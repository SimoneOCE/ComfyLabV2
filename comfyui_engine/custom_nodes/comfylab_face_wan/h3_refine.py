"""Helper nodes for the H3 face-refine engine (handler.py's build_h3_refine_payload).

The H3 engine is ComfyUI-H3-FaceRefine's own chain - Track + Crop, Inject
Video Latent, H3PerFrameDenoise, Stitch - with one change: the sampler gets
H3PerFrameDenoise's LATENT (its per-frame mask: full strength on small faces,
zero on faces 120px and up) but NOT its patched MODEL. Our ComfyUI (0.35+)
applies a per-frame H3 mask natively (comfy/model_base.py MiniMaxH3:
_denoise_mask_conds / scale_latent_inpaint, and the mask-scaled velocity in
comfy/ldm/minimax/model.py); the node's two model patches redo that by hand
for older ComfyUI and, stacked on the native handling, sent a negative
timestep then NaN into H3 (pack issue #19).

  ComfyLabH3AudioLock   - puts the clip's own soundtrack in the latent's
                          audio stream and holds it (mask 0) so lip sync
                          follows the real audio - the audio of exactly the
                          frames the crop covers (the tracker drops frames
                          the person isn't in). Core audio VAE + core
                          resample only (the old MiniMaxH3NativeAudioLock
                          needed torchaudio).
  ComfyLabStrengthWeights - per-frame stitch weights from the same strength
                          curve H3PerFrameDenoise uses, so frames it leaves
                          alone keep the video's own pixels (not a VAE
                          round-trip of them).
  ComfyLabH3StepCheck   - logs the timestep H3 receives and stops the job at
                          the first step with a bad timestep or NaN output,
                          instead of failing later with "KeyError: nan".
  ComfyLabSaveRefineCrops / ComfyLabLoadRefineCrops - hand the redrawn
                          crops and their transforms from the redraw prompt
                          to the stitch prompt through a file, so the worker
                          can unload H3 and clear ComfyUI's cache in between
                          (H3 + text encoder still in RAM during the stitch
                          and upscale ran a 15s clip out of RAM).
"""

import logging
import math

import numpy as np
import torch

import comfy.audio
import comfy.nested_tensor
import comfy.patcher_extension
import folder_paths

TAG = "[ComfyLabH3Refine]"
MAX_SUBJECTS = 4
WEIGHT_RAMP = 0.2  # same hand-over as the Wan engine: full paste once strength reaches this


def _log(msg):
    print(f"{TAG} {msg}", flush=True)
    logging.info(f"{TAG} {msg}")


class ComfyLabH3AudioLock:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"av_latent": ("LATENT",), "audio_vae": ("VAE",), "audio": ("AUDIO",)},
                "optional": {"transform": ("H3FACEXFORM",), "fps": ("FLOAT", {"default": 24.0})}}

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("av_latent", "report")
    FUNCTION = "run"
    CATEGORY = "ComfyLab"

    @staticmethod
    def _crop_audio(waveform, sr, source, fps):
        """The audio under each frame the crop holds, in order - so a crop
        that starts at frame 264 hears frame 264's audio, not the clip's
        first seconds."""
        per = sr / float(fps)
        pieces = []
        for f in source:
            a, b = int(round(int(f) * per)), int(round((int(f) + 1) * per))
            piece = waveform[..., a:b]
            if piece.shape[-1] < b - a:  # past the end of the soundtrack
                piece = torch.nn.functional.pad(piece, (0, b - a - piece.shape[-1]))
            pieces.append(piece)
        return torch.cat(pieces, dim=-1)

    def run(self, av_latent, audio_vae, audio, transform=None, fps=24.0):
        samples = av_latent["samples"]
        if not isinstance(samples, comfy.nested_tensor.NestedTensor):
            raise ValueError("Expected a MiniMax H3 joint AV latent (NestedTensor)")
        video, latent_audio = list(samples.unbind())[:2]
        waveform = audio["waveform"]
        sr = int(audio["sample_rate"])
        source = (transform or {}).get("source")
        span = ""
        if source:
            waveform = self._crop_audio(waveform, sr, source, fps)
            span = f" (frames {int(source[0])}-{int(source[-1])}, {len(source)} kept)"
        vae_sr = int(getattr(audio_vae, "audio_sample_rate", 32000))
        if sr != vae_sr:
            waveform = comfy.audio.resample(waveform, sr, vae_sr)
        # Same call as core's _encode_ref_audio (comfy_extras/nodes_minimax_h3.py).
        z = audio_vae.encode(waveform[:1].movedim(1, -1))  # [1, 32, 2, T]
        z = z.to(latent_audio.device, latent_audio.dtype)
        t_need, t_got = latent_audio.shape[-1], z.shape[-1]
        if t_got >= t_need:
            z = z[..., :t_need]
        else:
            # Hold the last latent frame (replicate) for the missing tail.
            z = torch.cat([z, z[..., -1:].expand(*z.shape[:-1], t_need - t_got)], dim=-1)
        if not torch.isfinite(z).all():
            raise RuntimeError("The clip's audio encoded to non-finite values")
        z = z.expand(latent_audio.shape[0], -1, -1, -1).contiguous()
        out = dict(av_latent)
        out["samples"] = comfy.nested_tensor.NestedTensor((video, z))
        out["noise_mask"] = comfy.nested_tensor.NestedTensor((torch.ones_like(video), torch.zeros_like(z)))
        report = (f"audio lock: {waveform.shape[-1] / vae_sr:.2f}s{span} at {vae_sr}Hz -> {t_got} latent frames "
                  f"(latent needs {t_need}), held at mask 0")
        _log(report)
        return (out, report)


def pfd_strength(transform, denoise_multiplier_small_face, denoise_multiplier_large_face,
                 face_px_small, face_px_large, gamma, smooth_frames):
    """Per-frame strength exactly as H3PerFrameDenoise.run computes it
    (absolute_px), plus each frame's face height."""
    from . import _pack_module
    pack = _pack_module()
    boxes = transform["boxes"]
    cf = float(transform.get("crop_factor", 3.0)) or 3.0
    face = np.array([b[3] / cf for b in boxes], dtype=np.float64)
    lo, hi = float(face_px_small), float(face_px_large)
    t = np.zeros_like(face) if hi - lo < 1e-6 else np.clip((face - lo) / (hi - lo), 0.0, 1.0)
    t = t ** float(gamma)
    strength = denoise_multiplier_small_face + (denoise_multiplier_large_face - denoise_multiplier_small_face) * t
    segs = transform.get("segments") or [(0, len(strength))]
    segs = [(int(a), int(b)) for a, b in segs if int(a) < len(strength)]
    strength = pack._smooth_seg(strength, int(smooth_frames), "gaussian", segs or [(0, len(strength))])
    absent = transform.get("absent")
    if absent and len(absent) == len(strength):
        strength[np.array(absent, dtype=bool)] = 0.0
    return np.clip(strength, 0.0, 1.0), face


class ComfyLabStrengthWeights:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "transform": ("H3FACEXFORM",),
            # Must match the H3PerFrameDenoise inputs in the same graph.
            "denoise_multiplier_small_face": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
            "denoise_multiplier_large_face": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0}),
            "face_px_small": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 1000.0}),
            "face_px_large": ("FLOAT", {"default": 120.0, "min": 2.0, "max": 2000.0}),
            "gamma": ("FLOAT", {"default": 1.0, "min": 0.2, "max": 4.0}),
            "smooth_frames": ("INT", {"default": 9, "min": 1, "max": 61}),
        }}

    RETURN_TYPES = ("H3FACEXFORM", "STRING")
    RETURN_NAMES = ("transform", "report")
    FUNCTION = "run"
    CATEGORY = "ComfyLab"

    def run(self, transform, denoise_multiplier_small_face, denoise_multiplier_large_face,
            face_px_small, face_px_large, gamma, smooth_frames):
        strength, face = pfd_strength(transform, denoise_multiplier_small_face, denoise_multiplier_large_face,
                                      face_px_small, face_px_large, gamma, smooth_frames)
        out = dict(transform)
        base = out.get("weights") or [1.0] * len(strength)
        ramp = np.clip(strength / WEIGHT_RAMP, 0.0, 1.0)
        out["weights"] = [float(w) * float(r) for w, r in zip(base, ramp)]
        kept = int((ramp <= 0.0).sum())
        report = (f"stitch weights: face {face.min():.0f}-{face.max():.0f}px, strength max {strength.max():.2f} "
                  f"mean {strength.mean():.2f}; {kept}/{len(strength)} frames keep the original pixels")
        _log(report)
        return (out, report)


class ComfyLabH3StepCheck:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",), "label": ("STRING", {"default": "h3 refine"})}}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "run"
    CATEGORY = "ComfyLab"

    def run(self, model, label):
        m = model.clone()
        state = {"calls": 0}

        def check(executor, x, timestep, *args, **kwargs):
            state["calls"] += 1
            n = state["calls"]
            ts = timestep.detach().float().flatten()
            t_min, t_max = float(ts.min()), float(ts.max())
            if n <= 2:
                _log(f"{label}: call {n} timestep {t_min:.2f}..{t_max:.2f}")
            if not math.isfinite(t_min) or not math.isfinite(t_max) or t_min < 0.0 or t_max > 1000.5:
                raise RuntimeError(f"{label}: H3 got timestep {t_min:.2f}..{t_max:.2f} on call {n} "
                                   "(expected 0..1000) - stopping before it turns into NaN")
            out = executor(x, timestep, *args, **kwargs)
            streams = out if isinstance(out, (list, tuple)) else [out]
            for k, s in enumerate(streams):
                if torch.is_tensor(s) and not torch.isfinite(s).all():
                    raise RuntimeError(f"{label}: H3 output stream {k} has NaN/inf on call {n} "
                                       f"(timestep {t_min:.2f}..{t_max:.2f})")
            return out

        m.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, "comfylab_h3_step_check", check)
        return (m,)


KEEP_BELOW = 0.02   # strength under which a frame keeps its original crop (as the Wan engine)
REDRAW_MAX = 512    # crops bigger than this are redrawn at this size (as the Wan engine)
H3_MULTIPLE = 32    # H3 canvases are multiples of 32 (comfy_extras/nodes_minimax_h3.py CANVAS_MULTIPLE)


def h3_grid(n):
    """Smallest H3 frame count (17k+5) holding n frames."""
    n = max(5, int(n))
    while n % 17 != 5:
        n += 1
    return n


class ComfyLabH3FaceRedraw:
    """The H3 engine's redraw, built like ComfyLabWanFaceRedraw:

    - one clip per person over their whole track by default; split_shots
      gives one clip per person per shot instead. Per shot was built after
      job 848f54fc put the shot-3 builder's face on the shot-4 mother, but
      testing showed the whole-scene prompt caused that (the generic prompt
      fixed it with one clip per person) and per-shot clips made the faces
      clearly worse (3b749775 vs 36c1d9a7), so it is off;
    - each clip covers the person's whole tracked stretch of that shot
      (padded to H3's 17k+5 grid), including frames where their face is big
      enough to be left alone: those are held unchanged (zero strength) but
      stay in the clip so H3 sees the person's clear face while redrawing
      the small-face frames - the identity anchor for someone walking toward
      the camera. A shot where none of the person's frames need redrawing is
      skipped;
    - redrawn at no more than REDRAW_MAX, resized back after;
    - the prompt is encoded once for every clip;
    - each clip goes through the pack's H3PerFrameDenoise (its slice of the
      tracker's transform) for its per-frame strength - its latent, not its
      patched model (see the module docstring) - and ComfyLabH3AudioLock for
      the audio of exactly its frames;
    - frames not redrawn keep the original crop, and the returned transforms'
      stitch weights follow the same strength curve (zero where nothing was
      redrawn).

    Not ported from the Wan node: batching clips of the same size. H3's own
    mask handling (comfy/model_base.py MiniMaxH3._denoise_mask_values) takes
    the per-frame mask of the FIRST batch row for every row's timesteps, so
    batched clips would all run at the first clip's strengths. Clips run one
    at a time."""

    @classmethod
    def INPUT_TYPES(cls):
        optional = {}
        for i in range(1, MAX_SUBJECTS):
            optional[f"crops_{i}"] = ("IMAGE",)
            optional[f"transform_{i}"] = ("H3FACEXFORM",)
        return {
            "required": {
                "crops_0": ("IMAGE",),
                "transform_0": ("H3FACEXFORM",),
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "audio_vae": ("VAE",),
                "audio": ("AUDIO",),
                "fps": ("FLOAT", {"default": 24.0}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "denoise": ("FLOAT", {"default": 0.4, "min": 0.05, "max": 1.0, "step": 0.01}),
                "steps": ("INT", {"default": 8, "min": 1, "max": 50}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "sampler_name": ("STRING", {"default": "er_sde"}),
                "denoise_multiplier_small_face": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
                "denoise_multiplier_large_face": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0}),
                "face_px_small": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 1000.0}),
                "face_px_large": ("FLOAT", {"default": 120.0, "min": 2.0, "max": 2000.0}),
                "gamma": ("FLOAT", {"default": 1.0, "min": 0.2, "max": 4.0}),
                "smooth_frames": ("INT", {"default": 9, "min": 1, "max": 61}),
            },
            "optional": {
                **optional,
                # On: one clip per person per shot. Off (default): one clip per
                # person over their whole track, across cuts - per-shot clips
                # made the faces clearly worse in testing.
                "split_shots": ("BOOLEAN", {"default": False}),
                # The detector's own face boxes, for the duplicate-person check.
                "face_pick": ("H3FACEPICK",),
            },
        }

    RETURN_TYPES = ("IMAGE",) * MAX_SUBJECTS + ("STRING",) + ("H3FACEXFORM",) * MAX_SUBJECTS
    RETURN_NAMES = (tuple(f"crops_{i}" for i in range(MAX_SUBJECTS)) + ("report",)
                    + tuple(f"transform_{i}" for i in range(MAX_SUBJECTS)))
    FUNCTION = "redraw"
    CATEGORY = "ComfyLab"

    def redraw(self, model, clip, vae, audio_vae, audio, fps, prompt, denoise, steps, seed, sampler_name,
               denoise_multiplier_small_face, denoise_multiplier_large_face, face_px_small, face_px_large,
               gamma, smooth_frames, split_shots=False, **subjects):
        import time
        start = time.time()
        curve = dict(denoise_multiplier_small_face=denoise_multiplier_small_face,
                     denoise_multiplier_large_face=denoise_multiplier_large_face,
                     face_px_small=face_px_small, face_px_large=face_px_large,
                     gamma=gamma, smooth_frames=smooth_frames)
        report = []
        work = []
        for i in range(MAX_SUBJECTS):
            crops, transform = subjects.get(f"crops_{i}"), subjects.get(f"transform_{i}")
            if crops is None or transform is None:
                continue
            n = min(crops.shape[0], len(transform["boxes"]))
            strength, face = pfd_strength(transform, **curve)
            strength = strength[:n]
            line = (f"subject {i}: {n} frames, face {face.min():.0f}-{face.max():.0f}px, "
                    f"strength max {strength.max():.2f} mean {strength.mean():.2f}, "
                    f"kept as-is {int((strength < KEEP_BELOW).sum())}/{n} frames")
            _log(line)
            report.append(line)
            work.append((i, crops, strength, transform))

        # Two trackers on one face would redraw and paste it twice.
        from . import duplicates
        work, silenced, dup_report = duplicates.apply(work, subjects.get("face_pick"))
        for line in dup_report:
            _log(line)
            report.append(line)

        # One clip per person per shot - the whole shot, so the frames where
        # their face is big stay in as (unchanged) context. Shots with nothing
        # to redraw are skipped.
        clips = []
        for i, crops, strength, transform in work:
            n = len(strength)
            segments = (transform.get("segments") or [(0, n)]) if split_shots else [(0, n)]
            for a, b in segments:
                a, b = max(0, int(a)), min(n, int(b))
                if b > a and (strength[a:b] >= KEEP_BELOW).any():
                    clips.append((i, a, b))

        outputs = {i: crops for i, crops, _, _ in work}
        outputs.update({i: subjects[f"crops_{i}"] for i in silenced})
        redrawn = {i: np.zeros(len(st), dtype=bool) for i, _, st, _ in work}
        if clips:
            text = prompt.strip() or ("close-up of a real person's face, natural realistic facial features, "
                                      "sharp clear eyes, detailed natural skin texture, natural mouth")
            t0 = time.time()
            # What MiniMaxH3ReferenceToVideo does with no references - once, for every clip.
            positive = clip.encode_from_tokens_scheduled(clip.tokenize(text, minimax_ref_items=[]))
            report.append(f"prompt encoded once in {time.time() - t0:.1f}s")
            _log(report[-1])
            crops_by = {i: c for i, c, _, _ in work}
            xform_by = {i: t for i, _, _, t in work}
            results = {i: c.clone() for i, c in crops_by.items()}
            for i, clip_start, clip_end in clips:
                t1 = time.time()
                rw, rh = self._redraw_clip(model, vae, audio_vae, audio, fps, positive, crops_by[i], xform_by[i],
                                           clip_start, clip_end, results[i], denoise, steps, seed, sampler_name,
                                           curve)
                redrawn[i][clip_start:clip_end] = True
                report.append(f"person {i} frames {clip_start}-{clip_end - 1} - redrawn at {rw}x{rh}, "
                              f"{time.time() - t1:.1f}s")
                _log(report[-1])
            for i, result in results.items():
                strength = next(st for j, _, st, _ in work if j == i)
                keep = torch.from_numpy((strength < KEEP_BELOW) | ~redrawn[i]).to(result.device)
                keep = keep[:result.shape[0]]
                result[:keep.shape[0]][keep] = crops_by[i][:keep.shape[0]][keep]
                outputs[i] = result
        else:
            report.append("every tracked face is already large enough - nothing redrawn")
            _log(report[-1])

        # Stitch weights from the same strength, zero wherever nothing was redrawn.
        transforms = {}
        for i, crops, strength, transform in work:
            t = dict(transform)
            base = list(t.get("weights") or [1.0] * len(strength))
            ramp = np.clip(strength / WEIGHT_RAMP, 0.0, 1.0) * redrawn[i]
            t["weights"] = [float(w) * float(r) for w, r in zip(base, ramp)] + base[len(ramp):]
            transforms[i] = t
        transforms.update(silenced)
        report.append(f"node total {time.time() - start:.1f}s")
        _log(report[-1])

        first = outputs.get(0, subjects.get("crops_0"))
        crops_out = tuple(outputs.get(i, first) for i in range(MAX_SUBJECTS))
        first_t = transforms.get(0, subjects.get("transform_0"))
        xforms = tuple(transforms.get(i, first_t) for i in range(MAX_SUBJECTS))
        return crops_out + ("\n".join(report),) + xforms

    @staticmethod
    def _redraw_clip(model, vae, audio_vae, audio, fps, positive, crops, transform, start, end, target,
                     denoise, steps, seed, sampler_name, curve):
        import torch.nn.functional as F
        import nodes
        from comfy_extras.nodes_custom_sampler import (BasicGuider, BasicScheduler, KSamplerSelect,
                                                       RandomNoise, SamplerCustomAdvanced)
        from comfy_extras.nodes_minimax_h3 import _empty_av_latent
        from . import _pack_module
        pack = _pack_module()

        span = end - start
        length = h3_grid(span)
        pad = length - span
        frames = crops[start:end, :, :, :3]
        h, w = frames.shape[1:3]
        scale = min(1.0, REDRAW_MAX / max(h, w))
        rh = max(H3_MULTIPLE, int(round(h * scale / H3_MULTIPLE)) * H3_MULTIPLE)
        rw = max(H3_MULTIPLE, int(round(w * scale / H3_MULTIPLE)) * H3_MULTIPLE)
        if (h, w) != (rh, rw):
            frames = F.interpolate(frames.movedim(-1, 1).float(), size=(rh, rw), mode="area").movedim(1, -1)
        if pad:
            frames = torch.cat([frames, frames[-1:].repeat(pad, 1, 1, 1)], dim=0)

        # The clip's slice of the tracker's transform; padding frames are marked
        # absent, so H3PerFrameDenoise gives them zero strength.
        boxes = list(transform["boxes"])[start:end]
        sub = dict(transform)
        sub["boxes"] = boxes + [boxes[-1]] * pad
        # The clip's own shot boundaries (one when split per shot), shifted to
        # the clip and capped by the padding.
        sub["segments"] = [(max(0, int(a) - start), min(span, int(b) - start))
                           for a, b in (transform.get("segments") or [(0, len(transform["boxes"]))])
                           if int(b) > start and int(a) < end] or [(0, span)]
        if pad:
            sub["segments"] = sub["segments"] + [(span, length)]
        sub["absent"] = (list(transform.get("absent") or [False] * len(transform["boxes"]))[start:end]
                         + [True] * pad)
        source = list(transform.get("source") or range(len(transform["boxes"])))[start:end]
        source = source + [source[-1]] * pad

        latent, _ = _empty_av_latent(rw, rh, length)
        latent, _ = ComfyLabH3AudioLock().run(latent, audio_vae, audio, transform={"source": source}, fps=fps)
        latent, _ = pack.H3InjectVideoLatent().run(latent, frames, vae)
        latent, _, _ = pack.H3PerFrameDenoise().run(model, latent, sub, scale_mode="absolute_px", **curve)

        sigmas = BasicScheduler.execute(model, "simple", steps, denoise).args[0]
        sampled = SamplerCustomAdvanced.execute(
            RandomNoise.execute(seed).args[0], BasicGuider.execute(model, positive).args[0],
            KSamplerSelect.execute(sampler_name).args[0], sigmas, latent).args[0]
        images = nodes.VAEDecode().decode(vae, sampled)[0][:span]
        if (h, w) != (rh, rw):
            images = F.interpolate(images.movedim(-1, 1).float(), size=(h, w), mode="bicubic",
                                   align_corners=False, antialias=True).movedim(1, -1)
        target[start:end, :, :, :3] = images.to(target.device, target.dtype).clamp(0.0, 1.0)
        return rw, rh


def _handoff_path(name):
    if not name or "/" in name or "\\" in name or name.startswith("."):
        raise ValueError(f"bad hand-off name {name!r}")
    # ComfyUI's temp dir, on the container's own disk (faster than the
    # volume). Usually under 0.5GB; up to ~2.5GB for 4 people at 768 on a
    # 15s clip. The worker deletes it once the refine finishes, and ComfyUI
    # empties its temp dir on every start.
    return f"{folder_paths.get_temp_directory()}/refine_handoff/{name}.pt"


class ComfyLabSaveRefineCrops:
    @classmethod
    def INPUT_TYPES(cls):
        optional = {}
        for i in range(1, MAX_SUBJECTS):
            optional[f"crops_{i}"] = ("IMAGE",)
            optional[f"transform_{i}"] = ("H3FACEXFORM",)
        return {"required": {"name": ("STRING", {"default": ""}),
                             "crops_0": ("IMAGE",), "transform_0": ("H3FACEXFORM",)},
                "optional": optional}

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = "ComfyLab"

    def run(self, name, **subjects):
        import os
        crops, transforms = [], []
        for i in range(MAX_SUBJECTS):
            c, t = subjects.get(f"crops_{i}"), subjects.get(f"transform_{i}")
            if c is None or t is None:
                break
            # 8-bit, like the video they end up in.
            crops.append((c[..., :3].clamp(0, 1) * 255.0).round().to(torch.uint8).cpu())
            transforms.append(t)
        path = _handoff_path(name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({"crops": crops, "transforms": transforms}, path)
        mb = sum(c.numel() for c in crops) / 1e6
        _log(f"saved {len(crops)} redrawn crop clip(s), {mb:.0f}MB, for the stitch prompt")
        return {"ui": {"saved": [name], "subjects": [len(crops)]}}


class ComfyLabLoadRefineCrops:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"name": ("STRING", {"default": ""})}}

    RETURN_TYPES = ("IMAGE",) * MAX_SUBJECTS + ("H3FACEXFORM",) * MAX_SUBJECTS
    RETURN_NAMES = (tuple(f"crops_{i}" for i in range(MAX_SUBJECTS))
                    + tuple(f"transform_{i}" for i in range(MAX_SUBJECTS)))
    FUNCTION = "run"
    CATEGORY = "ComfyLab"

    @classmethod
    def IS_CHANGED(cls, name):
        import os
        try:
            return os.path.getmtime(_handoff_path(name))
        except OSError:
            return float("nan")

    def run(self, name):
        data = torch.load(_handoff_path(name), weights_only=False)
        crops = [c.float() / 255.0 for c in data["crops"]]
        transforms = list(data["transforms"])
        if not crops:
            raise ValueError("hand-off file holds no crops")
        while len(crops) < MAX_SUBJECTS:  # unused outputs - never wired
            crops.append(crops[0])
            transforms.append(transforms[0])
        return tuple(crops) + tuple(transforms)


NODE_CLASS_MAPPINGS = {
    "ComfyLabH3AudioLock": ComfyLabH3AudioLock,
    "ComfyLabStrengthWeights": ComfyLabStrengthWeights,
    "ComfyLabH3StepCheck": ComfyLabH3StepCheck,
    "ComfyLabSaveRefineCrops": ComfyLabSaveRefineCrops,
    "ComfyLabLoadRefineCrops": ComfyLabLoadRefineCrops,
    "ComfyLabH3FaceRedraw": ComfyLabH3FaceRedraw,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ComfyLabH3AudioLock": "ComfyLab H3 Audio Lock",
    "ComfyLabStrengthWeights": "ComfyLab Strength -> Stitch Weights",
    "ComfyLabH3StepCheck": "ComfyLab H3 Step Check",
    "ComfyLabSaveRefineCrops": "ComfyLab Save Refine Crops (hand-off)",
    "ComfyLabLoadRefineCrops": "ComfyLab Load Refine Crops (hand-off)",
    "ComfyLabH3FaceRedraw": "ComfyLab H3 Face Redraw",
}
