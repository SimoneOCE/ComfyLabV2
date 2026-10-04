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
        from . import _pack_module
        pack = _pack_module()
        # H3PerFrameDenoise.run's curve (absolute_px), step for step.
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
        strength = np.clip(strength, 0.0, 1.0)

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


def _handoff_path(name):
    if not name or "/" in name or "\\" in name or name.startswith("."):
        raise ValueError(f"bad hand-off name {name!r}")
    # On the volume (the output dir): up to ~2.5GB for 4 people at 768 on a
    # 15s clip, more than the container's own small disk can spare. The
    # worker deletes it once the refine finishes.
    return f"{folder_paths.get_output_directory()}/refine_handoff/{name}.pt"


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
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ComfyLabH3AudioLock": "ComfyLab H3 Audio Lock",
    "ComfyLabStrengthWeights": "ComfyLab Strength -> Stitch Weights",
    "ComfyLabH3StepCheck": "ComfyLab H3 Step Check",
    "ComfyLabSaveRefineCrops": "ComfyLab Save Refine Crops (hand-off)",
    "ComfyLabLoadRefineCrops": "ComfyLab Load Refine Crops (hand-off)",
}
