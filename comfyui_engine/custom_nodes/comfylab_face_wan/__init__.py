"""ComfyLab Wan face redraw - the "Refine faces (Wan)" engine.

Redraws the face crops that ComfyUI-H3-FaceRefine's H3FaceTrackCrop produces
with Wan 2.2's low-noise 14B model (+ its 4-step lightx2v LoRA) instead of
H3, then hands them back for H3FaceStitch to paste in. Only core ComfyUI
model code is used, so none of H3's denoise-mask internals are involved.

Per-frame strength, so already-good faces are not redrawn: each frame's
noise mask comes from that frame's face height, on the same curve as the
pack's H3PerFrameDenoise (full strength at <= face_px_small, none at >=
face_px_large, smoothed per shot, zero where the tracker lost the face). It
goes through the sampler's standard noise mask (SetLatentNoiseMask's
mechanism). Frames that aren't redrawn keep the VIDEO's own pixels: the
returned transforms scale the stitch's per-frame weights by that strength.

Clips: one per person per shot (Wan never sees a hard cut), covering only
the stretch where the face is small plus a few frames of context. Crops are
redrawn at no more than REDRAW_MAX - the tracker sizes a canvas for a
person's largest face, which is never redrawn. Clips of the same length and
size are batched into one Wan call (one at a time if that runs out of VRAM).

Memory: one node does every subject so Wan loads once per refine, and it
loads, samples and unloads inside the node. Nothing Wan-related is left in
ComfyUI's output cache (only the image tensors are returned), and on the way
out the Wan model, text encoder and VAE are unloaded from VRAM and RAM, so
an H3 generation after a refine is not sharing memory with Wan.
"""

import gc
import logging
import sys
import time

import numpy as np
import torch

import comfy.model_management as mm
import nodes

from . import duplicates
from comfy_extras.nodes_model_advanced import ModelSamplingSD3

TAG = "[ComfyLabWanFace]"
MAX_SUBJECTS = 4
KEEP_BELOW = 0.02  # mask value under which a frame keeps its original crop
RANGE_MARGIN = 4   # extra frames kept either side of the active range, as context
REDRAW_MAX = 512   # crops bigger than this are redrawn at this size (see redraw())
WEIGHT_RAMP = 0.2  # paste weight reaches full once a frame's strength is this high


def _log(msg):
    print(f"{TAG} {msg}", flush=True)
    logging.info(f"{TAG} {msg}")


def _ram_gb():
    try:
        import psutil
        vm = psutil.virtual_memory()
        return f"RAM used {(vm.total - vm.available) / 1e9:.1f}/{vm.total / 1e9:.1f} GB"
    except Exception:
        return "RAM n/a"


def _vram_gb():
    if not torch.cuda.is_available():
        return "VRAM n/a"
    free, total = torch.cuda.mem_get_info()
    return f"VRAM used {(total - free) / 1e9:.1f}/{total / 1e9:.1f} GB"


def face_strength_curve(transform, n, face_px_small, face_px_large, smooth_frames, face_px_min=0.0):
    """Per-frame strength 0..1 from the tracker's boxes - same inputs and ramp
    as the pack's H3PerFrameDenoise, ending at 0 for large faces."""
    boxes = transform["boxes"]
    cf = float(transform.get("crop_factor", 3.0)) or 3.0
    face = np.array([b[3] / cf for b in boxes], dtype=np.float64)
    if face.size == 0:
        raise ValueError("transform has no boxes")
    if face.size != n:
        face = np.interp(np.linspace(0, face.size - 1, n), np.arange(face.size), face)
    span = max(1e-6, float(face_px_large) - float(face_px_small))
    strength = 1.0 - np.clip((face - float(face_px_small)) / span, 0.0, 1.0)

    k = int(smooth_frames)
    if k > 1:
        k += 1 - k % 2  # odd window
        segs = transform.get("segments") or [(0, n)]
        smoothed = strength.copy()
        for a, b in segs:
            a, b = int(a), min(int(b), n)
            seg = strength[a:b]
            if seg.size == 0:
                continue
            padded = np.pad(seg, k // 2, mode="edge")
            smoothed[a:b] = np.convolve(padded, np.ones(k) / k, mode="valid")[: seg.size]
        strength = smoothed

    absent = transform.get("absent")
    if absent is not None and len(absent) == n:
        strength[np.array(absent, dtype=bool)] = 0.0
    # Faces under face_px_min are too small to repair (and are where false
    # detections come from): left exactly as they are.
    if face_px_min > 0:
        strength[face < float(face_px_min)] = 0.0
    return np.clip(strength, 0.0, 1.0), face


class ComfyLabWanFaceRedraw:
    @classmethod
    def INPUT_TYPES(cls):
        optional = {}
        for i in range(1, MAX_SUBJECTS):
            optional[f"crops_{i}"] = ("IMAGE",)
            optional[f"transform_{i}"] = ("H3FACEXFORM",)
        # The detector's own face boxes, for the duplicate-person check.
        optional["face_pick"] = ("H3FACEPICK",)
        optional["face_px_min"] = ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1000.0})
        return {
            "required": {
                "crops_0": ("IMAGE",),
                "transform_0": ("H3FACEXFORM",),
                "unet_name": ("STRING", {"default": "wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors"}),
                "lora_name": ("STRING", {"default": "wan2.2_t2v_lightx2v_4steps_lora_v1.1_low_noise.safetensors"}),
                "clip_name": ("STRING", {"default": "umt5_xxl_fp8_e4m3fn_scaled.safetensors"}),
                "vae_name": ("STRING", {"default": "wan_2.1_vae.safetensors"}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "denoise": ("FLOAT", {"default": 0.5, "min": 0.05, "max": 1.0, "step": 0.01}),
                "steps": ("INT", {"default": 4, "min": 1, "max": 50}),
                "shift": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "face_px_small": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 1000.0}),
                "face_px_large": ("FLOAT", {"default": 120.0, "min": 2.0, "max": 2000.0}),
                "smooth_frames": ("INT", {"default": 9, "min": 1, "max": 61}),
                "sage_attention": ("BOOLEAN", {"default": True}),
            },
            "optional": optional,
        }

    RETURN_TYPES = ("IMAGE",) * MAX_SUBJECTS + ("STRING",) + ("H3FACEXFORM",) * MAX_SUBJECTS
    RETURN_NAMES = (tuple(f"crops_{i}" for i in range(MAX_SUBJECTS)) + ("report",)
                    + tuple(f"transform_{i}" for i in range(MAX_SUBJECTS)))
    FUNCTION = "redraw"
    CATEGORY = "ComfyLab"

    def redraw(self, unet_name, lora_name, clip_name, vae_name, prompt, denoise, steps, shift, seed,
               face_px_small, face_px_large, smooth_frames, sage_attention, **subjects):
        report = []
        start = time.time()
        _log(f"start: {_ram_gb()}, {_vram_gb()}")

        work = []
        for i in range(MAX_SUBJECTS):
            crops = subjects.get(f"crops_{i}")
            transform = subjects.get(f"transform_{i}")
            if crops is None or transform is None:
                continue
            n = crops.shape[0]
            strength, face = face_strength_curve(transform, n, face_px_small, face_px_large, smooth_frames,
                                                 subjects.get("face_px_min") or 0.0)
            line = (f"subject {i}: {n} frames, face {face.min():.0f}-{face.max():.0f}px, "
                    f"strength max {strength.max():.2f} mean {strength.mean():.2f}, "
                    f"kept as-is {int((strength < KEEP_BELOW).sum())}/{n} frames")
            _log(line)
            report.append(line)
            work.append((i, crops, strength, transform))

        # Two trackers on one face (an extra "person" from a stray detection)
        # would redraw and paste it twice - see duplicates.py.
        work, silenced, dup_report = duplicates.apply(work, subjects.get("face_pick"))
        for line in dup_report:
            _log(line)
            report.append(line)

        outputs = {i: crops for i, crops, _, _ in work}
        outputs.update({i: subjects[f"crops_{i}"] for i in silenced})
        # Frames this refine doesn't redraw must keep the video's own pixels.
        # Otherwise H3FaceStitch still pastes the tracker's crop back - and a
        # big face's crop has been squeezed into the canvas and blown back up,
        # which softens a close-up that needed nothing. The stitch's per-frame
        # weights (undetected_frames="fade_out") already do exactly this, so
        # the weights are scaled by the frame's strength, ramping to full by
        # WEIGHT_RAMP so the hand-over is a fade, not a pop.
        transforms = {}
        for i, crops, strength, transform in work:
            t = dict(transform)
            base = t.get("weights") or [1.0] * len(strength)
            ramp = np.clip(strength / WEIGHT_RAMP, 0.0, 1.0)
            t["weights"] = [float(w) * float(r) for w, r in zip(base, ramp)]
            transforms[i] = t
        transforms.update(silenced)

        # One redraw clip per person per SHOT: a person's crops can span hard
        # cuts (the tracker strings their shots together), and Wan generating
        # them as one continuous video blends the face across each cut. Within
        # a shot, only the stretch where the face is small (plus a few frames
        # of context) is generated.
        clips = []
        for i, crops, strength, transform in work:
            n = crops.shape[0]
            segs = transform.get("segments") or [(0, n)]
            for a, b in segs:
                a, b = max(0, int(a)), min(n, int(b))
                idx = np.flatnonzero(strength[a:b] >= KEEP_BELOW)
                if idx.size == 0:
                    continue
                clip_start = max(a, a + int(idx[0]) - RANGE_MARGIN)
                clip_end = min(b, a + int(idx[-1]) + 1 + RANGE_MARGIN)
                clips.append((i, clip_start, clip_end))
        if not clips:
            report.append("every tracked face is already large enough - Wan not loaded, crops unchanged")
            _log(report[-1])
            return self._result(outputs, transforms, subjects, report)

        unet = model = clip = vae = None
        try:
            t0 = time.time()
            unet = nodes.UNETLoader().load_unet(unet_name, "default")[0]
            model = nodes.LoraLoaderModelOnly().load_lora_model_only(unet, lora_name, 1.0)[0]
            model = ModelSamplingSD3().patch(model, shift)[0]
            if sage_attention:
                sage = nodes.NODE_CLASS_MAPPINGS.get("PathchSageAttentionKJ")
                if sage is None:
                    _log("PathchSageAttentionKJ not available - running without Sage")
                else:
                    try:
                        model = sage().patch(model, "auto", False)[0]
                    except Exception as e:
                        _log(f"Sage patch failed ({e}) - running without Sage")
            clip = nodes.CLIPLoader().load_clip(clip_name, "wan")[0]
            vae = nodes.VAELoader().load_vae(vae_name)[0]
            text = prompt.strip() or (
                "close-up of a real person's face, natural realistic facial features, "
                "sharp clear eyes, detailed natural skin texture, natural mouth, cinematic lighting")
            positive = nodes.CLIPTextEncode().encode(clip, text)[0]
            negative = nodes.CLIPTextEncode().encode(clip, "")[0]
            report.append(f"Wan loaded in {time.time() - t0:.1f}s ({_ram_gb()}, {_vram_gb()})")
            _log(report[-1])

            crops_by = {i: c for i, c, _, _ in work}
            strength_by = {i: st for i, _, st, _ in work}
            results = {i: c.clone() for i, c in crops_by.items()}
            # Size the redraw from the frames actually redrawn: the tracker sizes
            # a person's canvas for their LARGEST face (a close-up can push it to
            # 768), but those frames are never redrawn. Redrawn faces are under
            # face_px_large, so their crops fit REDRAW_MAX with room to spare.
            groups = {}
            for i, clip_start, clip_end in clips:
                h, w = crops_by[i].shape[1:3]
                scale = min(1.0, REDRAW_MAX / max(h, w))
                rh, rw = (max(16, int(round(h * scale / 16)) * 16), max(16, int(round(w * scale / 16)) * 16))
                span = clip_end - clip_start
                span_pad = ((span - 1 + 3) // 4) * 4 + 1  # Wan's VAE takes 4k+1 frames
                groups.setdefault((span_pad, rh, rw), []).append((i, clip_start, clip_end))
            for (span_pad, rh, rw), group in groups.items():
                t1 = time.time()
                try:
                    self._redraw_clips(model, vae, positive, negative, group, crops_by, strength_by, results,
                                       span_pad, rh, rw, denoise, steps, seed)
                    report.append(self._clip_line(group, rh, rw, time.time() - t1))
                except mm.OOM_EXCEPTION:
                    if len(group) == 1:
                        raise
                    mm.soft_empty_cache(True)
                    report.append(f"{len(group)} clips at {rw}x{rh}: batch did not fit in VRAM - one at a time")
                    _log(report[-1])
                    for clip_ in group:
                        t2 = time.time()
                        self._redraw_clips(model, vae, positive, negative, [clip_], crops_by, strength_by, results,
                                           span_pad, rh, rw, denoise, steps, seed)
                        report.append(self._clip_line([clip_], rh, rw, time.time() - t2))
                        _log(report[-1])
                    continue
                _log(report[-1])
            for i, result in results.items():
                keep = torch.from_numpy(strength_by[i] < KEEP_BELOW).to(result.device)
                result[keep] = crops_by[i][keep]
                outputs[i] = result
        finally:
            self._release(unet, clip, vae)
            del unet, model, clip, vae
            gc.collect()
            mm.cleanup_models()
            mm.soft_empty_cache()
            report.append(f"Wan unloaded ({_ram_gb()}, {_vram_gb()}); node total {time.time() - start:.1f}s")
            _log(report[-1])

        return self._result(outputs, transforms, subjects, report)

    @staticmethod
    def _clip_line(group, rh, rw, seconds):
        clips = ", ".join(f"person {i} frames {a}-{b - 1}" for i, a, b in group)
        return f"{clips} - redrawn at {rw}x{rh}{' (batched)' if len(group) > 1 else ''}, {seconds:.1f}s"

    @staticmethod
    def _redraw_clips(model, vae, positive, negative, group, crops_by, strength_by, results,
                      span_pad, rh, rw, denoise, steps, seed):
        """group: [(person, start, end)] - clips of the same padded length,
        redrawn together at rw x rh. Writes into results[person][start:end]."""
        import torch.nn.functional as F

        latents, masks = [], []
        for i, start, end in group:
            frames = crops_by[i][start:end, :, :, :3]
            h, w = frames.shape[1:3]
            if (h, w) != (rh, rw):
                frames = F.interpolate(frames.movedim(-1, 1), size=(rh, rw), mode="area").movedim(1, -1)
            span = end - start
            if span_pad > span:
                frames = torch.cat([frames, frames[-1:].repeat(span_pad - span, 1, 1, 1)], dim=0)
            latents.append(nodes.VAEEncode().encode(vae, frames)[0]["samples"])
            st = np.pad(strength_by[i][start:end], (0, span_pad - span), mode="edge")
            masks.append(torch.from_numpy(st).float().view(1, 1, span_pad, 1, 1).expand(1, 1, span_pad, rh, rw))
        # The sampler's standard noise mask (SetLatentNoiseMask's mechanism):
        # comfy.utils.reshape_mask maps [B,1,T,H,W] onto the video latent.
        latent = {"samples": torch.cat(latents, dim=0), "noise_mask": torch.cat(masks, dim=0).contiguous()}
        sampled = nodes.common_ksampler(model, seed, steps, 1.0, "euler", "simple",
                                        positive, negative, latent, denoise=denoise)[0]["samples"]
        for b, (i, start, end) in enumerate(group):
            images = nodes.VAEDecode().decode(vae, {"samples": sampled[b:b + 1]})[0][:end - start]
            h, w = crops_by[i].shape[1:3]
            if (h, w) != (rh, rw):
                images = F.interpolate(images.movedim(-1, 1).float(), size=(h, w), mode="bicubic",
                                       align_corners=False, antialias=True).movedim(1, -1)
            target = results[i]
            target[start:end, :, :, :3] = images.to(target.device, target.dtype).clamp(0.0, 1.0)

    @staticmethod
    def _release(*objects):
        for obj in objects:
            if obj is None:
                continue
            patcher = getattr(obj, "patcher", obj)
            try:
                mm.unload_model_and_clones(patcher)
            except Exception as e:
                _log(f"unload of {type(obj).__name__} failed: {e}")

    @staticmethod
    def _result(outputs, transforms, subjects, report):
        first = outputs.get(0, subjects.get("crops_0"))
        crops = tuple(outputs.get(i, first) for i in range(MAX_SUBJECTS))
        first_t = transforms.get(0, subjects.get("transform_0"))
        xforms = tuple(transforms.get(i, subjects.get(f"transform_{i}", first_t)) for i in range(MAX_SUBJECTS))
        return crops + ("\n".join(report),) + xforms


def _pack_module():
    select_cls = nodes.NODE_CLASS_MAPPINGS.get("H3FaceSelect")
    if select_cls is None:
        raise RuntimeError("ComfyUI-H3-FaceRefine's H3FaceSelect is not loaded")
    return sys.modules[select_cls.__module__]


def shot_face_sizes(face_pick, face_px_large, face_px_min=0.0):
    """Per shot: median height of each repairable face, largest first - the
    k-th largest face per frame, medianed over the frames that have one.
    Logging only (tuning face_px_min / face_px_large)."""
    boxes = face_pick["boxes"]
    out = []
    for a, b in face_pick["segments"]:
        ranks = {}
        for f in range(int(a), min(int(b), len(boxes))):
            heights = sorted((float(q[3]) - float(q[1]) for q in boxes[f]), reverse=True)
            for k, h in enumerate(h for h in heights if face_px_min <= h < face_px_large):
                ranks.setdefault(k, []).append(h)
        out.append([round(float(np.median(v))) for k, v in sorted(ranks.items())
                    if len(v) >= max(3, int(np.ceil(0.05 * max(1, int(b) - int(a)))))])
    return out


def shot_face_counts(face_pick, face_px_large, face_px_min=0.0):
    """Per shot: (small, large) - how many faces from face_px_min up to under
    face_px_large tall, and at-or-over face_px_large, are on screen together.
    Faces under face_px_min aren't counted at all. Robust to stray
    detections: a count only counts if at least max(3, 5% of the shot's)
    frames reach it."""
    boxes = face_pick["boxes"]
    out = []
    for a, b in face_pick["segments"]:
        a, b = int(a), int(b)
        small, large = [], []
        for f in range(a, min(b, len(boxes))):
            heights = [float(q[3]) - float(q[1]) for q in boxes[f]]
            small.append(sum(face_px_min <= h < face_px_large for h in heights))
            large.append(sum(h >= face_px_large for h in heights))
        need = max(3, int(np.ceil(0.05 * max(1, len(small)))))

        def robust(counts):
            if len(counts) < need:
                return max(counts, default=0)
            return int(sorted(counts, reverse=True)[need - 1])
        out.append((robust(small), robust(large)))
    return out


class ComfyLabSmallFaceCount:
    """How many people the refine should redraw: the most faces under
    face_px_large tall on screen together in any one shot, capped at
    max_people (the largest small faces win when there are more). Reads the
    face_pick from H3 Load Video + Face Select, so nothing is detected again.
    The report's first line is machine-read by the worker."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"face_pick": ("H3FACEPICK",),
                             "face_px_large": ("FLOAT", {"default": 120.0, "min": 2.0, "max": 2000.0}),
                             "max_people": ("INT", {"default": 4, "min": 1, "max": 4})},
                "optional": {"face_px_min": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1000.0})}}

    RETURN_TYPES = ("INT", "STRING")
    RETURN_NAMES = ("people", "report")
    FUNCTION = "run"
    CATEGORY = "ComfyLab"

    def run(self, face_pick, face_px_large, max_people, face_px_min=0.0):
        counts = shot_face_counts(face_pick, face_px_large, face_px_min)
        sizes = shot_face_sizes(face_pick, face_px_large, face_px_min)
        found = max((sm for sm, _ in counts), default=0)
        people = min(found, int(max_people))
        lines = [f"small_face_people={people} found={found}"]
        for k, (sm, lg) in enumerate(counts):
            px = f" - repairable faces ~{', '.join(f'{h}px' for h in sizes[k])}" if k < len(sizes) and sizes[k] else ""
            lines.append(f"shot {k + 1}: {sm} small face(s), {lg} large{px}")
        if face_px_min > 0:
            lines.append(f"faces under {face_px_min:.0f}px or from {face_px_large:.0f}px up are left alone")
        if found > people:
            lines.append(f"{found} small faces in one shot - the {people} largest of them are refined")
        report = "\n".join(lines)
        _log(report.replace("\n", " | "))
        return (people, report)


class ComfyLabFacePickIndex:
    """One detection pass, many people: takes the face_pick from the pack's
    H3 Load Video + Face Select (which detects every face and cut once) and
    re-picks it for person `index` (largest-face order, per shot), so each
    H3 Face Track + Crop reuses those boxes instead of detecting the video
    again. Shots that never hold index+1 faces are marked as not containing
    this person - their frames keep their original pixels - instead of the
    pack's fallback of re-using the last face in the shot (which redraws
    the same person several times). If no shot holds that many faces, the
    pack's fallback is kept so the tracker still has something to follow."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"face_pick": ("H3FACEPICK",),
                             "index": ("INT", {"default": 0, "min": 0, "max": 63})},
                "optional": {
                    # On: `index` counts SMALL faces only - person 0 is the largest
                    # face under face_px_large in each shot, skipping the faces
                    # that are already big (the refine leaves those alone anyway).
                    "skip_large": ("BOOLEAN", {"default": False}),
                    "face_px_large": ("FLOAT", {"default": 120.0, "min": 2.0, "max": 2000.0}),
                    "face_px_min": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1000.0})}}

    RETURN_TYPES = ("H3FACEPICK", "STRING")
    RETURN_NAMES = ("face_pick", "report")
    FUNCTION = "run"
    CATEGORY = "ComfyLab"

    def run(self, face_pick, index, skip_large=False, face_px_large=120.0, face_px_min=0.0):
        pack = _pack_module()
        boxes, confs = face_pick["boxes"], face_pick["confs"]
        segs = [(int(a), int(b)) for a, b in face_pick["segments"]]
        width, height = face_pick["src_size"]
        rank = pack._review_select("largest_face")
        if skip_large:
            # Per shot, rank among ALL faces = the shot's large faces + index.
            counts = shot_face_counts(face_pick, face_px_large, face_px_min)
            picks, present = [], []
            for k, (a, b) in enumerate(segs):
                small, large = counts[k]
                picks.append(pack._auto_pick(boxes, confs, [(a, b)], width, height, rank,
                                             large + int(index))[0])
                present.append(index < small)
        else:
            picks = pack._auto_pick(boxes, confs, segs, width, height, rank, int(index))
            present = [max((len(boxes[f]) for f in range(a, b)), default=0) > index for a, b in segs]
        if any(present):
            for k, (a, b) in enumerate(segs):
                if not present[k]:
                    picks[k] = {"segment": [a, b], "frame": -1, "box": -1,
                                "index": pack._ABSENT, "absent": True}
        out = dict(face_pick)
        out["picks"] = picks
        shots = [k + 1 for k, here in enumerate(present) if here]
        report = (f"person {index}: in shot(s) {shots or 'none'} of {len(segs)}"
                  + ("" if any(present) else " - fewer faces than that anywhere, re-using the last face"))
        _log(report)
        return (out, report)


try:
    from . import timing as _timing
    _timing.install()
except Exception as _e:  # logging-only helper; never block the nodes
    logging.warning(f"[ComfyLabTiming] not installed: {_e}")


from .save_nvenc import ComfyLabSaveVideoNVENC  # noqa: E402
from . import h3_refine as _h3_refine  # noqa: E402


NODE_CLASS_MAPPINGS = {
    "ComfyLabSaveVideoNVENC": ComfyLabSaveVideoNVENC,
    "ComfyLabWanFaceRedraw": ComfyLabWanFaceRedraw,
    "ComfyLabFacePickIndex": ComfyLabFacePickIndex,
    "ComfyLabSmallFaceCount": ComfyLabSmallFaceCount,
    **_h3_refine.NODE_CLASS_MAPPINGS,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ComfyLabWanFaceRedraw": "ComfyLab Wan Face Redraw",
    "ComfyLabFacePickIndex": "ComfyLab Face Pick (person N)",
    "ComfyLabSmallFaceCount": "ComfyLab Small Face Count",
    "ComfyLabSaveVideoNVENC": "ComfyLab Save Video (NVENC)",
    **_h3_refine.NODE_DISPLAY_NAME_MAPPINGS,
}
