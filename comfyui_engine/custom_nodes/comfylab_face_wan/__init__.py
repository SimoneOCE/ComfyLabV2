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
mechanism), and frames whose mask is ~0 get their original crop back
untouched. A subject whose mask is zero everywhere is skipped outright.

Speed: every subject with the same crop size and frame count is redrawn in ONE batched Wan
call, and only over the frames that need it - leading/trailing frames where
nobody's face is small (or the face was lost) are not generated at all
(their crops come back untouched). If the batch doesn't fit in VRAM it falls
back to one subject at a time.

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
from comfy_extras.nodes_model_advanced import ModelSamplingSD3

TAG = "[ComfyLabWanFace]"
MAX_SUBJECTS = 4
KEEP_BELOW = 0.02  # mask value under which a frame keeps its original crop
RANGE_MARGIN = 4   # extra frames kept either side of the active range, as context


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


def face_strength_curve(transform, n, face_px_small, face_px_large, smooth_frames):
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
    return np.clip(strength, 0.0, 1.0), face


class ComfyLabWanFaceRedraw:
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

    RETURN_TYPES = ("IMAGE",) * MAX_SUBJECTS + ("STRING",)
    RETURN_NAMES = tuple(f"crops_{i}" for i in range(MAX_SUBJECTS)) + ("report",)
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
            strength, face = face_strength_curve(transform, n, face_px_small, face_px_large, smooth_frames)
            line = (f"subject {i}: {n} frames, face {face.min():.0f}-{face.max():.0f}px, "
                    f"strength max {strength.max():.2f} mean {strength.mean():.2f}, "
                    f"kept as-is {int((strength < KEEP_BELOW).sum())}/{n} frames")
            _log(line)
            report.append(line)
            work.append((i, crops, strength))

        outputs = {i: crops for i, crops, _ in work}
        todo = [(i, c, s) for i, c, s in work if s.max() >= KEEP_BELOW]
        if not todo:
            report.append("every tracked face is already large enough - Wan not loaded, crops unchanged")
            _log(report[-1])
            return self._result(outputs, subjects, report)

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

            groups = {}
            for item in todo:
                # Batch only crops with the same frame count AND size: the tracker
                # drops frames of shots a person isn't in, so two people can have
                # different numbers of crops even at the same canvas size.
                groups.setdefault(tuple(item[1].shape[0:3]), []).append(item)
            for (_, h, w), group in groups.items():
                t1 = time.time()
                ids = [i for i, _, _ in group]
                try:
                    redrawn, span = self._redraw_batch(model, vae, positive, negative, group,
                                                       denoise, steps, seed)
                    outputs.update(redrawn)
                    report.append(f"subjects {ids} ({w}x{h}): batched, frames {span[0]}-{span[1] - 1} "
                                  f"of {group[0][1].shape[0]} generated, {time.time() - t1:.1f}s")
                except mm.OOM_EXCEPTION:
                    if len(group) == 1:
                        raise
                    mm.soft_empty_cache(True)
                    report.append(f"subjects {ids}: batch did not fit in VRAM - one at a time")
                    _log(report[-1])
                    for item in group:
                        t2 = time.time()
                        redrawn, span = self._redraw_batch(model, vae, positive, negative, [item],
                                                           denoise, steps, seed)
                        outputs.update(redrawn)
                        report.append(f"subject {item[0]}: frames {span[0]}-{span[1] - 1} generated, "
                                      f"{time.time() - t2:.1f}s")
                        _log(report[-1])
                    continue
                _log(report[-1])
        finally:
            self._release(unet, clip, vae)
            del unet, model, clip, vae
            gc.collect()
            mm.cleanup_models()
            mm.soft_empty_cache()
            report.append(f"Wan unloaded ({_ram_gb()}, {_vram_gb()}); node total {time.time() - start:.1f}s")
            _log(report[-1])

        return self._result(outputs, subjects, report)

    @staticmethod
    def _redraw_batch(model, vae, positive, negative, group, denoise, steps, seed):
        """group: [(subject, crops [n,h,w,c], strength [n])] with one crop size.
        Generates only the span of frames where some subject's strength is
        above KEEP_BELOW (plus a little context), as one batch. Returns
        ({subject: crops}, (start, end))."""
        n, h, w = group[0][1].shape[0], group[0][1].shape[1], group[0][1].shape[2]
        if h % 16 or w % 16:
            raise ValueError(f"crop size {w}x{h} must be a multiple of 16 for Wan")
        active = np.zeros(n, dtype=bool)
        for _, _, strength in group:
            active |= strength >= KEEP_BELOW
        idx = np.flatnonzero(active)
        start = max(0, int(idx[0]) - RANGE_MARGIN)
        end = min(n, int(idx[-1]) + 1 + RANGE_MARGIN)
        span = end - start
        # Wan's video VAE takes 4k+1 frames: pad by repeating the last frame.
        span_pad = ((span - 1 + 3) // 4) * 4 + 1

        latents, masks = [], []
        for _, crops, strength in group:
            frames = crops[start:end, :, :, :3]
            if span_pad > span:
                frames = torch.cat([frames, frames[-1:].repeat(span_pad - span, 1, 1, 1)], dim=0)
            latents.append(nodes.VAEEncode().encode(vae, frames)[0]["samples"])
            m = np.pad(strength[start:end], (0, span_pad - span), mode="edge")
            masks.append(torch.from_numpy(m).float().view(1, 1, span_pad, 1, 1).expand(1, 1, span_pad, h, w))
        # The sampler's standard noise mask (SetLatentNoiseMask's mechanism):
        # comfy.utils.reshape_mask maps [B,1,T,H,W] onto the video latent.
        latent = {"samples": torch.cat(latents, dim=0), "noise_mask": torch.cat(masks, dim=0).contiguous()}
        sampled = nodes.common_ksampler(model, seed, steps, 1.0, "euler", "simple",
                                        positive, negative, latent, denoise=denoise)[0]["samples"]

        out = {}
        for b, (subject, crops, strength) in enumerate(group):
            images = nodes.VAEDecode().decode(vae, {"samples": sampled[b:b + 1]})[0][:span]
            result = crops.clone()
            result[start:end, :, :, :3] = images.to(crops.device, crops.dtype).clamp(0.0, 1.0)
            keep = torch.from_numpy(strength < KEEP_BELOW).to(result.device)
            result[keep] = crops[keep]
            out[subject] = result
        return out, (start, end)

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
    def _result(outputs, subjects, report):
        first = outputs.get(0, subjects.get("crops_0"))
        crops = tuple(outputs.get(i, first) for i in range(MAX_SUBJECTS))
        return crops + ("\n".join(report),)


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
                             "index": ("INT", {"default": 0, "min": 0, "max": 63})}}

    RETURN_TYPES = ("H3FACEPICK", "STRING")
    RETURN_NAMES = ("face_pick", "report")
    FUNCTION = "run"
    CATEGORY = "ComfyLab"

    def run(self, face_pick, index):
        select_cls = nodes.NODE_CLASS_MAPPINGS.get("H3FaceSelect")
        if select_cls is None:
            raise RuntimeError("ComfyUI-H3-FaceRefine's H3FaceSelect is not loaded")
        pack = sys.modules[select_cls.__module__]
        boxes, confs = face_pick["boxes"], face_pick["confs"]
        segs = [(int(a), int(b)) for a, b in face_pick["segments"]]
        width, height = face_pick["src_size"]
        picks = pack._auto_pick(boxes, confs, segs, width, height,
                                pack._review_select("largest_face"), int(index))
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


NODE_CLASS_MAPPINGS = {
    "ComfyLabWanFaceRedraw": ComfyLabWanFaceRedraw,
    "ComfyLabFacePickIndex": ComfyLabFacePickIndex,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ComfyLabWanFaceRedraw": "ComfyLab Wan Face Redraw",
    "ComfyLabFacePickIndex": "ComfyLab Face Pick (person N)",
}
