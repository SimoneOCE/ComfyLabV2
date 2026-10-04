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
                          follows the real audio. Core audio VAE + core
                          resample only (the old MiniMaxH3NativeAudioLock
                          needed torchaudio).
  ComfyLabStrengthWeights - per-frame stitch weights from the same strength
                          curve H3PerFrameDenoise uses, so frames it leaves
                          alone keep the video's own pixels (not a VAE
                          round-trip of them).
  ComfyLabH3StepCheck   - logs the timestep H3 receives and stops the job at
                          the first step with a bad timestep or NaN output,
                          instead of failing later with "KeyError: nan".
"""

import logging
import math

import numpy as np
import torch

import comfy.audio
import comfy.nested_tensor
import comfy.patcher_extension

TAG = "[ComfyLabH3Refine]"
WEIGHT_RAMP = 0.2  # same hand-over as the Wan engine: full paste once strength reaches this


def _log(msg):
    print(f"{TAG} {msg}", flush=True)
    logging.info(f"{TAG} {msg}")


class ComfyLabH3AudioLock:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"av_latent": ("LATENT",), "audio_vae": ("VAE",), "audio": ("AUDIO",)}}

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("av_latent", "report")
    FUNCTION = "run"
    CATEGORY = "ComfyLab"

    def run(self, av_latent, audio_vae, audio):
        samples = av_latent["samples"]
        if not isinstance(samples, comfy.nested_tensor.NestedTensor):
            raise ValueError("Expected a MiniMax H3 joint AV latent (NestedTensor)")
        video, latent_audio = list(samples.unbind())[:2]
        waveform = audio["waveform"]
        sr = int(audio["sample_rate"])
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
            z = torch.nn.functional.pad(z, (0, t_need - t_got), mode="replicate")
        if not torch.isfinite(z).all():
            raise RuntimeError("The clip's audio encoded to non-finite values")
        z = z.expand(latent_audio.shape[0], -1, -1, -1).contiguous()
        out = dict(av_latent)
        out["samples"] = comfy.nested_tensor.NestedTensor((video, z))
        out["noise_mask"] = comfy.nested_tensor.NestedTensor((torch.ones_like(video), torch.zeros_like(z)))
        report = (f"audio lock: {waveform.shape[-1] / vae_sr:.2f}s at {vae_sr}Hz -> {t_got} latent frames "
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


NODE_CLASS_MAPPINGS = {
    "ComfyLabH3AudioLock": ComfyLabH3AudioLock,
    "ComfyLabStrengthWeights": ComfyLabStrengthWeights,
    "ComfyLabH3StepCheck": ComfyLabH3StepCheck,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ComfyLabH3AudioLock": "ComfyLab H3 Audio Lock",
    "ComfyLabStrengthWeights": "ComfyLab Strength -> Stitch Weights",
    "ComfyLabH3StepCheck": "ComfyLab H3 Step Check",
}
