"""ComfyLab H3 audio lock, used by the face refine pass (handler.py run_face_refine).

Puts the source clip's real audio into the AUDIO half of a MiniMax H3 joint AV
latent and masks it as "keep", so sampling only redraws video while the video
branch still attends to the real soundtrack (that's what keeps lip movement
in sync during the refine).

Our own replacement for MiniMaxH3NativeAudioLock (Shrek3OnVH5 workflow repo):
that node failed to import on the worker (its only extra dependency is a
module-level `import torchaudio`). This version uses only ComfyUI core:
comfy.audio.resample, the same helper core's own H3 reference-audio encode
uses (comfy_extras/nodes_minimax_h3.py _encode_ref_audio). It also drops the
original's "minimax_h3_lock_audio_clean" transformer option, which our pinned
ComfyUI never reads. ComfyUI-H3-FaceRefine's H3PerFrameDenoise keeps the audio
side of this node's mask as-is and handles the timestep side itself.
"""

import torch
import torch.nn.functional as F

import comfy.audio
import comfy.nested_tensor


class ComfyLabH3AudioLock:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "av_latent": ("LATENT",),
                "audio_vae": ("VAE",),
                "audio": ("AUDIO",),
            }
        }

    RETURN_TYPES = ("MODEL", "LATENT")
    RETURN_NAMES = ("model", "av_latent")
    FUNCTION = "lock_audio"
    CATEGORY = "ComfyLab/MiniMax H3"
    DESCRIPTION = "Encode real audio into an H3 AV latent and mask it so only video is denoised."

    def lock_audio(self, model, av_latent, audio_vae, audio):
        samples = av_latent.get("samples")
        if samples is None or not getattr(samples, "is_nested", False):
            raise ValueError("ComfyLabH3AudioLock needs a joint MiniMax H3 AV latent.")
        video_latent, audio_template = samples.unbind()[:2]

        waveform = audio["waveform"][:1]
        sample_rate = int(audio["sample_rate"])
        vae_rate = int(getattr(audio_vae, "audio_sample_rate", 32000))
        if sample_rate != vae_rate:
            waveform = comfy.audio.resample(waveform, sample_rate, vae_rate)

        audio_latent = audio_vae.encode(waveform.movedim(1, -1))
        target_t = audio_template.shape[-1]
        if audio_latent.shape[-1] > target_t:
            audio_latent = audio_latent[..., :target_t]
        elif audio_latent.shape[-1] < target_t:
            audio_latent = F.pad(audio_latent, (0, target_t - audio_latent.shape[-1]))
        audio_latent = audio_latent.to(device=audio_template.device, dtype=audio_template.dtype)

        locked = dict(av_latent)
        locked["samples"] = comfy.nested_tensor.NestedTensor((video_latent, audio_latent))
        locked["noise_mask"] = comfy.nested_tensor.NestedTensor(
            (torch.ones_like(video_latent), torch.zeros_like(audio_latent))
        )
        return (model, locked)


NODE_CLASS_MAPPINGS = {"ComfyLabH3AudioLock": ComfyLabH3AudioLock}
NODE_DISPLAY_NAME_MAPPINGS = {"ComfyLabH3AudioLock": "ComfyLab H3 Audio Lock"}
