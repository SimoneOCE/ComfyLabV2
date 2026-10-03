"""ComfyLab debug nodes - logging only, used by the face refine graph.

Diagnosing the face refine's "KeyError: nan" in H3's rows_to_mod_index: these
nodes print tensor statistics (shape, dtype, min/max/mean, NaN/Inf counts) at
each stage of the refine chain, and what the H3 diffusion model actually
receives on its first calls. They change no values: every node passes its
input straight through, and the model hook only reads its arguments before
calling the real forward unchanged.
"""

import logging

import torch

import comfy.patcher_extension

TAG = "[ComfyLabDebug]"


def _stats(t):
    if t is None:
        return "None"
    if getattr(t, "is_nested", False):
        return " | ".join(f"[{i}] {_stats(p)}" for i, p in enumerate(t.unbind()))
    if not torch.is_tensor(t):
        return repr(t)[:200]
    if t.numel() == 0:
        return f"shape={tuple(t.shape)} dtype={t.dtype} (empty)"
    f = t.detach().float()
    nan = int(torch.isnan(f).sum())
    inf = int(torch.isinf(f).sum())
    finite = f[torch.isfinite(f)]
    if finite.numel():
        rng = f"min={finite.min().item():.6g} max={finite.max().item():.6g} mean={finite.mean().item():.6g}"
    else:
        rng = "no finite values"
    return f"shape={tuple(t.shape)} dtype={t.dtype} {rng} nan={nan} inf={inf} of {t.numel()}"


def _log(msg):
    # print() as well as logging: the worker log shows ComfyUI's stdout.
    print(f"{TAG} {msg}", flush=True)


class ComfyLabDebugImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE",), "label": ("STRING", {"default": "images"})}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "ComfyLab/debug"

    def run(self, images, label):
        _log(f"{label}: {_stats(images)}")
        return (images,)


class ComfyLabDebugLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"latent": ("LATENT",), "label": ("STRING", {"default": "latent"})}}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "run"
    CATEGORY = "ComfyLab/debug"

    def run(self, latent, label):
        _log(f"{label} samples: {_stats(latent.get('samples'))}")
        _log(f"{label} noise_mask: {_stats(latent.get('noise_mask'))}")
        return (latent,)


class ComfyLabDebugModel:
    """Logs what H3's diffusion model receives on its first `calls` forward calls."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",), "label": ("STRING", {"default": "model"}),
                             "calls": ("INT", {"default": 2, "min": 1, "max": 50})}}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "run"
    CATEGORY = "ComfyLab/debug"

    def run(self, model, label, calls):
        state = {"n": 0}

        def wrapper(executor, *args, **kwargs):
            if state["n"] < calls:
                state["n"] += 1
                try:
                    x = args[0] if args else kwargs.get("x")
                    timestep = args[1] if len(args) > 1 else kwargs.get("timestep")
                    context = args[2] if len(args) > 2 else kwargs.get("context")
                    to = args[3] if len(args) > 3 else kwargs.get("transformer_options", {})
                    _log(f"{label} call {state['n']}: timestep {_stats(timestep)}"
                         f" values={timestep.flatten()[:4].tolist() if torch.is_tensor(timestep) else timestep}")
                    if isinstance(x, (list, tuple)):
                        for i, part in enumerate(x):
                            _log(f"{label} call {state['n']}: x[{i}] {_stats(part)}")
                    else:
                        _log(f"{label} call {state['n']}: x {_stats(x)}")
                    _log(f"{label} call {state['n']}: context {_stats(context)}")
                    _log(f"{label} call {state['n']}: denoise_mask {_stats(kwargs.get('denoise_mask'))}")
                    _log(f"{label} call {state['n']}: audio_denoise_mask {_stats(kwargs.get('audio_denoise_mask'))}")
                    payload = kwargs.get("minimax_payload") or {}
                    _log(f"{label} call {state['n']}: payload keys={sorted(payload.keys())}"
                         f" audio_scale={payload.get('audio_scale')}")
                    if isinstance(to, dict):
                        shifts = {k: v for k, v in to.items() if "minimax" in k}
                        _log(f"{label} call {state['n']}: transformer_options minimax keys={shifts}")
                except Exception as e:  # logging must never break the run
                    _log(f"{label}: logging failed: {e!r}")
            return executor(*args, **kwargs)

        m = model.clone()
        m.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, "comfylab_debug", wrapper)
        return (m,)


NODE_CLASS_MAPPINGS = {
    "ComfyLabDebugImage": ComfyLabDebugImage,
    "ComfyLabDebugLatent": ComfyLabDebugLatent,
    "ComfyLabDebugModel": ComfyLabDebugModel,
}
