"""RTX Video Super Resolution with NVIDIA's full quality-mode list.

Same processing as Comfy-Org/Nvidia_RTX_Nodes_ComfyUI's RTXVideoSuperResolution
(pinned at 892515e - execute() below mirrors its loop line for line), except
the quality input takes any upscaling mode by name from nvvfx's own
QualityLevel enum. The official node only offers LOW/MEDIUM/HIGH/ULTRA, which
NVIDIA documents as upscale + compression-artifact suppression for
streamed/compressed video; the HIGHBITRATE_* modes are their upscale for
clean sources (like freshly generated frames) that skip that suppression.
DENOISE_*/DEBLUR_* are left out on purpose: they only work at the same
resolution, not as an upscale.
"""

import torch
import nvvfx
from typing_extensions import override

from comfy_api.latest import ComfyExtension, io

QUALITY_MODES = [
    "LOW", "MEDIUM", "HIGH", "ULTRA",
    "HIGHBITRATE_LOW", "HIGHBITRATE_MEDIUM", "HIGHBITRATE_HIGH", "HIGHBITRATE_ULTRA",
]


class ComfyLabRTXVideoSuperResolution(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ComfyLabRTXVideoSuperResolution",
            display_name="RTX Video Super Resolution (all modes)",
            category="image/upscaling",
            inputs=[
                io.Image.Input("images"),
                io.Float.Input("scale", default=2.0, min=1.0, max=4.0, step=0.01),
                io.Combo.Input("quality", options=QUALITY_MODES, default="HIGHBITRATE_ULTRA"),
            ],
            outputs=[io.Image.Output("upscaled_images")],
        )

    @classmethod
    def execute(cls, images: torch.Tensor, scale: float, quality: str) -> io.NodeOutput:
        b, h, w, c = images.shape

        output_width = max(8, round(int(w * scale) / 8) * 8)
        output_height = max(8, round(int(h * scale) / 8) * 8)

        MAX_PIXELS = 1024 * 1024 * 16
        batch_size = max(1, MAX_PIXELS // (output_width * output_height))

        # By name, so a mode missing from the installed nvidia-vfx fails
        # loudly (AttributeError) instead of silently falling back.
        selected_quality = getattr(nvvfx.effects.QualityLevel, quality)

        with nvvfx.VideoSuperRes(selected_quality) as sr:
            sr.output_width = output_width
            sr.output_height = output_height
            sr.load()

            out_tensor = torch.empty((images.shape[0], output_height, output_width, c), device=images.device, dtype=images.dtype)
            for i in range(0, images.shape[0], batch_size):
                batch = images[i:i + batch_size]

                batch_cuda = batch.cuda().permute(0, 3, 1, 2).float().contiguous()

                for j in range(batch_cuda.shape[0]):
                    input_frame = batch_cuda[j]
                    dlpack_out = sr.run(input_frame).image
                    out_tensor[i + j: i + j + 1] = torch.from_dlpack(dlpack_out).movedim(0, -1).unsqueeze(0)

        return io.NodeOutput(out_tensor)


class ComfyLabRTXVSRExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [ComfyLabRTXVideoSuperResolution]


async def comfy_entrypoint() -> ComfyLabRTXVSRExtension:
    return ComfyLabRTXVSRExtension()
