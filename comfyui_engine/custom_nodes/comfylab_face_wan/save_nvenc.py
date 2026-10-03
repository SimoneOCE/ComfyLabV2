"""ComfyLab Save Video (NVENC) - SaveVideo with GPU H.264 encoding.

Core SaveVideo encodes H.264 on the CPU (libx264, crf 18); at 2x size that
is a big share of a refine. This encodes the same 8-bit BT.709 MP4 with the
GPU's NVENC encoder (h264_nvenc, constant-quality 19 - close to crf 18 in
look, somewhat larger files). If NVENC isn't usable on this worker (FFmpeg
built without it, or the container has no video driver capability), it
falls back to exactly what core SaveVideo does (libx264, crf 18), so the
output is never worse than before. The encoder used is logged and returned
in the UI result. History output shape matches SaveVideo ("images" list).
"""

import io
import logging
import math
import os
from fractions import Fraction

import av
import numpy as np
from av.video.reformatter import ColorPrimaries, ColorRange, ColorTrc

import folder_paths

NVENC_OPTIONS = {"preset": "p5", "tune": "hq", "rc": "vbr", "cq": "19", "b": "0"}
CPU_OPTIONS = {"crf": "18"}
BT709_NCL = 1  # same constant core's video_types.py uses
_NVENC_OK = None


def nvenc_available():
    """Encodes one small frame once per process to see if NVENC really works."""
    global _NVENC_OK
    if _NVENC_OK is None:
        try:
            buf = io.BytesIO()
            with av.open(buf, mode="w", format="mp4") as out:
                stream = out.add_stream("h264_nvenc", rate=24)
                stream.width, stream.height, stream.pix_fmt = 256, 256, "yuv420p"
                stream.options = dict(NVENC_OPTIONS)
                frame = av.VideoFrame.from_ndarray(np.zeros((256, 256, 3), np.uint8), format="rgb24")
                for p in stream.encode(frame.reformat(format="yuv420p")):
                    out.mux(p)
                for p in stream.encode(None):
                    out.mux(p)
            _NVENC_OK = True
        except Exception as e:
            logging.warning(f"[ComfyLabSaveVideoNVENC] NVENC not usable, using CPU libx264: {e}")
            _NVENC_OK = False
    return _NVENC_OK


class ComfyLabSaveVideoNVENC:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"video": ("VIDEO",),
                             "filename_prefix": ("STRING", {"default": "video/ComfyLab"})}}

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "ComfyLab"

    def save(self, video, filename_prefix):
        comps = video.get_components()
        images = comps.images
        height, width = int(images.shape[1]), int(images.shape[2])
        folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory(), width, height)
        file = f"{filename}_{counter:05}_.mp4"
        path = os.path.join(folder, file)

        use_nvenc = nvenc_available()
        encoder = "h264_nvenc" if use_nvenc else "h264"
        frame_rate = Fraction(round(float(comps.frame_rate) * 1000), 1000)
        with av.open(path, mode="w", format="mp4", options={"movflags": "use_metadata_tags+faststart"}) as out:
            stream = out.add_stream(encoder, rate=frame_rate)
            stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
            stream.options = dict(NVENC_OPTIONS if use_nvenc else CPU_OPTIONS)
            cc = stream.codec_context   # BT.709 limited range, as core SaveVideo writes sRGB video
            cc.color_primaries, cc.color_trc, cc.colorspace = ColorPrimaries.BT709, ColorTrc.BT709, BT709_NCL
            cc.color_range = ColorRange.MPEG

            audio = comps.audio
            audio_stream = None
            if audio:
                rate = int(audio["sample_rate"])
                waveform = audio["waveform"][0, :, :math.ceil((rate / frame_rate) * images.shape[0])]
                layout = {1: "mono", 2: "stereo", 6: "5.1"}.get(waveform.shape[0], "stereo")
                audio_stream = out.add_stream("aac", rate=rate, layout=layout)

            for frame in images:
                img = (frame * 255).clamp(0, 255).byte().cpu().numpy()
                vf = av.VideoFrame.from_ndarray(img, format="rgb24").reformat(
                    format="yuv420p", dst_colorspace=BT709_NCL)
                for p in stream.encode(vf):
                    out.mux(p)
            for p in stream.encode(None):
                out.mux(p)

            if audio_stream is not None:
                af = av.AudioFrame.from_ndarray(waveform.float().cpu().contiguous().numpy(), format="fltp", layout=layout)
                af.sample_rate = rate
                af.pts = 0
                for p in audio_stream.encode(af):
                    out.mux(p)
                for p in audio_stream.encode(None):
                    out.mux(p)

        logging.info(f"[ComfyLabSaveVideoNVENC] {file}: {width}x{height}, {images.shape[0]} frames, encoder {encoder}")
        print(f"[ComfyLabSaveVideoNVENC] {file}: encoder {encoder}", flush=True)
        return {"ui": {"images": [{"filename": file, "subfolder": subfolder, "type": "output"}],
                       "animated": (True,), "encoder": [encoder]}}
