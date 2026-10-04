"""Per-node run times, logging only - for finding where a refine's time goes.

Wraps ComfyUI's execution.get_output_data (the call that runs one node) to
time it, keyed by prompt id, and serves the result at
GET /comfylab/timings/{prompt_id}. Changes no values and no behaviour: the
wrapped function is awaited exactly as before. Cached nodes don't run, so
they don't appear.

Also times the parts of H3 Load Video + Face Select (from
ComfyUI-H3-FaceRefine) that happen before it checks any frame - loading
the video, importing the detector library (ultralytics), loading the
detector model - and logs them, "first in this worker" or not. A refine on
a fresh worker once spent 41.8s there instead of ~2s (job 10e921bd); these
lines say which part. Logging only: the pack's own functions run unchanged
and their results are returned as-is.
"""

import logging
import sys
import time
from collections import OrderedDict

_TIMINGS = OrderedDict()   # prompt_id -> [(node_id, class_name, seconds)]
_KEEP = 20                 # most recent prompts kept


def _log(msg):
    print(f"[ComfyLabTiming] {msg}", flush=True)
    logging.info(f"[ComfyLabTiming] {msg}")


_FACE_SELECT_PATCHED = False


def _time_face_select_loaders():
    """Wraps the pack's _load_video_components and _load_detector (module
    globals, so H3FaceSelect picks the wrappers up). Done on the first node
    run, once every custom node has been imported."""
    global _FACE_SELECT_PATCHED
    _FACE_SELECT_PATCHED = True
    try:
        import nodes
        cls = nodes.NODE_CLASS_MAPPINGS.get("H3FaceSelect")
        pack = sys.modules.get(cls.__module__) if cls is not None else None
        if pack is None:
            return
        load_video = getattr(pack, "_load_video_components", None)
        load_detector = getattr(pack, "_load_detector", None)
        if load_video is not None:
            def timed_load_video(path, *a, **k):
                start = time.perf_counter()
                out = load_video(path, *a, **k)
                frames = getattr(out[0], "shape", ["?"])[0] if out else "?"
                _log(f"face finding: video loaded in {time.perf_counter() - start:.1f}s ({frames} frames)")
                return out
            pack._load_video_components = timed_load_video
        if load_detector is not None:
            def timed_load_detector(name, *a, **k):
                cached = name in getattr(pack, "_DETECTOR_CACHE", {})
                if cached:
                    return load_detector(name, *a, **k)
                first_import = "ultralytics" not in sys.modules
                start = time.perf_counter()
                if first_import:
                    import ultralytics  # noqa: F401 - what the pack imports next; timed on its own
                    _log(f"face finding: detector library (ultralytics) imported in "
                         f"{time.perf_counter() - start:.1f}s - first import in this worker")
                start = time.perf_counter()
                out = load_detector(name, *a, **k)
                _log(f"face finding: detector {name} loaded in {time.perf_counter() - start:.1f}s"
                     f"{' - first in this worker' if first_import else ''}")
                return out
            pack._load_detector = timed_load_detector
    except Exception as e:  # logging only - never in the way of a refine
        logging.warning(f"[ComfyLabTiming] face-finding timers not installed: {e}")


def install():
    import execution
    if getattr(execution.get_output_data, "_comfylab_timed", False):
        return
    original = execution.get_output_data

    async def timed(prompt_id, unique_id, obj, *args, **kwargs):
        if not _FACE_SELECT_PATCHED:
            _time_face_select_loaders()
        start = time.perf_counter()
        try:
            return await original(prompt_id, unique_id, obj, *args, **kwargs)
        finally:
            rows = _TIMINGS.setdefault(prompt_id, [])
            rows.append((str(unique_id), type(obj).__name__, round(time.perf_counter() - start, 2)))
            while len(_TIMINGS) > _KEEP:
                _TIMINGS.popitem(last=False)

    timed._comfylab_timed = True
    execution.get_output_data = timed

    try:
        from aiohttp import web
        from server import PromptServer

        @PromptServer.instance.routes.get("/comfylab/timings/{prompt_id}")
        async def get_timings(request):
            return web.json_response(_TIMINGS.get(request.match_info["prompt_id"], []))
    except Exception as e:  # timing still logs even if the route can't be added
        logging.warning(f"[ComfyLabTiming] route not added: {e}")
