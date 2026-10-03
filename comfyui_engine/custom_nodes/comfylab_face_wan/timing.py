"""Per-node run times, logging only - for finding where a refine's time goes.

Wraps ComfyUI's execution.get_output_data (the call that runs one node) to
time it, keyed by prompt id, and serves the result at
GET /comfylab/timings/{prompt_id}. Changes no values and no behaviour: the
wrapped function is awaited exactly as before. Cached nodes don't run, so
they don't appear.
"""

import logging
import time
from collections import OrderedDict

_TIMINGS = OrderedDict()   # prompt_id -> [(node_id, class_name, seconds)]
_KEEP = 20                 # most recent prompts kept


def install():
    import execution
    if getattr(execution.get_output_data, "_comfylab_timed", False):
        return
    original = execution.get_output_data

    async def timed(prompt_id, unique_id, obj, *args, **kwargs):
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
