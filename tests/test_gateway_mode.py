"""Worker gateway mode, run locally with no GPU, no ComfyUI and no keys.

A fake website (the /api/worker/* routes, server.js "Worker gateway") runs
on 127.0.0.1; handler.py's ComfyUI steps are stubbed. Checks that a session
with a pass runs entirely through the gateway (start, marks, heartbeats,
claim, input links, upload link, progress, result, end), that the worker
has no master-key code left (2026-10-12), and that a job without a pass, or
with a malformed one, is refused before anything runs.

    python3 tests/test_gateway_mode.py
"""
import json
import os
import sys
import tempfile
import threading
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Stand-ins for packages the GPU image has but a laptop may not.
for name in ("runpod",):
    if name not in sys.modules:
        try:
            __import__(name)
        except ImportError:
            mod = types.ModuleType(name)
            sys.modules[name] = mod
sys.modules["runpod"].serverless = getattr(sys.modules["runpod"], "serverless", types.SimpleNamespace(start=lambda *_: None))

os.environ.pop("SUPABASE_SERVICE_ROLE_KEY", None)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import handler  # noqa: E402

SESSION = "11111111-2222-4333-8444-555555555555"
OWNER = "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"
JOB = "99999999-8888-4777-8666-555555555555"
PASS = "p" * 43
IMG = f"inputs/{OWNER}/12345678-1234-4123-8123-123456789012.png"
calls = []
state = {"active": True, "job_given": False, "uploaded": {}, "finished": None, "progress": [], "marks": [], "end": None}


class Fake(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/file/img":
            data = b"\x89PNG fake picture"
            self.send_response(200)
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        else:
            self._send(404, {})

    def do_PUT(self):
        n = int(self.headers.get("content-length") or 0)
        state["uploaded"][self.path] = (self.rfile.read(n), self.headers.get("content-type"))
        self._send(200, {})

    def do_POST(self):
        n = int(self.headers.get("content-length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        path = self.path
        calls.append(path)
        if self.headers.get("authorization") != f"Bearer {PASS}":
            return self._send(401, {"error": "Not allowed"})
        if path == "/api/worker/start":
            if body.get("runpodJobId") == "rp-blocked":
                return self._send(403, {"error": "This GPU machine is blocked", "blocked": True})
            if body.get("runpodJobId") != "rp-1":
                return self._send(403, {})
            return self._send(200, {"sessionId": SESSION, "userId": OWNER, "isTrial": False})
        if path == "/api/worker/state":
            return self._send(200, {"active": state["active"]})
        if not state["active"] and path not in ("/api/worker/end",) and "/finish" not in path and "/flags" not in path:
            return self._send(410, {"active": False})
        if path == "/api/worker/mark":
            state["marks"].append(body["event"])
            return self._send(200, {"ok": True})
        if path in ("/api/worker/heartbeat", "/api/worker/alert"):
            return self._send(200, {"ok": True})
        if path == "/api/worker/end":
            state["end"] = body.get("reason")
            return self._send(200, {"ok": True})
        if path == "/api/worker/jobs/claim":
            if state["job_given"]:
                return self._send(200, {"job": None})
            state["job_given"] = True
            return self._send(200, {"job": {"id": JOB, "session_id": SESSION, "user_id": OWNER,
                                            "input": {"v": 2, "mode": "generate"}, "cancel_requested": False}})
        if path == f"/api/worker/jobs/{JOB}/flags":
            return self._send(200, {"cancel_requested": False, "force_cancel_requested": False})
        if path == f"/api/worker/jobs/{JOB}/inputs":
            return self._send(200, {"files": {IMG: {"url": f"{BASE}/file/img", "size": 17}}, "refineNotAllowed": False})
        if path == f"/api/worker/jobs/{JOB}/progress":
            state["progress"].append(body["progress"])
            return self._send(200, {"ok": True})
        if path == f"/api/worker/jobs/{JOB}/upload":
            return self._send(200, {"url": f"{BASE}/put/out.mp4", "key": "12345678-aaaa-4bbb-8ccc-123456789012.mp4",
                                    "headers": {"Content-Type": "video/mp4"}})
        if path == f"/api/worker/jobs/{JOB}/finish":
            state["finished"] = body["output"]
            state["finish_body"] = body
            state["active"] = False  # the user ends the session after their video
            return self._send(200, {"written": True})
        self._send(404, {})


server = ThreadingHTTPServer(("127.0.0.1", 0), Fake)
BASE = f"http://127.0.0.1:{server.server_address[1]}"
threading.Thread(target=server.serve_forever, daemon=True).start()

tmp = tempfile.mkdtemp()
handler.OUTPUT_DIR = os.path.join(tmp, "out")
handler.COMFYUI_INPUT_DIR = os.path.join(tmp, "in")
handler.SESSION_POLL_INTERVAL_SECONDS = 0.05
for name in ("ensure_comfyui_engine", "symlink_models_to_volume", "start_comfyui_if_needed"):
    setattr(handler, name, lambda *a, **k: None)
handler.run_warmup = lambda *a, **k: None
handler.is_comfyui_ready = lambda: True
handler.validate_job = lambda job_input, owner_id, is_trial=False: {"mode": "generate", "trial": False, "seed": 1}


def fake_generation(job, report, should_cancel=None, should_force_kill=None, loaded_model=None):
    report({"stage": "Preparing your inputs"})
    assert handler._object_size(IMG) == 17
    os.makedirs(handler.COMFYUI_INPUT_DIR, exist_ok=True)
    path = os.path.join(handler.COMFYUI_INPUT_DIR, "pic")
    handler.storage_download(IMG, path, 1000)
    assert open(path, "rb").read().startswith(b"\x89PNG")
    assert not should_cancel() and not should_force_kill()
    report({"stage": "Saving"})
    key = handler.upload_result_and_get_key(b"finished video bytes")
    # As a trial video would: the clean copy's key beside the result.
    return {"videoKey": key, "storage": "wasabi", "kind": "generate", "watermarked": True, "seed": 1,
            "cleanKey": "99999999-aaaa-4bbb-8ccc-123456789012.mp4"}, "base"


handler.run_generation = fake_generation

failures = 0


def check(name, cond, extra=None):
    global failures
    print(("  ok   " if cond else "  FAIL ") + name + ("" if cond else f"  {extra}"))
    failures += 0 if cond else 1


print("gateway mode")
result = handler.handler({"id": "rp-1", "input": {"session_id": SESSION, "gateway": {"url": BASE, "pass": PASS}}})
check("the session ran and ended when the website said so", result.get("reason") == "stopped" and result.get("jobs_processed") == 1, result)
check("start was asked first, with the RunPod job id", calls and calls[0] == "/api/worker/start", calls[:3])
check("execution and ready were marked", state["marks"][:2] == ["execution_started", "worker_started"], state["marks"])
check("the picture came through its read link", any(p.endswith("/inputs") for p in calls))
check("the video went to the upload link, as MP4", state["uploaded"].get("/put/out.mp4", (b"", ""))[1] == "video/mp4"
      and state["uploaded"]["/put/out.mp4"][0] == b"finished video bytes", state["uploaded"])
check("the result names the key the website picked", (state["finished"] or {}).get("videoKey") == "12345678-aaaa-4bbb-8ccc-123456789012.mp4", state["finished"])
check("a trial video's clean copy is sent beside the result, never inside it",
      (state.get("finish_body") or {}).get("cleanKey") == "99999999-aaaa-4bbb-8ccc-123456789012.mp4"
      and "cleanKey" not in (state["finished"] or {}), state.get("finish_body"))
check("the worker no longer asks whether to skip the watermark", not any(p.endswith("/watermark") for p in calls), calls)
check("progress was sent", [p.get("stage") for p in state["progress"]][:1] == ["Preparing your inputs"], state["progress"])
check("the pass is forgotten after the job", handler._GW is None)

print("saving a finished video (2026-10-13)")
uploads = []
fail_first = [False]


def fake_upload(raw):
    if fail_first[0]:
        fail_first[0] = False
        raise RuntimeError("upload link refused")
    uploads.append(raw)
    return f"{len(uploads):08d}-aaaa-4bbb-8ccc-123456789012.mp4"


real_upload, real_watermark = handler.upload_result_and_get_key, handler.apply_watermark
handler.upload_result_and_get_key = fake_upload
handler.apply_watermark = lambda raw: b"WM:" + raw
video, clean, marked = handler.save_result({"trial": True}, b"raw")
check("trial: the clean file goes first, then the watermarked one; the result is the watermarked one",
      uploads == [b"raw", b"WM:raw"] and marked is True and clean == "00000001-aaaa-4bbb-8ccc-123456789012.mp4"
      and video == "00000002-aaaa-4bbb-8ccc-123456789012.mp4", (uploads, video, clean))
uploads.clear()
fail_first[0] = True
video, clean, marked = handler.save_result({"trial": True}, b"raw")
check("trial, the clean upload fails: still delivered watermarked, no clean copy", uploads == [b"WM:raw"] and clean is None and marked is True, uploads)
uploads.clear()
video, clean, marked = handler.save_result({"trial": False}, b"raw")
check("a plan's video: one upload, no watermark, no clean copy", uploads == [b"raw"] and clean is None and marked is False, uploads)
handler.apply_watermark = lambda raw: (_ for _ in ()).throw(RuntimeError("watermark failed"))
uploads.clear()
try:
    handler.save_result({"trial": True}, b"raw")
    check("trial, the watermark fails: the job fails (never an unmarked video)", False, uploads)
except RuntimeError:
    check("trial, the watermark fails: the job fails (never an unmarked video)", uploads == [b"raw"], uploads)
handler.upload_result_and_get_key, handler.apply_watermark = real_upload, real_watermark

print("no master-key code left")
src = open(handler.__file__).read()
for gone in ("SUPABASE_SERVICE_ROLE_KEY\")", "sb_get", "sb_patch", "s3_client", "boto3", "_sb_headers"):
    check(f"handler.py has no {gone}", gone not in src.replace('"SUPABASE_SERVICE_ROLE_KEY", "SUPABASE_URL"', ""))

print("refusals")
calls.clear()
r = handler.handler({"id": "rp-1", "input": {"session_id": SESSION}})
check("a job without a pass -> refused before anything runs", "error" in r and calls == [], (r, calls))
r = handler.handler({"id": "rp-1", "input": {"session_id": SESSION, "gateway": {"url": "http://evil.example", "pass": PASS}}})
check("a gateway address that isn't https -> refused", "error" in r, r)
r = handler.handler({"id": "rp-1", "input": {"session_id": SESSION, "gateway": {"url": BASE, "pass": "short"}}})
check("a malformed pass -> refused", "error" in r, r)
calls.clear()
state.update(active=True, job_given=True)
handler.RUNPOD_JOB_ID_WAIT_SECONDS = 1
r = handler.handler({"id": "rp-other", "input": {"session_id": SESSION, "gateway": {"url": BASE, "pass": PASS}}})
check("another RunPod job -> the gateway refuses, nothing runs", r.get("reason") == "not_active" and calls == ["/api/worker/start"], (r, calls))
r = handler.handler({"id": "rp-1", "input": {"session_id": SESSION, "gateway": {"url": BASE, "pass": "q" * 43}}})
check("a wrong pass -> refused, nothing runs", r.get("reason") == "not_active", r)

calls.clear()
r = handler.handler({"id": "rp-blocked", "input": {"session_id": SESSION, "gateway": {"url": BASE, "pass": PASS}}})
check("a blocked machine -> nothing runs, and RunPod is asked to replace this worker",
      r.get("reason") == "machine_blocked" and r.get("refresh_worker") is True and calls == ["/api/worker/start"], (r, calls))
r = handler.handler({"id": "rp-other", "input": {"session_id": SESSION, "gateway": {"url": BASE, "pass": PASS}}})
check("an ordinary refusal doesn't replace the worker", "refresh_worker" not in r, r)

print("\nall passed" if not failures else f"\n{failures} FAILED")
sys.exit(1 if failures else 0)
