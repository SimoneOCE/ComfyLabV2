import runpod
import subprocess
import threading
import time
import requests
import os
import json
import shutil
import uuid
import boto3
from botocore.client import Config

COMFYUI_URL = "http://127.0.0.1:8188"
VOLUME_DIR = "/runpod-volume"

# ComfyUI + KJNodes are baked into the image (see Dockerfile) - unlike
# speedlabv2's koboldcpp_engine, there's no extraction/build step for these,
# they're just already there when the container starts.
COMFYUI_DIR = "/opt/comfylab/ComfyUI"
ENGINE_SCRIPT = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "comfyui_engine", "ensure_comfyui_engine.sh"
)
WORKFLOW_TEMPLATE_PATH = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "comfyui_engine", "workflow_template.json"
)

# SageAttention + the MiniMax H3 model weights live on the persistent volume
# (see ensure_comfyui_engine.sh - marker-gated, built/downloaded once per
# volume, every later worker boot just reuses them). Model weights
# specifically go under VOLUME_DIR/models rather than COMFYUI_DIR/models
# because COMFYUI_DIR is inside this worker's own container filesystem,
# which doesn't survive past its lifetime - symlink_models_to_volume() below
# points ComfyUI's normal model lookup paths at the volume copies instead of
# re-downloading ~40GB on every cold boot.
VOLUME_MODELS_DIR = os.path.join(VOLUME_DIR, "models")

OUTPUT_DIR = os.path.join(VOLUME_DIR, "outputs")

# Wasabi (third-party S3-compatible storage), NOT RunPod's own S3-compatible
# volume storage - that only exists in 15 specific datacenters (see RunPod's
# docs), and EUR-IS-2 (the one datacenter with real RTX 5090 + CUDA 13.0
# availability) isn't one of them, confirmed the hard way with a real
# EndpointConnectionError against a guessed s3api-eur-is-2.runpod.io hostname
# that doesn't even resolve. Wasabi decouples storage from whichever
# datacenter the GPU worker happens to land in - the bucket lives in Wasabi's
# own eu-west-1 (UK), picked as the shortest real network path from Iceland
# (FARICE-1 submarine cable runs Iceland -> Scotland, backhauled to London).
S3_ACCESS_KEY = os.environ.get("S3_ACCESS_KEY")
S3_SECRET_KEY = os.environ.get("S3_SECRET_KEY")
S3_ENDPOINT = os.environ.get("S3_ENDPOINT", "https://s3.eu-west-1.wasabisys.com")
S3_BUCKET = os.environ.get("S3_BUCKET", "comfylab-outputs")
S3_REGION = os.environ.get("S3_REGION", "eu-west-1")

s3_client = boto3.client(
    "s3",
    endpoint_url=S3_ENDPOINT,
    aws_access_key_id=S3_ACCESS_KEY,
    aws_secret_access_key=S3_SECRET_KEY,
    region_name=S3_REGION,
    config=Config(signature_version="s3v4"),
)

# Node IDs from the real API-format export (video_minimax_h3_t2v_3.json,
# captured off the proven RunPod CUDA 13.0 pod). This graph is TEXT-TO-VIDEO
# ONLY - MiniMaxH3ImageToVideo (105:104) has no first_frame/last_frame wired
# in this export, so I2V isn't supported by this handler yet. Adding it
# means capturing a second export with those inputs connected and branching
# build_prompt_payload() on whether the job provides an image.
NODE_IDS = {
    "prompt_and_dims": "105:104",  # MiniMaxH3ImageToVideo - prompt/width/height/length inputs
    "duration_seconds": "105:111", # PrimitiveFloat feeding ComfyMathExpression's 17n+5 frame-count snap
    "seed": "105:15",              # RandomNoise - noise_seed
    "steps": "105:9",              # BasicScheduler - steps
    "output": "92",                # SaveVideo - terminal output node
}

comfyui_process = None
comfyui_process_lock = threading.Lock()

# --- Supabase (session mode only) ---------------------------------------
# Mirrors minimax-h3-worker/handler.py's own session-mode Supabase block
# exactly in mechanism (poll-based REST, service-role key, same held-open
# loop shape) - but points at comfylab_gpu_sessions/comfylab_active_gpu_
# sessions/comfylab_gpu_session_jobs, three tables added specifically for
# this test tool rather than the production gpu_sessions/active_gpu_
# sessions/gpu_session_jobs tables (which are coupled to real user auth
# and billing/credits in that same Supabase project). No user_id at all
# here - this is a single-tester local test tool, not a multi-tenant
# product, so comfylab_active_gpu_sessions uses one fixed global claim
# slot ('default') instead of one row per user.
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

SESSION_POLL_INTERVAL_SECONDS = 3
# See minimax-h3-worker/handler.py's own SESSION_IDLE_TIMEOUT_SECONDS
# comment for the full reasoning (uniform regardless of whether the
# session has ever had a job). Shorter than production's 15 minutes since
# this is a manual test tool, not a paying session someone might step
# away from mid-prompt - idling too long here just burns GPU-minutes on a
# dev test with nobody watching.
SESSION_IDLE_TIMEOUT_SECONDS = 10 * 60
SESSION_SAFETY_MAX_SECONDS = 23 * 60 * 60
HEARTBEAT_INTERVAL_SECONDS = 20

# Deliberately NOT replicating minimax-h3-worker's keep-warm ping
# (KEEPWARM_INTERVAL_SECONDS / warmup_kobold's "keep-warm" mode): that
# mechanism exists because a real investigation found koboldcpp/CUDA
# specifically loses first-inference warmup benefit after sitting idle a
# while. No equivalent investigation has been done for ComfyUI/PyTorch on
# this stack - adding an unverified periodic throwaway generation here
# would be copying a fix without copying the evidence that motivated it.
# Worth testing for real once this held-open pattern proves out.


def _sb_headers():
    return {
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "Content-Type": "application/json",
    }


def sb_get(table, params, timeout=10):
    r = requests.get(
        f"{SUPABASE_URL}/rest/v1/{table}",
        headers=_sb_headers(),
        params=params,
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()


def sb_patch(table, params, body, timeout=10):
    headers = _sb_headers()
    headers["Prefer"] = "return=representation"
    r = requests.patch(
        f"{SUPABASE_URL}/rest/v1/{table}",
        headers=headers,
        params=params,
        json=body,
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()


def sb_delete(table, params, timeout=10):
    r = requests.delete(
        f"{SUPABASE_URL}/rest/v1/{table}",
        headers=_sb_headers(),
        params=params,
        timeout=timeout,
    )
    r.raise_for_status()


def ensure_comfyui_engine():
    """Runs ensure_comfyui_engine.sh (marker-gated on the volume - see that
    script's own docstring for why SageAttention specifically has to be
    installed here rather than baked into the Dockerfile). Safe to call on
    every cold start; it no-ops past the GPU check once the marker exists.
    """
    print("Ensuring ComfyUI engine (SageAttention + model weights) is ready...")
    subprocess.run(
        ["bash", ENGINE_SCRIPT, VOLUME_DIR, COMFYUI_DIR],
        check=True,
    )


def symlink_models_to_volume():
    """Points ComfyUI's normal models/{diffusion_models,text_encoders,vae}
    lookup paths at the volume-backed copies ensure_comfyui_engine.sh just
    downloaded, instead of ComfyUI looking in its own (ephemeral) models/
    dir baked into the image.
    """
    for subdir in ("diffusion_models", "text_encoders", "vae"):
        link_path = os.path.join(COMFYUI_DIR, "models", subdir)
        target_path = os.path.join(VOLUME_MODELS_DIR, subdir)
        if os.path.islink(link_path):
            continue
        if os.path.isdir(link_path):
            # Not actually empty - ComfyUI ships a placeholder file in each
            # of these (e.g. put_diffusion_model_files_here) to keep the
            # directory tracked in git, so os.rmdir() (empty dirs only)
            # fails with ENOTEMPTY. rmtree since it's being replaced by a
            # symlink regardless of what's in it.
            shutil.rmtree(link_path)
        os.makedirs(os.path.dirname(link_path), exist_ok=True)
        os.symlink(target_path, link_path)
        print(f"Symlinked {link_path} -> {target_path}")


def is_comfyui_ready():
    try:
        r = requests.get(f"{COMFYUI_URL}/system_stats", timeout=5)
        return r.status_code == 200
    except requests.exceptions.RequestException:
        return False


def start_comfyui_if_needed():
    global comfyui_process
    with comfyui_process_lock:
        if is_comfyui_ready():
            return
        if comfyui_process is not None and comfyui_process.poll() is None:
            # Already starting, just not ready yet - fall through to the
            # wait loop below instead of spawning a second instance.
            pass
        else:
            print("Starting ComfyUI...")
            comfyui_process = subprocess.Popen(
                ["python3", "main.py", "--listen", "0.0.0.0", "--port", "8188"],
                cwd=COMFYUI_DIR,
            )

    for _ in range(180):  # up to 3 minutes for the server itself to come up
        if is_comfyui_ready():
            print("ComfyUI is ready.")
            return
        time.sleep(1)
    raise RuntimeError("ComfyUI did not become ready in time.")


def build_prompt_payload(job_input):
    with open(WORKFLOW_TEMPLATE_PATH, "r") as f:
        workflow = json.load(f)

    prompt_text = job_input["prompt"]
    width = job_input.get("width", 1280)
    height = job_input.get("height", 736)
    duration = job_input.get("duration", 7.29)
    steps = job_input.get("steps", 20)
    seed = job_input.get("seed", uuid.uuid4().int & 0xFFFFFFFF)

    prompt_node = workflow[NODE_IDS["prompt_and_dims"]]["inputs"]
    prompt_node["prompt"] = prompt_text
    # Overwrites the link to ResolutionSelector (node "115") with a literal
    # value - ResolutionSelector becomes unreachable from the output node
    # and simply won't execute, which ComfyUI tolerates fine.
    prompt_node["width"] = width
    prompt_node["height"] = height

    # The graph itself does the 17n+5 frame-count snap via ComfyMathExpression
    # (node "105:107") fed by this raw duration-in-seconds value - no need
    # to replicate that math here, just hand it the seconds.
    workflow[NODE_IDS["duration_seconds"]]["inputs"]["value"] = duration

    workflow[NODE_IDS["steps"]]["inputs"]["steps"] = steps
    workflow[NODE_IDS["seed"]]["inputs"]["noise_seed"] = seed

    return workflow


def submit_and_wait(workflow, timeout_seconds=1200):
    client_id = str(uuid.uuid4())
    resp = requests.post(
        f"{COMFYUI_URL}/prompt",
        json={"prompt": workflow, "client_id": client_id},
        timeout=30,
    )
    resp.raise_for_status()
    prompt_id = resp.json()["prompt_id"]

    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        r = requests.get(f"{COMFYUI_URL}/history/{prompt_id}", timeout=30)
        r.raise_for_status()
        history = r.json()
        if prompt_id in history:
            return history[prompt_id]
        time.sleep(2)
    raise TimeoutError(f"ComfyUI generation did not finish within {timeout_seconds}s.")


def fetch_output_video(history_entry):
    outputs = history_entry["outputs"]
    output_node = outputs[NODE_IDS["output"]]
    # Verified against ComfyUI's actual source (comfy_api/latest/_ui.py,
    # PreviewVideo.as_dict()) rather than assumed: SaveVideo's history
    # output list is keyed "images" even though it's video - that's
    # PreviewVideo's own key choice, not a ComfyLabV2 convention.
    video_info = output_node["images"][0]

    r = requests.get(
        f"{COMFYUI_URL}/view",
        params={
            "filename": video_info["filename"],
            "subfolder": video_info.get("subfolder", ""),
            "type": video_info.get("type", "output"),
        },
        timeout=60,
    )
    r.raise_for_status()
    return r.content, video_info["filename"]


def upload_result_and_get_key(raw_bytes, filename):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ext = os.path.splitext(filename)[1].lstrip(".") or "mp4"
    out_filename = f"{uuid.uuid4()}.{ext}"
    filepath = os.path.join(OUTPUT_DIR, out_filename)

    with open(filepath, "wb") as f:
        f.write(raw_bytes)

    content_type = "video/mp4" if ext == "mp4" else "application/octet-stream"
    key = f"outputs/{out_filename}"
    s3_client.upload_file(filepath, S3_BUCKET, key, ExtraArgs={"ContentType": content_type})
    return out_filename


def run_generation(job_input):
    """One full generation: build the workflow, run it, upload the result.
    Shared by both the classic one-shot handler() path and run_session()'s
    per-job loop below - identical either way, since ComfyUI itself only
    ever has one thing loaded/running at a time regardless of which path
    queued it.
    """
    workflow = build_prompt_payload(job_input)
    history_entry = submit_and_wait(workflow)
    raw_bytes, filename = fetch_output_video(history_entry)
    video_key = upload_result_and_get_key(raw_bytes, filename)
    return {"videoKey": video_key}


# --- Session mode (held-open worker) -------------------------------------
# Mirrors minimax-h3-worker/handler.py's run_session() pattern: one RunPod
# job that never returns until the session ends, which is what keeps this
# worker excluded from RunPod's pool for anyone/anything else's /run call
# the whole time. New generation requests can't reach an already-busy
# worker through RunPod's own routing, so they arrive here by polling
# Supabase instead (see claim_next_queued_job). Deliberately NOT
# replicating production's cancel/force-cancel or step-level progress
# reporting - this is a pre-port speed test of the held-open pattern
# itself, not a full port of the generation UX. Add those when this
# actually gets ported.


def is_session_active(session_id):
    """A session is active only while comfylab_gpu_sessions.ended_at is
    still null AND the single active_gpu_sessions-style claim slot still
    points at THIS session (not just any claim - a later session could in
    principle have re-claimed the slot after this one ended)."""
    try:
        sessions = sb_get(
            "comfylab_gpu_sessions",
            {"id": f"eq.{session_id}", "select": "ended_at"},
        )
        if not sessions or sessions[0].get("ended_at") is not None:
            return False

        claims = sb_get(
            "comfylab_active_gpu_sessions",
            {"slot": "eq.default", "select": "session_id"},
        )
        return len(claims) > 0 and claims[0].get("session_id") == session_id
    except Exception as e:
        print(f"Could not check session state (treating as still active): {e}")
        return True


def touch_session_heartbeat(session_id):
    try:
        sb_patch(
            "comfylab_active_gpu_sessions",
            {"slot": "eq.default", "session_id": f"eq.{session_id}"},
            {"last_activity_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
        )
    except Exception as e:
        print(f"Could not write heartbeat for session {session_id}: {e}")


def mark_session_worker_started(session_id):
    try:
        sb_patch(
            "comfylab_active_gpu_sessions",
            {"slot": "eq.default", "session_id": f"eq.{session_id}"},
            {"worker_started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
        )
    except Exception as e:
        print(f"Could not mark worker started for session {session_id}: {e}")


def mark_session_ended(session_id, reason):
    try:
        sb_patch(
            "comfylab_gpu_sessions",
            {"id": f"eq.{session_id}", "ended_at": "is.null"},
            {"ended_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "end_reason": reason},
        )
    except Exception as e:
        print(f"Could not mark session {session_id} ended: {e}")
    try:
        sb_delete("comfylab_active_gpu_sessions", {"slot": "eq.default", "session_id": f"eq.{session_id}"})
    except Exception as e:
        print(f"Could not release active_gpu_sessions claim for {session_id}: {e}")


def claim_next_queued_job(session_id):
    queued = sb_get(
        "comfylab_gpu_session_jobs",
        {
            "session_id": f"eq.{session_id}",
            "status": "eq.queued",
            "order": "created_at.asc",
            "limit": "1",
        },
    )
    if not queued:
        return None

    job_row = queued[0]
    claimed = sb_patch(
        "comfylab_gpu_session_jobs",
        {"id": f"eq.{job_row['id']}", "status": "eq.queued"},
        {"status": "processing"},
    )
    if not claimed:
        return None
    return claimed[0]


def finish_job(job_row_id, output):
    status = "failed" if output.get("error") else "completed"
    try:
        sb_patch(
            "comfylab_gpu_session_jobs",
            {"id": f"eq.{job_row_id}"},
            {"status": status, "output": output},
        )
    except Exception as e:
        print(f"Could not write final result for job {job_row_id}: {e}")


def run_session(session_id):
    """The held-open loop - see the module docstring above this section."""
    session_start = time.time()
    last_activity = time.time()
    last_heartbeat = 0.0
    jobs_processed = 0
    print(f"Session {session_id}: held-open loop starting.")

    try:
        ensure_comfyui_engine()
        symlink_models_to_volume()
        start_comfyui_if_needed()
    except Exception as e:
        print(f"Session {session_id}: ComfyUI failed to start ({e}) - ending session.")
        mark_session_ended(session_id, "error")
        return {
            "sessionEnded": True,
            "reason": "worker_error",
            "session_id": session_id,
            "jobs_processed": 0,
            "session_duration_seconds": round(time.time() - session_start, 1),
            "last_error": str(e),
        }

    # Absorbs the one-time model-load/CUDA warmup cost here, during the
    # "Starting..." wait, instead of the user's first real prompt - a tiny
    # 1-step throwaway generation at the model's minimum practical size.
    # Mirrors warmup_kobold()'s role in minimax-h3-worker/handler.py.
    warmup_seconds = None
    try:
        warmup_start = time.time()
        run_generation({
            "prompt": "warmup",
            "width": 320,
            "height": 320,
            "duration": 1.0,
            "steps": 1,
        })
        warmup_seconds = round(time.time() - warmup_start, 1)
        print(f"Session {session_id}: warmup generation done ({warmup_seconds}s).")
    except Exception as e:
        print(f"Session {session_id}: warmup generation failed, continuing anyway ({e}).")

    mark_session_worker_started(session_id)

    def session_summary(reason, **extra):
        summary = {
            "sessionEnded": True,
            "reason": reason,
            "session_id": session_id,
            "jobs_processed": jobs_processed,
            "warmup_seconds": warmup_seconds,
            "session_duration_seconds": round(time.time() - session_start, 1),
        }
        summary.update(extra)
        return summary

    while True:
        now = time.time()
        if now - last_heartbeat > HEARTBEAT_INTERVAL_SECONDS:
            touch_session_heartbeat(session_id)
            last_heartbeat = now

        if now - session_start > SESSION_SAFETY_MAX_SECONDS:
            print(f"Session {session_id}: hit the {SESSION_SAFETY_MAX_SECONDS}s safety cutoff, ending.")
            mark_session_ended(session_id, "safety_timeout")
            return session_summary("safety_timeout")

        if not is_session_active(session_id):
            print(f"Session {session_id}: no longer active, ending loop.")
            return session_summary("stopped")

        try:
            job_row = claim_next_queued_job(session_id)
        except Exception as e:
            print(f"Session {session_id}: could not check queue: {e}")
            job_row = None

        if job_row is None:
            if time.time() - last_activity > SESSION_IDLE_TIMEOUT_SECONDS:
                print(f"Session {session_id}: idle past {SESSION_IDLE_TIMEOUT_SECONDS}s, ending.")
                mark_session_ended(session_id, "timeout")
                return session_summary("timeout")
            time.sleep(SESSION_POLL_INTERVAL_SECONDS)
            continue

        print(f"Session {session_id}: processing job {job_row['id']}.")

        heartbeat_stop = threading.Event()

        def keep_heartbeat_alive():
            while not heartbeat_stop.wait(HEARTBEAT_INTERVAL_SECONDS):
                touch_session_heartbeat(session_id)

        heartbeat_thread = threading.Thread(target=keep_heartbeat_alive, daemon=True)
        heartbeat_thread.start()
        try:
            try:
                result = run_generation(job_row["input"])
            except Exception as e:
                result = {"error": str(e)}
        finally:
            heartbeat_stop.set()
            heartbeat_thread.join(timeout=5)

        finish_job(job_row["id"], result)
        jobs_processed += 1
        last_activity = time.time()


def handler(job):
    job_input = job["input"]

    # Session mode: this one RunPod job IS the held-open worker for the
    # named session - stays inside run_session() until the session ends.
    session_id = job_input.get("session_id")
    if session_id:
        return run_session(session_id)

    ensure_comfyui_engine()
    symlink_models_to_volume()
    start_comfyui_if_needed()

    return run_generation(job_input)


runpod.serverless.start({"handler": handler})
