"""Bizzle.ai production GPU worker (ComfyUI + MiniMax H3).

One RunPod job = one held-open GPU session for one user (see run_session).
The website (minimax-h3-website/server.js) starts the session, validates
every job, and queues it in Supabase's gpu_session_jobs; this worker polls
that queue, re-validates everything (it is the last line of defence: the
gpu_session_jobs table, the RunPod API key or the website could all be
misused or buggy), runs the job on ComfyUI and writes the result back.

Models (only one MiniMax H3 is ever resident - ~21GB each, 89GB of RAM):
  - base (FL2VA): loaded at session start. Text-to-video, start/end frames,
    reference pictures, and pictures + frames (MiniMaxH3AddGuide).
  - Ref2VA: motion swap only. Swapped in automatically when a job with a
    reference video arrives (~23-28s), swapped back by the next base job.
The user never picks a model, LoRA, attention mode or engine: the worker
derives all of it from the job's mode and "speed".

History: this file started as the ComfyLabV2 test worker. Test-only paths
(the standalone upscale job, the upscale-only session job, the model/LoRA/
attention/refine-engine pickers, DaSiWa, the comfylab_* tables) are gone -
see MERGE_NOTES.md "Ported to production".
"""

import runpod
import subprocess
import threading
import time
import requests
import os
import json
import re
import shutil
import urllib.parse
import uuid

COMFYUI_URL = "http://127.0.0.1:8188"
COMFYUI_WS_URL = "ws://127.0.0.1:8188/ws"
VOLUME_DIR = "/runpod-volume"

# ComfyUI + KJNodes + H3-FaceRefine are baked into the image (see Dockerfile).
COMFYUI_DIR = "/opt/comfylab/ComfyUI"
COMFYUI_INPUT_DIR = os.path.join(COMFYUI_DIR, "input")
ENGINE_SCRIPT = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "comfyui_engine", "ensure_comfyui_engine.sh"
)
WORKFLOW_TEMPLATE_PATH = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "comfyui_engine", "workflow_template.json"
)

# SageAttention's wheel and every model file live on the persistent volume
# (ensure_comfyui_engine.sh, marker-gated); symlink_models_to_volume() points
# ComfyUI's model folders at them.
VOLUME_MODELS_DIR = os.path.join(VOLUME_DIR, "models")
OUTPUT_DIR = os.path.join(VOLUME_DIR, "outputs")
# ComfyUI's own output dir, redirected off the small container disk; each file
# is deleted once uploaded (fetch_output_video).
COMFYUI_OUTPUT_DIR = os.path.join(VOLUME_DIR, "comfyui_output")

# --- Storage --------------------------------------------------------------
# outputs/<uuid>.mp4  - finished videos; inputs/<user_id>/<uuid>.<ext> - the
# user's uploads. The worker holds no storage keys (2026-10-12): it reads and
# writes only through short-lived links the website's gateway gives it for
# the current job (gw_download / gw_upload below).

# Node ids from workflow_template.json (the API export of the proven graph).
NODE_IDS = {
    "prompt_and_dims": "105:104",  # MiniMaxH3ImageToVideo, or MiniMaxH3ReferenceToVideo with references
    "duration_seconds": "105:111", # PrimitiveFloat feeding the 17n+5 frame snap (ComfyMathExpression 105:107)
    "seed": "105:15",              # RandomNoise
    "steps": "105:9",              # BasicScheduler
    "unet_loader": "105:6",        # UNETLoader
    "clip_loader": "105:13",       # CLIPLoader
    "vae_loader": "105:11",        # VAELoader (video)
    "audio_vae_loader": "105:24",  # VAELoader (audio)
    "sage_attention": "105:120",   # PathchSageAttentionKJ (always on in production)
    "guider": "105:16",            # BasicGuider
    "sampler": "105:14",           # SamplerCustomAdvanced
    "video_decode": "105:10",      # VAEDecode
    "create_video": "105:91",      # CreateVideo
    "output": "92",                # SaveVideo
}

MODEL_CHOICES = {
    "base": {"filename": "minimax_h3_fl2va_pruned_int8_convrot.safetensors"},
    # Motion swap only. Public repo; downloaded to the volume by the first
    # swap job if it isn't there yet (never at session start).
    "ref2va": {
        "filename": "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
        "repo": "Comfy-Org/MiniMax-H3",
        "repo_path": "diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors",
        "reference_node": True,
    },
}

# Fixed map - never chosen by the user directly (see plan_generation).
LORA_CHOICES = {
    # base: Comfy-Org's 8-step turbo ("Turbo" speed on the site).
    "turbo": {
        "filename": "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
        "multiplier": 1.0,
        "steps": 8,
    },
    # Ref2VA 4-step v0.1 (544p-trained): the motion swap's "Fast" option.
    "ref2v_turbo": {
        "filename": "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
        "repo": "Comfy-Org/MiniMax-H3",
        "repo_path": "loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
        "multiplier": 1.0,
        "steps": 4,
    },
    # Ref2VA 8-step v1.0 (768p-trained, lightx2v): the motion swap default.
    # Trained at flow shift 6/3, so MiniMaxH3SigmaShift is spliced in.
    "ref2v_turbo_8": {
        "filename": "minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors",
        "repo": "lightx2v/Minimax-h3-Turbo",
        "repo_path": "minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors",
        "multiplier": 1.0,
        "steps": 8,
        "shift": (6.0, 3.0),
    },
}

STANDARD_STEPS = 20  # base without a turbo LoRA ("Standard" speed, the default)

# H3's native ~1MP canvas (768p class). Generations use the three user-facing
# presets; a motion swap picks the preset closest to its reference video's
# shape (incl. 4:3 / 3:4, which the user can't pick).
ASPECT_PRESETS = {"16:9": (1344, 768), "9:16": (768, 1344), "1:1": (992, 992)}
MOTION_PRESETS = [(1344, 768), (768, 1344), (992, 992), (1024, 768), (768, 1024)]

# RTX Video Super Resolution, 2x only. Spliced into the generation's own graph
# (H3 stays resident). 2x of every preset stays inside 4K and H.264 level 5.x
# (largest: 2688x1536 / 1984x1984), so the result plays on iPhone/Mac. 4x is
# not offered: it produced level 6.0 files Apple devices can't decode.
NVIDIA_VSR_SCALE = 2.0
NVIDIA_VSR_QUALITY = "ULTRA"

# --- Job limits (mirrors server.js's validateJobRequest) ------------------
PROMPT_MAX_CHARS = 20000
DURATION_MIN = 1.0
DURATION_MAX = 15.0
MAX_REF_IMAGES = 9            # <Picture 1..9>
MOTION_MAX_PICTURES = 3
MAX_FRAMES = 362              # H3's trained maximum (17n+5)
SEED_MAX = 0xFFFFFFFF
IMAGE_MAX_BYTES = 50 * 1024 * 1024
IMAGE_MAX_SIDE = 16384
IMAGE_MAX_PIXELS = 200_000_000
# Pillow only refuses images over twice MAX_IMAGE_PIXELS (and merely warns
# above it), so a ~1MB PNG declaring 16384x16384 would decode to ~1GB. This
# is checked from the header before anything is decoded: 60MP covers a 48MP
# phone photo.
IMAGE_MAX_DECODE_PIXELS = 60_000_000
REF_IMAGE_MAX_SIDE = 1536     # references are downscaled to this before use
VIDEO_MAX_BYTES = 800 * 1024 * 1024
# A face fix's source is one of our own finished videos (the gateway's
# upload cap is the same 1GB).
REFINE_SOURCE_MAX_BYTES = 1024 * 1024 * 1024
VIDEO_MAX_SIDE = 4096
REF_VIDEO_MAX_SECONDS = 15.5  # H3's trained range tops out at 15s
UUID_RE = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
OUTPUT_KEY_RE = re.compile(rf"^{UUID_RE}\.mp4$")

# Free trial (gpu_sessions.is_trial, set by server.js at session start — never
# taken from the job): generate only, at most 5s (3s until 2026-10-12) and 2 pictures, no start/end
# frames, no motion swap, no upscale, no face fix, and a burned-in watermark.
# server.js and the job-input trigger enforce the same rules first.
TRIAL_MAX_DURATION = 5.0
TRIAL_MAX_PICTURES = 2

GENERATE_FIELDS = {"v", "mode", "prompt", "aspect", "duration", "speed", "seed", "upscale",
                   "ref_images", "start_frame", "end_frame", "ref_video", "is_trial"}
REFINE_FIELDS = {"v", "mode", "source_video_key", "seed", "is_trial"}
JOB_VERSION = 2

comfyui_process = None
comfyui_process_lock = threading.Lock()


class JobRejected(ValueError):
    """A job that fails validation - reported to the user as is."""


def force_kill_comfyui():
    """Kills ComfyUI outright (Force Cancellation, or the session ending mid
    job): a graceful /interrupt only lands at a step boundary, which can be
    minutes away. The session loop restarts and re-warms it if the session
    is still running."""
    global comfyui_process
    with comfyui_process_lock:
        proc = comfyui_process
        comfyui_process = None
    if proc is None:
        print("Force kill requested but no ComfyUI process handle on record.")
        return
    try:
        proc.kill()
        proc.wait(timeout=10)
        print("ComfyUI process force-killed.")
    except Exception as e:
        print(f"Error force-killing ComfyUI process: {e}")


# --- Session timing -------------------------------------------------------
# The worker holds no database key (2026-10-12): everything about the
# session goes through the website's gateway (see "Worker gateway" below).
SESSION_POLL_INTERVAL_SECONDS = 3
# 10 minutes with no job ends the session (user decision 2026-10-10, was 45:
# an idle session holds a GPU someone else could use). Never while a job is
# queued or rendering: the clock starts once the worker is READY (after
# warmup) and restarts when each job finishes. SESSION_IDLE_TIMEOUT_SECONDS on
# the endpoint overrides it without a rebuild.
SESSION_IDLE_TIMEOUT_SECONDS = int(os.environ.get("SESSION_IDLE_TIMEOUT_SECONDS") or 10 * 60)
# Self-return well before RunPod's suspected 24h job ceiling.
SESSION_SAFETY_MAX_SECONDS = 23 * 60 * 60
HEARTBEAT_INTERVAL_SECONDS = 20
PROGRESS_WRITE_INTERVAL_SECONDS = 2
COMFY_JOB_TIMEOUT_SECONDS = int(os.environ.get("COMFY_JOB_TIMEOUT_SECONDS", "3600"))


def utc_now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# --- Worker gateway (2026-10-12) -------------------------------------------
# Every RunPod job's input carries {"gateway": {"url", "pass"}}: a pass for
# THIS session only. The worker does everything through the website's
# /api/worker/* routes (server.js, "Worker gateway"). It holds no database or
# storage keys: the master-key path was removed on 2026-10-12 (owner), so a
# job without a valid pass is refused.
_GW = None
_gw_job = {"id": None, "files": None}
GW_TIMEOUT_SECONDS = 15
GW_PASS_RE = re.compile(r"^[A-Za-z0-9_-]{43}$")
GW_URL_RE = re.compile(r"^(https://[A-Za-z0-9.-]+(:\d+)?|http://(localhost|127\.0\.0\.1)(:\d+)?)$")


class GatewaySessionGone(Exception):
    pass


def gateway_from_input(job_input):
    """The job's gateway settings, checked. Raises ValueError when they're
    missing or malformed: the job is refused (there is no other way to run)."""
    gw = job_input.get("gateway")
    if (not isinstance(gw, dict) or not isinstance(gw.get("url"), str) or not isinstance(gw.get("pass"), str)
            or not GW_URL_RE.match(gw["url"]) or not GW_PASS_RE.match(gw["pass"])):
        raise ValueError("Missing or malformed gateway settings")
    return {"url": gw["url"], "pass": gw["pass"]}


def gw_call(path, body=None, timeout=GW_TIMEOUT_SECONDS):
    r = requests.post(f"{_GW['url']}/api/worker{path}", json=body or {},
                      headers={"Authorization": f"Bearer {_GW['pass']}"}, timeout=timeout)
    if r.status_code == 410:
        raise GatewaySessionGone("The session has ended")
    r.raise_for_status()
    return r.json()


def gw_set_job(job_id):
    _gw_job["id"] = job_id
    _gw_job["files"] = None


def gw_input_files():
    """This job's file links (fetched once per job): {key: {url, size} or
    {missing: true}}, plus whether a face fix's source is allowed."""
    if _gw_job["files"] is None:
        data = gw_call(f"/jobs/{_gw_job['id']}/inputs")
        _gw_job["files"] = data.get("files") or {}
        _gw_job["refine_not_allowed"] = bool(data.get("refineNotAllowed"))
    return _gw_job["files"]


def gw_download(key, path, max_bytes):
    entry = gw_input_files().get(key)
    if not entry or entry.get("missing") or not entry.get("url"):
        raise JobRejected("An uploaded file is missing - upload it again")
    written = 0
    with requests.get(entry["url"], stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                written += len(chunk)
                if written > max_bytes:
                    raise JobRejected("A file is larger than allowed")
                f.write(chunk)


def gw_upload(filepath):
    size = os.path.getsize(filepath)
    link = gw_call(f"/jobs/{_gw_job['id']}/upload", {"size": size})
    with open(filepath, "rb") as f:
        r = requests.put(link["url"], data=f,
                         headers={**(link.get("headers") or {}), "Content-Length": str(size)}, timeout=600)
    r.raise_for_status()
    return link["key"]


# Storage helpers: the only places the worker reads or writes files.
def storage_size(key):
    entry = gw_input_files().get(key)
    if not entry or entry.get("missing"):
        raise JobRejected("An uploaded file is missing - upload it again")
    return int(entry.get("size") or 0)


def storage_download(key, path, max_bytes):
    """key: an input key, or a face fix's bare output key."""
    return gw_download(key, path, max_bytes)


# --- ComfyUI process -------------------------------------------------------

def ensure_comfyui_engine():
    """SageAttention wheel + base model files (marker-gated on the volume)."""
    print("Ensuring ComfyUI engine (SageAttention + model weights) is ready...")
    subprocess.run(["bash", ENGINE_SCRIPT, VOLUME_DIR, COMFYUI_DIR], check=True)


_face_detector_reported = False


def report_face_detector_problem():
    """Tell the website when the engine script left no face detector (it
    failed its fingerprint check, or couldn't be downloaded): Fix faces is off
    on this worker, and the website emails the owner (ops_events kind
    worker_alert, no account data). Once per worker; never stops a session."""
    global _face_detector_reported
    if _face_detector_reported:
        return
    detector = os.path.join(VOLUME_MODELS_DIR, "ultralytics", "bbox", FACE_DETECTOR)
    if os.path.isfile(detector):
        return
    _face_detector_reported = True
    problem = "face_detector_rejected" if os.path.exists(detector + ".mismatch") else "face_detector_missing"
    print(f"Face detector problem: {problem} - Fix faces is off on this worker.")
    try:
        gw_call("/alert", {"problem": problem}, timeout=5)
    except Exception as e:
        print(f"Couldn't report the face detector problem: {e}")


def symlink_models_to_volume():
    for subdir in ("diffusion_models", "text_encoders", "vae", "loras", "ultralytics"):
        link_path = os.path.join(COMFYUI_DIR, "models", subdir)
        target_path = os.path.join(VOLUME_MODELS_DIR, subdir)
        if os.path.islink(link_path):
            continue
        if os.path.isdir(link_path):
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


# ComfyUI decodes user-uploaded pictures and videos with FFmpeg and its own
# libraries. It never needs the database or storage keys, so it doesn't get
# them: a bug in a decoder must not hand over the service-role key.
_COMFYUI_SECRET_ENV = ("SUPABASE_SERVICE_ROLE_KEY", "SUPABASE_URL", "S3_ACCESS_KEY", "S3_SECRET_KEY",
                       "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "RUNPOD_API_KEY",
                       "RUNPOD_AI_API_KEY", "RUNPOD_WEBHOOK_GET_JOB", "RUNPOD_WEBHOOK_POST_OUTPUT",
                       "RUNPOD_WEBHOOK_PING")


def comfyui_env():
    return {k: v for k, v in os.environ.items() if k not in _COMFYUI_SECRET_ENV}


def start_comfyui_if_needed():
    global comfyui_process
    with comfyui_process_lock:
        if is_comfyui_ready():
            return
        if comfyui_process is None or comfyui_process.poll() is not None:
            print("Starting ComfyUI...")
            os.makedirs(COMFYUI_OUTPUT_DIR, exist_ok=True)
            comfyui_process = subprocess.Popen(
                ["python3", "main.py", "--listen", "127.0.0.1", "--port", "8188",
                 "--output-directory", COMFYUI_OUTPUT_DIR],
                cwd=COMFYUI_DIR,
                env=comfyui_env(),
            )
    for _ in range(180):
        if is_comfyui_ready():
            print("ComfyUI is ready.")
            return
        time.sleep(1)
    raise RuntimeError("ComfyUI did not become ready in time.")


def ensure_model_file(spec, subdir):
    """Downloads a HuggingFace file to the volume on first use (.part + rename,
    so an interrupted download is never mistaken for a complete file)."""
    if "repo" not in spec:
        return False
    dest_dir = os.path.join(VOLUME_MODELS_DIR, subdir)
    dest = os.path.join(dest_dir, spec["filename"])
    if os.path.exists(dest):
        return False
    os.makedirs(dest_dir, exist_ok=True)
    url = f"https://huggingface.co/{spec['repo']}/resolve/main/{urllib.parse.quote(spec['repo_path'])}"
    headers = {}
    if os.environ.get("HF_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['HF_TOKEN']}"
    tmp = dest + ".part"
    print(f"Downloading {spec['repo']}/{spec['repo_path']} -> {dest}...")
    start = time.time()
    with requests.get(url, headers=headers, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=16 * 1024 * 1024):
                f.write(chunk)
    os.replace(tmp, dest)
    print(f"Downloaded {spec['filename']} in {round(time.time() - start)}s.")
    return True


def model_files_present(model):
    names = [(MODEL_CHOICES[model], "diffusion_models")]
    if model == "ref2va":
        names += [(LORA_CHOICES["ref2v_turbo"], "loras"), (LORA_CHOICES["ref2v_turbo_8"], "loras")]
    return all(os.path.exists(os.path.join(VOLUME_MODELS_DIR, sub, spec["filename"])) for spec, sub in names)


def ensure_model(model):
    downloaded = ensure_model_file(MODEL_CHOICES[model], "diffusion_models")
    if model == "ref2va":
        downloaded |= ensure_model_file(LORA_CHOICES["ref2v_turbo"], "loras")
        downloaded |= ensure_model_file(LORA_CHOICES["ref2v_turbo_8"], "loras")
    return downloaded


# --- Validation (the worker's own copy of every rule) ---------------------

def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value == value \
        and value not in (float("inf"), float("-inf"))


def _input_key_re(owner_id, exts):
    return re.compile(rf"^inputs/{re.escape(owner_id)}/{UUID_RE}\.({exts})$")


def resolve_seed(raw):
    if raw is None or raw == "":
        return uuid.uuid4().int & SEED_MAX
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise JobRejected("seed must be a whole number")
    if not 0 <= raw <= SEED_MAX:
        raise JobRejected(f"seed must be between 0 and {SEED_MAX}")
    return raw


def validate_job(job_input, owner_id, is_trial=False):
    """Whitelists and normalises a gpu_session_jobs.input. server.js already
    did all of this before inserting; repeated here because the table can be
    written by other paths (a direct insert under RLS before the cutover
    migration, a future bug in server.js). Anything unexpected is refused,
    never silently "fixed" into something the user didn't ask for - except
    numeric ranges, which are clamped."""
    if not isinstance(job_input, dict):
        raise JobRejected("Malformed job")
    if not re.fullmatch(UUID_RE, owner_id or ""):
        raise JobRejected("Session owner unknown")
    if job_input.get("v") != JOB_VERSION:
        raise JobRejected("This job was made for an older version of the site - refresh the page and try again")
    # is_trial is stamped by the job-input trigger from the session row; the
    # worker uses its own read of the session (is_trial argument) and only
    # requires the stamp to agree.
    if job_input.get("is_trial") not in (None, False, True) or bool(job_input.get("is_trial")) != bool(is_trial):
        raise JobRejected("This job doesn't match its session")
    mode = job_input.get("mode")

    if mode == "face_refine":
        if is_trial:
            raise JobRejected("Fix faces is a subscriber feature")
        unknown = set(job_input) - REFINE_FIELDS
        if unknown:
            raise JobRejected(f"Unknown fields: {', '.join(sorted(unknown))}")
        key = job_input.get("source_video_key")
        if not isinstance(key, str) or not OUTPUT_KEY_RE.match(key):
            raise JobRejected("Invalid source video")
        return {"mode": "face_refine", "source_video_key": key, "seed": resolve_seed(job_input.get("seed"))}

    if mode != "generate":
        raise JobRejected("Unknown job mode")
    unknown = set(job_input) - GENERATE_FIELDS
    if unknown:
        raise JobRejected(f"Unknown fields: {', '.join(sorted(unknown))}")

    prompt = job_input.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise JobRejected("A prompt is required")
    prompt = prompt.strip()
    if len(prompt) > PROMPT_MAX_CHARS:
        raise JobRejected(f"The prompt is too long (max {PROMPT_MAX_CHARS} characters)")

    aspect = job_input.get("aspect")
    if aspect not in ASPECT_PRESETS:
        raise JobRejected("Unknown aspect ratio")
    duration = job_input.get("duration")
    if not _is_number(duration):
        raise JobRejected("Duration must be a number")
    duration = round(min(DURATION_MAX, max(DURATION_MIN, float(duration))), 2)
    speed = job_input.get("speed")
    if speed not in ("standard", "fast"):
        raise JobRejected("Unknown speed")
    upscale = job_input.get("upscale")
    if not isinstance(upscale, bool):
        raise JobRejected("upscale must be true or false")

    image_re = _input_key_re(owner_id, "jpg|png|webp")
    video_re = _input_key_re(owner_id, "mp4|mov")
    ref_images = job_input.get("ref_images") or []
    if not isinstance(ref_images, list) or len(ref_images) > MAX_REF_IMAGES:
        raise JobRejected(f"At most {MAX_REF_IMAGES} reference pictures")
    for key in ref_images:
        if not isinstance(key, str) or not image_re.match(key):
            raise JobRejected("Invalid reference picture")
    frames = {}
    for name in ("start_frame", "end_frame"):
        key = job_input.get(name)
        if key is None:
            continue
        if not isinstance(key, str) or not image_re.match(key):
            raise JobRejected(f"Invalid {name.replace('_', ' ')}")
        frames[name] = key
    ref_video = job_input.get("ref_video")
    if ref_video is not None:
        if not isinstance(ref_video, str) or not video_re.match(ref_video):
            raise JobRejected("Invalid reference video")
        if not 1 <= len(ref_images) <= MOTION_MAX_PICTURES:
            raise JobRejected(f"A motion swap needs 1-{MOTION_MAX_PICTURES} pictures")
        if frames:
            raise JobRejected("Start/end frames can't be combined with a motion swap")
        upscale = False  # off for motion swaps for now (user, 2026-10-06)

    if is_trial:
        if ref_video is not None:
            raise JobRejected("Motion swap is a subscriber feature")
        if frames:
            raise JobRejected("Start/end frames are a subscriber feature")
        if len(ref_images) > TRIAL_MAX_PICTURES:
            raise JobRejected(f"The free trial allows up to {TRIAL_MAX_PICTURES} reference pictures")
        upscale = False
        duration = min(duration, TRIAL_MAX_DURATION)

    return {
        "mode": "generate",
        "prompt": prompt,
        "aspect": aspect,
        "duration": duration,
        "speed": speed,
        "seed": resolve_seed(job_input.get("seed")),
        "upscale": upscale,
        "ref_images": list(ref_images),
        "start_frame": frames.get("start_frame"),
        "end_frame": frames.get("end_frame"),
        "ref_video": ref_video,
        "trial": bool(is_trial),
    }


# The free-trial banner (owner, 2026-10-12: "more branded ... a little lower
# in the video not dead centre but still high enough that it cant be cropped
# easily ... also say demo test"): the wordmark as the site draws it
# ("Bizzle" white, ".Studio" gold) after the gold infinity mark, and an
# outlined gold "FREE DEMO" tag like the admin console's "ADMIN" one (owner,
# 2026-10-12: "also have our infinity logo ... instead of demo test, Free
# Demo"), repeated along a dark band with thin gold edges, scrolling across
# the frame.
WATERMARK_BRAND = ("Bizzle", ".Studio")
WATERMARK_TAG = "FREE DEMO"
# The site's infinity loop (web BrandMark.tsx LOOP, viewBox 1.6 6.6 20.8
# 10.8) as absolute cubic segments: (start, control 1, control 2, end).
WATERMARK_LOOP = (
    ((7, 8), (5, 8), (3, 9.8), (3, 12)),
    ((3, 12), (3, 14.2), (4.8, 16), (7, 16)),
    ((7, 16), (9.5, 16), (10.5, 14), (12, 12)),
    ((12, 12), (13.5, 10), (14.5, 8), (17, 8)),
    ((17, 8), (19.2, 8), (21, 9.8), (21, 12)),
    ((21, 12), (21, 14.2), (19.2, 16), (17, 16)),
    ((17, 16), (14.5, 16), (13.5, 14), (12, 12)),
    ((12, 12), (10.5, 10), (9.5, 8), (7, 8)),
)
WATERMARK_TEXT = "".join(WATERMARK_BRAND) + " " + WATERMARK_TAG
WATERMARK_GOLD = (235, 189, 101)  # the site's --bz-gold, #ebbd65


def build_watermark_mark(height):
    """The gold infinity mark, `height` px tall (RGBA), drawn 4x large and
    scaled down so its curves are smooth."""
    from PIL import Image, ImageDraw
    k = 4
    scale = height * k / 10.8
    w = round(20.8 * scale)
    img = Image.new("RGBA", (w, round(10.8 * scale)), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    pts = []
    for p0, c1, c2, p1 in WATERMARK_LOOP:
        for i in range(24):
            t = i / 24
            u = 1 - t
            x = u ** 3 * p0[0] + 3 * u * u * t * c1[0] + 3 * u * t * t * c2[0] + t ** 3 * p1[0]
            y = u ** 3 * p0[1] + 3 * u * u * t * c1[1] + 3 * u * t * t * c2[1] + t ** 3 * p1[1]
            pts.append(((x - 1.6) * scale, (y - 6.6) * scale))
    pts.append(pts[0])
    draw.line(pts, fill=(0, 0, 0, 120), width=round(3.4 * scale), joint="curve")
    draw.line(pts, fill=WATERMARK_GOLD + (235,), width=round(2.4 * scale), joint="curve")
    return img.resize((max(1, round(w / k)), max(1, height)), Image.LANCZOS)
WATERMARK_FONT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "assets", "watermark-font.ttf")
# The banner crosses the frame once every this many seconds (owner,
# 2026-10-12: 4s was "moving too fast"; a 5s trial clip now shows it
# drift about half a frame).
WATERMARK_CROSS_SECONDS = 10.0
# Where the band's middle sits, as a share of the frame's height: below
# centre, but far enough from the bottom that cropping it off would throw
# away a third of the picture.
WATERMARK_CENTRE_Y = 0.64


def build_watermark_strip(w, h):
    """The free-trial banner as one long strip: per repeat, the gold infinity
    mark, "Bizzle" (white) ".Studio" (gold), an outlined gold "FREE DEMO" tag
    and a gold dot, on a dark
    translucent band with gold edges, so it reads on light and dark scenes
    alike. Returns the strip (RGBA, long enough to cover the frame plus one
    repeat), the width of one repeat, and the strip's top edge."""
    from PIL import Image, ImageDraw, ImageFont
    size = max(18, round(min(w, h) * 0.058))
    tag_size = max(12, round(size * 0.52))
    try:
        font = ImageFont.truetype(WATERMARK_FONT_PATH, size)
        tag_font = ImageFont.truetype(WATERMARK_FONT_PATH, tag_size)
        try:
            font.set_variation_by_axes([800])
            tag_font.set_variation_by_axes([800])
        except Exception:
            pass
    except OSError:
        font = ImageFont.load_default(size=size)
        tag_font = ImageFont.load_default(size=tag_size)
    stroke = max(1, round(size / 24))
    brand, studio = WATERMARK_BRAND
    b_l, b_t, b_r, b_b = font.getbbox(brand, stroke_width=stroke)
    s_l, s_t, s_r, s_b = font.getbbox(studio, stroke_width=stroke)
    # The tag's letters, spaced out like the site's small caps badges.
    spacing = round(tag_size * 0.14)
    letters = [(ch, tag_font.getbbox(ch)) for ch in WATERMARK_TAG]
    t_t = min(bb[1] for _, bb in letters)
    t_b = max(bb[3] for _, bb in letters)
    space_w = round(tag_size * 0.32)
    letters_w = sum((space_w if ch == " " else tag_font.getlength(ch)) for ch, _ in letters) + spacing * (len(letters) - 1)
    text_h = max(b_b, s_b) - min(b_t, s_t)
    pad_x, pad_y = round(tag_size * 0.85), round(tag_size * 0.45)
    tag_w, tag_h = round(letters_w) + 2 * pad_x, (t_b - t_t) + 2 * pad_y
    mark = build_watermark_mark(round(text_h * 0.74))
    gap = round(size * 0.55)
    dot_r = max(2, round(size * 0.09))
    tile_w = mark.width + round(gap * 0.45) + (b_r - b_l) + (s_r - s_l) + gap + tag_w + 2 * gap + 2 * dot_r + gap
    band_h = round(text_h * 2.1)
    edge = max(1, round(band_h * 0.04))
    repeats = -(-w // tile_w) + 2
    strip = Image.new("RGBA", (tile_w * repeats, band_h), (8, 8, 10, 92))
    draw = ImageDraw.Draw(strip)
    gold = WATERMARK_GOLD
    draw.rectangle((0, 0, strip.width, edge - 1), fill=gold + (170,))
    draw.rectangle((0, band_h - edge, strip.width, band_h), fill=gold + (170,))
    mid = band_h / 2
    text_y = round(mid - text_h / 2) - min(b_t, s_t)
    border = max(1, round(tag_size * 0.1))
    for i in range(repeats):
        x = i * tile_w + gap // 2
        strip.alpha_composite(mark, dest=(x, round(mid - mark.height / 2)))
        x += mark.width + round(gap * 0.45)
        draw.text((x - b_l, text_y), brand, font=font, fill=(255, 255, 255, 205),
                  stroke_width=stroke, stroke_fill=(0, 0, 0, 130))
        x += b_r - b_l
        draw.text((x - s_l, text_y), studio, font=font, fill=gold + (225,),
                  stroke_width=stroke, stroke_fill=(0, 0, 0, 130))
        x += (s_r - s_l) + gap
        ty = round(mid - tag_h / 2)
        draw.rounded_rectangle((x, ty, x + tag_w, ty + tag_h), radius=tag_h // 2, fill=(14, 11, 6, 170),
                               outline=gold + (230,), width=border)
        lx = x + pad_x
        for ch, bb in letters:
            if ch == " ":
                lx += space_w + spacing
                continue
            draw.text((lx, ty + pad_y - t_t), ch, font=tag_font, fill=gold + (240,))
            lx += tag_font.getlength(ch) + spacing
        x += tag_w + gap
        draw.ellipse((x, mid - dot_r, x + 2 * dot_r, mid + dot_r), fill=gold + (190,))
    top = round(h * WATERMARK_CENTRE_Y - band_h / 2)
    return strip, tile_w, max(0, min(h - band_h, top))


def apply_watermark(raw_bytes):
    """Burns the free-trial banner (infinity mark, "Bizzle.Studio", "FREE DEMO", repeated)
    across every frame a little below the middle (WATERMARK_CENTRE_Y), scrolling left to right without a
    break, so it can't be cropped out or covered by a corner logo (PyAV +
    Pillow; audio copied untouched). Returns the new MP4 bytes. Raises on
    any failure: an unmarked trial video must never be uploaded (fail
    closed)."""
    import av
    from PIL import Image
    work = os.path.join(OUTPUT_DIR, f"wm_{uuid.uuid4().hex}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    src, dst = work + "_in.mp4", work + "_out.mp4"
    try:
        with open(src, "wb") as f:
            f.write(raw_bytes)
        with av.open(src) as inp:
            vin = inp.streams.video[0]
            w, h = vin.codec_context.width, vin.codec_context.height
            strip, tile_w, band_y = build_watermark_strip(w, h)
            fps = vin.average_rate or 24
            speed = w / (float(fps) * WATERMARK_CROSS_SECONDS)  # px per frame
            ain = inp.streams.audio[0] if inp.streams.audio else None
            with av.open(dst, mode="w") as out:
                vout = out.add_stream("libx264", rate=fps)
                vout.width, vout.height, vout.pix_fmt = w, h, "yuv420p"
                vout.options = {"crf": "18", "preset": "medium"}
                aout = out.add_stream_from_template(ain) if ain is not None else None
                frames = 0
                for packet in inp.demux(*([vin] + ([ain] if ain is not None else []))):
                    if packet.stream is ain:
                        if packet.dts is None:
                            continue
                        packet.stream = aout
                        out.mux(packet)
                        continue
                    for frame in packet.decode():
                        img = frame.to_image().convert("RGBA")
                        # Moving right: the window into the strip slides
                        # left by one repeat over and over, seamlessly.
                        offset = int(round(frames * speed)) % tile_w
                        band = strip.crop((tile_w - offset, 0, tile_w - offset + w, strip.height))
                        img.alpha_composite(band, dest=(0, band_y))
                        new = av.VideoFrame.from_image(img.convert("RGB")).reformat(format="yuv420p")
                        new.pts, new.time_base = frame.pts, frame.time_base
                        for p in vout.encode(new):
                            out.mux(p)
                        frames += 1
                for p in vout.encode():
                    out.mux(p)
        if frames == 0:
            raise RuntimeError("no frames")
        with open(dst, "rb") as f:
            return f.read()
    except Exception as e:
        raise RuntimeError(f"The free-trial watermark could not be applied ({type(e).__name__})")
    finally:
        for path in (src, dst):
            try:
                os.remove(path)
            except OSError:
                pass


def job_kind(job):
    if job["mode"] == "face_refine":
        return "refine"
    if job.get("ref_video"):
        return "motion"
    if job.get("ref_images"):
        return "reference"
    return "generate"


def refine_source_allowed(owner_id, key):
    """A face refine may only read a video this user owns and that face
    refine is offered on (not reference-picture or motion-swap videos). The
    website checks it (refineSourceAllowed) and only then links the file;
    the key itself was already format-checked (OUTPUT_KEY_RE)."""
    files = gw_input_files()
    return not _gw_job.get("refine_not_allowed") and key in files and not files[key].get("missing")


# --- Inputs ----------------------------------------------------------------

def _object_size(key):
    try:
        return storage_size(key)
    except JobRejected:
        raise
    except Exception:
        raise JobRejected("An uploaded file is missing - upload it again")


def load_input_image(key, target_size=None):
    """Downloads one uploaded picture, decodes it fully (rejecting anything
    that isn't a real image), and writes a clean PNG into ComfyUI's input dir.
    target_size: (w, h) to centre-crop and resize to - start/end frames must
    match the output canvas exactly, or H3 stretches them."""
    from PIL import Image, ImageOps
    size = _object_size(key)
    if size <= 0 or size > IMAGE_MAX_BYTES:
        raise JobRejected("A picture is too large (max 50MB)")
    os.makedirs(COMFYUI_INPUT_DIR, exist_ok=True)
    tmp = os.path.join(COMFYUI_INPUT_DIR, f"dl_{uuid.uuid4().hex}")
    try:
        storage_download(key, tmp, IMAGE_MAX_BYTES)
        if os.path.getsize(tmp) > IMAGE_MAX_BYTES:
            raise JobRejected("A picture is too large (max 50MB)")
        Image.MAX_IMAGE_PIXELS = IMAGE_MAX_PIXELS
        try:
            # formats=: only these decoders ever look at the file.
            with Image.open(tmp, formats=["JPEG", "PNG", "WEBP"]) as probe:
                if probe.format not in ("JPEG", "PNG", "WEBP"):
                    raise JobRejected("Pictures must be JPEG, PNG or WebP")
                probe.verify()
            with Image.open(tmp, formats=["JPEG", "PNG", "WEBP"]) as img:
                w, h = img.size
                if min(w, h) < 16 or max(w, h) > IMAGE_MAX_SIDE or w * h > IMAGE_MAX_DECODE_PIXELS:
                    raise JobRejected("A picture's size is out of range")
                img = ImageOps.exif_transpose(img).convert("RGB")
                if target_size:
                    img = ImageOps.fit(img, target_size, method=Image.LANCZOS)
                elif max(img.size) > REF_IMAGE_MAX_SIDE:
                    img.thumbnail((REF_IMAGE_MAX_SIDE, REF_IMAGE_MAX_SIDE), Image.LANCZOS)
                filename = f"in_{uuid.uuid4().hex}.png"
                img.save(os.path.join(COMFYUI_INPUT_DIR, filename), "PNG")
                return filename
        except JobRejected:
            raise
        except Exception as e:
            raise JobRejected(f"A picture could not be read ({type(e).__name__})")
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


# The reference video is opened ONLY as MP4/MOV with a known codec. FFmpeg
# otherwise picks a demuxer by sniffing the content (a file starting with
# "#EXTM3U" could be read as a playlist), and every extra demuxer/decoder is
# attack surface next to the worker's keys.
MP4_FIRST_BOXES = {b"ftyp", b"moov", b"wide", b"free", b"skip", b"mdat", b"pnot"}
VIDEO_CODECS = {"h264", "hevc", "prores", "mpeg4", "av1", "vp9", "mjpeg"}
AUDIO_CODECS = {"aac", "mp3", "alac", "opus", "ac3", "eac3", "flac"}


def probe_video(path):
    import av
    with open(path, "rb") as f:
        head = f.read(8)
    if len(head) < 8 or head[4:8] not in MP4_FIRST_BOXES:
        raise JobRejected("The reference video must be an MP4 or MOV file")
    with av.open(path, format="mov") as c:
        if not c.streams.video:
            raise JobRejected("The reference video has no picture")
        if not (c.format.name or "").startswith("mov"):
            raise JobRejected("The reference video must be an MP4 or MOV file")
        v = c.streams.video[0]
        if v.codec_context.name not in VIDEO_CODECS:
            raise JobRejected("The reference video's format isn't supported - export it as H.264 or HEVC")
        for a in c.streams.audio:
            name = a.codec_context.name or ""
            if name not in AUDIO_CODECS and not name.startswith("pcm_"):
                raise JobRejected("The reference video's sound format isn't supported - export it with AAC audio")
            if a.duration and a.time_base and float(a.duration * a.time_base) > REF_VIDEO_MAX_SECONDS + 2:
                raise JobRejected("The reference video's sound is longer than its picture - trim it to 15s or less")
        fps = float(v.average_rate or 0)
        # The picture's own length (a soundtrack can run past the frames).
        if v.frames and fps:
            seconds = v.frames / fps
        elif v.duration and v.time_base:
            seconds = float(v.duration * v.time_base)
        else:
            seconds = float(c.duration / av.time_base) if c.duration else 0.0
        return {
            "width": v.codec_context.width,
            "height": v.codec_context.height,
            "fps": fps,
            "seconds": seconds,
            "frames": int(v.frames or round(seconds * fps)),
            "has_audio": bool(c.streams.audio),
        }


def load_ref_video(key):
    """Downloads the motion swap's reference video and checks it: 24fps (the
    node takes frames 1:1 at 24fps), at most 15s, sane size. Returns
    (input_filename, info)."""
    size = _object_size(key)
    if size <= 0 or size > VIDEO_MAX_BYTES:
        raise JobRejected("The reference video is too large (max 800MB)")
    os.makedirs(COMFYUI_INPUT_DIR, exist_ok=True)
    filename = f"refvideo_{uuid.uuid4().hex}{os.path.splitext(key)[1]}"
    path = os.path.join(COMFYUI_INPUT_DIR, filename)
    try:
        storage_download(key, path, VIDEO_MAX_BYTES)
    except Exception:
        if os.path.exists(path):
            os.remove(path)
        raise
    try:
        info = probe_video(path)
    except JobRejected:
        os.remove(path)
        raise
    except Exception as e:
        os.remove(path)
        raise JobRejected(f"The reference video could not be read ({type(e).__name__})")
    if not 23.5 <= info["fps"] <= 24.5:
        os.remove(path)
        raise JobRejected(f"The reference video is {info['fps']:.2f}fps - export it at 24fps")
    if info["seconds"] > REF_VIDEO_MAX_SECONDS:
        os.remove(path)
        raise JobRejected(f"The reference video is {info['seconds']:.1f}s - trim it to 15s or less")
    if not info["width"] or not info["height"] or max(info["width"], info["height"]) > VIDEO_MAX_SIDE:
        os.remove(path)
        raise JobRejected("The reference video's size is out of range")
    return filename, info


def motion_frames(seconds):
    """17n+5 frame count for a swap of a `seconds`-long reference video, never
    past 362. H3 needs 17n+5 frames; rounding up would ask for up to 16 frames
    the video doesn't have, so it rounds up only when that's at most 2 frames
    past the video (the tested 15s swaps: 361 frames available -> 362) and
    otherwise down. The graph's own snap rounds UP, so the duration handed to
    it is exactly frames/24."""
    available = max(5, int(seconds * 24 + 1e-6))
    down = ((available - 5) // 17) * 17 + 5
    up = down if down == available else down + 17
    frames = up if up - available <= 2 else down
    return max(5, min(MAX_FRAMES, frames))


def motion_size(width, height):
    import math
    aspect = math.log(width / height)
    return min(MOTION_PRESETS, key=lambda wh: abs(math.log(wh[0] / wh[1]) - aspect))


# --- Graph building ----------------------------------------------------------

def plan_generation(job):
    """Model, LoRA and steps for a validated generate job - decided here,
    never by the user."""
    if job.get("ref_video"):
        return "ref2va", ("ref2v_turbo" if job["speed"] == "fast" else "ref2v_turbo_8")
    return "base", ("turbo" if job["speed"] == "fast" else None)


def build_prompt_payload(job, model, lora_key, inputs, width, height, duration):
    """inputs: {"ref_images": [filenames], "start_frame": filename|None,
    "end_frame": filename|None, "ref_video": filename|None,
    "ref_video_has_audio": bool}."""
    with open(WORKFLOW_TEMPLATE_PATH, "r") as f:
        workflow = json.load(f)

    ref_images = inputs.get("ref_images") or []
    ref_video = inputs.get("ref_video")
    reference_node = MODEL_CHOICES[model].get("reference_node", False)

    # Reference pictures and the reference video go through
    # MiniMaxH3ReferenceToVideo (<Picture N>, <Video 1>, <Audio 1>); start/end
    # frames alone go through MiniMaxH3ImageToVideo's first/last frame slots;
    # frames together with references are pinned with MiniMaxH3AddGuide.
    if ref_images or ref_video or reference_node:
        ref_inputs = {}
        for i, filename in enumerate(ref_images):
            workflow[f"_ref_image_load_{i}"] = {"inputs": {"image": filename}, "class_type": "LoadImage"}
            # Autogrow API key "<input id>.<template name>" (pinned commit).
            ref_inputs[f"ref_images.ref_image_{i}"] = [f"_ref_image_load_{i}", 0]
        if ref_video:
            workflow["_ref_video_load"] = {"inputs": {"file": ref_video}, "class_type": "LoadVideo"}
            workflow["_ref_video_parts"] = {"inputs": {"video": ["_ref_video_load", 0]},
                                            "class_type": "GetVideoComponents"}
            ref_inputs["ref_videos.ref_video_0"] = ["_ref_video_parts", 0]
            if inputs.get("ref_video_has_audio"):
                # <Audio 1>, and the finished video keeps the original
                # soundtrack (H3's re-generated audio of a song is worse).
                ref_inputs["ref_video_audios.ref_video_audio_0"] = ["_ref_video_parts", 1]
                workflow[NODE_IDS["create_video"]]["inputs"]["audio"] = ["_ref_video_parts", 1]
        workflow[NODE_IDS["prompt_and_dims"]] = {
            "inputs": {
                "clip": [NODE_IDS["clip_loader"], 0],
                "vae": [NODE_IDS["vae_loader"], 0],
                "prompt": job["prompt"],
                "width": width,
                "height": height,
                "length": ["105:107", 1],
                "ref_image_size": "match",
                "audio_vae": [NODE_IDS["audio_vae_loader"], 0],
                **ref_inputs,
            },
            "class_type": "MiniMaxH3ReferenceToVideo",
        }
        conditioning = [NODE_IDS["prompt_and_dims"], 0]
        for name, frame_idx in (("start_frame", 0), ("end_frame", -1)):
            filename = inputs.get(name)
            if not filename:
                continue
            workflow[f"_{name}_load"] = {"inputs": {"image": filename}, "class_type": "LoadImage"}
            workflow[f"_{name}_guide"] = {
                "inputs": {
                    "positive": conditioning,
                    "latent": [NODE_IDS["prompt_and_dims"], 1],
                    "vae": [NODE_IDS["vae_loader"], 0],
                    "image": [f"_{name}_load", 0],
                    "frame_idx": frame_idx,
                },
                "class_type": "MiniMaxH3AddGuide",
            }
            conditioning = [f"_{name}_guide", 0]
        workflow[NODE_IDS["guider"]]["inputs"]["conditioning"] = conditioning
    else:
        prompt_node = workflow[NODE_IDS["prompt_and_dims"]]["inputs"]
        prompt_node["prompt"] = job["prompt"]
        # Literal size replaces the link to ResolutionSelector ("115"), which
        # then never runs.
        prompt_node["width"] = width
        prompt_node["height"] = height
        if inputs.get("start_frame"):
            workflow["_start_frame_load"] = {"inputs": {"image": inputs["start_frame"]}, "class_type": "LoadImage"}
            prompt_node["first_frame"] = ["_start_frame_load", 0]
        if inputs.get("end_frame"):
            workflow["_end_frame_load"] = {"inputs": {"image": inputs["end_frame"]}, "class_type": "LoadImage"}
            prompt_node["last_frame"] = ["_end_frame_load", 0]

    # The graph snaps the duration to 17n+5 frames (105:107).
    workflow[NODE_IDS["duration_seconds"]]["inputs"]["value"] = duration
    workflow[NODE_IDS["seed"]]["inputs"]["noise_seed"] = job["seed"]
    workflow[NODE_IDS["unet_loader"]]["inputs"]["unet_name"] = MODEL_CHOICES[model]["filename"]

    # UNETLoader -> [turbo LoRA] -> [sigma shift] -> Sage -> guider/scheduler.
    model_src = [NODE_IDS["unet_loader"], 0]
    steps = STANDARD_STEPS
    if lora_key:
        preset = LORA_CHOICES[lora_key]
        workflow["_lora"] = {
            "inputs": {"model": model_src, "lora_name": preset["filename"], "strength_model": preset["multiplier"]},
            "class_type": "LoraLoaderModelOnly",
        }
        model_src = ["_lora", 0]
        if preset.get("shift"):
            workflow["_sigma_shift"] = {
                "inputs": {"model": model_src, "shift_video": preset["shift"][0], "shift_audio": preset["shift"][1]},
                "class_type": "MiniMaxH3SigmaShift",
            }
            model_src = ["_sigma_shift", 0]
        steps = preset["steps"]
    workflow[NODE_IDS["sage_attention"]]["inputs"]["model"] = model_src
    workflow[NODE_IDS["steps"]]["inputs"]["steps"] = steps

    # A fresh SaveVideo prefix per submission: ComfyUI caches node outputs, and
    # an exact repeat would otherwise point /history at a deleted file.
    workflow[NODE_IDS["output"]]["inputs"]["filename_prefix"] = f"video/MiniMax_H3_{uuid.uuid4().hex[:12]}"

    if job.get("upscale"):
        workflow["_nvidia_vsr"] = {
            "inputs": {
                "images": [NODE_IDS["video_decode"], 0],
                "resize_type": "scale by multiplier",
                "resize_type.scale": NVIDIA_VSR_SCALE,
                "quality": NVIDIA_VSR_QUALITY,
            },
            "class_type": "RTXVideoSuperResolution",
        }
        workflow[NODE_IDS["create_video"]]["inputs"]["images"] = ["_nvidia_vsr", 0]
    return workflow, steps


# --- Running a graph ---------------------------------------------------------

STAGE_BY_NODE = {
    NODE_IDS["video_decode"]: "Decoding the video",
    "_nvidia_vsr": "Upscaling 2x",
    NODE_IDS["output"]: "Saving",
    "r_wan": "Redrawing faces",
    "r_vsr": "Upscaling back",
}


class ProgressWatcher:
    """Follows ComfyUI's websocket for one prompt and reports sampler steps
    and stages through on_update(dict). Best-effort: if the websocket can't
    be opened, the job still runs, just without step-level progress."""

    def __init__(self, client_id, on_update, sampler_nodes=(NODE_IDS["sampler"],)):
        self.client_id = client_id
        self.on_update = on_update
        self.sampler_nodes = set(sampler_nodes)
        self.prompt_id = None
        self._stop = threading.Event()
        self._ws = None
        self._thread = None

    def start(self):
        try:
            import websocket  # websocket-client
            self._ws = websocket.create_connection(f"{COMFYUI_WS_URL}?clientId={self.client_id}", timeout=5)
            self._ws.settimeout(1)
            self._timeout_exc = websocket.WebSocketTimeoutException
        except Exception as e:
            print(f"Progress websocket unavailable ({e}) - continuing without step progress.")
            self._ws = None
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            try:
                msg = self._ws.recv()
            except self._timeout_exc:
                continue
            except Exception:
                return
            if not isinstance(msg, str):
                continue  # binary preview frames
            try:
                data = json.loads(msg)
            except ValueError:
                continue
            body = data.get("data") or {}
            if self.prompt_id and body.get("prompt_id") not in (None, self.prompt_id):
                continue
            try:
                if data.get("type") == "progress" and body.get("node") in self.sampler_nodes:
                    self.on_update({"step": int(body.get("value") or 0), "steps": int(body.get("max") or 0)})
                elif data.get("type") == "executing" and body.get("node") in STAGE_BY_NODE:
                    self.on_update({"stage": STAGE_BY_NODE[body["node"]]})
            except Exception as e:
                print(f"Progress callback failed: {e}")

    def stop(self):
        self._stop.set()
        try:
            if self._ws:
                self._ws.close()
        except Exception:
            pass


def submit_and_wait(workflow, should_cancel=None, should_force_kill=None, on_progress=None,
                    sampler_nodes=(NODE_IDS["sampler"],)):
    client_id = str(uuid.uuid4())
    watcher = ProgressWatcher(client_id, on_progress, sampler_nodes) if on_progress else None
    if watcher:
        watcher.start()
    try:
        resp = requests.post(f"{COMFYUI_URL}/prompt", json={"prompt": workflow, "client_id": client_id}, timeout=30)
        if resp.status_code == 400:
            try:
                detail = resp.json()
            except ValueError:
                detail = resp.text
            print(f"ComfyUI rejected the prompt: {json.dumps(detail)[:4000]}")
            raise RuntimeError("The generation could not be set up (ComfyUI rejected the workflow)")
        resp.raise_for_status()
        prompt_id = resp.json()["prompt_id"]
        if watcher:
            watcher.prompt_id = prompt_id

        interrupted = False
        deadline = time.time() + COMFY_JOB_TIMEOUT_SECONDS
        while time.time() < deadline:
            if should_force_kill and should_force_kill():
                force_kill_comfyui()
                return {"force_killed": True}
            if not interrupted and should_cancel and should_cancel():
                try:
                    requests.post(f"{COMFYUI_URL}/interrupt", json={"prompt_id": prompt_id}, timeout=10)
                except requests.exceptions.RequestException as e:
                    print(f"Could not send /interrupt for {prompt_id}: {e}")
                interrupted = True
            r = requests.get(f"{COMFYUI_URL}/history/{prompt_id}", timeout=30)
            r.raise_for_status()
            history = r.json()
            if prompt_id in history:
                entry = history[prompt_id]
                status = entry.get("status") or {}
                if status.get("status_str") == "error":
                    for kind, data in status.get("messages") or []:
                        if kind == "execution_error":
                            print(f"ComfyUI failed in node {data.get('node_id')} ({data.get('node_type')}): "
                                  f"{data.get('exception_type')}: {data.get('exception_message')}")
                            if interrupted:
                                return {"cancelled": True}
                            raise RuntimeError(f"Generation failed ({data.get('exception_type')}: "
                                               f"{str(data.get('exception_message'))[:300]})")
                    # A run stopped by /interrupt ends as an error whose
                    # message is "execution_interrupted", not
                    # "execution_error": it was the user's cancel, not a
                    # failure (2026-10-12: cancels were recorded as
                    # "Generation failed").
                    if interrupted:
                        return {"cancelled": True}
                    raise RuntimeError("Generation failed")
                if interrupted and NODE_IDS["output"] not in entry.get("outputs", {}):
                    return {"cancelled": True}
                return entry
            time.sleep(1 if interrupted else 2)
        try:
            requests.post(f"{COMFYUI_URL}/interrupt", json={"prompt_id": prompt_id}, timeout=10)
        except requests.exceptions.RequestException as e:
            print(f"Could not send /interrupt for timed-out {prompt_id}: {e}")
        raise TimeoutError(f"The generation did not finish within {COMFY_JOB_TIMEOUT_SECONDS // 60} minutes")
    finally:
        if watcher:
            watcher.stop()


def _output_video_info_and_path(history_entry):
    output_node = history_entry["outputs"][NODE_IDS["output"]]
    # SaveVideo's history output is keyed "images" even for video.
    video_info = output_node["images"][0]
    raw_path = os.path.join(COMFYUI_OUTPUT_DIR, video_info.get("subfolder", ""), video_info["filename"])
    return video_info, raw_path


def fetch_output_video(history_entry):
    video_info, raw_path = _output_video_info_and_path(history_entry)
    r = requests.get(
        f"{COMFYUI_URL}/view",
        params={"filename": video_info["filename"], "subfolder": video_info.get("subfolder", ""),
                "type": video_info.get("type", "output")},
        timeout=120,
    )
    r.raise_for_status()
    try:
        os.remove(raw_path)
    except OSError as e:
        print(f"Could not clean up {raw_path}: {e}")
    return r.content


def save_result(job, raw):
    """Uploads a finished video; returns (video key, clean key or None,
    watermarked). Free trial (owner, 2026-10-13): the customer sees the
    watermarked video; the clean file is uploaded first and kept locked by
    the website until they remove the watermark ($1) or subscribe. If the
    clean upload fails the video is still delivered, watermarked, with no
    offer; the watermark itself is never skipped (apply_watermark failing
    fails the job, as before)."""
    watermarked = bool(job.get("trial"))
    clean_key = None
    if watermarked:
        try:
            clean_key = upload_result_and_get_key(raw)
        except Exception as e:
            print(f"Clean copy not uploaded ({type(e).__name__}: {e}); the video stays watermarked with no offer")
        raw = apply_watermark(raw)
    return upload_result_and_get_key(raw), clean_key, watermarked


def upload_result_and_get_key(raw_bytes):
    """Uploads a finished video through the gateway's upload link and returns
    the bare key the website chose (what VALID_VIDEO_KEY / OUTPUT_KEY_RE accept)."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_filename = f"{uuid.uuid4()}.mp4"
    filepath = os.path.join(OUTPUT_DIR, out_filename)
    with open(filepath, "wb") as f:
        f.write(raw_bytes)
    try:
        # The website picks the key and links only this job's upload.
        return gw_upload(filepath)
    finally:
        try:
            os.remove(filepath)
        except OSError:
            pass


def remove_inputs(filenames):
    for name in filenames:
        if not name:
            continue
        try:
            os.remove(os.path.join(COMFYUI_INPUT_DIR, name))
        except OSError:
            pass


def run_warmup(model="base"):
    """1-step 320x320 throwaway generation: absorbs model load and CUDA warmup
    before the user's first real job. Never uploaded."""
    job = {"prompt": "warmup", "seed": 1, "upscale": False}
    workflow, _ = build_prompt_payload(job, model, None, {}, 320, 320, 1.0)
    workflow[NODE_IDS["steps"]]["inputs"]["steps"] = 1
    result = submit_and_wait(workflow)
    _, raw_path = _output_video_info_and_path(result)
    try:
        os.remove(raw_path)
    except OSError:
        pass


def run_generation(job, report, should_cancel=None, should_force_kill=None, loaded_model=None):
    """One validated generate job. report(dict) writes progress. Returns the
    job output dict; model_used is the H3 model now resident."""
    model, lora_key = plan_generation(job)
    staged = []
    try:
        if model != loaded_model:
            if model == "ref2va":
                if not model_files_present("ref2va"):
                    report({"stage": "First-time setup: downloading the motion model (one time only)"})
                    ensure_model("ref2va")
                report({"stage": "Loading the motion model (about 30s)"})
            elif loaded_model is not None:
                report({"stage": "Switching back to the standard model (about 30s)"})
        report({"stage": "Preparing your inputs"})

        inputs = {"ref_images": []}
        if job.get("ref_video"):
            video_file, info = load_ref_video(job["ref_video"])
            staged.append(video_file)
            width, height = motion_size(info["width"], info["height"])
            frames = motion_frames(min(info["seconds"], DURATION_MAX))
            duration = frames / 24.0
            inputs["ref_video"] = video_file
            inputs["ref_video_has_audio"] = info["has_audio"]
        else:
            width, height = ASPECT_PRESETS[job["aspect"]]
            duration = job["duration"]
        for key in job["ref_images"]:
            name = load_input_image(key)
            staged.append(name)
            inputs["ref_images"].append(name)
        for name in ("start_frame", "end_frame"):
            if job.get(name):
                filename = load_input_image(job[name], target_size=(width, height))
                staged.append(filename)
                inputs[name] = filename

        workflow, steps = build_prompt_payload(job, model, lora_key, inputs, width, height, duration)
        report({"stage": "Loading the motion model (about 30s)" if model != loaded_model and model == "ref2va"
                else "Generating", "step": 0, "steps": steps})

        def on_progress(update):
            if "step" in update:
                update = {"stage": "Generating", **update}
            report(update)

        comfy_start = time.time()
        result = submit_and_wait(workflow, should_cancel, should_force_kill, on_progress)
        comfy_seconds = round(time.time() - comfy_start, 1)
        if result.get("force_killed"):
            return {"cancelled": True, "force_killed": True}, None
        if result.get("cancelled"):
            return {"cancelled": True}, model
        report({"stage": "Saving"})
        raw = fetch_output_video(result)
        video_key, clean_key, watermarked = save_result(job, raw)
        output = {
            "videoKey": video_key,
            "storage": "wasabi",
            "kind": job_kind(job),
            "trial": bool(job.get("trial")),
            "watermarked": watermarked,
            "seed": job["seed"],
            "width": width * (2 if job["upscale"] else 1),
            "height": height * (2 if job["upscale"] else 1),
            "base_width": width,
            "base_height": height,
            "duration": round(duration, 2),
            "speed": job["speed"],
            "upscaled": bool(job["upscale"]),
            "comfy_seconds": comfy_seconds,
        }
        if clean_key:
            # Sent beside the result, never inside it (finish_job).
            output["cleanKey"] = clean_key
        return output, model
    finally:
        remove_inputs(staged)


# --- Face refine (button only, Wan engine, official settings) --------------
# Signed off 2026-10-04 (MERGE_NOTES "Official face-refine settings"): Wan 2.2
# low-noise + lightx2v 4 steps; strength 0.3, 0.6 for faces typically under
# 32px; full up to 60px, untouched from 80px; crop 2x the face; detector 0.25;
# 18px floor to pick a person, 12px once tracked; up to 4 people per shot;
# detection gaps up to 8 frames bridged (inside ComfyLabWanFaceRedraw).
REFINE_MAX_SUBJECTS = 4
REFINE_FACE_PX_MIN = 18.0
REFINE_FACE_PX_MIN_TRACKED = 12.0
REFINE_FACE_PX_TINY = 32.0
REFINE_FACE_PX_SMALL = 60.0
REFINE_FACE_PX_LARGE = 80.0
REFINE_CROP_FACTOR = 2.0
REFINE_DETECT_CONFIDENCE = 0.25
FACE_DETECTOR = "face_yolov8m.pt"
WAN_REFINE_DENOISE = 0.3
WAN_REFINE_SMALL_FACE_DENOISE = 0.6
WAN_REFINE_STEPS = 4
WAN_REFINE_SHIFT = 5.0
WAN_REFINE_FILES = {
    "unet": ({"filename": "wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors",
              "repo": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
              "repo_path": "split_files/diffusion_models/wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors"},
             "diffusion_models"),
    "lora": ({"filename": "wan2.2_t2v_lightx2v_4steps_lora_v1.1_low_noise.safetensors",
              "repo": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
              "repo_path": "split_files/loras/wan2.2_t2v_lightx2v_4steps_lora_v1.1_low_noise.safetensors"},
             "loras"),
    "clip": ({"filename": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
              "repo": "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
              "repo_path": "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors"},
             "text_encoders"),
    "vae": ({"filename": "wan_2.1_vae.safetensors",
             "repo": "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
             "repo_path": "split_files/vae/wan_2.1_vae.safetensors"},
            "vae"),
}


def ensure_wan_refine_files(report):
    missing = [(spec, sub) for spec, sub in WAN_REFINE_FILES.values()
               if not os.path.exists(os.path.join(VOLUME_MODELS_DIR, sub, spec["filename"]))]
    if not missing:
        return False
    report({"stage": "First-time setup: downloading the face models (one time only)"})
    for spec, sub in missing:
        ensure_model_file(spec, sub)
    return True


def downscale_video_for_refine(src_path, dst_path, max_w=1344, max_h=768):
    """Re-encodes a (usually 2x-upscaled) video back to H3's native canvas
    before refining, audio copied untouched. Returns the downscale factor."""
    import av
    with av.open(src_path) as inp:
        vin = inp.streams.video[0]
        w, h = vin.codec_context.width, vin.codec_context.height
        # Fit the larger side to 1344 and the smaller to 768 whatever the
        # orientation, so portrait and square videos aren't over-shrunk.
        long_side, short_side = max(w, h), min(w, h)
        scale = min(1.0, max_w / long_side, max_h / short_side)
        if scale >= 1.0:
            shutil.copyfile(src_path, dst_path)
            return 1.0
        tw = max(32, int(round(w * scale / 32)) * 32)
        th = max(32, int(round(h * scale / 32)) * 32)
        ain = inp.streams.audio[0] if inp.streams.audio else None
        with av.open(dst_path, mode="w") as out:
            vout = out.add_stream("libx264", rate=24)
            vout.width, vout.height, vout.pix_fmt = tw, th, "yuv420p"
            vout.options = {"crf": "12", "preset": "medium"}
            aout = out.add_stream_from_template(ain) if ain is not None else None
            streams = [vin] + ([ain] if ain is not None else [])
            for packet in inp.demux(*streams):
                if packet.stream is ain:
                    if packet.dts is None:
                        continue
                    packet.stream = aout
                    out.mux(packet)
                    continue
                for frame in packet.decode():
                    small = frame.reformat(width=tw, height=th, format="yuv420p", interpolation="AREA")
                    for p in vout.encode(small):
                        out.mux(p)
            for p in vout.encode():
                out.mux(p)
    return w / tw


def prepare_refine_source(video_key):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tmp_path = os.path.join(OUTPUT_DIR, f"refine_src_{uuid.uuid4().hex}.mp4")
    try:
        storage_download(video_key, tmp_path, REFINE_SOURCE_MAX_BYTES)
    except Exception:
        raise JobRejected("The source video could not be found")
    os.makedirs(COMFYUI_INPUT_DIR, exist_ok=True)
    filename = f"refine_{uuid.uuid4().hex[:12]}.mp4"
    try:
        factor = downscale_video_for_refine(tmp_path, os.path.join(COMFYUI_INPUT_DIR, filename))
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
    return filename, factor


def refine_select_node(source_filename):
    return {"class_type": "H3FaceSelect", "inputs": {
        "video": source_filename, "detector": FACE_DETECTOR,
        "confidence": REFINE_DETECT_CONFIDENCE,
        "select": "largest_face", "select_index": 0, "confirmed_pick": "",
        "cut_detection": "auto (pyscenedetect)", "cut_threshold": 3.0,
        "skip_first_frames": 0, "frame_load_cap": 0, "select_every_nth": 1,
        "identity_model": "insightface", "identity_threshold": 0.28,
        "X": 0, "Y": 0, "frame_index": 0}}


def stitch_feather():
    return max(8, int(round(24 * REFINE_CROP_FACTOR / 3.0)))


def build_people_count_payload(source_filename):
    return {
        "r_select": refine_select_node(source_filename),
        "r_count": {"class_type": "ComfyLabSmallFaceCount", "inputs": {
            "face_pick": ["r_select", 2], "face_px_large": REFINE_FACE_PX_LARGE,
            "face_px_min": REFINE_FACE_PX_MIN, "max_people": REFINE_MAX_SUBJECTS}},
        "r_count_report": {"class_type": "PreviewAny", "inputs": {"source": ["r_count", 1]}},
    }


def build_refine_payload(source_filename, subjects, seed, upscale_scale):
    wf = {"r_select": refine_select_node(source_filename)}
    source = ["r_select", 0]
    redraw_inputs = {
        "unet_name": WAN_REFINE_FILES["unet"][0]["filename"],
        "lora_name": WAN_REFINE_FILES["lora"][0]["filename"],
        "clip_name": WAN_REFINE_FILES["clip"][0]["filename"],
        "vae_name": WAN_REFINE_FILES["vae"][0]["filename"],
        "prompt": "", "denoise": WAN_REFINE_DENOISE, "steps": WAN_REFINE_STEPS,
        "shift": WAN_REFINE_SHIFT, "seed": seed,
        "face_px_small": REFINE_FACE_PX_SMALL, "face_px_large": REFINE_FACE_PX_LARGE, "smooth_frames": 9,
        "face_px_min": REFINE_FACE_PX_MIN_TRACKED,
        "small_denoise": WAN_REFINE_SMALL_FACE_DENOISE, "small_face_px": REFINE_FACE_PX_TINY,
        "sage_attention": True,
        "face_pick": ["r_select", 2],
    }
    for i in range(subjects):
        p = f"r{i}_"
        wf[p + "pick"] = {"class_type": "ComfyLabFacePickIndex", "inputs": {
            "face_pick": ["r_select", 2], "index": i,
            "skip_large": True, "face_px_large": REFINE_FACE_PX_LARGE,
            "face_px_min": REFINE_FACE_PX_MIN}}
        wf[p + "track"] = {"class_type": "H3FaceTrackCrop", "inputs": {
            "images": source, "face_pick": [p + "pick", 0],
            "detector": FACE_DETECTOR, "confidence": REFINE_DETECT_CONFIDENCE,
            "crop_factor": REFINE_CROP_FACTOR,
            "canvas_width": 768, "canvas_height": 768, "canvas_mode": "auto_capped_768",
            "smooth_window": 21, "size_smooth_window": 51,
            "smooth_method": "gaussian", "size_mode": "per_frame",
            "identity_track": False, "identity_threshold": 0.28,
            "select": "largest_face", "select_index": i, "fallback_detector": "none",
            "fallback_head_frac": 0.5, "identity_model": "insightface",
            "cut_detection": "auto (pyscenedetect)", "cut_threshold": 3.0,
            "absent_shots": "off", "X": 0, "Y": 0, "frame_index": 0}}
        redraw_inputs[f"crops_{i}"] = [p + "track", 0]
        redraw_inputs[f"transform_{i}"] = [p + "track", 1]
    wf["r_wan"] = {"class_type": "ComfyLabWanFaceRedraw", "inputs": redraw_inputs}
    images = source
    for i in range(subjects):
        p = f"r{i}_"
        wf[p + "stitch"] = {"class_type": "H3FaceStitch", "inputs": {
            "base_images": images, "refined_crops": ["r_wan", i], "transform": ["r_wan", 5 + i],
            "paste_region": "face_only", "mask_dilation": 24, "feather": stitch_feather(), "colour_match": 1.0,
            "blend": 1.0, "undetected_frames": "fade_out", "feather_scales_with_crop": False}}
        images = [p + "stitch", 0]
    if upscale_scale:
        wf["r_vsr"] = {"class_type": "RTXVideoSuperResolution", "inputs": {
            "images": images, "resize_type": "scale by multiplier",
            "resize_type.scale": upscale_scale, "quality": NVIDIA_VSR_QUALITY}}
        images = ["r_vsr", 0]
    wf["r_create"] = {"class_type": "CreateVideo", "inputs": {
        "fps": ["r_select", 6], "bit_depth": 8, "images": images, "audio": ["r_select", 1]}}
    wf[NODE_IDS["output"]] = {"class_type": "ComfyLabSaveVideoNVENC", "inputs": {
        "filename_prefix": f"video/FaceRefineWan_{uuid.uuid4().hex[:12]}",
        "video": ["r_create", 0]}}
    return wf


def run_face_refine(job, owner_id, report, should_cancel=None, should_force_kill=None):
    source_key = job["source_video_key"]
    if not refine_source_allowed(owner_id, source_key):
        raise JobRejected("Face fix is only available on your own generated videos (not reference or motion swap videos)")
    downloaded = ensure_wan_refine_files(report)
    report({"stage": "Finding faces"})
    source_filename, factor = prepare_refine_source(source_key)
    upscale_scale = NVIDIA_VSR_SCALE if abs(factor - NVIDIA_VSR_SCALE) < 0.05 else None
    comfy_start = time.time()
    try:
        counted = submit_and_wait(build_people_count_payload(source_filename), should_cancel, should_force_kill)
        if counted.get("force_killed"):
            return {"cancelled": True, "force_killed": True}
        if counted.get("cancelled"):
            return {"cancelled": True}
        text = counted.get("outputs", {}).get("r_count_report", {}).get("text")
        count_report = (text[0] if isinstance(text, list) else text) or ""
        match = re.search(r"small_face_people=(\d+)", count_report)
        if not match:
            raise RuntimeError("Could not count the faces in this video")
        subjects = min(REFINE_MAX_SUBJECTS, int(match.group(1)))
        if subjects == 0:
            return {
                "videoKey": source_key, "storage": "wasabi", "kind": "refine", "mode": "face_refine",
                "no_small_faces": True,
                "message": "No small faces needed fixing - your video was left as it is.",
                "source_video_key": source_key, "subjects": 0,
                "comfy_seconds": round(time.time() - comfy_start, 1),
            }
        report({"stage": f"Fixing faces ({subjects} {'person' if subjects == 1 else 'people'})"})
        result = submit_and_wait(build_refine_payload(source_filename, subjects, job["seed"], upscale_scale),
                                 should_cancel, should_force_kill, on_progress=report, sampler_nodes=("r_wan",))
    finally:
        remove_inputs([source_filename])
    if result.get("force_killed"):
        return {"cancelled": True, "force_killed": True}
    if result.get("cancelled"):
        return {"cancelled": True}
    report({"stage": "Saving"})
    video_key = upload_result_and_get_key(fetch_output_video(result))
    return {
        "videoKey": video_key, "storage": "wasabi", "kind": "refine", "mode": "face_refine",
        "source_video_key": source_key, "subjects": subjects, "seed": job["seed"],
        "upscaled_back": bool(upscale_scale), "first_time_download": downloaded,
        "comfy_seconds": round(time.time() - comfy_start, 1),
    }


# --- Session state (through the gateway) -----------------------------------

# The website records the RunPod job id right after RunPod accepts the job,
# so a worker can start a moment before it's there (gateway_start retries).
RUNPOD_JOB_ID_WAIT_SECONDS = 20

# If the session can't be read for this long in a row (an outage), the
# worker stops: a GPU nobody can bill or stop must not keep running. The
# billing-meter backstop (end as out of credit when the meter stalls past
# the balance) runs on the website, in the gateway's /state.
SESSION_CHECK_FAIL_CLOSED_SECONDS = 3 * 60
_session_check_failing_since = {}


def is_session_active(session_id):
    """The website answers (and runs the meter backstop itself). Fails
    closed: after SESSION_CHECK_FAIL_CLOSED_SECONDS of errors in a row it
    answers False."""
    try:
        active = bool(gw_call("/state").get("active"))
    except GatewaySessionGone:
        return False
    except Exception as e:
        failing_since = _session_check_failing_since.setdefault(session_id, time.time())
        if time.time() - failing_since > SESSION_CHECK_FAIL_CLOSED_SECONDS:
            print(f"Could not check session state for {SESSION_CHECK_FAIL_CLOSED_SECONDS}s - stopping: {e}")
            return False
        print(f"Could not check session state (still active for now): {e}")
        return True
    _session_check_failing_since.pop(session_id, None)
    return active


def touch_session_heartbeat(session_id):
    """active_gpu_sessions.last_activity_at (through the gateway) -
    server.js's reapDeadWorkers treats 5 minutes without it as a dead worker."""
    try:
        gw_call("/heartbeat")
    except Exception as e:
        print(f"Could not write heartbeat for session {session_id}: {e}")


def mark_session_execution_started(session_id):
    """Sets execution_started_at the moment this RunPod job starts executing:
    billing starts here (server.js meterActiveSessions). Only queue time,
    before a worker picked the job up, is free; worker startup, model load
    and warmup are billed, matching RunPod's own execution time."""
    try:
        gw_call("/mark", {"event": "execution_started"})
    except Exception as e:
        print(f"Could not mark execution started for session {session_id}: {e}")


def mark_session_worker_started(session_id):
    """Sets worker_started_at: the UI's "ready" signal (generating is allowed
    from here)."""
    try:
        gw_call("/mark", {"event": "worker_started"})
    except Exception as e:
        print(f"Could not mark worker started for session {session_id}: {e}")


def reset_worker_started(session_id):
    try:
        gw_call("/mark", {"event": "worker_reset"})
    except Exception as e:
        print(f"Could not reset worker_started for session {session_id}: {e}")


def mark_session_ended(session_id, reason):
    """reason: timeout or error (the gateway refuses others, out_of_credit included)."""
    try:
        gw_call("/end", {"reason": reason})
    except GatewaySessionGone:
        pass
    except Exception as e:
        print(f"Could not mark session {session_id} ended: {e}")


def claim_next_queued_job(session_id):
    job = gw_call("/jobs/claim").get("job")
    if job:
        gw_set_job(job["id"])
    return job


_gw_flags = {}


def _job_flag(job_row_id, column):
    cached = _gw_flags.get(job_row_id)
    if cached is None or time.time() - cached[0] > 1.0:
        try:
            cached = (time.time(), gw_call(f"/jobs/{job_row_id}/flags"))
        except Exception as e:
            print(f"Could not check {column} for job {job_row_id}: {e}")
            return False
        _gw_flags.clear()
        _gw_flags[job_row_id] = cached
    return bool(cached[1].get(column))


def make_progress_writer(job_row_id):
    """Writes gpu_session_jobs.progress, merged and throttled. Shape kept
    compatible with the site's progress readers: {stage, step, steps,
    progress, state: {sampling_step, sampling_steps}}."""
    current = {}
    last_write = [0.0]
    lock = threading.Lock()

    def report(update):
        with lock:
            current.update(update)
            if "stage" in update and "step" not in update and update.get("stage") != "Generating":
                current.pop("step", None)
                current.pop("steps", None)
            body = dict(current)
            if body.get("steps"):
                body["progress"] = round(min(1.0, body.get("step", 0) / body["steps"]), 3)
                body["state"] = {"sampling_step": body.get("step", 0), "sampling_steps": body["steps"]}
            now = time.time()
            stage_changed = "stage" in update
            if not stage_changed and now - last_write[0] < PROGRESS_WRITE_INTERVAL_SECONDS:
                return
            last_write[0] = now
        try:
            gw_call(f"/jobs/{job_row_id}/progress", {"progress": body})
        except Exception as e:
            print(f"Could not write progress for job {job_row_id}: {e}")

    return report


def finish_job(job_row_id, output):
    """The website writes it only while the job is still processing: once it
    has closed the job, a late result is dropped. A trial video's clean copy
    (cleanKey) goes beside the result, never in it: the result is readable by
    the customer, the clean copy stays locked until they remove the
    watermark."""
    output = dict(output)
    body = {"output": output}
    clean_key = output.pop("cleanKey", None)
    if clean_key:
        body["cleanKey"] = clean_key
    try:
        if not gw_call(f"/jobs/{job_row_id}/finish", body).get("written"):
            print(f"Job {job_row_id} was already closed by the website - result not written.")
    except Exception as e:
        print(f"Could not write final result for job {job_row_id}: {e}")


def gateway_start(runpod_job_id):
    """The session's owner and trial flag from the gateway, only for the
    RunPod job the website started (it may still be writing the job id:
    retried for up to RUNPOD_JOB_ID_WAIT_SECONDS). None = refused."""
    deadline = time.time() + RUNPOD_JOB_ID_WAIT_SECONDS
    while True:
        try:
            r = requests.post(f"{_GW['url']}/api/worker/start", json={"runpodJobId": runpod_job_id},
                              headers={"Authorization": f"Bearer {_GW['pass']}"}, timeout=GW_TIMEOUT_SECONDS)
        except Exception as e:
            print(f"Gateway start failed: {e}")
            r = None
        if r is not None and r.status_code == 200:
            data = r.json()
            if not isinstance(data.get("userId"), str) or not re.fullmatch(UUID_RE, data["userId"]):
                return None
            return {"user_id": data["userId"], "is_trial": bool(data.get("isTrial"))}
        if r is not None and r.status_code == 403:
            # The owner blocked the machine this worker runs on (the website
            # asked RunPod which one; 2026-10-12, GPU machines).
            try:
                if r.json().get("blocked") is True:
                    return {"blocked": True}
            except Exception:
                pass
            return None
        if r is not None and r.status_code not in (409, 429, 500, 502, 503, 504):
            return None
        if time.time() > deadline:
            return None
        time.sleep(2)


def run_session(session_id, runpod_job_id=None):
    session_start = time.time()
    last_heartbeat = 0.0
    jobs_processed = 0
    comfy_restarts = 0
    print(f"Session {session_id}: held-open loop starting.")

    def ended(reason, **extra):
        summary = {
            "sessionEnded": True,
            "reason": reason,
            "session_id": session_id,
            "jobs_processed": jobs_processed,
            "comfy_restarts": comfy_restarts,
            # Everything from this job's start to "ready" (engine setup,
            # ComfyUI start, model load, warmup). Informational only: the
            # whole execution time is billed.
            "warmup_seconds": warmup_seconds,
            "session_duration_seconds": round(time.time() - session_start, 1),
        }
        summary.update(extra)
        return summary

    warmup_seconds = 0.0
    # The website checks the pass, that the session is running and that
    # this is the RunPod job it started for it.
    owner = gateway_start(runpod_job_id)
    if owner is None:
        print(f"Session {session_id}: the gateway refused this session - refusing.")
        return ended("not_active")
    if owner.get("blocked"):
        # The website already ended the session (refunded in full). Ask
        # RunPod to replace this worker so the next session isn't sent here.
        print(f"Session {session_id}: this machine is blocked by the owner - stopping this worker.")
        return ended("machine_blocked", refresh_worker=True)
    owner_id = owner["user_id"]
    is_trial = bool(owner.get("is_trial"))
    mark_session_execution_started(session_id)

    # Heartbeat through startup too: a first boot on a fresh volume (Sage
    # build, model downloads) can take longer than the reaper's 5-minute
    # heartbeat window.
    startup_heartbeat_stop = threading.Event()

    def startup_heartbeat():
        touch_session_heartbeat(session_id)
        while not startup_heartbeat_stop.wait(HEARTBEAT_INTERVAL_SECONDS):
            touch_session_heartbeat(session_id)

    threading.Thread(target=startup_heartbeat, daemon=True).start()
    try:
        ensure_comfyui_engine()
        report_face_detector_problem()
        symlink_models_to_volume()
        start_comfyui_if_needed()
        run_warmup("base")
    except Exception as e:
        print(f"Session {session_id}: ComfyUI failed to start ({e}) - ending session.")
        warmup_seconds = round(time.time() - session_start, 1)
        mark_session_ended(session_id, "error")
        return ended("worker_error", last_error=str(e))
    finally:
        startup_heartbeat_stop.set()

    warmup_seconds = round(time.time() - session_start, 1)
    if not is_session_active(session_id):
        # Stopped while starting (Stop pressed, no worker found in time...).
        return ended("stopped")
    mark_session_worker_started(session_id)
    loaded_model = "base"
    # The idle clock starts here, after the warmup - a first-time download or
    # a slow model load never eats into the user's 10 idle minutes.
    last_activity = time.time()
    print(f"Session {session_id}: ready after {warmup_seconds}s.")

    while True:
        now = time.time()
        if now - last_heartbeat > HEARTBEAT_INTERVAL_SECONDS:
            touch_session_heartbeat(session_id)
            last_heartbeat = now

        if now - session_start > SESSION_SAFETY_MAX_SECONDS:
            print(f"Session {session_id}: hit the safety cutoff, ending.")
            mark_session_ended(session_id, "timeout")
            return ended("safety_timeout")

        if not is_session_active(session_id):
            print(f"Session {session_id}: no longer active, ending loop.")
            return ended("stopped")

        try:
            job_row = claim_next_queued_job(session_id)
        except Exception as e:
            print(f"Session {session_id}: could not check queue: {e}")
            job_row = None

        if job_row is None:
            if time.time() - last_activity > SESSION_IDLE_TIMEOUT_SECONDS:
                print(f"Session {session_id}: idle past {SESSION_IDLE_TIMEOUT_SECONDS}s, ending.")
                mark_session_ended(session_id, "timeout")
                return ended("timeout")
            time.sleep(SESSION_POLL_INTERVAL_SECONDS)
            continue

        job_id = job_row["id"]
        print(f"Session {session_id}: processing job {job_id}.")
        report = make_progress_writer(job_id)

        if job_row.get("cancel_requested"):
            finish_job(job_id, {"cancelled": True})
            jobs_processed += 1
            last_activity = time.time()
            continue

        heartbeat_stop = threading.Event()

        def keep_heartbeat_alive():
            while not heartbeat_stop.wait(HEARTBEAT_INTERVAL_SECONDS):
                touch_session_heartbeat(session_id)

        heartbeat_thread = threading.Thread(target=keep_heartbeat_alive, daemon=True)
        heartbeat_thread.start()
        try:
            try:
                if job_row.get("user_id") != owner_id or job_row.get("session_id") != session_id:
                    raise JobRejected("This job does not belong to this session")
                job = validate_job(job_row.get("input"), owner_id, is_trial)
                # Stop GPU / out of credit / reaper mid-job: the session is
                # over, so kill the generation now rather than finishing it.
                kwargs = {
                    "should_cancel": lambda: _job_flag(job_id, "cancel_requested"),
                    "should_force_kill": lambda: (_job_flag(job_id, "force_cancel_requested")
                                                  or not is_session_active(session_id)),
                }
                if job["mode"] == "face_refine":
                    result = run_face_refine(job, owner_id, report, **kwargs)
                else:
                    result, new_model = run_generation(job, report, loaded_model=loaded_model, **kwargs)
                    loaded_model = new_model
            except JobRejected as e:
                result = {"error": str(e)}
            except Exception as e:
                print(f"Session {session_id}: job {job_id} failed: {e}")
                result = {"error": str(e)[:500] or "The generation failed"}
        finally:
            heartbeat_stop.set()
            heartbeat_thread.join(timeout=5)

        finish_job(job_id, result)
        jobs_processed += 1

        if not is_session_active(session_id):
            print(f"Session {session_id}: stopped during job {job_id}, ending loop.")
            return ended("stopped")

        # ComfyUI is down after a force-kill, or after an unplanned crash
        # (e.g. out of memory): restart and re-warm, showing "starting" again.
        if result.get("force_killed") or (result.get("error") and not is_comfyui_ready()):
            why = "force-killed" if result.get("force_killed") else "crashed"
            print(f"Session {session_id}: ComfyUI {why} - restarting and re-warming.")
            reset_worker_started(session_id)
            # Heartbeat through the restart too: the website charges GPU time
            # only up to a minute past the last heartbeat.
            restart_heartbeat_stop = threading.Event()

            def restart_heartbeat():
                touch_session_heartbeat(session_id)
                while not restart_heartbeat_stop.wait(HEARTBEAT_INTERVAL_SECONDS):
                    touch_session_heartbeat(session_id)

            threading.Thread(target=restart_heartbeat, daemon=True).start()
            try:
                start_comfyui_if_needed()
                run_warmup("base")
                loaded_model = "base"
                comfy_restarts += 1
                mark_session_worker_started(session_id)
            except Exception as e:
                print(f"Session {session_id}: restart failed ({e}) - ending session.")
                mark_session_ended(session_id, "error")
                return ended("worker_error", last_error=str(e))
            finally:
                restart_heartbeat_stop.set()

        # Restarted after each job FINISHES, never at claim time: a long job
        # must not count as idle time.
        last_activity = time.time()


def handler(job):
    job_input = job.get("input") or {}
    session_id = job_input.get("session_id")
    # Session mode only. Anyone holding the RunPod API key can send any job
    # shape, so a request without a real, active session is refused here,
    # before touching ComfyUI or the GPU.
    if not isinstance(session_id, str) or not re.fullmatch(UUID_RE, session_id):
        return {"error": "This endpoint only accepts GPU sessions started by the website."}
    runpod_job_id = job.get("id")
    global _GW
    try:
        gateway = gateway_from_input(job_input)
    except ValueError:
        return {"error": "This job has no valid gateway pass."}
    _GW = gateway
    gw_set_job(None)
    try:
        return run_session(session_id, runpod_job_id if isinstance(runpod_job_id, str) else None)
    finally:
        _GW = None
        gw_set_job(None)


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
