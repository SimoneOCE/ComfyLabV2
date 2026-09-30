import runpod
import subprocess
import threading
import time
import requests
import os
import json
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

S3_ACCESS_KEY = os.environ.get("RUNPOD_S3_ACCESS_KEY")
S3_SECRET_KEY = os.environ.get("RUNPOD_S3_SECRET_KEY")
S3_ENDPOINT = os.environ.get("RUNPOD_S3_ENDPOINT")
S3_VOLUME_ID = os.environ.get("RUNPOD_VOLUME_ID")

s3_client = boto3.client(
    "s3",
    endpoint_url=S3_ENDPOINT,
    aws_access_key_id=S3_ACCESS_KEY,
    aws_secret_access_key=S3_SECRET_KEY,
    config=Config(signature_version="s3v4"),
)

# TODO: these node IDs are placeholders. Once the real API-format workflow
# export lands (see workflow_template.json's _comment), open it and fill in
# the actual numeric string IDs ComfyUI assigned to each of these nodes -
# they're stable per-workflow-file but arbitrary, not something to guess.
NODE_IDS = {
    "positive_prompt": "TODO",  # CLIPTextEncode (positive) node - "text" input
    "width_height": "TODO",     # the node holding the video's width/height widgets
    "frame_count": "TODO",      # the node holding num_frames (or length) for duration
    "seed": "TODO",             # KSampler/BasicScheduler-adjacent seed source
    "steps": "TODO",            # BasicScheduler - "steps" widget
    "output": "TODO",           # SaveVideo / VHS_VideoCombine - the terminal output node
    "first_frame": "TODO",      # image input node - I2V start frame (optional per job)
    "last_frame": "TODO",       # image input node - I2V end frame (optional per job)
}

comfyui_process = None
comfyui_process_lock = threading.Lock()


def duration_to_frame_count(duration_seconds):
    """MiniMax H3 requires frame counts of the form 17n+5, not an arbitrary
    integer. 24fps target, snapped to the nearest valid count - e.g.
    duration=7.29 -> round(7.29*24)=175 -> already 17*10+5, matches the
    stress-test baseline this was validated against.
    """
    raw = max(5, round(duration_seconds * 24))
    n = round((raw - 5) / 17)
    return 17 * n + 5


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
    downloaded, instead of ComfyUI looking in its own (empty, ephemeral)
    models/ dir baked into the image.
    """
    for subdir in ("diffusion_models", "text_encoders", "vae"):
        link_path = os.path.join(COMFYUI_DIR, "models", subdir)
        target_path = os.path.join(VOLUME_MODELS_DIR, subdir)
        if os.path.islink(link_path):
            continue
        if os.path.isdir(link_path):
            os.rmdir(link_path)  # the empty dir ComfyUI ships with by default
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

    workflow[NODE_IDS["positive_prompt"]]["inputs"]["text"] = prompt_text
    workflow[NODE_IDS["width_height"]]["inputs"]["width"] = width
    workflow[NODE_IDS["width_height"]]["inputs"]["height"] = height
    workflow[NODE_IDS["frame_count"]]["inputs"]["length"] = duration_to_frame_count(duration)
    workflow[NODE_IDS["steps"]]["inputs"]["steps"] = steps
    workflow[NODE_IDS["seed"]]["inputs"]["seed"] = seed

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
    # ComfyUI's video-output nodes (SaveVideo/VHS_VideoCombine) list produced
    # files under a "gifs" or "videos" key depending on node type - checked
    # at runtime against the real exported workflow rather than assumed here.
    video_info = (output_node.get("videos") or output_node.get("gifs"))[0]

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
    s3_client.upload_file(filepath, S3_VOLUME_ID, key, ExtraArgs={"ContentType": content_type})
    return out_filename


def handler(job):
    job_input = job["input"]

    ensure_comfyui_engine()
    symlink_models_to_volume()
    start_comfyui_if_needed()

    workflow = build_prompt_payload(job_input)
    history_entry = submit_and_wait(workflow)
    raw_bytes, filename = fetch_output_video(history_entry)
    video_key = upload_result_and_get_key(raw_bytes, filename)

    return {"videoKey": video_key}


runpod.serverless.start({"handler": handler})
