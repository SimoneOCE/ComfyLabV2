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
