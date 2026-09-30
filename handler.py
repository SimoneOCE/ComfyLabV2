import runpod
import subprocess
import threading
import time
import requests
import os
import json
import shutil
import uuid
import base64
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

# Where ComfyUI itself writes generated videos - NOT the ComfyLabV2-facing
# OUTPUT_DIR above (that's our own re-encoded copy on its way to Wasabi).
# See start_comfyui_if_needed()'s --output-directory comment for why this
# has to be redirected off the container's own small disk at all.
COMFYUI_OUTPUT_DIR = os.path.join(VOLUME_DIR, "comfyui_output")

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
# captured off the proven RunPod CUDA 13.0 pod), extended with node classes
# verified directly against comfyanonymous/ComfyUI's own source
# (comfy_extras/nodes_minimax_h3.py) rather than guessed - see the
# ComfyLabV2 migration plan for the research trail.
NODE_IDS = {
    "prompt_and_dims": "105:104",  # MiniMaxH3ImageToVideo (or MiniMaxH3ReferenceToVideo when a ref image is given) - prompt/width/height/length/first_frame/last_frame inputs
    "duration_seconds": "105:111", # PrimitiveFloat feeding ComfyMathExpression's 17n+5 frame-count snap
    "seed": "105:15",              # RandomNoise - noise_seed
    "steps": "105:9",              # BasicScheduler - steps
    "unet_loader": "105:6",        # UNETLoader - base diffusion model, LoRA's "model" input source
    "clip_loader": "105:13",       # CLIPLoader
    "vae_loader": "105:11",        # VAELoader (video)
    "sage_attention": "105:120",   # PathchSageAttentionKJ - "model" input rewired to the LoRA node's output when a LoRA is active
    "output": "92",                # SaveVideo - terminal output node
}

# koboldcpp/stable-diffusion.cpp parsed a `<lora:filename:mult>` tag straight
# out of the prompt string, hidden from the user by inserting it server-side.
# ComfyUI has no equivalent - a LoRA is a real graph node (LoraLoaderModelOnly,
# verified in comfyanonymous/ComfyUI's nodes.py), so "toggling" one here means
# conditionally splicing that node into the workflow per job instead, done in
# build_prompt_payload() below. default_sampler is deliberately NOT carried
# over from production's LORA_CHOICES - koboldcpp's sampler names ("Euler",
# "er_sde") aren't verified against ComfyUI's KSamplerSelect option list, and
# guessing that mapping risks a silently-wrong sampler rather than an error.
LORA_CHOICES = {
    # Comfy-Org's OWN turbo LoRAs, published in the same HF repo as our
    # checkpoint - replaced the original production-ported picks after a
    # real run confirmed those weren't actually compatible with this
    # checkpoint at all (see the adaln_proj shape-mismatch investigation:
    # a LoRA trained for adaln_proj.linear.in_features=2688 hit our
    # checkpoint's actual in_features=8 on every single block). Verified
    # before swapping in, not assumed: pulled these files' safetensors
    # headers directly - they're already keyed "diffusion_model.*" (no
    # naming-convention mismatch), they don't touch adaln_proj at all
    # (sidesteps that exact incompatibility), and their embedded metadata
    # states target_format "ComfyUI generic LoRA" and base_model
    # "Comfy-Org/MiniMax-H3 minimax_h3_fl2va_bf16.safetensors" - a
    # deliberate, documented conversion for this exact model family, not
    # a random community LoRA.
    "turbo": {
        "filename": "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
        "url": "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
        "multiplier": 1.0,
        "default_steps": 8,
    },
    "fast": {
        # Filename says "768p" - possibly tuned for that resolution
        # specifically rather than resolution-agnostic; worth confirming
        # at other sizes rather than assuming it's fine everywhere.
        "filename": "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors",
        "url": "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/loras/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors",
        "multiplier": 1.0,
        "default_steps": 4,
    },
}

# Real-ESRGAN's 2x model, verified as a real, resolving GitHub release
# asset (xinntao/Real-ESRGAN v0.2.1) before wiring it in - not guessed at.
# Not the 4x model: ComfyUI's ImageUpscaleWithModel accumulates its whole
# output tensor - every frame of the video at once - on
# comfy.model_management.intermediate_device(), which is system RAM
# (CPU), not VRAM, unless ComfyUI is started with --gpu-only (confirmed
# in comfy/model_management.py). At 4x that's 16x the source video's
# pixel volume held in RAM simultaneously, which is what was OOM-killing
# the container even on a 141GB-RAM machine, and why it was slow (every
# frame tiled at the heavier 4x cost before anything returns). 2x cuts
# that to 4x the source pixel volume - a real, verified reduction, and it
# matches koboldcpp's own proven default (upscaling_resize=2 on the same
# ESRGAN family). NVIDIA's newer upscaler is a planned follow-up once this
# one is proven; swapping it in later only means changing this
# filename/url and re-running ensure_comfyui_engine.sh's download, the
# node mechanism (UpscaleModelLoader + ImageUpscaleWithModel) stays the
# same for any spandrel-supported architecture.
UPSCALE_MODEL = {
    "filename": "RealESRGAN_x2plus.pth",
    "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
}

# FlashVSR (naxci1/ComfyUI-FlashVSR_Stable, MIT) - the second upscale
# option, added to compare against ESRGAN 2x rather than replace it. Not
# a small CNN like ESRGAN - it's a real diffusion model (Wan2.1-1.3B
# based per its paper, 11-18GB of its own VRAM), so it can't just splice
# into the same graph submission as MiniMax H3's own ~40GB the way ESRGAN
# does. See run_generation's "flashvsr" branch: it runs as a genuinely
# separate submission, with an explicit /free {"unload_models": true}
# between them to evict MiniMax H3's weights first - not a repeat of the
# two-submission design mentioned in build_prompt_payload's history below
# (that one was chasing a VRAM theory later proven wrong), this one has
# an actual second model that needs the room.
#
# Picked over the original repo's other community ComfyUI ports (several
# exist, under GPL-3.0/Apache-2.0/MIT) specifically because its CHANGELOG
# says it replaced the original Block-Sparse-Attention with Sparse_Sage
# Attention "for RTX 50 series support" - this worker's actual GPU (see
# the EUR-IS-2 comment above) - using the same SageAttention family
# MiniMax H3 already runs on here, not an unverified guess at Blackwell
# compatibility.
FLASHVSR_MODEL_NAME = "FlashVSR-v1.1"

# NVIDIA RTX Video Super Resolution (Comfy-Org/Nvidia_RTX_Nodes_ComfyUI,
# Apache-2.0 node) - the third upscale option, test-only, NOT recommended
# for production. Its one dependency (nvidia-vfx, the pip wheel the node
# imports as "nvvfx") is PyPI-classified Development Status :: 3 - Alpha,
# and NVIDIA's own SDK license (developer.download.nvidia.com/licenses/
# Maxine_SDK_License_1Apr2021_updated.pdf, section 1.4) says pre-release
# SDKs "are not intended for use in production or business-critical
# systems" - that's their license, not caution added here. Wired in
# anyway for this comparison tool specifically because that restriction
# is about shipping it live, not about testing it: the node itself
# processes one frame at a time internally (verified against its actual
# __init__.py) rather than accumulating a whole-video batch, and is a
# small dedicated SR effect, not a diffusion model - much lighter than
# FlashVSR - but it still gets the same separate-submission + /free
# treatment for consistency, since its real VRAM footprint on this exact
# hardware/driver combination hasn't been measured yet either.
NVIDIA_VSR_QUALITY = "ULTRA"

comfyui_process = None
comfyui_process_lock = threading.Lock()


def force_kill_comfyui():
    """Force Cancellation: kills ComfyUI outright rather than waiting for
    /interrupt's step-boundary stop. Mirrors minimax-h3-worker's
    force_kill_kobold() - the next generation on this worker pays a fresh
    model-load cost since VRAM state is gone, which is why run_session()
    restarts and re-warms right away rather than waiting for the next job
    to discover the process is dead."""
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
    """Points ComfyUI's normal models/{diffusion_models,text_encoders,vae,
    loras} lookup paths at the volume-backed copies ensure_comfyui_engine.sh
    just downloaded, instead of ComfyUI looking in its own (ephemeral)
    models/ dir baked into the image.
    """
    for subdir in ("diffusion_models", "text_encoders", "vae", "loras", "upscale_models"):
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
            os.makedirs(COMFYUI_OUTPUT_DIR, exist_ok=True)
            comfyui_process = subprocess.Popen(
                [
                    "python3", "main.py",
                    "--listen", "0.0.0.0",
                    "--port", "8188",
                    # Verified via comfy/cli_args.py - without this,
                    # ComfyUI's default output dir is base_path/output
                    # INSIDE the container's own small ephemeral disk
                    # (folder_paths.py: output_directory = os.path.join
                    # (base_path, "output")), never cleaned up between
                    # jobs. Every video from every job in a real test
                    # session - warmups, T2V runs, LoRA tests, and the
                    # much larger 4x-upscaled file - piled up there
                    # unnoticed, very likely what actually filled a
                    # small container disk right around the upscale
                    # test. Pointed at the volume instead, which is
                    # sized for this; fetch_output_video() below also
                    # deletes each file here once its bytes are safely
                    # uploaded to Wasabi, so even the volume doesn't
                    # grow unbounded.
                    "--output-directory", COMFYUI_OUTPUT_DIR,
                ],
                cwd=COMFYUI_DIR,
            )

    for _ in range(180):  # up to 3 minutes for the server itself to come up
        if is_comfyui_ready():
            print("ComfyUI is ready.")
            return
        time.sleep(1)
    raise RuntimeError("ComfyUI did not become ready in time.")


def save_input_image(b64_data, prefix):
    """Writes a base64-encoded image to ComfyUI's input/ directory and
    returns the bare filename. Verified against ComfyUI's own LoadImage
    node (nodes.py): it resolves its "image" widget value as a plain
    filename inside folder_paths.get_input_directory() - this is the
    standard way any ComfyUI API workflow feeds in an uploaded image,
    not a ComfyLabV2-specific convention."""
    input_dir = os.path.join(COMFYUI_DIR, "input")
    os.makedirs(input_dir, exist_ok=True)
    filename = f"{prefix}_{uuid.uuid4()}.png"
    with open(os.path.join(input_dir, filename), "wb") as f:
        f.write(base64.b64decode(b64_data))
    return filename


def build_prompt_payload(job_input, include_upscale=False):
    """Builds the full workflow graph for one generation. include_upscale
    splices the ESRGAN nodes into the SAME submission as everything else
    (matching how production's koboldcpp path does upscale - one request,
    not two). An earlier two-submission design (base gen + a separate
    follow-up upscale request) got reverted at the time on the theory that
    splitting into two HTTP requests was itself what caused a real upscale
    test to OOM - that theory turned out to be wrong: the actual cause,
    confirmed directly against comfy/model_management.py, was ESRGAN's
    output tensor landing on intermediate_device() (system RAM) for the
    WHOLE video at once regardless of submission count, fixed by switching
    to 2x (see UPSCALE_MODEL's comment). ESRGAN stays single-submission
    because there's no reason to split it now that the real cause is
    fixed, not because two submissions are inherently unsafe - see
    run_generation's "flashvsr" branch for a case where a second
    submission is the right call for a different, real reason.
    """
    with open(WORKFLOW_TEMPLATE_PATH, "r") as f:
        workflow = json.load(f)

    prompt_text = job_input["prompt"]
    width = job_input.get("width", 1280)
    height = job_input.get("height", 736)
    duration = job_input.get("duration", 7.29)
    steps = job_input.get("steps", 20)
    seed = job_input.get("seed", uuid.uuid4().int & 0xFFFFFFFF)
    ref_image = job_input.get("ref_image")
    start_frame = job_input.get("start_frame")
    end_frame = job_input.get("end_frame")

    # REF FRAME vs START/END FRAME are NOT the same mechanism on ComfyUI,
    # unlike whatever unified handling koboldcpp had - verified against
    # comfy_extras/nodes_minimax_h3.py:
    #   - start_frame/end_frame -> MiniMaxH3ImageToVideo's first_frame/
    #     last_frame optional Image inputs (same node as plain T2V).
    #   - ref_image -> a DIFFERENT node, MiniMaxH3ReferenceToVideo, whose
    #     ref_images go through the prompt via <Picture i> tags rather than
    #     anchoring a specific frame. The two aren't composable without
    #     chaining MiniMaxH3AddGuide on top (not implemented here yet) - a
    #     ref_image request takes the ReferenceToVideo path and ignores any
    #     start/end frame given alongside it in this first pass.
    if ref_image:
        ref_filename = save_input_image(ref_image, "ref")
        workflow["_ref_image_load"] = {
            "inputs": {"image": ref_filename},
            "class_type": "LoadImage",
        }
        workflow[NODE_IDS["prompt_and_dims"]] = {
            "inputs": {
                "clip": [NODE_IDS["clip_loader"], 0],
                "vae": [NODE_IDS["vae_loader"], 0],
                "prompt": prompt_text,
                "width": width,
                "height": height,
                "length": ["105:107", 1],
                "ref_image_size": "match",
                # UNVERIFIED wire format: Autogrow inputs are presented to
                # execute() as a dict (ref_images={...}), built server-side
                # from flat "ref_image_0"/"ref_image_1"/... keys per
                # io.Autogrow.TemplatePrefix(prefix="ref_image_") - this is
                # the standard Autogrow API-JSON shape, but there's no
                # captured real export to confirm it for THIS node the way
                # the plain T2V graph was verified. Test this path first.
                "ref_image_0": ["_ref_image_load", 0],
            },
            "class_type": "MiniMaxH3ReferenceToVideo",
        }
    else:
        prompt_node = workflow[NODE_IDS["prompt_and_dims"]]["inputs"]
        prompt_node["prompt"] = prompt_text
        # Overwrites the link to ResolutionSelector (node "115") with a
        # literal value - ResolutionSelector becomes unreachable from the
        # output node and simply won't execute, which ComfyUI tolerates fine.
        prompt_node["width"] = width
        prompt_node["height"] = height
        if start_frame:
            start_filename = save_input_image(start_frame, "start")
            workflow["_start_frame_load"] = {
                "inputs": {"image": start_filename},
                "class_type": "LoadImage",
            }
            prompt_node["first_frame"] = ["_start_frame_load", 0]
        if end_frame:
            end_filename = save_input_image(end_frame, "end")
            workflow["_end_frame_load"] = {
                "inputs": {"image": end_filename},
                "class_type": "LoadImage",
            }
            prompt_node["last_frame"] = ["_end_frame_load", 0]

    # The graph itself does the 17n+5 frame-count snap via ComfyMathExpression
    # (node "105:107") fed by this raw duration-in-seconds value - no need
    # to replicate that math here, just hand it the seconds.
    workflow[NODE_IDS["duration_seconds"]]["inputs"]["value"] = duration

    workflow[NODE_IDS["steps"]]["inputs"]["steps"] = steps
    workflow[NODE_IDS["seed"]]["inputs"]["noise_seed"] = seed

    # LoRA: no prompt-string tag parsing on ComfyUI (see LORA_CHOICES'
    # comment) - splice a LoraLoaderModelOnly node between the base UNET
    # loader and PathchSageAttentionKJ only when one's requested, leaving
    # today's direct wiring untouched otherwise.
    lora_key = job_input.get("lora")
    if lora_key and lora_key in LORA_CHOICES:
        preset = LORA_CHOICES[lora_key]
        workflow["_lora"] = {
            "inputs": {
                "model": [NODE_IDS["unet_loader"], 0],
                "lora_name": preset["filename"],
                "strength_model": preset["multiplier"],
            },
            "class_type": "LoraLoaderModelOnly",
        }
        workflow[NODE_IDS["sage_attention"]]["inputs"]["model"] = ["_lora", 0]
        if "steps" not in job_input:
            workflow[NODE_IDS["steps"]]["inputs"]["steps"] = preset["default_steps"]

    # ESRGAN upscale - verified nodes (UpscaleModelLoader + ImageUpscale
    # WithModel, comfy_extras/nodes_upscale_model.py). Spliced between
    # VAEDecode's frame output and CreateVideo's input, in the same
    # submission as everything else - ComfyUI's own dynamic VRAM
    # management frees the diffusion model automatically once nothing
    # downstream needs it anymore, same as every other node transition.
    if include_upscale:
        workflow["_upscale_loader"] = {
            "inputs": {"model_name": UPSCALE_MODEL["filename"]},
            "class_type": "UpscaleModelLoader",
        }
        workflow["_upscale_apply"] = {
            "inputs": {
                "upscale_model": ["_upscale_loader", 0],
                "image": ["105:10", 0],
            },
            "class_type": "ImageUpscaleWithModel",
        }
        workflow["105:91"]["inputs"]["images"] = ["_upscale_apply", 0]

    return workflow


def submit_and_wait(workflow, timeout_seconds=1200, should_cancel=None, should_force_kill=None):
    client_id = str(uuid.uuid4())
    resp = requests.post(
        f"{COMFYUI_URL}/prompt",
        json={"prompt": workflow, "client_id": client_id},
        timeout=30,
    )
    resp.raise_for_status()
    prompt_id = resp.json()["prompt_id"]

    # ComfyUI's own native cancel endpoint (POST /interrupt, optionally
    # scoped to a prompt_id) - verified in comfyanonymous/ComfyUI's
    # server.py, nothing custom needed. Force cancel is a harder stop
    # (kill the whole process, see force_kill_comfyui) for a request that
    # doesn't respond to a graceful interrupt.
    interrupted = False
    deadline = time.time() + timeout_seconds
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
            if interrupted and NODE_IDS["output"] not in entry.get("outputs", {}):
                # Stopped before the output node ran - a real cancel, not a
                # generation failure.
                return {"cancelled": True}
            return entry
        time.sleep(1 if interrupted else 2)
    raise TimeoutError(f"ComfyUI generation did not finish within {timeout_seconds}s.")


def _output_video_info_and_path(history_entry):
    """Shared by fetch_output_video and stage_video_for_reload: resolves a
    finished submission's SaveVideo output to (video_info, its real path
    under COMFYUI_OUTPUT_DIR)."""
    outputs = history_entry["outputs"]
    output_node = outputs[NODE_IDS["output"]]
    # Verified against ComfyUI's actual source (comfy_api/latest/_ui.py,
    # PreviewVideo.as_dict()) rather than assumed: SaveVideo's history
    # output list is keyed "images" even though it's video - that's
    # PreviewVideo's own key choice, not a ComfyLabV2 convention.
    video_info = output_node["images"][0]
    raw_path = os.path.join(COMFYUI_OUTPUT_DIR, video_info.get("subfolder", ""), video_info["filename"])
    return video_info, raw_path


def fetch_output_video(history_entry):
    video_info, raw_path = _output_video_info_and_path(history_entry)

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

    # Bytes are safely in hand now (about to be re-uploaded to Wasabi by
    # the caller) - delete ComfyUI's own copy so COMFYUI_OUTPUT_DIR on the
    # volume doesn't grow unbounded across a long session's worth of jobs,
    # same reasoning as redirecting it off the container disk in the first
    # place (see start_comfyui_if_needed).
    try:
        os.remove(raw_path)
    except OSError as e:
        print(f"Could not clean up {raw_path}: {e}")

    return r.content, video_info["filename"]


def stage_video_for_reload(history_entry):
    """FlashVSR path only: moves a just-finished base generation's SaveVideo
    output from COMFYUI_OUTPUT_DIR into ComfyUI's input/ directory - same
    convention save_input_image() already uses for ref/start/end frame
    images - so the second submission's LoadVideo node can read it back in
    by filename. Returns the bare filename LoadVideo should reference."""
    video_info, raw_path = _output_video_info_and_path(history_entry)
    input_dir = os.path.join(COMFYUI_DIR, "input")
    os.makedirs(input_dir, exist_ok=True)
    staged_filename = f"flashvsr_src_{uuid.uuid4()}{os.path.splitext(video_info['filename'])[1]}"
    shutil.move(raw_path, os.path.join(input_dir, staged_filename))
    return staged_filename


def free_comfyui_models():
    """POST /free {"unload_models": true} - ComfyUI's own endpoint
    (server.py's post_free route), which sets a flag main.py's worker loop
    consumes right after its current item finishes, calling
    comfy.model_management.unload_all_models() (moves resident weights
    from VRAM to CPU RAM, not deletes them). Used between the base
    generation and a FlashVSR submission so FlashVSR's own VRAM footprint
    doesn't have to coexist with MiniMax H3's - see FLASHVSR_MODEL_NAME's
    comment for why that's a real constraint here, unlike ESRGAN."""
    requests.post(f"{COMFYUI_URL}/free", json={"unload_models": True}, timeout=30).raise_for_status()


def build_flashvsr_payload(input_filename):
    """Second-submission graph for the FlashVSR path: LoadVideo/GetVideo
    Components (native ComfyUI nodes, comfy_extras/nodes_video.py) reload
    the base generation's staged file as IMAGE frames, FlashVSRNodeInitPipe/
    FlashVSRNodeAdv (naxci1/ComfyUI-FlashVSR_Stable, verified against its
    actual nodes.py) upscale them, CreateVideo/SaveVideo re-encode - same
    output shape as the base graph's own tail, so fetch_output_video works
    on this submission's result unchanged.

    Parameters left at the node's own recommended defaults (tuned by its
    maintainer, not guessed here) except: scale=2, to match ESRGAN's scale
    for a fair side-by-side comparison; enable_debug=True, to surface the
    per-run VRAM/timing summary this test is specifically comparing
    against ESRGAN's; keep_models_on_cpu/force_offload left True (their
    own defaults) since offloading its own weights afterward is exactly
    what's wanted here.
    """
    return {
        "_load_video": {
            "inputs": {"file": input_filename},
            "class_type": "LoadVideo",
        },
        "_video_components": {
            "inputs": {"video": ["_load_video", 0]},
            "class_type": "GetVideoComponents",
        },
        "_flashvsr_pipe": {
            "inputs": {
                "model": FLASHVSR_MODEL_NAME,
                "mode": "tiny",
                "vae_model": "Wan2.1",
                "force_offload": True,
                "precision": "auto",
                "device": "auto",
                "attention_mode": "sparse_sage_attention",
            },
            "class_type": "FlashVSRNodeInitPipe",
        },
        "_flashvsr_upscale": {
            "inputs": {
                "pipe": ["_flashvsr_pipe", 0],
                "frames": ["_video_components", 0],
                "scale": 2,
                "color_fix": True,
                "tiled_vae": True,
                "tiled_dit": True,
                "tile_size": 256,
                "tile_overlap": 24,
                "unload_dit": False,
                "sparse_ratio": 2.0,
                "kv_ratio": 3.0,
                "local_range": 11,
                "seed": 0,
                "frame_chunk_size": 0,
                "enable_debug": True,
                "keep_models_on_cpu": True,
                "resize_factor": 1.0,
            },
            "class_type": "FlashVSRNodeAdv",
        },
        "_flashvsr_create_video": {
            "inputs": {
                "fps": ["_video_components", 2],
                "bit_depth": 8,
                "images": ["_flashvsr_upscale", 0],
                "audio": ["_video_components", 1],
            },
            "class_type": "CreateVideo",
        },
        NODE_IDS["output"]: {
            "inputs": {
                "filename_prefix": "video/FlashVSR",
                "format": "auto",
                "codec": "auto",
                "video": ["_flashvsr_create_video", 0],
            },
            "class_type": "SaveVideo",
        },
    }


def build_nvidia_vsr_payload(input_filename):
    """Second-submission graph for the NVIDIA RTX VSR path - see
    NVIDIA_VSR_QUALITY's comment for why this still gets its own
    submission despite being a lighter-weight node than FlashVSR.
    RTXVideoSuperResolution (Comfy-Org/Nvidia_RTX_Nodes_ComfyUI, verified
    against its actual __init__.py) takes a single IMAGE input and a
    DynamicCombo "resize_type" choosing between scale-by-multiplier and
    target-dimensions.

    UNVERIFIED wire format, same caveat as build_prompt_payload's
    ref_image DynamicCombo-adjacent note: the node's execute() signature
    takes resize_type as ONE merged dict (`resize_type["resize_type"]`,
    `resize_type["scale"]` read off the same object), unlike SaveVideo's
    format/codec split in the base graph's own template - no real API-
    format export of this node exists to confirm this shape against, so
    this is reasoned from source, not captured. Test this path first.
    scale=2.0 matches ESRGAN/FlashVSR's scale for a fair comparison.
    """
    return {
        "_load_video": {
            "inputs": {"file": input_filename},
            "class_type": "LoadVideo",
        },
        "_video_components": {
            "inputs": {"video": ["_load_video", 0]},
            "class_type": "GetVideoComponents",
        },
        "_nvidia_vsr": {
            "inputs": {
                "images": ["_video_components", 0],
                "resize_type": {"resize_type": "scale by multiplier", "scale": 2.0},
                "quality": NVIDIA_VSR_QUALITY,
            },
            "class_type": "RTXVideoSuperResolution",
        },
        "_nvidia_vsr_create_video": {
            "inputs": {
                "fps": ["_video_components", 2],
                "bit_depth": 8,
                "images": ["_nvidia_vsr", 0],
                "audio": ["_video_components", 1],
            },
            "class_type": "CreateVideo",
        },
        NODE_IDS["output"]: {
            "inputs": {
                "filename_prefix": "video/NvidiaVSR",
                "format": "auto",
                "codec": "auto",
                "video": ["_nvidia_vsr_create_video", 0],
            },
            "class_type": "SaveVideo",
        },
    }


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


# Upscale methods that need their own separate submission rather than
# splicing into the base generation's graph (unlike "esrgan2x") - see
# FLASHVSR_MODEL_NAME's and NVIDIA_VSR_QUALITY's comments for why each one
# does. Maps to the workflow-builder function each one uses.
SECOND_STAGE_UPSCALE_BUILDERS = {
    "flashvsr": build_flashvsr_payload,
    "nvidia_vsr": build_nvidia_vsr_payload,
}


def run_second_stage_upscale(base_result, build_payload_fn, should_cancel, should_force_kill):
    """Shared by every entry in SECOND_STAGE_UPSCALE_BUILDERS: stages the
    base generation's saved video into ComfyUI's input dir, evicts MiniMax
    H3's weights from VRAM, submits build_payload_fn's graph against it,
    and cleans up the staged input file. Returns that submission's raw
    result (still needs the same force_killed/cancelled handling as any
    submit_and_wait result, and fetch_output_video/upload - this only owns
    the staging/free/submit/cleanup part, shared regardless of which
    upscaler runs)."""
    staged_filename = stage_video_for_reload(base_result)
    free_comfyui_models()
    workflow = build_payload_fn(staged_filename)
    result = submit_and_wait(workflow, should_cancel=should_cancel, should_force_kill=should_force_kill)

    staged_path = os.path.join(COMFYUI_DIR, "input", staged_filename)
    try:
        os.remove(staged_path)
    except OSError as e:
        print(f"Could not clean up {staged_path}: {e}")

    return result


def run_generation(job_input, should_cancel=None, should_force_kill=None):
    """One full generation. upscale_method selects how the tail is built:
    "none"/"esrgan2x" stay a single submission (see build_prompt_payload's
    docstring); anything in SECOND_STAGE_UPSCALE_BUILDERS ("flashvsr",
    "nvidia_vsr") runs the base generation, then a genuinely separate
    second submission for that upscaler (see run_second_stage_upscale).
    Shared by both the classic one-shot handler() path and run_session()'s
    per-job loop below - identical either way, since ComfyUI itself only
    ever has one thing loaded/running at a time regardless of which path
    queued it. should_cancel/should_force_kill default to None (never
    cancels) for the classic path, which has no per-job cancel flag to
    poll.

    No separate upscale-only timing here (production's koboldcpp path
    doesn't expose that either) - upscale's cost is folded into the same
    total time as everything else, same as any other node in the graph.
    FlashVSR's enable_debug=True prints its own VRAM/timing summary to the
    worker log for this comparison, since that can't ride through
    gpu_session_jobs.output the same simple way.
    """
    upscale_method = job_input.get("upscale_method", "none")
    workflow = build_prompt_payload(job_input, include_upscale=(upscale_method == "esrgan2x"))
    result = submit_and_wait(workflow, should_cancel=should_cancel, should_force_kill=should_force_kill)
    if result.get("force_killed"):
        return {"cancelled": True, "force_killed": True}
    if result.get("cancelled"):
        return {"cancelled": True}

    build_payload_fn = SECOND_STAGE_UPSCALE_BUILDERS.get(upscale_method)
    if build_payload_fn:
        result = run_second_stage_upscale(result, build_payload_fn, should_cancel, should_force_kill)
        if result.get("force_killed"):
            return {"cancelled": True, "force_killed": True}
        if result.get("cancelled"):
            return {"cancelled": True}

    raw_bytes, filename = fetch_output_video(result)
    video_key = upload_result_and_get_key(raw_bytes, filename)
    return {"videoKey": video_key}


# --- Session mode (held-open worker) -------------------------------------
# Mirrors minimax-h3-worker/handler.py's run_session() pattern: one RunPod
# job that never returns until the session ends, which is what keeps this
# worker excluded from RunPod's pool for anyone/anything else's /run call
# the whole time. New generation requests can't reach an already-busy
# worker through RunPod's own routing, so they arrive here by polling
# Supabase instead (see claim_next_queued_job). cancel_requested/
# force_cancel_requested are wired through to run_generation() below, same
# shape as production. Still NOT replicated: step-level progress reporting
# (needs ComfyUI's websocket API, not yet built) and production's
# keep-warm ping (see this file's own comment on that above).


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


def is_job_cancel_requested(job_row_id):
    try:
        rows = sb_get(
            "comfylab_gpu_session_jobs",
            {"id": f"eq.{job_row_id}", "select": "cancel_requested"},
        )
        return bool(rows and rows[0].get("cancel_requested"))
    except Exception as e:
        print(f"Could not check cancel flag for job {job_row_id}: {e}")
        return False


def is_job_force_cancel_requested(job_row_id):
    try:
        rows = sb_get(
            "comfylab_gpu_session_jobs",
            {"id": f"eq.{job_row_id}", "select": "force_cancel_requested"},
        )
        return bool(rows and rows[0].get("force_cancel_requested"))
    except Exception as e:
        print(f"Could not check force cancel flag for job {job_row_id}: {e}")
        return False


def finish_job(job_row_id, output):
    if output.get("cancelled"):
        status = "cancelled"
    elif output.get("error"):
        status = "failed"
    else:
        status = "completed"
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
                result = run_generation(
                    job_row["input"],
                    should_cancel=lambda: is_job_cancel_requested(job_row["id"]),
                    should_force_kill=lambda: is_job_force_cancel_requested(job_row["id"]),
                )
            except Exception as e:
                result = {"error": str(e)}
        finally:
            heartbeat_stop.set()
            heartbeat_thread.join(timeout=5)

        finish_job(job_row["id"], result)
        jobs_processed += 1

        # Two separate ways ComfyUI can be down after a job: we killed it
        # ourselves (force_killed), or it crashed on its own - a real run
        # hit this via what looked like an OOM mid-upscale (ComfyUI's HTTP
        # server itself stopped responding, not a clean in-graph error).
        # Without this check, an unplanned crash left every later job in
        # the session failing the same way forever, since nothing else
        # here ever re-checks whether ComfyUI is still alive between jobs.
        needs_restart = result.get("force_killed") or (result.get("error") and not is_comfyui_ready())
        if needs_restart:
            reason = "force-killed" if result.get("force_killed") else "crashed unexpectedly"
            print(f"Session {session_id}: ComfyUI {reason} - restarting and re-warming.")
            try:
                start_comfyui_if_needed()
                run_generation({
                    "prompt": "warmup",
                    "width": 320,
                    "height": 320,
                    "duration": 1.0,
                    "steps": 1,
                })
            except Exception as e:
                print(f"Session {session_id}: failed to restart after ComfyUI {reason} ({e}) - ending session.")
                mark_session_ended(session_id, "error")
                return session_summary("worker_error", last_error=str(e))

        last_activity = time.time()


def handler(job):
    job_input = job["input"]

    # Session mode ONLY - this one RunPod job IS the held-open worker for
    # the named session, and stays inside run_session() until the session
    # ends. There is deliberately no classic one-shot fallback: a frontend-
    # side check ("don't call /run without a session") is not a real
    # boundary, since anyone holding the RunPod API key can send any job
    # shape directly, bypassing whatever the test page's or website's own
    # JS does or doesn't allow. The only enforcement that actually means
    # anything lives here, in the one place that decides whether GPU work
    # happens at all - a request with no session_id gets rejected outright,
    # before touching ComfyUI or the GPU, rather than silently running a
    # full generation for whoever sent it.
    session_id = job_input.get("session_id")
    if not session_id:
        return {"error": "This endpoint only accepts session-mode jobs (session_id required). Start a GPU session first."}

    return run_session(session_id)


runpod.serverless.start({"handler": handler})
