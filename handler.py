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
    "guider": "105:16",            # BasicGuider - "model" input (the attention patch point)
    "scheduler": "105:9",          # BasicScheduler - "model" input (only reads model_sampling)
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

# NVIDIA RTX Video Super Resolution (Comfy-Org/Nvidia_RTX_Nodes_ComfyUI) -
# the one upscale option, chosen after a side-by-side test against ESRGAN 2x
# and FlashVSR (both since removed). Spliced into the SAME submission as the
# base generation - no /free, no second submission. It's a small per-frame
# SR network, not a diffusion model, so it doesn't need MiniMax H3 evicted
# to fit, and keeping it in one graph means the base model stays resident
# in VRAM for the next job in the session (the whole point of a warm
# session). Confirmed on a real run: clean 2x output, and a following
# no-upscale job stayed at native resolution.
NVIDIA_VSR_SCALE = 2.0
NVIDIA_VSR_QUALITY = "ULTRA"
# "nvidia_vsr_hb": same upscale, NVIDIA's HIGHBITRATE_ULTRA mode - their
# upscale for clean sources, which skips the compression-artifact
# suppression the standard modes apply. The official node can't select it,
# so this goes through our own ComfyLabRTXVideoSuperResolution
# (comfyui_engine/custom_nodes/comfylab_rtx_vsr) - same processing loop,
# full mode list.
NVIDIA_VSR_HB_QUALITY = "HIGHBITRATE_ULTRA"
UPSCALE_METHODS = {"none", "nvidia_vsr", "nvidia_vsr_hb"}
# Attention A/B (see apply_attention_mode). ComfyUI's own H3 guide
# (docs.comfy.org/tutorials/video/minimax/minimax-h3, "Quality degradation
# with INT8 attention") says Sage's INT8 attention causes morphing late in a
# clip on H3: its last blocks concentrate the key signal in a few channels
# that per-row INT8 scaling loses. The official templates ship without Sage.
#   "sage"   - PathchSageAttentionKJ, sage_attention=auto (the previous default)
#   "off"    - no patch: ComfyUI's standard attention, exact
#   "sparse" - no Sage; ComfyUI's built-in Model Sparse Attention
#              (BlockSparseAttention) in sol-attn mode, the mode the docs say
#              the base H3 weights use. Node defaults: tau 1.3, dense for the
#              first 20% of steps, text/audio/ref rows kept exact.
ATTENTION_MODES = {"sage", "off", "sparse"}
# Allowed upscale_scale values. 4x holds the whole upscaled clip in system
# RAM at once (the node writes to the decoded frames' device, which is
# ComfyUI's intermediate_device() - system RAM - same mechanism that
# OOM-killed ESRGAN 4x): ~31.6GB of float32 frames for a 1280x736, ~175
# frame clip, before encoding. Fine at low base resolutions; risky at high.
UPSCALE_SCALES = {2.0, 4.0}

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
    for subdir in ("diffusion_models", "text_encoders", "vae", "loras"):
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


def build_prompt_payload(job_input, upscale_method="none"):
    """Builds the full workflow graph for one generation. upscale_method
    "nvidia_vsr" / "nvidia_vsr_hb" splices RTX VSR into the SAME submission
    as everything else (matching how production's koboldcpp path does upscale - one
    request, not two).
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

    apply_attention_mode(workflow, job_input.get("attention", "sage"))

    # Unique SaveVideo prefix per submission. ComfyUI caches node outputs, so
    # an exact repeat of a previous job (same prompt/seed/settings) skipped
    # every node including SaveVideo and pointed /history at the previous
    # job's file - which fetch_output_video had already deleted after
    # uploading it, so /view 404'd (seen on a real run). A changed prefix
    # forces just SaveVideo to re-run and write a fresh file; everything
    # upstream (the actual generation) is still served from the cache.
    workflow[NODE_IDS["output"]]["inputs"]["filename_prefix"] = f"video/MiniMax_H3_{uuid.uuid4().hex[:12]}"

    # NVIDIA RTX VSR - spliced between VAEDecode's frame output and
    # CreateVideo's input. resize_type is a DynamicCombo: API-format prompts carry
    # it as FLAT keys - the selected option's key under "resize_type" and
    # that option's own inputs under "resize_type.<name>" - which ComfyUI
    # nests back into one dict before execute() (verified against
    # comfy_api/latest/_io.py at the pinned commit: DynamicCombo's
    # _expand_schema_for_dynamic matches live_inputs["resize_type"] against
    # the option keys, finalize_prefix joins sub-inputs with ".", and
    # build_nested_inputs rebuilds the dict). Same convention the template
    # already uses for ComfyMathExpression's "values.a".
    if upscale_method == "nvidia_vsr":
        workflow["_nvidia_vsr"] = {
            "inputs": {
                "images": ["105:10", 0],
                "resize_type": "scale by multiplier",
                "resize_type.scale": job_input.get("upscale_scale", NVIDIA_VSR_SCALE),
                "quality": NVIDIA_VSR_QUALITY,
            },
            "class_type": "RTXVideoSuperResolution",
        }
        workflow["105:91"]["inputs"]["images"] = ["_nvidia_vsr", 0]
    elif upscale_method == "nvidia_vsr_hb":
        workflow["_nvidia_vsr"] = {
            "inputs": {
                "images": ["105:10", 0],
                "scale": job_input.get("upscale_scale", NVIDIA_VSR_SCALE),
                "quality": NVIDIA_VSR_HB_QUALITY,
            },
            "class_type": "ComfyLabRTXVideoSuperResolution",
        }
        workflow["105:91"]["inputs"]["images"] = ["_nvidia_vsr", 0]

    return workflow


def apply_attention_mode(workflow, mode):
    """Rewires the model path for the attention A/B (see ATTENTION_MODES).
    Runs after the LoRA splice, so it picks up whichever model (base or
    LoRA'd) currently feeds the Sage node."""
    if mode == "sage":
        return
    sage_id = NODE_IDS["sage_attention"]
    model_src = workflow[sage_id]["inputs"]["model"]
    del workflow[sage_id]
    workflow[NODE_IDS["scheduler"]]["inputs"]["model"] = model_src
    if mode == "off":
        workflow[NODE_IDS["guider"]]["inputs"]["model"] = model_src
    elif mode == "sparse":
        # Only the guider needs the patch; the scheduler just builds sigmas.
        # DynamicCombo "selection" in ComfyUI's flat API format (same
        # convention as RTX VSR's resize_type). verbose=True logs, per
        # attention shape, whether it actually ran sparse or why it stayed
        # dense - the docs warn it silently falls back to dense without the
        # comfy-kitchen sol_attn kernel.
        workflow["_sparse_attention"] = {
            "inputs": {
                "model": model_src,
                "selection": "sol-attn",
                "selection.tau": 1.3,
                "start_percent": 0.2,
                "end_percent": 1.0,
                "dense_blocks": "",
                "min_tokens": 12288,
                "extra_tokens": 256,
                "sink_conditioning": "exact_kv_and_rows",
                "verbose": True,
            },
            "class_type": "BlockSparseAttention",
        }
        workflow[NODE_IDS["guider"]]["inputs"]["model"] = ["_sparse_attention", 0]


# Per-job ceiling on one ComfyUI submission. Was a hardcoded 1200s, which a
# 15s 768p clip with Sage off (standard attention, ~20-30 min sampling +
# decode) blows straight through - seen on a real run (job dcd1a245, failed
# at 1205s). Overridable per endpoint via COMFY_JOB_TIMEOUT_SECONDS. Stop GPU
# and the job's cancel flags still end a job early at any point.
COMFY_JOB_TIMEOUT_SECONDS = int(os.environ.get("COMFY_JOB_TIMEOUT_SECONDS", "3600"))


def submit_and_wait(workflow, timeout_seconds=None, should_cancel=None, should_force_kill=None):
    if timeout_seconds is None:
        timeout_seconds = COMFY_JOB_TIMEOUT_SECONDS
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
    # Stop the orphaned generation before giving up on it - without this,
    # ComfyUI kept sampling the timed-out prompt on the GPU and the next job
    # in the session queued behind it.
    try:
        requests.post(f"{COMFYUI_URL}/interrupt", json={"prompt_id": prompt_id}, timeout=10)
    except requests.exceptions.RequestException as e:
        print(f"Could not send /interrupt for timed-out {prompt_id}: {e}")
    raise TimeoutError(f"ComfyUI generation did not finish within {timeout_seconds}s.")


def _output_video_info_and_path(history_entry):
    """Used by fetch_output_video: resolves a finished submission's
    SaveVideo output to (video_info, its real path
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


# RandomNoise's own noise_seed range (comfy_extras/nodes_custom_sampler.py
# at the pinned commit: min=0, max=0xffffffffffffffff).
SEED_MAX = 0xFFFFFFFFFFFFFFFF


def resolve_seed(raw):
    """None/"" -> a fresh random seed; otherwise must be a whole number in
    RandomNoise's range (an int, or a string of digits). Anything else
    fails the job with a clear error rather than ComfyUI's generic 400."""
    if raw is None or raw == "":
        return uuid.uuid4().int & 0xFFFFFFFF
    if isinstance(raw, bool):
        raise ValueError(f"seed must be a whole number, got {raw!r}")
    if isinstance(raw, int):
        seed = raw
    elif isinstance(raw, float) and raw.is_integer():
        seed = int(raw)
    elif isinstance(raw, str) and raw.strip().isdigit():
        seed = int(raw.strip())
    else:
        raise ValueError(f"seed must be a whole number, got {raw!r}")
    if not 0 <= seed <= SEED_MAX:
        raise ValueError(f"seed must be between 0 and {SEED_MAX}, got {seed}")
    return seed


def run_generation(job_input, should_cancel=None, should_force_kill=None, upload=True):
    """One full generation, always a single ComfyUI submission -
    upscale_method "nvidia_vsr" just adds RTX VSR to that same graph (see
    build_prompt_payload). Shared by both the classic one-shot handler()
    path and run_session()'s per-job loop below - identical either way,
    since ComfyUI itself only ever has one thing loaded/running at a time
    regardless of which path queued it. should_cancel/should_force_kill
    default to None (never cancels) for the classic path, which has no
    per-job cancel flag to poll.
    """
    # Resolved here (not left to build_prompt_payload's fallback) so the
    # seed actually used can be returned with the result - needed to re-run
    # the exact same video, e.g. with vs. without upscale.
    job_input = dict(job_input)
    job_input["seed"] = resolve_seed(job_input.get("seed"))
    upscale_method = job_input.get("upscale_method", "none")
    # Unknown values used to fall through build_prompt_payload's if/elif
    # and silently produce an un-upscaled video - seen on a real run when a
    # newer test page sent "nvidia_vsr_hb" to a worker still on the build
    # before that option existed. Fail the job loudly instead.
    if upscale_method not in UPSCALE_METHODS:
        raise ValueError(
            f"Unknown upscale_method {upscale_method!r} - this worker supports {sorted(UPSCALE_METHODS)}"
        )
    try:
        upscale_scale = float(job_input.get("upscale_scale", NVIDIA_VSR_SCALE))
    except (TypeError, ValueError):
        upscale_scale = None
    if upscale_scale not in UPSCALE_SCALES:
        raise ValueError(
            f"Unsupported upscale_scale {job_input.get('upscale_scale')!r} - this worker supports {sorted(UPSCALE_SCALES)}"
        )
    job_input["upscale_scale"] = upscale_scale
    attention = job_input.get("attention", "sage")
    if attention not in ATTENTION_MODES:
        raise ValueError(f"Unknown attention {attention!r} - this worker supports {sorted(ATTENTION_MODES)}")
    workflow = build_prompt_payload(job_input, upscale_method=upscale_method)
    result = submit_and_wait(workflow, should_cancel=should_cancel, should_force_kill=should_force_kill)
    if result.get("force_killed"):
        return {"cancelled": True, "force_killed": True}
    if result.get("cancelled"):
        return {"cancelled": True}

    if not upload:
        # Warmup runs: the output is a throwaway 1-step 320x320 clip. It used
        # to go through the same upload as a real job, so every session start
        # (and every re-warm after a crash/force-kill) left an orphaned, blurry
        # video in Wasabi with no record of where it came from. Just delete
        # ComfyUI's local copy instead.
        _, raw_path = _output_video_info_and_path(result)
        try:
            os.remove(raw_path)
        except OSError as e:
            print(f"Could not clean up warmup output {raw_path}: {e}")
        return {"seed": job_input["seed"]}

    raw_bytes, filename = fetch_output_video(result)
    video_key = upload_result_and_get_key(raw_bytes, filename)
    return {"videoKey": video_key, "seed": job_input["seed"], "attention": attention}


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
        }, upload=False)
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
                # Stop GPU mid-generation counts as a force cancel. Stop only
                # ends the session (ended_at + claim row deleted) - it never
                # sets this job's own cancel flags - and the loop's own
                # is_session_active() check above only runs BETWEEN jobs, so
                # without this a running generation kept the GPU busy until
                # it finished or hit submit_and_wait's 20-minute timeout
                # (seen on a real run: a huge-resolution job kept going after
                # Stop until it was cancelled by hand in RunPod). There's no
                # reaper on this test endpoint to backstop it the way
                # production's REAP_STOP_GRACE_MS does. Force-kill, not a
                # graceful /interrupt: the session is over either way, and an
                # interrupt only lands at a step boundary, which can be
                # minutes away at high resolution.
                result = run_generation(
                    job_row["input"],
                    should_cancel=lambda: is_job_cancel_requested(job_row["id"]),
                    should_force_kill=lambda: (
                        is_job_force_cancel_requested(job_row["id"])
                        or not is_session_active(session_id)
                    ),
                )
            except Exception as e:
                result = {"error": str(e)}
        finally:
            heartbeat_stop.set()
            heartbeat_thread.join(timeout=5)

        finish_job(job_row["id"], result)
        jobs_processed += 1

        # Session stopped during this job (see should_force_kill above) -
        # end now rather than restarting and re-warming a ComfyUI that was
        # just killed for a session nobody is using anymore.
        if not is_session_active(session_id):
            print(f"Session {session_id}: stopped during job {job_row['id']}, ending loop.")
            return session_summary("stopped")

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
                }, upload=False)
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
