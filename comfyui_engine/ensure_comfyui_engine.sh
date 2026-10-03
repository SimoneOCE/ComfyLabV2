#!/bin/bash
# Installs, at runtime on a worker with a real GPU attached, the one piece of
# this stack that a Dockerfile build genuinely cannot produce correctly:
# SageAttention's compiled CUDA kernel. Mirrors speedlabv2's
# koboldcpp_engine/build_convrot_cuda_engine.sh pattern exactly - marker-file
# gated, runs once per persistent RunPod network volume, every subsequent
# worker boot just reuses the cached install.
#
# Why SageAttention can't be baked into the Dockerfile: its setup.py detects
# the attached GPU's compute capability via torch.cuda.get_device_capability()
# at build time (same shape as koboldcpp's ConvRot build using NVCCFLAGS
# -arch=native) - `docker build` has no GPU attached, so a build-time install
# would either produce no matching kernel or require hardcoding an arch,
# reintroducing the exact GPU-mismatch risk the -arch=native/runtime-install
# pattern exists to avoid.
#
# The CPATH fix below was proven on a live RunPod "ComfyUI - CUDA 13.0" pod
# (2026-09-30 session): its apt-installed CUDA toolkit was deliberately
# minimal (no library dev headers) because PyTorch's own pip wheel bundles
# its own copies of them - so the build fails with "fatal error: cusparse.h:
# No such file or directory" unless CPATH is pointed at those bundled
# headers first.
#
# This session briefly ran on cu128 instead of cu130 (this worker's driver
# didn't support cu130 until the endpoint's CUDA Version floor was raised to
# 13.0) - now reverted back to cu130 to match the endpoint and the proven
# pod. See the Dockerfile's FROM line for that full story. The marker file
# and wheel cache below are deliberately CUDA-version-scoped so that
# reverting doesn't silently install into torch cu130 a wheel that was
# actually compiled against cu128's ABI - a real, not cosmetic, mismatch
# risk that a bare marker file would have hidden completely on any volume
# that already has a cached build from before this change.
set -euo pipefail

SAGEATTENTION_REPO_URL="https://github.com/thu-ml/SageAttention.git"
# Pinned to the exact commit proven to build and import cleanly (2.2.0) on
# the live pod this session - not floating on main, for the same reason
# speedlabv2 pins its koboldcpp source commit.
SAGEATTENTION_COMMIT="d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5"

VOLUME_DIR="${1:?usage: ensure_comfyui_engine.sh <persistent-volume-dir> <comfyui-dir>}"
COMFYUI_DIR="${2:?usage: ensure_comfyui_engine.sh <persistent-volume-dir> <comfyui-dir>}"

# Scopes the marker/wheel cache to the actual torch CUDA build baked into
# THIS image, derived at runtime rather than hardcoded - so switching CUDA
# versions (as already happened once this session, cu128 -> cu130) can never
# silently reuse a wheel compiled against a different torch ABI. A wrong
# match here wouldn't necessarily fail loudly; it could just run with the
# wrong kernel or crash confusingly deep inside a generation. Old
# differently-scoped markers/wheel dirs from a prior CUDA version are simply
# never looked at again - harmless leftover, not cleaned up automatically,
# but never a correctness risk either.
TORCH_CUDA_TAG=$(python3 -c "import torch; print(torch.version.cuda)")
if [ -z "$TORCH_CUDA_TAG" ] || [ "$TORCH_CUDA_TAG" = "None" ]; then
    echo "[comfylab-engine] FATAL: could not determine torch's CUDA version (torch.version.cuda) - cannot safely scope the wheel cache." >&2
    exit 1
fi
echo "[comfylab-engine] Torch CUDA build: $TORCH_CUDA_TAG"

MARKER_FILE="$VOLUME_DIR/.comfylab_engine_ready_cu${TORCH_CUDA_TAG}"
# The actual built wheel is cached here, on the volume - NOT just a marker.
# `pip install` writes into this container's own local site-packages, which
# is ephemeral (gone the moment this worker's container is replaced), so a
# marker file alone would make every worker AFTER the first one skip the
# build (marker already exists) while never actually having SageAttention
# installed. Caching the real wheel here lets every worker boot do a fast
# local install (no compilation) from it instead.
WHEEL_CACHE_DIR="$VOLUME_DIR/sageattention_wheel_cu${TORCH_CUDA_TAG}"
# Deliberately NOT $COMFYUI_DIR/models - COMFYUI_DIR is inside the image's
# own container filesystem (ComfyUI is baked into the Dockerfile), which
# doesn't survive past this worker's lifetime. Model weights (~40GB) go on
# the persistent volume so they're downloaded once total, not once per
# worker boot; handler.py symlinks $COMFYUI_DIR/models/* to these paths
# before starting ComfyUI so it finds them at its normal lookup location.
MODELS_DIR="$VOLUME_DIR/models"

echo "[comfylab-engine] Checking for a visible NVIDIA GPU..."
if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi -L 2>/dev/null | grep -qE '^GPU [0-9]+:'; then
    echo "[comfylab-engine] FATAL: no NVIDIA GPU visible to this container - refusing to build (SageAttention's setup.py detects the GPU's compute capability at build time and needs one present to target it correctly)." >&2
    exit 1
fi
nvidia-smi -L

if [ -f "$MARKER_FILE" ]; then
    echo "[comfylab-engine] Marker found at $MARKER_FILE - wheel already built on this volume, skipping the build step."
else
    echo "[comfylab-engine] First run on this volume - building the SageAttention wheel from source..."

    # PyTorch's pip wheel bundles its own copies of the CUDA library headers
    # (cusparse.h etc.) under dist-packages/nvidia/*/include - this pod's
    # apt toolkit doesn't ship them itself, so nvcc can't find them without
    # this.
    # Searches both site-packages and dist-packages: which one pip installs
    # into depends on how Python itself was installed (upstream python.org
    # builds use site-packages; Debian/Ubuntu's apt-installed python3, which
    # is what the original proven pod used, patches pip to use
    # dist-packages instead) - hardcoding one guessed wrong here once
    # already (this image's python:3.12-slim base uses site-packages, not
    # the pod's dist-packages), so search both instead of assuming.
    #
    # The trailing `; true` is load-bearing under `set -euo pipefail`: find
    # exits non-zero when a starting path doesn't exist (exactly one of the
    # two won't, on any given image), and that non-zero status would
    # otherwise propagate through the pipe to paste and abort the whole
    # script right here - silently, before the -z check below ever runs,
    # which is exactly what happened on the previous attempt (the script
    # died with no FATAL message printed at all). `; true` resets the
    # command substitution's own exit status to 0 regardless, while still
    # capturing whatever real output find did produce.
    NVIDIA_INCLUDE_DIRS=$(find /usr/local/lib/python3.12/site-packages/nvidia /usr/local/lib/python3.12/dist-packages/nvidia -maxdepth 2 -type d -name include 2>/dev/null | paste -sd: -; true)
    if [ -z "$NVIDIA_INCLUDE_DIRS" ]; then
        echo "[comfylab-engine] FATAL: could not find PyTorch's bundled nvidia/*/include dirs under dist-packages - CPATH fix has nothing to point at. Toolkit layout may have changed." >&2
        exit 1
    fi
    export CPATH="${NVIDIA_INCLUDE_DIRS}:${CPATH:-}"
    echo "[comfylab-engine] CPATH set to: $CPATH"

    # nvcc itself (the compiler, not just headers) is baked into the
    # Dockerfile now - CUDA_HOME/PATH are already set via ENV in the image
    # (/usr/local/cuda-13.0). Earlier attempts tried pip-installing nvcc at
    # runtime instead: nvidia-cuda-nvcc (CUDA 13.x line) works, but
    # nvidia-cuda-nvcc-cu12 (the 12.x line) turns out to not ship an nvcc
    # binary at all when its wheel is inspected directly - confirmed, not
    # assumed, back when this was briefly on cu128. NVIDIA's apt packages
    # don't have that inconsistency, and installing the compiler itself
    # needs no GPU, so it belongs in the Dockerfile alongside everything
    # else that's safe to bake in. Just verify it's actually where the
    # image's ENV vars claim before relying on it.
    if ! command -v nvcc >/dev/null 2>&1; then
        echo "[comfylab-engine] FATAL: nvcc not found on PATH (expected \$CUDA_HOME/bin from the image's baked-in cuda-nvcc-13-0) - Dockerfile's CUDA toolkit install may have changed or failed silently." >&2
        exit 1
    fi
    echo "[comfylab-engine] Using nvcc: $(command -v nvcc) ($(nvcc --version | tail -1))"

    mkdir -p "$WHEEL_CACHE_DIR"

    # --no-build-isolation: pip's isolated build sandbox can't see the
    # already-installed torch, which SageAttention's build needs to detect
    # the GPU arch and link against.
    # `pip wheel` (not `pip install`) - builds the .whl into WHEEL_CACHE_DIR
    # without installing it into this container's local site-packages,
    # which is what actually gets cached on the volume for every future
    # worker boot to install from (see WHEEL_CACHE_DIR's comment above).
    pip wheel --no-build-isolation -w "$WHEEL_CACHE_DIR" "git+${SAGEATTENTION_REPO_URL}@${SAGEATTENTION_COMMIT}" \
        2>&1 | tee /tmp/sageattention_build.log

    touch "$MARKER_FILE"
    echo "[comfylab-engine] SageAttention wheel built OK, cached at $WHEEL_CACHE_DIR, marker written to $MARKER_FILE."
fi

# Runs on every boot, not just the first - a fast local install (no
# compilation, no network) from the cached wheel into this specific
# container's own site-packages, since that's ephemeral and doesn't survive
# from whichever worker did the original build.
echo "[comfylab-engine] Installing SageAttention from the cached wheel..."
pip uninstall -y sageattention >/dev/null 2>&1 || true
pip install --no-index --find-links "$WHEEL_CACHE_DIR" sageattention

echo "[comfylab-engine] Verifying the install (from outside any SageAttention source clone, to avoid picking up its own sageattention/ subfolder instead of the installed package)..."
(cd /tmp && python3 -c "import sageattention; print(sageattention)")

echo "[comfylab-engine] Ensuring MiniMax H3 model files are present on the volume..."
mkdir -p "$MODELS_DIR/diffusion_models" "$MODELS_DIR/text_encoders" "$MODELS_DIR/vae" "$MODELS_DIR/loras"

# HF_TOKEN is optional - Comfy-Org/MiniMax-H3 is a public repo (verified:
# HTTP 200 with zero credentials), so this isn't required for these specific
# files to download. It's here as a defensive measure in case anonymous
# downloads ever get rate-limited/throttled differently than authenticated
# ones - set the HF_TOKEN env var on the endpoint if that turns out to
# matter; wget just won't send the header at all if it's unset.
HF_AUTH_HEADER=()
if [ -n "${HF_TOKEN:-}" ]; then
    HF_AUTH_HEADER=(--header="Authorization: Bearer ${HF_TOKEN}")
fi

download_if_missing() {
    local url="$1"
    local dest_dir="$2"
    # Optional 3rd arg: save under this filename instead of the URL's own
    # basename - needed for the LoRA files below, whose HF repo filenames
    # don't match LORA_CHOICES' friendlier names in handler.py.
    local filename="${3:-$(basename "$url")}"
    if [ -f "$dest_dir/$filename" ]; then
        echo "[comfylab-engine]   $filename already present, skipping."
    else
        echo "[comfylab-engine]   downloading $filename..."
        # Downloads to a .part file first, only renamed to the real
        # filename on success - if wget dies partway (rate limit, network
        # blip, worker preemption, anything), the truncated data stays
        # under the .part name, so the `-f` check above won't mistake it
        # for a complete file on the next run and silently skip
        # re-downloading a corrupt model.
        wget -q --show-progress "${HF_AUTH_HEADER[@]}" -O "$dest_dir/$filename.part" "$url"
        mv "$dest_dir/$filename.part" "$dest_dir/$filename"
    fi
}

download_if_missing "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors" "$MODELS_DIR/diffusion_models"
download_if_missing "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors" "$MODELS_DIR/text_encoders"
download_if_missing "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_video_vae_fp16.safetensors" "$MODELS_DIR/vae"
download_if_missing "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_audio_vae_fp32.safetensors" "$MODELS_DIR/vae"

# Turbo LoRAs - Comfy-Org's own conversions, published alongside the
# checkpoint itself (see handler.py's LORA_CHOICES comment for why the
# original production-ported picks got swapped out: a real run showed
# they weren't actually compatible with this checkpoint's adaln_proj
# layers at all). Filenames here match LORA_CHOICES exactly, since these
# are already named sensibly upstream - no renaming needed this time.
download_if_missing "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors" "$MODELS_DIR/loras"
download_if_missing "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/loras/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors" "$MODELS_DIR/loras"

# Face detector for the refine pass (ComfyUI-H3-FaceRefine's H3FaceTrackCrop).
# It looks in models/ultralytics/bbox, symlinked to the volume by handler.py.
mkdir -p "$MODELS_DIR/ultralytics/bbox"
download_if_missing "https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8m.pt" "$MODELS_DIR/ultralytics/bbox"

# NVIDIA RTX Video Super Resolution (Comfy-Org/Nvidia_RTX_Nodes_ComfyUI) -
# the one upscale option (see handler.py's NVIDIA_VSR comment). No separate
# model-weight download needed - its VFX SDK libraries ship bundled inside
# the nvidia-vfx wheel itself.
#
# Both pinned now. COMFYUI_DIR is inside the container (not the volume), so
# this clone + pip install re-runs on every fresh worker - unpinned, that
# silently picked up whatever was newest at boot time (nvidia-vfx 0.2.0.0
# shipped 2026-09-30 16:36 UTC, ~30 min before RTX VSR was first wired in
# here). NVIDIA_VFX_VERSION is overridable from the RunPod endpoint's env
# vars so versions can be A/B'd without a rebuild; the installed version is
# printed below so every worker log shows which one actually ran.
NVIDIA_VSR_NODE_COMMIT="892515e3eb9a4920a131a502a047e47adca9eb0d"
NVIDIA_VFX_VERSION="${NVIDIA_VFX_VERSION:-0.2.0.0}"
NVIDIA_VSR_NODE_DIR="$COMFYUI_DIR/custom_nodes/Nvidia_RTX_Nodes_ComfyUI"
if [ -d "$NVIDIA_VSR_NODE_DIR" ]; then
    echo "[comfylab-engine] NVIDIA RTX VSR custom node already present, skipping clone."
else
    echo "[comfylab-engine] Cloning NVIDIA RTX VSR custom node..."
    git clone https://github.com/Comfy-Org/Nvidia_RTX_Nodes_ComfyUI.git "$NVIDIA_VSR_NODE_DIR"
    git -C "$NVIDIA_VSR_NODE_DIR" checkout "$NVIDIA_VSR_NODE_COMMIT"
fi
pip install "nvidia-vfx==$NVIDIA_VFX_VERSION"

# Our own copy of the RTX VSR node with NVIDIA's HIGHBITRATE modes exposed
# (see comfyui_engine/custom_nodes/comfylab_rtx_vsr/__init__.py). Shipped in
# this repo, so it's copied fresh from the image on every worker boot -
# overwritten, never "already present", so an updated image always wins.
COMFYLAB_NODES_SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/custom_nodes/comfylab_rtx_vsr"
rm -rf "$COMFYUI_DIR/custom_nodes/comfylab_rtx_vsr"
cp -r "$COMFYLAB_NODES_SRC" "$COMFYUI_DIR/custom_nodes/comfylab_rtx_vsr"
echo "[comfylab-engine] Installed comfylab_rtx_vsr custom node."

# Logging-only debug nodes for the face refine graph (see their docstring).
COMFYLAB_DEBUG_SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/custom_nodes/comfylab_debug"
rm -rf "$COMFYUI_DIR/custom_nodes/comfylab_debug"
cp -r "$COMFYLAB_DEBUG_SRC" "$COMFYUI_DIR/custom_nodes/comfylab_debug"
echo "[comfylab-engine] Installed comfylab_debug custom node."

# "Refine faces (Wan)" redraw node (see its docstring). Code only - the Wan
# model files are downloaded by the handler on the first Wan refine.
COMFYLAB_FACE_WAN_SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/custom_nodes/comfylab_face_wan"
rm -rf "$COMFYUI_DIR/custom_nodes/comfylab_face_wan"
cp -r "$COMFYLAB_FACE_WAN_SRC" "$COMFYUI_DIR/custom_nodes/comfylab_face_wan"
echo "[comfylab-engine] Installed comfylab_face_wan custom node."

echo "[comfylab-engine] nvidia-vfx installed: $(pip show nvidia-vfx 2>/dev/null | grep '^Version:' || echo 'NOT INSTALLED')"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true

echo "[comfylab-engine] Done. Engine + models ready on volume."
