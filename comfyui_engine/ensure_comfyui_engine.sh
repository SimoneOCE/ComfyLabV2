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
# Proven on a live RunPod "ComfyUI - CUDA 13.0" pod (2026-09-30 session):
# the pod's apt-installed CUDA toolkit is deliberately minimal
# (cuda-minimal-build-13-0, no library dev headers) because PyTorch's own pip
# wheel bundles its own copies under dist-packages/nvidia/*/include - so the
# build fails with "fatal error: cusparse.h: No such file or directory"
# unless CPATH is pointed at those bundled headers first.
set -euo pipefail

SAGEATTENTION_REPO_URL="https://github.com/thu-ml/SageAttention.git"
# Pinned to the exact commit proven to build and import cleanly (2.2.0) on
# the live pod this session - not floating on main, for the same reason
# speedlabv2 pins its koboldcpp source commit.
SAGEATTENTION_COMMIT="d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5"

VOLUME_DIR="${1:?usage: ensure_comfyui_engine.sh <persistent-volume-dir> <comfyui-dir>}"
COMFYUI_DIR="${2:?usage: ensure_comfyui_engine.sh <persistent-volume-dir> <comfyui-dir>}"

MARKER_FILE="$VOLUME_DIR/.comfylab_engine_ready"
# The actual built wheel is cached here, on the volume - NOT just a marker.
# `pip install` writes into this container's own local site-packages, which
# is ephemeral (gone the moment this worker's container is replaced), so a
# marker file alone would make every worker AFTER the first one skip the
# build (marker already exists) while never actually having SageAttention
# installed. Caching the real wheel here lets every worker boot do a fast
# local install (no compilation) from it instead.
WHEEL_CACHE_DIR="$VOLUME_DIR/sageattention_wheel"
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

    # CPATH alone isn't enough - it only helps the compiler find headers.
    # SageAttention's build (via torch.utils.cpp_extension) also needs
    # CUDA_HOME to locate the nvcc compiler binary itself, which this pod's
    # apt toolkit provided but nothing in this image does - PyTorch's pip
    # wheel bundles CUDA *runtime* libraries, not the compiler toolchain.
    # nvidia-cuda-nvcc is NVIDIA's own pip-installable nvcc, pinned to the
    # 13.0.x line to match the pinned torch build (2.10.0+cu130) - installs
    # into the same nvidia/cu13/ namespace dir the CPATH headers were just
    # found under (verified by downloading and inspecting the wheel: nvcc
    # lands at <site-packages>/nvidia/cu13/bin/nvcc).
    pip install "nvidia-cuda-nvcc==13.0.88"

    NVCC_PATH=$(find /usr/local/lib/python3.12/site-packages/nvidia/cu13/bin /usr/local/lib/python3.12/dist-packages/nvidia/cu13/bin -maxdepth 1 -type f -name nvcc 2>/dev/null | head -1; true)
    if [ -z "$NVCC_PATH" ]; then
        echo "[comfylab-engine] FATAL: nvcc not found after installing nvidia-cuda-nvcc - package layout may have changed." >&2
        exit 1
    fi
    export CUDA_HOME
    CUDA_HOME=$(dirname "$(dirname "$NVCC_PATH")")
    export PATH="$CUDA_HOME/bin:$PATH"
    echo "[comfylab-engine] CUDA_HOME set to: $CUDA_HOME"

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
mkdir -p "$MODELS_DIR/diffusion_models" "$MODELS_DIR/text_encoders" "$MODELS_DIR/vae"

download_if_missing() {
    local url="$1"
    local dest_dir="$2"
    local filename
    filename=$(basename "$url")
    if [ -f "$dest_dir/$filename" ]; then
        echo "[comfylab-engine]   $filename already present, skipping."
    else
        echo "[comfylab-engine]   downloading $filename..."
        wget -q --show-progress -P "$dest_dir" "$url"
    fi
}

download_if_missing "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors" "$MODELS_DIR/diffusion_models"
download_if_missing "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors" "$MODELS_DIR/text_encoders"
download_if_missing "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_video_vae_fp16.safetensors" "$MODELS_DIR/vae"
download_if_missing "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_audio_vae_fp32.safetensors" "$MODELS_DIR/vae"

echo "[comfylab-engine] Done. Engine + models ready on volume."
