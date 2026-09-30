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
    echo "[comfylab-engine] Marker found at $MARKER_FILE - engine already installed on this volume, skipping build."
else
    echo "[comfylab-engine] First run on this volume - installing SageAttention from source..."

    # PyTorch's pip wheel bundles its own copies of the CUDA library headers
    # (cusparse.h etc.) under dist-packages/nvidia/*/include - this pod's
    # apt toolkit doesn't ship them itself, so nvcc can't find them without
    # this.
    NVIDIA_INCLUDE_DIRS=$(find /usr/local/lib/python3.12/dist-packages/nvidia -maxdepth 2 -type d -name include 2>/dev/null | paste -sd: -)
    if [ -z "$NVIDIA_INCLUDE_DIRS" ]; then
        echo "[comfylab-engine] FATAL: could not find PyTorch's bundled nvidia/*/include dirs under dist-packages - CPATH fix has nothing to point at. Toolkit layout may have changed." >&2
        exit 1
    fi
    export CPATH="${NVIDIA_INCLUDE_DIRS}:${CPATH:-}"
    echo "[comfylab-engine] CPATH set to: $CPATH"

    pip uninstall -y sageattention >/dev/null 2>&1 || true

    # --no-build-isolation: pip's isolated build sandbox can't see the
    # already-installed torch, which SageAttention's build needs to detect
    # the GPU arch and link against.
    pip install --no-build-isolation "git+${SAGEATTENTION_REPO_URL}@${SAGEATTENTION_COMMIT}" \
        2>&1 | tee /tmp/sageattention_build.log

    echo "[comfylab-engine] Verifying the install (from outside any SageAttention source clone, to avoid picking up its own sageattention/ subfolder instead of the installed package)..."
    (cd /tmp && python3 -c "import sageattention; print(sageattention)")

    touch "$MARKER_FILE"
    echo "[comfylab-engine] SageAttention build OK, marker written to $MARKER_FILE."
fi

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
