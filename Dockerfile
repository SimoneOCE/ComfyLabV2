# Lean by design: ComfyUI + KJNodes are pure-Python (no GPU-arch-specific
# compilation), so they're safe to bake into the image for fast cold boots.
# SageAttention is NOT built here — its setup.py detects the attached GPU's
# compute capability at build time (torch.cuda.get_device_capability()), the
# same pattern our koboldcpp ConvRot build uses via -arch=native. A
# `docker build` has no GPU attached, so building it here would either fail
# to compile a matching kernel or require hardcoding an arch (risky if a
# worker ever lands on a GPU other than the one this was tuned for).
# SageAttention's actual *build* runs at runtime instead, on a worker with
# the real target GPU present — see comfyui_engine/ensure_comfyui_engine.sh,
# same pattern as speedlabv2's koboldcpp_engine/build_convrot_cuda_engine.sh.
#
# The CUDA *compiler* (nvcc) itself is a different matter and IS baked in
# below - merely installing it needs no GPU, only running SageAttention's
# build against it does. Confirmed the hard way: a pip-installable nvcc
# package exists for CUDA 13.x (nvidia-cuda-nvcc) but the CUDA 12.x line's
# equivalent (nvidia-cuda-nvcc-cu12) turns out to not actually ship an nvcc
# binary at all when its wheel is inspected directly - NVIDIA's pip
# packaging isn't consistent across CUDA major versions. Their apt packages
# are.
#
# Model weights (MiniMax H3 diffusion/text-encoder/VAE safetensors, ~40GB
# combined) are likewise never baked in — downloaded once at runtime to the
# persistent RunPod network volume by ensure_comfyui_engine.sh.
#
# Pinned to slim-bookworm (Debian 12), not the floating `python:3.12-slim`
# tag (which currently resolves to Debian 13 "trixie", confirmed via
# docker-library/python's own Dockerfile source) - NVIDIA's CUDA apt repo
# only publishes CUDA 13.x packages for trixie, nothing in the 12.x line,
# and this worker's actual GPU driver (confirmed via RunPod's own fitness
# check and nvidia-smi output on a live worker) only supports up to CUDA
# 12.8, not 13.0 - torch built for cu130 fails to initialize CUDA at all on
# this hardware ("CUDA initialization: The NVIDIA driver on your system is
# too old"), which is what actually caused SageAttention's build to see
# zero compute capabilities and fail. bookworm's NVIDIA repo has the needed
# cuda-nvcc-12-8 package; trixie's doesn't.
FROM python:3.12-slim-bookworm

# NOT the commits `git rev-parse HEAD` reported on the proven RunPod pod -
# those (700a8f66.../a3250418...) turned out to be unreachable in the real
# public repos ("fatal: unable to read tree", confirmed via a real build
# failure), meaning RunPod's pod template bakes in its own fork/rebuild
# rather than a plain clone of upstream. Pinned instead to each repo's real
# current HEAD (verified by cloning both directly and confirming the exact
# node classes this workflow needs - ComfyMathExpression/MiniMaxH3ImageToVideo/
# ResolutionSelector/SaveVideo in ComfyUI, PathchSageAttentionKJ in KJNodes -
# actually exist at these commits), which is the closest available
# reproduction of the proven pod's behavior.
ARG COMFYUI_COMMIT=fb2315f11db0ebfaafa9099a5df5227dc6bb42bc
ARG KJNODES_COMMIT=d3cfe21625e5170126ce06fbfcfe1d88108688c3

WORKDIR /opt/comfylab

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl wget ca-certificates gnupg \
    && rm -rf /var/lib/apt/lists/*

# NVIDIA's CUDA apt repo + the nvcc compiler for CUDA 12.8 specifically -
# matched to this worker's actual driver ceiling (see the FROM line's
# comment above for why 12.8, not 13.0). cuda-keyring is NVIDIA's own
# signed-repo bootstrap package; installing cuda-nvcc-12-8 needs no GPU,
# it's just file extraction - only SageAttention's own build step (which
# runs against this compiler at runtime) needs one present.
RUN curl -fsSL -o /tmp/cuda-keyring.deb \
    https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i /tmp/cuda-keyring.deb \
    && rm /tmp/cuda-keyring.deb \
    && apt-get update \
    && apt-get install -y --no-install-recommends cuda-nvcc-12-8 \
    && rm -rf /var/lib/apt/lists/*
ENV CUDA_HOME=/usr/local/cuda-12.8
ENV PATH="${CUDA_HOME}/bin:${PATH}"

# ComfyUI itself
RUN git clone https://github.com/comfyanonymous/ComfyUI.git ComfyUI \
    && cd ComfyUI \
    && git checkout "$COMFYUI_COMMIT"

# Torch pinned to CUDA 12.8, not the pod's own 13.0 build - this worker's
# actual GPU driver doesn't support cu130 (see the FROM line's comment).
# ComfyUI's own requirements.txt doesn't pin a CUDA build, and pulling from
# the default PyPI index would grab a CPU-only or mismatched-CUDA wheel.
# torchvision/torchaudio left unpinned to whatever version pip resolves as
# compatible with this exact torch build.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.10.0+cu128 torchvision torchaudio

RUN pip install --no-cache-dir -r ComfyUI/requirements.txt

# KJNodes - provides the "Patch Sage Attention" node used to actually engage
# the SageAttention kernel installed at runtime. Pure Python, no compiled
# CUDA extension of its own (verified: its requirements.txt is just
# pillow/color-matcher/matplotlib/opencv-python-headless), so unlike
# SageAttention it's safe to bake in.
RUN git clone https://github.com/kijai/ComfyUI-KJNodes.git ComfyUI/custom_nodes/ComfyUI-KJNodes \
    && cd ComfyUI/custom_nodes/ComfyUI-KJNodes \
    && git checkout "$KJNODES_COMMIT" \
    && pip install --no-cache-dir -r requirements.txt

COPY comfyui_engine ./comfyui_engine
COPY handler.py .
COPY test_input.json .

RUN pip install --no-cache-dir runpod requests boto3 psutil

ENTRYPOINT []
CMD ["python3", "-u", "handler.py"]
