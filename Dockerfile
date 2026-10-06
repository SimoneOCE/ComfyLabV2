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
# so bookworm is the one base that covers both (verified directly: bookworm's
# repo has cuda-nvcc-12-8 through cuda-nvcc-13-4, trixie's has only
# cuda-nvcc-13-1 and up, no 13-0 at all).
#
# Runs on CUDA 13.0 now - the endpoint's CUDA Version floor has been raised
# to 13.0, guaranteeing every worker RunPod schedules for it has a driver new
# enough. Before that floor was set, a live worker's driver only supported up
# to CUDA 12.8 (confirmed via RunPod's own fitness check and nvidia-smi),
# which made torch built for cu130 fail to initialize CUDA at all
# ("CUDA initialization: The NVIDIA driver on your system is too old") - that
# was the actual root cause of SageAttention's build seeing zero compute
# capabilities and failing. Reverting to cu130 now re-enables ComfyUI's
# comfy_kitchen accelerated backend for this checkpoint's int8-convrot
# quantized ops, which cu128 ran on the slower "eager" fallback path instead
# (visible in a real run's logs: "Found comfy_kitchen backend cuda:
# {'available': True, 'disabled': True, ...}" - disabled specifically because
# it needs cu130+). Real numbers from that cu128 run: 541-563s per
# generation vs the proven pod's 336.92s baseline (both on cu130) - if this
# reintroduces that speed, the endpoint's narrower CUDA floor was worth the
# trade explained when it was raised (a smaller eligible worker pool, some
# queue-time risk).
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

# libgl1/libglib2.0-0: OpenCV's GUI build (opencv-python, pulled in by
# ultralytics and scenedetect for the face refine nodes) links libGL.so.1,
# libglib/libgthread and libxcb, none of which ship in slim-bookworm. Without
# them `import cv2` fails for everything - seen on the worker as KJNodes'
# "OpenCV not installed" after the refine deps went in.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl wget ca-certificates gnupg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# NVIDIA's CUDA apt repo + the nvcc compiler for CUDA 13.0 - matches the
# endpoint's CUDA Version floor (now raised to 13.0, so every worker RunPod
# schedules here has a driver new enough). cuda-keyring is NVIDIA's own
# signed-repo bootstrap package; installing cuda-nvcc-13-0 needs no GPU,
# it's just file extraction - only SageAttention's own build step (which
# runs against this compiler at runtime) needs one present. Verified this
# exact package (13.0.88-1) exists in bookworm's repo and installs to
# /usr/local/cuda-13.0/bin/nvcc by downloading and inspecting the .deb
# directly, same as was done for the 12.8 package it replaces.
RUN curl -fsSL -o /tmp/cuda-keyring.deb \
    https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i /tmp/cuda-keyring.deb \
    && rm /tmp/cuda-keyring.deb \
    && apt-get update \
    && apt-get install -y --no-install-recommends cuda-nvcc-13-0 \
    && rm -rf /var/lib/apt/lists/*
ENV CUDA_HOME=/usr/local/cuda-13.0
# "video" makes the NVIDIA container runtime mount the driver's NVENC/NVDEC
# libraries (libnvidia-encode), which the refine's GPU H.264 encoding needs
# (ComfyLabSaveVideoNVENC; it falls back to CPU libx264 without them).
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,video
ENV PATH="${CUDA_HOME}/bin:${PATH}"

# ComfyUI itself
RUN git clone https://github.com/comfyanonymous/ComfyUI.git ComfyUI \
    && cd ComfyUI \
    && git checkout "$COMFYUI_COMMIT"

# Torch pinned to CUDA 13.0 - matches the endpoint's CUDA Version floor
# (see the FROM line's comment) and the proven pod's own build
# (torch.__version__ == "2.10.0+cu130"). ComfyUI's own requirements.txt
# doesn't pin a CUDA build, and pulling from the default PyPI index would
# grab a CPU-only or mismatched-CUDA wheel. torchvision pinned to torch
# 2.10's own release so it can't drift. No torchaudio: core ComfyUI dropped
# it, and its only user here (MiniMaxH3NativeAudioLock, for the old H3 face
# refine) is gone. If it ever comes back, pin it to 2.10.0+cu130 - an
# unpinned torchaudio resolved to 2.11 and failed to load against torch 2.10.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu130 \
    torch==2.10.0+cu130 torchvision==0.25.0+cu130

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

# Face refine (post-generation "refine faces" pass - see handler.py's
# run_face_refine). Only the face tracker (H3FaceTrackCrop) and stitch-back
# (H3FaceStitch) of Carasibana/ComfyUI-H3-FaceRefine (MIT) are used; the
# redraw is our comfylab_face_wan node. Pure Python, and its nodes import
# their detector/scene-cut deps lazily (only when a refine runs).
# insightface is deliberately NOT installed (needs a C++ build): identity
# matching is off, which never imports it.
ARG FACEREFINE_COMMIT=d8521d14fe0d721d80cd9417fff5a559cbc21aba
RUN git clone https://github.com/Carasibana/ComfyUI-H3-FaceRefine.git ComfyUI/custom_nodes/ComfyUI-H3-FaceRefine \
    && cd ComfyUI/custom_nodes/ComfyUI-H3-FaceRefine \
    && git checkout "$FACEREFINE_COMMIT" \
    && pip install --no-cache-dir "ultralytics==8.4.171" scipy "scenedetect==0.7.1"
# Fail the BUILD, not a user's job, if cv2 (the tracker's detector needs it;
# it broke on a worker once for want of libGL) can't import.
RUN python -c "import torch, cv2; print('torch', torch.__version__, '| cv2', cv2.__version__)"

COPY comfyui_engine ./comfyui_engine
COPY handler.py .
# Free-trial watermark (handler.py apply_watermark).
COPY assets ./assets
COPY test_input.json .

# websocket-client: step progress from ComfyUI's websocket (handler.py
# ProgressWatcher). Pinned like everything else in this image.
RUN pip install --no-cache-dir runpod requests boto3 psutil "websocket-client==1.8.0"

ENTRYPOINT []
CMD ["python3", "-u", "handler.py"]
