# Lean by design: ComfyUI + KJNodes are pure-Python (no GPU-arch-specific
# compilation), so they're safe to bake into the image for fast cold boots.
# SageAttention is NOT baked in here — its setup.py detects the attached
# GPU's compute capability at build time (torch.cuda.get_device_capability()),
# the same pattern our koboldcpp ConvRot build uses via -arch=native. A
# `docker build` has no GPU attached, so baking it in would either fail to
# compile a matching kernel or require hardcoding an arch (risky if a worker
# ever lands on a GPU other than the one this was tuned for). SageAttention
# is installed at runtime instead, on a worker with the real target GPU
# present — see comfyui_engine/ensure_comfyui_engine.sh, same pattern as
# speedlabv2's koboldcpp_engine/build_convrot_cuda_engine.sh.
#
# Model weights (MiniMax H3 diffusion/text-encoder/VAE safetensors, ~40GB
# combined) are likewise never baked in — downloaded once at runtime to the
# persistent RunPod network volume by ensure_comfyui_engine.sh.
FROM python:3.12-slim

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
    git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ComfyUI itself
RUN git clone https://github.com/comfyanonymous/ComfyUI.git ComfyUI \
    && cd ComfyUI \
    && git checkout "$COMFYUI_COMMIT"

# Torch pinned to the exact version+build confirmed on the proven pod
# (torch.__version__ == "2.10.0+cu130") - ComfyUI's own requirements.txt
# doesn't pin a CUDA build, and pulling from the default PyPI index would
# grab a CPU-only or mismatched-CUDA wheel. torchvision/torchaudio left
# unpinned to whatever version pip resolves as compatible with that exact
# torch build.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu130 \
    torch==2.10.0+cu130 torchvision torchaudio

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
