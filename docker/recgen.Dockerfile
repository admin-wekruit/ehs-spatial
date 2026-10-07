# RecGen object models, on-prem: the image of panoptes-serving modal_apps/lucida_assets.py (gpu_image) as a Dockerfile, with
# the resolved pins of the cached Modal image. RecGen code fe3c931 (TRI-ML/recgen), DINOv2 code 7764ea0; weights TRI-ML/RecGen
# @ bc0df7d + DINOv2 ViT-L/14 reg4 via scripts/onprem/fetch_weights.py --models recgen (DIR/recgen; run_stage.py --weights DIR
# gives /cache a run-private overlay of it, so the job copies RecGen writes to /cache/jobs (photo data) stay out of DIR and are
# deleted when the stage ends).
# LICENCE: RecGen code is Toyota Research Institute non-commercial and its weights CC-BY-NC-4.0 - an on-prem build is not a
# licence to use them commercially.
# (The vendored upstream Dockerfile, outputs/candidate-evaluation/lucida-replica-01/generation/vendor-recgen/Dockerfile, uses
# Python 3.11, torch 2.4.1, unpinned spconv/flash-attn and training extras; the reports were made with this image instead.)
# Pins = `pip freeze` of the cached Modal image(s) named below, read on 2026-10-05 (modal_apps/onprem_image_proof.py records the
# image's own freeze again at proof time), installed with --no-deps. No modal client: scripts/onprem/run_stage.py imports the
# apps against its stub (scripts/onprem/modal_stub/modal.py), so nothing unpinned is installed. Weights are never in an image:
# scripts/onprem/fetch_weights.py.
# Build context SRC = a directory holding this repository as workcell/ and panoptes-serving as serving/: make it with
# scripts/onprem/stage_context.sh SRC (code only, ~8 MB, with .dockerignore). Base image pinned by digest (index, 2026-10-05).
# Air-gapped servers: build on a connected machine, then scripts/onprem/airgap.sh save / load (docs/workcell-photo/ONPREM.md).
#   docker build -f SRC/workcell/docker/recgen.Dockerfile -t panoptes-recgen SRC
#   docker run --rm --gpus all --network none -v /srv/panoptes-weights:/weights -v $PWD/data:/data panoptes-recgen \
#       python /workcell/scripts/onprem/run_stage.py --weights /weights /serving/modal_apps/lucida_assets.py --mode check \
#       --output-dir /data/check
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04@sha256:7012e535a47883527d402da998384c30b936140c05e2537158c80b8143ee7425

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 TORCH_CUDA_ARCH_LIST=8.0 MAX_JOBS=4 ATTN_BACKEND=xformers \
    PYTHONPATH=/opt/recgen SPCONV_ALGO=native HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_OFFLINE=1 \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python PATH=/opt/recgen-py/bin:$PATH
RUN apt-get update && apt-get install -y --no-install-recommends git build-essential ninja-build libgl1 libglib2.0-0 curl \
        ca-certificates && rm -rf /var/lib/apt/lists/*
# Python 3.10.13 as Modal's add_python='3.10' (python-build-standalone), via uv
RUN curl -LsSf https://astral.sh/uv/0.5.14/install.sh | env UV_INSTALL_DIR=/usr/local/bin INSTALLER_NO_MODIFY_PATH=1 sh \
    && uv python install 3.10.13 && uv venv --seed --python 3.10.13 /opt/recgen-py
RUN uv pip install --python /opt/recgen-py/bin/python --no-deps --index-strategy unsafe-best-match \
        --extra-index-url https://download.pytorch.org/whl/cu121 \
        aiohappyeyeballs==2.4.3 aiohttp==3.10.8 aiosignal==1.3.1 async-timeout==4.0.3 attrs==24.2.0 \
        ccimport==0.4.4 certifi==2024.8.30 charset-normalizer==3.5.1 cumm-cu120==0.4.11 easydict==1.13 \
        einops==0.8.1 filelock==3.32.3 fire==0.7.1 frozenlist==1.4.1 fsspec==2026.7.0 \
        grpclib==0.4.7 h2==4.1.0 hf-xet==1.6.0 hpack==4.0.0 huggingface-hub==0.36.0 \
        hyperframe==6.0.1 idna==3.10 Jinja2==3.1.6 lark==1.3.1 MarkupSafe==3.0.3 \
        mpmath==1.3.0 multidict==6.1.0 networkx==3.4.2 ninja==1.13.2 numpy==1.26.4 \
        nvidia-cublas-cu12==12.1.3.1 nvidia-cuda-cupti-cu12==12.1.105 nvidia-cuda-nvrtc-cu12==12.1.105 nvidia-cuda-runtime-cu12==12.1.105 nvidia-cudnn-cu12==9.1.0.70 \
        nvidia-cufft-cu12==11.0.2.54 nvidia-curand-cu12==10.3.2.106 nvidia-cusolver-cu12==11.4.5.107 nvidia-cusparse-cu12==12.1.0.106 nvidia-nccl-cu12==2.20.5 \
        nvidia-nvjitlink-cu12==12.9.86 nvidia-nvtx-cu12==12.1.105 opencv-python-headless==4.11.0.86 packaging==26.3 pccm==0.4.16 \
        pillow==11.3.0 plyfile==1.1.2 portalocker==4.3.0 protobuf==5.29.6 pybind11==3.1.0 \
        PyYAML==6.0.3 requests==2.34.2 safetensors==0.6.2 scipy==1.15.3 setuptools==68.1.2 \
        spconv-cu120==2.3.6 sympy==1.14.0 termcolor==3.3.0 torch==2.4.0+cu121 torchvision==0.19.0+cu121 \
        tqdm==4.67.1 trimesh==4.7.4 triton==3.0.0 typing_extensions==4.12.2 urllib3==2.7.0 \
        xformers==0.0.27.post2 yarl==1.13.1
RUN git clone https://github.com/TRI-ML/recgen.git /opt/recgen && git -C /opt/recgen checkout --detach fe3c9315b439c50ada8b60c12b469d739fd722db \
    && git clone https://github.com/facebookresearch/dinov2.git /opt/dinov2 && git -C /opt/dinov2 checkout --detach 7764ea0f912e53c92e82eb78a2a1631e92725fc8 \
    && uv pip install --python /opt/recgen-py/bin/python --no-deps -e /opt/recgen \
    && python -c 'import xformers.ops as xops; assert xops.fmha.BlockDiagonalMask; from recgen_inference import build_recgen, generate; from recgen_inference.recgen_modules.models.structured_latent_vae import SLatMeshDecoder; print("RecGen import ready")'
COPY serving/modal_apps/lucida_assets.py /serving/modal_apps/lucida_assets.py
COPY serving/scripts/research /serving/scripts/research
COPY workcell/scripts/onprem /workcell/scripts/onprem
COPY workcell/scripts/workcell_recgen_worker.py /workcell/scripts/workcell_recgen_worker.py
WORKDIR /serving
CMD ["python", "/workcell/scripts/onprem/run_stage.py", "--self-test"]
