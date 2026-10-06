# Panoptes workcell photo pipeline - Depth Anything 3 joint geometry (alternative to Pi3X), on-prem (no Modal, no SaaS API, no
# internet at run time).
#   /opt/da3 = the environment the fair A/B ran DA3 in (research-notes/geometry-licence-ab-fair-2026-10-05, fair_ab_modal.py
#              gpu_image): its `pip freeze --all` read on 2026-10-05, installed with --no-deps; minus uniception (MapAnything only),
#              pip, wheel and uv. torch 2.5.1 cu124 (CUDA runtime inside the wheels): NVIDIA driver >= 525 + NVIDIA Container
#              Toolkit; Ampere or newer for bf16 autocast (A100 / L40S / RTX 6000 Ada). DA3-LARGE-1.1 peaks at 3.7 GiB, DA3-BASE 1.5 GiB.
#   /vendor/depth-anything-3 = ByteDance-Seed/Depth-Anything-3 @ 3d835ec (Apache-2.0), the revision the runner checks.
#   The stage is serving scripts/candidate_geometry_backend.py (DA3Runner through MapAnythingAdapter -> RUN/geometry frame schema).
# LICENCE of the weights (scripts/onprem/fetch_weights_da3.py): DA3-BASE Apache-2.0. DA3-LARGE-1.1 is DISPUTED - its Hugging Face card
# says apache-2.0 but the official README lists it CC BY-NC 4.0; do not use it commercially until the authors confirm.
# Weights are never in the image: fetch once with scripts/onprem/fetch_weights_da3.py --cache DIR (internet), then mount DIR.
# Build context SRC = scripts/onprem/stage_context.sh SRC (this repository as workcell/, panoptes-serving as serving/, ~8 MB).
#   docker build -f SRC/workcell/docker/da3.Dockerfile -t panoptes-workcell-da3 SRC
#   docker run --rm --gpus all --network none -v /srv/panoptes-weights:/weights:ro -v $PWD/data:/data panoptes-workcell-da3 \
#       /opt/da3/bin/python /serving/scripts/candidate_geometry_backend.py --vendor-dir /vendor/depth-anything-3 \
#       --model-dir /weights/local/da3-large-1.1 --model-id depth-anything/DA3-LARGE-1.1 --device cuda \
#       --inputs /data/RUN/evidence/canonical/frame_0001.png /data/RUN/evidence/canonical/frame_0002.png --output /data/RUN/geometry
# Air-gapped servers: build on a connected machine, then scripts/onprem/airgap.sh save / load (docs/workcell-photo/ONPREM.md).
FROM python:3.11.10-slim-bookworm@sha256:840e180ebcc6e5c8efab209c43f5e40fd2af98cb49db5c7103c90539c56bb30e

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility
RUN apt-get update && apt-get install -y --no-install-recommends git libgl1 libglib2.0-0 libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git /vendor/depth-anything-3 \
    && git -C /vendor/depth-anything-3 checkout 3d835ec1a5802d64a8b8b15f817a1ab54809bfe4
RUN python -m venv /opt/da3 && /opt/da3/bin/pip install --no-deps \
        addict==2.4.0 aiohappyeyeballs==2.4.3 aiohttp==3.10.8 aiosignal==1.3.1 annotated-types==0.8.0 \
        antlr4-python3-runtime==4.9.3 attrs==24.2.0 certifi==2024.8.30 charset-normalizer==3.5.2 cloudpickle==3.1.2 \
        contourpy==1.3.3 cycler==0.12.1 einops==0.8.0 filelock==4.0.12 fonttools==4.66.1 \
        frozenlist==1.4.1 fsspec==2026.9.0 grpclib==0.4.7 h2==4.1.0 hf-xet==1.6.0 \
        hpack==4.0.0 huggingface-hub==0.36.0 hydra-core==1.3.2 hyperframe==6.0.1 idna==3.10 \
        imageio==2.36.0 jaxtyping==0.2.36 Jinja2==3.1.6 joblib==1.6.0 kiwisolver==1.5.1 \
        MarkupSafe==3.0.4 matplotlib==3.9.2 mpmath==1.3.0 multidict==6.1.0 natsort==8.4.0 \
        networkx==3.6.1 numpy==1.26.4 nvidia-cublas-cu12==12.4.5.8 nvidia-cuda-cupti-cu12==12.4.127 nvidia-cuda-nvrtc-cu12==12.4.127 \
        nvidia-cuda-runtime-cu12==12.4.127 nvidia-cudnn-cu12==9.1.0.70 nvidia-cufft-cu12==11.2.1.3 nvidia-curand-cu12==10.3.5.147 nvidia-cusolver-cu12==11.6.1.9 \
        nvidia-cusparse-cu12==12.3.1.170 nvidia-nccl-cu12==2.21.5 nvidia-nvjitlink-cu12==12.4.127 nvidia-nvtx-cu12==12.4.127 omegaconf==2.3.0 \
        opencv-python-headless==4.10.0.84 orjson==3.10.11 packaging==26.3 pillow==11.0.0 plyfile==1.1 \
        protobuf==5.29.2 pydantic==2.9.2 pydantic_core==2.23.4 pyparsing==3.3.3 python-box==7.2.0 \
        python-dateutil==2.9.0.post0 PyYAML==6.0.3 requests==2.34.2 safetensors==0.4.5 scikit-learn==1.5.2 \
        scipy==1.14.1 setuptools==65.5.1 six==1.17.0 sympy==1.13.1 termcolor==2.5.0 \
        threadpoolctl==3.7.0 timm==1.0.11 torch==2.5.1 torchvision==0.20.1 tqdm==4.67.1 \
        trimesh==5.1.0 triton==3.1.0 typing_extensions==4.12.2 urllib3==2.8.0 yarl==1.13.1

# Only the files the DA3 stage and the weight fetch use: an unrelated edit elsewhere does not invalidate the layers above.
COPY serving/ehs_spatial /serving/ehs_spatial
COPY serving/scripts/candidate_geometry_backend.py /serving/scripts/candidate_geometry_backend.py
COPY workcell/scripts/onprem/fetch_weights.py workcell/scripts/onprem/fetch_weights_da3.py /workcell/scripts/onprem/
ENV HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
WORKDIR /workcell
CMD ["/opt/da3/bin/python", "scripts/onprem/fetch_weights_da3.py", "--self-test"]
