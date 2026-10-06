# Panoptes workcell photo pipeline - GPU models, on-prem (no Modal, no SaaS API, no internet at run time). One venv per model,
# each the exact environment of its Modal image:
#   /opt/pi3x = serving modal_apps/pi3x_geometry.py (Pi3X joint geometry, torch 2.5.1 cu124; Pi3 code /vendor/pi3 @ 9fa3ddb)
#   /opt/sam3 = modal_apps/workcell_mask_transfer.py (SAM 3 via transformers 5.17.0, torch 2.14.0 cu13); runs
#               scripts/workcell_sam_worker.py (text prompts: objects, floor), the box-prompt mask transfer, and the tiled
#               text-prompt apps modal_apps/workcell_estop_mask.py (e-stop) and workcell_part_masks.py (object parts)
#   /opt/moge = modal_apps/workcell_moge_check.py (MoGe-3 metric depth, torch 2.13.0 cu13, FlexGEMM on Triton)
# CUDA: the runtime libraries come inside the torch wheels (cu124 and cu13). The host needs an NVIDIA driver that supports
# CUDA 13 (>= 580) and the NVIDIA Container Toolkit; Ampere or newer (A100 / L40S / RTX 6000 Ada; bf16 autocast). gcc is for
# Triton's JIT (MoGe-3).
# Pins = `pip freeze` of the cached Modal image(s) named below, read on 2026-10-05 (modal_apps/onprem_image_proof.py records the
# image's own freeze again at proof time), installed with --no-deps. No modal client: scripts/onprem/run_stage.py imports the
# apps against its stub (scripts/onprem/modal_stub/modal.py), so nothing unpinned is installed. Weights are never in an image:
# scripts/onprem/fetch_weights.py.
# Build context SRC = a directory holding this repository as workcell/ and panoptes-serving as serving/: make it with
# scripts/onprem/stage_context.sh SRC (code only, ~8 MB, with .dockerignore). Base image pinned by digest (index, 2026-10-05).
# Air-gapped servers: build on a connected machine, then scripts/onprem/airgap.sh save / load (docs/workcell-photo/ONPREM.md).
#   docker build -f SRC/workcell/docker/workcell-gpu.Dockerfile -t panoptes-workcell-gpu SRC
#   docker run --rm --gpus all --network none -v /srv/panoptes-weights:/weights -v $PWD/data:/data panoptes-workcell-gpu \
#       /opt/pi3x/bin/python scripts/onprem/run_stage.py --weights /weights /serving/modal_apps/pi3x_geometry.py --run /data/RUN
FROM python:3.11.10-slim-bookworm@sha256:840e180ebcc6e5c8efab209c43f5e40fd2af98cb49db5c7103c90539c56bb30e

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility
RUN apt-get update && apt-get install -y --no-install-recommends git build-essential libgl1 libglib2.0-0 libgomp1 libx11-6 \
        ca-certificates && rm -rf /var/lib/apt/lists/*

# pi3x-geometry (debian_slim 3.11 + git libgl1 libglib2.0-0; Pi3 9fa3ddb cloned at /vendor/pi3)
RUN git clone https://github.com/yyfz/Pi3.git /vendor/pi3 && git -C /vendor/pi3 checkout 9fa3ddb3f8d53041f8b2738df404f62223bbaa7b
RUN python -m venv /opt/pi3x && /opt/pi3x/bin/pip install --no-deps \
        aiohappyeyeballs==2.4.3 aiohttp==3.10.8 aiosignal==1.3.1 annotated-types==0.8.0 attrs==24.2.0 \
        certifi==2024.8.30 charset-normalizer==3.5.2 einops==0.8.0 filelock==4.0.12 frozenlist==1.4.1 \
        fsspec==2026.9.0 grpclib==0.4.7 h2==4.1.0 hf-xet==1.6.0 hpack==4.0.0 \
        huggingface-hub==0.36.0 hyperframe==6.0.1 idna==3.10 Jinja2==3.1.6 MarkupSafe==3.0.4 \
        mpmath==1.3.0 multidict==6.1.0 networkx==3.6.1 numpy==1.26.4 nvidia-cublas-cu12==12.4.5.8 \
        nvidia-cuda-cupti-cu12==12.4.127 nvidia-cuda-nvrtc-cu12==12.4.127 nvidia-cuda-runtime-cu12==12.4.127 nvidia-cudnn-cu12==9.1.0.70 nvidia-cufft-cu12==11.2.1.3 \
        nvidia-curand-cu12==10.3.5.147 nvidia-cusolver-cu12==11.6.1.9 nvidia-cusparse-cu12==12.3.1.170 nvidia-nccl-cu12==2.21.5 nvidia-nvjitlink-cu12==12.4.127 \
        nvidia-nvtx-cu12==12.4.127 opencv-python-headless==4.10.0.84 packaging==26.3 pillow==11.0.0 plyfile==1.1 \
        protobuf==5.29.2 pydantic==2.9.2 pydantic_core==2.23.4 PyYAML==6.0.3 requests==2.34.2 \
        safetensors==0.4.5 setuptools==65.5.1 sympy==1.13.1 torch==2.5.1 torchvision==0.20.1 \
        tqdm==4.70.1 trimesh==5.1.0 triton==3.1.0 typing_extensions==4.12.2 urllib3==2.8.0 \
        yarl==1.13.1

# workcell-mask-transfer (debian_slim 3.11; torch 2.14.0 transformers 5.17.0 accelerate pillow numpy<2.3 opencv 4.10.0.84)
RUN python -m venv /opt/sam3 && /opt/sam3/bin/pip install --no-deps \
        accelerate==1.15.0 aiohappyeyeballs==2.4.3 aiohttp==3.10.8 aiosignal==1.3.1 annotated-doc==0.0.5 \
        anyio==4.15.1 attrs==24.2.0 certifi==2024.8.30 click==8.5.0 cuda-bindings==13.4.3 \
        cuda-pathfinder==1.8.3 cuda-toolkit==13.0.3.0 filelock==4.0.12 frozenlist==1.4.1 fsspec==2026.9.0 \
        grpclib==0.4.7 h11==0.16.0 h2==4.1.0 hf-xet==1.6.0 hpack==4.0.0 \
        httpcore==1.0.9 httpx==0.28.1 huggingface_hub==1.33.0 hyperframe==6.0.1 idna==3.10 \
        Jinja2==3.1.6 markdown-it-py==4.2.0 MarkupSafe==3.0.4 mdurl==0.1.2 mpmath==1.3.0 \
        multidict==6.1.0 networkx==3.6.1 numpy==2.2.6 nvidia-cublas==13.1.1.3 nvidia-cuda-cupti==13.0.85 \
        nvidia-cuda-nvrtc==13.0.88 nvidia-cuda-runtime==13.0.96 nvidia-cudnn-cu13==9.24.0.43 nvidia-cufft==12.0.0.61 nvidia-cufile==1.15.1.6 \
        nvidia-curand==10.4.0.35 nvidia-cusolver==12.0.4.66 nvidia-cusparse==12.6.3.3 nvidia-cusparselt-cu13==0.8.1 nvidia-nccl-cu13==2.30.7 \
        nvidia-nvjitlink==13.4.92 nvidia-nvshmem-cu13==3.4.5 nvidia-nvtx==13.0.85 opencv-python-headless==4.10.0.84 packaging==26.3 \
        pillow==12.3.0 protobuf==5.29.2 psutil==7.2.2 Pygments==2.21.0 PyYAML==6.0.3 \
        regex==2026.9.29 rich==15.0.0 safetensors==0.8.0 setuptools==84.0.0 shellingham==1.5.4 \
        sympy==1.14.0 tokenizers==0.23.2 torch==2.14.0 torchvision==0.29.1 tqdm==4.70.1 \
        transformers==5.17.0 triton==3.8.0 typer==0.27.2 typing_extensions==4.16.0 yarl==1.13.1

# workcell-moge-check (moge3_app image: debian_slim 3.11 + git build-essential, torch torchvision MoGe huggingface_hub opencv
# trimesh, then libgl1 libgomp1 libx11-6 + open3d 0.19.0); the unpinned Modal definition resolved to these
RUN python -m venv /opt/moge && /opt/moge/bin/pip install --no-deps \
        addict==2.4.0 aiohappyeyeballs==2.4.3 aiohttp==3.10.8 \
        aiosignal==1.3.1 annotated-doc==0.0.5 annotated-types==0.8.0 \
        anyio==4.14.2 asttokens==3.0.2 attrs==24.2.0 \
        blinker==1.9.0 brotli==1.2.0 certifi==2024.8.30 \
        charset-normalizer==3.5.1 click==8.5.0 cloudpickle==3.1.2 \
        comm==0.2.3 ConfigArgParse==1.8.0 contourpy==1.3.3 \
        cuda-bindings==13.3.1 cuda-pathfinder==1.7.0 cuda-toolkit==13.0.3.0 \
        cycler==0.12.1 dash==4.4.1 executing==2.2.1 \
        fastapi==0.141.1 fastjsonschema==2.22.2 filelock==3.32.4 \
        Flask==3.1.3 "flex_gemm @ git+https://github.com/JeffreyXiang/FlexGEMM.git@b2fadb29d41846c7981ade6801ffc689fae119cf" fonttools==4.63.0 \
        frozenlist==1.4.1 fsspec==2026.7.0 glcontext==3.0.0 \
        gradio==6.26.0 gradio_client==2.6.1 groovy==0.1.2 \
        grpclib==0.4.7 h11==0.16.0 h2==4.1.0 \
        hf-gradio==0.4.1 hf-xet==1.6.0 hpack==4.0.0 \
        httpcore==1.0.9 httpx==0.28.1 huggingface_hub==1.28.0 \
        hyperframe==6.0.1 idna==3.10 importlib_metadata==9.0.1 \
        ipython==9.17.1 ipython_pygments_lexers==1.1.1 ipywidgets==8.1.9 \
        itsdangerous==2.2.0 janus==2.0.0 jedi==0.20.0 \
        Jinja2==3.1.6 joblib==1.6.0 jsonschema-specifications==2025.9.1 \
        jsonschema==4.26.0 jupyter_core==5.9.1 jupyterlab_widgets==3.0.17 \
        kiwisolver==1.5.0 markdown-it-py==4.2.0 MarkupSafe==3.0.3 \
        matplotlib-inline==0.2.2 matplotlib==3.11.1 mdurl==0.1.2 \
        moderngl==5.12.0 "moge @ git+https://github.com/microsoft/MoGe.git@74fbce054ebed49800de42d0ad0e83495065719a" mpmath==1.3.0 \
        multidict==6.1.0 narwhals==2.26.0 nbformat==5.11.1 \
        nest-asyncio==1.6.0 networkx==3.6.1 numpy==2.4.6 \
        nvidia-cublas==13.1.1.3 nvidia-cuda-cupti==13.0.85 nvidia-cuda-nvrtc==13.0.88 \
        nvidia-cuda-runtime==13.0.96 nvidia-cudnn-cu13==9.20.0.48 nvidia-cufft==12.0.0.61 \
        nvidia-cufile==1.15.1.6 nvidia-curand==10.4.0.35 nvidia-cusolver==12.0.4.66 \
        nvidia-cusparse==12.6.3.3 nvidia-cusparselt-cu13==0.8.1 nvidia-nccl-cu13==2.29.7 \
        nvidia-nvjitlink==13.3.33 nvidia-nvshmem-cu13==3.4.5 nvidia-nvtx==13.0.85 \
        open3d==0.19.0 opencv-python-headless==5.0.0.93 orjson==3.12.0 \
        packaging==26.3 pandas==3.0.5 parso==0.8.7 \
        pexpect==4.9.0 pillow==12.3.0 "pipeline @ git+https://github.com/EasternJournalist/pipeline.git@1c511390d90226c00c101f34b84df26a0f8789b4" \
        platformdirs==4.12.3 plotly==7.1.0 prompt_toolkit==3.0.53 \
        protobuf==5.29.2 psutil==7.2.2 ptyprocess==0.7.0 \
        pure_eval==0.2.4 pydantic==2.13.4 pydantic_core==2.46.4 \
        pydub==0.25.1 Pygments==2.21.0 pyparsing==3.3.2 \
        pyquaternion==0.9.9 python-dateutil==2.9.0.post0 python-multipart==0.0.32 \
        pytz==2026.3.post1 PyYAML==6.0.3 referencing==0.37.0 \
        requests==2.34.2 retrying==1.4.2 rich==15.0.0 \
        rpds-py==2026.9.1 safehttpx==0.1.7 scikit-learn==1.9.1 \
        scipy==1.17.1 semantic-version==2.10.0 setuptools==84.0.0 \
        shellingham==1.5.4 six==1.17.0 stack-data==0.6.3 \
        starlette==1.6.0 sympy==1.14.0 threadpoolctl==3.7.0 \
        tomlkit==0.14.0 torch==2.13.0 torchvision==0.28.0 \
        tqdm==4.70.0 traitlets==5.16.1 trimesh==5.0.0 \
        triton==3.7.1 typer==0.27.1 typing-inspection==0.4.4 \
        typing_extensions==4.16.0 urllib3==2.7.0 "utils3d_moge @ git+https://github.com/EasternJournalist/utils3d-moge.git@62f09d58509485564e24d5d9f6aac9ee9ebc0c37" \
        uvicorn==0.52.4 wcwidth==0.9.2 Werkzeug==3.1.9 \
        widgetsnbextension==4.0.16 yarl==1.13.1 zipp==4.1.1

# Container paths of the Modal images' add_local_* (so every function body runs unchanged), then only the files the GPU
# stages use: an unrelated edit elsewhere in the repositories does not invalidate these heavy layers.
COPY serving/ehs_spatial /serving/ehs_spatial
COPY serving/scripts/candidate_pi3x_backend.py serving/scripts/candidate_geometry_backend.py /serving/scripts/
COPY serving/modal_apps/pi3x_geometry.py /serving/modal_apps/pi3x_geometry.py
COPY workcell/scripts/workcell_sam_worker.py /sam/workcell_sam_worker.py
COPY workcell/scripts/workcell_shape_check.py /sam/shape_core.py
COPY workcell/scripts/workcell_checks/transfer.py /sam/transfer.py
COPY workcell/scripts/workcell_shape_check.py /check/shape_core.py
COPY workcell/scripts/workcell_sam_worker.py workcell/scripts/workcell_shape_check.py /workcell/scripts/
COPY workcell/scripts/workcell_checks/transfer.py /workcell/scripts/workcell_checks/transfer.py
COPY workcell/modal_apps/workcell_moge_check.py workcell/modal_apps/moge3_app.py workcell/modal_apps/workcell_mask_transfer.py /workcell/modal_apps/
COPY workcell/modal_apps/workcell_estop_mask.py workcell/modal_apps/workcell_part_masks.py /workcell/modal_apps/
COPY workcell/scripts/onprem /workcell/scripts/onprem
ENV HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
WORKDIR /workcell
CMD ["/opt/pi3x/bin/python", "scripts/onprem/run_stage.py", "--self-test"]
