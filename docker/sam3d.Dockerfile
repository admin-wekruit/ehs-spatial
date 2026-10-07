# SAM 3D Objects (facebook/sam-3d-objects, SAM License), on-prem, mesh only: the licence-clean RecGen replacement candidate
# (research-notes/completion-licence-ab-2026-10-05). Runs modal_apps/sam3d_research.py's SAM3DObjects unchanged through
# scripts/onprem/run_stage.py (mesh decoder only, reviewed mesh-only patch, internal depth model disabled: only the pointmap
# we pass places the object). No Modal, no SaaS, no network at run time.
# Pins: SAM 3D code facebookresearch/sam-3d-objects f91db411 (+ modal_apps/sam3d_mesh_only.patch, receipt in site-packages
# sam3d_objects/panoptes_mesh_build.json); weights facebook/sam-3d-objects @ 2e73555 via scripts/onprem/fetch_weights_sam3d.py.
# Python packages = `pip freeze` of the research image (modal_apps/sam3d_research.image, as fast_report/sam3d-constraints.txt
# records it), installed with --no-deps, MINUS:
#   bpy 4.3.0 (GPL-3.0), pymeshfix 0.17.0 (GPL-3.0), smplx 0.1.28 (MPI non-commercial)  - upstream requirements.txt
#   igraph 0.11.8 (GPL-2+), plyfile 1.1.3 (GPL-3), point-cloud-utils 0.29.5 (GPL-3), pdoc3 0.10.0 (AGPL-3)
#   gsplat (Gaussian splatting renderer; the mesh-only path never imports it)
# All of them are imported only by the Gaussian / texture-baking code the mesh-only patch makes lazy, or not at all
# (grep of the pinned sources; the proof records sys.modules). ffmpeg (apt) is not installed either.
# DINOv2 (Apache-2.0): SAM 3D's four embedders call torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg',
# source='github'). The code is vendored here at 7764ea0 (= its main today) as torch.hub's own cache directory
# $TORCH_HOME/hub/facebookresearch_dinov2_main, and scripts/onprem/run_stage.py turns every torch.hub.load(..., source='github')
# into source='local' on that directory (and refuses torch.hub downloads), so there is no GitHub default-branch probe even when
# the container has a network (torch.hub._parse_repo_info would otherwise open github.com first). The ViT-L/14 reg4 weights
# (1.2 GB) are NOT in the image: fetch_weights_sam3d.py puts them in the weights directory and run_stage.py --weights links
# $TORCH_HOME/hub/checkpoints to them (that path must not exist in the image).
# GPU: >= 32 GB (upstream), Ampere or newer; extensions built for sm_80 (runs on 8.x) and sm_90. Host driver >= 525 (CUDA 12.1).
# Build context SRC = this repository as workcell/ and panoptes-serving as serving/ (scripts/onprem/stage_context.sh SRC).
# Base image pinned by digest (index, 2026-10-05). Build needs the internet (PyPI, download.pytorch.org, the kaolin index,
# GitHub); air-gapped servers: build on a connected machine, then scripts/onprem/airgap.sh save / load.
#   docker build -f SRC/workcell/docker/sam3d.Dockerfile -t panoptes-sam3d SRC
#   python scripts/onprem/fetch_weights_sam3d.py --cache /srv/panoptes-weights [--source HF_CACHE/hub]     (once, online)
#   docker run --rm --gpus all --network none -v /srv/panoptes-weights:/weights -v $PWD/data:/data panoptes-sam3d \
#       python /workcell/scripts/onprem/run_stage.py --weights /weights DRIVER.py ...   (DRIVER calls SAM3DObjects().run.remote)
# LICENCE: SAM License (Meta, custom: no ITAR / military / nuclear / espionage use; ship the licence text with the weights,
# fetch_weights_sam3d.py keeps LICENSE beside them). Not legal advice.
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04@sha256:21196d81f56b48dbee70494d5f10322e1a77cc47ffe202a3bf68eab81533c20f

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONUNBUFFERED=1 \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python PATH=/opt/sam3d/bin:$PATH CUDA_HOME=/usr/local/cuda
RUN apt-get update && apt-get install -y --no-install-recommends git curl ca-certificates build-essential ninja-build \
        libgl1 libglib2.0-0 libegl1 libxrender1 libxext6 libsm6 libxi6 libxkbcommon0 libxxf86vm1 libx11-6 libgomp1 \
    && rm -rf /var/lib/apt/lists/*
# Python 3.11 (python-build-standalone, as Modal's add_python='3.11') via uv; extension builds look for Python.h under the
# interpreter's build prefix /install: point it at the real headers (as fast_report/sam3d.py does)
RUN curl -LsSf https://astral.sh/uv/0.5.14/install.sh | env UV_INSTALL_DIR=/usr/local/bin INSTALLER_NO_MODIFY_PATH=1 sh \
    && uv python install 3.11.10 && uv venv --seed --python 3.11.10 /opt/sam3d \
    && mkdir -p /install/include && ln -sfn "$(/opt/sam3d/bin/python -c 'import glob, sys; print(glob.glob(sys.base_prefix + "/include/python3.11")[0])')" /install/include/python3.11 \
    && uv pip install --python /opt/sam3d/bin/python pip==24.3.1 setuptools==75.8.0 wheel==0.45.1
# torch 2.5.1 cu121 and what it imports, then the two compiled pieces that need it: flash-attn 2.8.3 (prebuilt) and
# pytorch3d @ 75ebeea (upstream requirements.p3d.txt), each in its own layer
RUN uv pip install --python /opt/sam3d/bin/python --no-deps --index-strategy unsafe-best-match \
        --extra-index-url https://download.pytorch.org/whl/cu121 \
        filelock==3.32.3 fsspec==2025.12.0 Jinja2==3.1.6 MarkupSafe==3.0.3 mpmath==1.3.0 \
        networkx==3.6.1 ninja==1.13.2 numpy==1.26.4 nvidia-cublas-cu12==12.1.3.1 nvidia-cuda-cupti-cu12==12.1.105 \
        nvidia-cuda-nvcc-cu12==12.1.105 nvidia-cuda-nvrtc-cu12==12.1.105 nvidia-cuda-runtime-cu12==12.1.105 nvidia-cudnn-cu12==9.1.0.70 nvidia-cufft-cu12==11.0.2.54 \
        nvidia-curand-cu12==10.3.2.106 nvidia-cusolver-cu12==11.4.5.107 nvidia-cusparse-cu12==12.1.0.106 nvidia-nccl-cu12==2.21.5 nvidia-nvjitlink-cu12==12.9.86 \
        nvidia-nvtx-cu12==12.1.105 pillow==11.3.0 sympy==1.13.1 torch==2.5.1+cu121 torchaudio==2.5.1+cu121 \
        torchvision==0.20.1+cu121 triton==3.1.0 typing_extensions==4.16.0
RUN uv pip install --python /opt/sam3d/bin/python --no-deps \
        "flash_attn @ https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
RUN TORCH_CUDA_ARCH_LIST="8.0;9.0" FORCE_CUDA=1 MAX_JOBS=8 CC=gcc CXX=g++ /opt/sam3d/bin/pip install --no-build-isolation --no-deps \
        "pytorch3d @ git+https://github.com/facebookresearch/pytorch3d.git@75ebeeaea0908c5527e7b1e305fbc7681382db47"
# everything else of the research image's freeze (minus the packages listed at the top); kaolin from NVIDIA's wheel index
RUN uv pip install --python /opt/sam3d/bin/python --no-deps --index-strategy unsafe-best-match \
        --extra-index-url https://download.pytorch.org/whl/cu121 \
        --find-links https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html \
        absl-py==2.5.0 accelerate==1.15.0 addict==2.4.0 aiofiles==24.1.0 aiohappyeyeballs==2.7.1 \
        aiohttp==3.14.3 aiosignal==1.4.0 annotated-doc==0.0.5 annotated-types==0.8.0 antlr4-python3-runtime==4.9.3 \
        anyio==4.15.1 argon2-cffi==25.1.0 argon2-cffi-bindings==26.1.0 arrow==1.4.0 astor==0.8.1 \
        asttokens==3.0.2 async-lru==2.3.0 async-timeout==4.0.3 attrs==23.2.0 audioread==3.1.0 \
        auto_gptq==0.7.1 autoflake==2.3.1 av==12.0.0 azure-core==1.41.0 azure-identity==1.25.3 \
        azure-storage-blob==12.30.3 azure-storage-file-datalake==12.25.0 babel==2.18.0 bcrypt==5.0.0 beautifulsoup4==4.15.0 \
        bitsandbytes==0.43.0 black==24.3.0 bleach==6.4.0 blinker==1.9.0 boto3==1.43.102 \
        botocore==1.43.102 braceexpand==0.1.7 brotli==1.2.0 calmsize==0.1.3 ccimport==0.4.4 \
        certifi==2024.8.30 cffi==2.1.1 charset-normalizer==3.5.1 circuitbreaker==2.1.3 click==8.5.0 \
        cloudpickle==3.1.2 colorama==0.4.6 coloredlogs==15.0.1 comm==0.2.3 conda-pack==0.7.1 \
        ConfigArgParse==1.7.7 contourpy==1.3.3 cramjam==2.12.1 crc32c==2.8 crcmod==1.7 \
        cryptography==50.0.1 cuda-python==12.1.0 cumm-cu121==0.7.11 cycler==0.12.1 cyclopts==5.0.0 \
        Cython==3.3.0 dash==4.4.1 dataclasses==0.6 dataclasses-json==0.6.7 datasets==5.0.1 \
        debugpy==1.8.22 decorator==5.3.1 decord==0.6.0 defusedxml==0.7.1 Deprecated==1.3.1 \
        deprecation==2.1.0 dill==0.4.1 dnspython==2.8.0 docker==7.2.0 docopt==0.6.2 \
        docstring_parser==0.18.0 easydict==1.13 einops==0.8.2 einops-exts==0.0.4 entrypoints==0.4 \
        exceptiongroup==1.2.0 executing==2.2.1 fastapi==0.141.1 fastavro==1.9.4 fasteners==0.19 \
        fastjsonschema==2.22.2 ffmpy==1.0.0 fire==0.7.1 flake8==7.0.0 Flask==3.0.3 \
        fonttools==4.66.0 fqdn==1.5.1 freetype-py==2.5.1 frozenlist==1.4.1 ftfy==6.2.0 \
        fvcore==0.1.5.post20221221 gdown==5.2.0 gekko==1.3.2 gitdb==4.0.12 GitPython==3.1.62 \
        glcontext==3.0.0 google-api-core==2.33.0 google-auth==2.58.1 google-cloud-core==2.7.0 google-cloud-storage==2.10.0 \
        google-crc32c==1.9.0 google-pasta==0.2.0 google-resumable-media==2.10.2 googleapis-common-protos==1.75.0 gradio==5.49.0 \
        gradio_client==1.13.3 groovy==0.1.2 grpcio==1.84.0 grpclib==0.4.7 h11==0.16.0 \
        h2==4.1.0 h5py==3.12.1 hatch-requirements-txt==0.4.1 hatchling==1.32.4 hdfs==2.7.3 \
        hf-xet==1.6.0 hpack==4.0.0 httpcore==1.0.9 httplib2==0.22.0 httpx==0.28.1 \
        huggingface_hub==0.36.2 humanfriendly==10.0 hydra-core==1.3.2 hydra-submitit-launcher==1.2.0 hyperframe==6.0.1 \
        idna==3.10 ImageIO==2.37.4 imath==0.0.2 importlib-metadata==6.11.0 iniconfig==2.3.0 \
        iopath==0.1.10 ipycanvas==0.14.3 ipyevents==2.0.4 ipykernel==6.29.5 ipython==9.17.1 \
        ipython_pygments_lexers==1.1.1 ipywidgets==8.1.9 isodate==0.7.2 isoduration==20.11.0 itsdangerous==2.2.0 \
        janus==2.0.0 jaxtyping==0.3.11 jedi==0.20.0 jmespath==1.1.0 joblib==1.6.0 \
        json5==0.15.0 jsonlines==4.0.0 jsonpickle==3.0.4 jsonpointer==2.4 jsonschema==4.22.0 \
        jsonschema-specifications==2025.9.1 jupyter==1.1.1 jupyter-console==6.6.3 jupyter-events==0.12.1 jupyter-lsp==2.3.1 \
        jupyter_builder==1.2.3 jupyter_client==7.4.9 jupyter_core==5.9.1 jupyter_server==2.21.1 jupyter_server_terminals==0.5.4 \
        jupyterlab==4.6.4 jupyterlab_pygments==0.3.0 jupyterlab_server==2.28.1 jupyterlab_widgets==3.0.17 kaolin==0.17.0 \
        kiwisolver==1.5.1 lark==1.3.1 lazy-loader==0.6 libcst==1.9.0 librosa==0.10.1 \
        lightning==2.3.3 lightning-utilities==0.15.3 llvmlite==0.49.0 loguru==0.7.2 Mako==1.4.3 \
        Markdown==3.11 markdown-it-py==4.2.0 marshmallow==3.26.2 matplotlib==3.11.2 matplotlib-inline==0.2.2 \
        mccabe==0.7.0 mdurl==0.1.2 mistune==3.3.4 mock==4.0.3 moderngl==5.12.0 \
        moreorless==0.6.0 mosaicml-streaming==0.7.5 msal==1.39.0 msal-extensions==1.3.1 msgpack==1.2.2 \
        multidict==6.1.0 multiprocess==0.70.19 mypy_extensions==1.1.0 narwhals==2.26.0 nbclient==0.11.0 \
        nbconvert==7.17.1 nbformat==5.11.1 nest-asyncio==1.6.0 notebook==7.6.3 notebook_shim==0.2.4 \
        numba==0.67.0 nvidia-ml-py==13.615.71 objsize==0.7.0 oci==2.187.0 omegaconf==2.3.0 \
        open3d==0.18.0 opencv-python==4.9.0.80 OpenEXR==3.3.3 optimum==1.18.1 optree==0.14.1 \
        orjson==3.10.0 overrides==7.7.0 packaging==24.2 Panda3D==1.10.16 panda3d-gltf==1.2.1 \
        panda3d-simplepbr==0.13.1 pandas==2.3.3 pandocfilters==1.5.1 paramiko==3.5.1 parso==0.8.7 \
        pathos==0.3.5 pathspec==1.1.1 pccm==0.4.16 peft==0.10.0 pexpect==4.9.0 \
        pip-system-certs==4.0 platformdirs==4.11.13 plotly==7.1.0 pluggy==1.6.0 polyscope==2.3.0 \
        pooch==1.9.0 portalocker==4.4.0 pox==0.3.7 ppft==1.7.8 prometheus_client==0.26.0 \
        prompt_toolkit==3.0.53 propcache==0.5.4 proto-plus==1.28.2 protobuf==5.29.6 psutil==7.2.2 \
        ptyprocess==0.7.0 pure_eval==0.2.4 pyarrow==25.0.1 pyasn1==0.6.4 pyasn1_modules==0.4.2 \
        pybind11==3.1.0 pycocotools==2.0.7 pycodestyle==2.11.1 pycparser==3.0 pydantic==2.11.10 \
        pydantic_core==2.33.2 pydot==1.4.2 pydub==0.25.1 pyflakes==3.2.0 pyglet==2.1.16 \
        pygltflib==1.16.5 Pygments==2.21.0 PyJWT==2.15.0 pymongo==4.6.3 PyNaCl==1.6.2 \
        pynvml==13.0.1 PyOpenGL==3.1.0 pyOpenSSL==26.4.0 pyparsing==3.3.3 pyquaternion==0.9.9 \
        pyrender==0.1.45 PySocks==1.7.1 pytest==8.1.1 python-dateutil==2.9.0.post0 python-dotenv==1.2.3 \
        python-json-logger==4.2.0 python-multipart==0.0.32 python-pycg==0.9.2 python-snappy==0.7.3 pytorch-lightning==2.6.6 \
        pytz==2026.4 pyvista==0.49.0 pyvista-validation==0.2.2 PyYAML==6.0.3 pyzmq==27.2.0 \
        randomname==0.2.1 referencing==0.37.0 regex==2026.9.10 requests==2.34.2 retrying==1.4.2 \
        rfc3339-validator==0.1.4 rfc3986-validator==0.1.1 rich==14.3.4 rich-rst==2.1.0 roma==1.5.1 \
        rootutils==1.0.7 rouge==1.0.1 rpds-py==2026.6.3 Rtree==1.3.0 ruff==0.16.9 \
        s3transfer==0.19.2 safehttpx==0.1.7 safetensors==0.8.0 sagemaker==2.242.0 sagemaker-core==1.0.78 \
        schema==0.7.8 scikit-image==0.23.1 scikit-learn==1.9.1 scipy==1.17.1 scooby==0.12.0 \
        screeninfo==0.8.1 seaborn==0.13.2 semantic-version==2.10.0 Send2Trash==2.1.0 sentence-transformers==2.6.1 \
        sentencepiece==0.2.2 sentry-sdk==2.70.0 setproctitle==1.3.7 shellingham==1.5.4 simplejson==3.19.2 \
        six==1.17.0 smdebug-rulesconfig==1.0.1 smmap==5.0.3 soundfile==0.14.0 soupsieve==2.10 \
        soxr==1.1.0 spconv-cu121==2.3.8 stack-data==0.6.3 starlette==0.52.1 stdlibs==2026.9.3 \
        submitit==1.5.4 tabulate==0.10.0 tblib==3.2.2 tensorboard==2.16.2 tensorboard-data-server==0.7.2 \
        termcolor==3.3.0 terminado==0.18.1 texttable==1.7.0 threadpoolctl==3.7.0 tifffile==2026.3.3 \
        timm==0.9.16 tinycss2==1.5.1 tokenizers==0.15.2 toml==0.10.2 tomli==2.0.1 \
        tomlkit==0.13.3 torchmetrics==1.9.0 tornado==6.5.10 tqdm==4.70.1 trailrunner==1.4.0 \
        traitlets==5.16.1 transformers==4.39.3 trimesh==5.1.0 trove-classifiers==2026.9.21.13 typer==0.27.2 \
        typing-inspect==0.9.0 typing-inspection==0.4.4 tzdata==2026.4 uri-template==1.3.0 urllib3==2.8.0 \
        usd-core==26.8 usort==1.0.8.post1 uv==0.12.19 uvicorn==0.54.0 vtk==9.7.0 \
        wadler_lindig==0.1.7 wandb==0.20.0 warp-lang==1.17.0 wcwidth==0.2.14 webcolors==1.13 \
        webdataset==0.2.86 webencodings==0.6.1 websocket-client==1.9.2 websockets==15.0.1 Werkzeug==3.0.6 \
        widgetsnbextension==4.0.16 wrapt==2.4.1 xatlas==0.0.9 xformers==0.0.28.post3 xxhash==3.8.1 \
        yacs==0.1.8 yarl==1.25.1 zipp==4.1.0 zstandard==0.25.0 zstd==1.5.7.2
# git-pinned: utils3d (MoGe's pin), MoGe (upstream requirements.txt; only its geometry utils are imported, the depth model is
# disabled), then SAM 3D itself with pip so its distribution keeps the git provenance (direct_url.json)
RUN uv pip install --python /opt/sam3d/bin/python --no-deps \
        "utils3d @ git+https://github.com/EasternJournalist/utils3d.git@3913c65d81e05e47b9f367250cf8c0f7462a0900" \
        "moge @ git+https://github.com/microsoft/MoGe.git@a8c37341bc0325ca99b9d57981cc3bb2bd3e255b" \
    && /opt/sam3d/bin/pip install --no-build-isolation --no-deps \
        "sam3d_objects @ git+https://github.com/facebookresearch/sam-3d-objects.git@f91db411c50efee93d8db7aeb323885650f6f722" \
    && python -c "import hydra; assert hydra.__version__ == '1.3.2'" \
    && curl -fsSL https://raw.githubusercontent.com/gleize/hydra/78f00766b5f37672aa7232ebbf01bdd74246bd60/hydra/core/utils.py \
        -o "$(python -c 'import hydra, os; print(os.path.join(os.path.dirname(hydra.__file__), "core", "utils.py"))')"
# DINOv2 code for torch.hub, pinned (weights come from the weights directory, see the top)
ENV TORCH_HOME=/opt/torch-hub
RUN git clone https://github.com/facebookresearch/dinov2.git /opt/torch-hub/hub/facebookresearch_dinov2_main \
    && git -C /opt/torch-hub/hub/facebookresearch_dinov2_main checkout --detach 7764ea0f912e53c92e82eb78a2a1631e92725fc8 \
    && echo facebookresearch_dinov2 > /opt/torch-hub/hub/trusted_list \
    && test ! -e /opt/torch-hub/hub/checkpoints
# the reviewed mesh-only patch, applied once to the installed source, with its receipt
COPY workcell/scripts/prepare_sam3d_mesh_source.py /workcell/scripts/prepare_sam3d_mesh_source.py
COPY workcell/modal_apps/sam3d_mesh_only.patch /workcell/modal_apps/sam3d_mesh_only.patch
RUN python /workcell/scripts/prepare_sam3d_mesh_source.py
# no modal client: scripts/onprem/run_stage.py imports modal_apps/sam3d_research.py against its stub (scripts/onprem/modal_stub)
RUN LIDRA_SKIP_INIT=true python -c "import pytorch3d, kaolin, flash_attn, spconv, xformers, moge, utils3d; import sam3d_objects.pipeline.inference_pipeline_pointmap; print('SAM 3D import ready')"
COPY workcell/modal_apps/sam3d_research.py /workcell/modal_apps/sam3d_research.py
COPY workcell/scripts/onprem /workcell/scripts/onprem
ENV LIDRA_SKIP_INIT=true HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /workcell
CMD ["python", "scripts/onprem/run_stage.py", "--self-test"]
