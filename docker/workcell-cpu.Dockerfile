# Panoptes workcell photo pipeline - CPU stages, on-prem (no Modal, no SaaS API, no internet at run time).
#   /opt/checks   = modal_apps/workcell_view_checks.py image (floor / lines / plane_stereo / transfer checks, Python 3.11.10);
#                   also modal_apps/workcell_shape_check.py (its Modal image resolved scipy 1.17.1 instead of 1.14.1: only difference)
#   /opt/assemble = serving modal_apps/assemble_scene.py image (scene assembly; capture freeze/evidence and the public scene
#                   build use the same packages), Python 3.12.6
# Pins = `pip freeze` of the cached Modal image(s) named below, read on 2026-10-05 (modal_apps/onprem_image_proof.py records the
# image's own freeze again at proof time), installed with --no-deps; then the modal client 1.5.4, used only as a library by
# scripts/onprem/run_stage.py (no account, token or network). Weights are never in an image: scripts/onprem/fetch_weights.py.
# Build context SRC = a directory holding this repository as workcell/ and panoptes-serving as serving/.
#   docker build -f SRC/workcell/docker/workcell-cpu.Dockerfile -t panoptes-workcell-cpu SRC
#   docker run --rm --network none panoptes-workcell-cpu                      # self-tests
#   docker run --rm --network none -v $PWD/data:/data panoptes-workcell-cpu /opt/checks/bin/python scripts/onprem/run_stage.py \
#       --offline /data/bundle modal_apps/workcell_view_checks.py --checks floor --view /data/view.json ... --out /data/floor
FROM python:3.11.10-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 UV_PYTHON_INSTALL_DIR=/opt/uv-python
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libgomp1 libx11-6 libglib2.0-0 git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# workcell-view-checks (debian_slim 3.11 + libgl1 libgomp1 libx11-6; numpy<2.3 opencv 4.10.0.84 open3d 0.19.0 matplotlib 3.9.2 trimesh 4.4.9 scipy 1.14.1)
RUN python -m venv /opt/checks && /opt/checks/bin/pip install --no-deps \
        addict==2.4.0 aiohappyeyeballs==2.4.3 aiohttp==3.10.8 aiosignal==1.3.1 annotated-types==0.8.0 \
        asttokens==3.0.2 attrs==24.2.0 blinker==1.9.0 certifi==2024.8.30 charset-normalizer==3.5.2 \
        click==8.5.0 cloudpickle==3.1.2 comm==0.2.3 ConfigArgParse==1.8.0 contourpy==1.3.3 \
        cycler==0.12.1 dash==4.4.1 executing==2.2.1 fastjsonschema==2.22.2 Flask==3.1.3 \
        fonttools==4.66.1 frozenlist==1.4.1 grpclib==0.4.7 h2==4.1.0 hpack==4.0.0 \
        hyperframe==6.0.1 idna==3.10 importlib_metadata==9.0.1 ipython==9.17.1 ipython_pygments_lexers==1.1.1 \
        ipywidgets==8.1.9 itsdangerous==2.2.0 janus==2.0.0 jedi==0.20.0 Jinja2==3.1.6 \
        joblib==1.6.0 jsonschema-specifications==2025.9.1 jsonschema==4.26.0 jupyter_core==5.9.1 jupyterlab_widgets==3.0.17 \
        kiwisolver==1.5.1 MarkupSafe==3.0.4 matplotlib-inline==0.2.2 matplotlib==3.9.2 multidict==6.1.0 \
        narwhals==2.26.0 nbformat==5.11.1 nest-asyncio==1.6.0 numpy==2.2.6 open3d==0.19.0 \
        opencv-python-headless==4.10.0.84 packaging==26.3 pandas==3.0.6 parso==0.8.7 pexpect==4.9.0 \
        pillow==12.3.0 platformdirs==4.12.3 plotly==7.1.0 prompt_toolkit==3.0.53 protobuf==5.29.2 \
        psutil==7.2.2 ptyprocess==0.7.0 pure_eval==0.2.4 pydantic==2.13.5 pydantic_core==2.46.5 \
        Pygments==2.21.0 pyparsing==3.3.3 pyquaternion==0.9.9 python-dateutil==2.9.0.post0 PyYAML==6.0.3 \
        referencing==0.37.0 requests==2.34.2 retrying==1.4.2 rpds-py==2026.9.1 scikit-learn==1.9.1 \
        scipy==1.14.1 setuptools==65.5.1 six==1.17.0 stack-data==0.6.3 threadpoolctl==3.7.0 \
        tqdm==4.70.1 traitlets==5.16.1 trimesh==4.4.9 typing-inspection==0.4.4 typing_extensions==4.16.0 \
        urllib3==2.8.0 wcwidth==0.9.2 Werkzeug==3.1.9 widgetsnbextension==4.0.16 yarl==1.13.1 \
        zipp==4.1.1 \
    && /opt/checks/bin/pip install modal==1.5.4

# assemble-scene (debian_slim 3.12 + libgl1 libgomp1 libglib2.0-0 libx11-6; numpy 2.1.3 opencv 4.10.0.84 open3d 0.19.0 pillow 11.0.0 scipy 1.14.1 trimesh 4.4.9)
RUN pip install uv==0.5.14 && uv python install 3.12.6 && uv venv --python 3.12.6 /opt/assemble \
    && uv pip install --python /opt/assemble/bin/python --no-deps \
        addict==2.4.0 aiohappyeyeballs==2.4.3 aiohttp==3.10.8 aiosignal==1.3.1 annotated-types==0.8.0 \
        asttokens==3.0.2 attrs==24.2.0 blinker==1.9.0 certifi==2024.8.30 charset-normalizer==3.5.2 \
        click==8.5.0 cloudpickle==3.1.2 comm==0.2.3 ConfigArgParse==1.8.0 contourpy==1.4.0 \
        cycler==0.12.1 dash==4.4.1 executing==2.2.1 fastjsonschema==2.22.2 Flask==3.1.3 \
        fonttools==4.66.1 frozenlist==1.4.1 grpclib==0.4.7 h2==4.1.0 hpack==4.0.0 \
        hyperframe==6.0.1 idna==3.10 importlib_metadata==9.0.1 ipython==9.17.1 ipython_pygments_lexers==1.1.1 \
        ipywidgets==8.1.9 itsdangerous==2.2.0 janus==2.0.0 jedi==0.20.0 Jinja2==3.1.6 \
        joblib==1.6.0 jsonschema-specifications==2025.9.1 jsonschema==4.26.0 jupyter_core==5.9.1 jupyterlab_widgets==3.0.17 \
        kiwisolver==1.5.1 MarkupSafe==3.0.4 matplotlib-inline==0.2.2 matplotlib==3.11.2 multidict==6.1.0 \
        narwhals==2.26.0 nbformat==5.11.1 nest-asyncio==1.6.0 numpy==2.1.3 open3d==0.19.0 \
        opencv-python-headless==4.10.0.84 packaging==26.3 pandas==3.0.6 parso==0.8.7 pexpect==4.9.0 \
        pillow==11.0.0 platformdirs==4.12.3 plotly==7.1.0 prompt_toolkit==3.0.53 protobuf==5.29.2 \
        psutil==7.2.2 ptyprocess==0.7.0 pure_eval==0.2.4 pydantic==2.13.5 pydantic_core==2.46.5 \
        Pygments==2.21.0 pyparsing==3.3.3 pyquaternion==0.9.9 python-dateutil==2.9.0.post0 PyYAML==6.0.3 \
        referencing==0.37.0 requests==2.34.2 retrying==1.4.2 rpds-py==2026.9.1 scikit-learn==1.9.1 \
        scipy==1.14.1 setuptools==84.0.0 six==1.17.0 stack-data==0.6.3 threadpoolctl==3.7.0 \
        tqdm==4.70.1 traitlets==5.16.1 trimesh==4.4.9 typing-inspection==0.4.4 typing_extensions==4.16.0 \
        urllib3==2.8.0 wcwidth==0.9.2 Werkzeug==3.1.9 widgetsnbextension==4.0.16 yarl==1.13.1 \
        zipp==4.1.1 \
    && uv pip install --python /opt/assemble/bin/python modal==1.5.4

# Container paths of the Modal images' add_local_* (so every function body runs unchanged), then the repositories.
COPY workcell/scripts/workcell_shape_check.py /check/shape_core.py
COPY workcell/scripts/workcell_checks /check/workcell_checks
COPY workcell/scripts /workcell/scripts
COPY workcell/modal_apps /workcell/modal_apps
COPY serving/scripts /serving/scripts
COPY serving/modal_apps /serving/modal_apps
COPY serving/ehs_spatial /serving/ehs_spatial
ENV HF_HUB_OFFLINE=1
WORKDIR /workcell
CMD ["/opt/checks/bin/python", "scripts/onprem/run_stage.py", "--self-test"]
