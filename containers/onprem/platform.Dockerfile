# Panoptes platform API (import / edit / publish / export, PostgreSQL + local blobs) and the read-only publication service,
# on-prem: a customer Linux server, no Modal, no SaaS API, no internet at run time. One image, two commands (docker-compose.yml).
# Build context = the panoptes-platform checkout; its root .dockerignore is an allowlist (read by BuildKit and the legacy builder),
# so .platform/ (capabilities, credentials), runs/, outputs/ and .env never enter the build. Internet at build time only.
#   docker build -f containers/onprem/platform.Dockerfile -t panoptes-platform:onprem .
# Pins: base image by digest (3.12.13 = the platform .venv), uv by wheel hash, every Python package by uv.lock (hashes, --frozen).
FROM python:3.12.13-slim-bookworm@sha256:4766d8b510c428e595d74b9cc5bbb2fae8e26316fffb4adc89908d79aacd58a2

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    UV_PYTHON_DOWNLOADS=never UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 UV_PROJECT_ENVIRONMENT=/app/.venv
# open3d needs libgomp1 + libgl1 (the same apt set as modal_apps/publication_site.py)
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 libgl1 && rm -rf /var/lib/apt/lists/*
RUN printf 'uv==0.10.9 --hash=sha256:7c9d6deb30edbc22123be75479f99fb476613eaf38a8034c0e98bba24a344179 --hash=sha256:24b1ce6d626e06c4582946b6af07b08a032fcccd81fe54c3db3ed2d1c63a97dc\n' > /tmp/uv.txt \
    && pip install --require-hashes -r /tmp/uv.txt && rm /tmp/uv.txt

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project && rm -rf /root/.cache/uv
COPY ehs_spatial ehs_spatial
COPY panoptes_worker panoptes_worker
COPY scripts scripts
COPY containers/onprem/serve_publications.py serve_publications.py

# Non-root at run time (compose: user: panoptes). New named volumes inherit these owners.
RUN useradd --system --uid 10001 --home-dir /data panoptes \
    && mkdir -p /data/blobs /data/imports /catalog /publication-http /feedback \
    && chown -R panoptes /data /catalog /publication-http /feedback
# The venv stays off PATH (commands name /app/.venv/bin/... explicitly); the system python is left as the base image ships it.
ENV PYTHONPATH=/app HF_HUB_OFFLINE=1 \
    PANOPTES_BLOB_ROOT=/data/blobs PANOPTES_EXECUTOR_BACKEND=local
EXPOSE 8792 8793
# The platform API migrates its schema at startup (advisory lock). The publication service: serve_publications.py.
CMD ["/app/.venv/bin/uvicorn", "ehs_spatial.platform.runtime:application", "--factory", "--host", "0.0.0.0", "--port", "8792"]
