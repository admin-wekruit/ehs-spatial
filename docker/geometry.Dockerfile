# Panoptes workcell photo pipeline - the licence-clean geometry route (replaces Pi3X), on-prem: CPU only, no CUDA or NVIDIA
# component, no Modal, no SaaS API, no internet at run time.
#   DA3-BASE start (docker/da3.Dockerfile, its own GPU stage) -> THIS IMAGE: RoMa v1 outdoor dense matches -> bundle adjustment
#   with one focal per photo (numpy Schur Levenberg-Marquardt, modal_apps/bundle_adjust.py) -> two-view triangulation of RoMa's
#   dense warp (CERT 0.05, conf = 1 on kept pixels) = scripts/onprem/run_stage.py geometry. RoMa runs on CPU (fp32), no GPU is
#   needed: the whole stage took 158 s for 090 (3 photos, 3 pairs) and 58 s for 030 (2 photos) on 8 CPUs, 2026-10-06.
#   /opt/geometry = docker/geometry-requirements.txt, pip --require-hashes --no-deps (every wheel pinned by SHA-256; torch and
#                   torchvision are the CPU builds). numpy 1.26.4, scipy 1.14.1, OpenCV 4.10.0.84. No pycolmap (GPL-2.0+
#                   SuiteSparse in its wheel), vggt, plyfile or Pi3.
#   /vendor/romatch = Parskatt/RoMa @ 77f8d68 (MIT): the romatch package + LICENSE from the commit archive (SHA-256 pinned), on
#                   sys.path through a .pth file. estimate_pose / estimate_pose_uncalibrated are deleted (their lineage runs
#                   through LGPL-2.1 DenseMatching to Magic Leap code; off the runtime path), and romatch/benchmarks, which calls them.
#   /licences/THIRD_PARTY_NOTICES.md = docker/THIRD_PARTY_NOTICES-geometry.md (every component and its licence).
# Weights are never in the image: scripts/onprem/fetch_weights_geometry.py --cache DIR (internet), or copy the mirror, then mount
# DIR read-only. The stage SHA-256-checks both RoMa files before loading and refuses a mismatch.
# Build context SRC = scripts/onprem/stage_context.sh SRC (this repository as workcell/, panoptes-serving as serving/).
#   docker build -f SRC/workcell/docker/geometry.Dockerfile -t panoptes-workcell-geometry SRC
#   docker run --rm --network none --cpus 8 -v /srv/panoptes-weights:/weights:ro -v $PWD/data:/data panoptes-workcell-geometry \
#       /opt/geometry/bin/python scripts/onprem/run_stage.py geometry --weights /weights --run /data/RUN \
#       --start /data/RUN/geometry-da3-base --output /data/RUN/geometry-route
# Air-gapped servers: build on a connected machine, then scripts/onprem/airgap.sh save / load (docs/workcell-photo/ONPREM.md).
FROM python:3.11.10-slim-bookworm@sha256:840e180ebcc6e5c8efab209c43f5e40fd2af98cb49db5c7103c90539c56bb30e

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

COPY workcell/docker/geometry-requirements.txt /tmp/geometry-requirements.txt
RUN python -m venv /opt/geometry && /opt/geometry/bin/pip install --no-deps -r /tmp/geometry-requirements.txt \
    && rm /tmp/geometry-requirements.txt

ARG ROMA=77f8d68803526dcddfd9b7a46bc76125bdc25f15
ARG ROMA_SHA256=7b5264cd729c93524cb1eeabba4a335896a6d5481c3a7f3eef707203b10cb1e6
# utils.py lines 28-76 = the DenseMatching comment through estimate_pose_uncalibrated (the archive bytes are pinned above)
RUN python -c "import hashlib, io, tarfile, urllib.request; \
d = urllib.request.urlopen('https://codeload.github.com/Parskatt/RoMa/tar.gz/$ROMA', timeout=120).read(); \
h = hashlib.sha256(d).hexdigest(); assert h == '$ROMA_SHA256', 'RoMa archive SHA-256 ' + h + ' is not the pin'; \
tarfile.open(fileobj=io.BytesIO(d)).extractall('/tmp/roma', filter='data')" \
    && mkdir -p /vendor/romatch && mv /tmp/roma/RoMa-$ROMA/romatch /tmp/roma/RoMa-$ROMA/LICENSE /vendor/romatch/ && rm -rf /tmp/roma \
    && cd /vendor/romatch && sed -i '28,76d' romatch/utils/utils.py && sed -i '/estimate_pose/d' romatch/utils/__init__.py \
    && rm -rf romatch/benchmarks && ! grep -rn estimate_pose romatch && grep -q '^def unnormalize_coords' romatch/utils/utils.py \
    && echo /vendor/romatch > /opt/geometry/lib/python3.11/site-packages/romatch.pth \
    && cd / && /opt/geometry/bin/python -c "import importlib.util as u, cv2, kornia, romatch, scipy, torch; \
assert torch.version.cuda is None, torch.version.cuda; \
bad = [m for m in ('pycolmap', 'vggt', 'plyfile', 'pi3', 'triton') if u.find_spec(m)]; assert not bad, bad"

# Only the files the geometry stage uses: an unrelated edit elsewhere does not invalidate the layers above.
COPY workcell/docker/THIRD_PARTY_NOTICES-geometry.md /licences/THIRD_PARTY_NOTICES.md
COPY workcell/scripts/onprem/run_stage.py workcell/scripts/onprem/fetch_weights.py workcell/scripts/onprem/fetch_weights_da3.py \
     workcell/scripts/onprem/fetch_weights_geometry.py /workcell/scripts/onprem/
COPY workcell/scripts/onprem/modal_stub /workcell/scripts/onprem/modal_stub
COPY workcell/modal_apps/geometry_clean_ab.py workcell/modal_apps/bundle_adjust.py workcell/modal_apps/moge3_app.py /workcell/modal_apps/
ENV HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
WORKDIR /workcell
CMD ["/opt/geometry/bin/python", "scripts/onprem/run_stage.py", "--self-test"]
