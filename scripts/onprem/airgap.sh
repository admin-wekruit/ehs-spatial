#!/usr/bin/env bash
# Air-gapped install of the on-prem images: no registry and no internet on the customer server.
# Connected build machine, after docker build of docker/*.Dockerfile:
#   airgap.sh save OUT [IMAGE...]         docker save | gzip -> OUT/IMAGE.tar.gz, OUT/SHA256SUMS
#   airgap.sh wheelhouse OUT [IMAGE...]   optional: every venv of each image as pinned wheels -> OUT/IMAGE/VENV/*.whl
# Air-gapped server (carry OUT and the weights directory written by fetch_weights.py):
#   airgap.sh load OUT                    sha256sum -c SHA256SUMS, then gunzip | docker load of the tarballs it lists (only those)
#   airgap.sh install-wheels DIR PYTHON   pip install --no-index --no-deps DIR/*.whl into PYTHON's environment
# Inside an image (what wheelhouse runs): airgap.sh venv-wheels OUT [PYTHON...]   (default /opt/*/bin/python)
#   airgap.sh self-test SCRATCH           load: only listed + checked tarballs reach a stand-in docker (needs sha256sum, gzip)
# Default images: panoptes-workcell-cpu panoptes-workcell-gpu panoptes-recgen. Wheels do not cover the git checkouts
# (/vendor/pi3, /opt/recgen, /opt/dinov2): docker cp them out of the image, or clone them at the commits in the Dockerfiles.
set -euo pipefail
usage() { sed -n '2,12p' "$0" >&2; exit 2; }
[ $# -ge 2 ] || usage
cmd=$1; out=$2; shift 2
images() { if [ $# -gt 0 ]; then echo "$@"; else echo panoptes-workcell-cpu panoptes-workcell-gpu panoptes-recgen; fi; }
case $cmd in
  save)
    mkdir -p "$out"
    for i in $(images "$@"); do
      docker save "$i" | gzip -1 > "$out/$i.tar.gz.part"; mv "$out/$i.tar.gz.part" "$out/$i.tar.gz"
    done
    (cd "$out" && sha256sum ./*.tar.gz > SHA256SUMS && cat SHA256SUMS) ;;
  load)  # only what SHA256SUMS lists and checks: a stray or unlisted tarball in OUT is never loaded
    names=$(awk '{ n = $2; sub(/^\*/, "", n); sub(/^\.\//, "", n); print n }' "$out/SHA256SUMS")
    [ -n "$names" ] || { echo "airgap: $out/SHA256SUMS lists nothing" >&2; exit 1; }
    for n in $names; do
      case $n in */*|*..*|-*) echo "airgap: refusing path $n in SHA256SUMS" >&2; exit 1 ;; *.tar.gz) ;; *) echo "airgap: $n is not a .tar.gz" >&2; exit 1 ;; esac
    done
    (cd "$out" && sha256sum -c --strict SHA256SUMS)
    for n in $names; do gunzip -c "$out/$n" | docker load; done ;;
  wheelhouse)
    for i in $(images "$@"); do
      mkdir -p "$out/$i"
      docker run --rm -e HOME=/tmp -v "$(cd "$out/$i" && pwd):/wheelhouse" "$i" bash /workcell/scripts/onprem/airgap.sh venv-wheels /wheelhouse
    done ;;
  venv-wheels)
    [ $# -gt 0 ] || set -- /opt/*/bin/python
    for py in "$@"; do
      [ -x "$py" ] || continue
      dir="$out/$(basename "$(dirname "$(dirname "$py")")")"; mkdir -p "$dir"
      "$py" -m pip --version >/dev/null 2>&1 || uv pip install --python "$py" pip >/dev/null  # uv venvs have no pip
      "$py" -m pip freeze --all --exclude-editable > "$dir/requirements.txt"
      cu=$(grep -o '+cu[0-9]*' "$dir/requirements.txt" | head -1 || true)  # local torch builds: their PyTorch index only
      "$py" -m pip wheel --no-deps -q -w "$dir" -r "$dir/requirements.txt" ${cu:+--extra-index-url "https://download.pytorch.org/whl/${cu#+}"}
      echo "$dir: $(ls "$dir"/*.whl | wc -l | tr -d ' ') wheels for $(grep -vc '^#' "$dir/requirements.txt") pins"
    done
    if [ "$(id -u)" = 0 ]; then chown -R --reference="$out" "$out" 2>/dev/null || true; fi ;;
  install-wheels)
    [ $# -eq 1 ] || usage
    "$1" -m pip install --no-index --no-deps "$out"/*.whl ;;
  self-test)
    t=$(mktemp -d "$out/airgap-test.XXXXXX"); trap 'rm -rf "$t"' EXIT
    mkdir -p "$t/bin" "$t/out"
    printf '#!/bin/sh\n[ "$1" = load ] && cat >> "%s/loaded"\n' "$t" > "$t/bin/docker"; chmod +x "$t/bin/docker"
    echo listed | gzip > "$t/out/a.tar.gz"; echo stray | gzip > "$t/out/b.tar.gz"
    (cd "$t/out" && sha256sum ./a.tar.gz > SHA256SUMS)
    PATH="$t/bin:$PATH" bash "$0" load "$t/out" >/dev/null
    [ "$(cat "$t/loaded")" = listed ] || { echo "airgap self-test: loaded $(cat "$t/loaded")" >&2; exit 1; }
    rm "$t/loaded"; echo tampered | gzip > "$t/out/a.tar.gz"
    if PATH="$t/bin:$PATH" bash "$0" load "$t/out" >/dev/null 2>&1 || [ -e "$t/loaded" ]; then echo "airgap self-test: tampered tarball loaded" >&2; exit 1; fi
    echo "0000  ../escape.tar.gz" > "$t/out/SHA256SUMS"
    if PATH="$t/bin:$PATH" bash "$0" load "$t/out" >/dev/null 2>&1 || [ -e "$t/loaded" ]; then echo "airgap self-test: path escape accepted" >&2; exit 1; fi
    echo "airgap self-test passed: load reads only SHA256SUMS-listed tarballs, refuses a changed one and a path outside OUT" ;;
  *) usage ;;
esac
