#!/usr/bin/env bash
# Assemble the minimal docker build context of docker/*.Dockerfile (the one modal_apps/onprem_image_proof.py builds from):
#   SRC/workcell/{docker,scripts,modal_apps,configs}  from this repository
#   SRC/serving/{scripts,modal_apps,ehs_spatial}   from panoptes-serving
#   SRC/.dockerignore                              this repository's .dockerignore (same whitelist)
# code only: no outputs/, .venv/, node_modules/, __pycache__, .git (about 8 MB instead of the 24 GB serving checkout).
#   scripts/onprem/stage_context.sh SRC [SERVING_DIR]        SERVING_DIR defaults to $PANOPTES_SERVING
#   docker build -f SRC/workcell/docker/workcell-cpu.Dockerfile -t panoptes-workcell-cpu SRC
set -euo pipefail
here=$(cd "$(dirname "$0")/../.." && pwd)
out=${1:?usage: stage_context.sh SRC [SERVING_DIR]}
serving=${2:-${PANOPTES_SERVING:?pass SERVING_DIR or set PANOPTES_SERVING to the panoptes-serving checkout}}
if [ -d "$out" ] && [ -n "$(ls -A "$out")" ]; then echo "stage_context: $out is not empty" >&2; exit 1; fi
mkdir -p "$out/workcell" "$out/serving"
copy() {  # copy DIR/NAME to DEST/NAME without caches, outputs or environments (tar: same flags on GNU and BSD)
  tar -C "$1" --exclude=__pycache__ --exclude='*.pyc' --exclude=node_modules --exclude=outputs --exclude=.venv \
      --exclude=.DS_Store -cf - "$2" | tar -C "$3" -xf -
}
for d in docker scripts modal_apps configs; do copy "$here" "$d" "$out/workcell"; done
for d in scripts modal_apps ehs_spatial; do copy "$serving" "$d" "$out/serving"; done
cp "$here/.dockerignore" "$out/.dockerignore"
du -sh "$out" >&2
