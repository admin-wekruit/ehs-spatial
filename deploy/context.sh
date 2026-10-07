#!/usr/bin/env bash
# Assemble the docker build context of deploy/Dockerfile.* : ehs-spatial's own stage_context.sh layout (SRC/workcell = the
# ehs-spatial checkout's docker, scripts, modal_apps, configs; SRC/serving = this repo's scripts, modal_apps, ehs_spatial;
# SRC/.dockerignore = its whitelist), plus this repo's serving/ and deploy/ (which that whitelist does not list).
#   deploy/context.sh SRC [WORKCELL]        WORKCELL = the ehs-spatial checkout, default $PANOPTES_WORKCELL
set -euo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
out=${1:?usage: deploy/context.sh SRC [WORKCELL]}
workcell=${2:-${PANOPTES_WORKCELL:?pass WORKCELL or set PANOPTES_WORKCELL to the ehs-spatial checkout}}
"$workcell/scripts/onprem/stage_context.sh" "$out" "$here"
mkdir -p "$out/serving"
for d in serving deploy; do
  tar -C "$here" --exclude=__pycache__ --exclude='*.pyc' --exclude=.DS_Store -cf - "$d" | tar -C "$out/serving" -xf -
done
printf '!serving/serving\n!serving/deploy\n' >> "$out/.dockerignore"
du -sh "$out" >&2
