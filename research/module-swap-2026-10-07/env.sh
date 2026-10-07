# Environment of the module-swap pipeline (source this; every script reads these, defaults = the original machine).
#   source research/module-swap-2026-10-07/env.sh            # from the ehs-spatial checkout
# One checkout: ehs-spatial (platform + workcell checks + on-prem kit + model services + this pipeline), branch main.
export PANOPTES_SERVING="${PANOPTES_SERVING:-$(cd "$(dirname "${BASH_SOURCE[0]:-${(%):-%x}}")/../.." && pwd)}"
export PANOPTES_WORKCELL="${PANOPTES_WORKCELL:-$PANOPTES_SERVING}   # one repository: platform, checks, services and the pipeline live together"
export PANOPTES_PLATFORM="${PANOPTES_PLATFORM:-$PANOPTES_WORKCELL}"
export SWAP_ROOT="$PANOPTES_SERVING/research/module-swap-2026-10-07"
export SWAP_NOTES="$SWAP_ROOT/notes"
export SWAP_SCRATCH="${SWAP_SCRATCH:-$SWAP_ROOT/data}"          # the frozen inputs live here; outputs go next to them
export PANOPTES_RUNS="${PANOPTES_RUNS:-$SWAP_SCRATCH/runs}"      # lucida-replica-01 (090), bor1-030-01 (030): photos, masks, manifest
export PANOPTES_PAGES="${PANOPTES_PAGES:-$SWAP_SCRATCH/measurement-layer}"   # where the report layers are written
export PY="${PY:-$PANOPTES_SERVING/.venv/bin/python}"
export PG="${PG:-/usr/lib/postgresql/16/bin}"                      # pg_ctl / pg_isready of the platform database
export PGDATA_DIR="${PGDATA_DIR:-$SWAP_SCRATCH/pgdata}"
export WEIGHTS="${WEIGHTS:-$SWAP_SCRATCH/weights}"                 # scripts/onprem/fetch_weights*.py --cache $WEIGHTS
# GPU / compute backend. Default = the on-prem runner: every Modal app runs in this process on this machine's GPU, no Modal.
# (Original machine: MODAL_RUN="$PANOPTES_PLATFORM/.venv/bin/modal run".)
export MODAL_RUN="${MODAL_RUN:-$PY $PANOPTES_WORKCELL/scripts/onprem/run_stage.py --weights $WEIGHTS}"
export M="${M:-$PANOPTES_PLATFORM/.venv/bin/modal}"; export MS="${MS:-$M}"
export PANOPTES_ONPREM=1 HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
mkdir -p "$SWAP_SCRATCH/swap-runs" "$PANOPTES_PAGES"
