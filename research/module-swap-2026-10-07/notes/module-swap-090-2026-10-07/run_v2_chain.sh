#!/bin/zsh
# Tier-2 for the assembly-v2 (floor-contact hinge) run copies: the ORIGINAL downstream stages, unchanged, per version.
#   swap-runs/v2/<name>/result (assemble_scene v2, already run) -> pin_run -> build_capture_report -> local platform import/export
#   (variant v2-<name>) -> serve_export -> run_stages -> compare_json / compare_layers
#   ./run_v2_chain.sh NAME [NAME ...]     NAME in pi3x-recgen pi3x-sam3d mvs-recgen mvs-sam3d mvs-fill-sam3d
set -e
SP=${SWAP_SCRATCH:-/private/tmp/claude-501/-Users-adam-Desktop-panoptes-public/1fd9a1db-e580-4bfc-8110-119a1cc38a99/scratchpad}
PY=${PY:-/Users/adam/Desktop/panoptes-public/panoptes-serving/.venv/bin/python}
RN=${SWAP_NOTES:-/Users/adam/Desktop/panoptes-public/research-notes}
HERE=$RN/module-swap-090-2026-10-07
SERV=${PANOPTES_SERVING:-/Users/adam/Desktop/panoptes-public/panoptes-serving}
PLAT=${PANOPTES_PLATFORM:-/Users/adam/Desktop/Tesla/panoptes-platform}
PG=${PG:-/opt/homebrew/opt/postgresql@16/bin}
filt() { grep -viE "capabilit|token|secret|password|postgres://"; }
step() { echo; echo "===== $1  $(date '+%H:%M:%S')"; }

for NAME in "$@"; do
  V=v2-$NAME
  RUN=$SP/swap-runs/v2/$NAME
  case $NAME in
    pi3x-*) SCALE=$PLAT/.platform/estop-scale-20261004/plan.json ;;   # the published Pi3X calibration (same world)
    *)      SCALE=$SP/mvs090/estop-scale2.json ;;                      # the MVS run's own e-stop fit (fill: identical scale)
  esac
  TITLE="090 module swap, assembly v2 (floor contact): $NAME (local, unpublished)"
  test -f $RUN/result/comparisons.json

  step "$NAME: pins + report"
  cd $HERE
  BEFORE=$(shasum -a 256 $RUN/manifest.json | cut -c1-64)
  nice $PY pin_run.py $RUN
  if [ "$(shasum -a 256 $RUN/manifest.json | cut -c1-64)" != "$BEFORE" ] && [ -d $RUN/public ]; then rm -rf "${RUN:?}/public"; fi
  cd $SERV
  if [ ! -f $RUN/public/scene.json ]; then $PY scripts/research/build_capture_report.py --run $RUN --label "$TITLE" 2>&1 | tail -2; fi
  test -f $RUN/public/scene.json

  step "$NAME: local platform import + export (September DB, briefly)"
  cd $PLAT
  if [ ! -f .platform/swap-20261007/$V/result.json ]; then
    LC_ALL=en_US.UTF-8 $PG/pg_ctl -D ${PGDATA_DIR:-/opt/homebrew/var/postgresql@16} -o "-p 55432 -c listen_addresses=127.0.0.1" -l ${SWAP_SCRATCH:-/private/tmp/claude-501}/pg55432.log start 2>&1 | filt | tail -1 || true
    for i in $(seq 1 40); do $PG/pg_isready -h 127.0.0.1 -p 55432 -q && break; $PY -c "import time; time.sleep(1)"; done
    $PG/pg_isready -h 127.0.0.1 -p 55432
    set +e
    .venv/bin/python .platform/publish-swap-20261007.py $V $RUN $SCALE "$TITLE" 2>&1 | filt | tail -3
    RC=${pipestatus[1]}
    set -e
    LC_ALL=en_US.UTF-8 $PG/pg_ctl -D ${PGDATA_DIR:-/opt/homebrew/var/postgresql@16} stop -m fast 2>&1 | filt | tail -1 || true
    for i in $(seq 1 40); do $PG/pg_isready -h 127.0.0.1 -p 55432 -q || break; $PY -c "import time; time.sleep(1)"; done
    test $RC -eq 0
  fi
  test -f .platform/swap-20261007/$V/result.json

  step "$NAME: serve + original checks (run_stages, Modal CPU)"
  cd $HERE
  $PY serve_export.py $V 2>&1 | tail -1
  $PY run_stages.py $V 2>&1 | filt | tail -12
done

step "tables"
cd $HERE
nice $PY compare_json.py > /dev/null && nice $PY compare_layers.py > /dev/null
grep -c "组装v2" table.md layers-table.md
echo "V2 CHAIN DONE $(date '+%H:%M:%S')"
