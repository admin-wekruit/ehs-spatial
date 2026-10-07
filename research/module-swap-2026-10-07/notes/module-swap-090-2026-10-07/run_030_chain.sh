#!/bin/zsh
# Cell 030 through the same module-swap pipeline (assembly v2 only): three versions
#   pi3x-recgen      the published September run bor1-030-01 (Pi3X + RecGen) re-assembled with the floor-contact hinge
#   pi3x-sam3d       SAM 3D candidates on the Pi3X run (swap-runs/030/pi3x-ab), uniform selection, generation/ contract
#   mvs-fill-sam3d   SAM 3D candidates on the filled MVS export (checks/bbab-export-030-mvs-fill, swap-runs/030/mvs-fill-ab)
# then S6-S14 unchanged (report -> local import/export as variant 030-v2-<name> -> original checks -> layer).
#   ./run_030_chain.sh [NAME ...]
set -e
SP=${SWAP_SCRATCH:-/private/tmp/claude-501/-Users-adam-Desktop-panoptes-public/1fd9a1db-e580-4bfc-8110-119a1cc38a99/scratchpad}
PY=${PY:-/Users/adam/Desktop/panoptes-public/panoptes-serving/.venv/bin/python}
MS=${MS:-/Users/adam/Desktop/panoptes-public/panoptes-serving/.venv/bin/modal}
M=${M:-/Users/adam/Desktop/Tesla/panoptes-platform/.venv/bin/modal}
RN=${SWAP_NOTES:-/Users/adam/Desktop/panoptes-public/research-notes}
HERE=$RN/module-swap-090-2026-10-07
SERV=${PANOPTES_SERVING:-/Users/adam/Desktop/panoptes-public/panoptes-serving}
PLAT=${PANOPTES_PLATFORM:-/Users/adam/Desktop/Tesla/panoptes-platform}
PG=${PG:-/opt/homebrew/opt/postgresql@16/bin}
BOR=$SERV/outputs/candidate-evaluation/bor1-030-01
FILLX=$SP/checks/bbab-export-030-mvs-fill
OBJS="cart guard left_light_curtain left_post right_fence right_light_curtain right_post robot"
MODAL_RUN=${MODAL_RUN:-$M run}   # on-prem: "python $WT/scripts/onprem/run_stage.py --weights DIR"
filt() { grep -viE "capabilit|token|secret|password|postgres://"; }
step() { echo; echo "===== $1  $(date '+%H:%M:%S')"; }
NAMES=${@:-pi3x-recgen pi3x-sam3d mvs-fill-sam3d}

for NAME in ${=NAMES}; do
  V=030-v2-$NAME
  RUN=$SP/swap-runs/030/$NAME
  case $NAME in
    pi3x-*) SCALE=$PLAT/.platform/cell030-20261005/estop-scale-v4.json; SRC=$BOR; AB=$SP/swap-runs/030/pi3x-ab; CMP=$HERE/cmp-030-pi3x ;;
    *)      SCALE=$SP/mvs030/estop-scale.json;                           SRC=$FILLX; AB=$SP/swap-runs/030/mvs-fill-ab; CMP=$HERE/cmp-030-mvs-fill ;;
  esac
  TITLE="030 module swap, assembly v2 (floor contact): $NAME (local, unpublished)"

  if [[ $NAME == *sam3d ]]; then
    step "$NAME: assemble candidates (completion_ab --stage assemble, CPU Modal)"
    cd $RN/completion-licence-ab-2026-10-05
    # only the candidate folders that hold meshes (a photo variant with no object has no folder content and breaks the stage)
    VARS=$(cd $AB && for d in sam3d sam3d-frame_0001 sam3d-frame_0002; do [ -n "$(ls $d/*.npz 2>/dev/null)" ] && echo $d; done | paste -sd, -)
    echo "candidate variants: $VARS"
    if [ ! -f $AB/assembly/sam3d/comparisons.json ]; then
      AB_CELL=030 AB_RUN=$SRC AB_OUT=$AB ${=MODAL_RUN} completion_ab.py --stage assemble --variants $VARS 2>&1 | filt | grep -E "estimateUsd|Error|rror:" | tail -3
    fi
    test -f $AB/assembly/sam3d/comparisons.json
    step "$NAME: uniform selection + sheets (compare.py)"
    mkdir -p $CMP; cd $RN/completion-ab-090-2026-10-06
    # CMP_SCALE: the Pi3X run's floor.json has no e-stop scale of its own (the published calibration file has nativeToMeters)
    if [ ! -f $CMP/results.json ]; then CMP_NOTES=$CMP CMP_SCALE=$SCALE AB_CELL=030 AB_RUN=$SRC AB_OUT=$AB nice $PY compare.py ${=OBJS} 2>&1 | grep -vE "findfont" | grep -E "Traceback|Error|rror:|决定|decision" | tail -4; fi
    test -f $CMP/results.json
    step "$NAME: materialise the run directory"
    cd $HERE
    if [ ! -d $RUN/generation ]; then nice $PY swap_generation.py 030/$NAME 2>&1 | tail -3; fi
  else
    step "$NAME: copy of the published run"
    if [ ! -d $RUN ]; then cp -c -R $BOR $RUN; rm -rf "${RUN:?}/result" "${RUN:?}/public"; fi
  fi

  step "$NAME: pins + assembly v2 + report"
  cd $HERE
  BEFORE=$(shasum -a 256 $RUN/manifest.json | cut -c1-64)
  nice $PY pin_run.py $RUN
  if [ "$(shasum -a 256 $RUN/manifest.json | cut -c1-64)" != "$BEFORE" ] && [ -d $RUN/public ]; then rm -rf "${RUN:?}/public"; fi
  cd $SERV
  if [ ! -f $RUN/result/comparisons.json ]; then ${=MODAL_RUN} modal_apps/assemble_scene.py --run $RUN 2>&1 | filt | grep -E "estimateUsd|Error|rror:" | tail -2; fi
  test -f $RUN/result/comparisons.json
  if [ ! -f $RUN/public/scene.json ]; then $PY scripts/research/build_capture_report.py --run $RUN --label "$TITLE" 2>&1 | tail -1; fi
  test -f $RUN/public/scene.json

  step "$NAME: local platform import + export"
  cd $PLAT
  if [ ! -f .platform/swap-20261007/$V/result.json ]; then
    LC_ALL=en_US.UTF-8 $PG/pg_ctl -D ${PGDATA_DIR:-/opt/homebrew/var/postgresql@16} -o "-p 55432 -c listen_addresses=127.0.0.1" -l ${SWAP_SCRATCH:-/private/tmp/claude-501}/pg55432.log start 2>&1 | filt | tail -1 || true
    for i in $(seq 1 40); do $PG/pg_isready -h 127.0.0.1 -p 55432 -q && break; $PY -c "import time; time.sleep(1)"; done
    $PG/pg_isready -h 127.0.0.1 -p 55432
    set +e
    .venv/bin/python .platform/publish-swap-20261007.py $V $RUN $SCALE "$TITLE" 2>&1 | filt | tail -2
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
  STAGES_CELL=030 $PY run_stages.py $V shape floor lines plane_stereo transfer clearance lower_edge box_faces obvious_errors plane_facets 2>&1 | filt | tail -12
  $PY build_swap_layer.py $V 2>&1 | tail -1
done
echo "030 CHAIN DONE $(date '+%H:%M:%S')"
