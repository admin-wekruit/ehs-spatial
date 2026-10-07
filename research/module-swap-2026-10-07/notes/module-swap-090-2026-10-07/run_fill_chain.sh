#!/bin/zsh
# After completion_ab.py --stage sam3d on the filled geometry: the ORIGINAL downstream stages, unchanged, for the 5th version
# mvs-fill+sam3d (same commands as the other three variants; see README "文件").
#   S4 selection: completion_ab assemble (CPU Modal) -> compare.py uniform rule -> swap_generation (generation/ contract) + pin_run
#   S5 assemble_scene (CPU Modal) -> S6 build_capture_report -> S7 local platform import/export (September DB, port 55432, briefly)
#   S8-S14 run_stages (original check modules via workcell_layer_trial) -> Tier-1 / Tier-2 tables
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
V=mvs-fill-sam3d
FILLX=$SP/checks/bbab-export-090-mvs-fill
AB=$SP/swap-runs/mvs-fill-ab
RUN=$SP/swap-runs/$V
TITLE="090 module swap: MVS + MoGe-3 in-mask fill + SAM 3D (local, unpublished)"
MODAL_RUN=${MODAL_RUN:-$M run}   # on-prem: "python $WT/scripts/onprem/run_stage.py --weights DIR"
filt() { grep -viE "capabilit|token|secret|password|postgres://"; }
step() { echo; echo "===== $1  $(date '+%H:%M:%S')"; }

step "assemble candidates (completion_ab --stage assemble, CPU Modal)"
cd $RN/completion-licence-ab-2026-10-05
test -d $AB/sam3d && test -d $AB/sam3d-frame_0003
if [ ! -f $AB/assembly/sam3d-frame_0003/comparisons.json ]; then
  AB_CELL=090 AB_RUN=$FILLX AB_OUT=$AB ${=MODAL_RUN} completion_ab.py --stage assemble --variants sam3d,sam3d-frame_0001,sam3d-frame_0002,sam3d-frame_0003 2>&1 | filt | tail -6
fi
ls $AB/assembly

step "uniform selection + per-object sheets (compare.py, no box gates)"
mkdir -p $HERE/cmp-mvs-fill
cd $RN/completion-ab-090-2026-10-06
if [ ! -f $HERE/cmp-mvs-fill/results.json ]; then
  CMP_NOTES=$HERE/cmp-mvs-fill AB_RUN=$FILLX AB_OUT=$AB nice $PY compare.py left_light_curtain right_light_curtain left_fence right_fence left_post right_post robot cart guard 2>&1 | tail -14
fi

step "materialise the run directory (swap_generation + pin_run)"
cd $HERE
if [ ! -d $RUN/generation ]; then nice $PY swap_generation.py $V 2>&1 | tail -4; fi
BEFORE=$(shasum -a 256 $RUN/manifest.json | cut -c1-64)
nice $PY pin_run.py $RUN
if [ "$(shasum -a 256 $RUN/manifest.json | cut -c1-64)" != "$BEFORE" ] && [ -d $RUN/public ]; then
  echo "manifest re-pinned: rebuilding public/ (scene.json pins the manifest sha)"; rm -rf "${RUN:?}/public"
fi

step "assembly (modal_apps/assemble_scene.py, CPU Modal)"
cd $SERV
if [ ! -f $RUN/result/comparisons.json ]; then ${=MODAL_RUN} modal_apps/assemble_scene.py --run $RUN 2>&1 | filt | tail -4; fi
test -f $RUN/result/comparisons.json

step "report (build_capture_report.py)"
if [ ! -f $RUN/public/scene.json ]; then $PY scripts/research/build_capture_report.py --run $RUN --label "$TITLE" 2>&1 | tail -4; fi
test -f $RUN/public/scene.json

step "local platform import + export (September DB, briefly)"
cd $PLAT
if [ ! -f .platform/swap-20261007/$V/result.json ]; then
  LC_ALL=en_US.UTF-8 $PG/pg_ctl -D ${PGDATA_DIR:-/opt/homebrew/var/postgresql@16} -o "-p 55432 -c listen_addresses=127.0.0.1" -l ${SWAP_SCRATCH:-/private/tmp/claude-501}/pg55432.log start 2>&1 | filt | tail -1 || true
  for i in $(seq 1 40); do $PG/pg_isready -h 127.0.0.1 -p 55432 -q && break; $PY -c "import time; time.sleep(1)"; done
  $PG/pg_isready -h 127.0.0.1 -p 55432
  set +e
  .venv/bin/python .platform/publish-swap-20261007.py $V $RUN $SP/mvs090/estop-scale2.json "$TITLE" 2>&1 | filt | tail -3
  RC=${pipestatus[1]}
  set -e
  LC_ALL=en_US.UTF-8 $PG/pg_ctl -D ${PGDATA_DIR:-/opt/homebrew/var/postgresql@16} stop -m fast 2>&1 | filt | tail -1 || true
  for i in $(seq 1 40); do $PG/pg_isready -h 127.0.0.1 -p 55432 -q || break; $PY -c "import time; time.sleep(1)"; done
  test $RC -eq 0
fi
test -f .platform/swap-20261007/$V/result.json

step "serve the export + original checks (run_stages, Modal CPU)"
cd $HERE
$PY serve_export.py $V 2>&1 | tail -2
$PY run_stages.py $V 2>&1 | filt | tail -14

step "tables"
nice $PY compare_json.py > /dev/null && nice $PY compare_layers.py > /dev/null
grep -c "mvs-fill+sam3d" table.md layers-table.md
echo "CHAIN DONE $(date '+%H:%M:%S')"
