#!/bin/zsh
# The whole module-swap pipeline for one cell on this machine (GPU through scripts/onprem/run_stage.py; nothing on Modal):
#   S2 geometry  MVS route (RoMa + DA3-BASE start + numpy LM BA + two-view triangulation) -> fair evaluator -> export
#                -> MoGe-3 in-mask fill -> fair evaluator again -> export -> field-value gate
#   S4 completion SAM 3D Objects candidates from every masked photo -> uniform selection -> generation/ contract
#   S5-S14       assembly v2 (floor-contact hinge) -> report -> platform import/export -> original checks -> layer -> tables
#   source env.sh; ./run_all.sh 090 | 030            (idempotent: finished steps are skipped)
set -e
: ${SWAP_ROOT:?source env.sh first}
CELL=${1:?cell: 090 or 030}
N=$SWAP_NOTES; SP=$SWAP_SCRATCH
step() { echo; echo "===== $CELL $1  $(date '+%H:%M:%S')"; }
case $CELL in
  090) RUNNAME=lucida-replica-01; OBJS="left_light_curtain right_light_curtain left_fence right_fence left_post right_post robot cart guard"; FRAMES=sam3d,sam3d-frame_0001,sam3d-frame_0002,sam3d-frame_0003 ;;
  030) RUNNAME=bor1-030-01; OBJS="cart guard left_light_curtain left_post right_fence right_light_curtain right_post robot"; FRAMES=sam3d,sam3d-frame_0001,sam3d-frame_0002 ;;
esac

step "S2a RoMa matches + DA3-BASE start + MoGe-3 depth (GPU) -> checks/clean-gpu/$CELL"
if [ ! -f $SP/checks/clean-gpu/$CELL/moge-frame_0001.npz ]; then
  cd $N/geometry-backbone-ab-2026-10-06 && ${=MODAL_RUN} prod_route_modal.py --stage route --cells $CELL 2>&1 | tail -3
fi
step "S2b bundle adjustment + two-view triangulation (CPU) -> checks/bbab-geom/$CELL-mvs-da3-base-padded"
cd $N/geometry-backbone-ab-2026-10-06
[ -d $SP/checks/bbab-geom/$CELL-mvs-da3-base-padded/geometry ] || $PY mvs_route.py da3-base 2>&1 | tail -2
step "S2c fair evaluator (floor, e-stop scale, field values) + export"
[ -f $SP/checks/bbab-analyse/$CELL-mvs-da3-base-padded.json ] || ${=MODAL_RUN} backbone_ab_modal.py analyse $CELL-mvs-da3-base-padded 2>&1 | tail -2
[ -d $SP/checks/bbab-export-$CELL-mvs-scipyba ] || { $PY backbone_ab_modal.py export mvs-da3-base $CELL 2>&1 | tail -1; mv $SP/checks/bbab-export-$CELL-mvs-da3-base $SP/checks/bbab-export-$CELL-mvs-scipyba; }
step "S2d MoGe-3 in-mask fill (CPU) + fair evaluator + export + field-value gate"
cd $N/module-swap-090-2026-10-07
[ -d $SP/checks/bbab-geom/$CELL-mvs-fill-padded/geometry ] || FILL_CELL=$CELL $PY fill_geometry.py 2>&1 | tail -3
cd $N/geometry-backbone-ab-2026-10-06
[ -f $SP/checks/bbab-analyse/$CELL-mvs-fill-padded.json ] || ${=MODAL_RUN} backbone_ab_modal.py analyse $CELL-mvs-fill-padded 2>&1 | tail -2
[ -d $SP/checks/bbab-export-$CELL-mvs-fill ] || $PY backbone_ab_modal.py export mvs-fill $CELL 2>&1 | tail -1
cd $N/module-swap-090-2026-10-07 && $PY field_values_fill.py 2>&1 | tail -2
"$PY" - <<'EOF'
import json, os
f = json.load(open('field-values-mvs-fill.json')); a, b = f['mvs-da3-base'], f['mvs-fill']
for k in ('housing090R_cm', 'fence090R_cm', 'housing030L_cm', 'housing030R_cm', 'estopNativeToMeters090', 'estopNativeToMeters030'):
    print(f'  {k:24s} mvs {a.get(k)}  fill {b.get(k)}')
assert b['maeCm4values'] <= 1.56 and b['maxAbsErrCm'] <= 3.0, 'field-value gate failed: the fill changed a measurement'
print('  GATE PASSED (MAE %.2f cm, max %.2f cm)' % (b['maeCm4values'], b['maxAbsErrCm']))
EOF

FILLX=$SP/checks/bbab-export-$CELL-mvs-fill; AB=$SP/swap-runs/$CELL/mvs-fill-ab; V=$CELL-v2-mvs-fill-sam3d; RUN=$SP/swap-runs/$CELL/mvs-fill-sam3d
step "S4a SAM 3D Objects candidates from every masked photo (GPU) -> $AB"
mkdir -p $AB; cd $N/completion-licence-ab-2026-10-05
[ -f $AB/sam3d/record.json ] || AB_CELL=$CELL AB_RUN=$FILLX AB_OUT=$AB AB_FRAME=all ${=MODAL_RUN} completion_ab.py --stage sam3d 2>&1 | tail -2
VARS=$(cd $AB && for d in ${(s:,:)FRAMES}; do [ -n "$(ls $d/*.npz 2>/dev/null)" ] && echo $d; done | paste -sd, -)
step "S4b candidate assembly (CPU) + uniform selection -> cmp-$CELL-mvs-fill"
[ -f $AB/assembly/sam3d/comparisons.json ] || AB_CELL=$CELL AB_RUN=$FILLX AB_OUT=$AB ${=MODAL_RUN} completion_ab.py --stage assemble --variants $VARS 2>&1 | tail -2
CMP=$N/module-swap-090-2026-10-07/cmp-$CELL-mvs-fill; mkdir -p $CMP; cd $N/completion-ab-090-2026-10-06
SCALE=$([ $CELL = 090 ] && echo $SP/mvs090/estop-scale2.json || echo $SP/mvs030/estop-scale.json)
[ -f $CMP/results.json ] || CMP_NOTES=$CMP CMP_SCALE=$SCALE AB_CELL=$CELL AB_RUN=$FILLX AB_OUT=$AB $PY compare.py ${=OBJS} 2>&1 | grep -E "Traceback|rror:" | tail -3
step "S4c generation/ contract + pins"
cd $N/module-swap-090-2026-10-07
[ -d $RUN/generation ] || $PY swap_generation.py $CELL/mvs-fill-sam3d 2>&1 | tail -2
$PY pin_run.py $RUN | tail -1

step "S5 assembly v2 (CPU, floor-contact hinge) + S6 report"
cd $PANOPTES_SERVING
[ -f $RUN/result/comparisons.json ] || ${=MODAL_RUN} modal_apps/assemble_scene.py --run $RUN 2>&1 | grep -E "estimateUsd|rror" | tail -2
[ -f $RUN/public/scene.json ] || $PY scripts/research/build_capture_report.py --run $RUN --label "$CELL: MVS + MoGe-3 fill + SAM 3D, assembly v2" --pages-root $PANOPTES_WORKCELL/web 2>&1 | tail -1

step "S7 platform import + export (Postgres at 127.0.0.1:55432)"
cd $PANOPTES_PLATFORM
if [ ! -f .platform/swap-20261007/$V/result.json ]; then
  $PG/pg_isready -h 127.0.0.1 -p 55432 -q || LC_ALL=en_US.UTF-8 $PG/pg_ctl -D $PGDATA_DIR -o "-p 55432 -c listen_addresses=127.0.0.1" -l $SP/pg55432.log start
  for i in $(seq 1 40); do $PG/pg_isready -h 127.0.0.1 -p 55432 -q && break; $PY -c "import time; time.sleep(1)"; done
  $PY $SWAP_ROOT/platform/publish-swap-20261007.py $V $RUN $SCALE "$CELL module swap: MVS + MoGe-3 fill + SAM 3D, assembly v2" 2>&1 | tail -1
fi

step "S8-S14 original checks + layer + tables"
cd $N/module-swap-090-2026-10-07
$PY serve_export.py $V 2>&1 | tail -1
STAGES_CELL=$CELL $PY run_stages.py $V 2>&1 | tail -11
$PY build_swap_layer.py $V 2>&1 | tail -1
if [ $CELL = 090 ]; then $PY compare_json.py > /dev/null; $PY compare_layers.py > /dev/null; echo "tables: table.md layers-table.md"; else $PY tables_030.py > /dev/null; echo "table: table-030.md"; fi
echo "DONE $CELL $(date '+%H:%M:%S')"
