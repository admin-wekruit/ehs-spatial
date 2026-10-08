"""Run the original layer-stage checks (S8-S14) on a locally served module-swapped publication, each as the ORIGINAL check module
through modal_apps/workcell_layer_trial.py (ephemeral Modal CPU), with the options the 090 report used:
  shape (standard multi-photo shape check, every model), floor, lines, plane_stereo, transfer, clearance, lower_edge (the 090
  targets + click part masks remapped to this publication's entity ids), box_faces, obvious_errors, plane_facets (guard).

    python run_stages.py VARIANT [stage ...]      (after serve_export.py VARIANT)
Outputs: swap-runs/<VARIANT>-stages/<stage>/{results.json, spend-ledger.json, files}
"""
import json
import os
from pathlib import Path
import subprocess
import sys

SP = Path(os.environ['SWAP_SCRATCH'])   # env.sh (env.template section 6)
RN = Path(os.environ['SWAP_NOTES'])
WT = Path(os.environ['PANOPTES_WORKCELL'])
HERE = Path(__file__).resolve().parent
MODAL_RUN = os.environ['MODAL_RUN'].split()   # env.sh: 'python WT/scripts/onprem/run_stage.py --weights DIR' on-prem ('modal run' on the original machine)
API = 'https://layer-trial.invalid/report/api'
CELL = os.environ.get('STAGES_CELL', '090')                # STAGES_CELL=030: the published 030 report's view; no click part masks there
OLD_VIEW = SP / ('sept/new-view.json' if CELL == '090' else 'checks/cd84-view.json')   # the published report: entity -> object for the lower-edge targets
PART_MASKS = RN / 'workcell-lower-edge-2026-10-05/090-part-masks.json' if CELL == '090' else None
LOWER_EDGE_TARGETS = {'right_light_curtain': ['recessed yellow housing', 'yellow front plate'], 'left_light_curtain': ['recessed yellow housing', 'yellow front plate'],
                      'right_fence': [None], 'left_fence': [None]}   # as loweredge-final-090 (object-level target = part None)


def entity_map(doc):
    """object_id -> entity id: the import lineage (offline_import sourceRecordId), else the model artifact's source record
    (the published September document keeps the object id only in the representations' sourceRefs)."""
    out = {}
    for e in doc['entities']:
        for l in e.get('lineage') or []:
            if isinstance(l, dict) and l.get('operation') == 'offline_import' and l.get('sourceRecordId') and not l['sourceRecordId'].startswith('object_'):
                out.setdefault(l['sourceRecordId'], e['id'])
        for r in e.get('representations') or []:
            for sref in r.get('sourceRefs') or []:
                if isinstance(sref, dict) and sref.get('role') == 'model_artifact' and sref.get('sourceRecordId'):
                    out.setdefault(sref['sourceRecordId'], e['id'])
    return out


def main(variant, stages):
    served = json.loads((SP / 'swap-runs' / f'{variant}-served.json').read_text())
    doc = json.loads(Path(served['view']).read_bytes())['publication']['snapshot']['revision']['document']
    ents = entity_map(doc)
    old = json.loads(OLD_VIEW.read_bytes())['publication']['snapshot']['revision']['document']
    old_ents = {v: k for k, v in entity_map(old).items()}   # old entity id -> object id
    pm_old = json.loads(PART_MASKS.read_text()) if PART_MASKS else {}
    part_masks = {ents[old_ents[k]]: v for k, v in pm_old.items() if k in old_ents and old_ents[k] in ents}
    # targets = every part that has a click mask (the current module refuses a part without one) + the object itself
    targets = [{'entityId': ents[o], 'part': p} for o in LOWER_EDGE_TARGETS if o in ents for p in sorted(part_masks.get(ents[o]) or {})] + \
              [{'entityId': ents[o]} for o in LOWER_EDGE_TARGETS if o in ents]
    guard = [ents[o] for o in ('guard',) if o in ents]
    STAGES = {
        'shape': (HERE / 'shape_all_task.py', {'points': 12000}),
        'floor': (WT / 'scripts/workcell_checks/floor.py', {}),
        'lines': (WT / 'scripts/workcell_checks/lines.py', {}),
        'plane_stereo': (WT / 'scripts/workcell_checks/plane_stereo.py', {}),
        'transfer': (WT / 'scripts/workcell_checks/transfer.py', {}),
        'clearance': (WT / 'scripts/workcell_checks/clearance.py', {'api': API}),
        'lower_edge': (WT / 'scripts/workcell_checks/lower_edge.py', {'api': API, 'targets': targets, 'partMasks': part_masks}),
        'box_faces': (WT / 'scripts/workcell_checks/box_faces.py', {'api': API}),
        'obvious_errors': (WT / 'scripts/workcell_checks/obvious_errors.py', {'api': API, 'draw': True}),
        'plane_facets': (WT / 'scripts/workcell_checks/plane_facets.py', {'only': guard}),
    }
    out_root = SP / 'swap-runs' / f'{variant}-stages'
    out_root.mkdir(exist_ok=True)
    (out_root / 'entity-map.json').write_text(json.dumps(ents, indent=1))
    procs = []
    for name in stages or list(STAGES):
        task, opts = STAGES[name]
        out = out_root / name
        if (out / 'results.json').exists():
            print(name, 'exists'); continue
        opts_path = out_root / f'{name}.opts.json'; opts_path.write_text(json.dumps(opts))
        cmd = MODAL_RUN + [str(WT / 'modal_apps/workcell_layer_trial.py'), '--task', str(task), '--view', served['view'],
               '--photos-dir', served['photosDir'], '--photo', served['photos'], '--api', API, '--served-dir', served['served'],
               '--opts', str(opts_path), '--out', str(out)]
        log = open(out_root / f'{name}.log', 'w')
        procs.append((name, subprocess.Popen(cmd, cwd=WT, stdout=log, stderr=subprocess.STDOUT), log))
        if len(procs) >= 3:  # three uploads at a time
            for n, p, l in procs:
                p.wait(); l.close(); print(n, 'exit', p.returncode, flush=True)
            procs = []
    for n, p, l in procs:
        p.wait(); l.close(); print(n, 'exit', p.returncode, flush=True)
    missing = [name for name in stages or list(STAGES) if not (out_root / name / 'results.json').exists()]
    for name in stages or list(STAGES):
        print(name, 'MISSING' if name in missing else 'ok')
    if missing:  # a check without results.json failed: exit non-zero so no report is built from it (customer review 2026-10-08)
        sys.exit(f'run_stages: {len(missing)} check(s) without results.json: {" ".join(missing)}')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2:])
