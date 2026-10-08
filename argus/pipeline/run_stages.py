"""Run the original layer-stage checks (S8-S14) on a locally served module-swapped publication, each as the ORIGINAL check module
through modal_apps/workcell_layer_trial.py (ephemeral Modal CPU), with the options the 090 report used:
  shape (standard multi-photo shape check, every model), floor, lines, plane_stereo, transfer, clearance, lower_edge (the 090
  targets + click part masks remapped to this publication's entity ids), box_faces, obvious_errors, plane_facets (guard).

    python run_stages.py VARIANT [stage ...]      (after serve_export.py VARIANT)
Outputs: swap-runs/<VARIANT>-stages/<stage>/{results.json, spend-ledger.json, files}
"""
from argus import ROOT
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

SP = Path(os.environ['PANOPTES_DATA_ROOT'])   # env.sh (env.template section 6)
HERE = ROOT / 'argus/checks'
API = 'https://layer-trial.invalid/report/api'
CELL = os.environ.get('STAGES_CELL', '090')                # STAGES_CELL=030: the published 030 report's view; no click part masks there
CONFIG = json.loads((ROOT / f'argus/pipeline/cells/{CELL}.json').read_text())
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
    part_masks = {ents[oid]: masks for oid, masks in CONFIG['partMasks'].items() if oid in ents}
    # targets = every part that has a click mask (the current module refuses a part without one) + the object itself
    targets = [{'entityId': ents[o], 'part': p} for o in LOWER_EDGE_TARGETS if o in ents for p in sorted(part_masks.get(ents[o]) or {})] + \
              [{'entityId': ents[o]} for o in LOWER_EDGE_TARGETS if o in ents]
    guard = [ents[o] for o in ('guard',) if o in ents]
    STAGES = {
        'shape': (HERE / 'shape_all.py', {'points': 12000}),
        'floor': (HERE / 'floor.py', {}),
        'lines': (HERE / 'lines.py', {}),
        'plane_stereo': (HERE / 'plane_stereo.py', {}),
        'transfer': (HERE / 'transfer.py', {}),
        'clearance': (HERE / 'clearance.py', {'api': API}),
        'lower_edge': (HERE / 'lower_edge.py', {'api': API, 'targets': targets, 'partMasks': part_masks}),
        'box_faces': (HERE / 'box_faces.py', {'api': API}),
        'obvious_errors': (HERE / 'obvious_errors.py', {'api': API, 'draw': True}),
        'plane_facets': (HERE / 'plane_facets.py', {'only': guard}),
    }
    out_root = SP / 'swap-runs' / f'{variant}-stages'
    out_root.mkdir(exist_ok=True)
    (out_root / 'entity-map.json').write_text(json.dumps(ents, indent=1))
    procs, failed = [], []
    for name in stages or list(STAGES):
        task, opts = STAGES[name]
        out = out_root / name
        if (out / 'results.json').exists():
            print(name, 'exists'); continue
        opts_path = out_root / f'{name}.opts.json'; opts_path.write_text(json.dumps(opts))
        cmd = [sys.executable, '-m', 'argus.checks.workcell_layer_trial', '--task', str(task), '--view', served['view'],
               '--photos-dir', served['photosDir'], '--photo', served['photos'], '--api', API, '--served-dir', served['served'],
               '--opts', str(opts_path), '--out', str(out)]
        log = open(out_root / f'{name}.log', 'w')
        procs.append((name, subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT), log))
        if len(procs) >= 3:  # three uploads at a time
            for n, p, l in procs:
                if p.wait():
                    failed.append(n)
                l.close(); print(n, 'exit', p.returncode, flush=True)
            procs = []
    for n, p, l in procs:
        if p.wait():
            failed.append(n)
        l.close(); print(n, 'exit', p.returncode, flush=True)
    missing = [name for name in stages or list(STAGES) if not (out_root / name / 'results.json').exists()]
    for name in stages or list(STAGES):
        print(name, 'FAILED' if name in failed else 'MISSING' if name in missing else 'ok')
    if failed:
        sys.exit(f'run_stages: {len(failed)} check(s) exited non-zero: {" ".join(failed)}')
    if missing:  # a check without results.json failed: exit non-zero so no report is built from it (customer review 2026-10-08)
        sys.exit(f'run_stages: {len(missing)} check(s) without results.json: {" ".join(missing)}')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2:])
