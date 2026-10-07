"""Regression gate between the published run and a candidate run of the same scene: nothing gets worse silently.

Every catalog object (floor excepted) is scored the same way in both runs: its displayed model against the union of its
observation masks in each photo, occlusion-aware (x7.score_view: the model is hidden only where the photo's depth is in
front of it outside the mask). The candidate fails when, without being listed in --accept with a reason:

  model      an object that had a model has none;
  fit        an object's mean IoU drops by more than IOU_DROP;
  sheet      the V-guard boards are not one connected sheet (neighbouring boards' models farther than CONNECTED apart);
  contact    an object sits deeper than scene_contacts.CONTACT inside a gantry member and the run does not record it.

python scripts/workcell_regression_gate.py --baseline PUBLISHED_RUN --candidate NEW_RUN [--accept OBJECT=REASON ...] [--out FILE]
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

IOU_DROP, CONNECTED = .05, .03
PAIRS = (('v-guard-left', 'v-guard-center'), ('v-guard-center', 'v-guard-right'))


def _meshes(root, item):
    model = item.get('model') or {}
    if not model.get('nodes'):
        return []
    scene = trimesh.load(Path(root) / model['file'], force='scene')
    out = []
    for node in model['nodes']:
        if node in scene.graph.nodes_geometry:
            matrix, geom = scene.graph[node]
            mesh = scene.geometry[geom].copy()
            mesh.apply_transform(matrix)
            out.append(mesh)
    return out


def _distance(a, b):
    """Shortest distance between two mesh sets, from surface samples (seeded)."""
    sample = lambda meshes: np.concatenate([np.asarray(m.vertices) for m in meshes] + [trimesh.sample.sample_surface(m, 1500, seed=0)[0] for m in meshes if m.area > 0])
    from scipy.spatial import cKDTree
    return float(cKDTree(sample(b)).query(sample(a))[0].min())


def score(root):
    """{objectId: {'model': bool, 'iouByPhoto': {...}, 'meanIou': float|None}} plus sheet and contact findings."""
    from scripts.workcell_extra_models import _clean
    from scripts.workcell_guard_plates import _silhouette_iou
    from scripts.workcell_recgen_objects import _frames
    root = Path(root)
    catalog = json.loads((root / 'objects.json').read_text())
    geometry = json.loads((root / 'geometry.json').read_text())
    segmentation = json.loads((root / 'sam3.json').read_text())
    photos = sorted({o['photo'] for item in catalog['objects'] for o in item['observations']})
    frames = _frames(root, photos)
    loaded = {}

    def inputs(name):
        if name not in loaded:
            loaded[name] = np.load(root / name)
        return loaded[name]
    objects, meshes = {}, {}
    for item in catalog['objects']:
        if item['id'] == 'floor':
            continue
        meshes[item['id']] = _meshes(root, item)
        masks = {}
        for o in item['observations']:
            _, _, mask, _ = _clean(o, frames[o['photo']], geometry, segmentation, inputs)
            masks[o['photo']] = masks.get(o['photo'], np.zeros_like(mask)) | mask
        iou = {str(p): _silhouette_iou(meshes[item['id']], frames[p], m) for p, m in masks.items()} if meshes[item['id']] else {}
        objects[item['id']] = {'model': bool(meshes[item['id']]), 'iouByPhoto': iou,
                               'meanIou': round(float(np.mean(list(iou.values()))), 4) if iou else None}
    sheet = [{'boards': [a, b], 'gapNative': round(_distance(meshes[a], meshes[b]), 4)}
             for a, b in PAIRS if meshes.get(a) and meshes.get(b)]
    contacts = (catalog['coverage'].get('contacts') or {})
    return {'objects': objects, 'sheet': sheet, 'contactsStatus': contacts.get('status'), 'contactsUnresolved': contacts.get('unresolved', [])}


def compare(baseline, candidate, accept=None):
    """Failures of the candidate against the baseline (both from score()); accepted objects are reported, not failed."""
    accept, failures, notes = accept or {}, [], []
    for ident, before in baseline['objects'].items():
        after = candidate['objects'].get(ident)
        reason = accept.get(ident)
        found = []
        if after is None or (before['model'] and not after['model']):
            found.append(f'{ident}: lost its model')
        elif before['meanIou'] is not None and after['meanIou'] is not None and after['meanIou'] < before['meanIou'] - IOU_DROP:
            found.append(f"{ident}: mean IoU {before['meanIou']} -> {after['meanIou']} (more than {IOU_DROP} worse)")
        (notes if reason else failures).extend(f'{f} [accepted: {reason}]' if reason else f for f in found)
    for row in candidate['sheet']:
        if row['gapNative'] > CONNECTED and 'sheet' not in accept:
            failures.append(f"V-guard {row['boards'][0]} / {row['boards'][1]} not connected: {row['gapNative']} native apart")
    if candidate['contactsStatus'] not in (None, 'resolved', 'no gantry') and not candidate['contactsUnresolved']:
        failures.append(f"contacts status {candidate['contactsStatus']} without recorded unresolved overlaps")
    return failures, notes


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--accept', action='append', default=[], help='OBJECT=REASON: a deliberate, explained change')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    accept = dict(a.split('=', 1) for a in args.accept)
    before, after = score(args.baseline), score(args.candidate)
    failures, notes = compare(before, after, accept)
    report = {'baseline': str(args.baseline), 'candidate': str(args.candidate), 'rule': {'iouDrop': IOU_DROP, 'connectedNative': CONNECTED},
              'objects': {i: {'before': before['objects'][i], 'after': after['objects'].get(i)} for i in before['objects']},
              'sheet': {'before': before['sheet'], 'after': after['sheet']}, 'failures': failures, 'accepted': notes, 'passed': not failures}
    if args.out:
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    for ident, row in report['objects'].items():
        b, a = row['before'], row['after'] or {}
        if b['meanIou'] is not None or (a and a.get('meanIou') is not None):
            print(f"  {ident:20s} {b['meanIou']} -> {a.get('meanIou')}")
    print('sheet', [r['gapNative'] for r in after['sheet']], '| failures:', failures or 'none', '| accepted:', notes or 'none')
    sys.exit(1 if failures else 0)
