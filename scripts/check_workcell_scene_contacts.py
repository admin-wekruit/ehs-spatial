"""CPU check of scripts/workcell_scene_contacts.py: the geometry rules on synthetic boxes, and with --run a finished run:
no modelled object is more than mounting contact inside a gantry member unless the run records it as unresolved.

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:scripts python scripts/check_workcell_scene_contacts.py [--run RUN ...]
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import workcell_scene_contacts as contacts

POST = (np.array([0., 0., 1.5]), np.eye(3), np.array([.06, .06, 1.5]))  # a 0.12 x 0.12 x 3 vertical post


def check_geometry():
    # A lamp-sized blob half inside the post, in front of it (toward a camera on -y): every candidate clears it,
    # the shortest comes first, and the line of sight is among the candidates.
    lamp = np.random.default_rng(0).uniform([-.03, -.10, 2.0], [.03, -.03, 2.3], (500, 3))
    candidates, before = contacts.clearing_shifts(POST, lamp, toward=[POST[0] - np.array([0, -4., 1.5])])
    assert before > 0 and candidates, before
    lengths = [float(np.linalg.norm(s)) for s in candidates]
    assert all(b >= a - 1e-9 for a, b in zip(lengths, lengths[1:])), 'shortest first'
    assert all(not contacts.inside(lamp, (POST[0] + s, POST[1], POST[2])).any() for s in candidates)
    assert all(abs(s[2]) < 1e-12 for s in candidates), 'shifts stay perpendicular to the member axis'
    assert any(abs(s[1]) > .99 * np.linalg.norm(s) for s in candidates), 'a line-of-sight candidate exists'
    assert contacts.clearing_shifts(POST, lamp + [1, 0, 0]) == ([], 0), 'nothing inside: no shift'
    # Slide toward the camera: the smallest step that clears, every point stays on its own camera ray.
    camera = np.array([0, -4., 1.5])
    s = contacts.slide_scale(lamp, camera, [POST])
    moved = camera + s * (lamp - camera)
    assert s is not None and s < 1 and not contacts.inside(moved, POST).any()
    assert contacts.inside(camera + (s + 2 * contacts.SLIDE_STEP) * (lamp - camera), POST).any(), 'the smallest clearing slide'
    rays = (lamp - camera) / np.linalg.norm(lamp - camera, axis=1, keepdims=True)
    assert np.allclose((moved - camera) / np.linalg.norm(moved - camera, axis=1, keepdims=True), rays), 'outline unchanged'
    assert contacts.slide_scale(np.array([[0, 0., 1.5]]), camera + [0, 3.9, 0], [POST]) is None, 'a camera inside cannot clear'
    # Boxes read back from a GLB of box nodes.
    scene = trimesh.Scene()
    box = trimesh.creation.box(extents=[.12, .12, 3.])
    box.apply_transform(trimesh.transformations.rotation_matrix(np.radians(20), [0, 0, 1]))
    box.apply_translation([1, 2, 1.5])
    scene.add_geometry(box, node_name='gantry-post-1', geom_name='gantry-post-1')
    (centre, axes, half), = contacts.boxes_of(trimesh.load(trimesh.util.wrap_as_stream(scene.export(file_type='glb')), file_type='glb', force='scene')).values()
    assert np.allclose(centre, [1, 2, 1.5], atol=1e-6) and np.allclose(sorted(half), [.06, .06, 1.5], atol=1e-6)


def check_run(root):
    """Every modelled object is under mounting contact inside every gantry member, or recorded as unresolved."""
    root = Path(root)
    catalog = json.loads((root / 'objects.json').read_text())
    record = catalog['coverage'].get('contacts')
    assert record and record['status'] in ('resolved', 'partly resolved', 'no gantry'), record
    items = {item['id']: item for item in catalog['objects']}
    if 'gantry' not in items:
        return record
    boxes = contacts.boxes_of(trimesh.load(root / items['gantry']['model']['file'], force='scene'))
    unresolved = {(u.get('member'), o) for u in record['unresolved'] for o in u.get('objects', [])}
    unresolved |= {(m, u['lamp']) for u in record['unresolved'] if 'lamp' in u for m in u['members']}
    worst = 0.
    for ident, item in items.items():
        if ident in ('gantry', 'floor') or not (item.get('model') or {}).get('nodes'):
            continue
        points = contacts._samples(root / item['model']['file'], item['model']['nodes'])
        for node, box in boxes.items():
            share = float(contacts.inside(points, box).mean()) if len(points) else 0.
            assert share < contacts.CONTACT or (node, ident) in unresolved, (ident, node, round(share, 4))
            worst = max(worst, share)
    for row in record['duplicates']:
        assert row['node'] not in items[row['object']]['model']['nodes'], row
    return {**record, 'worstInsideFraction': round(worst, 4)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', type=Path, action='append', default=[])
    args = parser.parse_args()
    check_geometry()
    for run in args.run:
        result = check_run(run)
        print(run.name, result['status'], 'worst inside', result.get('worstInsideFraction'), 'duplicates', len(result['duplicates']))
    print('check_workcell_scene_contacts passed')
