"""Contacts, not interpenetration: what is mounted on the gantry (lamps, light curtains, buttons, fence ends) may touch
a gantry member but never sit inside it, and one physical member has one model.

Each object is placed on its own evidence, so nothing stopped a lamp from landing inside the post that carries it
(2026-10-04: 87 % of one lamp inside a post, which hid it) or a fence end-frame from duplicating a gantry post. The
gantry is the coarse layer (square sections from silhouette widths); light curtains and fences carry the report's
measured endpoints and lamps sit on their own masks and depth. After every model is placed:

1. duplicate: a fence member >= DUPLICATE inside a gantry member and parallel to it is that member; the gantry keeps
   it and the fence stops displaying that node (members an endpoint reads, role '*lower*', are never dropped);
2. contact: under CONTACT of an object inside a member is mounting contact and is kept (recorded);
3. otherwise the member shifts perpendicular to its axis -- shortest clearing shift first, including the photos' lines
   of sight, the direction a photo constrains least -- and the shift is kept only when the gantry's mask IoU drops by
   at most IOU_DROP in every photo and nothing it merely touched gets deeper;
4. a lamp still inside a member slides toward the camera of the photo that observes it best, as a similarity about the
   camera centre (its outline in that photo is unchanged), by the smallest step that clears it, kept only when every
   observed photo still passes the lamp acceptance rule.

Everything is recorded in coverage.contacts and on the items (contactResolution). Display geometry only: no measured
endpoint, floor or scale changes (finalize re-reads endpoints from the models afterwards).

python scripts/workcell_scene_contacts.py --root RUN   (applies to a run directory in place, as the oneshot stage)
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DIRECTIONS, STEP, MAX_SHIFT = 24, .005, .15   # member shift search (native)
IOU_DROP = .03                                # largest gantry IoU loss per photo a shift may cost
CANDIDATES = 12                               # clearing shifts tried per member, shortest first
CONTACT = .06                                 # under this share of an object inside a member: mounting contact, kept
DUPLICATE, PARALLEL_DEG = .9, 10              # a structural member this far inside and this parallel is the same member
DUPLICATE_MARGIN = .02                        # ... allowing 2 cm-scale section differences between the two fits
SLIDE_STEP, MIN_SCALE = .0025, .85            # lamp slide toward its camera: scale about the camera centre
MARGIN = .005                                 # samples within this of a member's faces count as inside
SAMPLES = 4000                                # surface samples per object (seeded)


def boxes_of(scene):
    """{node: (centre, axes (columns), half extents)} of a gantry GLB whose nodes are boxes."""
    out = {}
    for node in scene.graph.nodes_geometry:
        matrix, geom = scene.graph[node]
        mesh = scene.geometry[geom].copy()
        mesh.apply_transform(matrix)
        obb = mesh.bounding_box_oriented.primitive  # exact for a box mesh
        out[node] = (obb.transform[:3, 3].copy(), obb.transform[:3, :3].copy(), np.asarray(obb.extents, float) / 2)
    return out


def inside(points, box, margin=MARGIN):
    centre, axes, half = box
    local = (np.asarray(points, float) - centre) @ axes
    return (np.abs(local) <= half + margin).all(1)


def clearing_shifts(box, points, toward=(), directions=DIRECTIONS, step=STEP, max_shift=MAX_SHIFT):
    """Shifts perpendicular to the box's long axis that leave none of `points` inside it: for each direction (DIRECTIONS
    evenly spaced, plus the `toward` directions, e.g. the photos' lines of sight, projected into that plane) the
    smallest one, sorted by length. Returns (candidates, samples inside before); no candidates when nothing clears."""
    centre, axes, half = box
    points = np.asarray(points, float)
    points = points[(np.abs((points - centre) @ axes) <= half + max_shift + MARGIN).all(1)]  # only what a shift can reach
    before = int(inside(points, box).sum())
    if not before:
        return [], 0
    long = axes[:, int(np.argmax(half))]
    u, v = (axes[:, k] for k in range(3) if k != int(np.argmax(half)))
    candidates = [np.cos(a) * u + np.sin(a) * v for a in np.arange(directions) * 2 * np.pi / directions]
    for d in toward:  # both senses of each line of sight, in the plane across the member
        d = np.asarray(d, float) - (np.asarray(d, float) @ long) * long
        if np.linalg.norm(d) > 1e-9:
            candidates += [d / np.linalg.norm(d), -d / np.linalg.norm(d)]
    found = []
    for direction in candidates:
        for magnitude in np.arange(step, max_shift + step / 2, step):
            if not inside(points, (centre + magnitude * direction, axes, half)).any():
                found.append(magnitude * direction)
                break
    return sorted(found, key=lambda s: (round(float(np.linalg.norm(s)), 9), tuple(np.round(s, 9)))), before


def slide_scale(vertices, camera, boxes, step=SLIDE_STEP, minimum=MIN_SCALE):
    """Largest s < 1 such that camera + s (vertices - camera) leaves no vertex inside any box; None if none clears."""
    vertices, camera = np.asarray(vertices, float), np.asarray(camera, float)
    for s in np.arange(1 - step, minimum - step / 2, -step):
        moved = camera + s * (vertices - camera)
        if not any(inside(moved, box).any() for box in boxes):
            return float(round(s, 6))
    return None


def _samples(path, nodes, seed=0):
    scene = trimesh.load(path, force='scene')
    parts = []
    for node in nodes:
        if node in scene.graph.nodes_geometry:
            matrix, geom = scene.graph[node]
            mesh = scene.geometry[geom].copy()
            mesh.apply_transform(matrix)
            parts.append(mesh)
    if not parts:
        return np.zeros((0, 3))
    mesh = trimesh.util.concatenate(parts)
    if mesh.area <= 0:
        return np.asarray(mesh.vertices, float)
    return np.concatenate([np.asarray(mesh.vertices, float), trimesh.sample.sample_surface(mesh, SAMPLES, seed=seed)[0]])


def _gantry_validation(root, item, boxes, photos):
    """workcell_gantry's own per-photo validation (IoU, unoccluded IoU, depth agreement) of these member boxes."""
    from scripts import workcell_gantry as gantry
    from scripts.workcell_photo_objects import decode_mask
    frames = gantry._frames(root, photos)
    masks = gantry.masks_from_segmentation(json.loads((root / 'sam3.json').read_text()), photos)
    out = {}
    for p in photos:
        mask = gantry.gantry_mask(item, masks, frames[p], p)
        everything = np.zeros_like(mask)
        for rles in masks.get(p, {}).values():
            for rle in rles:
                everything |= decode_mask(frames[p]['raw'], rle)
        result = gantry._validate(list(boxes.values()), list(boxes), frames[p], mask, everything)
        out[str(p)] = {k: v for k, v in result.items() if k != 'members'}
    return out


def _write_gantry(root, item, scene, shifts):
    for node, shift in shifts.items():
        matrix, geom = scene.graph[node]
        mesh = scene.geometry[geom]
        local = np.linalg.inv(matrix)[:3, :3] @ shift  # node geometry is stored in its node frame
        mesh.vertices = mesh.vertices + local
    (root / item['model']['file']).write_bytes(scene.export(file_type='glb'))


def _lamp_checks(root, record, transform):
    """Scores of a moved lamp in every observed photo, with the placement's own targets and acceptance rule."""
    from fast_report import x7
    from scripts import workcell_recgen_objects as small
    mesh = trimesh.load(root / record['glb'], force='mesh', process=False)
    v, f = np.asarray(mesh.vertices, float), np.asarray(mesh.faces, np.int64)
    caster, checks = x7.Caster(v, f), {}
    with np.load(root / record['input']) as z:
        for p in record['observedPhotos']:
            depth = z[f't{p}_depth']
            view = {'rays': x7.rays(z[f't{p}_K'], z[f't{p}_c2w'], depth.shape[1], depth.shape[0]), 'target': z[f't{p}_mask'] > 0, 'depth': depth}
            score = x7.score_view(caster, transform, view)
            checks[str(p)] = {'iou': round(score['iou'], 4), 'depthP50Relative': None if score['p50'] is None else round(score['p50'], 4)}
    up = np.asarray(record['floorNormal'], float)
    moved = x7.transformed(v, transform)
    ratio = float(np.ptp(np.percentile(moved @ up, [2, 98]))) / record['supportVerticalExtentNative'] if record['supportVerticalExtentNative'] else None
    lo, hi = small.ACCEPT['verticalExtentRatio']
    passes = all(c['iou'] >= small.ACCEPT['iou'] and c['depthP50Relative'] is not None and c['depthP50Relative'] <= small.ACCEPT['depthP50Relative']
                 for c in checks.values()) and ratio is not None and lo <= ratio <= hi
    return checks, passes, moved, f, mesh


def _node_samples(scene, node, count=800, seed=0):
    matrix, geom = scene.graph[node]
    mesh = scene.geometry[geom].copy()
    mesh.apply_transform(matrix)
    points = np.asarray(mesh.vertices, float)
    if mesh.area > 0:
        points = np.concatenate([points, trimesh.sample.sample_surface(mesh, count, seed=seed)[0]])
    obb = mesh.bounding_box_oriented.primitive
    return points, obb.transform[:3, int(np.argmax(obb.extents))]


def resolve(root):
    """Apply the rules to a run directory in place (the oneshot stage after every model is placed); returns the record."""
    from scripts.workcell_gantry import _frames
    root = Path(root)
    text = (root / 'objects.json').read_text()
    catalog = json.loads(text)
    items = {item['id']: item for item in catalog['objects']}
    gantry = items.get('gantry')
    record = {'rule': f'contacts, not interpenetration: under {CONTACT:.0%} of an object inside a gantry member is mounting contact; '
                      f'a structural member >= {DUPLICATE:.0%} inside and parallel is the same physical member (one model stays); '
                      f'otherwise the member shifts aside (gantry IoU drop <= {IOU_DROP}) and lamps slide toward their camera',
              'duplicates': [], 'members': [], 'lamps': [], 'contacts': [], 'unresolved': []}
    previous = catalog['coverage'].get('contacts')

    def save():
        record['status'] = 'resolved' if not record['unresolved'] else 'partly resolved'
        if previous:  # a later pass (e.g. after new guard plates) keeps what earlier passes changed
            record['earlierPasses'] = (previous.get('earlierPasses') or []) + [{k: v for k, v in previous.items() if k != 'earlierPasses'}]
        catalog['coverage']['contacts'] = record
        (root / 'objects.json').write_text(json.dumps(catalog, ensure_ascii=False, indent=2, allow_nan=False) + ('\n' if text.endswith('\n') else ''))
        return record
    if not gantry:
        record['unresolved'] = []
        record['status'] = 'no gantry'
        return save()
    photos = sorted({o['photo'] for o in gantry['observations']})
    scene = trimesh.load(root / gantry['model']['file'], force='scene')
    boxes = boxes_of(scene)
    geometry = json.loads((root / 'geometry.json').read_text())
    protected = {row['id'] for row in geometry.get('fence', {}).get('continuations', []) if 'lower' in row.get('role', '')}
    # 1. A fence member lying inside a gantry member, along it, is that same physical member: the gantry keeps it.
    for ident, item in items.items():
        if item.get('kind') != 'safety fence' or not (item.get('model') or {}).get('nodes'):
            continue
        fence = trimesh.load(root / item['model']['file'], force='scene')
        keep = []
        for node in item['model']['nodes']:
            points, axis = _node_samples(fence, node)
            twin = next((m for m, box in boxes.items() if node not in protected and inside(points, box, DUPLICATE_MARGIN).mean() >= DUPLICATE
                         and abs(axis @ box[1][:, int(np.argmax(box[2]))]) >= np.cos(np.radians(PARALLEL_DEG))), None)
            if twin:
                record['duplicates'].append({'object': ident, 'node': node, 'sameAs': twin})
            else:
                keep.append(node)
        if keep and len(keep) < len(item['model']['nodes']):
            item['model']['nodes'] = keep
            item['contactResolution'] = [row for row in record['duplicates'] if row['object'] == ident]
    samples = {ident: _samples(root / item['model']['file'], item['model']['nodes'])
               for ident, item in items.items() if ident not in ('gantry', 'floor') and (item.get('model') or {}).get('nodes')}
    lamps = {ident for ident, item in items.items() if (item.get('recgenModel') or {}).get('accepted')}
    fixed = {ident: points for ident, points in samples.items() if ident not in lamps and len(points)}
    validation = _gantry_validation(root, gantry, boxes, photos)
    iou = {p: v['iou'] for p, v in validation.items()}
    cameras = {p: _frames(root, [p])[p]['pose'][:3, 3] for p in photos}
    shifts = {}
    # 2. A member that holds more than mounting contact of a modelled object shifts aside, shortest first, if the gantry fit allows.
    for node, box in list(boxes.items()):
        share = {ident: float(inside(points, box).mean()) for ident, points in fixed.items()}
        owners = [ident for ident, s in share.items() if s >= CONTACT]
        record['contacts'] += [{'member': node, 'object': ident, 'insideFraction': round(s, 4)} for ident, s in share.items() if 0 < s < CONTACT]
        if not owners:
            continue
        target = np.concatenate([fixed[i] for i in owners])
        guard = np.concatenate([fixed[i] for i, s in share.items() if 0 < s < CONTACT] or [np.zeros((0, 3))])
        allowed = int(inside(guard, box).sum())
        candidates, before = clearing_shifts(box, target, toward=[box[0] - c for c in cameras.values()])
        row = {'member': node, 'overlaps': {i: round(share[i], 4) for i in owners}, 'insideSamplesBefore': before, 'iouBefore': iou, 'kept': False}
        for shift in candidates[:CANDIDATES]:
            moved = (box[0] + shift, box[1], box[2])
            if int(inside(guard, moved).sum()) > allowed:  # never pushed deeper into what it only touched
                continue
            trial = dict(boxes, **{node: moved})
            trial_validation = _gantry_validation(root, gantry, trial, photos)
            trial_iou = {p: v['iou'] for p, v in trial_validation.items()}
            if all(trial_iou[p] >= iou[p] - IOU_DROP for p in iou):
                row.update(shiftNative=np.round(shift, 5).tolist(), iouAfter=trial_iou, kept=True)
                boxes, validation, iou, shifts[node] = trial, trial_validation, trial_iou, shift
                break
        record['members'].append(row)
        if not row['kept']:
            record['unresolved'].append({'member': node, 'objects': owners, 'reason': f'no shift within {MAX_SHIFT} native (shortest '
                                         f'{CANDIDATES} of {len(candidates)} that clear) keeps the gantry IoU within {IOU_DROP}'})
    if shifts:
        _write_gantry(root, gantry, scene, shifts)
        gantry['gantryModel']['validation'] = validation
        gantry['contactResolution'] = [row for row in record['members'] if row['kept']]
    # 3. A lamp still more than touching a member slides toward the camera that sees it best (outline there unchanged).
    frames = _frames(root, photos)
    for ident in sorted(lamps):
        item = items[ident]
        lamp = json.loads((root / f"{item['recgenModel']['kind']}.json").read_text())
        mesh = trimesh.load(root / lamp['glb'], force='mesh', process=False)
        hits = [node for node, box in boxes.items() if inside(mesh.vertices, box).mean() >= CONTACT]
        if not hits:
            continue
        view = max(lamp['observedPhotos'], key=lambda p: lamp['sourceChecks'][str(p)].get('maskPixels', 0))
        camera = frames[view]['pose'][:3, 3]
        s = slide_scale(mesh.vertices, camera, [boxes[n] for n in hits])
        row = {'lamp': ident, 'inside': hits, 'photo': view, 'scaleAboutCamera': s, 'kept': False}
        if s is not None:
            transform = np.eye(4)
            transform[:3, :3] *= s
            transform[:3, 3] = (1 - s) * camera
            checks, passes, moved, _, original = _lamp_checks(root, lamp, transform)
            row.update(checks=checks, kept=bool(passes),
                       movedNative=round(float(np.linalg.norm(np.asarray(original.vertices).mean(0) - moved.mean(0))), 4))
            if passes:
                mesh.vertices = moved
                (root / lamp['glb']).write_bytes(mesh.export(file_type='glb'))
                lamp['contactResolution'] = row
                lamp['sourceChecks'] = {**lamp['sourceChecks'], **{p: {**lamp['sourceChecks'][p], **c} for p, c in checks.items()}}
                (root / f"{lamp['kind']}.json").write_text(json.dumps(lamp, indent=2))
                item['recgenModel'] = {**item['recgenModel'], 'sourceChecks': lamp['sourceChecks'], 'contactResolution': row}
        record['lamps'].append(row)
        if not row['kept']:
            record['unresolved'].append({'lamp': ident, 'members': hits, 'reason': 'no slide toward the camera keeps the acceptance rule'})
    return save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, required=True)
    print(json.dumps(resolve(parser.parse_args().root), indent=2))
