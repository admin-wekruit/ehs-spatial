"""V-guard boards as folded plates fitted to their own photo support -- no generated mesh in the loop.

With two photos per cell the RecGen guard came out as one thick lump: the two photos' pointmaps put the same board a
few centimetres apart and the generated mesh fuses both layers (2026-10-04: a second "face" parallel to the first,
1-2 degrees apart, in 090). A board is a flat sheet with at most one fold, so per board:

  support  the cleaned objects-stage support of each of its observations (workcell_extra_models._clean), all photos;
  faces    sequential RANSAC planes, tolerance TOL x the median camera depth (wide enough to merge the photos' parallel
           layers into one sheet); a second face only when it holds >= SECOND of the points and turns >= FOLD_DEG
           from the first (a real fold, not a depth layer);
  plates   each face is its 1-99 % oriented rectangle in its plane, clipped at the line where the two planes meet;
  texture  each plate from its best photo (workcell_photo_texture.texture_planar_mesh, the board's own masks);
  strips   a board whose support is a near-line (second spread < ELONGATED x the first) faces the camera that sees it;
  accept   the plates replace the RecGen part when their mean silhouette IoU against the board's masks is >= FLOOR_IOU
           and no worse than the part's by more than MARGIN (same measure); else the part stays, reason recorded.

Display geometry and a conditional fold angle (the report's bend analysis reads the displayed faces); not surveyed.

python scripts/workcell_guard_plates.py --root RUN --sources PHOTO ...   (in place, as the oneshot stage)
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

GUARDS = ('v-guard-left', 'v-guard-center', 'v-guard-right')
TOL, SECOND, FOLD_DEG, ITERATIONS = .012, .2, 15., 600
SHARED = .25            # a fold's second face needs this share of its points from each of two photos (multi-photo boards)
ELONGATED = .3          # second principal spread under this x the first: a strip, whose tilt about its length is unobserved
FLOOR_IOU, MARGIN = .35, .05  # plates replace the RecGen part when their mean IoU >= FLOOR_IOU and >= the part's mean - MARGIN
FILE = 'guard-plates.glb'


def _ransac(points, tol, rng, iterations=ITERATIONS):
    best = None
    for _ in range(iterations):
        a, b, c = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(b - a, c - a)
        length = np.linalg.norm(normal)
        if length < 1e-12:
            continue
        inliers = np.abs((points - a) @ (normal / length)) <= tol
        if best is None or inliers.sum() > best.sum():
            best = inliers
    centre = points[best].mean(0)
    normal = np.linalg.svd(points[best] - centre, full_matrices=False)[2][2]
    return normal, centre, np.abs((points - centre) @ normal) <= tol


def fit_faces(points, tol, camera=None, seed=0, photos=None):
    """[(normal, centre, inlier points)]: one face, or two meeting at a real fold. A strip (elongated support) faces the
    camera that sees it: its tilt about its own length is not observed, and a RANSAC plane through a near-line is noise.
    With `photos` (each point's photo) and a board seen in more than one photo, a fold needs its second face seen by more
    than one photo too (>= SHARED of that face's points from each of two photos): a face that only one photo supports
    is the photos disagreeing about the board's tilt (2026-10-04, 030 left: 30 degrees between the photos' planes)."""
    points = np.asarray(points, float)
    centre = points.mean(0)
    _, spread, axes = np.linalg.svd(points - centre, full_matrices=False)
    if camera is not None and spread[1] < ELONGATED * spread[0]:
        view = np.asarray(camera, float) - centre
        normal = view - (view @ axes[0]) * axes[0]
        return [(normal / np.linalg.norm(normal), centre, points)]
    rng = np.random.default_rng(seed)
    n1, c1, in1 = _ransac(points, tol, rng)
    faces = [(n1, c1, points[in1])]
    rest = points[~in1]
    if len(rest) >= max(30, SECOND * len(points)):
        n2, c2, in2 = _ransac(rest, tol, rng)
        seen = True
        if photos is not None and len(set(np.asarray(photos).tolist())) > 1:
            tags = np.asarray(photos)[~in1][in2]
            seen = sum(np.mean(tags == p) >= SHARED for p in set(tags.tolist())) >= 2
        if seen and in2.sum() >= SECOND * len(points) and np.degrees(np.arccos(min(1., abs(float(n1 @ n2))))) >= FOLD_DEG:
            faces.append((n2, c2, rest[in2]))
    return faces


def _clip(polygon, g):
    """Sutherland-Hodgman: the part of a convex 2D polygon where the affine function g(q) >= 0."""
    out = []
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
        ga, gb = g(a), g(b)
        if ga >= 0:
            out.append(a)
        if (ga >= 0) != (gb >= 0):
            out.append(a + (b - a) * ga / (ga - gb))
    return np.asarray(out)


def face_polygon(face, other=None):
    """3D polygon of one face: its points' 1-99 % oriented rectangle in its plane, clipped at the fold with `other`."""
    from scripts.workcell_extra_models import _rectangle
    normal, centre, points = face
    u = np.linalg.svd(points - centre, full_matrices=False)[2][0]
    v = np.cross(normal, u)
    xy = np.c_[(points - centre) @ u, (points - centre) @ v]
    polygon = _rectangle(xy)
    if other is not None:
        n2, c2, _ = other
        f = lambda q: float((centre + q[0] * u + q[1] * v - c2) @ n2)
        side = np.sign(np.median([f(q) for q in xy])) or 1.
        polygon = _clip(polygon, lambda q: side * f(q))
    return centre + polygon[:, :1] * u + polygon[:, 1:2] * v


def _sheet(polygon, color):
    faces = [[0, i, i + 1] for i in range(1, len(polygon) - 1)]
    mesh = trimesh.Trimesh(np.asarray(polygon, float), faces, process=False)
    mesh.visual.vertex_colors = np.tile(np.r_[color, 255], (len(polygon), 1)).astype(np.uint8)
    return mesh


def _silhouette_iou(meshes, frame, mask):
    """x7.score_view IoU: the model is hidden only where the photo's depth is in front of it outside the mask."""
    from fast_report import x7
    mesh = trimesh.util.concatenate(meshes)
    h, w = mask.shape
    view = {'rays': x7.rays(frame['K'], frame['pose'], w, h), 'target': np.asarray(mask, bool), 'depth': frame['depth']}
    return round(float(x7.score_view(x7.Caster(mesh.vertices, mesh.faces), np.eye(4), view)['iou']), 4)


SLIDE_MAX = .25  # a one-photo board's distance is fitted within 1 +- this (scale about its camera)
BRIDGE_GAP = .25  # neighbouring plates at most this far apart are joined by a bridge; farther apart they are not one sheet
PAIRS = (('v-guard-left', 'v-guard-center'), ('v-guard-center', 'v-guard-right'))


def _outline(polygon, per=60):
    return np.concatenate([a + np.linspace(0, 1, per, endpoint=False)[:, None] * (b - a) for a, b in zip(polygon, np.roll(polygon, -1, axis=0))])


def gap(a, b):
    """Shortest distance between two polygons' outlines."""
    return float(np.min(np.linalg.norm(_outline(a)[:, None] - _outline(b)[None], axis=2)))


def _nearest_edge(polygon, other):
    k, outline = len(polygon), _outline(other)
    i = min(range(k), key=lambda i: float(np.min(np.linalg.norm(outline - (polygon[i] + polygon[(i + 1) % k]) / 2, axis=1))))
    return polygon[i], polygon[(i + 1) % k]


def bridge(a, b):
    """Two triangles closing the gap between neighbouring plates (each plate's edge nearest the other, joined); the plates
    themselves keep the extent their photos support. Empty when they already touch."""
    if gap(a, b) < 1e-3:
        return []
    (a0, a1), (b0, b1) = _nearest_edge(a, b), _nearest_edge(b, a)
    if np.linalg.norm(a0 - b0) + np.linalg.norm(a1 - b1) > np.linalg.norm(a0 - b1) + np.linalg.norm(a1 - b0):
        b0, b1 = b1, b0
    return [np.array([a0, a1, b1]), np.array([a0, b1, b0])]


def planes_line(first, second, min_angle=10.):
    """(point, direction) of the line where two planes (normal, point) meet; None when they are near parallel."""
    (n1, c1), (n2, c2) = first, second
    d = np.cross(n1, n2)
    if np.linalg.norm(d) < np.sin(np.radians(min_angle)):
        return None
    d /= np.linalg.norm(d)
    return np.linalg.solve(np.stack([n1, n2, d]), [n1 @ c1, n2 @ c2, d @ (c1 + c2) / 2]), d


def _model_meshes(root, item):
    model = item.get('model') or {}
    if not model.get('nodes'):
        return []
    scene = trimesh.load(Path(root) / model['file'], force='scene')
    meshes = []
    for node in model['nodes']:
        matrix, geom = scene.graph[node]
        mesh = scene.geometry[geom].copy()
        mesh.apply_transform(matrix)
        meshes.append(mesh)
    return meshes


def _at_height(line, height, up, offset):
    """Point of a line at a height above the floor (h(p) = p . up + offset)."""
    point, d = line
    return point + (height - offset - point @ up) / (d @ up) * d


def assemble(planes, points, up, offset):
    """The V-guard as one bent sheet: centre band and two wings joined at the lines where their planes meet, one shared
    bottom and top edge (lowest 2nd / highest 98th height percentile of all three boards' support), each wing as wide as
    its own support (98th percentile away from its fold). planes/points: {'left'|'center'|'right': ...}; returns
    {side: polygon} or None when a fold is missing or near-vertical lines cannot be formed."""
    heights = {s: p @ up + offset for s, p in points.items()}
    bottom = min(float(np.percentile(h, 2)) for h in heights.values())
    top = max(float(np.percentile(h, 98)) for h in heights.values())
    folds = {}
    for wing in ('left', 'right'):
        line = planes_line(planes[wing], planes['center'])
        if line is None or abs(line[1] @ up) < .5:  # the fold of a standing sheet is near vertical
            return None
        folds[wing] = line
    corners = {wing: (_at_height(folds[wing], bottom, up, offset), _at_height(folds[wing], top, up, offset)) for wing in folds}
    polygons = {'center': np.array([corners['left'][0], corners['right'][0], corners['right'][1], corners['left'][1]])}
    for wing in ('left', 'right'):
        normal = planes[wing][0]
        point, d = folds[wing]
        w = np.cross(normal, d)
        w /= np.linalg.norm(w)
        away = (points[wing] - point) @ w
        if np.median(away) < 0:
            w, away = -w, -away
        width = float(np.percentile(away, 98))
        if width <= 0:
            return None
        b, t = corners[wing]
        polygons[wing] = np.array([b, b + width * w, t + width * w, t])
    return polygons


def _fit_assembly(boards, frames, up, offset, record):
    """Assemble the three faces; a board one photo sees keeps its outline there but its distance is fitted (scale about
    that camera, 1 +- SLIDE_MAX) so the assembled sheet best matches every board's masks (occlusion-aware mean IoU)."""
    sides = {'v-guard-left': 'left', 'v-guard-center': 'center', 'v-guard-right': 'right'}
    if set(boards) != set(sides) or any(len(b['faces']) != 1 for b in boards.values()):
        record['assembly'] = {'status': 'skipped', 'reason': 'needs exactly one fitted face for each of the three boards'}
        return None
    base = {sides[i]: (b['faces'][0][0], b['faces'][0][1]) for i, b in boards.items()}
    support = {sides[i]: b['points'] for i, b in boards.items()}
    free = [i for i, b in boards.items() if len(b['photos']) == 1]
    grids = [np.round(np.arange(1 - SLIDE_MAX, 1 + SLIDE_MAX + 1e-9, .05), 4) if i in free else np.array([1.]) for i in boards]
    best = None
    for combo in np.array(np.meshgrid(*grids)).T.reshape(-1, len(boards)):
        planes, pts = dict(base), dict(support)
        for (ident, b), s in zip(boards.items(), combo):
            if s != 1.:
                camera = frames[b['photos'][0]]['pose'][:3, 3]
                planes[sides[ident]] = (base[sides[ident]][0], camera + s * (base[sides[ident]][1] - camera))
                pts[sides[ident]] = camera + s * (support[sides[ident]] - camera)
        polygons = assemble(planes, pts, up, offset)
        if polygons is None:
            continue
        scores = [_silhouette_iou([_sheet(polygons[sides[i]], b['color'])], frames[p], b['masks'][p]) for i, b in boards.items() for p in b['masks']]
        key = (round(float(np.mean(scores)), 4), -float(np.abs(np.asarray(combo) - 1).sum()))
        if best is None or key > best[0]:
            best = (key, combo, polygons)
    if best is None:
        record['assembly'] = {'status': 'failed', 'reason': 'no fold lines between the boards\' planes'}
        return None
    record['assembly'] = {'status': 'assembled', 'meanIou': best[0][0],
                          'scaleAboutCamera': {i: float(s) for i, s in zip(boards, best[1]) if i in free}}
    return {i: best[2][sides[i]] for i in boards}


def build(root, sources):
    """Fit, join, texture and (when accepted) install the plates of every V-guard board of a run; returns the record."""
    from scripts.workcell_extra_models import _clean
    from scripts.workcell_photo_texture import source_texture_frames, texture_planar_mesh
    from scripts.workcell_recgen_objects import _frames
    root = Path(root)
    text = (root / 'objects.json').read_text()
    catalog = json.loads(text)
    items = {item['id']: item for item in catalog['objects']}
    geometry = json.loads((root / 'geometry.json').read_text())
    segmentation = json.loads((root / 'sam3.json').read_text())
    photos = sorted({o['photo'] for ident in GUARDS if ident in items for o in items[ident]['observations']})
    frames = _frames(root, photos)
    textured = source_texture_frames(root, {p: {'K': frames[p]['K'], 'pose': frames[p]['pose']} for p in photos},
                                     [Path(s) for s in sources])
    loaded = {}

    def inputs(name):
        if name not in loaded:
            loaded[name] = np.load(root / name)
        return loaded[name]
    scene = trimesh.Scene()
    record = {'rule': f'RANSAC faces (tolerance {TOL} x median depth; a second face holds >= {SECOND:.0%}, turns >= {FOLD_DEG:.0f} deg and '
                      f'is seen by two photos when two see the board; a strip faces its camera); one bent sheet: centre band and wings '
                      f'joined at the lines where their planes meet, one shared bottom and top edge, wing widths from their own support; '
                      f'a one-photo board keeps its outline but its distance (scale about its camera, 1 +- {SLIDE_MAX}) is fitted to all '
                      f'masks; faces replace the RecGen parts when their occlusion-aware mean IoU >= {FLOOR_IOU} and >= the parts - {MARGIN}',
              'boards': []}
    boards = {}
    for ident in GUARDS:  # 1. faces from each board's own photo support
        item = items.get(ident)
        if not item:
            continue
        points, colors, masks, tags = [], [], {}, []
        for o in item['observations']:
            p, c, mask, _ = _clean(o, frames[o['photo']], geometry, segmentation, inputs)
            points.append(p)
            tags.append(np.full(len(p), o['photo']))
            colors.append(c)
            masks[o['photo']] = masks.get(o['photo'], np.zeros_like(mask)) | mask
        points = np.concatenate(points)
        row = {'objectId': ident, 'supportPoints': int(len(points)), 'photos': sorted(masks)}
        if len(points) < 100:
            row.update(kept=False, reason='too little cleaned support for a plane fit')
            record['boards'].append(row)
            continue
        camera = frames[item['observations'][0]['photo']]['pose']
        depth = float(np.median((points - camera[:3, 3]) @ camera[:3, 2]))
        faces = fit_faces(points, TOL * depth, camera[:3, 3], photos=np.concatenate(tags))  # two close photos leave a band's tilt open too
        polygons = [face_polygon(face, faces[1 - k] if len(faces) == 2 else None) for k, face in enumerate(faces)]
        boards[ident] = {'row': row, 'faces': faces, 'polygons': [q for q in polygons if len(q) >= 3], 'masks': masks, 'points': points,
                         'photos': sorted(masks), 'color': np.median(np.concatenate(colors), axis=0).astype(np.uint8), 'depth': depth}
    from scripts.workcell_extra_models import _floor_basis
    up, offset, _, _ = _floor_basis(geometry)
    # 2. one sheet: the assembled bent sheet, or the independent plates with their small gaps closed -- the better connected one
    assembled = _fit_assembly(boards, frames, up, offset, record)
    singles = {i: b['polygons'][0] for i, b in boards.items() if len(b['polygons']) == 1}
    bridged, joints = None, []
    if len(singles) == len(boards) == 3:
        bridged = {i: [q] for i, q in singles.items()}
        for a, b in PAIRS:
            joints.append({'boards': [a, b], 'gapNative': round(gap(singles[a], singles[b]), 4)})
            if joints[-1]['gapNative'] <= BRIDGE_GAP:  # the bridge belongs to the wing
                bridged[b if b != 'v-guard-center' else a] += bridge(singles[a], singles[b])
    def mean_iou(polygons):
        return float(np.mean([_silhouette_iou([_sheet(q, b['color']) for q in polygons[i]], frames[p], b['masks'][p])
                              for i, b in boards.items() for p in b['masks']]))
    candidates = {}
    if assembled:
        candidates['assembled sheet'] = {i: [q] for i, q in assembled.items()}
    if bridged and all(j['gapNative'] <= BRIDGE_GAP for j in joints):
        candidates['plates with bridges'] = bridged
    scores = {name: round(mean_iou(polys), 4) for name, polys in candidates.items()}
    record['candidates'] = {'scores': scores, 'joints': joints}
    if scores:
        choice = max(scores, key=lambda name: (scores[name], name == 'assembled sheet'))
        record['candidates']['chosen'] = choice
        for ident, polygons in candidates[choice].items():
            boards[ident]['polygons'] = polygons
    assembled = bool(scores)  # the decision below is for one connected sheet
    scored = {}
    for ident, board in boards.items():  # 3. texture and score every board
        item, row, faces = items[ident], board['row'], board['faces']
        sheets = [texture_planar_mesh(_sheet(q, board['color']), textured, board['masks'])[0] for q in board['polygons']]
        iou = {str(p): _silhouette_iou(sheets, frames[p], board['masks'][p]) for p in board['masks']} if sheets else {}
        baseline = dict(item, model=(item.get('recgenGuard') or {}).get('model') or item.get('model'))
        current = _model_meshes(root, baseline)
        before = {str(p): _silhouette_iou(current, frames[p], board['masks'][p]) for p in board['masks']} if current else {}
        fold = (float(np.degrees(np.arccos(min(1., abs(float(faces[0][0] @ faces[1][0])))))) if len(faces) == 2 else None)
        mean, previous = (float(np.mean(list(v.values()))) if v else 0. for v in (iou, before))
        kept = bool(sheets) and mean >= FLOOR_IOU and mean >= previous - MARGIN
        scored[ident] = (sheets, iou, before)
        row.update(faces=[{'normal': np.round(f[0], 5).tolist(), 'inlierFraction': round(len(f[2]) / row['supportPoints'], 4)} for f in faces],
                   foldBetweenFaceNormalsDeg=None if fold is None else round(fold, 2), toleranceNative=round(TOL * board['depth'], 4),
                   iouByPhoto=iou, recgenPartIouByPhoto=before, kept=kept)
    if assembled:  # one sheet: all three faces replace the parts, or none do (same measure, all boards and photos together)
        mine = [v for _, iou, _ in scored.values() for v in iou.values()]
        theirs = [v for _, _, before in scored.values() for v in before.values()]
        whole = bool(mine) and np.mean(mine) >= FLOOR_IOU and np.mean(mine) >= (np.mean(theirs) if theirs else 0.) - MARGIN
        record['candidates'].update(meanIouAllBoards=round(float(np.mean(mine)), 4) if mine else None,
                                    recgenMeanIouAllBoards=round(float(np.mean(theirs)), 4) if theirs else None, kept=bool(whole))
        for ident in boards:
            boards[ident]['row']['kept'] = bool(whole)
    for ident, board in boards.items():  # 4. install what is kept
        item, row = items[ident], board['row']
        sheets, iou, before = scored[ident]
        mean, previous = (float(np.mean(list(v.values()))) if v else 0. for v in (iou, before))
        kept = row['kept']
        if kept:
            nodes = []
            for k, sheet in enumerate(sheets):
                node = f'{ident}-face-{k + 1}'
                scene.add_geometry(sheet, node_name=node, geom_name=node)
                nodes.append(node)
            item.setdefault('recgenGuard', {'model': item.get('model'), 'representation': item.get('representation')})
            item['model'] = {'file': FILE, 'nodes': nodes}
            item['representation'] = (f'plate fitted to this board\'s photo support ({len(nodes)} face{"s" if len(nodes) > 1 else ""}), joined to '
                                      'its neighbours where the photos allow; photo texture')
            item['guardPlates'] = row
        else:
            row['reason'] = row.get('reason') or f'plate silhouette mean IoU {mean:.3f} (RecGen part {previous:.3f}) fails the rule: the RecGen part stays'
        record['boards'].append(row)
    if scene.geometry:
        (root / FILE).write_bytes(scene.export(file_type='glb'))
    record['status'] = 'plates' if record['boards'] and all(b.get('kept') for b in record['boards']) else \
        'partly plates' if any(b.get('kept') for b in record['boards']) else 'kept RecGen'
    catalog['coverage']['guardPlates'] = record
    (root / 'objects.json').write_text(json.dumps(catalog, ensure_ascii=False, indent=2, allow_nan=False) + ('\n' if text.endswith('\n') else ''))
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--sources', type=Path, nargs='+', required=True, help='original photos of the scene, in photo order')
    args = parser.parse_args()
    print(json.dumps(build(args.root, args.sources), indent=2))
