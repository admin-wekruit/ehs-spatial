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
    from fast_report import x7
    mesh = trimesh.util.concatenate(meshes)
    caster = x7.Caster(mesh.vertices, mesh.faces)
    h, w = mask.shape
    hit = np.isfinite(caster.depth(np.eye(4), x7.rays(frame['K'], frame['pose'], w, h)))
    return round(float((hit & mask).sum() / max(1, (hit | mask).sum())), 4)


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


def build(root, sources):
    """Fit, texture and (when accepted) install the plates of every V-guard board of a run; returns the record."""
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
    scene, record = trimesh.Scene(), {'rule': f'RANSAC faces (tolerance {TOL} x median depth; a second face holds >= {SECOND:.0%} and '
                                              f'turns >= {FOLD_DEG:.0f} deg; a strip faces its camera); 1-99 % rectangles clipped at the fold; they replace the RecGen '
                                              f'part when their mean silhouette IoU >= {FLOOR_IOU} and >= the part\'s - {MARGIN}', 'boards': []}
    for ident in GUARDS:
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
        faces = fit_faces(points, TOL * depth, camera[:3, 3], photos=np.concatenate(tags))
        color = np.median(np.concatenate(colors), axis=0).astype(np.uint8)
        sheets = []
        for k, face in enumerate(faces):
            polygon = face_polygon(face, faces[1 - k] if len(faces) == 2 else None)
            if len(polygon) < 3:
                continue
            sheet, _ = texture_planar_mesh(_sheet(polygon, color), textured, masks)
            sheets.append(sheet)
        iou = {str(p): _silhouette_iou(sheets, frames[p], masks[p]) for p in masks} if sheets else {}
        current = _model_meshes(root, item)
        before = {str(p): _silhouette_iou(current, frames[p], masks[p]) for p in masks} if current else {}
        fold = (float(np.degrees(np.arccos(min(1., abs(float(faces[0][0] @ faces[1][0])))))) if len(faces) == 2 else None)
        mean, previous = (float(np.mean(list(v.values()))) if v else 0. for v in (iou, before))
        kept = bool(sheets) and mean >= FLOOR_IOU and mean >= previous - MARGIN
        row.update(faces=[{'normal': np.round(f[0], 5).tolist(), 'inlierFraction': round(len(f[2]) / len(points), 4)} for f in faces],
                   foldBetweenFaceNormalsDeg=None if fold is None else round(fold, 2), toleranceNative=round(TOL * depth, 4),
                   iouByPhoto=iou, recgenPartIouByPhoto=before, kept=kept)
        if kept:
            nodes = []
            for k, sheet in enumerate(sheets):
                node = f'{ident}-face-{k + 1}'
                scene.add_geometry(sheet, node_name=node, geom_name=node)
                nodes.append(node)
            item['recgenGuard'] = {'model': item.get('model'), 'representation': item.get('representation')}
            item['model'] = {'file': FILE, 'nodes': nodes}
            item['representation'] = f'folded plate fitted to this board\'s photo support ({len(nodes)} face{"s" if len(nodes) > 1 else ""}); photo texture'
            item['guardPlates'] = row
        else:
            row['reason'] = row.get('reason') or f'plate silhouette mean IoU {mean:.3f} (RecGen part {previous:.3f}) fails the rule: the RecGen part stays'
        record['boards'].append(row)
    if scene.geometry:
        (root / FILE).write_bytes(scene.export(file_type='glb'))
    record['status'] = 'plates' if all(b.get('kept') for b in record['boards']) else 'partly plates' if any(b.get('kept') for b in record['boards']) else 'kept RecGen'
    catalog['coverage']['guardPlates'] = record
    (root / 'objects.json').write_text(json.dumps(catalog, ensure_ascii=False, indent=2, allow_nan=False) + ('\n' if text.endswith('\n') else ''))
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--sources', type=Path, nargs='+', required=True, help='original photos of the scene, in photo order')
    args = parser.parse_args()
    print(json.dumps(build(args.root, args.sources), indent=2))
