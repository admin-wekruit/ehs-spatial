"""RecGen models for small catalog objects (lamps, signs, buttons) from all views of ONE scene, placed by a multi-view fit.

Scene-agnostic: a run directory (frame_*.json.gz, sam3.json, geometry.json, objects.json), the photos of one scene and
catalog object ids. Per object:

  inputs  build_inputs: one npz in scripts/workcell_recgen_worker.py's format, v{photo}_rgb/depth/mask/K/c2w for every
          scene photo where the catalog observes the object. Full frames, no crop (RecGen crops itself); mask = the
          cleaned objects-stage support of the object's observations in that photo (workcell_extra_models._clean:
          support_mask eroded 1 px, camera depth within 3 MAD); depth = the frame's camera depth (0 where invalid or
          >= 30, as the oneshot's inputs). With the original photos a view is resampled from the original on s x the
          canonical grid, s = ceil(MIN_SIDE / support long side) <= MAX_SCALE: a 13 x 40 px lamp otherwise reaches RecGen
          as a 3x blow-up of its 172 px minimum crop and loses a third of its width to RecGen's 5 px mask erosion.
          Depth and mask are nearest-upsampled, K follows the finer grid. t{photo}_mask/depth/K/c2w/A for EVERY scene
          photo on the canonical grid: the placement target (the un-eroded support; empty where not observed).
  plan    plans: one group per object with all its views, most cleaned support first (RecGen's SLAT conditions on the
          first pair, its SS stage on every anchor pair), objects dealt across the GPUs.
  place   place: the worker mesh from the anchor camera into the run's native world, x7.light, x7.refine(uniform_scale=
          True) against every observed view, upright kinds stood on the floor normal (stand_upright), x7.score_view
          per scene photo; {kind}.glb (the light copy, vertex colours)
          and {kind}.json (views, IoU and depth error per view, scale, seconds, accepted or proxy fallback).
  review  overlay: the placed model projected into each original photo (CPU raster, no WebGL).

Generated models are display layers placed to photo evidence, never measurements; nothing here is surveyed physical
accuracy. RecGen: internal research licence.
"""
import io
import json
from pathlib import Path
import re
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

GROUP = 'model'
MIN_SIDE, MAX_SCALE = 144, 4  # RecGen's crop side is max(2.4 x half the mask's long side, 518 / 3): >= 144 px the object sets it
MIN_PIXELS = 30               # cleaned canonical pixels with depth for a view to condition RecGen
# verticalExtentRatio: model height over the visible support's (both 2-98 %). Set from the 2026-10-04 lamp review (8 models,
# no held-out set): the two visibly broken canonical-grid controls the IoU rule passed sat at 1.43 and 1.50, the accepted
# lamps at 1.06-1.19 (the support misses cap rows that SAM leaves out or the frame cuts).
ACCEPT = {'iou': .6, 'depthP50Relative': .05, 'verticalExtentRatio': (.75, 1.33)}
UPRIGHT = ('signal light',)  # catalog kinds mounted upright: stand_upright
ELONGATED = 1.5               # stand_upright needs a long axis: largest principal spread >= 1.5 x the next
SUMMARY_KEYS = ('kind', 'objectId', 'scene', 'control', 'views', 'observedPhotos', 'skippedViews', 'viewInputs', 'sourceChecks',
                'scale', 'verticalExtentRatio', 'principalAxisTiltDeg', 'upright', 'modelExtentNative', 'accepted', 'reasons', 'decision',
                'glb', 'generationSeconds', 'modelLoadSeconds', 'placementSeconds')
BASIS = 'shape-preserving similarity alignment to source masks and estimated depth; not surveyed physical accuracy'


def _frames(root, photos):
    """Frame records as workcell_extra_models.consolidate builds them, plus the oneshot's camera depth."""
    from scripts.workcell_photo_oneshot import _array, _frame
    out = {}
    for p in photos:
        raw = _frame(root, p)
        points = _array(raw['pts3d'])
        pose, K = _array(raw['camera_poses']).astype(float), _array(raw['intrinsics']).astype(float)
        valid = _array(raw['non_ambiguous_mask']).astype(bool) & np.isfinite(points).all(2)
        with np.errstate(invalid='ignore'):
            depth = ((points - pose[:3, 3]) @ pose[:3, :3])[..., 2]
            good = valid & np.isfinite(depth) & (depth > 0) & (depth < 30)  # RecGen reads depth > 30 as millimetres
        out[p] = {'raw': raw, 'points': points, 'rgb': _array(raw['image']), 'valid': valid, 'pose': pose, 'K': K,
                  'depth': np.where(good, depth, 0).astype(np.float32)}
    return out


def _source_rgb(path, raw, s):
    """The original photo on s x the canonical grid: MapAnything's resize then crop (input_mask_transform), s times finer."""
    t = raw['input_mask_transform']
    (h, w), (x0, y0, x1, y1) = t['resized_shape_hw'], t['crop_xyxy']
    with Image.open(path) as image:
        rgb = np.asarray(ImageOps.exif_transpose(image).convert('RGB'))
    H, W = rgb.shape[:2]
    if [H, W] != [raw['original_image']['height'], raw['original_image']['width']]:
        raise ValueError(f'{path}: original size differs from the frame record')
    expected = [[w / W, 0, .5 * w / W - .5 - x0], [0, h / H, .5 * h / H - .5 - y0]]
    if not np.allclose(np.asarray(t['input_to_canonical_pixel_centres'], float)[:2], expected, atol=1e-6):
        raise ValueError('input_mask_transform is not the recorded resize + crop')
    return np.ascontiguousarray(cv2.resize(rgb, (w * s, h * s), interpolation=cv2.INTER_AREA)[y0 * s:y1 * s, x0 * s:x1 * s])


def finer_K(K, s):
    """K for s x the grid, pixel centres kept: u' = s (u + 0.5) - 0.5."""
    K = np.asarray(K, float).copy()
    K[:2, :2] *= s
    K[:2, 2] = s * (K[:2, 2] + .5) - .5
    return K


def _up(a, s):
    return np.repeat(np.repeat(a, s, 0), s, 1) if s > 1 else a


def build_inputs(root, photos, object_ids, sources=None, scene=None):
    """-> [{'record', 'arrays'}] per object: the worker npz content and the record place() completes.

    photos: the photos of ONE scene; sources: the run's original photos (sources[photo - 1]) or None (canonical frames)."""
    from scripts.workcell_extra_models import _clean
    root, photos = Path(root), sorted({int(p) for p in photos})
    scene = scene or 'photos-' + '-'.join(map(str, photos))
    catalog = json.loads((root / 'objects.json').read_text())
    geometry = json.loads((root / 'geometry.json').read_text())
    segmentation = json.loads((root / 'sam3.json').read_text())
    items = {item['id']: item for item in catalog['objects']}
    frames = _frames(root, photos)
    loaded = {}

    def inputs(name):
        if name not in loaded:
            loaded[name] = np.load(root / name)
        return loaded[name]
    up = np.asarray(geometry['floor']['normal'], float)
    up /= np.linalg.norm(up)
    jobs = []
    for ident in object_ids:
        item = items[ident]
        kind = re.sub(r'[^A-Za-z0-9_-]', '-', f'{scene}-{ident}')
        arrays, views, skipped, points = {}, {}, {}, []
        for p in photos:
            f = frames[p]
            arrays.update({f't{p}_depth': f['depth'], f't{p}_K': f['K'], f't{p}_c2w': f['pose'],
                           f't{p}_A': np.asarray(f['raw']['input_mask_transform']['input_to_canonical_pixel_centres'], float)})
            support, cleaned, used = np.zeros(f['valid'].shape, bool), np.zeros(f['valid'].shape, bool), []
            for o in item['observations']:
                if o['photo'] == p:
                    kept, _, mask, pixels = _clean(o, f, geometry, segmentation, inputs)
                    support |= mask
                    cleaned[pixels[:, 1].astype(int), pixels[:, 0].astype(int)] = True
                    points.append(kept)
                    used.append(o['source'])
            arrays[f't{p}_mask'] = support.astype(np.uint8) * 255
            if not used:
                continue
            cleaned &= f['depth'] > 0
            n = int(cleaned.sum())
            ys, xs = np.nonzero(cleaned)
            side = int(max(np.ptp(xs), np.ptp(ys)) + 1) if n else 0
            finest = f['raw']['original_image']['width'] // f['rgb'].shape[1]  # never upsample the original
            s = 1 if sources is None or not n else int(max(1, min(MAX_SCALE, finest, np.ceil(MIN_SIDE / side))))
            mask = _up(cleaned, s)
            # RecGen erodes its mask 5 x 5 before reading depth: a view must keep foreground depth after that.
            if n < MIN_PIXELS or not (cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & (_up(f['depth'], s) > 0)).any():
                skipped[p] = f'{n} cleaned pixels with depth; too few for RecGen (>= {MIN_PIXELS}, nonempty after its 5 px erosion)'
                continue
            arrays.update({f'v{p}_rgb': f['rgb'] if s == 1 else _source_rgb(sources[p - 1], f['raw'], s),
                           f'v{p}_depth': _up(f['depth'], s), f'v{p}_mask': mask.astype(np.uint8) * 255,
                           f'v{p}_K': finer_K(f['K'], s), f'v{p}_c2w': f['pose']})
            views[p] = {'observations': used, 'supportPixels': int(support.sum()), 'cleanedPixels': n,
                        'supportLongSidePx': side, 'gridScale': s, 'rgbShape': list(arrays[f'v{p}_rgb'].shape)}
        heights = np.concatenate(points) @ up if points else np.zeros(0)
        record = {'schemaVersion': 1, 'kind': kind, 'objectId': ident, 'category': item.get('kind'), 'scene': scene,
                  'scenePhotos': photos, 'views': sorted(views, key=lambda p: (-views[p]['cleanedPixels'], p)),
                  'viewInputs': {str(p): v for p, v in views.items()}, 'skippedViews': {str(p): r for p, r in skipped.items()},
                  'observedPhotos': [p for p in photos if arrays[f't{p}_mask'].any()],
                  'supportVerticalExtentNative': float(np.ptp(np.percentile(heights, [2, 98]))) if len(heights) else None,
                  'floorNormal': up.tolist(), 'input': f'{kind}-input.npz',
                  'proxyFallback': item.get('model'), 'proxyRepresentation': item.get('representation')}
        if not views:
            record['decision'] = 'fallback: no view can condition RecGen; keep the existing proxy'
        jobs.append({'record': record, 'arrays': arrays})
    return jobs


def npz_bytes(arrays):
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    return buffer.getvalue()


def plans(records, gpus=2):
    """scripts/workcell_recgen_worker.py plans, objects dealt across the GPUs; paths relative to the plan file's directory."""
    out = [[] for _ in range(gpus)]
    for n, r in enumerate(r for r in records if r['views']):
        out[n % gpus].append({'kind': r['kind'], 'source': r['input'], 'target': r['kind'], 'groups': [[GROUP, r['views']]]})
    return [plan for plan in out if plan]


def _visible(caster, transform, view):
    """score_view's visible model pixels: hidden only where the scene is in front of it outside the mask."""
    pred = caster.depth(transform, view['rays'])
    return np.isfinite(pred) & (pred > 0) & ~((~view['target']) & (view['depth'] > 0) & (pred > view['depth'] * 1.04))


def place(directory, record, stats=None, load_seconds=None):
    """Read the worker mesh back ({kind}-model.npz, anchor camera) and place it; writes {kind}.glb and {kind}.json."""
    import trimesh
    from fast_report import x7
    directory = Path(directory)
    kind, views = record['kind'], record['views']
    started = time.perf_counter()
    with np.load(directory / record['input']) as z, np.load(directory / f'{kind}-{GROUP}.npz') as mesh:
        arrays = {k: z[k] for k in z.files if k.startswith('t') or k == f'v{views[0]}_c2w'}
        raw_faces = int(len(mesh['faces']))
        vertices = x7.transformed(mesh['vertices'].astype(float), arrays[f'v{views[0]}_c2w'].astype(float))
        v, f, c = x7.light(vertices, mesh['faces'].astype(np.int64), mesh['colors'][:, :3].astype(float))
    targets = {}
    for p in record['scenePhotos']:
        depth = arrays[f't{p}_depth']
        targets[p] = {'rays': x7.rays(arrays[f't{p}_K'], arrays[f't{p}_c2w'], depth.shape[1], depth.shape[0]),
                      'target': arrays[f't{p}_mask'] > 0, 'depth': depth}
    observed = record['observedPhotos']
    transform, fit = x7.refine(v, f, [targets[p] for p in observed], uniform_scale=True)
    up = np.asarray(record['floorNormal'], float)
    if record.get('category') in UPRIGHT:
        transform = stand_upright(v, f, [targets[p] for p in observed], transform, up, record)
    caster = x7.Caster(v, f)
    checks = {}
    for p, view in targets.items():
        visible = _visible(caster, transform, view)
        if p in observed:
            score = x7.score_view(caster, transform, view)
            checks[str(p)] = {'iou': round(score['iou'], 4), 'depthP50Relative': None if score['p50'] is None else round(score['p50'], 4),
                              'boundary': round(score['boundary'], 5), 'maskPixels': int(view['target'].sum()),
                              'visibleModelPixels': int(visible.sum()),
                              'targetMedianDepthNative': round(float(np.median(view['depth'][view['target'] & (view['depth'] > 0)])), 4)
                              if (view['target'] & (view['depth'] > 0)).any() else None}
        else:  # not observed in this photo: the model should be out of frame or behind scene depth
            checks[str(p)] = {'observed': False, 'visibleModelPixels': int(visible.sum())}
    placed = x7.transformed(v, transform)
    colors = np.clip(np.rint(c), 0, 255).astype(np.uint8)
    (directory / f'{kind}.glb').write_bytes(trimesh.Trimesh(placed, f, vertex_colors=colors, process=False).export(file_type='glb'))
    vertical = float(np.ptp(np.percentile(placed @ up, [2, 98])))
    ratio = vertical / record['supportVerticalExtentNative'] if record['supportVerticalExtentNative'] else None
    record.update(
        glb=f'{kind}.glb', transform=transform.tolist(), refine={**fit, 'targets': observed},
        sourceChecks=checks, scale=round(float(np.cbrt(np.linalg.det(transform[:3, :3]))), 4),
        modelExtentNative=np.ptp(placed, 0).round(4).tolist(), modelVerticalExtentNative=round(vertical, 4),
        verticalExtentRatio=None if ratio is None else round(ratio, 3), principalAxisTiltDeg=principal_tilt(placed, up),
        faces={'recgen': raw_faces, 'exported': int(len(f))},
        generationSeconds=None if stats is None else stats['generationSeconds'], recgenStages=None if stats is None else stats.get('stages'),
        modelLoadSeconds=load_seconds, placementSeconds=round(time.perf_counter() - started, 3), basis=BASIS)
    decide(record)
    (directory / f'{kind}.json').write_text(json.dumps(record, indent=2))
    return record


def stand_upright(v, f, views, free, up, record):
    """Lamps are mounted upright, and one silhouette cannot see a lean along the line of sight (a 2026-10-04 lamp came out
    21.5 degrees off). Turn the free placement about its centre so its principal axis meets the floor normal, refit with
    turns about the normal only, and keep that when every observed view still passes ACCEPT; else the free one stays."""
    from fast_report import x7
    placed = x7.transformed(v, free)
    centre = placed.mean(0)
    _, spread, basis = np.linalg.svd(placed - centre, full_matrices=False)
    rule = ('principal axis to the floor normal, refit turning about the normal only; kept when every observed view '
            'passes the acceptance rule (IoU, depth, vertical extent)')
    if spread[0] < ELONGATED * spread[1]:  # a squat model has no long axis to stand on the floor normal
        record['upright'] = {'rule': rule, 'freeTiltDeg': principal_tilt(placed, up), 'kept': False,
                             'reason': f'no defined long axis (principal spreads {spread[0]:.3g} vs {spread[1]:.3g})'}
        return free
    n, a = x7.unit(up), x7.unit(basis[0] if basis[0] @ up > 0 else -basis[0])
    k = np.cross(a, n)
    skew = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    stand = np.eye(4)
    stand[:3, :3] = np.eye(3) + skew + skew @ skew / (1 + a @ n)  # a -> n (a . n > 0 by the sign choice)
    stand[:3, 3] = centre - stand[:3, :3] @ centre
    step, fit = x7.refine(x7.transformed(placed, stand), f, views, uniform_scale=True, axis=n)
    upright = step @ stand @ free
    caster = x7.Caster(v, f)
    scores = [x7.score_view(caster, upright, view) for view in views]
    (lo, hi), support = ACCEPT['verticalExtentRatio'], record['supportVerticalExtentNative']
    ratio = float(np.ptp(np.percentile(x7.transformed(v, upright) @ n, [2, 98]))) / support if support else None
    kept = all(s['iou'] >= ACCEPT['iou'] and s['p50'] is not None and s['p50'] <= ACCEPT['depthP50Relative'] for s in scores) \
        and ratio is not None and lo <= ratio <= hi
    record['upright'] = {'rule': rule, 'freeTiltDeg': principal_tilt(placed, up), 'kept': kept, 'refine': fit,
                         'iou': [round(s['iou'], 4) for s in scores], 'verticalExtentRatio': None if ratio is None else round(ratio, 3)}
    return upright if kept else free


def principal_tilt(vertices, up):
    """Degrees between the model's principal axis and the floor normal (meaningful for elongated objects only). Silhouettes
    of near-frontal views barely see a lean along the line of sight; this number shows it."""
    centred = np.asarray(vertices, float) - np.mean(vertices, 0)
    axis = np.linalg.svd(centred, full_matrices=False)[2][0]
    return round(float(np.degrees(np.arccos(np.clip(abs(axis @ up) / np.linalg.norm(up), 0, 1)))), 1)


def decide(record):
    """The acceptance rule on a placed record's numbers: IoU and relative depth error in every observed view, and the
    model's vertical extent against the visible support's (not wildly mis-scaled). Fallback: the existing proxy."""
    checks, (lo, hi), ratio = record['sourceChecks'], ACCEPT['verticalExtentRatio'], record['verticalExtentRatio']
    reasons = [f"photo {p}: IoU {checks[str(p)]['iou']:.3f} < {ACCEPT['iou']}" for p in record['observedPhotos']
               if checks[str(p)]['iou'] < ACCEPT['iou']]
    reasons += [f"photo {p}: relative depth error {checks[str(p)]['depthP50Relative']} > {ACCEPT['depthP50Relative']}"
                for p in record['observedPhotos'] if checks[str(p)]['depthP50Relative'] is None
                or checks[str(p)]['depthP50Relative'] > ACCEPT['depthP50Relative']]
    if ratio is None or not lo <= ratio <= hi:
        reasons.append(f"vertical extent {record['modelVerticalExtentNative']} native is {ratio} x the visible support (outside {lo:.2f}-{hi:.2f})")
    record.update(acceptance=ACCEPT, accepted=not reasons, reasons=reasons,
                  decision='accepted: one RecGen model for this object in this scene' if not reasons
                  else 'fallback: keep the existing proxy (' + json.dumps(record.get('proxyFallback')) + ')')
    return record


def overlay(arrays, record, glb, sources, path, tile=300):
    """Review sheet: per scene photo, the original around the object; left: support (green) and placed model (magenta)
    outlines, right: the model painted over the photo (painter's order, Lambert-shaded vertex colours)."""
    import trimesh
    from fast_report.x7 import crop_rgb
    mesh = trimesh.load(glb, force='mesh')
    V, F = np.asarray(mesh.vertices, float), np.asarray(mesh.faces)
    C = np.asarray(mesh.visual.vertex_colors)[:, :3].astype(float)
    normals = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    rows = []
    for p in record['scenePhotos']:
        A, K, pose = arrays[f't{p}_A'], arrays[f't{p}_K'], arrays[f't{p}_c2w'].astype(float)
        Ko = np.linalg.inv(A) @ K  # canonical -> original pixel centres
        with Image.open(sources[p - 1]) as image:
            photo = np.asarray(ImageOps.exif_transpose(image).convert('RGB'))
        cam = (V - pose[:3, 3]) @ pose[:3, :3]
        uv = (cam @ Ko.T)[:, :2] / np.maximum(cam[:, 2:], 1e-9)
        front = cam[:, 2] > 1e-6
        target = arrays[f't{p}_mask'] > 0
        check = record['sourceChecks'][str(p)]
        H, W = photo.shape[:2]
        inside = front & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
        if target.any():  # the support's box in original pixels
            ys, xs = np.nonzero(target)
            (bx0, by0, _), (bx1, by1, _) = np.array([[xs.min(), ys.min(), 1], [xs.max() + 1, ys.max() + 1, 1]]) @ np.linalg.inv(A).T
        elif inside.any():  # not observed: where the model lands in the frame
            (bx0, by0), (bx1, by1) = uv[inside].min(0), uv[inside].max(0)
        label = (f"photo {p}: IoU {check['iou']:.2f}, depth p50 {check['depthP50Relative']}" if 'iou' in check
                 else f"photo {p}: not observed; {check['visibleModelPixels']} visible model px")
        if not target.any() and not inside.any():
            canvas = np.full((tile // 3, 2 * tile, 3), 40, np.uint8)
            cv2.putText(canvas, label + '; model outside the frame', (6, 30), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1, cv2.LINE_AA)
            rows.append(canvas)
            continue
        side = int(max(64, 2.5 * max(bx1 - bx0, by1 - by0)))
        left, top = int((bx0 + bx1 - side) / 2), int((by0 + by1 - side) / 2)
        k = tile / side
        crop = cv2.resize(crop_rgb(photo, left, top, side), (tile, tile), interpolation=cv2.INTER_AREA)
        to_tile = np.array([[k, 0, -k * left], [0, k, -k * top]])
        shown = cv2.warpAffine(target.astype(np.uint8), to_tile @ np.r_[np.linalg.inv(A)[:2], [[0, 0, 1]]], (tile, tile), flags=cv2.INTER_NEAREST)
        pix = (uv - [left, top]) * k
        render, silhouette = crop.copy(), np.zeros((tile, tile), np.uint8)
        tri = pix[F]
        keep = front[F].all(1) & (tri.max(1) >= 0).all(1) & (tri.min(1) < tile).all(1)
        ray = V[F].mean(1) - pose[:3, 3]
        shade = .35 + .65 * np.abs((normals * ray).sum(1) / np.linalg.norm(ray, axis=1))
        colour = np.clip(C[F].mean(1) * shade[:, None], 0, 255)
        order = np.flatnonzero(keep)[np.argsort(-cam[F[keep], 2].mean(1))]
        for i in order:
            points = np.rint(tri[i] * 4).astype(np.int32)
            cv2.fillConvexPoly(render, points, colour[i].tolist(), cv2.LINE_8, 2)
            cv2.fillConvexPoly(silhouette, points, 1, cv2.LINE_8, 2)
        outline = crop.copy()
        for mask, rgb in ((shown, (0, 255, 0)), (silhouette, (255, 0, 255))):
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(outline, contours, -1, rgb, 1)
        blend = np.where(silhouette[..., None] > 0, (.25 * crop + .75 * render).astype(np.uint8), crop)
        row = np.concatenate([outline, blend], 1)
        cv2.rectangle(row, (0, 0), (2 * tile, 18), (0, 0, 0), -1)
        cv2.putText(row, label, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, .42, (255, 255, 255), 1, cv2.LINE_AA)
        rows.append(row)
    head = np.zeros((22, 2 * tile, 3), np.uint8)
    cv2.putText(head, f"{record['kind']}: {'ACCEPTED' if record.get('accepted') else 'FALLBACK (proxy)'}; scale {record.get('scale')}",
                (4, 15), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 0), 1, cv2.LINE_AA)
    sheet = np.concatenate([head, *rows], 0)
    Image.fromarray(sheet).save(path, quality=88)
    return path


if __name__ == '__main__':
    raise SystemExit('Library stage: modal run modal_apps/workcell_recgen_objects.py (see its docstring); '
                     'check: python scripts/check_workcell_recgen_objects.py')
