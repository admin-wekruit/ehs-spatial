"""CPU check of scripts/workcell_recgen_objects.py: input building and placement read-back on a synthetic run (a small red
box seen by two cameras, a third camera looking away); with --out also the outputs of a real call (records, GLBs read
back against their recorded IoU, overlays, outcome, spend ledger), rebuilding the targets from --run.

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:scripts python scripts/check_workcell_recgen_objects.py [--out OUT --run RUN]
"""
import argparse
import base64
import gzip
import json
from pathlib import Path
import sys
import tempfile

import cv2
import numpy as np
from PIL import Image
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fast_report import x7
from scripts import workcell_recgen_objects as objects

W, H, FINE = 60, 80, 4                      # canonical grid; the "original" photos are 4x finer
K = np.array([[300., 0, 29.5], [0, 300., 39.5], [0, 0, 1]])
A = np.array([[1 / FINE, 0, .5 / FINE - .5], [0, 1 / FINE, .5 / FINE - .5], [0, 0, 1]])  # original -> canonical centres
CENTRE, EXTENTS = np.array([0., 0., 3.]), np.array([.1, .3, .1])
BOX = trimesh.creation.box(extents=EXTENTS).subdivide().subdivide().subdivide()
BOX.apply_translation(CENTRE)
WALL = trimesh.creation.box(extents=[8., 8., .1])
WALL.apply_translation([0, 0, 4.25])


def _look(eye, target):
    """OpenCV camera (x right, y down, z forward) at eye looking at target; world y points down."""
    z = (np.asarray(target, float) - eye) / np.linalg.norm(np.asarray(target, float) - eye)
    x = np.cross([0., 1., 0.], z)
    x /= np.linalg.norm(x)
    c2w = np.eye(4)
    c2w[:3, :3], c2w[:3, 3] = np.c_[x, np.cross(z, x), z], eye
    return c2w


CAMERAS = {1: _look(np.zeros(3), CENTRE), 2: _look(np.array([-.9, -.2, .3]), CENTRE), 3: _look(np.zeros(3), [5., 0, 0])}


def _cast(c2w, k, w, h):
    import open3d as o3d
    scene = o3d.t.geometry.RaycastingScene()
    box = scene.add_triangles(o3d.core.Tensor(BOX.vertices.astype(np.float32)), o3d.core.Tensor(BOX.faces.astype(np.uint32)))
    scene.add_triangles(o3d.core.Tensor(WALL.vertices.astype(np.float32)), o3d.core.Tensor(WALL.faces.astype(np.uint32)))
    r = x7.rays(k, c2w, w, h)
    hit = scene.cast_rays(o3d.core.Tensor(r.astype(np.float32)))
    t = hit['t_hit'].numpy()
    points = r[..., :3] + np.where(np.isfinite(t), t, np.nan)[..., None] * r[..., 3:]
    return points, (hit['geometry_ids'].numpy() == box) & np.isfinite(t)


def _encode(a):
    a = np.ascontiguousarray(a)
    return {'data': base64.b64encode(a.tobytes()).decode(), 'dtype': str(a.dtype), 'shape': list(a.shape)}


def _rle(mask):
    flat = mask.flatten(order='F').astype(np.int8)
    counts = np.diff(np.concatenate(([0], np.flatnonzero(np.diff(flat)) + 1, [flat.size]))).tolist()
    return json.dumps({'size': list(mask.shape), 'counts': ([0] if flat[0] else []) + counts})


def synthetic_run(root):
    """A run directory in the oneshot's formats; returns the original photo paths (PNG: lossless)."""
    sources, results = [], []
    for p, c2w in CAMERAS.items():
        points, box = _cast(c2w, np.linalg.inv(A) @ K, W * FINE, H * FINE)
        stripes = (np.nan_to_num(points[..., 1]) * 60).astype(int) % 2
        rgb = np.where(box[..., None], np.stack([200 + 0 * stripes, 40 + 80 * stripes, 30 + 0 * stripes], -1),
                       np.stack([110 + 40 * stripes] * 3, -1)).astype(np.uint8)
        rgb[~np.isfinite(points).all(2)] = 20
        Image.fromarray(rgb).save(root / f'source-{p}.png')
        sources.append(root / f'source-{p}.png')
        canonical, _ = _cast(c2w, K, W, H)
        frame = {'image': _encode(cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA)), 'pts3d': _encode(canonical.astype(np.float32)),
                 'non_ambiguous_mask': _encode(np.isfinite(canonical).all(2)), 'camera_poses': _encode(c2w.astype(np.float32)),
                 'intrinsics': _encode(K.astype(np.float32)), 'original_image': {'width': W * FINE, 'height': H * FINE},
                 'input_mask_transform': {'resized_shape_hw': [H, W], 'crop_xyxy': [0, 0, W, H], 'input_to_canonical_pixel_centres': A.tolist()}}
        with gzip.open(root / f'frame_{p:04d}.json.gz', 'wt') as stream:
            json.dump(frame, stream)
        results.append([{'rle': [_rle(box)], 'scores': [.9]} if box.any() else {'rle': [], 'scores': []}])
    (root / 'sam3.json').write_text(json.dumps({'prompts': [{'text': 'signal light'}], 'results': results}))
    (root / 'geometry.json').write_text(json.dumps({'floor': {'normal': [0, -1, 0], 'offset': 1.5, 'residualP95Native': .01}}))
    (root / 'objects.json').write_text(json.dumps({'objects': [{
        'id': 'lamp-1', 'kind': 'signal light', 'model': {'file': 'object-proxies.glb', 'nodes': ['lamp-1']}, 'representation': 'proxy: box',
        'observations': [{'photo': p, 'source': 'SAM: signal light; instance 0'} for p in (1, 2)]}]}))
    return sources


def check_inputs(root, sources):
    from scripts.workcell_extra_models import _clean
    from workcell_recgen_worker import validate_plan
    [job] = objects.build_inputs(root, [3, 1, 2], ['lamp-1'], sources=sources, scene='synthetic')
    [canonical] = objects.build_inputs(root, [1, 2, 3], ['lamp-1'], scene='canonical')
    record, arrays = job['record'], job['arrays']
    assert record['kind'] == 'synthetic-lamp-1' and record['scenePhotos'] == [1, 2, 3] and record['observedPhotos'] == [1, 2]
    frames = objects._frames(root, [1, 2, 3])
    geometry, sam = (json.loads((root / n).read_text()) for n in ('geometry.json', 'sam3.json'))
    observation = {'source': 'SAM: signal light; instance 0'}
    for p in (1, 2):
        _, _, support, pixels = _clean({**observation, 'photo': p}, frames[p], geometry, sam, None)
        cleaned = np.zeros_like(support)
        cleaned[pixels[:, 1].astype(int), pixels[:, 0].astype(int)] = True
        s = record['viewInputs'][str(p)]['gridScale']
        assert s == FINE and 0 < cleaned.sum() < support.sum(), (s, cleaned.sum(), support.sum())  # small object: finest grid; eroded
        assert np.array_equal(arrays[f'v{p}_rgb'], np.asarray(Image.open(sources[p - 1])))  # 4 x canonical = the original
        assert np.array_equal(arrays[f'v{p}_mask'], np.repeat(np.repeat(cleaned, s, 0), s, 1).astype(np.uint8) * 255)
        assert np.array_equal(arrays[f't{p}_mask'] > 0, support) and record['viewInputs'][str(p)]['cleanedPixels'] == int(cleaned.sum())
        assert np.array_equal(arrays[f'v{p}_depth'], np.repeat(np.repeat(arrays[f't{p}_depth'], s, 0), s, 1))
        z = ((frames[p]['points'] - CAMERAS[p][:3, 3]) @ CAMERAS[p][:3, :3])[..., 2]
        assert np.allclose(arrays[f't{p}_depth'][support], z[support], atol=1e-5) and (arrays[f't{p}_depth'][~frames[p]['valid']] == 0).all()
        X = np.r_[CENTRE + [.03, -.1, -.05], 1] @ np.linalg.inv(CAMERAS[p]).T
        uv, fine = (K @ X[:3])[:2] / X[2], (arrays[f'v{p}_K'] @ X[:3])[:2] / X[2]
        assert np.allclose(fine, s * (uv + .5) - .5)  # the finer K keeps pixel centres
        assert np.array_equal(canonical['arrays'][f'v{p}_rgb'], frames[p]['rgb']) and canonical['record']['viewInputs'][str(p)]['gridScale'] == 1
    assert not arrays['t3_mask'].any() and 'v3_rgb' not in arrays
    counts = [record['viewInputs'][str(p)]['cleanedPixels'] for p in record['views']]
    assert counts == sorted(counts, reverse=True) and sorted(record['views']) == [1, 2]  # most support first
    assert abs(record['supportVerticalExtentNative'] - EXTENTS[1]) < .05, record['supportVerticalExtentNative']
    [[plan]] = objects.plans([record], gpus=2)
    assert plan == {'kind': record['kind'], 'source': record['input'], 'target': record['kind'], 'groups': [[objects.GROUP, record['views']]]}
    np.savez_compressed(root / record['input'], **arrays)
    (root / 'plan.json').write_text(json.dumps([plan]))
    [prepared] = validate_plan(root / 'plan.json')  # the worker's own preflight accepts the npz
    assert sorted(prepared['views']) == [1, 2]
    return record


def check_placement(root, record):
    """A worker mesh in the anchor camera, 8 % too large, 4 degrees and 2 cm off: place() must recover the box."""
    anchor = record['views'][0]
    angle = np.radians(4)
    rotation = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
    world = (BOX.vertices - CENTRE) @ (1.08 * rotation).T + CENTRE + [.015, -.01, .02]
    c2w = CAMERAS[anchor]
    np.savez_compressed(root / f"{record['kind']}-{objects.GROUP}.npz", vertices=((world - c2w[:3, 3]) @ c2w[:3, :3]).astype(np.float32),
                        faces=BOX.faces.astype(np.uint32), colors=np.tile([[200, 40, 30]], (len(world), 1)).astype(np.uint8))
    placed = objects.place(root, dict(record), {'generationSeconds': 1.5, 'stages': {}}, 3.)
    checks = placed['sourceChecks']
    assert placed['refine']['moved'] and placed['refine']['targets'] == [1, 2], placed['refine']
    for p, before in zip((1, 2), placed['refine']['iou_before']):
        assert checks[str(p)]['iou'] >= .85 and checks[str(p)]['iou'] > before and checks[str(p)]['depthP50Relative'] <= .01, checks
    assert checks['3'] == {'observed': False, 'visibleModelPixels': 0}, checks['3']
    assert abs(placed['scale'] - 1 / 1.08) < .03 and placed['accepted'] and not placed['reasons'], (placed['scale'], placed['reasons'])
    assert placed['principalAxisTiltDeg'] < 6, placed['principalAxisTiltDeg']  # an upright box, 4 degrees off before the fit
    assert placed['generationSeconds'] == 1.5 and placed['modelLoadSeconds'] == 3. and placed['decision'].startswith('accepted')
    glb = trimesh.load(root / placed['glb'], force='mesh', process=False)
    assert len(glb.faces) == len(BOX.faces) and np.asarray(glb.visual.vertex_colors)[:, :3].tolist()[0] == [200, 40, 30]
    transform = np.asarray(placed['transform'])
    assert np.abs(np.asarray(glb.vertices) - (world @ transform[:3, :3].T + transform[:3, 3])).max() < 1e-5  # exact read-back
    # Near the true box: IoU saturates at 1.0 on a 10 x 30 px silhouette and refine's loss sees depth only as a median
    # relative error (0.7 % = 2 cm at 3 m here), so the residual sits along the viewing depth: 3 canonical px.
    error = np.abs(np.asarray(glb.vertices) - CENTRE) - EXTENTS / 2
    assert np.abs(error.max(1)).max() < .03, np.abs(error.max(1)).max()
    assert json.loads((root / f"{record['kind']}.json").read_text())['sourceChecks'] == checks
    readback = readback_iou(root / placed['glb'], {p: {k: np.load(root / record['input'])[f't{p}_{k}'] for k in ('mask', 'depth', 'K', 'c2w')} for p in (1, 2, 3)})
    assert all(abs(readback[p] - checks[str(p)]['iou']) < 2e-3 for p in (1, 2)), (readback, checks)
    bad = dict(record, supportVerticalExtentNative=1.)  # a model 0.3 tall against 1.0 of visible support: mis-scaled
    assert not objects.place(root, bad)['accepted']
    # A lamp leaning 15 degrees along the line of sight: stood upright (kept: both views still pass); other kinds are not.
    lean = np.radians(15)
    tilt = np.array([[1, 0, 0], [0, np.cos(lean), -np.sin(lean)], [0, np.sin(lean), np.cos(lean)]])
    world = (BOX.vertices - CENTRE) @ tilt.T + CENTRE
    np.savez_compressed(root / f"{record['kind']}-{objects.GROUP}.npz", vertices=((world - c2w[:3, 3]) @ c2w[:3, :3]).astype(np.float32),
                        faces=BOX.faces.astype(np.uint32), colors=np.tile([[200, 40, 30]], (len(world), 1)).astype(np.uint8))
    stood, free = objects.place(root, dict(record)), objects.place(root, dict(record, category='sign'))
    assert 'upright' not in free and free['principalAxisTiltDeg'] > 5, free['principalAxisTiltDeg']
    assert stood['upright']['kept'] and stood['upright']['freeTiltDeg'] == free['principalAxisTiltDeg'], stood['upright']
    assert stood['principalAxisTiltDeg'] < 1 and stood['accepted'], (stood['principalAxisTiltDeg'], stood['reasons'])
    assert all(stood['sourceChecks'][str(p)]['iou'] >= .85 for p in (1, 2)), stood['sourceChecks']
    # The upright pose must pass the whole rule: one that fails the vertical extent never replaces a passing free pose.
    lo = objects.ACCEPT['verticalExtentRatio'][0]
    support = (stood['modelVerticalExtentNative'] + free['modelVerticalExtentNative']) / 2 / lo
    narrow = objects.place(root, dict(record, supportVerticalExtentNative=support))
    assert not narrow['upright']['kept'] and narrow['accepted'] and narrow['principalAxisTiltDeg'] == free['principalAxisTiltDeg'], narrow['upright']
    # A squat model has no long axis to stand up: left as fitted, with the reason.
    squat = trimesh.creation.box(extents=[.1, .11, .1]).subdivide().subdivide()
    world = squat.vertices + CENTRE
    np.savez_compressed(root / f"{record['kind']}-{objects.GROUP}.npz", vertices=((world - c2w[:3, 3]) @ c2w[:3, :3]).astype(np.float32),
                        faces=squat.faces.astype(np.uint32), colors=np.tile([[200, 40, 30]], (len(world), 1)).astype(np.uint8))
    assert 'no defined long axis' in objects.place(root, dict(record))['upright']['reason']
    try:
        x7.refine(BOX.vertices, BOX.faces, [], axis=[0, -1, 0])
    except ValueError:
        pass
    else:
        raise AssertionError('axis without uniform_scale must be refused')


def check_failure_isolation(root):
    """A lamp failure (inputs or a GPU worker) keeps every proxy, kills the other worker and never fails the oneshot."""
    from modal_apps import workcell_photo_all as oneshot
    catalog = json.loads((root / 'objects.json').read_text())
    catalog['coverage'] = {}
    before = json.dumps(catalog['objects'], sort_keys=True)

    class Worker:
        killed = False
        def poll(self):
            return None
        def kill(self):
            Worker.killed = True
    saved = objects.build_inputs, oneshot._start, oneshot._finish
    try:
        objects.build_inputs = lambda *a, **k: (_ for _ in ()).throw(RuntimeError('inputs broke'))
        out = oneshot._small_objects(root, json.loads(json.dumps(catalog)), 3, [])
        assert out['coverage']['smallObjectModels']['status'] == 'failed' and 'inputs broke' in out['coverage']['smallObjectModels']['error']
        assert json.dumps(out['objects'], sort_keys=True) == before
        objects.build_inputs = saved[0]
        oneshot._start = lambda command, gpu, **k: Worker()
        oneshot._finish = lambda proc, label, timeout=1200: (_ for _ in ()).throw(RuntimeError(f'{label} failed: OOM'))
        out = oneshot._small_objects(root, json.loads(json.dumps(catalog)), 3, [root / f'source-{p}.png' for p in (1, 2, 3)])
        assert out['coverage']['smallObjectModels']['status'] == 'failed' and 'OOM' in out['coverage']['smallObjectModels']['error']
        assert json.dumps(out['objects'], sort_keys=True) == before and Worker.killed, 'proxies kept; the running worker is stopped'
    finally:
        objects.build_inputs, oneshot._start, oneshot._finish = saved


def readback_iou(glb, views):
    """IoU of an exported GLB (already in the native world) with each view's target: the placement as delivered."""
    mesh = trimesh.load(glb, force='mesh')
    caster = x7.Caster(np.asarray(mesh.vertices), np.asarray(mesh.faces))
    out = {}
    for p, v in views.items():
        if v['mask'].any():
            view = {'rays': x7.rays(v['K'], v['c2w'], v['depth'].shape[1], v['depth'].shape[0]), 'target': v['mask'] > 0, 'depth': v['depth']}
            out[p] = x7.score_view(caster, np.eye(4), view)['iou']
    return out


def check_run(out, run):
    """A real call's outputs: every object record, GLB, overlay and ledger entry consistent; IoU read back from the GLBs."""
    out = Path(out)
    outcome = json.loads((out / 'outcome.json').read_text())
    ledger = json.loads((out / 'spend-ledger.json').read_text())
    rate = ledger['usdPerSecond']
    assert abs(rate - .0017752) < 1e-9 and ledger['actualBilledUsd'] is None and ledger['calls']
    for call in ledger['calls']:
        assert call['status'] in ('completed', 'failed') and str(call.get('appId') or '').startswith('ap-'), call
        for key in ('functionSeconds', 'callSeconds'):
            usd = call[key.replace('Seconds', 'WindowEstimateUsd')]
            assert (call[key] is None and usd is None) or abs(usd - round(call[key] * rate, 4)) < 1e-9, call
    for key in ('functionSeconds', 'callSeconds', 'functionWindowEstimateUsd', 'callWindowEstimateUsd'):
        assert abs(ledger['totals'][key] - round(sum(c.get(key) or 0 for c in ledger['calls']), 4)) < 1e-6, key
    targets = {}
    for spec in outcome['scenes']:
        for job in objects.build_inputs(run, spec['photos'], spec['objectIds'], scene=spec['scene']):
            targets[job['record']['objectId'], spec['scene']] = job['arrays']
    lo, hi = objects.ACCEPT['verticalExtentRatio']
    for row in outcome['objects']:
        if not row['glb']:
            assert row['decision'].startswith('fallback'), row
            continue
        record = json.loads((out / f"{row['kind']}.json").read_text())
        assert set(record['sourceChecks']) == {str(p) for p in record['scenePhotos']} and (out / f"{row['kind']}-overlay.jpg").is_file()
        passed = all(record['sourceChecks'][str(p)]['iou'] >= objects.ACCEPT['iou'] and record['sourceChecks'][str(p)]['depthP50Relative'] is not None
                     and record['sourceChecks'][str(p)]['depthP50Relative'] <= objects.ACCEPT['depthP50Relative'] for p in record['observedPhotos'])
        passed &= record['verticalExtentRatio'] is not None and lo <= record['verticalExtentRatio'] <= hi
        assert record['accepted'] == passed == (not record['reasons']) and record['decision'].startswith('accepted' if passed else 'fallback'), row['kind']
        mesh = trimesh.load(out / record['glb'], force='mesh')
        assert len(mesh.faces) == record['faces']['exported'] and np.asarray(mesh.visual.vertex_colors).shape == (len(mesh.vertices), 4)
        scene = row['scene'].removesuffix('-canonical')
        arrays = targets[record['objectId'], scene]
        readback = readback_iou(out / record['glb'], {p: {k: arrays[f't{p}_{k}'] for k in ('mask', 'depth', 'K', 'c2w')} for p in record['scenePhotos']})
        for p, iou in readback.items():
            assert abs(iou - record['sourceChecks'][str(p)]['iou']) < 5e-3, (row['kind'], p, iou, record['sourceChecks'][str(p)])
    print(f"workcell_recgen_objects: {len(outcome['objects'])} objects, {len(ledger['calls'])} ledger calls and GLB read-back consistent in {out}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', type=Path, help='output directory of a real call (outcome.json, spend-ledger.json)')
    parser.add_argument('--run', type=Path, help='the run directory that call read (targets are rebuilt from it)')
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='recgen-objects-check-') as directory:
        root = Path(directory)
        record = check_inputs(root, synthetic_run(root))
        check_placement(root, record)
        check_failure_isolation(root)
    print('workcell_recgen_objects: synthetic input building, worker preflight, placement read-back and lamp failure isolation passed')
    if args.out:
        if not args.run:
            parser.error('--out needs --run')
        check_run(args.out, args.run)
