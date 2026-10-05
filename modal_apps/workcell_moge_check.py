"""MoGe-3 (Ruicheng/moge-3-vitl, MIT, metric depth) as an independent second opinion on a published four-view report.

(a) Scale. Per photo: MoGe metric camera-z depth / Pi3X native camera-z depth over the pixels Pi3X marks valid (its canonical
    518 grid, mapped into the photo with the run's input_to_canonical_pixel_centres). The median is a MoGe-implied
    nativeToMeters, per photo and pooled per report, compared with the report's e-stop scale. Report only: |deviation| > 10 %
    is flagged; nothing in the report changes.
(b) Object depth. Every displayed model is ray-cast from every photo with all other models as occluders. Inside its visible
    silhouette (and its report mask when that photo has one), MoGe depth converted to native with that photo's ratio is compared
    with the model's depth: median relative residual, + = MoGe sees the surface farther from the camera than the model.

MoGe is given each photo's known horizontal FOV (report K). Ephemeral L4 run; nothing is deployed.

modal run modal_apps/workcell_moge_check.py --view VIEW.json --photos-dir DIR --photo IMAGE_ID=FILE,... --layer-url URL \
    --api ORIGIN --run-dir RUN --out NEW_DIR
"""
import io
import json
from pathlib import Path
import sys
import time

import modal
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from moge3_app import MODEL, REVISION, image as moge_image, volume  # noqa: E402  same MoGe-3 image, cached weights and HF pin

REPO = Path(__file__).resolve().parents[1]
RATE = .000222 + 4 * .0000131 + 16 * .00000222  # L4 + 4 CPU + 16 GiB list rate (USD/s); not an invoice
MAX_SIDE = 2048  # MoGe input raster; depth is sampled back at photo pixel centres
FLAG = .10  # report-only flag on |MoGe scale / e-stop scale - 1|
STRIDE, MIN_PIXELS = 4, 30  # object grid step (photo pixels) and fewest grid pixels for a residual

app = modal.App('workcell-moge-check')
image = (moge_image.apt_install('libgl1', 'libgomp1', 'libx11-6')  # open3d's CPU module links them
         .pip_install('open3d==0.19.0')
         .add_local_file(REPO / 'scripts/workcell_shape_check.py', '/check/shape_core.py')
         .add_local_python_source('moge3_app'))


def core():
    import importlib.util
    path = next(p for p in (Path('/check/shape_core.py'), REPO / 'scripts/workcell_shape_check.py') if p.exists())
    spec = importlib.util.spec_from_file_location('shape_core', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def sample(img, uv):
    """Bilinear at pixel-centre coordinates; nan outside or next to an invalid (nan) pixel. remap maps stay under SHRT_MAX."""
    import cv2
    n = len(uv); m = -(-n // 1024) * 1024 or 1024
    maps = np.full((m, 2), -10, np.float32); maps[:n] = uv
    return cv2.remap(img.astype(np.float32), maps.reshape(-1, 1024, 2), None, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=float('nan')).ravel()[:n]


def stats(x):
    q = np.percentile(x, [25, 50, 75]) if len(x) else [float('nan')] * 3
    return dict(median=float(q[1]), p25=float(q[0]), p75=float(q[2]), n=int(len(x)))


def photo_ratio(moge, f, pi3x, A):
    """MoGe metric / Pi3X native z-depth at each valid canonical pixel; moge is the photo's depth on a raster scaled by f."""
    ys, xs = np.nonzero(np.isfinite(pi3x) & (pi3x > 0))
    photo = np.linalg.solve(np.asarray(A, float), np.vstack([xs, ys, np.ones(len(xs))]))[:2].T
    z = sample(moge, (photo + .5) * f - .5)
    ok = np.isfinite(z) & (z > 0)
    r = z[ok] / pi3x[ys[ok], xs[ok]]
    ratio = np.full(pi3x.shape, np.nan, np.float32); ratio[ys[ok], xs[ok]] = r
    return r, ratio, float(ok.mean()) if len(ok) else 0.0


def depth_trend(ratio, pi3x, bins=10):
    """MoGe/Pi3X ratio against Pi3X depth: medians in depth-decile bins and a line in log-log, ratio = e^a z^b (b = 0: one scale)."""
    ok = np.isfinite(ratio) & np.isfinite(pi3x)
    z, r = np.log(pi3x[ok]), np.log(ratio[ok].astype(float))
    edges = np.percentile(z, np.linspace(0, 100, bins + 1)); idx = np.clip(np.searchsorted(edges, z, 'right') - 1, 0, bins - 1)
    full = [i for i in range(bins) if (idx == i).any()]  # repeated depths leave bins empty
    zc, rc = [np.median(z[idx == i]) for i in full], [np.median(r[idx == i]) for i in full]
    b, a = np.polyfit(zc, rc, 1) if np.ptp(zc) > 0 else (0.0, float(np.median(r)))
    return dict(a=float(a), b=float(b), bins=[dict(zNative=float(np.exp(x)), ratio=float(np.exp(y))) for x, y in zip(zc, rc)])


def object_depths(wsc, cams, objects, moge, f, ratios, pi3x, trends, stride=STRIDE):
    """Per object and photo: model z-depth (ray cast, all models occlude) vs MoGe depth / photo ratio inside the visible
    silhouette, eroded one grid pixel, and inside the report mask when that photo has one. Two context columns on the same
    pixels: MoGe converted with the photo's depth trend instead of one ratio, and Pi3X's own point-map depth."""
    import cv2
    import open3d as o3d
    keep = [i for i, o in enumerate(objects) if len(o['mesh'][1])]
    scene = wsc.raycast_scene([objects[i]['mesh'] for i in keep])  # geometry id g = keep[g]
    rows = []
    for k, cam in enumerate(cams):
        U, Vg = np.meshgrid(np.arange(stride / 2 - .5, cam['w'], stride), np.arange(stride / 2 - .5, cam['h'], stride))
        uv = np.c_[U.ravel(), Vg.ravel()]
        d = (np.c_[uv, np.ones(len(uv))] @ np.linalg.inv(cam['K']).T) @ cam['R']  # world directions, camera z = 1: t = z-depth
        hit = scene.cast_rays(o3d.core.Tensor(np.hstack([np.repeat(cam['C'][None], len(d), 0), d]).astype(np.float32)))
        t, g = hit['t_hit'].numpy().reshape(U.shape), hit['geometry_ids'].numpy().reshape(U.shape)
        metric = sample(moge[k], (uv + .5) * f[k] - .5).reshape(U.shape); zm = metric / ratios[k]
        a, b = trends[k]['a'], trends[k]['b']; zw = (metric * np.exp(-a)) ** (1 / (1 + b))
        A = np.asarray(pi3x[k]['A'], float); zp = sample(pi3x[k]['depth'], uv @ A[:2, :2].T + A[:2, 2]).reshape(U.shape)
        ui, vi = np.clip(U.astype(int), 0, cam['w'] - 1), np.clip(Vg.astype(int), 0, cam['h'] - 1)
        for gid, i in enumerate(keep):
            o = objects[i]; vis = (g == gid) & np.isfinite(t)
            if not vis.any():
                continue
            area = vis.copy()
            if k in o['masks']:
                area &= o['masks'][k][vi, ui]
            area = cv2.erode(area.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & np.isfinite(zm) & (zm > 0)
            rel = zm[area] / t[area] - 1
            other = lambda z: stats(z[area & np.isfinite(z)] / t[area & np.isfinite(z)] - 1)
            rows.append(dict(entityId=o['id'], label=o['label'], photo=k + 1, maskInPhoto=k in o['masks'], silhouettePixels=int(vis.sum()),
                             status='ok' if area.sum() >= MIN_PIXELS else 'too_few_pixels', residual=stats(rel),
                             within5pct=float((np.abs(rel) < .05).mean()) if len(rel) else float('nan'),
                             residualDepthTrend=other(zw), residualPi3x=other(zp),
                             modelDepthNative=float(np.median(t[area])) if len(rel) else float('nan'),
                             mogeDepthNative=float(np.median(zm[area])) if len(rel) else float('nan'), gridStridePx=stride))
    return rows


@app.function(image=image, gpu='L4', cpu=4, memory=16 * 1024, volumes={'/cache': volume}, timeout=1800, retries=0,
              min_containers=0, max_containers=1)
def run(view: bytes, photos: dict, layer_url: str, api: str, pi3x: list) -> dict:
    import cv2
    import torch
    from moge.model.v3 import MoGeModel
    start = time.monotonic()
    wsc = core(); _check()
    ctx = wsc.load_report(view, photos, layer_url, api)
    model = MoGeModel.from_pretrained(MODEL, revision=REVISION).to('cuda').eval()
    moge, f, per_photo, ratio_maps, pooled = [], [], [], {}, []
    for k, cam in enumerate(ctx['cams']):
        bgr = cv2.imdecode(np.frombuffer(photos[cam['imageId']], np.uint8), cv2.IMREAD_COLOR)  # same decode as load_report
        assert bgr.shape[:2] == (cam['h'], cam['w']), (bgr.shape, cam['h'], cam['w'])
        s = min(1.0, MAX_SIDE / max(cam['w'], cam['h'])); size = (round(cam['w'] * s), round(cam['h'] * s))
        rgb = cv2.cvtColor(cv2.resize(bgr, size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        fov = float(np.degrees(2 * np.arctan(cam['w'] / 2 / cam['K'][0, 0])))
        with torch.inference_mode():
            out = model.infer(torch.from_numpy(rgb.copy()).float().permute(2, 0, 1).cuda() / 255, fov_x=fov, use_fp16=True)
        depth, mask = out['depth'].float().cpu().numpy(), out['mask'].cpu().numpy().astype(bool)
        K = out['intrinsics'].float().cpu().numpy()
        moge.append(np.where(mask & np.isfinite(depth) & (depth > 0), depth, np.nan).astype(np.float32)); f.append(size[0] / cam['w'])
        r, ratio_maps[f'photo{k + 1}'], coverage = photo_ratio(moge[-1], f[-1], pi3x[k]['depth'], pi3x[k]['A'])
        pooled.append(r)
        per_photo.append(dict(photo=k + 1, imageId=cam['imageId'], pi3xFrame=pi3x[k]['frame'], fovXDegGiven=fov,
                              depthTrend=depth_trend(ratio_maps[f'photo{k + 1}'], pi3x[k]['depth']),
                              fovXDegMoge=float(np.degrees(2 * np.arctan(.5 / K[0, 0]))), mogeRaster=list(size), mogeValid=float(mask.mean()),
                              pi3xPixelsCompared=coverage, ratio=stats(r)))
    S = ctx['S']; allr = np.concatenate(pooled)
    for p in per_photo:
        p['deviationVsEstop'] = p['ratio']['median'] / S - 1; p['flag'] = abs(p['deviationVsEstop']) > FLAG
    report = dict(pooled=stats(allr), photoMedians=[p['ratio']['median'] for p in per_photo], estopNativeToMeters=S)
    report['deviationVsEstop'] = report['pooled']['median'] / S - 1; report['flag'] = abs(report['deviationVsEstop']) > FLAG
    rows = object_depths(wsc, ctx['cams'], ctx['objects'], moge, f, [p['ratio']['median'] for p in per_photo], pi3x,
                         [p['depthTrend'] for p in per_photo])
    buf = io.BytesIO(); np.savez_compressed(buf, **{k: v.astype(np.float16) for k, v in ratio_maps.items()})
    return dict(model=MODEL, modelRevision=REVISION, scale=dict(report=report, photos=per_photo, flagAbove=FLAG), objects=rows,
                objectMaskPhotos={o['id']: sorted(k + 1 for k in o['masks']) for o in ctx['objects']}, skipped=ctx['skipped'],
                layerRevision=(ctx['layer'] or {}).get('revisionId'), containerSeconds=time.monotonic() - start, ratioMaps=buf.getvalue())


def pi3x_depths(run_dir: Path, doc_cams: list, files: dict) -> list:
    """Pi3X native camera-z depth on its canonical grid (nan = not valid) for each report camera, plus the photo->canonical affine."""
    manifest = json.loads((run_dir / 'manifest.json').read_text())
    frames = {Path(fr['input']).name: (i, fr) for i, fr in enumerate(manifest['frames'])}
    out = []
    for cam in doc_cams:
        i, fr = frames[files[cam['imageId']]]; g = run_dir / 'geometry/frames' / f'frame_{i + 1:04d}'
        M = np.load(g / 'camera_to_world.npy')
        assert np.allclose(M, cam['cameraToWorld'], atol=1e-6), 'report camera is not the Pi3X camera: ' + g.name
        z = ((np.load(g / 'pts3d.npy') - M[:3, 3]) @ M[:3, :3])[..., 2]
        valid = np.load(g / 'valid_mask.npy') & np.load(g / 'content_valid_mask.npy') & (z > 0)
        out.append(dict(depth=np.where(valid, z, np.nan).astype(np.float32), A=fr['input_to_canonical_pixel_centres'], frame=g.name))
    return out


@app.local_entrypoint()
def main(view: str, photos_dir: str, photo: str, layer_url: str, api: str, run_dir: str, out: str):
    destination = Path(out)
    if destination.exists():
        raise ValueError('Choose a fresh output directory')
    _check()
    files = dict(item.split('=', 1) for item in photo.split(','))
    photos = {image_id: (Path(photos_dir) / name).read_bytes() for image_id, name in files.items()}
    view_bytes = Path(view).read_bytes()
    doc_cams = json.loads(view_bytes)['publication']['snapshot']['revision']['document']['cameras']
    start = time.monotonic()
    result = run.remote(view_bytes, photos, layer_url, api, pi3x_depths(Path(run_dir), doc_cams, files))
    destination.mkdir(parents=True)
    (destination / 'ratio-maps.npz').write_bytes(result.pop('ratioMaps'))
    (destination / 'results.json').write_text(json.dumps(result, indent=1, ensure_ascii=False) + '\n')
    ledger = {'mode': 'ephemeral modal run', 'hardware': 'L4, 4 CPU, 16 GiB', 'functionSeconds': result['containerSeconds'],
              'callSeconds': time.monotonic() - start, 'estimateUsd': RATE * result['containerSeconds'], 'actualBilledUsd': None,
              'rateSource': 'https://modal.com/pricing'}
    (destination / 'spend-ledger.json').write_text(json.dumps(ledger, indent=2) + '\n')
    rep = result['scale']['report']
    print(f"MoGe scale {rep['pooled']['median']:.4f} vs e-stop {rep['estopNativeToMeters']:.4f} ({rep['deviationVsEstop']:+.1%})"
          f"{' FLAG' if rep['flag'] else ''}; per photo {[round(x, 4) for x in rep['photoMedians']]}; depth exponent b "
          f"{[round(p['depthTrend']['b'], 3) for p in result['scale']['photos']]}")
    for r in result['objects']:
        print(f"photo {r['photo']} {r['label'][:28]:28s} {r['entityId'][:8]} {r['status']:14s} n {r['residual']['n']:6d} "
              f"residual {r['residual']['median']:+.3f} [{r['residual']['p25']:+.3f},{r['residual']['p75']:+.3f}] trend "
              f"{r['residualDepthTrend']['median']:+.3f} pi3x {r['residualPi3x']['median']:+.3f} mask {r['maskInPhoto']}")
    print(json.dumps(ledger))


def _check():
    """Synthetic: Pi3X sees a wall at z=3 and a box face at z=1.9 (native); MoGe sees both 1.2x farther (metric) and the box
    another 5 % farther. The photo ratio must come back 1.2 and the box residual +5 %."""
    wsc = core()
    w, h, fx = 400, 300, 300.
    K = np.array([[fx, 0, (w - 1) / 2], [0, fx, (h - 1) / 2], [0, 0, 1]])
    cam = wsc.camera(dict(cameraToWorld=np.eye(4).tolist(), K=K.tolist(), width=w, height=h, imageId='a'))
    def native(u, v):  # z-depth at photo pixel centres
        x, y = (u - K[0, 2]) / fx, (v - K[1, 2]) / fx
        box = (np.abs(x * 1.9) <= .25) & (np.abs(y * 1.9) <= .25)
        return np.where(box, 1.9, 3.0), box
    A = np.array([[.5, 0, -.25], [0, .5, -.25], [0, 0, 1]])  # canonical grid = photo / 2, pixel-centre convention
    yc, xc = np.mgrid[0:h // 2, 0:w // 2].astype(float)
    pi3x, _ = native(*(np.linalg.solve(A, np.vstack([xc.ravel(), yc.ravel(), np.ones(xc.size)]))[:2]))
    f = .5; yr, xr = np.mgrid[0:h // 2, 0:w // 2].astype(float)
    z, box = native((xr + .5) / f - .5, (yr + .5) / f - .5); moge = (1.2 * z * np.where(box, 1.05, 1)).astype(np.float32)
    pi3x = pi3x.reshape(xc.shape).astype(np.float32)
    r, _, _ = photo_ratio(moge, f, pi3x, A)
    assert abs(np.median(r) - 1.2) < 1e-3, np.median(r)
    zs = np.linspace(1, 9, 5000); trend = depth_trend(1.3 * zs ** -.2, zs)  # a known power law comes back
    assert abs(trend['b'] + .2) < 1e-3 and abs(np.exp(trend['a']) - 1.3) < 1e-3, trend
    V, F = wsc.primitive_mesh(dict(kind='box', dimensions=[.5, .5, .2])); V = V + [0, 0, 2.0]
    rows = object_depths(wsc, [cam], [dict(id='box', label='box', mesh=(V, F), masks={})], [moge], [f], [float(np.median(r))],
                         [dict(depth=pi3x, A=A)], [dict(a=float(np.log(1.2)), b=0.)])
    assert len(rows) == 1 and rows[0]['status'] == 'ok' and abs(rows[0]['residual']['median'] - .05) < .005, rows
    assert abs(rows[0]['residualDepthTrend']['median'] - .05) < .005 and abs(rows[0]['residualPi3x']['median']) < .005, rows
    print('moge check self-test passed: ratio', round(float(np.median(r)), 4), 'box residual', round(rows[0]['residual']['median'], 4))


if __name__ == '__main__':
    _check()
