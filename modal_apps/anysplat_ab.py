"""A/B research probe (not in any report): AnySplat feed-forward 3DGS on a cell's sparse photos vs the report's Pi3X geometry.

AnySplat (arXiv 2505.23716, MIT) code pinned at InternRobotics/AnySplat@CODE_REV, weights lhjiang/anysplat@WEIGHTS_REV.
One ephemeral A100-80GB call runs every scene twice: 'square448' = upstream process_image (448x448 centre crop, the trained
input) and 'full448' = the whole photo resized to long side 448 (multiples of 14). Locally (numpy) it compares with Pi3X:
camera poses after a similarity alignment of camera centres, and AnySplat's rendered depth (and its depth head) vs Pi3X
z-depth at each input view after one global scale fit.

modal run modal_apps/anysplat_ab.py --scenes SCENES.json --out NEW_DIR
SCENES.json = [{"name", "view", "photosDir", "photoMap" (file: IMAGE_ID=FILE,...), "runDir"}]
"""
import io
import json
from pathlib import Path
import time

import modal
import numpy as np

CODE_REV = '5f5e208a7dd57d52e43ea0d553a95eab526e8775'
WEIGHTS_REV = 'd2e8c343672646041ad4ea518184968f94362f01'
GSPLAT = 'https://github.com/nerfstudio-project/gsplat/releases/download/v1.4.0/gsplat-1.4.0%2Bpt22cu121-cp310-cp310-linux_x86_64.whl'
RATE = .000694 + 8 * .0000131 + 32 * .00000222  # A100-80GB + 8 CPU + 32 GiB list rate (USD/s), same as the Pi3X ledger; not an invoice

app = modal.App('anysplat-ab')
volume = modal.Volume.from_name('anysplat-ab-hf-cache', create_if_missing=True)
image = (modal.Image.debian_slim(python_version='3.10').apt_install('git', 'libgl1', 'libglib2.0-0')
         .pip_install('torch==2.2.0', 'torchvision==0.17.0', index_url='https://download.pytorch.org/whl/cu121')
         .pip_install('numpy==1.25.0', 'xformers==0.0.24', 'torch_scatter==2.1.2', GSPLAT, 'huggingface_hub==0.34.4', 'safetensors==0.4.5',
                      'einops==0.8.0', 'jaxtyping==0.2.36', 'omegaconf==2.3.0', 'scipy==1.11.4', 'matplotlib==3.8.4', 'e3nn==0.5.1',
                      'colorspacious==1.1.2', 'Pillow==10.4.0', 'dacite==1.8.1', 'imageio==2.34.2', 'lightning==2.2.5',
                      'scikit-video==1.1.11', 'tqdm==4.66.5', 'opencv-python-headless==4.10.0.84', find_links='https://data.pyg.org/whl/torch-2.2.0+cu121.html')
         # ponytail: upstream builds a VGGT-1B from the hub only to be overwritten by AnySplat's own weights; build it empty
         # instead (5 GB less) and load AnySplat with strict=True so any key the checkpoint lacks fails loudly.
         .run_commands(f'git clone https://github.com/InternRobotics/AnySplat.git /anysplat && cd /anysplat && git checkout {CODE_REV}',
                       """sed -i 's|VGGT.from_pretrained("facebook/VGGT-1B")|VGGT()|' /anysplat/src/model/encoder/anysplat.py""",
                       'grep -q "model_full = VGGT()" /anysplat/src/model/encoder/anysplat.py')
         .env({'HF_HOME': '/cache/huggingface'}))


@app.function(image=image, gpu='A100-80GB', cpu=8, memory=32 * 1024, volumes={'/cache': volume}, timeout=3600, retries=0,
              min_containers=0, max_containers=1)
def infer(scenes: dict) -> dict:
    """scenes: {name: [photo bytes]} -> {name: {variant: npz bytes}} plus timings."""
    import os, sys, tempfile
    start = time.monotonic()
    sys.path.insert(0, '/anysplat'); os.chdir('/anysplat')
    import torch, torchvision
    from PIL import Image, ImageOps
    from src.model.model.anysplat import AnySplat
    from src.utils.image import process_image
    model = AnySplat.from_pretrained('lhjiang/anysplat', revision=WEIGHTS_REV, strict=True).to('cuda').eval()
    model.encoder.distill = False
    volume.commit()
    load_s = time.monotonic() - start

    def inputs(photos, variant):
        tensors, to_photo = [], []
        for data in photos:
            im = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert('RGB'); W, H = im.size
            if variant == 'square448':  # upstream preprocessing, verbatim; its size arithmetic repeated for the pixel map
                path = tempfile.mktemp(suffix='.png'); im.save(path)
                tensors.append((process_image(path) + 1) / 2)
                nw, nh = (int(W * 448 / H), 448) if W > H else (448, int(H * 448 / W))
                left, top = (nw - 448) // 2, (nh - 448) // 2
            else:
                nw, nh = (448, round(448 * H / W / 14) * 14) if W >= H else (round(448 * W / H / 14) * 14, 448)
                tensors.append(torchvision.transforms.ToTensor()(im.resize((nw, nh), Image.BICUBIC))); left = top = 0
            sx, sy = W / nw, H / nh  # input pixel centre (u, v) -> photo pixel centre
            to_photo.append([[sx, 0, (left + .5) * sx - .5], [0, sy, (top + .5) * sy - .5]])
        return torch.stack(tensors)[None].cuda(), np.array(to_photo)

    def once(x):
        v, (h, w) = x.shape[1], x.shape[-2:]
        torch.cuda.synchronize(); t0 = time.monotonic()
        with torch.no_grad():
            enc = model.encoder(x, global_step=0, visualization_dump=None)
            torch.cuda.synchronize(); t1 = time.monotonic()
            pose = enc.pred_context_pose
            dec = model.decoder.forward(enc.gaussians, pose['extrinsic'], pose['intrinsic'], torch.full((1, v), .01, device='cuda'),
                                        torch.full((1, v), 100., device='cuda'), (h, w), 'depth')
            torch.cuda.synchronize(); t2 = time.monotonic()
        return enc, dec, t1 - t0, t2 - t1

    first = next(iter(scenes.values()))
    once(inputs(first, 'square448')[0])  # warm-up (CUDA kernels, gsplat); not timed
    out = {}
    for name, photos in scenes.items():
        out[name] = {}
        for variant in ('square448', 'full448'):
            x, to_photo = inputs(photos, variant)
            torch.cuda.reset_peak_memory_stats()
            enc, dec, enc_s, render_s = once(x)
            f = lambda t: t[0].float().cpu().numpy()
            buffer = io.BytesIO()
            np.savez_compressed(buffer, c2w=f(enc.pred_context_pose['extrinsic']), Kn=f(enc.pred_context_pose['intrinsic']),
                                color=(f(dec.color).transpose(0, 2, 3, 1) * 255).round().astype(np.uint8),
                                depth=f(dec.depth).astype(np.float32), alpha=f(dec.alpha).astype(np.float16),
                                head=f(enc.depth_dict['depth'])[..., 0].astype(np.float32), to_photo=to_photo)
            out[name][variant] = dict(npz=buffer.getvalue(), encoderSeconds=enc_s, renderSeconds=render_s, gaussians=int(enc.gaussians.means.shape[1]),
                                      inputHW=list(x.shape[-2:]), peakGB=torch.cuda.max_memory_allocated() / 1e9)
    return dict(scenes=out, loadSeconds=load_s, containerSeconds=time.monotonic() - start, gpu=torch.cuda.get_device_name(0))


# ---- local comparison (numpy only) ----

def rot_angle(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def align(Ra, Ca, Rp, Cp):
    """Similarity Cp ~ s Q Ca + t. Umeyama on the camera centres when they span a plane; with 2 (or collinear) centres the
    rotation about the baseline is free, so Q = the mean orientation offset (SO3 projection of sum Rp Ra^T) instead."""
    ma, mp = Ca.mean(0), Cp.mean(0); A, B = Ca - ma, Cp - mp
    sv = np.linalg.svd(A, compute_uv=False)
    planar = float(sv[1] / sv[0]) if len(Ca) >= 3 else 0.0
    if planar > .1:
        U, S, Vt = np.linalg.svd(B.T @ A); D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
        Q, s, how = U @ D @ Vt, float((S * np.diag(D)).sum() / (A ** 2).sum()), 'umeyama on camera centres'
    else:
        U, _, Vt = np.linalg.svd(sum(rp @ ra.T for ra, rp in zip(Ra, Rp))); D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
        Q, s, how = U @ D @ Vt, float(np.sqrt((B ** 2).sum() / (A ** 2).sum())), 'centre scale+centroid, mean orientation offset (centres collinear)'
    return s, Q, mp - s * Q @ ma, how, planar


def pose_errors(c2w_a, c2w_p):
    Ra, Ca, Rp, Cp = c2w_a[:, :3, :3], c2w_a[:, :3, 3], c2w_p[:, :3, :3], c2w_p[:, :3, 3]
    s, Q, t, how, planar = align(Ra, Ca, Rp, Cp)
    n = len(Ca); base = np.mean([np.linalg.norm(Cp[i] - Cp[j]) for i in range(n) for j in range(i + 1, n)])
    unit = lambda x: x / np.linalg.norm(x)
    pairs = [dict(i=i, j=j, relRotationErrorDeg=rot_angle((Ra[i].T @ Ra[j]).T @ (Rp[i].T @ Rp[j])),
                  relTranslationDirErrorDeg=float(np.degrees(np.arccos(np.clip(unit(Ra[i].T @ (Ca[j] - Ca[i])) @ unit(Rp[i].T @ (Cp[j] - Cp[i])), -1, 1)))))
             for i in range(n) for j in range(i + 1, n)]
    return dict(alignment=how, centresPlanarity=planar, scale=s,
                rotationErrorDeg=[rot_angle((Q @ Ra[i]).T @ Rp[i]) for i in range(n)],
                centreResidualOverBaseline=[float(np.linalg.norm(s * Q @ Ca[i] + t - Cp[i]) / base) for i in range(n)], pairs=pairs)


def depth_agreement(da, dp):
    """Lists of matched AnySplat / Pi3X depths per view -> one global scale (median ratio), per-view and overall stats."""
    s = float(np.median(np.concatenate([p / a for a, p in zip(da, dp)])))

    def stats(a, p):
        r = (s * a - p) / p
        return dict(n=int(r.size), medianAbsRel=float(np.median(np.abs(r))), within5pct=float((np.abs(r) < .05).mean()),
                    medianSignedRel=float(np.median(r)), viewScaleOverGlobal=float(np.median(p / a) / s))
    a, p = np.concatenate(da), np.concatenate(dp); edges = np.quantile(p, [1 / 3, 2 / 3])
    bins = np.digitize(p, edges)  # Pi3X depth terciles over all views: near / mid / far
    return dict(globalScale=s, perView=[stats(x, y) for x, y in zip(da, dp)], overall=stats(a, p), terciles=dict(
        edges=[float(e) for e in edges], near=stats(a[bins == 0], p[bins == 0]), mid=stats(a[bins == 1], p[bins == 1]), far=stats(a[bins == 2], p[bins == 2])))


def pi3x_reference(cams, run_dir):
    frames = {f['frame_id']: f for f in json.loads((run_dir / 'manifest.json').read_text())['frames']}
    refs = []
    for c in cams:
        fid = c['sourceRefs'][0]['sourceCameraId']; d = run_dir / 'geometry/frames' / fid
        M = np.load(d / 'camera_to_world.npy').astype(np.float64)
        assert np.allclose(M, c['cameraToWorld'], atol=1e-5), f'{fid}: Pi3X frame camera != report camera'
        z = ((np.load(d / 'pts3d.npy') - M[:3, 3]) @ M[:3, :3])[..., 2]
        valid = np.load(d / 'valid_mask.npy') & np.load(d / 'content_valid_mask.npy') & (z > 0)
        refs.append(dict(c2w=np.array(c['cameraToWorld'], float), K=np.array(c['K'], float), W=c['width'], H=c['height'], depth=z, valid=valid,
                         A=np.array(frames[fid]['input_to_canonical_pixel_centres'], float)))
    return refs


def matched(ref, to_photo, shape):
    """Pi3X depth at every AnySplat input pixel (nearest canonical pixel) and its validity."""
    h, w = shape; u, v = np.meshgrid(np.arange(w, dtype=float), np.arange(h, dtype=float))
    x, y = (to_photo[0, 0] * u + to_photo[0, 1] * v + to_photo[0, 2], to_photo[1, 0] * u + to_photo[1, 1] * v + to_photo[1, 2])
    A = ref['A']; ix = np.rint(A[0, 0] * x + A[0, 1] * y + A[0, 2]).astype(int); iy = np.rint(A[1, 0] * x + A[1, 1] * y + A[1, 2]).astype(int)
    hh, ww = ref['depth'].shape; inside = (ix >= 0) & (ix < ww) & (iy >= 0) & (iy < hh)
    ix, iy = np.clip(ix, 0, ww - 1), np.clip(iy, 0, hh - 1)
    return ref['depth'][iy, ix], inside & ref['valid'][iy, ix]


def compare(refs, z):
    n = len(refs); c2w_p = np.stack([r['c2w'] for r in refs])
    poses = pose_errors(z['c2w'].astype(np.float64), c2w_p)
    k = z['Kn']; tp = z['to_photo']; h, w = z['depth'].shape[1:]
    poses['focalRelError'] = [float(k[i, 0, 0] * w * tp[i, 0, 0] / refs[i]['K'][0, 0] - 1) for i in range(n)]
    out, maps = dict(poses=poses), []
    for key in ('depth', 'head'):
        da, dp, cover = [], [], []
        for i in range(n):
            p, ok = matched(refs[i], tp[i], (h, w))
            a = z['depth'][i] / np.maximum(z['alpha'][i].astype(np.float32), 1e-6) if key == 'depth' else z['head'][i]
            ok = ok & (a > 0) & ((z['alpha'][i] >= .5) if key == 'depth' else True)
            da.append(a[ok]); dp.append(p[ok]); cover.append(float(ok.mean()))
            if key == 'depth':
                maps.append((a, p, ok))
        out['renderedDepth' if key == 'depth' else 'headDepth'] = dict(**depth_agreement(da, dp), comparedPixelFraction=cover)
    return out, maps


def montage(z, maps, s):
    import cv2
    rows = []
    for i, (a, p, ok) in enumerate(maps):
        lo, hi = np.percentile(p[ok], [2, 98])
        col = lambda d, m: np.where(m[..., None], cv2.applyColorMap(np.uint8(np.clip((d - lo) / (hi - lo), 0, 1) * 255), cv2.COLORMAP_TURBO), 40)
        err = np.where(ok, np.abs(s * a - p) / np.where(ok, p, 1), 0)
        tiles = [cv2.cvtColor(z['color'][i], cv2.COLOR_RGB2BGR), col(s * a, ok), col(p, ok),
                 np.where(ok[..., None], cv2.applyColorMap(np.uint8(np.clip(err / .2, 0, 1) * 255), cv2.COLORMAP_INFERNO), 40)]
        rows.append(np.hstack([cv2.resize(t.astype(np.uint8), None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA) for t in tiles]))
    return cv2.imencode('.png', np.vstack(rows))[1].tobytes()


def _check():
    rng = np.random.default_rng(0)
    for n in (2, 3):
        Ra = []
        for _ in range(n):
            R = np.linalg.qr(rng.normal(size=(3, 3)))[0]; Ra.append(R if np.linalg.det(R) > 0 else -R)
        Ca = rng.normal(size=(n, 3)); Q = Ra[0] @ Ra[-1].T; s, t = 2.7, rng.normal(size=3)
        a = np.tile(np.eye(4), (n, 1, 1)); p = a.copy()
        a[:, :3, :3], a[:, :3, 3] = Ra, Ca
        p[:, :3, :3], p[:, :3, 3] = [Q @ R for R in Ra], s * Ca @ Q.T + t
        e = pose_errors(a, p)
        assert max(e['rotationErrorDeg']) < 1e-4 and max(e['centreResidualOverBaseline']) < 1e-6 and abs(e['scale'] - s) < 1e-6, e
        assert all(q['relRotationErrorDeg'] < 1e-4 and q['relTranslationDirErrorDeg'] < 1e-4 for q in e['pairs']), e
    dp = [rng.uniform(1, 5, 100), rng.uniform(1, 5, 50)]
    r = depth_agreement([d / 2.5 for d in dp], dp)
    assert abs(r['globalScale'] - 2.5) < 1e-9 and r['overall']['medianAbsRel'] < 1e-9 and r['overall']['within5pct'] == 1.0, r


@app.local_entrypoint()
def main(scenes: str, out: str):
    _check()
    destination = Path(out)
    if destination.exists():
        raise ValueError('Choose a fresh output directory')
    inputs, refs, pi3x = {}, {}, {}
    for s in json.loads(Path(scenes).read_text()):
        cams = json.loads(Path(s['view']).read_bytes())['publication']['snapshot']['revision']['document']['cameras']
        files = dict(item.split('=', 1) for item in Path(s['photoMap']).read_text().strip().split(','))
        inputs[s['name']] = [(Path(s['photosDir']) / files[c['imageId']]).read_bytes() for c in cams]
        run = Path(s['runDir']); refs[s['name']] = pi3x_reference(cams, run)
        ledger = run / 'geometry/spend-ledger.json'
        pi3x[s['name']] = dict(candidateManifest={k: v for k, v in json.loads((run / 'geometry/candidate_manifest.json').read_text()).items()
                                                  if k in ('model_id', 'model_revision', 'code_revision', 'device', 'precision', 'model_load_seconds',
                                                           'inference_and_encoding_seconds')},
                               spendLedger=json.loads(ledger.read_text()) if ledger.exists() else None)
    start = time.monotonic()
    result = infer.remote(inputs)
    call_s = time.monotonic() - start
    destination.mkdir(parents=True)
    report = dict(anysplat=dict(code=f'InternRobotics/AnySplat@{CODE_REV}', weights=f'lhjiang/anysplat@{WEIGHTS_REV}', gpu=result['gpu'],
                                loadSeconds=result['loadSeconds']), pi3xReference=pi3x, scenes={})
    for name, variants in result['scenes'].items():
        report['scenes'][name] = {}
        for variant, v in variants.items():
            (destination / f'{name}-{variant}.npz').write_bytes(v['npz'])
            z = np.load(io.BytesIO(v.pop('npz')))
            metrics, maps = compare(refs[name], z)
            report['scenes'][name][variant] = dict(**v, **metrics)
            if variant == 'full448':
                (destination / f'{name}-full448-depth.png').write_bytes(montage(z, maps, metrics['renderedDepth']['globalScale']))
    spend = {'mode': 'ephemeral modal run', 'hardware': 'A100-80GB, 8 CPU, 32 GiB', 'functionSeconds': result['containerSeconds'],
             'callSeconds': call_s, 'estimateUsd': RATE * result['containerSeconds'], 'actualBilledUsd': None, 'rateSource': 'https://modal.com/pricing'}
    report['spend'] = spend
    (destination / 'results.json').write_text(json.dumps(report, indent=1) + '\n')
    (destination / 'spend-ledger.json').write_text(json.dumps(spend, indent=2) + '\n')
    print(json.dumps(spend))
