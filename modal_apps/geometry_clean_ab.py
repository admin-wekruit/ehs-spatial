"""Licence-clean geometry A/B, scored by the fair harness of geometry-licence-ab-fair-2026-10-05 (2026-10-06).

Candidates (every one on the same frozen canonical frames of both runs; nothing deployed; retries=0, min_containers=0):
  vggt          facebook/VGGT-1B-Commercial, feed-forward on the padded 518 frames (depth head unprojected with its cameras)
  moge          Ruicheng/moge-3-vitl (MIT) per photo on the 392x518 content crop, registered by sim(3) RANSAC on RoMa matches
  da3-base      depth-anything/DA3-BASE (Apache-2.0), the fair A/B's geometry (da3fair-geom), unchanged
  mapanything   facebook/map-anything-apache (Apache-2.0), the first A/B's geometry (geomab-*-mapanything), pixel-identical frames
  pi3x          published Pi3X geometry (CC BY-NC): control only, to separate what the refinement does from what the backbone does
Refinement (same code for every backbone): RoMa (MIT, outdoor weights) dense matches on the content crops -> USAC F-matrix
inliers -> 2-view tracks, 3D initialised from the backbone's own depth -> bundle adjustment (modal_apps/bundle_adjust.py: numpy
Schur Levenberg-Marquardt, the GPL-free copy of the pycolmap call), Cauchy loss, gauge TWO_CAMS_FROM_WORLD, cameras only ('ba')
or cameras + one focal per photo ('ba-f') -> each photo's backbone depth is rescaled by the median ratio of the BA depths at its
observations and re-cast through the BA camera. No depth-prior residual: the backbone depth is the prior through the initial
points and the dense surface it rescales. refine_pycolmap (--ba pycolmap) is the research comparison only: the PyPI pycolmap
4.2.1 wheel statically links GPL-2.0+ SuiteSparse, and pycolmap is imported nowhere else.
RoMa weights: roma_model() loads the mirrored roma_outdoor.pth + dinov2_vitl14_pretrain.pth (Modal volume
'panoptes-geometry-weights' at /weights = scripts/onprem/fetch_weights_geometry.py --cache DIR), SHA-256 checked, passed
explicitly to roma_outdoor(weights=..., dinov2_weights=...): no run-time download.

  modal run modal_apps/geometry_clean_ab.py --stage access    can the 'huggingface' secret read the gated VGGT-1B-Commercial?
  modal run modal_apps/geometry_clean_ab.py --stage infer     L4 x2 in parallel: VGGT + RoMa matches, MoGe-3 -> SCR/checks/clean-*
  modal run modal_apps/geometry_clean_ab.py --stage refine [--ba pycolmap]   CPU map: MoGe sim(3) init + BA for every backbone
  modal run modal_apps/geometry_clean_ab.py --stage dense-infer   L4: RoMa's full warp both ways at every content pixel
  modal run modal_apps/geometry_clean_ab.py --stage mvs --bases B,..   local: two-view triangulation with each B-ba-f camera set
  modal run modal_apps/geometry_clean_ab.py --stage fuse --bases B,..  local: MVS depth, holes = backbone depth x smoothed MVS ratio
  modal run modal_apps/geometry_clean_ab.py --stage analyse   CPU map: fair_ab_modal.analyse_one on every geometry
  python modal_apps/geometry_clean_ab.py check                 local self-test of the refinement (synthetic scene)
"""
from __future__ import annotations

import io
from itertools import combinations
import json
from pathlib import Path
import sys
import tarfile
import time

import modal
import numpy as np

FAIR = Path('/Users/adam/Desktop/panoptes-public/research-notes/geometry-licence-ab-fair-2026-10-05')
sys.path[:0] = [str(Path(__file__).parent), str(FAIR), str(Path(__file__).resolve().parents[1] / 'scripts/onprem')]
import fair_ab_modal as fam  # noqa: E402  frozen inputs, frames_for, geometry_tar, analyse_one, cpu_image (unchanged)
from moge3_app import MODEL as MOGE, REVISION as MOGE_REV, image as moge_base, volume as moge_volume  # noqa: E402

SCR = fam.SCR
GPUD, GEOM, OUT = SCR / 'checks/clean-gpu', SCR / 'checks/clean-geom', SCR / 'checks/clean-analyse'
VGGT, VGGT_REV, VGGT_CODE = 'facebook/VGGT-1B-Commercial', 'ebb29a532abe92960eeb6903a5530f16990ef4ab', 'a288dd0f14786c93483e45524328726ab7b1b4ce'
ROMA_CODE = '77f8d68803526dcddfd9b7a46bc76125bdc25f15'
N_MATCH, F_PX = 10000, 1.0  # RoMa samples per pair; USAC-MAGSAC F-matrix threshold (canonical px)
BA_DROP_PX = 2.0  # points whose mean reprojection error exceeds this after the first BA are removed, then BA again
L4_RATE = .000222 + 4 * .0000131 + 32 * .00000222  # L4 + 4 CPU + 32 GiB list rate (USD/s); not an invoice
CPU_RATE = 8 * .0000131 + 16 * .00000222
INIT = {'da3-base': lambda c: SCR / f'checks/da3fair-geom/{c}-da3-base-padded/geometry',
        'mapanything': lambda c: SCR / f'checks/geomab-{c}-mapanything/geometry',
        'pi3x': lambda c: fam.RUNS / fam.CELLS[c]['run'] / 'geometry',
        'vggt': lambda c: GEOM / f'{c}-vggt/geometry', 'moge': lambda c: GEOM / f'{c}-moge/geometry'}

app = modal.App('geometry-clean-ab')
SOURCES = ('fair_ab_modal', 'moge3_app', 'bundle_adjust', 'fetch_weights_geometry', 'fetch_weights', 'fetch_weights_da3')
WEIGHTS, WEIGHTS_DIR = modal.Volume.from_name('panoptes-geometry-weights'), '/weights'  # only read, SHA-256 checked first; on-prem: the --cache DIR
vr_image = (modal.Image.debian_slim(python_version='3.11')
            .apt_install('git', 'libgl1', 'libglib2.0-0')
            .pip_install('torch==2.5.1', 'torchvision==0.20.1', 'numpy==1.26.4', 'pillow==11.0.0', 'opencv-python-headless==4.10.0.84',
                         'huggingface_hub==0.36.0', 'safetensors==0.4.5', 'einops==0.8.0', 'kornia==0.7.4', 'loguru==0.7.2', 'tqdm==4.67.1')
            .pip_install(f'git+https://github.com/facebookresearch/vggt.git@{VGGT_CODE}', f'git+https://github.com/Parskatt/RoMa.git@{ROMA_CODE}',
                         extra_options='--no-deps')  # ponytail: only the runtime imports; romatch's wandb/h5py/poselib are training-only
            .add_local_python_source(*SOURCES))
moge_image = moge_base.add_local_python_source(*SOURCES)
refine_image = (modal.Image.debian_slim(python_version='3.11')  # = docker/geometry-requirements.txt's numpy / scipy / opencv pins
                .pip_install('numpy==1.26.4', 'scipy==1.14.1', 'opencv-python-headless==4.10.0.84')
                .add_local_file(FAIR / 'fair_ab.py', '/check/fair_ab.py')
                .add_local_python_source(*SOURCES))
pycolmap_image = (modal.Image.debian_slim(python_version='3.11')  # research comparison only (GPL SuiteSparse in the wheel)
                  .pip_install('numpy==2.2.6', 'opencv-python-headless==4.10.0.84', 'pycolmap==4.2.1')
                  .add_local_file(FAIR / 'fair_ab.py', '/check/fair_ab.py')
                  .add_local_python_source(*SOURCES))
analyse_image = fam.cpu_image.add_local_python_source(*SOURCES)
hf = [modal.Secret.from_name('huggingface')]


def png(arr) -> bytes:
    from PIL import Image
    buf = io.BytesIO(); Image.fromarray(arr).save(buf, format='PNG'); return buf.getvalue()


def npz(**arrays) -> bytes:
    buf = io.BytesIO(); np.savez_compressed(buf, **arrays); return buf.getvalue()


def rgb(data: bytes):
    from PIL import Image
    return np.asarray(Image.open(io.BytesIO(data)).convert('RGB'))


# ---------------------------------------------------------------- GPU
@app.function(image=vr_image, cpu=1, memory=2048, timeout=300, retries=0, min_containers=0, secrets=hf)
def access() -> dict:
    from huggingface_hub import hf_hub_download
    try:
        hf_hub_download(VGGT, 'config.json', revision=VGGT_REV, local_dir='/tmp/vggt')
        return dict(repo=VGGT, revision=VGGT_REV, access=True)
    except Exception as error:  # noqa: BLE001 - reported, never the token
        return dict(repo=VGGT, revision=VGGT_REV, access=False, error=type(error).__name__ + ': ' + str(error).splitlines()[0][:300])


@app.function(image=vr_image, gpu='L4', cpu=4, memory=32 * 1024, timeout=1800, retries=0, min_containers=0, secrets=hf,
              volumes={WEIGHTS_DIR: WEIGHTS})
def vggt_roma(inputs: dict, with_vggt: bool) -> dict:
    """inputs = {cell: {'padded': {name: png}, 'unpadded': {name: png}}} (fair_ab_modal.frames_for, the frozen bytes).
    VGGT on the padded frames (only when the gated checkpoint is readable); RoMa on the content crops, matches returned on the
    518 grid (pixel centres at integers)."""
    import torch
    from PIL import Image
    torch.set_float32_matmul_precision('highest'); torch.manual_seed(0)
    start = time.monotonic(); out = {c: {} for c in inputs}; timing = {}
    if with_vggt:
        from huggingface_hub import snapshot_download
        from vggt.models.vggt import VGGT as Model
        from vggt.utils.geometry import unproject_depth_map_to_point_map
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        wdir = snapshot_download(VGGT, revision=VGGT_REV, allow_patterns=['config.json', 'model.safetensors', 'LICENSE'], local_dir='/tmp/vggt')
        model = Model.from_pretrained(wdir).to('cuda').eval(); timing['vggtLoad'] = time.monotonic() - start
        for cell, fr in inputs.items():
            names = sorted(fr['padded']); x = torch.from_numpy(np.stack([rgb(fr['padded'][n]) for n in names])).permute(0, 3, 1, 2).float().div(255).cuda()
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                pred = model(x)
            E, K = pose_encoding_to_extri_intri(pred['pose_enc'], x.shape[-2:])
            E, K = E[0].float().cpu().numpy(), K[0].float().cpu().numpy()
            depth, conf = pred['depth'][0, ..., 0].float().cpu().numpy(), pred['depth_conf'][0].float().cpu().numpy()
            pts = unproject_depth_map_to_point_map(depth[..., None], E, K)
            out[cell]['vggt'] = {n: npz(pts3d=pts[k].astype(np.float32), conf=conf[k].astype(np.float32), valid=np.isfinite(depth[k]) & (depth[k] > 1e-8),
                                        K=K[k].astype(np.float64), c2w=np.linalg.inv(np.r_[E[k], [[0, 0, 0, 1]]])) for k, n in enumerate(names)}
        timing['vggt'] = time.monotonic() - start
        del model; torch.cuda.empty_cache()
    roma = roma_model('cuda')
    for cell, fr in inputs.items():
        names = sorted(fr['unpadded']); ims = [Image.open(io.BytesIO(fr['unpadded'][n])).convert('RGB') for n in names]
        out[cell]['roma'] = {f'{i}-{j}': npz(**roma_pair(roma, ims[i], ims[j], 'cuda', dense=False)['sparse'])
                             for i, j in combinations(range(len(names)), 2)}
    timing['total'] = time.monotonic() - start
    return dict(out=out, timing=timing, gpu=torch.cuda.get_device_name(0), containerSeconds=time.monotonic() - start)


@app.function(image=moge_image, gpu='L4', cpu=4, memory=32 * 1024, volumes={'/cache': moge_volume}, timeout=1800, retries=0,
              min_containers=0)
def moge(inputs: dict) -> dict:
    """inputs = {cell: {name: content-crop png}}: MoGe-3 points in each photo's own camera frame, its mask and intrinsics."""
    import torch
    from moge.model.v3 import MoGeModel
    start = time.monotonic()
    model = MoGeModel.from_pretrained(MOGE, revision=MOGE_REV).to('cuda').eval()
    out = {}
    for cell, frames in inputs.items():
        out[cell] = {}
        for n, data in sorted(frames.items()):
            im = rgb(data)
            with torch.inference_mode():
                o = model.infer(torch.from_numpy(im.copy()).float().permute(2, 0, 1).cuda() / 255, use_fp16=True)
            out[cell][n] = npz(points=o['points'].float().cpu().numpy(), mask=o['mask'].cpu().numpy().astype(bool),
                               intrinsicsNormalised=o['intrinsics'].float().cpu().numpy())
    return dict(out=out, gpu=torch.cuda.get_device_name(0), containerSeconds=time.monotonic() - start)


@app.function(image=vr_image, gpu='L4', cpu=4, memory=32 * 1024, timeout=1800, retries=0, min_containers=0, volumes={WEIGHTS_DIR: WEIGHTS})
def roma_dense(inputs: dict) -> dict:
    """inputs = {cell: {name: content-crop png}}. RoMa's full warp, sampled at every crop pixel centre, both directions:
    uvAB = where A's pixel lands in B (518 grid, pixel centres at integers), certA; uvBA, certB the other way."""
    import torch
    from PIL import Image
    torch.set_float32_matmul_precision('highest'); torch.manual_seed(0)
    start = time.monotonic(); roma = roma_model('cuda'); out = {}
    for cell, frames in inputs.items():
        names = sorted(frames); ims = [Image.open(io.BytesIO(frames[n])).convert('RGB') for n in names]
        out[cell] = {f'{i}-{j}': npz(**roma_pair(roma, ims[i], ims[j], 'cuda', sparse=False)['dense']) for i, j in combinations(range(len(names)), 2)}
    return dict(out=out, gpu=torch.cuda.get_device_name(0), containerSeconds=time.monotonic() - start)


def roma_model(device, weights=WEIGHTS_DIR):
    """RoMa v1 outdoor (romatch@ROMA_CODE, plain correlation: no fused-local-corr CUDA extension) built from the mirrored
    weights, SHA-256 checked and passed explicitly (fetch_weights_geometry.roma_state_dicts): nothing is downloaded."""
    import torch
    from romatch import roma_outdoor
    import fetch_weights_geometry as fwg
    torch.set_float32_matmul_precision('highest')  # romatch refuses anything else
    return roma_outdoor(device=device, use_custom_corr=False, **fwg.roma_state_dicts(Path(weights), device))


def roma_pair(roma, im_a, im_b, device, sparse=True, dense=True) -> dict:
    """One RoMa match of two content crops (PIL, same size). sparse: N_MATCH balanced samples, uvA / uvB on the 518 grid
    (pixel centres at integers) + certainty; dense: the full warp sampled at every crop pixel centre both ways,
    uvAB = where A's pixel lands in B, certA; uvBA, certB the other way."""
    import torch
    import torch.nn.functional as F
    warp, cert = roma.match(im_a, im_b, device=device); W, H = im_a.size; out = {}
    if sparse:
        m, c = roma.sample(warp, cert, num=N_MATCH)
        a, b = roma.to_pixel_coordinates(m, H, W, H, W)  # continuous coords, pixel centres at +0.5
        shift = np.array([fam.X0 - .5, -.5])
        out['sparse'] = dict(uvA=a.cpu().numpy().astype(np.float64) + shift, uvB=b.cpu().numpy().astype(np.float64) + shift,
                             certainty=c.cpu().numpy().astype(np.float32))
    if dense:
        ws = cert.shape[-1] // 2
        ys, xs = torch.meshgrid((torch.arange(H, device=device) + .5) / H * 2 - 1, (torch.arange(W, device=device) + .5) / W * 2 - 1, indexing='ij')
        g = torch.stack((xs, ys), -1)[None].float(); res = {}
        for half, key in ((0, 'AB'), (1, 'BA')):  # left half: grid over A, B coords in [2:4]; right half: grid over B, A coords in [0:2]
            w = warp[0, :, half * ws:(half + 1) * ws]; tgt = w[..., 2:4] if half == 0 else w[..., 0:2]
            uv = F.grid_sample(tgt.permute(2, 0, 1)[None].float(), g, align_corners=False, mode='bilinear')[0].permute(1, 2, 0)
            c = F.grid_sample(cert[:, None, :, half * ws:(half + 1) * ws].float(), g, align_corners=False, mode='bilinear')[0, 0]
            pix = (uv + 1) / 2 * torch.tensor([W, H], device=device) - .5 + torch.tensor([fam.X0, 0.], device=device)
            res[f'uv{key}'] = pix.cpu().numpy().astype(np.float32); res[f'cert{key[0]}'] = c.cpu().numpy().astype(np.float16)
        out['dense'] = res
    return out


# ---------------------------------------------------------------- refinement (CPU, numpy + scipy + cv2)
def bilinear(img, uv):
    """img HxW with nan = invalid, uv Nx2 (pixel centres at integers) -> N, nan outside or next to an invalid pixel."""
    H, W = img.shape; x, y = uv[:, 0], uv[:, 1]
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int); ok = (x0 >= 0) & (y0 >= 0) & (x0 < W - 1) & (y0 < H - 1)
    x0, y0 = np.clip(x0, 0, W - 2), np.clip(y0, 0, H - 2); fx, fy = x - x0, y - y0
    v = (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x0 + 1] * fx * (1 - fy) + img[y0 + 1, x0] * (1 - fx) * fy + img[y0 + 1, x0 + 1] * fx * fy)
    return np.where(ok, v, np.nan)


def backproject(K, M, uv, z):
    d = np.c_[uv, np.ones(len(uv))] @ np.linalg.inv(K).T * np.asarray(z)[:, None]
    return d @ M[:3, :3].T + M[:3, 3]


def zdepth(fr):
    M = fr['c2w']; return ((fr['pts3d'].astype(float) - M[:3, 3]) @ M[:3, :3])[..., 2]


def grid(h, w):
    v, u = np.mgrid[0:h, 0:w]; return np.c_[u.ravel(), v.ravel()].astype(float)


def umeyama(A, B):
    """s, R, t with B ~ s R A + t (least squares)."""
    ma, mb = A.mean(0), B.mean(0); A0, B0 = A - ma, B - mb
    U, S, Vt = np.linalg.svd(B0.T @ A0 / len(A)); D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt; s = float(np.trace(np.diag(S) @ D) / (A0 ** 2).sum(1).mean())
    return s, R, mb - s * R @ ma


def sim3_ransac(A, B, rel=.05, iters=2000, seed=0):
    """Robust B ~ s R A + t; inlier when the residual is below rel x the distance of B from the origin (camera 1)."""
    rng = np.random.default_rng(seed); thr = rel * np.linalg.norm(B, axis=1); best = None
    for _ in range(iters):
        idx = rng.choice(len(A), 3, replace=False); s, R, t = umeyama(A[idx], B[idx])
        inl = np.linalg.norm(B - (s * A @ R.T + t), axis=1) < thr
        if best is None or inl.sum() > best.sum():
            best = inl
    for _ in range(2):
        s, R, t = umeyama(A[best], B[best]); best = np.linalg.norm(B - (s * A @ R.T + t), axis=1) < thr
    return s, R, t, best


def moge_frames(raw: dict, matches: dict, alphas: list):
    """MoGe per-photo camera-frame points embedded on the 518 grid, registered to photo 1 by sim(3) RANSAC on the matches."""
    import fair_ab as fa
    frames, X0, XW = [], fam.X0, fam.XW
    for n in sorted(raw):
        d = np.load(io.BytesIO(raw[n])); P, m = d['points'].astype(np.float32), d['mask'] & np.isfinite(d['points']).all(-1)
        h, w = m.shape; assert (h, w) == (518, XW), (h, w)
        uv = grid(h, w)[m.ravel()]; Q = P[m]  # intrinsics from MoGe's own points: u = fx X/Z + cx (exactly its projection)
        fx, cx = np.linalg.lstsq(np.c_[Q[:, 0] / Q[:, 2], np.ones(len(Q))], uv[:, 0], rcond=None)[0]
        fy, cy = np.linalg.lstsq(np.c_[Q[:, 1] / Q[:, 2], np.ones(len(Q))], uv[:, 1], rcond=None)[0]
        fr = fa.embed_unpadded(dict(pts3d=np.where(m[..., None], P, 0), conf=m.astype(np.float32), valid=m,
                                    K=np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]]), c2w=np.eye(4)), X0, 518)
        frames.append(fr)
    report = {}
    for j in range(1, len(frames)):
        uv0, uvj = matches[(0, j)][:2]
        z0, zj = bilinear(np.where(frames[0]['valid'], zdepth(frames[0]), np.nan), uv0), bilinear(np.where(frames[j]['valid'], zdepth(frames[j]), np.nan), uvj)
        ok = np.isfinite(z0) & np.isfinite(zj)
        A = backproject(frames[j]['K'], np.eye(4), uvj[ok], zj[ok]); B = backproject(frames[0]['K'], np.eye(4), uv0[ok], z0[ok])
        s, R, t, inl = sim3_ransac(A, B)
        M = np.eye(4); M[:3, :3], M[:3, 3] = R, t
        frames[j] = dict(frames[j], pts3d=(s * frames[j]['pts3d'].astype(float) @ R.T + t).astype(np.float32), c2w=M)
        report[f'photo{j + 1}'] = dict(scaleToPhoto1=s, matchesWithDepth=int(ok.sum()), inliers=int(inl.sum()))
    return frames, report


def rot_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def relative(frames):
    """Gauge-free pose summary: relative rotation and baseline direction (in camera i) for every pair."""
    out = {}
    for i, j in combinations(range(len(frames)), 2):
        Mi, Mj = frames[i]['c2w'], frames[j]['c2w']; b = Mi[:3, :3].T @ (Mj[:3, 3] - Mi[:3, 3])
        out[f'{i}-{j}'] = dict(R=Mi[:3, :3].T @ Mj[:3, :3], b=b / np.linalg.norm(b))
    return out


def refine(frames: list, contents: list, matches: dict, focal: bool, size=(518, 518)):
    """Bundle-adjust the backbone's cameras on the matches, then rescale each photo's backbone depth to the BA depths.
    frames: [dict(pts3d, conf, valid, K, c2w)] on the canonical grid; contents: the harness content masks (depth used there);
    matches: {(i, j): (uv_i, uv_j)} on the same grid. Returns refined frames (same masks) and a report; rep['converged'] /
    rep['usable'] are False when a BA pass stopped without meeting a tolerance (bundle_adjust module doc)."""
    import cv2
    import bundle_adjust as ba
    n = len(frames); H, W = size
    Z = [np.where(c & (z > 0), z, np.nan) for c, z in zip(contents, map(zdepth, frames))]
    cam, pt, uv, X, rep = [], [], [], [], dict(pairs={}, ba_backend=ba.BACKEND)
    for (i, j), (ui, uj) in sorted(matches.items()):
        _, inl = cv2.findFundamentalMat(ui, uj, cv2.USAC_MAGSAC, F_PX, .9999, 20000)
        inl = inl.ravel().astype(bool); zi, zj = bilinear(Z[i], ui), bilinear(Z[j], uj); ok = inl & np.isfinite(zi) & np.isfinite(zj)
        X.append((backproject(frames[i]['K'], frames[i]['c2w'], ui[ok], zi[ok]) + backproject(frames[j]['K'], frames[j]['c2w'], uj[ok], zj[ok])) / 2)
        p = sum(map(len, X[:-1])) + np.arange(ok.sum())
        cam += [np.full(ok.sum(), i), np.full(ok.sum(), j)]; pt += [p, p]; uv += [ui[ok], uj[ok]]
        rep['pairs'][f'{i + 1}-{j + 1}'] = dict(sampled=len(ui), fInliers=int(inl.sum()), withDepth=int(ok.sum()))
    cam, pt, uv, X = np.concatenate(cam), np.concatenate(pt), np.concatenate(uv).astype(float), np.concatenate(X)
    Ks = [fr['K'] if not focal else np.diag([(fr['K'][0, 0] + fr['K'][1, 1]) / 2] * 2 + [1.]) + np.c_[np.zeros((3, 2)), [fr['K'][0, 2], fr['K'][1, 2], 0]]
          for fr in frames]
    w2cs = [np.linalg.inv(fr['c2w']) for fr in frames]
    rep['reprojPxInit'] = float(ba.point_errors(Ks, w2cs, X, cam, pt, uv).mean()); rep['ba'] = []
    for it in range(2):  # pass 1, drop points with mean reprojection error > BA_DROP_PX (behind a camera: sqrt(DBL_MAX)), pass 2
        Ks, w2cs, X, info = ba.bundle_adjust(Ks, w2cs, X, cam, pt, uv, focal)
        err = ba.point_errors(Ks, w2cs, X, cam, pt, uv); bad = err > BA_DROP_PX
        rep['ba'].append(dict(pass_=it + 1, report=info.pop('report'), usable=info.pop('usable'), termination=info.pop('termination'),
                              reprojPx=float(err.mean()), points=len(X), dropped=int(bad.sum()) if it == 0 else 0, **info))
        if it == 0:
            new = np.cumsum(~bad) - 1; keep = ~bad[pt]
            X, cam, pt, uv = X[~bad], cam[keep], new[pt[keep]], uv[keep]
    rep['converged'], rep['usable'] = all(p['converged'] for p in rep['ba']), all(p['usable'] for p in rep['ba'])
    out, before = [], relative(frames); rep['photos'] = []
    for i, fr in enumerate(frames):
        M, K = np.linalg.inv(w2cs[i]), Ks[i]; sel = cam == i; u, Xi = uv[sel], X[pt[sel]]
        z_ba = (Xi - M[:3, 3]) @ M[:3, 2]; z_bb = bilinear(Z[i], u); ok = np.isfinite(z_bb) & (z_ba > 0)
        r = z_ba[ok] / z_bb[ok]; s_i = float(np.median(r))
        slope = float(np.polyfit(np.log(z_bb[ok]), np.log(r), 1)[0])  # 0 = one scale fits every depth
        z = zdepth(fr) * s_i
        pts = backproject(K, M, grid(H, W), z.ravel()).reshape(H, W, 3).astype(np.float32)
        out.append(dict(fr, pts3d=pts, K=K, c2w=M))
        rep['photos'].append(dict(photo=i + 1, observations=int(ok.sum()), depthScale=s_i, depthRatioIqr=[float(x) for x in np.percentile(r, [25, 75])],
                                  depthRatioLogSlope=slope, focalBefore=[float(fr['K'][0, 0]), float(fr['K'][1, 1])], focalAfter=[float(K[0, 0]), float(K[1, 1])]))
    after = relative(out)
    rep['poseChange'] = {k: dict(rotationDeg=rot_deg(before[k]['R'].T @ after[k]['R']),
                                 baselineDirDeg=float(np.degrees(np.arccos(np.clip(before[k]['b'] @ after[k]['b'], -1, 1))))) for k in before}
    return out, rep


def refine_pycolmap(frames: list, contents: list, matches: dict, focal: bool, size=(518, 518)):
    """Research comparison only (--ba pycolmap): the original refine with pycolmap 4.2.1's Ceres BA, whose PyPI wheel statically
    links GPL-2.0+ SuiteSparse. Same inputs, tracks and outputs as refine; never on the production path."""
    import cv2
    import pycolmap
    n = len(frames); H, W = size
    Z = [np.where(c & (z > 0), z, np.nan) for c, z in zip(contents, map(zdepth, frames))]
    kps = [[] for _ in range(n)]; tracks = []; rep = dict(pairs={})
    for (i, j), (ui, uj) in sorted(matches.items()):
        _, inl = cv2.findFundamentalMat(ui, uj, cv2.USAC_MAGSAC, F_PX, .9999, 20000)
        inl = inl.ravel().astype(bool); zi, zj = bilinear(Z[i], ui), bilinear(Z[j], uj); ok = inl & np.isfinite(zi) & np.isfinite(zj)
        X = (backproject(frames[i]['K'], frames[i]['c2w'], ui[ok], zi[ok]) + backproject(frames[j]['K'], frames[j]['c2w'], uj[ok], zj[ok])) / 2
        for a, b, x in zip(ui[ok], uj[ok], X):
            tracks.append(((i, len(kps[i])), (j, len(kps[j])), x)); kps[i].append(a); kps[j].append(b)
        rep['pairs'][f'{i + 1}-{j + 1}'] = dict(sampled=len(ui), fInliers=int(inl.sum()), withDepth=int(ok.sum()))
    rec = pycolmap.Reconstruction()
    for i, fr in enumerate(frames):
        K = fr['K']
        cam = (pycolmap.Camera(model='SIMPLE_PINHOLE', width=W, height=H, params=[(K[0, 0] + K[1, 1]) / 2, K[0, 2], K[1, 2]], camera_id=i + 1) if focal
               else pycolmap.Camera(model='PINHOLE', width=W, height=H, params=[K[0, 0], K[1, 1], K[0, 2], K[1, 2]], camera_id=i + 1))
        rec.add_camera_with_trivial_rig(cam)
        w2c = np.linalg.inv(fr['c2w'])
        rec.add_image_with_trivial_frame(pycolmap.Image(name=str(i), keypoints=np.array(kps[i], float).reshape(-1, 2), camera_id=i + 1, image_id=i + 1),
                                         pycolmap.Rigid3d(pycolmap.Rotation3d(w2c[:3, :3]), w2c[:3, 3]))
    for (i, a), (j, b), x in tracks:
        rec.add_point3D(x, pycolmap.Track([pycolmap.TrackElement(i + 1, a), pycolmap.TrackElement(j + 1, b)]))
    opts = pycolmap.BundleAdjustmentOptions(); opts.refine_focal_length = focal; opts.refine_principal_point = False
    opts.refine_extra_params = False; opts.ceres.loss_function_type = pycolmap.LossFunctionType.CAUCHY; opts.ceres.loss_function_scale = 1.0
    opts.ceres.solver_options.max_num_iterations = 200
    rec.update_point_3d_errors(); rep['reprojPxInit'] = rec.compute_mean_reprojection_error(); rep['ba'] = []
    for it in range(2):
        cfg = pycolmap.BundleAdjustmentConfig()
        for i in range(n):
            cfg.add_image(i + 1)
        cfg.fix_gauge(pycolmap.BundleAdjustmentGauge.TWO_CAMS_FROM_WORLD)
        s = pycolmap.create_default_bundle_adjuster(opts, cfg, rec).solve()
        rec.update_point_3d_errors()
        bad = [pid for pid, p in rec.points3D.items() if p.error > BA_DROP_PX]
        rep['ba'].append(dict(pass_=it + 1, report=s.brief_report(), usable=bool(s.is_solution_usable()), termination=str(s.termination_type), reprojPx=rec.compute_mean_reprojection_error(), points=rec.num_points3D(),
                              dropped=len(bad) if it == 0 else 0))
        if it == 0:
            for pid in bad:
                rec.delete_point3D(pid)
    out, before = [], relative(frames); rep['photos'] = []
    for i, fr in enumerate(frames):
        img, cam = rec.image(i + 1), rec.camera(i + 1); w2c = np.r_[img.cam_from_world().matrix(), [[0, 0, 0, 1]]]; M = np.linalg.inv(w2c)
        p = list(cam.params); K = np.array([[p[0], 0, p[1]], [0, p[0], p[2]], [0, 0, 1]]) if focal else np.array([[p[0], 0, p[2]], [0, p[1], p[3]], [0, 0, 1]])
        uv, X = [], []
        for p2 in img.points2D:
            if p2.has_point3D():
                uv.append(p2.xy); X.append(rec.points3D[p2.point3D_id].xyz)
        uv, X = np.array(uv), np.array(X)
        z_ba = (X - M[:3, 3]) @ M[:3, 2]; z_bb = bilinear(Z[i], uv); ok = np.isfinite(z_bb) & (z_ba > 0)
        r = z_ba[ok] / z_bb[ok]; s_i = float(np.median(r))
        slope = float(np.polyfit(np.log(z_bb[ok]), np.log(r), 1)[0])  # 0 = one scale fits every depth
        z = zdepth(fr) * s_i
        pts = backproject(K, M, grid(H, W), z.ravel()).reshape(H, W, 3).astype(np.float32)
        out.append(dict(fr, pts3d=pts, K=K, c2w=M))
        rep['photos'].append(dict(photo=i + 1, observations=int(ok.sum()), depthScale=s_i, depthRatioIqr=[float(x) for x in np.percentile(r, [25, 75])],
                                  depthRatioLogSlope=slope, focalBefore=[float(fr['K'][0, 0]), float(fr['K'][1, 1])], focalAfter=[float(K[0, 0]), float(K[1, 1])]))
    after = relative(out)
    rep['poseChange'] = {k: dict(rotationDeg=rot_deg(before[k]['R'].T @ after[k]['R']),
                                 baselineDirDeg=float(np.degrees(np.arccos(np.clip(before[k]['b'] @ after[k]['b'], -1, 1))))) for k in before}
    return out, rep


MVS = dict(CERT=.5, FB_PX=1.0, ANGLE_DEG=2.0, REPROJ_PX=1.0)  # fixed before any MVS number was computed


def triangulate(Ka, Ma, ua, Kb, Mb, ub):
    """Midpoint of the two rays; returns X, camera-z in a and b, and the ray angle (deg)."""
    da = np.c_[ua, np.ones(len(ua))] @ np.linalg.inv(Ka).T @ Ma[:3, :3].T; db = np.c_[ub, np.ones(len(ub))] @ np.linalg.inv(Kb).T @ Mb[:3, :3].T
    w0 = Ma[:3, 3] - Mb[:3, 3]; a, b, c = (da * da).sum(1), (da * db).sum(1), (db * db).sum(1); d, e = da @ w0, db @ w0
    den = a * c - b * b; ta, tb = (b * e - c * d) / den, (a * e - b * d) / den  # unit camera-z rays: t = camera-z depth
    X = (Ma[:3, 3] + ta[:, None] * da + Mb[:3, 3] + tb[:, None] * db) / 2
    ang = np.degrees(np.arccos(np.clip(b / np.sqrt(a * c), -1, 1)))
    return X, ta, tb, ang


def project(K, M, X):
    c = (X - M[:3, 3]) @ M[:3, :3]; return c[:, :2] / c[:, 2:] * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]


def mvs(frames: list, dense: dict, x0=None, w=None, size=(518, 518)):
    """Dense two-view triangulation with the given (BA) cameras: each crop pixel of photo a is matched by RoMa's warp into
    every partner b, kept if certainty, forward-backward, ray angle and reprojection pass MVS; the partner with the widest ray
    angle wins. pts3d/valid/conf (= certainty) replace the backbone's; K and c2w are kept."""
    x0 = fam.X0 if x0 is None else x0; w = fam.XW if w is None else w; H, W = size
    gv, gu = np.mgrid[0:H, x0:x0 + w]; ua = np.c_[gu.ravel(), gv.ravel()].astype(float)
    best = [dict(ang=np.zeros(H * w), X=np.zeros((H * w, 3)), c=np.zeros(H * w)) for _ in frames]; stats = {}
    for (i, j), (uvAB, cA, uvBA, cB) in sorted(dense.items()):
        for a, b, fwd, ca, bwd in ((i, j, uvAB, cA, uvBA), (j, i, uvBA, cB, uvAB)):
            ub = fwd.reshape(-1, 2).astype(float); back = np.stack([bilinear(bwd[..., k].astype(float), ub - [x0, 0]) for k in (0, 1)], 1)
            fb = np.linalg.norm(back - ua, axis=1)
            Fa, Fb = frames[a], frames[b]
            X, za, zb, ang = triangulate(Fa['K'], Fa['c2w'], ua, Fb['K'], Fb['c2w'], ub)
            with np.errstate(invalid='ignore', divide='ignore'):
                rep = np.maximum(np.linalg.norm(project(Fa['K'], Fa['c2w'], X) - ua, axis=1), np.linalg.norm(project(Fb['K'], Fb['c2w'], X) - ub, axis=1))
                ok = (ca.ravel() >= MVS['CERT']) & (fb < MVS['FB_PX']) & (ang >= MVS['ANGLE_DEG']) & (za > 0) & (zb > 0) & (rep < MVS['REPROJ_PX'])
            take = ok & (ang > best[a]['ang'])
            for k, v in (('ang', ang), ('X', X), ('c', ca.ravel().astype(float))):
                best[a][k][take] = v[take]
            stats[f'{a + 1}<-{b + 1}'] = dict(kept=float(ok.mean()), medianAngleDeg=float(np.median(ang[ok])) if ok.any() else None)
    out = []
    for fr, bst in zip(frames, best):
        pts = np.zeros((H, W, 3), np.float32); conf = np.zeros((H, W), np.float32); valid = np.zeros((H, W), bool)
        v = bst['ang'] > 0; pts[gv.ravel()[v], gu.ravel()[v]] = bst['X'][v]; conf[gv.ravel()[v], gu.ravel()[v]] = bst['c'][v]
        valid[gv.ravel()[v], gu.ravel()[v]] = True
        out.append(dict(fr, pts3d=pts, conf=conf, valid=valid))
    stats['validFraction'] = [float(o['valid'][:, x0:x0 + w].mean()) for o in out]
    return out, stats


FUSE_SIGMA_PX = 16.  # fixed before any fused number was computed


def fuse(bb: list, mv: list, contents: list, sigma=FUSE_SIGMA_PX):
    """Same cameras: keep the triangulated depth where MVS has it; elsewhere the backbone depth times the MVS/backbone ratio,
    smoothed (normalised Gaussian convolution of the log ratio, sigma px; global median where no MVS pixel is near)."""
    import cv2
    out, rep = [], []
    for b, m, c in zip(bb, mv, contents):
        zb, zm = zdepth(b), zdepth(m); okb = c & (zb > 0); okm = m['valid'] & okb & (zm > 0)
        lr = np.where(okm, np.log(np.where(okm, zm, 1) / np.where(okb, zb, 1)), 0.)
        num, den = cv2.GaussianBlur(lr, (0, 0), sigma), cv2.GaussianBlur(okm.astype(float), (0, 0), sigma)
        g = float(np.median(lr[okm])); ls = np.where(den > 1e-3, num / np.maximum(den, 1e-3), g)
        z = np.where(okm, zm, zb * np.exp(ls)); H, W = z.shape
        pts = backproject(b['K'], b['c2w'], grid(H, W), z.ravel()).reshape(H, W, 3).astype(np.float32)
        out.append(dict(b, pts3d=pts, valid=b['valid'] & okb, conf=np.where(okb, b['conf'], 0).astype(np.float32)))
        rep.append(dict(mvsFraction=float(okm.sum() / max(okb.sum(), 1)), medianRatio=float(np.exp(g)), ratioP5P95=[float(np.exp(x)) for x in np.percentile(lr[okm], [5, 95])]))
    return out, rep


# ---------------------------------------------------------------- frames on disk / in transit
NAMES = ('pts3d', 'conf', 'valid_mask', 'intrinsics', 'camera_to_world')
KEYS = ('pts3d', 'conf', 'valid', 'K', 'c2w')


def read_geometry(geom: Path) -> list:
    out = []
    for f in sorted((geom / 'frames').iterdir()):
        a = {k: np.load(f / f'{n}.npy') for k, n in zip(KEYS, NAMES)}
        out.append(dict(pts3d=a['pts3d'].astype(np.float32), conf=a['conf'].astype(np.float32), valid=a['valid'].astype(bool),
                        K=a['K'].astype(float), c2w=a['c2w'].astype(float)))
    return out


def write_geometry(geom: Path, cell: str, frames: list, manifest: dict):
    man = json.loads((fam.DATA / cell / 'manifest.json').read_text())
    for fr, f in zip(frames, man['frames']):
        d = geom / 'frames' / f['frame_id']; d.mkdir(parents=True, exist_ok=True)
        for k, n in zip(KEYS, NAMES):
            np.save(d / f'{n}.npy', fr[k])
        (d / 'canonical.png').write_bytes((fam.DATA / cell / 'canonical' / f"{f['frame_id']}.png").read_bytes())
    (geom / 'candidate_manifest.json').write_text(json.dumps(manifest, indent=1, default=float) + '\n')


def pack_frames(frames):
    return npz(**{f'{k}{i}': fr[k] for i, fr in enumerate(frames) for k in KEYS}, n=np.array(len(frames)))


def unpack_frames(data):
    d = np.load(io.BytesIO(data)); return [{k: d[f'{k}{i}'] for k in KEYS} for i in range(int(d['n']))]


def load_matches(cell):
    out = {}
    for f in sorted((GPUD / cell).glob('roma-*.npz')):
        i, j = map(int, f.stem.split('-')[1:]); d = np.load(f); out[(i, j)] = (d['uvA'], d['uvB'])
    return out


def _refine_job(job: dict, fn) -> dict:
    sys.path.insert(0, '/check')
    start = time.monotonic()
    frames, matches = unpack_frames(job['frames']), {tuple(map(int, k.split('-'))): (np.asarray(a), np.asarray(b)) for k, (a, b) in job['matches'].items()}
    out, rep = fn(frames, [np.asarray(c) for c in job['contents']], matches, job['focal'])
    return dict(frames=pack_frames(out), report=rep, containerSeconds=time.monotonic() - start)


@app.function(image=refine_image, cpu=4, memory=8 * 1024, timeout=1800, retries=0, min_containers=0)
def refine_job(job: dict) -> dict:
    return _refine_job(job, refine)


@app.function(image=pycolmap_image, cpu=4, memory=8 * 1024, timeout=1800, retries=0, min_containers=0)
def refine_job_pycolmap(job: dict) -> dict:  # research comparison only
    return _refine_job(job, refine_pycolmap)


@app.function(image=analyse_image, cpu=8, memory=16 * 1024, timeout=1800, retries=0, min_containers=0)
def analyse(job: dict) -> dict:
    sys.path[:0] = ['/check', '/serving']
    import tempfile
    start = time.monotonic()
    import fair_ab as fa
    fa._check()
    with tempfile.TemporaryDirectory() as tmp:
        with tarfile.open(fileobj=io.BytesIO(job['geometry']), mode='r:gz') as tar:
            tar.extractall(tmp, filter='data')
        out = fam.analyse_one(job['cell'], job['backbone'], 'padded', Path(tmp) / 'geometry', Path('/data'))
    out['containerSeconds'] = time.monotonic() - start
    return out


def contents_for(cell, frames):
    import fair_ab as fa
    man = json.loads((fam.DATA / cell / 'manifest.json').read_text())
    return [fa.content_mask(fr['pts3d'], fr['valid'], fr['conf'], np.load(fam.DATA / cell / 'canonical' / f"{f['frame_id']}_alpha.npy"))
            for fr, f in zip(frames, man['frames'])]


def ledger(path: Path, **kw):
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(dict(mode='ephemeral modal run', rateSource='https://modal.com/pricing', **kw), indent=1) + '\n')
    print(json.dumps(kw)[:600])


@app.local_entrypoint()
def main(stage: str, cells: str = '090,030', bases: str = 'moge,da3-base,mapanything,pi3x', only: str = '', vggt: bool = False, ba: str = 'numpy'):
    cells = cells.split(',')
    if stage == 'access':
        print(json.dumps(access.remote())); return
    if stage == 'infer':
        if GPUD.exists():
            raise ValueError(f'{GPUD} exists; choose a fresh one')
        t = time.monotonic(); fr = {c: fam.frames_for(c) for c in cells}
        h = moge.spawn({c: fr[c]['unpadded'] for c in cells}); r = vggt_roma.remote(fr, vggt); m = h.get()
        for c in cells:
            (GPUD / c).mkdir(parents=True)
            for k, data in r['out'][c]['roma'].items():
                (GPUD / c / f'roma-{k}.npz').write_bytes(data)
            for n, data in m['out'][c].items():
                (GPUD / c / f'moge-{Path(n).stem}.npz').write_bytes(data)
            if 'vggt' not in r['out'][c]:
                continue
            frames = [{k: np.load(io.BytesIO(r['out'][c]['vggt'][n]))[k] for k in KEYS} for n in sorted(r['out'][c]['vggt'])]
            write_geometry(GEOM / f'{c}-vggt' / 'geometry', c, frames, dict(model=VGGT, revision=VGGT_REV, code=VGGT_CODE, gpu=r['gpu'], precision='bf16 autocast',
                                                                              points='depth head unprojected with its own cameras', input='padded canonical 518x518'))
        sec = r['containerSeconds'] + m['containerSeconds']
        ledger(GPUD / 'spend-ledger.json', hardware='L4, 4 CPU, 32 GiB (x2: RoMa (+ VGGT if readable), MoGe)', vggt=vggt, functionSeconds=sec, callSeconds=time.monotonic() - t,
               estimateUsd=L4_RATE * sec, timing=r['timing'], mogeSeconds=m['containerSeconds'], gpu=[r['gpu'], m['gpu']])
        return
    if stage == 'refine':
        jobs, reports = [], {}
        for c in cells:
            matches = load_matches(c)
            if 'moge' in bases.split(',') and not INIT['moge'](c).exists():
                raw = {f.stem.split('-', 1)[1] + '.png': f.read_bytes() for f in sorted((GPUD / c).glob('moge-*.npz'))}
                frames, rep = moge_frames(raw, matches, None)
                write_geometry(INIT['moge'](c), c, frames, dict(model=MOGE, revision=MOGE_REV, input='392x518 content crop, per photo',
                                                               registration='sim(3) RANSAC to photo 1 on RoMa matches', report=rep))
            for b in bases.split(','):
                frames = read_geometry(INIT[b](c)); cont = contents_for(c, frames)
                for v, focal in (('ba', False), ('ba-f', True)):
                    jobs.append(dict(cell=c, base=b, variant=v, focal=focal, frames=pack_frames(frames), contents=cont,
                                     matches={f'{i}-{j}': (a, bb) for (i, j), (a, bb) in matches.items()}))
        if only:
            jobs = [j for j in jobs if f"{j['cell']}-{j['base']}-{j['variant']}" in only.split(',')]
        t = time.monotonic(); total = 0.
        fn = {'numpy': refine_job, 'pycolmap': refine_job_pycolmap}[ba]
        for job, r in zip(jobs, fn.map(jobs, order_outputs=True, return_exceptions=True)):
            name = f"{job['cell']}-{job['base']}-{job['variant']}" + ('' if ba == 'numpy' else '-pycolmap')
            if isinstance(r, Exception):
                print(name, 'FAILED', repr(r)[:800]); continue
            write_geometry(GEOM / name / 'geometry', job['cell'], unpack_frames(r['frames']), dict(base=job['base'], init=str(INIT[job['base']](job['cell'])),
                                                                                                 refinement=job['variant'], report=r['report']))
            total += r['containerSeconds']; reports[name] = r['report']
            print(name, f"{r['containerSeconds']:.0f}s", json.dumps({k: r['report'][k] for k in ('reprojPxInit', 'poseChange')}, default=float)[:300],
                  [round(p['depthScale'], 4) for p in r['report']['photos']])
        (GEOM / f'refine-reports-{int(time.time())}.json').write_text(json.dumps(reports, indent=1, default=float) + '\n')
        ledger(GEOM / f'spend-ledger-refine-{int(time.time())}.json', hardware='4 CPU, 8 GiB per job', jobs=len(jobs), functionSeconds=total,
               callSeconds=time.monotonic() - t, estimateUsd=(4 * .0000131 + 8 * .00000222) * total)
        return
    if stage == 'dense-infer':
        t = time.monotonic(); r = roma_dense.remote({c: fam.frames_for(c)['unpadded'] for c in cells})
        for c in cells:
            for k, data in r['out'][c].items():
                (GPUD / c / f'dense-{k}.npz').write_bytes(data)
        ledger(GPUD / 'spend-ledger-dense.json', hardware='L4, 4 CPU, 32 GiB', functionSeconds=r['containerSeconds'], callSeconds=time.monotonic() - t,
               estimateUsd=L4_RATE * r['containerSeconds'], gpu=r['gpu'])
        return
    if stage == 'mvs':  # local CPU, numpy only: dense triangulation with each BA-f geometry's cameras
        for c in cells:
            dense = {}
            for f in sorted((GPUD / c).glob('dense-*.npz')):
                i, j = map(int, f.stem.split('-')[1:]); d = np.load(f); dense[(i, j)] = (d['uvAB'], d['certA'], d['uvBA'], d['certB'])
            for b in bases.split(','):
                src = GEOM / f'{c}-{b}-ba-f' / 'geometry'; out, st = mvs(read_geometry(src), dense)
                write_geometry(GEOM / f'{c}-{b}-ba-f-mvs' / 'geometry', c, out, dict(cameras=str(src), points='RoMa dense warp, two-view midpoint triangulation',
                                                                                       thresholds=MVS, report=st))
                print(c, b, json.dumps(st))
        return
    if stage == 'fuse':  # local CPU: MVS depth + backbone depth corrected by the smoothed MVS ratio, same BA-f cameras
        for c in cells:
            for b in bases.split(','):
                src = GEOM / f'{c}-{b}-ba-f' / 'geometry'; bb = read_geometry(src); mv = read_geometry(GEOM / f'{c}-{b}-ba-f-mvs' / 'geometry')
                out, rep = fuse(bb, mv, contents_for(c, bb))
                write_geometry(GEOM / f'{c}-{b}-ba-f-fused' / 'geometry', c, out, dict(cameras=str(src), sigmaPx=FUSE_SIGMA_PX, report=rep,
                                                                                         points='MVS depth where triangulated, else backbone depth x smoothed MVS ratio'))
                print(c, b, json.dumps(rep))
        return
    if stage == 'analyse':
        OUT.mkdir(parents=True, exist_ok=True); jobs = []
        for c in cells:
            names = [f'{c}-mapanything-raw'] + sorted(p.name for p in GEOM.glob(f'{c}-*') if p.is_dir())
            for name in names:
                geom = INIT['mapanything'](c) if name.endswith('mapanything-raw') else GEOM / name / 'geometry'
                jobs.append(dict(cell=c, backbone=name.split('-', 1)[1], geometry=fam.geometry_tar(geom)))
        if only:
            jobs = [j for j in jobs if f"{j['cell']}-{j['backbone']}" in only.split(',')]
        t = time.monotonic(); total = 0.
        for job, r in zip(jobs, analyse.map(jobs, order_outputs=True, return_exceptions=True)):
            name = f"{job['cell']}-{job['backbone']}"
            if isinstance(r, Exception):
                print(name, 'FAILED', repr(r)[:800]); continue
            (OUT / f'{name}.json').write_text(json.dumps(r, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o)) + '\n')
            total += r['containerSeconds']; print(name, f"{r['containerSeconds']:.0f}s")
        ledger(OUT / f'spend-ledger-{int(time.time())}.json', hardware='8 CPU, 16 GiB per job', jobs=len(jobs), functionSeconds=total,
               callSeconds=time.monotonic() - t, estimateUsd=CPU_RATE * total)
        return
    raise ValueError(stage)


# ---------------------------------------------------------------- self-test
def _check():
    """Synthetic: floor + wall seen by three cameras; the 'backbone' has each photo's depth off by its own factor and its
    cameras perturbed. BA must recover the relative poses and the depth rescale must make the three photos agree."""
    rng = np.random.default_rng(0); H = W = 518; K = np.array([[400., 0, 258.5], [0, 400, 258.5], [0, 0, 1]])

    def look(C, target):
        z = target - C; z /= np.linalg.norm(z); x = np.cross(z, [0, 0, 1.]); x /= np.linalg.norm(x); y = np.cross(z, x)
        M = np.eye(4); M[:3, :3] = np.c_[x, y, z]; M[:3, 3] = C; return M
    true = [look(np.array(c, float), np.array([0., 3, .5])) for c in ([0, 0, 1.5], [1.2, .3, 1.5], [-1., .5, 1.4])]
    pts = np.r_[np.c_[rng.uniform(-2, 2, 3000), rng.uniform(1.5, 4.5, 3000), np.zeros(3000)],
                np.c_[rng.uniform(-2, 2, 3000), np.full(3000, 4.5), rng.uniform(0, 2, 3000)]]

    def depth_map(M):  # ray-cast floor z=0 and wall y=4.5
        g = grid(H, W); d = np.c_[g, np.ones(len(g))] @ np.linalg.inv(K).T @ M[:3, :3].T; C = M[:3, 3]
        tf = np.where(d[:, 2] < 0, -C[2] / np.minimum(d[:, 2], -1e-9), np.inf); tw = np.where(d[:, 1] > 0, (4.5 - C[1]) / np.maximum(d[:, 1], 1e-9), np.inf)
        return np.minimum(tf, tw).reshape(H, W)  # camera z = t because d has unit z in the camera
    factors = [1.0, 1.08, .93]; frames, contents = [], []
    for M, f in zip(true, factors):
        P = np.eye(4); P[:3, :3] = cv_rot(rng.normal(size=3) * np.radians(.8)); P[:3, 3] = rng.normal(size=3) * .04
        Mb = P @ M; z = depth_map(M) * f
        ok = np.isfinite(z) & (z < 50)
        frames.append(dict(pts3d=backproject(K, Mb, grid(H, W), np.where(ok, z, 1).ravel()).reshape(H, W, 3).astype(np.float32),
                           conf=np.ones((H, W), np.float32), valid=ok, K=K.copy(), c2w=Mb)); contents.append(ok)
    matches = {}
    for i, j in combinations(range(3), 2):
        uv = []
        for M in (true[i], true[j]):
            c = (pts - M[:3, 3]) @ M[:3, :3]; uv.append(c[:, :2] / c[:, 2:] * 400 + 258.5)
        ok = np.all([(u > 2).all(1) & (u < W - 3).all(1) for u in uv], 0) & ((pts - true[i][:3, 3]) @ true[i][:3, 2] > 0) & ((pts - true[j][:3, 3]) @ true[j][:3, 2] > 0)
        matches[(i, j)] = tuple(u[ok] + rng.normal(scale=.3, size=u[ok].shape) for u in uv)
    out, rep = refine(frames, contents, matches, focal=False)
    rt, ro = relative([dict(c2w=M) for M in true]), relative(out)
    for k in rt:
        assert rot_deg(rt[k]['R'].T @ ro[k]['R']) < .05 and np.degrees(np.arccos(min(1, rt[k]['b'] @ ro[k]['b']))) < .3, (k, rep['poseChange'])
    s = np.array([p['depthScale'] * f for p, f in zip(rep['photos'], factors)]); assert np.ptp(s / s[0]) < .005, s
    assert rep['ba'][-1]['reprojPx'] < .6 and rep['converged'] and rep['usable'], rep['ba']
    # focal refinement: photo 2's focal 6 % long must come back to 400 within 1 %
    frames[1] = dict(frames[1], K=np.diag([424., 424, 1]) + np.array([[0, 0, 258.5], [0, 0, 258.5], [0, 0, 0]]))
    _, rep_f = refine(frames, contents, matches, focal=True)
    assert all(abs(p['focalAfter'][0] / 400 - 1) < .01 for p in rep_f['photos']) and rep_f['converged'], ([p['focalAfter'] for p in rep_f['photos']], rep_f['ba'])
    # sim(3): exact recovery with 30 % outliers
    A = rng.normal(size=(500, 3)) + [0, 0, 4]; R0 = cv_rot([.1, -.2, .3]); B = 1.7 * A @ R0.T + [.3, -.1, .2]; o = rng.normal(size=(150, 3)); B[:150] += 2 * o / np.linalg.norm(o, axis=1, keepdims=True)
    s_, R_, t_, inl = sim3_ransac(A, B); assert abs(s_ - 1.7) < 1e-6 and rot_deg(R_.T @ R0) < 1e-4 and inl[150:].all() and not inl[:150].any(), (s_, inl.sum())
    # dense triangulation with the true cameras: warps from the true depth must give the true depth back
    tf = [dict(K=K, c2w=M) for M in true]; Ztrue = [depth_map(M) for M in true]; dense = {}
    for i, j in ((0, 1),):
        def warp(a, b):
            z = Ztrue[a]; Xa = backproject(K, true[a], grid(H, W), np.where(np.isfinite(z) & (z < 50), z, 1).ravel())
            return project(K, true[b], Xa).reshape(H, W, 2).astype(np.float32)
        dense[(i, j)] = (warp(i, j), np.ones((H, W), np.float16), warp(j, i), np.ones((H, W), np.float16))
    mv, st = mvs(tf, dense, x0=0, w=W)
    zt, zm = Ztrue[0][mv[0]['valid']], zdepth(mv[0])[mv[0]['valid']]
    assert st['validFraction'][0] > .5 and np.abs(zm / zt - 1).max() < 1e-4, (st, np.abs(zm / zt - 1).max())
    # fusion: backbone depth 10 % long with a smooth tilt, MVS exact on the left half -> right half corrected to < 1 %
    zt = Ztrue[0]; okz = np.isfinite(zt) & (zt < 50); zt = np.where(okz, zt, 1.); tilt = 1.1 + .05 * grid(H, W)[:, 1].reshape(H, W) / H
    bbf = dict(K=K, c2w=true[0], pts3d=backproject(K, true[0], grid(H, W), (zt * tilt).ravel()).reshape(H, W, 3).astype(np.float32), conf=np.ones((H, W), np.float32), valid=okz)
    half = okz.copy(); half[:, W // 2:] = False
    mvf = dict(bbf, pts3d=backproject(K, true[0], grid(H, W), zt.ravel()).reshape(H, W, 3).astype(np.float32), valid=half)
    fz, _ = fuse([bbf], [mvf], [okz]); err = np.abs(zdepth(fz[0]) / zt - 1)
    near = okz.copy(); near[:, W // 2 + 16:] = False  # within one sigma of MVS: local ratio; beyond ~3 sigma: global median
    assert err[half].max() < 1e-5 and err[near].max() < .005 and err[okz].max() < .03, (err[near].max(), err[okz].max())
    print('geometry_clean_ab self-test passed', json.dumps(rep['poseChange'], default=float)[:200], [round(x, 4) for x in s / s[0]])


def cv_rot(v):
    import cv2
    return cv2.Rodrigues(np.asarray(v, float))[0]


if __name__ == '__main__' and sys.argv[1:] == ['check']:
    _check()
