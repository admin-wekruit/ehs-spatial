"""Geometry-stage contract + backbone adapters for the 2026-10-06 backbone A/B.

CONTRACT (every backbone, identical; the evaluator fair_ab_modal.analyse is unchanged)
  input   the run's frozen canonical frames: padded 518x518 RGB PNGs, byte-identical to <run>/evidence/canonical/frame_000N.png,
          in frame order, through MapAnythingAdapter's runner seam (data URIs). Optional: our K (one 3x3 per cell, canonical grid).
  output  <geometry>/frames/frame_000N/ for N = 1..frames, the MapAnythingAdapter files:
            canonical.png        the input frame, pixel-identical
            pts3d.npy            HxWx3 float32 world points on the canonical grid (pixel (u, v) -> its 3D point)
            conf.npy             HxW float32, higher = more confident (the harness keeps conf >= 0.1)
            valid_mask.npy       HxW bool
            intrinsics.npy       3x3 K on the canonical grid (integer pixel centres)
            camera_to_world.npy  4x4 OpenCV c2w (right / down / forward), rigid
          + candidate_manifest.json (provenance). Units are native (any scale): the e-stop sets metres per native unit.
check_arrays() / check_geometry() assert this before anything is scored.
"""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path
import shutil
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image

SIZE = 518


# ---------------------------------------------------------------- contract
def check_arrays(geom: Path, n_frames: int, size: int = SIZE) -> dict:
    """Shapes, dtypes, finiteness, K and rigid c2w, frame count; returns the point-map/camera consistency per frame (px)."""
    geom = Path(geom); ids = sorted(p.name for p in (geom / 'frames').iterdir())
    assert ids == [f'frame_{i:04d}' for i in range(1, n_frames + 1)], ids
    out = {}
    for fid in ids:
        g = geom / 'frames' / fid
        img = np.asarray(Image.open(g / 'canonical.png').convert('RGB'))
        pts, conf, valid = np.load(g / 'pts3d.npy'), np.load(g / 'conf.npy'), np.load(g / 'valid_mask.npy')
        K, M = np.load(g / 'intrinsics.npy').astype(float), np.load(g / 'camera_to_world.npy').astype(float)
        assert img.shape == (size, size, 3) and pts.shape == (size, size, 3) and pts.dtype == np.float32, (fid, img.shape, pts.shape, pts.dtype)
        assert conf.shape == valid.shape == (size, size) and valid.dtype == bool, (fid, conf.shape, valid.dtype)
        assert valid.mean() > .05, (fid, 'valid fraction', valid.mean())  # MVS point maps are sparse; this catches empty outputs
        assert np.isfinite(pts[valid]).all() and np.isfinite(conf[valid]).all(), (fid, 'non-finite valid values')
        assert K.shape == (3, 3) and np.isfinite(K).all() and np.allclose(K[2], [0, 0, 1]) and min(K[0, 0], K[1, 1]) > 0, (fid, K)
        assert abs(K[0, 1]) < 1e-3 * K[0, 0] and 0 < K[0, 2] < size and 0 < K[1, 2] < size, (fid, K)
        R = M[:3, :3]
        assert M.shape == (4, 4) and np.isfinite(M).all() and np.allclose(M[3], [0, 0, 0, 1]), (fid, M)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-3) and abs(np.linalg.det(R) - 1) < 1e-3, (fid, 'c2w rotation not orthonormal')
        v, u = np.nonzero(valid); Xc = (pts[v, u].astype(float) - M[:3, 3]) @ R  # world -> camera
        front = Xc[:, 2] > 0
        assert front.mean() > .99, (fid, 'points behind the camera', 1 - front.mean())
        uv = (Xc[front] / Xc[front, 2:]) @ K.T
        err = np.hypot(uv[:, 0] - u[front], uv[:, 1] - v[front])
        out[fid] = dict(validFraction=float(valid.mean()), reprojMedianPx=float(np.median(err)), reprojP95Px=float(np.quantile(err, .95)))
        assert out[fid]['reprojMedianPx'] < 5., (fid, 'point map is not on the canonical grid of (K, c2w)', out[fid])
    return out


def check_geometry(geom: Path, run: Path) -> dict:
    """check_arrays + the frame ids and canonical images of the frozen run (pixel-identical input)."""
    man = json.loads((Path(run) / 'manifest.json').read_text())
    out = check_arrays(geom, len(man['frames']))
    for f in man['frames']:
        seen = np.asarray(Image.open(Path(geom) / 'frames' / f['frame_id'] / 'canonical.png').convert('RGB'))
        assert np.array_equal(seen, np.asarray(Image.open(Path(run) / f['canonical']).convert('RGB'))), (f['frame_id'], 'not the frozen frame')
    return out


def finish(out: Path, meta: dict):
    """Manifest + the same raw-copy cleanup as the fair harness (frames/ keeps every array), then the array contract."""
    out = Path(out)
    (out / 'candidate_manifest.json').write_text(json.dumps(meta, indent=2, default=str) + '\n')
    for junk in ('provider', 'map_anything_response.json', 'point_cloud.glb'):
        p = out / junk; shutil.rmtree(p) if p.is_dir() else p.unlink(missing_ok=True)
    meta['contract'] = check_arrays(out, len(list((out / 'frames').iterdir())))
    (out / 'candidate_manifest.json').write_text(json.dumps(meta, indent=2, default=str) + '\n')


def decode_inputs(input: dict) -> list[np.ndarray]:
    sources = input.get('inputs')
    assert isinstance(sources, list) and 1 <= len(sources) <= 4, 'one to four canonical PNG data URIs'
    images = []
    for uri in sources:
        assert isinstance(uri, str) and uri.startswith('data:image/png;base64,'), 'canonical PNG data URI only'
        with Image.open(io.BytesIO(base64.b64decode(uri.split(',', 1)[1], validate=True))) as im:
            assert im.format == 'PNG' and im.mode == 'RGB' and im.size == (SIZE, SIZE), (im.format, im.mode, im.size)
            images.append(np.array(im))
    return images


def response(depth, images, conf, K, w2c):
    """Depth + cameras -> the frozen frame schema, through the DA3 runner's own converter (same mask rule, same unprojection)."""
    from candidate_geometry_backend import prediction_to_response
    return prediction_to_response(SimpleNamespace(depth=np.asarray(depth, np.float32), processed_images=np.asarray(images, np.uint8),
                                                  conf=np.asarray(conf, np.float32), intrinsics=np.asarray(K, np.float32),
                                                  extrinsics=np.asarray(w2c, np.float32)))


# ---------------------------------------------------------------- classic: our K + ALIKED/LightGlue poses + MoGe-3 depth
class MoGe3:
    REPO, REVISION = 'Ruicheng/moge-3-vitl', '184008f877d7ad1ad4c2cd2182a9bd1f63d0e5be'

    def __init__(self, device='cuda'):
        import torch
        from moge.model.v3 import MoGeModel
        self.model = MoGeModel.from_pretrained(self.REPO, revision=self.REVISION).to(device).eval()
        self.torch, self.device = torch, device

    def __call__(self, rgb: np.ndarray, fx: float | None = None) -> dict:
        """fx (pixels) given: MoGe-3's own fov_x input, so its depth is consistent with that K; None: MoGe-3 estimates it."""
        import math
        t = self.torch.from_numpy(np.ascontiguousarray(rgb)).float().permute(2, 0, 1).div(255).to(self.device)
        fov = None if fx is None else math.degrees(2 * math.atan(rgb.shape[1] / 2 / fx))
        o = self.model.infer(t, use_fp16=True, fov_x=fov)  # defaults: resolution_level 9, refine_steps 3, mask applied
        h, w = rgb.shape[:2]; Kn = o['intrinsics'].float().cpu().numpy()
        depth = o['depth'].float().cpu().numpy(); mask = o['mask'].cpu().numpy().astype(bool) & np.isfinite(depth) & (depth > 0)
        return dict(depth=np.where(mask, depth, 0).astype(np.float32), fx=float(Kn[0, 0] * w), fy=float(Kn[1, 1] * h),
                    cx=float(Kn[0, 2] * w - .5), cy=float(Kn[1, 2] * h - .5))


def our_k(focals, x0, xw, size=SIZE):
    """One K per cell: median MoGe-3 focal of the cell's frames, principal point at the content centre (integer pixel centres)."""
    f = float(np.median(focals))
    return [[f, 0., x0 + (xw - 1) / 2], [0., f, (size - 1) / 2], [0., 0., 1.]]


class ClassicRunner:
    """Permissive route: our K; ALIKED + LightGlue matches; essential-matrix / PnP poses + bundle adjustment (K fixed);
    MoGe-3 depth of each frame scaled to the triangulated points (one scale per frame). Every licence: MIT / Apache-2.0 / BSD-3."""

    def __init__(self, moge: MoGe3, *, device='cuda', content=(63, 392), max_keypoints=4096, focal=None):
        import torch
        from lightglue import ALIKED, LightGlue
        self.torch, self.device, self.moge, self.content, self.focal = torch, device, moge, content, focal
        self.extractor = ALIKED(max_num_keypoints=max_keypoints).eval().to(device)
        self.matcher = LightGlue(features='aliked').eval().to(device)
        self.metadata = dict(model_id='classic: ALIKED-n16 + LightGlue (aliked) poses, MoGe-3 (Ruicheng/moge-3-vitl) depth',
                             code_revision=dict(MoGe='74fbce054ebed49800de42d0ad0e83495065719a', LightGlue='eb42fee2d71449efb0aa5c10549752b5d75384d8'),
                             model_revision=dict(moge=MoGe3.REVISION), torch_version=torch.__version__, device=device,
                             licences='MoGe-3 MIT (DINOv2 parts Apache-2.0); LightGlue Apache-2.0; ALIKED BSD-3; OpenCV Apache-2.0; SciPy BSD-3',
                             contentRect=list(content), maxKeypoints=max_keypoints, extractorResize=None,
                             ourK='one focal per device (fx = fy, principal point = content centre): the focal sweep of a three-view cell '
                                  '(median reprojection after bundle adjustment, parabola at the minimum); a two-view cell takes that device focal',
                             poses='USAC_MAGSAC essential matrix (1 px) per pair with our K, base pair = most inliers, PnP for the rest, bundle '
                                   'adjustment (Huber 1 px, K fixed), observations > 2 px dropped and re-adjusted',
                             depth='MoGe-3 (fov_x = our K) on the content crop of each frame, embedded on the canonical grid; one scale per frame = median '
                                   'z_triangulated / z_moge at its keypoints; world scale = median MoGe metric scale',
                             mask_semantics='MoGe-3 mask, finite positive depth, relative depth edge < 0.08 (DA3 converter)',
                             confidence_semantics='1 inside the mask (no learned confidence)')

    def __call__(self, model_identifier, *, input):
        torch = self.torch; t0 = time.perf_counter(); x0, xw = self.content
        images = decode_inputs(input); crops = [im[:, x0:x0 + xw] for im in images]
        free = [self.moge(c) for c in crops]  # MoGe-3's own focal: diagnostic only
        K_moge = np.array(our_k([(m['fx'] + m['fy']) / 2 for m in free], x0, xw)); K = K_moge.copy()
        kps, feats = [], []
        with torch.inference_mode():
            for c in crops:
                f = self.extractor.extract(torch.from_numpy(c).permute(2, 0, 1).float().div(255).to(self.device), resize=None)
                feats.append(f); kps.append(f['keypoints'][0].cpu().numpy().astype(float) + [x0, 0])
            matches = {}
            for i in range(len(images)):
                for j in range(i + 1, len(images)):
                    m = self.matcher({'image0': feats[i], 'image1': feats[j]})
                    matches[(i, j)] = m['matches'][0].cpu().numpy()
        sweep = focal_sweep(kps, matches, K) if len(images) >= 3 else (None, None, False)
        assert self.focal is not None or sweep[2], 'two-view cell needs the device focal; three-view sweep had no interior minimum'
        K[0, 0] = K[1, 1] = self.focal if self.focal is not None else sweep[0]
        rec = sfm(kps, matches, K); self.kps, self.matches = kps, matches
        mono = [self.moge(c, fx=K[0, 0]) for c in crops]  # depth under our K
        depth = np.zeros((len(images), SIZE, SIZE), np.float32)
        for i, m in enumerate(mono):
            depth[i, :, x0:x0 + xw] = m['depth']
        scales, align = mono_scales(depth, rec, K)
        g = float(np.median(scales))  # world in MoGe-metric-like units
        w2c = [np.c_[R, t / g] for R, t in zip(rec['R'], rec['t'])]
        depth *= (np.asarray(scales) / g)[:, None, None]
        out = response(depth, images, (depth > 0).astype(np.float32), [K] * len(images), w2c)
        self.metadata.update(ourK=K.tolist(), ourKSource='device focal given' if self.focal is not None else 'focal sweep of this cell',
                             focalSweep=dict(bestPx=sweep[0], interiorMinimum=sweep[2], curve=sweep[1]), mogeMedianK=K_moge.tolist(), mogeFocalPx=[dict(fx=m['fx'], fy=m['fy'], cx=m['cx'] + x0, cy=m['cy']) for m in free],
                             mogeFocalUnderOurK=[m['fx'] for m in mono],
                             keypoints=[len(k) for k in kps], sfm={k: v for k, v in rec.items() if k not in ('R', 't', 'X', 'obs', 'kps')},
                             alignment=align, worldScaleDivisor=g, inference_and_encoding_seconds=time.perf_counter() - t0)
        return out


def _triangulate(Ps, uvs):
    A = np.concatenate([np.stack([uv[0] * P[2] - P[0], uv[1] * P[2] - P[1]]) for P, uv in zip(Ps, uvs)])
    X = np.linalg.svd(A)[2][-1]
    return X[:3] / X[3]


def _project(K, R, t, X):
    Xc = X @ R.T + t
    return (Xc[:, :2] / Xc[:, 2:]) @ K[:2, :2].T + K[:2, 2], Xc[:, 2]


def sfm(kps, matches, K, ransac_px=1., drop_px=2.):
    """Two- or three-view reconstruction with fixed K. kps[i]: (n_i, 2) pixels; matches[(i, j)]: (m, 2) index pairs.
    Returns w2c R[i], t[i] (camera of the base pair's first view = identity), points X, observations and diagnostics."""
    import cv2
    from scipy.optimize import least_squares
    from scipy.sparse import lil_matrix
    from scipy.spatial.transform import Rotation
    K = np.array(K, float); n = len(kps); pairs = {}
    for (i, j), m in matches.items():
        if len(m) < 8:
            continue
        p, q = kps[i][m[:, 0]], kps[j][m[:, 1]]
        E, inl = cv2.findEssentialMat(p, q, K, method=cv2.USAC_MAGSAC, prob=.99999, threshold=ransac_px)
        if E is None or E.shape != (3, 3):
            continue
        _, R, t, inl = cv2.recoverPose(E, p, q, K, mask=inl.copy())
        keep = inl.ravel() > 0
        pairs[(i, j)] = dict(R=R, t=t.ravel(), m=m[keep], matches=len(m), inliers=int(keep.sum()))
    # tracks: union-find over (view, keypoint) nodes of the inlier matches
    parent = {}

    def find(node):
        while parent.setdefault(node, node) != node:
            node = parent[node]
        return node
    for (i, j), pr in pairs.items():
        for a, b in pr['m']:
            ra, rb = find((i, int(a))), find((j, int(b)))
            if ra != rb:
                parent[ra] = rb
    groups = {}
    for node in list(parent):
        groups.setdefault(find(node), []).append(node)
    tracks = []
    for nodes in groups.values():
        views = [v for v, _ in nodes]
        if len(nodes) >= 2 and len(set(views)) == len(views):
            tracks.append(dict(nodes))
    a, b = max(pairs, key=lambda k: pairs[k]['inliers'])
    R = {a: np.eye(3), b: pairs[(a, b)]['R']}; t = {a: np.zeros(3), b: pairs[(a, b)]['t']}
    P = lambda v: K @ np.c_[R[v], t[v]]
    X = {}
    for k, tr in enumerate(tracks):
        if a in tr and b in tr:
            X[k] = _triangulate([P(a), P(b)], [kps[a][tr[a]], kps[b][tr[b]]])
    for v in range(n):
        if v in R:
            continue
        ks = [k for k in X if v in tracks[k]]
        assert len(ks) >= 12, f'view {v}: {len(ks)} 2D-3D matches'
        ok, rv, tv, inl = cv2.solvePnPRansac(np.array([X[k] for k in ks]), np.array([kps[v][tracks[k][v]] for k in ks]), K, None,
                                             reprojectionError=2., iterationsCount=2000, confidence=.9999)
        assert ok and inl is not None and len(inl) >= 12, f'view {v}: PnP failed'
        R[v], t[v] = cv2.Rodrigues(rv)[0], tv.ravel()
    for k, tr in enumerate(tracks):  # (re)triangulate every track from all its views
        X[k] = _triangulate([P(v) for v in tr], [kps[v][i] for v, i in tr.items()])
    obs = [(k, v, i) for k, tr in enumerate(tracks) for v, i in tr.items()]
    free = [v for v in range(n) if v != a]

    def adjust(obs):
        ks = sorted({k for k, _, _ in obs}); kix = {k: i for i, k in enumerate(ks)}
        x0 = np.concatenate([np.r_[Rotation.from_matrix(R[v]).as_rotvec(), t[v]] for v in free] + [X[k] for k in ks])
        cv_ = np.array([v for _, v, _ in obs]); pk = np.array([kix[k] for k, _, _ in obs]); uv = np.array([kps[v][i] for _, v, i in obs])

        def res(x):
            Rs = {a: np.eye(3)}; ts = {a: np.zeros(3)}
            for j, v in enumerate(free):
                Rs[v] = Rotation.from_rotvec(x[6 * j:6 * j + 3]).as_matrix(); ts[v] = x[6 * j + 3:6 * j + 6]
            Xs = x[6 * len(free):].reshape(-1, 3); r = np.empty((len(obs), 2))
            for v in range(n):
                s = cv_ == v
                if s.any():
                    r[s] = _project(K, Rs[v], ts[v], Xs[pk[s]])[0] - uv[s]
            return r.ravel()
        J = lil_matrix((2 * len(obs), len(x0)), dtype=int)
        for o, (v, p) in enumerate(zip(cv_, pk)):
            if v != a:
                J[2 * o:2 * o + 2, 6 * free.index(v):6 * free.index(v) + 6] = 1
            J[2 * o:2 * o + 2, 6 * len(free) + 3 * p:6 * len(free) + 3 * p + 3] = 1
        sol = least_squares(res, x0, jac_sparsity=J, loss='huber', f_scale=1., x_scale='jac', method='trf', max_nfev=200)
        for j, v in enumerate(free):
            R[v] = Rotation.from_rotvec(sol.x[6 * j:6 * j + 3]).as_matrix(); t[v] = sol.x[6 * j + 3:6 * j + 6]
        for k, p in zip(ks, sol.x[6 * len(free):].reshape(-1, 3)):
            X[k] = p
        return np.hypot(*sol.fun.reshape(-1, 2).T)
    err = adjust(obs)
    obs = [o for o, e in zip(obs, err) if e <= drop_px]
    counts = {}
    for k, _, _ in obs:
        counts[k] = counts.get(k, 0) + 1
    obs = [o for o in obs if counts[o[0]] >= 2]
    err = adjust(obs)
    ks = sorted({k for k, _, _ in obs})
    z = {v: _project(K, R[v], t[v], np.array([X[k] for k in ks]))[1] for v in range(n)}
    return dict(R=[R[v] for v in range(n)], t=[t[v] for v in range(n)], X={k: X[k] for k in ks}, obs=obs, kps=kps, basePair=[a, b],
                K=K.tolist(),
                pairs={f'{i}-{j}': dict(matches=p['matches'], inliers=p['inliers']) for (i, j), p in pairs.items()},
                tracks=len(ks), observations=len(obs), reprojMedianPx=float(np.median(err)), reprojP95Px=float(np.quantile(err, .95)),
                allInFront=bool(all((z[v] > 0).all() for v in range(n))),
                baselineOverDepth={str(v): float(np.linalg.norm(R[v].T @ t[v]) / np.median(z[a])) for v in range(n) if v != a})


def focal_sweep(kps, matches, K, grid=np.arange(300., 481., 6.)):
    """Self-calibration of one shared focal (fx = fy, principal point fixed): a fixed-K reconstruction at each grid focal,
    cost = median reprojection error after bundle adjustment, parabola through the minimum and its neighbours. Needs >= 3 views:
    with two views the cost is flat (any focal fits), so a two-view cell takes the device focal of a three-view cell."""
    import cv2
    fixed = {}  # one K-independent inlier set for every focal: fundamental-matrix MAGSAC (1 px)
    for (i, j), m in matches.items():
        if len(m) >= 8:
            _, inl = cv2.findFundamentalMat(kps[i][m[:, 0]], kps[j][m[:, 1]], cv2.USAC_MAGSAC, 1., .99999)
            fixed[(i, j)] = m[inl.ravel() > 0]
    cost = []
    for f in grid:
        Kf = np.array(K, float); Kf[0, 0] = Kf[1, 1] = f
        try:
            cost.append(sfm(kps, fixed, Kf, ransac_px=3., drop_px=np.inf)['reprojMedianPx'])
        except AssertionError:
            cost.append(np.inf)
    c = np.array(cost); i = int(np.argmin(c)); f = float(grid[i])
    if 0 < i < len(grid) - 1 and np.isfinite(c[i - 1:i + 2]).all():
        a, b, _ = np.polyfit(grid[i - 1:i + 2], c[i - 1:i + 2], 2)
        f = float(-b / (2 * a)) if a > 0 else f
    return f, [[float(g), float(x)] for g, x in zip(grid, c)], bool(0 < i < len(grid) - 1)


def mono_scales(depth, rec, K):
    """One scale per frame: median of triangulated z over mono z at the frame's observed keypoints."""
    kps = rec['kps']; s, info = [], []
    for v in range(len(depth)):
        z_t, z_m = [], []
        for k, vv, i in rec['obs']:
            if vv != v:
                continue
            u, w = kps[v][i]; d = depth[v, int(round(w)), int(round(u))]
            if d > 0:
                z_t.append(_project(K, rec['R'][v], rec['t'][v], rec['X'][k][None])[1][0]); z_m.append(d)
        r = np.array(z_t) / np.array(z_m); assert len(r) >= 20, f'frame {v}: {len(r)} depth samples'
        s.append(float(np.median(r))); lr = np.log(r / s[-1])
        info.append(dict(samples=len(r), scale=s[-1], logRatioMAD=float(np.median(np.abs(lr))), logRatioP90=float(np.quantile(np.abs(lr), .9))))
    return s, info


# ---------------------------------------------------------------- self-test
def _check():
    import tempfile
    rng = np.random.default_rng(0)
    # sfm: three cameras around a box of points; recovered relative rotations / translation directions match the truth
    K = np.array([[390., 0, 258.5], [0, 390, 258.5], [0, 0, 1]])
    Xw = rng.uniform([-1, -1, 4], [1, 1, 6], (600, 3))
    from scipy.spatial.transform import Rotation
    Rs = [np.eye(3), Rotation.from_euler('y', 12, degrees=True).as_matrix(), Rotation.from_euler('xy', [4, -10], degrees=True).as_matrix()]
    ts = [np.zeros(3), np.array([-1., 0, .2]), np.array([.9, .1, .1])]
    kps, idx = [], []
    for R, t in zip(Rs, ts):
        uv, z = _project(K, R, t, Xw); kps.append(uv + rng.normal(0, .3, uv.shape)); idx.append(np.arange(len(Xw)))
    perm = [rng.permutation(len(Xw)) for _ in Rs]  # shuffle keypoint order per view
    kps = [k[p] for k, p in zip(kps, perm)]; inv = [np.argsort(p) for p in perm]
    matches = {(i, j): np.c_[inv[i], inv[j]][:500] for i in range(3) for j in range(i + 1, 3)}
    rec = sfm(kps, matches, K)
    a = rec['basePair'][0]
    for v in range(3):
        Rrel_true = Rs[v] @ Rs[a].T; Rrel = rec['R'][v] @ rec['R'][a].T
        assert np.degrees(np.arccos(np.clip((np.trace(Rrel_true.T @ Rrel) - 1) / 2, -1, 1))) < .2, v
    C = [-R.T @ t for R, t in zip(rec['R'], rec['t'])]; Ct = [-R.T @ t for R, t in zip(Rs, ts)]
    s = np.linalg.norm(Ct[1] - Ct[0]) / np.linalg.norm(C[1] - C[0])
    Ra = Rs[a].T  # rec world = camera a's frame; true world -> camera a: Rs[a]
    for v in range(3):
        assert np.linalg.norm(Ra @ (C[v] * s) + Ct[a] - Ct[v]) < .02, v
    assert rec['reprojMedianPx'] < .5 and rec['allInFront'], rec['reprojMedianPx']
    f, curve, interior = focal_sweep(kps, matches, K, grid=np.arange(330., 461., 10.))
    assert interior and abs(f / 390 - 1) < .03, (f, curve)
    # contract: a synthetic depth-unprojected geometry passes; a shifted grid fails
    with tempfile.TemporaryDirectory() as tmp:
        g = Path(tmp) / 'frames' / 'frame_0001'; g.mkdir(parents=True)
        yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(float); z = 3 + .001 * xx
        pts = np.stack([(xx - 258.5) / 390 * z, (yy - 258.5) / 390 * z, z], -1).astype(np.float32)
        M = np.eye(4); M[:3, 3] = [.1, .2, .3]; pts = (pts + M[:3, 3]).astype(np.float32)
        Image.fromarray(np.zeros((SIZE, SIZE, 3), np.uint8)).save(g / 'canonical.png')
        for name, arr in (('pts3d', pts), ('conf', np.ones((SIZE, SIZE), np.float32)), ('valid_mask', np.ones((SIZE, SIZE), bool)),
                          ('intrinsics', K.astype(np.float32)), ('camera_to_world', M.astype(np.float32))):
            np.save(g / f'{name}.npy', arr)
        assert check_arrays(Path(tmp), 1)['frame_0001']['reprojMedianPx'] < .01
        np.save(g / 'pts3d.npy', np.roll(pts, 12, axis=1))
        try:
            check_arrays(Path(tmp), 1); raise RuntimeError('shifted grid passed')
        except AssertionError:
            pass
        np.save(g / 'pts3d.npy', pts); M[:3, :3] *= 1.01; np.save(g / 'camera_to_world.npy', M.astype(np.float32))
        try:
            check_arrays(Path(tmp), 1); raise RuntimeError('non-rigid c2w passed')
        except AssertionError:
            pass
    print('backbones self-test passed')


if __name__ == '__main__':
    _check()
