"""A dense, single-layer colour point map from official LingBot-Map predictions, in the report's world frame.

Input: a lingbot_room.py run whose predictions stay on the Modal volume (prepare --video V --run-id R; execute), the
report's cameras (DROID prediction.npz poses_c2w, one per clip frame), the clip's moving-entity masks and its uncropped
source-full.mp4. On Modal (CPU): a robust Sim3 (least-median, then Umeyama on camera centres) puts LingBot's cameras on
the report's; frames it rejects (on ME340: the cut-away shot filmed with another lens) only vote. Each fused frame's
confident, non-edge, static depth becomes points (2x bilinear samples of LingBot's 518-px depth), coloured from the
full-resolution frame. A frame adds a point only where no earlier point lies on the surface it sees at that pixel
(within --tolerance of its depth) and replaces earlier points it resolves >1.4x finer: one layer per surface, every
point from one view. Points more later views contradict than confirm are dropped; one point per --cell (grown until
the web file holds <= --max-points). `evaluate` then measures it locally against the report geometry.

  python scripts/lingbot_dense_map.py diagnose --run-id R --droid-run D --clip C --masks M --depth-run F --output DIR
  python scripts/lingbot_dense_map.py build    --run-id R --droid-run D --clip C --masks M --depth-run F --mesh PLY --output DIR [--conf 1.06]
  python scripts/lingbot_dense_map.py evaluate --droid-run D --clip C --masks M --depth-run F --mesh PLY --surface GLB --points GLB --output DIR
  python scripts/lingbot_dense_map.py --self-check
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import struct
import sys
import tarfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'modal_apps'), '/root']
from lingbot_room import app, digest, image, save, volume  # noqa: E402

FULL_WH, CROP_X0, CROP_SCALE = (1280, 720), 160.25, 1.5  # clip pixel -> full pixel: x*1.5+160.25, y*1.5+0.25
SUBTITLE_ROWS = (646, 706)  # burned-in captions of the uncropped frame (fill_scene_holes --overlay-rows 648:704, +2 px)
MASK_GROW = 7                # clip pixels a moving-entity mask is grown by
EDGE_JUMP = .03              # relative depth jump to a 4-neighbour that marks a flying pixel (mono_room.unreliable)
MIN_COS = .12                # grazing view (> ~83 deg to the normal): depth unreliable
REPLACE = .7                 # a later view replaces a point when its own sample is this much finer
MAX_SPLAT = 8                # samples; ponytail: coverage discs are capped here (a coarse point seen very close leaves gaps
                             # a finer view fills by replacing it instead)
dense_image = image.add_local_file(ROOT / 'modal_apps' / 'lingbot_room.py', '/root/lingbot_room.py')
for _name in ['build_lingbot_replay.py', 'build_replay_scene.py', 'reconstruct_room_rgb.py', 'reconstruct_tum_room.py', 'video_motion.py']:
    dense_image = dense_image.add_local_file(ROOT / 'scripts' / _name, '/review/' + _name)


# ---------------------------------------------------------------- geometry
def umeyama(src, dst):
    """Similarity (s, R, t) minimising |s R src + t - dst|^2 (Umeyama 1991)."""
    ms, md = src.mean(0), dst.mean(0)
    a, b = src - ms, dst - md
    u, sv, vt = np.linalg.svd(b.T @ a / len(src))
    d = np.diag([1., 1., np.sign(np.linalg.det(u @ vt))])
    r = u @ d @ vt
    s = float(np.trace(np.diag(sv) @ d) / a.var(0).sum())
    return s, r, md - s * r @ ms


def robust_sim3(src, dst, trials=2000, seed=0):
    """Least-median start from random centre triplets, then Umeyama on the centres within 2.5x the median residual."""
    rng = np.random.default_rng(seed)
    residual = lambda f: np.linalg.norm(src @ (f[0] * f[1]).T + f[2] - dst, axis=1)
    best = min((umeyama(src[i], dst[i]) for i in (rng.choice(len(src), 3, replace=False) for _ in range(trials))),
               key=lambda f: np.median(residual(f)))
    keep = np.ones(len(src), bool)
    for _ in range(20):
        r = residual(best)
        new = r <= 2.5 * np.median(r)
        if np.array_equal(new, keep):
            break
        keep = new
        best = umeyama(src[keep], dst[keep])
    return best, residual(best), keep


def cell_keys(points, cell):
    """One int64 per cell (21 bits per axis, offset so negative coordinates fit)."""
    q = np.floor(points / cell).astype(np.int64) + (1 << 20)
    return (q[:, 0] << 42) | (q[:, 1] << 21) | q[:, 2]


def best_per_cell(keys, rank):
    """Indices of one row per key: the one with the smallest rank."""
    order = np.lexsort((rank, keys))
    first = np.ones(len(keys), bool)
    first[1:] = keys[order][1:] != keys[order][:-1]
    return order[first]


def edges(depth, jump=EDGE_JUMP):
    """Flying pixels: a 4-neighbour's depth differs by more than `jump` of this one (as mono_room.unreliable), grown 1 px."""
    import cv2
    pad = np.pad(depth, 1, mode='edge')
    step = np.max([np.abs(depth - pad[a:a + depth.shape[0], b:b + depth.shape[1]]) for a, b in [(0, 1), (2, 1), (1, 0), (1, 2)]], 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        bad = ~(step <= jump * depth)
    return cv2.dilate(bad.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0


def facing(points):
    """|cos| between each pixel's viewing ray and its surface normal (from the camera-frame point map)."""
    du = np.gradient(points, axis=1)
    dv = np.gradient(points, axis=0)
    n = np.cross(du, dv)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.abs(np.sum(n * points, -1)) / (np.linalg.norm(n, axis=-1) * np.linalg.norm(points, axis=-1))


def disc(radius):
    import cv2
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


class Fusion:
    """Accepted points and the per-frame rule that keeps them one layer thick."""

    FIELDS = {'xyz': (np.float32, 3), 'rgb': (np.uint8, 3), 'radius': (np.float32, None), 'fine': (np.float32, None),
              'conf': (np.float32, None), 'frame': (np.int32, None), 'support': (np.int32, None), 'against': (np.int32, None),
              'fill': (bool, None)}

    def __init__(self, cell, tolerance, tolerance_floor, confident=1.):
        """Samples with conf <= `confident` are fill: they never replace or vote down a confident point, and a confident
        sample of the same surface replaces them."""
        self.cell, self.tol, self.floor, self.confident = cell, tolerance, tolerance_floor, confident
        self.n, self.cap = 0, 1 << 20
        self.a = {k: np.zeros((self.cap, w) if w else self.cap, t) for k, (t, w) in self.FIELDS.items()}
        self.alive = np.zeros(self.cap, bool)
        self.stats = {'replaced': 0, 'suppressed_samples': 0, 'added': 0}

    def grow(self, extra):
        while self.n + extra > self.cap:
            self.cap *= 2
            for k, v in self.a.items():
                self.a[k] = np.resize(v, (self.cap,) + v.shape[1:])
            self.alive = np.resize(self.alive, self.cap)
            self.alive[self.n:] = False

    def compact(self):
        keep = np.flatnonzero(self.alive[:self.n])
        for k in self.a:
            self.a[k][:len(keep)] = self.a[k][keep]
        self.alive[:] = False
        self.alive[:len(keep)] = True
        self.n = len(keep)

    def add_frame(self, frame, depth, valid, k, c2w, colour, conf, cos, dry=False):
        """depth/valid/colour/conf/cos: this frame's sample grid; k: its intrinsics; c2w: report-world camera.

        Every earlier point this frame sees at a valid sample votes: its depth agrees within the tolerance (support), or
        this frame sees up to 3x the tolerance nearer/farther (a duplicate layer or a wrong depth: against), or far beyond
        it (seen through: against), or far in front (occluded: no vote); fill samples vote only on fill points.
        dry: only count, change nothing."""
        import cv2
        h, w = depth.shape
        rot, pos = c2w[:3, :3], c2w[:3, 3]
        fine_here = depth / (.5 * (k[0, 0] + k[1, 1]))  # world size of one sample at its depth
        live = np.flatnonzero(self.alive[:self.n])
        cam = (self.a['xyz'][live] - pos) @ rot
        z = cam[:, 2]
        with np.errstate(divide='ignore', invalid='ignore'):
            u = np.rint(k[0, 0] * cam[:, 0] / z + k[0, 2])
            v = np.rint(k[1, 1] * cam[:, 1] / z + k[1, 2])
        inside = (z > 1e-6) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        live, z, u, v = live[inside], z[inside], u[inside].astype(np.int64), v[inside].astype(np.int64)
        seen = valid[v, u]
        live, z, u, v = live[seen], z[seen], u[seen], v[seen]
        observed = depth[v, u]
        tol = np.maximum(self.floor, self.tol * z)
        same = np.abs(observed - z) <= tol
        against = ~same & (observed > z - 3 * tol)
        votes = {'same': int(same.sum()), 'near_or_through': int(against.sum()), 'occluded': int((observed <= z - 3 * tol).sum())}
        if dry:
            return votes
        sure, point_fill = conf[v, u] > self.confident, self.a['fill'][live]
        replace = same & (sure | point_fill) & ((fine_here[v, u] < REPLACE * self.a['fine'][live]) | (sure & point_fill))
        self.alive[live[replace]] = False
        self.stats['replaced'] += int(replace.sum())
        voter = sure | point_fill
        np.add.at(self.a['support'], live[same & ~replace & voter], 1)
        np.add.at(self.a['against'], live[against & voter], 1)
        keep = same & ~replace  # these already represent the surface this frame sees; they cover their footprint
        covered = np.zeros((h, w), np.uint8)
        splat = np.minimum(np.rint(self.a['radius'][live[keep]] * k[0, 0] / z[keep]), MAX_SPLAT).astype(int)
        for r in np.unique(splat):
            layer = np.zeros((h, w), np.uint8)
            layer[v[keep][splat == r], u[keep][splat == r]] = 1
            covered |= cv2.dilate(layer, disc(int(r))) if r else layer
        take = valid & ~covered.astype(bool)
        self.stats['suppressed_samples'] += int((valid & covered.astype(bool)).sum())
        yy, xx = np.nonzero(take)
        d = depth[yy, xx]
        local = np.stack([(xx - k[0, 2]) / k[0, 0] * d, (yy - k[1, 2]) / k[1, 1] * d, d], 1)
        xyz = local @ rot.T + pos
        chosen = best_per_cell(cell_keys(xyz, self.cell), d - 1e3 * (conf[yy, xx] > self.confident))  # per cell: confident, then closest
        xyz, yy, xx, d = xyz[chosen], yy[chosen], xx[chosen], d[chosen]
        fine = fine_here[yy, xx]
        m = len(xyz)
        self.grow(m)
        s = slice(self.n, self.n + m)
        self.a['xyz'][s], self.a['rgb'][s], self.a['fine'][s] = xyz, colour[yy, xx], fine
        # a point stands for its cell or its sample, stretched on oblique surfaces; .75 x spacing leaves no gap between discs
        self.a['radius'][s] = .75 * np.maximum(self.cell, fine / np.maximum(cos[yy, xx], .3))
        self.a['conf'][s], self.a['frame'][s], self.a['support'][s], self.a['against'][s] = conf[yy, xx], frame, 0, 0
        self.a['fill'][s] = conf[yy, xx] <= self.confident
        self.alive[s] = True
        self.n += m
        self.stats['added'] += m
        if self.n > 2 * self.alive[:self.n].sum() + 1_000_000:
            self.compact()
        return votes | {'added': m}

    def result(self, cell, max_points):
        """Drop points more later views contradict than confirm, then one per cell (the finest sample; fill points, mostly
        ceiling and far background, per cell twice as wide), cells grown together until the map fits."""
        self.compact()
        a = {k: v[:self.n] for k, v in self.a.items()}
        seen_through = (a['against'] >= 2) & (a['against'] > a['support'])
        a = {k: v[~seen_through] for k, v in a.items()}
        tiers = [np.flatnonzero(~a['fill']), np.flatnonzero(a['fill'])]
        while True:
            chosen = np.concatenate([t[best_per_cell(cell_keys(a['xyz'][t], cell * width), a['fine'][t] - 1e-9 * a['conf'][t])]
                                     for t, width in zip(tiers, (1, 2))])
            if len(chosen) <= max_points:
                return {k: v[chosen] for k, v in a.items()}, cell, int(seen_through.sum())
            cell *= 1.1


def layer_spread(points, centres, radius, min_points=20):
    """Per patch (the points whose nearest centre is within `radius`): p90-p10 of their distances to the patch's own
    best-fit plane. One clean layer gives its noise; stacked layers give their separation."""
    from scipy.spatial import cKDTree
    distance, patch = cKDTree(centres).query(points, distance_upper_bound=radius)
    hit = np.isfinite(distance)
    patch, pts = patch[hit], np.asarray(points[hit], np.float64)
    order = np.argsort(patch, kind='stable')
    patch, pts = patch[order], pts[order]
    bounds = np.searchsorted(patch, np.arange(len(centres) + 1))
    spread = np.full(len(centres), np.nan)
    for i in range(len(centres)):
        p = pts[bounds[i]:bounds[i + 1]]
        if len(p) >= min_points:
            p = p - p.mean(0)
            normal = np.linalg.eigh(p.T @ p)[1][:, 0]
            spread[i] = np.subtract(*np.percentile(p @ normal, [90, 10]))
    return spread


def self_spread(points, radius, count=2000, seed=0):
    """layer_spread around `count` of the map's own points, so every map is judged on the surfaces it has."""
    centres = points[np.random.default_rng(seed).choice(len(points), min(count, len(points)), replace=False)]
    return layer_spread(points, centres, radius)


def spread_summary(spread, metres):
    spread = np.array([np.nan if x is None else x for x in spread], float)
    s = spread[np.isfinite(spread)] * metres * 100
    if not len(s):
        return {'patches': 0}
    return {'patches': int(len(s)), 'median_cm': round(float(np.median(s)), 2), 'p75_cm': round(float(np.percentile(s, 75)), 2),
            'p90_cm': round(float(np.percentile(s, 90)), 2), 'share_over_2cm': round(float((s > 2).mean()), 3),
            'share_over_4cm': round(float((s > 4).mean()), 3)}


# ---------------------------------------------------------------- files
def write_points_glb(path, xyz, rgb):
    """GLB points: float32 POSITION + normalized uint8 RGBA COLOR_0, one buffer, no compression."""
    n = len(xyz)
    pos = np.ascontiguousarray(xyz, '<f4')
    col = np.ascontiguousarray(np.concatenate([rgb, np.full((n, 1), 255, np.uint8)], 1))
    blob = pos.tobytes() + col.tobytes()
    gltf = {'asset': {'version': '2.0', 'generator': 'panoptes lingbot_dense_map.py'}, 'scene': 0, 'scenes': [{'nodes': [0]}],
            'nodes': [{'name': 'lingbot-dense-points', 'mesh': 0}],
            'meshes': [{'name': 'lingbot-dense-points', 'primitives': [{'attributes': {'POSITION': 0, 'COLOR_0': 1}, 'mode': 0}]}],
            'buffers': [{'byteLength': len(blob)}],
            'bufferViews': [{'buffer': 0, 'byteOffset': 0, 'byteLength': 12 * n, 'target': 34962},
                            {'buffer': 0, 'byteOffset': 12 * n, 'byteLength': 4 * n, 'target': 34962}],
            'accessors': [{'bufferView': 0, 'componentType': 5126, 'count': n, 'type': 'VEC3',
                           'min': pos.min(0).astype(float).tolist(), 'max': pos.max(0).astype(float).tolist()},
                          {'bufferView': 1, 'componentType': 5121, 'normalized': True, 'count': n, 'type': 'VEC4'}]}
    head = json.dumps(gltf, separators=(',', ':')).encode()
    head += b' ' * (-len(head) % 4)
    blob += b'\0' * (-len(blob) % 4)
    with open(path, 'wb') as out:
        out.write(struct.pack('<III', 0x46546C67, 2, 28 + len(head) + len(blob)))
        out.write(struct.pack('<II', len(head), 0x4E4F534A) + head)
        out.write(struct.pack('<II', len(blob), 0x004E4942) + blob)


def read_points_glb(path):
    """Positions and RGB of a single POINTS primitive (the layout write_points_glb and trimesh write)."""
    raw = Path(path).read_bytes()
    size = struct.unpack('<I', raw[12:16])[0]
    gltf = json.loads(raw[20:20 + size])
    blob = raw[28 + size:]
    prim = gltf['meshes'][0]['primitives'][0]

    def accessor(index, dtype, width):
        a = gltf['accessors'][index]
        view = gltf['bufferViews'][a['bufferView']]
        start = view.get('byteOffset', 0) + a.get('byteOffset', 0)
        return np.frombuffer(blob, dtype, a['count'] * width, start).reshape(-1, width)
    return accessor(prim['attributes']['POSITION'], '<f4', 3), accessor(prim['attributes']['COLOR_0'], np.uint8, 4)[:, :3]


# ---------------------------------------------------------------- the frames, on Modal
def sample_grid(lb_shape, sample):
    """Full-frame pixel (x, y) of every sample of the `sample`x grid over LingBot's raster."""
    h, w = lb_shape
    v2, u2 = np.mgrid[0:sample * h, 0:sample * w].astype(np.float64)
    lb_u, lb_v = (u2 + .5) / sample - .5, (v2 + .5) / sample - .5  # cv2.resize INTER_LINEAR pixel centres
    fx = (lb_u + .5) * FULL_WH[0] / w - .5  # the official loader's PIL resize, pixel centres
    fy = (lb_v + .5) * FULL_WH[1] / h - .5
    return fx.astype(np.float32), fy.astype(np.float32)


def moving_lookup(mask_png_bytes, fx, fy):
    """Moving-entity mask on the sample grid: clip masks grown, and where a mask touches the crop's side the whole
    uncropped band beside it in those rows (the masks only cover the 4:3 crop)."""
    import cv2
    clip = np.zeros((480, 640), np.uint8)
    for raw in mask_png_bytes:
        clip |= (cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)
    if not clip.any():
        return np.zeros(fx.shape, bool)
    clip = cv2.dilate(clip, disc(MASK_GROW))
    cx = np.rint((fx - CROP_X0) / CROP_SCALE).astype(int)
    cy = np.clip(np.rint((fy - .25) / CROP_SCALE).astype(int), 0, 479)
    inside = (cx >= 0) & (cx < 640)
    out = np.zeros(fx.shape, bool)
    out[inside] = clip[cy[inside], cx[inside]] > 0
    left, right = clip[:, :3].any(1), clip[:, -3:].any(1)
    out |= (cx < 0) & left[cy]
    out |= (cx >= 640) & right[cy]
    return out


def lingbot_cameras(result_dir, files):
    w2c = []
    for f in files:
        with np.load(result_dir / f['file']) as data:
            w2c.append((data['w2c'].astype(np.float64), data['k'].astype(np.float64)))
    c2w = []
    for m, _ in w2c:
        full = np.eye(4)
        full[:3] = m
        c2w.append(np.linalg.inv(full))  # saved extrinsic is W2C (build_lingbot_replay.read_prediction)
    return np.array(c2w), np.array([k for _, k in w2c])


def align(result_dir, files, poses):
    """Robust Sim3 from LingBot camera centres to the report's, and what it leaves."""
    from scipy.spatial.transform import Rotation
    c2w, ks = lingbot_cameras(result_dir, files)
    report = poses[[f['sourceFrame'] for f in files]].astype(np.float64)
    (s, r, t), residual, inliers = robust_sim3(c2w[:, :3, 3], report[:, :3, 3])
    turned = np.einsum('ij,njk->nik', r, c2w[:, :3, :3])
    angle = np.degrees(Rotation.from_matrix(np.einsum('nji,njk->nik', turned, report[:, :3, :3])).magnitude())
    extent = np.linalg.svd(report[:, :3, 3] - report[:, :3, 3].mean(0), compute_uv=False) / np.sqrt(len(files))
    fov = np.degrees(2 * np.arctan([[k[0, 2] / k[0, 0], k[1, 2] / k[1, 1]] for k in ks]))
    return {'scale': s, 'rotation': r.tolist(), 'translation': t.tolist(), 'frames': len(files), 'inliers': int(inliers.sum()),
            'residual_native': {'median': float(np.median(residual)), 'p90': float(np.percentile(residual, 90)), 'max': float(residual.max()),
                                'rms_inliers': float(np.sqrt(np.mean(residual[inliers] ** 2)))},
            'rotation_disagreement_deg': {'median': float(np.median(angle)), 'p90': float(np.percentile(angle, 90)), 'max': float(angle.max())},
            'report_trajectory_rms_extent_native': extent.tolist(),
            'lingbot_fov_deg_median_xy': np.median(fov, 0).tolist(), 'lingbot_fov_deg_p10_p90_x': np.percentile(fov[:, 0], [10, 90]).tolist(),
            'lingbot_fov_deg_median_x_inliers': float(np.median(fov[inliers, 0])), 'lingbot_fov_deg_median_x_outliers': float(np.median(fov[~inliers, 0])) if (~inliers).any() else None,
            'outlier_source_frames': [f['sourceFrame'] for f, ok in zip(files, inliers) if not ok]}, (s, r, t), inliers


def frame_stream(root, files, sim3, clip_k, conf_floor, sample=3, overlay=SUBTITLE_ROWS, masks_tar='dynamic-masks.tar'):
    """Per LingBot frame, in the report world: sample-grid depth, validity, K, camera, full-res colour, conf, facing.
    overlay: rows [y0, y1) of the uncropped frame under burned-in captions, never used (y0 == y1: none).
    masks_tar: the moving-entity masks upload_masks put on the volume (named by their content)."""
    import cv2
    s, r, t = sim3
    masks = {}
    with tarfile.open(root / masks_tar) as archive:
        for m in archive.getmembers():
            masks.setdefault(int(m.name.split('-')[0]), []).append(archive.extractfile(m).read())
    wanted = {f['sourceFrame']: f for f in files}
    capture = cv2.VideoCapture(str(root / 'source.mp4'))
    grid = None
    index = -1
    try:
        while wanted:
            ok, bgr = capture.read()
            index += 1
            if not ok:
                raise ValueError(f'video ended before frame {min(wanted)}')
            f = wanted.pop(index, None)
            if f is None:
                continue
            with np.load(root / 'result' / f['file']) as data:
                depth, conf, k, w2c = data['depth'][..., 0].astype(np.float64), data['depth_conf'].astype(np.float64), data['k'].astype(np.float64), data['w2c'].astype(np.float64)
            h, w = depth.shape
            if grid is None:
                fx, fy = sample_grid((h, w), sample)
                grid = (fx, fy, (fy >= overlay[0]) & (fy < overlay[1]))
            fx, fy, subtitles = grid
            full = np.eye(4)
            full[:3] = w2c
            lb_c2w = np.linalg.inv(full)
            c2w = np.eye(4)
            c2w[:3, :3] = r @ lb_c2w[:3, :3]
            c2w[:3, 3] = s * r @ lb_c2w[:3, 3] + t
            depth = s * depth
            moving = moving_lookup(masks.get(index, []), fx, fy)
            blocked = moving | subtitles
            ok_lb = np.isfinite(depth) & (depth > 0) & (conf > conf_floor) & ~edges(depth) & ~blocked[::sample, ::sample]
            k2 = k.copy()
            k2[:2, :2] *= sample
            k2[:2, 2] = sample * k[:2, 2] + (sample - 1) / 2  # sample u2 <-> LingBot u = (u2 + .5)/sample - .5
            size = (sample * w, sample * h)
            d2 = cv2.resize(depth.astype(np.float32), size, interpolation=cv2.INTER_LINEAR)
            ok2 = cv2.resize(ok_lb.astype(np.float32), size, interpolation=cv2.INTER_LINEAR) > .999  # all 4 source pixels valid
            conf2 = cv2.resize(conf.astype(np.float32), size, interpolation=cv2.INTER_LINEAR)
            yy, xx = np.mgrid[0:size[1], 0:size[0]]
            cam = np.stack([(xx - k2[0, 2]) / k2[0, 0] * d2, (yy - k2[1, 2]) / k2[1, 1] * d2, d2], -1)
            cos = facing(cam)
            valid = ok2 & ~blocked & (cos > MIN_COS)
            rgb = cv2.remap(bgr, fx, fy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)[..., ::-1]
            yield {'source': index, 'depth': d2, 'valid': valid, 'k': k2, 'c2w': c2w, 'rgb': rgb, 'conf': conf2, 'cos': cos,
                   'masks': masks.get(index, []), 'lb': {'depth': depth, 'conf': conf, 'k': k, 'c2w': c2w, 'ok': ok_lb}}
    finally:
        capture.release()


def load_run(run_id):
    root = Path('/artifact') / run_id
    execution = json.loads((root / 'result' / 'run.json').read_text())
    if execution['status'] != 'inference_complete' or execution['plan_sha256'] != digest(root / 'plan.json'):
        raise ValueError('Require a complete, plan-bound LingBot execution')
    return root, execution


@app.function(image=dense_image, cpu=(4, 4), memory=(16384, 16384), timeout=1800, retries=0, volumes={'/artifact': volume})
def diagnose_remote(run_id, poses, clip_k, every=5, lag=5, conf_floor=1., overlay=SUBTITLE_ROWS, masks_tar='dynamic-masks.tar'):
    """Alignment, confidence distribution, and how far neighbouring views' depths disagree (per confidence decile)."""
    volume.reload()
    root, execution = load_run(run_id)
    files = execution['frames']
    report, sim3, _ = align(root / 'result', files, np.asarray(poses))
    confs, pairs = [], []
    views = {}
    for f in frame_stream(root, files, sim3, clip_k, conf_floor, overlay=overlay, masks_tar=masks_tar):
        lb = f['lb']
        views[f['source']] = lb
        confs.append(lb['conf'][::4, ::4].ravel())
    sources = sorted(views)
    for i in range(0, len(sources) - lag, every):
        a, b = views[sources[i]], views[sources[i + lag]]
        ok = a['ok']
        yy, xx = np.nonzero(ok)
        d = a['depth'][yy, xx]
        local = np.stack([(xx - a['k'][0, 2]) / a['k'][0, 0] * d, (yy - a['k'][1, 2]) / a['k'][1, 1] * d, d], 1)
        world = local @ a['c2w'][:3, :3].T + a['c2w'][:3, 3]
        cam = (world - b['c2w'][:3, 3]) @ b['c2w'][:3, :3]
        z = cam[:, 2]
        with np.errstate(divide='ignore', invalid='ignore'):
            u = np.rint(b['k'][0, 0] * cam[:, 0] / z + b['k'][0, 2])
            v = np.rint(b['k'][1, 1] * cam[:, 1] / z + b['k'][1, 2])
        h, w = b['depth'].shape
        inside = (z > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        u, v, z, c = u[inside].astype(int), v[inside].astype(int), z[inside], a['conf'][yy, xx][inside]
        seen = b['ok'][v, u]
        pairs.append(np.stack([(b['depth'][v, u][seen] - z[seen]) / z[seen], c[seen], z[seen]], 1)[::7])
    confs = np.concatenate(confs)
    rel = np.concatenate(pairs)
    deciles = np.percentile(confs, np.arange(0, 101, 10))
    by_conf = []
    for lo, hi in zip(deciles[:-1], deciles[1:]):
        x = rel[(rel[:, 1] >= lo) & (rel[:, 1] <= hi), 0]
        if len(x):
            by_conf.append({'conf_from': float(lo), 'conf_to': float(hi), 'samples': int(len(x)), 'median_abs_rel': float(np.median(np.abs(x))),
                            'share_within_2pct': float((np.abs(x) < .02).mean()), 'share_within_4pct': float((np.abs(x) < .04).mean()),
                            'share_within_8pct': float((np.abs(x) < .08).mean())})
    a = np.abs(rel[:, 0])
    c2w, ks = lingbot_cameras(root / 'result', files)
    return {'alignment': report, 'frames': len(files), 'pair_lag_frames': lag, 'pairs': len(pairs),
            'cameras': {'source': [f['sourceFrame'] for f in files], 'k': ks.tolist(), 'c2w': c2w.tolist()},
            'conf_quantiles': {str(q): float(np.percentile(confs, q)) for q in [1, 5, 10, 25, 50, 75, 90]},
            'share_conf_over': {str(c): float((confs > c).mean()) for c in [1.2, 1.5, 2, 3, 5, 10]},
            'neighbour_rel_depth_diff': {'samples': int(len(a)), 'median_abs': float(np.median(a)), 'p75_abs': float(np.percentile(a, 75)),
                                         'p90_abs': float(np.percentile(a, 90)), 'median_signed': float(np.median(rel[:, 0])),
                                         'share_within': {str(p): float((a < p).mean()) for p in [.01, .02, .03, .04, .06, .08, .12]}},
            'by_conf_decile': by_conf}


def build_map(root, files, poses, clip_k, config, out):
    """Fuse the Sim3-inlier frames into one layer; the outlier frames only vote (dry) against the finished map."""
    import cv2
    import tempfile
    from build_lingbot_replay import check_temporal_geometry
    started = time.time()
    report, sim3, inliers = align(root / 'result', files, np.asarray(poses))
    excluded = lambda s: any(a <= s < b for a, b in config.get('exclude', ()))  # a cut-away shot: its cameras are wrong, never fused
    fused = {f['sourceFrame'] for f, ok in zip(files, inliers) if ok and not excluded(f['sourceFrame'])}
    fusion = Fusion(config['cell'], config['tolerance'], config['tolerance_floor'], config['conf'])
    naive = [np.zeros((0, 3), np.float32)], [np.zeros(0, np.float32)]  # every valid LingBot-resolution sample, no suppression
    flow_masks = Path(tempfile.mkdtemp(prefix='flow-masks-'))
    analysis = {'frames': [], 'height': FULL_WH[1], 'width': FULL_WH[0], '_base': str(flow_masks)}
    full_grid = [g.astype(np.float32) for g in np.meshgrid(np.arange(FULL_WH[0]), np.arange(FULL_WH[1]))]
    votes, held, valid_share, step = {}, [], [], config['sample']
    for f in frame_stream(root, files, sim3, clip_k, config['fill_conf'], step, tuple(config.get('overlay_rows', SUBTITLE_ROWS)),
                          config.get('masks_tar', 'dynamic-masks.tar')):
        objects = []
        if f['masks']:  # the project's cross-view check leaves out the same moving pixels, as source-raster RGBA masks
            alpha = moving_lookup(f['masks'], *full_grid).astype(np.uint8) * 255
            name = f"{f['source']:05d}.png"
            assert cv2.imwrite(str(flow_masks / name), np.dstack([alpha, alpha, alpha, alpha]))
            objects.append({'maskUrl': name})
        analysis['frames'].append({'sourceFrame': f['source'], 'objects': objects})
        if f['source'] not in fused:
            held.append({k: f[k] for k in ('source', 'depth', 'valid', 'k', 'c2w', 'rgb', 'conf', 'cos')})
            continue
        votes[f['source']] = fusion.add_frame(f['source'], f['depth'], f['valid'], f['k'], f['c2w'], f['rgb'], f['conf'], f['cos'])
        valid_share.append(float(f['valid'].mean()))
        yy, xx = np.nonzero(f['valid'][::step, ::step])
        yy, xx = yy * step, xx * step
        d = f['depth'][yy, xx]
        local = np.stack([(xx - f['k'][0, 2]) / f['k'][0, 0] * d, (yy - f['k'][1, 2]) / f['k'][1, 1] * d, d], 1)
        naive[0].append((local @ f['c2w'][:3, :3].T + f['c2w'][:3, 3]).astype(np.float32))
        naive[1].append(d.astype(np.float32))
        if len(votes) % 10 == 0 or len(votes) == len(fused):
            xyz, depth = np.concatenate(naive[0]), np.concatenate(naive[1])
            chosen = best_per_cell(cell_keys(xyz, config['cell']), depth)  # the naive map keeps the closest per cell
            naive = [xyz[chosen]], [depth[chosen]]
    held_votes = {h['source']: fusion.add_frame(h['source'], h['depth'], h['valid'], h['k'], h['c2w'], h['rgb'], h['conf'], h['cos'], dry=True) for h in held}
    del held
    temporal = check_temporal_geometry(root / 'result', analysis)
    points, cell, dropped = fusion.result(config['cell'], config['max_points'])
    out.mkdir(exist_ok=True)
    write_points_glb(out / 'dense-points.glb', points['xyz'], points['rgb'])
    np.savez_compressed(out / 'point-attributes.npz', source_frame=points['frame'].astype(np.int16), fill=points['fill'],
                        spacing=(points['radius'] / .75).astype(np.float16), conf=points['conf'].astype(np.float16),
                        support=np.minimum(points['support'], 32767).astype(np.int16))
    mine = self_spread(points['xyz'], config['patch_radius'])
    naive_spread = self_spread(naive[0][0], config['patch_radius'])
    agree = lambda v: {k: int(sum(x[k] for x in v.values())) for k in ('same', 'near_or_through', 'occluded')}
    finite = lambda a: [x if x == x else None for x in a.tolist()]  # None: patch too sparse
    remote = {'alignment': report, 'config': config, 'cell_native_final': cell, 'points': int(len(points['xyz'])),
              'fill_points': int(points['fill'].sum()), 'dropped_contradicted': dropped, 'fusion': fusion.stats,
              'frames': len(files), 'fused_frames': len(votes), 'added_per_frame_median': float(np.median([v['added'] for v in votes.values()])),
              'valid_sample_share_median': float(np.median(valid_share)),
              'votes_of_fused_frames_on_earlier_points': agree(votes), 'votes_of_held_out_frames_on_map': agree(held_votes),
              'held_out_votes_per_frame': {str(k): v for k, v in held_votes.items()},
              'support_views_median': float(np.median(points['support'])), 'support_zero_share': float((points['support'] == 0).mean()),
              'conf_of_points_median': float(np.median(points['conf'])), 'naive_points': int(len(naive[0][0])),
              'layer_spread_native': {'this_map': finite(mine), 'lingbot_naive_same_frames_and_filters': finite(naive_spread)},
              'cross_view_check': temporal, 'elapsed_seconds': time.time() - started}
    save(out / 'remote.json', remote)
    return {k: v for k, v in remote.items() if k not in ('layer_spread_native', 'held_out_votes_per_frame', 'cross_view_check')}


@app.function(image=dense_image, cpu=(4, 4), memory=(24576, 24576), timeout=3600, retries=0, volumes={'/artifact': volume})
def build_remote(run_id, poses, clip_k, config):
    sys.path.insert(0, '/review')
    volume.reload()
    root, execution = load_run(run_id)
    summary = build_map(root, execution['frames'], np.asarray(poses), clip_k, config, root / 'dense')
    volume.commit()
    return summary


# ---------------------------------------------------------------- local: inputs, metrics, images
def report_inputs(droid_run, clip):
    data = np.load(droid_run / 'prediction.npz')
    k = json.loads((clip / 'clip.json').read_text())['K']
    return data['poses_c2w'].astype(np.float64), [float(x) for x in k], data


def full_k(clip_k):
    fx, fy, cx, cy = clip_k
    return np.array([[CROP_SCALE * fx, 0, CROP_SCALE * cx + CROP_X0], [0, CROP_SCALE * fy, CROP_SCALE * cy + .25], [0, 0, 1.]])


def raster_k(clip_k):
    """The report's 640x480 working raster (droid_room.prepare_image at scale 2: resize to 704x512, crop 32/16 px)."""
    fx, fy, cx, cy = clip_k
    return np.array([[fx * 1.1, 0, cx * 1.1 - 32], [0, fy * 512 / 480, cy * 512 / 480 - 16], [0, 0, 1.]])


def masks_tar_name(masks):
    """The volume name of one masks set: dynamic-masks-<digest of its (file name, bytes)>.tar."""
    h = hashlib.sha256()
    for path in sorted(masks.glob('*.png')):
        h.update(path.name.encode() + b'\0' + hashlib.sha256(path.read_bytes()).digest())
    return f'dynamic-masks-{h.hexdigest()[:16]}.tar'


def upload_masks(run_id, masks):
    """Put these masks on the volume under the run id, named by their content, and return that name: a run id built
    again with other masks gets their own tar (it used to reuse whatever {run_id}/dynamic-masks.tar held)."""
    name = masks_tar_name(masks)
    if name in {e.path.split('/')[-1] for e in volume.iterdir(run_id)}:
        return name
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w') as archive:
        for path in sorted(masks.glob('*.png')):
            archive.add(path, arcname=path.name)
    with volume.batch_upload() as batch:
        batch.put_file(io.BytesIO(buffer.getvalue()), f'{run_id}/{name}')
    return name


def fetch(run_id, name, target):
    with open(target, 'xb') as out:
        for block in volume.read_file(f'{run_id}/{name}'):
            out.write(block)


def mesh_samples(mesh_path, count, seed=0):
    import trimesh
    mesh = trimesh.load(mesh_path, process=False)
    return np.asarray(trimesh.sample.sample_surface(mesh, count, seed=seed)[0])


def project(points, k, c2w, wh, near=.03, exact=False):
    cam = (points - c2w[:3, 3]) @ c2w[:3, :3]
    z = cam[:, 2]
    with np.errstate(divide='ignore', invalid='ignore'):
        u, v = k[0, 0] * cam[:, 0] / z + k[0, 2], k[1, 1] * cam[:, 1] / z + k[1, 2]
    inside = (z > near) & (u > -.5) & (u < wh[0] - .5) & (v > -.5) & (v < wh[1] - .5)
    if exact:
        return u[inside], v[inside], z[inside], inside
    return np.rint(u[inside]).astype(int), np.rint(v[inside]).astype(int), z[inside], inside


def point_coverage(points, size, k, c2w, wh):
    """Pixels whose centre falls inside some point's square of side `size` (scalar or per point), as projected (any
    depth, like a ray hit); a point always covers its own pixel. Rectangles are summed with a 2D difference array."""
    u, v, z, inside = project(points, k, c2w, wh, exact=True)
    half = .5 * np.broadcast_to(size, len(points))[inside] * k[0, 0] / z
    x0, x1 = np.ceil(u - half), np.floor(u + half)
    y0, y1 = np.ceil(v - half), np.floor(v + half)
    x0, x1 = np.where(x1 < x0, np.rint(u), x0), np.where(x1 < x0, np.rint(u), x1)
    y0, y1 = np.where(y1 < y0, np.rint(v), y0), np.where(y1 < y0, np.rint(v), y1)
    x0, y0 = np.clip(x0, 0, wh[0] - 1).astype(int), np.clip(y0, 0, wh[1] - 1).astype(int)
    x1, y1 = np.clip(x1, 0, wh[0] - 1).astype(int) + 1, np.clip(y1, 0, wh[1] - 1).astype(int) + 1
    diff = np.zeros((wh[1] + 1, wh[0] + 1), np.int64)
    for dy, dx, sign in ((y0, x0, 1), (y0, x1, -1), (y1, x0, -1), (y1, x1, 1)):
        np.add.at(diff, (dy, dx), sign)
    return diff.cumsum(0).cumsum(1)[:wh[1], :wh[0]] > 0


def surface_scene(path):
    """Ray-cast scene of a GLB's triangles with what colours each hit: per geometry, texture+UV or vertex colours."""
    import open3d as o3d
    import trimesh
    loaded = trimesh.load(path, process=False)
    parts, offset, verts, faces = [], 0, [], []
    for name, g in loaded.geometry.items():
        visual = g.visual
        texture = getattr(getattr(visual, 'material', None), 'baseColorTexture', None)
        colour = (np.asarray(texture.convert('RGB')), np.asarray(visual.uv)) if texture is not None and visual.uv is not None else np.asarray(visual.vertex_colors)[:, :3]
        parts.append((offset, offset + len(g.faces), np.asarray(g.faces), colour))
        verts.append(np.asarray(g.vertices))
        faces.append(np.asarray(g.faces) + sum(len(x) for x in verts[:-1]))
        offset += len(g.faces)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(np.concatenate(verts).astype(np.float32)), o3d.core.Tensor(np.concatenate(faces).astype(np.uint32)))
    return scene, parts


def cast(scene, k, c2w, u, v):
    import open3d as o3d
    d = np.stack([(u - k[0, 2]) / k[0, 0], (v - k[1, 2]) / k[1, 1], np.ones_like(u, float)], -1).reshape(-1, 3) @ c2w[:3, :3].T
    rays = np.hstack([np.broadcast_to(c2w[:3, 3], d.shape), d]).astype(np.float32)
    return scene.cast_rays(o3d.core.Tensor(rays))


def render_surface(scene, parts, k, c2w, wh):
    v, u = np.mgrid[0:wh[1], 0:wh[0]].astype(float)
    hit = cast(scene, k, c2w, u, v)
    prim = hit['primitive_ids'].numpy().astype(np.int64)
    bary = hit['primitive_uvs'].numpy()
    image_out = np.zeros((prim.size, 3), np.uint8)
    for start, stop, tri, colour in parts:
        sel = np.flatnonzero((prim >= start) & (prim < stop))
        corners = tri[prim[sel] - start]
        w = np.column_stack([1 - bary[sel].sum(1), bary[sel]])
        if isinstance(colour, tuple):
            texture, uv = colour
            st = np.einsum('nk,nkj->nj', w, uv[corners])
            x = np.clip((st[:, 0] * texture.shape[1]).astype(int), 0, texture.shape[1] - 1)
            y = np.clip(((1 - st[:, 1]) * texture.shape[0]).astype(int), 0, texture.shape[0] - 1)
            image_out[sel] = texture[y, x]
        else:
            image_out[sel] = np.einsum('nk,nkj->nj', w, colour[corners].astype(float)).astype(np.uint8)
    return image_out.reshape(wh[1], wh[0], 3)


def render_points(xyz, rgb, size, k, c2w, wh, cap=12):
    """Nearest point per pixel, each drawn as the square of side `size` (scalar or per point) it projects to."""
    u, v, z, inside = project(xyz, k, c2w, wh)
    colour = rgb[inside]
    half = np.minimum(np.ceil(.5 * np.broadcast_to(size, len(xyz))[inside] * k[0, 0] / z - .5), cap).astype(int)  # gap-free
    us, vs, zs, cs = [u], [v], [z], [colour]
    for h in range(1, half.max() + 1 if len(half) else 1):
        big = np.flatnonzero(half >= h)
        for dy in range(-h, h + 1):
            for dx in range(-h, h + 1):
                if max(abs(dx), abs(dy)) == h:
                    us.append(u[big] + dx), vs.append(v[big] + dy), zs.append(z[big]), cs.append(colour[big])
    u, v, z, colour = map(np.concatenate, (us, vs, zs, cs))
    ok = (u >= 0) & (u < wh[0]) & (v >= 0) & (v < wh[1])
    pixel = v[ok] * wh[0] + u[ok]
    order = np.lexsort((z[ok], pixel))
    first = order[np.r_[True, pixel[order][1:] != pixel[order][:-1]]]
    out = np.zeros((wh[1] * wh[0], 3), np.uint8)
    out[pixel[first]] = colour[ok][first]
    return out.reshape(wh[1], wh[0], 3)


def decode_frames(video, wanted):
    import cv2
    capture, frames, index = cv2.VideoCapture(str(video)), {}, 0
    while len(frames) < len(wanted):
        ok, bgr = capture.read()
        if not ok:
            break
        if index in wanted:
            frames[index] = bgr[..., ::-1].copy()
        index += 1
    capture.release()
    return frames


def da3_naive_spread(droid_run, depth_run, masks, radius, cell, skip=lambda index: False):
    """The report's DA3 views as a naive per-pixel map (edges and moving pixels out, closest point per cell, merged every
    20 views to bound memory), measured on its own points like the other maps."""
    import mono_room
    mono_room.use_clip(droid_run)
    data = np.load(droid_run / 'prediction.npz')
    retained = np.load(depth_run / 'droid-support.npz')['retained']
    keys = {int(index): key for key, index in enumerate(data['keyframe_source_indices'])}
    droid, poses, keyframe_poses = data['keyframe_final_fullres_depth'], data['poses_c2w'], data['keyframe_c2w']
    kr = np.asarray(mono_room.prepare_image(np.zeros((480, 640, 3), np.uint8), mono_room.CALIBRATION, 2)[1], np.float64)
    k = np.array([[kr[0], 0, kr[2]], [0, kr[1], kr[3]], [0, 0, 1.]])
    paths = sorted((depth_run / 'mono').glob('*.npz'))
    load = lambda p: (lambda m: np.where(m['mask'], m['depth'], 0).astype(np.float32))(np.load(p))
    fitted = {int(p.stem): mono_room.anchor_scale(load(p)[::4, ::4], droid[keys[int(p.stem)]][::2, ::2], retained[keys[int(p.stem)]])[0]
              for p in paths if int(p.stem) in keys}  # as mono_room.load: own DROID anchors, else the median scale
    median = float(np.median([s for s in fitted.values() if s]))
    v, u = np.mgrid[0:480, 0:640].astype(np.float32)
    xyz, depth, used = [np.zeros((0, 3), np.float32)], [np.zeros(0, np.float32)], 0
    for p in paths:
        index = int(p.stem)
        if skip(index):
            continue
        used += 1
        view = load(p) * (fitted.get(index) or median)
        keep = (view > 0) & ~mono_room.unreliable(view, None, None, EDGE_JUMP) & ~mono_room.moving_mask(masks, index)
        c2w = (keyframe_poses[keys[index]] if index in keys else poses[index]).astype(np.float64)
        d = view[keep]
        local = np.stack([(u[keep] - k[0, 2]) / k[0, 0] * d, (v[keep] - k[1, 2]) / k[1, 1] * d, d], 1)
        xyz.append((local @ c2w[:3, :3].T + c2w[:3, 3]).astype(np.float32))
        depth.append(d)
        if used % 20 == 0:
            all_xyz, all_depth = np.concatenate(xyz), np.concatenate(depth)
            chosen = best_per_cell(cell_keys(all_xyz, cell), all_depth)
            xyz, depth = [all_xyz[chosen]], [all_depth[chosen]]
    all_xyz, all_depth = np.concatenate(xyz), np.concatenate(depth)
    points = all_xyz[best_per_cell(cell_keys(all_xyz, cell), all_depth)]
    return self_spread(points, radius), used, len(points)


def evaluate(args):
    """Alignment against the fused mesh, completeness against the published surface and points, layering, images."""
    import bisect
    import cv2
    import open3d as o3d
    import trimesh
    from scipy.spatial import cKDTree
    out = args.output
    info = json.loads((out / 'points.json').read_text())
    remote = json.loads((out / 'remote.json').read_text())
    metres = info['metres_per_native_unit']
    cm = lambda x: round(float(x) * metres * 100, 2)
    xyz, rgb = read_points_glb(out / 'dense-points.glb')
    xyz = xyz.astype(np.float64)
    attrs = np.load(out / 'point-attributes.npz')
    fill = attrs['fill']
    cell = remote['cell_native_final']
    point_cell = np.where(fill, 2 * cell, cell)  # fill points were kept one per double-width cell
    own = np.maximum(point_cell, attrs['spacing'].astype(np.float64))  # what each point's source sample spans
    poses, clip_k, data = report_inputs(args.droid_run, args.clip)

    # (a) alignment: unsigned distance of every point to the report's fused mesh; "both have" = within 25 (10) cm
    fused = trimesh.load(args.mesh, process=False)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(np.asarray(fused.vertices, np.float32)), o3d.core.Tensor(np.asarray(fused.faces, np.uint32)))
    dist = scene.compute_distance(o3d.core.Tensor(xyz.astype(np.float32))).numpy()
    back = cKDTree(xyz).query(mesh_samples(args.mesh, 200_000, seed=1), distance_upper_bound=.25 / metres)[0]

    def distances(sel):
        row = {'points': int(sel.sum()), 'all_cm': {'median': cm(np.median(dist[sel])), 'p90': cm(np.percentile(dist[sel], 90))}}
        for limit in (.25, .10):
            near = sel & (dist <= limit / metres)
            row[f'within_{int(limit * 100)}cm'] = {'share': round(float(near.sum() / sel.sum()), 4), 'median_cm': cm(np.median(dist[near])),
                                                   'p90_cm': cm(np.percentile(dist[near], 90))}
        return row
    ok = np.isfinite(back)
    alignment = {'reference': str(args.mesh), 'note': 'agreement of two monocular reconstructions, not accuracy; no ground truth',
                 'all_points': distances(np.ones(len(xyz), bool)), 'confident_points': distances(~fill), 'fill_points': distances(fill),
                 'mesh_to_points_within_25cm': {'share_of_mesh_area': round(float(ok.mean()), 4), 'median_cm': cm(np.median(back[ok])),
                                                'p90_cm': cm(np.percentile(back[ok], 90))}}

    # (b) completeness: the fill_scene_holes views and 4-px grid, plus the 20 DROID keyframes and the uncropped frame
    keys = {int(i): kk for kk, i in enumerate(data['keyframe_source_indices'])}
    camera = lambda i: (data['keyframe_c2w'][keys[i]] if i in keys else poses[i]).astype(np.float64)
    da3_views = sorted(int(p.stem) for p in (args.depth_run / 'mono').glob('*.npz'))
    judged, keyframes = da3_views[::5], sorted(keys)
    published_xyz = read_points_glb(args.points)[0].astype(np.float64)
    published_cell = json.loads((args.depth_run / 'fuse-metrics.json').read_text())['voxel_native']
    surface, parts = surface_scene(args.surface)
    kr, kf = raster_k(clip_k), full_k(clip_k)
    bounds = [0] + info['shot_cuts_source_frames']
    segment = lambda i: bisect.bisect_right(bounds, i) - 1
    held = set(info['held_out_frames']['source_frames'])
    fused_segments = {segment(f['sourceFrame']) for f in json.loads((out / 'plan.json').read_text())['frames'] if f['sourceFrame'] not in held}
    cut = lambda i: segment(i) not in fused_segments  # a shot none of whose frames is in this map

    def coverage(views, k, wh, step):
        gv, gu = np.mgrid[step // 2:wh[1]:step, step // 2:wh[0]:step]
        pick = lambda covered: covered[gv, gu].mean()
        rows = {'this_map_cell_footprint': [], 'this_map_own_sample_footprint': [], 'this_map_confident_only_cell_footprint': [],
                'published_points_cell_footprint': [], 'published_surface_ray_hit': []}
        for i in views:
            c2w = camera(i)
            rows['this_map_cell_footprint'].append(pick(point_coverage(xyz, point_cell, k, c2w, wh)))
            rows['this_map_own_sample_footprint'].append(pick(point_coverage(xyz, own, k, c2w, wh)))
            rows['this_map_confident_only_cell_footprint'].append(pick(point_coverage(xyz[~fill], cell, k, c2w, wh)))
            rows['published_points_cell_footprint'].append(pick(point_coverage(published_xyz, published_cell, k, c2w, wh)))
            rows['published_surface_ray_hit'].append(np.isfinite(cast(surface, k, c2w, gu.astype(float), gv.astype(float))['t_hit'].numpy()).mean())
        stats = lambda x: {'mean': round(float(np.mean(x)), 4), 'median': round(float(np.median(x)), 4), 'worst': round(float(np.min(x)), 4)} if len(x) else None
        groups = {'all': np.ones(len(views), bool), 'shots_in_map': np.array([not cut(i) for i in views]), 'shots_not_in_map': np.array([cut(i) for i in views])}
        return {g: {'views': int(m.sum()), **{name: stats(np.asarray(x)[m]) for name, x in rows.items()}} for g, m in groups.items()}
    completeness = {'rule': ('share of a view\'s grid pixels whose ray meets geometry: points as the square of their cell (as asked) or of '
                             'their own source sample, at any depth; the surface by ray casting (fill_scene_holes.coverage). Report '
                             'cameras and K. The cut-away shot was filmed with another lens, so the report\'s single K does not fit it, '
                             'and this map holds none of its frames'),
                    'fill_scene_holes_views_640x480_raster_4px': coverage(judged, kr, (640, 480), 4),
                    'droid_keyframes_640x480_raster_4px': coverage(keyframes, kr, (640, 480), 4),
                    'fill_scene_holes_views_uncropped_1280x720_8px': coverage(judged, kf, FULL_WH, 8)}

    # layering: each map on its own points (this map and the naive LingBot map remotely; the naive DA3 map here)
    radius = info['patch_radius_native']
    da3, da3_views_used, da3_count = da3_naive_spread(args.droid_run, args.depth_run, args.masks, radius, cell)
    da3_wide, da3_wide_used, da3_wide_count = da3_naive_spread(args.droid_run, args.depth_run, args.masks, radius, cell, skip=cut)
    layering = {'rule': f"per map, 2000 of its own points as patch centres: p90-p10 of the distances of the map's points within {cm(radius)} cm "
                        "to the patch's best-fit plane (>=20 points). One layer gives its noise and curvature; stacked layers their separation",
                'cell_native_all_maps': cell,
                'this_map': {**spread_summary(remote['layer_spread_native']['this_map'], metres), 'points': len(xyz)},
                'lingbot_naive_same_frames_and_filters': {**spread_summary(remote['layer_spread_native']['lingbot_naive_same_frames_and_filters'], metres),
                                                          'points': remote['naive_points'], 'rule': 'every valid LingBot-resolution sample of the fused frames, closest per cell, no suppression'},
                'da3_naive_report_views': {**spread_summary(da3, metres), 'views': da3_views_used, 'points': da3_count},
                'da3_naive_views_of_shots_in_this_map': {**spread_summary(da3_wide, metres), 'views': da3_wide_used, 'points': da3_wide_count}}

    # comparison images: real frame | this map | published room surface, four keyframes of the mapped shots
    candidates = [i for i in keyframes if not cut(i)]  # the report's single K fits only the shots in this map
    shown = [candidates[i] for i in np.linspace(1, len(candidates) - 2, 4).round().astype(int)]
    frames = decode_frames(args.clip / 'source-full.mp4', set(shown))
    images = []
    for i in shown:
        c2w = camera(i)
        panels = [frames[i], render_points(xyz, rgb, own, kf, c2w, FULL_WH), render_surface(surface, parts, kf, c2w, FULL_WH)]
        strip = np.hstack([cv2.resize(p, (640, 360), interpolation=cv2.INTER_AREA) for p in panels])
        for x, label in zip((10, 650, 1290), (f'frame {i}', 'LingBot dense points (this)', 'published room surface')):
            cv2.putText(strip, label, (x, 26), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 0), 2)
        name = f'compare-{i:05d}.jpg'
        cv2.imwrite(str(out / name), strip[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 88])
        images.append(name)
    check = remote['cross_view_check']
    within = [p for p in check['pairs'] if segment(p['source_frames'][0]) == segment(p['source_frames'][1])]
    info['metrics'] = {'alignment_to_fused_mesh': alignment, 'completeness': completeness, 'layering': layering,
                       'cross_view_check': {k: check[k] for k in ('passed', 'passing_pairs', 'threshold_median_pixels', 'required_pair_fraction', 'method')}
                       | {'pairs_checked': len(check['pairs']), 'median_of_pair_medians_px': round(float(np.median([p['median_reprojection_error_px'] for p in check['pairs']])), 3),
                          'note': f'the project\'s check (build_lingbot_replay.check_temporal_geometry) run unchanged on all {remote["frames"]} LingBot frames at a 10-frame lag; '
                                  'pairs across a shot cut find too few tracks and are skipped by the check itself',
                          'pairs_within_one_shot': len(within)}}
    info['comparison_images'] = {'files': images, 'panels': 'real uncropped frame | this map, each point the square of its own sample (z-buffered) | '
                                                            f'published room surface ({args.surface.parent.name}/{args.surface.name}), ray cast', 'camera': 'report camera and full-frame K'}
    save(out / 'points.json', info)
    print(json.dumps(info['metrics'], indent=1))


def shot_cuts(video, factor=8):
    """Frames that start a new shot: mean grey change from the previous frame (160x90) above `factor` x the median."""
    import cv2
    capture, change, previous = cv2.VideoCapture(str(video)), [], None
    while True:
        ok, bgr = capture.read()
        if not ok:
            break
        grey = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (160, 90), interpolation=cv2.INTER_AREA).astype(np.float32)
        change.append(np.abs(grey - previous).mean() if previous is not None else 0.)
        previous = grey
    capture.release()
    change = np.array(change)
    return [int(i) for i in np.flatnonzero(change > factor * np.median(change[1:]))]


def build(args, diagnose_only=False):
    poses, clip_k, _ = report_inputs(args.droid_run, args.clip)
    metres = json.loads((args.depth_run / 'metric-scale.json').read_text())['metres_per_native_unit']
    masks_tar = upload_masks(args.run_id, args.masks)
    started = time.time()
    if diagnose_only:
        with app.run():
            result = diagnose_remote.remote(args.run_id, poses, clip_k, overlay=args.overlay_rows, masks_tar=masks_tar)
        result['wall_seconds'] = time.time() - started
        save(args.output / 'diagnose.json', result)
        print(json.dumps(result, indent=1))
        return
    cuts = shot_cuts(args.clip / 'source-full.mp4')
    config = {'exclude': [[int(v) for v in span.split(':')] for span in args.exclude_frames], 'conf': args.conf, 'fill_conf': args.fill_conf, 'sample': args.sample, 'cell': args.cell / metres, 'tolerance': args.tolerance,
              'tolerance_floor': args.tolerance_floor / metres, 'max_points': args.max_points, 'patch_radius': .03 / metres, 'overlay_rows': list(args.overlay_rows),
              'masks_tar': masks_tar}
    with app.run():
        summary = build_remote.remote(args.run_id, poses, clip_k, config)
    wall = time.time() - started
    for name in ('dense/dense-points.glb', 'dense/point-attributes.npz', 'dense/remote.json', 'result/run.json'):
        fetch(args.run_id, name, args.output / ('lingbot-run.json' if name == 'result/run.json' else Path(name).name))
    run = json.loads((args.output / 'lingbot-run.json').read_text())
    plan = json.loads((args.output / 'plan.json').read_text())
    info = {'schema': 'phase2-dense-points-v1', 'coordinate_frame': 'droid_final_native_world', 'units': 'uncalibrated_monocular (native)',
            'metres_per_native_unit': metres, 'file': 'dense-points.glb', 'file_bytes': (args.output / 'dense-points.glb').stat().st_size,
            'file_sha256': digest(args.output / 'dense-points.glb'), 'count': summary['points'],
            'cell_native': summary['cell_native_final'], 'cell_cm': round(summary['cell_native_final'] * metres * 100, 3),
            'fill_cell_cm': round(2 * summary['cell_native_final'] * metres * 100, 3),
            'confidence_threshold': args.conf, 'fill_points': summary['fill_points'],
            'confidence_rule': (f'samples with LingBot depth_conf > {args.conf} are confident: the confidence deciles below it '
                                'were the only ones where under 90% of neighbouring views agreed within 4% (diagnose.json by_conf_decile; '
                                'ME340: the three lowest, conf < 1.061); the official demo shows conf > 1.5. Samples with '
                                f'{args.fill_conf} < conf <= {args.conf} (mostly ceiling and far background) only fill what no confident '
                                'sample covers and are replaced by any confident sample of the same surface'),
            'tolerance_relative': args.tolerance, 'tolerance_floor_cm': args.tolerance_floor * 100,
            'tolerance_rule': '4% of depth held 91% of neighbouring-view depth differences (diagnose.json neighbour_rel_depth_diff)',
            'samples_per_lingbot_pixel_axis': args.sample, 'attributes_file': 'point-attributes.npz (per point: source_frame, fill, spacing, conf, support)',
            'patch_radius_native': config['patch_radius'],
            'alignment': {'method': 'robust Sim3 (least-median triplets, then Umeyama on centres within 2.5x median residual), LingBot camera centres -> report poses_c2w of the same frames',
                          'scale': summary['alignment']['scale'], 'inliers': summary['alignment']['inliers'], 'frames': summary['alignment']['frames'],
                          'residual_native': summary['alignment']['residual_native'],
                          'residual_cm': {k: round(v * metres * 100, 2) for k, v in summary['alignment']['residual_native'].items()},
                          'rotation_disagreement_deg': summary['alignment']['rotation_disagreement_deg'],
                          'lingbot_fov_deg_median_xy': summary['alignment']['lingbot_fov_deg_median_xy'], 'rotation': summary['alignment']['rotation'],
                          'translation': summary['alignment']['translation']},
            'frames_used': {'lingbot_input_frames': len(plan['frames']), 'stride': plan['stride'], 'first_last': [plan['frames'][0]['sourceFrame'], plan['frames'][-1]['sourceFrame']],
                            'fused_into_map': summary['fused_frames'], 'held_out': len(summary['alignment']['outlier_source_frames']),
                            'input': f"uncropped source-full.mp4 1280x720, every {plan['stride']} frame(s) -> official crop mode 518x294",
                            'lingbot': {k: plan[k] for k in ('code_revision', 'weights_revision', 'weights_sha256')} | {'configuration': plan['configuration']},
                            'lingbot_run': {k: run[k] for k in ('gpu', 'inference_seconds', 'elapsed_seconds', 'peak_gpu_bytes', 'native_prediction_sha256')}
                                           | {'file': 'lingbot-run.json', 'predictions': f'Modal volume panoptes-lingbot-map:/{args.run_id}/result'}},
            'fusion': summary['fusion'] | {k: summary[k] for k in ('dropped_contradicted', 'support_views_median', 'support_zero_share', 'fused_frames',
                                                                   'votes_of_fused_frames_on_earlier_points', 'votes_of_held_out_frames_on_map',
                                                                   'valid_sample_share_median', 'naive_points')},
            'shot_cuts_source_frames': cuts,
            'held_out_frames': {'source_frames': summary['alignment']['outlier_source_frames'],
                                'why': 'robust-Sim3 outliers (camera centre disagrees with the report beyond 2.5x the median residual); they only vote against the finished map',
                                'lingbot_fov_deg_median_x': summary['alignment']['lingbot_fov_deg_median_x_outliers']},
            'lingbot_fov_deg_median_x_fused_frames': summary['alignment']['lingbot_fov_deg_median_x_inliers'],
            'build_wall_seconds': wall, 'remote_elapsed_seconds': summary['elapsed_seconds'],
            'note': (f'single-view-per-point display geometry: every point is one view\'s LingBot depth sample ({args.sample}x bilinear over the '
                     '518-px depth), coloured from that full-resolution frame; not averaged or verified across views. A view adds points only '
                     'where no earlier point lies on the surface it sees, so surfaces stay one layer; moving-entity and caption pixels are left out.')}
    save(args.output / 'points.json', info)
    print(json.dumps(summary, indent=1))


def self_check():
    rng = np.random.default_rng(3)
    # Sim3: recovered through noise and 15% gross outliers
    src = rng.normal(size=(300, 3)) * [3, 1, .2]
    angle = .7
    r_true = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]) @ np.array([[1, 0, 0], [0, np.cos(.3), -np.sin(.3)], [0, np.sin(.3), np.cos(.3)]])
    dst = 2.5 * src @ r_true.T + [1, -2, .5] + rng.normal(scale=.002, size=src.shape)
    dst[:45] += rng.normal(scale=2, size=(45, 3))
    (s, r, t), residual, inliers = robust_sim3(src, dst)
    assert abs(s - 2.5) < 1e-3 and np.allclose(r, r_true, atol=1e-3) and np.allclose(t, [1, -2, .5], atol=1e-2), (s, t)
    assert inliers[45:].mean() > .9 and inliers[:45].mean() < .1
    # one row per cell, the best-ranked
    pts = np.array([[.01, .01, .01], [.02, .02, .02], [.3, .3, .3]])
    assert sorted(best_per_cell(cell_keys(pts, .1), np.array([2., 1., 5.]))) == [1, 2]
    # fusion: a wall seen by two views whose depths disagree by 3% stays one layer; a box in front of it is kept
    k = np.array([[200., 0, 100], [0, 200, 75], [0, 0, 1]])
    v, u = np.mgrid[0:150, 0:200]
    wall = np.full((150, 200), 4.)
    box = (u > 80) & (u < 120) & (v > 55) & (v < 95)
    fusion = Fusion(cell=.01, tolerance=.05, tolerance_floor=.01)
    for frame, (shift, bias) in enumerate([(0., 1.), (.05, 1.03)]):
        c2w = np.eye(4)
        c2w[0, 3] = shift
        depth = np.where(box, 3., wall * bias).astype(np.float32)
        ok = ~edges(depth)
        fusion.add_frame(frame, depth, ok, k, c2w, np.zeros((150, 200, 3), np.uint8), np.full((150, 200), 2., np.float32), np.ones((150, 200)))
    points, _, _ = fusion.result(.01, 10 ** 7)
    z = points['xyz'][:, 2]
    on_wall = z > 3.5
    assert (z[on_wall] < 4.02).mean() > .97, 'second view added a layer behind the first'
    assert ((z > 2.9) & (z < 3.2)).sum() > 1000, 'the box in front was suppressed'
    spread = layer_spread(points['xyz'], np.array([[-1.5, -1.2, 4.]]), .3)
    assert spread[0] < .02, spread
    # fill tier: a confident view replaces fill points of the same surface; a later fill view cannot displace it
    fusion = Fusion(cell=.01, tolerance=.05, tolerance_floor=.01, confident=1.5)
    for frame, (bias, confidence) in enumerate([(1., 1.2), (1.01, 3.), (.99, 1.2)]):
        c2w = np.eye(4)
        c2w[0, 3] = .02 * frame
        fusion.add_frame(frame, (wall * bias).astype(np.float32), np.ones((150, 200), bool), k, c2w, np.zeros((150, 200, 3), np.uint8),
                         np.full((150, 200), confidence, np.float32), np.ones((150, 200)))
    points, _, _ = fusion.result(.01, 10 ** 7)
    assert (points['frame'] == 1).mean() > .9 and points['fill'].mean() < .1, (np.bincount(points['frame']), points['fill'].mean())
    # GLB round trip
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        xyz, rgb = rng.normal(size=(50, 3)).astype(np.float32), rng.integers(0, 255, (50, 3), dtype=np.uint8)
        write_points_glb(Path(tmp) / 'p.glb', xyz, rgb)
        back_xyz, back_rgb = read_points_glb(Path(tmp) / 'p.glb')
        assert np.array_equal(back_xyz, xyz) and np.array_equal(back_rgb, rgb)
    print('self-check passed: robust Sim3, per-cell choice, single-layer fusion keeps foreground, fill tier yields to confident samples, layering metric, GLB round trip')


if __name__ == '__main__':
    if '--self-check' in sys.argv:
        self_check()
        sys.exit()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('mode', choices=['diagnose', 'build', 'evaluate'])
    # a lingbot_room.py output directory names its run (submission.json): the report runner passes the producer's directory
    p.add_argument('--run-id', type=lambda v: json.loads((Path(v) / 'submission.json').read_text())['run_id'] if (Path(v) / 'submission.json').is_file() else v)
    for name in ['droid-run', 'clip', 'masks', 'output', 'mesh', 'surface', 'points', 'depth-run']:
        p.add_argument('--' + name, type=Path)
    p.add_argument('--conf', type=float, default=1.06, help='LingBot depth_conf above which a sample is confident (see diagnose)')
    p.add_argument('--fill-conf', type=float, default=1., help='samples above this and at most --conf only fill holes (--fill-conf = --conf: none)')
    p.add_argument('--sample', type=int, default=3, help='bilinear depth samples per LingBot pixel along each axis')
    p.add_argument('--cell', type=float, default=.0075, help='metres; one point per cell (grown until --max-points fit)')
    p.add_argument('--tolerance', type=float, default=.04, help='relative depth band that counts as the same surface')
    p.add_argument('--tolerance-floor', type=float, default=.015, help='metres; minimum same-surface band')
    p.add_argument('--max-points', type=int, default=4_000_000)
    p.add_argument('--exclude-frames', nargs='*', default=[], metavar='START:END', help='source frames never fused, e.g. a cut-away shot (14:226 on ME340)')
    p.add_argument('--overlay-rows', type=lambda s: tuple(int(v) for v in s.split(':')), default=SUBTITLE_ROWS, metavar='Y0:Y1',
                   help='rows of the uncropped frame under burned-in captions, never used (default: ME340\'s 646:706; 0:0 for none)')
    a = p.parse_args()
    if a.mode == 'evaluate':
        evaluate(a)
    else:
        build(a, diagnose_only=a.mode == 'diagnose')
