"""SAM 3D Objects vs RecGen on cell 090, one object at a time (experiment; nothing published is touched).

Both meshes are placed by the same unchanged assembly (completion_ab.py --stage assemble, AB_CELL=090). Per object:
  photos     every photo of the object: mask (white), RecGen (red), SAM 3D (orange) final silhouettes, IoU + depth p50
  side view  world vertices against the unified report floor (boxes-090.v3.json), with the measured box (grey)
  gates      floor: lowest point < -2 cm = sinks; floor-contact box and lowest > 3 cm = floats
             collapse: any gravity extent < 0.5x or > 2x the measured box, or IoU < 0.5 in any photo
             fix shown: floor-contact snap (lift/drop along up so lowest = box bottom); collapse -> measured box replaces model

  /Users/adam/Desktop/panoptes-public/panoptes-serving/.venv/bin/python compare.py left_light_curtain [...]

Any run (defaults unchanged): AB_RUN / AB_OUT as completion_ab.py; CMP_NOTES = where results.json and the sheets go;
CMP_BOXES = a boxes file (box_faces layer format; its 'objectIds' {object_id: entity id} maps run objects to boxes) or 'none'
(no measured boxes yet: the box gates are skipped, nativeToMeters from CMP_SCALE, the floor from the run's floor.json);
CMP_REFERENCE = the model drawn against SAM 3D ('recgen', default) or 'none'; CMP_PLANE_ANCHOR=1: the fixed metrics after the
1-DOF plane anchor (plane_anchor), their depth residual then the z-buffer measure of the moved model.
"""
import json
import os
from pathlib import Path
import sys

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import trimesh  # noqa: E402

HERE = Path(os.environ.get('CMP_NOTES') or Path(__file__).resolve().parent)
CHECKS = Path(os.environ.get('SWAP_SCRATCH', '/private/tmp/claude-501/-Users-adam-Desktop-panoptes-public/1fd9a1db-e580-4bfc-8110-119a1cc38a99/scratchpad')) / 'checks'
OUT = Path(os.environ['AB_OUT']) if os.environ.get('AB_OUT') else CHECKS / 'completionAB-090'
RUN = Path(os.environ['AB_RUN']) if os.environ.get('AB_RUN') else Path(os.environ.get('PANOPTES_RUNS', '/Users/adam/Desktop/panoptes-public/panoptes-serving/outputs/candidate-evaluation')) / 'lucida-replica-01'
# another run (AB_RUN) is another world: never the Pi3X-world boxes or a RecGen arm by default
_BOXES = os.environ.get('CMP_BOXES') or ('none' if os.environ.get('AB_RUN') else str(CHECKS / 'mvp-final/boxes-090.v3.json'))
BOXES = None if _BOXES == 'none' else json.loads(Path(_BOXES).read_text())
REF = os.environ.get('CMP_REFERENCE') or ('none' if os.environ.get('AB_RUN') else 'recgen')
OBJ_ORDER = [o['object_id'] for o in json.loads((RUN / 'evidence/objects.json').read_text())['objects']]
if BOXES:
    N2M = BOXES['nativeToMeters']
    UP, OFF = np.asarray(BOXES['floor']['normal']), BOXES['floor']['offset']
    # objectIds when the boxes file maps run objects to entities; else objects.json order = the first boxes (the published 090 layer)
    BOX = ({oid: BOXES['boxes'][eid] for oid, eid in BOXES['objectIds'].items() if eid in BOXES['boxes']} if BOXES.get('objectIds')
           else dict(zip(OBJ_ORDER, list(BOXES['boxes'].values())[:len(OBJ_ORDER)])))
else:  # no measured boxes: the run's own floor (cameras on the + side) and its scale (CMP_SCALE, else floor.json's e-stop scale)
    _floor = json.loads((RUN / 'evidence/floor.json').read_text())
    N2M = json.loads(Path(os.environ['CMP_SCALE']).read_text())['nativeToMeters'] if os.environ.get('CMP_SCALE') else _floor['estopNativeToMeters']
    _plane = np.asarray(_floor['plane_native'], float)
    _plane /= np.linalg.norm(_plane[:3])
    UP, OFF = _plane[:3], float(_plane[3])
    BOX = {}
if os.environ.get('AB_RUN'):  # one world: the run's own floor and e-stop scale, whatever the boxes file says
    _floor = json.loads((RUN / 'evidence/floor.json').read_text())
    _plane = np.asarray(_floor['plane_native'], float); _plane /= np.linalg.norm(_plane[:3])
    assert abs(abs(float(UP @ _plane[:3])) - 1) < 1e-6 and abs(abs(OFF) - abs(_plane[3])) < 1e-6, 'boxes are on another floor (another world)'
    if 'estopNativeToMeters' in _floor:
        assert abs(N2M / _floor['estopNativeToMeters'] - 1) < 1e-9, 'boxes use another scale (another world)'
FIELD_CM = {'left_light_curtain': ('罩壳下沿', 24.0), 'right_light_curtain': ('罩壳下沿', 24.0),
            'left_fence': ('网栏下横杆', 20.0), 'right_fence': ('网栏下横杆', 20.0)}
COL = {'target': (255, 255, 255), 'recgen': (60, 60, 230), 'sam3d': (40, 170, 255)}  # BGR


def comparison(model, oid):
    doc = json.loads((OUT / 'assembly' / model / 'comparisons.json').read_text())['comparisons']['objects']
    return next((c for c in doc if c['object_id'] == oid), None)


def world_mesh(model, oid, c):
    if model == 'recgen':
        mesh = trimesh.load(RUN / 'generation' / oid / 'object.ply', process=False)
        V, F = np.asarray(mesh.vertices, float), np.asarray(mesh.faces)
    else:
        d = np.load(OUT / model / f'{oid}.npz')
        V, F = d['vertices'].astype(float), d['faces']
    M = np.asarray(c['final_object_to_world'], float)
    return trimesh.Trimesh(V @ M[:3, :3].T + M[:3, 3], F, process=False)


def floor_fix(mesh, oid):
    """Generic, one floor for all: nothing below the floor (cut), and a floor-contact object that floats is lowered onto it."""
    notes, h = [], (mesh.vertices @ UP + OFF) * N2M
    lo = float(np.percentile(h, 0.1))  # a sunk foot is visible even when the report's p0.5 lowest point is above the floor
    if lo < -0.02:
        mesh = trimesh.intersections.slice_mesh_plane(mesh, plane_normal=UP, plane_origin=-OFF * UP)  # open cut; the hidden underside is not shown
        notes.append(f'切掉地面以下 {-100 * lo:.1f} cm')
    elif (BOX.get(oid) or {}).get('floorContact') and lo > 0.03:
        mesh = mesh.copy(); mesh.vertices -= UP * (lo / N2M)
        notes.append(f'下移贴地 {100 * lo:.1f} cm')
    return mesh, notes


ANCHOR = os.environ.get('CMP_PLANE_ANCHOR') == '1'  # off by default (the 090 comparison's rule)


def model_hits(mesh, oid, n=300000):
    """Per photo of the object: its own mask's point-map samples (world, valid) and, at the same pixels, the model's first-hit
    depth along the camera ray (z-buffer of dense surface samples; NaN where the model is not drawn)."""
    pts = trimesh.sample.sample_surface(mesh, n, seed=0)[0]; out = []
    for fid in VIEWS[oid]:
        g = RUN / 'geometry/frames' / fid
        P, c2w, K = np.load(g / 'pts3d.npy').astype(float), np.load(g / 'camera_to_world.npy').astype(float), np.load(g / 'intrinsics.npy').astype(float)
        H, Wd = P.shape[:2]; w2c = np.linalg.inv(c2w)
        cam = pts @ w2c[:3, :3].T + w2c[:3, 3]; cam = cam[cam[:, 2] > 1e-3]
        u = np.round(cam[:, 0] / cam[:, 2] * K[0, 0] + K[0, 2]).astype(int); v = np.round(cam[:, 1] / cam[:, 2] * K[1, 1] + K[1, 2]).astype(int)
        ok = (u >= 0) & (u < Wd) & (v >= 0) & (v < H); zbuf = np.full((H, Wd), np.inf); np.minimum.at(zbuf, (v[ok], u[ok]), cam[ok, 2])
        mask = np.load(RUN / 'evidence/objects' / oid / fid / 'canonical_mask.npy').astype(bool) & np.load(g / 'content_valid_mask.npy').astype(bool)
        X = P[mask]; zx = X @ w2c[2, :3] + w2c[2, 3]; zm = zbuf[mask]
        out.append((fid, X, zx, np.where(np.isfinite(zm), zm, np.nan), c2w[:3, 3]))
    return out


def plane_anchor(mesh, oid):
    """1-DOF plane anchor: when >= 50 % of the object's own mask samples lie on one near-vertical plane (box_faces.dominant_plane:
    5 cm, or 5 % of the extent), the model moves along that plane's normal by the median signed distance, over those plane
    samples, from the model's first hit to the sample (the model's photographed face onto its own plane). Returns (mesh, note,
    {photo: relative depth residual p50 after}, shift cm) or (mesh, [], None, None) when no plane holds most samples."""
    sys.path.insert(0, os.environ.get('PANOPTES_WORKCELL', '/Users/adam/.codex/worktrees/panoptes-workcell-photo-speed') + '/scripts')
    from workcell_checks.box_faces import dominant_plane, PLANE_TOL_M, PLANE_MAJORITY, PLANE_VERTICAL
    hits = model_hits(mesh, oid)
    P = np.concatenate([h[1] for h in hits]) if hits else np.zeros((0, 3))
    if len(P) < 200:
        return mesh, [], None, None
    flat = P - np.outer(P @ UP, UP); extent = float(np.linalg.norm(np.percentile(flat, 95, 0) - np.percentile(flat, 5, 0)))
    pl = dominant_plane(P, min(PLANE_TOL_M / N2M, .05 * extent))
    if pl is None or pl[2] < PLANE_MAJORITY or abs(float(pl[0] @ UP)) >= PLANE_VERTICAL:
        return mesh, [], None, None
    nrm, off = pl[0], pl[1]; tol = min(PLANE_TOL_M / N2M, .05 * extent); d = []
    for fid, X, zx, zm, C in hits:
        on = (np.abs(X @ nrm + off) < tol) & np.isfinite(zm)
        Y = C + (X[on] - C) * (zm[on] / zx[on])[:, None]  # the model's first hit on the same ray
        d.append((X[on] - Y) @ nrm)
    d = np.concatenate(d) if d else np.zeros(0)
    if len(d) < 100:
        return mesh, [], None, None
    shift = float(np.median(d)); moved = mesh.copy(); moved.vertices = moved.vertices + nrm * shift
    after = {}
    for fid, X, zx, zm, C in model_hits(moved, oid):
        ok = np.isfinite(zm); after[fid] = round(float(np.median(np.abs(zm[ok] / zx[ok] - 1))), 4) if ok.sum() >= 30 else None
    return moved, [f'沿自身平面法向移 {100 * shift * N2M:+.1f} cm 贴到照片平面（{100 * pl[2]:.0f}% 的点在该平面上）'], after, round(100 * shift * N2M, 1)


VIEWS = {o['object_id']: [v['frame_id'] for v in (o['views'].values() if isinstance(o['views'], dict) else o['views'])]
         for o in json.loads((RUN / 'evidence/objects.json').read_text())['objects']}


def free_space(mesh, oid, n=300000):
    """Per photo: share of the model's pixels where the model sits in front of a surface the photo sees (by > max(5 %, 8 cm))
    outside the object's mask, i.e. the model fills space the photo shows to be empty. Occluders in front do not count."""
    pts = trimesh.sample.sample_surface(mesh, n, seed=0)[0]
    out = {}
    for fid in VIEWS[oid]:
        g = RUN / 'geometry/frames' / fid
        P, c2w, K = np.load(g / 'pts3d.npy'), np.load(g / 'camera_to_world.npy').astype(float), np.load(g / 'intrinsics.npy').astype(float)
        H, Wd = P.shape[:2]
        w2c = np.linalg.inv(c2w)
        zobs = P @ w2c[2, :3] + w2c[2, 3]
        seen = np.load(g / 'valid_mask.npy').astype(bool) & np.isfinite(zobs) & (zobs > 0)
        cam = pts @ w2c[:3, :3].T + w2c[:3, 3]
        cam = cam[cam[:, 2] > 1e-3]
        u = np.round(cam[:, 0] / cam[:, 2] * K[0, 0] + K[0, 2]).astype(int)
        v = np.round(cam[:, 1] / cam[:, 2] * K[1, 1] + K[1, 2]).astype(int)
        ok = (u >= 0) & (u < Wd) & (v >= 0) & (v < H)
        zbuf = np.full((H, Wd), np.inf)
        np.minimum.at(zbuf, (v[ok], u[ok]), cam[ok, 2])
        mask = cv2.dilate(np.load(RUN / 'evidence/objects' / oid / fid / 'canonical_mask.npy').astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
        # 3 px margin (518 grid): a thin object's slight misplacement along its outline is not empty space filled
        model = np.isfinite(zbuf)
        tol = np.maximum(.05 * zobs, .08 / N2M)
        bad = model & ~mask & seen & (zbuf < zobs - tol)
        out[fid] = round(float(bad.sum() / max(model.sum(), 1)), 3)
    return out


def extents(W, box):
    """Gravity-aligned extents (m) along the measured box's axes (L, W horizontal; H up) and lowest/top above floor (m).
    Without a box: the model's own horizontal principal axes."""
    h = (W @ UP + OFF) * N2M
    if box is None:
        flat = W - np.outer(W @ UP, UP); flat -= flat.mean(0)
        e = np.linalg.eigh(flat.T @ flat)[1][:, ::-1]
        box = {'axes': [e[:, 0].tolist(), e[:, 1].tolist(), UP.tolist()]}
    ax = np.asarray(box['axes'], float)
    horiz = [(W @ ax[i]) * N2M for i in (0, 1)]
    lo, hi = np.percentile(h, [0.5, 99.5])  # the report's lowest point is p0.5 (workcell_checks.floor.assess)
    body = [float(np.ptp(np.percentile(p, [5, 95]))) for p in horiz]  # without thin flanges / feet, for the too-large check
    return [float(np.ptp(np.percentile(p, [0.5, 99.5]))) for p in horiz] + [float(hi - lo)], float(lo), float(hi), h, horiz, body


def gates(oid, m):
    box = BOX.get(oid)
    out = []
    if m['minCm'] < -2:
        out.append(f"下沉 {-m['minCm']:.1f} cm")
    if box is None:  # no measured box (yet): only the gates that need none
        return out + photo_gates(m)
    if box.get('floorContact') and m['lowestCm'] > 3:
        out.append(f"悬空 {m['lowestCm']:.1f} cm")
    elif (box['dims']['bottom']['confidence'] == 'high' and abs(m['lowestCm'] - 100 * box['bottomM']) > 5) or (
            box['dims']['bottom']['confidence'] in ('medium', 'low') and m['lowestCm'] - 100 * box['bottomM'] > 15):
        # a low-confidence bottom still catches a missing lower part; reaching further down (to the floor) is not an error there  # measured box bottom: the model's lower part is missing or too long
        out.append(f"底部与测量盒差 {m['lowestCm'] - 100 * box['bottomM']:+.1f} cm（{'下部缺失' if m['lowestCm'] > 100 * box['bottomM'] else '下部过长'}）")
    hb, hc = box['sizeM'][2], box['dims']['H']['confidence']
    if hc not in ('high', 'unverified') and hb > .1 and not .75 <= m['extentM'][2] / hb <= 1.33:
        out.append(f"高度 {m['extentM'][2] * 100:.0f} vs 盒子 {hb * 100:.0f} cm（{'缺一段' if m['extentM'][2] < hb else '过长'}）")
    for name, v, b in zip('LH', [m['extentM'][0], m['extentM'][2]], [box['sizeM'][0], box['sizeM'][2]]):
        # G7 of workcell_checks/obvious_errors.py, the one definition: medium/high box dim, > 30 % and > 2 cm (W is a warning there)
        if box['dims'][name]['confidence'] in ('medium', 'high') and abs(v - b) > .02 and abs(v / b - 1) > .30:
            out.append(f"{dict(L='长', H='高')[name]} {v * 100:.0f} vs 盒子 {b * 100:.0f} cm（{'塌陷' if v < b else '过大'}）")
    if EXPLODE_GATE and 'fullM' in m:  # G6 of obvious_errors.py (exploded): full vertex extent > 3 x the box size (5 cm minimum), any axis
        for name, v, b in zip('LWH', m['fullM'], box['sizeM']):
            if v > 3 * max(b, .05):
                out.append(f"炸开：{dict(L='长', W='宽', H='高')[name]}向 {v * 100:.0f} cm，超过测量盒 {b * 100:.0f} cm 的 3 倍")
    return out + photo_gates(m)


EXPLODE_GATE = os.environ.get('CMP_EXPLODE_GATE') == '1'  # off by default (the 090 comparison's rule)
EXTENT_GATE_CM = float(os.environ.get('CMP_EXTENT_GATE_CM', '0'))  # 0 = off (the 090 comparison's rule, unchanged default)


def observed_heights(oid):
    """2nd / 98th percentile height (cm) of the geometry's own points inside the object's masks (evidence points), or None."""
    obj = next(o for o in json.loads((RUN / 'evidence/objects.json').read_text())['objects'] if o['object_id'] == oid)
    P = [np.load(RUN / v['points_path']) for v in obj['views'] if v.get('partial_point_count', 1)]
    P = np.concatenate(P) if P else np.zeros((0, 3))
    return [round(float(x), 1) for x in np.percentile((P @ UP + OFF) * N2M * 100, [2, 98])] if len(P) >= 50 else None


def photo_gates(m):
    out = []
    obs = m.get('observedCm')
    if EXTENT_GATE_CM and obs:  # box-free: a capped box is the model's own box and cannot tell a missing part
        if m['lowestCm'] - obs[0] > EXTENT_GATE_CM:
            out.append(f"下部缺失（模型底比照片里看到的物体最低处高 {m['lowestCm'] - obs[0]:.0f} cm）")
        if obs[1] - m['topCm'] > EXTENT_GATE_CM:
            out.append(f"上部缺失（模型顶比照片里看到的物体最高处低 {obs[1] - m['topCm']:.0f} cm）")
    fs = [f'照片{fid[-1]} {100 * f:.0f}%' for fid, f in m['freeSpace'].items() if f > .15]
    if fs:
        out.append('模型占了照片里看得见的空地 ' + ', '.join(fs))
    if m['meanDepthP50'] > .06:
        out.append(f"深度与照片不符（p50 {100 * m['meanDepthP50']:.1f}%）")
    bad = [f'照片{fid[-1]} IoU {p["iou"]:.2f}' for fid, p in m['perPhoto'].items() if p['iou'] is not None and p['iou'] < .5]
    if bad:
        out.append('轮廓不符 ' + ', '.join(bad))
    return out


def metrics(model, oid, fix=False):
    c = comparison(model, oid)
    if c is None:
        return None
    mesh, notes = floor_fix(world_mesh(model, oid, c), oid) if fix else (world_mesh(model, oid, c), [])
    anchored = None
    if fix and ANCHOR:
        mesh, more, anchored, shift = plane_anchor(mesh, oid); notes = notes + more
    ext, lo, hi, h, horiz, body = extents(np.asarray(mesh.vertices), BOX.get(oid))
    if BOX.get(oid):  # full vertex extent along the box axes (G6's measure)
        Wv = np.asarray(mesh.vertices); ax = np.asarray(BOX[oid]['axes'], float)
        full = [float(np.ptp(Wv @ ax[i]) * N2M) for i in range(3)]
    m = {'perPhoto': {v['frame_id']: {'iou': round(v['generated_refined']['visible_iou'], 3),
                                       'depthP50': round((anchored or {}).get(v['frame_id']) or v['generated_refined']['relative_depth_p50'], 4)} for v in c['views']},
         'minCm': round(100 * float(np.percentile(h, 0.1)), 1), 'extentM': [round(x, 3) for x in ext], 'bodyM': [round(x, 3) for x in body], 'lowestCm': round(100 * lo, 1), 'topCm': round(100 * hi, 1)}
    if anchored:  # the depth residual after the plane anchor (model first hit vs the mask's samples, the z-buffer measure)
        m['anchorCm'] = shift; m['depthSource'] = 'after the plane anchor (z-buffer of the moved model vs the mask samples)'
    m['meanIou'] = round(float(np.mean([p['iou'] for p in m['perPhoto'].values()])), 3)
    m['meanDepthP50'] = round(float(np.mean([p['depthP50'] for p in m['perPhoto'].values()])), 4)
    m['freeSpace'] = free_space(mesh, oid)
    if EXTENT_GATE_CM:
        m['observedCm'] = observed_heights(oid)
    if BOX.get(oid):
        m['fullM'] = [round(x, 3) for x in full]
    m['gates'], m['fix'] = gates(oid, m), notes
    if model != 'recgen':
        m['coverage'] = coverage(model, oid)
    box = BOX.get(oid)
    if box and (box.get('floorContact') or box['bottomM'] > 0):  # one floor for all: snap lowest to the measured bottom
        m['snapCm'] = round(100 * box['bottomM'] - m['lowestCm'], 1)
    if oid in FIELD_CM:
        m['fieldErrorCm'] = round(m['lowestCm'] - FIELD_CM[oid][1], 1)
    m['_plot'] = (h, horiz)
    return m


def photo_tiles(oid, variant):
    sil = {'sam3d': np.load(OUT / 'assembly' / variant / 'silhouettes.npz')}
    if REF != 'none':
        sil[REF] = np.load(OUT / 'assembly' / REF / 'silhouettes.npz')
    base = sil['sam3d']  # every assembly writes the same targets
    tiles = []
    for fid in sorted({k.split('__')[1] for k in base.files if k.startswith(oid + '__')}):
        key = f'{oid}__{fid}__'
        hw = tuple(json.loads(bytes(base[key + 'shape'])))
        unpack = lambda a: np.unpackbits(a)[:hw[0] * hw[1]].reshape(hw).astype(np.uint8)
        target, big = unpack(base[key + 'target']), 4
        img = cv2.imread(str(RUN / 'geometry/frames' / fid / 'canonical.png'))  # full resolution; grid masks scaled up
        big = img.shape[0] / hw[0]
        up = lambda m: (cv2.resize(m.astype(np.float32), img.shape[1::-1], interpolation=cv2.INTER_LINEAR) > .5).astype(np.uint8)
        for name, mask in [('target', target)] + [(k, unpack(sil[k][key + 'final'])) for k in (REF, 'sam3d') if k in sil and key + 'final' in sil[k]]:
            cs, _ = cv2.findContours(up(mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(img, cs, -1, COL[name], max(2, int(img.shape[0] / 500)))
        ys, xs = np.nonzero(target)
        cx, cy = (xs.min() + xs.max()) / 2 * big, (ys.min() + ys.max()) / 2 * big
        half = max(np.ptp(xs), np.ptp(ys)) * big * .58 + 20
        crop = img[int(max(cy - half, 0)):int(cy + half), int(max(cx - half, 0)):int(cx + half)]
        tiles.append((fid, cv2.cvtColor(cv2.resize(crop, (420, 420), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)))
    return tiles


def decide(sam, fixed):
    if not sam['gates']:
        return 'SAM 3D 原样'
    if not fixed['gates']:
        return 'SAM 3D + 地面修正（' + '，'.join(fixed['fix']) + '）'
    return '用测量盒替代（' + '；'.join(fixed['gates']) + '）'


def gen_photo(variant, oid):
    if variant.startswith('sam3d-'):
        return variant[-1]
    return json.loads((OUT / 'sam3d/record.json').read_text())['objects'][oid]['frame_id'][-1]


COVERAGE_WEIGHT = float(os.environ.get('CMP_COVERAGE_WEIGHT', '0'))  # 0 = the 090 comparison's rule (unchanged default)


def coverage(variant, oid):
    """Share of the generation photo's mask with a point-map depth (SAM 3D's input), from the generation record; None if unknown."""
    meta = json.loads((OUT / variant / 'record.json').read_text())['objects'].get(oid) or {}
    # the MVS share (before any MoGe fill of the input: workcell_rebuild_report.sam3d_input records both)
    return round(meta.get('mask_mvs_pixels', meta['mask_depth_pixels']) / meta['mask_pixels'], 3) if meta.get('mask_pixels') else None


def score(m):
    """Fewest obvious errors first, then silhouette minus depth residual (every photo of the object), plus CMP_COVERAGE_WEIGHT x the
    generation photo's point-map coverage (a candidate generated from a photo whose mask the geometry barely covers is at risk)."""
    return (len(m['gates']), -(m['meanIou'] - 2 * m['meanDepthP50'] + COVERAGE_WEIGHT * (m.get('coverage') or 0)))


def long_axis(oid):
    box = BOX.get(oid)
    return (1 if box['sizeM'][1] >= box['sizeM'][0] else 0) if box else 0  # without a box: the model's own principal axis


def object_label(oid):
    box = BOX.get(oid)
    return box['label'] if box else next(o['label'] for o in json.loads((RUN / 'evidence/objects.json').read_text())['objects'] if o['object_id'] == oid)


def sheet(oid, rec, sam, fixed, variant, tried, cands=None):
    """rec: the reference model's metrics (RecGen), or None (CMP_REFERENCE=none: SAM 3D only)."""
    tiles = photo_tiles(oid, variant)
    fig = plt.figure(figsize=(4.4 * len(tiles) + 6.4, 9.6))
    gs = fig.add_gridspec(3, len(tiles) + 2, height_ratios=[5, 2.4, 1.8], width_ratios=[4.4] * len(tiles) + [3.2, 3.2])
    # candidates strip: SAM 3D from each photo, side view after the floor fix; orange = chosen, grey = not chosen
    strip = gs[1, :].subgridspec(1, max(len(cands or {}), 1) + 1, wspace=.35)
    ax0 = fig.add_subplot(strip[0, 0]); ax0.axis('off')
    ax0.text(0, .5, '补全候选\n（每张照片\n各生成一次）', fontsize=10, va='center')
    for j, (v, (_, f)) in enumerate((cands or {}).items()):
        ax = fig.add_subplot(strip[0, j + 1])
        h, horiz = f['_plot']
        la = long_axis(oid)
        k = np.random.default_rng(0).choice(len(h), min(len(h), 8000), replace=False)
        ax.scatter(100 * (horiz[la][k] - np.median(horiz[la])), 100 * h[k], s=1, c='#ff9f1c' if v == variant else '#999999',
                   alpha=.3, linewidths=0)
        ax.axhline(0, color='#2a7', lw=1)
        ax.set_aspect('equal', adjustable='datalim'); ax.tick_params(labelsize=6)
        err = '；'.join(f['gates']) or '无明显错误'
        mark = ('（最好，但仍有明显错误）' if fixed['gates'] else ' ✓ 采用') if v == variant else ''
        cov = f" 点图 {100 * f['coverage']:.0f}%" if f.get('coverage') is not None else ''
        ax.set_title(f"照片 {gen_photo(v, oid)} 生成{mark}\nIoU {f['meanIou']:.2f} 深度 {100 * f['meanDepthP50']:.1f}%{cov}\n{err[:28]}",
                     fontsize=8, color='#c06000' if v == variant else '#444')
    for i, (fid, t) in enumerate(tiles):
        ax = fig.add_subplot(gs[0, i]); ax.imshow(t); ax.axis('off')
        s = sam['perPhoto'].get(fid, {})
        ref = f'RecGen {(rec or {}).get("perPhoto", {}).get(fid, {}).get("iou", "-")}  /  ' if rec else ''
        ax.set_title(f'照片 {fid[-1]}   IoU  {ref}SAM 3D {s.get("iou", "-")}', fontsize=10)
    box = BOX.get(oid)
    la = long_axis(oid)
    centre = np.median(sam['_plot'][1][la])
    panels = ([('RecGen', [(rec, '#e63c3c')])] if rec else []) + [('SAM 3D（灰 = 修正前）', [(sam, '#bbbbbb'), (fixed, '#ff9f1c')])]
    for j, (name, layers) in enumerate(panels):
        ax = fig.add_subplot(gs[0, len(tiles) + j])
        for m, colour in layers:
            h, horiz = m['_plot']
            k = np.random.default_rng(0).choice(len(h), min(len(h), 20000), replace=False)
            ax.scatter(100 * (horiz[la][k] - centre), 100 * h[k], s=2, c=colour, alpha=.25, linewidths=0)
        h = np.concatenate([m['_plot'][0] for m, _ in layers])
        bw = max(box['sizeM'][:2]) if box else max(fixed['extentM'][:2])
        if box:
            ax.add_patch(plt.Rectangle((-50 * bw, 100 * box['bottomM']), 100 * bw, 100 * box['sizeM'][2], fill=False, ec='#666', lw=1.2))
        ax.axhline(0, color='#2a7', lw=1.5)
        if oid in FIELD_CM:
            ax.axhline(FIELD_CM[oid][1], color='#2a7', lw=.8, ls='--')
        ax.set_title(f'{name} 侧视（cm，绿 = 地面{"，灰框 = 测量盒" if box else ""}）', fontsize=10)
        ax.set_xlim(-max(45, 75 * bw + 10), max(45, 75 * bw + 10)); ax.tick_params(labelsize=7); ax.grid(alpha=.25)
        ax.set_ylim(min(-10, 100 * h.min() - 5), max(100 * h.max(), 100 * (box['bottomM'] + box['sizeM'][2]) if box else 0) + 8)
    if not rec:
        ax = fig.add_subplot(gs[0, len(tiles) + 1]); ax.axis('off')
    ax = fig.add_subplot(gs[2, :]); ax.axis('off')
    def line(name, m):
        e = m['extentM']
        g = '；'.join(m['gates']) or '无明显错误'
        f = f"  （现场{FIELD_CM[oid][0]} {FIELD_CM[oid][1]:.0f} cm，最低点差 {m['fieldErrorCm']:+.1f}）" if oid in FIELD_CM else ''
        return (f"{name}: IoU 平均 {m['meanIou']:.2f}  深度 p50 {100 * m['meanDepthP50']:.1f}%  最低点 {m['lowestCm']:+.1f} cm  顶 {m['topCm']:.0f} cm  "
                f"尺寸 {e[0] * 100:.0f}×{e[1] * 100:.0f}×{e[2] * 100:.0f} cm{f}  |  检查: {g}")
    txt = (f"{object_label(oid)}（{oid}）  " + (f"测量盒 {box['sizeM'][0] * 100:.0f}×{box['sizeM'][1] * 100:.0f}×{box['sizeM'][2] * 100:.0f} cm，"
           f"离地 {box['bottomM'] * 100:.1f} cm" if box else '还没有测量盒') + '\n' + (line('RecGen', rec) + '\n' if rec else '')
           + line(f'SAM 3D（照片 {gen_photo(variant, oid)} 生成）', sam) + '\n'
           + line('SAM 3D 地面修正后', fixed) + f"\n→ 采用：{decide(sam, fixed)}"
           + (f"\n   SAM 3D 各输入照片：" + '；'.join(tried) if len(tried) > 1 else ''))
    ax.text(0, .95, txt, va='top', fontsize=10.5, family=['Arial Unicode MS', 'Noto Sans CJK SC', 'sans-serif'])
    fig.text(.01, .985, ('白 = 掩码   红 = RecGen   橙 = SAM 3D（同一组装，同一地面）。' if rec else '白 = 掩码   橙 = SAM 3D（组装后，统一地面）。') + '最低点是模型最低处（光幕含下部护板、围栏含立柱脚），'
             '不是罩壳下沿/下横杆；那两个值由照片近面测量给出，不随补全模型变', fontsize=9.5, va='top')
    plt.rcParams['font.family'] = ['Arial Unicode MS', 'Noto Sans CJK SC', 'sans-serif']
    fig.tight_layout(rect=(0, 0, 1, .97))
    path = HERE / f'{oid}.jpg'
    fig.savefig(path, dpi=100, pil_kwargs={'quality': 85})
    plt.close(fig)
    return path


if __name__ == '__main__':
    plt.rcParams['font.family'] = ['Arial Unicode MS', 'Noto Sans CJK SC', 'sans-serif']
    results = json.loads((HERE / 'results.json').read_text()) if (HERE / 'results.json').exists() else {}
    for oid in sys.argv[1:]:
        rec = metrics(REF, oid) if REF != 'none' else None
        cands = {v: (metrics(v, oid), metrics(v, oid, fix=True)) for v in ['sam3d'] + [f'sam3d-frame_000{i}' for i in (1, 2, 3)]
                 if (OUT / 'assembly' / v / 'comparisons.json').exists() and comparison(v, oid) is not None}
        if (rec is None and REF != 'none') or not cands:
            print(oid, 'not assembled yet'); continue
        variant = min(cands, key=lambda v: score(cands[v][1]))
        sam, fixed = cands[variant]
        tried = [f"照片 {gen_photo(v, oid)}：IoU {f['meanIou']:.2f} 深度 {100 * f['meanDepthP50']:.1f}%" + (f" 点图覆盖 {100 * f['coverage']:.0f}%" if f.get('coverage') is not None else '')
                 + f" 错误 {len(f['gates'])}" for v, (_, f) in cands.items()]
        path = sheet(oid, rec, sam, fixed, variant, tried, cands)
        results[oid] = {k: {kk: vv for kk, vv in m.items() if kk != '_plot'} for k, m in (('recgen', rec), ('sam3d', sam), ('sam3dFixed', fixed)) if m}
        results[oid].update(decision=decide(sam, fixed), variant=variant, generationPhoto=gen_photo(variant, oid),
                            tried={v: {'meanIou': f['meanIou'], 'meanDepthP50': f['meanDepthP50'], 'coverage': f.get('coverage'), 'gates': f['gates']} for v, (_, f) in cands.items()},
                            coverageWeight=COVERAGE_WEIGHT)
        print(oid, path, json.dumps(results[oid], ensure_ascii=False))
    (HERE / 'results.json').write_text(json.dumps(results, indent=1, ensure_ascii=False) + '\n')
