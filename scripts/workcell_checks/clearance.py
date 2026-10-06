"""Bottom clearance, method A: how high an object's lowest visible edge is above the floor, from its own photo masks.

Per photo with a mask, per image column the lowest mask pixel. A column is dropped when it lies in the mask's left/right 10 %
(feet, posts, ends), when the frame cuts the mask, when a pixel just below (1-3 px) lies in another object's mask, or when the
ray through the pixel below meets another displayed model in front of this object (ray cast). The bottom edge of each kept
boundary pixel is back-projected and met with the object's own displayed model (first hit); a ray that misses is met with
the plane fitted (RANSAC, 1 cm) to the model hits 5-60 px above the boundary in that photo. Height = n.X + d (report floor)
x nativeToMeters. Per photo median / IQR over columns; combined = median over photos with >= 10 columns. The model's plain
bottom (0.5 percentile of its vertices) is given for comparison.

Variants, fixed before any measurement and all reported: `overlap` ignores another mask that also covers the boundary pixel
itself (two masks claiming the same pixels say nothing about occlusion); `plane` meets every kept ray with the fitted plane.
Added after the first run (post-hoc, from failures seen in the photos, not from the field values): `verticals` takes the
lowest pixel along each projected 3D vertical (line through the nadir vanishing point) instead of each image column, since a
tilted camera makes columns cut a post's slanted side edges; `verticalsGap` also counts another mask up to 20 px below as
'just below' (neighbouring report masks leave 7-10 px gaps, so a post hidden behind a bollard was not caught at 1-3 px).
Diagnostics: an independent floor near the object (RANSAC, 1 cm, on the report's Pi3X point maps outside every mask, within
0.75 m of the object's boundary footprint and +-15 cm of the report floor; 1.5 m if too few points) and the heights above it;
the photo's strongest vertical intensity step within +-10 px of the mask boundary; Pi3X range just above the boundary
against the range used; cm of height per px of edge and per cm of range along the ray.
Generic: no per-object rules; the scale is read, never changed. Pure numpy/cv2/open3d/scipy: runs in the Modal check
container or on-prem (python clearance.py --view ... ; no arguments = self-test)."""
import io
import json
import math
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np

try:
    import shape_core as wsc  # the check container (modal_apps/workcell_view_checks.py)
except ImportError:  # locally / on-prem: scripts/workcell_shape_check.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import workcell_shape_check as wsc

EDGE, BELOW, BAND, BAND_STEP, MAX_PLANE_COLS = .10, (1, 2, 3), (5, 60), 5, 300
PLANE_TOL_M, MIN_PLANE, MIN_COLS = .01, 20, 10
FLOOR_R_M, FLOOR_R2_M, FLOOR_BAND_M, FLOOR_TOL_M, FLOOR_MAX_DEG, FLOOR_MIN, FLOOR_DILATE_PX = .75, 1.5, .15, .01, 20, 100, 8
STEP_WIN, STEP_CONTRAST, PI3X_ABOVE_PX = 10, 3.0, 15
GAP_PX = 20  # post-hoc variant: report masks of neighbours leave gaps of ~7-10 px, so 'just below' means within 20 px


def rays(cam, u, v):
    d = np.c_[u, v, np.ones(len(u))] @ np.linalg.inv(cam['K']).T @ cam['R']  # rows of R^T K^-1 x
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def scene_of(meshes):
    import open3d as o3d
    scene = o3d.t.geometry.RaycastingScene()
    for V, F in meshes:  # geometry id = index
        scene.add_triangles(o3d.core.Tensor(V.astype(np.float32)), o3d.core.Tensor(F.astype(np.uint32)))
    return scene


def hits(scene, C, D, own):
    """Nearest hit of the own model and of any other model along each unit ray (native units, inf = none)."""
    import open3d as o3d
    own_t, oth_t = np.full(len(D), np.inf), np.full(len(D), np.inf)
    if len(D):
        r = scene.list_intersections(o3d.core.Tensor(np.hstack([np.broadcast_to(C, D.shape), D]).astype(np.float32)))
        ids, g, t = (r[k].numpy().astype(np.float64) for k in ('ray_ids', 'geometry_ids', 't_hit'))
        ids = ids.astype(np.int64); mine = g == own
        np.minimum.at(own_t, ids[mine], t[mine]); np.minimum.at(oth_t, ids[~mine], t[~mine])
    return own_t, oth_t


def ransac_plane(P, tol, rng, iters=1000, up=None, max_deg=90):
    """(unit n, d, inliers) of the plane with most points within tol, refit by SVD on its inliers; n.up >= 0 if up given."""
    if len(P) < 3:
        return None
    tri = P[rng.integers(0, len(P), (iters, 3))]
    N = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]); k = np.linalg.norm(N, axis=1); ok = k > 1e-12
    N, tri = N[ok] / k[ok, None], tri[ok]
    if up is not None:
        ok = np.abs(N @ up) >= math.cos(math.radians(max_deg)); N, tri = N[ok], tri[ok]
    best, count = None, 2
    for n, a in zip(N, tri[:, 0]):  # ponytail: plain loop, ~1000 hypotheses x <= 1e5 points
        c = int((np.abs((P - a) @ n) < tol).sum())
        if c > count:
            best, count = (n, a), c
    if best is None:
        return None
    n, a = best; Q = P[np.abs((P - a) @ n) < tol]; c = Q.mean(0); n = np.linalg.svd(Q - c, full_matrices=False)[2][2]
    if up is not None and n @ up < 0:
        n = -n
    return n, float(-n @ c), np.abs((P - c) @ n) < tol


def stats(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if not len(x):
        return dict(n=0)
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    return dict(n=int(len(x)), medianCm=round(float(med), 2), iqrCm=[round(float(q1), 2), round(float(q3), 2)])


def own_camera(P, cams):
    """(camera index, grid->pixel affine 2x3) of the camera a Pi3X point map was predicted for: its grid maps affinely onto
    that camera's pixels (median residual < 5 px)."""
    H, W = P.shape[:2]; jj, ii = np.meshgrid(np.arange(W), np.arange(H)); Q = P.reshape(-1, 3)
    good = np.isfinite(Q).all(1) & (np.abs(Q).sum(1) > 0); best = (np.inf, None, None)
    for k, c in enumerate(cams):
        uv, z = wsc.project(c, np.where(good[:, None], Q, 0))
        ok = good & (z > 0) & (uv[:, 0] >= 0) & (uv[:, 1] >= 0) & (uv[:, 0] < c['w']) & (uv[:, 1] < c['h'])
        if ok.sum() < 100:
            continue
        A = np.c_[jj.ravel()[ok], ii.ravel()[ok], np.ones(ok.sum())]; M = np.linalg.lstsq(A, uv[ok], rcond=None)[0]
        r = float(np.median(np.linalg.norm(uv[ok] - A @ M, axis=1)))
        if r < best[0]:
            best = (r, k, M.T)
    return (best[1], best[2]) if best[0] < 5 else (None, None)


def pi3x_frames(ctx, api):
    """The report's Pi3X point maps (native world) with the camera each was predicted for."""
    get = lambda url: urllib.request.urlopen(url, timeout=120).read()
    out, seen = [], set()
    for a in ctx['doc'].get('assets', []):
        md = a.get('metadata') or {}
        if not (md.get('logicalName') == 'pts3d.npy' or str(md.get('sourcePath', '')).endswith('/pts3d.npy')) or a.get('sha256') in seen:
            continue
        seen.add(a.get('sha256'))
        url = json.loads(get(f"{api}/api/assets/{a['id']}"))['url']
        P = np.load(io.BytesIO(get(url if url.startswith('http') else api + url))).astype(np.float64)
        k, M = own_camera(P, ctx['cams'])
        if k is not None:
            out.append(dict(assetId=a['id'], camera=k, P=P, affine=M))
    return out


def floor_candidates(ctx, frames):
    """Pi3X points outside every object mask (dilated) of their own photo, within +-FLOOR_BAND of the report floor."""
    n, d = ctx['floor']; S = ctx['S']; out = []
    for f in frames:
        k, cam = f['camera'], ctx['cams'][f['camera']]; Q = f['P'].reshape(-1, 3)
        Q = Q[np.isfinite(Q).all(1) & (np.abs(Q).sum(1) > 0)]
        Q = Q[np.abs(Q @ n + d) * S < FLOOR_BAND_M]
        uv, z = wsc.project(cam, Q)
        ok = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 1] >= 0) & (uv[:, 0] < cam['w']) & (uv[:, 1] < cam['h']); Q, uv = Q[ok], uv[ok]
        masked = np.zeros(len(Q), bool)
        for du, dv in ((0, 0), (FLOOR_DILATE_PX, 0), (-FLOOR_DILATE_PX, 0), (0, FLOOR_DILATE_PX), (0, -FLOOR_DILATE_PX)):
            ui = np.clip(np.round(uv[:, 0] + du).astype(int), 0, cam['w'] - 1); vi = np.clip(np.round(uv[:, 1] + dv).astype(int), 0, cam['h'] - 1)
            for o in ctx['objects']:
                if k in o['masks']:
                    masked |= o['masks'][k][vi, ui]
        out.append((k, Q[~masked]))
    return out


def local_floor(ctx, cands, footprint, rng):
    """Independent floor near an object: RANSAC plane on the candidates within FLOOR_R of its boundary footprint."""
    from scipy.spatial import cKDTree
    n, d = ctx['floor']; S = ctx['S']
    flat = lambda X: X - (X @ n + d)[:, None] * n
    tree = cKDTree(flat(footprint))
    for radius in (FLOOR_R_M, FLOOR_R2_M):
        parts = [(k, Q[tree.query(flat(Q), distance_upper_bound=radius / S)[0] < np.inf]) for k, Q in cands]
        P = np.vstack([Q for _, Q in parts]) if parts else np.zeros((0, 3))
        if len(P) >= FLOOR_MIN:
            break
    else:
        return None, dict(status='too_few_floor_points', points=int(len(P)))
    fit = ransac_plane(P, FLOOR_TOL_M / S, rng, 1500, n, FLOOR_MAX_DEG)
    if fit is None or fit[2].sum() < FLOOR_MIN:
        return None, dict(status='no_plane', points=int(len(P)))
    nl, dl, inl = fit; frames, start = {}, 0
    for k, Q in parts:
        sel = inl[start:start + len(Q)]; start += len(Q)
        if sel.sum():
            frames[str(k + 1)] = dict(inliers=int(sel.sum()), reportHeightCm=round(float(np.median(Q[sel] @ n + d) * S * 100), 2))
    off = (footprint @ n + d - (footprint @ nl + dl)) * S * 100
    return (nl, dl), dict(status='ok', radiusM=radius, points=int(len(P)), inliers=int(inl.sum()),
                          tiltDeg=round(math.degrees(math.acos(min(1, abs(float(nl @ n))))), 2),
                          inlierReportHeightCm=round(float(np.median(P[inl] @ n + d) * S * 100), 2),
                          floorOffsetCm=round(float(np.median(off)), 2), floorOffsetRangeCm=[round(float(off.min()), 2), round(float(off.max()), 2)],
                          byPhoto=frames)


def boundary(mask):
    """Pre-registered: per image column the lowest mask pixel. Returns pixel (u, v), image 'down' (du, dv), line coordinate."""
    cols = np.nonzero(mask.any(0))[0]; low = mask.shape[0] - 1 - np.argmax(mask[::-1, cols], axis=0); one = np.ones(len(cols))
    return cols, low, 0 * one, one, cols.astype(float)


def verticals(mask, cam, n):
    """Post-hoc variant: per projected 3D vertical (image line through the nadir vanishing point, ~1 px apart at the mask)
    the mask pixel lowest along it. A tilted camera makes image columns cut a vertical post's side edges; these lines do not."""
    vp = cam['K'] @ (cam['R'] @ -n)
    ys, xs = np.nonzero(mask); xs, ys = xs.astype(float), ys.astype(float)
    if abs(vp[2]) < 1e-6 * np.linalg.norm(vp):  # down parallel to the image plane: parallel lines
        p, s = np.array([cam['w'] / 2, cam['h'] / 2]) + 1e8 * vp[:2] / np.linalg.norm(vp[:2]), 1.0
    else:
        p, s = vp[:2] / vp[2], float(np.sign(vp[2]))
    dx, dy = s * (p[0] - xs), s * (p[1] - ys); r = np.hypot(dx, dy)  # image 'down' at each pixel
    phi = np.arctan2(dy, dx); mean = np.arctan2(np.sin(phi).mean(), np.cos(phi).mean()); phi = np.angle(np.exp(1j * (phi - mean)))
    b = np.floor((phi - phi.min()) * np.median(r)).astype(np.int64)  # ~1 px wide bins at the mask
    order = np.lexsort((-s * r, b)); last = np.r_[b[order][1:] != b[order][:-1], True]; i = order[last]
    return xs[i].astype(int), ys[i].astype(int), dx[i] / r[i], dy[i] / r[i], b[i].astype(float)


def measure_photo(ctx, k, obj, gid, scene, rng, frames, mode='columns', below_px=BELOW):
    cam, mask, S = ctx['cams'][k], obj['masks'][k], ctx['S']; n, d = ctx['floor']; h, w = mask.shape
    u, v, du, dv, coord = boundary(mask) if mode == 'columns' else verticals(mask, cam, n)
    if len(u) < 5:
        return None
    lo, span = coord.min(), coord.max() - coord.min()
    edge = (coord < lo + EDGE * span) | (coord > lo + span - EDGE * span)
    at = lambda s: (np.round(u + s * du).astype(int), np.round(v + s * dv).astype(int))
    inside = lambda a, b: (a >= 0) & (b >= 0) & (a < w) & (b < h)
    frame = ~inside(*at(max(BELOW) + 1))  # the pixel past the checked ones is in the frame (pre-registered rule)
    other, other_overlap = np.zeros(len(u), bool), np.zeros(len(u), bool)
    for o in ctx['objects']:
        if o is not obj and k in o['masks']:
            m = o['masks'][k]; below = np.zeros(len(u), bool)
            for s in below_px:
                a, b = at(s); below |= m[np.clip(b, 0, h - 1), np.clip(a, 0, w - 1)]
            other |= below; other_overlap |= below & ~m[v, u]
    C = cam['C']; Db, Dn = rays(cam, u + .5 * du, v + .5 * dv), rays(cam, u + 1.5 * du, v + 1.5 * dv)
    t_own = hits(scene, C, Db, gid)[0]; t_oth_below = hits(scene, C, Dn, gid)[1]
    # plane of the model hits just above the boundary (for rays that miss the model)
    pick = np.nonzero(~edge & ~frame)[0]
    pick = pick[np.unique(np.linspace(0, len(pick) - 1, min(len(pick), MAX_PLANE_COLS)).astype(int))] if len(pick) else pick
    uu, vv = [], []
    for off in range(BAND[0], BAND[1] + 1, BAND_STEP):
        a, b = np.round(u[pick] - off * du[pick]).astype(int), np.round(v[pick] - off * dv[pick]).astype(int)
        ok = inside(a, b); ok[ok] = mask[b[ok], a[ok]]; uu.append(a[ok]); vv.append(b[ok])
    uu, vv = np.concatenate(uu), np.concatenate(vv); Dp = rays(cam, uu, vv); tp = hits(scene, C, Dp, gid)[0]; good = np.isfinite(tp)
    plane = ransac_plane(C + tp[good, None] * Dp[good], PLANE_TOL_M / S, rng) if good.sum() >= MIN_PLANE else None
    plane = plane if plane is not None and plane[2].sum() >= MIN_PLANE else None
    t_plane = np.full(len(u), np.nan)
    if plane is not None:
        with np.errstate(divide='ignore', invalid='ignore'):
            tt = -(plane[0] @ C + plane[1]) / (Db @ plane[0])
        t_plane = np.where(tt > 0, tt, np.nan)
    t = np.where(np.isfinite(t_own), t_own, t_plane)
    occluded = t_oth_below < t
    base = ~edge & ~frame & ~occluded & np.isfinite(t)
    kept, kept_overlap = base & ~other, base & ~other_overlap
    X = C + t[:, None] * Db; hgt = (X @ n + d) * S * 100
    with np.errstate(invalid='ignore'):
        hgt_plane = ((C + t_plane[:, None] * Db) @ n + d) * S * 100
    cm_per_px = S * 100 * ((Dn - Db) @ n) * t; cm_per_cm_range = Db @ n
    # photo's strongest intensity step along 'down' near the boundary
    sw = np.arange(-STEP_WIN, STEP_WIN + 2)
    uv = np.c_[(u[:, None] + sw * du[:, None]).ravel(), (v[:, None] + sw * dv[:, None]).ravel()]
    prof = wsc.bilinear(ctx['images'][k], np.clip(uv, 0, [w - 1, h - 1])).reshape(len(u), len(sw))
    g = np.abs(np.diff(prof, axis=1)); i = g.argmax(1)
    contrast = g.max(1) / (np.median(g, 1) + 1e-3); step_px = (i - STEP_WIN).astype(float)
    sharp = kept & (contrast >= STEP_CONTRAST)
    # Pi3X range just above the boundary
    pi3x = None
    f = next((f for f in frames if f['camera'] == k), None)
    if f is not None and kept.any():
        A = cv2.invertAffineTransform(f['affine'].astype(np.float64)); uq, vq = at(-PI3X_ABOVE_PX); uq, vq = uq[kept], vq[kept]
        ok = inside(uq, vq); ok[ok] = mask[vq[ok], uq[ok]]
        gj = np.round(A[0, 0] * uq + A[0, 1] * vq + A[0, 2]).astype(int); gi = np.round(A[1, 0] * uq + A[1, 1] * vq + A[1, 2]).astype(int)
        H, W = f['P'].shape[:2]; ok &= (gj >= 0) & (gi >= 0) & (gj < W) & (gi < H)
        if ok.sum():
            Pp = f['P'][gi[ok], gj[ok]]; dr = ((Pp - C) * Db[kept][ok]).sum(1) - t[kept][ok]
            pi3x = dict(n=int(ok.sum()), rangeDiffCm=round(float(np.median(dr) * S * 100), 2),
                        impliedHeightCm=round(float(np.median(dr * cm_per_cm_range[kept][ok]) * S * 100), 2))
    reasons = dict(lines=int(len(u)), edge10pct=int(edge.sum()), frameCut=int((frame & ~edge).sum()),
                   otherMaskBelow=int((other & ~edge & ~frame).sum()), occludedBelow=int((occluded & ~edge & ~frame).sum()),
                   noDepth=int((~np.isfinite(t) & ~edge & ~frame).sum()))
    row = dict(photo=k + 1, **stats(hgt[kept]), dropped=reasons, planeFallback=int((kept & ~np.isfinite(t_own)).sum()),
               plane=dict(inliers=int(plane[2].sum()), hits=int(good.sum())) if plane is not None else None,
               overlap=stats(hgt[kept_overlap]), planeVariant=stats(hgt_plane[kept]),
               cmPerPx=round(float(np.median(cm_per_px[kept])), 3) if kept.any() else None,
               cmHeightPerCmRange=round(float(np.median(cm_per_cm_range[kept])), 3) if kept.any() else None,
               rangeM=round(float(np.median(t[kept]) * S), 3) if kept.any() else None,
               photoStep=dict(sharpLines=int(sharp.sum()), medianPx=float(np.median(step_px[sharp])),
                              medianCm=round(float(np.median(step_px[sharp] * cm_per_px[sharp])), 2)) if sharp.any() else None,
               pi3xRange=pi3x)
    return row, dict(X=X[kept], Xo=X[kept_overlap], u=u, v=v, kept=kept, plane_used=kept & ~np.isfinite(t_own),
                     other=other & ~edge & ~frame, occluded=occluded & ~edge & ~frame, edge=edge, nodepth=~np.isfinite(t) & ~edge & ~frame)


def crop(gray, arr, label):
    """Small JPEG of the boundary: kept green (plane fallback cyan), other mask red, occluded magenta, no depth blue, edge grey."""
    u, v = arr['u'], arr['v']; sel = ~arr['edge'] if (~arr['edge']).any() else np.ones(len(u), bool)
    cu, cv = np.percentile(u[sel], [2, 98]), np.percentile(v[sel], [2, 98]); size = max(cu[1] - cu[0], cv[1] - cv[0]) * .6 + 120
    x0, x1 = max(0, int(cu[0] - size)), min(gray.shape[1], int(cu[1] + size))
    y0, y1 = max(0, int(cv[0] - size)), min(gray.shape[0], int(cv[1] + size))
    img = cv2.cvtColor(np.clip(gray[y0:y1, x0:x1], 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    colour = [((128, 128, 128), arr['edge']), ((255, 0, 0), arr['nodepth']), ((255, 0, 255), arr['occluded']),
              ((0, 0, 255), arr['other']), ((0, 200, 0), arr['kept'] & ~arr['plane_used']), ((255, 255, 0), arr['plane_used'])]
    for bgr, which in colour:
        for a, b in zip(u[which], v[which]):
            if x0 <= a < x1 and y0 <= b < y1:
                img[max(0, b - y0 - 1):b - y0 + 2, max(0, a - x0 - 1):a - x0 + 2] = bgr
    s = 600 / max(img.shape[:2])
    if s < 1:
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    cv2.putText(img, label, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 1)
    return cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()


def combined(rows, key=None):
    meds = [(r[key] if key else r).get('medianCm') for r in rows if (r[key] if key else r).get('n', 0) >= MIN_COLS]
    return round(float(np.median(meds)), 2) if meds else None


def run(ctx, opts):
    if not ctx['floor']:
        return dict(status='no_floor', objects={})
    n, d = ctx['floor']; S = ctx['S']
    modes = dict(columns=('columns', BELOW), verticals=('verticals', BELOW), verticalsGap=('verticals', tuple(range(1, GAP_PX + 1))))
    rngs = {m: np.random.default_rng(i) for i, m in enumerate(list(modes) + ['floor'])}
    scene = scene_of([o['mesh'] for o in ctx['objects']])
    frames = pi3x_frames(ctx, opts['api']) if opts.get('api') else []
    cands = floor_candidates(ctx, frames) if frames else []
    want = tuple(opts.get('crops') or ())
    out, files = {}, {}
    for gid, o in enumerate(ctx['objects']):
        V = o['mesh'][0]; entry = dict(label=o['label'], kind=o['kind'], modelBottomCm=round(float(np.percentile((V @ n + d) * S * 100, .5)), 2))
        per = {}
        for mode, (lines, below_px) in modes.items():
            rows, pts, pts_o = [], [], []
            for k in sorted(o['masks']):
                res = measure_photo(ctx, k, o, gid, scene, rngs[mode], frames, lines, below_px)
                if res is None:
                    continue
                row, arr = res; rows.append(row); pts.append(arr['X']); pts_o.append(arr['Xo'])
                if want and o['id'].startswith(want):
                    files[f"{o['id'][:8]}-{mode}-photo{k + 1}.jpg"] = crop(ctx['gray'][k], arr, f"{o['id'][:8]} {mode} photo {k + 1}")
            per[mode] = (rows, pts, pts_o)
            entry[mode] = dict(photos=rows, combinedCm=combined(rows), combinedOverlapCm=combined(rows, 'overlap'),
                               combinedPlaneCm=combined(rows, 'planeVariant'))
        # independent floor from the pre-registered (columns) footprint; the verticals footprint only if that is empty
        foot = next((np.vstack(p) for mode in modes for p in per[mode][1:] if p and len(np.vstack(p))), None)
        if cands and foot is not None:
            plane, diag = local_floor(ctx, cands, foot, rngs['floor']); entry['localFloor'] = diag
            if plane is not None:
                nl, dl = plane
                for mode, (rows, pts, pts_o) in per.items():
                    for row, X, Xo in zip(rows, pts, pts_o):
                        row['localFloor'] = stats((X @ nl + dl) * S * 100); row['localFloorOverlap'] = stats((Xo @ nl + dl) * S * 100)
                    entry[mode]['combinedLocalFloorCm'] = combined(rows, 'localFloor')
                    entry[mode]['combinedLocalFloorOverlapCm'] = combined(rows, 'localFloorOverlap')
        out[o['id']] = entry
    params = dict(edgeFraction=EDGE, belowPx=list(BELOW), planeBandPx=list(BAND), planeTolM=PLANE_TOL_M, minColumns=MIN_COLS,
                  floorRadiusM=[FLOOR_R_M, FLOOR_R2_M], floorBandM=FLOOR_BAND_M, floorTolM=FLOOR_TOL_M, floorMaxDeg=FLOOR_MAX_DEG,
                  floorDilatePx=FLOOR_DILATE_PX, stepWindowPx=STEP_WIN, stepContrast=STEP_CONTRAST, pi3xAbovePx=PI3X_ABOVE_PX,
                  gapPx=GAP_PX, preRegistered='columns (with its overlap / plane variants); verticals and verticalsGap added after the first run')
    return dict(status='ok', nativeToMeters=S, params=params, pi3xFrames=[dict(assetId=f['assetId'], photo=f['camera'] + 1) for f in frames],
                objects=out, files=files)


def _look(C, target, f=500., w=640, h=480, roll=0.):
    z = target - C; z /= np.linalg.norm(z); x = np.cross(z, [0, 0, 1.]); x /= np.linalg.norm(x); y = np.cross(z, x)
    c, s = math.cos(roll), math.sin(roll); R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ np.array([x, y, z]); C = np.asarray(C, float)
    return dict(K=np.array([[f, 0, (w - 1) / 2], [0, f, (h - 1) / 2], [0, 0, 1]]), R=R, t=-R @ C, C=C, w=w, h=h, imageId=str(C))


def _box(lo, hi):
    V, F = wsc.primitive_mesh(dict(kind='box', dimensions=list(np.subtract(hi, lo))))
    return V + (np.add(lo, hi) / 2), F


def _render(cams, meshes):
    """Per camera: grey image and one mask per mesh (first hit), background 100."""
    import open3d as o3d
    truth = scene_of(meshes); gray, masks = [], []
    for c in cams:
        ys, xs = np.mgrid[0:c['h'], 0:c['w']]; D = rays(c, xs.ravel().astype(float), ys.ravel().astype(float))
        g = truth.cast_rays(o3d.core.Tensor(np.hstack([np.broadcast_to(c['C'], D.shape), D]).astype(np.float32)))['geometry_ids'].numpy()
        g = g.astype(np.int64).reshape(c['h'], c['w']); g[g >= len(meshes)] = -1
        gray.append(cv2.GaussianBlur(np.choose(g + 1, [100.] + [200. - 150 * i for i in range(len(meshes))]).astype(np.float32), (0, 0), 1.0))
        masks.append([g == i for i in range(len(meshes))])
    return gray, masks


def _check():
    """Panel 24 cm above the floor z = 0 behind a post standing on it, two cameras: the panel must read 24 cm (the lines behind
    the post dropped), the post ~0 cm, in both line families; with the panel model lifted 3 cm (rays miss) the fitted plane must
    still give 24 cm. A thin vertical post 24 cm up seen by a camera rolled 12 deg: image columns read its slanted side edges
    (too high), the verticals must read 24 cm. A Pi3X-like grid must find its camera; the floor RANSAC a 1 cm offset."""
    rng = np.random.default_rng(0); floor = (np.array([0, 0, 1.]), 0.)
    panel, post = _box((-.6, 2.95, .24), (.6, 3.0, 1.0)), _box((-.05, 2.0, 0), (.05, 2.1, .9))
    cams = [_look(np.array([-.4, 0, 1.5]), np.array([0, 3, .4])), _look(np.array([.5, .2, 1.4]), np.array([0, 3, .4]))]
    gray, masks = _render(cams, [panel, post])
    for lift in (0, .03):
        objs = [dict(id='panel', label='panel', kind='box', mesh=(panel[0] + [0, 0, lift], panel[1]), masks={0: masks[0][0], 1: masks[1][0]}),
                dict(id='post', label='post', kind='box', mesh=post, masks={0: masks[0][1], 1: masks[1][1]})]
        res = run(dict(cams=cams, gray=gray, images=gray, objects=objs, S=1.0, floor=floor, doc={'assets': []}), {})['objects']
        for mode in ('columns', 'verticals', 'verticalsGap'):
            p = res['panel'][mode]
            assert abs(p['combinedCm'] - 24) < .3, (mode, p)
            assert all(r['dropped']['otherMaskBelow'] > 0 for r in p['photos']), (mode, p['photos'])
            if lift:
                assert all(r['planeFallback'] > .9 * r['n'] for r in p['photos']) and abs(p['combinedPlaneCm'] - 24) < .3, (mode, p)
            else:
                assert abs(res['post'][mode]['combinedCm']) < .5 and abs(res['panel']['modelBottomCm'] - 24) < 1e-6, res['post']
    thin, rolled = _box((-.06, 2.5, .24), (.06, 2.56, 1.3)), [_look(np.array([-.3, 0, 1.5]), np.array([0, 2.5, .7]), roll=math.radians(12))]
    g2, m2 = _render(rolled, [thin])
    res = run(dict(cams=rolled, gray=g2, images=g2, objects=[dict(id='thin', label='thin', kind='box', mesh=thin, masks={0: m2[0][0]})],
                   S=1.0, floor=floor, doc={'assets': []}), {})['objects']['thin']
    assert res['columns']['combinedCm'] > 30 and abs(res['verticals']['combinedCm'] - 24) < .5, res
    # Pi3X-like point map of camera 2 at 3 m on a 64 x 48 grid, 10 px per cell
    jj, ii = np.meshgrid(np.arange(64), np.arange(48)); u, v = 10 * jj.ravel() + 5., 10 * ii.ravel() + 5.
    P = (cams[1]['C'] + 3 * rays(cams[1], u, v)).reshape(48, 64, 3)
    k, M = own_camera(P, cams); assert k == 1 and abs(M[0, 0] - 10) < .01, (k, M)
    Q = np.c_[rng.uniform(-1, 1, (3000, 2)), .01 + rng.normal(0, .002, 3000)]; Q[:900, 2] = rng.uniform(-.1, .3, 900)
    nl, dl, _ = ransac_plane(Q, .01, rng, 500, np.array([0, 0, 1.]), 20); assert nl[2] > .999 and abs(dl + .01) < .002, (nl, dl)
    print('clearance self-test passed: panel', p['combinedCm'], 'cm; thin post rolled camera: columns', res['columns']['combinedCm'],
          'verticals', res['verticals']['combinedCm'])


if __name__ == '__main__':
    if len(sys.argv) == 1:
        _check()
    else:  # on-prem: python clearance.py --view V.json --photos-dir D --photo ID=FILE,... --layer-url URL --api ORIGIN --out O.json [--opts J]
        import argparse
        a = argparse.ArgumentParser(); [a.add_argument(f'--{x}', default='') for x in ('view', 'photos-dir', 'photo', 'layer-url', 'api', 'out', 'opts')]
        a = a.parse_args(); pairs = dict(p.split('=', 1) for p in a.photo.split(','))
        ctx = wsc.load_report(Path(a.view).read_bytes(), {i: (Path(a.photos_dir) / f).read_bytes() for i, f in pairs.items()}, a.layer_url, a.api)
        result = run(ctx, {'api': a.api, **(json.loads(Path(a.opts).read_text()) if a.opts else {})})
        out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
        for name, data in result.pop('files').items():
            (out.parent / f'clearance-{name}').write_bytes(data)
        out.write_text(json.dumps(result, indent=1, ensure_ascii=False) + '\n')
