"""Plane stereo: are the flat parts of each model where the photos say they are?

Each displayed model is split into planar parts (area-weighted RANSAC over its faces; nothing object-specific). For every
part seen in two or more photos, the photo where the part covers most usable pixels face-on (inside the object's mask, not
hidden by other models, textured) is the reference; when the object has masks in two or more photos, only those photos are
used. The part's plane is swept along its normal (default +-80 cm, steps of
about one pixel of motion), every other photo is warped onto the reference through that plane's homography, and windowed
NCC over the part's reference pixels scores the plane. Visibility in the other photo comes from the object's mask there when
it has one (a sample outside the mask cannot support the plane, so every plane is scored on the same samples); without a
mask, samples hidden behind other models are left out. Per-pixel best depths give a RANSAC plane, i.e. the normal the photos
measure (a model normal can be 10-20 deg off); the curve is redone along it and the better of the two normals is kept.

Offsets are in cm, positive = farther from the reference camera (along the part normal, at the part centre); rayChangeCm is
the median change along the reference camera's rays (comparable with the shape check's depth change); planeDeviationCm is the
90th-percentile distance of the best plane from the model plane over the part (offset and tilt together).
Verdict per part: ok / offset / ambiguous (a second distinct peak about as good: repeating stripes, grids, or photo pairs
that disagree) / inconclusive (too few pixels, weak or flat curve, peak at the end of the range).
"""
from __future__ import annotations

import io
import math
import time

import cv2
import numpy as np

try:
    import shape_core as core  # Modal: /check/shape_core.py
except ImportError:  # local: scripts/workcell_shape_check.py
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import workcell_shape_check as core

WIN, VAR_MIN, MIN_PX, BUDGET = 11, 10.0, 400, 4e5  # NCC window (working px), texture floor, min pixels, working bbox px


def planar_parts(V, F, S, rng, dist_m=.02, max_angle=30, min_area_m2=.01, min_frac=.03, max_parts=8, iters=256):
    """Face labels (-1 = no part) and planes {n, c: n.X = c} of the largest planar parts (|normal| test: winding-agnostic)."""
    tri = V[F]; cr = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]); dbl = np.linalg.norm(cr, axis=1)
    area = dbl / 2 * S * S; nrm = cr / np.maximum(dbl, 1e-30)[:, None]; cen = tri.mean(1)
    tol, cmin = dist_m / S, math.cos(math.radians(max_angle))
    free, label, parts = area > 0, np.full(len(F), -1), []
    need = max(min_area_m2, min_frac * area.sum())
    inl = lambda n, c, pool: pool & (np.abs(cen @ n - c) < tol) & (np.abs(nrm @ n) > cmin)
    for _ in range(max_parts):
        idx = np.nonzero(free)[0]
        if area[idx].sum() < need:
            break
        p = area[idx] / area[idx].sum(); sub = rng.choice(idx, 20000, p=p)  # area-weighted: counts ~ area
        best, hits = None, -1
        for s in rng.choice(idx, iters, p=p):
            h = int(((np.abs(cen[sub] @ nrm[s] - nrm[s] @ cen[s]) < tol) & (np.abs(nrm[sub] @ nrm[s]) > cmin)).sum())
            if h > hits:
                best, hits = (nrm[s], nrm[s] @ cen[s]), h
        n, c = best
        for _ in range(3):  # refit: area-weighted plane through the inlier faces' corners
            m = inl(n, c, free); w = np.repeat(area[m], 3); P = tri[m].reshape(-1, 3); mu = (w[:, None] * P).sum(0) / w.sum()
            n = np.linalg.eigh(((P - mu) * w[:, None]).T @ (P - mu))[1][:, 0]; c = n @ mu
        m = inl(n, c, free)
        if area[m].sum() < need:
            break
        label[m] = len(parts); free &= ~m
        parts.append(dict(n=n, c=float(c), mu=mu, areaM2=float(area[m].sum()), faces=int(m.sum())))
    return label, parts


def cast(scene, cam, K, box):
    """z-depth (inf = miss) and triangle id of the first hit for the pixel centres of box (x0, y0, x1, y1) under K."""
    import open3d as o3d
    x0, y0, x1, y1 = box; ys, xs = np.mgrid[y0:y1, x0:x1]
    if scene is None:
        return np.full(xs.shape, np.inf, np.float32), np.full(xs.shape, -1)
    q = np.linalg.solve(K, np.vstack([xs.ravel() + 0., ys.ravel(), np.ones(xs.size)]))  # camera rays with z = 1
    rays = np.hstack([np.repeat(cam['C'][None], xs.size, 0), (cam['R'].T @ q).T]).astype(np.float32)
    hit = scene.cast_rays(o3d.core.Tensor(rays))
    prim = hit['primitive_ids'].numpy().astype(np.int64); z = hit['t_hit'].numpy()
    prim[~np.isfinite(z)] = -1
    return z.reshape(xs.shape), prim.reshape(xs.shape)


def views(ctx, obj, V, F, label, others):
    """Per photo at one working scale: image, mask, other models' depth, own part labels / depth / occlusion in the box."""
    cams, S = ctx['cams'], ctx['S']; boxes = []
    for k, cam in enumerate(cams):
        uv, z = core.project(cam, V); b = None
        if (z > 0).any():
            lo = np.clip(uv[z > 0].min(0), 0, [cam['w'], cam['h']]); hi = np.clip(uv[z > 0].max(0), 0, [cam['w'], cam['h']])
            b = [*lo, *hi] if (hi - lo).min() > 2 else None
        if k in obj['masks'] and obj['masks'][k].any():
            ys, xs = np.nonzero(obj['masks'][k]); mb = [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]
            b = mb if b is None else [min(b[0], mb[0]), min(b[1], mb[1]), max(b[2], mb[2]), max(b[3], mb[3])]
        boxes.append(b)
    areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes if b]
    if not areas:
        return None, []
    s = float(np.clip(math.sqrt(BUDGET / max(areas)), .15, 1.)); own = core.raycast_scene([(V, F)]); out = []
    for k, (cam, b) in enumerate(zip(cams, boxes)):
        if b is None:
            out.append(None); continue
        W, H = int(round(cam['w'] * s)), int(round(cam['h'] * s)); sx, sy = W / cam['w'], H / cam['h']
        K = np.array([[sx, 0, (sx - 1) / 2], [0, sy, (sy - 1) / 2], [0, 0, 1.]]) @ cam['K']
        img = cv2.GaussianBlur(cv2.resize(ctx['images'][k], (W, H), interpolation=cv2.INTER_AREA), (0, 0), 1.)  # peaks ~2 px wide
        mask = cv2.resize(obj['masks'][k].astype(np.float32), (W, H), interpolation=cv2.INTER_AREA) > .5 if k in obj['masks'] else None
        box = (int(b[0] * sx), int(b[1] * sy), min(W, int(math.ceil(b[2] * sx)) + 1), min(H, int(math.ceil(b[3] * sy)) + 1))
        bw, bh = box[2] - box[0], box[3] - box[1]
        pad = (max(0, box[0] - bw // 2 - 30), max(0, box[1] - bh // 2 - 30), min(W, box[2] + bw // 2 + 30), min(H, box[3] + bh // 2 + 30))
        z_oth = np.full((H, W), np.inf, np.float32); z_oth[pad[1]:pad[3], pad[0]:pad[2]] = cast(others, cam, K, pad)[0]
        z_own, prim = cast(own, cam, K, box); lab = np.where(prim >= 0, label[np.maximum(prim, 0)], -2)
        zref = np.where(np.isfinite(z_own), z_own, np.median(z_own[np.isfinite(z_own)]) if np.isfinite(z_own).any() else np.inf)
        occ = z_oth[box[1]:box[3], box[0]:box[2]] < zref - (.01 / S + .005 * zref)
        mu = cv2.boxFilter(img, -1, (WIN, WIN)); var = cv2.boxFilter(img * img, -1, (WIN, WIN)) - mu * mu
        ok = ~occ & (mask[box[1]:box[3], box[0]:box[2]] if mask is not None else True)
        out.append(dict(k=k, K=K, img=img, mask=mask, z_oth=z_oth, box=box, lab=lab, ok=ok,
                        textured=var[box[1]:box[3], box[0]:box[2]] > VAR_MIN,
                        usable=int(max((ok & (lab >= -1)).sum(), (ok & (mask[box[1]:box[3], box[0]:box[2]] if mask is not None else False)).sum()))))
    return s, out


def sweep(ctx, ref, others, region, plane0, range_cm, rng, max_px=40000, max_planes=400):
    """Plane-level NCC curves along the model normal and along the normal the photos measure (RANSAC on per-pixel depths).

    Offsets are sampled so that the warped pixels move about one pixel per step (stripes and sharp texture have sub-window
    correlation peaks: a fixed 1 cm step can jump over them)."""
    cams, S = ctx['cams'], ctx['S']; A = cams[ref['k']]; Kri = np.linalg.inv(ref['K'])
    ys, xs = np.nonzero(region); ys = ys + ref['box'][1]; xs = xs + ref['box'][0]
    if len(xs) > max_px:
        keep = rng.choice(len(xs), max_px, replace=False); ys, xs = ys[keep], xs[keep]
    h = WIN // 2 + 1; x0, y0 = max(0, xs.min() - h), max(0, ys.min() - h)
    x1, y1 = min(ref['img'].shape[1], xs.max() + h + 1), min(ref['img'].shape[0], ys.max() + h + 1)
    patch = ref['img'][y0:y1, x0:x1]; box = lambda a: cv2.boxFilter(a, -1, (WIN, WIN))
    mr = box(patch); vr = box(patch * patch) - mr * mr; yc, xc = ys - y0, xs - x0
    q = Kri @ np.vstack([xs + 0., ys, np.ones(len(xs))]); T = np.array([[1, 0, x0], [0, 1, y0], [0, 0, 1.]])
    n0, c0 = plane0['n'], plane0['c']
    if n0 @ A['C'] < c0:
        n0, c0 = -n0, -c0  # normal faces the reference camera
    dw = A['R'].T @ (Kri @ [np.median(xs), np.median(ys), 1.])
    lam = lambda d: (c0 - d / 100 / S - n0 @ A['C']) / (n0 @ dw)  # centre ray parameter of the model plane moved d cm back
    zmin = .2 * lam(0)
    rel = [(o, cams[o['k']]['R'] @ A['R'].T) for o in others]; rel = [(o, R, cams[o['k']]['t'] - R @ A['t']) for o, R in rel]

    def plane(d, n):  # normal n through the centre-ray point of the model plane moved d cm back
        return n, n @ (A['C'] + lam(d) * dw)

    def depth(n, c):
        return (c - n @ A['C']) / ((A['R'] @ n) @ q)

    def project(o, R, t, z):
        Xo = R @ (q * z) + t[:, None]; zo = Xo[2]
        return (o['K'] @ Xo)[:2] / np.where(zo > 0, zo, np.nan), zo

    def score(n, c):
        """Pixel-weighted mean NCC over the photo pairs, per-pair means and counts, and per-pixel NCC (nan = not usable)."""
        nr = A['R'] @ n; cr = c - n @ A['C']; z = cr / (nr @ q)
        if not (z > zmin).all():
            return np.nan, [np.nan] * len(rel), [(0, 0)] * len(rel), np.full((len(rel), len(xs)), np.nan)
        vals, cnt, per = [], [], []
        for o, R, t in rel:
            uv, zo = project(o, R, t, z); Hh, Ww = o['img'].shape; r_all = np.full(len(xs), np.nan)
            ok = (zo > 0) & (uv[0] >= h) & (uv[1] >= h) & (uv[0] < Ww - h) & (uv[1] < Hh - h)
            ui = np.nan_to_num(np.round(uv)).astype(int).clip(0, [[Ww - 1], [Hh - 1]])
            hidden = ok & (o['z_oth'][ui[1], ui[0]] < zo - (.01 / S + .005 * zo))  # behind another model in that photo
            if o['mask'] is None:  # with a mask, the photo itself says where the object is visible (scored below)
                ok &= ~hidden
            if ok.sum() < max(200, .15 * len(xs)):
                vals.append(np.nan); cnt.append((int(ok.sum()), int(hidden.sum()))); per.append(r_all); continue
            H = o['K'] @ (R + np.outer(t, nr) / cr) @ Kri @ T
            w = cv2.warpPerspective(o['img'], H, patch.shape[::-1], flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
            mw = box(w); vw = box(w * w) - mw * mw; cov = box(patch * w) - mr * mw
            r = np.where(vw > VAR_MIN, cov / np.sqrt(np.maximum(vr * vw, 1e-6)), 0.)[yc[ok], xc[ok]]
            if o['mask'] is not None:  # a sample outside the object's mask in that photo cannot support the plane
                r = np.where(o['mask'][ui[1][ok], ui[0][ok]], r, np.minimum(r, 0.))
            r_all[ok] = r; vals.append(float(r.mean())); cnt.append((int(ok.sum()), int(hidden.sum()))); per.append(r_all)
        v, k = np.array(vals), np.array([c[0] for c in cnt], float); good = ~np.isnan(v)
        return (float((v[good] * k[good]).sum() / k[good].sum()) if good.any() else np.nan), vals, cnt, np.array(per)

    # offset step: about one pixel of motion in the photo that moves most
    z0 = depth(*plane(0, n0)); z1 = depth(*plane(1, n0)); motion = 0.
    for o, R, t in rel:
        a, _ = project(o, R, t, z0); b, _ = project(o, R, t, z1); m = np.linalg.norm(b - a, axis=0)
        motion = max(motion, float(np.nanmedian(m)) if np.isfinite(m).any() else 0.)
    step = float(np.clip(1 / max(motion, 1e-6), .2, 2.)); step = max(step, 2 * range_cm / max_planes)
    K = int(range_cm // step); offs = np.arange(-K, K + 1) * step
    first = [score(*plane(d, n0)) for d in offs]
    # per-pixel depths along the model normal (both pairs must agree when there are two), RANSAC plane through them
    P = np.array([np.where(np.isnan(r[3]), np.inf, r[3]).min(0) for r in first]); P[~np.isfinite(P)] = -1.
    bi = P.argmax(0); peak = P.max(0); idx = np.arange(len(offs))[:, None]
    second = np.where(np.abs(idx - bi[None]) > 3, P, -1.).max(0)
    conf = (peak > .5) & (peak - second > .05) & (bi > 0) & (bi < len(offs) - 1)
    n_meas, inliers = None, 0
    if conf.sum() >= max(100, .05 * len(xs)):
        zc = np.array([depth(*plane(offs[i], n0))[j] for j, i in zip(np.nonzero(conf)[0], bi[conf])])
        X = A['C'] + (A['R'].T @ (q[:, conf] * zc)).T; thr = max(1.5, 1.5 * step) / 100 / S; best = None
        for _ in range(1500):
            a, b, c = X[rng.choice(len(X), 3, replace=False)]; n = np.cross(b - a, c - a); L = np.linalg.norm(n)
            if L > 1e-12:
                n /= L; m = np.abs((X - a) @ n) < thr
                if best is None or m.sum() > best.sum():
                    best = m
        if best is not None and best.sum() >= max(60, .3 * conf.sum()):
            Q = X[best]; _, sv, vt = np.linalg.svd(Q - Q.mean(0), full_matrices=False)
            if sv[1] / sv[0] > .1 and sv[1] / math.sqrt(len(Q)) * S > .05:  # a strip (narrower than ~17 cm) does not fix a plane
                n_meas, inliers = (vt[2] if vt[2] @ n0 > 0 else -vt[2]), int(best.sum())
    final, n_best, used = first, n0, 'model'
    if n_meas is not None and math.degrees(math.acos(min(1., n_meas @ n0))) > .5:
        second_try = [score(*plane(d, n_meas)) for d in offs]
        if np.nanmax([r[0] for r in second_try] + [-9]) > np.nanmax([r[0] for r in first] + [-9]):
            final, n_best, used = second_try, n_meas, 'measured'
    comb = np.array([r[0] for r in final]); ib = int(np.nanargmax(comb)) if not np.isnan(comb).all() else K
    z_model = depth(n0, c0); z_best = depth(*plane(offs[ib], n_best))
    return dict(offsets=offs.tolist(), final=comb.tolist(), modelCurve=[r[0] for r in first], atModel=first[K][0],
                pairs=[[r[1][j] for r in final] for j in range(len(rel))], pairPixels=[max(r[2][j][0] for r in final) for j in range(len(rel))],
                pairUsable=[[round(r[2][j][0] / len(xs), 2) for r in final] for j in range(len(rel))],  # fraction of samples usable
                pairHidden=[[round(r[2][j][1] / len(xs), 2) for r in final] for j in range(len(rel))],  # ... hidden by other models
                stepCm=step, pxPerCm=motion, normalFrom=used, confidentPixels=int(conf.sum()), planeInliers=inliers,
                tiltDeg=float(np.degrees(np.arccos(min(1., abs(n_best @ n0))))), cosRef=float(abs(n0 @ dw) / np.linalg.norm(dw)),
                rayChangeCm=float(np.median((z_best - z_model) * np.linalg.norm(q, axis=0)) * S * 100),
                planeDeviationCm=float(np.percentile(np.abs(n0 @ (A['C'][:, None] + A['R'].T @ (q * z_best)) - c0), 90) * S * 100),
                distanceM=float(lam(0) * np.linalg.norm(dw) * S), normal=n0.tolist(), bestNormal=n_best.tolist(), pixels=int(len(xs)))


def second_peak(x, y, sep=6.):
    """Best distinct local maximum farther than sep from the global one (the curve dips >= .05 between); range ends excluded."""
    y = np.array(y, float); i = int(np.nanargmax(y)); second, at = np.nan, None
    for j in range(1, len(y) - 1):
        if np.isnan(y[j]) or abs(x[j] - x[i]) <= sep or not (y[j] >= np.nan_to_num(y[j - 1], nan=-9) and y[j] >= np.nan_to_num(y[j + 1], nan=-9)):
            continue
        lo, hi = sorted((i, j))
        if np.nanmin(y[lo:hi + 1]) < y[j] - .05 and not y[j] <= np.nan_to_num(second, nan=-9):
            second, at = float(y[j]), float(x[j])
    return float(y[i]), second, at


def assess(sw, tol_cm):
    """Verdict from the agreement-versus-offset curve (along the better of the model and measured normals)."""
    offsets = np.array(sw['offsets']); c = np.array(sw['final'], float); valid = ~np.isnan(c)
    if valid.sum() < 5:
        return dict(verdict='inconclusive', reason='too_few_valid_planes')
    i = int(np.nanargmax(c)); peak = float(c[i]); best = float(offsets[i]); contrast = peak - float(np.nanmin(c))
    at0 = None if sw['atModel'] is None or math.isnan(sw['atModel']) else float(sw['atModel'])
    ppeak, second, second_at = second_peak(offsets, c, max(4., 4 * sw['stepCm']))
    pair_best = [float(offsets[int(np.nanargmax(q))]) if not np.isnan(q).all() else None for q in (np.array(x, float) for x in sw['pairs'])]
    row = dict(bestOffsetCm=best, nccAtBest=peak, nccAtModel=at0, curveContrast=contrast, secondPeakNcc=None if math.isnan(second) else second,
               secondPeakOffsetCm=second_at, pairBestOffsetCm=pair_best, toleranceCm=round(tol_cm, 1))
    spread = [q for q in pair_best if q is not None]
    if i in (0, len(c) - 1) or np.isnan(c[i - 1]) or np.isnan(c[i + 1]):
        row.update(verdict='inconclusive', reason='peak_at_range_end')
    elif peak < .3 or contrast < .2:
        row.update(verdict='inconclusive', reason='weak_or_flat_curve')
    elif not math.isnan(second) and second > peak - .1:
        row.update(verdict='ambiguous', reason='second_peak')
    elif len(spread) > 1 and max(spread) - min(spread) > 2 * tol_cm:
        row.update(verdict='ambiguous', reason='photo_pairs_disagree')
    elif sw['planeDeviationCm'] > tol_cm and (at0 is None or peak - at0 > .05):
        row.update(verdict='offset')
    else:
        row.update(verdict='ok')
    return row


def check_object(ctx, obj, mesh, others, range_cm, rng):
    V, F = mesh; S = ctx['S']; label, parts = planar_parts(V, F, S, rng)
    if not parts:
        return dict(verdict='inconclusive', reason='no_planar_part', parts=[])
    s, vs = views(ctx, obj, V, F, label, others)
    seen = [v for v in vs if v and v['usable'] >= MIN_PX]
    if sum(v['mask'] is not None for v in seen) >= 2:  # masks say where the object is visible: unmasked photos only as a fallback
        seen = [v for v in seen if v['mask'] is not None]
    if len(seen) < 2:
        return dict(verdict='inconclusive', reason='seen_in_fewer_than_2_photos', workScale=s, parts=[dict(areaM2=p['areaM2']) for p in parts])
    rows = []
    for j, p in enumerate(parts):
        regions = {v['k']: cv2.erode((v['ok'] & v['textured'] & (v['lab'] == j)).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0 for v in seen}
        cos = {v['k']: abs(p['n'] @ (ctx['cams'][v['k']]['C'] - p['mu'])) / np.linalg.norm(ctx['cams'][v['k']]['C'] - p['mu']) for v in seen}
        masked = [v for v in seen if v['mask'] is not None and regions[v['k']].sum() >= MIN_PX]
        ref = max(masked or seen, key=lambda v: regions[v['k']].sum() * cos[v['k']])  # many pixels, seen face-on
        row = dict(part=j, areaM2=round(p['areaM2'], 4), faces=p['faces'], refPhoto=ref['k'])
        if ctx.get('floor'):
            row['tiltFromHorizontalDeg'] = round(float(np.degrees(np.arccos(min(1., abs(p['n'] @ ctx['floor'][0]))))), 1)
        sw = sweep(ctx, ref, [v for v in seen if v is not ref], regions[ref['k']], p, range_cm, rng) if regions[ref['k']].sum() >= MIN_PX else None
        if sw is None:
            rows.append(dict(row, verdict='inconclusive', reason='too_few_pixels_in_two_photos', pixels=int(regions[ref['k']].sum()))); continue
        if ctx.get('floor'):
            row['bestTiltFromHorizontalDeg'] = round(float(np.degrees(np.arccos(min(1., abs(np.array(sw['bestNormal']) @ ctx['floor'][0]))))), 1)
        tol = max(3., sw['distanceM'], 2 * sw['stepCm'])  # 1 % of the distance, at least 3 cm
        row.update(assess(sw, tol), photos=[ref['k']] + [v['k'] for v in seen if v is not ref],
                   **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in sw.items() if k not in ('final', 'atModel', 'pairs', 'modelCurve', 'offsets')},
                   offsets=sw['offsets'], curve=sw['final'], modelCurve=sw['modelCurve'], pairCurves=sw['pairs'])
        row['photoPairs'] = [[ref['k'], v['k']] for v in seen if v is not ref]
        rows.append(row)
    return dict(summarize(rows), workScale=round(s, 3), photosSeen=[v['k'] for v in seen], parts=rows)


def summarize(rows):
    dec = [r for r in rows if r['verdict'] in ('ok', 'offset')]
    if not dec:
        return dict(verdict='ambiguous' if any(r['verdict'] == 'ambiguous' for r in rows) else 'inconclusive')
    off = [r for r in dec if r['verdict'] == 'offset']; w_off = sum(r['pixels'] for r in off); w_ok = sum(r['pixels'] for r in dec) - w_off
    out = dict(verdict='ok' if not off else 'offset' if w_off > 2 * w_ok else 'mixed', decisiveParts=len(dec), offsetParts=len(off))
    if off:
        out['medianRayChangeCm'] = float(np.median([r['rayChangeCm'] for r in off]))
    return out


def plot(objects):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    items = [(k, v) for k, v in objects.items() if any('curve' in p for p in v.get('parts', []))]
    if not items:
        return None
    cols = 4; rows = -(-len(items) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 2.6 * rows), squeeze=False)
    for ax, (k, v) in zip(axes.ravel(), items):
        for p in v['parts']:
            if 'curve' in p:
                ax.plot(p['offsets'], p['curve'], lw=1, label=f"{p['part']}: {p['verdict'][:4]} {p.get('bestOffsetCm', math.nan):+.0f}")
        ax.axvline(0, color='k', lw=.5); ax.set_title(f"{k[:8]} {v['verdict']}", fontsize=8); ax.legend(fontsize=6); ax.tick_params(labelsize=6)
    for ax in axes.ravel()[len(items):]:
        ax.axis('off')
    fig.supxlabel('plane offset along normal, cm (+ = farther from reference camera)', fontsize=8); fig.supylabel('windowed NCC', fontsize=8)
    fig.tight_layout(); buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=70); plt.close(fig)
    return buf.getvalue()


def run(ctx, opts):
    """opts: rangeCm (80), only ([entity ids]), original (True: also check the imported model where a layer replaced
    it), workers (8 threads, one object each)."""
    from concurrent.futures import ThreadPoolExecutor
    range_cm = float(opts.get('rangeCm', 80))

    def one(i):
        o = ctx['objects'][i]; t0 = time.monotonic(); rng = np.random.default_rng(i)
        others = [p['mesh'] for j, p in enumerate(ctx['objects']) if j != i]
        scene = core.raycast_scene(others) if others else None
        row = dict(label=o['label'], **check_object(ctx, o, o['mesh'], scene, range_cm, rng))
        if opts.get('original', True) and o['original'] is not o['mesh']:
            row['originalModel'] = check_object(ctx, o, o['original'], scene, range_cm, rng)
        return o['id'], dict(row, seconds=round(time.monotonic() - t0, 1))

    todo = [i for i, o in enumerate(ctx['objects']) if not opts.get('only') or o['id'] in opts['only']]
    threads = cv2.getNumThreads(); cv2.setNumThreads(1)  # ponytail: one object per thread; cv2/numpy release the GIL
    try:
        with ThreadPoolExecutor(opts.get('workers', 8)) as pool:
            out = dict(pool.map(one, todo))
    finally:
        cv2.setNumThreads(threads)
    png = plot(out)
    return dict(method='plane stereo: RANSAC planar parts; homography sweep along the model normal (about 1 px of motion per step), '
                       'per-pixel depths -> RANSAC plane for the measured normal, curve along the better normal; windowed NCC (11 px) '
                       'over the part pixels inside the masks, other models as occluders', rangeCm=range_cm, objects=out,
                **({'files': {'curves.png': png}} if png else {}))


def _synthetic(texture, offset_m, tilt_deg=0.):
    """Two cameras 0.8 apart looking at the textured plane z = 3; the model plane is placed offset_m farther, turned tilt_deg."""
    K = np.array([[800, 0, 400], [0, 800, 300], [0, 0, 1.]]); cams, images = [], []
    for x in (-.4, .4):
        M = np.eye(4); M[:3, 3] = [x, 0, 0]
        cams.append(core.camera(dict(cameraToWorld=M.tolist(), K=K.tolist(), width=800, height=600, imageId=str(x))))
        ys, xs = np.mgrid[0:600, 0:800]; rays = (np.linalg.inv(K) @ np.vstack([xs.ravel(), ys.ravel(), np.ones(xs.size)])).T
        Q = cams[-1]['C'] + rays * 3; images.append(texture(Q).reshape(600, 800).astype(np.float32))
    a = math.radians(tilt_deg); V = np.array([[-.7, -.5, 0], [.7, -.5, 0], [.7, .5, 0], [-.7, .5, 0]]); F = np.array([[0, 1, 2], [0, 2, 3]])
    V = V @ np.array([[math.cos(a), 0, -math.sin(a)], [0, 1, 0], [math.sin(a), 0, math.cos(a)]]).T + [0, 0, 3 + offset_m]
    mask = np.zeros((600, 800), bool); mask[120:480, 200:600] = True  # the object's outline in photo 0
    obj = dict(id='plane', label='plane', kind='primitive', mesh=(V, F), masks={0: mask}); obj['original'] = obj['mesh']
    ctx = dict(cams=cams, gray=images, images=images, objects=[obj], S=1., floor=None)
    return run(ctx, {'rangeCm': 40})['objects']['plane']


def _check():
    rng = np.random.default_rng(0)
    tex = cv2.GaussianBlur((rng.random((600, 600)) * 255).astype(np.float32), (0, 0), 2)
    noise = lambda Q: core.bilinear(tex, np.c_[(Q[:, 0] + 1.5) * 200, (Q[:, 1] + 1.5) * 200])
    r = _synthetic(noise, .10)
    p = r['parts'][0]
    assert p['verdict'] == 'offset' and abs(p['bestOffsetCm'] + 10) <= p['stepCm'] and p['tiltDeg'] <= 2.1, {k: v for k, v in p.items() if 'urve' not in k}
    t = _synthetic(noise, 0., 10.)['parts'][0]  # model turned 10 deg about its centre: the tilt search must undo it
    assert t['verdict'] == 'offset' and abs(t['tiltDeg'] - 10) <= 2.1 and abs(t['bestNormal'][2] + 1) < 1e-3, {k: v for k, v in t.items() if 'urve' not in k}
    assert p['nccAtBest'] > .8 and p['nccAtBest'] - p['nccAtModel'] > .3 and r['verdict'] == 'offset', p
    stripes = lambda Q: 128 + 100 * np.sin(2 * np.pi * Q[:, 0] / .045)  # disparity alias every ~17 cm
    s = _synthetic(stripes, 0.)['parts'][0]
    assert s['verdict'] == 'ambiguous', {k: v for k, v in s.items() if 'urve' not in k}
    print('plane stereo self-test passed: best', p['bestOffsetCm'], 'cm, tilted model', t['tiltDeg'], 'deg, ncc', round(p['nccAtModel'], 2), '->', round(p['nccAtBest'], 2),
          '| stripes:', s['verdict'], s['bestOffsetCm'], s['secondPeakOffsetCm'])


if __name__ == '__main__':
    _check()
