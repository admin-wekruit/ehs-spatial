"""Plane facets: the flat faces of an object and the folds between them, from the photos alone (no model shape is used).

Each photo with the object's mask is the reference once: plane sweep in inverse depth over the displayed model's depth range
in that camera (widened by depthFactor), fronto-parallel and along each planar part's normal of the displayed models of the
object and its parts (the models give search directions and the range only, never a position), 13 px NCC against every other
photo, the minimum over the other photos (all must agree: repeating stripes alias at different depths in different pairs), a
reprojection outside the object's mask in a photo that has one scores -1, a distinct peak (best - best farther than 4 % in
depth > .08) and parabolic sub-step refinement. A point is kept when another reference has one within confirmCm.
Sequential RANSAC (thrCm, local kNN normals within 30 deg of the plane) gives the planes, each refit by SVD. Per plane: unit
normal (towards the cameras), inclination to the floor, height range above it, in-plane width / slant, RMS, support (points per
reference photo). Folds: every plane pair whose supports both come within 8 cm of their intersection line, on one side of it,
along a common stretch: the interior angle between the two half-planes and the fold line (end points, heights, length).
This is the September guard measurement (research-notes/sept-guard-shape-2026-10-05/code sweep.py / compare.py / facets.py)
as a generic check.
"""
from __future__ import annotations

import io
import math

import cv2
import numpy as np

import argus.checks.shape_core as core

WIN = 13


def scaled(cam, f):
    K = np.array([[f, 0, (f - 1) / 2], [0, f, (f - 1) / 2], [0, 0, 1.]]) @ cam['K']  # pixel centres
    return dict(K=K, R=cam['R'], t=cam['t'], C=cam['C'])


def sweep_ref(ctx, obj, r, f, zlo, zhi, planes, normals):
    """Confident 3D points (native world) and their working-scale pixels for reference photo r.

    Sweep directions: fronto-parallel plus each world normal in `normals` (slanted boards keep their texture under the plane's
    homography; a fronto-parallel window smears it). Per pixel the best plane over every direction wins; it must beat the best
    plane farther than 4 % in depth (any direction) by .08."""
    cams = ctx['cams']; A = scaled(cams[r], f)
    img = [cv2.resize(g, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) for g in ctx['gray']]
    msk = {k: cv2.resize(m.astype(np.float32), img[k].shape[::-1], interpolation=cv2.INTER_AREA) for k, m in obj['masks'].items() if m.any()}
    m = msk[r] > .5; ys, xs = np.nonzero(m); h = WIN
    x0, y0, x1, y1 = max(0, xs.min() - h), max(0, ys.min() - h), min(m.shape[1], xs.max() + h + 1), min(m.shape[0], ys.max() + h + 1)
    ref = img[r][y0:y1, x0:x1]; box = lambda a: cv2.boxFilter(a, -1, (WIN, WIN), borderType=cv2.BORDER_REFLECT)
    mr = box(ref); vr = box(ref * ref) - mr * mr; inside = m[y0:y1, x0:x1]
    T = np.array([[1, 0, x0], [0, 1, y0], [0, 0, 1.]]); Kinv = np.linalg.inv(A['K'])
    gy, gx = np.mgrid[y0:y1, x0:x1]; q = np.einsum('ij,jhw->ihw', Kinv, np.stack([gx + 0., gy, np.ones_like(gx, float)]))  # rays, z = 1
    P, Z, S2 = (np.full(ref.shape, v, np.float32) for v in (-1., np.nan, -1.))
    rel = [(o, scaled(cams[o], f)) for o in range(len(cams)) if o != r]
    for nw in [None] + list(normals):
        nr = np.array([0, 0, 1.]) if nw is None else A['R'] @ np.asarray(nw, float)
        cq = np.einsum('i,ihw->hw', nr, q)
        if np.median(cq[inside]) < 0:
            nr, cq = -nr, -cq
        use = inside & (cq > .15)  # not seen edge-on from the reference
        if use.sum() < 200:
            continue
        dlo, dhi = zlo * cq[use].min(), zhi * cq[use].max()  # every pixel's depth range is covered
        n = int(min(3 * planes, planes * cq[use].max() * (1 / dlo - 1 / dhi) / (1 / zlo - 1 / zhi)))  # fronto-parallel step at worst
        inv = np.linspace(1 / dhi, 1 / dlo, n)  # 1 / d: inverse depth is linear in it at every pixel
        both = np.ones((n,) + ref.shape, np.float32)
        for o, B in rel:
            R = B['R'] @ A['R'].T; t = B['t'] - R @ A['t']
            for i, w_ in enumerate(inv):
                H = B['K'] @ (R + np.outer(t, nr) * w_) @ Kinv @ T  # ref pixel -> other pixel for the plane nr . X = 1 / w_
                w = cv2.warpPerspective(img[o], H, ref.shape[::-1], flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderValue=-1)
                valid = cv2.erode((w >= 0).astype(np.uint8), np.ones((WIN, WIN), np.uint8)) > 0
                mw = box(w); vw = box(w * w) - mw * mw; c = (box(ref * w) - mr * mw) / np.sqrt(np.maximum(vr * vw, 1e-6))
                c[~valid | (vr < 20) | (vw < 20)] = -1
                if o in msk:  # the photo's own outline of the object: a match outside it is not on the object
                    c[cv2.warpPerspective(msk[o], H, ref.shape[::-1], flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP) < .5] = -1
                both[i] = np.minimum(both[i], c)
        best = both.argmax(0); peak = np.take_along_axis(both, best[None], 0)[0]; second = np.full(peak.shape, -1., np.float32)
        for i, w_ in enumerate(inv):
            second = np.where(np.abs(w_ / inv[best] - 1) > .04, np.maximum(second, both[i]), second)
        b = np.clip(best, 1, n - 2); ym, y0_, yp = (np.take_along_axis(both, (b + k)[None], 0)[0] for k in (-1, 0, 1))
        den = ym - 2 * y0_ + yp; delta = np.clip(np.where(den < 0, .5 * (ym - yp) / np.where(den < 0, den, -1), 0), -.5, .5)
        z = 1 / (cq * (inv[b] + delta * (inv[1] - inv[0]))); edge = (best == 0) | (best == n - 1)
        peak[~use | edge] = -1; z[~use] = np.nan
        far = ~(np.abs(z / Z - 1) <= .04)  # nan (no previous winner) counts as far
        win = peak > P
        S2 = np.where(win, np.maximum(second, np.where(far, P, S2)), np.maximum(S2, np.where(far, peak, second)))
        Z = np.where(win, z, Z); P = np.where(win, peak, P)
    ok = inside & (P > .6) & (P - S2 > .08) & np.isfinite(Z)
    yy, xx = np.nonzero(ok); Xc = q[:, yy, xx] * Z[yy, xx]
    X = (A['R'].T @ (Xc - A['t'][:, None])).T
    return X, np.c_[xx + x0, yy + y0] / f, dict(maskPx=int(m.sum()), confident=int(ok.sum()), medianPeak=round(float(np.median(P[ok])), 3) if ok.any() else None)


def sweep_normals(ctx, obj, rng, sep_deg=8.):
    """Search directions from the displayed models of the object and its parts (planar parts' normals, deduplicated)."""
    try:
        from argus.checks.plane_stereo import planar_parts
    except ImportError:
        from argus.checks.plane_stereo import planar_parts
    kids = {e['id'] for e in (ctx.get('doc') or {}).get('entities', []) if e.get('parentEntityId') == obj['id']}
    out = []
    for o in [obj] + [x for x in ctx['objects'] if x['id'] in kids]:
        for p in planar_parts(*o['mesh'], ctx['S'], rng)[1]:
            if all(abs(p['n'] @ n) < math.cos(math.radians(sep_deg)) for n in out):
                out.append(p['n'])
    return out


def local_normals(X, k=24):
    from scipy.spatial import cKDTree
    _, idx = cKDTree(X).query(X, k=min(k, len(X)))
    Q = X[idx] - X[idx].mean(1, keepdims=True)
    return np.linalg.eigh(np.einsum('nki,nkj->nij', Q, Q))[1][:, :, 0]


def ransac(P, N, thr, rng, iters=3000, cmin=math.cos(math.radians(30))):
    sub = rng.choice(len(P), min(len(P), 20000), replace=False); best, hits = None, -1
    for _ in range(iters):
        a, b, c = P[rng.choice(len(P), 3, replace=False)]; n = np.cross(b - a, c - a); L = np.linalg.norm(n)
        if L < 1e-12:
            continue
        n /= L; s = int(((np.abs((P[sub] - a) @ n) < thr) & (np.abs(N[sub] @ n) > cmin)).sum())
        if s > hits:
            best, hits = (n, a), s
    n, c = best
    for _ in range(3):
        inl = (np.abs((P - c) @ n) < thr) & (np.abs(N @ n) > cmin); c = P[inl].mean(0); n = np.linalg.svd(P[inl] - c, full_matrices=False)[2][-1]
    return n, c, (np.abs((P - c) @ n) < thr) & (np.abs(N @ n) > cmin)


def plane_row(n, c, P, src, S, up, off, Cm):
    n = n if n @ (Cm - c) > 0 else -n
    hd = np.cross(up, n); hd = hd / np.linalg.norm(hd) if np.linalg.norm(hd) > 1e-6 else np.cross(n, [1., 0, 0]) / np.linalg.norm(np.cross(n, [1., 0, 0]))
    vd = np.cross(n, hd); hh, vv = (P - c) @ hd, (P - c) @ vd; h = (P @ up + off) * S
    (h0, h1), (v0, v1) = np.percentile(hh, [2, 98]), np.percentile(vv, [2, 98])
    corners = [(c + a * hd + b * vd).tolist() for a, b in ((h0, v0), (h1, v0), (h1, v1), (h0, v1))]
    return dict(normal=n.tolist(), center=c.tolist(), offsetNative=float(n @ c), inclinationDeg=round(float(np.degrees(np.arccos(min(1., abs(n @ up))))), 1),
                heightM=[round(float(np.percentile(h, 2)), 3), round(float(np.percentile(h, 98)), 3)], widthM=round(float((h1 - h0) * S), 3),
                slantM=round(float((v1 - v0) * S), 3), rmsCm=round(float(np.sqrt(np.mean(((P - c) @ n) ** 2)) * S * 100), 2), points=int(len(P)),
                pointsPerPhoto={int(k): int((src == k).sum()) for k in np.unique(src)}, cornersNative=corners, horizontalDir=hd.tolist(), slantDir=vd.tolist())


def fold(a, b, Pa, Pb, S, up, off, near_m=.08, min_pts=30):
    """Fold between two measured planes, or None when their supports do not meet along a common stretch on one side each."""
    na, ca, nb, cb = map(np.asarray, (a['normal'], a['center'], b['normal'], b['center']))
    d = np.cross(na, nb); L = np.linalg.norm(d)
    if L < math.sin(math.radians(15)):
        return None
    d /= L; p0 = np.linalg.solve(np.array([na, nb, d]), [na @ ca, nb @ cb, 0])
    near = lambda P: P[np.linalg.norm(np.cross(P - p0, d), axis=1) < near_m / S]
    Qa, Qb = near(Pa), near(Pb)
    if min(len(Qa), len(Qb)) < min_pts:
        return None
    sa, sb = (Qa - p0) @ d, (Qb - p0) @ d
    lo, hi = max(np.percentile(sa, 5), np.percentile(sb, 5)), min(np.percentile(sa, 95), np.percentile(sb, 95))
    if (hi - lo) * S < .05:
        return None

    def leg(n, P):  # in-plane direction from the fold line into the face, and the share of the face's nearby support on that side
        Q = P[np.linalg.norm(np.cross(P - p0, d), axis=1) < .3 / S]; v = Q - p0; v -= np.outer(v @ d, d); v -= np.outer(v @ n, n)
        u = v.mean(0); u /= np.linalg.norm(u)
        return u, float((v @ u > 0).mean())
    (ua, fa), (ub, fb) = leg(na, Pa), leg(nb, Pb)
    if min(fa, fb) < .8:
        return None  # a face crosses the line: a T or a crossing, not a fold edge
    a0, a1 = p0 + d * lo, p0 + d * hi; ht = lambda X: round(float((X @ up + off) * S), 3)
    return dict(interiorDeg=round(float(np.degrees(np.arccos(np.clip(ua @ ub, -1, 1)))), 1), lineNative=[a0.tolist(), a1.tolist()],
                lineHeightM=[ht(a0), ht(a1)], lengthM=round(float((hi - lo) * S), 3), legs=[ua.tolist(), ub.tolist()], oneSided=[round(fa, 3), round(fb, 3)],
                nearPoints=[int(len(Qa)), int(len(Qb))], lineSlopeDeg=round(float(np.degrees(np.arcsin(min(1., abs(d @ up))))), 1))


def check_object(ctx, obj, opts, rng):
    from scipy.spatial import cKDTree
    S = ctx['S']; refs = [k for k, m in obj['masks'].items() if m.any()]
    if len(refs) < 2:
        return dict(verdict='inconclusive', reason='mask_in_fewer_than_2_photos')
    up, off = ctx['floor'] if ctx.get('floor') else (np.array([0, -1., 0]), 0.)
    V = obj['mesh'][0]; clouds, uvs, stats = {}, {}, {}
    normals = [np.asarray(n, float) for n in opts['normals']] if opts.get('normals') else sweep_normals(ctx, obj, rng) if opts.get('modelNormals', True) else []
    for r in refs:
        cam = ctx['cams'][r]; uv, z = core.project(cam, V); m = obj['masks'][r]
        ui = np.clip(np.round(uv).astype(int), 0, [cam['w'] - 1, cam['h'] - 1]); inm = (z > 0) & m[ui[:, 1], ui[:, 0]]; z = z[inm] if inm.sum() >= 10 else z[z > 0]
        if opts.get('depthRangeM'):
            zlo, zhi = np.array(opts['depthRangeM'], float) / S
        else:
            zlo, zhi = np.percentile(z, 2) / opts.get('depthFactor', 1.5), np.percentile(z, 98) * opts.get('depthFactor', 1.5)
        X, uv, st = sweep_ref(ctx, obj, r, opts.get('scale', .5), zlo, zhi, int(opts.get('planes', 480)), normals)
        clouds[r], uvs[r] = X, uv; stats[r] = dict(st, depthRangeM=[round(float(zlo * S), 2), round(float(zhi * S), 2)])
    trees = {r: cKDTree(X) for r, X in clouds.items() if len(X)}; keep = []
    for r, X in clouds.items():
        ok = np.zeros(len(X), bool)
        for o, t in trees.items():
            if o != r and len(X):
                ok |= t.query(X, distance_upper_bound=opts.get('confirmCm', 1.5) / 100 / S)[0] < np.inf
        stats[r]['confirmed'] = int(ok.sum()); keep.append((X[ok], np.full(ok.sum(), r), uvs[r][ok]))
    X = np.vstack([k[0] for k in keep]); src = np.concatenate([k[1] for k in keep]); UV = np.vstack([k[2] for k in keep])
    if len(X) < 200:
        return dict(verdict='inconclusive', reason='too_few_confirmed_points', photos=stats)
    N = local_normals(X); Cm = np.mean([c['C'] for c in ctx['cams']], 0); thr = opts.get('thrCm', 1.2) / 100 / S
    free = np.ones(len(X), bool); planes, members = [], []
    for _ in range(int(opts.get('maxPlanes', 8))):
        idx = np.nonzero(free)[0]
        if len(idx) < opts.get('minPoints', 600):
            break
        n, c, inl = ransac(X[idx], N[idx], thr, rng)
        if inl.sum() < opts.get('minPoints', 600):
            break
        sel = idx[inl]; free[sel] = False; members.append(sel)
        planes.append(plane_row(n, c, X[sel], src[sel], S, up, off, Cm))
    folds = []
    for i in range(len(planes)):
        for j in range(i + 1, len(planes)):
            f = fold(planes[i], planes[j], X[members[i]], X[members[j]], S, up, off)
            if f:
                folds.append(dict(planes=[i, j], **f))
    for p, sel in zip(planes, members):
        pick = rng.choice(len(sel), min(len(sel), 1500), replace=False)
        p['pointsSampleNative'] = np.round(X[sel[pick]], 5).tolist(); p['pixelsSample'] = [[int(r_), round(float(a), 1), round(float(b), 1)] for r_, (a, b) in zip(src[sel[pick]], UV[sel[pick]])]
    return dict(verdict='measured' if planes else 'inconclusive', confirmedPoints=int(len(X)), unassigned=int(free.sum()), photos=stats,
                sweepNormals=[np.round(n, 4).tolist() for n in normals],
                planes=planes, folds=folds, _draw=(X, src, UV, members))


def overlay(ctx, obj, draw, k, width=900):
    X, src, UV, members = draw; cols = [(0, 0, 255), (0, 200, 0), (255, 0, 0), (0, 200, 255), (255, 0, 255), (255, 255, 0), (128, 0, 255), (0, 128, 128)]
    im = cv2.cvtColor(ctx['gray'][k].astype(np.uint8), cv2.COLOR_GRAY2BGR); uv, z = core.project(ctx['cams'][k], X)
    for j, sel in enumerate(members):
        for u, v in uv[sel][z[sel] > 0][::2].astype(int):
            cv2.circle(im, (u, v), 5, cols[j % 8], -1)
        q = np.median(uv[sel], 0).astype(int); cv2.putText(im, str(j), tuple(int(x) for x in q), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 12)
        cv2.putText(im, str(j), tuple(int(x) for x in q), cv2.FONT_HERSHEY_SIMPLEX, 4, cols[j % 8], 5)
    m = obj['masks'].get(k)
    if m is not None and m.any():
        ys, xs = np.nonzero(m); im = im[max(0, ys.min() - 150):ys.max() + 150, max(0, xs.min() - 150):xs.max() + 150]
    im = cv2.resize(im, (width, int(im.shape[0] * width / im.shape[1])), interpolation=cv2.INTER_AREA)
    return cv2.imencode('.jpg', im, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()


def run(ctx, opts):
    """opts: only ([entity ids]; default: every object with masks in two or more photos), planes (480), depthFactor (1.5),
    normals ([world normals] as sweep directions instead of the models' planar parts; modelNormals False: fronto-parallel only),
    depthRangeM ([lo, hi] m, instead of the model range), scale (.5), thrCm (1.2), confirmCm (1.5), maxPlanes (8), minPoints (600),
    overlays (True: plane colours on each masked photo), cloud (False: every confirmed point as <id8>-cloud.npz)."""
    out, files = {}, {}
    for i, o in enumerate(ctx['objects']):
        if opts.get('only') and o['id'] not in opts['only']:
            continue
        row = check_object(ctx, o, opts, np.random.default_rng(i)); draw = row.pop('_draw', None)
        out[o['id']] = dict(label=o['label'], **row)
        if draw is not None and opts.get('overlays', True):
            for k in sorted(o['masks']):
                files[f'{o["id"][:8]}-photo{k + 1}.jpg'] = overlay(ctx, o, draw, k)
        if draw is not None and opts.get('cloud'):  # every confirmed point: native world, reference photo, its pixel there, plane (-1 = none)
            X, src, UV, members = draw; lab = np.full(len(X), -1, np.int16)
            for j, sel in enumerate(members):
                lab[sel] = j
            buf = io.BytesIO(); np.savez_compressed(buf, X=X.astype(np.float32), src=src.astype(np.int8), uv=UV.astype(np.float32), plane=lab)
            files[f'{o["id"][:8]}-cloud.npz'] = buf.getvalue()
    return dict(method='plane facets: fronto-parallel plane sweep (inverse depth) per masked reference photo, 13 px NCC, minimum over the other '
                       'photos, distinct peak, sub-step refinement; cross-reference confirmation; sequential RANSAC with local-normal gate; folds '
                       'between planes whose supports meet along their intersection line', objects=out, **({'files': files} if files else {}))


def _synthetic(fold_deg=90.):
    """Three cameras 0.4 apart looking at a textured V (two vertical boards meeting at x = 0, z = 3, interior angle fold_deg)."""
    rng = np.random.default_rng(0); tex = cv2.GaussianBlur((rng.random((700, 700)) * 255).astype(np.float32), (0, 0), 2)
    k = 1 / math.tan(math.radians(fold_deg / 2))  # z = 3 + k |x|
    W, H = 640, 480; K = np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1.]]); cams, gray, masks = [], [], {}
    ys, xs = np.mgrid[0:H, 0:W]; rays = (np.linalg.inv(K) @ np.vstack([xs.ravel(), ys.ravel(), np.ones(xs.size)])).T
    for j, x in enumerate((-.4, 0., .4)):
        M = np.eye(4); M[:3, 3] = [x, 0, 0]; cams.append(core.camera(dict(cameraToWorld=M.tolist(), K=K.tolist(), width=W, height=H, imageId=str(j))))
        best = np.full(len(rays), np.inf); Q = np.zeros((len(rays), 3))
        for sgn in (-1, 1):  # half-plane z = 3 + k * sgn * x for sgn * x >= 0
            s = (3 + k * sgn * x) / (rays[:, 2] - k * sgn * rays[:, 0]); P = np.c_[x + s * rays[:, 0], s * rays[:, 1], s * rays[:, 2]]
            hit = (s > 0) & (sgn * P[:, 0] >= 0) & (s < best); best[hit] = s[hit]; Q[hit] = P[hit]
        inside = np.isfinite(best) & (np.abs(Q[:, 0]) < .8) & (np.abs(Q[:, 1]) < .6)
        g = core.bilinear(tex, np.c_[(Q[:, 0] + Q[:, 2] * .3 + 1.5) * 200, (Q[:, 1] + 1.5) * 200]) * inside + 90 * ~inside
        gray.append(g.reshape(H, W).astype(np.float32)); masks[j] = inside.reshape(H, W)
    V = np.array([[-.8, -.6, 3 + .8 * k], [0, -.6, 3], [0, .6, 3], [-.8, .6, 3 + .8 * k], [.8, -.6, 3 + .8 * k], [.8, .6, 3 + .8 * k]])
    F = np.array([[0, 1, 2], [0, 2, 3], [1, 4, 5], [1, 5, 2]])
    obj = dict(id='v', label='v', kind='primitive', mesh=(V, F), masks=masks); obj['original'] = obj['mesh']
    ctx = dict(cams=cams, gray=gray, images=gray, objects=[obj], S=1., floor=(np.array([0, -1., 0]), 1.))
    return run(ctx, {'overlays': False, 'minPoints': 300, 'planes': 240})['objects']['v']


def _check():
    r = _synthetic(90.)
    big = sorted(r['planes'], key=lambda p: -p['points'])[:2]
    assert len(big) == 2 and all(abs(p['inclinationDeg'] - 90) < 2 and p['rmsCm'] < 1.5 for p in big), [(p['inclinationDeg'], p['rmsCm'], p['points']) for p in r['planes']]
    f = [x for x in r['folds'] if set(x['planes']) == {r['planes'].index(big[0]), r['planes'].index(big[1])}]
    assert f and abs(f[0]['interiorDeg'] - 90) < 2 and .9 < f[0]['lengthM'] < 1.25, r['folds']
    print('plane facets self-test passed:', len(r['planes']), 'planes, fold', f[0]['interiorDeg'], 'deg, length', f[0]['lengthM'], 'm, rms', [p['rmsCm'] for p in big])


if __name__ == '__main__':
    _check()
