"""Method B clearance: bottom ends of light-curtain housings and bottom edges of fence panels from the report's own cameras.

Generic (no per-type template): an object's report mask, straight image structure and the floor normal as the vertical.
  post  (housing): axis = line through the vertical vanishing point fitted to the mask's row centres (all photos with a
        mask); 3D axis = vertical line of the back-projected axis planes. Bottom end per photo: along lines through the
        vanishing point across the central 80 % of the mask width, the colour step that leaves the mask's own colour near
        the mask boundary, located sub-pixel by a blurred-step fit in the channel with the largest step. Lines whose end
        touches another object's mask are occluded. Bottom point = the BOTTOM_PCT-percentile (lowest) edge point.
        3D: DLT of the bottom points of all photos with a usable bottom (>= 2), else that photo's ray closest to the axis.
  panel (fence): guide = RANSAC line through the mask's per-column bottom (columns whose bottom touches another mask are
        dropped); LSD segments within ANGLE_DEG and +-BAND_PX of it, grouped by offset; the group nearest the guide with
        coverage >= MIN_COVER is the bottom edge (all groups reported). 3D: two-view plane intersection when conditioned,
        else the horizontal 3D line fitted to both image lines, else the image line on the vertical plane fitted to the
        Pi3X points just above the line (one photo).
Floors: R = report floor; P = plane RANSAC on the Pi3X point maps near the object (all frames); W = two-view plane sweep
of the floor tiles near the object in the report cameras. Heights in metres on each.

Parameters below were fixed on 2026-10-05 before any height was computed and are not tuned to field values.
"""
from __future__ import annotations

import io
import math

import cv2
import numpy as np
from scipy.optimize import curve_fit, least_squares
from scipy.special import ndtr

P = dict(
    AXIS_ROWS=.5,          # axis fitted to row centres of the lower half of the mask
    WIDTH_FRAC=.8,         # bottom sampled across the central 80 % of the mask width, 1 px apart
    T_UP=120, T_DN=80,     # sampling range along each line (px above / below the mask end)
    SEARCH=(-30, 30),      # colour-step search window around the mask end along each line (px)
    COLOUR_K=3., COLOUR_MIN=30., COLOUR_MAX=60.,  # reported only: the object's dominant (k-means) colour spread
    FIT_HALF=8,            # blurred-step fit half window (px)
    MIN_AMP=20, SIGMA=(.3, 4.), MAX_SHIFT=4,  # accepted step: amplitude (8-bit), blur, fit vs run end
    OCCL_BELOW=25, OCCL_ABOVE=10,  # a nearer object within this range (px) of a line's end = occluded line
    MARGIN=.05,            # 'nearer' = Pi3X depth more than 5 cm-native in front of the object's own depth
    MIN_ACCEPT=.6, MAX_OCCL=.2, MAX_LATERAL=.5, LATERAL_WARN=.15,  # bottom usable: >= 60 % lines accepted, <= 20 % occluded, nearer object along < 50 % of a side
    EDGE_TOL=2.5, MIN_EDGE_SUPPORT=.2,  # bottom = lowest straight edge (sequential RANSAC, >= 20 % of lines); point = its axis crossing
    GUIDE_STEP=4, GUIDE_MIN=40, GUIDE_TOL=12., GUIDE_OCCL=20,  # fence guide line from mask bottoms
    BAND_PX=50., ANGLE_DEG=6., LSD_MIN=25., GROUP_GAP=3., MIN_COVER=.2,
    MIN_DIHEDRAL_DEG=2., MAX_CM_PER_PX=1.,
    PI3X_BAND=(-80, -3), PI3X_TOL=.015,          # Pi3X points 3-80 px above the bottom line; vertical-plane RANSAC tol (native)
    FLOOR_R=.5, FLOOR_H=.10, FLOOR_TOL=.006, FLOOR_MAX_TILT=10.,  # local floor: radius, height window, RANSAC tol (native)
    SWEEP_RANGE=.04, SWEEP_STEP=.001, TILE=160, TILE_MIN=.5, SWEEP_NCC=.4, SWEEP_CONTRAST=.2,
)


# ---------------------------------------------------------------- geometry
def proj_matrix(cam):
    return cam['K'] @ np.c_[cam['R'], cam['t']]


def project(cam, X):
    x = proj_matrix(cam) @ np.r_[X, 1.]
    return x[:2] / x[2]


def ray(cam, uv):
    d = cam['R'].T @ np.linalg.inv(cam['K']) @ np.array([uv[0], uv[1], 1.])
    return cam['C'], d / np.linalg.norm(d)


def back_plane(cam, line):
    pi = proj_matrix(cam).T @ line
    return pi / np.linalg.norm(pi[:3])


def vanishing(cam, up):
    return cam['K'] @ cam['R'] @ up


def down_dir(vp, p):
    """Unit image direction at p of the line through the vertical vanishing point, pointing down the image."""
    d = vp[:2] / vp[2] - p if abs(vp[2]) > 1e-9 else vp[:2].copy()
    d = d / np.linalg.norm(d)
    return d if d[1] > 0 else -d


def dlt(cams, uvs):
    A = []
    for cam, (u, v) in zip(cams, uvs):
        M = proj_matrix(cam); A += [u * M[2] - M[0], v * M[2] - M[1]]
    X = np.linalg.svd(np.array(A))[2][-1]
    return X[:3] / X[3]


def vertical_line(planes, up, prior=None):
    """Vertical 3D line (direction up) on every plane; with one plane, the line in it nearest the prior point."""
    A = [p[:3] for p in planes]; b = [-p[3] for p in planes]
    if len(planes) == 1:
        h = prior - (prior @ up) * up  # move the prior within the plane perpendicular to up
        w = np.cross(planes[0][:3], up); w /= np.linalg.norm(w)
        A.append(w); b.append(w @ h)
    A.append(up); b.append(0.)
    X, *_ = np.linalg.lstsq(np.array(A), np.array(b), rcond=None)
    return X


def closest_on_line(C, r, X0, D):
    """Point of line X0 + s D closest to the ray C + a r, and the gap between the two lines."""
    w = C - X0; a, b, c = r @ r, r @ D, D @ D; d, e = r @ w, D @ w
    den = a * c - b * b
    s = (a * e - b * d) / den; t = (b * e - c * d) / den
    return X0 + s * D, float(np.linalg.norm(X0 + s * D - (C + t * r))), s


def plane_line(p1, p2):
    D = np.cross(p1[:3], p2[:3]); D /= np.linalg.norm(D)
    X0, *_ = np.linalg.lstsq(np.array([p1[:3], p2[:3], D]), np.array([-p1[3], -p2[3], 0.]), rcond=None)
    return X0, D


def floor_basis(n, d):
    e1 = np.cross(n, [1., 0, 0]) if abs(n[0]) < .9 else np.cross(n, [0, 1., 0]); e1 /= np.linalg.norm(e1)
    return -d * n, e1, np.cross(n, e1)


# ---------------------------------------------------------------- image measurements
def sample(img, pts):
    pts = np.asarray(pts, np.float32).reshape(-1, 1, 2)
    out = cv2.remap(img, pts[..., 0], pts[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return out.reshape(len(pts), -1).astype(float)


def inside(mask, pts):
    ij = np.round(pts).astype(int)
    ok = (ij[:, 0] >= 0) & (ij[:, 1] >= 0) & (ij[:, 0] < mask.shape[1]) & (ij[:, 1] < mask.shape[0])
    out = np.zeros(len(pts), bool); out[ok] = mask[ij[ok, 1], ij[ok, 0]]
    return out


def step(x, a, b, x0, s):
    return a + b * ndtr((x - x0) / s)


def fit_step(t, p, guess):
    try:
        (a, b, x0, s), _ = curve_fit(step, t, p, p0=[p[:3].mean(), p[-3:].mean() - p[:3].mean(), guess, 1.2],
                                     bounds=([-50, -400, t[0], .2], [350, 400, t[-1], 8]), maxfev=4000)
        return float(x0), float(s), float(b)
    except (RuntimeError, ValueError):
        return None


def fit_axis(mask, vp, rows=None):
    """Image line through vp fitted to the mask's row centres (lower AXIS_ROWS of the mask)."""
    ys, xs = np.nonzero(mask); top, bot = ys.min(), ys.max()
    lo = bot - P['AXIS_ROWS'] * (bot - top)
    cs = []
    for r in range(int(lo), int(bot) - 4):
        c = np.nonzero(mask[r])[0]
        if len(c) >= 4:
            cs.append((c.mean(), r))
    cs = np.array(cs, float); yref = float(bot)
    A = np.cross([1., 0, 0], vp); B = np.cross([0., yref, 1.], vp)
    keep = np.ones(len(cs), bool); hc = np.c_[cs, np.ones(len(cs))]
    for _ in range(3):
        a, b = hc[keep] @ A, hc[keep] @ B; x = -(a @ b) / (a @ a)
        l = x * A + B; res = hc @ l / np.linalg.norm(l[:2])
        mad = np.median(np.abs(res[keep])) + .5; keep = np.abs(res) < 3 * mad
    l = l / np.linalg.norm(l[:2])
    return l, np.array([x, yref]), dict(rows=int(keep.sum()), residualPx=float(np.median(np.abs(res[keep]))))


def ransac_line(pts, tol, rng, iters=300):
    best = None
    for _ in range(iters):
        i, j = rng.choice(len(pts), 2, replace=False)
        l = np.cross(np.r_[pts[i], 1], np.r_[pts[j], 1]); L = np.linalg.norm(l[:2])
        if L < 1e-9:
            continue
        inl = np.abs(np.c_[pts, np.ones(len(pts))] @ (l / L)) < tol
        if best is None or inl.sum() > best.sum():
            best = inl
    return tls(pts[best], np.ones(best.sum())), best


def housing_bottom(rgb, mask, occl, vp):
    """Sub-pixel bottom end of a post-like object along lines through the vertical vanishing point.

    Per line: walk down from inside the mask while the colour stays the mask's own (gaps <= GAP_PX allowed: screws,
    washers), the run end is refined by a blurred-step fit. The end points are fitted by a RANSAC line (the dominant
    bottom edge); bottom point = where the axis crosses it. Lines whose end has a nearer object below are occluded;
    a nearer object along the mask's sides = lateral occlusion (the axis and the bottom may be partial)."""
    line, p_axis, axis_diag = fit_axis(mask, vp)
    ys, _ = np.nonzero(mask); top, bot = ys.min(), ys.max(); H = bot - top
    counts = [mask[r].sum() for r in range(int(bot - .25 * H), int(bot) - 10)]
    W = float(np.median([c for c in counts if c > 0]))
    rr, cc = np.nonzero(mask[int(bot - .2 * H):int(bot) - 10]); er = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    sel = er[rr + int(bot - .2 * H), cc]
    ref_px = rgb[rr[sel] + int(bot - .2 * H), cc[sel]].astype(np.float32)
    _, lab, centres = cv2.kmeans(ref_px, 3, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, .5), 3, cv2.KMEANS_PP_CENTERS)
    big = int(np.bincount(lab.ravel()).argmax()); ref = centres[big].astype(float)  # the object's dominant colour
    spread = float(np.median(np.linalg.norm(ref_px[lab.ravel() == big] - ref, axis=1)))
    tau = float(np.clip(P['COLOUR_K'] * spread, P['COLOUR_MIN'], P['COLOUR_MAX']))
    d0 = down_dir(vp, p_axis); e = np.array([-d0[1], d0[0]])
    ts = np.arange(-P['T_UP'], P['T_DN'], .5)
    rows = []
    for w in np.arange(-P['WIDTH_FRAC'] / 2 * W, P['WIDTH_FRAC'] / 2 * W + 1e-9, 1.):
        p0 = p_axis + w * e; dw = down_dir(vp, p0); pts = p0 + ts[:, None] * dw
        ins = inside(mask, pts)
        if not ins[:80].any():
            continue
        kb = int(np.nonzero(ins)[0].max())
        prof = sample(rgb, pts); dist = np.linalg.norm(prof - ref, axis=1)
        row = dict(w=float(w), tMask=float(ts[kb]), accepted=False, occluded=False)
        # strongest colour step near the mask end whose upper side is nearer the object's dominant colour than its lower side
        best = None
        for k in np.nonzero((ts >= ts[kb] + P['SEARCH'][0]) & (ts <= ts[kb] + P['SEARCH'][1]))[0]:
            if k < 10 or k + 11 > len(ts):
                continue
            up, dn = prof[k - 10:k].mean(0), prof[k + 1:k + 11].mean(0); sc = np.linalg.norm(dn - up)
            if dist[k - 10:k].mean() < dist[k + 1:k + 11].mean() and (best is None or sc > best[0]):
                best = (sc, k)
        if best is None:
            rows.append(row); continue
        last = best[1]
        up, dn = prof[last - 10:last - 1].mean(0), prof[last + 2:last + 11].mean(0); ch = int(np.argmax(np.abs(dn - up)))
        win = np.abs(ts - ts[last]) <= P['FIT_HALF']
        f = fit_step(ts[win], prof[win, ch], ts[last])
        if f and abs(f[2]) >= P['MIN_AMP'] and P['SIGMA'][0] <= f[1] <= P['SIGMA'][1] and abs(f[0] - ts[last]) <= P['MAX_SHIFT']:
            t0 = f[0]; q = p0 + t0 * dw
            near = (ts >= t0 - P['OCCL_ABOVE']) & (ts <= t0 + P['OCCL_BELOW'])
            row.update(accepted=True, t0=t0, uv=q.tolist(), sigma=f[1], amp=f[2], channel='RGB'[ch],
                       occluded=bool(inside(occl, pts[near]).any()))
        rows.append(row)
    acc = [r for r in rows if r['accepted']]; vis = [r for r in acc if not r['occluded']]
    lat = []
    for side in (-1, 1):
        hits = n = 0
        for r in range(int(bot - .5 * H), int(bot) - 5, 4):
            c = np.nonzero(mask[r])[0]
            if len(c) < 4:
                continue
            x = c.max() if side > 0 else c.min(); n += 1
            hits += bool(occl[r, max(x + side * 3, 0):x + side * 9 + 1].any() if side > 0 else occl[r, max(x - 9, 0):max(x - 2, 0)].any())
        lat.append(hits / max(n, 1))
    out = dict(axis=line.tolist(), axisPoint=p_axis.tolist(), axisFit=axis_diag, widthPx=W, colourRef=ref.tolist(), colourTau=tau,
               lines=len(rows), accepted=len(acc), occluded=sum(r['occluded'] for r in acc), lateral=lat, maskBottomPx=float(bot))
    if len(vis) >= 5:
        V = np.array([r['uv'] for r in vis]); rest = np.ones(len(V), bool); found = []; rng = np.random.default_rng(0)
        for _ in range(4):  # sequential RANSAC: straight bottom-edge segments with >= MIN_EDGE_SUPPORT of the visible lines
            if rest.sum() < 5:
                break
            el, inl_r = ransac_line(V[rest], P['EDGE_TOL'], rng); inl = np.zeros(len(V), bool); inl[np.nonzero(rest)[0][inl_r]] = True
            if inl.sum() < max(5, P['MIN_EDGE_SUPPORT'] * len(V)):
                break
            found.append((el, inl)); rest &= ~inl
        if found:
            def point(el, inl):  # axis crossing when the axis meets the segment, else the segment end nearest the axis
                X = np.cross(el, line); X = X[:2] / X[2]; seg = V[inl]; dirn = np.array([el[1], -el[0]])
                s, s0 = seg @ dirn, X @ dirn
                if s.min() - 10 <= s0 <= s.max() + 10:
                    return X, True
                return seg[np.argmin(np.abs(s - s0))], False
            segs = [dict(line=el.tolist(), inliers=int(inl.sum()), medianV=float(np.median(V[inl][:, 1])), point=point(el, inl)[0].tolist(),
                         onAxis=point(el, inl)[1], residualPx=float(np.median(np.abs(np.c_[V[inl], np.ones(inl.sum())] @ el)))) for el, inl in found]
            low = max(range(len(segs)), key=lambda i: segs[i]['medianV']); dom = max(range(len(segs)), key=lambda i: segs[i]['inliers'])
            inl = found[low][1]
            out.update(edgeSegments=segs, edgeLine=segs[low]['line'], edgeInliers=segs[low]['inliers'], bottom=segs[low]['point'],
                       bottomDominant=segs[dom]['point'], bottomLowest=V[inl][np.argmax(V[inl][:, 1])].tolist(),
                       bottomMedian=V[np.argsort(V[:, 1])[len(V) // 2]].tolist(), edgeResidualPx=segs[low]['residualPx'],
                       spreadPx=float(np.percentile(V[:, 1], 90) - np.percentile(V[:, 1], 10)),
                       sigmaPx=float(np.median([r['sigma'] for r in vis])), channels=''.join(sorted({r['channel'] for r in vis})),
                       edge=V.tolist(), edgeIsInlier=inl.tolist())
    ok_lines = len(rows) > 0 and len(acc) >= P['MIN_ACCEPT'] * len(rows)
    out['lateralWarning'] = bool(max(lat) > P['LATERAL_WARN'])  # coarse masks often touch neighbours: warning only
    if max(lat) >= P['MAX_LATERAL']:
        out['status'] = 'lateral'  # a nearer object along half a side: the visible end may be another face
    elif out['occluded'] > P['MAX_OCCL'] * max(len(acc), 1):
        out['status'] = 'occluded'
    elif not ok_lines or 'edgeLine' not in out:
        out['status'] = 'no_step'
    else:
        out['status'] = 'ok'
    return out


def guide_line(mask, occl):
    xs = np.nonzero(mask.any(0))[0]; pts = []
    for c in range(int(xs.min()), int(xs.max()) + 1, P['GUIDE_STEP']):
        r = np.nonzero(mask[:, c])[0]
        if len(r) < P['GUIDE_MIN']:
            continue
        b = int(r.max())
        if occl[max(b - 5, 0):b + P['GUIDE_OCCL'], c].any():
            continue
        pts.append((c, b))
    pts = np.array(pts, float); rng = np.random.default_rng(0); best = None
    for _ in range(800):
        i, j = rng.choice(len(pts), 2, replace=False)
        if pts[i, 0] == pts[j, 0]:
            continue
        a = (pts[j, 1] - pts[i, 1]) / (pts[j, 0] - pts[i, 0]); b = pts[i, 1] - a * pts[i, 0]
        inl = np.abs(pts[:, 1] - (a * pts[:, 0] + b)) <= P['GUIDE_TOL']
        if best is None or inl.sum() > best.sum():
            best = inl
    a, b = np.polyfit(pts[best, 0], pts[best, 1], 1)
    u0, u1 = np.percentile(pts[best, 0], [2, 98])
    return float(a), float(b), float(u0), float(u1), dict(columns=len(pts), inliers=int(best.sum()))


def tls(points, weights):
    c = (points * weights[:, None]).sum(0) / weights.sum()
    d = np.linalg.svd((points - c) * np.sqrt(weights)[:, None], full_matrices=False)[2][0]
    l = np.cross(np.r_[c, 1], np.r_[c + d, 1]); l /= np.linalg.norm(l[:2])
    return l if l[1] > 0 else -l  # l . (u, v, 1) > 0 below the line


def fence_bottom(gray, mask, occl):
    """Bottom edge of a panel-like object: straight image lines (LSD, grouped into collinear lines) within +-BAND_PX
    and ANGLE_DEG of the RANSAC line through the mask's per-column bottom; the line nearest the guide wins."""
    a, b, u0, u1, gdiag = guide_line(mask, occl)
    th = math.atan(a); g = np.array([math.cos(th), math.sin(th)]); nrm = np.array([-g[1], g[0]])  # nrm points down
    v0, v1 = a * u0 + b, a * u1 + b; O = np.array([u0, v0]); span = float(np.hypot(u1 - u0, v1 - v0))
    x0, x1 = int(max(u0 - 30, 0)), int(min(u1 + 30, gray.shape[1]))
    y0, y1 = int(max(min(v0, v1) - P['BAND_PX'] - 40, 0)), int(min(max(v0, v1) + P['BAND_PX'] + 40, gray.shape[0]))
    crop = np.clip(gray[y0:y1, x0:x1], 0, 255).astype(np.uint8)
    segs = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(crop)[0]
    segs = (segs.reshape(-1, 4) + [x0, y0, x0, y0]) if segs is not None else np.zeros((0, 4))
    d = segs[:, 2:] - segs[:, :2]; L = np.hypot(d[:, 0], d[:, 1])
    ang = np.degrees(np.abs(np.arctan2(d[:, 0] * g[1] - d[:, 1] * g[0], d[:, 0] * g[0] + d[:, 1] * g[1]))); ang = np.minimum(ang, 180 - ang)
    mid = (segs[:, :2] + segs[:, 2:]) / 2; off = (mid - O) @ nrm; s_mid = (mid - O) @ g
    keep = (L >= P['LSD_MIN']) & (ang <= P['ANGLE_DEG']) & (np.abs(off) <= P['BAND_PX']) & (s_mid >= 0) & (s_mid <= span)
    S, Lk = segs[keep], L[keep]
    # greedy collinear grouping, longest segment first: members within GROUP_GAP px of the group's line (both ends)
    free = list(np.argsort(-Lk)); groups = []
    while free:
        seed = free.pop(0); members = [seed]; l = tls(np.array([S[seed, :2], S[seed, 2:]]), np.ones(2))
        for _ in range(2):
            near = [i for i in free if max(abs(l @ np.r_[S[i, :2], 1]), abs(l @ np.r_[S[i, 2:], 1])) <= P['GROUP_GAP']]
            if not near:
                break
            members += near; free = [i for i in free if i not in near]
            pts = np.concatenate([np.linspace(S[i, :2], S[i, 2:], max(int(Lk[i] // 4), 2)) for i in members])
            l = tls(pts, np.ones(len(pts)))
        groups.append((members, l))
    cands = []
    for members, l in groups:
        iv = sorted(tuple(sorted(((S[i, :2] - O) @ g, (S[i, 2:] - O) @ g))) for i in members)
        cov, (lo, hi) = 0., iv[0]
        for s0, s1 in iv[1:]:
            if s0 > hi:
                cov += hi - lo; lo, hi = s0, s1
            else:
                hi = max(hi, s1)
        cov += hi - lo
        if cov / span < P['MIN_COVER']:
            continue
        ext = [max(iv[0][0], 0.), min(max(x[1] for x in iv), span)]
        foot = lambda s: (lambda p: p - (l @ np.r_[p, 1]) * l[:2])(O + s * g)  # point of the line at guide position s
        ends = [foot(s) for s in ext]; midp = foot((ext[0] + ext[1]) / 2)
        along = [foot(s) for s in np.linspace(*ext, 20)]
        above, below = sample(gray, [p - 6 * l[:2] for p in along]), sample(gray, [p + 6 * l[:2] for p in along])
        cands.append(dict(offsetPx=float((midp - O) @ nrm), coverage=float(min(cov / span, 1.)), segments=len(members), line=l.tolist(),
                          ends=[x.tolist() for x in ends], aboveMinusBelow=float(above.mean() - below.mean())))
    cands.sort(key=lambda c: c['offsetPx'])
    out = dict(guide=[a, b, u0, u1], guideFit=gdiag, segments=int(keep.sum()), status='no_line', candidates=cands)
    if cands:
        k = int(np.argmin([abs(c['offsetPx']) for c in cands]))
        out.update(status='ok', chosen=k, line=cands[k]['line'], ends=cands[k]['ends'])
    return out


def occluders(ctx, k, obj, box=None):
    """Pixels of other objects' masks that are nearer to camera k than the object itself (Pi3X depth, MARGIN native)."""
    from scipy.spatial import cKDTree
    cam = ctx['cams'][k]; dm = depth_map(ctx, k)
    m = np.zeros((cam['h'], cam['w']), bool)
    for o in ctx['objects']:
        if o['id'] != obj['id'] and k in o['masks']:
            m |= o['masks'][k]
    if box is not None:
        x0, y0, x1, y1 = [int(v) for v in box]; keep = np.zeros_like(m); keep[max(y0, 0):y1, max(x0, 0):x1] = True; m &= keep
    rr, cc = np.nonzero(m[::3, ::3]); rr, cc = rr * 3, cc * 3  # ponytail: 3 px grid, filled as 3x3 blocks
    if not len(rr):
        return m
    uv, z = dm
    er = cv2.erode(obj['masks'][k].astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    own = inside(er, uv); tree = cKDTree(uv[own]); zo = z[own]
    zt = np.full(len(rr), np.nan); q = np.c_[cc, rr].astype(float)
    for radius in (60., 150., 400.):
        todo = np.isnan(zt)
        if not todo.any():
            break
        nb = tree.query_ball_point(q[todo], radius)
        zt[todo] = [np.median(zo[i]) if len(i) else np.nan for i in nb]
    za = z[cKDTree(uv).query(q)[1]]
    front = za < zt - P['MARGIN']
    out = np.zeros_like(m)
    for dy in range(3):
        for dx in range(3):
            out[np.minimum(rr[front] + dy, m.shape[0] - 1), np.minimum(cc[front] + dx, m.shape[1] - 1)] = True
    return out & m


def depth_map(ctx, k):
    cache = ctx.setdefault('_depth', {})
    if k not in cache:
        fr = ctx['pi3x'][k]; cam = ctx['cams'][k]; X = fr['pts3d'].reshape(-1, 3).astype(float)
        ok = (fr['valid'] & fr['content']).ravel(); X = X[ok]
        x = (cam['K'] @ (cam['R'] @ X.T + cam['t'][:, None])).T
        cache[k] = (x[:, :2] / x[:, 2:], x[:, 2])
    return cache[k]


# ---------------------------------------------------------------- 3D
def height(X, floor):
    return float(floor[0] @ X + floor[1])


def ransac_vertical_plane(X, up, tol, rng, iters=400):
    best = None
    for _ in range(iters):
        i, j = rng.choice(len(X), 2, replace=False)
        m = np.cross(X[j] - X[i], up); L = np.linalg.norm(m)
        if L < 1e-6:
            continue
        m /= L; inl = np.abs((X - X[i]) @ m) < tol
        if best is None or inl.sum() > best.sum():
            best = inl
    Q = X[best]; Qh = Q - np.outer(Q @ up, up); c = Qh.mean(0)
    u, s, vt = np.linalg.svd(Qh - c, full_matrices=False)
    dirn = vt[0] - (vt[0] @ up) * up; dirn /= np.linalg.norm(dirn); m = np.cross(dirn, up)
    return np.r_[m, -m @ c], int(best.sum()), len(X)


def ransac_plane(X, n_ref, tol, rng, max_tilt_deg, iters=500):
    best = None; cmax = math.cos(math.radians(max_tilt_deg))
    for _ in range(iters):
        a, b, c = X[rng.choice(len(X), 3, replace=False)]
        m = np.cross(b - a, c - a); L = np.linalg.norm(m)
        if L < 1e-9:
            continue
        m /= L
        if abs(m @ n_ref) < cmax:
            continue
        inl = np.abs((X - a) @ m) < tol
        if best is None or inl.sum() > best.sum():
            best = inl
    Q = X[best]; c = Q.mean(0); m = np.linalg.svd(Q - c, full_matrices=False)[2][2]
    m = m if m @ n_ref > 0 else -m
    return (m, float(-m @ c)), int(best.sum()), len(X)


def line_heights(X0, D, s_range, floor, n=21):
    s = np.linspace(*s_range, n)
    h = np.array([height(X0 + t * D, floor) for t in s])
    return dict(median=float(np.median(h)), ends=[float(h[0]), float(h[-1])], slopeDeg=float(math.degrees(math.asin(np.clip(D @ floor[0], -1, 1)))),
                lengthNative=float(abs(s_range[1] - s_range[0]))), X0 + np.median(s) * D


def extent_on_line(cam, ends, X0, D):
    ss = [closest_on_line(*ray(cam, e), X0, D)[2] for e in ends]
    return sorted(ss)


def horizontal_fit(cams, lines, ends_list, floor, init):
    """Horizontal 3D line (height h, heading phi, offset rho) best fitting the image lines (px residuals at each photo's ends)."""
    O, e1, e2 = floor_basis(*floor); n = floor[0]

    def build(p):
        h, phi, rho = p; dirn = math.cos(phi) * e1 + math.sin(phi) * e2; nn = -math.sin(phi) * e1 + math.cos(phi) * e2
        return O + h * n + rho * nn, dirn

    def res(p):
        X0, D = build(p); r = []
        for cam, l, ends in zip(cams, lines, ends_list):
            for e in ends:
                Xc = closest_on_line(*ray(cam, e), X0, D)[0]; x = project(cam, Xc)
                r.append(l @ np.r_[x, 1.])
        return np.array(r)
    X0, D = init; Dh = D - (D @ n) * n; Dh /= np.linalg.norm(Dh)
    phi = math.atan2(Dh @ e2, Dh @ e1); nn = -math.sin(phi) * e1 + math.cos(phi) * e2
    p0 = [height(X0, floor), phi, (X0 - O) @ nn]
    sol = least_squares(res, p0, x_scale=[.1, .1, .1])
    return build(sol.x), float(np.sqrt(np.mean(sol.fun ** 2)))


def plane_intersection(pi, Q):
    return plane_line(pi, Q)


# ---------------------------------------------------------------- floors
def object_masks_union(ctx, k, skip=None):
    m = np.zeros((ctx['cams'][k]['h'], ctx['cams'][k]['w']), bool)
    for o in ctx['objects']:
        if o['id'] != skip and k in o['masks']:
            m |= o['masks'][k]
    return m


def pi3x_floor(ctx, pi3x, anchors, rng):
    """Local floor plane from the Pi3X point maps of every frame (and per frame) near the anchor points."""
    n, d = ctx['floor']; feet = np.array([a - height(a, ctx['floor']) * n for a in anchors])
    allX, per = [], {}
    for k, fr in pi3x.items():
        X = fr['pts3d'].reshape(-1, 3).astype(float); ok = (fr['valid'] & fr['content']).ravel() & (fr['conf'].ravel() >= np.median(fr['conf']))
        h = X @ n + d; ok &= np.abs(h) <= P['FLOOR_H']
        Xh = X - np.outer(h, n)
        dist = np.min(np.linalg.norm(Xh[:, None, :] - feet[None], axis=2), axis=1) if len(feet) < 40 else None
        ok &= dist <= P['FLOOR_R']
        idx = np.nonzero(ok)[0]
        if len(idx):
            cam = ctx['cams'][k]; x = (cam['K'] @ (cam['R'] @ X[idx].T + cam['t'][:, None])).T; uv = x[:, :2] / x[:, 2:]
            occ = inside(object_masks_union(ctx, k), uv); idx = idx[~occ]
        per[k] = X[idx]; allX.append(X[idx])
    out = {}
    for name, X in [('all', np.concatenate(allX))] + [(f'frame{k + 1}', X) for k, X in per.items()]:
        if len(X) < 200:
            out[name] = dict(status='too_few', points=int(len(X))); continue
        (m, e), ninl, tot = ransac_plane(X, n, P['FLOOR_TOL'], rng, P['FLOOR_MAX_TILT'])
        offs = [float(m @ a + e) for a in anchors]  # anchor height above the local plane
        hR = [height(a, ctx['floor']) for a in anchors]
        out[name] = dict(status='ok', points=int(tot), inliers=ninl, tiltDeg=float(math.degrees(math.acos(min(1., m @ n)))),
                         reportMinusLocalNative=float(np.median(np.array(hR) - np.array(offs))), plane=[*m.tolist(), e])
    return out


def sweep_floor(ctx, anchors):
    """Two-view plane sweep of floor tiles near the anchors: the height (native, + = above the report floor) of the plane
    that best maps each reference tile into the other photos (NCC), median over accepted tiles."""
    n, d = ctx['floor']; feet = np.array([a - height(a, ctx['floor']) * n for a in anchors]); cams = ctx['cams']
    images = ctx['images']; best_ref = None
    for k, cam in enumerate(cams):
        ys, xs = np.mgrid[0:cam['h']:8, 0:cam['w']:8]; uv = np.c_[xs.ravel(), ys.ravel()].astype(float)
        Xf = floor_points(cam, uv, ctx['floor'], 0.)
        near = np.min(np.linalg.norm(Xf[:, None] - feet[None], axis=2), axis=1) <= P['FLOOR_R']
        near &= ~inside(object_masks_union(ctx, k), uv)
        if best_ref is None or near.sum() > best_ref[1]:
            best_ref = (k, int(near.sum()))
    r = best_ref[0]; cam = cams[r]; T = P['TILE']; offs = np.arange(-P['SWEEP_RANGE'], P['SWEEP_RANGE'] + 1e-9, P['SWEEP_STEP'])
    occ = {k: object_masks_union(ctx, k) for k in range(len(cams))}
    tiles = []
    for y in range(0, cam['h'] - T + 1, T):
        for x in range(0, cam['w'] - T + 1, T):
            ys, xs = np.mgrid[y:y + T:2, x:x + T:2]; uv = np.c_[xs.ravel(), ys.ravel()].astype(float)
            Xf = floor_points(cam, uv, ctx['floor'], 0.)
            ok = np.min(np.linalg.norm(Xf[:, None] - feet[None], axis=2), axis=1) <= P['FLOOR_R']
            ok &= ~inside(occ[r], uv)
            if ok.mean() < P['TILE_MIN']:
                continue
            uv = uv[ok]; ref = sample(images[r], uv)[:, 0]
            curve = []
            for dh in offs:
                Xf = floor_points(cam, uv, ctx['floor'], dh); num, den = 0., 0
                for k, c2 in enumerate(cams):
                    if k == r:
                        continue
                    x = (c2['K'] @ (c2['R'] @ Xf.T + c2['t'][:, None])).T; uv2 = x[:, :2] / x[:, 2:]
                    m = (x[:, 2] > 0) & (uv2[:, 0] > 1) & (uv2[:, 1] > 1) & (uv2[:, 0] < c2['w'] - 2) & (uv2[:, 1] < c2['h'] - 2)
                    m[m] &= ~inside(occ[k], uv2[m])
                    if m.sum() < 500:
                        continue
                    a, b = ref[m], sample(images[k], uv2[m])[:, 0]
                    a, b = a - a.mean(), b - b.mean(); c = float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-9))
                    num += c * m.sum(); den += m.sum()
                curve.append(num / den if den else np.nan)
            curve = np.array(curve)
            if np.isnan(curve).all():
                continue
            j = int(np.nanargmax(curve)); peak = curve[j]; contrast = peak - np.nanmin(curve)
            okt = peak >= P['SWEEP_NCC'] and contrast >= P['SWEEP_CONTRAST'] and 0 < j < len(offs) - 1 and not np.isnan(curve[j - 1:j + 2]).any()
            dh = offs[j]
            if okt:
                a_, b_, c_ = curve[j - 1:j + 2]; den_ = a_ - 2 * b_ + c_
                dh = offs[j] + (.5 * (a_ - c_) / den_ * P['SWEEP_STEP'] if den_ < 0 else 0)
            tiles.append(dict(tile=[x, y], heightNative=float(dh), ncc=float(peak), contrast=float(contrast), accepted=bool(okt)))
    acc = [t['heightNative'] for t in tiles if t['accepted']]
    return dict(referencePhoto=r + 1, tiles=tiles, accepted=len(acc),
                floorAboveReportNative=float(np.median(acc)) if acc else None,
                iqrNative=[float(np.percentile(acc, 25)), float(np.percentile(acc, 75))] if acc else None)


def floor_points(cam, uv, floor, dh):
    n, d = floor; Kinv = np.linalg.inv(cam['K'])
    r = (cam['R'].T @ (Kinv @ np.c_[uv, np.ones(len(uv))].T)).T
    lam = -(n @ cam['C'] + d - dh) / (r @ n)
    return cam['C'] + lam[:, None] * r


# ---------------------------------------------------------------- per object
def measure_post(ctx, obj, up):
    cams = ctx['cams']; per = {}
    for k, mask in sorted(obj['masks'].items()):
        vp = vanishing(cams[k], up); ys, xs = np.nonzero(mask); H = ys.max() - ys.min()
        box = (xs.min() - 150, ys.max() - .5 * H - 150, xs.max() + 150, ys.max() + 200)
        per[k] = housing_bottom(ctx['rgb'][k], mask, occluders(ctx, k, obj, box), vp)
    axis_photos = list(per)
    planes = [back_plane(cams[k], np.array(per[k]['axis'])) for k in axis_photos]
    prior = None
    if len(planes) < 2:
        k = axis_photos[0] if axis_photos else next(iter(per))
        prior = pi3x_mask_point(ctx, k, obj['masks'][k]); planes = [back_plane(cams[k], np.array(per[k]['axis']))]
        axis_photos = [k]
    X0 = vertical_line(planes, up, prior)
    usable = [k for k, b in per.items() if b['status'] == 'ok']
    res = dict(kind='post', photos={k + 1: summarize_post(b) for k, b in per.items()}, usablePhotos=[k + 1 for k in usable],
               axisPhotos=[k + 1 for k in axis_photos], axisFromPi3xPrior=prior is not None)
    S = ctx['S']; floor = ctx['floor']
    axis_h = {}
    for k in usable:
        for key in ('bottom', 'bottomDominant', 'bottomLowest', 'bottomMedian'):
            Xc, gap, _ = closest_on_line(*ray(cams[k], per[k][key]), X0, up)
            axis_h.setdefault(key, {})[k + 1] = dict(heightM=height(Xc, floor) * S, rayGapCm=gap * S * 100, X=Xc.tolist())
        # sensitivity of the axis height to 1 px of the bottom point (down the image)
        bb = np.array(per[k]['bottom']); Xa = closest_on_line(*ray(cams[k], bb), X0, up)[0]; Xb = closest_on_line(*ray(cams[k], bb + [0, 1]), X0, up)[0]
        axis_h['bottom'][k + 1]['cmPerPx'] = abs(height(Xa, floor) - height(Xb, floor)) * S * 100
        # the visible bottom edge lies on the near face, up to half the silhouette width in front of the axis; the ray keeps
        # descending to the axis, so the axis height can read low by up to (w/2) tan(depression)
        C, r = ray(cams[k], bb); z = float((cams[k]['R'] @ Xa + cams[k]['t'])[2]); w = per[k]['widthPx'] * z / cams[k]['K'][0, 0]
        dep = -(r @ up); tan = dep / math.sqrt(max(1 - dep * dep, 1e-9))
        axis_h['bottom'][k + 1].update(nearFaceBiasMaxCm=w / 2 * tan * S * 100, depressionDeg=math.degrees(math.atan(tan)), depthM=z * S)
    res['axisHeights'] = axis_h
    if len(usable) >= 2:
        for key in ('bottom', 'bottomDominant', 'bottomLowest', 'bottomMedian'):
            X = dlt([cams[k] for k in usable], [per[k][key] for k in usable])
            rep = [float(np.linalg.norm(project(cams[k], X) - per[k][key])) for k in usable]
            res.setdefault('dlt', {})[key] = dict(heightM=height(X, floor) * S, reprojPx=rep, X=X.tolist())
    if usable:  # primary: mean over usable photos of the bottom point's height on the triangulated vertical axis
        hs = [axis_h['bottom'][k + 1]['heightM'] for k in usable]
        res['primary'] = dict(method='axis_mean_photos_' + ''.join(str(k + 1) for k in usable), heightM=float(np.mean(hs)),
                              perPhotoM=hs, X=np.mean([axis_h['bottom'][k + 1]['X'] for k in usable], 0).tolist())
    else:
        res['primary'] = dict(method='none', heightM=None)
    res['axisLine'] = dict(X0=X0.tolist(), planeResidualsCm=[float(abs(p[:3] @ X0 + p[3]) * S * 100) for p in planes])
    return res, per


def summarize_post(b):
    keep = ('status', 'lines', 'accepted', 'occluded', 'lateral', 'lateralWarning', 'widthPx', 'colourTau', 'axisFit', 'bottom', 'bottomDominant', 'bottomLowest', 'bottomMedian', 'edgeSegments',
            'edgeInliers', 'edgeResidualPx', 'spreadPx', 'sigmaPx', 'channels', 'maskBottomPx')
    return {k: b[k] for k in keep if k in b}


def pi3x_mask_point(ctx, k, mask, band=None):
    fr = ctx['pi3x'][k]; X = fr['pts3d'].reshape(-1, 3).astype(float); ok = (fr['valid'] & fr['content']).ravel()
    cam = ctx['cams'][k]; x = (cam['K'] @ (cam['R'] @ X.T + cam['t'][:, None])).T; uv = x[:, :2] / x[:, 2:]
    ok &= inside(mask, uv)
    return np.median(X[ok], 0)


def measure_panel(ctx, obj, up, rng):
    cams = ctx['cams']; per = {}
    for k, mask in sorted(obj['masks'].items()):
        ys, xs = np.nonzero(mask)
        per[k] = fence_bottom(ctx['gray'][k], mask, occluders(ctx, k, obj, (xs.min() - 100, ys.min(), xs.max() + 100, ys.max() + 150)))
    ok = [k for k, b in per.items() if b['status'] == 'ok']
    S, floor = ctx['S'], ctx['floor']
    res = dict(kind='panel', photos={k + 1: {kk: v for kk, v in b.items() if kk != 'candidates'} | dict(candidates=[{c: x[c] for c in ('offsetPx', 'coverage', 'segments', 'aboveMinusBelow', 'ends')} for x in b['candidates']]) for k, b in per.items()},
               usablePhotos=[k + 1 for k in ok], variants={})
    V = res['variants']
    # single-view: image line on a vertical plane (Pi3X points just above the line; the displayed model)
    model_plane = None
    if obj.get('mesh') is not None:
        model_plane = model_vertical_plane(obj['mesh'], up, floor, rng)
    for k in ok:
        l = np.array(per[k]['line']); pi = back_plane(cams[k], l)
        Q, ninl, tot = pi3x_line_plane(ctx, k, obj['masks'][k], l, up, rng)
        for name, plane in (('pi3xPlane', Q), ('modelPlane', model_plane)):
            if plane is None:
                continue
            X0, D = plane_line(pi, plane); s = extent_on_line(cams[k], per[k]['ends'], X0, D)
            hs, mid = line_heights(X0, D, s, floor)
            hs = dict(medianM=hs['median'] * S, endsM=[x * S for x in hs['ends']], slopeDeg=hs['slopeDeg'], lengthM=hs['lengthNative'] * S)
            hs['inliers'] = [ninl, tot] if name == 'pi3xPlane' else None
            V.setdefault(name, {})[k + 1] = dict(**hs, mid=mid.tolist())
    if len(ok) >= 2:
        a, b = ok[:2]
        la, lb = np.array(per[a]['line']), np.array(per[b]['line'])
        pa, pb = back_plane(cams[a], la), back_plane(cams[b], lb)
        dihedral = math.degrees(math.acos(min(1., abs(pa[:3] @ pb[:3]))))
        X0, D = plane_line(pa, pb)
        sa, sb = extent_on_line(cams[a], per[a]['ends'], X0, D), extent_on_line(cams[b], per[b]['ends'], X0, D)
        lo, hi = max(sa[0], sb[0]), min(sa[1], sb[1]); overlap = hi > lo
        rng_s = (lo, hi) if overlap else (min(sa[0], sb[0]), max(sa[1], sb[1]))
        hs, mid = line_heights(X0, D, rng_s, floor)
        sens = []
        for which in (a, b):
            l2 = [la.copy(), lb.copy()]; i = 0 if which == a else 1; l2[i][2] -= 1.  # 1 px shift of one image line
            Xs, Ds = plane_line(back_plane(cams[a], l2[0]), back_plane(cams[b], l2[1]))
            m2 = closest_on_line(*ray(cams[a], np.mean(per[a]['ends'], 0)), Xs, Ds)[0]
            m1 = closest_on_line(*ray(cams[a], np.mean(per[a]['ends'], 0)), X0, D)[0]
            sens.append(abs(height(m2, floor) - height(m1, floor)) * S * 100)
        V['twoView'] = dict(photos=[a + 1, b + 1], dihedralDeg=dihedral, overlap=bool(overlap), medianM=hs['median'] * S,
                            endsM=[x * S for x in hs['ends']], slopeDeg=hs['slopeDeg'], lengthM=hs['lengthNative'] * S,
                            cmPerPx=float(math.hypot(*sens)), mid=mid.tolist())
        (Xh, Dh), rms = horizontal_fit([cams[a], cams[b]], [la, lb], [per[a]['ends'], per[b]['ends']], floor, (X0, D))
        sens = []
        for i in (0, 1):
            l2 = [la.copy(), lb.copy()]; l2[i][2] -= 1.
            (Xs, _), _ = horizontal_fit([cams[a], cams[b]], l2, [per[a]['ends'], per[b]['ends']], floor, (Xh, Dh))
            sens.append(abs(height(Xs, floor) - height(Xh, floor)) * S * 100)
        V['horizontal'] = dict(photos=[a + 1, b + 1], heightM=height(Xh, floor) * S, rmsPx=rms, cmPerPx=float(math.hypot(*sens)),
                               mid=closest_on_line(*ray(cams[a], np.mean(per[a]['ends'], 0)), Xh, Dh)[0].tolist())
        tv, hz = V['twoView'], V['horizontal']
        if tv['dihedralDeg'] >= P['MIN_DIHEDRAL_DEG'] and tv['cmPerPx'] <= P['MAX_CM_PER_PX'] and tv['overlap']:
            res['primary'] = dict(method='twoView', heightM=tv['medianM'], endsM=tv['endsM'], X=tv['mid'])
        elif hz['cmPerPx'] <= P['MAX_CM_PER_PX']:
            res['primary'] = dict(method='horizontal', heightM=hz['heightM'], X=hz['mid'])
    if 'primary' not in res and V.get('pi3xPlane'):
        vals = [v['medianM'] for v in V['pi3xPlane'].values()]
        k = next(iter(V['pi3xPlane']))
        res['primary'] = dict(method='pi3xPlane', heightM=float(np.median(vals)), endsM=V['pi3xPlane'][k]['endsM'], X=V['pi3xPlane'][k]['mid'])
    res.setdefault('primary', dict(method='none', heightM=None))
    return res, per


def pi3x_line_plane(ctx, k, mask, l, up, rng):
    fr = ctx['pi3x'][k]; X = fr['pts3d'].reshape(-1, 3).astype(float); ok = (fr['valid'] & fr['content']).ravel()
    cam = ctx['cams'][k]; x = (cam['K'] @ (cam['R'] @ X.T + cam['t'][:, None])).T; uv = x[:, :2] / x[:, 2:]
    sd = np.c_[uv, np.ones(len(uv))] @ l
    ok &= inside(mask, uv) & (sd >= P['PI3X_BAND'][0]) & (sd <= P['PI3X_BAND'][1])
    if ok.sum() < 20:
        return None, int(ok.sum()), int(ok.sum())
    return ransac_vertical_plane(X[ok], up, P['PI3X_TOL'], rng)


def model_vertical_plane(mesh, up, floor, rng):
    V, F = mesh; tri = V[F]; c = tri.mean(1); nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]); area = np.linalg.norm(nrm, axis=1)
    ok = area > 0; nrm[ok] /= area[ok, None]
    h = c @ floor[0] + floor[1]; lo = np.percentile(h, 0.5)
    sel = ok & (np.abs(nrm @ up) < .3) & (h < lo + .6 * (np.percentile(h, 99.5) - lo))
    if sel.sum() < 20:
        return None
    pick = rng.choice(np.nonzero(sel)[0], min(4000, sel.sum()), replace=False, p=area[sel] / area[sel].sum())
    return ransac_vertical_plane(c[pick], up, .01, rng)[0]


# ---------------------------------------------------------------- report
def run(ctx, opts):
    """ctx: load_report output plus rgb (list of HxWx3 float32 RGB) and pi3x {k: {pts3d, conf, valid, content}}."""
    up = ctx['floor'][0]; rng = np.random.default_rng(0)
    out, details = {}, {}
    for o in ctx['objects']:
        kind = (opts.get('targets') or {}).get(o['id'])
        if not kind:
            continue
        r, per = (measure_post(ctx, o, up) if kind == 'post' else measure_panel(ctx, o, up, rng))
        r['label'] = o['label']; details[o['id']] = per
        X = r['primary'].get('X')
        if X is not None:
            anchors = [np.array(X)]
            if kind == 'panel':
                anchors = panel_anchors(ctx, r, per)
            fP = pi3x_floor(ctx, ctx['pi3x'], anchors, np.random.default_rng(1))
            fW = sweep_floor(ctx, anchors) if opts.get('sweep', True) else None
            r['floor'] = dict(pi3x=fP, sweep=fW)
            h = r['primary']['heightM']; S = ctx['S']
            r['heightsM'] = dict(reportFloor=h,
                                 pi3xFloor=(h - fP['all']['reportMinusLocalNative'] * S) if fP['all'].get('status') == 'ok' else None,
                                 sweepFloor=(h - fW['floorAboveReportNative'] * S) if fW and fW['floorAboveReportNative'] is not None else None)
        out[o['id']] = r
    return dict(params=P, nativeToMeters=ctx['S'], floor=dict(normal=ctx['floor'][0].tolist(), offset=float(ctx['floor'][1])),
                objects=out), details


def panel_anchors(ctx, r, per):
    X = np.array(r['primary']['X']); return [X]


def overlay(rgb, mask, item, kind, pad=120, scale=None):
    """JPEG crop: mask outline (magenta), edge points / chosen line (green), other candidates (yellow), guide (cyan)."""
    img = cv2.cvtColor(np.clip(rgb, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR).copy()
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, cs, -1, (255, 0, 255), 1)
    if kind == 'post':
        for u, v in item.get('edge', []):
            cv2.circle(img, (int(round(u)), int(round(v))), 1, (0, 255, 0), -1)
        if 'bottom' in item:
            u, v = item['bottom']; cv2.drawMarker(img, (int(round(u)), int(round(v))), (0, 0, 255), cv2.MARKER_CROSS, 18, 1)
        cx, cy = item['axisPoint']; box = [cx - 220, cy - 260, cx + 220, cy + 120]
    else:
        a, b, u0, u1 = item['guide']; cv2.line(img, (int(u0), int(a * u0 + b)), (int(u1), int(a * u1 + b)), (255, 255, 0), 1)
        for i, c in enumerate(item['candidates']):
            (x0, y0), (x1, y1) = c['ends']
            cv2.line(img, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0) if i == item.get('chosen') else (0, 255, 255), 1)
        box = [u0 - pad, min(a * u0 + b, a * u1 + b) - pad - 80, u1 + pad, max(a * u0 + b, a * u1 + b) + pad]
    x0, y0 = int(max(box[0], 0)), int(max(box[1], 0)); x1, y1 = int(min(box[2], img.shape[1])), int(min(box[3], img.shape[0]))
    crop = img[y0:y1, x0:x1]
    if scale is None:
        scale = min(1., 900 / max(crop.shape[:2]))
    crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale != 1 else crop
    return cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()


# ---------------------------------------------------------------- self-test
def _cam(C, target, f=1500., w=1200, h=1600):
    z = target - C; z /= np.linalg.norm(z); x = np.cross(z, [0, 0, -1.]); x /= np.linalg.norm(x); y = np.cross(z, x)
    R = np.array([x, y, z]); K = np.array([[f, 0, w / 2 - .5], [0, f, h / 2 - .5], [0, 0, 1.]])
    return dict(K=K, R=R, t=-R @ C, C=np.asarray(C, float), w=w, h=h)


def _check():
    up = np.array([0, 0, 1.]); floor = (up, 0.)
    cams = [_cam(np.array([-.4, -2.5, 1.5]), np.array([0, 0, .3])), _cam(np.array([.5, -2.2, 1.4]), np.array([0, 0, .3]))]
    # geometry: a vertical post at (0.1, 0.2) ending 0.24 above the floor; a horizontal edge at 0.20
    B = np.array([.1, .2, .24]); T = B + [0, 0, 1.]
    planes = [back_plane(c, np.cross(np.r_[project(c, B), 1], np.r_[project(c, T), 1])) for c in cams]
    X0 = vertical_line(planes, up)
    for c in cams:
        Xc, gap, _ = closest_on_line(*ray(c, project(c, B)), X0, up); assert abs(height(Xc, floor) - .24) < 1e-6 and gap < 1e-6
    assert abs(height(dlt(cams, [project(c, B) for c in cams]), floor) - .24) < 1e-6
    A1, A2 = np.array([-.5, .6, .2]), np.array([.4, 1.6, .2])
    lines = [np.cross(np.r_[project(c, A1), 1], np.r_[project(c, A2), 1]) for c in cams]
    lines = [l / np.linalg.norm(l[:2]) * (1 if l[1] > 0 else -1) for l in lines]
    X0, D = plane_line(*[back_plane(c, l) for c, l in zip(cams, lines)])
    s = extent_on_line(cams[0], [project(cams[0], A1), project(cams[0], A2)], X0, D)
    hs, _ = line_heights(X0, D, s, floor); assert abs(hs['median'] - .2) < 1e-6 and abs(hs['slopeDeg']) < 1e-4
    ends = [[project(c, A1), project(c, A2)] for c in cams]
    (Xh, Dh), rms = horizontal_fit(cams, lines, ends, floor, (X0 + [0, 0, .03], D)); assert abs(height(Xh, floor) - .2) < 1e-5 and rms < 1e-3
    # image: a yellow bar on grey, rendered 4x and averaged (anti-aliased bottom at v = 700.3)
    H, W = 1000, 400; big = np.full((H * 4, W * 4, 3), 140., np.float32); yb = 700.3
    big[:int(round((yb + .5) * 4)), 160 * 4:240 * 4] = [230, 200, 40]
    rgb = cv2.resize(big, (W, H), interpolation=cv2.INTER_AREA); rgb = cv2.GaussianBlur(rgb, (0, 0), 1.)
    mask = np.zeros((H, W), bool); mask[300:704, 162:239] = True
    b = housing_bottom(rgb, mask, np.zeros((H, W), bool), np.array([200., 1e7, 1.]))
    assert b['status'] == 'ok' and abs(b['bottom'][1] - yb) < .25, (b['status'], b.get('bottom'))
    occl = np.zeros((H, W), bool); occl[705:760, 150:250] = True
    assert housing_bottom(rgb, mask, occl, np.array([200., 1e7, 1.]))['status'] == 'occluded'
    # fence: bright rail whose lower edge is the line v = 600 + 0.2 (u - 100), above a darker floor
    img = np.full((H * 4, W * 4), 80., np.float32); vv, uu = np.mgrid[0:H * 4, 0:W * 4] / 4.
    lower = 600 + .2 * (uu - 100); img[(vv < lower + .5) & (vv > lower - 25)] = 210.
    g = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    m = (vv[::4, ::4] < lower[::4, ::4] + 6) & (vv[::4, ::4] > lower[::4, ::4] - 200) & (uu[::4, ::4] > 30) & (uu[::4, ::4] < 370)
    fb = fence_bottom(g, m, np.zeros((H, W), bool)); l = np.array(fb['line'])
    for u in (60., 300.):
        v = -(l[0] * u + l[2]) / l[1]; assert abs(v - (600 + .2 * (u - 100))) < .3, (u, v)
    print('clearance_b self-test passed: housing bottom err %.3f px, fence line ok, geometry exact' % (b['bottom'][1] - yb))


if __name__ == '__main__':
    _check()
