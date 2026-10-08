"""Fair geometry-backbone A/B: Pi3X vs DA3-LARGE-1.1 vs DA3-BASE, scored against the field values (2026-10-05).

Fairness rules (fixed before any number was computed):
  input    every backbone sees the same frozen canonical frames of the run, (a) padded 518x518 as stored, (b) unpadded: the
           392x518 content crop [:, 63:455] of the same pixels; (b) outputs are put back on the 518 grid (pad = invalid, cx + 63)
  floor    three rules on the same frozen floor masks, all from the backbone's own points:
             lowest      the current rule (prepare_capture_evidence.floor: _ransac_floor_plane, lowest supported plane + SVD)
             maxInlier   largest-consensus plane (same candidate gate: normal within 20 deg of the camera up, cameras above),
                         same threshold, two SVD refits on its inliers
             lsq         total least squares over every mask point
  scale    each backbone's own e-stop (estop_cylinder.fit + joint_scale, red 4 + yellow 8 cm), up = that floor's normal; a run
           whose e-stop deviates >= 4 % is excluded from every aggregate
  photos   the Pi3X QA sets, pinned: housing 090R photos 1+3, 030R photo 1, 030L photo 2 (090L listed only); fence 090R 1+3;
           e-stop 090 photos 1-3, 030 photo 2; camera heights all five photos
Scores:
  housing  recessed-housing lower edge (field 24 cm): clearance-B image edge lines (model-free, fixed), each edge pixel's ray cast
           onto the backbone's own point-map surface just above the edge (robust plane on its points in the post mask, 8-64 px
           above the line), height = (n.X + d) x S, median over the edge pixels, mean over the pinned photos
  fence    090R mesh-fence bottom rail (field 20 cm): the two clearance-B image lines back-projected and intersected (model-free)
  e-stop   yellow/red width ratio (2.00, scale-free) and visible height above the bracket (~8.5 cm; not a scale anchor)
  cameras  camera-centre heights (secondary; ASSUMES one photographer holding the phone at a similar height)
"""
from __future__ import annotations

import math

import cv2
import numpy as np

FIELD_CM = dict(housing=24., fence=20., yellowOverRed=2., visible=8.5)
GATE = .04
P = dict(EDGE_HALF=60, ABOVE=6, BAND=(8., 64.), ERODE=8, MIN_PTS=12, MAD_K=2.5,  # housing near-face surface
         FLOOR_ITERS=4000, FLOOR_MAX=60000, FLOOR_COS=math.cos(math.radians(20.)), FLOOR_SEED=0)
PIN = {'090': dict(housing={'169518d8': [1, 3]}, listed={'5163a9b0': [1]}, fence={'ce9516a5': [1, 3]}),
       '030': dict(housing={'606109af': [2], 'd72e25ef': [1]}, listed={}, fence={})}


# ---------------------------------------------------------------- frames
def embed_unpadded(fr: dict, x0: int, width: int = 518) -> dict:
    """Outputs of the 392-wide crop back on the canonical grid: pad = invalid, principal point shifted by x0."""
    h, w = fr['pts3d'].shape[:2]
    out = dict(fr)
    for key, fill in (('pts3d', 0.), ('conf', 0.), ('valid', False)):
        a = fr[key]; full = np.full((h, width) + a.shape[2:], fill, a.dtype); full[:, x0:x0 + w] = a; out[key] = full
    K = np.array(fr['K'], float).copy(); K[0, 2] += x0; out['K'] = K
    return out


def content_mask(pts, valid, conf, alpha):
    """prepare_capture_evidence.RULE: native valid & declared alpha & finite & nonzero points & conf >= 0.1."""
    return valid & alpha & np.isfinite(pts).all(-1) & (np.linalg.norm(pts, axis=-1) > 1e-6) & (conf >= .1)


def camera(K, M):
    M = np.asarray(M, float); R = M[:3, :3].T; C = M[:3, 3]
    return dict(K=np.asarray(K, float), R=R, t=-R @ C, C=C, M=M)


def ray_dirs(cam, uv):
    d = (np.linalg.inv(cam['K']) @ np.c_[uv, np.ones(len(uv))].T).T @ cam['R']
    return d / np.linalg.norm(d, axis=1, keepdims=True)


# ---------------------------------------------------------------- floors
def _orient(n, d, up, centres):
    s = 1. if n @ up >= 0 else -1.
    return n * s, d * s


def _svd_plane(Q):
    c = Q.mean(0); n = np.linalg.svd(Q - c, full_matrices=False)[2][-1]
    return n, -float(n @ c)


def floor_rows(points, centres, up, thr, lowest_plane):
    """{rule: (n, d)} with n towards the cameras' up. lowest_plane = geometry_ab.floor_fit output for the current rule."""
    out = {'lowest': (np.asarray(lowest_plane['normal'], float), float(lowest_plane['offset']))}
    X = points[::max(1, len(points) // P['FLOOR_MAX'])].astype(float)
    rng = np.random.default_rng(P['FLOOR_SEED']); idx = rng.integers(0, len(X), (P['FLOOR_ITERS'], 3))
    a, b, c = X[idx[:, 0]], X[idx[:, 1]], X[idx[:, 2]]
    N = np.cross(b - a, c - a); L = np.linalg.norm(N, axis=1); ok = L > 1e-12; N, a = N[ok] / L[ok, None], a[ok]
    al = N @ up; keep = np.abs(al) >= P['FLOOR_COS']; N, a = N[keep] * np.sign(al[keep])[:, None], a[keep]
    D = -np.einsum('ij,ij->i', N, a)
    above = ((centres @ N.T + D) > 0).all(0)
    counts = np.zeros(len(N), int)
    for s in range(0, len(N), 128):
        counts[s:s + 128] = (np.abs(X @ N[s:s + 128].T + D[s:s + 128]) < thr).sum(0)
    counts[~above] = -1
    i = int(np.argmax(counts)); n, d = N[i], float(D[i])
    for _ in range(2):  # refit on the consensus set
        inl = np.abs(points @ n + d) < thr; n, d = _svd_plane(points[inl].astype(float)); n, d = _orient(n, d, up, centres)
    out['maxInlier'] = (n, d)
    out['lsq'] = _orient(*_svd_plane(points.astype(float)), up, centres)
    return out


def floor_stats(points, centres, n, d, thr):
    r = points @ n + d
    return dict(normal=[float(x) for x in n], offset=float(d), residualP95Native=float(np.quantile(np.abs(r), .95)),
                residualMedianSignedNative=float(np.median(r)), inlierFraction=float((np.abs(r) < thr).mean()),
                cameraHeightsNative=[float(c @ n + d) for c in centres])


def angle_deg(a, b):
    return math.degrees(math.acos(min(1., abs(float(np.dot(a, b)) / np.linalg.norm(a) / np.linalg.norm(b)))))


# ---------------------------------------------------------------- housing lower edge on the backbone's own surface
def edge_pixels(line, xc, mask, half=None, above=None):
    """housing_edge.py: pixels of the image line within +-half px of the segment point, kept where the mask is set just above."""
    half = P['EDGE_HALF'] if half is None else half; above = P['ABOVE'] if above is None else above
    a, b, c = line; xs = np.arange(xc - half, xc + half + 1, 1.0); ys = -(a * xs + c) / b
    ok = (ys >= 0) & (ys < mask.shape[0] - 1) & (xs >= 0) & (xs < mask.shape[1] - 1)
    xs, ys = xs[ok], ys[ok]
    inside = mask[np.clip(np.round(ys - above).astype(int), 0, mask.shape[0] - 1), xs.astype(int)]
    return np.c_[xs[inside], ys[inside]]


def robust_plane(X):
    keep = np.ones(len(X), bool)
    for _ in range(3):
        n, d = _svd_plane(X[keep]); r = X @ n + d
        keep = np.abs(r) <= max(P['MAD_K'] * 1.4826 * np.median(np.abs(r[keep])), 1e-6)
    n, d = _svd_plane(X[keep])
    return n, d, int(keep.sum())


def surface_points(fr, A, mask, line, xc):
    """The backbone's points on the canonical grid inside the (eroded) post mask, 8-64 original px above the edge line."""
    a, b, c = line; H, W = fr['content'].shape; E = P['ERODE']
    er = cv2.erode(mask.astype(np.uint8), np.ones((2 * E + 1, 2 * E + 1), np.uint8)) > 0
    y_line = -(a * xc + c) / b; half = P['EDGE_HALF']
    corners = np.array([[xc - half, y_line - P['BAND'][1] - 30, 1], [xc + half, y_line - P['BAND'][1] - 30, 1],
                        [xc - half, y_line + 30, 1], [xc + half, y_line + 30, 1]]) @ np.asarray(A, float).T
    u0, u1 = int(max(0, np.floor(corners[:, 0].min()))), int(min(W - 1, np.ceil(corners[:, 0].max())))
    v0, v1 = int(max(0, np.floor(corners[:, 1].min()))), int(min(H - 1, np.ceil(corners[:, 1].max())))
    vv, uu = np.mgrid[v0:v1 + 1, u0:u1 + 1]; uu, vv = uu.ravel(), vv.ravel()
    p = np.c_[uu, vv, np.ones(len(uu))] @ np.linalg.inv(np.asarray(A, float)).T
    px, py = p[:, 0], p[:, 1]
    above = (-(a * px + c) / b) - py
    ok = (np.abs(px - xc) <= P['EDGE_HALF']) & (above >= P['BAND'][0]) & (above <= P['BAND'][1])
    ok &= (px >= 0) & (py >= 0) & (px < mask.shape[1] - .5) & (py < mask.shape[0] - .5)
    ok[ok] &= er[np.round(py[ok]).astype(int), np.round(px[ok]).astype(int)]
    ok &= fr['content'][vv, uu]
    return fr['pts3d'][vv[ok], uu[ok]].astype(float), int(ok.sum())


def housing_photo(cam, fr, A, mask, seg, floor, S, up):
    line = np.asarray(seg['line'], float); xc = float(seg['point'][0])
    uv = edge_pixels(line, xc, mask)
    X, npts = surface_points(fr, A, mask, line, xc)
    row = dict(edgePixels=len(uv), surfacePoints=npts)
    if len(uv) < 5 or npts < P['MIN_PTS']:
        return dict(row, status='too few pixels or points', heightCm=None)
    n_p, d_p, kept = robust_plane(X)
    D = ray_dirs(cam, uv); C = cam['C']; n, d = floor
    m_v = n_p - (n_p @ up) * up; m_v /= np.linalg.norm(m_v)  # variant: same points, normal constrained horizontal
    out = {}
    for name, (m, e) in (('freePlane', (n_p, d_p)), ('verticalPlane', (m_v, -float(np.median(X @ m_v))))):
        t = -(m @ C + e) / (D @ m); hit = t > 0
        h = ((C + t[hit, None] * D[hit]) @ n + d) * S * 100
        out[name] = float(np.median(h)) if hit.sum() >= 5 else None
    depth = float(np.median((X - C) @ cam['R'][2]))  # camera z of the surface (native)
    return dict(row, status='ok', planeInliers=kept, planeTiltFromVerticalDeg=float(90 - angle_deg(n_p, up)),
                heightCm=out['freePlane'], verticalPlaneCm=out['verticalPlane'], surfaceDepthNative=depth)


# ---------------------------------------------------------------- fence bottom rail, two-view lines (clearance_b geometry)
def fence_two_view(cams, lines, ends, floor, S, cb):
    ca, cb_ = [dict(K=c['K'], R=c['R'], t=c['t'], C=c['C']) for c in cams]
    la, lb = (np.asarray(x, float) for x in lines)
    pa, pb = cb.back_plane(ca, la), cb.back_plane(cb_, lb)
    dihedral = math.degrees(math.acos(min(1., abs(pa[:3] @ pb[:3]))))
    X0, D = cb.plane_line(pa, pb)
    sa, sb = cb.extent_on_line(ca, ends[0], X0, D), cb.extent_on_line(cb_, ends[1], X0, D)
    lo, hi = max(sa[0], sb[0]), min(sa[1], sb[1]); overlap = hi > lo
    hs, _ = cb.line_heights(X0, D, (lo, hi) if overlap else (min(sa[0], sb[0]), max(sa[1], sb[1])), floor)
    sens = []
    for i in (0, 1):  # 1 px shift of one image line
        l2 = [la.copy(), lb.copy()]; l2[i][2] -= 1.
        Xs, Ds = cb.plane_line(cb.back_plane(ca, l2[0]), cb.back_plane(cb_, l2[1]))
        m = np.mean(ends[0], 0)
        sens.append(abs(cb.height(cb.closest_on_line(*cb.ray(ca, m), Xs, Ds)[0], floor) - cb.height(cb.closest_on_line(*cb.ray(ca, m), X0, D)[0], floor)) * S * 100)
    return dict(heightCm=hs['median'] * S * 100, endsCm=[x * S * 100 for x in hs['ends']], slopeDeg=hs['slopeDeg'], dihedralDeg=dihedral,
                overlap=bool(overlap), cmPerPx=float(math.hypot(*sens)))


# ---------------------------------------------------------------- e-stop: scale, ratio, visible height
def scaled_K(K, sx, sy):  # pixel-centre convention: x' = (x + .5) s - .5
    K = np.array(K, float); return np.array([[K[0, 0] * sx, 0, (K[0, 2] + .5) * sx - .5], [0, K[1, 1] * sy, (K[1, 2] + .5) * sy - .5], [0, 0, 1]])


def triangulate_axis(views):
    Aq, bq = [], []
    for v in views:
        M = v['M']; d = M[:3, :3] @ (np.linalg.inv(v['K']) @ [v['seed'][0], v['seed'][1], 1]); d /= np.linalg.norm(d)
        Pm = np.eye(3) - np.outer(d, d); Aq.append(Pm); bq.append(Pm @ M[:3, 3])
    return np.linalg.solve(np.sum(Aq, 0), np.sum(bq, 0))


def pointmap_point(fr, A, box):
    pts = []
    for y in range(*box[1], 4):
        for x in range(*box[0], 6):
            u, v, _ = A @ [x, y, 1]; ui, vi = int(round(u)), int(round(v))
            if fr['content'][vi, ui]:
                pts.append(fr['pts3d'][vi, ui])
    return np.median(pts, axis=0)


SPEC = (('red lip 4 cm', 'red lip', .04), ('yellow body 8 cm', 'yellow body', .08))


def estop(views, P3, up, fit, joint_scale, images):
    """Widths by estop_cylinder.fit (unchanged), joint scale, and the visible height of each photo on that scale."""
    rows, per, fits = [], {}, {}
    for v in views:
        M = v['M']; res = fit(images[v['name']], v['K'], M, up, P3, v['seed'], v['top'], v['bot'])
        z = float((M[:3, :3].T @ (P3 - M[:3, 3]))[2]); fx = float(v['K'][0, 0]); w = {k: r['symmetricWidthPx'] for k, r in res['parts'].items()}
        per[v['name']] = dict(widthsPx=w, depthNative=z, fx=fx, yellowOverRed=w['yellow body'] / w['red lip'], axisTiltDeg=res['vanishingTiltDeg'])
        fits[v['name']] = res
        rows += [dict(photo=v['name'], feature=f, specM=s, measuredNative=w[part] * z / fx) for f, part, s in SPEC]
    js = joint_scale(rows)
    for v in views:
        try:
            per[v['name']]['visibleHeightCm'] = visible_height(images[v['name']], v['K'], v['M'], up, P3, per[v['name']]['depthNative'],
                                                               js['nativeToMeters'], fits[v['name']], per[v['name']]['widthsPx'])
        except Exception as error:  # noqa: BLE001 - reported per photo
            per[v['name']]['visibleHeightCm'] = dict(error=str(error)[:200])
    return dict(P3=[float(x) for x in P3], perPhoto=per, **js)


def visible_height(img, K, M, up, P3, z, S, res, ws):
    """estop-dimensions-2026-10-05 (estop_dims_final.py) on the full image, with this backbone's pose, up, depth, scale, widths:
    red silhouette top (far rim of the lip disc; flat top) / apex (domed top) to the grey base's near rim on the bracket."""
    from scipy.optimize import brentq
    sp = res['spans']; Kc = np.asarray(K, float)
    rgb = cv2.GaussianBlur(img, (0, 0), .7).astype(np.float32); hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    H, Sat, Val = hsv[..., 0], hsv[..., 1] / 255, hsv[..., 2] / 255
    like_red = cv2.GaussianBlur((Sat * np.exp(-np.square(np.minimum(H, 180 - H) / 8)) * (Val > .25)).astype(np.float32), (0, 0), 1.0)

    def remap(im, xs, ys):
        X = np.asarray(xs, np.float32).reshape(1, -1); Y = np.asarray(ys, np.float32).reshape(1, -1)
        return cv2.remap(im, X, Y, cv2.INTER_LINEAR).reshape(np.shape(xs) + im.shape[2:])

    def prof(im, rows, x_off, half):
        o = []
        for r in rows:
            p, n = res['axis'](r); offs = x_off + np.arange(-half, half + .01, 1.0)
            o.append(remap(im, p[0] + offs * n[0], p[1] + offs * n[1]).mean(0))
        return np.array(o)

    def cross(rows, y, i_from, i_to, mid):
        st = 1 if i_to > i_from else -1; s0 = np.sign(y[i_from] - mid)
        for i in range(i_from + st, i_to + st, st):
            if np.sign(y[i] - mid) != s0:
                f = (y[i - st] - mid) / (y[i - st] - y[i]); return float(rows[i - st] + st * f * (rows[1] - rows[0]))
        return None

    def ellipse(xs, ys, a):
        xs, ys = np.array(xs), np.array(ys); keep = np.ones(len(xs), bool)
        for _ in range(3):
            Am = np.stack([np.ones(keep.sum()), np.sqrt(1 - np.square(xs[keep] / a))], 1)
            (c0, b), *_ = np.linalg.lstsq(Am, ys[keep], rcond=None); r = ys - (c0 + b * np.sqrt(1 - np.square(xs / a)))
            keep = np.abs(r) <= max(3 * 1.4826 * np.median(np.abs(r[keep])), .5)
        return float(c0), float(b)
    Q = 4
    a_g = ws['grey cylinder'] / 2; rows_b = np.arange(sp['grey'][0] + 2, sp['grey'][1] + 10, 1 / Q); bx, by = [], []
    for xo in np.linspace(-.95 * a_g, .95 * a_g, 39):
        L = prof(rgb, rows_b, xo, 1.0).mean(1); gl = np.median(L[:4 * Q]); k = int(np.argmax(L < .6 * gl))
        if k < 7 * Q or L[k] >= .6 * gl:
            continue
        mid = (np.median(L[k - 7 * Q:k - 3 * Q]) + L[k:k + 5 * Q].min()) / 2; yb = cross(rows_b, L, k - 3 * Q, k, mid)
        if yb is not None:
            bx.append(xo); by.append(yb)
    yb0, bb = ellipse(bx, by, a_g); bot_axis = yb0 + bb
    rows_t = np.arange(sp['red'][0] - 16, sp['red'][0] + .5 * (sp['red'][1] - sp['red'][0]), 1 / Q)
    Lr = prof(like_red, rows_t, 0, .2 * ws['red lip']); Pp = prof(rgb, rows_t, 0, .2 * ws['red lip'])
    hi = Lr >= .85 * np.percentile(Lr, 90); run = np.convolve(hi, np.ones(3 * Q), 'valid') == 3 * Q; s0 = int(np.argmax(run))
    face = np.median(Pp[s0 - 5 * Q:s0 - 2 * Q], 0); side = np.median(Pp[s0 + 2 * Q:s0 + 5 * Q], 0); ch = int(np.argmax(np.abs(face - side)))
    near = cross(rows_t, Pp[:, ch], s0 - 4 * Q, s0 + 3 * Q, (face[ch] + side[ch]) / 2)
    Dd = np.linalg.norm(Pp - face, axis=1); i_n = int(np.searchsorted(rows_t, near))
    i_dep = next((i for i in range(i_n - 2 * Q, 0, -1) if Dd[i] > .5 * np.linalg.norm(face - side)), None)
    far = None
    if i_dep is not None and i_dep - int(2.5 * Q) >= 0:
        bg = np.median(Pp[i_dep - int(2.5 * Q):i_dep - Q], 0); far = cross(rows_t, Dd, i_n - 2 * Q, max(i_dep - Q, 0), .5 * np.linalg.norm(bg - face))
    R_ = M[:3, :3]; C = M[:3, 3]; up = up / np.linalg.norm(up)
    pm, _ = res['axis'](((far if far is not None else near) + bot_axis) / 2); r_ = R_ @ (np.linalg.inv(Kc) @ [pm[0], pm[1], 1]); Ax = C + r_ / (R_.T @ r_)[2] * z
    proj = lambda X: (lambda x: x[:2] / x[2])(Kc @ (R_.T @ (X - C)))
    u_img = res['axis'](0)[0] - res['axis'](1)[0]; u_img /= np.linalg.norm(u_img)
    e1 = np.cross(up, [1, 0, 0]); e1 /= np.linalg.norm(e1); e2 = np.cross(up, e1); ph = np.linspace(0, 2 * np.pi, 721)
    rim = np.cos(ph)[:, None] * e1 + np.sin(ph)[:, None] * e2; fx = float(Kc[0, 0]); rn = {k: ws[k] * z / fx / 2 for k in ws}

    def hrim(row, r, ext):
        return brentq(lambda s: getattr(np, ext)(np.stack([proj(Ax + s * up + r * q) for q in rim]) @ u_img) - res['axis'](row)[0] @ u_img, -.4, .4)
    cm = lambda native: 100 * native * S
    h_bot = hrim(bot_axis, rn['grey cylinder'], 'min')
    out = dict(faceSideBoundaryFlatTop=cm(hrim(near, rn['red lip'], 'min') - h_bot))
    if far is not None:
        out.update(silhouetteFlatTop=cm(hrim(far, rn['red lip'], 'max') - h_bot), silhouetteDomedTop=cm(hrim(far, 0, 'max') - h_bot))
    lo, hi_ = (out['silhouetteFlatTop'], out['silhouetteDomedTop']) if far is not None else (out['faceSideBoundaryFlatTop'],) * 2
    view = (Ax - C) / np.linalg.norm(Ax - C)
    return dict(flatTop=lo, domedTop=hi_, mid=(lo + hi_) / 2, cameraDepressionDeg=float(np.degrees(np.arcsin(-view @ up))),
                rows=dict(far=far, near=near, bottom=bot_axis), **out)


# ---------------------------------------------------------------- self-test
def _check():
    rng = np.random.default_rng(1)
    # floors: a tilted plane plus 20 % clutter 5-30 cm above it; maxInlier recovers the plane, lsq is pulled up, lowest = plane
    n = np.array([.05, -.1, .99]); n /= np.linalg.norm(n); d = .4
    e1 = np.cross(n, [1, 0, 0]); e1 /= np.linalg.norm(e1); e2 = np.cross(n, e1)
    F = -d * n + rng.uniform(-1, 1, (8000, 1)) * e1 + rng.uniform(-1, 1, (8000, 1)) * e2
    clutter = -d * n + rng.uniform(-.5, .5, (2000, 1)) * e1 + rng.uniform(-.5, .5, (2000, 1)) * e2 + rng.uniform(.05, .3, (2000, 1)) * n
    pts = np.r_[F, clutter]; centres = np.array([-d * n + 1.5 * n, -d * n + 1.5 * n + .3 * e1])
    rows = floor_rows(pts, centres, n, .01, dict(normal=n, offset=d))
    m, dd = rows['maxInlier']; assert angle_deg(m, n) < .05 and abs(dd - d) < 1e-3, (m, dd)
    m, dd = rows['lsq']; assert (centres[0] @ m + dd) < 1.5 - .01, 'lsq should be pulled towards the clutter'
    st = floor_stats(pts, centres, *rows['maxInlier'], .01); assert abs(st['cameraHeightsNative'][0] - 1.5) < 1e-3 and st['inlierFraction'] > .79
    # housing: a vertical face y = 2 (camera looks along +y), lower edge at height 0.24 above the floor z = 0; pixels above the
    # edge see the face; the ray-plane height of the edge pixels is exact, and the point band picks the face, not the floor
    K = np.array([[1000., 0, 500], [0, 1000, 500], [0, 0, 1]])
    Rw = np.array([[1., 0, 0], [0, 0, -1], [0, 1, 0]])  # world z up -> camera y down; camera looks along world +y
    C = np.array([0., 0, 1.2]); M = np.eye(4); M[:3, :3] = Rw.T; M[:3, 3] = C; cam = camera(K, M)
    Hh, Ww = 1000, 1000; v_edge = 500 + 1000 * (1.2 - .24) / 2.  # image row of the edge
    yy, xx = np.mgrid[0:Hh, 0:Ww].astype(float); dirs = (np.stack([xx, yy, np.ones_like(xx)], -1) @ np.linalg.inv(K).T) @ Rw
    t = 2. / dirs[..., 1]; pts3 = C + t[..., None] * dirs; below = pts3[..., 2] < .24
    tf = -C[2] / np.minimum(dirs[..., 2], -1e-9); pts3[below] = (C + tf[..., None] * dirs)[below]  # below the edge the rays hit the floor
    fr = dict(pts3d=pts3.astype(np.float32), content=np.ones((Hh, Ww), bool))
    mask = np.zeros((Hh, Ww), bool); mask[:int(v_edge) + 1, 400:600] = True
    row = housing_photo(cam, fr, np.eye(3), mask, dict(line=[0., 1., -v_edge], point=[500., v_edge]), (np.array([0., 0, 1]), 0.), 1., np.array([0., 0, 1]))
    assert row['status'] == 'ok' and abs(row['heightCm'] - 24.) < .05 and abs(row['verticalPlaneCm'] - 24.) < .05, row
    # unpadded embedding: pad invalid, K shifted, pixel (v, u) of the crop lands at (v, u + 63)
    crop = dict(pts3d=rng.normal(size=(4, 5, 3)).astype(np.float32), conf=np.ones((4, 5), np.float32), valid=np.ones((4, 5), bool), K=np.eye(3), c2w=np.eye(4))
    full = embed_unpadded(crop, 63, 70)
    assert full['pts3d'].shape == (4, 70, 3) and not full['valid'][:, :63].any() and np.array_equal(full['pts3d'][:, 63:68], crop['pts3d']) and full['K'][0, 2] == 63
    print('fair_ab self-test passed')


if __name__ == '__main__':
    _check()
