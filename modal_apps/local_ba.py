"""X11 local bundle adjustment for an X4 refine spot (fixes X4's 0/15: the coarse cameras are ~4 cm off).

The spot's crops are matched (ALIKED + LightGlue, every pair, fundamental-matrix MAGSAC at 1 px) and the matches chained
into tracks. Then the cameras (6 DOF each; the coarse pose is a prior, moves bounded to X4's ICP refusal limits 3 deg /
10 cm), one DA3 depth scale + shift per view and the track points are re-solved jointly: soft-L1 reprojection error of
every track + the DA3 depth at each track near the spot (box + 30 cm) + the priors. 'before' = the same objective with
the cameras and depth frozen (only the points move): the best the coarse cameras can do with these matches.
'+tri': after the BA, each view's depth is also bent onto the triangulated tracks near the spot (a Gaussian-weighted
field of the track-vs-DA3 depth residuals, 40 px, shrunk to 0 away from tracks): depth from the matched multi-view
triangulation where there are tracks, DA3's shape between them.

  python modal_apps/local_ba.py --self-check     # numpy/scipy/cv2: synthetic cameras 2 deg / 6 cm off, depth x1.07 + 3 cm
Scale is 'estimated' (the coarse map's floor + an assumed 1.6 m camera height): m / cm here are estimated values.
"""
import sys
import time

import numpy as np

SIGMA_PX, SIGMA_R_DEG, SIGMA_T_M = 1., 1.5, .05
MAX_R_DEG, MAX_T_M = 3., .10  # X4's per-view ICP refusal limits
SIGMA_LS, SIGMA_B_M, MAX_LS, MAX_B_M = .05, .05, .2, .2
DEPTH_SIG = (.01, .01)  # DA3 depth at a track: 1 cm + 1 % of the depth
REGION_M, F_SCALE, RANSAC_PX = .3, 2., 1.
FIELD_PX, FIELD_SHRINK = 40., .25
LICENCES = {"ALIKED (code + aliked-n16 weights)": "BSD-3-Clause, github.com/Shiaoming/ALIKED",
            "LightGlue (code + aliked_lightglue weights)": "Apache-2.0, github.com/cvg/LightGlue",
            "kornia (LightGlue dependency)": "Apache-2.0"}


# ---------- geometry (numpy) ----------

def so3_exp(w):
    w = np.atleast_2d(np.asarray(w, np.float64))
    th = np.linalg.norm(w, axis=1)
    k = w / np.maximum(th, 1e-12)[:, None]
    Kx = np.zeros((len(w), 3, 3))
    Kx[:, 0, 1], Kx[:, 0, 2], Kx[:, 1, 2] = -k[:, 2], k[:, 1], -k[:, 0]
    Kx[:, 1, 0], Kx[:, 2, 0], Kx[:, 2, 1] = k[:, 2], -k[:, 1], k[:, 0]
    s, c = np.sin(th)[:, None, None], np.cos(th)[:, None, None]
    return np.eye(3) + s * Kx + (1 - c) * (Kx @ Kx)


def moved(c2w, w, t):
    """c2w (n,4,4): rotation Exp(w) about each camera centre (world axes), centre + t."""
    out = np.array(c2w, np.float64)
    out[:, :3, :3] = so3_exp(w) @ out[:, :3, :3]
    out[:, :3, 3] += t
    return out


def proj(R, C, K, X):
    """per observation: c2w rotation R (N,3,3), centre C (N,3), K (N,3,3), point X (N,3) -> u, v, z."""
    Xc = np.einsum("nji,nj->ni", R, X - C)
    z = Xc[:, 2]
    zs = np.where(z > 1e-3, z, 1e-3)
    return K[:, 0, 0] * Xc[:, 0] / zs + K[:, 0, 2], K[:, 1, 1] * Xc[:, 1] / zs + K[:, 1, 2], z


def sample_depth(D, u, v, edge=.05):
    """Bilinear depth at (u, v); nan off the image, on a hole, or across a depth edge (> 5 % spread of the 4 pixels)."""
    h, w = D.shape
    x0, y0 = np.floor(u).astype(int), np.floor(v).astype(int)
    ok = (x0 >= 0) & (y0 >= 0) & (x0 < w - 1) & (y0 < h - 1)
    x0, y0 = np.clip(x0, 0, w - 2), np.clip(y0, 0, h - 2)
    q = np.stack([D[y0, x0], D[y0, x0 + 1], D[y0 + 1, x0], D[y0 + 1, x0 + 1]], 1).astype(np.float64)
    ax, ay = u - x0, v - y0
    d = q[:, 0] * (1 - ax) * (1 - ay) + q[:, 1] * ax * (1 - ay) + q[:, 2] * (1 - ax) * ay + q[:, 3] * ax * ay
    ok &= (q.min(1) > 0) & (q.max(1) <= q.min(1) * (1 + edge))
    return np.where(ok, d, np.nan)


def build_tracks(nkp, pairs):
    """nkp keypoints per view, pairs [(i, j, idx_i, idx_j)] of verified matches -> (view, kp, track) per observation.
    Matches chain into tracks (connected components); a track with two keypoints in one view is ambiguous: dropped."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    e = np.zeros(0, int)
    pairs = [p for p in pairs if len(p[2])]
    if not pairs:
        return e, e, e
    off = np.concatenate([[0], np.cumsum(nkp)]).astype(int)
    a = np.concatenate([off[i] + np.asarray(x, int) for i, _, x, _ in pairs])
    b = np.concatenate([off[j] + np.asarray(y, int) for _, j, _, y in pairs])
    N = int(off[-1])
    _, lab = connected_components(coo_matrix((np.ones(len(a)), (a, b)), shape=(N, N)), directed=False)
    nodes = np.unique(np.concatenate([a, b]))
    view = np.searchsorted(off, nodes, side="right") - 1
    comp, n, L = lab[nodes], len(nkp), int(lab.max()) + 1
    per = np.bincount(comp, minlength=L)
    uniq = np.bincount(np.unique(comp * n + view) // n, minlength=L)
    keep = ((per == uniq) & (per >= 2))[comp]
    _, track = np.unique(comp[keep], return_inverse=True)
    return view[keep], nodes[keep] - off[view[keep]], track.astype(int)


def init_points(view, uv, track, K, c2w, dobs, T):
    """Per track: the median of its observations back-projected through the DA3 depth; DLT with the given cameras where
    no observation has depth. -> X (T,3), has (T,) bool."""
    ray = np.einsum("nij,nj->ni", np.linalg.inv(K[view]), np.c_[uv, np.ones(len(uv))])
    Pw = np.einsum("nij,nj->ni", c2w[view, :3, :3], ray * np.nan_to_num(dobs)[:, None]) + c2w[view, :3, 3]
    X, has = np.zeros((T, 3)), np.zeros(T, bool)
    order = np.argsort(track, kind="stable")
    starts = np.searchsorted(track[order], np.arange(T + 1))
    Pmat = np.einsum("nij,njk->nik", K, np.linalg.inv(c2w)[:, :3])  # (views,3,4)
    for t in range(T):
        o = order[starts[t]:starts[t + 1]]
        d = o[np.isfinite(dobs[o])]
        if len(d):
            X[t], has[t] = np.median(Pw[d], 0), True
        elif len(o) >= 2:
            P = Pmat[view[o]]
            A = np.concatenate([uv[o, :1, None] * P[:, 2:3] - P[:, 0:1], uv[o, 1:, None] * P[:, 2:3] - P[:, 1:2]], 1).reshape(-1, 4)
            h = np.linalg.svd(A)[2][-1]
            if abs(h[3]) > 1e-9:
                X[t], has[t] = h[:3] / h[3], True
    return X, has


# ---------- the solve (scipy) ----------

def skew(y):
    S = np.zeros(y.shape[:-1] + (3, 3))
    S[..., 0, 1], S[..., 0, 2], S[..., 1, 2] = -y[..., 2], y[..., 1], -y[..., 0]
    S[..., 1, 0], S[..., 2, 0], S[..., 2, 1] = y[..., 2], -y[..., 1], y[..., 0]
    return S


def left_jac(w):
    """SO(3) left Jacobian: Exp(w + d) = Exp(J_l(w) d) Exp(w) to first order."""
    th = np.linalg.norm(w, axis=1)[:, None, None]
    W = skew(w)
    small = th < 1e-6
    a = np.where(small, .5, (1 - np.cos(th)) / np.where(small, 1, th) ** 2)
    b = np.where(small, 1 / 6, (th - np.sin(th)) / np.where(small, 1, th) ** 3)
    return np.eye(3) + a * W + b * (W @ W)


def kabsch(P, Q):
    """Rigid (R, t) minimising |R P + t - Q|^2 over rows."""
    mp, mq = P.mean(0), Q.mean(0)
    U, _, Vt = np.linalg.svd((P - mp).T @ (Q - mq))
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, mq - R @ mp


def cam_points(c2w, arm=.5):
    """Each camera as its centre + its three axes at arm m: 4 points (a rigid fit on these is well posed for a line of cameras)."""
    c2w = np.asarray(c2w)
    return np.concatenate([c2w[:, :3, 3]] + [c2w[:, :3, 3] + arm * c2w[:, :3, k] for k in range(3)])


def solve(view, uv, track, K, c2w0, dobs, dsel, X0, cams=True, max_nfev=100, fns=False):
    """Joint solve -> dict(c2w, ls, b, X, residual stats). cams=False: only the points move (the 'before' diagnostic).
    Parameters [w 3n | t 3n | log-scale n | shift n | X 3T]; residuals [u N | v N | depth Nd | priors 8n], each in sigmas.
    Analytic Jacobian (the rotation's through the SO(3) left Jacobian).
    The result is put back in the coarse frame (gauge): the rigid transform that best maps the solved cameras onto the
    coarse ones is applied to cameras and points (reprojection and depth unchanged; it is reported as 'gauge')."""
    from scipy.sparse import coo_matrix
    n, T, N = len(c2w0), len(X0), len(view)
    c2w0 = np.asarray(c2w0, np.float64)
    R0, C0 = c2w0[:, :3, :3], c2w0[:, :3, 3]
    di = np.flatnonzero(dsel)
    Nd = len(di)
    dval = dobs[di]
    dsig = DEPTH_SIG[0] + DEPTH_SIG[1] * dval
    sr = np.radians(SIGMA_R_DEG)
    prior_sig = np.concatenate([np.full(3 * n, sr), np.full(3 * n, SIGMA_T_M), np.full(n, SIGMA_LS), np.full(n, SIGMA_B_M)])

    def unpack(p):
        return (p[:3 * n].reshape(n, 3), p[3 * n:6 * n].reshape(n, 3), p[6 * n:7 * n], p[7 * n:8 * n], p[8 * n:].reshape(T, 3))

    def cam_frame(p):
        w, t, ls, b, X = unpack(p)
        R = so3_exp(w) @ R0
        C = C0 + t
        Rv, y = R[view], X[track] - C[view]
        return w, t, ls, b, X, Rv, y, np.einsum("nji,nj->ni", Rv, y)

    def residual(p):
        w, t, ls, b, X, Rv, y, Xc = cam_frame(p)
        z = Xc[:, 2]
        zs = np.where(z > 1e-3, z, 1e-3)
        u, v = K[view, 0, 0] * Xc[:, 0] / zs + K[view, 0, 2], K[view, 1, 1] * Xc[:, 1] / zs + K[view, 1, 2]
        pred = np.exp(ls[view[di]]) * dval + b[view[di]]
        return np.concatenate([(u - uv[:, 0]) / SIGMA_PX, (v - uv[:, 1]) / SIGMA_PX, (z[di] - pred) / dsig, p[:8 * n] / prior_sig])

    cam = lambda k, vs: np.stack([3 * k * n + 3 * vs + a for a in range(3)], 1)  # noqa: E731  block k (w, t): 3 columns per view
    Xcol = lambda ts: 8 * n + 3 * ts[:, None] + np.arange(3)  # noqa: E731
    cols = [np.concatenate([cam(0, view), cam(1, view), Xcol(track)], 1)] * 2
    cols.append(np.concatenate([cam(0, view[di]), cam(1, view[di]), (6 * n + view[di])[:, None], (7 * n + view[di])[:, None], Xcol(track[di])], 1))
    rows = np.concatenate([np.repeat(np.arange(N), 9), np.repeat(N + np.arange(N), 9), np.repeat(2 * N + np.arange(Nd), 11), 2 * N + Nd + np.arange(8 * n)])
    cols = np.concatenate([c.ravel() for c in cols] + [np.arange(8 * n)])
    shape = (2 * N + Nd + 8 * n, 8 * n + 3 * T)

    def jacobian(p):
        w, t, ls, b, X, Rv, y, Xc = cam_frame(p)
        Rt = Rv.transpose(0, 2, 1)
        z = np.where(Xc[:, 2] > 1e-3, Xc[:, 2], 1e-3)
        blocks = np.concatenate([Rt @ skew(y) @ left_jac(w)[view], -Rt, Rt], 2)  # d camera-frame point / d (w, t, X): (N, 3, 9)
        fx, fy = K[view, 0, 0], K[view, 1, 1]
        zero = np.zeros(N)
        du = np.stack([fx / z, zero, -fx * Xc[:, 0] / z ** 2], 1) / SIGMA_PX
        dv = np.stack([zero, fy / z, -fy * Xc[:, 1] / z ** 2], 1) / SIGMA_PX
        Ju, Jv = np.einsum("nk,nkj->nj", du, blocks), np.einsum("nk,nkj->nj", dv, blocks)
        bz = blocks[di, 2] / dsig[:, None]
        Jd = np.concatenate([bz[:, :6], (-np.exp(ls[view[di]]) * dval / dsig)[:, None], (-1 / dsig)[:, None], bz[:, 6:]], 1)
        vals = np.concatenate([Ju.ravel(), Jv.ravel(), Jd.ravel(), 1 / prior_sig])
        return coo_matrix((vals, (rows, cols)), shape=shape).tocsr()
    p0 = np.concatenate([np.zeros(8 * n), np.asarray(X0, np.float64).ravel()])
    if fns:  # the self-check compares the Jacobian with finite differences
        return residual, jacobian, p0
    bound = np.concatenate([np.full(3 * n, np.radians(MAX_R_DEG)), np.full(3 * n, MAX_T_M), np.full(n, MAX_LS), np.full(n, MAX_B_M)])
    t0 = time.perf_counter()
    p, it, lm_info = lm(residual, jacobian, p0, 8 * n, T, bound, cams, max_nfev)
    w, t, ls, b, X = unpack(p)
    raw = moved(c2w0, w, t)
    at_bound = bool(cams and np.any(np.abs(p[:6 * n]) >= .999 * bound[:6 * n]))
    Rg, tg = kabsch(cam_points(raw), cam_points(c2w0))  # the gauge back to the coarse frame
    c2w = raw.copy()
    c2w[:, :3, :3] = Rg @ raw[:, :3, :3]
    c2w[:, :3, 3] = raw[:, :3, 3] @ Rg.T + tg
    X = X @ Rg.T + tg
    u, v, z = proj(c2w[view, :3, :3], c2w[view, :3, 3], K[view], X[track])
    e = np.hypot(u - uv[:, 0], v - uv[:, 1])
    dres = z[di] - (np.exp(ls[view[di]]) * dval + b[view[di]])
    rot = [float(np.degrees(np.arccos(np.clip((np.trace(a[:3, :3].T @ q[:3, :3]) - 1) / 2, -1, 1)))) for a, q in zip(c2w, c2w0)]
    mv = c2w[:, :3, 3] - C0
    return {"c2w": c2w, "ls": ls, "b": b, "X": X, "err_px": e, "depth_res_m": dres, "z": z, "s": time.perf_counter() - t0,
            "iterations": it, **lm_info, "at_bound": at_bound,
            "rot_deg": np.asarray(rot), "move_cm": 100 * np.linalg.norm(mv, axis=1), "move_vec_cm": 100 * mv,
            "gauge": {"rot_deg": round(float(np.degrees(np.arccos(np.clip((np.trace(Rg) - 1) / 2, -1, 1)))), 3),
                      "move_cm": round(float(100 * np.linalg.norm(tg + (Rg - np.eye(3)) @ C0.mean(0))), 2)}}


def lm(residual, jacobian, p0, nc, T, bound, cams=True, iters=100):
    """Levenberg-Marquardt with the points eliminated (Schur complement: 3x3 blocks per track, a dense nc x nc camera
    system), IRLS for the soft-L1 on the data rows (the last nc rows are the quadratic priors), camera parameters
    clipped to +-bound after each step. cams=False: only the points move. -> p, iterations, info."""
    from scipy.sparse import diags
    F2 = F_SCALE ** 2

    def cost(r):
        d = r[:-nc]
        return float(np.sum(2 * F2 * (np.sqrt(1 + d * d / F2) - 1)) + np.sum(r[-nc:] ** 2))
    p = p0.copy()
    r = residual(p)
    c = c0 = cost(r)
    lam, it, done = 1e-3, 0, False
    for it in range(1, iters + 1):
        wts = np.ones(len(r))
        wts[:-nc] = 1 / np.sqrt(1 + r[:-nc] ** 2 / F2)
        sw = np.sqrt(wts)
        J = diags(sw) @ jacobian(p)
        rw = sw * r
        Jp = J[:, nc:].tocsc()
        gp = (Jp.T @ rw).reshape(T, 3)
        H = (Jp.T @ Jp).tocoo()
        Hpp = np.zeros((T, 3, 3))
        np.add.at(Hpp, (H.row // 3, H.row % 3, H.col % 3), H.data)
        if cams:
            Jc = J[:, :nc].toarray()
            Hcc, gc = Jc.T @ Jc, Jc.T @ rw
            Hpc = np.asarray(Jp.T @ Jc).reshape(T, 3, nc)
        while True:
            dg = np.einsum("tii->ti", Hpp)
            V = Hpp + (lam * dg + 1e-9)[:, :, None] * np.eye(3)
            Vinv = np.linalg.inv(V)
            if cams:
                Y = np.einsum("tij,tjk->tik", Vinv, Hpc)
                S = Hcc + np.diag(lam * np.diag(Hcc) + 1e-9) - np.einsum("tia,tib->ab", Hpc, Y)
                dc = np.linalg.solve(S, -gc + np.einsum("tia,ti->a", Y, gp))
                dp = -np.einsum("tij,tj->ti", Vinv, gp + np.einsum("tia,a->ti", Hpc, dc))
            else:
                dc, dp = np.zeros(nc), -np.einsum("tij,tj->ti", Vinv, gp)
            pn = p.copy()
            pn[:nc] = np.clip(p[:nc] + dc, -bound, bound)
            pn[nc:] += dp.ravel()
            rn = residual(pn)
            cn = cost(rn)
            if cn < c:
                done = c - cn < 1e-8 * c
                p, r, c, lam = pn, rn, cn, max(lam / 3, 1e-9)
                break
            lam *= 4
            if lam > 1e9:
                done = True
                break
        if done:
            break
    return p, it, {"cost_start": round(c0, 3), "cost_end": round(c, 3), "lambda_end": lam}


def resect(uv, K, c2w0, X, max_nfev=60):
    """One camera's pose from 2D-3D (points fixed), the coarse pose as a prior, the same bounds -> c2w, errors before/after."""
    from scipy.optimize import least_squares
    R0, C0, sr = c2w0[:3, :3], c2w0[:3, 3], np.radians(SIGMA_R_DEG)
    Kn = np.repeat(K[None], len(X), 0)

    def res(p):
        R = so3_exp(p[:3])[0] @ R0
        u, v, _ = proj(np.repeat(R[None], len(X), 0), np.repeat((C0 + p[3:])[None], len(X), 0), Kn, X)
        return np.concatenate([(u - uv[:, 0]) / SIGMA_PX, (v - uv[:, 1]) / SIGMA_PX, p[:3] / sr, p[3:] / SIGMA_T_M])
    hi = np.r_[np.full(3, np.radians(MAX_R_DEG)), np.full(3, MAX_T_M)]
    r = least_squares(res, np.zeros(6), bounds=(-hi, hi), method="trf", loss="soft_l1", f_scale=F_SCALE, max_nfev=max_nfev)
    err = lambda p: np.hypot(*res(p)[:2 * len(X)].reshape(2, -1)) * SIGMA_PX  # noqa: E731
    return moved(c2w0[None], r.x[None, :3], r.x[None, 3:])[0], err(np.zeros(6)), err(r.x), r.x


def field(shape, u, v, r, sigma=FIELD_PX, shrink=FIELD_SHRINK):
    """Dense correction from sparse residuals r at (u, v): Gaussian-weighted mean, pulled to 0 away from them
    (a lone sample keeps 1 / (1 + shrink) of its residual at its own pixel)."""
    import cv2
    h, w = shape
    num, den = np.zeros(shape, np.float64), np.zeros(shape, np.float64)
    ui, vi = np.clip(np.round(u).astype(int), 0, w - 1), np.clip(np.round(v).astype(int), 0, h - 1)
    np.add.at(num, (vi, ui), r)
    np.add.at(den, (vi, ui), 1.)
    k = int(6 * sigma) | 1
    num, den = cv2.GaussianBlur(num, (k, k), sigma), cv2.GaussianBlur(den, (k, k), sigma)
    peak = 1 / (2 * np.pi * sigma ** 2)
    return num / (den + shrink * peak)


def stats(e, q=(50, 90)):
    if len(e) == 0:
        return None
    return {"median": round(float(np.median(e)), 4), "p90": round(float(np.percentile(e, q[1])), 4),
            "rms": round(float(np.sqrt(np.mean(np.square(e)))), 4), "n": int(len(e))}


# ---------- the matcher (container: torch, lightglue) ----------

class Matcher:
    """ALIKED keypoints + LightGlue on the crops (weights via torch.hub into TORCH_HOME, a Modal volume)."""

    def __init__(self, dev, max_kp=2048, thr=.05):
        from lightglue import ALIKED, LightGlue
        self.dev = dev
        self.ext = ALIKED(max_num_keypoints=max_kp, detection_threshold=thr).eval().to(dev)
        self.lg = LightGlue(features="aliked").eval().to(dev)

    def features(self, imgs_bgr):
        import torch
        out = []
        with torch.inference_mode():
            for im in imgs_bgr:
                t = torch.from_numpy(np.ascontiguousarray(im[..., ::-1])).to(self.dev).permute(2, 0, 1).float() / 255
                out.append(self.ext.extract(t, resize=None))  # the crop's own pixels (504), no upsampling
        return out

    def match(self, f0, f1):
        import torch
        with torch.inference_mode():
            r = self.lg({"image0": f0, "image1": f1})
        m = r["matches"][0].cpu().numpy()
        return m[:, 0], m[:, 1]


def verified(k0, k1, i0, i1):
    """Fundamental-matrix MAGSAC at RANSAC_PX -> inlier index pairs."""
    import cv2
    if len(i0) < 12:
        return i0[:0], i1[:0]
    _, mk = cv2.findFundamentalMat(k0[i0], k1[i1], cv2.USAC_MAGSAC, RANSAC_PX, .999, 10000)
    if mk is None:
        return i0[:0], i1[:0]
    mk = mk.ravel().astype(bool)
    return i0[mk], i1[mk]


def in_box(P, box, margin=0.):
    return np.all((P >= np.asarray(box[0]) - margin) & (P <= np.asarray(box[1]) + margin), 1)


def fit_line(z, d, iters=5):
    """Robust z = s d + b (Huber IRLS, 1 cm knee) -> s, b."""
    s_, b_ = float(np.median(z / d)), 0.
    for _ in range(iters):
        r = z - (s_ * d + b_)
        w = np.minimum(1, .01 / np.maximum(np.abs(r), 1e-9))
        A = np.stack([d, np.ones_like(d)], 1) * np.sqrt(w)[:, None]
        s_, b_ = np.linalg.lstsq(A, z * np.sqrt(w), rcond=None)[0]
    return float(s_), float(b_)


def local_ba(m, imgs, depth, K, c2w, kept, box, resect_view=None, tri=False, rerun_da3=None):
    """The spot's crops (kept views) -> depth with scale/shift (and the '+tri' field), refined c2w, info (JSON-able).
    resect_view (img, K, c2w): a held-out crop registered to the solved tracks (its pose only, its depth never used).
    rerun_da3(c2w) -> depth: '+da3', DA3 posed again with the solved cameras, then each view's scale/shift fitted to the
    solved tracks' depths near the spot (all of the view's tracks when < 8 there; the BA depth kept when < 8 at all)."""
    import torch
    t0 = time.perf_counter()
    V = list(kept)
    n = len(V)
    info = {"views": V, "licences": LICENCES}
    feats = m.matcher.features([imgs[i] for i in V] + ([resect_view[0]] if resect_view is not None else []))
    kp = [f["keypoints"][0].cpu().numpy().astype(np.float64) for f in feats]
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    pairs, pinfo = [], []
    for a in range(n):
        for b in range(a + 1, n):
            i0, i1 = m.matcher.match(feats[a], feats[b])
            j0, j1 = verified(kp[a], kp[b], i0, i1)
            pairs.append((a, b, j0, j1))
            pinfo.append({"pair": [V[a], V[b]], "lightglue": int(len(i0)), "magsac_inliers": int(len(j0))})
    t2 = time.perf_counter()
    info["keypoints"] = [int(len(k)) for k in kp[:n]]
    info["pairs"] = pinfo
    view, kpi, track = build_tracks([len(k) for k in kp[:n]], pairs)
    T = int(track.max()) + 1 if len(track) else 0
    Kv, C0 = np.asarray(K, np.float64)[V], np.asarray(c2w, np.float64)[V]
    Dv = [depth[i] for i in V]
    uv, dobs = np.zeros((len(view), 2)), np.full(len(view), np.nan)
    for k in range(n):
        o = view == k
        uv[o] = kp[k][kpi[o]]
        dobs[o] = sample_depth(Dv[k], uv[o, 0], uv[o, 1])
    X0, has = init_points(view, uv, track, Kv, C0, dobs, T) if T else (np.zeros((0, 3)), np.zeros(0, bool))
    if T and not has.all():  # tracks with neither depth nor a triangulation: out
        keep_t = np.flatnonzero(has)
        remap = -np.ones(T, int)
        remap[keep_t] = np.arange(len(keep_t))
        o = has[track]
        view, kpi, track, uv, dobs = view[o], kpi[o], remap[track[o]], uv[o], dobs[o]
        X0, T = X0[keep_t], len(keep_t)
    region = in_box(X0, box, REGION_M) if T else np.zeros(0, bool)
    dsel = np.isfinite(dobs) & region[track] if T else np.zeros(0, bool)
    lens = np.bincount(np.bincount(track)) if T else np.zeros(0, int)
    info.update(tracks=T, observations=int(len(view)), tracks_in_region=int(region.sum()), depth_observations=int(dsel.sum()),
                track_length_hist={int(k): int(v) for k, v in enumerate(lens) if v and k})
    t3 = time.perf_counter()
    if T < 12 or len(set(view.tolist())) < 2:
        info["status"] = f"too few tracks ({T}): coarse cameras kept"
        info["time_s"] = {"features": round(t1 - t0, 3), "match_verify": round(t2 - t1, 3), "tracks": round(t3 - t2, 3), "total": round(t3 - t0, 3)}
        return depth, np.asarray(c2w, np.float64), info
    after = solve(view, uv, track, Kv, C0, dobs, dsel, X0, cams=True)
    t4 = time.perf_counter()
    before = solve(view, uv, track, Kv, C0, dobs, dsel, X0, cams=False)  # diagnostic: outside the product time
    t5 = time.perf_counter()
    info["status"] = "solved"
    info["reprojection_px_at_504_crop"] = {"before": stats(before["err_px"]), "after": stats(after["err_px"]),
                                           "within_2px_before": round(float((before["err_px"] <= 2).mean()), 4),
                                           "within_2px_after": round(float((after["err_px"] <= 2).mean()), 4)}
    info["depth_residual_m"] = {"before": stats(np.abs(before["depth_res_m"])), "after": stats(np.abs(after["depth_res_m"]))}
    info["pose_change"] = {"rot_deg": np.round(after["rot_deg"], 3).tolist(), "move_cm": np.round(after["move_cm"], 2).tolist(),
                           "move_vec_cm": np.round(after["move_vec_cm"], 2).tolist(), "at_bound": after["at_bound"],
                           "gauge_removed": after["gauge"], "note": "after the rigid gauge fit back onto the coarse cameras"}
    info["depth_scale"] = np.round(np.exp(after["ls"]), 4).tolist()
    info["depth_shift_m"] = np.round(after["b"], 4).tolist()
    info["solver"] = {"iterations": after["iterations"], "cost_start": after["cost_start"], "cost_end": after["cost_end"], "s": round(after["s"], 3),
                      "before_iterations": before["iterations"], "before_s": round(before["s"], 3)}
    out_c2w = np.asarray(c2w, np.float64).copy()
    out_d = np.array(depth, np.float32, copy=True)
    for k, i in enumerate(V):
        out_c2w[i] = after["c2w"][k]
        on = out_d[i] > 0
        out_d[i][on] = np.exp(after["ls"][k]) * out_d[i][on] + after["b"][k]
    if rerun_da3 is not None:
        t_d = time.perf_counter()
        D2 = rerun_da3(out_c2w)
        rinfo = []
        for k, i in enumerate(V):
            for sel in ((view == k) & region[track], view == k):
                o = np.flatnonzero(sel)
                d2 = sample_depth(D2[i], uv[o, 0], uv[o, 1])
                ok = np.isfinite(d2)
                if ok.sum() >= 8:
                    break
            if ok.sum() < 8:
                rinfo.append({"view": i, "tracks": int(ok.sum()), "applied": False})
                continue
            s_, b_ = fit_line(after["z"][o][ok], d2[ok])
            on = D2[i] > 0
            out_d[i] = np.where(on, s_ * D2[i] + b_, 0).astype(np.float32)
            res = after["z"][o][ok] - (s_ * d2[ok] + b_)
            rinfo.append({"view": i, "tracks": int(ok.sum()), "near_spot": bool(sel is not None and o.size and region[track[o]].all()),
                          "scale": round(s_, 4), "shift_m": round(b_, 4), "residual_median_abs_m": round(float(np.median(np.abs(res))), 4), "applied": True})
        info["da3_rerun"] = {"views": rinfo, "s": round(time.perf_counter() - t_d, 3)}
    if tri:  # the triangulated tracks near the spot set the depth, DA3 fills between them
        tinfo = []
        for k, i in enumerate(V):
            o = np.flatnonzero((view == k) & region[track])
            if len(o) < 5:
                tinfo.append({"view": i, "tracks": int(len(o)), "applied": False})
                continue
            d_now = sample_depth(out_d[i], uv[o, 0], uv[o, 1], edge=1.)
            ok = np.isfinite(d_now)
            r = after["z"][o][ok] - d_now[ok]
            f = field(out_d[i].shape, uv[o, 0][ok], uv[o, 1][ok], r)
            on = out_d[i] > 0
            out_d[i][on] += f[on].astype(np.float32)
            tinfo.append({"view": i, "tracks": int(ok.sum()), "residual_median_abs_m": round(float(np.median(np.abs(r))), 4) if ok.any() else None,
                          "field_abs_p90_m": round(float(np.percentile(np.abs(f[on]), 90)), 4) if on.any() else None, "applied": True})
        info["tri_field"] = tinfo
    t6 = time.perf_counter()
    if resect_view is not None:  # the held-out crop: matched to the kept crops, its pose from the solved track points
        img_h, K_h, c2w_h = resect_view
        tk = [dict() for _ in range(n)]
        for o in range(len(view)):
            tk[view[o]][int(kpi[o])] = int(track[o])
        uv_h, tr_h = [], {}
        for a in range(n):
            i0, i1 = m.matcher.match(feats[n], feats[a])
            j0, j1 = verified(kp[n], kp[a], i0, i1)
            for p, q in zip(j0, j1):
                t = tk[a].get(int(q))
                if t is not None and t not in tr_h:
                    tr_h[t] = len(uv_h)
                    uv_h.append(kp[n][p])
        rs = {"correspondences": len(uv_h)}
        if len(uv_h) >= 12:
            tr = np.array(list(tr_h.keys()))
            c2w_new, e0, e1, p = resect(np.asarray(uv_h), np.asarray(K_h, np.float64), np.asarray(c2w_h, np.float64), after["X"][tr])
            rs.update(c2w=c2w_new.tolist(), reprojection_px_before=stats(e0), reprojection_px_after=stats(e1),
                      rot_deg=round(float(np.degrees(np.linalg.norm(p[:3]))), 3), move_cm=round(float(100 * np.linalg.norm(p[3:])), 2))
        else:
            rs["status"] = "too few correspondences: coarse camera kept"
        info["resect"] = rs
    t7 = time.perf_counter()
    info["time_s"] = {"features": round(t1 - t0, 3), "match_verify": round(t2 - t1, 3), "tracks": round(t3 - t2, 3), "solve": round(t4 - t3, 3),
                      "apply_depth": round(t6 - t5, 3), "resect": round(t7 - t6, 3), "before_diagnostic": round(t5 - t4, 3),
                      "product": round((t4 - t0) + (t6 - t5) + (t7 - t6), 3)}
    return out_d, out_c2w, info


def self_check():
    rng = np.random.default_rng(3)
    assert np.allclose(so3_exp([[0, 0, np.pi / 2]])[0] @ [1, 0, 0], [0, 1, 0])
    # tracks: 0-1-2 chain -> one track of 3; a view-conflict (two kps of view 0 in one component) -> dropped
    v, k, t = build_tracks([3, 3, 3], [(0, 1, [0, 1], [0, 1]), (1, 2, [0], [2]), (0, 1, [2], [1])])
    got = sorted(zip(t.tolist(), v.tolist(), k.tolist()))
    assert got == [(0, 0, 0), (0, 1, 0), (0, 2, 2)], got
    # a synthetic spot: 400 points on two planes 2-3 m away, 5 cameras on an arc; the coarse cameras 2 deg / 6 cm off,
    # the DA3 depth x1.07 + 3 cm; the solve must bring the cameras back and the scale/shift out
    Kc = np.array([[700., 0, 251.5], [0, 700, 251.5], [0, 0, 1]])
    X = np.concatenate([np.c_[rng.uniform(-.6, .6, 250), rng.uniform(-.6, .6, 250), np.full(250, 2.5)],
                        np.c_[rng.uniform(-.6, .6, 150), np.full(150, .6), rng.uniform(2., 3., 150)]])
    ang = np.radians([-20, -10, 0, 10, 20])
    c2w = np.repeat(np.eye(4)[None], 5, 0)
    for i, a in enumerate(ang):
        R = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
        c2w[i, :3, :3], c2w[i, :3, 3] = R, np.array([0, 0, 2.5]) - R @ [0, 0, 2.5]
    view, track, uv, dobs = [], [], [], []
    for i in range(5):
        u, v_, z = proj(np.repeat(c2w[i:i + 1, :3, :3], len(X), 0), np.repeat(c2w[i:i + 1, :3, 3], len(X), 0), np.repeat(Kc[None], len(X), 0), X)
        view += [i] * len(X)
        track += list(range(len(X)))
        uv.append(np.c_[u, v_] + rng.normal(0, .3, (len(X), 2)))
        dobs.append(1.07 * z + .03)
    view, track, uv, dobs = np.array(view), np.array(track), np.concatenate(uv), np.concatenate(dobs)
    Ks = np.repeat(Kc[None], 5, 0)
    w = rng.normal(0, 1, (5, 3))
    w *= np.radians(2) / np.linalg.norm(w, axis=1, keepdims=True)
    tt = rng.normal(0, 1, (5, 3))
    tt *= .06 / np.linalg.norm(tt, axis=1, keepdims=True)
    coarse = moved(c2w, w, tt)
    X0, has = init_points(view, uv, track, Ks, coarse, dobs, len(X))
    assert has.all()
    dsel = np.ones(len(view), bool)
    b4 = solve(view, uv, track, Ks, coarse, dobs, dsel, X0, cams=False)
    af = solve(view, uv, track, Ks, coarse, dobs, dsel, X0, cams=True)
    # the analytic Jacobian vs central differences, at zero and off zero
    res, jac, p0 = solve(view, uv, track, Ks, coarse, dobs, dsel, X0, fns=True)
    for scale_w in (0., np.radians(1.)):
        pp = p0 + rng.normal(0, 1e-3, len(p0))
        pp[:15] = rng.normal(0, scale_w, 15) if scale_w else 0.
        J = jac(pp).toarray()
        cols = rng.choice(len(p0), 40, replace=False)
        for c in cols:
            e = np.zeros(len(p0))
            e[c] = 1e-6
            fd = (res(pp + e) - res(pp - e)) / 2e-6
            tol = 1e-4
            assert np.abs(fd - J[:, c]).max() <= tol * max(1., np.abs(fd).max()), (c, scale_w, np.abs(fd - J[:, c]).max(), np.abs(fd).max())
    assert np.median(af["err_px"]) < .5 < np.median(b4["err_px"]), (np.median(b4["err_px"]), np.median(af["err_px"]))
    # relative geometry recovered up to one global scale s (the gauge: the camera prior vs the depth-scale prior settle
    # it between 1 and 1.07; the scale is 'estimated' anyway): camera-to-camera distances / s within 1 cm, and every
    # view's depth scale the same (exp(ls) x 1.07 = s)
    dist = lambda C: np.linalg.norm(C[:, None, :3, 3] - C[None, :, :3, 3], axis=-1)  # noqa: E731
    iu = np.triu_indices(5, 1)
    s = float(np.median(dist(af["c2w"])[iu] / dist(c2w)[iu]))
    assert np.abs(dist(af["c2w"])[iu] / s - dist(c2w)[iu]).max() < .01 and 1 <= s <= 1.08, (s, np.abs(dist(af["c2w"])[iu] / s - dist(c2w)[iu]).max())
    assert np.allclose(np.exp(af["ls"]) * 1.07 / s, 1, atol=.01), (np.exp(af["ls"]), s)
    # resect: a camera 2 deg / 6 cm off comes back from its 2D-3D pairs
    c_new, e0, e1, _ = resect(uv[view == 2], Kc, coarse[2], X)
    assert np.median(e1) < .6 < np.median(e0) and np.linalg.norm(c_new[:3, 3] - c2w[2, :3, 3]) < .01
    # field: a lone residual of 5 cm keeps 1/(1+shrink) at its pixel, ~0 far away
    f = field((100, 100), np.array([50.]), np.array([50.]), np.array([.05]), sigma=5)
    assert abs(f[50, 50] - .05 / (1 + FIELD_SHRINK)) < 5e-3 and abs(f[5, 5]) < 1e-6, (f[50, 50], f[5, 5])
    D = np.full((10, 10), 2.)
    D[:, 6:] = 3.
    assert abs(sample_depth(D, np.array([2.5]), np.array([2.5]))[0] - 2) < 1e-9 and np.isnan(sample_depth(D, np.array([5.5]), np.array([2.]))[0])
    print("local_ba self-check ok", {"before_px": round(float(np.median(b4["err_px"])), 3), "after_px": round(float(np.median(af["err_px"])), 3),
                                     "solve_s": round(af["s"], 3)})


if __name__ == "__main__":
    assert sys.argv[1:] == ["--self-check"], __doc__
    self_check()
