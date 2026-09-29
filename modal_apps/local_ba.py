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
SIGMA_LS, SIGMA_B_M, MAX_LS, MAX_B_M = .1, .05, .5, .2  # per-view depth scale / shift
SIGMA_LS_MEAN, SIGMA_LSIG, MAX_LSIG = .005, 1., .3  # the mean depth scale (views with depth data) held at the coarse map's;
# the baseline scale free within x0.74..1.35 (a solve at that bound is refused: the coarse cameras are kept)
START_MAX_PX = 50.
MIN_DEPTH_OBS, OUTLIER_LS = 10, .2  # a view's depth scale counts with >= 10 observations; > 0.2 off the median: its depth is dropped
MAX_TRACKS = 800  # all tracks near the spot, then the longest others
DEPTH_SIG = (.01, .01)  # DA3 depth at a track: 1 cm + 1 % of the depth
REGION_M, F_SCALE, RANSAC_PX = .3, 2., 1.
FIELD_PX, FIELD_SHRINK, TRI_MAX_M = 40., .25, .10
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


def sane(view, uv, track, K, c2w, X, zmin=.1, max_px=START_MAX_PX):
    """Observations whose start point is in front of the view (> 10 cm) and within 50 px of the keypoint: a point behind
    a camera (a DLT start at infinity) makes the projection discontinuous and stalls the solve (run 007: 4 spots)."""
    u, v, z = proj(c2w[view, :3, :3], c2w[view, :3, 3], K[view], X[track])
    return (z > zmin) & (np.hypot(u - uv[:, 0], v - uv[:, 1]) <= max_px)


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


def solve(view, uv, track, K, c2w0, dobs, dsel, X0, cams=True, max_nfev=60, fns=False):
    """Joint solve -> dict(c2w, ls, b, X, residual stats). cams=False: only the points move (the 'before' diagnostic).
    Parameters [w 3n | t 3n | log depth scale n | depth shift n | log baseline scale 1 | X 3T]; camera i's centre is
    c + e^lsig (C0_i - c) + t_i (c the coarse centres' mean): the baseline scale takes any scale disagreement between
    the coarse trajectory and the depth, the mean log depth scale is held at 0 (the coarse map's scale), and the
    bounded t_i are what is left. Residuals [u N | v N | depth Nd | priors 8n + 2], each in sigmas; analytic Jacobian
    (rotation through the SO(3) left Jacobian). The result is put back on the coarse cameras by the rigid transform
    that best maps the solved cameras onto them (applied to cameras and points; reported as 'gauge')."""
    from scipy.sparse import coo_matrix
    n, T, N = len(c2w0), len(X0), len(view)
    c2w0 = np.asarray(c2w0, np.float64)
    R0, C0 = c2w0[:, :3, :3], c2w0[:, :3, 3]
    cbar = C0.mean(0)
    di = np.flatnonzero(dsel)
    Nd = len(di)
    dview = np.bincount(view[di], minlength=n) >= MIN_DEPTH_OBS  # the views whose depth scale is measured
    if not dview.any():
        dview[:] = True
    wmean = dview / dview.sum()
    dval = dobs[di]
    dsig = DEPTH_SIG[0] + DEPTH_SIG[1] * dval
    sr = np.radians(SIGMA_R_DEG)
    prior_sig = np.concatenate([np.full(3 * n, sr), np.full(3 * n, SIGMA_T_M), np.full(n, SIGMA_LS), np.full(n, SIGMA_B_M)])
    nc = 8 * n + 1

    def unpack(p):
        return (p[:3 * n].reshape(n, 3), p[3 * n:6 * n].reshape(n, 3), p[6 * n:7 * n], p[7 * n:8 * n], p[8 * n], p[nc:].reshape(T, 3))

    def cam_frame(p):
        w, t, ls, b, lsig, X = unpack(p)
        R = so3_exp(w) @ R0
        C = cbar + np.exp(lsig) * (C0 - cbar) + t
        Rv, y = R[view], X[track] - C[view]
        return w, t, ls, b, lsig, X, Rv, y, np.einsum("nji,nj->ni", Rv, y)

    def residual(p):
        w, t, ls, b, lsig, X, Rv, y, Xc = cam_frame(p)
        z = Xc[:, 2]
        zs = np.where(z > 1e-3, z, 1e-3)
        u, v = K[view, 0, 0] * Xc[:, 0] / zs + K[view, 0, 2], K[view, 1, 1] * Xc[:, 1] / zs + K[view, 1, 2]
        pred = np.exp(ls[view[di]]) * dval + b[view[di]]
        return np.concatenate([(u - uv[:, 0]) / SIGMA_PX, (v - uv[:, 1]) / SIGMA_PX, (z[di] - pred) / dsig, p[:8 * n] / prior_sig,
                               [ls @ wmean / SIGMA_LS_MEAN, lsig / SIGMA_LSIG]])

    cam = lambda k, vs: np.stack([3 * k * n + 3 * vs + a for a in range(3)], 1)  # noqa: E731  block k (w, t): 3 columns per view
    Xcol = lambda ts: nc + 3 * ts[:, None] + np.arange(3)  # noqa: E731
    G = lambda m: np.full((m, 1), 8 * n)  # noqa: E731  the baseline-scale column
    cols = [np.concatenate([cam(0, view), cam(1, view), G(N), Xcol(track)], 1)] * 2
    cols.append(np.concatenate([cam(0, view[di]), cam(1, view[di]), (6 * n + view[di])[:, None], (7 * n + view[di])[:, None], G(Nd),
                                Xcol(track[di])], 1))
    P0 = 2 * N + Nd
    rows = np.concatenate([np.repeat(np.arange(N), 10), np.repeat(N + np.arange(N), 10), np.repeat(2 * N + np.arange(Nd), 12),
                           P0 + np.arange(8 * n), np.full(n, P0 + 8 * n), [P0 + 8 * n + 1]])
    cols = np.concatenate([c.ravel() for c in cols] + [np.arange(8 * n), 6 * n + np.arange(n), [8 * n]])
    shape = (P0 + 8 * n + 2, nc + 3 * T)

    def jacobian(p):
        w, t, ls, b, lsig, X, Rv, y, Xc = cam_frame(p)
        Rt = Rv.transpose(0, 2, 1)
        z = np.where(Xc[:, 2] > 1e-3, Xc[:, 2], 1e-3)
        dsc = -np.einsum("nij,nj->ni", Rt, np.exp(lsig) * (C0 - cbar)[view])[:, :, None]
        blocks = np.concatenate([Rt @ skew(y) @ left_jac(w)[view], -Rt, dsc, Rt], 2)  # d camera-frame point / d (w, t, lsig, X): (N, 3, 10)
        fx, fy = K[view, 0, 0], K[view, 1, 1]
        zero = np.zeros(N)
        du = np.stack([fx / z, zero, -fx * Xc[:, 0] / z ** 2], 1) / SIGMA_PX
        dv = np.stack([zero, fy / z, -fy * Xc[:, 1] / z ** 2], 1) / SIGMA_PX
        Ju, Jv = np.einsum("nk,nkj->nj", du, blocks), np.einsum("nk,nkj->nj", dv, blocks)
        bz = blocks[di, 2] / dsig[:, None]
        Jd = np.concatenate([bz[:, :6], (-np.exp(ls[view[di]]) * dval / dsig)[:, None], (-1 / dsig)[:, None], bz[:, 6:]], 1)
        vals = np.concatenate([Ju.ravel(), Jv.ravel(), Jd.ravel(), 1 / prior_sig, wmean / SIGMA_LS_MEAN, [1 / SIGMA_LSIG]])
        return coo_matrix((vals, (rows, cols)), shape=shape).tocsr()
    p0 = np.concatenate([np.zeros(nc), np.asarray(X0, np.float64).ravel()])
    if fns:  # the self-check compares the Jacobian with finite differences
        return residual, jacobian, p0
    bound = np.concatenate([np.full(3 * n, np.radians(MAX_R_DEG)), np.full(3 * n, MAX_T_M), np.full(n, MAX_LS), np.full(n, MAX_B_M), [MAX_LSIG]])
    t0 = time.perf_counter()
    try:  # one BLAS thread: the dense parts are 41 x 41; threads only add contention beside vLLM
        from threadpoolctl import threadpool_limits
        limit = threadpool_limits(1)
    except ImportError:
        limit = None
    p, it, lm_info = lm(residual, jacobian, p0, nc, T, bound, cams, max_nfev if cams else 30, n_prior=8 * n + 2)
    if limit is not None:
        limit.unregister()
    w, t, ls, b, lsig, X = unpack(p)
    raw = moved(c2w0, w, np.zeros((n, 3)))
    raw[:, :3, 3] = cbar + np.exp(lsig) * (C0 - cbar) + t
    at_bound = bool(cams and np.any(np.abs(p[:6 * n]) >= .999 * bound[:6 * n]))
    Rg, tg = kabsch(cam_points(raw), cam_points(c2w0))  # the rigid gauge back to the coarse frame
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
            "iterations": it, **lm_info, "at_bound": at_bound, "baseline_scale": float(np.exp(lsig)), "depth_views": dview,
            "scale_at_bound": bool(cams and abs(lsig) >= .999 * MAX_LSIG),
            "rot_deg": np.asarray(rot), "move_cm": 100 * np.linalg.norm(mv, axis=1), "move_vec_cm": 100 * mv,
            "gauge": {"rot_deg": round(float(np.degrees(np.arccos(np.clip((np.trace(Rg) - 1) / 2, -1, 1)))), 3),
                      "move_cm": round(float(100 * np.linalg.norm(tg + (Rg - np.eye(3)) @ C0.mean(0))), 2)}}


def lm(residual, jacobian, p0, nc, T, bound, cams=True, iters=60, tol=1e-4, n_prior=None):  # tol 1e-4: same cameras as 1e-6 within 0.001 deg (bench), 4x fewer iterations
    """Levenberg-Marquardt with the points eliminated (Schur complement: 3x3 blocks per track, a dense nc x nc camera
    system), IRLS for the soft-L1 on the data rows (the last n_prior rows are the quadratic priors), camera parameters kept
    within +-bound by an active set (a parameter that would cross is held at the bound and the rest re-solved).
    cams=False: only the points move. -> p, iterations, info."""
    from scipy.sparse import diags
    F2 = F_SCALE ** 2
    npr = n_prior or nc

    def cost(r):
        d = r[:-npr]
        return float(np.sum(2 * F2 * (np.sqrt(1 + d * d / F2) - 1)) + np.sum(r[-npr:] ** 2))
    p = p0.copy()
    r = residual(p)
    c = c0 = cost(r)
    lam, it, done, evals = 1e-3, 0, False, 1
    for it in range(1, iters + 1):
        wts = np.ones(len(r))
        wts[:-npr] = 1 / np.sqrt(1 + r[:-npr] ** 2 / F2)
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
                rhs = -gc + np.einsum("tia,ti->a", Y, gp)
                act = np.zeros(nc, bool)
                dc = np.linalg.solve(S, rhs)
                for _ in range(4):  # active set on the bounds
                    out = ~act & (np.abs(p[:nc] + dc) > bound)
                    if not out.any():
                        break
                    act |= out
                    dc[act] = np.clip(p[:nc] + dc, -bound, bound)[act] - p[:nc][act]
                    f = ~act
                    dc[f] = np.linalg.solve(S[np.ix_(f, f)], rhs[f] - S[np.ix_(f, act)] @ dc[act])
                dp = -np.einsum("tij,tj->ti", Vinv, gp + np.einsum("tia,a->ti", Hpc, dc))
            else:
                dc, dp = np.zeros(nc), -np.einsum("tij,tj->ti", Vinv, gp)
            pn = p.copy()
            pn[:nc] = np.clip(p[:nc] + dc, -bound, bound)
            pn[nc:] += dp.ravel()
            rn = residual(pn)
            cn = cost(rn)
            evals += 1
            if cn < c:
                done = c - cn < tol * c
                p, r, c, lam = pn, rn, cn, max(lam / 3, 1e-9)
                break
            lam *= 4
            if lam > 1e9:
                done = True
                break
        if done:
            break
    return p, it, {"cost_start": round(c0, 3), "cost_end": round(c, 3), "lambda_end": lam, "evaluations": evals}


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


MVS_PASSES = ((np.linspace(-.06, .06, 25), 7), (np.linspace(-.02, .02, 21), 9))  # (offsets m, NCC window px): coarse, then fine
MVS_NCC_MIN, MVS_PEAK_MIN, MVS_TEX_MIN, MVS_FIELD_PX = .5, .05, .02, 12.


def sweep(g, i, src, D, K, c2w, offsets, win):
    """Plane sweep of view i's depth D (torch H x W) over offsets against the source views' grey images -> accepted
    mask, offset (m), best NCC (the mean of the best two sources, parabolic sub-step)."""
    import torch
    import torch.nn.functional as F
    dev = g.device
    H, W = D.shape
    pool = lambda x: F.avg_pool2d(x[None], win, 1, win // 2, count_include_pad=False)[0]  # noqa: E731
    v, u = torch.meshgrid(torch.arange(H, device=dev, dtype=torch.float32), torch.arange(W, device=dev, dtype=torch.float32), indexing="ij")
    Ki = torch.tensor(K[i], dtype=torch.float32, device=dev)
    ray = torch.stack([(u - Ki[0, 2]) / Ki[0, 0], (v - Ki[1, 2]) / Ki[1, 1], torch.ones_like(u)], -1)
    ref = g[i]
    mr = pool(ref[None])[0]
    vr = (pool((ref * ref)[None])[0] - mr * mr).clamp_min(0)
    maps = []
    for j in src:
        w2c = np.linalg.inv(c2w[j])
        M = torch.tensor(K[j] @ w2c[:3, :3] @ c2w[i][:3, :3], dtype=torch.float32, device=dev)
        B = torch.tensor(K[j] @ (w2c[:3, :3] @ c2w[i][:3, 3] + w2c[:3, 3]), dtype=torch.float32, device=dev)
        maps.append((ray @ M.T, B))
    cost = []
    for o in offsets:
        d = (D + float(o)).clamp_min(.05)[..., None]
        nccs = []
        for (a, B), j in zip(maps, src):
            x = a * d + B
            z = x[..., 2]
            uj, vj = x[..., 0] / z.clamp_min(1e-3), x[..., 1] / z.clamp_min(1e-3)
            w_ = F.grid_sample(g[j][None, None], torch.stack([uj / (W - 1) * 2 - 1, vj / (H - 1) * 2 - 1], -1)[None],
                               align_corners=True, padding_mode="border")[0, 0]
            ok = (z > .05) & (uj >= 0) & (uj <= W - 1) & (vj >= 0) & (vj <= H - 1)
            mw = pool(w_[None])[0]
            vw = (pool((w_ * w_)[None])[0] - mw * mw).clamp_min(0)
            cv = pool((ref * w_)[None])[0] - mr * mw
            nccs.append(torch.where(ok, cv / torch.sqrt(vr * vw + 1e-6), torch.full_like(cv, -1.)))
        cost.append(torch.stack(nccs).topk(min(2, len(src)), 0).values.mean(0))
    C = torch.stack(cost)
    best, k = C.max(0)
    med = C.median(0).values
    kc = k.clamp(1, len(offsets) - 2)
    c0, c1, c2 = (C.gather(0, (kc + a)[None])[0] for a in (-1, 0, 1))
    den = c0 - 2 * c1 + c2
    sub = torch.where(den < -1e-6, .5 * (c0 - c2) / den, torch.zeros_like(den)).clamp(-.5, .5)
    off = float(offsets[0]) + (kc.float() + sub) * float(offsets[1] - offsets[0])
    acc = (best >= MVS_NCC_MIN) & (best - med >= MVS_PEAK_MIN) & (k > 0) & (k < len(offsets) - 1) & (vr.sqrt() >= MVS_TEX_MIN) & (D > 0)
    return acc, off, best


def mvs_refine(dev, imgs, depth, K, c2w, kept, box, margin=.15):
    """'+mvs': depth near the spot from multi-view photometric triangulation through the solved cameras, DA3 only as the
    starting surface. Per kept view, two plane sweeps (+-6 cm in 5 mm, then +-2 cm in 2 mm around the result) against
    the other kept views' grey images: a pixel is accepted where the NCC peak >= 0.5, beats the sweep's median by
    >= 0.05, lies inside the range and the reference patch has texture (grey std >= 0.02). The accepted offsets are
    spread as a Gaussian-weighted field (12 px, shrunk to 0 away from them) over the spot region: the textured pixels
    set the depth, DA3's shape carries it across the textureless ones. -> new depth (n,R,R) float32, per-view stats."""
    import torch
    g = torch.from_numpy(np.stack([im[..., ::-1].astype(np.float32).mean(-1) / 255 for im in imgs])).to(dev)
    n, H, W = g.shape
    out = np.array(depth, np.float32, copy=True)
    stats = []
    for i in kept:
        P, (vv, uu) = backproject_np(depth[i], K[i], c2w[i])
        region = np.zeros((H, W), bool)
        region[vv, uu] = in_box(P, box, margin)
        src = [j for j in kept if j != i]
        st = {"view": i, "region_px": int(region.sum())}
        if region.sum() < 50 or not src:
            stats.append({**st, "accepted_share": [0.]})
            continue
        D = torch.from_numpy(np.ascontiguousarray(depth[i])).to(dev)
        reg = torch.from_numpy(region).to(dev)
        st["accepted_share"], st["field_abs_median_m"], st["best_ncc_median"] = [], [], []
        for offsets, win in MVS_PASSES:
            acc, off, best = sweep(g, i, src, D, K, c2w, offsets, win)
            acc &= reg
            a = acc.cpu().numpy()
            f = field((H, W), *np.nonzero(a)[::-1], off.cpu().numpy()[a], sigma=MVS_FIELD_PX) if a.any() else np.zeros((H, W))
            D = torch.where(reg & (D > 0), D + torch.from_numpy(f.astype(np.float32)).to(dev), D)
            st["accepted_share"].append(round(float(a.sum() / region.sum()), 4))
            st["field_abs_median_m"].append(round(float(np.median(np.abs(f[region]))), 4))
            st["best_ncc_median"].append(round(float(np.median(best.cpu().numpy()[region])), 3))
        out[i] = D.cpu().numpy()
        stats.append(st)
    return out, stats


def consistency(depth, K, c2w, kept, box, cap=20000):
    """Signed disagreement per ordered pair (i -> j): view i's in-box points re-projected into view j, z - depth_j
    there (> 0: behind j's surface); within 10 cm. -> per pair median / MAD, and the median of |median| over pairs
    (a systematic offset between views) against the median |dz| (everything)."""
    rows, allabs = [], []
    for i in kept:
        P, _ = backproject_np(depth[i], K[i], c2w[i])
        P = P[in_box(P, box)]
        if len(P) > cap:
            P = P[np.random.default_rng(i).choice(len(P), cap, replace=False)]
        for j in kept:
            if i == j or not len(P):
                continue
            w2c = np.linalg.inv(c2w[j])
            X = P @ w2c[:3, :3].T + w2c[:3, 3]
            z = X[:, 2]
            u, v = K[j][0, 0] * X[:, 0] / np.maximum(z, 1e-6) + K[j][0, 2], K[j][1, 1] * X[:, 1] / np.maximum(z, 1e-6) + K[j][1, 2]
            h, w = depth[j].shape
            ok = (z > .05) & (u >= 0) & (u <= w - 1) & (v >= 0) & (v <= h - 1)
            dj = depth[j][np.round(v[ok]).astype(int), np.round(u[ok]).astype(int)]
            dz = (z[ok] - dj)[dj > 0]
            dz = dz[np.abs(dz) <= .10]
            if len(dz) < 50:
                continue
            med = float(np.median(dz))
            rows.append({"pair": [int(i), int(j)], "signed_median_m": round(med, 4), "mad_m": round(float(np.median(np.abs(dz - med))), 4), "n": int(len(dz))})
            allabs.append(np.abs(dz))
    if not rows:
        return None
    return {"pairs": rows, "median_abs_pair_offset_m": round(float(np.median([abs(r["signed_median_m"]) for r in rows])), 4),
            "median_pair_mad_m": round(float(np.median([r["mad_m"] for r in rows])), 4),
            "median_abs_dz_m": round(float(np.median(np.concatenate(allabs))), 4)}


def mvs_self_check(dev):
    """GPU: a textured plane 2 m away seen by 3 cameras 20 cm apart; view 0's depth given 3 cm too far must come back
    to within 5 mm (run in the container's warm-up; no torch locally)."""
    import cv2
    rng = np.random.default_rng(0)
    tex = cv2.GaussianBlur(rng.random((1500, 1500)).astype(np.float32), (0, 0), 2)
    tex = (tex - tex.min()) / np.ptp(tex)
    R, f = 504, 500.
    K = np.array([[f, 0, (R - 1) / 2], [0, f, (R - 1) / 2], [0, 0, 1]])
    c2w = np.repeat(np.eye(4)[None], 3, 0)
    c2w[:, 0, 3] = [0., -.2, .2]
    v, u = np.mgrid[:R, :R].astype(np.float32)
    imgs = []
    for c in c2w:  # the plane z = 2: world x, y at 2 mm per texel, centred
        x = (u - K[0, 2]) / f * 2 + c[0, 3]
        y = (v - K[1, 2]) / f * 2
        g = cv2.remap(tex, (x / .002 + 750).astype(np.float32), (y / .002 + 750).astype(np.float32), cv2.INTER_LINEAR)
        imgs.append(np.repeat((g * 255).astype(np.uint8)[..., None], 3, -1))
    depth = np.full((3, R, R), 2., np.float32)
    depth[0] += .03
    out, st = mvs_refine(dev, imgs, depth, np.repeat(K[None], 3, 0), c2w, [0, 1, 2], [np.array([-3, -3, 1.]), np.array([3, 3, 3.])])
    inner = out[0][100:-100, 100:-100]
    err = float(np.median(np.abs(inner - 2)))
    assert err < .005 and st[0]["accepted_share"][0] > .5, (err, st[0])
    return {"median_abs_err_m": round(err, 4), "accepted_share": st[0]["accepted_share"]}


def backproject_np(depth, K, c2w):
    ok = np.isfinite(depth) & (depth > 0)
    v, u = np.nonzero(ok)
    z = depth[v, u]
    X = np.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z], 1)
    return X @ np.asarray(c2w)[:3, :3].T + np.asarray(c2w)[:3, 3], (v, u)


def outlier_views(ls, dview, thr=OUTLIER_LS):
    """Views with measured depth whose log depth scale is > thr from their median: the DA3 crop disagrees in scale."""
    return np.flatnonzero(dview & (np.abs(ls - np.median(ls[dview])) > thr))


def fit_line(z, d, iters=5):
    """Robust z = s d + b (Huber IRLS, 1 cm knee) -> s, b."""
    s_, b_ = float(np.median(z / d)), 0.
    for _ in range(iters):
        r = z - (s_ * d + b_)
        w = np.minimum(1, .01 / np.maximum(np.abs(r), 1e-9))
        A = np.stack([d, np.ones_like(d)], 1) * np.sqrt(w)[:, None]
        s_, b_ = np.linalg.lstsq(A, z * np.sqrt(w), rcond=None)[0]
    return float(s_), float(b_)


def local_ba(m, imgs, depth, K, c2w, kept, box, resect_view=None, tri=False, rerun_da3=None, mvs=False):
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
    if T:  # tracks with neither depth nor a triangulation, or whose start point sits behind / far off a view: out
        ok_o = sane(view, uv, track, Kv, C0, X0) & has[track]
        good = np.bincount(track[ok_o], minlength=T) >= 2
        info["tracks_dropped_at_start"] = int(T - good.sum())
        keep_t = np.flatnonzero(good)
        remap = -np.ones(T, int)
        remap[keep_t] = np.arange(len(keep_t))
        o = ok_o & good[track]
        view, kpi, track, uv, dobs = view[o], kpi[o], remap[track[o]], uv[o], dobs[o]
        X0, T = X0[keep_t], len(keep_t)
    region = in_box(X0, box, REGION_M) if T else np.zeros(0, bool)
    if T > MAX_TRACKS:  # ponytail: a cap for time; the spot's own tracks always stay
        L = np.bincount(track, minlength=T)
        order = np.lexsort((-L, ~region))
        keep_t = np.sort(order[:max(MAX_TRACKS, int(region.sum()))])
        remap = -np.ones(T, int)
        remap[keep_t] = np.arange(len(keep_t))
        o = remap[track] >= 0
        view, kpi, track, uv, dobs = view[o], kpi[o], remap[track[o]], uv[o], dobs[o]
        X0, region, T = X0[keep_t], region[keep_t], len(keep_t)
        info["tracks_before_cap"] = int(len(L))
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
    dv = after["depth_views"]
    bad = outlier_views(after["ls"], dv)
    if len(bad) and len(bad) < dv.sum():
        dsel = dsel & ~np.isin(view, bad)
        after = solve(view, uv, track, Kv, C0, dobs, dsel, X0, cams=True)
    info["depth_dropped_views"] = [V[k] for k in bad] if len(bad) < dv.sum() else []
    t4 = time.perf_counter()
    before = solve(view, uv, track, Kv, C0, dobs, dsel, X0, cams=False)  # diagnostic: outside the product time
    t5 = time.perf_counter()
    info["status"] = "solved"
    if after["scale_at_bound"]:
        info["status"] = f"refused: baseline scale {after['baseline_scale']:.3f} at the x{np.exp(-MAX_LSIG):.2f}..{np.exp(MAX_LSIG):.2f} bound"
    info["reprojection_px_at_504_crop"] = {"before": stats(before["err_px"]), "after": stats(after["err_px"]),
                                           "within_2px_before": round(float((before["err_px"] <= 2).mean()), 4),
                                           "within_2px_after": round(float((after["err_px"] <= 2).mean()), 4)}
    info["depth_residual_m"] = {"before": stats(np.abs(before["depth_res_m"])), "after": stats(np.abs(after["depth_res_m"]))}
    info["pose_change"] = {"rot_deg": np.round(after["rot_deg"], 3).tolist(), "move_cm": np.round(after["move_cm"], 2).tolist(),
                           "move_vec_cm": np.round(after["move_vec_cm"], 2).tolist(), "at_bound": after["at_bound"],
                           "gauge_removed": after["gauge"], "note": "after the rigid gauge fit back onto the coarse cameras"}
    # the physical floor: two-view triangulation depth noise at the spot for 0.5 px of matching error, z^2 dpx / (f B)
    ctr = (np.asarray(box[0]) + np.asarray(box[1])) / 2
    Cs = after["c2w"][:, :3, 3]
    zc = float(np.median(np.linalg.norm(Cs - ctr, axis=1)))
    B = float(np.max(np.linalg.norm(Cs[:, None] - Cs[None], axis=-1)))
    f = float(np.median(Kv[:, 0, 0]))
    info["geometry"] = {"distance_m": round(zc, 3), "baseline_max_m": round(B, 3), "focal_px_504": round(f, 1),
                        "tri_sigma_m_at_0_5px": round(zc ** 2 * .5 / (f * max(B, 1e-6)), 4)}
    info["depth_scale"] = np.round(np.exp(after["ls"]), 4).tolist()
    info["baseline_scale"] = round(after["baseline_scale"], 4)  # > 1: the coarse trajectory is too short for this depth
    info["depth_shift_m"] = np.round(after["b"], 4).tolist()
    info["solver"] = {"iterations": after["iterations"], "cost_start": after["cost_start"], "cost_end": after["cost_end"], "s": round(after["s"], 3),
                      "before_iterations": before["iterations"], "before_s": round(before["s"], 3)}
    out_c2w = np.asarray(c2w, np.float64).copy()
    out_d = np.array(depth, np.float32, copy=True)
    if info["status"] != "solved":  # refused: the numbers above are reported, the coarse cameras and anchor depth kept
        info["time_s"] = {"features": round(t1 - t0, 3), "match_verify": round(t2 - t1, 3), "tracks": round(t3 - t2, 3), "solve": round(t4 - t3, 3),
                          "product": round(t4 - t0, 3)}
        return depth, np.asarray(c2w, np.float64), info
    for k, i in enumerate(V):
        out_c2w[i] = after["c2w"][k]
        on = out_d[i] > 0
        out_d[i][on] = np.exp(after["ls"][k]) * out_d[i][on] + after["b"][k]
        if i in info["depth_dropped_views"]:
            out_d[i][:] = 0  # its pose stays solved (from its matches); its depth goes into no map
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
            d_now = sample_depth(out_d[i], uv[o, 0], uv[o, 1])  # not across a depth edge
            ok = np.isfinite(d_now) & (np.abs(after["z"][o] - np.nan_to_num(d_now)) <= TRI_MAX_M)  # > 10 cm: another surface
            r = after["z"][o][ok] - d_now[ok]
            f = np.clip(field(out_d[i].shape, uv[o, 0][ok], uv[o, 1][ok], r), -TRI_MAX_M / 2, TRI_MAX_M / 2)
            on = out_d[i] > 0
            out_d[i][on] += f[on].astype(np.float32)
            tinfo.append({"view": i, "tracks": int(ok.sum()), "residual_median_abs_m": round(float(np.median(np.abs(r))), 4) if ok.any() else None,
                          "field_abs_p90_m": round(float(np.percentile(np.abs(f[on]), 90)), 4) if on.any() else None, "applied": True})
        info["tri_field"] = tinfo
    if mvs:
        t_m = time.perf_counter()
        out_d, ms = mvs_refine(m.dev, imgs, out_d, np.asarray(K, np.float64), out_c2w, V, box)
        info["mvs"] = {"views": ms, "s": round(time.perf_counter() - t_m, 3)}
    t6 = time.perf_counter()
    info["consistency_before"] = consistency(depth, np.asarray(K, np.float64), np.asarray(c2w, np.float64), V, box)  # diagnostics
    info["consistency_after"] = consistency(out_d, np.asarray(K, np.float64), out_c2w, V, box)
    t6b = time.perf_counter()
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
                      "apply_depth": round(t6 - t5, 3), "resect": round(t7 - t6b, 3), "before_diagnostic": round(t5 - t4, 3),
                      "consistency_diagnostic": round(t6b - t6, 3), "product": round((t4 - t0) + (t6 - t5) + (t7 - t6b), 3)}
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
    assert has.all() and sane(view, uv, track, Ks, coarse, X0).mean() > .98  # 2 deg / 6 cm / 7 % off: a few starts past 50 px
    Xb = X0.copy()
    Xb[0] = [0, 0, -2.]  # behind every camera
    sb, s0 = sane(view, uv, track, Ks, coarse, Xb), sane(view, uv, track, Ks, coarse, X0)
    assert not sb[track == 0].any() and np.array_equal(sb[track != 0], s0[track != 0])
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
    # a view without depth data is not the slack of the mean-scale hold; a view 1.35x off the others is the outlier
    d2 = dobs.copy()
    d2[view == 3] *= 1.35
    ds2 = dsel & (view != 4)
    af2 = solve(view, uv, track, Ks, coarse, d2, ds2, X0, cams=True)
    assert abs(af2["ls"][4]) < .02 and list(outlier_views(af2["ls"], af2["depth_views"])) == [3], (af2["ls"], af2["depth_views"])
    # resect: a camera 2 deg / 6 cm off comes back from its 2D-3D pairs
    c_new, e0, e1, _ = resect(uv[view == 2], Kc, coarse[2], X)
    assert np.median(e1) < .6 < np.median(e0) and np.linalg.norm(c_new[:3, 3] - c2w[2, :3, 3]) < .01
    # field: a lone residual of 5 cm keeps 1/(1+shrink) at its pixel, ~0 far away
    f = field((100, 100), np.array([50.]), np.array([50.]), np.array([.05]), sigma=5)
    assert abs(f[50, 50] - .05 / (1 + FIELD_SHRINK)) < 5e-3 and abs(f[5, 5]) < 1e-6, (f[50, 50], f[5, 5])
    # consistency: a plane seen twice, the second view's depth 2 cm long -> the pair offsets are -+2 cm, no spread
    Kp = np.repeat(np.array([[400., 0, 99.5], [0, 400, 99.5], [0, 0, 1]])[None], 2, 0)
    cp = np.repeat(np.eye(4)[None], 2, 0)
    cp[1, 0, 3] = .2
    dp = np.full((2, 200, 200), 2., np.float32)
    dp[1] += .02
    cons = consistency(dp, Kp, cp, [0, 1], [np.array([-5, -5, 0.]), np.array([5, 5, 5.])])
    assert [r["signed_median_m"] for r in cons["pairs"]] == [-.02, .02] and cons["median_pair_mad_m"] == 0, cons
    D = np.full((10, 10), 2.)
    D[:, 6:] = 3.
    assert abs(sample_depth(D, np.array([2.5]), np.array([2.5]))[0] - 2) < 1e-9 and np.isnan(sample_depth(D, np.array([5.5]), np.array([2.]))[0])
    print("local_ba self-check ok", {"before_px": round(float(np.median(b4["err_px"])), 3), "after_px": round(float(np.median(af["err_px"])), 3),
                                     "solve_s": round(af["s"], 3)})


if __name__ == "__main__":
    assert sys.argv[1:] == ["--self-check"], __doc__
    self_check()
