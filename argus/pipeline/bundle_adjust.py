"""GPL-free bundle adjustment: the production BA of geometry_clean_ab.refine (licence-clean geometry route, 2026-10-06).

Levenberg-Marquardt on the camera / point Schur complement (what Ceres' DENSE_SCHUR does for these 2-3 photo problems), analytic
Jacobians, exact 3x3 point and <= 21x21 camera solves, numpy + scipy.spatial.transform only. No pycolmap, SuiteSparse or Ceres:
the PyPI pycolmap 4.2.1 wheel statically links SuiteSparseQR / CHOLMOD (GPL-2.0-or-later). It reproduces the pycolmap call it
replaced (research-notes/geometry-backbone-ab-2026-10-06, BA table: field values within 0.0004 cm):
  camera      one focal per photo, f0 = (fx + fy) / 2, when focal; else fx, fy fixed. Principal point fixed
  refined     every photo's world-to-camera rotation and translation, its focal (focal=True), every 3D point
  loss        Cauchy(1 px) on each observation's 2D residual: cost = 0.5 sum log(1 + |e|^2), Ceres' cost (comparable to its
              report). LM on the exact gradient with IRLS weights rho'(s)
  cheirality  an observation with camera depth < eps has zero residual and zero Jacobian (COLMAP's reprojection functor);
              point_errors gives it sqrt(DBL_MAX) (COLMAP CalculateSquaredReprojectionError), so refine's 2 px cut drops the point
  gauge       COLMAP 4.2.1 TWO_CAMS_FROM_WORLD: photo 1's pose fixed; photo 2's translation component argmax |t(cam1_from_cam2)| fixed
  stopping    Ceres' three tests at Ceres' default tolerances, on every trial step (accepted or not; each one is an iteration, as
              Ceres counts): gradient max-norm <= GTOL; step norm <= PTOL (|x| + PTOL); |cost change| <= FTOL cost. MAX_ITER trial
              steps without one of them = 'max iterations': converged and usable are False, and refine reports it
  python modal_apps/bundle_adjust.py     self-test (synthetic, asserts)
"""
import time

import numpy as np
from scipy.spatial.transform import Rotation

BACKEND = 'numpy Schur Levenberg-Marquardt (modal_apps/bundle_adjust.py; no pycolmap / SuiteSparse / Ceres)'
CAUCHY = 1.0  # px, pycolmap opts.ceres.loss_function_scale
FTOL, GTOL, PTOL = 1e-6, 1e-10, 1e-8  # Ceres Solver::Options defaults: function_tolerance, gradient_tolerance, parameter_tolerance
MAX_ITER = 500
CONVERGED = ('gradient tolerance', 'parameter tolerance', 'function tolerance')
EPS, BEHIND = np.finfo(float).eps, np.sqrt(np.finfo(float).max)


def _poses(w2cs):
    return np.array([np.r_[Rotation.from_matrix(M[:3, :3]).as_rotvec(), M[:3, 3]] for M in w2cs])


def _project(fxy, c, pose, X, cam, pt):
    R = Rotation.from_rotvec(pose[:, :3]).as_matrix()
    Xc = np.einsum('oij,oj->oi', R[cam], X[pt]) + pose[cam, 3:]
    front = Xc[:, 2] >= EPS
    return Xc[:, :2] / np.where(front, Xc[:, 2], 1.)[:, None] * fxy[cam] + c[cam], front


def point_errors(Ks, w2cs, X, cam, pt, uv):
    """COLMAP Point3D.error: mean reprojection error over the track, sqrt(DBL_MAX) for an observation behind the camera."""
    fxy = np.array([[K[0, 0], K[1, 1]] for K in Ks]); c = np.array([[K[0, 2], K[1, 2]] for K in Ks])
    p, front = _project(fxy, c, _poses(w2cs), X, cam, pt)
    e = np.where(front, np.linalg.norm(p - uv, axis=1), BEHIND)
    return np.bincount(pt, e, len(X)) / np.bincount(pt, minlength=len(X))


def _skew(v):
    S = np.zeros(v.shape[:-1] + (3, 3)); S[..., 0, 1], S[..., 0, 2], S[..., 1, 2] = -v[..., 2], v[..., 1], -v[..., 0]
    return S - np.swapaxes(S, -1, -2)


def bundle_adjust(Ks, w2cs, X, cam, pt, uv, focal, max_iter=MAX_ITER):
    """pycolmap's BA (module doc) on observations uv[o] of point pt[o] in photo cam[o]. Ks / w2cs: lists of 3x3 / 4x4.
    Returns refined Ks, w2cs, X and a summary: cost [initial, final] (Ceres' 0.5 sum rho), iterations, termination, converged,
    usable (= converged with a finite cost no higher than the start)."""
    n, O, P = len(Ks), len(uv), len(X)
    R = Rotation.from_matrix(np.array([M[:3, :3] for M in w2cs])).as_matrix(); t = np.array([M[:3, 3] for M in w2cs], float)
    k = int(np.argmax(np.abs((w2cs[0] @ np.linalg.inv(w2cs[1]))[:3, 3])))  # COLMAP: baseline = t(cam1_from_cam2)
    free = np.ones((n, 7), bool); free[0, :6] = False; free[1, 3 + k] = False; free[:, 6] = focal  # per photo: rotation, translation, focal
    q = int(free.sum()); idx = -np.ones((n, 7), int); idx[free] = np.arange(q); cols = idx[cam]; on = cols >= 0
    c = np.array([[K[0, 2], K[1, 2]] for K in Ks]); fxy = np.array([[K[0, 0], K[1, 1]] for K in Ks], float)
    fxy = np.repeat(fxy.mean(1, keepdims=True), 2, 1) if focal else fxy
    X = np.array(X, float); b = CAUCHY ** 2

    def evaluate(R, t, fxy, X):
        Xc = np.einsum('oij,oj->oi', R[cam], X[pt]) + t[cam]; front = Xc[:, 2] >= EPS; z = np.where(front, Xc[:, 2], 1.)
        p = Xc[:, :2] / z[:, None]; e = np.where(front[:, None], p * fxy[cam] + c[cam] - uv, 0.); s = (e * e).sum(1)
        return .5 * float((b * np.log1p(s / b)).sum()), dict(Xc=Xc, front=front, z=z, p=p, e=e, s=s)

    cost, st = evaluate(R, t, fxy, X); cost0, lam, nu, it, why, tm = cost, 1e-4, 2., 0, 'max iterations', time.monotonic()
    grad, step, linearise = np.inf, np.nan, True
    while it < max_iter:
        if linearise:  # at the start and after every accepted step
            f, z, p, fr = fxy[cam], st['z'], st['p'], st['front'][:, None, None]
            Pj = np.zeros((O, 2, 3)); Pj[:, 0, 0], Pj[:, 1, 1] = f[:, 0] / z, f[:, 1] / z; Pj[:, 0, 2], Pj[:, 1, 2] = -f[:, 0] * p[:, 0] / z, -f[:, 1] * p[:, 1] / z
            Pj *= fr  # behind the camera: zero residual and zero Jacobian (COLMAP's functor)
            JX = Pj @ R[cam]
            J7 = np.concatenate([-Pj @ _skew(st['Xc'] - t[cam]), Pj, (p * fr[:, 0])[:, :, None]], 2)  # d/d(rotation, translation, focal)
            Jc = np.zeros((O, 2, q)); o = np.nonzero(on)[0]; Jc[o, :, cols[on]] = np.swapaxes(J7, 1, 2)[on]
            w = 1 / (1 + st['s'] / b); we = w[:, None] * st['e']  # rho'(s): IRLS weights, exact gradient
            U = np.einsum('o,oai,oaj->ij', w, Jc, Jc); g_c = np.einsum('oai,oa->i', Jc, we)
            V = np.zeros((P, 3, 3)); np.add.at(V, pt, w[:, None, None] * np.swapaxes(JX, 1, 2) @ JX)
            Wm = np.zeros((P, q, 3)); np.add.at(Wm, pt, w[:, None, None] * np.swapaxes(Jc, 1, 2) @ JX)
            g_p = np.zeros((P, 3)); np.add.at(g_p, pt, np.einsum('oai,oa->oi', JX, we))
            grad = max(np.abs(g_c).max(initial=0), np.abs(g_p).max(initial=0))
            if grad <= GTOL:
                why = 'gradient tolerance'; break
            dU, dV = np.clip(np.diag(U), 1e-6, 1e32), np.clip(np.diagonal(V, 0, 1, 2), 1e-6, 1e32)
            # ponytail: |x| as Ceres' state norm (each free rotation a unit quaternion); photo 1's pose is constant, as in Ceres
            xnorm = np.sqrt(n - 1 + (t[1:] ** 2).sum() + focal * (fxy[:, 0] ** 2).sum() + (X ** 2).sum())
        it += 1
        Vi = np.linalg.pinv(V + lam * dV[:, :, None] * np.eye(3), rcond=1e-12, hermitian=True); WVi = Wm @ Vi  # rank-2 block (a view behind): pseudo-inverse, as Ceres
        dc = np.linalg.solve(U + lam * np.diag(dU) - np.einsum('pij,pkj->ik', WVi, Wm), -g_c + np.einsum('pij,pj->i', WVi, g_p))
        dp = -np.einsum('pij,pj->pi', Vi, g_p + np.einsum('pij,i->pj', Wm, dc))
        d7 = np.zeros((n, 7)); d7[free] = dc
        R2 = Rotation.from_rotvec(d7[:, :3]).as_matrix() @ R; t2 = t + d7[:, 3:6]; f2 = fxy + d7[:, 6:7]; X2 = X + dp
        new, st2 = evaluate(R2, t2, f2, X2)
        step = float(np.sqrt(dc @ dc + (dp * dp).sum()))
        if step <= PTOL * (xnorm + PTOL):
            why = 'parameter tolerance'; break
        small = abs(cost - new) <= FTOL * cost
        linearise = new < cost
        if linearise:  # accepted (a decrease at the function tolerance is kept: the lower cost)
            pred = .5 * (lam * ((dU * dc * dc).sum() + (dV * dp * dp).sum()) - g_c @ dc - (g_p * dp).sum())
            rho = (cost - new) / max(pred, 1e-300); lam = max(lam * max(1 / 3, 1 - (2 * rho - 1) ** 3), 1e-16); nu = 2.  # Ceres: radius <= 1e16
            R, t, fxy, X, st, cost = R2, t2, f2, X2, st2, new
        else:
            lam *= nu; nu *= 2
        if small:
            why = 'function tolerance'; break
    converged = why in CONVERGED
    usable = bool(converged and np.isfinite(cost) and cost <= cost0)
    Ks = [np.array([[fxy[i, 0], 0, c[i, 0]], [0, fxy[i, 1], c[i, 1]], [0, 0, 1.]]) for i in range(n)]
    w2cs = [np.r_[np.c_[R[i], t[i]], [[0, 0, 0, 1.]]] for i in range(n)]
    info = dict(report=f'{BACKEND}: iterations {it}, initial cost {cost0:.6e}, final cost {cost:.6e}, termination {why}', converged=converged,
                usable=usable, termination=why, cost=[cost0, cost], iterations=it, gradientInf=float(grad), lastStepNorm=step,
                tolerances=dict(function=FTOL, gradient=GTOL, parameter=PTOL, maxIterations=max_iter), seconds=time.monotonic() - tm,
                gaugeFixedTranslationDim=k, parameters=q + 3 * P, residuals=2 * O)
    return Ks, w2cs, X, info


# ---------------------------------------------------------------- self-test
def _check():
    """Synthetic floor + wall seen by three cameras (geometry_clean_ab._check's scene), every pair matched as 2-view tracks:
    exact recovery; Cauchy cost and pose / focal accuracy with noise + outliers; convergence reporting; cheirality."""
    rng = np.random.default_rng(0); F = 400.

    def look(C, target):
        z = target - C; z /= np.linalg.norm(z); x = np.cross(z, [0, 0, 1.]); x /= np.linalg.norm(x); y = np.cross(z, x)
        M = np.eye(4); M[:3, :3] = np.c_[x, y, z]; M[:3, 3] = C; return M
    true = [look(np.array(C, float), np.array([0., 3, .5])) for C in ([0, 0, 1.5], [1.2, .3, 1.5], [-1., .5, 1.4])]
    w2c_true = [np.linalg.inv(M) for M in true]
    K_true = [np.array([[F, 0, 258.5], [0, F, 258.5], [0, 0, 1]])] * 3
    P = np.r_[np.c_[rng.uniform(-2, 2, 3000), rng.uniform(1.5, 4.5, 3000), np.zeros(3000)], np.c_[rng.uniform(-2, 2, 3000), np.full(3000, 4.5), rng.uniform(0, 2, 3000)]]
    depth = lambda M, X: (X - M[:3, 3]) @ M[:3, 2]
    proj = lambda M, X: ((X - M[:3, 3]) @ M[:3, :3])[:, :2] / depth(M, X)[:, None] * F + 258.5
    jitter = lambda M, deg, m: np.r_[np.c_[Rotation.from_rotvec(rng.normal(size=3) * np.radians(deg)).as_matrix(), rng.normal(size=3) * m], [[0, 0, 0, 1.]]] @ M
    rot_deg = lambda A, B: float(np.degrees(np.linalg.norm(Rotation.from_matrix(A.T @ B).as_rotvec())))

    def gauge_free(c2ws):  # relative rotations and baseline directions (in camera i), and the baseline-length ratio
        out = {}
        for i, j in ((0, 1), (0, 2), (1, 2)):
            bb = c2ws[i][:3, :3].T @ (c2ws[j][:3, 3] - c2ws[i][:3, 3]); out[i, j] = (c2ws[i][:3, :3].T @ c2ws[j][:3, :3], bb / np.linalg.norm(bb))
        return out, np.linalg.norm(c2ws[2][:3, 3] - c2ws[0][:3, 3]) / np.linalg.norm(c2ws[1][:3, 3] - c2ws[0][:3, 3])
    rt, ratio_t = gauge_free(true)

    def poses_ok(w2cs, rot, base, rat):
        ro, ratio = gauge_free([np.linalg.inv(M) for M in w2cs])
        return all(rot_deg(rt[k][0], ro[k][0]) < rot and np.degrees(np.arccos(min(1, rt[k][1] @ ro[k][1]))) < base for k in rt) and abs(ratio / ratio_t - 1) < rat

    def tracks(noise, outliers):  # -> cam, pt, uv, backbone-like initial points, exact points
        cam, pt, uv, Xs, Xe = [], [], [], [], []
        for i, j in ((0, 1), (0, 2), (1, 2)):
            ui, uj = proj(true[i], P), proj(true[j], P)
            ok = np.all([(u > 2).all(1) & (u < 515).all(1) for u in (ui, uj)], 0) & (depth(true[i], P) > 0) & (depth(true[j], P) > 0)
            ui, uj = ui[ok] + rng.normal(scale=noise, size=(ok.sum(), 2)), uj[ok] + rng.normal(scale=noise, size=(ok.sum(), 2))
            bad = rng.random(ok.sum()) < outliers; uj[bad] += rng.uniform(-30, 30, (bad.sum(), 2))
            p = sum(map(len, Xs)) + np.arange(ok.sum()); Xs.append(P[ok] + rng.normal(scale=.03, size=(ok.sum(), 3))); Xe.append(P[ok])
            cam += [np.full(ok.sum(), i), np.full(ok.sum(), j)]; pt += [p, p]; uv += [ui, uj]
        return np.concatenate(cam), np.concatenate(pt), np.concatenate(uv), np.concatenate(Xs), np.concatenate(Xe)
    init = [w2c_true[0]] + [np.linalg.inv(jitter(M, .8, .04)) for M in true[1:]]
    K0 = [np.diag([F * f, F * f, 1]) + np.c_[np.zeros((3, 2)), [258.5, 258.5, 0]] for f in (1.03, 1.06, .97)]
    # 1) exact observations: poses (gauge-free), baseline ratio and every focal come back; converged by a Ceres test
    cam, pt, uv, X, _ = tracks(0., 0.)
    Ks, w2cs, Xr, info = bundle_adjust(K0, init, X, cam, pt, uv, focal=True)
    assert info['converged'] and info['usable'] and info['termination'] in CONVERGED, info
    assert poses_ok(w2cs, 1e-4, 1e-4, 1e-6) and all(abs(Kr[0, 0] / F - 1) < 1e-6 for Kr in Ks), ([Kr[0, 0] for Kr in Ks], info)
    e = point_errors(Ks, w2cs, Xr, cam, pt, uv); behind = e > 1e100  # a point stepped behind a camera has zero residual (COLMAP)
    assert info['cost'][1] < 1e-12 and e[~behind].max() < 1e-5 and behind.mean() < 1e-3, (info, e[~behind].max(), behind.sum())
    k = info['gaugeFixedTranslationDim']  # gauge: photo 1 untouched, photo 2's fixed translation component untouched
    assert np.allclose(w2cs[0], init[0], atol=1e-12) and abs(w2cs[1][k, 3] - init[1][k, 3]) < 1e-12
    # 2) 0.3 px noise + 5 % gross outliers: poses and focals within geometry_clean_ab._check's tolerances (Cauchy), cost =
    #    Ceres' 0.5 sum log(1 + |e|^2); the same run cut at 3 trial steps reports 'max iterations', not converged, not usable
    cam, pt, uv, X, _ = tracks(.3, .05)
    Ks, w2cs, Xr, info = bundle_adjust(K0, init, X, cam, pt, uv, focal=True)
    assert info['converged'] and info['usable'] and info['iterations'] < MAX_ITER, info
    assert poses_ok(w2cs, .05, .3, 5e-3) and all(abs(Kr[0, 0] / F - 1) < 5e-3 for Kr in Ks), ([Kr[0, 0] for Kr in Ks], info)
    fxy = np.array([[Kr[0, 0]] * 2 for Kr in Ks]); s = (np.square(_project(fxy, np.full((3, 2), 258.5), _poses(w2cs), Xr, cam, pt)[0] - uv)).sum(1)
    assert abs(.5 * np.log1p(s).sum() / info['cost'][1] - 1) < 1e-9, (.5 * np.log1p(s).sum(), info['cost'])
    cut = bundle_adjust(K0, init, X, cam, pt, uv, focal=True, max_iter=3)[3]
    assert (cut['termination'], cut['converged'], cut['usable'], cut['iterations']) == ('max iterations', False, False, 3), cut
    # 3) cheirality (COLMAP: an observation behind its camera has zero residual). 20 extra tracks whose point is in front of
    #    photo 1 and behind photo 2, seen exactly in photo 1 and at an arbitrary pixel in photo 2:
    #    a) at the true cameras and points the cost is exactly 0; one point moved across photo 2's image plane to depth +0.05
    #       adds exactly its photo-2 term 0.5 log(1 + |e|^2), and back behind it adds nothing again
    #    b) from the jittered start the BA converges to the true cameras and focals at zero cost, the 20 still behind photo 2
    #    c) point_errors marks exactly the 20 (sqrt(DBL_MAX)), so refine's 2 px cut drops them
    cam, pt, uv, X, Xe = tracks(0., 0.)
    Q = rng.uniform([1.5, -.5, .8], [4., 2., 2.2], (20000, 3)); Q = Q[(depth(true[0], Q) > .3) & (depth(true[1], Q) < -.2)][:20]
    assert len(Q) == 20, len(Q)
    ids = len(X) + np.arange(20); far = rng.uniform(0, 518, (20, 2))
    cam, pt, uv = np.r_[cam, np.zeros(20, int), np.ones(20, int)], np.r_[pt, ids, ids], np.r_[uv, proj(true[0], Q), far]
    X, Xe = np.r_[X, Q], np.r_[Xe, Q]
    at = lambda Xp: bundle_adjust(K_true, w2c_true, Xp, cam, pt, uv, focal=True, max_iter=0)[3]['cost'][0]
    assert at(Xe) < 1e-18, at(Xe)
    C1, d1, z2 = true[0][:3, 3], Q[0] - true[0][:3, 3], true[1][:3, 2]
    on_ray = lambda dz: C1 + (dz - (C1 - true[1][:3, 3]) @ z2) / (d1 @ z2) * d1  # photo 1's ray through Q[0], at photo-2 depth dz
    for dz in (.05, -.05):
        Xc = Xe.copy(); Xc[ids[0]] = on_ray(dz)
        term = .5 * np.log1p(np.sum((proj(true[1], Xc[ids[:1]]) - far[:1]) ** 2)) if dz > 0 else 0.
        assert abs(at(Xc) - term) < 1e-9 and (term > .5 or dz < 0), (dz, at(Xc), term)
    Ks, w2cs, Xr, info = bundle_adjust(K0, init, X, cam, pt, uv, focal=True)
    assert info['converged'] and info['cost'][1] < 1e-12 and poses_ok(w2cs, 1e-4, 1e-4, 1e-6), info
    assert all(abs(Kr[0, 0] / F - 1) < 1e-6 for Kr in Ks) and (depth(np.linalg.inv(w2cs[1]), Xr[ids]) < 0).all(), [Kr[0, 0] for Kr in Ks]
    e = point_errors(Ks, w2cs, Xr, cam, pt, uv)
    assert (e[ids] > BEHIND / 4).all() and np.delete(e, ids)[np.delete(e, ids) < 1e100].max() < 1e-5, (e[ids].min(), info)
    print('bundle_adjust self-test passed', {k: v for k, v in info.items() if k in ('iterations', 'termination', 'cost')}, round(info['seconds'], 1), 's')


if __name__ == '__main__':
    _check()
