"""Research reference for the GPL-free bundle adjustment of the licence-clean MVS route (2026-10-06).

pycolmap 4.2.1's PyPI wheel statically links SuiteSparseQR and CHOLMOD (GPL-2.0-or-later, VERIFY.md). The production BA that
replaced it now lives in the kit: worktree modal_apps/bundle_adjust.py (numpy Levenberg-Marquardt with the camera / point Schur
complement; Ceres' default tolerances; cheirality and convergence self-tests), called by geometry_clean_ab.refine. This module
keeps only what the research harnesses need on top of it:
  bundle_adjust_lsq  reference: scipy.optimize.least_squares (trf + lsmr, sparse finite-difference Jacobian) on the same
                     objective, gauge and cheirality rule. The LM's solution is a minimum of it (self-test). On the real 090
                     problems it is not usable as the stage (Modal, 2026-10-06): the DA3-BASE start hit the 3600 s timeout, the
                     MoGe start's pass 1 stopped at 1000 evaluations unconverged (gradient 2 243) and dropped 2 509 points; 030
                     converged in ~30 s
  refine             = geometry_clean_ab.refine (the production stage; ba_scipy_modal.py and older harnesses call bs.refine)
  bundle_adjust, point_errors, CAUCHY, EPS, BEHIND   re-exported from bundle_adjust

  python ba_scipy.py     LM vs least_squares on the synthetic 3-camera scene (asserts); the production self-tests are
                         python modal_apps/bundle_adjust.py and python modal_apps/geometry_clean_ab.py check
"""
from itertools import combinations
import os
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import csr_matrix
from scipy.spatial.transform import Rotation

sys.path[:0] = [str(Path(__file__).resolve().parent), os.environ.get('PANOPTES_WORKCELL', '/Users/adam/.codex/worktrees/panoptes-workcell-photo-speed') + '/modal_apps']
import geometry_clean_ab as gc  # noqa: E402  helpers; refine = the production stage (numpy BA, no pycolmap)
from bundle_adjust import BEHIND, CAUCHY, EPS, _project, bundle_adjust, point_errors  # noqa: E402,F401  the production BA

refine = gc.refine
MAX_NFEV = 1000  # reference: trf's subspace steps progress less per evaluation than Ceres' (self-test pass 1 needed ~400)
TOL = 1e-10  # ftol / xtol / gtol
LSMR = dict(atol=1e-12, btol=1e-12)  # lsmr's default 1e-6 gives inexact Gauss-Newton steps that stall far from the minimum (self-test)


def bundle_adjust_lsq(Ks, w2cs, X, cam, pt, uv, focal, max_nfev=MAX_NFEV, tol=TOL):
    """Reference solver (module doc): scipy least_squares on the same objective, gauge and cheirality rule."""
    n, O = len(Ks), len(uv)
    pose0 = np.array([np.r_[Rotation.from_matrix(M[:3, :3]).as_rotvec(), M[:3, 3]] for M in w2cs])
    k = int(np.argmax(np.abs((w2cs[0] @ np.linalg.inv(w2cs[1]))[:3, 3])))  # COLMAP: baseline = t(cam1_from_cam2)
    free = np.ones((n, 6), bool); free[0] = False; free[1, 3 + k] = False
    c = np.array([[K[0, 2], K[1, 2]] for K in Ks]); fxy0 = np.array([[K[0, 0], K[1, 1]] for K in Ks])
    nf, npose = (n if focal else 0), int(free.sum())
    x0 = np.r_[fxy0.mean(1) if focal else [], pose0[free], np.asarray(X, float).ravel()]
    b = CAUCHY ** 2

    def unpack(x):
        pose = pose0.copy(); pose[free] = x[nf:nf + npose]
        return (np.repeat(x[:n, None], 2, 1) if focal else fxy0), pose, x[nf + npose:].reshape(-1, 3)

    def fun(x):
        fxy, pose, Xp = unpack(x)
        p, front = _project(fxy, c, pose, Xp, cam, pt)
        e = np.where(front[:, None], p - uv, 0.); s = (e * e).sum(1)
        w = np.sqrt(np.divide(b * np.log1p(s / b), s, out=np.ones_like(s), where=s > 0))
        return (e * w[:, None]).ravel()

    pidx = -np.ones((n, 6), int); pidx[free] = nf + np.arange(npose)
    cols = np.concatenate([cam[:, None] if focal else np.empty((O, 0), int), pidx[cam], nf + npose + 3 * pt[:, None] + np.arange(3)], 1)
    rows = np.repeat(np.arange(O)[:, None], cols.shape[1], 1); ok = cols >= 0
    r, cc = rows[ok], cols[ok]
    A = csr_matrix((np.ones(2 * len(r)), (np.r_[2 * r, 2 * r + 1], np.r_[cc, cc])), shape=(2 * O, len(x0)))
    t = time.monotonic(); cost0 = .5 * float(fun(x0) @ fun(x0))
    sol = least_squares(fun, x0, jac_sparsity=A, method='trf', tr_solver='lsmr', x_scale='jac', ftol=tol, xtol=tol, gtol=tol, max_nfev=max_nfev,
                        tr_options=LSMR)
    fxy, pose, Xp = unpack(sol.x)
    Ks = [np.array([[fxy[i, 0], 0, c[i, 0]], [0, fxy[i, 1], c[i, 1]], [0, 0, 1.]]) for i in range(n)]
    w2cs = [np.r_[np.c_[Rotation.from_rotvec(pose[i, :3]).as_matrix(), pose[i, 3:]], [[0, 0, 0, 1.]]] for i in range(n)]
    info = dict(report=f'scipy least_squares trf/lsmr: nfev {sol.nfev}, njev {sol.njev}, initial cost {cost0:.6e}, final cost {sol.cost:.6e}, status {sol.status}',
                usable=bool(sol.status > 0), termination=f'status {sol.status}: {sol.message}', cost=[cost0, float(sol.cost)], iterations=int(sol.nfev), gradientInf=float(sol.optimality),
                seconds=time.monotonic() - t, gaugeFixedTranslationDim=k, parameters=len(x0), residuals=2 * O)
    return Ks, w2cs, Xp, info


# ---------------------------------------------------------------- self-test
def _check():
    """The production LM ends at a minimum of the least_squares reference: synthetic floor + wall seen by three cameras
    (geometry_clean_ab._check's scene), every pair matched as 2-view tracks, 0.3 px noise, no outliers. Restarted at the LM's
    solution, the reference finds no lower cost and leaves cameras and focals in place. (From the far start the reference can
    step a few points behind a camera, where their residual is zero, so the two runs from the start are not compared by cost.)"""
    rng = np.random.default_rng(0); F = 400.; K = np.array([[F, 0, 258.5], [0, F, 258.5], [0, 0, 1]])

    def look(C, target):
        z = target - C; z /= np.linalg.norm(z); x = np.cross(z, [0, 0, 1.]); x /= np.linalg.norm(x); y = np.cross(z, x)
        M = np.eye(4); M[:3, :3] = np.c_[x, y, z]; M[:3, 3] = C; return M
    true = [look(np.array(c, float), np.array([0., 3, .5])) for c in ([0, 0, 1.5], [1.2, .3, 1.5], [-1., .5, 1.4])]
    P = np.r_[np.c_[rng.uniform(-2, 2, 3000), rng.uniform(1.5, 4.5, 3000), np.zeros(3000)], np.c_[rng.uniform(-2, 2, 3000), np.full(3000, 4.5), rng.uniform(0, 2, 3000)]]
    jitter = lambda M, deg, m: np.r_[np.c_[Rotation.from_rotvec(rng.normal(size=3) * np.radians(deg)).as_matrix(), rng.normal(size=3) * m], [[0, 0, 0, 1.]]] @ M
    cam, pt, uv, Xs = [], [], [], []
    for i, j in combinations(range(3), 2):
        ui, uj = gc.project(K, true[i], P), gc.project(K, true[j], P)
        ok = np.all([(u > 2).all(1) & (u < 515).all(1) for u in (ui, uj)], 0) & ((P - true[i][:3, 3]) @ true[i][:3, 2] > 0) & ((P - true[j][:3, 3]) @ true[j][:3, 2] > 0)
        ui, uj = ui[ok] + rng.normal(scale=.3, size=(ok.sum(), 2)), uj[ok] + rng.normal(scale=.3, size=(ok.sum(), 2))
        p = sum(map(len, Xs)) + np.arange(ok.sum()); Xs.append(P[ok] + rng.normal(scale=.03, size=(ok.sum(), 3)))  # backbone-like init
        cam += [np.full(ok.sum(), i), np.full(ok.sum(), j)]; pt += [p, p]; uv += [ui, uj]
    cam, pt, uv, X = np.concatenate(cam), np.concatenate(pt), np.concatenate(uv), np.concatenate(Xs)
    init = [true[0]] + [jitter(M, .8, .04) for M in true[1:]]
    K0 = [np.diag([F * f, F * f, 1]) + np.c_[np.zeros((3, 2)), [258.5, 258.5, 0]] for f in (1.03, 1.06, .97)]
    import bundle_adjust as ba
    saved = ba.FTOL, ba.GTOL, ba.PTOL
    ba.FTOL, ba.GTOL, ba.PTOL = 0., 1e-8, 0.  # precision rule (the 2026-10-06 scipyba run's): as close to the minimum as the LM gets
    try:
        Ks, w2cs, Xr, info = bundle_adjust(K0, [np.linalg.inv(M) for M in init], X, cam, pt, uv, focal=True)
    finally:
        ba.FTOL, ba.GTOL, ba.PTOL = saved
    Ks2, w2cs2, _, info2 = bundle_adjust_lsq(Ks, w2cs, Xr, cam, pt, uv, focal=True)
    assert info2['cost'][1] > info['cost'][1] * (1 - 1e-7) and max(abs(a[0, 0] - b[0, 0]) for a, b in zip(Ks, Ks2)) < 1e-3, (info, info2)
    assert max(gc.rot_deg(a[:3, :3].T @ b[:3, :3]) for a, b in zip(w2cs, w2cs2)) < 1e-4, 'LM and least_squares disagree on the cameras'
    assert max(np.abs(a[:3, 3] - b[:3, 3]).max() for a, b in zip(w2cs, w2cs2)) < 1e-5, 'LM and least_squares disagree on the cameras'
    # the production rule (Ceres' default tolerances) stops converged, within its function tolerance of that minimum
    prod = bundle_adjust(K0, [np.linalg.inv(M) for M in init], X, cam, pt, uv, focal=True)[3]
    assert prod['converged'] and prod['cost'][1] < info2['cost'][1] * (1 + 10 * ba.FTOL), (prod, info2['cost'])
    print('ba_scipy self-test passed; LM (precision rule) vs least_squares (cost, cost, s, s)', (info['cost'][1], info2['cost'][1], round(info['seconds'], 1), round(info2['seconds'], 1)),
          '| production rule', prod['termination'], prod['iterations'], f"relative cost above the minimum {prod['cost'][1] / info2['cost'][1] - 1:.1e}")


if __name__ == '__main__':
    _check()
