"""r5b (models): what a display model may call 'observed', one rule for every model (a primitive's faces, a cylinder's arc
sectors and caps, an open frame's members, a generated mesh's vertices): a sample of its surface is observed only when

  faced   a camera that saw the object looked at it within FACE_DEG of its outward normal (the ray from the sample to the
          camera less than 75 deg off the normal: the back of a panel, a far face never counts),
  clear   that camera's depth there is not nearer than the sample by more than the tolerance (nothing, the object's own front
          included, stood between the camera and it),
  on      an observed point of the object lies within the tolerance of it (the video saw a surface there, not a guess).

Everything else is drawn as guessed. A face, sector or member is observed when >= SEEN_SHARE of its samples are.

    python -m fast_report.observed --self-check
"""
import numpy as np

FACE_DEG, SEEN_SHARE = 75., .25
TOL_M, TOL_REL = .03, .03  # the depth test's and the point test's tolerance: max(3 cm, 3 % of the camera distance)
MAX_VIEWS = 12


def spread(views, n=MAX_VIEWS):
    """Up to n of the object's views, evenly over their order (the first and the last kept)."""
    views = list(views)
    if len(views) <= n:
        return views
    return [views[i] for i in np.unique(np.round(np.linspace(0, len(views) - 1, n)).astype(int))]


def seen(P, N, views, pts, tol_m=TOL_M, tol_rel=TOL_REL):
    """P (n, 3) surface samples and N (n, 3) their outward normals in the shot's frame; views [(c2w (4, 4), K (3, 3) at the depth
    map's grid, depth (h, w) m or None)]; pts (m, 3) the object's observed points (same frame). -> (n,) bool (module docstring)."""
    from scipy.spatial import cKDTree
    P, N = np.asarray(P, float).reshape(-1, 3), np.asarray(N, float).reshape(-1, 3)
    n = len(P)
    if not n or not len(views):
        return np.zeros(n, bool)
    N = N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)
    cos_face = np.cos(np.radians(FACE_DEG))
    clear, dist = np.zeros(n, bool), np.full(n, np.inf)
    for c2w, K, depth in views:
        c2w, K = np.asarray(c2w, float), np.asarray(K, float)
        to_cam = c2w[:3, 3] - P
        d = np.linalg.norm(to_cam, axis=1)
        dist = np.minimum(dist, d)
        faced = (to_cam * N).sum(1) >= cos_face * np.maximum(d, 1e-9)
        X = (P - c2w[:3, 3]) @ c2w[:3, :3]
        z = X[:, 2]
        ok = faced & (z > .05)
        if depth is None or not ok.any():
            continue
        h, w = depth.shape[-2:]
        u = K[0, 0] * X[ok, 0] / z[ok] + K[0, 2]
        v = K[1, 1] * X[ok, 1] / z[ok] + K[1, 2]
        iu, iv = np.floor(u).astype(int), np.floor(v).astype(int)
        inb = (iu >= 0) & (iv >= 0) & (iu < w) & (iv < h)
        idx = np.flatnonzero(ok)[inb]
        D = np.asarray(depth[iv[inb], iu[inb]], float)
        good = np.isfinite(D) & (D > 0) & (z[idx] <= D + np.maximum(tol_m, tol_rel * D))
        clear[idx[good]] = True
    pts = np.asarray(pts, float).reshape(-1, 3)
    if not len(pts):
        return np.zeros(n, bool)
    near = cKDTree(pts).query(P, distance_upper_bound=float(np.max(np.maximum(tol_m, tol_rel * np.where(np.isfinite(dist), dist, 0)))))[0]
    on = near <= np.maximum(tol_m, tol_rel * np.where(np.isfinite(dist), dist, 0))
    return clear & on


def vertex_normals(V, F):
    """Area-weighted vertex normals, oriented outward from the mesh's centroid where the winding is inconsistent."""
    V, F = np.asarray(V, float), np.asarray(F, np.int64)
    Nv = np.zeros_like(V)
    if len(F):
        fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
        for k in range(3):
            np.add.at(Nv, F[:, k], fn)
    Nv /= np.maximum(np.linalg.norm(Nv, axis=1, keepdims=True), 1e-12)
    out = (V - V.mean(0))
    flip = (Nv * out).sum(1) < 0
    if flip.mean() > .5:  # the generator wound its faces inward
        Nv = -Nv
    return Nv


def mesh_seen(V, F, views, pts, **kw):
    """Per vertex of a mesh (the shot's frame): observed by the module's rule."""
    return seen(V, vertex_normals(V, F), views, pts, **kw)


def gate_seen(src, key, vertices, faces):
    """A generated mesh placed in a staged shot (sam3d.FastSource: the gate processes' view of the report): per vertex observed by
    the module's rule over the object's own keyframes (their DA3 depth, K and cameras) and its observed points (x7's lift of its
    masks). -> (n,) bool."""
    from fast_report.x7 import observed_points
    frames = spread(src.objects[key]["frames"])
    views = [(np.asarray(src.rows[f]["c2w"], float), np.asarray(src.clip.k_raster, float), np.asarray(src.rows[f]["mono"], np.float32))
             for f in frames if f in src.rows]
    return mesh_seen(vertices, faces, views, observed_points(src, key, frames, cap=20000))


def self_check():
    # a 1 x 1 m panel facing -y at y = 0, a camera 3 m in front of it; the depth map is the panel itself
    K = np.array([[250., 0, 126], [0, 250., 70], [0, 0, 1]])

    def look(eye, target):
        z = np.asarray(target, float) - eye
        z /= np.linalg.norm(z)
        x = np.cross(z, [0, 0, 1.])
        x /= np.linalg.norm(x)
        c2w = np.eye(4)
        c2w[:3, :3], c2w[:3, 3] = np.stack([x, np.cross(z, x), z], 1), eye
        return c2w
    c2w = look(np.array([0., -3., 0.]), [0, 0, 0])
    depth = np.full((140, 252), 3.)  # the panel's plane, as the camera sees it (approx.)
    g = np.stack(np.meshgrid(np.linspace(-.5, .5, 11), np.linspace(-.5, .5, 11)), -1).reshape(-1, 2)
    front = np.c_[g[:, 0], np.zeros(len(g)), g[:, 1]]
    pts = front + np.random.default_rng(0).normal(0, .003, front.shape)
    s_front = seen(front, np.tile([0, -1., 0], (len(g), 1)), [(c2w, K, depth)], pts)
    back = front + [0, .05, 0]  # the back face of a 5 cm slab: faced away, and behind the front's depth
    s_back = seen(back, np.tile([0, 1., 0], (len(g), 1)), [(c2w, K, depth)], pts)
    assert s_front.mean() > .9 and not s_back.any(), (s_front.mean(), s_back.mean())
    # a side face (normal +x): 90 deg off the ray: never faced from here
    side = np.c_[np.full(11, .5), np.linspace(0, .05, 11), np.zeros(11)]
    assert not seen(side, np.tile([1., 0, 0], (11, 1)), [(c2w, K, depth)], pts).any()
    # occluded: a box 1 m in front of the panel covers it in the depth map -> not clear
    near = np.full((140, 252), 2.)
    assert not seen(front, np.tile([0, -1., 0], (len(g), 1)), [(c2w, K, near)], pts).any()
    # no points there: faced and clear but nothing observed on it
    far_pts = pts + [0, 0, 5.]
    assert not seen(front, np.tile([0, -1., 0], (len(g), 1)), [(c2w, K, depth)], far_pts).any()
    # a mesh: a closed box (12 triangles) around the panel; only its front face's vertices are seen
    V = np.array([[x, y, z] for x in (-.5, .5) for y in (0., .05) for z in (-.5, .5)])
    F = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    m = mesh_seen(V, F, [(c2w, K, depth)], pts)
    assert not m[V[:, 1] > .01].any(), m
    assert spread(range(30), 12) == [0, 3, 5, 8, 11, 13, 16, 18, 21, 24, 26, 29] and spread([4, 5], 12) == [4, 5]
    print("observed self-check ok: faced, clear sight and points each required (front seen; back, side, occluded and empty not)")


if __name__ == "__main__":
    import sys
    assert sys.argv[1:] == ["--self-check"], __doc__
    self_check()
