"""r5 (models): what every object gets from its own observed points (the cards' points: the lift's stride-2 samples of DA3's
504x280 depth inside the object's SAM 3 masks, main cluster).

observed_mesh  the surface the video saw (a display model, never a measurement): a small TSDF of the object's own depth, its
               points put back on the keyframe grids they were lifted from, 1-4 cm voxels by the object's size and distance,
               marching cubes, components under 5 % of the triangles dropped, vertex colours from the keyframes. Nothing is added
               where no view looked: holes stay holes, the unseen bulk stays the primitive's (display_model, drawn translucent).
planar_parts   the observed points cut into planar parts (open3d RANSAC, normals agreeing, connected), each part's angle to the
               floor and the angle between touching parts, value +- u by the cards' angle rule (view-subset spread, the fit's own
               term, the shot's plumb term): measurements, separate from the display model.

    python -m fast_report.surface --self-check
"""
import time

import numpy as np

STRIDE, GRID_HW = 2, (280, 504)  # the lift's pixel stride on DA3's grid
VOXEL_M, MAX_TRIANGLES, MIN_COMPONENT = (.01, .04), 20000, .05
PART_SAMPLE, MAX_PARTS, NORMAL_AGREE_DEG, MIN_PART_SHARE, MIN_PART_POINTS, MIN_PART_SIDE_M = 5000, 6, 30., .05, 40, .05
TOUCH_PAIRS, MIN_FOLD_DEG, SUBSET_POINTS = 5, 5., 30
# u's k for a part's angle (r5 bench 001, ground truth: 265 parts of ARKitScenes 42445448 / 47333932 and TUM fr1 room matched to
# GT parts; k = 1 covered 78 %, 1.75 covers 90 %; left-one-sequence-out k 1.55-2.05); a fit term over FIT_MAX_DEG: a curved patch
SURFACE_K, FIT_MAX_DEG = 1.75, 15.


def _cam_points(world, c2w):
    return (np.asarray(world, float) - c2w[:3, 3]) @ c2w[:3, :3]


def observed_mesh(world, frame, c2w, K, rgb=None, views=None, hw=GRID_HW, stride=STRIDE):
    """world (N,3) shot-frame m, frame (N,) local keyframe of each point, c2w (n,4,4), K (n,3,3) at hw; rgb(local) -> that
    keyframe's BGR image or None; views: the local keyframes to integrate (default all). -> (vertices, faces, colours uint8, info)."""
    import open3d as o3d
    t0 = time.perf_counter()
    world, frame = np.asarray(world, float), np.asarray(frame)
    use = sorted(set(np.unique(frame).tolist()) & (set(views) if views is not None else set(np.unique(frame).tolist())))
    h, w = hw[0] // stride, hw[1] // stride
    empty = (np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32), np.zeros((0, 3), np.uint8))
    if not use or len(world) < 10:
        return (*empty, {"views": len(use), "reason": "too few points", "s": round(time.perf_counter() - t0, 4)})
    lo, hi = np.percentile(world, 2, 0), np.percentile(world, 98, 0)
    fx = float(np.median(K[use, 0, 0]))
    zmed = float(np.median([np.linalg.norm(np.median(world[frame == v], 0) - c2w[v][:3, 3]) for v in use]))
    vox = float(np.clip(max((hi - lo).max() / 64, zmed * stride / fx), *VOXEL_M))
    vol = o3d.pipelines.integration.ScalableTSDFVolume(voxel_length=vox, sdf_trunc=4 * vox, color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    strides = []
    for v in use:
        cam = _cam_points(world[frame == v], c2w[v])
        z = cam[:, 2]
        ok = z > .05
        cam, z = cam[ok], z[ok]
        u, vv = K[v][0, 0] * cam[:, 0] / z + K[v][0, 2], K[v][1, 1] * cam[:, 1] / z + K[v][1, 2]
        sv = view_stride(u, vv, stride)  # a view the point cap thinned out: a coarser grid, so its depth image stays whole
        strides.append(sv)
        h, w = hw[0] // sv, hw[1] // sv
        i, j = np.round(u / sv).astype(int), np.round(vv / sv).astype(int)
        inb = (i >= 0) & (j >= 0) & (i < w) & (j < h)
        i, j, z, u, vv = i[inb], j[inb], z[inb], u[inb], vv[inb]
        order = np.argsort(-z)  # the nearest point of a pixel is written last
        D = np.zeros((h, w), np.float32)
        D[j[order], i[order]] = z[order]
        C = np.full((h, w, 3), 128, np.uint8)
        img = rgb(v) if rgb is not None else None
        if img is not None:
            H, W = img.shape[:2]
            px = np.clip(np.round(u * W / hw[1]).astype(int), 0, W - 1)
            py = np.clip(np.round(vv * H / hw[0]).astype(int), 0, H - 1)
            C[j[order], i[order]] = img[py[order], px[order], ::-1]
        intr = o3d.camera.PinholeCameraIntrinsic(w, h, K[v][0, 0] / sv, K[v][1, 1] / sv, K[v][0, 2] / sv, K[v][1, 2] / sv)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(C), o3d.geometry.Image(D), depth_scale=1., depth_trunc=1e4,
                                                                  convert_rgb_to_intensity=False)
        vol.integrate(rgbd, intr, np.linalg.inv(c2w[v]))
    mesh = vol.extract_triangle_mesh()
    n0 = len(mesh.triangles)
    if n0:
        cl, counts, _ = mesh.cluster_connected_triangles()
        counts = np.asarray(counts)
        mesh.remove_triangles_by_mask(counts[np.asarray(cl)] < MIN_COMPONENT * n0)
        mesh.remove_unreferenced_vertices()
        if len(mesh.triangles) > MAX_TRIANGLES:
            mesh = mesh.simplify_quadric_decimation(MAX_TRIANGLES)
    V, F = np.asarray(mesh.vertices, np.float32), np.asarray(mesh.triangles, np.int32)
    Cv = (np.asarray(mesh.vertex_colors) * 255).round().astype(np.uint8) if len(V) else empty[2]
    return V, F, Cv, {"views": len(use), "voxel_m": round(vox, 4), "triangles_raw": n0, "triangles": int(len(F)), "stride_median": float(np.median(strides)),
                      "s": round(time.perf_counter() - t0, 4)}


def view_stride(u, v, base=STRIDE, strides=(2, 4, 8, 16)):
    """The finest grid stride (>= base) at which a view's points still fill their own footprint: the occupied cells at stride s over
    4 x those at 2 s (1 when the points are dense) >= 0.5."""
    occ = lambda s: len(np.unique(np.round(u / s).astype(np.int64) * 100003 + np.round(v / s).astype(np.int64)))  # noqa: E731
    for s in [x for x in strides if x >= base]:
        if occ(s) >= .5 * 4 * occ(2 * s):
            return s
    return strides[-1]


def _normals(Q, k=12):
    from scipy.spatial import cKDTree
    d, nn = cKDTree(Q).query(Q, k=min(k + 1, len(Q)))
    nb = Q[nn] - Q[nn].mean(1, keepdims=True)
    ev, vec = np.linalg.eigh(np.einsum("nki,nkj->nij", nb, nb) / nb.shape[1])
    return vec[:, :, 0], np.sqrt(np.maximum(ev[:, 0], 0)), float(np.median(d[:, 1]))


def _components(Q, radius):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    pairs = cKDTree(Q).query_pairs(radius, output_type="ndarray")
    g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(Q), len(Q))) if len(pairs) else coo_matrix((len(Q), len(Q)))
    return connected_components(g, directed=False)[1]


def _plane(X):
    """PCA plane of X -> (centre, unit normal, in-plane axes (2,3), p2-p98 extents (2), p95 |offset|)."""
    c = X.mean(0)
    _, _, vt = np.linalg.svd(X - c, full_matrices=False)
    loc = (X - c) @ vt[:2].T
    ext = np.percentile(loc, 98, 0) - np.percentile(loc, 2, 0)
    return c, vt[2], vt[:2], ext, float(np.percentile(np.abs((X - c) @ vt[2]), 95))


def tilt_deg(n):
    """The angle between a plane with normal n and the floor (z up): 0 horizontal, 90 vertical."""
    return float(np.degrees(np.arccos(np.clip(abs(n[2]) / max(np.linalg.norm(n), 1e-12), 0, 1))))


def planar_parts(P, frame, subsets, cams, plumb_u_deg=None, k=None, keep=False):
    """P (N,3) floor-frame points (z up, floor z = 0), frame (N,) their local keyframes, subsets: lists of local keyframes (the
    card's view subsets), cams (n,3) floor-frame camera centres by local keyframe. -> {"parts": [...], "bends": [...], "s"} or a
    status with its reason. Each part: tilt_deg (value +- u), area_m2, centre_m, normal (towards the cameras that saw it), share.
    keep: also the sampled points and each one's part (-1: none) under '_points' / '_labels' (for pictures, never serialised)."""
    import open3d as o3d
    from fast_report import cards
    t0 = time.perf_counter()
    k = k or {"angle": SURFACE_K}
    P, frame = np.asarray(P, float), np.asarray(frame)
    if len(P) > PART_SAMPLE:
        ix = np.sort(np.random.default_rng(0).choice(len(P), PART_SAMPLE, replace=False))
        P, frame = P[ix], frame[ix]
    n = len(P)
    if n < 2 * MIN_PART_POINTS:
        return {"status": "not measurable", "reason": f"{n} points: too few for planar parts", "s": round(time.perf_counter() - t0, 4)}
    normals, noise, spacing = _normals(P)
    tau = max(.01, 2.5 * float(np.median(noise)))
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    o3d.utility.random.seed(0)
    left, parts, cos_agree = np.arange(n), [], np.cos(np.radians(NORMAL_AGREE_DEG))
    for _ in range(3 * MAX_PARTS):
        if len(parts) == MAX_PARTS or len(left) < max(MIN_PART_POINTS, MIN_PART_SHARE * n):
            break
        plane, inl = pcd.select_by_index(left.tolist()).segment_plane(tau, 3, 200)
        nrm = np.asarray(plane[:3]) / np.linalg.norm(plane[:3])
        inl = left[np.asarray(inl, int)]
        inl = inl[np.abs(normals[inl] @ nrm) >= cos_agree]
        if len(inl) < MIN_PART_POINTS:
            break
        lab = _components(P[inl], 3 * spacing)
        comp = inl[lab == np.bincount(lab).argmax()]
        if len(comp) < max(MIN_PART_POINTS, MIN_PART_SHARE * n):
            left = np.setdiff1d(left, inl)
            continue
        c, nrm, axes, ext, res95 = _plane(P[comp])
        if ext.min() < MIN_PART_SIDE_M:
            left = np.setdiff1d(left, inl)  # no part here: these points leave the pool
            continue
        left = np.setdiff1d(left, comp)
        seen = cams[np.unique(frame[comp])].mean(0)
        if (seen - c) @ nrm < 0:
            nrm = -nrm
        parts.append({"ix": comp, "c": c, "n": nrm, "ext": ext, "res95": res95})
    if not parts:
        return {"status": "not measurable", "reason": "no planar part (>= 5 % of the points, >= 5 cm a side)", "s": round(time.perf_counter() - t0, 4)}
    fit = lambda p: float(np.degrees(np.arctan(2 * p["res95"] / max(p["ext"].min(), 1e-6))))  # noqa: E731  measure_observed_points' plane term

    def per_subset(p):
        out = []
        for sv in subsets or []:
            q = p["ix"][np.isin(frame[p["ix"]], sv)]
            if len(q) >= SUBSET_POINTS:
                _, m, _, e, _ = _plane(P[q])
                if e.min() >= .5 * p["ext"].min():
                    out.append(m if m @ p["n"] >= 0 else -m)
        return out
    rows = []
    for p in parts:
        p["subs"] = per_subset(p)
        vals = [tilt_deg(m) for m in p["subs"]]
        v = cards.value(tilt_deg(p["n"]), {"views": cards.spread(vals) if len(vals) >= 2 else None, "fit": fit(p), "plumb": plumb_u_deg},
                        "angle", k, vals if len(vals) >= 2 else None, unit="deg", scale=cards.SCALE_FREE)
        if fit(p) > FIT_MAX_DEG:  # the cards' rule for any angle: a patch this rough is curved, no angle
            v = {"status": "not measurable", "reason": f"a curved patch: fit term {fit(p):.0f} deg > {FIT_MAX_DEG:g}", "value_if_flat": v["value"]}
        rows.append({"tilt_deg": v, "area_m2": round(float(np.prod(p["ext"])), 3), "centre_m": np.round(p["c"], 3).tolist(),
                     "normal": np.round(p["n"], 4).tolist(), "sides_m": np.round(p["ext"], 3).tolist(), "share": round(len(p["ix"]) / n, 3)})
    bends = []
    from scipy.spatial import cKDTree
    for a in range(len(parts)):
        ta = cKDTree(P[parts[a]["ix"]])
        for b in range(a + 1, len(parts)):
            touch = sum(len(x) for x in ta.query_ball_point(P[parts[b]["ix"]], 2.5 * spacing))
            fold = float(np.degrees(np.arccos(np.clip(parts[a]["n"] @ parts[b]["n"], -1, 1))))
            if touch < TOUCH_PAIRS or fold < MIN_FOLD_DEG:
                continue
            subs = [float(np.degrees(np.arccos(np.clip(x @ y, -1, 1)))) for x, y in zip(parts[a]["subs"], parts[b]["subs"])]
            if max(fit(parts[a]), fit(parts[b])) > FIT_MAX_DEG:
                continue
            v = cards.value(180 - fold, {"views": cards.spread(subs) if len(subs) >= 2 else None, "fit": float(np.hypot(fit(parts[a]), fit(parts[b])))},
                            "angle", k, [180 - s for s in subs] if len(subs) >= 2 else None, unit="deg", scale=cards.SCALE_FREE,
                            note="the angle between the two parts (180 = flat)")
            bends.append({"parts": [a, b], "angle_deg": v})
    out = {"parts": rows, "bends": bends, "points": n, "tolerance_m": round(tau, 4), "s": round(time.perf_counter() - t0, 4),
           "note": "planar parts of the observed points (the video's own surface), each to the floor: 0 = horizontal, 90 = vertical"}
    if keep:
        lab = np.full(n, -1)
        for i, p in enumerate(parts):
            lab[p["ix"]] = i
        out.update(_points=P, _labels=lab)
    return out


def decided(rec, threshold):
    """A value +- u clear of a threshold (e.g. a 30 deg rule): True / False, None without a value (or not measurable)."""
    if not rec or "value" not in rec:
        return None
    return abs(rec["value"] - threshold) > rec["u"]


# ---------------------------------------------------------------- the report's display model (fast_report.core.surfaces_job)
DISPLAY_TRIANGLES, OBSERVED_ALPHA, PRIMITIVE_ALPHA, PRIMITIVE_RGB = 8000, 255, 60, (255, 184, 77)


def mesh_glb(V, F, rgba):
    """One-primitive GLB: float32 POSITION, normalised uint8 RGBA COLOR_0, uint32 indices, one double-sided BLEND material."""
    import json
    import struct
    V, F, rgba = np.ascontiguousarray(V, "<f4"), np.ascontiguousarray(F, "<u4"), np.ascontiguousarray(rgba, np.uint8)
    n = len(V)
    blob = V.tobytes() + rgba.tobytes() + F.tobytes()
    gltf = {"asset": {"version": "2.0", "generator": "panoptes fast_report.surface"}, "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}],
            "materials": [{"pbrMetallicRoughness": {"baseColorFactor": [1, 1, 1, 1], "metallicFactor": 0, "roughnessFactor": 1}, "alphaMode": "BLEND",
                           "doubleSided": True}],
            "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "COLOR_0": 1}, "indices": 2, "material": 0, "mode": 4}]}],
            "buffers": [{"byteLength": len(blob)}],
            "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": 12 * n, "target": 34962},
                            {"buffer": 0, "byteOffset": 12 * n, "byteLength": 4 * n, "target": 34962},
                            {"buffer": 0, "byteOffset": 16 * n, "byteLength": F.nbytes, "target": 34963}],
            "accessors": [{"bufferView": 0, "componentType": 5126, "count": n, "type": "VEC3", "min": V.min(0).astype(float).tolist(),
                           "max": V.max(0).astype(float).tolist()},
                          {"bufferView": 1, "componentType": 5121, "normalized": True, "count": n, "type": "VEC4"},
                          {"bufferView": 2, "componentType": 5125, "count": int(F.size), "type": "SCALAR"}]}
    head = json.dumps(gltf, separators=(",", ":")).encode()
    head += b" " * (-len(head) % 4)
    blob += b"\0" * (-len(blob) % 4)
    return (struct.pack("<III", 0x46546C67, 2, 28 + len(head) + len(blob)) + struct.pack("<II", len(head), 0x4E4F534A) + head
            + struct.pack("<II", len(blob), 0x004E4942) + blob)


def display_glb(V, F, C, model=None):
    """The card's display model as one GLB about its own centre: the observed mesh opaque (decimated to DISPLAY_TRIANGLES), the
    card's primitive (display_model's record: its shape, seen faces too) translucent for the unseen bulk. -> (bytes, centre, info)."""
    import open3d as o3d
    parts_v, parts_f, parts_c, n_obs = [], [], [], 0
    if len(F):
        if len(F) > DISPLAY_TRIANGLES:
            m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(np.asarray(V, np.float64)), o3d.utility.Vector3iVector(np.asarray(F, np.int32)))
            m.vertex_colors = o3d.utility.Vector3dVector(np.asarray(C, np.float64) / 255)
            m = m.simplify_quadric_decimation(DISPLAY_TRIANGLES)
            V, F, C = np.asarray(m.vertices), np.asarray(m.triangles), (np.asarray(m.vertex_colors) * 255).round()
        parts_v.append(np.asarray(V, float))
        parts_f.append(np.asarray(F, np.int64))
        parts_c.append(np.c_[np.asarray(C, float)[:, :3], np.full(len(V), OBSERVED_ALPHA)])
        n_obs = int(len(F))
    if model and model.get("kind"):
        import r4_models
        T, _ = r4_models.model_tris(model)
        PV = T.reshape(-1, 3)
        parts_f.append(np.arange(len(PV)).reshape(-1, 3) + sum(len(v) for v in parts_v))
        parts_v.append(PV)
        parts_c.append(np.tile([*PRIMITIVE_RGB, PRIMITIVE_ALPHA], (len(PV), 1)))
    if not parts_v:
        return None, None, {"reason": "no observed surface and no primitive"}
    V, F, C = np.concatenate(parts_v), np.concatenate(parts_f), np.concatenate(parts_c)
    centre = (V.min(0) + V.max(0)) / 2
    half = (V.max(0) - V.min(0)) / 2
    return mesh_glb(V - centre, F, C.clip(0, 255)), centre, {"triangles": int(len(F)), "observed_triangles": n_obs,
                                                           "bounds": {"min": np.round(-half, 4).tolist(), "max": np.round(half, 4).tolist()}}


def card_display(job):
    """A worker's card: (id, world points, their local keyframes, c2w, K, frames .npy path, the shot's source keys, the card's model
    record) -> (id, GLB bytes or None, centre, info). The card's main cluster is the cards' own (cards.prepare)."""
    cid, world, frame, c2w, K, frames_path, keys, fr, fx, model = job
    from fast_report import cards
    t0 = time.perf_counter()
    world = np.asarray(world, float)
    frame = np.asarray(frame)
    if len(world) < 10:
        return cid, None, None, {"reason": "too few points", "s": 0.}
    z = np.einsum("ni,ni->n", world - c2w[frame, :3, 3], c2w[frame, :3, 2])
    x = cards.prepare({"world": world, "frame": frame, "z": z}, fr, fx)
    Pw = x["P"] @ fr["R"] + fr["origin"]
    frames = np.load(frames_path, mmap_mode="r") if frames_path else None
    V, F, C, info = observed_mesh(Pw, x["frame"], c2w, K, rgb=(lambda v: frames[keys[v]]) if frames is not None else None)
    glb, centre, dinfo = display_glb(V, F, C, model)
    return cid, glb, None if centre is None else np.round(centre, 4).tolist(), {**info, **dinfo, "s": round(time.perf_counter() - t0, 4)}


# ---------------------------------------------------------------- self-check (synthetic)
def _sheet(o, a, b, n, rng, noise=.003):
    s, t = rng.random(n), rng.random(n)
    return o + s[:, None] * a + t[:, None] * b + rng.normal(0, noise, (n, 3))


def _look_at(eye, target):
    z = np.asarray(target, float) - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1.])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    c2w = np.eye(4)
    c2w[:3, :3], c2w[:3, 3] = np.stack([x, y, z], 1), eye
    return c2w


def self_check():
    rng = np.random.default_rng(1)
    # a bent guard: a sheet 30 deg to the floor meeting a vertical sheet (the angle between them 120 deg), seen from -y
    a = _sheet(np.zeros(3), np.array([.8, 0, 0]), np.array([0, .5 * np.cos(np.radians(30)), .5 * np.sin(np.radians(30))]), 1500, rng)
    top = np.array([0, .5 * np.cos(np.radians(30)), .5 * np.sin(np.radians(30))])
    b = _sheet(top, np.array([.8, 0, 0]), np.array([0, 0, .6]), 1500, rng)
    P = np.concatenate([a, b]) + [0, 0, .5]
    frame = np.r_[np.tile([0, 1, 2], 1000)]
    cams = np.array([[-.5, -2, 1.6], [.4, -2.2, 1.5], [1.3, -2, 1.7]])
    out = planar_parts(P, frame, [[0], [1], [2]], cams, plumb_u_deg=.5)
    tilts = sorted(p["tilt_deg"]["value"] for p in out["parts"])
    assert len(tilts) == 2 and abs(tilts[0] - 30) < 2 and abs(tilts[1] - 90) < 2, tilts
    assert len(out["bends"]) == 1 and abs(out["bends"][0]["angle_deg"]["value"] - 120) < 3, out["bends"]
    p30 = min(out["parts"], key=lambda p: p["tilt_deg"]["value"])["tilt_deg"]
    assert 0 < p30["u"] < 3 and p30["n_subsets"] == 3 and decided(p30, 45) and not decided({"value": 30.5, "u": 1.}, 30)
    # a box seen on two sides and the top: 0 and 90 deg parts, 90 deg between touching faces
    box = np.concatenate([_sheet(np.zeros(3), np.array([.6, 0, 0]), np.array([0, 0, .5]), 1200, rng),
                          _sheet(np.array([.6, 0, 0]), np.array([0, .4, 0]), np.array([0, 0, .5]), 1200, rng),
                          _sheet(np.array([0, 0, .5]), np.array([.6, 0, 0]), np.array([0, .4, 0]), 1200, rng)])
    o2 = planar_parts(box, np.tile([0, 1], 1800), [[0], [1]], np.array([[2, -2, 1.6], [2.5, -1, 1.2]]))
    assert sorted(round(p["tilt_deg"]["value"] / 45) * 45 for p in o2["parts"]) == [0, 90, 90], [p["tilt_deg"]["value"] for p in o2["parts"]]
    assert all(abs(b_["angle_deg"]["value"] - 90) < 3 for b_ in o2["bends"]) and len(o2["bends"]) == 3
    assert planar_parts(box[:30], np.zeros(30, int), [], np.zeros((1, 3)))["status"] == "not measurable"
    # observed mesh: the box's front face seen by 3 cameras; the mesh stays on the seen face, the back is not invented
    K = np.array([[250., 0, 252], [0, 250., 140], [0, 0, 1]])
    c2ws = np.stack([_look_at(np.array([x, -2., .8]), [.3, 0, .25]) for x in (-.4, .3, 1.)])
    face = _sheet(np.zeros(3), np.array([.6, 0, 0]), np.array([0, 0, .5]), 20000, rng, noise=.002)
    fr = rng.integers(0, 3, len(face))
    V, F, C, info = observed_mesh(face, fr, c2ws, np.repeat(K[None], 3, 0), rgb=lambda v: np.full((720, 1280, 3), (40, 90, 200), np.uint8))
    assert len(F) > 100 and np.abs(V[:, 1]).max() < 4 * info["voxel_m"], (len(F), np.abs(V[:, 1]).max())
    assert V[:, 0].min() > -.05 and V[:, 0].max() < .65 and V[:, 2].max() < .55 and np.allclose(C.mean(0), (200, 90, 40), atol=8)
    thin = rng.random(len(face)) < .08  # the point cap's random sample: a sparse speckle on each view's grid
    _, Ft, _, it = observed_mesh(face[thin], fr[thin], c2ws, np.repeat(K[None], 3, 0))
    assert it["stride_median"] > 2 and len(Ft) > 50, it
    _, F1, _, i1 = observed_mesh(face, fr, c2ws, np.repeat(K[None], 3, 0), views=[1])
    assert i1["views"] == 1 and 0 < len(F1) and observed_mesh(face[:5], fr[:5], c2ws, np.repeat(K[None], 3, 0))[3]["reason"] == "too few points"
    import struct
    rec = {"kind": "box", "size_m": [.6, .1, .5], "faces": {f: "seen" for f in ("-x", "+x", "-y", "+y", "-z", "+z")}, "position": [.3, 0, .25],
           "quaternion": [0, 0, 0, 1]}
    import sys
    sys.path.append(str(__import__("pathlib").Path(__file__).resolve().parents[1] / "scripts"))
    glb, centre, dinfo = display_glb(V, F, C, rec)
    assert glb[:4] == b"glTF" and struct.unpack("<I", glb[8:12])[0] == len(glb) and dinfo["triangles"] == len(F) + 12 and dinfo["observed_triangles"] == len(F)
    assert np.allclose(centre, [.3, 0, .25], atol=.06), centre
    print(f"surface self-check ok: a bent guard (30 / 90 deg parts, 120 deg between them, u {p30['u']}), a box (0 / 90 / 90, 90 deg edges), "
          f"the observed mesh on the seen face only ({len(F)} triangles, {info['voxel_m'] * 100:.1f} cm voxels, {info['s'] * 1000:.0f} ms)")


if __name__ == "__main__":
    import sys
    assert sys.argv[1:] == ["--self-check"], __doc__
    self_check()
