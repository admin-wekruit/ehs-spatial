"""X7: one model per unique object of a video from its best views (the photo-workcell recipe on video), on the core's
merged objects (fb/a-core's lift: one object per shot-level 3D component, never per window).

Per object (op 'select', CPU pool, fb/b's FastSource over the core's hand-off in /dev/shm):
  good views  in view and clear of the caption band and the person, mostly reliable depth (complete_video_objects.failing
              without 'small'), and unoccluded by coarse depth (OCCLUDED: share of the mask's outer ring that is nearer
              than the object's own edge); scored as complete_video_objects.pick_views (area x relative sharpness x
              frontality x solidity^2) x (1 - occluded share)
  directions  greedy by score, each view's camera direction from the object's centroid >= MIN_SEP_DEG from every view
              taken: up to MAX_GEN_VIEWS + 1; the last one taken is HELD OUT (never generates, never places, only gates)
  jobs        (a) RecGen generate_multiview on the generation views (source-frame crops, depth on the crop grid, crop K,
              complete_video_objects.build_input's recipe); one view: RecGen generate
              (b) SAM 3D s1cfg12 on the single best view, if that view passes today's prepare rule (failing(): incl. 'small')
              (c) only on a (b) prepare rejection: SAM 3D on the next-best view that passes it
              (p) a parametric shape for boxes, posts, shelves, partitions (box / upright cylinder / plane slab) from the
              generation views' points, with its measurable facts (tilt, height above the floor, clearance), 'estimated'
Per model (op 'gate'): the source camera places it (RecGen: its anchor camera; SAM 3D: its view's camera), then the
workcell's bounded render-and-compare (panoptes-serving scripts/research/assemble_lucida_scene.refine: Nelder-Mead,
translation <= 0.3 x size, rotation <= 0.65 rad, per-axis scale x e^+-0.4) on the GENERATION views only, then the
source-consistency gate (build_lingbot_object_model.evaluate: silhouette IoU >= 0.65, relative depth median <= 0.04,
p95 <= 0.10, >= 300 supported px) on the HELD-OUT view. Generated models are display layers, never measurements; every
metre is estimated (floor plane + assumed camera height). RecGen: non-commercial research licence (demo only).

  python -m fast_report.x7 --self-check      # CPU: selection, placement, gate, parametric fits on a synthetic shot
"""
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "scripts", ROOT / "modal_apps"):
    if str(_p) not in sys.path:
        sys.path.append(str(_p))

MIN_SEP_DEG, MAX_GEN_VIEWS, OCCLUDED, NEARER = 15., 4, .2, .9
GOOD_FAILS = {"side", "top/bottom", "raster edge", "depth", "person", "caption"}  # failing() minus 'small'
SEED, TILE, RECGEN_MAX_DEPTH = 42, 128, 30.  # RecGen reads depth > 30 as millimetres: beyond it, no depth
PARAM = {"post": ("post", "pole", "column", "pillar", "bollard", "stanchion"),
         "shelf": ("shelf", "shelves", "shelving", "rack", "gondola"),
         "partition": ("partition", "divider", "barrier", "fence", "wall panel", "door", "board", "screen", "panel"),
         "box": ("box", "boxes", "carton", "crate", "pallet", "bin", "container", "case", "cabinet", "package", "tote", "stack")}
RECGEN_PY, RECGEN_CODE, RECGEN_DIR = "/opt/recgen-venv/bin/python", "fe3c9315b439c50ada8b60c12b469d739fd722db", "/opt/recgen"
RECGEN_MODEL, RECGEN_REV, RECGEN_NAME = "TRI-ML/RecGen", "bc0df7de2e43314830039a35a720731d4c4fac65", "recgen_base.multiview_stereo"
DINO_REV, RECGEN_CACHE = "7764ea0f912e53c92e82eb78a2a1631e92725fc8", "/cache"


# ---------------------------------------------------------------- small geometry
def unit(v):
    v = np.asarray(v, float)
    return v / max(np.linalg.norm(v), 1e-12)


def angle_deg(a, b):
    return float(np.degrees(np.arccos(np.clip(unit(a) @ unit(b), -1, 1))))


def spread_views(ranked, centre, cam_of, n_max=MAX_GEN_VIEWS + 1, min_sep=MIN_SEP_DEG):
    """Greedy by rank: a view is taken if its camera direction from the object's centre is >= min_sep from every view
    taken. -> (generation views, held-out view or None). The held-out view is the last one taken (>= min_sep from all)."""
    taken = []
    for m in ranked:
        d = cam_of(m) - centre
        if len(taken) < n_max and all(angle_deg(d, cam_of(t) - centre) >= min_sep for t in taken):
            taken.append(m)
    if len(taken) < 2:
        return taken, None
    return taken[:-1], taken[-1]


def occlusion_share(mask, depth, nearer=NEARER):
    """Share of the mask's outer ring (3 px) with depth nearer than `nearer` x the depth of the closest mask-edge pixel:
    something in front of the object's outline. NaN when the ring has no depth."""
    import cv2
    from scipy.ndimage import distance_transform_edt
    k = np.ones((7, 7), np.uint8)
    m = mask.astype(np.uint8)
    ring = (cv2.dilate(m, k) > 0) & ~mask & (depth > 0)
    edge = mask & ~(cv2.erode(m, np.ones((5, 5), np.uint8)) > 0) & (depth > 0)
    if not ring.any() or not edge.any():
        return float("nan")
    _, (iy, ix) = distance_transform_edt(~edge, return_indices=True)
    near = depth[ring] < nearer * depth[iy[ring], ix[ring]]
    return float(near.mean())


def transformed(vertices, t):
    return np.asarray(vertices) @ t[:3, :3].T + t[:3, 3]


def lift_points(depth, k, c2w, mask):
    v, u = np.nonzero(mask & (depth > 0))
    z = depth[v, u].astype(np.float64)
    local = np.stack([(u - k[0, 2]) / k[0, 0] * z, (v - k[1, 2]) / k[1, 1] * z, z], 1)
    return local @ np.asarray(c2w)[:3, :3].T + np.asarray(c2w)[:3, 3]


# ---------------------------------------------------------------- render-and-compare (workcell, ported)
def rays(k, c2w, w, h):
    """World rays of a pinhole view, directions with unit camera z (t_hit = z depth)."""
    import open3d as o3d
    return o3d.t.geometry.RaycastingScene.create_rays_pinhole(o3d.core.Tensor(np.asarray(k, np.float64)),
                                                             o3d.core.Tensor(np.linalg.inv(np.asarray(c2w, np.float64))), w, h).numpy()


class Caster:
    """One BVH per mesh; a pose moves the rays into the mesh's frame instead (assemble_lucida_scene.cast_depth)."""

    def __init__(self, vertices, faces):
        import open3d as o3d
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(o3d.core.Tensor(np.asarray(vertices, np.float32)), o3d.core.Tensor(np.asarray(faces, np.uint32)))

    def depth(self, transform, r):
        import open3d as o3d
        inv = np.linalg.inv(transform)
        local = np.concatenate([transformed(r[..., :3], inv), r[..., 3:] @ inv[:3, :3].T], -1).astype(np.float32)
        return self.scene.cast_rays(o3d.core.Tensor(local))["t_hit"].numpy()


def score_view(caster, transform, view):
    """assemble_lucida_scene.score_view: IoU of the visible model with the mask (the model hidden only where the scene
    is in front of it outside the mask), boundary error / image height, relative depth p50; loss = 1 - IoU + 2 boundary + p50."""
    import cv2
    pred = caster.depth(transform, view["rays"])
    target, depth = view["target"], view["depth"]
    finite = np.isfinite(pred) & (pred > 0)
    visible = finite & ~((~target) & (depth > 0) & (pred > depth * 1.04))
    iou = float((visible & target).sum() / max(1, (visible | target).sum()))
    sup = visible & target & (depth > 0)
    rel = np.abs(pred[sup] - depth[sup]) / depth[sup]
    p50 = float(np.median(rel)) if len(rel) else None

    def edge(mask):
        return mask & ~cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    a, b = edge(visible), edge(target)
    if a.any() and b.any():
        da = cv2.distanceTransform((~a).astype(np.uint8), cv2.DIST_L2, 3)
        db = cv2.distanceTransform((~b).astype(np.uint8), cv2.DIST_L2, 3)
        boundary = float((np.mean(da[b]) + np.mean(db[a])) / (2 * len(target)))
    else:
        boundary = 1.
    return {"iou": iou, "boundary": boundary, "p50": p50, "loss": float(1 - iou + 2 * boundary + min(p50 if p50 is not None else 1, 1))}


def refine(vertices, faces, views, max_iterations=100, *, uniform_scale=False, axis=None):
    """assemble_lucida_scene.refine: bounded Nelder-Mead over translation (x 0.3 of the model's size), rotation vector
    (<= 0.65 rad) and per-axis log scale (<= 0.4) about the model's centre, from the source-camera placement. Returns
    (4x4 applied to world vertices, record); identity when nothing beats the start.
    uniform_scale uses a Sim(3) transform, preserving the mesh's intrinsic angles; axis limits the rotation to turns
    about that one world direction (an upright model stays upright)."""
    from scipy.optimize import minimize
    from scipy.spatial.transform import Rotation
    if axis is not None and not uniform_scale:
        raise ValueError('axis needs uniform_scale: per-axis world scaling would tilt the model off that axis')
    caster = Caster(vertices, faces)
    centre = np.asarray(vertices).mean(0)
    radius = max(float(np.linalg.norm(np.ptp(vertices, 0))), 1e-4)
    turns = 3 if axis is None else 1

    def candidate(x):
        t = np.eye(4)
        scales = np.repeat(np.exp(x[3 + turns]), 3) if uniform_scale else np.exp(x[3 + turns:])
        rotation = x[3:6] if axis is None else x[3] * unit(axis)
        t[:3, :3] = Rotation.from_rotvec(rotation).as_matrix() @ np.diag(scales)
        t[:3, 3] = centre - t[:3, :3] @ centre + x[:3] * radius
        return t

    def score(t):
        per = [score_view(caster, t, v) for v in views]
        return float(np.mean([p["loss"] for p in per])), per
    start = time.perf_counter()
    initial_loss, initial = score(np.eye(4))
    count = [0]

    def objective(x):
        if np.max(np.abs(x[:3])) > .3 or np.linalg.norm(x[3:3 + turns]) > .65 or np.max(np.abs(x[3 + turns:])) > .4:
            return 100. + float(x @ x)
        count[0] += 1
        return score(candidate(x))[0]
    n = 3 + turns + (1 if uniform_scale else 3)
    simplex = np.zeros((n + 1, n))
    simplex[1:] = np.diag([.015] * 3 + [.04] * (n - 3))
    fit = minimize(objective, np.zeros(n), method="Nelder-Mead", options={"initial_simplex": simplex, "maxiter": max_iterations,
                                                                            "xatol": .002, "fatol": .001})
    final = candidate(fit.x) if fit.fun < initial_loss else np.eye(4)
    final_loss, per = score(final)
    return final, {"seconds": round(time.perf_counter() - start, 3), "evaluations": count[0], "loss_before": round(initial_loss, 4),
                   "loss_after": round(final_loss, 4), "iou_before": [round(p["iou"], 3) for p in initial],
                   "iou_after": [round(p["iou"], 3) for p in per], "moved": bool(fit.fun < initial_loss)}


# ---------------------------------------------------------------- parametric shapes
def floor_frame(up):
    up = unit(up)
    a = unit(np.cross(up, [1., 0, 0] if abs(up[0]) < .9 else [0, 1., 0]))
    return up, a, np.cross(up, a)


def fit_plane(points):
    """Trimmed SVD plane (3 rounds, 2.5 x median residual): (centre, normal)."""
    keep = points
    for _ in range(3):
        c = keep.mean(0)
        n = np.linalg.svd(keep - c, full_matrices=False)[2][2]
        r = np.abs((keep - c) @ n)
        keep = keep[r <= 2.5 * max(np.median(r), 1e-9)]
    return keep.mean(0), unit(n)


def fit_axis(points):
    """Principal axis of the points (a post's length)."""
    c = points.mean(0)
    return c, unit(np.linalg.svd(points - c, full_matrices=False)[2][0])


def fit_circle(xy):
    """Kasa least-squares circle -> (centre 2, radius)."""
    a = np.c_[2 * xy, np.ones(len(xy))]
    b = (xy ** 2).sum(1)
    sol = np.linalg.lstsq(a, b, rcond=None)[0]
    return sol[:2], float(np.sqrt(max(sol[2] + sol[:2] @ sol[:2], 1e-12)))


def cylinder_mesh(bottom, axis, radius, height, sides=32):
    _, a, b = floor_frame(axis)
    ang = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    ring = np.cos(ang)[:, None] * a + np.sin(ang)[:, None] * b
    v = np.concatenate([bottom + radius * ring, bottom + height * axis + radius * ring, [bottom, bottom + height * axis]])
    i = np.arange(sides)
    j = (i + 1) % sides
    faces = np.concatenate([np.stack([i, j, sides + j], 1), np.stack([i, sides + j, sides + i], 1),
                            np.stack([np.full(sides, 2 * sides), j, i], 1), np.stack([np.full(sides, 2 * sides + 1), sides + i, sides + j], 1)])
    return v, faces


def box_from_axes(lo, hi, axes):
    """Box of local extents [lo, hi] on row axes (3x3) -> (world vertices, faces)."""
    import trimesh
    unitbox = trimesh.creation.box(bounds=[lo, hi])
    return np.asarray(unitbox.vertices) @ axes, np.asarray(unitbox.faces)


def box_distance(points, lo, hi, axes):
    """|distance| of points to the surface of the box (local extents on row axes)."""
    q = points @ axes.T
    c, h = (lo + hi) / 2, (hi - lo) / 2
    d = np.abs(q - c) - h
    outside = np.linalg.norm(np.maximum(d, 0), axis=1)
    inside = np.minimum(d.max(1), 0)
    return np.abs(outside + inside)


def parametric(kind, points, up, floor_point, rng=None, boots=30):
    """Fit one shape to an object's observed points (world, estimated metres). -> (vertices, faces, record). Facts carry a
    bootstrap spread (points resampled `boots` times): the ± of a fact is that spread's standard deviation."""
    rng = np.random.default_rng(0) if rng is None else rng
    up, a, b = floor_frame(up)
    height_of = lambda p: (p - floor_point) @ up  # noqa: E731
    lo_pct, hi_pct = 1, 99

    def spread(fn):
        vals = [fn(points[rng.integers(0, len(points), len(points))]) for _ in range(boots)]
        return float(np.std(vals))
    t = time.perf_counter()
    if kind == "post":
        centre, axis = fit_axis(points)
        axis = axis if axis @ up >= 0 else -axis
        planar = points - np.outer((points - centre) @ axis, axis)
        _, pa, pb = floor_frame(axis)
        xy = np.c_[(planar - centre) @ pa, (planar - centre) @ pb]
        c2, radius = fit_circle(xy)
        base = centre + c2[0] * pa + c2[1] * pb
        s = (points - base) @ axis
        s0, s1 = np.percentile(s, lo_pct), np.percentile(s, hi_pct)
        vertices, faces = cylinder_mesh(base + s0 * axis, axis, radius, s1 - s0)
        radial = np.linalg.norm((points - base) - np.outer(s, axis), axis=1)
        residual = np.abs(radial - radius)
        tilt = lambda p: angle_deg(fit_axis(p)[1] * np.sign(fit_axis(p)[1] @ up), up)  # noqa: E731
        facts = {"tilt_from_vertical_deg": (angle_deg(axis, up), spread(tilt)), "radius_m": (radius, None), "height_m": (s1 - s0, None),
                 "top_above_floor_m": (float(np.percentile(height_of(points), hi_pct)), spread(lambda p: np.percentile(height_of(p), hi_pct)))}
    elif kind == "partition":
        centre, n = fit_plane(points)
        u = unit(np.cross(n, up)) if abs(n @ up) < .95 else a
        w = np.cross(n, u)
        loc = np.c_[(points - centre) @ u, (points - centre) @ w, (points - centre) @ n]
        lo, hi = np.percentile(loc, lo_pct, 0), np.percentile(loc, hi_pct, 0)
        lo[2], hi[2] = -.01, .01  # a 2 cm slab
        axes = np.stack([u, w, n])
        vertices, faces = box_from_axes(lo, hi, axes)
        vertices = vertices + centre
        residual = np.abs((points - centre) @ n)
        lean = lambda p: abs(90. - angle_deg(fit_plane(p)[1], up))  # noqa: E731
        facts = {"tilt_from_vertical_deg": (lean(points), spread(lean)),
                 "top_above_floor_m": (float(np.percentile(height_of(points), hi_pct)), spread(lambda p: np.percentile(height_of(p), hi_pct))),
                 "bottom_above_floor_m": (float(np.percentile(height_of(points), lo_pct)), spread(lambda p: np.percentile(height_of(p), lo_pct)))}
    else:  # box, shelf: gravity-aligned box, yaw by the least median surface distance (1 deg steps; box_mesh's minAreaRect
        # yaw tilts the box when only two faces are seen: their hull is a triangle); a shelf's lean from its front plane
        def box_at(deg):
            x = np.cos(np.radians(deg)) * a + np.sin(np.radians(deg)) * b
            axes = np.stack([x, np.cross(up, x), up])
            loc = points @ axes.T
            lo, hi = np.percentile(loc, lo_pct, 0), np.percentile(loc, hi_pct, 0)
            hi = np.maximum(hi, lo + .02)
            return float(np.median(box_distance(points, lo, hi, axes))), lo, hi, axes
        _, lo, hi, axes = min((box_at(d) for d in range(90)), key=lambda r: r[0])
        vertices, faces = box_from_axes(lo, hi, axes)
        residual = box_distance(points, lo, hi, axes)
        facts = {"top_above_floor_m": (float(np.percentile(height_of(points), hi_pct)), spread(lambda p: np.percentile(height_of(p), hi_pct))),
                 "bottom_above_floor_m": (float(np.percentile(height_of(points), lo_pct)), spread(lambda p: np.percentile(height_of(p), lo_pct))),
                 "footprint_m": ([round(float(v), 3) for v in (hi - lo)[:2]], None)}
        if kind == "shelf":
            lean = lambda p: abs(90. - angle_deg(fit_plane(p)[1], up))  # noqa: E731
            facts["tilt_from_vertical_deg"] = (lean(points), spread(lean))
    record = {"kind": kind, "fit_s": round(time.perf_counter() - t, 3), "points": int(len(points)),
              "residual_median_m": round(float(np.median(residual)), 4), "residual_p90_m": round(float(np.percentile(residual, 90)), 4),
              "facts": {k: {"value": v if isinstance(v, list) else round(float(v), 3), "pm": None if s is None else round(s, 3)}
                        for k, (v, s) in facts.items()},
              "status": "estimated: scale from the floor plane and an assumed 1.6 m camera height; no measurement"}
    return vertices, faces, record


def param_kind(labels):
    import complete_video_objects as cvo
    for kind in ("post", "shelf", "partition", "box"):
        if any(lab and cvo.matches(lab, PARAM[kind]) for lab in labels):
            return kind
    return None


# ---------------------------------------------------------------- views of one object (CPU pool, fb/b's FastSource)
def view_data(src, key, frame, scale=1.):
    """(depth, mask, K, c2w) of one object view on the DA3 grid (x scale): the reliable depth (edges and the moving person
    out, complete_video_objects.reliable), the object's mask without specks and without that person."""
    import cv2
    import complete_video_objects as cvo
    row = src.rows[frame]
    depth, moving = cvo.reliable(row, src.clip, None)
    mask = cvo.main_parts(src.clip.raster_mask(cvo.mask_path(src.args.masks, f"{key}:{frame}:0"))) & ~moving
    k = src.clip.k_raster.copy()
    if scale != 1:
        h, w = depth.shape
        size = (int(round(w * scale)), int(round(h * scale)))
        depth = cv2.resize(depth, size, interpolation=cv2.INTER_NEAREST)
        mask = cv2.resize(mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
        k[:2] *= [[size[0] / w], [size[1] / h]]
    return depth, mask, k, np.asarray(row["c2w"], float)


def lowres_masks(src, key, frames):
    """The object's masks on the 4:3 centre crop at 160x120 (the delivered masks' raster / 4), packed: for overlap with the
    delivered report's objects."""
    import cv2
    from fast_report.sam3d import mask_from_logits
    logits = np.load(src.root / "logits" / f"{key}.npy", mmap_mode="r")
    W, H = src.clip.full_size
    x0 = (W - H * 4 // 3) // 2
    out = []
    for i, f in enumerate(frames):
        m = mask_from_logits(logits[i], (W, H))[:, x0:W - x0]
        out.append(np.packbits(cv2.resize(m.astype(np.uint8), (160, 120), interpolation=cv2.INTER_AREA) > 0))
    return np.stack(out) if out else np.zeros((0, 2400), np.uint8)


def crop_square(mask_full, pad=1.4):
    ys, xs = np.nonzero(mask_full)
    lo, hi = np.array([xs.min(), ys.min()], float), np.array([xs.max(), ys.max()], float) + 1
    side = int(np.ceil((hi - lo).max() * pad))
    left, top = np.floor((lo + hi - side) / 2).astype(int)
    return int(left), int(top), side


def crop_rgb(rgb, left, top, side):
    out = np.zeros((side, side, 3), np.uint8)
    y0, x0, y1, x1 = max(top, 0), max(left, 0), min(top + side, rgb.shape[0]), min(left + side, rgb.shape[1])
    out[y0 - top:y1 - top, x0 - left:x1 - left] = rgb[y0:y1, x0:x1]
    return out


def jpeg(rgb, size=TILE):
    import cv2
    img = cv2.resize(np.ascontiguousarray(rgb), (size, size), interpolation=cv2.INTER_AREA)
    return cv2.imencode(".jpg", img[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()


def recgen_view(src, view, frame_rgb):
    """complete_video_objects.build_input's RecGen crop: source-frame square crop, the mask carried onto it, the view's
    reliable depth on the crop grid (0 beyond RECGEN_MAX_DEPTH), crop K. None when the crop keeps no mask with depth."""
    from scipy.ndimage import binary_erosion
    import complete_video_objects as cvo
    from ehs_spatial.platform.recgen import source_grid_crop
    depth, _ = cvo.reliable(src.rows[view["frame"]], src.clip, None)
    depth = np.where(depth > RECGEN_MAX_DEPTH, 0, depth).astype(np.float32)
    full = src.full_mask(view)
    crop = source_grid_crop(frame_rgb, full, depth, src.clip.k_raster, src.clip.full_to_raster)
    left, top, right, bottom = crop["pixelMapping"]["sourceCropXYXY"]
    yy, xx = np.mgrid[top:bottom, left:right]
    inside = (xx >= 0) & (yy >= 0) & (xx < full.shape[1]) & (yy < full.shape[0])
    mask = np.zeros(inside.shape, bool)
    mask[inside] = full[yy[inside], xx[inside]]
    if not (binary_erosion(mask, structure=np.ones((5, 5), bool)) & (crop["depth"] > 0)).any():
        return None
    return {"rgb": np.ascontiguousarray(crop["rgb"]), "depth": crop["depth"], "mask": mask.astype(np.uint8) * 255,
            "camera_intrinsics": np.asarray(crop["K"], np.float64)}


def select(src, key, obj, min_sep=MIN_SEP_DEG):
    """One object's views and jobs (module docstring). obj: the fixture's object record (centroid_m, labels)."""
    import cv2
    import complete_video_objects as cvo
    from fast_report import sam3d
    t0 = time.time()
    entity = src.entities[key]
    src.write_masks(key)
    metrics = sam3d.view_metrics(src, entity, {})
    frames_all = src.objects[key]["frames"]
    out = {"frames": frames_all, "lowres": lowres_masks(src, key, frames_all), "views_with_depth": len(metrics)}
    if not metrics:
        return {**out, "eligible": False, "reason": "no mask with depth", "select_s": time.time() - t0}
    t_metrics = time.time() - t0
    rgb = dict(src.clip.frames({m["frame"] for m in metrics}))
    for m in metrics:
        gray = cv2.cvtColor(rgb[m["frame"]], cv2.COLOR_RGB2GRAY)
        (x0, y0), (x1, y1) = (np.reshape(m["clipBox"], (2, 2)) * np.diag(src.clip.clip_to_full)[:2] + src.clip.clip_to_full[:2, 2]).round().astype(int)
        m["sharpness"] = float(cv2.Laplacian(gray[max(y0, 0):y1 + 1, max(x0, 0):x1 + 1], cv2.CV_64F).var())
        depth, mask, _, _ = view_data(src, key, m["frame"])
        m["occluded"] = occlusion_share(mask, depth)
        m["fails"] = sorted(cvo.failing(m))
    top = max(m["sharpness"] for m in metrics)
    for m in metrics:
        occ = 0. if not np.isfinite(m["occluded"]) else m["occluded"]
        m["score"] = m["area"] * m["sharpness"] / max(top, 1e-9) * max(m["frontal"], .1) * min(m["solidity"], 1) ** 2 * (1 - occ)
    t_views = time.time() - t0 - t_metrics
    good = sorted([m for m in metrics if not set(m["fails"]) & GOOD_FAILS and not (m["occluded"] > OCCLUDED)], key=lambda m: -m["score"])
    centre = np.asarray(obj["centroid_m"], float)
    cam = lambda m: np.asarray(src.rows[m["frame"]]["c2w"])[:3, 3]  # noqa: E731
    gen, held = spread_views(good, centre, cam, min_sep=min_sep)
    if held is None and min_sep == 0 and (good or metrics):  # r5b integrate: a one-sided object still gets a model from its best view;
        pool = good or sorted(metrics, key=lambda m: -m["score"])  # its check view is that view (no held-out view exists)
        gen, held = pool[:MAX_GEN_VIEWS], pool[0]
        out["held_out"] = False
    out["directions_at_deg"] = {str(d): len(sum(spread_views(good, centre, cam, min_sep=d)[0:1], []) + ([1] if spread_views(good, centre, cam, min_sep=d)[1] else []))
                                for d in (5, 10, 15)}  # sensitivity of the rule: views a looser separation would take
    brief = lambda m: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items() if k in  # noqa: E731
                       ("frame", "area", "sharpness", "frontal", "solidity", "occluded", "score", "fails", "sourceShortSide")}
    out.update(good_views=len(good), gen=[brief(m) for m in gen], held=brief(held) if held else None,
               min_pair_deg=round(min((angle_deg(cam(a) - centre, cam(b) - centre) for i, a in enumerate(gen + [held]) for b in (gen + [held])[:i]),
                                      default=0.), 1) if held else None)
    if held is None:
        return {**out, "eligible": False, "reason": f"{len(good)} good view(s), fewer than 2 directions {min_sep:g} deg apart",
                "select_s": time.time() - t0}
    frames = dict(src.clip.frames({m["frame"] for m in gen + [held]}))
    # (a) RecGen on the generation views (a view whose crop keeps no mask with depth is dropped; the anchor first)
    views = [(m["frame"], recgen_view(src, m, frames[m["frame"]])) for m in gen]
    views = [(f, v) for f, v in views if v is not None]
    out["recgen_views"] = [f for f, _ in views]
    out["recgen_job"] = {"views": [v for _, v in views], "seed": SEED} if views else None
    # (b) SAM 3D on the single best view if today's prepare rule passes it; (c) on a rejection, the next-best that does
    rest = [m for m in sorted(metrics, key=lambda m: -m["score"]) if m["frame"] != held["frame"]]
    best = gen[0]
    chosen, method = (best, "sam3d_b") if not best["fails"] else (next((m for m in rest if not m["fails"] and m["frame"] != best["frame"]), None), "sam3d_c")
    out["sam3d_b_rejected"] = None if not best["fails"] else ", ".join(best["fails"])
    if chosen is not None:
        if chosen["frame"] not in frames:
            frames.update(src.clip.frames({chosen["frame"]}))
        payload = sam3d.build_input(src, chosen, frames[chosen["frame"]])
        out["sam3d_job"] = {"method": method, "frame": chosen["frame"], "rgb": payload["fullRgb"], "mask": payload["fullMask"].astype(bool),
                            "pointmap": cvo.sam3d_pointmap(payload["fullDepth"], payload["fullK"].astype(float)), "seed": cvo.SEED}
    else:
        out["sam3d_job"] = None
    # the held-out view's crop for the contact sheet
    hm = src.full_mask(held)
    left, top_, side = crop_square(hm)
    out["held_tile"] = jpeg(crop_rgb(frames[held["frame"]], left, top_, side))
    out["colour"] = np.median(frames[gen[0]["frame"]][src.full_mask(gen[0])], 0).tolist()  # a parametric shape's paint
    out["held_crop"] = [left, top_, side]
    return {**out, "eligible": True, "select_s": time.time() - t0,
            "select_parts_s": {"view_metrics": round(t_metrics, 2), "sharp_occlusion": round(t_views, 2), "jobs": round(time.time() - t0 - t_metrics - t_views, 2)}}


def observed_points(src, key, frames, cap=60000):
    pts = []
    for f in frames:
        depth, mask, k, c2w = view_data(src, key, f)
        pts.append(lift_points(depth, k, c2w, mask))
    pts = np.concatenate(pts) if pts else np.zeros((0, 3))
    if len(pts) > cap:
        pts = pts[np.random.default_rng(0).choice(len(pts), cap, replace=False)]
    return pts


def held_gate(src, key, held, vertices, faces):
    """build_lingbot_object_model.evaluate on the held-out view (DA3 grid), its thresholds unchanged; its 300-pixel floor
    (640x480 clip raster = 2.25 source px each) kept as the same source-frame area on this grid (MIN_SUPPORTED px, 6.53
    source px each). evaluate's own 300-on-this-grid verdict (fb/b's reading) stays as accepted_300px_on_grid."""
    from build_lingbot_object_model import evaluate
    depth, mask, k, c2w = view_data(src, key, held)
    g = evaluate(np.asarray(vertices, np.float64), np.asarray(faces), depth, np.full(depth.shape, 2., np.float32), depth > 0, mask, k, c2w)
    floor = min_supported(src.clip.full_size, depth.shape[::-1])
    g["accepted_300px_on_grid"] = g["accepted_source_consistency"]
    g["min_supported_pixels"] = floor
    g["accepted_source_consistency"] = bool(g["supported_pixels"] >= floor and g["silhouette_iou"] >= .65
                                            and g["relative_depth_median"] is not None and g["relative_depth_median"] <= .04 and g["relative_depth_p95"] <= .10)
    return g


def min_supported(full_wh, grid_wh, clip_px=300):
    """The delivered gate's pixel floor (300 px of the 640x480 raster of a 960x720 centre crop) as the same source-frame area on
    a grid covering the whole frame."""
    source_px = clip_px * (960 * 720) / (640 * 480) * (full_wh[1] / 720) ** 2
    return int(round(source_px / (full_wh[0] * full_wh[1] / (grid_wh[0] * grid_wh[1]))))


def render_tile(src, held, crop, vertices, faces, colors, observed):
    """The placed model rendered from the held-out camera into its crop (inferred faces tinted, complete_video_objects.render)."""
    import complete_video_objects as cvo
    left, top, side = crop
    k = src.clip.k_full.copy()
    k[0, 2] -= left
    k[1, 2] -= top
    scale = min(1., 320 / side)
    k[:2] *= scale
    size = max(8, int(round(side * scale)))
    c2w = np.asarray(src.rows[held]["c2w"], float)
    image, _ = cvo.render(np.asarray(vertices, np.float64), np.asarray(faces), np.asarray(colors, np.float64), observed, k, c2w, (size, size),
                          np.full((size, size, 3), 40, np.uint8))
    return jpeg(image)


def light(vertices, faces, colors, cells=80):
    """Vertex clustering (open3d, colours averaged) to cells^3 at most across the model's longest side: generators return
    0.2-1.2 M faces, a 504x280 view resolves far less; ~10x fewer faces for every BVH after this."""
    import open3d as o3d
    if len(faces) <= 60000:
        return vertices, faces, colors
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(np.asarray(vertices, np.float64)), o3d.utility.Vector3iVector(np.asarray(faces, np.int32)))
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.asarray(colors, np.float64) / 255)
    mesh = mesh.simplify_vertex_clustering(float(np.ptp(vertices, 0).max()) / cells).remove_degenerate_triangles().remove_unreferenced_vertices()
    return np.asarray(mesh.vertices), np.asarray(mesh.triangles, np.int64), np.asarray(mesh.vertex_colors) * 255


def display_glb(vertices, faces, colors, observed, path):
    """Decimated display copy (open3d vertex clustering to ~1/100 of the size), observed vertices opaque, the rest translucent."""
    import open3d as o3d
    import complete_video_objects as cvo
    from scipy.spatial import cKDTree
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(np.asarray(vertices, np.float64)), o3d.utility.Vector3iVector(np.asarray(faces, np.int32)))
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.asarray(colors, np.float64) / 255)
    if len(faces) > 60000:
        mesh = mesh.simplify_vertex_clustering(float(np.ptp(vertices, 0).max()) / 100)
    v, f, c = np.asarray(mesh.vertices), np.asarray(mesh.triangles), np.asarray(mesh.vertex_colors)
    alpha = np.where(observed, cvo.OBSERVED_ALPHA, cvo.INFERRED_ALPHA)[cKDTree(vertices).query(v)[1]]
    centre = v.mean(0)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(cvo.glb(v - centre, f, np.column_stack([(c * 255).round(), alpha]).astype(np.uint8), blend=True))
    return {"faces": int(len(f)), "centre": centre.round(4).tolist()}


def gate_model(src, msg, observed_voxel):
    """Place (source camera, then bounded refine on the generation views) and gate (held-out view) one generated mesh."""
    import complete_video_objects as cvo
    from scipy.spatial import cKDTree
    t0 = time.time()
    key, gen, held = msg["key"], msg["gen"], msg["held"]
    g = msg["mesh"]
    c2w = np.asarray(src.rows[msg["source_frame"]]["c2w"], float)
    if msg["kind"] == "recgen":  # posed in the anchor's OpenCV camera
        vertices = transformed(g["vertices"], c2w)
    else:  # SAM 3D: its pose puts the mesh in the pointmap's PyTorch3D camera
        vertices = cvo.sam3d_to_world(g["vertices"], g["objectToCamera"], c2w)
    faces, colors = np.asarray(g["faces"], np.int64), np.asarray(g["colors"], np.float64)
    t_light = time.time()
    vertices, faces, colors = light(vertices, faces, colors)  # placed, judged and shipped: the same light copy (the deliverable)
    t_light = time.time() - t_light
    views = []
    for f in gen:
        depth, mask, k, cw = view_data(src, key, f, .5)
        views.append({"rays": rays(k, cw, depth.shape[1], depth.shape[0]), "target": mask, "depth": depth})
    before = held_gate(src, key, held, vertices, faces)
    moved_by, placement = refine(vertices, faces, views)
    vertices = transformed(vertices, moved_by)
    gate = held_gate(src, key, held, vertices, faces)
    out = {"gate": {k: gate.get(k) for k in ("accepted_source_consistency", "silhouette_iou", "relative_depth_median", "relative_depth_p95",
                                            "supported_pixels", "min_supported_pixels", "accepted_300px_on_grid")},
           "gate_before_placement": {k: before.get(k) for k in ("accepted_source_consistency", "silhouette_iou", "relative_depth_median")},
           "placement": placement, "faces": int(len(faces)), "light_s": round(t_light, 3)}
    if gate["accepted_source_consistency"]:
        pts = observed_points(src, key, gen)
        observed = cKDTree(pts).query(vertices, distance_upper_bound=2 * observed_voxel)[0] < np.inf if len(pts) else np.zeros(len(vertices), bool)
        out["observed_vertex_share"] = round(float(observed.mean()), 3)
        out["tile"] = render_tile(src, held, msg["crop"], vertices, faces, colors, observed)
        out["glb"] = display_glb(vertices, faces, colors, observed, msg["glb_path"])
    return {**out, "gate_s": round(time.time() - t0, 3)}


def fit_param(src, msg):
    """A parametric shape from the generation views' points, gated on the held-out view like the models."""
    t0 = time.time()
    key, gen, held = msg["key"], msg["gen"], msg["held"]
    pts = observed_points(src, key, gen)
    if len(pts) < 50:
        return {"error": f"only {len(pts)} observed points", "gate_s": round(time.time() - t0, 3)}
    c = np.median(pts, 0)
    d = np.linalg.norm(pts - c, axis=1)
    pts = pts[d <= np.median(d) + 3 * 1.4826 * np.median(np.abs(d - np.median(d)))]  # far strays out (a mask's bleed onto the background)
    vertices, faces, record = parametric(msg["kind"], pts, msg["up"], np.asarray(msg["floor_point"], float))
    gate = held_gate(src, key, held, vertices, faces)
    out = {**record, "gate": {k: gate.get(k) for k in ("accepted_source_consistency", "silhouette_iou", "relative_depth_median",
                                                      "relative_depth_p95", "supported_pixels", "min_supported_pixels", "accepted_300px_on_grid")}}
    if gate["accepted_source_consistency"]:
        colors = np.tile(msg["colour"], (len(vertices), 1)).astype(np.float64)
        out["tile"] = render_tile(src, held, msg["crop"], vertices, faces, colors, np.ones(len(vertices), bool))
    return {**out, "gate_s": round(time.time() - t0, 3)}


def cpu_worker():
    """One CPU process (the /opt/gate venv): 'select', 'gate', 'fit' on the staged hand-off, sources cached per shot."""
    os.nice(5)
    import complete_video_objects as cvo
    from fast_report import sam3d
    sources = {}

    def source(spec, shot):
        key = (json.dumps(spec, sort_keys=True), shot)
        if key not in sources:
            cvo.VOXEL, cvo.OCCLUSION = sam3d.VOXEL_M, sam3d.VOXEL_M / 2  # the gate's tolerances in estimated metres (fb/b)
            cvo.FIT_GATE["max_fit_median_native"] = sam3d.VOXEL_M
            sources[key] = sam3d.FastSource(spec, shot)
        return sources[key]

    def handle(message, _):
        start = time.time()
        src = source(message["src"], message["shot"])
        op = message["op"]
        out = select(src, message["key"], message["obj"], message.get("min_sep", MIN_SEP_DEG)) if op == "select" else gate_model(src, message, sam3d.VOXEL_M) if op == "gate" \
            else fit_param(src, message)
        return {**out, "start_unix": start, "end_unix": time.time(), "pid": os.getpid()}
    def boot():  # resident code: the first object must not pay the imports
        import cv2
        cv2.setNumThreads(1)  # CPU_PROCS processes share the cores: no per-process thread pools
        import open3d  # noqa: F401
        import scipy.optimize  # noqa: F401
        import trimesh  # noqa: F401
        import build_lingbot_object_model  # noqa: F401
        from ehs_spatial.platform import recgen  # noqa: F401
        return {"ready": True, "pid": os.getpid()}
    sam3d.serve(handle, boot)


# ---------------------------------------------------------------- RecGen processes (/opt/recgen-venv)
def load_recgen():
    """lucida_assets.generate_object's loader: RecGen's pinned snapshot and DINOv2 from the weights volume (manifest pins
    checked; the files are not re-hashed on every start: ponytail, the volume is written once by prepare_weights)."""
    from unittest.mock import patch
    import torch
    from recgen_inference import build_recgen
    weights = json.loads((Path(RECGEN_CACHE) / "recgen-weights-manifest.json").read_text())
    assert weights["model_revision"] == RECGEN_REV and weights["code_revision"] == RECGEN_CODE, "RecGen pins changed"
    original = torch.hub.load

    def pinned_hf(repo_id, filename, *args, **kwargs):
        if repo_id != RECGEN_MODEL or args or filename not in weights["files"]:
            raise ValueError("unexpected Hugging Face request: " + filename)
        return str(Path(weights["snapshot"]) / filename)

    def pinned_dino(repo_or_dir, model, *args, **kwargs):
        if repo_or_dir != "facebookresearch/dinov2" or model != "dinov2_vitl14_reg":
            raise ValueError("unexpected torch.hub model")
        return original("/opt/dinov2", model, *args, source="local", weights=weights["dino"]["path"], **kwargs)
    with patch("huggingface_hub.hf_hub_download", pinned_hf), patch("torch.hub.load", pinned_dino):
        return build_recgen.build(RECGEN_NAME)


def run_recgen(pipeline, views, seed):
    """One RecGen call: generate (1 view) or generate_multiview (anchor first). The posed mesh is in the anchor's OpenCV camera."""
    import torch
    from recgen_inference import generate, generate_multiview
    started = time.monotonic()
    if len(views) == 1:
        v = views[0]
        result = generate(pipeline, image=v["rgb"], depth=v["depth"], mask=v["mask"], intrinsics=v["camera_intrinsics"], seed=seed,
                          mask_erosion_enabled=True)
    else:
        result = generate_multiview(pipeline, anchor_view=views[0], second_views=views[1:], seed=seed, mask_erosion_enabled=True)
    torch.cuda.synchronize()
    posed = result.mesh
    colors = np.asarray(posed.visual.vertex_colors)[:, :3] if hasattr(posed.visual, "vertex_colors") else np.full((len(posed.vertices), 3), 160)
    return {"vertices": np.asarray(posed.vertices, np.float32), "faces": np.asarray(posed.faces, np.uint32), "colors": colors.astype(np.uint8),
            "views": len(views), "seconds": time.monotonic() - started}


def synthetic_views(n=2, side=256):
    """A red box on a grey plane 2 m away, seen from n slightly shifted crops: the warm-up call's input."""
    views = []
    for i in range(n):
        rgb = np.full((side, side, 3), 90, np.uint8)
        rgb[80:180, 70 + 6 * i:190 + 6 * i] = (200, 60, 40)
        mask = np.zeros((side, side), np.uint8)
        mask[80:180, 70 + 6 * i:190 + 6 * i] = 255
        depth = np.where(mask > 0, 1.8, 2.).astype(np.float32)
        views.append({"rgb": rgb, "depth": depth, "mask": mask, "camera_intrinsics": np.array([[300., 0, side / 2], [0, 300., side / 2], [0, 0, 1]])})
    return views


def recgen_worker():
    import torch
    from fast_report.sam3d import serve
    state = {}

    def boot():
        t = time.time()
        state["pipeline"] = load_recgen()
        torch.cuda.synchronize()
        loaded = time.time() - t
        t, warm_error = time.time(), None
        try:  # the one-off kernel and allocator set-up; a synthetic input may fail where real ones do not: recorded, not fatal
            run_recgen(state["pipeline"], synthetic_views(2), SEED)
        except Exception as error:  # noqa: BLE001
            warm_error = repr(error)[-500:]
        warm_reserved = torch.cuda.max_memory_reserved()
        torch.cuda.empty_cache()
        return {"ready": True, "load_s": round(loaded, 1), "warm_s": round(time.time() - t, 1), "warm_error": warm_error, "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "reserved_after_warm_gb": round(warm_reserved / 2 ** 30, 2), "idle_reserved_gb": round(torch.cuda.memory_reserved() / 2 ** 30, 2)}

    def handle(message, _):
        torch.cuda.reset_peak_memory_stats()
        start = time.time()
        out = run_recgen(state["pipeline"], message["views"], message["seed"])
        peak = round(torch.cuda.max_memory_reserved() / 2 ** 30, 2)
        torch.cuda.empty_cache()  # r5: the call's cache back to the device (two processes beside the report's core took GPU 0 to 75.9 GiB)
        return {**out, "start_unix": start, "end_unix": time.time(), "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), "max_reserved_gb": peak}
    serve(handle, boot)


def with_recgen(image):
    """RecGen's official lock (lucida_assets.gpu_image) in its own venv: Python 3.10, torch 2.4.0 cu121, xformers 0.0.27.post2,
    spconv-cu120 2.3.6; code and DINOv2 at their pinned commits. Weights: the panoptes-lucida-weights volume at /cache."""
    py, idx = RECGEN_PY, "https://download.pytorch.org/whl/cu121"
    uv = f"uv pip install --python {py}"
    return image.run_commands(
        "uv venv --python 3.10 /opt/recgen-venv",
        f"{uv} torch==2.4.0 torchvision==0.19.0 --index-url {idx}",
        f"{uv} numpy==1.26.4 pillow==11.3.0 scipy==1.15.3 easydict==1.13 tqdm==4.67.1 safetensors==0.6.2 huggingface_hub==0.36.0 "
        "spconv-cu120==2.3.6 trimesh==4.7.4 plyfile==1.1.2 einops==0.8.1 opencv-python-headless==4.11.0.86",
        f"{uv} xformers==0.0.27.post2 --index-url {idx}",
        f"git clone https://github.com/TRI-ML/recgen.git {RECGEN_DIR} && git -C {RECGEN_DIR} checkout --detach {RECGEN_CODE}",
        f"git clone https://github.com/facebookresearch/dinov2.git /opt/dinov2 && git -C /opt/dinov2 checkout --detach {DINO_REV}",
        f"{uv} --no-deps -e {RECGEN_DIR}",
        f"ATTN_BACKEND=xformers SPCONV_ALGO=native PYTHONPATH={RECGEN_DIR} {py} -c 'import xformers.ops as xops; assert xops.fmha.BlockDiagonalMask; "
        "from recgen_inference import build_recgen, generate, generate_multiview; print(\"RecGen import ready\")'")


# ---------------------------------------------------------------- self-check (local, CPU)
def _scene(tmp, tilt_deg=0.):
    """A synthetic shot: a 0.6 x 0.4 x 1.0 m box on a floor, 7 keyframes on an arc around it; depth and masks raycast."""
    import open3d as o3d
    import trimesh
    from fast_report import sam3d
    up = np.array([0., -1., 0.])  # OpenCV-like world: y down
    box = trimesh.creation.box(extents=[.6, 1.0, .4])
    R = trimesh.transformations.rotation_matrix(np.radians(tilt_deg), [0, 0, 1])
    box.apply_transform(R)
    box.apply_translation([0, -.5, 3.])  # sits on the floor y = 0
    floor = trimesh.creation.box(extents=[8, .02, 8])
    floor.apply_translation([0, .01, 3.])
    scene_mesh = trimesh.util.concatenate([box, floor])
    k = np.array([[226.6, 0, 252], [0, 222.9, 140], [0, 0, 1.]])
    keys = [0, 6, 12, 18, 24, 30, 36]
    c2ws, depths, logits = [], [], []
    obj_cast = o3d.t.geometry.RaycastingScene()
    obj_cast.add_triangles(o3d.core.Tensor(np.asarray(box.vertices, np.float32)), o3d.core.Tensor(np.asarray(box.faces, np.uint32)))
    all_cast = o3d.t.geometry.RaycastingScene()
    all_cast.add_triangles(o3d.core.Tensor(np.asarray(scene_mesh.vertices, np.float32)), o3d.core.Tensor(np.asarray(scene_mesh.faces, np.uint32)))
    import cv2
    for i, _ in enumerate(keys):
        ang = np.radians(-45 + 15 * i)
        eye = np.array([3 * np.sin(ang), -1.4, 3 - 3 * np.cos(ang)])
        z = unit(np.array([0, -.5, 3.]) - eye)
        x = unit(np.cross([0, -1., 0], z) * -1)
        c2w = np.eye(4)
        c2w[:3, :3], c2w[:3, 3] = np.stack([x, np.cross(z, x), z], 1), eye
        r = o3d.t.geometry.RaycastingScene.create_rays_pinhole(o3d.core.Tensor(k), o3d.core.Tensor(np.linalg.inv(c2w)), 504, 280)
        d_all = all_cast.cast_rays(r)["t_hit"].numpy()
        d_obj = obj_cast.cast_rays(r)["t_hit"].numpy()
        depths.append(np.where(np.isfinite(d_all), d_all, 0).astype(np.float32))
        m = np.isfinite(d_obj) & (np.abs(d_obj - d_all) < 1e-3)
        c2ws.append(c2w)
        lg = np.where(cv2.resize(m.astype(np.uint8), (288, 288), interpolation=cv2.INTER_NEAREST) > 0, 8., -8.).astype(np.float16)
        logits.append(lg)
    rng = np.random.default_rng(0)
    frames = np.lib.format.open_memmap(tmp / "frames.npy", mode="w+", dtype=np.uint8, shape=(40, 720, 1280, 3))
    frames[:] = rng.integers(0, 255, (1, 720, 1280, 3), dtype=np.uint8)
    shot = {"index": 0, "keys": keys, "depth_m": np.stack(depths), "c2w_m": np.stack(c2ws), "K": np.repeat(k[None], len(keys), 0),
            "person": np.zeros((len(keys), 280, 504), bool)}
    lo, hi = box.bounds
    obj = {"id": "box", "shot": 0, "word": "box", "box_min_m": lo.tolist(), "box_max_m": hi.tolist(), "centroid_m": box.centroid.tolist(),
           "masks_lr": {f: logits[i] for i, f in enumerate(keys)}}
    spec, names = sam3d.stage([obj], [shot], str(tmp / "frames.npy"), tmp / "staged")
    return spec, names["box"], obj, box, up


def self_check():
    import tempfile
    import complete_video_objects as cvo
    from fast_report import sam3d
    # spread: 10-degree steps -> every other view, the last one taken is held out
    cams = {i: np.array([np.sin(np.radians(10 * i)), 0, np.cos(np.radians(10 * i))]) for i in range(8)}
    gen, held = spread_views(list(range(8)), np.zeros(3), lambda m: cams[m])
    assert gen == [0, 2, 4, 6] and held is None or (gen, held) == ([0, 2, 4], 6), (gen, held)
    assert spread_views([0, 1], np.zeros(3), lambda m: cams[m]) == ([0], None)
    # occlusion: a bar in front of a box's left edge
    depth = np.full((60, 80), 5., np.float32)
    mask = np.zeros((60, 80), bool)
    mask[20:40, 20:60] = True
    depth[mask] = 3.
    assert occlusion_share(mask, depth) == 0.
    depth[15:45, 14:20] = 1.  # a bar just left of the box, in front of it
    assert occlusion_share(mask, depth) > .1
    # parametric: a post leaning 6 deg, a partition leaning 4 deg, a box 1.2 m tall on the floor
    rng = np.random.default_rng(1)
    up = np.array([0., 0, 1])
    t = np.radians(6)
    axis = np.array([np.sin(t), 0, np.cos(t)])
    s, th = rng.uniform(0, 2, 4000), rng.uniform(-np.pi / 2, np.pi / 2, 4000)  # the half the camera sees
    _, pa, pb = floor_frame(axis)
    post = s[:, None] * axis + .1 * (np.cos(th)[:, None] * pa + np.sin(th)[:, None] * pb)
    _, _, rec = parametric("post", post + rng.normal(0, .003, post.shape), up, np.zeros(3), boots=10)
    assert abs(rec["facts"]["tilt_from_vertical_deg"]["value"] - 6) < 1 and abs(rec["facts"]["radius_m"]["value"] - .1) < .02, rec
    lean = np.radians(4)
    wall = np.c_[rng.uniform(-1, 1, 3000), np.zeros(3000), rng.uniform(0, 2, 3000)]
    wall = wall @ np.array([[1, 0, 0], [0, np.cos(lean), np.sin(lean)], [0, -np.sin(lean), np.cos(lean)]]).T
    _, _, rec = parametric("partition", wall, up, np.zeros(3), boots=10)
    assert abs(rec["facts"]["tilt_from_vertical_deg"]["value"] - 4) < .5, rec
    cube = rng.uniform([0, 0, 0], [.5, .4, 1.2], (3000, 3))
    cube[:, 0] = np.where(rng.random(3000) < .5, 0, cube[:, 0])  # two faces seen
    cube[:, 1] = np.where(rng.random(3000) < .5, 0, cube[:, 1])
    _, _, rec = parametric("box", cube, up, np.zeros(3), boots=5)
    assert abs(rec["facts"]["top_above_floor_m"]["value"] - 1.2) < .03 and rec["residual_median_m"] < .01, rec
    assert min_supported((1280, 720), (504, 280)) == 103 and min_supported((1280, 720), (1280, 720)) == 675  # 300 x 2.25 / 6.53; 300 x 2.25
    assert param_kind(["carton stack"]) == "box" and param_kind(["pallet rack"]) == "shelf" and param_kind(["drill press"]) is None
    # the whole path on a synthetic shot: selection, the true box placed and gated, a displaced box refined, a wrong one rejected
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        spec, key, obj, box, up_world = _scene(tmp)
        cvo.VOXEL, cvo.OCCLUSION = sam3d.VOXEL_M, sam3d.VOXEL_M / 2
        src = sam3d.FastSource(spec, 0)
        sel = select(src, key, obj)
        assert sel["eligible"] and len(sel["gen"]) >= 2 and sel["held"]["frame"] not in [g["frame"] for g in sel["gen"]], sel.get("reason")
        assert sel["min_pair_deg"] >= MIN_SEP_DEG, sel["min_pair_deg"]
        assert sel["recgen_job"] and sel["recgen_job"]["views"][0]["mask"].dtype == np.uint8 and sel["sam3d_job"]["method"] == "sam3d_b"
        gen, held = [g["frame"] for g in sel["gen"]], sel["held"]["frame"]
        g = held_gate(src, key, held, box.vertices, box.faces)
        assert g["accepted_source_consistency"] and g["silhouette_iou"] > .9, g
        shifted = np.asarray(box.vertices) + [.12, 0, 0]
        views = []
        for f in gen:
            d, m, k, cw = view_data(src, key, f, .5)
            views.append({"rays": rays(k, cw, d.shape[1], d.shape[0]), "target": m, "depth": d})
        moved, rec = refine(shifted, np.asarray(box.faces), views)
        assert rec["moved"] and np.mean(rec["iou_after"]) > np.mean(rec["iou_before"]) + .05, rec
        assert held_gate(src, key, held, transformed(shifted, moved), box.faces)["silhouette_iou"] > held_gate(src, key, held, shifted, box.faces)["silhouette_iou"]
        wrong = np.asarray(box.vertices) * [1, .3, 1] + [0, -.35, 0]
        assert not held_gate(src, key, held, wrong, box.faces)["accepted_source_consistency"]
        pts = observed_points(src, key, gen)
        _, _, rec = parametric("box", pts, up_world, np.zeros(3), boots=5)
        assert abs(rec["facts"]["top_above_floor_m"]["value"] - 1.0) < .05 and rec["residual_median_m"] < .02, rec
    print("fast_report.x7 self-check passed: spread, occlusion, post/partition/box fits, select, held-out gate, bounded refine")


if __name__ == "__main__":
    assert sys.argv[1:] == ["--self-check"], __doc__
    self_check()
