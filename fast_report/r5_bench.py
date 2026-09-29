"""r5 (models): the model bench's CPU work, in the gate venv's processes (fast_report.sam3d.Pool), on an r5 report dump
(fast_report.fixture) staged like the SAM 3D gate's (sam3d.stage). Per card (op in brackets):
  surface   (a) the observed-surface mesh and (d) its planar parts from the card's own points, each timed (every card)
  heldout   X7's view selection's held-out view (never used below): (a) rebuilt without every view within the separation of it,
            the r4 primitive refitted on the same points, (c) the splat cut by the card's box; each rendered from the held-out
            camera and judged by X7's gate (silhouette IoU >= 0.65, relative depth p50 <= 0.04, p95 <= 0.10, pixel floor)
  gate      a posed generated mesh (RecGen: its anchor camera; SAM 3D: its pointmap pose): X7's bounded placement on the
            generation views, the same gate, a tile, and the workcell measurement of the placed surface against (d)
  gate_free a pose-free mesh (TRELLIS, TripoSR; canonical z up): upright yaw search scaled by the anchor view's silhouette and
            moved onto the observed points, then the same placement, gate, colours from the generation views, tile, measurement

    python -m fast_report.r5_bench --self-check      # CPU: alignment of a pose-free mesh, primitive triangles, splat crop
"""
import json
import os
from pathlib import Path
import time

import numpy as np

HELD_SEP_DEG = 15.
TILE = 160


# ---------------------------------------------------------------- the card's own data
def card_data(fx, card, src):
    """World points (shot frame), their local keyframes, the floor frame, floor-frame points (the card's main cluster), cameras."""
    from fast_report import cards, fixture
    world, frame = fixture.card_points(fx, card)
    keys = list(src.rows)
    c2w = np.stack([src.rows[k]["c2w"] for k in keys])
    sh = next(s for s in fx["meta"]["cards_shots"] if s["index"] == card["shot"])
    fr = cards.floor_frame(c2w[0], np.asarray(sh["normal"], float), np.asarray(sh["point_m"], float))
    z = np.einsum("ni,ni->n", world - c2w[frame, :3, 3], c2w[frame, :3, 2]) if len(world) else np.zeros(0)
    x = cards.prepare({"world": world, "frame": frame, "z": z}, fr, float(src.clip.k_raster[0, 0]))
    keep = np.isin(np.arange(len(world)), np.flatnonzero(np.isin(frame, np.unique(x["frame"])))) if len(world) else np.zeros(0, bool)
    # the main cluster in world coordinates: floor -> shot
    Pw = x["P"] @ fr["R"] + fr["origin"] if len(x["P"]) else np.zeros((0, 3))
    at = {k: i for i, k in enumerate(keys)}
    subsets = [[at[k] for k in sv if k in at] for sv in card.get("views", {}).get("subsets", [])]
    return {"world": Pw, "frame": np.asarray(x["frame"]), "P": x["P"], "fr": fr, "keys": keys, "c2w": c2w,
            "K": np.repeat(src.clip.k_raster[None], len(keys), 0), "cams": cards.to_floor(c2w[:, :3, 3], fr), "subsets": subsets, "raw_n": int(keep.sum())}


def rgb_of(src, keys):
    return lambda v: np.asarray(src.frames[keys[v]])  # BGR, as decoded


def surface(src, fx, card, plumb_u):
    from fast_report import surface as sf
    d = card_data(fx, card, src)
    t = time.perf_counter()
    V, F, C, info = sf.observed_mesh(d["world"], d["frame"], d["c2w"], d["K"], rgb=rgb_of(src, d["keys"]))
    t_mesh = time.perf_counter() - t
    t = time.perf_counter()
    parts = sf.planar_parts(d["P"], d["frame"], d["subsets"], d["cams"], plumb_u)
    return {"points": int(len(d["P"])), "mesh": {**info, "s": round(t_mesh, 4)}, "parts": parts, "parts_s": round(time.perf_counter() - t, 4)}


PART_BGR = ((60, 200, 255), (255, 120, 40), (80, 220, 80), (220, 80, 220), (40, 40, 230), (230, 230, 60))


def parts_tile(src, fx, card, frame_key, side=320):
    """The card's planar parts drawn on one of its best views: each part's points in its own colour, its tilt to the floor
    (value +- u) and the angle between touching parts written on top (for looking at tilted parts by eye)."""
    import cv2
    from fast_report import surface as sf
    from fast_report.x7 import crop_rgb
    d = card_data(fx, card, src)
    out = sf.planar_parts(d["P"], d["frame"], d["subsets"], d["cams"], keep=True)
    img = np.asarray(src.frames[frame_key]).copy()  # BGR
    c2w = np.asarray(src.rows[frame_key]["c2w"], float)
    k = src.clip.k_full
    if "parts" not in out:
        return None
    W = out["_points"] @ d["fr"]["R"] + d["fr"]["origin"]
    cam = (W - c2w[:3, 3]) @ c2w[:3, :3]
    ok = cam[:, 2] > .05
    uv = cam[:, :2] / np.maximum(cam[:, 2:], 1e-6) * [k[0, 0], k[1, 1]] + k[:2, 2]
    for i in range(len(out["parts"])):
        for x, y in uv[ok & (out["_labels"] == i)]:
            cv2.circle(img, (int(x), int(y)), 2, PART_BGR[i % len(PART_BGR)], -1)
    inb = ok & (uv[:, 0] > 0) & (uv[:, 1] > 0) & (uv[:, 0] < img.shape[1]) & (uv[:, 1] < img.shape[0])
    if not inb.any():
        return None
    lo, hi = uv[inb].min(0), uv[inb].max(0)
    s = int(max(hi - lo) * 1.3) + 20
    left, top = int((lo[0] + hi[0]) / 2 - s / 2), int((lo[1] + hi[1]) / 2 - s / 2)
    crop = cv2.resize(crop_rgb(img, left, top, s), (side, side), interpolation=cv2.INTER_AREA)
    y = 14
    for i, p in enumerate(out["parts"]):
        t = p["tilt_deg"]
        cv2.putText(crop, f"{i + 1}: {t['value']:.0f}+-{t['u']:.0f}", (4, y), cv2.FONT_HERSHEY_SIMPLEX, .45, PART_BGR[i % len(PART_BGR)], 1, cv2.LINE_AA)
        y += 16
    for b in out["bends"][:3]:
        cv2.putText(crop, f"{b['parts'][0] + 1}-{b['parts'][1] + 1}: {b['angle_deg']['value']:.0f}+-{b['angle_deg']['u']:.0f}", (4, y),
                    cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1, cv2.LINE_AA)
        y += 16
    return cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()


# ---------------------------------------------------------------- held-out tests
def held_views(d, src, held, sep=HELD_SEP_DEG):
    """Local keyframes of the card's points more than `sep` from the held-out view (camera direction from the object's centre)."""
    from fast_report.x7 import angle_deg
    c = np.median(d["world"], 0)
    h = src.rows[held]["c2w"][:3, 3] - c
    return [v for v in np.unique(d["frame"]).tolist() if angle_deg(d["c2w"][v][:3, 3] - c, h) > sep]


def primitive_tris(rec):
    import r4_models
    T, seen = r4_models.model_tris(rec)
    V = T.reshape(-1, 3)
    return V, np.arange(len(V)).reshape(-1, 3), np.repeat(np.asarray(seen, bool), 3)


def tile(src, held, crop, vertices, faces, colors, observed, dim=.35):
    """The model from the held-out camera over the dimmed held-out crop (unseen faces tinted, complete_video_objects.render)."""
    import complete_video_objects as cvo
    from fast_report.x7 import crop_rgb, jpeg
    left, top, side = crop
    k = src.clip.k_full.copy()
    k[0, 2] -= left
    k[1, 2] -= top
    s = TILE / side
    k[:2] *= s
    rgb = np.asarray(src.frames[held])[..., ::-1]
    bg = cvo_resize(crop_rgb(rgb, left, top, side), TILE) * dim
    if len(faces) == 0:
        return jpeg(bg.astype(np.uint8), TILE)
    img, _ = cvo.render(np.asarray(vertices, np.float64), np.asarray(faces), np.asarray(colors, np.float64), np.asarray(observed, bool), k,
                        np.asarray(src.rows[held]["c2w"], float), (TILE, TILE), bg)
    return jpeg(img, TILE)


def cvo_resize(img, side):
    import cv2
    return cv2.resize(np.ascontiguousarray(img), (side, side), interpolation=cv2.INTER_AREA).astype(np.float64)


def gate_row(g):
    return {k: g.get(k) for k in ("accepted_source_consistency", "silhouette_iou", "relative_depth_median", "relative_depth_p95", "supported_pixels",
                                  "min_supported_pixels")}


def heldout(src, fx, card, key, gen, held, crop, splat=None, sep=HELD_SEP_DEG):
    from fast_report import display_model, surface as sf
    from fast_report.x7 import held_gate
    d = card_data(fx, card, src)
    use = held_views(d, src, held, sep)
    out = {"views_all": int(len(np.unique(d["frame"]))), "views_used": len(use), "tiles": {}}
    t = time.perf_counter()
    V, F, C, info = sf.observed_mesh(d["world"], d["frame"], d["c2w"], d["K"], rgb=rgb_of(src, d["keys"]), views=use)
    out["observed"] = {**info, "s": round(time.perf_counter() - t, 4)}
    if len(F):
        out["observed"]["gate"] = gate_row(held_gate(src, key, held, V, F))
        out["tiles"]["observed"] = tile(src, held, crop, V, F, C, np.ones(len(V), bool))
    # the r4 primitive refitted on the same points (the card's type decides the shape, as display_model.for_card)
    keep = np.isin(d["frame"], use)
    if keep.sum() >= 10:
        fs = display_model.fits(d["P"][keep], d["cams"][np.unique(d["frame"][keep])])
        rec = display_model.record(fs, *display_model.choose(fs, (card.get("class") or {}).get("class_word")), d["fr"])
        PV, PF, seen = primitive_tris(rec)
        out["primitive"] = {"kind": rec["kind"], "gate": gate_row(held_gate(src, key, held, PV, PF))}
        out["tiles"]["primitive"] = tile(src, held, crop, PV, PF, np.tile([255, 184, 77], (len(PV), 1)), seen)
    if splat is not None and card["shot"] == splat["shot"]:
        out["splat"], out["tiles"]["splat"] = splat_crop(src, key, held, crop, card, splat)
    return out


# ---------------------------------------------------------------- (c) the splat cut by the card's box
def splat_crop(src, key, held, crop, card, sp, grow=1.1, margin=.03):
    """Gaussians whose centre lies in the card's drawn box (x grow + margin), drawn far to near as discs (2 sigma) from the held-out
    camera: silhouette IoU with the held-out mask (DA3 grid) and the bleed (share of the drawn alpha outside the mask dilated 3 px).
    The splat trained on every keyframe of its shot: a look test, not a held-out one."""
    import cv2
    from fast_report.x7 import view_data
    from scipy.spatial.transform import Rotation
    b = card["physical"]["box"]
    R = Rotation.from_quat(b["quaternion"]).as_matrix()
    loc = (sp["positions"] - b["center_m"]) @ R
    inside = np.all(np.abs(loc) <= np.asarray(b["size_m"]) * grow / 2 + margin, axis=1)
    depth, mask, k, c2w = view_data(src, key, held)
    h, w = mask.shape

    def draw(K, size, bg=None):
        cam = (sp["positions"][inside] - c2w[:3, 3]) @ c2w[:3, :3]
        z = cam[:, 2]
        ok = z > .05
        cam, z = cam[ok], z[ok]
        uv = cam[:, :2] / z[:, None] * [K[0, 0], K[1, 1]] + K[:2, 2]
        r = np.clip(2 * sp["scales"][inside][ok].mean(1) * K[0, 0] / z, .5, size[0] / 4)
        col, a = sp["rgb"][inside][ok] * 255, sp["opacity"][inside][ok]
        img = np.zeros((size[1], size[0], 3)) if bg is None else bg.astype(np.float64).copy()
        alpha = np.zeros((size[1], size[0]))
        for i in np.argsort(-z):
            m = np.zeros((size[1], size[0]), np.uint8)
            cv2.circle(m, tuple(np.round(uv[i] * 4).astype(int)), int(round(r[i] * 4)), 1, -1, cv2.LINE_8, 2)
            s = m > 0
            img[s] = img[s] * (1 - a[i]) + col[i][::-1] * a[i]
            alpha[s] = alpha[s] * (1 - a[i]) + a[i]
        return img, alpha
    _, alpha = draw(k, (w, h))
    sil = alpha > .5
    iou = float((sil & mask).sum() / max((sil | mask).sum(), 1))
    grown = cv2.dilate(mask.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    bleed = float(alpha[~grown].sum() / max(alpha.sum(), 1e-9))
    left, top, side = crop
    kt = src.clip.k_full.copy()
    kt[0, 2] -= left
    kt[1, 2] -= top
    kt[:2] *= TILE / side
    from fast_report.x7 import crop_rgb, jpeg
    bg = cvo_resize(crop_rgb(np.asarray(src.frames[held])[..., ::-1], left, top, side), TILE) * .35
    img, _ = draw(kt, (TILE, TILE), bg[..., ::-1])
    return {"gaussians": int(inside.sum()), "silhouette_iou": round(iou, 3), "bleed": round(bleed, 3)}, jpeg(img[..., ::-1].clip(0, 255).astype(np.uint8), TILE)


# ---------------------------------------------------------------- generated meshes
def observed_flags(src, key, gen, vertices, voxel):
    from scipy.spatial import cKDTree
    from fast_report.x7 import observed_points
    pts = observed_points(src, key, gen)
    return cKDTree(pts).query(vertices, distance_upper_bound=2 * voxel)[0] < np.inf if len(pts) else np.zeros(len(vertices), bool)


def paint(src, key, views, vertices, faces, fallback):
    """Per-vertex colour from the generation views where the vertex is in sight and inside the mask (ray-cast), else `fallback`."""
    import open3d as o3d
    from fast_report.x7 import view_data
    V = np.asarray(vertices, np.float64)
    acc, cnt = np.zeros((len(V), 3)), np.zeros(len(V))
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(V.astype(np.float32)), o3d.core.Tensor(np.asarray(faces, np.uint32)))
    for f in views:
        _, mask, k, c2w = view_data(src, key, f)
        cam = (V - c2w[:3, 3]) @ c2w[:3, :3]
        z = cam[:, 2]
        u, v = k[0, 0] * cam[:, 0] / np.maximum(z, 1e-6) + k[0, 2], k[1, 1] * cam[:, 1] / np.maximum(z, 1e-6) + k[1, 2]
        ok = (z > .05) & (u >= 0) & (v >= 0) & (u < mask.shape[1] - 1) & (v < mask.shape[0] - 1)
        ok[ok] &= mask[np.round(v[ok]).astype(int), np.round(u[ok]).astype(int)]
        d = V[ok] - c2w[:3, 3]
        dist = np.linalg.norm(d, axis=1)
        rays = np.c_[np.repeat(c2w[None, :3, 3], len(d), 0), d / dist[:, None]].astype(np.float32)
        hit = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
        vis = np.flatnonzero(ok)[hit >= dist - max(.01, .01 * float(np.median(dist)))]
        img = np.asarray(src.frames[f])
        H, W = img.shape[:2]
        px = np.clip(np.round(u[vis] * W / mask.shape[1]).astype(int), 0, W - 1)
        py = np.clip(np.round(v[vis] * H / mask.shape[0]).astype(int), 0, H - 1)
        acc[vis] += img[py, px, ::-1]
        cnt[vis] += 1
    out = np.asarray(fallback, np.float64).copy() if fallback is not None else np.full((len(V), 3), 150.)
    out[cnt > 0] = acc[cnt > 0] / cnt[cnt > 0, None]
    return out, cnt > 0


def align_free(src, key, gen, up, mesh, yaws=24):
    """A canonical (z up, or y up) mesh onto the observed points of the generation views: for each up axis and yaw, the scale that
    matches the anchor view's silhouette size and a translation-only ICP of the observed points onto its surface; the best by
    the observed points' median distance to the surface. -> (4x4 transform, record)."""
    import open3d as o3d
    from fast_report.x7 import observed_points, view_data
    V0 = np.asarray(mesh["vertices"], np.float64)
    F = np.asarray(mesh["faces"], np.int64)
    O = observed_points(src, key, gen, cap=8000)
    if len(O) < 30 or len(F) == 0:
        return None, {"error": f"{len(O)} observed points / {len(F)} faces"}
    _, mask, k, c2w = view_data(src, key, gen[0])
    ys, xs = np.nonzero(mask)
    zc = float(np.median((O - c2w[:3, 3]) @ c2w[:3, 2]))
    real = np.array([np.ptp(xs) / k[0, 0], np.ptp(ys) / k[1, 1]]) * zc  # silhouette width, height at the object's depth (m)
    up = np.asarray(up, float) / np.linalg.norm(up)
    right0 = np.cross(c2w[:3, 2], up)
    right0 /= max(np.linalg.norm(right0), 1e-9)
    fwd0 = np.cross(up, right0)
    base = np.stack([right0, fwd0, up], 1)  # canonical x, y, z -> world (yaw 0: canonical x along the camera's right)
    tm = o3d.t.geometry.TriangleMesh()
    tm.vertex.positions = o3d.core.Tensor(V0.astype(np.float32))
    tm.triangle.indices = o3d.core.Tensor(F.astype(np.int32))
    best = None
    for up_axis, U in (("z", np.eye(3)), ("y", np.array([[1., 0, 0], [0, 0, -1], [0, 1, 0]]))):
        for yi in range(yaws):
            a = 2 * np.pi * yi / yaws
            Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1.]])
            R = base @ Rz @ U
            Vr = V0 @ R.T
            cam = Vr @ c2w[:3, :3]
            ext = np.ptp(cam[:, :2], 0)
            s = float(np.median(real / np.maximum(ext, 1e-6)))
            Vw = s * Vr
            m = tm.clone()
            m.vertex.positions = o3d.core.Tensor(Vw.astype(np.float32))
            scene = o3d.t.geometry.RaycastingScene()
            scene.add_triangles(m)
            t = np.median(O, 0) - np.median(Vw, 0)
            for _ in range(6):  # translation-only ICP: the observed points onto the surface
                q = scene.compute_closest_points(o3d.core.Tensor((O - t).astype(np.float32)))["points"].numpy()
                t = t + np.median(O - t - q, 0)
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = s * R, t
            dist = scene.compute_distance(o3d.core.Tensor((O - t).astype(np.float32))).numpy()
            score = float(np.median(dist)) / max(float(np.ptp(O, 0).max()), 1e-6)
            if best is None or score < best[0]:
                best = (score, T, up_axis, a, s)
    score, T, up_axis, a, s = best
    return T, {"up_axis": up_axis, "yaw_deg": round(float(np.degrees(a)), 1), "scale": round(s, 4), "median_distance_rel": round(score, 4)}


def gate_mesh(src, fx, card, msg):
    """Place (posed: the source camera; pose-free: align_free), X7's bounded refine on the generation views, the held-out gate,
    colours, a tile, and the workcell measurement against (d)."""
    from fast_report import sam3d
    from fast_report.x7 import held_gate, light, rays, refine, transformed, view_data
    import complete_video_objects as cvo
    t0 = time.time()
    key, gen, held, g = msg["key"], msg["gen"], msg["held"], msg["mesh"]
    out = {}
    faces, colors = np.asarray(g["faces"], np.int64), (np.asarray(g["colors"], np.float64) if g.get("colors") is not None else None)
    if msg["kind"] == "recgen":
        vertices = transformed(g["vertices"], np.asarray(src.rows[msg["source_frame"]]["c2w"], float))
    elif msg["kind"] == "sam3d":
        vertices = cvo.sam3d_to_world(g["vertices"], g["objectToCamera"], np.asarray(src.rows[msg["source_frame"]]["c2w"], float))
    else:
        T, out["align"] = align_free(src, key, gen, msg["up"], g)
        if T is None:
            return {**out, "error": out["align"]["error"], "gate_s": round(time.time() - t0, 3)}
        vertices = transformed(g["vertices"], T)
    if colors is None:
        colors = np.full((len(vertices), 3), 150.)
    vertices, faces, colors = light(vertices, faces, colors)
    views = []
    for f in gen:
        depth, mask, k, cw = view_data(src, key, f, .5)
        views.append({"rays": rays(k, cw, depth.shape[1], depth.shape[0]), "target": mask, "depth": depth})
    out["gate_before_placement"] = gate_row(held_gate(src, key, held, vertices, faces))
    moved, out["placement"] = refine(vertices, faces, views)
    vertices = transformed(vertices, moved)
    out["gate"] = gate_row(held_gate(src, key, held, vertices, faces))
    if msg["kind"] in ("trellis", "triposr"):
        colors, _ = paint(src, key, gen, vertices, faces, colors if msg["kind"] == "triposr" else None)
    obs = observed_flags(src, key, gen, vertices, sam3d.VOXEL_M)
    out["observed_vertex_share"] = round(float(obs.mean()), 3)
    out["tile"] = tile(src, held, msg["crop"], vertices, faces, colors, obs)
    out["faces"] = int(len(faces))
    out["gate_s"] = round(time.time() - t0, 3)
    if out["gate"]["accepted_source_consistency"]:
        out["measure"] = measure_against_parts(src, fx, card, vertices, faces, obs)
    return out


def measure_against_parts(src, fx, card, vertices, faces, observed):
    """The workcell's measurement (scene_measurements.fitted_plane) of the placed model's surface next to each observed planar
    part (d), and whether the model's faces there were seen: the angle a generated model gives where its faces agree with the
    observed points, next to the observed one. The model's unseen faces are listed apart: never a measurement."""
    from ehs_spatial.platform.scene_measurements import fitted_plane
    from fast_report import surface as sf
    d = card_data(fx, card, src)
    parts = sf.planar_parts(d["P"], d["frame"], d["subsets"], d["cams"])
    if "parts" not in parts:
        return {"parts": [], "reason": parts.get("reason")}
    Vf = (np.asarray(vertices) - d["fr"]["origin"]) @ d["fr"]["R"].T  # shot -> floor
    tri = Vf[np.asarray(faces)]
    cen = tri.mean(1)
    seen_face = np.asarray(observed)[np.asarray(faces)].all(1)
    rows = []
    for p in parts["parts"]:
        n, c = np.asarray(p["normal"]), np.asarray(p["centre_m"])
        near = (np.abs((cen - c) @ n) <= .04) & (np.linalg.norm(cen - c, axis=1) <= .6 * np.linalg.norm(p["sides_m"]) + .05)
        row = {"observed_tilt": p["tilt_deg"]["value"], "observed_u": p["tilt_deg"]["u"], "model_faces": int(near.sum()),
               "model_faces_seen_share": round(float(seen_face[near].mean()), 3) if near.any() else None}
        try:
            fp = fitted_plane(tri[near]) if near.sum() >= 3 else None
            row["model_tilt"] = round(sf.tilt_deg(fp["normal"]), 2) if fp else None
        except Exception as e:  # noqa: BLE001  the workcell's own refusal (no stable plane)
            row["model_tilt"], row["refused"] = None, str(e)[:80]
        rows.append(row)
    return {"parts": rows}


# ---------------------------------------------------------------- the worker
def cpu_worker():
    os.nice(5)
    import complete_video_objects as cvo
    from fast_report import fixture, sam3d, x7
    sources, fixtures, splats = {}, {}, {}

    def source(spec, shot):
        k = (json.dumps(spec, sort_keys=True), shot)
        if k not in sources:
            cvo.VOXEL, cvo.OCCLUSION = sam3d.VOXEL_M, sam3d.VOXEL_M / 2
            cvo.FIT_GATE["max_fit_median_native"] = sam3d.VOXEL_M
            sources[k] = sam3d.FastSource(spec, shot)
        return sources[k]

    def fx_of(path):
        if path not in fixtures:
            fixtures[path] = fixture.load(path, light=True)
        return fixtures[path]

    def splat_of(path):
        if path and path not in splats:
            z = np.load(path)
            splats[path] = {k: z[k] for k in z.files} | {"shot": int(z["shot"])}
        return splats.get(path)

    def handle(m, _):
        start = time.time()
        src = source(m["src"], m["shot"])
        op = m["op"]
        if op == "select":
            out = x7.select(src, m["key"], m["obj"], m.get("min_sep", HELD_SEP_DEG))
            if not m.get("jobs"):  # the CPU arm: no generator inputs back (a SAM 3D pointmap is 11 MB)
                for k in ("recgen_job", "sam3d_job", "lowres"):
                    out.pop(k, None)
            elif out.get("eligible"):
                from fast_report.gen3d import rgba_crop
                full = [src.full_mask({"observation": f"{m['key']}:{g['frame']}:0", "frame": g["frame"]}) for g in out["gen"]]
                out["rgba"] = [rgba_crop(np.asarray(src.frames[g["frame"]])[..., ::-1], f) for g, f in zip(out["gen"], full) if f.any()]
                out.pop("lowres", None)
        else:
            fx = fx_of(m["fixture"])
            card = m["card"]
            if op == "surface":
                out = surface(src, fx, card, m.get("plumb_u"))
            elif op == "parts_tile":
                out = {"tile": parts_tile(src, fx, card, m["frame_key"])}
            elif op == "heldout":
                out = heldout(src, fx, card, m["key"], m["gen"], m["held"], m["crop"], splat_of(m.get("splat")), m.get("sep", HELD_SEP_DEG))
            else:
                out = gate_mesh(src, fx, card, m)
        return {**out, "start_unix": start, "end_unix": time.time(), "pid": os.getpid()}

    def boot():
        import cv2
        cv2.setNumThreads(1)
        import open3d  # noqa: F401
        import scipy.optimize  # noqa: F401
        import build_lingbot_object_model  # noqa: F401
        from ehs_spatial.platform import recgen, scene_measurements  # noqa: F401
        from fast_report import cards, display_model, surface as sf  # noqa: F401
        return {"ready": True, "pid": os.getpid()}
    sam3d.serve(handle, boot)


# ---------------------------------------------------------------- self-check (CPU, synthetic)
def self_check():
    import open3d as o3d
    # a pose-free box (canonical z up, unit size) found back upright on its observed front: yaw and scale
    box = o3d.geometry.TriangleMesh.create_box(.6, .4, 1.).translate([-.3, -.2, -.5])
    V, F = np.asarray(box.vertices), np.asarray(box.triangles)

    class Src:  # the few FastSource fields align_free reads, through x7's helpers (patched below)
        pass
    import fast_report.x7 as x7m
    rng = np.random.default_rng(0)
    true_s, true_yaw, true_t = 1.3, np.radians(30), np.array([2., 5., .65])
    Rz = np.array([[np.cos(true_yaw), -np.sin(true_yaw), 0], [np.sin(true_yaw), np.cos(true_yaw), 0], [0, 0, 1.]])
    c2w = np.eye(4)
    c2w[:3, :3] = np.array([[1., 0, 0], [0, 0, 1], [0, -1, 0]])  # looking along +y, image y down = -z
    c2w[:3, 3] = [2., 1., .9]
    k = np.array([[300., 0, 252], [0, 300., 140], [0, 0, 1]])
    front = np.c_[rng.uniform(-.3, .3, 3000), np.full(3000, -.2), rng.uniform(-.5, .5, 3000)] @ Rz.T * true_s + true_t
    side = np.c_[np.full(1500, .3), rng.uniform(-.2, .2, 1500), rng.uniform(-.5, .5, 1500)] @ Rz.T * true_s + true_t
    O = np.concatenate([front, side])
    cam = (O - c2w[:3, 3]) @ c2w[:3, :3]
    uv = cam[:, :2] / cam[:, 2:] * 300 + [252, 140]
    mask = np.zeros((280, 504), bool)
    mask[np.clip(uv[:, 1].round().astype(int), 0, 279), np.clip(uv[:, 0].round().astype(int), 0, 503)] = True
    orig = (x7m.observed_points, x7m.view_data)
    x7m.observed_points = lambda src, key, frames, cap=60000: O
    x7m.view_data = lambda src, key, frame, scale=1.: (None, mask, k, c2w)
    try:
        T, rec = align_free(Src(), "o0", [0], [0, 0, 1.], {"vertices": V, "faces": F})
    finally:
        x7m.observed_points, x7m.view_data = orig
    s = np.linalg.norm(T[:3, 0])
    assert rec["up_axis"] == "z" and abs(s / true_s - 1) < .15 and np.linalg.norm(T[:3, 3] - true_t) < .15, (rec, T)
    Rr = T[:3, :3] / s
    assert min(abs(((np.degrees(np.arctan2(Rr[1, 0], Rr[0, 0])) - 30 + 90) % 180) - 90), 90) < 20, rec  # yaw mod 180 (box symmetry), coarse: the placement refines it
    # the r4 primitive's triangles: a box record back to 12 triangles, seen flags per vertex
    from fast_report import display_model
    fs = display_model.fits(front, c2w[None, :3, 3])
    rec2 = display_model.record(fs, "box", "t", {"R": np.eye(3), "origin": np.zeros(3)})
    PV, PF, seen = primitive_tris(rec2)
    assert PF.shape == (12, 3) and seen.shape == (36,) and 0 < seen.sum() < 36
    print("r5_bench self-check ok: a pose-free box aligned upright (scale, position, yaw mod 180), primitive triangles with seen flags")


if __name__ == "__main__":
    import sys
    assert sys.argv[1:] == ["--self-check"], __doc__
    self_check()
