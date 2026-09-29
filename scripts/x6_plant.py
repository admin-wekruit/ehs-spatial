"""Planted changes for X6's recall: a real clip with three rendered cardboard boxes on its floor, one that disappears, one
that appears and one that moves 1.2 m, each at a known time. The walk-throughs themselves hold no change the camera
sees twice (runs/fx-x6-windows-time-003), so without a plant the change rule's recall is unmeasured.

The boxes are drawn with the delivered report's cameras (DROID poses and lens, metric by its own floor-plane scale):
0.8 m cubes standing on the delivered floor plane, flat-shaded cardboard faces (painter's order), placed where DROID's
depth saw free space behind every corner in view in every keyframe that sees >= 5 of its 8 corners and centre (>= 1 such), >= 0.7 m beside the camera path. People or
things passing in front of a box are not drawn over it (the places are picked in open floor). A plant, not a finding:
every number measured on these clips is labelled 'planted'.

    python scripts/x6_plant.py SITE OUT.mp4 TRUTH.json
    python scripts/x6_plant.py --self-check
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "scripts"), str(REPO / "modal_apps"), str(REPO)]

PHASE2 = Path("/Users/adam/Desktop/panoptes-public/research-notes/phase2")
CLIPS = {"me340": "me340-165", "samsclub-a2": "samsclub-337", "walmart": "walmart-190"}
SCALE = {"me340": "da3-posed-me340-223-shotc", "samsclub-a2": "da3-posed-samsclub-a2-281", "walmart": "da3-posed-walmart-251-shot383"}
SIDE, MOVE, CLEAR = .8, 1.2, .7  # metres: cube side, the move, distance kept from the camera path
FACES = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]  # bottom, top, sides


def cube(centre_floor, up, side_dir, size):
    """8 corners: bottom 0-3, top 4-7, the bottom on the floor at centre_floor."""
    a = side_dir / np.linalg.norm(side_dir)
    b = np.cross(up, a)
    h = size / 2
    base = [centre_floor + sx * h * a + sy * h * b for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    return np.array(base + [p + size * up for p in base])


def project(pts, c2w, K):
    c = (pts - c2w[:3, 3]) @ c2w[:3, :3]
    return np.stack([K[0] * c[:, 0] / c[:, 2] + K[2], K[1] * c[:, 1] / c[:, 2] + K[3]], 1), c[:, 2]


def to_source(xy):
    """640x480 raster (the 4:3 centre crop of 1280x720, x1.5) -> source pixels."""
    return np.stack([xy[:, 0] * 1.5 + 160, xy[:, 1] * 1.5], 1)


def draw(img, corners, c2w, K, seed):
    """Visible faces, far to near, cardboard shades with tape and a label."""
    xy, z = project(corners, c2w, K)
    if (z <= .2).any():
        return False
    src = to_source(xy)
    centre, eye = corners.mean(0), c2w[:3, 3]
    order = sorted(range(6), key=lambda f: -np.linalg.norm(corners[list(FACES[f])].mean(0) - eye))
    rng = np.random.default_rng(seed)
    base = np.array([60, 120, 175], float)  # BGR cardboard
    drawn = False
    for f in order:
        idx = list(FACES[f])
        fc = corners[idx].mean(0)
        if np.dot(fc - centre, eye - fc) <= 0:  # facing away
            continue
        shade = [1.0, 1.15, .8, .9, .75, .85][f]
        poly = src[idx].astype(np.int32)
        mask = np.zeros(img.shape[:2], np.uint8)
        cv2.fillConvexPoly(mask, poly, 1)
        if not mask.any():
            continue
        col = np.clip(base * shade + rng.normal(0, 6, 3), 0, 255)
        tex = np.clip(col + rng.normal(0, 7, (mask.sum(), 3)), 0, 255)
        img[mask > 0] = tex.astype(np.uint8)
        q = src[idx]
        cv2.line(img, tuple(((q[0] + q[1]) / 2).astype(int)), tuple(((q[2] + q[3]) / 2).astype(int)), (70, 150, 200), max(2, int(np.ptp(q[:, 0]) / 25)))
        cv2.polylines(img, [poly], True, (35, 70, 105), 2)
        drawn = True
    return drawn


def load(site):
    import fast_report_eval as fe
    ref = fe.reference(site)
    scale = json.loads((PHASE2 / "runs" / SCALE[site] / "metric-scale.json").read_text())
    assert abs(scale["metres_per_native_unit"] - ref["mpn"]) < 1e-9, "the reference's own scale file"
    d = np.load(ref["droid"])
    clip = json.loads((PHASE2 / "data/clips" / CLIPS[site] / "clip.json").read_text())
    up = -np.asarray(scale["up_native"], float)
    up = -up if up @ np.array([0, -1, 0]) < 0 else up  # points away from the floor, towards the camera's -y
    return ref, d, clip, up / np.linalg.norm(up), np.asarray(scale["plane_point_native"], float)


def free_everywhere(corners, d, frames):
    """DROID keyframe depth sees past every corner and the centre (with 10 % slack) in every keyframe that sees them."""
    keys, depth, K = d["keyframe_source_indices"], d["keyframe_final_fullres_depth"], d["keyframe_final_fullres_intrinsics"]
    pts = np.vstack([corners, corners.mean(0)])
    seen = 0
    for j, f in enumerate(keys):
        if f not in frames:
            continue
        c2w = d["poses_c2w"][f].astype(np.float64)
        k = K[j] if K.ndim == 2 else K
        xy, z = project(pts, c2w, np.array([k[0], k[1], k[2], k[3]]) if len(k) == 4 else np.array([k[0, 0], k[1, 1], k[0, 2], k[1, 2]]))
        h, w = depth[j].shape
        ok = (z > .1) & (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
        if ok.sum() < 5:
            continue
        seen += 1
        u, v = xy[ok, 0].astype(int), xy[ok, 1].astype(int)
        obs = np.where(d["keyframe_final_fullres_valid"][j][v, u], depth[j][v, u], 0)
        if ((obs > 0) & (obs < z[ok] * 1.1)).any():
            return False, seen
    return seen >= 1, seen


def visible(corners, poses, frames, Kr, mpn):
    """Frames where the cube is 1.2-7 m from the camera (run 005's 0.6 m boxes, seen at 5-11 m, were never lifted; a
    floor object is below the view of a level camera 1.6 m up closer than ~3.5 m) with >= 6 of its 8 corners inside the
    raster and its centre above the bottom 15 % (the camera operator's own cart fills the bottom of Sam's Club's
    frames). >= 40 frames wanted: a window before the change and one after."""
    out = []
    for f in frames:
        xy, z = project(corners, poses[f], Kr)
        inside = (xy[:, 0] > 0) & (xy[:, 0] < 640) & (xy[:, 1] > 0) & (xy[:, 1] < 480)
        if (z * mpn > 1.2).all() and (z * mpn < 7.).all() and inside.sum() >= 6 and xy[:, 1].mean() < .85 * 480:
            out.append(f)
    return out


def ok_views(cs, up, fwd, size, poses, frames, Kr, mpn):
    """>= 40 close views of the (first) place; a moved box's new place has >= 20 close views after the change."""
    v = visible(cube(cs[0], up, fwd, size), poses, frames, Kr, mpn)
    if len(v) < 40:
        return False
    return len(cs) < 2 or sum(f >= v[len(v) // 2] for f in visible(cube(cs[1], up, fwd, size), poses, frames, Kr, mpn)) >= 20


def plan(site):
    ref, d, clip, up, p0 = load(site)
    mpn, poses, frames = ref["mpn"], d["poses_c2w"].astype(np.float64), ref["segment"]
    Kr = 2 * d["keyframe_final_fullres_intrinsics"][0].astype(float)  # DROID's own lens (its poses were solved with it), at 640 x 480
    size, move, clear = SIDE / mpn, MOVE / mpn, CLEAR / mpn
    path = np.array([poses[f][:3, 3] for f in frames])
    on_floor = lambda p: p - np.multiply.outer((p - p0) @ up, up)  # noqa: E731
    events = []
    used = []
    for kind, share in (("disappeared", .25), ("appeared", .5), ("moved", .72)):
        best = None
        for sh in sorted(np.arange(.05, .96, .05), key=lambda x: abs(x - share)):
            f0 = frames[int(np.clip(sh, 0, .95) * len(frames))]
            c2w = poses[f0]
            fwd = on_floor(c2w[:3, 3] + c2w[:3, 2]) - on_floor(c2w[:3, 3])
            fwd /= np.linalg.norm(fwd)
            side = np.cross(up, fwd)
            for dist in (3., 3.5, 2.5, 4., 4.5, 5., 2., 6.):
                for lat in (0., .6, -.6, .9, -.9, 1.2, -1.2, 1.5, -1.5):
                    c = on_floor(c2w[:3, 3]) + (dist * fwd + lat * side) / mpn
                    cs = [c] + ([c + move * side * (1 if lat <= 0 else -1)] if kind == "moved" else [])
                    if any(np.min(np.linalg.norm(on_floor(path) - x, axis=1)) < clear for x in cs) or \
                            any(np.linalg.norm(x - u) < 2 * size for x in cs for u in used):
                        continue
                    oks = [free_everywhere(cube(x, up, fwd, size), d, set(frames)) for x in cs]
                    if all(o for o, _ in oks) and ok_views(cs, up, fwd, size, poses, frames, Kr, mpn):
                        best = (cs, dist, lat, [n for _, n in oks])
                        break
                if best:
                    break
            if best:
                break
        if best is None:
            events.append({"kind": kind, "placed": False})
            continue
        used += best[0]
        cs, dist, lat, seen = best
        vis = visible(cube(cs[0], up, fwd, size), poses, frames, Kr, mpn)  # the change: mid-way through the close views of its place
        t_frame = vis[len(vis) // 2] if vis else f0
        events.append({"kind": kind, "placed": True, "centre_native": [x.tolist() for x in cs], "centre_m": [(x * mpn).tolist() for x in cs],
                       "change_frame": int(t_frame), "visible_frames": [int(vis[0]), int(vis[-1])] if vis else None, "droid_keyframes_judged": seen,
                       "distance_m": dist, "lateral_m": lat, "side_dir": fwd.tolist()})
    return events, (ref, d, clip, up, p0)


def render(site, out_mp4, truth_json):
    events, (ref, d, clip, up, p0) = plan(site)
    mpn, poses, frames = ref["mpn"], d["poses_c2w"].astype(np.float64), set(ref["segment"])
    Kr = 2 * d["keyframe_final_fullres_intrinsics"][0].astype(float)
    src = PHASE2 / "data/clips" / CLIPS[site] / "source-full.mp4"
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"avc1"), fps, (w, h))
    assert writer.isOpened()
    f, drawn = 0, {i: 0 for i in range(len(events))}
    while True:
        ok, img = cap.read()
        if not ok:
            break
        if f in frames:
            for i, e in enumerate(events):
                if not e["placed"]:
                    continue
                before = f < e["change_frame"]
                if e["kind"] == "disappeared" and not before or e["kind"] == "appeared" and before:
                    continue
                c = np.array(e["centre_native"][0 if (before or e["kind"] != "moved") else 1])
                drawn[i] += draw(img, cube(c, up, np.array(e["side_dir"]), SIDE / mpn), poses[f], Kr, seed=i)
        writer.write(img)
        f += 1
    writer.release()
    for i, e in enumerate(events):
        e.update(frames_drawn=drawn[i], change_s=round(e["change_frame"] / fps, 3) if e["placed"] else None)
    truth = {"site": site, "source": str(src.relative_to(PHASE2)), "fps": fps, "frames": f, "side_m": SIDE, "move_m": MOVE, "events": events,
             "frame": "DROID native (the delivered camera); *_m = native x metres_per_native_unit (estimated, assumed 1.6 m camera height)",
             "label": "planted: rendered boxes, not a real change"}
    Path(truth_json).write_text(json.dumps(truth, indent=1))
    return truth


def self_check():
    up, fwd = np.array([0, -1., 0]), np.array([0, 0, 1.])
    c = cube(np.zeros(3), up, fwd, 2.)
    assert np.allclose(c[:4, 1], 0) and np.allclose(c[4:, 1], -2) and np.allclose(c.mean(0), [0, -1, 0])
    img = np.zeros((720, 1280, 3), np.uint8)
    c2w = np.eye(4)
    c2w[:3, 3] = [0, -1, -5]
    assert draw(img, c, c2w, np.array([500., 500, 320, 240]), 0) and img[360, 640].any(), "the cube is drawn in front of the camera"
    assert not draw(np.zeros_like(img), c + np.array([0, 0, -10.]), c2w, np.array([500., 500, 320, 240]), 0), "behind the camera: nothing"
    xy = to_source(np.array([[0., 0], [640, 480]]))
    assert np.allclose(xy, [[160, 0], [1120, 720]])
    print("plant self-check ok: cube on the floor, drawn in view, nothing behind the camera, raster -> source")


if __name__ == "__main__":
    if sys.argv[1:] == ["--self-check"]:
        self_check()
    else:
        t = render(*sys.argv[1:4])
        print(json.dumps(t["events"], indent=1))
