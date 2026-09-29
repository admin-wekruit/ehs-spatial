"""Viewer fixture for the click MVP (builder C, CLICK-MVP-SPEC section 9 C): a recorded fb/integrate report plus `pick`,
`object_cards` and `judgements` layers in the spec's formats (sections 3.3, 4.9, 5.6), written as a new recording that
`python -m fast_report.layers serve ROOT --replay OUT REPORT` plays at its (simulated) times. A's and B's real layers replace it.

What is real and what is not (the layers say so in data.fixture and in their labels):
  pick     the recording's outlines rasterised at 640x360, smaller mask on top (spec 3.2). People have no masks in the
           recording: each track point is a 0.5 x 1.75 m board standing at the track's floor point, painted last. [fixture]
  depth    the full room mesh's vertices projected into each keyframe camera, nearest per 4x4 DA3 block (not DA3 depth).
  cards    floor-frame values from the recording's (inflated, L1) boxes; views, times, distances and azimuth spread from the
           outlines and cameras; the class size check and the identity cross-check are the spec's tables. The view-subset
           values are SYNTHETIC (the recording keeps no per-view points), so every u here is a UI test value.
  judge    J1 / J4 / J7 / J8 by the spec's banded rule on those values; VLM probabilities are SYNTHETIC; evidence images are
           set-of-marks crops of the real frames.
  times    new layers at simulated times [E] after the recorded objects/outlines layers.

  python scripts/mvp_viewer_fixture.py SRC_RUN REPORT OUT_ROOT [--as NAME]
  python scripts/mvp_viewer_fixture.py --self-check
"""
import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fast_report.layers import put_blob, _write_json  # noqa: E402

PICK_W, PICK_H = 640, 360
DEPTH_W, DEPTH_H = 126, 70
NOTE = ("fixture: pick, cards and judgements are viewer test data built from a recorded fast report; view-subset values and "
        "VLM probabilities are synthetic; not measurements")
SCALE = "estimated (floor plane + assumed 1.6 m camera height)"  # A's cards.SCALE; B's rows say "estimated"

# Spec 4.2 seed table: (names, longest side min, max in m, placement). Multi-word names first so 'stacked boxes' beats 'box'.
CLASS_SIZE = [
    (("stacked boxes", "pallet of goods", "loaded pallet"), .3, 3.5, "on_floor"),
    (("shopping cart", "pallet jack", "cart", "trolley"), .5, 2.2, "on_floor"),
    (("exit sign",), .1, .6, None), (("fire extinguisher",), .3, 1.0, None),
    (("display screen", "monitor", "tv"), .2, 1.6, None), (("ceiling light", "light fixture"), .1, 2.5, "high"),
    (("drill bit", "marker", "eraser", "wrench", "screwdriver", "hammer", "bolt", "knob", "switch", "battery", "bottle", "can",
      "cup", "glove", "tape"), .01, .6, None),
    (("sign", "label"), .05, 2.0, None), (("box", "carton", "package", "crate", "tote", "bin", "bag"), .05, 1.5, None),
    (("pallet",), .8, 1.4, "on_floor"), (("chair",), .3, 1.3, None), (("stool",), .3, .8, None),
    (("table", "desk", "workbench"), .5, 3.5, None), (("cabinet", "locker"), .3, 2.5, None), (("door",), .6, 3.0, None),
    (("ladder",), .5, 6, None), (("cnc machine", "machine", "lathe", "mill", "forklift"), .5, 6, "on_floor"),
    (("display rack", "shelving", "shelf", "rack", "conveyor", "duct", "pipe"), .3, 30, None),
    (("cable", "hose", "cord", "wire"), .1, 30, None), (("spill",), .05, 5, "flat_floor"),
]
MOBILITY = {"fixed": {"wall", "shelf", "rack", "machine", "door", "column", "conveyor", "cabinet", "sign", "display rack", "shelving"},
            "movable rigid": {"box", "pallet", "cart", "ladder", "chair", "bin", "tool", "crate", "carton", "stacked boxes", "material cart"},
            "deformable": {"cable", "hose", "cord", "wire", "rope", "chain", "strap", "curtain", "wrap"},
            "agent": {"person", "forklift", "pallet jack", "agv", "robot arm"}}
CATEGORY = {"F payload": {"box", "carton", "pallet", "crate", "stacked boxes", "cart", "bag", "tote", "bin", "pallet of goods"},
            "E information": {"sign", "exit sign", "label"}, "C guards": {"guard", "fence"},
            "B control": {"control panel", "switch", "button", "emergency stop"}, "A sensing": {"light curtain", "sensor"}}


def heads(name):
    """The name, then its head noun and simple singulars (spec 4.2's head-noun match)."""
    name = name.lower().strip()
    last = name.split()[-1] if name else name
    out = [name, last, last[:-1] if last.endswith("s") else last, last[:-2] if last.endswith("es") else last]
    return out + ([last[:-3] + "f"] if last.endswith("ves") else [])


def lookup(name, table):
    for h in heads(name):
        for key, value in table:
            if h in key:
                return key, value
    return None, None


def size_class(name):
    for h in heads(name):
        for names, lo, hi, place in CLASS_SIZE:
            if h in names:
                return h, lo, hi, place
    return None, 0.0, 6.0, None


def kind_of(name):
    mob = next((m for h in heads(name) for m, words in MOBILITY.items() if h in words), None)
    cat = next((c for h in heads(name) for c, words in CATEGORY.items() if h in words), "other")
    return {"category": cat, "mobility": mob or "not classified", "mobility_source": "class prior" if mob else "no class prior"}


def rle(labels):
    """uint16 (value, run) pairs in raster order; runs over 65535 are split (spec 3.3)."""
    flat = np.asarray(labels, np.uint16).ravel()
    starts = np.r_[0, np.flatnonzero(np.diff(flat)) + 1]
    lengths = np.diff(np.r_[starts, flat.size])
    reps = (lengths + 65534) // 65535
    idx = np.repeat(np.arange(len(starts)), reps)
    first = np.cumsum(reps) - reps
    piece = np.arange(idx.size) - first[idx]
    runs = np.where(piece < reps[idx] - 1, 65535, lengths[idx] - 65535 * (reps[idx] - 1))
    return np.stack([flat[starts][idx], runs], 1).astype("<u2")


def unrle(pairs, w, h):
    return np.repeat(pairs[:, 0], pairs[:, 1].astype(np.int64)).reshape(h, w)


def seeded(*key):
    return np.random.default_rng(int(hashlib.sha256("|".join(map(str, key)).encode()).hexdigest()[:12], 16))


def r(v, d=3):
    return None if v is None else round(float(v), d)


def quantity(v, parts, unit="m", subsets=(), bound=None, scale=True):
    """{value, u, unit, level, scale, n_subsets, parts}; u = sqrt(sum parts^2) with k = 1 (spec 4.4)."""
    parts = {k: float(p) for k, p in parts.items() if p is not None}
    out = {"value": r(v), "u": r(math.sqrt(sum(p * p for p in parts.values()))), "unit": unit, "level": "coarse",
           "scale": SCALE if scale else "scale-free (angle)", "n_subsets": max(len(subsets), 1), "subsets": [r(s) for s in subsets],
           "parts": {k: r(p) for k, p in parts.items()}}
    if len(subsets) < 2:
        out["note"] = "one view set: uncertainty from model terms only (likely understated)"
    if bound:
        out["status"] = bound  # A: "at least" / "at most" / "needs review" carry a value
    return out


def subsets_for(value, n_views, *key):
    """SYNTHETIC block values (the recording has no per-view points): S = 2 for 4-8 views, 3 for 9+ (spec 4.4)."""
    s = 0 if n_views < 4 else 2 if n_views <= 8 else 3
    if not s:
        return []
    return list(value * (1 + seeded(*key).uniform(-.08, .08, s)))


def views_part(sub):
    return (max(sub) - min(sub)) if len(sub) >= 2 else None


def floor_frame(shot):
    c2w0 = np.asarray(shot["c2w"][0], float)
    n, p, c = np.asarray(shot["floor"]["normal"], float), np.asarray(shot["floor"]["point_m"], float), c2w0[:3, 3]
    n = n if (c - p) @ n > 0 else -n
    origin = c - ((c - p) @ n) * n
    f = c2w0[:3, 2] - (c2w0[:3, 2] @ n) * n
    x = f / np.linalg.norm(f)
    return {"origin": origin, "x": x, "y": np.cross(n, x), "z": n}


def to_floor(F, P):
    P = np.asarray(P, float) - F["origin"]
    return np.stack([P @ F["x"], P @ F["y"], P @ F["z"]], -1)


def project(K, c2w, P):
    """World points -> (u, v, z) on the 504x280 DA3 grid."""
    w2c = np.linalg.inv(np.asarray(c2w, float))
    X = np.asarray(P, float) @ w2c[:3, :3].T + w2c[:3, 3]
    K = np.asarray(K, float)
    z = X[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        return K[0, 0] * X[:, 0] / z + K[0, 2], K[1, 1] * X[:, 1] / z + K[1, 2], z


def read_mesh_vertices(path, meta):
    lay = meta["byteLayout"]
    return np.frombuffer(Path(path).read_bytes(), "<f4", lay["vertexCount"] * lay["stride"]).reshape(-1, lay["stride"])[:, :3]


def poly_to_rect_dist(pt, rect):
    import cv2
    return max(0.0, -cv2.pointPolygonTest(rect.astype(np.float32), (float(pt[0]), float(pt[1])), True))


def band(v, u, t, kind):
    """video.banded_verdict with band = u: max-type FAIL if v-u > T, PASS if v+u < T; min-type mirrored."""
    if kind == "max":
        return "FAIL" if v - u > t else "PASS" if v + u < t else "NEEDS_REVIEW"
    return "FAIL" if v + u < t else "PASS" if v - u > t else "NEEDS_REVIEW"


ORDER = {"FAIL": 0, "NEEDS_REVIEW": 1, "NO_DATA": 2, "PASS": 3}


def worst(verdicts):
    """Severity FAIL > NEEDS_REVIEW > NO_DATA > PASS (the worst_verdict fix, spec 5.2)."""
    return min(verdicts, key=ORDER.__getitem__) if verdicts else None


class Recording:
    def __init__(self, src, report):
        self.src, self.report = Path(src), report
        folder = self.src / "reports" / report
        self.patches = [json.loads(p.read_text()) for p in sorted((folder / "patches").glob("*.json"))]
        self.written = json.loads((folder / "written.json").read_text()) if (folder / "written.json").exists() else {}
        self.run = json.loads((folder / "run.json").read_text()) if (folder / "run.json").exists() else None
        self.first, self.latest = {}, {}
        for p in self.patches:
            self.first.setdefault(p["layer"], p)
            self.latest[p["layer"]] = p

    def blob(self, ref):
        return self.src / "blobs" / "sha256" / ref["sha256"]

    def json_of(self, patch, key, role):
        return patch["data"][key] if isinstance(patch["data"].get(key), (dict, list)) else json.loads(self.blob(patch["blobs"][role]).read_text())


def build(src, report, out, as_report=None, mp4=None):
    import cv2
    rec, out = Recording(src, report), Path(out)
    as_report = as_report or report + "-mvp-fixture"
    cams = rec.first["cameras"]["data"]
    shots = {s["index"]: s for s in cams["shots"]}
    frames_of = {s["index"]: {k: i for i, k in enumerate(s["keys"])} for s in cams["shots"]}
    F = {i: floor_frame(s) for i, s in shots.items()}
    objects = rec.first["objects"]["data"]["objects"]
    analysis = rec.json_of(rec.first["outlines"], "analysis", "analysis")
    people = rec.first["people"]["data"]
    room = rec.latest["room"]
    ate = ((rec.run or {}).get("quality", {}).get("rows", {}).get("cameras", {}) or {}).get("ate_m")
    u_pose = max(.04, ate or .04)
    clip_end = analysis["frames"][-1]["endTimeSec"]

    # ---- frames: which shot/key each outline frame is, and each entity's polygons there
    def shot_key(source_frame):
        for i, keys in frames_of.items():
            if source_frame in keys:
                return i, keys[source_frame]
        return None, None

    entity_ids = [None] + [o["id"] for o in objects] + ["person:" + t["id"] for t in people["tracks"]]
    index = {e: i for i, e in enumerate(entity_ids) if e}
    track_at = {}
    for t in people["tracks"]:
        for p in t["points"]:
            track_at.setdefault(p["frame"], []).append((t, p))

    pick_frames, runs, depths, per_obj = [], [], [], {}
    verts = {i: read_mesh_vertices(rec.blob(room["blobs"][f"mesh-{i}"]), room["blobs"][f"mesh-{i}"]) for i in shots if f"mesh-{i}" in room["blobs"]}
    sx, sy = PICK_W / analysis["width"], PICK_H / analysis["height"]
    for fi, fr in enumerate(analysis["frames"]):
        shot, key = shot_key(fr["sourceFrame"])
        canvas = np.zeros((PICK_H, PICK_W), np.uint16)
        painted = []
        for ob in fr["objects"]:
            if not ob.get("entityId") or ob["entityId"] not in index:
                continue
            polys = [np.round(np.asarray(p, float) * [sx, sy]).astype(np.int32) for p in ob["polygons"] if len(p) >= 3]
            area = sum(abs(cv2.contourArea(p.astype(np.float32))) for p in polys)
            painted.append((area, ob["entityId"], polys))
            full = [np.asarray(p, float) for p in ob["polygons"] if len(p) >= 3]
            if full:
                allp = np.vstack(full)
                per_obj.setdefault(ob["entityId"], []).append({"fi": fi, "t": fr["timeSec"], "t_end": fr["endTimeSec"], "frame": fr["sourceFrame"],
                    "shot": shot, "key": key, "source": ob.get("source", "segmented"), "area": area / (sx * sy),
                    "bbox": [allp[:, 0].min(), allp[:, 1].min(), allp[:, 0].max(), allp[:, 1].max()]})
        for area, eid, polys in sorted(painted, key=lambda x: -x[0]):  # larger first: the smaller mask wins
            cv2.fillPoly(canvas, polys, int(index[eid]))
        if shot is not None:  # people last: they beat objects (spec 3.2)
            K, c2w, n = shots[shot]["K"][key], shots[shot]["c2w"][key], F[shot]["z"]
            c = np.asarray(c2w, float)[:3, 3]
            for t, p in track_at.get(fr["sourceFrame"], []):
                if t["shot"] != shot:
                    continue
                g = np.asarray(p["xyz"], float)
                d = g - c
                d -= (d @ n) * n
                side = np.cross(n, d / max(np.linalg.norm(d), 1e-6)) * .25
                corners = np.array([g - side, g + side, g + side + 1.75 * n, g - side + 1.75 * n])
                u, v, z = project(K, c2w, corners)
                if (z > .2).all():
                    poly = np.round(np.stack([u * PICK_W / 504, v * PICK_H / 280], 1)).astype(np.int32)
                    cv2.fillPoly(canvas, [poly], int(index["person:" + t["id"]]))
        pairs = rle(canvas)
        pick_frames.append({"t": fr["timeSec"], "t_end": fr["endTimeSec"], "frame": fr["sourceFrame"], "shot": shot, "key": key,
                            "w": PICK_W, "h": PICK_H, "source": fr.get("source", "segmented"), "offset": sum(len(x) for x in runs), "pairs": len(pairs)})
        runs.append(pairs)
        depth = np.zeros(DEPTH_W * DEPTH_H, np.float64) + np.inf
        if shot is not None and shot in verts:
            u, v, z = project(shots[shot]["K"][key], shots[shot]["c2w"][key], verts[shot])
            ok = (z > .1) & (u >= 0) & (u < 504) & (v >= 0) & (v < 280)
            cell = (v[ok] // 4).astype(np.int64) * DEPTH_W + (u[ok] // 4).astype(np.int64)
            np.minimum.at(depth, cell, z[ok])
        depths.append(np.where(np.isfinite(depth), np.clip(np.round(depth * 1000), 1, 65535), 0).astype("<u2"))
    pick_blob = gzip.compress(np.concatenate(runs).tobytes(), 6)
    depth_blob = gzip.compress(np.concatenate(depths).tobytes(), 6)

    # ---- cards
    walked = {}
    for i, s in shots.items():
        walked[(i, "camera")] = to_floor(F[i], [np.asarray(c, float)[:3, 3] for c in s["c2w"]])[:, :2]
    for t in people["tracks"]:
        if len(t["points"]) >= 2:
            walked[(t["shot"], "person:" + t["id"])] = to_floor(F[t["shot"]], [p["xyz"] for p in t["points"]])[:, :2]

    def cam_centre(shot, key):
        return np.asarray(shots[shot]["c2w"][key], float)[:3, 3]

    cards, cards_v2, rects = [], [], {}
    for o in objects:
        s, oid, name = o["shot"], o["id"], o["word"]
        Fs = F[s]
        lo, hi = np.asarray(o["box_min_m"], float), np.asarray(o["box_max_m"], float)
        corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        fc = to_floor(Fs, corners)
        base, top = fc[:, 2].min(), fc[:, 2].max()
        (cx, cy), (rw, rh), ang = cv2.minAreaRect(fc[:, :2].astype(np.float32))
        rect = rects[oid] = cv2.boxPoints(((cx, cy), (rw, rh), ang))
        centre = to_floor(Fs, [o["centroid_m"]])[0]
        seen = [v for v in per_obj.get(oid, []) if v["shot"] == s and v["area"] >= 100]
        det = [v for v in seen if v["source"] == "segmented"]
        cams_xy = np.array([to_floor(Fs, [cam_centre(s, v["key"])])[0] for v in det]) if det else np.zeros((0, 3))
        dist = np.linalg.norm(cams_xy - centre, axis=1) if len(det) else np.array([])
        z_med = float(np.median(dist)) if len(dist) else float(np.linalg.norm(to_floor(Fs, [cam_centre(s, 0)])[0] - centre))
        az = np.sort(np.degrees(np.arctan2(cams_xy[:, 1] - centre[1], cams_xy[:, 0] - centre[0]))) if len(det) else np.array([])
        spread = float(360 - max(np.diff(np.r_[az, az[0] + 360]))) if len(az) > 1 else 0.0
        # width = the rectangle side more perpendicular to the mean viewing direction
        e1 = rect[1] - rect[0]
        view = (centre[:2] - cams_xy[:, :2].mean(0)) if len(det) else np.array([1.0, 0])
        view = view / max(np.linalg.norm(view), 1e-6)
        s1, s2 = float(np.linalg.norm(e1)), float(np.linalg.norm(rect[2] - rect[1]))
        par1 = abs(e1 @ view) / max(s1, 1e-6)
        width, depth = (s1, s2) if par1 < .7071 else (s2, s1)
        n = len(det)
        cls, cmin, cmax, place = size_class(name)
        longest = max(top - base, width, depth)
        span = max(width, depth) if cls == "pallet" else longest
        reasons = []
        if span > cmax:
            reasons.append(f"implausible for a {cls or name} ({span:.2f} m > {cmax:g} m): needs review")
        if span < cmin:
            reasons.append(f"implausible for a {cls or name} ({span:.2f} m < {cmin:g} m): needs review")
        if place in ("on_floor", "flat_floor") and base > .3:
            reasons.append(f"a {cls} should stand on the floor; base {base:.2f} m above it: needs review")
        if place == "high" and base < 1.8:
            reasons.append(f"a {cls} should hang at 1.8 m or higher; base {base:.2f} m: needs review")
        size_check = {"status": "implausible" if reasons else "plausible", "class": cls or "any other word", "class_range_m": [cmin, cmax],
                      "longest_m": r(span, 2), **({"reason": "; ".join(reasons)} if reasons else {})}
        cam_h = float(np.median(cams_xy[:, 2])) if len(det) else 1.6
        res = 2 * 2 * z_med / float(np.asarray(shots[s]["K"][0])[0][0])
        u_floor = .02  # fixture: the recording keeps no floor residual

        def q(field, v, depth_part, extra=(), bound=None):
            sub = subsets_for(v, n, oid, field)
            parts = {"views": views_part(sub), "depth": depth_part, **dict(extra), "scale": .2 * abs(v)}
            return quantity(float(np.median(sub)) if sub else v, parts, subsets=sub, bound=bound)

        def cut(axis):
            lo_i, hi_i = (1, 3) if axis == "y" else (0, 2)
            limit = analysis["height"] if axis == "y" else analysis["width"]
            return all(v["bbox"][lo_i] <= 2 or v["bbox"][hi_i] >= limit - 2 for v in det) if det else True

        phys = {"position_xy": {**quantity(0, {"pose": u_pose, "depth": .05 * z_med, "scale": .2 * z_med}), "value": [r(cx), r(cy)]},
                "base_above_floor": q("base", base, .05 * abs(cam_h - base), {"floor": u_floor}),
                "top_above_floor": q("top", top, .05 * abs(cam_h - top), {"floor": u_floor})}
        phys["height"] = q("height", top - base, .05 * (top - base), {"resolution": res}, bound="at least" if cut("y") else None)
        if phys["height"].get("status"):
            phys["height"]["reason"] = "cut by the frame edge in every view"
        phys["width"] = q("width", width, .05 * width, {"resolution": res}, bound="at least" if cut("x") else None)
        if phys["width"].get("status"):
            phys["width"]["reason"] = "cut by the frame edge in every view"
        if spread >= 30:
            phys["depth"] = q("depth", depth, .05 * depth, {"resolution": res})
            phys["footprint_m2"] = quantity(width * depth, {"depth": .1 * width * depth, "scale": .4 * width * depth}, unit="m²")
        else:
            phys["depth"] = {"status": "not observed", "reason": f"seen from one side (azimuth spread {spread:.0f}°)"}
            phys["footprint_m2"] = quantity(width * depth, {"depth": .1 * width * depth, "scale": .4 * width * depth}, unit="m²", bound="at least")
        best = min(((poly_to_rect_dist(p, rect), path) for (sh, path), pts in walked.items() if sh == s for p in pts), default=None)
        if best:
            phys["nearest_walked_path"] = {**quantity(best[0], {"pose": u_pose, "depth": .05 * z_med, "scale": .2 * best[0]}), "path": best[1]}
        phys["walkway"] = {"status": "no marked walkway detected"}
        phys["principal_axis_tilt_deg"] = {"status": "not measurable", "reason": "the fixture has no per-view points"}
        phys["planar_slope_deg"] = {"status": "not measurable", "reason": "the fixture has no per-view points"}
        phys["footprint_xy"] = {"value": [[r(a), r(b)] for a, b in rect], "unit": "m", "frame": f"floor frame of shot {s}", "scale": SCALE}
        phys["level"] = "coarse"
        phys["size_check"] = size_check
        phys["box"] = {"center_m": [r(cx), r(cy), r((base + top) / 2)], "quaternion": [0, 0, r(math.sin(math.radians(ang) / 2), 4), r(math.cos(math.radians(ang) / 2), 4)],
                       "size_m": [r(rw), r(rh), r(top - base)]}
        phys["dropped_share"] = None
        phys["merged_from"] = []
        best_views = [v["frame"] for v in sorted(det, key=lambda v: -v["area"])[:3]]
        intervals = []
        for v in sorted(seen, key=lambda v: v["t"]):
            if intervals and v["t"] - intervals[-1][1] <= .5:
                intervals[-1][1] = v["t_end"]
            else:
                intervals.append([v["t"], v["t_end"]])
        last = seen[-1]["t"] if seen else None
        if n >= 4:
            state = {"state": "static", "evidence": None}
        else:
            state = {"state": f"last seen at {r(last, 2)} s", "last_seen_reason": "out of view" if intervals and intervals[-1][1] < clip_end - .5 else "end of the shot",
                     "evidence": None}
        card = {"id": oid, "kind": "object", "shot": s,
                "identity": {"name": name, "confidence": None, "calibrated": False, "alternatives": [], "decided_by": "sam3 vote",
                             "note": "detected word, unverified", "candidates_struck": [], "label": "inferred"},
                "class": kind_of(name), "physical": phys,
                "views": {"n": n, "keyframes": [v["frame"] for v in det], "best": best_views,
                          "distance_m": [r(dist.min(), 2), r(dist.max(), 2)] if len(dist) else None, "azimuth_spread_deg": r(spread, 1)},
                "time": {"first_seen_s": r(seen[0]["t"], 2) if seen else None, "last_seen_s": r(last, 2), "intervals": [[r(a, 2), r(b, 2)] for a, b in intervals],
                         "detected_keyframes": [v["frame"] for v in det], **state},
                "observed": ["masks", "views", "time"], "estimated": ["physical"], "inferred": ["identity", "class"]}
        cards.append(card)
        # v2 identity: the SAM 3 votes and the cascade label as options, each struck when its size or placement fails (spec 4.7);
        # the probabilities are the normalised votes, uncalibrated (no decider in the fixture).
        cand = [w for w, _ in sorted(o.get("votes", {}).items(), key=lambda x: -x[1])[:3]]
        if (o.get("cascade") or {}).get("label") and o["cascade"]["label"] not in cand:
            cand.append(o["cascade"]["label"])
        struck, keep = [], []
        for w in cand:
            c2, lo2, hi2, pl2 = size_class(w)
            why = (f"{span:.2f} m is not a {w}'s size" if not (lo2 <= longest <= hi2) else
                   "not on the floor" if pl2 in ("on_floor", "flat_floor") and base > .3 else
                   "not high enough" if pl2 == "high" and base < 1.8 else None)
            (struck.append([w, why]) if why else keep.append(w))
        votes = {w: float(o.get("votes", {}).get(w, .2)) for w in keep}
        total = sum(votes.values()) or 1
        opts = sorted(((w, v / total) for w, v in votes.items()), key=lambda x: -x[1])
        ident = ({"name": opts[0][0], "confidence": r(opts[0][1], 2), "calibrated": False, "alternatives": [[w, r(p, 2)] for w, p in opts[1:]],
                  "decided_by": "sam3 vote", "candidates_struck": struck, "label": "inferred"} if opts else
                 {"name": name, "confidence": None, "calibrated": False, "alternatives": [], "decided_by": "sam3 vote",
                  "status": "every candidate failed the size check", "candidates_struck": struck, "label": "inferred"})
        cards_v2.append({**card, "identity": ident, "class": kind_of(ident["name"])})

    # people cards
    rules = people.get("rules", [])
    pcards = []
    for t in people["tracks"]:
        pid = "person:" + t["id"]
        pts = np.asarray([p["xyz"] for p in t["points"]], float)
        fl = to_floor(F[t["shot"]], pts)[:, :2]
        mine = [x for x in rules if x.get("shot") == t["shot"] and t["id"].split("-", 1)[1] in (x.get("tracks") or [])]
        near = sorted(((min(poly_to_rect_dist(p, rects[c["id"]]) for p in fl), c["id"], c["identity"]["name"])
                       for c in cards if c["shot"] == t["shot"] and c["physical"]["size_check"]["status"] == "plausible"), key=lambda x: x[0])[:3]
        length = float(np.linalg.norm(np.diff(fl, axis=0), axis=1).sum()) if len(fl) > 1 else 0.
        pcards.append({"id": pid, "kind": "person", "shot": t["shot"],  # A's people_cards shape
                       "identity": {"name": "person", "decided_by": "sam3 person + tracker", "label": "observed", "track": t["id"]},
                       "class": {"category": "other", "mobility": "agent", "mobility_source": "class prior"},
                       "time": {"first_seen_s": r(t["t0"], 2), "last_seen_s": r(t["t1"], 2), "detections": t["detections"], "state": "agent: position per keyframe"},
                       "physical": {"path_length": quantity(length, {"scale": .2 * length, "depth": .05 * length}), "level": "coarse"},
                       "rules": mine, "nearest_objects": [[i, r(d, 2)] for d, i, _ in near], "ppe": None,
                       "observed": ["masks", "track"], "estimated": ["physical", "positions"], "inferred": []})
        pcards[-1]["physical"]["path_length"].pop("note", None)  # a path length has no view-subset meaning (A: views_term=False)

    # ---- judgements (B's rules on the fixture values; spec 5.1, 5.4)
    frames_needed = {}
    rows = []

    def forced(card, *fields):
        ph = card["physical"]
        why = []
        if ph["size_check"]["status"] == "implausible":
            why.append(ph["size_check"]["reason"])
        if any(ph[f].get("n_subsets", 1) < 2 for f in fields if "value" in ph[f]):
            why.append("one view set: no PASS or FAIL on model terms alone")
        return why

    def evidence(card, check):
        ev = []
        for k, frame in enumerate(card["views"]["best"][:2]):
            role = f"ev-{check}-{card['id']}-{k}"
            t = next(v["t"] for v in per_obj[card["id"]] if v["frame"] == frame)
            ev.append({"key": frame, "t": r(t, 2), **({"image": role} if k == 0 else {})})
            if k == 0:
                frames_needed.setdefault(frame, []).append((role, card["id"]))
        return ev

    for card in cards:
        name, ph = card["identity"]["name"], card["physical"]
        cls = size_class(name)[0]
        mob = card["class"]["mobility"]
        if cls in ("stacked boxes", "pallet of goods", "loaded pallet"):
            top = ph["top_above_floor"]
            g = band(top["value"], top["u"], 2.5, "max")
            why = forced(card, "top_above_floor")
            verdict = "NEEDS_REVIEW" if why and g in ("PASS", "FAIL") else g
            rows.append({"id": f"J1:{card['id']}", "check": "J1", "title": "stack height", "subject": card["id"], "object": None,
                         "verdict": verdict, "severity": "major",
                         "geometry": {"quantity": "top above floor", "value": top["value"], "u": top["u"], "unit": "m", "threshold": 2.5, "direction": "max",
                                      "result": g, "scale": SCALE},
                         "reasons": [f"top {top['value']:.2f} ± {top['u']:.2f} m against 2.5 m (max)"] + why,
                         "evidence": evidence(card, "J1"), "rule_source": "policy MAX_HEIGHT; starter rule 'stacked pallets ≤ 2.5 m'",
                         "interval_s": card["time"]["intervals"][0] if card["time"]["intervals"] else None})
        if mob == "deformable" and "nearest_walked_path" in ph:
            d, base = ph["nearest_walked_path"], ph["base_above_floor"]
            on_floor = base["value"] <= .05 + base["u"]
            g = ("FAIL" if on_floor and d["value"] + d["u"] < .5 else "PASS" if d["value"] - d["u"] > 1.0 else "NEEDS_REVIEW")
            why = forced(card, "nearest_walked_path", "base_above_floor")
            if g == "PASS" and ph["depth"].get("status") == "not observed":
                why.append("footprint seen from one side: the gap may be smaller")
            verdict = "NEEDS_REVIEW" if why and g in ("PASS", "FAIL") else g
            rows.append({"id": f"J4:{card['id']}", "check": "J4", "title": "cable or hose across a walked path", "subject": card["id"], "object": "path:" + d["path"],
                         "verdict": verdict, "severity": "major",
                         "geometry": {"quantity": "distance to walked path", "value": d["value"], "u": d["u"], "unit": "m", "threshold": .5, "direction": "min",
                                      "result": g, "scale": SCALE},
                         "reasons": [f"distance {d['value']:.2f} ± {d['u']:.2f} m to the {d['path']} path; FAIL below 0.5 m, PASS beyond 1.0 m",
                                     f"base {base['value']:.2f} ± {base['u']:.2f} m: {'on' if on_floor else 'off'} the floor"] + why,
                         "evidence": evidence(card, "J4"), "rule_source": "MVP; OSHA 1910.176(a) (aisles kept clear)",
                         "interval_s": card["time"]["intervals"][0] if card["time"]["intervals"] else None})
        if cls == "ladder":
            rows.append({"id": f"J7:{card['id']}", "check": "J7", "title": "ladder lean", "subject": card["id"], "object": None, "verdict": "NO_DATA",
                         "severity": "major", "geometry": {"quantity": "principal axis tilt", "value": None, "u": None, "unit": "deg", "threshold": 10,
                                                            "direction": "max", "result": "NO_DATA", "scale": "scale-free"},
                         "reasons": ["tilt not measurable from this video: " + ph["principal_axis_tilt_deg"]["reason"]],
                         "evidence": evidence(card, "J7"), "rule_source": "policy MAX_TILT", "interval_s": None})
    for pc in pcards:  # B's person_rule_rows: J8 per rule, aggregated over the track's findings
        by_rule = {}
        for x in pc["rules"]:
            by_rule.setdefault(x["rule"], []).append(x)
        for rule, rs in sorted(by_rule.items()):
            v = worst([x["verdict"] for x in rs])
            reasons = sorted({f"{x['verdict']}: {x.get('reason') or 'scale not measured (scale_gated)'}" for x in rs})
            rows.append({"id": f"J8-{rule}:{pc['id']}", "check": "J8", "title": f"person rule {rule}", "subject": pc["id"], "object": None,
                         "verdict": v, "severity": "major", "geometry": {"quantity": rule, "value": None, "result": v, "scale": "estimated", "reasons": reasons},
                         "vlm": None, "reasons": reasons, "evidence": [{"key": x["frame"], "t": r(x["t"], 2)} for x in rs[:3]],
                         "rule_source": "video.py R1-R3, unchanged (scale gated)", "interval_s": [rs[0]["t"], rs[-1]["t"]]})

    def layer(rs):
        by = {}
        for x in rs:
            by.setdefault(x["subject"], []).append(x["verdict"])
        counts = {k: sum(x["verdict"] == k for x in rs) for k in ORDER}
        return {"schema": "panoptes-judgements-v1", "checks_version": "mvp-1", "calibration": {"file_sha256": None}, "fixture": NOTE,
                "rows": rs, "by_object": {k: worst(v) for k, v in by.items()}, "counts": counts}

    # v2: synthetic VLM answers on J1/J4 (uncalibrated, so 'unsure' and never deciding: spec 5.4)
    rows_v2 = []
    for x in rows:
        if x["check"] in ("J1", "J4"):
            rng = seeded(x["id"], "vlm")
            per = []
            for ev in x["evidence"][:2]:
                p = rng.dirichlet([2, 2, .6])
                per.append({"keys": [ev["key"]], "probs": [r(v) for v in p], "mass": r(rng.uniform(.6, .99), 2), "p_hazard_raw": r(p[0], 4), "calibrated": None})
            x = {**x, "vlm": {"question": "q2" if x["check"] == "J1" else "q1", "text": "(fixture question)", "options": ["yes", "no", "cannot tell"],
                              "decider": "fixture (synthetic probabilities)", "per_view": per, "answer": "unsure", "calibration": "none", "also": []},
                 "reasons": x["reasons"] + ["picture: uncalibrated probabilities never decide"]}
        rows_v2.append(x)

    # ---- evidence images: set-of-marks crops of the real frames (white over black, never red; subject [1])
    mp4 = Path(mp4) if mp4 else next(Path(src).glob("input-*.mp4"))
    images = {}
    cap, i, want = cv2.VideoCapture(str(mp4)), 0, sorted(frames_needed)
    by_frame = {fr["sourceFrame"]: fr for fr in analysis["frames"]}
    for target in want:
        while i <= target:
            ok, img = cap.read()
            i += 1
            if not ok:
                break
        if not ok:
            break
        fr = by_frame[target]
        for role, oid in frames_needed[target]:
            canvas = img.copy()
            marks = [ob for ob in fr["objects"] if ob.get("entityId")]
            marks.sort(key=lambda ob: ob["entityId"] != oid)
            for k, ob in enumerate(marks):
                polys = [np.asarray(p, np.int32) for p in ob["polygons"] if len(p) >= 3]
                cv2.polylines(canvas, polys, True, (0, 0, 0), 4)
                cv2.polylines(canvas, polys, True, (255, 255, 255), 2)
                if polys:
                    x0, y0 = np.vstack(polys).min(0)
                    for col, th in (((0, 0, 0), 4), ((255, 255, 255), 1)):
                        cv2.putText(canvas, f"[{k + 1}]", (int(x0), int(max(y0 - 4, 14))), cv2.FONT_HERSHEY_SIMPLEX, .6, col, th)
            sub = next(ob for ob in marks if ob["entityId"] == oid)
            allp = np.vstack([np.asarray(p, float) for p in sub["polygons"]])
            (x0, y0), (x1, y1) = allp.min(0), allp.max(0)
            cxm, cym, half = (x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, y1 - y0, 60) * .8
            a0, b0 = int(max(cxm - half, 0)), int(max(cym - half, 0))
            crop = canvas[b0:int(min(cym + half, img.shape[0])), a0:int(min(cxm + half, img.shape[1]))]
            scale = 320 / max(crop.shape[:2])
            crop = cv2.resize(crop, (max(1, int(crop.shape[1] * scale)), max(1, int(crop.shape[0] * scale))), interpolation=cv2.INTER_AREA)
            for qual in (80, 65, 50):
                jpg = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, qual])[1].tobytes()
                if len(jpg) <= 30_000:
                    break
            images[role] = jpg

    # ---- write: the recording's patches renumbered by time with the new ones in, blobs hard-linked
    (out / "blobs" / "sha256").mkdir(parents=True, exist_ok=True)
    for p in rec.patches:
        for ref in p["blobs"].values():
            dst = out / "blobs" / "sha256" / ref["sha256"]
            if not dst.exists():
                try:
                    os.link(rec.blob(ref), dst)
                except OSError:
                    dst.write_bytes(rec.blob(ref).read_bytes())

    def ref(payload, **meta):
        return {"sha256": put_blob(out, payload), "bytes": len(payload), **meta}

    def inline_or_blob(data, key, role):
        text = json.dumps(data[key], separators=(",", ":"))
        if len(text) < 1_000_000:
            return data, {}
        return {**data, key: "blob"}, {role: ref(text.encode(), mediaType="application/json", format="json")}

    objs, outl = rec.first["objects"], rec.first["outlines"]
    w = lambda p: rec.written.get(str(p["seq"]), p["sent_s"])
    pick_data = {"format": "panoptes-pick-v1", "source_wh": [analysis["width"], analysis["height"]], "entities": entity_ids, "frames": pick_frames,
                 "depth": {"w": DEPTH_W, "h": DEPTH_H, "unit": "mm", "scale": SCALE, "source": "room mesh vertices (fixture), not DA3"},
                 "offset_unit": "pairs", "note": "segmented frames: SAM 3 masks (observed); projected frames: carried from 3D (estimated); people: fixture boards",
                 "fixture": NOTE}
    shots_meta = [{"index": i, "floor_frame": {"origin_m": [r(v) for v in F[i]["origin"]], "x": [r(v, 5) for v in F[i]["x"]], "z": [r(v, 5) for v in F[i]["z"]]},
                   "u_pose_m": r(u_pose), "u_floor_m": .02, "plumb_deg": None, "angles_usable": False,
                   "scale": {"status": "estimated", "source": cams["scale"]["source"], "u_rel": .2}} for i in shots]

    def cards_layer(cs, version):
        data = {"schema": "panoptes-object-cards-v1", "version_of": {"objects": 1, "pick": 1}, "fixture": NOTE,
                "calibration": {"file_sha256": None, "k": {"height": 1, "extent": 1, "position": 1, "angle": 1}}, "shots": shots_meta, "cards": cs + pcards}
        return inline_or_blob(data, "cards", "cards")

    c1, cb1 = cards_layer(cards, 1)
    c2, cb2 = cards_layer(cards_v2, 2)
    ev_refs = {role: ref(jpg, mediaType="image/jpeg", format="jpeg") for role, jpg in images.items()}
    ev_of = lambda rs: {x["image"]: ev_refs[x["image"]] for row in rs for x in row["evidence"] if x.get("image") in ev_refs}
    new = [  # (sent_s, written_s, layer, data, blobs)
        (outl["sent_s"] + .3, w(outl) + .5, "pick", pick_data, {"pick": ref(pick_blob, mediaType="application/octet-stream", format="panoptes-pick-v1"),
                                                               "depth": ref(depth_blob, mediaType="application/octet-stream", format="panoptes-pick-depth-v1")}),
        (objs["sent_s"] + 5.0, objs["sent_s"] + 6.5, "object_cards", c1, cb1),
        (objs["sent_s"] + 6.0, objs["sent_s"] + 7.5, "judgements", layer(rows), ev_of(rows)),
        (objs["sent_s"] + 38.0, objs["sent_s"] + 39.5, "object_cards", c2, cb2),
        (objs["sent_s"] + 42.0, objs["sent_s"] + 43.5, "judgements", layer(rows_v2), ev_of(rows_v2)),
    ]
    stream = [(p["sent_s"], w(p), p["layer"], p) for p in rec.patches] + [(s_, w_, l_, (d_, b_)) for s_, w_, l_, d_, b_ in new]
    stream.sort(key=lambda x: x[0])
    folder = out / "reports" / as_report
    (folder / "patches").mkdir(parents=True, exist_ok=True)
    for old in (folder / "patches").glob("*.json"):
        old.unlink()
    versions, written = {}, {}
    template = rec.patches[0]
    for seq, (sent, wr, lay, body) in enumerate(stream, 1):
        versions[lay] = versions.get(lay, 0) + 1
        if isinstance(body, dict):
            patch = {**body, "report": as_report, "seq": seq, "version": versions[lay]}
        else:
            patch = {"schema": template["schema"], "report": as_report, "seq": seq, "layer": lay, "version": versions[lay], "status": "fixture",
                     "labels": [NOTE], "t0_unix": template["t0_unix"], "queued_s": round(sent, 3), "sent_s": round(sent, 3), "data": body[0], "blobs": body[1]}
        _write_json(folder / "patches" / f"{seq:06d}-{lay}.json", patch)
        written[str(seq)] = round(wr, 3)
    _write_json(folder / "written.json", written)
    if rec.run:
        _write_json(folder / "run.json", rec.run)
    stats = {"report": as_report, "patches": len(stream), "pick_frames": len(pick_frames), "pick_gz_mb": round(len(pick_blob) / 1e6, 3),
             "depth_gz_mb": round(len(depth_blob) / 1e6, 3), "cards": len(cards), "people_cards": len(pcards), "rows": len(rows),
             "verdicts": layer(rows)["counts"], "implausible": sum(c["physical"]["size_check"]["status"] == "implausible" for c in cards),
             "evidence_images": len(images), "evidence_kb_max": round(max(map(len, images.values()), default=0) / 1e3, 1),
             "cards_inline": not cb1}
    return stats


def self_check():
    rng = np.random.default_rng(0)
    m = np.zeros((360, 640), np.uint16)
    m[10:50, 20:200] = 3
    m[30:40, 50:60] = 7
    pairs = rle(m)
    assert (unrle(pairs, 640, 360) == m).all() and pairs[:, 1].max() <= 65535
    big = np.zeros((360, 640), np.uint16)  # 230400 zeros: split into 65535 runs
    p = rle(big)
    assert p[:, 1].tolist() == [65535, 65535, 65535, 230400 - 3 * 65535] and (unrle(p, 640, 360) == 0).all()
    noise = rng.integers(0, 4, (360, 640)).astype(np.uint16)
    assert (unrle(rle(noise), 640, 360) == noise).all()
    assert size_class("stacked boxes")[0] == "stacked boxes" and size_class("water bottles")[0] == "bottle"
    assert size_class("shelves")[0] == "shelf" and size_class("control panel")[0] is None
    assert kind_of("cable")["mobility"] == "deformable" and kind_of("carton")["category"] == "F payload"
    assert band(3.0, .3, 2.5, "max") == "FAIL" and band(2.0, .3, 2.5, "max") == "PASS" and band(2.4, .3, 2.5, "max") == "NEEDS_REVIEW"
    assert band(.2, .1, .5, "min") == "FAIL" and band(1.0, .1, .5, "min") == "PASS"
    assert worst(["PASS", "NO_DATA"]) == "NO_DATA" and worst(["NO_DATA", "NEEDS_REVIEW", "PASS"]) == "NEEDS_REVIEW"
    shot = {"c2w": [np.eye(4).tolist()], "floor": {"normal": [0, 1, 0], "point_m": [0, 1.6, 0]}}  # opencv: y down, floor 1.6 m below
    F = floor_frame(shot)
    assert np.allclose(F["z"], [0, -1, 0]) and np.allclose(to_floor(F, [[0, 0, 0]])[0], [0, 0, 1.6])
    assert np.allclose(to_floor(F, [[0, 1.6, 2]])[0], [2, 0, 0])
    q = quantity(1.0, {"depth": .05, "scale": .2}, subsets=[.97, 1.03])
    assert q["n_subsets"] == 2 and abs(q["u"] - math.hypot(.05, .2)) < 1e-3 and "note" not in q
    print("mvp_viewer_fixture self-check passed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src", nargs="?")
    ap.add_argument("report", nargs="?")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--as", dest="as_report")
    ap.add_argument("--self-check", action="store_true")
    a = ap.parse_args()
    if a.self_check:
        self_check()
    else:
        print(json.dumps(build(a.src, a.report, a.out, a.as_report)))
