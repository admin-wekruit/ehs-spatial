"""Stand-in object cards for the judgement engine (MVP spec 9: B works against a stub of A's cards), from the fast path's own
objects (today's inflated boxes, L1), cameras, outlines and people. A's cards (mvp/a-cards) replace this file: they have the
eroded, clustered points, view subsets, CLASS_SIZE and the calibrated identity. Here every metric value is one view set
(n_subsets 1), so no metric check can PASS or FAIL on a stub card, and depth is 'not observed' unless the views spread
over 30 deg of azimuth. Names are SAM 3's words ('detected word, unverified').

  python -m fast_report.cards_stub RUNS/fb-integrate-me340-006 REPORT [--out judgements.json]   # local: stub cards + v1
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

from fast_report.judge import DEFORMABLE, SCALE_REL, is_a, to_floor

STUB_MAX_M = {"box": 1.5, "carton": 1.5, "crate": 1.5, "bin": 1.5, "cart": 2.2, "pallet": 1.4, "boxes": 3.5, "stacked boxes": 3.5,
              "chair": 1.3, "ladder": 6., "monitor": 1.6, "sign": 2., "fire extinguisher": 1.}  # a few of A's CLASS_SIZE rows
F_PX = 262.  # DA3's focal length at 504 x 280 (spec 4.4's resolution term)


def _area(poly):
    p = np.asarray(poly, float).reshape(-1, 2)
    return .5 * abs(np.dot(p[:, 0], np.roll(p[:, 1], 1)) - np.dot(p[:, 1], np.roll(p[:, 0], 1)))


def _val(value, parts, unit="m", scale="estimated"):
    return {"value": round(float(value), 3), "u": round(math.sqrt(sum(v * v for v in parts.values())), 3), "unit": unit, "level": "coarse",
            "scale": scale, "n_subsets": 1, "parts": {k: round(v, 4) for k, v in parts.items()}, "stub": True}


def cards(objects, cam_rows, outline_frames, people, ctx):
    from shapely.geometry import MultiPoint
    frames = {s["index"]: s["floor_frame"] for s in ctx["shots"]}
    cams = {s["index"]: s for s in cam_rows}
    seen = {}
    for f in outline_frames:
        for e in f["objects"]:
            seen.setdefault(e["entityId"], []).append((f["sourceFrame"], f["timeSec"], f["source"], sum(_area(p) for p in e["polygons"])))
    out = []
    for o in objects:
        if o["shot"] not in frames:
            continue
        fr, cam = frames[o["shot"]], cams[o["shot"]]
        lo, hi = np.asarray(o["box_min_m"]), np.asarray(o["box_max_m"])
        corners = to_floor(fr, [[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        centre = to_floor(fr, o["centroid_m"])[0]
        rect = MultiPoint(corners[:, :2]).minimum_rotated_rectangle
        ring = np.asarray(rect.exterior.coords)[:4] if rect.geom_type == "Polygon" else corners[:4, :2]
        views = sorted(seen.get(o["id"], []), key=lambda v: -v[3])
        detected = [v for v in views if v[2] == "segmented"]
        key_pos = {k: i for i, k in enumerate(cam["keys"])}
        cam_xyz = {k: to_floor(fr, np.asarray(cam["c2w"][key_pos[k]])[:3, 3])[0] for k, *_ in views if k in key_pos}
        az = {k: math.degrees(math.atan2(*(centre[:2] - c[:2])[::-1])) for k, c in cam_xyz.items()}
        dist = float(np.median([np.linalg.norm(centre - c) for c in cam_xyz.values()])) if cam_xyz else 5.
        dz = float(np.median([abs(c[2] - centre[2]) for c in cam_xyz.values()])) if cam_xyz else 1.6
        spread = (lambda a: 0. if len(a) < 2 else min(360 - (max(a) - min(a)), max(a) - min(a)) if max(a) - min(a) > 180 else max(a) - min(a))(
            [az[k] for k, *_ in detected if k in az])
        best = []
        for k, *_ in detected:  # largest outlines, >= 15 deg apart where possible (X7's spread_views, greedily)
            if all(abs((az.get(k, 0) - az.get(b, 0) + 180) % 360 - 180) >= 15 for b in best):
                best.append(k)
            if len(best) == 3:
                break
        best += [k for k, *_ in detected if k not in best][:max(0, 1 - len(best))]
        top, base = corners[:, 2].max(), corners[:, 2].min()
        e1, e2 = ring[1] - ring[0], ring[2] - ring[1]
        view_dir = centre[:2] - np.mean([c[:2] for c in cam_xyz.values()], 0) if cam_xyz else np.array([1., 0])
        cos = lambda e: abs(e @ view_dir) / (np.linalg.norm(e) * np.linalg.norm(view_dir) + 1e-9)  # noqa: E731
        width, depth = (np.linalg.norm(e1), np.linalg.norm(e2)) if cos(e1) < cos(e2) else (np.linalg.norm(e2), np.linalg.norm(e1))
        res = 4 * dist / F_PX
        pose, floor = ctx["shots"][0].get("u_pose_m", .04), .02
        name = (o.get("label") or o.get("word") or "").lower()
        longest = max(top - base, width, depth)
        cap = next((v for w, v in STUB_MAX_M.items() if is_a(name, (w,))), 30. if is_a(name, DEFORMABLE + ("shelf", "rack", "shelving")) else 6.)
        ext = lambda v: _val(v, {"depth": .05 * v, "resolution": res, "scale": SCALE_REL * v})  # noqa: E731
        phys = {"top_above_floor": _val(top, {"depth": .05 * dz, "floor": floor, "pose": pose, "scale": SCALE_REL * abs(top)}),
                "base_above_floor": _val(base, {"depth": .05 * dz, "floor": floor, "pose": pose, "scale": SCALE_REL * abs(base)}),
                "height": ext(top - base), "width": ext(width),
                "depth": ext(depth) if spread >= 30 else {"status": "not observed", "reason": f"seen from one side (azimuth spread {spread:.0f} deg)"},
                "position_xy": {**_val(0, {"depth": .05 * dist, "pose": pose, "scale": SCALE_REL * float(np.linalg.norm(centre[:2]))}),
                                "value": [round(float(v), 3) for v in np.asarray(rect.centroid.coords[0])]},
                "footprint_xy": [[round(float(x), 3), round(float(y), 3)] for x, y in ring],
                "size_check": {"status": "implausible" if longest > cap else "plausible", "longest_m": round(float(longest), 2), "max_m": cap,
                               "note": "stub: a few of A's CLASS_SIZE rows"},
                "principal_axis_tilt_deg": {"status": "not measurable", "reason": "stub cards: no view subsets"},
                "planar_slope_deg": {"status": "not measurable", "reason": "stub cards: no view subsets"},
                "overhang": {"status": "not measurable", "reason": "stub cards: no points"}}
        out.append({"id": o["id"], "kind": "object", "shot": o["shot"], "stub": True,
                    "identity": {"name": name, "confidence": None, "calibrated": False, "decided_by": "sam3 vote", "label": "detected word, unverified"},
                    "class": {}, "physical": phys,
                    "views": {"n": len(detected), "keyframes": [k for k, *_ in sorted(detected)], "best": best, "distance_m": round(dist, 2),
                              "azimuth_spread_deg": round(spread, 1)},
                    "time": {"first_seen_s": min((v[1] for v in views), default=None), "last_seen_s": max((v[1] for v in views), default=None)}})
    rules = (people or {}).get("rules", [])
    for tr in (people or {}).get("tracks", []):
        name = tr["id"].split("-", 1)[1]
        out.append({"id": f"person:{tr['id']}", "kind": "person", "shot": tr["shot"], "track": tr["id"], "points": tr["points"],
                    "rules": [r for r in rules if r.get("shot") == tr["shot"] and name in (r.get("tracks") or [])],
                    "identity": {"name": "person"}, "time": {"first_seen_s": tr["t0"], "last_seen_s": tr["t1"]}})
    return out


def glb_points(raw):
    """POSITION of a glb-points blob (core.points_glb's layout) -> (n, 3) float32."""
    import struct
    n_json = struct.unpack_from("<I", raw, 12)[0]
    gltf = json.loads(raw[20:20 + n_json])
    body = 20 + n_json + 8
    view = gltf["bufferViews"][gltf["accessors"][0]["bufferView"]]
    return np.frombuffer(raw, "<f4", count=3 * gltf["accessors"][0]["count"], offset=body + view["byteOffset"]).reshape(-1, 3)


def load(run_dir, report):
    """A recorded report's newest version of each layer -> {layer: (data, blobs)}."""
    folder = Path(run_dir) / "reports" / report / "patches"
    out = {}
    for p in sorted(folder.glob("*.json")):
        patch = json.loads(p.read_text())
        out[patch["layer"]] = (patch["data"], patch["blobs"])
    return out


def replay(run_dir, report, with_frames=False):
    """Local: stub cards + ctx from a recorded report (the room points from its GLB blobs, frames decoded on demand)."""
    from fast_report import judge
    run_dir = Path(run_dir)
    lay = load(run_dir, report)
    blob = lambda b: (run_dir / "blobs/sha256" / b["sha256"]).read_bytes()  # noqa: E731
    cams, objs = lay["cameras"][0], lay["objects"][0]
    outlines = json.loads(blob(lay["outlines"][1]["analysis"]))
    room = {int(k.split("-")[1]): glb_points(blob(b)) for k, b in lay["room"][1].items() if k.startswith("points-")}
    frames = None
    if with_frames:
        import cv2
        mp4 = sorted(run_dir.glob("input-*.mp4"))[0]

        class Frames(dict):
            def __missing__(self, k):
                cap = cv2.VideoCapture(str(mp4))
                cap.set(cv2.CAP_PROP_POS_FRAMES, k)
                ok, img = cap.read()
                cap.release()
                if not ok:
                    raise KeyError(k)
                self[k] = img
                return img
        frames = Frames()
    ctx = judge.context(cams["shots"], outlines["frames"], lay["people"][0], frames, room, cams["fps"], (outlines["width"], outlines["height"]))
    return cards(objs["objects"], cams["shots"], outlines["frames"], lay["people"][0], ctx), ctx


if __name__ == "__main__":
    from fast_report import judge
    cs, ctx = replay(sys.argv[1], sys.argv[2])
    rows = judge.evaluate(cs, ctx)
    lay = judge.layer(rows, judge.load_calibration())
    print(json.dumps({"cards": len(cs), "rows": len(rows), "counts": lay["counts"], "by_check": lay["by_check"],
                      "implausible": sum((c.get("physical") or {}).get("size_check", {}).get("status") == "implausible" for c in cs)}, indent=1))
    if "--out" in sys.argv:
        Path(sys.argv[sys.argv.index("--out") + 1]).write_text(json.dumps({"cards": cs, "judgements": lay}, default=str))
