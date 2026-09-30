"""X6 evaluation on the Mac (numpy): the runs of modal_apps/x6_windows_app.py next to the delivered reports.

Per run JSON: ATE of the keyframe cameras vs DROID (the delivered camera, metric by its own scale; E1's Sim3 with
orientations), on the map frame holding most of the reference shot's keyframes; duplicates vs the delivered object list
(the 22 / 67 / 40 objects of the merged models, centroids from the named object map): our global objects, and the
per-window instances before association, carried into DROID's frame by that Sim3, within 0.3 / 0.5 m of each
delivered object; timing marks; per-GPU peaks. Changes are drawn on a contact sheet: before and after keyframes from
the source MP4, the object's points projected (green where it was seen, red where it was judged gone / new).

    python scripts/x6_evaluate.py RUN_DIR [RUN_DIR ...] --out OUT_DIR
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "scripts"), str(REPO / "modal_apps"), str(REPO)]
import fast_report_eval as fe  # noqa: E402  references (DROID, scale, object map) from the delivered fixtures
import m3_exp_geometry as geo  # noqa: E402

PHASE2 = Path("/Users/adam/Desktop/panoptes-public/research-notes/phase2")
CLIPS = {"me340": "me340-165", "samsclub-a2": "samsclub-337", "walmart": "walmart-190"}
MERGED = {"me340": "me340-object-models-303-merged", "samsclub-a2": "samsclub-a2-object-models-303-merged", "walmart": "walmart-object-models-303-merged"}
GRID = (252, 140)  # the state grid (DA3's 504 x 280 at stride 2)


def delivered(site, ref):
    ids = list(json.loads((PHASE2 / "runs" / MERGED[site] / "merge.json").read_text())["choice"])
    ents = {e["entityId"]: e for e in json.loads(ref["object_map"].read_text())["entities"]}
    names = json.loads(ref["names"].read_text())
    return [{"id": i, "name": names.get(i, {}).get("category"), "xyz": np.array(ents[i]["centroidNative"]) * ref["mpn"],
             "box": np.array(ents[i]["boundsNative"]) * ref["mpn"]} for i in ids if i in ents]


def fragments(points, dl, sim, pad=.1):
    """Per delivered object: how many of `points` lie inside its box (+ pad): 2 or more = one object in pieces."""
    s, R, t = sim
    if not len(points):
        return {"covered": 0, "of": len(dl), "in_pieces": 0}
    xyz = (s * (R @ np.asarray(points, float).T)).T + t
    per = np.array([np.all((xyz >= o["box"][0] - pad) & (xyz <= o["box"][1] + pad), axis=1).sum() for o in dl])
    return {"covered": int((per >= 1).sum()), "of": len(dl), "in_pieces": int((per >= 2).sum()), "pieces_per_covered_median": float(np.median(per[per >= 1])) if (per >= 1).any() else None,
            "ours_inside": int(per.sum())}


def split_candidates(objs, radius=.3):
    """Pairs of our objects that are probably one object split across windows: same map frame and word, never seen in one
    window together, some per-window positions within `radius`."""
    n = 0
    for i, a in enumerate(objs):
        wa = {p["window"] for p in a["positions"]}
        pa = np.array([p["centroid"] for p in a["positions"]])
        for b in objs[i + 1:]:
            if b["frame"] != a["frame"] or b["label"] != a["label"] or wa & {p["window"] for p in b["positions"]}:
                continue
            pb = np.array([p["centroid"] for p in b["positions"]])
            n += bool((np.linalg.norm(pa[:, None] - pb[None], axis=2) <= radius).any())
    return n


def baseline_me340():
    """Today's core on ME340 (fb-a-core-011, the whole-shot lift) in the same shape as a run, for the duplicate rows."""
    rep = PHASE2 / "runs/fb-a-core-011/fb-me340-e84efffd-1790639941/patches"
    cams = json.loads((rep / "000002-cameras.json").read_text())["data"]["shots"]
    objs = json.loads((rep / "000008-objects.json").read_text())["data"]["objects"]
    cameras = {str(k): {"frame": s["index"], "c2w": c} for s in cams for k, c in zip(s["keyframes"], s["c2w"])}
    return {"cameras": cameras, "keyframes": [k for s in cams for k in s["keyframes"]],
            "objects": [{"frame": o["shot"], "label": o["word"], "positions": [{"window": 0, "centroid": o["centroid_m"]}]} for o in objs]}


def align(run, ref):
    """Sim3 our map frame -> DROID metres on the reference shot's keyframes, for the frame with most of them."""
    frames = set(ref["frames"])
    by = {}
    for k, c in run["cameras"].items():
        if int(k) in frames:
            by.setdefault(c["frame"], []).append((int(k), np.array(c["c2w"])))
    if not by:
        return None
    fr, rows = max(by.items(), key=lambda kv: len(kv[1]))
    rows.sort()
    ours = np.stack([c for _, c in rows])
    target = np.load(ref["droid"])["poses_c2w"][[k for k, _ in rows]].astype(np.float64)
    target[:, :3, 3] *= ref["mpn"]
    s, R, t = geo.align_sim3(ours, target)
    err = np.linalg.norm((s * (R @ ours[:, :3, 3].T)).T + t - target[:, :3, 3], axis=1)
    rot = [geo.angle_deg(tg[:3, :3].T @ R @ o[:3, :3]) for o, tg in zip(ours, target)]
    path = float(np.linalg.norm(np.diff(target[:, :3, 3], axis=0), axis=1).sum())
    n_ref = len({k for k in map(int, run["cameras"]) if k in frames} | {k for k in run["keyframes"] if k in frames})
    return {"frame": fr, "keyframes": len(rows), "reference_keyframes": n_ref, "coverage": round(len(rows) / max(n_ref, 1), 3),
            "ate_m": round(float(np.sqrt((err ** 2).mean())), 4), "path_m": round(path, 3), "ate_share_of_path": round(float(np.sqrt((err ** 2).mean())) / path, 4) if path else None,
            "rotation_error_deg_median": round(float(np.median(rot)), 2), "ours_metres_over_reference": round(1 / s, 4),
            "frames_touching_reference": len(by), "sim3": (s, R, t)}


def duplicates(points, labels, dl, sim, radius):
    """Per delivered object: how many of `points` (our frame) lie within radius after the Sim3 (and with a matching name)."""
    s, R, t = sim
    if not len(points):
        return {"matched": 0, "of": len(dl), "duplicated": 0}
    xyz = (s * (R @ np.asarray(points, float).T)).T + t
    d = np.linalg.norm(np.stack([o["xyz"] for o in dl])[:, None] - xyz[None], axis=2)
    near = d <= radius
    named = near & np.array([[fe.same_name(lb, o["name"] or "") for lb in labels] for o in dl])
    per = near.sum(1)
    per_n = named.sum(1)
    return {"matched": int((per >= 1).sum()), "of": len(dl), "duplicated": int((per >= 2).sum()), "ours_per_matched_median": float(np.median(per[per >= 1])) if (per >= 1).any() else None,
            "matched_named": int((per_n >= 1).sum()), "duplicated_named": int((per_n >= 2).sum()), "ours_near_any": int(near.any(0).sum()), "ours": int(len(xyz))}


def evaluate(run, site):
    ref = fe.reference(site)
    dl = delivered(site, ref)
    row = {"site": site, "geometry": run["opts"]["geometry"], "threshold": run["opts"]["threshold"], "error": run["error"]}
    if run["error"]:
        return row, None
    s = run["summary"]
    marks = s["marks"]
    row["timing_s"] = {k: marks.get(k) for k in ("first_window_dispatched", "first_window_facts", "first_objects", "decoded", "cuts_final", "da3_done", "all_windows_facts")}
    row["client_wall_s"] = run.get("client_wall_s")
    row["gpu_peak_gb"] = [{"gpu": g["gpu"], "peak_gb": g["peak_gb"], "total_gb": g["total_gb"], "at_s": g["at_s"]} for g in run["clock"]["gpu_peak"]]
    row["flags"] = run["clock"]["flags"]
    stages = {}
    for st in run["clock"]["stages"]:
        base = st["stage"].split("@")[0].rstrip("0123456789")
        x = stages.setdefault(base, {"n": 0, "s": 0., "peak_gb": [0, 0]})
        x["n"] += 1
        x["s"] = round(x["s"] + st["s"], 3)
        x["peak_gb"] = [max(a or 0, b or 0) for a, b in zip(x["peak_gb"], st.get("peak_gb") or [0, 0])]
    row["stages"] = stages
    W = run["windows"]
    row["windows"] = {"n": len(W), "with_objects": sum(1 for w in W if w["instances"]), "skipped": sum(1 for w in W if w.get("skipped")),
                      "keys_median": float(np.median([len(w["keys"]) for w in W])), "gpu": [w.get("gpu") for w in W],
                      "frames": len({w["frame"] for w in W if w["frame"] is not None}), "spanning_final_cut": s["windows_spanning_final_cut"],
                      "window_s": [round(w["done_s"] - w["started_s"], 2) for w in W if w.get("done_s") and w.get("started_s")]}
    st = [x for x in s["stitches"] if x.get("accepted") is not None]
    row["stitch"] = {"tried": len(st), "accepted": sum(x["accepted"] for x in st), "no_carry_or_depth": len(s["stitches"]) - len(st),
                     "rotation_residual_deg_median": round(float(np.median([x["rotation_residual_deg"] for x in st])), 3) if st else None,
                     "centre_residual_m_median": round(float(np.median([x["centre_residual_m"] for x in st])), 4) if st else None,
                     "scale_median": round(float(np.median([x["s"] for x in st])), 4) if st else None,
                     "refused": [{k: x[k] for k in ("window", "rotation_residual_deg", "centre_residual_m", "centre_limit_m", "s")} for x in st if not x["accepted"]]}
    a = align(run, ref)
    if a:
        sim = a.pop("sim3")
        row["ate_vs_droid"] = a
        objs = [o for o in run["objects"] if o["frame"] == a["frame"]]
        inst = [x for x in run["per_window_instances"] if x["frame"] == a["frame"]]
        cent = [o["positions"][0]["centroid"] for o in objs]
        row["duplicates_vs_delivered"] = {f"{r}m": {"global_objects": duplicates(cent, [o["label"] for o in objs], dl, sim, r),
                                                    "per_window_instances": duplicates([x["centroid"] for x in inst], [x["label"] for x in inst], dl, sim, r)}
                                          for r in (.3, .5)}
        row["pieces_in_delivered_boxes"] = {"global_objects": fragments(cent, dl, sim), "per_window_instances": fragments([x["centroid"] for x in inst], dl, sim)}
        if site == "me340":
            b = baseline_me340()
            ab = align(b, ref)
            if ab:
                bs = ab.pop("sim3")
                bo = [o for o in b["objects"] if o["frame"] == ab["frame"]]
                row["today_core_fb_a_core_011"] = {"ate_m": ab["ate_m"], "objects_in_frame": len(bo),
                                                   "pieces_in_delivered_boxes": fragments([o["positions"][0]["centroid"] for o in bo], dl, bs),
                                                   "within_0.3m": duplicates([o["positions"][0]["centroid"] for o in bo], [o["label"] for o in bo], dl, bs, .3)}
    objs = run["objects"]
    row["objects"] = {"global": len(objs), "per_window_instances": len(run["per_window_instances"]),
                      "seen_in_2plus_windows": sum(o["windows_observed"] >= 2 for o in objs),
                      "split_candidates_0.3m_same_word": split_candidates(objs),
                      "states": count_states(objs)}
    row["changes"] = [{k: c.get(k) for k in ("object", "label", "kind", "window", "t_before", "t_after", "before_key", "after_key", "distance_m", "appearance_cos")}
                      | {"free_views": c["place"]["free_views"], "best_free_share": c["place"]["best_free_share"], "withdrawn_in_window": c.get("withdrawn_in_window")}
                      for c in run["changes"]]
    row["tau_move"] = s["tau_move_calibrated"]
    row["facts"] = run["facts"]
    return row, (ref, dl)


def count_states(objs):
    out = {}
    for o in objs:
        for iv in o["intervals"]:
            k = iv["state"].split(":")[0] if not iv["state"].startswith("not-observed") else iv["state"]
            out[k] = out.get(k, 0) + 1
    return out


# ---------- contact sheets ----------

def frame_at(cap, k):
    cap.set(cv2.CAP_PROP_POS_FRAMES, k)
    return cap.read()[1]


def draw(img, pts, cam, color):
    """Project world points (map frame) with the keyframe's camera (grid K at stride 2) onto the 1280x720 frame."""
    if cam is None or pts is None or not len(pts):
        return img
    c2w, K = np.array(cam["c2w"]), np.array(cam["K_grid_stride2"])
    c = (np.asarray(pts) - c2w[:3, 3]) @ c2w[:3, :3]
    ok = c[:, 2] > .05
    u = (K[0, 0] * c[ok, 0] / c[ok, 2] + K[0, 2] + .5) * img.shape[1] / GRID[0] - .5
    v = (K[1, 1] * c[ok, 1] / c[ok, 2] + K[1, 2] + .5) * img.shape[0] / GRID[1] - .5
    p = np.stack([u, v], 1).astype(np.int32)
    p = p[(p[:, 0] > -2000) & (p[:, 0] < 4000) & (p[:, 1] > -2000) & (p[:, 1] < 4000)]
    if len(p) >= 3:
        cv2.polylines(img, [cv2.convexHull(p)], True, color, 4)
    for x, y in p[::2]:
        cv2.circle(img, (int(x), int(y)), 4, color, -1)
    return img


def label(img, text):
    cv2.rectangle(img, (0, 0), (img.shape[1], 44), (0, 0, 0), -1)
    cv2.putText(img, text, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return img


def sheet(run, site, out_path, max_rows=12):
    ch = run["changes"][:max_rows]
    if not ch:
        return None
    cap = cv2.VideoCapture(str(PHASE2 / "data/clips" / CLIPS[site] / "source-full.mp4"))
    fps = run["fps"]
    rows = []
    for c in ch:
        cb, ca = run["cameras"].get(str(c["before_key"])), run["cameras"].get(str(c["after_key"]))
        b = frame_at(cap, c["before_key"]) if c["before_key"] is not None else np.zeros((720, 1280, 3), np.uint8)
        a = frame_at(cap, c["after_key"]) if c["after_key"] is not None else np.zeros((720, 1280, 3), np.uint8)
        if c["kind"] == "appeared":  # before: its place, empty; after: the new object
            draw(b, c["points_after"], cb, (0, 0, 255))
            draw(a, c["points_after"], ca, (0, 255, 0))
        else:  # disappeared / moved: before: the object; after: its old place (red), and for moved the new place (green)
            draw(b, c["points_before"], cb, (0, 255, 0))
            draw(a, c["points_before"], ca, (0, 0, 255))
            if c["kind"] == "moved":
                draw(a, c["points_after"], ca, (0, 255, 0))
        wd = " (withdrawn later)" if c.get("withdrawn_in_window") is not None else ""
        label(b, f"{c['object']} {c['label']}: {c['kind']}{wd} | before {c['before_key'] / fps:.1f} s")
        label(a, f"after {c['after_key'] / fps:.1f} s, free share {c['place']['best_free_share']}, views {c['place']['free_views']}"
              if c["after_key"] is not None else "after: -")
        rows.append(np.hstack([cv2.resize(b, (640, 360)), cv2.resize(a, (640, 360))]))
    img = np.vstack(rows)
    cv2.imwrite(str(out_path), img, [cv2.IMWRITE_JPEG_QUALITY, 72])
    return str(out_path)


def facts_sheet(run, site, out_path):
    fs = run["facts"]
    if not fs:
        return None
    cap = cv2.VideoCapture(str(PHASE2 / "data/clips" / CLIPS[site] / "source-full.mp4"))
    rows = []
    for f in fs:
        tiles = []
        for k in f["evidence_keys"][:3]:
            img = frame_at(cap, k)
            draw(img, run["fact_points"].get(f["object"]), run["cameras"].get(str(k)), (0, 200, 255))
            label(img, f"{k / run['fps']:.1f} s")
            tiles.append(cv2.resize(img, (426, 240)))
        while len(tiles) < 3:
            tiles.append(np.zeros((240, 426, 3), np.uint8))
        head = np.zeros((40, 1278, 3), np.uint8)
        cv2.putText(head, f"{f['label']}: {f['property']} = {f['value']} +- {f['uncertainty']} over {f['interval_s'][0]}-{f['interval_s'][1]} s ({f['status']})",
                    (8, 28), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2)
        rows += [head, np.hstack(tiles)]
    cv2.imwrite(str(out_path), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 72])
    return str(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sheets", action="store_true")
    ap.add_argument("--windows", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    rows = {}
    for d in args.runs:
        for p in sorted(Path(d).glob("x6-*.json")):
            run = json.loads(p.read_text())
            site = run["opts"]["site"]
            row, _ = evaluate(run, site)
            if not run["error"] and run["opts"].get("planted"):
                row["planted"] = planted(run, d)
            if not run["error"] and args.windows and run["opts"]["geometry"] == "a":
                row["window_rule_vs_da3"] = window_compare(run)
            rows[p.stem] = row
            if args.sheets and not run["error"]:
                row["contact_sheet"] = sheet(run, site, out / f"{p.stem}-changes.jpg")
                row["facts_sheet"] = facts_sheet(run, site, out / f"{p.stem}-facts.jpg")
            print(p.stem, json.dumps({k: row.get(k) for k in ("timing_s", "ate_vs_droid", "stitch", "objects")}, default=str)[:1500], flush=True)
    (out / "evaluation.json").write_text(json.dumps(rows, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))



def sweep_table(sweep_dir, runs_dir):
    """Per sweep config and run: objects, cross-window split candidates, delivered objects in pieces, change claims."""
    out = {}
    for p in sorted(Path(sweep_dir).glob("sweep-x6-*.json")):
        rid = p.stem[len("sweep-"):]
        run = json.loads(next(Path(runs_dir).glob(f"{rid}.json")).read_text())
        site = run["opts"]["site"]
        ref = fe.reference(site)
        dl = delivered(site, ref)
        a = align(run, ref)
        sim = a.pop("sim3")
        rows = []
        for r in json.loads(p.read_text()):
            objs = [dict(o, positions=o["positions"]) for o in r["timelines"]]
            inframe = [o for o in objs if o["frame"] == a["frame"]]
            ch = r["changes"]
            rows.append({"cfg": r["cfg"], "objects": len(objs), "split_candidates": split_candidates(objs),
                         "pieces": fragments([o["positions"][0]["centroid"] for o in inframe], dl, sim),
                         "claims": {k: sum(c["kind"] == k for c in ch) for k in ("disappeared", "appeared", "moved")},
                         "claims_withdrawn": sum(c.get("withdrawn_in_window") is not None for c in ch), "s": r["s"]})
        out[rid] = rows
    return out


# ---------- planted changes (recall) ----------

def hull_mask(xy):
    m = np.zeros((720, 1280), np.uint8)
    xy = np.asarray(xy)
    xy = xy[np.isfinite(xy).all(1) & (np.abs(xy) < 5000).all(1)]
    if len(xy) >= 3:
        cv2.fillConvexPoly(m, cv2.convexHull(xy.astype(np.int32)), 1)
    return m > 0


def ours_xy(pts, cam):
    """Our object's points (map frame) on the 1280 x 720 source through our keyframe camera (the contact sheet's rule)."""
    if cam is None or pts is None:
        return np.zeros((0, 2))
    c2w, K = np.array(cam["c2w"]), np.array(cam["K_grid_stride2"])
    c = (np.asarray(pts) - c2w[:3, 3]) @ c2w[:3, :3]
    c = c[c[:, 2] > .05]
    return np.stack([(K[0, 0] * c[:, 0] / c[:, 2] + K[0, 2] + .5) * 1280 / GRID[0] - .5, (K[1, 1] * c[:, 1] / c[:, 2] + K[1, 2] + .5) * 720 / GRID[1] - .5], 1)


def planted_match(changes, cameras, truth, site, iou_min=.25):
    """Planted events vs change claims, in the image: at the claim's before (after, for 'appeared') keyframe the box
    drawn with the delivered camera and our object's points drawn with ours overlap by IoU >= iou_min, the kinds agree,
    and the event time lies between the claim's before and after keyframes (+- 1 s). Image space, so the map-to-map
    alignment (ATE 4-22 cm, 11-17 % scale) does not decide it."""
    import x6_plant as xp
    ref, d, clip, up, p0 = xp.load(site)
    poses, mpn = d["poses_c2w"].astype(np.float64), ref["mpn"]
    Kr = 2 * d["keyframe_final_fullres_intrinsics"][0].astype(float)

    def box_xy(e, pos, f):
        xy, z = xp.project(xp.cube(np.array(e["centre_native"][pos]), up, np.array(e["side_dir"]), xp.SIDE / mpn), poses[f], Kr)
        return xp.to_source(xy) if (z > 0).all() else np.zeros((0, 2))
    rows, used = [], set()
    for e in truth["events"]:
        if not e["placed"]:
            rows.append({"kind": e["kind"], "placed": False})
            continue
        best = None
        for i, c in enumerate(changes):
            if c["kind"] != e["kind"] or not (c["t_before"] - 1 <= e["change_s"] <= c["t_after"] + 1):
                continue
            if e["kind"] == "appeared":
                f, ours, box = c["after_key"], c.get("points_after"), box_xy(e, 0, c["after_key"])
            else:
                f, ours, box = c["before_key"], c.get("points_before"), box_xy(e, 0, c["before_key"])
            if f is None or f not in ref["segment"]:
                continue
            a, b = hull_mask(ours_xy(ours, cameras.get(str(f)))), hull_mask(box)
            iou = (a & b).sum() / max((a | b).sum(), 1)
            if iou >= iou_min and (best is None or iou > best[1]):
                best = (i, round(float(iou), 3))
        if best:
            used.add(best[0])
        rows.append({"kind": e["kind"], "placed": True, "change_s": e["change_s"], "found": best is not None,
                     "claim": changes[best[0]]["object"] if best else None, "iou": best[1] if best else None,
                     "claim_t": [changes[best[0]]["t_before"], changes[best[0]]["t_after"]] if best else None})
    other = [c for i, c in enumerate(changes) if i not in used]
    return {"events": rows, "placed": sum(r["placed"] for r in rows), "found": sum(bool(r.get("found")) for r in rows),
            "claims_not_planted": len(other), "claims_not_planted_detail": [(c["object"], c.get("label"), c["kind"], c["t_after"]) for c in other]}


def planted(run, run_dir):
    site = run["opts"]["site"]
    truth = json.loads((Path(run_dir) / f"planted-{site}.json").read_text())
    return planted_match(run["changes"], run["cameras"], truth, site)


# ---------- window rule: ORB co-visibility next to DA3's geometric overlap ----------

def window_compare(run, thresholds=(.4, .55, .7)):
    """(a) runs carry DA3's overlap of each keyframe with the next 60 of its shot: the window rule on it, and its
    correlation with the ORB co-visibility of the same keyframe pairs."""
    from fast_report import windows as win
    rows = run["summary"].get("overlap_da3") or []
    if not rows:
        return None
    site = run["opts"]["site"]
    cap = cv2.VideoCapture(str(PHASE2 / "data/clips" / CLIPS[site] / "source-full.mp4"))
    out = {"shots": []}
    orb_all, da3_all = [], []
    for sh in rows:
        keys, M = sh["keys"], sh["rows"]
        feats = []
        for k in keys:
            cap.set(cv2.CAP_PROP_POS_FRAMES, k)
            bgr = cap.read()[1]
            h, w = bgr.shape[:2]
            x0 = (w - h * 4 // 3) // 2
            feats.append(win.features(cv2.cvtColor(cv2.resize(bgr[:, x0:w - x0], (640, 480), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)))
        index = {id(f): i for i, f in enumerate(feats)}

        def da3(a, b):
            i, j = index[id(a)], index[id(b)]
            return M[i][j - i - 1] if 0 < j - i <= len(M[i]) else 0.
        for i in range(0, len(keys) - 1, 3):
            for j in (i + 1, i + 5, i + 10, i + 20):
                if j < len(keys) and j - i <= len(M[i]):
                    orb_all.append(win.covisibility(feats[i], feats[j]))
                    da3_all.append(M[i][j - i - 1])
        s = {"shot": sh["shot"], "keyframes": len(keys)}
        for thr in thresholds:
            wa = win.run(keys, feats, threshold=thr, covis=da3)
            wo = win.run(keys, feats, threshold=thr)
            s[str(thr)] = {"da3_windows": len(wa), "orb_windows": len(wo), "da3_span_s_median": round(float(np.median([(x["keys"][-1] - x["keys"][x["carried"]]) / run["fps"] for x in wa])), 2),
                           "orb_span_s_median": round(float(np.median([(x["keys"][-1] - x["keys"][x["carried"]]) / run["fps"] for x in wo])), 2)}
        out["shots"].append(s)
    o, d = np.array(orb_all), np.array(da3_all)
    out["pairs"] = len(o)
    out["pearson_orb_da3"] = round(float(np.corrcoef(o, d)[0, 1]), 3) if len(o) > 2 else None
    out["da3_overlap_when_orb_below"] = {str(x): round(float(np.median(d[o < x])), 3) if (o < x).any() else None for x in thresholds}
    out["orb_when_da3_below"] = {str(x): round(float(np.median(o[d < x])), 3) if (d < x).any() else None for x in thresholds}
    return out


if __name__ == "__main__":
    main()
