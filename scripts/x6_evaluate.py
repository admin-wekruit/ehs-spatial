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
    return [{"id": i, "name": names.get(i, {}).get("category"), "xyz": np.array(ents[i]["centroidNative"]) * ref["mpn"]} for i in ids if i in ents]


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
    row["timing_s"] = {k: marks.get(k) for k in ("first_window_dispatched", "first_window_facts", "decoded", "cuts_final", "da3_done", "all_windows_facts")}
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
    objs = run["objects"]
    row["objects"] = {"global": len(objs), "per_window_instances": len(run["per_window_instances"]),
                      "seen_in_2plus_windows": sum(o["windows_observed"] >= 2 for o in objs),
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
    args = ap.parse_args()
    out = Path(args.out)
    rows = {}
    for d in args.runs:
        for p in sorted(Path(d).glob("x6-*.json")):
            run = json.loads(p.read_text())
            site = run["opts"]["site"]
            row, _ = evaluate(run, site)
            rows[p.stem] = row
            if args.sheets and not run["error"]:
                row["contact_sheet"] = sheet(run, site, out / f"{p.stem}-changes.jpg")
                row["facts_sheet"] = facts_sheet(run, site, out / f"{p.stem}-facts.jpg")
            print(p.stem, json.dumps({k: row.get(k) for k in ("timing_s", "ate_vs_droid", "stitch", "objects")}, default=str)[:1500], flush=True)
    (out / "evaluation.json").write_text(json.dumps(rows, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


if __name__ == "__main__":
    main()
