"""mvp2 physical info (R3 tops, R4 people, R5 shape side, R7 card contract): the rows that say whether the fixes hold, per
mirrored call. Numbers are at estimated scale; 'delivered' is the delivered report, itself model-made (agreement, not truth).

    python scripts/mvp2_physical_eval.py OUT site=RUN_DIR:REPORT[:SHIFTED_RUN_DIR:SHIFTED_REPORT:OFFSET] ...
    python scripts/mvp2_physical_eval.py --self-check

Rows per call:
  tops / bases vs delivered on same-object pairs (fast_report_eval.match_same_object: plan-footprint IoU + agreeing names;
    heights never enter the match): signed median, |delta| median / p90, coverage (|delta| <= u without its scale part), for
    the card's value and, when the call carries the cards' diagnostics, for the points' p98 / p2 on the same objects;
  distance test (no reference needed): per object, a view's top minus the nearest view's top against the extra distance;
    a shaved rim reads lower the farther the view (about one pixel per 262 px of focal length: ~1 cm per extra metre);
  people: detections, feet heights (rays at the body's range vs the lowest pixels' own depth) where the whole body is in
    view, masks measured as pictures, tracks confirmed by motion, tracks named likely pictures;
  long objects: visible lengths, short sides not measurable; card contract violations (cards.contract);
  repeatability (warm vs shifted window): coverage at the calibrated k for height / extent / position.
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "scripts")]
import fast_report_eval as ev  # noqa: E402
from fast_report import cards as fc  # noqa: E402


def cards_layer(run_dir, report):
    """The newest object_cards version with its cards and, when present, its diagnostics blob."""
    vs = ev.patch_versions(run_dir, report, "object_cards")
    if not vs:
        return None, None
    p = vs[-1]
    data = ev.patch_data(run_dir, p, "cards")
    diag = json.loads(ev.blob_bytes(run_dir, p["blobs"]["diagnostics"]["sha256"])) if "diagnostics" in (p.get("blobs") or {}) else None
    return data, diag


def summary(xs):
    a = np.asarray([x for x in xs if x is not None], float)
    return {"n": int(a.size), **({"median": round(float(np.median(a)), 3), "p90": round(float(np.percentile(a, 90)), 3)} if a.size else {})}


def vs_delivered(ours, ref, align, boxes, wh, diag=None):
    """Same-object pairs: top / base signed and absolute deltas, coverage; with diag, the points' p98 / p2 on the same pairs."""
    pairs, n_ref, n_iou = ev.match_same_object(ours, ref, boxes, wh)
    out = {"pairs": len(pairs), "iou_pairs": n_iou, "delivered_clear": n_ref, "scale_ours_to_delivered": round(align[0], 4),
           "note": "heights above the floor: 'raw' compares each report in its own metres (both anchor the floor 1.6 m under the camera), "
                   "'path' multiplies ours by the camera-path Sim3 scale (fast_report_eval.physical_row's rule)"}
    for q, (est, s) in [(q, x) for q in ("top", "base") for x in (("card", 1.), ("card_path", align[0])) + ((("points", 1.), ("points_path", align[0])) if diag else ())]:
        if True:
            d, cov = [], []
            for o, e, _, _ in pairs:
                dv = ev.delivered_values(e, ref["mpn"])[q]
                if not o[q]:
                    continue
                v = o[q][0]
                if est.startswith("points"):
                    r = (diag.get("rim") or {}).get(o["id"])
                    if not r:
                        continue
                    v = r["top_p98"] if q == "top" else r["base_p2"]
                d.append(s * v - dv)
                if o[q][2] is not None:
                    cov.append(abs(s * v - dv) <= s * o[q][2])
            out[f"{q}_{est}"] = {"signed_median": round(float(np.median(d)), 3) if d else None, **{k: v for k, v in summary(np.abs(d)).items()},
                                 "coverage": round(float(np.mean(cov)), 3) if cov else None}
    out["pairs_list"] = [[o["id"], e["entityId"], round(i, 2), n] for o, e, i, n in pairs]
    return out


def distance_test(diag):
    """Per object with >= 3 usable views spanning >= 1.5 m of distance: (view top - nearest view top) / (extra distance), for the
    points' p98 and the edge estimate; medians over objects (m per m). A shaved rim gives a negative slope."""
    slopes = {"p98": [], "edge": []}
    for r in (diag or {}).get("rim", {}).values():
        v = [x for x in r.get("views") or [] if x[2] is not None]
        if len(v) < 3:
            continue
        v = sorted(v)
        if v[-1][0] - v[0][0] < 1.5:
            continue
        near = v[0]
        for key, i in (("p98", 1), ("edge", 2)):
            dd = np.array([x[0] - near[0] for x in v[1:]])
            dt = np.array([x[i] - near[i] for x in v[1:]])
            slopes[key].append(float(np.sum(dd * dt) / np.sum(dd * dd)))
    return {k: {"objects": len(x), "median_m_per_m": round(float(np.median(x)), 4) if x else None,
                "share_negative": round(float(np.mean(np.asarray(x) < 0)), 3) if x else None} for k, x in slopes.items()}


def people_rows(layers, cards):
    ppl = layers.get("people") or {}
    det = [q.get("foot_surface") or {} for t in ppl.get("tracks_full", []) for q in t["points"]]
    whole = [g for g in det if g.get("feet_visible")]
    person_cards = [c for c in cards if c.get("kind") == "person" and c["id"] != "person:untracked"]
    rej = ppl.get("rejected") or []
    return {"tracks": len(ppl.get("tracks_full", [])), "detections": len(det), "whole_body": len(whole),
            "feet_rays_m": summary([g["h_m"] for g in whole]), "feet_rays_signed_median": round(float(np.median([g["h_m"] for g in whole])), 3) if whole else None,
            "feet_pixels_m": summary([g.get("h_pixels_m") for g in whole]),
            "feet_u_median": round(float(np.median([g["u_m"] for g in whole])), 3) if whole else None,
            "feet_within_u_of_0": round(float(np.mean([abs(g["h_m"]) <= g["u_m"] for g in whole])), 3) if whole else None,
            "stature_m": summary([g.get("stature_m") for g in whole]),
            "rejected_masks": len(rej), "rejected_reasons": sorted({r["reason"].split(":")[-1].strip() for r in rej}),
            "confirmed_by_motion": sum(bool((c.get("identity") or {}).get("confirmed_by")) for c in person_cards),
            "likely_pictures": sum((c.get("identity") or {}).get("name", "").startswith("person?") for c in person_cards),
            "person_cards": len(person_cards), "gate_mps": ppl.get("association_gate_mps")}


def judgement_rows(run_dir, report, checks=("J1", "J3a", "J8")):
    """Verdict counts of the newest judgements version for the checks the physical fixes feed (J1 reads the top, J3a the feet)."""
    vs = ev.patch_versions(run_dir, report, "judgements")
    if not vs:
        return None
    rows = ev.patch_data(run_dir, vs[-1], "rows").get("rows") or []
    out = {}
    for r in rows:
        if r.get("check") in checks:
            out.setdefault(r["check"], {}).setdefault(r["verdict"], 0)
            out[r["check"]][r["verdict"]] += 1
    return {"version": vs[-1]["version"], **out}


def one_call(run_dir, report, site, shifted=None):
    layers = ev.load_layers(run_dir, report)
    data, diag = cards_layer(run_dir, report)
    layers["object_cards"] = data
    vs = ev.patch_versions(run_dir, report, "people")
    if vs:
        layers["people"]["tracks_full"] = ev.patch_data(run_dir, vs[-1])["tracks"]
    ref = ev.reference(site)
    _, _, align = ev.camera_rows(layers, ref)
    ours = ev.ours_objects(layers)
    cards = (data or {}).get("cards") or []
    objs = [c for c in cards if c.get("kind") == "object"]
    viol = [v for c in cards for v in fc.contract(c)]
    boxes = ev.image_boxes(layers.get("outlines"), (data or {}).get("aliases"))
    wh = (layers["video"]["width"], layers["video"]["height"])
    row = {"run": [str(run_dir), report], "vs_delivered": vs_delivered(ours, ref, align, boxes, wh, diag), "distance_test": distance_test(diag),
           "people": people_rows(layers, cards),
           "long_objects": {"visible_length": sum("visible_length" in c["physical"] for c in objs),
                            "short_side_not_measurable": sum(any((c["physical"].get(n) or {}).get("status") == "not measurable" and "long object" in
                                                                 (c["physical"].get(n) or {}).get("reason", "") for n in ("width", "depth")) for c in objs),
                            "objects": len(objs)},
           "contract": {"cards": len(cards), "violations": len(viol), "examples": viol[:10]}, "judgements": judgement_rows(run_dir, report),
           "tops_moved_by_edges": summary([r["top"] - r["top_p98"] for r in ((diag or {}).get("rim") or {}).values()]),
           "bases_moved_by_edges": summary([r["base_p2"] - r["base"] for r in ((diag or {}).get("rim") or {}).values()])}
    if shifted:
        rb, repb, off = shifted
        row["repeat_shifted"] = ev.repeat_row(layers, ev.load_layers(rb, repb), int(off))[0]
        row["repeat_shifted"].pop("shots", None)
    return row


def sheet(tiles, path, cols=6, h=320, w=220):
    """Contact sheet: tiles [(bgr crop, caption lines)] -> one JPEG."""
    import cv2
    cells = []
    for img, cap in tiles:
        c = np.full((h + 16 * len(cap), w, 3), 255, np.uint8)
        k = min(w / img.shape[1], h / img.shape[0])
        im = cv2.resize(img, (max(1, int(img.shape[1] * k)), max(1, int(img.shape[0] * k))))
        c[:im.shape[0], :im.shape[1]] = im
        for i, line in enumerate(cap):
            cv2.putText(c, line, (2, h + 12 + 16 * i), cv2.FONT_HERSHEY_SIMPLEX, .38, (0, 0, 0), 1, cv2.LINE_AA)
        cells.append(c)
    hh = max(c.shape[0] for c in cells)
    cells = [np.pad(c, ((0, hh - c.shape[0]), (0, 4), (0, 0)), constant_values=255) for c in cells]
    rows = [np.concatenate(cells[i:i + cols] + [np.full_like(cells[0], 255)] * (cols - len(cells[i:i + cols])), 1) for i in range(0, len(cells), cols)]
    # ponytail: one JPEG per sheet; split into pages if a sheet grows past ~40 tiles
    cv2.imwrite(str(path), np.concatenate(rows, 0), [cv2.IMWRITE_JPEG_QUALITY, 85])


def people_sheets(out, site, run_dir, report, n=40, seed=0):
    """Agent-label sheets for people: (a) detections whose true bottom is in view (feet on the floor? the new and old feet
    heights under each crop, a red line at the mask's bottom), (b) masks measured as pictures. Frames from the clip's
    source-full.mp4 (the warm call's window). Writes people-<site>.jpg / rejected-<site>.jpg and their tile lists."""
    import cv2
    vs = ev.patch_versions(run_dir, report, "people")
    ppl = ev.patch_data(run_dir, vs[-1])
    cap = cv2.VideoCapture(str(ev.PHASE2 / "data/clips" / ev.CLIPS[site] / "source-full.mp4"))
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def crop(frame, bbox):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, img = cap.read()
        if not ok:
            return None
        x0, y0, x1, y1 = bbox[0] * W, bbox[1] * H, bbox[2] * W, bbox[3] * H
        mx, my = max(.3 * (x1 - x0), 24), max(.15 * (y1 - y0), 24)
        a, b, c, d = int(max(0, x0 - mx)), int(max(0, y0 - my)), int(min(W, x1 + mx)), int(min(H, y1 + my))
        img = img.copy()
        cv2.line(img, (int(x0), int(y1)), (int(x1), int(y1)), (0, 0, 255), 1)
        return img[b:d, a:c]
    rng = np.random.default_rng(seed)
    dets = [(t["id"], q) for t in ppl["tracks"] for q in t["points"] if (q.get("foot_surface") or {}).get("feet_visible")]
    pick = [dets[i] for i in sorted(rng.choice(len(dets), min(n, len(dets)), replace=False))] if dets else []
    tiles, rows = [], []
    for i, (tid, q) in enumerate(pick):
        g = q["foot_surface"]
        img = crop(q["frame"], g["bbox"])
        if img is None:
            continue
        tiles.append((img, [f"#{i} {tid} t{q['t']:.1f}", f"feet {g['foot_h_m']:+.2f}+-{g['u_m']:.2f} old {g.get('h_pixels_m')}",
                            f"h {g['stature_m']:.2f} below {g.get('below_h_m')}"]))
        rows.append({"tile": i, "track": tid, "t": q["t"], "frame": q["frame"], "foot_h_m": g["foot_h_m"], "u_m": g["u_m"], "h_pixels_m": g.get("h_pixels_m"),
                     "feet_visible": g.get("feet_visible"), "stature_m": g.get("stature_m"), "label": None})
    if tiles:
        sheet(tiles, Path(out) / f"people-{site}.jpg")
    (Path(out) / f"people-{site}.json").write_text(json.dumps({"labels": "on floor | raised | feet hidden | not a person | unclear", "tiles": rows}, indent=1))
    rej = ppl.get("rejected") or []
    tiles, rows = [], []
    for i, r in enumerate(rej[:n]):
        g = r["geometry"]
        img = crop(r["frame"], g["bbox"])
        if img is None:
            continue
        tiles.append((img, [f"#{i} t{r['t']:.1f} s{r['score']:.2f}", f"h {g['stature_m']:.2f}+-{g['u_stature_m']:.2f}", f"foot {g['foot_h_m']:+.2f}"]))
        rows.append({"tile": i, "t": r["t"], "frame": r["frame"], "reason": r["reason"], "label": None})
    if tiles:
        sheet(tiles, Path(out) / f"rejected-{site}.jpg")
    (Path(out) / f"rejected-{site}.json").write_text(json.dumps({"labels": "picture | real person | unclear", "tiles": rows}, indent=1))
    return len(pick), len(rej)


TIMED = ("cameras", "people", "objects v1", "pick v1", "cards v1", "judgements v1", "cards v2", "judgements v2", "pick v2", "cards v3",
         "first SAM 3D model", "splat preview")
STAGES = ("lift", "people.shot0", "people.shot1", "cards.v1", "densify.lift")


def timing(bench_dir):
    """Per call of a bench run (its summary.json): layer times (s from the MP4 bytes in the container), per-GPU peaks, the
    stages this branch touched (s, peak GiB) and every >90% flag."""
    d = json.loads((Path(bench_dir) / "summary.json").read_text())
    out = {"boot_s": (d.get("boot") or {}).get("ready_s") or (d.get("boot") or {}).get("cold_start_s"), "usd_estimate_upper": d.get("usd_estimate_upper"),
           "calls": []}
    for c in d["calls"]:
        lay = (c.get("mvp_latency") or {}).get("layers") or {}
        st = c.get("stages") or {}
        out["calls"].append({"kind": c["kind"], "report": c["report"], "first_call_after_boot": c.get("first_call_after_boot"),
                             "layers": {k: (lay.get(k) or {}).get("written_s") for k in TIMED},
                             "gpu_peak_gib": [g.get("peak_gb") for g in c.get("gpu_peak") or []], "flags": c.get("flags"),
                             "stages": {k: {"s": (st.get(k) or {}).get("s"), "peak_gb": (st.get(k) or {}).get("peak_gb")} for k in STAGES if k in st}})
    return out


def main(argv):
    out = Path(argv[0])
    out.mkdir(parents=True, exist_ok=True)
    rows = {}
    for a in argv[1:]:
        site, rest = a.split("=", 1)
        parts = rest.split(":")
        shifted = (parts[2], parts[3], parts[4]) if len(parts) >= 5 else None
        rows[site] = one_call(parts[0], parts[1], site, shifted)
        print(site, json.dumps({k: v for k, v in rows[site].items() if k not in ("run",)}, default=str)[:3000], flush=True)
    (out / "physical.json").write_text(json.dumps(rows, indent=1, default=str))
    return rows


def self_check():
    diag = {"rim": {"a": {"views": [[2., 1.00, 1.05], [4., .98, 1.05], [6., .96, 1.05]], "top": 1.05, "top_p98": .98, "base": 0., "base_p2": .02},
                    "b": {"views": [[3., 1., 1.], [3.5, 1., 1.]]}}}
    t = distance_test(diag)
    assert t["p98"]["objects"] == 1 and abs(t["p98"]["median_m_per_m"] + .01) < 1e-9 and t["edge"]["median_m_per_m"] == 0.
    assert summary([1., None, 3.])["median"] == 2.
    print("mvp2_physical_eval self-check ok: distance slope, summaries")


if __name__ == "__main__":
    if sys.argv[1:] == ["--self-check"]:
        self_check()
    elif sys.argv[1:2] == ["--sheets"]:
        for a in sys.argv[3:]:
            site_, rest = a.split("=", 1)
            print(site_, people_sheets(sys.argv[2], site_, *rest.split(":")[:2]))
    else:
        main(sys.argv[1:])
