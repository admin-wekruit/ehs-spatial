"""r5b: the cards' timelines on a retail run (modal_apps/fast_report_app.py::accuracy with calls labelled on / off / planted):
every change claim with its evidence frames (a contact sheet per report: the claimed card's pick region on its before / after
keyframes, for the agent's labels), planted events found (the planted box's outline, drawn with the delivered DROID cameras
as scripts/x6_plant.py drew it, against the claimed card's pick region on the claim's evidence keyframe: IoU >= IOU_MIN, the
kind agrees, the planted time lies inside the claim's before-after span +- 1 s), recall / precision per kind, false claims,
and the cards' added seconds (timeline on vs off: Volume commit of cards v1 / v3, and the cards stages).

    python scripts/r5b_timeline_eval.py RUN_DIR --plants DIR --out OUT_DIR [--labels LABELS.json]
    python scripts/r5b_timeline_eval.py --self-check

LABELS.json (agent-labelled, by eye on the sheets): {"claims": {"<report>/<card>/<kind>": {"label": "true" | "false", "why": "..."}},
"real_events": [{"site", "kind", "t0", "t1", "what", "evidence_frames": [..]}]}.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "scripts"), str(REPO / "modal_apps")]
IOU_MIN, MATCH = .25, {"disappeared": ("disappeared", "moved"), "appeared": ("appeared", "moved"), "moved": ("moved",)}


def reports(run_dir):
    out = []
    for rj in sorted((Path(run_dir) / "reports").glob("*/run.json")):
        r = json.loads(rj.read_text())
        if r.get("error") or "call" not in r:
            continue
        out.append({"report": rj.parent.name, "site": r["call"]["site"], "label": r["call"]["options"].get("label"), "mp4": r["call"]["mp4"],
                    "first_call": bool(r.get("first_call")), "run": r})
    return out


def claims_of(cards):
    rows = []
    for c in cards:
        tl = (c.get("time") or {}).get("timeline")
        if c.get("kind") != "object" or not tl:
            continue
        for ch in tl["changes"]:
            rows.append({"card": c["id"], "shot": c["shot"], "name": (c.get("identity") or {}).get("name"), "views": (c.get("views") or {}).get("n"),
                         **{k: ch.get(k) for k in ("kind", "window", "t_before", "t_after", "before_key", "after_key", "new_place_key", "free_views",
                                                   "distance_m", "distance_u_m", "to", "from")}})
    return rows


def places(cards):
    """How often the place test ran and what it saw (X6 VERIFY: report the judgements, not only the claims): per state, over the
    windows before a card's first sighting ('before') and after it while unseen ('after')."""
    out = {"before": {}, "after": {}}
    for c in cards:
        tl = (c.get("time") or {}).get("timeline")
        if c.get("kind") != "object" or not tl:
            continue
        seen = False
        for w in tl["windows"]:
            seen = seen or w.get("seen", 0) > 0
            if "place" in w:
                k = out["after" if seen else "before"]
                k[w["place"]["state"]] = k.get(w["place"]["state"], 0) + 1
    return out


def card_mask(pick, key, ids):
    """The card's pick region (any of ids: the card and the objects merged into it) on source frame `key`, or None."""
    i = next((j for j, f in enumerate(pick.frames) if f["frame"] == key), None)
    if i is None:
        return None
    ent = pick.data["entities"]
    codes = [k for k, e in enumerate(ent) if e in ids]
    return np.isin(pick.map(i), codes) if codes else None


def box_mask(site, event, place, frame, shape):
    """The planted box's outline on source frame `frame` (x6_plant's cube and DROID camera), at the pick map's size."""
    import cv2
    import x6_plant as xp
    ref, d, clip, up, p0 = _plant_ctx(site)
    poses, mpn = d["poses_c2w"].astype(np.float64), ref["mpn"]
    if frame >= len(poses):
        return None
    Kr = 2 * d["keyframe_final_fullres_intrinsics"][0].astype(float)
    xy, z = xp.project(xp.cube(np.array(event["centre_native"][place]), up, np.array(event["side_dir"]), xp.SIDE / mpn), poses[frame], Kr)
    if (z <= 0).any():
        return None
    src = xp.to_source(xy)
    m = np.zeros(shape, np.uint8)
    cv2.fillConvexPoly(m, cv2.convexHull((src * np.array([shape[1] / 1280, shape[0] / 720])).astype(np.int32)), 1)
    return m > 0


_CTX = {}


def _plant_ctx(site):
    if site not in _CTX:
        import x6_plant as xp
        _CTX[site] = xp.load(site)
    return _CTX[site]


def iou(a, b):
    return float((a & b).sum() / max((a | b).sum(), 1)) if a is not None and b is not None else 0.


def planted(site, truth, claims, pick, aliases):
    """Each planted event: found by a claim (kind, time, image IoU on the claim's evidence keyframe), and whether the box got a
    card at all (a pick region with IoU >= IOU_MIN on a frame it is drawn on: a miss is then the change rule's, not detection's)."""
    rows, used = [], set()
    members = lambda cid: {cid} | {a for a, b in aliases.items() if b == cid}  # noqa: E731
    for e in truth["events"]:
        if not e.get("placed"):
            rows.append({"kind": e["kind"], "placed": False})
            continue
        best = None
        for i, c in enumerate(claims):
            if c["kind"] not in MATCH[e["kind"]] or c["t_before"] is None or not (c["t_before"] - 1 <= e["change_s"] <= (c["t_after"] or c["t_before"]) + 1):
                continue
            appeared_side = c["kind"] == "appeared" or c["kind"] == "moved" and c.get("from")
            key = c["after_key"] if appeared_side else c["before_key"]
            place = 1 if (e["kind"] == "moved" and appeared_side) else 0
            m = card_mask(pick, key, members(c["card"]))
            b = box_mask(site, e, place, key, m.shape) if m is not None else None
            v = iou(m, b)
            if v >= IOU_MIN and (best is None or v > best[1]):
                best = (i, round(v, 3))
        detected = []
        for f in range(e["change_frame"] - 60, e["change_frame"] + 60, 6):
            i = next((j for j, fr in enumerate(pick.frames) if fr["frame"] == f), None)
            if i is None:
                continue
            m = pick.map(i)
            b = box_mask(site, e, 0 if (f < e["change_frame"] or e["kind"] != "moved") else 1, f, m.shape)
            if b is None or (e["kind"] == "disappeared" and f >= e["change_frame"]) or (e["kind"] == "appeared" and f < e["change_frame"]):
                continue
            codes, n = np.unique(m[b], return_counts=True)
            for code, cnt in zip(codes, n):
                if code and cnt / max(b.sum(), 1) >= .3:
                    detected.append(pick.data["entities"][code])
        if best:
            used.add(best[0])
        rows.append({"kind": e["kind"], "placed": True, "change_s": e["change_s"], "found": best is not None, "claim": claims[best[0]] if best else None,
                     "iou": best[1] if best else None, "box_cards": sorted({x for x in detected if x and x.startswith("obj-")})})
    return rows, used


def sheet(mp4, claims, pick, aliases, out_path, width=1600):
    """Per claim a row: before / after (/ new place) keyframes, the card's pick region outlined (green before, red after)."""
    import cv2
    if not claims:
        return None
    keys = sorted({k for c in claims for k in (c["before_key"], c["after_key"], c.get("new_place_key")) if k is not None})
    cap, frames, f = cv2.VideoCapture(mp4), {}, 0
    while len(frames) < len(keys):
        ok, img = cap.read()
        if not ok:
            break
        if f in keys:
            frames[f] = img
        f += 1
    rows = []
    for c in claims:
        ids = {c["card"]} | {a for a, b in aliases.items() if b == c["card"]}
        tiles = []
        for k, col, lab in ((c["before_key"], (0, 255, 0), "before"), (c["after_key"], (0, 0, 255), "after"), (c.get("new_place_key"), (255, 160, 0), "new place")):
            if k is None or k not in frames:
                continue
            img = frames[k].copy()
            m = card_mask(pick, k, ids)
            if m is not None and m.any():
                cs, _ = cv2.findContours(cv2.resize(m.astype(np.uint8), (1280, 720), interpolation=cv2.INTER_NEAREST), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img, cs, -1, col, 4)
            cv2.putText(img, f"{c['card']} {c['name']} {c['kind']} {lab} key {k}", (12, 44), 0, 1.1, (0, 255, 255), 3)
            tiles.append(cv2.resize(img, (640, 360)))
        while len(tiles) < 3:
            tiles.append(np.zeros((360, 640, 3), np.uint8))
        rows.append(np.hstack(tiles))
    img = np.vstack(rows)
    if img.shape[1] > width:
        img = cv2.resize(img, (width, int(img.shape[0] * width / img.shape[1])))
    cv2.imwrite(str(out_path), img, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return str(out_path)


def timing(reps):
    """Per site: cards v1 / v3 (Volume commit, s from the MP4 bytes) and the cards stages, timeline on vs off (warm calls)."""
    out = {}
    for r in reps:
        if r["first_call"] or r["label"] not in ("on", "off"):
            continue
        m = {k: (v or {}).get("written_s") for k, v in (r["run"].get("milestones") or {}).items()}
        st = {}
        for s in r["run"].get("stages", []):
            if s["stage"] in ("cards.inputs", "cards.v1", "cards.v3"):
                st[s["stage"]] = round(s["end_s"] - s["start_s"], 3)
        out.setdefault(r["site"], {})[r["label"]] = {"cards_v1": m.get("cards_v1"), "cards_v3": m.get("cards_v3"), "objects_v3": m.get("objects_v3"),
                                                      "stages": st, "report": r["report"]}
    for site, x in out.items():
        if "on" in x and "off" in x:
            x["added_s"] = {k: round(x["on"][k] - x["off"][k], 2) for k in ("cards_v1", "cards_v3") if x["on"][k] is not None and x["off"][k] is not None}
            x["added_stage_s"] = {k: round(x["on"]["stages"].get(k, 0) - x["off"]["stages"].get(k, 0), 2) for k in ("cards.inputs", "cards.v1", "cards.v3")}
    return out


def precision(claims_rows, labels):
    """Per kind: claims, labelled true / false / unlabelled (agent labels; planted matches count as true)."""
    out = {}
    for c in claims_rows:
        k = out.setdefault(c["kind"], {"claims": 0, "true": 0, "false": 0, "unlabelled": 0})
        k["claims"] += 1
        lab = c.get("planted_match") and "true" or (labels.get(f"{c['report']}/{c['card']}/{c['kind']}") or {}).get("label")
        k[lab if lab in ("true", "false") else "unlabelled"] += 1
    return out


def self_check():
    cards = [{"id": "obj-0-1", "kind": "object", "shot": 0, "identity": {"name": "box"}, "views": {"n": 5},
              "time": {"timeline": {"changes": [{"kind": "disappeared", "window": 2, "t_before": 3., "t_after": 4., "before_key": 75, "after_key": 100}]}}},
             {"id": "obj-0-2", "kind": "object", "shot": 0, "time": {"timeline": {"changes": []}}}]
    cl = claims_of(cards)
    assert len(cl) == 1 and cl[0]["kind"] == "disappeared" and cl[0]["before_key"] == 75
    cl[0].update(report="r", planted_match=True)
    p = precision(cl + [{"kind": "appeared", "report": "r", "card": "obj-0-3"}], {"r/obj-0-3/appeared": {"label": "false"}})
    assert p == {"disappeared": {"claims": 1, "true": 1, "false": 0, "unlabelled": 0}, "appeared": {"claims": 1, "true": 0, "false": 1, "unlabelled": 0}}, p
    a, b = np.zeros((4, 4), bool), np.zeros((4, 4), bool)
    a[:2], b[1:3] = True, True
    assert abs(iou(a, b) - 1 / 3) < 1e-9 and iou(a, None) == 0.
    t = timing([{"first_call": False, "label": "on", "site": "s", "report": "a", "run": {"milestones": {"cards_v1": {"written_s": 30.}, "cards_v3": {"written_s": 70.}},
                                                                                         "stages": [{"stage": "cards.v1", "start_s": 20., "end_s": 28.}]}},
                {"first_call": False, "label": "off", "site": "s", "report": "b", "run": {"milestones": {"cards_v1": {"written_s": 29.}, "cards_v3": {"written_s": 68.}},
                                                                                          "stages": [{"stage": "cards.v1", "start_s": 20., "end_s": 27.}]}}])
    assert t["s"]["added_s"] == {"cards_v1": 1., "cards_v3": 2.} and t["s"]["added_stage_s"]["cards.v1"] == 1.
    print("r5b timeline eval self-check ok: claims from cards, precision with planted matches and labels, IoU, added seconds on vs off")


if __name__ == "__main__":
    if sys.argv[1:] == ["--self-check"]:
        self_check()
        sys.exit()
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--plants", type=Path, required=True, help="truth of calls labelled 'planted' (scripts/r5b_plant.py)")
    ap.add_argument("--plants-x6", type=Path, help="truth of calls labelled 'planted-x6' (X6's run-006 plants)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--labels", type=Path)
    a = ap.parse_args()
    import fast_report_eval as ev
    a.out.mkdir(parents=True, exist_ok=True)
    labels = json.loads(a.labels.read_text()) if a.labels else {}
    reps = reports(a.run)
    result = {"timing": timing(reps), "reports": [], "planted": []}
    all_claims = []
    for r in reps:
        if r["label"] not in ("on", "planted", "planted-x6"):
            continue
        L = ev.load_layers(a.run, r["report"])
        oc = L["object_cards"]
        cards = oc["cards"]
        pick = ev.run_picks(a.run, r["report"], L)[-1][1]
        aliases = oc.get("aliases") or {}
        cl = claims_of(cards)
        for c in cl:
            c.update(report=r["report"], site=r["site"], label=r["label"])
        stats = (oc.get("stats") or {}).get("timeline")
        rec = {"report": r["report"], "site": r["site"], "label": r["label"], "first_call": r["first_call"], "cards_version": oc.get("version"),
               "objects": sum(c.get("kind") == "object" for c in cards), "with_timeline": sum(bool((c.get("time") or {}).get("timeline")) for c in cards),
               "claims": len(cl), "by_kind": {k: sum(c["kind"] == k for c in cl) for k in ("appeared", "disappeared", "moved")}, "stats": stats,
               "places_judged": places(cards),
               "windows": ((r["run"].get("summary") or {}).get("timeline") or {}).get("shots")}
        if r["label"] in ("planted", "planted-x6"):
            truth = json.loads(((a.plants if r["label"] == "planted" else a.plants_x6) / f"planted-{r['site']}.json").read_text())
            rows, used = planted(r["site"], truth, cl, pick, aliases)
            for i in used:
                cl[i]["planted_match"] = True
            result["planted"].append({"report": r["report"], "site": r["site"], "label": r["label"], "events": rows})
        rec["sheet"] = sheet(r["mp4"], cl, pick, aliases, a.out / f"claims-{r['report']}.jpg") if cl else None
        result["reports"].append(rec)
        all_claims += cl
    result["claims"] = all_claims
    result["precision"] = precision([c for c in all_claims if not next(x for x in result["reports"] if x["report"] == c["report"])["first_call"]],
                                    labels.get("claims", {}))
    result["recall_planted"] = {}
    for lab in ("planted", "planted-x6"):
        ev_rows = [e for p in result["planted"] if p["label"] == lab for e in p["events"] if e.get("placed")]
        result["recall_planted"][lab] = {k: {"events": sum(e["kind"] == k for e in ev_rows), "found": sum(e["kind"] == k and e["found"] for e in ev_rows),
                                             "box_got_a_card": sum(e["kind"] == k and bool(e["box_cards"]) for e in ev_rows)} for k in ("disappeared", "appeared", "moved")}
    result["real_events"] = labels.get("real_events", [])
    (a.out / "timeline-eval.json").write_text(json.dumps(result, indent=1, default=float))
    print(json.dumps({k: result[k] for k in ("timing", "precision", "recall_planted")}, indent=1, default=float))
    for r in result["reports"]:
        print(r["report"], r["claims"], r["by_kind"], r["sheet"])
