"""r5b (models): the measurements of the models round, on a bench's mirror (scripts/fast_report_bench.py runs).

  python scripts/r5b_models.py sheets MIRROR REPORT OUT [--n 30 --strat 20 --seed 7]
      30 random + 20 stratified (by model tier) object cards of the report's final cards version: per card the best view with the
      object's outline | the SHOWN model (generated, checked primitive or observed surface) drawn from that camera | r4b's model
      (display_model.r4_record on the same fits: mostly a box); sheets OUT/sheet-*.jpg and OUT/labels.json (to fill by eye:
      right | partial | wrong, 'agent-labelled')
  python scripts/r5b_models.py tilted MIRROR REPORT OUT          # every planar part 15-75 deg to the floor, drawn on its card's view
  python scripts/r5b_models.py stats MIRROR REPORT               # tiers, routes, times (written_s: the Volume commit), GPU peaks
  python scripts/r5b_models.py --self-check

Labels by eye are the agent's ('agent-labelled'), never ground truth.
"""
import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "scripts")]
import r5b_render as rr  # noqa: E402

TIERS = ("generated", "primitive", "observed surface", "none")


# ---------------------------------------------------------------- a report's layers on a mirror
def patches(mirror, report):
    """{layer: [patches]} in version order."""
    out = {}
    for p in sorted((Path(mirror) / "reports" / report / "patches").glob("*.json")):
        d = json.loads(p.read_text())
        out.setdefault(d["layer"], []).append(d)
    for v in out.values():
        v.sort(key=lambda d: (d["version"], d["seq"]))
    return out


def blob(mirror, ref):
    path = Path(mirror) / "blobs" / "sha256" / ref["sha256"]
    if not path.exists():
        return None
    raw = path.read_bytes()
    return gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw


def final_cards(mirror, P):
    p = P["object_cards"][-1]
    cs = p["data"]["cards"]
    if cs == "blob":
        cs = json.loads(blob(mirror, p["blobs"]["cards"]))
    return cs, p["data"].get("aliases") or {}


def outlines(mirror, P):
    """{entity: {source frame: [polygons]}} of the last outlines version."""
    p = P["outlines"][-1]
    data = json.loads(blob(mirror, p["blobs"]["analysis"])) if (p.get("blobs") or {}).get("analysis") else p["data"]
    out = {}
    for f in data.get("frames", []):
        for o in f.get("objects", []):
            if o.get("polygons"):
                out.setdefault(o["entityId"], {}).setdefault(int(f["sourceFrame"]), []).extend(o["polygons"])
    return out


def cameras(P):
    return {s["index"]: s for s in P["cameras"][-1]["data"]["shots"]}


def video_path(mirror, P):
    return Path(mirror) / "blobs" / "sha256" / P["video"][-1]["blobs"]["video"]["sha256"]


def frames_at(mp4, frames):
    import cv2
    cap, out = cv2.VideoCapture(str(mp4)), {}
    for f in sorted(set(int(x) for x in frames)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if ok:
            out[f] = img
    cap.release()
    return out


# ---------------------------------------------------------------- the shown model and r4b's, as world triangles
def shown(card, mirror, P, cache):
    """-> (tier, triangles (n, 3, 3) in the shot frame, per-triangle seen flags or None)."""
    import r4_models
    from scipy.spatial.transform import Rotation
    models = (P.get("models") or [{}])[-1]
    rows = {r["object"]: r for r in (models.get("data") or {}).get("models", [])}
    r = rows.get(card["id"])
    if r is not None:
        rep = r.get("reuse_of") or card["id"]
        raw = blob(mirror, models["blobs"][f"model-{rep}"])
        if raw is not None:
            base = next(x for x in models["data"]["models"] if x["object"] == rep)["transform"]["position"]
            T, rgba = r4_models.glb_tris(raw, base)
            if r.get("reuse_of"):
                t = r["transform"]
                Rm = Rotation.from_quat(t["quaternion"]).as_matrix()
                T = ((T - base) * t["scale"][0]) @ Rm.T + t["position"]
            return "generated", T, rgba[:, 3] > 200
    m = card.get("model") or {}
    if m.get("tier") == "primitive":
        T, seen = r4_models.model_tris(m)
        return "primitive", T, seen
    sf = next((x for x in ((P.get("surfaces") or [{}])[-1].get("data") or {}).get("surfaces", []) if x["object"] == card["id"]), None)
    if sf and sf.get("blob"):
        key = (P["surfaces"][-1]["seq"], sf["blob"])
        if key not in cache:
            raw = blob(mirror, P["surfaces"][-1]["blobs"][sf["blob"]])
            cache[key] = rr.glb_nodes(raw) if raw is not None else {}
        got = cache[key].get(card["id"])
        if got is not None:
            V, F, _ = got
            return "observed surface", V[F], None
    ph = card.get("physical") or {}
    if ph.get("box_min_m") and ph.get("box_max_m"):  # the viewer's fallback: the see-through box (axis-aligned in the shot frame)
        from r4_models import FACE_AXIS, box_tris
        lo, hi = np.asarray(ph["box_min_m"], float), np.asarray(ph["box_max_m"], float)
        T, _ = box_tris(lo, hi, dict.fromkeys(FACE_AXIS, False))
        return "none", np.asarray(T, float).reshape(-1, 3, 3), None
    return "none", np.zeros((0, 3, 3)), None


def r4b(card):
    """r4b's display model on the card's own fits (its type's shape or the residual's: mostly a box)."""
    import r4_models
    from fast_report import cards, display_model
    idn = card.get("identity") or {}
    cls = idn.get("canonical")
    key = None if cls == cards.NOT_OBJECT else cls or idn.get("name")
    rec = display_model.r4_record(card.get("raw") or {}, key)
    if not rec or not rec.get("kind"):
        return None, np.zeros((0, 3, 3))
    T, _ = r4_models.model_tris(rec)
    return rec["kind"], T


def tier_of(card, P):
    models = (P.get("models") or [{}])[-1]
    if any(r["object"] == card["id"] for r in (models.get("data") or {}).get("models", [])):
        return "generated"
    t = (card.get("model") or {}).get("tier")
    return "primitive" if t == "primitive" else "observed surface" if t == 0 else "none"


# ---------------------------------------------------------------- sheets
def sample(cs, P, n=30, strat=20, seed=7):
    """n random object cards (with 3D points), then `strat` more spread over the tiers (the rarer tiers first), no repeats."""
    rng = np.random.default_rng(seed)
    obj = [c for c in cs if c.get("kind") == "object" and (c.get("raw") or {}).get("model_fits")]
    pick = [obj[i] for i in sorted(rng.choice(len(obj), min(n, len(obj)), replace=False))]
    rows = [(c, "random") for c in pick]
    left = [c for c in obj if c not in pick]
    by = {t: [c for c in left if tier_of(c, P) == t] for t in TIERS}
    order = sorted([t for t in TIERS if by[t]], key=lambda t: len(by[t]))
    k = 0
    while k < strat and any(by[t] for t in order):
        for t in order:
            if by[t] and k < strat:
                c = by[t].pop(int(rng.integers(len(by[t]))))
                rows.append((c, f"stratified:{t}"))
                k += 1
    return rows


def tile_row(card, how, mirror, P, cams, outl, aliases, imgs, cache, side=240):
    s = cams[card["shot"]]
    keys = s["keys"]
    best = [int(k) for k in (card.get("views") or {}).get("best", []) if int(k) in keys]
    ids = [card["id"], *((card.get("physical") or {}).get("merged_from") or [])] + [a for a, b in aliases.items() if b == card["id"]]
    polys = {}
    for i in ids:
        for q, p in outl.get(i, {}).items():
            polys.setdefault(q, []).extend(p)
    q = next((k for k in best if k in polys), None) or (max(polys, key=lambda k: len(polys[k])) if polys else (best or keys)[0])
    img = imgs[q]
    v = keys.index(q)
    c2w = np.asarray(s["c2w"][v], float)
    K = rr.k_full(np.asarray(s["K"][v], float), s["wh"], s["source_wh"])
    import cv2
    mask = np.zeros(img.shape[:2], np.uint8)
    for p in polys.get(q, []):
        cv2.fillPoly(mask, [np.round(np.asarray(p, float).reshape(-1, 2)).astype(np.int32)], 1)
    mask = mask > 0
    tier, T, _ = shown(card, mirror, P, cache)
    kind4, T4 = r4b(card)
    cov, _ = rr.raster(T, c2w, K, s["source_wh"]) if len(T) else (np.zeros(mask.shape, bool), None)
    cov4, _ = rr.raster(T4, c2w, K, s["source_wh"]) if len(T4) else (np.zeros(mask.shape, bool), None)
    box = rr.crop_box(mask if mask.any() else cov | cov4)
    name = str((card.get("identity") or {}).get("name"))[:22]
    a = rr.label(rr.cut(rr.outline(img, mask), box, side), f"{card['id']} {name}", y=14)
    a = rr.label(a, how[:24])
    b = rr.label(rr.cut(rr.outline(rr.overlay(img, cov, tint=(80, 220, 120)), mask), box, side), f"shown: {tier}"
                 + (f" ({(card.get('model') or {}).get('kind')})" if tier == "primitive" else ""))
    c = rr.label(rr.cut(rr.outline(rr.overlay(img, cov4), mask), box, side), f"r4b: {kind4}")
    return np.hstack([a, b, c]), {"card": card["id"], "name": (card.get("identity") or {}).get("name"), "sample": how, "tier": tier,
                                  "kind": (card.get("model") or {}).get("kind"), "r4b_kind": kind4, "view": q, "shown": None, "r4b": None, "note": ""}


def sheets(mirror, report, out, n=30, strat=20, seed=7, per=6, only=None):
    import cv2
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    P = patches(mirror, report)
    cs, aliases = final_cards(mirror, P)
    cams, outl = cameras(P), outlines(mirror, P)
    rows = sample(cs, P, n, strat, seed) if not only else \
        [(c, f"tier:{only}") for c in cs if c.get("kind") == "object" and tier_of(c, P) == only][:strat]
    need = set()
    for c, _ in rows:
        keys = cams[c["shot"]]["keys"]
        need |= {int(k) for k in (c.get("views") or {}).get("best", [])[:1]} | {int(keys[0])}
        need |= {k for i in [c["id"], *((c.get("physical") or {}).get("merged_from") or [])] for k in outl.get(i, {})}
    imgs = frames_at(video_path(mirror, P), need)
    tiles, labels, cache = [], [], {}
    for c, how in rows:
        try:
            t, lab = tile_row(c, how, mirror, P, cams, outl, aliases, imgs, cache)
        except Exception as e:  # noqa: BLE001  a card that cannot be drawn is listed with why
            labels.append({"card": c["id"], "sample": how, "error": repr(e)[:200]})
            continue
        lab["row"] = len(tiles) % per + 1
        lab["sheet"] = len(tiles) // per + 1
        tiles.append(t)
        labels.append(lab)
    for i in range(0, len(tiles), per):
        cv2.imwrite(str(out / f"sheet-{i // per + 1:02d}.jpg"), np.vstack(tiles[i:i + per]), [cv2.IMWRITE_JPEG_QUALITY, 82])
    doc = {"labeller": "agent-labelled", "report": report, "question": "does the model's outline match the object's outline in this view? "
           "right | partial | wrong (shown = the r5b model, r4b = r4b's model on the same fits)", "rows": labels}
    lp = out / "labels.json"
    if not lp.exists():
        lp.write_text(json.dumps(doc, indent=1))
    print(json.dumps({"cards": len(rows), "drawn": len(tiles), "sheets": -(-len(tiles) // per), "tiers": {t: sum(r.get("tier") == t for r in labels) for t in TIERS}}))


# ---------------------------------------------------------------- tilted parts, for a look
TILT_WORDS = ("guard", "ramp", "board", "panel", "shield", "cover", "door", "sign", "screen", "lid", "hood", "chute", "slope", "incline", "plate",
              "sheet", "tray", "monitor", "display")


def tilted(mirror, report, out, cap=48, side=300):
    """Every planar part the report reads 15-75 deg to the floor, and the parts of cards whose names suggest a slanted panel (guards,
    ramps, boards, panels, doors, signs ...), drawn as their plane's rectangle on the card's best view with the letter and angle: a
    sheet to say by eye whether the part is truly tilted and roughly at what angle (OUT/tilted-*.jpg, OUT/tilted-labels.json)."""
    import cv2
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    P = patches(mirror, report)
    cs, aliases = final_cards(mirror, P)
    by = {c["id"]: c for c in cs if c.get("kind") == "object"}
    rows = {r["object"]: r for r in ((P.get("surfaces") or [{}])[-1].get("data") or {}).get("surfaces", [])}
    shots = {s["index"]: s for s in P["object_cards"][-1]["data"]["shots"]}
    cams, outl = cameras(P), outlines(mirror, P)
    items = []
    for cid, r in rows.items():
        c = by.get(cid)
        sp = (c or {}).get("physical", {}).get("surface_parts") or r.get("parts") or {}
        if c is None or not sp.get("parts"):
            continue
        named = any(w in str((c.get("identity") or {}).get("name")).lower() for w in TILT_WORDS)
        for p in sp["parts"]:
            t = (p.get("tilt_deg") or {}).get("value")
            if t is not None and (15 <= t <= 75 or named):
                items.append((0 if 15 <= t <= 75 else 1, -p.get("share", 0), cid, p))
    items = sorted(items, key=lambda x: x[:2])[:cap]
    need = set()
    for _, _, cid, _ in items:
        need |= {int(k) for k in (by[cid].get("views") or {}).get("best", [])[:1]}
    imgs = frames_at(video_path(mirror, P), need)
    tiles, labels = [], []
    for kind, _, cid, p in items:
        c = by[cid]
        s = cams[c["shot"]]
        q = int((c.get("views") or {}).get("best", [s["keys"][0]])[0])
        if q not in imgs:
            continue
        v = s["keys"].index(q)
        ff = shots[c["shot"]]["floor_frame"]
        x, z = np.asarray(ff["x"], float), np.asarray(ff["z"], float)
        R, o = np.stack([x, np.cross(z, x), z]), np.asarray(ff["origin_m"], float)
        a1, a2 = np.asarray(p.get("axes") or [[1, 0, 0], [0, 1, 0]], float)
        s1, s2 = p["sides_m"]
        quad = np.array([np.asarray(p["centre_m"]) + i * a1 * s1 / 2 + j * a2 * s2 / 2 for i, j in ((-1, -1), (1, -1), (1, 1), (-1, 1))]) @ R + o
        K = rr.k_full(np.asarray(s["K"][v], float), s["wh"], s["source_wh"])
        c2w = np.asarray(s["c2w"][v], float)
        X = (quad - c2w[:3, 3]) @ c2w[:3, :3]
        img = imgs[q].copy()
        mask = np.zeros(img.shape[:2], np.uint8)
        for poly in [pp for i in [cid, *c["physical"].get("merged_from", [])] for pp in outl.get(i, {}).get(q, [])]:
            cv2.fillPoly(mask, [np.round(np.asarray(poly, float).reshape(-1, 2)).astype(np.int32)], 1)
        if (X[:, 2] > .05).all():
            uv = np.c_[K[0, 0] * X[:, 0] / X[:, 2] + K[0, 2], K[1, 1] * X[:, 1] / X[:, 2] + K[1, 2]]
            cv2.polylines(img, [np.round(uv).astype(np.int32)], True, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.polylines(img, [np.round(uv).astype(np.int32)], True, (0, 255, 255), 2, cv2.LINE_AA)
        box = rr.crop_box(mask > 0) if mask.any() else (0, 0, img.shape[1], img.shape[0])
        t = p["tilt_deg"]
        tile = rr.label(rr.cut(rr.outline(img, mask > 0), box, side), f"{cid} {str((c.get('identity') or {}).get('name'))[:20]}", y=14)
        tiles.append(rr.label(tile, f"{p.get('name')} {t['value']:.0f}+-{t['u']:.0f} deg to floor ({'15-75' if kind == 0 else 'by name'})"))
        labels.append({"card": cid, "name": (c.get("identity") or {}).get("name"), "part": p.get("name"), "tilt": t["value"], "u": t["u"],
                       "listed": "15-75" if kind == 0 else "by name", "sheet": (len(tiles) - 1) // 16 + 1, "cell": (len(tiles) - 1) % 16 + 1,
                       "truly_tilted": None, "eye_deg": None, "note": ""})
    for i in range(0, len(tiles), 16):
        chunk = tiles[i:i + 16]
        chunk += [np.full_like(tiles[0], 255)] * (-len(chunk) % 4)
        cv2.imwrite(str(out / f"tilted-{i // 16 + 1:02d}.jpg"), np.vstack([np.hstack(chunk[j:j + 4]) for j in range(0, len(chunk), 4)]),
                    [cv2.IMWRITE_JPEG_QUALITY, 82])
    lp = out / "tilted-labels.json"
    if not lp.exists():
        lp.write_text(json.dumps({"labeller": "agent-labelled", "report": report, "question": "is this part truly tilted (not level, not "
                                  "vertical)? if so, roughly how many degrees to the floor (by eye, +-10 deg at best)?", "rows": labels}, indent=1))
    print(json.dumps({"parts": len(labels), "sheets": -(-len(tiles) // 16)}))


# ---------------------------------------------------------------- tiers, routes, times and GPU peaks of one call
def stats(mirror, report, call=None):
    """One call's numbers: tier shares of the final object cards, primitive kinds, the router's record, the times each layer was
    WRITTEN (the Volume commit returned: run.json's written_s, from the MP4 bytes in the container) and the GPU peaks per stage."""
    P = patches(mirror, report)
    cs, _ = final_cards(mirror, P)
    obj = [c for c in cs if c.get("kind") == "object"]
    tiers = {t: sum(tier_of(c, P) == t for c in obj) for t in TIERS}
    kinds = {}
    for c in obj:
        if tier_of(c, P) == "primitive":
            kinds[c["model"]["kind"]] = kinds.get(c["model"]["kind"], 0) + 1
    md = ((P.get("models") or [{}])[-1].get("data") or {})
    models_rows = md.get("models") or []
    run = None
    if call and Path(call).exists():
        run = json.loads(Path(call).read_text()).get("run")
    elif (Path(mirror) / "reports" / report / "run.json").exists():
        run = json.loads((Path(mirror) / "reports" / report / "run.json").read_text())
    t = {}
    if run:
        rows = run.get("layers") or []
        seq_of = {d["seq"]: d for v in P.values() for d in v}

        def first(layer, pred=lambda d: True):
            got = [r for r in rows if r["layer"] == layer and r.get("written_s") is not None and pred(seq_of.get(r["seq"], {}))]
            return min((r["written_s"] for r in got), default=None)

        def last(layer, pred=lambda d: True):
            got = [r for r in rows if r["layer"] == layer and r.get("written_s") is not None and pred(seq_of.get(r["seq"], {}))]
            return max((r["written_s"] for r in got), default=None)
        t = {"cards_v1": first("object_cards"), "tier0_v1": first("surfaces", lambda d: (d.get("data") or {}).get("version_of_cards") == 1),
             "tier0_v3": first("surfaces", lambda d: (d.get("data") or {}).get("version_of_cards") == 3), "cards_final": last("object_cards"),
             "tier1_routes": first("models"), "first_generated": first("models", lambda d: bool((d.get("data") or {}).get("models"))),
             "tier1_done": first("models", lambda d: bool((d.get("data") or {}).get("final"))), "splat_preview": first("splat"),
             "call_end": run.get("elapsed_s")}
        t["tier0_after_cards_v1_s"] = round(t["tier0_v1"] - t["cards_v1"], 2) if t["tier0_v1"] and t["cards_v1"] else None
        marks = run.get("marks") or {}
        t["marks"] = {k: marks.get(k) for k in ("cards_v1_put", "tier0_v1_put", "tier1_started", "tier1_generating", "densify_sam3_done",
                                                  "cards_v3_put", "tier0_v3_put", "first_model_put", "models_final_put") if k in marks}
    gpu = None
    if run:
        over = [s for s in run.get("stages") or [] if any(run_over for run_over in (s.get("over_90") or []))]
        per_stage = {}
        for s in run.get("stages") or []:
            for g, v in enumerate(s.get("peak_gb") or []):
                if v is not None:
                    key = s["stage"].split("@")[0]
                    per_stage.setdefault(g, {}).setdefault(key, 0.)
                    per_stage[g][key] = max(per_stage[g][key], v)
        gpu = {"peak": [g["peak_gb"] for g in run.get("gpu_peak") or []], "flags": run.get("flags"),
               "stages_over_72": sorted({s["stage"] for s in over}),
               "top_stages": {g: sorted(v.items(), key=lambda x: -x[1])[:6] for g, v in per_stage.items()}}
    plan = (md.get("plan") or {})
    return {"report": report, "cards": len(obj), "tiers": tiers, "tier_share": {k: round(v / max(1, len(obj)), 3) for k, v in tiers.items()},
            "primitive_kinds": kinds, "generated_rows": len(models_rows), "reused_rows": sum(bool(r.get("reuse_of")) for r in models_rows),
            "plan": plan, "reuse": md.get("reuse"), "generator": md.get("generator"),
            "tried": {"accepted": sum(bool(r.get("accepted")) for r in md.get("tried") or []), "rows": len(md.get("tried") or [])},
            "times_written_s": t, "gpu": gpu, "usd_estimate": (run or {}).get("usd_estimate")}


# ---------------------------------------------------------------- planar-part angles against ground truth, every part counted
SURFACE_K, FIT_MAX_DEG = 1.75, 15.  # fast_report.surface's (records of the r5 bench carry k = 1: cal() applies the pipeline's k)


def cal(rec, k=SURFACE_K, recorded_k=1.):
    """A part angle record as the pipeline shows it: None when not measurable or curved (fit term > FIT_MAX_DEG), else u at k."""
    if not rec or "value" not in rec or (rec.get("parts") or {}).get("fit", 0) > FIT_MAX_DEG:
        return None
    return {**rec, "u": round(rec["u"] * k / recorded_k, 3)}


def match_all(ours, gt, size):
    """Every one of our parts and every GT part, paired by their centres only (Hungarian on centre distance / size, a pair only
    within 0.5 x size + 0.1 m: the same spatial gate as r5's, WITHOUT r5's 30 deg normal gate). -> [(i, j)], ours unmatched, gt unmatched."""
    from scipy.optimize import linear_sum_assignment
    if not ours or not gt:
        return [], list(range(len(ours))), list(range(len(gt)))
    D = np.array([[np.linalg.norm(np.asarray(a["c"]) - np.asarray(b["centre_m"])) for b in gt] for a in ours])
    cost = np.where(D <= .5 * size + .1, D / max(size, 1e-3), 1e6)
    r, c = linear_sum_assignment(cost)
    pairs = [(i, j) for i, j in zip(r, c) if cost[i, j] < 1e6]
    return pairs, sorted(set(range(len(ours))) - {i for i, _ in pairs}), sorted(set(range(len(gt))) - {j for _, j in pairs})


def gt_rows(dump_run, inputs, parts_of, recorded_k=1.):
    """Per GT report (sites 'gt-<seq>-w<i>') of dump_run: each card's GT points (accuracy_gt.regions: its pick regions lifted with GT depth
    and pose, GT floor frame) cut into planar parts the pipeline's way; our parts (parts_of(site, report) -> {card id: parts record}) put
    in the GT frame (the shots' floor yaw fit, the true camera height's scale), matched by centre only. -> rows (one per pair or miss)."""
    import accuracy_gt as ag
    import fast_report_eval as ev
    from fast_report import cards as fc
    from fast_report.core import CAMERA_HEIGHT_M
    from fast_report.surface import planar_parts, tilt_deg
    rows = []
    dump_run, inputs = Path(dump_run), Path(inputs)
    for rep in sorted((dump_run / "reports").iterdir()):
        run = json.loads((rep / "run.json").read_text())
        site = (run.get("call") or {}).get("site") or run.get("site") or ""
        if not site.startswith("gt-"):
            continue
        ours_by = parts_of(site, rep.name)
        if not ours_by:
            continue
        name, wi = site.split("-")[1], int(site.rsplit("-w", 1)[1])
        seq = ag.Seq(inputs / name)
        w = seq.g["windows"][wi]
        a0, hold = w["source_frames"][0], w["hold"]
        src = lambda f: a0 + int(f) // hold  # noqa: E731
        L = ev.load_layers(dump_run, rep.name)
        pick = ev.run_picks(dump_run, rep.name, L)[-1][1]
        oc = L["object_cards"]
        alias = oc.get("aliases") or {}
        merged = {}
        for e, r in ag.regions(pick, seq, src).items():
            cid = alias.get(e, e)
            while cid in alias and alias[cid] != cid:
                cid = alias[cid]
            merged.setdefault(cid, []).extend(r["P"])
        shots = {}
        for s in L["cameras"]["shots"]:
            keys, C = s["keys"], np.asarray(s["c2w"], float)[:, :3, 3]
            G = [seq.c2w(src(f)) for f in keys]
            ok = np.array([g is not None for g in G])
            Gc = np.array([g[:3, 3] for g in G if g is not None])
            fr = seq.floor_frame(src(keys[0]))
            ff = next((x["floor_frame"] for x in oc["shots"] if x["index"] == s["index"]), None)
            h_true = float(np.median((Gc - np.asarray(seq.floor["point"])) @ np.asarray(seq.floor["normal"])))
            mine = ag.to_card_frame(C[ok], ff)
            R, t = ag.rigid2(h_true / CAMERA_HEIGHT_M * mine[:, :2], fc.to_floor(Gc, fr)[:, :2])
            shots[s["index"]] = {"fr": fr, "R": R, "t": t, "s": h_true / CAMERA_HEIGHT_M, "cams": fc.to_floor(Gc, fr)}
        for c in oc["cards"]:
            sp = ours_by.get(c["id"]) or {}
            if not merged.get(c["id"]) or c["shot"] not in shots or not sp.get("parts"):
                continue
            sh = shots[c["shot"]]
            Q = fc.to_floor(np.concatenate(merged[c["id"]]), sh["fr"])
            Q = Q[fc.main_cluster(Q, fc.EPS_MIN)]
            if len(Q) < 200:
                continue
            g = planar_parts(Q, np.zeros(len(Q), int), [], np.array([np.median(sh["cams"], 0)]))
            gparts = [q for q in (g.get("parts") or []) if "value" in q["tilt_deg"]]
            R3 = np.eye(3)
            R3[:2, :2] = sh["R"]
            ours = []
            for i, p in enumerate(sp["parts"]):
                t = cal(p["tilt_deg"], recorded_k=recorded_k)
                if t is None:
                    continue
                n = R3 @ np.asarray(p["normal"])
                ours.append({"i": i, "t": t, "n": n, "c": np.r_[sh["R"] @ (sh["s"] * np.asarray(p["centre_m"][:2])) + sh["t"], sh["s"] * p["centre_m"][2]],
                             "sub": t.get("n_subsets") or 0, "share": p.get("share"), "area": p.get("area_m2")})
            if not ours and not gparts:
                continue
            size = max([max(q["sides_m"]) for q in g.get("parts") or []] + [.3])
            pairs, lone_o, lone_g = match_all(ours, gparts, size)
            base = {"seq": name, "window": wi, "card": c["id"], "name": (c.get("identity") or {}).get("name")}
            for i, j in pairs:
                o, q = ours[i], gparts[j]
                v, u, gv = o["t"]["value"], o["t"]["u"], q["tilt_deg"]["value"]
                rows.append({**base, "kind": "pair", "tilt": v, "u": u, "gt": gv, "err": abs(v - gv), "covered": abs(v - gv) <= u, "subsets": o["sub"],
                             "share": o["share"], "area_m2": o["area"],
                             "normal_deg": round(float(np.degrees(np.arccos(np.clip(abs(o["n"] @ np.asarray(q["normal"])), 0, 1)))), 2)})
            rows += [{**base, "kind": "ours unmatched", "tilt": ours[i]["t"]["value"], "u": ours[i]["t"]["u"], "covered": False, "subsets": ours[i]["sub"],
                      "share": ours[i]["share"], "area_m2": ours[i]["area"], "gt_parts_on_card": len(gparts)} for i in lone_o]
            rows += [{**base, "kind": "gt unmatched", "gt": gparts[j]["tilt_deg"]["value"], "covered": False} for j in lone_g]
    return rows


def gt_table(rows):
    def med(v, q=50):
        return round(float(np.percentile(v, q)), 2) if len(v) else None
    out = {}
    for seq in sorted({r["seq"] for r in rows}) + ["all"]:
        rs = [r for r in rows if seq in ("all", r["seq"])]
        pr = [r for r in rs if r["kind"] == "pair"]
        ours = [r for r in rs if r["kind"] in ("pair", "ours unmatched")]
        gts = [r for r in rs if r["kind"] in ("pair", "gt unmatched")]
        out[seq] = {"our_parts": len(ours), "gt_parts": len(gts), "pairs": len(pr), "ours_unmatched": len(ours) - len(pr), "gt_unmatched": len(gts) - len(pr),
                    "err_median_deg": med([r["err"] for r in pr]), "err_p90_deg": med([r["err"] for r in pr], 90), "u_median_deg": med([r["u"] for r in pr]),
                    "coverage_pairs": round(float(np.mean([r["covered"] for r in pr])), 3) if pr else None,
                    "coverage_all_ours": round(float(np.mean([r["covered"] for r in ours])), 3) if ours else None,
                    "pairs_over_30deg_normal": sum(r["normal_deg"] > 30 for r in pr), "gt_tilted_15_75": sum(15 <= r["gt"] <= 75 for r in gts),
                    "err_on_gt_tilted_15_75": med([r["err"] for r in pr if 15 <= r["gt"] <= 75])}
    return out


# ---------------------------------------------------------------- every card's outline agreement, measured (the eye labels' proxy)
def ious(mirror, report, out=None, limit=None):
    """Every object card at its best view: the shown model's and r4b's silhouettes against the object's outline there (the outlines
    layer's polygons: SAM 3's masks) -> per card IoU, coverage (outline share inside the silhouette), overflow (silhouette share
    outside the outline); medians by tier. At 1/2 resolution (640 x 360)."""
    import cv2
    P = patches(mirror, report)
    cs, aliases = final_cards(mirror, P)
    cams, outl = cameras(P), outlines(mirror, P)
    obj = [c for c in cs if c.get("kind") == "object" and (c.get("raw") or {}).get("model_fits")][:limit]
    need = {}
    for c in obj:
        keys = cams[c["shot"]]["keys"]
        ids = [c["id"], *((c.get("physical") or {}).get("merged_from") or [])] + [a for a, b in aliases.items() if b == c["id"]]
        polys = {}
        for i in ids:
            for q, p in outl.get(i, {}).items():
                polys.setdefault(q, []).extend(p)
        best = [int(k) for k in (c.get("views") or {}).get("best", []) if int(k) in keys and int(k) in polys]
        q = best[0] if best else (max(polys, key=lambda k: len(polys[k])) if polys else None)
        if q is not None:
            need[c["id"]] = (q, polys[q])
    rows, cache = [], {}
    for c in obj:
        if c["id"] not in need:
            continue
        q, polys = need[c["id"]]
        s = cams[c["shot"]]
        v = s["keys"].index(q)
        W, H = s["source_wh"][0] // 2, s["source_wh"][1] // 2
        K = rr.k_full(np.asarray(s["K"][v], float), s["wh"], (W, H))
        c2w = np.asarray(s["c2w"][v], float)
        mask = np.zeros((H, W), np.uint8)
        for p in polys:
            cv2.fillPoly(mask, [np.round(np.asarray(p, float).reshape(-1, 2) / 2).astype(np.int32)], 1)
        mask = mask > 0
        tier, T, _ = shown(c, mirror, P, cache)
        _, T4 = r4b(c)
        row = {"card": c["id"], "tier": tier, "kind": (c.get("model") or {}).get("kind"), "mask_px": int(mask.sum())}
        for tag, TT in (("shown", T), ("r4b", T4)):
            cov = rr.raster(TT, c2w, K, (W, H))[0] if len(TT) else np.zeros((H, W), bool)
            inter, uni = (cov & mask).sum(), (cov | mask).sum()
            row[tag] = {"iou": round(inter / max(uni, 1), 3), "coverage": round(inter / max(mask.sum(), 1), 3), "overflow": round((cov & ~mask).sum() / max(cov.sum(), 1), 3)}
        rows.append(row)
    def med(rs, tag, k):
        return round(float(np.median([r[tag][k] for r in rs])), 3) if rs else None
    summary = {}
    for t in ("all",) + TIERS:
        rs = [r for r in rows if t == "all" or r["tier"] == t]
        if rs:
            summary[t] = {"n": len(rs), **{f"{tag}_{k}": med(rs, tag, k) for tag in ("shown", "r4b") for k in ("iou", "coverage", "overflow")},
                          "shown_iou_ge_0.7": round(float(np.mean([r["shown"]["iou"] >= .7 for r in rs])), 3),
                          "r4b_iou_ge_0.7": round(float(np.mean([r["r4b"]["iou"] >= .7 for r in rs])), 3),
                          "shown_iou_lt_0.3": round(float(np.mean([r["shown"]["iou"] < .3 for r in rs])), 3),
                          "r4b_iou_lt_0.3": round(float(np.mean([r["r4b"]["iou"] < .3 for r in rs])), 3)}
    if out:
        Path(out).write_text(json.dumps({"report": report, "summary": summary, "rows": rows}, indent=1))
    return summary


# ---------------------------------------------------------------- the eye labels, counted
def label_counts(path, key):
    """labels.json rows -> {right, partial, wrong, unclear, n} of one column ('shown' or 'r4b'), over all rows and per sample kind."""
    rows = [r for r in json.loads(Path(path).read_text())["rows"] if r.get(key)]
    out = {}
    for name, rs in (("all", rows), ("random", [r for r in rows if r["sample"] == "random"]),
                     ("stratified", [r for r in rows if r["sample"].startswith("stratified")])):
        out[name] = {v: sum(r[key] == v for r in rs) for v in ("right", "partial", "wrong", "unclear")} | {"n": len(rs)}
    out["by_tier"] = {t: {v: sum(r[key] == v for r in rows if r.get("tier") == t) for v in ("right", "partial", "wrong")}
                      for t in sorted({r.get("tier") for r in rows if r.get("tier")})}
    return out


def tilted_counts(path):
    rows = [r for r in json.loads(Path(path).read_text())["rows"] if r.get("truly_tilted") is not None]
    tilt = [r for r in rows if r["truly_tilted"] is True and r.get("eye_deg") is not None]
    err = [abs(r["tilt"] - r["eye_deg"]) for r in tilt]
    return {"looked_at": len(rows), "truly_tilted": sum(r["truly_tilted"] is True for r in rows), "listed_15_75_not_tilted":
            sum(r["truly_tilted"] is False and r["listed"] == "15-75" for r in rows),
            "err_vs_eye_median_deg": round(float(np.median(err)), 1) if err else None, "err_vs_eye_max_deg": round(float(np.max(err)), 1) if err else None,
            "within_u_plus_10_of_eye": sum(abs(r["tilt"] - r["eye_deg"]) <= r["u"] + 10 for r in tilt), "n_eye": len(tilt)}


def self_check():
    cs = [{"id": f"o{i}", "kind": "object", "raw": {"model_fits": {"box": {}}}, "model": {"tier": 0 if i % 3 else "primitive"}} for i in range(40)]
    P = {"models": [{"data": {"models": [{"object": "o4"}]}}]}
    rows = sample(cs, P, 10, 6, 1)
    assert len(rows) == 16 and len({c["id"] for c, _ in rows}) == 16
    assert any(h == "stratified:generated" for c, h in rows) or any(c["id"] == "o4" for c, h in rows[:10])
    assert tier_of(cs[4], P) == "generated" and tier_of(cs[3], P) == "primitive" and tier_of(cs[1], P) == "observed surface"
    pairs, lo, lg = match_all([{"c": [0, 0, 0]}, {"c": [5, 5, 5]}], [{"centre_m": [.05, 0, 0]}, {"centre_m": [2, 0, 0]}], .5)
    assert pairs == [(0, 0)] and lo == [1] and lg == [1], (pairs, lo, lg)
    t = gt_table([{"seq": "tum", "kind": "pair", "err": 1., "u": 2., "gt": 90., "covered": True, "normal_deg": 3.},
                  {"seq": "tum", "kind": "ours unmatched", "u": 2., "covered": False}, {"seq": "tum", "kind": "gt unmatched", "gt": 30., "covered": False}])
    assert t["tum"]["our_parts"] == 2 and t["tum"]["gt_parts"] == 2 and t["tum"]["coverage_all_ours"] == .5 and t["all"]["gt_tilted_15_75"] == 1
    print("r5b_models self-check ok: sampling, tiers, every-part matching, GT table")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", nargs="?")
    ap.add_argument("args", nargs="*")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--strat", type=int, default=20)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--only", help="sheets: only the cards of this tier (e.g. generated), up to --strat")
    ap.add_argument("--self-check", action="store_true")
    a = ap.parse_args()
    if a.self_check:
        return self_check()
    if a.cmd == "sheets":
        sheets(*a.args, n=a.n, strat=a.strat, seed=a.seed, only=a.only)
    elif a.cmd == "ious":  # MIRROR REPORT [OUT]
        print(json.dumps(ious(*a.args), indent=1))
    elif a.cmd == "tilted":  # MIRROR REPORT OUT
        tilted(*a.args)
    elif a.cmd == "stats":  # MIRROR REPORT [CALL_JSON]
        print(json.dumps(stats(*a.args), indent=1, default=str))
    elif a.cmd == "gt-r5":  # BENCH DUMP_RUN INPUTS OUT: r5-models-bench-001's parts (k = 1 records) re-scored with every part
        bench, dump, inputs, out = a.args

        def parts_of(site, _report):
            p = Path(bench) / f"{site}-cpu.json"
            return {c["id"]: (c.get("surface") or {}).get("parts") or {} for c in json.loads(p.read_text())["cards"]} if p.exists() else {}
        rows = gt_rows(dump, inputs, parts_of, recorded_k=1.)
        Path(out).write_text(json.dumps({"source": f"{bench} (our parts) on {dump} (GT reports)", "rows": rows, "table": gt_table(rows)}, indent=1))
        print(json.dumps(gt_table(rows), indent=1))
    elif a.cmd == "gt":  # RUN INPUTS OUT: a GT run of this round (its surfaces layer's parts, the pipeline's k)
        run, inputs, out = a.args

        def parts_of(_site, report):
            P = patches(run, report)
            return {r["object"]: r.get("parts") or {} for r in (P.get("surfaces") or [{}])[-1].get("data", {}).get("surfaces", [])}
        rows = gt_rows(run, inputs, parts_of, recorded_k=SURFACE_K)
        Path(out).write_text(json.dumps({"source": run, "rows": rows, "table": gt_table(rows)}, indent=1))
        print(json.dumps(gt_table(rows), indent=1))


if __name__ == "__main__":
    main()
