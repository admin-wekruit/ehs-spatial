"""X8: labelled decision sets for the calibrated-decider test (Jev-Omni vs Qwen3-VL vs SigLIP 2), CPU only.

  python scripts/x8_sets.py build OUT_DIR      # sets a (same object?), b (which label?) from the fast core + delivered maps
  python scripts/x8_sets.py tiles OUT_DIR      # set c tiles cut from X2's contact sheets (labels: sets/c-labels.json, by eye)
  python scripts/x8_sets.py frames OUT_DIR     # set d keyframes/crops (labels: sets/d-labels.json, by eye)
  python scripts/x8_sets.py --self-check

Sources (read only): the fast core's objects + outlines (fb-integrate-* reports, the fb/a-core code), the delivered object
maps / names / masks (tests/fixtures/delivered-303 via fast_report_eval.reference), the 30 s clips, X2's contact sheets.
Truth for a and b is the delivered map ('reference'); c and d are 'agent-labelled' (by looking), saved as contact sheets.
Every image written is a small JPEG; nothing heavy is copied.
"""
import json
import random
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "scripts"), str(REPO / "modal_apps"), str(REPO)]
PHASE2 = Path("/Users/adam/Desktop/panoptes-public/research-notes/phase2")
RUNS = PHASE2 / "runs"
SITES = {  # the fast core's latest three-video reports (fb/integrate = fb/a-core + store/viewer/harness)
    "me340": (RUNS / "fb-integrate-me340-006", "fb-me340-e84efffd-1790649860", "me340-165"),
    "samsclub-a2": (RUNS / "fb-integrate-samsclub-002", "fb-samsclub-a2-d5e0c855-1790650595", "samsclub-337"),
    "walmart": (RUNS / "fb-integrate-walmart-001", "fb-walmart-c0761a2a-1790651216", "walmart-190"),
}
IOU, MIN_GAP_S, NEG_MIN_M = .5, 1., .5  # match to a delivered mask; positives >= 1 s apart; look-alikes >= 0.5 m apart
PER_SITE = {"pos": 50, "hard": 32, "easy": 18}  # 100 pairs a site, half positive


def norm_name(s):  # fast_report_app.norm_name, verbatim
    words = [w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w for w in (s or "").lower().replace("-", " ").split()]
    return " ".join(words)


def name_match(ours, theirs):
    """fast_report_app.name_match (spec section 10): same head noun, or one name inside the other as whole words."""
    a, b = norm_name(ours), norm_name(theirs)
    if not a or not b:
        return False
    return a.split()[-1] == b.split()[-1] or f" {a} " in f" {b} " or f" {b} " in f" {a} "


def tum(poly):
    """1280x720 source pixels -> the delivered 640x480 raster (x 160-1120, x 2/3)."""
    p = np.asarray(poly, np.float64)
    return np.stack([(p[:, 0] - 160) * 640 / 960, p[:, 1] * 480 / 720], 1)


def fill(polys, size, fn=None):
    import cv2
    m = np.zeros(size, np.uint8)
    for p in polys:
        p = fn(p) if fn else np.asarray(p, np.float64)
        if len(p) >= 3:
            cv2.fillPoly(m, [np.round(p).astype(np.int32)], 1)
    return m.astype(bool)


def crop_pair(bgr, mask, side=448):
    """The core's VLM crop (discover.crop_jpeg: 1.5 x box, red outline, long side 448) and its SigLIP input (cascade.Embedder:
    square box + 10 %, mid grey outside the mask, 224) -> (jpeg, jpeg)."""
    import cv2
    H, W = mask.shape
    ys, xs = np.nonzero(mask)
    cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2
    half = max(ys.max() - ys.min(), xs.max() - xs.min(), 96) * .75
    y0, y1, x0, x1 = int(max(0, cy - half)), int(min(H, cy + half)), int(max(0, cx - half)), int(min(W, cx + half))
    img = bgr.copy()
    cv2.drawContours(img, cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], -1, (0, 0, 255), 2)
    crop = img[y0:y1, x0:x1]
    sc = side / max(crop.shape[:2])
    crop = cv2.resize(crop, (max(1, int(crop.shape[1] * sc)), max(1, int(crop.shape[0] * sc))), interpolation=cv2.INTER_AREA if sc < 1 else cv2.INTER_CUBIC)
    half2 = max(ys.max() - ys.min() + 1, xs.max() - xs.min() + 1) * 1.2 / 2
    pad = int(half2) + 2
    big = cv2.copyMakeBorder(np.where(mask[..., None], bgr, 128).astype(np.uint8), pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(128, 128, 128))
    cyi, cxi, h2 = int(round(cy)) + pad, int(round(cx)) + pad, int(round(half2))
    sq = cv2.resize(big[cyi - h2:cyi + h2, cxi - h2:cxi + h2], (224, 224), interpolation=cv2.INTER_AREA)
    enc = lambda a, q: cv2.imencode(".jpg", a, [cv2.IMWRITE_JPEG_QUALITY, q])[1].tobytes()  # noqa: E731
    return enc(crop, 88), enc(sq, 95)


def decode(mp4, frames):
    import cv2
    cap, out, i, want = cv2.VideoCapture(str(mp4)), {}, 0, set(frames)
    while len(out) < len(want):
        ok, im = cap.read()
        if not ok:
            break
        if i in want:
            out[i] = im
        i += 1
    return out


def observations(site):
    """The core's segmented outlines matched to the delivered masks: [{oid, frame, ent, iou, polys}] (IoU >= 0.5, the
    delivered frame <= 1 frame away), plus the core objects and the delivered entities."""
    import cv2
    import fast_report_eval as fe
    run_dir, report, clip = SITES[site]
    lay = fe.load_layers(run_dir, report)
    ref = fe.reference(site)
    names = json.loads(ref["names"].read_text())
    omap = json.loads(ref["object_map"].read_text())
    ents = {e["entityId"]: e for e in omap["entities"] if e["entityId"].startswith("object-")}
    root = ref["mask_root"]
    have = {int(p.name.split("-")[1]) for p in root.iterdir() if p.name.startswith("frame-")}
    by_frame = {}
    for eid, e in ents.items():
        for o in e["observations"]:
            _, f, inst = o.split(":")
            by_frame.setdefault(int(f), []).append((eid, int(inst)))
    obs = []
    for fr in lay["outlines"]["frames"]:
        if fr["source"] != "segmented":
            continue
        f = fr["sourceFrame"]
        g = min(have, key=lambda x: abs(x - f))
        if abs(g - f) > 1 or g not in by_frame:
            continue
        theirs = [(eid, cv2.imread(str(root / f"frame-{g:05d}" / f"instance-{i}-mask.png"), cv2.IMREAD_GRAYSCALE)) for eid, i in by_frame[g]]
        theirs = [(eid, m > 0) for eid, m in theirs if m is not None and (m > 0).sum() >= 300]
        for o in fr["objects"]:
            m = fill(o["polygons"], (480, 640), tum)
            if m.sum() < 300:
                continue
            best = max(((float((m & t).sum() / max((m | t).sum(), 1)), eid) for eid, t in theirs), default=(0., None))
            if best[0] >= IOU:
                obs.append({"site": site, "oid": o["entityId"], "frame": f, "ent": best[1], "iou": round(best[0], 3), "polys": o["polygons"]})
    objs = {o["id"]: o for o in lay["objects"]["objects"]}
    vocab = lay["objects"].get("vocabulary") or lay["objects"].get("words") or []
    return obs, objs, ents, names, ref, vocab, lay


def centroid_m(e, mpn):
    return np.asarray(e["centroidNative"], float) * mpn


def category(names, eid):
    n = names.get(eid)
    return n["category"] if n and n["status"] in ("clear", "partial") else None


def pick_pairs(obs, ents, names, mpn, fps, rng):
    """Positives: two observations of one delivered entity >= 1 s apart (half across two core objects = a merge the core
    missed, when there are enough). Hard negatives: two different delivered entities with matching names, >= 0.5 m apart.
    Easy negatives: different entities whose names do not match. -> [(kind, obs_i, obs_j)]"""
    by_ent = {}
    for i, o in enumerate(obs):
        by_ent.setdefault(o["ent"], []).append(i)
    pos_same, pos_cross = [], []
    for eid, idx in by_ent.items():
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                i, j = idx[a], idx[b]
                if abs(obs[i]["frame"] - obs[j]["frame"]) >= MIN_GAP_S * fps:
                    (pos_same if obs[i]["oid"] == obs[j]["oid"] else pos_cross).append((i, j))
    named = [i for i, o in enumerate(obs) if category(names, o["ent"])]
    hard, easy = [], []
    for a in range(len(named)):
        for b in range(a + 1, len(named)):
            i, j = named[a], named[b]
            ei, ej = obs[i]["ent"], obs[j]["ent"]
            if ei == ej:
                continue
            if name_match(category(names, ei), category(names, ej)):
                if np.linalg.norm(centroid_m(ents[ei], mpn) - centroid_m(ents[ej], mpn)) >= NEG_MIN_M:
                    hard.append((i, j))
            else:
                easy.append((i, j))

    def spread(cands, n, key):
        """n pairs, no entity (or entity pair) used more than needed: round-robin over the key's groups."""
        groups = {}
        for c in cands:
            groups.setdefault(key(c), []).append(c)
        for g in groups.values():
            rng.shuffle(g)
        keys = sorted(groups)
        rng.shuffle(keys)
        out = []
        while len(out) < n and any(groups[k] for k in keys):
            for k in keys:
                if groups[k] and len(out) < n:
                    out.append(groups[k].pop())
        return out
    n_pos = PER_SITE["pos"]
    cross = spread(pos_cross, n_pos // 2, lambda c: obs[c[0]]["ent"])
    same = spread(pos_same, n_pos - len(cross), lambda c: obs[c[0]]["ent"])
    hard_p = spread(hard, PER_SITE["hard"], lambda c: tuple(sorted((obs[c[0]]["ent"], obs[c[1]]["ent"]))))
    easy_p = spread(easy, PER_SITE["easy"] + PER_SITE["hard"] - len(hard_p), lambda c: obs[c[0]]["ent"])
    return ([("pos-cross-object", *c) for c in cross] + [("pos-same-object", *c) for c in same] +
            [("neg-lookalike", *c) for c in hard_p] + [("neg-other-class", *c) for c in easy_p])


def sheet(items, path, cols=6, tile=220):
    """[(jpeg bytes, caption)] -> contact sheet JPEG (discover.contact_sheet's layout)."""
    import cv2
    ims = []
    for jpg, cap in items:
        img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
        s = tile / max(img.shape[:2])
        ims.append((cv2.resize(img, (max(1, int(img.shape[1] * s)), max(1, int(img.shape[0] * s))), interpolation=cv2.INTER_AREA), cap))
    rows = [ims[k:k + cols] for k in range(0, len(ims), cols)]
    heights = [max(im.shape[0] for im, _ in r) + 30 for r in rows]  # a row is as tall as its tallest tile
    out = np.full((sum(heights), cols * tile, 3), 255, np.uint8)
    y = 0
    for r, h in zip(rows, heights):
        for c, (img, cap) in enumerate(r):
            x = c * tile
            out[y:y + img.shape[0], x:x + img.shape[1]] = img
            for k, line in enumerate([cap[:int(tile / 6.3)], cap[int(tile / 6.3):int(tile / 3.15)]]):
                cv2.putText(out, line, (x + 3, y + h - 18 + 12 * k), cv2.FONT_HERSHEY_SIMPLEX, .38, (0, 0, 0), 1, cv2.LINE_AA)
        y += h
    cv2.imwrite(str(path), out, [cv2.IMWRITE_JPEG_QUALITY, 80])


def side_by_side(a, b, h=224):
    import cv2
    ims = [cv2.imdecode(np.frombuffer(x, np.uint8), cv2.IMREAD_COLOR) for x in (a, b)]
    ims = [cv2.resize(im, (max(1, int(im.shape[1] * h / im.shape[0])), h)) for im in ims]
    gap = np.full((h, 8, 3), 255, np.uint8)
    return cv2.imencode(".jpg", np.hstack([ims[0], gap, ims[1]]), [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()


def build(out):
    """Sets a and b -> OUT/sets/{a,b}.json + crops + contact sheets."""
    out = Path(out)
    crops = out / "sets/crops"
    crops.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)
    a_items, b_items, stats = [], [], {}
    for site, (run_dir, report, clip) in SITES.items():
        obs, objs, ents, names, ref, vocab, lay = observations(site)
        fps = json.loads((PHASE2 / "data/clips" / clip / "source-full.json").read_text())["fps"]
        pairs = pick_pairs(obs, ents, names, ref["mpn"], fps, rng)
        # b: each core object's best view (the VLM crop's frame) matched to one delivered named object, then up to 2
        # more views of the same object (>= 1 s apart) so every site gives enough items; the view index is kept
        by_oid = {}
        for i, o in enumerate(obs):
            if category(names, o["ent"]):
                by_oid.setdefault(o["oid"], []).append(i)
        b_pick = []
        for oid, idx in by_oid.items():
            best = objs.get(oid, {}).get("best_key")
            idx = sorted(idx, key=lambda i: (obs[i]["frame"] != best, -obs[i]["iou"]))
            kept = []
            for i in idx:
                if all(abs(obs[i]["frame"] - obs[k]["frame"]) >= fps for k in kept) and len(kept) < 3:
                    kept.append(i)
            b_pick += [(i, v) for v, i in enumerate(kept)]
        need = sorted({i for _, i, j in pairs for i in (i, j)} | {i for i, _ in b_pick})
        frames = decode(PHASE2 / "data/clips" / clip / "source-full.mp4", sorted({obs[i]["frame"] for i in need}))
        cid = {}
        for i in need:
            o = obs[i]
            m = fill(o["polys"], (720, 1280))
            if m.sum() < 50 or o["frame"] not in frames:
                continue
            vlm, sig = crop_pair(frames[o["frame"]], m)
            key = f"{site}-{o['frame']:04d}-{o['oid']}"
            (crops / f"{key}.jpg").write_bytes(vlm)
            (crops / f"{key}.sig.jpg").write_bytes(sig)
            cid[i] = key
        for kind, i, j in pairs:
            if i in cid and j in cid:
                a_items.append({"id": f"a-{len(a_items):03d}", "site": site, "kind": kind, "truth": int(kind.startswith("pos")),
                                "crops": [cid[i], cid[j]], "frames": [obs[i]["frame"], obs[j]["frame"]], "core_objects": [obs[i]["oid"], obs[j]["oid"]],
                                "delivered": [obs[i]["ent"], obs[j]["ent"]], "categories": [category(names, obs[i]["ent"]), category(names, obs[j]["ent"])],
                                "iou": [obs[i]["iou"], obs[j]["iou"]], "truth_source": "reference: delivered object map identity (mask IoU >= 0.5)"})
        for i, view in b_pick:
            if i not in cid:
                continue
            cat = category(names, obs[i]["ent"])
            o = objs.get(obs[i]["oid"], {})
            b_items.append({"id": f"b-{len(b_items):03d}", "site": site, "crop": cid[i], "frame": obs[i]["frame"], "core_object": obs[i]["oid"],
                            "view": view, "delivered": obs[i]["ent"], "delivered_category": cat, "vocabulary": vocab,
                            "correct_words": [w for w in vocab if name_match(w, cat)], "sam3_word": o.get("word"),
                            "cascade_label": (o.get("cascade") or {}).get("label") or o.get("label") or o.get("word"),
                            "cascade_source": (o.get("cascade") or {}).get("source"),
                            "truth_source": "reference: delivered report name (names.json), name rule of FAST-BUILD-SPEC section 10"})
        stats[site] = {"observations_matched": len(obs), "core_objects": len(objs), "delivered_objects": len(ents), "vocabulary": len(vocab),
                       "pairs": {k: sum(p[0] == k for p in pairs) for k in ("pos-cross-object", "pos-same-object", "neg-lookalike", "neg-other-class")},
                       "b_items": sum(1 for x in b_items if x["site"] == site)}
        print(site, json.dumps(stats[site]))
    (out / "sets/a.json").write_text(json.dumps({"items": a_items, "stats": stats}, indent=1))
    (out / "sets/b.json").write_text(json.dumps({"items": b_items}, indent=1))
    write_sheets(out)
    return stats


C_SHEETS = [("me340", "amg", False, 1), ("me340", "amg", True, 1), ("me340", "vlm-ground", False, 1), ("me340", "sam3-generic", False, 1),
            ("samsclub-a2", "amg16", False, 2), ("samsclub-a2", "amg16", True, 1), ("walmart", "amg16", False, 1), ("walmart", "amg16", True, 1)]
X2_RUN = RUNS / "fx-x2-discover-002"


def tiles(out):
    """Set c: X2 run 002's contact-sheet tiles (220 px crops, red outline) cut back out, with X2's name and its Qwen judge
    letter; the tile order is discover.run_design's (seeded permutation of the design's objects, 48 a page)."""
    import cv2
    out = Path(out)
    crops = out / "sets/crops"
    crops.mkdir(parents=True, exist_ok=True)
    items = []
    for site, design, expansion, pages in C_SHEETS:
        rec = json.loads((X2_RUN / site / "discover.json").read_text())["designs"][design]
        objs = rec["expansion_objects" if expansion else "objects"]
        order = np.random.default_rng(0).permutation(len(objs)).tolist()
        for page in range(pages):
            img = cv2.imread(str(X2_RUN / site / f"sheet-{design}{'-expansion' if expansion else ''}-{page}.jpg"))
            for k in range(48):
                g = page * 48 + k
                if g >= len(objs):
                    break
                r, c = divmod(k, 6)
                cell = img[r * 254:r * 254 + 220, c * 220:(c + 1) * 220]
                ink = (cell < 235).any(2)
                if not ink.any():
                    continue
                h, w = int(np.flatnonzero(ink.any(1))[-1]) + 1, int(np.flatnonzero(ink.any(0))[-1]) + 1
                o = objs[order[g]]
                key = f"c-{site}-{design.replace(':', '_')}{'-x' if expansion else ''}-{order[g]:03d}"
                cv2.imwrite(str(crops / f"{key}.jpg"), cell[:h, :w], [cv2.IMWRITE_JPEG_QUALITY, 92])
                items.append({"id": f"c-{len(items):03d}", "site": site, "design": design, "expansion": expansion, "object": order[g], "crop": key,
                              "x2_name": o.get("label"), "x2_qwen_judge": o.get("judge"), "x2_qwen_judge_name": o.get("judge_name")})
    (out / "sets/c.json").write_text(json.dumps({"items": items, "source": str(X2_RUN), "note": "tiles are 220 px (the sheet's), X2's judge saw 448 px"}, indent=1))
    for k in range(0, len(items), 48):
        sheet([((crops / f"{x['crop']}.jpg").read_bytes(), f"{x['id']} {x['x2_name']}") for x in items[k:k + 48]], out / f"sheets/c-{k // 48}.jpg")
    return len(items)


D_QUESTIONS = {  # the task's four EHS questions, worded for one image; q5 added because the footage has its positives
    "q1": "Is there a cable or hose lying across the floor where people walk?",
    "q2": "Are boxes or goods stacked unstably, so that they could fall?",
    "q3": "Is a person standing on a shelf, rack or pallet?",
    "q4": "Is the aisle or exit route blocked or narrowed by an object standing in it?",
    "q5": "Is a person standing on a raised platform, step or box instead of the floor?"}
DC, DP = PHASE2 / "data/clips", PHASE2 / "data/public-clips"
D_SOURCES = {"samsclub": DC / "samsclub-337/source-full.mp4", "walmart": DC / "walmart-190/source-full.mp4", "lightning": DC / "lightning-3585/source-rgb.mp4",
             "me340": DC / "me340-165/source-full.mp4", "tank": DP / "tankodrome-walk.webm", "nasa": DP / "nasa-pace-cleanroom-dolly1.mp4",
             "agv": DP / "agv-ice-depot.webm", "tum": PHASE2 / "data/rgbd_dataset_freiburg1_room/rgb"}
# (source, frame, {question: label}) labels by looking: 'Y'/'N' clear, 'y'/'n' a judgement call (kept, flagged)
D_ITEMS = [
    ("samsclub", 25, "q4=Y q2=n q3=N q1=N"), ("samsclub", 75, "q4=Y q2=N q3=N q1=N"), ("samsclub", 125, "q4=N q2=N q1=N"), ("samsclub", 175, "q4=N q2=N q1=N"),
    ("samsclub", 225, "q4=y q1=N"), ("samsclub", 275, "q4=y q1=N"), ("samsclub", 325, "q4=Y q2=n q1=N"), ("samsclub", 375, "q4=Y q2=n q1=N"),
    ("samsclub", 425, "q4=y q2=N"), ("samsclub", 475, "q4=y q2=N"), ("samsclub", 525, "q4=y"), ("samsclub", 575, "q4=y"),
    ("walmart", 25, "q4=N q1=N"), ("walmart", 75, "q4=N"), ("walmart", 125, "q4=N"), ("walmart", 175, "q4=N q2=N"), ("walmart", 225, "q4=N"),
    ("walmart", 275, "q4=n q2=N"), ("walmart", 325, "q4=n"), ("walmart", 375, "q4=n q2=n"), ("walmart", 425, "q4=N q1=N q2=N"), ("walmart", 475, "q4=N"),
    ("walmart", 525, "q4=N"), ("walmart", 575, "q4=N"),
    ("lightning", 30, "q4=N q3=N q1=N"), ("lightning", 90, "q4=N q1=N"), ("lightning", 150, "q4=n q2=N"), ("lightning", 210, "q4=y q2=N"),
    ("lightning", 270, "q4=y q2=N"), ("lightning", 330, "q4=y q3=N"), ("lightning", 390, "q4=Y q1=N"), ("lightning", 450, "q4=N q3=N"),
    ("lightning", 509, "q4=n q3=N"), ("lightning", 569, "q4=y"), ("lightning", 629, "q4=y"), ("lightning", 689, "q4=n q3=N"), ("lightning", 749, "q1=n"),
    ("tank", 539, "q4=y"), ("tank", 1019, "q4=N"), ("tank", 1499, "q4=n"), ("tank", 1978, "q4=N"), ("tank", 2458, "q4=N"), ("tank", 2937, "q4=N"),
    ("tank", 3417, "q4=N"), ("tank", 3896, "q4=N"), ("tank", 4376, "q4=N"), ("tank", 4855, "q4=y"), ("tank", 5335, "q4=N"),
    ("me340", 30, "q1=n q3=N q4=N"), ("me340", 90, "q1=n"), ("me340", 270, "q1=N q4=N"), ("me340", 330, "q4=N"), ("me340", 390, "q4=N"),
    ("me340", 450, "q1=N q4=N"), ("me340", 509, "q4=N"), ("me340", 569, "q4=N"), ("me340", 629, "q1=n"), ("me340", 689, "q1=n"),
    ("nasa", 30, "q1=n q5=N"), ("nasa", 210, "q3=N q5=N q1=N"), ("nasa", 569, "q5=N"), ("nasa", 1289, "q1=n"), ("nasa", 1469, "q5=N"),
    ("nasa", 2368, "q5=N"), ("nasa", 2907, "q5=Y q3=n"), ("nasa", 3087, "q5=Y q3=n"), ("nasa", 3267, "q5=N"), ("nasa", 3447, "q5=N"),
    ("nasa", 6503, "q1=y"), ("nasa", 6683, "q1=y"),
    ("agv", 120, "q2=N"), ("agv", 480, "q2=N"), ("agv", 1019, "q2=N"), ("agv", 1203, "q2=n"),
    ("tum", "1305031942.771708", "q1=y"), ("tum", "1305031926.769141", "q1=y"), ("tum", "1305031934.769124", "q1=N"), ("tum", "1305031913.433081", "q1=N"),
    ("tum", "1305031937.438211", "q3=N"),
]


def frames(out):
    """Set d: the listed frames (long side <= 960 px) and one item per (frame, question) with its label."""
    import cv2
    out = Path(out)
    crops = out / "sets/crops"
    crops.mkdir(parents=True, exist_ok=True)
    items, keys = [], {}
    for src, f, labels in D_ITEMS:
        key = f"d-{src}-{f}"
        if key not in keys:
            if src == "tum":
                im = cv2.imread(str(D_SOURCES["tum"] / f"{f}.png"))
            else:
                cap = cv2.VideoCapture(str(D_SOURCES[src]))
                cap.set(cv2.CAP_PROP_POS_FRAMES, f)
                ok, im = cap.read()
                assert ok, key
            s = min(1., 960 / max(im.shape[:2]))
            im = cv2.resize(im, (int(im.shape[1] * s), int(im.shape[0] * s)), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(crops / f"{key}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 88])
            keys[key] = True
        for tok in labels.split():
            q, v = tok.split("=")
            items.append({"id": f"d-{len(items):03d}", "source": src, "frame": f, "crop": key, "question_id": q, "question": D_QUESTIONS[q],
                          "truth": int(v.upper() == "Y"), "clear": v.isupper(), "labeller": "agent-labelled"})
    (out / "sets/d.json").write_text(json.dumps({"items": items, "questions": D_QUESTIONS, "sources": {k: str(v) for k, v in D_SOURCES.items()},
                                                 "note": "Q2 and Q3 have no positives in any footage available here: they measure false alarms only"}, indent=1))
    sheet([((crops / f"{x['crop']}.jpg").read_bytes(), f"{x['id']} {x['question_id']}={'Y' if x['truth'] else 'N'}{'' if x['clear'] else '?'} {x['crop'][2:]}")
           for x in items if x["question_id"] in ("q4", "q5") or x["truth"]], out / "sheets/d-positives-and-aisles.jpg", cols=6, tile=220)
    return {q: {"n": sum(x["question_id"] == q for x in items), "yes": sum(x["truth"] for x in items if x["question_id"] == q)} for q in D_QUESTIONS}


def write_sheets(out, per=18):
    """Audit sheets: pairs side by side (18 a page, readable at full size), b crops 48 a page."""
    out = Path(out)
    crops, sheets = out / "sets/crops", out / "sheets"
    sheets.mkdir(exist_ok=True)
    a_items = json.loads((out / "sets/a.json").read_text())["items"]
    b_items = json.loads((out / "sets/b.json").read_text())["items"]
    for k in range(0, len(a_items), per):
        sheet([(side_by_side((crops / f"{x['crops'][0]}.jpg").read_bytes(), (crops / f"{x['crops'][1]}.jpg").read_bytes()),
                f"{x['id']} {x['kind']} truth={x['truth']} {x['categories'][0]} | {x['categories'][1]}") for x in a_items[k:k + per]],
              sheets / f"a-{k // per:02d}.jpg", cols=3, tile=420)
    for k in range(0, len(b_items), 48):
        sheet([((crops / f"{x['crop']}.jpg").read_bytes(), f"{x['id']} {x['delivered_category']}") for x in b_items[k:k + 48]], sheets / f"b-{k // 48}.jpg")


def self_check():
    assert name_match("cabinet", "tool cabinet") and name_match("paper towel pack", "packs") and not name_match("cabinet", "lathe")
    assert np.allclose(tum([[160, 0], [1120, 720]]), [[0, 0], [640, 480]])
    rng = random.Random(0)
    obs = [{"oid": "o1", "frame": 0, "ent": "e1"}, {"oid": "o2", "frame": 60, "ent": "e1"}, {"oid": "o1", "frame": 90, "ent": "e1"},
           {"oid": "o3", "frame": 0, "ent": "e2"}, {"oid": "o4", "frame": 0, "ent": "e3"}]
    ents = {"e1": {"centroidNative": [0, 0, 0]}, "e2": {"centroidNative": [1, 0, 0]}, "e3": {"centroidNative": [0, 0, 5]}}
    names = {"e1": {"category": "box", "status": "clear"}, "e2": {"category": "cardboard box", "status": "clear"}, "e3": {"category": "lathe", "status": "clear"}}
    kinds = {k for k, *_ in pick_pairs(obs, ents, names, 1., 30, rng)}
    assert kinds == {"pos-cross-object", "pos-same-object", "neg-lookalike", "neg-other-class"}, kinds
    m = np.zeros((720, 1280), bool)
    m[300:340, 600:700] = True
    vlm, sig = crop_pair(np.zeros((720, 1280, 3), np.uint8), m)
    assert vlm[:2] == sig[:2] == b"\xff\xd8"
    print("x8 sets self-check ok: name rule, raster map, pair kinds, crops")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--self-check"]:
        self_check()
    elif sys.argv[1:2] == ["tiles"]:
        print(tiles(sys.argv[2]))
    elif sys.argv[1:2] == ["frames"]:
        print(frames(sys.argv[2]))
    elif sys.argv[1:2] == ["sheets"]:
        write_sheets(sys.argv[2])
    elif sys.argv[1:2] == ["build"]:
        print(json.dumps(build(sys.argv[2]), indent=1))
    else:
        print(__doc__)
