"""Contact sheets for the click MVP's agent audits (CLICK-MVP-SPEC 8.1, 8.3, 8.4), each with a label template beside it.
The agent (not a person, not ground truth) fills the 'label' fields; fast_report_eval reads them back.

  clicks:      60 object clicks a video that returned an entity (seed 0): the eval frame with the reference outlined (green)
               and the click marked | the pick frame with the picked entity lit (cyan) | the picked entity's best view.
               labels: same object (the lit entity is the referenced thing) | part of it (one is a part of the other: a
               mill's table and the mill, a label on a cabinet) | different (another thing, e.g. the bench under a board) | unclear
  boxes:       30 flagged (implausible) boxes and the 30 largest unflagged a video, on their best view with the outline.
               labels: size right | inflated | unclear
  identity:    40 object cards a video seen on >= 3 keyframes (seed 0) on their best view, with the card's name and route.
               labels: right | close (the right family, or a part / the whole of it) | wrong | unclear
  judgements:  every FAIL, then PASS / NEEDS_REVIEW rows per check (>= 60 over the videos): evidence images + the geometry line.
               labels: present | absent | cannot tell

    python scripts/mvp_sheets.py clicks OUT --site me340 [--run RUN_DIR:REPORT] [--variant v1]
    python scripts/mvp_sheets.py boxes OUT --site me340 [--run RUN_DIR:REPORT]
    python scripts/mvp_sheets.py judgements OUT --runs me340=RUN_DIR:REPORT,...
    python scripts/mvp_sheets.py --self-check
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parent)]
import fast_report_eval as ev  # noqa: E402

PANEL, PER_SHEET, COLS = 220, 12, 2
GREEN, CYAN, YELLOW, WHITE = (60, 220, 60), (255, 220, 0), (0, 230, 255), (255, 255, 255)


def sheet(items, path, cols=COLS, tile=PANEL * 3):
    """[(bgr image, caption)] -> contact sheet JPEG (x8_sets.sheet's layout: rows as tall as their tallest tile, two caption lines)."""
    import cv2
    rows = [items[k:k + cols] for k in range(0, len(items), cols)]
    heights = [max(im.shape[0] for im, _ in r) + 34 for r in rows]
    out = np.full((sum(heights), cols * tile, 3), 255, np.uint8)
    y = 0
    for r, h in zip(rows, heights):
        for c, (img, cap) in enumerate(r):
            x = c * tile
            out[y:y + img.shape[0], x:x + img.shape[1]] = img[:, :tile]
            n = int(tile / 6.6)
            for k, line in enumerate([cap[:n], cap[n:2 * n]]):
                cv2.putText(out, line, (x + 3, y + h - 20 + 13 * k), cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 0, 0), 1, cv2.LINE_AA)
        y += h
    cv2.imwrite(str(path), out, [cv2.IMWRITE_JPEG_QUALITY, 82])


def frames_bgr(mp4, frames):
    import cv2
    cap, out, i, want = cv2.VideoCapture(str(mp4)), {}, 0, set(int(f) for f in frames)
    while len(out) < len(want):
        ok, image = cap.read()
        if not ok:
            break
        if i in want:
            out[i] = image
        i += 1
    cap.release()
    return out


def crop(img, box, side=PANEL):
    """Square crop around box (x0, y0, x1, y1) at 1.6x, at least 120 px, resized to side."""
    import cv2
    H, W = img.shape[:2]
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    half = max(box[2] - box[0], box[3] - box[1], 120) * .8
    x0, x1, y0, y1 = int(max(0, cx - half)), int(min(W, cx + half)), int(max(0, cy - half)), int(min(H, cy + half))
    c = img[y0:y1, x0:x1]
    s = side / max(c.shape[:2])
    c = cv2.resize(c, (max(1, int(c.shape[1] * s)), max(1, int(c.shape[0] * s))))
    return cv2.copyMakeBorder(c, 0, side - c.shape[0], 0, side - c.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255))


def outline(img, mask, color, tint=False):
    import cv2
    img = img.copy()
    if tint:
        img[mask] = (.5 * img[mask] + .5 * np.array(color)).astype(np.uint8)
    cv2.drawContours(img, cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], -1, color, 2)
    return img


def bbox(mask):
    ys, xs = np.nonzero(mask)
    return (xs.min(), ys.min(), xs.max(), ys.max()) if len(xs) else (0, 0, 1, 1)


def poly_mask(polys, shape=(720, 1280)):
    import cv2
    m = np.zeros(shape, np.uint8)
    ps = [np.round(np.asarray(p, float)).astype(np.int32) for p in polys or [] if len(p) >= 3]
    if ps:
        cv2.fillPoly(m, ps, 1)
    return m.astype(bool)


def entity_polys(outlines):
    """{(source frame, entity id): polygons} from an outlines analysis."""
    return {(f["sourceFrame"], o["entityId"]): o.get("polygons") for f in outlines.get("frames", []) for o in f["objects"]}


def best_view(entity, objects, polys):
    """(frame, polygons) of an entity's best view: the objects layer's best_key when it has an outline there, else the
    keyframe where its outline is largest."""
    import cv2
    o = objects.get(entity) or {}
    key = (o.get("views") or {}).get("best", [o.get("best_key")])[0] if o.get("views") else o.get("best_key")
    if (key, entity) in polys:
        return key, polys[(key, entity)]
    own = [(f, p) for (f, e), p in polys.items() if e == entity and p]
    if not own:
        return key, None
    return max(own, key=lambda fp: sum(abs(cv2.contourArea(np.asarray(q, np.float32))) for q in fp[1] if len(q) >= 3))


def clicks(out, site, run_dir, report, variant="v1", n=60, mp4=None):
    """The click audit: n object clicks that returned an entity (seeded; a miss needs no audit), three panels a tile."""
    import cv2
    out = Path(out)
    recs = [r for r in json.loads((out / f"clicks-{site}-{variant}.json").read_text()) if r["kind"] == "object" and r["entity"]]
    pick_idx = np.random.default_rng(0).permutation(len(recs))[:n]
    recs = [recs[i] for i in sorted(pick_idx, key=lambda i: (recs[i]["frame"], recs[i]["x"]))]
    refs = ev.load_refs(out / f"refs-{site}.npz")
    layers = ev.load_layers(run_dir, report)
    picks = dict(ev.run_picks(run_dir, report, layers))
    pick = next(p for name, p in picks.items() if name.split(" ")[1] == variant)
    polys = entity_polys(layers.get("outlines") or {})
    objects = {o["id"]: o for o in layers["objects"]["objects"]}
    names = {o["id"]: o["name"] for o in ev.ours_objects(layers)}
    views = {r["entity"]: best_view(r["entity"], objects, polys) for r in recs if r["entity"]}
    need = {r["frame"] for r in recs} | {r["pick_frame"] for r in recs if r["pick_frame"] is not None} | {v[0] for v in views.values() if v[0] is not None}
    imgs = frames_bgr(mp4 or ev.PHASE2 / "data/clips" / ev.CLIPS[site] / "source-full.mp4", need)
    tiles, labels = [], {}
    for k, r in enumerate(recs):
        ref = ev.ref_mask(refs, r["ref"])
        box = bbox(ref)
        a = outline(imgs[r["frame"]], ref, GREEN)
        cv2.drawMarker(a, (r["x"], r["y"]), YELLOW, cv2.MARKER_CROSS, 28, 3)
        panels = [crop(a, box)]
        if r["entity"]:
            i = int(pick.frame_at(r["frame"] / layers["video"]["fps"]))
            f = pick.frames[i]
            region = cv2.resize((pick.map(i) == pick.data["entities"].index(r["entity"])).astype(np.uint8), (1280, 720), interpolation=cv2.INTER_NEAREST).astype(bool)
            b = outline(imgs[r["pick_frame"]], region, CYAN, tint=True)
            cv2.drawMarker(b, (r["x"], r["y"]), YELLOW, cv2.MARKER_CROSS, 28, 3)
            panels.append(crop(b, box))
            key, ps = views[r["entity"]]
            m = poly_mask(ps)
            panels.append(crop(outline(imgs[key], m, CYAN), bbox(m)) if key in imgs and m.any() else np.full((PANEL, PANEL, 3), 235, np.uint8))
        else:
            f = None
            panels += [np.full((PANEL, PANEL, 3), 235, np.uint8)] * 2
        tid = f"{site}-{variant}-{k:02d}"
        auto = "correct" if r["correct"] else "unknown" if r["entity"] is None else "wrong"
        cap = (f"{tid} ref:{refs['words'][refs['word'][r['ref']]] if refs['word'][r['ref']] >= 0 else 'person'} f{r['frame']} | "
               f"auto {auto} iou {r['iou']} | {r['entity']} '{names.get(r['entity'], '')}' key f{r['pick_frame']} {f['source'] if f else ''}")
        tiles.append((np.hstack(panels), cap))
        labels[tid] = {"auto": bool(r["correct"]), "entity": r["entity"], "frame": r["frame"], "iou": r["iou"], "cover": r["cover"], "label": ""}
    (out / "sheets").mkdir(exist_ok=True)
    for s in range(0, len(tiles), PER_SHEET):
        sheet(tiles[s:s + PER_SHEET], out / "sheets" / f"clicks-{site}-{variant}-{s // PER_SHEET}.jpg")
    path = out / f"labels-clicks-{site}.json"
    if not path.exists():  # never overwrite labels already made
        path.write_text(json.dumps(labels, indent=1))
    return len(tiles)


def boxes(out, site, run_dir, report, n=30, mp4=None):
    """The L1 audit: n flagged (implausible) boxes and the n largest unflagged boxes, on their best view."""
    out = Path(out)
    layers = ev.load_layers(run_dir, report)
    ours = [o for o in ev.ours_objects(layers) if o["longest"] is not None]
    rng = np.random.default_rng(0)
    flagged = [o for o in ours if not o["plausible"]]
    flagged = [flagged[i] for i in sorted(rng.permutation(len(flagged))[:n])]
    large = sorted((o for o in ours if o["plausible"]), key=lambda o: -o["longest"])[:n]
    polys = entity_polys(layers.get("outlines") or {})
    objects = {o["id"]: o for o in layers["objects"]["objects"]}
    views = {o["id"]: best_view(o["id"], objects, polys) for o in flagged + large}
    imgs = frames_bgr(mp4 or ev.PHASE2 / "data/clips" / ev.CLIPS[site] / "source-full.mp4", {v[0] for v in views.values() if v[0] is not None})
    tiles, labels = [], {}
    for group, objs in (("flagged", flagged), ("largest-unflagged", large)):
        for o in objs:
            key, ps = views[o["id"]]
            m = poly_mask(ps)
            img = crop(outline(imgs[key], m, CYAN), bbox(m)) if key in imgs and m.any() else np.full((PANEL, PANEL, 3), 235, np.uint8)
            sides = "x".join(f"{s[0]:.2f}" for s in o["sides"]) if o["sides"] else "?"
            cls, lo, hi, _ = ev.size_class(o.get("word") or o["name"])
            tid = f"{site}-box-{o['id']}"
            tiles.append((img, f"{group} {o['id']} '{o.get('word') or o['name']}' ({cls} {lo:g}-{hi:g} m) longest {o['longest']:.2f} m footprint {sides} f{key}"))
            labels[tid] = {"group": group, "longest_m": round(o["longest"], 3), "class": cls, "label": ""}
    (out / "sheets").mkdir(exist_ok=True)
    per = 24
    for s in range(0, len(tiles), per):
        sheet(tiles[s:s + per], out / "sheets" / f"boxes-{site}-{s // per}.jpg", cols=6, tile=PANEL)
    path = out / f"labels-boxes-{site}.json"
    if not path.exists():
        path.write_text(json.dumps(labels, indent=1))
    return len(tiles)


def judgements(out, runs, n=60):
    """Every FAIL, then PASS / NEEDS_REVIEW rows per check (fast_report_eval.judgement_sample), evidence images and the
    geometry line."""
    import cv2
    out = Path(out)
    by_video, blobs = {}, {}
    for site, (run_dir, report) in runs.items():
        vs = ev.patch_versions(run_dir, report, "judgements")
        if vs:
            by_video[site] = ev.patch_data(run_dir, vs[-1])
            blobs[site] = (run_dir, vs[-1].get("blobs") or {})
    tiles, labels = [], {}
    for site, r in ev.judgement_sample(by_video, n):
        run_dir, bl = blobs[site]
        ims = []
        for e in r.get("evidence", [])[:3]:
            b = bl.get(e.get("image"))
            if b:
                im = cv2.imdecode(np.frombuffer(ev.blob_bytes(run_dir, b["sha256"]), np.uint8), cv2.IMREAD_COLOR)
                ims.append(cv2.resize(im, (PANEL, int(im.shape[0] * PANEL / im.shape[1]))))
        h = max([im.shape[0] for im in ims] + [PANEL])
        ims = [cv2.copyMakeBorder(im, 0, h - im.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=WHITE) for im in ims] or [np.full((h, PANEL, 3), 235, np.uint8)]
        g = r.get("geometry") or {}
        geo = f"{g.get('quantity')} {g.get('value')} +- {g.get('u')} {g.get('unit', '')} vs {g.get('threshold')} ({g.get('direction')})" if g else "no geometry"
        tid = f"{site}:{r['id']}"
        tiles.append((np.hstack(ims + [np.full((h, PANEL * 3 - sum(i.shape[1] for i in ims), 3), 255, np.uint8)])[:, :PANEL * 3],
                      f"{tid} {r['verdict']} {r.get('title', '')} | {geo} | vlm {((r.get('vlm') or {}).get('answer'))}"))
        labels[tid] = {"verdict": r["verdict"], "check": r["check"], "label": ""}
    (out / "sheets").mkdir(exist_ok=True)
    for s in range(0, len(tiles), PER_SHEET):
        sheet(tiles[s:s + PER_SHEET], out / "sheets" / f"judgements-{s // PER_SHEET}.jpg")
    path = out / "labels-judgements.json"
    if not path.exists():
        path.write_text(json.dumps(labels, indent=1))
    return len(tiles)


def identity(out, site, run_dir, report, n=40, mp4=None):
    """The identity audit (mvp/integrate): n object cards seen on >= 3 keyframes (seeded), each on its best view with the
    outline and the card's name, route and probability. labels: right | close (the right family, or a part / the whole of
    it) | wrong | unclear."""
    out = Path(out)
    layers = ev.load_layers(run_dir, report)
    cards = [c for c in (layers.get("object_cards") or {}).get("cards", []) if c.get("kind") == "object"
             and len((c.get("time") or {}).get("detected_keyframes") or []) >= 3]
    polys = entity_polys(layers.get("outlines") or {})
    objects = {c["id"]: c for c in cards}
    views = {c["id"]: best_view(c["id"], objects, polys) for c in cards}
    cards = [c for c in cards if views[c["id"]][1]]
    cards = [cards[i] for i in sorted(np.random.default_rng(0).permutation(len(cards))[:n])]
    imgs = frames_bgr(mp4 or ev.PHASE2 / "data/clips" / ev.CLIPS[site] / "source-full.mp4", {views[c["id"]][0] for c in cards})
    tiles, labels = [], {}
    for c in cards:
        key, ps = views[c["id"]]
        m = poly_mask(ps)
        idn = c["identity"]
        img = crop(outline(imgs[key], m, CYAN), bbox(m), side=PANEL * 2) if key in imgs and m.any() else np.full((PANEL * 2, PANEL * 2, 3), 235, np.uint8)
        tid = f"{site}-id-{c['id']}"
        p = idn.get("confidence")
        tiles.append((img, f"{c['id']} '{idn.get('name')}' by {idn.get('decided_by')}" + (f" p {p:.2f}" if p is not None else "") + f" f{key}"))
        labels[tid] = {"id": c["id"], "name": idn.get("name"), "route": idn.get("decided_by"), "p": p, "label": ""}
    (out / "sheets").mkdir(parents=True, exist_ok=True)
    per = 12
    for s in range(0, len(tiles), per):
        sheet(tiles[s:s + per], out / "sheets" / f"identity-{site}-{s // per}.jpg", cols=4, tile=PANEL * 2)
    path = out / f"labels-identity-{site}.json"
    if not path.exists():
        path.write_text(json.dumps(labels, indent=1))
    return len(tiles)


def self_check():
    import tempfile
    img = np.full((720, 1280, 3), 90, np.uint8)
    m = np.zeros((720, 1280), bool)
    m[100:200, 300:500] = True
    c = crop(outline(img, m, GREEN), bbox(m))
    assert c.shape == (PANEL, PANEL, 3) and (c == GREEN).all(2).any(), "the outline shows in the crop"
    assert poly_mask([[[0, 0], [10, 0], [10, 10], [0, 10]]])[5, 5] and not poly_mask([[[0, 0], [1, 1]]]).any()
    polys = {(6, "a"): [[[0, 0], [9, 0], [9, 9]]], (12, "a"): [[[0, 0], [99, 0], [99, 99]]]}
    assert best_view("a", {"a": {"best_key": 6}}, polys)[0] == 6 and best_view("a", {"a": {"best_key": 3}}, polys)[0] == 12
    with tempfile.TemporaryDirectory() as tmp:
        sheet([(np.hstack([c] * 3), "caption " * 20)] * 3, Path(tmp) / "s.jpg")
        assert (Path(tmp) / "s.jpg").stat().st_size > 1000
    print("mvp_sheets self-check passed: crops, outlines, polygons, best view, sheet")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("what", nargs="?", choices=("clicks", "boxes", "judgements", "identity"))
    p.add_argument("out", nargs="?", type=Path)
    p.add_argument("--site", choices=sorted(ev.CLIPS))
    p.add_argument("--run", default="", help="RUN_DIR:REPORT (default: fb/integrate's warm call of the site)")
    p.add_argument("--runs", default="", help="judgements: site=RUN_DIR:REPORT,...")
    p.add_argument("--variant", default="v1")
    p.add_argument("--self-check", action="store_true")
    a = p.parse_args()
    if a.self_check:
        self_check()
        sys.exit()
    run = (Path(a.run.rsplit(":", 1)[0]), a.run.rsplit(":", 1)[1]) if a.run else (ev.PHASE2 / "runs" / ev.FB_RUNS[a.site][0], ev.FB_RUNS[a.site][1]) if a.site else None
    if a.what == "clicks":
        print(clicks(a.out, a.site, *run, variant=a.variant), "tiles")
    elif a.what == "boxes":
        print(boxes(a.out, a.site, *run), "tiles")
    elif a.what == "identity":
        print(identity(a.out, a.site, *run), "tiles")
    elif a.what == "judgements":
        print(judgements(a.out, {s: (Path(v.rsplit(":", 1)[0]), v.rsplit(":", 1)[1]) for s, v in (x.split("=", 1) for x in a.runs.split(","))}), "tiles")
