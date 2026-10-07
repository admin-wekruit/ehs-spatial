"""mvp3 integrate: fresh audit of random object cards (a new seed): each card's best view with its outline beside what the card
says (name, the physical values with +-u and their scale label, its judgement rows), for the agent to label by eye:
name right | close | wrong | unclear; physical plausible | implausible | unclear (against the picture, at the stated u);
judgement right | wrong (its PASS / FAIL rows) | undecided (NEEDS_REVIEW / NO_DATA only) | none (no row) | unclear. Labels are agent-made.

  python scripts/card_audit_mvp3.py sheets --run RUN --out DIR [--kind warm] [--n 20] [--seed 4242]
  python scripts/card_audit_mvp3.py score --out DIR [DIR ...]
"""
import argparse
import glob
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "scripts")]
import judge_offline as jo  # noqa: E402

FIELDS = ("top_above_floor", "base_above_floor", "height", "width", "depth", "visible_length")
LABELS = {"name": ("right", "close", "wrong", "unclear"), "physical": ("plausible", "implausible", "unclear"),
          "judgement": ("right", "wrong", "undecided", "none", "unclear")}  # right / wrong: a PASS or FAIL row; undecided: NEEDS_REVIEW / NO_DATA only


def value(f):
    if not isinstance(f, dict):
        return "-"
    if f.get("value") is None:
        return f"{f.get('status', '-')}" + (f" ({str(f.get('reason'))[:34]})" if f.get("reason") else "")
    if f.get("bound") or f.get("status") in ("at most", "at least"):
        return f"{f.get('bound') or f['status']} {f['value']:.2f} {f.get('unit', '')}"
    st = f" [{f['status']}]" if f.get("status") else ""
    return f"{f['value']:.2f} +- {f.get('u', 0):.2f} {f.get('unit', '')}{st}"


def lines(card, rows):
    idn = card.get("identity") or {}
    out = [f"{card['id']}", f"name: {idn.get('name')}", f"  by {idn.get('decided_by') or '-'}", f"class: {(card.get('class') or {}).get('category')}"]
    out += [f"{k.replace('_above_floor', '')}: {value((card.get('physical') or {}).get(k))}" for k in FIELDS if k != "visible_length" or (card.get("physical") or {}).get(k)]
    out.append("scale: estimated (floor + 1.6 m camera)")
    for r in rows[:4]:
        g = r.get("geometry") or {}
        v = f"{g['value']:.2f}+-{g.get('u') or 0:.2f}" if isinstance(g.get("value"), (int, float)) else "-"
        out.append(f"{r['check']} {r['verdict']}: {v} vs {g.get('threshold')}")
    if not rows:
        out.append("judgement: no check")
    return out


def tile(frame, polys, text, w=640, h=360, tw=420):
    """The frame with the outline, a zoom around it (the outline's box, 1.6 x, at least 160 source px), the card's text."""
    import cv2
    s = w / frame.shape[1]
    img = cv2.resize(frame, (w, round(frame.shape[0] * s)))
    img = np.pad(img, ((0, max(0, h - img.shape[0])), (0, 0), (0, 0)))[:h]
    full = frame.copy()
    for p in polys or []:
        q = np.asarray(p, float).reshape(-1, 2)
        for im, k in ((img, s), (full, 1.)):
            cv2.polylines(im, [np.round(q * k).astype(np.int32)], True, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.polylines(im, [np.round(q * k).astype(np.int32)], True, (0, 255, 255), 2, cv2.LINE_AA)
    zoom = np.zeros((h, h, 3), np.uint8)
    if polys:
        xy = np.concatenate([np.asarray(p, float).reshape(-1, 2) for p in polys])
        c, r = (xy.min(0) + xy.max(0)) / 2, max(80., .8 * float((xy.max(0) - xy.min(0)).max()))
        x0, y0 = int(max(0, c[0] - r)), int(max(0, c[1] - r))
        crop = full[y0:int(c[1] + r), x0:int(c[0] + r)]
        if crop.size:
            k = h / max(crop.shape[:2])
            crop = cv2.resize(crop, (max(1, round(crop.shape[1] * k)), max(1, round(crop.shape[0] * k))))
            zoom[:crop.shape[0], :crop.shape[1]] = crop
    pane = np.full((h, tw, 3), 245, np.uint8)
    for i, t in enumerate(text):
        cv2.putText(pane, t[:52], (6, 20 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 1, cv2.LINE_AA)
    return np.hstack([img, zoom, pane])


def sheets(a):
    import cv2
    run = Path(a.run)
    c = next(json.loads(Path(f).read_text()) for f in glob.glob(str(run / "call-*.json")) if json.loads(Path(f).read_text())["kind"] == a.kind)
    report, site = c["run"]["report"], c["site"]
    d = jo.load(run=run, report=report)
    js = [json.loads(Path(f).read_text())["data"] for f in sorted(glob.glob(str(run / "mirror/reports" / report / "patches/*-judgements.json")))]
    rows = js[-1]["rows"]
    by = {}
    for r in rows:
        by.setdefault(r["subject"], []).append(r)
    outl = {f["sourceFrame"]: {e["entityId"]: e["polygons"] for e in f["objects"]} for f in d["outlines"]}
    objs = [x for x in d["cards"] if x["kind"] == "object"]
    pick = random.Random(a.seed).sample(objs, min(a.n, len(objs)))
    a.out.mkdir(parents=True, exist_ok=True)
    index, tiles = [], []
    for i, card in enumerate(pick):
        keys = [k for k in (card.get("views") or {}).get("best") or [] if card["id"] in outl.get(k, {})] or \
               sorted((k for k, v in outl.items() if card["id"] in v), key=lambda k: -sum(len(p) for p in outl[k][card["id"]]))
        k = keys[0] if keys else None
        frame = d["ctx"]["frames"][k] if k is not None else np.zeros((720, 1280, 3), np.uint8)
        t = tile(frame, outl.get(k, {}).get(card["id"]), [f"#{i:02d} f{k}"] + lines(card, by.get(card["id"], [])))
        tiles.append(t)
        index.append({"i": i, "id": card["id"], "frame": k, "name": (card.get("identity") or {}).get("name"),
                      "rows": [(r["check"], r["verdict"]) for r in by.get(card["id"], [])]})
    for s0 in range(0, len(tiles), 5):
        cv2.imwrite(str(a.out / f"cards-{site}-{a.kind}-s{a.seed}-{s0 // 5}.jpg"), np.vstack(tiles[s0:s0 + 5]), [cv2.IMWRITE_JPEG_QUALITY, 85])
    (a.out / "index.json").write_text(json.dumps({"run": str(run), "report": report, "site": site, "kind": a.kind, "seed": a.seed,
                                                  "of_objects": len(objs), "cards": index}, indent=1))
    print(site, a.kind, "cards", len(index), "of", len(objs))


def score(dirs):
    out = {}
    for d in dirs:
        d = Path(d)
        idx = json.loads((d / "index.json").read_text())
        lab = {x["i"]: x for x in map(json.loads, (d / "labels.jsonl").read_text().splitlines())}
        assert all(lab[c["i"]][k] in v for c in idx["cards"] for k, v in LABELS.items()), f"{d}: unlabelled or unknown labels"
        cnt = {k: dict(Counter(lab[c["i"]][k] for c in idx["cards"])) for k in LABELS}
        n = len(idx["cards"])
        out[idx["site"]] = {"n": n, "of_objects": idx["of_objects"], "kind": idx["kind"], "seed": idx["seed"], **cnt,
                            "name_right_or_close": round((cnt["name"].get("right", 0) + cnt["name"].get("close", 0)) / n, 2),
                            "physical_plausible": round(cnt["physical"].get("plausible", 0) / n, 2)}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("sheets", "score"))
    ap.add_argument("dirs", nargs="*")
    ap.add_argument("--run")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--kind", default="warm")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=4242)
    a = ap.parse_args()
    if a.cmd == "sheets":
        sheets(a)
    else:
        print(json.dumps(score(a.dirs), indent=1))
