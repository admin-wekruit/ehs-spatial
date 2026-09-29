"""r4 (physical): does every card carry the object layer's physical fields, and do the sizes look right against the picture?

    python scripts/r4_physical.py completeness RUN_DIR [...]            # every cards version of every mirrored report -> JSON
    python scripts/r4_physical.py sheet MIRROR REPORT MP4 OUT_DIR [--n 30 --seed 4]   # contact sheets for the audit by eye
    python scripts/r4_physical.py --self-check

sheet: n cards drawn at random (seeded) among the final cards with a pick region (objects of every identity state and people),
each on the keyframe where its region is largest, outlined, with its name, identity state and physical values (each +-u,
'>=' / '<=' for bounds, 'n/o' not observed, 'n/m' not measurable); labels.json lists them for the agent's labels
(plausible | implausible | unclear, by looking; 'agent-labelled').
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "scripts")]
SHOW = (("height", "H"), ("width", "W"), ("depth", "D"), ("visible_length", "VL"), ("top_above_floor", "top"), ("base_above_floor", "base"))


def short(f):
    """One field as the sheet shows it."""
    if not isinstance(f, dict):
        return "-"
    st = f.get("status")
    if "value" not in f:
        vis = f" ({short(f['visible'])})" if isinstance(f.get("visible"), dict) else ""  # a one-side depth's visible lower bound
        return {"not observed": "n/o", "not measurable": "n/m"}.get(st, st or "-") + vis
    v = f["value"]
    if isinstance(v, list):
        return "(" + ", ".join(f"{x:.1f}" for x in v) + f") +-{f['u']:.1f}"
    pre = ">=" if st == "at least" or f.get("bound") == "at least" else "<=" if st == "at most" or f.get("bound") == "at most" else ""
    return f"{pre}{v:.2f}" + ("" if f.get("bound") and not f.get("u") else f"+-{f['u']:.2f}") + ("?" if st == "needs review" else "")


def completeness(run_dir):
    """{report: {cards version: {required (cards.completeness), contract violations, cards}}} over every mirrored report."""
    from fast_report import cards as fc
    out = {}
    for rep, r in fc.summarize_run(Path(run_dir)).items():
        out[rep] = {v: {"required": x["required"], "violations": x["contract"]["violations"], "examples": x["contract"]["examples"][:5],
                        "cards": x["contract"]["cards"], "sent_s": x["sent_s"]} for v, x in r["cards"].items()}
    return out


def sheet(mirror, report, mp4, out_dir, n=30, seed=4, per=10):
    import cv2
    import fast_report_eval as ev
    from fast_report import cards as fc
    L = ev.load_layers(mirror, report)
    pick = ev.run_picks(mirror, report, L)[-1][1]
    oc = L["object_cards"]
    alias = oc.get("aliases") or {}
    ent = pick.data["entities"]
    owner = {}
    for e in ent:  # a pick entity -> the card that holds it (merged fragments point at their kept card)
        c = e
        while c in alias and alias[c] != c:
            c = alias[c]
        owner[e] = c
    best = {}
    for i, f in enumerate(pick.frames):
        if f.get("source") != "segmented":
            continue
        m = pick.map(i)
        idx, cnt = np.unique(m[m > 0], return_counts=True)
        for j, a in zip(idx, cnt):
            cid = owner.get(ent[int(j)])
            if cid and a > best.get(cid, (0,))[0]:
                best[cid] = (int(a), i)
    cards = [c for c in oc["cards"] if c.get("kind") in ("object", "person") and c["id"] in best]
    pickd = [cards[i] for i in np.random.default_rng(seed).choice(len(cards), min(n, len(cards)), replace=False)]
    cap = cv2.VideoCapture(str(mp4))
    tiles, rows = [], []
    for k, c in enumerate(pickd):
        i = best[c["id"]][1]
        f = pick.frames[i]
        cap.set(cv2.CAP_PROP_POS_FRAMES, f["frame"])
        ok, img = cap.read()
        img = cv2.resize(img if ok else np.zeros((360, 640, 3), np.uint8), (640, 360))
        m = pick.map(i)
        mk = np.isin(m, [j for j, e in enumerate(ent) if e and owner.get(e) == c["id"]]).astype(np.uint8)
        mk = cv2.resize(mk, (640, 360), interpolation=cv2.INTER_NEAREST)
        cs, _ = cv2.findContours(mk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cs, -1, (255, 255, 255), 3)
        cv2.drawContours(img, cs, -1, (0, 160, 255), 1)
        ph = c.get("physical") or {}
        state = "person" if c["kind"] == "person" else fc.identified(c)
        lines = [f"#{k} {c['id']} [{state}]", f"{(c.get('identity') or {}).get('name')}"[:48],
                 "  ".join(f"{lab} {short(ph.get(fld))}" for fld, lab in SHOW[:4]),
                 "  ".join(f"{lab} {short(ph.get(fld))}" for fld, lab in SHOW[4:]) + f"  @{(c.get('views') or {}).get('distance_m', ['?'])[0]} m"]
        band = np.zeros((6 + 20 * len(lines), 640, 3), np.uint8)  # the text above the frame: it never hides a region
        for j, t in enumerate(lines):
            cv2.putText(band, t, (6, 18 + 20 * j), cv2.FONT_HERSHEY_SIMPLEX, .48, (255, 255, 255), 1, cv2.LINE_AA)
        img = np.vstack([band, img])
        tiles.append(img)
        rows.append({"k": k, "card": c["id"], "kind": c["kind"], "state": state, "name": (c.get("identity") or {}).get("name"), "frame": f["frame"],
                     "physical": {fld: ph.get(fld) for fld, _ in SHOW}, "size_check": (ph.get("size_check") or {}).get("status"),
                     "label": None, "note": None})
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for s in range(0, len(tiles), per):
        t = tiles[s:s + per]
        t += [np.zeros_like(t[0])] * (-len(t) % 2)
        cv2.imwrite(str(out_dir / f"sheet-{s // per}.jpg"), np.vstack([np.hstack(t[q:q + 2]) for q in range(0, len(t), 2)]), [cv2.IMWRITE_JPEG_QUALITY, 75])
    (out_dir / "labels.json").write_text(json.dumps({"report": report, "seed": seed, "labelled_by": "agent (by looking at the sheets)", "rows": rows}, indent=1))
    return rows


FIELD_ORDER = ("position_xy", "top_above_floor", "base_above_floor", "height", "width", "depth", "principal_axis_tilt_deg", "planar_slope_deg")


def completeness_table(req, title):
    """Markdown: per card kind / identity state, complete share, and per REQUIRED field the share carried as a number
    (value or bound) vs a status with its reason (never missing)."""
    md = f"\n{title}\n\n| cards | n | complete | " + " | ".join(FIELD_ORDER) + " |\n|---|---|---|" + "---|" * len(FIELD_ORDER) + "\n"
    for key, r in req.items():
        cells = []
        for f in FIELD_ORDER:
            t = r["fields"][f]
            num = sum(v for k, v in t.items() if k in ("value", "at least", "at most", "needs review"))
            why = sum(v for k, v in t.items() if k in ("not observed", "not measurable"))
            bad = sum(v for k, v in t.items() if k in ("missing", "broken"))
            cells.append(f"{num / r['cards']:.2f} / {why / r['cards']:.2f}" + (f" / **{bad} missing**" if bad else ""))
        md += f"| {key} | {r['cards']} | {r['complete_share']:.2f} | " + " | ".join(cells) + " |\n"
    return md


def gt_tables(score):
    """Markdown: per sequence and field, median |error| as delivered (assumed 1.6 m camera height) and at the true camera
    height, coverage within +-u, n; the true camera heights; bounds holding."""
    md = "\n| sequence | field | n | median abs err (a) assumed 1.6 m | (b) true height | cov (a) | cov (b) | u median |\n|---|---|---|---|---|---|---|---|\n"
    for t in score["table"]:
        cm = "deg" if t["field"].endswith("_deg") else "cm"
        k = 1. if cm == "deg" else 100.
        md += (f"| {t['seq']} | {t['field']} | {t['n']} | {k * t['a_med']:.1f} {cm} | {k * t['b_med']:.1f} {cm} | {t['a_cov']:.2f} | {t['b_cov']:.2f} | "
               f"{k * t['u_med']:.1f} {cm} |\n")
    hs = {}
    for r in score["reports"]:
        for x in r["shots"].values():
            hs.setdefault(r["seq"], []).append(x["gt_camera_height_m"])
    md += "\nTrue camera height over the floor (the delivered scale assumes 1.6 m): " + "; ".join(
        f"{q} {min(v):.2f}-{max(v):.2f} m" for q, v in sorted(hs.items())) + "\n"
    b = {}
    for r in score["reports"]:
        if r.get("first_call"):
            continue
        for x in r["bounds"]:
            if x["gt_cover"] < .5:
                continue
            key = (x["seq"], x["field"], x["status"])
            a = b.setdefault(key, [0, 0, 0])
            a[0] += 1
            a[1] += x["holds_a"]
            a[2] += x.get("holds_b", x["holds_a"])
    md += "\n| bounds (warm calls, GT on >= 50 % of the mask) | n | hold (a) | hold (b) |\n|---|---|---|---|\n"
    for (q, f, st), (n_, ha, hb) in sorted(b.items()):
        md += f"| {q} {f} {st} | {n_} | {ha} | {hb} |\n"
    return md


def audit_table(labels):
    """{video: labels.json} -> markdown counts of the agent's labels."""
    md = "\n| video | cards looked at | plausible | implausible | unclear |\n|---|---|---|---|---|\n"
    for v, d in labels.items():
        c = [r.get("label") for r in d["rows"]]
        md += f"| {v} | {len(c)} | {c.count('plausible')} | {c.count('implausible')} | {c.count('unclear')} |\n"
    return md


def self_check():
    assert short({"value": 1.234, "u": .1, "status": "at least"}) == ">=1.23+-0.10"
    assert short({"value": .5, "u": 0., "bound": "at most", "status": "needs review"}) == "<=0.50?"
    assert short({"status": "not observed", "reason": "x"}) == "n/o" and short(None) == "-"
    assert short({"status": "not observed", "reason": "x", "visible": {"value": .4, "u": .1, "status": "at least"}}) == "n/o (>=0.40+-0.10)"
    assert short({"value": [1., 2.], "u": .3}) == "(1.0, 2.0) +-0.3"
    req = {"object": {"cards": 2, "complete_share": 1., "fields": {f: {"value": 1, "not observed": 1} for f in FIELD_ORDER}}}
    assert "| object | 2 | 1.00 | 0.50 / 0.50 |" in completeness_table(req, "t")
    req["object"]["fields"]["depth"] = {"missing": 2}
    assert "0.00 / 0.00 / **2 missing**" in completeness_table(req, "t")
    sc = {"table": [{"seq": "tum", "field": "height", "n": 3, "a_med": .05, "b_med": .02, "a_cov": 1., "b_cov": .9, "u_med": .1}],
          "reports": [{"seq": "tum", "first_call": False, "shots": {"0": {"gt_camera_height_m": 1.5}},
                       "bounds": [{"seq": "tum", "field": "depth", "status": "at least", "gt_cover": .8, "holds_a": True, "holds_b": False}]}]}
    md = gt_tables(sc)
    assert "| tum | height | 3 | 5.0 cm | 2.0 cm |" in md and "tum 1.50-1.50 m" in md and "| tum depth at least | 1 | 1 | 0 |" in md, md
    assert "| v | 2 | 1 | 0 | 1 |" in audit_table({"v": {"rows": [{"label": "plausible"}, {"label": "unclear"}]}})
    print("r4_physical self-check ok: sheet field format, completeness / GT / audit tables")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("cmd", nargs="?", choices=("completeness", "sheet"))
    p.add_argument("args", nargs="*")
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--seed", type=int, default=4)
    p.add_argument("--out", type=Path)
    p.add_argument("--self-check", action="store_true")
    a = p.parse_args()
    if a.self_check:
        self_check()
    elif a.cmd == "completeness":
        res = {d: completeness(d) for d in a.args}
        (a.out.write_text(json.dumps(res, indent=1)) if a.out else print(json.dumps(res, indent=1)))
    else:
        sheet(*a.args, n=a.n, seed=a.seed)
