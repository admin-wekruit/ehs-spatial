"""r5b: width bounds against metric ground truth next to value coverage, on the same cards, for round 5's rule ('at most' only
when both ends were seen bounded by free space; otherwise 'at least' the extent seen when resolved, else 'not observed') and
round 4's ('at most' = measured + u for every unresolved width, recovered from each field's 'measured'). GT per card as
scripts/accuracy_gt.py: its own pick regions lifted with GT depth and poses, the width along the card's own footprint axes.
(a) as delivered (assumed 1.6 m camera height), (b) at the true camera height (x s, u without its scale part).
Warm calls only, cards with GT on >= 50 % of their mask pixels.

    python scripts/r5b_width_gt.py RUN_DIR [RUN_DIR ...] --inputs RUNS/mvp2-accuracy-inputs --out OUT.json [--md OUT.md]
    python scripts/r5b_width_gt.py --self-check
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "scripts")]


def card_rows(run_dir, report, inputs, cards_override=None):
    """Per object card with GT: its width record, the GT width, the shot's true-scale factor s, GT cover. cards_override: a
    cards.build output ({cards, aliases}) scored in place of the report's own (a replay of its dump under another rule)."""
    import accuracy_gt as ag
    import fast_report_eval as ev
    from fast_report.core import CAMERA_HEIGHT_M
    run = json.loads((run_dir / "reports" / report / "run.json").read_text())
    call = run["call"]
    name, wi = call["site"].split("-")[1], int(call["site"].rsplit("-w", 1)[1])
    seq = ag.Seq(Path(inputs) / name)
    w = seq.g["windows"][wi]
    a0, hold = w["source_frames"][0], w["hold"]
    src = lambda f: a0 + int(f) // hold  # noqa: E731
    L = ev.load_layers(run_dir, report)
    pick = ev.run_picks(run_dir, report, L)[-1][1]
    oc = cards_override or L["object_cards"]
    alias = oc.get("aliases") or {}
    shots = {}
    for s in L["cameras"]["shots"]:
        G = [seq.c2w(src(f)) for f in s["keys"]]
        Gc = np.array([g[:3, 3] for g in G if g is not None])
        n, p0 = np.asarray(seq.floor["normal"]), np.asarray(seq.floor["point"])
        shots[s["index"]] = {"s": float(np.median((Gc - p0) @ n)) / CAMERA_HEIGHT_M, "fr": seq.floor_frame(src(s["keys"][0]))}
    merged = {}
    for e, r in ag.regions(pick, seq, src).items():
        cid = alias.get(e, e)
        while cid in alias and alias[cid] != cid:
            cid = alias[cid]
        m = merged.setdefault(cid, {"P": [], "px": 0, "ok": 0})
        m["px"] += r["px"]
        m["ok"] += r["ok"]
        m["P"] += r["P"]
    out = []
    for c in oc["cards"]:
        ph = c.get("physical") or {}
        if c.get("kind") != "object" or "footprint_xy" not in ph or c["id"] not in merged or c["shot"] not in shots:
            continue
        m = merged[c["id"]]
        if not m["P"]:
            continue
        P = np.concatenate(m["P"])
        if len(P) < ag.MIN_GT_POINTS:
            continue
        wf = dict(ph.get("width") or {})
        if isinstance(wf.get("measured"), dict):  # the GT side is the one nearest the card's pooled width (a bound's value is not it)
            ph = {**ph, "width": {**wf, "value": wf["measured"]["value"]}}
        gt = ag.gt_card(P, {**c, "physical": ph}, shots[c["shot"]]["fr"])
        out.append({"report": report, "seq": name, "card": c["id"], "first_call": bool(run.get("first_call")), "cover": m["ok"] / max(m["px"], 1),
                    "s": shots[c["shot"]]["s"], "gt": gt["width"], "width": wf, "views": (c.get("views") or {}).get("n")})
    return out


def u_ns(f):
    """u without its scale part (fast_report_eval.fact's rule for a ground-truth u rule: the scale part is not multiplied)."""
    sc = float((f.get("parts") or {}).get("scale", 0.))
    return float(np.sqrt(max(float(f["u"]) ** 2 - sc ** 2, 0.)))


def judge(r):
    """One card's width under both rules -> {new: (kind, holds_a, holds_b), old: (kind, holds_a, holds_b)}."""
    f, gt, s = r["width"], r["gt"], r["s"]
    st = f.get("status")
    if "value" in f and st not in ("at least", "at most", "not observed", "not measurable") and not f.get("bound"):
        v, u = float(f["value"]), float(f["u"])
        new = ("value", abs(v - gt) <= u, abs(s * v - gt) <= s * u_ns(f))
    elif st == "at most":
        new = ("at most", gt <= f["value"], gt <= s * f["value"])
    elif st == "at least" and "value" in f:
        new = ("at least", gt >= f["value"] - f["u"], gt >= s * (f["value"] - u_ns(f)))
    else:
        new = (st or "none", None, None)
    m = f.get("measured")
    if isinstance(m, dict) and (st in ("at most", "not observed") or f.get("bound") == "at least" and "ends" in f):
        top = m["value"] + m["u"]  # round 4: every unresolved width 'at most measured + u'
        old = ("at most", gt <= top, gt <= s * top)
    else:
        old = new
    return {"new": new, "old": old}


def table(rows):
    """Per sequence: width values (n, coverage a/b), 'at most' / 'at least' / 'not observed' counts and holds, both rules."""
    out = {}
    for r in rows:
        if r["first_call"] or r["cover"] < .5:
            continue
        j = judge(r)
        t = out.setdefault(r["seq"], {"cards": 0, "new": {}, "old": {}})
        t["cards"] += 1
        for rule in ("new", "old"):
            k, ha, hb = j[rule]
            x = t[rule].setdefault(k, {"n": 0, "hold_a": 0, "hold_b": 0})
            x["n"] += 1
            x["hold_a"] += bool(ha)
            x["hold_b"] += bool(hb)
    return out


def md(tab, gt_heights=None):
    s = "| sequence | cards | rule | width values: n, within +-u (a) / (b) | 'at most': n, hold (a) / (b) | 'at least': n, hold (a) / (b) | 'not observed' / 'not measurable' |\n|---|---|---|---|---|---|---|\n"
    for q, t in sorted(tab.items()):
        for rule, name in (("old", "round 4: at most = measured + u"), ("new", "r5b: at most only with both ends free")):
            x = t[rule]
            cell = lambda k: (f"{x[k]['n']}, {x[k]['hold_a']} / {x[k]['hold_b']}" if k in x else "0")  # noqa: E731
            no = sum(x[k]["n"] for k in x if k in ("not observed", "not measurable", "none"))
            s += f"| {q} | {t['cards']} | {name} | {cell('value')} | {cell('at most')} | {cell('at least')} | {no} |\n"
    return s


def self_check():
    f = {"value": .5, "u": .1, "parts": {"scale": .05}}
    assert judge({"width": f, "gt": .55, "s": 1.})["new"] == ("value", True, True)
    am = {"status": "at most", "bound": "at most", "value": .3, "u": 0., "measured": {"value": .1, "u": .2}, "ends": {"lo": "free", "hi": "free"}}
    assert judge({"width": am, "gt": .2, "s": .8})["new"] == ("at most", True, True) and judge({"width": am, "gt": .25, "s": .8})["new"] == ("at most", True, False)
    no = {"status": "not observed", "measured": {"value": .1, "u": .2}, "ends": {"lo": "free", "hi": "occupied"}}
    j = judge({"width": no, "gt": .5, "s": 1.})
    assert j["new"][0] == "not observed" and j["old"] == ("at most", False, False), j
    al = {"status": "at least", "bound": "at least", "value": .4, "u": .1, "parts": {"scale": .1}, "measured": {"value": .4, "u": .5}, "ends": {"lo": "occupied", "hi": "free"}}
    j = judge({"width": al, "gt": 1.2, "s": 1.})
    assert j["new"] == ("at least", True, True) and j["old"] == ("at most", False, False), j
    print("r5b width GT self-check ok: values, both rules' bounds, round 4's 'at most' recovered from 'measured'")


if __name__ == "__main__":
    if sys.argv[1:] == ["--self-check"]:
        self_check()
        sys.exit()
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--md", type=Path)
    a = ap.parse_args()
    rows = []
    for rd in a.runs:
        for rj in sorted((rd / "reports").glob("*/run.json")):
            if json.loads(rj.read_text()).get("error"):
                continue
            rows += card_rows(rd, rj.parent.name, a.inputs)
            print(rj.parent.name, len(rows), flush=True)
    tab = table(rows)
    a.out.write_text(json.dumps({"table": tab, "rows": rows}, indent=1, default=float))
    if a.md:
        a.md.write_text(md(tab))
    print(md(tab))
