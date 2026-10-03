"""mvp3/judge: the judge on a finished run's final cards, off the GPU (judge_offline.load), with the run's own stored Gemini
hazard answers replayed (a question the run never asked stays unanswered: no Qwen), then per call: objects, objects with a
check, objects with an overall PASS / FAIL, verdict counts per check, why the others have none, and the rows whose verdict
changed against the run's final judgements.

  python scripts/judge_replay_mvp3.py --out DIR [--runs me340=RUN,samsclub-a2=RUN,walmart=RUN]
"""
import argparse
import glob
import json
import sys
from collections import Counter
from concurrent.futures import Future
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "scripts")]
import judge_offline as jo  # noqa: E402
from fast_report import judge  # noqa: E402

RUNS = {"me340": "mvp2-integrate-me340-007", "samsclub-a2": "mvp2-integrate-samsclub-a2-007", "walmart": "mvp2-integrate-walmart-007"}


def calls(run):
    """-> [(kind, report)] first, warm from the bench's call files."""
    out = []
    for f in sorted(glob.glob(str(run / "call-*.json"))):
        c = json.loads(Path(f).read_text())
        out.append((c["kind"], c["run"]["report"]))
    return out


def stored(run, report):
    """The call's final judgements (the last patch with VLM answers) -> (rows by id, {(object, question): gemini answer})."""
    js = [json.loads(Path(f).read_text())["data"] for f in sorted(glob.glob(str(run / "mirror/reports" / report / "patches/*-judgements.json")))]
    last = [d for d in js if d.get("vlm_answers")][-1]
    ans = {}
    for r in last["rows"]:
        v = r.get("vlm") or {}
        if v.get("question") and v.get("p_yes") is not None and "gemini" in (v.get("decider") or "").lower():
            ans[(r["subject"], v["question"])] = {"p": v["p_yes"], "why": v.get("why")}
    return {r["id"]: r for r in last["rows"]}, ans


def replay(ans):
    def ask(reqs):
        out = {}
        for r in reqs:
            f = Future()
            f.set_result({"status": "completed", "usage": {}, "output_text": json.dumps({"answers": [
                {"id": i, "question": q, "p_yes": ans[(i, q)]["p"], "why": ans[(i, q)]["why"]} for i, q in r["ids"] if (i, q) in ans]})})
            out[r["key"]] = f
        return out
    return ask


def unanswered(jpegs, prompt, n, priority):
    f = Future()
    f.set_result({"probs": None, "mass": 0.})
    return f


def why_none(card, ctx, cards):
    """An object with no row: its placement, as the checks read it."""
    b, pd = judge.fact(card, "base_above_floor"), judge.path_distance(card, ctx)
    if judge.class_of(card) == "not an object":
        return "not an object"
    if judge.footprint(card) is None:
        return "no footprint (2d only)"
    off = b is not None and b["value"] - b["u"] > judge.FLOOR_BASE_M
    far = pd is None or pd[0] - pd[1] > judge.NEAR_PATH_M
    return ("off the floor" if off else "on the floor" if b is not None else "no base") + (", no walked path near" if far else ", at a walked path")


def summarize(rows, cards, ctx, old=None):
    objs = [c for c in cards if c.get("kind") == "object"]
    by = {}
    for r in rows:
        by.setdefault(r["subject"], []).append(r["verdict"])
    checked = [c for c in objs if c["id"] in by]
    overall = Counter(judge.worst_verdict(dict(enumerate(by[c["id"]]))) for c in checked)
    per = {}
    for r in rows:
        per.setdefault(r["check"], Counter())[r["verdict"]] += 1
    rep = {"objects": len(objs), "objects_with_a_check": len(checked), "objects_overall": dict(overall),
           "objects_pass_or_fail": overall.get("PASS", 0) + overall.get("FAIL", 0),
           "objects_with_any_pass_or_fail": sum(1 for c in checked if {"PASS", "FAIL"} & set(by[c["id"]])),
           "rows": len(rows), "counts": dict(Counter(r["verdict"] for r in rows)), "by_check": {k: dict(v) for k, v in sorted(per.items())},
           "no_check": dict(Counter(why_none(c, ctx, cards) for c in objs if c["id"] not in by)),
           "priority": sum(1 for r in rows if r.get("priority")), "fails": [(r["id"], r["subject_name"], r["reasons"][:3]) for r in rows if r["verdict"] == "FAIL"]}
    if old is not None:
        rep["changed"] = Counter(f"{old[r['id']]['verdict'] if r['id'] in old else 'new'} -> {r['verdict']} ({r['check']})" for r in rows
                                 if r["id"] not in old or old[r["id"]]["verdict"] != r["verdict"])
        rep["dropped"] = Counter(f"{r['verdict']} ({r['check']})" for i, r in old.items() if i not in {x["id"] for x in rows})
    return rep


def main(a):
    runs = dict(x.split("=") for x in a.runs.split(",")) if a.runs else RUNS
    out = {}
    for site, run in runs.items():
        run = jo.RUNS / run
        for kind, report in calls(run):
            d = jo.load(run=run, report=report)
            old, ans = stored(run, report)
            w = judge._Writer()
            judge.run(d["cards"], d["ctx"], w, judge._Clock(), ask=unanswered, hazard_ask=replay(ans))
            rows = w.puts[-1][1]["rows"]
            rep = summarize(rows, d["cards"], d["ctx"], old)
            rep["report"] = report
            out[f"{site} {kind}"] = rep
            print(site, kind, {k: rep[k] for k in ("objects", "objects_with_a_check", "objects_pass_or_fail", "counts")}, flush=True)
            if a.out:
                a.out.mkdir(parents=True, exist_ok=True)
                (a.out / f"rows-{site}-{kind}.json").write_text(json.dumps(rows, default=str))
    if a.out:
        (a.out / "replay.json").write_text(json.dumps(out, indent=1, default=str))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--runs")
    main(ap.parse_args())
