"""Round-2 judgements on the round-1 warm runs, off the GPU, audited against agent labels: the judge's rules on the final
cards (judge_offline.load), the hazard judge replayed from stored answers (Gemini v2: runs/mvp2-judge-hazard-002; Qwen:
the round-1 runs' own letter probabilities), each video's cuts fitted WITHOUT that video (set d + the other two videos:
held out), then per check: coverage, verdict counts, audited precision of PASS and FAIL (labels: mvp2-judge-hazard-001/
labels-agent.jsonl, agent-made from contact sheets), and round 1's verdicts on the same rows with the same labels.

  python scripts/judge_eval_r2.py --out RUNS/mvp2-judge-eval-001 [--decider gemini|qwen] [--cuts heldout|all]
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
import hazard_calibrate as hc  # noqa: E402
import judge_offline as jo  # noqa: E402
from fast_report import hazard, judge  # noqa: E402

RUNS = jo.RUNS
LABEL_Q = {"J1": "h25", "J2": "q2", "J4": "q1", "J5": "q4", "J6": "q6", "J7": "q7"}


def truth(lab, site, row):
    """-> 1 hazard present, 0 absent, None unverifiable / not labelled."""
    q = LABEL_Q.get(row["check"])
    x = lab.get((site, row["subject"], q)) if q else None
    if x is None:
        return None
    v = x["label"]
    if q == "h25":
        return {"below": 0, "above": 1}.get(v)
    return v if v in (0, 1) else None


def cuts_for(site, mode):
    """calibration.json's hazard section as fitted without `site` (held out) or on everything."""
    sets = hc.sets()
    if mode == "heldout":
        sets = {d: [x for x in xs if x["source"] != site] for d, xs in sets.items()}
    return {"hazard": hazard.fit(sets)}


def qwen_round1(site):
    """{(row id, question): {p, ...}} from the round-1 run's last judgements with VLM answers."""
    run, rep = jo.WARM[site]
    out = {}
    for f in sorted(glob.glob(str(RUNS / run / "mirror/reports" / rep / "patches/*-judgements.json"))):
        p = json.loads(Path(f).read_text())["data"]
        if not p.get("vlm_answers"):
            continue
        for r in p["rows"]:
            v = r.get("vlm") or {}
            for a in [v] + list(v.get("also") or []):
                ps = [x.get("p_hazard_raw") for x in a.get("per_view") or [] if x.get("p_hazard_raw") is not None]
                if a.get("question") and ps:
                    out[(r["id"], a["question"])] = {"p": max(ps), "why": None, "decider": "qwen", "keys": None}
    return out


def replay_gemini(site, run="mvp2-judge-hazard-002"):
    ans = json.loads((RUNS / run / f"answers-{site}.json").read_text())["answers"]

    def ask(reqs):
        out = {}
        for r in reqs:
            f = Future()
            f.set_result({"status": "completed", "usage": {}, "output_text": json.dumps({"answers": [
                {"id": i, "question": q, "p_yes": ans[f"{i}|{q}"]["p"], "why": ans[f"{i}|{q}"]["why"]} for i, q in r["ids"] if f"{i}|{q}" in ans]})})
            out[r["key"]] = f
        return out
    return ask


def unanswered(jpegs, prompt, n, priority):
    f = Future()
    f.set_result({"probs": None, "mass": 0.})
    return f


def round1_rows(site):
    run, rep = jo.WARM[site]
    js = sorted(glob.glob(str(RUNS / run / "mirror/reports" / rep / "patches/*-judgements.json")))
    return {r["id"]: r for r in json.loads(Path(js[-1]).read_text())["data"]["rows"]}


def audit(rows, lab, site):
    out = {}
    for r in rows:
        c = out.setdefault(r["check"], {"rows": 0, "verdicts": Counter(), "pass_audited": 0, "pass_right": 0, "pass_unverifiable": 0,
                                        "fail_audited": 0, "fail_right": 0, "fail_unverifiable": 0, "false_pass": [], "false_fail": []})
        c["rows"] += 1
        c["verdicts"][r["verdict"]] += 1
        if r["verdict"] in ("PASS", "FAIL"):
            t = truth(lab, site, r)
            k = "pass" if r["verdict"] == "PASS" else "fail"
            if t is None:
                c[f"{k}_unverifiable"] += 1
            else:
                c[f"{k}_audited"] += 1
                right = (t == 0) if k == "pass" else (t == 1)
                c[f"{k}_right"] += right
                if not right:
                    c["false_pass" if k == "pass" else "false_fail"].append(r["subject"])
    return out


def main(a):
    lab = hc.labels()
    report = {"decider": a.decider, "cuts": a.cuts, "sites": {}}
    for site in jo.WARM:
        d = jo.load(site)
        cal = {**judge.load_calibration(), **cuts_for(site, a.cuts)}
        carried = {}
        if a.decider == "qwen":
            carried = {("qwen", rid, q): v for (rid, q), v in qwen_round1(site).items()}
        w = jo_writer()
        stats = judge.run(d["cards"], d["ctx"], w, judge._Clock(), ask=unanswered, cal=cal, carried=carried,
                          hazard_ask=replay_gemini(site) if a.decider == "gemini" else None)
        rows = w.puts[-1][1]["rows"]
        objects = [c for c in d["cards"] if c.get("kind") == "object"]
        checked = {r["subject"] for r in rows if not r["subject"].startswith("person")}
        r1 = round1_rows(site)
        r1_same = [r1[r["id"]] for r in rows if r["id"] in r1]
        rep = {"objects": len(objects), "objects_with_a_check": len(checked), "coverage": round(len(checked) / max(1, len(objects)), 3),
               "rows": len(rows), "counts": dict(Counter(r["verdict"] for r in rows)), "stats": {k: v for k, v in stats.items() if not isinstance(v, dict)},
               "by_check": {k: {**v, "verdicts": dict(v["verdicts"])} for k, v in audit(rows, lab, site).items()},
               "round1_same_rows": {k: {**v, "verdicts": dict(v["verdicts"])} for k, v in audit(r1_same, lab, site).items()},
               "round1_all": {"rows": len(r1), "counts": dict(Counter(r["verdict"] for r in r1.values()))},
               "changed": [{"id": r["id"], "verdict": r["verdict"], "reasons": r["reasons"][-3:], "vlm": {k: (r.get("vlm") or {}).get(k) for k in ("decider", "p_yes", "answer")}}
                           for r in rows if r["verdict"] in ("FAIL",) or ((r.get("vlm") or {}).get("answer") in ("hazard", "likely"))]}
        report["sites"][site] = rep
        print(site, rep["coverage"], rep["counts"], {k: (v["verdicts"], f"PASS {v['pass_right']}/{v['pass_audited']} (+{v['pass_unverifiable']} unverifiable)",
                                                         f"FAIL {v['fail_right']}/{v['fail_audited']}") for k, v in rep["by_check"].items()}, flush=True)
    if a.out:
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / f"eval-{a.decider}-{a.cuts}.json").write_text(json.dumps(report, indent=1, default=str))


def jo_writer():
    return judge._Writer()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--decider", default="gemini", choices=("gemini", "qwen"))
    ap.add_argument("--cuts", default="heldout", choices=("heldout", "all"))
    main(ap.parse_args())
