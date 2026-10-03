"""mvp3/judge: a fresh audit of a bench run's final judgements (round-2 review D7: no carried labels). Every PASS / FAIL row
of each call (one item per stack for J1, per object and check otherwise) is sampled with a new seed and drawn on blind
contact sheets (the check's question and the object's 2 x 2 evidence, never the verdict); the agent labels every sheet by eye
(labels.jsonl: {n, label 0|1|null, note}); `score` joins them back and reports per call and check the PASS / FAIL precision.

  python scripts/judge_audit_mvp3.py sheets --runs RUN[,RUN..] --out DIR [--seed 2027] [--per-call 30]
  python scripts/judge_audit_mvp3.py score --out DIR
"""
import argparse
import glob
import json
import random
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "scripts")]
import judge_eval_r2 as je  # noqa: E402
import judge_offline as jo  # noqa: E402

je.LABEL_Q["J9"] = "q9"
je.QTEXT.update(q9="low object lying on the floor where people walk (trip)?",
                h25="the stack [1] is part of (floor up): top above 2.5 m? / off the floor: taller than 2.5 m?")


def final_rows(run, report):
    js = [json.loads(Path(f).read_text())["data"] for f in sorted(glob.glob(str(run / "mirror/reports" / report / "patches/*-judgements.json")))]
    return ([d for d in js if d.get("vlm_answers")] or js)[-1]["rows"]


def items(rows):
    """PASS / FAIL rows, one per stack (J1) or per object and check."""
    out, seen = [], set()
    for r in rows:
        if r["verdict"] not in ("PASS", "FAIL") or r["check"] not in je.LABEL_Q:
            continue
        key = (r["check"], (r["geometry"] or {}).get("stack_top") or r["subject"], r["verdict"])
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def sheets(a):
    rng, index = random.Random(a.seed), []
    for run in a.runs.split(","):
        run = jo.RUNS / run
        for f in sorted(glob.glob(str(run / "call-*.json"))):
            c = json.loads(Path(f).read_text())
            report, kind, site = c["run"]["report"], c["kind"], c["site"]
            todo = items(final_rows(run, report))
            fails = [r for r in todo if r["verdict"] == "FAIL"]
            passes = [r for r in todo if r["verdict"] == "PASS"]
            rng.shuffle(passes)
            pick = fails + passes[:a.per_call]
            rng.shuffle(pick)
            d = jo.load(run=run, report=report)
            idx = je.sheets(pick, d, a.out / "sheets", f"{site}-{kind}")
            by = {(r["subject"], je.LABEL_Q[r["check"]]): r for r in pick}
            for x in idx:
                r = by[(x["id"], x["q"])]
                index.append({**x, "run": run.name, "kind": kind, "row": r["id"], "verdict": r["verdict"], "of_pass": len(passes), "of_fail": len(fails)})
            print(site, kind, "PASS items", len(passes), "FAIL items", len(fails), "sheeted", len(idx), flush=True)
    (a.out / "index.json").write_text(json.dumps(index, indent=1))


def score(a):
    index = json.loads((a.out / "index.json").read_text())
    lab = {}
    for line in (a.out / "labels.jsonl").read_text().splitlines():
        x = json.loads(line)
        lab[(x["sheet"], x["n"])] = x
    out = {}
    for x in index:
        call = f"{x['call'].split('-')[0]} {x['kind']}"
        rec = out.setdefault(call, {"of_pass": x["of_pass"], "of_fail": x["of_fail"], "checks": {}})
        c = rec["checks"].setdefault(x["check"], Counter())
        y = (lab.get((x["sheet"], x["n"])) or {}).get("label")
        k = x["verdict"].lower()
        c[f"{k}_audited" if y is not None else f"{k}_unverifiable"] += 1
        if y is not None:
            c[f"{k}_right"] += (y == 0) if k == "pass" else (y == 1)
    for rec in out.values():
        tot = sum(rec["checks"].values(), Counter())
        rec["all"] = dict(tot)
        rec["checks"] = {k: dict(v) for k, v in rec["checks"].items()}
    (a.out / "score.json").write_text(json.dumps(out, indent=1))
    for k, v in out.items():
        t = v["all"]
        print(k, f"PASS {t.get('pass_right', 0)}/{t.get('pass_audited', 0)} (+{t.get('pass_unverifiable', 0)})",
              f"FAIL {t.get('fail_right', 0)}/{t.get('fail_audited', 0)} (+{t.get('fail_unverifiable', 0)})", v["checks"])
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=("sheets", "score"))
    ap.add_argument("--runs")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--per-call", type=int, default=30)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    sheets(a) if a.cmd == "sheets" else score(a)
