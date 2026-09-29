"""Hazard-judge thresholds (fast_report.hazard.fit) from X8 set d (whole frames, agent-labelled) and the agent-labelled object
items of mvp2-judge-hazard-001 (labels-agent.jsonl; the last line per item wins, revisions are marked), per decider:
'gemini' (v2 answers of hazard-002 on the object items; X8's stated-p answers on set d) and 'qwen' (round-1 option-letter
answers on the same objects; mvp-b-judge-decider-001 on set d). Writes the 'hazard' section of fast_report/calibration.json
and a held-out report (leave one source clip out).

  python scripts/hazard_calibrate.py [--write]
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO)]
from fast_report import hazard  # noqa: E402

RUNS = Path("/Users/adam/Desktop/panoptes-public/research-notes/phase2/runs")
X8 = RUNS / "fx-x8-jev-001"
SITES = ("me340", "samsclub", "walmart")


def labels():
    out = {}
    for line in (RUNS / "mvp2-judge-hazard-001/labels-agent.jsonl").read_text().splitlines():
        x = json.loads(line)
        out[(x["site"], x["id"], x["q"])] = x
    return out


def answers(run):
    out = {}
    for s in SITES:
        for k, v in json.loads((RUNS / run / f"answers-{s}.json").read_text())["answers"].items():
            i, q = k.split("|")
            out[(s, i, q)] = v["p"]
    return out


def sets(object_run="mvp2-judge-hazard-002"):
    lab = labels()
    d = json.loads((X8 / "sets/d.json").read_text())["items"]
    gd = json.loads((X8 / "gemini-d.json").read_text())["answers"]
    qd = {x["id"]: x for x in json.loads((RUNS / "mvp-b-judge-decider-001/answers.json").read_text())["items"]}
    g = [{"question": x["question_id"], "p": gd[x["id"]]["p_yes"], "truth": x["truth"], "source": "setd-" + x["source"], "set": "d"} for x in d if x["id"] in gd]
    q = [{"question": x["question_id"], "p": qd[x["id"]]["p"], "truth": x["truth"], "source": "setd-" + x["source"], "set": "d"} for x in d
         if qd.get(x["id"], {}).get("p") is not None]
    ga = answers(object_run)
    g += [{"question": k[2], "p": ga[k], "truth": v["label"], "source": k[0], "set": "objects", "id": k[1]} for k, v in lab.items() if k in ga]
    for x in json.loads((RUNS / "mvp2-judge-hazard-001/qwen-round1-object-answers.json").read_text()):
        k = (x["site"], x["id"], x["q"])
        if k in lab:
            q.append({"question": x["q"], "p": x["p"], "truth": lab[k]["label"], "source": x["site"], "set": "objects", "id": x["id"]})
    return {"gemini": g, "qwen": q}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    s = sets()
    f = hazard.fit(s)
    # the object-level held-out view: thresholds fitted on set d + the other two videos, applied to each video's objects
    by_video = {}
    for dec, items in s.items():
        for q in sorted({x["question"] for x in items}):
            xs = [x for x in items if x["question"] == q and x["truth"] in (0, 1)]
            res = {"hazard_calls": 0, "hazard_right": 0, "clear_calls": 0, "clear_right": 0, "n": 0, "positives": 0, "by": {}}
            for v in SITES:
                test = [x for x in xs if x["source"] == v]
                if not test:
                    continue
                t = (hazard.validate if dec == "gemini" else hazard.thresholds)([x for x in xs if x["source"] != v])
                th, tc = (t["hazard"] or {}).get("t", 2.), (t["clear"] or {}).get("t", -1.)
                b = {"n": len(test), "positives": sum(x["truth"] for x in test), "thresholds": [None if th > 1 else th, None if tc < 0 else tc],
                     "hazard_calls": sum(x["p"] >= th for x in test), "hazard_right": sum(x["p"] >= th and x["truth"] == 1 for x in test),
                     "clear_calls": sum(x["p"] <= tc for x in test), "clear_right": sum(x["p"] <= tc and x["truth"] == 0 for x in test)}
                res["by"][v] = b
                for k in ("n", "positives", "hazard_calls", "hazard_right", "clear_calls", "clear_right"):
                    res[k] += b[k]
            if res["n"]:
                by_video.setdefault(dec, {})[q] = res
    report = {"note": "labels agent-made (contact sheets); set d = X8 whole frames, other wording and input budget; object items = "
                      "hazard-002 (Gemini v2) and round-1 Qwen answers on the same objects. Held out: leave one source clip out.",
              "fit": f, "held_out_by_video": by_video}
    if a.out:
        a.out.write_text(json.dumps(report, indent=1))
    for dec in f:
        for q, r in f[dec].items():
            h = r["held_out"]
            print(dec, q, "n", r["n"], "pos", r["positives"], "hazard", r["hazard"], "clear", r["clear"], "veto", r["veto"],
                  "| held-out: hazard", h["hazard_right"], "/", h["hazard_calls"], "recall", h["hazard_recall"], "clear", h["clear_right"], "/", h["clear_calls"])
    if a.write:
        path = REPO / "fast_report/calibration.json"
        cal = json.loads(path.read_text())
        cal["hazard"] = {dec: {q: {k: r[k] for k in ("hazard", "clear", "veto", "n", "positives", "sources")} for q, r in qs.items()} for dec, qs in f.items()}
        cal["hazard_note"] = ("hazard judge thresholds (fast_report.hazard.fit): 'hazard' = calibrated precision >= 0.8 (may join a measured "
                              "value to make a FAIL), 'clear' = NPV >= 0.97, 'veto' = the p at which a PASS becomes NEEDS_REVIEW (uncalibrated "
                              "0.8 where no positives exist). Fitted on X8 set d + agent-labelled object items (hazard-001 labels, Gemini v2 "
                              "answers hazard-002, Qwen round-1 answers); scripts/hazard_calibrate.py")
        path.write_text(json.dumps(cal, indent=1))
        print("written", path)
