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
        if a.decider == "qwen":  # Qwen's answers by object and question, carried into every row that asks it
            qa = {(x["id"], x["q"]): x for x in hc.qwen_objects() if x["site"] == site}
            rows0 = judge.evaluate(d["cards"], d["ctx"])
            carried = {("qwen", r["id"], hazard.CHECK_Q[r["check"]]): {"p": qa[(r["subject"], hazard.CHECK_Q[r["check"]])]["p"], "why": None,
                                                                       "decider": "qwen", "keys": None}
                       for r in rows0 if hazard.CHECK_Q.get(r["check"]) and (r["subject"], hazard.CHECK_Q[r["check"]]) in qa}
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


def id_map(site, d):
    """New run's object id -> the round-1 warm run's id where the same id sits within 0.2 m (73% of ids keep their place across
    calls of one video, Sam's Club round 1): labels follow only those."""
    import numpy as np
    r1 = {c["id"]: c for c in jo.load(site)["cards"] if c.get("kind") == "object"}
    out = {}
    for c in d["cards"]:
        o = r1.get(c["id"])
        a, b = ((c.get("physical") or {}).get("box") or {}).get("center_m"), ((o or {}).get("physical") or {}).get("box", {}).get("center_m")
        if o is not None and a and b and np.linalg.norm(np.subtract(a, b)) < .2 and c.get("shot") == o.get("shot"):
            out[c["id"]] = c["id"]
    return out


def audit_run(run, report, site, lab, extra=None):
    """A bench run's last judgements, audited: labels of the round-1 object at the same place, else `extra` labels made on this
    run's own sheets (labels-run.jsonl). -> the site record and the unlabelled PASS / FAIL rows."""
    d = jo.load(run=run, report=report)
    js = sorted(glob.glob(str(Path(run) / "mirror/reports" / report / "patches/*-judgements.json")))
    data = json.loads(Path(js[-1]).read_text())["data"]
    rows = data["rows"]
    m = id_map(site, d)
    usable = {k: v for k, v in lab.items() if k[0] == site and k[1] in m}
    usable.update({(site, x["id"], x["q"]): x for x in (extra or {}).values() if x.get("report") == report})  # this call's own sheets
    objects = [c for c in d["cards"] if c.get("kind") == "object"]
    checked = {r["subject"] for r in rows if not r["subject"].startswith("person")}
    by_obj = Counter(v for k, v in (data.get("by_object") or {}).items() if not k.startswith("person"))
    rep = {"report": report, "objects": len(objects), "objects_with_a_check": len(checked), "coverage": round(len(checked) / max(1, len(objects)), 3),
           "objects_by_summary": dict(by_obj), "objects_decided": by_obj.get("PASS", 0) + by_obj.get("FAIL", 0),
           "rows": len(rows), "counts": dict(Counter(r["verdict"] for r in rows)), "vlm": data.get("vlm"),
           "by_check": {k: {**v, "verdicts": dict(v["verdicts"])} for k, v in audit(rows, usable, site).items()},
           "id_kept": len(m)}
    todo = [r for r in rows if r["verdict"] in ("PASS", "FAIL") and truth(usable, site, r) is None and LABEL_Q.get(r["check"])
            and (site, r["subject"], LABEL_Q[r["check"]]) not in usable]
    return rep, todo, d


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--decider", default="gemini", choices=("gemini", "qwen"))
    ap.add_argument("--cuts", default="heldout", choices=("heldout", "all"))
    main(ap.parse_args())


QTEXT = {"h25": "top more than 2.5 m above the floor (stack on the floor) / taller than 2.5 m (off the floor)?",
         "q2": "stacked unstably, could fall?", "q1": "cable/hose on the floor where people walk?",
         "q4": "blocks/narrows the walkway to < 0.7 m?", "q6": "within 60 cm of a machine guard/fence?", "q7": "ladder unsafe?"}


def sheets(todo, d, out, site, per=4):
    """Blind contact sheets for rows to label: the object's 2 x 2 evidence (hazard.evidence) and the check's question, never the
    verdict. -> index rows [{n, site, id, q, check, sheet}]."""
    import cv2
    import numpy as np
    ctx = d["ctx"]
    ctx["outlines_by_frame"] = {f["sourceFrame"]: f["objects"] for f in d["outlines"]}
    by = {c["id"]: judge.follow_name(c) for c in d["cards"]}
    items, seen = [], set()
    for r in todo:
        q = LABEL_Q[r["check"]]
        if (r["subject"], q) in seen or r["subject"] not in by:
            continue
        seen.add((r["subject"], q))
        jpg, _ = hazard.evidence(by[r["subject"]], ctx, judge.frame_at, judge.marks_on, judge.som)
        if jpg is not None:
            items.append((r, q, cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)))
    out.mkdir(parents=True, exist_ok=True)
    index = []
    for k in range(0, len(items), per):
        panels = []
        for j, (r, q, img) in enumerate(items[k:k + per]):
            cap = np.full((70, 560, 3), 255, np.uint8)
            cv2.putText(cap, f"#{k + j} {site} {r['subject']} '{r.get('subject_name')}'", (6, 24), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(cap, QTEXT[q][:80], (6, 52), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 160), 1, cv2.LINE_AA)
            panels.append(np.vstack([cap, cv2.resize(img, (560, 560), interpolation=cv2.INTER_AREA)]))
            index.append({"n": k + j, "site": site.split("-")[0], "call": site, "report": d["report"], "id": r["subject"], "q": q, "check": r["check"],
                          "sheet": f"{site}-{k // per:03d}"})
        while len(panels) < per:
            panels.append(np.full_like(panels[0], 255))
        cv2.imwrite(str(out / f"{site}-{k // per:03d}.jpg"), np.vstack([np.hstack(panels[:2]), np.hstack(panels[2:])]), [cv2.IMWRITE_JPEG_QUALITY, 80])
    return index
