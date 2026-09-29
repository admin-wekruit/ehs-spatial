"""MVP B: the judgement layer of recorded fast-report runs -> summary.json + summary.md (small; every number recomputable
from the run folders' patches and run.json). Per call: layer times (written, from the MP4 bytes in the container), the judge's
stages and GPU peaks, verdict counts per check and per object type, the VLM's raw answers per question, and the people's foot
surfaces (J3a's cue).

  python scripts/mvp_judge_summary.py OUT_DIR RUN_DIR [RUN_DIR ...]

'recomputed' = judge.evaluate of this checkout on the same recorded layers (stand-in cards rebuilt from them): the verdicts a
rules fix after the run gives; the VLM answers cannot change them while no question is calibrated (all advisory).
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

V = ("FAIL", "NEEDS_REVIEW", "PASS", "NO_DATA")


def patches(report_dir):
    out = defaultdict(list)
    for p in sorted((report_dir / "patches").glob("*.json")):
        d = json.loads(p.read_text())
        out[d["layer"]].append(d)
    return out


def one(run_dir, row):
    rep = run_dir / "reports" / row["report"]
    run = json.loads((rep / "run.json").read_text())
    pt = patches(rep)
    written = json.loads((rep / "written.json").read_text()) if (rep / "written.json").exists() else {}
    w = lambda p: written.get(str(p["seq"]))  # noqa: E731
    judg = pt.get("judgements", [])
    obj = pt["objects"][0] if pt.get("objects") else None
    times = {"objects_v1": w(obj) if obj else None, "outlines": w(pt["outlines"][0]) if pt.get("outlines") else None,
             "judgements_v1": w(judg[0]) if judg else None, "judgements_v2": w(judg[-1]) if len(judg) > 1 else None}
    for k in ("judgements_v1", "judgements_v2"):
        if times[k] is not None and times["objects_v1"] is not None:
            times[k + "_after_objects"] = round(times[k] - times["objects_v1"], 2)
    stages = [{k: s[k] for k in ("stage", "start_s", "end_s", "s", "n", "peak_gb") if k in s} for s in run["stages"]
              if s["stage"].startswith(("judge.", "cards.stub"))]
    last = judg[-1]["data"] if judg else {}
    by_type = defaultdict(Counter)
    for r in last.get("rows", []):
        by_type[(r.get("subject_name") or "?", r["check"])][r["verdict"]] += 1
    vlm_q = defaultdict(list)
    for r in last.get("rows", []):
        v = r.get("vlm") or {}
        for q in [v, *(v.get("also") or [])] if v else []:
            for pv in q.get("per_view") or []:
                if pv.get("p_hazard_raw") is not None:
                    vlm_q[q["question"]].append(pv["p_hazard_raw"])
    feet = [p.get("foot_surface") for t in (pt["people"][-1]["data"]["tracks"] if pt.get("people") else []) for p in t["points"]]
    hs = [f["h_m"] for f in feet if f and f.get("h_m") is not None]
    contact = [f["contact"] for f in feet if f and f.get("h_m") is not None]
    vis = [f for f in feet if f and f.get("h_m") is not None and f.get("feet_visible")]
    vis_h = [f["h_m"] for f in vis if f["contact"]]
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from fast_report import cards_stub, judge
    cs, ctx = cards_stub.replay(run_dir, row["report"])
    again = judge.layer(judge.evaluate(cs, ctx), judge.load_calibration())
    by_type_again = defaultdict(Counter)
    for r in again["rows"]:
        by_type_again[(r.get("subject_name") or "?", r["check"])][r["verdict"]] += 1
    return {"report": row["report"], "first_call": row["first_call"], "error": run.get("error"),
            "recomputed": {"counts": again["counts"], "by_check": again["by_check"], "by_object_summary": dict(Counter(again["by_object"].values())),
                           "by_type": {f"{n}, {c}": dict(v) for (n, c), v in sorted(by_type_again.items(), key=lambda x: -sum(x[1].values()))}},
            "judge_summary": (run.get("summary") or {}).get("judge"), "times_s": times,
            "milestones": row.get("milestones"), "gpu_peak_gib": [{k: g[k] for k in ("gpu", "peak_gb", "at_s", "stages_active")} for g in run["gpu_peak"]],
            "flags": run["flags"], "usd_estimate": run.get("usd_estimate"), "judge_stages": stages,
            "counts": last.get("counts"), "by_check": last.get("by_check"), "calibration": last.get("calibration"), "vlm": last.get("vlm"),
            "by_object_summary": dict(Counter(last.get("by_object", {}).values())),
            "by_type": {f"{n}, {c}": dict(v) for (n, c), v in sorted(by_type.items(), key=lambda x: -sum(x[1].values()))},
            "vlm_raw_p_hazard": {q: {"n": len(p), "median": round(float(np.median(p)), 4), "share_ge_0.5": round(float(np.mean(np.array(p) >= .5)), 3),
                                     "share_ge_0.9": round(float(np.mean(np.array(p) >= .9)), 3)} for q, p in sorted(vlm_q.items())},
            "foot_surface": {"detections": len(feet), "measured": len(hs), "contact_share": round(float(np.mean(contact)), 3) if contact else None,
                             "h_m_p5_p50_p95": np.percentile(hs, [5, 50, 95]).round(3).tolist() if hs else None,
                             "feet_visible": len(vis), "feet_visible_and_contact": len(vis_h),
                             "visible_h_m_p5_p50_p95": np.percentile(vis_h, [5, 50, 95]).round(3).tolist() if vis_h else None,
                             "visible_contact_over_0.30_plus_u": sum(1 for f in vis if f["contact"] and f["h_m"] - f["u_m"] > .3),
                             "note": "nobody stands on an object in these clips (the reference check of J3a's cue): every visible, "
                                     "resting foot should read 0 m within u"},
            "evidence_images": sum(1 for r in last.get("rows", []) for e in r.get("evidence", []) if e.get("image"))}


def main(out, runs):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    res = []
    for rd in map(Path, runs):
        summ = json.loads((rd / "summary.json").read_text())
        for row in summ["runs"]:
            res.append({"run": rd.name, **one(rd, row)})
    (out / "summary.json").write_text(json.dumps({"schema": "mvp-b-judge-summary-v1", "calls": res}, indent=1, default=str))
    lines = ["# MVP B judgement runs", "", "Times: seconds from the MP4 bytes in the container to the layer written (Volume commit returned).", "",
             "| run | call | objects v1 | judgements v1 (+obj) | judgements v2 (+obj) | questions | GPU peaks GiB | flags | USD est. |", "|---|---|---|---|---|---|---|---|---|"]
    for r in res:
        t = r["times_s"]
        lines.append(f"| {r['run']} | {'first' if r['first_call'] else 'warm'} | {t['objects_v1']} | {t['judgements_v1']} ({t.get('judgements_v1_after_objects')}) | "
                     f"{t['judgements_v2']} ({t.get('judgements_v2_after_objects')}) | {(r['vlm'] or {}).get('questions')} | "
                     f"{' / '.join(str(g['peak_gb']) for g in r['gpu_peak_gib'])} | {len(r['flags'])} | {r['usd_estimate']} |")
    for r in res:
        if r["first_call"]:
            continue
        lines += ["", f"## {r['run']} (warm): verdicts (in the run / recomputed with this checkout)", "", "| check | " + " | ".join(V) + " |", "|---|---|---|---|---|"]
        for c in sorted(set(r["by_check"] or {}) | set(r["recomputed"]["by_check"])):
            v, w = (r["by_check"] or {}).get(c, {}), r["recomputed"]["by_check"].get(c, {})
            lines.append(f"| {c} | " + " | ".join(f"{v.get(x, 0)} / {w.get(x, 0)}" for x in V) + " |")
        lines += ["", "| object type, check (recomputed) | " + " | ".join(V) + " |", "|---|---|---|---|---|"]
        for k, v in list(r["recomputed"]["by_type"].items())[:25]:
            lines.append(f"| {k} | " + " | ".join(str(v.get(x, 0)) for x in V) + " |")
        lines += ["", f"VLM raw p(hazard) per question: {json.dumps(r['vlm_raw_p_hazard'])}", "",
                  f"Foot surfaces (J3a): {json.dumps(r['foot_surface'])}", "", f"Judge stages: {json.dumps(r['judge_stages'])}"]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
