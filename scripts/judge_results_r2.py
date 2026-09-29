"""Round-2 judgements: one page from the bench runs (timing per judgements version, per-stage GPU peaks), the audit of the final
warm calls against agent labels, the held-out offline evaluation (judge_eval_r2) and the hazard calibration report.

  python scripts/judge_results_r2.py --bench RUNS/mvp2-judge-bench-003 --fallback RUNS/mvp2-judge-bench-001 --out RUNS/mvp2-judge-results
"""
import argparse
import glob
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "scripts")]
import hazard_calibrate as hc  # noqa: E402
import judge_eval_r2 as je  # noqa: E402

SITE = {"me340": "me340", "samsclub-a2": "samsclub", "walmart": "walmart"}
FLAG_GIB = 72.


def timing(bench):
    """Per call: the judgements puts (written_s, the cards version they judge, with or without VLM answers), cards and objects v1,
    the judge's stages, per-GPU peaks per stage (flag > 72 GiB)."""
    out = []
    for f in sorted(glob.glob(str(bench / "call-*.json"))):
        c = json.loads(Path(f).read_text())
        r = c["run"]
        pdir = bench / "mirror/reports" / r["report"] / "patches"
        meta = {}
        for pf in sorted(pdir.glob("*-judgements.json")):
            p = json.loads(pf.read_text())
            meta[p["seq"]] = {"of": (p["data"].get("version_of") or {}).get("object_cards"), "vlm": p["data"].get("vlm_answers"),
                              "stats": p["data"].get("vlm"), "counts": p["data"].get("counts")}
        puts = [{"version": l["version"], "written_s": l["written_s"], **meta.get(l["seq"], {})} for l in r["layers"] if l["layer"] == "judgements"]
        first = lambda layer: next((l["written_s"] for l in r["layers"] if l["layer"] == layer), None)  # noqa: E731
        stages = [s for s in r["stages"] if s["stage"].startswith("judge") or s["stage"].startswith("vlm.identity")]
        peaks = {}
        for s in r["stages"]:
            for g, v in enumerate(s.get("peak_gb") or []):
                peaks[g] = max(peaks.get(g, 0.), v or 0.)
        out.append({"site": c["site"], "kind": c["kind"], "report": r["report"], "first_call_after_boot": (r.get("boot") or {}).get("first_call_after_boot"),
                    "objects_v1_s": first("objects"), "cards_v1_s": first("object_cards"),
                    "cards_last_s": max((l["written_s"] for l in r["layers"] if l["layer"] == "object_cards"), default=None),
                    "judgements_first_s": puts[0]["written_s"] if puts else None,
                    "judgements_first_vlm_s": next((p["written_s"] for p in puts if p.get("vlm")), None),
                    "judgements_final_s": puts[-1]["written_s"] if puts else None, "puts": puts,
                    "judge_stages": [{k: s.get(k) for k in ("stage", "start_s", "end_s", "s", "n", "peak_gb")} for s in stages],
                    "gpu_peak_gib": [round(peaks[g], 2) for g in sorted(peaks)],
                    "stage_peaks_over_72": [{"stage": s["stage"], "peak_gb": s["peak_gb"]} for s in r["stages"] if any((v or 0) > FLAG_GIB for v in s.get("peak_gb") or [])],
                    "flags": [f for f in r.get("flags") or [] if "unknown stage name" not in f], "elapsed_s": r.get("elapsed_s"),
                    "usd_estimate": r.get("usd_estimate"), "error": r.get("error")})
    return out


def audits(bench, extra):
    lab = hc.labels()
    out, todo_all = {}, {}
    for t in timing(bench):
        site = SITE[t["site"]]
        rep, todo, d = je.audit_run(bench, t["report"], site, lab, extra)
        out[f"{site} {t['kind']}"] = rep
        todo_all[f"{site} {t['kind']}"] = (todo, d, site)
    return out, todo_all


def load_extra(path):
    out = {}
    if path and path.exists():
        for line in path.read_text().splitlines():
            x = json.loads(line)
            out[(x["site"], x["id"], x["q"])] = x
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bench", type=Path, required=True)
    ap.add_argument("--fallback", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sheets", action="store_true", help="write blind contact sheets for the unlabelled PASS / FAIL rows of the warm calls")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    extra = load_extra(a.out / "labels-run.jsonl")
    res = {"bench": str(a.bench), "timing": timing(a.bench), "fallback_timing": timing(a.fallback) if a.fallback else None}
    res["audit"], todo = audits(a.bench, extra)
    if a.sheets:
        index = []
        for k, (rows, d, site) in todo.items():
            if k.endswith("warm"):
                index += je.sheets(rows, d, a.out / "sheets-run", site)
        (a.out / "sheets-run-index.json").write_text(json.dumps(index, indent=0))
        print("sheets", len(index))
    (a.out / "run-results.json").write_text(json.dumps(res, indent=1, default=str))
    for t in res["timing"]:
        print(t["site"], t["kind"], "objects", t["objects_v1_s"], "judgements first", t["judgements_first_s"], "first with VLM", t["judgements_first_vlm_s"],
              "final", t["judgements_final_s"], "peaks", t["gpu_peak_gib"], "over72", len(t["stage_peaks_over_72"]))
    for k, v in res["audit"].items():
        print(k, v["counts"], {c: (b["verdicts"], f"P {b['pass_right']}/{b['pass_audited']}+{b['pass_unverifiable']}", f"F {b['fail_right']}/{b['fail_audited']}+{b['fail_unverifiable']}")
                                for c, b in v["by_check"].items() if c not in ("J8", "J3a", "J3b")})
