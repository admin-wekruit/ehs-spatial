"""r4/coverage results: analysis times per call (first call after boot, warm with coverage, warm without: one container), the
added time (warm on - warm off, and the coverage stages' own seconds per GPU), per-GPU peaks (stages over 72 GiB flagged),
what the coverage pass did (densify's 'coverage' record), the card contract on every cards version, and the audit tallies.

    python scripts/r4_coverage_results.py BENCH_DIR --audit AUDIT_DIR [AUDIT_DIR ...] --out RESULTS.json
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "scripts")]
import mvp2_click_times as ct  # noqa: E402
import mvp2_results as mr  # noqa: E402
import r4_coverage_audit as au  # noqa: E402

ROWS = ("first 3D", "objects v1", "pick v1", "cards v1", "identity complete", "pick v2", "cards v3", "final judgements", "splat preview")


def per_call(bench):
    out = {}
    for f in sorted(Path(bench).glob("call-*.json")):
        c = json.loads(f.read_text())
        run = c["run"]
        t = ct.times(run)
        st = run.get("stages") or []
        cov = {}
        for s in st:
            if s["stage"].startswith("coverage."):
                cov[s["stage"]] = round(cov.get(s["stage"], 0.) + s["s"], 2)
        lift = sum(s["s"] for s in st if s["stage"] == "densify.lift")
        dens = ((run.get("summary") or {}).get("densify") or {})
        out.setdefault(c["site"], {})[c["kind"]] = {
            "report": run["report"], "error": (run.get("error") or None) and str(run["error"])[-600:], **{k: t.get(k) for k in ROWS},
            "densify_sam3_done": (run.get("marks") or {}).get("densify_sam3_done"), "densify_lift_s": round(lift, 2),
            "coverage_stage_s": cov, "gpu_peak_gib": t["gpu_peak_gib"], "over_72": t["over_72"],
            "coverage_peaks_gib": {s["stage"]: s["peak_gb"] for s in st if s["stage"].startswith("coverage.")},
            "coverage": dens.get("coverage"), "densify_new_objects": dens.get("new_objects"), "objects_final": (dens.get("cards") or {}),
            "usd_estimate": run.get("usd_estimate")}
    return out


def checks(bench):
    """mvp2_results.checks (R1 and the card contract on every cards version) per call of a bench that holds several sites."""
    out = {}
    real = mr.calls
    try:
        for f in sorted(Path(bench).glob("call-*.json")):
            c = json.loads(f.read_text())
            mr.calls = lambda d, c=c: [(c["kind"], c)]  # ponytail: one call at a time through the round 2 checker
            out.setdefault(c["site"], {})[c["kind"]] = mr.checks({c["site"]: bench})[c["site"]][c["kind"]]
    finally:
        mr.calls = real
    return out


def added(calls):
    """warm (coverage on) minus warm-off, per row."""
    on, off = calls.get("warm"), calls.get("warm-off")
    if not (on and off):
        return None
    return {k: round(on[k] - off[k], 1) for k in (*ROWS, "densify_sam3_done", "densify_lift_s") if on.get(k) is not None and off.get(k) is not None}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("bench", type=Path)
    p.add_argument("--audit", nargs="*", default=[])
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    calls = per_call(a.bench)
    summary = json.loads((a.bench / "summary.json").read_text()) if (a.bench / "summary.json").exists() else {}
    res = {"bench": str(a.bench), "calls": calls, "added_warm_on_minus_off_s": {s: added(c) for s, c in calls.items()},
           "card_checks": checks(a.bench),
           "audit": au.score(a.audit) if a.audit else None,
           "boot": json.loads((a.bench / "boot.json").read_text()) if (a.bench / "boot.json").exists() else None,
           "usd_estimate_upper": (summary.get("meta") or summary).get("usd_estimate_upper")}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({"added": res["added_warm_on_minus_off_s"], "usd": res["usd_estimate_upper"]}, indent=1))


if __name__ == "__main__":
    main()
