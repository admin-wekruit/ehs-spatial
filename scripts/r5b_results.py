"""r5b time: the results folder's tables from the runs' own records (spend from the containers' lives at list prices,
times from the Volume commits, per-GPU peaks) and the evaluators' JSON (r5b_width_gt.py, r5b_timeline_eval.py).

    python scripts/r5b_results.py --runs RUN_DIR [RUN_DIR ...] --width WIDTH.json --timeline TIMELINE.json [--gt-claims GT.json] --out OUT.json
    python scripts/r5b_results.py --self-check
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "scripts")]


def run_record(run_dir):
    """A container's calls: milestones (written_s), per-GPU peaks and flags, and the spend upper bound: its whole life (boot
    entered -> the last call's end + the 60 s scale-down window) at list prices (fast_report.instrument.usd_per_s)."""
    from fast_report.instrument import usd_per_s
    run_dir = Path(run_dir)
    boot = json.loads((run_dir / "boot.json").read_text())
    calls = []
    for rj in sorted((run_dir / "reports").glob("*/run.json"), key=lambda p: json.loads(p.read_text())["t0_unix"]):
        r = json.loads(rj.read_text())
        calls.append({"report": rj.parent.name, "site": r["call"]["site"], "label": r["call"]["options"].get("label"), "first_call": bool(r.get("first_call")),
                      "error": (r.get("error") or "")[-300:] or None, "t0_unix": r["t0_unix"], "elapsed_s": r["elapsed_s"],
                      "milestones": {k: (v or {}).get("written_s") for k, v in (r.get("milestones") or {}).items()},
                      "gpu_peak_gib": [g["peak_gb"] for g in r.get("gpu_peak", [])], "flags": r.get("flags"),
                      "over_72": [s["stage"] for s in r.get("stages", []) if any((x or 0) > 72 for x in (s.get("peak_gb") or []))]})
    end = max(c["t0_unix"] + c["elapsed_s"] for c in calls)
    life = end - boot["entered_unix"] + 60
    return {"run": run_dir.name, "boot_ready_s": boot.get("ready_s"), "calls": calls, "life_s": round(life, 1),
            "usd_upper": round(life * usd_per_s(), 2)}


def self_check():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "reports" / "a").mkdir(parents=True)
        (d / "boot.json").write_text(json.dumps({"entered_unix": 100., "ready_s": 90.}))
        (d / "reports" / "a" / "run.json").write_text(json.dumps({"call": {"site": "s", "options": {"label": "on"}}, "first_call": True, "t0_unix": 200.,
                                                                     "elapsed_s": 40., "milestones": {"cards_v1": {"written_s": 30.}},
                                                                     "gpu_peak": [{"peak_gb": 60.}, {"peak_gb": 50.}], "stages": [], "flags": []}))
        r = run_record(d)
        assert r["life_s"] == 200. and r["calls"][0]["milestones"]["cards_v1"] == 30. and r["usd_upper"] > 0
    print("r5b results self-check ok: container life -> spend, calls' Volume-commit milestones and peaks")


if __name__ == "__main__":
    if sys.argv[1:] == ["--self-check"]:
        self_check()
        sys.exit()
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", type=Path, required=True)
    ap.add_argument("--width", type=Path)
    ap.add_argument("--timeline", type=Path)
    ap.add_argument("--gt-claims", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    out = {"runs": [run_record(r) for r in a.runs]}
    out["usd_upper_total"] = round(sum(r["usd_upper"] for r in out["runs"]), 2)
    for k, p in (("width", a.width), ("timeline", a.timeline), ("gt_claims", a.gt_claims)):
        if p:
            d = json.loads(p.read_text())
            out[k] = {kk: v for kk, v in d.items() if kk != "rows"} if isinstance(d, dict) else d
    a.out.write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({"usd_upper_total": out["usd_upper_total"], "runs": [(r["run"], r["life_s"], r["usd_upper"]) for r in out["runs"]]}, indent=1))
