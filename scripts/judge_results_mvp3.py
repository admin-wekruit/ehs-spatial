"""mvp3/judge results page: per video and call of the benches, the judgement coverage (objects with a check, with an overall
PASS / FAIL, verdict counts per check, why the others have none), the walked paths the judge used, the D6 contract counts,
analysis times (s from the MP4 bytes in the container) with per-GPU stage peaks, the offline replay on round 2's run 007,
the fresh audit's precision, and spend.

  python scripts/judge_results_mvp3.py --bench me340=RUN,samsclub-a2=RUN,walmart=RUN --replay DIR --audit DIR[,DIR..] --out DIR
"""
import argparse
import glob
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "scripts")]
import judge_audit_mvp3 as ja  # noqa: E402
import judge_offline as jo  # noqa: E402
import judge_replay_mvp3 as jr  # noqa: E402
import mvp2_results as mr  # noqa: E402

NAME = {"me340": "ME340", "samsclub-a2": "Sam's Club", "walmart": "Walmart"}
TIMES = ("objects v1", "cards v1", "identity complete", "densify names", "final judgements", "cards v3")


def d6(cards):
    """Sizes marked 'needs review' with u 0 (round 2: 9-14 a video), person distances whose u >= max(value, 1 m)."""
    nr = sum(1 for c in cards for f in (c.get("physical") or {}).values() if isinstance(f, dict) and f.get("status") == "needs review" and f.get("u") == 0)
    pm = sum(1 for c in cards if c.get("kind") == "person" for k in ("moved", "path_length")
             if "value" in ((c.get("physical") or {}).get(k) or {}) and c["physical"][k]["u"] >= max(abs(c["physical"][k]["value"]), 1.))
    return {"needs_review_u0": nr, "person_distance_u_over_value": pm}


def main(a):
    benches = dict(x.split("=") for x in a.bench.split(","))
    benches = {k: jo.RUNS / v for k, v in benches.items()}
    res = {"times": mr.times(benches), "checks": mr.checks(benches), "calls": {}, "spend": {}}
    for site, run in benches.items():
        s = json.loads((run / "summary.json").read_text())
        res["spend"][site] = s.get("usd_estimate_upper")
        res.setdefault("boot", {})[site] = (s.get("boot") or {}).get("ready_s")
        for kind, c in mr.calls(run):
            report = c["run"]["report"]
            d = jo.load(run=run, report=report)
            rows = ja.final_rows(run, report)
            rep = jr.summarize(rows, d["cards"], d["ctx"])
            rep.update(d6(d["cards"]), walked={str(k): sorted(v) for k, v in d["ctx"]["walked"].items()},
                       person_walked={x["id"]: (x.get("walked_path") or {}).get("used") for x in d["cards"] if x.get("kind") == "person" and "walked_path" in x},
                       report=report)
            res["calls"][f"{site} {kind}"] = rep
    res["replay_007"] = json.loads((a.replay / "replay.json").read_text()) if a.replay else None
    res["audit"] = {k: v for d in (a.audit or "").split(",") if d and (Path(d) / "score.json").exists()
                    for k, v in json.loads((Path(d) / "score.json").read_text()).items()} or None
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "results.json").write_text(json.dumps(res, indent=1, default=str))
    (a.out / "tables.md").write_text(md(res, benches))
    print((a.out / "tables.md").read_text())


def pct(n, d):
    return f"{n} / {d} ({100 * n / max(1, d):.1f}%)"


def md(res, benches):
    sites = list(benches)
    L = ["## Judgement coverage per call (final judgements patch)", "",
         "| call | objects with a check | objects with an overall PASS / FAIL | objects with any PASS / FAIL row | verdict rows | 'likely hazard' priority rows |",
         "|---|---|---|---|---|---|"]
    for k, v in res["calls"].items():
        L.append(f"| {k} | {pct(v['objects_with_a_check'], v['objects'])} | {pct(v['objects_pass_or_fail'], v['objects'])} | "
                 f"{pct(v['objects_with_any_pass_or_fail'], v['objects'])} | {json.dumps(v['counts'])} | {v['priority']} |")
    L += ["", "| call | per check | objects with no row, by placement | FAIL rows |", "|---|---|---|---|"]
    for k, v in res["calls"].items():
        L.append(f"| {k} | {json.dumps(v['by_check'])} | {json.dumps(v['no_check'])} | {len(v['fails'])} |")
    L += ["", "## Walked paths the judge used (D2), contract counts (D6)", "",
          "| call | walked paths per shot | person cards: walked path used | 'needs review' sizes with u 0 | person moved / path with u >= max(value, 1 m) | card contract violations (all versions) | rows on a check the final class does not apply |",
          "|---|---|---|---|---|---|---|"]
    for k, v in res["calls"].items():
        site, kind = k.split()
        ch = res["checks"][site][kind]
        L.append(f"| {k} | {json.dumps(v['walked'])} | {sum(1 for x in v['person_walked'].values() if x)} of {len(v['person_walked'])} | "
                 f"{v['needs_review_u0']} | {v['person_distance_u_over_value']} | {ch['contract_violations']} | {ch['final_rows_on_a_check_the_final_class_does_not_apply']} |")
    L += ["", "## Analysis time (s from the MP4 bytes in the container), warm (first call)", "",
          "| | " + " | ".join(NAME.get(s, s) for s in sites) + " |", "|---|" + "---|" * len(sites)]
    for row in TIMES:
        cells = []
        for s in sites:
            t = res["times"].get(s, {})
            w, f = (t.get("warm") or {}).get(row), (t.get("first") or {}).get(row)
            cells.append(f"{mr.cell(w)} ({mr.cell(f)})")
        L.append(f"| {row} | " + " | ".join(cells) + " |")
    peaks = []
    for s in sites:
        t = res["times"].get(s, {})
        pk = {kind: [max((v[g] for v in (t.get(kind) or {}).get("stage_peaks_gib", {}).values() if len(v) > g and v[g] is not None), default=None)
                     for g in (0, 1)] for kind in ("warm", "first")}
        over = sorted({st for kind in ("warm", "first") for st, v in ((t.get(kind) or {}).get("stage_peaks_gib") or {}).items() if any((x or 0) > mr.OVER_GIB for x in v)})
        peaks.append(f"{pk['warm']} ({pk['first']}); over 72 GiB: {', '.join(over) or 'none'}")
    L.append("| GPU peaks GiB [gpu0, gpu1] per stage max, warm (first) | " + " | ".join(peaks) + " |")
    L.append("| Modal list-price upper bound, $ | " + " | ".join(str(res["spend"].get(s)) for s in sites) + " |")
    L.append("| cold start (boot, not analysis), s | " + " | ".join(str(res.get("boot", {}).get(s)) for s in sites) + " |")
    if res.get("replay_007"):
        L += ["", "## Offline replay on round 2's run 007 cards (CPU; the run's own Gemini answers; no new questions)", "",
              "| call | objects with a check | overall PASS / FAIL | verdict rows | FAIL rows |", "|---|---|---|---|---|"]
        for k, v in res["replay_007"].items():
            L.append(f"| {k} | {pct(v['objects_with_a_check'], v['objects'])} | {pct(v['objects_pass_or_fail'], v['objects'])} | {json.dumps(v['counts'])} | {len(v['fails'])} |")
    if res.get("audit"):
        L += ["", "## Fresh audit (new seed, every sampled item looked at by eye; agent-labelled, blind to verdicts)", "",
              "| call | PASS right / audited (+ unverifiable), of PASS items | FAIL right / audited (+ unverifiable), of FAIL items | per check |", "|---|---|---|---|"]
        for k, v in res["audit"].items():
            t = v["all"]
            L.append(f"| {k} | {t.get('pass_right', 0)} / {t.get('pass_audited', 0)} (+{t.get('pass_unverifiable', 0)}), of {v['of_pass']} | "
                     f"{t.get('fail_right', 0)} / {t.get('fail_audited', 0)} (+{t.get('fail_unverifiable', 0)}), of {v['of_fail']} | {json.dumps(v['checks'])} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bench", required=True)
    ap.add_argument("--replay", type=Path)
    ap.add_argument("--audit", help="audit folders (judge_audit_mvp3.py score), comma-separated")
    ap.add_argument("--out", type=Path, required=True)
    main(ap.parse_args())
