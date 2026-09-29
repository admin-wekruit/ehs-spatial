"""mvp2/integrate results page: one summary.json + summary.md from the integrated benches (one per video: a first call after boot
and a warm call), the ground-truth run, and the labelled audits (click audit, judgement audit, viewer check). The evaluations
that need no labels are run here with the branches' own code; the labelled ones are read from their folders.

    python scripts/mvp2_results.py OUT --bench me340=DIR,samsclub-a2=DIR,walmart=DIR --gt DIR --clicks DIR --judge DIR --viewer DIR \
        [--variant NAME=me340=DIR,...]      # e.g. the --discover benches, timed and click-audited beside
    python scripts/mvp2_results.py --self-check

Every metric is at estimated scale (floor plane + assumed 1.6 m camera height) unless it says 'true scale'; labels are
agent-made (contact sheets), not ground truth.
"""
import argparse
import glob
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "scripts")]
PHASE2 = Path("/Users/adam/Desktop/panoptes-public/research-notes/phase2")
SITES = ("me340", "samsclub-a2", "walmart")
NAME = {"me340": "ME340", "samsclub-a2": "Sam's Club", "walmart": "Walmart"}
ROWS = ["first 3D", "objects v1", "pick v1", "cards v1", "identity complete", "densify names", "final judgements", "pick v2", "cards v3",
        "first SAM 3D model", "splat preview"]
# round 1 (runs/mvp-results, warm; first call in brackets where it was reported)
ROUND1 = {"first 3D": (17.7, 15.2, 15.1), "objects v1": (36.3, 27.2, 20.2), "pick v1": (39.9, 27.2, 22.4), "cards v1": (42.0, 33.8, 25.4),
          "identity complete": (60.3, 88.3, 69.5), "final judgements": (82.2, 117.3, 55.9), "pick v2": (73.8, 56.1, 43.0),
          "cards v3": (75.5, 60.1, 45.6), "first SAM 3D model": (168.3, 159.7, 71.0), "splat preview": (223.3, 205.9, 190.8)}
OVER_GIB = 72.


def calls(bench):
    """[(kind, call record)] of a bench folder, first call first."""
    out = [json.loads(Path(f).read_text()) for f in sorted(glob.glob(str(Path(bench) / "call-*.json")))]
    return [(c["kind"], c) for c in sorted(out, key=lambda c: c["call"])]


def times(benches):
    import mvp2_click_times as ct
    out = {}
    for site, d in benches.items():
        for kind, c in calls(d):
            t = ct.times(c["run"])
            stages = c["run"].get("stages") or []
            t["stage_peaks_gib"] = {s["stage"]: s["peak_gb"] for s in stages if s.get("peak_gb")}
            t["error"] = c["run"].get("error")
            t["report"] = c["run"]["report"]
            out.setdefault(site, {})[kind] = t
    return out


def checks(benches):
    """R1 (a card's class, size check and checks follow its final name; identity_study.consistency) on the final patches, and
    the card contract (cards.contract) on every cards version."""
    from fast_report import cards, judge
    out = {}
    for site, d in benches.items():
        for kind, c in calls(d):
            rep = Path(d) / "mirror/reports" / c["run"]["report"] / "patches"
            ps = [json.loads(p.read_text()) for p in sorted(rep.glob("*.json"))]

            def cards_of(p):
                cs = p["data"]["cards"]
                return json.loads(next(Path(d).rglob(p["blobs"]["cards"]["sha256"])).read_bytes()) if cs == "blob" else cs
            cps = [p for p in ps if p["layer"] == "object_cards"]
            final = {x["id"]: x for x in cards_of(cps[-1])}
            rows = [p for p in ps if p["layer"] == "judgements"][-1]["data"]["rows"]
            bad_checks = [r["id"] for r in rows if r.get("check") not in ("J0", "J8", "J3a") and r["subject"] in final
                          and r["check"] not in judge.applicable(final[r["subject"]])]
            bad_class = [x["id"] for x in final.values() if x["kind"] == "object" and "raw" in x and cards.apply_name(json.loads(json.dumps(x))) != x]
            n = viol = 0
            for p in cps:
                cs = cards_of(p)
                n += len(cs)
                viol += sum(len(cards.contract(x)) for x in cs)
            out.setdefault(site, {})[kind] = {"final_rows_on_a_check_the_final_class_does_not_apply": len(bad_checks),
                                              "final_cards_not_derived_from_their_name": len(bad_class),
                                              "cards_versions": len(cps), "cards_checked": n, "contract_violations": viol}
    return out


def identity(benches):
    """The fresh held-out items (runs/mvp2-identity-final-001, agent-labelled blind before this branch existed) graded on the
    warm and first calls' final names."""
    import identity_study as ids
    out = {}
    for kind in ("warm", "first"):
        now = {s: (Path(d).name, dict(calls(d))[kind]["run"]["report"]) for s, d in benches.items() if kind in dict(calls(d))}
        r = ids.score_final(PHASE2 / "runs/mvp2-identity-final-001", now)
        grades = {}
        for x in r["per_item"]:
            if x["grade"] is not None:  # an item labelled 'unclear' is not graded
                grades.setdefault(x["id"].split(":")[0], []).append(x["grade"])
        out[kind] = {"by_site": {s: {"n": len(g), "right": sum(v == "right" for v in g), "close": sum(v == "close" for v in g),
                                     "wrong": sum(v == "wrong" for v in g)} for s, g in grades.items()},
                     "shares": r["final_names"], "round1_same_objects": r["round1_names_same_objects"], "hazard_names": r["hazard_names"],
                     "not_matched": r["not_matched"]}
    return out


def physical_gt(gt):
    import accuracy_gt as ag
    sc = json.loads((Path(gt) / "accuracy/score.json").read_text())
    rows = json.loads((Path(gt) / "accuracy/rows.json").read_text())
    return {"table": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in sc["table"]],
            "shots": {r["report"]: r["shots"] for r in sc["reports"]},
            "end_to_end": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in ag.end_to_end(rows)],
            "bounds_hold": [sum(b["holds_a"] for r in sc["reports"] for b in r.get("bounds", [])), sum(len(r.get("bounds", [])) for r in sc["reports"])],
            "held_out": "k_geo was fitted on the two ARKit sequences (mvp2/accuracy); TUM is held out. The runs are new (this branch's code)."}


def physical_delivered(benches):
    import mvp2_physical_eval as pe
    out = {}
    for site, d in benches.items():
        warm = dict(calls(d)).get("warm")
        if warm:
            r = pe.one_call(str(d), warm["run"]["report"], site, None)
            out[site] = {k: v for k, v in r.items() if k != "run"}
    return out


def spend(benches, gt, variants):
    dirs = list(benches.values()) + [d for v in variants.values() for d in v.values()]
    modal = {Path(d).name: json.loads((Path(d) / "summary.json").read_text()).get("usd_estimate_upper") for d in dirs if (Path(d) / "summary.json").exists()}
    if gt:
        modal[Path(gt).name] = round(sum(r.get("usd_estimate") or 0 for r in json.loads((Path(gt) / "summary.json").read_text())["runs"]), 2)
    tok = [0, 0]
    for d in dirs:
        for f in glob.glob(str(Path(d) / "namer/*/*.json")):
            u = json.loads(Path(f).read_text()).get("usage") or {}
            tok[0] += u.get("prompt_token_count") or 0
            tok[1] += (u.get("candidates_token_count") or 0) + (u.get("thoughts_token_count") or 0)
        for line in (Path(d) / "hazard-relay-events.jsonl").read_text().splitlines() if (Path(d) / "hazard-relay-events.jsonl").exists() else []:
            e = json.loads(line)
            if e.get("phase") == "budget_evidence":
                tok[0] += e["data"].get("inputTokens") or 0
    return {"modal_usd_upper": modal, "modal_usd_upper_sum": round(sum(v or 0 for v in modal.values()), 2), "gemini_tokens_in_out": tok,
            "note": "Modal list-price upper bound over each container's life (GPU queue waits included); the GT run's per-call estimates. "
                    "Gemini: input tokens of every relayed request (hazard: counted input only), not priced (no price for gemini-3.5-flash in the repo)."}


def cell(v, fmt="{:.1f}"):
    return "—" if v is None else fmt.format(v) if isinstance(v, (int, float)) else str(v)


def pair(t, row):
    w, f = (t.get("warm") or {}).get(row), (t.get("first") or {}).get(row)
    return f"{cell(w)} ({cell(f)})"


def md_times(res):
    L = ["| s from MP4 bytes in, warm (first call) | " + " | ".join(NAME[s] for s in SITES) + " | round 1 warm |", "|---|---|---|---|---|"]
    for row in ROWS:
        L.append(f"| {row} | " + " | ".join(pair(res["times"].get(s, {}), row) for s in SITES) + f" | {' / '.join(cell(x) for x in ROUND1.get(row, ())) or '—'} |")
    L.append("| GPU peaks GiB gpu0/gpu1, warm (first) | " + " | ".join(
        "{} ({})".format(*("/".join(cell(g) for g in (res["times"].get(s, {}).get(k) or {}).get("gpu_peak_gib", [])) or "—" for k in ("warm", "first")))
        for s in SITES) + " | — |")
    L.append("| stages over 72 GiB | " + " | ".join(", ".join(sorted({x for k in ("warm", "first") for x in (res["times"].get(s, {}).get(k) or {}).get("over_72", [])})) or "none"
                                             for s in SITES) + " | — |")
    return L


def self_check():
    t = {"warm": {"cards v1": 41.24, "gpu_peak_gib": [61.3, 52.7], "over_72": []}, "first": {"cards v1": 46.0, "gpu_peak_gib": [66.0, 55.0], "over_72": ["x"]}}
    assert pair(t, "cards v1") == "41.2 (46.0)" and pair(t, "identity complete") == "— (—)"
    md = md_times({"times": {"me340": t}})
    assert "| cards v1 | 41.2 (46.0) | — (—) | — (—) | 42.0 / 33.8 / 25.4 |" in md and md[-1].startswith("| stages over 72 GiB | x |"), md
    print("mvp2_results self-check ok: cells, times table")


def main(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    bench = dict(x.split("=", 1) for x in a.bench.split(","))
    variants = {}
    for v in a.variant or []:
        name, rest = v.split("=", 1)
        variants[name] = dict(x.split("=", 1) for x in rest.split(","))
    res = {"schema": "mvp2-integrate-results-v1", "bench": bench, "gt": a.gt, "variants": variants,
           "times": times(bench), "checks": checks(bench), "identity": identity(bench), "physical_gt": physical_gt(a.gt) if a.gt else None,
           "physical_delivered": physical_delivered(bench), "spend": spend(bench, a.gt, variants),
           "variant_times": {k: times(v) for k, v in variants.items()}}
    for key, path, name in (("click_audit", a.clicks, "score.json"), ("judgements", a.judge, "judge.json"), ("viewer", a.viewer, "viewer.json")):
        res[key] = json.loads((Path(path) / name).read_text()) if path and (Path(path) / name).exists() else None
    (out / "summary.json").write_text(json.dumps(res, indent=1, default=str))
    print("\n".join(md_times(res)))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", nargs="?")
    ap.add_argument("--bench")
    ap.add_argument("--gt")
    ap.add_argument("--clicks")
    ap.add_argument("--judge")
    ap.add_argument("--viewer")
    ap.add_argument("--variant", action="append")
    ap.add_argument("--self-check", action="store_true")
    a = ap.parse_args()
    self_check() if a.self_check else main(a)
