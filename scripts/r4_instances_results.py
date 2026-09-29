"""r4/instances results page: the GPU runs (before: r4-instances-dump-001, the pipeline without the r4 rules; after:
r4-instances-final-001) scored by r4_instances_eval on their own pick maps and cards, the offline replay's ablation
(r4_instances_replay.matrix on the dumped masks), the agent audit by eye, the lift / cards timings and per-stage GPU peaks
from the runs, and the spend.

    python scripts/r4_instances_results.py --before RUNS/r4-instances-dump-001 --after RUNS/r4-instances-final-001 \
        --matrix DIR_WITH_matrix-*.json --audit AUDIT_DIR --out RUNS/r4-instances-results
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r4_instances_eval as ev  # noqa: E402

SITES = ("me340", "samsclub-a2", "walmart")
STAGES = ("dedupe", "lift", "densify.lift", "cards.v1", "cards.v3")
MARKS = ("objects_lifted", "objects_v1_put", "cards_v1_put", "pick_v2_put", "cards_v3_put")
FLAG_GIB = 72.


def calls(run_dir):
    out = []
    for p in sorted(Path(run_dir).glob("call-*.json")):
        r = json.loads(p.read_text())
        out.append(r)
    return out


def score_call(run_dir, rec, dl_cache):
    run = rec["run"]
    maps, entities, cards = ev.run_maps(Path(run_dir) / "mirror", run["report"])
    parent = {c["id"]: c["part_of"]["id"] for c in cards["cards"] if (c.get("part_of") or {}).get("kind") == "part"}
    res, dl = ev.evaluate(rec["site"], maps, entities, cards.get("aliases") or {}, parent, dl=dl_cache.get(rec["site"]))
    dl_cache[rec["site"]] = dl
    shown = [c for c in cards["cards"] if c["kind"] == "object"]
    stages = {}
    for st in run["stages"]:
        name = st["stage"]
        if name in STAGES:
            x = stages.setdefault(name, {"s": 0., "peak_gib": [0., 0.]})
            x["s"] = round(x["s"] + st["s"], 3)
            x["peak_gib"] = [max(a, b or 0.) for a, b in zip(x["peak_gib"], st.get("peak_gb") or [0, 0])]
    peaks = [{"gpu": g["gpu"], "peak_gib": g["peak_gb"], "at_s": g["at_s"], "over_72": g["peak_gb"] > FLAG_GIB} for g in run["gpu_peak"]]
    stage_peaks = {}
    for st in run["stages"]:
        for gi, pk in enumerate(st.get("peak_gb") or []):
            if pk and pk > FLAG_GIB:
                stage_peaks.setdefault(st["stage"].split("@")[0], []).append((gi, pk))
    lift_rec = (run.get("summary") or {}).get("lift") or []
    return {"report": run["report"], "kind": rec["kind"], "first_call_after_boot": run["boot"].get("first_call_after_boot"),
            "eval": {k: {kk: v[kk] for kk in ("delivered", "covered", "missed", "in_pieces", "pieces_per_covered_mean", "wrong_merge_cards",
                                              "delivered_in_wrong_merges", "wholes_over_several")}
                     for k, v in res.items() if isinstance(v, dict) and "covered" in v},
            "frames_shared": res["frames_shared"], "cards": len(shown), "parts": sum((c.get("part_of") or {}).get("kind") == "part" for c in shown),
            "contents": sum((c.get("part_of") or {}).get("kind") == "contents" for c in shown), "cards_stats": cards.get("stats"),
            "marks_s": {m: run["marks"].get(m) for m in MARKS}, "stages": stages, "gpu_peaks": peaks, "stages_over_72_gib": stage_peaks,
            "lift_seams": [x.get("seams") for x in lift_rec], "lift_reproject": [(x.get("reproject_links"), x.get("reproject_s")) for x in lift_rec],
            "usd_estimate": run.get("usd_estimate")}


def audit(audit_dir):
    out = {}
    for p in sorted(Path(audit_dir).glob("labels-*.json")):
        _, which, site = p.stem.split("-", 2)
        lab = json.loads(p.read_text())["labels"]
        counts = {}
        for v in lab.values():
            counts[v] = counts.get(v, 0) + 1
        out.setdefault(site, {})[which] = {"n": len(lab), "counts": counts}
    return out


def fmt(e):
    return f"{e['covered']}/{e['delivered']}, {e['in_pieces']}, {e['wrong_merge_cards']} ({e['delivered_in_wrong_merges']})"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--before", type=Path, required=True)
    p.add_argument("--after", type=Path, required=True)
    p.add_argument("--matrix", type=Path)
    p.add_argument("--audit", type=Path)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    dl_cache, gpu = {}, {}
    for tag, d in (("before", a.before), ("after", a.after)):
        for rec in calls(d):
            gpu.setdefault(rec["site"], {}).setdefault(tag, []).append(score_call(d, rec, dl_cache))
    bench = {tag: json.loads((d / "summary.json").read_text()) for tag, d in (("before", a.before), ("after", a.after))}
    matrix = {s: json.loads((a.matrix / f"matrix-final-{s}.json").read_text()) for s in SITES} if a.matrix else {}
    aud = audit(a.audit) if a.audit else {}
    out = {"gpu": gpu, "matrix": {s: {k: {kk: {x: v[kk][x] for x in ("delivered", "covered", "missed", "in_pieces", "wrong_merge_cards",
                                                                      "delivered_in_wrong_merges", "wholes_over_several")}
                                          for kk in ("models", "all", "clean")} | {"cards": v["cards"], "parts": v["parts"], "build_s": v["cards_build_s"]}
                                      for k, v in m.items()} for s, m in matrix.items()},
           "audit": aud, "spend_usd_upper": {tag: b.get("usd_estimate_upper") for tag, b in bench.items()}}
    (a.out / "summary.json").write_text(json.dumps(out, indent=1, default=str))
    lines = ["| video | run | call | cards (parts, contents) | models: covered, in pieces, wrong merges (delivered in them) | clean: same | all: same | lift s | cards v1 at s | cards v3 at s | peak GiB (GPU0, GPU1) |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in SITES:
        for tag in ("before", "after"):
            for r in gpu.get(s, {}).get(tag, []):
                e = r["eval"]
                lines.append(f"| {s} | {tag} | {r['kind']}{' (first after boot)' if r['first_call_after_boot'] else ''} | {r['cards']} ({r['parts']}, {r['contents']}) | "
                             f"{fmt(e['models'])} | {fmt(e['clean'])} | {fmt(e['all'])} | {r['stages'].get('lift', {}).get('s')} | {r['marks_s']['cards_v1_put']} | "
                             f"{r['marks_s']['cards_v3_put']} | {', '.join(str(g['peak_gib']) for g in r['gpu_peaks'])} |")
    (a.out / "tables.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
