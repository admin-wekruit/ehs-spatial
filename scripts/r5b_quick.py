"""r5b integrate: one video's checkpoint for the user's first look (quick.json): a bench's warm call (the first call's times beside
it) and the alignment check's summary (modal_apps/r5b_align_app.py) -> times from the Volume commit (written_s) incl. when every
generated model is in, the display-model tiers of the non-simple objects, alignment IoU / offsets, VLM requests by purpose, RecGen
work and GPU peaks.

    python scripts/r5b_quick.py BENCH_DIR SITE ALIGN_DIR OUT.json
"""
import collections
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "scripts")]
import r5b_models as rm  # noqa: E402
import r5b_vocab as rv  # noqa: E402


def tiers(mirror, report):
    """Per object card its shown model's tier; the non-simple objects (routed to a generated model: a group's own or its look-alike
    copy, incl. copies whose fit check failed) and what they show."""
    P = rm.patches(mirror, report)
    cs, _ = rm.final_cards(mirror, P)
    md = ((P.get("models") or [{}])[-1].get("data") or {})
    routes, rows = md.get("routes") or {}, {r["object"]: r for r in md.get("models", [])}
    obj = [c for c in cs if c.get("kind") == "object"]
    ns = [c for c in obj if c["id"] in rows or (routes.get(c["id"]) or [""])[0] in ("generated", "observed surface")
          or "a look-alike of" in str((routes.get(c["id"]) or ["", ""])[1])]
    shown = collections.Counter("generated (own)" if c["id"] in rows and not rows[c["id"]].get("reuse_of") else "generated (look-alike copy)"
                                if c["id"] in rows else rm.tier_of(c, P) for c in ns)
    checked = [r for r in rows.values() if not r.get("reuse_of")]
    return {"object_cards": len(obj), "all_cards_by_tier": dict(collections.Counter(rm.tier_of(c, P) for c in obj)),
            "non_simple": len(ns), "non_simple_shown": dict(shown),
            "non_simple_generated_share": round((shown["generated (own)"] + shown["generated (look-alike copy)"]) / max(len(ns), 1), 3),
            "generated_models": len(checked), "copies": len(rows) - len(checked),
            "passed_view_check": sum(bool((r.get("gate") or {}).get("accepted_source_consistency")) for r in checked),
            "held_out_view": sum((r.get("gate") or {}).get("held_out", True) is not False for r in checked),
            "plan": {k: v for k, v in (md.get("plan") or {}).items() if k in ("cards", "candidates", "groups", "jev_asked", "jev_s", "generated_groups",
                                                                            "kept", "members", "by_route", "soft_goods", "plan_s")},
            "reuse": md.get("reuse"), "display_rule": md.get("display"), "final": md.get("final")}


def run_times(run):
    t = rv.times(run)
    m = run.get("marks") or {}
    t["tier 0 surfaces v1"] = rv.written_after(run, "surfaces", m.get("tier0_v1_put"))
    t["tier 1 generating (mark)"] = m.get("tier1_generating")
    t["all generated models in (models final)"] = t.pop("models final")
    t["first generated model"] = t.pop("first SAM 3D model")
    return t


def gpu(run):
    return {"peak_gib": [max((x.get("peak_gb") or [0, 0])[g] or 0 for x in run["stages"]) for g in (0, 1)],
            "stages_over_72": sorted({x["stage"] for x in run["stages"] if any((p or 0) > 72 for p in x.get("peak_gb") or [])}),
            "recgen": {k: rv.stage_sum(run, k) for k in ("recgen.select", "recgen.generate", "recgen.gate", "tier1.jev")}}


def main(bench, site, align_dir, out):
    bench = Path(bench)
    recs = [json.loads(p.read_text()) for p in sorted(bench.glob("call-*.json"))]
    recs = [r for r in recs if r.get("site") == site]
    warm = next(r for r in recs if r["kind"] == "warm")
    first = next((r for r in recs if r["kind"] == "first"), None)
    run = warm["run"]
    res = {"site": site, "report": run["report"], "bench": str(bench), "rule": "times: analysis s from the MP4 bytes in the container to the "
           "Volume commit (written_s); tiers: the final layers; alignment: scripts/r5b_align.py (every object card's shown model from its best "
           "keyframe's camera vs its SAM 3 outline)",
           "times_warm": run_times(run), "times_first": run_times(first["run"]) if first else None,
           "models": tiers(bench / "mirror", run["report"]), "vlm_requests": rv.vlm_requests(run), "gpu": gpu(run),
           "vocab": {k: v for k, v in ((run.get("summary") or {}).get("vocab") or {}).items() if k in ("source", "words", "s", "qwen", "error")},
           "tier1_summary": {k: v for k, v in ((run.get("summary") or {}).get("sam3d") or {}).items() if k not in ("records", "plan")},
           "usd_estimate_call": run.get("usd_estimate")}
    a = Path(align_dir) / run["report"] / "alignment.json"
    if a.exists():
        al = json.loads(a.read_text())
        res["alignment"] = {"summary": al["summary"], "worst_generated": al.get("worst_generated"), "overlays": al.get("overlays")}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(res, indent=1, default=str))
    return res


if __name__ == "__main__":
    r = main(*sys.argv[1:5])
    print(json.dumps({k: r[k] for k in ("times_warm", "models", "gpu")}, indent=1, default=str)[:4000])
