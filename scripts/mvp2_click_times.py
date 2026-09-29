"""mvp2 click: analysis times per call (first call after boot, warm, +5 s shifted) from bench call records, round 1 beside.

identity complete = the first object_cards patch queued after the identity pass for the other objects ended (vlm.identity.other),
final judgements = the last judgements patch written; both 'written' = Volume commit returned, s from the MP4 bytes in the
container (the bench's own clock). GPU peaks: max over stages per GPU, and the stages over 72 GiB.

    python scripts/mvp2_click_times.py RUN_DIR [RUN_DIR ...] [--json OUT.json]
    python scripts/mvp2_click_times.py --self-check
"""
import json
import sys
from pathlib import Path

LAYERS = [("first 3D", "cameras", 1), ("objects v1", "objects", 1), ("pick v1", "pick", 1), ("cards v1", "object_cards", 1),
          ("pick v2", "pick", 2)]
OVER_GIB = 72.


def times(run):
    """One call's run.json -> {row: s}."""
    lay = run["layers"]
    first = lambda name, v: next((x["written_s"] for x in lay if x["layer"] == name and x["version"] == v), None)  # noqa: E731
    out = {row: first(name, v) for row, name, v in LAYERS}
    st = {s["stage"]: s for s in run["stages"]}
    m = run.get("marks") or {}
    # mvp2/integrate: the first-pass names are Gemini's (mark identity_gemini_put) or the Qwen decider's (identity_qwen_*_put, or
    # the end of vlm.identity.other in round 1); densify's own objects are named later (identity_gemini_densify_put)
    other = max([m[k] for k in ("identity_gemini_put", "identity_qwen_ehs_put", "identity_qwen_other_put") if k in m] +
                [s["end_s"] for s in run["stages"] if s["stage"] == "vlm.identity.other"], default=None)
    cards = [x for x in lay if x["layer"] == "object_cards"]
    out["identity complete"] = next((x["written_s"] for x in cards if other is not None and x["queued_s"] >= other - .01), None)
    dens = max([m[k] for k in ("identity_gemini_densify_put", "identity_gemini_densify_left_put", "identity_qwen_densify_put") if k in m], default=None)
    out["densify names"] = next((x["written_s"] for x in cards if dens is not None and x["queued_s"] >= dens - .01), None)
    out["cards v3"] = next((x["written_s"] for x in cards if x["queued_s"] >= (st.get("cards.v3") or {}).get("end_s", 1e9) - .01), None)  # patch versions count puts
    out["final judgements"] = max((x["written_s"] for x in lay if x["layer"] == "judgements"), default=None)
    out["first SAM 3D model"], out["splat preview"], out["splat start"] = m.get("first_model_put"), m.get("splat_preview_put"), m.get("splat_started")
    peaks = [max((s["peak_gb"][g] or 0) for s in run["stages"] if s.get("peak_gb")) for g in (0, 1)]
    over = sorted({s["stage"] for s in run["stages"] if any((p or 0) > OVER_GIB for p in s.get("peak_gb") or [])})
    return {k: (round(v, 1) if isinstance(v, (int, float)) else v) for k, v in out.items()} | {"gpu_peak_gib": peaks, "over_72": over}


def table(run_dirs):
    rows = []
    for d in run_dirs:
        for f in sorted(Path(d).glob("call-*.json")):
            rec = json.loads(f.read_text())
            rows.append({"run": Path(d).name, "site": rec["site"], "kind": rec["kind"], "report": rec["run"]["report"],
                         "error": (rec["run"].get("error") or None) and str(rec["run"]["error"])[:200], **times(rec["run"])})
    return rows


def self_check():
    run = {"layers": [{"layer": "cameras", "version": 1, "queued_s": 10, "written_s": 12}, {"layer": "object_cards", "version": 2, "queued_s": 40, "written_s": 41},
                      {"layer": "object_cards", "version": 3, "queued_s": 50, "written_s": 52}, {"layer": "object_cards", "version": 4, "queued_s": 61, "written_s": 63},
                      {"layer": "judgements", "version": 1, "queued_s": 45, "written_s": 46}, {"layer": "judgements", "version": 3, "queued_s": 70, "written_s": 73}],
           "stages": [{"stage": "vlm.identity.other", "start_s": 45, "end_s": 60, "peak_gb": [70, 73]}, {"stage": "cards.v3", "start_s": 48, "end_s": 50, "peak_gb": [60, 50]}],
           "marks": {"splat_started": 75}}
    t = times(run)
    assert (t["identity complete"], t["cards v3"], t["final judgements"], t["first 3D"]) == (63, 52, 73, 12), t
    assert t["gpu_peak_gib"] == [70, 73] and t["over_72"] == ["vlm.identity.other"], t
    run["marks"].update(identity_gemini_put=40, identity_gemini_densify_put=61)  # Gemini names (no Qwen decider pass):
    run["stages"] = run["stages"][1:]                                              # the cards patches queued at the marks
    t = times(run)
    assert (t["identity complete"], t["densify names"]) == (41, 63), t
    print("mvp2_click_times self-check ok: identity complete, cards v3, final judgements, peaks")


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        self_check()
    else:
        args = [a for a in sys.argv[1:] if not a.startswith("--")]
        out = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None
        dirs = [a for a in args if a != out]
        rows = table(dirs)
        if out:
            Path(out).write_text(json.dumps(rows, indent=1))
        keys = ["first 3D", "objects v1", "cards v1", "identity complete", "final judgements", "pick v2", "cards v3", "first SAM 3D model",
                "splat preview", "gpu_peak_gib", "over_72"]
        print("| run | kind | " + " | ".join(keys) + " |")
        for r in rows:
            print(f"| {r['site']} | {r['kind']} | " + " | ".join(str(r.get(k)) for k in keys) + " |")
