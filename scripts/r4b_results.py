"""r4b integrate results: the user's four acceptance items per video (types, segmentation and clicks, physical, models), then
times, GPU peaks and VLM calls, from the final benches (one per video: a first call after boot, then a warm call), the fresh
audits by eye (clicks, cards: agent-labelled) and the instances scorer; summary.md's narrative is written by hand around tables.md.

    python scripts/r4b_results.py OUT --bench me340=DIR,samsclub-a2=DIR,walmart=DIR [--baseline me340=DIR,...] \
        [--clicks DIR] [--cards DIR] [--gt DIR] [--shots DIR]
    python scripts/r4b_results.py --self-check

--clicks DIR/<site>/ (click_audit.py sample + labels), --cards DIR/<site>/ (r4b_audit.py cards + labels), --gt a GT run
(accuracy_gt.py score wrote DIR/accuracy/score.json; a comma list), --baseline round 3's benches (the instances scorer on the same frames).
"""
import argparse
import collections
import glob
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "scripts")]
SITES = ("me340", "samsclub-a2", "walmart")
NAME = {"me340": "ME340", "samsclub-a2": "Sam's Club", "walmart": "Walmart"}
OVER_GIB = 72.


def calls(bench, site):
    out = [json.loads(Path(f).read_text()) for f in sorted(glob.glob(str(Path(bench) / "call-*.json")))]
    return {c["kind"]: c for c in sorted(out, key=lambda c: c["call"]) if c.get("site", site) == site}


def cards_versions(bench, report):
    """Every object_cards patch of a report -> [cards list] (out-of-line blobs read from the mirror)."""
    import fast_report_eval as ev
    out = []
    for p in ev.patch_versions(Path(bench) / "mirror", report, "object_cards"):
        cs = p["data"]["cards"]
        out.append(json.loads(ev.blob_bytes(Path(bench) / "mirror", p["blobs"]["cards"]["sha256"])) if cs == "blob" else cs)
    return out


def times(run):
    import mvp2_click_times as ct
    import r4_naming_results as nr
    t, n, m = ct.times(run), nr.times(run), run.get("marks") or {}
    r = lambda v: None if v is None else round(v, 1)  # noqa: E731
    return {"first 3D": t["first 3D"], "objects v1": t["objects v1"], "cards v1": t["cards v1"], "objects v3": r(m.get("objects_v3_put")),
            "cards v3": t["cards v3"], "types (first pass)": n["names (first pass)"], "types (densify pass)": n["names (densify pass)"],
            "first SAM 3D model": r(m.get("first_model_put")), "models final": r(m.get("models_final_put")), "splat preview": t["splat preview"],
            "call end": r(run.get("elapsed_s")), "gpu_peak_gib": t["gpu_peak_gib"], "over_72": t["over_72"]}


def vlm_calls(run):
    """Every VLM request of a call by purpose: naming (the cascade's medoid questions), scene vocabulary, event captions, hazard."""
    s = run.get("summary") or {}
    st = collections.Counter(x["stage"] for x in run["stages"])
    n = {x["stage"]: x.get("n") or {} for x in run["stages"]}
    return {"naming (cluster medoids)": sum(p.get("vlm_questions", 0) for p in (s.get("naming") or {}).get("passes", [])),
            "naming escalated": sum(p.get("vlm_questions_escalated", 0) for p in (s.get("naming") or {}).get("passes", [])),
            "scene vocabulary (one request)": st.get("vlm.vocab", 0), "event captions (windows)": (n.get("vlm.events") or {}).get("windows", 0),
            "hazard / judge": st.get("judge.qwen", 0) + st.get("judge.gemini", 0) + st.get("judge.vlm", 0),
            "qwen decider (identity_vlm)": st.get("vlm.identity", 0) + st.get("vlm.identity.ehs", 0) + st.get("vlm.identity.other", 0),
            "question log": s.get("vlm_questions")}


def models(bench, report):
    import fast_report_eval as ev
    ps = ev.patch_versions(Path(bench) / "mirror", report, "models")
    if not ps:
        return None
    d = ps[-1]["data"]
    return {"sam3d_accepted": len(d.get("models") or []), "eligible": d.get("eligible"), "attempted": d.get("attempted"),
            "first_pass": d.get("first_pass"), "final": d.get("final")}


def per_call(bench, site, kind, rec):
    import fast_report_eval as ev
    import r4b_audit as ra
    from fast_report import cards as fc
    run = rec["run"]
    L = ev.load_layers(Path(bench) / "mirror", run["report"])
    vs = cards_versions(bench, run["report"])
    comp = fc.completeness(L["object_cards"]["cards"])
    return {"report": run["report"], "error": run.get("error"), "first_call_after_boot": (run.get("boot") or {}).get("first_call_after_boot"),
            "share": {k: v for k, v in ra.share(L).items() if k != "completeness"},
            "physical": {"complete_share": {k: v["complete_share"] for k, v in comp.items()}, "cards": {k: v["cards"] for k, v in comp.items()},
                         "with_number_share_objects": (comp.get("object") or {}).get("with_number_share"),
                         "contract_violations": sum(len(fc.contract(x)) for cs in vs for x in cs), "cards_versions": len(vs)},
            "models": models(bench, run["report"]), "times": times(run), "vlm": vlm_calls(run),
            "stage_peaks_gib": {x["stage"]: x["peak_gb"] for x in run["stages"] if x.get("peak_gb")},
            "naming": {k: v for k, v in ((run.get("summary") or {}).get("naming") or {}).items() if k != "passes"},
            "usd_estimate_call": run.get("usd_estimate")}


def instances(bench, report, site):
    import r4_instances_eval as ie
    maps, entities, cs = ie.run_maps(Path(bench) / "mirror", report)
    parent = {c["id"]: c["part_of"]["id"] for c in cs["cards"] if (c.get("part_of") or {}).get("kind") == "part"}
    res, _ = ie.evaluate(site, maps, entities, cs.get("aliases") or {}, parent)
    return {k: ({kk: vv for kk, vv in v.items() if kk != "detail"} if isinstance(v, dict) else v) for k, v in res.items()} | \
        {"missed_ids": {k: v["detail"]["missed"] for k, v in res.items() if isinstance(v, dict) and "detail" in v}}


def collect(a):
    import click_audit as ca
    import r4_naming_results as nr
    import r4b_audit as ra
    bench = dict(x.split("=") for x in a.bench.split(","))
    base = dict(x.split("=") for x in a.baseline.split(",")) if a.baseline else {}
    res = {"videos": {}, "benches": bench, "baseline": base}
    for site, d in bench.items():
        v = res["videos"].setdefault(site, {"calls": {}})
        cs = calls(d, site)
        for kind, rec in cs.items():
            v["calls"][kind] = per_call(d, site, kind, rec)
        boot = json.loads((Path(d) / "boot.json").read_text())
        summ = json.loads((Path(d) / "summary.json").read_text()) if (Path(d) / "summary.json").exists() else {}
        v["boot"] = {"ready_s": boot.get("ready_s"), "submit_to_ready_s_two_clocks": boot.get("submit_to_ready_s_two_clocks"), "resident_gb": boot.get("resident_gb")}
        v["usd_estimate_upper"] = summ.get("usd_estimate_upper")
        if "warm" in cs:
            v["instances"] = {"r4b warm": instances(d, cs["warm"]["run"]["report"], site)}
            if site in base:
                bc = calls(base[site], site)
                if "warm" in bc:
                    v["instances"]["round 3 warm"] = instances(base[site], bc["warm"]["run"]["report"], site)
    if a.clicks:
        res["clicks"] = {Path(r["file"]).parent.name: r for r in ca.score(sorted(str(p) for p in Path(a.clicks).iterdir() if p.is_dir()))}
    if a.cards:
        res["cards_audit"] = ra.score(sorted(str(p) for p in Path(a.cards).iterdir() if (p / "labels.json").exists()))
    res["heldout_types"] = {}
    for kind in ("warm", "first"):
        g, miss, _ = nr.heldout(bench, Path(a.out) / f"heldout-{kind}", kind)
        res["heldout_types"][kind] = {"by_site": g, "not_matched": len(miss)}
    if base:  # round 3's warm calls on the same held-out items (the same scorer): the types before the cascade
        g, miss, _ = nr.heldout(base, Path(a.out) / "heldout-round3-warm", "warm")
        res["heldout_types"]["round 3 warm"] = {"by_site": g, "not_matched": len(miss)}
        v = {s: {"share": ra.share(__import__("fast_report_eval").load_layers(Path(d) / "mirror", calls(d, s)["warm"]["run"]["report"]))} for s, d in base.items()}
        res["round3_shares"] = {s: {k: x for k, x in v[s]["share"].items() if k != "completeness"} for s in v}
    for d in (a.gt or "").split(","):  # GT runs (accuracy_gt.py score wrote DIR/accuracy/score.json); several: one per sequence set
        if d and (Path(d) / "accuracy" / "score.json").exists():
            res.setdefault("gt", {})[Path(d).name] = json.loads((Path(d) / "accuracy" / "score.json").read_text())
    if a.shots:
        res["shots"] = {p.name: json.loads((p / "shots.json").read_text()) for p in sorted(Path(a.shots).iterdir()) if (p / "shots.json").exists()}
    return res


def pct(x):
    return "-" if x is None else f"{100 * x:.0f}%"


def md(res):
    V = res["videos"]
    cols = [s for s in SITES if s in V]
    head = "| | " + " | ".join(NAME[s] for s in cols) + " |\n|---|" + "---|" * len(cols) + "\n"

    def row(label, fn):
        cells = []
        for s in cols:
            try:
                cells.append(fn(s))
            except (KeyError, TypeError, ZeroDivisionError, StopIteration):
                cells.append("-")
        return f"| {label} | " + " | ".join(cells) + " |\n"
    w = lambda s: V[s]["calls"]["warm"]  # noqa: E731
    f = lambda s: V[s]["calls"]["first"]  # noqa: E731
    out = "## 1. Types (warm call; first call in brackets)\n\n" + head
    out += row("object cards", lambda s: f"{w(s)['share']['object_cards']} ({f(s)['share']['object_cards']})")
    out += row("typed by family (a specific name or '<family> (type only)')", lambda s: f"{pct(w(s)['share']['family_typed'])} ({pct(f(s)['share']['family_typed'])})")
    out += row("with a specific name", lambda s: f"{pct(w(s)['share']['specific_name'])} ({pct(f(s)['share']['specific_name'])})")
    out += row("shape type only (no family)", lambda s: f"{pct(w(s)['share']['shape_only'])} ({pct(f(s)['share']['shape_only'])})")
    out += row("round 3 warm: typed by family / specific name", lambda s: f"{pct(res['round3_shares'][s]['family_typed'])} / {pct(res['round3_shares'][s]['specific_name'])}")
    ca = res.get("cards_audit") or {}
    out += row("fresh card audit: type right / close / wrong / unclear (n)", lambda s: (lambda t: f"{t.get('right', 0)} / {t.get('close', 0)} / {t.get('wrong', 0)} / "
                                                                                           f"{t.get('unclear', 0)} ({ca[s]['n']})")(ca[s]["type"]))
    ho = res.get("heldout_types", {})
    for k in ("warm", "first", "round 3 warm"):
        out += row(f"held-out items (round 2's labels), {k}: type (family) right / n", lambda s, k=k: (lambda g: f"{g['family_right']}/{g['n']} = {g['family_right'] / g['n']:.2f}")(ho[k]["by_site"][s]))
        out += row(f"held-out items, {k}: specific names right / right-or-close / named", lambda s, k=k: (lambda g: f"{g['named_right']} / {g['named_right'] + g['named_close']} / {g['named_n']}")(ho[k]["by_site"][s]))
    out += "\n## 2. Segmentation and clicks (warm call)\n\n" + head
    out += row("object cards with a pick region (clickable)", lambda s: pct(w(s)["share"]["clickable"]))
    out += row("fresh card audit: outline right / partial / wrong", lambda s: (lambda t: f"{t.get('right', 0)} / {t.get('partial', 0)} / {t.get('wrong', 0)}")(ca[s]["outline"]))
    ck = res.get("clicks") or {}
    out += row("60 random clicks: right card / nothing / wrong / background right / background hit",
               lambda s: f"{ck[s]['correct']} / {ck[s]['miss']} / {ck[s]['wrong']} / {ck[s]['background']} / {ck[s]['background-hit']}")
    out += row("clicks on a thing that opened the right card", lambda s: f"{ck[s]['correct']}/{ck[s]['correct'] + ck[s]['miss'] + ck[s]['wrong']} = {ck[s]['on_object_correct']:.2f}")
    for ref in ("clean", "all", "models"):
        out += row(f"delivered objects ({ref}): covered, in pieces, wrong-merge cards; round 3 warm in brackets",
                   lambda s, ref=ref: (lambda a, b: f"{a['covered']}/{a['delivered']}, {a['in_pieces']}, {a['wrong_merge_cards']}"
                                       + (f" ({b['covered']}/{b['delivered']}, {b['in_pieces']}, {b['wrong_merge_cards']})" if b else ""))(
                       V[s]["instances"]["r4b warm"][ref], (V[s]["instances"].get("round 3 warm") or {}).get(ref)))
    out += "\n## 3. Physical (warm call)\n\n" + head
    out += row("object cards complete (every required field: value +-u, bound, or status + reason)", lambda s: pct(w(s)["physical"]["complete_share"]["object"]))
    out += row("person cards complete", lambda s: pct(w(s)["physical"]["complete_share"].get("person")))
    out += row("contract violations, every cards version (both calls)", lambda s: str(w(s)["physical"]["contract_violations"] + f(s)["physical"]["contract_violations"]))
    for fld in ("position_xy", "top_above_floor", "base_above_floor", "height", "width", "depth", "principal_axis_tilt_deg"):
        out += row(f"object cards with a number or bound: {fld}", lambda s, fld=fld: pct(w(s)["physical"]["with_number_share_objects"][fld]))
    out += row("fresh card audit: physical plausible / implausible / unclear", lambda s: (lambda t: f"{t.get('plausible', 0)} / {t.get('implausible', 0)} / {t.get('unclear', 0)}")(ca[s]["physical"]))
    if res.get("gt"):  # r4_physical's GT tables: median |error| as delivered (assumed 1.6 m camera) and at the true height, bounds holding
        from r4_physical import gt_tables
        out += "\n### Physical against ground truth (TUM fr1 room, ARKitScenes; warm calls)\n"
        for name, sc in res["gt"].items():
            out += f"\n{name}:\n" + gt_tables(sc)
    out += "\n## 4. Display models (warm call)\n\n" + head
    out += row("object cards with a display model", lambda s: pct(w(s)["share"]["model"]))
    out += row("model kinds", lambda s: ", ".join(f"{k} {v}" for k, v in sorted(w(s)["share"]["model_kinds"].items(), key=lambda x: -x[1])))
    out += row("SAM 3D accepted meshes / eligible / attempted", lambda s: f"{w(s)['models']['sam3d_accepted']} / {w(s)['models']['eligible']} / {w(s)['models']['attempted']}")
    out += row("fresh card audit: model plausible / implausible / absent / unclear", lambda s: (lambda t: f"{t.get('plausible', 0)} / {t.get('implausible', 0)} / "
                                                                                                    f"{t.get('absent', 0)} / {t.get('unclear', 0)}")(ca[s]["model"]))
    out += "\n## Times (s from the MP4 bytes in the container to the Volume commit; warm call, first call after boot in brackets)\n\n" + head
    for k in ("first 3D", "objects v1", "cards v1", "types (first pass)", "objects v3", "cards v3", "types (densify pass)", "first SAM 3D model", "models final",
              "splat preview", "call end"):
        out += row(k, lambda s, k=k: f"{w(s)['times'][k]} ({f(s)['times'][k]})")
    out += row("cold start (not analysis), s", lambda s: f"{V[s]['boot']['ready_s']}")
    out += "\n## GPU peaks (GiB, max over stages, GPU 0 / GPU 1; stages over 72 GiB)\n\n" + head
    out += row("warm", lambda s: f"{w(s)['times']['gpu_peak_gib'][0]:.1f} / {w(s)['times']['gpu_peak_gib'][1]:.1f}; over 72: {', '.join(w(s)['times']['over_72']) or 'none'}")
    out += row("first", lambda s: f"{f(s)['times']['gpu_peak_gib'][0]:.1f} / {f(s)['times']['gpu_peak_gib'][1]:.1f}; over 72: {', '.join(f(s)['times']['over_72']) or 'none'}")
    out += "\n## VLM requests per call (warm; first in brackets)\n\n" + head
    for k in ("naming (cluster medoids)", "naming escalated", "scene vocabulary (one request)", "event captions (windows)", "hazard / judge", "qwen decider (identity_vlm)"):
        out += row(k, lambda s, k=k: f"{w(s)['vlm'][k]} ({f(s)['vlm'][k]})")
    out += "\n## Spend (Modal list-price upper bound over each container's life)\n\n" + head
    out += row("bench, $", lambda s: str(V[s]["usd_estimate_upper"]))
    return out


def self_check():
    run = {"stages": [{"stage": "vlm.vocab", "n": {"frames": 5}}, {"stage": "vlm.events", "n": {"windows": 2}}, {"stage": "naming.vlm.first", "n": {"questions": 3}}],
           "summary": {"naming": {"passes": [{"vlm_questions": 3, "vlm_questions_escalated": 0}, {"vlm_questions": 2}]}}}
    v = vlm_calls(run)
    assert v["naming (cluster medoids)"] == 5 and v["scene vocabulary (one request)"] == 1 and v["event captions (windows)"] == 2 and v["hazard / judge"] == 0
    res = {"videos": {"me340": {"calls": {k: {"share": {"object_cards": 10, "family_typed": .9, "specific_name": .5, "shape_only": .1, "clickable": 1., "model": .95,
                                                        "model_kinds": {"box": 9}}} for k in ("first", "warm")}}}}
    t = md(res)
    assert "| typed by family (a specific name or '<family> (type only)') | 90% (90%) |" in t and "| object cards with a display model | 95% |" in t, t
    print("r4b_results self-check ok: VLM requests by purpose, the acceptance tables (missing parts shown as '-')")


if __name__ == "__main__":
    if sys.argv[1:] == ["--self-check"]:
        self_check()
        sys.exit()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out")
    for k in ("bench", "baseline", "clicks", "cards", "gt", "shots"):
        p.add_argument(f"--{k}")
    a = p.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    r = collect(a)
    (Path(a.out) / "summary.json").write_text(json.dumps(r, indent=1, default=str))
    (Path(a.out) / "tables.md").write_text(md(r))
    print(md(r))
