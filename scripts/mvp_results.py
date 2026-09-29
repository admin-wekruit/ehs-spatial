"""The click MVP's results page (mvp/integrate): one table per question across the three videos, from each video's bench
folder (scripts/fast_report_bench.py: summary.json + mvp/summary.json, D's scoring) and the viewer check (C's
tests/mvp-click-browser.mjs: click-check.json + screenshots), with fb/integrate and D1's baseline beside.

    python scripts/mvp_results.py OUT --bench me340=RUN_DIR,samsclub-a2=RUN_DIR,walmart=RUN_DIR --viewer CHECK_DIR [--plugged plugged.json]
    python scripts/mvp_results.py --self-check
"""
import argparse
import json
import sys
from pathlib import Path

PHASE2 = Path("/Users/adam/Desktop/panoptes-public/research-notes/phase2")
BASELINE = PHASE2 / "runs/mvp-d1-baseline-001/summary.json"
NAMES = {"me340": "ME340", "samsclub-a2": "Sam's Club", "walmart": "Walmart"}
LAYERS = [("first 3D (cameras + light room)", "cameras"), ("objects v1", "objects v1"), ("pick v1", "pick v1"), ("cards v1", "cards v1"),
          ("judgements v1", "judgements v1"), ("cards v2 (identity)", "cards v2"), ("judgements v2 (VLM)", "judgements v2"),
          ("pick v2 (densified)", "pick v2"), ("cards v3", "cards v3"), ("first SAM 3D model", "first SAM 3D model"), ("splat preview", "splat preview")]


def get(d, *path):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def calls(bench):
    """{kind: call record} (first call after boot, warm, shifted) of one video's bench."""
    return {c["kind"]: c for c in bench.get("calls", [])}


def layer_s(call, row):
    r = get(call, "mvp_latency", "layers", row) or {}
    return r.get("written_s"), r.get("ok")


def video_rows(site, bench, mvp, base):
    cs = calls(bench)
    first, warm, shifted = cs.get("first"), cs.get("warm"), cs.get("shifted")
    s, b = get(mvp, "sites", site) or {}, get(base, "sites", site) or {}
    out = {"times_s": {}, "gpu_peak_gib": {}, "clicks": {}, "identity": {}, "physical": {}, "repeat": {}, "cards": {}, "judgements": {}}
    for label, row in LAYERS:
        w, ok = layer_s(warm, row) if warm else (None, None)
        f, _ = layer_s(first, row) if first else (None, None)
        out["times_s"][label] = {"warm": w, "first_call": f, "target_ok_warm": ok}
    for kind, c in cs.items():
        out["gpu_peak_gib"][kind] = [g["peak_gb"] for g in c.get("gpu_peak") or []]
        out.setdefault("flags", {})[kind] = c.get("flags") or []
    out["cold_start_s"] = get(bench, "boot", "ready_s") or get(bench, "boot", "sam3d_ready_s")
    for name, r in (s.get("clicks") or {}).items():
        out["clicks"][name] = {"object_correct": get(r, "object", "correct_rate"), "object_correct_iou_only": get(r, "object", "correct_iou_only_rate"),
                               "object_hit": get(r, "object", "hit_rate"), "object_wrong_entity": get(r, "object", "wrong_entity_rate"),
                               "object_n": get(r, "object", "n"), "person_correct": get(r, "person", "correct_rate"), "person_n": get(r, "person", "n"),
                               "background_false_hit": get(r, "background", "false_hit_rate")}
    for name, r in (b.get("clicks") or {}).items():
        out["clicks"]["baseline " + name] = {"object_correct": get(r, "object", "correct_rate"), "object_correct_iou_only": get(r, "object", "correct_iou_only_rate"),
                                             "person_correct": get(r, "person", "correct_rate"), "background_false_hit": get(r, "background", "false_hit_rate")}
    out["click_audit"] = s.get("click_audit")
    idn = s.get("identity") or {}
    out["identity"] = {"matched": idn.get("matched"), "agreement": idn.get("agreement"), "by_route": idn.get("by_route"),
                       "baseline_agreement": get(b, "identity", "agreement"), "baseline_matched": get(b, "identity", "matched")}
    ph = s.get("physical") or {}
    out["physical"] = {"matched": ph.get("matched"), "agreement": ph.get("agreement"), "baseline_agreement": get(b, "physical", "agreement"),
                       "inflation": ph.get("inflation"), "baseline_inflation": get(b, "physical", "inflation"), "known_verticals": ph.get("known_verticals"),
                       "scale_ours_over_delivered": get(s, "scale", "ours_over_reference")}
    out["repeat"] = s.get("repeat")
    out["cards"] = s.get("card_check")
    out["judgements_counts"] = s.get("judgements_counts")
    out["box_audit"] = s.get("box_audit")
    out["acceptance"] = get(mvp, "acceptance", site)
    out["usd_estimate"] = sum(c.get("usd_estimate") or 0 for c in bench.get("calls", []))
    out["usd_upper_container_life"] = bench.get("usd_estimate_upper")
    out["reports"] = {k: c.get("report") for k, c in cs.items()}
    return out


def pct(x):
    return "—" if x is None else f"{100 * x:.0f}%"


def num(x, n=1):
    return "—" if x is None else f"{x:.{n}f}"


def md(summary):
    sites = list(summary["videos"])
    head = "| | " + " | ".join(NAMES.get(s, s) for s in sites) + " |\n|---|" + "---|" * len(sites) + "\n"
    v = summary["videos"]
    lines = ["# Click MVP (mvp/integrate): results", "", summary["note"], ""]
    lines += ["## Analysis time to each layer (s from the MP4 bytes in the container; warm call, first call after boot in brackets)", "", head.rstrip("\n")]
    for label, _ in LAYERS:
        cells = []
        for s in sites:
            t = v[s]["times_s"][label]
            ok = {True: " ✓", False: " ✗", None: ""}[t["target_ok_warm"]]
            cells.append(f"{num(t['warm'])} ({num(t['first_call'])}){ok}")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.append("| cold start (s, not analysis) | " + " | ".join(num(v[s]["cold_start_s"]) for s in sites) + " |")
    lines.append("| GPU peaks GiB (warm; first) | " + " | ".join(
        f"{'/'.join(num(x) for x in v[s]['gpu_peak_gib'].get('warm', []))}; {'/'.join(num(x) for x in v[s]['gpu_peak_gib'].get('first', []))}" for s in sites) + " |")
    lines += ["", "✓/✗: CLICK-MVP-SPEC section 7 target (cameras/people/events: fb/integrate ± 1 s; pick ≤ outlines + 1 s; cards v1 ≤ objects + 10 s; "
              "judgements v1 ≤ objects + 12 s; cards/judgements v2 ≤ objects + 60 s; pick v2 / cards v3 ≤ objects + 40 s; first model ≤ 120 s; splat ≤ 230 s).", ""]
    lines += ["## Click accuracy (SAM 3 references on X1's 60 eval frames a video, seed-0 clicks; D's rule: IoU ≥ 0.3 or covers ≥ 50%)", "", head.rstrip("\n")]
    names = sorted({n for s in sites for n in v[s]["clicks"]})
    for n in names:
        for key, lab in (("object_correct", "object clicks correct"), ("object_correct_iou_only", "correct, IoU ≥ 0.3 only"),
                         ("person_correct", "person clicks correct"), ("background_false_hit", "background false hits")):
            lines.append(f"| {n}: {lab} | " + " | ".join(pct(get(v[s]['clicks'], n, key)) for s in sites) + " |")
    lines += ["", "## Identity (1:1 match ≤ 0.5 m to the delivered report's clear objects, after the camera Sim3; agreement with its model-made names)", "", head.rstrip("\n")]
    lines.append("| matched | " + " | ".join(str(v[s]["identity"]["matched"]) for s in sites) + " |")
    lines.append("| name agreement, MVP | " + " | ".join(pct(v[s]["identity"]["agreement"]) for s in sites) + " |")
    lines.append("| name agreement, fb/integrate (D1) | " + " | ".join(pct(v[s]["identity"]["baseline_agreement"]) for s in sites) + " |")
    lines += ["", "## Physical info vs the delivered report (median / p90 |Δ| m, same definitions, after the camera Sim3; coverage = share with |Δ| ≤ u without its scale part)", "", head.rstrip("\n")]
    for q in ("top", "base", "long_side", "short_side"):
        cells = []
        for s in sites:
            a, bb = get(v[s]["physical"], "agreement", q) or {}, get(v[s]["physical"], "baseline_agreement", q) or {}
            cells.append(f"{num(a.get('median'), 2)} / {num(a.get('p90'), 2)} (cov {pct(a.get('coverage'))}, n {a.get('n')}); fb {num(bb.get('median'), 2)} / {num(bb.get('p90'), 2)}")
        lines.append(f"| {q} | " + " | ".join(cells) + " |")
    lines.append("| boxes > 3 m, all / shown non-large | " + " | ".join(
        f"{pct(get(v[s]['physical'], 'inflation', 'over_3m_share'))} / {pct(get(v[s]['physical'], 'inflation', 'shown_non_large_over_3m_share'))} "
        f"(fb {pct(get(v[s]['physical'], 'baseline_inflation', 'over_3m_share'))} / {pct(get(v[s]['physical'], 'baseline_inflation', 'shown_non_large_over_3m_share'))})" for s in sites) + " |")
    lines.append("| known verticals shown / outside ±u | " + " | ".join(
        f"{get(v[s]['physical'], 'known_verticals', 'shown')} / {len(get(v[s]['physical'], 'known_verticals', 'misses') or [])}" for s in sites) + " |")
    lines += ["", "## Repeatability (warm call vs the +5 s shifted window, matched through the Sim3 between their cameras)", "", head.rstrip("\n")]
    for q in ("top", "height", "long_side", "position"):
        lines.append(f"| median / p90 |Δ| {q} (m) | " + " | ".join(
            (lambda d: f"{num(d.get('median'), 3)} / {num(d.get('p90'), 3)} (n {d.get('n')})")(get(shift_of(v[s]["repeat"]), "delta", q) or {}) for s in sites) + " |")
    for fam in ("height", "extent", "position", "angle"):
        lines.append(f"| coverage k=1, {fam} | " + " | ".join(
            (lambda d: f"{pct(d.get('coverage'))} (n {d.get('n')})")(get(shift_of(v[s]["repeat"]), "coverage_k1", fam) or {}) for s in sites) + " |")
    lines.append(f"\nk_family (smallest k ≥ 1 with coverage ≥ 0.9, pooled over the videos): {json.dumps(summary.get('k_family'))}\n")
    lines += ["## Judgements", "", head.rstrip("\n")]
    for verdict in ("FAIL", "NEEDS_REVIEW", "PASS", "NO_DATA"):
        lines.append(f"| {verdict} rows | " + " | ".join(str((v[s]["judgements_counts"] or {}).get(verdict)) for s in sites) + " |")
    j = summary.get("judgement_accuracy") or {}
    if j:
        lines += ["", f"Agent-labelled accuracy ({j.get('labeller')}): {j.get('labelled')} rows labelled; false PASS total {j.get('false_pass_total')}.", "",
                  "| check | rows | labelled | verdict shares | FAIL precision (n) | false PASS |", "|---|---|---|---|---|---|"]
        for c, r in (j.get("checks") or {}).items():
            lines.append(f"| {c} | {r['rows']} | {r['labelled']} | {', '.join(f'{k} {pct(x)}' for k, x in r['shares'].items())} | "
                         f"{pct(r['fail_precision'])} ({r['fail_precision_n']}) | {r['false_pass']} |")
    lines += ["", "## Cards", "", head.rstrip("\n")]
    lines.append("| cards / fields / contract violations | " + " | ".join(
        f"{get(v[s], 'cards', 'cards')} / {get(v[s], 'cards', 'fields')} / {get(v[s], 'cards', 'violations')}" for s in sites) + " |")
    if summary.get("viewer"):
        lines += ["", "## Viewer (headless Chromium, recorded layer times)", "", head.rstrip("\n")]
        vw = summary["viewer"]
        lines.append("| click → card p50 / p95 ms (200 clicks) | " + " | ".join(
            (lambda r: f"{num(get(r, 'latencyMs', 'p50'))} / {num(get(r, 'latencyMs', 'p95'))}" if r else "—")(vw.get(s)) for s in sites) + " |")
        lines.append("| pick decode ms | " + " | ".join((lambda r: num(get(r, "pickDecodeMs")) if r else "—")(vw.get(s)) for s in sites) + " |")
        lines += ["", "Screenshots:"]
        for s in sites:
            for shot in (vw.get(s) or {}).get("shots", []):
                lines.append(f"- {NAMES.get(s, s)}: `{Path(shot['file']).name}`: aimed {shot['aimed']} → {shot.get('title')} {' '.join(shot.get('chips') or [])}")
    if summary.get("plugged"):
        lines += ["", "## Experiments plugged in, and not", ""] + [f"- **{k}**: {x}" for k, x in summary["plugged"].items()]
    lines += ["", f"Spend: {summary.get('spend')}", ""]
    return "\n".join(lines)


def shift_of(repeat):
    for name, r in (repeat or {}).items():
        if "shifted" in name:
            return r
    return next(iter((repeat or {}).values()), None)


def build(out, bench_dirs, viewer_dir=None, plugged=None):
    base = json.loads(BASELINE.read_text()) if BASELINE.exists() else {}
    videos, k_family = {}, {}
    for site, d in bench_dirs.items():
        d = Path(d)
        bench = json.loads((d / "summary.json").read_text())
        mvp = json.loads((d / "mvp/summary.json").read_text()) if (d / "mvp/summary.json").exists() else {}
        videos[site] = video_rows(site, bench, mvp, base)
        videos[site]["bench_dir"] = str(d)
        k_family[site] = mvp.get("k_family")
    viewer = None
    if viewer_dir and (Path(viewer_dir) / "click-check.json").exists():
        chk = json.loads((Path(viewer_dir) / "click-check.json").read_text())
        viewer = {}
        for r in chk["videos"]:
            site = next((s for s in bench_dirs if s in r["report"]), r["report"])
            viewer[site] = r
    summary = {"schema": "panoptes-mvp-results-v1", "note": "All numbers measured on 2 x A100-SXM4-80GB (MPS), ephemeral Modal runs, analysis time "
               "only (cold start apart). Physical values are at estimated scale (floor plane + assumed 1.6 m camera height); 'agreement' is with the "
               "delivered report, itself model-made and at estimated scale, not ground truth. Labels on clicks, boxes and judgements are agent-made.",
               "videos": videos, "k_family": k_family, "viewer": viewer, "plugged": plugged, "baseline": str(BASELINE)}
    judged = [json.loads((Path(d) / "mvp/summary.json").read_text()).get("judgements") for d in bench_dirs.values() if (Path(d) / "mvp/summary.json").exists()]
    summary["judgement_accuracy"] = merge_judgements([j for j in judged if j])
    usd = sum((v.get("usd_upper_container_life") or 0) for v in videos.values())
    summary["spend"] = f"about ${usd:.2f} at list price over the three containers' lives (upper bound; boot included); reconcile with `modal billing report`"
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    (out / "summary.md").write_text(md(summary))
    return summary


def merge_judgements(parts):
    """D's judgement_row per video folder -> one table (rows and labels summed per check)."""
    if not parts:
        return None
    checks = {}
    for p in parts:
        for c, r in (p.get("checks") or {}).items():
            a = checks.setdefault(c, {"rows": 0, "labelled": 0, "counts": {}, "fail_present": 0, "fail_n": 0, "false_pass": 0})
            a["rows"] += r["rows"]
            a["labelled"] += r["labelled"]
            for k, x in r["shares"].items():
                a["counts"][k] = a["counts"].get(k, 0) + round(x * r["rows"])
            if r.get("fail_precision") is not None:
                a["fail_present"] += round(r["fail_precision"] * r["fail_precision_n"])
                a["fail_n"] += r["fail_precision_n"]
            a["false_pass"] += r["false_pass"]
    out = {c: {"rows": a["rows"], "labelled": a["labelled"], "shares": {k: n / a["rows"] for k, n in sorted(a["counts"].items())},
               "fail_precision": a["fail_present"] / a["fail_n"] if a["fail_n"] else None, "fail_precision_n": a["fail_n"], "false_pass": a["false_pass"]}
           for c, a in sorted(checks.items())}
    return {"checks": out, "labelled": sum(a["labelled"] for a in out.values()), "false_pass_total": sum(a["false_pass"] for a in out.values()),
            "labeller": "agent-made labels (contact sheets); not ground truth"}


def self_check():
    j = merge_judgements([{"checks": {"J5": {"rows": 4, "labelled": 2, "shares": {"PASS": .5, "NEEDS_REVIEW": .5}, "fail_precision": None, "fail_precision_n": 0, "false_pass": 0}}},
                          {"checks": {"J5": {"rows": 2, "labelled": 2, "shares": {"FAIL": 1.}, "fail_precision": .5, "fail_precision_n": 2, "false_pass": 1}}}])
    assert j["checks"]["J5"]["rows"] == 6 and j["checks"]["J5"]["fail_precision"] == .5 and j["false_pass_total"] == 1
    assert abs(j["checks"]["J5"]["shares"]["FAIL"] - 2 / 6) < 1e-9
    call = {"kind": "warm", "mvp_latency": {"layers": {"pick v1": {"written_s": 40., "ok": True}}}, "gpu_peak": [{"peak_gb": 60.}, {"peak_gb": 50.}]}
    rows = video_rows("me340", {"calls": [call]}, {}, {})
    assert rows["times_s"]["pick v1"] == {"warm": 40., "first_call": None, "target_ok_warm": True} and rows["gpu_peak_gib"]["warm"] == [60., 50.]
    assert shift_of({"warm call 1": 1, "shifted call 2": 2}) == 2
    print("mvp_results self-check ok")


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        self_check()
        sys.exit()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path)
    ap.add_argument("--bench", required=True, help="site=RUN_DIR,...")
    ap.add_argument("--viewer", type=Path)
    ap.add_argument("--plugged", type=Path)
    a = ap.parse_args()
    b = dict(x.split("=", 1) for x in a.bench.split(","))
    s = build(a.out, b, a.viewer, json.loads(a.plugged.read_text()) if a.plugged else None)
    print((a.out / "summary.md").read_text())
