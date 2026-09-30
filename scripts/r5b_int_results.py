"""r5b integrate results: summary.json + the tables of summary.md, from the final benches (one per video: a first call after boot and a
warm call), the GT run (physical, width bounds, angles, a cross-visit pair), the audits by eye (agent-labelled) and r4b's results.

    python scripts/r5b_int_results.py OUT --bench me340=DIR,samsclub-a2=DIR,walmart=DIR --gt DIR --audit DIR --align DIR --visit JSON
"""
import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "scripts")]
RUNS = Path("/Users/adam/Desktop/panoptes-public/research-notes/phase2/runs")
SITES, NAME = ("me340", "samsclub-a2", "walmart"), {"me340": "ME340", "samsclub-a2": "Sam's Club", "walmart": "Walmart"}


def pct(x, d=0):
    return "-" if x is None else f"{100 * x:.{d}f}%"


def video(site, bench, align_dir):
    import r4b_results as rr
    import r5b_quick as rq
    import r5b_vocab as rv
    cs = rr.calls(bench, site)
    out = {"bench": str(bench)}
    for kind, rec in cs.items():
        out[kind] = rr.per_call(bench, site, kind, rec)
    w = cs["warm"]
    out["vocab_row"] = rv.call_row(Path(bench) / "mirror", w)  # held-out (both scorers), clean coverage, VLM requests by purpose
    out["quick"] = rq.main(bench, site, align_dir, Path(align_dir) / f"quick-{site}.json")
    boot = json.loads((Path(bench) / "boot.json").read_text())
    summ = json.loads((Path(bench) / "summary.json").read_text()) if (Path(bench) / "summary.json").exists() else {}
    out["boot"] = {k: boot.get(k) for k in ("ready_s", "submit_to_ready_s_two_clocks", "resident_gb")}
    out["usd_estimate_upper"] = summ.get("usd_estimate_upper")
    out["jev_usd_upper"] = (summ.get("jev") or {}).get("usd_upper")
    return out


def r4b_rows():
    s = json.loads((RUNS / "r4b-results" / "summary.json").read_text())
    f = json.loads((RUNS / "r5b-vocab-results" / "final.json").read_text())
    out = {}
    for site in SITES:
        w = s["videos"][site]["calls"]["warm"]
        out[site] = {"share": w["share"], "physical": w["physical"], "times": w["times"], "vlm": w["vlm"], "acceptance": s["acceptance"][site],
                     "clicks": s["clicks"].get(site), "cards_audit": s["cards_audit"].get(site),
                     "vocab_row": f["r4b_rows"][site]["warm"], "ram_row": f["now"][site]["warm"]}
    return out


def audits(d):
    """Tallies of the label files under DIR/<site>/ (clicks-*.json + labels-*.json: click_audit; cards/labels.json: r4b_audit;
    models/labels.json: r5b_models sheets; tilted/labels.json; timelines.json)."""
    import click_audit as ca
    import r4b_audit as ra
    import r5b_models as rm
    out = {}
    for site in SITES:
        s = Path(d) / site
        if not s.exists():
            continue
        o = out[site] = {}
        if (s / "clicks").exists() and list((s / "clicks").glob("labels-*.json")):
            o["clicks"] = ca.score([str(s / "clicks")])[0]
        if (s / "cards" / "labels.json").exists():
            o["cards"] = ra.score([str(s / "cards")]).get(site) or next(iter(ra.score([str(s / "cards")]).values()), None)
        if (s / "models" / "labels.json").exists():
            o["models"] = {"shown": rm.label_counts(s / "models" / "labels.json", "shown")}
        if (s / "tilted" / "labels.json").exists():
            o["tilted"] = json.loads((s / "tilted" / "labels.json").read_text()).get("tally")
        if (s / "timelines.json").exists():
            o["timelines"] = json.loads((s / "timelines.json").read_text()).get("tally")
    return out


def gt(gt_dir, out_dir):
    res = {}
    p = Path(gt_dir) / "accuracy" / "score.json"
    if p.exists():
        res["physical"] = [r for r in json.loads(p.read_text())["table"]]
    for k, f in (("width", "width-gt.json"), ("angles", "angle-gate.json")):
        q = Path(out_dir) / "gt" / f
        if q.exists():
            res[k] = json.loads(q.read_text())
    return res


def visit(layer_json, gt_json):
    """The cross-visit pair scored by scripts/r5b_visits_eval.score_pair (object match and change precision / recall on GT depth)."""
    import r5b_visits_eval as ve
    lay, g = json.loads(Path(layer_json).read_text()), json.loads(Path(gt_json).read_text())
    return {"score": ve.score_pair(lay, g), "record": {k: v for k, v in lay.items() if k not in ("objects",)}}


def md(res):
    V, B = res["videos"], res["r4b"]
    cols = [s for s in SITES if s in V]
    head = "| | " + " | ".join(NAME[s] for s in cols) + " |\n|---|" + "---|" * len(cols) + "\n"

    def row(label, fn):
        cells = []
        for s in cols:
            try:
                cells.append(str(fn(s)))
            except Exception:  # noqa: BLE001  a missing piece shows as '-'
                cells.append("-")
        return f"| {label} | " + " | ".join(cells) + " |\n"
    w = lambda s: V[s]["warm"]  # noqa: E731
    q = lambda s: V[s]["quick"]  # noqa: E731
    vr = lambda s: V[s]["vocab_row"]  # noqa: E731
    A = res.get("audits") or {}
    t = "### 1. Types (warm call; r4b warm in brackets)\n\n" + head
    t += row("object cards", lambda s: f"{w(s)['share']['object_cards']} ({B[s]['share']['object_cards']})")
    t += row("typed by family / specific name", lambda s: f"{pct(w(s)['share']['family_typed'])} / {pct(w(s)['share']['specific_name'])} "
             f"({pct(B[s]['share']['family_typed'])} / {pct(B[s]['share']['specific_name'])})")
    t += row("held-out family right (r5b scorer)", lambda s: (lambda h, g: f"{h['right_ext']}/{h['n']} = {h['acc_ext']:.2f} ({g['right_ext']}/{g['n']} = {g['acc_ext']:.2f})")(
        vr(s)["heldout"]["as shown"], B[s]["vocab_row"]["heldout"]["as shown"]))
    t += row("held-out family right (round 4 scorer)", lambda s: (lambda h, g: f"{h['right_r4b']}/{h['n']} ({g['right_r4b']}/{g['n']})")(
        vr(s)["heldout"]["as shown"], B[s]["vocab_row"]["heldout"]["as shown"]))
    t += row("fresh card audit: type right / close / wrong / unclear (n)", lambda s: (lambda c: f"{c['type'].get('right', 0)} / {c['type'].get('close', 0)} / "
             f"{c['type'].get('wrong', 0)} / {c['type'].get('unclear', 0)} ({c['n']})")(A[s]["cards"]))
    t += "\n### 2. Clicks and segmentation\n\n" + head
    t += row("random clicks: right card / nothing (should be) / wrong / background right / background hit (n)", lambda s: (lambda c: f"{c['correct']} / {c['miss']} / "
             f"{c['wrong']} / {c['background']} / {c['background-hit']} ({c['n']})")(A[s]["clicks"]))
    t += row("r4b clicks (60): right / miss / wrong / background / background hit", lambda s: (lambda c: f"{c['correct']} / {c['miss']} / {c['wrong']} / "
             f"{c['background']} / {c['background-hit']}")(B[s]["clicks"]))
    t += row("clean objects covered / delivered, in pieces, wrong merges (r4b)", lambda s: (lambda c, g: f"{c['covered']}/{c['delivered']}, {c['in_pieces']}, "
             f"{c['wrong_merge_cards']} ({g['covered']}/{g['delivered']}, {g['in_pieces']}, {g['wrong_merge_cards']})")(vr(s)["instances"]["clean"], B[s]["vocab_row"]["instances"]["clean"]))
    t += row("card audit outline right / partial / wrong", lambda s: (lambda c: f"{c['outline'].get('right', 0)} / {c['outline'].get('partial', 0)} / {c['outline'].get('wrong', 0)}")(A[s]["cards"]))
    t += "\n### 3. Physical\n\n" + head
    ph = lambda s: w(s)["physical"]["with_number_share_objects"]  # noqa: E731
    t += row("object cards with every field (value, bound or reason)", lambda s: pct(w(s)["physical"]["complete_share"]["object"]))
    t += row("with a number: height / width / depth (r4b)", lambda s: f"{pct(ph(s)['height'])} / {pct(ph(s)['width'])} / {pct(ph(s)['depth'])} "
             f"({pct(B[s]['physical']['with_number_share_objects']['height'])} / {pct(B[s]['physical']['with_number_share_objects']['width'])} / "
             f"{pct(B[s]['physical']['with_number_share_objects']['depth'])})")
    t += row("card audit physical plausible / implausible / unclear", lambda s: (lambda c: f"{c['physical'].get('plausible', 0)} / {c['physical'].get('implausible', 0)} / "
             f"{c['physical'].get('unclear', 0)}")(A[s]["cards"]))
    t += "\n### 4. Models (every non-simple object: RecGen FAST, internal profile)\n\n" + head
    t += row("non-simple objects: generated (own / look-alike copy) / observed surface", lambda s: (lambda m: f"{m['non_simple']}: "
             f"{m['non_simple_shown'].get('generated (own)', 0)} / {m['non_simple_shown'].get('generated (look-alike copy)', 0)} / "
             f"{m['non_simple_shown'].get('observed surface', 0)} ({pct(m['non_simple_generated_share'])} generated)")(q(s)["models"]))
    t += row("all cards by shown tier", lambda s: ", ".join(f"{k} {v}" for k, v in q(s)["models"]["all_cards_by_tier"].items()))
    t += row("RecGen models made / drawn (lined up) / not drawn", lambda s: (lambda m, d: f"{m['generated_models'] + (d or {}).get('not_lined_up', 0)} / "
             f"{m['generated_models']} / {(d or {}).get('not_lined_up', 0)}")(q(s)["models"], q(s)["models"].get("display_rule")))
    al = lambda s, k: q(s)["alignment"]["summary"].get(k) or {}  # noqa: E731
    t += row("alignment, every card at its best view: IoU median / p10", lambda s: f"{al(s, 'all')['iou_median']} / {al(s, 'all')['iou_p10']}")
    t += row("  generated models: IoU median / p10; offset median / p90 cm", lambda s: f"{al(s, 'generated')['iou_median']} / {al(s, 'generated')['iou_p10']}; "
             f"{al(s, 'generated')['offset_cm_median']} / {al(s, 'generated')['offset_cm_p90']}")
    t += row("  observed surfaces: IoU median / p10; offset median / p90 cm", lambda s: f"{al(s, 'observed surface')['iou_median']} / {al(s, 'observed surface')['iou_p10']}; "
             f"{al(s, 'observed surface')['offset_cm_median']} / {al(s, 'observed surface')['offset_cm_p90']}")
    t += row("  primitives: IoU median / p10", lambda s: f"{al(s, 'primitive')['iou_median']} / {al(s, 'primitive')['iou_p10']}")
    t += row("20 stratified models by eye: right / partial / wrong", lambda s: (lambda c: f"{c.get('right', 0)} / {c.get('partial', 0)} / {c.get('wrong', 0)}")(
        A[s]["models"]["shown"]["all"]))
    t += "\n### Times (s from the MP4 bytes in the container to the Volume commit; warm call, first call in brackets)\n\n" + head
    tw = lambda s: q(s)["times_warm"]  # noqa: E731
    tf = lambda s: q(s)["times_first"] or {}  # noqa: E731
    for label, k in (("first 3D", "first 3D"), ("cards v1", "cards v1"), ("all types (densify pass)", "types (densify pass)"),
                     ("tier 0 models (observed surfaces v1)", "tier 0 surfaces v1"), ("first generated model", "first generated model"),
                     ("all generated models in", "all generated models in (models final)"), ("call end", "call end")):
        t += row(label, lambda s, k=k: f"{tw(s).get(k)} ({tf(s).get(k)})")
    t += row("r4b warm: first 3D / cards v1 / types / models final", lambda s: " / ".join(str(B[s]["times"].get(k)) for k in ("first 3D", "cards v1", "types (densify pass)", "models final")))
    t += "\n### GPU and VLM (warm call)\n\n" + head
    t += row("GPU peak GiB 0 / 1; stages > 72", lambda s: f"{q(s)['gpu']['peak_gib'][0]} / {q(s)['gpu']['peak_gib'][1]}; {', '.join(q(s)['gpu']['stages_over_72']) or 'none'}")
    t += row("RecGen generations: n, span s", lambda s: (lambda g: f"{g['n']}, {g['span'][0]}-{g['span'][1]}")(q(s)["gpu"]["recgen"]["recgen.generate"]))
    for purpose in ("scene vocabulary", "naming (cluster medoids)", "event captions", "hazard / judge", "qwen identity decider"):
        t += row(f"VLM: {purpose} (requests [analysis s])", lambda s, p=purpose: (lambda v: f"{v['requests']} {v['spans_s'] if v['requests'] else ''}")(q(s)["vlm_requests"][p]))
    t += row("VLM requests before cards v1", lambda s: q(s)["vlm_requests"]["before cards v1 (all)"])
    return t


def vocab_md(res):
    V, B = res["videos"], res["r4b"]
    out = ["| video | source | wave-2 words | objects raw / cards | clean covered / delivered | held-out family right (r5b scorer) | vocab known s | cards v1 s | all types s |",
           "|---|---|---|---|---|---|---|---|---|"]
    for s in SITES:
        if s not in V:
            continue
        for label, r in (("r4b: Qwen words", B[s]["vocab_row"]), ("r5b/vocab: RAM++ (VLM-free)", B[s]["ram_row"]), ("this round: RAM++ then Qwen (union)", V[s]["vocab_row"])):
            h, c, tm = r["heldout"]["as shown"], r["instances"]["clean"], r["times"]
            out.append(f"| {NAME[s]} | {label} | {r['words']['wave2']} | {r['objects_raw']} / {r['shares']['object_cards']} | {c['covered']}/{c['delivered']} | "
                       f"{h['right_ext']}/{h['n']} = {h['acc_ext']:.2f} | {tm.get('vocab known (mark)')} | {tm.get('cards v1')} | {tm.get('types (densify pass)')} |")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--bench", required=True)
    ap.add_argument("--gt")
    ap.add_argument("--audit")
    ap.add_argument("--align", required=True)
    ap.add_argument("--visit", help="LAYER.json,GT.json")
    a = ap.parse_args()
    out = Path(a.out)
    res = {"videos": {}, "r4b": r4b_rows()}
    for site, d in (x.split("=") for x in a.bench.split(",")):
        res["videos"][site] = video(site, Path(d), a.align)
    if a.audit:
        res["audits"] = audits(a.audit)
    if a.gt:
        res["gt"] = gt(a.gt, out)
    if a.visit:
        res["visit"] = visit(*a.visit.split(","))
    (out / "summary.json").write_text(json.dumps(res, indent=1, default=str))
    (out / "tables.md").write_text(md(res) + "\n### Vocabulary sources (warm calls)\n\n" + vocab_md(res))
    print((out / "tables.md").read_text())


if __name__ == "__main__":
    main()
