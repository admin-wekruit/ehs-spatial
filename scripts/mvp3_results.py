"""mvp3 integrate results: summary.json + tables.md from the final benches (one per video: a first call after boot, then a warm
call), the fresh audits (clicks with on-demand, random object cards, every FAIL and a sample of PASS), the hazard-ask replay
(round 2's rule vs this branch's on round 2's run-007 cards) and round 2's own relay records. The physical-vs-GT rows are
round 2's (mvp2-results; no ground-truth run here). summary.md's narrative is written by hand around tables.md.

    python scripts/mvp3_results.py OUT --bench me340=DIR,samsclub-a2=DIR,walmart=DIR --clicks DIR --ondemand DIR --seed N \
        --cards DIR --judge DIR --asks DIR --viewer DIR
    python scripts/mvp3_results.py --self-check
"""
import argparse
import glob
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "scripts")]
import mvp2_results as mr  # noqa: E402

SITES, NAME = mr.SITES, mr.NAME
RUNS = mr.PHASE2 / "runs"
ROUND2 = {"me340": "mvp2-integrate-me340-007", "samsclub-a2": "mvp2-integrate-samsclub-a2-007", "walmart": "mvp2-integrate-walmart-007"}
ROWS = ("first 3D", "objects v1", "cards v1", "all names", "final judgements")


def times(benches):
    """mvp2's times per call, plus 'all names': the last naming pass's cards (densify's names, else the first pass's)."""
    t = mr.times(benches)
    for site in t.values():
        for x in site.values():
            x["all names"] = x.get("densify names") or x.get("identity complete")
    return t


def coverage(benches):
    import judge_audit_mvp3 as ja
    import judge_offline as jo
    import judge_replay_mvp3 as jr
    from judge_results_mvp3 import d6
    out = {}
    for site, run in benches.items():
        for kind, c in mr.calls(run):
            d = jo.load(run=Path(run), report=c["run"]["report"])
            rep = jr.summarize(ja.final_rows(Path(run), c["run"]["report"]), d["cards"], d["ctx"])
            out.setdefault(site, {})[kind] = {**{k: rep[k] for k in ("objects", "objects_with_a_check", "objects_pass_or_fail", "counts", "by_check", "no_check", "priority")},
                                              "fails": len(rep["fails"]), **d6(d["cards"])}
    return out


def asks_now(benches):
    """What each bench call asked its deciders: Qwen identity and judgement questions (vLLM's own counts and mean prompt tokens),
    the judge's hazard questions new / cached / not asked (decided), and Gemini (none: no key the report container may use)."""
    out = {}
    for site, run in benches.items():
        for kind, c in mr.calls(run):
            s, js = c["run"]["summary"], [j for j in c["run"]["summary"].get("judge") or [] if j]
            q = s.get("vlm_questions") or {}
            ident = {k: v for k, v in q.items() if k.startswith("identity")}
            judg = {k: v for k, v in q.items() if k.startswith("judgement")}
            out.setdefault(site, {})[kind] = {
                "gemini_requests": 0,
                "qwen_identity_questions": sum(v["n"] for v in ident.values()),
                "qwen_identity_prompt_tokens": round(sum(v["n"] * v["prompt_tokens_mean"] for v in ident.values())),
                "qwen_hazard_questions_new": sum(j.get("qwen_questions", 0) for j in js),
                "qwen_hazard_images_new": sum(j.get("qwen_images", 0) for j in js),
                "qwen_hazard_answers_reused": sum(j.get("qwen_cached", 0) for j in js),
                "hazard_rows_not_asked_decided": [j.get("skipped_decided") for j in js],
                "qwen_judgement_prompt_tokens": round(sum(v["n"] * v["prompt_tokens_mean"] for v in judg.values())),
                "judge_runs": len(js)}
    return out


def round2_asks():
    """Round 2's relayed Gemini traffic per video (both calls): naming requests (every copy), the objects on them (one sheet
    each), tokens; hazard requests and counted input tokens."""
    out = {}
    for site, name in ROUND2.items():
        run = RUNS / name
        nm = [json.loads(Path(f).read_text()) for f in glob.glob(str(run / "namer/*/*.json"))]
        u = [x.get("usage") or {} for x in nm]
        ev = [json.loads(x) for x in (run / "hazard-relay-events.jsonl").read_text().splitlines()]
        b = [e for e in ev if e.get("phase") == "budget_evidence"]
        out[site] = {"naming_requests": len(nm), "naming_images": sum(x.get("n") or 0 for x in nm),
                     "naming_prompt_tokens": sum(x.get("prompt_token_count") or 0 for x in u),
                     "naming_output_tokens": sum((x.get("candidates_token_count") or 0) + (x.get("thoughts_token_count") or 0) for x in u),
                     "hazard_requests": len(b), "hazard_input_tokens": sum(e["data"].get("inputTokens") or 0 for e in b)}
    return out


def replay_asks(asks):
    """Round 2's rule vs this branch's, replayed on round 2's run-007 cards (hazard_asks_mvp3.py), both calls summed; tokens at the
    run's own relayed tokens per evidence image under round 2's rule."""
    before = json.loads((Path(asks) / "before-mvp2-rule-on-run007.json").read_text())
    after = json.loads((Path(asks) / "after-rule-on-run007.json").read_text())
    out = {}
    for site, name in ROUND2.items():
        b, a = before[name], after[name]
        tot = lambda r: {k: sum(c[k] for c in r["calls"].values()) for k in ("requests", "images", "questions")}  # noqa: E731
        tb, ta = tot(b), tot(a)
        per_img = b["measured_both_calls"]["input_tokens"] / tb["images"] if tb["images"] else None
        out[site] = {"round2_rule": tb, "this_rule": ta, "round2_measured": b["measured_both_calls"],
                     "tokens_per_image": round(per_img) if per_img else None,
                     "this_rule_input_tokens_est": round(ta["images"] * per_img) if per_img else None}
    return out


def clicks(clicks_dir, ondemand_dir, seed):
    import click_audit as ca
    import click_ondemand as co
    base = {r["site"]: r for r in ca.score([str(Path(clicks_dir) / s) for s in SITES])}
    od = co.score(ondemand_dir, clicks_dir, seed=seed) if ondemand_dir else {}
    return {"baseline": base, "on_demand": od}


def md(res):
    L = ["## Per video (warm call; first call after boot in brackets)", "", "| | " + " | ".join(NAME[s] for s in SITES) + " |", "|---|---|---|---|"]
    t = res["times"]
    for row in ROWS:
        L.append(f"| {row}, s | " + " | ".join(mr.pair(t.get(s, {}), row) for s in SITES) + " |")
    L.append("| GPU peak GiB gpu0/gpu1 (max over stages) | " + " | ".join(
        "{} ({})".format(*("/".join(mr.cell(g) for g in (t.get(s, {}).get(k) or {}).get("gpu_peak_gib", [])) or "—" for k in ("warm", "first"))) for s in SITES) + " |")
    L.append("| stages over 72 GiB | " + " | ".join(", ".join(sorted({x for k in ("warm", "first") for x in (t.get(s, {}).get(k) or {}).get("over_72", [])})) or "none" for s in SITES) + " |")
    L.append("| cold start (not analysis), s | " + " | ".join(mr.cell(res["boot"].get(s)) for s in SITES) + " |")
    ca = res.get("clicks") or {}
    if ca:
        b, od = ca["baseline"], ca["on_demand"]
        L.append("| clicks: random, real objects opening the right thing, report only -> with on-demand | " + " | ".join(
            (lambda o: f"{o['real_right_baseline']}/{o['real_objects']} -> {o['real_right_with_on_demand']}/{o['real_objects']}" if o else "—")(od.get(s)) for s in SITES) + " |")
        L.append("| clicks: background kept unknown / fixture named / false object card | " + " | ".join(
            (lambda o: "{unknown} / {fixture named} / {false object} of {n}".format(n=o["background"], **o["background_with_on_demand"]) if o else "—")(od.get(s)) for s in SITES) + " |")
        L.append("| clicks: picked precision (report's own entities) | " + " | ".join(mr.cell((b.get(s) or {}).get("picked_precision"), "{:.2f}") for s in SITES) + " |")
        L.append("| on-demand time to card p50 / p95 / max, s | " + " | ".join(
            "/".join(str(x) for x in (od.get(s) or {}).get("time_to_card_s_p50_p95_max") or ["—"]) for s in SITES) + " |")
        L.append("| on-demand names on right/coarse masks: right / close / wrong / none | " + " | ".join(
            " / ".join(map(str, (od.get(s) or {}).get("on_demand_names_right_close_wrong_none") or [])) or "—" for s in SITES) + " |")
    idn = res.get("identity") or {}
    for kind in ("warm", "first"):
        if idn.get(kind):
            sh = idn[kind]["shares"]
            L.append(f"| identity held-out right, right-or-close ({kind}) | " + " | ".join(
                f"{sh[s]['right']:.2f}, {sh[s]['right_or_close']:.2f} (n {idn[kind]['by_site'][s]['n']})" if s in sh else "—" for s in SITES) + " |")
    cv = res.get("coverage") or {}
    L.append("| objects with a check (warm) | " + " | ".join(
        (lambda v: f"{v['objects_with_a_check']}/{v['objects']} ({100 * v['objects_with_a_check'] / max(1, v['objects']):.0f}%)" if v else "—")((cv.get(s) or {}).get("warm")) for s in SITES) + " |")
    L.append("| objects with an overall PASS / FAIL (warm) | " + " | ".join(
        (lambda v: f"{v['objects_pass_or_fail']}/{v['objects']} ({100 * v['objects_pass_or_fail'] / max(1, v['objects']):.1f}%)" if v else "—")((cv.get(s) or {}).get("warm")) for s in SITES) + " |")
    L.append("| verdict rows (warm) | " + " | ".join(json.dumps((cv.get(s) or {}).get("warm", {}).get("counts")) for s in SITES) + " |")
    ju = res.get("judge_audit") or {}
    for kind in ("warm", "first"):
        cells = []
        for s in SITES:
            v = ju.get(f"{s.split('-')[0]} {kind}")
            cells.append("—" if not v else f"PASS {v['all'].get('pass_right', 0)}/{v['all'].get('pass_audited', 0)} (of {v['of_pass']}); "
                                            f"FAIL {v['all'].get('fail_right', 0)}/{v['all'].get('fail_audited', 0)} (of {v['of_fail']})")
        L.append(f"| judgement audit ({kind}): right / audited | " + " | ".join(cells) + " |")
    cd = res.get("cards_audit") or {}
    L.append("| card audit (20 random, warm): name right/close/wrong/unclear; physical plausible/implausible/unclear; judgement right/wrong/none | " + " | ".join(
        (lambda v: "{}; {}; {}".format(*("/".join(str(v[k].get(x, 0)) for x in xs) for k, xs in (("name", ("right", "close", "wrong", "unclear")),
                                                                                          ("physical", ("plausible", "implausible", "unclear")),
                                                                                          ("judgement", ("right", "wrong", "none"))))) if v else "—")(cd.get(s)) for s in SITES) + " |")
    L.append("| Modal list-price upper bound per bench, $ | " + " | ".join(mr.cell(res["spend"].get(s), "{}") for s in SITES) + " |")
    ra, r2, now = res.get("replay_asks") or {}, res.get("round2_asks") or {}, res.get("asks_now") or {}
    if ra:
        L += ["", "## What goes to the VLMs (both calls of a video)", "",
              "| | " + " | ".join(NAME[s] for s in SITES) + " |", "|---|---|---|---|"]
        L.append("| round 2 measured, Gemini naming: requests (with copies) / images / prompt tokens | " + " | ".join(
            f"{r2[s]['naming_requests']} / {r2[s]['naming_images']} / {r2[s]['naming_prompt_tokens']}" for s in SITES) + " |")
        L.append("| round 2 measured, Gemini hazard: requests / counted input tokens | " + " | ".join(
            f"{r2[s]['hazard_requests']} / {r2[s]['hazard_input_tokens']}" for s in SITES) + " |")
        L.append("| replay on round 2's cards, round 2's rule: hazard requests / images / questions | " + " | ".join(
            "{requests} / {images} / {questions}".format(**ra[s]["round2_rule"]) for s in SITES) + " |")
        L.append("| replay, this branch's rule: hazard requests / images / questions (input tokens, est.) | " + " | ".join(
            "{requests} / {images} / {questions}".format(**ra[s]["this_rule"]) + f" ({ra[s]['this_rule_input_tokens_est']})" for s in SITES) + " |")
        L.append("| this round's benches: Gemini requests | " + " | ".join(str(sum(v["gemini_requests"] for v in (now.get(s) or {}).values())) for s in SITES) + " |")
        L.append("| this round's benches: Qwen identity questions (prompt tokens) | " + " | ".join(
            f"{sum(v['qwen_identity_questions'] for v in (now.get(s) or {}).values())} ({sum(v['qwen_identity_prompt_tokens'] for v in (now.get(s) or {}).values())})" for s in SITES) + " |")
        L.append("| this round's benches: Qwen hazard questions new / reused / images new | " + " | ".join(
            f"{sum(v['qwen_hazard_questions_new'] for v in (now.get(s) or {}).values())} / {sum(v['qwen_hazard_answers_reused'] for v in (now.get(s) or {}).values())} / "
            f"{sum(v['qwen_hazard_images_new'] for v in (now.get(s) or {}).values())}" for s in SITES) + " |")
    return L


def self_check():
    t = {"me340": {"warm": {"densify names": None, "identity complete": 50.0}, "first": {"densify names": 70.0, "identity complete": 60.0}}}
    for x in t["me340"].values():
        x["all names"] = x.get("densify names") or x.get("identity complete")
    assert mr.pair(t["me340"], "all names") == "50.0 (70.0)"
    L = md({"times": t, "boot": {}, "spend": {}})
    assert L[7].startswith("| all names, s | 50.0 (70.0) |"), L[7]
    print("mvp3_results self-check ok: all-names row, table")


def main(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    bench = {k: RUNS / v if not Path(v).is_absolute() else Path(v) for k, v in (x.split("=", 1) for x in a.bench.split(","))}
    res = {"schema": "mvp3-integrate-results-v1", "bench": {k: str(v) for k, v in bench.items()}, "times": times(bench), "checks": mr.checks(bench),
           "identity": mr.identity(bench, out / "identity-heldout"), "coverage": coverage(bench), "asks_now": asks_now(bench),
           "round2_asks": round2_asks(), "replay_asks": replay_asks(a.asks) if a.asks else None,
           "boot": {s: (json.loads((d / "summary.json").read_text()).get("boot") or {}).get("ready_s") for s, d in bench.items()},
           "spend": {s: json.loads((d / "summary.json").read_text()).get("usd_estimate_upper") for s, d in bench.items()},
           "physical_gt_round2": json.loads((RUNS / "mvp2-results/summary.json").read_text()).get("physical_gt")}
    res["clicks"] = clicks(a.clicks, a.ondemand, a.seed) if a.clicks else None
    res["cards_audit"] = __import__("card_audit_mvp3").score([str(Path(a.cards) / s) for s in SITES if (Path(a.cards) / s / "labels.jsonl").exists()]) if a.cards else None
    res["judge_audit"] = json.loads((Path(a.judge) / "score.json").read_text()) if a.judge and (Path(a.judge) / "score.json").exists() else None
    res["viewer"] = [v for f in sorted(glob.glob(f"{a.viewer}/*/click-check.json")) for v in json.loads(Path(f).read_text())["videos"]] if a.viewer else None
    (out / "summary.json").write_text(json.dumps(res, indent=1, default=str))
    (out / "tables.md").write_text("\n".join(md(res)) + "\n")
    print((out / "tables.md").read_text())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", nargs="?")
    for k in ("bench", "clicks", "ondemand", "cards", "judge", "asks", "viewer"):
        ap.add_argument(f"--{k}")
    ap.add_argument("--seed", type=int, default=61)
    ap.add_argument("--self-check", action="store_true")
    a = ap.parse_args()
    self_check() if a.self_check else main(a)
