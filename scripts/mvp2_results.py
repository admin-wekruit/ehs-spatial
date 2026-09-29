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


def identity(benches, work):
    """The fresh held-out items (runs/mvp2-identity-final-001, agent-labelled blind before this branch existed) graded on the
    warm and first calls' final names. identity_study.score_final writes its metrics beside the items: it runs on copies in
    `work` (the identity branch's folder is never written)."""
    import shutil
    import identity_study as ids
    src = PHASE2 / "runs/mvp2-identity-final-001"
    work.mkdir(parents=True, exist_ok=True)
    for f in ("items.json", "labels-final.json"):
        shutil.copy(src / f, work / f)
    out = {}
    for kind in ("warm", "first"):
        now = {s: (Path(d).name, dict(calls(d))[kind]["run"]["report"]) for s, d in benches.items() if kind in dict(calls(d))}
        r = ids.score_final(work, now)
        (work / f"metrics-{kind}.json").write_text((work / "metrics-final.json").read_text())
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
            # per status and sequence, warm calls: 'at least' = cut by the frame edge / seen in parts, 'at most' = an unresolved size
            "bounds": {f"{st}/{q}": [sum(b["holds_a"] for r in sc["reports"] if not r["first_call"] and r["seq"] == q for b in r.get("bounds", []) if b["status"] == st),
                                     sum(1 for r in sc["reports"] if not r["first_call"] and r["seq"] == q for b in r.get("bounds", []) if b["status"] == st)]
                       for st in ("at least", "at most") for q in ("arkit47", "arkit42", "tum")},
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


def md_tables(res):
    """The measured tables of summary.md (the narrative around them is written by hand)."""
    L = ["## Analysis time (s from the MP4 bytes in the container to the Volume commit)", ""] + md_times(res)
    for name, vt in (res.get("variant_times") or {}).items():
        L += ["", f"Variant '{name}' (same code, one flag), warm (first call):", "",
              "| | " + " | ".join(NAME[s] for s in SITES) + " |", "|---|---|---|---|"]
        for row in ("objects v1", "cards v1", "identity complete", "densify names", "final judgements", "pick v2", "cards v3"):
            L.append(f"| {row} | " + " | ".join(pair(vt.get(s, {}), row) for s in SITES) + " |")
    ck = res.get("checks") or {}
    L += ["", "## Card checks (every cards version of every call)", "", "| | " + " | ".join(NAME[s] for s in SITES) + " |", "|---|---|---|---|"]
    for key, label in (("final_rows_on_a_check_the_final_class_does_not_apply", "final judgement rows on a check the card's final class does not apply (R1), warm (first)"),
                       ("final_cards_not_derived_from_their_name", "final cards apply_name would change (R1), warm (first)"),
                       ("contract_violations", "card contract violations over every cards version, warm (first)"),
                       ("cards_checked", "cards checked (all versions), warm (first)")):
        L.append(f"| {label} | " + " | ".join(f"{cell((ck.get(s) or {}).get('warm', {}).get(key), '{}')} ({cell((ck.get(s) or {}).get('first', {}).get(key), '{}')})"
                                               for s in SITES) + " |")
    idn = res.get("identity") or {}
    if idn:
        L += ["", "## Identity on the held-out items (runs/mvp2-identity-final-001, agent-labelled blind before this branch)", "",
              "| call | " + " | ".join(NAME[s] for s in SITES) + " | all |", "|---|---|---|---|---|"]
        for kind in ("warm", "first"):
            b, sh = idn[kind]["by_site"], idn[kind]["shares"]
            tot = {k: sum(b.get(s, {}).get(k, 0) for s in SITES) for k in ("n", "right", "close", "wrong")}
            L.append(f"| {kind}: right / close / wrong (n) | " + " | ".join(
                f"{b[s]['right']} / {b[s]['close']} / {b[s]['wrong']} ({b[s]['n']})" if s in b else "—" for s in SITES)
                + f" | {tot['right']} / {tot['close']} / {tot['wrong']} ({tot['n']}) |")
            L.append(f"| {kind}: right, right or close | " + " | ".join(
                f"{sh[s]['right']:.2f}, {sh[s]['right_or_close']:.2f}" if s in sh else "—" for s in SITES)
                + f" | {sh['all']['right']:.2f}, {sh['all']['right_or_close']:.2f} |")
        r1 = idn["warm"]["round1_same_objects"]
        L.append("| round 1 on the same objects: right, right or close | " + " | ".join(
            f"{r1[s]['right']:.2f}, {r1[s]['right_or_close']:.2f} (n {r1[s]['n']})" if s in r1 else "—" for s in SITES)
            + f" | {r1['all']['right']:.2f}, {r1['all']['right_or_close']:.2f} (n {r1['all']['n']}) |")
        hz = idn["warm"]["hazard_names"]
        L += ["", f"Hazard-class names on these items (warm): {hz['shown']} shown, {hz['shown_right']} right or same family, "
                  f"{hz['shown_unclear_truth']} on items labelled unclear; {hz['held_back']} held back, {hz['held_back_but_true']} of them true."]
    gt = res.get("physical_gt")
    if gt:
        L += ["", "## Physical values against metric ground truth (runs/mvp2-integrate-gt-*, 3 indoor scenes)", "",
              "Median |error| (a) as delivered (assumed 1.6 m camera height) -> (b) at the true camera height; n = shown values with GT depth "
              "on >= 50 % of their mask; cov = share within +-u (as delivered).", "",
              "| field | " + " | ".join(("ARKit 47333932", "ARKit 42445448", "TUM fr1 room (held out)")) + " |", "|---|---|---|---|"]
        tab = {(r["seq"], r["field"]): r for r in gt["table"]}
        for f in ("top_above_floor", "base_above_floor", "height", "width", "depth", "position_xy", "planar_slope_deg", "principal_axis_tilt_deg"):
            unit, k = ("deg", 1) if f.endswith("_deg") else ("cm", 100)
            L.append(f"| {f} | " + " | ".join(
                (lambda r: f"{r['a_med'] * k:.1f} -> {r['b_med'] * k:.1f} {unit} (n {r['n']}, cov {r['a_cov']:.2f})" if r else "—")(tab.get((q, f)))
                for q in ("arkit47", "arkit42", "tum")) + " |")
        L += ["", "u coverage by family and view-set state (k_geo fitted on ARKit only; TUM held out), as delivered:", "",
              "| family / view sets | ARKit 47333932 | ARKit 42445448 | TUM (held out) |", "|---|---|---|---|"]
        e2e = {(r["family"], r["view_sets"], r["seq"]): r for r in gt["end_to_end"]}
        for fam in ("height", "extent", "position", "angle"):
            for st in ("sets", "one_set"):
                cells = [e2e.get((fam, st, q)) for q in ("arkit47", "arkit42", "tum")]
                if any(cells):
                    L.append(f"| {fam} / {st} | " + " | ".join(f"{c['coverage']:.2f} (n {c['n']})" if c else "—" for c in cells) + " |")
        L += ["", "Bounds holding against GT (warm calls): " + "; ".join(f"{k} {v[0]}/{v[1]}" for k, v in gt["bounds"].items() if v[1]) + "."]
    pdv = res.get("physical_delivered") or {}
    if pdv:
        L += ["", "## Physical values against the delivered reports (same-object pairs; agreement, not truth), warm call", "",
              "| | " + " | ".join(NAME[s] for s in SITES) + " |", "|---|---|---|---|"]
        def vd(s, q):
            x = ((pdv.get(s) or {}).get("vs_delivered") or {}).get(q) or {}
            return f"{x.get('signed_median')} / {x.get('median')} / {x.get('p90')} (cov {x.get('coverage')}, n {x.get('n')})" if x else "—"
        L.append("| same-object pairs | " + " | ".join(str(((pdv.get(s) or {}).get("vs_delivered") or {}).get("pairs")) for s in SITES) + " |")
        L.append("| top m: signed median / median / p90 abs | " + " | ".join(vd(s, "top_card") for s in SITES) + " |")
        L.append("| base m: signed median / median / p90 abs | " + " | ".join(vd(s, "base_card") for s in SITES) + " |")
        L.append("| people: feet read (n), median signed m, share within u of 0 | " + " | ".join(
            (lambda p: f"{(p.get('feet_rays_m') or {}).get('n')}, {p.get('feet_rays_signed_median')}, {p.get('feet_within_u_of_0')}")((pdv.get(s) or {}).get("people") or {})
            for s in SITES) + " |")
        L.append("| person masks measured as pictures | " + " | ".join(str(((pdv.get(s) or {}).get("people") or {}).get("rejected_masks")) for s in SITES) + " |")
        L.append("| long objects: visible length shown / short side not measurable | " + " | ".join(
            (lambda x: f"{x.get('visible_length')} / {x.get('short_side_not_measurable')}")((pdv.get(s) or {}).get("long_objects") or {}) for s in SITES) + " |")
    ca = res.get("click_audit")
    if ca:
        L += ["", "## Click audit (independent of SAM 3: random pixel on a random frame, the viewer's pick rule; held-out seed 2, agent-labelled)", "",
              "| warm call | clicks | correct | wrong | miss | background | background-hit | resolved right | picked precision |", "|---|---|---|---|---|---|---|---|---|"]
        for r in [r for r in ca if r["sample"] == "random"]:
            L.append(f"| {NAME.get(r['site'], r['site'])} | {r['n']} | {r['correct']} | {r['wrong']} | {r['miss']} | {r['background']} | {r['background-hit']} | "
                     f"{r['click_correct_share']:.2f} | {r['picked_precision']:.2f} |")
        L += ["", "| person clicks (inside SAM 3 person references, seed 11) | real people | real person opens the person | pictures of people | pictures opened as a person |", "|---|---|---|---|---|"]
        for r in [r for r in ca if r["sample"] == "person refs"]:
            L.append(f"| {NAME.get(r['site'], r['site'])} | {r['real_people']} | {r['real_person_correct']:.2f} | {r['pictures']} | {r['pictures_as_person']} |")
    ju = res.get("judgements")
    if ju:
        L += ["", "## Judgements (final judgements patch; audit labels agent-made, blind to verdicts)", "",
              "| call | objects with a check | verdicts (rows) | PASS right / audited (+ unverifiable) | FAIL right / audited (+ unverifiable) |", "|---|---|---|---|---|"]
        for site, calls_ in ju.items():
            for kind, rep in calls_.items():
                pr, pa, pu, fr, fa, fu = (sum(b[k] for b in rep["by_check"].values()) for k in
                                          ("pass_right", "pass_audited", "pass_unverifiable", "fail_right", "fail_audited", "fail_unverifiable"))
                L.append(f"| {kind} | {rep['objects_with_a_check']} / {rep['objects']} ({rep['coverage']:.0%}) | {json.dumps(rep['counts'])} | {pr} / {pa} (+{pu}) | {fr} / {fa} (+{fu}) |")
        L += ["", "| check, warm call | " + " | ".join(NAME[s] for s in SITES) + " |", "|---|---|---|---|"]
        warm = {s: next((rep for kind, rep in (ju.get(s) or {}).items() if kind.endswith("warm")), None) for s in SITES}
        checks_ = sorted({c for rep in warm.values() if rep for c in rep["by_check"]})
        for c in checks_:
            def one(rep):
                b = (rep or {}).get("by_check", {}).get(c)
                if not b:
                    return "—"
                v = b["verdicts"]
                n = sum(v.values())
                dec = (v.get("PASS", 0) + v.get("FAIL", 0)) / n if n else 0
                return f"{n} rows, decided {dec:.0%} ({json.dumps(v)}); PASS {b['pass_right']}/{b['pass_audited']}+{b['pass_unverifiable']}, FAIL {b['fail_right']}/{b['fail_audited']}+{b['fail_unverifiable']}"
            L.append(f"| {c} | " + " | ".join(one(warm[s]) for s in SITES) + " |")
    vw = res.get("viewer")
    if vw:
        L += ["", "## Viewer (headless Chromium, the recording replayed at recorded speed; screenshots in viewer/)", "",
              "| report | click -> card p50 / p95 ms | pick decode ms | aimed clicks (screenshot: what the card says) |", "|---|---|---|---|"]
        for v in vw:
            shots = "; ".join(f"{Path(x['file']).name}: aimed {x['aimed']} -> '{x['title']}' [{', '.join(x['chips'])}]" for x in v["shots"])
            L.append(f"| {v['report']} | {v['latencyMs']['p50']:.1f} / {v['latencyMs']['p95']:.1f} | {v['pickDecodeMs']:.0f} | {shots} |")
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
           "times": times(bench), "checks": checks(bench), "identity": identity(bench, out / "identity-heldout"), "physical_gt": physical_gt(a.gt) if a.gt else None,
           "physical_delivered": physical_delivered(bench), "spend": spend(bench, a.gt, variants),
           "variant_times": {k: times(v) for k, v in variants.items()}}
    res["click_audit"] = json.loads((Path(a.clicks) / "score.json").read_text()) if a.clicks else None
    # judge_results_r2 per bench (DIR/<site>/run-results.json): the audit of each call against the agent labels
    res["judgements"] = {Path(f).parent.name: json.loads(Path(f).read_text())["audit"] for f in sorted(glob.glob(f"{a.judge}/*/run-results.json"))} if a.judge else None
    res["viewer"] = [v for f in sorted(glob.glob(f"{a.viewer}/*/click-check.json")) for v in json.loads(Path(f).read_text())["videos"]] if a.viewer else None
    (out / "summary.json").write_text(json.dumps(res, indent=1, default=str))
    (out / "tables.md").write_text("\n".join(md_tables(res)) + "\n")
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
