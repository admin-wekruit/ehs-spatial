"""r4 integrate results: summary.json + tables.md from the final benches (one per video: a first call after boot, then a warm
call), the ground-truth run, and the fresh audits by eye. The user's acceptance comes first: objects with a type (and the
family right), segmentation (click coverage), physical info (completeness, GT errors), a display model. Then times, GPU
peaks, instance quality against the delivered objects (before = round 3's final benches), naming, the card audit, spend.

Folders under OUT the audits fill (agent-labelled by looking at contact sheets):
    clicks/<site>/       click_audit.py sample + labels-*.json
    naming/              r4_naming_results.py sheets + audit/labels-agent.json (blind: ids only)
    cards/<site>/        r4_physical.py sheet --ids (the naming sample): labels.json rows get extent / physical labels
    viewer/<site>/       web/tests/r4-integrate-shots.mjs shots.json + png

    python scripts/r4_integrate_results.py OUT --bench me340=DIR,samsclub-a2=DIR,walmart=DIR [--gt DIR] \
        [--before me340=DIR,...]
    python scripts/r4_integrate_results.py --self-check
"""
import argparse
import collections
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "scripts")]
import mvp2_results as mr  # noqa: E402

SITES, NAME, OVER_GIB = mr.SITES, mr.NAME, 72.
ROWS = ("first 3D", "objects v1", "cards v1", "cards v3", "all names")
EXTENT = ("one object", "several objects", "piece", "not an object", "unclear")
PHYS = ("plausible", "implausible", "unclear")


def times(benches):
    import mvp3_results as m3
    t = m3.times(benches)
    for site, d in benches.items():
        for kind, c in mr.calls(d):
            st = c["run"].get("stages") or []
            peaks = [max(((s.get("peak_gb") or [0, 0])[g] or 0) for s in st) for g in (0, 1)]
            t[site][kind].update(gpu_peak_gib=peaks, over_72=sorted({s["stage"] for s in st if any((p or 0) > OVER_GIB for p in s.get("peak_gb") or [])}),
                                 boot_ready_s=(c["run"].get("boot") or {}).get("ready_s"), usd_call=c["run"].get("usd_estimate"))
    return t


def instances(benches, work):
    """r4_instances_eval on each call's pick maps and cards (subprocess: it is a CLI with its own data paths)."""
    import subprocess
    out = {}
    for site, d in benches.items():
        for kind, c in mr.calls(d):
            f = work / f"{site}-{kind}.json"
            if not f.exists():
                subprocess.run([sys.executable, str(REPO / "scripts/r4_instances_eval.py"), "--run", str(d), "--report", c["run"]["report"],
                                "--site", site, "--out", str(f)], check=True, capture_output=True)
            x = json.loads(f.read_text())
            out.setdefault(site, {})[kind] = {k: {kk: x[k][kk] for kk in ("delivered", "covered", "missed", "in_pieces", "wrong_merge_cards",
                                                                          "delivered_in_wrong_merges", "wholes_over_several")}
                                              for k in ("models", "clean", "all")} | {"cards": x.get("cards"), "parts": x.get("parts")}
    return out


def physical(benches):
    import r4_physical as rp
    out = {}
    for site, d in benches.items():
        comp = rp.completeness(Path(d) / "mirror")
        for kind, c in mr.calls(d):
            vs = comp.get(c["run"]["report"]) or {}
            last = vs[sorted(vs, key=lambda v: int(v[1:]))[-1]] if vs else None
            out.setdefault(site, {})[kind] = None if last is None else {
                "version": sorted(vs, key=lambda v: int(v[1:]))[-1], "violations": sum(x["violations"] for x in vs.values()),
                "required": {k: {kk: v.get(kk) for kk in ("cards", "complete", "complete_share", "with_number_share")} for k, v in last["required"].items()}}
    return out


def models(benches):
    import r4_models as rm
    out = {}
    for site, d in benches.items():
        tab = rm.table(d)
        for kind, c in mr.calls(d):
            x = tab.get(c["run"]["report"]) or {}
            out.setdefault(site, {})[kind] = {k: x.get(k) for k in ("object_cards", "with_primitive", "share", "kinds", "sam3d", "residual_cm")}
    return out


def card_audit(root):
    """cards/<site>/labels.json rows labelled 'extent' (EXTENT) and 'label' (PHYS) -> counts; names come from the blind naming audit."""
    out = {}
    for site in SITES:
        f = root / "cards" / site / "labels.json"
        if not f.exists():
            continue
        rows = json.loads(f.read_text())["rows"]
        out[site] = {"n": len(rows), "extent": dict(collections.Counter(r.get("extent") for r in rows)),
                     "physical": dict(collections.Counter(r.get("label") for r in rows)),
                     "by_identity_state": {s: dict(collections.Counter(r.get("label") for r in rows if r["state"] == s)) for s in {r["state"] for r in rows}}}
    return out


def clicks(root):
    import click_audit as ca
    out = {}
    for site in SITES:
        d = root / "clicks" / site
        if d.exists():
            for r in ca.score([d]):
                on = r["correct"] + r["wrong"] + r["miss"]
                bg = r["background"] + r["background-hit"]
                out[site] = {**r, "on_object_n": on, "background_n": bg, "background_kept": round(r["background"] / bg, 3) if bg else None}
    return out


def viewer(root):
    out = {}
    for site in SITES:
        f = root / "viewer" / site / "shots.json"
        if f.exists():
            d = json.loads(f.read_text())
            out[site] = {"report": d["report"], "errors": d["errors"], "shots": [{k: r.get(k) for k in ("label", "title", "unknown", "file", "error", "why")} for r in d["rows"]]}
    return out


def spend(benches, gt):
    out = {s: json.loads((Path(d) / "summary.json").read_text()).get("usd_estimate_upper") for s, d in benches.items() if (Path(d) / "summary.json").exists()}
    if gt:
        runs = json.loads((Path(gt) / "summary.json").read_text()).get("runs") or []
        out["gt"] = round(sum(r.get("usd_estimate") or 0 for r in runs), 2)
    return out


def cell(v, fmt="{:.1f}"):
    return "—" if v is None else fmt.format(v) if isinstance(v, float) else str(v)


def share(v):
    return cell(v, "{:.2f}")


def pair(x, row):
    return f"{cell(x.get('warm', {}).get(row))} ({cell(x.get('first', {}).get(row))})"


def md(res):
    L = ["## The user's acceptance (warm call; first call after boot in brackets)", "",
         "| | " + " | ".join(NAME[s] for s in SITES) + " |", "|---|" + "---|" * len(SITES)]
    nm, cl, ph, mo, na = res["naming"].get("videos", {}), res["clicks"], res["physical"], res["models"], res["naming"]

    def row(label, f):
        L.append(f"| {label} | " + " | ".join(f(s) for s in SITES) + " |")

    def two(f):
        return lambda s: f"{f(s, 'warm')} ({f(s, 'first')})"
    row("object cards", two(lambda s, k: cell(((nm.get(s) or {}).get(k) or {}).get("cards"))))
    row("with a type (a family)", two(lambda s, k: share(((nm.get(s) or {}).get(k) or {}).get("typed_share (a family)"))))
    row("with a specific name", two(lambda s, k: share(((nm.get(s) or {}).get(k) or {}).get("named_share (a specific name)"))))
    fa = (na.get("fresh_audit_warm") or {})
    row("family right, fresh 30-card audit (warm, blind)", lambda s: f"{(fa.get(s) or {}).get('family_right', '—')}/{(fa.get(s) or {}).get('n', '—')}")
    hw = ((na.get("heldout") or {}).get("warm") or {}).get("by_site") or {}
    row("family right, held-out items (warm)", lambda s: f"{(hw.get(s) or {}).get('family_right', '—')}/{(hw.get(s) or {}).get('n', '—')}")
    row("clicks on real objects opening the right card (fresh, warm)", lambda s: f"{(cl.get(s) or {}).get('correct', '—')}/{(cl.get(s) or {}).get('on_object_n', '—')}")
    row("clicks on background left unknown", lambda s: f"{(cl.get(s) or {}).get('background', '—')}/{(cl.get(s) or {}).get('background_n', '—')}")
    row("object cards complete (every physical field a value, bound or reasoned n/o)",
        two(lambda s, k: share((((ph.get(s) or {}).get(k) or {}).get("required", {}).get("object") or {}).get("complete_share"))))
    row("object cards with a display model", two(lambda s, k: share(((mo.get(s) or {}).get(k) or {}).get("share"))))
    row("SAM 3D meshes accepted on cards", two(lambda s, k: cell((((mo.get(s) or {}).get(k) or {}).get("sam3d") or {}).get("accepted_on_cards"))))
    L += ["", "## Times (s from the MP4 bytes in the container to the layer written; warm (first))", "",
          "| | " + " | ".join(NAME[s] for s in SITES) + " |", "|---|" + "---|" * len(SITES)]
    t = res["times"]
    for r_ in ROWS:
        row(f"{r_}, s", lambda s: pair(t.get(s, {}), r_))
    row("GPU peak GiB gpu0/gpu1 (warm; first)", lambda s: "; ".join("/".join(cell(v) for v in (t[s].get(k) or {}).get("gpu_peak_gib", [])) for k in ("warm", "first")))
    row("stages over 72 GiB", lambda s: ", ".join(sorted(set((t[s].get("warm") or {}).get("over_72", []) + (t[s].get("first") or {}).get("over_72", [])))) or "none")
    row("cold start (not analysis), s", lambda s: cell((t[s].get("first") or {}).get("boot_ready_s")))
    L += ["", "## Instances vs the delivered objects (covered/n, objects in pieces, wrong-merge cards (objects in them)); round 3 final -> this (warm)", "",
          "| set | " + " | ".join(NAME[s] for s in SITES) + " |", "|---|" + "---|" * len(SITES)]
    ins, bef = res["instances"], res.get("before") or {}

    def fmt(x):
        return "—" if not x else f"{x['covered']}/{x['delivered']}, {x['in_pieces']}, {x['wrong_merge_cards']}({x['delivered_in_wrong_merges']})"
    for k in ("models", "clean", "all"):
        row(k, lambda s: f"{fmt(((bef.get(s) or {}).get('warm') or {}).get(k))} -> {fmt(((ins.get(s) or {}).get('warm') or {}).get(k))}")
    row("cards", lambda s: f"{((bef.get(s) or {}).get('warm') or {}).get('cards', '—')} -> {((ins.get(s) or {}).get('warm') or {}).get('cards', '—')}")
    L += ["", "## Naming (the VLM last)", "", "| | " + " | ".join(NAME[s] for s in SITES) + " |", "|---|" + "---|" * len(SITES)]
    row("VLM questions per call", two(lambda s, k: cell(((nm.get(s) or {}).get(k) or {}).get("vlm_questions"))))
    row("routes (warm)", lambda s: json.dumps(((nm.get(s) or {}).get("warm") or {}).get("routes"), sort_keys=True).replace("|", "/"))
    row("fresh audit (warm, blind): right / close / wrong / unidentified of n", lambda s: "/".join(str((fa.get(s) or {}).get(k, "—")) for k in ("right", "close", "wrong", "unidentified", "n")))
    row("held-out items (warm): right / close / wrong / unidentified of n", lambda s: "/".join(str((hw.get(s) or {}).get(k, "—")) for k in ("right", "close", "wrong", "unidentified", "n")))
    ca = res["card_audit"]
    L += ["", "## Card audit by eye (the same 30 random object cards per video, warm; agent-labelled, not blind for extent and physical)", "",
          "| | " + " | ".join(NAME[s] for s in SITES) + " |", "|---|" + "---|" * len(SITES)]
    row("extent: " + " / ".join(EXTENT), lambda s: " / ".join(str(((ca.get(s) or {}).get("extent") or {}).get(e, 0)) for e in EXTENT))
    row("physical values vs the image: " + " / ".join(PHYS), lambda s: " / ".join(str(((ca.get(s) or {}).get("physical") or {}).get(e, 0)) for e in PHYS))
    L += ["", "## Spend (Modal list price, upper bound), $", "", " ".join(f"{k} {v}" for k, v in res["spend"].items()), ""]
    return "\n".join(L)


def self_check():
    t = {"warm": {"cards v3": 70.04}, "first": {"cards v3": 75.5}}
    assert pair(t, "cards v3") == "70.0 (75.5)" and pair(t, "all names") == "— (—)"
    res = {"naming": {"videos": {"me340": {"warm": {"cards": 300, "typed_share (a family)": .97, "vlm_questions": 40, "routes": {"vlm": 3}}}},
                      "fresh_audit_warm": {"me340": {"family_right": 25, "n": 29, "right": 10, "close": 9, "wrong": 5, "unidentified": 5}}},
           "clicks": {"me340": {"correct": 30, "on_object_n": 40, "background": 18, "background_n": 20}},
           "physical": {"me340": {"warm": {"required": {"object": {"complete_share": 1.0}}}}}, "models": {"me340": {"warm": {"share": 1.0}}},
           "times": {s: {"warm": {"cards v3": 1.}, "first": {}} for s in SITES}, "instances": {}, "card_audit": {"me340": {"extent": {"one object": 20}}},
           "spend": {"me340": 1.5}}
    m = md(res)
    assert "| with a type (a family) | 0.97 (—) |" in m and "| clicks on real objects opening the right card (fresh, warm) | 30/40 |" in m, m
    assert "| family right, fresh 30-card audit (warm, blind) | 25/29 |" in m and "20 / 0 / 0 / 0 / 0" in m
    print("r4_integrate_results self-check ok: acceptance, times, instances, naming and audit tables")


def main(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    benches = dict(x.split("=", 1) for x in a.bench.split(","))
    import r4_naming_results as nr
    res = {"benches": benches, "gt": a.gt, "times": times(benches), "instances": instances(benches, out / "instances"),
           "physical": physical(benches), "models": models(benches), "clicks": clicks(out), "card_audit": card_audit(out),
           "viewer": viewer(out), "spend": spend(benches, a.gt)}
    if a.before:
        res["before"] = instances(dict(x.split("=", 1) for x in a.before.split(",")), out / "instances-before")
    res["naming"] = nr.score(benches, out / "naming")
    if a.gt and (Path(a.gt) / "accuracy/score.json").exists():
        import r4_physical as rp
        sc = json.loads((Path(a.gt) / "accuracy/score.json").read_text())
        res["gt_table"] = sc["table"]
        (out / "gt.md").write_text(rp.gt_tables(sc))
    (out / "summary.json").write_text(json.dumps(res, indent=1, default=str))
    (out / "tables.md").write_text(md(res))
    print((out / "tables.md").read_text())


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out", nargs="?")
    p.add_argument("--bench", default="")
    p.add_argument("--before", default="")
    p.add_argument("--gt")
    p.add_argument("--self-check", action="store_true")
    a = p.parse_args()
    self_check() if a.self_check else main(a)
