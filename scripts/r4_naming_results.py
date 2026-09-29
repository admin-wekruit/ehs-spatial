"""r4/naming results: the naming cascade (the VLM last) per video, first call and warm call, against round 2 (Gemini named
every object).

  sheets: a seeded sample of 30 object cards of each warm call -> OUT/audit/sheets/*.jpg (ids only: blind to the names)
          + OUT/audit/items.json; the agent labels them by looking (OUT/audit/labels-agent.json, identity_study's label format)
  score:  OUT/summary.json + OUT/tables.md: routes (cheap / VLM / copy / unidentified), VLM questions vs round 2's objects
          sent to Gemini, the held-out audited items (runs/mvp2-results/identity-heldout, agent-labelled blind before any
          round-4 run), the fresh audit, time to names, per-stage GPU peaks (> 72 GiB flagged)

    python scripts/r4_naming_results.py sheets --runs me340=RUNS/r4-naming-me340-001,... --out RUNS/r4-naming-results
    python scripts/r4_naming_results.py score --runs ... --out RUNS/r4-naming-results
"""
import argparse
import collections
import glob
import json
import shutil
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "scripts")]
from fast_report import cards  # noqa: E402

PHASE2 = Path("/Users/adam/Desktop/panoptes-public/research-notes/phase2")
CLIP = {"me340": "me340-165", "samsclub-a2": "samsclub-337", "walmart": "walmart-190"}
ROUND2 = {s: PHASE2 / f"runs/mvp2-integrate-{s}-007" for s in CLIP}
HELDOUT = PHASE2 / "runs/mvp2-results/identity-heldout"
N_AUDIT, SEED, OVER_GIB = 30, 4, 72.


def calls(bench):
    out = [json.loads(Path(f).read_text()) for f in sorted(glob.glob(str(Path(bench) / "call-*.json")))]
    return {c["kind"]: c for c in sorted(out, key=lambda c: c["call"])}


def final_cards(bench, report):
    import fast_report_eval as ev
    return [c for c in ev.load_layers(Path(bench), report)["object_cards"]["cards"] if c["kind"] == "object"]


LABEL_FAMILY = {v: k for k, v in cards.TYPE_LABEL.items()}


def type_family(name):
    """A shown name -> the family it claims: a '<label> (type only)' name its label's family, a shape type None, a specific
    name its class's family, 'not an object' itself."""
    if not name:
        return None
    if name.endswith(cards.TYPE_ONLY):
        return LABEL_FAMILY.get(name[:-len(cards.TYPE_ONLY)])
    return cards.NOT_OBJECT if cards.canonical(name) == cards.NOT_OBJECT else cards.family_of(name)


def is_named(name):
    return bool(name) and not name.endswith(cards.TYPE_ONLY) and name != cards.UNIDENTIFIED


def route(c):
    i = c["identity"]
    n = (i.get("naming") or {}).get("route")
    if not is_named(i.get("name")):
        kind = "type only (family)" if type_family(i.get("name")) else "type only (shape)"
        return kind if n in (None, "unidentified") else f"{kind}, name held back (hazard gate)"
    return n or i.get("decided_by")


def family_grade(name, label):
    if label["canon"] == "unclear":
        return None
    f = type_family(name)
    if label["canon"] == cards.NOT_OBJECT:
        return f == cards.NOT_OBJECT
    return f is not None and f in {cards.FAMILY.get(c) for c in [label["canon"], *label.get("also", [])] if cards.FAMILY.get(c)}


def times(run):
    """s from the MP4 bytes in the container to each patch written: cards v1, the first pass's names, densify's names."""
    lay, m = run["layers"], run.get("marks") or {}
    cps = [x for x in lay if x["layer"] == "object_cards"]
    after = lambda t: next((x["written_s"] for x in cps if t is not None and x["queued_s"] >= t - .01), None)  # noqa: E731
    st = collections.defaultdict(float)
    for s in run["stages"]:
        if s["stage"].startswith("naming."):
            st[s["stage"]] += s["s"]
    peaks = [max(((s.get("peak_gb") or [0, 0])[g] or 0) for s in run["stages"]) for g in (0, 1)]
    over = sorted({s["stage"] for s in run["stages"] if any((p or 0) > OVER_GIB for p in s.get("peak_gb") or [])})
    stage_peaks = {s["stage"]: s["peak_gb"] for s in run["stages"] if s["stage"].startswith("naming.") and s.get("peak_gb")}
    r = lambda v: None if v is None else round(v, 1)  # noqa: E731
    return {"objects v1": r(next((x["written_s"] for x in lay if x["layer"] == "objects" and x["version"] == 1), None)),
            "cards v1": r(next((x["written_s"] for x in cps if x["version"] == 1), None)),
            "names (first pass)": r(after(m.get("identity_cascade_put"))), "cards v3": r(m.get("cards_v3_put")),
            "names (densify pass)": r(after(m.get("identity_cascade_densify_put"))),
            "naming stage s": {k: round(v, 2) for k, v in st.items()}, "gpu_peak_gib": peaks, "over_72_gib": over, "naming_stage_peaks_gib": stage_peaks}


def round2(site):
    out = {}
    for kind, c in calls(ROUND2[site]).items():
        s = c["run"]["summary"]
        out[kind] = {"objects_sent_to_gemini": (s.get("identity") or {}).get("asked", 0) + (s.get("identity_densify") or {}).get("asked", 0),
                     "gemini_requests": (s.get("identity") or {}).get("requests", 0) + (s.get("identity_densify") or {}).get("requests", 0),
                     "cards": len(final_cards(ROUND2[site], c["run"]["report"]))}
    return out


def heldout(runs, work, kind):
    """identity_study.score_final on copies of the held-out items, the shown names of these runs."""
    import identity_study as ids
    work.mkdir(parents=True, exist_ok=True)
    for f in ("items.json", "labels-final.json"):
        shutil.copy(HELDOUT / f, work / f)
    now = {s: (str(Path(d).relative_to(PHASE2 / "runs")), calls(d)[kind]["run"]["report"]) for s, d in runs.items() if kind in calls(d)}
    if not now:  # score_final would grade the items' own round-2 runs instead
        return {}, [], []
    r = ids.score_final(work, now)
    lab = json.loads((HELDOUT / "labels-final.json").read_text())
    out = {}
    for x in r["per_item"]:
        if x["grade"] is None:
            continue
        o = out.setdefault(x["id"].split(":")[0], {"n": 0, "typed": 0, "family_right": 0, "right": 0, "close": 0, "wrong": 0, "unidentified": 0,
                                                   "named_n": 0, "named_right": 0, "named_close": 0})
        o["n"] += 1
        o[x["grade"]] += 1
        o["typed"] += type_family(x["name"]) is not None
        o["family_right"] += bool(family_grade(x["name"], lab[x["id"]]))
        if not is_named(x["name"]):
            o["unidentified"] += 1
        else:
            o["named_n"] += 1
            o["named_right"] += x["grade"] == "right"
            o["named_close"] += x["grade"] == "close"
    return out, r["not_matched"], r["per_item"]


def sheets(runs, out):
    """30 object cards of each warm call, seeded: the marked best view (judge.som) | the frame with the box; ids only."""
    import cv2
    from fast_report import judge
    import fast_report_eval as ev
    from mvp_sheets import frames_bgr, sheet
    dest = out / "audit"
    (dest / "sheets").mkdir(parents=True, exist_ok=True)
    rng, items = np.random.default_rng(SEED), []
    for site, d in runs.items():
        rep = calls(d)["warm"]["run"]["report"]
        L = ev.load_layers(Path(d), rep)
        outl = {}
        for f in L["outlines"]["frames"]:
            if f["source"] != "segmented":
                continue
            for o in f["objects"]:
                if o["polygons"]:
                    outl.setdefault(o["entityId"], {})[f["sourceFrame"]] = o["polygons"]
        cs = [c for c in L["object_cards"]["cards"] if c["kind"] == "object" and outl.get(c["id"])]
        pick = sorted(rng.choice(len(cs), min(N_AUDIT, len(cs)), replace=False).tolist())
        chosen = []
        for i in pick:
            c = cs[i]
            per = outl[c["id"]]
            q = next((k for k in (c.get("views") or {}).get("best", []) if k in per), None)
            if q is None:
                q = max(per, key=lambda k: np.prod(np.ptp(np.concatenate([np.asarray(p, float).reshape(-1, 2) for p in per[k]]), 0)))
            chosen.append((c, q, per[q]))
        imgs = frames_bgr(PHASE2 / "data/clips" / CLIP[site] / "source-full.mp4", {q for _, q, _ in chosen})
        tiles = []
        for c, q, polys in chosen:
            k = len(items)
            som = judge.som(imgs[q], {1: polys}, subject=1, side=420, encode=False)
            som = cv2.copyMakeBorder(som, 0, 420 - som.shape[0], 0, 420 - som.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255))
            fr = imgs[q].copy()
            xy = np.concatenate([np.asarray(p, float).reshape(-1, 2) for p in polys])
            (x0, y0), (x1, y1) = xy.min(0).astype(int), xy.max(0).astype(int)
            cv2.rectangle(fr, (x0, y0), (x1, y1), (0, 230, 255), 4)
            fr = cv2.copyMakeBorder(cv2.resize(fr, (420, 236)), 0, 184, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            tiles.append((np.hstack([som, fr]), f"R{k:02d}"))
            items.append({"id": f"R{k:02d}", "site": site, "report": rep, "card": c["id"], "frame": int(q),
                          "box_min_m": c["physical"].get("box_min_m"), "box_max_m": c["physical"].get("box_max_m")})
        print(site, len(chosen), "cards sampled of", len(cs), flush=True)
    for s in range(0, len(tiles), 10):
        sheet(tiles[s:s + 10], dest / "sheets" / f"audit-{s // 10:02d}.jpg", cols=2, tile=840)
    (dest / "items.json").write_text(json.dumps({"seed": SEED, "n_per_video": N_AUDIT, "items": items}, indent=1))


def fresh_audit(runs, out):
    """The agent's blind labels of the sampled warm-call cards -> the shown names graded (identity_study.grade)."""
    import identity_study as ids
    dest = out / "audit"
    if not (dest / "labels-agent.json").exists():
        return None
    items = json.loads((dest / "items.json").read_text())["items"]
    lab = json.loads((dest / "labels-agent.json").read_text())
    by = {}
    for site, d in runs.items():
        rep = calls(d)["warm"]["run"]["report"]
        by[site] = {c["id"]: c for c in final_cards(d, rep)}
    res, rows = {}, []
    for it in items:
        L, c = lab[it["id"]], by[it["site"]].get(it["card"])
        o = res.setdefault(it["site"], {"sampled": 0, "unclear": 0, "n": 0, "typed": 0, "family_right": 0, "right": 0, "close": 0, "wrong": 0,
                                        "unidentified": 0, "named_n": 0, "named_right": 0, "named_close": 0, "by_route": {}})
        o["sampled"] += 1
        if L["canon"] == "unclear" or c is None:
            o["unclear"] += 1
            continue
        name = c["identity"]["name"]
        g = ids.grade(name, L, name == cards.NOT_OBJECT)
        o["n"] += 1
        o[g] += 1
        fg = bool(family_grade(name, L))
        o["typed"] += type_family(name) is not None
        o["family_right"] += fg
        rt = route(c)
        o["by_route"].setdefault(rt, {"n": 0, "family_right": 0, "right": 0, "close": 0})
        o["by_route"][rt]["n"] += 1
        o["by_route"][rt]["family_right"] += fg
        o["by_route"][rt][g] = o["by_route"][rt].get(g, 0) + 1
        if not is_named(name):
            o["unidentified"] += 1
        else:
            o["named_n"] += 1
            o["named_right"] += g == "right"
            o["named_close"] += g == "close"
        rows.append({**it, "name": name, "route": rt, "truth": L["name"], "canon": L["canon"], "grade": g, "family_right": fg})
    (dest / "graded.json").write_text(json.dumps(rows, indent=1))
    return res


def score(runs, out):
    res = {"videos": {}, "round2": {s: round2(s) for s in runs}}
    for site, d in runs.items():
        for kind, c in calls(d).items():
            run, s = c["run"], c["run"]["summary"] or {}
            cs = final_cards(d, run["report"])
            rt = collections.Counter(route(x) for x in cs)
            rules = collections.Counter((x["identity"].get("naming") or {}).get("rule") for x in cs if (x["identity"].get("naming") or {}).get("route") == "cheap")
            nm = s.get("naming") or {}
            res["videos"].setdefault(site, {})[kind] = {
                "report": run["report"], "error": run.get("error"), "cards": len(cs), "routes": dict(rt), "cheap_rules": dict(rules),
                "typed_share (a family)": round(sum(type_family(x["identity"]["name"]) is not None for x in cs) / max(1, len(cs)), 3),
                "type_sources": dict(collections.Counter((x["identity"].get("type") or {}).get("source") for x in cs)),
                "named_share (a specific name)": round(sum(is_named(x["identity"]["name"]) for x in cs) / max(1, len(cs)), 3),
                "vlm_questions": sum(p.get("vlm_questions", 0) for p in nm.get("passes", [])), "passes": nm.get("passes"),
                "bank_rows_at_load": nm.get("bank_rows_at_load"), "bank_rows_now": nm.get("bank_rows_now"), "naming_error": nm.get("error"),
                "times": times(run), "boot_ready_s": (run.get("boot") or {}).get("ready_s"),
                "first_call_after_boot": (run.get("boot") or {}).get("first_call_after_boot")}
    res["heldout"] = {}
    for kind in ("warm", "first"):
        g, nm, per = heldout(runs, out / f"heldout-{kind}", kind)
        res["heldout"][kind] = {"by_site": g, "not_matched": nm}
        (out / f"heldout-{kind}" / "per-item.json").write_text(json.dumps(per, indent=1))
    res["fresh_audit_warm"] = fresh_audit(runs, out)
    (out / "summary.json").write_text(json.dumps(res, indent=1, default=str))
    return res


def vector_check(runs, npz_dir, x13_dir):
    """Do the pipeline's DINOv2-L card vectors meet the bank's (x13's crops of round 2's cards)? A card of this run and a
    round-2 card with box centres within 0.3 m are the same object: their cosine against the cosine to every other bank card
    of that video. npz_dir/<report>/naming-first.npz: the pass's vectors (fetched from the layers volume)."""
    import x13_naming as X
    D = X.Data(Path(x13_dir))
    V = D.vec("dinov2-l", "plain+masked")
    import fast_report_eval as ev
    out = {}
    for site, d in runs.items():
        rep = calls(d)["warm"]["run"]["report"]
        f = Path(npz_dir) / rep / "naming-first.npz"
        if not f.exists():
            continue
        z = np.load(f)
        vec = {i: v.astype(np.float32) for i, v in zip(z["ids"], z["dino"]) if np.any(v)}
        ctr = lambda c: np.mean([c["physical"]["box_min_m"], c["physical"]["box_max_m"]], 0) if "box_min_m" in c["physical"] else None  # noqa: E731
        now = {c["id"]: ctr(c) for c in final_cards(d, rep) if c["id"] in vec}
        r2 = {c["id"]: ctr(c) for c in ev.load_layers(ROUND2[site], calls(ROUND2[site])["warm"]["run"]["report"])["object_cards"]["cards"] if c["kind"] == "object"}
        at = {D.meta[i]["card"]: i for i in range(D.n) if D.site[i] == site}
        same, other, top1 = [], [], []
        for cid, c in now.items():
            if c is None:
                continue
            best = min(((float(np.linalg.norm(c - x)), k) for k, x in r2.items() if x is not None and k in at), default=None)
            if best and best[0] < .3:
                sims = V[[at[k] for k in at]] @ vec[cid]
                j = list(at).index(best[1])
                same.append(float(sims[j]))
                other.append(float(np.median(np.delete(sims, j))))
                top1.append(int(np.argmax(sims)) == j)
        out[site] = {"pairs": len(same), "cos_same_object_median": round(float(np.median(same)), 3) if same else None,
                     "cos_other_cards_median": round(float(np.median(other)), 3) if other else None,
                     "same_is_top1_share": round(float(np.mean(top1)), 3) if top1 else None}
    return out


NAME = {"me340": "ME340", "samsclub-a2": "Sam's Club", "walmart": "Walmart"}


def md(res):
    """tables.md: types first (the user's acceptance), then VLM calls, specific names, time."""
    sh = lambda a, b: "—" if not b else f"{a / b:.2f}"  # noqa: E731
    V, L = res["videos"], []
    sites = [s for s in NAME if s in V]
    head = "| | " + " | ".join(NAME[s] for s in sites) + " |"
    sep = "|---|" + "---|" * len(sites)

    def row(label, f):
        cells = []
        for s in sites:
            w, fi = V[s].get("warm"), V[s].get("first")
            cells.append(f"{f(w) if w else '—'} ({f(fi) if fi else '—'})")
        return f"| {label} | " + " | ".join(cells) + " |"
    L += ["## Every object's card, warm call (first call)", "", head, sep,
          row("object cards", lambda x: x["cards"]),
          row("share with a type (a family)", lambda x: x["typed_share (a family)"]),
          row("share with a specific name", lambda x: x["named_share (a specific name)"]),
          row("VLM questions (Qwen3-VL-8B, group medoids)", lambda x: x["vlm_questions"]),
          row("names (first pass) written, s", lambda x: x["times"]["names (first pass)"]),
          row("names (densify pass) written, s", lambda x: x["times"]["names (densify pass)"]),
          row("GPU peaks gpu0/gpu1 GiB", lambda x: "/".join(f"{g:.1f}" for g in x["times"]["gpu_peak_gib"])),
          row("stages over 72 GiB", lambda x: ", ".join(x["times"]["over_72_gib"]) or "none")]
    r2 = res["round2"]
    L += ["", "Round 2 (mvp2-integrate-*-007) sent every named object to Gemini: " + "; ".join(
        f"{NAME[s]} {r2[s]['warm']['objects_sent_to_gemini']} objects in {r2[s]['warm']['gemini_requests']} requests (warm)" for s in sites if s in r2) + "."]
    L += ["", "Routes (warm): " + "; ".join(f"{NAME[s]} {V[s]['warm']['routes']}" for s in sites if V[s].get("warm")),
          "", "Type sources (warm): " + "; ".join(f"{NAME[s]} {V[s]['warm']['type_sources']}" for s in sites if V[s].get("warm"))]
    for kind in ("warm", "first"):
        h = (res.get("heldout") or {}).get(kind, {}).get("by_site", {})
        if not h:
            continue
        L += ["", f"## Held-out audited items (runs/mvp2-results/identity-heldout, agent-labelled blind before round 4), {kind} call", "",
              "| | n | typed | family right | specific right | right or close | cards without a specific name | named: right / right or close |",
              "|---|---|---|---|---|---|---|---|"]
        for s in sites:
            if s in h:
                x = h[s]
                L.append(f"| {NAME[s]} | {x['n']} | {sh(x['typed'], x['n'])} | {sh(x['family_right'], x['n'])} | {sh(x['right'], x['n'])} | "
                         f"{sh(x['right'] + x['close'], x['n'])} | {x['unidentified']} | {sh(x['named_right'], x['named_n'])} / "
                         f"{sh(x['named_right'] + x['named_close'], x['named_n'])} (n {x['named_n']}) |")
    fa = res.get("fresh_audit_warm")
    if fa:
        L += ["", "## Fresh audit: 30 random object cards a video, warm call (agent-labelled blind from contact sheets)", "",
              "| | labelled (unclear) | typed | family right | specific right | right or close |", "|---|---|---|---|---|---|"]
        for s in sites:
            if s in fa:
                x = fa[s]
                L.append(f"| {NAME[s]} | {x['n']} ({x['unclear']}) | {sh(x['typed'], x['n'])} | {sh(x['family_right'], x['n'])} | {sh(x['right'], x['n'])} | "
                         f"{sh(x['right'] + x['close'], x['n'])} |")
        L += ["", "By route: " + "; ".join(f"{NAME[s]} {fa[s]['by_route']}" for s in sites if s in fa)]
    vc = res.get("vector_check")
    if vc:
        L += ["", "Pipeline vectors vs the bank's (same object in round 2, box centres within 0.3 m): " + "; ".join(
            f"{NAME[s]} {x}" for s, x in vc.items())]
    return "\n".join(L) + "\n"


def self_check():
    c = lambda name, r=None: {"identity": {"name": name, "naming": {"route": r} if r else None, "decided_by": "x"}}  # noqa: E731
    assert route(c("machine (type only)", "unidentified")) == "type only (family)" and route(c("flat panel (type only)")) == "type only (shape)"
    assert route(c("cable / hose / pipe (type only)", "vlm")) == "type only (family), name held back (hazard gate)"
    assert route(c("box", "copy")) == "copy" and route(c("box")) == "x"
    assert type_family("container / box / goods (type only)") == "goods" and type_family("lathe") == "machine" and type_family("flat panel (type only)") is None
    assert family_grade("tool (type only)", {"canon": "hand tool", "also": []}) and not family_grade("flat panel (type only)", {"canon": "sign"})
    run = {"layers": [{"layer": "objects", "version": 1, "written_s": 20., "queued_s": 19.}, {"layer": "object_cards", "version": 1, "written_s": 30., "queued_s": 29.},
                      {"layer": "object_cards", "version": 2, "written_s": 50., "queued_s": 49.5}],
           "marks": {"identity_cascade_put": 49.4}, "stages": [{"stage": "naming.signals", "s": 3., "peak_gb": [60., 75.]}]}
    t = times(run)
    assert t["names (first pass)"] == 50. and t["over_72_gib"] == ["naming.signals"] and t["naming stage s"] == {"naming.signals": 3.}
    print("r4_naming_results self-check ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sheets", "score"], nargs="?")
    ap.add_argument("--runs", default="")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--npz", help="folder with <report>/naming-first.npz (from the layers volume): the vector check")
    ap.add_argument("--x13", default="/private/tmp/claude-501/-Users-adam-Desktop-panoptes-public/1fd9a1db-e580-4bfc-8110-119a1cc38a99/scratchpad/x13")
    ap.add_argument("--self-check", action="store_true")
    a = ap.parse_args()
    if a.self_check:
        self_check()
    else:
        runs = dict(x.split("=") for x in a.runs.split(",") if x)
        a.out.mkdir(parents=True, exist_ok=True)
        if a.cmd == "sheets":
            sheets(runs, a.out)
        else:
            res = score(runs, a.out)
            if a.npz:
                res["vector_check"] = vector_check(runs, a.npz, a.x13)
            (a.out / "summary.json").write_text(json.dumps(res, indent=1, default=str))
            (a.out / "tables.md").write_text(md(res))
            print(md(res))
