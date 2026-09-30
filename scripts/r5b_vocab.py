"""r5b/vocab: the frozen naming bank (offline write-back only), the vocabulary comparison and the type results.

  bank freeze BASE.npz TEXT.npz OUT.npz     BASE's rows + TEXT's zero-shot classes (modal_apps/r5b_vocab.py) -> a snapshot
  bank merge BASE.npz ROWS.npz... OUT.npz   BASE + runs' bank-rows-<pass>.npz (the VLM answers a run wrote beside its report)
  score --runs site=RUN_DIR:REPORT,... --out OUT.json [--bank SNAPSHOT] [--replay]
                                            held-out and dev family accuracy of those runs' cards (both scorers, below);
                                            --replay: the family rule re-run on the cards' own votes (cards.family_vote)
  python scripts/r5b_vocab.py --self-check

Scorers (identity_study's held-out items, matched by box centre within 0.3 m, as round 4 did):
  r4b  a label's family is its class's (cards.FAMILY); an 'other:<name>' label has none, so it is never right (round 4's rule)
  ext  an 'other:<name>' label's family is the family its name maps to in the r5b taxonomy (OTHER below, fixed before any r5b
       run: the workshop classes and aliases added in r5b name these things)
"""
import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "scripts")]
from fast_report import cards  # noqa: E402

PHASE2 = Path("/Users/adam/Desktop/panoptes-public/research-notes/phase2")
SETS = {"test": PHASE2 / "runs/mvp2-results/identity-heldout", "dev": PHASE2 / "runs/mvp2-identity-study-001"}
# the 'other:' labels of both sets, each the family its thing belongs to (r5b, written before any r5b run)
OTHER = {"end mill holder": "tool", "tool holder item": "tool", "drill chuck": "tool", "air blow gun": "tool", "machine side panel": "machine",
         "machine window": "machine", "machine cover": "machine", "mill table": "machine", "way cover": "machine", "table leg": "furniture",
         "ceiling beam": "building", "roll of tape": None}
LABEL_FAMILY = {v: k for k, v in cards.TYPE_LABEL.items()}


# ---------------------------------------------------------------- the frozen bank

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze(base, text, out):
    """A snapshot: BASE's rows (emb, meta) with TEXT's zero-shot classes and text (every taxonomy class)."""
    b, t = np.load(base), np.load(text)
    np.savez(out, emb=b["emb"].astype(np.float16), meta=str(b["meta"]), text=t["zs_text"].astype(np.float32), text_scale=np.float32(t["scale"]),
             classes=np.array([str(c) for c in t["zs_classes"]]))
    return summary(out)


def merge(base, rows, out):
    """BASE + the rows files (bank-rows-<pass>.npz) -> a new snapshot; a row already in BASE (same video and card) is skipped."""
    b = np.load(base)
    emb, meta = [b["emb"].astype(np.float16)], json.loads(str(b["meta"]))
    have = {(m.get("video"), m.get("card")) for m in meta}
    for p in rows:
        z = np.load(p)
        m2 = json.loads(str(z["meta"]))
        keep = [i for i, m in enumerate(m2) if (m.get("video"), m.get("card")) not in have]
        emb.append(z["emb"][keep].astype(np.float16))
        meta += [m2[i] for i in keep]
        have |= {(m2[i].get("video"), m2[i].get("card")) for i in keep}
    np.savez(out, emb=np.concatenate(emb), meta=json.dumps(meta), text=b["text"], text_scale=b["text_scale"], classes=b["classes"])
    return summary(out)


def summary(path):
    z = np.load(path)
    meta = json.loads(str(z["meta"]))
    by = collections.Counter((m.get("family"), m.get("site"), str(m.get("source", ""))[:24]) for m in meta)
    return {"file": str(path), "sha256": sha(path), "rows": len(meta), "zero_shot_classes": len(z["classes"]),
            "rows_by_family_site_source": {" / ".join(map(str, k)): n for k, n in sorted(by.items())}}


# ---------------------------------------------------------------- scoring

def type_family(name):
    """A shown name -> the family it claims (r4_naming_results.type_family, plus r5b's 'unidentified (<shape>)': none)."""
    if not name or name.startswith("unidentified"):
        return None
    if name.endswith(cards.TYPE_ONLY):
        return LABEL_FAMILY.get(name[:-len(cards.TYPE_ONLY)])
    return cards.NOT_OBJECT if cards.canonical(name) == cards.NOT_OBJECT else cards.family_of(name)


def truth(label, scorer):
    """-> the set of right families, or None (unclear: not scored)."""
    if label["canon"] == "unclear":
        return None
    if label["canon"] == cards.NOT_OBJECT:
        return {cards.NOT_OBJECT}
    out = {cards.FAMILY.get(c) for c in [label["canon"], *label.get("also", [])] if cards.FAMILY.get(c)}
    if scorer == "ext" and label["canon"].startswith("other:"):
        out |= {OTHER.get(label["canon"][6:])} - {None}
    return out


def centres(run_dir, report):
    import fast_report_eval as ev
    out = {}
    for c in ev.load_layers(Path(run_dir), report)["object_cards"]["cards"]:
        if c["kind"] == "object":
            out[c["id"]] = (c, np.mean([c["physical"]["box_min_m"], c["physical"]["box_max_m"]], 0) if "box_min_m" in c["physical"] else None)
    return out


def match(which, now):
    """The labelled items of a set on the runs `now` {site: (run dir, report)}: [(item, label, card or None)], identity_study's
    rule (the item's card id when its box centre is within 0.3 m of the item's own run's, else the nearest centre within 0.3 m)."""
    d = SETS[which]
    items = json.loads((d / "items.json").read_text())
    lab = json.loads((d / ("labels-final.json" if which == "test" else "labels-heldout.json")).read_text())
    src = {s: centres(PHASE2 / "runs" / v[0], v[1]) for s, v in items["runs"].items() if s in now}
    cur = {s: centres(*v) for s, v in now.items()}
    out = []
    for r in items["items"]:
        s = r["site"]
        if s not in cur:
            continue
        ctr = src[s].get(r["card"], (None, None))[1]
        got = cur[s].get(r["card"])
        if got is None or ctr is None or got[1] is None or np.linalg.norm(got[1] - ctr) >= .3:
            cand = [(float(np.linalg.norm(c - ctr)), (cc, c)) for cc, c in cur[s].values() if c is not None and ctr is not None]
            best = min(cand, default=None, key=lambda x: x[0])
            got = best[1] if best is not None and best[0] < .3 else None
        out.append((r, lab[r["id"]], None if got is None else got[0]))
    return out


def score(rows, name_of=lambda c: c["identity"]["name"]):
    """-> {site: {n, matched, unmatched, typed, right (per scorer), precision (right / typed)}} over the labelled (not unclear) items."""
    out = {}
    for r, L, c in rows:
        if L["canon"] == "unclear":
            continue
        o = out.setdefault(r["site"], {"n": 0, "unmatched": 0, "typed": 0, "right_r4b": 0, "right_ext": 0, "n_ext": 0})
        if c is None:
            o["unmatched"] += 1
            continue
        o["n"] += 1
        f = type_family(name_of(c))
        o["typed"] += f is not None
        o["right_r4b"] += f is not None and f in truth(L, "r4b")
        o["right_ext"] += f is not None and f in truth(L, "ext")
    for o in out.values():
        o["acc_r4b"] = round(o["right_r4b"] / o["n"], 3) if o["n"] else None
        o["acc_ext"] = round(o["right_ext"] / o["n"], 3) if o["n"] else None
        o["precision_ext"] = round(o["right_ext"] / o["typed"], 3) if o["typed"] else None
    return out


# ---------------------------------------------------------------- one call's rows

def written_after(run, layer, t=None, version=None):
    """The Volume-commit time (written_s, analysis s) of a layer's first patch put at or after analysis time t (a mark), or of its
    version `version` (review finding 9: times are commits, not queue times)."""
    rows = sorted((x for x in run.get("layers") or [] if x["layer"] == layer and x.get("written_s") is not None), key=lambda x: x["seq"])
    if version is not None:
        return next((x["written_s"] for x in rows if x.get("version") == version), None)
    return None if t is None else next((x["written_s"] for x in rows if x.get("queued_s", x.get("sent_s", 0.)) >= t - .01), None)


def times(run):
    m = run.get("marks") or {}
    r = lambda v: None if v is None else round(v, 1)  # noqa: E731
    return {"first 3D": r(written_after(run, "room", 0.)), "objects v1": r(written_after(run, "objects", version=1)),
            "cards v1": r(written_after(run, "object_cards", version=1)), "types (first pass)": r(written_after(run, "object_cards", m.get("identity_cascade_put"))),
            "objects v3": r(written_after(run, "objects", m.get("objects_v3_put"))), "cards v3": r(written_after(run, "object_cards", m.get("cards_v3_put"))),
            "types (densify pass)": r(written_after(run, "object_cards", m.get("identity_cascade_densify_put"))),
            "first SAM 3D model": r(written_after(run, "models", m.get("first_model_put"))), "models final": r(written_after(run, "models", m.get("models_final_put"))),
            "splat preview": r(written_after(run, "splat", m.get("splat_preview_put"))), "call end": r(run.get("elapsed_s")),
            "vocab known (mark)": r(m.get("vocab_known")), "decode end": r(next((x["end_s"] for x in run["stages"] if x["stage"] == "decode"), None)),
            "sam3 done (mark)": r(m.get("sam3_done"))}


def stage_sum(run, prefix):
    xs = [x for x in run["stages"] if x["stage"].split("@")[0] == prefix]
    return {"n": len(xs), "gpu_s": round(sum(x["s"] for x in xs), 2), "span": [round(min(x["start_s"] for x in xs), 2), round(max(x["end_s"] for x in xs), 2)]} if xs else None


def vlm_requests(run):
    """Every VLM request of a call by purpose, with the analysis seconds it ran in (its stage's span), and how many ran before
    cards v1 was committed (the r5b target: 0)."""
    s = run.get("summary") or {}
    c1 = written_after(run, "object_cards", version=1)
    out = {}
    for purpose, prefixes, count in (("scene vocabulary", ("vlm.vocab",), lambda xs: len(xs)),
                                     ("event captions", ("vlm.events",), lambda xs: sum((x.get("n") or {}).get("windows", 0) for x in xs)),
                                     ("naming (cluster medoids)", ("naming.vlm.first", "naming.vlm.densify"), lambda xs: sum((x.get("n") or {}).get("questions", 0) for x in xs)),
                                     ("hazard / judge", ("judge.qwen", "judge.gemini", "judge.vlm"), lambda xs: len(xs)),
                                     ("qwen identity decider", ("vlm.identity", "vlm.identity.ehs", "vlm.identity.other"), lambda xs: len(xs))):
        xs = [x for x in run["stages"] if x["stage"].split("@")[0] in prefixes]
        n = count(xs)
        out[purpose] = {"requests": n, "spans_s": [[round(x["start_s"], 1), round(x["end_s"], 1)] for x in xs if count([x])],
                        "before_cards_v1": sum(count([x]) for x in xs if c1 is not None and x["start_s"] < c1)}
    out["naming escalated"] = sum(p.get("vlm_questions_escalated", 0) for p in (s.get("naming") or {}).get("passes", []))
    out["question log (by priority)"] = s.get("vlm_questions")
    out["before cards v1 (all)"] = sum(v["before_cards_v1"] for v in out.values() if isinstance(v, dict) and "before_cards_v1" in v)
    return out


def shares(cs):
    n = max(1, len(cs))
    fam = [type_family(c["identity"]["name"]) for c in cs]
    named = [c for c in cs if not c["identity"]["name"].endswith(cards.TYPE_ONLY) and not c["identity"]["name"].startswith("unidentified")]
    return {"object_cards": len(cs), "typed": round(sum(f is not None for f in fam) / n, 3), "specific_name": round(len(named) / n, 3),
            "unidentified": round(sum(c["identity"]["name"].startswith("unidentified") for c in cs) / n, 3),
            "top_names": collections.Counter(c["identity"]["name"] for c in cs).most_common(8)}


def call_row(mirror, rec, fit=None):
    """One call -> its rows: word source, objects, shares, clean coverage (the instances scorer), held-out and dev family accuracy
    (as shown; and with r5b's family rule replayed with `fit`), times (commits), SAM 3 work, VLM requests by purpose."""
    import fast_report_eval as ev
    import r4b_results as rr
    run, site = rec["run"], rec["site"]
    L = ev.load_layers(Path(mirror), run["report"])
    cs = [c for c in L["object_cards"]["cards"] if c["kind"] == "object"]
    now = {site: (str(mirror), run["report"])}
    test, dev = match("test", now), match("dev", now)
    voc = (run.get("summary") or {}).get("vocab") or {}
    out = {"site": site, "report": run["report"], "kind": rec.get("kind"), "source": rec.get("vocab") or voc.get("source"), "error": run.get("error"),
           "first_call_after_boot": (run.get("boot") or {}).get("first_call_after_boot"),
           "words": {"wave2": len((run.get("summary") or {}).get("vocab", {}).get("words") or []), "all": (run.get("summary") or {}).get("words"),
                     "list": voc.get("words"), "vocab_s": voc.get("s")},
           "objects_raw": len(L["objects"]["objects"]), "shares": shares(cs), "instances": rr.instances(Path(mirror).parent, run["report"], site),
           "heldout": {"as shown": score(test)[site] if site in score(test) else None},
           "dev": {"as shown": score(dev).get(site)}, "times": times(run),
           "sam3": {k: stage_sum(run, k) for k in ("sam3.person", "sam3.vocab.wave1", "sam3.vocab.wave2", "densify.sam3", "vocab.pe", "vlm.vocab")},
           "vlm": vlm_requests(run), "naming": {k: v for k, v in ((run.get("summary") or {}).get("naming") or {}).items() if k != "passes"},
           "gpu_peak_gib": [max((x.get("peak_gb") or [0, 0])[g] or 0 for x in run["stages"]) for g in (0, 1)],
           "over_72": sorted({x["stage"] for x in run["stages"] if any((p or 0) > 72 for p in x.get("peak_gb") or [])}),
           "usd_estimate_call": run.get("usd_estimate")}
    if fit is not None:
        out["heldout"]["r5b rule"] = score(test, lambda c: replay_name(c, fit)).get(site)
        out["dev"]["r5b rule"] = score(dev, lambda c: replay_name(c, fit)).get(site)
        out["shares_r5b_rule"] = shares([{**c, "identity": {**c["identity"], "name": replay_name(c, fit)}} for c in cs])
    return out


# ---------------------------------------------------------------- the family rule, replayed and fitted

def replay_name(c, fit):
    """A card's shown name under r5b's family rule (cards.family_vote with `fit`): a named card keeps its name; an unnamed one
    ('<family> (type only)' or a shape) gets its family type again from its own votes, or 'unidentified (<shape>)'."""
    i = c["identity"]
    n = i.get("naming") or {}
    if n.get("route") not in (None, "unidentified") or i.get("canonical"):
        return i["name"]
    fam, _ = cards.family_vote(sam3=(n.get("sam3") or [None])[0], yolo=(n.get("yolo") or [None])[0], zero_shot_family=n.get("zero_shot_family"),
                               bank=(n.get("bank") or [None])[0], fit=fit)
    return cards.TYPE_LABEL[fam] + cards.TYPE_ONLY if fam else cards.unidentified(cards.shape_type(c["physical"]))


def route_of(c):
    """The ungated family route an unnamed card takes ('zero-shot strong', 'sam3 word alone', ... or 'votes agree'), its family."""
    i = c["identity"]
    n = i.get("naming") or {}
    if n.get("route") not in (None, "unidentified") or i.get("canonical"):
        return "named", cards.FAMILY.get(i.get("canonical"))
    fam, src = cards.family_vote(sam3=(n.get("sam3") or [None])[0], yolo=(n.get("yolo") or [None])[0], zero_shot_family=n.get("zero_shot_family"),
                                 bank=(n.get("bank") or [None])[0], fit={})
    r = (src or "none").replace(" (family)", "")
    return (r if r in cards.SINGLE_ROUTES else "votes agree" if fam else "none"), fam


def fit_routes(rows, p_min=.6, n_min=3):
    """Dev items [(item, label, card)] -> the calibration: per single-vote route and family, n and right (ext scorer); a family's
    route is allowed at precision >= p_min with >= n_min items, else the route's pooled precision decides ('default')."""
    tab = {r: {} for r in cards.SINGLE_ROUTES}
    for r, L, c in rows:
        t = truth(L, "ext") if c is not None else None
        if t is None:
            continue
        route, fam = route_of(c)
        if route in tab and fam:
            x = tab[route].setdefault(fam, {"n": 0, "right": 0})
            x["n"] += 1
            x["right"] += fam in t
    out = {"schema": "panoptes-family-fit-v1", "p_min": p_min, "n_min": n_min, "routes": {}, "default": {}}
    for route, fams in tab.items():
        n, right = sum(x["n"] for x in fams.values()), sum(x["right"] for x in fams.values())
        out["default"][route] = bool(n) and right / n >= p_min
        out["routes"][route] = {f: {**x, "precision": round(x["right"] / x["n"], 3), "ok": x["right"] / x["n"] >= p_min}
                                for f, x in sorted(fams.items()) if x["n"] >= n_min}
        out["routes"][route]["_pooled"] = {"n": n, "right": right}
    return out


# ---------------------------------------------------------------- the comparison

NAME = {"me340": "ME340", "samsclub-a2": "Sam's Club", "walmart": "Walmart"}


def bench_calls(bench):
    """A bench folder's call records (call-*.json), prelude calls apart."""
    recs = [json.loads(p.read_text()) for p in sorted(Path(bench).glob("call-*.json"))]
    return [r for r in recs if r.get("kind") != "prelude"], [r for r in recs if r.get("kind") == "prelude"]


def compare(bench, r4b, out, fit_source="ram"):
    """Every word source's warm call per video, round 4's warm calls beside (the Qwen words as they were), the family rule fitted
    on the dev labels of fit_source's calls (all videos; and without ME340's, for ME340's no-leakage line) -> compare.json, .md."""
    calls, pre = bench_calls(bench)
    mirror = Path(bench) / "mirror"
    rows = [call_row(mirror, rec) for rec in calls]
    for site, d in r4b.items():
        rec = next(r for r in bench_calls(d)[0] if r["kind"] == "warm")
        rows.append({**call_row(Path(d) / "mirror", rec), "source": "r4b qwen (round 4 code)"})
    dev = [x for rec in calls if rec.get("vocab") == fit_source for x in match("dev", {rec["site"]: (str(mirror), rec["run"]["report"])})]
    fit = fit_routes(dev)
    fit_lovo = fit_routes([x for x in dev if x[0]["site"] != "me340"])
    for rec, row in zip(calls, rows):
        test = match("test", {rec["site"]: (str(mirror), rec["run"]["report"])})
        row["heldout"]["r5b rule"] = score(test, lambda c: replay_name(c, fit)).get(rec["site"])
        if rec["site"] == "me340":
            row["heldout"]["r5b rule, fit without ME340"] = score(test, lambda c: replay_name(c, fit_lovo)).get("me340")
    for row in rows[len(calls):]:
        test = match("test", {row["site"]: (str(Path(r4b[row["site"]]) / "mirror"), row["report"])})
        row["heldout"]["r5b rule"] = score(test, lambda c: replay_name(c, fit)).get(row["site"])
    res = {"bench": str(bench), "fit_source": fit_source, "fit": fit, "fit_without_me340": fit_lovo, "rows": rows,
           "prelude": [{"site": r["site"], "report": r["run"]["report"], "naming": ((r["run"].get("summary") or {}).get("naming") or {})} for r in pre]}
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / "compare.json").write_text(json.dumps(res, indent=1, default=str))
    (Path(out) / "compare.md").write_text(compare_md(res))
    return res


def compare_md(res):
    def f(x, k):
        return "-" if not x or x.get(k) is None else x[k]
    out = []
    for site in ("me340", "samsclub-a2", "walmart"):
        rs = [r for r in res["rows"] if r["site"] == site]
        if not rs:
            continue
        out.append(f"\n### {NAME[site]}\n\n| source | wave-2 words | objects (raw / cards) | clean covered / delivered, in pieces, wrong-merge | "
                   "held-out family right (r4b / ext scorer) / n, as shown | ... r5b rule (fit) | typed / specific | wave-2 + densify SAM 3 GPU s | "
                   "vocab known, s | cards v1, s | cards v3, s | types (densify), s | call end, s | VLM requests (vocab / events / naming) |\n"
                   "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in rs:
            cl, h, hr = r["instances"].get("clean") or {}, r["heldout"].get("as shown") or {}, r["heldout"].get("r5b rule") or {}
            w2, de = r["sam3"].get("sam3.vocab.wave2") or {}, r["sam3"].get("densify.sam3") or {}
            v = r["vlm"]
            out.append(f"| {r['source']} | {r['words']['wave2']} | {r['objects_raw']} / {r['shares']['object_cards']} | "
                       f"{f(cl, 'covered')}/{f(cl, 'delivered')}, {f(cl, 'in_pieces')}, {f(cl, 'wrong_merge_cards')} | "
                       f"{f(h, 'right_r4b')} / {f(h, 'right_ext')} / {f(h, 'n')} | {f(hr, 'right_r4b')} / {f(hr, 'right_ext')} / {f(hr, 'n')} "
                       f"(typed {f(hr, 'typed')}) | {r['shares']['typed']:.2f} / {r['shares']['specific_name']:.2f} | "
                       f"{round((w2.get('gpu_s') or 0) + (de.get('gpu_s') or 0), 1)} | {r['times']['vocab known (mark)']} | {r['times']['cards v1']} | "
                       f"{r['times']['cards v3']} | {r['times']['types (densify pass)']} | {r['times']['call end']} | "
                       f"{v['scene vocabulary']['requests']} / {v['event captions']['requests']} / {v['naming (cluster medoids)']['requests']} |")
    return "\n".join(out) + "\n"


def final(bench, r4b, out):
    """The final bench (r5b defaults) against round 4's benches, per video: types (typed share, held-out family accuracy, both
    scorers, precision), clean coverage, VLM requests by purpose with their times, milestone commits (the added seconds), GPU peaks,
    the on-demand clicks. -> final.json, final.md."""
    calls, _ = bench_calls(bench)
    rows = {}
    for rec in calls:
        row = call_row(Path(bench) / "mirror", rec)
        row["clicks"] = rec.get("clicks")
        rows.setdefault(rec["site"], {})[rec["kind"]] = row
    base = {}
    for site, d in r4b.items():
        for rec in bench_calls(d)[0]:
            base.setdefault(site, {})[rec["kind"]] = call_row(Path(d) / "mirror", rec)
    res = {"bench": str(bench), "r4b": r4b, "now": rows, "r4b_rows": base}
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / "final.json").write_text(json.dumps(res, indent=1, default=str))
    (Path(out) / "final.md").write_text(final_md(res))
    return res


def final_md(res):
    sites = [s for s in ("me340", "samsclub-a2", "walmart") if s in res["now"]]
    head = "| | " + " | ".join(NAME[s] for s in sites) + " |\n|---|" + "---|" * len(sites) + "\n"

    def row(label, fn):
        cells = []
        for s in sites:
            try:
                cells.append(fn(s))
            except (KeyError, TypeError, ZeroDivisionError, IndexError):
                cells.append("-")
        return f"| {label} | " + " | ".join(cells) + " |\n"
    now = lambda s: res["now"][s].get("warm") or next(iter(res["now"][s].values()))  # noqa: E731
    old = lambda s: res["r4b_rows"][s]["warm"]  # noqa: E731
    pct = lambda x: f"{100 * x:.0f}%"  # noqa: E731
    acc = lambda h, k: f"{h['right_' + k]}/{h['n']} = {h['right_' + k] / h['n']:.2f}"  # noqa: E731
    md = "### Types (warm calls; r4b in brackets)\n\n" + head
    md += row("object cards", lambda s: f"{now(s)['shares']['object_cards']} ({old(s)['shares']['object_cards']})")
    md += row("typed by family", lambda s: f"{pct(now(s)['shares']['typed'])} ({pct(old(s)['shares']['typed'])})")
    md += row("specific name", lambda s: f"{pct(now(s)['shares']['specific_name'])} ({pct(old(s)['shares']['specific_name'])})")
    md += row("'unidentified (<shape>)'", lambda s: f"{pct(now(s)['shares']['unidentified'])} ({pct(old(s)['shares']['unidentified'])})")
    for k, lab in (("r4b", "round 4's scorer: 'other:' labels never right"), ("ext", "r5b scorer: 'other:' labels by the r5b taxonomy")):
        md += row(f"held-out family right / n ({lab})", lambda s, k=k: f"{acc(now(s)['heldout']['as shown'], k)} ({acc(old(s)['heldout']['as shown'], k)})")
    md += row("held-out: typed / precision (ext)", lambda s: f"{now(s)['heldout']['as shown']['typed']} / {now(s)['heldout']['as shown']['precision_ext']} "
              f"({old(s)['heldout']['as shown']['typed']} / {old(s)['heldout']['as shown']['precision_ext']})")
    md += "\n### Clean delivered objects (the instances scorer; r4b in brackets)\n\n" + head
    cl = lambda r: r["instances"]["clean"]  # noqa: E731
    md += row("covered / delivered, in pieces, wrong-merge cards", lambda s: f"{cl(now(s))['covered']}/{cl(now(s))['delivered']}, {cl(now(s))['in_pieces']}, "
              f"{cl(now(s))['wrong_merge_cards']} ({cl(old(s))['covered']}/{cl(old(s))['delivered']}, {cl(old(s))['in_pieces']}, {cl(old(s))['wrong_merge_cards']})")
    md += "\n### VLM requests by purpose (warm call: count [analysis s spans]; r4b in brackets)\n\n" + head
    for purpose in ("scene vocabulary", "event captions", "naming (cluster medoids)", "hazard / judge", "qwen identity decider"):
        md += row(purpose, lambda s, p=purpose: f"{now(s)['vlm'][p]['requests']} {now(s)['vlm'][p]['spans_s'] or ''} ({old(s)['vlm'][p]['requests']})")
    md += row("before cards v1 (all purposes)", lambda s: f"{now(s)['vlm']['before cards v1 (all)']} ({old(s)['vlm']['before cards v1 (all)']})")
    md += "\n### Times (s from the MP4 bytes in the container to the Volume commit; warm call, r4b warm in brackets)\n\n" + head
    for k in ("first 3D", "objects v1", "cards v1", "types (first pass)", "objects v3", "cards v3", "types (densify pass)", "first SAM 3D model", "models final",
              "splat preview", "call end", "vocab known (mark)", "decode end"):
        md += row(k, lambda s, k=k: f"{now(s)['times'][k]} ({old(s)['times'][k]})")
    md += row("GPU peak GiB (0 / 1); stages > 72", lambda s: f"{now(s)['gpu_peak_gib'][0]:.1f} / {now(s)['gpu_peak_gib'][1]:.1f}; {', '.join(now(s)['over_72']) or 'none'}")
    return md


def self_check():
    import tempfile
    assert type_family("unidentified (compact object)") is None and type_family("machine (type only)") == "machine"
    assert type_family("vise") == "machine" and type_family("floor") == cards.NOT_OBJECT
    L = {"canon": "other:tool holder item", "also": [], "name": "mill tool holder"}
    assert truth(L, "r4b") == set() and truth(L, "ext") == {"tool"}, "an 'other:' label is scorable only by the ext scorer"
    assert all(v is None or cards.family_of(k) in (v, None) or k in ("drill chuck",) for k, v in OTHER.items()), \
        {k: cards.family_of(k) for k in OTHER}  # the taxonomy maps them where it knows them
    rows = [({"site": "s"}, {"canon": "vise", "also": []}, {"identity": {"name": "machine (type only)"}}),
            ({"site": "s"}, L, {"identity": {"name": "machine tool holder"}}),
            ({"site": "s"}, {"canon": "fan", "also": []}, {"identity": {"name": "unidentified (compact object)"}}),
            ({"site": "s"}, {"canon": "unclear"}, None), ({"site": "s"}, {"canon": "bin", "also": []}, None)]
    got = score(rows)["s"]
    assert (got["n"], got["unmatched"], got["typed"], got["right_r4b"], got["right_ext"]) == (3, 1, 2, 1, 2), got
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        np.savez(t / "base.npz", emb=np.ones((2, 4), np.float16), meta=json.dumps([{"video": "v", "card": "a", "family": "retail", "site": "s"},
                                                                                     {"video": "v", "card": "b", "family": "retail", "site": "s"}]))
        np.savez(t / "text.npz", zs_text=np.ones((3, 4), np.float32), zs_classes=np.array(["box", "vise", "fan"]), scale=np.float32(99.))
        a = freeze(t / "base.npz", t / "text.npz", t / "A.npz")
        assert a["rows"] == 2 and a["zero_shot_classes"] == 3 and a["sha256"] == sha(t / "A.npz")
        np.savez(t / "rows.npz", emb=np.zeros((2, 4), np.float16), meta=json.dumps([{"video": "v", "card": "a"}, {"video": "w", "card": "c", "family": "shop floor"}]))
        b = merge(t / "A.npz", [t / "rows.npz"], t / "B.npz")
        assert b["rows"] == 3 and b["sha256"] != a["sha256"], "a row already in the base is skipped"
    # the family rule replayed: a sure zero-shot storage over a lone 'metal part' word; the fit gates a route per family
    c = {"identity": {"name": "material / part (type only)", "canonical": None, "naming": {"route": "unidentified", "sam3": ["metal part", .7, "metal block"],
                                                                                         "zero_shot_family": ["storage", .99], "yolo": [None, 0.], "bank": [None, 0.]}},
         "physical": {"height": {"value": .5, "u": .01}, "width": {"value": .6, "u": .01}}}
    assert replay_name(c, {}) == "shelf / rack / storage (type only)" and route_of(c) == ("zero-shot strong", "storage")
    gate = {"routes": {"zero-shot strong": {"storage": {"ok": False}}, "sam3 word alone": {"material": {"ok": False}}}}
    assert replay_name(c, gate) == "unidentified (compact object)", "p 0.99 is the strong band's only (the bands are disjoint)"
    lab = {"canon": "cabinet", "also": []}
    f = fit_routes([({"site": "s"}, lab, c)] * 3 + [({"site": "s"}, {"canon": "fan", "also": []}, c)], p_min=.6, n_min=3)
    assert f["routes"]["zero-shot strong"]["storage"] == {"n": 4, "right": 3, "precision": .75, "ok": True} and f["default"]["zero-shot strong"]
    print("r5b_vocab self-check ok: type families (unidentified: none), both scorers, matching counts, bank freeze and merge, the family "
          "rule replayed and fitted")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", nargs="*")
    p.add_argument("--self-check", action="store_true")
    a = p.parse_args()
    if a.self_check:
        return self_check()
    if a.cmd[:2] == ["bank", "freeze"]:
        print(json.dumps(freeze(*a.cmd[2:5]), indent=1))
    elif a.cmd[:2] == ["bank", "merge"]:
        print(json.dumps(merge(a.cmd[2], a.cmd[3:-1], a.cmd[-1]), indent=1))
    elif a.cmd[:1] == ["final"]:  # final BENCH OUT r4b=site:DIR,...
        r4b = dict(x.split(":", 1) for x in a.cmd[3].split(","))
        print(final_md(final(a.cmd[1], r4b, a.cmd[2])))
    elif a.cmd[:1] == ["compare"]:  # compare BENCH OUT r4b=site:DIR,... [FIT_SOURCE]: the family fit goes to OUT/family_calibration.json
        r4b = dict(x.split(":", 1) for x in a.cmd[3].split(",")) if len(a.cmd) > 3 else {}
        res = compare(a.cmd[1], r4b, a.cmd[2], *(a.cmd[4:5] or ["ram"]))
        (Path(a.cmd[2]) / "family_calibration.json").write_text(json.dumps({**res["fit"], "fitted_on": f"dev labels (runs/mvp2-identity-study-001) "
                                                                          f"on {res['fit_source']}'s calls of {a.cmd[1]}"}, indent=1) + "\n")
        print(compare_md(res))
    else:
        p.error(f"unknown command {a.cmd}")


if __name__ == "__main__":
    main()
