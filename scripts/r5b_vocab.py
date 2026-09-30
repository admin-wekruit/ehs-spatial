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
    print("r5b_vocab self-check ok: type families (unidentified: none), both scorers, matching counts, bank freeze and merge")


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
    else:
        p.error(f"unknown command {a.cmd}")


if __name__ == "__main__":
    main()
