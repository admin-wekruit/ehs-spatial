"""r4/naming: fit the naming cascade's thresholds (CPU), held out by video, on the x13 crop dataset (round 2's warm cards,
Gemini's names as the labels) with the YOLOE detections (modal_apps/r4_naming.py); export the label bank's seed rows.

Rule set (the round-4 direction: the VLM last, never by default; X13 memo + its verifier corrections):
  - the SAM 3 word is the free first vote; a card is accepted only when a second, independent vote gives the same class:
      'sam3+bank'           the family bank's DINOv2-L k-NN label (earlier VLM answers of OTHER videos), score min(shares)
      'sam3+zero-shot/yolo' PE-Core-L zero-shot top-1 or the YOLOE class matched to the mask, score the SAM 3 share
    (no single-vote rule: 'sam3 word' alone was 0.28 held out; no 'bank alone' rule)
  - the rest: clustered per video (DINOv2-L, average linkage, cosine cut), each cluster split by its members' SAM 3 class;
    ONE VLM call per group medoid, its answer copied only to members whose SAM 3 word gives the same class; a member the
    answer contradicts is asked on its own (the pipeline's escalation); an escaped answer leaves the group 'unidentified'
  - thresholds from the same domain family's other video (retail: Sam's Club <-> Walmart); a family with no other
    audited video (ME340, shop floor) is uncalibrated: no cheap acceptance, clusters at the cut fitted on the other videos
  - bank rows: VLM answers only (here round 2's Gemini names), tagged with the family

  python scripts/r4_naming_fit.py X13_DIR --calibration fast_report/naming_calibration.json --results RUNS/r4-naming-fit/results.json \
      --bank X13_DIR/bank-dinov2-l-v1.npz
  python scripts/r4_naming_fit.py --self-check
"""
import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "scripts")]
import x13_naming as X  # noqa: E402
from fast_report import cards  # noqa: E402

FAMS = list(cards.TAXONOMY)
FAM_OF = np.array([FAMS.index(cards.FAMILY[c]) for c in X.CLASSES])

KNN_KEY, ZS_KEY = ("dinov2-l", "plain+masked"), ("pe-core-l", "masked")  # the memo's picks (the pipeline computes exactly these)
RULES = ("sam3+bank", "sam3+zero-shot/yolo")
TARGET = .8
FAMILY_OF_SITE = X.FAMILY_OF_SITE
CLIP = {"me340": "me340-165", "samsclub-a2": "samsclub-337", "walmart": "walmart-190"}


def det_vote(m, det, min_iou=.5):
    """x13's owl_vote on another detector's boxes: per class the best score per view (box IoU >= min_iou with the mask's
    box), averaged over the card's views -> (label, score)."""
    return X.owl_vote(m, det, min_iou)[:2]


def rule_scores(s3, kx, zs, yo):
    """Each rule's (label, score) for one card, None where it does not apply. s3 = (label, share, word)."""
    lab, share, _ = s3
    if not lab:
        return {r: None for r in RULES}
    return {"sam3+bank": (lab, min(share, kx["share"])) if lab == kx["label"] else None,
            "sam3+zero-shot/yolo": (lab, share) if lab in (zs[0], yo[0]) else None}


def tier1(sc, th):
    for r in RULES:
        x = sc[r]
        if x is not None and x[1] >= th[r]:
            return r, x[0]
    return None, None


def groups(D, V, idx, d):
    """The rest of one video -> groups: a visual cluster (x13.clusters, DINOv2-L average linkage at cosine distance d) split
    by the members' SAM 3 class; medoid first."""
    out = []
    for mem in X.clusters(V, idx, d):
        by = {}
        for i in mem:
            by.setdefault(X.sam3_vote(D.meta[i])[0], []).append(i)
        out += list(by.values())
    return out


def name_clusters(D, V, idx, d):
    """One VLM call per group (its medoid; Gemini's offline name stands in for the answer), copied to the group's members
    only when the answer has their SAM 3 class (verified); otherwise they stay 'unidentified'."""
    recs = {}
    for mem in groups(D, V, idx, d):
        med = next((i for i in mem if D.label[i]), mem[0])
        for i in mem:
            if i == med:
                recs[i] = {"tier": "vlm", "rule": "vlm", "label": D.label[i], "name": D.meta[i]["name"]}
            elif D.label[med] and X.sam3_vote(D.meta[i])[0] == D.label[med]:
                recs[i] = {"tier": "cluster", "rule": "cluster+sam3", "label": D.label[med], "name": D.meta[med]["name"], "medoid": med}
            else:
                recs[i] = {"tier": "unidentified", "rule": "unverified member", "label": None, "name": None}
    return recs


class Signals:
    def __init__(self, D, det):
        self.D, self.det = D, det
        self.V, self.P = D.vec(*KNN_KEY), D.zero_shot(*ZS_KEY)

    def rows(self, site, cross):
        """[(card i, rule scores)] in first-seen order; cross: bank rows [(card j, label, name)] (never this video's)."""
        D, q = self.D, list(X.first_seen_order(self.D, site))
        out = []
        for i, kx in zip(q, X.cross_knn(D, q, cross, self.V)):
            zs = (X.CLASSES[int(self.P[i].argmax())], float(self.P[i].max()))
            out.append((i, rule_scores(X.sam3_vote(D.meta[i]), kx, zs, det_vote(D.meta[i], self.det)), kx))
        return out


def bank_rows(D, exclude, family=None):
    """Bank rows: every Gemini-labelled card of the videos not in `exclude` (optionally of one family)."""
    return [(j, D.label[j], D.meta[j]["name"]) for j in range(D.n) if D.site[j] not in exclude and D.label[j]
            and (family is None or FAMILY_OF_SITE[D.site[j]] == family)]


def fit(S, train, test, target=TARGET):
    """Thresholds on the training videos (x13.fit's procedure: a training video's bank = every other non-test video, the
    only bank two retail videos allow), then the verified-copy cluster cut on what tier 1 leaves."""
    D = S.D
    ok = lambda a, b: a == b  # noqa: E731
    rows = []
    for s in train:
        rows += S.rows(s, bank_rows(D, (s, test)))
    th = {}
    for r in RULES:
        sc = [(x[1], ok(x[0], D.label[i])) for i, scs, _ in rows if D.label[i] for x in [scs[r]] if x is not None]
        th[r] = X.fit_threshold([a for a, _ in sc], [b for _, b in sc], target) if sc else float("inf")
    rest = [i for i, scs, _ in rows if tier1(scs, th)[0] is None]
    th["cluster_cut"] = fit_cut(S, rest, train, target)
    return th


CURVES = {}


def fit_cut(S, rest, train, target, min_n=20):
    """The largest cut whose verified copies reach `target` on the training videos (>= min_n copies); the curve is kept."""
    best, curve = 0., []
    for d in X.D_GRID:
        got, calls, unident = [], 0, 0
        for s in train:
            r_ = name_clusters(S.D, S.V, [i for i in rest if S.D.site[i] == s], d)
            got += [r["label"] == S.D.label[i] for i, r in r_.items() if r["tier"] == "cluster" and S.D.label[i]]
            calls += sum(r["tier"] == "vlm" for r in r_.values())
            unident += sum(r["tier"] == "unidentified" for r in r_.values())
        curve.append((float(d), len(got), round(float(np.mean(got)), 3) if got else None, calls, unident))
        if len(got) >= min_n and np.mean(got) >= target:
            best = float(d)
    CURVES[tuple(train)] = curve
    return best


def family_votes(S, site, cross):
    """{card i: the cheap votes at family level} (cards.family_vote's inputs), with the held-out bank."""
    D, out = S.D, {}
    for i, scs, kx in S.rows(site, cross):
        fz = np.bincount(FAM_OF, S.P[i], len(FAMS))
        out[i] = {"sam3": X.sam3_vote(D.meta[i])[0], "yolo": det_vote(D.meta[i], S.det)[0], "zero_shot_family": (FAMS[int(fz.argmax())], float(fz.max())),
                  "bank": kx["label"]}
    return out


def with_types(S, site, cross, recs):
    """Every card's type: the family of its accepted name, else cards.family_vote, else 'shape' (x13 keeps no geometry)."""
    fv = family_votes(S, site, cross)
    for i, r in recs.items():
        f = X.family(r["label"]) if r.get("label") and not r["label"].startswith("~") else None
        if f and f != "hazard":
            r.update(type=f, type_source="the name")
        else:
            f, src = cards.family_vote(**fv[i])
            r.update(type=f or "shape", type_source=src or "shape")
    return recs


def simulate(S, site, cross, th, ask_unverified=False):
    """One video: tier 1 at th -> the rest clustered, one VLM call per medoid, verified copies; unverified members stay
    'unidentified' (ask_unverified: each is its own VLM call, x13's route)."""
    recs, rest = {}, []
    for i, scs, kx in S.rows(site, cross):
        rule, label = tier1(scs, th)
        if rule:
            recs[i] = {"tier": "tier1", "rule": rule, "label": label, "name": X.sam3_vote(S.D.meta[i])[2]}
        else:
            rest.append(i)
    recs.update(name_clusters(S.D, S.V, rest, th["cluster_cut"]))
    if ask_unverified:
        for i, r in recs.items():
            if r["tier"] == "unidentified":
                recs[i] = {"tier": "vlm", "rule": "vlm (unverified member)", "label": S.D.label[i], "name": S.D.meta[i]["name"]}
    return with_types(S, site, cross, recs)


def summarise(D, recs):
    """Per video: tier shares, VLM calls, cheap names (tier 1 + copies) against Gemini's class, and the audited items:
    final names (an unidentified card counts wrong), named cards only, and Gemini's own."""
    out = {}
    for site in [*X.SITES, "all"]:
        ids = [i for i in recs if site == "all" or D.site[i] == site]
        if not ids:
            continue
        cheap = [i for i in ids if recs[i]["tier"] in ("tier1", "cluster") and D.label[i]]
        aud = [i for i in ids if D.meta[i]["audit"]]
        g = [X.grade_audit(recs[i]["name"], D.meta[i]["audit"]) for i in aud]
        gn = [X.grade_audit(recs[i]["name"], D.meta[i]["audit"]) for i in aud if recs[i]["name"]]
        g0 = [X.grade_audit(D.meta[i]["name"] or X.sam3_vote(D.meta[i])[2], D.meta[i]["audit"]) for i in aud]
        lab = [i for i in ids if D.label[i]]
        typed = [i for i in ids if recs[i]["type"] != "shape"]
        fam_ok = lambda i: recs[i]["type"] == X.family(D.label[i])  # noqa: E731
        ga = [family_grade(recs[i]["type"], D.meta[i]["audit"]) for i in aud]
        g0a = [family_grade(X.family(X.label_of(D.meta[i]["name"])) if D.meta[i]["name"] else None, D.meta[i]["audit"]) for i in aud]
        by_src = {}
        for i in lab:
            by_src.setdefault(recs[i]["type_source"], []).append(fam_ok(i))
        types = {"typed_share (a family)": round(len(typed) / len(ids), 3), "shape_only_share": round(1 - len(typed) / len(ids), 3),
                 "family_vs_gemini": {"n": len(lab), "agree": round(float(np.mean([fam_ok(i) for i in lab])), 3) if lab else None},
                 "family_vs_gemini_by_source": {k: {"n": len(v), "agree": round(float(np.mean(v)), 3)} for k, v in sorted(by_src.items(), key=lambda x: -len(x[1]))},
                 "audited_family_right": X._share(ga, (True,)), "gemini_audited_family_right": X._share(g0a, (True,)), "audited_n": len(aud)}
        by_rule = {}
        for i in cheap:
            by_rule.setdefault(recs[i]["rule"], []).append(recs[i]["label"] == D.label[i])
        out[site] = {"types": types, "cards": len(ids), "vlm_calls": sum(recs[i]["tier"] == "vlm" for i in ids),
                     "vlm_share": round(sum(recs[i]["tier"] == "vlm" for i in ids) / len(ids), 3),
                     "tier_share": {t: round(sum(recs[i]["tier"] == t for i in ids) / len(ids), 3) for t in ("tier1", "cluster", "vlm", "unidentified")},
                     "cheap_vs_gemini_class": {"n": len(cheap), "agree": X._share([recs[i]["label"] == D.label[i] for i in cheap], (True,))},
                     "by_rule": {r: {"n": len(v), "agree": round(float(np.mean(v)), 3)} for r, v in by_rule.items()},
                     "audited": {"n": len(aud), "final_right": X._share(g, ("right",)), "final_right_or_close": X._share(g, ("right", "close")),
                                 "named_n": len(gn), "named_right": X._share(gn, ("right",)), "named_right_or_close": X._share(gn, ("right", "close")),
                                 "gemini_right": X._share(g0, ("right",)), "gemini_right_or_close": X._share(g0, ("right", "close"))}}
    return out


def family_grade(fam, audit):
    """The type's family against the agent's label: right when it is the label's class's family (or an 'also' class's);
    None for an unclear label; a shape type (no family) is not right."""
    if not audit or audit["canon"] == "unclear":
        return None
    if audit["canon"] == cards.NOT_OBJECT:
        return fam == cards.NOT_OBJECT
    truth = {cards.FAMILY.get(c) for c in [audit["canon"], *audit.get("also", [])] if cards.FAMILY.get(c)}
    return fam in truth


def folds(S, target=TARGET):
    """Per test video: the thresholds the pipeline uses when it analyses that video (held out)."""
    D, out = S.D, {}
    for test in X.SITES:
        fam = FAMILY_OF_SITE[test]
        same = [s for s in X.SITES if s != test and FAMILY_OF_SITE[s] == fam]
        if same:
            th = fit(S, same, test, target)
            out[test] = {"family": fam, "trained_on": same, "uncalibrated": False, **{k: _num(v) for k, v in th.items()}}
        else:  # the first video of its family: no cheap acceptance; the cut is a visual-similarity property, fitted on the others
            other = [s for s in X.SITES if s != test]
            th = fit(S, other, test, target)
            out[test] = {"family": fam, "trained_on": [], "cut_trained_on": other, "uncalibrated": True,
                         **{r: None for r in RULES}, "cluster_cut": th["cluster_cut"],
                         "note": "first video of its domain family: no calibrated second vote, so no cheap acceptance; one VLM call per "
                                 "cluster medoid at the cut fitted on the other videos"}
    return out


def _num(v):
    return None if v == float("inf") else round(float(v), 4)


def th_of(f):
    return {**{r: (float("inf") if f[r] is None else f[r]) for r in RULES}, "cluster_cut": f["cluster_cut"]}


def sha(site):
    return hashlib.sha256((X.PHASE2 / "data/clips" / CLIP[site] / "source-full.mp4").read_bytes()).hexdigest()


def main(a):
    x13 = Path(a.x13)
    D = X.Data(x13)
    det = pickle.loads((x13 / "yoloe.pkl").read_bytes())
    S, So = Signals(D, det["yolo"]), Signals(D, D.owl)
    F = folds(S)
    res = {"schema": "r4-naming-fit-v1", "target": TARGET, "knn": KNN_KEY, "zero_shot": ZS_KEY, "rules": RULES, "folds": F,
           "yoloe_timing": det["timing"], "simulated_heldout": {}}
    for name, s_, ask in (("pipeline: yoloe, VLM on group medoids only, the rest typed", S, False),
                          ("variant: contradicted members also asked on their own (naming_escalate)", S, True),
                          ("variant: owlv2 instead of yoloe", So, False)):
        recs = {}
        for test in X.SITES:
            f = folds(s_)[test] if s_ is not S else F[test]
            fam = FAMILY_OF_SITE[test]
            recs.update(simulate(s_, test, bank_rows(D, (test,), fam), th_of(f), ask))
        res["simulated_heldout"][name] = summarise(D, recs)
    for cut in (0., .1, .2):  # sensitivity only (never used to choose): the uncalibrated video at fixed cuts
        recs = simulate(S, "me340", bank_rows(D, ("me340",), "shop floor"), {**th_of(F["me340"]), "cluster_cut": cut})
        res["simulated_heldout"][f"me340 at cut {cut} (sensitivity)"] = {"me340": summarise(D, recs)["me340"]}
    fam_fit = {}
    for fam in sorted(set(FAMILY_OF_SITE.values())):
        vids = [s for s in X.SITES if FAMILY_OF_SITE[s] == fam]
        if len(vids) >= 2:  # production: every audited video of the family (a new video of the family is held out by construction)
            th = fit(S, vids, None)
            fam_fit[fam] = {"trained_on": vids, "uncalibrated": False, **{k: _num(v) for k, v in th.items()}}
    calib = {"schema": "panoptes-naming-calibration-v1", "fitted_on": "x13 crop dataset (mvp2-integrate-*-007 warm cards, Gemini names as labels)",
             "target_class_agreement": TARGET, "knn": list(KNN_KEY), "zero_shot": list(ZS_KEY), "rules": list(RULES),
             "folds": F, "families": fam_fit,
             "uncalibrated": {**{r: None for r in RULES}, "cluster_cut": F["me340"]["cluster_cut"], "uncalibrated": True},
             "site_family": FAMILY_OF_SITE, "note": "folds[site]: held out (fitted without that video); families[f]: production"}
    res["cut_curves (train videos: cut, copies, copy precision, vlm calls, unidentified)"] = {"+".join(k): v for k, v in CURVES.items()}
    Path(a.calibration).write_text(json.dumps(calib, indent=1))
    Path(a.results).parent.mkdir(parents=True, exist_ok=True)
    Path(a.results).write_text(json.dumps(res, indent=1, default=str))
    V = D.vec(*KNN_KEY)
    lab = [i for i in range(D.n) if D.label[i]]
    shas = {s: sha(s) for s in X.SITES}
    meta = [{"label": D.label[i], "name": D.meta[i]["name"], "site": D.site[i], "family": FAMILY_OF_SITE[D.site[i]], "video": shas[D.site[i]],
             "card": D.meta[i]["card"], "source": "gemini (round 2 warm cards, mvp2-integrate-*-007)"} for i in lab]
    T = D.F["text"]["pe-core-l"]
    np.savez(a.bank, emb=V[lab].astype(np.float16), meta=json.dumps(meta), text=T.astype(np.float32), text_scale=np.float32(D.F["timing"]["pe-core-l"]["scale"]),
             classes=np.array(X.CLASSES))
    for name, s in res["simulated_heldout"].items():
        print("==", name)
        for v in [v for v in [*X.SITES, "all"] if v in s]:
            x = s[v]
            t = x["types"]
            print(f"  {v:12s} TYPED {t['typed_share (a family)']} family~gemini {t['family_vs_gemini']['agree']} audited family right {t['audited_family_right']} "
                  f"(gemini {t['gemini_audited_family_right']}, n {t['audited_n']}) | sources {t['family_vs_gemini_by_source']}")
            print(f"  {v:12s} vlm {x['vlm_share']:.3f} ({x['vlm_calls']})  tiers {x['tier_share']}  cheap {x['cheap_vs_gemini_class']} {x['by_rule']}  "
                  f"aud {x['audited']['final_right']}/{x['audited']['final_right_or_close']} named {x['audited']['named_n']}: "
                  f"{x['audited']['named_right']}/{x['audited']['named_right_or_close']} (gemini {x['audited']['gemini_right']}/{x['audited']['gemini_right_or_close']})")
    print(json.dumps(F, indent=1))
    print(json.dumps(fam_fit, indent=1))


def self_check():
    s3 = ("box", .7, "cardboard box")
    kx = {"label": "box", "share": .6}
    sc = rule_scores(s3, kx, ("shelf", .9), ("box", .3))
    assert sc["sam3+bank"] == ("box", .6) and sc["sam3+zero-shot/yolo"] == ("box", .7)
    assert tier1(sc, {"sam3+bank": .65, "sam3+zero-shot/yolo": .5}) == ("sam3+zero-shot/yolo", "box")
    assert tier1(sc, {"sam3+bank": .9, "sam3+zero-shot/yolo": .9}) == (None, None)
    assert rule_scores((None, 0., None), kx, ("box", 1.), ("box", 1.)) == {r: None for r in RULES}  # no SAM 3 word: nothing accepted
    assert th_of({"sam3+bank": None, "sam3+zero-shot/yolo": .3, "cluster_cut": .2})["sam3+bank"] == float("inf")
    assert family_grade("machine", {"canon": "lathe", "also": []}) is True and family_grade("goods", {"canon": "shelf", "also": ["pallet"]}) is True
    assert family_grade("shape", {"canon": "lathe"}) is False and family_grade("machine", {"canon": "unclear"}) is None
    print("r4_naming_fit self-check ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("x13", nargs="?")
    ap.add_argument("--calibration", default=str(REPO / "fast_report/naming_calibration.json"))
    ap.add_argument("--results")
    ap.add_argument("--bank")
    ap.add_argument("--self-check", action="store_true")
    a = ap.parse_args()
    self_check() if a.self_check else main(a)
