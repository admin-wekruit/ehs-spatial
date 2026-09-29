"""X8 metrics: truth (reference and agent-audited) x the deciders' saved probabilities -> accuracy, AUROC, ECE, reliability,
the 90 %-precision operating point (confidence threshold, share decided = 1 - escalation), latency, memory, cost.

  python scripts/x8_metrics.py RUN_DIR [DECISIONS_JSON]   # writes RUN_DIR/metrics.json and RUN_DIR/reliability.png
  python scripts/x8_metrics.py --self-check

'Precision' at the operating point = the share of the decider's own accepted answers (confidence >= threshold) that are
correct; the threshold is the lowest one that keeps it >= 0.9, found on the same items (in-sample, optimistic) and 2-fold
cross-fitted (fit on one half, scored on the other, both ways; the honest number).
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from x8_sets import name_match  # noqa: E402

NONE = "none of these"
TARGET = .9


def auroc(score, y):
    """Mann-Whitney AUROC with ties counted half; None when one class is missing."""
    score, y = np.asarray(score, float), np.asarray(y, bool)
    pos, neg = score[y], score[~y]
    if not len(pos) or not len(neg):
        return None
    gt = (pos[:, None] > neg[None]).sum() + .5 * (pos[:, None] == neg[None]).sum()
    return round(float(gt / (len(pos) * len(neg))), 4)


def ece(conf, correct, bins=10):
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    idx = np.minimum((conf * bins).astype(int), bins - 1)
    return round(float(sum(abs(conf[idx == b].mean() - correct[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any())), 4)


def reliability(conf, correct, bins=10):
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    idx = np.minimum((conf * bins).astype(int), bins - 1)
    return [{"bin": b, "n": int((idx == b).sum()), "mean_conf": round(float(conf[idx == b].mean()), 4), "accuracy": round(float(correct[idx == b].mean()), 4)}
            for b in range(bins) if (idx == b).any()]


def threshold_for(conf, correct, target=TARGET):
    """Lowest confidence threshold whose accepted answers are >= target correct (None if none reaches it)."""
    order = np.argsort(-np.asarray(conf, float))
    c, ok = np.asarray(conf, float)[order], np.asarray(correct, float)[order]
    run = np.cumsum(ok) / np.arange(1, len(ok) + 1)
    good = [i for i in range(len(c)) if run[i] >= target and (i == len(c) - 1 or c[i + 1] < c[i])]
    return float(c[max(good)]) if good else None


def operating_point(conf, correct, seed=0):
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    th = threshold_for(conf, correct)
    ins = {"threshold": None if th is None else round(th, 4), "decided_share": 0., "accuracy_decided": None}
    if th is not None:
        m = conf >= th
        ins.update(decided_share=round(float(m.mean()), 4), accuracy_decided=round(float(correct[m].mean()), 4))
    rng = np.random.default_rng(seed)
    half = rng.permutation(len(conf)) < len(conf) // 2
    dec, acc = 0, []
    for fit, test in ((half, ~half), (~half, half)):
        t = threshold_for(conf[fit], correct[fit])
        if t is not None:
            m = test & (conf >= t)
            dec += int(m.sum())
            acc += list(correct[m])
    cross = {"decided_share": round(dec / len(conf), 4), "accuracy_decided": round(float(np.mean(acc)), 4) if acc else None}
    return {"in_sample": ins, "cross_fitted_2fold": cross, "escalation_rate_cross_fitted": round(1 - dec / len(conf), 4)}


def score(probs, truth_sets, positive=None):
    """probs (n, k) per item; truth_sets: the correct option indices per item. -> metrics dict."""
    pred = [int(np.argmax(p)) for p in probs]
    conf = [float(np.max(p)) for p in probs]
    correct = [int(p in t) for p, t in zip(pred, truth_sets)]
    out = {"n": len(probs), "accuracy": round(float(np.mean(correct)), 4), "ece_10": ece(conf, correct),
           "auroc_confidence_vs_correct": auroc(conf, correct), "mean_confidence": round(float(np.mean(conf)), 4), **operating_point(conf, correct),
           "reliability": reliability(conf, correct)}
    if positive is not None:  # binary: the option index meaning 'yes'
        y = [positive in t for t in truth_sets]
        out["auroc_p_yes"] = auroc([p[positive] for p in probs], y)
        out["positives"] = int(sum(y))
        tp = [c for c, yy in zip(correct, y) if yy]
        tn = [c for c, yy in zip(correct, y) if not yy]
        out["balanced_accuracy"] = round(float((np.mean(tp) + np.mean(tn)) / 2), 4) if tp and tn else None
        out["said_yes_share"] = round(float(np.mean([p == positive for p in pred])), 4)
    return out, (conf, correct)


def platt_cv(s, y, folds=5, seed=0):
    """Cross-fitted logistic calibration of a raw score (SigLIP cosine) -> probabilities; labelled as fitted on our labels."""
    from scipy.optimize import minimize
    s, y = np.asarray(s, float), np.asarray(y, float)
    fold = np.random.default_rng(seed).permutation(len(s)) % folds
    p = np.zeros(len(s))
    for f in range(folds):
        tr = fold != f
        nll = lambda w: -np.sum(y[tr] * -np.logaddexp(0, -(w[0] * s[tr] + w[1])) + (1 - y[tr]) * -np.logaddexp(0, w[0] * s[tr] + w[1]))  # noqa: E731
        w = minimize(nll, [10., -9.], method="Nelder-Mead").x
        p[~tr] = 1 / (1 + np.exp(-(w[0] * s[~tr] + w[1])))
    return p


def metrics(run_dir, decisions=None):
    rd = Path(run_dir)
    D = json.loads(Path(decisions or rd / "decisions.json").read_text())
    res = D["res"]
    pass2 = rd / "decisions-d-pass2.json"  # set d grown after the first run (x8_sets.D_ITEMS second pass): its d answers replace run 1's
    if pass2.exists():
        P2 = json.loads(pass2.read_text())
        for src in ("jev", "jev_batched", "qwen"):
            res[src] = {k: v for k, v in res[src].items() if not k.startswith("d|")} | {k: v for k, v in P2["res"][src].items() if k.startswith("d|")}
        res["latency"] = res["latency"] | {f"pass2_{k}": v for k, v in P2["res"]["latency"].items() if "d" in k.split("_")[-1] or k.startswith("qwen_unbatched")}
        res["card_verification"] = P2["res"].get("card_verification")
    out, curves, B_aud = {"sets": {}}, {}, []

    preds = {}
    res["jevb"] = {k: {"probs": v} for k, v in res.get("jev_batched", {}).items()}

    def add(name, decider, probs, truth, positive=None):
        m, curve = score(probs, truth, positive)
        out["sets"].setdefault(name, {})[decider] = m
        if "batched" not in decider:
            curves.setdefault(name, {})[decider] = curve
        preds.setdefault(name, {})[decider] = ([int(np.argmax(q)) for q in probs], curve[0], truth)

    # ---- a: same object?
    A = json.loads((rd / "sets/a.json").read_text())["items"]
    audit = json.loads((rd / "sets/a-audit.json").read_text())
    for variant, keep in (("reference", lambda x: True), ("audited", lambda x: x["id"] not in audit["unsure"])):
        xs = [x for x in A if keep(x)]
        truth = [{0} if (x["truth"] and not (variant == "audited" and x["id"] in audit["flip_to_0"])) else {1} for x in xs]
        for dec, key, src in (("jev-two-images", "multi", "jev"), ("jev-side-by-side", "pair", "jev"),
                              ("jev-two-images-batched", "multi", "jevb"), ("jev-side-by-side-batched", "pair", "jevb")):
            if all(f"a|{key}|{x['id']}" in res[src] for x in xs):
                add(f"a:{variant}", dec, [res[src][f"a|{key}|{x['id']}"]["probs"] for x in xs], truth, 0)
        if all(f"a|multi|{x['id']}" in res["qwen"] for x in xs):
            add(f"a:{variant}", "qwen3-vl", [res["qwen"][f"a|multi|{x['id']}"]["probs"] for x in xs], truth, 0)
        if all(x["id"] in res["siglip"] for x in xs):
            cos = np.array([res["siglip"][x["id"]]["cos"] for x in xs])
            y = np.array([0 in t for t in truth], float)
            p = platt_cv(cos, y)
            add(f"a:{variant}", "siglip-cos (platt, cross-fitted on our labels)", [[q, 1 - q] for q in p], truth, 0)
            core_th = .9858  # the fast core's per-video negative q99 on ME340 (fb-a-core-011): 'same' above it
            out["sets"][f"a:{variant}"]["siglip-cos raw"] = {"auroc_cos": auroc(cos, y), "accuracy_at_core_threshold_0.9858": round(float(np.mean((cos >= core_th) == (y > 0))), 4),
                                                             "same_share_at_core_threshold": round(float(np.mean(cos >= core_th)), 4)}
        out["sets"][f"a:{variant}"]["_by_kind"] = {k: sum(1 for x in xs if x["kind"] == k) for k in sorted({x["kind"] for x in xs})}
        for dec, src, key in (("jev-two-images", "jev", "multi"), ("qwen3-vl", "qwen", "multi")):
            ok = [int(np.argmax(res[src][f"a|{key}|{x['id']}"]["probs"])) in t for x, t in zip(xs, truth)]
            out["sets"][f"a:{variant}"].setdefault("_accuracy_by_kind", {})[dec] = {
                k: round(float(np.mean([o for o, x in zip(ok, xs) if x["kind"] == k])), 4) for k in sorted({x["kind"] for x in xs})}

    # ---- b: which label? truth = options that name-match the reference name, else 'none of these'
    B = json.loads((rd / "sets/b.json").read_text())["items"]
    baud = json.loads((rd / "sets/b-audit.json").read_text())

    def truth_b(options, name):
        t = {i for i, o in enumerate(options) if o != NONE and name_match(o, name)}
        return t or {options.index(NONE)} if NONE in options else t

    for variant, xs, name_of in (("reference-all-views", B, lambda x: x["delivered_category"]),
                                 ("audited-best-view", [x for x in B if x["view"] == 0 and x["id"] not in baud["drop"]],
                                  lambda x: baud["rename"].get(x["id"], x["delivered_category"]))):
        key = f"b:{variant}"
        for dec, src, var in (("jev-shortlist", "jev", "short"), ("jev-full-vocabulary", "jev", "full"), ("qwen3-vl-shortlist", "qwen", "short"),
                              ("jev-shortlist-batched", "jevb", "short"), ("jev-full-vocabulary-batched", "jevb", "full")):
            if all(f"b|{var}|{x['id']}" in res[src] for x in xs):
                opts = [res["jev"][f"b|{var}|{x['id']}"]["options"] for x in xs]
                add(key, dec, [res[src][f"b|{var}|{x['id']}"]["probs"] for x in xs], [truth_b(o, name_of(x)) for o, x in zip(opts, xs)])
        if all(x["id"] in res["siglip"] for x in xs):
            probs, truth, dec_ok = [], [], []
            for x in xs:
                full = res["siglip"][x["id"]]["full"]
                vocab = list(full)
                probs.append([full[w] for w in vocab])
                truth.append({i for i, w in enumerate(vocab) if name_match(w, name_of(x))} or {-1})
                srt = sorted(full.values(), reverse=True)
                dec_ok.append(srt[0] >= .5 and srt[0] - srt[1] >= .25)
            add(key, "siglip-zero-shot-full-vocabulary", probs, truth)
            m = out["sets"][key]["siglip-zero-shot-full-vocabulary"]
            corr = curves[key]["siglip-zero-shot-full-vocabulary"][1]
            m["core_rule_p0.5_margin0.25"] = {"decided_share": round(float(np.mean(dec_ok)), 4),
                                              "accuracy_decided": round(float(np.mean([c for c, d in zip(corr, dec_ok) if d])), 4) if any(dec_ok) else None}
        if variant.startswith("audited"):
            B_aud = xs
        sl = [res["shortlist"][x["id"]] for x in xs]
        out["sets"][key]["_shortlist_recall"] = round(float(np.mean([any(name_match(w, name_of(x)) for w in s) for s, x in zip(sl, xs)])), 4)
        out["sets"][key]["_vocabulary_recall"] = round(float(np.mean([any(name_match(w, name_of(x)) for w in x["vocabulary"]) for x in xs])), 4)
        out["sets"][key]["_core_cascade_label_accuracy"] = round(float(np.mean([name_match(x["cascade_label"], name_of(x)) for x in xs])), 4)
        out["sets"][key]["_sam3_word_accuracy"] = round(float(np.mean([name_match(x["sam3_word"], name_of(x)) for x in xs])), 4)
        out["sets"][key]["_n"] = len(xs)

    # ---- c: object / part / background / group
    C = json.loads((rd / "sets/c.json").read_text())["items"]
    lab = json.loads((rd / "sets/c-labels.json").read_text())["labels"]
    truth = [{"ABCD".index(lab[x["id"]])} for x in C]
    for dec, src in (("jev", "jev"), ("qwen3-vl", "qwen"), ("jev-batched", "jevb")):
        if all(f"c|abcd|{x['id']}" in res[src] for x in C):
            add("c", dec, [res[src][f"c|abcd|{x['id']}"]["probs"] for x in C], truth)
    yA = [t == {0} for t in truth]  # the discovery gate's real question: one whole object (A) or not
    for dec, src in (("jev", "jev"), ("qwen3-vl", "qwen")):
        pA = [res[src][f"c|abcd|{x['id']}"]["probs"][0] for x in C]
        m, _ = score([[a, 1 - a] for a in pA], [{0} if y else {1} for y in yA], 0)
        out["sets"]["c"].setdefault("_whole_object_vs_rest", {})[dec] = {k: m[k] for k in ("accuracy", "balanced_accuracy", "auroc_p_yes", "ece_10", "in_sample", "cross_fitted_2fold")}
    x2 = [x["x2_qwen_judge"] for x in C]
    out["sets"]["c"]["_x2_generated_qwen_judge_accuracy"] = round(float(np.mean([j is not None and "ABCD".find(j) in t for j, t in zip(x2, truth)])), 4)
    out["sets"]["c"]["_labels"] = {k: sum(1 for t in truth if t == {i}) for i, k in enumerate("ABCD")}

    # ---- d: EHS yes/no
    Dd = json.loads((rd / "sets/d.json").read_text())["items"]
    gem = json.loads((rd / "gemini-d.json").read_text())["answers"] if (rd / "gemini-d.json").exists() else {}
    if gem:  # a stated probability, not a token probability
        res["gemini"] = {f"d|yn|{i}": {"probs": [min(max(float(a["p_yes"]), 0.), 1.), 1 - min(max(float(a["p_yes"]), 0.), 1.)]} for i, a in gem.items()}
    for variant, xs in (("all", Dd), ("clear-labels", [x for x in Dd if x["clear"]])):
        truth = [{0} if x["truth"] else {1} for x in xs]
        for dec, src in (("jev", "jev"), ("qwen3-vl", "qwen"), ("gemini-stated-p", "gemini"), ("jev-batched", "jevb")):
            if src not in res:
                continue
            if all(f"d|yn|{x['id']}" in res[src] for x in xs):
                add(f"d:{variant}", dec, [res[src][f"d|yn|{x['id']}"]["probs"] for x in xs], truth, 0)
        for q in sorted({x["question_id"] for x in xs}):
            qx = [x for x in xs if x["question_id"] == q]
            row = {"n": len(qx), "yes": sum(x["truth"] for x in qx)}
            for dec, src in (("jev", "jev"), ("qwen3-vl", "qwen"), ("gemini-stated-p", "gemini")):
                if src in res and all(f"d|yn|{x['id']}" in res[src] for x in qx):
                    p = [res[src][f"d|yn|{x['id']}"]["probs"][0] for x in qx]
                    row[dec] = {"accuracy": round(float(np.mean([(pp >= .5) == bool(x["truth"]) for pp, x in zip(p, qx)])), 4),
                                "auroc_p_yes": auroc(p, [x["truth"] for x in qx]), "mean_p_yes_on_no": round(float(np.mean([pp for pp, x in zip(p, qx) if not x["truth"]] or [np.nan])), 4)}
            out["sets"][f"d:{variant}"].setdefault("_per_question", {})[q] = row
        y = np.array([x["truth"] for x in xs], bool)
        for dec, src in (("jev", "jev"), ("qwen3-vl", "qwen"), ("gemini-stated-p", "gemini")):
            if src in res and all(f"d|yn|{x['id']}" in res[src] for x in xs):
                p = np.array([res[src][f"d|yn|{x['id']}"]["probs"][0] for x in xs])
                ths = np.sort(np.unique(p))[::-1]
                rec = [(t, (p[y] >= t).mean(), (p[~y] >= t).mean()) for t in ths]
                r80 = next(((t, r, f) for t, r, f in rec if r >= .8), None)
                f10 = max(((t, r, f) for t, r, f in rec if f <= .1), key=lambda z: z[1], default=None)
                out["sets"][f"d:{variant}"].setdefault("_screening", {})[dec] = {
                    "p_yes_threshold_for_recall_0.8": None if r80 is None else {"threshold": round(float(r80[0]), 5), "false_alarm_rate": round(float(r80[2]), 4)},
                    "recall_at_false_alarm_0.1": None if f10 is None else {"threshold": round(float(f10[0]), 5), "recall": round(float(f10[1]), 4)}}

    # ---- e: text routing (rule id, severity)
    if res.get("laya"):
        E = json.loads((rd / "sets/e.json").read_text())
        rules, sev = res["e_options"]["rule"], res["e_options"]["severity"]
        items = E["items"]
        tr = [{rules.index(x["rule"])} for x in items]
        sv = [x for x in items if x["severity"]]
        ts = [{sev.index(x["severity"])} for x in sv]
        for dec, src in (("laya", "laya"), ("jev-text", "jev_text"), ("qwen3-vl-text", "qwen_text")):
            add("e:rule", dec, [res[src][x["id"]]["rule"] for x in items], tr)
            add("e:severity", dec, [res[src][x["id"]]["severity"] for x in sv], ts)

    # two deciders agree -> accept, else escalate (the ladder's cheapest check); SigLIP is excluded (never says 'none'/'no')
    for name, ds in preds.items():
        jev = next((d for d in ds if d.startswith("jev") and "batched" not in d and "side-by-side" not in d and "full" not in d), None)
        for other in [d for d in ds if d.startswith("qwen") or d.startswith("gemini")]:
            if jev:
                (pj, cj, t), (po, co, _) = ds[jev], ds[other]
                agree = [a == b for a, b in zip(pj, po)]
                ok = [a in tt for a, tt in zip(pj, t)]
                out["sets"][name][f"_agree_{jev}+{other}"] = {"agree_share": round(float(np.mean(agree)), 4),
                                                            "accuracy_when_agree": round(float(np.mean([o for o, g in zip(ok, agree) if g])), 4) if any(agree) else None}
    if "b:audited-best-view" in preds:  # the 'none of these' rows: vocabulary without the object's name
        for d, (pr, _, t) in preds["b:audited-best-view"].items():
            if "siglip" in d:
                continue
            kind = "full" if "full" in d else "short"
            none_rows = [i for i, x in enumerate(B_aud) if t[i] == {res["jev"][f"b|{kind}|{x['id']}"]["options"].index(NONE)}]
            word_rows = [i for i in range(len(t)) if i not in none_rows]
            out["sets"]["b:audited-best-view"].setdefault("_by_truth_kind", {})[d] = {
                "correct_word_exists": f"{sum(pr[i] in t[i] for i in word_rows)}/{len(word_rows)}",
                "only_none_correct": f"{sum(pr[i] in t[i] for i in none_rows)}/{len(none_rows)}"}
    out["card_verification"] = res.get("card_verification")
    out["latency"] = res["latency"]
    out["video_16_frames"] = res.get("video")
    run = D["run"]
    out["memory"] = {"gpu_peak": run["gpu_peak"], "flags": run["flags"], "torch_peaks_gb": run.get("torch_peaks_gb"),
                     "stages": [{k: r[k] for k in ("stage", "s", "peak_gb", "over_90", "n") if k in r} for r in run["stages"]]}
    out["boot"] = run.get("boot")
    out["analysis_elapsed_s"] = run.get("elapsed_s")
    # batched vs unbatched agreement for Jev-Omni
    diffs = [max(abs(a - b) for a, b in zip(res["jev"][k]["probs"], res["jev_batched"][k])) for k in res["jev_batched"] if k in res["jev"]]
    if diffs:
        out["jev_batched_vs_unbatched_max_abs_prob_diff"] = {"median": round(float(np.median(diffs)), 5), "max": round(float(np.max(diffs)), 5), "n": len(diffs)}
    (rd / "metrics.json").write_text(json.dumps(out, indent=1))
    plot(curves, rd / "reliability.png")
    return out


COLORS = {"jev": "#2a78d6", "qwen": "#eb6834", "siglip": "#1baf7a", "laya": "#eda100", "gemini": "#e87ba4"}  # dataviz categorical slots 1-5


def plot(curves, path, min_bin=5):
    """One reliability diagram per set: bin accuracy vs mean confidence (bins with < min_bin items left out), the diagonal =
    perfect calibration; marker area follows the bin's item count."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = [n for n in curves if not n.startswith("a:reference") and not n.startswith("b:reference")]
    cols = 4
    rows = (len(names) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 3.5 * rows), squeeze=False)
    for ax in axes.flat[len(names):]:
        ax.axis("off")
    for ax, n in zip(axes.flat, names):
        ax.plot([0, 1], [0, 1], color="#b0afa6", lw=1, ls="--")
        for dec, (conf, corr) in curves[n].items():
            col = next((c for k, c in COLORS.items() if dec.startswith(k)), "#666")
            r = [b for b in reliability(conf, corr) if b["n"] >= min_bin]
            ls = ":" if "side-by-side" in dec or "full" in dec else "-"
            ax.plot([b["mean_conf"] for b in r], [b["accuracy"] for b in r], color=col, lw=2, ls=ls, label=dec[:26])
            ax.scatter([b["mean_conf"] for b in r], [b["accuracy"] for b in r], s=[8 + b["n"] / 2 for b in r], color=col, edgecolors="white", linewidths=1, zorder=3)
        ax.set_title(n, fontsize=9, color="#222")
        ax.set_xlim(0, 1.02), ax.set_ylim(0, 1.02)
        ax.set_xlabel("confidence (bin mean)", fontsize=8, color="#555")
        ax.set_ylabel("accuracy in bin", fontsize=8, color="#555")
        ax.tick_params(labelsize=7, colors="#555")
        ax.grid(color="#eeeeea", lw=.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.legend(fontsize=6.5, frameon=False, loc="upper left")
    fig.suptitle(f"X8 reliability (10 bins, bins with >= {min_bin} items; marker area = items)", fontsize=10, color="#222")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def self_check():
    assert auroc([.9, .8, .1], [1, 1, 0]) == 1. and auroc([.1, .9], [1, 0]) == 0. and auroc([.5, .5], [1, 0]) == .5
    assert ece([.9] * 10, [1] * 9 + [0]) == 0. and abs(ece([.9] * 10, [0] * 10) - .9) < 1e-9
    assert threshold_for([.99, .95, .9, .6, .5], [1, 1, 1, 0, 0]) == .9
    assert threshold_for([.99, .95], [0, 0]) is None
    op = operating_point(np.linspace(.5, 1, 40), np.r_[np.zeros(10), np.ones(30)])
    assert op["in_sample"]["decided_share"] == .825 and op["in_sample"]["accuracy_decided"] == .9091  # 30 right + 3 wrong on top
    m, _ = score([[.8, .2], [.3, .7], [.6, .4]], [{0}, {1}, {1}], positive=0)
    assert m["accuracy"] == round(2 / 3, 4) and m["positives"] == 1 and m["balanced_accuracy"] == .75
    p = platt_cv(np.r_[np.full(20, .99), np.full(20, .8)], np.r_[np.ones(20), np.zeros(20)])
    assert (p[:20] > .5).all() and (p[20:] < .5).all()
    print("x8 metrics self-check ok: AUROC, ECE, 90 % threshold, operating point, binary score, Platt")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--self-check"]:
        self_check()
    else:
        print(json.dumps({k: v for k, v in metrics(*sys.argv[1:3]).items() if k != "memory"}, indent=1)[:20000])
