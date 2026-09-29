"""Calibration for the click MVP (CLICK-MVP-SPEC sections 4.4, 4.7, 5.3, 8.2-8.4). D writes fast_report/calibration.json;
A reads the identity table and k, B reads the decider's Platt pairs. One JSON, three sections, each optional:

  decider:  {"source", "questions": {qid: {"a", "b", "fit": "per-question" | "pooled", "n", "positives",
                                           "raw": {...}, "calibrated_cv": {...}}}}   p' = sigmoid(a * logit(p_yes) + b)
  identity: {"routes": {route: {"bins": [[lo, hi, n, accuracy]], "n", "accuracy"}}, "ece_cv_by_video", ...}
  k:        {family: {"k", "n", "coverage_before", "coverage_after"}}   families: height, extent, position, angle

Metrics: Brier; ECE two ways, 10 bins: 'ece' = top-label confidence vs correct (X8's x8_metrics.ece, so numbers compare with
X8) and 'ece_p_yes' = binned p_yes vs the yes rate; AUROC of p_yes (Mann-Whitney, ties half). Cross-validation is grouped:
by source clip for the decider (spec 5.3), by video for identity (8.2). Nothing here is ground truth: labels are agent-made.

    python -m fast_report.calibration --self-check
"""
import json
import time
from pathlib import Path

import numpy as np

PATH = Path(__file__).with_name("calibration.json")
EPS, BINS, MIN_CLASS = 1e-4, 10, 3  # a question gets its own Platt pair only with >= 3 yes and >= 3 no; else the pooled pair


def logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1 / (1 + np.exp(-np.asarray(z, float)))


def platt_fit(p, y):
    """(a, b) minimising the log loss of sigmoid(a * logit(p) + b); a tiny ridge keeps separable sets finite."""
    from scipy.optimize import minimize
    z, y = logit(p), np.asarray(y, float)

    def nll(w):
        s = w[0] * z + w[1]
        return float(np.sum(np.logaddexp(0, s) - y * s) + 1e-3 * (w[0] - 1) ** 2 + 1e-3 * w[1] ** 2)
    w = minimize(nll, [1., 0.], method="BFGS").x
    return round(float(w[0]), 5), round(float(w[1]), 5)


def platt_apply(ab, p):
    return sigmoid(ab[0] * logit(p) + ab[1])


def auroc(score, y):
    """Mann-Whitney AUROC with ties counted half; None when one class is missing (x8_metrics.auroc)."""
    score, y = np.asarray(score, float), np.asarray(y, bool)
    pos, neg = score[y], score[~y]
    if not len(pos) or not len(neg):
        return None
    return round(float(((pos[:, None] > neg[None]).sum() + .5 * (pos[:, None] == neg[None]).sum()) / (len(pos) * len(neg))), 4)


def ece_conf(conf, correct, bins=BINS):
    """Top-label ECE (x8_metrics.ece)."""
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    idx = np.minimum((conf * bins).astype(int), bins - 1)
    return round(float(sum(abs(conf[idx == b].mean() - correct[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any())), 4)


def binary_metrics(p, y):
    p, y = np.asarray(p, float), np.asarray(y, float)
    if not len(p):
        return {"n": 0}
    said = p >= .5
    return {"n": int(len(p)), "positives": int(y.sum()), "brier": round(float(np.mean((p - y) ** 2)), 4),
            "ece": ece_conf(np.maximum(p, 1 - p), said == (y > .5)), "ece_p_yes": ece_conf(p, y),
            "auroc": auroc(p, y), "accuracy": round(float(np.mean(said == (y > .5))), 4), "said_yes_share": round(float(said.mean()), 4),
            "false_yes": int((said & (y < .5)).sum()), "false_no": int((~said & (y > .5)).sum())}


def _fit_pairs(q, p, y):
    """Per question: its own pair when it has both classes (>= MIN_CLASS each), else the pooled pair of all items."""
    pooled, pairs = platt_fit(p, y), {}
    for qid in sorted(set(q)):
        m = q == qid
        own = y[m].sum() >= MIN_CLASS and (1 - y[m]).sum() >= MIN_CLASS
        pairs[qid] = (platt_fit(p[m], y[m]), "per-question") if own else (pooled, "pooled")
    return pairs


def decider(items, probs, source, folds=5):
    """items: [{id, question_id, source, truth}]; probs: {id: p_yes} (items without a probability are left out).
    -> the 'decider' section: Platt pairs fitted on everything, and raw vs calibrated metrics where the calibrated
    probabilities are cross-fitted by source clip (sources sorted by size, dealt round-robin into `folds` folds)."""
    items = [i for i in items if i["id"] in probs]
    q = np.array([i["question_id"] for i in items])
    p = np.array([float(probs[i["id"]]) for i in items])
    y = np.array([float(i["truth"]) for i in items])
    src = [i["source"] for i in items]
    sizes = {s: src.count(s) for s in set(src)}
    fold_of = {s: k % folds for k, s in enumerate(sorted(sizes, key=lambda s: (-sizes[s], s)))}
    fold = np.array([fold_of[s] for s in src])
    cv = np.zeros(len(p))
    for f in range(folds):
        test = fold == f
        if not test.any():
            continue
        pairs = _fit_pairs(q[~test], p[~test], y[~test])
        pooled = platt_fit(p[~test], y[~test])
        for k in np.flatnonzero(test):
            cv[k] = platt_apply(pairs.get(q[k], (pooled, "pooled"))[0], p[k])
    pairs = _fit_pairs(q, p, y)
    out = {"source": source, "folds": folds, "fold_of_source": fold_of, "rule": "p' = sigmoid(a * logit(p_yes) + b)",
           "questions": {}, "all": {"raw": binary_metrics(p, y), "calibrated_cv": binary_metrics(cv, y)}}
    for qid, (ab, how) in pairs.items():
        m = q == qid
        out["questions"][qid] = {"a": ab[0], "b": ab[1], "fit": how, "n": int(m.sum()), "positives": int(y[m].sum()),
                                 "raw": binary_metrics(p[m], y[m]), "calibrated_cv": binary_metrics(cv[m], y[m])}
    return out


def apply_decider(cal, qid, p_yes):
    """Calibrated p_yes, or None when the file has no pair for this question (the answer stays uncalibrated)."""
    qs = ((cal or {}).get("decider") or {}).get("questions") or {}
    return None if qid not in qs else float(platt_apply((qs[qid]["a"], qs[qid]["b"]), p_yes))


def _bin(p, bins=BINS):
    return min(int(float(p) * bins), bins - 1)


def identity_table(rows, bins=BINS):
    """rows: [{route, p, correct}] -> {route: {bins: [[lo, hi, n, accuracy]], n, accuracy}}."""
    out = {}
    for route in sorted({r["route"] for r in rows}):
        rs = [r for r in rows if r["route"] == route]
        by = {}
        for r in rs:
            by.setdefault(_bin(r["p"], bins), []).append(r["correct"])
        out[route] = {"n": len(rs), "accuracy": round(float(np.mean([r["correct"] for r in rs])), 4),
                      "bins": [[b / bins, (b + 1) / bins, len(v), round(float(np.mean(v)), 4)] for b, v in sorted(by.items())]}
    return out


def identity_confidence(table, route, p):
    """Calibrated accuracy for (route, p): its bin's accuracy, else the route's, else None (the card says uncalibrated)."""
    t = (table or {}).get(route)
    if not t:
        return None
    return next((acc for lo, hi, n, acc in t["bins"] if lo <= float(p) < hi or (hi == 1 and p >= 1)), t["accuracy"])


def identity(rows):
    """rows: [{video, route, p, correct}] -> the 'identity' section: the table on everything, and its ECE when each video
    is scored with the table fitted on the other videos."""
    pred, correct = [], []
    for v in sorted({r["video"] for r in rows}):
        table = identity_table([r for r in rows if r["video"] != v])
        for r in rows:
            if r["video"] == v:
                c = identity_confidence(table, r["route"], r["p"])
                pred.append(r["p"] if c is None else c)
                correct.append(r["correct"])
    return {"routes": identity_table(rows), "n": len(rows), "videos": sorted({r["video"] for r in rows}),
            "ece_cv_by_video": ece_conf(pred, correct) if pred else None,
            "ece_stated": ece_conf([r["p"] for r in rows], [r["correct"] for r in rows]) if rows else None,
            "rule": "confidence = the accuracy of the (route, stated-probability bin) on the labelled matches"}


def k_family(pairs, target=.9):
    """pairs: {family: [(|delta|, u)]} with u = sqrt(u1^2 + u2^2) at k = 1 (scale parts removed) -> {family: k...}.
    k = the smallest factor >= 1 with coverage >= target: max(1, the target-quantile of |delta| / u)."""
    out = {}
    for fam, xs in pairs.items():
        d = np.array([x[0] for x in xs], float)
        u = np.array([max(x[1], 1e-9) for x in xs], float)
        if not len(d):
            out[fam] = {"k": 1., "n": 0, "coverage_before": None, "coverage_after": None}
            continue
        k = max(1., float(np.quantile(d / u, target, method="higher")))
        out[fam] = {"k": round(k, 3), "n": int(len(d)), "coverage_before": round(float(np.mean(d <= u)), 4),
                    "coverage_after": round(float(np.mean(d <= k * u + 1e-12)), 4)}
    return out


def load(path=PATH):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def write(sections, path=PATH):
    """Merge sections into the file (others kept); k may only go up (spec 4.4: D may only raise it)."""
    cal = load(path) or {"schema": "panoptes-calibration-v1"}
    if "k" in sections and cal.get("k"):
        for fam, v in sections["k"].items():
            v["k"] = max(v["k"], (cal["k"].get(fam) or {}).get("k", 1.))
    cal.update(sections, written_unix=time.time(), note="labels are agent-made; see each section's source")
    Path(path).write_text(json.dumps(cal, indent=1))
    return cal


def self_check():
    rng = np.random.default_rng(0)
    # a decider that is right but over-confident: Platt must lower its extremes and not hurt Brier (cross-fitted)
    items, probs = [], {}
    for k in range(400):
        y = int(rng.random() < .3)
        z = (1.5 if y else -1.5) + rng.normal(0, 1.2)
        items.append({"id": f"i{k}", "question_id": "q1" if k % 2 else "q4", "source": f"clip{k % 7}", "truth": y})
        probs[f"i{k}"] = float(sigmoid(8 * z))  # true log-odds ~2.1 z: 4x too sharp
    items.append({"id": "none", "question_id": "q2", "source": "clip0", "truth": 0})
    items += [{"id": f"z{k}", "question_id": "q2", "source": f"clip{k % 7}", "truth": 0} for k in range(20)]
    probs.update({f"z{k}": .02 for k in range(20)})
    d = decider(items, probs, "synthetic")
    assert d["questions"]["q2"]["fit"] == "pooled" and d["questions"]["q1"]["fit"] == "per-question", d["questions"]["q2"]
    assert 0 < d["questions"]["q1"]["a"] < .6, "an over-sharp decider gets a < 1"
    assert d["all"]["calibrated_cv"]["brier"] <= d["all"]["raw"]["brier"] + 1e-6, d["all"]
    assert d["all"]["calibrated_cv"]["ece"] < d["all"]["raw"]["ece"], d["all"]
    assert d["all"]["raw"]["n"] == 420, "items without a probability are left out"
    cal = {"decider": d}
    assert apply_decider(cal, "q9", .9) is None and 0 < apply_decider(cal, "q1", .999) < .999
    assert abs(float(platt_apply((1., 0.), .7)) - .7) < 1e-9
    # identity: a route whose 0.9 bin is right half the time reads 0.5; unknown route -> None
    rows = [{"video": v, "route": "vlm options", "p": .95, "correct": k % 2} for v in "abc" for k in range(10)]
    rows += [{"video": v, "route": "sam3 vote", "p": .35, "correct": 1} for v in "abc" for _ in range(4)]
    ident = identity(rows)
    assert identity_confidence(ident["routes"], "vlm options", .93) == .5 and identity_confidence(ident["routes"], "free text", .9) is None
    assert identity_confidence(ident["routes"], "vlm options", .2) == .5, "empty bin -> the route's accuracy"
    assert ident["ece_cv_by_video"] < ident["ece_stated"]
    # k: deltas 2x the stated u -> k = 2 brings coverage to >= 0.9
    k = k_family({"height": [(2 * u, u) for u in np.linspace(.05, .5, 20)], "angle": [(.5, 1.)] * 10, "extent": []})
    assert k["height"]["k"] == 2. and k["height"]["coverage_before"] == 0 and k["height"]["coverage_after"] == 1.
    assert k["angle"] == {"k": 1., "n": 10, "coverage_before": 1., "coverage_after": 1.} and k["extent"]["k"] == 1.
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "c.json"
        write({"k": {"height": {"k": 1.5}}}, f)
        assert write({"k": {"height": {"k": 1.2}}}, f)["k"]["height"]["k"] == 1.5, "k may only go up"
        assert write({"identity": ident}, f)["k"]["height"]["k"] == 1.5, "other sections kept"
    print("calibration self-check passed: Platt per question / pooled, grouped CV, metrics, identity bins, k, merge")


if __name__ == "__main__":
    import sys
    if "--self-check" in sys.argv:
        self_check()
