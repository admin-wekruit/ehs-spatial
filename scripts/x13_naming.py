"""X13: naming without a run-time VLM (no fine-tuning). Held out by video: the label bank and every threshold come from the
other videos; the third is the test; rotate.

  build OUT          crop dataset from the mvp2 warm runs (runs/mvp2-integrate-*-007, warm call): per object card up to K
                     views (plain tight crop + its mask, the best view's outlined crop for the decider) and the keyframes
                     for the detector; labels = the card's Gemini open name -> cards.canonical / cards.FAMILY (offline label
                     source); the 119 agent-labelled held-out items (runs/mvp2-results/identity-heldout) matched to their card
  score OUT          features.pkl (+ jev.pkl) -> results: (a) bank kNN, SAM 3 word vote, zero-shot; (b) OWLv2; (c) Jev-Omni;
                     (d) the cascade; (e) bank growth over video orders
  --self-check

Labels: Gemini's names are the pipeline's offline labels (0.81 right on the audited items), not truth; the audited items
carry the agent's labels (mvp2/identity, blind). Nothing here calls Gemini.
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "scripts")]
from fast_report import cards  # noqa: E402

PHASE2 = Path("/Users/adam/Desktop/panoptes-public/research-notes/phase2")
WARM = {"me340": ("mvp2-integrate-me340-007", "mvp-me340-e84efffd-1790690775"),
        "samsclub-a2": ("mvp2-integrate-samsclub-a2-007", "mvp-samsclub-a2-d5e0c855-1790691934"),
        "walmart": ("mvp2-integrate-walmart-007", "mvp-walmart-c0761a2a-1790692552")}
AUDIT = PHASE2 / "runs/mvp2-results/identity-heldout"
K_VIEWS, SIDE, PAD = 5, 384, .1
MODEL_IDS = {"siglip2-base": "google/siglip2-base-patch16-224", "siglip2-so400m": "google/siglip2-so400m-patch16-384",
             "pe-core-l": "timm/PE-Core-L-14-336 (open_clip 3.3.0)", "dinov2-l": "facebook/dinov2-large",
             "owlv2-base": "google/owlv2-base-patch16-ensemble", "jev-omni": "akhilaaa3/Jev-Omni"}


def label_of(name):
    """A free name -> the bank's label key: the canonical class, else '~' + the head noun (a class outside the taxonomy)."""
    if not name:
        return None
    c = cards.canonical(name)
    if c:
        return c
    h = cards.norm(name.split(" of ")[0]).split()
    return "~" + h[-1] if h else None


def family(label):
    return cards.FAMILY.get(label) if label and not label.startswith("~") else label


def polys_xy(polys):
    return np.concatenate([np.asarray(p, float).reshape(-1, 2) for p in polys]) if polys else None


def pick_views(card, per_frame, W, H, k=K_VIEWS):
    """The card's best views first, then the largest outlines not cut by the frame edge, >= 5 keyframes apart."""
    area = {}
    for q, polys in per_frame.items():
        xy = polys_xy(polys)
        if xy is None or not len(xy):
            continue
        (x0, y0), (x1, y1) = xy.min(0), xy.max(0)
        cut = x0 <= 2 or y0 <= 2 or x1 >= W - 3 or y1 >= H - 3
        area[q] = (x1 - x0) * (y1 - y0) * (.5 if cut else 1.)
    out = [q for q in card["views"].get("best", []) if q in area][:3]
    for q in sorted(area, key=lambda q: -area[q]):
        if len(out) >= k:
            break
        if all(abs(q - p) >= 5 for p in out):
            out.append(q)
    return out


def tight_crop(frame, polys, side=SIDE, pad=PAD):
    """Box + pad, squared, grey outside the frame, side x side -> (BGR crop, mask) (cascade.Embedder.crops' geometry)."""
    import cv2
    xy = polys_xy(polys)
    (x0, y0), (x1, y1) = xy.min(0), xy.max(0)
    half = max(x1 - x0, y1 - y0, 8) * (1 + 2 * pad) / 2
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    s = side / (2 * half)
    M = np.array([[s, 0, side / 2 - cx * s], [0, s, side / 2 - cy * s]], np.float32)
    img = cv2.warpAffine(frame, M, (side, side), flags=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT,
                         borderValue=(128, 128, 128))
    mask = np.zeros((side, side), np.uint8)
    cv2.fillPoly(mask, [np.round(np.asarray(p, float).reshape(-1, 2) * s + [side / 2 - cx * s, side / 2 - cy * s]).astype(np.int32) for p in polys], 255)
    return img, mask


def _centres(cards_list):
    out = {}
    for c in cards_list:
        if c["kind"] == "object" and "box_min_m" in c["physical"]:
            out[c["id"]] = np.mean([c["physical"]["box_min_m"], c["physical"]["box_max_m"]], 0)
    return out


def audited(site, warm_cards):
    """{warm card id: agent label} for the held-out items of `site`: the item's card id with its box centre within 0.3 m of
    the item's own run, else the nearest centre within 0.3 m (identity_study.score_final's rule)."""
    import fast_report_eval as ev
    items = json.loads((AUDIT / "items.json").read_text())
    lab = json.loads((AUDIT / "labels-final.json").read_text())
    d, rep = items["runs"][site]
    src = _centres(ev.load_layers(PHASE2 / "runs" / d, rep)["object_cards"]["cards"])
    cur = _centres(warm_cards)
    out = {}
    for r in items["items"]:
        if r["site"] != site or lab[r["id"]]["canon"] == "unclear" or r["card"] not in src:
            continue
        ctr = src[r["card"]]
        got = r["card"] if r["card"] in cur and np.linalg.norm(cur[r["card"]] - ctr) < .3 else None
        if got is None:
            best = min(((float(np.linalg.norm(c - ctr)), i) for i, c in cur.items()), default=None)
            got = best[1] if best and best[0] < .3 else None
        if got is not None:
            out.setdefault(got, {**lab[r["id"]], "item": r["id"]})
    return out


def build(out):
    import cv2
    import fast_report_eval as ev
    from fast_report import judge
    from mvp_sheets import frames_bgr
    out.mkdir(parents=True, exist_ok=True)
    meta, crops, frames_jpg = [], {}, {}
    for site, (d, rep) in WARM.items():
        run = PHASE2 / "runs" / d
        L = ev.load_layers(run, rep)
        W, H = L["outlines"]["width"], L["outlines"]["height"]
        outl = {}
        for f in L["outlines"]["frames"]:
            for o in f["objects"]:
                if o["polygons"]:
                    outl.setdefault(o["entityId"], {})[f["sourceFrame"]] = o["polygons"]
        votes = {o["id"]: o.get("votes") or {} for o in L["objects"]["objects"]}
        cs = [c for c in L["object_cards"]["cards"] if c["kind"] == "object"]
        aud = audited(site, cs)
        rows = []
        for c in cs:
            members = [c["id"], *[m for m in (c["physical"].get("merged_from") or []) if isinstance(m, str)]]
            per_frame = {}
            for m in members:
                for q, p in outl.get(m, {}).items():
                    per_frame.setdefault(q, []).extend(p)
            views = pick_views(c, per_frame, W, H)
            if not views:
                continue
            idn = c["identity"]
            gem = idn.get("decided_by") == "gemini open name"
            v = {}
            for m in members:
                for w, s in votes.get(m, {}).items():
                    v[w] = v.get(w, 0.) + float(s)
            rows.append((c, views, per_frame))
            meta.append({"key": f"{site}|{c['id']}", "site": site, "card": c["id"], "first_seen_s": c["time"].get("first_seen_s"),
                         "name": idn.get("name") if gem else None, "label": label_of(idn.get("name")) if gem else None,
                         "namer_status": (idn.get("namer") or {}).get("status"), "sam3_votes": v, "sam3_word": idn.get("proposed") if not gem else None,
                         "candidates": idn.get("candidates"), "views": [], "audit": aud.get(c["id"])})
        imgs = frames_bgr(run / "mirror/blobs/sha256" / L["video"]["sha256"], {q for _, vs, _ in rows for q in vs})
        for q, im in imgs.items():
            frames_jpg[f"{site}|{q}"] = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes()
        for m, (c, views, per_frame) in zip(meta[len(meta) - len(rows):], rows):
            for i, q in enumerate(views):
                img, mask = tight_crop(imgs[q], per_frame[q])
                crops[f"{m['key']}|{i}|plain"] = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()
                crops[f"{m['key']}|{i}|mask"] = cv2.imencode(".png", mask)[1].tobytes()
                xy = polys_xy(per_frame[q])
                m["views"].append({"frame": int(q), "box": [*map(float, xy.min(0)), *map(float, xy.max(0))]})
            crops[f"{m['key']}|som"] = judge.som(imgs[views[0]], {1: per_frame[views[0]]}, subject=1, side=448)
        print(site, len(rows), "cards", len(imgs), "frames", sum(1 for m in meta if m["site"] == site and m["audit"]), "audited", flush=True)
    with open(out / "dataset.pkl", "wb") as f:
        pickle.dump({"meta": meta, "crops": crops, "frames": frames_jpg}, f, protocol=4)
    (out / "dataset-meta.json").write_text(json.dumps(meta, indent=0))
    print(json.dumps({"cards": len(meta), "labelled": sum(m["label"] is not None for m in meta), "crops": len(crops), "frames": len(frames_jpg),
                      "mb": round((out / "dataset.pkl").stat().st_size / 1e6, 1)}))


# ---------------------------------------------------------------- scoring (CPU; labels stay here)

CLASSES = [c for fam in cards.TAXONOMY.values() for c in fam]  # the containers' text-embedding order
SITES = list(WARM)
ENCODERS = ("siglip2-base", "siglip2-so400m", "pe-core-l", "dinov2-l")
K, TAU = 10, .05  # kNN: 10 neighbours, weights exp((cos - 1) / TAU); set before any score was seen


class Data:
    """meta + per-card features. vec(enc, var): card vectors (views pooled: mean of unit view vectors, renormalised)."""

    def __init__(self, out):
        self.meta = json.loads((out / "dataset-meta.json").read_text())
        self.F = pickle.loads((out / "features.pkl").read_bytes())
        self.owl = pickle.loads((out / "owl.pkl").read_bytes())["owl"]
        self.site = np.array([m["site"] for m in self.meta])
        self.label = [m["label"] for m in self.meta]
        self.n = len(self.meta)
        self._v = {}

    def vec(self, enc, var):
        if (enc, var) not in self._v:
            if var == "plain+masked":
                v = self.vec(enc, "plain") + self.vec(enc, "masked")
            else:
                E = self.F["emb"][(enc, var)].astype(np.float32)
                if var == "som":
                    v = E
                else:
                    v = np.zeros((self.n, E.shape[1]), np.float32)
                    np.add.at(v, self.F["idx"][:, 0], E)
            self._v[(enc, var)] = v / np.linalg.norm(v, axis=1, keepdims=True)
        return self._v[(enc, var)]

    def zero_shot(self, enc, var):
        """(n, 102) softmax over the taxonomy classes (logit scale x cosine)."""
        T = self.F["text"][enc]
        z = self.F["timing"][enc]["scale"] * self.vec(enc, var) @ T.T
        z = np.exp(z - z.max(1, keepdims=True))
        return z / z.sum(1, keepdims=True)


def sam3_vote(m):
    """The SAM 3 word votes over the card's masks -> (label, share of the vote, the top word)."""
    v, best = {}, {}
    for w, s in (m.get("sam3_votes") or {}).items():
        lab = label_of(w)
        if lab:
            v[lab] = v.get(lab, 0.) + s
            if s > best.get(lab, ("", -1))[1]:
                best[lab] = (w, s)
    if not v:
        return None, 0., None
    top = max(v, key=v.get)
    return top, v[top] / sum(v.values()), best[top][0]


def iou(a, b):
    ix = max(0., min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0., min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def owl_vote(m, owl, min_iou=.5):
    """OWLv2 detections on the card's views whose box overlaps the mask's box (IoU >= min_iou): per class the best score per
    view, averaged over views -> (label, score, {label: score})."""
    acc = {}
    for v in m["views"]:
        best = {}
        for *b, c, sc in owl.get(f"{m['site']}|{v['frame']}", []):
            if iou(b, v["box"]) >= min_iou:
                best[CLASSES[c]] = max(best.get(CLASSES[c], 0.), sc)
        for c, sc in best.items():
            acc[c] = acc.get(c, 0.) + sc / len(m["views"])
    if not acc:
        return None, 0., {}
    top = max(acc, key=acc.get)
    return top, acc[top], acc


def knn(Q, B, blab, k=K, tau=TAU):
    """Similarity-weighted k-NN vote. Q (q,D), B (b,D) unit; blab: bank labels -> [{label, share, cos1, ranked, nn}]."""
    if not len(B):
        return [{"label": None, "share": 0., "cos1": -1., "ranked": [], "nn": None} for _ in range(len(Q))]
    S = Q @ B.T
    kk = min(k, len(B))
    top = np.argpartition(-S, kk - 1, 1)[:, :kk]
    out = []
    for i in range(len(Q)):
        js = top[i][np.argsort(-S[i, top[i]])]
        w = {}
        for j in js:
            w[blab[j]] = w.get(blab[j], 0.) + float(np.exp((S[i, j] - 1) / tau))
        ranked = sorted(w, key=lambda x: -w[x])
        out.append({"label": ranked[0], "share": w[ranked[0]] / sum(w.values()), "cos1": float(S[i, js[0]]), "ranked": ranked,
                    "nn": {lab: int(next(j for j in js if blab[j] == lab)) for lab in ranked}})
    return out


def fit_threshold(scores, correct, target, min_n=20):
    """The lowest score t whose accepted set {s >= t} reaches precision >= target with >= min_n items (the largest such
    prefix of the ranking); inf when none does. Fitted on OTHER videos only."""
    s, c = np.asarray(scores, float), np.asarray(correct, float)
    order = np.argsort(-s, kind="stable")
    prec = np.cumsum(c[order]) / np.arange(1, len(s) + 1)
    ok = [i for i in range(len(s)) if i + 1 >= min_n and prec[i] >= target and (i + 1 == len(s) or s[order[i + 1]] < s[order[i]])]
    return float(s[order[max(ok)]]) if ok else float("inf")


def grade_audit(name, audit):
    import identity_study as ids
    return ids.grade(name, audit) if audit and name else ("wrong" if audit else None)


def table(pred, D, mask=None, names=None):
    """pred: [label or None] per card -> per site: class / family agreement with Gemini (labelled cards; None = no answer,
    counted wrong), and on the audited items right / right-or-close (identity_study.grade on the agent's labels)."""
    out = {}
    for site in [*SITES, "all"]:
        rows = [i for i in range(D.n) if (site == "all" or D.site[i] == site) and (mask is None or mask[i])]
        lab = [i for i in rows if D.label[i] is not None]
        aud = [i for i in rows if D.meta[i]["audit"]]
        g = [grade_audit((names[i] if names else None) or pred[i], D.meta[i]["audit"]) for i in aud]
        out[site] = {"n": len(lab), "class": round(np.mean([pred[i] == D.label[i] for i in lab]), 3) if lab else None,
                     "family": round(np.mean([pred[i] is not None and family(pred[i]) == family(D.label[i]) for i in lab]), 3) if lab else None,
                     "audited_n": len(aud), "right": round(np.mean([x == "right" for x in g]), 3) if g else None,
                     "right_or_close": round(np.mean([x in ("right", "close") for x in g]), 3) if g else None}
    return out


JEV_STATE = ("This image is cropped from a video walk-through of a workplace (a workshop, warehouse or store). A white outline "
             "with a black edge, tagged 1, marks one region.")
JEV_Q, NONE = "What is the outlined object?", "none of these"
KNN_KEYS = (("dinov2-l", "plain+masked"), ("pe-core-l", "masked"))  # the union of the per-fold picks (nested on training videos)
ZS_KEYS = (("pe-core-l", "masked"), ("pe-core-l", "plain+masked"))


def first_seen_order(D, site):
    return sorted(np.where(D.site == site)[0], key=lambda i: (D.meta[i]["first_seen_s"] or 0., i))


def earlier_knn(D, V, site, labels=None):
    """In-video k-NN over the video's EARLIER cards (first-seen order) with their labels (default: Gemini's) -> {i: knn row}."""
    labels = labels or D.label
    out, idx = {}, first_seen_order(D, site)
    for k, i in enumerate(idx):
        b = [j for j in idx[:k] if labels[j] is not None]
        out[i] = knn(V[[i]], V[b], [labels[j] for j in b])[0]
        out[i]["names"] = {lab: D.meta[b[j]]["name"] for lab, j in out[i]["nn"].items()} if b else {}
    return out


def options(D, i, ki, zs, max_opts=12):
    """Jev's options for card i, fold-independent: SAM 3's top-3 labels, zero-shot top-4 (both ZS keys), OWLv2 top-2, the
    in-video bank's top-3 (both kNN keys, earlier cards) -> display names (a class word, or the neighbour's / SAM 3's own
    name for a label outside the taxonomy), alphabetical (no rank cue), + 'none of these'. -> (options, {option: label})"""
    m, disp = D.meta[i], {}
    v = {}
    for w, sc in (m.get("sam3_votes") or {}).items():
        lab = label_of(w)
        if lab:
            v[lab] = v.get(lab, 0.) + sc
            if lab.startswith("~"):
                disp.setdefault(lab, w)
    labs = sorted(v, key=lambda x: -v[x])[:3]
    for P in zs:
        labs += [CLASSES[j] for j in np.argsort(-P[i])[:4]]
    _, _, ow = owl_vote(m, D.owl)
    labs += sorted(ow, key=lambda x: -ow[x])[:2]
    for e in ki:
        r = e[i]
        labs += r["ranked"][:3]
        for lab in r["ranked"][:3]:
            if lab.startswith("~"):
                disp.setdefault(lab, r["names"].get(lab) or lab[1:])
    seen, out = set(), []
    for lab in labs:
        if lab not in seen and lab != cards.NOT_OBJECT:
            seen.add(lab)
            out.append(lab)
    out = out[:max_opts]
    names = {(disp.get(lab) or lab[1:]) if lab.startswith("~") else lab: lab for lab in out}
    return sorted(names) + [NONE], names


def questions(out):
    """One Jev-Omni question per card (its best view, outlined) -> OUT/questions.json (no labels in it)."""
    D = Data(out)
    zs = [D.zero_shot(*k) for k in ZS_KEYS]
    ki = [{i: r for s in SITES for i, r in earlier_knn(D, D.vec(*k), s).items()} for k in KNN_KEYS]
    qs, recall = [], []
    for i, m in enumerate(D.meta):
        opts, names = options(D, i, ki, zs)
        qs.append({"key": m["key"], "crop": f"{m['key']}|som", "state": JEV_STATE, "question": JEV_Q, "options": opts, "labels": names})
        if m["label"]:
            recall.append((m["site"], m["label"] in names.values()))
    (out / "questions.json").write_text(json.dumps(qs))
    print(json.dumps({"questions": len(qs), "options_median": float(np.median([len(q["options"]) for q in qs])),
                      "gemini_label_in_options": {s: round(np.mean([r for x, r in recall if x == s]), 3) for s in SITES}}))


# ---------------------------------------------------------------- (c) Jev-Omni's answers, (d) the cascade, (e) bank growth

RULES = ("sam3+cross bank", "sam3+zero-shot/owl", "sam3 word", "cross bank")  # no in-video names needed: fitted exactly as run
EMPTY = {"label": None, "share": 0., "cos1": -1., "ranked": [], "nn": {}}
KNN_CHOICES = [(e, v) for e in ("siglip2-so400m", "pe-core-l", "dinov2-l") for v in ("plain", "masked", "plain+masked")]
ZS_CHOICES = [(e, v) for e in ("siglip2-base", "siglip2-so400m", "pe-core-l") for v in ("plain", "masked", "plain+masked", "som")]
D_GRID = np.round(np.arange(.02, .62, .02), 2)  # cluster cut, cosine distance (average linkage)


def jev_answers(out, D):
    """{card i: (label or None for 'none of these', p of the top option, top option, [(option, p)])}."""
    qs = json.loads((out / "questions.json").read_text())
    probs = pickle.loads((out / "jev.pkl").read_bytes())["probs"]
    at = {m["key"]: i for i, m in enumerate(D.meta)}
    res = {}
    for q in qs:
        p = probs[q["key"]]
        k = int(np.argmax(p))
        top = q["options"][k]
        res[at[q["key"]]] = (None if top == NONE else q["labels"][top], float(p[k]), top, list(zip(q["options"], p)))
    return res


def pick_config(D, train):
    """Nested choice on the training videos only: the kNN key with the best in-video (earlier cards) k-NN agreement and the
    zero-shot key with the best zero-shot agreement, both against Gemini's labels."""
    def knn_acc(k):
        V = D.vec(*k)
        return np.mean([np.mean([r["label"] == D.label[i] for i, r in earlier_knn(D, V, s).items() if D.label[i]]) for s in train])

    def zs_acc(k):
        P = D.zero_shot(*k)
        return np.mean([np.mean([CLASSES[int(P[i].argmax())] == D.label[i] for i in np.where(D.site == s)[0] if D.label[i]]) for s in train])
    return {"knn": max(KNN_CHOICES, key=knn_acc), "zs": max(ZS_CHOICES, key=zs_acc)}


def cross_knn(D, idx, cross, V):
    """k-NN of cards idx against the cross-video bank [(card j, label, name)] -> [knn row with the neighbours' names]."""
    if not cross:
        return [dict(EMPTY, names={}) for _ in idx]
    cv = [j for j, _, _ in cross]
    rows = knn(V[idx], V[cv], [lab for _, lab, _ in cross])
    for r in rows:
        r["names"] = {lab: cross[j][2] for lab, j in r["nn"].items()}
    return rows


def signals(D, i, P, kx):
    return {"s3": sam3_vote(D.meta[i]), "zs": (CLASSES[int(P[i].argmax())], float(P[i].max())), "owl": owl_vote(D.meta[i], D.owl)[:2], "kx": kx}


def rule_scores(sg):
    """Each tier-1 rule's (label, score) for one card, None where the rule does not apply."""
    s3, s3s, _ = sg["s3"]
    kx = sg["kx"]
    return {"sam3+cross bank": (s3, min(s3s, kx["share"])) if s3 and s3 == kx["label"] else None,
            "sam3+zero-shot/owl": (s3, s3s) if s3 and s3 in (sg["zs"][0], sg["owl"][0]) else None,
            "sam3 word": (s3, s3s) if s3 else None,
            "cross bank": (kx["label"], kx["share"]) if kx["label"] else None}


def tier1(sg, th):
    for r, x in rule_scores(sg).items():
        if x is not None and x[1] >= th[r]:
            return r, x[0]
    return None, None


def clusters(V, idx, d):
    """Average-linkage clusters of cards idx at cosine distance d -> [[card]] (each list: members, medoid first)."""
    if len(idx) < 2 or d <= 0:
        return [[i] for i in idx]
    from sklearn.cluster import AgglomerativeClustering
    lab = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average", distance_threshold=float(d)).fit_predict(V[idx])
    out = []
    for c in np.unique(lab):
        mem = [idx[k] for k in np.where(lab == c)[0]]
        sim = V[mem] @ V[mem].T
        out.append(sorted(mem, key=lambda i: -sim[mem.index(i)].mean()))
    return out


def name_clusters(D, V, idx, d, verify=False):
    """One VLM call per cluster (its medoid, the labelled member most like the rest; Gemini's offline name stands in for the
    answer), the name copied to the members; verify: only to members whose SAM 3 word gives the same label (the others are
    left for the next step: 'rest'). A cluster without a Gemini-named member: each member is its own call."""
    recs = {}
    for mem in clusters(V, idx, d):
        med = next((i for i in mem if D.label[i]), None)
        for i in mem:
            if med is None or i == med:
                recs[i] = {"tier": "vlm", "rule": "vlm", "label": D.label[i], "name": D.meta[i]["name"]}
            elif verify and sam3_vote(D.meta[i])[0] != D.label[med]:
                recs[i] = {"tier": "rest"}
            else:
                recs[i] = {"tier": "cluster", "rule": "cluster+sam3" if verify else "cluster", "label": D.label[med],
                           "name": D.meta[med]["name"], "medoid": med}
    return recs


def fit_cut(D, V, rest, train, ok, target, verify):
    """The largest cluster cut whose copied names reach `target` (>= 20 copies over the training videos), 0 when none does."""
    best = 0.
    for d in D_GRID:
        got = []
        for s in train:
            got += [ok(r["label"], D.label[i]) for i, r in name_clusters(D, V, [i for i in rest if D.site[i] == s], d, verify).items()
                    if r["tier"] == "cluster" and D.label[i]]
        if len(got) >= 20 and np.mean(got) >= target:
            best = float(d)
    return best


def in_video(D, V, rest, th):
    """The in-video step on the cards tier 1 and Jev left: SAM 3-verified copies at th['cluster_verified'], then plain copies
    at th['cluster'] over what is still unnamed; every medoid is one VLM call."""
    recs = name_clusters(D, V, rest, th["cluster_verified"], verify=True) if th["cluster_verified"] > 0 else {i: {"tier": "rest"} for i in rest}
    left = [i for i in rest if recs[i]["tier"] == "rest"]
    recs.update(name_clusters(D, V, left, th["cluster"]))
    return recs


def fit(D, train, cfg, jev, target, level="class", test=None):
    """Thresholds from the training videos only (agreement with Gemini's label at `level` >= target on what each step
    accepts): the tier-1 rules (a training video's cross bank = every other non-test video), then Jev-Omni's p over the
    cards no rule accepted, then the cluster cut over the cards left (the largest cut whose copied names reach the target)."""
    V, P = D.vec(*cfg["knn"]), D.zero_shot(*cfg["zs"])
    ok = (lambda a, b: a == b) if level == "class" else (lambda a, b: a is not None and family(a) == family(b))
    rows = []
    for s in train:
        q = list(first_seen_order(D, s))
        cross = [(j, D.label[j], D.meta[j]["name"]) for j in range(D.n) if D.site[j] not in (s, test) and D.label[j]]
        rows += [(i, signals(D, i, P, kx)) for i, kx in zip(q, cross_knn(D, q, cross, V))]
    th = {}
    for r in RULES:
        sc = [(x[1], ok(x[0], D.label[i])) for i, sg in rows if D.label[i] for x in [rule_scores(sg)[r]] if x is not None]
        th[r] = fit_threshold([a for a, _ in sc], [b for _, b in sc], target) if sc else float("inf")
    rest = [i for i, sg in rows if tier1(sg, th)[0] is None]
    jj = [(jev[i][1], ok(jev[i][0], D.label[i])) for i in rest if jev[i][0] is not None and D.label[i]]
    th["jev"] = fit_threshold([a for a, _ in jj], [b for _, b in jj], target) if jj else float("inf")
    rest = [i for i in rest if not (jev[i][0] is not None and jev[i][1] >= th["jev"])]
    th["cluster_verified"] = fit_cut(D, V, rest, train, ok, target, True)
    if th["cluster_verified"] > 0:  # what the verified step leaves unnamed (per video, as run)
        rest = [i for s in train for i, r in name_clusters(D, V, [i for i in rest if D.site[i] == s], th["cluster_verified"], True).items() if r["tier"] == "rest"]
    th["cluster"] = fit_cut(D, V, rest, train, ok, target, False)
    return th


def simulate(D, site, cross, th, cfg, jev, use_jev=True, use_clusters=True):
    """One video, batch: tier 1 (SAM 3 word agreement and the cross-video bank of earlier VLM answers, at the fitted
    thresholds) -> Jev-Omni (top option not 'none', p >= its threshold) -> the rest clustered in-video, one VLM call per
    cluster, the answer copied to its members (write-back inside the video). Names accepted without a VLM never enter a bank."""
    V, P = D.vec(*cfg["knn"]), D.zero_shot(*cfg["zs"])
    idx = list(first_seen_order(D, site))
    recs, rest = {}, []
    for i, kx in zip(idx, cross_knn(D, idx, cross, V)):
        sg = signals(D, i, P, kx)
        rule, label = tier1(sg, th)
        if rule:
            name = sg["s3"][2] if rule.startswith("sam3") else kx["names"].get(label) or label.lstrip("~")
            recs[i] = {"tier": "tier1", "rule": rule, "label": label, "name": name}
        elif use_jev and jev[i][0] is not None and jev[i][1] >= th["jev"]:
            recs[i] = {"tier": "jev", "rule": "jev", "label": jev[i][0], "name": jev[i][2]}
        else:
            rest.append(i)
    recs.update(in_video(D, V, rest, th if use_clusters else {"cluster_verified": 0., "cluster": 0.}))
    return recs


def simulate_stream(D, site, cross, th, jev_unused=None):
    """The first design, kept for its failure: cards one at a time, EVERY final name written back into an in-video bank,
    accepted on that bank's k-NN vote share alone (th: share cut). With a near-empty bank one neighbour is a 100 % vote."""
    V = D.vec("dinov2-l", "plain+masked")
    bank, recs = [], {}
    for i in first_seen_order(D, site):
        r = knn(V[[i]], V[[j for j, _ in bank]], [lab for _, lab in bank])[0] if bank else EMPTY
        if r["label"] is not None and r["share"] >= th:
            recs[i] = {"tier": "tier1", "rule": "in-video bank", "label": r["label"], "name": r["label"].lstrip("~")}
        else:
            recs[i] = {"tier": "vlm", "rule": "vlm", "label": D.label[i], "name": D.meta[i]["name"]}
        if recs[i]["label"] is not None:
            bank.append((i, recs[i]["label"]))
    return recs


def summarise(D, recs):
    """Per video (and all): tier shares, VLM objects and requests (14 a request), accepted-name agreement with Gemini's class
    and family, and the agent-labelled audit: final names (accepted + Gemini's for the VLM share) and Gemini alone."""
    out = {}
    for site in [*SITES, "all"]:
        ids = [i for i in recs if site == "all" or D.site[i] == site]
        if not ids:
            continue
        acc = [i for i in ids if recs[i]["tier"] != "vlm"]
        lab = [i for i in acc if D.label[i]]
        aud = [i for i in ids if D.meta[i]["audit"]]
        g = [grade_audit(recs[i]["name"], D.meta[i]["audit"]) for i in aud]
        g0 = [grade_audit(D.meta[i]["name"] or sam3_vote(D.meta[i])[2], D.meta[i]["audit"]) for i in aud]
        ga = [grade_audit(recs[i]["name"], D.meta[i]["audit"]) for i in aud if recs[i]["tier"] != "vlm"]
        vlm = sum(recs[i]["tier"] == "vlm" for i in ids)
        by_rule = {}
        for i in [i for i in ids if recs[i]["tier"] != "vlm"]:
            by_rule.setdefault(recs[i]["rule"], []).append(i)
        out[site] = {
            "cards": len(ids), "vlm_share": round(vlm / len(ids), 3), "vlm_objects": vlm,
            "vlm_requests_14": sum(-(-sum(recs[i]["tier"] == "vlm" for i in ids if D.site[i] == s) // 14) for s in SITES if site in (s, "all")),
            "tier_share": {t: round(sum(recs[i]["tier"] == t for i in ids) / len(ids), 3) for t in ("tier1", "jev", "cluster", "vlm")},
            "accepted_vs_gemini": {"n": len(lab), "class": round(np.mean([recs[i]["label"] == D.label[i] for i in lab]), 3) if lab else None,
                                   "family": round(np.mean([family(recs[i]["label"]) == family(D.label[i]) for i in lab]), 3) if lab else None},
            "by_rule": {r: {"n": len(v), "class": round(np.mean([recs[i]["label"] == D.label[i] for i in v if D.label[i]]), 3) if any(D.label[i] for i in v) else None}
                        for r, v in by_rule.items()},
            "audited": {"n": len(aud), "final_right": _share(g, ("right",)), "final_right_or_close": _share(g, ("right", "close")),
                        "accepted_n": len(ga), "accepted_right": _share(ga, ("right",)), "accepted_right_or_close": _share(ga, ("right", "close")),
                        "gemini_right": _share(g0, ("right",)), "gemini_right_or_close": _share(g0, ("right", "close"))}}
    return out


def _share(g, ok):
    return round(float(np.mean([x in ok for x in g])), 3) if g else None


FAMILY_OF_SITE = {"me340": "shop floor", "samsclub-a2": "retail", "walmart": "retail"}
OPERATING = {  # name: (target, level, fit scope)
    "class>=0.90, pooled": (.9, "class", "pooled"), "class>=0.80, pooled": (.8, "class", "pooled"),
    "family>=0.95, pooled": (.95, "family", "pooled"), "class>=0.80, same domain": (.8, "class", "domain"),
    "class>=0.90, same domain": (.9, "class", "domain")}


def train_of(test, scope):
    """pooled: the other two videos; domain: the other video of the test's domain family when there is one (retail), else pooled."""
    other = [s for s in SITES if s != test]
    same = [s for s in other if FAMILY_OF_SITE[s] == FAMILY_OF_SITE[test]]
    return same if scope == "domain" and same else other


def curve(D, idx, score, pred, covs=(.2, .4, .6, .8, 1.)):
    """Precision (agreement with Gemini's class) of the top-`cov` share of the labelled cards ranked by score."""
    idx = [i for i in idx if D.label[i]]
    o = sorted(idx, key=lambda i: -score[i])
    return {str(c): round(float(np.mean([pred[i] == D.label[i] for i in o[:max(1, int(c * len(o)))]])), 3) for c in covs}


def ece(p, ok, bins=10):
    p, ok = np.asarray(p, float), np.asarray(ok, float)
    e = 0.
    for b in range(bins):
        m = (p >= b / bins) & (p < (b + 1) / bins + (b == bins - 1))
        if m.any():
            e += m.mean() * abs(p[m].mean() - ok[m].mean())
    return round(float(e), 4)


def score(out, results):
    import collections
    import time
    D = Data(out)
    J = jev_answers(out, D)
    lab = np.array([x is not None for x in D.label])
    R = {"dataset": {}, "a_retrieval_and_votes": {}, "b_detector": {}, "c_jev": {}, "d_cascade": {}, "e_bank_growth": {}, "latency": {}}
    for s in SITES:
        ids = np.where(D.site == s)[0]
        other = {D.label[i] for i in range(D.n) if D.site[i] != s and D.label[i]}
        R["dataset"][s] = {"cards": len(ids), "gemini_labelled": int(lab[ids].sum()), "audited": sum(1 for i in ids if D.meta[i]["audit"]),
                           "views_per_card": round(float(np.mean([len(D.meta[i]["views"]) for i in ids])), 2),
                           "top_labels": collections.Counter(D.label[i] for i in ids if D.label[i]).most_common(6),
                           "label_outside_taxonomy": round(float(np.mean([D.label[i].startswith("~") for i in ids if D.label[i]])), 3),
                           "class_present_in_other_videos": round(float(np.mean([D.label[i] in other for i in ids if D.label[i]])), 3)}
    # (a) held out by video: the bank = the other two videos' Gemini-labelled cards; in-video = earlier / all other cards of the video
    A = R["a_retrieval_and_votes"]
    maj = [None] * D.n
    for s in SITES:
        c = collections.Counter(D.label[i] for i in range(D.n) if D.site[i] != s and D.label[i]).most_common(1)[0][0]
        for i in np.where(D.site == s)[0]:
            maj[i] = c
    A["majority class of the other videos"] = table(maj, D)
    s3 = [sam3_vote(m) for m in D.meta]
    A["sam3 word vote (per-video Gemini vocabulary)"] = table([x[0] for x in s3], D, names=[x[2] for x in s3])
    for enc in ENCODERS:
        for var in ("plain", "masked", "plain+masked", "som"):
            V = D.vec(enc, var)
            if enc != "dinov2-l":
                P = D.zero_shot(enc, var)
                A[f"zero-shot over the taxonomy: {enc} {var}"] = table([CLASSES[int(j)] for j in P.argmax(1)], D)
            px, pi, pe = [None] * D.n, [None] * D.n, [None] * D.n
            for s in SITES:
                q = np.where(D.site == s)[0]
                b = np.where((D.site != s) & lab)[0]
                for i, r in zip(q, knn(V[q], V[b], [D.label[j] for j in b])):
                    px[i] = r["label"]
                for i in q:
                    bb = q[(q != i) & lab[q]]
                    pi[i] = knn(V[[i]], V[bb], [D.label[j] for j in bb])[0]["label"]
                for i, r in earlier_knn(D, V, s).items():
                    pe[i] = r["label"]
            A[f"kNN cross-video bank (other 2 videos): {enc} {var}"] = table(px, D)
            A[f"kNN in-video, earlier cards named (stream bound): {enc} {var}"] = table(pe, D)
            A[f"kNN in-video, all other cards named (upper bound): {enc} {var}"] = table(pi, D)
    # (b) OWLv2 with the taxonomy words, matched to the SAM 3 masks (box IoU >= 0.5), averaged over the card's views
    ow = [owl_vote(m, D.owl) for m in D.meta]
    R["b_detector"] = {"model": MODEL_IDS["owlv2-base"], "match": "OWLv2 box IoU >= 0.5 with the mask's box on each view; best score per class per view, mean over views",
                       "all cards (no match = wrong)": table([x[0] for x in ow], D),
                       "coverage (a matched box)": {s: round(float(np.mean([ow[i][0] is not None for i in np.where(D.site == s)[0]])), 3) for s in SITES},
                       "precision by score rank": {s: curve(D, list(np.where(D.site == s)[0]), {i: ow[i][1] for i in range(D.n)}, [x[0] for x in ow]) for s in SITES},
                       "agrees with the SAM 3 word": {s: {"share": round(float(np.mean([ow[i][0] is not None and ow[i][0] == s3[i][0] for i in np.where((D.site == s) & lab)[0]])), 3),
                                                           "agreement_with_gemini_when_they_agree": round(float(np.mean([s3[i][0] == D.label[i] for i in np.where((D.site == s) & lab)[0] if ow[i][0] == s3[i][0]] or [np.nan])), 3)} for s in SITES}}
    A["precision by score rank"] = {
        "sam3 vote share": {s: curve(D, list(np.where(D.site == s)[0]), {i: s3[i][1] for i in range(D.n)}, [x[0] for x in s3]) for s in SITES}}
    # (c) Jev-Omni over the candidates (SAM 3 top-3, zero-shot top-4 x 2 keys, OWLv2 top-2, in-video bank top-3 x 2 keys) + none
    qs = json.loads((out / "questions.json").read_text())
    C = R["c_jev"]
    C["candidates"] = {s: {"median_options": float(np.median([len(qs[i]["options"]) for i in np.where(D.site == s)[0]])),
                           "gemini_label_among_options": round(float(np.mean([D.label[i] in qs[i]["labels"].values() for i in np.where((D.site == s) & lab)[0]])), 3)} for s in SITES}
    C["forced choice (top option)"] = table([J[i][0] for i in range(D.n)], D, names=[J[i][2] if J[i][0] else None for i in range(D.n)])
    C["none_of_these_chosen"] = round(float(np.mean([J[i][0] is None for i in range(D.n)])), 4)
    C["precision by p"] = {}
    for s in [*SITES, "all"]:
        r = [i for i in range(D.n) if (s == "all" or D.site[i] == s) and D.label[i]]
        C["precision by p"][s] = {f"p>={t}": {"coverage": round(len(a) / len(r), 3), "class": round(float(np.mean([J[i][0] == D.label[i] for i in a])), 3) if a else None}
                                  for t in (.5, .7, .9, .95) for a in [[i for i in r if J[i][0] is not None and J[i][1] >= t]]}
        C["precision by p"][s]["ece_top_option_vs_gemini_class"] = ece([J[i][1] for i in r], [J[i][0] == D.label[i] for i in r])
    # (d) the cascade on each held-out video, thresholds fitted on the other video(s); cross bank = the other two videos
    Dd = R["d_cascade"]
    fits = {}
    for op, (target, level, scope) in OPERATING.items():
        recs, folds = {}, {}
        for test in SITES:
            tr = train_of(test, scope)
            cfg = pick_config(D, tr)
            th = fit(D, tr, cfg, J, target, level, test)
            fits[(op, test)] = (cfg, th)
            cross = [(j, D.label[j], D.meta[j]["name"]) for j in range(D.n) if D.site[j] != test and D.label[j]]
            recs.update(simulate(D, test, cross, th, cfg, J))
            folds[test] = {"trained_on": tr, "knn_key": cfg["knn"], "zero_shot_key": cfg["zs"], "thresholds": {k: (None if v == float("inf") else round(v, 4)) for k, v in th.items()}}
        Dd[op] = {"folds": folds, "per_video": summarise(D, recs)}
    base = "class>=0.80, pooled"
    for name, kw in (("without the in-video clusters (tier 1 + Jev, the rest to the VLM)", {"use_clusters": False}),
                     ("without Jev-Omni", {"use_jev": False})):
        recs = {}
        for test in SITES:
            cfg, th = fits[(base, test)]
            cross = [(j, D.label[j], D.meta[j]["name"]) for j in range(D.n) if D.site[j] != test and D.label[j]]
            recs.update(simulate(D, test, cross, th, cfg, J, **kw))
        Dd[f"{base}: {name}"] = {"per_video": summarise(D, recs)}
    recs = {}
    for test in SITES:
        recs.update(simulate_stream(D, test, [], .8))
    Dd["failed design: stream, every final name written back, accept on the in-video vote share >= 0.8"] = {"per_video": summarise(D, recs)}
    # (e) bank growth: every order of the three videos from an empty bank; the bank holds the VLM answers only
    import itertools
    E = R["e_bank_growth"]
    for op in ("class>=0.80, pooled", "class>=0.90, pooled", "class>=0.80, same domain"):
        E[op] = {}
        for order in itertools.permutations(SITES):
            bank, rows = [], []
            for pos, v in enumerate(order):
                cfg, th = fits[(op, v)]
                recs = simulate(D, v, bank, th, cfg, J)
                sm = summarise(D, recs)[v]
                rows.append({"video": v, "position": pos + 1, "bank_rows_before": len(bank), "vlm_share": sm["vlm_share"], "tier_share": sm["tier_share"],
                             "accepted_vs_gemini": sm["accepted_vs_gemini"], "audited": sm["audited"]})
                bank += [(i, r["label"], r["name"]) for i, r in recs.items() if r["tier"] == "vlm" and r["label"]]
            E[op][" -> ".join(order)] = rows
    # latency (per object, batched): encoders and OWLv2 measured in the features container; kNN and clustering here (CPU)
    T = D.F["timing"]
    vpc = len(D.F["idx"]) / D.n
    V = D.vec("dinov2-l", "plain+masked")
    b = np.where(lab)[0]
    t = time.perf_counter()
    for _ in range(20):
        S = V[:256] @ V[b].T
        np.argpartition(-S, K, 1)
    knn_ms = (time.perf_counter() - t) / 20 / 256 * 1000
    t = time.perf_counter()
    clusters(V, list(np.where(D.site == "walmart")[0]), .2)
    cl_s = time.perf_counter() - t
    jv = pickle.loads((out / "jev.pkl").read_bytes())
    trips = jv["trips"]
    ow_t = pickle.loads((out / "owl.pkl").read_bytes())["timing"]
    R["latency"] = {
        "views_per_card": round(vpc, 2),
        "encoders_ms_per_crop_A100": {k: T[k]["masked"]["ms_per_crop"] for k in ENCODERS if "masked" in T[k]},
        "encoders_ms_per_card (views x crops)": {"dinov2-l plain+masked": round(2 * vpc * T["dinov2-l"]["masked"]["ms_per_crop"], 1),
                                                 "pe-core-l masked": round(vpc * T["pe-core-l"]["masked"]["ms_per_crop"], 1),
                                                 "pe-core-l plain+masked": round(2 * vpc * T["pe-core-l"]["masked"]["ms_per_crop"], 1)},
        "encoder_load_s": D.F["load_s"],
        "owlv2_per_keyframe_s (102 queries, batch 8, incl. pre/post)": ow_t["per_frame_s"], "owlv2_model_only_s_per_keyframe": round(ow_t["model_s"] / ow_t["frames"], 4),
        "owlv2_ms_per_card (keyframes / cards)": round(1000 * ow_t["total_s"] / D.n, 1),
        "knn_ms_per_query_cpu (bank %d x %d)" % (len(b), V.shape[1]): round(knn_ms, 4),
        "cluster_s_walmart_720_cards_cpu": round(cl_s, 3),
        "jev": {"compute_s_per_question_batched_8": round(sum(x["compute_s"] for x in trips) / sum(x["n"] for x in trips), 4),
                "round_trip_s_per_question_in_64_question_requests": round(sum(x["rtt_s"] for x in trips) / sum(x["n"] for x in trips), 4),
                "hop_s_per_64_question_request_median": round(float(np.median([x["hop_s"] for x in trips])), 3),
                "request_mb_median": round(float(np.median([x["bytes"] for x in trips])) / 1e6, 2),
                "single_question_round_trip_s_median": round(float(np.median([x["rtt_s"] for x in jv["single"]])), 3),
                "single_question_compute_s_median": round(float(np.median([x["compute_s"] for x in jv["single"]])), 3),
                "cold_start_s (container + load)": jv["cold_s"], "model_load_s": jv["load_s"], "peak_gib_torch_reserved": jv["peak_gib"],
                "caller": "a CPU Modal container (the core's stand-in) calling the service container through Modal's function calls, same region"}}
    results.update(R)
    return R


def audit_sheets(out, dest, n=(("samsclub-a2", 24), ("walmart", 24), ("me340", 16)), seed=2):
    """Blind agent audit of the cheap tiers' disagreements with Gemini ('class>=0.80, same domain'): a seeded sample of
    accepted cards whose label differs from Gemini's (the 119 identity items left out) -> dest/items.json + id-only sheets."""
    import cv2
    from mvp_sheets import sheet
    D = Data(out)
    J = jev_answers(out, D)
    recs = {}
    for test in SITES:
        tr = train_of(test, "domain")
        cfg = pick_config(D, tr)
        th = fit(D, tr, cfg, J, .8, "class", test)
        recs.update(simulate(D, test, [(j, D.label[j], D.meta[j]["name"]) for j in range(D.n) if D.site[j] != test and D.label[j]], th, cfg, J))
    rng, items = np.random.default_rng(seed), []
    for s, k in n:
        pool = [i for i in range(D.n) if D.site[i] == s and recs[i]["tier"] != "vlm" and D.label[i] and recs[i]["label"] != D.label[i] and not D.meta[i]["audit"]]
        items += [{"id": D.meta[i]["key"], "i": int(i), "site": s, "pool": len(pool)} for i in sorted(rng.choice(pool, min(k, len(pool)), replace=False).tolist())]
    data = pickle.loads((out / "dataset.pkl").read_bytes())
    tiles = []
    for k, it in enumerate(items):
        m = D.meta[it["i"]]
        som = cv2.imdecode(np.frombuffer(data["crops"][f"{m['key']}|som"], np.uint8), 1)
        sc = 420 / max(som.shape[:2])
        som = cv2.resize(som, (int(som.shape[1] * sc), int(som.shape[0] * sc)))
        som = cv2.copyMakeBorder(som, 0, 420 - som.shape[0], 0, 420 - som.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255))
        v = m["views"][0]
        fr = cv2.imdecode(np.frombuffer(data["frames"][f"{m['site']}|{v['frame']}"], np.uint8), 1)
        b = [int(x) for x in v["box"]]
        cv2.rectangle(fr, (b[0], b[1]), (b[2], b[3]), (0, 230, 255), 4)
        fr = cv2.copyMakeBorder(cv2.resize(fr, (420, 236)), 0, 184, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        tiles.append((np.hstack([som, fr]), f"A{k:02d}  {it['id']}"))
    (dest / "sheets").mkdir(parents=True, exist_ok=True)
    for s in range(0, len(tiles), 8):
        sheet(tiles[s:s + 8], dest / "sheets" / f"disagree-{s // 8:02d}.jpg", cols=2, tile=840)
    (dest / "items.json").write_text(json.dumps({"items": [{**it, "cheap_name": recs[it["i"]]["name"], "cheap_label": recs[it["i"]]["label"],
                                                            "cheap_rule": recs[it["i"]]["rule"], "gemini_name": D.meta[it["i"]]["name"],
                                                            "gemini_label": D.label[it["i"]]} for it in items]}, indent=1))


def audit_grade(dest):
    """items.json + labels-agent.json (the agent's blind labels) -> per video: the cheap name's and Gemini's grades."""
    import identity_study as ids
    items = json.loads((dest / "items.json").read_text())["items"]
    lab = json.loads((dest / "labels-agent.json").read_text())
    out = {}
    for k, it in enumerate(items):
        L = lab[f"A{k:02d}"]
        o = out.setdefault(it["site"], {"sampled": 0, "unclear": 0, "labelled": 0, "cheap": {"right": 0, "close": 0, "wrong": 0},
                                        "gemini": {"right": 0, "close": 0, "wrong": 0}, "pool": it["pool"]})
        o["sampled"] += 1
        if L["canon"] == "unclear":
            o["unclear"] += 1
            continue
        o["labelled"] += 1
        o["cheap"][ids.grade(it["cheap_name"], L)] += 1
        o["gemini"][ids.grade(it["gemini_name"], L)] += 1
    return out


def self_check():
    assert label_of("Extension cords") == "cable" and label_of("bellows cover") == "~cover" and label_of(None) is None
    assert family("cabinet") == "storage" and family("~cover") == "~cover"
    assert pick_views({"views": {"best": [10, 30]}}, {10: [[[0, 0], [5, 5], [0, 5]]], 12: [[[100, 100], [300, 300], [100, 300]]],
                                                       30: [[[50, 50], [60, 60], [50, 60]]]}, 1280, 720) == [10, 30]
    img, mask = tight_crop(np.zeros((720, 1280, 3), np.uint8), [[[100, 100], [200, 100], [200, 300], [100, 300]]])
    assert img.shape == (SIDE, SIDE, 3) and mask[SIDE // 2, SIDE // 2] == 255 and mask[SIDE // 2, 5] == 0
    assert fit_threshold([.9, .8, .7, .6], [1, 1, 0, 0], .9, min_n=1) == .8 and fit_threshold([.9, .8], [0, 0], .5, min_n=1) == float("inf")
    r = knn(np.eye(3)[:1], np.eye(3), ["a", "b", "a"], k=3)[0]
    assert r["label"] == "a" and r["cos1"] == 1. and r["share"] > .9 and r["nn"]["a"] == 0
    assert sam3_vote({"sam3_votes": {"power cord": 1., "hose": .5, "cable": .4}})[:1] == ("cable",)
    assert iou([0, 0, 10, 10], [0, 0, 10, 10]) > .99 and iou([0, 0, 1, 1], [5, 5, 6, 6]) == 0
    V = np.array([[1, 0], [.99, .14], [0, 1], [.14, .99]], float)
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    assert sorted(map(sorted, clusters(V, [0, 1, 2, 3], .1))) == [[0, 1], [2, 3]] and clusters(V, [0, 1], 0.) == [[0], [1]]
    print("x13_naming self-check ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "questions", "score", "audit-sheets"], nargs="?")
    ap.add_argument("--audit", default=None, help="the audit folder (items.json, labels-agent.json)")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--results", default=None)
    ap.add_argument("--self-check", action="store_true")
    a = ap.parse_args()
    if a.self_check:
        self_check()
    elif a.cmd == "build":
        build(Path(a.out))
    elif a.cmd == "questions":
        questions(Path(a.out))
    elif a.cmd == "audit-sheets":
        audit_sheets(Path(a.out), Path(a.audit))
    elif a.cmd == "score":
        res = {}
        score(Path(a.out), res)
        if a.audit:
            res["agent_audit_of_disagreements"] = audit_grade(Path(a.audit))
        Path(a.results).write_text(json.dumps(res, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x)))
