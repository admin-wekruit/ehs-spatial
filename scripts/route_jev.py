"""Route each object card's display model: SIMPLE (a primitive: plane / box / cylinder / open frame) or COMPLEX (a generated
model: RecGen internally, SAM 3D / TRELLIS commercially). Which router is accurate and cheap, held out by video:
  (a) rules: the card's canonical class (cards.TAXONOMY via the naming cascade) + the primitive fits' geometry
  (b) zero-shot embeddings: PE-Core-L / SigLIP 2 so400m against text prompts (no training)
  (c) Jev-Omni: a calibrated multiple-choice decider as a service on its own A100 (modal_apps/route_jev.py)
  (d) the cascade: rules where confident, Jev-Omni for the rest, a VLM never or last
Truth: agent labels by looking at blind contact sheets (not ground truth). Thresholds and temperatures are fitted on the
other two videos and scored on the third, rotated. Nothing is trained.

  python scripts/route_jev.py build OUT SCRATCH   # cards + crops (r4 models bench, warm calls) -> OUT/meta.json, SCRATCH/dataset.pkl
  python scripts/route_jev.py sheets OUT SCRATCH  # the stratified labelled sample's blind sheets + label template
  modal run modal_apps/route_jev.py --data SCRATCH/dataset.pkl --out SCRATCH/answers.pkl
  modal run modal_apps/route_jev.py ... --stage qwen --out SCRATCH/qwen.pkl   # the last tier (the core's Qwen3-VL-8B)
  python scripts/route_jev.py export OUT SCRATCH  # raw answers + zero-shot logits -> OUT/answers.json (no crops needed after this)
  python scripts/route_jev.py score OUT           # OUT/results.json + OUT/results.md
  python scripts/route_jev.py --self-check
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "scripts")]
from fast_report import cards, display_model  # noqa: E402

PHASE2 = Path("/Users/adam/Desktop/panoptes-public/research-notes/phase2")
MIRROR = PHASE2 / "runs/r4-models-bench-001/mirror"
WARM = {"me340": "mvp-me340-e84efffd-1790713300", "samsclub-a2": "mvp-samsclub-a2-d5e0c855-1790713770",
        "walmart": "mvp-walmart-c0761a2a-1790714235"}
K_VIEWS, SIDE, PAD, PER_SITE = 2, 384, .1, 100

# ---------------------------------------------------------------- (a) the rules' class prior
# Written from the user's definition (SIMPLE: walls, floor, ceiling, doors, plain panels / boards, beams, poles / pipes, plain
# boxes / cartons, shelf / rack frames; COMPLEX: machines, carts, shaped guards / fences, equipment, tools, furniture, anything
# irregular) over cards.TAXONOMY, before any object was labelled. A class in neither set (or no class) is left to geometry.
SIMPLE_CLASSES = {"shelf", "rack", "display rack", "cabinet", "locker", "refrigerator", "box", "stacked boxes", "pallet", "crate", "drum",
                  "can", "trash can", "first aid kit", "exit sign", "safety sign", "sign", "label", "floor marking", "bollard", "pipe", "duct",
                  "cable tray", "control panel", "electrical outlet", "switch", "door", "window", "column", "wall panel", "floor drain", "vent",
                  "wooden board", "metal sheet", "whiteboard", "paper", "clipboard", "book", "spill"}
COMPLEX_FAMILIES = {"machine", "tool", "handling", "furniture", "ppe"}
COMPLEX_CLASSES = {"bag", "ladder", "step stool", "work platform", "stairs", "fire extinguisher", "eyewash station", "emergency stop button",
                   "fire alarm", "guard", "railing", "safety cone", "cable", "hose", "fan", "keyboard", "printer", "phone", "rag", "cup",
                   "mannequin", "wrap"}


# POST HOC (written after the labels, flagged wherever used): the pre-registered table minus the classes the labels showed to be
# shape-ambiguous (bag: pet-food bags are boxes; tool box / tool tray: boxes and trays; crate, display rack: mixed)
FIXED_SHAPE_POST_HOC = {"simple": SIMPLE_CLASSES - {"crate", "display rack"}, "drop": {"bag", "tool box", "tool tray"}}


def class_route(cls, post_hoc=False):
    """'simple' | 'complex' | None (no prior: geometry, or the decider, decides)."""
    if not cls or (post_hoc and cls in FIXED_SHAPE_POST_HOC["drop"]):
        return None
    if cls in (FIXED_SHAPE_POST_HOC["simple"] if post_hoc else SIMPLE_CLASSES):
        return "simple"
    if cls in COMPLEX_CLASSES or cards.FAMILY.get(cls) in COMPLEX_FAMILIES:
        return "complex"
    return None


def card_class(card):
    """The naming cascade's canonical class (identity.canonical, else the detector word's), or None."""
    idn = card.get("identity") or {}
    c = idn.get("canonical") or cards.canonical(idn.get("proposed") or idn.get("name") or "")
    return None if c in (None, cards.NOT_OBJECT) else c


def geometry(card):
    """The primitive fits' residuals (m) and the box's extents -> the rules' geometry features."""
    fits = (card.get("raw") or {}).get("model_fits") or {}
    if not fits:
        return None
    r = {k: float(f["residual_m"]) for k, f in fits.items()}
    cyl = fits["cylinder"]
    if not (cyl.get("fitted") and cyl.get("arc", "").count("1") >= display_model.MIN_ARC):
        r["cylinder"] = float("inf")  # display_model.choose's own gate: an unfitted circle is no cylinder
    e = np.sort(np.asarray(fits["box"]["hi"], float) - np.asarray(fits["box"]["lo"], float))
    r_min = min(r.values())
    return {"res": r, "r_min": r_min, "ext": e.round(4).tolist(), "longest": float(e[2]), "rel": r_min / max(float(e[2]), .05),
            "planar": r["plane"] / max(r["box"], 1e-6), "elong": float(e[2] / max(e[1], 1e-3)), "flat": float(e[0] / max(e[1], 1e-3))}


# ---------------------------------------------------------------- build: cards, views, crops

def polys_xy(polys):
    return np.concatenate([np.asarray(p, float).reshape(-1, 2) for p in polys])


def pick_views(card, per_frame, W, H, k=K_VIEWS):
    """The card's best views first, then the largest outlines not cut by the frame edge, >= 5 keyframes apart (X13's rule)."""
    area = {}
    for q, polys in per_frame.items():
        (x0, y0), (x1, y1) = polys_xy(polys).min(0), polys_xy(polys).max(0)
        area[q] = (x1 - x0) * (y1 - y0) * (.5 if x0 <= 2 or y0 <= 2 or x1 >= W - 3 or y1 >= H - 3 else 1.)
    out = [q for q in (card.get("views") or {}).get("best", []) if q in area][:k]
    for q in sorted(area, key=lambda q: -area[q]):
        if len(out) >= k:
            break
        if all(abs(q - p) >= 5 for p in out):
            out.append(q)
    return out


def tight_crop(frame, polys, side=SIDE, pad=PAD):
    """Box + pad, squared, grey outside the frame -> (BGR crop, mask) (X13's crop, cascade.Embedder.crops' geometry)."""
    import cv2
    xy = polys_xy(polys)
    (x0, y0), (x1, y1) = xy.min(0), xy.max(0)
    half = max(x1 - x0, y1 - y0, 8) * (1 + 2 * pad) / 2
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    s = side / (2 * half)
    M = np.array([[s, 0, side / 2 - cx * s], [0, s, side / 2 - cy * s]], np.float32)
    img = cv2.warpAffine(frame, M, (side, side), flags=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC, borderValue=(128, 128, 128))
    mask = np.zeros((side, side), np.uint8)
    cv2.fillPoly(mask, [np.round(np.asarray(p, float).reshape(-1, 2) * s + [side / 2 - cx * s, side / 2 - cy * s]).astype(np.int32) for p in polys], 255)
    return img, mask


def stratified(keys, strata, n, seed=0):
    """n of keys, sqrt(stratum size) allocation capped at the stratum (largest remainder), seeded -> (chosen, {stratum: (N, n)})."""
    groups = {}
    for k, s in zip(keys, strata):
        groups.setdefault(s, []).append(k)
    size = {s: len(v) for s, v in groups.items()}
    alloc = dict.fromkeys(groups, 0)
    left = min(n, len(keys))
    while left:
        open_ = [s for s in groups if alloc[s] < size[s]]
        w = np.array([np.sqrt(size[s]) for s in open_])
        want = w / w.sum() * left
        add = np.floor(want).astype(int)
        for i in np.argsort(-(want - add))[:left - add.sum()]:
            add[i] += 1
        for s, a in zip(open_, add):
            a = min(int(a), size[s] - alloc[s])
            alloc[s] += a
            left -= a
    rng = np.random.default_rng(seed)
    chosen = [k for s in sorted(groups) for k in sorted(rng.choice(groups[s], alloc[s], replace=False).tolist())]
    return chosen, {s: (size[s], alloc[s]) for s in groups}


def build(out, scratch):
    import cv2
    import fast_report_eval as ev
    from fast_report import judge
    from mvp_sheets import frames_bgr
    out.mkdir(parents=True, exist_ok=True)
    meta, crops, sample = [], {}, {}
    for site, rep in WARM.items():
        L = ev.load_layers(MIRROR, rep)
        W, H = L["outlines"]["width"], L["outlines"]["height"]
        alias = L["object_cards"].get("aliases") or {}
        root = lambda e: root(alias[e]) if e in alias and alias[e] != e else e  # noqa: E731
        outl = {}
        for f in L["outlines"]["frames"]:
            for o in f["objects"]:
                if o.get("polygons"):
                    outl.setdefault(root(o["entityId"]), {}).setdefault(f["sourceFrame"], []).extend(o["polygons"])
        rows = []
        for c in L["object_cards"]["cards"]:
            if c.get("kind") != "object" or c["id"] not in outl:
                continue
            views = pick_views(c, outl[c["id"]], W, H)
            cls, g, m = card_class(c), geometry(c), c.get("model") or {}
            idn = c.get("identity") or {}
            rows.append((c, views))
            meta.append({"key": f"{site}|{c['id']}", "site": site, "card": c["id"], "name": idn.get("name"), "proposed": idn.get("proposed"),
                         "cls": cls, "family": cards.FAMILY.get(cls), "class_route": class_route(cls), "geometry": g,
                         "model_kind": m.get("kind"), "model_chosen_by": m.get("chosen_by"), "seen_share": m.get("seen_share"),
                         "views_n": (c.get("views") or {}).get("n"), "azimuth_spread_deg": (c.get("views") or {}).get("azimuth_spread_deg"),
                         "depth_seen": (c.get("raw") or {}).get("depth_seen"), "sam3d": display_model.well_observed(c)[0] is not None,
                         "views": []})
        imgs = frames_bgr(MIRROR / "blobs/sha256" / L["video"]["sha256"], {q for _, vs in rows for q in vs})
        for m, (c, views) in zip(meta[len(meta) - len(rows):], rows):
            per = outl[c["id"]]
            for i, q in enumerate(views):
                img, mask = tight_crop(imgs[q], per[q])
                crops[f"{m['key']}|{i}|plain"] = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()
                crops[f"{m['key']}|{i}|mask"] = cv2.imencode(".png", mask)[1].tobytes()
                xy = polys_xy(per[q])
                m["views"].append({"frame": int(q), "box": [*map(float, xy.min(0)), *map(float, xy.max(0))]})
            crops[f"{m['key']}|som"] = judge.som(imgs[views[0]], {1: per[views[0]]}, subject=1, side=448)
            crops[f"{m['key']}|context"] = judge.som(imgs[views[0]], {1: per[views[0]]}, subject=1, side=448, scale=4.)
        mine = [m for m in meta if m["site"] == site]
        strata = [m["cls"] or "~" + cards.norm(m["proposed"] or "unknown") for m in mine]
        chosen, alloc = stratified([m["key"] for m in mine], strata, PER_SITE)
        sample[site] = {"keys": chosen, "strata": {s: {"N": a[0], "n": a[1]} for s, a in sorted(alloc.items())}}
        print(site, len(rows), "cards,", len(imgs), "frames,", len(chosen), "sampled over", len(alloc), "strata", flush=True)
    scratch.mkdir(parents=True, exist_ok=True)
    with open(scratch / "dataset.pkl", "wb") as f:
        pickle.dump({"meta": meta, "crops": crops}, f, protocol=4)
    (out / "meta.json").write_text(json.dumps(meta))
    (out / "sample.json").write_text(json.dumps({"rule": "per video 100 cards, strata = the canonical class (else '~' + the detector word): "
                                                         "sqrt(stratum size) allocation capped at the stratum, seed 0", "sites": sample}, indent=1))
    print(json.dumps({"cards": len(meta), "crops": len(crops), "mb": round((scratch / "dataset.pkl").stat().st_size / 1e6, 1)}))


# ---------------------------------------------------------------- (b) prompts and (c) questions, written before any label

PROMPTS = {  # zero-shot phrases per primitive / route (the user's words and plain synonyms; no class seen in the data added)
    "plane": {"route": "simple", "phrases": ["a wall", "a floor", "a ceiling", "a door", "a flat panel", "a wooden board", "a sign", "a metal plate"]},
    "box": {"route": "simple", "phrases": ["a cardboard box", "a carton", "a stack of boxes", "a plain box", "a pallet", "a block", "a beam"]},
    "cylinder": {"route": "simple", "phrases": ["a pipe", "a pole", "a column", "a drum"]},
    "open frame": {"route": "simple", "phrases": ["a shelf", "a storage rack", "a shelving unit"]},
    "complex": {"route": "complex", "phrases": ["a machine", "a cart", "a tool", "equipment", "furniture", "a chair", "a table",
                                                "a machine guard", "a fence", "an irregular object"]},
    "generic simple": {"route": "simple", "phrases": ["a simple geometric object such as a box, a panel, a pipe or a shelf"]},
    "generic complex": {"route": "complex", "phrases": ["a complex irregular object such as a machine, a tool, a cart or furniture"]},
}
JEV_STATE = ("This image is cropped from a video walk-through of a workplace (a workshop, warehouse or store). A white outline "
             "with a black edge, tagged 1, marks one region.")  # X13 naming's state
Q2 = ("Can the outlined object be shown well as one simple geometric shape, or does it need a detailed 3D model?",
      ["one simple shape: a flat panel, board, wall or door; a plain box or carton; a pipe or pole; or a shelf or rack frame",
       "a detailed 3D model: a machine, cart, tool, equipment, furniture, a shaped guard or fence, or any irregular object"])
Q5 = ("Which shape best represents the outlined object in a 3D model of the place?",
      ["a flat panel, board, sign, wall or door", "a plain box, carton, block or stack of boxes", "a cylinder: a pipe, pole, drum or roll",
       "an open frame: a shelf or rack", "none of these simple shapes: a machine, cart, tool, equipment, furniture or an irregular object"])
Q5_KIND = ["plane", "box", "cylinder", "open frame", None]
QYN = ("Is the outlined object a simple geometric shape, such as a panel, a box, a pipe or a shelf frame?", ["yes", "no"])


def questions(out):
    """OUT/questions.json (every card: Q2 in both option orders and Q5; the labelled sample also the yes / no form) and
    OUT/prompts.json. No labels in either."""
    meta = json.loads((out / "meta.json").read_text())
    sample = {k for s in json.loads((out / "sample.json").read_text())["sites"].values() for k in s["keys"]}
    qs = []
    for m in meta:
        forms = {"q2a": (Q2[0], Q2[1]), "q2b": (Q2[0], Q2[1][::-1]), "q5": Q5}
        if m["key"] in sample:
            forms["qyn"] = QYN
        qs += [{"key": f"{m['key']}|{f}", "card": m["key"], "form": f, "state": JEV_STATE, "question": q, "options": o} for f, (q, o) in forms.items()]
    (out / "questions.json").write_text(json.dumps(qs))
    (out / "prompts.json").write_text(json.dumps(PROMPTS, indent=1))
    print(len(qs), "questions")


def topup(out, extra, seed=1):
    """A second stratified draw (the same rule, seed 1) from each video's unsampled cards, so that >= 300 labels are clear
    after the first sheets' unclear ones; sample.json's strata counts become the union's (weights N / n stay per stratum)."""
    meta = json.loads((out / "meta.json").read_text())
    doc = json.loads((out / "sample.json").read_text())
    for site, n in extra.items():
        s = doc["sites"][site]
        rest = [m for m in meta if m["site"] == site and m["key"] not in set(s["keys"])]
        chosen, _ = stratified([m["key"] for m in rest], [m["cls"] or "~" + cards.norm(m["proposed"] or "unknown") for m in rest], n, seed)
        s["topup"] = chosen
        for m in rest:
            if m["key"] in chosen:
                st = m["cls"] or "~" + cards.norm(m["proposed"] or "unknown")
                s["strata"][st]["n"] += 1
    doc["topup_rule"] = "after the first 300 labels: a second draw by the same rule (seed 1) from the unsampled cards: " + json.dumps(extra)
    (out / "sample.json").write_text(json.dumps(doc, indent=1))


# ---------------------------------------------------------------- scoring (CPU; labels stay here)

SITES = list(WARM)
RECGEN_S = 8.21  # X7 (runs/fx-x7-object-models-001): recgen.generate median s per object
PHRASES = [(g, p) for g, d in PROMPTS.items() for p in d["phrases"]]  # the container's text order
SETS = {"classes": [k for k, (g, _) in enumerate(PHRASES) if not g.startswith("generic")],
        "generic": [k for k, (g, _) in enumerate(PHRASES) if g.startswith("generic")]}


def sigmoid(z):
    return 1 / (1 + np.exp(-np.asarray(z, float)))


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def ece(p, y, bins=10):
    """Expected calibration error of P(complex) against the labels (10 equal-width bins)."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    e = 0.
    for b in range(bins):
        m = (p >= b / bins) & ((p < (b + 1) / bins) | (b == bins - 1))
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def platt(z, y):
    """(a, b) of sigmoid(a z + b) by the training NLL: a temperature and a bias (the only numbers fitted on a decider)."""
    from scipy.optimize import minimize
    z, y = np.asarray(z, float), np.asarray(y, float)

    def nll(ab):
        q = np.clip(sigmoid(ab[0] * z + ab[1]), 1e-9, 1 - 1e-9)
        return -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))
    return minimize(nll, [1., 0.], method="Nelder-Mead").x


def fit_threshold(s, y):
    """The cut on s (complex above it) with the best training accuracy; ties -> the middle of the best cuts."""
    s, y = np.asarray(s, float), np.asarray(y, int)
    if not len(s):
        return np.inf
    u = np.unique(s)
    cand = np.r_[u[0] - 1e-9, (u[1:] + u[:-1]) / 2, u[-1] + 1e-9]
    acc = np.array([((s > c) == y).mean() for c in cand])
    best = cand[acc >= acc.max() - 1e-12]
    return float(best[len(best) // 2])


class Data:
    """Cards, labels (1 complex, 0 simple, -2 unclear, -1 not labelled), stratum weights, and every router's raw score."""

    def __init__(self, out):
        self.meta = json.loads((out / "meta.json").read_text())
        self.n = len(self.meta)
        self.at = {m["key"]: i for i, m in enumerate(self.meta)}
        self.site = np.array([m["site"] for m in self.meta])
        self.stratum = np.array([m["cls"] or "~" + cards.norm(m["proposed"] or "unknown") for m in self.meta])
        self.y, self.prim, self.w = np.full(self.n, -1), [None] * self.n, np.zeros(self.n)
        for it in json.loads((out / "labels.json").read_text())["items"].values():
            i = self.at[it["key"]]
            self.y[i], self.prim[i] = {"complex": 1, "simple": 0, "unclear": -2}[it["label"]], it["primitive"]
        strata = json.loads((out / "sample.json").read_text())["sites"]
        for i in np.where(self.y != -1)[0]:
            s = strata[self.site[i]]["strata"][self.stratum[i]]
            self.w[i] = s["N"] / s["n"]  # Horvitz-Thompson: the sample is stratified (sqrt allocation), not proportional
        self.route = np.array([m["class_route"] or "" for m in self.meta])
        self.route_post_hoc = np.array([class_route(m["cls"], post_hoc=True) or "" for m in self.meta])
        self.rel = np.array([(m["geometry"] or {}).get("rel", np.inf) for m in self.meta])
        self.eligible = np.array([bool(m["sam3d"]) for m in self.meta])
        ans = json.loads((out / "answers.json").read_text())
        self.timing = ans
        for who, forms in (("jev", (("q2a", 1), ("q2b", 0), ("q5", 4), ("qyn", 1))), ("qwen", (("q2a", 1), ("q2b", 0), ("q5", 4)))):
            pr = ans[who]["probs"]
            d = {f: np.array([(pr.get(f"{m['key']}|{f}") or [np.nan] * 5)[k] for m in self.meta]) for f, k in forms}
            d["q2"] = (d["q2a"] + d["q2b"]) / 2  # both option orders: the position bias cancels
            setattr(self, who, d)
        q5 = np.array([(ans["jev"]["probs"].get(f"{m['key']}|q5") or [np.nan] * 5)[:4] for m in self.meta])
        self.jev_prim = np.array([Q5_KIND[int(np.argmax(r))] if np.isfinite(r).all() else None for r in q5], object)
        zs = ans["zero_shot"]
        self.zs = {tuple(k.split("|")): np.array(v) for k, v in zs["logit_complex"].items()}
        self.zs_prim = {tuple(k.split("|")): np.array(v) for k, v in zs["primitive"].items()}


def export(out, scratch):
    """The containers' raw outputs (SCRATCH answers.pkl, qwen.pkl) -> OUT/answers.json: option probabilities and timing, and per
    card the zero-shot logit of P(complex) (the phrases' softmax at the model's own scale) and shape group, so scoring needs no
    crops or embeddings (those stay in the scratchpad)."""
    ans, qw = pickle.loads((scratch / "answers.pkl").read_bytes()), pickle.loads((scratch / "qwen.pkl").read_bytes())
    meta = json.loads((out / "meta.json").read_text())
    E = ans["embed"]
    assert E["keys"] == [m["key"] for m in meta]
    groups, zs, prim = np.array([g for g, _ in PHRASES]), {}, {}
    for enc, T in E["text"].items():
        for var in ("plain", "masked", "som"):
            L = E["scale"][enc] * E["emb"][(enc, var)].astype(np.float32) @ T.astype(np.float32).T
            for name, cols in SETS.items():
                z = L[:, cols] - L[:, cols].max(1, keepdims=True)
                P = np.exp(z) / np.exp(z).sum(1, keepdims=True)
                zs[f"{enc}|{var}|{name}"] = np.round(logit(P[:, [PROMPTS[groups[c]]["route"] == "complex" for c in cols]].sum(1)), 4).tolist()
                if name == "classes":
                    shapes = ["plane", "box", "cylinder", "open frame"]
                    prim[f"{enc}|{var}"] = np.array(shapes)[np.argmax(np.stack([P[:, groups[cols] == g].sum(1) for g in shapes], 1), 1)].tolist()
    doc = {"jev": {"probs": ans["jev"]["probs"], **{k: v for k, v in ans["jev"].items() if k != "probs"}},
           "qwen": {"probs": qw["probs"], **{k: v for k, v in qw.items() if k != "probs"}},
           "embed": {"timing": E["timing"], "gpu": E["gpu"], "scale": E["scale"]}, "wall_s": ans["wall_s"],
           "zero_shot": {"phrases": PHRASES, "logit_complex": zs, "primitive": prim}}
    (out / "answers.json").write_text(json.dumps(doc))


def cell_rate(cells, y, tr, te):
    """P(complex) of a rule's cell: the training videos' complex share in it (Laplace-smoothed), 0.5 for an unseen cell."""
    out = np.full(te.sum(), .5)
    for c in np.unique(cells[te]):
        m = tr & (cells == c)
        out[cells[te] == c] = (y[m].sum() + 1) / (m.sum() + 2)
    return out


# each router: (D, tr, te) -> (complex? on te, P(complex) on te, tier on te, fitted numbers); tr = labelled clear training cards

def r_type(D, tr, te):
    none = D.route == ""
    default = bool(D.y[tr & none].mean() > .5) if (tr & none).any() else True
    pred = np.where(D.route == "complex", True, np.where(D.route == "simple", False, default))
    return pred[te], cell_rate(D.route, D.y, tr, te), np.full(te.sum(), "rules"), {"no_class_goes": "complex" if default else "simple"}


def r_geom(D, tr, te):
    t = fit_threshold(D.rel[tr], D.y[tr])
    return (D.rel > t)[te], cell_rate(np.where(D.rel > t, "high", "low"), D.y, tr, te), np.full(te.sum(), "rules"), {"rel_cut": t}


def r_type_geom(D, tr, te, veto=False):
    none, simple = D.route == "", D.route == "simple"
    t = fit_threshold(D.rel[tr & none], D.y[tr & none])
    t_hi = fit_threshold(D.rel[tr & simple], D.y[tr & simple]) if veto else np.inf
    pred = (D.route == "complex") | (none & (D.rel > t)) | (simple & (D.rel > t_hi))
    cells = np.where(none, np.where(D.rel > t, "none-high", "none-low"), np.where(simple & (D.rel > t_hi), "simple-high", D.route))
    return pred[te], cell_rate(cells, D.y, tr, te), np.full(te.sum(), "rules"), {"rel_cut_no_class": t, "rel_cut_simple_class": t_hi}


def calibrated(z, y, tr, te, fit=True):
    tr = tr & np.isfinite(z)  # the yes / no form was asked on the first sample only
    a, b = platt(z[tr], y[tr]) if fit else (1., 0.)
    return sigmoid(a * z[te] + b), {"platt_a": round(float(a), 4), "platt_b": round(float(b), 4)}


def r_zs(D, tr, te, key=None):
    """Zero-shot P(complex) (the phrases' softmax at the model's scale, summed over the complex phrases), Platt-calibrated on
    the training videos; the (encoder, crop, prompt set) picked on the training videos too unless given."""
    keys = [key] if key else sorted(D.zs)
    best = max(keys, key=lambda k: ((calibrated(D.zs[k], D.y, tr, tr)[0] > .5) == D.y[tr]).mean())
    p, prm = calibrated(D.zs[best], D.y, tr, te)
    return p > .5, p, np.full(te.sum(), "zero-shot"), {"key": list(best), **prm}


def r_jev(D, tr, te, form="q2", fit=True):
    p, prm = calibrated(logit(D.jev[form]), D.y, tr, te, fit)
    return p > .5, p, np.full(te.sum(), "jev"), {"form": form, **prm}


def r_qwen(D, tr, te, form="q2", fit=True):
    p, prm = calibrated(logit(D.qwen[form]), D.y, tr, te, fit)
    return p > .5, p, np.full(te.sum(), "vlm"), {"form": form, **prm}


def r_band(D, tr, te, first=r_jev, width=.2, **kw):
    """`first` decides unless its P(complex) is within `width` of 0.5; there the VLM (Qwen3-VL-8B, same question) decides."""
    pred, p, tier, prm = first(D, tr, te, **kw)
    q, qp, _, qprm = r_qwen(D, tr, te, form=kw.get("form", "q2"))
    band = (tier != "rules") & (np.abs(p - .5) < width)
    pred, p, tier = np.where(band, q, pred), np.where(band, qp, p), np.where(band, "vlm", tier)
    return pred, p, tier, {"first": prm, "vlm": qprm, "band": [.5 - width, .5 + width]}


def r_cascade(D, tr, te, second=r_jev, gate="class", post_hoc=False, **kw):
    """Tier 1 the rules where the class decides (gate 'class': any class prior; 'class+geom': a complex class, or a simple one
    whose points fit a primitive (relative residual <= the cut fitted on the simple classes)); tier 2 `second`, calibrated
    on the training cards that reach it; a VLM never."""
    route = D.route_post_hoc if post_hoc else D.route
    simple = route == "simple"
    t_hi = fit_threshold(D.rel[tr & simple], D.y[tr & simple]) if gate == "class+geom" else np.inf
    t1 = (route == "complex") | (simple & (D.rel <= t_hi))
    pred, p, tier = np.zeros(te.sum(), bool), np.zeros(te.sum()), np.full(te.sum(), "rules", object)
    c1 = (route == "complex")[te]
    pred[t1[te]] = c1[t1[te]]
    p[t1[te]] = cell_rate(np.where(t1, route, "-"), D.y, tr & t1, te & t1)
    pred[~t1[te]], p[~t1[te]], tier[~t1[te]], prm = second(D, tr & ~t1, te & ~t1, **kw)
    return pred, p, tier, {"gate": gate, "rel_cut_simple_class": t_hi, "tier2": prm}


def folds(D, router, **kw):
    """Held out by video: fitted on the other two videos' labels, applied to every card of the third."""
    pred, p, tier, prm = np.zeros(D.n, bool), np.full(D.n, np.nan), np.full(D.n, "", object), {}
    for v in SITES:
        tr, te = (D.site != v) & (D.y >= 0), D.site == v
        a, b, c, d = router(D, tr, te, **kw)
        pred[te], p[te], tier[te], prm[v] = a, b, c, d
    return pred, p, tier, prm


def metrics(D, pred, p, tier, m):
    y, pr, w = D.y[m], pred[m], D.w[m]
    tp, fp, fn = int((pr & (y == 1)).sum()), int((pr & (y == 0)).sum()), int((~pr & (y == 1)).sum())
    out = {"n": int(m.sum()), "accuracy": round(float((pr == y).mean()), 3), "accuracy_weighted": round(float((w * (pr == y)).sum() / w.sum()), 3),
           "precision_complex": round(tp / max(tp + fp, 1), 3), "recall_complex": round(tp / max(tp + fn, 1), 3),
           "missed_complex": fn, "false_complex": fp, "ece": round(ece(p[m], y), 3)}
    out["tiers"] = {t: round(float((tier[m] == t).mean()), 3) for t in sorted(set(tier[m]))}
    band = (tier[m] != "rules") & (np.abs(p[m] - .5) < .2)  # where a last-resort VLM would be asked (P(complex) 0.3-0.7)
    out["vlm_band_share"] = round(float(band.mean()), 3)
    out["accuracy_outside_vlm_band"] = round(float((pr == y)[~band].mean()), 3) if (~band).any() else None
    return out


def load(D, pred, v):
    """RecGen load on every card of video v: routed there, and those the SAM 3D / RecGen view gate would take at all."""
    c = D.site == v
    lab = c & (D.y >= 0)
    return {"cards": int(c.sum()), "routed_complex": int((pred & c).sum()), "routed_complex_and_well_observed": int((pred & c & D.eligible).sum()),
            "recgen_gpu_s_at_x7_median": round(float((pred & c).sum()) * RECGEN_S, 1),
            "labels_estimate_complex": round(float((D.w * (D.y == 1))[lab].sum()), 1),
            "labels_estimate_unclear": round(float((D.w * (D.y == -2))[c].sum()), 1)}


def score(out):
    import time
    D = Data(out)
    lab = D.y >= 0
    routers = {"a0 rules: class only": (r_type, {}), "a1 rules: geometry only": (r_geom, {}),
               "a2 rules: class, else geometry": (r_type_geom, {}), "a3 rules: class, else geometry, simple class vetoed by geometry": (r_type_geom, {"veto": True}),
               "b zero-shot (encoder, crop, prompts picked on training videos)": (r_zs, {})}
    for enc in ("pe-core-l", "siglip2-so400m"):
        routers[f"b zero-shot {enc} masked classes"] = (r_zs, {"key": (enc, "masked", "classes")})
    for f in ("q2", "q2a", "q2b", "q5", "qyn"):
        routers[f"c jev {f} raw"] = (r_jev, {"form": f, "fit": False})
        routers[f"c jev {f} calibrated"] = (r_jev, {"form": f})
    routers["d cascade: class -> jev q2"] = (r_cascade, {"form": "q2"})
    routers["d cascade: class+geometry -> jev q2"] = (r_cascade, {"gate": "class+geom", "form": "q2"})
    routers["d cascade: class -> jev q5"] = (r_cascade, {"form": "q5"})
    routers["d cascade: class -> zero-shot"] = (r_cascade, {"second": r_zs})
    routers["c' vlm qwen3-vl-8b q2 calibrated (reference: a VLM for every card)"] = (r_qwen, {"form": "q2"})
    routers["c' vlm qwen3-vl-8b q5 raw (reference)"] = (r_qwen, {"form": "q5", "fit": False})
    routers["d jev q2 calibrated -> vlm on jev's band (P 0.3-0.7)"] = (r_band, {"form": "q2"})
    routers["d cascade: class -> jev q2 -> vlm on jev's band"] = (r_band, {"first": r_cascade, "form": "q2"})
    routers["d cascade: class -> jev q2 (POST HOC class table)"] = (r_cascade, {"form": "q2", "post_hoc": True})
    routers["d cascade: class -> jev q5 raw (POST HOC class table)"] = (r_cascade, {"form": "q5", "fit": False, "post_hoc": True})
    res, outs = {"routers": {}, "baseline_all_simple": {"accuracy": round(float((D.y[lab] == 0).mean()), 3),
                                                         "accuracy_weighted": round(float((D.w * (D.y == 0))[lab].sum() / D.w[lab].sum()), 3)}}, {}
    for name, (fn, kw) in routers.items():
        m = lab & np.isfinite(D.jev["qyn"]) if "qyn" in name else lab
        pred, p, tier, prm = folds(D, fn, **kw)
        outs[name] = (pred, p)
        row = {"pooled": metrics(D, pred, p, tier, m), "per_video": {v: metrics(D, pred, p, tier, m & (D.site == v)) for v in SITES},
               "well_observed": metrics(D, pred, p, tier, m & D.eligible), "fitted": prm}
        if "qyn" not in name:
            row["load"] = {v: load(D, pred, v) for v in SITES}
        res["routers"][name] = row
    for name in ("c jev q2 calibrated", "c jev q5 raw"):
        res.setdefault("operating_curve", {})[name] = curve(D, outs[name][1])
    ref = outs["c jev q2 calibrated"][0]
    res["paired_bootstrap_accuracy_vs_jev_q2_calibrated"] = {
        k: bootstrap(D, ref, outs[k][0]) for k in ("a0 rules: class only", "a2 rules: class, else geometry",
                                                   "b zero-shot (encoder, crop, prompts picked on training videos)", "c jev q5 raw",
                                                   "d cascade: class -> jev q2", "d cascade: class -> jev q2 (POST HOC class table)",
                                                   "c' vlm qwen3-vl-8b q2 calibrated (reference: a VLM for every card)",
                                                   "d jev q2 calibrated -> vlm on jev's band (P 0.3-0.7)")}
    t = time.perf_counter()
    for _ in range(100):
        folds(D, r_type_geom)
    res["rules_s_per_card_incl_fitting"] = (time.perf_counter() - t) / 100 / D.n
    res["primitive"] = primitives(D)
    res["latency"] = latency(D)
    res["labels"] = {v: {"clear": int(((D.site == v) & lab).sum()), "complex": int(((D.site == v) & (D.y == 1)).sum()),
                         "simple": int(((D.site == v) & (D.y == 0)).sum()), "unclear": int(((D.site == v) & (D.y == -2)).sum())} for v in SITES}
    return D, res


def curve(D, p, cuts=(.2, .3, .4, .5, .6, .7)):
    """A calibrated P(complex) cut t is the cost ratio: route to the generator when P > t, i.e. t = cost(false complex) /
    (cost(false complex) + cost(missed complex)). Per cut: precision / recall of COMPLEX and the RecGen load per video."""
    lab = D.y >= 0
    out = {}
    for t in cuts:
        pr = p > t
        tp, fp, fn = int((pr & lab & (D.y == 1)).sum()), int((pr & lab & (D.y == 0)).sum()), int((~pr & lab & (D.y == 1)).sum())
        out[str(t)] = {"accuracy": round(float((pr[lab] == D.y[lab]).mean()), 3), "precision_complex": round(tp / max(tp + fp, 1), 3),
                       "recall_complex": round(tp / max(tp + fn, 1), 3),
                       "routed_complex": {v: int((pr & (D.site == v)).sum()) for v in SITES},
                       "routed_complex_well_observed": {v: int((pr & (D.site == v) & D.eligible).sum()) for v in SITES}}
    return out


def bootstrap(D, a, b, n=2000, seed=0):
    """Paired bootstrap over the labelled cards (resampled within each video): accuracy(b) - accuracy(a), 95 % interval."""
    rng = np.random.default_rng(seed)
    idx = {v: np.where((D.y >= 0) & (D.site == v))[0] for v in SITES}
    ok_a, ok_b = (a == (D.y == 1)), (b == (D.y == 1))
    d = []
    for _ in range(n):
        s = np.concatenate([rng.choice(ix, len(ix)) for ix in idx.values()])
        d.append(ok_b[s].mean() - ok_a[s].mean())
    lab = D.y >= 0
    return {"difference": round(float(ok_b[lab].mean() - ok_a[lab].mean()), 3), "ci95": [round(float(q), 3) for q in np.percentile(d, [2.5, 97.5])]}


def primitives(D):
    """Which primitive, on the cards labelled SIMPLE: the card's display model (display_model.choose by its class and the fits),
    Jev-Omni's Q5 argmax over the four shapes, the zero-shot group argmax."""
    m = np.where(D.y == 0)[0]
    truth = np.array([D.prim[i] for i in m])
    out = {"n": len(m), "label_counts": {k: int((truth == k).sum()) for k in sorted(set(truth))}}
    for name, got in (("display model (cards)", np.array([D.meta[i]["model_kind"] for i in m])), ("jev q5", D.jev_prim[m]),
                      *((f"zero-shot {k[0]} {k[1]}", v[m]) for k, v in D.zs_prim.items() if k[1] == "masked")):
        conf = {}
        for a, b in zip(truth, got):
            conf[f"{a} -> {b}"] = conf.get(f"{a} -> {b}", 0) + 1
        out[name] = {"accuracy": round(float((got == truth).mean()), 3),
                     "per_video": {v: round(float((got == truth)[D.site[m] == v].mean()), 3) for v in SITES},
                     "confusions": dict(sorted(((k, c) for k, c in conf.items() if k.split(" -> ")[0] != k.split(" -> ")[1]), key=lambda kv: -kv[1]))}
    return out


def latency(D):
    j, e = D.timing["jev"], D.timing["embed"]["timing"]
    q = D.timing["qwen"]
    n = sum(t["n"] for t in j["trips"])
    views = np.mean([len(m["views"]) for m in D.meta])
    return {"jev": {"questions": n, "requests": len(j["trips"]), "gpu": j["gpu"], "compute_s_per_question_batched_8": round(sum(t["compute_s"] for t in j["trips"]) / n, 4),
                    "round_trip_s_per_question_in_64_question_requests": round(sum(t["rtt_s"] for t in j["trips"]) / n, 4),
                    "hop_s_per_request_median": round(float(np.median([t["hop_s"] for t in j["trips"]])), 3),
                    "request_mb_median": round(float(np.median([t["bytes"] for t in j["trips"]])) / 1e6, 2),
                    "single_question_round_trip_s_median": round(float(np.median([t["rtt_s"] for t in j["single"]])), 3),
                    "single_question_compute_s_median": round(float(np.median([t["compute_s"] for t in j["single"]])), 3),
                    "cold_start_s": j["cold_s"], "model_load_s": j["load_s"], "peak_gib_torch_reserved": j["peak_gib"], "all_questions_s": j["total_s"]},
            "qwen_core_sidecar": {"s_per_question_16_parallel_in_container": q["s_per_question_16_parallel"], "single_s_median":
                                  round(float(np.median(q["single_s"])), 4), "load_s": q["load_s"], "gpu": q["gpu"], "questions": len(q["probs"])},
            "embed": {"views_per_card": round(float(views), 2), "gpu": D.timing["embed"]["gpu"],
                      **{enc: {"ms_per_crop": {v: e[enc][v]["ms_per_crop"] for v in ("plain", "masked", "som")}, "load_s": e[enc]["load_s"],
                               "ms_per_card_masked": round(e[enc]["masked"]["ms_per_crop"] * views, 2), "text_s_once": e[enc]["text_s"]}
                         for enc in ("pe-core-l", "siglip2-so400m")}}}


# ---------------------------------------------------------------- the recommended rule (results.md)

P_COMPLEX_CUT = .5  # the cost knob: generate when P(complex) > cut = cost(wasted generation) / (cost(wasted) + cost(ugly primitive))


def route(card, q5=None, cut=P_COMPLEX_CUT):
    """-> (display model, why): 'primitive', 'generated' or 'ask jev' (then call again with Jev-Omni's Q5 probabilities: raw,
    one question on the card's outlined best view). 1) A card no generator takes (display_model.well_observed fails) keeps its
    primitive, no question. 2) A class that fixes the shape decides (FIXED_SHAPE_POST_HOC: machines, tools, carts, furniture,
    cables -> generated; boxes, pallets, shelves, racks, signs, pipes, boards -> primitive). 3) Jev-Omni Q5 for the rest:
    P(none of the simple shapes) > cut -> generated. No VLM."""
    score, why = display_model.well_observed(card)
    if score is None:
        return "primitive", f"no generator takes it ({why})"
    cls = card_class(card)
    r = class_route(cls, post_hoc=True)
    if r:
        return ("generated" if r == "complex" else "primitive"), f"the class fixes the shape ({cls})"
    if q5 is None:
        return "ask jev", f"no class fixes the shape ({cls}): Q5 on the outlined best view"
    return ("generated" if q5[4] > cut else "primitive"), f"jev q5: P(none of the simple shapes) {q5[4]:.2f} vs {cut}"


# ---------------------------------------------------------------- results.json / results.md

ROWS = ["a0 rules: class only", "a1 rules: geometry only", "a2 rules: class, else geometry", "b zero-shot (encoder, crop, prompts picked on training videos)",
        "b zero-shot pe-core-l masked classes", "b zero-shot siglip2-so400m masked classes", "c jev q2 raw", "c jev q2 calibrated", "c jev q5 raw",
        "c jev q5 calibrated", "c jev qyn calibrated", "c' vlm qwen3-vl-8b q2 calibrated (reference: a VLM for every card)",
        "c' vlm qwen3-vl-8b q5 raw (reference)", "d cascade: class -> jev q2", "d cascade: class -> jev q5", "d cascade: class -> zero-shot",
        "d jev q2 calibrated -> vlm on jev's band (P 0.3-0.7)", "d cascade: class -> jev q2 -> vlm on jev's band",
        "d cascade: class -> jev q2 (POST HOC class table)", "d cascade: class -> jev q5 raw (POST HOC class table)"]
REC, BEST_SINGLE = "d cascade: class -> jev q5 raw (POST HOC class table)", "c jev q5 raw"
BEST_RULES, BEST_ZS, QWEN_ROW = "a2 rules: class, else geometry", "b zero-shot (encoder, crop, prompts picked on training videos)", "c' vlm qwen3-vl-8b q5 raw (reference)"
RES = {"jev": (1, 4, 48), "embed": (1, 8, 64), "caller": (0, 2, 8), "qwen": (1, 8, 64)}  # (A100s, cores, GiB) per container
APPS = {"ap-Xi5OzB9alMGmLdwgWXcPwn jev + embed smoke": (76, ("jev", "embed", "caller")),
        "ap-1ZRbWyiSU9hhigfzyWTiuw jev + embed": (604, ("jev", "embed", "caller")),
        "ap-HuRpnB4gBtMvnT7ygV1PUw qwen smoke": (109, ("qwen",)), "ap-4JDeG9ZLrs5xkioIBW2Va9 qwen": (221, ("qwen",))}  # modal app list: wall s


def spend():
    from fast_report.instrument import PRICE
    usd = {a: round(w * sum(g * PRICE["A100-80GB"] + c * PRICE["cpu_core"] + m * PRICE["gib"] for g, c, m in (RES[k] for k in ks)), 3)
           for a, (w, ks) in APPS.items()}
    return {"apps": usd, "total_usd_upper_bound": round(sum(usd.values()), 2),
            "method": "fast_report.instrument.PRICE (Modal list prices) x each ephemeral app's wall time (modal app list) for every container of it"}


def per_object(name, r, lat):
    """Latency per object: Jev questions per card x the batched round trip, or the tier's own cost."""
    t, j = r["pooled"]["tiers"], lat["jev"]["round_trip_s_per_question_in_64_question_requests"]
    if name.startswith("a"):
        return "<1 us CPU"
    if name.startswith("b"):
        return f"{lat['embed']['pe-core-l']['ms_per_card_masked']:.0f} ms A100"
    if name.startswith("c'"):
        return f"{lat['qwen_core_sidecar']['s_per_question_16_parallel_in_container'] * (2 if 'q2' in name else 1):.3f} s in the core"
    per_q = 2 if ("q2" in name and "q2a" not in name and "q2b" not in name) else 1
    jq = per_q * t.get("jev", 0) + (per_q * t.get("vlm", 0) if "band" in name else 0)
    return f"{jq:.2f} q x {j:.3f} s = {jq * j:.3f} s" + (" + VLM" if "vlm" in t else "") + (" + 6 ms" if "zero-shot" in t else "")


CAVEATS = [
    "Labels are the agent's (Claude), by looking at blind sheets (the outlined view and a wider one, a code only), not ground truth. 27 "
    "of 340 were unclear (a region across objects, a sliver, a floor line) and are left out; 23 of them on ME340, whose cards are often "
    "parts or fragments of machines and benches.",
    "Load-bearing labelling calls: full pet-food bags and shrink-wrapped packs are boxes (SIMPLE); shelf boards / lips and signs are "
    "planes; workbenches are furniture (COMPLEX, the user's list). If the 23 labelled bags were COMPLEX instead, Jev Q5 alone would drop "
    "0.875 -> 0.834 (Walmart 0.885 -> 0.760), the pre-registered cascade (bag -> complex) would rise 0.815 -> 0.888 and the "
    "recommended one fall 0.904 -> 0.863: bags are a policy decision, so put 'bag' in the class list whichever way it goes.",
    "The recommended class list is POST HOC: the pre-registered list (commit 1b61ed1, before any label) minus 5 entries the labels "
    "showed to be shape-ambiguous (bag, tool box, tool tray, crate, display rack). Its 0.904 is optimistic; the held-out numbers are "
    "0.815 for the pre-registered list and 0.875 for Jev alone. On the cards the post-hoc list decides, the class is right 0.975 vs "
    "Jev 0.920 (ME340 0.948 vs 0.793: Jev calls workbenches, hand tools and machine parts simple). Only 15 of the list's classes had "
    "labelled cards here (box, stacked boxes, pallet, shelf, rack, sign, spill, pipe, metal sheet, wooden board; machine, hand tool, power "
    "tool, cable, workbench); the other entries are the user's definition, untested.",
    "Per video n is 102-107 labelled; a difference under ~5 points is under ~5 cards; the bootstrap intervals are above.",
    "Jev-Omni's raw Q2 answers lean hard to 'a detailed 3D model' and move with the option order (raw 0.744 vs 0.591 by order): Q2 needs "
    "its two-number Platt calibration, whose intercept moved from -2.6 to -4.2 with the training videos' mix. Q5's raw probabilities "
    "need none (ECE 0.075), hence Q5 in the rule. The yes / no form ('a simple geometric shape?') is unusable (Jev says no to almost all).",
    "RecGen load is dominated by Walmart's wall of shoes: ~300 COMPLEX cards, ~100 of them well-observed -> ~100 generations per video "
    "(~15 min of one GPU at X7's 8.2 s median). A per-video cap or reuse of one generated model across same-type cards (X9) is needed "
    "whatever the router.",
    "The rule routes only well-observed cards (display_model.well_observed, the generators' own view gate): 352 of 1598 cards. If a "
    "single-view generator (SAM 3D from one image, TRELLIS) is adopted, drop the gate: Jev then answers every non-class card "
    "(~0.1 s each batched).",
    "Which primitive is a separate, weaker spot: the cards' display model agrees with the SIMPLE label 0.71 (planes drawn as boxes, boxes as "
    "cylinders), PE-Core-L zero-shot's shape group 0.82, Jev's Q5 argmax 0.63. Many plane / box disagreements are thin boards and lips, "
    "where a thin box and a slab look alike.",
    "Data: the r4-models-bench-001 warm calls (the only r4 runs with primitive fits); no r4b-* run existed; r4-physical-results has no fits.",
    "The VLM tier was measured, not assumed: the core's own Qwen3-VL-8B (same image and questions, letter log-probs) is worse than Jev "
    "alone (0.802-0.837) and deciding Jev's uncertain band (P 0.3-0.7, 13 % of cards) with it changes nothing (0.872 vs 0.875).",
    "Jev's one-question round trip (0.49 s) is mostly the hop: batch each cards version's questions (0.10 s a question in 64-question "
    "requests).",
]


def curve_rec(D, cuts=(.2, .3, .4, .5, .6, .7)):
    """The recommended cascade's Jev cut (the class tier fixed): accuracy, COMPLEX precision / recall, load per video."""
    t1, cp = D.route_post_hoc != "", D.route_post_hoc == "complex"
    return curve(D, np.where(t1, cp.astype(float), D.jev["q5"]), cuts)


def report(out, D, res):
    lat, R = res["latency"], res["routers"]
    res["spend_usd"] = spend()
    res["operating_curve"]["recommended cascade (the Jev cut)"] = cur = curve_rec(D)
    res["recommended"] = {"router": REC, "rule": route.__doc__.strip(), "cut": P_COMPLEX_CUT, "fixed_shape_simple": sorted(FIXED_SHAPE_POST_HOC["simple"]),
                          "fixed_shape_complex": {"families": sorted(COMPLEX_FAMILIES), "classes": sorted(COMPLEX_CLASSES - FIXED_SHAPE_POST_HOC["drop"])},
                          "to_jev": "every other class and cards with no class", "question": {"state": JEV_STATE, "question": Q5[0], "options": Q5[1]},
                          "caveats": CAVEATS}
    (out / "results.json").write_text(json.dumps(res, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x)))
    f = lambda x: f"{x:.3f}"  # noqa: E731
    j, q, e = lat["jev"], lat["qwen_core_sidecar"], lat["embed"]
    rt = j["round_trip_s_per_question_in_64_question_requests"]
    wo = {v: int(((D.site == v) & D.eligible).sum()) for v in SITES}
    asked = {v: int(((D.site == v) & D.eligible & (D.route_post_hoc == "")).sum()) for v in SITES}
    rec, one, c5 = R[REC], R[BEST_SINGLE], cur[str(P_COMPLEX_CUT)]
    per = lambda d: " / ".join(str(d[v]) for v in SITES)  # noqa: E731
    L = ["# route/jev: a primitive or a generated model, per object card", "",
         "Question: can the display model be routed per object (SIMPLE: a primitive; COMPLEX: RecGen / SAM 3D) by rules, zero-shot "
         "embeddings, Jev-Omni or a cascade, accurately and cheaply? Code: scripts/route_jev.py, modal_apps/route_jev.py (branch route/jev). "
         "Labels: the agent's, by looking at blind sheets; not ground truth. Every threshold / temperature is fitted on two videos and scored "
         "on the third (rotated); nothing is trained.", "",
         f"**Answer.** Jev-Omni can route: alone it is the best single router held out ({f(one['pooled']['accuracy'])}, COMPLEX precision "
         f"{one['pooled']['precision_complex']:.2f} / recall {one['pooled']['recall_complex']:.2f}, ECE {f(one['pooled']['ece'])}, no fitted number) against "
         f"{f(R[BEST_RULES]['pooled']['accuracy'])} for the best rules, {f(R[BEST_ZS]['pooled']['accuracy'])} for zero-shot embeddings and "
         f"{f(R[QWEN_ROW]['pooled']['accuracy'])} for the core's Qwen3-VL-8B. Recommended: the cascade, a class decides where it fixes the shape "
         f"and Jev-Omni Q5 decides the rest, on the well-observed cards only, no VLM: {f(rec['pooled']['accuracy'])} (post-hoc class list, see the "
         f"caveats), Jev asked about {per(asked)} cards per video ({' / '.join(f'{asked[v] * rt:.1f}' for v in SITES)} s of the service), "
         f"{per(c5['routed_complex_well_observed'])} RecGen calls per video (ME340 / Sam's Club / Walmart). Spend {res['spend_usd']['total_usd_upper_bound']} USD.", "",
         "## Labels", "", "| video | cards | labelled clear | complex | simple | unclear (left out) | labels' estimate of complex cards (weighted) |",
         "|---|---|---|---|---|---|---|"]
    for v in SITES:
        lb, ld = res["labels"][v], one["load"][v]
        L.append(f"| {v} | {ld['cards']} | {lb['clear']} | {lb['complex']} | {lb['simple']} | {lb['unclear']} | {ld['labels_estimate_complex']:.0f} |")
    pc = res["primitive"]["label_counts"]
    L += ["", "Sample: 100 cards per video stratified by class (sqrt allocation, seed 0) plus a top-up draw (seed 1) of 30 / 5 / 5 for the "
              f"unclear ones; weights N / n per stratum for the per-card numbers. Primitive labels on the SIMPLE ones: {', '.join(f'{k} {n}' for k, n in pc.items())}. "
              f"Baseline 'every card a primitive': {f(res['baseline_all_simple']['accuracy'])} (weighted {f(res['baseline_all_simple']['accuracy_weighted'])}).", "",
          "## Comparison (held out by video, pooled over the three test videos)", "",
          "| router | acc | acc per card (weighted) | COMPLEX precision / recall | missed / false complex | ECE | tiers | acc ME340 / Sam's / Walmart | "
          "routed to RecGen, all cards ME340 / Sam's / Walmart | of those well-observed | latency per object |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name in ROWS:
        r, ld = R[name], R[name].get("load")
        p = r["pooled"]
        L.append(f"| {'**' + name + '**' if name == REC else name} | {f(p['accuracy'])} | {f(p['accuracy_weighted'])} | {p['precision_complex']:.2f} / "
                 f"{p['recall_complex']:.2f} | {p['missed_complex']} / {p['false_complex']} | {f(p['ece'])} | "
                 f"{', '.join(f'{k} {v:.0%}' for k, v in p['tiers'].items())} | {' / '.join(f(r['per_video'][v]['accuracy']) for v in SITES)} | "
                 f"{' / '.join(str(ld[v]['routed_complex']) for v in SITES) if ld else '-'} | "
                 f"{' / '.join(str(ld[v]['routed_complex_and_well_observed']) for v in SITES) if ld else '-'} | {per_object(name, r, lat)} |")
    L += ["", "Rows: a = rules (the pre-registered class prior over cards.TAXONOMY; geometry = the best primitive's fit residual over the longest "
              "side, cut fitted on the training videos); b = zero-shot text prompts (Platt on the training videos); c = Jev-Omni (Q2: simple shape or "
              "detailed model, both option orders averaged; Q5: which of four shapes or none; qyn: yes / no), raw or Platt-calibrated; c' = the "
              "core's Qwen3-VL-8B on the same image and questions; d = cascades. POST HOC rows use the class list revised after labelling.", "",
          "Paired bootstrap (resampled within each video), accuracy minus that of 'c jev q2 calibrated':", ""]
    L += [f"- {k}: {v['difference']:+.3f} (95 % {v['ci95'][0]:+.3f} to {v['ci95'][1]:+.3f})" for k, v in res["paired_bootstrap_accuracy_vs_jev_q2_calibrated"].items()]
    L += ["", "## Recommended rule", "", "```", route.__doc__.strip(), "",
          f"generated: families {', '.join(sorted(COMPLEX_FAMILIES))}; classes {', '.join(sorted(COMPLEX_CLASSES - FIXED_SHAPE_POST_HOC['drop']))}",
          f"primitive: {', '.join(sorted(FIXED_SHAPE_POST_HOC['simple']))}", "Jev-Omni (every other class, and cards with no class):",
          f"  state: {JEV_STATE}", f"  question: {Q5[0]}", *[f"  {'ABCDE'[k]}. {o}" for k, o in enumerate(Q5[1])],
          f"  generated when P(E) > {P_COMPLEX_CUT} (raw probabilities, no fitted number)", "```", "",
          f"Pooled {f(rec['pooled']['accuracy'])}, COMPLEX precision {rec['pooled']['precision_complex']:.2f} / recall {rec['pooled']['recall_complex']:.2f}, "
          f"per video {' / '.join(f(rec['per_video'][v]['accuracy']) for v in SITES)}; on the well-observed cards (the ones it routes) "
          f"{f(rec['well_observed']['accuracy'])} (n {rec['well_observed']['n']}). The class tier decides {rec['pooled']['tiers']['rules']:.0%} of the labelled "
          "cards. The cut is the cost knob (P > cut goes to the generator); its effect with the class tier fixed:", "",
          "| cut | acc | COMPLEX precision / recall | routed, all cards ME340 / Sam's / Walmart | of those well-observed (RecGen calls) |", "|---|---|---|---|---|"]
    for t, c in cur.items():
        L.append(f"| {t} | {f(c['accuracy'])} | {c['precision_complex']:.2f} / {c['recall_complex']:.2f} | {per(c['routed_complex'])} | "
                 f"{per(c['routed_complex_well_observed'])} |")
    L += ["", "## Latency", "",
          f"- Jev-Omni service, alone on one {j['gpu'].split(':')[1].split('(')[0].strip()}, called from a CPU container (the core's stand-in) "
          f"through Modal: {j['compute_s_per_question_batched_8']} s GPU per question (batches of 8), {rt} s per question round trip in 64-question requests "
          f"(hop {j['hop_s_per_request_median']} s a request of {j['request_mb_median']} MB); one question alone {j['single_question_round_trip_s_median']} s round "
          f"trip ({j['single_question_compute_s_median']} s of it compute). Cold start {j['cold_start_s']} s (load {j['model_load_s']} s), {j['peak_gib_torch_reserved']} GiB.",
          f"- The recommended rule asks Jev about {per(asked)} cards per video (well-observed {per(wo)}, all cards "
          f"{' / '.join(str(int((D.site == v).sum())) for v in SITES)}): {' / '.join(f'{asked[v] * rt:.1f}' for v in SITES)} s of the service per video.",
          f"- The VLM (the core's Qwen3-VL-8B sidecar alone on an A100): {q['s_per_question_16_parallel_in_container']} s per question at 16 parallel inside the core "
          f"(no hop), load {q['load_s']} s. Zero-shot: PE-Core-L {e['pe-core-l']['ms_per_crop']['masked']} ms, SigLIP 2 so400m "
          f"{e['siglip2-so400m']['ms_per_crop']['masked']} ms per crop on an A100. Rules: {res['rules_s_per_card_incl_fitting'] * 1e6:.2f} us per card.", "",
          "## RecGen load per video (the recommended rule, cut 0.5)", "",
          "| video | cards | well-observed | RecGen calls (routed and well-observed) | routed, any card | GPU s at X7's 8.2 s median | labels' estimate of complex cards |",
          "|---|---|---|---|---|---|---|"]
    for v in SITES:
        L.append(f"| {v} | {one['load'][v]['cards']} | {wo[v]} | {c5['routed_complex_well_observed'][v]} | {c5['routed_complex'][v]} | "
                 f"{c5['routed_complex_well_observed'][v] * RECGEN_S:.0f} | {one['load'][v]['labels_estimate_complex']:.0f} |")
    pr = res["primitive"]
    L += ["", "## Which primitive (the SIMPLE-labelled cards)", "", "| picker | agrees with the label | ME340 / Sam's / Walmart | top confusions (label -> picked) |",
          "|---|---|---|---|"]
    for k in ("display model (cards)", "jev q5", "zero-shot pe-core-l masked", "zero-shot siglip2-so400m masked"):
        L.append(f"| {k} | {f(pr[k]['accuracy'])} | {' / '.join(f(pr[k]['per_video'][v]) for v in SITES)} | "
                 f"{', '.join(f'{cc} {n}' for cc, n in list(pr[k]['confusions'].items())[:3])} |")
    L += ["", "## Spend", "", f"{res['spend_usd']['total_usd_upper_bound']} USD upper bound ({res['spend_usd']['method']}): "
          f"{', '.join(f'{k} {v}' for k, v in res['spend_usd']['apps'].items())}. No Gemini. The cost ledger was not edited.", "",
          "## Caveats and open issues", ""] + [f"- {cv}" for cv in CAVEATS]
    (out / "results.md").write_text("\n".join(L) + "\n")


# ---------------------------------------------------------------- blind sheets for the agent's labels

LABEL_GUIDE = {"S:plane": "a wall, floor, ceiling, door, plain panel / board / sign / plate, a flat patch (a slab as thick as it is)",
               "S:box": "a plain box / carton / block / beam / pallet / wrapped pack / stack of boxes",
               "S:cylinder": "a pole / pipe / column / drum / roll / plain can",
               "S:frame": "a shelf / rack frame (posts and levels)",
               "C": "a machine, cart, shaped guard / fence, equipment, tool, furniture (table, chair, bench), or anything irregular",
               "U": "unclear: the outline is no one thing (a fragment across objects, a shadow) or it cannot be seen"}


def sheets(out, scratch, per=15, cols=3, side=256, seed=1):
    """Per video, the sample in a seeded random order (no class blocks), each tile the outlined view (1.6x the box) and a wider
    view (4x) with only a code under it: no name, class or geometry (the labels stay blind to every router's inputs)."""
    import cv2
    data = pickle.loads((scratch / "dataset.pkl").read_bytes())
    sample = json.loads((out / "sample.json").read_text())["sites"]
    (out / "sheets").mkdir(parents=True, exist_ok=True)
    template, rng = {}, np.random.default_rng(seed)

    def fit(jpeg):
        im = cv2.imdecode(np.frombuffer(jpeg, np.uint8), 1)
        s = side / max(im.shape[:2])
        im = cv2.resize(im, (max(1, round(im.shape[1] * s)), max(1, round(im.shape[0] * s))), interpolation=cv2.INTER_AREA)
        return cv2.copyMakeBorder(im, 0, side - im.shape[0], 0, side - im.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255))

    for site, s in sample.items():
        keys = [s["keys"][i] for i in rng.permutation(len(s["keys"]))] + s.get("topup", [])
        tiles = []
        for k, key in enumerate(keys):
            if k < len(s["keys"]) and s.get("topup"):
                continue  # the first sheets are labelled already: only the top-up's are drawn (codes continue)
            code = f"{site[0].upper()}{k:03d}"
            template[code] = {"key": key, "label": None, "note": None}
            t = np.hstack([fit(data["crops"][f"{key}|som"]), np.full((side, 6, 3), 255, np.uint8), fit(data["crops"][f"{key}|context"])])
            t = np.vstack([t, np.full((26, t.shape[1], 3), 255, np.uint8)])
            cv2.putText(t, code, (4, side + 19), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 0), 2, cv2.LINE_AA)
            tiles.append(np.pad(t, ((0, 8), (0, 10), (0, 0)), constant_values=255))
        for p in range(0, len(tiles), per):
            t = tiles[p:p + per]
            t += [np.full_like(t[0], 255)] * (-len(t) % cols)
            cv2.imwrite(str(out / "sheets" / f"{site}-{'topup-' if s.get('topup') else ''}{p // per:02d}.jpg"), np.vstack([np.hstack(t[q:q + cols]) for q in range(0, len(t), cols)]),
                        [cv2.IMWRITE_JPEG_QUALITY, 85])
    if any(s.get("topup") for s in sample.values()):
        template = {**json.loads((out / "labels-template.json").read_text())["items"], **template}
    (out / "labels-template.json").write_text(json.dumps({"labelled_by": "agent, by looking at the blind sheets (not ground truth)",
                                                          "guide": LABEL_GUIDE, "items": template}, indent=1))


# ---------------------------------------------------------------- self-check

def self_check():
    # the rule: the generators' view gate first (no question), then a class that fixes the shape, then Jev-Omni's Q5
    good = {"views": {"n": 5, "azimuth_spread_deg": 40, "distance_m": [1.5, 3.]}, "physical": {"top_above_floor": {"value": 1.}, "dropped_share": .05},
            "raw": {"size": {"longest": .6}}, "model": {"kind": "box"}}
    card = lambda name, **kw: {**good, "identity": {"canonical": cards.canonical(name), "proposed": name}, **kw}  # noqa: E731
    assert route(card("lathe", views={**good["views"], "azimuth_spread_deg": 5}))[0] == "primitive"  # no generator takes it
    assert route(card("lathe"))[0] == "generated" and route(card("cardboard box"))[0] == "primitive"  # the class fixes the shape
    assert route(card("shoe"))[0] == "ask jev" and route(card("pet food bag"))[0] == "ask jev"  # merchandise, bag: it does not
    q = [.05, .05, .05, .05, .8]
    assert route(card("shoe"), q)[0] == "generated" and route(card("shoe"), q[::-1])[0] == "primitive"
    assert route(card("shoe"), [.1, .1, .1, .1, .6], cut=.7)[0] == "primitive"  # the cost knob
    assert route({**card("unidentified object"), "identity": {"name": "unidentified object"}}, q)[0] == "generated"  # no class: Jev
    # the class tables: pre-registered (held out) and post hoc (the rule)
    assert class_route("bag") == "complex" and class_route("bag", post_hoc=True) is None and class_route("merchandise") is None
    assert class_route("drill press") == "complex" and class_route("pallet") == "simple" and class_route("crate", post_hoc=True) is None
    # the fitted numbers: a cut, a temperature and bias, the calibration error
    assert fit_threshold([.1, .2, .8, .9], [0, 0, 1, 1]) == .5 and fit_threshold([], []) == np.inf
    rng = np.random.default_rng(0)
    z = rng.normal(0, 2, 4000)
    y = (rng.random(4000) < sigmoid(z)).astype(int)
    a, b = platt(z, y)
    assert abs(a - 1) < .15 and abs(b) < .15, (a, b)
    assert ece(sigmoid(z), y) < .03 and ece(np.full(4000, .9), np.zeros(4000)) > .85
    # the stratified sample: sqrt allocation, capped at the stratum, exactly n, seeded
    keys, strata = [f"k{i}" for i in range(130)], ["a"] * 100 + ["b"] * 25 + ["c"] * 5
    ch, al = stratified(keys, strata, 30)
    assert len(set(ch)) == 30 and al["a"][1] > al["b"][1] > al["c"][1] and ch == stratified(keys, strata, 30)[0]
    assert stratified(keys[:102], ["a"] * 100 + ["b"] * 2, 30)[1]["b"] == (2, 2)
    # the geometry feature: the best primitive's residual over the longest side; an unfitted circle is no cylinder
    fits = {"box": {"residual_m": .02, "lo": [0, 0, 0], "hi": [1., .5, .2]}, "plane": {"residual_m": .05},
            "cylinder": {"residual_m": .001, "fitted": False, "arc": "1" * 36}, "open frame": {"residual_m": .1}}
    g = geometry({"raw": {"model_fits": fits}})
    assert abs(g["rel"] - .02) < 1e-9 and g["res"]["cylinder"] == np.inf and g["ext"] == [.2, .5, 1.] and geometry({"raw": {}}) is None
    print("route_jev self-check ok: the rule (view gate, class, Jev Q5, the cut), class tables, cut / Platt / ECE, stratified sample, geometry")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a == ["--self-check"]:
        self_check()
    elif a[0] == "build":
        build(Path(a[1]), Path(a[2]))
    elif a[0] == "sheets":
        sheets(Path(a[1]), Path(a[2]))
    elif a[0] == "topup":
        topup(Path(a[1]), json.loads(a[2]))
    elif a[0] == "questions":
        questions(Path(a[1]))
    elif a[0] == "export":
        export(Path(a[1]), Path(a[2]))
    elif a[0] == "score":
        report(Path(a[1]), *score(Path(a[1])))
    else:
        sys.exit(__doc__)
