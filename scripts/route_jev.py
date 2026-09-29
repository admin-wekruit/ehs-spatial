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
  python scripts/route_jev.py score OUT SCRATCH   # OUT/results.json + OUT/results.md
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


def class_route(cls):
    """'simple' | 'complex' | None (no prior: geometry, or the decider, decides)."""
    if not cls:
        return None
    if cls in SIMPLE_CLASSES:
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
        keys = [s["keys"][i] for i in rng.permutation(len(s["keys"]))]
        tiles = []
        for k, key in enumerate(keys):
            code = f"{site[0].upper()}{k:03d}"
            template[code] = {"key": key, "label": None, "note": None}
            t = np.hstack([fit(data["crops"][f"{key}|som"]), np.full((side, 6, 3), 255, np.uint8), fit(data["crops"][f"{key}|context"])])
            t = np.vstack([t, np.full((26, t.shape[1], 3), 255, np.uint8)])
            cv2.putText(t, code, (4, side + 19), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 0), 2, cv2.LINE_AA)
            tiles.append(np.pad(t, ((0, 8), (0, 10), (0, 0)), constant_values=255))
        for p in range(0, len(tiles), per):
            t = tiles[p:p + per]
            t += [np.full_like(t[0], 255)] * (-len(t) % cols)
            cv2.imwrite(str(out / "sheets" / f"{site}-{p // per:02d}.jpg"), np.vstack([np.hstack(t[q:q + cols]) for q in range(0, len(t), cols)]),
                        [cv2.IMWRITE_JPEG_QUALITY, 85])
    (out / "labels-template.json").write_text(json.dumps({"labelled_by": "agent, by looking at the blind sheets (not ground truth)",
                                                          "guide": LABEL_GUIDE, "items": template}, indent=1))


if __name__ == "__main__":
    a = sys.argv[1:]
    if a == ["--self-check"]:
        self_check()
    elif a[0] == "build":
        build(Path(a[1]), Path(a[2]))
    elif a[0] == "sheets":
        sheets(Path(a[1]), Path(a[2]))
    elif a[0] == "score":
        score(Path(a[1]), Path(a[2]))
    else:
        sys.exit(__doc__)
