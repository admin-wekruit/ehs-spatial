"""X10 evaluation (local numpy): the audited 'everything' reference and every method's recall / precision against it.

  python scripts/x10_eval.py pool   RUN_DIR JPG_DIR     # every method's masks per reference frame -> candidate clusters
                                                        # (IoU >= CLUSTER_IOU) -> audit sheets (RUN_DIR/audit/*.jpg)
  python scripts/x10_eval.py score  RUN_DIR             # recall / precision / time per method and stack -> results
  python scripts/x10_eval.py --self-check

Reference = the clusters labelled W (one whole thing) or K (a block of repeated things counted once) on the audit
sheets (agent-labelled: RUN_DIR/audit/labels.json), plus things added by hand (boxes -> the pool mask that fits best).
"""
import json
import sys
from pathlib import Path

import numpy as np

CLUSTER_IOU = .6
FOUND_IOU = .5               # the harness's rule (fast_report_eval.IOU_FOUND)
GRID = 2                     # IoU on a stride-2 grid of the evaluation grid (360x640 -> 180x320)
PERSON_INSIDE = .5
FLOORS = {"vocab": .3, "generic": .3, "ram": .3, "owlw": .3, "aux": .3, "labelw": .3, "labelw_t": .3, "generic_t": .3}
MIN_PX = 24                  # evaluation-grid pixels (~5 x 5 at 640 wide, ~10 x 10 in the 1280 source): nothing smaller is identifiable
MAX_SHARE = .6
METHODS = ["vocab", "generic", "ram", "owlw", "owlbox", "amg16", "amg32", "amg16c1", "amg32c1", "geo", "geosam", "ridge", "listing"]
FAMILY = {"vocab": "sam3-words", "generic": "sam3-generic", "ram": "tagger", "owlw": "tagger", "owlbox": "owl-boxes",
          "amg16": "amg", "amg32": "amg", "amg16c1": "amg", "amg32c1": "amg", "geo": "geometry", "geosam": "geometry",
          "ridge": "ridge", "listing": "listing"}


def unpack(bits, w):
    return np.unpackbits(bits, axis=-1, count=w).astype(bool)


NB_PLAN = {}  # {video: {ref frame: [before, after]}}: frames.json's neighbours, set by the caller


class Video:
    def __init__(self, run_dir, name):
        self.name = name
        self.rec = json.loads((Path(run_dir) / f"{name}.json").read_text())
        self.z = np.load(Path(run_dir) / f"{name}.npz")
        self.hw = tuple(self.rec["eval_hw"])
        self.run_dir, self.extra = Path(run_dir), {}

    def get(self, f, m):
        k = f"{f}/{m}/bits"
        z = self.z
        if m in EXTRA_FILES:  # methods run after run 001: RUN/<video>-<file>.npz
            path = self.run_dir / f"{self.name}-{EXTRA_FILES[m]}.npz"
            z = self.extra.setdefault(path, np.load(path) if path.exists() else {})
        if k not in z:
            return None
        bits = z[k]
        masks = unpack(bits, self.hw[1]) if len(bits) else np.zeros((0, *self.hw), bool)
        return masks, z[f"{f}/{m}/score"], [str(x) for x in z[f"{f}/{m}/label"]]

    def rec_neighbours(self, f):
        return [x for x in self.rec.get("neighbours_of", {}).get(str(f), [])] or self.nb_from_plan(f)

    def nb_from_plan(self, f):
        return NB_PLAN.get(self.name, {}).get(str(f), [])

    def aux(self, f, word, floor):
        m, s, lab = self.get(f, "aux")
        sel = np.array([l == word for l in lab], bool) & (s >= floor)
        return m[sel].any(0) if sel.any() else np.zeros(self.hw, bool)

    def method(self, f, m, floor=None):
        """A method's masks on frame f as used: score floor (SAM 3 word methods), people out, size limits."""
        got = self.get(f, m)
        if got is None:
            return None
        masks, s, lab = got
        keep = np.ones(len(masks), bool)
        fl = FLOORS.get(m) if floor is None else floor
        if fl is not None:
            keep &= s >= fl
        if not len(masks):
            return masks, s, lab
        if m in WORD_METHODS:  # the core's rule (segment.dedupe): floods across words and doubles within a word go
            keep &= dedupe_mask(masks, s, lab, keep)
        area = masks.reshape(len(masks), -1).sum(1)
        person = self.aux(f, "person", .4)
        inside = (masks & person).reshape(len(masks), -1).sum(1) / np.maximum(area, 1)
        keep &= (area >= MIN_PX) & (area <= MAX_SHARE * masks[0].size if len(masks) else True) & (inside < PERSON_INSIDE)
        return masks[keep], s[keep], [l for l, k in zip(lab, keep) if k]


WORD_METHODS = {"vocab", "generic", "ram", "owlw", "labelw", "labelw_t", "generic_t"}
EXTRA_FILES = {"ridge2": "ridge2", "labelw": "words", "labelw_t": "words", "generic_t": "words"}
FLOOD_IOU, INSIDE_SAME = .8, .5  # fast_report.segment: FLOOD_IOU, INSIDE


def dedupe_mask(masks, score, words, valid):
    """fast_report.segment.dedupe in numpy, one frame: best score first; a mask with IoU > 0.8 with a kept mask of
    another word, or at least half inside a kept mask of the same word, is dropped. -> keep flags."""
    idx = np.flatnonzero(valid)
    if not len(idx):
        return np.zeros(len(masks), bool)
    idx = idx[np.argsort(-score[idx], kind="stable")]
    x = masks[idx][:, ::2, ::2].reshape(len(idx), -1).astype(np.float32)
    inter, area = x @ x.T, x.sum(1)
    w = np.array(words)[idx]
    kept = []
    for a in range(len(idx)):
        if kept:
            k = np.array(kept)
            other = w[k] != w[a]
            iou = inter[a, k] / np.maximum(area[a] + area[k] - inter[a, k], 1)
            if (other & (iou > FLOOD_IOU)).any() or (~other & (inter[a, k] >= INSIDE_SAME * max(area[a], 1))).any():
                continue
        kept.append(a)
    out = np.zeros(len(masks), bool)
    out[idx[kept]] = True
    return out


def iou_matrix(a, b, grid=GRID):
    """(n, h, w), (m, h, w) bool -> (n, m) IoU on a stride-grid, plus |a & b| / |a| (share of a inside b)."""
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b))), np.zeros((len(a), len(b)))
    x = a[:, ::grid, ::grid].reshape(len(a), -1).astype(np.float32)
    y = b[:, ::grid, ::grid].reshape(len(b), -1).astype(np.float32)
    inter = x @ y.T
    sa, sb = x.sum(1), y.sum(1)
    union = sa[:, None] + sb[None] - inter
    return inter / np.maximum(union, 1), inter / np.maximum(sa[:, None], 1)


def dilate(masks, r=1):
    """Binary dilation by r px (square), numpy only: thin things get a tolerance of r px on each side."""
    out = masks.copy()
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dy or dx:
                out |= np.roll(np.roll(masks, dy, axis=-2), dx, axis=-1)
    return out


def cluster(masks, sources, iou=CLUSTER_IOU):
    """Greedy: masks sorted by how many sources' masks overlap them at >= iou (support), then each unassigned mask
    starts a cluster and takes every unassigned mask at >= iou with it. -> [(representative index, member indices)]."""
    m, _ = iou_matrix(masks, masks)
    close = m >= iou
    fam = np.array(sources)
    support = np.array([len(set(fam[close[i]])) for i in range(len(masks))])
    area = masks.reshape(len(masks), -1).sum(1)
    order = sorted(range(len(masks)), key=lambda i: (-support[i], -area[i]))
    done = np.zeros(len(masks), bool)
    out = []
    for i in order:
        if done[i]:
            continue
        members = [j for j in np.nonzero(close[i] & ~done)[0]]
        done[members] = True
        out.append((i, members))
    return out


# ---------- pool: candidate clusters + audit sheets ----------

POOL_FLOORS = {}  # the pool = every method's masks as used, so each used mask sits in exactly one cluster
TILE_W, TILE_H, COLS, ROWS = 200, 150, 8, 6


def frame_pool(v, f):
    """Every method's masks on frame f (pool floors) -> (masks, [(method, index in the method, label)])."""
    ms, src = [], []
    for m in METHODS:
        got = v.method(f, m, POOL_FLOORS.get(m))
        if got is None:
            continue
        masks, s, lab = got
        ms.append(masks)
        src += [(m, i, lab[i] if i < len(lab) else "", float(s[i])) for i in range(len(masks))]
    return (np.concatenate(ms) if ms else np.zeros((0, *v.hw), bool)), src


def hint(members, src):
    """A name for the tile: the listing's label, else the most common SAM 3 word, else ''."""
    labs = [src[j][2].split("|")[0] for j in members if src[j][0] == "listing"]
    labs = labs or [src[j][2] for j in members if src[j][0] in ("vocab", "generic", "ram", "owlw") and src[j][2]]
    return max(set(labs), key=labs.count) if labs else ""


def tile(img, mask, label):
    """Crop around the mask (padding: half its size, at least 40 px), letterboxed to TILE_W x TILE_H, outline drawn."""
    import cv2
    H, W = img.shape[:2]
    m = cv2.resize(mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
    ys, xs = np.nonzero(m)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    pad = max(40 * W // 1280, int(.5 * max(x1 - x0, y1 - y0)))
    cw, ch = x1 - x0 + 2 * pad, y1 - y0 + 2 * pad
    aspect = TILE_W / (TILE_H - 14)
    cw, ch = max(cw, ch * aspect), max(ch, cw / aspect)  # the tile's aspect: no letterbox
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    cx0, cy0 = int(max(0, min(W - cw, cx - cw / 2))), int(max(0, min(H - ch, cy - ch / 2)))
    cx1, cy1 = int(min(W, cx0 + cw)), int(min(H, cy0 + ch))
    crop, cm = img[cy0:cy1, cx0:cx1], m[cy0:cy1, cx0:cx1]
    s = min(TILE_W / crop.shape[1], (TILE_H - 14) / crop.shape[0])
    size = (max(1, int(crop.shape[1] * s)), max(1, int(crop.shape[0] * s)))
    crop = cv2.resize(crop, size, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    cm = cv2.resize(cm, size, interpolation=cv2.INTER_NEAREST)
    cs, _ = cv2.findContours(cm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(crop, cs, -1, (0, 0, 0), 3)
    cv2.drawContours(crop, cs, -1, (255, 255, 0), 1)
    out = np.full((TILE_H, TILE_W, 3), 40, np.uint8)
    out[14:14 + size[1], :size[0]] = crop
    cv2.putText(out, label[:30], (2, 11), cv2.FONT_HERSHEY_SIMPLEX, .38, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def sheets(tiles_list, path_fmt):
    import cv2
    per = COLS * ROWS
    paths = []
    for p in range(0, len(tiles_list), per):
        page = np.full((ROWS * TILE_H, COLS * TILE_W, 3), 0, np.uint8)
        for k, t in enumerate(tiles_list[p:p + per]):
            r, c = divmod(k, COLS)
            page[r * TILE_H:(r + 1) * TILE_H, c * TILE_W:(c + 1) * TILE_W] = t
            cv2.rectangle(page, (c * TILE_W, r * TILE_H), ((c + 1) * TILE_W - 1, (r + 1) * TILE_H - 1), (90, 90, 90), 1)
        path = str(path_fmt).format(p // per)
        cv2.imwrite(path, page, [cv2.IMWRITE_JPEG_QUALITY, 85])
        paths.append(path)
    return paths


def pool(run_dir, jpg_dir, rate, videos=None):
    """Clusters per reference frame -> the audit sample: seeded uniform samples of two strata, clusters whose mask is
    long and narrow ('thin') and the rest, at rate[stratum] each (weight = 1 / rate). Tiles in random order within each frame; sheets of 48.
    Saves RUN_DIR/audit/clusters-<video>.npz (every cluster's representative mask) and clusters.json (stratum,
    sampled, weight = 1 / inclusion probability, sheet position)."""
    import cv2
    run_dir, jpg_dir = Path(run_dir), Path(jpg_dir)
    audit = run_dir / "audit"
    audit.mkdir(exist_ok=True)
    import zlib
    for vp in sorted(run_dir.glob("*.npz")):
        if videos and vp.stem not in videos:
            continue
        v = Video(run_dir, vp.stem)
        rng = np.random.default_rng(zlib.crc32(v.name.encode()))  # per video: videos can be pooled one at a time
        totals = {}
        reps, tiles_list, rows_v = {}, [], []
        for f in v.rec["reference"]:
            masks, src = frame_pool(v, f)
            cl = cluster(masks, [FAMILY[x[0]] for x in src])
            img = cv2.imread(str(jpg_dir / f"{v.name}-{f:05d}.jpg"))
            stuff = v.aux(f, "floor", .3) | v.aux(f, "wall", .3) | v.aux(f, "ceiling", .3)
            rows = []
            for k, (rep, members) in enumerate(cl):
                cid = f"{v.name}-{f}-{k}"
                area = int(masks[rep].sum())
                length, width = elongation(masks[rep])
                thin = length >= 5 * width and width <= 6 * v.hw[1] / 640
                row = {"id": cid, "frame": f, "area_px": area, "share": round(area / masks[rep].size, 5),
                       "families": sorted({FAMILY[src[j][0]] for j in members}), "methods": sorted({src[j][0] for j in members}),
                       "members": [[src[j][0], src[j][1]] for j in members], "hint": hint(members, src), "stuff_inside": round(float((masks[rep] & stuff).sum() / max(area, 1)), 3),
                       "bbox": [int(x) for x in bbox(masks[rep])], "stratum": "thin" if thin else "rest"}
                p = rate[row["stratum"]]
                row["sampled"] = bool(rng.random() < p)
                row["weight"] = 1 / p
                rows.append(row)
                reps[cid] = masks[rep]
            order = [r for r in rows if r["sampled"]]
            rng.shuffle(order)
            for row in order:
                tiles_list.append(tile(img, reps[row["id"]], f"{f}.{row['id'].rsplit('-', 1)[1]} {row['hint']}"))
                row["sheet_pos"] = len(tiles_list) - 1
            rows_v += rows
            totals[f"{v.name}-{f}"] = {"pool_masks": len(masks), "clusters": len(cl), "thin_stratum": sum(r["stratum"] == "thin" for r in rows),
                                       "sampled": len(order)}
            print(v.name, f, totals[f"{v.name}-{f}"], flush=True)
        (audit / f"clusters-{v.name}.json").write_text(json.dumps(rows_v))
        (audit / f"pool-totals-{v.name}.json").write_text(json.dumps({"rate_rest": rate, "frames": totals}, indent=1))
        ids = list(reps)
        np.savez_compressed(audit / f"clusters-{v.name}.npz", ids=np.array(ids), bits=np.packbits(np.array([reps[i] for i in ids]), axis=-1))
        paths = sheets(tiles_list, audit / f"sheet-{v.name}-{{:03d}}.jpg")
        print(v.name, len(tiles_list), "tiles,", len(paths), "sheets", flush=True)


def bbox(m):
    ys, xs = np.nonzero(m)
    return (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1) if len(xs) else (0, 0, 0, 0)


# ---------- reference + scoring ----------

THIN_WORDS = ("cable", "cord", "wire", "hose", "rope", "strap", "chain", "tube", "pipe", "conduit", "lead", "belt", "band", "string")
SIZE_EDGES = (.0005, .005, .05)   # share of the frame: tiny < 0.05% <= small < 0.5% <= medium < 5% <= large
SMALL_SHARE = .0025               # 'small' shape class: not thin, under 0.25% of the frame (~24 x 24 px at 640 x 360)
PART_INSIDE, BACKGROUND_MAX, GROUP_COVER = .8, .2, .5


def size_class(share):
    return ("tiny", "small", "medium", "large")[int(np.searchsorted(SIZE_EDGES, share, side="right"))]


def elongation(mask):
    """(length px, mean width px) from the mask's principal axes: length = extent along the first axis."""
    ys, xs = np.nonzero(mask)
    if len(xs) < 3:
        return float(len(xs)), 1.
    p = np.stack([xs, ys], 1).astype(float)
    c = p - p.mean(0)
    axis = np.linalg.svd(c, full_matrices=False)[2][0]
    proj = c @ axis
    length = float(proj.max() - proj.min() + 1)
    return length, len(xs) / length


def shape_class(mask, name, thin_flag=None):
    """thin: the audit said so, or a thin word, or long and narrow (length >= 5 x width, width <= 6 px at 640 wide);
    small: otherwise under SMALL_SHARE of the frame; normal: the rest."""
    length, width = elongation(mask)
    geo_thin = length >= 5 * width and width <= 6 * mask.shape[1] / 640
    word_thin = any(w in (name or "").lower() for w in THIN_WORDS)
    if thin_flag if thin_flag is not None else (geo_thin or word_thin):
        return "thin"
    return "small" if mask.sum() / mask.size < SMALL_SHARE else "normal"


def classify(masks, ref, ref_thin):
    """Every mask of a method vs the frame's reference items -> (per item best IoU (tolerant for thin items),
    per mask category whole | part | group | background | other)."""
    if not len(ref):
        return np.zeros(0), np.array(["background"] * len(masks))
    iou, inside = iou_matrix(masks, ref)
    if ref_thin.any() and len(masks):
        iou_t, _ = iou_matrix(dilate(masks), dilate(ref[ref_thin]))
        iou[:, ref_thin] = np.maximum(iou[:, ref_thin], iou_t)
    best_item = iou.max(0) if len(masks) else np.zeros(len(ref))
    if not len(masks):
        return best_item, np.array([], dtype="U10")
    union = ref.any(0)
    on_ref = (masks & union)[:, ::GRID, ::GRID].reshape(len(masks), -1).sum(1) / np.maximum(masks[:, ::GRID, ::GRID].reshape(len(masks), -1).sum(1), 1)
    _, cover = iou_matrix(ref, masks)  # share of each item inside each mask
    cats = []
    for i in range(len(masks)):
        if iou[i].max() >= FOUND_IOU:
            cats.append("whole")
        elif inside[i].max() >= PART_INSIDE:
            cats.append("part")
        elif (cover[:, i] >= GROUP_COVER).sum() >= 2:
            cats.append("group")
        elif on_ref[i] <= BACKGROUND_MAX:
            cats.append("background")
        else:
            cats.append("other")
    return best_item, np.array(cats)


REF_LETTERS = {"W": "whole", "T": "thin", "K": "block"}   # audit letters that make a reference item
AUDIT_LETTERS = set("WTKPBMX")


def load_audit(run_dir):
    """audit/labels.json {'sheets': {'<video>-<page>': 48 letters}, 'names': {cluster id: name}} + clusters.json ->
    (rows by video with 'letter' set on audited clusters, representative masks by cluster id)."""
    audit = Path(run_dir) / "audit"
    labels = json.loads((audit / "labels.json").read_text())
    index = {p.stem.split("-", 1)[1]: json.loads(p.read_text()) for p in sorted(audit.glob("clusters-*.json"))}
    per = COLS * ROWS
    reps = {}
    for video, rows in index.items():
        by_pos = {r["sheet_pos"]: r for r in rows if r.get("sampled")}
        for sheet, text in labels["sheets"].items():
            v, page = sheet.rsplit("-", 1)
            if v != video:
                continue
            for k, ch in enumerate(text.replace(" ", "")):
                assert ch in AUDIT_LETTERS, (sheet, k, ch)
                if int(page) * per + k in by_pos:
                    by_pos[int(page) * per + k]["letter"] = ch
        z = np.load(audit / f"clusters-{video}.npz")
        w = int(json.loads((Path(run_dir) / f"{video}.json").read_text())["eval_hw"][1])
        for cid, bits in zip(z["ids"], z["bits"]):
            reps[str(cid)] = bits
        for r in rows:
            r["name"] = labels.get("names", {}).get(r["id"], r["hint"])
            r["w"] = w
    return index, reps


def reference_items(index, reps):
    """Audited W / T / K clusters -> {(video, frame): [item]}: item = {id, mask, kind, name, weight, size, shape}; two
    items with IoU >= 0.5 on one frame are one (the bigger kept)."""
    ref = {}
    for video, rows in index.items():
        for r in rows:
            if r.get("letter") in REF_LETTERS:
                m = unpack(reps[r["id"]][None], r["w"])[0]
                ref.setdefault((video, r["frame"]), []).append({"id": r["id"], "mask": m, "kind": REF_LETTERS[r["letter"]], "name": r["name"],
                                                               "weight": r["weight"], "thin_letter": r["letter"] == "T"})
    for key, items in ref.items():
        items.sort(key=lambda it: -it["mask"].sum())
        ms = np.array([it["mask"] for it in items])
        iou, _ = iou_matrix(ms, ms)
        keep = []
        for i in range(len(items)):
            if all(iou[i, j] < FOUND_IOU for j in keep):
                keep.append(i)
        ref[key] = [items[i] for i in keep]
        for it in ref[key]:
            it["size"] = size_class(it["mask"].sum() / it["mask"].size)
            it["shape"] = shape_class(it["mask"], it["name"], True if it["thin_letter"] else None)
    return ref


def found_by(masks, items):
    """best IoU of any mask with each item (1 px tolerance for thin items)."""
    if not items:
        return np.zeros(0)
    rm = np.array([it["mask"] for it in items])
    iou, _ = iou_matrix(masks, rm)
    thin = np.array([it["shape"] == "thin" for it in items])
    if thin.any() and len(masks):
        iou_t, _ = iou_matrix(dilate(masks), dilate(rm[thin]))
        iou[:, thin] = np.maximum(iou[:, thin], iou_t)
    return iou.max(0) if len(masks) else np.zeros(len(items))


def score_recall(run_dir, ref, sets, floors=None):
    """sets: {name: [methods]} (a stack = the union of its methods' masks) -> {name: [(item, found, best iou)]}."""
    floors = floors or {}
    out = {k: [] for k in sets}
    for (video, f), items in sorted(ref.items()):
        v = VIDEOS_CACHE.setdefault(video, Video(run_dir, video))
        got = {}
        for name, parts in sets.items():
            ms = []
            for p in parts:
                if p not in got:
                    x = v.method(f, p, floors.get(p))
                    got[p] = x[0] if x is not None else np.zeros((0, *v.hw), bool)
                ms.append(got[p])
            best = found_by(np.concatenate(ms), items)
            out[name] += [(it, bool(best[i] >= FOUND_IOU), float(best[i])) for i, it in enumerate(items)]
    return out


def recall_table(rows):
    """weighted recall (1 / inclusion probability of each audited item) overall and by class."""
    groups = {"all": lambda it: True, **{f"size:{k}": (lambda it, k=k: it["size"] == k) for k in ("tiny", "small", "medium", "large")},
              **{f"shape:{k}": (lambda it, k=k: it["shape"] == k) for k in ("thin", "small", "normal")},
              **{f"kind:{k}": (lambda it, k=k: it["kind"] == k) for k in ("whole", "thin", "block")}}
    out = {}
    for g, sel in groups.items():
        r = [(it["weight"], found) for it, found, _ in rows if sel(it)]
        wsum = sum(w for w, _ in r)
        out[g] = {"items": len(r), "found": sum(f for _, f in r), "weighted_recall": round(sum(w for w, f in r if f) / wsum, 3) if wsum else None}
    return out


def precision_table(index, methods):
    """Label composition of each method's masks, over the audited clusters (weighted): whole (W/T/K), part (P),
    background (B), group (M: several things or a thing plus background), unclear (X); masks per frame."""
    cat = {"W": "whole", "T": "whole", "K": "whole", "P": "part", "B": "background", "M": "group", "X": "unclear"}
    out = {}
    frames = {(v, r["frame"]) for v, rows in index.items() for r in rows}
    for m in methods:
        acc, n_masks = {}, 0
        for rows in index.values():
            for r in rows:
                k = sum(1 for mm, _ in r["members"] if mm in m)
                n_masks += k
                if k and r.get("letter"):
                    acc[cat[r["letter"]]] = acc.get(cat[r["letter"]], 0.) + k * r["weight"]
        tot = sum(acc.values())
        out["+".join(m)] = {"masks_per_frame": round(n_masks / max(len(frames), 1), 1),
                            **{c: round(acc.get(c, 0.) / tot, 3) if tot else None for c in ("whole", "part", "background", "group", "unclear")}}
    return out


# method timing: what one frame costs on one A100 (median of the per-frame rows), and what it ADDS to the fast core
# (SAM 3 vision features and DA3 depth are computed by the core already on its keyframes)
TIME_PARTS = {"vocab": ["sam3_vision", "vocab"], "generic": ["sam3_vision", "generic"], "ram": ["ram++", "sam3_vision", "ram"],
              "owlw": ["owl", "sam3_vision", "owlw"], "owlbox": ["owl", "sam2_embed", "owlbox_decode"],
              "amg16": ["amg16"], "amg32": ["amg32"], "amg16c1": ["amg16c1"], "amg32c1": ["amg32c1"],
              "geo": ["geo_da3_and_planes"], "geosam": ["geo_da3_and_planes", "sam2_embed", "geosam_decode"],
              "ridge": ["sam2_embed", "ridge_filter", "ridge_decode"], "listing": ["listing_qwen_per_frame", "sam2_embed"]}
IN_CORE = {"sam3_vision", "geo_da3"}   # already paid by the core on object keyframes


def method_times(recs, ram):
    """{part: median seconds per frame over every video} and {method: (per frame, added to the core per frame)}."""
    rows = {}
    for rec in recs:
        for k, v in rec["per_frame_s"].items():
            rows.setdefault(k, []).extend(x["s"] for x in v["rows"])
        lst = rec.get("listing") or {}
        if lst:
            rows.setdefault("listing_qwen_per_frame", []).append(lst["wall_s"] / max(len(rec["reference"]), 1))
        for g in (rec.get("geo") or {}).values():
            rows.setdefault("geo_da3", []).append(g["da3_s"])
            rows.setdefault("geo_planes", []).append(g.get("planes_s", 0))
    rows["ram++"] = list(ram["per_frame_s"])
    med = {k: float(np.median(v)) for k, v in rows.items()}
    out = {}
    for m, parts in TIME_PARTS.items():
        total = sum(med.get(p, 0.) for p in parts)
        added = total - sum(med.get(p, 0.) for p in parts if p in IN_CORE)
        if m in ("geo", "geosam"):
            added -= med.get("geo_da3", 0.)  # DA3 on the keyframes is the core's own work
        out[m] = {"per_frame_s": round(total, 4), "added_per_frame_s": round(max(added, 0.), 4)}
    return {k: round(v, 4) for k, v in med.items()}, out


VIDEOS_CACHE = {}


NB_METHODS = ["vocab", "generic", "ram", "owlw", "owlbox", "amg32", "amg32c1", "ridge"]


def warp_to(jpg_dir, video, f, n, hw):
    """DIS optical flow from frame f to frame n on the evaluation grid -> remap maps that pull n's pixels onto f."""
    import cv2
    g = [cv2.cvtColor(cv2.resize(cv2.imread(str(Path(jpg_dir) / f"{video}-{x:05d}.jpg")), (hw[1], hw[0]), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
         for x in (f, n)]
    flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM).calc(g[0], g[1], None)
    yy, xx = np.mgrid[:hw[0], :hw[1]].astype(np.float32)
    return xx + flow[..., 0], yy + flow[..., 1]


def five_fps(run_dir, jpg_dir, ref, floors=None):
    """Object keyframes at 5 fps instead of every 3rd: a reference item counts as found when the method finds it on
    its frame OR on the 5 fps keyframe before / after, that frame's masks warped onto the reference frame by optical
    flow (a lower bound: warping blurs thin masks). -> per method: recall by shape, reference-only vs + neighbours."""
    import cv2
    floors = floors or {}
    out = {}
    for (video, f), items in sorted(ref.items()):
        v = VIDEOS_CACHE.setdefault(video, Video(run_dir, video))
        nbs = [x for x in v.rec_neighbours(f) if x is not None]
        rmask = np.array([it["mask"] for it in items])
        thin = np.array([it["shape"] == "thin" for it in items])
        maps = {n: warp_to(jpg_dir, video, f, n, v.hw) for n in nbs}
        for m in NB_METHODS:
            base = v.method(f, m, floors.get(m))
            if base is None:
                continue
            best0, _ = classify(base[0], rmask, thin)
            best = best0.copy()
            extra_masks = 0
            for n in nbs:
                x = v.method(n, m, floors.get(m))
                if x is None or not len(x[0]):
                    continue
                mx, my = maps[n]
                warped = np.array([cv2.remap(mk.astype(np.uint8), mx, my, cv2.INTER_NEAREST) > 0 for mk in x[0]])
                b, _ = classify(warped, rmask, thin)
                best = np.maximum(best, b)
                extra_masks += len(x[0])
            r = out.setdefault(m, {"frames": 0, "extra_masks": 0, "rows": []})
            r["frames"] += 1
            r["extra_masks"] += extra_masks
            r["rows"] += [(it["shape"], bool(best0[i] >= FOUND_IOU), bool(best[i] >= FOUND_IOU)) for i, it in enumerate(items)]
    summary = {}
    for m, r in out.items():
        row = {}
        for shape in ("thin", "small", "normal", "all"):
            sel = [x for x in r["rows"] if shape == "all" or x[0] == shape]
            row[shape] = {"of": len(sel), "ref_only": sum(x[1] for x in sel), "with_5fps_neighbours": sum(x[2] for x in sel),
                          "gain": sum(x[2] for x in sel) - sum(x[1] for x in sel)}
        summary[m] = {**row, "neighbour_masks_per_ref_frame": round(r["extra_masks"] / max(r["frames"], 1), 1)}
    return summary


def uncovered(run_dir, jpg_dir, videos=None):
    """Per reference frame: pixels that NO method's mask (as used, people out) touches keep their colour, the rest is
    dimmed; a 10% grid with labels so a missed thing can be located. Two frames per sheet (audit/uncovered-*.jpg) ->
    the look for things every method missed."""
    import cv2
    run_dir, jpg_dir = Path(run_dir), Path(jpg_dir)
    out = []
    for vp in sorted(run_dir.glob("*.npz")):
        if videos and vp.stem not in videos:
            continue
        v = Video(run_dir, vp.stem)
        panels = []
        for f in v.rec["reference"]:
            masks, _ = frame_pool(v, f)
            cov = masks.any(0) | v.aux(f, "person", .4)
            img = cv2.imread(str(jpg_dir / f"{v.name}-{f:05d}.jpg"))
            img = cv2.resize(img, (800, int(800 * img.shape[0] / img.shape[1])), interpolation=cv2.INTER_AREA)
            c = cv2.resize(cov.astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
            dim = img.copy()
            dim[c] = (dim[c] * .3).astype(np.uint8)
            h, w = dim.shape[:2]
            for k in range(1, 10):
                cv2.line(dim, (w * k // 10, 0), (w * k // 10, h), (0, 200, 255), 1)
                cv2.line(dim, (0, h * k // 10), (w, h * k // 10), (0, 200, 255), 1)
                cv2.putText(dim, str(k), (w * k // 10 + 2, 12), cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 255, 255), 1)
                cv2.putText(dim, str(k), (2, h * k // 10 - 2), cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 255, 255), 1)
            both = np.hstack([img, dim])
            cv2.putText(both, f"{v.name} {f}  uncovered {100 * (1 - c.mean()):.0f}%", (5, h - 8), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 0), 2)
            panels.append(both)
        for i in range(0, len(panels), 2):
            path = run_dir / "audit" / f"uncovered-{v.name}-{i // 2:02d}.jpg"
            cv2.imwrite(str(path), np.vstack(panels[i:i + 2]), [cv2.IMWRITE_JPEG_QUALITY, 85])
            out.append(str(path))
    return out


SINGLE = ["vocab", "generic", "ram", "owlw", "owlbox", "amg16", "amg32", "amg16c1", "amg32c1", "geo", "geosam", "ridge", "listing"]
STACKS = {"vocab+generic": ["vocab", "generic"], "vocab+ram": ["vocab", "ram"], "vocab+owlw": ["vocab", "owlw"],
          "vocab+owlbox": ["vocab", "owlbox"], "vocab+amg16": ["vocab", "amg16"], "vocab+amg32": ["vocab", "amg32"],
          "vocab+amg32c1": ["vocab", "amg32c1"], "vocab+geosam": ["vocab", "geosam"], "vocab+ridge": ["vocab", "ridge"],
          "vocab+generic+owlbox": ["vocab", "generic", "owlbox"], "vocab+generic+ridge": ["vocab", "generic", "ridge"],
          "vocab+generic+owlbox+ridge": ["vocab", "generic", "owlbox", "ridge"],
          "vocab+generic+owlw+owlbox+ridge": ["vocab", "generic", "owlw", "owlbox", "ridge"],
          "vocab+generic+owlbox+ridge+amg16": ["vocab", "generic", "owlbox", "ridge", "amg16"],
          "vocab+generic+owlbox+ridge+amg32": ["vocab", "generic", "owlbox", "ridge", "amg32"],
          "vocab+generic+amg32c1": ["vocab", "generic", "amg32c1"],
          "all_but_listing": [m for m in SINGLE if m != "listing"], "pool (all + listing)": SINGLE}
LATER = ["ridge2", "labelw", "labelw_t", "generic_t"]  # run after the audit pool: the pool reference under-counts what they alone find
CORE3 = ["vocab", "generic", "owlbox"]
STACKS.update({"vocab+generic+owlbox+ridge2": CORE3 + ["ridge2"], "vocab+generic+owlbox+labelw": CORE3 + ["labelw"],
               "vocab+generic+owlbox+labelw+ridge2": CORE3 + ["labelw", "ridge2"],
               "vocab+generic+owlbox+labelw+amg16": CORE3 + ["labelw", "amg16"],
               "vocab+generic+owlbox+labelw+ridge+amg16": CORE3 + ["labelw", "ridge", "amg16"],
               "vocab+generic+owlbox+labelw_t+generic_t": CORE3 + ["labelw_t", "generic_t"],
               "vocab+generic+owlbox+labelw_t+generic_t+ridge2": CORE3 + ["labelw_t", "generic_t", "ridge2"],
               "vocab+generic+owlbox+amg32": CORE3 + ["amg32"],
               "everything (all + later)": SINGLE + LATER})


def weighted_count(index):
    """Estimated things per frame and label shares over the whole pool (Horvitz-Thompson: sum of 1 / p)."""
    tot, by = 0., {}
    n_frames = len({(v, r["frame"]) for v, rows in index.items() for r in rows})
    for rows in index.values():
        for r in rows:
            if r.get("letter"):
                by[r["letter"]] = by.get(r["letter"], 0.) + r["weight"]
                tot += r["weight"]
    return {"frames": n_frames, "audited_clusters": sum(1 for rows in index.values() for r in rows if r.get("letter")),
            "clusters_total": sum(len(rows) for rows in index.values()),
            "estimated_per_frame": {k: round(v / n_frames, 1) for k, v in sorted(by.items())},
            "estimated_share": {k: round(v / tot, 3) for k, v in sorted(by.items())}}


def score(run_dir, jpg_dir, frames_json):
    run_dir = Path(run_dir)
    plan = json.loads(Path(frames_json).read_text())
    NB_PLAN.update({v: p["neighbours"] for v, p in plan.items()})
    index, reps = load_audit(run_dir)
    ref = reference_items(index, reps)
    n_items = sum(len(v) for v in ref.values())
    sets = {m: [m] for m in SINGLE + LATER} | STACKS
    rec = score_recall(run_dir, ref, sets)
    recall = {k: recall_table(v) for k, v in rec.items()}
    by_video = {}
    for k, rows in rec.items():
        for video in plan:
            sel = [(it, f, i) for it, f, i in rows if it["id"].startswith(video + "-")]
            by_video.setdefault(k, {})[video] = recall_table(sel)["all"]
    prec = precision_table(index, [[m] for m in SINGLE] + [v for v in STACKS.values() if not set(v) & set(LATER)])
    recs = [json.loads((run_dir / f"{v}.json").read_text()) for v in plan]
    ram = json.loads((run_dir / "ram.json").read_text())
    med, times = method_times(recs, ram)
    missed = {k: [{"id": it["id"], "name": it["name"], "shape": it["shape"], "size": it["size"]} for it, f, _ in v if not f and it["weight"] <= 4]
              for k, v in rec.items() if k in ("vocab", "vocab+generic+owlbox+ridge", "all_but_listing")}
    items = [{"id": it["id"], "kind": it["kind"], "name": it["name"], "shape": it["shape"], "size": it["size"], "weight": it["weight"],
              "area_px": int(it["mask"].sum())} for key in sorted(ref) for it in ref[key]]
    out = {"reference": {**weighted_count(index), "items": n_items, "items_by_shape": {s: sum(1 for i in items if i["shape"] == s) for s in ("thin", "small", "normal")},
                         "items_by_size": {s: sum(1 for i in items if i["size"] == s) for s in ("tiny", "small", "medium", "large")},
                         "items_by_kind": {s: sum(1 for i in items if i["kind"] == s) for s in ("whole", "thin", "block")}},
           "recall": recall, "recall_by_video": by_video, "precision": prec, "time_parts_median_s": med, "time_per_frame": times,
           "missed_examples": missed, "items": items}
    return out, ref


def five_fps_run(run_dir, jpg_dir, frames_json, ref):
    plan = json.loads(Path(frames_json).read_text())
    NB_PLAN.update({v: p["neighbours"] for v, p in plan.items()})
    return five_fps(run_dir, jpg_dir, ref)


def look_views(jpg_dir, frames_json, out_dir):
    """One image per reference frame for the own look: the frame at 1600 px wide (4:3 clips 1200) with a 5% grid,
    every other line numbered 0-20 on both axes, so a thing can be boxed in grid units."""
    import cv2
    plan = json.loads(Path(frames_json).read_text())
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for v, p in plan.items():
        for f in p["reference"]:
            img = cv2.imread(str(Path(jpg_dir) / f"{v}-{f:05d}.jpg"))
            W = 1600 if img.shape[1] / img.shape[0] > 1.5 else 1200
            img = cv2.resize(img, (W, int(W * img.shape[0] / img.shape[1])), interpolation=cv2.INTER_CUBIC)
            h, w = img.shape[:2]
            for k in range(1, 20):
                col = (0, 220, 255) if k % 2 == 0 else (0, 140, 180)
                cv2.line(img, (w * k // 20, 0), (w * k // 20, h), col, 1)
                cv2.line(img, (0, h * k // 20), (w, h * k // 20), col, 1)
                if k % 2 == 0:
                    for (x, y) in ((w * k // 20 + 2, 14), (w * k // 20 + 2, h - 4)):
                        cv2.putText(img, str(k), (x, y), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 255, 255), 1, cv2.LINE_AA)
                    for (x, y) in ((2, h * k // 20 - 3), (w - 22, h * k // 20 - 3)):
                        cv2.putText(img, str(k), (x, y), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.imwrite(str(out_dir / f"look-{v}-{f:05d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 88])


def look_score(run_dir, look_json, extra=None):
    """Own-look items (boxes in 5% grid units [x0, y0, x1, y1], 0-20) -> per method / stack: found when one of its
    masks has its box centre inside the item's box and box IoU >= LOOK_IOU (lenient: the boxes are read off a grid)."""
    look = json.loads(Path(look_json).read_text())
    extra = extra or {}
    sets = {m: [m] for m in SINGLE + LATER} | STACKS | {k: v for k, v in extra.items()}
    rows = {k: [] for k in sets}
    for key, items in look["frames"].items():
        video, f = key.rsplit("-", 1)
        f = int(f)
        v = VIDEOS_CACHE.setdefault(video, Video(run_dir, video))
        H, W = v.hw
        boxes = np.array([[b[0] * W / 20, b[1] * H / 20, b[2] * W / 20, b[3] * H / 20] for b in (it["box"] for it in items)], float)
        got = {}
        for name, parts in sets.items():
            ms = []
            for p in parts:
                if p not in got:
                    x = v.method(f, p)
                    got[p] = x[0] if x is not None else np.zeros((0, H, W), bool)
                ms.append(got[p])
            m = np.concatenate(ms)
            mb = np.array([bbox(x) for x in m], float).reshape(-1, 4)
            iou = box_iou_np(mb, boxes) if len(mb) else np.zeros((0, len(boxes)))
            cx, cy = (mb[:, 0] + mb[:, 2]) / 2, (mb[:, 1] + mb[:, 3]) / 2
            inside = (cx[:, None] >= boxes[None, :, 0]) & (cx[:, None] <= boxes[None, :, 2]) & (cy[:, None] >= boxes[None, :, 1]) & (cy[:, None] <= boxes[None, :, 3])
            hit = ((iou >= LOOK_IOU) & inside).any(0) if len(mb) else np.zeros(len(boxes), bool)
            rows[name] += [(it, bool(h)) for it, h in zip(items, hit)]
    out = {}
    for name, r in rows.items():
        cls = {}
        for it, h in r:
            for c in ("all", it["class"]):
                cls.setdefault(c, []).append(h)
        out[name] = {c: {"found": int(sum(x)), "of": len(x), "share": round(sum(x) / len(x), 3)} for c, x in sorted(cls.items())}
    return out, rows


LOOK_IOU = .3


def box_iou_np(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ix = np.maximum(0, np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]))
    iy = np.maximum(0, np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]))
    inter = ix * iy
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])  # noqa: E731
    return inter / np.maximum(area(a)[:, None] + area(b)[None] - inter, 1e-9)


LETTER_COLOUR = {"W": (0, 220, 0), "T": (0, 220, 0), "K": (0, 180, 255), "P": (200, 200, 200), "B": (0, 0, 255), "M": (255, 0, 255), "X": (128, 128, 128)}


def export(run_dir, jpg_dir, out_dir):
    """The audit, saved small for a human to check: reference items as polygons (evaluation grid), every audited
    cluster's letter and box, audit sheets with each tile's letter stamped, own-look frames with the boxes drawn."""
    import cv2
    run_dir, out_dir = Path(run_dir), Path(out_dir)
    (out_dir / "audit-sheets").mkdir(parents=True, exist_ok=True)
    (out_dir / "own-look").mkdir(exist_ok=True)
    index, reps = load_audit(run_dir)
    ref = reference_items(index, reps)
    items = []
    for (video, f), its in sorted(ref.items()):
        for it in its:
            cs, _ = cv2.findContours(it["mask"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            polys = [c[:, 0].tolist() for c in cs if len(c) >= 3]
            items.append({"id": it["id"], "video": video, "frame": f, "name": it["name"], "kind": it["kind"], "shape": it["shape"], "size": it["size"],
                          "weight": it["weight"], "area_px": int(it["mask"].sum()), "polygons": polys})
    audited = [{"id": r["id"], "letter": r["letter"], "stratum": r["stratum"], "weight": r["weight"], "bbox": r["bbox"], "methods": sorted({m for m, _ in r["members"]}),
                "hint": r["hint"]} for rows in index.values() for r in rows if r.get("letter")]
    hw = {v: json.loads((run_dir / f"{v}.json").read_text())["eval_hw"] for v in index}
    (out_dir / "reference.json").write_text(json.dumps({"agent_labelled": True, "grid_hw": hw, "labels": json.loads((run_dir / "audit/labels.json").read_text())["rules"],
                                                        "items": items, "audited_clusters": audited}))
    # sheets again, letter stamped (same order as the audit)
    for video, rows in index.items():
        z = np.load(run_dir / "audit" / f"clusters-{video}.npz")
        w = hw[video][1]
        pos = {str(c): i for i, c in enumerate(z["ids"])}
        tiles_list = []
        cache = {}
        for r in sorted((r for r in rows if r.get("sampled")), key=lambda r: r["sheet_pos"]):
            img = cache.setdefault(r["frame"], cv2.imread(str(Path(jpg_dir) / f"{video}-{r['frame']:05d}.jpg")))
            t = tile(img, unpack(z["bits"][pos[r["id"]]][None], w)[0], f"{r['frame']}.{r['id'].rsplit('-', 1)[1]} {r['hint']}")
            ch = r.get("letter", "?")
            cv2.rectangle(t, (TILE_W - 22, TILE_H - 22), (TILE_W - 1, TILE_H - 1), (0, 0, 0), -1)
            cv2.putText(t, ch, (TILE_W - 19, TILE_H - 5), cv2.FONT_HERSHEY_SIMPLEX, .6, LETTER_COLOUR.get(ch, (255, 255, 255)), 2)
            tiles_list.append(t)
        for path in sheets(tiles_list, out_dir / "audit-sheets" / f"sheet-{video}-{{:03d}}.jpg"):
            im = cv2.imread(path)
            cv2.imwrite(path, im, [cv2.IMWRITE_JPEG_QUALITY, 62])
    look = json.loads((run_dir / "look.json").read_text())
    (out_dir / "own-look.json").write_text(json.dumps(look, indent=1))
    for key, its in look["frames"].items():
        video, f = key.rsplit("-", 1)
        img = cv2.imread(str(Path(jpg_dir) / f"{video}-{int(f):05d}.jpg"))
        W = 1200 if img.shape[1] / img.shape[0] > 1.5 else 900
        img = cv2.resize(img, (W, int(W * img.shape[0] / img.shape[1])), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]
        for k, it in enumerate(its):
            b = it["box"]
            p0, p1 = (int(b[0] * w / 20), int(b[1] * h / 20)), (int(b[2] * w / 20), int(b[3] * h / 20))
            cv2.rectangle(img, p0, p1, (0, 255, 255), 1)
            cv2.putText(img, f"{k} {it['class']}", (p0[0], max(10, p0[1] - 2)), cv2.FONT_HERSHEY_SIMPLEX, .33, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(out_dir / "own-look" / f"look-{key}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return len(items), len(audited)


def results(scratch, out_dir):
    """Every reported number -> OUT/results.json (reads the scratch run folders: run000 trial, run001 main, stack002 /
    stack004 stack times, words003 word sets)."""
    sc, out_dir = Path(scratch), Path(out_dir)
    r1 = sc / "run001"
    scores = json.loads((r1 / "scores.json").read_text())
    look = json.loads((r1 / "look-scores.json").read_text())
    five = json.loads((r1 / "five_fps.json").read_text())
    meta = json.loads((r1 / "meta.json").read_text())
    plan = json.loads((sc / "frames.json").read_text())
    ram = json.loads((r1 / "ram.json").read_text())
    recs = {v: json.loads((r1 / f"{v}.json").read_text()) for v in plan}
    own = json.loads((r1 / "look.json").read_text())
    stacks = []
    for name in ("stack002", "stack004"):
        d = json.loads((sc / name / "stack-time.json").read_text())
        for r in d["runs"]:
            if "added_s" in r:
                stacks.append({"run": name, "video": r["video"], "components_added_to_core": r["components"], "keyframes": r["keyframes"],
                               "keyframes_kind": r["keyframes_kind"], "added_s": r["added_s"], "component_gpu_thread_s": r["component_gpu_s_sum"],
                               "masks_per_frame": r["masks_per_frame"], "gpu_peak_gb": [g["peak_gb"] for g in r["gpu_peak"]], "flags_over_90": r["flags"],
                               "core_state_s_not_added": r["pre_core_state_s_not_added"]})
    by_stack = {}
    for r in stacks:
        by_stack.setdefault(("+".join(r["components_added_to_core"]), r["keyframes_kind"]), []).append(r["added_s"])
    words = json.loads((sc / "words003" / "words-time.json").read_text())
    ridge2 = json.loads((sc / "stack002" / "stack-time.json").read_text())
    ridge2_s = [t for r in ridge2["runs"] if "ridge2_per_frame_s" in r for t in r["ridge2_per_frame_s"].values()]
    later_s = {k: round(float(np.median([t for v in words["videos"].values() for t in v[k].values()])), 4) for k in words["sets"]}
    later_s["ridge2 (embed + filter + multimask decode)"] = round(float(np.median(ridge2_s)), 4)
    gpu = {v: {"stages": [{"stage": st["stage"], "s": st["s"], "n": st.get("n"), "peak_gb": [round(x, 2) for x in st["peak_gb"]], "over_90": st["over_90"]}
                          for st in rec["timing"]["stages"]], "gpu_peak": rec["timing"]["gpu_peak"], "flags": rec["timing"]["flags"],
               "torch_reserved_peak_gb": rec["torch_reserved_peak_gb"]} for v, rec in recs.items()}
    spend = {"run000_trial_lightning": 1.559, "run001_all_methods": meta["usd_upper"], "stack002_times_ridge2": ridge2["usd_upper"],
             "words003_label_and_tiles": words["usd_upper"], "stack004_recommended_stack": json.loads((sc / "stack004" / "stack-time.json").read_text())["usd_upper"],
             "setup_and_ram_probes_estimated": .45}
    spend["total_usd_upper"] = round(sum(spend.values()), 2)
    res = {
        "experiment": "X10 discover-all: can we segment 'everything' without categories?", "branch": "fx/x10-discover-all",
        "policy": {"scale": "estimated (floor plane + assumed 1.6 m camera height) wherever metres appear (geometry lane only)",
                   "agent_labelled": "the reference letters and the own-look boxes are agent-labelled (Claude looking at images): reference.json, own-look.json, audit-sheets/*.jpg (letter stamped bottom right of each tile), own-look/*.jpg",
                   "observed_vs_inferred": "observed: every recall, precision, per-frame and stack time below; inferred: lines marked 'inferred'"},
        "setup": {"videos": {v: {"video": p["video"], "size_wh": p["size_wh"], "frames": p["frames"], "object_keyframes": len(p["object_keys"]),
                                 "reference_keyframes": p["reference"], "reference_t_s": p["reference_t_s"]} for v, p in plan.items()},
                  "keyframe_rule": "the core's: 5 fps = sharpest of each 6-frame block, object keyframes = every 3rd; 12 reference keyframes spread evenly over the object keyframes; 5 fps neighbours = the 5 fps keyframe before and after",
                  "evaluation_grid": "half resolution (640 x 360) for the 1280 x 720 clips, native 640 x 480 for lightning; every method's masks resized there (area > 0.3)",
                  "methods_as_used": {"score floors": FLOORS, "SAM 3 word methods": "the core's dedupe rule (segment.dedupe: IoU > 0.8 across words, >= 0.5 inside within a word)",
                                      "all": f"people out (>= {PERSON_INSIDE} inside a SAM 3 person mask), masks < {MIN_PX} px or > {MAX_SHARE} of the frame dropped"},
                  "hardware": meta["boot"]["gpus"], "cold_start_s_not_analysis": {k: v for k, v in meta["boot"].items() if k.endswith("_s")},
                  "resident_gb_after_boot": meta["boot"]["resident_gb"], "licences": meta["licences"],
                  "ram_plus": {"load_s": ram["load_s"], "note": ram["note"]}},
        "reference_pool": {**scores["reference"], "how": "every method's masks (as used) on the 48 reference frames, clustered at IoU >= 0.6 (19,575 clusters); a seeded stratified sample audited on contact sheets (thin-shaped clusters at 30%, the rest at 12%): W whole thing, T thin whole thing, K block of repeated things, P part, B background, M mixture, X unclear; W/T/K = reference items, weight 1/p",
                           "matching": f"found = a method mask with IoU >= {FOUND_IOU} with the item (thin items: 1 px dilation on both), on a stride-{GRID} grid; recall weighted by 1/p",
                           "bias": "the pool is made of the methods' own masks: it cannot contain what no method outlined (see own_look), and it favours methods with many masks; the four later methods (ridge2, labelw, labelw_t, generic_t) were not in the pool: their pool recall is a lower bound"},
        "own_look": {"frames": len(own["frames"]), "items": sum(len(v) for v in own["frames"].values()),
                     "by_class": {c: sum(1 for v in own["frames"].values() for it in v if it["class"] == c) for c in ("cable", "tool", "label", "floor", "small")},
                     "rule": own["rule"], "matching": "found = a mask whose box centre is inside the look box and box IoU >= 0.3 (the look boxes are read off a 5% grid)",
                     "note": "independent of every method's masks: the completeness check of the pool reference"},
        "recall_pool_reference": scores["recall"], "recall_pool_reference_by_video": scores["recall_by_video"],
        "recall_own_look": look,
        "precision_pool_audit": {**scores["precision"], "_rule": "share of each method's masks (weighted by the audit sample) whose cluster was labelled whole (W/T/K), part (P), background (B), group/mixture (M) or unclear (X)"},
        "time_per_frame_one_a100": {"parts_median_s": scores["time_parts_median_s"], "per_method": scores["time_per_frame"], "later_methods_median_s": later_s,
                                    "note": "median over the 36 frames per video (48 for AMG 16 / c1, listing and geometry); 'added' = minus what the core already runs on its keyframes (SAM 3 vision features, DA3 depth, the vocabulary words' detection)"},
        "stack_added_s_two_a100": {"runs": stacks, "summary": {f"{k[0]} | {k[1]}": {"min": min(v), "max": max(v), "videos": len(v)} for k, v in by_stack.items()},
                                   "note": "t0 = the core's state ready (keyframes on their GPU, SAM 3 features, person/floor): wall seconds until both GPUs finish; 2 threads, frames alternate; no MPS"},
        "five_fps_neighbours": {**five, "rule": "a reference item counts as found when found on its keyframe or on the 5 fps keyframe before / after, those masks warped onto the keyframe by DIS optical flow (lower bound: warping blurs thin masks)"},
        "gpu_memory_run001": gpu,
        "spend_usd_upper": spend,
    }
    (out_dir / "results.json").write_text(json.dumps(res, indent=1))
    return res


def self_check():
    a = np.zeros((4, 20, 20), bool)
    a[0, 2:10, 2:10] = True
    a[1, 2:10, 2:11] = True    # ~ a[0]
    a[2, 12:18, 12:18] = True
    a[3, 2:6, 2:6] = True      # inside a[0]
    iou, inside = iou_matrix(a, a, grid=1)
    assert abs(iou[0, 1] - 64 / 72) < 1e-6 and inside[3, 0] == 1 and iou[0, 2] == 0
    cl = cluster(a, ["x", "y", "x", "x"], iou=.6)
    assert sorted(sorted(m) for _, m in cl) == [[0, 1], [2], [3]], cl
    assert cl[0][0] in (0, 1) and set(cl[0][1]) == {0, 1}
    line = np.zeros((1, 10, 10), bool)
    line[0, 5, :] = True
    assert dilate(line)[0, 4:7, :].all() and dilate(line).sum() == 30
    ref = np.zeros((3, 40, 40), bool)
    ref[0, 2:12, 2:12] = True                # box
    ref[1, 20:30, 2:12] = True               # box
    ref[2, 35, 5:35] = True                  # 1 px cable
    m = np.zeros((6, 40, 40), bool)
    m[0, 2:12, 2:12] = True                  # whole
    m[1, 3:6, 3:6] = True                    # part
    m[2, 0:32, 0:14] = True                  # group (both boxes)
    m[3, 30:40, 30:40] = True                # background (cable end grazes it)
    m[4, 36, 5:35] = True                    # cable 1 px off: tolerant IoU finds it
    m[5, 20:30, 8:20] = True                 # half on a box: other
    best, cats = classify(m, ref, np.array([False, False, True]))
    assert list(cats) == ["whole", "part", "group", "background", "whole", "other"], cats
    assert best[0] == 1 and best[1] < .5 and best[2] >= .5, best
    wide = np.zeros((360, 640), bool)
    wide[100:103, 50:300] = True             # a 3 px wide, 250 px long cable at 640 wide
    blob = np.zeros((360, 640), bool)
    blob[10:25, 10:25] = True
    assert shape_class(wide, "") == "thin" and shape_class(blob, "") == "small" and shape_class(ref[0], "box") == "normal"
    dm = np.zeros((4, 20, 20), bool)
    dm[0, :10, :10] = dm[1, :10, :10] = dm[2, 2:5, 2:5] = dm[3, 2:5, 2:5] = True
    assert list(dedupe_mask(dm, np.array([.9, .8, .7, .6]), ["a", "b", "a", "b"], np.ones(4, bool))) == [True, False, False, True]
    assert shape_class(blob, "power cord") == "thin" and size_class(.0001) == "tiny" and size_class(.06) == "large"
    print("x10_eval self-check ok")


if __name__ == "__main__":
    if sys.argv[1:] == ["--self-check"]:
        self_check()
    elif sys.argv[1] == "score":
        res, ref = score(sys.argv[2], sys.argv[3], sys.argv[4])
        if "--5fps" in sys.argv:
            res["five_fps"] = five_fps_run(sys.argv[2], sys.argv[3], sys.argv[4], ref)
        (Path(sys.argv[2]) / "scores.json").write_text(json.dumps(res, indent=1))
        print(json.dumps({k: v for k, v in res.items() if k not in ("items", "missed_examples", "recall_by_video")}, indent=0)[:12000])
    elif sys.argv[1] == "results":
        r = results(sys.argv[2], sys.argv[3])
        print(json.dumps(r["stack_added_s_two_a100"]["summary"], indent=1), json.dumps(r["spend_usd_upper"]))
    elif sys.argv[1] == "export":
        print(export(sys.argv[2], sys.argv[3], sys.argv[4]))
    elif sys.argv[1] == "look-views":
        look_views(sys.argv[2], sys.argv[3], sys.argv[4])
    elif sys.argv[1] == "uncovered":
        print(uncovered(sys.argv[2], sys.argv[3], sys.argv[4].split(",") if len(sys.argv) > 4 else None))
    elif sys.argv[1] == "pool":
        rate = {k: float(x) for k, x in (kv.split(":") for kv in sys.argv[4].split(","))}  # e.g. thin:0.3,rest:0.12
        pool(sys.argv[2], sys.argv[3], rate, sys.argv[5].split(",") if len(sys.argv) > 5 else None)
