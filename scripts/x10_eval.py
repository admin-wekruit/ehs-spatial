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
FLOORS = {"vocab": .3, "generic": .3, "ram": .3, "owlw": .3, "aux": .3}
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

    def get(self, f, m):
        k = f"{f}/{m}/bits"
        if k not in self.z:
            return None
        bits = self.z[k]
        masks = unpack(bits, self.hw[1]) if len(bits) else np.zeros((0, *self.hw), bool)
        return masks, self.z[f"{f}/{m}/score"], [str(x) for x in self.z[f"{f}/{m}/label"]]

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


WORD_METHODS = {"vocab", "generic", "ram", "owlw"}
FLOOD_IOU, INSIDE_SAME = .8, .5  # fast_report.segment: FLOOD_IOU, INSIDE


def dedupe_mask(masks, score, words, valid):
    """fast_report.segment.dedupe in numpy, one frame: best score first; a mask with IoU > 0.8 with a kept mask of
    another word, or at least half inside a kept mask of the same word, is dropped. -> keep flags."""
    idx = np.flatnonzero(valid)
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

POOL_FLOORS = {"vocab": .2, "generic": .2, "ram": .2, "owlw": .2}  # the pool is wider than any one method as used
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
    pad = max(40, int(.5 * max(x1 - x0, y1 - y0)))
    cx0, cy0, cx1, cy1 = max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(H, y1 + pad)
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


def pool(run_dir, jpg_dir):
    """Clusters per reference frame (saved: RUN_DIR/audit/clusters-<video>.npz + clusters.json) and audit sheets."""
    import cv2
    run_dir, jpg_dir = Path(run_dir), Path(jpg_dir)
    audit = run_dir / "audit"
    audit.mkdir(exist_ok=True)
    index = {}
    for vp in sorted(run_dir.glob("*.npz")):
        v = Video(run_dir, vp.stem)
        reps, tiles_list = {}, []
        for f in v.rec["reference"]:
            masks, src = frame_pool(v, f)
            cl = cluster(masks, [FAMILY[s[0]] for s in src])
            img = cv2.imread(str(jpg_dir / f"{v.name}-{f:05d}.jpg"))
            stuff = v.aux(f, "floor", .3) | v.aux(f, "wall", .3) | v.aux(f, "ceiling", .3)
            rows = []
            for k, (rep, members) in enumerate(cl):
                cid = f"{v.name}-{f}-{k}"
                area = int(masks[rep].sum())
                row = {"id": cid, "frame": f, "area_px": area, "share": round(area / masks[rep].size, 5),
                       "families": sorted({FAMILY[src[j][0]] for j in members}), "methods": sorted({src[j][0] for j in members}),
                       "hint": hint(members, src), "stuff_inside": round(float((masks[rep] & stuff).sum() / max(area, 1)), 3),
                       "bbox": [int(x) for x in bbox(masks[rep])]}
                rows.append(row)
                reps[cid] = masks[rep]
            # sheet order: big first within a frame, so parts sit after their whole
            for row in sorted(rows, key=lambda r: -r["area_px"]):
                tiles_list.append(tile(img, reps[row["id"]], f"{row['id'].split('-', 1)[1]} {row['hint']}"))
                row["sheet_pos"] = len(tiles_list) - 1
            index.setdefault(v.name, []).extend(rows)
            print(v.name, f, "pool", len(masks), "clusters", len(cl), flush=True)
        ids = list(reps)
        np.savez_compressed(audit / f"clusters-{v.name}.npz", ids=np.array(ids), bits=np.packbits(np.array([reps[i] for i in ids]), axis=-1))
        paths = sheets([t for t in tiles_list], audit / f"sheet-{v.name}-{{:03d}}.jpg")
        print(v.name, len(tiles_list), "tiles,", len(paths), "sheets", flush=True)
    (audit / "clusters.json").write_text(json.dumps(index, indent=0))


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


def load_reference(run_dir):
    """audit/labels.json ({'sheets': {sheet name: 48-letter string}, 'added': [...]}) + clusters -> {(video, frame):
    [item]}; item = {id, mask, kind, name, thin, size, shape}. Two reference items with IoU >= 0.5 on one frame are one
    (the bigger kept): the pool's 0.6 clustering can leave a thing twice."""
    audit = Path(run_dir) / "audit"
    labels = json.loads((audit / "labels.json").read_text())
    index = json.loads((audit / "clusters.json").read_text())
    ref = {}
    letters = {}
    for video, rows in index.items():
        by_pos = {r["sheet_pos"]: r for r in rows}
        per = COLS * ROWS
        for sheet, text in labels["sheets"].items():
            v, page = sheet.rsplit("-", 1)
            if v != video:
                continue
            text = text.replace(" ", "")
            for k, ch in enumerate(text):
                assert ch in AUDIT_LETTERS, (sheet, k, ch)
                row = by_pos.get(int(page) * per + k)
                if row is not None:
                    letters[row["id"]] = ch
        z = np.load(audit / f"clusters-{video}.npz")
        hw = None
        ids = [str(x) for x in z["ids"]]
        w = int(json.loads((Path(run_dir) / f"{video}.json").read_text())["eval_hw"][1])
        masks = unpack(z["bits"], w)
        pos = {i: n for n, i in enumerate(ids)}
        for r in rows:
            ch = letters.get(r["id"])
            if ch in REF_LETTERS:
                m = masks[pos[r["id"]]]
                name = labels.get("names", {}).get(r["id"], r["hint"])
                ref.setdefault((video, r["frame"]), []).append({"id": r["id"], "mask": m, "kind": REF_LETTERS[ch], "name": name,
                                                               "thin": ch == "T", "support": len(r["families"])})
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
            share = it["mask"].sum() / it["mask"].size
            it["size"] = size_class(share)
            it["shape"] = shape_class(it["mask"], it["name"], True if it["thin"] else None)
    unlabelled = [r["id"] for rows in index.values() for r in rows if r["id"] not in letters]
    return ref, letters, labels.get("added", []), unlabelled


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


def score_frames(run_dir, ref, methods, floors=None, stacks=None):
    """-> per method: item hits (list of (video, frame, item index, found, best iou)), mask categories."""
    floors = floors or {}
    stacks = stacks or {}
    res = {}
    for (video, f), items in sorted(ref.items()):
        v = VIDEOS_CACHE.setdefault(video, Video(run_dir, video))
        rmask = np.array([it["mask"] for it in items])
        thin = np.array([it["shape"] == "thin" for it in items])
        got = {}
        for m in methods:
            x = v.method(f, m, floors.get(m))
            got[m] = x[0] if x is not None else np.zeros((0, *v.hw), bool)
        for name, parts in stacks.items():
            got[name] = np.concatenate([got[p] if p in got else v.method(f, p, floors.get(p))[0] for p in parts])
        for m, masks in got.items():
            best, cats = classify(masks, rmask, thin)
            r = res.setdefault(m, {"items": [], "cats": [], "masks": 0})
            r["items"] += [(video, f, i, bool(best[i] >= FOUND_IOU), float(best[i])) for i in range(len(items))]
            r["cats"] += list(cats)
            r["masks"] += len(masks)
    return res


VIDEOS_CACHE = {}


def summarise(res, ref, n_frames):
    """recall overall / by size / by shape / by video, precision categories, masks per frame."""
    out = {}
    for m, r in res.items():
        rows = {"all": [], **{f"size:{k}": [] for k in ("tiny", "small", "medium", "large")}, **{f"shape:{k}": [] for k in ("thin", "small", "normal")},
                **{f"kind:{k}": [] for k in ("whole", "thin", "block")}}
        by_video = {}
        for video, f, i, found, _ in r["items"]:
            it = ref[(video, f)][i]
            for key in ("all", f"size:{it['size']}", f"shape:{it['shape']}", f"kind:{it['kind']}"):
                rows[key].append(found)
            by_video.setdefault(video, []).append(found)
        cats = np.array(r["cats"])
        n = max(len(cats), 1)
        out[m] = {"recall": {k: {"found": int(sum(v)), "of": len(v), "share": round(sum(v) / max(len(v), 1), 3)} for k, v in rows.items()},
                  "recall_by_video": {k: round(sum(v) / max(len(v), 1), 3) for k, v in by_video.items()},
                  "masks_per_frame": round(r["masks"] / n_frames, 1),
                  "precision": {c: round(float((cats == c).sum()) / n, 3) for c in ("whole", "part", "group", "background", "other")}}
    return out


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
    elif sys.argv[1] == "pool":
        pool(sys.argv[2], sys.argv[3])
