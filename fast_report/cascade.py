"""Object naming cascade, the VLM last (r4/naming: the X13 memo with its verifier corrections; the user's round-4 direction:
find the objects, say what each one is, and when that cannot be settled keep it as 'an object' with its physical card).

Per object card, cheapest first:
  1. votes: the SAM 3 word (free: its canonical class's share of the card's mask votes), the YOLOE-26L class matched to the
     card's masks (box IoU >= MATCH_IOU on its views), PE-Core-L zero-shot over the canonical taxonomy (masked crops), and the
     label bank's DINOv2-L k-NN over earlier VLM answers of OTHER videos of the same domain family (plus this video's own
     VLM answers from an earlier pass)
  2. accept only when a second, independent vote gives the SAM 3 word's class, at the thresholds fitted on the family's
     other videos (naming_calibration.json; folds[site] never saw that video). A family with no other audited video is
     uncalibrated: nothing is accepted cheaply there (ME340 at the pooled 0.80 point agreed with the VLM 0.23, x13)
  3. the rest: clustered per pass (DINOv2-L, average linkage, the fitted cosine cut), ONE VLM question per cluster medoid
     (the local Qwen3-VL-8B decider over the cluster's cheap candidates, sizes struck by the measured geometry); the answer
     is copied only to members whose SAM 3 word gives the same class
  4. everything else stays 'unidentified object': its physical card is complete; its words stay unverified candidates
  5. the bank grows by the VLM's answers only (cheap names never enter it); a row carries video, site and family. r5b: the bank a
     run reads is a frozen snapshot (read only, its sha256 in the run's record); a run's VLM rows go to its report folder
     (bank-rows-<pass>.npz) and join a new snapshot offline only (scripts/r5b_vocab.py bank)

Crops follow x13's recipe (scripts/x13_naming.tight_crop, Encoder.image): the mask's box + 10 % a side, squared, grey
outside the frame; 'masked' = grey outside the mask; up to K_VIEWS views (PE-Core: PE_VIEWS); a card vector is the sum of
its unit view vectors (plain and masked each renormalised, then added), so the pipeline's vectors meet the bank's.
"""
import json
import threading
import time
from pathlib import Path

import numpy as np

from fast_report import cards

DINO, PE_CORE, X13_HF = "facebook/dinov2-large", "hf-hub:timm/PE-Core-L-14-336", "/v/x13/hf"
YOLO_PT = "/v/r4/yolo/yoloe-26l-seg-taxonomy.pt"  # modal_apps/r4_naming.py bakes the taxonomy in (AGPL-3.0, accepted for now)
# r5b: the frozen snapshot every run reads (never written by a run; scripts/r5b_vocab.py bank builds the next one offline). Its
# 'classes' / 'text' are the zero-shot's (every taxonomy class, x13's prompt ensemble); round 4's growing file was dinov2-l-v1.npz
BANK = "/v/layers/label-bank/frozen/r5b-A.npz"
CALIBRATION = Path(__file__).with_name("naming_calibration.json")
K_VIEWS, PE_VIEWS, CROP_PAD, BATCH = 5, 3, .1, 64
DINO_SIDE, PE_SIDE, DINO_MEAN, DINO_STD = 224, 336, (.485, .456, .406), (.229, .224, .225)
K, TAU = 10, .05                                    # bank k-NN (x13: set before any score was seen)
YOLO_IMGSZ, YOLO_CONF, MATCH_IOU = 960, .05, .5
RULES = ("sam3+bank", "sam3+zero-shot/yolo")
MAX_OPTIONS = 8
# r5b: the zero-shot's classes are the bank snapshot's own ('classes' beside 'text': every taxonomy class, r5b-A on); YOLOE's are the
# names baked into its weights (r4: the 102 of mvp2's taxonomy, the model's own list). Both only name taxonomy classes.
# ponytail: YOLOE is not re-baked for the classes added since ('wrap', 'curtain', 'machine tool holder'): they get no YOLOE vote.
CLASSES = [c for fam in cards.TAXONOMY.values() for c in fam]  # a new snapshot's zero-shot classes (scripts/r5b_vocab.py bank)
FAMILIES = list(cards.TAXONOMY)
VLM_SOURCE = "qwen3-vl-8b decider (cluster medoid)"


# ---------------------------------------------------------------- models (GPU)

class Encoders:
    """DINOv2-L, PE-Core-L's vision tower and YOLOE-26L (the taxonomy baked in) resident on one GPU. The taxonomy's PE-Core
    text embeddings come with the bank file (x13's prompt ensemble, computed once)."""

    def __init__(self, dev):
        import os
        import cv2
        import open_clip
        import torch
        from transformers import AutoModel
        os.environ.setdefault("YOLO_OFFLINE", "1")
        threads = cv2.getNumThreads()
        from ultralytics import YOLOE
        cv2.setNumThreads(threads)  # ultralytics sets 0 on import: the core's own cv2 work keeps its threads
        self.dev, self.torch = dev, torch
        self.dino = AutoModel.from_pretrained(DINO, cache_dir=X13_HF, torch_dtype=torch.bfloat16).to(dev).eval()
        pe, _, pre = open_clip.create_model_and_transforms(PE_CORE, cache_dir=X13_HF)
        self.pe = pe.to(dev).eval().to(torch.bfloat16)
        norm = [t for t in pre.transforms if type(t).__name__ == "Normalize"][0]
        self.pe_mean, self.pe_std = tuple(norm.mean), tuple(norm.std)
        self.yolo = YOLOE(YOLO_PT)
        self.yolo_names = self.yolo.names if isinstance(self.yolo.names, list) else [self.yolo.names[i] for i in range(len(self.yolo.names))]
        assert all(n in cards.FAMILY for n in self.yolo_names), "YOLOE's baked classes are not taxonomy classes"

    def _norm(self, x, mean, std):
        t = self.torch
        return ((x - t.tensor(mean, device=x.device)[:, None, None]) / t.tensor(std, device=x.device)[:, None, None]).to(t.bfloat16)

    def dino_embed(self, x):
        """(B,3,224,224) float RGB [0,1] -> (B,D) unit."""
        with self.torch.inference_mode():
            return self.torch.nn.functional.normalize(self.dino(pixel_values=self._norm(x, DINO_MEAN, DINO_STD)).pooler_output.float(), dim=-1)

    def pe_embed(self, x):
        with self.torch.inference_mode():
            return self.torch.nn.functional.normalize(self.pe.encode_image(self._norm(x, self.pe_mean, self.pe_std)).float(), dim=-1)

    def detect(self, bgr_frames):
        """[BGR uint8 frame] -> [(n,6) array: x0, y0, x1, y1 (source px), class index, score]."""
        out = []
        for s in range(0, len(bgr_frames), 16):
            for r in self.yolo.predict(bgr_frames[s:s + 16], imgsz=YOLO_IMGSZ, conf=YOLO_CONF, half=True, verbose=False, max_det=300,
                                       device=f"cuda:{self.dev.index}"):
                b = r.boxes
                out.append(np.concatenate([b.xyxy.cpu().numpy(), b.cls.cpu().numpy()[:, None], b.conf.cpu().numpy()[:, None]], 1)
                           if len(b) else np.zeros((0, 6), np.float32))
        return out

    def warm(self):
        t = self.torch
        x = t.rand((8, 3, DINO_SIDE, DINO_SIDE), device=self.dev)
        self.dino_embed(x)
        self.pe_embed(t.nn.functional.interpolate(x, size=(PE_SIDE, PE_SIDE)))
        self.detect([np.random.default_rng(0).integers(0, 255, (720, 1280, 3), np.uint8)] * 2)
        t.cuda.synchronize(self.dev)


def pick_views(area, cut, frame, k=K_VIEWS):
    """Member masks of one object -> up to k of them: largest first (a mask touching the frame edge counts half, x13's
    pick_views), one per keyframe. Returns indices into the inputs."""
    order = np.argsort(-(np.asarray(area, float) * np.where(cut, .5, 1.)), kind="stable")
    out, seen = [], set()
    for i in order:
        if frame[i] not in seen:
            seen.add(frame[i])
            out.append(int(i))
            if len(out) == k:
                break
    return out


def signals(enc, frames, frame_of, masks, owner, n_obj, frame_bgr):
    """Per-object naming signals from its SAM 3 masks.
    frames (N,H,W,3) uint8 BGR (any GPU); frame_of (M,) index into frames; masks (M,h,w) bool (a grid over the same view,
    on the frames' GPU); owner (M,) object index; frame_bgr(i) -> BGR numpy frame i (YOLOE's input). Crops are cut on the
    frames' GPU, the encoders run on enc.dev.
    -> {"dino": (n,D) float32 (zero rows: no view), "pe": (n,D), "yolo": [{class: score}], "views": [[frame i]], "record"}"""
    import torch
    from torchvision.ops import roi_align
    dev0 = masks.device
    (h, w), (H, W) = masks.shape[1:], frames.shape[1:3]
    frame_of, owner = np.asarray(frame_of), np.asarray(owner)
    rows, cols = masks.any(2), masks.any(1)
    ys, xs = torch.arange(h, device=dev0), torch.arange(w, device=dev0)
    box = torch.stack([torch.where(cols, xs, w).amin(1), torch.where(rows, ys, h).amin(1),
                       torch.where(cols, xs, -1).amax(1) + 1, torch.where(rows, ys, -1).amax(1) + 1], 1).float().cpu().numpy()
    area = masks.sum((1, 2)).cpu().numpy()
    sx, sy = W / w, H / h
    src = box * [sx, sy, sx, sy]  # the mask's box in source px
    cut = (box[:, 0] <= 0) | (box[:, 1] <= 0) | (box[:, 2] >= w) | (box[:, 3] >= h)
    by_obj = {}
    for i, o in enumerate(owner):
        by_obj.setdefault(int(o), []).append(i)
    views = [[] for _ in range(n_obj)]
    for o, idx in by_obj.items():
        views[o] = [idx[j] for j in pick_views(area[idx], cut[idx], frame_of[idx])]
    sel = np.array([i for v in views for i in v], int)
    sel_owner = np.array([o for o, v in enumerate(views) for _ in v], int)
    first_pe = np.array([j < PE_VIEWS for v in views for j in range(len(v))], bool)
    D = enc.dino.config.hidden_size
    dino_p, dino_m, pe_m = (np.zeros((n_obj, D), np.float32) for _ in range(3))

    def crop(ix, side):
        """Crops of masks ix at side x side on enc.dev: plain RGB [0,1] (grey outside the frame), masked (grey outside the mask)."""
        c = src[ix]
        cx, cy = (c[:, 0] + c[:, 2]) / 2, (c[:, 1] + c[:, 3]) / 2
        half = np.maximum(c[:, 2] - c[:, 0], c[:, 3] - c[:, 1]).clip(8) * (1 + 2 * CROP_PAD) / 2
        boxes = torch.tensor(np.stack([cx - half, cy - half, cx + half, cy + half], 1), dtype=torch.float32, device=dev0)
        fi, inv = np.unique(frame_of[ix], return_inverse=True)
        img = frames[torch.tensor(fi, device=dev0)].permute(0, 3, 1, 2).flip(1).float().div_(255)
        rb = torch.cat([torch.tensor(inv, dtype=torch.float32, device=dev0)[:, None], boxes], 1)
        x = roi_align(img, rb, side, 1., sampling_ratio=-1, aligned=True)  # adaptive sampling: area-like, no aliasing
        inside = roi_align(torch.ones_like(img[:, :1]), rb, side, 1., sampling_ratio=2, aligned=True) > .5
        x = torch.where(inside, x, .5)
        mb = torch.cat([torch.arange(len(ix), device=dev0, dtype=torch.float32)[:, None], boxes / torch.tensor([sx, sy, sx, sy], device=dev0)], 1)
        mk = roi_align(masks[torch.tensor(ix, device=dev0)][:, None].float(), mb, side, 1., sampling_ratio=2, aligned=True) > .5
        x = x.to(enc.dev)
        return x, torch.where(mk.to(enc.dev), x, .5)
    t0 = time.perf_counter()
    for s in range(0, len(sel), BATCH):
        x, xm = crop(sel[s:s + BATCH], DINO_SIDE)
        np.add.at(dino_p, sel_owner[s:s + BATCH], enc.dino_embed(x).cpu().numpy())
        np.add.at(dino_m, sel_owner[s:s + BATCH], enc.dino_embed(xm).cpu().numpy())
    pe_sel, pe_owner = sel[first_pe], sel_owner[first_pe]
    for s in range(0, len(pe_sel), BATCH):
        _, xm = crop(pe_sel[s:s + BATCH], PE_SIDE)
        np.add.at(pe_m, pe_owner[s:s + BATCH], enc.pe_embed(xm).cpu().numpy())
    t1 = time.perf_counter()
    uf = sorted(set(frame_of[sel].tolist()))
    dets = dict(zip(uf, enc.detect([frame_bgr(i) for i in uf])))
    t2 = time.perf_counter()
    yolo = [{} for _ in range(n_obj)]
    for o, v in enumerate(views):
        acc = {}
        for i in v:
            best = {}
            for x0, y0, x1, y1, c, p in dets[int(frame_of[i])]:
                if iou((x0, y0, x1, y1), src[i]) >= MATCH_IOU:
                    n = enc.yolo_names[int(c)]
                    best[n] = max(best.get(n, 0.), float(p))
            for c, p in best.items():
                acc[c] = acc.get(c, 0.) + p / len(v)
        yolo[o] = {c: round(p, 4) for c, p in sorted(acc.items(), key=lambda x: -x[1])[:5]}
    rec = {"views": int(len(sel)), "pe_views": int(first_pe.sum()), "yolo_frames": len(uf), "encode_s": round(t1 - t0, 3), "yolo_s": round(t2 - t1, 3)}
    return {"dino": unit(unit(dino_p) + unit(dino_m)), "pe": unit(pe_m), "yolo": yolo, "views": [[int(frame_of[i]) for i in v] for v in views],
            "record": rec}


def unit(a):
    n = np.linalg.norm(a, axis=-1, keepdims=True)
    return np.where(n > 0, a / np.maximum(n, 1e-12), 0.).astype(np.float32)


def iou(a, b):
    ix = max(0., min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0., min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


# ---------------------------------------------------------------- the bank (CPU)

class Bank:
    """A frozen snapshot of VLM answers keyed by DINOv2-L card vectors, with the zero-shot's PE-Core text embeddings ('classes',
    'text'). Read only (r5b: round 4's runs grew one shared file during the benches, 1798 -> 1998 rows): `sha256` names what a run
    read. Lookups use the rows of the same domain family, never this video's (its own earlier answers come in as settle()'s
    `in_video` rows) and, with exclude_site (a site or several; the benches: no row of the scored site), never that site's. A
    row's label follows the current taxonomy (label_of its name)."""

    def __init__(self, path=BANK, exclude_site=None):
        import hashlib
        import io
        raw = Path(path).read_bytes()
        self.path, self.sha256 = str(path), hashlib.sha256(raw).hexdigest()
        self.exclude = {exclude_site} if isinstance(exclude_site, str) else set(exclude_site or ())
        z = np.load(io.BytesIO(raw))
        self.emb, self.meta = z["emb"].astype(np.float32), json.loads(str(z["meta"]))
        for m in self.meta:
            m["label"] = label_of(m.get("name")) or m.get("label")
        self.text, self.scale = z["text"].astype(np.float32), float(z["text_scale"])
        self.classes = [str(c) for c in z["classes"]]
        assert len(self.classes) == len(self.text) and all(c in cards.FAMILY for c in self.classes), "zero-shot classes: taxonomy classes"
        self.family_of = np.array([FAMILIES.index(cards.FAMILY[c]) for c in self.classes])  # zero-shot summed per family
        self.rows_at_load = len(self.meta)

    def rows(self, family, video):
        keep = [i for i, m in enumerate(self.meta) if m["family"] == family and m["video"] != video and m.get("site") not in self.exclude]
        return self.emb[keep], [self.meta[i] for i in keep]

    def zero_shot(self, pe):
        z = self.scale * np.asarray(pe, np.float32) @ self.text.T
        z = np.exp(z - z.max(-1, keepdims=True))
        return z / z.sum(-1, keepdims=True)

    def record(self):
        return {"path": self.path, "sha256": self.sha256, "rows": self.rows_at_load, "zero_shot_classes": len(self.classes),
                "exclude_sites": sorted(self.exclude), "read_only": True}


def write_rows(path, rows):
    """A pass's bank rows [(vector, meta)] -> one npz beside the report (offline write-back only: scripts/r5b_vocab.py bank)."""
    if rows:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, emb=np.stack([np.asarray(e, np.float32) for e, _ in rows]).astype(np.float16), meta=json.dumps([m for _, m in rows]))
    return len(rows)


def calibration(site, family, path=CALIBRATION):
    """The thresholds for this video: folds[site] (fitted without it: the held-out runs), else the family's, else the
    uncalibrated rule (no cheap acceptance)."""
    c = json.loads(Path(path).read_text())
    f = c["folds"].get(site) or c["families"].get(family) or c["uncalibrated"]
    return {**{r: (float("inf") if f.get(r) is None else float(f[r])) for r in RULES}, "cluster_cut": float(f.get("cluster_cut") or 0.),
            "uncalibrated": bool(f.get("uncalibrated")), "trained_on": f.get("trained_on"),
            "source": "folds (held out)" if site in c["folds"] else "family" if family in c["families"] else "uncalibrated"}


# ---------------------------------------------------------------- decisions (CPU)

def label_of(name):
    """A free name -> the bank's label key: the canonical class, else '~' + the head noun (x13)."""
    if not name:
        return None
    c = cards.canonical(name)
    if c:
        return c
    h = cards.norm(name.split(" of ")[0]).split()
    return "~" + h[-1] if h else None


def sam3_vote(votes):
    """SAM 3 word votes -> (label, its share, its best word); (None, 0, None) when there are none."""
    v, best = {}, {}
    for w, s in (votes or {}).items():
        lab = label_of(w)
        if lab and lab != cards.NOT_OBJECT:
            v[lab] = v.get(lab, 0.) + float(s)
            if s > best.get(lab, ("", -1.))[1]:
                best[lab] = (w, s)
    if not v:
        return None, 0., None
    top = max(v, key=v.get)
    return top, v[top] / sum(v.values()), best[top][0]


def knn(q, B, labels, k=K, tau=TAU):
    """Similarity-weighted k-NN vote (x13) -> {label, share, ranked, nn (label -> row)}."""
    if not len(B) or not np.any(q):
        return {"label": None, "share": 0., "ranked": [], "nn": {}}
    s = B @ q
    top = np.argsort(-s)[:k]
    wt = {}
    for j in top:
        wt[labels[j]] = wt.get(labels[j], 0.) + float(np.exp((s[j] - 1) / tau))
    ranked = sorted(wt, key=lambda x: -wt[x])
    return {"label": ranked[0], "share": wt[ranked[0]] / sum(wt.values()), "ranked": ranked, "cos1": float(s[top[0]]),
            "nn": {lab: int(next(j for j in top if labels[j] == lab)) for lab in ranked}}


def tier1(v, th):
    """The first rule whose second vote agrees with the SAM 3 class at its threshold -> (rule, label) or (None, None)."""
    lab, share, _ = v["sam3"]
    if not lab:
        return None, None
    scores = {"sam3+bank": min(share, v["bank"]["share"]) if lab == v["bank"]["label"] else None,
              "sam3+zero-shot/yolo": share if lab in (v["zero_shot"][0], v["yolo"][0]) else None}
    for r in RULES:
        if scores[r] is not None and scores[r] >= th[r]:
            return r, lab
    return None, None


def clusters(V, ids, cut):
    """Average-linkage clusters of the rows ids of V at cosine distance `cut` -> [[row]] (medoid first: the member most like
    the rest). cut 0: every row alone."""
    alone = [[i] for i in ids if not np.any(V[i])]  # no vector (no view): its own group
    ids = [i for i in ids if np.any(V[i])]
    if len(ids) < 2 or cut <= 0:
        return [[i] for i in ids] + alone
    from scipy.cluster.hierarchy import fcluster, linkage
    lab = fcluster(linkage(V[ids], method="average", metric="cosine"), t=cut, criterion="distance")
    out = []
    for c in np.unique(lab):
        mem = [ids[k] for k in np.flatnonzero(lab == c)]
        sim = V[mem] @ V[mem].T
        out.append([mem[j] for j in np.argsort(-sim.mean(1), kind="stable")])
    return out + alone


def candidates(words, v, bank_meta, measured_m=None, classes=CLASSES):
    """A medoid's options: the cluster's SAM 3 words (most voted first), the medoid's zero-shot top 3, YOLOE top 2 and bank
    top 3 (a class word, or the neighbour's own name for a class outside the taxonomy); one per class; struck when the measured
    size rules the class out (the geometry first); alphabetical (no rank cue, x13), at most MAX_OPTIONS.
    -> (options, struck)"""
    names = list(words[:6])
    if v["probs"] is not None:
        names += [classes[j] for j in np.argsort(-v["probs"])[:3]]
    names += [c for c, _ in sorted(v["yolo_all"].items(), key=lambda x: -x[1])[:2]]
    names += [bank_meta[v["bank"]["nn"][lab]]["name"] if lab.startswith("~") else lab for lab in v["bank"]["ranked"][:3]]
    seen, out = set(), []
    for n in names:
        lab = label_of(n)
        if n and lab and lab != cards.NOT_OBJECT and lab not in seen:
            seen.add(lab)
            out.append(n)
    struck = {w for w, _ in cards.strike(out, measured_m)}
    return sorted(w for w in out if w not in struck)[:MAX_OPTIONS], sorted(struck)


def settle(items, bank, th, video, family, in_video=()):
    """Tier 1, then the rest in groups: a visual cluster (DINOv2-L at the fitted cut) split by the members' SAM 3 class, one
    VLM question per group medoid (the most central member with a view). items: [{"id", "votes", "dino", "pe", "yolo",
    "measured_m", "askable"}]; in_video: this video's earlier VLM rows [(vector, meta)] (a later pass reads them like the
    bank's). -> (records {id: rec}, questions [(medoid id, the group's other members, options)], every askable rest card's own
    options {id: options} (a member the group's answer contradicts is asked on its own with them: resolve))"""
    B, meta = bank.rows(family, video)
    if in_video:
        B, meta = np.concatenate([B, np.stack([e for e, _ in in_video])]), meta + [m for _, m in in_video]
    labels = [m["label"] for m in meta]
    recs, rest, vs = {}, [], {}
    for it in items:
        probs = bank.zero_shot(it["pe"]) if np.any(it["pe"]) else None
        yo = max(it["yolo"].items(), key=lambda x: x[1]) if it["yolo"] else (None, 0.)
        v = vs[it["id"]] = {"sam3": sam3_vote(it["votes"]), "bank": knn(it["dino"], B, labels), "probs": probs, "yolo": yo, "yolo_all": it["yolo"],
                            "zero_shot": (bank.classes[int(np.argmax(probs))], float(np.max(probs))) if probs is not None else (None, 0.)}
        rule, lab = tier1(v, th)
        fz = np.bincount(bank.family_of, probs, len(FAMILIES)) if probs is not None else None
        rec = recs[it["id"]] = {"sam3": [v["sam3"][0], round(v["sam3"][1], 3), v["sam3"][2]], "bank": [v["bank"]["label"], round(v["bank"]["share"], 3)],
                                "zero_shot": [v["zero_shot"][0], round(v["zero_shot"][1], 3)], "yolo": [yo[0], round(float(yo[1]), 3)],
                                "zero_shot_family": [FAMILIES[int(fz.argmax())], round(float(fz.max()), 3)] if fz is not None else [None, 0.]}
        if rule and lab not in cards.HAZARD:  # a hazard class is shown only once a VLM named it (cards.hazard_gate): to the groups
            rec.update(route="cheap", rule=rule, label=lab, name=v["sam3"][2])
        else:
            rest.append(it)
    V = np.stack([it["dino"] for it in rest]) if rest else np.zeros((0, 1))
    qs, g = [], 0
    for c, mem in enumerate(clusters(V, list(range(len(rest))), th["cluster_cut"])):
        by = {}  # a visual cluster split by its members' SAM 3 class: one question per class present
        for i in mem:
            by.setdefault(recs[rest[i]["id"]]["sam3"][0], []).append(rest[i])
        for its in by.values():
            for it in its:
                recs[it["id"]].update(cluster=c, group=g, group_size=len(its))
            g += 1
            med = next((it for it in its if it["askable"]), None)
            if med is None:
                continue
            wv = {}
            for it in its:
                for wd, sc in (it["votes"] or {}).items():
                    wv[wd] = wv.get(wd, 0.) + float(sc)
            opts, struck = candidates([wd for wd, _ in sorted(wv.items(), key=lambda x: -x[1])], vs[med["id"]], meta, med.get("measured_m"), bank.classes)
            recs[med["id"]].update(options=opts, struck=struck)
            qs.append((med["id"], [it["id"] for it in its if it is not med], opts))
    own = {it["id"]: candidates([wd for wd, _ in sorted((it["votes"] or {}).items(), key=lambda x: -x[1])], vs[it["id"]], meta, it.get("measured_m"),
                                bank.classes)[0]
           for it in rest if it["askable"]}
    return recs, qs, own


def resolve(recs, qs, answers, items_by_id, video, site, family, own=None):
    """answers: {asked id: the chosen option (a candidate word) or None (an escape option, no answer)}. A group's answer is
    copied to members whose SAM 3 word gives its class; a member it contradicts is escalated when `own` (settle's options) is
    given: asked on its own (its options plus the look-alike's answer), else it stays 'unidentified'; a group whose medoid
    escaped stays unidentified. The escalation round (own=None) sets every remaining route.
    -> (records, the bank rows to add [(vector, meta)]: the VLM's answers only, the escalation questions)"""
    rows, esc = [], []
    for med, members, _ in qs:
        a = answers.get(med)
        lab = label_of(a) if a else None
        if lab:
            recs[med].update(route="vlm", label=lab, name=a)
            rows.append((items_by_id[med]["dino"], {"label": lab, "name": a, "site": site, "family": family, "video": video, "card": med,
                                                    "source": VLM_SOURCE, "t": round(time.time())}))
        for m in members:
            if lab and recs[m]["sam3"][0] == lab:  # the member's own word, when it names the class; else the VLM's
                o = recs[m]["sam3"][2]
                recs[m].update(route="copy", label=lab, name=o if o and label_of(o) == lab else a, medoid=med)
            elif lab and own is not None and m in own:
                opts = own[m] + ([a] if lab not in {label_of(x) for x in own[m]} else [])
                recs[m].update(escalated_from=med, options=sorted(opts))
                esc.append((m, [], sorted(opts)))
    if own is None:
        for r in recs.values():
            r.setdefault("route", "unidentified")
    return recs, rows, esc


def identity(base, rec, decided=None):
    """A card's identity from its cascade record (cards.apply_name derives the shown name, class and checks from it).
    decided: cards.decide_identity's output for a medoid (the VLM's own record: options, probabilities)."""
    naming = {k: rec.get(k) for k in ("route", "rule", "label", "sam3", "bank", "zero_shot", "zero_shot_family", "yolo", "cluster", "group", "group_size", "medoid",
                                      "escalated_from", "options", "struck")
              if rec.get(k) is not None}
    if rec["route"] == "vlm" and decided is not None:
        return {**decided, "naming": naming}
    if rec["route"] in ("cheap", "copy", "vlm"):
        how = {"cheap": f"cascade: {rec.get('rule')}", "copy": cards.COPIED, "vlm": "vlm options"}[rec["route"]]
        note = {"cheap": f"the SAM 3 word with a second agreeing vote ({rec.get('rule')}) at the family's fitted threshold; no VLM",
                "copy": "the cluster medoid's VLM answer, copied because this object's own SAM 3 word gives the same class",
                "vlm": "the VLM's choice among the cheap votes' candidates"}[rec["route"]]
        return {**base, "proposed": rec["name"], "name": rec["name"], "decided_by": how, "confidence": None, "calibrated": False, "note": note,
                "naming": naming}
    out = {**base, "proposed": cards.UNIDENTIFIED, "name": cards.UNIDENTIFIED, "decided_by": "cascade: unsettled", "confidence": None,
           "calibrated": False, "naming": naming,
           "note": "an object: no two cheap votes agreed and no verified VLM answer; its detected words stay unverified candidates"}
    if decided is not None and decided.get("decider"):  # a medoid the VLM answered with an escape (or not at all): its record stays
        out["decider"] = decided["decider"]
        out["note"] = f"an object: the VLM answered '{decided['decider'].get('answer')}'; its detected words stay unverified candidates"
    return out


def self_check():
    """The rules end to end on a toy video: tier 1 needs a second vote, one question per cluster, copies only to members
    whose SAM 3 word agrees, the rest stay unidentified, the bank never feeds a video its own rows (nor, excluded, its site's),
    is read only and its new rows (VLM answers only) go to a file; the view pick and the calibration lookup."""
    assert "machine tool holder" in CLASSES and len(CLASSES) == len(set(CLASSES)) == len(cards.FAMILY)
    import hashlib
    import tempfile
    rng = np.random.default_rng(0)
    d = 16
    base = unit(rng.normal(size=(4, d)))
    with tempfile.TemporaryDirectory() as tmp:
        text = unit(rng.normal(size=(len(CLASSES), d)))
        meta = [{"label": "box", "name": "cardboard box", "site": "a", "family": "retail", "video": "va", "card": "c0", "source": "gemini"},
                {"label": "shelf", "name": "shelf", "site": "self", "family": "retail", "video": "vself", "card": "c1", "source": "gemini"},
                {"label": "lathe", "name": "lathe", "site": "m", "family": "shop floor", "video": "vm", "card": "c2", "source": "gemini"}]
        p = Path(tmp) / "bank.npz"
        np.savez(p, emb=np.stack([base[0], base[1], base[2]]).astype(np.float16), meta=json.dumps(meta), text=text, text_scale=np.float32(50.),
                 classes=np.array(CLASSES))
        bank = Bank(p)
        assert [m["card"] for m in bank.rows("retail", "vself")[1]] == ["c0"], "same family, never this video"
        assert bank.sha256 == hashlib.sha256(p.read_bytes()).hexdigest() and bank.record()["read_only"]
        assert Bank(p, exclude_site="a").rows("retail", "vother")[1] == [meta[1]], "the scored site's rows never (benches)"
        assert Bank(p, exclude_site=["a", "self"]).rows("retail", "vother")[1] == [] and Bank(p).record()["exclude_sites"] == []
        pe_box = text[CLASSES.index("box")]
        items = [  # o0: SAM 3 box + bank box -> cheap; o1..o3 look alike (SAM 3: pallet, pallet, cart); o4 agrees with nothing, no view
            {"id": "o0", "votes": {"cardboard box": 1.}, "dino": base[0], "pe": pe_box, "yolo": {}, "askable": True},
            {"id": "o1", "votes": {"pallet": 1.}, "dino": unit(base[3] + .01), "pe": pe_box, "yolo": {}, "askable": True},
            {"id": "o2", "votes": {"wooden pallet": .9, "box": .1}, "dino": unit(base[3] + .015), "pe": pe_box, "yolo": {}, "askable": True},
            {"id": "o3", "votes": {"cart": 1.}, "dino": unit(base[3] + .02), "pe": pe_box, "yolo": {}, "askable": True},
            {"id": "o4", "votes": {"sign": 1.}, "dino": unit(-base[3]), "pe": pe_box, "yolo": {"label": .9}, "askable": False}]
        th = {"sam3+bank": .5, "sam3+zero-shot/yolo": .6, "cluster_cut": .2}
        recs, qs, own = settle(items, bank, th, "vself", "retail")
        assert recs["o0"]["route"] == "cheap" and recs["o0"]["label"] == "box" and recs["o0"]["name"] == "cardboard box", recs["o0"]
        assert [(q[0], q[1]) for q in qs] == [("o2", ["o1"]), ("o3", [])], qs  # the cluster o1-o3 split by SAM 3 class; o2 the medoid
        assert "pallet" in qs[0][2] and "wooden pallet" not in qs[0][2], qs[0][2]  # one option per class
        assert set(own) == {"o1", "o2", "o3"}, "o4 has no view: never asked"
        recs, rows, esc = resolve(recs, qs, {"o2": "pallet", "o3": None}, {it["id"]: it for it in items}, "vself", "self", "retail", own)
        assert esc == [] and "route" not in recs["o3"], "an escaped medoid is not escalated; routes are set by the last round"
        recs, rows2, _ = resolve(recs, esc, {}, {}, "vself", "self", "retail")
        routes = {k: r["route"] for k, r in recs.items()}
        assert routes == {"o0": "cheap", "o1": "copy", "o2": "vlm", "o3": "unidentified", "o4": "unidentified"}, routes
        assert recs["o1"]["name"] == "pallet" and recs["o1"]["medoid"] == "o2" and rows2 == []
        assert len(rows) == 1 and rows[0][1]["label"] == "pallet" and rows[0][1]["source"] == VLM_SOURCE  # only the VLM's answer
        r2 = {"a": {"sam3": ["box", 1., "box"]}, "b": {"sam3": ["box", 1., "carton"]}}  # the VLM contradicts the group's word
        r2, _, esc = resolve(r2, [("a", ["b"], [])], {"a": "shelf"}, {"a": {"dino": base[0]}}, "v", "s", "retail", {"b": ["box", "crate"]})
        assert r2["a"]["route"] == "vlm" and esc == [("b", [], ["box", "crate", "shelf"])], esc  # b asked on its own, shelf offered
        r2, rows3, _ = resolve(r2, esc, {"b": "crate"}, {"b": {"dino": base[1]}}, "v", "s", "retail")
        assert r2["b"]["route"] == "vlm" and r2["b"]["name"] == "crate" and rows3[0][1]["label"] == "crate"
        before = p.read_bytes()
        assert write_rows(Path(tmp) / "r" / "bank-rows-first.npz", rows) == 1 and p.read_bytes() == before, "rows go beside the report"
        z = np.load(Path(tmp) / "r" / "bank-rows-first.npz")
        assert json.loads(str(z["meta"]))[0]["label"] == "pallet" and z["emb"].shape == (1, d) and write_rows(Path(tmp) / "x.npz", []) == 0
        again = Bank(p)
        assert again.rows_at_load == 3 and again.sha256 == bank.sha256
        th_u = {"sam3+bank": float("inf"), "sam3+zero-shot/yolo": float("inf"), "cluster_cut": 0.}
        recs, qs, _ = settle(items[:1], again, th_u, "vother", "retail")
        assert "route" not in recs["o0"] and len(qs) == 1, "uncalibrated: nothing accepted cheaply, the medoid is asked"
        idn = identity({"proposed": "sign", "name": "sign", "candidates": ["sign"]}, {"route": "unidentified", "sam3": ["sign", 1., "sign"]})
        assert idn["name"] == cards.UNIDENTIFIED and idn["decided_by"] == "cascade: unsettled"
        c = {"folds": {"s1": {"sam3+bank": .2, "sam3+zero-shot/yolo": None, "cluster_cut": .3}}, "families": {},
             "uncalibrated": {"sam3+bank": None, "sam3+zero-shot/yolo": None, "cluster_cut": .1, "uncalibrated": True}}
        (Path(tmp) / "c.json").write_text(json.dumps(c))
        assert calibration("s1", "retail", Path(tmp) / "c.json")["sam3+zero-shot/yolo"] == float("inf")
        assert calibration("new", "x", Path(tmp) / "c.json")["uncalibrated"] is True
    V = unit(np.array([[1., 0], [.99, .1], [0, 0], [0, 1.]]))
    assert clusters(V, [0, 1, 2, 3], .1) == [[0, 1], [3], [2]] or clusters(V, [0, 1, 2, 3], .1) == [[1, 0], [3], [2]]  # no vector: alone
    assert pick_views([10, 50, 40, 30], [False, True, False, False], [0, 1, 1, 2], k=2) == [2, 3]  # edge halves 50; one per frame
    v = {"sam3": ("box", .7, "box"), "bank": {"label": "shelf", "share": .9}, "zero_shot": ("shelf", .9), "yolo": ("box", .2)}
    assert tier1(v, {"sam3+bank": .1, "sam3+zero-shot/yolo": .5}) == ("sam3+zero-shot/yolo", "box")
    assert tier1({**v, "yolo": (None, 0.)}, {"sam3+bank": .1, "sam3+zero-shot/yolo": .5}) == (None, None), "one vote is not enough"
    print("cascade self-check ok: second vote, one question per cluster, verified copies, unidentified, frozen bank (read only, "
          "its hash, rows beside the report, never its own video or excluded site), calibration lookup, view pick")
