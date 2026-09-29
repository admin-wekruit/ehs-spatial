"""SAM 3 for the fast report: one resident model per GPU, one priority queue both GPUs take from (E9's dynamic split),
the vocabulary in two waves (wave 1 = EHS core + this site's cached words, known at t = 0, run inside the person/floor
task while its vision features are in hand; wave 2 = only the words the VLM adds, on the cached features), the flood of
masks that generic words bring (E2b caveat), the E7 lift, naming, and the video outlines (keyframes 'segmented',
in-between 5 fps keyframes 'projected' with E6b's 'pair' rule, people cut out).
"""
import itertools
import queue
import threading

import numpy as np

DA3_HW, SAM_SIDE, LR = (280, 504), 1008, 288
PERSON_FRAMES, PAIRS, OBJECT_EVERY = 8, 80, 3
PERSON_SCORE, PERSON_TOP, VOCAB_SCORE = .4, 12, .3  # person/floor: the production rule E3/E9 validated; vocabulary: E2b, no cap
FLOOD_IOU, INSIDE = .8, .5
LIFT_VOXEL, MIN_PIXELS, MOVING, MATCH_MIN, MATCH_MAX, CONFIRMED = .05, 16, .5, .5, .2, 2  # E9's lift rule
GENERIC = {"tool", "tools", "machine", "machinery", "equipment", "object", "item", "part", "metal part", "container", "device",
           "thing", "material", "structure", "unit", "component", "hardware", "supplies", "fixture"}
EMPTY = 2 ** 63 - 1
HOLE_ITERS = 3


class Sam3:
    """SAM 3 image model (bf16) resident on one GPU. Batch methods: text(), vision(), detect()."""

    def __init__(self, model, processor, dev):
        import torch
        self.model, self.proc, self.dev, self.texts = model, processor, dev, {}
        ip = processor.image_processor
        assert (ip.size["height"], ip.size["width"]) == (SAM_SIDE, SAM_SIDE), ip.size
        self.mean = torch.tensor(ip.image_mean, device=dev).view(1, 3, 1, 1)
        self.std = torch.tensor(ip.image_std, device=dev).view(1, 3, 1, 1)
        self.lock = threading.Lock()

    def text(self, words):
        """Encoded once per word list (0.015-0.018 s, E2b) and kept."""
        import torch
        key = tuple(words)
        with self.lock:
            if key not in self.texts:
                tok = self.proc(text=list(words), return_tensors="pt").to(self.dev)
                with torch.cuda.device(self.dev), torch.inference_mode():
                    enc = self.model.get_text_features(input_ids=tok["input_ids"], attention_mask=tok["attention_mask"])
                wrapped = hasattr(enc, "pooler_output")
                self.texts[key] = (enc.pooler_output if wrapped else enc, tok["attention_mask"], wrapped)
            return self.texts[key]

    def vision(self, frames_bgr):
        """(n,H,W,3) uint8 BGR on this GPU -> vision features (the processor's resize + normalise, on the GPU, E2)."""
        import torch
        import torch.nn.functional as F
        x = frames_bgr.permute(0, 3, 1, 2).flip(1).float() / 255
        x = F.interpolate(x, size=(SAM_SIDE, SAM_SIDE), mode="bilinear", antialias=True, align_corners=False)
        return self.model.get_vision_features(pixel_values=((x - self.mean) / self.std).to(torch.bfloat16))

    @staticmethod
    def pick(vision, idx, clone=False):
        return type(vision)(**{k: tuple((t[idx].clone() if clone else t[idx]) for t in v) for k, v in vision.items() if k.startswith("fpn_")})

    def detect(self, vision, n_frames, words, floor, top=None, logits=False):
        """Every frame x every word, at most PAIRS (frame, word) pairs per forward. Keeps queries with score >= floor
        (top `top` per pair when set); only kept masks are upsampled, straight to the DA3 grid.
        -> {frame, word (index into words), score, mask (DA3 grid), [logits (LR x LR fp16)]}"""
        import types
        import torch
        import torch.nn.functional as F
        feats, amask, wrapped = self.text(words)
        per_w = min(len(words), PAIRS)
        per_f = max(1, PAIRS // per_w)
        out = []
        for f0 in range(0, n_frames, per_f):
            fs = list(range(f0, min(f0 + per_f, n_frames)))
            v = self.pick(vision, fs) if (f0 or len(fs) < n_frames) else vision
            for w0 in range(0, len(words), per_w):
                ws = list(range(w0, min(w0 + per_w, len(words))))
                k = len(ws)
                f = feats[ws].repeat(len(fs), 1, 1)
                o = self.model(vision_embeds=type(v)(**{n: tuple(t.repeat_interleave(k, 0) for t in x) for n, x in v.items() if n.startswith("fpn_")}),
                               attention_mask=amask[ws].repeat(len(fs), 1), text_embeds=types.SimpleNamespace(pooler_output=f) if wrapped else f)
                scores = o.pred_logits.float().sigmoid()
                if getattr(o, "presence_logits", None) is not None:
                    scores = scores * o.presence_logits.float().sigmoid()
                if top:
                    s, q = scores.topk(top, dim=1)
                    pair, rank = torch.nonzero(s >= floor, as_tuple=True)
                    q, s = q[pair, rank], s[pair, rank]
                else:
                    pair, q = torch.nonzero(scores >= floor, as_tuple=True)
                    s = scores[pair, q]
                lr = o.pred_masks[pair, q]
                m = (F.interpolate(lr[None].float(), size=DA3_HW, mode="bilinear", align_corners=False)[0] > 0) if len(pair) else \
                    torch.zeros((0, *DA3_HW), dtype=torch.bool, device=self.dev)
                r = {"frame": torch.tensor(fs, device=self.dev)[pair // k], "word": torch.tensor(ws, device=self.dev)[pair % k], "score": s, "mask": m}
                if logits:
                    r["logits"] = lr.half()
                out.append(r)
        return {key: torch.cat([r[key] for r in out]) for key in out[0]}


class SamWork:
    """The SAM 3 task queue of one run, any GPU takes any task, in this order:
      0 ('person', chunk start): {person, floor} on PERSON_FRAMES keyframes; keeps the vision features of the object
        keyframes among them and queues their wave 1 (run 002: wave 1 inside this task held the people, and so the
        geometry layers, back by 8-10 s once the site cache made wave 1 58 words long);
      1 ('wave1', chunk start): wave 1 on those object keyframes, on the kept features;
      2 ('wave2', keyframe): the VLM's new words on one object keyframe, on the kept features.
    No GPU is held for vLLM: holding GPU 1 while the vocabulary decoded (runs 003-005) made it 1.5x faster but left 7 s
    of GPU 1 idle, and total SAM 3 work, not the vocabulary, is what the objects wait for once it starts at t = 0.5 s."""

    def __init__(self, sams, dev_out, clock, wave1):
        self.sams, self.dev_out, self.clock, self.wave1, self.wave2 = sams, dev_out, clock, list(wave1), None
        self.tasks, self.order, self.lock = queue.PriorityQueue(), itertools.count(), threading.Lock()
        self.chunks = {d: [] for d in sams}
        self.cache, self.person, self.vocab = {}, [], []
        self.total, self.done = {"person": None, "wave1": None, "wave2": None}, {"person": 0, "wave1": 0, "wave2": 0}
        self.decoded, self.vocab_known, self.person_ready, self.all_ready = (threading.Event() for _ in range(4))
        self.by_worker, self.error = {}, None

    @property
    def words(self):
        return self.wave1 + (self.wave2 or [])

    def add_chunk(self, frames):
        """frames: list of (720,1280,3) uint8 keyframes, the next chunk; uploaded once to every GPU."""
        import torch
        chunk = torch.from_numpy(np.stack(frames))
        start = len(self.chunks[self.dev_out]) * PERSON_FRAMES
        for d in self.sams:
            self.chunks[d].append(chunk.to(d, non_blocking=False))
        self.tasks.put((0, next(self.order), "person", start))

    def seal_decode(self):
        with self.lock:
            self.total["person"] = n = len(self.chunks[self.dev_out])
            n_keys = sum(len(c) for c in self.chunks[self.dev_out])
            self.total["wave1"] = sum(any((x + j) % OBJECT_EVERY == 0 for j in range(min(PERSON_FRAMES, n_keys - x)))
                                      for x in range(0, n * PERSON_FRAMES, PERSON_FRAMES))
            self.decoded.set()
            self._check()

    def set_wave2(self, words, n_keys):
        """The VLM's words not already in wave 1 -> one task per object keyframe (none when nothing is new)."""
        new = [w for w in dict.fromkeys(words) if w not in self.wave1]
        with self.lock:
            self.wave2 = new
            xs = [x for x in range(0, n_keys, OBJECT_EVERY)] if new else []
            self.total["wave2"] = len(xs)
            for x in xs:
                self.tasks.put((2, next(self.order), "wave2", x))
            self.vocab_known.set()
            self._check()
        return new

    def _check(self):  # under lock
        if self.decoded.is_set() and self.done["person"] == self.total["person"]:
            self.person_ready.set()
            if self.vocab_known.is_set() and self.done["wave1"] == self.total["wave1"] and self.done["wave2"] == self.total["wave2"]:
                self.all_ready.set()

    def fail(self, error):
        self.error = error
        for e in (self.decoded, self.vocab_known, self.person_ready, self.all_ready):
            e.set()

    def wait(self, event):
        event.wait()
        if self.error is not None:
            raise RuntimeError("SAM 3 worker failed") from self.error

    def worker(self, dev, role, until=None):
        try:
            self._worker(dev, role, until)
        except BaseException as error:  # noqa: BLE001  the main thread re-raises it from wait()
            self.fail(error)
            raise

    def _features(self, x, dev):
        """Kept vision features of object keyframe x, on dev (copied across GPUs when the other one kept them)."""
        with self.lock:
            v = self.cache.get(x)
        return None if v is None else type(v)(**{k: tuple(t.to(dev) for t in val) for k, val in v.items()})

    def _worker(self, dev, role, until):
        import torch
        sam, got, gpu = self.sams[dev], {"person": 0, "wave1": 0, "wave2": 0}, dev.index
        with torch.cuda.device(dev), torch.inference_mode():
            while not (until is not None and until.is_set()) and self.error is None and not self.all_ready.is_set():
                try:
                    _, _, kind, x = self.tasks.get(timeout=.01)
                except queue.Empty:
                    continue
                if kind == "person":
                    chunk = self.chunks[dev][x // PERSON_FRAMES]
                    real = len(chunk)
                    if real < PERSON_FRAMES:  # fixed shapes for the allocator
                        chunk = torch.cat([chunk, chunk[-1:].expand(PERSON_FRAMES - real, -1, -1, -1)])
                    with self.clock.stage(f"sam3.person@gpu{gpu}", gpu=dev, n={"frames": real, "pairs": 2 * PERSON_FRAMES}):
                        vision = sam.vision(chunk)
                        r = sam.detect(vision, PERSON_FRAMES, ("person", "floor"), PERSON_SCORE, top=PERSON_TOP)
                        r = {k: v[r["frame"] < real] for k, v in r.items()}
                        r["frame"] = r["frame"] + x
                        self.person.append({k: v.to(self.dev_out) for k, v in r.items()})
                        objs = [j for j in range(real) if (x + j) % OBJECT_EVERY == 0]
                        with self.lock:
                            for j in objs:
                                self.cache[x + j] = sam.pick(vision, [j], clone=True)
                    if objs:
                        self.tasks.put((1, next(self.order), "wave1", x))
                elif kind == "wave1":
                    objs = [x + j for j in range(PERSON_FRAMES) if (x + j) % OBJECT_EVERY == 0 and x + j in self.cache]
                    with self.clock.stage(f"sam3.vocab.wave1@gpu{gpu}", gpu=dev, n={"frames": len(objs), "words": len(self.wave1)}):
                        fs = [self._features(q, dev) for q in objs]
                        v = type(fs[0])(**{k: tuple(torch.cat([f[k][i] for f in fs]) for i in range(len(fs[0][k]))) for k in fs[0]})
                        r = sam.detect(v, len(objs), self.wave1, VOCAB_SCORE, logits=True)
                        r["frame"] = torch.tensor(objs, device=dev)[r["frame"]]
                        self.vocab.append({k: t.to(self.dev_out) for k, t in r.items()})
                else:
                    v = self._features(x, dev)
                    with self.clock.stage(f"sam3.vocab.wave2@gpu{gpu}", gpu=dev, n={"frames": 1, "words": len(self.wave2), "cached_features": v is not None}):
                        if v is None:
                            chunk = self.chunks[dev][x // PERSON_FRAMES]
                            v = sam.vision(chunk[x % PERSON_FRAMES:x % PERSON_FRAMES + 1])
                        r = sam.detect(v, 1, self.wave2, VOCAB_SCORE, logits=True)
                        r["frame"] = r["frame"] + x
                        r["word"] = r["word"] + len(self.wave1)
                        self.vocab.append({k: t.to(self.dev_out) for k, t in r.items()})
                torch.cuda.current_stream(dev).synchronize()  # results visible before anyone is told
                got[kind] += 1
                with self.lock:
                    self.done[kind] += 1
                    self._check()
        with self.lock:
            w = self.by_worker.setdefault(f"{role} gpu{gpu}", {"person_chunks": 0, "wave1_chunks": 0, "wave2_frames": 0})
            for k, n in (("person_chunks", got["person"]), ("wave1_chunks", got["wave1"]), ("wave2_frames", got["wave2"])):
                w[k] += n

    def gathered(self, kind):
        import torch
        rows = self.person if kind == "person" else self.vocab
        return {k: torch.cat([r[k] for r in rows]) for k in rows[0]} if rows else None


# ---------- the flood of masks ----------

def dedupe(frame, word, score, masks, stride=2):
    """Per frame, best score first (E2b caveat: generic words flood 150-390 masks per frame):
      - a mask with IoU > FLOOD_IOU with a kept mask of ANOTHER word is the same thing: dropped, its word votes for it;
      - a mask at least INSIDE within a kept mask of the SAME word is a part or a double (fast5_dedupe): dropped.
    -> (kept indices, votes: list of (kept index, word, score) including each kept mask's own)."""
    import torch
    frame_np, word_np, score_np = frame.cpu().numpy(), word.cpu().numpy(), score.float().cpu().numpy()
    kept_all, votes = [], []
    for f in np.unique(frame_np):
        idx = np.flatnonzero(frame_np == f)
        idx = idx[np.argsort(-score_np[idx], kind="stable")]
        m = masks[torch.from_numpy(idx).to(masks.device)][:, ::stride, ::stride].flatten(1).float()
        area = m.sum(1)
        inter = (m @ m.T).cpu().numpy()
        area = area.cpu().numpy()
        kept = []
        for a in range(len(idx)):
            if kept:
                k = np.array(kept)
                other = word_np[idx[k]] != word_np[idx[a]]
                iou = inter[a, k] / np.maximum(area[a] + area[k] - inter[a, k], 1)
                same = np.flatnonzero(other & (iou > FLOOD_IOU))
                if len(same):
                    votes.append((idx[k[same[0]]], word_np[idx[a]], score_np[idx[a]]))
                    continue
                if (~other & (inter[a, k] >= INSIDE * max(area[a], 1))).any():
                    continue
            kept.append(a)
            votes.append((idx[a], word_np[idx[a]], score_np[idx[a]]))
        kept_all += idx[kept].tolist()
    return np.array(sorted(kept_all), int), votes


def is_generic(word, candidates):
    return word in GENERIC or any(c != word and c.endswith(" " + word) for c in candidates)


def name(votes, order):
    """Among well-supported words (>= half the best vote), a specific word before a generic one, then the VLM's order."""
    if not votes:
        return None
    top = max(votes.values())
    cands = [w for w, v in votes.items() if v >= .5 * top]
    return min(cands, key=lambda w: (is_generic(w, cands), order.index(w) if w in order else len(order)))


# ---------- the E7 lift ----------

def backproject(depth, K, c2w, f, vy, vx, stride):
    """Pixels (frame f, row vy, col vx of a stride grid over depth's full resolution) -> world points."""
    import torch
    u, v = vx.float() * stride + (stride - 1) / 2, vy.float() * stride + (stride - 1) / 2
    z = depth[f, vy * stride, vx * stride]
    k = K[f]
    cam = torch.stack([(u - k[:, 0, 2]) / k[:, 0, 0] * z, (v - k[:, 1, 2]) / k[:, 1, 1] * z, z], 1)
    return (c2w[f, :3, :3] @ cam[:, :, None])[:, :, 0] + c2w[f, :3, 3]


def lift(masks, frame_of, depth_m, K, c2w_m, dyn, stride=2, emb=None, emb_min=None):
    """E7's lift (E9's code): pixels -> 5 cm voxels -> cross-frame voxel overlap -> connected components.
    emb (len(masks), D) unit, optional (X2 proposals): an edge also needs the two masks' cosine >= emb_min.
    -> (component per input mask, -1 = not lifted; per component arrays; stats)."""
    import torch
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    dev, n = depth_m.device, len(depth_m)
    comp = np.full(len(masks), -1)
    m = masks[:, ::stride, ::stride]
    dy = dyn[:, ::stride, ::stride][frame_of]
    moving = (m & dy).sum((1, 2)) >= MOVING * m.sum((1, 2)).clamp(min=1)
    m = m & (depth_m[:, ::stride, ::stride] > 0)[frame_of] & ~dy
    idx = torch.nonzero(~moving & (m.sum((1, 2)) >= MIN_PIXELS)).squeeze(1)
    stats = {"masks_in": len(masks), "masks_dropped_moving": int(moving.sum()), "masks_lifted": len(idx)}
    if len(idx) < 2:
        return comp, None, stats
    mid, vy, vx = torch.nonzero(m[idx], as_tuple=True)
    fr = frame_of[idx]
    world = backproject(depth_m, K, c2w_m, fr[mid], vy, vx, stride)
    ijk = torch.floor(world / LIFT_VOXEL).long()
    B, OFF = 1 << 21, 1 << 20
    vox, vid = torch.unique(((ijk[:, 0] + OFF) * B + (ijk[:, 1] + OFF)) * B + (ijk[:, 2] + OFF), return_inverse=True)
    pair = torch.unique(mid * len(vox) + vid)
    pm, pv = pair // len(vox), pair % len(vox)
    size = torch.bincount(pm, minlength=len(idx)).float()
    A = torch.sparse_coo_tensor(torch.stack([pm, pv]), torch.ones_like(pm, dtype=torch.float32), (len(idx), len(vox))).coalesce()
    inter = torch.sparse.mm(A, A.t()).coalesce()
    (a, b), c = inter.indices(), inter.values()
    edge = (a < b) & (fr[a] != fr[b]) & (c >= MATCH_MIN * torch.minimum(size[a], size[b])) & (c >= MATCH_MAX * torch.maximum(size[a], size[b]))
    if emb is not None:
        e = emb[idx]
        edge &= (e[a] * e[b]).sum(1) >= emb_min
    a, b = a[edge].cpu().numpy(), b[edge].cpu().numpy()
    ncomp, label = connected_components(coo_matrix((np.ones(len(a)), (a, b)), shape=(len(idx), len(idx))), directed=False)
    lab = torch.from_numpy(label).to(dev)
    ov = torch.unique(lab[pm] * len(vox) + pv)
    oc, ovid = ov // len(vox), ov % len(vox)
    centre = torch.stack([vox // (B * B) - OFF, (vox // B) % B - OFF, vox % B - OFF], 1).float()[ovid] * LIFT_VOXEL + LIFT_VOXEL / 2
    nvox = torch.bincount(oc, minlength=ncomp).float()
    cent = torch.zeros(ncomp, 3, device=dev).index_add_(0, oc, centre) / nvox.clamp(min=1)[:, None]
    lo = torch.full((ncomp, 3), 1e9, device=dev).scatter_reduce(0, oc[:, None].expand(-1, 3), centre, "amin")
    hi = torch.full((ncomp, 3), -1e9, device=dev).scatter_reduce(0, oc[:, None].expand(-1, 3), centre, "amax")
    nframes = torch.bincount(torch.unique(lab * n + fr) // n, minlength=ncomp)
    comp[idx.cpu().numpy()] = label
    arrays = {k: x.cpu().numpy() for k, x in (("frames", nframes), ("voxels", nvox), ("centroid", cent), ("lo", lo), ("hi", hi))}
    stats.update(points=int(len(mid)), voxels=int(len(vox)), edges=int(len(a)), components=int(ncomp))
    return comp, arrays, stats


# ---------- outlines ----------

def polygons(mask, sx, sy, min_area=4.):
    """bool (h,w) numpy -> outer polygons in source pixels (x * sx, y * sy at pixel centres)."""
    import cv2
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        c = cv2.approxPolyDP(c, .75, True)[:, 0].astype(np.float64)
        out.append(np.round(np.stack([(c[:, 0] + .5) * sx - .5, (c[:, 1] + .5) * sy - .5], 1), 1).tolist())
    return out


def label_polygons(lab, sx, sy):
    """int label map (0 = nothing) -> {label: polygons}. A top-level function: the outlines run it in worker processes."""
    return {int(v): poly for v in np.unique(lab) if v > 0 for poly in [polygons(lab == v, sx, sy)] if poly}


def paint(masks):
    """(n,H,W) bool -> (H,W) label map, 0 = nothing, i + 1 = masks[i]; where masks overlap, the smaller one wins."""
    import torch
    order = torch.argsort(masks.flatten(1).sum(1), descending=True)
    rank = torch.empty_like(order)
    rank[order] = torch.arange(1, len(order) + 1, device=masks.device)
    top = (masks * rank[:, None, None]).amax(0)
    inv = torch.zeros(len(order) + 1, dtype=torch.long, device=masks.device)
    inv[1:] = order + 1
    return inv[top]


def splat(world, ids, K, w2c, hw):
    """E6b's z-buffer splat of world points into one view: per pixel int64 key (z in 0.1 mm << 20 | id), EMPTY where none;
    the rounded pixel, then the 2x2 floor block where that left gaps."""
    import torch
    H, W = hw
    cam = world @ w2c[:3, :3].T + w2c[:3, 3]
    front = cam[:, 2] > 1e-3
    cam, ids = cam[front], ids[front]
    x, y = cam[:, 0] / cam[:, 2] * K[0, 0] + K[0, 2], cam[:, 1] / cam[:, 2] * K[1, 1] + K[1, 2]
    key = (cam[:, 2] * 1e4).long().clamp(0, 2 ** 40) * 2 ** 20 + ids

    def scatter(out, xi, yi):
        inside = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
        return out.scatter_reduce_(0, (yi * W + xi)[inside], key[inside], reduce="amin")
    near = scatter(torch.full((H * W,), EMPTY, dtype=torch.int64, device=world.device), torch.round(x).long(), torch.round(y).long())
    wide = torch.full_like(near, EMPTY)
    fx, fy = torch.floor(x).long(), torch.floor(y).long()
    for dx in (0, 1):
        for dy in (0, 1):
            scatter(wide, fx + dx, fy + dy)
    return torch.where(near != EMPTY, near, wide)


def finish(key, hw):
    """Packed keys -> id map (-1 = nothing); holes up to 3 px take a neighbour's id (E6b)."""
    import torch
    lab = torch.where(key == EMPTY, -1, key % 2 ** 20).float().view(1, 1, *hw)
    for _ in range(HOLE_ITERS):
        lab = torch.where(lab < 0, torch.nn.functional.max_pool2d(lab, 3, 1, 1), lab)
    return lab[0, 0].long()


def project_pair(ids, depth_m, K, c2w_m, keys_local, target):
    """E6b 'pair': the two object keyframes around `target` (local index), the nearer first; the other fills what the
    nearer cannot see. ids: (n,H,W) id maps (valid on keys_local). -> (H,W) long, -1 = nothing, 0 = background."""
    import torch
    before = [k for k in keys_local if k < target]
    after = [k for k in keys_local if k > target]
    order = sorted([x for x in (before[-1:] + after[:1])], key=lambda k: (abs(k - target), k))
    w2c = torch.linalg.inv(c2w_m[target])
    key = None
    for k in order:
        z = depth_m[k]
        v, u = torch.nonzero(z > 0, as_tuple=True)
        world = backproject(depth_m, K, c2w_m, torch.full_like(v, k), v, u, 1)
        s = splat(world, ids[k][v, u], K[target], w2c, DA3_HW)
        key = s if key is None else torch.where(key != EMPTY, key, s)
    return finish(key, DA3_HW) if key is not None else torch.full(DA3_HW, -1, dtype=torch.long, device=depth_m.device)


def self_check():
    """Naming and polygons anywhere; the flood rules where torch is installed (the Modal setup function runs them too)."""
    order = ["fire extinguisher", "tool cabinet", "cabinet", "machine", "lathe"]
    assert name({"cabinet": 2., "tool cabinet": 1.5, "lathe": .1}, order) == "tool cabinet"  # specific over its head noun
    assert name({"machine": 3., "lathe": 2.}, order) == "lathe"  # 'machine' is generic
    assert name({"machine": 3., "lathe": 1.}, order) == "machine"  # lathe lacks support
    assert polygons(np.pad(np.ones((4, 4), bool), 2), 2., 2.)[0][0] == [4.5, 4.5]
    try:
        import torch
    except ImportError:
        print("segment self-check ok: naming, polygons (flood rules skipped: no torch here)")
        return
    masks = torch.zeros((5, 280, 504), dtype=torch.bool)
    masks[0, 10:50, 10:50] = True   # 'cabinet' .9
    masks[1, 11:50, 10:50] = True   # 'machine' .8, IoU > .8 with 0: merged, votes for 0
    masks[2, 12:30, 12:30] = True   # 'cabinet' .7, inside 0: dropped
    masks[3, 100:140, 100:140] = True  # 'cabinet' .6, elsewhere: kept
    masks[4, 10:50, 10:50] = True   # frame 1: kept
    kept, votes = dedupe(torch.tensor([0, 0, 0, 0, 1]), torch.tensor([0, 1, 0, 0, 1]), torch.tensor([.9, .8, .7, .6, .5]), masks)
    assert kept.tolist() == [0, 3, 4], kept
    assert sorted((int(a), int(w)) for a, w, _ in votes) == [(0, 0), (0, 1), (3, 0), (4, 1)], votes
    lab = paint(masks[[0, 2, 3]])  # 2 lies inside 0: the smaller one keeps its pixels
    assert int(lab[20, 20]) == 2 and int(lab[45, 45]) == 1 and int(lab[120, 120]) == 3 and int(lab[200, 400]) == 0
    two = torch.zeros((2, 280, 504), dtype=torch.bool)  # one mask seen twice from one camera: joined, unless SigLIP disagrees (X2)
    two[:, 100:160, 200:260] = True
    geo = (torch.full((2, 280, 504), 2.), torch.tensor([[300., 0, 252], [0, 300, 140], [0, 0, 1]]).repeat(2, 1, 1), torch.eye(4).repeat(2, 1, 1),
           torch.zeros((2, 280, 504), dtype=torch.bool))
    assert lift(two, torch.tensor([0, 1]), *geo)[0].tolist() == [0, 0]
    assert lift(two, torch.tensor([0, 1]), *geo, emb=torch.eye(2), emb_min=.5)[0].tolist() == [0, 1]
    print("segment self-check ok: flood merge/drop, naming, polygons, lift with the SigLIP gate")
