"""SAM 3 for the fast report: one resident model per GPU, one priority queue both GPUs take from (E9's dynamic split),
the vocabulary in two waves (wave 1 = EHS core + this site's cached words, known at t = 0, run inside the person/floor
task while its vision features are in hand; wave 2 = only the words the VLM adds, on the cached features), the flood of
masks that generic words bring (E2b caveat), the E7 lift, naming, and the video outlines (keyframes 'segmented',
in-between 5 fps keyframes 'projected' with E6b's 'pair' rule, people cut out).
"""
import itertools
import queue
import threading
import time

import numpy as np

DA3_HW, SAM_SIDE, LR = (280, 504), 1008, 288
PERSON_FRAMES, PAIRS, OBJECT_EVERY = 8, 80, 3
PERSON_SCORE, PERSON_TOP, VOCAB_SCORE = .4, 12, .3  # person/floor: the production rule E3/E9 validated; vocabulary: E2b, no cap
FLOOD_IOU, INSIDE = .8, .5
LIFT_VOXEL, MIN_PIXELS, MOVING, MATCH_MIN, MATCH_MAX, CONFIRMED = .05, 16, .5, .5, .2, 2  # E9's lift rule
GENERIC = {"tool", "tools", "machine", "machinery", "equipment", "object", "item", "part", "metal part", "container", "device",
           "thing", "material", "structure", "unit", "component", "hardware", "supplies", "fixture"}
EMPTY = 2 ** 63 - 1
EDGE_JUMP, ERODE_KEEP, DEPTH_REL = .05, .3, .05  # click MVP section 4.2: m3_exp_geometry.EDGE_JUMP; photo erosion rule; DA3 ~5 %
# mvp2 (R3): the raw mask's upper / lower edge counts when it lies at most this many DA3 px beyond the kept points' own top /
# bottom in that column: the depth-edge pixel (1) + the 1 px erosion (1) + the stride-2 sample (1) + 1 of slack. Further
# out the mask is taken to have bled and the kept point stands. Set from the cleaning rule before any run, not tuned.
EDGE_GAP_PX = 4
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


def clean(masks, frame_of, depth_m, stride=2, batch=2048):
    """Click MVP section 4.2 steps 1: mask pixels at depth edges out (a 4-neighbour jumps by > EDGE_JUMP x the depth:
    m3_exp_geometry's flying-pixel rule, X1's per-view extents), then a 1 DA3 px (3x3) erosion unless < ERODE_KEEP of
    the mask would remain (the photo rule: scene_inventory erodes 2 px at full resolution). Evaluated only at the stride
    grid's pixels. -> (n, H/stride, W/stride) bool."""
    import torch
    import torch.nn.functional as F
    pad = F.pad(depth_m[:, None], (1, 1, 1, 1), mode="replicate")[:, 0]
    nb = torch.stack([pad[:, 1:-1, :-2], pad[:, 1:-1, 2:], pad[:, :-2, 1:-1], pad[:, 2:, 1:-1]])
    edge = (nb - depth_m).abs().amax(0) > EDGE_JUMP * depth_m
    H, W = depth_m.shape[1:]
    out = []
    for b0 in range(0, len(masks), batch):
        m = masks[b0:b0 + batch] & ~edge[frame_of[b0:b0 + batch]]
        p = F.pad(m.to(torch.uint8), (1, 1, 1, 1), value=1).bool()  # outside the image counts as inside: a frame cut is not a rim
        er = torch.ones_like(m[:, ::stride, ::stride])
        for dy in (0, 1, 2):
            for dx in (0, 1, 2):
                er &= p[:, dy:dy + H:stride, dx:dx + W:stride]
        m2 = m[:, ::stride, ::stride]
        ok = er.sum((1, 2)) >= ERODE_KEEP * m2.sum((1, 2))
        out.append(torch.where(ok[:, None, None], er, m2))
    return torch.cat(out) if out else masks[:, ::stride, ::stride]


def trim(z, mid, n_masks):
    """Section 4.2 step 2, per mask: keep points whose camera depth lies in [p15 - m, p85 + m], m = 0.25 (p85 - p15) +
    DEPTH_REL z_med (the photo rule with its fixed 5 cm as 5 % of the depth). -> bool per point."""
    import torch
    order = torch.argsort(z)
    order = order[torch.argsort(mid[order], stable=True)]
    zs, ms = z[order], mid[order]
    cnt = torch.bincount(ms, minlength=n_masks)
    start = torch.cumsum(cnt, 0) - cnt
    last = (cnt - 1).clamp(min=0).float()
    q = lambda p: zs[(start + (last * p).long()).clamp(max=max(len(zs) - 1, 0))]  # noqa: E731
    p15, p50, p85 = q(.15), q(.5), q(.85)
    m = .25 * (p85 - p15) + DEPTH_REL * p50
    ok = (zs >= (p15 - m)[ms]) & (zs <= (p85 + m)[ms])
    keep = torch.empty_like(ok)
    keep[order] = ok
    return keep


def mask_points(masks, frame_of, depth_m, K, c2w_m, dyn, stride=2):
    """The lift's point step (click MVP section 4.2 steps 1-2): masks mostly on people out (E9), depth-edge pixels and a
    1 px rim out, >= MIN_PIXELS left, per-mask depth tails trimmed, back-projected (stride grid).
    -> ({lifted (L) input mask index, mid (N) index into lifted, world (N,3) metres, z (N) camera depth, fr (L) frame,
         border (L,4) top/bottom/left/right edge touched, pixels (L) raw mask pixels} or None, stats)."""
    import torch
    raw = masks[:, ::stride, ::stride]
    dy = dyn[:, ::stride, ::stride][frame_of]
    moving = (raw & dy).sum((1, 2)) >= MOVING * raw.sum((1, 2)).clamp(min=1)
    m = clean(masks, frame_of, depth_m, stride) & (depth_m[:, ::stride, ::stride] > 0)[frame_of] & ~dy
    idx = torch.nonzero(~moving & (m.sum((1, 2)) >= MIN_PIXELS)).squeeze(1)
    stats = {"masks_in": len(masks), "masks_dropped_moving": int(moving.sum()), "masks_lifted": len(idx), "pixels_raw": int(raw.sum()),
             "pixels_clean": int(m.sum())}
    if not len(idx):
        return None, stats
    mid, vy, vx = torch.nonzero(m[idx], as_tuple=True)
    fr = frame_of[idx]
    z = depth_m[fr[mid], vy * stride, vx * stride]
    t = trim(z, mid, len(idx))
    mid, vy, vx, z = mid[t], vy[t], vx[t], z[t]
    stats["points_trimmed"] = int((~t).sum())
    mk = masks[idx]
    border = torch.stack([mk[:, :2].flatten(1).any(1), mk[:, -2:].flatten(1).any(1), mk[:, :, :2].flatten(1).any(1), mk[:, :, -2:].flatten(1).any(1)], 1)
    return {"lifted": idx, "mid": mid, "world": backproject(depth_m, K, c2w_m, fr[mid], vy, vx, stride), "z": z, "fr": fr,
            "border": border, "pixels": raw[idx].sum((1, 2)), "edge": edges(mk, mid, vy, vx, z, fr, K, c2w_m, stride)}, stats


def edges(mk, mid, vy, vx, z, fr, K, c2w, stride=2, batch=2048):
    """mvp2 (R3: tops read low; the cleaning shaves the rim): per lifted mask and stride column, the raw (un-eroded) mask's
    top and bottom pixel edge, placed at the camera depth of the object's own topmost / bottommost kept point in that
    column (depth consistency: the rim's own depth is the blend the cleaning dropped), used only within EDGE_GAP_PX of that
    point (else the kept point itself). A column whose raw mask reaches the image border gives no edge (cut, not seen).
    mk: the lifted masks at full DA3 size; mid/vy/vx/z: the kept points (mask_points, after the trim). -> {top, bottom:
    (lifted-mask index, world points)}."""
    import torch
    L, H, W = mk.shape
    Wc = W // stride
    key = mid * Wc + vx
    out = {}
    for name in ("top", "bottom"):
        top = name == "top"
        at = torch.full((L * Wc,), H if top else -1, dtype=vy.dtype, device=vy.device).scatter_reduce(0, key, vy, "amin" if top else "amax")
        hit = vy == at[key]  # the one kept point per (mask, column) on that row
        k, zk = key[hit], z[hit]
        rows = []  # the raw mask's first / last row per (mask, stride column), in batches (full-height masks)
        for b0 in range(0, L, batch):
            c = mk[b0:b0 + batch, :, ::stride][:, :, :Wc].to(torch.uint8)
            rows.append(c.argmax(1) if top else H - 1 - c.flip(1).argmax(1))
        raw = torch.cat(rows).flatten()[k] if rows else torch.zeros_like(k)
        kept = at[k] * stride  # the kept point's full-resolution row (its column holds a raw pixel there, so raw is on its side)
        use = ((kept - raw) if top else (raw - kept)) <= EDGE_GAP_PX
        # the raw pixel's outer boundary (backproject's convention: pixel i spans [i, i + 1)); else the kept point's own centre
        v = torch.where(use, raw.float() + (0. if top else 1.), kept.float() + (stride - 1) / 2)
        seen = raw != (0 if top else H - 1)
        m, col, zk, v = (k // Wc)[seen], (k % Wc)[seen], zk[seen], v[seen]
        f = fr[m]
        kk = K[f]
        u = col.float() * stride + (stride - 1) / 2
        cam = torch.stack([(u - kk[:, 0, 2]) / kk[:, 0, 0] * zk, (v - kk[:, 1, 2]) / kk[:, 1, 1] * zk, zk], 1)
        out[name] = (m, (c2w[f, :3, :3] @ cam[:, :, None])[:, :, 0] + c2w[f, :3, 3])
    return out


VB, VOFF = 1 << 21, 1 << 20
R4 = {"seam": True, "reproject": True, "join_guard": True}  # r4/instances switches (fast_report.instances); the offline replay turns them off
BRIDGE_SHARE = .25  # r4: a densify mask with this share of its voxels in a second object is over two things


def voxel_codes(world):
    import torch
    ijk = torch.floor(world / LIFT_VOXEL).long()
    return ((ijk[:, 0] + VOFF) * VB + (ijk[:, 1] + VOFF)) * VB + (ijk[:, 2] + VOFF)


def lift(masks, frame_of, depth_m, K, c2w_m, dyn, stride=2):
    """E7's lift (E9's code): pixels -> 5 cm voxels -> cross-frame voxel overlap -> connected components, on points
    cleaned first (mask_points: depth-edge pixels and a 1 px rim out, per-mask depth tails trimmed), so flying pixels
    neither bridge components nor stretch boxes (L1).
    -> (component per input mask, -1 = not lifted; per component arrays; stats; points: mask_points' dict plus label (L)
        component per lifted mask and obj_voxel (component, voxel code) pairs)."""
    import torch
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    dev, n = depth_m.device, len(depth_m)
    comp = np.full(len(masks), -1)
    p, stats = mask_points(masks, frame_of, depth_m, K, c2w_m, dyn, stride)
    if p is None or len(p["lifted"]) < 2:
        return comp, None, stats, None
    idx, mid, fr = p["lifted"], p["mid"], p["fr"]
    B, OFF = VB, VOFF
    vox, vid = torch.unique(voxel_codes(p["world"]), return_inverse=True)
    pair = torch.unique(mid * len(vox) + vid)
    pm, pv = pair // len(vox), pair % len(vox)
    size = torch.bincount(pm, minlength=len(idx)).float()
    A = torch.sparse_coo_tensor(torch.stack([pm, pv]), torch.ones_like(pm, dtype=torch.float32), (len(idx), len(vox))).coalesce()
    inter = torch.sparse.mm(A, A.t()).coalesce()
    (a, b), c = inter.indices(), inter.values()
    edge = (a < b) & (fr[a] != fr[b]) & (c >= MATCH_MIN * torch.minimum(size[a], size[b])) & (c >= MATCH_MAX * torch.maximum(size[a], size[b]))
    if R4["reproject"]:  # r4/instances: small things whose 5 cm voxels miss each other (depth noise > their size) link by projection
        t_ = time.perf_counter()
        ra, rb, rw = reproject_links(masks[idx][:, ::stride, ::stride], p["world"], mid, fr, depth_m, K, c2w_m, stride)
        stats.update(reproject_links=int(len(ra)), reproject_s=round(time.perf_counter() - t_, 3))
        if len(ra):
            a, b, c = torch.cat([a, ra]), torch.cat([b, rb]), torch.cat([c, torch.zeros_like(ra, dtype=c.dtype)])
            edge = torch.cat([edge, torch.ones_like(ra, dtype=torch.bool)])
            link_w = torch.cat([torch.zeros(int(edge.sum()) - len(ra), device=dev), rw])  # voxel links keep their IoU below
    if R4["seam"]:  # r4/instances: groups SAM 3 keeps as separate masks on >= SEAM_MIN keyframes stay apart, a mask over both is left out
        from fast_report import instances
        keep = (a < b) & ((fr[a] == fr[b]) | edge)
        w = c[edge] / (size[a[edge]] + size[b[edge]] - c[edge]).clamp(min=1)
        if R4["reproject"] and stats.get("reproject_links"):
            w = torch.where(c[edge] > 0, w, link_w)
        w = w.cpu().numpy()
        t_ = time.perf_counter()
        label, seam_rec = instances.seam_labels(len(idx), fr.cpu().numpy(), size.cpu().numpy(), (a[keep].cpu().numpy(), b[keep].cpu().numpy(),
                                                c[keep].cpu().numpy()), (a[edge].cpu().numpy(), b[edge].cpu().numpy(), w))
        ncomp = int(label.max()) + 1 if len(label) else 0
        stats["seams"] = {**seam_rec, "s": round(time.perf_counter() - t_, 3)}
        a = a[edge].cpu().numpy()
    else:
        a, b = a[edge].cpu().numpy(), b[edge].cpu().numpy()
        ncomp, label = connected_components(coo_matrix((np.ones(len(a)), (a, b)), shape=(len(idx), len(idx))), directed=False)
    lab = torch.from_numpy(label.astype(np.int64)).to(dev)  # scipy gives int32: lab * voxels overflows (X1, fx/x1-fps lift_big)
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
    return comp, arrays, stats, {**p, "label": lab, "obj_voxel": (oc, vox[ovid])}


REPROJ_HOPS, REPROJ_IN, REPROJ_COVER, REPROJ_DEPTH = 2, .5, .3, .1  # r4: next keyframes looked at, shares, depth agreement (DA3 ~5 %, x 2)


def reproject_links(m2, world, mid, fr, depth_m, K, c2w, stride=2):
    """r4/instances: links between masks of nearby keyframes by projection, not voxels: a mask's cleaned points projected into
    each of the next REPROJ_HOPS keyframes of this lift (points the depth there does not see, within REPROJ_DEPTH x the depth,
    left out) land >= REPROJ_IN on one mask there and cover >= REPROJ_COVER of its points, both ways (mutual). The projection
    of a nearby view barely moves with a depth error, so a 5 cm tool 4 m away (depth noise 20 cm, its voxels never meet)
    still links. m2: (L, H/stride, W/stride) raw masks; world/mid: the cleaned points; fr: local keyframe per mask.
    -> (a, b, weight) tensors, a < b, weight = the weaker of the four shares / 2 (below a voxel link's IoU)."""
    import torch
    dev = world.device
    L, h, w = m2.shape
    n_pts = torch.bincount(mid, minlength=L).float()
    frames = torch.unique(fr).tolist()
    lab = {}  # keyframe -> (h, w) lifted-mask index, -1 = none; the smaller mask wins
    for f in frames:
        sel = torch.nonzero(fr == f).squeeze(1)
        pm = paint(m2[sel])
        lab[f] = torch.where(pm > 0, sel[(pm - 1).clamp(min=0)], torch.full_like(pm, -1))
    pt_fr = fr[mid]
    fwd = {}  # (src, dst) -> points of src landing on dst
    for i, f in enumerate(frames):
        src = torch.nonzero(pt_fr == f).squeeze(1)
        if not len(src):
            continue
        for g in frames[max(0, i - REPROJ_HOPS):i] + frames[i + 1:i + 1 + REPROJ_HOPS]:
            w2c = torch.linalg.inv(c2w[g])
            cam = world[src] @ w2c[:3, :3].T + w2c[:3, 3]
            z = cam[:, 2]
            u = (K[g, 0, 0] * cam[:, 0] / z.clamp(min=1e-6) + K[g, 0, 2]) / stride
            v = (K[g, 1, 1] * cam[:, 1] / z.clamp(min=1e-6) + K[g, 1, 2]) / stride
            ui, vi = u.floor().long(), v.floor().long()
            ok = (z > 0) & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
            d = depth_m[g, (vi.clamp(0, h - 1) * stride), (ui.clamp(0, w - 1) * stride)]
            ok &= (d > 0) & ((z - d).abs() <= REPROJ_DEPTH * d)
            hit = torch.where(ok, lab[g][vi.clamp(0, h - 1), ui.clamp(0, w - 1)], torch.full_like(ui, -1))
            key = mid[src[ok]] * L + hit[ok]
            key = key[hit[ok] >= 0]
            k, cnt = torch.unique(key, return_counts=True)
            seen = torch.bincount(mid[src[ok]], minlength=L).float()
            for kk, cc in zip((k // L).tolist(), zip((k % L).tolist(), cnt.tolist())):
                if seen[kk] > 0:
                    fwd[(kk, cc[0])] = (cc[1] / float(seen[kk]), cc[1] / max(float(n_pts[cc[0]]), 1.))
    a, b, wt = [], [], []
    for (x, y), (in_xy, cov_xy) in fwd.items():
        if x < y and (y, x) in fwd:
            in_yx, cov_yx = fwd[(y, x)]
            m = min(in_xy, cov_xy, in_yx, cov_yx)
            if in_xy >= REPROJ_IN and in_yx >= REPROJ_IN and cov_xy >= REPROJ_COVER and cov_yx >= REPROJ_COVER:
                a.append(x), b.append(y), wt.append(m / 2)
    t = lambda x, dt=torch.long: torch.tensor(x, dtype=dt, device=dev)  # noqa: E731
    return t(a), t(b), t(wt, torch.float32)


def join(p, codes, owner):
    """Densify (click MVP section 7): new masks' cleaned points against the objects' voxel sets (codes sorted, owner = the
    object of each code): a mask joins the object holding the most of its voxels when that is >= MATCH_MIN of them (the
    lift's rule, mask to object). -> (object per lifted mask, -1 = none; its share)."""
    import torch
    L = len(p["lifted"])
    vox, vid = torch.unique(voxel_codes(p["world"]), return_inverse=True)
    pair = torch.unique(p["mid"] * len(vox) + vid)
    pm, pv = pair // len(vox), pair % len(vox)
    size = torch.bincount(pm, minlength=L).float()
    if not len(codes):
        return torch.full((L,), -1, dtype=torch.long, device=vox.device), torch.zeros(L, device=vox.device)
    pos = torch.searchsorted(codes, vox).clamp(max=len(codes) - 1)
    ob = torch.where(codes[pos] == vox, owner[pos], torch.full_like(pos, -1))[pv]
    ok = ob >= 0
    n_obj = int(owner.max()) + 1
    key, cnt = torch.unique(pm[ok] * n_obj + ob[ok], return_counts=True)
    km, ko = key // n_obj, key % n_obj
    best = torch.zeros(L, device=vox.device).scatter_reduce(0, km, cnt.float(), "amax")
    who = torch.full((L,), -1, dtype=torch.long, device=vox.device)
    top = cnt.float() == best[km]
    who[km[top]] = ko[top]
    share = best / size.clamp(min=1)
    ok = share >= MATCH_MIN
    if R4["join_guard"]:  # r4/instances: a mask over two objects (its second object holds >= BRIDGE_SHARE of it) joins neither
        second = torch.zeros(L, device=vox.device).scatter_reduce(0, km[~top], cnt[~top].float(), "amax")
        ok &= second / size.clamp(min=1) < BRIDGE_SHARE
    return torch.where(ok, who, torch.full_like(who, -1)), share


def pack(m):
    """(n,H,W) bool -> (n,H,W/8) uint8 numpy, np.packbits' bit order (X1's layout: densify masks wait on the CPU)."""
    import torch
    w = torch.tensor([128, 64, 32, 16, 8, 4, 2, 1], dtype=torch.uint8, device=m.device)
    return (m.reshape(m.shape[0], m.shape[1], m.shape[2] // 8, 8).to(torch.uint8) * w).sum(-1, dtype=torch.uint8).cpu().numpy()


def unpack(p, dev):
    import torch
    x = torch.from_numpy(np.ascontiguousarray(p)).to(dev)
    w = torch.tensor([128, 64, 32, 16, 8, 4, 2, 1], dtype=torch.uint8, device=dev)
    return ((x[..., None] & w) > 0).reshape(x.shape[0], x.shape[1], x.shape[2] * 8)


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


def label_rle(lab):
    """(h,w) int label map (0 = nothing, < 65536) -> uint16 little-endian (value, run) pairs in raster order, runs over
    65535 split (panoptes-pick-v1)."""
    x = np.asarray(lab).ravel()
    starts = np.r_[0, np.flatnonzero(np.diff(x)) + 1]
    runs = np.diff(np.r_[starts, len(x)])
    reps = (runs + 65534) // 65535
    r = np.full(int(reps.sum()), 65535, np.int64)
    r[np.cumsum(reps) - 1] = runs - (reps - 1) * 65535
    return np.stack([np.repeat(x[starts], reps), r], 1).astype("<u2").ravel()


def pick_frame(lab):
    """One pick map (process pool): (RLE bytes, entity values present, their pixel counts)."""
    v, c = np.unique(lab, return_counts=True)
    return label_rle(lab).tobytes(), v.tolist(), c.tolist()


def rle_decode(pairs, h, w):
    p = np.asarray(pairs, np.int64).reshape(-1, 2)
    return np.repeat(p[:, 0], p[:, 1]).reshape(h, w)


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
    lab = np.zeros((360, 640), np.int64)
    lab[10:20, 5:600] = 3
    lab[200:, :] = 65535  # a run longer than 65535 pixels is split
    enc = label_rle(lab)
    assert enc.dtype == np.dtype("<u2") and np.array_equal(rle_decode(enc, 360, 640), lab) and (enc.reshape(-1, 2)[:, 1] > 0).all()
    try:
        import torch
    except ImportError:
        print("segment self-check ok: naming, polygons, pick RLE (flood rules, cleaning, trim skipped: no torch here)")
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
    # cleaning: a depth step inside mask 0 drops the edge pixels there; a 1 px rim goes; a 2 px wide mask keeps its pixels
    depth = torch.full((2, 280, 504), 4.)
    depth[0, :, 30:] = 6.
    thin = torch.zeros((1, 280, 504), dtype=torch.bool)
    thin[0, 100:200, 300:302] = True
    cm = clean(torch.cat([masks[:1], thin]), torch.tensor([0, 1]), depth)
    full = masks[0, ::2, ::2]
    assert cm.shape == (2, 140, 252) and not cm[0, 7:25, 14].any() and not cm[0, 7:25, 5].any() and cm[0, 12, 10] and cm[0].sum() < full.sum()
    assert cm[1].sum() == thin[0, ::2, ::2].sum(), "a mask the erosion would empty keeps its pixels"
    z = torch.tensor([1., 1.02, 1.01, .99, 1., 3., 2., 2.01, 2.02, 1.99, 2., 2.])
    keep = trim(z, torch.tensor([0] * 6 + [1] * 6), 2)
    assert keep.tolist() == [True] * 5 + [False] + [True] * 6, keep  # a flying 3 m point in a 1 m mask goes
    bits = torch.rand(3, 280, 504) > .5
    assert torch.equal(unpack(pack(bits), "cpu"), bits) and np.array_equal(pack(bits), np.packbits(bits.numpy(), axis=2))
    # join: object 0 owns voxels around x = 0, object 1 around x = 1; mask 0 lies 80 % in object 1, mask 1 nowhere
    w0 = torch.tensor([[1.01 + .05 * i, 0., 0.] for i in range(5)])
    w1 = torch.tensor([[3. + .05 * i, 0., 0.] for i in range(4)])
    codes = torch.cat([voxel_codes(torch.tensor([[.01 + .05 * i, 0., 0.] for i in range(4)])), voxel_codes(w0[:4])])
    order = torch.argsort(codes)
    who, share = join({"lifted": torch.arange(2), "mid": torch.tensor([0] * 5 + [1] * 4), "world": torch.cat([w0, w1])},
                      codes[order], torch.tensor([0] * 4 + [1] * 4)[order])
    assert who.tolist() == [1, -1] and abs(float(share[0]) - .8) < 1e-6, (who, share)
    # mvp2 edges: an 80 x 100 px face at 4 m before a far wall; the cleaning shaves its rim (4-5 cm here), the raw edge at the
    # face's own depth gives the pixel boundary back; a face cut by the image top gives no top edge
    dm = torch.full((2, 280, 504), 8.)
    dm[:, 100:180, 200:300] = 4.
    dm[1, :180, 200:300] = 4.
    face = torch.zeros((2, 280, 504), dtype=torch.bool)
    face[0, 100:180, 200:300] = True
    face[1, :180, 200:300] = True
    Kt = torch.tensor([[262., 0, 252], [0, 262., 140], [0, 0, 1]]).repeat(2, 1, 1)
    p, _ = mask_points(face, torch.tensor([0, 1]), dm, Kt, torch.eye(4).repeat(2, 1, 1), torch.zeros_like(face))
    want = (100 - 140) / 262 * 4., (180 - 140) / 262 * 4.
    ys = p["world"][p["mid"] == 0, 1]
    assert ys.min() > want[0] + .02 and ys.max() < want[1] - .02, "the cleaning shaves the rim"
    (mt, top), (mb, bot) = p["edge"]["top"], p["edge"]["bottom"]
    assert (mt == 0).all() and (mt == 0).sum() >= 45 and (top[:, 1] - want[0]).abs().max() < 1e-4, top[:, 1]
    assert ((mb == 0).sum() >= 45 and (mb == 1).sum() >= 45 and (bot[:, 1] - want[1]).abs().max() < 1e-4 and (bot[:, 2] - 4.).abs().max() < 1e-4)
    print("segment self-check ok: flood merge/drop, naming, polygons, pick RLE, depth-edge cleaning, depth-tail trim, mask packing, densify join, "
          "un-eroded edges at the object's depth")
