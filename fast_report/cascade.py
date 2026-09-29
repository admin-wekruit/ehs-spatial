"""Object naming cascade on top of SAM 3 (the user's request): cheap and certain first, the big VLM only for the rest.

  1. every member mask of a confirmed object -> a masked crop -> SigLIP 2 image embedding (batched on the GPU); the
     object's embedding is the mean of its masks' (the 3D lift already joined the same object across nearby frames)
  2. cache: the persistent cross-video cache (earlier VLM answers, keyed by embedding, never this same video)
  3. zero-shot SigLIP 2 probabilities over this video's vocabulary: accept when top-1 >= P_MIN, top-1 - top-2 >= MARGIN
     and top-1 is not a generic word ('tool', 'metal part': the report wants the specific name, spec section 12)
  4. cache: objects of THIS video already settled (repeated objects: same cabinet model, second shot of a machine)
  5. the rest -> clustered by embedding, one crop per cluster to Qwen3-VL (vLLM); answers go back to both caches

The cache threshold is calibrated on each video without any name: two objects seen together on one keyframe are two
objects, so their cosines are known negatives, and the threshold is their 99th percentile (never below CACHE_FLOOR). Run
001 showed why: at a fixed 0.92, masked crops of different objects of one video matched each other (61 objects became
'handle'). P_MIN and MARGIN are set before any run and never tuned on ME340's names: the run saves every probability,
so the trade-off between accuracy and the share sent to the VLM is read off afterwards (--evaluate).
Classes are model-resident with batch methods (the shape of a Ray actor); they run as threads in the one container
because every input is already a CUDA tensor there (an actor would copy it through the object store).
"""
import json
import time
from pathlib import Path

import numpy as np

SIGLIP, SIGLIP_REV = "google/siglip2-base-patch16-224", None
TEMPLATE = "a photo of a {}."
P_MIN, MARGIN, CACHE_FLOOR, NEGATIVE_Q = .5, .25, .9, .99  # a priori; see the docstring
CROP_PAD, BATCH = .1, 512


class Embedder:
    """SigLIP 2 (ViT-B/16, 224) resident on one GPU. crops(): masked crops of many masks at once; text(): prompts."""

    def __init__(self, dev, cache_dir):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.dev = dev
        self.model = AutoModel.from_pretrained(SIGLIP, cache_dir=cache_dir, torch_dtype=torch.bfloat16).to(dev).eval()
        self.tok = AutoTokenizer.from_pretrained(SIGLIP, cache_dir=cache_dir)
        self.side = self.model.config.vision_config.image_size
        self.scale = float(self.model.logit_scale.exp())
        self.dim = self.text(["object"]).shape[1]

    @staticmethod
    def _pooled(out):
        return out.pooler_output if hasattr(out, "pooler_output") else out

    def text(self, words):
        import torch
        t = self.tok([TEMPLATE.format(w) for w in words], padding="max_length", max_length=64, truncation=True, return_tensors="pt").to(self.dev)
        with torch.inference_mode():
            e = self._pooled(self.model.get_text_features(**t)).float()
        return torch.nn.functional.normalize(e, dim=-1)

    def crops(self, frames, frame_of, masks, masked=True):
        """frames (N,720,1280,3) uint8 BGR on dev; frame_of (M,) index into frames; masks (M,h,w) bool on dev at any grid
        over the same 16:9 view. Box around the mask + CROP_PAD, squared, background outside the mask set to mid grey
        (SigLIP's normalisation zero) when `masked`, resized to side x side -> (M,D) unit embeddings."""
        import torch
        from torchvision.ops import roi_align
        m_count, (h, w) = len(masks), masks.shape[1:]
        H, W = frames.shape[1:3]
        out = []
        ys = torch.arange(h, device=self.dev)
        xs = torch.arange(w, device=self.dev)
        for s in range(0, m_count, BATCH):
            mk = masks[s:s + BATCH]
            rows, cols = mk.any(2), mk.any(1)
            y0 = torch.where(rows, ys, h).amin(1).float()
            y1 = torch.where(rows, ys, -1).amax(1).float() + 1
            x0 = torch.where(cols, xs, w).amin(1).float()
            x1 = torch.where(cols, xs, -1).amax(1).float() + 1
            sy, sx = H / h, W / w
            cy, cx = (y0 + y1) / 2 * sy, (x0 + x1) / 2 * sx
            half = torch.maximum((y1 - y0) * sy, (x1 - x0) * sx) * (1 + 2 * CROP_PAD) / 2
            boxes = torch.stack([cx - half, cy - half, cx + half, cy + half], 1)
            fi = frame_of[s:s + BATCH].float()[:, None]
            img = roi_align(self._rgb(frames), torch.cat([fi, boxes], 1), self.side, 1., aligned=True)
            mm = roi_align(mk[:, None].float(), torch.cat([torch.arange(len(mk), device=self.dev)[:, None].float(),
                                                             boxes / torch.tensor([sx, sy, sx, sy], device=self.dev)], 1),
                           self.side, 1., aligned=True) > .5
            x = (torch.where(mm, (img - .5) / .5, 0.) if masked else (img - .5) / .5).to(torch.bfloat16)
            with torch.inference_mode():
                out.append(torch.nn.functional.normalize(self._pooled(self.model.get_image_features(pixel_values=x)).float(), dim=-1))
        return torch.cat(out) if out else torch.zeros((0, self.dim), device=self.dev)

    def _rgb(self, frames):
        """(N,H,W,3) uint8 BGR -> (N,3,H,W) float RGB in [0,1], kept for the run (the frames do not change)."""
        key = (frames.data_ptr(), frames.shape)
        if getattr(self, "_rgb_key", None) != key:
            self._rgb_key, self._rgb_val = key, frames.permute(0, 3, 1, 2).flip(1).float().div_(255)
        return self._rgb_val

    def release(self):
        self._rgb_key = self._rgb_val = None

    def zero_shot(self, img, txt):
        """Softmax over the vocabulary of logit_scale x cosine (SigLIP's bias cancels in a softmax)."""
        return (self.scale * img @ txt.T).softmax(-1)


class LabelCache:
    """Persistent labels keyed by embedding: every VLM answer, with the video it came from. Lookups skip entries of the
    video being analysed. ponytail: brute-force cosine over all entries; an ANN index when it passes ~1e5 entries."""

    def __init__(self, path):
        self.path = Path(path)
        try:
            z = np.load(self.path)
            self.emb, self.meta = z["emb"].astype(np.float32), json.loads(str(z["meta"]))
        except (OSError, ValueError, KeyError):
            self.emb, self.meta = np.zeros((0, 0), np.float32), []
        self.new = 0

    def __len__(self):
        return len(self.meta)

    def nearest(self, emb, video_sha):
        """emb (M,D) unit -> (cos (M,), meta or None per row); -1 where nothing usable."""
        m = len(emb)
        usable = np.array([x["video"] != video_sha for x in self.meta], bool)
        if not usable.any() or self.emb.shape[1] != emb.shape[1]:
            return np.full(m, -1.), [None] * m
        e = self.emb[usable]
        meta = [x for x, u in zip(self.meta, usable) if u]
        sims = emb @ e.T
        best = sims.argmax(1)
        return sims[np.arange(m), best], [meta[i] for i in best]

    def add(self, emb, labels, video_sha, site, source):
        if not len(emb):
            return
        self.emb = np.concatenate([self.emb.reshape(-1, emb.shape[1]), emb.astype(np.float32)])
        self.meta += [{"label": l, "video": video_sha, "site": site, "source": source, "t": round(time.time())} for l in labels]
        self.new += len(labels)

    def save(self):
        if not self.new:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp.npz")
        np.savez(tmp, emb=self.emb.astype(np.float16), meta=json.dumps(self.meta))
        tmp.replace(self.path)


def calibrate(emb, together):
    """together: lists of objects seen on one keyframe (distinct objects: known negatives). -> (threshold, record)."""
    a, b = [], []
    for objs in together:
        objs = sorted(set(objs))
        for i in range(len(objs)):
            a += [objs[i]] * (len(objs) - i - 1)
            b += objs[i + 1:]
    if not a:
        return 1.01, {"negative_pairs": 0, "threshold": None, "note": "no evidence: cache off"}
    cos = (emb[a] * emb[b]).sum(1)
    q = np.quantile(cos, [.5, .9, NEGATIVE_Q, 1.])
    tau = max(CACHE_FLOOR, float(q[2]))
    return tau, {"negative_pairs": len(a), "negative_cos_q50_q90_q99_max": [round(float(x), 4) for x in q], "threshold": round(tau, 4)}


def decide(obj_emb, probs, words, sam3_names, cache, video_sha, tau, generic, use_cache=True):
    """Steps 2-4 for every object -> (records, uncertain object indices). obj_emb (n,D) unit numpy; probs (n,V) numpy;
    generic(word) -> bool."""
    n = len(obj_emb)
    rec = [None] * n
    order = np.argsort(-probs, 1)[:, :3] if len(probs) else np.zeros((0, 3), int)
    top = [[(words[j], round(float(probs[i, j]), 4)) for j in order[i]] for i in range(n)]
    cos, hit = cache.nearest(obj_emb, video_sha) if use_cache else (np.full(n, -1.), [None] * n)
    for i in range(n):
        p1 = top[i][0][1] if top[i] else 0.
        p2 = top[i][1][1] if len(top[i]) > 1 else 0.
        base = {"sam3": sam3_names[i], "zero_shot_top3": top[i], "cross_cos": round(float(cos[i]), 4)}
        if cos[i] >= tau:
            rec[i] = {**base, "label": hit[i]["label"], "source": "cache:cross-video", "from_site": hit[i]["site"]}
        elif p1 >= P_MIN and p1 - p2 >= MARGIN and not generic(top[i][0][0]):
            rec[i] = {**base, "label": top[i][0][0], "source": "zero-shot"}
        else:
            rec[i] = {**base, "label": None, "source": None}
    settled = np.array([r["label"] is not None for r in rec], bool)
    for i in np.flatnonzero(~settled):
        if settled.any():
            s = obj_emb[settled] @ obj_emb[i]
            j = np.flatnonzero(settled)[s.argmax()]
            if s.max() >= tau:
                rec[i].update(label=rec[j]["label"], source="cache:in-video", repeat_of=int(j), repeat_cos=round(float(s.max()), 4))
    return rec, [i for i in range(n) if rec[i]["label"] is None]


def clusters(emb, idx, tau):
    """Greedy: each uncertain object joins the first representative within tau -> {rep: [members]}."""
    reps = {}
    for i in idx:
        for r in reps:
            if float(emb[r] @ emb[i]) >= tau:
                reps[r].append(i)
                break
        else:
            reps[i] = [i]
    return reps


def self_check():
    import tempfile
    rng = np.random.default_rng(0)
    e = rng.normal(size=(6, 8))
    e /= np.linalg.norm(e, axis=1, keepdims=True)
    e[3] = e[0] + .05 * rng.normal(size=8)  # a repeat of object 0
    e[3] /= np.linalg.norm(e[3])
    words = ["cabinet", "lathe", "vise", "tool"]
    probs = np.array([[.9, .05, .05, 0], [.4, .35, .25, 0], [.1, .8, .1, 0], [.4, .3, .3, 0], [.34, .33, .33, 0], [.05, .05, 0, .9]])
    generic = lambda w: w == "tool"  # noqa: E731
    tau, rec_ = calibrate(e, [[0, 1, 2], [1, 4, 5]])
    assert rec_["negative_pairs"] == 6 and tau >= CACHE_FLOOR, rec_
    assert calibrate(e, [[0]])[0] > 1, "no negatives: the cache stays off"
    with tempfile.TemporaryDirectory() as d:
        cache = LabelCache(Path(d) / "c.npz")
        assert len(cache) == 0
        rec, unsure = decide(e, probs, words, ["box"] * 6, cache, "v1", .95, generic)
        assert [r["source"] for r in rec[:4]] == ["zero-shot", None, "zero-shot", "cache:in-video"], rec
        assert rec[3]["label"] == "cabinet" and unsure == [1, 4, 5], unsure  # 5: 'tool' is sure but generic
        cache.add(e[[1]], ["drill press"], "v0", "shop", "vlm")
        cache.add(e[[4]], ["own answer"], "v1", "shop", "vlm")  # same video: never used for v1
        cache.save()
        again = LabelCache(Path(d) / "c.npz")
        rec, unsure = decide(e, probs, words, ["box"] * 6, again, "v1", .95, generic)
        assert rec[1]["source"] == "cache:cross-video" and rec[1]["label"] == "drill press" and 4 in unsure, rec
    assert clusters(e, [0, 3, 1], .95) == {0: [0, 3], 1: [1]}
    print("cascade self-check ok: calibrated threshold, accept rule (no generic words), cross-video cache skips its own video, "
          "in-video repeats, clustering")
