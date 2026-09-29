"""X9: same object -> do not re-segment. The reuse ladder over the fast core's poses + depth, one 15 fps frame at a time:

  1. predict  known objects into the new frame: E6b's pair splat of the two previous outputs (their per-pixel depth,
              z-buffered), plus the 5 cm voxels of known objects missing from both (memory: re-entering, un-occluded);
              every predicted pixel is depth-checked against the new frame's depth: occluded (something in front) or
              conflicting (the surface seen is farther: the object is gone) pixels are not predicted.
  2. check    each prediction cheaply: DINOv2 (masked pooling of one dense forward per frame) vs the object's cached
              embedding, depth agreement, visible share, edge agreement, look-alike margin -> logistic p(right),
              calibrated on labelled rows of other videos (probe runs, SAM 3 masks of the same frame as truth).
  3. agree    same object: the name is reused; the mask is the projection, or ONE SAM 3 tracker box prompt around it.
  4. new      pixels no accepted prediction explains (not seen by any segmented frame, or a rejected prediction):
              past a share, SAM 3's text search runs on the frame; its masks join known objects (2D IoU with the
              prediction, then 3D voxel overlap, then appearance re-identification of a recently lost object = moved).
  5. grey     -> a decider: SAM 3 (re-segment), reuse, or Qwen3-VL (two outlined crops, 'same object?').
  6. refresh  every N frames SAM 3 runs anyway; deformable/moving classes are re-detected on every frame, never reused.

Everything here is torch on one GPU (or the CPU in the self-check); numbers are in the DA3 grid (280 x 504).
"""
import math
import time

import numpy as np

HW = (280, 504)
TOL_REL, TOL_ABS = .10, .05          # depth test: |z_pred - z| <= 10 % + 5 cm (DA3 cross-view depth spread ~5 %, X1)
MIN_PX, MIN_CHECK, MIN_BOX = 30, 40, 40  # DA3-grid pixels: a mask at all / checked / box-prompted
MATCH_MIN, MATCH_MAX, VOX_MATCH, REID_COS, REID_WINDOW = .5, .2, .5, .85, 10
COVER_FRAMES, OPEN_PX = 3, 2
THETA_BAD = .3  # rejected predictions mostly mark what SAM 3 would not segment on this frame either (probe rows)
MEM_MIN_VOX, MEM_CAP = 6, 3000
FEATURES = ["cos", "depth_ok", "vis_share", "edge_rel", "log_area", "age", "margin", "memory"]
LIFT_VOXEL = .05


# ---------- numpy: calibration and identity metrics (local self-check) ----------

def fit_logistic(X, y, l2=1e-2, iters=50):
    """IRLS logistic regression on standardised features -> {'mu','sd','w','b'}."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.c_[(X - mu) / sd, np.ones(len(X))]
    w = np.zeros(Z.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(Z @ w, -30, 30)))
        g = Z.T @ (p - y) + l2 * np.r_[w[:-1], 0]
        H = (Z * (p * (1 - p))[:, None]).T @ Z + l2 * np.eye(len(w))
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return {"mu": mu.tolist(), "sd": sd.tolist(), "w": w[:-1].tolist(), "b": float(w[-1])}


def predict_logistic(c, X):
    X = np.asarray(X, float)
    z = ((X - np.asarray(c["mu"])) / np.asarray(c["sd"])) @ np.asarray(c["w"]) + c["b"]
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def auc(p, y):
    """Mann-Whitney AUC."""
    p, y = np.asarray(p, float), np.asarray(y, bool)
    if y.all() or (~y).all():
        return None
    r = np.argsort(np.argsort(p)) + 1.
    for v in np.unique(p):  # ties: average ranks
        m = p == v
        r[m] = r[m].mean()
    n1, n0 = y.sum(), (~y).sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def truth_label(iou, hi=.5, lo=.2):
    """1 = the prediction is on a SAM 3 mask of this frame (IoU >= .5), 0 = on none (< .2), None = in between."""
    return 1 if iou >= hi else 0 if iou < lo else None


def delivered_box_to_da3(box):
    """Delivered raster box (640x480 = the 4:3 centre of 1280x720, scaled by 2/3) -> DA3 grid (504x280 of 16:9)."""
    x0, y0, x1, y1 = box
    sx, sy = HW[1] / 1280, HW[0] / 720
    return ((160 + x0 * 1.5) * sx, y0 * 1.5 * sy, (160 + x1 * 1.5) * sx, y1 * 1.5 * sy)


def identity_metrics(seqs, lookalike=frozenset()):
    """seqs: {entity: [(frame, our id or None)]} (sorted). -> switches (our id changes between consecutive matched
    observations), fragments (distinct our ids per entity), coverage (matched share), look-alike switches."""
    obs = matched = switches = look = 0
    frag = []
    for seq in seqs.values():
        ids = [i for _, i in seq if i is not None]
        obs += len(seq)
        matched += len(ids)
        sw = [(a, b) for a, b in zip(ids, ids[1:]) if a != b]
        switches += len(sw)
        look += sum((min(a, b), max(a, b)) in lookalike for a, b in sw)
        if len(ids) >= 5:
            frag.append(len(set(ids)))
    return {"entities": len(seqs), "observations": obs, "matched": matched, "coverage": round(matched / max(obs, 1), 4),
            "switches": switches, "switches_per_100_matched": round(100 * switches / max(matched, 1), 2),
            "lookalike_switches": look, "entities_5plus": len(frag), "ids_per_entity_mean": round(float(np.mean(frag)), 3) if frag else None,
            "ids_per_entity_median": float(np.median(frag)) if frag else None}


def numpy_self_check():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(4000, 3))
    y = (X @ [2., -1., 0.] + .5 + rng.logistic(size=4000) > 0).astype(float)
    c = fit_logistic(X, y)
    p = predict_logistic(c, X)
    assert auc(p, y) > .75 and abs(p.mean() - y.mean()) < .02, (auc(p, y), p.mean(), y.mean())
    w = np.asarray(c["w"]) / np.asarray(c["sd"])
    assert w[0] > 1.5 and w[1] < -.7 and abs(w[2]) < .2, w
    assert auc([.1, .4, .35, .8], [0, 0, 1, 1]) == .75
    assert truth_label(.6) == 1 and truth_label(.1) == 0 and truth_label(.3) is None
    b = delivered_box_to_da3((0, 0, 640, 480))
    assert abs(b[0] - 63) < 1e-6 and abs(b[2] - 441) < 1e-6 and abs(b[3] - 280) < 1e-6
    m = identity_metrics({"e1": [(0, 1), (3, 1), (6, 2), (9, None), (12, 2), (15, 2)], "e2": [(0, 5), (3, 6)]}, frozenset({(5, 6)}))
    assert m["switches"] == 2 and m["lookalike_switches"] == 1 and m["ids_per_entity_mean"] == 2 and m["coverage"] == .875, m
    print("x9 numpy self-check ok: logistic calibration, AUC, truth labels, delivered boxes, identity metrics")


# ---------- torch helpers ----------

def backproject(z, K, c2w, vy, vx):
    import torch
    x = (vx.float() - K[0, 2]) / K[0, 0] * z
    y = (vy.float() - K[1, 2]) / K[1, 1] * z
    return torch.stack([x, y, z], 1) @ c2w[:3, :3].T + c2w[:3, 3]


def splat(world, ids, K, w2c):
    from fast_report import segment as sg
    return sg.splat(world, ids, K, w2c, HW)


def finish_with_z(key):
    """Packed splat keys -> (label (-1 none, holes <= 3 px filled), z of the directly splatted pixels (nan elsewhere))."""
    import torch
    from fast_report import segment as sg
    lab = sg.finish(key, HW)
    z = torch.where(key == sg.EMPTY, torch.nan, (key >> 20).float() * 1e-4).view(HW)
    return lab, z


def open_mask(m, r=OPEN_PX):
    import torch.nn.functional as F
    k = 2 * r + 1
    x = m[None, None].float()
    x = -F.max_pool2d(-x, k, 1, r)
    return F.max_pool2d(x, k, 1, r)[0, 0] > 0


def erode(m, r=1):
    import torch.nn.functional as F
    return -F.max_pool2d(-m[:, None].float(), 2 * r + 1, 1, r)[:, 0] > 0


def dilate(m, r):
    import torch.nn.functional as F
    x = F.max_pool2d(m[:, None].float(), (1, 2 * r + 1), 1, (0, r))  # separable: rows, then columns
    return F.max_pool2d(x, (2 * r + 1, 1), 1, (r, 0))[:, 0] > 0


def pool(feats, masks):
    """feats (h,w,C) (DINOv2 patch grid), masks (k,H,W) bool -> (k,C) unit embeddings (area-weighted mean of patches)."""
    import torch
    import torch.nn.functional as F
    h, w, C = feats.shape
    if not len(masks):
        return torch.zeros((0, C), device=feats.device)
    W = F.adaptive_avg_pool2d(masks[:, None].float(), (h, w)).flatten(1)
    e = W @ feats.reshape(h * w, C).float()
    return F.normalize(e / W.sum(1, keepdim=True).clamp(min=1e-6), dim=1)


def edge_ratio(grad, masks):
    """mean image gradient on each mask's boundary / on its 5 px surroundings (an outline on real edges > 1)."""
    import torch
    if not len(masks):
        return torch.zeros(0, device=grad.device)
    bnd = masks & ~erode(masks)
    ring = dilate(masks, 5)
    g = grad.flatten()
    eb = (bnd.flatten(1).float() @ g) / bnd.flatten(1).sum(1).clamp(min=1)
    er = (ring.flatten(1).float() @ g) / ring.flatten(1).sum(1).clamp(min=1)
    return eb / er.clamp(min=1e-6)


def iou_matrix(a, b, stride=2):
    """(m,H,W) x (k,H,W) bool -> (m,k) IoU on a stride grid."""
    import torch
    if not len(a) or not len(b):
        return torch.zeros((len(a), len(b)), device=a.device if len(a) else b.device)
    A = a[:, ::stride, ::stride].flatten(1).float()
    B = b[:, ::stride, ::stride].flatten(1).float()
    inter = A @ B.T
    return inter / (A.sum(1)[:, None] + B.sum(1)[None] - inter).clamp(min=1)


def pack_bits(m):
    """(n,H,W) bool -> (n,H,W/8) uint8 on the CPU (np.packbits order); fresh masks kept for crops only."""
    import torch
    w = torch.tensor([128, 64, 32, 16, 8, 4, 2, 1], dtype=torch.uint8, device=m.device)
    return (m.reshape(m.shape[0], m.shape[1], m.shape[2] // 8, 8).to(torch.uint8) * w).sum(-1, dtype=torch.uint8).cpu().numpy()


def fresh_mask(d):
    """an object's last fresh mask (numpy bool H x W) and its frame."""
    t, store, j = d["fresh_mask"]
    return t, np.unpackbits(store[j], axis=-1).astype(bool)


def vox_codes(world):
    import torch
    ijk = torch.floor(world / LIFT_VOXEL).long()
    B, OFF = 1 << 21, 1 << 20
    return ((ijk[:, 0] + OFF) * B + (ijk[:, 1] + OFF)) * B + (ijk[:, 2] + OFF)


def code_centres(codes):
    import torch
    B, OFF = 1 << 21, 1 << 20
    return torch.stack([codes // (B * B) - OFF, (codes // B) % B - OFF, codes % B - OFF], 1).float() * LIFT_VOXEL + LIFT_VOXEL / 2


def paint_ids(masks, ids):
    """(k,H,W) bool + ids -> (H,W) int32 map, 0 = nothing, id + 1 (smaller mask wins where they overlap)."""
    import torch
    from fast_report import segment as sg
    if not len(masks):
        return torch.zeros(HW, dtype=torch.int32, device=masks.device)
    lab = sg.paint(masks)
    table = torch.cat([torch.zeros(1, dtype=torch.long, device=masks.device), torch.as_tensor(ids, device=masks.device).long() + 1])
    return table[lab].int()


# ---------- the ladder ----------

class Ladder:
    """One configuration over one video. S (the site) gives: shots {si: [15 fps frames]}, geo(f) -> (depth_m, K, c2w_m),
    dyn(f), sam(f) -> dict(masks, word, score, idx), votes(idx) -> [(word, score)], feats(f), grad(f), words,
    always (set of word indices), tracker (optional box prompts: box(f, boxes) -> (masks, scores)), qwen (optional)."""

    def __init__(self, S, cfg, calib=None, log_rows=False, qwen=None, clock=None):
        import torch
        self.S, self.cfg, self.calib, self.log_rows, self.qwen = S, cfg, calib, log_rows, qwen
        self.dev = S.dev
        self.rows, self.events, self.frames_log, self.qwen_rows, self.confusion, self.truth_s = [], [], [], [], {}, 0.
        self.event_refs, self.qwen_refs = [], []  # for the contact sheets
        self.t = {k: 0. for k in ("predict", "check", "cover", "sam_assoc", "box", "qwen", "paint", "update")}
        self.out = {}
        self.next_id = 0
        self.objs = {}  # id -> dict
        self.E = torch.zeros((0, 768), device=self.dev)
        self.counts = {k: 0 for k in ("frames", "sam_frames", "sam_forced", "sam_triggered", "pred", "checked", "agree", "reject", "grey", "grey_agree",
                                      "grey_reject", "small_carried", "memory_pred", "tightened", "box_prompts", "fresh", "new_objects", "reid_moved",
                                      "matched_2d", "matched_3d", "deformable_redetected", "dropped_unmatched_on_sam", "qwen_calls")}

    # --- object state ---
    def emb_rows(self, need):
        import torch
        if need > len(self.E):
            self.E = torch.cat([self.E, torch.zeros((max(need - len(self.E), 256), self.E.shape[1]), device=self.dev)])

    def new_obj(self, si):
        i = self.next_id
        self.next_id += 1
        self.emb_rows(i + 1)
        self.objs[i] = {"shot": si, "votes": {}, "fresh": 0, "passed": 0, "last_fresh": None, "last_seen": None, "edge": 1., "cents": [],
                        "fresh_mask": None, "rejected_at": None, "reject_reason": None, "occluded_since": None, "occ_episodes": 0, "occ_kept": 0}
        self.counts["new_objects"] += 1
        return i

    def deformable(self, i):
        v = self.objs[i]["votes"]
        return bool(v) and max(v, key=v.get) in self.S.always

    # --- one video ---
    def run(self):
        import torch
        with torch.inference_mode():
            for si, frames in self.S.shots.items():
                self.vox_code = torch.zeros(0, dtype=torch.long, device=self.dev)
                self.vox_owner = torch.zeros(0, dtype=torch.long, device=self.dev)
                self.cover_src, self.prev, self.obj_codes = [], [], {}
                for i, t in enumerate(frames):
                    self.step(si, i, t, frames)
        return self

    def timed(self, key):
        import contextlib
        import torch
        dev = self.dev

        @contextlib.contextmanager
        def ctx():
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
            t0 = time.perf_counter()
            yield
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
            self.t[key] += time.perf_counter() - t0
        return ctx()

    def step(self, si, i, t, frames):
        import torch
        S, cfg, dev = self.S, self.cfg, self.dev
        z_t, K_t, c_t = S.geo(t)
        dyn = S.dyn(t)
        valid = (z_t > 0) & ~dyn
        w2c = torch.linalg.inv(c_t)
        self.counts["frames"] += 1
        log = {"frame": t, "shot": si}

        # 1. predict
        with self.timed("predict"):
            pids, pvis, pstat, pmem, pproj = self.predict(si, i, t, frames, z_t, K_t, w2c, dyn)
        self.counts["pred"] += len(pids)
        self.counts["memory_pred"] += int(pmem.sum()) if len(pids) else 0

        # 2. check
        with self.timed("check"):
            decision, p, feats = self.check(t, pids, pvis, pstat, pmem)
        pmask = {j: pvis[j] if decision[j] != "reject" else pproj[j] for j in range(len(pids))}
        # 5. grey -> decider
        orig = list(decision)
        grey = [j for j, d in enumerate(decision) if d == "grey"]
        if grey and cfg["grey"] == "qwen" and self.qwen is not None:
            with self.timed("qwen"):
                yes = self.qwen(self, t, [pids[j] for j in grey], pvis[grey])
            self.counts["qwen_calls"] += len(grey)
            for j, y in zip(grey, yes):
                decision[j] = "agree" if y else "reject"
                self.counts["grey_agree" if y else "grey_reject"] += 1
        elif grey and cfg["grey"] == "reuse":
            for j in grey:
                decision[j] = "agree"
                self.counts["grey_agree"] += 1
        for d in decision:
            if d in self.counts:
                self.counts[d] += 1
        for j, d in enumerate(decision):
            o = self.objs[pids[j]]
            if d == "reject":
                o["rejected_at"], o["reject_reason"] = t, feats[j] if feats is not None else None
            if d in ("agree", "small"):
                o["passed"] += 1

        # 4. what no accepted prediction explains
        with self.timed("cover"):
            covered = self.coverage(t, K_t, w2c, z_t)
            rej = [j for j, d in enumerate(decision) if d in ("reject", "grey")]
            acc = [j for j, d in enumerate(decision) if d in ("agree", "small", "deformable")]
            bad = torch.stack([pmask[j] for j in rej]).any(0) if rej else torch.zeros(HW, dtype=torch.bool, device=dev)
            good = pvis[acc].any(0) if acc else torch.zeros(HW, dtype=torch.bool, device=dev)
            nv = max(float(valid.sum()), 1.)
            # new content: pixels no segmented frame saw at this depth, that no accepted prediction explains
            share = float((open_mask(valid & ~covered & ~good) & valid).sum()) / nv
            # disagreement: pixels of rejected / grey predictions that no accepted prediction explains
            share_bad = float((open_mask(valid & bad & ~good) & valid).sum()) / nv
            log.update(novel=round(share, 4), bad=round(share_bad, 4))
        forced = i == 0 or (cfg["N"] and i % cfg["N"] == 0)
        trig = cfg.get("theta") is not None and (share >= cfg["theta"] or share_bad >= cfg.get("theta_bad", THETA_BAD))
        do_sam = forced or trig
        log.update(share=round(share, 4), sam=bool(do_sam), forced=bool(forced), pred=len(pids),
                   agree=decision.count("agree") + decision.count("small"), reject=decision.count("reject"), grey=len(grey))

        # truth (SAM 3 of this frame, all words): the label of each checked prediction (rows for the calibration; a
        # confusion count of decision x label for every configuration; the decider's answers against it)
        chk = [j for j in range(len(pids)) if orig[j] in ("agree", "reject", "grey")]
        if chk and (self.log_rows or self.cfg.get("truth", True)):
            t_truth = time.perf_counter()  # evaluation only: subtracted from the ladder's time
            sam_all = S.sam(t)
            iou = (iou_matrix(pvis[chk], sam_all["masks"]).amax(1) if len(sam_all["masks"]) else torch.zeros(len(chk), device=dev)).tolist()
            for q, j in enumerate(chk):
                lab = truth_label(iou[q])
                key = f"{orig[j]}->{decision[j]}|{'none' if lab is None else lab}"
                self.confusion[key] = self.confusion.get(key, 0) + 1
                if self.log_rows:
                    self.rows.append({"frame": t, "obj": pids[j], "x": feats[j], "iou": iou[q], "p": float(p[j]), "decision": decision[j], "word": self.name(pids[j])})
                if orig[j] == "grey" and cfg["grey"] == "qwen":
                    self.qwen_rows.append({"frame": t, "obj": pids[j], "p": round(float(p[j]), 3), "answer": decision[j], "iou": round(iou[q], 3)})
                    if len(self.qwen_refs) < 48:
                        self.qwen_refs.append((self.objs[pids[j]]["fresh_mask"], t, pack_bits(pvis[j:j + 1])))
            self.truth_s += time.perf_counter() - t_truth

        keep_masks, keep_ids, fresh = [], [], []
        agreed = [j for j, d in enumerate(decision) if d in ("agree", "small") and not self.deformable(pids[j])]
        if do_sam:
            self.counts["sam_frames"] += 1
            self.counts["sam_forced" if forced else "sam_triggered"] += 1
            sm = S.sam(t)
            with self.timed("sam_assoc"):
                taken = self.associate(si, t, sm, pids, pvis, decision, z_t, K_t, c_t, dyn, fresh)
            # where SAM 3 ran its masks are the output; a prediction it did not confirm is dropped here (the object
            # stays known: the older output and its voxels can bring it back)
            self.counts["dropped_unmatched_on_sam"] += sum(pids[j] not in taken for j in agreed)
            self.cover_src = (self.cover_src + [self.world_of(t, valid)])[-COVER_FRAMES:]
        else:
            sm = S.sam(t, always=True)
            with self.timed("sam_assoc"):
                self.associate(si, t, sm, pids, pvis, decision, z_t, K_t, c_t, dyn, fresh, deformable_only=True)
            self.counts["deformable_redetected"] += len(fresh)
            masks = pvis[agreed] if agreed else torch.zeros((0, *HW), dtype=torch.bool, device=dev)
            if cfg.get("box") and len(agreed) and S.tracker is not None:
                with self.timed("box"):
                    masks = self.tighten(t, masks, dyn)
            keep_masks += list(masks)
            keep_ids += [pids[j] for j in agreed]
        with self.timed("paint"):
            ms = [m for _, m in fresh] + keep_masks
            ids = [k for k, _ in fresh] + keep_ids
            stack = torch.stack(ms) & ~dyn[None] if ms else torch.zeros((0, *HW), dtype=torch.bool, device=dev)
            out = paint_ids(stack, ids) if ms else torch.zeros(HW, dtype=torch.int32, device=dev)
            self.out[t] = out
            self.prev = [(t, ids, stack)] + self.prev[:1]
        # occlusion episodes (hard case): >= half of the projection hidden by something in front, later seen again
        present = set(ids)
        pst = pstat.tolist()
        for j, oid in enumerate(pids):
            o = self.objs[oid]
            occ = pst[j][2] / max(pst[j][0], 1.)
            if occ >= .5 and o["occluded_since"] is None:
                o["occluded_since"] = t
                o["occ_episodes"] += 1
            elif o["occluded_since"] is not None and oid in present and occ < .2:
                o["occ_kept"] += 1
                o["occluded_since"] = None
        for oid in present:
            self.objs[oid]["last_seen"] = t
        log.update(out_objects=len(present), fresh=len(fresh))
        self.frames_log.append(log)

    # --- 1. prediction ---
    def predict(self, si, i, t, frames, z_t, K_t, w2c, dyn):
        """Every object of the two previous outputs, warped: each target pixel takes the source pixel the depth splat
        puts there (z-buffered, holes <= 3 px filled), so one warp moves every (overlapping) object mask at once; plus
        the objects missing from both, from their voxels. -> ids, visible masks (k,H,W), stats (k,4: projected, ok,
        occluded, conflict), memory flags (k,), projected masks without the occluded pixels (k,H,W)."""
        import torch
        S, dev = self.S, self.dev
        n = HW[0] * HW[1]
        none = torch.zeros((0, *HW), dtype=torch.bool, device=dev)
        empty = ([], none, torch.zeros((0, 4), device=dev), torch.zeros(0, dtype=torch.bool, device=dev), none)
        if not self.prev:
            return empty
        src = torch.full((n,), -1, dtype=torch.long, device=dev)
        pix = torch.full((n,), -1, dtype=torch.long, device=dev)
        zp = torch.full((n,), torch.nan, device=dev)
        for s_, (k, ids_k, _) in enumerate(self.prev):  # the newest (nearest) first
            zk, Kk, ck = S.geo(k)
            v, u = torch.nonzero(zk > 0, as_tuple=True)  # background too: the z-buffer, not the hole fill, decides edges
            lab, z = finish_with_z(splat(backproject(zk[v, u], Kk, ck, v, u), v * HW[1] + u + 1, K_t, w2c))
            lab, z = lab.flatten(), z.flatten()
            take = (src < 0) & (lab > 0)
            src[take], pix[take], zp[take] = s_, lab[take] - 1, z[take]
        ids = sorted(set().union(*[set(p_[1]) for p_ in self.prev]))
        pos = {o: j for j, o in enumerate(ids)}
        M = torch.zeros((len(ids), n), dtype=torch.bool, device=dev)
        for s_, (k, ids_k, masks_k) in enumerate(self.prev):
            where = torch.nonzero(src == s_).squeeze(1)
            if not len(ids_k) or not len(where):
                continue
            rows = torch.tensor([pos[o] for o in ids_k], device=dev)
            M[rows[:, None], where[None]] = masks_k.flatten(1)[:, pix[where]]
        M = M.view(-1, *HW)
        zp = zp.view(HW)
        keep = M.flatten(1).any(1)
        ids = [o for o, k_ in zip(ids, keep.tolist()) if k_]
        M = M[keep]
        # memory: known objects of this shot missing from both outputs, from their 5 cm voxels
        inside = set(ids)
        missing = [o for o in self.obj_codes if o not in inside]
        if missing:  # frustum cull on the last fresh centroid first (the voxels of the whole shot are many)
            cen = torch.tensor([self.objs[o]["cents"][-1][1] for o in missing], device=dev)
            cc = cen @ w2c[:3, :3].T + w2c[:3, 3]
            xx = cc[:, 0] / cc[:, 2].clamp(min=1e-3) * K_t[0, 0] + K_t[0, 2]
            yy = cc[:, 1] / cc[:, 2].clamp(min=1e-3) * K_t[1, 1] + K_t[1, 2]
            ok_c = (cc[:, 2] > .1) & (xx > -.25 * HW[1]) & (xx < 1.25 * HW[1]) & (yy > -.25 * HW[0]) & (yy < 1.25 * HW[0])
            missing = [o for o, k_ in zip(missing, ok_c.tolist()) if k_]
        Mm, zm, mem_ids = None, None, []
        if missing:
            codes_m = [self.obj_codes[o] for o in missing]
            world = code_centres(torch.cat(codes_m))
            own = torch.repeat_interleave(torch.tensor(missing, device=dev), torch.tensor([len(c) for c in codes_m], device=dev))
            cam = world @ w2c[:3, :3].T + w2c[:3, 3]
            zc = cam[:, 2]
            x = (cam[:, 0] / zc.clamp(min=1e-3) * K_t[0, 0] + K_t[0, 2]).round().long()
            y = (cam[:, 1] / zc.clamp(min=1e-3) * K_t[1, 1] + K_t[1, 2]).round().long()
            ins = (zc > .1) & (x >= 0) & (x < HW[1]) & (y >= 0) & (y < HW[0])
            zo = torch.where(ins, z_t[y.clamp(0, HW[0] - 1), x.clamp(0, HW[1] - 1)], torch.zeros_like(zc))
            vis = ins & (zo > 0) & ((zc - zo).abs() <= TOL_REL * zo + TOL_ABS + LIFT_VOXEL)
            cnt = torch.bincount(own[vis], minlength=self.next_id).tolist()
            mem_ids = [o for o in missing if cnt[o] >= MEM_MIN_VOX]
            if mem_ids:
                kv = vis & torch.isin(own, torch.tensor(mem_ids, device=dev))
                mlab, zm = finish_with_z(splat(world[kv], own[kv] + 1, K_t, w2c))
                Mm = mlab[None] == (torch.tensor(mem_ids, device=dev)[:, None, None] + 1)
                keepm = Mm.flatten(1).sum(1) >= MIN_PX
                mem_ids = [o for o, k_ in zip(mem_ids, keepm.tolist()) if k_]
                Mm = Mm[keepm]
        tol = TOL_REL * z_t + TOL_ABS

        def test(masks, z):
            hole = torch.isnan(z)
            ok = hole | (z_t <= 0) | ((z - z_t).abs() <= tol)
            occ = (~hole & (z_t > 0) & (z > z_t + tol)) | dyn  # something (a person) in front hides the object
            conf = ~hole & (z_t > 0) & (z < z_t - tol)  # the surface seen is farther: the object is not there
            vis = masks & ok[None] & ~dyn[None]
            stat = torch.stack([masks.flatten(1).sum(1), vis.flatten(1).sum(1), (masks & occ[None]).flatten(1).sum(1),
                                (masks & conf[None]).flatten(1).sum(1)], 1).float()
            return vis, stat, masks & ~occ[None]
        vis, stat, proj = test(M, zp)
        mem = torch.zeros(len(ids), dtype=torch.bool, device=dev)
        if mem_ids:
            v2, s2, p2 = test(Mm, zm)
            ids, vis, stat, proj = ids + mem_ids, torch.cat([vis, v2]), torch.cat([stat, s2]), torch.cat([proj, p2])
            mem = torch.cat([mem, torch.ones(len(mem_ids), dtype=torch.bool, device=dev)])
        if not ids:
            return empty
        return ids, vis, stat, mem, proj

    # --- 2. check ---
    def check(self, t, ids, vis, stat, mem):
        import torch
        S, dev = self.S, self.dev
        n = len(ids)
        if not n:
            return [], np.zeros(0), None
        area = vis.flatten(1).sum(1)
        emb = pool(S.feats(t), vis[:, ::2, ::2])
        idt = torch.tensor(ids, device=dev)
        cos = (emb * self.E[idt]).sum(1)
        shot_ids = [o for o, d in self.objs.items() if d["shot"] == self.objs[ids[0]]["shot"] and d["fresh"] > 0]
        if len(shot_ids) > 1:
            sims = emb @ self.E[torch.tensor(shot_ids, device=dev)].T
            same = torch.tensor(shot_ids, device=dev)[None] == idt[:, None]
            other = torch.where(same, torch.full_like(sims, -1.), sims).amax(1)
        else:
            other = torch.full_like(cos, -1.)
        er = edge_ratio(S.grad(t)[::2, ::2], vis[:, ::2, ::2])
        ref = torch.tensor([self.objs[o]["edge"] for o in ids], device=dev)
        age = torch.tensor([min(30, self.steps_since(o, t)) for o in ids], device=dev, dtype=torch.float32)
        X = torch.stack([cos, stat[:, 1] / (stat[:, 1] + stat[:, 3]).clamp(min=1), stat[:, 1] / stat[:, 0].clamp(min=1),
                         (er / ref.clamp(min=1e-3)).clamp(0, 3), torch.log(area.float().clamp(min=1)), age / 30, cos - other, mem.float()], 1).cpu().numpy()
        if self.cfg.get("check", "calib") == "off" or self.calib is None:
            p = np.ones(n)
        else:
            p = predict_logistic(self.calib, X)
        dec = []
        area, stat = area.tolist(), stat.tolist()
        for j in range(n):
            o = ids[j]
            if self.deformable(o):
                dec.append("deformable")
            elif area[j] < MIN_CHECK and stat[j][3] >= .5 * stat[j][0] and stat[j][0] >= MIN_CHECK:
                dec.append("reject")  # the surface seen there is farther than the object: it is gone (moved)
            elif area[j] < MIN_PX:
                dec.append("hidden")  # (almost) all of it behind something or out of view: not output, not rejected
            elif area[j] < MIN_CHECK:
                dec.append("small")
                self.counts["small_carried"] += 1
            else:
                self.counts["checked"] += 1
                dec.append("agree" if p[j] >= self.cfg.get("p_hi", .7) else "reject" if p[j] < self.cfg.get("p_lo", .3) else "grey")
        return dec, p, X.tolist()

    def steps_since(self, o, t):
        lf = self.objs[o]["last_fresh"]
        return 30 if lf is None else sum(1 for f in self.S.b2_index_range(lf, t))

    # --- coverage of segmented frames ---
    def world_of(self, t, valid):
        import torch
        z, K, c = self.S.geo(t)
        v, u = torch.nonzero(valid[::2, ::2], as_tuple=True)
        v, u = v * 2, u * 2
        return backproject(z[v, u], K, c, v, u)

    def coverage(self, t, K_t, w2c, z_t):
        """pixels a segmented frame already saw, at the same depth (something new in front of it is not 'seen')."""
        import torch
        key = None
        for w in self.cover_src:
            s = splat(w, torch.ones(len(w), dtype=torch.long, device=self.dev), K_t, w2c)
            key = s if key is None else torch.minimum(key, s)
        if key is None:
            return torch.zeros(HW, dtype=torch.bool, device=self.dev)
        lab, zc = finish_with_z(key)
        same = torch.isnan(zc) | (z_t <= 0) | ((zc - z_t).abs() <= TOL_REL * z_t + TOL_ABS)
        return (lab > 0) & same

    # --- 4/6. SAM 3 masks of this frame -> known objects or new ones ---
    def associate(self, si, t, sm, pids, pvis, decision, z_t, K_t, c_t, dyn, fresh, deformable_only=False):
        """SAM 3's masks of frame t -> objects: (1) the lift's overlap rule in 2D against the predictions (several masks
        may join one object, as the lift joins parts and wholes), (2) 5 cm voxel overlap with the known objects of the
        shot, (3) appearance re-identification of an object that was rejected where it had been (moved), (4) new."""
        import torch
        dev = self.dev
        masks = sm["masks"]
        if not len(masks):
            return set()
        area = masks.flatten(1).sum(1)
        on_person = (masks & dyn[None]).flatten(1).sum(1)
        masks = masks & ~dyn[None]
        keep = (on_person < .5 * area.clamp(min=1)) & (masks.flatten(1).sum(1) >= MIN_PX)
        idx = torch.nonzero(keep).squeeze(1)
        if not len(idx):
            return set()
        masks = masks[idx]
        midx = [int(sm["idx"][j]) for j in idx.tolist()]
        m = len(midx)
        assign = [None] * m
        cand = list(range(len(pids))) if not deformable_only else [j for j in range(len(pids)) if self.deformable(pids[j]) or decision[j] in ("reject", "grey")]
        if cand:
            A = masks[:, ::2, ::2].flatten(1).float()
            B = pvis[cand][:, ::2, ::2].flatten(1).float()
            inter = A @ B.T
            sa, sb = A.sum(1)[:, None], B.sum(1)[None]
            iou = inter / (sa + sb - inter).clamp(min=1)
            ok = (inter >= MATCH_MIN * torch.minimum(sa, sb)) & (inter >= MATCH_MAX * torch.maximum(sa, sb))
            best = torch.where(ok, iou, torch.full_like(iou, -1.)).max(1)
            for a, (v, b) in enumerate(zip(best.values.tolist(), best.indices.tolist())):
                if v >= 0:
                    assign[a] = pids[cand[b]]
                    self.counts["matched_2d"] += 1
        a_, v, u = torch.nonzero(masks[:, ::2, ::2] & (z_t > 0)[::2, ::2][None], as_tuple=True)
        w_all = backproject(z_t[v * 2, u * 2], K_t, c_t, v * 2, u * 2)
        codes = vox_codes(w_all)
        un = [a for a in range(m) if assign[a] is None]
        if un and len(self.vox_code):
            sel = torch.isin(a_, torch.tensor(un, device=dev))
            pc = torch.unique(torch.stack([a_[sel], codes[sel]], 1), dim=0)
            n_codes = torch.bincount(pc[:, 0], minlength=m)
            lo = torch.searchsorted(self.vox_code, pc[:, 1])
            hi = torch.searchsorted(self.vox_code, pc[:, 1], right=True)
            rep = torch.repeat_interleave(torch.arange(len(pc), device=dev), hi - lo)
            if len(rep):
                start = torch.cumsum(hi - lo, 0) - (hi - lo)
                pos_ = lo[rep] + torch.arange(len(rep), device=dev) - start[rep]
                cnt = torch.bincount(pc[rep, 0] * self.next_id + self.vox_owner[pos_], minlength=m * self.next_id).view(m, self.next_id)
                bc, bo = cnt.max(1)
                good = (bc >= VOX_MATCH * n_codes) & (bc > 0)
                for a, g, o in zip(range(m), good.tolist(), bo.tolist()):
                    if g and assign[a] is None:
                        assign[a] = o
                        self.counts["matched_3d"] += 1
        taken = {a for a in assign if a is not None}
        un = [a for a in range(m) if assign[a] is None]
        if un:  # appearance re-identification of a recently lost object: moved
            emb_m = pool(self.S.feats(t), masks[un][:, ::2, ::2])
            lost_all = [o for o, d in self.objs.items() if d["shot"] == si and o not in taken and d["rejected_at"] is not None
                        and self.S.b2_steps(d["rejected_at"], t) <= REID_WINDOW and d["reject_reason"] is not None
                        and (d["reject_reason"][1] < .5 or d["reject_reason"][0] < .5)]
            for q, a in enumerate(un):
                word = self.S.word_of(midx[a])
                lost = [o for o in lost_all if o not in taken and self.name(o) == word]
                if not lost:
                    continue
                cs = (self.E[torch.tensor(lost, device=dev)] @ emb_m[q]).tolist()
                b = int(np.argmax(cs))
                if cs[b] >= REID_COS:
                    o = lost[b]
                    self.events.append({"kind": "moved", "frame": t, "obj": o, "cos": round(cs[b], 3), "rejected_at": self.objs[o]["rejected_at"],
                                        "from_m": self.objs[o]["cents"][-1][1] if self.objs[o]["cents"] else None, "mask": midx[a]})
                    self.event_refs.append((self.objs[o]["fresh_mask"], t, midx[a]))
                    assign[a] = o
                    taken.add(o)
                    self.counts["reid_moved"] += 1
        with self.timed("update"):
            for a in range(m):
                if assign[a] is None:
                    assign[a] = self.new_obj(si)
            owners = sorted(set(assign))
            opos = {o: j for j, o in enumerate(owners)}
            own = torch.tensor([opos[o] for o in assign], device=dev)
            um = torch.zeros((len(owners), *HW), dtype=torch.int32, device=dev).index_add_(0, own, masks.int()) > 0
            emb = pool(self.S.feats(t), um[:, ::2, ::2])
            er = edge_ratio(self.S.grad(t)[::2, ::2], um[:, ::2, ::2]).tolist()
            pown = own[a_]
            npts = torch.bincount(pown, minlength=len(owners)).float()
            cent = (torch.zeros((len(owners), 3), device=dev).index_add_(0, pown, w_all) / npts.clamp(min=1)[:, None]).tolist()
            npts = npts.tolist()
            oid = torch.tensor(owners, device=dev)
            self.E[oid] = emb
            store = pack_bits(um)
            for j, o in enumerate(owners):
                d = self.objs[o]
                d["fresh"] += 1
                d["last_fresh"] = t
                d["rejected_at"] = None
                d["fresh_mask"] = (t, store, j)
                d["edge"] = er[j]
                if npts[j]:
                    d["cents"].append((t, cent[j]))
                fresh.append((o, um[j]))
                self.counts["fresh"] += 1
            for a in range(m):
                d = self.objs[assign[a]]
                for w, s_ in self.S.votes(midx[a]):
                    d["votes"][w] = d["votes"].get(w, 0.) + s_
            key = torch.unique(torch.cat([torch.stack([self.vox_code, self.vox_owner], 1), torch.stack([codes, oid[pown]], 1)]), dim=0)
            self.vox_code, self.vox_owner = key[:, 0].contiguous(), key[:, 1].contiguous()
            order = torch.argsort(pown, stable=True)
            for j, c in enumerate(torch.split(codes[order], torch.bincount(pown, minlength=len(owners)).tolist())):
                if not len(c):
                    continue
                o = owners[j]
                u = torch.unique(torch.cat([self.obj_codes[o], c]) if o in self.obj_codes else c)
                # ponytail: at most MEM_CAP voxels per object for the memory projection (a coarse mask is all it gives)
                self.obj_codes[o] = u[::int(math.ceil(len(u) / MEM_CAP))] if len(u) > MEM_CAP else u
        return taken

    # --- 3. one box prompt per reused object ---
    def tighten(self, t, masks, dyn):
        import torch
        big = masks.flatten(1).sum(1) >= MIN_BOX
        if not big.any():
            return masks
        sel = torch.nonzero(big).squeeze(1)
        m = masks[sel]
        ys = torch.arange(HW[0], device=self.dev)
        xs = torch.arange(HW[1], device=self.dev)
        rows, cols = m.any(2), m.any(1)
        y0 = torch.where(rows, ys, HW[0]).amin(1).float()
        y1 = torch.where(rows, ys, -1).amax(1).float() + 1
        x0 = torch.where(cols, xs, HW[1]).amin(1).float()
        x1_ = torch.where(cols, xs, -1).amax(1).float() + 1
        padx, pady = (x1_ - x0) * .05 + 1, (y1 - y0) * .05 + 1
        boxes = torch.stack([x0 - padx, y0 - pady, x1_ + padx, y1 + pady], 1)
        tm, score = self.S.tracker(t, boxes)
        self.counts["box_prompts"] += len(sel)
        tm = tm & ~dyn[None]
        inter = (tm & m).flatten(1).sum(1).float()
        iou = inter / (tm | m).flatten(1).sum(1).float().clamp(min=1)
        ok = (iou >= .5) & (score >= .5)
        out = masks.clone()
        out[sel[ok]] = tm[ok]
        self.counts["tightened"] += int(ok.sum())
        return out

    def name(self, o):
        v = self.objs[o]["votes"]
        return max(v, key=v.get) if v else None

    # --- results ---
    def objects(self):
        """confirmed objects (>= 2 SAM views, the lift's rule) with centroid (metres, estimated), word, counts."""
        out = []
        for o, d in self.objs.items():
            out.append({"id": o, "shot": d["shot"], "word": self.name(o), "fresh": d["fresh"], "passed": d["passed"],
                        "centroid_m": np.mean([c for _, c in d["cents"]], 0).round(3).tolist() if d["cents"] else None,
                        "cents": [(f, np.round(c, 3).tolist()) for f, c in d["cents"]], "occ_episodes": d["occ_episodes"], "occ_kept": d["occ_kept"]})
        return out


# ---------- synthetic self-check (torch, CPU is enough) ----------

class Synthetic:
    """A wall at 4 m, two IDENTICAL boxes (A, B) and a cabinet (C) at 3 m, the camera sliding sideways; box A moves at
    frame MOVE; a person passes in front of C over OCC. SAM 3 = the rendered masks; DINOv2 = one-hot appearance."""
    MOVE, OCC = 12, range(15, 21)

    def __init__(self, n=30):
        import torch
        self.dev = torch.device("cpu")
        self.words = ["box", "cabinet", "cable"]
        self.always = {"cable"}
        self.shots = {0: list(range(n))}
        self.K = torch.tensor([[300., 0, 252], [0, 300, 140], [0, 0, 1]])
        self.tracker = None
        self.cache = {}
        self.sam_calls = 0

    def objects_at(self, f):
        a = (-1.0, -.6) if f < self.MOVE else (-.45, -.05)
        return [("A", a, (-.3, .3), 0), ("B", (.2, .6), (-.3, .3), 0), ("C", (1.0, 1.3), (-.5, .2), 1)]

    def render(self, f):
        import torch
        if f in self.cache:
            return self.cache[f]
        H, W = HW
        cx = .02 * f
        c2w = torch.eye(4)
        c2w[0, 3] = cx
        v, u = torch.meshgrid(torch.arange(H).float(), torch.arange(W).float(), indexing="ij")
        rx, ry = (u - 252) / 300, (v - 140) / 300
        z = torch.full((H, W), 4.)
        app = torch.zeros((H, W), dtype=torch.long)  # 0 wall
        masks = {}
        for name, (x0, x1), (y0, y1), a in self.objects_at(f):
            X, Y = rx * 3 + cx, ry * 3
            m = (X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1)
            z[m] = 3.
            app[m] = 1 + a
            masks[name] = m
        person = torch.zeros((H, W), dtype=torch.bool)
        if f in self.OCC:  # in front of C (rays 0.15..0.4 wide cover C's 0.2..0.33 at these camera positions)
            person = (rx >= .15) & (rx <= .4) & (ry >= -.3) & (ry <= .4)
            z[person] = 1.5
            app[person] = 3
        vis = {k: m & ~person for k, m in masks.items()}
        self.cache[f] = (z, c2w, app, vis, person)
        return self.cache[f]

    def geo(self, f):
        z, c2w, *_ = self.render(f)
        return z, self.K, c2w

    def dyn(self, f):
        return self.render(f)[4]

    def sam(self, f, always=False):
        import torch
        _, _, _, vis, _ = self.render(f)
        if always:
            return {"masks": torch.zeros((0, *HW), dtype=torch.bool), "idx": []}
        self.sam_calls += 1
        names = [k for k in "ABC" if vis[k].sum() >= MIN_PX]
        self.last_names = names
        return {"masks": torch.stack([vis[k] for k in names]) if names else torch.zeros((0, *HW), dtype=torch.bool),
                "idx": [f * 10 + "ABC".index(k) for k in names]}

    def word_of(self, idx):
        return "cabinet" if idx % 10 == 2 else "box"

    def votes(self, idx):
        return [(self.word_of(idx), 1.)]

    def feats(self, f):
        import torch
        import torch.nn.functional as F
        app = self.render(f)[2]
        oh = F.one_hot(app, 768).float().permute(2, 0, 1)[None]
        return F.adaptive_avg_pool2d(oh, (36, 64))[0].permute(1, 2, 0)

    def grad(self, f):
        import torch
        a = self.render(f)[2].float()
        g = torch.zeros_like(a)
        g[:, 1:] += (a[:, 1:] - a[:, :-1]).abs()
        g[1:] += (a[1:] - a[:-1]).abs()
        return g + .01

    def b2_steps(self, a, b):
        return abs(b - a)

    def b2_index_range(self, a, b):
        return range(a, b)


def ladder_self_check():
    """Identical boxes keep their own ids (geometry, not appearance); the moved box is rejected where it was and
    re-identified where it went (same id, no new object); the cabinet keeps its id through the occlusion; SAM runs
    far less often than every frame."""
    import torch
    S = Synthetic()
    calib = {"mu": [0.] * 8, "sd": [1.] * 8, "w": [8., 6., 0, 0, 0, 0, 0, 0], "b": -9.}  # p high iff cos and depth agreement are high
    L = Ladder(S, {"N": 0, "theta": .05, "box": False, "check": "calib", "grey": "sam", "p_hi": .7, "p_lo": .3}, calib, log_rows=True).run()
    ids = {}
    for f in S.shots[0]:
        _, _, _, vis, _ = S.render(f)
        for k in "ABC":
            if vis[k].sum() < MIN_PX:
                continue
            got = L.out[f][vis[k]]
            got = got[got > 0]
            if len(got):
                ids.setdefault(k, []).append(int(torch.mode(got).values) - 1)
    assert len(set(ids["A"])) == 1 and len(set(ids["B"])) == 1 and ids["A"][0] != ids["B"][0], ids
    assert len(set(ids["C"])) == 1, ids["C"]
    assert L.counts["new_objects"] == 3, L.counts
    assert L.counts["reid_moved"] == 1 and L.events[0]["frame"] >= S.MOVE and L.events[0]["obj"] == ids["A"][0], (L.counts, L.events)
    assert L.counts["sam_frames"] < len(S.shots[0]) / 3, L.counts
    hidden = [f for f in S.OCC if S.render(f)[3]["C"].sum() < MIN_PX]
    assert len(hidden) >= 3 and L.counts["memory_pred"] >= 1, (hidden, L.counts)  # C comes back from its voxels, same id
    after = [f for f in S.shots[0] if f > max(S.OCC)]
    assert all((L.out[f][S.render(f)[3]["C"]] == ids["C"][0] + 1).float().mean() > .8 for f in after)
    print("x9 ladder self-check ok:", {k: L.counts[k] for k in ("sam_frames", "new_objects", "reid_moved", "agree", "reject", "grey", "memory_pred", "matched_2d", "matched_3d")})
    return {"counts": L.counts, "log": L.frames_log, "rows": L.rows[:0]}
