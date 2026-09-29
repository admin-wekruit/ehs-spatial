"""X11 local bundle adjustment for the X4 refine spots (X4: 0/15 confirmed, multi-view disagreement 4.6 cm vs a 1.5 cm rule).

One resident container on ONE A100-80GB (vLLM Qwen3-VL-8B on the same card for X4's VLM check, as in X4). Per video:
  coarse     X4's coarse pass, unchanged (the fast core's DA3 + SAM 3 + 3 cm TSDF), never part of a spot's time
  spots      X4's find_spots and a vectorised one on the same detections: both timed, the picks must be identical
  per spot   frames A = X4's view selection; frames B = the same selector with A's frames (+-5) ruled out (repeatability)
             x methods: 'anchor' (X4's product: posed DA3 on wide crops, per-view scale to the coarse map),
             'anchor+ba' (+ local BA: ALIKED + LightGlue tracks, cameras / per-view depth scale+shift / points re-solved,
             the held-out crop's pose registered to the other views' tracks), 'anchor+ba+tri' (+ the depth bent onto the
             triangulated tracks near the spot), 'anchor+ba+da3' (DA3 posed again with the solved cameras, its scale/shift
             fitted to the tracks); X4's confirmation (x4_refine.refine_spot) unchanged for every one
  modal run modal_apps/x11_local_ba.py --out RUNS/fx-x11-local-ba-NNN [--clips me340-165,...] [--spots 8] [--methods ...]
  python modal_apps/x11_local_ba.py --self-check      # find_spots_fast == x4.find_spots on synthetic detections
  python modal_apps/x11_local_ba.py --summarise RUN_DIR [EXTRA.json]
Scale is 'estimated' (floor plane + an assumed 1.6 m camera height): every m / cm / mm here is an estimated value.
"""
import json
import sys
import time
import traceback
from pathlib import Path

import modal
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "scripts"), str(HERE.parent)]
import fast_report_app as fra  # noqa: E402
import local_ba  # noqa: E402
import x4_refine as x4  # noqa: E402

LG_REV = "eb42fee2d71449efb0aa5c10549752b5d75384d8"  # cvg/LightGlue HEAD on 2026-09-28
CLIPS = {"me340-165": "source-full.mp4", "samsclub-337": "source-full.mp4", "walmart-190": "source-full.mp4",
         "lightning-3585": "source-rgb.mp4"}  # the factory clip has no 1280x720 cut: its 640x480 playback file
METHODS = ("anchor", "anchor+ba", "anchor+ba+tri", "anchor+ba+da3", "anchor+ba+da3+tri", "anchor+ba+mvs")
CPU, MEM_GIB = 16, 64
FACE_M = .03  # tilt repeatability: points within 3 cm of frames A's plane
PRICE_S = fra.PRICE["A100-80GB"] + CPU * fra.PRICE["cpu_core"] + MEM_GIB * fra.PRICE["gib"]

app = modal.App("panoptes-x11-local-ba")
# fast_report_app's image, step for step (cached layers), + kornia and LightGlue before the local sources (Modal allows
# no build step after add_local_*)
image = (modal.Image.from_registry("nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04", add_python="3.11")
         .apt_install("git", "libgl1", "libglib2.0-0", "libgomp1")
         .pip_install("torch==2.14.0", "torchvision", "xformers", "transformers==5.17.0", "accelerate", "addict", "pillow", "scipy",
                      "open3d==0.19.0", "shapely", "pydantic", "opencv-python-headless", "sentencepiece",
                      f"git+https://github.com/ByteDance-Seed/Depth-Anything-3.git@{fra.DA3_CODE}")
         .run_commands("python -m venv /opt/vllm && PIP_EXTRA_INDEX_URL= /opt/vllm/bin/pip install -q vllm==0.11.0 transformers==4.57.1 pillow")
         .env({"HF_HUB_OFFLINE": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
         .pip_install("kornia==0.8.3", "threadpoolctl==3.6.0")
         .run_commands(f"pip install --no-deps git+https://github.com/cvg/LightGlue.git@{LG_REV}")
         .env({"TORCH_HOME": "/v/models/torch"})
         .add_local_python_source("detect_shot_cuts", "m3_exp_geometry", "sam3_app", "video_events", "fast_report", "ehs_spatial",
                                  "fast_report_app", "lingbot_room", "x4_refine", "local_ba"))


# ---------- spot finding, vectorised ----------

def fit_plane_fast(P, thr, rng, iters=300, up=None):
    """x4.fit_plane with the same rng draws, every hypothesis scored in one matrix product."""
    tri = np.stack([rng.choice(len(P), 3, replace=False) for _ in range(iters)])
    a, b, c = P[tri[:, 0]], P[tri[:, 1]], P[tri[:, 2]]
    n = np.cross(b - a, c - a)
    nn = np.linalg.norm(n, axis=1)
    ok = nn >= 1e-12
    n = n / np.where(ok, nn, 1)[:, None]
    if up is not None:
        ok &= np.abs(n @ up) <= .5
    cnt = np.where(ok, (np.abs(P @ n.T - (a * n).sum(1)) <= thr).sum(0), -1)
    best = None
    if ok.any():
        k = int(np.argmax(cnt))
        best = np.flatnonzero(np.abs((P - a[k]) @ n[k]) <= thr)
    nrm, cen = x4.ls_normal(P[best if best is not None else np.arange(len(P))])
    inl = np.flatnonzero(np.abs((P - cen) @ nrm) <= thr)
    nrm, cen = x4.ls_normal(P[inl])
    return nrm, cen, inl, float(np.sqrt((((P[inl] - cen) @ nrm) ** 2).mean()))

def plane_tilt_fast(P, up, thr, seed=0, boot=200, cap=4000, vertical=True):
    """x4.plane_tilt, same result (same rng draws; the RANSAC hypotheses and the bootstrap fits batched)."""
    if P is None or len(P) < 50:
        return None
    rng = np.random.default_rng(seed)
    n, c, inl, rms = fit_plane_fast(P, thr, rng, up=up if vertical else None)
    if len(inl) < 30:
        return None
    kind, deg = x4.tilt(n, up)
    pool = P[inl] if len(inl) <= cap else P[rng.choice(inl, cap, replace=False)]
    idx = np.stack([rng.integers(0, len(pool), len(pool)) for _ in range(boot)])
    Q = pool[idx]
    Q = Q - Q.mean(1, keepdims=True)
    nb = np.linalg.eigh(np.matmul(Q.transpose(0, 2, 1), Q))[1][:, :, 0]
    th = np.degrees(np.arccos(np.clip(np.abs(nb @ np.asarray(up)), 0, 1)))
    d = np.where(th > 45, 90 - th, th)
    kb = np.where(th > 45, "vertical", "horizontal")
    boots = np.where(kb == kind, d, 90 - d)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"kind": kind, "tilt_deg": round(deg, 3), "ci95_deg": [round(float(lo), 3), round(float(hi), 3)],
            "ci_width_deg": round(float(hi - lo), 3), "inliers": int(len(inl)), "points": int(len(P)), "rms_mm": round(rms * 1000, 2),
            "normal": np.round(n, 5).tolist(), "centre": np.round(c, 4).tolist()}


def find_spots_fast(c, max_spots):
    """x4.find_spots, same picks: leader clustering per (group, shot) against an array of leaders, negatives per keyframe
    as one matrix product, and the per-cluster work (ball counts, coarse tilt) only where a cluster can be picked
    (>= 2 keyframes; the tilt only past the view-spread gate). The shared rng is advanced for skipped capped clusters
    exactly as find_spots does, so the capped clusters that count see the same random subset. The coarse tilt (only a
    ranking trigger) is computed lazily: the rest is taken from a heap keyed by the best trigger count a cluster could
    still reach, a tilt computed only when a cluster reaches the top. -> spots, sub-step times."""
    from scipy.spatial import cKDTree
    t0 = time.perf_counter()
    rng = np.random.default_rng(0)
    clusters, lead = [], {}
    for d in sorted(c.dets, key=lambda d: -d["score"]):
        key = (x4.GROUP_OF[d["word"]], d["si"])
        L = lead.get(key)
        if L is not None:
            dist = np.sqrt(((L[0][:L[1]] - d["centroid"]) ** 2).sum(1))
            j = int(np.argmin(dist))
            if dist[j] <= x4.CLUSTER_M:
                L[2][j]["members"].append(d)
                continue
        cl = {"group": key[0], "si": d["si"], "members": [d], "centroid": d["centroid"]}
        clusters.append(cl)
        if L is None:
            L = lead[key] = [np.zeros((64, 3)), 0, []]
        if L[1] == len(L[0]):
            L[0] = np.concatenate([L[0], np.zeros_like(L[0])])
        L[0][L[1]] = d["centroid"]
        L[1] += 1
        L[2].append(cl)
    for i, cl in enumerate(clusters):
        cl["id"] = i
        for d in cl["members"]:
            d["cluster"] = i
    t1 = time.perf_counter()
    by_q = {}
    for d in c.dets:
        by_q.setdefault(d["q"], []).append(d)
    neg = []
    for ds in by_q.values():
        if len(ds) > 1 and all("emb" in d for d in ds):
            E = np.stack([d["emb"] for d in ds])
            cl_ = np.array([d["cluster"] for d in ds])
            iu = np.triu_indices(len(ds), 1)
            neg.append((E @ E.T)[iu][cl_[iu[0]] != cl_[iu[1]]])
    neg = np.concatenate(neg) if neg else np.zeros(0)
    c.identity_threshold = max(.9, float(np.percentile(neg, 99))) if len(neg) else .9
    c.identity_negatives = int(len(neg))
    t2 = time.perf_counter()
    out, n_tilt = [], 0
    for cl in clusters:
        g = c.G[cl["si"]]
        mem = cl["members"]
        keyframes = sorted({d["q"] for d in mem})
        pts = voxel_down_fast(np.concatenate([d["pts"] for d in mem]), .02)
        ext = np.percentile(pts, 95, 0) - np.percentile(pts, 5, 0)
        capped = False
        if ext.max() > x4.SPOT_CAP_M:
            if len(keyframes) < 2:  # never picked; only the rng must move as in find_spots
                if len(pts) >= 4000:
                    rng.choice(len(pts), 4000, replace=False)
                continue
            sub = pts if len(pts) < 4000 else pts[rng.choice(len(pts), 4000, replace=False)]
            cnt = cKDTree(pts).query_ball_point(sub, .2, return_length=True)
            pts = pts[np.linalg.norm(pts - sub[int(np.argmax(cnt))], axis=1) <= x4.SPOT_CAP_M / 2]
            capped = True
        if len(keyframes) < 2 or len(pts) < 20:
            continue
        lo, hi = np.percentile(pts, 3, 0) - .03, np.percentile(pts, 97, 0) + .03
        height = (pts - g["p0"]) @ g["up"]
        w = max(set(d["word"] for d in mem), key=lambda x: sum(d["word"] == x for d in mem))
        if cl["group"] == "floor_item" and (np.percentile(height, 95) > .4 or (hi - lo).max() > .8):
            continue
        conf = [d["conf"] for d in mem if d["conf"] is not None]
        ev = np.linalg.eigvalsh(np.cov((pts - pts.mean(0)).T))
        sp = {"id": f"s{len(out)}", "cluster": cl["id"], "group": cl["group"], "word": w, "si": cl["si"], "pts": pts, "box": [lo, hi],
              "centre": np.median(pts, 0), "extent_m": np.round(hi - lo, 3).tolist(), "capped_to_1_5_m": capped,
              "keyframes": keyframes, "n_keyframes": len(keyframes), "max_score": max(d["score"] for d in mem),
              "mean_conf": float(np.mean(conf)) if conf else None, "holes": float(np.mean([d["holes"] for d in mem])),
              "planarity": float(ev[0] / max(ev[1], 1e-12)), "members": mem,
              "emb": None if "emb" not in mem[0] else (lambda e: e / np.linalg.norm(e))(np.mean([d["emb"] for d in mem], 0))}
        sp["normal"] = np.linalg.eigh(np.cov((pts - pts.mean(0)).T))[1][:, 0] if sp["planarity"] < .15 else None
        dirs = np.stack([g["c2w_np"][d["l"]][:3, 3] for d in mem]) - sp["centre"]
        dirs /= np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-9)
        sp["view_spread_deg"] = round(float(np.degrees(np.arccos(np.clip((dirs @ dirs.T).min(), -1, 1)))), 2)
        sp["rank"] = sp["max_score"] * np.log2(1 + sp["n_keyframes"]) * min(1., sp["view_spread_deg"] / 30)
        if sp["view_spread_deg"] < x4.ANGLE_MIN:  # never picked either
            continue
        tr = [f"semantic: {cl['group']} ('{w}')"]
        if c.conf_p25 is not None and sp["mean_conf"] is not None and sp["mean_conf"] < c.conf_p25:
            tr.append(f"quality: mean DA3 confidence {sp['mean_conf']:.2f} < the video's 25th percentile {c.conf_p25:.2f}")
        if sp["holes"] > .15:
            tr.append(f"quality: {sp['holes']:.0%} of the mask's depth dropped as flying pixels (holes)")
        sp["triggers"], sp["_tilt_due"] = tr, cl["group"] in x4.SURFACE and g["plane_ok"]
        out.append(sp)
    t3 = time.perf_counter()

    def settle(sp):  # the coarse tilt and its rule trigger, once
        nonlocal n_tilt
        if sp.pop("_tilt_due", False):
            t = plane_tilt_fast(sp["pts"], c.G[sp["si"]]["up"], .03, boot=60)
            n_tilt += 1
            sp["coarse_tilt"] = t
            if t and (t["ci95_deg"][0] <= x4.TILT_RULE <= t["ci95_deg"][1] or abs(t["tilt_deg"] - x4.TILT_RULE) <= 2):
                sp["triggers"].append(f"rule-borderline: coarse tilt {t['tilt_deg']:.1f} deg, CI {t['ci95_deg']} vs an assumed {x4.TILT_RULE:g} deg rule")
        return -sum(t.startswith(("quality", "rule")) for t in sp["triggers"])
    picked = []
    for grp in x4.GROUPS:
        cand = sorted([s for s in out if s["group"] == grp], key=lambda s: -s["rank"])
        if cand:
            picked.append(cand[0])
    for s_ in picked:
        settle(s_)
    taken = {s["id"] for s in picked}
    import heapq
    heap = [(-(sum(t.startswith(("quality", "rule")) for t in s["triggers"]) + bool(s.get("_tilt_due"))), -s["rank"], k, not s.get("_tilt_due"))
            for k, s in enumerate(out) if s["id"] not in taken]
    heapq.heapify(heap)
    rest = []
    while heap and len(rest) < max_spots - len(picked):
        key, nr, k, exact = heapq.heappop(heap)
        if exact:
            rest.append(out[k])
        else:
            heapq.heappush(heap, (settle(out[k]), nr, k, True))
    picked = (picked + rest)[:max_spots]
    for i, s in enumerate(picked):
        s["id"] = f"s{i}"
    c.clusters, c.candidates = clusters, out
    return picked, {"cluster_s": round(t1 - t0, 4), "negatives_s": round(t2 - t1, 4), "per_cluster_s": round(t3 - t2, 4),
                    "pick_s": round(time.perf_counter() - t3, 4), "clusters": len(clusters), "pickable": len(out), "coarse_tilts": n_tilt}


def voxel_down_fast(P, s):
    """x4.voxel_down, same points in the same order: one int64 key per voxel instead of a row-wise unique."""
    if len(P) == 0:
        return P
    q = np.floor(P / s).astype(np.int64)
    q -= q.min(0)
    span = q.max(0) + 1
    _, i = np.unique((q[:, 0] * span[1] + q[:, 1]) * span[2] + q[:, 2], return_index=True)
    return P[np.sort(i)]


def spot_key(s):
    return [s["group"], s["word"], int(s["n_keyframes"]), np.round(s["centre"], 4).tolist()]


# ---------- the resident container ----------

@app.cls(image=image, gpu="A100-80GB", cpu=CPU, memory=MEM_GIB * 1024, volumes=fra.VOLUMES, timeout=3600, retries=0,
         max_containers=1, scaledown_window=30)
class LocalBA:
    @modal.enter()
    def boot(self):  # x4_refine.Refine.boot minus LingBot and the resolution sweep, plus the matcher
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor
        import cv2
        import torch
        from depth_anything_3.api import DepthAnything3
        from transformers import Sam3Model, Sam3Processor
        from fast_report import cascade, core, segment, vlm
        t0 = time.perf_counter()
        b = self.boot_record = {"entered_unix": time.time()}
        lap = lambda k: b.__setitem__(k, round(time.perf_counter() - t0, 2))  # noqa: E731
        vlm.VLLM_SHARE = .3
        self.vllm = vlm.start(0, mps=False)
        self.procs = ProcessPoolExecutor(12, mp_context=multiprocessing.get_context("spawn"))
        list(self.procs.map(core.warm_worker, range(12)))
        self.dev = torch.device("cuda:0")
        model = DepthAnything3.from_pretrained(fra.DA3_MODEL, revision=fra.DA3_REV, cache_dir="/v/da3/huggingface/hub").eval().to(self.dev)
        self.api, self.da3 = model, core.Da3(model, self.dev)
        proc = Sam3Processor.from_pretrained(fra.sam3_app.MODEL_ID, revision=fra.sam3_app.REVISION, cache_dir="/v/sam3/huggingface/hub")
        sam = Sam3Model.from_pretrained(fra.sam3_app.MODEL_ID, revision=fra.sam3_app.REVISION, cache_dir="/v/sam3/huggingface/hub",
                                        torch_dtype=torch.bfloat16).eval().to(self.dev)
        self.sam = segment.Sam3(sam, proc, self.dev)
        self.emb = cascade.Embedder(self.dev, "/v/models/hf")
        lap("models_s")
        self.matcher = local_ba.Matcher(self.dev)  # ALIKED + LightGlue weights: torch.hub into /v/models/torch
        fra.VOLUMES["/v/models"].commit()
        lap("matcher_s")
        vlm.wait(self.vllm)
        lap("vllm_ready_s")
        try:  # a failing warm-up is recorded, never a crash loop
            rng = np.random.default_rng(0)
            with torch.inference_mode():
                x4.da3_shot(self.da3, torch.randint(0, 255, (16, 720, 1280, 3), dtype=torch.uint8, device=self.dev))
                r = x4.DEFAULT_RES
                for n in (3, 4, 5):
                    w2c = np.repeat(np.eye(4)[None], n, 0)
                    w2c[:, :3, 3] = rng.normal(0, .3, (n, 3))
                    K = np.repeat(np.array([[r, 0, r / 2], [0, r, r / 2], [0, 0, 1.]])[None], n, 0)
                    self.posed([rng.integers(0, 255, (r, r, 3), np.uint8) for _ in range(n)], w2c, K, r)
                v = self.sam.vision(torch.randint(0, 255, (8, 720, 1280, 3), dtype=torch.uint8, device=self.dev))
                self.sam.detect(v, 8, ("person", "floor"), segment.PERSON_SCORE, top=segment.PERSON_TOP)
                self.sam.detect(self.sam.pick(v, [0, 3, 6]), 3, tuple(x4.WORDS), segment.VOCAB_SCORE)
                v = self.sam.vision(torch.randint(0, 255, (5, r, r, 3), dtype=torch.uint8, device=self.dev))
                self.sam.detect(v, 5, ("cable",), .25, logits=True)
                f = self.matcher.features([rng.integers(0, 255, (r, r, 3), np.uint8) for _ in range(2)])
                self.matcher.match(f[0], f[1])
                torch.cuda.synchronize()
            b["mvs_self_check"] = local_ba.mvs_self_check(self.dev)  # also warms the sweep
            jpg = cv2.imencode(".jpg", rng.integers(0, 255, (504, 504, 3), np.uint8))[1].tobytes()
            vlm.chat([vlm.image_block(jpg), {"type": "text", "text": "Describe."}], max_tokens=8)
        except Exception:  # noqa: BLE001
            b["warm_error"] = traceback.format_exc()[-1500:]
        torch.cuda.empty_cache()
        lap("ready_s")
        b["gpus"] = fra.gpu_listing()
        b["resident_gib"] = round((torch.cuda.mem_get_info(0)[1] - torch.cuda.mem_get_info(0)[0]) / 2 ** 30, 2)

    @modal.exit()
    def stop(self):
        if getattr(self, "vllm", None) is not None:
            self.vllm.terminate()

    @modal.method()
    def boot_info(self):
        return self.boot_record

    def posed(self, crops_rgb, w2c, K, R, own=False):  # x4_refine.Refine.posed
        import torch
        with torch.inference_mode():
            out = self.api.inference(list(crops_rgb), extrinsics=np.asarray(w2c, np.float32), intrinsics=np.asarray(K, np.float32),
                                     align_to_input_ext_scale=not own, process_res=R)
        assert out.depth.shape == (len(crops_rgb), R, R), out.depth.shape
        return np.asarray(out.depth, np.float32), np.asarray(out.conf, np.float32), np.asarray(out.intrinsics, np.float64), np.asarray(out.extrinsics, np.float64)

    @modal.method()
    def run(self, clip, mp4, opts):
        import torch
        from fast_report.instrument import Clock, Vram
        clock, vram = Clock(), Vram([0]).start()
        torch.cuda.reset_peak_memory_stats(0)
        out, error = {}, None
        try:
            out = analyse(self, clip, mp4, opts, clock)
        except Exception:  # noqa: BLE001  reported, never retried
            error = traceback.format_exc()[-4000:]
        vram.stop()
        rep = clock.report(vram, price_per_s=PRICE_S)
        rep["flags"] = [f for f in rep["flags"] if not f.startswith("unknown stage name")]
        torch.cuda.empty_cache()
        return {**out, "clip": clip, "error": error, "timing": rep, "boot": self.boot_record,
                "torch_reserved_peak_gib": round(torch.cuda.max_memory_reserved(0) / 2 ** 30, 2)}


def slim(r):
    """A refine_spot record without the bulky propagation boxes."""
    rp = (r.get("propagation") or {}).get("reproject")
    if rp:
        rp.pop("boxes_grid_504x280", None)
    for v in r.get("variants", []):
        for b in (v.get("ba"), (v.get("holdout") or {}).get("ba")):
            if b:
                b.pop("licences", None)
    return r


def analyse(m, clip, mp4, opts, clock):
    c = x4.coarse(m, mp4, clock)
    c.store, c.patched, c.store_raw = {}, {}, {}
    ns = opts.get("spots", 8)
    t = time.perf_counter()
    with clock.stage("refine.spots.x4"):
        ref = x4.find_spots(c, ns)
    t_ref = time.perf_counter() - t
    ref_keys, n_cand_ref = [spot_key(s) for s in ref], len(c.candidates)
    t = time.perf_counter()
    with clock.stage("refine.spots.fast"):
        try:  # one BLAS thread for the small matrices (as in the BA)
            from threadpoolctl import threadpool_limits
            with threadpool_limits(1):
                spots, sub = find_spots_fast(c, ns)
        except ImportError:
            spots, sub = find_spots_fast(c, ns)
    t_fast = time.perf_counter() - t
    same = [spot_key(s) for s in spots] == ref_keys
    if not same:  # X4's picks stay the product; the mismatch is reported
        spots = ref
    finder = {"x4_s": round(t_ref, 3), "fast_s": round(t_fast, 3), "same_picks": same, "fast_substeps": sub,
              "x4_candidates": n_cand_ref, "detections": len(c.dets), "picks": ref_keys}
    methods = opts.get("methods", METHODS)
    recs, rows = [], []
    for sp in ([] if opts.get("finder_only") else spots):
        head = {k: sp[k] for k in ("id", "group", "word", "si", "extent_m", "capped_to_1_5_m", "n_keyframes", "view_spread_deg", "max_score",
                                   "mean_conf", "holes", "planarity", "triggers")}
        head.update(centre_m=np.round(sp["centre"], 3).tolist(), box_m=[np.round(b, 3).tolist() for b in sp["box"]], coarse_tilt=sp.get("coarse_tilt"))
        t = time.perf_counter()
        sel = {"A": x4.select_views(c, sp)}
        ts = {"A": round(time.perf_counter() - t, 3)}
        t = time.perf_counter()
        sel["B"] = x4.select_views(c, sp, exclude=[v["frame"] for v in sel["A"]["views"]])
        ts["B"] = round(time.perf_counter() - t, 3)
        runs, tiles = {}, {}
        for fs in ("A", "B") if opts.get("repeat", True) else ("A",):
            for mi, meth in enumerate(methods):
                if fs == "B" and meth not in opts.get("methods_b", methods):
                    continue
                s2 = dict(sp, select=sel[fs], select_s=ts[fs], id=f"{sp['id']}{fs}{mi}")  # unique stage names per run
                c.patched = {}
                try:
                    r = x4.refine_spot(m, c, s2, clock, None, {"method": meth, "sweep": False})
                except Exception:  # noqa: BLE001
                    r = {"error": traceback.format_exc()[-3000:]}
                tl = r.pop("_tiles", None)
                if fs == "A":
                    tiles[meth] = tl
                runs[f"{fs}:{meth}"] = slim(r)
        g = c.G[sp["si"]]
        for mi, meth in enumerate(methods):  # the same face in A and B: frames A's fine plane (+-3 cm), refitted in each map
            ra, rb = runs.get(f"A:{meth}"), runs.get(f"B:{meth}")
            pa = ((((ra or {}).get("variants") or [{}])[0].get("plane")) or {}).get("fine")
            if not pa or rb is None:
                continue
            nrm, ctr = np.asarray(pa["normal"]), np.asarray(pa["centre"])
            fm = {}
            for fs, P in (("A", c.store.get(f"{sp['id']}A{mi}-fine")), ("B", c.store.get(f"{sp['id']}B{mi}-fine")), ("coarse", g["pts"])):
                if P is None or not len(P):
                    continue
                P = P[x4.in_box(P, sp["box"])]
                t = x4.plane_tilt(P[np.abs((P - ctr) @ nrm) <= FACE_M], g["up"], .03 if fs == "coarse" else .01)
                fm[fs] = None if t is None else {k: t[k] for k in ("tilt_deg", "ci95_deg", "inliers")}
            rb["face_matched_tilt"] = fm
        recs.append({**head, "frames_A": [v["frame"] for v in sel["A"]["views"]], "frames_B": [v["frame"] for v in sel["B"]["views"]], "runs": runs})
        rows.append((sp, runs, tiles))
    return {"coarse": {**c.cuts, "trigger_detections": len(c.dets), "clusters": len(c.clusters)}, "spot_finding": finder,
            "spots": recs, "jpg": sheet(rows, methods)}


def sheet(rows, methods):
    """Per spot: frames A best crop + the BA outline | coarse 3 cm relief | fine 1 cm relief per method (the spot box)."""
    import cv2
    T = 224
    out = []
    for sp, runs, tiles in rows:
        base = tiles.get(methods[-1]) or tiles.get(methods[0])
        if not base:
            continue
        ims = [base[0], base[1]] + [(tiles.get(mt) or (None,) * 3)[2] for mt in methods]
        labels = ["best view + outline (" + methods[-1] + ")", "coarse 3 cm"] + [f"fine 1 cm {mt}" for mt in methods]
        row = []
        for im, lab in zip(ims, labels):
            im = np.zeros((T, T, 3), np.uint8) if im is None else cv2.resize(im if im.ndim == 3 else cv2.cvtColor(im, cv2.COLOR_GRAY2BGR), (T, T))
            cv2.putText(im, lab, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, .38, (0, 255, 255), 1, cv2.LINE_AA)
            row.append(im)
        band = np.zeros((34, T * len(row), 3), np.uint8)
        verd = "  ".join(f"{mt}: {((runs.get('A:' + mt) or {}).get('verdict') or {}).get('status', '?')} "
                         f"mv {(((runs.get('A:' + mt) or {}).get('variants') or [{}])[0].get('multi_view') or {}).get('median_m')}" for mt in methods)
        cv2.putText(band, f"{sp['id']} {sp['group']} '{sp['word']}'", (4, 13), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(band, verd[:150], (4, 28), cv2.FONT_HERSHEY_SIMPLEX, .36, (255, 255, 255), 1, cv2.LINE_AA)
        out.append(np.concatenate([band, np.concatenate(row, 1)]))
    return cv2.imencode(".jpg", np.concatenate(out), [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes() if out else None


def plain(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o.item() if hasattr(o, "item") else str(o)


@app.local_entrypoint()
def main(out: str, clips: str = ",".join(CLIPS), spots: int = 8, methods: str = ",".join(METHODS), repeat: bool = True,
         methods_b: str = "anchor,anchor+ba,anchor+ba+mvs", finder_only: bool = False):
    import shutil
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    assert shutil.disk_usage(out).free > 8 * 2 ** 30, "under 8 GB free: stop"
    r = LocalBA()
    boot = r.boot_info.remote()
    (out / "boot.json").write_text(json.dumps(boot, indent=1, default=plain))
    print("ready:", json.dumps({k: v for k, v in boot.items() if k.endswith("_s") or k in ("resident_gib", "warm_error")}), flush=True)
    for clip in clips.split(","):
        mp4 = (x4.PHASE2 / "data/clips" / clip / CLIPS[clip]).read_bytes()
        t = time.time()
        res = r.run.remote(clip, mp4, {"spots": spots, "methods": tuple(methods.split(",")), "repeat": repeat, "methods_b": tuple(methods_b.split(",")),
                                        "finder_only": finder_only})
        res["client_wall_s"] = round(time.time() - t, 2)
        res["source_file"] = f"data/clips/{clip}/{CLIPS[clip]}"
        jpg = res.pop("jpg", None)
        if jpg:
            (out / f"{clip}-sheet.jpg").write_bytes(jpg)
        (out / f"{clip}.json").write_text(json.dumps(res, indent=1, default=plain))
        print(clip, "error" if res.get("error") else "ok", res.get("client_wall_s"), json.dumps(res.get("spot_finding", {}))[:400], flush=True)
        if res.get("error"):
            print(res["error"][-3000:], flush=True)
        for s in res.get("spots", []):
            line = [s["id"], s["group"], s["word"]]
            for k, v in s["runs"].items():
                v0 = (v.get("variants") or [{}])[0]
                ba = v0.get("ba") or {}
                line.append(f"{k}={(v.get('verdict') or {}).get('status')}/{(v0.get('multi_view') or {}).get('median_m')}"
                            f"/rp{((ba.get('reprojection_px_at_504_crop') or {}).get('after') or {}).get('median')}"
                            + (f" ERR {v['error'][-300:]}" if v.get("error") else ""))
            print("  ", " | ".join(str(x) for x in line), flush=True)


def self_check():
    """find_spots_fast picks what x4.find_spots picks on synthetic detections (incl. capped and single-keyframe clusters)."""
    rng = np.random.default_rng(5)

    c = type("C", (), {})()
    g = {"p0": np.zeros(3), "up": np.array([0., -1, 0]), "plane_ok": True, "c2w_np": np.repeat(np.eye(4)[None], 40, 0)}
    g["c2w_np"][:, :3, 3] = np.c_[np.linspace(-3, 3, 40), np.full(40, -1.5), np.zeros(40)]
    c.G, c.conf_p25, c.dets = [g], .5, []
    words = x4.WORDS
    for k in range(400):
        w = words[rng.integers(len(words))]
        ctr = np.r_[rng.uniform(-4, 4), rng.uniform(-1.5, .2), rng.uniform(3, 6)]
        big = rng.random() < .1
        n = 5000 if big else 60
        P = ctr + rng.normal(0, 1.2 if big else .05, (n, 3)) * ([1, .02, 1] if big else 1)
        q = int(rng.integers(0, 40))
        e = rng.normal(0, 1, 16)
        c.dets.append({"q": q, "l": q, "si": 0, "word": w, "score": float(rng.uniform(.3, .9)), "pts": P, "centroid": np.median(P, 0),
                       "conf": float(rng.uniform(0, 1)), "holes": float(rng.uniform(0, .3)), "emb": e / np.linalg.norm(e)})
    ref = x4.find_spots(c, 8)
    thr_ref = c.identity_threshold
    fast, sub = find_spots_fast(c, 8)
    assert [spot_key(s) for s in fast] == [spot_key(s) for s in ref] and len(ref) >= 5, ([spot_key(s) for s in ref], [spot_key(s) for s in fast])
    assert [s["triggers"] for s in fast] == [s["triggers"] for s in ref] and sub["coarse_tilts"] < 30, sub
    for k in range(10):
        P = rng.normal(0, 1, (int(rng.integers(1, 3000)), 3)) * rng.uniform(.01, 3)
        assert np.array_equal(voxel_down_fast(P, .02), x4.voxel_down(P, .02)), k
    up = np.array([0., -1, 0])
    for k in range(20):  # the batched plane tilt is the same function
        m = int(rng.integers(60, 2000))
        a = np.radians(rng.uniform(0, 20))
        yy, xx = rng.uniform(-1, 1, (2, m))
        P = np.stack([xx, yy, 3 + yy * np.tan(a)], 1) + rng.normal(0, rng.uniform(.002, .05), (m, 3))
        assert x4.plane_tilt(P, up, .03, boot=60) == plane_tilt_fast(P, up, .03, boot=60), k
    assert abs(c.identity_threshold - thr_ref) < 1e-9 and sub["pickable"] <= sub["clusters"]
    print("x11 self-check ok", {"picks": len(ref), **sub})


def _med(xs, q=4):
    xs = [x for x in xs if x is not None]
    return round(float(np.median(xs)), q) if xs else None


def spot_row(s, stages, methods):
    """One spot of a <clip>.json -> the per-spot table (frames A per method, BA numbers, A-vs-B tilt, times)."""
    import re
    out = {"id": s["id"], "group": s["group"], "word": s["word"], "centre_m_estimated": s["centre_m"], "frames_A": s["frames_A"],
           "frames_B": s["frames_B"], "methods": {}, "tilt_repeatability": {}}
    for key, r in s["runs"].items():
        fs, meth = key.split(":", 1)
        if "variants" not in r:
            out["methods"][key] = {"error": (r.get("error") or "")[-600:], "verdict": r.get("verdict")}
            continue
        v = r["variants"][0]
        ho = v.get("holdout") or {}
        pl = ((v.get("plane") or {}).get("fine")) or {}
        ba = v.get("ba") or {}
        hba = ho.get("ba") or {}
        sid = f"{s['id']}{fs}{methods.index(meth) if meth in methods else '?'}"
        peaks = [x["peak_gb"][0] for x in stages if re.search(rf"\.{sid}(\.|$)", x["stage"]) and x.get("peak_gb") and x["peak_gb"][0] is not None]
        met = [x["s"] for x in stages if x["stage"].startswith(f"refine.metrics.{sid}.")]
        row = {"verdict": r["verdict"]["status"], "failed": r["verdict"].get("failed"), "checks": r["verdict"].get("checks"),
               "multi_view_median_m": v["multi_view"]["median_m"], "multi_view_within_2cm": v["multi_view"]["within_2cm"],
               "multi_view_occl10_median_m": v["multi_view"]["occlusion_cut_10cm"]["median_m"],
               "fine_to_coarse_nn_median_m": v["agreement_with_coarse"]["fine_to_coarse_nn_median_m"],
               "holdout": {"photometric_ratio": (ho.get("photometric_l1") or {}).get("ratio_fine_over_coarse"),
                           "outline_iou_fine": (ho.get("outline_iou") or {}).get("fine"), "outline_iou_coarse": (ho.get("outline_iou") or {}).get("coarse"),
                           "resect": {k: x for k, x in (hba.get("resect") or {}).items() if k != "c2w"} or None, "skipped": ho.get("skipped")},
               "plane_fine": {k: pl.get(k) for k in ("tilt_deg", "ci95_deg", "ci_width_deg", "inliers")} if pl else None,
               "plane_coarse": {k: ((v.get("plane") or {}).get("coarse") or {}).get(k) for k in ("tilt_deg", "ci95_deg")} if v.get("plane") else None,
               "thin": v.get("thin"), "vlm": (r.get("vlm") or {}).get("answer"), "points_in_box": v["points_in_box"],
               "anchor": v.get("anchor"), "gpu0_peak_gib": max(peaks) if peaks else None,
               "time_s": {**v["time_s"], "metrics": round(sum(met), 3) if met else None, "vlm": r["time_s_product"]["vlm"],
                          "propagate": r["time_s_product"]["propagate"], "select": r["time_s_product"]["select"],
                          "total_without_sweeps": r["time_s_product"]["total_without_sweeps"]}}
        if ba:
            t, th = ba.get("time_s") or {}, hba.get("time_s") or {}
            row["ba"] = {k: ba.get(k) for k in ("status", "views", "keypoints", "pairs", "tracks", "observations", "tracks_in_region",
                                                "depth_observations", "track_length_hist", "reprojection_px_at_504_crop", "depth_residual_m",
                                                "pose_change", "depth_scale", "depth_shift_m", "baseline_scale", "solver", "tri_field",
                                                "da3_rerun", "mvs", "geometry", "tracks_before_cap")}
            for k in ("consistency_before", "consistency_after"):  # the per-pair lists stay in <clip>.json
                row["ba"][k] = {x: y for x, y in (ba.get(k) or {}).items() if x != "pairs"} or None
            row["time_split_s"] = {"matching": round(t.get("features", 0) + t.get("match_verify", 0), 3),
                                   "ba": round(t.get("tracks", 0) + t.get("solve", 0) + t.get("apply_depth", 0), 3),
                                   "fuse_tsdf_1cm": v["time_s"]["tsdf_1cm"],
                                   "checks": round(v["time_s"]["holdout_check"] + (row["time_s"]["metrics"] or 0) + r["time_s_product"]["vlm"], 3),
                                   "of_which_holdout_matching_ba_resect": round(sum(th.get(k, 0) for k in ("features", "match_verify", "tracks", "solve",
                                                                                                             "apply_depth", "resect")), 3),
                                   "da3_crops_sam3": round(v["time_s"]["fine_i_iii"] - v["time_s"]["anchor_icp"] - v["time_s"]["tsdf_1cm"], 3)}
        out["methods"][key] = row
    for meth in methods:
        fm = (s["runs"].get(f"B:{meth}") or {}).get("face_matched_tilt") or {}
        if (fm.get("A") or {}).get("tilt_deg") is not None and (fm.get("B") or {}).get("tilt_deg") is not None:
            out.setdefault("face_matched_tilt", {})[meth] = {**fm, "abs_diff_deg": round(abs(fm["A"]["tilt_deg"] - fm["B"]["tilt_deg"]), 3)}
        a, b = (out["methods"].get(f"{f}:{meth}") or {} for f in "AB")
        ta, tb = (a.get("plane_fine") or {}).get("tilt_deg"), (b.get("plane_fine") or {}).get("tilt_deg")
        if ta is not None and tb is not None:
            out["tilt_repeatability"][meth] = {"A": ta, "B": tb, "abs_diff_deg": round(abs(ta - tb), 3),
                                               "ci_A": (a.get("plane_fine") or {}).get("ci95_deg"), "ci_B": (b.get("plane_fine") or {}).get("ci95_deg"),
                                               "cis_overlap": bool(a["plane_fine"]["ci95_deg"][0] <= b["plane_fine"]["ci95_deg"][1]
                                                                   and b["plane_fine"]["ci95_deg"][0] <= a["plane_fine"]["ci95_deg"][1])}
    return out


def summarise(run_dir, extra=None, x4_run="fx-x4-refine-006"):
    """<clip>.json -> results.json: every reported number, per spot and aggregated."""
    run_dir = Path(run_dir)
    boot = json.loads((run_dir / "boot.json").read_text())
    x4dir = run_dir.parent / x4_run
    videos, allrows = {}, []
    for clip in CLIPS:
        p = run_dir / f"{clip}.json"
        if not p.exists():
            continue
        r = json.loads(p.read_text())
        t = r["timing"]
        stages = [{k: s.get(k) for k in ("stage", "start_s", "end_s", "s", "peak_gb", "over_90", "n")} for s in t["stages"]]
        methods = list(dict.fromkeys(k.split(":", 1)[1] for s in r.get("spots", []) for k in s["runs"]))  # the run's order (stage ids)
        x4_spots = []
        if (x4dir / f"{clip}.json").exists():
            x4_spots = json.loads((x4dir / f"{clip}.json").read_text()).get("spots", [])
        rows = []
        for s in r.get("spots", []):
            row = spot_row(s, stages, methods)
            near = min(x4_spots, key=lambda q: np.linalg.norm(np.subtract(q["centre_m"], s["centre_m"])), default=None)
            if near is not None:
                d = float(np.linalg.norm(np.subtract(near["centre_m"], s["centre_m"])))
                row["x4_run006"] = {"id": near["id"], "word": near["word"], "centre_shift_m": round(d, 4), "same_spot": d < .01 and near["id"] == s["id"],
                                    "verdict_006": (near.get("verdict") or {}).get("status"),
                                    "multi_view_median_m_006": ((near.get("variants") or [{}])[0].get("multi_view") or {}).get("median_m")}
            row["clip"] = clip
            rows.append(row)
        allrows += rows
        coarse_s = max((s["end_s"] for s in t["stages"] if not s["stage"].startswith("refine.")), default=None)
        videos[clip] = {"error": r.get("error"), "source_file": r.get("source_file"), "coarse": r.get("coarse"), "coarse_pass_s": coarse_s,
                        "analysis_wall_s": t["elapsed_s"], "client_wall_s": r.get("client_wall_s"), "spot_finding": r.get("spot_finding"),
                        "methods": methods, "spots": rows, "gpu_peak": t["gpu_peak"], "flags": t["flags"],
                        "torch_reserved_peak_gib": r.get("torch_reserved_peak_gib"), "sheet_jpg": f"{clip}-sheet.jpg"}

    def col(fs, meth, f):
        return [f(x["methods"][f"{fs}:{meth}"]) for x in allrows if "verdict" in x["methods"].get(f"{fs}:{meth}", {}) and "error" not in x["methods"][f"{fs}:{meth}"]]
    agg = {}
    for meth in METHODS:
        for fs in "AB":
            rows = col(fs, meth, lambda m: m)
            if not rows:
                continue
            fails = {}
            for m in rows:
                for k in m["failed"] or []:
                    fails[k] = fails.get(k, 0) + 1
            a = {"spots": len(rows), "confirmed": sum(m["verdict"] == "CONFIRMED" for m in rows), "failed_checks": fails,
                 "median_multi_view_m": _med([m["multi_view_median_m"] for m in rows]),
                 "median_multi_view_within_2cm": _med([m["multi_view_within_2cm"] for m in rows]),
                 "multi_view_le_1_5cm": sum((m["multi_view_median_m"] or 9) <= .015 for m in rows),
                 "median_fine_to_coarse_m": _med([m["fine_to_coarse_nn_median_m"] for m in rows]),
                 "median_photometric_ratio": _med([m["holdout"]["photometric_ratio"] for m in rows]),
                 "median_outline_iou_fine": _med([m["holdout"]["outline_iou_fine"] for m in rows]),
                 "median_plane_ci_width_deg": _med([(m["plane_fine"] or {}).get("ci_width_deg") for m in rows]),
                 "median_total_without_sweeps_s": _med([m["time_s"]["total_without_sweeps"] for m in rows], 3)}
            bas = [m["ba"] for m in rows if m.get("ba") and m["ba"].get("status") == "solved"]
            if [m for m in rows if m.get("ba")]:
                rp = lambda k, w: _med([((b["reprojection_px_at_504_crop"] or {}).get(w) or {}).get(k) for b in bas])  # noqa: E731
                a["ba"] = {"solved": len(bas), "not_solved": [m["ba"]["status"] for m in rows if m.get("ba") and m["ba"].get("status") != "solved"],
                           "median_tracks": _med([b["tracks"] for b in bas]), "median_tracks_in_region": _med([b["tracks_in_region"] for b in bas]),
                           "median_reprojection_px_before": rp("median", "before"), "median_reprojection_px_after": rp("median", "after"),
                           "median_p90_reprojection_px_before": rp("p90", "before"), "median_p90_reprojection_px_after": rp("p90", "after"),
                           "median_rot_deg": _med([x for b in bas for x in b["pose_change"]["rot_deg"]]),
                           "median_move_cm": _med([x for b in bas for x in b["pose_change"]["move_cm"]]),
                           "max_move_cm": max([x for b in bas for x in b["pose_change"]["move_cm"]], default=None),
                           "any_at_bound": sum(bool(b["pose_change"]["at_bound"]) for b in bas),
                           "median_depth_scale": _med([x for b in bas for x in b["depth_scale"]]),
                           "median_depth_shift_m": _med([x for b in bas for x in b["depth_shift_m"]]),
                           "median_baseline_scale": _med([b.get("baseline_scale") for b in bas]),
                           "baseline_scale_range": [min([b["baseline_scale"] for b in bas if b.get("baseline_scale")], default=None),
                                                    max([b["baseline_scale"] for b in bas if b.get("baseline_scale")], default=None)],
                           "median_tri_sigma_m_at_0_5px": _med([(b.get("geometry") or {}).get("tri_sigma_m_at_0_5px") for b in bas]),
                           "median_consistency_before": {k: _med([(b.get("consistency_before") or {}).get(k) for b in bas])
                                                         for k in ("median_abs_pair_offset_m", "median_pair_mad_m", "median_abs_dz_m")},
                           "median_consistency_after": {k: _med([(b.get("consistency_after") or {}).get(k) for b in bas])
                                                        for k in ("median_abs_pair_offset_m", "median_pair_mad_m", "median_abs_dz_m")},
                           "median_depth_residual_m_before": _med([(b["depth_residual_m"]["before"] or {}).get("median") for b in bas]),
                           "median_depth_residual_m_after": _med([(b["depth_residual_m"]["after"] or {}).get("median") for b in bas]),
                           "median_resect_px_before": _med([((m["holdout"]["resect"] or {}).get("reprojection_px_before") or {}).get("median") for m in rows]),
                           "median_resect_px_after": _med([((m["holdout"]["resect"] or {}).get("reprojection_px_after") or {}).get("median") for m in rows]),
                           "median_time_split_s": {k: _med([m["time_split_s"][k] for m in rows if m.get("time_split_s")], 3)
                                                   for k in ("matching", "ba", "fuse_tsdf_1cm", "checks", "of_which_holdout_matching_ba_resect", "da3_crops_sam3")}}
            agg[f"{fs}:{meth}"] = a
    rep = {}
    for meth in METHODS:
        d = [x["tilt_repeatability"][meth]["abs_diff_deg"] for x in allrows if meth in x["tilt_repeatability"]]
        if d:
            rep[meth] = {"surface_spots": len(d), "median_abs_diff_deg": _med(d, 3), "max_abs_diff_deg": round(max(d), 3),
                         "within_1deg": sum(v <= 1 for v in d), "cis_overlap": sum(x["tilt_repeatability"][meth]["cis_overlap"] for x in allrows if meth in x["tilt_repeatability"])}
    for meth in METHODS:
        d = [x["face_matched_tilt"][meth]["abs_diff_deg"] for x in allrows if meth in x.get("face_matched_tilt", {})]
        if d:
            rep.setdefault(meth, {})["face_matched"] = {"surface_spots": len(d), "median_abs_diff_deg": _med(d, 3), "max_abs_diff_deg": round(max(d), 3),
                                                        "within_1deg": sum(v <= 1 for v in d), "within_2deg": sum(v <= 2 for v in d)}
    out = {"schema": "panoptes-x11-local-ba-results-v1", "run": run_dir.name, "boot": boot, "units": {
        "*_s": "seconds of wall time inside the container (perf_counter), models resident, cold start excluded",
        "*_m / *_cm / *_mm": "estimated: floor plane + an assumed 1.6 m camera height; never measured",
        "reprojection_px_at_504_crop": "pixels of the 504 px crop (x 720/504 = 1.43 for source pixels of the 1280x720 videos; 480/504 for the 640x480 factory clip)",
        "before / after (BA)": "before = the same objective with the coarse cameras and the anchor depth frozen (only the track points move); after = the joint solve",
        "tilt_repeatability": "frames A = X4's selection, frames B = the same selector with A's frames +-5 excluded; fine plane tilt |A - B|",
        "*_gib / peak_gb": "GiB, whole device (nvidia-smi, incl. vLLM), per stage window; over_90 flags > 90 %"},
        "confirmation_rule": x4.PASS, "matcher": {"lightglue_commit": LG_REV, "extractor": "ALIKED aliked-n16, 2048 keypoints, threshold 0.05, on the 504 px crop",
                                                  "verification": f"fundamental matrix MAGSAC {local_ba.RANSAC_PX} px", "licences": local_ba.LICENCES},
        "ba_settings": {"sigma_px": local_ba.SIGMA_PX, "pose_prior_sigma": [local_ba.SIGMA_R_DEG, local_ba.SIGMA_T_M],
                        "bounds_per_component": [local_ba.MAX_R_DEG, local_ba.MAX_T_M], "depth_sigma": local_ba.DEPTH_SIG,
                        "depth_scale_shift_prior_sigma": [local_ba.SIGMA_LS, local_ba.SIGMA_B_M], "depth_term_region_m": local_ba.REGION_M,
                        "loss": f"soft_l1, knee {local_ba.F_SCALE} sigma", "tri_field": [local_ba.FIELD_PX, local_ba.FIELD_SHRINK]},
        "hardware": "1 x A100-80GB (Modal), vLLM Qwen3-VL-8B on the same card at 30 %", "videos": videos, "aggregate": agg,
        "tilt_repeatability": rep, **(extra or {})}
    (run_dir / "results.json").write_text(json.dumps(out, indent=1, default=plain))
    return out


if __name__ == "__main__":
    if sys.argv[1:2] == ["--summarise"]:
        summarise(sys.argv[2], json.loads(Path(sys.argv[3]).read_text()) if len(sys.argv) > 3 else None)
    else:
        assert sys.argv[1:] == ["--self-check"], __doc__
        self_check()
