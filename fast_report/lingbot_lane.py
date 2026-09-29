"""X3, the LingBot fine lane: official LingBot-Map (lingbot_room's pins) per shot on one GPU -> a dense point layer, then
(when the fast core's DA3 frame is at hand) the camera-centre Sim3, the capped ICP and the D8 display gate.

A spawned worker process owns one GPU (CUDA_VISIBLE_DEVICES), loads LingBot once (boot, not analysis time) and runs jobs:
decode + shot cuts (fast_report.core's rules) -> per shot every `stride`-th frame, 518x294 crop mode on the GPU ->
inference_streaming (keyframe_interval = lingbot_room.streaming_interval, point head dropped: depth + camera only, the
lingbot_room contract) -> D7 confidence rule (report_runner.decide.conf_from_deciles over neighbour-view depth agreement)
-> multi-view filter (a pixel is kept when >= 1 neighbour view agrees within 4 % and no more views contradict than agree)
-> one point per voxel of one pixel footprint at the shot's median depth -> GLB (core.points_glb).
Everything is in LingBot's own units until align() puts it in the shot's DA3 metre frame (scale 'estimated').

    python -c "from fast_report import lingbot_lane; lingbot_lane.self_check()"   # needs torch (runs in the container)
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

LB_HW = (294, 518)       # 1280x720 in official crop mode: width 518, height round(720*518/1280/14)*14
TOL = .04                # lingbot_dense_map's tolerance_relative (4 % held 91 % of neighbour-view differences)
LAG_SRC = 10             # D7 diagnose pairs: source frames apart
NEIGHBOURS_SRC = (-20, -10, 10, 20)  # views each pixel is checked against, source frames apart
EDGE_JUMP, MIN_COS = .03, .12        # lingbot_dense_map: flying pixel / grazing view
CHUNK = 24               # frames per GPU batch in the filter
MAX_POINTS = 3_000_000   # per shot, the web layer's cap (the delivered ME340 map: 2.9M)


# ---------------------------------------------------------------- the worker process

def worker(gpu, compile_, jobs, results, weights, procs=8):
    """Spawned: owns physical GPU `gpu`; one job at a time; every reply is a dict with 'kind'."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    try:
        t = time.time()
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
        from fast_report import core
        pools = (ThreadPoolExecutor(4), ProcessPoolExecutor(procs, mp_context=multiprocessing.get_context("spawn")))
        list(pools[1].map(core.warm_worker, range(procs)))
        import torch
        model = load_model(torch.device("cuda:0"), weights, compile_)
        loaded = time.time()
        warm(model, compile_)
        self_check()
        torch.cuda.empty_cache()  # idle: the weights only, so a shared GPU's other tenants see what LingBot really holds
        results.put({"kind": "ready", "load_s": round(loaded - t, 2), "warm_s": round(time.time() - loaded, 2),
                     "gpu": str(torch.cuda.get_device_name(0)), "torch": str(torch.__version__), "compile": compile_,
                     "resident_gb": round(torch.cuda.memory_reserved(0) / 1e9, 2)})
    except BaseException as e:  # noqa: BLE001
        import traceback
        results.put({"kind": "error", "error": traceback.format_exc()[-3000:]})
        return
    while True:
        job = jobs.get()
        if job is None:
            return
        try:
            lane(model, job, results, pools)
        except BaseException:  # noqa: BLE001  reported, never retried
            import traceback
            results.put({"kind": "error", "id": job.get("id"), "error": traceback.format_exc()[-3000:]})
        torch.cuda.empty_cache()


def load_model(dev, weights, compile_):
    import torch
    from types import SimpleNamespace
    sys.path.insert(0, "/opt/lingbot")
    from demo import load_model as official_load
    args = SimpleNamespace(mode="streaming", image_size=518, patch_size=14, enable_3d_rope=True, max_frame_num=1024,
                           kv_cache_sliding_window=64, num_scale_frames=8, use_sdpa=True, camera_num_iterations=4, model_path=weights)
    model = official_load(args, dev)
    model.aggregator = model.aggregator.to(dtype=torch.bfloat16)  # lingbot_room's setting
    if compile_:  # demo.compile_model's targets, default mode: its reduce-overhead (CUDA graphs) fails under torch 2.14
        agg = model.aggregator  # ('accessing tensor output of CUDAGraphs that has been overwritten', run fx-x3-lingbot-001 A2)
        for blocks in (agg.frame_blocks, agg.patch_embed.blocks):
            for i, b in enumerate(blocks):
                blocks[i] = torch.compile(b)
        for b in agg.global_blocks:
            for name in ("attn_pre", "ffn_residual"):
                if hasattr(b, name):
                    setattr(b, name, torch.compile(getattr(b, name)))
            b.attn.proj = torch.compile(b.attn.proj)
    model.point_head = None  # depth + camera only (lingbot_room: 'no point head'); gct_profile drops it too
    return model.eval()


def warm(model, compile_):
    """Kernels/allocator at the run shapes; with compile_, the demo's CUDA-graph warm-up for both keyframe paths."""
    import torch
    imgs = torch.rand(40, 3, *LB_HW, device="cuda:0")
    for ki in ((1, 2, 3) if compile_ else (2,)):  # every keyframe path the runs take (compile: traced here, not in a run)
        infer(model, imgs[:24], ki)
    torch.cuda.synchronize()


# ---------------------------------------------------------------- one video

def decode_and_cut(mp4, pools):
    """core.analyse's decode loop without keyframes or SAM: frames (BGR list), cuts, shots [a, b] inclusive, fps."""
    import cv2
    from fast_report import core
    import detect_shot_cuts as dsc
    cap = cv2.VideoCapture(mp4)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames, grays, futures, a = [], [], [], 0
    threads, procs = pools
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(bgr)
        grays.append(threads.submit(core.gray_sharp, bgr))
        if len(frames) == a + core.CHUNK + dsc.SPAN + 1:
            a0 = max(0, a - 2)
            futures.append(procs.submit(core.measure_chunk, [g.result()[0] for g in grays[a0:]], a, a + core.CHUNK, a0, len(frames)))
            a += core.CHUNK
    cap.release()
    n = len(frames)
    decoded = time.time()
    gray_all = [g.result()[0] for g in grays]
    for a1 in range(a, n, core.TAIL):
        b1, lo = min(a1 + core.TAIL, n), max(0, a1 - 2)
        futures.append(procs.submit(core.measure_chunk, gray_all[lo:min(n, b1 + dsc.SPAN + 1)], a1, b1, lo, min(n, b1 + dsc.SPAN + 1)))
    cuts = core.cuts_from(core.stitch([f.result() for f in futures]), n)
    shots = [(int(a), int(b)) for a, b in cuts["segments"] if b - a + 1 >= core.MIN_SHOT]
    return frames, cuts, shots, fps, decoded


def preprocess(frames_bgr, dev):
    """(n,720,1280,3) uint8 BGR numpy -> (n,3,294,518) RGB [0,1] on dev: official crop mode's resize, bicubic with
    antialias on the GPU (PIL's bicubic in the official loader; not byte-identical)."""
    import torch
    import torch.nn.functional as F
    out = []
    for i in range(0, len(frames_bgr), 32):
        x = torch.from_numpy(np.stack(frames_bgr[i:i + 32])).to(dev, non_blocking=True)
        x = x.permute(0, 3, 1, 2).flip(1).float() / 255
        out.append(F.interpolate(x, size=LB_HW, mode="bicubic", antialias=True, align_corners=False).clamp_(0, 1))
    return torch.cat(out)


def infer(model, imgs, ki):
    """-> depth (n,H,W), conf (n,H,W), c2w (n,4,4), K (n,3,3) on the GPU, float32. c2w is pose_encoding_to_extri_intri's
    matrix itself: lingbot_room saves demo.postprocess's inverse of it as 'w2c' and lingbot_dense_map inverts that back."""
    import torch
    sys.path.insert(0, "/opt/lingbot")
    from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        pred = model.inference_streaming(imgs, num_scale_frames=8, keyframe_interval=ki, output_device=imgs.device)
    with torch.inference_mode():
        ext, K = pose_encoding_to_extri_intri(pred["pose_enc"], imgs.shape[-2:])
        n = imgs.shape[0]
        c2w = torch.eye(4, device=imgs.device).repeat(n, 1, 1)
        c2w[:, :3, :4] = ext.reshape(n, 3, 4).float()
        return pred["depth"].reshape(n, *LB_HW).float(), pred["depth_conf"].reshape(n, *LB_HW).float(), c2w, K.reshape(n, 3, 3).float()


def lane(model, job, results, pools):
    """job: {id, mp4, stride, t0_unix, tmp, out (volume dir for the raw record), align (dir where the core's DA3 files appear)
    or None}. Replies: one 'shot' per shot as its layer part is ready (longest shot first), then 'points' (the record), then
    'done' once the raw record is saved (not analysis time)."""
    import torch
    from fast_report import core
    from report_runner.decide import conf_from_deciles
    dev = torch.device("cuda:0")
    stamps = {"t0_unix": job["t0_unix"], "picked_up_unix": time.time()}
    stage = lambda k, a, b, **n: stamps.setdefault("stages", []).append({"stage": k, "start_unix": a, "end_unix": b, **({"n": n} if n else {})})  # noqa: E731
    t = time.time()
    frames, cuts, shots, fps, decoded = decode_and_cut(job["mp4"], pools)
    stage("lingbot.decode", t, decoded, frames=len(frames))
    stage("lingbot.cuts", decoded, time.time(), shots=len(shots))
    stride, out_shots = job["stride"], []
    torch.cuda.reset_peak_memory_stats(dev)
    for si, (a, b) in sorted(enumerate(shots), key=lambda x: x[1][0] - x[1][1]):  # longest shot first: the one the report is about
        idx = list(range(a, b + 1, stride))
        if len(idx) < 8:
            continue
        assert len(idx) <= 1000, "LingBot positional limit (max_frame_num 1024)"
        t = time.time()
        imgs = preprocess([frames[i] for i in idx], dev)
        torch.cuda.synchronize()
        t1 = time.time()
        ki = (len(idx) + 319) // 320  # lingbot_room.streaming_interval
        depth, conf, c2w, K = infer(model, imgs, ki)
        torch.cuda.synchronize()
        t2 = time.time()
        stage(f"lingbot.pre.shot{si}", t, t1, frames=len(idx))
        stage(f"lingbot.infer.shot{si}", t1, t2, frames=len(idx), keyframe_interval=ki)
        peak_infer = torch.cuda.max_memory_allocated(dev)
        diag = diagnose(depth, conf, c2w, K, max(1, round(LAG_SRC / stride)))
        thr = conf_from_deciles(diag["by_conf_decile"])
        t3 = time.time()
        stage(f"lingbot.conf.shot{si}", t2, t3)
        sh = {"shot": si, "frames": [a, b], "source_frames": idx, "keyframe_interval": ki, "conf_threshold": thr, "diagnose": diag,
              "peak_alloc_infer_gb": round(peak_infer / 1e9, 2), "fps_infer": round(len(idx) / (t2 - t1), 2), "raw": (depth, conf, c2w, K)}
        out_shots.append(sh)
        if thr is None:  # D7: no decile agrees -> no dense map for this shot
            sh["withheld"] = "no confidence decile reaches 90 % within 4 %"
            results.put({"kind": "shot", "row": {"shot": si, "frames": [a, b], "withheld": sh["withheld"]}, "blob": None})
            continue
        sh["points"] = pts = points(depth, conf, c2w, K, imgs, thr, offsets(stride))
        torch.cuda.synchronize()
        t4 = time.time()
        stage(f"lingbot.points.shot{si}", t3, t4, points=int(len(pts["xyz"])), filtered=pts["stats"]["kept"], one_layer=pts["stats"]["after_one_layer"])
        al = {}
        if job.get("align"):
            al = sh["align"] = align_shot(sh, job["align"])
            t5 = time.time()
            stage(f"lingbot.align.shot{si}", t4 + al.get("waited_for_core_s", 0.), t5, use=al.get("use"))
            if al.get("waited_for_core_s"):
                stage(f"lingbot.wait_core.shot{si}", t4, t4 + al["waited_for_core_s"])
        t5 = time.time()
        if job.get("align") and not al.get("use"):  # D8: the gate failed (raw and capped ICP): no dense points for this shot
            results.put({"kind": "shot", "row": {"shot": si, "frames": [a, b], "withheld": "D8 display gate failed against the DA3 TSDF",
                                                 "gate": {k: al.get(k) for k in ("metrics", "steps", "sim3", "error")}}, "blob": None})
            continue
        xyz = pts["xyz"]
        if al.get("use"):
            sc, R, tr = al["transform"]["s"], np.array(al["transform"]["R"]), np.array(al["transform"]["t"])
            xyz = (sc * xyz.astype(np.float64) @ R.T + tr).astype(np.float32)
        path = Path(job["tmp"]) / f"points-{si}.glb"
        raw, meta = core.points_glb(xyz, pts["rgb"], float(pts["cell"] * (al["transform"]["s"] if al.get("use") else 1.)))
        path.write_bytes(raw)
        frame = f"shot-{al['da3_shot']}" if al.get("use") else f"lingbot-shot-{si}"
        row = {"shot": si, "frames": [a, b], "frame": frame, "points": int(len(xyz)), "conf_threshold": thr, "cell_native": pts["cell"],
               "units": "DA3 metres (estimated)" if al.get("use") else "LingBot native (uncalibrated)",
               "gate": {k: al.get(k) for k in ("use", "metrics", "steps", "sim3", "error")} if al else None, "points_blob": f"points-{si}"}
        stage(f"lingbot.pack.shot{si}", t5, time.time())
        results.put({"kind": "shot", "row": row, "blob": (f"points-{si}", str(path), {**meta, "frame": frame})})
    stamps["peak_alloc_gb"] = round(torch.cuda.max_memory_allocated(dev) / 1e9, 2)
    stamps["peak_reserved_gb"] = round(torch.cuda.max_memory_reserved(dev) / 1e9, 2)
    results.put({"kind": "points", "id": job["id"], "stamps": stamps, "cuts": cuts, "shots": shots, "fps": fps,
                 "shot_stats": [{k: v for k, v in s.items() if k not in ("points", "raw")} | ({"points_stats": s["points"]["stats"]} if "points" in s else {})
                                for s in out_shots]})
    t = time.time()  # after the layer: the raw record for the evaluation (not analysis time)
    out = Path(job["out"])
    out.mkdir(parents=True, exist_ok=True)
    for s in out_shots:
        if "points" not in s:
            continue
        d, c, cw, k = (x.cpu().numpy() for x in s["raw"])
        p = s["points"]
        np.savez(out / f"shot{s['shot']}.npz", depth=d.astype(np.float16), conf=c.astype(np.float16), c2w=cw, K=k,
                 source_frames=np.array(s["source_frames"]), xyz=p["xyz"], rgb=p["rgb"], frame=p["frame"], pconf=p["conf"], agree=p["agree"],
                 cell=p["cell"], thr=s["conf_threshold"])
    results.put({"kind": "done", "id": job["id"], "saved_s": round(time.time() - t, 2)})


# ---------------------------------------------------------------- geometry on the GPU

def offsets(stride):
    """NEIGHBOURS_SRC in LingBot frames of this stride (at least one frame apart, signs kept)."""
    return [max(1, round(abs(o) / stride)) * (1 if o > 0 else -1) for o in NEIGHBOURS_SRC]


def world_points(depth, K, c2w):
    """(B,H,W) depth, (B,3,3), (B,4,4) -> (B,H,W,3) world, and the camera-frame points."""
    import torch
    B, H, W = depth.shape
    v, u = torch.meshgrid(torch.arange(H, device=depth.device, dtype=torch.float32), torch.arange(W, device=depth.device, dtype=torch.float32), indexing="ij")
    x = (u - K[:, 0, 2, None, None]) / K[:, 0, 0, None, None] * depth
    y = (v - K[:, 1, 2, None, None]) / K[:, 1, 1, None, None] * depth
    cam = torch.stack([x, y, depth], -1)
    return torch.einsum("bhwc,bdc->bhwd", cam, c2w[:, :3, :3]) + c2w[:, None, None, :3, 3], cam


def project(world, K, c2w):
    """world (B,N,3) into cameras (B,...) -> u, v, z (B,N)."""
    import torch
    cam = torch.einsum("bnd,bdc->bnc", world - c2w[:, None, :3, 3], c2w[:, :3, :3])
    z = cam[..., 2]
    zs = torch.where(z > 1e-6, z, torch.ones_like(z))
    return K[:, 0, 0, None] * cam[..., 0] / zs + K[:, 0, 2, None], K[:, 1, 1, None] * cam[..., 1] / zs + K[:, 1, 2, None], z


def lookup(img, u, v):
    """Nearest pixel of img (B,H,W) at (u, v) (B,N) -> values, inside mask."""
    import torch
    B, H, W = img.shape
    ui, vi = torch.round(u).long(), torch.round(v).long()
    inside = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    flat = (vi.clamp(0, H - 1) * W + ui.clamp(0, W - 1))
    return torch.gather(img.reshape(B, -1), 1, flat), inside


def usable(depth, conf, thr):
    """Confident, finite, positive, not a flying pixel (4-neighbour jump > EDGE_JUMP, grown 1 px), not grazing."""
    import torch
    import torch.nn.functional as F
    pad = F.pad(depth[:, None], (1, 1, 1, 1), mode="replicate")[:, 0]
    nb = torch.stack([pad[:, 1:-1, :-2], pad[:, 1:-1, 2:], pad[:, :-2, 1:-1], pad[:, 2:, 1:-1]])
    edge = (nb - depth).abs().amax(0) > EDGE_JUMP * depth
    edge = F.max_pool2d(edge[:, None].float(), 3, 1, 1)[:, 0] > 0
    return torch.isfinite(depth) & (depth > 0) & (conf > thr) & ~edge


def facing(cam):
    """|cos| between viewing ray and surface normal from camera-frame points (B,H,W,3) (lingbot_dense_map.facing)."""
    import torch
    du = torch.gradient(cam, dim=2)[0]
    dv = torch.gradient(cam, dim=1)[0]
    n = torch.cross(du, dv, dim=-1)
    return (n * cam).sum(-1).abs() / (n.norm(dim=-1) * cam.norm(dim=-1)).clamp(min=1e-12)


def diagnose(depth, conf, c2w, K, lag, every=5):
    """lingbot_dense_map.diagnose_remote's by_conf_decile on the GPU: frame i vs i + lag (LingBot frames), every 5th i,
    both pixels usable at conf > 1, rel = (d_b - z) / z; share within 4 % per confidence decile."""
    import torch
    n = len(depth)
    conf_s = conf[:, ::4, ::4].reshape(-1)
    conf_s = conf_s[torch.isfinite(conf_s)]
    rel, cc = [], []
    for i in range(0, n - lag, every):
        ok_a = usable(depth[i:i + 1], conf[i:i + 1], 1.)[0]
        ok_b = usable(depth[i + lag:i + lag + 1], conf[i + lag:i + lag + 1], 1.)
        w, _ = world_points(depth[i:i + 1], K[i:i + 1], c2w[i:i + 1])
        pts = w[0][ok_a][None]
        u, v, z = project(pts, K[i + lag:i + lag + 1], c2w[i + lag:i + lag + 1])
        d_b, inside = lookup(depth[i + lag:i + lag + 1], u, v)
        seen, _ = lookup(ok_b.float(), u, v)
        m = inside & (z > 0) & (seen > 0)
        rel.append(((d_b - z) / z)[m])
        cc.append(conf[i][ok_a][None][m])
    rel, cc = torch.cat(rel), torch.cat(cc)
    q = torch.quantile(conf_s[torch.randperm(len(conf_s), device=conf_s.device)[:1_000_000]].float(), torch.linspace(0, 1, 11, device=conf_s.device))
    rows = []
    for lo, hi in zip(q[:-1].tolist(), q[1:].tolist()):
        x = rel[(cc >= lo) & (cc <= hi)].abs()
        if len(x):
            rows.append({"conf_from": lo, "conf_to": hi, "samples": int(len(x)), "median_abs_rel": float(x.median()),
                         "share_within_2pct": float((x < .02).float().mean()), "share_within_4pct": float((x < .04).float().mean()),
                         "share_within_8pct": float((x < .08).float().mean())})
    a = rel.abs()
    return {"pairs_lag_frames": lag, "samples": int(len(a)), "median_abs": float(a.median()) if len(a) else None,
            "share_within_4pct": float((a < .04).float().mean()) if len(a) else None, "by_conf_decile": rows}


def points(depth, conf, c2w, K, imgs, thr, offsets, one_layer=True, max_points=MAX_POINTS):
    """Multi-view filter, then (one_layer) lingbot_dense_map's 'one layer per surface' as a GPU z-buffer pass, then one point
    per voxel (finest sample wins; the cell = one pixel footprint at the shot's median depth, grown x1.25 until <= max_points).
    -> numpy xyz (float32, LingBot units), rgb uint8, per point: frame index, conf, agreeing views; cell; stats."""
    import torch
    n, H, W = depth.shape
    ok_all = usable(depth, conf, thr)
    good_all = torch.zeros((n, H * W), dtype=torch.bool, device=depth.device)
    agree_all = torch.zeros((n, H * W), dtype=torch.int8, device=depth.device)
    tallies = {"usable": 0, "grazing": 0, "contradicted": 0, "no_agreeing_view": 0, "kept": 0}
    med = float(depth[ok_all].median())
    cell = med / float(K[:, 0, 0].median())
    for i0 in range(0, n, CHUNK):
        i1 = min(n, i0 + CHUNK)
        ok = ok_all[i0:i1]
        world, cam = world_points(depth[i0:i1], K[i0:i1], c2w[i0:i1])
        graze = facing(cam) <= MIN_COS
        ok_ng = ok & ~graze
        tallies["usable"] += int(ok.sum())
        tallies["grazing"] += int((ok & graze).sum())
        B = i1 - i0
        flat = world.reshape(B, -1, 3)
        agree = torch.zeros(flat.shape[:2], dtype=torch.int8, device=depth.device)
        contra = torch.zeros_like(agree)
        for o in offsets:
            j = torch.arange(i0, i1, device=depth.device) + o
            has = (j >= 0) & (j < n)
            jj = j.clamp(0, n - 1)
            u, v, z = project(flat, K[jj], c2w[jj])
            dj, inside = lookup(depth[jj], u, v)
            okj, _ = lookup(ok_all[jj].float(), u, v)
            m = has[:, None] & inside & (z > 0) & (okj > 0)
            r = (z - dj) / dj.clamp(min=1e-9)
            agree += (m & (r.abs() <= TOL)).to(torch.int8)
            contra += (m & (r < -TOL)).to(torch.int8)
        cand = ok_ng.reshape(B, -1)
        good_all[i0:i1] = cand & (agree >= 1) & (agree >= contra)
        agree_all[i0:i1] = agree
        tallies["contradicted"] += int((cand & (contra > agree)).sum())
        tallies["no_agreeing_view"] += int((cand & (agree == 0) & (contra == 0)).sum())
    tallies["kept"] = int(good_all.sum())
    fr, px = one_layer_pass(depth, K, c2w, good_all) if one_layer else torch.nonzero(good_all, as_tuple=True)
    tallies["after_one_layer"] = int(len(fr)) if one_layer else None
    d = depth.reshape(n, -1)[fr, px]
    vy, vx = (px // W).float(), (px % W).float()
    k = K[fr]
    cam = torch.stack([(vx - k[:, 0, 2]) / k[:, 0, 0] * d, (vy - k[:, 1, 2]) / k[:, 1, 1] * d, d], 1)
    xyz = (c2w[fr, :3, :3] @ cam[:, :, None])[:, :, 0] + c2w[fr, :3, 3]
    fp = d / k[:, 0, 0]
    order = torch.argsort(fp)
    while True:
        ks, perm = torch.sort(voxel_keys(xyz, cell)[order], stable=True)
        first = torch.ones_like(ks, dtype=torch.bool)
        first[1:] = ks[1:] != ks[:-1]
        sel = order[perm[first]]
        if len(sel) <= max_points:
            break
        cell *= 1.25
    fr, px = fr[sel], px[sel]
    out = {"xyz": xyz[sel].cpu().numpy(), "rgb": (imgs.permute(0, 2, 3, 1).reshape(n, -1, 3)[fr, px] * 255).round().to(torch.uint8).cpu().numpy(),
           "frame": fr.int().cpu().numpy(), "conf": conf.reshape(n, -1)[fr, px].half().cpu().numpy(), "agree": agree_all[fr, px].cpu().numpy(), "cell": cell,
           "pixel": px.cpu().numpy()}
    out["stats"] = {**tallies, "points": int(len(sel)), "cell_native": cell, "median_depth_native": med, "offsets_lb_frames": list(offsets),
                    "one_layer": one_layer, "max_points": max_points}
    return out


def one_layer_pass(depth, K, c2w, good, tally=None):
    """lingbot_dense_map's rule on the GPU. Frames in order; every accepted point is projected into this view and compared
    with this view's depth at its pixel: on the surface (within TOL) it confirms, in front of it (this view sees through it)
    it is contradicted. A filtered pixel near (3x3) a confirming point whose footprint is no coarser than 1/0.7 of its own is
    skipped; where only coarser points confirm, this finer sample replaces them (REPLACE .7: they die); elsewhere it is added.
    Points more views contradict than confirm are dropped at the end. -> (frame, pixel) of the surviving points."""
    import torch
    import torch.nn.functional as F
    n, H, W = depth.shape
    dflat = depth.reshape(n, -1)
    cap = int(good.sum())
    dev = depth.device
    P = torch.empty((cap, 3), device=dev)
    FP = torch.empty(cap, device=dev)
    FR = torch.empty(cap, dtype=torch.long, device=dev)
    PX = torch.empty(cap, dtype=torch.long, device=dev)
    alive = torch.zeros(cap, dtype=torch.bool, device=dev)
    confirm = torch.zeros(cap, dtype=torch.int16, device=dev)
    contra = torch.zeros(cap, dtype=torch.int16, device=dev)
    m = 0
    for i in range(n):
        px = torch.nonzero(good[i], as_tuple=True)[0]
        if not len(px):
            continue
        d = dflat[i, px]
        k = K[i]
        vy, vx = (px // W).float(), (px % W).float()
        cam = torch.stack([(vx - k[0, 2]) / k[0, 0] * d, (vy - k[1, 2]) / k[1, 1] * d, d], 1)
        world = cam @ c2w[i, :3, :3].T + c2w[i, :3, 3]
        fpi = d / k[0, 0]
        if m:
            u, v, z = project(P[:m][None], K[i:i + 1], c2w[i:i + 1])
            ui, vi, z = torch.round(u[0]).long(), torch.round(v[0]).long(), z[0]
            sel = torch.nonzero(alive[:m] & (z > 0) & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H), as_tuple=True)[0]
            idx, zs = vi[sel] * W + ui[sel], z[sel]
            dimg = torch.zeros(H * W, device=dev)
            dimg[px] = d
            dd = dimg[idx]
            seen = dd > 0
            match = seen & ((zs - dd).abs() <= TOL * dd)
            front = seen & (zs < dd * (1 - TOL))
            confirm[sel[match]] += 1
            contra[sel[front]] += 1
            fb = torch.full((H * W,), float("inf"), device=dev).scatter_reduce(0, idx[match], FP[sel[match]], "amin")
            fb3 = -F.max_pool2d(-fb.view(1, 1, H, W), 3, 1, 1).view(-1)
            covered = torch.isfinite(fb3[px])
            skip = covered & (fb3[px] * .7 <= fpi)
            replace = covered & ~skip
            if replace.any():  # the coarser confirming points around a finer sample die
                rimg = torch.zeros(H * W, dtype=torch.bool, device=dev)
                rimg[px[replace]] = True
                rimg = F.max_pool2d(rimg.view(1, 1, H, W).float(), 3, 1, 1).view(-1) > 0
                kill = match & rimg[idx] & (FP[sel] * .7 > dd / k[0, 0])
                alive[sel[kill]] = False
            if tally is not None:  # per-frame decisions (GPU syncs: diagnostics only)
                tally.append({"frame": i, "candidates": int(len(d)), "covered": int(covered.sum()), "skip": int(skip.sum()), "replace": int(replace.sum()),
                              "points_confirming": int(match.sum()), "points_seen_through": int(front.sum()), "alive": int(alive[:m].sum())})
            px, world, fpi = px[~skip], world[~skip], fpi[~skip]
        c = len(px)
        P[m:m + c], FP[m:m + c], PX[m:m + c] = world, fpi, px
        FR[m:m + c] = i
        alive[m:m + c] = True
        m += c
    keep = torch.nonzero(alive[:m] & (contra[:m] <= confirm[:m]), as_tuple=True)[0]
    return FR[keep], PX[keep]


def densify(depth, K, c2w, frame, pixel, full_rgb, s=2):
    """Display density (the delivered map samples 3x): each point's pixel -> s x s sub-samples of its own view, depth
    bilinear (the pixel's own depth where a neighbour is off its surface by more than TOL), colour bilinear from the
    full-resolution frame (full_rgb (n,Hf,Wf,3) uint8, same frame order). No new geometry. -> xyz, rgb (torch)."""
    import torch
    n, H, W = depth.shape
    Hf, Wf = full_rgb.shape[1:3]
    o = (torch.arange(s, device=depth.device) + .5) / s - .5
    oy, ox = (t.reshape(-1) for t in torch.meshgrid(o, o, indexing="ij"))
    f = frame[:, None].expand(-1, s * s)
    u, v = (pixel % W).float()[:, None] + ox, (pixel // W).float()[:, None] + oy

    def bilinear(img, x, y):
        x0, y0 = x.floor().clamp(0, img.shape[2] - 2).long(), y.floor().clamp(0, img.shape[1] - 2).long()
        wx, wy = (x - x0).clamp(0, 1), (y - y0).clamp(0, 1)
        g = lambda yy, xx: img[f, yy, xx]  # noqa: E731
        if img.dim() == 4:
            wx, wy = wx[..., None], wy[..., None]
        return (g(y0, x0) * (1 - wx) + g(y0, x0 + 1) * wx) * (1 - wy) + (g(y0 + 1, x0) * (1 - wx) + g(y0 + 1, x0 + 1) * wx) * wy
    dc = depth.reshape(n, -1)[frame, pixel][:, None]
    db = bilinear(depth, u, v)
    d = torch.where((db - dc).abs() <= TOL * dc, db, dc)
    k = K[f]
    cam = torch.stack([(u - k[..., 0, 2]) / k[..., 0, 0] * d, (v - k[..., 1, 2]) / k[..., 1, 1] * d, d], -1)
    xyz = torch.einsum("npc,npdc->npd", cam, c2w[f, :3, :3]) + c2w[f, :3, 3]
    rgb = bilinear(full_rgb.float(), (u + .5) * Wf / W - .5, (v + .5) * Hf / H - .5)
    return xyz.reshape(-1, 3), rgb.reshape(-1, 3).round().clamp(0, 255).to(torch.uint8)


def voxel_keys(xyz, cell):
    import torch
    q = torch.floor(xyz / cell).long() + (1 << 20)
    return (q[:, 0] << 42) | (q[:, 1] << 21) | q[:, 2]


# ---------------------------------------------------------------- alignment to the fast core's DA3 frame

def align_shot(s, folder, wait_s=120.):
    """The core's shot (most shared frames) from FOLDER/da3.json + da3-points-<shot>.npy (written by the main process when
    its cameras and room layers exist): robust Sim3 on camera centres, D8 gate, capped ICP when the gate fails."""
    folder = Path(folder)
    end = time.time() + wait_s
    while not (folder / "ready").exists():
        if time.time() > end:
            return {"use": None, "error": "the core's DA3 layers never arrived"}
        time.sleep(.02)
    t_waited = time.time()
    da3 = json.loads((folder / "da3.json").read_text())
    lb_frames = {f: i for i, f in enumerate(s["source_frames"])}
    best = max(da3["shots"], key=lambda d: sum(k in lb_frames for k in d["keyframes"]))
    room = np.load(folder / f"da3-points-{best['index']}.npy")
    c2w = s["raw"][2].cpu().numpy().astype(np.float64)
    rec = align_points(s["points"]["xyz"], s["points"]["conf"], s["conf_threshold"], c2w, lb_frames, best, room)
    rec["waited_for_core_s"] = round(t_waited - (end - wait_s), 3)
    return rec


def align_points(xyz, pconf, thr, lb_c2w, lb_frames, da3_shot, room, icp_always=False):
    """-> {use: raw|icp|None, transform {s,R,t} (LingBot -> DA3 metres), sim3 residuals, metrics per stage, steps}."""
    from lingbot_dense_map import robust_sim3
    keys = [k for k in da3_shot["keyframes"] if k in lb_frames]
    rec = {"da3_shot": da3_shot["index"], "matched_keyframes": len(keys)}
    if len(keys) < 8:
        return {**rec, "use": None, "error": "fewer than 8 shared keyframes"}
    import m3_exp_geometry as geo
    dst_c2w = np.array([da3_shot["c2w_m"][da3_shot["keyframes"].index(k)] for k in keys], np.float64)
    src_c2w = lb_c2w[[lb_frames[k] for k in keys]]
    dst, src = dst_c2w[:, :3, 3], src_c2w[:, :3, 3]
    (s, R, t), resid, inl = robust_sim3(src, dst)
    floor = da3_shot["floor"]
    # two camera Sim3s: the delivered rule (robust, centres only) and, on its inliers, rotation from the camera orientations
    # (m3_exp_geometry.align_sim3: centres alone leave the roll about a straight walk free); the one the gate prefers wins
    cands = {"centres": (s, R, t), "orientations": geo.align_sim3(src_c2w[inl], dst_c2w[inl])}
    rec["sim3"], rec["metrics"] = {}, {}
    for name, (cs, cR, ct) in cands.items():
        r = np.linalg.norm(src @ (cs * cR).T + ct - dst, axis=1)
        rec["sim3"][name] = {"scale_m_per_native": cs, "inliers_of_centres_fit": int(inl.sum()),
                             "residual_m": {"median": float(np.median(r)), "p90": float(np.percentile(r, 90)), "max": float(r.max())}}
        rec["metrics"][f"raw_{name}"] = gate_metrics(cs * xyz.astype(np.float64) @ cR.T + ct, room, floor)
    rec["path_m"] = float(np.linalg.norm(np.diff(dst, axis=0), axis=1).sum())
    ok = [n for n in cands if passes(rec["metrics"][f"raw_{n}"])]
    pick = ok[0] if ok else max(cands, key=lambda n: rec["metrics"][f"raw_{n}"]["within_25cm_all_points"])
    s, R, t = cands[pick]
    rec["camera_sim3_used"] = pick
    rec["metrics"]["raw"] = rec["metrics"][f"raw_{pick}"]
    pts = s * xyz.astype(np.float64) @ R.T + t
    use = "raw" if ok else None
    if use is None or icp_always:
        (s2, R2, t2), steps, refused = icp(pts[pconf.astype(np.float32) > thr], room)
        rec["steps"] = {"iterations": len(steps), "max_step_deg": max(x["step_deg"] for x in steps), "max_step_move_m": max(x["step_move_m"] for x in steps),
                        "refused": refused, "last_median_cm": steps[-1]["median_cm"]}
        if not refused:
            pts_icp = s2 * pts @ R2.T + t2
            rec["metrics"]["icp"] = gate_metrics(pts_icp, room, floor)
            if use is None and passes(rec["metrics"]["icp"]):
                use = "icp"
                s, R, t = s2 * s, R2 @ R, s2 * R2 @ t + t2
    rec["use"] = use
    rec["transform"] = {"s": float(s), "R": np.asarray(R).tolist(), "t": np.asarray(t).tolist()}
    return rec


def gate_metrics(pts, room, floor, sample=300_000):
    """D8's two numbers against the DA3 TSDF points (3 cm) in place of the report's fused mesh: share of points within 25 cm
    (a 300k sample), floor offset (report_runner.decide.floor_offset_m on a 1M sample, 10 cm plane tolerance, metres)."""
    from scipy.spatial import cKDTree
    from report_runner import decide
    rng = np.random.default_rng(0)
    q = pts[rng.choice(len(pts), min(sample, len(pts)), replace=False)]
    d, _ = cKDTree(room).query(q, distance_upper_bound=.25, workers=8)
    scale = {"metres_per_native_unit": 1., "up_native": floor["normal"], "plane_point_native": floor["point_m"], "plane_tolerance_native": .1}
    fq = pts[rng.choice(len(pts), min(1_000_000, len(pts)), replace=False)]  # a median of cell medians: a 1M sample
    off, cells = decide.floor_offset_m(fq, room, scale) if floor else (None, 0)
    return {"within_25cm_all_points": round(float(np.isfinite(d).mean()), 4), "floor_offset_m": off, "floor_cells": cells,
            "median_cm_within_25": round(float(np.median(d[np.isfinite(d)])) * 100, 2) if np.isfinite(d).any() else None}


def passes(m):
    from report_runner.decide import DENSE_FLOOR_M, DENSE_WITHIN_25CM
    return m["within_25cm_all_points"] >= DENSE_WITHIN_25CM and m["floor_offset_m"] is not None and abs(m["floor_offset_m"]) <= DENSE_FLOOR_M


def icp(pts, target, sub=50_000):
    """lingbot_icp_refine.main's loop on arrays (metres): trimmed similarity ICP, reach 1.0 -> 0.5 -> 0.25 m, the per-step
    caps; 50k query points (the script: 400k) against the DA3 TSDF points (the script: 3M mesh samples)."""
    from scipy.spatial import cKDTree
    from lingbot_dense_map import umeyama
    from lingbot_icp_refine import MAX_STEP_DEG, MAX_STEP_MOVE_M, TRIM, angle_deg
    tree = cKDTree(target)
    rng = np.random.default_rng(0)
    q = pts[rng.choice(len(pts), min(sub, len(pts)), replace=False)]
    s, R, t, steps, refused = 1., np.eye(3), np.zeros(3), [], None
    for reach, iterations in ((1., 15), (.5, 15), (.25, 20)):
        for _ in range(iterations):
            cur = s * q @ R.T + t
            d, idx = tree.query(cur, distance_upper_bound=reach, workers=8)
            ok = np.isfinite(d)
            if ok.sum() < 100:
                break
            keep = ok & (d <= np.quantile(d[ok], TRIM))
            ds, dR, dt = umeyama(cur[keep], target[idx[keep]])
            step = {"step_deg": angle_deg(dR), "step_move_m": float(np.median(np.linalg.norm(ds * cur[keep] @ dR.T + dt - cur[keep], axis=1))),
                    "reach_m": reach, "median_cm": round(float(np.median(d[keep])) * 100, 2)}
            steps.append(step)
            if step["step_deg"] > MAX_STEP_DEG or step["step_move_m"] > MAX_STEP_MOVE_M:
                refused = step
                break
            s, R, t = ds * s, dR @ R, ds * dR @ t + dt
            if abs(ds - 1) < 1e-6 and angle_deg(dR) < 1e-4 and np.linalg.norm(dt) < 1e-4:
                break
        if refused:
            break
    return (s, R, t), steps or [{"step_deg": 0., "step_move_m": 0., "median_cm": None}], refused


# ---------------------------------------------------------------- check

def self_check():
    """A plane seen by 3 cameras: every pixel agrees; a floater in one view is contradicted; voxel dedupe keeps one per
    cell; the Sim3 recovers a known similarity; the D8 gate passes a copy of the room."""
    import torch
    H, W = LB_HW
    K = torch.tensor([[200., 0, W / 2], [0, 200., H / 2], [0, 0, 1]]).repeat(3, 1, 1)
    c2w = torch.eye(4).repeat(3, 1, 1)
    c2w[1, 0, 3], c2w[2, 0, 3] = .05, .10
    depth = torch.full((3, H, W), 2.)
    conf = torch.full((3, H, W), 5.)
    depth[1, 100:110, 200:210] = 1.  # something in view 1 only: views 0 and 2 see through it
    imgs = torch.rand(3, 3, H, W)
    p = points(depth, conf, c2w, K, imgs, 1.5, [-1, 1], one_layer=False)
    assert p["stats"]["contradicted"] >= 30 and not (np.abs(p["xyz"][:, 2] - 1.) < .01).any(), p["stats"]
    assert abs(float(np.median(p["xyz"][:, 2])) - 2.) < 1e-4
    assert offsets(2) == [-10, -5, 5, 10] and offsets(1) == [-20, -10, 10, 20] and offsets(40) == [-1, -1, 1, 1]
    lay = points(depth, conf, c2w, K, imgs, 1.5, [-1, 1], one_layer=True)
    assert lay["stats"]["after_one_layer"] < .5 * lay["stats"]["kept"], lay["stats"]  # the plane is seen 3x: one layer keeps ~1/3 of it
    fr0, px0 = one_layer_pass(depth, K, c2w, torch.ones((3, H * W), dtype=torch.bool))  # view 1's floater: views 0 and 2 see through it... later only view 2
    square = [(y * W + x) for y in range(100, 110) for x in range(200, 210)]
    floaters = int(((fr0 == 1) & torch.isin(px0, torch.tensor(square))).sum())
    assert floaters <= 20 and len(fr0) < 1.3 * H * W, (floaters, len(fr0))
    full = torch.randint(0, 255, (3, 720, 1280, 3), dtype=torch.uint8)
    dx, dr = densify(depth, K, c2w, torch.from_numpy(p["frame"]).long()[:500], torch.from_numpy(p["pixel"]).long()[:500], full)
    assert dx.shape == (2000, 3) and dr.shape == (2000, 3) and float((dx[:, 2] - 2).abs().max()) < 1e-4
    near = torch.eye(4).repeat(2, 1, 1)
    near[1, 2, 3] = 2.  # the second view walked 2 m towards a wall 4 m away: its samples are 2x finer
    fr, _ = one_layer_pass(torch.tensor([4., 2.])[:, None, None].expand(2, H, W).contiguous(), K[:2], near, torch.ones((2, H * W), dtype=torch.bool))
    far_left, near_kept = int((fr == 0).sum()), int((fr == 1).sum())
    assert near_kept == H * W and .6 * H * W < far_left < .85 * H * W, (far_left, near_kept)  # the far view keeps only its outer ring
    keys = voxel_keys(torch.from_numpy(p["xyz"]), p["cell"])
    assert len(torch.unique(keys)) == len(keys)
    d = diagnose(depth, conf, c2w, K, 1, every=1)
    assert d["share_within_4pct"] > .95, d
    rng = np.random.default_rng(1)
    room = rng.uniform(-3, 3, (20000, 3))
    room[:, 1] = np.where(rng.random(20000) < .5, 1.6, room[:, 1])  # a floor plane at y = 1.6 (DA3: +y down)
    lb_c2w = np.repeat(np.eye(4)[None], 30, 0)
    lb_c2w[:, :3, 3] = rng.normal(size=(30, 3))
    shot = {"index": 0, "keyframes": list(range(30)), "c2w_m": [], "floor": {"normal": [0, -1., 0], "point_m": [0, 1.6, 0]}}
    for m in lb_c2w:
        g = np.eye(4)
        g[:3, 3] = 2. * m[:3, 3] + [1, 2, 3]
        shot["c2w_m"].append(g.tolist())
    rec = align_points((room - [1, 2, 3]) / 2., np.full(len(room), 5.), 1.5, lb_c2w, {f: f for f in range(30)}, shot, room)
    assert rec["use"] == "raw" and abs(rec["sim3"]["centres"]["scale_m_per_native"] - 2) < 1e-6 and rec["metrics"]["raw"]["within_25cm_all_points"] == 1, rec
    assert abs(rec["sim3"]["orientations"]["scale_m_per_native"] - 2) < 1e-6
    print("lingbot_lane self-check ok")
