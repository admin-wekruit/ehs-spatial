"""M3 follow-up E5b: faster gsplat previews of ME340's walk shot, scored like runs/me340-splat-232 (30.82 dB delivered).

Same data, loss and scoring as modal_apps/splat_train.py: DROID cameras with pose refinement (zero-mean knots), MCMC densification,
elongation cap 4, the presenter and caption box out of the loss; the cut-away 14-225 skipped; the 86 held-out frames (every 8th) scored
at 1280x720 on the corrected cameras. What changes, one flag each:
  --init surface|da3|lingbot   seeds: the report's surfaces (splat_train's), the DA3-GIANT any-view TSDF points of M3 E1/E7 (round 1,
                               runs/m3-exp-e1-e7-geometry-004, already in the DROID frame) or the LingBot dense map (runs/me340-lingbot-map-222)
  --schedule 4:.25,2:.5,1:1    coarse-to-fine: 1/4 size until 25% of the run, 1/2 until 50%, then full size
  --seconds 120                a wall-clock budget: learning-rate decay, densification stop and schedule follow elapsed time, not steps
  --gpu H100 --gpus 2          data parallel: each GPU renders its own training views, gradients all-reduced (NCCL) every step
Always on (--slow turns them off), and checked against splat_train's versions at the start of every call: the pose correction without
host syncs (boolean-mask mean and torch.linalg.matrix_exp each stall the CPU every step; closed-form Rodrigues instead) and a
separable SSIM window. Snapshots (--snapshots seconds) are scored after training, off the clock.

  python modal_apps/m3_e5b/fast_splat.py --prep                            # once, local CPU: inputs cache
  python modal_apps/m3_e5b/fast_splat.py --bench --gpu H100 --output DIR   # ms/step of each speed-up and size
  python modal_apps/m3_e5b/fast_splat.py --output DIR --gpu H100 --init da3 --seconds 120 --schedule 4:.25,2:.5,1:1
  python modal_apps/m3_e5b/fast_splat.py --self-check
"""
import argparse
import json
import pickle
from pathlib import Path
import sys
import time

import modal
import numpy as np

if modal.is_local():  # in the container both modules sit next to this file
    sys.path[:0] = [str(Path(__file__).resolve().parent.parent), str(Path(__file__).resolve().parents[2] / "scripts")]
import splat_train as st  # noqa: E402

RUNS = st.ART / "runs"
DA3 = RUNS / "m3-exp-e1-e7-geometry-004/giant-504_cam-tsdf-points.npz"  # DA3-GIANT any-view, 112 keyframes, 3 cm TSDF, metres, DROID frame
DA3_VOXEL_M = .03
LINGBOT = RUNS / "me340-lingbot-map-222"
CACHE = RUNS / "m3-fu-e5b-splat-inputs/inputs.pkl"
SKIP = range(14, 226)  # ME340's cut-away shot, as runs/me340-splat-232
TICK = 25  # steps between clock checks (and, on two GPUs, one broadcast of rank 0's clock)
SNAPSHOTS = (30, 60, 90, 120, 150, 180, 240, 300, 420, 600, 900, 1200, 1500, 1800, 2400)
USD_PER_S = {"H100": .001097, "A100-80GB": .000694, "cpu_core": .0000131, "memory_gib": .00000222}  # modal.com/pricing
CPU_PER_GPU, MEMORY_MIB = 4, 32768

image = st.image.add_local_file(st.__file__, "/root/splat_train.py")
app = modal.App("panoptes-m3fu-e5b-splat")


# ---------------------------------------------------------------- GPU side
def pose_of(knots, weights, i):
    """splat_train.timeline for one frame without host syncs: the knots' mean over the trained frames' knots is a weighted sum."""
    k = knots - (knots * weights).sum(0)
    j, w = divmod(int(i), st.KNOT_EVERY)
    w /= st.KNOT_EVERY
    return (1 - w) * k[j] + w * k[j + 1]


def viewmat_fast(c2w, pose):
    """splat_train.viewmat with Rodrigues' formula for the rotation (matrix_exp picks its series degree on the host)."""
    import torch
    w, z = pose[:3], torch.zeros((), device=c2w.device)
    skew = torch.stack([torch.stack([z, -w[2], w[1]]), torch.stack([w[2], z, -w[0]]), torch.stack([-w[1], w[0], z])])
    t = torch.sqrt((w * w).sum() + 1e-12)
    rot = torch.eye(3, device=c2w.device) + torch.sin(t) / t * skew + 2 * torch.sin(t / 2) ** 2 / t ** 2 * skew @ skew
    R, c = rot @ c2w[:3, :3], c2w[:3, 3] + pose[3:]
    out = torch.eye(4, device=c2w.device)
    out[:3, :3], out[:3, 3] = R.T, -R.T @ c
    return out


def ssim_fast(a, b, size=11, sigma=1.5):
    """splat_train.ssim_map with the Gaussian window applied as two 1-D passes over all five maps at once."""
    import torch
    import torch.nn.functional as F
    g = torch.exp(-(torch.arange(size, device=a.device) - size // 2) ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    x, y = a.permute(2, 0, 1)[None], b.permute(2, 0, 1)[None]
    maps = torch.cat([x, y, x * x, y * y, x * y], 1)
    maps = F.conv2d(maps, g.view(1, 1, 1, size).expand(15, 1, 1, size), padding=(0, size // 2), groups=15)
    mx, my, xx, yy, xy = F.conv2d(maps, g.view(1, 1, size, 1).expand(15, 1, size, 1), padding=(size // 2, 0), groups=15).split(3, 1)
    sxx, syy, sxy = xx - mx ** 2, yy - my ** 2, xy - mx * my
    c1, c2 = .01 ** 2, .03 ** 2
    return ((2 * mx * my + c1) * (2 * sxy + c2) / ((mx ** 2 + my ** 2 + c1) * (sxx + syy + c2)))[0].mean(0)


def check_equivalence(c2w):
    """The sync-free pose and the separable SSIM agree with splat_train's."""
    import torch
    torch.manual_seed(1)
    knots, used = torch.randn(12, 6, device="cuda") * 1e-2, torch.tensor([1, 1, 0, 1] * 3, dtype=torch.bool, device="cuda")
    for i in (9, 40, 87):
        p = pose_of(knots, (used.float() / used.sum())[:, None], i)
        assert torch.allclose(p, st.timeline(knots, torch.tensor([i], device="cuda"), used=used)[0], atol=1e-7), "pose timeline"
        assert torch.allclose(viewmat_fast(c2w, p), st.viewmat(c2w, p), atol=2e-6), "Rodrigues = matrix_exp"
    assert torch.allclose(viewmat_fast(c2w, torch.zeros(6, device="cuda")), st.viewmat(c2w), atol=1e-6), "zero correction"
    a, b = torch.rand(72, 128, 3, device="cuda"), torch.rand(72, 128, 3, device="cuda")
    assert (ssim_fast(a, b) - st.ssim_map(a, b)).abs().max() < 1e-5, "separable SSIM"


def fit(cfg, data, rank=0, world=1):
    """splat_train.fit's training with seeds, schedule, clock and GPU count from cfg; returns params, knots, stats, snapshots."""
    import torch
    import torch.distributed as dist
    import torch.nn.functional as F
    from gsplat import MCMCStrategy, rasterization
    from gsplat.optimizers import SelectiveAdam
    from scipy.spatial import cKDTree
    torch.manual_seed(0)  # every rank draws the same random numbers: backgrounds, MCMC relocation and noise stay identical
    started_init = time.time()
    xyz, rgb, scale = data["xyz"], data["rgb"], data["scale"]
    if scale is None:  # gsplat's MCMC recipe, as splat_train: a tenth of the RMS distance to 3 neighbours
        scale = .1 * np.sqrt((cKDTree(xyz).query(xyz, k=4)[0][:, 1:] ** 2).mean(1)).clip(1e-4)
    tensors = {"means": torch.tensor(xyz), "scales": torch.log(torch.tensor(scale, dtype=torch.float32))[:, None].repeat(1, 3),
               "quats": torch.rand(len(xyz), 4), "opacities": torch.logit(torch.full((len(xyz),), .5)), "colors": torch.logit(torch.tensor(rgb).clamp(.02, .98))}
    params = torch.nn.ParameterDict({k: torch.nn.Parameter(v.float().cuda()) for k, v in tensors.items()})
    n_knots = len(data["c2w"]) // st.KNOT_EVERY + 2
    knots = torch.zeros(n_knots, 6, device="cuda", requires_grad=cfg["pose"])
    trained = torch.tensor(data["train"], device="cuda") // st.KNOT_EVERY
    used = torch.zeros(n_knots, dtype=torch.bool, device="cuda").index_fill_(0, torch.cat([trained, trained + 1]), True)
    weights = (used.float() / used.sum())[:, None]
    centres = data["c2w"][data["train"], :3, 3]
    lr = {**st.LR, "means": st.LR["means"] * 1.1 * float((centres - centres.mean(0)).norm(dim=1).max())}
    adam = SelectiveAdam if cfg.get("selective_adam") else torch.optim.Adam
    optimizers = {k: adam([{"params": params[k], "lr": lr[k], "name": k}], eps=1e-15, betas=(.9, .999)) for k in params}
    refine = torch.optim.Adam([{"params": [knots], "lr": lr["pose"], "name": "pose"}]) if cfg["pose"] else None
    budget, steps = cfg.get("seconds"), cfg.get("steps")
    # per rendered view as in splat_train (refine from view 500, every 100 views, until 5/6 of the run); a step renders `world` x `batch` views
    per_step = world * cfg.get("batch", 1)
    strategy = MCMCStrategy(cap_max=cfg["cap"], refine_start_iter=cfg.get("refine_start", 500) // per_step, refine_every=100 // per_step,
                            refine_stop_iter=10 ** 9 if budget else steps * 5 // 6)
    strategy.check_sanity(params, optimizers)
    state = strategy.initialize_state()
    frames, excluded, K, c2w = data["frames"], data["excluded"], data["K"], data["c2w"]
    Ks = {s: torch.cat([K[:2] / s, K[2:]]) for s, _ in cfg["schedule"]}  # gsplat's +0.5 pixel centres scale with the pixels
    mine, rng, order = data["train"][rank::world], np.random.default_rng(rank), []
    view_of = viewmat_fast if cfg["fast"] else st.viewmat
    ssim_of = ssim_fast if cfg["fast"] else st.ssim_map
    wanted, snaps, losses, per_scale = [t for t in cfg.get("snapshots", ()) if not budget or t < budget], [], [], {}
    init_seconds = time.time() - started_init
    if world > 1:
        dist.barrier()
    torch.cuda.synchronize()
    started = time.time()
    step, progress, timed = 0, 0., None
    while True:
        if step % TICK == 0:
            elapsed = time.time() - started
            if world > 1 and budget:  # every rank stops, stops densifying and changes size on the same step
                clock = torch.tensor([elapsed], device="cuda")
                dist.broadcast(clock, 0)
                elapsed = float(clock)
            while rank == 0 and wanted and elapsed >= wanted[0]:
                snaps.append({"at_s": wanted.pop(0), "step": step, "elapsed_s": round(elapsed, 1),
                              "params": {k: v.detach().clone() for k, v in params.items()}, "knots": knots.detach().clone()})
            if budget:
                progress = elapsed / budget
                if progress >= 5 / 6:
                    strategy.refine_stop_iter = min(strategy.refine_stop_iter, step)
        if not budget:
            progress = step / steps
        if progress >= 1:
            break
        if step == cfg.get("time_from"):
            torch.cuda.synchronize()
            timed = time.time()
        small = next(s for s, until in cfg["schedule"] if progress < until)
        picks = []
        for _ in range(cfg.get("batch", 1)):  # views rendered together in one rasterization call on this GPU
            if not order:
                order = [mine[j] for j in rng.permutation(len(mine))]
            picks.append(order.pop())
        decay = .01 ** progress
        for group in optimizers["means"].param_groups + (refine.param_groups if refine else []):
            group["lr"] = lr[group["name"]] * decay
        pose = lambda i: pose_of(knots, weights, i) if cfg["fast"] else st.timeline(knots, torch.tensor([i], device="cuda"), used=used)[0]
        views = torch.stack([view_of(c2w[i], pose(i)) if cfg["pose"] else st.viewmat(c2w[i]) for i in picks])
        m = st.activated(params)
        colours, alphas, info = rasterization(m["means"], m["quats"], m["scales"], m["opacities"], m["colors"], views, Ks[small][None].expand(len(picks), 3, 3),
                                              st.W // small, st.H // small, packed=cfg.get("packed", False), rasterize_mode="classic",
                                              tile_size=16 // small if cfg.get("small_tiles") else 16)
        loss = .01 * m["opacities"].mean() + .01 * m["scales"].mean()
        for b, i in enumerate(picks):
            # the random background composited here as gsplat's `backgrounds` does (colour + (1 - alpha) bg): packed mode rejects it in 1.5.3
            image = colours[b] + (1 - alphas[b]) * torch.rand(3, device="cuda")
            target, valid = frames[i].float() / 255, (~excluded[i]).float()
            if small > 1:  # area-downsampled target; a block with any excluded pixel is excluded (round-1 E5's --train-scale)
                target = F.avg_pool2d(target.permute(2, 0, 1)[None], small)[0].permute(1, 2, 0)
                valid = -F.max_pool2d(-valid[None, None], small)[0, 0]
            l1 = ((image - target).abs().mean(-1) * valid).sum() / valid.sum()
            ssim = (ssim_of(image, target) * valid).sum() / valid.sum()
            loss = loss + (.8 * l1 + .2 * (1 - ssim)) / len(picks)
        loss.backward()
        if world > 1:  # data parallel: the mean gradient of every rank's view, so all ranks take the same Adam step
            grads = [p.grad for p in params.values()] + ([knots.grad] if cfg["pose"] else [])
            flat = torch.cat([g.reshape(-1) for g in grads])
            dist.all_reduce(flat)
            flat /= world
            for g, part in zip(grads, flat.split([g.numel() for g in grads])):
                g.copy_(part.view_as(g))
        visible = (info["radii"] > 0).all(-1).any(0) if cfg.get("selective_adam") else None
        for optimizer in optimizers.values():
            optimizer.step(visible) if visible is not None else optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        if refine:
            refine.step()
            refine.zero_grad(set_to_none=True)
        if cfg.get("max_elongation"):
            with torch.no_grad():
                ordered, axes = params["scales"].sort(1)
                params["scales"].scatter_(1, axes[:, 2:], torch.minimum(ordered[:, 2:], ordered[:, 1:2] + np.log(cfg["max_elongation"])))
        strategy.step_post_backward(params, optimizers, state, step, info, lr=lr["means"] * decay)
        losses.append(loss.detach())
        per_scale[small] = per_scale.get(small, 0) + 1
        step += 1
    torch.cuda.synchronize()
    ended = time.time()
    stats = {"steps_done": step, "views_rendered": step * per_step, "train_seconds": round(ended - started, 2), "init_seconds": round(init_seconds, 2),
             "steps_per_scale": {str(k): v for k, v in per_scale.items()}, "gaussians": len(params["means"]), "seeds": len(xyz),
             "loss_per_500_steps": [round(float(torch.stack(losses[k:k + 500]).mean()), 4) for k in range(0, len(losses), 500)]}
    if timed is not None:
        stats["ms_per_step_after_warmup"] = round(1000 * (ended - timed) / (step - cfg["time_from"]), 2)
    if world > 1:  # the ranks must hold the same model: largest difference of any Gaussian position across ranks
        gathered = [torch.empty_like(params["means"]) for _ in range(world)]
        dist.all_gather(gathered, params["means"].detach().contiguous())
        stats["rank_max_abs_diff_means"] = float(max((g - gathered[0]).abs().max() for g in gathered))
    return params, {"pose": knots.detach(), "used": used}, stats, snaps


def to_gpu(shared):
    import torch
    cuda = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32, device="cuda")
    return {"frames": shared["frames"].cuda(), "excluded": shared["excluded"].cuda(), "K": cuda(shared["K"]), "c2w": cuda(shared["c2w"]),
            "xyz": shared["xyz"], "rgb": shared["rgb"], "scale": shared["scale"], "train": shared["train"], "held": shared["held"], "depth": None}


def worker(rank, world, shared, cfg):
    """One GPU's process; rank 0 scores and writes /tmp/result.pkl."""
    import lpips
    import torch
    import torch.distributed as dist
    torch.cuda.set_device(rank)
    if world > 1:
        dist.init_process_group("nccl", init_method="tcp://127.0.0.1:29533", rank=rank, world_size=world)
    t = time.time()
    data = to_gpu(shared)
    check_equivalence(data["c2w"][shared["train"][0]])
    out = {"to_gpu_seconds": round(time.time() - t, 2), "gpu": torch.cuda.get_device_name(), "gpus": world, "runs": {}}
    if world > 1:
        out["peer_access"] = torch.cuda.can_device_access_peer(0, 1)
    for name, variant in (cfg["bench"].items() if cfg.get("bench") else [("final", {})]):
        c = {**cfg, **variant}
        params, knots, stats, snaps = fit(c, data, rank, world)
        if rank == 0 and not cfg.get("bench"):
            t = time.time()
            net = lpips.LPIPS(net="alex", spatial=True, verbose=False).cuda().eval()
            scoring = {"pose": c["pose"], "exposure": False}
            stats["trained_model"] = st.evaluate(st.activated(params), knots, scoring, data, net)[0]["summary"]
            stats["snapshots"] = [{k: v for k, v in s.items() if k not in ("params", "knots")} |
                                  {"held_out": st.evaluate(st.activated(s["params"]), {"pose": s["knots"], "used": knots["used"]}, scoring, data, net)[0]["summary"]}
                                  for s in snaps]
            records, stats["pruning"] = st.export(params, {"cap": max(c["cap"], len(params["means"]))}, data)
            as_file = st.as_model(records.tobytes())
            scored, pngs = st.evaluate(as_file, knots, scoring, data, net, st.SHOW)
            stats["exported_file"] = scored["summary"]
            stats["files"] = {"pngs": pngs, "splat": records.tobytes() if c.get("keep_splat") else None}
            stats["eval_seconds"] = round(time.time() - t, 1)
        out["runs"][name] = stats
        del params
        torch.cuda.empty_cache()
    if rank == 0:
        Path("/tmp/result.pkl").write_bytes(pickle.dumps(out))
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


@app.function(image=image, gpu="H100", cpu=CPU_PER_GPU, memory=MEMORY_MIB, timeout=3600, retries=0, max_containers=6, scaledown_window=2,
              volumes={"/ckpt": st.volume})  # /ckpt/torch: the LPIPS backbone splat_train cached
def train_remote(inputs, cfg):
    """Decode once on the CPU (the parent never touches CUDA), then one forked process per GPU."""
    import torch
    import torch.multiprocessing as mp
    started = time.time()
    st.use_frame(inputs["frame"])
    Path("/tmp/video.mp4").write_bytes(inputs["video"])
    import cv2
    n, capture = len(inputs["c2w"]), cv2.VideoCapture("/tmp/video.mp4")
    frames, excluded = torch.empty((n, st.H, st.W, 3), dtype=torch.uint8), torch.empty((n, st.H, st.W), dtype=torch.bool)
    masks = np.unpackbits(inputs["masks"], axis=1)[:, :480 * 640].reshape(n, 480, 640).astype(bool)
    for i in range(n):
        ok, bgr = capture.read()
        assert ok and bgr.shape == (st.H, st.W, 3), f"video frame {i} missing or not {st.W}x{st.H}"
        frames[i] = torch.from_numpy(np.ascontiguousarray(bgr[..., ::-1]))
        excluded[i] = torch.from_numpy(st.excluded_full(masks[i], inputs["captions"][i]))
    shared = {k: inputs[k] for k in ("K", "c2w", "xyz", "rgb", "scale", "train", "held")} | {"frames": frames, "excluded": excluded}
    shared["K"] = np.asarray(inputs["K"]) + [[0, 0, .5], [0, 0, .5], [0, 0, 0]]  # gsplat's pixel centres, as splat_train.load
    decode_seconds = time.time() - started
    world = cfg["gpus"]
    if world == 1:
        worker(0, 1, shared, cfg)
    else:
        mp.start_processes(worker, args=(world, shared, cfg), nprocs=world, start_method="fork", join=True)
    out = pickle.loads(Path("/tmp/result.pkl").read_bytes())
    out |= {"decode_seconds": round(decode_seconds, 1), "container_seconds": round(time.time() - started, 1)}
    return out


# ---------------------------------------------------------------- local side
def prep(cache):
    """Everything a call needs, as splat_train.run builds it for ME340 (--skip 14-225): video, masks, caption boxes, cameras, frame
    split, and the three seed sets (report surfaces as splat_train's seed_points, DA3 TSDF points in native units, LingBot map
    without its held-out and skipped source frames)."""
    import cv2
    import open3d as o3d
    st.use_frame(st.frame_of(st.CLIP))
    clip = json.loads((st.CLIP / "clip.json").read_text())
    K, c2w = st.full_k(clip["K"]), np.load(st.DROID / "prediction.npz")["poses_c2w"].astype(np.float64)
    n, video = len(c2w), st.CLIP / "source-full.mp4"
    held, train = [i for i in range(0, n, st.HOLD_OUT) if i not in SKIP], [i for i in range(n) if i % st.HOLD_OUT and i not in SKIP]
    masks = np.zeros((n, 480, 640), bool)
    for i in range(n):
        for path in st.MASKS.glob(f"{i:05d}-*.png"):
            masks[i] |= cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) > 0
    captions = st.caption_boxes(video, n, st.CAPTIONS)
    surface = st.seed_points(video, K, c2w, train[::16], lambda i: st.excluded_full(masks[i], captions[i]))
    metres = json.loads(st.SCALE.read_text())["metres_per_native_unit"]
    da3 = np.load(DA3)
    assert abs(float(da3["metres_per_native"]) - metres) < 1e-6, "DA3 points use the report's metric scale"
    da3_xyz = (da3["points_m_droid_frame"] / metres).astype(np.float32)
    lingbot = st.lingbot_map(LINGBOT, set(held) | set(SKIP))
    scene = st.report_surfaces()[2]
    near = lambda p: float(np.median(scene.compute_distance(o3d.core.Tensor(p[::max(1, len(p) // 50000)].astype(np.float32))).numpy())) * metres
    out = {"video": video.read_bytes(), "masks": np.packbits(masks.reshape(n, -1), axis=1), "captions": captions, "K": K, "c2w": c2w, "train": train,
           "held": held, "skip": list(SKIP), "metres": metres, "frame": st.frame_now(),
           "surface": surface[:2], "da3": (da3_xyz, da3["colors"].astype(np.float32) / 255), "lingbot": lingbot,
           "meta": {"surface_seeds": surface[2], "da3_points": len(da3_xyz), "da3_keyframes": da3["keyframes"].tolist(), "lingbot_points": len(lingbot[0]),
                    "median_distance_to_report_surfaces_m": {"da3": near(da3_xyz), "lingbot": near(lingbot[0]), "surface": near(surface[0])}}}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(pickle.dumps(out))
    print(json.dumps(out["meta"] | {"da3_keyframes": len(out["meta"]["da3_keyframes"]), "held_out": len(held), "trained": len(train)}))


def seeds(cache, init, voxel, share):
    """(xyz, rgb, scale or None): surface as splat_train (scale from neighbours); dense points one per voxel, scale a share of
    max(sample size, voxel)."""
    if init == "surface":
        return (*cache["surface"], None)
    xyz, rgb, size = cache["lingbot"] if init == "lingbot" else (*cache["da3"], np.full(len(cache["da3"][0]), DA3_VOXEL_M / cache["metres"], np.float32))
    return st.voxel_seeds(xyz, rgb, size, voxel, share)


def parse_schedule(text):
    """'4:.25,2:.5,1:1' -> [(4, .25), (2, .5), (1, 1.)]: train size 1/4 until a quarter of the run, ... ; must end at 1:1."""
    out = [(int(a), float(b)) for a, b in (part.split(":") for part in text.split(","))]
    assert out[-1] == (1, 1.) and all(x[1] < y[1] for x, y in zip(out, out[1:])), "increasing fractions, ending at full size 1:1"
    return out


def usd(gpu, gpus, seconds):
    return seconds * (gpus * USD_PER_S[gpu] + gpus * CPU_PER_GPU * USD_PER_S["cpu_core"] + MEMORY_MIB / 1024 * USD_PER_S["memory_gib"])


def run(args):
    cache = pickle.loads(args.cache.read_bytes())
    st.use_frame(cache["frame"])
    xyz, rgb, scale = seeds(cache, args.init, args.voxel, args.share)
    inputs = {k: cache[k] for k in ("video", "masks", "captions", "K", "c2w", "train", "held", "frame")} | {"xyz": xyz, "rgb": rgb, "scale": scale}
    cfg = {"gpus": args.gpus, "cap": max(args.cap, len(xyz)), "pose": not args.no_pose, "max_elongation": 4., "fast": not args.slow, "packed": args.packed,
           "selective_adam": args.selective_adam, "small_tiles": args.small_tiles, "batch": args.batch,
           "schedule": parse_schedule(args.schedule), "seconds": args.seconds, "steps": args.steps, "snapshots": args.snapshots, "keep_splat": args.keep_splat}
    if args.bench:  # same seeds each time, no densification (fixed count), 60 warm-up steps then timed
        base = {"steps": 360, "seconds": None, "time_from": 60, "refine_start": 10 ** 9, "snapshots": ()}
        full = {**base, "schedule": [(1, 1.)]}
        cfg["bench"] = {"splat_train_step": {**full, "fast": False}, "fast_pose_ssim": full, "fast_packed": {**full, "packed": True},
                        "fast_selective_adam": {**full, "selective_adam": True}, "fast_no_pose": {**full, "pose": False},
                        "fast_half_size": {**base, "schedule": [(2, 1.)]}, "fast_quarter_size": {**base, "schedule": [(4, 1.)]},
                        "fast_half_size_tile8": {**base, "schedule": [(2, 1.)], "small_tiles": True},
                        "fast_quarter_size_tile4": {**base, "schedule": [(4, 1.)], "small_tiles": True},
                        "fast_selective_adam_quarter_tile4": {**base, "schedule": [(4, 1.)], "small_tiles": True, "selective_adam": True},
                        "fast_batch2": {**full, "batch": 2}, "fast_batch4": {**full, "batch": 4}, "fast_selective_adam_batch2": {**full, "batch": 2, "selective_adam": True}}
    assert args.seconds or args.steps or args.bench, "--seconds or --steps"
    args.output.mkdir(parents=True, exist_ok=False)
    launch = {"init": args.init, "voxel_native": args.voxel, "share": args.share, "seeds": len(xyz), "gpu": args.gpu, "cfg": cfg, "prep": cache["meta"],
              "held_out_frames": len(cache["held"]), "trained_frames": len(cache["train"])}
    (args.output / "launch.json").write_text(json.dumps(launch, indent=1, default=str))
    started = time.time()
    with modal.enable_output(), app.run():
        result = train_remote.with_options(gpu=f"{args.gpu}:{args.gpus}" if args.gpus > 1 else args.gpu, cpu=CPU_PER_GPU * args.gpus,
                                           timeout=int(60 * args.max_minutes)).remote(inputs, cfg)
    call_seconds = time.time() - started
    for name, r in result["runs"].items():
        for frame, png in (r.get("files") or {}).get("pngs", {}).items():
            (args.output / f"{name}-heldout-{frame:05d}.png").write_bytes(png)
        splat = (r.pop("files", None) or {}).get("splat")
        if splat:
            (args.output / f"{name}.splat").write_bytes(splat)
    kind = "H100" if "H100" in result["gpu"] else "A100-80GB"
    result |= {"call_seconds": round(call_seconds, 1), "cold_start_and_transfer_s": round(call_seconds - result["container_seconds"], 1),
               "usd_estimate": round(usd(kind, args.gpus, result["container_seconds"] + 60), 3)}
    (args.output / "result.json").write_text(json.dumps(launch | {"result": result}, indent=1, default=str))
    brief = {n: {k: r.get(k) for k in ("train_seconds", "steps_done", "gaussians", "ms_per_step_after_warmup", "rank_max_abs_diff_means")} |
             {"psnr": (r.get("exported_file") or {}).get("corrected_pose", {}).get("psnr") if not args.no_pose else (r.get("exported_file") or {}).get("report_pose", {}).get("psnr"),
              "snapshots": [(s["at_s"], s["held_out"].get("corrected_pose", s["held_out"].get("report_pose"))["psnr"]) for s in r.get("snapshots", [])]}
             for n, r in result["runs"].items()}
    print(json.dumps({"gpu": result["gpu"], "gpus": args.gpus, "decode_s": result["decode_seconds"], "container_s": result["container_seconds"],
                      "call_s": round(call_seconds), "usd": result["usd_estimate"], "runs": brief}, indent=1))


def self_check():
    assert parse_schedule("4:.25,2:.5,1:1") == [(4, .25), (2, .5), (1, 1.)] and parse_schedule("1:1") == [(1, 1.)]
    for bad in ("2:.5", "1:1,2:.5"):
        try:
            parse_schedule(bad)
        except AssertionError:
            continue
        raise AssertionError(f"{bad} accepted")
    cache = {"surface": (np.zeros((2, 3), np.float32), np.zeros((2, 3), np.float32)), "metres": 3.,
             "da3": (np.array([[0, 0, 0], [.001, 0, 0], [.05, 0, 0]], np.float32), np.eye(3, dtype=np.float32))}
    xyz, rgb, scale = seeds(cache, "da3", .01, .5)
    assert len(xyz) == 2 and np.allclose(scale, .5 * .01), "one DA3 seed per voxel, scale half of max(TSDF voxel in native = .01, voxel)"
    assert seeds(cache, "surface", .01, .5)[2] is None, "surface seeds keep splat_train's neighbour scale"
    assert abs(usd("H100", 2, 100) - 100 * (2 * .001097 + 8 * .0000131 + 32 * .00000222)) < 1e-9
    print("fast_splat check passed (GPU equivalence of the sync-free pose and separable SSIM runs at the start of every call)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--prep", action="store_true", help="build the inputs cache (local CPU)")
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--bench", action="store_true", help="ms/step of splat_train's step and each speed-up (fixed Gaussian count)")
    parser.add_argument("--init", choices=("surface", "da3", "lingbot"), default="surface")
    parser.add_argument("--voxel", type=float, default=.01, help="dense seeds: one per voxel of this size (native units)")
    parser.add_argument("--share", type=float, default=.5, help="dense seeds: scale as a share of max(sample size, voxel)")
    parser.add_argument("--cap", type=int, default=500000, help="most Gaussians (raised to the seed count if below it)")
    parser.add_argument("--schedule", default="1:1", help="coarse-to-fine sizes and until which share of the run, e.g. 4:.25,2:.5,1:1")
    parser.add_argument("--seconds", type=float, help="wall-clock training budget (schedules follow elapsed time)")
    parser.add_argument("--steps", type=int, help="or a step count (each step renders one view per GPU)")
    parser.add_argument("--snapshots", type=float, nargs="*", default=list(SNAPSHOTS), help="also score the model at these training seconds")
    parser.add_argument("--gpu", default="H100", choices=("H100", "A100-80GB"))
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--batch", type=int, default=1, help="views per step on each GPU, rendered in one call (loss averaged)")
    parser.add_argument("--no-pose", action="store_true")
    parser.add_argument("--packed", action="store_true")
    parser.add_argument("--selective-adam", action="store_true", help="Adam steps only the Gaussians the view sees (gsplat SelectiveAdam)")
    parser.add_argument("--small-tiles", action="store_true", help="at 1/s size rasterize with 16/s px tiles (same pixels per tile as full size)")
    parser.add_argument("--slow", action="store_true", help="splat_train's pose and SSIM (host syncs, 2-D window)")
    parser.add_argument("--keep-splat", action="store_true", help="write the exported splat32 file")
    parser.add_argument("--max-minutes", type=float, default=20)
    parser.add_argument("--output", type=Path)
    a = parser.parse_args()
    self_check() if a.self_check else prep(a.cache) if a.prep else run(a)
