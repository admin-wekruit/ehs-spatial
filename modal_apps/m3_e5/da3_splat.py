"""E5(b): a feed-forward splat, DA3-GIANT-1.1's Gaussian head (weights CC BY-NC 4.0: demo only; DA3 code Apache-2.0), on ME340
keyframes, scored on the same held-out frames and masks as modal_apps/splat_train.py.

Context: ~100 walk-shot training frames (every 8th frame is held out and never shown to the network; the cut-away shot 14-225 is
skipped). Two variants in one A100-80GB call: `posed` (DROID cameras condition the network) and `unposed` (DA3 estimates them; the
fast path's case). The Gaussians live in the network's own frame, so each held-out DROID camera is carried into it by a Sim(3)
fitted on the context cameras' centres (Umeyama); its residual is reported. Held-out renders are black where no context view saw
the scene, which the score counts. PSNR/SSIM over the pixels splat_train scores (presenter masks dilated 15 px and the caption box
left out), at the video's 1280x720 and at the network's 504x280.

  python modal_apps/m3_e5/da3_splat.py --output NEW_DIR [--views 100]
  python modal_apps/m3_e5/da3_splat.py --self-check
"""
import argparse
import io
import json
from pathlib import Path
import sys
import time

import modal
import numpy as np

if modal.is_local():  # in the container both modules sit next to this file
    sys.path[:0] = [str(Path(__file__).resolve().parent.parent), str(Path(__file__).resolve().parents[2] / "scripts")]
import splat_train as st  # noqa: E402  clip frame, K, masks and caption boxes exactly as the gsplat runs score them

MODEL, REVISION = "depth-anything/DA3-GIANT-1.1", "72ee9f89ce4e50d704e9d55ee9c646ec8dc25a19"  # mono_room.DA3_REVISIONS
CODE = "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4"  # ByteDance-Seed/Depth-Anything-3, 2026-07-27
USD_PER_S = .000694 + 4 * .0000131 + 32 * .00000222  # A100-80GB + 4 cores + 32 GiB, modal.com/pricing
image = (modal.Image.debian_slim(python_version="3.10").apt_install("git", "libgl1", "libglib2.0-0")
         .pip_install("torch==2.4.1", "torchvision==0.19.1", "xformers==0.0.28.post1", index_url="https://download.pytorch.org/whl/cu124")
         .pip_install("numpy==1.26.4", "opencv-python-headless==4.10.0.84", "addict", "packaging==24.2",
                      f"https://github.com/nerfstudio-project/gsplat/releases/download/v{st.GSPLAT}/gsplat-{st.GSPLAT}%2Bpt24cu124-cp310-cp310-linux_x86_64.whl",
                      f"git+https://github.com/ByteDance-Seed/Depth-Anything-3.git@{CODE}")
         .env({"HF_HOME": "/cache/huggingface", "HF_HUB_OFFLINE": "1"})
         .add_local_file(st.__file__, "/root/splat_train.py", copy=True).add_local_file(st.splat_to_web.__file__, "/root/splat_to_web.py", copy=True))
app = modal.App("panoptes-m3-e5-da3-splat")
volume = modal.Volume.from_name("moge3-hf-cache")  # DA3 weights mono_room.py cached


def umeyama(src, dst):
    """Similarity (s, R, t) with dst ~ s R src + t, least squares over rows."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    a, b = src - mu_s, dst - mu_d
    u, d, vt = np.linalg.svd(b.T @ a / len(src))
    e = np.eye(3)
    e[2, 2] = np.sign(np.linalg.det(u @ vt))
    R = u @ e @ vt
    s = (d * np.diag(e)).sum() / (a ** 2).sum(1).mean()
    return s, R, mu_d - s * R @ mu_s


def carried(c2w, s, R, t):
    """A camera-to-world moved by the similarity: same view, new frame (rotation R, centre s R c + t)."""
    out = np.eye(4)
    out[:3, :3], out[:3, 3] = R @ c2w[:3, :3], s * R @ c2w[:3, 3] + t
    return out


@app.function(image=image, gpu="A100-80GB", cpu=4, memory=32768, timeout=1200, retries=0, max_containers=1, volumes={"/cache": volume})
def splat_remote(video, context, held, w2c, K, excluded, show, context_excluded, nearest):
    entered = time.time()
    import cv2
    import torch
    from depth_anything_3.api import DepthAnything3
    from depth_anything_3.utils.pose_align import batch_align_poses_umeyama
    from gsplat import rasterization
    t = time.perf_counter()
    model = DepthAnything3.from_pretrained(MODEL, revision=REVISION).to("cuda").eval()
    load = time.perf_counter() - t
    Path("/tmp/v.mp4").write_bytes(video)
    cap, frames = cv2.VideoCapture("/tmp/v.mp4"), {}
    for i in range(max(context + held) + 1):
        ok, bgr = cap.read()
        assert ok
        if i in context or i in held:
            frames[i] = bgr[..., ::-1].copy()
    images = [frames[i] for i in context]
    Ks = np.repeat(np.asarray(K, np.float32)[None], len(context), 0)

    def forward(views, extrinsics, intrinsics):
        torch.cuda.synchronize()
        a = time.perf_counter()
        imgs_cpu, ex, ix = model._preprocess_inputs(views, extrinsics, intrinsics, 504, "upper_bound_resize")
        imgs, ex_t, in_t = model._prepare_model_inputs(imgs_cpu, ex, ix)
        norm = model._normalize_extrinsics(ex_t.clone()) if ex_t is not None else None
        b = time.perf_counter()
        raw = model._run_model_forward(imgs, norm, in_t, [], True, False, "saddle_balanced")
        torch.cuda.synchronize()
        pred = raw.extrinsics[0].float()
        pred = torch.cat([pred[:, :3], torch.tensor([[[0, 0, 0, 1.]]], device=pred.device).expand(len(pred), 1, 4)], 1)
        scale = 1.
        if norm is not None:  # posed: the Gaussian head rescales its cameras to the given ones (gs_adapter 1.2), so the splats do too
            scale = float(batch_align_poses_umeyama(norm.float(), pred[None])[2].clamp(1 / 3., 3.)[0])
        c2w = torch.linalg.inv(pred).double().cpu().numpy()
        c2w[:, :3, 3] *= scale
        return raw, c2w, {"preprocess_s": b - a, "forward_with_gs_head_s": time.perf_counter() - b, "hw": list(imgs.shape[-2:]), "views": len(views)}

    forward(images[:4], None, None)  # warm-up: kernels, allocator
    ssim = lambda x, y, m: float((st.ssim_map(x, y) * m).sum() / m.sum())
    droid_c2w = np.linalg.inv(np.asarray(w2c, np.float64))
    nc = len(context)
    out = {"gpu": torch.cuda.get_device_name(), "load_s": load, "variants": {}}
    variants = {"posed": (images, np.asarray(w2c, np.float32)[:nc], Ks), "unposed": (images, None, None),
                "unposed_heldout_in_input": (images + [frames[i] for i in held], None, None)}
    for name, (views, ex, ix) in variants.items():
        torch.cuda.reset_peak_memory_stats()
        raw, pred_c2w, timing = forward(views, ex, ix)
        timing["peak_gb"] = torch.cuda.max_memory_allocated() / 1e9
        g = raw.gaussians
        s, R, tr = umeyama(droid_c2w[:nc, :3, 3], pred_c2w[:nc, :3, 3])
        fit = np.linalg.norm(s * droid_c2w[:nc, :3, 3] @ R.T + tr - pred_c2w[:nc, :3, 3], axis=1)
        extent = np.linalg.norm(pred_c2w[:nc, :3, 3] - pred_c2w[:nc, :3, 3].mean(0), axis=1).max()
        rot = [np.degrees(np.arccos(np.clip((np.trace((R @ d[:3, :3]).T @ p[:3, :3]) - 1) / 2, -1, 1))) for d, p in zip(droid_c2w, pred_c2w)]
        per_view = timing["hw"][0] * timing["hw"][1]
        keep_n = nc * per_view  # the Gaussians of the context views only (they are laid out view by view)
        means, quats, scales = g.means[0, :keep_n].float(), g.rotations[0, :keep_n].float(), g.scales[0, :keep_n].float()
        opac = g.opacities[0, :keep_n].float()
        colours = g.harmonics[0, :keep_n].float().permute(0, 2, 1).contiguous()  # g xyz n -> g n xyz
        degree = int(round(colours.shape[1] ** .5)) - 1
        hw = tuple(timing["hw"])
        static = torch.cat([torch.nn.functional.interpolate(torch.tensor(np.unpackbits(m)[:frames[context[0]].shape[0] * frames[context[0]].shape[1]].reshape(
            frames[context[0]].shape[:2]), dtype=torch.float32, device="cuda")[None, None], hw, mode="area")[0, 0].ravel() == 0 for m in context_excluded])
        order = np.argsort(np.abs(np.asarray(context)[None] - np.asarray(held)[:, None]), 1, kind="stable")[:, :nearest]
        span = lambda views: torch.cat([torch.arange(v * per_view, (v + 1) * per_view, device="cuda") for v in sorted(views)])
        # every context Gaussian; without those from the presenter and caption pixels of their view (splat_train's exclusion);
        # the same, only from the `nearest` context views in time (what one held-out view can use without other views' misplaced copies)
        subsets = {"all": lambda h: None, "static": lambda h: static.nonzero()[:, 0],
                   f"static_nearest{nearest}": lambda h: (lambda idx: idx[static[idx]])(span(order[h]))}
        own = name.endswith("in_input")  # held-out views rendered from the network's own cameras for them
        rows, pngs = [], {}
        with torch.no_grad():
            for h, i in enumerate(held):
                cam = pred_c2w[nc + h] if own else carried(droid_c2w[nc + h], s, R, tr)
                view = torch.tensor(np.linalg.inv(cam), dtype=torch.float32, device="cuda")
                target = torch.tensor(frames[i], device="cuda").float() / 255
                H, W = target.shape[:2]
                keep = torch.tensor(~np.unpackbits(excluded[h])[:H * W].reshape(H, W).astype(bool), device="cuda").float()
                row = {"frame": i}
                for subset, pick in subsets.items():
                    idx = pick(h)
                    sub = [x if idx is None else x[idx] for x in (means, quats, scales, opac, colours)]
                    row[subset] = {}
                    for tag, (w, hh) in {"1280x720": (W, H), "504x280": (504, 280)}.items():
                        k = torch.tensor(K, dtype=torch.float32, device="cuda") * torch.tensor([[w / W], [hh / H], [1.]], device="cuda")
                        k[:2, 2] += .5  # OpenCV centres on integers, gsplat on halves (as splat_train.load)
                        img = rasterization(*sub, view[None], k[None], w, hh, packed=False, sh_degree=degree, rasterize_mode="classic")[0][0].clamp(0, 1)
                        small = w != W
                        tgt = torch.nn.functional.interpolate(target.permute(2, 0, 1)[None], (hh, w), mode="area")[0].permute(1, 2, 0) if small else target
                        m = (torch.nn.functional.interpolate(keep[None, None], (hh, w), mode="area")[0, 0] > .999).float() if small else keep
                        mse = (((img - tgt) ** 2).mean(-1) * m).sum() / m.sum()
                        row[subset][tag] = {"psnr": float(-10 * torch.log10(mse)), "ssim": ssim(img, tgt, m)}
                        if i in show and not small:
                            both = torch.cat([tgt, img], 1).mul(255).round().byte().cpu().numpy()
                            pngs[f"{subset}-{i:05d}"] = cv2.imencode(".jpg", both[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes()
                rows.append(row)
        out["variants"][name] = {"timing": timing, "gaussians": int(means.shape[0]), "sh_degree": degree, "per_frame": rows, "pngs": pngs,
                                 "held_out_cameras": "network's own (held-out frames were network inputs; only context Gaussians rendered)" if own
                                 else "DROID, carried into the splat frame by the context cameras' Sim(3)",
                                 "sim3_droid_to_splat": {"scale": float(s), "centre_rms_over_extent": float(np.sqrt((fit ** 2).mean()) / extent),
                                                         "rotation_deg_mean": float(np.mean(rot[:nc]))},
                                 "static_gaussians": int(static.sum()),
                                 "summary": {sub: {tag: {m: float(np.mean([r[sub][tag][m] for r in rows])) for m in ("psnr", "ssim")} for tag in ("1280x720", "504x280")}
                                             for sub in subsets}}
        del raw, g, means, quats, scales, opac, colours
        torch.cuda.empty_cache()
    out["container_s"] = time.time() - entered
    out["entered_unix"] = entered
    return out


def run(args):
    import cv2
    st.use_frame(st.frame_of(args.clip))
    clip = json.loads((args.clip / "clip.json").read_text())
    K, c2w = st.full_k(clip["K"]), np.load(args.droid_run / "prediction.npz")["poses_c2w"].astype(np.float64)
    n, video = len(c2w), args.clip / "source-full.mp4"
    skip = set(range(14, 226))
    held = [i for i in range(0, n, st.HOLD_OUT) if i not in skip]
    train = [i for i in range(n) if i % st.HOLD_OUT and i not in skip]
    context = [train[int(round(j))] for j in np.linspace(0, len(train) - 1, args.views)]
    masks = np.zeros((n, 480, 640), bool)
    for i in context + held:
        for path in args.masks.glob(f"{i:05d}-*.png"):
            masks[i] |= cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) > 0
    captions = st.caption_boxes(video, n, st.CAPTIONS)
    excluded = [np.packbits(st.excluded_full(masks[i], captions[i]).ravel()) for i in held]
    context_excluded = [np.packbits(st.excluded_full(masks[i], captions[i]).ravel()) for i in context]
    w2c = np.linalg.inv(c2w[context + held])  # context first, then held-out
    args.output.mkdir(parents=True, exist_ok=False)
    launch = {"model": MODEL, "revision": REVISION, "code": CODE, "licence": "DA3-GIANT-1.1 weights CC BY-NC 4.0 (demo only); code Apache-2.0",
              "context_frames": context, "held_out_frames": held, "views": len(context), "gpu": "A100-80GB", "retries": 0}
    (args.output / "launch.json").write_text(json.dumps(launch, indent=1))
    submitted = time.time()
    with modal.enable_output(), app.run():
        result = splat_remote.remote(video.read_bytes(), context, held, w2c, K, excluded, list(st.SHOW), context_excluded, args.nearest)
    returned = time.time()
    for name, v in result["variants"].items():
        for key, jpg in v.pop("pngs").items():
            (args.output / f"{name}-{key}.jpg").write_bytes(jpg)
    summary = {"gpu": result["gpu"], "load_s_M": round(result["load_s"], 1), "container_s_M": round(result["container_s"], 1),
               "submit_to_entry_s_M": round(result["entered_unix"] - submitted, 1), "client_wall_s_M": round(returned - submitted, 1),
               "usd_estimate": round(USD_PER_S * (result["container_s"] + 30), 3),
               "variants": {k: {"summary": v["summary"], "timing": v["timing"], "gaussians": v["gaussians"], "static_gaussians": v["static_gaussians"], "sim3_droid_to_splat": v["sim3_droid_to_splat"], "held_out_cameras": v["held_out_cameras"]} for k, v in result["variants"].items()},
               "reference": {"gsplat_60k_delivered_psnr": 30.823, "same_frames": 86, "note": "splat_train held-out protocol, 1280x720"}}
    (args.output / "result.json").write_text(json.dumps({**launch, **summary, "per_frame": {k: v["per_frame"] for k, v in result["variants"].items()}}, indent=1))
    print(json.dumps(summary, indent=1))


def self_check():
    rng = np.random.default_rng(0)
    src = rng.normal(size=(20, 3))
    R = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    R *= np.sign(np.linalg.det(R))
    s, R2, t = umeyama(src, 2.5 * src @ R.T + [1, 2, 3])
    assert abs(s - 2.5) < 1e-9 and np.allclose(R2, R) and np.allclose(t, [1, 2, 3])
    pose = np.eye(4)
    pose[:3, 3] = [1, 0, 0]
    moved = carried(pose, s, R2, t)
    assert np.allclose(moved[:3, 3], 2.5 * R @ [1, 0, 0] + [1, 2, 3]) and np.allclose(moved[:3, :3], R)
    print("da3 splat check passed: the similarity is recovered and carries cameras")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--clip", type=Path, default=st.CLIP)
    parser.add_argument("--droid-run", type=Path, default=st.DROID)
    parser.add_argument("--masks", type=Path, default=st.MASKS)
    parser.add_argument("--views", type=int, default=100)
    parser.add_argument("--nearest", type=int, default=8, help="also score each held-out view from only this many context views nearest in time")
    parser.add_argument("--output", type=Path)
    a = parser.parse_args()
    self_check() if a.self_check else run(a)
