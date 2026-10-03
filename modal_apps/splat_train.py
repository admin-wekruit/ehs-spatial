"""Gaussian splats of a video report's static scene (gsplat, Apache-2.0) on Modal, in the report's world frame, for the web viewer.

The report's room is a TSDF of monocular depth textured from 640x480 crops, so it looks soft. This fits 3D Gaussians to the uncropped
video frames (the clip's source-full.mp4, whatever its size: source-full.json places the 640x480 raster in it) seen from the report's own cameras (DROID poses_c2w with the video-frame intrinsics), seeded from the report's own
surfaces, so the splats share the world of its meshes and cameras. Kept out of the loss and of every metric: the presenter (the dynamic
masks, which cover only the centre crop, mapped to the video frame and dilated) and each frame's burned-in caption box. Optional refinement,
linear in time between knots every 8 frames and zero-mean over the clip so the world frame cannot drift: camera pose (a rotation about the
camera centre and a world translation) and exposure (a 3x3 colour matrix and an offset on the render). Every 8th frame is held out and never
trained on; it takes pose and exposure from the knots around it, which only training frames set. MCMC densification up to a cap, one base
colour per Gaussian (the web format has no view-dependent colour), classic rasterization with the 0.3 px low-pass the viewer draws with.
For views away from the walked path: an optional cap on elongation (needles hide end-on from the path and streak from anywhere else) and
an optional clipped depth pull onto the report's surfaces. With --lingbot, a LingBot dense point map (lingbot_dense_map.py, same world
frame) replaces those surfaces: its points (one per voxel, scaled to their sample size) are the seeds, and the depth pull is towards the
map drawn from each training camera. One GPU call, retries 0, an absolute deadline; checkpoints on a Modal volume, and a restarted call
resumes from them. --clean scores cleanup rules on a finished run (held-out metrics, fixed views), --pick writes one; --compare renders
splat files side by side in the same fixed views (three orbits like the report's default view, three short drags off the path).

  python modal_apps/splat_train.py --output NEW_DIR --steps 500 --cap 300000 --pose --exposure       # smoke: every code path, minutes
  python modal_apps/splat_train.py --output NEW_DIR --steps 7000 --cap 600000 --ablation             # plain, +exposure, +exposure+pose
  python modal_apps/splat_train.py --output NEW_DIR --steps 60000 --cap 2500000 --pose              # best held-out metrics on ME340
  python modal_apps/splat_train.py --output NEW_DIR --steps 40000 --cap 2500000 --pose --max-elongation 4 --depth-weight .1
  python modal_apps/splat_train.py --output NEW_DIR --steps 60000 --cap 2500000 --pose --max-elongation 4 --depth-weight .1 --lingbot MAP_DIR
  python modal_apps/splat_train.py --output NEW_DIR --steps 60000 --cap 2500000 --pose --resume RUN  # evaluate and export a run again
  python modal_apps/splat_train.py --clean RUN_DIR [--pick RULE]                                     # cleanup rules, then splats-clean.*
  python modal_apps/splat_train.py --compare A=a.splat B=b.splat --output DIR                        # side-by-side sheets
  python modal_apps/splat_train.py --self-check
"""
import argparse
import json
import pickle
from pathlib import Path
import sys
import time

import modal
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import splat_to_web  # noqa: E402  the viewer's byte format

ART = Path("/Users/adam/Desktop/panoptes-public/research-notes/phase2")
CLIP, DROID = ART / "data/clips/me340-165", ART / "runs/droid-me340-165-171"
MESH = ART / "runs/da3-posed-me340-189-fused-dynamic/mono-anchored-mesh.ply"
FILL = ART / "runs/me340-filled-213/textured-scene.glb"  # the report's room: that mesh plus its single-view-depth fill
MASKS = ART / "runs/me340-dynamic-masks-188/masks"
SCALE = ART / "runs/da3-posed-me340-189-fused-dynamic/metric-scale.json"
FRAME_ID = "droid_final_native_world"
# the video frame and where the clip's 640x480 raster sits in it (source-full.json raster_in_video_xywh): ME340's 1280x720 with the
# raster its centre 960x720 (x1.5); use_frame(frame_of(clip)) sets them for another clip, and every remote call gets its clip's frame
W, H, SIDE, TOP, ZOOM = 1280, 720, 160, 0, 1.5
CAPTIONS = (648, 704, 160, 1120)  # y0 y1 x0 x1 band of ME340's burned-in captions (text rows 661-689, one centred line, widest x 228-1050)
HOLD_OUT, KNOT_EVERY, DILATE = 8, 8, 15
HELD_OUT_NOTE = ("validation, not an untouched test: the settings (Gaussian cap, steps, pose refinement, elongation cap, cleanup rule) were "
                 "chosen on these frames and the seed surface was fused from views that include them; each is one frame from a "
                 "training frame, so it scores the walked path, not views away from it")
OFF = {"far": 1e9, "min_views": 0, "needle_ratio": 1e9, "needle_opacity": 0, "huge": 1e9, "huge_opacity": 0}  # a cleanup rule that drops nothing
SHOW = (240, 400, 560, 784)  # held-out frames saved as real | render (ME340 frames 14-225 are another shot)
# ME340's benches from a short drag off the walked path: clip frame, a bench pixel in its video frame, that pixel's depth in the LingBot
# map (native), eye offset from that camera right/up/back (native, 0.14-0.18 in all)
MODERATE = ((240, (560, 450), .566, (.12, .08, 0.)), (520, (613, 453), .560, (-.15, .10, 0.)), (760, (960, 420), .812, (-.08, .08, .10)))
LR = {"means": 1.6e-4, "scales": 5e-3, "quats": 1e-3, "opacities": 5e-2, "colors": 2.5e-3, "pose": 2e-5, "exposure": 5e-4}  # means x scene scale
GSPLAT = "1.5.3"
GPUS = ["H100", "A100-80GB", "A100-40GB"]  # H100 first: 0.022 s/step at 2.5M Gaussians against the A100-80GB's 0.051, cheaper per step
CPU, MEMORY_MIB = 2.0, 12288
USD_PER_S = {"H100": .001097, "A100-80GB": .000694, "A100-40GB": .000583, "cpu_core": .0000131, "memory_gib": .00000222}  # modal.com/pricing

image = (modal.Image.debian_slim(python_version="3.10")
         .pip_install("torch==2.4.1", "torchvision==0.19.1", index_url="https://download.pytorch.org/whl/cu124")
         .pip_install("numpy==1.26.4", "opencv-python-headless==4.10.0.84", "scipy==1.13.1", "lpips==0.1.4", "packaging==24.2",  # prebuilt kernels: no CUDA compile
                      f"https://github.com/nerfstudio-project/gsplat/releases/download/v{GSPLAT}/gsplat-{GSPLAT}%2Bpt24cu124-cp310-cp310-linux_x86_64.whl")
         .env({"TORCH_HOME": "/ckpt/torch"})
         .add_local_python_source("splat_to_web"))
app = modal.App("panoptes-splat-train-once")
volume = modal.Volume.from_name("panoptes-splat-train", create_if_missing=True)


def frame_of(clip):
    """The clip's video frame (source-full.json): its size and where the 640x480 raster sits in it (one zoom for both axes)."""
    full = json.loads((Path(clip) / "source-full.json").read_text())
    x, y, w, h = full["raster_in_video_xywh"]
    assert abs(w / 640 - h / 480) < 1e-9, f"the raster is scaled unevenly in the video: {w}x{h}"
    return {"W": int(full["width"]), "H": int(full["height"]), "SIDE": int(round(x)), "TOP": int(round(y)), "ZOOM": w / 640}


def use_frame(frame):
    """Set the video frame every function here works in (a remote call gets the local one in its inputs)."""
    global W, H, SIDE, TOP, ZOOM
    W, H, SIDE, TOP, ZOOM = (frame[k] for k in ("W", "H", "SIDE", "TOP", "ZOOM"))


def frame_now():
    return {"W": W, "H": H, "SIDE": SIDE, "TOP": TOP, "ZOOM": ZOOM}


def full_k(k_clip):
    """Video-frame intrinsics from the clip frame's: x_video = z x_clip + SIDE + (z - 1)/2 (pixel centres), y likewise with TOP; on
    ME340 (z = 1.5) x_video = 1.5 x_clip + 160.25, y_video = 1.5 y_clip + 0.25."""
    fx, fy, cx, cy = k_clip
    z, half = ZOOM, (ZOOM - 1) / 2
    return np.array([[z * fx, 0, z * cx + SIDE + half], [0, z * fy, z * cy + TOP + half], [0, 0, 1.]])


def excluded_full(clip_mask, caption=CAPTIONS, dilate=DILATE):
    """Video-frame pixels left out of the loss and the metrics: the moving person from a clip-frame mask, dilated, and the caption box.

    The masks cover only the crop: a person touching its left or right edge may go on into that side strip, so the whole strip goes too.
    """
    import cv2
    full = np.zeros((H, W), np.uint8)
    right, bottom = SIDE + int(round(640 * ZOOM)), TOP + int(round(480 * ZOOM))
    full[TOP:bottom, SIDE:right] = cv2.resize(clip_mask.astype(np.uint8), (right - SIDE, bottom - TOP), interpolation=cv2.INTER_NEAREST_EXACT)
    if clip_mask[:, 0].any():
        full[:, :SIDE] = 1
    if clip_mask[:, -1].any():
        full[:, right:] = 1
    if dilate:
        full = cv2.dilate(full, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate + 1, 2 * dilate + 1)))
    y0, y1, x0, x1 = caption
    full[y0:y1, x0:x1] = 1
    return full > 0


def caption_half_width(bgr, band=CAPTIONS, centre=None):
    """Half-width of the caption line in one video frame (0: none): white text pixels beside the dark caption box inside the band,
    mirrored about the centre the lines are set on, since text over a bright floor can go unseen."""
    import cv2
    y0, y1, x0, x1 = band
    if y1 <= y0 or x1 <= x0:
        return 0
    centre = W // 2 if centre is None else centre
    rows = bgr[y0:y1, x0:x1]
    text = (rows.min(2) > 200) & (cv2.dilate((rows.max(2) < 70).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0)
    cols = np.nonzero(text.sum(0) >= 2)[0] + x0
    return int(np.abs(cols - centre).max()) if len(cols) else 0


def caption_boxes(video, n, band=CAPTIONS, pad=14, window=3):
    """Per-frame caption box (y0, y1, x0, x1): the widest half-width within +-window frames (a line changing, a missed frame), padded."""
    import cv2
    capture = cv2.VideoCapture(str(video))
    half = np.array([caption_half_width(capture.read()[1], band) for _ in range(n)])
    half = np.array([half[max(0, i - window):i + window + 1].max() for i in range(n)])
    y0, y1, x0, x1 = band
    return np.array([[y0, y1, max(x0, W // 2 - h - pad), min(x1, W // 2 + h + pad)] if h else [y0, y0, x0, x0] for h in half])


def timeline(knots, frames, every=KNOT_EVERY, used=None):
    """Per-frame corrections linear in time between knots every `every` frames, minus the mean of the knots the trained frames use:
    the clip as a whole stays put."""
    k = knots - (knots if used is None else knots[used]).mean(0)
    w = (frames % every) / every
    return (1 - w)[..., None] * k[frames // every] + w[..., None] * k[frames // every + 1]


def report_surfaces():
    """The report's room (fused mesh + single-view-depth fill): vertices, vertex colours 0..1 and one ray-casting scene of both.
    The fill is one vertex-coloured geometry, or (fill_scene_holes.py --video) several textured ones, sampled to vertex colours."""
    import open3d as o3d
    import trimesh
    mesh = o3d.io.read_triangle_mesh(str(MESH))
    fills = [g for name, g in trimesh.load(FILL, process=False).geometry.items() if name.startswith("single-view-depth-fill")]
    parts = [(np.asarray(mesh.vertices), np.asarray(mesh.triangles), np.asarray(mesh.vertex_colors))] + [
        (np.asarray(g.vertices), np.asarray(g.faces), (g.visual.to_color() if g.visual.kind == "texture" else g.visual).vertex_colors[:, :3] / 255.) for g in fills]
    scene = o3d.t.geometry.RaycastingScene()
    for v, f, _ in parts:
        scene.add_triangles(o3d.core.Tensor(v.astype(np.float32)), o3d.core.Tensor(f.astype(np.uint32)))
    return [p[0] for p in parts], [p[2] for p in parts], scene


def seed_points(video, K, c2w, frames, excluded_of, step=8, voxel=.01):
    """Seeds: the report's surface vertices, plus points where those surfaces leave a video frame empty (mostly the side strips the
    report never saw, as it was built from the crop): depth carried along the image row from the surface hits (inverse depth
    interpolated, exact for planes), colour from the frame. One point per voxel."""
    import cv2
    import open3d as o3d
    points, colours, scene = report_surfaces()
    v, u = np.mgrid[step // 2:H:step, step // 2:W:step]
    rays = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones(u.shape)], -1)  # camera z = 1, so the hit distance is depth
    capture, wanted, added = cv2.VideoCapture(str(video)), set(frames), 0
    for i in range(max(wanted) + 1):
        ok, bgr = capture.read()
        assert ok, "frame missing from the video"
        if i not in wanted:
            continue
        R, t = c2w[i][:3, :3], c2w[i][:3, 3]
        d = rays @ R.T
        cast = np.concatenate([np.broadcast_to(t, d.shape), d], -1).reshape(-1, 6).astype(np.float32)
        z = scene.cast_rays(o3d.core.Tensor(cast))["t_hit"].numpy().reshape(u.shape)
        inverse = np.full(u.shape, np.nan)
        for r in range(len(z)):
            hit = np.isfinite(z[r])
            if hit.sum() >= 2:
                inverse[r] = np.interp(u[r], u[r, hit], 1 / z[r, hit])
        new = ~np.isfinite(z) & (inverse > 0) & ~excluded_of(i)[v, u]
        points.append((rays[new] / inverse[new][:, None]) @ R.T + t)
        colours.append(bgr[v[new], u[new], ::-1] / 255.)
        added += int(new.sum())
    points, colours = np.concatenate(points), np.concatenate(colours)
    _, first = np.unique(np.floor(points / voxel).astype(np.int64), axis=0, return_index=True)
    return points[first].astype(np.float32), colours[first].astype(np.float32), {"surface_vertices": len(points) - added, "frame_seeds": added,
                                                                                  "after_voxel": len(first), "voxel_native": voxel}


def surface_depth(K, c2w, step=8):
    """Depth of the report's surfaces seen from each camera through the centres of step x step pixel blocks (0: no surface)."""
    import open3d as o3d
    scene = report_surfaces()[2]
    v, u = np.mgrid[:H // step, :W // step] * step + (step - 1) / 2
    rays = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones(u.shape)], -1)  # camera z = 1: the hit distance is depth
    out = np.zeros((len(c2w),) + u.shape, np.float16)
    for i, pose in enumerate(c2w):
        d = rays @ pose[:3, :3].T
        cast = np.concatenate([np.broadcast_to(pose[:3, 3], d.shape), d], -1).reshape(-1, 6).astype(np.float32)
        z = scene.cast_rays(o3d.core.Tensor(cast))["t_hit"].numpy().reshape(u.shape)
        out[i] = np.where(np.isfinite(z), z, 0)
    return out


def lingbot_map(folder, unseen=()):
    """A LingBot dense point map (lingbot_dense_map.py build, the report's world frame): positions, colours 0..1 and each point's
    sample size (native), without the points taken from the `unseen` source frames (held-out or skipped: never seen by the fit)."""
    import trimesh
    cloud = next(iter(trimesh.load(folder / "dense-points.glb", process=False).geometry.values()))
    attributes = np.load(folder / "point-attributes.npz")
    keep = ~np.isin(attributes["source_frame"], np.asarray(list(unseen), int))
    return (np.asarray(cloud.vertices, np.float32)[keep], (np.asarray(cloud.colors)[:, :3] / np.float32(255))[keep],
            attributes["spacing"].astype(np.float32)[keep])


def voxel_seeds(xyz, rgb, size, voxel, share=.5):
    """One seed per voxel, the finest sample in it, its scale a share of its sample size or the voxel, whichever is larger (.5: a
    closed surface; .1: gsplat's MCMC recipe, small seeds that grow)."""
    order = np.argsort(size, kind="stable")
    first = order[np.unique(np.floor(xyz[order] / voxel).astype(np.int64), axis=0, return_index=True)[1]]
    return xyz[first], rgb[first], np.maximum(size[first], voxel) * share


# ---------------------------------------------------------------- GPU side
def ssim_map(a, b, size=11, sigma=1.5):
    """Per-pixel SSIM of two HxWx3 images in 0..1, Gaussian window and zero padding as in 3DGS, mean over channels."""
    import torch
    import torch.nn.functional as F
    g = torch.exp(-(torch.arange(size, device=a.device) - size // 2) ** 2 / (2 * sigma ** 2))
    window = ((g[:, None] * g[None]) / g.sum() ** 2).repeat(3, 1, 1, 1)
    blur = lambda z: F.conv2d(z, window, padding=size // 2, groups=3)
    x, y = a.permute(2, 0, 1)[None], b.permute(2, 0, 1)[None]
    mx, my = blur(x), blur(y)
    sxx, syy, sxy = blur(x * x) - mx ** 2, blur(y * y) - my ** 2, blur(x * y) - mx * my
    c1, c2 = .01 ** 2, .03 ** 2
    return ((2 * mx * my + c1) * (2 * sxy + c2) / ((mx ** 2 + my ** 2 + c1) * (sxx + syy + c2)))[0].mean(0)


def activated(p):
    import torch
    return {"means": p["means"], "quats": p["quats"], "scales": torch.exp(p["scales"]), "opacities": torch.sigmoid(p["opacities"]),
            "colors": torch.sigmoid(p["colors"])}


def viewmat(c2w, pose=None):
    """World-to-camera of a camera-to-world, optionally corrected by pose = (rotation vector about the camera centre, world shift)."""
    import torch
    R, t = c2w[:3, :3], c2w[:3, 3]
    if pose is not None:
        w, z = pose[:3], torch.zeros((), device=c2w.device)
        skew = torch.stack([torch.stack([z, -w[2], w[1]]), torch.stack([w[2], z, -w[0]]), torch.stack([-w[1], w[0], z])])
        R, t = torch.linalg.matrix_exp(skew) @ R, t + pose[3:]
    out = torch.eye(4, device=c2w.device)
    out[:3, :3], out[:3, 3] = R.T, -R.T @ t
    return out


def expose(image, gain):
    import torch
    return image @ (torch.eye(3, device=image.device) + gain[:9].view(3, 3)).T + gain[9:]


def render(model, view, K, background=None, mode="RGB", size=None):
    from gsplat import rasterization
    colours, alphas, info = rasterization(model["means"], model["quats"], model["scales"], model["opacities"], model["colors"], view[None], K[None],
                                          *(size or (W, H)), packed=False, rasterize_mode="classic", render_mode=mode, backgrounds=None if background is None else background[None])
    return colours[0], info


def map_depth(data, xyz, sigma, step=8):
    """Depth of a dense point map seen from each training camera through step x step pixel blocks (0: not covered): its points drawn
    as opaque Gaussians of their own size, so nearer surfaces hide farther ones; mean expected depth of the well covered pixels."""
    import torch
    import torch.nn.functional as F
    from gsplat import rasterization
    n = len(xyz)
    ones = torch.ones(n, device="cuda")
    points = {"means": torch.tensor(xyz, device="cuda"), "quats": torch.tensor([[1., 0, 0, 0]], device="cuda").expand(n, 4).contiguous(),
              "scales": torch.tensor(sigma, device="cuda")[:, None].expand(n, 3).contiguous(), "opacities": ones, "colors": ones[:, None]}
    pool = lambda x: F.avg_pool2d(x[None, None], step)[0, 0]
    out = torch.zeros((len(data["c2w"]), H // step, W // step), dtype=torch.float16, device="cuda")
    with torch.no_grad():
        for i in data["train"]:
            depth, alpha, _ = rasterization(**points, viewmats=viewmat(data["c2w"][i])[None], Ks=data["K"][None], width=W, height=H, packed=False,
                                            rasterize_mode="classic", render_mode="ED")
            covered = (alpha[0, ..., 0] > .9).float()
            share = pool(covered)
            out[i] = torch.where(share > .9, pool(depth[0, ..., 0] * covered) / share.clamp(min=1e-6), 0).half()
    return out


def corrections(knots, cfg, frame):
    """Pose (6) and exposure (12) of one frame, None where that refinement is off."""
    import torch
    f, used = torch.tensor([frame], device=knots["pose"].device), knots.get("used")
    return (timeline(knots["pose"], f, used=used)[0] if cfg["pose"] else None), (timeline(knots["exposure"], f, used=used)[0] if cfg["exposure"] else None)


def load(inputs):
    import cv2
    import torch
    use_frame(inputs.get("frame") or frame_now())  # the local side's video frame (older inputs: ME340's)
    Path("/tmp/video.mp4").write_bytes(inputs["video"])
    n, capture = len(inputs["c2w"]), cv2.VideoCapture("/tmp/video.mp4")
    frames, excluded = torch.empty((n, H, W, 3), dtype=torch.uint8, device="cuda"), torch.empty((n, H, W), dtype=torch.bool, device="cuda")
    masks = np.unpackbits(inputs["masks"], axis=1)[:, :480 * 640].reshape(n, 480, 640).astype(bool)
    for i in range(n):
        ok, bgr = capture.read()
        assert ok and bgr.shape == (H, W, 3), f"video frame {i} missing or not {W}x{H}"
        frames[i] = torch.from_numpy(np.ascontiguousarray(bgr[..., ::-1]))
        excluded[i] = torch.from_numpy(excluded_full(masks[i], inputs["captions"][i]))
    assert not capture.read()[0], "the video has more frames than cameras"
    cuda = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32, device="cuda")
    # gsplat puts pixel centres at +0.5; K_video is OpenCV's (centres on integers), as the report's meshes and viewer use it
    data = {"frames": frames, "excluded": excluded, "K": cuda(np.asarray(inputs["K"]) + [[0, 0, .5], [0, 0, .5], [0, 0, 0]]), "c2w": cuda(inputs["c2w"]),
            "xyz": inputs["xyz"], "rgb": inputs["rgb"], "scale": inputs.get("scale"), "train": inputs["train"], "held": inputs["held"], "metres": inputs["metres"],
            "depth": None if inputs.get("depth") is None else torch.from_numpy(inputs["depth"]).cuda(), "skip": inputs.get("skip", [])}
    if inputs.get("prior") is not None:  # a dense point map to pull depth towards, drawn from each training camera here
        data["depth"] = map_depth(data, *inputs["prior"])
    return data


def fit(cfg, data, ckpt, stop_at):
    """Train from the seeds (or the checkpoint a restarted call left); returns parameters, knots and training stats."""
    import torch
    from gsplat import MCMCStrategy
    from scipy.spatial import cKDTree
    torch.manual_seed(0)
    steps, n_knots = cfg["steps"], len(data["c2w"]) // KNOT_EVERY + 2
    if ckpt.exists():
        saved = torch.load(ckpt, map_location="cuda")
        tensors, knots, start = saved["params"], saved["knots"], saved["step"]
    else:
        xyz, rgb, scale = data["xyz"], data["rgb"], data.get("scale")
        if scale is None:  # gsplat's MCMC recipe: a tenth of the RMS distance to 3 neighbours
            scale = .1 * np.sqrt((cKDTree(xyz).query(xyz, k=4)[0][:, 1:] ** 2).mean(1)).clip(1e-4)
        tensors = {"means": torch.tensor(xyz), "scales": torch.log(torch.tensor(scale, dtype=torch.float32))[:, None].repeat(1, 3),
                   "quats": torch.rand(len(xyz), 4), "opacities": torch.logit(torch.full((len(xyz),), .5)), "colors": torch.logit(torch.tensor(rgb).clamp(.02, .98))}
        knots, start = {k: torch.zeros(n_knots, d, device="cuda") for k, d in (("pose", 6), ("exposure", 12))}, 0
        trained = torch.tensor(data["train"], device="cuda") // KNOT_EVERY
        knots["used"] = torch.zeros(n_knots, dtype=torch.bool, device="cuda").index_fill_(0, torch.cat([trained, trained + 1]), True)
    params = torch.nn.ParameterDict({k: torch.nn.Parameter(v.float().cuda()) for k, v in tensors.items()})
    centres = data["c2w"][data["train"], :3, 3]
    lr = {**LR, "means": LR["means"] * 1.1 * float((centres - centres.mean(0)).norm(dim=1).max())}  # gsplat's scene scale
    optimizers = {k: torch.optim.Adam([{"params": params[k], "lr": lr[k], "name": k}], eps=1e-15) for k in params}
    refined = [k for k in ("pose", "exposure") if cfg[k]]
    for k in refined:
        knots[k].requires_grad_(True)
    refine = torch.optim.Adam([{"params": [knots[k]], "lr": lr[k], "name": k} for k in refined]) if refined else None
    strategy = MCMCStrategy(cap_max=cfg["cap"], refine_start_iter=min(500, steps // 4), refine_stop_iter=steps * 5 // 6)
    strategy.check_sanity(params, optimizers)
    state = strategy.initialize_state()
    frames, excluded, K, c2w = data["frames"], data["excluded"], data["K"], data["c2w"]
    decay, order, losses, pulls, started, done = .01 ** (1 / steps), [], [], [], time.time(), start
    small = cfg.get("train_scale", 1)  # train on 1/small of the video frame's width and height
    assert small == 1 or not (cfg.get("depth_weight") and data["depth"] is not None), "the depth pull is on full-size 8 px blocks"
    K = K.clone()
    K[2, 2] = small  # K / small scales focal lengths and centres (gsplat's +0.5 centres scale with the pixels) and keeps K[2, 2] = 1
    for step in range(start, steps):
        if time.time() > stop_at:
            break
        if not order:
            order = [data["train"][j] for j in torch.randperm(len(data["train"])).tolist()]
        i = order.pop()
        for group in optimizers["means"].param_groups + (refine.param_groups if refine else []):
            group["lr"] = lr[group["name"]] * decay ** step
        pose, gain = corrections(knots, cfg, i)
        prior = cfg.get("depth_weight") and data["depth"] is not None
        image, info = render(activated(params), viewmat(c2w[i], pose), K / small, background=torch.rand(3, device="cuda"), mode="RGB+ED" if prior else "RGB",
                             size=(W // small, H // small))
        image, depth = (image[..., :3], image[..., 3]) if prior else (image, None)
        image = image if gain is None else expose(image, gain)
        target, valid = frames[i].float() / 255, (~excluded[i]).float()
        if small > 1:  # ponytail: area-downsampled target, a block with any excluded pixel is excluded; train only, evaluation stays full size
            target = torch.nn.functional.avg_pool2d(target.permute(2, 0, 1)[None], small)[0].permute(1, 2, 0)
            valid = -torch.nn.functional.max_pool2d(-valid[None, None], small)[0, 0]
        l1 = ((image - target).abs().mean(-1) * valid).sum() / valid.sum()
        ssim = (ssim_map(image, target) * valid).sum() / valid.sum()
        loss = .8 * l1 + .2 * (1 - ssim) + .01 * torch.sigmoid(params["opacities"]).mean() + .01 * torch.exp(params["scales"]).mean()
        if prior:  # weak pull onto the report's surfaces or the dense map: clipped, so a surface it lacks or misplaces is not fought
            pool = lambda x: torch.nn.functional.avg_pool2d(x[None, None], 8)[0, 0]
            d, ref = pool(depth), data["depth"][i].float()
            ok = (ref > 0) & (d > 1e-3) & (pool(valid) > .99)
            if ok.any():
                pulls.append((torch.log(d[ok]) - torch.log(ref[ok])).abs().clamp(max=.2).mean())
                loss = loss + cfg["depth_weight"] * pulls[-1]
        loss.backward()
        for optimizer in list(optimizers.values()) + ([refine] if refine else []):
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        if cfg.get("max_elongation"):  # no needles: seen end-on from the path they hide, from anywhere else they streak
            with torch.no_grad():
                ordered, axes = params["scales"].sort(1)
                params["scales"].scatter_(1, axes[:, 2:], torch.minimum(ordered[:, 2:], ordered[:, 1:2] + np.log(cfg["max_elongation"])))
        strategy.step_post_backward(params, optimizers, state, step, info, lr=lr["means"] * decay ** step)
        losses.append(loss.detach())  # no per-step GPU sync
        done = step + 1
        if done % 500 == 0:
            print(f"{cfg['name']} step {done}/{steps} loss {float(torch.stack(losses[-500:]).mean()):.4f} gaussians {len(params['means'])} "
                  f"{(time.time() - started) / (done - start):.3f} s/step" + (f" depth {float(torch.stack(pulls[-500:]).mean()):.4f}" if pulls else ""), flush=True)
        if done % cfg["checkpoint_every"] == 0 or done == steps or time.time() > stop_at:
            torch.save({"step": done, "params": {k: v.detach() for k, v in params.items()}, "knots": {k: v.detach() for k, v in knots.items()}}, ckpt)
            volume.commit()
    stats = {"steps_done": done, "resumed_from_step": start, "train_seconds": round(time.time() - started, 1), "gaussians": len(params["means"]),
             "loss_per_500_steps": [round(float(torch.stack(losses[k:k + 500]).mean()), 4) for k in range(0, len(losses), 500)], "lr": lr,
             "depth_pull_per_500_steps": [round(float(torch.stack(pulls[k:k + 500]).mean()), 4) for k in range(0, len(pulls), 500)]}
    trained = torch.tensor(data["train"], device="cuda")
    if cfg["pose"]:
        p = timeline(knots["pose"].detach(), trained, used=knots.get("used"))
        angle, shift = torch.rad2deg(p[:, :3].norm(dim=1)), p[:, 3:].norm(dim=1)
        stats["pose_correction"] = {"mean_rotation_deg": float(angle.mean()), "max_rotation_deg": float(angle.max()), "mean_translation_native": float(shift.mean()),
                                    "mean_translation_m": float(shift.mean()) * data["metres"], "max_translation_m": float(shift.max()) * data["metres"],
                                    "net_over_clip": "zero by construction (knots minus their mean)"}
    if cfg["exposure"]:
        e = timeline(knots["exposure"].detach(), trained, used=knots.get("used"))
        gain = 1 + (e[:, 0] + e[:, 4] + e[:, 8]) / 3
        stats["exposure_correction"] = {"mean_gain_min": float(gain.min()), "mean_gain_max": float(gain.max()), "mean_gain_std": float(gain.std()),
                                        "offset_min": float(e[:, 9:].min()), "offset_max": float(e[:, 9:].max())}
    return params, knots, stats


def evaluate(model, knots, cfg, data, net, show=()):
    """Masked PSNR / SSIM / LPIPS of every held-out frame for each way of rendering it; `show` frames come back as real | render PNGs."""
    import cv2
    import torch
    variants = {"report_pose": (False, cfg["exposure"])}
    if cfg["pose"]:
        variants["corrected_pose"] = (True, cfg["exposure"])
    if cfg["exposure"]:
        variants["report_pose_no_exposure"] = (False, False)
    primary = "corrected_pose" if cfg["pose"] else "report_pose"
    per, pngs = {name: [] for name in variants}, {}
    with torch.no_grad():
        for i in data["held"]:
            pose, gain = corrections(knots, cfg, i)
            target, valid = data["frames"][i].float() / 255, (~data["excluded"][i]).float()
            renders = {}
            for name, (corrected, exposed) in variants.items():
                if corrected not in renders:
                    renders[corrected] = render(model, viewmat(data["c2w"][i], pose if corrected else None), data["K"])[0]
                image = (expose(renders[corrected], gain) if exposed else renders[corrected]).clamp(0, 1)
                mse = (((image - target) ** 2).mean(-1) * valid).sum() / valid.sum()
                lp = net(image.permute(2, 0, 1)[None] * 2 - 1, target.permute(2, 0, 1)[None] * 2 - 1)[0, 0]
                per[name].append({"frame": i, "psnr": float(-10 * torch.log10(mse)), "ssim": float((ssim_map(image, target) * valid).sum() / valid.sum()),
                                  "lpips": float((lp * valid).sum() / valid.sum())})
                if i in show and name == primary:
                    both = torch.cat([target, image], 1).mul(255).round().byte().cpu().numpy()
                    pngs[i] = cv2.imencode(".png", both[..., ::-1])[1].tobytes()
    summary = {name: {m: round(float(np.mean([r[m] for r in rows])), 4) for m in ("psnr", "ssim", "lpips")} | {"frames": len(rows)} for name, rows in per.items()}
    return {"primary": primary, "summary": summary, "per_frame": per}, pngs


def export(params, cfg, data):
    """splat32 records of the Gaussians an 8-bit alpha can still show, most important first, at most the cap. Strays the MCMC noise
    threw beyond three times the farthest report surface (seen from the camera path's centre) go too."""
    m = {k: v.detach().cpu().numpy() for k, v in activated(params).items()}
    centre = data["c2w"][:, :3, 3].mean(0).cpu().numpy()
    far = 3 * float(np.linalg.norm(data["xyz"] - centre, axis=1).max())
    index = splat_to_web.keep(m["means"], m["scales"], m["opacities"], cfg["cap"], centre=centre, max_distance=far)
    records = splat_to_web.pack(m["means"][index], m["scales"][index], m["colors"][index], m["opacities"][index], m["quats"][index])
    return records, {"trained": len(m["means"]), "kept": len(index), "alpha_byte_zero": int((m["opacities"] < .5 / 255).sum()),
                     "farther_than_native": [far, int((np.linalg.norm(m["means"] - centre, axis=1) > far).sum())]}


@app.function(image=image, gpu=GPUS, cpu=CPU, memory=MEMORY_MIB, volumes={"/ckpt": volume}, timeout=3600, retries=0, max_containers=1, scaledown_window=2)
def train_remote(inputs, configs, run, deadline):
    """Trains each config in turn from the same seeds; the exported one comes back as splat32 bytes, PNGs and metrics."""
    import gsplat
    import lpips
    import torch
    started = time.time()
    folder = Path("/ckpt") / run
    folder.mkdir(parents=True, exist_ok=True)
    data = load(inputs)
    net = lpips.LPIPS(net="alex", spatial=True, verbose=False).cuda().eval()
    out = {"gpu": torch.cuda.get_device_name(), "gsplat": gsplat.__version__, "torch": str(torch.__version__), "load_seconds": round(time.time() - started, 1), "runs": {}}
    print(json.dumps({k: out[k] for k in ("gpu", "gsplat", "load_seconds")}), flush=True)
    for name, cfg in configs.items():
        stop_at = deadline - cfg["reserve_s"]
        if time.time() > stop_at:
            out["runs"][name] = {"skipped": "deadline reached before this config started"}
            continue
        params, knots, stats = fit({**cfg, "name": name}, data, folder / f"{name}.pt", stop_at)
        stats["trained_model"], _ = evaluate(activated(params), knots, cfg, data, net)
        if cfg["export"]:
            records, stats["pruning"] = export(params, cfg, data)
            s = splat_to_web.unpack(records.tobytes())
            as_file = {"means": s["positions"], "quats": s["quats"], "scales": s["scales"], "opacities": s["opacity"], "colors": s["rgb"]}
            stats["exported_file"], pngs = evaluate({k: torch.tensor(v, dtype=torch.float32, device="cuda") for k, v in as_file.items()}, knots, cfg, data, net, SHOW)
            stats["files"] = {"splat": records.tobytes(), "pngs": pngs}
            if cfg["pose"]:  # the cameras the splats agree with: the report's, corrected by the knots
                with torch.no_grad():
                    p = timeline(knots["pose"], torch.arange(len(data["c2w"]), device="cuda"), used=knots.get("used"))
                    p[data["skip"]] = 0  # frames of another shot keep the report's camera
                    stats["files"]["refined_c2w"] = torch.stack([torch.linalg.inv(viewmat(c, q)) for c, q in zip(data["c2w"], p)]).cpu().numpy()
            for key, value in stats["files"].items():
                (folder / f"{name}-{key}.pkl").write_bytes(pickle.dumps(value))  # for recovery if the local client is lost
        (folder / f"{name}.json").write_text(json.dumps({k: v for k, v in stats.items() if k != "files"}))
        volume.commit()
        print(json.dumps({name: {k: stats[k] for k in ("steps_done", "gaussians", "train_seconds")} | {"held_out": stats["trained_model"]["summary"]}}), flush=True)
        out["runs"][name] = stats
    out["container_seconds"] = round(time.time() - started, 1)
    return out


def splat_evidence(model, data, every=2, scale=2):
    """Per splat: its blending weight summed over training views (every 2nd, half resolution, unmasked pixels only) and the views that
    give it any (> 0.001 px); elongation (longest / middle axis); how closely the long axis points at the nearest camera. A splat's weight
    in a view is the gradient of the kept image's sum with respect to its colour, so what the surfaces in front hide counts for nothing,
    and dropping splats of total weight below t changes no training view by more than t pixels' worth."""
    import torch
    from gsplat import rasterization
    n, (w, h) = len(model["means"]), (W // scale, H // scale)
    K = data["K"].clone()
    K[:2] /= scale  # gsplat's pixel-centre convention scales with the raster
    colours = model["colors"].detach().clone().requires_grad_(True)
    views, weight = torch.zeros(n, dtype=torch.int32, device="cuda"), torch.zeros(n, device="cuda")
    for i in data["train"][::every]:
        image = rasterization(model["means"], model["quats"], model["scales"], model["opacities"], colours, viewmat(data["c2w"][i])[None], K[None],
                              w, h, packed=False)[0][0]
        (image * ~data["excluded"][i][::scale, ::scale, None]).sum().backward()
        views += (colours.grad.mean(1) > 1e-3).int()
        weight += colours.grad.mean(1)
        colours.grad = None
    with torch.no_grad():
        sizes, axis = model["scales"].sort(1)
        q = model["quats"] / model["quats"].norm(dim=1, keepdim=True)
        rw, rx, ry, rz = q.unbind(1)
        rot = torch.stack([torch.stack([1 - 2 * (ry * ry + rz * rz), 2 * (rx * ry - rw * rz), 2 * (rx * rz + rw * ry)], 1),
                           torch.stack([2 * (rx * ry + rw * rz), 1 - 2 * (rx * rx + rz * rz), 2 * (ry * rz - rw * rx)], 1),
                           torch.stack([2 * (rx * rz - rw * ry), 2 * (ry * rz + rw * rx), 1 - 2 * (rx * rx + ry * ry)], 1)], 1)  # rows of R
        longest = torch.gather(rot, 2, axis[:, 2].view(-1, 1, 1).expand(-1, 3, 1))[..., 0]  # column of R along the longest scale
        cams = data["c2w"][::8, :3, 3]
        nearest = torch.cat([cams[torch.cdist(chunk, cams).argmin(1)] for chunk in model["means"].split(262144)])
        to_camera = torch.nn.functional.normalize(nearest - model["means"], dim=1)
    return {"views": views, "weight": weight, "ratio": sizes[:, 2] / sizes[:, 1], "longest": sizes[:, 2], "opacity": model["opacities"], "aligned": (longest * to_camera).sum(1).abs()}


def drops(rule, f):
    """Splats a cleanup rule removes: any clause whose thresholds are all given."""
    out = (f["distance"] > rule["far"]) & (f["views"] < rule["min_views"])
    out |= (f["ratio"] > rule["needle_ratio"]) & (f["opacity"] < rule["needle_opacity"])
    out |= (f["longest"] > rule["huge"]) & (f["opacity"] < rule["huge_opacity"])
    if rule.get("aligned_ratio"):
        out |= (f["ratio"] > rule["aligned_ratio"]) & (f["aligned"] > rule["aligned_cos"])
    if rule.get("faint"):  # negligible to every training view, wherever it sits (or only away from the report's surfaces)
        out |= (f["weight"] < rule["faint"]) & (f["distance"] > rule.get("faint_far", -1))
    return out


def as_model(records):
    """splat32 bytes as the activated Gaussians render() takes, on the GPU."""
    import torch
    s = splat_to_web.unpack(records)
    return {k: torch.tensor(s[v], dtype=torch.float32, device="cuda") for k, v in
            (("means", "positions"), ("quats", "quats"), ("scales", "scales"), ("opacities", "opacity"), ("colors", "rgb"))}


def shoot(model, views, ks, quality=88):
    """JPEGs of a model from camera-to-world views with their intrinsics."""
    import cv2
    import torch
    cuda = lambda a: torch.tensor(a, dtype=torch.float32, device="cuda")
    with torch.no_grad():
        shots = [render(model, torch.linalg.inv(cuda(c)), cuda(k))[0].clamp(0, 1) for c, k in zip(views, ks)]
    return [cv2.imencode(".jpg", (x * 255).round().byte().cpu().numpy()[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tobytes() for x in shots]


@app.function(image=image, gpu=GPUS, cpu=CPU, memory=MEMORY_MIB, volumes={"/ckpt": volume}, timeout=1800, retries=0, max_containers=1, scaledown_window=2)
def clean_remote(inputs, records, distance, rules, views, ks, png_rules=("none",)):
    """Held-out metrics and renders from fixed views of the exported splats under each cleanup rule (real | render PNGs for png_rules);
    returns the kept splats per rule as bits."""
    import lpips
    import torch
    started = time.time()
    data = load(inputs)
    net = lpips.LPIPS(net="alex", spatial=True, verbose=False).cuda().eval()
    model = as_model(records)
    f = splat_evidence(model, data) | {"distance": torch.tensor(distance, device="cuda")}
    dummy, off = {"pose": torch.zeros(1, device="cuda")}, {"pose": False, "exposure": False}  # inputs["c2w"] are already the refined cameras
    out = {"evidence": {"views_quantiles_1_10_50": torch.quantile(f["views"].float()[::16], torch.tensor([.01, .1, .5], device="cuda")).tolist(),
                        "views_under_2": int((f["views"] < 2).sum()), "weight_quantiles_10_50_90": torch.quantile(f["weight"][::16],
                        torch.tensor([.1, .5, .9], device="cuda")).tolist(),
                        "ratio_over_10": int((f["ratio"] > 10).sum()), "aligned_needles": int(((f["ratio"] > 5) & (f["aligned"] > .9)).sum())}, "rules": {}}
    for name, rule in {"none": None, **rules}.items():
        keep = torch.ones(len(model["means"]), dtype=torch.bool, device="cuda") if rule is None else ~drops(rule, f)
        kept = {k: v[keep] for k, v in model.items()}
        scored, pngs = evaluate(kept, dummy, off, data, net, SHOW if name in png_rules else ())
        metrics = scored["summary"]["report_pose"]
        out["rules"][name] = {"kept": int(keep.sum()), "removed": int((~keep).sum()), "held_out": metrics, "view_jpgs": shoot(kept, views, ks),
                              "keep_bits": np.packbits(keep.cpu().numpy()), "pngs": pngs}
        print(name, {k: v for k, v in out["rules"][name].items() if k in ("kept", "removed", "held_out")}, flush=True)
    out["container_seconds"] = round(time.time() - started, 1)
    return out


@app.function(image=image, gpu=GPUS, cpu=CPU, memory=MEMORY_MIB, timeout=900, retries=0, max_containers=1, scaledown_window=2)
def render_remote(files, views, ks, frame=None):
    """JPEGs of each named splat32 file from each view, in the given video frame."""
    use_frame(frame or frame_now())
    started = time.time()
    jpgs = {name: shoot(as_model(records), views, ks, quality=92) for name, records in files.items()}
    return {"jpgs": jpgs, "container_seconds": round(time.time() - started, 1)}


# ---------------------------------------------------------------- local side
def look_at(eye, target, up):
    """Camera-to-world (OpenCV axes) at eye looking at target, image y down along -up."""
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    return np.vstack([np.column_stack([x, np.cross(z, x), z, eye]), [0, 0, 0, 1]])


def orbit_views(c2w, back=1.0, rise=.7, ahead=2., yaw=35.):
    """Three free cameras like the report's default view: raised above and behind the start of the path, looking along it
    (camera-to-world, OpenCV axes), turned -yaw, 0, +yaw about the start."""
    up = -c2w[:, :3, 1].mean(0)  # the video cameras' image y points down
    up /= np.linalg.norm(up)
    start, forward = c2w[0, :3, 3], c2w[-1, :3, 3] - c2w[0, :3, 3]
    forward -= up * (forward @ up)
    forward /= np.linalg.norm(forward)
    return np.array([look_at(start - back * (np.cos(a) * forward + np.sin(a) * np.cross(up, forward)) + rise * up, start + ahead * forward, up)
                     for a in np.radians([-yaw, 0., yaw])])


def moderate_views(c2w, K, picks=MODERATE):
    """Free cameras a short drag off the path: each eye moved from a path camera by a small offset (right, up, back), looking at the
    point that camera saw at a pixel (given its depth)."""
    views = []
    inside = lambda p: p[0] < len(c2w) and 0 <= p[1][0] < W and 0 <= p[1][1] < H  # ponytail: ME340's picks; another clip keeps those inside it
    for frame, pixel, depth, offset in filter(inside, picks):
        R, t = c2w[frame][:3, :3], c2w[frame][:3, 3]
        views.append(look_at(t + R @ (np.array(offset) * [1, -1, -1]), R @ (np.linalg.inv(K) @ [*pixel, 1.] * depth) + t, -R[:, 1]))
    return np.array(views)


def comparison_views(report_c2w, K):
    """The views every model is compared in, from the report's cameras so they are the same for every run: three orbits like the
    report's default view (70 degrees across) and three moderate drags off the path (the viewer's free camera, 60 degrees high)."""
    k = lambda focal: np.array([[focal, 0, W / 2], [0, focal, H / 2], [0, 0, 1.]])
    moderate = moderate_views(report_c2w, K)
    return (np.concatenate([orbit_views(report_c2w), moderate]),
            np.array([k(W / 2 / np.tan(np.radians(35)))] * 3 + [k(H / 2 / np.tan(np.radians(30)))] * len(moderate)),
            [f"orbit-{j}" for j in range(3)] + [f"moderate-{j}" for j in range(len(moderate))])


def clean(args):
    """Cleanup rules on a finished run's splats.splat: held-out metrics and orbit views per rule on Modal, then splats-clean.* locally."""
    import cv2
    import open3d as o3d
    run_dir = args.clean
    use_frame(frame_of(args.clip))
    state = json.loads((run_dir / "train.json").read_text())
    records = (run_dir / "splats.splat").read_bytes()
    saved = np.load(run_dir / "refined-cameras.npz")
    cameras = saved["poses_c2w"].astype(np.float64)
    n = len(cameras)
    positions = splat_to_web.unpack(records)["positions"]
    surfaces, _, scene = report_surfaces()
    if state.get("lingbot"):  # "far" means off the dense map the run was pulled onto
        from scipy.spatial import cKDTree
        distance = cKDTree(lingbot_map(Path(state["lingbot"]))[0]).query(positions, workers=-1)[0].astype(np.float32)
    else:
        distance = scene.compute_distance(o3d.core.Tensor(positions)).numpy()
    centre = cameras[:, :3, 3].mean(0)
    far = 3 * float(np.linalg.norm(np.concatenate(surfaces) - centre, axis=1).max())
    stray = np.linalg.norm(positions - centre, axis=1) > far
    output = run_dir / "clean"
    output.mkdir(exist_ok=True)
    np.save(output / "keep-far.npy", np.packbits(~stray))
    records = np.frombuffer(records, splat_to_web.RECORD)[~stray].tobytes()  # far strays go from every variant, the main file included
    distance = distance[~stray]
    masks = np.zeros((n, 480, 640), bool)
    for i in range(n):
        for path in args.masks.glob(f"{i:05d}-*.png"):
            masks[i] |= cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) > 0
    video = args.clip / "source-full.mp4"
    inputs = {"video": video.read_bytes(), "masks": np.packbits(masks.reshape(n, -1), axis=1), "captions": caption_boxes(video, n, tuple(state["caption_band_y0_y1_x0_x1"])),
              "K": np.array(state["K_video"]), "c2w": cameras, "xyz": None, "rgb": None, "metres": state["metres_per_native_unit"], "frame": frame_now()}
    skip = set(state.get("skip_frames", []))
    inputs.update(train=[i for i in range(n) if i % HOLD_OUT and i not in skip], held=[i for i in range(0, n, HOLD_OUT) if i not in skip], skip=sorted(skip))
    off = OFF
    rules = {"unseen_far": {**off, "far": .035, "min_views": 2},  # the viewer's suggestion; views: every 2nd training frame
             "negligible_0.05": {**off, "faint": .05}, "negligible_0.2": {**off, "faint": .2}, "negligible_1": {**off, "faint": 1.},
             "negligible_far_1": {**off, "faint": 1., "faint_far": .035}, "negligible_far_5": {**off, "faint": 5., "faint_far": .035}}
    views, ks, names = comparison_views(saved["report_poses_c2w"].astype(np.float64), inputs["K"])
    with modal.enable_output(), app.run():
        result = clean_remote.remote(inputs, records, distance, rules, views, ks, ("none", "negligible_1"))
    kind, cost = usd("H100", result["container_seconds"] + 60)  # upper bound: the H100 rate
    summary = {"far_strays_removed": int(stray.sum()), "far_native": far, "rules": {k: v for k, v in rules.items()}, "evidence": result["evidence"],
               "container_seconds": result["container_seconds"], "usd_estimate_upper": round(cost, 3), "results": {}}
    for name, r in result["rules"].items():
        np.save(output / f"keep-{name}.npy", r.pop("keep_bits"))
        for view, jpg in zip(names, r.pop("view_jpgs")):
            (output / f"{view}-{name}.jpg").write_bytes(jpg)
        for frame, png in r.pop("pngs").items():
            (output / f"heldout-{frame:05d}-{name}.png").write_bytes(png)
        summary["results"][name] = r
    (output / "clean.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "rules"}, indent=1))


def pick(args):
    """splats-clean.* next to the run's splats.*: its splats under the chosen scored rule, with that rule's held-out metrics."""
    run_dir = args.clean
    summary = json.loads((run_dir / "clean" / "clean.json").read_text())
    info = json.loads((run_dir / "splats.json").read_text())
    records = np.frombuffer((run_dir / "splats.splat").read_bytes(), splat_to_web.RECORD)
    for name in ("far", args.pick):
        records = records[np.unpackbits(np.load(run_dir / "clean" / f"keep-{name}.npy"))[:len(records)].astype(bool)]
    state = json.loads((run_dir / "train.json").read_text())
    r, cfg = summary["results"][args.pick], state["configs"]["final"]
    meta = {k: v for k, v in info.items() if k not in ("format", "count", "bytesPerGaussian", "bounds", "metrics")}
    meta["training"] = {"capGaussians": cfg["cap"], "maxElongation": cfg.get("max_elongation"), "depthWeight": cfg.get("depth_weight"),
                        "depthPrior": state.get("depth_prior"), "seeds": state.get("seeds"), "lingbotMap": state.get("lingbot")}
    metrics = {"heldOut": {**info["metrics"]["heldOut"], **r["held_out"], "note": HELD_OUT_NOTE}, "heldOutBeforeCleanup": summary["results"]["none"]["held_out"]}
    cleanup = {"rule": {k: v for k, v in summary["rules"][args.pick].items() if OFF.get(k) != v}, "removed": r["removed"] + summary["far_strays_removed"],
               "evidence": "weight: a splat's blending weight summed over training views (every 2nd, half resolution, unmasked pixels), in pixels; "
                           f"far: distance to the {'LingBot map points' if state.get('lingbot') else 'report surfaces'} (native); views: training views giving it any weight",
               "views": f"clean/orbit-*-{args.pick}.jpg and clean/moderate-*-{args.pick}.jpg against the same *-none.jpg"}
    splat_to_web.write(run_dir, records, name="splats-clean", **meta, metrics=metrics, cleanup=cleanup)
    shots = sorted((run_dir / "clean").glob(f"heldout-*-{args.pick}.png"))
    if shots:  # real | render of the file just written, in place of the uncleaned run's
        for old in run_dir.glob("heldout-*.png"):
            old.unlink()
        for shot in shots:
            (run_dir / shot.name.replace(f"-{args.pick}", "")).write_bytes(shot.read_bytes())
    write_cameras(run_dir, json.loads((args.clip / "clip.json").read_text())["playback"]["fps"])
    print(json.dumps({rule: summary["results"][rule]["held_out"] | {"kept": summary["results"][rule]["kept"]} for rule in ("none", args.pick)}))


def write_cameras(run_dir, fps):
    """refined-cameras.json: the cameras the splats agree with, one per clip frame the run used (row-major 4x4 camera-to-world, OpenCV
    axes, the report's world and units). Pose corrections are one piecewise-linear function of time fitted to the trained frames, so a
    held-out frame gets its value between the neighbouring trained frames."""
    cameras = np.load(run_dir / "refined-cameras.npz")
    skip = set(json.loads((run_dir / "train.json").read_text()).get("skip_frames", []))
    frames = [i for i in range(len(cameras["poses_c2w"])) if i not in skip]
    refined, raw = cameras["poses_c2w"][frames].astype(np.float64), cameras["report_poses_c2w"][frames].astype(np.float64)
    turn = np.einsum("nij,nkj->nik", refined[:, :3, :3], raw[:, :3, :3])  # refined rotation times the raw one's inverse
    angle = np.degrees(np.arccos(np.clip((np.trace(turn, axis1=1, axis2=2) - 1) / 2, -1, 1)))
    shift = np.linalg.norm(refined[:, :3, 3] - raw[:, :3, 3], axis=1)
    (run_dir / "refined-cameras.json").write_text(json.dumps({
        "fps": fps, "coordinateFrameId": FRAME_ID, "frames": frames, "c2w": refined.reshape(len(frames), 16).round(7).tolist(),
        "frameIndex": "clip frame i = MP4 frame i of source-rgb.mp4 / source-full.mp4", "convention": "OpenCV camera: x right, y down, z forward",
        "meanCorrection": {"rotationDeg": float(angle.mean()), "maxRotationDeg": float(angle.max()), "translationNative": float(shift.mean()),
                           "maxTranslationNative": float(shift.max())}}))


def compare(args):
    """Side-by-side sheets (one per fixed view, and all views) of splat files given as LABEL=PATH, rendered in one GPU call."""
    import cv2
    use_frame(frame_of(args.clip))
    views, ks, names = comparison_views(np.load(args.droid_run / "prediction.npz")["poses_c2w"].astype(np.float64),
                                        full_k(json.loads((args.clip / "clip.json").read_text())["K"]))
    files = dict(item.split("=", 1) for item in args.compare)
    with modal.enable_output(), app.run():
        result = render_remote.remote({label: Path(path).read_bytes() for label, path in files.items()}, views, ks, frame_now())
    args.output.mkdir(parents=True, exist_ok=True)

    def tile(jpg, label, size):
        image = cv2.resize(cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR), size, interpolation=cv2.INTER_AREA)
        for colour, width in (((0, 0, 0), 5), ((255, 255, 255), 2)):
            cv2.putText(image, label, (14, 34), cv2.FONT_HERSHEY_SIMPLEX, .9, colour, width, cv2.LINE_AA)
        return image
    rows, big, small = [], (960, round(960 * H / W)), (640, round(640 * H / W))  # the video frame's aspect
    for v, view in enumerate(names):
        cv2.imwrite(str(args.output / f"compare-{view}.jpg"), np.hstack([tile(result["jpgs"][f][v], f"{f}  {view}", big) for f in files]),
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
        rows.append(np.hstack([tile(result["jpgs"][f][v], f"{f}  {view}", small) for f in files]))
    cv2.imwrite(str(args.output / "compare-all.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
    _, cost = usd("H100", result["container_seconds"] + 60)  # upper bound: the H100 rate
    (args.output / "compare.json").write_text(json.dumps({"files": files, "views": names, "c2w": views.tolist(), "K": ks.tolist(),
                                                          "moderate": MODERATE, "container_seconds": result["container_seconds"], "usd_estimate_upper": round(cost, 3)}, indent=1))
    print(json.dumps({"sheets": str(args.output), "container_seconds": result["container_seconds"], "usd_estimate_upper": round(cost, 3)}))


def usd(gpu_name, seconds):
    kind = "H100" if "H100" in gpu_name else "A100-80GB" if "80GB" in gpu_name else "A100-40GB"
    return kind, seconds * (USD_PER_S[kind] + CPU * USD_PER_S["cpu_core"] + MEMORY_MIB / 1024 * USD_PER_S["memory_gib"])


def run(args):
    import cv2
    use_frame(frame_of(args.clip))
    clip = json.loads((args.clip / "clip.json").read_text())
    K, c2w = full_k(clip["K"]), np.load(args.droid_run / "prediction.npz")["poses_c2w"].astype(np.float64)
    n, video = len(c2w), args.clip / "source-full.mp4"
    skip = sorted({i for a, b in (r.split("-") for r in args.skip) for i in range(int(a), int(b) + 1)})
    held, train = [i for i in range(0, n, HOLD_OUT) if i not in skip], [i for i in range(n) if i % HOLD_OUT and i not in skip]
    masks = np.zeros((n, 480, 640), bool)
    for i in range(n):
        for path in args.masks.glob(f"{i:05d}-*.png"):
            masks[i] |= cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) > 0
    captions = caption_boxes(video, n, tuple(args.captions))
    points = None
    if args.lingbot:  # seeds and depth pull from the dense map instead of the report's surfaces
        points, colours, sizes = lingbot_map(args.lingbot, set(held) | set(skip))
        xyz, rgb, scale = voxel_seeds(points, colours, sizes, args.seed_voxel, args.seed_scale)
        seeds = {"lingbot_points": len(points), "voxel_native": args.seed_voxel, "after_voxel": len(xyz), "scale": f"{args.seed_scale} x max(sample size, voxel)"}
    else:
        (xyz, rgb, seeds), scale = seed_points(video, K, c2w, train[::16], lambda i: excluded_full(masks[i], captions[i])), None
    metres = json.loads(SCALE.read_text())["metres_per_native_unit"]
    inputs = {"video": video.read_bytes(), "masks": np.packbits(masks.reshape(n, -1), axis=1), "captions": captions, "K": K, "c2w": c2w,
              "xyz": xyz, "rgb": rgb, "scale": scale, "train": train, "held": held, "metres": metres, "skip": skip, "frame": frame_now(),
              "depth": surface_depth(K, c2w) if args.depth_weight and not args.lingbot else None,
              "prior": (points, sizes / 2) if args.depth_weight and args.lingbot else None}
    base = {"steps": args.steps, "cap": args.cap, "checkpoint_every": 5000, "reserve_s": 120 if args.ablation else 420, "max_elongation": args.max_elongation,
            "depth_weight": args.depth_weight, "train_scale": args.train_scale}
    configs = ({"plain": {**base, "pose": False, "exposure": False, "export": False}, "exposure": {**base, "pose": False, "exposure": True, "export": False},
                "exposure_pose": {**base, "pose": True, "exposure": True, "export": False}} if args.ablation else
               {"final": {**base, "pose": args.pose, "exposure": args.exposure, "export": True}})
    run_name = args.resume or f"{args.output.name}-{int(time.time())}"
    args.output.mkdir(parents=True, exist_ok=True)
    assert not (args.output / "train.json").exists() and (args.resume or not (args.output / "launch.json").exists()), \
        "output already holds a run, or a launched one that may still be running (--resume it)"
    state = {"status": "gpu_running", "run": run_name, "volume": "panoptes-splat-train", "configs": configs, "gpu_fallback": [args.gpu] if args.gpu else GPUS, "retries": 0,
             "max_minutes": args.max_minutes, "seeds": seeds, "clip": str(args.clip), "droid_run": str(args.droid_run), "masks": str(args.masks),
             "mesh": str(MESH), "fill": str(FILL), "caption_band_y0_y1_x0_x1": list(args.captions),
             "caption_box_mean_width_px": float((captions[:, 3] - captions[:, 2]).clip(0).mean()), "trained_frames": len(train), "held_out_frames": len(held),
             "K_video": K.tolist(), "video_frame": frame_now(), "metres_per_native_unit": metres, "skip_frames": skip, "lingbot": str(args.lingbot) if args.lingbot else None,
             "depth_prior": None if not args.depth_weight else "LingBot map drawn from each training camera (points as opaque Gaussians, sigma half the sample size)"
             if args.lingbot else "report surfaces ray cast from each training camera"}
    (args.output / "launch.json").write_text(json.dumps(state, indent=1))
    started = time.time()
    with modal.enable_output(), app.run(detach=True):  # detached: a lost local client leaves the call running; results also land on the volume
        state["app_id"] = app.app_id
        (args.output / "launch.json").write_text(json.dumps(state, indent=1))
        deadline = time.time() + 60 * args.max_minutes
        result = train_remote.with_options(timeout=int(60 * args.max_minutes) + 120, **({"gpu": args.gpu} if args.gpu else {})).remote(inputs, configs, run_name, deadline)
    call_seconds = time.time() - started
    kind, cost = usd(result["gpu"], result["container_seconds"] + 60)  # + container start and imports, which the function cannot time
    state.update(status="complete", gpu=result["gpu"], gpu_kind=kind, gsplat=result["gsplat"], torch=result["torch"], container_seconds=result["container_seconds"],
                 call_seconds=round(call_seconds, 1), usd_estimate=round(cost, 3), usd_rates=USD_PER_S, runs=result["runs"])
    for name, r in result["runs"].items():
        if "files" not in r:
            continue
        files = r.pop("files")
        records = np.frombuffer(files["splat"], splat_to_web.RECORD)
        for frame, png in files["pngs"].items():
            (args.output / f"heldout-{frame:05d}.png").write_bytes(png)
        if "refined_c2w" in files:
            np.savez(args.output / "refined-cameras.npz", poses_c2w=files["refined_c2w"], report_poses_c2w=c2w.astype(np.float32))
        primary = r["exported_file"]["primary"]
        r["alignment"] = alignment(records, metres, points)
        splat_to_web.write(args.output, records, coordinateFrameId=FRAME_ID, units="native (metres_per_native_unit %.4f, assumed camera height)" % metres,
                           metrics={"heldOut": r["exported_file"]["summary"][primary] | {"pose": primary, "exposure": "interpolated from training-frame knots" if configs[name]["exposure"] else "none",
                                                                                         "excluded": "presenter (dynamic masks, dilated 15 px) and each frame's caption box", "note": HELD_OUT_NOTE},
                                    "heldOutAllVariants": r["exported_file"]["summary"], "alignmentToReportSurfaces": r["alignment"]},
                           trainedFrames=train, heldOutFrames=held, skippedFrames=skip, gsplatVersion=result["gsplat"], steps=r["steps_done"],
                           poseRefinement=r.get("pose_correction") and {**r["pose_correction"], "refinedCameras": "refined-cameras.npz poses_c2w, one per clip frame"},
                           exposureCompensation=bool(configs[name]["exposure"]))
    (args.output / "train.json").write_text(json.dumps(state, indent=1))
    print(json.dumps({"gpu": result["gpu"], "call_seconds": round(call_seconds), "usd_estimate": round(cost, 3),
                      **{name: (r.get("exported_file") or r.get("trained_model") or r).get("summary", r) for name, r in result["runs"].items()}}, indent=1))


def alignment(records, metres, points=None):
    """How far the opaque splats sit from the report's own surfaces (they were never told to stay on them) and from a dense map's points."""
    import open3d as o3d
    from scipy.spatial import cKDTree
    s = splat_to_web.unpack(records.tobytes())
    opaque = s["positions"][s["opacity"] >= .5][::max(1, int((s["opacity"] >= .5).sum()) // 300000)]
    out = {"opaque_splats_sampled": len(opaque)}
    for name, distance in [("", report_surfaces()[2].compute_distance(o3d.core.Tensor(opaque.astype(np.float32))).numpy())] + \
            ([("_to_lingbot", cKDTree(points).query(opaque, workers=-1)[0])] if points is not None else []):
        out |= {f"{q}_distance{name}_m": float(np.percentile(distance, p)) * metres for q, p in (("median", 50), ("p75", 75), ("p90", 90))}
    return out


def self_check():
    one = np.zeros((480, 640), bool)
    one[200, 100] = True
    hit = {tuple(p) for p in np.argwhere(excluded_full(one, (0, 0, 0, 0), dilate=0))}
    assert (300, 310) in hit <= {(300, 310), (301, 310), (300, 311), (301, 311)}, "clip pixel (100, 200) covers video [310, 311.5) x [300, 301.5)"
    grown = excluded_full(one, (0, 0, 0, 0))
    assert grown[300, 325] and not grown[300, 326] and not grown[:, :SIDE].any(), "dilated 15 px, side strips untouched"
    edge = one.copy()
    edge[240, 0] = True
    assert excluded_full(edge, (0, 0, 0, 0))[:, :SIDE].all() and not excluded_full(edge, (0, 0, 0, 0))[:, W - SIDE:].any(), "a person at the crop edge takes that side strip"
    assert excluded_full(np.zeros((480, 640), bool))[CAPTIONS[0]:CAPTIONS[1], CAPTIONS[2]:CAPTIONS[3]].all(), "caption box excluded"
    frame = np.full((H, W, 3), 120, np.uint8)
    frame[655:695, 600:900] = 30  # a dark caption box, its white text seen only right of the centre
    frame[665:685, 700:880:4] = 255
    assert caption_half_width(frame) == 876 - W // 2 and caption_half_width(np.full((H, W, 3), 120, np.uint8)) == 0, "text mirrored about the centre"

    k_clip = [377.5, 377.5, 319.5, 239.5]
    K, Kc = full_k(k_clip), np.array([[k_clip[0], 0, k_clip[2]], [0, k_clip[1], k_clip[3]], [0, 0, 1]])
    angle = .3
    c2w = np.eye(4)
    c2w[:3, :3] = [[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]]
    c2w[:3, 3] = [.4, -.2, 1.]
    pixel, depth = np.array([1000.5, 123.25]), 3.
    centre = c2w[:3, :3] @ (np.linalg.inv(K) @ [*pixel, 1] * depth) + c2w[:3, 3]
    records = splat_to_web.pack(centre[None].astype(np.float32), np.full((1, 3), .01, np.float32), np.full((1, 3), .5), np.array([.9]), np.array([[1., 0, 0, 0]]))
    back = splat_to_web.unpack(records.tobytes())["positions"][0].astype(np.float64)
    project = lambda Km, p: (lambda q: q[:2] / q[2])(Km @ (c2w[:3, :3].T @ (p - c2w[:3, 3])))
    assert np.abs(project(K, back) - pixel).max() < 1e-3, "a Gaussian written and read back projects onto its pixel with K_video"
    assert np.allclose(project(K, back), 1.5 * project(Kc, back) + [SIDE + .25, .25]), "K_video agrees with the clip-to-video pixel map"

    knots = np.arange(10.).reshape(5, 2) ** 2
    k = knots - knots.mean(0)
    t = timeline(knots, np.array([0, 4, 8, 31]))
    assert np.allclose(t[0], k[0]) and np.allclose(t[1], (k[0] + k[1]) / 2) and np.allclose(t[2], k[1]) and np.allclose(t[3], k[3] + (k[4] - k[3]) * 7 / 8), "linear between knots"
    assert np.allclose(timeline(knots + 5, np.array([4, 31])), timeline(knots, np.array([4, 31]))), "a shift shared by every knot changes nothing: no drift"
    used = np.array([True, True, False, True, True])
    moved = knots.copy()
    moved[2] += 100  # an unused knot (frames of another shot) must not shift the corrections of the used ones
    assert np.allclose(timeline(moved, np.array([4]), used=used), timeline(knots, np.array([4]), used=used)), "unused knots do not move the frame"
    assert all(i % HOLD_OUT == 0 for i in SHOW), "shown frames are held-out frames"
    path = np.tile(np.eye(4), (5, 1, 1))
    path[:, 2, 3] = np.arange(5.)  # walking along +z, image y down
    for view in orbit_views(path):
        eye, look = view[:3, 3], view[:3, 2]
        assert eye[1] < 0 and eye[2] < 0 and look[2] > 0 and look[1] > 0 and np.allclose(view[:3, :3].T @ view[:3, :3], np.eye(3)) \
            and np.isclose(np.linalg.det(view[:3, :3]), 1), "orbit cameras: raised, behind the start, looking along the path and down"
    xyz = np.array([[.001, 0, 0], [.004, 0, 0], [.012, 0, 0]], np.float32)
    seeds = voxel_seeds(xyz, np.eye(3, dtype=np.float32), np.array([.003, .002, .02], np.float32), .01)
    assert np.allclose(seeds[0], xyz[1:]) and np.allclose(seeds[2], [.005, .01]), "one seed per voxel, its finest sample, scale half its size or voxel"
    assert np.allclose(voxel_seeds(xyz, np.eye(3, dtype=np.float32), np.array([.003, .002, .02], np.float32), .01, .1)[2], [.001, .002]), "scale share"
    walk = np.tile(np.eye(4), (900, 1, 1))
    walk[:, 2, 3] = np.linspace(0, 2, 900)
    for view, (frame, pixel, depth, offset) in zip(moderate_views(walk, K), MODERATE):
        target = walk[frame][:3, :3] @ (np.linalg.inv(K) @ [*pixel, 1.] * depth) + walk[frame][:3, 3]
        eye = view[:3, 3]
        assert .1 <= np.linalg.norm(eye - walk[frame][:3, 3]) <= .2 and eye[1] < walk[frame][1, 3] and np.allclose(view[:3, 2], (target - eye) / np.linalg.norm(target - eye)) \
            and np.isclose(np.linalg.det(view[:3, :3]), 1), "moderate views: 0.1-0.2 off the path camera, raised, looking at its bench pixel"
    f = {"distance": np.array([.5, .5, 0., 0.]), "views": np.array([1, 9, 1, 9]), "ratio": np.array([1., 1, 1, 20]), "longest": np.full(4, .01),
         "opacity": np.array([.5, .5, .5, .05]), "aligned": np.zeros(4), "weight": np.array([0., 5, 0, 5])}
    rule = {"far": .035, "min_views": 3, "needle_ratio": 10, "needle_opacity": .1, "huge": .1, "huge_opacity": .1}
    assert drops(rule, f).tolist() == [True, False, False, True], "far and unseen, or a faint needle, is dropped; seen or on the surface stays"
    off = OFF
    assert drops({**off, "faint": 1.}, f).tolist() == [True, False, True, False] and drops({**off, "faint": 1., "faint_far": .1}, f).tolist() == [True, False, False, False], \
        "negligible weight is dropped, optionally only away from the surfaces"
    splat_to_web.self_check()
    print("splat train check passed: masks map to the video frame, K_video projects round-tripped splats onto their pixel, corrections stay zero-mean")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--clip", type=Path, default=CLIP)
    parser.add_argument("--droid-run", type=Path, default=DROID)
    parser.add_argument("--masks", type=Path, default=MASKS, help="dynamic masks SOURCEINDEX-*.png in clip-frame pixels")
    parser.add_argument("--mesh", type=Path, default=MESH, help="the report's fused room mesh (mono-anchored-mesh.ply): seeds, depth pull, cleanup distance")
    parser.add_argument("--fill", type=Path, default=FILL, help="fill_scene_holes.py textured-scene.glb of that mesh: its single-view-depth-fill* geometries")
    parser.add_argument("--scale", type=Path, default=SCALE, help="metric-scale.json of the depth run: metres per native unit")
    parser.add_argument("--captions", type=int, nargs=4, default=CAPTIONS, metavar=("Y0", "Y1", "X0", "X1"), help="burned-in caption band, video pixels (0 0 0 0: none)")
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--cap", type=int, default=1500000, help="most Gaussians trained and exported")
    parser.add_argument("--pose", action="store_true", help="refine camera poses (zero-mean over the clip)")
    parser.add_argument("--exposure", action="store_true", help="per-frame colour/exposure correction (zero-mean over the clip)")
    parser.add_argument("--ablation", action="store_true", help="train plain, +exposure, +exposure+pose; metrics only")
    parser.add_argument("--max-elongation", type=float, help="cap every Gaussian's longest axis at this multiple of its middle one")
    parser.add_argument("--depth-weight", type=float, default=0, help="weight of a clipped log-depth pull onto the report's surfaces (0: off)")
    parser.add_argument("--lingbot", type=Path, help="a LingBot dense map run dir (dense-points.glb, point-attributes.npz): seeds and depth pull from it")
    parser.add_argument("--seed-voxel", type=float, default=.005, help="with --lingbot: one seed per voxel of this size (native)")
    parser.add_argument("--seed-scale", type=float, default=.5, help="with --lingbot: seed scale as a share of max(sample size, voxel)")
    parser.add_argument("--compare", nargs="*", metavar="LABEL=PATH", help="splat files to render side by side in the fixed views (GPU), sheets in --output")
    parser.add_argument("--max-minutes", type=float, default=40, help="hard GPU budget of the call; training stops early to leave time to evaluate")
    parser.add_argument("--resume", help="run name on the volume whose checkpoints to continue")
    parser.add_argument("--clean", type=Path, help="a finished run's directory: score cleanup rules on its splats.splat (GPU)")
    parser.add_argument("--pick", help="with --clean: write splats-clean.* under this scored rule (local, no GPU)")
    parser.add_argument("--skip", nargs="*", default=[], metavar="A-B", help="clip frames (inclusive ranges) left out of training and evaluation")
    parser.add_argument("--train-scale", type=int, default=1, help="train on 1/N of the frame's width and height (N=2: half size); held-out scoring stays full size")
    parser.add_argument("--gpu", help="one Modal GPU type instead of the H100-first fallback list, e.g. A100-80GB")
    parser.add_argument("--output", type=Path)
    a = parser.parse_args()
    MESH, FILL, SCALE = a.mesh, a.fill, a.scale  # the defaults are ME340's; another clip passes its own
    self_check() if a.self_check else compare(a) if a.compare else pick(a) if a.pick else clean(a) if a.clean else run(a)
