"""M3 fast-path experiments E1 + E7 on ME340 (docs/phase2/FAST-PATH-PLAN.md section 7). Research probe, not a stage.

E1: DA3 any-view (unposed: predicts cameras + depth) on the walk shot's keyframes, full 16:9 frames.
    Timing (cold load, first and warm call, pure forward), peak VRAM; poses vs DROID (Sim3 ATE, rotation: agreement,
    not accuracy); depth vs the DA3-posed reference; GT-free photometric reprojection on keyframe pairs and on
    held-out frames, for this chain and for the reference chain (DROID poses + DA3-posed depth) on the same pixels.
E7: on each DA3 output, Open3D CUDA TSDF at 3 cm + point/mesh extraction, compared with the reference mesh; held-out
    depth rendered from the fused mesh; GPU lift of the reference SAM 2 masks on the keyframes, voxel-overlap merge.

Keyframes: 6-frame blocks over the walk shot; in each block the sharpest (Laplacian variance) frame among those that
have reference DA3-posed depth and SAM 2 masks (every 3rd frame), so every keyframe and held-out frame has a reference.
Metric units: DA3 -> DROID native by the trajectory Sim3, then the reference run's metres_per_native_unit (assumed 1.6 m
camera height); the fast path would get scale from its own floor rule, so absolute metres here are borrowed.

  modal run modal_apps/m3_exp_geometry.py --smoke-only              # T4: open3d CUDA + DA3 import check
  modal run modal_apps/m3_exp_geometry.py --models base,giant --out RUN_DIR
  python modal_apps/m3_exp_geometry.py                              # local self-check (numpy only)
"""
import io
import json
import time
from pathlib import Path

import modal
import numpy as np

PHASE2 = Path("/Users/adam/Desktop/panoptes-public/research-notes/phase2")
WALK = (226, 899)  # ME340 walk shot [start, end); 14-225 is the cut-away
BLOCK = 6  # keyframe every 6th frame = 5 fps
MODELS = {"giant": "depth-anything/DA3-GIANT-1.1", "base": "depth-anything/DA3-BASE"}
REVISIONS = {"depth-anything/DA3-GIANT-1.1": "72ee9f89ce4e50d704e9d55ee9c646ec8dc25a19",  # mono_room.py pins
             "depth-anything/DA3-BASE": "f4a6c9b3c95e41c82048423d3493a81ec3fa810e"}
TSDF_VOXEL, TSDF_DEPTH_MAX, EDGE_JUMP = 0.03, 15.0, 0.05
LIFT_VOXEL, MIN_PIXELS, MOVING, MATCH_MIN, MATCH_MAX, CONFIRMED = 0.05, 32, 0.5, 0.5, 0.2, 3
# raster (640x480 = source crop x 160..1120, scale 2/3) grid at [::2, ::2], in 1280x720 source pixel coordinates
GX = 160 + (2 * np.arange(320) + 0.5) * 1.5 - 0.5
GY = (2 * np.arange(240) + 0.5) * 1.5 - 0.5
PRICE = {"A100-80GB": 0.000694, "T4": 0.000164, "cpu_core": 0.0000131, "gib": 0.00000222}  # Modal list $/s

app = modal.App("panoptes-m3-exp-geometry")
volume = modal.Volume.from_name("moge3-hf-cache", create_if_missing=True)  # DA3 weights already cached by mono_room.py
image = (modal.Image.debian_slim(python_version="3.11").apt_install("git", "libgl1", "libglib2.0-0")
         .pip_install("torch", "torchvision", "xformers")
         .pip_install("git+https://github.com/ByteDance-Seed/Depth-Anything-3.git", "addict")  # same recipe as mono_room.py
         .apt_install("libgomp1").pip_install("open3d==0.19.0", "scipy")
         .env({"HF_HOME": "/cache/huggingface"}))


# ---------- pure numpy helpers (also run by the local self-check) ----------

def umeyama(src, dst):
    """dst ~ s R src + t (least squares)."""
    ms, md = src.mean(0), dst.mean(0)
    xs, xd = src - ms, dst - md
    u, sv, vt = np.linalg.svd(xd.T @ xs / len(src))
    d = np.diag([1, 1, np.sign(np.linalg.det(u @ vt))])
    r = u @ d @ vt
    s = np.trace(np.diag(sv) @ d) / (xs ** 2).sum(1).mean()
    return s, r, md - s * r @ ms


def align_sim3(c2w_src, c2w_dst):
    """dst ~ s R src + t with R from the camera orientations (chordal mean), s and t from the centres.

    Umeyama on centres alone leaves the roll about a nearly straight walk unconstrained; orientations fix it."""
    u, _, vt = np.linalg.svd(sum(d @ q.T for d, q in zip(c2w_dst[:, :3, :3], c2w_src[:, :3, :3])))
    r = u @ np.diag([1, 1, np.sign(np.linalg.det(u @ vt))]) @ vt
    x, y = c2w_src[:, :3, 3], c2w_dst[:, :3, 3]
    xc, yc = x - x.mean(0), y - y.mean(0)
    s = float(((xc @ r.T) * yc).sum() / (xc ** 2).sum())
    return s, r, y.mean(0) - s * r @ x.mean(0)


def angle_deg(r):
    return float(np.degrees(np.arccos(np.clip((np.trace(r) - 1) / 2, -1, 1))))


def interp_c2w(c2w_a, c2w_b, alpha):
    from scipy.spatial.transform import Rotation, Slerp
    rot = Slerp([0, 1], Rotation.from_matrix([c2w_a[:3, :3], c2w_b[:3, :3]]))([alpha]).as_matrix()[0]
    out = np.eye(4)
    out[:3, :3], out[:3, 3] = rot, (1 - alpha) * c2w_a[:3, 3] + alpha * c2w_b[:3, 3]
    return out


def pick_keyframes(candidates, sharpness):
    """Sharpest candidate in each BLOCK-frame block of the walk shot."""
    keys = []
    for start in range(WALK[0], WALK[1], BLOCK):
        block = [f for f in candidates if start <= f < start + BLOCK]
        if block:
            keys.append(max(block, key=lambda f: sharpness[f]))
    return keys


def self_check():
    rng = np.random.default_rng(0)
    src = rng.normal(size=(50, 3))
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    q *= np.sign(np.linalg.det(q))
    s, r, t = umeyama(src, 2.5 * src @ q.T + [1, 2, 3])
    assert abs(s - 2.5) < 1e-9 and np.allclose(r, q) and np.allclose(t, [1, 2, 3]) and angle_deg(r @ q.T) < 1e-6
    poses = np.tile(np.eye(4), (6, 1, 1))
    for i in range(6):
        poses[i, :3, :3] = np.linalg.qr(rng.normal(size=(3, 3)))[0] * [1, 1, 1]
        poses[i, :3, :3] *= np.sign(np.linalg.det(poses[i, :3, :3]))
        poses[i, :3, 3] = [i, 0, 0]  # a straight walk: Umeyama cannot fix the roll about x
    moved = poses.copy()
    moved[:, :3, :3] = q @ poses[:, :3, :3]
    moved[:, :3, 3] = 2.5 * poses[:, :3, 3] @ q.T + [1, 2, 3]
    s, r, t = align_sim3(poses, moved)
    assert abs(s - 2.5) < 1e-9 and np.allclose(r, q) and np.allclose(t, [1, 2, 3])
    a, b = np.eye(4), np.eye(4)
    b[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    b[:3, 3] = [2, 0, 0]
    mid = interp_c2w(a, b, .5)
    assert abs(angle_deg(mid[:3, :3]) - 45) < 1e-6 and np.allclose(mid[:3, 3], [1, 0, 0])
    assert pick_keyframes([226, 228, 231, 234], {226: 1, 228: 5, 231: 2, 234: 0}) == [228, 234]
    print("self-check ok")


# ---------- remote ----------

@app.function(image=image, gpu="T4", timeout=600, retries=0)
def smoke():
    import inspect
    import open3d as o3d
    import open3d.core as o3c
    import torch
    from depth_anything_3.api import DepthAnything3
    dev = o3c.Device("CUDA:0")
    vbg = o3d.t.geometry.VoxelBlockGrid(attr_names=("tsdf", "weight", "color"), attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
                                        attr_channels=((1), (1), (3)), voxel_size=.03, block_resolution=16, block_count=1000, device=dev)
    depth = o3d.t.geometry.Image(o3c.Tensor(np.full((48, 64), 2., np.float32), device=dev))
    color = o3d.t.geometry.Image(o3c.Tensor(np.full((48, 64, 3), .5, np.float32), device=dev))
    k = o3c.Tensor(np.array([[50., 0, 32], [0, 50, 24], [0, 0, 1]]))
    e = o3c.Tensor(np.eye(4))
    coords = vbg.compute_unique_block_coordinates(depth, k, e, depth_scale=1., depth_max=5.)
    vbg.integrate(coords, depth, color, k, e, depth_scale=1., depth_max=5.)
    return {"open3d": o3d.__version__, "o3d_cuda": o3c.cuda.is_available(), "points": len(vbg.extract_point_cloud(0.).point.positions),
            "torch": str(torch.__version__), "inference_sig": str(inspect.signature(DepthAnything3.inference)),
            "has_forward_hook": hasattr(DepthAnything3, "_run_model_forward")}


@app.function(image=image, gpu="A100-80GB", cpu=8, memory=32768, volumes={"/cache": volume}, timeout=2400, retries=0,
              max_containers=1, scaledown_window=120)
def run(mode, payload=None, model_key=None):
    entered = time.time()
    if mode == "ping":
        t = time.perf_counter()
        import open3d  # noqa: F401
        import torch
        from depth_anything_3.api import DepthAnything3  # noqa: F401
        torch.zeros(1, device="cuda")
        return {"entered_unix": entered, "imports_and_cuda_init_s": time.perf_counter() - t}
    return experiment(payload, model_key, entered)


def experiment(p, model_key, entered):
    import cv2
    import torch
    from depth_anything_3.api import DepthAnything3
    T = {}
    t = time.perf_counter()
    Path("/tmp/clip.mp4").write_bytes(p["mp4"])
    cap, frames = cv2.VideoCapture("/tmp/clip.mp4"), {}
    ref_frames = p["ref_frames"]
    timing_set = [int(round(f)) for f in np.linspace(WALK[0], WALK[1] - 1, 150)]
    keep = set(ref_frames) | set(timing_set)
    index = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if index in keep:
            frames[index] = np.ascontiguousarray(bgr[..., ::-1])
        index += 1
    T["decode_all_frames_s"] = time.perf_counter() - t
    t = time.perf_counter()
    sharp = {f: float(cv2.Laplacian(cv2.cvtColor(cv2.resize(frames[f], (640, 360)), cv2.COLOR_RGB2GRAY), cv2.CV_32F).var()) for f in ref_frames}
    keys = pick_keyframes(ref_frames, sharp)
    T["sharpness_and_pick_s"] = time.perf_counter() - t
    held = [f for f in ref_frames if f not in set(keys) and keys[0] < f < keys[-1]]
    first_of_block = pick_keyframes(ref_frames, {f: -f for f in ref_frames})  # the block's first candidate: what no sharpness pick gives
    info = {"frames_decoded": index, "keyframes": keys, "held_out": held, "n_keyframes": len(keys), "n_held_out": len(held),
            "sharpness_median_picked_vs_first": float(np.median([sharp[a] / max(sharp[b], 1e-6) for a, b in zip(keys, first_of_block)]))}

    torch.cuda.synchronize()
    t = time.perf_counter()
    model = DepthAnything3.from_pretrained(MODELS[model_key], revision=REVISIONS[MODELS[model_key]]).to("cuda").eval()
    torch.cuda.synchronize()
    T["model_load_cold_s"] = time.perf_counter() - t
    forward = {}
    if hasattr(model, "_run_model_forward"):
        inner = model._run_model_forward

        def timed(*a, **k):
            torch.cuda.synchronize()
            s = time.perf_counter()
            r = inner(*a, **k)
            torch.cuda.synchronize()
            forward["s"] = time.perf_counter() - s
            return r
        model._run_model_forward = timed

    def infer(ids, res, ray):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        s = time.perf_counter()
        with torch.inference_mode():
            out = model.inference([frames[f] for f in ids], process_res=res, use_ray_pose=ray)  # noqa: F821  (del model runs after the last call)
        torch.cuda.synchronize()
        return out, {"views": len(ids), "process_res": res, "use_ray_pose": ray, "call_s": time.perf_counter() - s,
                     "forward_s": forward.get("s"), "peak_alloc_gb": torch.cuda.max_memory_allocated() / 1e9,
                     "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1e9, "hw": list(out.depth.shape[1:])}

    runs, outs = {}, {}
    _, runs["first_call_504_cam"] = infer(keys, 504, False)
    outs["504_cam"], runs["504_cam"] = infer(keys, 504, False)
    outs["504_ray"], runs["504_ray"] = infer(keys, 504, True)
    _, runs["timing_150_504_cam"] = infer(timing_set, 504, False)
    if model_key == "giant":
        _, runs["first_call_672_cam"] = infer(keys, 672, False)
        outs["672_cam"], runs["672_cam"] = infer(keys, 672, False)
    del model
    torch.cuda.empty_cache()

    import traceback
    results, reference = {}, {}
    try:  # a failing evaluation must not lose the paid inference numbers
        ctx = Context(p, frames, keys, held)
    except Exception:
        ctx, reference = None, {"error": traceback.format_exc()}
    for i, (name, out) in enumerate(outs.items() if ctx else ()):
        try:
            results[name] = ctx.evaluate(out, twice=(i == 0), save_points=(name == "504_cam"))
        except Exception:
            results[name] = {"metrics": {"error": traceback.format_exc()}, "files": {}}
    try:
        reference = ctx.reference_chain() if ctx else reference
    except Exception:
        reference = {"error": traceback.format_exc()}
    return {"model": MODELS[model_key], "revision": REVISIONS[MODELS[model_key]], "entered_unix": entered, "timing": T,
            "info": info, "inference": runs, "results": {k: v["metrics"] for k, v in results.items()},
            "files": {f"{k}-{n}": b for k, v in results.items() for n, b in v["files"].items()},
            "reference_chain": reference, "remote_wall_s": time.time() - entered}


class Context:
    """Reference data on the GPU plus the evaluations every DA3 output gets."""

    def __init__(self, p, frames, keys, held):
        import cv2
        import open3d as o3d
        import torch
        self.p, self.keys, self.held, self.mpn = p, keys, held, p["metres_per_native"]
        self.ref_index = {f: i for i, f in enumerate(p["ref_frames"])}
        self.ref_depth = torch.from_numpy(p["ref_depth"].astype(np.float32)).cuda()  # (M,240,320) native units
        self.dyn = {f: cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_GRAYSCALE) > 0 for f, b in p["dynamic"].items()}  # 480x640
        self.dyn_grid = {f: torch.from_numpy(m[::2, ::2].copy()).cuda() for f, m in self.dyn.items()}
        self.gray = {f: torch.from_numpy(cv2.cvtColor(frames[f], cv2.COLOR_RGB2GRAY).astype(np.float32)).cuda() for f in p["ref_frames"]}
        self.rgb = frames
        self.droid = np.asarray(p["droid_c2w"], np.float64)
        fx, fy, cx, cy = p["raster_K"]  # 640x480 raster -> source pixels
        self.K_ref = np.array([[fx * 1.5, 0, 160 + (cx + .5) * 1.5 - .5], [0, fy * 1.5, (cy + .5) * 1.5 - .5], [0, 0, 1]])
        gx, gy = np.meshgrid(GX, GY)
        self.pix = torch.from_numpy(np.stack([gx.ravel(), gy.ravel(), np.ones(gx.size)])).float().cuda()  # (3, 76800)
        Path("/tmp/ref.ply").write_bytes(p["ref_mesh"])
        self.ref_mesh = o3d.io.read_triangle_mesh("/tmp/ref.ply")
        self.ref_mesh_m = o3d.geometry.TriangleMesh(self.ref_mesh).scale(self.mpn, center=(0, 0, 0))
        self.ref_simple = self.reference_tsdf()
        self.masks = {f: [cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_GRAYSCALE)[::2, ::2] > 0 for b in p["sam_masks"][f]] for f in keys}

    # --- sampling helpers ---
    def grid_depth(self, depth):
        """(N,H,W) depth at process resolution -> (N,240,320) at the raster grid (nearest)."""
        import torch
        h, w = depth.shape[1:]
        xs = torch.from_numpy(np.clip(np.rint((GX + .5) * w / 1280 - .5), 0, w - 1).astype(np.int64)).cuda()
        ys = torch.from_numpy(np.clip(np.rint((GY + .5) * h / 720 - .5), 0, h - 1).astype(np.int64)).cuda()
        return depth[:, ys][:, :, xs]

    def sample(self, gray, u, v):
        import torch
        grid = torch.stack([u / 1279 * 2 - 1, v / 719 * 2 - 1], -1)[None, None]
        return torch.nn.functional.grid_sample(gray[None, None], grid, align_corners=True)[0, 0, 0]

    def warp(self, z, K_i, c2w_i, c2w_j, K_j, f_i, f_j):
        """Photometric reprojection of the raster grid of frame i into frame j: (abs error, valid), both (76800,)."""
        import torch
        c = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device="cuda")
        X = (c(np.linalg.inv(K_i)) @ self.pix) * z.reshape(1, -1)
        rel = np.linalg.inv(c2w_j) @ c2w_i
        Xj = c(rel[:3, :3]) @ X + c(rel[:3, 3])[:, None]
        uv = c(K_j) @ Xj
        u, v = uv[0] / uv[2].clamp(min=1e-6), uv[1] / uv[2].clamp(min=1e-6)
        valid = (z.reshape(-1) > 0) & (Xj[2] > 1e-6) & (u >= 0) & (u <= 1279) & (v >= 0) & (v <= 719)
        src = self.sample(self.gray[f_i], self.pix[0], self.pix[1])
        return (self.sample(self.gray[f_j], u.clamp(0, 1279), v.clamp(0, 719)) - src).abs(), valid

    def identity_error(self, f_i, f_j):
        return (self.sample(self.gray[f_j], self.pix[0], self.pix[1]) - self.sample(self.gray[f_i], self.pix[0], self.pix[1])).abs()

    # --- the reference chain alone (same metrics where they apply) ---
    def reference_chain(self):
        return {"note": "DROID poses + DA3-posed depth (240 views) + reference mesh built from all views, held-out frames included",
                "held_out_depth_from_reference_mesh": self.render_depth_eval(self.ref_mesh, lambda h: self.droid[h], lambda h: self.K_ref, 1.0),
                "same_method_reference_tsdf": {
                    "timing": self.ref_simple["timing"], "points": self.ref_simple["n_points"],
                    "vs_published_mesh": self.surface(self.ref_simple, lambda x: x),
                    "held_out_depth": self.render_depth_eval(self.ref_simple["mesh"], lambda h: self.metric_pose(self.droid[h], self.mpn),
                                                             lambda h: self.K_ref, 1 / self.mpn)}}

    def render_depth_eval(self, mesh, c2w_of, K_of, to_native):
        """Raycast a legacy mesh into every held-out frame's raster grid; compare with the reference depth there."""
        import open3d as o3d
        import open3d.core as o3c
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3c.Tensor(np.asarray(mesh.vertices, np.float32)), o3c.Tensor(np.asarray(mesh.triangles, np.uint32)))
        rel, cover, total = [], 0, 0
        pix = self.pix.cpu().numpy()
        for h in self.held:
            c2w, K = c2w_of(h), K_of(h)
            d = c2w[:3, :3] @ (np.linalg.inv(K) @ pix)
            rays = np.concatenate([np.repeat(c2w[:3, 3][None], d.shape[1], 0), d.T], 1).astype(np.float32)
            z = scene.cast_rays(o3c.Tensor(rays))["t_hit"].numpy() * to_native
            ref = self.ref_depth[self.ref_index[h]].reshape(-1).cpu().numpy()
            ok = (ref > 0) & ~self.dyn_grid[h].reshape(-1).cpu().numpy()
            hit = ok & np.isfinite(z)
            total, cover = total + ok.sum(), cover + hit.sum()
            rel.append(np.abs(z[hit] - ref[hit]) / ref[hit])
        rel = np.concatenate(rel)
        return {"coverage": float(cover / max(total, 1)), "absrel_median": float(np.median(rel)),
                "within_5pct": float((rel < .05).mean()), "within_10pct": float((rel < .10).mean()), "frames": len(self.held)}

    # --- one DA3 output ---
    def evaluate(self, out, twice, save_points):
        import torch
        keys, m, files = self.keys, {}, {}
        n, h, w = out.depth.shape
        w2c = np.tile(np.eye(4), (n, 1, 1))
        w2c[:, :3] = np.asarray(out.extrinsics, np.float64)[:, :3]
        c2w = np.linalg.inv(w2c)
        K = np.asarray(out.intrinsics, np.float64).copy()
        K[:, 0] *= 1280 / w
        K[:, 1] *= 720 / h  # source pixel coordinates
        depth = torch.from_numpy(np.asarray(out.depth, np.float32)).cuda()

        # poses vs DROID (agreement)
        droid = self.droid[keys]
        s0, R0, t0 = umeyama(c2w[:, :3, 3], droid[:, :3, 3])
        err0 = np.linalg.norm((s0 * (R0 @ c2w[:, :3, 3].T)).T + t0 - droid[:, :3, 3], axis=1)
        rot0 = [angle_deg(droid[i, :3, :3].T @ R0 @ c2w[i, :3, :3]) for i in range(n)]
        s, R, t = align_sim3(c2w, droid)  # used for everything downstream
        aligned = (s * (R @ c2w[:, :3, 3].T)).T + t
        err = np.linalg.norm(aligned - droid[:, :3, 3], axis=1)
        rot = [angle_deg(droid[i, :3, :3].T @ R @ c2w[i, :3, :3]) for i in range(n)]
        rel_rot = [abs(angle_deg(c2w[i, :3, :3].T @ c2w[i + 1, :3, :3]) - angle_deg(droid[i, :3, :3].T @ droid[i + 1, :3, :3])) for i in range(n - 1)]
        rel_rot_err = [angle_deg((droid[i, :3, :3].T @ droid[i + 1, :3, :3]).T @ (c2w[i, :3, :3].T @ c2w[i + 1, :3, :3])) for i in range(n - 1)]
        path = float(np.linalg.norm(np.diff(droid[:, :3, 3], axis=0), axis=1).sum())
        m["pose_vs_droid"] = {"ate_rmse_native": float(np.sqrt((err ** 2).mean())), "ate_rmse_m": float(np.sqrt((err ** 2).mean()) * self.mpn),
                              "ate_max_m": float(err.max() * self.mpn), "ate_over_path_length": float(np.sqrt((err ** 2).mean()) / path),
                              "path_length_m": path * self.mpn, "rot_err_deg_median": float(np.median(rot)), "rot_err_deg_max": float(np.max(rot)),
                              "rel_rot_err_deg_median": float(np.median(rel_rot_err)), "rel_rot_err_deg_p90": float(np.percentile(rel_rot_err, 90)),
                              "rel_rot_angle_diff_deg_median": float(np.median(rel_rot)), "sim3_scale_da3_to_native": float(s),
                              "alignment": "R from orientations, s and t from centres (align_sim3)",
                              "umeyama_centres_only": {"ate_rmse_m": float(np.sqrt((err0 ** 2).mean()) * self.mpn), "rot_err_deg_median": float(np.median(rot0)),
                                                       "rot_err_deg_max": float(np.max(rot0))},
                              "da3_fx_source_px_median": float(np.median(K[:, 0, 0])), "da3_fx_source_px_p10_p90": np.percentile(K[:, 0, 0], [10, 90]).tolist(),
                              "clip_fx_source_px_moge": float(self.K_ref[0, 0])}
        s_m = s * self.mpn  # DA3 units -> metres (borrowed scale)

        # depth vs the DA3-posed reference on the keyframes (4:3 centre, raster grid)
        zg = self.grid_depth(depth)
        rows = torch.tensor([self.ref_index[f] for f in keys], device="cuda")
        ref = self.ref_depth[rows]
        ok = (ref > 0) & (zg > 0) & ~torch.stack([self.dyn_grid[f] for f in keys])
        ratio = (zg * s / ref)[ok]
        per_frame = torch.stack([torch.median((ref[i] / zg[i])[ok[i]]) for i in range(n)])
        ratio_pf = (zg * per_frame[:, None, None] / ref)[ok]
        m["depth_vs_da3_posed_keyframes"] = {
            "absrel_median_global_scale": float((ratio - 1).abs().median()), "delta_1.05_global": float((torch.maximum(ratio, 1 / ratio) < 1.05).float().mean()),
            "delta_1.25_global": float((torch.maximum(ratio, 1 / ratio) < 1.25).float().mean()),
            "absrel_median_per_frame_scale": float((ratio_pf - 1).abs().median()),
            "per_frame_scale_spread_mad_rel": float(((per_frame / s - 1).abs()).median()) if s else None}

        # photometric reprojection: keyframe pairs and held-out frames, both chains on the same pixels
        def photometric(pairs, da3_pose_of, ref_pose_of):
            e_da3, e_ref, e_mix, e_id, n_px = [], [], [], [], 0
            for f_i, f_j in pairs:
                i, (pose_j, K_j) = keys.index(f_i), da3_pose_of(f_j)
                a, va = self.warp(zg[i], K[i], c2w[i], pose_j, K_j, f_i, f_j)
                b, vb = self.warp(self.ref_depth[self.ref_index[f_i]], self.K_ref, self.droid[f_i], ref_pose_of(f_j), self.K_ref, f_i, f_j)
                x, vx = self.warp(zg[i] * s, K[i], self.droid[f_i], ref_pose_of(f_j), K[i], f_i, f_j)  # any-view depth on DROID cameras
                v = va & vb & vx & ~self.dyn_grid[f_i].reshape(-1)
                e_da3.append(a[v])
                e_ref.append(b[v])
                e_mix.append(x[v])
                e_id.append(self.identity_error(f_i, f_j)[v])
                n_px += int(v.sum())
            a, b, c, x = (torch.cat(e) for e in (e_da3, e_ref, e_id, e_mix))
            return {"da3_mean_L1": float(a.mean()), "ref_mean_L1": float(b.mean()), "identity_mean_L1": float(c.mean()),
                    "da3_depth_on_droid_cameras_mean_L1": float(x.mean()),
                    "da3_over_ref": float(a.mean() / b.mean()), "da3_median_L1": float(a.median()), "ref_median_L1": float(b.median()),
                    "pixels": n_px, "pairs": len(pairs)}

        m["photometric_keyframe_pairs"] = photometric(list(zip(keys[:-1], keys[1:])), lambda f: (c2w[keys.index(f)], K[keys.index(f)]), lambda f: self.droid[f])

        def interp(f):
            j = int(np.searchsorted(keys, f))
            a, b = keys[j - 1], keys[j]
            return interp_c2w(c2w[j - 1], c2w[j], (f - a) / (b - a)), K[j - 1] if f - a <= b - f else K[j]
        nearest = lambda f: min(keys, key=lambda k: abs(k - f))
        m["photometric_held_out"] = photometric([(nearest(f), f) for f in self.held], interp, lambda f: self.droid[f])

        # E7 TSDF (full 16:9 and the reference's 4:3 centre)
        tsdf = self.tsdf(depth * s_m, K, c2w, s_m, centre_only=False)
        if twice:
            first = tsdf["timing"]
            tsdf = self.tsdf(depth * s_m, K, c2w, s_m, centre_only=False)
            tsdf["timing_first_call"] = first
        centre = self.tsdf(depth * s_m, K, c2w, s_m, centre_only=True)
        to_nm = lambda x: x @ R.T + t * self.mpn  # metric DA3 frame -> metric DROID frame
        m["tsdf_full"] = {"timing": tsdf["timing"], "timing_first_call": tsdf.get("timing_first_call"), **self.surface(tsdf, to_nm)}
        m["tsdf_centre_4x3"] = {"timing": centre["timing"], **self.surface(centre, to_nm)}
        m["same_method_reference_tsdf"] = {"note": "same fusion code on DA3-posed depth + DROID cameras, same keyframes, 4:3 raster grid",
                                           "full_vs_it": self.surface(tsdf, to_nm, self.ref_simple["mesh"]),
                                           "centre_4x3_vs_it": self.surface(centre, to_nm, self.ref_simple["mesh"])}
        m["held_out_depth_from_tsdf"] = {
            "interpolated_da3_pose": self.render_depth_eval(tsdf["mesh"], lambda f: self.metric_pose(interp(f)[0], s_m), lambda f: interp(f)[1], 1 / self.mpn),
            "droid_pose_via_sim3": self.render_depth_eval(tsdf["mesh"], lambda f: self.droid_in_da3(f, R, t), lambda f: self.K_ref, 1 / self.mpn)}
        if save_points:
            buf = io.BytesIO()
            np.savez_compressed(buf, points_m_droid_frame=to_nm(np.asarray(tsdf["points"])).astype(np.float32), colors=tsdf["colors"],
                                keyframes=np.array(keys), c2w_da3=c2w, K_source_px=K, sim3=np.array([s, *R.ravel(), *t]), metres_per_native=self.mpn)
            files["tsdf-points.npz"] = buf.getvalue()

        # E7 mask lift
        lift = self.lift(zg * s_m, K, c2w, s_m)
        m["mask_lift"] = {k: v for k, v in lift.items() if k != "objects"}
        m["mask_lift"].update(self.object_recall(lift["objects"], to_nm))
        files["objects.json"] = json.dumps([{**o, "centroid_m_droid_frame": to_nm(np.array(o["centroid_m"])).tolist()} for o in lift["objects"]]).encode()
        return {"metrics": m, "files": files}

    def metric_pose(self, c2w, s_m):
        out = c2w.copy()
        out[:3, 3] *= s_m
        return out

    def droid_in_da3(self, f, R, t):
        """DROID camera of frame f in the metric DA3 frame (inverse of the trajectory Sim3)."""
        out = np.eye(4)
        out[:3, :3] = R.T @ self.droid[f][:3, :3]
        out[:3, 3] = R.T @ (self.droid[f][:3, 3] * self.mpn - t * self.mpn)
        return out

    def tsdf(self, depth_m, K, c2w, s_m, centre_only):
        """Open3D CUDA VoxelBlockGrid at TSDF_VOXEL on metric DA3 depth; people (dynamic masks, 4:3 centre) and depth edges removed."""
        import torch
        n, h, w = depth_m.shape
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        # dynamic mask lookup: process pixel -> raster pixel (only the 4:3 centre has masks)
        xs = (np.arange(w) + .5) * 1280 / w - .5
        ys = (np.arange(h) + .5) * 720 / h - .5
        u = np.rint((xs - 160 + .5) / 1.5 - .5).astype(int)
        v = np.clip(np.rint((ys + .5) / 1.5 - .5).astype(int), 0, 479)
        inside = (u >= 0) & (u < 640)
        dyn = torch.from_numpy(np.stack([self.dyn[f][v][:, np.clip(u, 0, 639)] & inside[None] for f in self.keys])).cuda()
        d = depth_m.clone()
        d[dyn] = 0
        if centre_only:
            d[:, :, ~torch.from_numpy(inside).cuda()] = 0
        edge_filter(d)
        colors = torch.from_numpy(np.stack([cv2_resize(self.rgb[f], w, h) for f in self.keys]).astype(np.float32) / 255).cuda()
        torch.cuda.synchronize()
        out = fuse(d, K * [[w / 1280], [h / 720], [1]], np.stack([self.metric_pose(c, s_m) for c in c2w]), colors)
        prep = time.perf_counter() - t0 - out["timing"]["total_gpu_s"]
        out["timing"] = {"mask_and_edge_filter_s": prep, **out["timing"], "fusion_plus_points_s": prep + out["timing"]["integrate_s"] + out["timing"]["extract_points_s"]}
        out["timing"]["total_gpu_s"] += prep
        return out

    def reference_tsdf(self):
        """Control: the same fusion on the reference chain (DA3-posed depth on DROID cameras, same keyframes, 4:3 raster grid)."""
        import torch
        rows = [self.ref_index[f] for f in self.keys]
        d = self.ref_depth[rows] * self.mpn
        d[torch.stack([self.dyn_grid[f] for f in self.keys])] = 0
        edge_filter(d)
        k = self.K_ref.copy()
        k[0], k[1] = k[0] / 3, k[1] / 3
        k[0, 2], k[1, 2] = (self.K_ref[0, 2] - GX[0]) / 3, (self.K_ref[1, 2] - GY[0]) / 3  # grid index = (source px - first grid px) / 3
        c2w = self.droid[self.keys].copy()
        c2w[:, :3, 3] *= self.mpn
        xi, yi = np.rint(GX).astype(int), np.rint(GY).astype(int)
        colors = torch.from_numpy(np.stack([self.rgb[f][yi][:, xi] for f in self.keys]).astype(np.float32) / 255).cuda()
        return fuse(d, np.repeat(k[None], len(rows), 0), c2w, colors)

    def surface(self, tsdf, to_nm, ref_mesh=None):
        """Fused surface vs a reference mesh (default: the published one), in metres in the DROID frame (trajectory Sim3, then + rigid ICP)."""
        import open3d as o3d
        import open3d.core as o3c
        ref_mesh = self.ref_mesh_m if ref_mesh is None else ref_mesh
        ours = to_nm(tsdf["points"].astype(np.float64))
        mesh = o3d.geometry.TriangleMesh(tsdf["mesh"])
        mesh.vertices = o3d.utility.Vector3dVector(to_nm(np.asarray(mesh.vertices)))
        o3d.utility.random.seed(0)
        ref_pts = np.asarray(ref_mesh.sample_points_uniformly(300000).points)

        def dist_to(m, q):
            scene = o3d.t.geometry.RaycastingScene()
            scene.add_triangles(o3c.Tensor(np.asarray(m.vertices, np.float32)), o3c.Tensor(np.asarray(m.triangles, np.uint32)))
            return scene.compute_distance(o3c.Tensor(q.astype(np.float32))).numpy()

        ref_lo, ref_hi = np.asarray(ref_mesh.vertices).min(0), np.asarray(ref_mesh.vertices).max(0)

        def scores(ours_pts, ours_mesh):
            acc, comp = dist_to(ref_mesh, ours_pts), dist_to(ours_mesh, ref_pts)
            box = np.all((ours_pts > ref_lo - .25) & (ours_pts < ref_hi + .25), axis=1)  # the reference only covers what it kept
            out = {"accuracy_median_m": float(np.median(acc)), "completeness_median_m": float(np.median(comp)),
                   "share_of_ours_in_reference_box": float(box.mean()), "accuracy_median_in_reference_box_m": float(np.median(acc[box])),
                   "precision_5cm_in_reference_box": float((acc[box] < .05).mean()), "precision_10cm_in_reference_box": float((acc[box] < .10).mean())}
            for th in (.05, .10):
                p, r = float((acc < th).mean()), float((comp < th).mean())
                out[f"precision_{int(th * 100)}cm"], out[f"recall_{int(th * 100)}cm"], out[f"fscore_{int(th * 100)}cm"] = p, r, 2 * p * r / max(p + r, 1e-9)
            return out
        res = {"points": len(ours), "triangles": tsdf["n_triangles"], "sim3_from_trajectory": scores(ours, mesh)}
        src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(ours)).voxel_down_sample(.05)  # ICP speed only
        dst = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(ref_pts))
        icp = o3d.pipelines.registration.registration_icp(src, dst, .15, np.eye(4))
        T = icp.transformation
        mesh.transform(T)
        res["plus_rigid_icp"] = {**scores(ours @ T[:3, :3].T + T[:3, 3], mesh), "icp_translation_m": float(np.linalg.norm(T[:3, 3])),
                                 "icp_rotation_deg": angle_deg(T[:3, :3])}
        return res

    def lift(self, zg_m, K, c2w, s_m):
        """SAM 2 keyframe masks -> 3D voxels (GPU) -> merge by voxel overlap across frames -> objects."""
        import torch
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        keys = self.keys
        stack, frame_of = [], []
        for i, f in enumerate(keys):
            for mk in self.masks[f]:
                stack.append(mk)
                frame_of.append(i)
        masks_np = np.stack(stack)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        masks = torch.from_numpy(masks_np).cuda()
        frame_t = torch.tensor(frame_of, device="cuda")
        dyn = torch.stack([self.dyn_grid[f] for f in keys])[frame_t]
        valid_depth = (zg_m > 0)[frame_t]
        moving = (masks & dyn).sum((1, 2)) >= MOVING * masks.sum((1, 2)).clamp(min=1)
        masks = masks & valid_depth & ~dyn
        keep = ~moving & (masks.sum((1, 2)) >= MIN_PIXELS)
        idx = torch.nonzero(keep).squeeze(1)
        mid, vy, vx = torch.nonzero(masks[idx], as_tuple=True)
        f = frame_t[idx][mid]
        Kinv = torch.as_tensor(np.linalg.inv(K), dtype=torch.float32, device="cuda")[f]
        pix = torch.stack([self.pix[0].reshape(240, 320)[vy, vx], self.pix[1].reshape(240, 320)[vy, vx], torch.ones_like(vx, dtype=torch.float32)], 1)
        cam = (Kinv @ pix[:, :, None])[:, :, 0] * zg_m[f, vy, vx][:, None]
        pose = torch.as_tensor(np.stack([self.metric_pose(c, s_m) for c in c2w]), dtype=torch.float32, device="cuda")[f]
        world = (pose[:, :3, :3] @ cam[:, :, None])[:, :, 0] + pose[:, :3, 3]
        ijk = torch.floor(world / LIFT_VOXEL).long()
        B, OFF = 1 << 21, 1 << 20
        key = ((ijk[:, 0] + OFF) * B + (ijk[:, 1] + OFF)) * B + (ijk[:, 2] + OFF)
        vox, vid = torch.unique(key, return_inverse=True)
        pair = torch.unique(mid * len(vox) + vid)
        pm, pv = pair // len(vox), pair % len(vox)
        size = torch.bincount(pm, minlength=len(idx)).float()
        A = torch.sparse_coo_tensor(torch.stack([pm, pv]), torch.ones_like(pm, dtype=torch.float32), (len(idx), len(vox))).coalesce()
        inter = torch.sparse.mm(A, A.t()).coalesce()
        (a, b), c = inter.indices(), inter.values()
        fr = frame_t[idx]
        small, large = torch.minimum(size[a], size[b]), torch.maximum(size[a], size[b])
        edge = (a < b) & (fr[a] != fr[b]) & (c >= MATCH_MIN * small) & (c >= MATCH_MAX * large)
        a, b = a[edge].cpu().numpy(), b[edge].cpu().numpy()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        ncomp, label = connected_components(coo_matrix((np.ones(len(a)), (a, b)), shape=(len(idx), len(idx))), directed=False)
        t2 = time.perf_counter()
        lab = torch.from_numpy(label).cuda()
        ov = torch.unique(lab[pm] * len(vox) + pv)
        oc, ovid = ov // len(vox), ov % len(vox)
        centre = torch.stack([(vox // (B * B)) - OFF, (vox // B) % B - OFF, vox % B - OFF], 1).float()[ovid] * LIFT_VOXEL + LIFT_VOXEL / 2
        nvox = torch.bincount(oc, minlength=ncomp).float()
        cent = torch.zeros(ncomp, 3, device="cuda").index_add_(0, oc, centre) / nvox.clamp(min=1)[:, None]
        lo = torch.full((ncomp, 3), 1e9, device="cuda").scatter_reduce(0, oc[:, None].expand(-1, 3), centre, "amin")
        hi = torch.full((ncomp, 3), -1e9, device="cuda").scatter_reduce(0, oc[:, None].expand(-1, 3), centre, "amax")
        nframes = torch.bincount(torch.unique(lab * len(keys) + fr) // len(keys), minlength=ncomp)
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        nmask, nframes, nvox, cent, ext = (x.cpu().numpy() for x in (torch.bincount(lab, minlength=ncomp), nframes, nvox, cent, hi - lo + LIFT_VOXEL))
        objects = [{"id": i, "frames": int(nframes[i]), "masks": int(nmask[i]), "voxels": int(nvox[i]), "centroid_m": cent[i].tolist(),
                    "extent_m": ext[i].tolist(), "confirmed": bool(nframes[i] >= CONFIRMED)} for i in range(ncomp)]
        return {"timing": {"gpu_lift_voxelise_overlap_s": t1 - t0, "cpu_components_s": t2 - t1, "gpu_object_stats_s": t3 - t2, "total_s": t3 - t0},
                "masks_in": len(stack), "masks_dropped_moving": int(moving.sum()), "masks_dropped_small": int((~moving & ~keep).sum()),
                "masks_lifted": len(idx), "points_lifted": int(len(mid)), "voxels": int(len(vox)), "overlap_pairs": int(len(c)), "edges": int(len(a)),
                "objects": objects, "n_objects": ncomp, "n_confirmed": int(sum(o["confirmed"] for o in objects)),
                "config": {"voxel_m": LIFT_VOXEL, "min_pixels_320x240": MIN_PIXELS, "match_overlap_of_smaller": MATCH_MIN,
                           "min_overlap_of_larger": MATCH_MAX, "confirmed_min_frames": CONFIRMED}}

    def object_recall(self, objects, to_nm):
        ref = self.p["ref_entities"]
        ours = np.array([to_nm(np.array(o["centroid_m"])) for o in objects if o["confirmed"]]).reshape(-1, 3)
        out = {"reference_entities_walk_only": len(ref)}
        if not len(ours) or not ref:
            return out
        rc = np.array([e["centroid_native"] for e in ref]) * self.mpn
        d = np.linalg.norm(rc[:, None] - ours[None], axis=2)
        near_ref, near_ours = d.min(1), d.min(0)
        clear = np.array([e["status"] == "clear" for e in ref])
        for th in (.3, .5):
            out[f"reference_recall_{th}m"] = float((near_ref < th).mean())
            out[f"reference_clear_named_recall_{th}m"] = float((near_ref[clear] < th).mean()) if clear.any() else None
            out[f"ours_confirmed_near_reference_{th}m"] = float((near_ours < th).mean())
        out["reference_centroid_offset_median_m"] = float(np.median(near_ref))
        return out


def edge_filter(d):
    """Drop depth pixels whose 4-neighbourhood jumps by more than EDGE_JUMP (flying pixels), in place."""
    import torch
    pad = torch.nn.functional.pad(d[:, None], (1, 1, 1, 1), mode="replicate")[:, 0]
    neighbours = torch.stack([pad[:, 1:-1, :-2], pad[:, 1:-1, 2:], pad[:, :-2, 1:-1], pad[:, 2:, 1:-1]])
    d[((neighbours - d).abs().amax(0) > EDGE_JUMP * d)] = 0


def fuse(d, K_pix, c2w_m, colors):
    """Open3D CUDA VoxelBlockGrid TSDF: d (N,H,W) metres on the GPU, K_pix (N,3,3) at H,W, c2w_m (N,4,4) metric, colors (N,H,W,3) in [0,1]."""
    import open3d as o3d
    import open3d.core as o3c
    import torch.utils.dlpack as tdl
    o3c.cuda.synchronize()
    t1 = time.perf_counter()
    vbg = o3d.t.geometry.VoxelBlockGrid(attr_names=("tsdf", "weight", "color"), attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
                                        attr_channels=((1), (1), (3)), voxel_size=TSDF_VOXEL, block_resolution=16, block_count=80000,
                                        device=o3c.Device("CUDA:0"))
    for i in range(len(d)):
        di = o3d.t.geometry.Image(o3c.Tensor.from_dlpack(tdl.to_dlpack(d[i].contiguous())))
        ci = o3d.t.geometry.Image(o3c.Tensor.from_dlpack(tdl.to_dlpack(colors[i].contiguous())))
        k = o3c.Tensor(np.asarray(K_pix[i], np.float64))
        e = o3c.Tensor(np.linalg.inv(c2w_m[i]))
        coords = vbg.compute_unique_block_coordinates(di, k, e, depth_scale=1., depth_max=TSDF_DEPTH_MAX)
        vbg.integrate(coords, di, ci, k, e, depth_scale=1., depth_max=TSDF_DEPTH_MAX)
    o3c.cuda.synchronize()
    t2 = time.perf_counter()
    pcd = vbg.extract_point_cloud(weight_threshold=3.)
    o3c.cuda.synchronize()
    t3 = time.perf_counter()
    mesh = vbg.extract_triangle_mesh(weight_threshold=3.)
    o3c.cuda.synchronize()
    t4 = time.perf_counter()
    points = pcd.point.positions.cpu().numpy()
    legacy = mesh.to_legacy()
    return {"timing": {"integrate_s": t2 - t1, "extract_points_s": t3 - t2, "extract_mesh_s": t4 - t3, "total_gpu_s": t4 - t1,
                       "views": len(d), "voxel_m": TSDF_VOXEL},
            "points": points, "colors": (pcd.point.colors.cpu().numpy() * 255).clip(0, 255).astype(np.uint8), "mesh": legacy,
            "n_points": len(points), "n_triangles": len(legacy.triangles)}


def cv2_resize(img, w, h):
    import cv2
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


# ---------- local ----------

def payload():
    runs = PHASE2 / "runs"
    mono, sam = runs / "da3-posed-me340-223-shotc/mono", runs / "me340-masks-194/object-a"
    ref_frames = sorted(int(q.stem) for q in mono.glob("*.npz") if WALK[0] <= int(q.stem) < WALK[1] and (sam / f"frame-{int(q.stem):05d}").is_dir())
    names = json.loads((runs / "me340-entity-names-200/names.json").read_text())
    omap = json.loads((runs / "me340-entity-names-200/object-map.json").read_text())
    entities = [{"id": e["entityId"], "centroid_native": e["centroidNative"], "status": names.get(e["entityId"], {}).get("status"),
                 "category": names.get(e["entityId"], {}).get("category")}  # seen >= 3 times in the walk shot, never in the cut-away
                for e in omap["entities"] if e["entityId"].startswith("object-") and sum(WALK[0] <= f < WALK[1] for f in e["sourceFrames"]) >= 3
                and not any(14 <= f < WALK[0] for f in e["sourceFrames"])]
    clip = json.loads((PHASE2 / "data/clips/me340-165/clip.json").read_text())
    return {"mp4": (PHASE2 / "data/clips/me340-165/source-full.mp4").read_bytes(), "ref_frames": ref_frames,
            "ref_depth": np.stack([np.load(mono / f"{f:05d}.npz")["depth"][::2, ::2] for f in ref_frames]).astype(np.float16),
            "dynamic": {f: (runs / f"me340-dynamic-masks-188/masks/{f:05d}-0.png").read_bytes() for f in ref_frames},
            "sam_masks": {f: [q.read_bytes() for q in sorted((sam / f"frame-{f:05d}").glob("instance-*-mask.png"))] for f in ref_frames},
            "ref_mesh": (runs / "da3-posed-me340-223-shotc/mono-anchored-mesh.ply").read_bytes(),
            "droid_c2w": np.load(runs / "droid-me340-165-171/prediction.npz")["poses_c2w"].astype(np.float64),
            "raster_K": clip["K"], "ref_entities": entities,
            "metres_per_native": json.loads((runs / "da3-posed-me340-223-shotc/metric-scale.json").read_text())["metres_per_native_unit"]}


@app.local_entrypoint()
def main(smoke_only: bool = False, models: str = "base,giant", out: str = ""):
    if smoke_only:
        t = time.time()
        print(json.dumps(smoke.remote(), indent=1))
        print(json.dumps({"smoke_wall_s": time.time() - t}))
        return
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=False)  # never reuse a run dir
    started = t = time.time()
    p = payload()
    print(json.dumps({"payload_build_s": time.time() - t, "ref_frames": len(p["ref_frames"]),
                      "payload_mb": sum(len(v) if isinstance(v, bytes) else 0 for v in p.values()) / 1e6 + p["ref_depth"].nbytes / 1e6}))
    t = time.time()
    ping = run.remote("ping")
    ping.update(local_wall_s=time.time() - t)
    record = {"app_id": app.app_id, "ping_cold_container": ping, "calls": {}}
    (out_dir / "record.json").write_text(json.dumps(record, indent=1))
    for key in models.split(","):
        t = time.time()
        result = run.remote("experiment", p, key)
        wall = time.time() - t
        for name, data in result.pop("files").items():
            (out_dir / f"{key}-{name}").write_bytes(data)
        result.update(local_wall_s=wall, upload_and_overhead_s_estimate=wall - result["remote_wall_s"])
        record["calls"][key] = result
        record["app_wall_s_so_far"] = time.time() - started
        (out_dir / "record.json").write_text(json.dumps(record, indent=1))
        print(json.dumps({key: {"local_wall_s": wall, "remote_wall_s": result["remote_wall_s"], "timing": result["timing"],
                                "inference": result["inference"]}}, indent=1))


if __name__ == "__main__":
    self_check()
