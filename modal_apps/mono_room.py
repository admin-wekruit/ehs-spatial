"""Pretrained monocular depth, anchored to an existing DROID run, fused with its cameras.

Hypothesis: DROID cameras are reliable (Sim3 ATE ~0.04 m) but its flow depth has no
cross-view support on textureless walls/floor. MoGe depth with the *known* FOV is dense;
one scale per keyframe, fitted only on DROID-supported pixels, puts it in DROID units.
Cameras, K and scale anchors come from DROID. No GT or sensor depth enters `infer`/`fuse`;
`evaluate` reads sensor depth strictly as an independent check.

  python modal_apps/mono_room.py infer --droid-run RUN --output NEW_RUN [--stride 15]
  python modal_apps/mono_room.py fuse --droid-run RUN --support NPZ --output RUN_FROM_INFER
  python modal_apps/mono_room.py evaluate --droid-run RUN --support NPZ --output RUN_FROM_INFER
  python modal_apps/mono_room.py self-check
"""
import argparse
import io
import json
from pathlib import Path
import sys
import time

import modal
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from droid_room import CLIPS, DATASET, SOURCE_D, SOURCE_K, save, sha  # noqa: E402
from droid_room import prepare_image as tum_raster  # noqa: E402
from moge3_app import image as moge_image, volume  # noqa: E402  same MoGe-3 image and cached weights

image = moge_image.add_local_python_source("droid_room", "moge3_app")

app = modal.App("panoptes-mono-room-once")
MODEL = "Ruicheng/moge-3-vitl"
CALIBRATION = {"source_K_fx_fy_cx_cy": SOURCE_K, "source_distortion": SOURCE_D}


EVALUATION_DEPTH = "evaluation_only/highres_depth"  # a posed clip's reference depth, millimetres, read by evaluate only
RASTER, METRIC_CAMERAS, SOURCE_WH = "tum", False, (640, 480)  # "tum": official undistort/resize/crop; "resize": a calibrated pinhole image scaled to 640x480


def prepare_image(image, calibration, scale=2):
    """Every image, mask and depth of a clip lives on one 640x480 raster; returns it with its fx, fy, cx, cy."""
    import cv2
    if RASTER == "tum":
        return tum_raster(image, calibration, scale)
    fx, fy, cx, cy = calibration["source_K_fx_fy_cx_cy"]
    height, width = image.shape[:2]
    assert (width, height) == SOURCE_WH and abs(width / height - 4 / 3) < 1e-6 and not np.any(calibration["source_distortion"]), "resize raster needs the clip's undistorted 4:3 image"
    return cv2.resize(image, (640, 480), interpolation=cv2.INTER_AREA), np.array([fx, fy, cx, cy], np.float32) * (640 / width)


def use_clip(droid_run):
    """Dataset, calibration, raster and camera scale follow the clip the run recorded (runs before clips existed are the room clip).

    A run may define its own clip (run.json clip_definition): posed multi-view captures with metric cameras need no SLAM run.
    """
    global DATASET, SOURCE_K, SOURCE_D, CALIBRATION, RASTER, METRIC_CAMERAS, SOURCE_WH, EVALUATION_DEPTH
    run = json.loads((droid_run / "run.json").read_text())
    clip = run.get("clip_definition") or CLIPS[run.get("clip", "fr1-room")]
    DATASET, SOURCE_K, SOURCE_D = Path(clip["dataset"]), clip["K"], clip["D"]
    RASTER, METRIC_CAMERAS, SOURCE_WH = clip.get("raster", "tum"), bool(clip.get("metric_cameras")), tuple(clip.get("source_wh", (640, 480)))
    CALIBRATION = {"source_K_fx_fy_cx_cy": SOURCE_K, "source_distortion": SOURCE_D}
    EVALUATION_DEPTH = clip.get("evaluation_depth", EVALUATION_DEPTH)
    manifest = json.loads((droid_run / "input-manifest.json").read_text())
    assert manifest["source_K_fx_fy_cx_cy"] == list(SOURCE_K) and manifest["source_distortion"] == list(SOURCE_D), "clip calibration differs from the run"
MIN_ANCHORS = 500  # supported 160x120 pixels needed to fit one scale
MAX_ANCHOR_RESIDUAL = .05  # failure criterion: median |s*mono/droid-1| per frame


@app.function(image=image, gpu="L4", volumes={"/cache": volume}, timeout=900, retries=0, max_containers=1)
def infer_remote(frames, fov_x_deg):
    """frames: [(source_index, png_bytes)] -> [(source_index, npz_bytes)] of depth/mask/model K."""
    import cv2
    import torch
    from moge.model.v3 import MoGeModel
    model = MoGeModel.from_pretrained(MODEL).to("cuda").eval()
    results = []
    for index, png in frames:
        rgb = cv2.cvtColor(cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        tensor = (torch.from_numpy(rgb.copy()).float().permute(2, 0, 1) / 255).cuda()
        out = model.infer(tensor, fov_x=fov_x_deg, use_fp16=True)  # unknown kwarg raises: no silent free-FOV run
        buffer = io.BytesIO()
        np.savez_compressed(buffer, depth=out["depth"].cpu().numpy().astype(np.float32),
                            mask=out["mask"].cpu().numpy().astype(bool),
                            model_intrinsics=out["intrinsics"].cpu().numpy())
        results.append((index, buffer.getvalue()))
    return results


da3_image = (modal.Image.debian_slim(python_version="3.11").apt_install("git", "libgl1", "libglib2.0-0")
             .pip_install("torch", "torchvision", "xformers")
             .pip_install("git+https://github.com/ByteDance-Seed/Depth-Anything-3.git", "addict")  # addict: imported by da3.py, missing from its pyproject
             .env({"HF_HOME": "/cache/huggingface"}).add_local_python_source("droid_room", "moge3_app"))


@app.function(image=da3_image, gpu="A100-80GB", volumes={"/cache": volume}, timeout=1200, retries=0, max_containers=1)
def infer_da3_remote(frames, w2c, K, model_name):
    """Posed multi-view depth: the given cameras condition the network and fix the output scale."""
    import cv2
    import torch
    from depth_anything_3.api import DepthAnything3
    model = DepthAnything3.from_pretrained(model_name).to("cuda")
    images = [cv2.cvtColor(cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) for _, png in frames]
    with torch.inference_mode():
        out = model.inference(images, extrinsics=np.asarray(w2c, np.float32), intrinsics=np.asarray(K, np.float32),
                              align_to_input_ext_scale=True, process_res=504)
    assert out.depth.shape == (len(frames), 378, 504), out.depth.shape  # pure .7875 resize of 640x480: K scales, no crop
    assert np.allclose(out.extrinsics[:, :3], np.asarray(w2c)[:, :3], atol=1e-4), "returned cameras are not the given ones"
    results = []
    for (index, _), depth, conf in zip(frames, out.depth, out.conf):
        buffer = io.BytesIO()
        np.savez_compressed(buffer, depth=cv2.resize(depth, (640, 480), interpolation=cv2.INTER_LINEAR).astype(np.float32),
                            conf=cv2.resize(conf, (640, 480), interpolation=cv2.INTER_LINEAR).astype(np.float16),
                            mask=np.ones((480, 640), bool), model_intrinsics=out.intrinsics[0], peak_gb=torch.cuda.max_memory_allocated() / 1e9)
        results.append((index, buffer.getvalue()))
    return results


def infer(droid_run, output, stride, da3_model=None, midframes=False, every=None):
    import cv2
    manifest = json.loads((droid_run / "input-manifest.json").read_text())
    assert not manifest["groundtruth_included"] and not manifest["depth_included"]
    prediction = np.load(droid_run / "prediction.npz")
    keyframes = prediction["keyframe_source_indices"][::stride]
    c2w = dict(enumerate(prediction["poses_c2w"].astype(np.float64)))  # official filler cameras for non-keyframes
    c2w.update(zip(map(int, prediction["keyframe_source_indices"]), prediction["keyframe_c2w"].astype(np.float64)))
    if midframes:  # one extra view halfway between keyframes: more baselines where the camera turns fast
        keyframes = np.unique(np.concatenate([keyframes, (keyframes[:-1] + keyframes[1:]) // 2]))
    if every:  # a steady time sampling for moving entities: a nearly static camera yields very few keyframes
        keyframes = np.unique(np.concatenate([keyframes, np.arange(0, len(prediction["poses_c2w"]), every)]))
    (output / "mono").mkdir(parents=True, exist_ok=True)
    frames, k = [], None
    for index in map(int, keyframes):
        if (output / "mono" / f"{index:05d}.npz").exists():
            continue  # paid once
        record = manifest["frames"][index]
        assert record["source_index"] == index and sha(DATASET / record["relative_path"]) == record["sha256"]
        rectified, k = prepare_image(cv2.imread(str(DATASET / record["relative_path"])), CALIBRATION, 2)
        frames.append((index, cv2.imencode(".png", rectified)[1].tobytes()))
    if not frames:
        return print("all requested keyframes cached")
    fov = float(np.degrees(2 * np.arctan(rectified.shape[1] / 2 / k[0])))
    state = {"status": "gpu_running", "model": da3_model or MODEL, "fov_x_deg": fov, "requested": len(frames),
             "gpu": "A100-80GB" if da3_model else "L4", "timeout_s": 1200 if da3_model else 900, "retries": 0, "sensor_depth_uploaded": False, "groundtruth_uploaded": False,
             "droid_run": str(droid_run), "script_sha256": sha(Path(__file__))}
    with app.run():
        state["app_id"] = app.app_id
        save(output / f"infer-{int(time.time())}.json", state)
        started = time.time()
        size = 400 if da3_model else 30  # DA3: all keyframes share one posed forward pass (59 views peaked at 14 GB)
        chunks = [frames[i:i + size] for i in range(0, len(frames), size)]
        if da3_model:
            own = prediction["keyframe_final_fullres_intrinsics"].astype(np.float64) * 2  # 640x480 raster
            K = lambda i: [[own[i][0], 0, own[i][2]], [0, own[i][1], own[i][3]], [0, 0, 1]] if METRIC_CAMERAS else [[k[0], 0, k[2]], [0, k[1], k[3]], [0, 0, 1]]
            calls = infer_da3_remote.starmap([(c, [np.linalg.inv(c2w[i]) for i, _ in c], [K(i) for i, _ in c], da3_model) for c in chunks])
        else:
            calls = infer_remote.map(chunks, kwargs={"fov_x_deg": fov})
        for chunk in calls:
            for index, data in chunk:
                (output / "mono" / f"{index:05d}.npz").write_bytes(data)
        state.update(status="complete", wall_seconds=time.time() - started)
    save(output / "infer.json", state)
    print(json.dumps(state))


def anchor_scale(mono, droid, supported):
    """One scalar per frame, fitted only where DROID depth has cross-view support."""
    ok = supported & np.isfinite(mono) & (mono > 0) & (droid > 0)
    if ok.sum() < MIN_ANCHORS:
        return None, None, int(ok.sum())
    scale = float(np.median(droid[ok] / mono[ok]))
    return scale, float(np.median(np.abs(scale * mono[ok] / droid[ok] - 1))), int(ok.sum())


def load(droid_run, support, output):
    data = np.load(droid_run / "prediction.npz")
    if support is None and not METRIC_CAMERAS:  # the pinned viewer rule on the official stride-2 raster, exactly as build_droid_replay does
        support = output / "droid-support.npz"
        if not support.exists():
            from build_droid_replay import depth_support
            np.savez_compressed(support, retained=depth_support(data["keyframe_c2w"], data["keyframe_final_fullres_inverse_depth"][:, ::2, ::2],
                                                                data["keyframe_final_fullres_intrinsics"] / 2)[2])
    retained = None if METRIC_CAMERAS else np.load(support)["retained"]  # device-posed captures have no SLAM depth to support
    rows, keys = [], {int(index): key for key, index in enumerate(data["keyframe_source_indices"])}
    # An opened .npz decodes an array again on every access. Read per view inside the loop, each keyframe kept its own
    # 54 MB copy of the DROID depth stack alive through its strided view: 177 copies, 9.6 GB.
    droid_depth = None if METRIC_CAMERAS else data["keyframe_final_fullres_depth"]
    poses, keyframe_poses, intrinsics = data["poses_c2w"], data["keyframe_c2w"], data["keyframe_final_fullres_intrinsics"]
    for path in sorted((output / "mono").glob("*.npz")):
        index = int(path.stem)
        key = None if METRIC_CAMERAS else keys.get(index)  # None: no DROID depth for this view, so no own anchors
        mono = np.load(path)
        depth = np.where(mono["mask"], mono["depth"], 0).astype(np.float32)
        assert depth.shape == (480, 640)
        conf = mono["conf"].astype(np.float32) if "conf" in mono.files else None
        droid = None if key is None else droid_depth[key][::2, ::2]
        scale, residual, anchors = (None, None, 0) if key is None else anchor_scale(depth[::4, ::4], droid, retained[key])
        rows.append({"key": key, "source_index": index, "scale": scale, "anchor_residual": residual,
                     "anchors": anchors, "mono": depth, "conf": conf, "droid": droid, "retained": None if key is None else retained[key],
                     "c2w": (poses[index] if key is None else keyframe_poses[key]).astype(np.float64),
                     "k": intrinsics[index if METRIC_CAMERAS else 0].astype(np.float64)})  # device captures refocus per frame
    assert rows, "run infer first"
    if METRIC_CAMERAS:  # posed depth is already in the cameras' metres; there is no SLAM depth to anchor to
        for r in rows:
            r.update(scale=1., scale_source="metric_cameras")
        return rows
    fitted = [r["scale"] for r in rows if r["scale"]]
    assert fitted, "no keyframe has enough DROID-supported anchors"
    for r in rows:  # ponytail: global median for anchorless frames; temporal interpolation if scale drifts
        r["scale_source"] = "own_anchors" if r["scale"] else "global_median"
        r["scale"] = r["scale"] or float(np.median(fitted))
    return rows


def overlapping_views(cameras, depth, k, minimum=.1):
    """Keyframes, at any time, that see at least `minimum` of a keyframe's coarse depth samples."""
    height, width = depth.shape[1:]
    v, u = np.indices((height, width))[:, 5::10, 5::10]
    ks = np.broadcast_to(np.asarray(k, float), (len(cameras), 4)) if np.ndim(k) == 1 else np.asarray(k, float)  # one K, or one per view
    world = [(np.stack([(u - q[2]) / q[0], (v - q[3]) / q[1], np.ones_like(u, float)], -1) * d[5::10, 5::10, None]).reshape(-1, 3) @ c[:3, :3].T + c[:3, 3]
             for d, c, q in zip(depth, cameras, ks)]
    neighbours = []
    for i, points in enumerate(world):
        seen = []
        for j, c in enumerate(cameras):
            local, k = (points - c[:3, 3]) @ c[:3, :3], ks[j]
            with np.errstate(divide="ignore", invalid="ignore"):
                x, y = local[:, 0] / local[:, 2] * k[0] + k[2], local[:, 1] / local[:, 2] * k[1] + k[3]
            inside = (local[:, 2] > 0) & (x >= 0) & (x < width - 1) & (y >= 0) & (y < height - 1)
            if j != i and inside.mean() >= minimum:
                seen.append(j)
        neighbours.append(seen)
    return neighbours


def unreliable(depth, conf, conf_floor, edge_jump):
    """Pixels the depth model itself doubts, and in-between depths at object borders (flying pixels)."""
    pad = np.pad(depth, 1, mode="edge")
    jump = np.max([np.abs(depth - pad[a:a + depth.shape[0], b:b + depth.shape[1]]) for a, b in [(0, 1), (2, 1), (1, 0), (1, 2)]], 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        bad = jump / depth > edge_jump
    return bad | (conf < conf_floor if conf is not None and conf_floor is not None else False)


def floor_support(rows, plane, k_of, views=2, cell=.05):
    """Per view, the depth of the pixels that lie on the verified floor plane in a floor cell at least `views` views see.

    The cross-view rule compares depth along the ray. A floor seen at a grazing angle turns a small error across the
    plane into a large one along the ray, so floor that many views agree on was thrown away. For a plane already shown
    to be one plane (metric: consensus_plane), agreement is judged across it: distance to the plane, and how many views
    put a point in the same floor cell. Such a pixel gets the depth where its ray meets the plane (it was within the
    plane's tolerance already), so the views fuse into one flat floor instead of a wobbling one. No cell fewer views saw
    is filled, and nothing farther from the plane than its tolerance is touched.
    """
    up, origin, tolerance = np.array(plane["up_native"]), np.array(plane["plane_point_native"]), 1.5 * plane["plane_tolerance_native"]
    a = np.cross(up, [1., 0, 0]); a /= np.linalg.norm(a)
    b = np.cross(up, a)
    v, u = np.indices((480, 640))

    def key_of(relative):  # one non-negative integer per floor cell
        key = np.floor(np.stack([relative @ a, relative @ b], -1) / cell).astype(np.int64) + 50000
        return key[..., 0] * 100003 + key[..., 1]

    def cells_of(r):  # -1 off the floor; recomputed per view rather than held for hundreds of views
        k, z = k_of(r), r["scale"] * r["mono"]
        world = np.stack([(u - k[2]) / k[0] * z, (v - k[3]) / k[1] * z, z], -1) @ r["c2w"][:3, :3].T + r["c2w"][:3, 3] - origin
        return np.where((z > 0) & (np.abs(world @ up) < tolerance), key_of(world), -1)

    def floor_depth(r):
        """(mask, depth): supported floor pixels and the depth at which their ray meets the plane."""
        k = k_of(r)
        rays = np.stack([(u - k[2]) / k[0], (v - k[3]) / k[1], np.ones((480, 640))], -1) @ r["c2w"][:3, :3].T
        with np.errstate(divide="ignore", invalid="ignore"):
            meets = ((origin - r["c2w"][:3, 3]) @ up) / (rays @ up)
        return np.isin(cells_of(r), agreed), meets

    votes = {}
    for r in rows:
        key = cells_of(r)
        for c in np.unique(key[key >= 0]):
            votes[int(c)] = votes.get(int(c), 0) + 1
    agreed = np.array([c for c, n in votes.items() if n >= views], np.int64)
    on_agreed_floor = lambda points: (np.abs((points - origin) @ up) < tolerance) & np.isin(key_of(points - origin), agreed)
    return floor_depth, on_agreed_floor, {"floor_cells_seen": len(votes), "floor_cells_kept": len(agreed), "views_needed": views, "cell_native": cell,
                                          "plane_tolerance_native": tolerance, "depth_on_supported_floor": "ray-plane intersection"}


def fuse(droid_run, support, output, voxel, relative, base_scene=None, all_views=False, dynamic_masks=None,
         conf_percentile=None, edge_jump=None, carve=False, video=None, floor_plane=None):
    import cv2
    import open3d as o3d
    from build_droid_replay import depth_support
    from reconstruct_room_rgb import integrate, new_volume
    rows = load(droid_run, support, output)
    manifest = json.loads((droid_run / "input-manifest.json").read_text())
    conf_floor = (float(np.percentile(np.concatenate([r["conf"][::4, ::4].ravel() for r in rows]), conf_percentile))
                  if conf_percentile and rows[0]["conf"] is not None else None)
    if conf_floor is not None or edge_jump:
        for r in rows:  # removed before voting, so doubtful depth can neither survive nor support a neighbour
            r["mono"] = np.where(unreliable(r["mono"], r["conf"], conf_floor, edge_jump or np.inf), 0, r["mono"])
    aligned = np.stack([r["scale"] * r["mono"][::4, ::4] for r in rows])
    with np.errstate(divide="ignore"):
        disparity = np.where(aligned > 0, 1 / aligned, np.nan)
    # Same viewer rule and thresholds as the DROID review, so the fractions are comparable.
    cameras, ks = np.stack([r["c2w"] for r in rows]), np.stack([r["k"] / 2 for r in rows])
    _, _, same_rule = depth_support(cameras, disparity, ks)
    neighbours = overlapping_views(cameras, aligned, ks) if all_views else None
    mono_supported = same_rule if relative is None and not all_views else depth_support(
        cameras, disparity, ks, tolerance_fraction=relative, neighbours=neighbours)[2]
    volume_ = new_volume(voxel)
    points, colors, masked_views, exposures = [], [], set(), []
    raster_k = prepare_image(np.zeros((480, 640, 3), np.uint8) if RASTER == "tum" else np.zeros((SOURCE_WH[1], SOURCE_WH[0], 3), np.uint8), CALIBRATION, 2)[1]
    on_floor, agreed_floor, floor_report = floor_support(rows, json.loads(floor_plane.read_text()), lambda r: r["k"] * 2 if METRIC_CAMERAS else raster_k) if floor_plane else (None, None, None)
    for r, keep in zip(rows, mono_supported):
        bgr, k = prepare_image(cv2.imread(str(DATASET / manifest["frames"][r["source_index"]]["relative_path"])), CALIBRATION, 2)
        k = r["k"] * 2 if METRIC_CAMERAS else k  # the device refocuses per frame; prepare_image only knows the clip median
        keep = cv2.resize(keep.astype(np.uint8), (640, 480), interpolation=cv2.INTER_NEAREST).astype(bool)
        if on_floor is not None:
            flat, meets = on_floor(r)
            keep |= flat
            r["mono"] = np.where(flat, meets / r["scale"], r["mono"]).astype(np.float32)
        moving = moving_mask(dynamic_masks, r["source_index"])  # source pixels -> depth raster
        if moving.any():
            keep &= ~cv2.dilate(moving.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
            masked_views.add(r["source_index"])
        K = np.array([[k[0], 0, k[2]], [0, k[1], k[3]], [0, 0, 1.]])
        integrate(volume_, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), np.where(keep, r["scale"] * r["mono"], 0), K, r["c2w"])
        exposures.append((np.where(keep, r["scale"] * r["mono"], 0)[::2, ::2], K / [[2], [2], [1]], r["c2w"]))
        v, u = np.indices((480, 640))[:, ::4, ::4]  # display cloud: the 160x120 support raster, source colours
        z = r["scale"] * r["mono"][::4, ::4]
        local = np.stack([(u - k[2]) / k[0] * z, (v - k[3]) / k[1] * z, z], -1)
        points.append(np.where(keep[::4, ::4, None], local @ r["c2w"][:3, :3].T + r["c2w"][:3, 3], np.nan).reshape(-1, 3))
        colors.append(bgr[::4, ::4, ::-1].reshape(-1, 3))
    mesh = volume_.extract_triangle_mesh()
    assert not mesh.is_empty()
    carved = None
    if carve:  # the existing static-surface rule: enough views agree, and few views see straight through the vertex
        from filter_video_static_surfaces import depth_evidence
        vertices = np.asarray(mesh.vertices)
        positive, negative = np.zeros(len(vertices), int), np.zeros(len(vertices), int)
        for depth, k2, c2w in exposures:
            pos, neg = depth_evidence(vertices, depth, depth <= 0, k2, c2w, tolerance=0, depth_range=(0, np.inf), relative_tolerance=.04)
            positive[pos] += 1; negative[neg] += 1
        drop = ~((positive >= 3) & (negative <= np.maximum(2, positive * .15)))
        if floor_plane:  # both carve tests run along the ray, which is what fails on a grazing floor; its support is the vote across the plane
            drop &= ~agreed_floor(vertices)
        carved = {"rule": "filter_video_static_surfaces: positive>=3 and negative<=max(2, .15*positive), 4% relative",
                  "vertices": len(vertices), "removed_vertices": int(drop.sum()), "removed_for_free_space_contradiction": int(((positive >= 3) & drop).sum())}
        mesh.remove_vertices_by_mask(drop)
    o3d.io.write_triangle_mesh(str(output / "mono-anchored-mesh.ply"), mesh)
    o3d.io.write_point_cloud(str(output / "mono-anchored-points.ply"), volume_.extract_point_cloud())
    sizes = np.bincount(np.asarray(mesh.cluster_connected_triangles()[0]))
    residuals = [r["anchor_residual"] for r in rows if r["anchor_residual"] is not None] or [float("nan")]
    scales = np.array([r["scale"] for r in rows if r["scale_source"] == "own_anchors"] or [float("nan")])
    metrics = {"views": len(rows), "keyframes": sum(r["key"] is not None for r in rows), "voxel_native": voxel, "frames_with_own_anchors": len(residuals),
               "anchor_residual_median": float(np.median(residuals)), "anchor_residual_p90": float(np.percentile(residuals, 90)),
               "anchor_residual_criterion": MAX_ANCHOR_RESIDUAL,
               "anchor_criterion_passed": bool(np.median(residuals) <= MAX_ANCHOR_RESIDUAL),
               "scale_native_per_mono_metre_median": float(np.median(scales)),
               "scale_relative_spread_mad": float(np.median(np.abs(scales / np.median(scales) - 1))),
               "droid_support_fraction_keyframes": float(np.mean([r["retained"].mean() for r in rows if r["key"] is not None] or [float("nan")])),
               "cameras": "metric, from the capture device" if METRIC_CAMERAS else "DROID, scale-anchored per keyframe",
               "mono_support_fraction_same_rule": float(same_rule.mean()),
               "fusion_support_views": "every keyframe seeing >=10% of this one" if all_views else "pinned six temporal neighbours",
               "fusion_support_views_median": int(np.median([len(n) for n in neighbours])) if all_views else 6,
               "model_confidence_percentile_removed": conf_percentile if conf_floor is not None else None, "model_confidence_floor": conf_floor,
               "depth_edge_relative_jump_removed": edge_jump, "free_space_carving": carved, "floor_plane_support": floor_report,
               "dynamic_masks": str(dynamic_masks) if dynamic_masks else None, "views_with_dynamic_mask_removed": len(masked_views),
               "fusion_support_relative_tolerance": relative, "fusion_support_fraction": float(mono_supported.mean()),
               "mesh_triangles": len(mesh.triangles), "mesh_components": len(sizes), "largest_component_triangles": int(sizes.max()),
               "note": "Support/anchor numbers measure cross-view consistency, not accuracy; visual review still decides."}
    metrics = {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in metrics.items()}
    save(output / "fuse-metrics.json", metrics)
    print(json.dumps(metrics, indent=1))
    if base_scene or video:
        import trimesh
        points, colors = np.concatenate(points), np.concatenate(colors)
        ids = np.flatnonzero(np.isfinite(points[:, 0]))
        # one real source pixel per 2 cm-native cell: an even cloud instead of dense-near/sparse-far pixel striding
        ids = ids[np.sort(np.unique(np.floor(points[ids] / .02).astype(np.int64), axis=0, return_index=True)[1])]
        ids = ids[::max(1, int(np.ceil(len(ids) / 300000)))]
        trimesh.Scene(trimesh.points.PointCloud(points[ids], colors=colors[ids])).export(output / "supported-keyframe-points.glb")
        trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles), process=False,
                        vertex_colors=(np.asarray(mesh.vertex_colors) * 255).astype(np.uint8)).export(output / "predicted-scene.glb")
        if base_scene:  # same DROID cameras and world, so the replay frames carry over unchanged
            scene = json.loads(base_scene.read_text())
            assert scene["provenance"]["source_run"] == str(droid_run), "base scene must replay the same DROID cameras"
        else:  # the same frame records build_droid_replay writes: verified media times and the official filler cameras
            from build_replay_scene import media_spans
            from reconstruct_room_rgb import digest
            spans, cameras_all = media_spans(video)[0], np.load(droid_run / "prediction.npz")["poses_c2w"]
            assert len(spans) == len(cameras_all) == len(manifest["frames"]), "video, cameras and RGB manifest must list the same frames"
            scene = {"schema": "phase2-replay-scene-v1", "coordinate_frame": "arkit_device_world" if METRIC_CAMERAS else "droid_final_native_world",
                     "units": "meters" if METRIC_CAMERAS else "uncalibrated_monocular",  # the viewer contract's spelling
                     "source_video_sha256": digest(video), "meshUrl": "predicted-scene.glb", "pointCloudUrl": "supported-keyframe-points.glb",
                     "frames": [{"sourceFrame": i, "timeSec": span[0], "endTimeSec": span[1], "c2w": cameras_all[i].tolist(),
                                 "poseSource": "device_arkit" if METRIC_CAMERAS else "droid_motion_only_filler", "objects": []} for i, span in enumerate(spans)]}
        model = json.loads((output / "infer.json").read_text())["model"]
        scene.update(method=(f"Device ARKit cameras (metric, unchanged) + {model} posed depth; TSDF of cross-view supported depth" if METRIC_CAMERAS else
                             f"DROID cameras (unchanged) + {model} depth; per-keyframe scale from DROID-supported pixels; TSDF of cross-view supported depth"),
                     points=[[int(i), *points[i].tolist()] for i in ids], pointCloudCount=len(ids), complete_room_accepted=False,
                     quality_status="not_validated", provenance={"cameras_and_frames_from": str(base_scene or droid_run), "depth_model": model,
                     "gt_pose_input": False, "sensor_depth_input": False, "fuse_metrics": metrics,
                     "point_id": "keyframe order, then row-major 160x120 raster of the 640x480 depth sampled [::4, ::4]"},
                     limitations=[("相机位姿与尺度来自采集设备（ARKit，米制）；不是从视频估计的。" if METRIC_CAMERAS else "单目原生尺度未标定；尺寸和距离不是米。"),
                                  ("深度来自预训练多视角模型，仅保留与≥2个其他视图一致的像素。" if METRIC_CAMERAS else "相机沿用DROID；深度来自另一预训练模型，仅保留与≥2个邻近关键帧一致的像素。"),
                                  "跨视角一致不代表独立几何精度；完整房间尚未验收。", "没有人体或物体掩码；场景中的运动物体未被剔除。"])
        (output / "scene.json").write_text(json.dumps(scene, allow_nan=False, separators=(",", ":")))


def consensus_plane(points, tolerance, iterations=2000):
    """Largest-consensus plane of points already segmented as one surface (deterministic).

    ponytail: not geometry._ransac_floor_plane - that one picks the LOWEST supported plane to skip
    furniture in an unmasked cloud; on floor-masked points it locks onto the noise band under the floor.
    """
    index = np.random.default_rng(0).integers(0, len(points), (iterations, 3))
    a, b, c = points[index[:, 0]], points[index[:, 1]], points[index[:, 2]]
    normals = np.cross(b - a, c - a)
    length = np.linalg.norm(normals, axis=1)
    normals, a = normals[length > 1e-12] / length[length > 1e-12, None], a[length > 1e-12]
    sample = points[::max(1, len(points) // 4000)]
    best = int(np.argmax((np.abs(sample @ normals.T - np.einsum("ij,ij->i", normals, a)) < tolerance).sum(0)))
    inliers = np.abs(points @ normals[best] - normals[best] @ a[best]) < tolerance
    for _ in range(3):
        centre = points[inliers].mean(0)
        normal = np.linalg.svd(points[inliers] - centre, full_matrices=False)[2][-1]
        inliers = np.abs((points - centre) @ normal) < tolerance
    return centre, normal, inliers


def metric(droid_run, support, output, floor_masks, camera_height, metric_depth_run=None):
    """Metres from one stated assumption: the camera is carried `camera_height` above the segmented floor."""
    import cv2
    rows = {r["source_index"]: r for r in load(droid_run, support, output)}
    points, views, patches = [], [], []
    for folder in sorted(floor_masks.glob("frame-*")):
        r = rows.get(int(folder.name.split("-")[1]))
        masks = sorted(folder.glob("instance-*-mask.png"))
        if r is None or not masks:
            continue
        floor = np.any([prepare_image(cv2.imread(str(m), cv2.IMREAD_COLOR), CALIBRATION, 2)[0][..., 0] > 0 for m in masks], 0)
        z = np.where(floor & ~unreliable(r["mono"], None, None, .03), r["scale"] * r["mono"], 0)[::4, ::4]
        v, u = np.indices((480, 640))[:, ::4, ::4]
        k = r["k"] * 2
        local = np.stack([(u - k[2]) / k[0] * z, (v - k[3]) / k[1] * z, z], -1)[z > 0]
        world = local @ r["c2w"][:3, :3].T + r["c2w"][:3, 3]
        if len(world) >= 200:  # minPointsPerView of estimate_native_ground
            normal = np.linalg.svd(world - world.mean(0), full_matrices=False)[2][-1]
            patches.append((r["source_index"], world, normal * np.sign((r["c2w"][:3, 3] - world.mean(0)) @ normal), float(np.median(z[z > 0]))))
    reference = max(patches, key=lambda patch: len(patch[1]))[2]
    rejected = [i for i, _, n, _ in patches if np.degrees(np.arccos(np.clip(n @ reference, -1, 1))) > 5]  # maxCrossViewAngleDegrees
    patches = [patch for patch in patches if patch[0] not in rejected]
    assert len(patches) >= 2, "fewer than two views agree on the floor direction"
    points, views = np.concatenate([patch[1] for patch in patches]), [patch[0] for patch in patches]
    cameras = np.load(droid_run / "prediction.npz")["poses_c2w"][:, :3, 3].astype(np.float64)
    tolerance = .02 * float(np.median([patch[3] for patch in patches]))  # predicted depth: the same 2% used for cross-view support
    centre, up, inliers = consensus_plane(points, tolerance)
    up *= np.sign(np.median((cameras - centre) @ up))  # same convention as estimate_native_ground: up is toward the cameras
    one_plane = inliers.mean() >= .75  # minInlierFraction of estimate_native_ground
    heights = (cameras - centre) @ up
    model_scale = (1 / json.loads((metric_depth_run / "fuse-metrics.json").read_text())["scale_native_per_mono_metre_median"]
                   if metric_depth_run else None)
    report = {"floor_is_one_plane_within_tolerance": bool(one_plane),
              "floor_view_heights_native": {str(i): float(np.median((world - centre) @ up)) for i, world, _, _ in patches},
              "model_estimated_metres_per_native_unit": model_scale,
              "model_estimate_source": str(metric_depth_run) if metric_depth_run else None,
              "height_anchor_vs_model_estimate": None if model_scale is None else camera_height / float(np.median(heights)) / model_scale - 1,
              "scale_sources_disagree_over_10pct": None if model_scale is None else bool(abs(camera_height / float(np.median(heights)) / model_scale - 1) > .1),
              "camera_height_implied_by_model_scale_m": None if model_scale is None else model_scale * float(np.median(heights)),
              "floor_views_rejected_normal_over_5deg": rejected, "plane_tolerance_native": tolerance,
              "assumption": f"camera carried {camera_height} m above the floor (stated by the operator, not measured)",
              "floor_views": views, "floor_points": len(points), "plane_inlier_fraction": float(inliers.mean()),
              "plane_point_native": centre.tolist(), "up_native": up.tolist(),
              "camera_height_native_median": float(np.median(heights)), "camera_height_native_p10_p90": np.percentile(heights, [10, 90]).tolist(),
              "metres_per_native_unit": camera_height / float(np.median(heights)), "scale_status": "assumed_camera_height" if one_plane else "assumed_camera_height_floor_views_disagree",
              "height_spread_note": "a handheld camera moves up and down; p10-p90 spread bounds the scale uncertainty of this anchor"}
    save(output / "metric-scale.json", report)
    print(json.dumps(report, indent=1))


def moving_mask(masks, source_index):
    """Union of the cached entity masks of one source frame, on the 640x480 depth raster."""
    import cv2
    found = sorted(masks.glob(f"{source_index:05d}-*.png")) if masks else []
    return np.any([prepare_image(cv2.imread(str(path), cv2.IMREAD_COLOR), CALIBRATION, 2)[0][..., 0] > 0 for path in found], 0) if found else np.zeros((480, 640), bool)


def entity_depth(row, entity_run, moving):
    """Depth for moving pixels: the map's own posed depth, or a single-frame model tied to it on this view's static pixels.

    Multi-view depth assumes a rigid scene; a single-frame model does not, but has its own scale per image. One robust
    ratio over the static pixels of the same view puts it in the map's units without touching the map.
    """
    own = row["scale"] * row["mono"]
    if entity_run is None:
        return own
    single = np.load(entity_run / "mono" / f"{row['source_index']:05d}.npz")
    single = np.where(single["mask"], single["depth"], 0).astype(np.float32)
    static = ~moving & (own > 0) & (single > 0)
    assert static.sum() >= 5000, "too few static pixels to tie the single-frame depth to the map"
    return single * float(np.median(own[static] / single[static]))


def rectified_pixels(points):
    """Source-image pixels -> the 640x480 depth raster: the same undistort, 704x512 resize and crop as prepare_image at scale 2."""
    import cv2
    fx, fy, cx, cy = SOURCE_K
    k = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]])
    flat = cv2.undistortPoints(np.asarray(points, np.float64).reshape(-1, 1, 2), k, np.asarray(SOURCE_D, np.float64), P=k).reshape(-1, 2)
    return flat * [704 / 640, 512 / 480] - [32, 16] if RASTER == "tum" else flat * (640 / SOURCE_WH[0])


def dynamic(droid_run, support, output, analysis, entity_run=None):
    """Moving entities as time-stamped visible surfaces in the static map's frame.

    One surface per mask per view, from that view's own depth; no mask means no object at that time
    (nothing is carried forward), and nothing behind the visible side is invented.
    """
    import cv2
    from build_video_object_models import observed_surface
    from build_video_surfaces import export_surface
    from build_replay_scene import surface_joints
    rows = {r["source_index"]: r for r in load(droid_run, support, output)}
    manifest = json.loads((droid_run / "input-manifest.json").read_text())
    scene = json.loads((output / "scene.json").read_text())  # written by fuse --base-scene
    frames = {f["sourceFrame"]: f for f in scene["frames"]}
    (output / "dynamic").mkdir(exist_ok=True)
    surfaces = empty = joints_lifted = joints_seen = 0
    for observed in json.loads(analysis.read_text())["frames"]:
        r, frame = rows.get(observed["sourceFrame"]), frames.get(observed["sourceFrame"])
        if r is None or frame is None or not observed["objects"]:
            continue
        bgr, k = prepare_image(cv2.imread(str(DATASET / manifest["frames"][r["source_index"]]["relative_path"])), CALIBRATION, 2)
        K = np.array([[k[0], 0, k[2]], [0, k[1], k[3]], [0, 0, 1.]])
        depth = entity_depth(r, entity_run, moving_mask(analysis.parent / "masks", r["source_index"]))
        depth = np.where(unreliable(depth, None, None, .03), 0, depth)
        frame["objects"] = []
        for number, entity in enumerate(observed["objects"]):
            if not entity.get("maskUrl"):
                continue
            mask = prepare_image(cv2.imread(str(analysis.parent / entity["maskUrl"]), cv2.IMREAD_COLOR), CALIBRATION, 2)[0][..., 0] > 0
            if not (mask & (depth > 0)).any():
                empty += 1
                continue
            reach = float(np.median(depth[mask & (depth > 0)]))
            vertices, faces, colors, _, _ = observed_surface(bgr[..., ::-1], depth, mask, K, r["c2w"], max_edge_m=.04 * reach, depth_range=(0, np.inf))
            if not len(faces):
                empty += 1
                continue
            # 2D skeleton joints onto the visible surface: a joint outside the mask or on inconsistent depth stays empty
            flat = entity.get("keypoints") or []
            joints = np.column_stack([rectified_pixels([j[:2] for j in flat]), [j[2] for j in flat]]) if flat else np.zeros((0, 3))
            keypoints3d = surface_joints(joints, depth, mask, K, r["c2w"], max_depth_spread=.075 * reach)  # .12 m at the 1.6 m the helper was tuned for
            joints_seen += sum(j[2] >= .3 for j in flat); joints_lifted += sum(j is not None for j in keypoints3d)
            path = output / "dynamic" / f"entity-{r['source_index']:05d}-{number}.glb"
            export_surface(path, vertices, faces, colors)
            frame["objects"].append({"entityId": entity["entityId"], "keypoints3d": keypoints3d, "bones": entity.get("bones", []) if flat else [], "representation": "estimated_monocular_surface",
                                     "world_motion": "insufficient_evidence", "centroid": np.median(vertices, 0).tolist(),
                                     "surface": {"meshUrl": f"dynamic/{path.name}", "sourceFrame": r["source_index"],
                                                 "representation": "visible_monocular_surface", "sha256": sha(path), "triangles": len(faces)}})
            surfaces += 1
    # Same convention as the accepted walking replay: scene frames are the sampled views, each lasting until the next one,
    # so an entity is shown only for its own sampling interval and never carried across a view that lacks its mask.
    sampled = [frames[i] for i in sorted(rows) if i in frames]
    for current, following in zip(sampled, sampled[1:]):
        current["endTimeSec"] = following["timeSec"]
    scene["frames"] = sampled
    scene["humanSurfaces"] = {"representation": "visible_monocular_surface", "count": surfaces, "masks_without_usable_depth": empty, "skeleton": {"source": "cached RTMPose COCO17 2D joints lifted onto the visible surface (build_replay_scene.surface_joints)",
                              "confident_2d_joints": int(joints_seen), "lifted_3d_joints": int(joints_lifted), "note": "surface points, not joint centres inside the body"},
                              "coordinateFrame": scene["coordinate_frame"], "hidden_body_completed": False,
                              "depth": ("the static map's own posed depth of the same view" if entity_run is None else
                                        f"single-frame depth of {entity_run.name}, scaled per view on its static pixels to the map's posed depth")
                                       + "; identity is the cached short tracklet, not a person"}
    scene["limitations"] = scene["limitations"] + ["移动对象只有被掩码标出的可见一侧表面，逐视图独立，无观测的时刻不显示；短轨迹ID不是持久身份。"]
    (output / "scene.json").write_text(json.dumps(scene, allow_nan=False, separators=(",", ":")))
    print(json.dumps(scene["humanSurfaces"], ensure_ascii=False))


def evaluate(droid_run, support, output, dynamic_masks=None, entity_run=None):
    """Independent check only: TUM sensor depth, nearest timestamp, same rectification raster."""
    import cv2
    rows = load(droid_run, support, output)
    manifest = json.loads((droid_run / "input-manifest.json").read_text())
    if not METRIC_CAMERAS:
        listing = [line.split() for line in (DATASET / "depth.txt").read_text().splitlines() if not line.startswith("#")]
        times = np.array([float(t) for t, _ in listing])
        K = np.array([[SOURCE_K[0], 0, SOURCE_K[2]], [0, SOURCE_K[1], SOURCE_K[3]], [0, 0, 1.]])
        map_x, map_y = cv2.initUndistortRectifyMap(K, np.asarray(SOURCE_D), None, K, (640, 480), cv2.CV_32FC1)
    pairs = {"mono": [], "mono_on_moving_entities": [], "mono_static_only": [], "droid_all": [], "droid_supported": []}
    for r in rows:
        stamp = manifest["frames"][r["source_index"]]["timestamp_text"]
        if METRIC_CAMERAS:  # the capture's reference depth of the same frame, millimetres, on the source raster
            reference = DATASET / EVALUATION_DEPTH / f"{stamp}.png"
            if not reference.exists():
                continue  # a frame the device gave no confident depth for
            sensor = cv2.imread(str(reference), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000
            sensor = cv2.resize(sensor, (640, 480), interpolation=cv2.INTER_NEAREST)[::4, ::4]
        else:
            nearest = int(np.argmin(np.abs(times - float(stamp))))
            if abs(times[nearest] - float(stamp)) > .02:
                continue
            sensor = cv2.imread(str(DATASET / listing[nearest][1]), cv2.IMREAD_UNCHANGED).astype(np.float32) / 5000
            sensor = cv2.remap(sensor, map_x, map_y, cv2.INTER_NEAREST)
            sensor = cv2.resize(sensor, (704, 512), interpolation=cv2.INTER_NEAREST)[16:-16, 32:-32][::4, ::4]
        seen = sensor > 0
        moving_full = moving_mask(dynamic_masks, r["source_index"])
        moving = moving_full[::4, ::4]
        if entity_run is not None and moving.any():
            pairs.setdefault("single_frame_on_moving_entities", [])
            tied = entity_depth(r, entity_run, moving_full)[::4, ::4]
            picked = seen & moving & (tied > 0)
            if picked.any():
                pairs["single_frame_on_moving_entities"].append((tied[picked], sensor[picked], picked.sum() / seen.sum()))
        for name, depth, mask in [("mono", r["scale"] * r["mono"][::4, ::4], seen), ("mono_on_moving_entities", r["scale"] * r["mono"][::4, ::4], seen & moving),
                                  ("mono_static_only", r["scale"] * r["mono"][::4, ::4], seen & ~moving)] + ([] if r["key"] is None else [
                ("droid_all", r["droid"], seen), ("droid_supported", r["droid"], seen & r["retained"])]):
            mask = mask & (depth > 0)
            if mask.any():
                pairs[name].append((depth[mask], sensor[mask], mask.sum() / seen.sum()))
    # One global native->metre factor per method (median over all pixels): no per-frame GT fitting.
    report = {"frames": len(pairs["mono"]), "sensor_depth_role": "evaluation only"}
    scale_to_sensor = float(np.median(np.concatenate([t for _, t, _ in pairs["mono_static_only"]]) / np.concatenate([p for p, _, _ in pairs["mono_static_only"]])))
    for name, items in pairs.items():
        if not items:
            continue
        predicted, truth = np.concatenate([p for p, _, _ in items]), np.concatenate([t for _, t, _ in items])
        # moving entities are judged in the static scene's scale, so a per-entity depth offset cannot hide in its own fit
        factor = scale_to_sensor if name.endswith("on_moving_entities") else np.median(truth / predicted)
        relative = np.abs(predicted * factor / truth - 1)
        if METRIC_CAMERAS:  # metric cameras claim metres, so the unfitted error is the one that counts
            report.setdefault("unfitted_metres", {})[name] = {"abs_rel_median": float(np.median(np.abs(predicted / truth - 1))), "signed_rel_median": float(np.median(predicted / truth - 1)),
                                                              "within_5pct": float((np.abs(predicted / truth - 1) < .05).mean()), "abs_error_median_m": float(np.median(np.abs(predicted - truth)))}
        report[name] = {"coverage_of_sensor_pixels": float(np.mean([c for _, _, c in items])),
                        "signed_rel_median": float(np.median(predicted * factor / truth - 1)), "abs_rel_median": float(np.median(relative)), "abs_rel_mean": float(relative.mean()),
                        "within_5pct": float((relative < .05).mean()), "within_10pct": float((relative < .10).mean())}
    save(output / "sensor-depth-evaluation.json", report)
    print(json.dumps(report, indent=1))


def self_check():
    rng = np.random.default_rng(0)
    droid = rng.uniform(.5, 3, (120, 160))
    supported = rng.random((120, 160)) < .3
    mono = droid / 1.7
    mono[~supported] *= 9  # unsupported pixels must not influence the anchor
    scale, residual, anchors = anchor_scale(mono, droid, supported)
    assert abs(scale - 1.7) < 1e-9 and residual < 1e-9 and anchors == supported.sum()
    assert anchor_scale(mono, droid, np.zeros_like(supported)) == (None, None, 0)
    outlier = mono.copy(); outlier[:20] *= 4  # a robust fit ignores a wrong sixth of the anchors
    assert abs(anchor_scale(outlier, droid, supported)[0] - 1.7) < .05
    import tempfile
    with tempfile.TemporaryDirectory() as folder:  # the map's posed depth is wrong on the mover; only static pixels may set the scale
        moving = np.zeros((480, 640), bool); moving[100:300, 200:400] = True
        truth = np.where(moving, 1.5, 3.).astype(np.float32)
        (Path(folder) / "mono").mkdir()
        np.savez(Path(folder) / "mono" / "00007.npz", depth=truth / 1.7, mask=np.ones_like(moving))
        tied = entity_depth({"scale": 1., "mono": np.where(moving, 9., 3.).astype(np.float32), "source_index": 7}, Path(folder), moving)
        assert np.allclose(tied, truth, atol=1e-5), "single-frame depth must be scaled by static pixels only"
    global SOURCE_K, SOURCE_D
    saved, SOURCE_K, SOURCE_D = (SOURCE_K, SOURCE_D), [500., 500., 320., 240.], [0.] * 5  # without distortion the mapping is the plain resize and crop
    assert np.allclose(rectified_pixels([[0, 0], [640, 480], [320, 240]]), [[-32, -16], [672, 496], [320, 240]])
    SOURCE_K, SOURCE_D = saved
    rng = np.random.default_rng(1)
    tilted = np.array([.1, -.2, 1.]) / np.linalg.norm([.1, -.2, 1.])
    flat = rng.uniform(-2, 2, (3000, 3)); flat -= np.outer(flat @ tilted, tilted); flat += tilted * .7 + rng.normal(0, .003, (3000, 3))
    clutter = rng.uniform(-2, 2, (900, 3))
    centre, normal, inliers = consensus_plane(np.vstack([flat, clutter]), .02)
    assert abs(abs(normal @ tilted) - 1) < 1e-4 and abs((centre @ tilted) - .7) < .005 and inliers[:3000].mean() > .99, "plane not recovered under 23% clutter"
    step = np.full((6, 8), 2.); step[:, 4:] = 3.; step[:, 4] = 2.5  # an in-between depth invented at a border
    bad = unreliable(step, np.where(np.arange(8) == 0, .1, 9.) * np.ones((6, 8)), 1., .1)
    assert bad[:, [0, 3, 4, 5]].all() and not bad[:, [1, 2, 6, 7]].any(), bad[0]
    front, back = np.eye(4), np.diag([-1., 1, -1, 1])  # same place, facing away
    views = overlapping_views(np.stack([front, back] * 4 + [front]), np.full((9, 120, 160), 2.), [140., 140, 80, 60])
    assert views[0] == [2, 4, 6, 8] and views[1] == [3, 5, 7], views[:2]
    plane = {"up_native": [0., -1., 0.], "plane_point_native": [0., 1., 0.], "plane_tolerance_native": .02}  # floor 1 below the cameras (y points down)
    def looking_down(x):
        c = np.eye(4); c[0, 3] = x
        return {"scale": 1., "c2w": c, "mono": np.where(np.indices((480, 640))[0] > 300, 140. / np.maximum(np.indices((480, 640))[0] - 240., 1e-6), 5.).astype(np.float32)}
    rows_ = [looking_down(x) for x in (0., .05, .1, 30.)]  # three views of the same floor, one far away seeing its own
    rows_[1]["mono"] = rows_[1]["mono"] * 1.01  # a view whose floor depth is 1% off: within the plane's tolerance, so it still votes and is put on the plane
    kept, on_agreed, report_ = floor_support(rows_, plane, lambda r: np.array([140., 140., 320., 240.]))
    flat, meets = kept(rows_[1])
    assert flat[400, 320] and not flat[100, 320] and not kept(rows_[3])[0].any() and report_["floor_cells_kept"] < report_["floor_cells_seen"], "floor kept only on the plane and only where views agree on the cell"
    assert abs(meets[400, 320] - 140. / 160) < 1e-6 and on_agreed(np.array([[0., 1., 140. / 160]]))[0] and not on_agreed(np.array([[0., .5, 1.]]))[0], "supported floor takes the ray-plane depth"
    print("mono room check passed: grazing floor supported across its plane and fused flat; moving-pixel depth tied on static pixels only; consensus floor plane under clutter; low-confidence and border-jump pixels dropped; revisits at any time vote and opposite views do not; anchors only from supported pixels, anchorless frames refused, median robust to outliers")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["infer", "fuse", "metric", "dynamic", "evaluate", "self-check"])
    parser.add_argument("--droid-run", type=Path)
    parser.add_argument("--support", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--every", type=int, help="also infer every Nth source frame (filler cameras)")
    parser.add_argument("--midframes", action="store_true", help="also infer the frame halfway between consecutive keyframes (filler cameras)")
    parser.add_argument("--da3-model", help="e.g. depth-anything/DA3-GIANT-1.1 (CC BY-NC) or depth-anything/DA3-BASE (Apache-2.0); default MoGe-3")
    parser.add_argument("--voxel-length-native", type=float, default=.03)  # same as the DROID review
    parser.add_argument("--support-relative", type=float, help="fusion keeps depth agreeing with >=2 neighbours within this fraction; default: pinned .005 native rule")
    parser.add_argument("--support-all-views", action="store_true", help="let every overlapping keyframe vote, not only six temporal neighbours")
    parser.add_argument("--conf-percentile", type=float, help="drop this lowest share of the depth model's own confidence (DA3 official default: 40)")
    parser.add_argument("--edge-jump", type=float, help="drop pixels whose depth jumps by more than this fraction to a 4-neighbour (flying pixels)")
    parser.add_argument("--carve", action="store_true", help="apply the existing static-surface support/free-space rule to the fused mesh")
    parser.add_argument("--entity-depth-run", type=Path, help="run holding single-frame mono/*.npz to use for moving pixels instead of the posed depth")
    parser.add_argument("--analysis", type=Path, help="cached analysis.json with per-frame entity masks; 'dynamic' lifts them into the fused scene")
    parser.add_argument("--floor-masks", type=Path, help="discover_video_keyframes output for the prompt 'floor'")
    parser.add_argument("--metric-depth-run", type=Path, help="fused run of a metric depth model (e.g. MoGe) to report its scale beside the height anchor")
    parser.add_argument("--camera-height", type=float, help="assumed carrying height in metres")
    parser.add_argument("--dynamic-masks", type=Path, help="directory of SOURCEINDEX-*.png person/object masks in source pixels; removed before fusion")
    parser.add_argument("--video", type=Path, help="source video; without --base-scene, fuse writes the replay frames itself from its CFR times and the DROID cameras")
    parser.add_argument("--floor-plane", type=Path, help="metric-scale.json of a verified floor plane: floor pixels are supported across the plane (>=3 views per floor cell), not along the ray")
    parser.add_argument("--base-scene", type=Path, help="existing replay scene.json of the same DROID run; fuse then also writes a viewer scene")
    a = parser.parse_args()
    if a.command == "self-check":
        self_check()
        sys.exit()
    use_clip(a.droid_run)
    if a.command == "infer":
        infer(a.droid_run, a.output, a.stride, a.da3_model, a.midframes, a.every)
    elif a.command == "metric":
        metric(a.droid_run, a.support, a.output, a.floor_masks, a.camera_height, a.metric_depth_run)
    elif a.command == "fuse":
        fuse(a.droid_run, a.support, a.output, a.voxel_length_native, a.support_relative, a.base_scene, a.support_all_views, a.dynamic_masks, a.conf_percentile, a.edge_jump, a.carve, a.video, a.floor_plane)
    elif a.command == "dynamic":
        dynamic(a.droid_run, a.support, a.output, a.analysis, a.entity_depth_run)
    else:
        evaluate(a.droid_run, a.support, a.output, a.dynamic_masks, a.entity_depth_run)
