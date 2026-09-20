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
from droid_room import DATASET, SOURCE_D, SOURCE_K, prepare_image, save, sha  # noqa: E402
from moge3_app import image as moge_image, volume  # noqa: E402  same MoGe-3 image and cached weights

image = moge_image.add_local_python_source("droid_room", "moge3_app")

app = modal.App("panoptes-mono-room-once")
MODEL = "Ruicheng/moge-3-vitl"
CALIBRATION = {"source_K_fx_fy_cx_cy": SOURCE_K, "source_distortion": SOURCE_D}
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


def infer(droid_run, output, stride, da3_model=None, midframes=False):
    import cv2
    manifest = json.loads((droid_run / "input-manifest.json").read_text())
    assert not manifest["groundtruth_included"] and not manifest["depth_included"]
    prediction = np.load(droid_run / "prediction.npz")
    keyframes = prediction["keyframe_source_indices"][::stride]
    c2w = dict(enumerate(prediction["poses_c2w"].astype(np.float64)))  # official filler cameras for non-keyframes
    c2w.update(zip(map(int, prediction["keyframe_source_indices"]), prediction["keyframe_c2w"].astype(np.float64)))
    if midframes:  # one extra view halfway between keyframes: more baselines where the camera turns fast
        keyframes = np.unique(np.concatenate([keyframes, (keyframes[:-1] + keyframes[1:]) // 2]))
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
            K = [[k[0], 0, k[2]], [0, k[1], k[3]], [0, 0, 1]]
            calls = infer_da3_remote.starmap([(c, [np.linalg.inv(c2w[i]) for i, _ in c], [K] * len(c), da3_model) for c in chunks])
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
    retained = np.load(support)["retained"]
    rows, keys = [], {int(index): key for key, index in enumerate(data["keyframe_source_indices"])}
    for path in sorted((output / "mono").glob("*.npz")):
        index = int(path.stem)
        key = keys.get(index)  # None: a filler-camera frame without DROID depth, so no own anchors
        mono = np.load(path)
        depth = np.where(mono["mask"], mono["depth"], 0).astype(np.float32)
        assert depth.shape == (480, 640)
        conf = mono["conf"].astype(np.float32) if "conf" in mono.files else None
        droid = None if key is None else data["keyframe_final_fullres_depth"][key][::2, ::2]
        scale, residual, anchors = (None, None, 0) if key is None else anchor_scale(depth[::4, ::4], droid, retained[key])
        rows.append({"key": key, "source_index": index, "scale": scale, "anchor_residual": residual,
                     "anchors": anchors, "mono": depth, "conf": conf, "droid": droid, "retained": None if key is None else retained[key],
                     "c2w": (data["poses_c2w"][index] if key is None else data["keyframe_c2w"][key]).astype(np.float64),
                     "k": data["keyframe_final_fullres_intrinsics"][0].astype(np.float64)})
    assert rows, "run infer first"
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
    rays = np.stack([(u - k[2]) / k[0], (v - k[3]) / k[1], np.ones_like(u, float)], -1)
    world = [(rays * d[5::10, 5::10, None]).reshape(-1, 3) @ c[:3, :3].T + c[:3, 3] for d, c in zip(depth, cameras)]
    neighbours = []
    for i, points in enumerate(world):
        seen = []
        for j, c in enumerate(cameras):
            local = (points - c[:3, 3]) @ c[:3, :3]
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


def fuse(droid_run, support, output, voxel, relative, base_scene=None, all_views=False, dynamic_masks=None,
         conf_percentile=None, edge_jump=None, carve=False):
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
    neighbours = overlapping_views(cameras, aligned, ks[0]) if all_views else None
    mono_supported = same_rule if relative is None and not all_views else depth_support(
        cameras, disparity, ks, tolerance_fraction=relative, neighbours=neighbours)[2]
    volume_ = new_volume(voxel)
    points, colors, masked_views, exposures = [], [], set(), []
    for r, keep in zip(rows, mono_supported):
        bgr, k = prepare_image(cv2.imread(str(DATASET / manifest["frames"][r["source_index"]]["relative_path"])), CALIBRATION, 2)
        keep = cv2.resize(keep.astype(np.uint8), (640, 480), interpolation=cv2.INTER_NEAREST).astype(bool)
        for path in sorted(dynamic_masks.glob(f"{r['source_index']:05d}-*.png")) if dynamic_masks else []:
            moving = prepare_image(cv2.imread(str(path), cv2.IMREAD_COLOR), CALIBRATION, 2)[0][..., 0] > 0  # source pixels -> depth raster
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
        carved = {"rule": "filter_video_static_surfaces: positive>=3 and negative<=max(2, .15*positive), 4% relative",
                  "vertices": len(vertices), "removed_vertices": int(drop.sum()), "removed_for_free_space_contradiction": int(((positive >= 3) & drop).sum())}
        mesh.remove_vertices_by_mask(drop)
    o3d.io.write_triangle_mesh(str(output / "mono-anchored-mesh.ply"), mesh)
    o3d.io.write_point_cloud(str(output / "mono-anchored-points.ply"), volume_.extract_point_cloud())
    sizes = np.bincount(np.asarray(mesh.cluster_connected_triangles()[0]))
    residuals = [r["anchor_residual"] for r in rows if r["anchor_residual"] is not None]
    scales = np.array([r["scale"] for r in rows if r["scale_source"] == "own_anchors"])
    metrics = {"views": len(rows), "keyframes": sum(r["key"] is not None for r in rows), "voxel_native": voxel, "frames_with_own_anchors": len(residuals),
               "anchor_residual_median": float(np.median(residuals)), "anchor_residual_p90": float(np.percentile(residuals, 90)),
               "anchor_residual_criterion": MAX_ANCHOR_RESIDUAL,
               "anchor_criterion_passed": bool(np.median(residuals) <= MAX_ANCHOR_RESIDUAL),
               "scale_native_per_mono_metre_median": float(np.median(scales)),
               "scale_relative_spread_mad": float(np.median(np.abs(scales / np.median(scales) - 1))),
               "droid_support_fraction_keyframes": float(np.mean([r["retained"].mean() for r in rows if r["key"] is not None])),
               "mono_support_fraction_same_rule": float(same_rule.mean()),
               "fusion_support_views": "every keyframe seeing >=10% of this one" if all_views else "pinned six temporal neighbours",
               "fusion_support_views_median": int(np.median([len(n) for n in neighbours])) if all_views else 6,
               "model_confidence_percentile_removed": conf_percentile if conf_floor is not None else None, "model_confidence_floor": conf_floor,
               "depth_edge_relative_jump_removed": edge_jump, "free_space_carving": carved,
               "dynamic_masks": str(dynamic_masks) if dynamic_masks else None, "views_with_dynamic_mask_removed": len(masked_views),
               "fusion_support_relative_tolerance": relative, "fusion_support_fraction": float(mono_supported.mean()),
               "mesh_triangles": len(mesh.triangles), "mesh_components": len(sizes), "largest_component_triangles": int(sizes.max()),
               "note": "Support/anchor numbers measure cross-view consistency, not accuracy; visual review still decides."}
    save(output / "fuse-metrics.json", metrics)
    print(json.dumps(metrics, indent=1))
    if base_scene:  # same DROID cameras and world, so the replay frames carry over unchanged
        import trimesh
        points, colors = np.concatenate(points), np.concatenate(colors)
        ids = np.flatnonzero(np.isfinite(points[:, 0]))
        # one real source pixel per 2 cm-native cell: an even cloud instead of dense-near/sparse-far pixel striding
        ids = ids[np.sort(np.unique(np.floor(points[ids] / .02).astype(np.int64), axis=0, return_index=True)[1])]
        ids = ids[::max(1, int(np.ceil(len(ids) / 300000)))]
        trimesh.Scene(trimesh.points.PointCloud(points[ids], colors=colors[ids])).export(output / "supported-keyframe-points.glb")
        trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles), process=False,
                        vertex_colors=(np.asarray(mesh.vertex_colors) * 255).astype(np.uint8)).export(output / "predicted-scene.glb")
        scene = json.loads(base_scene.read_text())
        assert scene["provenance"]["source_run"] == str(droid_run), "base scene must replay the same DROID cameras"
        model = json.loads((output / "infer.json").read_text())["model"]
        scene.update(method=f"DROID cameras (unchanged) + {model} depth; per-keyframe scale from DROID-supported pixels; TSDF of cross-view supported depth",
                     points=[[int(i), *points[i].tolist()] for i in ids], pointCloudCount=len(ids), complete_room_accepted=False,
                     quality_status="not_validated", provenance={"cameras_and_frames_from": str(base_scene), "depth_model": model,
                     "gt_pose_input": False, "sensor_depth_input": False, "fuse_metrics": metrics,
                     "point_id": "keyframe order, then row-major 160x120 raster of the 640x480 depth sampled [::4, ::4]"},
                     limitations=["单目原生尺度未标定；尺寸和距离不是米。", "相机沿用DROID；深度来自另一预训练模型，仅保留与≥2个邻近关键帧一致的像素。",
                                  "跨视角一致不代表独立几何精度；完整房间尚未验收。", "没有人体或物体掩码；场景中的运动物体未被剔除。"])
        (output / "scene.json").write_text(json.dumps(scene, allow_nan=False, separators=(",", ":")))


def metric(droid_run, support, output, floor_masks, camera_height, metric_depth_run=None):
    """Metres from one stated assumption: the camera is carried `camera_height` above the segmented floor."""
    import cv2
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from ehs_spatial.geometry import _ransac_floor_plane
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
    center = np.median(points, 0)
    normal = np.linalg.svd(points - center, full_matrices=False)[2][-1]
    normal *= np.sign(np.median((cameras - center) @ normal))  # same convention as estimate_native_ground: up is toward the cameras
    extent = float(np.quantile(np.linalg.norm(points - center, axis=1), .95))
    tolerance = .02 * float(np.median([patch[3] for patch in patches]))  # predicted depth: the same 2% used for cross-view support
    inliers = _ransac_floor_plane(points, cameras, normal, tolerance)
    one_plane = inliers is not None and len(inliers) >= .5 * len(points)
    if not one_plane:  # report it, anchored on the nearest (most reliable) floor view, instead of hiding the disagreement
        inliers = np.arange(len(min(patches, key=lambda patch: patch[3])[1])) + sum(
            len(patch[1]) for patch in patches[:patches.index(min(patches, key=lambda patch: patch[3]))])
    centre = points[inliers].mean(0)
    up = np.linalg.svd(points[inliers] - centre, full_matrices=False)[2][-1]
    up *= np.sign(up @ normal)
    heights = (cameras - centre) @ up
    model_scale = (1 / json.loads((metric_depth_run / "fuse-metrics.json").read_text())["scale_native_per_mono_metre_median"]
                   if metric_depth_run else None)
    report = {"floor_is_one_plane_within_tolerance": bool(one_plane),
              "floor_view_heights_native": {str(i): float(np.median((world - centre) @ up)) for i, world, _, _ in patches},
              "model_estimated_metres_per_native_unit": model_scale,
              "model_estimate_source": str(metric_depth_run) if metric_depth_run else None,
              "camera_height_implied_by_model_scale_m": None if model_scale is None else model_scale * float(np.median(heights)),
              "floor_views_rejected_normal_over_5deg": rejected, "plane_tolerance_native": tolerance,
              "assumption": f"camera carried {camera_height} m above the floor (stated by the operator, not measured)",
              "floor_views": views, "floor_points": len(points), "plane_inlier_fraction": len(inliers) / len(points),
              "plane_point_native": centre.tolist(), "up_native": up.tolist(),
              "camera_height_native_median": float(np.median(heights)), "camera_height_native_p10_p90": np.percentile(heights, [10, 90]).tolist(),
              "metres_per_native_unit": camera_height / float(np.median(heights)), "scale_status": "assumed_camera_height" if one_plane else "assumed_camera_height_floor_views_disagree",
              "height_spread_note": "a handheld camera moves up and down; p10-p90 spread bounds the scale uncertainty of this anchor"}
    save(output / "metric-scale.json", report)
    print(json.dumps(report, indent=1))


def evaluate(droid_run, support, output):
    """Independent check only: TUM sensor depth, nearest timestamp, same rectification raster."""
    import cv2
    rows = load(droid_run, support, output)
    manifest = json.loads((droid_run / "input-manifest.json").read_text())
    listing = [line.split() for line in (DATASET / "depth.txt").read_text().splitlines() if not line.startswith("#")]
    times = np.array([float(t) for t, _ in listing])
    K = np.array([[SOURCE_K[0], 0, SOURCE_K[2]], [0, SOURCE_K[1], SOURCE_K[3]], [0, 0, 1.]])
    map_x, map_y = cv2.initUndistortRectifyMap(K, np.asarray(SOURCE_D), None, K, (640, 480), cv2.CV_32FC1)
    pairs = {"mono": [], "droid_all": [], "droid_supported": []}
    for r in rows:
        nearest = int(np.argmin(np.abs(times - float(manifest["frames"][r["source_index"]]["timestamp_text"]))))
        if abs(times[nearest] - float(manifest["frames"][r["source_index"]]["timestamp_text"])) > .02:
            continue
        sensor = cv2.imread(str(DATASET / listing[nearest][1]), cv2.IMREAD_UNCHANGED).astype(np.float32) / 5000
        sensor = cv2.remap(sensor, map_x, map_y, cv2.INTER_NEAREST)
        sensor = cv2.resize(sensor, (704, 512), interpolation=cv2.INTER_NEAREST)[16:-16, 32:-32][::4, ::4]
        seen = sensor > 0
        for name, depth, mask in [("mono", r["scale"] * r["mono"][::4, ::4], seen)] + ([] if r["key"] is None else [
                ("droid_all", r["droid"], seen), ("droid_supported", r["droid"], seen & r["retained"])]):
            mask = mask & (depth > 0)
            pairs[name].append((depth[mask], sensor[mask], mask.sum() / seen.sum()))
    # One global native->metre factor per method (median over all pixels): no per-frame GT fitting.
    report = {"frames": len(pairs["mono"]), "sensor_depth_role": "evaluation only"}
    for name, items in pairs.items():
        predicted, truth = np.concatenate([p for p, _, _ in items]), np.concatenate([t for _, t, _ in items])
        relative = np.abs(predicted * np.median(truth / predicted) / truth - 1)
        report[name] = {"coverage_of_sensor_pixels": float(np.mean([c for _, _, c in items])),
                        "abs_rel_median": float(np.median(relative)), "abs_rel_mean": float(relative.mean()),
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
    step = np.full((6, 8), 2.); step[:, 4:] = 3.; step[:, 4] = 2.5  # an in-between depth invented at a border
    bad = unreliable(step, np.where(np.arange(8) == 0, .1, 9.) * np.ones((6, 8)), 1., .1)
    assert bad[:, [0, 3, 4, 5]].all() and not bad[:, [1, 2, 6, 7]].any(), bad[0]
    front, back = np.eye(4), np.diag([-1., 1, -1, 1])  # same place, facing away
    views = overlapping_views(np.stack([front, back] * 4 + [front]), np.full((9, 120, 160), 2.), [140., 140, 80, 60])
    assert views[0] == [2, 4, 6, 8] and views[1] == [3, 5, 7], views[:2]
    print("mono room check passed: low-confidence and border-jump pixels dropped; revisits at any time vote and opposite views do not; anchors only from supported pixels, anchorless frames refused, median robust to outliers")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["infer", "fuse", "metric", "evaluate", "self-check"])
    parser.add_argument("--droid-run", type=Path)
    parser.add_argument("--support", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--midframes", action="store_true", help="also infer the frame halfway between consecutive keyframes (filler cameras)")
    parser.add_argument("--da3-model", help="e.g. depth-anything/DA3-GIANT-1.1 (CC BY-NC) or depth-anything/DA3-BASE (Apache-2.0); default MoGe-3")
    parser.add_argument("--voxel-length-native", type=float, default=.03)  # same as the DROID review
    parser.add_argument("--support-relative", type=float, help="fusion keeps depth agreeing with >=2 neighbours within this fraction; default: pinned .005 native rule")
    parser.add_argument("--support-all-views", action="store_true", help="let every overlapping keyframe vote, not only six temporal neighbours")
    parser.add_argument("--conf-percentile", type=float, help="drop this lowest share of the depth model's own confidence (DA3 official default: 40)")
    parser.add_argument("--edge-jump", type=float, help="drop pixels whose depth jumps by more than this fraction to a 4-neighbour (flying pixels)")
    parser.add_argument("--carve", action="store_true", help="apply the existing static-surface support/free-space rule to the fused mesh")
    parser.add_argument("--floor-masks", type=Path, help="discover_video_keyframes output for the prompt 'floor'")
    parser.add_argument("--metric-depth-run", type=Path, help="fused run of a metric depth model (e.g. MoGe) to report its scale beside the height anchor")
    parser.add_argument("--camera-height", type=float, help="assumed carrying height in metres")
    parser.add_argument("--dynamic-masks", type=Path, help="directory of SOURCEINDEX-*.png person/object masks in source pixels; removed before fusion")
    parser.add_argument("--base-scene", type=Path, help="existing replay scene.json of the same DROID run; fuse then also writes a viewer scene")
    a = parser.parse_args()
    if a.command == "self-check":
        self_check()
    elif a.command == "infer":
        infer(a.droid_run, a.output, a.stride, a.da3_model, a.midframes)
    elif a.command == "metric":
        metric(a.droid_run, a.support, a.output, a.floor_masks, a.camera_height, a.metric_depth_run)
    elif a.command == "fuse":
        fuse(a.droid_run, a.support, a.output, a.voxel_length_native, a.support_relative, a.base_scene, a.support_all_views, a.dynamic_masks, a.conf_percentile, a.edge_jump, a.carve)
    else:
        evaluate(a.droid_run, a.support, a.output)
