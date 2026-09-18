"""One local 8-view MapAnything camera-conditioning probe, without GT or pose fitting.

Uses saved native DROID keyframe RGB, K and c2w only. Outputs remain raw model
predictions: camera conditioning is not a hard camera constraint. A diagnostic
image-grid mesh preserves native predicted XYZ; it is not a DROID-aligned room.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import threading
import time

import numpy as np
from PIL import Image

from reconstruct_room_rgb import digest, pointmap_residuals, validate_frame


def relative_poses(poses):
    return np.linalg.inv(poses[0]) @ poses


def camera_comparison(authority, predicted):
    """Compare gauge-invariant poses using the vendor's declared normalization.

    No fitted transform: each trajectory is expressed in its own first camera,
    and translation is divided by its mean nonzero first-camera baseline.
    """
    original, output = relative_poses(authority), relative_poses(predicted)
    input_lengths = np.linalg.norm(original[1:, :3, 3], axis=1)
    output_lengths = np.linalg.norm(output[1:, :3, 3], axis=1)
    input_norm = input_lengths.sum() / max(int((input_lengths > 0).sum()), 1)
    output_norm = output_lengths.sum() / max(int((output_lengths > 0).sum()), 1)
    if not np.isfinite([input_norm, output_norm]).all() or min(input_norm, output_norm) <= 1e-8:
        raise ValueError("Camera-baseline normalization requires nondegenerate translations")
    a, b = original[:, :3, 3] / input_norm, output[:, :3, 3] / output_norm
    delta = original[:, :3, :3].transpose(0, 2, 1) @ output[:, :3, :3]
    angles = np.degrees(np.arccos(np.clip((np.trace(delta, axis1=1, axis2=2) - 1) / 2, -1, 1)))
    pair_i, pair_j = np.triu_indices(len(a), 1)
    input_lengths = np.linalg.norm(a[pair_i] - a[pair_j], axis=1)
    output_lengths = np.linalg.norm(b[pair_i] - b[pair_j], axis=1)
    usable = input_lengths > 1e-8
    ratios = output_lengths[usable] / input_lengths[usable]
    return {
        "normalization": "inverse first c2w, then divide translations by mean nonzero first-camera baseline; source geometry.py::normalize_pose_translations",
        "fitted_alignment_applied": False, "input_mean_baseline_native": float(input_norm),
        "output_mean_baseline_predicted": float(output_norm), "output_to_input_baseline_scale": float(output_norm / input_norm),
        "relative_rotation_errors_degrees": angles.tolist(),
        "normalized_translation_errors": np.linalg.norm(a - b, axis=1).tolist(),
        "dimensionless_pair_baseline_ratio_quantiles_0_50_100": np.quantile(ratios, [0, .5, 1]).tolist(),
        "raw_pose_max_abs_difference": float(np.max(np.abs(authority - predicted))),
        "input_first_c2w": authority[0].tolist(), "output_first_c2w": predicted[0].tolist(),
    }


def ray_reprojection(rays, K, valid):
    projected = rays @ K.T
    y, x = np.indices(valid.shape)
    with np.errstate(divide="ignore", invalid="ignore"):
        error = np.linalg.norm(projected[..., :2] / projected[..., 2:] - np.stack([x, y], -1), axis=-1)
    usable = valid & np.isfinite(error) & (rays[..., 2] > 0)
    if not usable.any():
        raise ValueError("No positive finite rays for camera comparison")
    return float(np.percentile(error[usable], 95))


def run(args):
    if args.output.exists():
        raise ValueError("Probe output must be a new immutable directory")
    started = time.perf_counter()
    args.output.mkdir(parents=True)
    (args.output / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    metrics = {
        "status": "running", "stage": "verify_native_input", "script_sha256": digest(__file__),
        "requested_views": 8, "processed_views": 0, "frames": [], "timings": {},
        "metric_scale_known": False, "groundtruth_read": False, "sensor_depth_read": False,
        "droid_depth_used": False, "fitted_alignment_applied": False, "input_cameras_changed": False,
        "claim_scope": "camera-conditioning probe; no hard-constraint guarantee or accepted room reconstruction",
        "parameters": {"is_metric_scale": False, "size_wh": [336, 252], "use_amp": False,
                       "memory_efficient_inference": True, "minibatch_size": 1, "depth_edge_limit": .08},
        "memory": {"mps_limit_changed": False, "sampled_peak_driver_bytes": 0, "sampled_peak_allocated_bytes": 0},
        "contract": {
            "inputs": "intrinsics tensor[1,3,3], camera_poses tensor[1,4,4] OpenCV c2w, is_metric_scale bool tensor[1] false",
            "poses": "Input poses are conditioned relative to first camera; translations normalized by mean nonzero baseline. Nonmetric scale encoding is zeroed.",
            "outputs": "Fresh learned rays/depth/pose/scale; output K is a regression from output rays. No input pose/K overwrite or world-gauge restore.",
            "preservation_check": "Numerical equality only: relative rotation <=0.01deg, normalized translation <=0.0001, K abs<=0.001px + rel<=0.0001, ray reprojection p95<=0.01px; not field accuracy limits.",
        },
    }
    stop, sampler = threading.Event(), None

    def save():
        (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")

    save()
    try:
        native = args.droid_run
        remote = json.loads((native / "remote-run.json").read_text())
        if (remote["status"] != "inference_complete" or remote["scale"] != "uncalibrated_monocular"
                or remote["groundtruth_uploaded"] or remote["sensor_depth_uploaded"] or remote["groundtruth_alignment_applied"]):
            raise ValueError("Require completed unaligned RGB-only native DROID run")
        for name in ("prediction.npz", "frames.jsonl", "preprocessing.json", "pose-contract.json"):
            if digest(native / name) != remote["artifacts_sha256"][name]:
                raise ValueError("Changed native artifact: " + name)
        ledger = [json.loads(line) for line in (native / "frames.jsonl").read_text().splitlines()]
        pre = json.loads((native / "preprocessing.json").read_text())
        with np.load(native / "prediction.npz", allow_pickle=False) as data:
            source_indices = data["keyframe_source_indices"]
            native_poses = data["keyframe_c2w"]
            native_k = data["keyframe_intrinsics_fx_fy_cx_cy"]
            native_bgr = data["keyframe_model_bgr"]
        if (len(source_indices) != remote["keyframe_count"] or not np.all(np.diff(source_indices) > 0)
                or native_bgr.shape != (len(source_indices), 240, 320, 3) or native_bgr.dtype != np.uint8):
            raise ValueError("Native keyframe order or exact model RGB domain is invalid")
        source_times = np.array([float(ledger[int(i)]["source_timestamp_text"]) for i in source_indices])
        selected = [int(np.argmin(abs(source_times - t))) for t in np.linspace(source_times[0], source_times[-1], 8)]
        if len(set(selected)) != 8:
            raise ValueError("Sampling did not produce 8 distinct native keyframes")
        metrics["input"] = {"run": str(native), "prediction_sha256": digest(native / "prediction.npz"),
                            "preprocessing_sha256": digest(native / "preprocessing.json"), "preprocessing": pre,
                            "selected_native_keyframe_indices": selected, "selection": "nearest native keyframes to 8 equally spaced timestamps, including first and last"}
        sx, sy = 336 / 320, 252 / 240
        pixel_transform = np.array([[sx, 0, (sx - 1) / 2], [0, sy, (sy - 1) / 2], [0, 0, 1.]])
        images, Ks, poses, paths = [], [], [], []
        for index, key in enumerate(selected):
            source = int(source_indices[key]); row = ledger[source]
            if hashlib.sha256(native_bgr[key].tobytes()).hexdigest() != row["model_bgr_sha256"]:
                raise ValueError("Saved DROID RGB differs from exact inference input")
            directory = args.output / "frames" / f"frame_{index:02d}"
            directory.mkdir(parents=True)
            rgb = Image.fromarray(native_bgr[key, :, :, ::-1])
            rgb.save(directory / "droid-rgb.png")
            rgb = rgb.resize((336, 252), Image.Resampling.LANCZOS)
            rgb.save(directory / "rgb.png")
            color = np.asarray(rgb)
            fx, fy, cx, cy = native_k[key]
            K = pixel_transform @ np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]])
            validate_frame(color, np.ones(color.shape[:2]), np.ones(color.shape[:2]), np.ones(color.shape[:2], bool), K, native_poses[key])
            images.append(color); Ks.append(K.astype(np.float32)); poses.append(native_poses[key]); paths.append(directory / "rgb.png")
            np.savez_compressed(directory / "camera-input.npz", K=Ks[-1], c2w=poses[-1], is_metric_scale=np.array([False]), native_K_fx_fy_cx_cy=native_k[key])
            metrics["frames"].append({"frame_id": directory.name, "native_keyframe_index": key, "source_frame_index": source,
                                      "timestamp": source_times[key], "source_original_rgb_sha256": row["rgb_sha256"],
                                      "source_path": str(directory / "droid-rgb.png"), "source_sha256": digest(directory / "droid-rgb.png"),
                                      "source_bgr_sha256": row["model_bgr_sha256"], "source_size_wh": [320, 240],
                                      "processed_size_wh": [336, 252], "processed_rgb_sha256": digest(paths[-1]),
                                      "source_to_processed_pixel_centres": pixel_transform.tolist(),
                                      "camera_input_sha256": digest(directory / "camera-input.npz")})
        poses = np.asarray(poses)
        metrics["stage"] = "load_model"
        save()
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        sys.path.insert(0, str(args.serving_root / "scripts"))
        from candidate_mapanything_backend import MapAnythingRunner, _numpy, _require_identity_grid
        import torch

        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS unavailable; no alternate execution path")
        # ponytail: sampled peaks can miss sub-0.2s transients; exact peaks need allocator instrumentation.
        def sample():
            while not stop.is_set():
                for name, method in (("sampled_peak_driver_bytes", torch.mps.driver_allocated_memory),
                                     ("sampled_peak_allocated_bytes", torch.mps.current_allocated_memory)):
                    metrics["memory"][name] = max(metrics["memory"][name], method())
                stop.wait(.2)
        sampler = threading.Thread(target=sample, daemon=True); sampler.start()
        t0 = time.perf_counter()
        runner = MapAnythingRunner(args.vendor_dir, args.model_dir, device="mps")
        metrics["timings"]["model_load_seconds"] = time.perf_counter() - t0
        metrics["model"] = {k: runner.metadata[k] for k in ("model_id", "model_revision", "weights_sha256", "code_revision", "device", "torch_version")}
        metrics["contract"]["source_files"] = {str(p.relative_to(args.vendor_dir)): digest(p) for p in (
            args.vendor_dir / "mapanything/utils/inference.py", args.vendor_dir / "mapanything/utils/geometry.py",
            args.vendor_dir / "mapanything/models/mapanything/model.py")}
        views = runner.load_images([str(p) for p in paths], resize_mode="fixed_size", size=(336, 252), norm_type="dinov2", patch_size=14)
        for view, color, K, pose in zip(views, images, Ks, poses):
            _require_identity_grid(runner.rgb(view["img"], "dinov2")[0], color)
            view.update(intrinsics=torch.from_numpy(K[None]), camera_poses=torch.from_numpy(pose[None].copy()),
                        is_metric_scale=torch.tensor([False], dtype=torch.bool))
        if len(views) != 8:
            raise ValueError("Official preprocessing dropped a view")
        metrics["stage"] = "single_camera_conditioned_inference"; save()
        print("Running one 8-view native DROID camera-conditioned MapAnything probe", flush=True)
        t0 = time.perf_counter()
        with torch.inference_mode():
            predictions = runner.model.infer(views, memory_efficient_inference=True, minibatch_size=1, use_amp=False,
                                             apply_mask=False, mask_edges=False, ignore_calibration_inputs=False, ignore_pose_inputs=False)
        torch.mps.synchronize()
        metrics["timings"]["inference_seconds"] = time.perf_counter() - t0
        if len(predictions) != 8:
            raise ValueError("Inference did not return all 8 outputs")
        from mapanything.utils.hf_utils.viz import image_mesh
        import trimesh

        output_poses, meshes, preserved_K, ray_errors = [], [], [], []
        for pred, color, K, path, record in zip(predictions, images, Ks, paths, metrics["frames"]):
            directory = path.parent
            np.savez_compressed(directory / "native-output.npz", **{k: v.detach().cpu().numpy() for k, v in pred.items()})
            _require_identity_grid(_numpy(pred["img_no_norm"]), color)
            depth, conf, mask = _numpy(pred["depth_z"])[..., 0], _numpy(pred["conf"]), _numpy(pred["non_ambiguous_mask"])
            k_out, pose_out, points, rays = [_numpy(pred[k]) for k in ("intrinsics", "camera_poses", "pts3d", "ray_directions")]
            valid = validate_frame(color, depth, conf, mask, k_out, pose_out) & np.isfinite(points).all(-1)
            gy, gx = np.gradient(depth)
            valid &= np.hypot(gx, gy) / np.maximum(depth, 1e-6) < .08
            if not valid.any():
                raise ValueError("No native finite geometry survives the declared edge mask")
            output_poses.append(pose_out)
            error = ray_reprojection(rays, K, valid); ray_errors.append(error)
            preserved_K.append(bool(np.allclose(k_out, K, atol=.001, rtol=.0001)))
            np.savez_compressed(directory / "prediction.npz", depth_z=depth, confidence=conf, native_mask=mask,
                                valid_mask=valid, K=k_out, c2w=pose_out, pts3d=points, ray_directions=rays,
                                authoritative_K=K, authoritative_c2w=poses[len(output_poses) - 1])
            record.update(prediction_sha256=digest(directory / "prediction.npz"), native_output_sha256=digest(directory / "native-output.npz"),
                          native_output_keys=list(pred), valid_pixels=int(valid.sum()), total_pixels=int(valid.size),
                          **pointmap_residuals(points, depth, k_out, pose_out, valid),
                          output_rays_under_authoritative_K_error_p95_px=error,
                          intrinsics_max_abs_difference_px=float(np.max(abs(k_out - K))),
                          intrinsics_focal_relative_errors=((k_out[[0, 1], [0, 1]] / K[[0, 1], [0, 1]]) - 1).tolist(),
                          metric_scaling_factor=_numpy(pred["metric_scaling_factor"]).tolist())
            # Native helper triangulates neighboring valid samples; do not call
            # predictions_to_glb, which applies an unrelated 180deg world rotation.
            faces, vertices, colors = image_mesh(points, color.astype(float) / 255, mask=valid, tri=True)
            meshes.append(trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=np.rint(colors * 255).astype(np.uint8), process=False))
            metrics["processed_views"] += 1; save()
        comparison = camera_comparison(poses.astype(float), np.asarray(output_poses, dtype=float))
        pose_ok = max(comparison["relative_rotation_errors_degrees"]) <= .01 and max(comparison["normalized_translation_errors"]) <= .0001
        preserved = pose_ok and all(preserved_K) and max(ray_errors) <= .01
        metrics["camera_comparison"] = comparison
        metrics["camera_contract_preserved_after_documented_gauge_normalization"] = preserved
        metrics["authority_fusion_performed"] = False
        metrics["acceptance"] = "camera_contract_preserved_geometry_quality_not_evaluated" if preserved else "rejected_camera_contract_drift"
        mesh = trimesh.util.concatenate(meshes)
        for name in ("native-predicted-surface.ply", "native-predicted-surface.glb"):
            mesh.export(args.output / name)
        loaded = trimesh.load(args.output / "native-predicted-surface.glb", force="mesh", process=False)
        if not np.array_equal(loaded.faces, mesh.faces) or not np.allclose(loaded.vertices, mesh.vertices, atol=1e-6):
            raise ValueError("Native diagnostic mesh did not survive export readback")
        metrics["diagnostic_mesh"] = {"vertices": len(mesh.vertices), "faces": len(mesh.faces),
                                      "coordinates": "unaltered predicted MapAnything world; not aligned into DROID world",
                                      "method": "native image-grid triangles; no TSDF, K reprojection, pose rewrite, registration or hole fill",
                                      "outputs": {name: {"sha256": digest(args.output / name), "bytes": (args.output / name).stat().st_size}
                                                  for name in ("native-predicted-surface.ply", "native-predicted-surface.glb")}}
        metrics.update(status="probe_complete", stage="complete", all_requested_views_processed=True)
    except BaseException as error:
        metrics.update(status="failed", error=repr(error)); raise
    finally:
        stop.set()
        if sampler is not None:
            sampler.join(timeout=1)
        metrics["timings"]["total_seconds"] = time.perf_counter() - started
        metrics["memory"]["process_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        save()
    print(json.dumps({k: metrics[k] for k in ("status", "acceptance", "camera_comparison", "timings", "memory")}, indent=2), flush=True)


def self_check():
    poses = np.repeat(np.eye(4)[None], 3, axis=0)
    poses[1:, :3, 3] = [[1, 0, 0], [0, 2, 0]]
    gauge = np.eye(4); gauge[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]; gauge[:3, 3] = [3, 4, 5]
    predicted = poses.copy(); predicted[:, :3, 3] *= 7; predicted = gauge @ predicted
    same = camera_comparison(poses, predicted)
    assert max(same["relative_rotation_errors_degrees"]) < 1e-6
    assert max(same["normalized_translation_errors"]) < 1e-10
    np.testing.assert_allclose(same["dimensionless_pair_baseline_ratio_quantiles_0_50_100"], 1)
    predicted[2, :3, 3] += [.5, 0, 0]
    assert max(camera_comparison(poses, predicted)["normalized_translation_errors"]) > .01
    K = np.array([[10., 0, 3.5], [0, 10., 2.5], [0, 0, 1.]])
    y, x = np.indices((6, 8)); rays = np.stack([(x - 3.5) / 10, (y - 2.5) / 10, np.ones_like(x)], -1)
    assert ray_reprojection(rays, K, np.ones((6, 8), bool)) < 1e-10
    wrong_K = K.copy(); wrong_K[0, 0] *= 2
    assert ray_reprojection(rays, wrong_K, np.ones((6, 8), bool)) > 2
    print("self-check passed: documented gauge/scale normalization, relative drift rejection and input-K ray mismatch", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    for name in ("droid-run", "output", "serving-root", "vendor-dir", "model-dir"):
        parser.add_argument("--" + name, type=Path)
    args = parser.parse_args()
    if args.self_check:
        self_check()
    elif any(getattr(args, k) is None for k in ("droid_run", "output", "serving_root", "vendor_dir", "model_dir")):
        parser.error("All cached run/model/source/output paths are required")
    else:
        run(args)
