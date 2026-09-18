"""Estimate a TUM RGB-D trajectory on CPU and fuse its observed surfaces.

Sensor-depth control only: no training, learned depth, ground-truth mapping,
loop closure, relocalization, or semantic object reconstruction.
"""

import argparse
from bisect import bisect_left, bisect_right
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np


TUM_K = np.array([[525., 0, 319.5], [0, 525., 239.5], [0, 0, 1.]])


def associate(first, second, maximum=0.02):
    """Nearest timestamp pairs, each input used once; same greedy rule as TUM."""
    first, second = np.asarray(first), np.asarray(second)
    if (maximum <= 0 or not np.isfinite(first).all() or not np.isfinite(second).all()
            or np.any(np.diff(first) <= 0) or np.any(np.diff(second) <= 0)):
        raise ValueError("Expected finite, strictly increasing timestamps and positive tolerance")
    candidates = []
    for i, stamp in enumerate(first):
        for j in range(bisect_left(second, stamp - maximum), bisect_right(second, stamp + maximum)):
            difference = abs(float(stamp - second[j]))
            if difference < maximum:
                candidates.append((difference, i, j))
    used_first, used_second, pairs = set(), set(), []
    for _, i, j in sorted(candidates):
        if i not in used_first and j not in used_second:
            used_first.add(i)
            used_second.add(j)
            pairs.append((i, j))
    return sorted(pairs)


def advance_pose(previous_c2w, previous_to_current):
    """Open3D source=previous,target=current; TSDF later consumes inverse c2w."""
    for pose in (previous_c2w, previous_to_current):
        if (pose.shape != (4, 4) or not np.isfinite(pose).all()
                or not np.allclose(pose[3], [0, 0, 0, 1])
                or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-4)
                or not np.isclose(np.linalg.det(pose[:3, :3]), 1, atol=1e-4)):
            raise ValueError("Expected finite rigid camera transforms")
    return previous_c2w @ np.linalg.inv(previous_to_current)


def metric_ate(estimated, reference):
    """Posthoc SE3 alignment only: scale remains exactly 1, positions in meters."""
    estimated, reference = np.asarray(estimated, dtype=float), np.asarray(reference, dtype=float)
    if (estimated.shape != reference.shape or estimated.ndim != 2 or estimated.shape[1] != 3
            or len(estimated) < 3 or not np.isfinite([estimated, reference]).all()):
        raise ValueError("ATE needs at least three paired finite XYZ positions")
    x, y = estimated - estimated.mean(0), reference - reference.mean(0)
    u, _, vt = np.linalg.svd(x.T @ y)
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(vt.T @ u.T)
    rotation = vt.T @ correction @ u.T
    translation = reference.mean(0) - rotation @ estimated.mean(0)
    errors = np.linalg.norm(estimated @ rotation.T + translation - reference, axis=1)
    return {"matched_poses": len(errors), "alignment": "SE3 rigid, posthoc evaluation only",
            "scale_fit": False, "scale": 1.0, "rmse_m": float(np.sqrt(np.mean(errors ** 2))),
            "median_m": float(np.median(errors)), "p95_m": float(np.percentile(errors, 95)),
            "max_m": float(errors.max()), "alignment_rotation": rotation.tolist(),
            "alignment_translation": translation.tolist()}


def read_rows(path, columns):
    rows = [line.split() for line in path.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")]
    if not rows or any(len(row) != columns for row in rows):
        raise ValueError(f"Expected {columns} columns in {path}")
    stamps = [float(row[0]) for row in rows]
    if not np.isfinite(stamps).all() or np.any(np.diff(stamps) <= 0):
        raise ValueError(f"Timestamps must be finite and increasing: {path}")
    return rows


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def reconstruct(args):
    import cv2
    import open3d as o3d
    import trimesh
    from scipy.spatial.transform import Rotation

    root, output = args.dataset.resolve(), args.output.resolve()
    rgb_rows, depth_rows = read_rows(root / "rgb.txt", 2), read_rows(root / "depth.txt", 2)
    pairs = associate([float(row[0]) for row in rgb_rows], [float(row[0]) for row in depth_rows], args.max_time_difference)
    selected = pairs[::args.stride]
    if args.max_frames:
        selected = selected[:args.max_frames]
    if len(selected) < 2:
        raise ValueError("Need at least two associated RGB-D frames")
    for rows in (rgb_rows, depth_rows):
        for _, relative in rows:
            path = (root / relative).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise ValueError(f"Missing or unsafe dataset image path: {relative}")
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    # Pixel centers must follow OpenCV resize, including the half-pixel offset.
    scale = args.width / 640
    height = args.width * 3 // 4
    k = TUM_K.copy()
    k[0, 0] *= scale
    k[1, 1] *= scale
    k[:2, 2] = (k[:2, 2] + .5) * scale - .5
    intrinsic = o3d.camera.PinholeCameraIntrinsic(args.width, height, k[0, 0], k[1, 1], k[0, 2], k[1, 2])
    odometry_option = o3d.pipelines.odometry.OdometryOption(
        iteration_number_per_pyramid_level=o3d.utility.IntVector([20, 10, 5]),
        depth_min=0., depth_max=args.depth_max, depth_diff_max=.07)
    tsdf = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=args.voxel_size, sdf_trunc=4 * args.voxel_size,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    summary = {
        "status": "running", "dataset": str(root), "coordinate_frame": "first accepted RGB camera; OpenCV right/down/forward",
        "method": "Open3D legacy CPU hybrid RGB-D odometry estimated poses + ScalableTSDFVolume",
        "tracking_gate": "native odometry success AND evaluate_registration fitness >= min_fitness and inlier_rmse <= max_rmse; depth point clouds sampled every 2 pixels; correspondence radius 0.07m",
        "input": "TUM registered sensor RGB-D; not ordinary RGB-only video",
        "training": False, "paid_compute": False, "groundtruth_used_for_mapping": False,
        "metric_scale_source": "TUM sensor depth PNG / 5000; no fitted scale or extra Freiburg correction",
        "calibration": {"original_K": TUM_K.tolist(), "processed_K": k.tolist(), "undistorted": False,
                        "source_size": [640, 480], "processed_size": [args.width, height],
                        "resize": "OpenCV INTER_AREA RGB; INTER_NEAREST depth; pixel-center K transform"},
        "parameters": {key: value for key, value in vars(args).items() if key not in {"dataset", "output"}},
        "rgb_count": len(rgb_rows), "depth_count": len(depth_rows), "associated_pairs": len(pairs),
        "selected_pairs": len(selected), "accepted_frames": 0, "integrated_frames": 0,
        "rgb_span_seconds": float(rgb_rows[-1][0]) - float(rgb_rows[0][0]),
        "associated_rgb_span_seconds": float(rgb_rows[pairs[-1][0]][0]) - float(rgb_rows[pairs[0][0]][0]),
        "open3d_version": o3d.__version__, "opencv_version": cv2.__version__, "trimesh_version": trimesh.__version__,
        "index_sha256": {name: digest(root / name) for name in ["rgb.txt", "depth.txt"]},
        "script_sha256": digest(Path(__file__).resolve()),
        "limitations": ["Sensor depth control does not validate RGB-only reconstruction or ORB-SLAM3.",
                        "Sequential odometry has no loop closure, global optimization, Atlas save/reload, or relocalization.",
                        "A local fitness gate can miss drift or wrong correspondences; inspect posthoc ATE and surfaces.",
                        "Dynamic content may leave ghost surfaces; unobserved space remains missing.",
                        "A colored surface mesh does not establish semantic entities, editable parts, or completion."],
    }
    records, pose, previous, previous_points, previous_timestamp = [], np.eye(4), None, None, None
    timing = {"decode_seconds": 0.0, "odometry_seconds": 0.0, "integration_seconds": 0.0}
    progress_path = output / "progress.jsonl"
    try:
        with progress_path.open("x") as progress:
            for frame_index, (rgb_index, depth_index) in enumerate(selected):
                stamp, color_path = rgb_rows[rgb_index]
                depth_stamp, depth_path = depth_rows[depth_index]
                record = {"frame": frame_index, "rgb_timestamp": float(stamp), "depth_timestamp": float(depth_stamp),
                          "rgb": color_path, "depth": depth_path, "status": "tracking", "integrated": False}
                begin = time.perf_counter()
                color = cv2.imread(str(root / color_path), cv2.IMREAD_COLOR)
                depth = cv2.imread(str(root / depth_path), cv2.IMREAD_UNCHANGED)
                if color is None or depth is None or color.shape != (480, 640, 3) or depth.shape != (480, 640) or depth.dtype != np.uint16:
                    raise ValueError(f"Expected TUM 640x480 RGB and uint16 depth at frame {frame_index}")
                color = cv2.cvtColor(cv2.resize(color, (args.width, height), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
                depth = cv2.resize(depth, (args.width, height), interpolation=cv2.INTER_NEAREST)
                current = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    o3d.geometry.Image(color), o3d.geometry.Image(depth), depth_scale=5000., depth_trunc=args.depth_max)
                current_points = o3d.geometry.PointCloud.create_from_depth_image(
                    o3d.geometry.Image(depth), intrinsic, depth_scale=5000., depth_trunc=args.depth_max, stride=2)
                timing["decode_seconds"] += time.perf_counter() - begin
                if previous is not None:
                    if float(stamp) - previous_timestamp > args.max_frame_gap:
                        record.update(status="lost_tracking", reason="input frame gap exceeds declared limit")
                    else:
                        begin = time.perf_counter()
                        success, relative, information = o3d.pipelines.odometry.compute_rgbd_odometry(
                            previous, current, intrinsic, np.eye(4),
                            o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(), odometry_option)
                        result = o3d.pipelines.registration.evaluate_registration(previous_points, current_points, .07, relative)
                        timing["odometry_seconds"] += time.perf_counter() - begin
                        record.update(odometry_success=success,
                                      fitness=float(result.fitness) if np.isfinite(result.fitness) else None,
                                      inlier_rmse=float(result.inlier_rmse) if np.isfinite(result.inlier_rmse) else None,
                                      previous_to_current=relative.tolist() if np.isfinite(relative).all() else None)
                        if (not success or not np.isfinite([result.fitness, result.inlier_rmse]).all()
                                or not np.isfinite(relative).all()
                                or result.fitness < args.min_fitness or result.inlier_rmse > args.max_rmse):
                            record.update(status="lost_tracking", reason="odometry fitness/RMSE failed declared gate")
                        else:
                            pose = advance_pose(pose, relative)
                if record["status"] == "lost_tracking":
                    summary.update(status="lost_tracking", failed_frame=record)
                    progress.write(json.dumps(record, allow_nan=False) + "\n")
                    progress.flush()
                    print(json.dumps(record), flush=True)
                    break
                record.update(status="accepted", camera_to_world=pose.tolist())
                if frame_index % args.integrate_every == 0 or frame_index == len(selected) - 1:
                    begin = time.perf_counter()
                    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                        o3d.geometry.Image(color), o3d.geometry.Image(depth), depth_scale=5000.,
                        depth_trunc=args.depth_max, convert_rgb_to_intensity=False)
                    # Same c2w -> inverse(c2w) integration convention as reconstruct_workcell.py.
                    tsdf.integrate(rgbd, intrinsic, np.linalg.inv(pose))
                    timing["integration_seconds"] += time.perf_counter() - begin
                    record["integrated"] = True
                    summary["integrated_frames"] += 1
                records.append(record)
                summary["accepted_frames"] = len(records)
                progress.write(json.dumps(record, allow_nan=False) + "\n")
                progress.flush()
                previous, previous_points, previous_timestamp = current, current_points, float(stamp)
                if frame_index % 30 == 0:
                    print(json.dumps({"frame": frame_index, "selected": len(selected), "accepted": len(records),
                                      "integrated": summary["integrated_frames"], "fitness": record.get("fitness"),
                                      "elapsed_seconds": round(time.perf_counter() - started, 2)}), flush=True)
            else:
                summary["status"] = "complete"
    except Exception as error:
        summary.update(status="failed", error=f"{type(error).__name__}: {error}")
    finally:
        # Always preserve accepted poses and failure state; never add a failed frame at a stale pose.
        (output / "trajectory.json").write_text(json.dumps(records, indent=2, allow_nan=False) + "\n")
        with (output / "trajectory_estimated.txt").open("w") as stream:
            stream.write("# timestamp tx ty tz qx qy qz qw; estimated c2w in meters\n")
            for record in records:
                p = np.asarray(record["camera_to_world"])
                values = [record["rgb_timestamp"], *p[:3, 3], *Rotation.from_matrix(p[:3, :3]).as_quat()]
                stream.write(" ".join(f"{value:.9f}" for value in values) + "\n")
        summary.update(timing)
        summary["tracking_seconds"] = time.perf_counter() - started
        (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")

    try:
        begin = time.perf_counter()
        mesh = tsdf.extract_triangle_mesh()
        mesh.remove_degenerate_triangles().remove_duplicated_triangles().remove_unreferenced_vertices()
        mesh.compute_vertex_normals()
        vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
        if not len(faces) or not np.isfinite(vertices).all():
            raise ValueError("TSDF produced an empty or nonfinite surface")
        if not o3d.io.write_triangle_mesh(str(output / "surface.ply"), mesh, write_ascii=False):
            raise IOError("Cannot save surface.ply")
        colors = np.rint(np.clip(np.asarray(mesh.vertex_colors), 0, 1) * 255).astype(np.uint8)
        trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=colors, process=False).export(output / "room.glb")
        check = trimesh.load(output / "room.glb", force="mesh", process=False)
        if (not np.allclose(check.vertices, vertices, atol=1e-6) or not np.array_equal(check.faces, faces)
                or check.visual.kind != "vertex"):
            raise ValueError("GLB readback differs from estimated TSDF mesh")
        summary.update(vertices=len(vertices), faces=len(faces), mesh_bounds_m=[vertices.min(0).tolist(), vertices.max(0).tolist()],
                       extraction_export_seconds=time.perf_counter() - begin, glb_readback_verified=True)
        # Ground truth is opened only after all estimated poses and the mesh are finalized.
        gt_rows = read_rows(root / "groundtruth.txt", 8)
        gt_pairs = associate([record["rgb_timestamp"] for record in records], [float(row[0]) for row in gt_rows], args.max_time_difference)
        if len(gt_pairs) >= 3:
            estimated = [np.asarray(records[i]["camera_to_world"])[:3, 3] for i, _ in gt_pairs]
            reference = [[float(value) for value in gt_rows[j][1:4]] for _, j in gt_pairs]
            summary["ate"] = metric_ate(estimated, reference)
            summary["ate"]["groundtruth_sha256"] = digest(root / "groundtruth.txt")
        else:
            summary["ate"] = {"status": "insufficient_matched_poses", "matched_poses": len(gt_pairs)}
        summary["artifacts"] = {name: {"sha256": digest(output / name), "bytes": (output / name).stat().st_size}
                                for name in ["room.glb", "surface.ply", "trajectory.json", "trajectory_estimated.txt", "progress.jsonl"]}
    except Exception as error:
        summary.update(status="failed", finalization_error=f"{type(error).__name__}: {error}")
    summary.update(elapsed_seconds=time.perf_counter() - started,
                   accepted_associated_fraction=len(records) / len(pairs),
                   peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    checks = {"full_associated_sequence_selected": len(selected) == len(pairs),
              "execution_complete": summary["status"] == "complete",
              "tracking_at_least_95_percent": summary["accepted_associated_fraction"] >= .95,
              "metric_ate_rmse_at_most_0_15m": summary.get("ate", {}).get("rmse_m", float("inf")) <= .15,
              "finite_nonempty_mesh": summary.get("glb_readback_verified", False)}
    summary["acceptance"] = {"passed": all(checks.values()), "checks": checks,
                             "threshold_basis": "predeclared local RGB-D sanity experiment targets; not a published accuracy guarantee"}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2, allow_nan=False), flush=True)
    return 0 if summary["acceptance"]["passed"] else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=320, choices=[320, 640])
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--integrate-every", type=int, default=5)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--max-time-difference", type=float, default=.02)
    parser.add_argument("--max-frame-gap", type=float, default=.15)
    parser.add_argument("--voxel-size", type=float, default=.02)
    parser.add_argument("--depth-max", type=float, default=4.)
    parser.add_argument("--min-fitness", type=float, default=.2)
    parser.add_argument("--max-rmse", type=float, default=.05)
    args = parser.parse_args()
    if (args.stride < 1 or args.integrate_every < 1 or args.max_frames < 0 or args.max_time_difference <= 0
            or args.max_frame_gap <= 0 or args.voxel_size <= 0 or args.depth_max <= 0
            or not 0 < args.min_fitness <= 1 or args.max_rmse <= 0):
        parser.error("Frame counts and geometry/timing thresholds must be positive; fitness must lie in (0,1]")
    return reconstruct(args)


if __name__ == "__main__":
    sys.exit(main())
