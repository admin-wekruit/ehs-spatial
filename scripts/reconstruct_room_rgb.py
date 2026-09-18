"""Local RGB-only 2-32-view MapAnything room experiment; no SLAM or metric truth claim.

Run with panoptes-serving/.venv/bin/python and explicit cached serving/vendor/model
paths. --self-check tests z-depth, XYZ reprojection diagnostics and c2w fusion.
Inputs are either --dataset (rgb.txt + RGB PNGs only) or --video (selected decoded RGB
frames). No depth or ground truth is read. --prepare-only validates video/RGB
preprocessing without loading or running the model.
"""

import argparse
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import threading
import time
import tempfile

import numpy as np
from PIL import Image


LONGEST_SIDE = 336
DEFAULT_VIEWS = 8
VOXEL = 0.04
TRUNCATION = 0.16


def _file_version(stat):
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


@lru_cache(maxsize=4096)
def _file_digest(path, version):
    # ponytail: process-local metadata cache for stable local files, not file bytes.
    # A new process or changed inode/size/timestamps must read and hash again.
    with Path(path).open("rb") as stream:
        if _file_version(os.fstat(stream.fileno())) != version:
            raise OSError("File changed before hashing")
        value = hashlib.file_digest(stream, "sha256").hexdigest()
        if (_file_version(os.fstat(stream.fileno())) != version
                or _file_version(Path(path).stat()) != version):
            raise OSError("File changed while hashing")
    return value


def digest(path):
    path = Path(path).resolve(strict=True)
    return _file_digest(str(path), _file_version(path.stat()))


def validate_frame(color, depth, confidence, native_mask, K, c2w):
    height, width = depth.shape
    if (color.shape != (height, width, 3) or color.dtype != np.uint8
            or confidence.shape != depth.shape or native_mask.shape != depth.shape
            or not np.isin(native_mask, [0, 1]).all()):
        raise ValueError("RGB, depth, confidence and binary validity must share the same pixel domain")
    if (K.shape != (3, 3) or not np.isfinite(K).all()
            or not np.allclose(K[2], [0, 0, 1]) or min(K[0, 0], K[1, 1]) <= 0
            or not np.allclose([K[0, 1], K[1, 0]], 0)
            or not (-0.5 <= K[0, 2] < width - 0.5 and -0.5 <= K[1, 2] < height - 0.5)):
        raise ValueError("Expected positive, zero-skew pinhole K in the processed RGB domain")
    if (c2w.shape != (4, 4) or not np.isfinite(c2w).all()
            or not np.allclose(c2w[3], [0, 0, 0, 1])
            or not np.allclose(c2w[:3, :3].T @ c2w[:3, :3], np.eye(3), atol=1e-3)
            or not np.isclose(np.linalg.det(c2w[:3, :3]), 1, atol=1e-3)):
        raise ValueError("Expected a finite rigid OpenCV camera-to-world transform")
    valid = native_mask.astype(bool) & np.isfinite(depth) & (depth > 0) & np.isfinite(confidence)
    if not valid.any():
        raise ValueError("Frame has no finite, positive native z-depth")
    return valid


def integrate(volume, color, depth, K, c2w):
    import open3d as o3d

    height, width = depth.shape
    intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, K[0, 0], K[1, 1], K[0, 2], K[1, 2])
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(np.ascontiguousarray(color)),
        o3d.geometry.Image(np.ascontiguousarray(depth, dtype=np.float32)),
        depth_scale=1.0, depth_trunc=float(depth.max()) + volume.sdf_trunc,
        convert_rgb_to_intensity=False)
    volume.integrate(rgbd, intrinsic, np.linalg.inv(c2w))


def pointmap_residuals(points, depth, K, c2w, valid):
    """Measure the recovered pinhole approximation; do not alter native rays."""
    local = (np.asarray(points, dtype=float) - c2w[:3, 3]) @ c2w[:3, :3]
    projected = local @ K.T
    yy, xx = np.indices(depth.shape)
    with np.errstate(divide="ignore", invalid="ignore"):
        error = np.linalg.norm(projected[..., :2] / projected[..., 2:] - np.stack([xx, yy], axis=-1), axis=-1)
    projectable = valid & (local[..., 2] > 0) & np.isfinite(error)
    return {
        "pointmap_camera_z_residual_p95": float(np.percentile(np.abs(local[..., 2][valid] - depth[valid]), 95)),
        "pointmap_xyz_reprojection_error_p95_px": float(np.percentile(error[projectable], 95)) if projectable.any() else None,
        "pointmap_reprojectable_pixels": int(projectable.sum()),
        "pointmap_reprojection_invalid_pixels": int((valid & ~projectable).sum()),
    }


def new_volume(voxel_length=VOXEL):
    import open3d as o3d

    if not np.isfinite(voxel_length) or voxel_length <= 0:
        raise ValueError("TSDF voxel length must be finite and positive")
    return o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_length, sdf_trunc=4 * voxel_length,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)


def selected_rgb(dataset, views=DEFAULT_VIEWS):
    root = (dataset / "rgb").resolve()
    entries = []
    for line in (dataset / "rgb.txt").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        timestamp, relative = line.split()
        path = (dataset / relative).resolve()
        if path.parent != root or path.suffix.lower() != ".png" or not path.is_file():
            raise ValueError("RGB index must reference existing PNGs directly inside dataset/rgb")
        entries.append((float(timestamp), path))
    times = np.array([x[0] for x in entries])
    if len(entries) < views or not np.isfinite(times).all() or not np.all(np.diff(times) > 0):
        raise ValueError("Expected enough uniquely timestamped, chronological RGB frames")
    indices = [int(np.argmin(np.abs(times - t))) for t in np.linspace(times[0], times[-1], views)]
    if len(set(indices)) != views:
        raise ValueError("Evenly spaced sampling did not yield the requested distinct frames")
    return [entries[i] for i in indices], len(entries)


def selected_video(video, output, views=DEFAULT_VIEWS):
    import cv2

    if not video.is_file():
        raise ValueError("Video file does not exist")
    capture = cv2.VideoCapture(str(video))
    try:
        if not capture.isOpened():
            raise ValueError("Video could not be opened")
        count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not np.isfinite([count, fps]).all() or int(count) != count or count < views or fps <= 0:
            raise ValueError("Video requires enough indexed frames and valid FPS metadata")
        indices = np.rint(np.linspace(0, int(count) - 1, views)).astype(int).tolist()
        selected, records = [], []
        directory = output / "decoded_rgb"
        directory.mkdir()
        # ponytail: sequential decode avoids inaccurate container seeks; time is
        # O(video frames). Indexed decoding needs verified seek/frame alignment.
        for index in range(int(count)):
            if not capture.grab():
                raise ValueError(f"Video ended or failed decoding at frame {index} of {int(count)}")
            if index not in indices:
                continue
            ok, bgr = capture.retrieve()
            timestamp = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if not ok or bgr is None or not np.isfinite(timestamp) or timestamp < 0:
                raise ValueError(f"Invalid RGB or timestamp at selected frame {index}")
            if selected and timestamp <= selected[-1][0]:
                raise ValueError("Selected video timestamps must increase; FPS is not a timestamp substitute")
            path = directory / f"source_{index:08d}.png"
            Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).save(path)
            selected.append((timestamp, path))
            records.append({"frame_index": index, "timestamp_seconds": timestamp,
                            "decoded_rgb_sha256": digest(path), "decoded_size_wh": list(bgr.shape[1::-1])})
        if len(selected) != views:
            raise ValueError("Video did not yield all requested RGB frames")
        return selected, int(count), {"path": str(video.resolve()), "sha256": digest(video),
                                     "reported_fps": fps, "reported_frame_count": int(count),
                                     "sampling": f"{views} equally spaced frame indices including first and last",
                                     "timestamps": "actual OpenCV CAP_PROP_POS_MSEC after sequential decode",
                                     "selected_frames": records}
    finally:
        capture.release()


def reconstruct(args):
    if not 2 <= args.views <= 32:
        raise ValueError("This bounded RGB experiment accepts 2 to 32 views")
    if args.output.exists():
        raise ValueError("Output must be a new immutable run directory")
    if args.dataset and not args.dataset.is_dir():
        raise ValueError("RGB dataset does not exist; inference was not started")
    args.output.mkdir(parents=True)
    (args.output / "reconstruct_room_rgb.py").write_bytes(Path(__file__).read_bytes())
    started = time.perf_counter()
    metrics = {
        "status": "running", "quality_status": "not_validated", "stage": "prepare_rgb", "script_sha256": digest(__file__),
        "input_kind": "video" if args.video else "rgb_dataset",
        "input_contract": "RGB only; depth, ground truth, known calibration and trajectories are never read",
        "requested_views": args.views, "processed_views": 0,
        "claim_scope": "joint sparse whole-room geometry experiment; no tracking, persistent map, relocalization or object graph",
        "metric_scale_known": False, "units": "native predicted scale; not independently calibrated",
        "parameters": {"longest_side": LONGEST_SIDE, "memory_efficient_inference": True, "minibatch_size": 1,
                       "use_amp": False, "voxel_length": VOXEL, "sdf_trunc": TRUNCATION,
                       "relative_depth_edge_limit": 0.08, "confidence_threshold": None},
        "depth_contract": "native predicted camera z-depth with K regressed from native rays; pinhole approximation measured by full XYZ reprojection, not assumed exact; TSDF uses inverse OpenCV c2w",
        "mask_contract": "native non_ambiguous_mask AND finite positive depth/confidence/points AND relative depth edge <0.08",
        "confidence_semantics": "native learned confidence, not a calibrated probability",
        "memory": {"mps_limit_changed": False, "sample_interval_seconds": 0.2,
                   "sampled_peak_mps_allocated_bytes": 0, "sampled_peak_mps_driver_bytes": 0},
        "frames": [], "timings": {},
    }
    stop = threading.Event()
    sampler = None

    def save():
        (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")

    save()
    try:
        if args.video:
            selected, available, video = selected_video(args.video, args.output, args.views)
            metrics.update(video=video, sampling=video["sampling"])
        else:
            selected, available = selected_rgb(args.dataset, args.views)
            metrics.update(dataset=str(args.dataset.resolve()), rgb_index_sha256=digest(args.dataset / "rgb.txt"),
                           sampling=f"nearest RGB timestamps to {args.views} equally spaced times including first and last")
        metrics.update(available_rgb_frames=available, sample_span_seconds=selected[-1][0] - selected[0][0])
        with Image.open(selected[0][1]) as first:
            source_size = first.size
        size = tuple(max(14, int(value * LONGEST_SIDE / max(source_size)) // 14 * 14) for value in source_size)
        metrics["parameters"]["image_size_wh"] = size
        canonical = []
        for index, (timestamp, path) in enumerate(selected):
            frame = args.output / "frames" / f"frame_{index:02d}"
            frame.mkdir(parents=True)
            with Image.open(path) as source:
                if source.format != "PNG" or source.size != source_size:
                    raise ValueError("Selected RGB PNG frames must share one source pixel domain")
                rgb = source.convert("RGB").resize(size, Image.Resampling.LANCZOS)
                rgb.save(frame / "rgb.png")
            sx, sy = size[0] / source_size[0], size[1] / source_size[1]
            metrics["frames"].append({"frame_id": frame.name, "timestamp": timestamp, "source_path": str(path),
                                      "source_sha256": digest(path), "source_size_wh": source_size,
                                      "processed_size_wh": size, "processed_rgb_sha256": digest(frame / "rgb.png"),
                                      "source_to_processed_pixel_centres": [[sx, 0, (sx - 1) / 2], [0, sy, (sy - 1) / 2], [0, 0, 1]]})
            if args.video:
                metrics["frames"][-1]["source_video_frame_index"] = video["selected_frames"][index]["frame_index"]
            canonical.append(frame / "rgb.png")
        metrics["timings"]["prepare_rgb_seconds"] = time.perf_counter() - started
        if args.prepare_only:
            metrics.update(status="prepared", stage="rgb_prepared", all_requested_views_processed=False,
                           prepared_views=len(canonical), model_calls=0)
            print(f"Prepared {len(canonical)} RGB frames at {size}; no model loaded", flush=True)
            return
        metrics["stage"] = "load_model"
        save()
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        sys.path.insert(0, str(args.serving_root / "scripts"))
        from candidate_mapanything_backend import MapAnythingRunner, _numpy, _require_identity_grid
        import torch

        if not torch.backends.mps.is_available():
            raise RuntimeError("This experiment requires local MPS; no CPU/cloud substitution")
        metrics["memory"]["mps_recommended_max_bytes"] = torch.mps.recommended_max_memory()
        metrics["memory"]["mps_high_watermark_env"] = os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO")

        # ponytail: 0.2s polling records sampled MPS peaks, not allocator-exact
        # transients; use allocator instrumentation if exact GPU peaks are needed.
        def sample():
            while not stop.is_set():
                memory = metrics["memory"]
                memory["sampled_peak_mps_allocated_bytes"] = max(memory["sampled_peak_mps_allocated_bytes"], torch.mps.current_allocated_memory())
                memory["sampled_peak_mps_driver_bytes"] = max(memory["sampled_peak_mps_driver_bytes"], torch.mps.driver_allocated_memory())
                stop.wait(0.2)

        sampler = threading.Thread(target=sample, daemon=True)
        sampler.start()
        t0 = time.perf_counter()
        runner = MapAnythingRunner(args.vendor_dir, args.model_dir, device="mps")
        torch.mps.synchronize()
        metrics["timings"]["model_load_seconds"] = time.perf_counter() - t0
        metrics["model"] = {k: runner.metadata[k] for k in (
            "model_id", "model_revision", "weights_sha256", "code_revision", "torch_version", "uniception_version", "device", "strict_state_dict")}
        views = runner.load_images([str(p) for p in canonical], resize_mode="fixed_size", size=size,
                                   norm_type="dinov2", patch_size=14)
        if len(views) != args.views:
            raise ValueError("Official preprocessing did not retain all requested RGB views")
        images = [np.asarray(Image.open(p).convert("RGB")) for p in canonical]
        for view, image in zip(views, images):
            _require_identity_grid(runner.rgb(view["img"], "dinov2")[0], image)
        metrics["stage"] = "joint_rgb_inference"
        save()
        print(f"Starting joint {args.views}-view RGB-only inference on MPS ({size[0]}x{size[1]}, float32)", flush=True)
        t0 = time.perf_counter()
        with torch.inference_mode():
            predictions = runner.model.infer(views, memory_efficient_inference=True, minibatch_size=1,
                                             use_amp=False, apply_mask=False, mask_edges=False)
        torch.mps.synchronize()
        metrics["timings"]["inference_seconds"] = time.perf_counter() - t0
        metrics["returned_views"] = len(predictions)
        if len(predictions) != args.views:
            raise ValueError("Joint inference did not return all requested predictions")
        metrics["stage"] = "save_native_predictions"
        save()
        for prediction, image, record, path in zip(predictions, images, metrics["frames"], canonical):
            color = _require_identity_grid(_numpy(prediction["img_no_norm"]), image)
            depth = _numpy(prediction["depth_z"]).squeeze(-1)
            conf, native = _numpy(prediction["conf"]), _numpy(prediction["non_ambiguous_mask"])
            K, c2w = _numpy(prediction["intrinsics"]), _numpy(prediction["camera_poses"])
            points = _numpy(prediction["pts3d"])
            valid = validate_frame(color, depth, conf, native, K, c2w)
            if points.shape != (*depth.shape, 3):
                raise ValueError("Native world points do not share the RGB grid")
            gy, gx = np.gradient(depth)
            valid &= np.isfinite(points).all(-1) & (np.hypot(gx, gy) / np.maximum(depth, 1e-6) < 0.08)
            if not valid.any():
                raise ValueError("Native depth edge filter removed all support")
            np.savez_compressed(path.parent / "prediction.npz", depth_z=depth, confidence=conf,
                                native_mask=native, valid_mask=valid, K=K, c2w=c2w, pts3d=points)
            record.update(prediction_sha256=digest(path.parent / "prediction.npz"), valid_pixels=int(valid.sum()),
                          total_pixels=int(depth.size), valid_depth_min_max=[float(depth[valid].min()), float(depth[valid].max())],
                          **pointmap_residuals(points, depth, K, c2w, valid))
            metrics["processed_views"] += 1
            save()
        del predictions, runner, views
        torch.mps.empty_cache()
        metrics["stage"] = "tsdf_fusion"
        save()
        import open3d as o3d
        import trimesh

        t0 = time.perf_counter()
        volume = new_volume()
        for image, path in zip(images, canonical):
            with np.load(path.parent / "prediction.npz", allow_pickle=False) as native:
                integrate(volume, image, np.where(native["valid_mask"], native["depth_z"], 0), native["K"], native["c2w"])
        mesh = volume.extract_triangle_mesh()
        mesh.remove_degenerate_triangles().remove_duplicated_triangles().remove_unreferenced_vertices()
        mesh.compute_vertex_normals()
        vertices, faces, colors = np.asarray(mesh.vertices), np.asarray(mesh.triangles), np.asarray(mesh.vertex_colors)
        if not len(faces) or not np.isfinite(vertices).all() or not mesh.has_vertex_colors():
            raise ValueError("TSDF did not produce a finite, colored surface")
        ply, glb = args.output / "surface.ply", args.output / "room.glb"
        if not o3d.io.write_triangle_mesh(str(ply), mesh):
            raise IOError("Could not save PLY mesh")
        trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=np.rint(colors * 255).astype(np.uint8), process=False).export(glb)
        loaded = trimesh.load(glb, force="mesh", process=False)
        loaded_ply = o3d.io.read_triangle_mesh(str(ply))
        if (not np.allclose(loaded.vertices, vertices, atol=1e-5) or not np.array_equal(loaded.faces, faces)
                or not np.allclose(np.asarray(loaded_ply.vertices), vertices)
                or not np.array_equal(np.asarray(loaded_ply.triangles), faces)):
            raise ValueError("Exported PLY/GLB readback did not preserve mesh")
        metrics.update(status="execution_complete", stage="complete", all_requested_views_processed=metrics["processed_views"] == args.views,
                       mesh={"vertices": len(vertices), "faces": len(faces), "bounds": [vertices.min(0).tolist(), vertices.max(0).tolist()],
                             "ply_glb_readback_passed": True, "hole_filling": False},
                       outputs={p.name: {"bytes": p.stat().st_size, "sha256": digest(p)} for p in (ply, glb)})
        metrics["timings"]["fusion_export_seconds"] = time.perf_counter() - t0
    except BaseException as error:
        metrics.update(status="failed", error=repr(error), all_requested_views_processed=metrics["processed_views"] == args.views)
        raise
    finally:
        stop.set()
        if sampler is not None:
            sampler.join(timeout=1)
        metrics["memory"]["process_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        metrics["timings"]["total_seconds"] = time.perf_counter() - started
        save()
    print(json.dumps({"status": metrics["status"], "quality_status": metrics["quality_status"], "processed_views": metrics["processed_views"], "output": str(args.output),
                      "timings": metrics["timings"], "memory": metrics["memory"]}, indent=2), flush=True)


def self_check():
    height, width = 60, 80
    K = np.array([[80., 0, 39.5], [0, 80., 29.5], [0, 0, 1.]])
    c2w = np.eye(4)
    angle = 0.3
    c2w[:3, :3] = [[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]]
    c2w[:3, 3] = [0.4, -0.2, 0.1]
    color = np.full((height, width, 3), [180, 60, 20], np.uint8)
    depth, conf, mask = np.full((height, width), 2.), np.ones((height, width)), np.ones((height, width), bool)
    assert validate_frame(color, depth, conf, mask, K, c2w).all()
    yy, xx = np.indices(depth.shape)
    native_local = np.stack([(xx - K[0, 2]) / K[0, 0] * depth, (yy - K[1, 2]) / K[1, 1] * depth, depth], axis=-1)
    points = native_local @ c2w[:3, :3].T + c2w[:3, 3]
    assert pointmap_residuals(points, depth, K, c2w, mask)["pointmap_xyz_reprojection_error_p95_px"] < 1e-10
    native_local[..., 0] += .25
    wrong_xy = native_local @ c2w[:3, :3].T + c2w[:3, 3]
    residuals = pointmap_residuals(wrong_xy, depth, K, c2w, mask)
    assert residuals["pointmap_camera_z_residual_p95"] < 1e-10
    assert abs(residuals["pointmap_xyz_reprojection_error_p95_px"] - 10) < 1e-10
    assert residuals["pointmap_reprojection_invalid_pixels"] == 0
    for which in ("negative_z", "invalid_K", "wrong_domain", "reflected_pose"):
        d, k, pose, rgb = depth.copy(), K.copy(), c2w.copy(), color
        if which == "negative_z": d[:] = -2
        if which == "invalid_K": k[0, 0] = 0
        if which == "wrong_domain": rgb = color[:-1]
        if which == "reflected_pose": pose[:3, 0] *= -1
        try:
            validate_frame(rgb, d, conf, mask, k, pose)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Invalid {which} was admitted")
    volume = new_volume()
    integrate(volume, color, depth, K, c2w)
    mesh = volume.extract_triangle_mesh()
    local = (np.asarray(mesh.vertices) - c2w[:3, 3]) @ c2w[:3, :3]
    assert len(mesh.triangles) > 100 and np.quantile(np.abs(local[:, 2] - 2), .95) < VOXEL
    import cv2

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        video = root / "check.mp4"
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (80, 60))
        assert writer.isOpened()
        for index in range(16):
            writer.write(np.full((60, 80, 3), [index * 10, 30, 180], np.uint8))
        writer.release()
        selected, count, meta = selected_video(video, root)
        expected = np.rint(np.linspace(0, 15, DEFAULT_VIEWS)).astype(int).tolist()
        assert count == 16 and len(selected) == DEFAULT_VIEWS
        assert [x["frame_index"] for x in meta["selected_frames"]] == expected
        np.testing.assert_allclose([x[0] for x in selected], np.array(expected) / 10, atol=0.002)
        for (_, path), index in zip(selected, expected):
            pixel = np.asarray(Image.open(path)).mean(axis=(0, 1))
            np.testing.assert_allclose(pixel, [180, 30, index * 10], atol=6)
        empty = root / "empty.mp4"
        empty.touch()
        try:
            selected_video(empty, root)
        except ValueError:
            pass
        else:
            raise AssertionError("Empty video was admitted")
    print("self-check passed: correct Z/wrong XY produces 10px residual; z/K/domain/pose, inverse-c2w TSDF, MP4 indices/times/RGB, empty video rejection. This does not validate reconstruction quality.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--views", type=int, default=DEFAULT_VIEWS, choices=range(2, 33))
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--dataset", type=Path)
    source.add_argument("--video", type=Path)
    for name in ("output", "serving-root", "vendor-dir", "model-dir"):
        parser.add_argument(f"--{name}", type=Path)
    args = parser.parse_args()
    if args.self_check:
        self_check()
    elif ((args.dataset is None and args.video is None) or args.output is None
          or (not args.prepare_only and any(getattr(args, k) is None for k in ("serving_root", "vendor_dir", "model_dir")))):
        parser.error("Choose --dataset or --video and --output; inference also requires serving/vendor/model paths")
    else:
        reconstruct(args)
