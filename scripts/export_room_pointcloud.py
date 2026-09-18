"""Export saved native RGB point predictions, with no reprojection or fusion.

Usage: python scripts/export_room_pointcloud.py --run RUN --output NEW_DIRECTORY
       python scripts/export_room_pointcloud.py --self-check
Only the saved valid_mask selects points. Full PLY retains every selected point;
the browser binary uses a deterministic linear-pixel stride, with provenance.
"""

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image

from reconstruct_room_rgb import digest


PLY_DTYPE = np.dtype([
    ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
    ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ("frame_index", "<u4"), ("pixel_index", "<u4"), ("confidence", "<f4"),
])


def read_frame(run, record):
    frame_id = record["frame_id"]
    if not isinstance(frame_id, str) or Path(frame_id).name != frame_id or frame_id in (".", ".."):
        raise ValueError("Frame ID must name a directory directly under frames")
    frame = run / "frames" / frame_id
    for filename, key in (("prediction.npz", "prediction_sha256"), ("rgb.png", "processed_rgb_sha256")):
        if digest(frame / filename) != record[key]:
            raise ValueError(f"{frame_id}/{filename} differs from the recorded source hash")
    with np.load(frame / "prediction.npz", allow_pickle=False) as saved:
        points, valid, confidence = saved["pts3d"], saved["valid_mask"], saved["confidence"]
    with Image.open(frame / "rgb.png") as source:
        if source.mode != "RGB":
            raise ValueError("Canonical RGB must already be RGB; no color conversion is performed")
        rgb = np.asarray(source)
    if (valid.ndim != 2 or points.shape != (*valid.shape, 3) or rgb.shape != points.shape
            or confidence.shape != valid.shape or points.dtype != np.float32
            or confidence.dtype != np.float32 or rgb.dtype != np.uint8
            or not np.isin(valid, [0, 1]).all()):
        raise ValueError("Expected native float32 XYZ/confidence and uint8 RGB sharing a binary validity grid")
    valid = valid.astype(bool)
    if (not np.isfinite(points[valid]).all() or not np.isfinite(confidence[valid]).all()
            or int(valid.sum()) != record["valid_pixels"] or valid.size != record["total_pixels"]
            or list(valid.shape[::-1]) != record["processed_size_wh"]):
        raise ValueError("Saved valid points must be finite and match the recorded pixel domain/count")
    return points.reshape(-1, 3), rgb.reshape(-1, 3), valid.ravel(), confidence.ravel()


def export(run, output, max_display=300_000):
    run, output = run.resolve(), output.resolve()
    if output.exists():
        raise ValueError("Output must be a new directory; saved runs and exports are immutable")
    if output.parent != run:
        raise ValueError("Output must be a new direct child of the saved run for relative source links")
    if not 1 <= max_display <= 300_000:
        raise ValueError("Browser display limit must be between 1 and 300000 points")
    metrics_path = run / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    records = metrics["frames"]
    if (not records or len(records) != metrics["processed_views"]
            or len({record["frame_id"] for record in records}) != len(records)):
        raise ValueError("Expected a complete, uniquely identified list of saved frames")
    masks = []
    for record in records:
        _, _, valid, _ = read_frame(run, record)
        masks.append(valid)
    total = sum(int(mask.sum()) for mask in masks)
    if not total:
        raise ValueError("Run has no valid native points")
    # ponytail: a fixed linear-pixel stride is bounded and reproducible, not
    # spatially uniform. Spatial sampling would require a separate explicit task.
    stride = max(1, (sum(mask.size for mask in masks) + max_display - 1) // max_display)
    while sum((mask.size + stride - 1) // stride for mask in masks) > max_display:
        stride += 1
    selections = [np.flatnonzero(mask & (np.arange(mask.size) % stride == 0)) for mask in masks]
    if not all(len(indices) for indices in selections):
        raise ValueError("Display stride removed all support from a frame; increase the display limit")
    output.mkdir(parents=True)
    header = ("ply\nformat binary_little_endian 1.0\n"
              "comment Native predicted XYZ; units and shared geometry unverified\n"
              "comment frame_index refers to manifest frames; pixel_index is row-major in canonical RGB\n"
              f"element vertex {total}\nproperty float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\n"
              "property uint frame_index\nproperty uint pixel_index\nproperty float confidence\nend_header\n").encode()
    frames = []
    bounds_min, bounds_max = np.full(3, np.inf), np.full(3, -np.inf)
    full_offset = display_offset = 0
    with (output / "points.ply").open("wb") as full, (output / "points.bin").open("wb") as display, (output / "points.provenance.bin").open("wb") as provenance:
        full.write(header)
        for frame_index, (record, selection) in enumerate(zip(records, selections)):
            points, rgb, valid, confidence = read_frame(run, record)
            indices = np.flatnonzero(valid)
            vertices = np.empty(len(indices), dtype=PLY_DTYPE)
            for axis, name in enumerate(("x", "y", "z")):
                vertices[name] = points[indices, axis]
            for channel, name in enumerate(("red", "green", "blue")):
                vertices[name] = rgb[indices, channel]
            vertices["frame_index"], vertices["pixel_index"] = frame_index, indices
            vertices["confidence"] = confidence[indices]
            vertices.tofile(full)
            packed = np.zeros((len(selection), 9), dtype="<f4")
            packed[:, :3], packed[:, 6:] = points[selection], rgb[selection].astype(np.float32) / 255
            packed.tofile(display)
            np.column_stack((np.full(len(selection), frame_index, dtype="<u4"), selection.astype("<u4"))).astype("<u4").tofile(provenance)
            bounds_min = np.minimum(bounds_min, points[indices].min(0))
            bounds_max = np.maximum(bounds_max, points[indices].max(0))
            frames.append({
                "frame_index": frame_index, "frame_id": record["frame_id"], "timestamp": record["timestamp"],
                "canonical_size_wh": record["processed_size_wh"],
                "prediction": f"../frames/{record['frame_id']}/prediction.npz",
                "prediction_sha256": record["prediction_sha256"],
                "canonical_rgb": f"../frames/{record['frame_id']}/rgb.png",
                "canonical_rgb_sha256": record["processed_rgb_sha256"],
                "source_path": record["source_path"], "source_sha256": record["source_sha256"],
                "source_to_processed_pixel_centres": record["source_to_processed_pixel_centres"],
                "valid_mask_sha256": hashlib.sha256(valid.astype(np.uint8).tobytes()).hexdigest(),
                "valid_mask_hash_encoding": "row-major uint8, one byte per pixel, 0/1",
                "total_pixels": len(valid), "valid_points": len(indices),
                "full_point_offset": full_offset, "display_point_offset": display_offset,
                "display_points": len(selection),
            })
            full_offset += len(indices)
            display_offset += len(selection)
    import trimesh

    display = np.fromfile(output / "points.bin", dtype="<f4").reshape(-1, 9)
    trimesh.points.PointCloud(
        display[:, :3], colors=np.rint(display[:, 6:] * 255).astype(np.uint8),
        metadata={"label": "Native predicted RGB point cloud", "metric_scale_known": False,
                  "units": "native predicted scale; not independently calibrated"},
    ).export(output / "points.glb")
    manifest = {
        "version": 1, "run_id": run.name, "source_metrics": "../metrics.json",
        "source_metrics_sha256": digest(metrics_path), "export_script_sha256": digest(__file__),
        "label": "Native predicted RGB point cloud", "model": metrics.get("model"),
        "coordinates": "Saved pts3d native predicted world XYZ, copied unchanged",
        "metric_scale_known": False, "units": "native predicted scale; not independently calibrated",
        "shared_coordinate_geometry_verified": False, "dynamic_object_mask_applied": False,
        "selection": "Saved valid_mask exactly; no additional geometry or confidence filtering",
        "saved_mask_contract": metrics.get("mask_contract"),
        "confidence_semantics": metrics.get("confidence_semantics"),
        "operations": {"model_calls": 0, "depth_reprojection": False, "pose_transform": False,
                       "icp": False, "denoising": False, "voxel_merge": False, "triangulation": False},
        "bounds": {"min": bounds_min.tolist(), "max": bounds_max.tolist()},
        "full_cloud": {"path": "points.ply", "point_count": total, "vertex_bytes": PLY_DTYPE.itemsize,
                       "header_bytes": len(header), "frame_provenance": "frame_index and pixel_index per vertex"},
        "display": {"path": "points.bin", "point_count": display_offset, "maximum_points": max_display,
                    "glb": "points.glb", "glb_primitive_mode": 0,
                    "byte_offset": 0, "stride": 9, "dtype": "little-endian float32",
                    "layout": ["x", "y", "z", "unused_normal_x", "unused_normal_y", "unused_normal_z", "red", "green", "blue"],
                    "normal_values": "zero placeholders; render unlit points, not a surface",
                    "rgb_encoding": "canonical RGB uint8 / 255, stored as float32",
                    "sampling": "Per frame, keep valid row-major pixel indices divisible by pixel_stride",
                    "pixel_stride": stride, "provenance_path": "points.provenance.bin",
                    "provenance_dtype": "little-endian uint32", "provenance_stride": 2,
                    "provenance_layout": ["frame_index", "pixel_index"],
                    "provenance_order": "One record per display point, in identical order"},
        "frames": frames,
    }
    export_frame_clouds(run, output, manifest)
    verify(run, output, manifest)
    manifest["verification"] = {"full_points_exact_xyz_rgb_confidence_frame_pixel": True,
                                "display_points_exact_xyz_rgb_frame_pixel": True,
                                "glb_points_mode_and_exact_xyz_rgb_order": True,
                                "frame_glbs_exact_display_subsets": True,
                                "display_within_limit": True, "source_prediction_rgb_hashes_match": True}
    manifest["outputs"] = {name: {"bytes": (output / name).stat().st_size, "sha256": digest(output / name)}
                           for name in ["points.ply", "points.bin", "points.provenance.bin", "points.glb"]
                           + [frame["display"]["glb"] for frame in frames]}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest


def export_frame_clouds(run, output, manifest):
    """Expose the same display subsets per source frame, without changing points."""
    import trimesh

    (output / "frames").mkdir()
    display = np.fromfile(output / "points.bin", dtype="<f4").reshape(-1, 9)
    for frame in manifest["frames"]:
        start = frame["display_point_offset"]
        selected = display[start:start + frame["display_points"]]
        relative = f"frames/{frame['frame_id']}.glb"
        trimesh.points.PointCloud(
            selected[:, :3], colors=np.rint(selected[:, 6:] * 255).astype(np.uint8),
            metadata={"frame_id": frame["frame_id"], "frame_index": frame["frame_index"],
                      "metric_scale_known": False, "units": "native predicted scale; not independently calibrated"},
        ).export(output / relative)
        with np.load(run / "frames" / frame["frame_id"] / "prediction.npz", allow_pickle=False) as saved:
            K, c2w = saved["K"], saved["c2w"]
        if K.shape != (3, 3) or c2w.shape != (4, 4) or not np.isfinite(K).all() or not np.isfinite(c2w).all():
            raise ValueError("Source camera must have finite saved K and c2w")
        frame.update(width=frame["canonical_size_wh"][0], height=frame["canonical_size_wh"][1],
                     K=K.tolist(), camera_to_world=c2w.tolist(),
                     camera_pixel_domain="Canonical RGB; saved predicted K/c2w, not verified calibration",
                     display={"glb": relative, "point_count": len(selected), "glb_primitive_mode": 0})


def verify_glb(path, display):
    import trimesh

    glb = path.read_bytes()
    document = json.loads(glb[20:20 + int.from_bytes(glb[12:16], "little")])
    assert len(document["meshes"]) == 1 and document["meshes"][0]["primitives"][0]["mode"] == 0
    loaded = trimesh.load(path, force="scene", process=False)
    assert len(loaded.geometry) == 1
    cloud = next(iter(loaded.geometry.values()))
    assert np.array_equal(cloud.vertices, display[:, :3])
    assert np.array_equal(cloud.colors[:, :3], np.rint(display[:, 6:] * 255).astype(np.uint8))


def verify(run, output, manifest):
    """Read every exported point back and compare directly with its saved pixel."""
    full_meta, display_meta = manifest["full_cloud"], manifest["display"]
    full = np.memmap(output / "points.ply", mode="r", dtype=PLY_DTYPE, offset=full_meta["header_bytes"])
    display = np.fromfile(output / "points.bin", dtype="<f4").reshape(-1, 9)
    provenance = np.fromfile(output / "points.provenance.bin", dtype="<u4").reshape(-1, 2)
    assert len(full) == full_meta["point_count"]
    assert len(display) == len(provenance) == display_meta["point_count"] <= display_meta["maximum_points"]
    assert np.isfinite(display).all() and np.all(display[:, 3:6] == 0)
    verify_glb(output / "points.glb", display)
    records = json.loads((run / "metrics.json").read_text())["frames"]
    for frame, record in zip(manifest["frames"], records):
        points, rgb, valid, confidence = read_frame(run, record)
        indices = np.flatnonzero(valid)
        start = frame["full_point_offset"]
        vertices = full[start:start + len(indices)]
        assert np.array_equal(np.column_stack([vertices[name] for name in ("x", "y", "z")]), points[indices])
        assert np.array_equal(np.column_stack([vertices[name] for name in ("red", "green", "blue")]), rgb[indices])
        assert np.array_equal(vertices["pixel_index"], indices)
        assert np.all(vertices["frame_index"] == frame["frame_index"])
        assert np.array_equal(vertices["confidence"], confidence[indices])
        selection = np.flatnonzero(valid & (np.arange(len(valid)) % display_meta["pixel_stride"] == 0))
        start = frame["display_point_offset"]
        selected = display[start:start + len(selection)]
        selected_provenance = provenance[start:start + len(selection)]
        assert np.array_equal(selected[:, :3], points[selection])
        assert np.array_equal(selected[:, 6:], rgb[selection].astype(np.float32) / 255)
        assert np.all(selected_provenance[:, 0] == frame["frame_index"])
        assert np.array_equal(selected_provenance[:, 1], selection)
        verify_glb(output / frame["display"]["glb"], selected)


def self_check():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        records = []
        for index in range(2):
            frame = root / "frames" / f"frame_{index:02d}"
            frame.mkdir(parents=True)
            points = (np.arange(36, dtype=np.float32).reshape(3, 4, 3) + index * 100) / 7
            rgb = (np.arange(36).reshape(3, 4, 3) * 7 + index).astype(np.uint8)
            valid = np.ones((3, 4), dtype=bool)
            valid[0, 1] = False
            points[0, 1] = np.nan
            np.savez(frame / "prediction.npz", pts3d=points, valid_mask=valid,
                     confidence=np.full((3, 4), 0.25 + index, np.float32), K=np.eye(3), c2w=np.eye(4))
            Image.fromarray(rgb).save(frame / "rgb.png")
            records.append({"frame_id": frame.name, "timestamp": index, "valid_pixels": 11, "total_pixels": 12,
                            "processed_size_wh": [4, 3], "prediction_sha256": digest(frame / "prediction.npz"),
                            "processed_rgb_sha256": digest(frame / "rgb.png"), "source_path": str(frame / "rgb.png"),
                            "source_sha256": digest(frame / "rgb.png"), "source_to_processed_pixel_centres": np.eye(3).tolist()})
        (root / "metrics.json").write_text(json.dumps({"frames": records, "processed_views": 2}))
        manifest = export(root, root / "cloud", max_display=7)
        assert manifest["full_cloud"]["point_count"] == 22
        assert manifest["display"]["point_count"] == 6 and manifest["display"]["pixel_stride"] == 4
        provenance = np.fromfile(root / "cloud" / "points.provenance.bin", dtype="<u4").reshape(-1, 2)
        assert provenance.tolist() == [[0, 0], [0, 4], [0, 8], [1, 0], [1, 4], [1, 8]]
        damaged = np.memmap(root / "cloud" / "points.bin", mode="r+", dtype="<f4")
        damaged[0] += 1
        damaged.flush()
        try:
            verify(root, root / "cloud", manifest)
        except AssertionError:
            pass
        else:
            raise AssertionError("Readback verification admitted modified XYZ")
    print("self-check passed: full native XYZ/RGB/confidence/frame/pixel readback; deterministic bounded display; corrupted XYZ rejected")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-display", type=int, default=300_000)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
    elif args.run is None or args.output is None:
        parser.error("Choose --run and --output, or --self-check")
    else:
        result = export(args.run, args.output, args.max_display)
        print(json.dumps({"output": str(args.output), "full_points": result["full_cloud"]["point_count"],
                          "display_points": result["display"]["point_count"], "verification": result["verification"]}, indent=2))
