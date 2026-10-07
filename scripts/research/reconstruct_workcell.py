"""Fuse existing joint Pi3X RGB-D observations with Open3D TSDF; no inference.

  .venv/bin/python scripts/research/reconstruct_workcell.py --self-check
  .venv/bin/python scripts/research/reconstruct_workcell.py --selection <selection.json> --output <new directory>
"""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import open3d as o3d
from PIL import Image
import trimesh


VOXEL = 0.012
TRUNCATION = 0.048


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check_pose(pose):
    if (pose.shape != (4, 4) or not np.isfinite(pose).all()
            or not np.allclose(pose[3], [0, 0, 0, 1])
            or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-4)
            or not np.isclose(np.linalg.det(pose[:3, :3]), 1, atol=1e-4)):
        raise ValueError("Expected a finite rigid OpenCV camera-to-world transform")


def camera_points(points, pose):
    world_to_camera = np.linalg.inv(pose)
    return points @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]


def rectangle_mask(shape, rectangle):
    height, width = shape
    if len(rectangle) != 4 or any(int(v) != v for v in rectangle):
        raise ValueError("Canonical rectangles must contain four integer LTRB coordinates")
    left, top, right, bottom = map(int, rectangle)
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        raise ValueError("Rectangle lies outside the canonical image")
    mask = np.zeros(shape, dtype=bool)
    mask[top:bottom, left:right] = True
    return mask


def selected_depth(points, confidence, valid, pose, reference_pose, selection, frame_id):
    check_pose(pose)
    check_pose(reference_pose)
    if points.shape != (*valid.shape, 3) or confidence.shape != valid.shape:
        raise ValueError("Point maps, confidence and validity must share one grid")
    local = camera_points(points, pose)
    reference = camera_points(points, reference_pose)
    lo, hi = (np.asarray(selection[key], dtype=float) for key in ("crop_box_camera_min", "crop_box_camera_max"))
    if lo.shape != (3,) or hi.shape != (3,) or not np.isfinite([lo, hi]).all() or np.any(lo >= hi):
        raise ValueError("Invalid first-camera crop bounds")
    mask = np.ones(valid.shape, dtype=bool)
    counts = {"total_pixels": int(mask.size)}

    def keep(label, condition):
        before = int(mask.sum())
        mask[:] &= condition
        counts[label] = {"removed": before - int(mask.sum()), "remaining": int(mask.sum())}

    keep("finite", np.isfinite(points).all(-1) & np.isfinite(confidence) & np.isfinite(local).all(-1))
    keep("original_valid", valid.astype(bool))
    keep("real_image_content", rectangle_mask(valid.shape, selection["content_rect"]))
    keep("confidence", confidence >= selection["minimum_confidence"])
    keep("positive_camera_z", local[..., 2] > 0)
    excluded = np.zeros(valid.shape, dtype=bool)
    for rectangle in selection["exclude_rects"][frame_id]:
        excluded |= rectangle_mask(valid.shape, rectangle)
    keep("dynamic_exclusions", ~excluded)
    keep("reference_camera_crop", ((reference >= lo) & (reference <= hi)).all(-1))
    depth = np.where(mask, local[..., 2], 0).astype(np.float32)
    return depth, mask, counts


def volume(voxel=VOXEL, truncation=TRUNCATION):
    return o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel, sdf_trunc=truncation,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)


def integrate(tsdf, color, depth, K, pose):
    height, width = depth.shape
    if color.shape != (height, width, 3) or color.dtype != np.uint8:
        raise ValueError("Expected pixel-aligned uint8 RGB")
    if (K.shape != (3, 3) or not np.isfinite(K).all() or not np.allclose(K[2], [0, 0, 1])
            or min(K[0, 0], K[1, 1]) <= 0 or K[0, 1] != 0 or K[1, 0] != 0):
        raise ValueError("TSDF requires finite positive pinhole intrinsics with zero skew")
    intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, K[0, 0], K[1, 1], K[0, 2], K[1, 2])
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(np.ascontiguousarray(color)), o3d.geometry.Image(np.ascontiguousarray(depth)),
        depth_scale=1.0, depth_trunc=float(depth.max()) + TRUNCATION, convert_rgb_to_intensity=False)
    tsdf.integrate(rgbd, intrinsic, np.linalg.inv(pose))


def reconstruct(selection_path, output):
    if output.exists():
        raise ValueError("Output must be a new immutable directory")
    selection = json.loads(selection_path.read_text())
    if selection["minimum_confidence"] != 0.1 or selection["metric_scale_known"] is not False:
        raise ValueError("Expected fixed confidence=0.1 and uncalibrated model units")
    if selection["content_rect"] != [63, 0, 455, 518]:
        raise ValueError("This experiment uses the verified BOR1 letterbox content rectangle")
    source = Path(selection["source_geometry"])
    frames = sorted((source / "frames").glob("frame_*"))
    if len(frames) != 4:
        raise ValueError("Expected the four frozen source frame caches")
    include = selection["include_frames"]
    if not isinstance(include, list) or not include or len(include) != len(set(include)) or not set(include) <= {f.name for f in frames}:
        raise ValueError("include_frames must explicitly select unique existing BOR1 frames")
    if not set(include) <= set(selection["exclude_rects"]):
        raise ValueError("Every included frame requires an explicit exclusion entry")
    frames = [source / "frames" / name for name in include]
    if selection["reference_frame"] in include and selection["exclude_rects"][selection["reference_frame"]]:
        raise ValueError("The reference frame must retain its observed dynamic state")
    reference_pose = np.load(source / "frames" / selection["reference_frame"] / "camera_to_world.npy", allow_pickle=False)
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=False)
    metrics = {"status": "running", "source_geometry": str(source), "selection_sha256": digest(selection_path),
               "selection": selection, "open3d_version": o3d.__version__, "trimesh_version": trimesh.__version__,
               "include_frames": include,
               "parameters": {"voxel_length": VOXEL, "sdf_trunc": TRUNCATION, "minimum_confidence": 0.1,
                              "units": "native model estimated units; not physically calibrated", "color": "RGB8"},
               "metric_scale_known": False, "world_coordinates_changed": False,
               "method": "ScalableTSDFVolume integration of masked z-depth with inverse native c2w; marching-cubes surface extraction",
               "depth_contract": "Z from native world points in each camera; fitted native K defines integration rays. Native point-map XY is not identical to the fitted pinhole grid.",
               "unknown_space": "Zero-depth pixels contribute no observation; no Poisson or mesh hole filling",
               "frames": [], "fusion_seconds": 0.0}
    if "crop_repair_diagnostics" in selection:
        metrics["crop_repair_diagnostics"] = selection["crop_repair_diagnostics"]
    tsdf = volume()
    try:
        for frame in frames:
            names = ["pts3d.npy", "conf.npy", "valid_mask.npy", "camera_to_world.npy", "intrinsics.npy", "canonical.png"]
            hashes = {name: digest(frame / name) for name in names}
            points, conf, valid, pose, K = [np.load(frame / name, allow_pickle=False) for name in names[:-1]]
            color = np.asarray(Image.open(frame / "canonical.png").convert("RGB"))
            if color.shape != (518, 518, 3):
                raise ValueError("Expected frozen 518x518 canonical RGB")
            depth, mask, counts = selected_depth(points, conf, valid, pose, reference_pose, selection, frame.name)
            if not mask.any():
                raise ValueError(f"No selected depth in {frame.name}")
            local = camera_points(points, pose)
            projected = local @ K.T
            yy, xx = np.indices(mask.shape)
            residual = np.linalg.norm(projected[mask, :2] / projected[mask, 2:] - np.stack([xx, yy], -1)[mask], axis=-1)
            t0 = time.perf_counter()
            integrate(tsdf, color, depth, K, pose)
            elapsed = time.perf_counter() - t0
            metrics["fusion_seconds"] += elapsed
            metrics["frames"].append({"frame_id": frame.name, "input_sha256": hashes, "selection_counts": counts,
                                       "integrate_seconds": elapsed, "selected_depth_min_max": [float(depth[mask].min()), float(depth[mask].max())],
                                       "selected_native_pinhole_residual_p95_px": float(np.percentile(residual, 95))})
            print(f"Integrated {frame.name}: {int(mask.sum())} selected pixels in {elapsed:.3f}s", flush=True)
        t0 = time.perf_counter()
        mesh = tsdf.extract_triangle_mesh()
        metrics["extraction_seconds"] = time.perf_counter() - t0
        metrics["extracted_vertex_count"] = len(mesh.vertices)
        # Keep the requested crop at the TSDF interpolation boundary as well; no coordinate change.
        ref_vertices = camera_points(np.asarray(mesh.vertices), reference_pose)
        outside = ((ref_vertices < selection["crop_box_camera_min"]) | (ref_vertices > selection["crop_box_camera_max"])).any(-1)
        metrics["boundary_vertices_removed"] = int(outside.sum())
        mesh.remove_vertices_by_mask(outside)
        mesh.remove_degenerate_triangles().remove_duplicated_triangles().remove_unreferenced_vertices()
        mesh.compute_vertex_normals()
        vertices, faces, colors = np.asarray(mesh.vertices), np.asarray(mesh.triangles), np.asarray(mesh.vertex_colors)
        if not len(vertices) or not len(faces) or not np.isfinite(vertices).all() or not np.isfinite(colors).all():
            raise ValueError("TSDF produced an empty or nonfinite surface")
        metrics.update(vertex_count=len(vertices), face_count=len(faces), mesh_finite=True,
                       world_bounds=[vertices.min(0).tolist(), vertices.max(0).tolist()], has_vertex_colors=mesh.has_vertex_colors())
        ply = output / "surface.ply"
        if not o3d.io.write_triangle_mesh(str(ply), mesh, write_ascii=False):
            raise IOError("Open3D could not write surface.ply")
        colors_u8 = np.rint(np.clip(colors, 0, 1) * 255).astype(np.uint8)
        exported = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=colors_u8, process=False)
        exported.export(output / "workcell.glb")
        read_ply = o3d.io.read_triangle_mesh(str(ply))
        read_glb = trimesh.load(output / "workcell.glb", force="mesh", process=False)
        checks = {"ply_vertex_count": len(read_ply.vertices), "ply_face_count": len(read_ply.triangles),
                  "ply_vertices_match": bool(np.allclose(np.asarray(read_ply.vertices), vertices)),
                  "ply_faces_match": bool(np.array_equal(np.asarray(read_ply.triangles), faces)),
                  "ply_has_colors": read_ply.has_vertex_colors(), "glb_vertex_count": len(read_glb.vertices),
                  "glb_face_count": len(read_glb.faces), "glb_finite": bool(np.isfinite(read_glb.vertices).all()),
                  "glb_vertices_match": bool(np.allclose(read_glb.vertices, vertices, atol=1e-6)),
                  "glb_faces_match": bool(np.array_equal(read_glb.faces, faces)), "glb_has_vertex_colors": read_glb.visual.kind == "vertex"}
        if not all(checks[key] for key in ["ply_vertices_match", "ply_faces_match", "ply_has_colors", "glb_finite", "glb_vertices_match", "glb_faces_match", "glb_has_vertex_colors"]):
            raise ValueError(f"Mesh readback validation failed: {checks}")
        metrics["readback"] = checks
        metrics["outputs"] = {p.name: {"bytes": p.stat().st_size, "sha256": digest(p)} for p in [ply, output / "workcell.glb"]}
        metrics["status"] = "complete"
    except Exception as error:
        metrics["status"], metrics["error"] = "failed", repr(error)
        raise
    finally:
        metrics["elapsed_seconds"] = time.perf_counter() - started
        (output / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(output), "vertices": len(vertices), "faces": len(faces), "seconds": metrics["elapsed_seconds"]}), flush=True)


def self_check():
    height, width = 60, 80
    K = np.array([[80., 0, 39.5], [0, 80., 29.5], [0, 0, 1.]])
    angle = 0.3
    reference = np.eye(4)
    reference[:3, :3] = [[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]]
    reference[:3, 3] = [0.4, -0.2, 0.1]
    selection = {"crop_box_camera_min": [-0.7, -0.5, 1.9], "crop_box_camera_max": [0.7, 0.5, 2.1],
                 "content_rect": [0, 0, width, height], "minimum_confidence": 0.1,
                 "exclude_rects": {"one": [[30, 20, 40, 30]], "two": []}}
    yy, xx = np.indices((height, width))
    local = np.stack([(xx - K[0, 2]) / K[0, 0] * 2, (yy - K[1, 2]) / K[1, 1] * 2, np.full_like(xx, 2)], axis=-1)
    tsdf = volume(0.025, 0.1)
    for name, offset in [("one", 0), ("two", 0.15)]:
        pose = reference.copy()
        pose[:3, 3] += reference[:3, :3] @ [offset, 0, 0]
        world = local @ pose[:3, :3].T + pose[:3, 3]
        depth, mask, _ = selected_depth(world, np.ones((height, width)), np.ones((height, width), bool), pose, reference, selection, name)
        np.testing.assert_allclose(depth[mask], 2, atol=1e-6)
        expected = ((local[..., 0] + offset >= -0.7) & (local[..., 0] + offset <= 0.7) & (np.abs(local[..., 1]) <= 0.5))
        if name == "one":
            expected[20:30, 30:40] = False
        # Floating-point crop-boundary ties are excluded from this exact-mask assertion.
        interior = (np.abs(np.abs(local[..., 0] + offset) - 0.7) > 1e-6) & (np.abs(np.abs(local[..., 1]) - 0.5) > 1e-6)
        assert np.array_equal(mask[interior], expected[interior])
        integrate(tsdf, np.full((height, width, 3), [180, 60, 20], np.uint8), depth, K, pose)
    mesh = tsdf.extract_triangle_mesh()
    assert len(mesh.triangles) > 100 and np.isfinite(np.asarray(mesh.vertices)).all()
    reference_vertices = camera_points(np.asarray(mesh.vertices), reference)
    assert np.quantile(np.abs(reference_vertices[:, 2] - 2), .95) < 0.04
    print("self-check passed: nonidentity c2w, z-depth, first-camera ROI, dynamic exclusion, two-view plane TSDF", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
    else:
        if args.selection is None or args.output is None:
            parser.error("--selection and --output are required")
        reconstruct(args.selection, args.output)


if __name__ == "__main__":
    main()
