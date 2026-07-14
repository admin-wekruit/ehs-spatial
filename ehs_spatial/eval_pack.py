"""Generate deterministic calibrated EHS workcell evaluation scenes."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image


WIDTH = 512
HEIGHT = 384
FX = FY = 450.0
K = np.array(
    [
        [FX, 0.0, (WIDTH - 1) / 2],
        [0.0, FY, (HEIGHT - 1) / 2],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
FENCE = {"xmin": -1.5, "xmax": 1.5, "ymin": -1.2, "ymax": 1.2}
COLORS = {
    "factory floor": (0.58, 0.61, 0.64),
    "safety fence": (0.96, 0.72, 0.05),
    "industrial robot arm": (0.95, 0.24, 0.04),
    "step ladder": (0.12, 0.36, 0.86),
    "portable work platform": (0.05, 0.58, 0.52),
}


def _box(size: tuple[float, float, float], origin: tuple[float, float, float]):
    return o3d.geometry.TriangleMesh.create_box(*size).translate(origin)


def _sphere(center: tuple[float, float, float], radius: float):
    return o3d.geometry.TriangleMesh.create_sphere(
        radius, resolution=12
    ).translate(center)


def _cylinder_between(
    start: tuple[float, float, float] | np.ndarray,
    end: tuple[float, float, float] | np.ndarray,
    radius: float,
    resolution: int = 10,
):
    start_array = np.asarray(start, dtype=float)
    end_array = np.asarray(end, dtype=float)
    axis = end_array - start_array
    length = float(np.linalg.norm(axis))
    mesh = o3d.geometry.TriangleMesh.create_cylinder(
        radius, length, resolution=resolution
    )
    direction = axis / length
    z_axis = np.array([0.0, 0.0, 1.0])
    cross = np.cross(z_axis, direction)
    cross_norm = float(np.linalg.norm(cross))
    if cross_norm > 1e-12:
        angle = math.acos(float(np.clip(np.dot(z_axis, direction), -1.0, 1.0)))
        mesh.rotate(
            o3d.geometry.get_rotation_matrix_from_axis_angle(
                cross / cross_norm * angle
            )
        )
    elif float(np.dot(z_axis, direction)) < 0:
        mesh.rotate(o3d.geometry.get_rotation_matrix_from_xyz((math.pi, 0.0, 0.0)))
    return mesh.translate((start_array + end_array) / 2)


def _merge(parts: list[o3d.geometry.TriangleMesh], label: str):
    mesh = o3d.geometry.TriangleMesh()
    for part in parts:
        mesh += part
    mesh.paint_uniform_color(COLORS[label])
    mesh.compute_vertex_normals()
    return mesh


def _build_floor():
    return _merge([_box((8.0, 8.0, 0.05), (-4.0, -4.0, -0.05))], "factory floor")


def _build_fence():
    x0, x1 = FENCE["xmin"], FENCE["xmax"]
    y0, y1 = FENCE["ymin"], FENCE["ymax"]
    parts: list[o3d.geometry.TriangleMesh] = []
    post_xy = [(x, y) for x in (x0, 0.0, x1) for y in (y0, y1)]
    post_xy += [(x, y) for x in (x0, x1) for y in (0.0,)]
    parts.extend(
        _cylinder_between((x, y, 0.0), (x, y, 1.6), 0.035)
        for x, y in post_xy
    )
    for z in (0.12, 0.85, 1.55):
        parts.extend(
            [
                _cylinder_between((x0, y0, z), (x1, y0, z), 0.025),
                _cylinder_between((x0, y1, z), (x1, y1, z), 0.025),
                _cylinder_between((x0, y0, z), (x0, y1, z), 0.025),
                _cylinder_between((x1, y0, z), (x1, y1, z), 0.025),
            ]
        )
    # ponytail: sparse analytic bars stand in for real mesh fencing; densify here if wire-level geometry is evaluated.
    for x in np.linspace(x0 + 0.3, x1 - 0.3, 7):
        parts.extend(
            _cylinder_between((x, y, 0.12), (x, y, 1.55), 0.012, 8)
            for y in (y0, y1)
        )
    for y in np.linspace(y0 + 0.3, y1 - 0.3, 5):
        parts.extend(
            _cylinder_between((x, y, 0.12), (x, y, 1.55), 0.012, 8)
            for x in (x0, x1)
        )
    return _merge(parts, "safety fence")


def _build_robot():
    parts = [
        o3d.geometry.TriangleMesh.create_cylinder(0.32, 0.22, resolution=20).translate((-0.25, 0.2, 0.11)),
        o3d.geometry.TriangleMesh.create_cylinder(0.20, 0.34, resolution=16).translate((-0.25, 0.2, 0.39)),
    ]
    joints = [
        np.array([-0.25, 0.2, 0.55]),
        np.array([0.05, 0.15, 1.18]),
        np.array([0.48, 0.38, 1.02]),
        np.array([0.72, 0.48, 0.78]),
    ]
    parts.extend(_sphere(tuple(point), 0.15 if index < 2 else 0.11) for index, point in enumerate(joints))
    parts.extend(
        _cylinder_between(start, end, 0.11 if index == 0 else 0.08, 16)
        for index, (start, end) in enumerate(zip(joints, joints[1:]))
    )
    parts.append(_box((0.26, 0.16, 0.10), (0.68, 0.42, 0.72)))
    return _merge(parts, "industrial robot arm")


def _build_ladder(clearance_m: float):
    radius = 0.04
    front_x = FENCE["xmax"] + clearance_m + radius
    rear_x = front_x + 0.55
    top_x = (front_x + rear_x) / 2
    sides = (-0.38, 0.38)
    top_z = 1.45
    parts: list[o3d.geometry.TriangleMesh] = []
    for y in sides:
        parts.extend(
            [
                _cylinder_between((front_x, y, radius), (top_x, y, top_z), radius),
                _cylinder_between((rear_x, y, radius), (top_x, y, top_z), radius),
            ]
        )
    for z in np.linspace(0.28, 1.20, 5):
        x = front_x + (top_x - front_x) * z / top_z
        parts.append(_cylinder_between((x, sides[0], z), (x, sides[1], z), 0.025))
    parts.append(_box((0.48, 0.76, 0.07), (top_x - 0.24, sides[0], 1.30)))
    mesh = _merge(parts, "step ladder")
    measured_clearance = (
        float(mesh.get_axis_aligned_bounding_box().min_bound[0]) - FENCE["xmax"]
    )
    return mesh.translate((clearance_m - measured_clearance, 0.0, 0.0))


def _build_platform():
    parts = [_box((0.88, 0.64, 0.10), (0.38, -0.92, 0.58))]
    for x in (0.46, 1.18):
        for y in (-0.84, -0.36):
            parts.append(_cylinder_between((x, y, 0.03), (x, y, 0.58), 0.035))
    parts.extend(
        [
            _cylinder_between((0.42, -0.88, 0.30), (1.22, -0.88, 0.30), 0.025),
            _cylinder_between((0.42, -0.32, 0.30), (1.22, -0.32, 0.30), 0.025),
        ]
    )
    return _merge(parts, "portable work platform")


def _world_to_camera(
    eye: tuple[float, float, float],
    target: tuple[float, float, float],
):
    eye_array = np.asarray(eye, dtype=float)
    forward = np.asarray(target, dtype=float) - eye_array
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.stack([right, down, forward])
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = rotation
    extrinsic[:3, 3] = -rotation @ eye_array
    return extrinsic


def _normalized_bbox(mask: np.ndarray):
    rows, columns = np.nonzero(mask)
    return [
        float(columns.min() / WIDTH),
        float(rows.min() / HEIGHT),
        float((columns.max() + 1) / WIDTH),
        float((rows.max() + 1) / HEIGHT),
    ]


def _case_meshes(movable_label: str, distance_m: float):
    movable = (
        _build_ladder(distance_m)
        if movable_label == "step ladder"
        else _build_platform()
    )
    return {
        "factory floor": _build_floor(),
        "safety fence": _build_fence(),
        "industrial robot arm": _build_robot(),
        movable_label: movable,
    }


def _normal_cameras():
    target = (0.20, 0.0, 0.75)
    return [
        ((4.6, -4.4, 2.15), target),
        ((4.6, 4.4, 2.00), target),
        ((-4.4, 4.2, 2.25), target),
        ((-4.6, -4.1, 1.95), target),
    ]


def _occluded_cameras():
    return _normal_cameras()[:2] + [
        ((2.8, -2.7, 1.7), (4.4, -3.8, -0.3)),
        ((2.6, 2.3, 2.4), (4.4, 3.8, -0.8)),
    ]


def _write_frame(
    root: Path,
    case_dir: Path,
    frame_index: int,
    scene: o3d.t.geometry.RaycastingScene,
    geometry_labels: dict[int, str],
    eye: tuple[float, float, float],
    target: tuple[float, float, float],
):
    world_to_camera = _world_to_camera(eye, target)
    rays = scene.create_rays_pinhole(
        o3d.core.Tensor(K, dtype=o3d.core.Dtype.Float64),
        o3d.core.Tensor(world_to_camera, dtype=o3d.core.Dtype.Float64),
        WIDTH,
        HEIGHT,
    )
    cast = scene.cast_rays(rays)
    distances = cast["t_hit"].numpy()
    geometry_ids = cast["geometry_ids"].numpy().astype(np.uint32, copy=False)
    valid = np.isfinite(distances)

    ray_values = rays.numpy()
    points = np.full((HEIGHT, WIDTH, 3), np.nan, dtype=np.float32)
    points[valid] = (
        ray_values[..., :3][valid]
        + ray_values[..., 3:][valid] * distances[valid, None]
    )
    confidence = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    confidence[valid] = 1.0 / (1.0 + 0.03 * distances[valid])

    rgb = np.empty((HEIGHT, WIDTH, 3), dtype=np.uint8)
    rgb[:] = (226, 232, 238)
    masks: dict[str, np.ndarray] = {}
    for geometry_id, label in geometry_labels.items():
        mask = geometry_ids == geometry_id
        masks[label] = mask
        if np.any(mask):
            shade = np.clip(1.04 - 0.035 * distances[mask], 0.62, 1.0)
            base_color = np.asarray(COLORS[label]) * 255.0
            rgb[mask] = np.rint(base_color[None, :] * shade[:, None]).astype(
                np.uint8
            )

    stem = f"frame_{frame_index:02d}"
    rgb_path = case_dir / f"{stem}_rgb.png"
    points_path = case_dir / f"{stem}_pts3d.npy"
    confidence_path = case_dir / f"{stem}_confidence.npy"
    valid_path = case_dir / f"{stem}_valid.npy"
    Image.fromarray(rgb, mode="RGB").save(rgb_path)
    np.save(points_path, points, allow_pickle=False)
    np.save(confidence_path, confidence, allow_pickle=False)
    np.save(valid_path, valid, allow_pickle=False)

    label_paths: dict[str, str] = {}
    bboxes: dict[str, list[float]] = {}
    for label, mask in masks.items():
        mask_path = case_dir / f"{stem}_{label.replace(' ', '_')}_mask.png"
        Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(mask_path)
        label_paths[label] = mask_path.relative_to(root).as_posix()
        if np.any(mask):
            bboxes[label] = _normalized_bbox(mask)

    camera_to_world = np.linalg.inv(world_to_camera)
    return {
        "frame_id": frame_index,
        "rgb": rgb_path.relative_to(root).as_posix(),
        "pts3d": points_path.relative_to(root).as_posix(),
        "confidence": confidence_path.relative_to(root).as_posix(),
        "valid_mask": valid_path.relative_to(root).as_posix(),
        "label_masks": label_paths,
        "bboxes": bboxes,
        "K": K.tolist(),
        "camera_to_world": camera_to_world.tolist(),
    }


def generate_eval_pack(output_root: str | Path):
    """Write four deterministic CPU-raycast scenes and return their manifest."""

    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    case_specs = {
        "ladder_050": ("FAIL", 0.5, "step ladder", _normal_cameras()),
        "ladder_070": ("PASS", 0.7, "step ladder", _normal_cameras()),
        "platform_inside": (
            "FAIL",
            0.0,
            "portable work platform",
            _normal_cameras(),
        ),
        "fence_occluded": (
            "INSUFFICIENT_EVIDENCE",
            0.5,
            "step ladder",
            _occluded_cameras(),
        ),
    }
    manifest: dict[str, object] = {
        "image_size": {"width": WIDTH, "height": HEIGHT},
        "coordinate_system": "world metres; z up; camera +z forward",
        "renderer": "Open3D RaycastingScene (CPU)",
        "cases": {},
    }

    cases = manifest["cases"]
    assert isinstance(cases, dict)
    for case_id, (status, distance_m, movable_label, cameras) in case_specs.items():
        case_dir = root / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        meshes = _case_meshes(movable_label, distance_m)

        combined = o3d.geometry.TriangleMesh()
        scene = o3d.t.geometry.RaycastingScene()
        geometry_labels: dict[int, str] = {}
        for label, mesh in meshes.items():
            combined += mesh
            geometry_id = int(
                scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
            )
            geometry_labels[geometry_id] = label

        scene_path = case_dir / "scene.ply"
        if not o3d.io.write_triangle_mesh(
            str(scene_path), combined, write_vertex_colors=True
        ):
            raise RuntimeError(f"Could not write scene mesh: {scene_path}")

        frames = [
            _write_frame(
                root,
                case_dir,
                index,
                scene,
                geometry_labels,
                eye,
                target,
            )
            for index, (eye, target) in enumerate(cameras)
        ]
        cases[case_id] = {
            "expected_status": status,
            "expected_distance_m": distance_m,
            "movable_label": movable_label,
            "scene_ply": scene_path.relative_to(root).as_posix(),
            "frames": frames,
        }

    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest
