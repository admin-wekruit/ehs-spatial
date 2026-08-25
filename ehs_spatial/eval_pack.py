"""Generate deterministic calibrated EHS workcell evaluation scenes."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import open3d as o3d
from PIL import Image
from pydantic import TypeAdapter

from .artifacts import ArtifactStore
from .contracts import (
    Assessment,
    CaptureRun,
    Criterion,
    GeometryFrame,
    GroundedAnswer,
    Observation2D,
    SceneMap,
)
from .providers.base import ProviderError
from .scene import build_scene_and_assess


WIDTH = 512
HEIGHT = 384
FX = FY = 450.0
CAMERA_HEIGHT_M = 1.65
K = np.array(
    [
        [FX, 0.0, (WIDTH - 1) / 2],
        [0.0, FY, (HEIGHT - 1) / 2],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
FENCE = {"xmin": -1.5, "xmax": 1.5, "ymin": -1.2, "ymax": 1.2}
FENCE_POST_RADIUS_M = 0.035
COLORS = {
    "factory floor": (0.58, 0.61, 0.64),
    "safety fence": (0.96, 0.72, 0.05),
    "industrial robot arm": (0.95, 0.24, 0.04),
    "step ladder": (0.12, 0.36, 0.86),
    "portable work platform": (0.05, 0.58, 0.52),
}
_CASE_IDS = {
    "ladder_050",
    "ladder_070",
    "platform_inside",
    "fence_occluded",
}


def _pack_path(root: Path, value: object) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ValueError("artifact path must be a relative path inside the pack root")
    resolved_root = root.resolve()
    candidate = (resolved_root / value).resolve()
    if not candidate.is_relative_to(resolved_root):
        raise ValueError("artifact path must be a relative path inside the pack root")
    return candidate


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
        _cylinder_between((x, y, 0.0), (x, y, 1.6), FENCE_POST_RADIUS_M)
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
    # ponytail: this analytic synthetic fence is a domain simplification; real measured capture is the upgrade path.
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
    fence_outer_x = FENCE["xmax"] + FENCE_POST_RADIUS_M
    front_x = fence_outer_x + clearance_m + radius
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
        float(mesh.get_axis_aligned_bounding_box().min_bound[0]) - fence_outer_x
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
        ((4.6, -4.4, CAMERA_HEIGHT_M), target),
        ((4.6, 4.4, CAMERA_HEIGHT_M), target),
        ((-4.4, 4.2, CAMERA_HEIGHT_M), target),
        ((-4.6, -4.1, CAMERA_HEIGHT_M), target),
    ]


def _occluded_cameras():
    return _normal_cameras()[:2] + [
        ((2.8, -2.7, CAMERA_HEIGHT_M), (4.4, -3.8, -0.3)),
        ((2.6, 2.3, CAMERA_HEIGHT_M), (4.4, 3.8, -0.8)),
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
    primitive_normals = cast["primitive_normals"].numpy()
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
    light_direction = np.array([0.35, -0.45, 0.82], dtype=np.float32)
    light_direction /= np.linalg.norm(light_direction)
    diffuse = np.clip(primitive_normals @ light_direction, 0.0, 1.0)
    lighting = 0.35 + 0.65 * diffuse
    masks: dict[str, np.ndarray] = {}
    for geometry_id, label in geometry_labels.items():
        mask = geometry_ids == geometry_id
        masks[label] = mask
        if np.any(mask):
            base_color = np.asarray(COLORS[label]) * 255.0
            rgb[mask] = np.rint(
                base_color[None, :] * lighting[mask, None]
            ).astype(
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
        # 0.5 and 0.7 both sit inside the ±0.20 m multi-view error band
        # around the 0.6 m threshold, so the production verdict for BOTH is
        # NEEDS_REVIEW; the expected-distance gate still checks each case
        # lands on its own side of the threshold numerically.
        "ladder_050": ("NEEDS_REVIEW", 0.5, "step ladder", _normal_cameras()),
        "ladder_070": ("NEEDS_REVIEW", 0.7, "step ladder", _normal_cameras()),
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
        "camera_height_m": CAMERA_HEIGHT_M,
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
        round_trip = o3d.io.read_triangle_mesh(str(scene_path))
        round_trip_vertices = np.asarray(round_trip.vertices)
        round_trip_triangles = np.asarray(round_trip.triangles)
        round_trip_colors = np.asarray(round_trip.vertex_colors)
        if (
            not len(round_trip_vertices)
            or not len(round_trip_triangles)
            or round_trip_colors.shape != round_trip_vertices.shape
            or not np.isfinite(round_trip_colors).all()
        ):
            raise RuntimeError(f"Invalid scene PLY round-trip: {scene_path}")

        scene_glb_path = case_dir / "scene.glb"
        if not o3d.io.write_triangle_mesh(
            str(scene_glb_path), combined, write_vertex_colors=True
        ):
            raise RuntimeError(f"Could not write browser scene mesh: {scene_glb_path}")
        glb_payload = scene_glb_path.read_bytes()
        if (
            len(glb_payload) < 20
            or glb_payload[:4] != b"glTF"
            or int.from_bytes(glb_payload[4:8], "little") != 2
            or int.from_bytes(glb_payload[8:12], "little") != len(glb_payload)
        ):
            raise RuntimeError(
                f"Invalid browser scene GLB round-trip: {scene_glb_path}"
            )

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
            "scene_glb": scene_glb_path.relative_to(root).as_posix(),
            "frames": frames,
        }

    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def run_offline_benchmark(pack_root: str | Path) -> dict:
    """Assess a generated pack from its calibrated pointmaps and oracle masks."""

    root = Path(pack_root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    cases = manifest.get("cases")
    if not isinstance(cases, dict) or set(cases) != _CASE_IDS:
        raise ValueError("eval pack must contain exactly the four calibrated cases")
    for case in cases.values():
        for frame in case["frames"]:
            for key in ("rgb", "pts3d", "confidence", "valid_mask"):
                _pack_path(root, frame[key])
            for mask_path in frame["label_masks"].values():
                _pack_path(root, mask_path)

    camera_height_m = manifest["camera_height_m"]
    store = ArtifactStore(root)
    case_reports: dict[str, dict[str, Any]] = {}

    for case_id, case in cases.items():
        frames = []
        observations = []
        for frame in case["frames"]:
            frame_id = f"frame-{frame['frame_id']:02d}"
            frames.append(
                GeometryFrame(
                    frame_id=frame_id,
                    canonical_image_path=str(_pack_path(root, frame["rgb"])),
                    pts3d_path=str(_pack_path(root, frame["pts3d"])),
                    conf_path=str(_pack_path(root, frame["confidence"])),
                    valid_mask_path=str(_pack_path(root, frame["valid_mask"])),
                    camera_to_world=frame["camera_to_world"],
                    intrinsics=frame["K"],
                )
            )
            for label, bbox in sorted(frame["bboxes"].items()):
                observations.append(
                    Observation2D(
                        observation_id=f"{frame_id}-{label.replace(' ', '-')}",
                        frame_id=frame_id,
                        label=label,
                        instance_id=label,
                        mask_path=str(
                            _pack_path(root, frame["label_masks"][label])
                        ),
                        score=1.0,
                        bbox=bbox,
                        source_prompt=label,
                    )
                )

        paths = store.paths(case_id)
        scene, assessment = build_scene_and_assess(
            case_id,
            frames,
            observations,
            camera_height_m,
            Criterion(minimum_clearance_m=0.6),
            topdown_path=paths.topdown_png,
        )
        store.save_json(paths.scene_json, scene)
        store.save_json(paths.assessment_json, assessment)

        expected_status = case["expected_status"]
        expected_distance_m = case["expected_distance_m"]
        actual_distance_m = assessment.approximate_distance_m
        absolute_error_m = (
            None
            if actual_distance_m is None
            else abs(actual_distance_m - expected_distance_m)
        )
        sufficient = expected_status != "INSUFFICIENT_EVIDENCE"
        movables = [
            entity
            for entity in scene.entities
            if entity.label == case["movable_label"]
        ]
        robots = [
            entity
            for entity in scene.entities
            if entity.label == "industrial robot arm"
        ]
        fences = [
            entity for entity in scene.entities if entity.label == "safety fence"
        ]
        clearance_fact = next(
            (
                fact
                for fact in scene.facts
                if fact.predicate == "minimum_boundary_clearance"
            ),
            None,
        )
        evidence_passed = not sufficient or (
            len(movables) == 1
            and len(set(movables[0].evidence_frame_ids)) >= 2
            and bool(robots)
            and len(fences) == 1
            and len(set(fences[0].evidence_frame_ids)) >= 3
            and clearance_fact is not None
            and clearance_fact.subject_id == movables[0].entity_id
        )
        distance_passed = (
            actual_distance_m is None
            if not sufficient
            else absolute_error_m is not None and absolute_error_m <= 0.1
        )
        case_reports[case_id] = {
            "expected_status": expected_status,
            "actual_status": assessment.status.value,
            "expected_distance_m": expected_distance_m,
            "actual_distance_m": actual_distance_m,
            "absolute_error_m": absolute_error_m,
            "entity_labels": [entity.label for entity in scene.entities],
            "warnings": scene.warnings,
            "passed": assessment.status.value == expected_status
            and distance_passed
            and evidence_passed,
            "artifacts": {
                "scene_json": paths.scene_json.relative_to(root).as_posix(),
                "assessment_json": paths.assessment_json.relative_to(root).as_posix(),
                "topdown_png": paths.topdown_png.relative_to(root).as_posix(),
            },
        }

    report = {
        "passed": all(case["passed"] for case in case_reports.values()),
        "cases": case_reports,
    }
    (root / "offline_report.json").write_bytes(
        TypeAdapter(dict[str, Any]).dump_json(report, indent=2) + b"\n"
    )
    return report


def _redact_provider_values(message: str) -> str:
    for name in ("REPLICATE_API_TOKEN", "FAL_KEY", "GEMINI_API_KEY"):
        value = os.getenv(name)
        if value:
            message = message.replace(value, "[REDACTED]")
    return message


def _write_live_report(root: Path, report: dict[str, Any]) -> dict[str, Any]:
    (root / "live_report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def run_live_benchmark_case(
    pack_root: str | Path,
    case_id: str,
    *,
    pipeline: Any | None = None,
) -> dict[str, Any]:
    """Run one provider-backed case and write ``live_report.json``."""

    root = Path(pack_root).resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    cases = manifest.get("cases")
    if not isinstance(cases, dict) or set(cases) != _CASE_IDS:
        raise ValueError("eval pack must contain exactly the four calibrated cases")
    if case_id not in cases:
        raise ValueError(f"unknown calibrated case: {case_id}")
    case = cases[case_id]
    frames = case.get("frames")
    if not isinstance(frames, list) or len(frames) != 4:
        raise ValueError("live benchmark case must contain exactly four frames")
    image_paths = [_pack_path(root, frame.get("rgb")) for frame in frames]
    scene_ply = _pack_path(root, case.get("scene_ply"))
    scene_glb = _pack_path(root, case.get("scene_glb"))
    for path in [*image_paths, scene_ply, scene_glb]:
        if not path.is_file():
            raise FileNotFoundError(path)

    if pipeline is None:
        from .pipeline import EHSAssessmentPipeline

        pipeline = EHSAssessmentPipeline(store=ArtifactStore(root / "live_runs"))
    run_id = f"live-{case_id}-{uuid4().hex}"
    capture = CaptureRun(
        run_id=run_id,
        image_paths=[str(path) for path in image_paths],
        camera_height_m=manifest["camera_height_m"],
        criterion=Criterion(minimum_clearance_m=0.6),
    )

    # Drop any stale report first: only ProviderError is caught below, so an
    # unwrapped post-spend crash must not leave a previous invocation's verdict
    # as the pack's current live_report.json.
    (root / "live_report.json").unlink(missing_ok=True)

    try:
        assessment = Assessment.model_validate(pipeline.run_assessment(capture))
        paths = pipeline.store.paths(run_id)
        scene = pipeline.store.load_json(paths.scene_json, SceneMap)
        if scene.run_id != run_id:
            raise ValueError("provider SceneMap run_id does not match the capture")

        expected_status = case["expected_status"]
        expected_distance_m = float(case["expected_distance_m"])
        actual_distance_m = assessment.approximate_distance_m
        threshold_m = capture.criterion.minimum_clearance_m
        threshold_side_matches = (
            None
            if actual_distance_m is None
            else (actual_distance_m >= threshold_m)
            == (expected_distance_m >= threshold_m)
        )
        absolute_error_m = (
            None
            if actual_distance_m is None
            else abs(actual_distance_m - expected_distance_m)
        )

        movables = [
            entity
            for entity in scene.entities
            if entity.label == case["movable_label"]
        ]
        fences = [
            entity for entity in scene.entities if entity.label == "safety fence"
        ]
        robots = [
            entity
            for entity in scene.entities
            if entity.label == "industrial robot arm"
        ]
        clearance_fact = next(
            (
                fact
                for fact in scene.facts
                if fact.predicate == "minimum_boundary_clearance"
            ),
            None,
        )
        sufficient = expected_status != "INSUFFICIENT_EVIDENCE"
        entity_evidence_passed = not sufficient or (
            len(movables) == 1
            and len(set(movables[0].evidence_frame_ids)) >= 2
            and len(fences) == 1
            and len(set(fences[0].evidence_frame_ids)) >= 3
            and bool(robots)
            and clearance_fact is not None
            and clearance_fact.subject_id == movables[0].entity_id
        )
        scene_fact_ids = {fact.fact_id for fact in scene.facts}
        fact_grounding_passed = (
            set(assessment.fact_ids) <= scene_fact_ids
            and (
                assessment.climb_review is None
                or set(assessment.climb_review.fact_ids) <= scene_fact_ids
            )
        )

        chat_report = None
        chat_grounded = True
        if case_id == "ladder_050":
            answer = GroundedAnswer.model_validate(
                pipeline.answer_question(
                    run_id,
                    "Why did this workcell fail the clearance check?",
                )
            )
            chat_grounded = (
                bool(answer.fact_ids)
                and set(answer.fact_ids) <= scene_fact_ids
            )
            chat_report = {
                "answer": answer.answer,
                "fact_ids": answer.fact_ids,
                "grounded": chat_grounded,
            }

        artifact_paths = {
            "point_cloud_glb": paths.point_cloud_glb,
            "observations_json": paths.observations_json,
            "scene_json": paths.scene_json,
            "assessment_json": paths.assessment_json,
            "topdown_png": paths.topdown_png,
            "chat_jsonl": paths.chat_jsonl,
        }
        provider_artifacts: dict[str, str] = {}
        artifact_passed = True
        for name, path in artifact_paths.items():
            resolved = path.resolve()
            if not resolved.is_relative_to(root):
                raise ValueError("provider artifact escaped the eval pack root")
            provider_artifacts[name] = resolved.relative_to(root).as_posix()
            artifact_passed &= resolved.is_file() and resolved.stat().st_size > 0

        observation_payload = json.loads(
            paths.observations_json.read_text(encoding="utf-8")
        )
        observations = TypeAdapter(list[Observation2D]).validate_python(
            observation_payload
        )
        observations_by_label: dict[str, int] = {}
        for observation in observations:
            observations_by_label[observation.label] = (
                observations_by_label.get(observation.label, 0) + 1
            )

        report = {
            "passed": (
                assessment.status.value == expected_status
                and (not sufficient or threshold_side_matches is True)
                and entity_evidence_passed
                and fact_grounding_passed
                and chat_grounded
                and artifact_passed
            ),
            "case_id": case_id,
            "run_id": run_id,
            "expected_status": expected_status,
            "actual_status": assessment.status.value,
            "criterion_m": threshold_m,
            "expected_distance_m": expected_distance_m,
            "actual_distance_m": actual_distance_m,
            "absolute_error_m": absolute_error_m,
            "threshold_side_matches": threshold_side_matches,
            "observations": {
                "count": len(observations),
                "by_label": dict(sorted(observations_by_label.items())),
            },
            "entity_evidence_passed": entity_evidence_passed,
            "fact_grounding_passed": fact_grounding_passed,
            "entity_labels": [entity.label for entity in scene.entities],
            "entities": [
                {
                    "entity_id": entity.entity_id,
                    "label": entity.label,
                    "observation_ids": entity.observation_ids,
                    "evidence_frame_ids": entity.evidence_frame_ids,
                }
                for entity in scene.entities
            ],
            "assessment_fact_ids": assessment.fact_ids,
            "assessment_evidence_frame_ids": assessment.evidence_frame_ids,
            "warnings": scene.warnings,
            "climb_review": (
                None
                if assessment.climb_review is None
                else assessment.climb_review.model_dump(mode="json")
            ),
            "chat": chat_report,
            "source_artifacts": {
                "scene_ply": scene_ply.relative_to(root).as_posix(),
                "scene_glb": scene_glb.relative_to(root).as_posix(),
                "images": [path.relative_to(root).as_posix() for path in image_paths],
            },
            "provider_artifacts": provider_artifacts,
        }
        return _write_live_report(root, report)
    except ProviderError as exc:
        error = {
            "type": type(exc).__name__,
            "provider": exc.provider,
            "operation": exc.operation,
            "message": _redact_provider_values(exc.original_message),
        }
        return _write_live_report(
            root,
            {
                "passed": False,
                "case_id": case_id,
                "run_id": run_id,
                "error": error,
            },
        )
