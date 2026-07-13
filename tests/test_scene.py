from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ehs_spatial.contracts import Criterion, GeometryFrame, Observation2D


RAW_TO_METERS = 0.8


def _raw_rotation(tilted: bool) -> np.ndarray:
    if not tilted:
        return np.eye(3)
    angle = np.deg2rad(12.0)
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(angle), -np.sin(angle)],
            [0.0, np.sin(angle), np.cos(angle)],
        ]
    )


def _raw(points_m: np.ndarray, *, tilted: bool = False) -> np.ndarray:
    unrotated = np.asarray(points_m, dtype=np.float64) / RAW_TO_METERS
    return unrotated @ _raw_rotation(tilted).T


def _fence_points() -> np.ndarray:
    edge = np.linspace(0.0, 2.0, 26)
    inset = edge[1:-1]
    boundary = np.vstack(
        [
            np.column_stack([edge, np.zeros_like(edge)]),
            np.column_stack([edge, np.full_like(edge, 2.0)]),
            np.column_stack([np.zeros_like(inset), inset]),
            np.column_stack([np.full_like(inset, 2.0), inset]),
        ]
    )
    heights = np.linspace(0.08, 1.0, 9)
    return np.vstack(
        [
            np.column_stack([boundary, np.full(len(boundary), height)])
            for height in heights
        ]
    )


def _object_points(clearance_m: float, *, inside: bool = False) -> np.ndarray:
    x = (
        np.linspace(0.8, 1.0, 5)
        if inside
        else np.linspace(2.0 + clearance_m, 2.2 + clearance_m, 5)
    )
    y = np.linspace(0.8, 1.05, 6)
    z = np.linspace(0.08, 0.5, 5)
    return np.asarray(np.meshgrid(x, y, z, indexing="ij")).reshape(3, -1).T


def _write_mask(path: Path, shape: tuple[int, int], indices: np.ndarray) -> str:
    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    mask[indices] = 255
    Image.fromarray(mask.reshape(shape), mode="L").save(path)
    return str(path)


def _synthetic_scene(
    tmp_path: Path,
    clearance_m: float,
    *,
    floor_views: int = 4,
    fence_views: int = 3,
    object_views: int = 2,
    inside: bool = False,
    tilted: bool = False,
):
    shape = (40, 40)
    floor_xy = np.asarray(
        np.meshgrid(np.linspace(-1.0, 4.0, 20), np.linspace(-1.0, 3.0, 20))
    ).reshape(2, -1).T
    floor = np.column_stack([floor_xy, np.zeros(len(floor_xy))])
    fence = _fence_points()
    movable = _object_points(clearance_m, inside=inside)
    all_points = _raw(np.vstack([floor, fence, movable]), tilted=tilted)

    floor_indices = np.arange(len(floor))
    fence_indices = np.arange(len(floor), len(floor) + len(fence))
    object_indices = np.arange(len(floor) + len(fence), len(all_points))
    object_with_floor_leakage = np.concatenate([floor_indices[:30], object_indices])

    frames = []
    observations = []
    for frame_index in range(4):
        frame_id = f"frame-{frame_index + 1}"
        points = np.full((*shape, 3), np.nan, dtype=np.float64)
        points.reshape(-1, 3)[: len(all_points)] = all_points
        valid = np.zeros(shape, dtype=bool)
        valid.reshape(-1)[: len(all_points)] = True
        confidence = valid.astype(np.float32)

        image_path = tmp_path / f"{frame_id}.png"
        points_path = tmp_path / f"{frame_id}-pts3d.npy"
        valid_path = tmp_path / f"{frame_id}-valid.npy"
        confidence_path = tmp_path / f"{frame_id}-conf.npy"
        Image.new("RGB", (shape[1], shape[0]), "black").save(image_path)
        np.save(points_path, points)
        np.save(valid_path, valid)
        np.save(confidence_path, confidence)

        camera_to_world = np.eye(4)
        rotation = _raw_rotation(tilted)
        camera_to_world[:3, :3] = rotation
        camera_to_world[:3, 3] = rotation @ np.array(
            [0.1 * frame_index, 0.05 * frame_index, 2.0]
        )
        frames.append(
            GeometryFrame(
                frame_id=frame_id,
                canonical_image_path=str(image_path),
                pts3d_path=str(points_path),
                conf_path=str(confidence_path),
                valid_mask_path=str(valid_path),
                camera_to_world=camera_to_world.tolist(),
                intrinsics=[[100, 0, 20], [0, 100, 20], [0, 0, 1]],
            )
        )

        if frame_index < floor_views:
            floor_mask = _write_mask(
                tmp_path / f"{frame_id}-floor.png", shape, floor_indices
            )
            observations.append(
                Observation2D(
                    observation_id=f"floor-{frame_id}",
                    frame_id=frame_id,
                    label="factory floor",
                    instance_id="floor",
                    mask_path=floor_mask,
                    score=0.99,
                    bbox=(0, 0, 1, 1),
                    source_prompt="factory floor",
                )
            )

        if frame_index < fence_views:
            fence_mask = _write_mask(
                tmp_path / f"{frame_id}-fence.png", shape, fence_indices
            )
            observations.append(
                Observation2D(
                    observation_id=f"fence-{frame_id}",
                    frame_id=frame_id,
                    label="safety fence",
                    instance_id="fence",
                    mask_path=fence_mask,
                    score=0.98,
                    bbox=(0, 0, 1, 1),
                    source_prompt="safety fence",
                )
            )

        if frame_index < object_views:
            object_mask = _write_mask(
                tmp_path / f"{frame_id}-object.png",
                shape,
                object_with_floor_leakage,
            )
            observations.append(
                Observation2D(
                    observation_id=f"pallet-{frame_id}",
                    frame_id=frame_id,
                    label="pallet",
                    instance_id="pallet",
                    mask_path=object_mask,
                    score=0.97,
                    bbox=(0, 0, 1, 1),
                    source_prompt="pallet",
                )
            )

    return frames, observations


def test_external_clearance_below_minimum_fails(tmp_path):
    from ehs_spatial.scene import build_scene_and_assess

    frames, observations = _synthetic_scene(tmp_path, clearance_m=0.5)

    scene, assessment = build_scene_and_assess(
        run_id="clearance-050",
        frames=frames,
        observations=observations,
        camera_height_m=1.6,
        criterion=Criterion(),
    )

    assert assessment.status.value == "FAIL"
    assert assessment.approximate_distance_m == pytest.approx(0.5, abs=0.03)
    assert scene.scale_source == "camera_height"
    assert scene.scale_factor == pytest.approx(0.8, abs=0.01)


def test_external_clearance_above_minimum_passes(tmp_path):
    from ehs_spatial.scene import build_scene_and_assess

    frames, observations = _synthetic_scene(tmp_path, clearance_m=0.7)

    _, assessment = build_scene_and_assess(
        run_id="clearance-070",
        frames=frames,
        observations=observations,
        camera_height_m=1.6,
        criterion=Criterion(),
    )

    assert assessment.status.value == "PASS"
    assert assessment.approximate_distance_m == pytest.approx(0.7, abs=0.03)


def test_external_clearance_equal_to_minimum_passes(tmp_path):
    from ehs_spatial.scene import build_scene_and_assess

    frames, observations = _synthetic_scene(tmp_path, clearance_m=0.6)

    _, assessment = build_scene_and_assess(
        run_id="clearance-060",
        frames=frames,
        observations=observations,
        camera_height_m=1.6,
        criterion=Criterion(),
    )

    assert assessment.status.value == "PASS"
    assert assessment.approximate_distance_m == pytest.approx(0.6, abs=1e-12)


def test_object_inside_fence_fails_regardless_of_boundary_distance(tmp_path):
    from ehs_spatial.scene import build_scene_and_assess

    frames, observations = _synthetic_scene(
        tmp_path, clearance_m=0.0, inside=True
    )

    scene, assessment = build_scene_and_assess(
        run_id="inside-fence",
        frames=frames,
        observations=observations,
        camera_height_m=1.6,
        criterion=Criterion(),
    )

    inside_fact = next(
        fact for fact in scene.facts if fact.predicate == "inside_or_intersects"
    )
    assert assessment.status.value == "FAIL"
    assert assessment.approximate_distance_m > 0.6
    assert inside_fact.value == 1.0
    assert inside_fact.unit == "boolean"


def test_fence_with_fewer_than_three_views_is_insufficient(tmp_path):
    from ehs_spatial.scene import build_scene_and_assess

    frames, observations = _synthetic_scene(
        tmp_path, clearance_m=0.5, fence_views=2
    )

    scene, assessment = build_scene_and_assess(
        run_id="two-view-fence",
        frames=frames,
        observations=observations,
        camera_height_m=1.6,
        criterion=Criterion(),
    )

    assert assessment.status.value == "INSUFFICIENT_EVIDENCE"
    assert assessment.approximate_distance_m is None
    assert scene.facts == []
    assert any("safety fence" in warning for warning in scene.warnings)


def test_movable_seen_in_only_one_view_is_insufficient(tmp_path):
    from ehs_spatial.scene import build_scene_and_assess

    frames, observations = _synthetic_scene(
        tmp_path, clearance_m=0.5, object_views=1
    )

    scene, assessment = build_scene_and_assess(
        run_id="one-view-pallet",
        frames=frames,
        observations=observations,
        camera_height_m=1.6,
        criterion=Criterion(),
    )

    assert assessment.status.value == "INSUFFICIENT_EVIDENCE"
    assert assessment.approximate_distance_m is None
    assert scene.facts == []
    assert any("movable entity" in warning for warning in scene.warnings)


def test_same_object_in_two_views_reconciles_to_one_entity(tmp_path):
    from ehs_spatial.scene import build_scene_and_assess

    frames, observations = _synthetic_scene(tmp_path, clearance_m=0.5)

    scene, _ = build_scene_and_assess(
        run_id="reconciled-pallet",
        frames=frames,
        observations=observations,
        camera_height_m=1.6,
        criterion=Criterion(),
    )

    pallets = [entity for entity in scene.entities if entity.label == "pallet"]
    assert len(pallets) == 1
    assert pallets[0].observation_ids == ["pallet-frame-1", "pallet-frame-2"]
    assert pallets[0].evidence_frame_ids == ["frame-1", "frame-2"]


def test_camera_height_scale_normalizes_tilted_floor_and_removes_floor_leakage(
    tmp_path,
):
    from ehs_spatial.geometry import _build_geometry

    frames, observations = _synthetic_scene(
        tmp_path, clearance_m=0.5, tilted=True
    )

    geometry = _build_geometry(frames, observations, camera_height_m=1.6)

    assert geometry.transform is not None
    assert geometry.transform.scale_factor == pytest.approx(0.8, abs=0.01)
    floor_points = np.load(frames[0].pts3d_path)[:10, :].reshape(-1, 3)
    normalized_floor = geometry.transform.apply(floor_points)
    assert np.max(np.abs(normalized_floor[:, 2])) < 1e-8
    pallet = next(entity for entity in geometry.entities if entity.label == "pallet")
    pallet_x = [point[0] for point in pallet.footprint_xy]
    assert min(pallet_x) == pytest.approx(2.5, abs=0.03)
    assert max(pallet_x) == pytest.approx(2.7, abs=0.03)


def test_floor_seen_in_only_one_frame_is_insufficient_without_fake_plane(tmp_path):
    from ehs_spatial.scene import build_scene_and_assess

    frames, observations = _synthetic_scene(
        tmp_path, clearance_m=0.5, floor_views=1
    )

    scene, assessment = build_scene_and_assess(
        run_id="one-view-floor",
        frames=frames,
        observations=observations,
        camera_height_m=1.6,
        criterion=Criterion(),
    )

    assert assessment.status.value == "INSUFFICIENT_EVIDENCE"
    assert assessment.approximate_distance_m is None
    assert scene.floor_plane is None
    assert scene.scale_factor is None
    assert any("2 distinct frames" in warning for warning in scene.warnings)


def test_topdown_png_is_created_and_readable(tmp_path):
    from ehs_spatial.scene import build_scene_and_assess

    frames, observations = _synthetic_scene(tmp_path, clearance_m=0.7)
    topdown_path = tmp_path / "evidence" / "topdown.png"

    build_scene_and_assess(
        run_id="topdown",
        frames=frames,
        observations=observations,
        camera_height_m=1.6,
        criterion=Criterion(),
        topdown_path=topdown_path,
    )

    assert topdown_path.stat().st_size > 100
    with Image.open(topdown_path) as image:
        assert image.format == "PNG"
        assert image.size == (640, 640)
        image.verify()


def test_pre_floor_insufficiency_still_writes_text_topdown(tmp_path):
    from ehs_spatial.scene import build_scene_and_assess

    frames, observations = _synthetic_scene(
        tmp_path, clearance_m=0.5, floor_views=1
    )
    topdown_path = tmp_path / "insufficient.png"

    _, assessment = build_scene_and_assess(
        run_id="insufficient-topdown",
        frames=frames,
        observations=observations,
        camera_height_m=1.6,
        criterion=Criterion(),
        topdown_path=topdown_path,
    )

    assert assessment.status.value == "INSUFFICIENT_EVIDENCE"
    with Image.open(topdown_path) as image:
        assert image.format == "PNG"
        assert image.getbbox() is not None


def test_mask_pointmap_shape_mismatch_fails_loudly(tmp_path):
    from ehs_spatial.scene import build_scene_and_assess

    frames, observations = _synthetic_scene(tmp_path, clearance_m=0.5)
    corrupt_mask = next(
        observation.mask_path
        for observation in observations
        if observation.label == "factory floor"
    )
    Image.new("L", (8, 8), 255).save(corrupt_mask)

    with pytest.raises(ValueError, match="mask shape does not match pts3d"):
        build_scene_and_assess(
            run_id="corrupt-mask",
            frames=frames,
            observations=observations,
            camera_height_m=1.6,
            criterion=Criterion(),
        )
