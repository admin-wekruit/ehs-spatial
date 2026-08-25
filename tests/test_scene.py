from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ehs_spatial.contracts import Criterion, GeometryFrame, Observation2D


RAW_TO_METERS = 0.8

# OpenCV camera axes expressed in the unrotated metric world (z up): camera +Y
# points down so the geometry layer's average-camera-up prior recovers world up.
CAMERA_BASIS = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
    ]
)


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


def _raw(
    points_m: np.ndarray,
    *,
    tilted: bool = False,
    raw_to_meters: float = RAW_TO_METERS,
) -> np.ndarray:
    unrotated = np.asarray(points_m, dtype=np.float64) / raw_to_meters
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
    geometry_views: int = 4,
    inside: bool = False,
    tilted: bool = False,
    raw_to_meters: float = RAW_TO_METERS,
):
    shape = (40, 40)
    floor_xy = np.asarray(
        np.meshgrid(np.linspace(-1.0, 4.0, 20), np.linspace(-1.0, 3.0, 20))
    ).reshape(2, -1).T
    floor = np.column_stack([floor_xy, np.zeros(len(floor_xy))])
    fence = _fence_points()
    movable = _object_points(clearance_m, inside=inside)
    all_points = _raw(
        np.vstack([floor, fence, movable]),
        tilted=tilted,
        raw_to_meters=raw_to_meters,
    )

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
        if frame_index < geometry_views:
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
        camera_to_world[:3, :3] = rotation @ CAMERA_BASIS
        camera_center_m = np.array(
            [0.08 * frame_index, 0.04 * frame_index, 1.6]
        )
        camera_to_world[:3, 3] = rotation @ (camera_center_m / raw_to_meters)
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


def test_external_clearance_inside_band_needs_review(tmp_path):
    from ehs_spatial.scene import build_scene_and_assess

    frames, observations = _synthetic_scene(tmp_path, clearance_m=0.5)

    scene, assessment = build_scene_and_assess(
        run_id="clearance-050",
        frames=frames,
        observations=observations,
        camera_height_m=1.6,
        criterion=Criterion(),
    )

    # 0.5 m sits inside the ±0.20 m multi-view band around 0.6 m: the
    # honest verdict is NEEDS_REVIEW, not a confident FAIL.
    assert assessment.status.value == "NEEDS_REVIEW"
    assert assessment.approximate_distance_m == pytest.approx(0.5, abs=0.03)
    assert assessment.distance_error_budget_m == pytest.approx(0.20)
    assert scene.scale_source == "camera_height"
    assert scene.scale_factor == pytest.approx(0.8, abs=0.01)


def test_external_clearance_above_minimum_passes(tmp_path):
    from ehs_spatial.scene import build_scene_and_assess

    frames, observations = _synthetic_scene(tmp_path, clearance_m=0.9)

    _, assessment = build_scene_and_assess(
        run_id="clearance-090",
        frames=frames,
        observations=observations,
        camera_height_m=1.6,
        criterion=Criterion(),
    )

    assert assessment.status.value == "PASS"
    assert assessment.approximate_distance_m == pytest.approx(0.9, abs=0.03)


def test_external_clearance_equal_to_minimum_needs_review(tmp_path):
    from ehs_spatial.scene import build_scene_and_assess

    frames, observations = _synthetic_scene(tmp_path, clearance_m=0.6)

    _, assessment = build_scene_and_assess(
        run_id="clearance-060",
        frames=frames,
        observations=observations,
        camera_height_m=1.6,
        criterion=Criterion(),
    )

    # Exactly at the threshold is the centre of the error band; under band
    # semantics this is the canonical NEEDS_REVIEW case.
    assert assessment.status.value == "NEEDS_REVIEW"
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
    assert assessment.approximate_distance_m == 0.0
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


def test_one_cluster_point_does_not_count_as_a_second_evidence_view(tmp_path):
    from ehs_spatial.scene import build_scene_and_assess

    frames, observations = _synthetic_scene(tmp_path, clearance_m=0.5)
    second_mask_path = next(
        observation.mask_path
        for observation in observations
        if observation.observation_id == "pallet-frame-2"
    )
    second_mask = np.asarray(Image.open(second_mask_path)).astype(bool)
    second_points = np.load(frames[1].pts3d_path)
    object_indices = np.flatnonzero(
        second_mask.reshape(-1) & (second_points.reshape(-1, 3)[:, 2] > 0)
    )
    noise_m = np.column_stack(
        [
            10.0 + 0.3 * np.arange(len(object_indices) - 1),
            np.full(len(object_indices) - 1, 10.0),
            np.full(len(object_indices) - 1, 0.2),
        ]
    )
    second_points.reshape(-1, 3)[object_indices[1:]] = _raw(noise_m)
    np.save(frames[1].pts3d_path, second_points)

    scene, assessment = build_scene_and_assess(
        run_id="one-stray-cluster-point",
        frames=frames,
        observations=observations,
        camera_height_m=1.6,
        criterion=Criterion(),
    )

    pallet = next(entity for entity in scene.entities if entity.label == "pallet")
    assert assessment.status.value == "INSUFFICIENT_EVIDENCE"
    assert assessment.approximate_distance_m is None
    assert scene.facts == []
    assert pallet.observation_ids == ["pallet-frame-1"]
    assert pallet.evidence_frame_ids == ["frame-1"]


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


def test_floor_ransac_final_tolerance_is_three_centimeters_at_any_raw_scale(
    tmp_path,
):
    from ehs_spatial.scene import build_scene_and_assess

    for raw_to_meters in (0.08, 8.0):
        scene_dir = tmp_path / str(raw_to_meters)
        scene_dir.mkdir()
        frames, observations = _synthetic_scene(
            scene_dir,
            clearance_m=0.5,
            raw_to_meters=raw_to_meters,
        )
        for frame in frames:
            points = np.load(frame.pts3d_path)
            points.reshape(-1, 3)[:400:4, 2] += 0.08 / raw_to_meters
            np.save(frame.pts3d_path, points)

        scene, assessment = build_scene_and_assess(
            run_id=f"raw-scale-{raw_to_meters}",
            frames=frames,
            observations=observations,
            camera_height_m=1.6,
            criterion=Criterion(),
        )

        assert scene.scale_factor == pytest.approx(raw_to_meters, rel=0.002)
        assert assessment.status.value == "NEEDS_REVIEW"
        assert assessment.approximate_distance_m == pytest.approx(0.5, abs=0.01)


def test_floor_fit_requests_compact_svd_for_tall_point_matrix(
    tmp_path,
    monkeypatch,
):
    from ehs_spatial.scene import build_scene_and_assess

    frames, observations = _synthetic_scene(tmp_path, clearance_m=0.5)
    original_svd = np.linalg.svd
    seen_shapes = []

    def require_compact_svd(matrix, *args, **kwargs):
        seen_shapes.append(matrix.shape)
        assert kwargs.get("full_matrices") is False
        return original_svd(matrix, *args, **kwargs)

    monkeypatch.setattr(np.linalg, "svd", require_compact_svd)

    build_scene_and_assess(
        run_id="compact-floor-svd",
        frames=frames,
        observations=observations,
        camera_height_m=1.6,
        criterion=Criterion(),
    )

    assert seen_shapes
    assert all(rows > columns for rows, columns in seen_shapes)


def test_top_surface_only_object_height_is_measured_above_floor(tmp_path):
    from ehs_spatial.scene import build_scene_and_assess

    frames, observations = _synthetic_scene(tmp_path, clearance_m=0.5)
    for observation in observations:
        if observation.label != "pallet":
            continue
        frame = next(frame for frame in frames if frame.frame_id == observation.frame_id)
        points = np.load(frame.pts3d_path)
        mask = np.asarray(Image.open(observation.mask_path)).astype(bool)
        object_points = mask & (points[:, :, 2] > 0)
        points[:, :, 2][object_points] = 0.15 / RAW_TO_METERS
        np.save(frame.pts3d_path, points)

    scene, assessment = build_scene_and_assess(
        run_id="top-surface-height",
        frames=frames,
        observations=observations,
        camera_height_m=1.6,
        criterion=Criterion(),
    )

    pallet = next(entity for entity in scene.entities if entity.label == "pallet")
    assert pallet.height_m == pytest.approx(0.15, abs=0.01)
    assert assessment.status.value == "NEEDS_REVIEW"


def test_geometry_in_only_one_frame_is_insufficient_without_fake_plane(tmp_path):
    from ehs_spatial.scene import build_scene_and_assess

    frames, observations = _synthetic_scene(
        tmp_path, clearance_m=0.5, geometry_views=1
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
    assert any("2 distinct frame(s)" in warning for warning in scene.warnings)


@pytest.mark.parametrize(
    "invalid_raw_height",
    [pytest.param(np.nan, id="nan"), pytest.param(-1.0, id="below-plane")],
)
def test_invalid_camera_plane_height_is_insufficient(tmp_path, invalid_raw_height):
    from ehs_spatial.scene import build_scene_and_assess

    frames, observations = _synthetic_scene(tmp_path, clearance_m=0.5)
    pose = np.asarray(frames[0].camera_to_world).copy()
    pose[2, 3] = invalid_raw_height
    frames[0] = GeometryFrame.model_validate(
        {**frames[0].model_dump(), "camera_to_world": pose.tolist()}
    )

    scene, assessment = build_scene_and_assess(
        run_id="invalid-camera-height",
        frames=frames,
        observations=observations,
        camera_height_m=1.6,
        criterion=Criterion(),
    )

    assert assessment.status.value == "INSUFFICIENT_EVIDENCE"
    assert assessment.approximate_distance_m is None
    assert scene.scale_factor is None
    assert scene.facts == []
    assert any("camera-to-floor heights" in warning for warning in scene.warnings)


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
        tmp_path, clearance_m=0.5, geometry_views=1
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
        if observation.label == "safety fence"
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


def test_floor_ransac_prefers_lowest_supported_plane_over_dominant_tabletop():
    from ehs_spatial.geometry import _ransac_floor_plane

    rng = np.random.default_rng(7)
    tabletop = np.column_stack(
        [rng.uniform(0, 2, 4000), rng.uniform(0, 2, 4000), np.full(4000, 0.7)]
    )
    floor = np.column_stack(
        [rng.uniform(0, 2, 500), rng.uniform(0, 2, 500), np.zeros(500)]
    )
    points = np.vstack([tabletop, floor])
    cameras = np.array([[0.5, 0.5, 1.6], [1.5, 1.5, 1.7]])

    inliers = _ransac_floor_plane(points, cameras, np.array([0.0, 0.0, 1.0]), 0.03)

    assert inliers is not None
    assert float(np.mean(points[inliers][:, 2])) == pytest.approx(0.0, abs=0.02)


def test_topdown_always_draws_the_selected_entity(tmp_path):
    from ehs_spatial.contracts import Assessment, Entity3D
    from ehs_spatial.topdown import _render_topdown
    from PIL import Image

    # A single-frame movable passes the scaled rule gates on a 1-view capture
    # but fails the renderer's cosmetic 2-frame filter; selection must win.
    movable = Entity3D(
        entity_id="entity-ladder-01",
        label="step ladder",
        observation_ids=["obs-1"],
        centroid_xyz=(3.0, 1.0, 0.5),
        footprint_xy=[(2.8, 0.8), (3.2, 0.8), (3.2, 1.2), (2.8, 1.2)],
        height_m=1.2,
        evidence_frame_ids=["frame-1"],
    )
    path = tmp_path / "topdown.png"
    _render_topdown(
        path,
        [movable],
        [(0, 0), (2, 0), (2, 2), (0, 2)],
        "entity-ladder-01",
        Assessment(
            status="FAIL",
            fact_ids=[],
            evidence_frame_ids=["frame-1"],
            approximate_distance_m=0.8,
        ),
    )

    pixels = np.array(Image.open(path).convert("RGB")).reshape(-1, 3)
    assert (pixels == (255, 138, 101)).all(axis=1).any()  # selected fill #ff8a65


def test_scene_builder_emits_plan_view_and_semantic_ply(tmp_path):
    from ehs_spatial.scene import build_scene_and_assess

    frames, observations = _synthetic_scene(tmp_path, clearance_m=0.5)
    plan_path = tmp_path / "plan_view.png"
    ply_path = tmp_path / "semantic_cloud.ply"

    build_scene_and_assess(
        run_id="plan-artifacts",
        frames=frames,
        observations=observations,
        camera_height_m=1.6,
        criterion=Criterion(),
        plan_view_path=plan_path,
        semantic_ply_path=ply_path,
    )

    with Image.open(plan_path) as plan:
        assert plan.size == (760, 760)
    raw = ply_path.read_bytes()
    assert raw.startswith(b"ply\nformat binary_little_endian 1.0\n")
    header_end = raw.index(b"end_header\n") + len(b"end_header\n")
    count = int(
        [line for line in raw[:header_end].split(b"\n") if b"element vertex" in line][0]
        .split()[-1]
    )
    assert count > 0
    assert len(raw) - header_end == count * (12 + 3)


def test_scene_builder_renders_cloud_views_fail_soft(tmp_path):
    from ehs_spatial.scene import build_scene_and_assess

    frames, observations = _synthetic_scene(tmp_path, clearance_m=0.5)
    persp = tmp_path / "cloud_perspective.png"
    top = tmp_path / "cloud_topdown.png"

    # Must never raise, even on hosts where offscreen GL is unavailable.
    build_scene_and_assess(
        run_id="cloud-views",
        frames=frames,
        observations=observations,
        camera_height_m=1.6,
        criterion=Criterion(),
        cloud_views_paths=(persp, top),
    )

    if persp.exists():  # renderer available on this host
        with Image.open(persp) as image:
            assert image.size == (1280, 860)
        assert top.exists()


def test_model_native_scale_demotes_the_verdict_to_review(tmp_path):
    """An unanchored gauge cannot honestly certify PASS or FAIL: the same
    geometry that PASSes under an anchored scale is NEEDS_REVIEW when the
    scale is model-native."""
    from ehs_spatial.scene import build_scene_and_assess

    frames, observations = _synthetic_scene(tmp_path, clearance_m=0.9)

    _, anchored = build_scene_and_assess(
        run_id="native-a",
        frames=frames,
        observations=observations,
        camera_height_m=1.6,
        criterion=Criterion(),
        scale_source="moge_anchor",
    )
    native_scene, native = build_scene_and_assess(
        run_id="native-b",
        frames=frames,
        observations=observations,
        camera_height_m=1.6,
        criterion=Criterion(),
        scale_source="model_native",
    )

    assert anchored.status.value == "PASS"
    assert native.status.value == "NEEDS_REVIEW"
    assert any("model-native" in w for w in native_scene.warnings)
