import json
from pathlib import Path

import numpy as np
import open3d as o3d
import pytest
from PIL import Image

from ehs_spatial.contracts import Assessment, SceneMap
from ehs_spatial.eval_pack import (
    COLORS,
    generate_eval_pack,
    run_offline_benchmark,
)


WIDTH = 512
HEIGHT = 384
COMMON_LABELS = {"factory floor", "safety fence", "industrial robot arm"}
CASE_EXPECTATIONS = {
    # Both ladder cases sit inside the ±0.20 m multi-view band around the
    # 0.6 m threshold: production's honest verdict is NEEDS_REVIEW, and the
    # distance expectations still pin each case to its own side.
    "ladder_050": ("NEEDS_REVIEW", 0.5, "step ladder"),
    "ladder_070": ("NEEDS_REVIEW", 0.7, "step ladder"),
    "platform_inside": ("FAIL", 0.0, "portable work platform"),
    "fence_occluded": ("INSUFFICIENT_EVIDENCE", 0.5, "step ladder"),
}


@pytest.fixture(scope="module")
def generated_pack(tmp_path_factory):
    root = tmp_path_factory.mktemp("ehs-eval-pack")
    manifest = generate_eval_pack(root)
    return root, manifest


def _artifact(root: Path, relative_path: str) -> Path:
    path = root / relative_path
    assert path.is_file()
    return path


def _mask(root: Path, frame: dict, label: str) -> np.ndarray:
    with Image.open(_artifact(root, frame["label_masks"][label])) as image:
        assert image.format == "PNG"
        return np.asarray(image.convert("L"))


def test_generate_eval_pack_writes_four_cases_with_metric_truth(generated_pack):
    root, manifest = generated_pack

    assert json.loads((root / "manifest.json").read_text(encoding="utf-8")) == manifest
    assert set(manifest["cases"]) == set(CASE_EXPECTATIONS)
    assert manifest["image_size"] == {"width": WIDTH, "height": HEIGHT}
    assert manifest["camera_height_m"] == pytest.approx(1.65)

    for case_id, (status, distance_m, movable_label) in CASE_EXPECTATIONS.items():
        case = manifest["cases"][case_id]
        assert case["expected_status"] == status
        assert case["expected_distance_m"] == pytest.approx(distance_m)
        assert case["movable_label"] == movable_label
        assert len(case["frames"]) == 4


def test_generate_eval_pack_artifacts_are_pixel_aligned_and_calibrated(
    generated_pack,
):
    root, manifest = generated_pack

    for case in manifest["cases"].values():
        expected_labels = COMMON_LABELS | {case["movable_label"]}
        camera_poses = []
        rgb_payloads = []
        for frame in case["frames"]:
            with Image.open(_artifact(root, frame["rgb"])) as image:
                assert image.size == (WIDTH, HEIGHT)
                assert image.mode == "RGB"
                rgb_payloads.append(np.asarray(image).tobytes())
            points = np.load(_artifact(root, frame["pts3d"]), allow_pickle=False)
            confidence = np.load(
                _artifact(root, frame["confidence"]), allow_pickle=False
            )
            valid = np.load(_artifact(root, frame["valid_mask"]), allow_pickle=False)

            assert points.shape == (HEIGHT, WIDTH, 3)
            assert confidence.shape == valid.shape == (HEIGHT, WIDTH)
            assert valid.dtype == np.bool_
            assert np.isfinite(points[valid]).all()
            assert np.isnan(points[~valid]).all()
            assert np.all(confidence[valid] > 0)
            assert np.all(confidence[~valid] == 0)
            assert set(frame["label_masks"]) == expected_labels

            mask_union = np.zeros((HEIGHT, WIDTH), dtype=bool)
            for label in expected_labels:
                mask = _mask(root, frame, label)
                assert mask.shape == (HEIGHT, WIDTH)
                mask_union |= mask > 0
            assert np.array_equal(mask_union, valid)

            nonempty_labels = {
                label for label in expected_labels if np.any(_mask(root, frame, label))
            }
            assert set(frame["bboxes"]) == nonempty_labels
            for bbox in frame["bboxes"].values():
                x0, y0, x1, y1 = bbox
                assert 0.0 <= x0 < x1 <= 1.0
                assert 0.0 <= y0 < y1 <= 1.0

            intrinsics = np.asarray(frame["K"], dtype=float)
            camera_to_world = np.asarray(frame["camera_to_world"], dtype=float)
            assert intrinsics.shape == (3, 3)
            assert camera_to_world.shape == (4, 4)
            assert intrinsics[0, 0] > 0 and intrinsics[1, 1] > 0
            assert intrinsics[0, 2] == pytest.approx((WIDTH - 1) / 2)
            assert intrinsics[1, 2] == pytest.approx((HEIGHT - 1) / 2)
            assert np.allclose(camera_to_world[3], [0, 0, 0, 1])
            assert camera_to_world[2, 3] == pytest.approx(
                manifest["camera_height_m"]
            )
            camera_poses.append(camera_to_world)

        assert len({pose.tobytes() for pose in camera_poses}) == 4
        assert len(set(rgb_payloads)) == 4


def test_generate_eval_pack_normal_views_show_every_component(generated_pack):
    root, manifest = generated_pack

    for case_id in ("ladder_050", "ladder_070", "platform_inside"):
        case = manifest["cases"][case_id]
        labels = COMMON_LABELS | {case["movable_label"]}
        for frame in case["frames"]:
            assert all(np.count_nonzero(_mask(root, frame, label)) >= 20 for label in labels)


def test_generate_eval_pack_occlusion_comes_from_cameras_pointing_away(
    generated_pack,
):
    root, manifest = generated_pack
    frames = manifest["cases"]["fence_occluded"]["frames"]

    usable_fence_frames = [
        frame
        for frame in frames
        if np.count_nonzero(_mask(root, frame, "safety fence")) >= 20
    ]
    assert usable_fence_frames == frames[:2]

    workcell_center = np.array([0.0, 0.0, 0.8])
    for index, frame in enumerate(frames):
        camera_to_world = np.asarray(frame["camera_to_world"], dtype=float)
        camera_position = camera_to_world[:3, 3]
        camera_forward = camera_to_world[:3, 2]
        direction_to_cell = workcell_center - camera_position
        points_toward_cell = float(np.dot(camera_forward, direction_to_cell)) > 0
        assert points_toward_cell is (index < 2)


def test_generate_eval_pack_scene_meshes_round_trip_for_eval_and_viewer(
    generated_pack,
):
    root, manifest = generated_pack

    for case in manifest["cases"].values():
        scene = o3d.io.read_triangle_mesh(
            str(_artifact(root, case["scene_ply"]))
        )
        vertices = np.asarray(scene.vertices)
        triangles = np.asarray(scene.triangles)
        colors = np.asarray(scene.vertex_colors)
        assert len(vertices) > 0
        assert len(triangles) > 0
        assert colors.shape == vertices.shape
        assert np.isfinite(colors).all()

        assert _artifact(root, case["scene_glb"]).read_bytes()[:4] == b"glTF"


def test_generate_eval_pack_rejects_empty_scene_ply_round_trip(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        o3d.io,
        "read_triangle_mesh",
        lambda _path: o3d.geometry.TriangleMesh(),
    )

    with pytest.raises(RuntimeError, match="scene PLY round-trip"):
        generate_eval_pack(tmp_path)


@pytest.mark.parametrize(
    ("case_id", "expected_clearance_m"),
    [
        ("ladder_050", 0.5),
        ("ladder_070", 0.7),
        ("fence_occluded", 0.5),
    ],
)
def test_scene_ply_encodes_metric_ladder_clearance(
    generated_pack,
    case_id,
    expected_clearance_m,
):
    root, manifest = generated_pack
    scene = o3d.io.read_triangle_mesh(
        str(_artifact(root, manifest["cases"][case_id]["scene_ply"]))
    )
    vertices = np.asarray(scene.vertices)
    colors = np.asarray(scene.vertex_colors)

    def vertices_for(label):
        stored_color = np.rint(np.asarray(COLORS[label]) * 255.0) / 255.0
        selected = np.all(np.isclose(colors, stored_color, atol=1e-6), axis=1)
        assert np.any(selected)
        return vertices[selected]

    fence_vertices = vertices_for("safety fence")
    ladder_vertices = vertices_for("step ladder")
    measured_clearance_m = float(
        ladder_vertices[:, 0].min() - fence_vertices[:, 0].max()
    )
    assert measured_clearance_m == pytest.approx(expected_clearance_m, abs=1e-6)


def test_pts3d_reprojects_to_source_pixel_centers(generated_pack):
    root, manifest = generated_pack

    for case in manifest["cases"].values():
        for frame in case["frames"]:
            points = np.load(_artifact(root, frame["pts3d"]), allow_pickle=False)
            valid = np.load(_artifact(root, frame["valid_mask"]), allow_pickle=False)
            rows, columns = np.nonzero(valid)
            assert len(rows) > 0
            sample_indices = np.linspace(
                0,
                len(rows) - 1,
                num=min(128, len(rows)),
                dtype=int,
            )
            rows = rows[sample_indices]
            columns = columns[sample_indices]
            world_points = points[rows, columns]

            world_to_camera = np.linalg.inv(np.asarray(frame["camera_to_world"]))
            homogeneous = np.column_stack(
                [world_points, np.ones(len(world_points))]
            )
            camera_points = (world_to_camera @ homogeneous.T).T[:, :3]
            assert np.all(camera_points[:, 2] > 0)

            projected = (np.asarray(frame["K"]) @ camera_points.T).T
            projected = projected[:, :2] / projected[:, 2, None]
            source_pixel_centers = np.column_stack(
                [columns + 0.5, rows + 0.5]
            )
            assert np.allclose(projected, source_pixel_centers, atol=2e-4)


def test_generate_eval_pack_is_deterministic(generated_pack, tmp_path):
    first_root, first_manifest = generated_pack
    second_manifest = generate_eval_pack(tmp_path)
    assert second_manifest == first_manifest

    case_id = "ladder_050"
    first_case = first_manifest["cases"][case_id]
    second_case = second_manifest["cases"][case_id]
    assert (first_root / first_case["scene_ply"]).read_bytes() == (
        tmp_path / second_case["scene_ply"]
    ).read_bytes()

    first_frame = first_case["frames"][0]
    second_frame = second_case["frames"][0]
    assert (first_root / first_frame["rgb"]).read_bytes() == (
        tmp_path / second_frame["rgb"]
    ).read_bytes()
    first_points = np.load(first_root / first_frame["pts3d"], allow_pickle=False)
    second_points = np.load(tmp_path / second_frame["pts3d"], allow_pickle=False)
    assert np.array_equal(first_points, second_points, equal_nan=True)


def test_offline_benchmark_rejects_artifact_path_escapes_before_reads(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "pack"
    manifest = generate_eval_pack(root)
    frame = manifest["cases"]["ladder_050"]["frames"][0]
    absolute_rgb = str((root / frame["rgb"]).resolve())
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "escape").symlink_to(outside, target_is_directory=True)

    def fail_artifact_read(*_args, **_kwargs):
        pytest.fail("artifact read started before manifest validation")

    monkeypatch.setattr(np, "load", fail_artifact_read)
    monkeypatch.setattr(Image, "open", fail_artifact_read)

    for invalid_path in (
        absolute_rgb,
        "../outside/frame.png",
        "escape/frame.png",
        123,
    ):
        frame["rgb"] = invalid_path
        (root / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

        with pytest.raises(ValueError, match="relative path inside the pack root"):
            run_offline_benchmark(root)


def test_offline_benchmark_requires_exact_case_set_before_artifact_reads(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "pack"
    manifest = generate_eval_pack(root)
    cases = manifest["cases"]
    invalid_case_sets = (
        {},
        {key: value for key, value in cases.items() if key != "ladder_050"},
        {**cases, "unexpected_case": cases["ladder_050"]},
    )

    def fail_artifact_read(*_args, **_kwargs):
        pytest.fail("artifact read started before manifest validation")

    monkeypatch.setattr(np, "load", fail_artifact_read)
    monkeypatch.setattr(Image, "open", fail_artifact_read)

    for invalid_cases in invalid_case_sets:
        manifest["cases"] = invalid_cases
        (root / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

        with pytest.raises(ValueError, match="exactly the four calibrated cases"):
            run_offline_benchmark(root)


def test_offline_benchmark_matches_calibrated_metric_truth(generated_pack):
    root, _ = generated_pack

    report = run_offline_benchmark(root)

    assert report["passed"] is True
    assert report["cases"]["ladder_050"]["actual_status"] == "NEEDS_REVIEW"
    assert report["cases"]["ladder_050"]["actual_distance_m"] == pytest.approx(
        0.5, abs=0.1
    )
    assert report["cases"]["ladder_070"]["actual_status"] == "NEEDS_REVIEW"
    assert report["cases"]["ladder_070"]["actual_distance_m"] == pytest.approx(
        0.7, abs=0.1
    )
    assert report["cases"]["platform_inside"]["actual_status"] == "FAIL"
    assert report["cases"]["platform_inside"]["actual_distance_m"] == 0.0
    assert (
        report["cases"]["fence_occluded"]["actual_status"]
        == "INSUFFICIENT_EVIDENCE"
    )
    assert report["cases"]["fence_occluded"]["actual_distance_m"] is None

    for case_id in ("ladder_050", "ladder_070"):
        case_report = report["cases"][case_id]
        assert case_report["absolute_error_m"] <= 0.1


def test_offline_benchmark_records_reconciled_evidence_and_artifacts(
    generated_pack,
):
    root, manifest = generated_pack

    report = run_offline_benchmark(root)

    assert (
        json.loads((root / "offline_report.json").read_text(encoding="utf-8"))
        == report
    )
    assert set(report) == {"passed", "cases"}
    for case_id, case_manifest in manifest["cases"].items():
        case_report = report["cases"][case_id]
        assert set(case_report) == {
            "expected_status",
            "actual_status",
            "expected_distance_m",
            "actual_distance_m",
            "absolute_error_m",
            "entity_labels",
            "warnings",
            "passed",
            "artifacts",
        }
        assert case_report["expected_status"] == case_manifest["expected_status"]
        assert case_report["expected_distance_m"] == case_manifest[
            "expected_distance_m"
        ]
        assert case_report["passed"] is True
        assert case_report["artifacts"] == {
            "scene_json": f"{case_id}/scene.json",
            "assessment_json": f"{case_id}/assessment.json",
            "topdown_png": f"{case_id}/topdown.png",
        }

        scene_path = _artifact(root, case_report["artifacts"]["scene_json"])
        assessment_path = _artifact(
            root, case_report["artifacts"]["assessment_json"]
        )
        topdown_path = _artifact(root, case_report["artifacts"]["topdown_png"])
        scene = SceneMap.model_validate_json(scene_path.read_text(encoding="utf-8"))
        assessment = Assessment.model_validate_json(
            assessment_path.read_text(encoding="utf-8")
        )
        with Image.open(topdown_path) as topdown:
            assert topdown.format == "PNG"
            assert topdown.size == (640, 640)

        assert case_report["actual_status"] == assessment.status.value
        assert case_report["actual_distance_m"] == assessment.approximate_distance_m
        assert case_report["entity_labels"] == [
            entity.label for entity in scene.entities
        ]
        assert case_report["warnings"] == scene.warnings

        if case_id == "fence_occluded":
            continue

        movables = [
            entity
            for entity in scene.entities
            if entity.label == case_manifest["movable_label"]
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
            fact
            for fact in scene.facts
            if fact.predicate == "minimum_boundary_clearance"
        )

        assert len(movables) == 1
        assert len(set(movables[0].evidence_frame_ids)) >= 2
        assert robots
        assert len(fences) == 1
        assert len(set(fences[0].evidence_frame_ids)) >= 3
        assert clearance_fact.subject_id == movables[0].entity_id
        assert clearance_fact.subject_id not in {
            robot.entity_id for robot in robots
        }
