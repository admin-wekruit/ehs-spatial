import json
from pathlib import Path

import numpy as np
import open3d as o3d
import pytest
from PIL import Image

from ehs_spatial.eval_pack import generate_eval_pack


WIDTH = 512
HEIGHT = 384
COMMON_LABELS = {"factory floor", "safety fence", "industrial robot arm"}
CASE_EXPECTATIONS = {
    "ladder_050": ("FAIL", 0.5, "step ladder"),
    "ladder_070": ("PASS", 0.7, "step ladder"),
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


def test_generate_eval_pack_scene_ply_round_trips_with_colored_meshes(
    generated_pack,
):
    root, manifest = generated_pack

    for case in manifest["cases"].values():
        scene = o3d.io.read_triangle_mesh(str(_artifact(root, case["scene_ply"])))
        vertices = np.asarray(scene.vertices)
        triangles = np.asarray(scene.triangles)
        colors = np.asarray(scene.vertex_colors)
        assert len(vertices) > 0
        assert len(triangles) > 0
        assert colors.shape == vertices.shape
        assert np.isfinite(colors).all()
