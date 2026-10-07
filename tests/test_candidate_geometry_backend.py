"""Known camera/depth check, including the real production decoder; no model mock."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ehs_spatial.providers.map_anything import decode_encoded_array, parse_frame_json

SPEC = importlib.util.spec_from_file_location(
    "candidate_geometry_backend", Path(__file__).parents[1] / "scripts/candidate_geometry_backend.py")
backend = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(backend)


def test_known_cameras_depth_and_production_serialization(tmp_path):
    k = np.array([[2, 0, 1], [0, 3, 1], [0, 0, 1]], dtype=np.float32)
    c2w = np.array([[0, -1, 0, 4], [1, 0, 0, -2], [0, 0, 1, 3], [0, 0, 0, 1]], dtype=np.float32)
    prediction = SimpleNamespace(
        depth=np.full((2, 3, 4), 2, dtype=np.float32),
        processed_images=np.full((2, 3, 4, 3), 123, dtype=np.uint8),
        conf=np.full((2, 3, 4), 7, dtype=np.float32),
        intrinsics=np.stack([k, k]),
        extrinsics=np.stack([np.eye(4), np.linalg.inv(c2w)])[:, :3],
    )
    response = backend.prediction_to_response(prediction)
    assert len(response["data"]) == 2
    assert response["point_cloud"][:4] == b"glTF"
    for index, expected_pose in enumerate([np.eye(4), c2w]):
        source = tmp_path / f"frame{index}.json"
        source.write_bytes(response["data"][index])
        frame = parse_frame_json(source, tmp_path / f"decoded{index}", f"frame_{index:04d}")
        points = np.load(frame.pts3d_path)
        np.testing.assert_allclose(frame.camera_to_world, expected_pose, atol=1e-6)
        # At the principal point, z-depth 2 yields [0,0,2] in each camera.
        np.testing.assert_allclose(points[1, 1], expected_pose[:3, 3] + [0, 0, 2])
        world_to_camera = np.linalg.inv(expected_pose)
        camera_points = points @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
        projected = camera_points @ k.T
        projected = projected[..., :2] / projected[..., 2:]
        yy, xx = np.indices((3, 4))
        np.testing.assert_allclose(projected, np.stack([xx, yy], axis=-1), atol=1e-6)
        assert np.load(frame.valid_mask_path).all()
        payload = json.loads(response["data"][index])
        assert decode_encoded_array(payload["conf"])[0, 0] == 7  # Native score, no invented probability.
    prediction.extrinsics[0, 0, 0] = 2
    with pytest.raises(ValueError, match="orthonormal"):
        backend.prediction_to_response(prediction)
