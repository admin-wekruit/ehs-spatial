"""Actual upstream pinhole fit plus preservation of non-pinhole native geometry."""

import json
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

from ehs_spatial.providers.map_anything import decode_encoded_array
SPEC = importlib.util.spec_from_file_location(
    "candidate_pi3x_backend", Path(__file__).parents[1] / "scripts/candidate_pi3x_backend.py")
backend = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(backend)


def test_upstream_intrinsic_fit_and_native_point_preservation():
    torch = pytest.importorskip("torch")
    vendor = Path(__file__).parents[1] / "outputs/candidate-evaluation/vendor/pi3"
    if not vendor.is_dir():
        pytest.skip("This research check requires the pinned official Pi3X source checkout")
    sys.path.insert(0, str(vendor))
    h, w = 5, 7
    yy, xx = np.indices((h, w))
    k = np.array([[8, 0, 3], [0, 9, 2], [0, 0, 1]], dtype=np.float32)
    local = (np.stack([xx, yy, np.ones_like(xx)], axis=-1) @ np.linalg.inv(k).T * 2).astype(np.float32)
    pose = np.array([[0, -1, 0, 4], [1, 0, 0, -2], [0, 0, 1, 3], [0, 0, 0, 1]], dtype=np.float32)
    rgb = np.full((1, h, w, 3), 127, dtype=np.uint8)

    def convert(local_points):
        points = local_points @ pose[:3, :3].T + pose[:3, 3]
        return backend.pi3x_prediction_to_response({
            "local_points": torch.from_numpy(local_points)[None, None],
            "points": torch.from_numpy(points)[None, None],
            "camera_poses": torch.from_numpy(pose)[None, None],
            "conf": torch.zeros(1, 1, h, w, 1),
        }, rgb), points

    response, expected = convert(local)
    frame = json.loads(response["data"][0])
    np.testing.assert_allclose(decode_encoded_array(frame["intrinsics"]), k, atol=1e-5)
    np.testing.assert_allclose(decode_encoded_array(frame["camera_poses"]), pose, atol=1e-6)
    np.testing.assert_array_equal(decode_encoded_array(frame["pts3d"]), expected)
    assert response["native_pinhole_fit"][0]["max_px"] < 1e-5

    # A non-pinhole local ray perturbation must survive serialization, with a
    # nonzero fitting residual. Rebuilding points from K and z would hide this.
    local[1, 1, 0] += 0.2
    response, expected = convert(local)
    frame = json.loads(response["data"][0])
    np.testing.assert_array_equal(decode_encoded_array(frame["pts3d"]), expected)
    assert response["native_pinhole_fit"][0]["max_px"] > 0.5
