"""Native world points, c2w, masks and fixed pixels survive the production decoder."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from ehs_spatial.providers.map_anything import decode_encoded_array, parse_frame_json

SPEC = importlib.util.spec_from_file_location(
    "candidate_mapanything_backend", Path(__file__).parents[1] / "scripts/candidate_mapanything_backend.py")
backend = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(backend)
predictions_to_response = backend.predictions_to_response


def test_native_geometry_mask_and_identity_grid(tmp_path):
    height, width = 4, 5
    image = np.arange(height * width * 3, dtype=np.uint8).reshape(height, width, 3)
    k = np.array([[2, 0, 1], [0, 3, 1], [0, 0, 1]], dtype=np.float32)
    pose = np.array([[0, -1, 0, 4], [1, 0, 0, -2], [0, 0, 1, 3],
                     [0, 0, 0, 1]], dtype=np.float32)
    yy, xx = np.indices((height, width), dtype=np.float32)
    camera = np.stack([(xx - 1) / 2, (yy - 1) / 3, np.ones_like(xx)], -1) * 2
    points = camera @ pose[:3, :3].T + pose[:3, 3]
    native_mask = np.ones((height, width), dtype=bool)
    native_mask[0, 0] = False
    prediction = {
        "img_no_norm": image[None].astype(np.float32) / 255,
        "pts3d": points[None], "conf": np.full((1, height, width), 7, np.float32),
        "non_ambiguous_mask": native_mask[None],
        "depth_z": np.full((1, height, width, 1), 2, np.float32),
        "camera_poses": pose[None], "intrinsics": k[None],
    }
    response = predictions_to_response([prediction], [image])
    assert response["point_cloud"][:4] == b"glTF"
    source = tmp_path / "native.json"
    source.write_bytes(response["data"][0])
    frame = parse_frame_json(source, tmp_path / "decoded", "frame_0000")
    np.testing.assert_array_equal(np.load(frame.pts3d_path), points)
    np.testing.assert_array_equal(frame.camera_to_world, pose)
    np.testing.assert_array_equal(np.load(frame.valid_mask_path), native_mask)
    payload = json.loads(response["data"][0])
    np.testing.assert_array_equal(decode_encoded_array(payload["image"]), image)
    assert decode_encoded_array(payload["conf"])[0, 0] == 7
    recovered = (np.load(frame.pts3d_path) - pose[:3, 3]) @ pose[:3, :3]
    projected = recovered @ k.T
    np.testing.assert_allclose(projected[..., :2] / projected[..., 2:],
                               np.stack([xx, yy], -1), atol=1e-6)

    prediction["img_no_norm"] = prediction["img_no_norm"][:, :, ::-1].copy()
    with pytest.raises(ValueError, match="identity pixel grid"):
        predictions_to_response([prediction], [image])
    prediction["img_no_norm"] = image[None].astype(np.float32) / 255
    prediction["depth_z"][0, :, -1, 0] = 10
    response = predictions_to_response([prediction], [image])
    mask = decode_encoded_array(json.loads(response["data"][0])["non_ambiguous_mask"])
    assert not mask[:, -2:].any()  # Declared relative depth edge, not invented visibility.
    prediction["camera_poses"][0, 0, 0] = 2
    with pytest.raises(ValueError, match="rigid c2w"):
        predictions_to_response([prediction], [image])
