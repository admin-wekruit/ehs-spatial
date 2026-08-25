import base64
import email.message
import importlib
import io
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


def encoded_array(array):
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "data": base64.b64encode(array.tobytes()).decode("ascii"),
    }


def provider_frame_payload():
    image = np.array(
        [
            [[255, 0, 0], [0, 255, 0], [0, 0, 255]],
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        ],
        dtype=np.uint8,
    )
    pts3d = np.arange(18, dtype=np.float32).reshape(2, 3, 3)
    conf = np.arange(6, dtype=np.float32).reshape(2, 3) / 10
    valid = np.array([[True, False, True], [True, True, False]])
    return {
        "pts3d": encoded_array(pts3d),
        "pts3d_cam": encoded_array(pts3d + 100),
        "ray_directions": encoded_array(np.ones((2, 3, 3), dtype=np.float32)),
        "depth_along_ray": encoded_array(np.ones((2, 3), dtype=np.float32)),
        "cam_trans": encoded_array(np.array([1, 2, 3], dtype=np.float32)),
        "cam_quats": encoded_array(np.array([1, 0, 0, 0], dtype=np.float32)),
        "metric_scaling_factor": encoded_array(np.array(10, dtype=np.float32)),
        "conf": encoded_array(conf),
        "non_ambiguous_mask": encoded_array(valid),
        "non_ambiguous_mask_logits": encoded_array(conf + 1),
        "depth_z": encoded_array(np.ones((2, 3), dtype=np.float32) * 2),
        "intrinsics": encoded_array(
            np.array([[100, 0, 1], [0, 100, 1], [0, 0, 1]], dtype=np.float32)
        ),
        "camera_poses": encoded_array(np.eye(4, dtype=np.float32)),
        "mask": encoded_array(valid),
        "image": encoded_array(image),
        "alpha_mask": encoded_array(np.ones((2, 3), dtype=np.float32)),
        "original_image": {"width": 6, "height": 4, "frame_index": 0},
    }


def test_decode_encoded_array_restores_shape_dtype_and_values():
    map_anything = importlib.import_module("ehs_spatial.providers.map_anything")
    expected = np.arange(12, dtype=np.float32).reshape(2, 2, 3)

    decoded = map_anything.decode_encoded_array(encoded_array(expected))

    assert decoded.dtype == np.dtype("float32")
    np.testing.assert_array_equal(decoded, expected)


def test_decode_encoded_array_rejects_byte_count_shape_mismatch():
    map_anything = importlib.import_module("ehs_spatial.providers.map_anything")
    payload = {
        "shape": [2, 2],
        "dtype": "uint8",
        "data": base64.b64encode(b"\x00\x01\x02").decode("ascii"),
    }

    with pytest.raises(ValueError, match="byte count"):
        map_anything.decode_encoded_array(payload)


def test_parse_provider_frame_persists_pixel_aligned_geometry_without_rescaling(tmp_path):
    map_anything = importlib.import_module("ehs_spatial.providers.map_anything")
    payload = provider_frame_payload()
    source = tmp_path / "provider-frame.json"
    source.write_text(json.dumps(payload), encoding="utf-8")

    frame = map_anything.parse_frame_json(source, tmp_path / "frame-1", "frame-1")

    canonical = np.asarray(Image.open(frame.canonical_image_path))
    pts3d = np.load(frame.pts3d_path)
    assert canonical.shape[:2] == pts3d.shape[:2] == (2, 3)
    np.testing.assert_array_equal(canonical, map_anything.decode_encoded_array(payload["image"]))
    np.testing.assert_array_equal(pts3d, map_anything.decode_encoded_array(payload["pts3d"]))
    np.testing.assert_array_equal(np.load(frame.conf_path), map_anything.decode_encoded_array(payload["conf"]))
    np.testing.assert_array_equal(
        np.load(frame.valid_mask_path),
        map_anything.decode_encoded_array(payload["non_ambiguous_mask"]),
    )
    np.testing.assert_array_equal(np.load(tmp_path / "frame-1" / "camera_to_world.npy"), np.eye(4))
    np.testing.assert_array_equal(
        np.load(tmp_path / "frame-1" / "intrinsics.npy"),
        np.array([[100, 0, 1], [0, 100, 1], [0, 0, 1]]),
    )


def test_parse_provider_frame_rejects_non_aligned_pixel_arrays(tmp_path):
    map_anything = importlib.import_module("ehs_spatial.providers.map_anything")
    payload = provider_frame_payload()
    payload["pts3d"] = encoded_array(np.zeros((1, 3, 3), dtype=np.float32))
    source = tmp_path / "misaligned.json"
    source.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="pixel-aligned"):
        map_anything.parse_frame_json(source, tmp_path / "frame-1", "frame-1")


def test_adapter_persists_complete_provider_output_and_returns_geometry_frames(tmp_path):
    map_anything = importlib.import_module("ehs_spatial.providers.map_anything")
    image_paths = []
    data_paths = []
    for index in range(4):
        image_path = tmp_path / f"input-{index}.png"
        Image.new("RGB", (3, 2), (index, index, index)).save(image_path)
        image_paths.append(str(image_path))
        data_path = tmp_path / f"provider-{index}.json"
        data_path.write_text(json.dumps(provider_frame_payload()), encoding="utf-8")
        data_paths.append(str(data_path))
    mesh = tmp_path / "mesh.glb"
    mesh.write_bytes(b"mesh")
    point_cloud = tmp_path / "provider-point-cloud.glb"
    point_cloud.write_bytes(b"glTF-provider-point-cloud")
    provider_response = {
        "data": data_paths,
        "mesh": str(mesh),
        "point_cloud": str(point_cloud),
    }
    seen = {}

    def runner(model_identifier, *, input):
        seen["model_identifier"] = model_identifier
        seen["input_values"] = list(input["inputs"])
        seen["flags"] = {key: value for key, value in input.items() if key != "inputs"}
        return provider_response

    geometry_dir = tmp_path / "run" / "geometry"
    frames, saved_point_cloud = map_anything.MapAnythingAdapter(
        runner=runner, allow_local=True
    ).run(
        image_paths,
        geometry_dir,
    )

    assert seen == {
        "model_identifier": map_anything.MAP_ANYTHING_MODEL_ID,
        # The model's Cog wrapper cannot fetch authenticated Files-API URLs, so
        # images must travel as self-contained base64 data URIs.
        "input_values": [
            "data:image/png;base64,"
            + base64.b64encode(Path(path).read_bytes()).decode("ascii")
            for path in image_paths
        ],
        "flags": {
            "normals": False,
            "to_base64": True,
            "return_pcd": True,
            "return_mesh": False,
            "point_scales": False,
            "keys_to_exclude": "",
            "alpha_blend_onto": "white",
        },
    }
    assert len(frames) == 4
    assert [frame.frame_id for frame in frames] == [
        "frame_0001",
        "frame_0002",
        "frame_0003",
        "frame_0004",
    ]
    assert saved_point_cloud == geometry_dir / "point_cloud.glb"
    assert saved_point_cloud.read_bytes() == b"glTF-provider-point-cloud"
    np.testing.assert_array_equal(
        np.load(frames[2].pts3d_path),
        decode_fixture := map_anything.decode_encoded_array(provider_frame_payload()["pts3d"]),
    )
    assert decode_fixture[1, 2, 2] == 17
    assert json.loads((geometry_dir / "map_anything_request.json").read_text()) == {
        "model_identifier": map_anything.MAP_ANYTHING_MODEL_ID,
        "input": {**seen["flags"], "inputs": image_paths},
    }
    assert json.loads((geometry_dir / "map_anything_response.json").read_text()) == provider_response
    assert json.loads((geometry_dir / "provider" / "frame_0001.json").read_text())["original_image"] == {
        "width": 6,
        "height": 4,
        "frame_index": 0,
    }


def test_adapter_requires_one_to_four_local_images(tmp_path):
    map_anything = importlib.import_module("ehs_spatial.providers.map_anything")
    images = []
    for index in range(5):
        path = tmp_path / f"image-{index}.png"
        Image.new("RGB", (1, 1)).save(path)
        images.append(str(path))

    with pytest.raises(ValueError, match="one to four image paths"):
        map_anything.MapAnythingAdapter(runner=lambda *args, **kwargs: {}).run(
            images,
            tmp_path / "geometry",
        )
    with pytest.raises(ValueError, match="one to four image paths"):
        map_anything.MapAnythingAdapter(runner=lambda *args, **kwargs: {}).run(
            [],
            tmp_path / "geometry",
        )


def test_adapter_wraps_provider_failures_with_operation_context(tmp_path):
    map_anything = importlib.import_module("ehs_spatial.providers.map_anything")
    images = []
    for index in range(4):
        path = tmp_path / f"image-{index}.png"
        Image.new("RGB", (1, 1)).save(path)
        images.append(str(path))

    def failing_runner(*args, **kwargs):
        raise TimeoutError("provider timed out")

    with pytest.raises(map_anything.ProviderError) as caught:
        map_anything.MapAnythingAdapter(runner=failing_runner).run(
            images,
            tmp_path / "geometry",
        )

    assert caught.value.provider == "replicate"
    assert caught.value.operation == "map_anything.run"
    assert caught.value.original_message == "provider timed out"


def test_adapter_wraps_provider_file_read_value_error(tmp_path):
    map_anything = importlib.import_module("ehs_spatial.providers.map_anything")
    images = []
    data_paths = []
    for index in range(4):
        image_path = tmp_path / f"input-{index}.png"
        Image.new("RGB", (1, 1)).save(image_path)
        images.append(str(image_path))
        data_path = tmp_path / f"data-{index}.json"
        data_path.write_text(json.dumps(provider_frame_payload()))
        data_paths.append(str(data_path))
    mesh = tmp_path / "mesh.glb"
    mesh.write_bytes(b"mesh")
    point_cloud = tmp_path / "point-cloud.glb"
    point_cloud.write_bytes(b"glTF")

    class FailingRemoteFile:
        url = "https://replicate.delivery/fake/frame.json"

        def read(self):
            raise ValueError("provider stream could not be read")

    response = {
        "data": [FailingRemoteFile(), *data_paths[1:]],
        "mesh": str(mesh),
        "point_cloud": str(point_cloud),
    }

    with pytest.raises(map_anything.ProviderError) as caught:
        map_anything.MapAnythingAdapter(
            runner=lambda model_identifier, *, input: response
        ).run(images, tmp_path / "geometry")

    assert caught.value.provider == "replicate"
    assert caught.value.operation == "map_anything.download"
    assert caught.value.original_message == "provider stream could not be read"


@pytest.mark.parametrize(
    ("response", "message"),
    [
        ([], "MapAnything response must be an object"),
        (
            {"data": [], "point_cloud": "unused.glb"},
            "MapAnything response must contain one data file per input image (4)",
        ),
        (
            {"data": ["unused.json"] * 4},
            "MapAnything response is missing point_cloud",
        ),
    ],
)
def test_adapter_wraps_provider_response_shape_errors(tmp_path, response, message):
    map_anything = importlib.import_module("ehs_spatial.providers.map_anything")
    images = []
    for index in range(4):
        image_path = tmp_path / f"input-{index}.png"
        Image.new("RGB", (1, 1)).save(image_path)
        images.append(str(image_path))

    with pytest.raises(map_anything.ProviderError) as caught:
        map_anything.MapAnythingAdapter(
            runner=lambda model_identifier, *, input: response
        ).run(images, tmp_path / "geometry")

    assert caught.value.provider == "replicate"
    assert caught.value.operation == "map_anything.response"
    assert caught.value.original_message == message


def test_adapter_persists_all_raw_outputs_before_parsing_frames(tmp_path):
    map_anything = importlib.import_module("ehs_spatial.providers.map_anything")
    images = []
    data_paths = []
    for index in range(4):
        image_path = tmp_path / f"input-{index}.png"
        Image.new("RGB", (1, 1)).save(image_path)
        images.append(str(image_path))
        data_path = tmp_path / f"data-{index}.json"
        data_path.write_text("not-json" if index == 0 else json.dumps(provider_frame_payload()))
        data_paths.append(str(data_path))
    point_cloud = tmp_path / "provider.glb"
    point_cloud.write_bytes(b"glTF")
    response = {"data": data_paths, "mesh": None, "point_cloud": str(point_cloud)}
    geometry_dir = tmp_path / "geometry"

    with pytest.raises(map_anything.ProviderError) as caught:
        map_anything.MapAnythingAdapter(
            runner=lambda model_identifier, *, input: response, allow_local=True
        ).run(images, geometry_dir)

    assert caught.value.provider == "replicate"
    assert caught.value.operation == "map_anything.decode"
    assert "Expecting value" in caught.value.original_message
    assert [
        (geometry_dir / "provider" / f"frame_{index:04d}.json").read_bytes()
        for index in range(1, 5)
    ] == [Path(path).read_bytes() for path in data_paths]
    assert (geometry_dir / "point_cloud.glb").read_bytes() == b"glTF"


def test_default_runner_uses_polling_instead_of_blocking_wait(monkeypatch):
    import sys
    import types

    map_anything = importlib.import_module("ehs_spatial.providers.map_anything")
    seen = {}

    def fake_run(model_identifier, *, input, wait):
        seen["model"] = model_identifier
        seen["input"] = input
        seen["wait"] = wait
        return {"ok": True}

    monkeypatch.setitem(sys.modules, "replicate", types.SimpleNamespace(run=fake_run))

    result = map_anything._default_runner("m/x:abc", input={"k": 1})

    assert result == {"ok": True}
    assert seen == {"model": "m/x:abc", "input": {"k": 1}, "wait": False}


def test_provider_bytes_rejects_local_paths_and_foreign_hosts_by_default(tmp_path):
    map_anything = importlib.import_module("ehs_spatial.providers.map_anything")
    secret = tmp_path / "secret.txt"
    secret.write_bytes(b"leak")

    # O3: a hostile provider response must not read local files ...
    with pytest.raises(ValueError, match="unsupported provider file location"):
        map_anything._read_provider_bytes(str(secret))
    with pytest.raises(ValueError, match="unsupported provider file location"):
        map_anything._read_provider_bytes(f"file://{secret}")
    # ... nor fetch from arbitrary origins.
    with pytest.raises(ValueError, match="host not allowed"):
        map_anything._read_provider_bytes("https://evil.example.com/x.bin")
    # The explicit replay seam still works.
    assert map_anything._read_provider_bytes(str(secret), allow_local=True) == b"leak"
    assert (
        map_anything._read_provider_bytes(f"file://{secret}", allow_local=True)
        == b"leak"
    )


def test_provider_bytes_refuses_redirects_from_allowed_hosts(monkeypatch):
    map_anything = importlib.import_module("ehs_spatial.providers.map_anything")
    from urllib.request import HTTPSHandler
    from urllib.response import addinfourl

    class FakeTransport(HTTPSHandler):
        """Answers every https request with a 302 to a foreign origin, so the
        test exercises the real opener stack without touching the network."""

        def https_open(self, req):
            headers = email.message.Message()
            headers["Location"] = "http://169.254.169.254/latest/meta-data/"
            response = addinfourl(io.BytesIO(b""), headers, req.full_url, code=302)
            response.msg = "Found"
            return response

    monkeypatch.setattr(
        map_anything,
        "_OPENER",
        map_anything._redirect_refusing_opener(FakeTransport()),
    )

    # The initial host passes the allow-list; the redirect must still die.
    with pytest.raises(ValueError, match="redirect refused"):
        map_anything._read_provider_bytes("https://x.replicate.delivery/pc.glb")
