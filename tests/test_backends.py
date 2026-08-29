"""Backend routing: each env value dispatches to the right transport, and
the http branches speak the frozen contract in docs/BACKENDS.md."""

import base64
import io
import json

import pytest

from ehs_spatial import backends


class _Calls(list):
    response: dict = {}

    def set_response(self, r):
        self.response = r


@pytest.fixture
def http_calls(monkeypatch):
    calls = _Calls()

    def fake_http_json(url, payload, **_):
        calls.append((url, payload))
        return calls.response

    monkeypatch.setattr(backends, "http_json", fake_http_json)
    return calls


def test_service_backend_default_and_normalization(monkeypatch):
    monkeypatch.delenv("X_BACKEND", raising=False)
    assert backends.service_backend("X_BACKEND", "fal") == "fal"
    monkeypatch.setenv("X_BACKEND", "  Modal ")
    assert backends.service_backend("X_BACKEND", "fal") == "modal"
    monkeypatch.setenv("X_BACKEND", "")
    assert backends.service_backend("X_BACKEND", "fal") == "fal"


def test_sam_http_text_prompt(monkeypatch, http_calls):
    from ehs_spatial.providers import sam3

    monkeypatch.setenv("SAM3_BACKEND", "http")
    monkeypatch.setenv("SAM3_HTTP_URL", "http://gpu.internal/sam3")
    http_calls.set_response({"rle": [], "scores": []})
    image_b64 = base64.b64encode(b"png").decode()
    result = sam3.sam_subscribe(
        "fal-ai/sam3",
        arguments={
            "image_url": "data:image/png;base64," + image_b64,
            "prompt": "guard fence",
        },
    )
    assert result == {"rle": [], "scores": []}
    url, payload = http_calls[0]
    assert url == "http://gpu.internal/sam3"
    assert payload == {"image_b64": image_b64, "prompt": "guard fence"}


def test_sam_http_box_prompt(monkeypatch, http_calls):
    from ehs_spatial.providers import sam3

    monkeypatch.setenv("SAM3_BACKEND", "http")
    monkeypatch.setenv("SAM3_HTTP_URL", "http://gpu.internal/sam3")
    http_calls.set_response({"rle": ["x"], "scores": [0.9]})
    image_b64 = base64.b64encode(b"png").decode()
    sam3.sam_subscribe(
        "fal-ai/sam3",
        arguments={
            "image_url": "data:image/png;base64," + image_b64,
            "box_prompts": [
                {"x_min": 1, "y_min": 2, "x_max": 3, "y_max": 4}
            ],
        },
    )
    _, payload = http_calls[0]
    assert payload == {"image_b64": image_b64, "box": [1, 2, 3, 4]}


def test_sam_unknown_backend_fails_loudly(monkeypatch):
    from ehs_spatial.providers import sam3

    monkeypatch.setenv("SAM3_BACKEND", "banana")
    with pytest.raises(ValueError, match="banana"):
        sam3.sam_subscribe(
            "fal-ai/sam3",
            arguments={"image_url": "data:image/png;base64,QUJD", "prompt": "x"},
        )


def test_geometry_http_passes_replicate_input(monkeypatch, http_calls):
    from ehs_spatial.providers import map_anything

    monkeypatch.setenv("GEOMETRY_BACKEND", "http")
    monkeypatch.setenv("GEOMETRY_HTTP_URL", "http://gpu.internal/geometry")
    http_calls.set_response({"frames": []})
    payload_in = {"inputs": ["data:image/jpeg;base64,QUJD"], "apply_mask": True}
    result = map_anything._default_runner("model/id", input=payload_in)
    assert result == {"frames": []}
    url, payload = http_calls[0]
    assert url == "http://gpu.internal/geometry"
    assert payload == payload_in


def test_moge_http_reshapes_to_replicate_schema(monkeypatch, http_calls):
    from ehs_spatial.providers import moge

    monkeypatch.setenv("MOGE_BACKEND", "http")
    monkeypatch.setenv("MOGE_HTTP_URL", "http://gpu.internal/moge")
    ply = b"ply\nend_header\n"
    http_calls.set_response(
        {
            "ply_b64": base64.b64encode(ply).decode(),
            "intrinsics": [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]],
            "fov_x_deg": 62.0,
        }
    )
    image_b64 = base64.b64encode(b"png").decode()
    result = moge._default_runner(
        "model/id", input={"image": "data:image/png;base64," + image_b64}
    )
    _, payload = http_calls[0]
    assert payload == {"image_b64": image_b64}
    assert isinstance(result["pointcloud_ply"], io.BytesIO)
    assert result["pointcloud_ply"].read() == ply
    assert json.load(result["intrinsics_json"])["intrinsics"][0][0] == 1.0
    assert result["fov_x_deg"] == 62.0


def test_moge_replicate_branch_dispatches(monkeypatch):
    from ehs_spatial.providers import moge

    monkeypatch.setenv("MOGE_BACKEND", "replicate")
    seen = {}
    monkeypatch.setattr(
        moge,
        "_replicate_runner",
        lambda model, *, input: seen.update(model=model) or "replicate-output",
    )
    assert moge._default_runner("m/v", input={"image": "x"}) == "replicate-output"
    assert seen["model"] == "m/v"
