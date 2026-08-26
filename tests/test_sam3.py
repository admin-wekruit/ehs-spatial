import base64
import importlib
import json

import numpy as np
import pytest
from PIL import Image


def test_decode_coco_rle_supports_serialized_standard_object():
    sam3 = importlib.import_module("ehs_spatial.providers.sam3")
    serialized = json.dumps({"size": [2, 3], "counts": [1, 2, 3]})

    mask = sam3.decode_coco_rle(serialized)

    assert mask.dtype == np.uint8
    np.testing.assert_array_equal(
        mask,
        np.array([[0, 1, 0], [1, 0, 0]], dtype=np.uint8),
    )


def test_decode_coco_rle_supports_compressed_and_counts_only_forms():
    sam3 = importlib.import_module("ehs_spatial.providers.sam3")
    expected = np.array([[0, 1, 0], [1, 0, 0]], dtype=np.uint8)

    compressed_object = json.dumps({"size": [2, 3], "counts": "123"})
    np.testing.assert_array_equal(sam3.decode_coco_rle(compressed_object), expected)
    np.testing.assert_array_equal(
        sam3.decode_coco_rle("123", height=2, width=3),
        expected,
    )


def test_decode_coco_rle_supports_fal_one_indexed_row_major_pairs():
    sam3 = importlib.import_module("ehs_spatial.providers.sam3")

    mask = sam3.decode_coco_rle("1 2 5 1", height=2, width=3)

    np.testing.assert_array_equal(
        mask,
        np.array([[1, 1, 0], [0, 1, 0]], dtype=np.uint8),
    )


def test_decode_coco_rle_fails_loudly_for_unknown_object_shape():
    sam3 = importlib.import_module("ehs_spatial.providers.sam3")

    with pytest.raises(ValueError, match="requires size and counts"):
        sam3.decode_coco_rle(json.dumps({"size": [2, 2], "unexpected": []}))


def test_adapter_normalizes_complete_fal_response_and_resizes_masks_nearest(tmp_path):
    sam3 = importlib.import_module("ehs_spatial.providers.sam3")
    canonical_path = tmp_path / "canonical.png"
    Image.new("RGB", (4, 4), (10, 20, 30)).save(canonical_path)
    provider_response = {
        "rle": [
            json.dumps({"size": [2, 2], "counts": [0, 4]}),
            json.dumps({"size": [2, 2], "counts": [0, 1, 3]}),
        ],
        "boundingbox_frames_zip": {
            "url": "https://fal.media/example/boxes.zip",
            "content_type": "application/zip",
            "file_name": "boxes.zip",
            "file_size": 123,
        },
        "metadata": [
            {"index": 7, "score": 0.91, "box": [0.5, 0.5, 1.0, 1.0]},
            {"index": 8, "score": 0.82, "box": [0.25, 0.25, 0.5, 0.5]},
        ],
        "scores": [0.91, 0.82],
        "boxes": [[0.5, 0.5, 1.0, 1.0], [0.25, 0.25, 0.5, 0.5]],
    }
    seen = {}

    def subscriber(endpoint, *, arguments):
        seen["endpoint"] = endpoint
        seen["arguments"] = arguments
        return provider_response

    observations = sam3.SAM3Adapter(subscriber=subscriber).segment(
        canonical_path,
        prompt="pallet",
        frame_id="frame_0001",
        output_dir=tmp_path / "masks",
    )

    assert seen["endpoint"] == "fal-ai/sam-3-1/image-rle"
    assert {key: value for key, value in seen["arguments"].items() if key != "image_url"} == {
        "prompt": "pallet",
        "return_multiple_masks": True,
        "include_scores": True,
        "include_boxes": True,
    }
    prefix, encoded_image = seen["arguments"]["image_url"].split(",", 1)
    assert prefix == "data:image/png;base64"
    assert base64.b64decode(encoded_image) == canonical_path.read_bytes()
    assert [observation.instance_id for observation in observations] == ["7", "8"]
    assert [observation.score for observation in observations] == [0.91, 0.82]
    assert observations[1].bbox == (0.25, 0.25, 0.5, 0.5)
    assert all(observation.source_prompt == "pallet" for observation in observations)
    first_mask = np.asarray(Image.open(observations[0].mask_path))
    second_mask = np.asarray(Image.open(observations[1].mask_path))
    np.testing.assert_array_equal(first_mask, np.full((4, 4), 255, dtype=np.uint8))
    np.testing.assert_array_equal(
        second_mask,
        np.array(
            [
                [255, 255, 0, 0],
                [255, 255, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.uint8,
        ),
    )
    assert set(sam3.PROMPT_VOCABULARY) == {
        "factory floor",
        "safety fence",
        "industrial robot arm",
        "material cart",
        "pallet",
        "crate",
        "step ladder",
        "portable work platform",
        "safety sensor",
        "emergency stop button",
        "warning sign",
        "safety light",
    }


def test_adapter_normalizes_single_rle_string(tmp_path):
    sam3 = importlib.import_module("ehs_spatial.providers.sam3")
    canonical_path = tmp_path / "canonical.png"
    Image.new("RGB", (2, 2)).save(canonical_path)
    provider_response = {
        "rle": json.dumps({"size": [2, 2], "counts": [0, 4]}),
        "boundingbox_frames_zip": None,
        "metadata": [{"index": 3, "score": 0.75, "box": [0.5, 0.5, 1, 1]}],
        "scores": [0.75],
        "boxes": [[0.5, 0.5, 1, 1]],
    }

    observations = sam3.SAM3Adapter(
        subscriber=lambda endpoint, *, arguments: provider_response
    ).segment(
        canonical_path,
        prompt="crate",
        frame_id="frame_0002",
        output_dir=tmp_path / "masks",
    )

    assert len(observations) == 1
    assert observations[0].instance_id == "3"
    np.testing.assert_array_equal(
        np.asarray(Image.open(observations[0].mask_path)),
        np.full((2, 2), 255, dtype=np.uint8),
    )


def test_adapter_wraps_fal_failure_with_operation_context(tmp_path):
    sam3 = importlib.import_module("ehs_spatial.providers.sam3")
    canonical_path = tmp_path / "canonical.png"
    Image.new("RGB", (2, 2)).save(canonical_path)

    def failing_subscriber(*args, **kwargs):
        raise RuntimeError("queue unavailable")

    with pytest.raises(sam3.ProviderError) as caught:
        sam3.SAM3Adapter(subscriber=failing_subscriber).segment(
            canonical_path,
            prompt="pallet",
            frame_id="frame_0001",
            output_dir=tmp_path / "masks",
        )

    assert caught.value.provider == "fal"
    assert caught.value.operation == "sam3.subscribe"
    assert caught.value.original_message == "queue unavailable"


def test_adapter_rejects_prompts_outside_fixed_vocabulary(tmp_path):
    sam3 = importlib.import_module("ehs_spatial.providers.sam3")
    canonical_path = tmp_path / "canonical.png"
    Image.new("RGB", (2, 2)).save(canonical_path)

    with pytest.raises(ValueError, match="unsupported SAM 3 prompt"):
        sam3.SAM3Adapter(subscriber=lambda *args, **kwargs: {}).segment(
            canonical_path,
            prompt="anything nearby",
            frame_id="frame_0001",
            output_dir=tmp_path / "masks",
        )


@pytest.mark.parametrize(
    "frame_id",
    ["../escaped", "/absolute", "nested/path", ".", "..", "back\\slash", ""],
)
def test_adapter_rejects_unsafe_frame_id_before_provider_or_write(tmp_path, frame_id):
    sam3 = importlib.import_module("ehs_spatial.providers.sam3")
    canonical_path = tmp_path / "canonical.png"
    Image.new("RGB", (2, 2)).save(canonical_path)
    provider_called = False

    def subscriber(*args, **kwargs):
        nonlocal provider_called
        provider_called = True
        raise AssertionError("unsafe frame id reached provider")

    output_dir = tmp_path / "masks"
    with pytest.raises(ValueError, match="frame_id"):
        sam3.SAM3Adapter(subscriber=subscriber).segment(
            canonical_path,
            prompt="pallet",
            frame_id=frame_id,
            output_dir=output_dir,
        )

    assert provider_called is False
    assert not output_dir.exists()


def test_adapter_wraps_malformed_fal_response(tmp_path):
    sam3 = importlib.import_module("ehs_spatial.providers.sam3")
    canonical_path = tmp_path / "canonical.png"
    Image.new("RGB", (2, 2)).save(canonical_path)
    response = {"rle": 123, "metadata": [], "scores": [], "boxes": []}

    with pytest.raises(sam3.ProviderError) as caught:
        sam3.SAM3Adapter(
            subscriber=lambda endpoint, *, arguments: response
        ).segment(
            canonical_path,
            prompt="pallet",
            frame_id="frame_0001",
            output_dir=tmp_path / "masks",
        )

    assert caught.value.provider == "fal"
    assert caught.value.operation == "sam3.response"
    assert caught.value.original_message == (
        "SAM 3 response rle must be a string or list of strings"
    )


def test_adapter_wraps_invalid_provider_rle(tmp_path):
    sam3 = importlib.import_module("ehs_spatial.providers.sam3")
    canonical_path = tmp_path / "canonical.png"
    Image.new("RGB", (2, 2)).save(canonical_path)
    response = {
        "rle": json.dumps({"size": [2, 2], "counts": [5]}),
        "metadata": [{"index": 0, "score": 0.9, "box": [0.5, 0.5, 1, 1]}],
        "scores": [0.9],
        "boxes": [[0.5, 0.5, 1, 1]],
    }

    with pytest.raises(sam3.ProviderError) as caught:
        sam3.SAM3Adapter(
            subscriber=lambda endpoint, *, arguments: response
        ).segment(
            canonical_path,
            prompt="pallet",
            frame_id="frame_0001",
            output_dir=tmp_path / "masks",
        )

    assert caught.value.provider == "fal"
    assert caught.value.operation == "sam3.decode"
    assert caught.value.original_message == "invalid COCO RLE run lengths"


@pytest.mark.parametrize(
    ("scores", "boxes", "message"),
    [
        ([], [], "SAM 3 response is missing requested score or box"),
        ([2.0], [[0.5, 0.5, 1, 1]], "less than or equal to 1"),
    ],
)
def test_adapter_wraps_provider_normalization_errors(
    tmp_path, scores, boxes, message
):
    sam3 = importlib.import_module("ehs_spatial.providers.sam3")
    canonical_path = tmp_path / "canonical.png"
    Image.new("RGB", (2, 2)).save(canonical_path)
    response = {
        "rle": json.dumps({"size": [2, 2], "counts": [0, 4]}),
        "metadata": [{"index": 0}],
        "scores": scores,
        "boxes": boxes,
    }

    with pytest.raises(sam3.ProviderError) as caught:
        sam3.SAM3Adapter(
            subscriber=lambda endpoint, *, arguments: response
        ).segment(
            canonical_path,
            prompt="pallet",
            frame_id="frame_0001",
            output_dir=tmp_path / "masks",
        )

    assert caught.value.provider == "fal"
    assert caught.value.operation == "sam3.normalize"
    assert message in caught.value.original_message


def test_adapter_rejects_colliding_instance_ids(tmp_path):
    sam3 = importlib.import_module("ehs_spatial.providers.sam3")
    canonical_path = tmp_path / "canonical.png"
    Image.new("RGB", (4, 4), (10, 20, 30)).save(canonical_path)
    # Metadata covers only the first mask; the second falls back to its
    # ordinal (1), colliding with the first mask's explicit index 1. Without
    # the guard, both observations share one mask path and the second save
    # silently overwrites the first mask's pixels.
    response = {
        "rle": [
            json.dumps({"size": [2, 2], "counts": [0, 4]}),
            json.dumps({"size": [2, 2], "counts": [0, 1, 3]}),
        ],
        "metadata": [{"index": 1, "score": 0.91, "box": [0.5, 0.5, 1.0, 1.0]}],
        "scores": [0.91, 0.82],
        "boxes": [[0.5, 0.5, 1.0, 1.0], [0.25, 0.25, 0.5, 0.5]],
    }

    with pytest.raises(sam3.ProviderError) as caught:
        sam3.SAM3Adapter(
            subscriber=lambda endpoint, *, arguments: response
        ).segment(
            canonical_path,
            prompt="pallet",
            frame_id="frame_0001",
            output_dir=tmp_path / "masks",
        )

    assert caught.value.provider == "fal"
    assert caught.value.operation == "sam3.normalize"
    assert "duplicate SAM 3 mask instance id: 1" in caught.value.original_message


def test_label_prompt_registry_covers_vocabulary_with_canonical_first():
    sam3 = importlib.import_module("ehs_spatial.providers.sam3")

    assert set(sam3.LABEL_PROMPTS) == set(sam3.PROMPT_VOCABULARY)
    for label, prompts in sam3.LABEL_PROMPTS.items():
        assert prompts[0] == label


def test_adapter_maps_synonym_prompt_to_canonical_label(tmp_path):
    sam3 = importlib.import_module("ehs_spatial.providers.sam3")
    canonical_path = tmp_path / "canonical.png"
    Image.new("RGB", (4, 4), (10, 20, 30)).save(canonical_path)
    response = {
        "rle": [json.dumps({"size": [2, 2], "counts": [0, 4]})],
        "metadata": [{"index": 0, "score": 0.9, "box": [0.5, 0.5, 1.0, 1.0]}],
        "scores": [0.9],
        "boxes": [[0.5, 0.5, 1.0, 1.0]],
    }
    seen = {}

    def subscriber(endpoint, *, arguments):
        seen["prompt"] = arguments["prompt"]
        return response

    [observation] = sam3.SAM3Adapter(subscriber=subscriber).segment(
        canonical_path,
        prompt="barrier",
        label="safety fence",
        frame_id="frame_0001",
        output_dir=tmp_path / "masks",
    )

    assert seen["prompt"] == "barrier"
    assert observation.label == "safety fence"
    assert observation.source_prompt == "barrier"
    assert observation.observation_id == "frame_0001:safety_fence:0"
    assert observation.mask_path.endswith("frame_0001_safety_fence_0.png")


def test_adapter_rejects_prompt_not_registered_for_label(tmp_path):
    sam3 = importlib.import_module("ehs_spatial.providers.sam3")
    canonical_path = tmp_path / "canonical.png"
    Image.new("RGB", (4, 4), (10, 20, 30)).save(canonical_path)

    with pytest.raises(ValueError, match="unsupported SAM 3 prompt for label"):
        sam3.SAM3Adapter(subscriber=lambda endpoint, *, arguments: {}).segment(
            canonical_path,
            prompt="barrier",
            label="pallet",
            frame_id="frame_0001",
            output_dir=tmp_path / "masks",
        )


def test_adapter_retries_transient_billing_lock(tmp_path, monkeypatch):
    """fal's billing gate flaps: 'User is locked' must be retried, not fatal."""
    sam3 = importlib.import_module("ehs_spatial.providers.sam3")
    monkeypatch.setattr(sam3.time, "sleep", lambda _s: None)
    canonical_path = tmp_path / "canonical.png"
    Image.new("RGB", (2, 2)).save(canonical_path)

    calls = {"n": 0}

    def flapping_subscriber(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise RuntimeError("User is locked. Reason: TOP_UP.")
        return {"rle": [], "scores": [], "boxes": []}

    observations = sam3.SAM3Adapter(subscriber=flapping_subscriber).segment(
        canonical_path,
        prompt="pallet",
        frame_id="frame_0001",
        output_dir=tmp_path / "masks",
    )
    assert observations == []
    assert calls["n"] == 3


def test_adapter_does_not_retry_non_transient_failure(tmp_path, monkeypatch):
    sam3 = importlib.import_module("ehs_spatial.providers.sam3")
    monkeypatch.setattr(sam3.time, "sleep", lambda _s: None)
    canonical_path = tmp_path / "canonical.png"
    Image.new("RGB", (2, 2)).save(canonical_path)

    calls = {"n": 0}

    def failing_subscriber(*args, **kwargs):
        calls["n"] += 1
        raise RuntimeError("queue unavailable")

    with pytest.raises(sam3.ProviderError):
        sam3.SAM3Adapter(subscriber=failing_subscriber).segment(
            canonical_path,
            prompt="pallet",
            frame_id="frame_0001",
            output_dir=tmp_path / "masks",
        )
    assert calls["n"] == 1
