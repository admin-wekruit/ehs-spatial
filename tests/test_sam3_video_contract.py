"""CPU-only exact provenance/mask checks; importing this file never starts Modal."""

import copy
import asyncio
import json
import urllib.error

import numpy as np
import pytest

from ehs_spatial.providers.sam3 import decode_coco_rle
from modal_apps.sam3_video import check_access, encode_frame, validate_frames
from modal_apps.sam3_video_fal import drain_process


def test_existing_container_exec_drains_output_after_exit_and_streams_input():
    class Process:
        def __init__(self):
            self.stdin = self
            self.stdout = self
            self.stderr = self
            self.written = bytearray()
            self.eof = False
            self.exited = False

        async def wait(self):
            self.exited = True
            return 0

        async def __aiter__(self):
            await asyncio.sleep(0)
            assert self.exited
            yield '{"phase":"submitted"}\n'
            await asyncio.sleep(0)
            yield '{"phase":"provider_output_complete"}\n'

        async def read(self):
            await asyncio.sleep(0)
            return ""

        def write(self, data):
            assert len(data) <= 65536
            self.written.extend(data)

        def write_eof(self):
            self.eof = True

        async def drain(self):
            await asyncio.sleep(0)

    process, lines = Process(), []
    payload = b"x" * 700_000  # Exceeds macOS argv limits; stdin must retain every byte.
    assert asyncio.run(drain_process(process, payload, lines.append)) == (0, 0)
    assert process.written == payload and process.eof
    assert [json.loads(line)["phase"] for line in lines] == ["submitted", "provider_output_complete"]


def test_native_video_contract_preserves_masks_ids_boxes_empty_frames_and_source_indices():
    clip = {"width": 4, "height": 3, "frame_count": 2,
            "frame_timestamps_seconds": [0.0, 0.04]}
    masks = np.zeros((2, 3, 4), dtype=bool)
    masks[0, :2, 1:3] = True
    masks[1, 2, 3] = True
    original = {"frame_index": 0, "outputs": {
        "out_obj_ids": np.array([19, 4], dtype=np.int64),
        "out_probs": np.array([0.75, 0.8], dtype=np.float32),
        "out_boxes_xywh": np.array([[0.25, 0, 0.25, 1 / 3], [0.75, 2 / 3, 0, 0]], dtype=np.float32),
        "out_binary_masks": masks,
    }}
    frame = encode_frame(original, clip, source_start_frame=100)
    assert frame["frame_index"] == 0 and frame["source_frame_index"] == 100
    assert [item["track_id"] for item in frame["objects"]] == [19, 4]
    for i, item in enumerate(frame["objects"]):
        np.testing.assert_array_equal(decode_coco_rle(item["rle"]), masks[i])
        np.testing.assert_array_equal(item["box_xywh_normalized"], original["outputs"]["out_boxes_xywh"][i])
        assert item["score"] == float(original["outputs"]["out_probs"][i])
    empty = encode_frame({"frame_index": 1, "outputs": {
        "out_obj_ids": np.empty(0, dtype=np.int64), "out_probs": np.empty(0),
        "out_boxes_xywh": np.empty((0, 4)), "out_binary_masks": np.empty((0, 3, 4), dtype=bool),
    }}, clip, 100)
    assert empty["objects"] == [] and empty["source_frame_index"] == 101
    assert empty["timestamp_seconds"] == 0.04
    validate_frames([frame, empty], clip)
    with pytest.raises(ValueError, match="every input frame"):
        validate_frames([frame], clip)
    with pytest.raises(ValueError, match="every input frame"):
        validate_frames([frame, frame], clip)
    for key, bad in [("out_obj_ids", np.array([4, 4])),
                     ("out_binary_masks", masks.astype(np.uint8)),
                     ("out_probs", np.array([float("nan"), 0.8]))]:
        corrupted = copy.deepcopy(original)
        corrupted["outputs"][key] = bad
        with pytest.raises(ValueError):
            encode_frame(corrupted, clip)


def test_access_only_uses_head_and_redacts_auth_error_details(monkeypatch):
    seen = []

    class Opener:
        def open(self, request, timeout):
            seen.append(request)
            raise urllib.error.HTTPError(request.full_url, 403, "Forbidden", {
                "X-Error-Code": "GatedRepo",
                "X-Error-Message": "Not authorized: hf_fake_secret https://example.com/?signature=private",
            }, None)

    monkeypatch.setenv("HF_TOKEN", "hf_fake_secret")
    monkeypatch.setattr("urllib.request.build_opener", lambda handler: Opener())
    results = [check_access("3"), check_access("3.1")]
    assert [item["model_id"] for item in results] == ["facebook/sam3", "facebook/sam3.1"]
    assert all(item["http_status"] == 403 and item["error_code"] == "GatedRepo" for item in results)
    assert all(not item["accessible"] and not item["downloaded_weights"] for item in results)
    assert all(request.get_method() == "HEAD" for request in seen)
    serialized = json.dumps(results)
    assert "hf_fake_secret" not in serialized and "signature=private" not in serialized
