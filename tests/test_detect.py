"""Per-frame detection identities, bounded calls, and retained failures."""

import json
import base64
import hashlib
import io
from pathlib import Path
from threading import Barrier

import numpy as np
import pytest
from PIL import Image

from ehs_spatial.detect import DetectedBox, TaxonomySweep, detect_devices
from ehs_spatial.providers.sam3 import encode_coco_rle
from ehs_spatial.taxonomy import TAXONOMY


class FakeAdapter:
    """Stands in for GeminiAdapter: bulk sweep returns a canned envelope,
    single-item locate returns found=False (so retries stay quiet)."""

    def __init__(self, sweep: TaxonomySweep):
        self.sweep = sweep
        self.calls = []

    def _create(self, op, **kwargs):
        self.calls.append(op)
        return op

    def _parse(self, response, model, op):
        assert op == "detect.sweep", "unexpected extra semantic call"
        return self.sweep, None


@pytest.fixture
def run_dir(tmp_path):
    run = tmp_path / "runs" / "r1"
    (run / "input").mkdir(parents=True)
    Image.new("RGB", (200, 100), (90, 90, 90)).save(
        run / "input" / "image_01.png"
    )
    return tmp_path / "runs"


def _sweep(*boxes, not_visible=()):
    return TaxonomySweep(
        boxes=[DetectedBox(item_id=i, box_2d=b) for i, b in boxes],
        not_visible=list(not_visible),
    )


def _subscriber(endpoint, arguments):
    mask = np.zeros((100, 200), bool)
    box = arguments["box_prompts"][0]
    mask[box["y_min"] : box["y_max"], box["x_min"] : box["x_max"]] = True
    return {"rle": [encode_coco_rle(mask)], "scores": [0.9]}


def test_detect_walks_taxonomy_and_masks(run_dir):
    sweep = _sweep(
        ("a1", (100, 50, 600, 120)),
        ("d1", (700, 100, 950, 400)),
        not_visible=[t.item_id for t in TAXONOMY if t.item_id not in ("a1", "d1")],
    )
    envelope = detect_devices(
        "r1",
        runs_root=run_dir,
        adapter=FakeAdapter(sweep),
        subscriber=_subscriber,
    )
    found = [d for d in envelope["detections"] if "rle" in d]
    assert {d["item_id"] for d in found} == {"a1", "d1"}
    assert all(d["sam_score"] == 0.9 for d in found)
    # every non-optional item not found is reported missing — honest coverage
    expected_missing = {
        t.item_id
        for t in TAXONOMY
        if t.expect != "optional" and t.item_id not in ("a1", "d1")
    }
    assert {m["item_id"] for m in envelope["missing"]} == expected_missing
    # cached: second call must not re-detect
    adapter = FakeAdapter(sweep)
    detect_devices("r1", runs_root=run_dir, adapter=adapter)
    assert adapter.calls == []


def test_failed_sam_keeps_the_localized_instance(run_dir):
    envelope = detect_devices(
        "r1", runs_root=run_dir,
        adapter=FakeAdapter(_sweep(("b1", (100, 50, 600, 120)))),
        subscriber=lambda *a, **kw: {"rle": [], "scores": []},
    )
    detection = envelope["detections"][0]
    assert detection["label"] == "emergency stop button"
    assert detection["mask_status"] == "unavailable" and "no mask" in detection["sam_error"]
    assert detection["frame_id"] == "frame_0001"
    assert detection["semantic_verification"] == "single_sweep"
    assert detection["source_image_sha256"]
    assert envelope["last_execution"] == {"sweep_calls": 1, "sam_calls": 1}


def test_non_normalized_box_rejected(run_dir):
    sweep = _sweep(
        ("b1", (100, 50, 1600, 120)),  # >1000: convention drift
        not_visible=[t.item_id for t in TAXONOMY if t.item_id != "b1"],
    )
    envelope = detect_devices(
        "r1",
        runs_root=run_dir,
        adapter=FakeAdapter(sweep),
        subscriber=_subscriber,
    )
    assert envelope["rejected"][0]["reason"] == "bad box"


def test_overlay_written(run_dir):
    sweep = _sweep(
        ("c1", (100, 50, 600, 400)),
        not_visible=[t.item_id for t in TAXONOMY if t.item_id != "c1"],
    )
    detect_devices(
        "r1",
        runs_root=run_dir,
        adapter=FakeAdapter(sweep),
        subscriber=_subscriber,
    )
    assert (Path(run_dir) / "r1" / "detection" / "overlay_frame_0001.png").exists()
    envelope = json.loads(
        (Path(run_dir) / "r1" / "detection" / "detections.json").read_text()
    )
    assert envelope["detections"][0]["number"] == 1


def test_legacy_first_frame_cache_does_not_hide_new_photo_and_retries_cost_zero(run_dir):
    from ehs_spatial.object_evidence import build_object_evidence, load_candidate_mask

    run = run_dir / "r1"
    directory = run / "detection"
    directory.mkdir()
    mask1 = np.zeros((100, 200), bool)
    mask1[10:20, 10:20] = True
    old = {"run_id": "r1", "image_size": [200, 100], "detections": [
        {"item_id": "b1", "label": "emergency stop button", "box": [10, 10, 20, 20],
         "rle": encode_coco_rle(mask1), "sam_score": .9, "number": 1,
         "category": "B", "zh": "急停", "iso": ""}], "missing": [], "rejected": []}
    (directory / "detections.json").write_text(json.dumps(old))
    second = run / "input/image_02.png"
    Image.new("RGB", (120, 240), (200, 0, 0)).save(second)
    adapter = FakeAdapter(_sweep(("b1", (100, 100, 150, 150))))
    calls = []
    def subscriber(endpoint, *, arguments):
        payload = base64.b64decode(arguments["image_url"].split(",", 1)[1])
        assert payload == second.read_bytes()
        with Image.open(io.BytesIO(payload)) as image:
            width, height = image.size
        box = arguments["box_prompts"][0]
        mask = np.zeros((height, width), bool)
        mask[box["y_min"]:box["y_max"], box["x_min"]:box["x_max"]] = True
        calls.append((width, height))
        return {"rle": [encode_coco_rle(mask)], "scores": [.91]}
    result = detect_devices("r1", runs_root=run_dir, adapter=adapter, subscriber=subscriber)
    assert adapter.calls == ["detect.sweep"] and calls == [(120, 240)]
    assert len(result["frames"]) == 2 and len(result["detections"]) == 2
    first, found = result["detections"]
    assert first["source_binding"] == "legacy_first_frame_unverified" and first["source_image_sha256"] is None
    assert found["frame_id"] == "frame_0002" and found["box"] == [12, 24, 18, 36]
    assert found["source_image_sha256"] == hashlib.sha256(second.read_bytes()).hexdigest()
    assert result["last_execution"] == {"sweep_calls": 1, "sam_calls": 1}
    assert result["recall"] is None
    for frame in result["frames"]:
        with Image.open(run / frame["overlay_path"]) as image:
            assert list(image.size) == frame["image_size"]
    registry = build_object_evidence(run)
    assert len(registry["candidates"]) == 2
    native = next(c for c in registry["candidates"] if c["frame_id"] == "frame_0002")
    assert load_candidate_mask(run, native).shape == (240, 120)
    from types import SimpleNamespace
    from scripts.scene_inventory import _ingest_refinements
    (run / "inventory").mkdir()
    geometry = run / "geometry/frames/frame_0002"
    geometry.mkdir(parents=True)
    y, x = np.indices((240, 120))
    np.save(geometry / "pts3d.npy", np.stack([1+x*.01, 2+y*.01, np.ones_like(x)*2], -1))
    np.save(geometry / "valid_mask.npy", np.ones((240, 120), bool))
    frame = SimpleNamespace(frame_id="frame_0002", pts3d_path=geometry/"pts3d.npy",
                            valid_mask_path=geometry/"valid_mask.npy", camera_to_world=np.eye(4))
    entries, unresolved = [], []
    _ingest_refinements(run, frame, SimpleNamespace(apply=lambda p: p), None, entries, unresolved)
    row = json.loads((run / "refinements.json").read_text())[0]
    assert row["frame_id"] == "frame_0002" and row["source_image_sha256"] == found["source_image_sha256"]
    assert hashlib.sha256((run / "refinements" / f'{row["refine_slug"]}.json').read_bytes()).hexdigest() == row["mask_sha256"]
    rebuilt = build_object_evidence(run)
    native = next(c for c in rebuilt["candidates"] if c["frame_id"] == "frame_0002")
    assert native["mask"]["status"] == "available"
    again = detect_devices("r1", runs_root=run_dir, adapter=adapter, subscriber=subscriber)
    assert again["last_execution"] == {"sweep_calls": 0, "sam_calls": 0}
    assert len(adapter.calls) == 1 and len(calls) == 1
    Image.new("RGB", (120, 240), (0, 200, 0)).save(second)
    changed = detect_devices("r1", runs_root=run_dir, adapter=adapter, subscriber=subscriber)
    assert changed["last_execution"] == {"sweep_calls": 1, "sam_calls": 1}
    assert changed["detections"][1]["mask_path"] != found["mask_path"]
    assert len(adapter.calls) == 2 and len(calls) == 2


def test_cold_run_spends_one_sweep_per_frame(run_dir):
    run = run_dir / "r1"
    Image.new("RGB", (60, 80)).save(run / "input/image_02.png")
    adapter = FakeAdapter(_sweep())
    result = detect_devices("r1", runs_root=run_dir, adapter=adapter,
                            subscriber=lambda *a, **kw: pytest.fail("no detected boxes"))
    assert adapter.calls == ["detect.sweep", "detect.sweep"]
    assert result["last_execution"] == {"sweep_calls": 2, "sam_calls": 0}
    assert {m["frame_id"] for m in result["missing"]} == {"frame_0001", "frame_0002"}


def test_two_photos_sweep_together_but_keep_source_order(run_dir):
    Image.new("RGB", (60, 80)).save(run_dir / "r1/input/image_02.png")
    rendezvous = Barrier(2, timeout=2)

    class ConcurrentAdapter(FakeAdapter):
        def _create(self, op, **kwargs):
            rendezvous.wait()
            return super()._create(op, **kwargs)

    result = detect_devices("r1", runs_root=run_dir, adapter=ConcurrentAdapter(_sweep()),
                            subscriber=lambda *a, **kw: pytest.fail("no detected boxes"))
    assert [frame["frame_id"] for frame in result["frames"]] == ["frame_0001", "frame_0002"]
    assert result["last_execution"] == {"sweep_calls": 2, "sam_calls": 0}
