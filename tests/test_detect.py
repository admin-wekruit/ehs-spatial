"""Detection-layer tests: taxonomy walk, box validation, crop-check gate,
retry path, SAM wiring — all with injected fakes, zero live calls."""

import json
from pathlib import Path

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
        if op == "detect.sweep":
            return self.sweep, None
        if op == "detect.more":
            from ehs_spatial.detect import ExtraSweep

            return ExtraSweep(boxes=[]), None
        if op == "detect.split":
            from ehs_spatial.detect import SectionSplit

            return SectionSplit(boxes=[]), None
        from ehs_spatial.agent import LocatedObject

        return (
            LocatedObject(
                found=False, label_en="", box_2d=(0, 0, 1, 1), rationale="none"
            ),
            None,
        )


class Check:
    def __init__(self, matches, reason=""):
        self.matches = matches
        self.reason = reason


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
        verifier=lambda crop, text: Check(True),
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


def test_crop_check_rejects_bad_boxes(run_dir):
    sweep = _sweep(
        ("b1", (100, 50, 600, 120)),
        not_visible=[t.item_id for t in TAXONOMY if t.item_id != "b1"],
    )
    envelope = detect_devices(
        "r1",
        runs_root=run_dir,
        adapter=FakeAdapter(sweep),
        subscriber=_subscriber,
        verifier=lambda crop, text: Check(False, "wrong object"),
    )
    assert envelope["detections"] == [] or all(
        "rle" not in d for d in envelope["detections"]
    )
    assert envelope["rejected"][0]["item_id"] == "b1"


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
        verifier=lambda crop, text: Check(True),
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
        verifier=lambda crop, text: Check(True),
    )
    assert (Path(run_dir) / "r1" / "detection" / "overlay.png").exists()
    envelope = json.loads(
        (Path(run_dir) / "r1" / "detection" / "detections.json").read_text()
    )
    assert envelope["detections"][0]["number"] == 1
