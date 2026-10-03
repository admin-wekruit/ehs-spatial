"""Small detected devices remain evidence even when metric gates reject them."""
import json
import sys
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from scene_inventory import _ingest_refinements, _reconcile_enumeration
import scene_inventory
from ehs_spatial.providers.sam3 import encode_coco_rle


def test_detected_instances_survive_mask_and_footprint_gates(tmp_path):
    run = tmp_path / "run"
    for directory in ["input", "detection", "geometry", "inventory"]:
        (run / directory).mkdir(parents=True)
    Image.new("RGB", (64, 64)).save(run / "input/image_01.png")
    yy, xx = np.indices((64, 64))
    points = np.stack([1 + xx * .01, 2 + yy * .01, np.ones_like(xx)], -1)
    valid = np.ones((64, 64), bool)
    detections = []
    # The actual regression gates: 49 mask pixels, then 200 valid pixels with
    # a footprint too small to measure. A larger same-label sibling succeeds.
    cases = [("b1", "emergency stop button", [1, 1, 8, 8]),
             ("b2", "control button", [12, 1, 22, 21]),
             ("b2", "control button", [30, 30, 40, 50])]
    for item_id, label, box in cases:
        x1, y1, x2, y2 = box
        mask = np.zeros((64, 64), bool)
        mask[y1:y2, x1:x2] = True
        if box[0] == 12:
            points[mask, :2] = np.column_stack([1 + xx[mask] * .002, 2 + yy[mask] * .002])
        detections.append({"item_id": item_id, "label": label, "box": box,
                           "rle": encode_coco_rle(mask), "sam_score": .9})
    detections.append({"item_id": "e4", "label": "safety light", "box": [45, 1, 50, 9]})
    (run / "detection/detections.json").write_text(json.dumps({"detections": detections}))
    np.save(run / "geometry/points.npy", points)
    np.save(run / "geometry/valid.npy", valid)
    frame = SimpleNamespace(frame_id="frame_0001", camera_to_world=np.eye(4), pts3d_path=run / "geometry/points.npy",
                            valid_mask_path=run / "geometry/valid.npy")
    entries = []
    # None of these labels needs to be in the VLM's separate phrase list.
    unresolved = _reconcile_enumeration(run, frame, entries, [], live=False)
    _ingest_refinements(run, frame, SimpleNamespace(apply=lambda p: p), None, entries, unresolved)
    assert len(entries) == 1 and entries[0]["label"] == "control button"
    by_box = {tuple(record["box"]): record for record in unresolved}
    assert set(by_box) == {(1, 1, 8, 8), (12, 1, 22, 21), (45, 1, 50, 9)}
    assert by_box[(1, 1, 8, 8)]["stage"] == "mask"
    assert "49 pixels" in by_box[(1, 1, 8, 8)]["reason"]
    assert by_box[(12, 1, 22, 21)]["stage"] == "footprint"
    assert by_box[(45, 1, 50, 9)]["stage"] == "mask"
    assert all(record["source"] == "detection" and record["frame"] == "frame_0001" for record in unresolved)
    assert all("height_m" not in record and "footprint" not in record for record in unresolved)

    # A hashed reviewer cache for photo 2 must never ingest photo 1's boxes.
    Image.new("RGB", (64, 64), "red").save(run / "input/image_02.png")
    slug = "frame_0002__sourcehash__identityhash"
    mask = np.zeros((64, 64), bool)
    mask[1:8, 1:8] = True
    (run / "refinements" / f"{slug}.json").write_text(json.dumps({"rle": [encode_coco_rle(mask)], "frame_id": "frame_0002"}))
    rows = json.loads((run / "refinements.json").read_text())
    rows.append({"label": "new object", "box": [1, 1, 8, 8], "refine_slug": slug,
                 "frame_id": "frame_0002", "source": "reviewer"})
    (run / "refinements.json").write_text(json.dumps(rows))
    frame.frame_id = "frame_0002"
    unresolved2 = []
    _ingest_refinements(run, frame, SimpleNamespace(apply=lambda p: p), None, [], unresolved2)
    assert len(unresolved2) == 1 and unresolved2[0]["refine_slug"] == slug
    assert unresolved2[0]["frame"] == "frame_0002"


def test_text_sam_keeps_each_rejected_instance_before_refinement(tmp_path, monkeypatch):
    """The production main path retains both SAM instances of one label."""
    monkeypatch.chdir(tmp_path)
    run = tmp_path / "runs/test"
    run.mkdir(parents=True)
    (run / "observations.json").write_text("[]")
    yy, xx = np.indices((64, 64))
    points = np.stack([1 + xx * .002, 2 + yy * .002, np.ones_like(xx)], -1)
    np.save(run / "points.npy", points)
    np.save(run / "valid.npy", np.ones((64, 64), bool))
    frame = SimpleNamespace(frame_id="frame_0001", pts3d_path=run / "points.npy",
                            valid_mask_path=run / "valid.npy")
    masks = [np.zeros((64, 64), bool) for _ in range(2)]
    masks[0][1:8, 1:8] = True
    masks[1][12:32, 12:22] = True
    monkeypatch.setattr(scene_inventory, "_frames", lambda run: [frame])
    monkeypatch.setattr(scene_inventory, "_build_geometry", lambda *args, **kwargs:
                        SimpleNamespace(transform=SimpleNamespace(apply=lambda p: p)))
    monkeypatch.setattr(scene_inventory, "_enumerate_objects", lambda *args, **kwargs: ["button"])
    monkeypatch.setattr(scene_inventory, "_moge3_maps", lambda *args, **kwargs: None)
    monkeypatch.setattr(scene_inventory, "_segment", lambda *args, **kwargs:
                        {"rle": [encode_coco_rle(mask) for mask in masks]})
    captured = []
    class StopAfterMeasurement(Exception):
        pass
    def capture(run, frame, transform, moge, entries, unresolved):
        assert entries == []
        captured.extend(unresolved)
        raise StopAfterMeasurement
    monkeypatch.setattr(scene_inventory, "_ingest_refinements", capture)
    with pytest.raises(StopAfterMeasurement):
        scene_inventory.main(["--run", "test"])
    assert [(u["instance"], u["stage"]) for u in captured] == [(0, "mask"), (1, "footprint")]
    assert all(u["mask_path"] == "inventory/sam/frame_0001__button.json" for u in captured)
    assert all(u["source"] == "text-sam" and u["phrase"] == "button" for u in captured)


def test_inventory_segments_two_phrases_together(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run = tmp_path / "runs/test"
    run.mkdir(parents=True)
    (run / "observations.json").write_text("[]")
    np.save(run / "points.npy", np.ones((8, 8, 3)))
    np.save(run / "valid.npy", np.ones((8, 8), bool))
    frame = SimpleNamespace(frame_id="frame_0001", pts3d_path=run / "points.npy",
                            valid_mask_path=run / "valid.npy")
    monkeypatch.setattr(scene_inventory, "_frames", lambda _: [frame])
    monkeypatch.setattr(scene_inventory, "_build_geometry", lambda *a, **k:
                        SimpleNamespace(transform=SimpleNamespace(apply=lambda p: p)))
    monkeypatch.setattr(scene_inventory, "_enumerate_objects", lambda *a, **k: ["button", "fence"])
    monkeypatch.setattr(scene_inventory, "_moge3_maps", lambda *a, **k: None)
    rendezvous = Barrier(2, timeout=2)
    def segment(*args, **kwargs):
        rendezvous.wait()
        return None
    monkeypatch.setattr(scene_inventory, "_segment", segment)

    class StopAfterSegments(Exception):
        pass

    def stop(*args, **kwargs):
        raise StopAfterSegments
    monkeypatch.setattr(scene_inventory, "_reconcile_enumeration", stop)
    with pytest.raises(StopAfterSegments):
        scene_inventory.main(["--run", "test"])
