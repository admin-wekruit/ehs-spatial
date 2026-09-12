"""One end-to-end registry check: retain small objects and exact source identity."""
import base64
import hashlib
import json

import numpy as np
import pytest
from PIL import Image

from ehs_spatial.object_evidence import build_object_evidence, load_candidate_mask, write_object_evidence, _local
from ehs_spatial.providers.sam3 import encode_coco_rle


def test_registry_preserves_instances_and_source_pixel_contract(tmp_path):
    mounted = tmp_path / "volume-root" / "mounted-run"
    mounted.mkdir(parents=True)
    (mounted / "mask.png").write_bytes(b"path-contract")
    assert _local(mounted, "runs/mounted-run/mask.png") == mounted / "mask.png"
    with pytest.raises(ValueError, match="outside this run"):
        _local(mounted, "runs/other-run/mask.png")
    run = tmp_path / "runs" / "new-photo"
    for folder in ["input", "geometry/frames/frame_0001", "geometry/provider", "geometry/masks", "inventory/sam", "detection", "refinements"]:
        (run / folder).mkdir(parents=True, exist_ok=True)
    def save(name, value):
        (run / name).write_text(json.dumps(value))
    Image.new("RGB", (100, 80)).save(run / "input/image_01.png")
    geom = run / "geometry/frames/frame_0001"
    Image.new("RGB", (40, 40)).save(geom / "canonical.png")
    yy, xx = np.indices((40, 40))
    points = np.stack([(xx-20)/100, (yy-20)/100, np.ones_like(xx)], -1)
    for name, values in {"pts3d": points, "valid_mask": np.ones((40, 40), bool), "conf": np.ones((40, 40)),
                         "intrinsics": [[100., 0, 20], [0, 100, 20], [0, 0, 1]], "camera_to_world": np.eye(4)}.items():
        np.save(geom / f"{name}.npy", values)
    alpha = np.zeros((40, 40), np.uint8); alpha[:, 5:35] = 1
    save("geometry/provider/frame_0001.json", {"original_image": {"width": 100, "height": 80},
         "image": {"shape": [40, 40, 3]}, "alpha_mask": {"shape": [40, 40], "dtype": "uint8", "data": base64.b64encode(alpha.tobytes()).decode()}})
    left = np.zeros((40, 40), bool); left[8:18, 8:18] = True
    right = np.zeros_like(left); right[8:18, 23:33] = True
    observations = []
    for index, mask in enumerate([left, right]):
        path = run / f"geometry/masks/lamp{index}.png"
        Image.fromarray(mask.astype(np.uint8)*255).save(path)
        observations.append({"observation_id": f"lamp:{index}", "frame_id": "frame_0001", "label": "lamp", "mask_path": str(path)})
    save("observations.json", observations)
    save("inventory/sam/frame_0001__lamp.json", {"rle": [encode_coco_rle(left)], "scores": [.9]})
    save("inventory/inventory.json", {"phrases": ["lamp"], "objects": [{"inv": 0, "label": "lamp", "frame": "frame_0001", "instance": 0}]})
    save("scene.json", {"floor_plane": [0, -1, 0, 1.5], "scale_factor": 1.0})
    button = np.zeros((80, 100), bool); button[10:17, 20:27] = True
    save("detection/detections.json", {"detections": [
        {"item_id": "b1", "label": "stop button", "box": [20, 10, 27, 17], "rle": encode_coco_rle(button)},
        {"item_id": "e1", "label": "missing lamp", "box": [1, 2, 3, 4]}]})
    save("refinements/button-hash.json", {"rle": [encode_coco_rle(button)], "scores": [1.]})
    save("refinements.json", [{"label": "stop button", "box": [20, 10, 27, 17], "refine_slug": "button-hash", "frame_id": "frame_0001"}])
    save("inventory/unresolved.json", [
        {"phrase": "stop button", "frame": "frame_0001", "refine_slug": "button-hash", "stage": "mask", "reason": "measurement mask under 50 pixels"},
        {"phrase": "indicator light", "reason": "Localized but no accepted instance mask"}])
    before = {str(p.relative_to(run)): hashlib.sha256(p.read_bytes()).hexdigest() for p in run.rglob("*") if p.is_file()}
    registry = build_object_evidence(run)
    assert len(registry["candidates"]) == 5  # Two lamps, one button, one maskless detection, one unresolved claim.
    unresolved = next(c for c in registry["candidates"] if c["kind"] == "unresolved_claim")
    assert all(unresolved[key]["reason"] == "Localized but no accepted instance mask"
               for key in ["mask", "geometry", "generation"])
    lamps = [c for c in registry["candidates"] if c["label"] == "lamp"]
    assert len(lamps) == 2 and len({c["id"] for c in lamps}) == 2
    measured = next(c for c in lamps if c["inventory_indices"])
    assert len(measured["source_refs"]) == 2 and measured["geometry"]["status"] == "measured"
    measurement = measured["measurements"]
    assert measurement["status"] == "available"
    assert np.isclose(measurement["dimensions_native"]["height"], .09)
    assert measurement["scale"]["status"] == "uncalibrated"
    assert measurement["scale"]["m_per_native"] is None
    assert measurement["source"]["scene_sha256"] == before["scene.json"]
    assert measurement["source"]["pointmap_sha256"] == before["geometry/frames/frame_0001/pts3d.npy"]
    assert registry["measurement_reference"]["scene_sha256"] == before["scene.json"]
    stop = next(c for c in registry["candidates"] if c["label"] == "stop button")
    assert len(stop["source_refs"]) == 2 and stop["mask"]["bbox"] == [20, 10, 27, 17]
    assert stop["mask"]["shape_hw"] == [80, 100] and stop["mask"]["resolution"] == "original"
    assert np.array_equal(load_candidate_mask(run, stop), button)
    assert stop["measurement_rejections"][0]["stage"] == "mask"
    assert all(c["generation"]["status"] != "ready" for c in registry["candidates"])
    assert next(c for c in registry["candidates"] if c["label"] == "missing lamp")["generation"]["status"] == "blocked"
    frame = registry["frames"][0]
    assert (frame["width"], frame["height"], frame["canonical_width"], frame["canonical_height"]) == (100, 80, 40, 40)
    assert np.allclose(frame["input_to_canonical_pixel_centres"], [[.3, 0, 4.65], [0, .5, -.25], [0, 0, 1]])
    assert before == {str(p.relative_to(run)): hashlib.sha256(p.read_bytes()).hexdigest() for p in run.rglob("*") if p.is_file()}
    assert str(tmp_path) not in json.dumps(registry)
    assert json.loads(write_object_evidence(run).read_text())["candidates"] == registry["candidates"]
    save("scene.json", {"floor_plane": [0, -1, 0, 1.6], "scale_factor": 1.0})
    updated = build_object_evidence(run)
    assert updated["source_sha256"]["scene.json"] != registry["source_sha256"]["scene.json"]
    assert [c["id"] for c in updated["candidates"]] == [c["id"] for c in registry["candidates"]]
    # A changed RLE cannot be silently reused using the previous candidate identity.
    save("detection/detections.json", {"detections": []})
    with pytest.raises(ValueError, match="changed"):
        load_candidate_mask(run, stop)
