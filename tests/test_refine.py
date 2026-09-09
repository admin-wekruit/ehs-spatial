"""The refine module measures a boxed region through injected SAM."""
import base64
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from ehs_spatial.refine import RefineError, refine_region


def _fake_run(tmp_path: Path) -> Path:
    run = tmp_path / "runs" / "r1"
    (run / "input").mkdir(parents=True)
    (run / "geometry" / "frames" / "frame_0001").mkdir(parents=True)
    W, H = 64, 48
    Image.new("RGB", (W, H), (120, 120, 120)).save(run / "input" / "image_01.png")
    fd = run / "geometry" / "frames" / "frame_0001"
    Image.new("RGB", (W, H)).save(fd / "canonical.png")
    # synthetic floor scene: y-down camera 1.5 m above floor
    fx = 80.0
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    direction = np.stack(
        [(u - W / 2) / fx, (v - H / 2) / fx, np.ones_like(u, float)], axis=-1
    )
    t = np.where(direction[..., 1] > 1e-3, 1.5 / direction[..., 1], 8.0)
    pts = direction * t[..., None]
    np.save(fd / "pts3d.npy", pts.astype(np.float32))
    np.save(fd / "conf.npy", np.ones((H, W), np.float32))
    np.save(fd / "valid_mask.npy", np.ones((H, W), bool))
    np.save(fd / "camera_to_world.npy", np.eye(4, dtype=np.float32))
    np.save(fd / "intrinsics.npy",
            np.array([[fx, 0, W / 2], [0, fx, H / 2], [0, 0, 1]], np.float32))
    return run


def _rle(mask: np.ndarray) -> str:
    flat = mask.flatten()
    pairs, start = [], None
    for i, v in enumerate(flat):
        if v and start is None:
            start = i
        elif not v and start is not None:
            pairs += [start + 1, i - start]
            start = None
    if start is not None:
        pairs += [start + 1, len(flat) - start]
    return " ".join(map(str, pairs))


def test_refine_region_measures_box(tmp_path):
    run = _fake_run(tmp_path)
    W, H = 64, 48
    mask = np.zeros((H, W), bool)
    mask[10:40, 20:30] = True

    def subscriber(endpoint, *, arguments):
        assert "box_prompts" in arguments
        return {"rle": [_rle(mask)], "scores": [0.9]}

    result = refine_region(
        "r1", "safety fence", (18, 8, 32, 42),
        runs_root=tmp_path / "runs", subscriber=subscriber,
    )
    assert result["sam_score"] == 0.9
    assert result["points"] >= 60
    assert result["footprint_xy"]
    # cached: second call needs no subscriber
    again = refine_region(
        "r1", "safety fence", (18, 8, 32, 42), runs_root=tmp_path / "runs"
    )
    assert again["sam_score"] == 0.9


def test_refine_region_rejects_bad_box(tmp_path):
    run = _fake_run(tmp_path)
    with pytest.raises(RefineError):
        refine_region("r1", "x", (50, 10, 10, 40), runs_root=tmp_path / "runs")


def test_refine_region_rejects_wrong_grid_before_measurement(tmp_path, monkeypatch):
    from ehs_spatial.providers.sam3 import encode_coco_rle

    run = _fake_run(tmp_path)
    cache = run / "refinements" / "x_1_1_20_20.json"
    cache.parent.mkdir()
    cache.write_text(json.dumps({"rle": [encode_coco_rle(np.ones((24, 32), bool))]}))
    monkeypatch.setattr("ehs_spatial.refine.measure", lambda *a: pytest.fail("wrong-grid mask reached geometry"))
    with pytest.raises(RefineError, match="does not match input image"):
        refine_region("r1", "x", (1, 1, 20, 20), runs_root=tmp_path / "runs")
    assert not cache.with_suffix(".png").exists()


def test_measure_uses_padded_input_mask_on_native_points(tmp_path, monkeypatch):
    import ehs_spatial.refine as refine

    run = _fake_run(tmp_path)
    frame = run / "geometry" / "frames" / "frame_0001"
    y, x = np.indices((48, 80))
    points = np.stack((x / 10, y / 10, np.full_like(x, 2)), axis=-1)
    np.save(frame / "pts3d.npy", points)
    np.save(frame / "valid_mask.npy", np.ones((48, 80), bool))
    Image.new("RGB", (80, 48)).save(frame / "canonical.png")
    alpha = np.zeros((48, 80), np.float32)
    alpha[:, 8:72] = 1
    provider = run / "geometry" / "provider" / "frame_0001.json"
    provider.parent.mkdir()
    provider.write_text(json.dumps({
        "original_image": {"width": 64, "height": 48}, "image": {"shape": [48, 80, 3]},
        "alpha_mask": {"shape": list(alpha.shape), "dtype": str(alpha.dtype),
                       "data": base64.b64encode(alpha.tobytes()).decode("ascii")},
    }))
    monkeypatch.setattr(refine, "_build_geometry", lambda *a, **kw: SimpleNamespace(
        transform=SimpleNamespace(apply=lambda cloud: cloud)))
    mask = np.zeros((48, 64), bool)
    mask[8:32, 16:32] = True
    measured = refine.measure(run, mask, 1.5, None)
    assert measured["points"] == 384
    assert measured["centroid_xy"] == [3.15, 1.95]
    assert measured["extent_m"] == "1.50x2.30"
