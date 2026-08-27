"""The refine module measures a boxed region through injected SAM."""
import json
from pathlib import Path

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
