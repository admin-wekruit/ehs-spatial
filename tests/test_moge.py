"""MoGe anchor confidence semantics: dispersion across frames is the only
evidence of quality, so a single frame must cap confidence instead of
reporting the vacuous 1.0 that zero spread would imply."""

from pathlib import Path

import numpy as np
import pytest

from ehs_spatial.contracts import GeometryFrame
from ehs_spatial.providers.moge import MoGeAnchorAdapter


def _frame(tmp_path: Path, frame_id: str, depth: float) -> GeometryFrame:
    frame_dir = tmp_path / "frames" / frame_id
    frame_dir.mkdir(parents=True)
    points = np.zeros((16, 16, 3), dtype=np.float32)
    points[..., 2] = depth
    np.save(frame_dir / "pts3d.npy", points)
    np.save(frame_dir / "valid_mask.npy", np.ones((16, 16), dtype=bool))
    np.save(frame_dir / "conf.npy", np.ones((16, 16), dtype=np.float32))
    image = frame_dir / "canonical.png"
    image.write_bytes(b"not-read-by-the-fake")
    return GeometryFrame(
        frame_id=frame_id,
        canonical_image_path=str(image),
        pts3d_path=str(frame_dir / "pts3d.npy"),
        valid_mask_path=str(frame_dir / "valid_mask.npy"),
        conf_path=str(frame_dir / "conf.npy"),
        camera_to_world=np.eye(4).tolist(),
        intrinsics=np.eye(3).tolist(),
    )


@pytest.fixture
def adapter(tmp_path, monkeypatch):
    adapter = MoGeAnchorAdapter(runner=lambda *a, **k: None)
    monkeypatch.setattr(
        adapter, "_moge_median_range", lambda image, cache: 5.0
    )
    return adapter


def test_single_frame_anchor_caps_confidence(tmp_path, adapter):
    frames = [_frame(tmp_path, "frame_0001", depth=2.0)]

    anchor = adapter.anchor_scale(frames, tmp_path / "geom")

    assert anchor is not None
    assert anchor.scale == pytest.approx(2.5)
    # Zero spread from one sample is not evidence of quality.
    assert anchor.confidence == 0.5


def test_consistent_multi_frame_anchor_earns_high_confidence(tmp_path, adapter):
    frames = [
        _frame(tmp_path, "frame_0001", depth=2.0),
        _frame(tmp_path, "frame_0002", depth=2.0),
    ]

    anchor = adapter.anchor_scale(frames, tmp_path / "geom")

    assert anchor is not None
    assert anchor.confidence > 0.9
