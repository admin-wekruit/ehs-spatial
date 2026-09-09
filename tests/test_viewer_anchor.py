"""The 3D viewer opens anchored to the camera (PRODUCT_PLAN §2.2): the eye
sits where the phone was, the orbit pivot is the floor point 2.5 m ahead,
and the anchors are published in the HTML as JSON so this can be checked
without running WebGL."""

import json
import re
from pathlib import Path

import numpy as np
import pytest
from test_viewer import _synthetic_run

from ehs_spatial.geometry import _FloorTransform, _rotation_to_positive_z
from ehs_spatial.viewer import LOOK_AHEAD_M, build_viewer_html, camera_anchor

BOR1 = Path("runs") / "user-bor1-02"
_BLOB = re.compile(
    r'<script id="anchors" type="application/json">(.*?)</script>', re.S
)


def _anchors_in(html: str) -> list[dict]:
    match = _BLOB.search(html)
    assert match, "anchors blob missing from viewer.html"
    return json.loads(match.group(1))


def _floor_transform(scale: float) -> _FloorTransform:
    # OpenCV world: floor is the plane y = +1.5 (1.5 m below an identity
    # camera), so the floor normal pointing up is -Y.
    normal = np.array([0.0, -1.0, 0.0])
    return _FloorTransform(
        plane=(0.0, -1.0, 0.0, 1.5),
        scale_factor=scale,
        rotation=_rotation_to_positive_z(normal),
        origin=np.array([0.0, 1.5, 0.0]),
    )


def _check_pivot_rule(anchor: dict) -> None:
    position, forward, pivot = (
        np.asarray(anchor[k]) for k in ("position", "forward", "pivot")
    )
    assert np.isfinite(np.r_[position, forward, pivot]).all()
    assert pivot[2] == pytest.approx(0.0, abs=1e-6)
    assert np.linalg.norm(pivot[:2] - position[:2]) == pytest.approx(
        LOOK_AHEAD_M, abs=1e-3
    )
    heading = forward[:2] / np.linalg.norm(forward[:2])
    assert np.allclose((pivot[:2] - position[:2]) / LOOK_AHEAD_M, heading, atol=1e-3)


def test_camera_anchor_identity_camera_sits_above_floor_looking_level():
    anchor = camera_anchor(np.eye(4), _floor_transform(scale=2.0), "frame_0001")

    # Position is scaled (2 x 1.5 m above the floor); directions are not.
    assert anchor["frame_id"] == "frame_0001"
    assert anchor["position"] == pytest.approx([0.0, 0.0, 3.0], abs=1e-4)
    assert np.linalg.norm(anchor["forward"]) == pytest.approx(1.0, abs=1e-4)
    assert anchor["forward"][2] == pytest.approx(0.0, abs=1e-4)  # level gaze
    assert anchor["up"] == pytest.approx([0.0, 0.0, 1.0], abs=1e-4)  # floor normal
    _check_pivot_rule(anchor)


def test_camera_anchor_tilted_translated_camera_projects_heading_onto_floor():
    theta = np.radians(30.0)
    c2w = np.eye(4)
    # Pitch the OpenCV camera 30 degrees down (forward gains +Y, i.e. down).
    c2w[:3, :3] = [
        [1.0, 0.0, 0.0],
        [0.0, np.cos(theta), np.sin(theta)],
        [0.0, -np.sin(theta), np.cos(theta)],
    ]
    c2w[:3, 3] = [1.0, 0.2, 3.0]

    anchor = camera_anchor(c2w, _floor_transform(scale=1.0))

    assert anchor["position"][2] == pytest.approx(1.5 - 0.2, abs=1e-4)
    assert anchor["forward"][2] == pytest.approx(-np.sin(theta), abs=1e-4)
    assert anchor["up"][2] == pytest.approx(np.cos(theta), abs=1e-4)
    _check_pivot_rule(anchor)


def test_viewer_html_publishes_one_anchor_per_frame(tmp_path):
    run, frames, observations = _synthetic_run(tmp_path)

    summary = build_viewer_html(run, frames=frames, observations=observations)

    html = (run / "viewer.html").read_text(encoding="utf-8")
    anchors = _anchors_in(html)
    assert anchors == summary["anchors"]
    assert [a["frame_id"] for a in anchors] == ["frame_0001"]
    # Default camera height is 1.5 m; the synthetic floor is 1.5 m below it.
    assert anchors[0]["position"][2] == pytest.approx(1.5, abs=0.05)
    _check_pivot_rule(anchors[0])
    assert "__ANCHORS__" not in html


@pytest.mark.skipif(
    not (BOR1 / "geometry" / "frames").is_dir(), reason="BOR1 run not present"
)
def test_bor1_viewer_anchors_cover_every_frame(tmp_path):
    frame_dirs = sorted(
        p.name for p in (BOR1 / "geometry" / "frames").iterdir() if p.is_dir()
    )

    summary = build_viewer_html(BOR1, out_path=tmp_path / "viewer.html")

    anchors = _anchors_in((tmp_path / "viewer.html").read_text(encoding="utf-8"))
    assert [a["frame_id"] for a in anchors] == frame_dirs
    assert anchors == summary["anchors"]
    for anchor in anchors:
        _check_pivot_rule(anchor)
        # A hand-held phone: above the floor, roughly upright.
        assert 0.8 < anchor["position"][2] < 2.5
        assert anchor["up"][2] > 0.9
