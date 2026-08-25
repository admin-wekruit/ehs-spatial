"""Evidence artifacts built from a small synthetic scene: no network, no
cached provider data — a flat floor 1.5 m below an identity camera (OpenCV
+Y down) with a ~1 m tall box standing on it, so the floor fit, the object
measurements and the overlays are all checkable by construction."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ehs_spatial.contracts import GeometryFrame, Observation2D
from ehs_spatial.viewer import build_viewer_html, render_frame_overlays

SIDE = 48
OBJECT_ROWS = slice(8, 24)
OBJECT_COLS = slice(20, 36)


def write_synthetic_frame(frame_dir: Path, frame_id: str) -> GeometryFrame:
    """Write one frame whose reconstruction has a fittable floor plane
    (y = +1.5, i.e. 1.5 m below the camera) and a ~1 m box at ~2 m depth."""
    frame_dir.mkdir(parents=True, exist_ok=True)
    rows, cols = np.mgrid[0:SIDE, 0:SIDE]
    points = np.zeros((SIDE, SIDE, 3), dtype=np.float32)
    points[..., 0] = (cols - 24) / 12.0
    points[..., 1] = 1.5
    points[..., 2] = 1.0 + rows / 12.0
    box = np.zeros((SIDE, SIDE), dtype=bool)
    box[OBJECT_ROWS, OBJECT_COLS] = True
    points[..., 0][box] = 0.5 + (cols[box] - 20) * 0.02
    points[..., 1][box] = 1.5 - (23 - rows[box]) / 15.0
    points[..., 2][box] = 2.0 + (cols[box] - 20) * 0.005

    np.save(frame_dir / "pts3d.npy", points)
    np.save(frame_dir / "conf.npy", np.ones((SIDE, SIDE), dtype=np.float32))
    np.save(frame_dir / "valid_mask.npy", np.ones((SIDE, SIDE), dtype=bool))
    np.save(frame_dir / "camera_to_world.npy", np.eye(4))
    np.save(frame_dir / "intrinsics.npy", np.eye(3))
    canonical = np.zeros((SIDE, SIDE, 3), dtype=np.uint8)
    canonical[..., 0] = rows * 5
    canonical[..., 1] = cols * 5
    canonical[..., 2] = 128
    Image.fromarray(canonical).save(frame_dir / "canonical.png")

    return GeometryFrame(
        frame_id=frame_id,
        canonical_image_path=str(frame_dir / "canonical.png"),
        pts3d_path=str(frame_dir / "pts3d.npy"),
        conf_path=str(frame_dir / "conf.npy"),
        valid_mask_path=str(frame_dir / "valid_mask.npy"),
        camera_to_world=np.eye(4).tolist(),
        intrinsics=np.eye(3).tolist(),
    )


def write_box_mask(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = np.zeros((SIDE, SIDE), dtype=np.uint8)
    mask[OBJECT_ROWS, OBJECT_COLS] = 255
    Image.fromarray(mask).save(path)


def _synthetic_run(
    tmp_path: Path,
) -> tuple[Path, list[GeometryFrame], list[Observation2D]]:
    run = tmp_path / "runs" / "synth"
    frame = write_synthetic_frame(run / "geometry" / "frames" / "frame_0001", "frame_0001")

    masks_dir = run / "geometry" / "masks" / "frame_0001"
    pallet_mask = masks_dir / "pallet.png"
    write_box_mask(pallet_mask)
    wall = np.zeros((SIDE, SIDE), dtype=np.uint8)
    wall[:, :8] = 255
    wall_mask = masks_dir / "wall.png"
    Image.fromarray(wall).save(wall_mask)

    observations = [
        Observation2D(
            observation_id="frame_0001:pallet:0",
            frame_id="frame_0001",
            label="pallet",
            instance_id="0",
            mask_path=str(pallet_mask),
            score=0.9,
            bbox=[20 / SIDE, 8 / SIDE, 36 / SIDE, 24 / SIDE],
            source_prompt="pallet",
        ),
        Observation2D(
            observation_id="frame_0001:wall:0",
            frame_id="frame_0001",
            label="wall",
            instance_id="0",
            mask_path=str(wall_mask),
            score=0.7,
            bbox=[0.0, 0.0, 8 / SIDE, 1.0],
            source_prompt="wall",
        ),
    ]
    (run / "observations.json").write_text(
        json.dumps([item.model_dump(mode="json") for item in observations])
    )
    return run, [frame], observations


def test_build_viewer_html_from_disk_measures_the_box_and_excludes_scenery(tmp_path):
    run, _, _ = _synthetic_run(tmp_path)

    summary = build_viewer_html(run)

    html_path = run / "viewer.html"
    assert summary["path"] == html_path
    assert html_path.is_file()
    html = html_path.read_text(encoding="utf-8")
    assert "__PAYLOAD__" not in html
    assert summary["points"] == SIDE * SIDE
    [pallet] = summary["objects"]
    assert pallet["label"] == "pallet"
    # The box is 1.0 m tall and its footprint sits ~2.1 m from the camera.
    assert 0.7 < pallet["height_m"] < 1.1
    assert 1.5 < pallet["camera_dist_m"] < 3.0
    assert pallet["points"] >= 40
    assert '"label":"pallet"' in html
    # "wall" is scene context, never a selectable object.
    assert '"label":"wall"' not in html


def test_build_viewer_html_accepts_in_memory_data_and_out_path(tmp_path):
    run, frames, observations = _synthetic_run(tmp_path)
    out = tmp_path / "elsewhere" / "viewer.html"

    summary = build_viewer_html(
        run, frames=frames, observations=observations, out_path=out
    )

    assert summary["path"] == out
    assert out.is_file()
    assert [obj["label"] for obj in summary["objects"]] == ["pallet"]


def test_build_viewer_html_mirrors_the_assessed_scene_scale(tmp_path, monkeypatch):
    """A run whose scale came from the auto anchor must rebuild the viewer
    under that scale, not a fabricated camera height (regression: real
    surveillance-camera run failed its MAD gate under the 1.5 m default)."""
    import ehs_spatial.viewer as viewer_module

    run, frames, observations = _synthetic_run(tmp_path)
    (run / "scene.json").write_text(json.dumps({"scale_factor": 2.75}))
    seen = {}
    real_build = viewer_module._build_geometry

    def spy(frames_arg, observations_arg, camera_height_m, **kwargs):
        seen["camera_height_m"] = camera_height_m
        seen["override"] = kwargs.get("scale_factor_override")
        return real_build(
            frames_arg, observations_arg, camera_height_m, **kwargs
        )

    monkeypatch.setattr(viewer_module, "_build_geometry", spy)

    build_viewer_html(run, frames=frames, observations=observations,
                      camera_height_m=None)
    assert seen == {"camera_height_m": None, "override": 2.75}

    # An explicit override wins over the scene.json fallback.
    build_viewer_html(run, frames=frames, observations=observations,
                      scale_factor_override=1.0)
    assert seen["override"] == 1.0


def test_build_viewer_html_raises_without_floor_transform(tmp_path):
    run = tmp_path / "runs" / "empty"
    run.mkdir(parents=True)

    with pytest.raises(ValueError, match="floor transform"):
        build_viewer_html(run, frames=[], observations=[])


def test_overlays_tint_masked_pixels_and_leave_the_rest_untouched(tmp_path):
    run, frames, observations = _synthetic_run(tmp_path)

    written = render_frame_overlays(frames, observations, run / "evidence")

    assert [path.name for path in written] == ["frame_0001_overlay.png"]
    with Image.open(written[0]) as image:
        overlay = np.asarray(image.convert("RGB"))
    with Image.open(frames[0].canonical_image_path) as image:
        base = np.asarray(image.convert("RGB"))
    assert overlay.shape == base.shape
    assert not np.array_equal(overlay, base)
    # Bottom half of the box is tinted (top edge may carry the caption).
    assert (overlay[16:24, 20:36] != base[16:24, 20:36]).any(axis=-1).all()
    # Far corner: outside both masks and any caption.
    assert np.array_equal(overlay[40:, 40:], base[40:, 40:])


def test_overlays_skip_unreadable_masks_instead_of_failing(tmp_path):
    run, frames, observations = _synthetic_run(tmp_path)
    broken = observations[0].model_copy(
        update={"mask_path": str(run / "missing.png")}
    )

    written = render_frame_overlays(frames, [broken], run / "evidence")

    with Image.open(written[0]) as image:
        overlay = np.asarray(image.convert("RGB"))
    with Image.open(frames[0].canonical_image_path) as image:
        base = np.asarray(image.convert("RGB"))
    assert np.array_equal(overlay, base)


def test_cli_shell_builds_viewer_for_a_cached_run(tmp_path, monkeypatch, capsys):
    run, _, _ = _synthetic_run(tmp_path)
    monkeypatch.chdir(tmp_path)
    spec = importlib.util.spec_from_file_location(
        "build_viewer_cli", Path(__file__).parents[1] / "scripts" / "build_viewer.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.main(["--run", "synth"]) == 0

    assert (run / "viewer.html").is_file()
    printed = capsys.readouterr().out
    assert "pallet" in printed
    assert "1 selectable objects" in printed
