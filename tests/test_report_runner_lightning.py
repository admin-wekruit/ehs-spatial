"""The fixes the first Lightning one-shot run and its independent check asked for (2026-09-28). CPU only, no Modal call:
synthetic inputs, plus the cached Lightning runs read-only where they exist."""
import json
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "scripts"), str(REPO / "modal_apps")]
from report_runner import decide, stages  # noqa: E402
from test_report_runner_stages import RESEARCH, SITES, build, ctx, flag  # noqa: E402

RUNS = stages.art() / "runs"


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


# ---------------------------------------------------------------- 1. the inferred floor: never the untested convex outline
def test_floor_without_a_dense_map_is_withheld_with_its_reason(tmp_path):
    out = tmp_path / "floor"
    done = subprocess.run([sys.executable, str(REPO / "scripts/infer_room_floor.py"), "--fused", str(tmp_path / "no-such-run"), "--output", str(out)],
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr[-2000:]
    record = json.loads((out / "inferred-floor.json").read_text())
    assert record["kind"] == "inferred_floor_withheld" and "dense map" in record["reason"]
    assert not (out / "inferred-floor.glb").exists(), "the convex outline needs --legacy-convex"


def test_the_floor_decision_keeps_only_a_dense_validated_floor_on_a_verified_plane(tmp_path):
    model = {"kind": "inferred_floor_model", "basis": "b", "area_m2": 463.2}
    lens = lambda status: write(tmp_path / status / "lens.json", {"value": {"scale_status": status}})
    convex = write(tmp_path / "convex/inferred-floor.json", model)
    dense = write(tmp_path / "dense/inferred-floor.json", {**model, "dense": {"validation": {"closeM": .5}}})
    withheld = write(tmp_path / "withheld/inferred-floor.json", {"kind": "inferred_floor_withheld", "reason": "no validated dense map"})
    assert decide.inferred_floor(convex)["value"] is False, "the Lightning floor: a convex outline, never tested"
    assert decide.inferred_floor(dense, lens("intrinsics_uncertain"))["value"] is False, "Lightning's plane failed the lens gate (0.835 < 0.90)"
    assert decide.inferred_floor(withheld)["value"] is False and "dense" in decide.inferred_floor(withheld)["evidence"]["withheld"]
    assert decide.inferred_floor(dense, lens("assumed_camera_height"))["value"] is True and decide.inferred_floor(dense)["value"] is True


def test_the_runner_never_asks_for_the_convex_outline():
    values = {**SITES["walmart"][2], "dense_gate": {"use": None, "dir": None}}
    by = build("walmart", decisions=values)
    assert flag(by["floor_infer"], "--dense") is None and flag(by["floor_infer"], "--legacy-convex") is None
    assert "gate=@lens_gate:decision" in by["inferred_floor"].commands[0] and by["inferred_floor"].inputs["lens_gate"] == ("lens_gate", ())
    assert by["floor_infer"].version == 2
    delivered = build("walmart", profile=types.SimpleNamespace(**{**vars(RESEARCH), "name": "delivered", "omit": ("static_filter", "lens_gate")}))
    assert delivered["floor_infer"].version == 1 and "lens_gate" not in delivered["inferred_floor"].inputs, \
        "the delivered profile keys its adopted runs under the versions they were adopted with"


# ---------------------------------------------------------------- 2. no metric wording without a measured scale
def test_a_report_without_metres_states_no_metric_figure(tmp_path):
    import import_video_scene as importer
    checked = {"generator": "SAM 3D Objects", "fitResidualNative": .00248, "fitResidualCm": 2.08, "observedShare": .128}
    floor = {"basis": "floor plane (consensus over views) extended under and between things seen standing on it", "observed_share_of_this_floor": .3,
             "area_m2": 463.2}
    leaked = {"entities": [{"representations": [{"modelBasis": importer.model_basis(checked, False)}, {"modelBasis": importer.floor_basis(floor, False)}]}]}
    assert [p for p, _ in importer.metric_claims(leaked)] == [".entities[0].representations[0].modelBasis", ".entities[0].representations[1].modelBasis"], \
        "the Lightning import quoted 'residual 2.1 cm' and '463.2 m2' although its scale was uncalibrated"
    native = {"entities": [{"representations": [{"modelBasis": importer.model_basis(checked, True)}, {"modelBasis": importer.floor_basis(floor, True)}]}],
              "annotations": [{"limitations": ["sizes and distances are in native units, not metres."]}]}
    assert importer.metric_claims(native) == [] and "0.00248 native units" in native["entities"][0]["representations"][0]["modelBasis"]
    assert importer.unmeasured({"events": [{"near": ["about 2 meters from the cart"], "t0": 1.5}]}) == {"events": [{"near": ["about [distance not measured] from the cart"], "t0": 1.5}]}
    with pytest.raises(ValueError, match="metric figures"):
        importer.check_no_metres(leaked, uncalibrated=True)
    importer.check_no_metres(leaked, uncalibrated=False)  # a scale that claims metres may say so
    assert "check_no_metres(document, uncalibrated)" in (REPO / "scripts/import_video_scene.py").read_text(), "build_document runs the guard"

    folder = tmp_path / "floor"
    write(folder / "inferred-floor.json", {"dense": {"point_spacing_native": .002, "points": 10, "region": "floor seen ... gaps up to 0.5 m between them, on a 0.05 m grid",
                                                     "colour_rule": "median colour of the nearest floor seen within 0.5 m"}})
    (folder / "inferred-floor-points.glb").write_bytes(b"glb")
    shell = types.SimpleNamespace(bounds=np.zeros((2, 3)))
    points = lambda uncalibrated: importer.inferred_floor_points(folder, lambda *a: "asset", lambda *a: "id", {}, "source", shell, uncalibrated)
    assert importer.metric_claims(points(False)) and importer.metric_claims(points(True)) == []


def test_the_scale_sentence_quotes_only_a_measured_disagreement():
    import import_video_scene as importer
    scale = {"scale_status": "assumed_camera_height", "metres_per_native_unit": 8.38, "camera_height_native_median": .19,
             "model_estimated_metres_per_native_unit": 7., "height_anchor_vs_model_estimate": None}
    assert "disagrees" not in importer.scale_limitation(scale, device=False) and "not measured" in importer.scale_limitation(scale, device=False)


# ---------------------------------------------------------------- 3. any source size: the clip's own source-full.json
def clip_of(tmp_path, wh, raster_xywh, frames=3):
    """A clip folder with source-full.json and a source-full video of `frames` grey frames at `wh`."""
    import cv2
    clip = tmp_path / f"clip-{wh[0]}x{wh[1]}"
    write(clip / "source-full.json", {"video": "source-full.avi", "width": wh[0], "height": wh[1], "raster_wh": [640, 480], "raster_in_video_xywh": list(raster_xywh)})
    writer = cv2.VideoWriter(str(clip / "source-full.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 30, tuple(wh))
    for _ in range(frames):
        writer.write(np.full((wh[1], wh[0], 3), 90, np.uint8))
    writer.release()
    return clip


@pytest.fixture
def splat(monkeypatch):
    import splat_train
    for name in ("W", "H", "SIDE", "TOP", "ZOOM", "MESH", "FILL"):  # use_frame and the test set these; restored afterwards
        monkeypatch.setattr(splat_train, name, getattr(splat_train, name))
    return splat_train


def test_splats_train_on_a_640x480_source(tmp_path, splat):
    """Lightning's clip is a 640x480 re-encode with no crop: the frame, K_video, masks and seeds follow its source-full.json (the first
    run stopped on IndexError 484 vs 480 in seed_points, which assumed ME340's 1280x720)."""
    import trimesh
    clip = clip_of(tmp_path, (640, 480), (0, 0, 640, 480))
    splat.use_frame(splat.frame_of(clip))
    assert (splat.W, splat.H, splat.ZOOM) == (640, 480, 1.) and splat.frame_now()["SIDE"] == 0
    k_clip = [500., 500., 319.5, 239.5]
    assert np.allclose(splat.full_k(k_clip), [[500, 0, 319.5], [0, 500, 239.5], [0, 0, 1]]), "no crop, no zoom: the clip's own K"
    person = np.zeros((480, 640), bool)
    person[100:200, 0:50] = True
    excluded = splat.excluded_full(person, (0, 0, 0, 0), dilate=0)
    assert excluded.shape == (480, 640) and (excluded == person).all(), "the mask lands on the same pixels"
    wall = trimesh.creation.box((2., 1.2, .01))  # a wall 2 in front of the camera that leaves the frame's edges empty
    wall.apply_translation([0, 0, 2.])
    wall.export(tmp_path / "mesh.ply")
    trimesh.Scene({"room": wall}).export(tmp_path / "fill.glb")
    splat.MESH, splat.FILL = tmp_path / "mesh.ply", tmp_path / "fill.glb"
    c2w = np.tile(np.eye(4), (3, 1, 1))
    xyz, rgb, seeds = splat.seed_points(clip / "source-full.avi", splat.full_k(k_clip), c2w, [0, 2], lambda i: np.zeros((480, 640), bool))
    assert seeds["frame_seeds"] > 0 and len(xyz) == len(rgb), "rows beside the wall are seeded from the 640x480 frame"
    me340 = clip_of(tmp_path, (1280, 720), (160, 0, 960, 720), frames=1)
    splat.use_frame(splat.frame_of(me340))
    assert np.allclose(splat.full_k(k_clip), [[750, 0, 1.5 * 319.5 + 160.25], [0, 750, 1.5 * 239.5 + .25], [0, 0, 1]]), "ME340 unchanged"


def test_the_dense_map_follows_a_640x480_source(tmp_path):
    """LingBot's crop of a 4:3 source is 518x392; the cross-view check's masks must come from the clip's own frame (the first run
    drew them on ME340's 1280x720 and failed: (392,518) vs (294,518))."""
    import lingbot_dense_map as ldm
    from build_lingbot_replay import resize_mask
    clip = clip_of(tmp_path, (640, 480), (0, 0, 640, 480), frames=1)
    geometry = ldm.frame_geometry(clip)
    assert geometry == {"full_wh": [640, 480], "x0": 0., "y0": 0., "scale": 1.}
    assert ldm.frame_geometry(clip_of(tmp_path, (1280, 720), (160, 0, 960, 720), frames=1)) == ldm.ME340_GEOMETRY, "ME340 unchanged"
    analysis, grid = ldm.flow_frame(geometry, tmp_path)
    lingbot_raster = (392, 518)  # lingbot_room's official crop mode: width 518, height round(480 * 518 / 640 / 14) * 14
    assert resize_mask(np.zeros((analysis["height"], analysis["width"]), bool), (analysis["height"], analysis["width"])).shape == lingbot_raster
    fx, fy = ldm.sample_grid(lingbot_raster, 3, geometry["full_wh"])
    assert fx.min() > -1 and fx.max() < 640 and fy.min() > -1 and fy.max() < 480, "samples stay inside the 640x480 frame"
    import cv2
    person = np.zeros((480, 640), np.uint8)
    person[200:260, 100:160] = 255
    mask = ldm.moving_lookup([cv2.imencode(".png", person)[1].tobytes()], *grid, geometry)
    assert mask.shape == (480, 640) and mask[230, 130] and not mask[230, 300] and not mask[100, 130], "the mask lands where the person is"
    k = ldm.full_k([500., 500., 319.5, 239.5], geometry)
    assert np.allclose(k, [[500, 0, 319.5], [0, 500, 239.5], [0, 0, 1]])
