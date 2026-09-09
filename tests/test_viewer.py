"""Evidence artifacts built from a small synthetic scene: no network, no
cached provider data — a flat floor 1.5 m below an identity camera (OpenCV
+Y down) with a ~1 m tall box standing on it, so the floor fit, the object
measurements and the overlays are all checkable by construction."""

import importlib.util
import base64
import hashlib
import json
from pathlib import Path
import re
import subprocess

import numpy as np
import pytest
from PIL import Image

from ehs_spatial.contracts import GeometryFrame, Observation2D
from ehs_spatial.viewer import build_viewer_html, render_frame_overlays, inventory_lookup

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


def test_inventory_identity_uses_frame_instance_and_explicit_merges():
    by_instance, by_slug = inventory_lookup([
        {"label":"safety fence", "frame":"frame_0001", "instance":0, "merged_instances":[0,2]},
        {"label":"safety fence", "frame":"frame_0002", "instance":0},
        {"label":"safety fence", "refine_slug":"actual_box"},
        {"label":"safety fence", "frame":"frame_0002", "refine_slug":"actual_box"},
    ])
    assert by_instance == {("frame_0001","safety_fence",0):0,
                           ("frame_0001","safety_fence",2):0,
                           ("frame_0002","safety_fence",0):1}
    assert by_slug == {("frame_0001","actual_box"):2, ("frame_0002","actual_box"):3}


def test_viewer_masks_merge_only_explicit_inventory_members(tmp_path, monkeypatch):
    import ehs_spatial.viewer as viewer
    run, frames, _ = _synthetic_run(tmp_path)
    sam = run / "inventory" / "sam"
    sam.mkdir(parents=True)
    (sam.parent / "inventory.json").write_text(json.dumps({"objects":[
        {"label":"bollard", "frame":"frame_0001", "instance":0, "merged_instances":[0,2]}]}))
    (sam / "frame_0001__bollard.json").write_text(json.dumps({"rle":["first","rejected","last"]}))
    masks = {}
    for i, name in enumerate(["first","rejected","last"]):
        masks[name] = np.zeros((SIDE,SIDE),bool)
        masks[name][i*10:i*10+8,:8] = True
    monkeypatch.setattr(viewer,"decode_coco_rle",lambda rle,**_:masks[rle].copy())
    [(label, mask, inv)] = viewer._masks_for_frame(run,frames[0],(SIDE,SIDE),[])
    assert label == "bollard" and inv == 0
    assert np.array_equal(mask,masks["first"] | masks["last"])
    (sam.parent / "inventory.json").write_text('{"objects":[]}')
    assert viewer._masks_for_frame(run,frames[0],(SIDE,SIDE),[]) == []


def test_filtered_object_ids_are_dense_and_inventory_ids_remain_exact(tmp_path, monkeypatch):
    import ehs_spatial.viewer as viewer
    run, frames, _ = _synthetic_run(tmp_path)
    removed = np.zeros((SIDE,SIDE),bool); removed[24:44,:16] = True
    kept = np.zeros((SIDE,SIDE),bool); kept[OBJECT_ROWS,OBJECT_COLS] = True
    valid = np.ones((SIDE,SIDE),bool); valid[removed] = False
    np.save(frames[0].valid_mask_path, valid)
    (run / "inventory").mkdir()
    measured = {"label":"kept", "height_m":4.321,"size_m":"7.89 x 0.12","camera_dist_m":9.87,"tilt_deg":23}
    (run / "inventory" / "inventory.json").write_text(json.dumps({"objects":[measured]*13}))
    monkeypatch.setattr(viewer,"_masks_for_frame",lambda *args:[("removed",removed,8),("kept",kept,12)])
    summary = build_viewer_html(run)
    assert [o["id"] for o in summary["objects"]] == [1]
    assert summary["supported_inv"] == [12]
    assert summary["objects"][0]["height_m"] == 4.321
    assert summary["objects"][0]["size"] == "7.89 x 0.12 m"
    assert summary["objects"][0]["camera_dist_m"] == 9.87
    assert summary["objects"][0]["measurement_source"] == "inventory"
    payload = json.loads(re.search(r'const DATA = (.*);',summary["path"].read_text())[1])
    ids = np.frombuffer(base64.b64decode(payload["ids"]),dtype='<u2')
    assert set(ids) == {0,1}


def test_generated_viewer_javascript_executes_selection_protocol(tmp_path):
    from ehs_spatial.viewer import _VIEWER_TEMPLATE
    objects = [{"id":i+1,"inv":inv,"frame":"frame_0001","label":"same label",
                "height_m":1,"size":"1 x 1 m","camera_dist_m":2,"points":100,
                "tilt_deg":0,"color":[100,150,200]} for i,inv in enumerate([0,1,0,None])]
    payload = {"objects":objects,"supported_inv":[0,1],"inventory_count":3,
               "interactive_inv":[0,1,2],
               "unavailable":[{"inv":2,"label":"not observed","frame":"frame_0001","reason":"no points"}],
               "count":4,"origin":[0,0,0],"span":4,"run":"linked-check",
               "xyz":base64.b64encode(np.arange(12,dtype='<u2').tobytes()).decode(),
               "rgb":base64.b64encode(bytes([100]*12)).decode(),
               "ids":base64.b64encode(np.arange(1,5,dtype='<u2').tobytes()).decode()}
    path = tmp_path / "viewer.html"
    path.write_text(_VIEWER_TEMPLATE.replace("__PAYLOAD__",json.dumps(payload)).replace("__ANCHORS__","[]"))
    result = subprocess.run(["node",str(Path(__file__).with_name("test_viewer_js.mjs")),str(path)],
                            capture_output=True,text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)["status"] == "passed"


def test_zero_sentinels_and_recorded_padding_never_enter_viewer_payload(tmp_path):
    run, frames, _ = _synthetic_run(tmp_path)
    frame = frames[0]
    points = np.load(frame.pts3d_path)
    points[40:48,40:48] = 0
    np.save(frame.pts3d_path, points)
    alpha = np.ones((SIDE,SIDE),np.uint8); alpha[:,:2] = 0
    provider = run / "geometry" / "provider"; provider.mkdir()
    (provider / "frame_0001.json").write_text(json.dumps({"alpha_mask":{
        "dtype":"uint8","shape":list(alpha.shape),"data":base64.b64encode(alpha.tobytes()).decode()}}))
    summary = build_viewer_html(run)
    assert summary["points"] == SIDE*SIDE - 8*8 - SIDE*2
    assert summary["objects"][0]["label"] == "pallet"


def test_derived_refinement_is_bound_to_inventory_frame_and_source_hash(tmp_path):
    import ehs_spatial.viewer as viewer
    run, frames, _ = _synthetic_run(tmp_path)
    directory = run / "inventory" / "refinement_masks"; directory.mkdir(parents=True)
    (run / "refinements").mkdir()
    source = run / "refinements" / "box.json"; source.write_text('{}')
    mask = np.zeros((SIDE,SIDE),bool); mask[8:16,8:16] = True
    path = directory / "frame_0001__box.npy"; np.save(path,mask)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    item = {"label":"bollard","frame":"frame_0001","refine_slug":"box","refine_mask_sha256":digest}
    (run / "inventory" / "inventory.json").write_text(json.dumps({"objects":[item]}))
    path.with_suffix('.json').write_text(json.dumps({"frame_id":"frame_0001","coordinate_space":"canonical",
        "mask_sha256":digest,"source_sha256":hashlib.sha256(source.read_bytes()).hexdigest()}))
    result = viewer._masks_for_frame(run,frames[0],(SIDE,SIDE),[])
    assert result[0][2] == 0 and np.array_equal(result[0][1],mask)
    source.write_text('{"changed":true}')
    with pytest.raises(ValueError,match="accepted inventory/source"):
        viewer._masks_for_frame(run,frames[0],(SIDE,SIDE),[])


def test_accepted_thin_inventory_object_survives_viewer_sampling(tmp_path, monkeypatch):
    import ehs_spatial.viewer as viewer
    run, frames, _ = _synthetic_run(tmp_path)
    sam = run / "inventory" / "sam"; sam.mkdir(parents=True)
    item = {"label":"bolt","instance":0,"frame":"frame_0001","height_m":0.02,
            "size_m":"0.01x0.01","camera_dist_m":2.4}
    (sam.parent / "inventory.json").write_text(json.dumps({"objects":[item]}))
    (sam / "frame_0001__bolt.json").write_text('{"rle":["tiny"]}')
    tiny = np.zeros((SIDE,SIDE),bool); tiny[10,24] = True
    monkeypatch.setattr(viewer,"decode_coco_rle",lambda *args,**kwargs:tiny.copy())
    summary = build_viewer_html(run,max_points=80)
    assert summary["points"] == 80 and summary["supported_inv"] == [0]
    assert summary["objects"][0]["points"] == 1
    assert summary["objects"][0]["height_m"] == item["height_m"]
