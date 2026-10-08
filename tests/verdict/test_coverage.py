"""coverage_from_frames on a synthetic pinhole camera over a floor plane, and on the frozen 090 frames when they are present."""
import json
from pathlib import Path

import numpy as np
import pytest

from ehs_spatial.verdict.contracts import Obj, Scene
from ehs_spatial.verdict.layers.l1_scene.coverage import attach_coverage, coverage_from_frames, floor_level, plan_frame

ROOT = Path(__file__).resolve().parents[2]
FRAMES_090 = ROOT / "research/module-swap-2026-10-07/data/checks/bbab-geom/090-mvs-fill-padded/geometry/frames"
SCENE_090 = ROOT / "research/verdict-layer-trial-2026-10-07/out/scene-090.json"
NATIVE_TO_M_090 = 3.2371372068568487     # scale.nativeToMeters of the published measurement layer a9a6e0a0


def pinhole_floor_frame(tmp_path, n=64, cam=(0.0, 0.0, 1.5), pitch_deg=45.0, fov_deg=70.0, max_x=3.0):
    """A 64x64 camera at `cam` looking down `pitch_deg` along +x over the floor z = 0 (world up = +z); the floor patch in view is
    cut at x = max_x so there is a 'behind' to test. Returns (frame dir, floor hits N x 3)."""
    f = n / 2 / np.tan(np.radians(fov_deg / 2))
    p = np.radians(pitch_deg)
    R = np.stack([[0, -1, 0], [-np.sin(p), 0, -np.cos(p)], [np.cos(p), 0, -np.sin(p)]], 1)     # columns: right, down, forward
    c2w = np.eye(4)
    c2w[:3, :3], c2w[:3, 3] = R, cam
    v, u = np.mgrid[0:n, 0:n] + 0.5
    dirs = np.stack([(u - n / 2) / f, (v - n / 2) / f, np.ones_like(u)], -1) @ R.T
    pts = np.asarray(cam) + dirs * (-cam[2] / dirs[..., 2])[..., None]
    valid = (dirs[..., 2] < 0) & (pts[..., 0] <= max_x)
    d = tmp_path / "frame_0001"
    d.mkdir()
    np.save(d / "pts3d.npy", pts)
    np.save(d / "valid_mask.npy", valid)
    np.save(d / "camera_to_world.npy", c2w)
    np.save(d / "intrinsics.npy", np.array([[f, 0, n / 2], [0, f, n / 2], [0, 0, 1.0]]))
    return d, pts[valid]


GRID = dict(ground_normal=[0, 0, 1], basis=[[1, 0, 0], [0, 1, 0]], origin_xy=[-1.0, -3.0], cell_m=0.1, shape=[80, 60], stride=1)   # 64 px: every pixel is a ray


def cell(xy, origin=(-1.0, -3.0), cell_m=0.1):
    return tuple(int(v) for v in np.floor((np.asarray(xy) - origin) / cell_m))


def test_carved_cells_cover_the_floor_patch_in_view_and_nothing_behind_it(tmp_path):
    d, hits = pinhole_floor_frame(tmp_path)
    cov = coverage_from_frames([d], floor_level=0.0, **GRID)
    observed = {tuple(c) for c in cov.observed}
    hit_cells = {cell(h[:2]) for h in hits}
    assert hit_cells <= observed                                   # every floor point the camera saw
    xs = [ix for ix, _ in observed]
    assert max(xs) <= cell((3.0, 0))[0]                            # nothing behind the far edge of the patch (x > 3 m)
    assert min(xs) >= cell((0.0, 0))[0]                            # nothing behind the camera
    assert len(observed) > len(hit_cells) * 0.5 and cov.observed == sorted(cov.observed) and cov.shape == [80, 60]


def test_floor_level_is_estimated_from_the_frames_when_not_given(tmp_path):
    d, _ = pinhole_floor_frame(tmp_path)
    assert coverage_from_frames([d], **GRID).observed == coverage_from_frames([d], floor_level=0.0, **GRID).observed


def test_slab_excludes_samples_above_two_metres(tmp_path):
    d, _ = pinhole_floor_frame(tmp_path, cam=(0.0, 0.0, 2.6), pitch_deg=20.0)   # the first metres of every ray are above 2 m
    cov = coverage_from_frames([d], floor_level=0.0, **GRID)
    assert cell((0.3, 0.0)) not in {tuple(c) for c in cov.observed} and cov.observed


def box(oid, cls, center, size, bottom=0.0, views=("1", "2")):
    return Obj(id=oid, cls=cls, center_m=list(center), axes=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], size_m=list(size), bottom_m=bottom,
               top_m=bottom + size[2], views=list(views), floor_contact=bottom == 0.0)


def test_plan_frame_and_attach_coverage():
    scene = Scene(scene_id="t", objects=[box("a", "fence", (0, 0, 1.0), (2.0, 0.1, 2.0))], ground_normal=[0, 0, 1])
    basis, origin, shape = plan_frame(scene, cell_m=0.1, margin_m=1.0)
    assert np.allclose(np.abs(basis), [[0, 1, 0], [1, 0, 0]]) and shape == [21, 40] and floor_level(scene) == 0.0
    cov = coverage_from_frames([], [0, 0, 1], basis, origin, 0.1, shape, floor_level=0.0)
    assert attach_coverage(scene, cov).coverage.observed == [] and scene.coverage is None


def load_trial_scene(path):
    d = json.loads(Path(path).read_text())
    d["objects"] = [{**o, "views": [str(v) for v in o["views"]]} for o in d["objects"]]
    return Scene.model_validate({**d, "schema": "verdict/1"})


@pytest.mark.skipif(not (FRAMES_090.exists() and SCENE_090.exists()), reason="frozen 090 frames not checked out")
def test_frozen_090_frames_carve_the_floor_under_the_bollards():
    scene = load_trial_scene(SCENE_090)
    basis, origin, shape = plan_frame(scene)
    assert shape == [75, 55]                                        # the trial's grid (README of the trial)
    assert abs(floor_level(scene) + 1.461) < 0.005                  # = -ground.offset x nativeToMeters of the published layer
    cov = coverage_from_frames(sorted(FRAMES_090.glob("frame_*")), scene.ground_normal, basis, origin, 0.1, shape,
                               floor_level=floor_level(scene), native_to_m=NATIVE_TO_M_090, stride=8)
    observed = {tuple(c) for c in cov.observed}
    E, lo = np.asarray(basis), np.asarray(origin)
    for o in scene.objects:
        if o.floor_contact:                                          # the bollards stand on seen floor, in every photo
            assert tuple(np.floor((np.asarray(o.center_m) @ E.T - lo) / 0.1).astype(int)) in observed, o.label
    assert 0.05 < len(observed) / (shape[0] * shape[1]) < 0.95
