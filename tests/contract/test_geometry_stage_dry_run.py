"""The real geometry-mvs stage driver (serving/geometry_mvs_stage.py roma-refine) against the ehs-spatial function bodies on a
synthetic scene, with RoMa replaced by exact warps and torch by a stub: exercises the modal-stub import, content masks, the pair
files, geometry_clean_ab.refine (numpy LM bundle adjustment, cv2 USAC) and the BA-f output layout, without a GPU or weights.
Needs PANOPTES_WORKCELL (the ehs-spatial checkout) and cv2 + scipy; skipped otherwise."""
import json
import os
from itertools import combinations
from pathlib import Path
import sys
import types

import numpy as np
import pytest

WORKCELL = Path(os.environ.get('PANOPTES_WORKCELL', '/nonexistent'))
pytestmark = pytest.mark.skipif(not (WORKCELL / 'modal_apps/geometry_clean_ab.py').is_file(), reason='PANOPTES_WORKCELL not set')


@pytest.fixture(autouse=True)
def _isolated_modules():
    """These tests install the on-prem Modal stub and import the real run_stage / geometry modules; give every other test the
    interpreter state it started with (the provider tests assert that run_stage / modal are not pre-imported)."""
    before, path, meta, environ = dict(sys.modules), list(sys.path), list(sys.meta_path), dict(os.environ)   # run_stage also sets PANOPTES_ONPREM, HF_HUB_OFFLINE, hub paths, a torch.hub meta-path pin
    for name in [k for k in sys.modules if k == 'modal' or k.startswith('modal.')]:   # the real client, if another test imported it:
        del sys.modules[name]                                                        # use_stub() must be able to install the stub
    yield
    os.environ.clear(); os.environ.update(environ)
    def first_party(mod):   # the stage modules, the stub and the research-note modules they import (fair_ab_modal reads SWAP_SCRATCH at import)
        f = getattr(mod, '__file__', None) or ''
        return bool(f) and '/site-packages/' not in f and '/lib/python3' not in f
    for name, mod in [(k, v) for k, v in sys.modules.items() if k not in before]:   # drop those; keep third-party modules first imported
        if name == 'modal' or name.startswith('modal.') or first_party(mod):          # here (numpy.fft etc. are C extensions: one load per process)
            del sys.modules[name]
    sys.modules.update(before)   # every module object exactly as it was - the modal.* submodules removed above included: restoring
    sys.path[:] = path           # only the `modal` package leaves a half-imported client (modal.config re-executes into a circular import)
    sys.meta_path[:] = meta
cv2 = pytest.importorskip('cv2')
pytest.importorskip('scipy')
H = W = 64
K = np.array([[80., 0, 31.5], [0, 80, 31.5], [0, 0, 1]])


def look(C, target):
    z = target - C; z /= np.linalg.norm(z); x = np.cross(z, [0, 0, 1.]); x /= np.linalg.norm(x); y = np.cross(z, x)
    M = np.eye(4); M[:3, :3] = np.c_[x, y, z]; M[:3, 3] = C; return M


def scene(gc):
    """Floor + wall seen by three cameras (geometry_clean_ab._check's scene at 64 px): true depth maps, perturbed 'backbone'."""
    rng = np.random.default_rng(0)
    true = [look(np.array(c, float), np.array([0., 3, .5])) for c in ([0, 0, 1.5], [1.2, .3, 1.5], [-1., .5, 1.4])]

    def depth_map(M):
        g = gc.grid(H, W); d = np.c_[g, np.ones(len(g))] @ np.linalg.inv(K).T @ M[:3, :3].T; C = M[:3, 3]
        tf = np.where(d[:, 2] < 0, -C[2] / np.minimum(d[:, 2], -1e-9), np.inf); tw = np.where(d[:, 1] > 0, (4.5 - C[1]) / np.maximum(d[:, 1], 1e-9), np.inf)
        return np.minimum(tf, tw).reshape(H, W)
    frames = []
    for M, f in zip(true, [1.0, 1.08, .93]):
        P = np.eye(4); P[:3, :3] = cv2.Rodrigues(rng.normal(size=3) * np.radians(.8))[0]; P[:3, 3] = rng.normal(size=3) * .04
        z = depth_map(M) * f; ok = np.isfinite(z) & (z < 50)
        frames.append(dict(pts3d=gc.backproject(K, P @ M, gc.grid(H, W), np.where(ok, z, 1).ravel()).reshape(H, W, 3).astype(np.float32),
                           conf=np.ones((H, W), np.float32), valid=ok, K=K.copy(), c2w=P @ M, depth=np.where(ok, depth_map(M), np.nan)))
    return true, frames


def test_roma_refine_dry_run(tmp_path, monkeypatch):
    sys.path.insert(0, str(WORKCELL / 'scripts/onprem')); sys.path.insert(0, str(WORKCELL / 'modal_apps'))
    import run_stage
    run_stage.use_stub()
    fake_torch = types.ModuleType('torch'); fake_torch.__version__ = 'stub'; fake_torch.manual_seed = lambda s: None
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False, get_device_name=lambda i: 'none')
    fake_torch.Tensor = type('Tensor', (), {})  # scipy's array-API probe looks torch.Tensor up when a 'torch' module is loaded
    monkeypatch.setitem(sys.modules, 'torch', fake_torch)
    import geometry_clean_ab as gc
    true, frames = scene(gc)
    start = tmp_path / 'start'
    for i, fr in enumerate(frames):
        d = start / 'frames' / f'f{i}'; d.mkdir(parents=True)
        for name, v in zip(('pts3d', 'conf', 'valid_mask', 'intrinsics', 'camera_to_world'), (fr['pts3d'], fr['conf'], fr['valid'], fr['K'], fr['c2w'])):
            np.save(d / f'{name}.npy', v)
    (start / 'candidate_manifest.json').write_text(json.dumps({'model_id': 'synthetic'}))
    from PIL import Image
    canon, alphas = [], []
    for i, fr in enumerate(frames):
        p = tmp_path / f'f{i}.png'; Image.fromarray(np.full((H, W, 3), 90, np.uint8)).save(p); canon.append(p)
        a = tmp_path / f'f{i}_alpha.npy'; np.save(a, np.ones((H, W), bool)); alphas.append(a)

    def warp(a, b):  # the exact warp from the true depth: where a's pixels land in b
        z = frames[a]['depth']; Xa = gc.backproject(K, true[a], gc.grid(H, W), np.where(np.isfinite(z), z, 1).ravel())
        return gc.project(K, true[b], Xa).reshape(H, W, 2).astype(np.float32)

    def fake_pair(model, im_a, im_b, device, sparse=True, dense=True, x0=None):
        i, j = pair_ids.pop(0)  # the driver visits combinations(range(n), 2) in order
        uv = warp(i, j); vis = (uv[..., 0] > 1) & (uv[..., 0] < W - 2) & (uv[..., 1] > 1) & (uv[..., 1] < H - 2) & np.isfinite(frames[i]['depth'])
        ua = gc.grid(H, W)[vis.ravel()]; ub = uv.reshape(-1, 2)[vis.ravel()].astype(np.float64)
        rng = np.random.default_rng(i * 10 + j); pick = rng.choice(len(ua), min(2000, len(ua)), replace=False)
        return dict(sparse=dict(uvA=ua[pick] + rng.normal(scale=.2, size=(len(pick), 2)), uvB=ub[pick] + rng.normal(scale=.2, size=(len(pick), 2)), certainty=np.ones(len(pick), np.float32)),
                    dense=dict(uvAB=uv, certA=np.ones((H, W), np.float16), uvBA=warp(j, i), certB=np.ones((H, W), np.float16)))
    pair_ids = list(combinations(range(3), 2))
    monkeypatch.setattr(gc, 'roma_model', lambda device, weights: 'stub-roma')
    monkeypatch.setattr(gc, 'roma_pair', fake_pair)
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'serving'))
    import geometry_mvs_stage
    out_gpu, out_geom = tmp_path / 'checks/clean-gpu/cell', tmp_path / 'checks/clean-geom/cell-da3-base-ba-f/geometry'
    geometry_mvs_stage.main(['roma-refine', '--workcell', str(WORKCELL), '--weights', str(tmp_path), '--start', str(start),
                             '--canonicals', *map(str, canon), '--alphas', *map(str, alphas), '--frame-ids', 'f0', 'f1', 'f2',
                             '--x0', '0', '--x1', str(W), '--out-gpu', str(out_gpu), '--out-geom', str(out_geom), '--device', 'cpu'])
    assert sorted(p.name for p in out_gpu.iterdir()) == ['dense-0-1.npz', 'dense-0-2.npz', 'dense-1-2.npz', 'roma-0-1.npz', 'roma-0-2.npz', 'roma-1-2.npz']
    with np.load(out_gpu / 'roma-0-1.npz') as d:
        assert set(d.files) == {'uvA', 'uvB', 'certainty'}
    for i in range(3):
        assert sorted(p.name for p in (out_geom / 'frames' / f'f{i}').iterdir()) == ['camera_to_world.npy', 'canonical.png', 'conf.npy', 'intrinsics.npy', 'pts3d.npy', 'valid_mask.npy']
    man = json.loads((out_geom / 'candidate_manifest.json').read_text())
    assert man['refinement'] == 'ba-f' and man['baUsable'] and man['startModel'] == 'synthetic' and man['roma']['device'] == 'cpu'
    out = gc.read_geometry(out_geom)
    rt, rs, ro = gc.relative([dict(c2w=M) for M in true]), gc.relative(frames), gc.relative(out)
    for k in rt:  # the bundle adjustment moved every relative pose from the perturbed start towards the truth (64 px scene, focal refined)
        before, after = gc.rot_deg(rt[k]['R'].T @ rs[k]['R']), gc.rot_deg(rt[k]['R'].T @ ro[k]['R'])
        assert after < min(before, 1.), (k, before, after, man['report']['poseChange'])
        assert np.degrees(np.arccos(min(1, rt[k]['b'] @ ro[k]['b']))) < 2., k


def test_stub_runs_enter_on_attribute_access():
    """serving/sam3d_service.py loads the model with `model.load`: the stub's _Obj runs @enter on the first attribute access."""
    sys.path.insert(0, str(WORKCELL / 'scripts/onprem'))
    import run_stage
    modal = run_stage.use_stub()
    app, loads = modal.App('dry-run'), []

    @app.cls(gpu='A100')
    class Model:
        @modal.enter()
        def load(self):
            loads.append(1); self.pipeline = 'ready'

        @modal.method()
        def run(self, x):
            return self.pipeline, x
    m = Model()
    m.load  # noqa: B018
    assert loads == [1] and m.run.remote(3) == ('ready', 3) and loads == [1]
