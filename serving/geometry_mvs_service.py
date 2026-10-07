"""geometry-mvs service — docs/BACKENDS-v1.md `geometry-mvs` (port 8804): the GPU half of the licence-clean geometry route.

    POST /v1/geometry-mvs/jobs  {"input_sha256", "cell", "frames": [{"frame_id", "canonical_png_b64" | "_uri" (518x518 RGB),
                                 "alpha_npy_b64" | "_uri" (518x518 bool)}, ...], "options": {"start": "da3-base", "roma": "outdoor", "pairs": "all"}}
    -> 202 {"job_id", "input_sha256", "cached"}; GET /v1/jobs/{id} -> result {"result_uri" | "result_b64": <tar.gz>, "cell",
       "frames", "files", "stages", "seconds", "gpu", "model_info"}
    tar.gz layout (what research/.../mvs_route.py reads):
      checks/clean-gpu/<cell>/roma-I-J.npz, dense-I-J.npz, moge-<frame_id>.npz
      checks/da3fair-geom/<cell>-da3-base-padded/geometry/frames/<frame_id>/{pts3d,conf,valid_mask,intrinsics,camera_to_world}.npy + canonical.png, candidate_manifest.json
      checks/clean-geom/<cell>-da3-base-ba-f/geometry/...  (BA-f cameras: geometry_clean_ab.refine, numpy LM)

Stages, each a subprocess in the venv of its pinned on-prem image (deploy/Dockerfile.geometry; no model stays resident):
  1 DA3-BASE on the padded frames   PANOPTES_DA3_PYTHON  scripts/candidate_geometry_backend.py (DA3Runner, process_res 518)
  2 RoMa matches + dense warps, BA-f PANOPTES_GEOMETRY_PYTHON  serving/geometry_mvs_stage.py roma-refine
  3 MoGe-3 on the content crops      PANOPTES_MOGE_PYTHON  serving/geometry_mvs_stage.py moge
The content rect is read from the alpha masks (the columns with any alpha; one rect shared by every frame).
PANOPTES_FAKE_MODEL=1: the same layout with tiny arrays. Run:  cd serving && uvicorn geometry_mvs_service:app --port 8804
"""
from __future__ import annotations

import io
import json
import os
from itertools import combinations
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import tempfile
import time

import numpy as np
from PIL import Image
from pydantic import BaseModel

import v1_common as v1

service = v1.Service('geometry-mvs', loaded=False)
app = service.app
HERE = Path(__file__).resolve().parent
NAME = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$')
DEFAULTS = {'start': 'da3-base', 'roma': 'outdoor', 'pairs': 'all'}  # the only route this service implements
CANONICAL = 518
ENV = dict(workcell=os.environ.get('PANOPTES_WORKCELL', '/workcell'), serving=os.environ.get('PANOPTES_SERVING', str(HERE.parent)),
           da3_python=os.environ.get('PANOPTES_DA3_PYTHON', '/opt/da3/bin/python'), da3_vendor=os.environ.get('PANOPTES_DA3_VENDOR', '/vendor/depth-anything-3'),
           geometry_python=os.environ.get('PANOPTES_GEOMETRY_PYTHON', '/opt/geometry/bin/python'),
           moge_python=os.environ.get('PANOPTES_MOGE_PYTHON', '/opt/moge/bin/python'), roma_device=os.environ.get('PANOPTES_ROMA_DEVICE', 'auto'))
MOGE = next(ws for ws in service.entry['weights'] if ws['name'] == 'moge3')


class Frame(BaseModel):
    frame_id: str
    canonical_png_b64: str | None = None
    canonical_png_uri: str | None = None
    alpha_npy_b64: str | None = None
    alpha_npy_uri: str | None = None


class GeometryRequest(BaseModel):
    input_sha256: str
    cell: str
    frames: list[Frame]
    options: dict = {}


def startup():
    if v1.FAKE:
        return
    v1.verify_weights(service.entry)
    for key in ('da3_python', 'geometry_python', 'moge_python'):
        if not Path(ENV[key]).is_file():
            raise SystemExit(f'{key}: {ENV[key]} is not there (deploy/Dockerfile.geometry venvs)')
    for path in (Path(ENV['workcell']) / 'modal_apps/geometry_clean_ab.py', Path(ENV['serving']) / 'scripts/candidate_geometry_backend.py', Path(ENV['da3_vendor'])):
        if not path.exists():
            raise SystemExit(f'{path} is missing (PANOPTES_WORKCELL / PANOPTES_SERVING / PANOPTES_DA3_VENDOR)')


startup()
service.loaded = True


# ---------------------------------------------------------------- inputs
def decode(body: dict) -> dict:
    if not NAME.match(body['cell']):
        raise v1.bad_input('cell: letters, digits, _ . - only')
    options = {**DEFAULTS, **(body.get('options') or {})}
    if options != DEFAULTS:
        raise v1.bad_input(f'options: only {DEFAULTS} is implemented (got {options})')
    frames = body['frames']
    if len(frames) < 2:
        raise v1.bad_input('frames: at least two frames (RoMa pairs + bundle adjustment)')
    ids = [f['frame_id'] for f in frames]
    if len(set(ids)) != len(ids) or not all(NAME.match(i) for i in ids):
        raise v1.bad_input('frame_id: unique, letters, digits, _ . - only')
    out, shape, rect = [], None, None
    for f in frames:
        png = v1.read_binary(f, 'canonical_png')
        try:
            rgb = np.asarray(Image.open(io.BytesIO(png)).convert('RGB'))
        except (OSError, ValueError) as error:
            raise v1.bad_input(f'{f["frame_id"]} canonical_png: not a PNG ({error})') from None
        try:
            alpha = np.load(io.BytesIO(v1.read_binary(f, 'alpha_npy')), allow_pickle=False)
        except (OSError, ValueError) as error:
            raise v1.bad_input(f'{f["frame_id"]} alpha_npy: not a .npy ({error})') from None
        alpha = np.asarray(alpha).astype(bool)
        if alpha.shape != rgb.shape[:2]:
            raise v1.bad_input(f'{f["frame_id"]}: alpha {alpha.shape} is not the canonical grid {rgb.shape[:2]}')
        if shape is None:
            shape = rgb.shape[:2]
        elif rgb.shape[:2] != shape:
            raise v1.bad_input(f'{f["frame_id"]}: {rgb.shape[:2]} differs from the first frame {shape}')
        cols = np.flatnonzero(alpha.any(0))
        if not len(cols):
            raise v1.bad_input(f'{f["frame_id"]}: empty alpha')
        this = (int(cols[0]), int(cols[-1]) + 1)
        if rect is None:
            rect = this
        elif this != rect:
            raise v1.bad_input(f'{f["frame_id"]}: content columns {this} differ from the first frame {rect}')
        out.append(dict(frame_id=f['frame_id'], png=png, rgb=rgb, alpha=alpha))
    if not v1.FAKE and shape != (CANONICAL, CANONICAL):
        raise v1.bad_input(f'canonical frames must be {CANONICAL}x{CANONICAL} (got {shape[1]}x{shape[0]})')
    return dict(cell=body['cell'], frames=out, x0=rect[0], x1=rect[1], H=shape[0], W=shape[1])


# ---------------------------------------------------------------- the stages
def stage(name: str, cmd: list, env: dict | None = None) -> float:
    t = time.monotonic()
    p = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, env={**os.environ, **(env or {})})
    if p.returncode:
        raise v1.model_error(f'{name} exit {p.returncode}: ' + (p.stderr.strip() or p.stdout.strip())[-2000:])
    return round(time.monotonic() - t, 1)


def layout(work: Path, cell: str) -> dict:
    return dict(gpu=work / 'checks/clean-gpu' / cell, da3=work / f'checks/da3fair-geom/{cell}-da3-base-padded/geometry',
                geom=work / f'checks/clean-geom/{cell}-da3-base-ba-f/geometry')


def real(job: dict, work: Path) -> dict:
    d, ids, x0, x1, H = layout(work, job['cell']), [f['frame_id'] for f in job['frames']], job['x0'], job['x1'], job['H']
    inp = work / 'in'; inp.mkdir()
    pngs, alphas, crops = [], [], []
    for f in job['frames']:
        (inp / f'{f["frame_id"]}.png').write_bytes(f['png']); pngs.append(inp / f'{f["frame_id"]}.png')
        np.save(inp / f'{f["frame_id"]}_alpha.npy', f['alpha']); alphas.append(inp / f'{f["frame_id"]}_alpha.npy')
        Image.fromarray(f['rgb'][:, x0:x1]).save(inp / f'{f["frame_id"]}_crop.png'); crops.append(inp / f'{f["frame_id"]}_crop.png')
    weights = v1.weights_dir()
    seconds = {}
    # 1 DA3-BASE start geometry on the padded frames (= the fair A/B's da3fair-geom: candidate_geometry_backend.DA3Runner)
    seconds['da3'] = stage('da3-base', [ENV['da3_python'], Path(ENV['serving']) / 'scripts/candidate_geometry_backend.py', '--vendor-dir', ENV['da3_vendor'],
                                         '--model-dir', weights / 'local/da3-base', '--model-id', 'depth-anything/DA3-BASE', '--device', 'cuda',
                                         '--process-res', CANONICAL, '--inputs', *pngs, '--output', d['da3']], dict(HF_HUB_OFFLINE='1'))
    for junk in ('provider', 'map_anything_request.json', 'map_anything_response.json', 'point_cloud.glb'):  # raw copies; frames/ keeps every array
        p = d['da3'] / junk; shutil.rmtree(p) if p.is_dir() else p.unlink(missing_ok=True)
    frames_dir = d['da3'] / 'frames'  # MapAnythingAdapter names frames frame_0001.. by input order; the request's ids are the contract
    for index in range(1, len(ids) + 1):
        (frames_dir / f'frame_{index:04d}').rename(frames_dir / f'.in_{index:04d}')
    for index, fid in enumerate(ids, start=1):
        (frames_dir / f'.in_{index:04d}').rename(frames_dir / fid)
    manifest = d['da3'] / 'candidate_manifest.json'
    manifest.write_text(json.dumps(dict(json.loads(manifest.read_text()), inputVariant='padded', frameIds=ids, stage='geometry-mvs v1'), indent=2) + '\n')
    # 2 RoMa matches + dense warps on the content crops, then BA-f (geometry_clean_ab.roma_pair / refine)
    seconds['roma_refine'] = stage('roma-refine', [ENV['geometry_python'], HERE / 'geometry_mvs_stage.py', 'roma-refine', '--workcell', ENV['workcell'], '--weights', weights,
                                                    '--start', d['da3'], '--canonicals', *pngs, '--alphas', *alphas, '--frame-ids', *ids, '--x0', x0, '--x1', x1,
                                                    '--out-gpu', d['gpu'], '--out-geom', d['geom'], '--device', ENV['roma_device']])
    # 3 MoGe-3 on the content crops (geometry_clean_ab.moge)
    seconds['moge'] = stage('moge', [ENV['moge_python'], HERE / 'geometry_mvs_stage.py', 'moge', '--weights', weights, '--model', MOGE['repo'], '--revision', MOGE['revision'],
                                      '--crops', *crops, '--names', *ids, '--out', d['gpu']])
    shutil.rmtree(inp)
    return seconds


def fake(job: dict, work: Path) -> dict:
    """The same files with tiny arrays (4x4 grids), deterministic from the inputs."""
    d, ids, n = layout(work, job['cell']), [f['frame_id'] for f in job['frames']], len(job['frames'])
    h = w = 4
    d['gpu'].mkdir(parents=True)
    for i, j in combinations(range(n), 2):
        (d['gpu'] / f'roma-{i}-{j}.npz').write_bytes(v1.npz(uvA=np.full((8, 2), job['x0'], np.float64), uvB=np.full((8, 2), job['x0'] + 1, np.float64), certainty=np.ones(8, np.float32)))
        (d['gpu'] / f'dense-{i}-{j}.npz').write_bytes(v1.npz(uvAB=np.zeros((h, w, 2), np.float32), certA=np.ones((h, w), np.float16), uvBA=np.zeros((h, w, 2), np.float32), certB=np.ones((h, w), np.float16)))
    for f in job['frames']:
        (d['gpu'] / f'moge-{f["frame_id"]}.npz').write_bytes(v1.npz(points=np.ones((h, w, 3), np.float32), mask=np.ones((h, w), bool), intrinsicsNormalised=np.eye(3, dtype=np.float32)))
    for key, manifest in (('da3', dict(model_id='fake da3-base', stage='geometry-mvs v1 fake', frameIds=ids)), ('geom', dict(base='da3-base', refinement='ba-f', ba='fake', baUsable=True, frameIds=ids))):
        for k, f in enumerate(job['frames']):
            fd = d[key] / 'frames' / f['frame_id']; fd.mkdir(parents=True)
            arrays = dict(pts3d=np.full((h, w, 3), float(k + 1), np.float32), conf=np.ones((h, w), np.float32), valid_mask=np.ones((h, w), bool),
                          intrinsics=np.array([[400., 0, 258.5], [0, 400, 258.5], [0, 0, 1]]), camera_to_world=np.eye(4))
            for name, v in arrays.items():
                np.save(fd / f'{name}.npy', v)
            (fd / 'canonical.png').write_bytes(f['png'])
        (d[key] / 'candidate_manifest.json').write_text(json.dumps(dict(manifest, contentRect=[job['x0'], 0, job['x1'], job['H']]), indent=1) + '\n')
    return {'fake': 0.0}


def run(job: dict, input_sha256: str) -> dict:
    work = Path(tempfile.mkdtemp(prefix='geometry-mvs-', dir=os.environ.get('PANOPTES_SERVICE_WORK_DIR')))
    try:
        seconds = fake(job, work) if v1.FAKE else real(job, work)
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode='w:gz') as tar:
            tar.add(work / 'checks', arcname='checks')
        files = sorted(str(p.relative_to(work)) for p in (work / 'checks').rglob('*') if p.is_file())
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return dict(v1.publish_result('geometry-mvs', input_sha256, buf.getvalue()), cell=job['cell'], frames=[f['frame_id'] for f in job['frames']],
                content_rect_xyxy=[job['x0'], 0, job['x1'], job['H']], files=files, stages=seconds)


@app.post('/v1/geometry-mvs/jobs')
def geometry_job(request: GeometryRequest):
    body = request.model_dump()
    job = decode(body)  # in the request: bad_input is a 400 now, not an errored job
    return service.accepted(*service.submit(body['input_sha256'], lambda: run(job, body['input_sha256'])))
