"""The v1 contract against a live service: PANOPTES_CONTRACT_URL = a sam3d service root (http://host:8805), optionally
PANOPTES_CONTRACT_GEOMETRY_URL (http://host:8804); PANOPTES_SERVICE_API_KEY as the X-API-Key. Skipped when unset.
    PANOPTES_CONTRACT_URL=http://10.21.72.251:8805 PANOPTES_SERVICE_API_KEY=... pytest tests/contract/live.py -q
Real services run the real model (one SAM 3D candidate ~10 s on an A100, a geometry cell minutes), so the sam3d body is
the synthetic scene of modal_apps/sam3d_research.debug and the geometry check needs PANOPTES_CONTRACT_GEOMETRY_CELL=DIR
(a frozen run directory with manifest.json frames[] canonical / alpha, 518x518) to send real frames."""
import base64
import io
import json
import os
from pathlib import Path
import tarfile
import time

import httpx
import numpy as np
import pytest

import inputs

URL = os.environ.get('PANOPTES_CONTRACT_URL')
GEOMETRY_URL = os.environ.get('PANOPTES_CONTRACT_GEOMETRY_URL')
KEY = {'X-API-Key': os.environ.get('PANOPTES_SERVICE_API_KEY', '')}
pytestmark = pytest.mark.skipif(not URL, reason='PANOPTES_CONTRACT_URL not set')


@pytest.fixture(scope='module')
def client():
    with httpx.Client(base_url=URL, timeout=float(os.environ.get('PANOPTES_SERVICE_TIMEOUT_S', '1800'))) as c:
        yield c


def poll(client, job_id, timeout=3600) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        rec = client.get(f'/v1/jobs/{job_id}', headers=KEY).json()
        if rec['status'] in ('done', 'error'):
            return rec
        time.sleep(2)
    raise AssertionError('job did not finish')


def test_auth(client):
    r = client.get('/healthz')
    assert r.status_code == 401 and r.json()['error']['code'] == 'unauthorized'


def test_healthz_and_info(client):
    h = client.get('/healthz', headers=KEY).json()
    assert h['ok'] is True and h['model'] == 'sam3d' and {'revision', 'gpu', 'vram_free_mb', 'queue_depth', 'running'} <= set(h), h
    i = client.get('/v1/info', headers=KEY).json()
    assert {'model_id', 'model_revision', 'code_revision', 'weights_sha256', 'licence', 'code_sha', 'seed_policy'} <= set(i), i
    assert i['model_id'] == 'facebook/sam-3d-objects' and len(i['weights_sha256']) >= 4


def test_sam3d_sync_and_job(client):
    body = inputs.sam3d_body(size=256, salt=f'live-{int(time.time())}')
    r = client.post('/v1/sam3d', json=body, headers=KEY)
    assert r.status_code == 200, r.text
    result = r.json()
    with np.load(io.BytesIO(base64.b64decode(result['mesh_npz_b64']))) as d:
        assert set(d.files) == {'vertices', 'faces', 'colors', 'object_to_camera_p3d'} and d['vertices'].shape == (result['vertices'], 3)
    assert result['model_info']['model_revision'] and isinstance(result['seconds'], float) and result['gpu']
    j = client.post('/v1/sam3d/jobs', json=body, headers=KEY)
    assert j.status_code == 202 and j.json()['cached'] is True and j.json()['input_sha256'] == body['input_sha256']
    rec = poll(client, j.json()['job_id'])
    assert rec['status'] == 'done' and rec['result']['mesh_npz_b64'] == result['mesh_npz_b64']
    assert client.get('/v1/jobs/no-such-job', headers=KEY).status_code == 404


@pytest.mark.skipif(not GEOMETRY_URL, reason='PANOPTES_CONTRACT_GEOMETRY_URL not set')
def test_geometry_job():
    cell_dir = os.environ.get('PANOPTES_CONTRACT_GEOMETRY_CELL')
    if cell_dir:
        man = json.loads((Path(cell_dir) / 'manifest.json').read_text())
        frames, parts = [], []
        for f in man['frames']:
            png, alpha = (Path(cell_dir) / f['canonical']).read_bytes(), inputs.npy(np.load(Path(cell_dir) / f['alpha']))
            parts += [png, alpha]; frames.append(dict(frame_id=f['frame_id'], canonical_png_b64=inputs.b64(png), alpha_npy_b64=inputs.b64(alpha)))
        body = dict(input_sha256=inputs.sha256_of(*parts, dict(cell=man.get('cell', 'live'))), cell=str(man.get('cell', 'live')), frames=frames,
                    options={'start': 'da3-base', 'roma': 'outdoor', 'pairs': 'all'})
    else:
        body = inputs.geometry_body(cell='live', n=2)  # random 518x518 content: the stages run, the geometry is meaningless
    with httpx.Client(base_url=GEOMETRY_URL, timeout=1800) as c:
        assert c.get('/healthz', headers=KEY).json()['model'] == 'geometry-mvs'
        r = c.post('/v1/geometry-mvs/jobs', json=body, headers=KEY)
        assert r.status_code == 202, r.text
        rec = poll(c, r.json()['job_id'])
    assert rec['status'] == 'done', rec['error']
    result = rec['result']
    data = base64.b64decode(result['result_b64']) if 'result_b64' in result else None
    assert data is not None or result['result_uri'].startswith('s3://')
    if data is not None:
        with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as tar:
            names = {m.name for m in tar.getmembers() if m.isfile()}
        cell = body['cell']
        assert f'checks/clean-gpu/{cell}/roma-0-1.npz' in names and f'checks/clean-gpu/{cell}/dense-0-1.npz' in names
        assert any(n.startswith(f'checks/da3fair-geom/{cell}-da3-base-padded/geometry/frames/') for n in names)
        assert f'checks/clean-geom/{cell}-da3-base-ba-f/geometry/candidate_manifest.json' in names
