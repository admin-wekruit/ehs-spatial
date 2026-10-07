"""End-to-end proof of the handoff path with a real GPU: the sam3d /v1 service (serving/sam3d_service.py, real model, the same
code the customer's container runs) started inside Modal's SAM 3D image on an A100 behind a Modal tunnel (= the customer's
jump endpoint), and the pipeline's provider (ehs_spatial/providers/sam3d.py, SAM3D_BACKEND=http) called from this machine with
a real object input of the 090 filled-geometry run. Ephemeral; nothing deployed.

  python e2e_modal_sam3d.py serve 20          # Modal: service up, prints the tunnel URL, stays 20 minutes (timeout 1500 s)
  python e2e_modal_sam3d.py client URL OBJ    # here: provider -> service -> mesh; compared with the Modal-generated candidate
The API key for the run is in SP/e2e-sam3d.key (generated once here, never printed).
"""
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import time

import modal

SP = Path(os.environ.get('SWAP_SCRATCH', '/private/tmp/claude-501/-Users-adam-Desktop-panoptes-public/1fd9a1db-e580-4bfc-8110-119a1cc38a99/scratchpad'))
SERV = Path(os.environ.get('PANOPTES_SERVING', '/Users/adam/Desktop/panoptes-public/panoptes-serving'))
WORKCELL = Path(os.environ.get('PANOPTES_WORKCELL', str(SERV)))   # one repository
KEY_FILE = SP / 'e2e-sam3d.key'
try:                                  # local side; inside the Modal container the key arrives as the function argument
    if not KEY_FILE.exists():
        KEY_FILE.write_text(secrets.token_hex(24))
    API_KEY = KEY_FILE.read_text().strip()
except OSError:
    API_KEY = None

# the module is imported both here and inside the container (to hydrate `serve`): define the app unconditionally
sys.path[:0] = [str(WORKCELL / 'modal_apps'), '/workcell/modal_apps']
import sam3d_research as sr  # noqa: E402  its image (SAM 3D Objects at the pinned revisions), weights volume, GPU

image = sr.image.pip_install('fastapi==0.115.6', 'uvicorn==0.34.0', 'pydantic==2.10.4', 'pyyaml==6.0.2', 'httpx==0.28.1')
if SERV.is_dir():   # local side: ship the service, the client package and the on-prem runner into the image
    image = (image.add_local_dir(SERV / 'serving', '/serving/serving', ignore=['__pycache__'])
             .add_local_dir(SERV / 'ehs_spatial', '/serving/ehs_spatial', ignore=['__pycache__'])
             .add_local_dir(WORKCELL / 'scripts/onprem', '/workcell/scripts/onprem', ignore=['__pycache__'])
             .add_local_file(WORKCELL / 'modal_apps/sam3d_research.py', '/workcell/modal_apps/sam3d_research.py')
             .add_local_file(WORKCELL / 'modal_apps/sam3d_mesh_only.patch', '/workcell/modal_apps/sam3d_mesh_only.patch')
             .add_local_file(WORKCELL / 'scripts/prepare_sam3d_mesh_source.py', '/workcell/scripts/prepare_sam3d_mesh_source.py'))
app = modal.App('e2e-sam3d-service')


@app.function(image=image, gpu=sr.GPU, cpu=4, memory=32768, timeout=1500, retries=0, min_containers=0,
              volumes={'/weights': sr.weights}, secrets=[modal.Secret.from_name('huggingface')])
def serve(minutes: int, api_key: str):
    import urllib.request
    env = dict(os.environ, PANOPTES_WORKCELL='/workcell', PANOPTES_WEIGHTS_MODE='hf-cache', PANOPTES_SERVICE_API_KEY=api_key,
               PYTHONPATH='/serving:/serving/serving', PANOPTES_SERVICE_MAX_QUEUE='4')
    env.pop('PANOPTES_FAKE_MODEL', None)
    # the services import their siblings (v1_common, registry) as top-level modules: run from the serving directory, as the compose stacks do
    proc = subprocess.Popen([sys.executable, '-m', 'uvicorn', 'sam3d_service:app', '--host', '0.0.0.0', '--port', '8805'],
                            cwd='/serving/serving', env=env)
    with modal.forward(8805) as tunnel:
        print('TUNNEL', tunnel.url, flush=True)
        t0 = time.time(); ok = False
        while time.time() - t0 < 900 and proc.poll() is None:
            try:
                req = urllib.request.Request('http://127.0.0.1:8805/healthz', headers={'X-API-Key': api_key})
                with urllib.request.urlopen(req, timeout=5) as r:
                    h = json.loads(r.read()); print('HEALTHZ', json.dumps(h), flush=True)
                    if h.get('ok'):
                        ok = True; break
            except Exception as e:  # noqa: BLE001  not up yet
                print('waiting', type(e).__name__, flush=True)
            time.sleep(10)
        print('READY' if ok else 'NOT READY', flush=True)
        end = time.time() + 60 * minutes
        while ok and time.time() < end and proc.poll() is None:
            time.sleep(15)
    proc.terminate()
    return ok


def client(url, oid='left_post'):
    import numpy as np
    os.environ.update(AB_CELL='090', AB_RUN=str(SP / 'checks/bbab-export-090-mvs-fill'), AB_OUT=str(SP / 'swap-runs/e2e-sam3d'),
                      SAM3D_BACKEND='http', SAM3D_HTTP_URLS=url, PANOPTES_SERVICE_API_KEY=API_KEY, PANOPTES_SERVICE_TIMEOUT_S='900')
    sys.path.insert(0, str(SERV)); sys.path.insert(0, str(SERV / 'research/module-swap-2026-10-07/notes/completion-licence-ab-2026-10-05'))
    import completion_ab as C
    from ehs_spatial.providers import sam3d as P
    manifest = json.loads((C.RUN / 'manifest.json').read_text())
    objects = {o['object_id']: o for o in json.loads((C.RUN / 'evidence/objects.json').read_text())['objects']}
    rgb, mask, pointmap, meta = C.sam3d_inputs(objects[oid], manifest)
    t = time.time(); out = P.generate(rgb, mask, pointmap, 42)
    base = np.load(SP / 'swap-runs/mvs-fill-ab/sam3d' / f'{oid}.npz')
    ext = lambda v: (np.percentile(v, 99.5, 0) - np.percentile(v, .5, 0)).round(3).tolist()
    print(json.dumps({'object': oid, 'photo': meta.get('frame_id'), 'wall_s': round(time.time() - t, 1), 'service_seconds': out.get('seconds'),
                      'gpu': out.get('gpu'), 'model_info': {k: out.get('model_info', {}).get(k) for k in ('model_id', 'model_revision', 'code_sha')},
                      'service_url': out.get('service_url'), 'cached': out.get('cached'),
                      'http_result': {'vertices': int(len(out['vertices'])), 'faces': int(len(out['faces'])), 'extent': ext(np.asarray(out['vertices']))},
                      'modal_candidate_today': {'vertices': int(len(base['vertices'])), 'faces': int(len(base['faces'])), 'extent': ext(base['vertices'])}},
                     indent=1))


@app.local_entrypoint()
def main(minutes: int = 20):
    """modal run e2e_modal_sam3d.py --minutes 20"""
    print('serve ->', serve.remote(minutes, API_KEY))


if __name__ == '__main__' and sys.argv[1:2] == ['client']:
    client(sys.argv[2], *(sys.argv[3:4]))
