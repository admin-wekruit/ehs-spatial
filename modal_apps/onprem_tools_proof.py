"""Proof that the e-stop scale and measurement-layer CLIs run in the on-prem CPU image (docker/workcell-cpu.Dockerfile, /opt/checks)
with the network blocked, as `docker run --network none` would on a customer server. Modal is only the test bench.

Inputs travel as one tar (call argument) laid out as the CLIs' roots: notes/ (research-notes files), scratch/ (views, layer
patch), runs/ (run dirs), photos/, masks/, layers/. In the container: both self-tests, scripts/workcell_estop_scale.py for both
reports, scripts/workcell_layer_build.py for both reports from the given layers; files <= 100 kB come back, every output's sha256 too (block_network also blocks the
upload of a large return value).

  PANOPTES_SERVING=SERVING_DIR modal run modal_apps/onprem_tools_proof.py --inputs INPUTS.tar --spec SPEC.json --out NEW_DIR
SPEC = {"estop": {name: [args...]}, "layers": {name: [args...]}} with paths relative to /data (the unpacked tar).
PANOPTES_SERVING = the panoptes-serving checkout (the image's serving/ half of the build context); required, no machine default:
the image is built when this module is imported, before the entrypoint's flags exist.
"""
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import time

import modal

REPO = Path(__file__).resolve().parents[1]
SERVING = os.environ.get('PANOPTES_SERVING', '')
RATE = 8 * .0000131 + 16 * .00000222  # list rate (USD/s), not an invoice

if modal.is_local():
    import atexit
    if not SERVING or not Path(SERVING, 'ehs_spatial').is_dir():
        raise SystemExit(f'set PANOPTES_SERVING to the panoptes-serving checkout (got {SERVING!r})')
    CTX = Path(tempfile.mkdtemp(prefix='panoptes-onprem-context-'))
    subprocess.run(['bash', str(REPO / 'scripts/onprem/stage_context.sh'), str(CTX), SERVING], check=True, capture_output=True)
    atexit.register(shutil.rmtree, CTX, True)
    image = modal.Image.from_dockerfile(CTX / 'workcell/docker/workcell-cpu.Dockerfile', context_dir=CTX)
else:
    image = modal.Image.debian_slim()
app = modal.App('onprem-tools-proof')


@app.function(image=image, cpu=8, memory=16 * 1024, timeout=1800, retries=0, min_containers=0, block_network=True)
def run(inputs: bytes, spec: dict) -> dict:
    start = time.monotonic()
    tarfile.open(fileobj=io.BytesIO(inputs)).extractall('/data')
    env = {k: v for k, v in os.environ.items() if not k.startswith('MODAL')}
    py = '/opt/checks/bin/python'; steps = []

    def sh(name, args):
        t = time.monotonic(); p = subprocess.run([py, *args], cwd='/data', env=env, capture_output=True, text=True, timeout=1500)
        steps.append(dict(step=name, rc=p.returncode, seconds=round(time.monotonic() - t, 1), stdout=p.stdout[-4000:], stderr=p.stderr[-3000:]))
    probe = "import socket\ntry:\n    socket.create_connection(('1.1.1.1', 443), timeout=5); print('REACHABLE')\nexcept Exception as e:\n    print('blocked:', type(e).__name__)"
    sh('network probe', ['-c', probe])
    sh('versions', ['-c', 'import numpy, cv2, scipy; print(numpy.__version__, cv2.__version__, scipy.__version__)'])
    sh('estop self-test', ['/workcell/scripts/workcell_estop_scale.py', '--self-test'])
    sh('layer self-test', ['/workcell/scripts/workcell_layer_build.py', '--self-test'])
    for name, args in spec['estop'].items():
        sh(f'estop {name}', ['/workcell/scripts/workcell_estop_scale.py', *args])
    for name, args in spec['layers'].items():
        sh(f'layer {name}', ['/workcell/scripts/workcell_layer_build.py', *args])
    # block_network also blocks Modal's blob upload of a large return value: small files come back, the rest as sha256
    import hashlib
    out = [p for p in Path('/data/out').rglob('*') if p.is_file()]
    files = {str(p.relative_to('/data/out')): p.read_bytes() for p in out if p.stat().st_size <= 100_000}
    sha256 = {str(p.relative_to('/data/out')): hashlib.sha256(p.read_bytes()).hexdigest() for p in out}
    return dict(steps=steps, files=files, sha256=sha256, containerSeconds=time.monotonic() - start)


@app.local_entrypoint()
def main(inputs: str, spec: str, out: str):
    destination = Path(out)
    if destination.exists():
        raise ValueError('Choose a fresh output directory')
    start = time.monotonic()
    result = run.remote(Path(inputs).read_bytes(), json.loads(Path(spec).read_text()))
    for name, data in result.pop('files').items():
        (destination / name).parent.mkdir(parents=True, exist_ok=True); (destination / name).write_bytes(data)
    result['ledger'] = {'mode': 'ephemeral modal run', 'hardware': '8 CPU, 16 GiB, block_network', 'functionSeconds': result['containerSeconds'],
                        'callSeconds': time.monotonic() - start, 'estimateUsd': RATE * result['containerSeconds'], 'actualBilledUsd': None}
    (destination / 'results.json').write_text(json.dumps(result, indent=1, ensure_ascii=False) + '\n')
    for s in result['steps']:
        print(f"[{s['rc']}] {s['step']} ({s['seconds']} s): {(s['stdout'] or s['stderr']).strip().splitlines()[-1:] if (s['stdout'] or s['stderr']) else ''}")
    print(json.dumps(result['ledger']))
