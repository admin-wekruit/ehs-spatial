"""Scene assembly (scripts/research/assemble_lucida_scene.py, unchanged) on an ephemeral Modal CPU, so the laptop stays idle.

The run directory goes up as one archive; only its new result/ comes back.

modal run modal_apps/assemble_scene.py --run outputs/candidate-evaluation/RUN [--iterations 100 --eval-size 288]
"""
import io
import json
from pathlib import Path
import tarfile
import time

import modal

REPO = Path(__file__).resolve().parents[1]
RATE = 8 * .0000131 + 16 * .00000222  # 8 CPU, 16 GiB list rate (USD/s); not an invoice

app = modal.App('assemble-scene')
image = (modal.Image.debian_slim(python_version='3.12')
         .apt_install('libgl1', 'libgomp1', 'libglib2.0-0', 'libx11-6')
         .pip_install('numpy==2.1.3', 'opencv-python-headless==4.10.0.84', 'open3d==0.19.0', 'pillow==11.0.0', 'scipy==1.14.1', 'trimesh==4.4.9')
         .add_local_dir(REPO / 'scripts/research', '/serving/scripts/research', ignore=['__pycache__']))


@app.function(image=image, cpu=8, memory=16 * 1024, timeout=3600, retries=0, min_containers=0)
def assemble_remote(archive: bytes, name: str, iterations: int, eval_size: int) -> dict:
    import sys
    sys.path.insert(0, '/serving/scripts/research')
    from assemble_lucida_scene import assemble
    start = time.monotonic()
    root = Path('/tmp/run')
    with tarfile.open(fileobj=io.BytesIO(archive), mode='r:gz') as bundle:
        bundle.extractall(root, filter='data')
    run = root / name
    assemble(run, iterations, eval_size)
    out = io.BytesIO()
    with tarfile.open(fileobj=out, mode='w:gz') as bundle:
        bundle.add(run / 'result', arcname='result')
    return {'archive': out.getvalue(), 'containerSeconds': time.monotonic() - start}


@app.local_entrypoint()
def main(run: str, iterations: int = 100, eval_size: int = 288):
    root = Path(run).resolve()
    if (root / 'result').exists():
        raise ValueError('RUN/result already exists')
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w:gz') as bundle:
        bundle.add(root, arcname=root.name, filter=lambda info: None if '/generation/attempts' in info.name else info)
    start = time.monotonic()
    result = assemble_remote.remote(buffer.getvalue(), root.name, iterations, eval_size)
    with tarfile.open(fileobj=io.BytesIO(result.pop('archive')), mode='r:gz') as bundle:
        bundle.extractall(root, filter='data')
    ledger = {'mode': 'ephemeral modal run', 'hardware': '8 CPU, 16 GiB, no GPU', 'functionSeconds': result['containerSeconds'],
              'callSeconds': time.monotonic() - start, 'estimateUsd': RATE * result['containerSeconds'], 'actualBilledUsd': None,
              'rateSource': 'https://modal.com/pricing'}
    (root / 'result' / 'assembly-spend-ledger.json').write_text(json.dumps(ledger, indent=2) + '\n')
    print(json.dumps(ledger))
