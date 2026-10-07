"""Joint Pi3X geometry for a frozen capture on a private, ephemeral Modal GPU (no endpoint).

The same pinned runner and adapter as the local September run (scripts/candidate_pi3x_backend.py, MapAnythingAdapter):
Pi3 code 9fa3ddb, weights yyfz233/Pi3X@bb1deea (SHA-checked by the runner). Writes RUN/geometry exactly as the local CLI.

modal run modal_apps/pi3x_geometry.py --run outputs/candidate-evaluation/RUN
"""
import io
import json
from pathlib import Path
import tarfile
import time

import modal

REPO = Path(__file__).resolve().parents[1]
CODE_REV = '9fa3ddb3f8d53041f8b2738df404f62223bbaa7b'
MODEL_REV = 'bb1deea4d7423de5b30691739cb451a3f57dc1d5'
RATE = .000694 + 8 * .0000131 + 32 * .00000222  # A100-80GB + 8 CPU + 32 GiB list rate (USD/s); not an invoice

app = modal.App('pi3x-geometry')
image = (modal.Image.debian_slim(python_version='3.11')
         .apt_install('git', 'libgl1', 'libglib2.0-0')
         .pip_install('torch==2.5.1', 'torchvision==0.20.1', 'numpy==1.26.4', 'pillow==11.0.0', 'opencv-python-headless==4.10.0.84',
                      'plyfile==1.1', 'huggingface_hub==0.36.0', 'safetensors==0.4.5', 'pydantic==2.9.2', 'einops==0.8.0', 'trimesh==5.1.0')  # 5.x writes the single-node GLB the platform importer accepts
         .run_commands(f'git clone https://github.com/yyfz/Pi3.git /vendor/pi3 && git -C /vendor/pi3 checkout {CODE_REV}')
         .add_local_dir(REPO / 'ehs_spatial', '/serving/ehs_spatial')
         .add_local_file(REPO / 'scripts/candidate_pi3x_backend.py', '/serving/scripts/candidate_pi3x_backend.py')
         .add_local_file(REPO / 'scripts/candidate_geometry_backend.py', '/serving/scripts/candidate_geometry_backend.py'))


@app.function(image=image, gpu='A100-80GB', cpu=8, memory=32 * 1024, timeout=1800, retries=0, min_containers=0)
def infer(frames: dict) -> dict:
    import sys
    from huggingface_hub import hf_hub_download
    sys.path[:0] = ['/serving', '/serving/scripts']
    from candidate_pi3x_backend import Pi3XRunner
    from ehs_spatial.providers.map_anything import MapAnythingAdapter
    start = time.monotonic()
    model_dir = Path('/tmp/pi3x')
    hf_hub_download('yyfz233/Pi3X', 'model.safetensors', revision=MODEL_REV, local_dir=model_dir)
    inputs = []
    for name, data in sorted(frames.items()):
        path = Path('/tmp/in') / name; path.parent.mkdir(exist_ok=True); path.write_bytes(data); inputs.append(str(path))
    out = Path('/tmp/geometry')
    runner = Pi3XRunner('/vendor/pi3', model_dir, device='cuda')
    result, cloud = MapAnythingAdapter(runner=runner).run(inputs, out)
    (out / 'candidate_manifest.json').write_text(json.dumps(runner.metadata, indent=2) + '\n')
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w:gz') as bundle:
        bundle.add(out, arcname='geometry')
    return {'archive': buffer.getvalue(), 'frames': len(result), 'metadata': runner.metadata, 'containerSeconds': time.monotonic() - start}


@app.local_entrypoint()
def main(run: str):
    root = Path(run).resolve(); manifest = json.loads((root / 'manifest.json').read_text())
    if (root / 'geometry').exists():
        raise ValueError('RUN/geometry already exists')
    frames = {Path(f['canonical']).name: (root / f['canonical']).read_bytes() for f in manifest['frames']}
    start = time.monotonic()
    result = infer.remote(frames)
    with tarfile.open(fileobj=io.BytesIO(result.pop('archive')), mode='r:gz') as bundle:
        bundle.extractall(root, filter='data')
    ledger = {'mode': 'ephemeral modal run', 'hardware': 'A100-80GB, 8 CPU, 32 GiB', 'functionSeconds': result['containerSeconds'],
              'callSeconds': time.monotonic() - start, 'estimateUsd': RATE * result['containerSeconds'], 'actualBilledUsd': None,
              'rateSource': 'https://modal.com/pricing'}
    (root / 'geometry' / 'spend-ledger.json').write_text(json.dumps(ledger, indent=2) + '\n')
    print(json.dumps({'frames': result['frames'], 'device': result['metadata'].get('device'), 'load': result['metadata'].get('model_load_seconds'),
                      'infer': result['metadata'].get('inference_and_encoding_seconds'), 'pinhole': result['metadata'].get('native_pinhole_fit'), **ledger}, default=str)[:1200])
