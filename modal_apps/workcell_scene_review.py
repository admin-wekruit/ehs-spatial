"""Fully-loaded scene review of a live four-view report, in the cloud (nothing renders locally).

Runs web/checks/scene-review-check.mjs (Chromium, software WebGL): waits until every model reports loaded (or the cap),
records requests / failures / console errors, and captures the four-view, the 场景 3D pane and named object previews.

modal run modal_apps/workcell_scene_review.py --url LIVE_REPORT_URL --objects OBJECTS.json --out NEW_DIR [--cap 180]
OBJECTS.json = [["robot", "<entity id>"], ...]
"""
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile
import time

import modal

REPO = Path(__file__).resolve().parents[1]
PLAYWRIGHT = '1.55.0'
RATE = 4 * .0000131 + 8 * .00000222  # 4 CPU, 8 GiB list rate (USD/s); not an invoice

app = modal.App('workcell-scene-review')
image = (modal.Image.from_registry(f'mcr.microsoft.com/playwright:v{PLAYWRIGHT}-noble', add_python='3.11')
         .run_commands(f'mkdir -p /check && cd /check && npm init -y >/dev/null && npm install --silent playwright@{PLAYWRIGHT}')
         .add_local_file(REPO / 'web/checks/scene-review-check.mjs', '/check/scene-review-check.mjs'))


@app.function(image=image, cpu=4, memory=8 * 1024, timeout=1800, retries=0, min_containers=0)
def review(url: str, objects: str, cap: int):
    started = time.monotonic()
    out = Path('/out'); out.mkdir()
    result = subprocess.run(['node', '/check/scene-review-check.mjs', url, str(out), objects, str(cap)], capture_output=True, text=True,
                            env={**os.environ, 'PLAYWRIGHT_FROM': '/check/package.json'}, timeout=1600)
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode='w:gz') as bundle:
        for path in sorted(out.iterdir()):
            bundle.add(path, arcname=path.name)
    return {'returncode': result.returncode, 'stdout': result.stdout[-4000:], 'stderr': result.stderr[-4000:],
            'archive': archive.getvalue(), 'containerSeconds': time.monotonic() - started}


@app.local_entrypoint()
def main(url: str, objects: str, out: str, cap: int = 180):
    destination = Path(out)
    if destination.exists():
        raise ValueError('Choose a fresh output directory')
    pairs = json.loads(Path(objects).read_text())
    destination.mkdir(parents=True)
    start = time.monotonic()
    result = review.remote(url, json.dumps(pairs), cap)
    with tarfile.open(fileobj=io.BytesIO(result.pop('archive')), mode='r:gz') as bundle:
        bundle.extractall(destination, filter='data')
    ledger = {'mode': 'ephemeral modal run', 'hardware': '4 CPU, 8 GiB, no GPU', 'status': 'completed' if result['returncode'] == 0 else 'failed',
              'functionSeconds': result['containerSeconds'], 'callSeconds': time.monotonic() - start,
              'estimateUsd': RATE * result['containerSeconds'], 'callWindowEstimateUsd': RATE * (time.monotonic() - start),
              'actualBilledUsd': None, 'rateSource': 'https://modal.com/pricing'}
    (destination / 'spend-ledger.json').write_text(json.dumps(ledger, indent=2) + '\n')
    (destination / 'check-output.json').write_text(json.dumps({key: result[key] for key in ('returncode', 'stdout', 'stderr')}, indent=2) + '\n')
    print(json.dumps({'returncode': result['returncode'], 'tail': result['stdout'][-600:] or result['stderr'][-600:]}))
