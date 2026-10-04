"""Real-browser check of the published workcell report, run in the cloud (nothing renders locally).

A CPU container shallow-clones the public Pages repository at one commit, then runs
web/checks/photo-revision-check.mjs (Chromium, software WebGL) on the published folder
and returns its record and screenshots.

modal run modal_apps/workcell_browser_check.py --commit 7d0ebb9 --folder workcell-photo-direct --out NEW_DIR
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
SITE = 'https://github.com/admin-wekruit/panoptes-workcell-report.git'
RATE = 4 * .0000131 + 8 * .00000222  # 4 CPU, 8 GiB list rate (USD/s); not an invoice

app = modal.App('workcell-browser-check')
image = (modal.Image.from_registry(f'mcr.microsoft.com/playwright:v{PLAYWRIGHT}-noble', add_python='3.11')
         .apt_install('git')
         .run_commands(f'mkdir -p /check && cd /check && npm init -y >/dev/null && npm install --silent playwright@{PLAYWRIGHT}')
         .add_local_file(REPO / 'web/checks/photo-revision-check.mjs', '/check/photo-revision-check.mjs'))


@app.function(image=image, cpu=4, memory=8 * 1024, timeout=1800, retries=0, min_containers=0)
def check(commit: str, folder: str):
    started = time.monotonic()
    site = Path('/site')
    clone = ['git', 'clone', '--filter=blob:none', '--no-checkout', SITE, str(site)]
    subprocess.run(clone, check=True, capture_output=True)
    subprocess.run(['git', '-C', str(site), 'sparse-checkout', 'set', folder], check=True, capture_output=True)
    subprocess.run(['git', '-C', str(site), 'checkout', commit], check=True, capture_output=True)
    head = subprocess.run(['git', '-C', str(site), 'rev-parse', 'HEAD'], check=True, capture_output=True, text=True).stdout.strip()
    out = Path('/out'); out.mkdir()
    result = subprocess.run(['node', '/check/photo-revision-check.mjs', str(site / folder), str(out)], capture_output=True, text=True,
                            env={**os.environ, 'PLAYWRIGHT_BROWSERS_PATH': os.environ.get('PLAYWRIGHT_BROWSERS_PATH', '/ms-playwright'),
                                 'PLAYWRIGHT_FROM': '/check/package.json'}, timeout=1500)
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode='w:gz') as bundle:
        for path in sorted(out.iterdir()):
            if path.suffix in ('.json', '.png'):
                bundle.add(path, arcname=path.name)
    return {'returncode': result.returncode, 'stdout': result.stdout[-4000:], 'stderr': result.stderr[-4000:], 'commit': head,
            'archive': archive.getvalue(), 'containerSeconds': time.monotonic() - started}


@app.local_entrypoint()
def main(commit: str, folder: str, out: str):
    destination = Path(out)
    if destination.exists():
        raise ValueError('Choose a fresh output directory')
    destination.mkdir(parents=True)
    start = time.monotonic()
    result = check.remote(commit, folder)
    with tarfile.open(fileobj=io.BytesIO(result.pop('archive')), mode='r:gz') as bundle:
        bundle.extractall(destination, filter='data')
    ledger = {'mode': 'ephemeral modal run', 'hardware': '4 CPU, 8 GiB, no GPU', 'status': 'completed' if result['returncode'] == 0 else 'failed',
              'functionSeconds': result['containerSeconds'], 'callSeconds': time.monotonic() - start,
              'estimateUsd': RATE * result['containerSeconds'], 'callWindowEstimateUsd': RATE * (time.monotonic() - start),
              'actualBilledUsd': None, 'rateSource': 'https://modal.com/pricing', 'checkedCommit': result['commit']}
    (destination / 'spend-ledger.json').write_text(json.dumps(ledger, indent=2) + '\n')
    (destination / 'check-output.json').write_text(json.dumps({key: result[key] for key in ('returncode', 'stdout', 'stderr', 'commit')}, indent=2) + '\n')
    print(json.dumps({'returncode': result['returncode'], 'commit': result['commit'], 'tail': result['stdout'][-400:] or result['stderr'][-400:]}))
    if result['returncode']:
        raise SystemExit(1)
