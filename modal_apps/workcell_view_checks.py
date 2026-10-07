"""Generic photo checks of every model in a published four-view report, run in the cloud (CPU only, ephemeral).

Each check is one module in scripts/workcell_checks/ with run(ctx, opts) -> dict and a synthetic _check(); ctx is
scripts/workcell_shape_check.py load_report (cameras, photos, each object's displayed model and masks, scale, floor).
Checks never change the report; they return per-object findings that the measurement layer's confidence reads.

modal run modal_apps/workcell_view_checks.py --checks floor,lines --view VIEW.json --photos-dir DIR --photo IMAGE_ID=FILE,... \
    --layer-url URL --api ORIGIN --out NEW_DIR [--opts OPTS.json]
"""
import importlib
import json
from pathlib import Path
import time

import modal

REPO = Path(__file__).resolve().parents[1]
RATE = 8 * .0000131 + 16 * .00000222  # 8 CPU, 16 GiB list rate (USD/s); not an invoice

app = modal.App('workcell-view-checks')
image = (modal.Image.debian_slim(python_version='3.11')
         .apt_install('libgl1', 'libgomp1', 'libx11-6')  # open3d's CPU module links them
         .pip_install('numpy<2.3', 'opencv-python-headless==4.10.0.84', 'open3d==0.19.0', 'matplotlib==3.9.2', 'trimesh==4.4.9', 'scipy==1.14.1')
         .add_local_file(REPO / 'scripts/workcell_shape_check.py', '/check/shape_core.py')
         .add_local_dir(REPO / 'scripts/workcell_checks', '/check/workcell_checks', ignore=['__pycache__']))


@app.function(image=image, cpu=8, memory=16 * 1024, timeout=1800, retries=0, min_containers=0)
def run(checks: list, view: bytes, photos: dict, layer_url: str, api: str, opts: dict) -> dict:
    import sys
    sys.path.insert(0, '/check')
    import shape_core as wsc
    start = time.monotonic()
    modules = {name: importlib.import_module(f'workcell_checks.{name}') for name in checks}
    for module in modules.values():
        module._check()
    ctx = wsc.load_report(view, photos, layer_url, api)
    out = {name: module.run(ctx, opts.get(name) or {}) for name, module in modules.items()}
    return dict(results=out, objects={o['id']: o['label'] for o in ctx['objects']}, skipped=ctx['skipped'], nativeToMeters=ctx['S'],
                layerRevision=(ctx['layer'] or {}).get('revisionId'), containerSeconds=time.monotonic() - start)


@app.local_entrypoint()
def main(checks: str, view: str, photos_dir: str, photo: str, layer_url: str, api: str, out: str, opts: str = ''):
    destination = Path(out)
    if destination.exists():
        raise ValueError('Choose a fresh output directory')
    pairs = dict(item.split('=', 1) for item in photo.split(','))
    photos = {image_id: (Path(photos_dir) / name).read_bytes() for image_id, name in pairs.items()}
    start = time.monotonic()
    result = run.remote(checks.split(','), Path(view).read_bytes(), photos, layer_url, api, json.loads(Path(opts).read_text()) if opts else {})
    destination.mkdir(parents=True)
    files = {}
    for name, value in result['results'].items():  # a check may return binary attachments (plots) under 'files'
        for fname, data in (value.pop('files', None) or {}).items():
            (destination / f'{name}-{fname}').write_bytes(data); files[f'{name}-{fname}'] = len(data)
    (destination / 'results.json').write_text(json.dumps(result, indent=1, ensure_ascii=False) + '\n')
    ledger = {'mode': 'ephemeral modal run', 'hardware': '8 CPU, 16 GiB, no GPU', 'functionSeconds': result['containerSeconds'],
              'callSeconds': time.monotonic() - start, 'estimateUsd': RATE * result['containerSeconds'], 'actualBilledUsd': None,
              'rateSource': 'https://modal.com/pricing'}
    (destination / 'spend-ledger.json').write_text(json.dumps(ledger, indent=2) + '\n')
    print(json.dumps({'checks': list(result['results']), 'files': files, **ledger}))
