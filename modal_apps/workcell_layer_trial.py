"""Run photo checks against a LOCAL (unpublished) measurement layer, in the cloud (CPU only, ephemeral).

load_report (scripts/workcell_shape_check.py) reads the layer and its mesh files from a URL. This app serves a local layer
JSON and its files from memory under a fake page URL instead, so a proposed correction can be checked before it is
published. The work itself is a task script (run(ctx, opts) -> dict, optional 'files': {name: bytes}) sent as text; the
checks package (scripts/workcell_checks) and the shape-check core are importable from it as on the dispatcher.

modal run modal_apps/workcell_layer_trial.py --task TASK.py --view VIEW.json --photos-dir DIR --photo IMAGE_ID=FILE,... \
    --api ORIGIN --out NEW_DIR [--layer LAYER.json --layer-dir DIR_WITH_ITS_FILES] [--opts OPTS.json] [--served-dir DIR]
--served-dir: every file under DIR is served too, at its relative path under the fake page (an unpublished report: --api
https://layer-trial.invalid/report/api with DIR/api/api/assets/<id> = {"url": "/blobs/<sha>"} and DIR/api/blobs/<sha>).
"""
import json
from pathlib import Path
import time

import modal

REPO = Path(__file__).resolve().parents[1]
RATE = 8 * .0000131 + 16 * .00000222  # 8 CPU, 16 GiB list rate (USD/s); not an invoice
BASE = 'https://layer-trial.invalid/report/'  # never resolved: served from memory

app = modal.App('workcell-layer-trial')
image = (modal.Image.debian_slim(python_version='3.11')
         .apt_install('libgl1', 'libgomp1', 'libx11-6')  # open3d's CPU module links them
         .pip_install('numpy<2.3', 'opencv-python-headless==4.10.0.84', 'open3d==0.19.0', 'matplotlib==3.9.2', 'trimesh==4.4.9', 'scipy==1.14.1')
         .add_local_file(REPO / 'scripts/workcell_shape_check.py', '/check/shape_core.py')
         .add_local_dir(REPO / 'scripts/workcell_checks', '/check/workcell_checks', ignore=['__pycache__']))


@app.function(image=image, cpu=8, memory=16 * 1024, timeout=1800, retries=0, min_containers=0)
def run(task: str, view: bytes, photos: dict, layer_name: str, served: dict, api: str, opts: dict) -> dict:
    import io, sys, types, urllib.request
    sys.path.insert(0, '/check')
    import shape_core as wsc
    start = time.monotonic()
    real = urllib.request.urlopen

    def urlopen(url, *a, **k):
        u = url if isinstance(url, str) else url.full_url
        return io.BytesIO(served[u[len(BASE):]]) if u.startswith(BASE) else real(url, *a, **k)

    urllib.request.urlopen = urlopen  # ponytail: load_report fetches with urllib.request.urlopen; nothing else is patched
    ctx = wsc.load_report(view, photos, BASE + layer_name if layer_name else '', api)
    module = types.ModuleType('trial_task'); exec(compile(task, 'trial_task.py', 'exec'), module.__dict__)
    out = module.run(ctx, opts)
    return dict(result=out, layerRevision=(ctx['layer'] or {}).get('revisionId'), layerModels=sorted(((ctx['layer'] or {}).get('models') or {})),
                nativeToMeters=ctx['S'], skipped=ctx['skipped'], containerSeconds=time.monotonic() - start)


@app.local_entrypoint()
def main(task: str, view: str, photos_dir: str, photo: str, api: str, out: str, layer: str = '', layer_dir: str = '', opts: str = '',
         served_dir: str = ''):
    destination = Path(out)
    if destination.exists():
        raise ValueError('Choose a fresh output directory')
    pairs = dict(item.split('=', 1) for item in photo.split(','))
    photos = {image_id: (Path(photos_dir) / name).read_bytes() for image_id, name in pairs.items()}
    served, name = {}, ''
    if served_dir:
        served.update({p.relative_to(served_dir).as_posix(): p.read_bytes() for p in Path(served_dir).rglob('*') if p.is_file()})
    if layer:
        doc = json.loads(Path(layer).read_text()); name = f"measurement-layer/{doc['publicationId']}.json"; served[name] = Path(layer).read_bytes()
        for asset in doc.get('assets') or []:  # files named by the layer, relative to the page as the website serves them
            served[asset['url']] = (Path(layer_dir) / Path(asset['url']).name).read_bytes()
    start = time.monotonic()
    result = run.remote(Path(task).read_text(), Path(view).read_bytes(), photos, name, served, api, json.loads(Path(opts).read_text()) if opts else {})
    destination.mkdir(parents=True)
    for fname, data in (result['result'].pop('files', None) or {}).items():
        (destination / fname).write_bytes(data)
    (destination / 'results.json').write_text(json.dumps(result, indent=1, ensure_ascii=False, default=float) + '\n')
    ledger = {'mode': 'ephemeral modal run', 'hardware': '8 CPU, 16 GiB, no GPU', 'functionSeconds': result['containerSeconds'],
              'callSeconds': time.monotonic() - start, 'estimateUsd': RATE * result['containerSeconds'], 'actualBilledUsd': None,
              'rateSource': 'https://modal.com/pricing'}
    (destination / 'spend-ledger.json').write_text(json.dumps(ledger, indent=2) + '\n')
    print(json.dumps({'layerModels': result['layerModels'], 'layerRevision': result['layerRevision'], **ledger}))
