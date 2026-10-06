"""Method B clearance heights (light-curtain housing bottoms, fence panel bottom edges) for a published four-view report.

Code: research-notes/workcell-clearance-b-2026-10-05/clearance_b.py (run(ctx, opts)); ctx = scripts/workcell_shape_check.py
load_report plus the RGB photos and the run's Pi3X point maps (geometry/frames/frame_000k). Ephemeral CPU run, nothing deployed.

modal run modal_apps/workcell_clearance_b.py --view VIEW.json --photos-dir DIR --photo IMAGE_ID=FILE,... --layer-url URL \
    --api ORIGIN --run-dir RUN --targets ID=post,ID=panel,... --out NEW_DIR
"""
import io
import json
from pathlib import Path
import time

import modal

REPO = Path(__file__).resolve().parents[1]
NOTE = Path('/Users/adam/Desktop/panoptes-public/research-notes/workcell-clearance-b-2026-10-05/clearance_b.py')
RATE = 8 * .0000131 + 16 * .00000222  # 8 CPU, 16 GiB list rate (USD/s); not an invoice

app = modal.App('workcell-clearance-b')
image = (modal.Image.debian_slim(python_version='3.11')
         .apt_install('libgl1', 'libgomp1', 'libx11-6')
         .pip_install('numpy<2.3', 'opencv-python-headless==4.10.0.84', 'open3d==0.19.0', 'trimesh==4.4.9', 'scipy==1.14.1')
         .add_local_file(REPO / 'scripts/workcell_shape_check.py', '/check/shape_core.py')
         .add_local_file(NOTE, '/check/clearance_b.py'))


@app.function(image=image, cpu=8, memory=16 * 1024, timeout=1800, retries=0, min_containers=0)
def run(view: bytes, photos: dict, layer_url: str, api: str, pi3x: bytes, opts: dict):
    import sys
    sys.path.insert(0, '/check')
    import cv2
    import numpy as np
    import clearance_b as cb
    import shape_core as wsc
    start = time.monotonic()
    cb._check()
    ctx = wsc.load_report(view, photos, layer_url, api)
    ctx['rgb'] = [cv2.cvtColor(cv2.imdecode(np.frombuffer(photos[c['imageId']], np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(np.float32)
                  for c in ctx['cams']]
    z = np.load(io.BytesIO(pi3x))
    ctx['pi3x'] = {k: dict(pts3d=z[f'pts3d{k}'], conf=z[f'conf{k}'], valid=z[f'valid{k}'], content=z[f'content{k}']) for k in range(len(ctx['cams']))}
    res, details = cb.run(ctx, opts)
    files = {}
    for oid, per in details.items():
        o = next(x for x in ctx['objects'] if x['id'] == oid)
        for k, item in per.items():
            files[f'{oid[:8]}-p{k + 1}.jpg'] = cb.overlay(ctx['rgb'][k], o['masks'][k], item, res['objects'][oid]['kind'])
    res.update(containerSeconds=time.monotonic() - start, layerRevision=(ctx['layer'] or {}).get('revisionId'), skipped=ctx['skipped'],
               labels={o['id']: o['label'] for o in ctx['objects']})
    return res, files


@app.local_entrypoint()
def main(view: str, photos_dir: str, photo: str, layer_url: str, api: str, run_dir: str, targets: str, out: str, sweep: int = 1):
    import numpy as np
    destination = Path(out)
    if destination.exists():
        raise ValueError('Choose a fresh output directory')
    pairs = dict(item.split('=', 1) for item in photo.split(','))
    photos = {image_id: (Path(photos_dir) / name).read_bytes() for image_id, name in pairs.items()}
    doc = json.loads(Path(view).read_text())['publication']['snapshot']['revision']['document']
    arrays = {}
    for k in range(len(doc['cameras'])):
        f = Path(run_dir) / 'geometry/frames' / f'frame_{k + 1:04d}'
        for key, name in (('pts3d', 'pts3d'), ('conf', 'conf'), ('valid', 'valid_mask'), ('content', 'content_valid_mask')):
            arrays[f'{key}{k}'] = np.load(f / f'{name}.npy')
    buf = io.BytesIO(); np.savez_compressed(buf, **arrays)
    ids = {e['id'][:8]: e['id'] for e in doc['entities']}
    opts = dict(targets={ids[t.split('=')[0][:8]]: t.split('=')[1] for t in targets.split(',')}, sweep=bool(sweep))
    start = time.monotonic()
    res, files = run.remote(Path(view).read_bytes(), photos, layer_url, api, buf.getvalue(), opts)
    destination.mkdir(parents=True)
    (destination / 'spend-ledger.json').write_text(json.dumps({'functionSeconds': res['containerSeconds'], 'callSeconds': time.monotonic() - start,
                                                               'estimateUsd': RATE * res['containerSeconds']}) + '\n')
    for name, data in files.items():
        (destination / name).write_bytes(data)
    (destination / 'results.json').write_text(json.dumps(res, indent=1, ensure_ascii=False, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o)) + '\n')
    ledger = {'mode': 'ephemeral modal run', 'hardware': '8 CPU, 16 GiB, no GPU', 'functionSeconds': res['containerSeconds'],
              'callSeconds': time.monotonic() - start, 'estimateUsd': RATE * res['containerSeconds'], 'actualBilledUsd': None,
              'rateSource': 'https://modal.com/pricing'}
    (destination / 'spend-ledger.json').write_text(json.dumps(ledger, indent=2) + '\n')
    print(json.dumps({'files': len(files), **ledger}))
