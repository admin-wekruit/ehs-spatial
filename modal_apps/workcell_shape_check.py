"""Shape check of every model in a published four-view report, run in the cloud (CPU only, ephemeral).

scripts/workcell_shape_check.py does the work; this file feeds it the publication document, the report's photos, its
measurement layer, and the model meshes (downloaded from the publication service and the report website), and returns a
table plus the agreement-versus-depth curves.

modal run modal_apps/workcell_shape_check.py --view VIEW.json --photos-dir DIR --photo IMAGE_ID=FILE ... --layer-url URL --api ORIGIN --out NEW_DIR
    [--extra-masks MASKS.json]   # {entityId: {photoIndex0: polygons}}: masks transferred into photos the report has none for
"""
import json
from pathlib import Path
import time

import modal

REPO = Path(__file__).resolve().parents[1]
RATE = 8 * .0000131 + 16 * .00000222  # 8 CPU, 16 GiB list rate (USD/s); not an invoice

app = modal.App('workcell-shape-check')
image = (modal.Image.debian_slim(python_version='3.11')
         .apt_install('libgl1', 'libgomp1', 'libx11-6')  # open3d's CPU module links them
         .pip_install('numpy<2.3', 'opencv-python-headless==4.10.0.84', 'open3d==0.19.0', 'matplotlib==3.9.2', 'trimesh==4.4.9')
         .add_local_file(REPO / 'scripts/workcell_shape_check.py', '/check/shape_core.py'))  # own name: this app file is also workcell_shape_check.py


@app.function(image=image, cpu=8, memory=16 * 1024, timeout=1800, retries=0, min_containers=0)
def run(view: bytes, photos: dict, layer_url: str, api: str, points: int = 12000, extra_masks: str = ''):
    import hashlib, io, sys, urllib.parse, urllib.request
    import cv2, numpy as np
    sys.path.insert(0, '/check')
    import shape_core as wsc
    start = time.monotonic()
    wsc._check()
    ctx = wsc.load_report(view, photos, layer_url, api)
    cams, images, S, layer = ctx['cams'], ctx['images'], ctx['S'], ctx['layer']
    displayed = {o['id']: o['mesh'] for o in ctx['objects']}
    extra = json.loads(extra_masks) if extra_masks else {}  # {entityId: {photoIndex0: [polygons]}} masks transferred by projection
    scales = np.round(np.linspace(.75, 1.25, 51), 4)
    rows = []
    for o in ctx['objects']:
        e = {'id': o['id'], 'label': o['label']}
        masks = {k: cv2.erode(m.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool) for k, m in o['masks'].items()}
        for k, polygons in (extra.get(o['id']) or {}).items():
            k = int(k); m = wsc.polygon_mask(polygons, (cams[k]['h'], cams[k]['w']))
            m = cv2.erode(m.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool); masks[k] = masks[k] | m if k in masks else m
        others = wsc.raycast_scene([displayed[i] for i in displayed if i != e['id']])
        variants = [('displayed', o['mesh'])] + ([('september', o['original'])] if o['mesh'] is not o['original'] else [])
        for name, (V, F) in variants:
            rng = np.random.default_rng(int(hashlib.sha256((e['id'] + name).encode()).hexdigest()[:8], 16))
            P, N = wsc.sample_surface(V, F, points, rng)
            row = wsc.check_entity(e['id'], P, N, wsc.raycast_scene([(V, F)]), others, cams, images, masks, scales, .004)
            row.update(label=e.get('label'), model=name, maskPhotos=sorted(k + 1 for k in masks), transferredMaskPhotos=sorted(int(k) + 1 for k in (extra.get(o['id']) or {})), verdict=wsc.verdict(row),
                       depthChangeM=(row['bestScale'] - 1) * row['distanceToReference'] * S, referencePhoto=row['referencePhoto'] + 1)
            rows.append(row)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    n = len(rows); cols = 6; fig, axes = plt.subplots(-(-n // cols), cols, figsize=(3.2 * cols, 2.3 * -(-n // cols)), squeeze=False)
    for ax, r in zip(axes.ravel(), rows):
        xs, ys = zip(*r['curve']); ax.plot(xs, ys, color='#2a6f4e' if r['verdict'] == 'ok' else '#c0392b' if r['verdict'] in ('depth_off', 'not_photo_consistent') else '#888')
        ax.axvline(1, color='#999', lw=.6); ax.axhline(r['shiftedControl'], color='#bbb', lw=.6, ls='--')
        ax.set_title(f"{r['label'][:18]}{' (Sept)' if r['model'] == 'september' else ''}\n{r['verdict']} s*={r['bestScale']:.2f}", fontsize=7); ax.tick_params(labelsize=6); ax.set_ylim(-.3, 1)
    for ax in axes.ravel()[n:]:
        ax.axis('off')
    fig.tight_layout(); buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=110)
    return dict(rows=rows, skipped=ctx['skipped'], nativeToMeters=S, layerRevision=(layer or {}).get('revisionId'), containerSeconds=time.monotonic() - start, plot=buf.getvalue())


@app.local_entrypoint()
def main(view: str, photos_dir: str, photo: str, layer_url: str, api: str, out: str, points: int = 12000, extra_masks: str = ''):
    destination = Path(out)
    if destination.exists():
        raise ValueError('Choose a fresh output directory')
    pairs = dict(item.split('=', 1) for item in photo.split(','))
    photos = {image_id: (Path(photos_dir) / name).read_bytes() for image_id, name in pairs.items()}
    start = time.monotonic()
    result = run.remote(Path(view).read_bytes(), photos, layer_url, api, points, Path(extra_masks).read_text() if extra_masks else '')
    destination.mkdir(parents=True)
    (destination / 'curves.png').write_bytes(result.pop('plot'))
    (destination / 'results.json').write_text(json.dumps(result, indent=1, ensure_ascii=False) + '\n')
    ledger = {'mode': 'ephemeral modal run', 'hardware': '8 CPU, 16 GiB, no GPU', 'functionSeconds': result['containerSeconds'],
              'callSeconds': time.monotonic() - start, 'estimateUsd': RATE * result['containerSeconds'], 'actualBilledUsd': None,
              'rateSource': 'https://modal.com/pricing'}
    (destination / 'spend-ledger.json').write_text(json.dumps(ledger, indent=2) + '\n')
    for r in sorted(result['rows'], key=lambda r: (r['verdict'], r['label'])):  # noqa: E501
        print(f"{r['verdict']:22s} {r['label'][:24]:24s} {r['model']:9s} ref {r['referencePhoto']} masks {r['maskPhotos']} "
              f"ncc {r['nccAtModel']:.2f} -> {r['nccAtBest']:.2f} at s {r['bestScale']:.2f} ({r['depthChangeM']:+.2f} m) control {r['shiftedControl']:.2f}")
    for s in result['skipped']:
        print('skipped', s)
