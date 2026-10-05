"""Shape check of every model in a published four-view report, run in the cloud (CPU only, ephemeral).

scripts/workcell_shape_check.py does the work; this file feeds it the publication document, the report's photos, its
measurement layer, and the model meshes (downloaded from the publication service and the report website), and returns a
table plus the agreement-versus-depth curves.

modal run modal_apps/workcell_shape_check.py --view VIEW.json --photos-dir DIR --photo IMAGE_ID=FILE ... --layer-url URL --api ORIGIN --out NEW_DIR
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
def run(view: bytes, photos: dict, layer_url: str, api: str, points: int = 12000):
    import hashlib, io, sys, urllib.parse, urllib.request
    import cv2, numpy as np
    sys.path.insert(0, '/check')
    import shape_core as wsc
    start = time.monotonic()
    wsc._check()
    get = lambda url: urllib.request.urlopen(url, timeout=120).read()
    doc = json.loads(view)['publication']['snapshot']['revision']['document']
    layer = json.loads(get(layer_url)) if layer_url else None
    if layer and layer.get('revisionId') not in (None, json.loads(view)['publication']['sceneRevisionId']):
        layer = None
    page = layer_url.rsplit('/measurement-layer/', 1)[0] + '/' if layer else ''
    assets = {a['id']: a for a in doc['assets']}
    layer_assets = {a['id']: a for a in (layer or {}).get('assets', [])}
    cache = {}

    def asset_bytes(asset_id):
        if asset_id not in cache:
            if asset_id in layer_assets:
                cache[asset_id] = get(urllib.parse.urljoin(page, layer_assets[asset_id]['url']))
            else:
                url = json.loads(get(f'{api}/api/assets/{asset_id}'))['url']
                cache[asset_id] = get(url if url.startswith('http') else api + url)
        return cache[asset_id]

    def mesh(entity, rep):
        transform = entity.get('currentModelTransform') if rep is None or rep.get('id') == entity.get('activeModelRepresentationId') else None
        transform = transform or rep['transform']
        if rep['kind'] == 'primitive':
            V, F = wsc.primitive_mesh(rep['primitive'])
        else:
            meta = layer_assets.get(rep['assetId']) or assets[rep['assetId']]
            fmt = meta.get('format') or (meta.get('metadata') or {}).get('format')
            if fmt == 'panoptes-mesh-v1':
                V, F = wsc.read_packed(asset_bytes(rep['assetId']), meta.get('byteLayout') or meta['metadata']['byteLayout'])
            elif fmt == 'glb':  # as the viewer: one node when the representation names it, else every mesh in the file
                import trimesh
                scene = trimesh.load(io.BytesIO(asset_bytes(rep['assetId'])), file_type='glb', force='scene')
                if rep.get('node'):
                    matrix, geometry = scene.graph[rep['node']]; part = scene.geometry[geometry].copy(); part.apply_transform(matrix)
                else:
                    part = scene.dump(concatenate=True)
                V, F = np.asarray(part.vertices, float), np.asarray(part.faces, np.int64)
            else:
                raise ValueError(f'unsupported_format:{fmt}')
        return wsc.placed(V, transform), F

    cams = [wsc.camera(c) for c in doc['cameras']]
    images = []
    for c in cams:
        g = cv2.imdecode(np.frombuffer(photos[c['imageId']], np.uint8), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        assert g.shape == (c['h'], c['w']), (g.shape, c['h'], c['w'])
        images.append(cv2.GaussianBlur(g, (0, 0), 1.0))
    objects = [e for e in doc['entities'] if not e.get('sourceContext') and e.get('visible') is not False and e.get('activeModelRepresentationId')]
    displayed, original, skipped = {}, {}, []
    for e in objects:
        rep = next(r for r in e['representations'] if r['id'] == e['activeModelRepresentationId'])
        if rep['kind'] not in ('generated_mesh', 'primitive'):
            continue
        try:
            original[e['id']] = mesh(e, rep)
            over = (layer or {}).get('models', {}).get(e['id'])
            displayed[e['id']] = mesh({**e, 'currentModelTransform': over['representation']['transform']}, over['representation']) if over else original[e['id']]
        except Exception as error:  # noqa: BLE001 - reported per object
            skipped.append(dict(entityId=e['id'], label=e.get('label'), reason=str(error)[:200]))
    observations = {o['id']: o for o in doc['observations']}
    index = {c['imageId']: k for k, c in enumerate(cams)}
    S = next((f.get('scale') or {}).get('nativeToMeters') for f in doc['coordinateFrames']) or 1
    if layer:
        S = layer['scale']['nativeToMeters']
    scales = np.round(np.linspace(.75, 1.25, 51), 4)
    rows = []
    for e in objects:
        if e['id'] not in displayed:
            continue
        masks = {}
        for oid in e.get('observationRefs') or []:
            o = observations.get(oid)
            if o and o['imageId'] in index and o.get('originalPixelPolygons'):
                k = index[o['imageId']]; m = wsc.polygon_mask(o['originalPixelPolygons'], (cams[k]['h'], cams[k]['w']))
                m = cv2.erode(m.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
                masks[k] = masks[k] | m if k in masks else m
        others = wsc.raycast_scene([displayed[i] for i in displayed if i != e['id']])
        variants = [('displayed', displayed[e['id']])] + ([('september', original[e['id']])] if displayed[e['id']] is not original[e['id']] else [])
        for name, (V, F) in variants:
            rng = np.random.default_rng(int(hashlib.sha256((e['id'] + name).encode()).hexdigest()[:8], 16))
            P, N = wsc.sample_surface(V, F, points, rng)
            row = wsc.check_entity(e['id'], P, N, wsc.raycast_scene([(V, F)]), others, cams, images, masks, scales, .004)
            row.update(label=e.get('label'), model=name, maskPhotos=sorted(k + 1 for k in masks), verdict=wsc.verdict(row),
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
    return dict(rows=rows, skipped=skipped, nativeToMeters=S, layerRevision=(layer or {}).get('revisionId'), containerSeconds=time.monotonic() - start, plot=buf.getvalue())


@app.local_entrypoint()
def main(view: str, photos_dir: str, photo: str, layer_url: str, api: str, out: str, points: int = 12000):
    destination = Path(out)
    if destination.exists():
        raise ValueError('Choose a fresh output directory')
    pairs = dict(item.split('=', 1) for item in photo.split(','))
    photos = {image_id: (Path(photos_dir) / name).read_bytes() for image_id, name in pairs.items()}
    start = time.monotonic()
    result = run.remote(Path(view).read_bytes(), photos, layer_url, api, points)
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
