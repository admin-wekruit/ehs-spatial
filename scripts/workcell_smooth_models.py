"""Smooth the displayed models of the well-measured objects of a four-view report (high / medium confidence only).

For each object whose live-layer confidence is high or medium, its displayed mesh (the layer model if there is one, else the
publication asset; scripts/workcell_shape_check.py load_report) is smoothed with Taubin only (open3d, lambda 0.5 / mu -0.53,
positions only), 20 iterations, then 10 if 20 fails. Nothing else: no fragment removal, no decimation, so the vertex count,
the faces and the colours (hence the vertex density) stay those of the displayed model; a vertex no face uses stays put.
The result is accepted only if
- its box in the unified floor frame (u = floor normal, l / w = minimum-area rectangle of the original, heights above the
  report floor) moves by <= 0.5 cm on every side,
- its lowest point as the report publishes it (0.5th percentile of the vertex heights, box_faces / lower_edge modelBottomCm)
  moves by <= 0.3 cm,
- its silhouette IoU against the original model is >= 0.98 in every photo that sees it (full photo resolution).
Otherwise the object stays raw.
Never touched: low and unverified objects (their problems must stay visible); objects a saved bend angle measures (the
layer's 'bends', or a row of the publication's bend analysis {api}/api/revisions/{rev}/bend-analysis-v1 that is not
'unsupported' / 'skipped', with every entity their results reference: the angle was measured on that model); primitives
(already exact); photo-textured GLB surfaces (vertex colours would lose the texture).

Output: panoptes-mesh-v1 assets (entity-local position, normal, colour; uint32 indices) under the unchanged displayed
transform, and one layer patch per object, OUT/<entityId>.json = {"models": {entityId: entry}, "assets": [asset]}, in the
format of the existing layer models, so the publisher keeps any subset. The new representation does not carry what described
the old asset (planProjection, qualityEvidence, familyBinding, source lists). Nothing is published.

modal run scripts/workcell_smooth_models.py --view VIEW.json --photos-dir DIR --photo IMAGE_ID=FILE,... --layer-url URL \
    --api ORIGIN --out NEW_DIR
python scripts/workcell_smooth_models.py      # synthetic self-test (also runs in the container before every job)
"""
from __future__ import annotations

import functools
import hashlib
import json
from pathlib import Path
import sys
import time

import modal

HERE = Path(__file__).resolve().parent
RATE = 8 * .0000131 + 16 * .00000222  # 8 CPU, 16 GiB list rate (USD/s); not an invoice
BOX_TOL_CM, LOW_TOL_CM, LOW_PCT, IOU_MIN, ITERATIONS = .5, .3, .5, .98, (20, 10)
DROP = {'sourceFaceIndices', 'sourceVertexIndices', 'sourceRefs', 'lineage', 'planProjection', 'qualityEvidence', 'familyBinding'}
NO_BEND = ('unsupported', 'skipped')  # bend-analysis rows that measured nothing

app = modal.App('workcell-smooth-models')
image = (modal.Image.debian_slim(python_version='3.11')
         .apt_install('libgl1', 'libgomp1', 'libx11-6')  # open3d's CPU module links them
         .pip_install('numpy<2.3', 'opencv-python-headless==4.10.0.84', 'open3d==0.19.0', 'trimesh==4.4.9', 'scipy==1.14.1')
         .add_local_file(HERE / 'workcell_shape_check.py', '/check/shape_core.py'))


@functools.lru_cache(None)
def core():
    try:
        sys.path.insert(0, '/check'); import shape_core as wsc  # noqa: E401 - container
    except ImportError:
        sys.path.insert(0, str(HERE)); import workcell_shape_check as wsc  # noqa: E401 - local
    return wsc


# ---------------------------------------------------------------- geometry

def floor_axes(V, floor):
    """Rows l, w, u: u = floor normal, l / w along the minimum-area rectangle of the horizontal projection (l the longer)."""
    import cv2
    import numpy as np
    n = floor[0]; e1 = np.cross(n, [1., 0, 0] if abs(n[0]) < .9 else [0, 1., 0]); e1 /= np.linalg.norm(e1); e2 = np.cross(n, e1)
    (_, _), (a, b), ang = cv2.minAreaRect(np.c_[V @ e1, V @ e2].astype(np.float32))
    t = np.radians(ang + (90 if b > a else 0)); l = np.cos(t) * e1 + np.sin(t) * e2
    return np.array([l, np.cross(n, l), n])


def box_cm(V, A, floor, S):
    """[[min l, min w, bottom], [max l, max w, top]] in cm; heights above the floor."""
    import numpy as np
    X = V @ A.T; X[:, 2] += floor[1]
    return np.array([X.min(0), X.max(0)]) * S * 100


def lowest_cm(V, floor, S):
    """The published model lowest point: the 0.5th percentile of the vertex heights above the floor (cm)."""
    import numpy as np
    return float(np.percentile((V @ floor[0] + floor[1]) * S * 100, LOW_PCT))


def o3d_mesh(V, F, C):
    import numpy as np
    import open3d as o3d
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(np.asarray(V, float)), o3d.utility.Vector3iVector(np.asarray(F, np.int32)))
    m.vertex_colors = o3d.utility.Vector3dVector(np.asarray(C, float))
    return m


def smooth(V, F, iterations):
    """Taubin (positions only) on the vertices the faces use; same vertices in the same order, faces untouched."""
    import numpy as np
    import open3d as o3d
    used = np.unique(F); index = np.full(len(V), -1); index[used] = np.arange(len(used))  # an isolated vertex would turn NaN
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V[used]), o3d.utility.Vector3iVector(index[F].astype(np.int32)))
    m = m.filter_smooth_taubin(number_of_iterations=iterations, lambda_filter=.5, mu=-.53, filter_scope=o3d.geometry.FilterScope.Vertex)
    out = V.copy(); out[used] = np.asarray(m.vertices)
    assert np.isfinite(out).all()
    return out


def silhouette(cam, V, F):
    """Union of the projected triangles at full photo resolution (fillPoly with many polygons is even-odd: draw one by one)."""
    import cv2
    import numpy as np
    wsc = core(); uv, z = wsc.project(cam, V); w, h = cam['w'], cam['h']
    T = uv[F]; good = (z[F] > 0).all(1) & ~((T[..., 0] < 0).all(1) | (T[..., 1] < 0).all(1) | (T[..., 0] >= w).all(1) | (T[..., 1] >= h).all(1))
    m = np.zeros((h, w), np.uint8)
    for tri in np.round(np.clip(T[good], -1e6, 1e6) * 8).astype(np.int32):
        cv2.fillConvexPoly(m, tri, 1, cv2.LINE_8, 3)
    return m.astype(bool)


def ious(cams, ref_masks, V, F):
    rows = []
    for k, (cam, ref) in enumerate(zip(cams, ref_masks)):
        if ref.sum() == 0:
            continue
        m = silhouette(cam, V, F)
        rows.append(dict(photo=k + 1, areaPx=int(ref.sum()), iou=round(float((m & ref).sum() / (m | ref).sum()), 4)))
    return rows


def judge(box0, low0, V, A, floor, S, cams, ref_masks, F):
    import numpy as np
    delta = box_cm(V, A, floor, S) - box0; big = float(np.abs(delta).max()); dlow = lowest_cm(V, floor, S) - low0
    rows = ious(cams, ref_masks, V, F); worst = min((r['iou'] for r in rows), default=float('nan'))
    reason = ([f'box moved {big:.2f} cm > {BOX_TOL_CM}'] if big > BOX_TOL_CM else []) + \
        ([f'lowest point moved {dlow:+.2f} cm, more than {LOW_TOL_CM}'] if abs(dlow) > LOW_TOL_CM else []) + \
        ([f'silhouette IoU {worst:.3f} < {IOU_MIN}'] if rows and worst < IOU_MIN else []) + ([] if rows else ['no photo sees the model'])
    return dict(ok=not reason, reason='; '.join(reason), boxDeltaCm=np.round(delta, 3).tolist(), maxBoxDeltaCm=round(big, 3),
                lowestDeltaCm=round(dlow, 3), iou=rows, minIoU=worst)


def sizes(box):
    return dict(L=round(float(box[1, 0] - box[0, 0]), 2), W=round(float(box[1, 1] - box[0, 1]), 2), H=round(float(box[1, 2] - box[0, 2]), 2),
                bottom=round(float(box[0, 2]), 2), top=round(float(box[1, 2]), 2))


def process(V, F, cams, floor, S):
    """One object in native world coordinates -> (result dict, smoothed vertices or None)."""
    A = floor_axes(V, floor); box0 = box_cm(V, A, floor, S); low0 = lowest_cm(V, floor, S); ref = [silhouette(c, V, F) for c in cams]
    res = dict(faces=int(len(F)), vertices=int(len(V)), boxBeforeCm=sizes(box0), lowestBeforeCm=round(low0, 2), attempts=[])
    for it in ITERATIONS:
        V1 = smooth(V, F, it); verdict = judge(box0, low0, V1, A, floor, S, cams, ref, F)
        res['attempts'].append(dict(iterations=it, boxAfterCm=sizes(box_cm(V1, A, floor, S)), lowestAfterCm=round(lowest_cm(V1, floor, S), 2), **verdict))
        if verdict['ok']:
            res.update(accepted=True, iterations=it)
            return res, V1
    res.update(accepted=False, reason=res['attempts'][-1]['reason'])
    return res, None


def bend_entities(layer, analysis):
    """Entities a saved bend angle measures: every layer 'bends' row and every publication bend-analysis row that measured
    something (status not 'unsupported' / 'skipped'), with the entities their results reference."""
    rows = list((layer or {}).get('bends') or []) + [r for r in (analysis or {}).get('items') or [] if r.get('status') not in NO_BEND]
    return {r['entityId'] for r in rows} | {x['entityId'] for r in rows for x in (r.get('result') or {}).get('references') or []}


# ---------------------------------------------------------------- assets

def pack(V, F, C, transform):
    """panoptes-mesh-v1 bytes: entity-local position, normal, colour (float32 x 9) + uint32 indices, under `transform`."""
    import numpy as np
    wsc = core()
    m = o3d_mesh(V, F, C); m.compute_vertex_normals(); N = np.asarray(m.vertex_normals)
    R, s, p = wsc.quat_matrix(transform['quaternion']), np.array(transform['scale'], float), np.array(transform['position'], float)
    local = ((V - p) @ R) / s; ln = (N @ R) * s; ln /= np.linalg.norm(ln, axis=1, keepdims=True) + 1e-12  # inverse-transpose
    assert np.abs(wsc.placed(local, transform) - V).max() < 1e-6
    vertices = np.c_[local, ln, np.clip(C, 0, 1)].astype('<f4'); assert np.isfinite(vertices).all()
    blob = vertices.tobytes() + np.asarray(F, '<u4').tobytes()
    layout = {'stride': 9, 'indexType': 'uint32', 'byteOffset': 0, 'indexCount': int(F.size), 'vertexCount': int(len(V)), 'indexByteOffset': int(vertices.nbytes)}
    return blob, layout, {'min': local.min(0).tolist(), 'max': local.max(0).tolist()}


def displayed(ctx, entity, layer, api):
    """(representation, model entry or None, transform, asset meta, raw rows or None) of what the page shows for `entity`."""
    import io, urllib.parse, urllib.request  # noqa: E401
    import numpy as np
    get = lambda url: urllib.request.urlopen(url, timeout=120).read()  # noqa: E731
    entry = (layer or {}).get('models', {}).get(entity['id'])
    if entry:
        rep = entry['representation']; transform = rep['transform']
    else:
        rep = next(r for r in entity['representations'] if r['id'] == entity['activeModelRepresentationId'])
        transform = entity.get('currentModelTransform') or rep['transform']
    if rep['kind'] != 'generated_mesh':
        return rep, entry, transform, None, None
    layer_assets = {a['id']: a for a in (layer or {}).get('assets', [])}
    meta = layer_assets.get(rep['assetId']) or {a['id']: a for a in ctx['doc']['assets']}[rep['assetId']]
    if rep['assetId'] in layer_assets:
        data = get(urllib.parse.urljoin(ctx['page'], meta['url']))
    else:
        url = json.loads(get(f"{api}/api/assets/{rep['assetId']}"))['url']; data = get(url if url.startswith('http') else api + url)
    fmt = meta.get('format') or (meta.get('metadata') or {}).get('format')
    if fmt == 'panoptes-mesh-v1':
        lay = meta.get('byteLayout') or meta['metadata']['byteLayout']; vc, ic = lay['vertexCount'], lay['indexCount']
        rows = np.frombuffer(data, np.float32, vc * 9, lay.get('byteOffset', 0)).reshape(vc, 9).astype(np.float64)
        F = np.frombuffer(data, np.uint32, ic, lay['indexByteOffset']).reshape(-1, 3).astype(np.int64)
        return rep, entry, transform, meta, (rows[:, :3], rows[:, 6:9], F)
    if fmt == 'glb':  # vertex-coloured GLB is smoothable; a photo-textured one is not (mesh-v1 has no texture)
        import trimesh
        scene = trimesh.load(io.BytesIO(data), file_type='glb', force='scene')
        part = scene.dump(concatenate=True) if not rep.get('node') else scene.geometry[scene.graph[rep['node']][1]].copy().apply_transform(scene.graph[rep['node']][0])
        if getattr(part.visual, 'kind', None) != 'vertex':
            return rep, entry, transform, dict(meta, skip=f'photo-textured glb ({getattr(part.visual, "kind", None)}): vertex colours would lose the texture'), None
        return rep, entry, transform, meta, (np.asarray(part.vertices, float), np.asarray(part.visual.vertex_colors[:, :3], float) / 255, np.asarray(part.faces, np.int64))
    return rep, entry, transform, dict(meta, skip=f'unsupported format {fmt}'), None


def model_entry(rep, entry, transform, asset_id, bounds, frame_id, image_ids, res):
    """Layer model in the format of the existing layer models (030 guard): the displayed representation, new asset, without
    the old asset's descriptions (DROP)."""
    old_source = rep.get('placementSource') or {}
    r = {k: v for k, v in rep.items() if k not in DROP}
    r.update(id='rep-' + asset_id, assetId=asset_id, kind='generated_mesh', primitive=None, bounds=bounds, coordinateFrameId=frame_id,
             transform={**transform, 'coordinateFrameId': frame_id}, placementState='unconfirmed', placementReason='imported_proposal',
             placementSource={'type': 'measurement_layer_smoothed',
                              'sourceRepresentationId': old_source.get('sourceRepresentationId', rep['id']),
                              'sourceAssetId': old_source.get('sourceAssetId', rep['assetId']),
                              'smoothedFrom': {'representationId': rep['id'], 'assetId': rep['assetId'], 'placementSource': old_source or None},
                              'method': f"displayed mesh; Taubin {res['iterations']} iterations (lambda 0.5, mu -0.53, positions only; "
                                        f"vertices, faces and colours unchanged); accepted: box moved {res['maxBoxDeltaCm']:.2f} cm (<= {BOX_TOL_CM}), "
                                        f"lowest point (0.5th percentile) moved {round(res['lowestDeltaCm'], 2) + 0:+.2f} cm (<= {LOW_TOL_CM}), "
                                        f"silhouette IoU >= {res['minIoU']:.3f} in every photo (>= {IOU_MIN})"},
             sourceRefs=[{'role': 'measurement_layer', 'method': 'taubin-smoothing-v2', 'imageIds': image_ids}])
    note = (f"显示模型平滑：Taubin {res['iterations']} 次，只动顶点位置（{res['faces']} 面，顶点数不变）；"
            f"包围盒变化 ≤ {res['maxBoxDeltaCm']:.2f} cm，模型最低点变化 {round(res['lowestDeltaCm'], 2) + 0:+.2f} cm，各照片轮廓 IoU ≥ {res['minIoU']:.3f}")
    old_note = (entry or {}).get('note')
    return {**(entry or {}), 'representation': r, 'note': f'{old_note}；{note}' if old_note else note}


# ---------------------------------------------------------------- job

def smooth_report(view: bytes, photos: dict, layer_url: str, api: str) -> dict:
    import urllib.request
    import numpy as np
    wsc = core(); ctx = wsc.load_report(view, photos, layer_url, api)
    layer = ctx['layer']
    assert layer, 'the live layer must apply to this revision (its confidence decides what is smoothed)'
    rev = json.loads(view)['publication']['sceneRevisionId']
    analysis = json.loads(urllib.request.urlopen(f'{api}/api/revisions/{rev}/bend-analysis-v1', timeout=120).read())
    assert analysis.get('revisionId') == rev, 'the bend analysis is of another revision'
    bent = bend_entities(layer, analysis)
    ctx['page'] = layer_url.rsplit('/measurement-layer/', 1)[0] + '/'
    entities = {e['id']: e for e in ctx['doc']['entities']}; frame_id = ctx['doc']['coordinateFrames'][0]['id']
    image_ids = [c['imageId'] for c in ctx['cams']]; confidence = layer.get('confidence') or {}
    rows, bins, patches = [], {}, {}
    for o in ctx['objects']:
        t0 = time.monotonic(); level = (confidence.get(o['id']) or {}).get('level', 'unverified')
        row = dict(entityId=o['id'], label=o['label'], confidence=level)
        if level not in ('high', 'medium'):
            rows.append(dict(row, action='kept raw', reason=f'confidence {level}: never smoothed')); continue
        if o['id'] in bent:
            rows.append(dict(row, action='skipped', reason='a saved bend angle measures this model: kept as measured')); continue
        rep, entry, transform, meta, raw = displayed(ctx, entities[o['id']], layer, api)
        row['displayed'] = 'layer model' if entry else 'publication asset'
        if raw is None:
            rows.append(dict(row, action='skipped', reason=(meta or {}).get('skip') or f"{rep['kind']}: analytic primitive, already exact")); continue
        local, C, F = raw; V = wsc.placed(local, transform)
        assert len(F) == len(o['mesh'][1]) and len(V) == len(o['mesh'][0]) and np.abs(V - o['mesh'][0]).max() < 1e-6, 'decoded mesh differs from the displayed model'
        res, V1 = process(V, F, ctx['cams'], ctx['floor'], ctx['S'])
        row.update(res); row['seconds'] = round(time.monotonic() - t0, 1)
        if V1 is None:
            rows.append(dict(row, action='kept raw')); continue
        best = res['attempts'][-1]; row.update(maxBoxDeltaCm=best['maxBoxDeltaCm'], lowestDeltaCm=best['lowestDeltaCm'], minIoU=best['minIoU'])
        blob, layout, bounds = pack(V1, F, C, transform)
        sha = hashlib.sha256(blob).hexdigest(); aid = 'layer-' + sha[:24]; fname = f"smooth-{o['id'][:8]}.bin"
        meta_out = {'kind': 'generated_mesh', 'bounds': bounds, 'format': 'panoptes-mesh-v1', 'byteLayout': layout}
        asset = {'id': aid, 'url': f'measurement-layer/{fname}', 'kind': 'generated_mesh', 'sha256': sha, 'sizeBytes': len(blob),
                 'mediaType': 'application/octet-stream', 'format': 'panoptes-mesh-v1', 'byteLayout': layout, 'bounds': bounds, 'metadata': meta_out}
        patches[o['id']] = {'models': {o['id']: model_entry(rep, entry, transform, aid, bounds, frame_id, image_ids, row)}, 'assets': [asset]}
        if entry:  # the asset it replaces, for the publisher to drop if nothing else uses it
            row['replacesLayerAsset'] = rep['assetId']
        bins[fname] = blob; rows.append(dict(row, action='smoothed', patch=f"{o['id']}.json", file=fname, fileBytes=len(blob), assetId=aid))
    return dict(objects=rows, patches=patches, files=bins, nativeToMeters=ctx['S'], bentEntities=sorted(bent),
                floor=dict(normal=ctx['floor'][0].tolist(), offset=float(ctx['floor'][1])), layerRevision=layer.get('revisionId'),
                publicationId=layer.get('publicationId'), skippedByLoader=ctx['skipped'])


@app.function(image=image, cpu=8, memory=16 * 1024, timeout=1800, retries=0, min_containers=0)
def run(view: bytes, photos: dict, layer_url: str, api: str) -> dict:
    start = time.monotonic(); _check()
    out = smooth_report(view, photos, layer_url, api)
    out['containerSeconds'] = time.monotonic() - start
    return out


@app.local_entrypoint()
def main(view: str, photos_dir: str, photo: str, layer_url: str, api: str, out: str):
    destination = Path(out)
    if destination.exists():
        raise ValueError('Choose a fresh output directory')
    photos = {i: (Path(photos_dir) / n).read_bytes() for i, n in (x.split('=', 1) for x in photo.split(','))}
    start = time.monotonic(); result = run.remote(Path(view).read_bytes(), photos, layer_url, api)
    destination.mkdir(parents=True)
    for name, data in result.pop('files').items():
        (destination / name).write_bytes(data)
    for eid, patch in result.pop('patches').items():  # one patch per object: keep any subset
        (destination / f'{eid}.json').write_text(json.dumps(patch, indent=1, ensure_ascii=False) + '\n')
    (destination / 'results.json').write_text(json.dumps(result, indent=1, ensure_ascii=False) + '\n')
    ledger = {'mode': 'ephemeral modal run', 'hardware': '8 CPU, 16 GiB, no GPU', 'functionSeconds': result['containerSeconds'],
              'callSeconds': time.monotonic() - start, 'estimateUsd': RATE * result['containerSeconds'], 'actualBilledUsd': None,
              'rateSource': 'https://modal.com/pricing'}
    (destination / 'spend-ledger.json').write_text(json.dumps(ledger, indent=2) + '\n')
    for r in result['objects']:
        print(f"{r['entityId'][:8]} {r['confidence']:10s} {r['action']:9s} faces {r.get('faces', '')} box {r.get('maxBoxDeltaCm', '')} "
              f"low {r.get('lowestDeltaCm', '')} IoU {r.get('minIoU', '')} {r.get('fileBytes', '')} {r.get('reason', '')} {r['label']}")
    print(json.dumps(ledger))


# ---------------------------------------------------------------- self-test

def _check():
    """A noisy box with a vertex no face uses, seen by two cameras: Taubin keeps the box, the lowest point and the silhouettes;
    vertex count, faces, colours and the unused vertex stay; the packed asset round-trips; a 3 cm spike that smoothing would
    remove is refused (box), so is a 0.4 cm drop (lowest point); bend rows mark their entities; old-asset fields are dropped."""
    import numpy as np
    import open3d as o3d
    wsc = core(); rng = np.random.default_rng(0)
    K = [[900, 0, 320], [0, 900, 240], [0, 0, 1]]

    def cam(C, look):
        z = np.asarray(look, float) - C; z /= np.linalg.norm(z); x = np.cross(z, [0, 0, 1.]); x /= np.linalg.norm(x); y = np.cross(z, x)
        M = np.eye(4); M[:3, :3] = np.c_[x, y, z]; M[:3, 3] = C
        return wsc.camera(dict(cameraToWorld=M.tolist(), K=K, width=640, height=480, imageId=str(C)))
    cams = [cam(np.array([2.5, -1.5, 1.2]), [0, 0, .35]), cam(np.array([-1.5, -2.5, 1.0]), [0, 0, .35])]
    floor = (np.array([0, 0, 1.]), 0.)  # height = z
    box = o3d.geometry.TriangleMesh.create_box(.8, .5, .6).subdivide_midpoint(6)  # 49k triangles, welded
    box.translate([-.4, -.25, .1]); F = np.asarray(box.triangles)
    V = np.r_[np.asarray(box.vertices) + rng.normal(0, .0005, (len(box.vertices), 3)), [[0, 0, .4]]]  # last: used by no face
    C = rng.random((len(V), 3))
    # silhouettes are unions: two overlapping triangles drawn together must not cancel (cv2.fillPoly is even-odd)
    tri = np.array([[-.3, 0, .2], [.3, 0, .2], [0, 0, .7]]); two = np.r_[tri, tri + [.05, 0, 0]]
    assert silhouette(cams[0], two, np.array([[0, 1, 2], [3, 4, 5]])).sum() >= silhouette(cams[0], tri, np.array([[0, 1, 2]])).sum()
    res, V1 = process(V, F, cams, floor, 1.0)
    assert res['accepted'] and res['iterations'] == 20, res
    a = res['attempts'][-1]; assert a['maxBoxDeltaCm'] <= BOX_TOL_CM and abs(a['lowestDeltaCm']) <= LOW_TOL_CM and a['minIoU'] >= IOU_MIN and len(a['iou']) == 2, a
    assert V1.shape == V.shape and (V1[-1] == V[-1]).all() and np.abs(V1[:-1] - V[:-1]).max() > 1e-4, 'smoothed in place, unused vertex kept'
    # asset round trip under a non-uniform transform: same vertices, faces and colours
    tr = dict(position=[.1, -.2, .3], quaternion=[.1, .2, .3, (1 - .14) ** .5], scale=[1.3, .8, 1.1])
    blob, lay, _ = pack(V1, F, C, tr); Vr, Fr = wsc.read_packed(blob, lay)
    assert lay['vertexCount'] == len(V) and np.abs(wsc.placed(Vr, tr) - V1).max() < 1e-5 and (Fr == F).all()
    rows = np.frombuffer(blob, np.float32, lay['vertexCount'] * 9).reshape(-1, 9); assert np.allclose(rows[:, 6:], C, atol=1e-6)
    # a thin 3 cm spike sets the top; smoothing pulls it in -> refused (the box would change)
    Vs = np.asarray(box.vertices).copy(); top = np.argmax(Vs[:, 2] - np.abs(Vs[:, 0]) - np.abs(Vs[:, 1])); Vs[top, 2] += .03
    res2, out2 = process(Vs, F, cams, floor, 1.0)
    assert not res2['accepted'] and out2 is None and 'box moved' in res2['reason'], res2
    # the lowest-point gate alone: the model 0.4 cm lower moves the box by 0.4 cm (allowed) and the lowest point by 0.4 cm (not)
    A = floor_axes(V, floor); ref = [silhouette(c, V, F) for c in cams]
    v = judge(box_cm(V, A, floor, 1.), lowest_cm(V, floor, 1.), V - [0, 0, .004], A, floor, 1., cams, ref, F)
    assert not v['ok'] and v['reason'].startswith('lowest point') and v['maxBoxDeltaCm'] <= BOX_TOL_CM, v
    # bend rows: layer rows and measured / failed analysis rows mark their entity and its references; unsupported does not
    layer = {'bends': [{'entityId': 'a', 'status': 'measured', 'result': {'references': [{'entityId': 'b'}]}}]}
    analysis = {'items': [{'entityId': 'c', 'status': 'measured', 'result': {'references': [{'entityId': 'c'}]}},
                          {'entityId': 'd', 'status': 'unsupported', 'result': None}, {'entityId': 'e', 'status': 'failed'}]}
    assert bend_entities(layer, analysis) == {'a', 'b', 'c', 'e'}, bend_entities(layer, analysis)
    # the new representation does not carry the old asset's descriptions
    old = dict(id='r0', assetId='x0', kind='generated_mesh', transform=tr, planProjection={'outline': []}, familyBinding={'family': 'f'},
               qualityEvidence={'familyBinding': {'family': 'f'}}, qualityReviewRefs=['q0'])
    e = model_entry(old, None, tr, 'layer-y', {}, 'frame', ['img'], dict(res, maxBoxDeltaCm=a['maxBoxDeltaCm'], lowestDeltaCm=a['lowestDeltaCm'], minIoU=a['minIoU']))
    assert not {'planProjection', 'qualityEvidence', 'familyBinding'} & set(e['representation']) and 'familyBinding' not in json.dumps(e) \
        and e['representation']['assetId'] == 'layer-y', e
    print('smooth self-test passed:', res['faces'], 'faces kept; box', a['maxBoxDeltaCm'], 'cm; lowest', a['lowestDeltaCm'], 'cm; IoU', a['minIoU'],
          '| spike refused:', res2['reason'], '| drop refused:', v['reason'])


if __name__ == '__main__':
    _check()
