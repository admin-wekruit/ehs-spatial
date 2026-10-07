"""Generic fixes of the obvious display errors that scripts/workcell_checks/obvious_errors.py finds, as a measurement-layer patch.

Only the DISPLAY changes, only in these ways; no other geometry is touched and nothing is guessed:
F6 (fragments only)  when G6's fragments are the object's only obvious photo-fit / mesh / size failure, exactly the fragment
              components G6 counts (outside the object's box + 5 cm) are dropped from the displayed mesh; every other vertex,
              face and colour stays (a new panoptes-mesh-v1 asset under the same transform). Deviation from the lead's F6 (box):
              the body is right, and a gravity-aligned box of a tilted panel is a large slab (090 central guard panel 1.5 x 1.0
              x 0.27 m), worse than the fragments. A photo-textured GLB (vertex colours would lose the texture) -> flagged.
F5 / F6 / F7  an object with an obvious G5 (photo fit), G6 (exploded; fragments together with another failure) or G7 (L / H vs
              a verified box dimension) failure is displayed as its box_faces box: a 12-triangle box mesh (panoptes-mesh-v1, grey) at the box's centre,
              axes and size, labelled '模型不可信：显示为测量盒' (representation label and model note). Only an uncapped box is a
              measurement of the photos; a capped box (box_faces R3: the displayed model's own box), no box, or a displayed
              model a saved bend angle was measured on (layer 'bends', the publication's bend analysis) -> flagged only.
F2 / F3       the displayed model (or its box) is translated along the floor normal until its robust bottom (0.5th percentile
              of the vertex heights) is 0: a G2 sinker (< -2 cm) up, a G3 floor-contact floater (> 3 cm) down.
F4            each obvious G4 pair left after F2-F7: when one model is clearly misplaced (its best photo IoU >= 0.15 below the
              other's), it moves by the smallest translation (1 cm steps, <= 30 cm) along one of its box axes (+-l, +-w, +u;
              never down) that brings the overlap to <= 10 % of the smaller, keeps its bottom within the G2 / G3 limits and its
              best photo IoU within 0.02 of before; otherwise the pair is flagged.
Then every gate runs again on the patched display (in memory). Output files (a workcell_layer_trial.py task's 'files'):
patch.json = {"models": {entityId: entry}, "assets": [asset]} in the existing layer format (030 guard / smoothing patches),
fixbox-<id8>.bin (box meshes), trial-layer.json = the loaded layer with the patch merged, fix-results.json (before / after
gates and every action). A patched entry keeps the layer entry it replaces (note continued); a publication model becomes a new
layer entry without the old asset's descriptions (plan projection, quality evidence, source lists). Nothing is published.

modal run modal_apps/workcell_layer_trial.py --task scripts/workcell_fix_obvious.py --view VIEW.json --photos-dir DIR \
    --photo IMAGE_ID=FILE,... --api ORIGIN --layer LIVE_LAYER.json --layer-dir ITS_FILES --opts OPTS.json ({"api": ORIGIN}) --out NEW_DIR
python scripts/workcell_fix_obvious.py      # synthetic self-test"""
import copy
import hashlib
import json
import sys

import numpy as np

try:
    import shape_core as wsc  # the trial / check container
except ImportError:  # locally
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import workcell_shape_check as wsc
from workcell_checks import obvious_errors as oe

BOX_LABEL = '模型不可信：显示为测量盒'
IOU_MARGIN, STEP_M, MAX_SHIFT_M, IOU_KEEP, GREY = .15, .01, .30, .02, .62
DROP = {'sourceFaceIndices', 'sourceVertexIndices', 'sourceRefs', 'lineage', 'planProjection', 'qualityEvidence', 'familyBinding', 'bounds'}
REPLACE_GATES = ('G5', 'G6', 'G7')


def box_mesh(rec, S):
    """box_faces record -> (local vertices 24, faces 12, local normals, transform): centre, rotation (columns l, w, u), unit scale."""
    from scipy.spatial.transform import Rotation
    R = np.array(rec['axes'], float).T; assert np.linalg.det(R) > .99, 'box axes must be right-handed and orthonormal'
    half = np.array(rec['sizeM'], float) / S / 2; V, N, F = [], [], []
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        for s in (1, -1):
            quad = [(-1, -1), (1, -1), (1, 1), (-1, 1)][::s]; base = len(V)
            for a, b in quad:
                p = np.zeros(3); p[i], p[j], p[k] = s * half[i], a * half[j], b * half[k]; V.append(p); N.append(np.eye(3)[i] * s)
            F += [[base, base + 1, base + 2], [base, base + 2, base + 3]]
    transform = dict(position=[float(x) for x in rec['centerNative']], quaternion=[float(x) for x in Rotation.from_matrix(R).as_quat()], scale=[1., 1., 1.])
    return np.array(V), np.array(F, np.int64), np.array(N), transform


def pack(V, F, N, colour, colours=None):
    """panoptes-mesh-v1 bytes (local position, normal, colour as float32 x 9; uint32 indices), byteLayout, local bounds."""
    rows = np.c_[V, N, np.tile(colour, (len(V), 1)) if colours is None else colours].astype('<f4'); blob = rows.tobytes() + np.asarray(F, '<u4').tobytes()
    layout = {'stride': 9, 'indexType': 'uint32', 'byteOffset': 0, 'indexCount': int(F.size), 'vertexCount': int(len(V)), 'indexByteOffset': int(rows.nbytes)}
    return blob, layout, {'min': V.min(0).tolist(), 'max': V.max(0).tolist()}


def displayed_entry(ctx, eid):
    """(entry, representation, where) of what the page displays for eid: a copy of the layer model, else a new entry from the
    publication's active representation (its current transform, without the old asset's descriptions)."""
    layer = ctx.get('layer') or {}
    if eid in (layer.get('models') or {}):
        entry = copy.deepcopy(layer['models'][eid]); return entry, entry['representation'], 'layer model'
    e = next(x for x in ctx['doc']['entities'] if x['id'] == eid); rep = next(r for r in e['representations'] if r['id'] == e['activeModelRepresentationId'])
    r = {k: copy.deepcopy(v) for k, v in rep.items() if k not in DROP}; r['transform'] = copy.deepcopy(e.get('currentModelTransform') or rep['transform'])
    r.update(placementState='unconfirmed', placementReason='imported_proposal')
    return {'representation': r}, r, 'publication'


def raw_rows(ctx, rep, opts):
    """(rows: local position, normal, colour as floats x 9, faces) of the displayed mesh asset, or None (primitive, textured GLB).
    Layer assets resolve against opts['page'] (the report page the layer sits under), publication assets via opts['api']."""
    import io, urllib.parse, urllib.request  # noqa: E401
    if rep['kind'] != 'generated_mesh':
        return None
    lay = {a['id']: a for a in (ctx.get('layer') or {}).get('assets') or []}; meta = lay.get(rep['assetId']) or {a['id']: a for a in ctx['doc']['assets']}[rep['assetId']]
    data = (urllib.request.urlopen(urllib.parse.urljoin(opts['page'], meta['url']), timeout=120).read() if rep['assetId'] in lay
            else oe.fetch(opts['api'], rep['assetId']))
    fmt = meta.get('format') or (meta.get('metadata') or {}).get('format')
    if fmt == 'panoptes-mesh-v1':
        L = meta.get('byteLayout') or meta['metadata']['byteLayout']; vc, ic = L['vertexCount'], L['indexCount']
        return (np.frombuffer(data, np.float32, vc * 9, L.get('byteOffset', 0)).reshape(vc, 9).astype(np.float64),
                np.frombuffer(data, np.uint32, ic, L['indexByteOffset']).reshape(-1, 3).astype(np.int64))
    if fmt == 'glb' and not rep.get('node'):
        import trimesh
        m = trimesh.load(io.BytesIO(data), file_type='glb', force='scene').dump(concatenate=True)
        if getattr(m.visual, 'kind', None) == 'vertex':
            return np.c_[m.vertices, m.vertex_normals, np.asarray(m.visual.vertex_colors[:, :3], float) / 255], np.asarray(m.faces, np.int64)
    return None


def bend_assets(ctx, api):
    """Assets a saved bend angle was measured on: the layer's 'bends' and the measured rows of the publication's bend analysis."""
    import urllib.request
    layer = ctx.get('layer') or {}; rows = list(layer.get('bends') or []); rev = layer.get('revisionId')
    if api and rev:
        got = json.loads(urllib.request.urlopen(f'{api}/api/revisions/{rev}/bend-analysis-v1', timeout=120).read())
        assert got.get('revisionId') == rev, 'bend analysis of another revision'
        rows += [r for r in got.get('items') or [] if r.get('status') == 'measured']
    return {x.get('assetId') for r in rows for x in (r.get('result') or {}).get('references') or []}


def best_iou(ious):
    return max(ious.values()) if ious else None


def bottom_cm(ctx, V):
    n, d = ctx['floor']
    return float(np.percentile((V @ n + d) * ctx['S'] * 100, .5))


def search_shift(ctx, x, other, axes, ious_of, contact):
    """F4: smallest translation of object x (dict with id, mesh) along +-l, +-w, +u of its box axes (STEP_M steps up to MAX_SHIFT_M)
    that brings the overlap with `other` to <= oe.OVERLAP; bottom stays within the G2 / G3 limits, best IoU within IOU_KEEP.
    Returns (shift native vector, cm, axis name, overlap ratio, IoU after) or None."""
    S = ctx['S']; A = axes[x['id']]; sx, so = oe.solid(*x['mesh'], A, S), oe.solid(*other['mesh'], axes[other['id']], S)
    small_is_x = sx['volume'] <= so['volume']; vol = min(sx['volume'], so['volume']); b0 = bottom_cm(ctx, x['mesh'][0])
    if vol <= 0:
        return None
    cands = []
    for name, u in (('+l', A[0]), ('-l', -A[0]), ('+w', A[1]), ('-w', -A[1]), ('+u', A[2])):
        for i in range(1, int(round(MAX_SHIFT_M / STEP_M)) + 1):
            t = i * STEP_M; s = u * t / S; b = b0 + (100 * t if name == '+u' else 0.)
            if b < -oe.SINK_CM or (contact and b > oe.FLOAT_CM):
                break
            ov = oe.overlap(sx, so, S, offset=s) if small_is_x else oe.overlap(so, sx, S, offset=-s)  # x moved by s
            if ov / vol <= oe.OVERLAP:
                cands.append((t, name, s, ov / vol)); break
    before = best_iou(ious_of(x['mesh']))
    for t, name, s, r in sorted(cands, key=lambda c: c[0]):
        after = best_iou(ious_of((x['mesh'][0] + s, x['mesh'][1])))
        if before is None or (after is not None and after >= before - IOU_KEEP):
            return s, round(100 * t, 1), name, round(r, 3), after
    return None


def fix(ctx, opts, before=None):
    """Plan and apply F2-F7 on ctx (in place: the displayed meshes), return (patch, files, actions, before)."""
    n = ctx['floor'][0]; S = ctx['S']; before = before or oe.run(ctx, opts)
    boxes = (ctx.get('layer') or {}).get('boxes') or {}; objs = {o['id']: o for o in ctx['objects']}; fid = ctx['doc']['coordinateFrames'][0]['id']
    patch, files, actions, notes = {'models': {}, 'assets': []}, {}, [], {}
    axes_of = lambda eid: np.array(boxes[eid]['axes'], float) if eid in boxes else oe.floor_axes(objs[eid]['mesh'][0], n)  # noqa: E731
    image_ids = [c['imageId'] for c in ctx['cams']]; bent = bend_assets(ctx, opts.get('api'))

    def entry(eid):
        if eid not in patch['models']:
            patch['models'][eid] = displayed_entry(ctx, eid)[0]
        return patch['models'][eid]

    def translate(eid, cm, fix_id, why_en, why_zh):
        o = objs[eid]; s = n * (cm / 100 / S) if isinstance(cm, float) else cm[0]; dist = cm if isinstance(cm, float) else cm[1]
        o['mesh'] = (o['mesh'][0] + s, o['mesh'][1]); en = entry(eid); r = en['representation']
        r['transform'] = dict(r['transform'], position=[float(a + b) for a, b in zip(r['transform']['position'], s)])
        old = r.get('placementSource'); r['id'] = 'rep-fix-' + hashlib.sha256(json.dumps([eid, r['transform']], sort_keys=True).encode()).hexdigest()[:24]
        r['placementSource'] = {'type': 'measurement_layer_obvious_fix', 'fix': fix_id, 'translationNative': [float(x) for x in s], 'method': why_en,
                                'previousPlacementSource': old}
        notes.setdefault(eid, []).append(why_zh); actions.append(dict(entityId=eid, label=o['label'], fix=fix_id, translatedCm=round(float(np.linalg.norm(s)) * S * 100, 2), why=why_en))

    def add_asset(eid, blob, layout, bounds, prefix):
        sha = hashlib.sha256(blob).hexdigest(); aid = 'layer-' + sha[:24]; fname = f'{prefix}-{eid[:8]}.bin'
        meta = {'kind': 'generated_mesh', 'bounds': bounds, 'format': 'panoptes-mesh-v1', 'byteLayout': layout}
        patch['assets'].append({'id': aid, 'url': f'measurement-layer/{fname}', 'kind': 'generated_mesh', 'sha256': sha, 'sizeBytes': len(blob),
                                'mediaType': 'application/octet-stream', 'format': 'panoptes-mesh-v1', 'byteLayout': layout, 'bounds': bounds, 'metadata': meta})
        files[fname] = blob
        return aid, fname

    # F5 / F6 / F7: replace by the measured box (F6 fragments only: drop them)
    for eid, row in before['objects'].items():
        bad = [f for f in row['failed'] if f['gate'] in REPLACE_GATES and f['severity'] == 'obvious']
        if not bad:
            continue
        if all(f['gate'] == 'G6' and 'fragmentFaces' in f for f in bad):
            old, oldr, where = displayed_entry(ctx, eid); raw = raw_rows(ctx, oldr, opts) if oldr.get('assetId') not in bent else None
            if raw is None:
                actions.append(dict(entityId=eid, label=row['label'], fix='flag', gates=['G6'], why='fragments, but the displayed model is a primitive, '
                                    'a photo-textured GLB or a saved bend\'s model: not changed'))
                continue
            rows, F = raw; V, F0 = objs[eid]['mesh']; assert len(rows) == len(V) and len(F) == len(F0), 'decoded mesh differs from the displayed model'
            frac, _, nfrag, drop = oe.fragments(V, F, axes_of(eid), S, boxes.get(eid), faces_out=True)
            keep = F[~drop]; used = np.unique(keep); index = np.full(len(rows), -1); index[used] = np.arange(len(used)); F1 = index[keep]
            blob, layout, bounds = pack(rows[used, :3], F1, rows[used, 3:6], None, rows[used, 6:9])
            aid, fname = add_asset(eid, blob, layout, bounds, 'fixfrag')
            r = {k: copy.deepcopy(v) for k, v in oldr.items() if k not in DROP}
            r.update(id='rep-' + aid, assetId=aid, kind='generated_mesh', primitive=None, bounds=bounds, placementState='unconfirmed', placementReason='imported_proposal',
                     placementSource={'type': 'measurement_layer_obvious_fix', 'fix': 'F6', 'from': {'representationId': oldr['id'], 'assetId': oldr.get('assetId'), 'displayed': where},
                                      'method': f'displayed mesh without the {nfrag} fragment components outside its box + {100 * oe.GROW_M:.0f} cm '
                                                f'({100 * frac:.1f} % of the faces); every other vertex, face and colour unchanged'})
            patch['models'][eid] = {**old, 'representation': r}; objs[eid]['mesh'] = (V[used], F1)
            notes.setdefault(eid, []).append(f'显示修正：去掉主体盒外 {nfrag} 块碎片（{100 * frac:.1f}% 的面），其余不变')
            actions.append(dict(entityId=eid, label=row['label'], fix='F6', why=f'dropped {nfrag} fragments ({100 * frac:.1f} % of the faces)', asset=fname,
                                faces=[int(len(F)), int(len(F1))], replaces=oldr.get('assetId') if where == 'layer model' else None))
            continue
        rec, cap = boxes.get(eid), (row.get('box') or {}).get('capped'); shown = displayed_entry(ctx, eid)[1].get('assetId')
        if not rec or cap or shown in bent:
            actions.append(dict(entityId=eid, label=row['label'], fix='flag', gates=sorted({f['gate'] for f in bad}),
                                why='no box_faces box' if not rec else 'the box is capped (the displayed model\'s own box, not a photo measurement): not replaced'
                                if cap else 'a saved bend angle was measured on the displayed model: not replaced'))
            continue
        V, F, N, T = box_mesh(rec, S); blob, layout, bounds = pack(V, F, N, [GREY] * 3); aid, fname = add_asset(eid, blob, layout, bounds, 'fixbox')
        old, _, where = displayed_entry(ctx, eid); oldr = old['representation']; gates = sorted({f['gate'] for f in bad})
        why = '; '.join(f['text'] for f in bad); why_zh = '；'.join(f['zh'] for f in bad)
        rep = {'id': 'rep-' + aid, 'kind': 'generated_mesh', 'assetId': aid, 'coordinateFrameId': fid, 'transform': dict(T, coordinateFrameId=fid),
               'primitive': None, 'placementState': 'unconfirmed', 'placementReason': 'imported_proposal', 'bounds': bounds, 'label': BOX_LABEL,
               'placementSource': {'type': 'measurement_layer_obvious_fix', 'fix': '/'.join('F' + g[1] for g in gates), 'replaces': {
                   'representationId': oldr['id'], 'assetId': oldr.get('assetId'), 'kind': oldr['kind'], 'displayed': where},
                   'method': f'displayed model obviously wrong ({why}): shown as its box_faces box (centre, axes, size of the layer box; '
                             f'{rec["sizeM"]} m, confidence {row["box"]["confidence"]})'},
               'sourceRefs': [{'role': 'measurement_layer', 'method': 'box-faces-v3', 'imageIds': image_ids}]}
        patch['models'][eid] = {**old, 'representation': rep}
        notes.setdefault(eid, []).append(f'{BOX_LABEL}（{why_zh}）')
        objs[eid]['mesh'] = (wsc.placed(V, T), F)
        actions.append(dict(entityId=eid, label=row['label'], fix='/'.join('F' + g[1] for g in gates), why=why, asset=fname,
                            replaces=oldr.get('assetId') if where == 'layer model' else None))
    # F2 / F3: onto the floor along its normal
    for eid, row in before['objects'].items():
        b = bottom_cm(ctx, objs[eid]['mesh'][0])
        if b < -oe.SINK_CM:
            translate(eid, -b, 'F2', f'robust bottom {b:.2f} cm below the floor: translated {-b:.2f} cm up along the floor normal',
                      f'显示修正：模型穿进地面 {-b:.1f} cm，沿地面法向上移 {-b:.1f} cm')
        elif row['floorContact'] and b > oe.FLOAT_CM:
            translate(eid, -b, 'F3', f'floor-contact object floating {b:.2f} cm: translated {b:.2f} cm down along the floor normal',
                      f'显示修正：落地物体悬空 {b:.1f} cm，沿地面法向下移 {b:.1f} cm')
    # F4: interpenetration left after F2-F7
    current = [o for o in ctx['objects'] if o['id'] in before['objects']]
    axes = {o['id']: np.array(boxes[o['id']]['axes'], float) if o['id'] in boxes else oe.floor_axes(o['mesh'][0], n) for o in current}
    pairs, _ = oe.g4_pairs(ctx, current, axes); scene = oe.tr.scene_of([o['mesh'] for o in current]); ious = oe.g5_iou(ctx, current, scene)
    for p in (p for p in pairs if not p['siblings']):
        ia, ib = best_iou(ious[p['a']]), best_iou(ious[p['b']])
        x = p['a'] if ia is not None and ib is not None and ia <= ib - IOU_MARGIN else p['b'] if ia is not None and ib is not None and ib <= ia - IOU_MARGIN else None
        if x is None:
            actions.append(dict(entityId=p['a'], other=p['b'], fix='flag', gates=['G4'], ratio=p['ratio'], why=f'interpenetration {100 * p["ratio"]:.0f} %, neither clearly misplaced (best IoU {ia} / {ib})'))
            continue
        other = p['b'] if x == p['a'] else p['a']; idx = {o['id']: i for i, o in enumerate(current)}

        def ious_of(mesh, x=x):
            ms = [mesh if o['id'] == x else o['mesh'] for o in current]
            return oe.g5_iou(ctx, [dict(current[idx[x]], mesh=mesh)] + [o for o in current if o['id'] != x], oe.tr.scene_of([mesh] + [m for o, m in zip(current, ms) if o['id'] != x]))[x]
        hit = search_shift(ctx, objs[x], objs[other], axes, ious_of, before['objects'][x]['floorContact'])
        if hit is None:
            actions.append(dict(entityId=x, other=other, fix='flag', gates=['G4'], ratio=p['ratio'], why=f'no translation <= {100 * MAX_SHIFT_M:.0f} cm resolves it within the limits'))
            continue
        s, cm, axis, r, after = hit
        translate(x, (s, cm), 'F4', f'interpenetrated {other[:8]} ({100 * p["ratio"]:.0f} % of the smaller), lower photo fit: translated {cm} cm along {axis} '
                  f'(overlap now {100 * r:.0f} %, best IoU {after})', f'显示修正：与 {other[:8]} 互相穿插，照片吻合较差的一方沿 {axis} 平移 {cm} cm')
    for eid, zh in notes.items():
        en = patch['models'][eid]; old = (((ctx.get('layer') or {}).get('models') or {}).get(eid) or {}).get('note')
        en['note'] = '；'.join(([old] if old and en['representation'].get('label') != BOX_LABEL else []) + zh)
    return patch, files, actions, before


def merged_layer(layer, patch):
    out = copy.deepcopy(layer); out.setdefault('models', {}).update(copy.deepcopy(patch['models']))
    ids = {a['id'] for a in patch['assets']}; out['assets'] = [a for a in out.get('assets') or [] if a['id'] not in ids] + copy.deepcopy(patch['assets'])
    return out


def run(ctx, opts):
    """workcell_layer_trial.py task: gates, fixes, gates again on the patched display; files = patch, meshes, trial layer, results."""
    patch, files, actions, before = fix(ctx, opts)
    after = oe.run(ctx, opts)
    for tag, res in (('before', before), ('after', after)):  # the gates' drawings, per stage
        files.update({f'{tag}-{k}': v for k, v in (res.pop('files', None) or {}).items()})
    res = dict(before=before, after=after, actions=actions, patchModels=sorted(patch['models']), summaryBefore=before['summary'], summaryAfter=after['summary'])
    files.update({'patch.json': json.dumps(patch, indent=1, ensure_ascii=False).encode(),
                  'fix-results.json': json.dumps(res, indent=1, ensure_ascii=False, default=float).encode()})
    if ctx.get('layer'):
        files['trial-layer.json'] = json.dumps(merged_layer(ctx['layer'], patch), indent=1, ensure_ascii=False).encode()
    return dict(actions=actions, summaryBefore=before['summary'], summaryAfter=after['summary'], files=files)


def _check():
    """Synthetic z-up floor, S = 1: the box mesh round-trips (pack -> read_packed -> placed = the box corners); a sinker and a
    floater go to bottom 0; a model with a wrong height and an uncapped verified box is shown as the box; a misplaced model
    inside another moves out along one axis."""
    S = 1.; th = .4; l = np.array([np.cos(th), np.sin(th), 0]); w = np.cross([0, 0, 1.], l)
    rec = dict(axes=[l.tolist(), w.tolist(), [0, 0, 1.]], sizeM=[.3, .2, .9], centerNative=[1., 2., .45], bottomM=0., floorContact=True,
               dims={k: dict(confidence='high') for k in ('L', 'W', 'H', 'bottom')}, highlightReasons=[])
    V, F, N, T = box_mesh(rec, S); blob, layout, _ = pack(V, F, N, [GREY] * 3); V2, F2 = wsc.read_packed(blob, layout)
    W = wsc.placed(V2, T); assert np.allclose(np.sort(W @ l)[[0, -1]], [1 * l[0] + 2 * l[1] - .15, 1 * l[0] + 2 * l[1] + .15], atol=1e-6)
    assert abs(W[:, 2].min()) < 1e-6 and abs(W[:, 2].max() - .9) < 1e-6 and len(F2) == 12
    tri = W[F2]; nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]); assert ((nrm * (tri.mean(1) - W.mean(0))).sum(1) > 0).all(), 'faces point outwards'
    box = lambda lo, hi: oe._box(lo, hi)  # noqa: E731
    ident = dict(position=[0, 0, 0.], quaternion=[0, 0, 0, 1.], scale=[1, 1, 1.])
    ents = [dict(id=i, representations=[dict(id='r' + i, kind='generated_mesh', assetId='a' + i, transform=dict(ident, coordinateFrameId='f'))],
                 activeModelRepresentationId='r' + i) for i in ('sink', 'float', 'tall', 'in', 'big', 'frag')]
    (Va, Fa), (Vb, Fb) = box([5, 0, 0], [5.2, .2, .9]), box([6, 1, 0], [6.05, 1.05, .05]); frag = (np.r_[Va, Vb], np.r_[Fa, Fb + len(Va)])
    objs = [dict(id='sink', label='sink', mesh=box([0, 0, -.05], [.2, .2, .5])), dict(id='float', label='float', mesh=box([1, 0, .06], [1.2, .2, .5])),
            dict(id='tall', label='tall', mesh=box([2, 0, 0], [2.2, .2, 1.5])), dict(id='in', label='in', mesh=box([3.05, 0.05, 0], [3.15, .15, .3])),
            dict(id='big', label='big', mesh=box([3, 0, 0], [3.5, .5, .5])), dict(id='frag', label='frag', mesh=frag)]
    ctx = dict(floor=(np.array([0, 0, 1.]), 0.), S=S, cams=[], objects=objs, layer=dict(models={}, boxes={'tall': dict(rec, axes=np.eye(3).tolist(), sizeM=[.2, .2, .9], centerNative=[2.1, .1, .45]),
                                                                                              'frag': dict(rec, axes=np.eye(3).tolist(), sizeM=[.2, .2, .9], centerNative=[5.1, .1, .45])}),
               doc=dict(entities=ents, coordinateFrames=[dict(id='f')]))
    row = lambda fails, contact=False, capped=False: dict(label='x', failed=fails, floorContact=contact, box=dict(capped=capped, confidence={}))  # noqa: E731
    before = dict(objects={'sink': row([]), 'float': row([], True), 'tall': row([dict(gate='G7', severity='obvious', text='H', zh='高')]), 'in': row([]), 'big': row([]),
                           'frag': row([dict(gate='G6', severity='obvious', fragmentFaces=.5, text='frag', zh='碎片')])})
    ious = {'in': {1: .2}, 'big': {1: .8}}
    oe_g5, oe.g5_iou = oe.g5_iou, lambda ctx, objs, scene: {o['id']: ious.get(o['id'], {}) for o in objs}  # no cameras: fixed IoUs
    raw, g = globals()['raw_rows'], globals(); g['raw_rows'] = lambda ctx, rep, opts: (np.c_[frag[0], np.zeros((len(frag[0]), 3)), np.full((len(frag[0]), 3), .5)], frag[1])
    try:
        patch, files, actions, _ = fix(ctx, {}, before)
    finally:
        oe.g5_iou, g['raw_rows'] = oe_g5, raw
    got = {a['entityId']: a['fix'] for a in actions}
    assert got == {'tall': 'F7', 'sink': 'F2', 'float': 'F3', 'in': 'F4', 'frag': 'F6'}, actions
    assert len(objs[5]['mesh'][1]) == 12 and len(objs[5]['mesh'][0]) == 8 and objs[5]['mesh'][0][:, 0].max() <= 5.2, 'only the fragment dropped'
    m = {o['id']: o['mesh'][0] for o in objs}
    assert abs(bottom_cm(ctx, m['sink'])) < 1e-6 and abs(bottom_cm(ctx, m['float'])) < 1e-6 and abs(np.ptp(m['tall'][:, 2]) - .9) < 1e-6
    assert patch['models']['tall']['representation']['label'] == BOX_LABEL and sorted(files) == ['fixbox-tall.bin', 'fixfrag-frag.bin']
    assert np.allclose(patch['models']['sink']['representation']['transform']['position'], [0, 0, .05])
    a = oe.solid(*objs[3]['mesh'], np.eye(3), S); b = oe.solid(*objs[4]['mesh'], np.eye(3), S)
    assert oe.overlap(a, b, S) / a['volume'] <= oe.OVERLAP, 'the misplaced model is out'
    print('fix_obvious self-test passed:', actions)


if __name__ == '__main__':
    _check()
