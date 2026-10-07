"""The 090 comparison layer `<publicationId>.sam3d.json`: the published layer with every RecGen-derived display model replaced.
The published layer is read only and stays the report's own; the comparison opens with ?layer=sam3d.

  RecGen objects of run lucida-replica-01   SAM 3D as chosen by compare.py (results.json), floor fix applied; an object whose
                                            SAM 3D candidates all have an obvious error is shown as its measured box
  parts of those objects (source partitions) the parent's SAM 3D mesh inside the part's box, or the part's box
  other RecGen meshes (lamps, poster, ...)  their measured box (no SAM 3D input prepared for them yet)
Facts and confidence that described the RecGen model are dropped for those entities and replaced by one note with the SAM 3D
numbers; photo measurements (现场对照) are kept, with the model's lowest point rewritten. Scale, floor and e-stop facts unchanged.

  /Users/adam/Desktop/panoptes-public/panoptes-serving/.venv/bin/python build_layer.py

Reused for other reports (defaults unchanged): compare.py's environment (AB_RUN, AB_OUT, CMP_NOTES, CMP_BOXES); BL_PAGES = the
directory pipeline sheets are copied to, BL_PUB / BL_VIEW = the publication and its view file, BL_GEOMETRY = the '相机与深度'
stage text ({} = cm per native unit), BL_DEPTH = the depth the SAM 3D input used. pipeline() leaves out the RecGen comparison
when results.json has none.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import sys

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import trimesh

import compare as C

sys.path.insert(0, os.environ.get('PANOPTES_SERVING', '/Users/adam/Desktop/panoptes-public/panoptes-serving') + '/scripts/research')
from assemble_lucida_scene import mesh_bytes  # noqa: E402

HERE = Path(__file__).resolve().parent
PAGES = Path(os.environ.get('BL_PAGES') or '/Users/adam/Desktop/panoptes-public/panoptes-workcell-pages/workcell-photo-direct/report/measurement-layer')
PUB = os.environ.get('BL_PUB') or '4b58dbd2-3846-47f2-af97-57eaa108753c'
VIEW = json.loads(Path(os.environ.get('BL_VIEW') or C.CHECKS / '4b58-view.json').read_text())['publication']['snapshot']['revision']['document']
MAX_FACES = 80000
MODEL_FACTS = {'confidence', 'pipeline', 'shape_check', 'shape_validation', 'fold', 'correction'}


def recgen_entities():
    """Entity id -> (kind, source) for every displayed model that came from RecGen."""
    assets = {a['id']: a for a in VIEW['assets']}
    out = {}
    for e in VIEW['entities']:
        rep = next((r for r in e['representations'] if r['id'] == e.get('activeModelRepresentationId')), None)
        if rep is None:
            continue
        blob = json.dumps(rep) + json.dumps((assets.get(rep.get('assetId')) or {}).get('metadata'))
        refs = [s.get('sourceRecordId') for s in rep.get('sourceRefs') or [] if s.get('role') == 'model_artifact']
        if 'RecGen' in blob or any(refs):
            partition = (rep.get('placementSource') or {}).get('type') == 'derived_source_partition'
            out[e['id']] = {'label': e['label'], 'source': next((r for r in refs if r), None), 'partition': partition, 'rep': rep}
    return out


def coloured_world_mesh(variant, oid):
    c = C.comparison(variant, oid)
    d = np.load(C.OUT / variant / f'{oid}.npz')
    mesh = C.world_mesh(variant, oid, c)
    fixed, notes = C.floor_fix(mesh, oid)
    _, idx = cKDTree(mesh.vertices).query(fixed.vertices)  # colours survive the floor cut by nearest original vertex
    return fixed, d['colors'][idx], notes


def decimate(V, F, colours, faces=MAX_FACES):
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(F))
    m.vertex_colors = o3d.utility.Vector3dVector(colours.astype(np.float64) / 255)
    if len(F) > faces:
        m = m.simplify_quadric_decimation(faces)
    m.remove_unreferenced_vertices()
    return np.asarray(m.vertices), np.asarray(m.triangles), (np.asarray(m.vertex_colors) * 255).astype(np.uint8)


def crop_to_box(mesh, colours, box, margin=0.05):
    """Keep the faces whose centroid lies inside the (margin-expanded) measured box."""
    ax = np.asarray(box['axes'], float)
    half = (np.asarray(box['sizeM']) / 2 + margin) / C.N2M
    local = (mesh.triangles_center - np.asarray(box['centerNative'])) @ ax.T
    keep = (np.abs(local) <= half).all(1)
    sub = trimesh.Trimesh(mesh.vertices, mesh.faces[keep], process=False)
    used = np.unique(sub.faces)
    remap = -np.ones(len(mesh.vertices), int); remap[used] = np.arange(len(used))
    return trimesh.Trimesh(mesh.vertices[used], remap[sub.faces], process=False), colours[used]


def mesh_entry(eid, mesh, colours, frame):
    V, F, col = decimate(np.asarray(mesh.vertices), np.asarray(mesh.faces), colours)
    centre = (V.min(0) + V.max(0)) / 2  # local frame centred on the mesh, placed by the transform (as the layer's other meshes)
    V = V - centre
    bounds = {'min': V.min(0).tolist(), 'max': V.max(0).tolist()}
    tm = trimesh.Trimesh(V, F, vertex_colors=np.column_stack([col, np.full(len(col), 255, np.uint8)]), process=False)
    raw, layout = mesh_bytes(tm)
    sha = hashlib.sha256(raw).hexdigest()
    asset, name = f'sam3d-{sha[:24]}', f'sam3d-{eid[:8]}.bin'
    (PAGES / name).write_bytes(raw)
    byte_layout = {'stride': 9, 'indexType': 'uint32', 'byteOffset': 0, 'indexCount': layout['index_count'],
                   'vertexCount': layout['vertex_count'], 'indexByteOffset': layout['index_byte_offset']}
    a = {'id': asset, 'url': f'measurement-layer/{name}', 'kind': 'generated_mesh', 'sha256': sha, 'sizeBytes': len(raw),
         'mediaType': 'application/octet-stream', 'format': 'panoptes-mesh-v1', 'byteLayout': byte_layout, 'bounds': bounds,
         'metadata': {'kind': 'generated_mesh', 'bounds': bounds, 'format': 'panoptes-mesh-v1', 'byteLayout': byte_layout,
                      'entityId': eid}}
    rep = {'id': f'rep-{asset}', 'kind': 'generated_mesh', 'assetId': asset, 'coordinateFrameId': frame, 'primitive': None,
           'placementState': 'unconfirmed', 'placementReason': 'imported_proposal', 'bounds': bounds,
           'transform': {'coordinateFrameId': frame, 'position': centre.tolist(), 'quaternion': [0., 0., 0., 1.], 'scale': [1., 1., 1.]}}
    return a, rep, len(F)


def box_rep(eid, boxes, frame, colour=(0.62, 0.62, 0.6)):
    """A primitive box (or several, merged as one mesh, when the box stands for its parts) from the measured box(es)."""
    ax = np.asarray(boxes[0]['axes'], float)
    if len(boxes) == 1:
        R = ax.T.copy()
        if np.linalg.det(R) < 0:
            R[:, 1] *= -1
        b = boxes[0]
        half = (np.asarray(b['sizeM']) / C.N2M / 2).tolist()
        return None, {'id': f'rep-box-{eid[:8]}', 'kind': 'primitive', 'assetId': None, 'coordinateFrameId': frame,
                      'placementState': 'unconfirmed', 'placementReason': 'imported_proposal',
                      'bounds': {'min': [-x for x in half], 'max': half},
                      'primitive': {'kind': 'box', 'dimensions': (np.asarray(b['sizeM']) / C.N2M).tolist()},
                      'material': {'color': list(colour)},
                      'transform': {'coordinateFrameId': frame, 'position': list(map(float, b['centerNative'])),
                                    'quaternion': Rotation.from_matrix(R).as_quat().tolist(), 'scale': [1., 1., 1.]}}, 12 * len(boxes)
    parts = []
    for b in boxes:
        a = np.asarray(b['axes'], float)
        t = trimesh.creation.box(extents=np.asarray(b['sizeM']) / C.N2M)
        t.vertices = t.vertices @ a + np.asarray(b['centerNative'])
        parts.append(t)
    m = trimesh.util.concatenate(parts)
    return mesh_entry(eid, m, np.tile((np.asarray(colour) * 255).astype(np.uint8), (len(m.vertices), 1)), frame)


def rep_bounds_box(rep):
    """A box record (as boxes-090) from a representation's local bounds and transform."""
    t, b = rep['transform'], rep['bounds']
    R = Rotation.from_quat(t['quaternion']).as_matrix()
    lo, hi, s = np.asarray(b['min']), np.asarray(b['max']), np.asarray(t['scale'])
    return {'axes': R.T.tolist(), 'sizeM': ((hi - lo) * s * C.N2M).tolist(),
            'centerNative': (R @ ((lo + hi) / 2 * s) + np.asarray(t['position'])).tolist()}


LEVEL = {'high': '高', 'medium': '中', 'low': '低', 'unverified': '未验证'}
VIEWS = {o['object_id']: o['views'] for o in json.loads((C.RUN / 'evidence/objects.json').read_text())['objects']}
GEOMETRY = os.environ.get('BL_GEOMETRY') or ('Pi3X 前馈几何（权重 CC BY-NC，待替换）：三张照片的相机位姿与深度；尺度由急停定（1 原生单位 = {:.1f} cm）；'
                                             '全部物体共用一个 RANSAC 地面')
DEPTH = os.environ.get('BL_DEPTH') or 'Pi3X 深度点图'


def pipeline(eid, oid, parent, results, rec, layer, text):
    """Every stage for one object, photos to measurements, with its sheet image for the nine RecGen objects."""
    box = (layer.get('boxes') or {}).get(eid)
    stages = []
    if oid in VIEWS:
        stages.append({'label': '照片与掩码', 'text': 'SAM 3 文本提示分割（SAM License）：' + '，'.join(
            f"照片 {v['frame_id'][-1]} {v['mask_pixels'] / 1e3:.0f}k 像素" for v in VIEWS[oid])})
    stages.append({'label': '相机与深度', 'text': GEOMETRY.format(layer['scale']['nativeToMeters'] * 100)})
    url = None
    if oid in results:
        r = results[oid]
        stages.append({'label': '补全候选', 'text': f'SAM 3D Objects（SAM License）每张照片各生成一次，输入 = 照片 + 掩码 + {DEPTH}：' + '；'.join(
            f"照片 {C.gen_photo(v, oid)}：IoU {t['meanIou']:.2f}，深度 {100 * t['meanDepthP50']:.1f}%，" + (f"点图覆盖 {100 * t['coverage']:.0f}%，" if t.get('coverage') is not None else '')
            + (f"{len(t['gates'])} 个明显错误" if t['gates'] else '无明显错误')
            for v, t in r['tried'].items())})
        f = r['sam3dFixed']
        stages.append({'label': '组装', 'text': f"同一组装（9 自由度，全部照片的轮廓 + 深度）：照片 {C.gen_photo(r['variant'], oid)} 的候选轮廓 IoU "
                       + ' / '.join(f"{p['iou']:.2f}" for p in f['perPhoto'].values()) + f"，深度残差 p50 {100 * f['meanDepthP50']:.1f}%"})
        stages.append({'label': '明显错误检查', 'text': '下沉 / 悬空 / 塌陷 / 过大 / 高度 / 底部 / 占了看得见的空地 / 深度 / 轮廓：'
                       + ('；'.join(f['gates']) or '全部通过') + ('；修正：' + '，'.join(f['fix']) if f['fix'] else '')})
        stages.append({'label': '采用', 'text': r['decision'] + (f"（RecGen 对照：IoU {r['recgen']['meanIou']:.2f}，最低点 {r['recgen']['lowestCm']:.1f} cm"
                       + (f"，{'；'.join(r['recgen']['gates'])}" if r['recgen']['gates'] else '') + '）' if 'recgen' in r else '')})
        sheet = C.HERE / f'{oid}.jpg'
        if sheet.exists():
            (PAGES / f'pipeline-{eid[:8]}.jpg').write_bytes(sheet.read_bytes())
            url = f'measurement-layer/pipeline-{eid[:8]}.jpg'
    else:
        stages.append({'label': '模型', 'text': text})
    if box:
        d = box['dims']
        stages.append({'label': '测量（统一地面）', 'text': '长×宽×高 ' + '×'.join(f"{100 * d[k]['valueM']:.1f}" for k in 'LWH')
                       + f" cm，离地 {100 * d['bottom']['valueM']:.1f} cm；置信度 " + ' / '.join(f"{n} {LEVEL[d[k]['confidence']]}" for k, n in zip(('L', 'W', 'H', 'bottom'), '长宽高') ) + f" / 离地 {LEVEL[d['bottom']['confidence']]}"
                       + ('；需复核：' + '；'.join(box['highlightReasons']) if box.get('highlightReasons') else '')})
    check = next((x['text'] for x in layer['facts'].get(eid, []) if x.get('kind') == 'check'), None)
    if check:
        stages.append({'label': '现场对照', 'text': check})
    caption = '白 = 掩码，红 = RecGen，橙 = SAM 3D' if any('recgen' in r for r in results.values()) else '白 = 掩码，橙 = SAM 3D'
    return {'url': url, 'caption': caption + '；第二行 = 每张照片生成的候选（点开看大图）', 'stages': stages}


def rewrite_check(text, lowest_cm):
    text = re.sub(r'本次无可靠值：模型下部缺失（最低点 ([0-9.]+) cm）', r'原报告无可靠值（当时 RecGen 模型下部缺失，最低点 \1 cm）', text)
    return re.sub(r'模型最低点 [0-9.+−-]+ cm[^。｜]*', f'SAM 3D 模型最低点 {lowest_cm:.1f} cm', text)


def main():
    results = json.loads((HERE / 'results.json').read_text())
    published = json.loads((PAGES / f'{PUB}.json').read_text())
    layer = copy.deepcopy(published)
    frame = layer['coordinateFrameId']
    boxes = C.BOXES['boxes']
    eid_of = dict(zip(C.OBJ_ORDER, list(boxes)[:len(C.OBJ_ORDER)]))  # same order as compare.BOX (labels checked there)
    oid_of = {v: k for k, v in eid_of.items()}
    rec = recgen_entities()
    drop_assets = set()
    for eid in rec:  # RecGen-derived overrides in the published layer go too
        old = (layer.get('models') or {}).pop(eid, None)
        if old:
            drop_assets.add(old['representation'].get('assetId'))
    layer['assets'] = [a for a in layer.get('assets', []) if a['id'] not in drop_assets]
    sam_mesh, summary = {}, {}
    for oid, r in results.items():  # the nine RecGen objects first: parents of the partitions
        eid = eid_of[oid]
        if r['decision'].startswith('SAM 3D'):
            sam_mesh[oid] = coloured_world_mesh(r['variant'], oid)
    for eid, info in rec.items():
        box = boxes.get(eid)
        parent = info['source'] if info['source'] in results else None
        oid = oid_of.get(eid)
        if oid in results and oid in sam_mesh:  # the object itself
            mesh, colours, notes = sam_mesh[oid]
            asset, rep, nf = mesh_entry(eid, mesh, colours, frame)
            r = results[oid]['sam3dFixed']
            how = f"SAM 3D（照片 {results[oid]['generationPhoto']} 生成，SAM License）"
            text = (f"{how}｜摆放轮廓 IoU " + ' / '.join(f"{p['iou']:.2f}" for p in r['perPhoto'].values())
                    + f"｜深度残差 p50 {100 * r['meanDepthP50']:.1f}%｜最低点 {r['lowestCm']:.1f} cm｜尺寸 "
                    + '×'.join(f'{100 * x:.0f}' for x in r['extentM']) + ' cm' + (f"｜{'，'.join(notes)}" if notes else '')
                    + f"｜RecGen 对照：IoU {results[oid]['recgen']['meanIou']:.2f}，最低点 {results[oid]['recgen']['lowestCm']:.1f} cm"
                    + (f"（{'；'.join(results[oid]['recgen']['gates'])}）" if results[oid]['recgen']['gates'] else ''))
            level = 'high' if r['meanIou'] >= .85 and r['meanDepthP50'] <= .02 else 'medium'
            conf = {'level': level, 'label': {'high': '高', 'medium': '中'}[level],
                    'reasons': ['SAM 3D 摆放轮廓 IoU ' + ' / '.join(f"{p['iou']:.2f}" for p in r['perPhoto'].values()),
                                f"深度残差 p50 {100 * r['meanDepthP50']:.1f}%"], 'missing': []}
            lowest = r['lowestCm']
        elif parent in sam_mesh and box is not None:  # a part of a SAM 3D object
            mesh, colours = crop_to_box(sam_mesh[parent][0], sam_mesh[parent][1], box)
            asset, rep, nf = mesh_entry(eid, mesh, colours, frame) if len(mesh.faces) else box_rep(eid, [box], frame)
            text = f"「{rec[eid_of[parent]]['label']}」的 SAM 3D 模型在本部件测量盒内的部分"
            conf, lowest = None, float(np.percentile((mesh.vertices @ C.UP + C.OFF) * C.N2M, .5) * 100) if len(mesh.faces) else None
        else:  # obvious error in every SAM 3D candidate, or no SAM 3D input prepared: the measured box
            children = [boxes[k] for k, v in rec.items() if v['partition'] and v['source'] == info['source'] and k != eid and k in boxes
                        and info['source'] is not None and oid is not None]
            use = children or ([box] if box else [rep_bounds_box(info['rep'])])  # no measured box: the old model's bounding box
            asset, rep, nf = box_rep(eid, use, frame)
            why = (results[oid]['decision'] if oid in results else
                   f"「{rec[eid_of[parent]]['label']}」的 SAM 3D 候选都有明显错误，部件也用测量盒" if parent else
                   '这个物体还没有准备 SAM 3D 输入' + ('' if box else '（没有测量盒，暂用原模型的包围盒尺寸）'))
            text = f"测量盒代替模型：{why}" + ('（显示各部件的测量盒）' if children else '')
            conf = {'level': 'low', 'label': '低', 'reasons': [], 'missing': [why]} if oid in results else None
            lowest = None
        layer.setdefault('models', {})[eid] = {'representation': rep, 'note': text}
        if asset:
            layer['assets'].append(asset)
        facts = [f for f in layer.get('facts', {}).get(eid, []) if f.get('kind') not in MODEL_FACTS and '原生成模型' not in f.get('label', '')]
        for f in facts:
            if f.get('kind') == 'check' and lowest is not None:
                f['text'] = rewrite_check(f['text'], lowest)
        layer.setdefault('facts', {})[eid] = [{'label': '补全模型（对照版）', 'kind': 'note', 'text': text}] + facts
        if conf:
            layer.setdefault('confidence', {})[eid] = conf
        else:
            (layer.get('confidence') or {}).pop(eid, None)
        if eid in layer.get('boxes', {}):
            b = layer['boxes'][eid]
            b['highlightReasons'] = [x for x in b.get('highlightReasons') or [] if not x.startswith('图层：')] + \
                                    [f'图层：{m}' for m in (conf or {}).get('missing', [])]
            b['highlight'] = bool(b['highlightReasons'])
            if oid in sam_mesh:  # dims that showed the RecGen size now show the SAM 3D size
                ext = results[oid]['sam3dFixed']['extentM']
                for k, v in zip('LWH', ext):
                    if b['dims'][k].get('source') == 'model':
                        b['dims'][k]['valueM'] = round(v, 4)
                        b['sizeM']['LWH'.index(k)] = round(v, 4)
        summary[eid[:8]] = (info['label'], rep['kind'], nf, text[:60])
        layer.setdefault('pipelines', {})[eid] = pipeline(eid, oid, parent, results, rec, layer, text)
    layer['variant'] = {'id': 'sam3d', 'label': '对照版：所有 RecGen 模型已换成 SAM 3D（SAM License）或测量盒；几何仍是 Pi3X。原报告未改。'}
    for k in ('scale', 'ground'):
        assert layer[k] == published[k], k  # scale and floor unchanged
    for eid, fs in published.get('facts', {}).items():
        if eid not in rec:
            assert layer['facts'][eid] == fs, eid  # other entities' facts (e-stop hosts included) unchanged
    out = PAGES / f'{PUB}.sam3d.json'
    out.write_text(json.dumps(layer, ensure_ascii=False, indent=1))
    for k, v in summary.items():
        print(k, *v)
    print(out, round(out.stat().st_size / 1e6, 2), 'MB; mesh bins',
          round(sum(p.stat().st_size for p in PAGES.glob('sam3d-*.bin')) / 1e6, 1), 'MB')


if __name__ == '__main__':
    main()
