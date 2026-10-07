"""Measurement layer (web/src/measurement-layer.ts schema) for a locally published module-swap variant, from the ORIGINAL stage
results already computed for it (swap-runs/<V>-stages: box_faces, floor, shape, obvious_errors, lower_edge, plane_facets), the
run's assembly result, the completion selection (cmp-*/results.json + sheets) and the geometry's field values (results.json).
Per entity: confidence, facts (G1-G8 check, assembly, field value), the measured box (box_faces output = LayerBox), and the
"完整流程" pipeline panel (S2 geometry -> S4 completion -> S5 assembly -> S8 shape -> S9 floor -> S11 lower edge -> S12 box -> S14 gates)
with the per-object comparison sheet. Writes PAGES/measurement-layer/<publicationId>.json (+ pipeline-<eid>.jpg).

    python build_swap_layer.py V [V ...]      V = v2-mvs-fill-sam3d, v2-pi3x-recgen, ... (also the v1 names)
"""
import json
import os
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent
SP = Path(os.environ.get('SWAP_SCRATCH', '/private/tmp/claude-501/-Users-adam-Desktop-panoptes-public/1fd9a1db-e580-4bfc-8110-119a1cc38a99/scratchpad'))
PLATFORM = Path(os.environ.get('PANOPTES_PLATFORM', '/Users/adam/Desktop/Tesla/panoptes-platform'))
PAGES = Path(os.environ.get('PANOPTES_PAGES', '/Users/adam/Desktop/panoptes-public/panoptes-workcell-pages/workcell-photo-direct/report/measurement-layer'))
ZH = {'left_light_curtain': '左光幕', 'right_light_curtain': '右光幕', 'left_fence': '左围栏', 'right_fence': '右围栏', 'left_post': '左防撞柱',
      'right_post': '右防撞柱', 'robot': '机器人', 'cart': '料车', 'guard': '防护板', 'observed_floor': '观测地面'}
EN = {'left_light_curtain': 'left light curtain', 'right_light_curtain': 'right light curtain', 'left_fence': 'left fence', 'right_fence': 'right fence',
      'left_post': 'left bollard', 'right_post': 'right bollard', 'robot': 'robot', 'cart': 'cart', 'guard': 'safety guard'}
LEVEL = {'high': '高', 'medium': '中', 'low': '低', 'unverified': '未验证'}
VERDICT = {'ok': '一致', 'depth_off': '深度偏差', 'ambiguous': '含糊', 'inconclusive': '无法判断', 'not_photo_consistent': '不符'}
FLOOR = {'sinks': '插地', 'hovers': '悬空', 'on_floor': '贴地', 'above': '离地（挂装）'}
GEOM = {'pi3x': ('Pi3X（CC BY-NC，非商用）', '尺度 = 已发布急停标定'), 'mvs': ('MVS（RoMa + numpy LM BA，不含 GPL，可商用）', '尺度 = 本几何自己的急停拟合'),
        'mvs-fill': ('MVS + 掩码内补洞（MoGe-3，MIT；可商用）', '尺度 = 本几何自己的急停拟合（补洞不动地面与急停）')}
FIELD = {'right_light_curtain': ('housing090R', '右罩壳下沿', 24), 'right_fence': ('fence090R', '右围栏下横杆', 20), 'left_light_curtain': ('housing090L', '左罩壳下沿（半边被挡，不计分）', 24)}


def load(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def geometry_of(v):
    return 'mvs-fill' if 'mvs-fill' in v else 'mvs' if 'mvs' in v else 'pi3x'


def cmp_dir(v):
    if 'recgen' in v:
        return None
    cell = '030-' if v.startswith('030-') else ''
    return HERE / f"cmp-{cell}{'mvs-fill' if 'mvs-fill' in v else 'mvs' if 'mvs' in v else 'pi3x'}"


def build(v):
    res = load(PLATFORM / '.platform/swap-20261007' / v / 'result.json')
    doc = load(PLATFORM / res['view'] if not Path(res['view']).is_absolute() else res['view'])['publication']['snapshot']['revision']['document']
    run = Path(res['run']); st = SP / 'swap-runs' / f'{v}-stages'
    ents = load(st / 'entity-map.json')                       # object_id -> entity id
    stage = lambda name: (load(st / name / 'results.json') or {}).get('result') or {}
    boxes, floor, shape, obvious, lower = stage('box_faces').get('boxes') or {}, stage('floor').get('objects') or {}, stage('shape'), stage('obvious_errors'), stage('lower_edge')
    shape_rows = {r['entityId']: r for r in shape.get('rows', [])}
    comps = {c['object_id']: c for c in load(run / 'result/comparisons.json')['objects']}
    fl = load(run / 'evidence/floor.json'); S = res['nativeToMeters']; g = geometry_of(v)
    if v.startswith('030-'):   # the 030 cell: the fair evaluator's 030 housing values (MVS / fill rows of field_values_fill.py; Pi3X: none here)
        fv = ((load(HERE / 'field-values-mvs-fill.json') or {}).get({'mvs-fill': 'mvs-fill', 'mvs': 'mvs-da3-base'}.get(g, '')) or {})
        field = {'right_light_curtain': ('housing030R', '右罩壳下沿', 24), 'left_light_curtain': ('housing030L', '左罩壳下沿', 24)}
    else:
        fv = ((load(HERE / 'results.json') or {}).get('fieldValues') or {}).get(g) or {}
        field = FIELD
    cmp = load(cmp_dir(v) / 'results.json') if cmp_dir(v) else None
    v2 = v.startswith('v2-')
    frame = doc['coordinateFrames'][0]
    layer = {'schemaVersion': 1, 'publicationId': res['publicationId'], 'revisionId': res['revisionId'], 'coordinateFrameId': frame['id'],
             'scale': {'nativeToMeters': S, 'status': 'operator_anchored', 'source': (frame.get('scale') or {}).get('sourceRefs', [{}])[0].get('object', 'e-stop')},
             'ground': {'normal': fl['plane_native'][:3], 'offset': fl['plane_native'][3], 'plane': fl['plane_native'], 'source': fl.get('fit', 'run floor')},
             'variant': {'id': v, 'label': f"模块替换 {v}：几何 {GEOM[g][0]}，补全 {'RecGen（TRI 非商用）' if 'recgen' in v else 'SAM 3D Objects（SAM License）'}，组装 {'v2（地面接触罚项）' if v2 else 'v1'}",
                         'labelEn': f"module swap {v}: geometry {g}, completion {'RecGen' if 'recgen' in v else 'SAM 3D'}, assembly {'v2 (floor-contact hinge)' if v2 else 'v1'}"},
             'labels': {}, 'labelsEn': {}, 'confidence': {}, 'facts': {}, 'boxes': {}, 'pipelines': {}, 'assets': []}
    for oid, eid in ents.items():
        if oid not in comps:
            continue
        layer['labels'][eid] = ZH.get(oid, oid); layer['labelsEn'][eid] = EN.get(oid, oid)
        c = comps[oid]; views = {x['frame_id']: x['generated_refined'] for x in c['views']}
        ious = [x['visible_iou'] for x in views.values()]; deps = [x['relative_depth_p50'] for x in views.values() if x['relative_depth_p50'] is not None]
        iou_txt = '轮廓 IoU ' + ' / '.join(f"{views[f]['visible_iou']:.2f}" for f in sorted(views)) + f"（均 {sum(ious) / len(ious):.2f}）" + (f"，深度残差 p50 {100 * sum(deps) / len(deps):.1f}%" if deps else '')
        rf = c['refinement']; fin = (rf.get('final') or {}).get('floor') or {}
        lowest = f"，组装后最低点 {100 * S * fin['lowest_native']:+.1f} cm" if fin.get('lowest_native') is not None else ''
        ob = (obvious.get('objects') or {}).get(eid) or {}; fails = [x['zh'] for x in ob.get('failed', [])]
        fo = floor.get(eid) or {}; sh = shape_rows.get(eid) or {}; bx = boxes.get(eid)
        rec = load(run / 'generation' / oid / 'output.json') or {}; sel = rec.get('selection') if isinstance(rec.get('selection'), dict) else {}
        cm = (cmp or {}).get(oid) or {}
        # pipeline stages
        stages = [{'label': '几何（S2）', 'text': f"{GEOM[g][0]}；{GEOM[g][1]}（1 原生单位 = {100 * S:.1f} cm）"
                   + (f"；掩码内点图覆盖率 {100 * cm['sam3d']['coverage']:.0f}%" if cm.get('sam3d', {}).get('coverage') is not None else '')}]
        if 'recgen' in v:
            stages.append({'label': '补全（S4）', 'text': 'RecGen（TRI 非商用）：全部照片联合生成，一个模型'})
        else:
            tried = sel.get('tried') or {}
            parts = ['SAM 3D Objects：每张有掩码的照片各生成一个候选，按全部照片一致性择优（先比明显错误数，再 IoU − 2×深度 + 0.5×覆盖）']
            parts += [f"{('照片 ' + str(sel.get('generationPhoto', '?'))) if k == 'sam3d' else k.replace('sam3d-frame_000', '照片 ')}：IoU {t['meanIou']:.2f}，深度 {100 * t['meanDepthP50']:.1f}%{'，' + '、'.join(t['gates']) if t.get('gates') else ''}" for k, t in tried.items()]
            parts.append(f"采用：{sel.get('decision', 'SAM 3D')}（生成用照片 {str(sel.get('generationPhoto', '?')).replace('frame_000', '')}；择优里的地面修正只用于比较，进入组装的是原始候选）")
            stages.append({'label': '补全（S4）', 'text': '；'.join(parts)})
        stages.append({'label': '组装（S5）' + ('，v2：加地面接触罚项' if v2 else ''), 'text': f"同一组装（assemble_lucida_scene，9 自由度，全部照片轮廓 + 边界 + 深度{' + 地面接触罚项' if v2 else ''}）：{iou_txt}{lowest}"
                       + (f"；罚项终值 {fin['penalty']:.4f}" if v2 and fin.get('penalty') is not None else '')})
        if sh:
            stages.append({'label': '形状校验（S8）', 'text': f"三照片光度一致性：{VERDICT.get(sh.get('verdict'), sh.get('verdict'))}" + (f"（最佳深度 {100 * (sh.get('bestScale', 1) - 1) * sh.get('distanceToReference', 0) * S:+.0f} cm）" if sh.get('bestScale') else '')})
        if fo:
            stages.append({'label': '地面接触（S9）', 'text': f"模型最低点 {100 * fo['bottomM']:+.1f} cm：{FLOOR.get(fo['status'], fo['status'])}"})
        le_items = lower.get('targets') or lower.get('objects') or lower.get('results') or [] if lower else []
        le = [x for x in (le_items.values() if isinstance(le_items, dict) else le_items) if isinstance(x, dict) and x.get('entityId') == eid]
        for x in le:
            stages.append({'label': '下沿实测（S11）', 'text': f"{x.get('part') or '整体'}：" + (f"{x['valueCm']:.1f} cm" if x.get('valueCm') is not None else f"无可靠值（{x.get('reason')}" + (f"，估计 {x['estimateCm']:.1f} ± {x['sigmaCm']:.1f}" if x.get('estimateCm') else '') + '）')})
        if bx:
            d = bx['dims']
            stages.append({'label': '盒子（S12）', 'text': f"L×W×H {100 * bx['sizeM'][0]:.0f}×{100 * bx['sizeM'][1]:.0f}×{100 * bx['sizeM'][2]:.0f} cm，离地 {100 * bx['bottomM']:.1f} cm；置信度 "
                           + ' / '.join(f"{n} {LEVEL[d[k]['confidence']]}" for k, n in zip(('L', 'W', 'H', 'bottom'), ('长', '宽', '高', '离地'))) + ('；' + '；'.join(bx.get('highlightReasons') or []) if bx.get('highlight') else '')})
        stages.append({'label': '明显错误门 G1–G8（S14）', 'text': '；'.join(fails) if fails else '通过'})
        p = {'stages': stages, 'caption': '白 = 掩码，橙 = 选中的 SAM 3D 候选；第二行 = 每张照片生成的候选' if cmp else '原生成模型'}
        if cmp_dir(v) and (cmp_dir(v) / f'{oid}.jpg').exists():
            shutil.copy(cmp_dir(v) / f'{oid}.jpg', PAGES / f'pipeline-{eid[:8]}.jpg'); p['url'] = f'measurement-layer/pipeline-{eid[:8]}.jpg'
        layer['pipelines'][eid] = p
        # confidence + facts + box
        level = 'low' if fails else (bx or {}).get('confidence', 'unverified')
        layer['confidence'][eid] = {'level': level, 'label': LEVEL[level], 'reasons': list((bx or {}).get('highlightReasons') or []), 'missing': fails}
        facts = [{'label': '明显错误检查 G1–G8', 'kind': 'check', 'text': '；'.join(fails) if fails else '通过（无下沉、悬空、穿插、炸开、轮廓不符）'},
                 {'label': '组装', 'kind': 'note', 'text': iou_txt + lowest}]
        if oid in field and fv:
            key, name, field_cm = field[oid]
            val = fv.get(f'{key}_cm'); err = fv.get(f'{key}_err')
            if val is not None:
                facts.append({'label': '现场值（几何层评判器）', 'kind': 'field', 'text': f"{name}：评判器 {val} cm，现场 {field_cm} cm" + (f"（误差 {err:+} cm）" if err is not None else '')})
        layer['facts'][eid] = facts
        if bx:
            layer['boxes'][eid] = bx
    out = PAGES / f"{res['publicationId']}.json"
    out.write_text(json.dumps(layer, ensure_ascii=False, indent=1) + '\n')
    print(v, res['publicationId'], 'entities', len(layer['pipelines']), 'boxes', len(layer['boxes']), 'low', sum(1 for c in layer['confidence'].values() if c['level'] == 'low'), '->', out.name)


if __name__ == '__main__':
    for v in sys.argv[1:]:
        build(v)
