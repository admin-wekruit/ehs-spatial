"""Tier-2 comparison: the ORIGINAL layer-stage checks (S8-S14) on the published 090 report and on the three module-swapped
publications, per object. Old = the stage results the published report was built from (research-notes / scratch) and its layer;
variants = swap-runs/<variant>-stages/<stage>/results.json (modal_apps/workcell_layer_trial.py, result under 'result').

    nice python compare_layers.py   -> layers-table.md, layers-results.json (here)
"""
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
SP = Path(os.environ.get('SWAP_SCRATCH', '/private/tmp/claude-501/-Users-adam-Desktop-panoptes-public/1fd9a1db-e580-4bfc-8110-119a1cc38a99/scratchpad'))
RN = Path(os.environ.get('SWAP_NOTES', '/Users/adam/Desktop/panoptes-public/research-notes'))
PAGES = Path(os.environ.get('PANOPTES_PAGES', '/Users/adam/Desktop/panoptes-public/panoptes-workcell-pages/workcell-photo-direct/report/measurement-layer'))
OBJECTS = ['left_light_curtain', 'right_light_curtain', 'left_fence', 'right_fence', 'left_post', 'right_post', 'robot', 'cart', 'guard']
ZH = {'left_light_curtain': '左光幕', 'right_light_curtain': '右光幕', 'left_fence': '左围栏', 'right_fence': '右围栏', 'left_post': '左防撞柱',
      'right_post': '右防撞柱', 'robot': '机器人', 'cart': '料车', 'guard': '防护板'}
LV = {'high': '高', 'medium': '中', 'low': '低', 'unverified': '未验证', None: '—'}
VERDICT = {'ok': '一致', 'depth_off': '深度偏差', 'ambiguous': '含糊', 'inconclusive': '无法判断', 'not_photo_consistent': '不符', None: '—'}
FIELD = {'left_light_curtain': 24, 'right_light_curtain': 24, 'left_fence': 20, 'right_fence': 20}
VARIANTS = ['pi3x+sam3d', 'mvs+recgen', 'mvs+sam3d', 'mvs-fill+sam3d']
DIRS = {'pi3x+sam3d': 'pi3x-sam3d', 'mvs+recgen': 'mvs-recgen', 'mvs+sam3d': 'mvs-sam3d', 'mvs-fill+sam3d': 'mvs-fill-sam3d'}
# assembly v2 (floor-contact hinge) on a copy of every version's run directory: stage dirs swap-runs/v2-<name>-stages (run_v2_chain.sh)
for _k in ['pi3x+recgen'] + VARIANTS[:]:
    if (SP / 'swap-runs' / f"v2-{_k.replace('+', '-')}-stages").exists():
        VARIANTS.append(f'{_k} 组装v2'); DIRS[f'{_k} 组装v2'] = f"v2-{_k.replace('+', '-')}"


def load(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def entity_map(doc):
    out = {}
    for e in doc['entities']:
        for l in e.get('lineage') or []:
            if isinstance(l, dict) and l.get('operation') == 'offline_import' and l.get('sourceRecordId'):
                out.setdefault(l['sourceRecordId'], e['id'])
    return out


def stage(variant, name):
    r = load(SP / 'swap-runs' / f'{DIRS[variant]}-stages' / name / 'results.json')
    return (r or {}).get('result')


def old_stages():
    """The published report's stage results (workcell_view_checks.py wraps them under results[check])."""
    def vc(p, key):
        d = load(p); return (d or {}).get('results', {}).get(key) or (d or {}).get('result')
    return {'floor': vc(RN / 'workcell-floor-check-2026-10-05/090-results.json', 'floor'),
            'shape': {'rows': (load(RN / 'workcell-shape-check-2026-10-05-b/results.json') or {}).get('rows', [])},
            'box_faces': vc(SP / 'checks/mvpf-run-090-v4/results.json', 'box_faces'),
            'lower_edge': vc(SP / 'checks/loweredge-final-090/results.json', 'lower_edge'),
            'obvious_errors': vc(SP / 'checks/obvious-baseline-090-final/results.json', 'obvious_errors'),
            'plane_stereo': ((load(RN / 'workcell-plane-stereo-2026-10-05/results.json') or {}).get('reports', {}).get('090')),
            'lines': vc(SP / 'checks/lines-final-090/results.json', 'lines')}


def per_object(st, eid, layer=None):
    """World-independent facts of one object from the stage results (+ the layer's box / confidence when given)."""
    o = {}
    fl = ((st.get('floor') or {}).get('objects') or {}).get(eid)
    if fl:
        o['floor'] = {'bottomCm': round(100 * fl['bottomM'], 1), 'status': fl.get('status')}
    rows = [r for r in (st.get('shape') or {}).get('rows', []) if r.get('entityId') == eid and r.get('model') == 'displayed']
    if rows:
        r = rows[0]; o['shape'] = {'verdict': r.get('verdict'), 'depthChangeCm': round(100 * (r.get('depthChangeM') or 0), 1), 'ncc': r.get('nccAtModel')}
    bx = ((st.get('box_faces') or {}).get('boxes') or {}).get(eid) or ((layer or {}).get('boxes') or {}).get(eid)
    if bx:
        o['box'] = {'sizeCm': [round(100 * x, 1) for x in bx['sizeM']], 'bottomCm': round(100 * bx['bottomM'], 1),
                    'conf': {k: bx['dims'][k]['confidence'] for k in ('L', 'W', 'H', 'bottom')}, 'highlight': bx.get('highlight'),
                    'reasons': bx.get('highlightReasons') or [], 'capped': bool(bx.get('capped')) if 'capped' in bx else None}
    le = [t for t in (st.get('lower_edge') or {}).get('targets', []) if t.get('entityId') == eid]
    if le:
        o['lowerEdge'] = {(t.get('part') or 'object'): {'valueCm': t.get('valueCm'), 'reason': t.get('reason'), 'estimateCm': t.get('estimateCm'), 'sigmaCm': t.get('sigmaCm')} for t in le}
    oe = ((st.get('obvious_errors') or {}).get('objects') or {}).get(eid)
    if oe:
        o['obvious'] = {'severity': oe.get('severity'), 'failed': [f.get('zh') or f.get('text') for f in oe.get('failed', [])],
                        'bottomCm': oe.get('bottomCm'), 'iou': oe.get('iou')}
    ps = ((st.get('plane_stereo') or {}).get('objects') or {}).get(eid)
    if ps:
        parts = [p for p in ps.get('parts') or [] if p.get('verdict')]
        if parts:
            b = max(parts, key=lambda p: p.get('pixels', 0))
            o['planeStereo'] = {'verdict': b.get('verdict'), 'rayChangeCm': b.get('rayChangeCm'), 'ncc': b.get('nccAtBest')}
    pf = ((st.get('plane_facets') or {}).get('objects') or {}).get(eid)
    if pf:
        o['planeFacets'] = {'planes': [{'inclinationDeg': round(p.get('inclinationDeg', 0), 1), 'points': p.get('points')} for p in (pf.get('planes') or [])[:4]],
                            'folds': [{'interiorDeg': round(f.get('interiorDeg', 0), 1), 'lengthM': f.get('lengthM')} for f in (pf.get('folds') or [])[:4]],
                            'verdict': pf.get('verdict')}
    if layer and eid in (layer.get('confidence') or {}):
        c = layer['confidence'][eid]; o['confidence'] = {'level': c.get('level'), 'missing': c.get('missing') or []}
    return o


def cell_text(o):
    parts = []
    if 'box' in o:
        b = o['box']; parts.append('盒子 ' + '×'.join(f"{x:.0f}" for x in b['sizeCm']) + f" cm，离地 {b['bottomCm']:.1f}（{'/'.join(LV[b['conf'][k]] for k in ('L', 'W', 'H', 'bottom'))}）")
    if 'floor' in o:
        parts.append(f"最低点 {o['floor']['bottomCm']:+.1f} cm（{ {'sinks': '下沉', 'on_floor': '贴地', 'hovers': '悬空', 'above': '离地'}.get(o['floor']['status'], o['floor']['status'])}）")
    if 'shape' in o:
        dc = o['shape'].get('depthChangeCm')
        parts.append(f"形状校验 {VERDICT.get(o['shape']['verdict'], o['shape']['verdict'])}" + (f"（最佳深度 {dc:+.0f} cm）" if isinstance(dc, (int, float)) else ''))
    if 'planeStereo' in o:
        rc = o['planeStereo'].get('rayChangeCm')
        parts.append(f"平面立体 {o['planeStereo']['verdict']}" + (f"（{rc:+.0f} cm）" if isinstance(rc, (int, float)) else ''))
    if 'lowerEdge' in o:
        def le_text(k, v):
            if v['valueCm'] is not None:
                return f"{k}: {v['valueCm']:.1f} cm"
            if v.get('estimateCm') is not None:
                return f"{k}: 估 {v['estimateCm']:.0f}±{v.get('sigmaCm') or 0:.0f}（{v['reason']}）"
            return f"{k}: 无（{v['reason']}）"
        parts.append('下沿 ' + '；'.join(le_text(k, v) for k, v in o['lowerEdge'].items()))
    if 'planeFacets' in o:
        parts.append('三视图平面 ' + '，'.join(f"{p['inclinationDeg']:.0f}°" for p in o['planeFacets']['planes']) + '；折角 ' + '，'.join(f"{f['interiorDeg']:.0f}°" for f in o['planeFacets']['folds']))
    if 'obvious' in o:
        parts.append('G1–G8 ' + ('；'.join(o['obvious']['failed']) if o['obvious']['failed'] else '通过'))
    if 'confidence' in o:
        parts.append(f"置信度 {LV[o['confidence']['level']]}")
    return '<br>'.join(parts) if parts else '—'


G1 = {}


def g1_text(name):
    g = G1.get(name)
    if not g:
        return '—'
    import re
    m = re.search(r'covered (\d+) %', ' '.join(f[0] if isinstance(f, (list, tuple)) else str(f) for f in g.get('failed', [])) + ' ' + ' '.join(str(w) for w in g.get('warnings', [])))
    pct = m.group(1) + '%' if m else ('≥ 90%' if g.get('severity') != 'obvious' else '?')
    return f"{pct}（{ {'obvious': '明显错误', 'warning': '警告', None: '通过'}.get(g.get('severity'), g.get('severity')) }）"


def main():
    old_view = load(SP / 'sept/new-view.json')['publication']['snapshot']['revision']['document']
    old_ents = entity_map(old_view)
    old_layer = load(PAGES / '4b58dbd2-3846-47f2-af97-57eaa108753c.json')
    st_old = old_stages()
    results = {'pi3x+recgen': {o: per_object(st_old, old_ents[o], old_layer) for o in OBJECTS if o in old_ents}}
    summaries = {'pi3x+recgen': (st_old.get('obvious_errors') or {}).get('summary')}
    G1['pi3x+recgen'] = ((st_old.get('obvious_errors') or {}).get('report') or {}).get('G1')
    for v in VARIANTS:
        em = load(SP / 'swap-runs' / f'{DIRS[v]}-stages' / 'entity-map.json') or {}
        st = {name: stage(v, name) for name in ('floor', 'shape', 'box_faces', 'lower_edge', 'obvious_errors', 'plane_stereo', 'lines', 'transfer', 'clearance', 'plane_facets')}
        results[v] = {o: per_object(st, em[o]) for o in OBJECTS if o in em}
        summaries[v] = (st.get('obvious_errors') or {}).get('summary')
        G1[v] = ((st.get('obvious_errors') or {}).get('report') or {}).get('G1')
        results[v]['_stages'] = {k: bool(x) for k, x in st.items()}
        pf = st.get('plane_facets')
        if pf:
            results[v]['_planeFacets'] = pf.get('objects')
    (HERE / 'layers-results.json').write_text(json.dumps({'objects': results, 'obviousSummary': summaries}, indent=1, ensure_ascii=False) + '\n')
    names = ['pi3x+recgen'] + VARIANTS
    L = ['# 090 模块替换对比（第二层：原流程后段各项检查，逐物体）', '',
         '每格：盒子 L×W×H 与离地（括号内为长/宽/高/离地的置信度）；模型最低点对各自地面；三照片形状校验；平面立体；下沿实测（现场 24 / 20）；G1–G8；层置信度。',
         '原版 = 已发布报告当时的检查结果；四个新版本 = 同一检查脚本在各自本地导入的发布视图上重跑（mvs-fill+sam3d = MVS 几何在掩码内用 MoGe-3 补洞后重新生成）。', '',
         '| 物体 | ' + ' | '.join(names) + ' |', '|---|' + '---|' * len(names)]
    for o in OBJECTS:
        L.append(f"| {ZH[o]} | " + ' | '.join(cell_text(results[n].get(o, {})) for n in names) + ' |')
    L += ['', '## 报告级', '', '| 指标 | ' + ' | '.join(names) + ' |', '|---|' + '---|' * len(names)]
    L.append('| G1–G8 明显错误物体数 | ' + ' | '.join(str(len((summaries[n] or {}).get('obviousObjects', [])) if summaries[n] else '—') for n in names) + ' |')
    L.append('| G1 地面覆盖（照片里物体前/下看得见的地面有多少画了出来） | ' + ' | '.join(g1_text(n) for n in names) + ' |')
    L.append('| 跑了的检查 | ' + ' | '.join('全部（发布时）' if n == 'pi3x+recgen' else ', '.join(k for k, ok in results[n].get('_stages', {}).items() if ok) for n in names) + ' |')
    (HERE / 'layers-table.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
