"""Cell 030 summary of the three assembly-v2 versions (030-v2-pi3x-recgen, 030-v2-pi3x-sam3d, 030-v2-mvs-fill-sam3d) plus the
published report's own numbers where they exist: per object the assembly's silhouette IoU / depth residual / lowest point
(result/comparisons.json), the original checks' floor status, box and G1-G8 result (swap-runs/<V>-stages), and the geometry's
030 field values (field-values-mvs-fill.json). -> table-030.md, results-030.json (here)

    nice python tables_030.py
"""
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
SP = Path(os.environ.get('SWAP_SCRATCH', '/private/tmp/claude-501/-Users-adam-Desktop-panoptes-public/1fd9a1db-e580-4bfc-8110-119a1cc38a99/scratchpad'))
PLATFORM = Path(os.environ.get('PANOPTES_PLATFORM', '/Users/adam/Desktop/Tesla/panoptes-platform'))
OBJECTS = ['left_light_curtain', 'right_light_curtain', 'right_fence', 'left_post', 'right_post', 'robot', 'cart', 'guard']
ZH = {'left_light_curtain': '左光幕', 'right_light_curtain': '右光幕', 'right_fence': '右围栏', 'left_post': '左防撞柱', 'right_post': '右防撞柱',
      'robot': '机器人', 'cart': '料车', 'guard': '防护板'}
VERSIONS = {'pi3x+recgen 组装v2': '030-v2-pi3x-recgen', 'pi3x+sam3d 组装v2': '030-v2-pi3x-sam3d', 'mvs-fill+sam3d 组装v2': '030-v2-mvs-fill-sam3d'}
FLOOR = {'sinks': '插地', 'hovers': '悬空', 'on_floor': '贴地', 'above': '离地'}


def load(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def main():
    out = {'versions': {}, 'objects': {}}
    names = [n for n, v in VERSIONS.items() if (PLATFORM / '.platform/swap-20261007' / v / 'result.json').exists()]
    for n in names:
        v = VERSIONS[n]; res = load(PLATFORM / '.platform/swap-20261007' / v / 'result.json'); run = Path(res['run']); S = res['nativeToMeters']
        st = SP / 'swap-runs' / f'{v}-stages'; ents = load(st / 'entity-map.json') or {}
        stage = lambda name: (load(st / name / 'results.json') or {}).get('result') or {}
        ob, fl, bx = stage('obvious_errors'), stage('floor').get('objects') or {}, stage('box_faces').get('boxes') or {}
        comps = {c['object_id']: c for c in load(run / 'result/comparisons.json')['objects']}
        rows = {}
        for oid in OBJECTS:
            c = comps.get(oid)
            if not c:
                continue
            views = [x['generated_refined'] for x in c['views']]
            ious = [x['visible_iou'] for x in views]; deps = [x['relative_depth_p50'] for x in views if x['relative_depth_p50'] is not None]
            fin = (c['refinement'].get('final') or {}).get('floor') or {}
            eid = ents.get(oid); o = ((ob.get('objects') or {}).get(eid) or {}); f = fl.get(eid) or {}; b = bx.get(eid) or {}
            rows[oid] = dict(meanIou=round(sum(ious) / len(ious), 3), ious=[round(x, 2) for x in ious], depthP50=round(100 * sum(deps) / len(deps), 1) if deps else None,
                             lowestCm=round(100 * S * fin['lowest_native'], 1) if fin.get('lowest_native') is not None else None,
                             floor=FLOOR.get(f.get('status'), f.get('status')), bottomCm=round(100 * f['bottomM'], 1) if f else None,
                             obvious=[x['zh'] for x in o.get('failed', [])], box=([round(100 * x) for x in b['sizeM']] if b else None), boxConf=b.get('confidence'))
        summ = (ob.get('summary') or {})
        out['versions'][n] = dict(variant=v, publicationId=res['publicationId'], nativeToMeters=S, obviousObjects=len(summ.get('obviousObjects', [])),
                                  G1=((ob.get('report') or {}).get('G1') or {}).get('failed') and 'fail' or 'pass', rows=rows)
    fv = load(HERE / 'field-values-mvs-fill.json') or {}
    L = ['# 030 模块替换（组装 v2）：三个版本', '',
         '| 版本 | 几何 | 补全 | 发布 id | G1–G8 明显错误物体 | 轮廓 IoU 均值 | 最低点 < −2 cm 的物体 |', '|---|---|---|---|---|---|---|']
    for n in names:
        V = out['versions'][n]; rows = V['rows']
        mi = sum(r['meanIou'] for r in rows.values()) / max(len(rows), 1); sinks = [ZH[o] for o, r in rows.items() if r['lowestCm'] is not None and r['lowestCm'] < -2]
        g = 'MVS + 补洞' if 'mvs-fill' in n else 'Pi3X'; comp = 'RecGen' if 'recgen' in n else 'SAM 3D'
        L.append(f"| {n} | {g} | {comp} | `{V['publicationId']}` | {V['obviousObjects']} | {mi:.3f} | {', '.join(sinks) or '无'} |")
    L += ['', '## 逐物体', '', '| 物体 | 指标 | ' + ' | '.join(names) + ' |', '|---|---|' + '---|' * len(names)]
    for oid in OBJECTS:
        cell = lambda fn: ' | '.join((fn(out['versions'][n]['rows'][oid]) if oid in out['versions'][n]['rows'] else '—') for n in names)
        L.append(f"| {ZH[oid]} | 轮廓 IoU（照片 1/2） | " + cell(lambda r: f"{r['meanIou']:.2f}（" + '/'.join(f'{x:.2f}' for x in r['ious']) + '）') + ' |')
        L.append(f"|  | 深度残差 p50 | " + cell(lambda r: f"{r['depthP50']:.1f}%" if r['depthP50'] is not None else '—') + ' |')
        L.append(f"|  | 组装后最低点 / 地面接触 | " + cell(lambda r: (f"{r['lowestCm']:+.1f} cm" if r['lowestCm'] is not None else '—') + (f"，{r['floor']}" if r['floor'] else '')) + ' |')
        L.append(f"|  | 盒子 L×W×H cm（置信度） | " + cell(lambda r: ('×'.join(map(str, r['box'])) + f"（{r['boxConf']}）") if r['box'] else '—') + ' |')
        L.append(f"|  | G1–G8 | " + cell(lambda r: '；'.join(r['obvious']) or '通过') + ' |')
    L += ['', '## 几何层（公平评判器，030 现场值：左右罩壳下沿各 24 cm）', '', '| | MVS | MVS + 补洞 |', '|---|---|---|']
    a, b = fv.get('mvs-da3-base', {}), fv.get('mvs-fill', {})
    for k, lab in (('housing030L', '左罩壳下沿'), ('housing030R', '右罩壳下沿')):
        L.append(f"| {lab} | {a.get(k + '_cm')}（{a.get(k + '_err', 0):+}） | {b.get(k + '_cm')}（{b.get(k + '_err', 0):+}） |")
    L.append(f"| 急停 4 % 约束最大偏差 | {a.get('estopMaxDevPct030')}% | {b.get('estopMaxDevPct030')}% |")
    L.append(f"| 急停尺度（m / 原生单位） | {a.get('estopNativeToMeters030')} | {b.get('estopNativeToMeters030')} |")
    L.append(f"| 4 个现场值（090 + 030）平均 / 最大误差 | {a.get('maeCm4values')} / {a.get('maxAbsErrCm')} cm | {b.get('maeCm4values')} / {b.get('maxAbsErrCm')} cm |")
    (HERE / 'results-030.json').write_text(json.dumps(out, ensure_ascii=False, indent=1) + '\n')
    (HERE / 'table-030.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
