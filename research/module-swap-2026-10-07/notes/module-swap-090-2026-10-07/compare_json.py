"""Tier-1 comparison of the 090 module-swap versions from their JSON outputs only (no meshes loaded):
assembly result (per photo silhouette IoU, boundary error, depth residual; refinement), generation records, point coverage
inside the generation mask, floor statistics, camera heights, and the geometry-level field values from the fair evaluator.

Versions (same pipeline, modules swapped):  pi3x+recgen (published), pi3x+sam3d, mvs+recgen, mvs+sam3d, mvs-fill+sam3d
(geometry module v2: fill_geometry.py; field values of that geometry from field_values_fill.py).
    nice python compare_json.py  -> results.json, table.md (here)
"""
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
SP = Path(os.environ.get('SWAP_SCRATCH', '/private/tmp/claude-501/-Users-adam-Desktop-panoptes-public/1fd9a1db-e580-4bfc-8110-119a1cc38a99/scratchpad'))
RN = Path(os.environ.get('SWAP_NOTES', '/Users/adam/Desktop/panoptes-public/research-notes'))
LUCIDA = Path(os.environ.get('PANOPTES_RUNS', '/Users/adam/Desktop/panoptes-public/panoptes-serving/outputs/candidate-evaluation')) / 'lucida-replica-01'
MVS = SP / 'checks/bbab-export-090-mvs-scipyba'
FILL = SP / 'checks/bbab-export-090-mvs-fill'          # geometry module v2: the same MVS run + MoGe-3 in-mask hole fill (fill_geometry.py)
OBJECTS =['left_light_curtain', 'right_light_curtain', 'left_fence', 'right_fence', 'left_post', 'right_post', 'robot', 'cart', 'guard']
ZH = {'left_light_curtain': '左光幕', 'right_light_curtain': '右光幕', 'left_fence': '左围栏', 'right_fence': '右围栏', 'left_post': '左防撞柱',
      'right_post': '右防撞柱', 'robot': '机器人', 'cart': '料车', 'guard': '防护板'}
SCALE = {'pi3x': 1.28601, 'mvs': 3.2371372068568487, 'mvs-fill': 3.2371372068568487}   # published e-stop scale; the MVS run's own e-stop scale (m per native unit; the fill touches no e-stop / floor pixel)
GEOMS = [('pi3x', 'Pi3X'), ('mvs', 'MVS'), ('mvs-fill', 'MVS+补洞')]
PHOTO_ORDER = ['frame_0001', 'frame_0002', 'frame_0003']


def load(p):
    return json.loads(Path(p).read_text())


def comps(path):
    d = load(path)
    objs = d['objects'] if 'objects' in d else d['comparisons']['objects']
    return {o['object_id']: o for o in objs}


def selection(path, key='variant'):
    d = load(path)
    src = next((d[k] for k in ('entities', 'choices') if isinstance(d.get(k), dict)), d)
    return {oid: v[key] for oid, v in src.items() if isinstance(v, dict) and key in v and oid in OBJECTS}


def first_existing(*paths):
    return next((p for p in paths if p.exists()), paths[-1])


VERSIONS = {
    'pi3x+recgen': dict(geometry='pi3x', completion='recgen', run=LUCIDA, comps=lambda: comps(LUCIDA / 'result/comparisons.json'),
                        record=lambda oid: load(LUCIDA / 'generation' / oid / 'output.json'), variant=lambda oid: 'recgen'),
    'pi3x+sam3d': dict(geometry='pi3x', completion='sam3d', run=LUCIDA, out=SP / 'checks/completionAB-090',
                       sel=first_existing(HERE / 'cmp-pi3x/results.json', RN / 'completion-ab-090-2026-10-06/results.json')),
    'mvs+recgen': dict(geometry='mvs', completion='recgen', run=SP / 'swap-runs/mvs-recgen', out=SP / 'swap-runs/mvs-recgen-ab',
                       comps=lambda: comps(SP / 'swap-runs/mvs-recgen-ab/assembly/recgen/comparisons.json'),
                       record=lambda oid: load(SP / 'swap-runs/mvs-recgen/generation' / oid / 'output.json'), variant=lambda oid: 'recgen'),
    'mvs+sam3d': dict(geometry='mvs', completion='sam3d', run=MVS, out=SP / 'mvs090/completion2',
                      sel=first_existing(HERE / 'cmp-mvs/results.json', RN / 'workcell-clean-report-090-2026-10-06/data/compose-record.json')),
    'mvs-fill+sam3d': dict(geometry='mvs-fill', completion='sam3d', run=FILL, out=SP / 'swap-runs/mvs-fill-ab', sel=HERE / 'cmp-mvs-fill/results.json'),
}
V2 = SP / 'swap-runs/v2'   # assembly v2 (floor-contact hinge in assemble_lucida_scene.refine) re-run on a copy of every version's run directory


def v2_version(name, geometry, completion):
    run = V2 / name
    rec = lambda oid: load(run / 'generation' / oid / 'output.json')
    def variant(oid):
        if completion == 'recgen':
            return 'recgen'
        s = rec(oid).get('selection')
        return s.get('variant', 'sam3d') if isinstance(s, dict) else (s or 'sam3d')
    return dict(geometry=geometry, completion=completion, run=run, assembly='v2', comps=lambda: comps(run / 'result/comparisons.json'), record=rec, variant=variant)


VERSIONS.update({f'{k} 组装v2': v2_version(k.replace('+', '-'), VERSIONS[k]['geometry'], VERSIONS[k]['completion']) for k in list(VERSIONS)})


def sam3d_version(v):
    sel = selection(v['sel'])
    v['variant'] = lambda oid: sel.get(oid, 'sam3d')
    cache = {}
    def comps_sel():
        out = {}
        for oid in OBJECTS:
            var = sel.get(oid, 'sam3d')
            if var not in cache:
                cache[var] = comps(v['out'] / 'assembly' / var / 'comparisons.json')
            if oid in cache[var]:
                out[oid] = cache[var][oid]
        return out
    v['comps'] = comps_sel
    v['record'] = lambda oid: load(v['out'] / sel.get(oid, 'sam3d') / 'record.json')['objects'][oid]
    return v


def coverage(v):
    """Point-map coverage inside the generation mask per object (largest-mask view): SAM 3D record mask_depth_pixels / mask_pixels."""
    rec = load(v['out'] / 'sam3d/record.json')['objects']
    return {oid: rec[oid]['mask_depth_pixels'] / rec[oid]['mask_pixels'] for oid in OBJECTS if oid in rec}


def obj_row(c, rec, variant):
    views = {x['frame_id']: x for x in c['views']}
    per = {}
    for f in PHOTO_ORDER:
        if f in views:
            a, b = views[f]['generated_initial'], views[f]['generated_refined']
            per[f[-1]] = dict(iou0=round(a['visible_iou'], 3), iou=round(b['visible_iou'], 3), boundary=round(b['boundary_error_image_height'], 4),
                              depthP50=round(b['relative_depth_p50'], 4), depthP95=round(b['relative_depth_p95'], 4))
    ious = [p['iou'] for p in per.values()]
    r = c['refinement']
    row = dict(variant=variant, perPhoto=per, meanIou=round(sum(ious) / len(ious), 3), minIou=round(min(ious), 3),
               meanDepthP50=round(sum(p['depthP50'] for p in per.values()) / len(per), 4),
               meanBoundary=round(sum(p['boundary'] for p in per.values()) / len(per), 4),
               refine=dict(evaluations=r['evaluations'], seconds=round(r['seconds'], 1), loss0=round(r['initial']['loss'], 4), loss=round(r['final']['loss'], 4)),
               faces=c.get('faces'), vertices=c.get('vertices'))
    if rec:
        row['generation'] = dict(seconds=round(rec.get('inference_seconds', rec.get('seconds', 0)) or 0, 1),
                                 views=rec.get('view_count', 1), photo=(rec.get('anchor_frame') or rec.get('frame_id', ''))[-1:])
    return row


def floor_stats(run, scale):
    f = load(run / 'evidence/floor.json')
    return dict(points=f.get('total_observed_points'), inliers=f.get('inlier_points'),
                residualMedianCm=round(100 * scale * f['all_point_residual_median_native'], 2),
                residualP95Cm=round(100 * scale * f['all_point_residual_p95_native'], 2),
                cameraHeightsCm=[round(100 * scale * h, 1) for h in f['camera_signed_heights_native']])


def field_values():
    """090 field values per geometry from the fair evaluator (geometry-backbone-ab results rows, primary configuration)."""
    rows = load(RN / 'geometry-backbone-ab-2026-10-06/results.json')['rows']
    pick = {'pi3x': next(r for r in rows if r['label'] == 'Pi3X (published)'),
            'mvs': next(r for r in rows if r['label'].startswith('MVS, CERT 0.05 (DA3-BASE start)'))}
    out = {}
    for g, r in pick.items():
        v = r['values']
        out[g] = {'housing090R_cm': round(v['090R housing']['cm'], 1), 'housing090R_err': round(v['090R housing']['errCm'], 1),
                  'fence090R_cm': round(v['090R fence']['cm'], 1), 'fence090R_err': round(v['090R fence']['errCm'], 1),
                  'housing090L_cm': round(r['listed090L'], 1), 'estopMaxDevPct090': round(r['gate']['090']['maxDeviationPct'], 2),
                  'maeCm4values': round(r['maeCm'], 2), 'maxAbsErrCm': round(r['maxAbsErrCm'], 2),
                  'floorP95Cm090': r['floorP95Cm']['090'], 'cameraHeightRangeCm': round(r['cameraHeightRangeCm'], 1), 'label': r['label']}
    return out


METRICS = {  # world-independent per-object numbers from compare.py (meshes): extents / lowest point / gates; uniform rerun (cmp-*) preferred
    'pi3x+recgen': (first_existing(HERE / 'cmp-pi3x/results.json', RN / 'completion-ab-090-2026-10-06/results.json'), 'recgen'),
    'pi3x+sam3d': (first_existing(HERE / 'cmp-pi3x/results.json', RN / 'completion-ab-090-2026-10-06/results.json'), 'sam3dFixed'),
    'mvs+recgen': (HERE / 'cmp-mvs/results.json', 'recgen'),
    'mvs+sam3d': (first_existing(HERE / 'cmp-mvs/results.json', SP / 'mvs090/g2p2/results.json'), 'sam3dFixed'),
    'mvs-fill+sam3d': (HERE / 'cmp-mvs-fill/results.json', 'sam3dFixed'),
}


def metrics_table(names):
    L = ['', '## 还原结果（世界无关的数字；来自 compare.py，网格统计）', '',
         '尺寸 = 模型沿统一地面坐标的外接范围 L×W×H；最低点 = 顶点高度 p0.5 对各自地面；检查 = 下沉 / 悬空 / 深度 / 轮廓 / 占了看得见的空地（无测量盒门）。', '',
         '| 物体 | 指标 | ' + ' | '.join(names) + ' |', '|---|---|' + '---|' * len(names)]
    data = {}
    for n in names:
        if n not in METRICS:
            continue
        path, key = METRICS[n]
        if path.exists():
            d = load(path); data[n] = {oid: d[oid][key] for oid in OBJECTS if oid in d and key in d[oid]}
    if not data:
        return []
    for oid in OBJECTS:
        c = lambda fn: ' | '.join(fn(data[n][oid]) if n in data and oid in data[n] else '—' for n in names)
        L.append(f"| {ZH[oid]} | 尺寸 L×W×H cm | " + c(lambda m: '×'.join(f"{100 * x:.0f}" for x in m['extentM'])) + ' |')
        L.append(f"|  | 最低点 cm | " + c(lambda m: f"{m['lowestCm']:+.1f}") + ' |')
        L.append(f"|  | 检查 | " + c(lambda m: '；'.join(m['gates']) or '通过') + ' |')
    return L


def attribution(objects, names):
    """Which module each change comes from: completion effect = sam3d - recgen at fixed geometry; geometry effect = mvs - pi3x at
    fixed completion. Mean over the objects present in both versions."""
    def mean_delta(a, b, key, scale=1):
        ds = [scale * (objects[o][b][key] - objects[o][a][key]) for o in OBJECTS if a in objects.get(o, {}) and b in objects.get(o, {})]
        return (round(sum(ds) / len(ds), 3), len(ds)) if ds else (None, 0)
    pairs = [('补全效果（几何 = Pi3X）', 'pi3x+recgen', 'pi3x+sam3d'), ('补全效果（几何 = MVS）', 'mvs+recgen', 'mvs+sam3d'),
             ('几何效果（补全 = RecGen）', 'pi3x+recgen', 'mvs+recgen'), ('几何效果（补全 = SAM 3D）', 'pi3x+sam3d', 'mvs+sam3d'),
             ('补洞效果（几何 MVS → MVS+补洞，补全 = SAM 3D）', 'mvs+sam3d', 'mvs-fill+sam3d'), ('两个都换+补洞 对 原版', 'pi3x+recgen', 'mvs-fill+sam3d'),
             ('组装 v2 效果（地面接触约束；原版）', 'pi3x+recgen', 'pi3x+recgen 组装v2'), ('组装 v2 效果（mvs-fill+sam3d）', 'mvs-fill+sam3d', 'mvs-fill+sam3d 组装v2'),
             ('两个都换+补洞+组装 v2 对 原版', 'pi3x+recgen', 'mvs-fill+sam3d 组装v2'), ('两个都换+补洞+组装 v2 对 原版+组装 v2', 'pi3x+recgen 组装v2', 'mvs-fill+sam3d 组装v2')]
    L = ['', '## 归因：每处差别来自哪个模块（9 个物体的平均变化）', '', '| 对比 | 轮廓 IoU 均值 | 深度残差 p50 | 边界误差 | 物体数 |', '|---|---|---|---|---|']
    for label, a, b in pairs:
        iou, n = mean_delta(a, b, 'meanIou'); dep, _ = mean_delta(a, b, 'meanDepthP50', 100); bd, _ = mean_delta(a, b, 'meanBoundary', 100)
        if n:
            L.append(f"| {label}：{b} − {a} | {iou:+.3f} | {dep:+.2f} 点 | {bd:+.2f} 点 | {n} |")
        else:
            L.append(f"| {label}：{b} − {a} | — | — | — | 0 |")
    return L


def main():
    for k in ('pi3x+sam3d', 'mvs+sam3d', 'mvs-fill+sam3d'):
        sam3d_version(VERSIONS[k])
    results = {'objects': {}, 'runs': {}, 'missing': []}
    for name, v in VERSIONS.items():
        try:
            cs = v['comps']()
        except FileNotFoundError as e:
            results['missing'].append(f'{name}: {e}'); continue
        for oid in OBJECTS:
            if oid not in cs:
                results['missing'].append(f'{name}: {oid}'); continue
            try:
                rec = v['record'](oid)
            except (FileNotFoundError, KeyError):
                rec = None
            row = obj_row(cs[oid], rec, v['variant'](oid))
            fl = (cs[oid]['refinement'].get('final') or {}).get('floor')   # assembly v2 records the hinge's lowest point (0.5th percentile)
            if fl and fl.get('lowest_native') is not None:
                row['lowestCm'] = round(100 * SCALE[v['geometry']] * fl['lowest_native'], 1)
                row['floorPenalty'] = round(fl['penalty'], 4)
            results['objects'].setdefault(oid, {})[name] = row
    for g, k, run in (('pi3x', 'pi3x+sam3d', LUCIDA), ('mvs', 'mvs+sam3d', MVS), ('mvs-fill', 'mvs-fill+sam3d', FILL)):
        if not (VERSIONS[k]['out'] / 'sam3d/record.json').exists():
            results['missing'].append(f'{k}: sam3d/record.json'); continue
        cov = coverage(VERSIONS[k])
        results['runs'][g] = dict(scale=SCALE[g], floor=floor_stats(run, SCALE[g]), coverage={o: round(c, 3) for o, c in cov.items()},
                                  meanCoverage=round(sum(cov.values()) / len(cov), 3))
    results['fieldValues'] = field_values()
    if (HERE / 'field-values-mvs-fill.json').exists():   # field_values_fill.py: the same fair config() on the filled geometry's analysis
        results['fieldValues']['mvs-fill'] = load(HERE / 'field-values-mvs-fill.json')['mvs-fill']
    (HERE / 'results.json').write_text(json.dumps(results, indent=1, ensure_ascii=False) + '\n')

    names = list(VERSIONS)
    L = ['# 090 模块替换对比（第一层：组装 / 生成 / 几何 JSON，同一组装代码）', '',
         '## 列的含义（四个版本，流程相同，只换模块）', '',
         '| 列 | 几何模块（S2） | 补全模块（S4） | 用途 | 数据来源 |',
         '|---|---|---|---|---|',
         '| pi3x+recgen | Pi3X（CC BY-NC，非商用） | RecGen（TRI 非商用） | 原报告，基线 | `lucida-replica-01/result/comparisons.json`（已发布） |',
         '| pi3x+sam3d | Pi3X | SAM 3D（SAM License，可商用） | 只换补全：看补全模块的影响 | 同一组装代码跑 SAM 3D 候选：`checks/completionAB-090/assembly/*` |',
         '| mvs+recgen | MVS（可商用，不含 GPL） | RecGen | 只换几何：看几何模块的影响；内部对照，不发布 | `swap-runs/mvs-recgen`（RecGen 在 MVS 几何上重新生成）+ 同一组装 |',
         '| mvs+sam3d | MVS | SAM 3D | 两个都换：可商用版 | `mvs090/completion2/assembly/*`（SAM 3D 在 MVS 几何上生成）+ 同一组装 |',
         '| mvs-fill+sam3d | MVS + 掩码内补洞（同一 MVS 点图，掩码内的空洞用 MoGe-3（MIT）单目深度补上，按每张照片的全局比例 + 该物体自身 MVS 点的局部偏移场对齐；地面、急停不动）| SAM 3D | 两个都换 + 几何模块第 2 版：补全模块拿到完整深度 | `fill_geometry.py` → `checks/bbab-export-090-mvs-fill`；SAM 3D 在补洞几何上重新生成 `swap-runs/mvs-fill-ab` + 同一组装 |', '',
         'SAM 3D 是单张照片生成的，所以每张照片各生成一个候选，再按全部照片的一致性择优（规则四列相同：先比明显错误数，再比 IoU − 2×深度残差 + 0.5×点图覆盖率）。括号里的"照片 N"= 选中的候选是用哪张照片生成的。RecGen 本身用全部照片，没有候选。', '',
         '## 指标的含义', '',
         '| 指标 | 定义 | 方向 | 来源 |',
         '|---|---|---|---|',
         '| 轮廓 IoU | 组装后把模型投到每张照片，与该照片的掩码（SAM 3 分割）的交并比；括号内按照片 1/2/3 | 越高越好；< 0.5 算明显错误 | `comparisons.json` → views[].generated_refined.visible_iou |',
         '| 深度残差 p50 | 模型可见面深度与该几何点图深度的相对差的中位数（掩码内） | 越低越好；> 6% 算明显错误 | 同上 relative_depth_p50 |',
         '| 边界误差 | 模型轮廓与掩码轮廓的距离，按图像高度归一化 | 越低越好 | 同上 boundary_error_image_height |',
         '| 组装：迭代 / 损失 | 9 自由度优化的评估次数 / 最终损失（轮廓 + 深度项） | 损失越低越好；迭代只说明收敛快慢 | refinement.evaluations / final.loss |',
         '| 网格面数 | 补全模型的三角面数 | 仅说明模型复杂度 | comparisons.json faces |',
         '| 尺寸 L×W×H | 模型沿各自地面坐标系的外接范围（p0.5–p99.5），换算成 cm | 与原版对照看变化 | compare.py（网格统计） |',
         '| 最低点 | 顶点高度的 p0.5 相对各自报告地面 | 落地物体应接近 0；< −2 下沉，落地物体 > 3 悬空 | compare.py |',
         '| 检查 | 下沉 / 悬空 / 深度 > 6% / IoU < 0.5 / 模型占了照片里看得见的空地 > 15% | "通过" = 无明显错误 | compare.py，四列同一规则，不含测量盒门 |',
         '| 掩码内点图覆盖率 | 生成用照片的掩码里，几何模块给出有效深度的像素比例 | 越高越好；低 = 补全模块拿到的深度不完整 | SAM 3D 输入记录 mask_depth_pixels / mask_pixels |',
         '| 现场值 | 公平评判器从该几何直接量的现场尺寸（罩壳下沿 24、横杆 20），只做验证，不是输入 | 误差越小越好 | geometry-backbone-ab results.json |',
         '| 急停 4% 约束 | 红钮 4 cm + 黄体 8 cm 联合定尺度时各部位的最大偏差 | 必须 < 4% | 同上 |', '',
         '## 逐物体', '',
         '| 物体 | 指标 | ' + ' | '.join(names) + ' |', '|---|---|' + '---|' * len(names)]
    for oid in OBJECTS:
        rows = results['objects'].get(oid, {})
        def cell(fn):
            return ' | '.join(fn(rows[n]) if n in rows else '—' for n in names)
        L.append(f"| {ZH[oid]} | 轮廓 IoU 均值（照片 1/2/3） | " + cell(lambda r: f"{r['meanIou']:.2f}（" + '/'.join(f"{r['perPhoto'][p]['iou']:.2f}" for p in sorted(r['perPhoto'])) + (f"，照片 {r['generation']['photo']}" if r.get('generation') and r['variant'] != 'recgen' else '') + '）') + ' |')
        L.append(f"|  | 深度残差 p50 | " + cell(lambda r: f"{100 * r['meanDepthP50']:.1f}%") + ' |')
        L.append(f"|  | 边界误差 | " + cell(lambda r: f"{100 * r['meanBoundary']:.2f}%") + ' |')
        L.append(f"|  | 组装：迭代 / 损失 | " + cell(lambda r: f"{r['refine']['evaluations']} / {r['refine']['loss']:.3f}") + ' |')
        L.append(f"|  | 组装后最低点 cm（0.5 百分位；组装 v2 记录） | " + cell(lambda r: f"{r['lowestCm']:+.1f}" if 'lowestCm' in r else '—') + ' |')
        L.append(f"|  | 网格面数 | " + cell(lambda r: f"{(r['faces'] or 0) / 1000:.0f}k") + ' |')
    geoms = [(g, zh) for g, zh in GEOMS if g in results['runs']]
    L += ['', '## 几何层（与补全无关）', '', '| 指标 | ' + ' | '.join(zh for _, zh in geoms) + ' |', '|---|' + '---|' * len(geoms)]
    R = {g: results['runs'][g] for g, _ in geoms}
    fv = results['fieldValues']
    def grow(label, fn):
        L.append(f"| {label} | " + ' | '.join(fn(g) for g, _ in geoms) + ' |')
    grow('尺度（1 原生单位）', lambda g: f"{100 * R[g]['scale']:.2f} cm")
    grow('地面点 / 残差中位 / p95', lambda g: f"{R[g]['floor']['points']} / {R[g]['floor']['residualMedianCm']} / {R[g]['floor']['residualP95Cm']} cm")
    grow('三台相机离地', lambda g: ' / '.join(map(str, R[g]['floor']['cameraHeightsCm'])) + ' cm')
    grow('掩码内点图覆盖率（生成照片，9 物体均值）', lambda g: f"{100 * R[g]['meanCoverage']:.0f}%")
    for oid in OBJECTS:
        grow(f' · {ZH[oid]}', lambda g: f"{100 * R[g]['coverage'].get(oid, 0):.0f}%")
    fvv = lambda g, k: '—' if g not in fv else fv[g][k]
    grow('右罩壳下沿（现场 24）', lambda g: '—' if g not in fv else f"{fv[g]['housing090R_cm']}（{fv[g]['housing090R_err']:+}）")
    grow('右围栏下横杆（现场 20）', lambda g: '—' if g not in fv else f"{fv[g]['fence090R_cm']}（{fv[g]['fence090R_err']:+}）")
    grow('左罩壳下沿（半边被挡，不计分）', lambda g: f"{fvv(g, 'housing090L_cm')}")
    grow('急停 4% 约束最大偏差', lambda g: f"{fvv(g, 'estopMaxDevPct090')}%")
    grow('4 个现场值（含 030）平均 / 最大误差', lambda g: f"{fvv(g, 'maeCm4values')} / {fvv(g, 'maxAbsErrCm')} cm")
    grow('五张照片相机高度极差', lambda g: f"{fvv(g, 'cameraHeightRangeCm')} cm")
    L += metrics_table(names)
    L += attribution(results['objects'], names)
    if results['missing']:
        L += ['', '缺：' + '；'.join(results['missing'])]
    (HERE / 'table.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
