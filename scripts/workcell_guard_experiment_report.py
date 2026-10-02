"""Package frozen guard experiment outputs with the existing clickable viewer."""
import argparse
import html
import json
from pathlib import Path
import shutil
import sys

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
from scripts.workcell_photo_report import build as build_report
from scripts.workcell_photo_oneshot import _build_page, _export_metric_scene, _freeze_report_ui


def read(path):
    return json.loads(Path(path).read_text())


def structural_models(root, out, catalog):
    """Promote only the recorded source-fit decision; preserve actual assembly ownership."""
    import trimesh
    paths = list(root.rglob('results.json'))
    paths = [p for p in paths if read(p).get('route') == 'shared-angle-silhouette-prior']
    if len(paths) != 1: raise ValueError('Structural input must contain exactly one A4 result')
    path = paths[0]; result = read(path)
    (out/'structural-result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    if not result['promotionAllowed']: return path, result
    if not result['fitGate']['passed']: raise ValueError('Inconsistent A4 promotion and fit gate')
    records = {r['id']:r for r in result['objects']}
    if set(records) != {'v-guard-left','v-guard-right'}: raise ValueError('A4 must contain only left/right')
    ambiguity = '、'.join(f'{v:.2f}°' for v in result['ambiguity']['imageCompatibleProfileAnglesDeg'])
    for item in catalog['objects']:
        if item['id'] not in records: continue
        row = records[item['id']]; filename = row['mesh']
        if filename != item['model']['file']: raise ValueError('Unexpected structural model filename')
        shutil.copy2(path.parent/filename,out/filename)
        scene = trimesh.load(out/filename,force='scene')
        item['model'] = {'file':filename,'nodes':list(scene.graph.nodes_geometry)}
        item['representation'] = '用户同角先验 + 源照片轮廓拟合的双片模型；实物夹角不可唯一确定'
        item['notes'] = [
            f"模型共用 {result['sharedAngleDeg']:.2f}° 参数；相等由用户同规格先验施加，不是独立测量结果。",
            f'离散角度敏感性检查中，{ambiguity} 也通过照片贴合检查；这些点不是连续区间或统计置信区间。',
            '原生成护板总成的分片只初始化姿态和轮廓；最终两片平面按源照片分割轮廓重新拟合，并保留照片颜色。',
            '源照片相机和世界坐标保持原解；未知板厚以双面可见的零厚度平面表示。',
            '模型仅通过相对 A1 的照片贴合门槛；个别视图可能下降，详见实验页逐图对照。厘米和角度物理精度未验证。']
        item['structuralModel'] = {'source':'structural-result.json','modelAngleDeg':row['meshAngleReadbackDeg'],
                                   'measurementAngleDeg':None,'prior':result['prior'],'ambiguity':result['ambiguity']}
    shutil.copy2(out/'guard-partition.json',out/'guard-generated-initializer-partition.json')
    assembly = trimesh.Scene(); source_nodes = {}
    for side in ('left','center','right'):
        scene = trimesh.load(out/f'guard-{side}.glb',force='scene')
        source_nodes[side] = list(scene.graph.nodes_geometry)
        for node in source_nodes[side]:
            matrix, geometry = scene.graph.get(node)
            name = f'{side}:{node}'
            assembly.add_geometry(scene.geometry[geometry],node_name=name,geom_name=name,transform=matrix)
    assembly.export(out/'guard-multi.glb')
    # ponytail: record the exported scene's actual flattening order, not assumed insertion order.
    loaded = trimesh.load(out/'guard-multi.glb',force='scene'); offset = 0; ranges = {}
    for node in loaded.graph.nodes_geometry:
        _, geometry = loaded.graph.get(node); count = len(loaded.geometry[geometry].faces)
        ranges[node] = list(range(offset,offset+count)); offset += count
    partition = {'sourceFile':'guard-multi.glb','method':'assembly of final A4 left/right sheets and unchanged A1 center; actual exported face ownership',
                 'initializerProvenance':{'partition':'guard-generated-initializer-partition.json',
                    'models':'experiment-data/controls/A1-similarity/',
                    'scope':'Original generated assembly partition initializes A4; its face indices do not describe the final analytic sheets.'},
                 'parts':[{'side':side,'file':f'guard-{side}.glb',
                           'sourceFaceIndices':[i for node in source_nodes[side] for i in ranges[f'{side}:{node}']],
                           'representation':'unchanged A1 generated center' if side=='center' else 'A4 prior-constrained analytic two-sheet model'}
                          for side in ('left','center','right')]}
    (out/'guard-partition.json').write_text(json.dumps(partition,ensure_ascii=False)+'\n')
    shutil.copy2(out/'guard-placement.json',out/'guard-placement-initializer.json')
    placement = {'method':'A4 shared-angle structural fit to source silhouettes',
                 'result':'structural-result.json','initialization':'guard-placement-initializer.json',
                 'coordinateSystem':result['coordinateSystem'],'optimizedVariables':result['optimizedVariables'],
                 'prior':result['prior'],'sharedModelAngleDeg':result['sharedAngleDeg'],
                 'measurementAngleDeg':None,'centerPolicy':result['centerPolicy']}
    (out/'guard-placement.json').write_text(json.dumps(placement,ensure_ascii=False,indent=2)+'\n')
    return path, result


def structural_section(path, root, result):
    base = 'experiment-data/structural/'+str(path.parent.relative_to(root))
    compatible = '、'.join(f'{v:.2f}°' for v in result['ambiguity']['imageCompatibleProfileAnglesDeg']) or '无'
    selected = result['promotionAllowed']
    status = '已作为先验约束模型显示在主场景' if selected else '候选未替换主场景，主场景保留 A1'
    rows=[]; models=[]
    for obj in result['objects']:
        name = {'v-guard-left':'左','v-guard-right':'右'}[obj['id']]
        fit, old, initial = obj['sourceFit'], obj['a1SourceFit'], obj['analyticInitialSourceFit']
        models.append(f'<article><h3>{name}板：模型 {obj["meshAngleReadbackDeg"]:.2f}°</h3><p>平均 IoU：A1 {old["meanIoU"]:.3f} → 双片初值 {initial["meanIoU"]:.3f} → 拟合 {fit["meanIoU"]:.3f}。中位 IoU：{old["medianIoU"]:.3f} → {fit["medianIoU"]:.3f}。</p><a href="{base}/{html.escape(obj["mesh"],quote=True)}">下载实际双片模型 GLB（非物理测量）</a></article>')
        before = {v['photo']:v for v in old['views']}; start = {v['photo']:v for v in initial['views']}
        for view in fit['views']:
            photo = view['photo']; delta = view['iou']-before[photo]['iou']
            rows.append(f'<tr><td>{name}</td><td>{photo}</td><td>{before[photo]["iou"]:.3f}</td><td>{start[photo]["iou"]:.3f}</td><td>{view["iou"]:.3f}</td><td>{delta:+.3f}</td></tr>')
    profile=[]
    for row in result['profile']:
        fits = row.get('sourceFit',{})
        means = ' / '.join(f'{fits[s]["meanIoU"]:.3f}' for s in ('left','right')) if fits else '—'
        profile.append(f'<tr><td>{row["angleDeg"]:.2f}°</td><td>{html.escape(row["status"])}</td><td>{means}</td><td>{"通过" if row.get("imageCompatible") else "未通过"}</td></tr>')
    overlays=''.join(f'<figure><img src="{base}/{html.escape(name,quote=True)}" alt="照片 {i} 的源轮廓、A1 和双片候选叠图"><figcaption>照片 {i}</figcaption></figure>' for i,name in enumerate(result['overlays'],1))
    reasons='；'.join(result['fitGate']['reasons']) or '相对 A1 的逐板均值、中位数、逐图 IoU 门槛及收敛/模型注释检查均通过。'
    ambiguity = '实物角度仍不可唯一确定' if result['ambiguity']['ambiguous'] else '有限离散检查未发现多个通过角度；仍不构成物理精度验证'
    return f'''<section class="structural"><h2>A4 · 同角先验下的源照片轮廓拟合</h2><p class="lead"><strong>{status}；{ambiguity}。</strong></p>
<p>左右模型从求解开始共用一个夹角参数，当前选中 {result['sharedAngleDeg']:.2f}°。相等来自用户提供的同规格先验。生成总成的分片仅初始化姿态与轮廓；目标函数使用源照片分割轮廓，分别重新拟合两板姿态和可见轮廓尺寸；中间板保持 A1。</p>
<p><strong>仍通过照片贴合检查的离散角度：{compatible}。</strong>这是固定相机、有限模板边界下的敏感性检查，不是连续范围或统计置信区间。检查完成：{'是' if result['ambiguity']['profileComplete'] else '否'}。</p>
<div class="cards">{''.join(models)}</div><p>选择依据：{html.escape(reasons)}</p>
<h3>逐板逐图 IoU 对照</h3><p>IoU 为轮廓重合度，不是厘米或角度精度。下表保留下降的视图；缺少有效板实例的照片不计入该板。</p>
<table><thead><tr><th>板</th><th>照片</th><th>A1</th><th>双片初值</th><th>A4</th><th>相对 A1</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<details><summary>离散角度敏感性检查（非置信区间）</summary><table><thead><tr><th>模型角度</th><th>优化状态</th><th>左 / 右平均 IoU</th><th>照片贴合门槛</th></tr></thead><tbody>{''.join(profile)}</tbody></table></details>
<p>绿色：源照片轮廓；紫红：A1；青色：先验约束候选。</p><div class="cards">{overlays}</div>
<p>A4 缓存输入上的优化增量 {result['wallSeconds']:.2f} 秒；不是一次从照片开始的完整流程耗时。<a href="{base}/results.json">完整结果、参数边界与限制</a></p></section>'''


def build(baseline, controls, joint, out, viewer_assets, previous=(), extra=(), structural=None, measurements=None):
    from scripts.workcell_photo_calibration import apply_measurements, load_measurements
    measured = load_measurements(measurements) if measurements else None
    if out.exists(): raise ValueError('Report output must be new')
    out.mkdir(parents=True)
    for p in baseline.iterdir():
        if p.is_file() and (p.suffix in ('.glb','.png','.jpg') or p.name in ('geometry.json','objects.json','guard-partition.json','guard-input.npz','cart-input.npz','guard-mask-selection.json','measurements.json') or p.name.startswith('frame_')):
            if not p.name.startswith('entity-'): shutil.copy2(p,out/p.name)
    align = controls/'A1-similarity'
    for p in align.glob('*.glb'): shutil.copy2(p,out/p.name)
    shutil.copy2(align/'results.json',out/'guard-placement.json')
    catalog = read(out/'objects.json')
    for item in catalog['objects']:
        if item['id'].startswith('v-guard-'):
            item['representation'] = '生成模型 + 保形对齐（旋转、平移、统一缩放）；实物夹角尚未验证'
            item['notes'] += ['本轮取消逐轴缩放；模型夹角保持生成时形状。',
                              '左右模型角差仍约 3.41°。这是测量链路尚未解决的问题，不能据此判定两件实物不同。']
    structural_path, structural_result = structural_models(structural,out,catalog) if structural else (None,None)
    promoted = bool(structural_result and structural_result['promotionAllowed'])
    if promoted:
        center = next(item for item in catalog['objects'] if item['id']=='v-guard-center')
        center['notes'] = [note for note in center['notes'] if '左右模型角差' not in note]
        center['notes'].append('A4 仅约束左右板；本中间板保持 A1 生成分片和保形对齐，不参与共享夹角。')
    (out/'objects.json').write_text(json.dumps(catalog,ensure_ascii=False))
    if measured:
        apply_measurements(out, measured)
    data = build_report(out)
    _export_metric_scene(out,data)
    _freeze_report_ui(out, viewer_assets)
    metrics = read(baseline/'one-shot.json')
    metrics['geometry'] = data['geometry']
    page = _build_page(out,metrics)
    data['experiment'] = {'title':'本轮：保形对齐已应用，实物角度尚未测稳',
        'summary':'这里保留完整可点击场景，三块护板采用同一批生成模型的保形对齐结果。下面的角度来自模型；相同规格实物的夹角一致性仍未通过验证。全流程此前实测 334.6 秒，本轮重用已保存结果做对照。',
        'reportURL':'experiments.html','timingLabel':'对齐增量'}
    a1 = read(align/'results.json')
    data['timing'] = {'oneShotSeconds':a1['seconds']}
    if structural_result:
        angles = '、'.join(f'{v:.2f}°' for v in structural_result['ambiguity']['imageCompatibleProfileAnglesDeg'])
        data['experiment'].update(title='同角先验模型已应用；实物角度仍不可唯一确定' if promoted else '同角先验候选未通过；主场景保留 A1',
            summary=(f"左右双片模型共用 {structural_result['sharedAngleDeg']:.2f}°，相等由用户同规格先验施加，并按源照片轮廓拟合。" if promoted else '左右双片候选未通过照片贴合检查，主场景继续显示 A1 生成模型。')+
                    f'离散检查中的 {angles or "无角度"} 通过照片贴合门槛；这不是连续区间或置信区间，也不能确定真实角度。中间板保持 A1。耗时为缓存输入增量。',
            timingLabel='A4 缓存优化增量',structuralModel={'resultURL':'structural-result.json',
                'sharedAngleDeg':structural_result['sharedAngleDeg'],'promotionAllowed':promoted,
                'ambiguity':structural_result['ambiguity'],'measurementAngleDeg':None})
        data['timing'] = {'oneShotSeconds':structural_result['wallSeconds']}
        shutil.copy2(out/'structural-result.json',page/'structural-result.json')
        if promoted:
            for name in ('guard-generated-initializer-partition.json','guard-placement.json','guard-placement-initializer.json'):
                shutil.copy2(out/name,page/name)
    for target in (out/'scene-report.json',page/'scene-report.json'):
        target.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    evidence = page/'experiment-data'; evidence.mkdir()
    runs = list(dict.fromkeys(p.resolve() for p in [controls,*previous,joint,*extra,*([structural] if structural else [])]))
    route_roots = [(joint,'joint'),*[(root,f'extra-{i}') for i,root in enumerate(extra,1)]]
    for root, label in [(controls,'controls'),*route_roots,*([(structural,'structural')] if structural else [])]:
        for p in root.rglob('*'):
            if not p.is_file() or p.suffix not in ('.json','.glb','.jpg','.png'): continue
            if p.name in ('input-manifest.json','spend-ledger.json','run.json'): continue
            relative = Path(label)/p.relative_to(root)
            destination = evidence/relative; destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(p,destination)
    col = read(controls/'COLMAP/results.json')
    line_results = [read(p) for p in joint.glob('LIMAP*/results.json')]
    rows=[]
    for root,label in route_roots:
      for p in sorted(root.rglob('results.json')):
        d=read(p)
        if 'objects' not in d: continue
        names={'v-guard-left':'左','v-guard-center':'中','v-guard-right':'右'}
        parts=[]
        for obj in d['objects']:
            value=obj.get('measurementAngleDeg')
            text=f'{value:.2f}°（条件估计）' if value is not None else '测量未通过'
            mesh=obj.get('mesh')
            download=f' · <a href="experiment-data/{label}/{p.parent.relative_to(root)}/{html.escape(mesh,quote=True)}">未验收实验模型 GLB</a>' if mesh else ''
            parts.append(f'<li>{names[obj["id"]]}板：{text}{download}</li>')
        link=f'experiment-data/{label}/'+str(p.relative_to(root))
        rows.append(f'<article><h3>{html.escape(str(p.parent.relative_to(root)))}</h3><ul>{"".join(parts)}</ul><a href="{link}">查看支持点、留出误差与失败原因</a></article>')
    ledgers=[{'run':r.name,**read(r/'spend-ledger.json')} for r in runs]
    estimates=sum(r.get('estimateUsd',0) for r in ledgers)
    calls=sum(r.get('callWindowEstimateUsd',0) for r in ledgers)
    run_times=[r['functionSeconds'] for r in ledgers if r.get('functionSeconds') is not None]
    summary={'baselineFullPipelineSeconds':334.60,'alignmentSeconds':a1['seconds'],
             'colmapSeconds':col['seconds'],'functionSeconds':run_times,'functionCostEstimateUsd':estimates,
             'callWindowCostEstimateUsd':calls,'actualBilledUsd':None,'runs':ledgers,
             'structuralOptimizationSeconds':structural_result['wallSeconds'] if structural_result else None,
             'basis':'frozen-data ablation; reuses baseline inference; not a new full oneshot latency benchmark'}
    (evidence/'summary.json').write_text(json.dumps(summary,indent=2))
    after={b['side']:b['angleDeg'] for b in a1['boards']}
    line_text='；'.join(f'{r.get("scope","scene")}：{r["lines3D"]} 条稳定 3D 线' for r in line_results)
    dense_sections=[]
    for root,label in route_roots:
        for p in root.rglob('dense-results.json'):
            d=read(p)
            support=' / '.join(f'{r["side"]} {r["retainedDistinctTracks"]}' for r in d['trackMerge'])
            pic=f'experiment-data/{label}/{p.parent.relative_to(root)}/matches-3-4.jpg'
            dense_sections.append(f'<h3>LoFTR 密集匹配</h3><p>两块 GPU 并行匹配六对照片；去重后护板轨迹数：{support}。有效护板轨迹仅连接照片 3/4，尚未形成独立第三视角的验证。匹配增加不等于角度已经可用。</p><img src="{pic}" alt="照片3与4的真实密集匹配，用颜色区分三块护板">')
    structural_html = structural_section(structural_path,structural,structural_result) if structural_result else ''
    headline = '模型已一致，真实夹角仍无法唯一确定' if promoted else '两块同规格护板，为什么还没有测成一样？'
    current_scene = '主场景的左右板已采用上方 A4 先验约束模型，中间板保持 A1。' if promoted else '完整场景继续显示经过保形对齐的 A1 模型。'
    measurement_boundary = ('用户提供的按钮标准尺寸与独立离地真值详见主报告；仅按钮整体高度定尺度。围栏与光幕真值仅用于独立误差评估。照片几何仍有误差。' if data['geometry'].get('calibration') else '按钮整体高/宽 20 cm 仍是可调整输入假设。')
    page_text=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>护板角度实验 · Panoptes</title><style>body{{font:17px/1.65 system-ui,sans-serif;background:#f5f6f2;color:#182824;margin:0}}main{{max-width:1050px;margin:auto;padding:40px 22px}}h1{{font-size:36px;line-height:1.25}}h2{{margin-top:38px}}a{{color:#14675b}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(270px,1fr));gap:16px}}article{{background:white;border:1px solid #d6ddd5;border-radius:12px;padding:18px}}img{{width:100%;border-radius:10px}}.lead{{font-size:21px}}.note{{color:#53645f}}.action{{display:inline-block;background:#195e52;color:white;padding:12px 18px;border-radius:7px;text-decoration:none}}li{{margin:5px 0}}table{{border-collapse:collapse;width:100%;font-size:15px}}td,th{{padding:8px;text-align:left;border-bottom:1px solid #c9d4cc}}figure{{margin:0}}.structural{{background:#eef5eb;border:2px solid #557351;border-radius:12px;padding:20px}}</style><main>
<p>PANOPTES · 四张照片 · 同一批输入对照</p><h1>{headline}</h1>
<p class="lead">{html.escape(data["experiment"]["summary"])}</p>
<p><a class="action" href="index.html#scene">打开完整可点击 3D 模型</a></p>
{structural_html}
<h2>历史对照 1 · A1 保形对齐</h2><div class="cards"><article><h3>保形对齐</h3><p>左 {after['left']:.2f}° / 右 {after['right']:.2f}°，差 {abs(after['left']-after['right']):.2f}°。中间板仍不报可用夹角。</p><p>原版差 3.47°；这项改动避免继续改变模型夹角，尚未消除生成几何中的差异。</p></article>
<article><h3>照片贴合与速度</h3><p>照片 3/4 重合度 {a1['sourceChecks'][2]['iou']:.3f} / {a1['sourceChecks'][3]['iou']:.3f}；原版为 0.821 / 0.754。</p><p>对齐 {a1['seconds']:.2f} 秒。贴合基本保持；重合度不等于物理测量精度。</p></article></div>
<img src="experiment-data/angle-comparison.png" alt="原始生成模型、逐轴缩放与保形对齐的模型夹角对比">
<h2>历史对照 2 · 相机优化改善了什么</h2><p>COLMAP 在四图中得到 {col['afterBa']['points']} 个真实匹配点。拟合重投影误差从 {col['beforeBa']['meanErrorPx']:.2f} px 降到 {col['afterBa']['meanErrorPx']:.2f} px（1554 px 高的处理图）。这是参与拟合的误差，不能用作独立精度证明。</p>
<p>LIMAP 点线联合优化也已执行。{line_text}。少量场景线不能证明护板两面的夹角已恢复。</p>
<img src="experiment-data/feature-support.png" alt="照片3和4上实际用于护板几何拟合的RGB匹配点">
{''.join(dense_sections)}
<h2>历史对照 3 · 各条几何路线的实际结果</h2><p>A2 左右独立求解；A3 在相同观测上尝试共享夹角。没有足够支持的路线不会把强制相等当成测准。历史实验候选模型可下载；{current_scene}</p><details><summary>展开 {len(rows)} 条路线的结果、模型与失败原因</summary><div class="cards">{''.join(rows)}</div></details>
<h2>4. 本轮耗时与花费</h2><p>原完整照片流程实测 <strong>334.6 秒</strong>。本轮重用这次运行的相机初值、分割及生成模型，比较几何增量；未把缓存对照耗时冒充一次全流程速度。</p>
<p>COLMAP 增量 {col['seconds']:.1f} 秒；各临时云容器实际工作时段 {', '.join(f'{x:.1f}' for x in run_times)} 秒。按固定 2×A100 配置估算 ${estimates:.3f}；含调度调用窗口估算 ${calls:.3f}，均不是账单金额。</p>
<h2>测量边界</h2><p>{measurement_boundary}统一定尺度不能修正夹角。角度的现场真值尚未提供；本轮不宣称厘米或角度准确度达标。板厚及不可见完整尺寸保留未知。</p>
<p><a href="experiment-data/summary.json">下载本轮摘要</a> · <a href="https://github.com/colmap/colmap">COLMAP</a> · <a href="https://github.com/cvg/limap">LIMAP</a> · <a href="https://github.com/zju3dv/LoFTR">LoFTR</a></p></main></html>'''
    (page/'experiments.html').write_text(page_text)
    plots(baseline,a1,joint,evidence)
    return page


def plots(baseline, a1, joint, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image
    fig,ax=plt.subplots(figsize=(10,4),layout='constrained')
    for side,color,a0 in (('left','#167b70',144.9939),('right','#bc612c',141.5282)):
        row=next(b for b in a1['boards'] if b['side']==side)
        vals=[row['beforeDeg'],a0,row['angleDeg']]
        ax.plot(range(3),vals,'o-',label=side.title(),color=color,lw=2)
        for x,y in enumerate(vals):ax.annotate(f'{y:.2f}',(x,y),xytext=(0,10),textcoords='offset points',ha='center')
    ax.set_xticks(range(3),['Generated mesh','Old per-axis alignment','A1 uniform alignment'])
    ax.set_ylim(139,147);ax.set_ylabel('Model interior angle (degrees)');ax.grid(axis='y',alpha=.2)
    ax.set_title('Preserving shape prevents added distortion; the generated difference remains',loc='left')
    ax.legend();fig.savefig(out/'angle-comparison.png',dpi=140);plt.close(fig)
    obs=read(joint/'COLMAP-cameras/observations.json')['boards']
    fig,axes=plt.subplots(1,2,figsize=(10,6),layout='constrained')
    for ax,photo in zip(axes,(3,4)):
        ax.imshow(Image.open(baseline/f'photo-{photo}.png'))
        for side,color in (('left','#ff9d00'),('center','#65e4ed'),('right','#ee77b1')):
            points=np.array([o['uv'] for t in obs[side] for o in t['observations'] if o['photo']==photo])
            if len(points):ax.scatter(points[:,0],points[:,1],s=35,facecolors='none',edgecolors=color,label=f'{side}: {len(points)}')
        ax.set_xlim(35,320);ax.set_ylim(375,235);ax.set_title(f'Photo {photo}: actual COLMAP board observations')
        ax.legend(loc='lower left',fontsize=8);ax.axis('off')
    fig.suptitle('A low scene-wide pixel error does not supply missing evidence on the two panel faces')
    fig.savefig(out/'feature-support.png',dpi=150);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('baseline','controls','joint','out','viewer-assets'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--previous',type=Path,action='append',default=[])
    p.add_argument('--extra',type=Path,action='append',default=[])
    p.add_argument('--structural',type=Path,help='Frozen A4 run root; promotes only when its recorded source-fit gate passes')
    p.add_argument('--measurements',type=Path,help='Measured reference and independent evaluation JSON')
    a=p.parse_args();print(build(a.baseline,a.controls,a.joint,a.out,a.viewer_assets,a.previous,a.extra,a.structural,a.measurements))
