"""Package frozen guard experiment outputs with the existing clickable viewer."""
import argparse
import html
import json
from pathlib import Path
import shutil
import sys

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
from scripts.workcell_photo_report import build as build_report
from scripts.workcell_photo_oneshot import _build_page, _export_metric_scene


def read(path):
    return json.loads(Path(path).read_text())


def build(baseline, controls, joint, out, previous=(), extra=()):
    if out.exists(): raise ValueError('Report output must be new')
    out.mkdir(parents=True)
    for p in baseline.iterdir():
        if p.is_file() and (p.suffix in ('.glb','.png','.jpg') or p.name in ('geometry.json','objects.json','guard-partition.json','guard-input.npz','cart-input.npz','guard-mask-selection.json') or p.name.startswith('frame_')):
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
    (out/'objects.json').write_text(json.dumps(catalog,ensure_ascii=False))
    data = build_report(out)
    _export_metric_scene(out,data['geometry'])
    source = Path(__file__).resolve().parents[1]
    shutil.copytree(source/'web/dist-photo',out/'report-ui')
    page = _build_page(out,read(baseline/'one-shot.json'))
    data['experiment'] = {'title':'本轮：保形对齐已应用，实物角度尚未测稳',
        'summary':'这里保留完整可点击场景，三块护板采用同一批生成模型的保形对齐结果。下面的角度来自模型；相同规格实物的夹角一致性仍未通过验证。全流程此前实测 334.6 秒，本轮重用已保存结果做对照。',
        'reportURL':'experiments.html','timingLabel':'对齐增量'}
    a1 = read(align/'results.json')
    data['timing'] = {'oneShotSeconds':a1['seconds']}
    for target in (out/'scene-report.json',page/'scene-report.json'):
        target.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    evidence = page/'experiment-data'; evidence.mkdir()
    runs = [controls,*previous,joint,*extra]
    route_roots = [(joint,'joint'),*[(root,f'extra-{i}') for i,root in enumerate(extra,1)]]
    for root, label in [(controls,'controls'),*route_roots]:
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
    estimates=sum(read(r/'spend-ledger.json').get('estimateUsd',0) for r in runs)
    calls=sum(read(r/'spend-ledger.json').get('callWindowEstimateUsd',0) for r in runs)
    run_times=[read(r/'spend-ledger.json').get('functionSeconds') for r in runs]
    summary={'baselineFullPipelineSeconds':334.60,'alignmentSeconds':a1['seconds'],
             'colmapSeconds':col['seconds'],'functionSeconds':run_times,'functionCostEstimateUsd':estimates,
             'callWindowCostEstimateUsd':calls,'actualBilledUsd':None,
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
    page_text=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>护板角度实验 · Panoptes</title><style>body{{font:17px/1.65 system-ui,sans-serif;background:#f5f6f2;color:#182824;margin:0}}main{{max-width:1050px;margin:auto;padding:40px 22px}}h1{{font-size:36px;line-height:1.25}}h2{{margin-top:38px}}a{{color:#14675b}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(270px,1fr));gap:16px}}article{{background:white;border:1px solid #d6ddd5;border-radius:12px;padding:18px}}img{{width:100%;border-radius:10px}}.lead{{font-size:21px}}.note{{color:#53645f}}.action{{display:inline-block;background:#195e52;color:white;padding:12px 18px;border-radius:7px;text-decoration:none}}li{{margin:5px 0}}</style><main>
<p>PANOPTES · 四张照片 · 同一批输入对照</p><h1>两块同规格护板，为什么还没有测成一样？</h1>
<p class="lead">本轮已移除会改变夹角的逐轴缩放；多视图几何实验仍需通过两片板面的独立图像证据检查。</p>
<p><a class="action" href="index.html#scene">打开完整可点击 3D 模型</a></p>
<h2>1. 已验证的改动</h2><div class="cards"><article><h3>保形对齐</h3><p>左 {after['left']:.2f}° / 右 {after['right']:.2f}°，差 {abs(after['left']-after['right']):.2f}°。中间板仍不报可用夹角。</p><p>原版差 3.47°；这项改动避免继续改变模型夹角，尚未消除生成几何中的差异。</p></article>
<article><h3>照片贴合与速度</h3><p>照片 3/4 重合度 {a1['sourceChecks'][2]['iou']:.3f} / {a1['sourceChecks'][3]['iou']:.3f}；原版为 0.821 / 0.754。</p><p>对齐 {a1['seconds']:.2f} 秒。贴合基本保持；重合度不等于物理测量精度。</p></article></div>
<img src="experiment-data/angle-comparison.png" alt="原始生成模型、逐轴缩放与保形对齐的模型夹角对比">
<h2>2. 相机优化真正改善了什么</h2><p>COLMAP 在四图中得到 {col['afterBa']['points']} 个真实匹配点。拟合重投影误差从 {col['beforeBa']['meanErrorPx']:.2f} px 降到 {col['afterBa']['meanErrorPx']:.2f} px（1554 px 高的处理图）。这是参与拟合的误差，不能用作独立精度证明。</p>
<p>LIMAP 点线联合优化也已执行。{line_text}。少量场景线不能证明护板两面的夹角已恢复。</p>
<img src="experiment-data/feature-support.png" alt="照片3和4上实际用于护板几何拟合的RGB匹配点">
{''.join(dense_sections)}
<h2>3. 各条几何路线的实际结果</h2><p>A2 左右独立求解；A3 在相同观测上尝试共享夹角。没有足够支持的路线不会把强制相等当成测准。实验候选模型可下载；完整场景继续显示经过保形对齐的模型。</p><details><summary>展开 {len(rows)} 条路线的结果、模型与失败原因</summary><div class="cards">{''.join(rows)}</div></details>
<h2>4. 本轮耗时与花费</h2><p>原完整照片流程实测 <strong>334.6 秒</strong>。本轮重用这次运行的相机初值、分割及生成模型，比较几何增量；未把缓存对照耗时冒充一次全流程速度。</p>
<p>COLMAP 增量 {col['seconds']:.1f} 秒；各临时云容器实际工作时段 {', '.join(f'{x:.1f}' for x in run_times)} 秒。按固定 2×A100 配置估算 ${estimates:.3f}；含调度调用窗口估算 ${calls:.3f}，均不是账单金额。</p>
<h2>测量边界</h2><p>按钮整体高/宽 20 cm 仍是可调整输入假设。统一定尺度不能修正夹角。角度的现场真值尚未提供；本轮不宣称厘米或角度准确度达标。板厚及不可见完整尺寸保留未知。</p>
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
    for name in ('baseline','controls','joint','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--previous',type=Path,action='append',default=[])
    p.add_argument('--extra',type=Path,action='append',default=[])
    a=p.parse_args();print(build(a.baseline,a.controls,a.joint,a.out,a.previous,a.extra))
