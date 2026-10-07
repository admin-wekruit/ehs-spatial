"""Static evidence page for one oneshot round: latency, spend, readings, catalog changes, audits.

Measured numbers are read from the run, experiment and revision files named on the command line;
only the --finding sentences and --ui-build-seconds are typed. The page sits beside the packaged
report and links back to it.

python scripts/workcell_oneshot_round_page.py --run ONESHOT_RUN --final FINAL_RUN --revisions REVISIONS_DIR \
  --main-id MAIN --candidate-id CANDIDATE --post-shells EXPERIMENT [--spend LABEL LEDGER ...] \
  [--binding LABEL BEFORE_RERUN_JSON ...] [--previous-main OLD_SCENE_REPORT --previous-candidate OLD_SCENE_REPORT] \
  [--audit TITLE DIR ...] [--image FILE ...] [--finding TEXT ...] --ui-build-seconds S --out PAGE/oneshot-YYYY-MM-DD
"""
import argparse
from html import escape
import json
from pathlib import Path
import shutil


def _read(path):
    return json.loads(Path(path).read_text())


def _cm(report, row, key='heightNative'):
    scale = report['modelMeasurementScale']['nativeToMeters']
    return None if scale is None else row[key] * scale * 100


def _photos(report):
    """Photos of the run's one scene (legacy reports: one camera per photo)."""
    return (report.get('capture') or {}).get('photoCount') or len(report['revision']['document']['cameras'])


def _table(head, rows):
    cells = lambda tag, row: ''.join(f'<{tag}>{escape(str(value))}</{tag}>' for value in row)
    return ('<div class="scroll"><table><thead><tr>' + cells('th', head) + '</tr></thead><tbody>'
            + ''.join('<tr>' + cells('td', row) + '</tr>' for row in rows) + '</tbody></table></div>')


def _readings(report):
    """Endpoint id -> (label, cm): rows of two revisions align by measured point, not by display label."""
    return {row['id']: (row['label'], _cm(report, row)) for row in report['endpointEstimation']['endpoints']}


def _audit(title, source, out):
    """Copy an audit directory (images, JSON, README) and summarise its labels table."""
    target = out / source.name
    target.mkdir()
    figures = []
    for path in sorted(source.rglob('*')):
        if path.is_file() and path.suffix.lower() in ('.jpg', '.jpeg', '.png', '.json', '.md'):
            relative = path.relative_to(source)
            (target / relative).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target / relative)
            if path.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                caption = relative.with_suffix('').as_posix()
                figures.append(f'<figure><img loading="lazy" src="{source.name}/{relative.as_posix()}" alt="{escape(caption)}">'
                               f'<figcaption>{escape(caption)}</figcaption></figure>')
    parts = [f'<section><h2>{escape(title)}</h2>']
    readme = source / 'README.md'
    if readme.is_file():
        parts.append(f'<pre class="readme">{escape(readme.read_text())}</pre>')
    labels = source / 'labels.json'
    if labels.is_file():
        rows = _read(labels)
        rows = rows.get('labels', rows) if isinstance(rows, dict) else rows
        if isinstance(rows, list) and rows and all(isinstance(row, dict) for row in rows):
            keys = list(dict.fromkeys(key for row in rows for key in row))
            parts.append(_table(keys, [[json.dumps(row.get(key), ensure_ascii=False) if isinstance(row.get(key), (dict, list)) else row.get(key, '')
                                        for key in keys] for row in rows]))
    parts.append(f'<div class="pictures">{"".join(figures)}</div>')
    parts.append(f'<p class="files">数据：' + ' · '.join(f'<a href="{source.name}/{p.relative_to(source).as_posix()}">{escape(p.relative_to(source).as_posix())}</a>'
                                                     for p in sorted(source.rglob('*.json'))) + '</p></section>')
    return ''.join(parts)


def build(args):
    out = Path(args.out)
    if out.exists():
        raise ValueError('Output must be a new directory')
    out.mkdir(parents=True)
    run, final = Path(args.run), Path(args.final)
    metrics = _read(run / 'one-shot.json')
    stage = metrics['stageTiming']
    modal = _read(run / 'modal-timing.json')
    call = _read(run / 'modal-call.json')
    semantic_first = _read(run / 'semantic-experiment' / 'spend-ledger.json')
    semantic_call = _read(run / 'semantic-experiment' / 'modal-call.json')
    geometry_peak = _read(run / 'geometry-timing.json')
    encoder_peaks = {path.stem.removeprefix('timing-'): _read(path).get('peakAllocatedGiB')
                     for path in sorted((final / 'semantic-experiment').glob('timing-*.json'))}
    revisions = Path(args.revisions)
    page = revisions / args.main_id / 'page'
    main = _read(page / 'scene-report.json')
    candidate = _read(page / 'revisions' / args.candidate_id / 'scene-report.json')

    local_tail = metrics['oneShotWallSeconds'] - call['wallSeconds'] - semantic_call['wallSeconds']
    latency = [
        ['报告界面构建（vite）', f'{args.ui_build_seconds:.2f}', '本机，GPU 前'],
        ['Modal 调用（冷启动、排队、上传下载）', f"{call['wallSeconds'] - modal['containerWallSeconds']:.1f}", '调用窗口 − 容器内'],
        ['几何（MapAnything）', f"{stage['geometrySeconds']:.1f}", 'GPU 0'],
        ['分割（SAM 3，与几何并行）', f"{stage['segmentationSeconds']:.1f}", 'GPU 1'],
        ['小车提议（OWLv2）+ 掩码', f"{stage['owlSeconds'] + stage['cartMaskSeconds']:.1f}", 'GPU 0'],
        ['输入准备', f"{stage['prepareSeconds']:.1f}", 'CPU'],
        ['模型（RecGen，两卡）', f"{stage['modelSeconds']:.1f}", 'GPU 0+1'],
        ['后处理（物理下沿、护板拟合、目录、finalize）', f"{stage['metricTailSeconds']:.1f}", f"CPU；其中信号灯 RecGen {stage['smallObjectSeconds']:.1f} s 用 GPU" if 'smallObjectSeconds' in stage else 'CPU，GPU 容器内空等'],
        ['　其中：物理下沿提取', f"{_read(run / 'physical-clearances.json')['wallSeconds']:.1f}", '与护板拟合并行'],
        ['　其中：三块护板共享拟合', f"{_read(run / 'structural-result.json')['wallSeconds']:.1f}", '与物理下沿并行'],
        ['语义阶段（另一个临时容器）', f"{semantic_call['wallSeconds']:.1f}", f"容器内 {semantic_first['functionSeconds']:.1f} s"],
        ['本机打包与报告', f'{local_tail:.1f}', 'CPU'],
        ['完整 oneshot', f"{metrics['oneShotWallSeconds']:.1f}", f'{_photos(main)} 张原图 → 可打开的报告'],
    ]
    spend_rows = [['oneshot 主容器', 'completed', f"{modal['containerWallSeconds']:.1f}", f"{modal['wallSecondsIncludingColdStart']:.1f}",
                   f"{_read(run / 'spend-ledger.json')['estimate']['functionWindowEstimateUsd']:.3f}",
                   f"{_read(run / 'spend-ledger.json')['estimate']['callWindowEstimateUsd']:.3f}"],
                  ['oneshot 语义阶段', semantic_first['status'], f"{semantic_first['functionSeconds']:.1f}",
                   f"{semantic_first['callSeconds']:.1f}", f"{semantic_first['estimateUsd']:.3f}", f"{semantic_first['callWindowEstimateUsd']:.3f}"]]
    for label, path in args.spend or []:
        ledger = _read(path)
        spend_rows.append([label, ledger['status'], f"{ledger['functionSeconds']:.1f}", f"{ledger['callSeconds']:.1f}",
                           f"{ledger['estimateUsd']:.3f}", f"{ledger['callWindowEstimateUsd']:.3f}"])
    total = [sum(float(row[i]) for row in spend_rows) for i in (4, 5)]
    failed = [sum(float(row[i]) for row in spend_rows if row[1] == 'failed') for i in (4, 5)]
    spend_rows.append(['合计', '', '', '', f'{total[0]:.3f}', f'{total[1]:.3f}'])

    readings = [_readings(main), _readings(candidate)]
    previous = [_readings(_read(path)) if path else {} for path in (args.previous_main, args.previous_candidate)]
    fmt = lambda row: '—' if row is None or row[1] is None else f'{row[1]:.2f}'
    reading_rows = []
    for table, old in zip(readings, previous):
        for ident in dict.fromkeys([*table, *old]):
            label = (table.get(ident) or old[ident])[0]
            if old.get(ident) and old[ident][0] != label:
                label += f'（上一版标为“{old[ident][0]}”）'
            reading_rows.append([('候选 · ' if table is readings[1] else '主模型 · ') + label, fmt(table.get(ident)), fmt(old.get(ident))])
    fmt_cm = lambda value: '原生单位' if value is None else f'{value:.2f}'
    difference_rows = [[row['label'], fmt_cm(_cm(main, row, 'valueNative'))] for row in main['endpointEstimation']['differences']]
    difference_rows += [['候选：' + row['label'], fmt_cm(_cm(candidate, row, 'valueNative'))]
                        for row in candidate['endpointEstimation']['differences']]

    coverage = _read(final / 'objects.json')['coverage']
    review = coverage.get('unsupportedObservations', [])
    consolidation = coverage.get('extraConsolidation') or {'aliased': [], 'merged': []}
    bindings = [(label, _read(path)['semanticBinding']) for label, path in args.binding or []]
    extents = [item['visibleExtentFloor'] for item in main['objects']]
    changed = sum(row.get('observationsWithChangedSupport', 0) for row in extents)
    observations = sum(len(item['observations']) for item in main['objects'])
    for image in args.image or []:
        shutil.copyfile(image, out / Path(image).name)
    gates = []
    fits = Path(args.post_shells) / 'post-shells'
    for item in _read(fits / 'post-shells.json')['items']:
        rows = item.get('fitGate', {}).get('exportedModelReprojectionByPhoto', [])
        gates.append([item['id'], item['status'], ' / '.join(f"照片{r['photo']} {r['maxRawPx']:.1f}" for r in rows),
                      f"{rows[0]['thresholdRawPx']:.1f}" if rows else '—'])
    overlays = ''.join(f'<figure><img loading="lazy" src="candidate/{p.name}" alt="{p.stem}"><figcaption>{p.stem}</figcaption></figure>'
                       for p in sorted(fits.glob('post-box-*-photo-*.jpg')))
    (out / 'candidate').mkdir()
    for path in fits.iterdir():
        # housing-fit.json repeats post-shells.json (same content, other formatting).
        if path.suffix in ('.jpg', '.json') and path.name != 'housing-fit.json':
            shutil.copyfile(path, out / 'candidate' / path.name)
    for name in ('run.json', 'spend-ledger.json', 'input-manifest.json'):
        if (Path(args.post_shells) / name).is_file():
            shutil.copyfile(Path(args.post_shells) / name, out / 'candidate' / name)
    audits = ''.join(_audit(title, Path(directory), out) for title, directory in args.audit or [])
    report = '../'
    html = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Workcell oneshot {escape(args.main_id)}</title>
<style>:root{{--bg:#f4f6f2;--panel:#fff;--ink:#17231f;--muted:#55635e;--line:#d4dbd5;--accent:#12695c}}
@media (prefers-color-scheme:dark){{:root{{--bg:#101614;--panel:#17201d;--ink:#e1e8e4;--muted:#97a49f;--line:#2b3632;--accent:#4fb8a6}}}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.6 system-ui,"PingFang SC",sans-serif}}main{{max-width:1180px;margin:auto;padding:24px 16px 48px}}
a{{color:var(--accent)}}section{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:18px 20px;margin:16px 0}}
h1{{font-size:28px;margin:8px 0}}h2{{font-size:20px;margin:0 0 10px}}.scroll{{overflow-x:auto}}table{{border-collapse:collapse;font-variant-numeric:tabular-nums;min-width:100%}}
th,td{{border-bottom:1px solid var(--line);padding:6px 10px;text-align:left;vertical-align:top}}th{{color:var(--muted);font-weight:600}}
.pictures{{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:14px;margin-top:12px}}figure{{margin:0}}img{{max-width:100%;max-height:520px;object-fit:contain;border-radius:6px}}
figcaption,.files,small{{color:var(--muted);font-size:13px}}pre.readme{{white-space:pre-wrap;font:13px/1.55 ui-monospace,monospace;background:var(--bg);padding:12px;border-radius:8px;max-height:420px;overflow:auto}}
.tag{{display:inline-block;border:1px solid var(--line);border-radius:999px;padding:1px 10px;margin-right:6px;font-size:13px}}</style><main>
<p><a href="{report}">← 打开 3D 报告（主模型）</a> · <a href="{report}?version={escape(args.candidate_id)}">候选版本</a></p>
<h1>{_photos(main)} 张照片 oneshot 重跑（{escape(args.main_id)}）</h1>
<p><span class="tag">已有功能：一套 revision 一套事实</span><span class="tag">候选：G 光幕封闭外形，未接受</span><span class="tag">物理验证：无</span></p>
<p>所有离地值都是条件模型估计：{'原生单位，本场景没有按钮标尺' if main['modelMeasurementScale']['nativeToMeters'] is None else f"比例 {main['modelMeasurementScale']['nativeToMeters']:.5f} m/native"}（{escape(main['modelMeasurementScale']['status'])}；{escape(main['modelMeasurementScale']['source'])}），accepted physical scale 仍为 null。</p>
{('<section><h2>本轮结论</h2><ul>' + ''.join(f'<li>{escape(text)}</li>' for text in args.finding) + '</ul></section>') if args.finding else ''}
<section><h2>延迟（秒）</h2>{_table(['阶段', '秒', '说明'], latency)}
<p><small>GPU 显存峰值：MapAnything {geometry_peak['peakAllocatedGiB']:.2f} GiB（保留 {geometry_peak['peakReservedGiB']:.2f}）/ 80 GiB；语义编码器 {", ".join(f"{k} {v:.2f} GiB" for k, v in encoder_peaks.items() if v is not None)}；SAM 3 与 RecGen 进程未记录峰值。</small></p></section>
<section><h2>费用（Modal 列表价估计，非账单）</h2>{_table(['调用', '状态', '容器内 s', '调用窗口 s', '函数估计 $', '调用窗口估计 $'], spend_rows)}
<p><small>失败支出：函数 ${failed[0]:.3f} / 调用窗口 ${failed[1]:.3f}。actualBilledUsd 为 null。</small></p></section>
<section><h2>离地读数（cm，条件估计）</h2>{_table(['测点', '本轮', '上一版'], reading_rows)}
{_table(['差值', 'cm'], difference_rows)}
{_table(['不比较', '原因'], [[('候选：' if table is candidate else '') + row['id'], row['reason']] for table in (main, candidate) for row in table['endpointEstimation'].get('excludedComparisons', [])])}
<p><small>主模型 {escape(main['revision']['id'])} · 文档 {main['revision']['documentSha256'][:12]}…；候选 {escape(candidate['revision']['id'])} · 文档 {candidate['revision']['documentSha256'][:12]}…（父：主模型）。候选来自封闭挤出假设，严格跨图门槛未通过，不替换主模型。</small></p></section>
<section><h2>目录复核：伪观察移除</h2>
{_table(['对象', '照片', '支持像素', '框', '原因'], [[row['objectId'], row['photo'], row['supportedPixels'], row['box'], row['reason']] for row in review])}
{''.join(f'<p>{escape(label)}：重跑语义前绑定为 <code>{escape(binding["status"])}</code>（{escape(binding["reason"])}）。</p>' for label, binding in bindings)}
<p>最终语义绑定：{escape(main['semanticExperiment']['binding']['reuse'])}（实验运行 {escape(main['semanticExperiment']['sourceRevisionId'])}）。</p>
<div class="pictures">{''.join(f'<figure><img src="{Path(image).name}" alt="{Path(image).stem}"><figcaption>{Path(image).stem}</figcaption></figure>' for image in args.image or [])}</div></section>
<section><h2>一个对象一个模型：合并重复检测、并入宿主、替换原始碎片</h2>
<p>原来 39 个“原始单照片点云碎片”对象（标识、信号灯、线缆托架、地面胶带）按规则整理：{escape(json.dumps(consolidation.get('rule', {}), ensure_ascii=False))}</p>
{_table(['合并后的对象', '合并的检测', '规则'], [[row['objectId'], ', '.join(row['members']), row['rule']] for row in consolidation['merged']])}
{_table(['并入', '原对象', '类别', '原因'], [[row['to'], row['objectId'], row['category'], row['reason']] for row in consolidation['aliased']])}
<p>每个剩下的对象只有一个代理模型：地面胶带是贴地的照片纹理四边形；标识是照片纹理平面四边形；信号灯是按清理后支持点的盒子；超出拟合工作区地面的对象只保留照片观察，不放 3D 模型。厚度和不可见部分仍未知；挂在龙门架、天花板上的对象没有建出支撑结构。</p></section>
<section><h2>可见尺寸：按本版本帧与地面重测</h2>
<p>{len(extents)} 个对象、{observations} 个观察都用 objects 阶段的支持规则（SAM 实例掩码、围栏平面过滤、各自的掩码来源）在本版本帧和地面上重测；不用外轮廓多边形。状态：{escape(', '.join(sorted({row['status'] for row in extents})))}。与云端目录相比支持像素不同的观察 {changed} 个：云端用 OpenCV 最近邻重采样，本版本用与环境无关的确定性最近邻解码（逐元素 float64，恰在半像素处由浮点舍入决定取哪一侧）。</p></section>
<section><h2>候选外壳拟合（本轮重跑）</h2>{_table(['对象', '状态', '导出模型逐图最大重投影误差 px', '门槛 px'], gates)}
<div class="pictures">{overlays}</div><p class="files">数据：<a href="candidate/volume-candidates.json">volume-candidates.json</a> · <a href="candidate/post-shells.json">post-shells.json</a> · <a href="candidate/spend-ledger.json">spend-ledger</a></p></section>
{audits}
</main></html>
"""
    (out / 'index.html').write_text(html)
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ('--run', '--final', '--revisions', '--main-id', '--candidate-id', '--post-shells', '--out'):
        parser.add_argument(name, required=True)
    parser.add_argument('--spend', nargs=2, action='append', metavar=('LABEL', 'LEDGER'), help='Another GPU call of this round')
    parser.add_argument('--binding', nargs=2, action='append', metavar=('LABEL', 'JSON'), help='semantic-binding-before-rerun.json of a catalog change')
    parser.add_argument('--previous-main')
    parser.add_argument('--previous-candidate')
    parser.add_argument('--audit', nargs=2, action='append', metavar=('TITLE', 'DIR'))
    parser.add_argument('--image', action='append')
    parser.add_argument('--finding', action='append', help='One conclusion line shown at the top (already established elsewhere)')
    parser.add_argument('--ui-build-seconds', type=float, required=True)
    print(build(parser.parse_args()))
