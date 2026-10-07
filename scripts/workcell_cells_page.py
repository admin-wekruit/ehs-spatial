"""Evidence page for a capture that holds several scenes: one section per scene run, read from its files.

Measured numbers come from each run (capture.json, one-shot.json, stage-timing.json, spend-ledger.json,
objects.json, scene-report.json); only --finding sentences and image captions are typed.

python scripts/workcell_cells_page.py --cell "BL4-BOR1-090" RUN ../ --cell "BL4-BOR1-030" RUN2 ../cell-030/ \
  --image FILE CAPTION ... --finding TEXT ... --spend LEDGER_ENTRY.json --ui-build-seconds S --out PAGE/cells-YYYY-MM-DD
"""
import argparse
from html import escape
import json
from pathlib import Path
import shutil


def _read(path):
    return json.loads(Path(path).read_text())


def _table(head, rows):
    cells = lambda tag, row: ''.join(f'<{tag}>{escape(str(value))}</{tag}>' for value in row)
    return ('<div class="scroll"><table><thead><tr>' + cells('th', head) + '</tr></thead><tbody>'
            + ''.join('<tr>' + cells('td', row) + '</tr>' for row in rows) + '</tbody></table></div>')


def _iou(record):
    """'photo: IoU' list from a model record (multiViewModel.maskIoUByPhoto or recgenModel.sourceChecks)."""
    record = record or {}
    if record.get('maskIoUByPhoto'):
        return ' / '.join(f"照片{p} {value:.2f}" for p, value in sorted(record['maskIoUByPhoto'].items()))
    checks = record.get('sourceChecks') or {}
    return ' / '.join(f"照片{p} {c['iou']:.2f}" for p, c in sorted(checks.items()) if isinstance(c, dict) and c.get('iou') is not None) or '—'


STAGES = {'geometrySeconds': '容器内：几何（MapAnything）', 'owlSeconds': '容器内：OWLv2 检测', 'segmentationSeconds': '容器内：SAM 3 分割',
          'cartMaskSeconds': '容器内：小车掩码', 'prepareSeconds': '容器内：RecGen 输入准备', 'modelSeconds': '容器内：RecGen 多视角模型（机器人、小车、护板）',
          'metricTailSeconds': '容器内：模型之后的全部阶段（CPU 后处理与信号灯）'}
LOCAL = {'semanticStageSeconds': '语义阶段（第二个 Modal 调用，含排队）', 'evidenceSheetsSeconds': '本机：证据图', 'finalizeSeconds': '本机：报告定稿',
         'exportSeconds': '本机：导出', 'pageSeconds': '本机：打包页面'}


def _latency(run, metrics, ui_seconds):
    modal = _read(run / 'modal-timing.json')
    first = metrics['runs'][0]['wallSeconds']
    rows = [['完整 oneshot（本机计时，端到端）', metrics['oneShotWallSeconds']], ['报告界面构建（vite，一次，两个工位共用）', ui_seconds],
            ['主 Modal 调用（本机 modal run 计时）', first],
            ['其中 modal run 启动与收尾（本机）', first - modal['wallSecondsIncludingColdStart']],
            ['其中排队、冷启动与照片/结果传输', modal['wallSecondsIncludingColdStart'] - modal['containerWallSeconds']],
            ['其中容器运行', modal['containerWallSeconds']]]
    stage = metrics['stageTiming']
    rows += [[name, stage[key]] for key, name in STAGES.items() if key in stage]
    if 'smallObjectSeconds' in stage:  # inside metricTailSeconds (the lamps run after the CPU tail's catalog build)
        rows += [['　其中信号灯 RecGen 与放置', stage['smallObjectSeconds']],
                 ['　其中 CPU 后处理（地面、对象、龙门架、尺寸）', stage['metricTailSeconds'] - stage['smallObjectSeconds']]]
    rows += [[name, metrics['localStageTiming'][key]] for key, name in LOCAL.items() if key in metrics['localStageTiming']]
    return [[name, f'{value:.1f}'] for name, value in rows]


def _cell(label, run, url, ui_seconds):
    run = Path(run)
    capture, metrics, report = _read(run / 'capture.json'), _read(run / 'one-shot.json'), _read(run / 'scene-report.json')
    catalog = _read(run / 'objects.json')
    ledger = _read(run / 'spend-ledger.json')
    semantic = _read(run / 'semantic-experiment/spend-ledger.json') if (run / 'semantic-experiment/spend-ledger.json').is_file() else {}
    scale = report['modelMeasurementScale']
    factor = scale['nativeToMeters']
    photos = ', '.join(f"照片 {n + 1} = {row['name']}" for n, row in enumerate(capture['sources']))
    latency = _latency(run, metrics, ui_seconds)
    calls = [[run_['stage'], run_.get('returncode'), f"{run_['wallSeconds']:.1f}"] for run_ in ledger.get('runs', [])]
    estimate = ledger.get('estimate', {})
    models = []
    for item in catalog['objects']:
        multi = item.get('multiViewModel')
        if multi:
            models.append([item['id'], 'RecGen 多视角', ', '.join(map(str, multi.get('generationViews', []))), _iou(multi)])
        lamp = item.get('recgenModel')
        if lamp:
            upright = lamp.get('upright') or {}
            stood = (f"，立正：自由拟合倾斜 {upright['freeTiltDeg']}° → {lamp['principalAxisTiltDeg']}°" if upright.get('kept')
                     else f"，倾斜 {lamp.get('principalAxisTiltDeg')}°（立正未通过，保留自由拟合）" if upright else '')
            models.append([item['id'], f"RecGen（灯，{len(lamp.get('views') or [])} 个视角）" + ('' if lamp.get('accepted') else '，未通过→代理') + stood,
                           ', '.join(map(str, lamp.get('views') or [])), _iou(lamp)])
        plates = item.get('guardPlates')
        if plates:
            fmt = lambda v: ' / '.join(f"照片{p} {x:.2f}" for p, x in sorted(v.items())) or '—'
            models.append([item['id'], f"照片拟合平板（{len(plates['faces'])} 面），替换 RecGen 块（该块 IoU：{fmt(plates['recgenPartIouByPhoto'])}）",
                           ', '.join(map(str, plates['photos'])), fmt(plates['iouByPhoto'])])
        gantry = item.get('gantryModel')
        if gantry:
            validation = gantry.get('validation', {})
            models.append([item['id'], f"龙门架 {len(gantry.get('members', []))} 根构件", ', '.join(validation),
                           ' / '.join(f"照片{p} {v.get('iou', 0):.2f}" for p, v in validation.items())])
    # One guard model, partitioned into the left / centre / right boards; a promoted A4 fit keeps the RecGen record aside.
    placement = next((run / name for name in ('guard-placement-initializer.json', 'guard-placement.json') if (run / name).is_file()), None)
    guard = _read(placement) if placement else {}
    if guard.get('sourceChecks') and guard.get('generationViews') and not any(item.get('guardPlates') for item in catalog['objects']):
        models.append(['v-guard-left / center / right', 'RecGen 多视角（一个护板模型，按原图掩码分成三块）', ', '.join(map(str, guard['generationViews'])),
                       ' / '.join(f"照片{n + 1} {check['iou']:.2f}" for n, check in enumerate(guard['sourceChecks']))])
    endpoints = [[row['label'], f"{row['heightNative'] * factor * 100:.2f} cm" if factor else f"{row['heightNative']:.4f} native"]
                 for row in report.get('endpointEstimation', {}).get('endpoints', [])]
    return f"""<section><h2>{escape(label)}</h2>
<p>{escape(photos)}；参考照片 {capture['referencePhoto']}。<a href="{escape(url)}">打开这个工位的 3D 报告</a></p>
<p>{len(catalog['objects'])} 个对象，{sum(len(o['observations']) for o in catalog['objects'])} 个观察。尺度：{escape(scale['status'])}{(f"，{factor:.5f} m/native" + ("（条件比例，未验证）" if scale['status'] == 'conditional_unvalidated' else "")) if factor else "，无尺度参考，保持 native 单位，不借用其他工位的比例"}。</p>
{_table(['阶段', '秒'], latency)}
{_table(['Modal 调用', '返回码', '调用窗口 s'], calls)}
<p><small>估计费用（主调用 + 语义调用）：函数窗口 ${estimate.get('functionWindowEstimateUsd', 0) + semantic.get('estimateUsd', 0):.3f}，调用窗口 ${estimate.get('callWindowEstimateUsd', 0) + semantic.get('callWindowEstimateUsd', 0):.3f}（列表价，非账单）。</small></p>
{_table(['对象', '模型', '视角', '每张照片的掩码 IoU'], models)}
{_table(['测点', '离地（条件估计）'], endpoints) if endpoints else ''}
</section>"""


def build(args):
    out = Path(args.out)
    if out.exists():
        raise ValueError('Output must be a new directory')
    out.mkdir(parents=True)
    figures = []
    names = [Path(path).name for path, _ in args.image or []]
    if len(set(names)) != len(names):
        raise ValueError(f'--image files must have distinct names (they are published side by side): {names}')
    for path, caption in args.image or []:
        shutil.copyfile(path, out / Path(path).name)
        figures.append(f'<figure><img src="{escape(Path(path).name)}" alt="{escape(caption)}"><figcaption>{escape(caption)}</figcaption></figure>')
    runs = _read(args.spend)['runs']
    spend = [[r['run'], r['status'], f"{r['estimateUsd']:.3f}", f"{r['callWindowEstimateUsd']:.3f}"] for r in runs]
    failed = [r for r in runs if r['status'].startswith('failed')]
    spend += [['合计（含失败）', f'{len(runs)} 次调用，{len(failed)} 次失败', f"{sum(r['estimateUsd'] for r in runs):.3f}",
               f"{sum(r['callWindowEstimateUsd'] for r in runs):.3f}"],
              ['其中失败', '', f"{sum(r['estimateUsd'] for r in failed):.3f}", f"{sum(r['callWindowEstimateUsd'] for r in failed):.3f}"]]
    sections = ''.join(_cell(label, run, url, args.ui_build_seconds) for label, run, url in args.cell)
    html = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Workcell cells {escape(out.name)}</title>
<style>:root{{--bg:#f4f6f2;--panel:#fff;--ink:#17231f;--muted:#55635e;--line:#d4dbd5;--accent:#12695c}}
@media (prefers-color-scheme:dark){{:root{{--bg:#101614;--panel:#17201d;--ink:#e1e8e4;--muted:#97a49f;--line:#2b3632;--accent:#4fb8a6}}}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.6 system-ui,"PingFang SC",sans-serif}}main{{max-width:1180px;margin:auto;padding:24px 16px 48px}}
a{{color:var(--accent)}}section{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:18px 20px;margin:16px 0}}
h1{{font-size:28px;margin:8px 0}}h2{{font-size:20px;margin:0 0 10px}}.scroll{{overflow-x:auto}}table{{border-collapse:collapse;font-variant-numeric:tabular-nums;min-width:100%;margin:8px 0}}
th,td{{border-bottom:1px solid var(--line);padding:6px 10px;text-align:left;vertical-align:top}}th{{color:var(--muted);font-weight:600}}
.pictures{{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:14px}}figure{{margin:0}}img{{max-width:100%;border-radius:6px}}
figcaption,small{{color:var(--muted);font-size:13px}}</style><main>
<h1>四张照片其实是两个工位：按工位多视角重建</h1>
<section><h2>结论</h2><ul>{''.join(f'<li>{escape(text)}</li>' for text in args.finding or [])}</ul>
<div class="pictures">{''.join(figures)}</div></section>
{sections}
<section><h2>本轮全部 Modal 调用（GPU 与 CPU，含失败）</h2>{_table(['调用', '状态', '函数估计 $', '调用窗口估计 $'], spend)}
<p><small>列表价估计，非账单；actualBilledUsd 为 null。没有物理验证。</small></p></section>
</main></html>
"""
    (out / 'index.html').write_text(html)
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--cell', nargs=3, action='append', required=True, metavar=('LABEL', 'RUN', 'REPORT_URL'))
    parser.add_argument('--image', nargs=2, action='append', metavar=('FILE', 'CAPTION'))
    parser.add_argument('--finding', action='append')
    parser.add_argument('--spend', required=True, help='cost-ledger entry JSON: runs with run, status, estimateUsd, callWindowEstimateUsd')
    parser.add_argument('--ui-build-seconds', type=float, required=True)
    parser.add_argument('--out', required=True)
    print(build(parser.parse_args()))
