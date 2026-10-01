"""Evaluate frozen source-only metrology outputs and publish beside the existing 3D report."""
import argparse
import hashlib
import html
import json
import math
from pathlib import Path
import shutil

from scripts.workcell_photo_calibration import load_measurements

LABELS = {'fence-0': '围栏下横杆', 'post-box-1': '右光幕（按钮侧）', 'post-box-2': '左光幕'}
ROUTES = {'A': '已发布几何 + 10 cm 高度标尺', 'B': '只重新拟合按钮直径标尺',
          'C': '重新拟合实体下沿和附近地面', 'D': '新下沿、地面和直径标尺组合'}


def evaluate(result, measurements):
    """No solver import: check distances enter only after source predictions are frozen."""
    targets = {r['objectId']: r for r in measurements['evaluation']['targets']}
    rows = []
    for route, candidate in result['routes'].items():
        identities = [o['id'] for o in candidate['objects'] if o['id'] in targets]
        if len(identities) != len(set(identities)) or set(identities) != set(targets):
            raise ValueError('Each route must retain every check object exactly once, including unsupported objects')
        for obj in candidate['objects']:
            if obj['id'] not in targets:
                continue
            estimate = obj.get('heightM')
            if obj['status'] == 'unsupported' and estimate is not None:
                raise ValueError('Unsupported geometry cannot be presented as a measurement')
            if estimate is not None:
                scale, native = candidate.get('scaleMPerNative'), obj.get('heightNative')
                if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in (estimate, scale, native)) or scale <= 0:
                    raise ValueError('Finite metrology prediction requires its native height and positive common scale')
                if not math.isclose(estimate, native * scale, rel_tol=1e-8, abs_tol=1e-9):
                    raise ValueError('Metric and native predictions disagree')
            truth = targets[obj['id']]['groundTruthM']
            error = estimate - truth if estimate is not None else None
            rows.append({'route': route, 'routeLabel': candidate['label'], 'objectId': obj['id'],
                         'status': obj['status'], 'estimateM': estimate, 'groundTruthM': truth,
                         'signedErrorM': error, 'absoluteErrorM': abs(error) if error is not None else None,
                         'below3cm': abs(error) < .03 if error is not None else False,
                         'reason': obj.get('reason'), 'sourcePhotos': obj.get('sourcePhotos', [])})
    return rows


def build(baseline, runs, out, measurements):
    if not runs:
        raise ValueError('At least one recorded run is required')
    if out.exists():
        raise ValueError('Use a new report output directory')
    # Freeze all prediction bytes before reading their check-set errors.
    frozen = [(run, p, p.read_bytes()) for run in reversed(runs) for p in sorted(run.rglob('results.json'))
              if 'routes' in json.loads(p.read_text())]
    config = load_measurements(measurements)
    out.mkdir(parents=True)
    page = out / 'page'
    shutil.copytree(baseline / 'page', page)
    web = Path(__file__).resolve().parents[1] / 'web/dist-photo'
    if not (web / 'photo.html').is_file():
        raise ValueError('Build the current photo report UI first')
    shutil.copytree(web, page, dirs_exist_ok=True)
    shutil.copyfile(web / 'photo.html', page / 'index.html')
    evidence = page / 'metrology-data'; evidence.mkdir()
    reports, sections, ledger = [], [], []
    esc = lambda value: html.escape(str(value), quote=True)
    cm = lambda value: '未取得受支持估计' if value is None else f'{100 * value:.2f} cm'
    for run in runs:
        target = evidence / run.name; target.mkdir()
        for p in sorted(run.rglob('*')):
            if p.is_file() and p.suffix in ('.json', '.png', '.jpg', '.glb'):
                dest = target / p.relative_to(run); dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(p, dest)
        ledger.append({'run': run.name, **json.loads((run / 'spend-ledger.json').read_text())})
    for run, path, payload in frozen:
        result = json.loads(payload)
        rows = evaluate(result, config)
        base = Path('metrology-data') / run.name / path.parent.relative_to(run)
        source_hash = hashlib.sha256(payload).hexdigest()
        reports.append({'run': run.name, 'cameraRoute': result['cameraRoute'],
                        'predictionSha256': source_hash, 'source': str(base / path.name), 'comparisons': rows})
        table = []
        for row in rows:
            error = row['signedErrorM']
            table.append(f'<tr><th scope="row">{esc(row["route"])} · {esc(LABELS[row["objectId"]])}</th>'
                         f'<td>{cm(row["estimateM"])}</td><td>{cm(row["groundTruthM"])}</td>'
                         f'<td>{"—" if error is None else f"{100 * error:+.2f} cm"}</td>'
                         f'<td>{"是（仅此检查点）" if row["below3cm"] else "未达到"}</td>'
                         f'<td>{esc(row["reason"] or row["status"])}</td></tr>')
        models = ''.join(f'<li><a href="{esc(base / p.name)}">{esc(p.name)}</a> · 实验几何，按该路线的支持状态解释</li>'
                         for p in sorted(path.parent.glob('*.glb')))
        pictures = ''.join(f'<figure><img loading="lazy" src="{esc(base / picture["file"])}" alt="照片 {esc(picture["photo"])} 测量特征"><figcaption>照片 {esc(picture["photo"])} · {esc(picture["legend"])}</figcaption></figure>'
                           for picture in result.get('evidenceImages', []))
        labels = '；'.join(f'{esc(k)}：{esc(ROUTES[k])}' for k in result['routes'])
        manifest = json.loads((run / 'input-manifest.json').read_text())
        camera_name = ('原图推断相机' if path.parent.name == 'original-cameras' else
                       '方形像素约束的相机优化' if path.parent.name == 'square-pixel-cameras' or manifest.get('controlCameraModel') == 'SIMPLE_PINHOLE' else
                       '横纵焦距分别优化的相机')
        trial = runs.index(run) + 1
        review_file = run / 'review.json'
        review_note = json.loads(review_file.read_text()).get('summary', '') if review_file.is_file() else ''
        sections.append(f'<section><details><summary>第 {trial} 轮 · {esc(camera_name)} — 展开全部对照与原图</summary><p>{esc(review_note)}</p><p>{labels}</p>'
                        '<div class="scroll"><table><thead><tr><th>路线 / 对象</th><th>估计</th><th>现场检查值</th><th>误差</th><th>小于 3 cm</th><th>状态 / 原因</th></tr></thead>'
                        f'<tbody>{"".join(table)}</tbody></table></div><p><a href="{esc(base / path.name)}">完整源观测、标尺拟合与失败原因 JSON</a></p>'
                        f'<details><summary>拟合诊断</summary><pre>{esc(json.dumps({"calibration":result.get("calibration"),"ground":result.get("ground"),"limitations":result.get("limitations")},ensure_ascii=False,indent=2))}</pre></details>'
                        f'<h3>原图证据</h3><div class="pictures">{pictures}</div>'
                        f'{"<h3>本次导出的实验模型</h3><ul>" + models + "</ul>" if models else ""}</details></section>')
    pass_route = any(len(rows := [r for r in report['comparisons'] if r['route'] == key]) == len(LABELS)
                     and all(r['below3cm'] for r in rows)
                     for report in reports if report['run'] == runs[-1].name
                     for key in {r['route'] for r in report['comparisons']} - {'A'})
    headline = '本次有一条路线在三处检查点均小于 3 cm；仍需检查可观测性与模型一致性。' if pass_route else '现有四图实验尚未达到三处离地误差都小于 3 cm。'
    duration = sum(r['functionSeconds'] for r in ledger if r.get('functionSeconds') is not None)
    cost = sum(r['estimateUsd'] for r in ledger if r.get('estimateUsd') is not None)
    missing = sum(r.get('functionSeconds') is None or r.get('estimateUsd') is None for r in ledger)
    call_cost = sum(r['callWindowEstimateUsd'] for r in ledger if r.get('callWindowEstimateUsd') is not None)
    summary = {'schemaVersion': 1, 'targetAbsoluteErrorM': .03, 'allThreeBelow3cm': pass_route,
               'groundTruthUsedForFitOrSelection': False,
               'evaluationScope': 'Previously disclosed three-point development check set, not blind validation.',
               'reports': reports, 'ledger': ledger, 'knownFunctionSeconds': duration,
               'knownEstimatedFunctionCostUsd': cost, 'functionAccountingMissingRuns': missing,
               'functionSecondsTotal': None if missing else duration,
               'estimatedFunctionCostUsd': None if missing else cost,
               'callWindowEstimateUsd': call_cost, 'actualBilledUsd': None,
               'timingScope': 'Cached inference inputs; geometric ablation only. Not a new end-to-end one-shot benchmark.',
               'baselineEndToEndSeconds': 334.60}
    (out / 'metrology-evaluation.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    shutil.copy2(out / 'metrology-evaluation.json', page / 'metrology-evaluation.json')
    data = json.loads((page / 'scene-report.json').read_text())
    data['metrology'] = {'summary': headline + f' 完整场景保留此前模型；{len(runs)} 轮自动实验、测量结果和原图证据见实验页。', 'reportURL': 'metrology.html'}
    for target in (out / 'scene-report.json', page / 'scene-report.json'):
        target.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    latest = next((report for report in reports if report['run'] == runs[-1].name and 'original-cameras' in report['source']), None)
    latest_result = json.loads((page / latest['source']).read_text()) if latest else {}
    overview = []
    primary = 'D' if latest_result.get('calibration', {}).get('new', {}).get('status') == 'available' else 'C'
    scale_note = '通过源观测验证的圆直径尺度' if primary == 'D' else '原 10 cm 整体高度尺度'
    for identity, label in LABELS.items():
        if latest is None:
            truth = next(row['groundTruthM'] for row in config['evaluation']['targets'] if row['objectId'] == identity)
            overview.append(f'<tr><th scope="row">{label}</th><td>—</td><td>未输出预测</td><td>{cm(truth)}</td><td>最新原相机路线未完成；详见运行记录</td></tr>')
            continue
        old = next(row for row in latest['comparisons'] if row['route'] == 'A' and row['objectId'] == identity)
        new = next(row for row in latest['comparisons'] if row['route'] == primary and row['objectId'] == identity)
        outcome = ('未稳定识别同一个实体下沿' if new['estimateM'] is None else
                   f'绝对误差 {cm(new["absoluteErrorM"])}')
        overview.append(f'<tr><th scope="row">{label}</th><td>{cm(old["estimateM"])}</td>'
                        f'<td>{cm(new["estimateM"])}</td><td>{cm(new["groundTruthM"])}</td><td>{outcome}</td></tr>')
    detail_image = Path(latest['source']).parent / 'raw-image-features-4.jpg' if latest else None
    evidence_html = (f'<a href="{esc(detail_image)}"><img src="{esc(detail_image)}" alt="照片 4 的按钮轮廓、实体下沿与地面拟合点"></a>'
                     if detail_image and (page / detail_image).is_file() else '<p>最新原相机路线没有输出该证据图；未替换成历史图片。</p>')
    red = next((row for row in latest_result.get('calibration', {}).get('candidates', []) if row['feature'] == 'redActuatorDiameterM'), {})
    circle_note = ('红色圆边在留出照片中的最大投影偏差为 '
                   f'{max(v["maxErrorRawPx"] for v in red.get("heldOutPhotos", []) if v.get("maxErrorRawPx") is not None):.1f} 个原图像素。'
                   if any(v.get('maxErrorRawPx') is not None for v in red.get('heldOutPhotos', [])) else '')
    calibration_note = ('本轮有圆直径尺度通过源观测门槛，仍需结合实体下沿和附近地面核对真实测量。'
                        if primary == 'D' else '本轮圆直径尺度尚未通过源观测门槛，因此该尺度没有替换主报告。')
    latest_seconds = ledger[-1].get('functionSeconds')
    latest_timing = '未取得完整函数时间' if latest_seconds is None else f'{latest_seconds:.2f} 秒'
    markup = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>按钮标尺测量实验 · Panoptes</title>
<style>body{{font:16px/1.65 system-ui,sans-serif;margin:0;background:#f5f6f2;color:#20372f}}main{{max-width:1150px;margin:auto;padding:30px 20px}}h1{{font-size:32px;line-height:1.3}}h2{{margin-top:34px}}section{{margin:24px 0;padding:20px;background:white;border:1px solid #d7dfd8;border-radius:12px}}a{{color:#176750}}.lead{{font-size:21px}}table{{width:100%;border-collapse:collapse;font-size:14px}}td,th{{border-bottom:1px solid #d7dfd8;text-align:left;padding:9px}}.scroll{{overflow:auto}}img{{display:block;width:100%;height:auto}}figure{{margin:0}}.pictures{{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,400px),1fr));gap:16px}}pre{{max-height:500px;overflow:auto;white-space:pre-wrap;font-size:12px}}.action{{display:inline-block;padding:10px 16px;background:#235c48;color:white;border-radius:6px;text-decoration:none}}</style>
<main><p>PANOPTES · 2026-10-01 · 四张原图</p><h1>按钮标尺能把离地距离测到什么程度？</h1><p class="lead">{headline}</p>
<p>输入：现有四张照片；按钮红帽直径 4 cm、主体最大直径 8.5 cm、整体高度 10 cm。没有新增实测相机参数或拍摄高度。相机先从图像推断，再测试多视图优化。</p>
<p><a class="action" href="index.html#scene">打开完整 52 对象可旋转 3D</a> <a href="metrology-evaluation.json">下载本次对照与耗时</a></p>
<section><h2>现在实际测到什么</h2><div class="scroll"><table><thead><tr><th>对象</th><th>已发布估计</th><th>本次边线重建</th><th>现场检查值</th><th>本次结果</th></tr></thead><tbody>{''.join(overview)}</tbody></table></div>
<p>这里的本次边线重建使用原推断相机、{scale_note}，以及重新提取的下沿和附近地面。该地面仍未确认完全来自混凝土地面。优化相机的全部对照见下面实验记录。{calibration_note}</p>
<p>围栏 20 cm、光幕 24 cm 只在预测保存后用于计算误差，未进入求解器或候选选择；这三处已经公开给开发者，属于开发检查集。</p></section>
<section><h2>这轮检查了什么</h2><ol><li><strong>按钮直径能否跨照片成立。</strong>{circle_note}{calibration_note}已知尺寸需要和正确的圆边、相机投影对应起来。</li><li><strong>光幕真实下端能否匹配。</strong>检查原图实体边线及可见区间；部分照片有黑色护柱、端子和线缆遮挡。旧包围框最低点不能作为已修复的实体底边。</li><li><strong>地面是否有附近的原图支持。</strong>下方蓝点为拟合地面的入选点、红点为被排除点。地面误差会直接进入离地距离，低像素误差本身不能验证厘米精度。</li></ol></section>
<section><h2>照片 4 · 实际提取证据</h2><p>紫色：两个按钮部件的圆形轮廓；黄色：围栏；绿色/蓝色：两侧光幕。仅对通过边线验证且有附近地面支持的对象绘制离地垂线。</p>{evidence_html}</section>
<h2>{len(runs)} 轮完整实验记录</h2><p>A–D 是四个固定对照；每次预测和失败记录均保留。D 是现有相机下的新标尺与边线组合，本轮尚未实现按钮尺寸、相机、地面全部变量一起优化。</p>
{''.join(sections)}
<section><h2>实际用时与花费</h2><p>本次 {len(ledger)} 次临时云调用，均为 2×A100-80GB，各次调用内部并行比较相机路线。最新一轮函数执行：{latest_timing}；已知函数时间合计 {duration:.2f} 秒，按预留资源费率估算 ${cost:.3f}。{missing} 次缺少完整函数记录。所有调用窗口费用估算 ${call_cost:.3f}（含调度，非账单）。</p><p>这是复用分割和模型的几何增量实验。此前四照片完整报告为 334.60 秒，本次没有重跑完整神经网络流水线。完整 52 对象场景仍使用此前模型；新的实验几何未获验证，尚未替换进去。</p><p>每次调用、失败记录、源码哈希和估算依据见 <a href="metrology-evaluation.json">运行账目</a>。</p></section></main></html>'''
    (page / 'metrology.html').write_text(markup)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True, action='append')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--measurements', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.baseline, args.run, args.out, args.measurements), ensure_ascii=False))
