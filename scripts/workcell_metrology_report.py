"""Evaluate frozen source-only metrology outputs and publish beside the existing 3D report."""
import argparse
import hashlib
import html
import json
import math
from pathlib import Path
import shutil

from scripts.workcell_photo_calibration import load_measurements

LABELS = {'fence-0': '右围栏下横杆', 'fence-1': '左围栏下横杆',
          'post-box-1': '右光幕（按钮侧）', 'post-box-2': '左光幕'}
ROUTES = {'A': '本路线输入几何 + 10 cm 高度标尺（新深度分支会重新生成）', 'B': '只重新拟合按钮直径标尺',
          'C': '重新拟合实体下沿和附近地面', 'D': '新下沿、地面和直径标尺组合',
          'J': '三个已知尺寸与相机、场景特征共同求解'}


def _variant(path):
    parts = Path(path).parts
    if 'double-1036' in parts:
        return 'double-1036', '新深度 784×1036'
    if 'default-518' in parts:
        return 'default-518', '新深度 392×518'
    if any(part.startswith('cached-') for part in parts):
        return 'cached', '冻结的此前深度'
    return 'historical', '历史输入'


def _primary(reports, page, run_name):
    """Selection reads source-validation flags; it never inspects check-set errors."""
    latest = [row for row in reports if row['run'] == run_name]
    for variant in ('double-1036', 'default-518', 'cached', 'historical'):
        for row in latest:
            data = json.loads((page / row['source']).read_text())
            if (row['variant'] == variant and 'J' in data.get('routes', {}) and
                    data.get('jointReference', {}).get('status') == 'available'):
                return row
    # A rejected joint trial remains visible; never substitute a closer check-set answer.
    for name in ('cached-joint-cameras', 'joint-cameras', 'cached-original-cameras', 'original-cameras'):
        choices = [row for row in latest if Path(row['source']).parent.name == name]
        if choices:
            return sorted(choices, key=lambda row: (row['variant'] != 'cached', row['variant'] != 'double-1036', row['source']))[0]
    return None


def _preview_options(runs, page):
    """Each option owns one native world; joint candidates never share a floor group."""
    choices = []
    for run in runs:
        for geometry_path in sorted(run.glob('*/fresh-geometry/geometry.json')):
            variant, label = _variant(geometry_path)
            geometry = json.loads(geometry_path.read_text())
            base = Path('metrology-data') / run.name / geometry_path.parent.relative_to(run)
            assets = [str(base / name) for name in ('floor-fitted.glb', 'fence-fitted.glb') if (page / base / name).is_file()]
            if assets:
                stages_path = geometry_path.parent.parent / 'measurements/stages.json'
                stages = json.loads(stages_path.read_text()) if stages_path.is_file() else {}
                identity_note = '源对象身份匹配失败，不能据此替换原对象。' if stages.get('sourceIdentity', {}).get('status') == 'failed' else ''
                choices.append({'label': label + ' · 地面与围栏', 'world': str(base), 'kind': 'fresh-geometry',
                                'status': 'conditional', 'units': 'native', 'assets': assets,
                                'floor': geometry.get('floor'),
                                'note': '条件几何：该分支独立重建的地面和围栏；原生单位，尚未证明厘米精度。' + identity_note})
        for model in sorted(run.rglob('reference-candidate.glb')):
            sidecar = model.with_suffix('.json')
            if not sidecar.is_file():
                continue
            metadata = json.loads(sidecar.read_text())
            if metadata.get('units') != 'native' or (metadata.get('status') != 'available' and metadata.get('metricScaleMPerNative') is not None):
                raise ValueError('Candidate preview must preserve native units and reject unsupported metric scales')
            _, label = _variant(model)
            base = Path('metrology-data') / run.name / model.parent.relative_to(run)
            choices.append({'label': label + ' · 按钮候选（' + ('条件支持' if metadata.get('status') == 'available' else '未通过验证') + '）',
                            'world': str(base), 'kind': 'joint-reference', 'status': metadata.get('status', 'unsupported'),
                            'units': 'native', 'assets': [str(base / model.name)], 'floor': None,
                            'axis': metadata.get('axisNative'), 'base': metadata.get('baseNative'),
                            'note': '10 / 8.5 / 4 cm 是用户提供的尺寸约束，不是独立测量结果；灰色底座仍为图像拟合的形状假设。'
                                    '独立候选预览，网格不是现场地面；未通过验证的候选不显示厘米估值。'})
    return choices


def _preview_markup(choices):
    if not choices:
        return '<section><h2>本轮可旋转模型</h2><p>本轮尚无已导出的候选模型；完整 52 对象场景可从顶部入口打开。</p></section>'
    payload = json.dumps(choices, ensure_ascii=False).replace('<', '\\u003c')
    return '''<section id="models"><h2>本轮可旋转模型</h2><label for="model-choice">选择本轮模型：</label>
<select id="model-choice"></select><p id="model-note"></p><p>拖动旋转 · 滚轮缩放 · 右键平移。每次只加载一个坐标系。</p>
<canvas id="model-preview" aria-label="本轮三维候选模型，可拖动旋转" style="display:block;width:100%;height:460px;touch-action:none"></canvas>
<p id="model-status" role="status"></p></section>
<script type="importmap">{"imports":{"three":"./viewer-assets/three.module.js","three/addons/":"./viewer-assets/addons/"}}</script>
<script type="module">
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {GLTFLoader} from 'three/addons/loaders/GLTFLoader.js';
const choices = ''' + payload + ''';
const canvas=document.querySelector('#model-preview'), select=document.querySelector('#model-choice');
const status=document.querySelector('#model-status'), note=document.querySelector('#model-note');
const scene=new THREE.Scene(); scene.background=new THREE.Color('#17202a');
const renderer=new THREE.WebGLRenderer({canvas,antialias:true}); renderer.setPixelRatio(Math.min(devicePixelRatio,2));
const camera=new THREE.PerspectiveCamera(45,1,.001,10000), controls=new OrbitControls(camera,canvas);
controls.enableDamping=true; scene.add(new THREE.HemisphereLight(0xffffff,0x405060,2));
const light=new THREE.DirectionalLight(0xffffff,2); light.position.set(4,8,5); scene.add(light);
const loader=new GLTFLoader(); let active=new THREE.Group(), ticket=0; scene.add(active);
function dispose(group){group.traverse(o=>{o.geometry?.dispose(); for(const m of (Array.isArray(o.material)?o.material:[o.material])){if(m){for(const v of Object.values(m)) if(v?.isTexture)v.dispose();m.dispose();}}});}
for(const [index,row] of choices.entries()){const option=document.createElement('option');option.value=index;option.textContent=row.label;select.append(option);}
async function show(index){
  const current=++ticket, row=choices[index]; note.textContent=row.note;status.textContent='加载模型…';
  scene.remove(active);dispose(active);active=new THREE.Group();scene.add(active);
  const next=new THREE.Group();
  try{
    for(const url of row.assets){const gltf=await loader.loadAsync(url);next.add(gltf.scene);}
    if(current!==ticket){dispose(next);return;}
    if(row.floor){const normal=new THREE.Vector3(...row.floor.normal);const norm=normal.length();
      next.quaternion.setFromUnitVectors(normal.normalize(),new THREE.Vector3(0,1,0));next.position.y=row.floor.offset/norm;}
    else if(row.axis && row.base){next.quaternion.setFromUnitVectors(new THREE.Vector3(...row.axis).normalize(),new THREE.Vector3(0,1,0));
      next.position.copy(new THREE.Vector3(...row.base).applyQuaternion(next.quaternion).negate());}
    active.add(next);const box=new THREE.Box3().setFromObject(next), center=box.getCenter(new THREE.Vector3());
    const size=Math.max(...box.getSize(new THREE.Vector3()).toArray(),.01);
    const grid=new THREE.GridHelper(size*1.5,12,0x93abb1,0x334951);grid.position.set(center.x,row.floor?0:box.min.y,center.z);active.add(grid);
    const axes=new THREE.AxesHelper(size*.25);axes.position.copy(grid.position);active.add(axes);
    controls.target.copy(center);camera.position.copy(center).add(new THREE.Vector3(1,.7,1).multiplyScalar(size*1.3));
    camera.near=size/1000;camera.far=size*100;camera.updateProjectionMatrix();controls.update();
    status.textContent='已加载 '+row.assets.length+' 个模型 · '+row.label+' · 原生单位';
    canvas.dataset.world=row.world;canvas.dataset.kind=row.kind;
  }catch(error){dispose(next);if(current===ticket)status.textContent='模型加载失败：'+error.message;}
}
select.addEventListener('change',()=>show(Number(select.value)));
new ResizeObserver(()=>{const width=canvas.clientWidth,height=canvas.clientHeight;renderer.setSize(width,height,false);camera.aspect=width/height;camera.updateProjectionMatrix();}).observe(canvas);
renderer.setAnimationLoop(()=>{controls.update();renderer.render(scene,camera);});show(0);
</script>'''


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


def build(baseline, runs, out, measurements, depth_runs=()):
    if not runs:
        raise ValueError('At least one recorded run is required')
    if out.exists():
        raise ValueError('Use a new report output directory')
    # Freeze all prediction bytes before reading their check-set errors.
    frozen = [(run, p, p.read_bytes()) for run in reversed(runs) for p in sorted(run.rglob('results.json'))
              if 'routes' in json.loads(p.read_text())]
    config = load_measurements(measurements)
    check_ids = tuple(row['objectId'] for row in config['evaluation']['targets'])
    out.mkdir(parents=True)
    page = out / 'page'
    shutil.copytree(baseline / 'page', page)
    web = Path(__file__).resolve().parents[1] / 'web/dist-photo'
    if not (web / 'photo.html').is_file():
        raise ValueError('Build the current photo report UI first')
    shutil.copytree(web, page, dirs_exist_ok=True)
    shutil.copyfile(web / 'photo.html', page / 'index.html')
    evidence = page / 'metrology-data'; evidence.mkdir(exist_ok=True)
    reports, sections, ledger, execution = [], [], [], []
    esc = lambda value: html.escape(str(value), quote=True)
    cm = lambda value: '未取得受支持估计' if value is None else f'{100 * value:.2f} cm'
    for run in runs:
        target = evidence / run.name; target.mkdir(exist_ok=True)
        for p in sorted(run.rglob('*')):
            if p.is_file() and p.suffix in ('.json', '.png', '.jpg', '.glb'):
                dest = target / p.relative_to(run); dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(p, dest)
        ledger.append({'run': run.name, 'phase': 'geometry-metrology', **json.loads((run / 'spend-ledger.json').read_text())})
        run_record = json.loads((run / 'run.json').read_text()) if (run / 'run.json').is_file() else {}
        stage_records = []
        for stage_path in sorted(run.rglob('stages.json')):
            variant, label = _variant(stage_path.relative_to(run))
            stage_records.append({'variant': variant, 'variantLabel': label,
                                  'source': str(Path('metrology-data') / run.name / stage_path.relative_to(run)),
                                  'stages': json.loads(stage_path.read_text())})
        joint_records = []
        for joint_path in sorted(run.rglob('joint-reference.json')):
            candidate = json.loads(joint_path.read_text())
            variant, label = _variant(joint_path.relative_to(run))
            joint_records.append({'variant': variant, 'variantLabel': label, 'status': candidate.get('status'),
                                  'reason': candidate.get('reason'), 'diagnostics': candidate.get('diagnostics'),
                                  'source': str(Path('metrology-data') / run.name / joint_path.relative_to(run))})
        execution.append({'run': run.name, 'callStatus': ledger[-1].get('status'), 'callError': ledger[-1].get('error'),
                          'records': run_record.get('records', {}), 'branchStages': stage_records,
                          'jointCandidates': joint_records})
    depth = None
    for depth_run in depth_runs:
        depth = json.loads((depth_run / 'run.json').read_text())
        target = evidence / depth_run.name; target.mkdir(exist_ok=True)
        for filename in ('run.json', 'spend-ledger.json', 'input-manifest.json'):
            if (depth_run / filename).is_file():
                shutil.copyfile(depth_run / filename, target / filename)
        ledger.append({'run': depth_run.name, 'phase': 'neural-depth-ablation',
                       **json.loads((depth_run / 'spend-ledger.json').read_text())})
    for run, path, payload in frozen:
        result = json.loads(payload)
        rows = evaluate(result, config)
        base = Path('metrology-data') / run.name / path.parent.relative_to(run)
        source_hash = hashlib.sha256(payload).hexdigest()
        variant, variant_label = _variant(path.relative_to(run))
        reports.append({'run': run.name, 'cameraRoute': result['cameraRoute'],
                        'variant': variant, 'variantLabel': variant_label,
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
        camera_name = ('三个已知尺寸联合求解的相机' if path.parent.name.endswith('joint-cameras') else
                       '原图推断相机' if path.parent.name.endswith('original-cameras') else
                       '方形像素约束的相机优化' if path.parent.name == 'square-pixel-cameras' or manifest.get('controlCameraModel') == 'SIMPLE_PINHOLE' else
                       '横纵焦距分别优化的相机')
        trial = runs.index(run) + 1
        review_file = run / 'review.json'
        review_note = json.loads(review_file.read_text()).get('summary', '') if review_file.is_file() else ''
        sections.append(f'<section><details><summary>第 {trial} 轮 · {esc(variant_label)} · {esc(camera_name)} — 展开全部对照与原图</summary><p>{esc(review_note)}</p><p>{labels}</p>'
                        '<div class="scroll"><table><thead><tr><th>路线 / 对象</th><th>估计</th><th>现场检查值</th><th>误差</th><th>小于 3 cm</th><th>状态 / 原因</th></tr></thead>'
                        f'<tbody>{"".join(table)}</tbody></table></div><p><a href="{esc(base / path.name)}">完整源观测、标尺拟合与失败原因 JSON</a></p>'
                        f'<details><summary>拟合诊断</summary><pre>{esc(json.dumps({"jointReference":result.get("jointReference"),"calibration":result.get("calibration"),"ground":result.get("ground"),"limitations":result.get("limitations")},ensure_ascii=False,indent=2))}</pre></details>'
                        f'<h3>原图证据</h3><div class="pictures">{pictures}</div>'
                        f'{"<h3>本次导出的实验模型</h3><ul>" + models + "</ul>" if models else ""}</details></section>')
    latest = _primary(reports, page, runs[-1].name)
    latest_result = json.loads((page / latest['source']).read_text()) if latest else {}
    primary = 'J' if 'J' in latest_result.get('routes', {}) else ('D' if latest_result.get('calibration', {}).get('new', {}).get('status') == 'available' else 'C')
    primary_rows = [row for row in latest['comparisons'] if row['route'] == primary] if latest else []
    pass_route = len(primary_rows) == len(check_ids) and all(row['below3cm'] for row in primary_rows)
    headline = '当前主对照在三处检查点均小于 3 cm；仍需检查可观测性与模型一致性。' if pass_route else '现有四图实验尚未达到三处离地误差都小于 3 cm。'
    duration = sum(r['functionSeconds'] for r in ledger if r.get('functionSeconds') is not None)
    cost = sum(r['estimateUsd'] for r in ledger if r.get('estimateUsd') is not None)
    missing = sum(r.get('functionSeconds') is None or r.get('estimateUsd') is None for r in ledger)
    call_cost = sum(r['callWindowEstimateUsd'] for r in ledger if r.get('callWindowEstimateUsd') is not None)
    summary = {'schemaVersion': 1, 'targetAbsoluteErrorM': .03, 'allThreeBelow3cm': pass_route,
               'groundTruthUsedForFitOrSelection': False,
               'evaluationScope': 'Previously disclosed three-point development check set, not blind validation.',
               'reports': reports, 'ledger': ledger, 'knownFunctionSeconds': duration,
               'executionRecords': execution,
               'knownEstimatedFunctionCostUsd': cost, 'functionAccountingMissingRuns': missing,
               'functionSecondsTotal': None if missing else duration,
               'estimatedFunctionCostUsd': None if missing else cost,
               'callWindowEstimateUsd': call_cost, 'actualBilledUsd': None,
               'timingScope': ('Fresh paired neural depth inference plus separate geometry/metrology experiments; segmentation and complete 52-object reconstruction were not rerun. Not a new end-to-end one-shot benchmark.' if depth is not None else
                               'Cached inference inputs; geometric ablation only. Not a new end-to-end one-shot benchmark.'),
               'primarySource': latest['source'] if latest else None, 'primaryRoute': primary,
               'selectionRule': 'Prefer source-validated J: double-1036, default-518, cached, historical. If none pass, show rejected cached J (then current J); never select by check-set error.',
               'depthRun': depth, 'modelPreviews': _preview_options([runs[-1]], page),
               'baselineEndToEndSeconds': 334.60}
    (out / 'metrology-evaluation.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    shutil.copy2(out / 'metrology-evaluation.json', page / 'metrology-evaluation.json')
    data = json.loads((page / 'scene-report.json').read_text())
    data['metrology'] = {'summary': headline + f' 完整场景保留此前模型；{len(runs)} 轮自动实验、测量结果和原图证据见实验页。', 'reportURL': 'metrology.html'}
    for target in (out / 'scene-report.json', page / 'scene-report.json'):
        target.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    overview = []
    joint = latest_result.get('jointReference', {})
    scale_note = (('三个已知尺寸联合求解的尺度' if joint.get('status') == 'available' else '联合求解未通过；没有可用的新尺度') if primary == 'J' else
                  '通过源观测验证的圆直径尺度' if primary == 'D' else '原 10 cm 整体高度尺度')
    for identity in check_ids:
        label = LABELS[identity]
        if latest is None:
            truth = next(row['groundTruthM'] for row in config['evaluation']['targets'] if row['objectId'] == identity)
            overview.append(f'<tr><th scope="row">{label}</th><td>—</td><td>未输出预测</td><td>{cm(truth)}</td><td>最新原相机路线未完成；详见运行记录</td></tr>')
            continue
        old = next(row for row in latest['comparisons'] if row['route'] == 'A' and row['objectId'] == identity)
        new = next(row for row in latest['comparisons'] if row['route'] == primary and row['objectId'] == identity)
        outcome = (('联合求解未通过源观测验证' if primary == 'J' and joint.get('status') != 'available' else '未稳定识别同一个实体下沿或缺少地面支持') if new['estimateM'] is None else
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
    if primary == 'J':
        errors = [row['maxErrorRawPx'] for row in joint.get('heldOutPhotos', []) if row.get('maxErrorRawPx') is not None]
        circle_note = (f'整套按钮留出照片检查的最大偏差为 {max(errors):.1f} 个原图像素（共享相机初始化）。'
                       if errors else '整套按钮留出照片检查未完成。')
        calibration_note = ('三个已知尺寸共同约束相机和按钮；联合求解通过源观测门槛，仍须验证实体端点与地面。'
                            if joint.get('status') == 'available' else
                            '联合求解使用三个已知尺寸，源观测检查未通过，不输出该路线的厘米估值；失败阶段见完整记录。')
    camera_note = '三个尺寸共同约束后输出的候选相机' if primary == 'J' else '原推断相机'
    method_note = ('J 同时约束整体高度 10 cm、主体最大直径 8.5 cm、红帽直径 4 cm，并优化相机与场景特征。'
                   '按钮内部高度和底座形状由图像估计；三维形状假设仍需留出照片验证。地面与物体边线随后用新相机重新求解。'
                   if primary == 'J' else 'D 是现有相机下的新标尺与边线组合，未将按钮尺寸与相机变量一起优化。')
    joint_file = (Path(latest['source']).parent.parent / ('cached-joint-button' if latest and latest['variant'] == 'cached' else 'joint-button') / 'joint-reference.json') if latest else Path('missing')
    joint_link = f'<p><a href="{esc(joint_file)}">三个尺寸如何进入求解、拟合状态和留出照片检查</a></p>' if (page / joint_file).is_file() else ''
    latest_seconds = next(row for row in ledger if row['run'] == runs[-1].name).get('functionSeconds')
    latest_timing = '未取得完整函数时间' if latest_seconds is None else f'{latest_seconds:.2f} 秒'
    depth_section = ''
    timing_note = '这是复用分割和模型的几何增量实验；本次没有重跑神经深度推理。'
    if depth is not None:
        runtime_rows = []
        for name in ('default-518', 'double-1036'):
            branch = depth.get('branches', {}).get(name, {})
            timing = branch.get('timing', {})
            seconds = lambda key: '未取得' if timing.get(key) is None else f'{timing[key]:.2f} s'
            vram = '未取得' if timing.get('peakAllocatedGiB') is None else f'{timing["peakAllocatedGiB"]:.2f} GiB'
            runtime_status = {'completed': '完成', 'failed': '失败', 'timeout': '超时'}.get(branch.get('status'), '未完成')
            runtime_rows.append(f'<tr><th scope="row">{esc(_variant(name)[1])}</th><td>{seconds("inferenceSeconds")}</td>'
                                f'<td>{seconds("modelLoadSeconds")}</td><td>{seconds("wallSecondsInContainer")}</td>'
                                f'<td>{vram}</td><td>{runtime_status}</td></tr>')
        comparison_rows = []
        current_execution = execution[-1]
        for variant in ('cached', 'default-518', 'double-1036'):
            found_routes = set()
            for item in reports:
                if item['run'] != runs[-1].name or item['variant'] != variant:
                    continue
                route = 'J' if any(row['route'] == 'J' for row in item['comparisons']) else 'C'
                found_routes.add(route)
                chosen = [row for row in item['comparisons'] if row['route'] == route]
                values = {row['objectId']: row for row in chosen}
                cells = ''.join(f'<td>{cm(values[key]["estimateM"])}</td>' if key in values else '<td>未输出</td>' for key in check_ids)
                source_status = '联合标定候选' if route == 'J' else '原相机 + 新下沿 / 地面（条件估计）'
                status_text = '、'.join({'conditional': '条件估计', 'available': '源观测支持', 'unsupported': '未知 / 未通过'}.get(value, value) for value in sorted({row['status'] for row in chosen}))
                comparison_rows.append(f'<tr><th scope="row">{esc(item["variantLabel"])} · {route}</th>{cells}'
                                       f'<td>{source_status}；{esc(status_text)}</td></tr>')
            joint_record = next((row for row in current_execution['jointCandidates'] if row['variant'] == variant), None)
            failures = [f'{name}: {stage.get("error") or stage.get("reason") or stage["status"]}'
                        for row in current_execution['branchStages'] if row['variant'] == variant
                        for name, stage in row['stages'].items() if stage.get('status') in ('failed', 'unsupported', 'timeout')]
            if not found_routes or ('J' not in found_routes and joint_record is not None):
                call_error = current_execution.get('callError')
                detail = '; '.join(failures) or (joint_record or {}).get('reason') or (
                    f'云调用失败：{call_error}' if call_error else '该分支未导出测量；请查看运行阶段记录')
                label = '冻结的此前深度' if variant == 'cached' else _variant(variant)[1]
                suffix = ' · J' if joint_record is not None else ''
                comparison_rows.append(f'<tr><th scope="row">{esc(label)}{suffix}</th><td colspan="3">未导出受支持测量</td>'
                                       f'<td>{esc(detail)}</td></tr>')
        depth_ledger = next(row for row in reversed(ledger) if row['phase'] == 'neural-depth-ablation')
        depth_seconds = depth_ledger.get('functionSeconds')
        depth_time = '未取得' if depth_seconds is None else f'{depth_seconds:.2f} 秒'
        depth_section = f'''<section><h2>本轮：同四张原图，深度分辨率加倍</h2>
<p>两个分支使用同一权重版本、同一原图、相同推理参数，双 A100 各跑一条。每个分支重新求自己的相机、地面和围栏；没有把旧坐标直接贴到新深度上。分辨率变化也会影响上游边缘过滤，因此这是整条分辨率管线的对照，不能单独归因于深度网络。</p>
<div class="scroll"><table><thead><tr><th>分支</th><th>四图纯推理</th><th>加载模型</th><th>Worker 内完整处理</th><th>峰值分配显存</th><th>状态</th></tr></thead><tbody>{''.join(runtime_rows)}</tbody></table></div>
<p>神经深度云函数 {depth_time}；后续几何 / 测量云函数 {latest_timing}。两次调用分别计时、计费；这不是完整 52 对象 oneshot 的新成绩。</p>
<div class="scroll"><table><thead><tr><th>深度 / 测量路线</th><th>围栏</th><th>右光幕</th><th>左光幕</th><th>依据 / 状态</th></tr></thead><tbody>{''.join(comparison_rows)}</tbody></table></div>
<p>新分支的 A 是该分支重新生成的几何与高度标尺，不能称作此前已发布估计。C 仍为条件估计；J 未通过源观测门槛时保持未知。完整误差与照片见折叠记录。</p></section>'''
        timing_note = '本轮重跑了两个分辨率的神经深度推理，并单独运行几何 / 测量实验；复用了此前分割，没有重跑完整 52 对象建模流水线。'
    preview_markup = _preview_markup(summary['modelPreviews'])
    selection_label = latest['variantLabel'] if latest else '未输出'
    markup = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>按钮标尺测量实验 · Panoptes</title>
<style>body{{font:16px/1.65 system-ui,sans-serif;margin:0;background:#f5f6f2;color:#20372f}}main{{max-width:1150px;margin:auto;padding:30px 20px}}h1{{font-size:32px;line-height:1.3}}h2{{margin-top:34px}}section{{margin:24px 0;padding:20px;background:white;border:1px solid #d7dfd8;border-radius:12px}}a{{color:#176750}}.lead{{font-size:21px}}table{{width:100%;border-collapse:collapse;font-size:14px}}td,th{{border-bottom:1px solid #d7dfd8;text-align:left;padding:9px}}.scroll{{overflow:auto}}img{{display:block;width:100%;height:auto}}figure{{margin:0}}.pictures{{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,400px),1fr));gap:16px}}pre{{max-height:500px;overflow:auto;white-space:pre-wrap;font-size:12px}}.action{{display:inline-block;padding:10px 16px;background:#235c48;color:white;border-radius:6px;text-decoration:none}}</style>
<main><p>PANOPTES · 2026-10-01 · 四张原图</p><h1>按钮标尺能把离地距离测到什么程度？</h1><p class="lead">{headline}</p>
<p>输入：现有四张照片；用户提供的约束为按钮红帽直径 4 cm、主体最大直径 8.5 cm、整体高度 10 cm。这些不是本轮独立测出的尺寸；灰色底座仍为图像拟合的形状假设。没有新增实测相机参数或拍摄高度。相机先从图像推断，再测试多视图优化。</p>
<p><a class="action" href="index.html#scene">打开完整 52 对象可旋转 3D</a> <a href="metrology-evaluation.json">下载本次对照与耗时</a></p>
{preview_markup}
{depth_section}
<section><h2>当前主对照 · {selection_label} · {primary}</h2><p>选择顺序固定：源观测验证通过的 J 优先取 784×1036，再取 392×518，再取冻结深度；全部未通过时展示未通过的冻结 J。选择不读取现场误差。</p><div class="scroll"><table><thead><tr><th>对象</th><th>该输入原几何估计</th><th>本次边线重建</th><th>现场检查值</th><th>本次结果</th></tr></thead><tbody>{''.join(overview)}</tbody></table></div>
<p>这里的本次边线重建使用{camera_note}、{scale_note}，以及重新提取的下沿和附近地面。该地面仍未确认完全来自混凝土地面。优化相机的全部对照见下面实验记录。{calibration_note}</p>
<p>围栏 20 cm、光幕 24 cm 只在预测保存后用于计算误差，未进入求解器或候选选择；这三处已经公开给开发者，属于开发检查集。</p></section>
<section><h2>这轮检查了什么</h2><ol><li><strong>按钮直径能否跨照片成立。</strong>{circle_note}{calibration_note}已知尺寸需要和正确的圆边、相机投影对应起来。</li><li><strong>光幕真实下端能否匹配。</strong>检查原图实体边线及可见区间；部分照片有黑色护柱、端子和线缆遮挡。旧包围框最低点不能作为已修复的实体底边。</li><li><strong>地面是否有附近的原图支持。</strong>下方蓝点为拟合地面的入选点、红点为被排除点。地面误差会直接进入离地距离，低像素误差本身不能验证厘米精度。</li></ol></section>
<section><h2>照片 4 · 实际提取证据</h2><p>紫色：两个按钮部件的圆形轮廓；黄色：围栏；绿色/蓝色：两侧光幕。仅对通过边线验证且有附近地面支持的对象绘制离地垂线。</p>{evidence_html}</section>
<h2>{len(runs)} 轮完整实验记录</h2><p>A–D 是四个固定对照；每次预测和失败记录均保留。{method_note}</p>{joint_link}
{''.join(sections)}
<section><h2>实际用时与花费</h2><p>收录 {len(ledger)} 次临时双 A100 云调用（含失败和历史对照）。最新几何 / 测量函数执行：{latest_timing}；已知函数时间合计 {duration:.2f} 秒，按预留资源费率估算 ${cost:.3f}。{missing} 次缺少完整函数记录。所有调用窗口费用估算 ${call_cost:.3f}（含调度，非账单）。</p><p>{timing_note}此前四照片完整报告为 334.60 秒。完整 52 对象场景仍使用此前模型；本轮候选只在本页独立预览，尚未替换完整场景。</p><p>没有完整传播分割、相机和地面误差；源观测门槛不等于物理测量置信区间。</p><details><summary>各次调用、阶段失败及联合求解原因</summary><pre>{esc(json.dumps(execution,ensure_ascii=False,indent=2))}</pre></details><p>每次调用、失败记录、源码哈希和估算依据见 <a href="metrology-evaluation.json">运行账目</a>。</p></section></main></html>'''
    (page / 'metrology.html').write_text(markup)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True, action='append')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--measurements', type=Path, required=True)
    parser.add_argument('--depth-run', type=Path, action='append', default=[], help='Paired neural-depth run directory; repeat chronologically, latest is displayed')
    args = parser.parse_args()
    print(json.dumps(build(args.baseline, args.run, args.out, args.measurements, depth_runs=args.depth_run), ensure_ascii=False))
