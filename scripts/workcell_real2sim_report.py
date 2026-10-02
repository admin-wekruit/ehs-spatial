"""Read recorded real2sim experiments and package a separate Chinese report.

No inference or geometry fitting. Run --check for a tiny schema/failure check.
"""
import argparse
from collections import Counter
import hashlib
import html
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
from scripts.workcell_metrology_report import _preview_markup


LABELS = {'textures': '护板照片纹理', 'post-faces': '光幕可见面', 'camera-control': '原姿态相机对照',
          'independent-poses': '独立姿态重建', 'old-selection': '原轨迹选择',
          'balanced-selection': '连通性平衡轨迹', 'independent-button': '独立姿态 + 按钮'}
OBJECTS = {'v-guard-left': '左护板', 'v-guard-right': '右护板', 'post-box-1': '右光幕（按钮侧）', 'post-box-2': '左光幕'}
STATUS = {'completed': '执行完成', 'failed': '执行失败', 'partial': '部分完成', 'available': '条件支持',
          'unsupported': '未获支持', 'by_construction': '源视图构造自洽',
          'not_supported': '未获跨图支持', 'consistent_with_detected_end_pair': '与检测端边一致'}


def _esc(value):
    return html.escape(str(value), quote=True)


def _number(value, digits=3):
    if value is None:
        return '未提供'
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('Report numbers must be finite recorded values')
    return f'{value:.{digits}f}'


def _required(data, fields, label):
    if not isinstance(data, dict):
        raise ValueError(f'{label}: expected an object')
    missing = [field for field in fields if field not in data]
    if missing:
        raise ValueError(f'{label}: missing required fields: {", ".join(missing)}')
    return data


def _read(path, fields):
    return _required(json.loads(path.read_text()), fields, str(path))


def _table(headings, rows):
    return '<div class="scroll"><table><thead><tr>' + ''.join('<th>' + _esc(v) + '</th>' for v in headings) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join('<td>' + _esc(v) + '</td>' for v in row) + '</tr>' for row in rows) + '</tbody></table></div>'


def _edge_diagnostics(data, sources):
    """Draw saved detector segments only; never rerun detection or fit geometry."""
    import cv2
    import numpy as np
    if len(sources) != 4:
        raise ValueError('Diagnostic overlays need the four original JPEGs')
    _required(data, ('sources', 'sourceEdgeDiagnostics'), 'face source diagnostics')
    metadata = {row['photo']: row for row in data['sources']}
    objects = [row for row in data['sourceEdgeDiagnostics'] if row['id'].startswith('post-box-')]
    images, rows = [], []
    for photo, source in enumerate(map(Path, sources), 1):
        if hashlib.sha256(source.read_bytes()).hexdigest() != metadata[photo]['sha256']:
            raise ValueError(f'Photo {photo} differs from the recorded face-detector source')
        rgb = cv2.imread(str(source))
        if rgb is None or list(rgb.shape[:2]) != metadata[photo]['rawShape']:
            raise ValueError('Diagnostic source dimensions disagree with the recorded camera input')
        tiles = []
        for obj in objects:
            name = 'RIGHT' if obj['id'] == 'post-box-1' else 'LEFT'
            found = {kind: [edge for edge in obj[kind + 'Candidates'] if edge['photo'] == photo] for kind in ('bottom', 'top')}
            ids = {kind: sorted({tuple(sorted(edge['faceSideIds'])) for edge in edges}) for kind, edges in found.items()}
            rows.append([photo, OBJECTS[obj['id']], len(found['bottom']), len(found['top']),
                         '；'.join('/'.join(map(str, pair)) for pair in ids['bottom']) or '无',
                         '；'.join('/'.join(map(str, pair)) for pair in ids['top']) or '无',
                         len(set(ids['bottom']) & set(ids['top']))])
            view = next((view for view in obj['views'] if view['photo'] == photo), {})
            for kind, edges in found.items():
                rejected = view.get(kind, {}).get('nearEndRejectedEdges', []) if not edges else []
                segments = [segment for edge in edges for segment in edge.get('rawSegments', [edge['rawEnds']])]
                crop_edges = segments or [edge['rawEnds'] for edge in rejected]
                if crop_edges:
                    points = np.asarray(crop_edges, float).reshape(-1, 2)
                    lo = np.maximum(np.floor(points.min(0) - 80).astype(int), 0)
                    hi = np.minimum(np.ceil(points.max(0) + 80).astype(int), rgb.shape[1::-1])
                else:
                    lo, hi = np.array([0, 0]), np.array(rgb.shape[1::-1])
                crop = rgb[lo[1]:hi[1], lo[0]:hi[0]].copy()
                color = (230, 220, 0) if kind == 'bottom' else (200, 55, 255)
                for index, edge in enumerate(edges):
                    for segment in edge.get('rawSegments', [edge['rawEnds']]):
                        p = np.rint(np.asarray(segment) - lo).astype(np.int32)
                        cv2.polylines(crop, [p], False, color, 5 if index == 0 else 2)
                    for segment in edge['faceSideEdgesRaw']:
                        cv2.polylines(crop, [np.rint(np.asarray(segment) - lo).astype(np.int32)], False, color, 1)
                for edge in rejected:
                    cv2.polylines(crop, [np.rint(np.asarray(edge['rawEnds']) - lo).astype(np.int32)], False, (145, 145, 145), 2)
                factor = min(480 / crop.shape[1], 260 / crop.shape[0])
                crop = cv2.resize(crop, None, fx=factor, fy=factor)
                tile = np.full((320, 480, 3), 25, np.uint8)
                x = (480 - crop.shape[1]) // 2
                tile[58:58 + crop.shape[0], x:x + crop.shape[1]] = crop
                best = '/'.join(map(str, edges[0]['faceSideIds'])) if edges else 'none'
                cv2.putText(tile, f'Photo {photo} | {name} {kind} | {len(edges)} candidates', (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .48, (255, 255, 255), 1)
                cv2.putText(tile, f'Best side IDs: {best}' + (' | gray: rejected lines' if not edges else ''), (8, 45), cv2.FONT_HERSHEY_SIMPLEX, .44, color, 1)
                tiles.append(tile)
        if len(tiles) != 4:
            raise ValueError('Source diagnostics must retain both light-curtain objects')
        images.append((f'photo-{photo}-detector.jpg', np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])) )
    return images, rows


def build(baseline, run, out, main_report='../index.html', retry_run=None, sources=None):
    """Package actual run JSON, selected models/overlays and the existing viewer."""
    baseline, run, out = map(Path, (baseline, run, out))
    retry = Path(retry_run) if retry_run else None
    if any(out.resolve() == path.resolve() or path.resolve() in out.resolve().parents for path in (baseline, run, *([retry] if retry else []))):
        raise ValueError('Report must be in a separate output directory')
    if out.exists():
        raise ValueError('Report output must be new')
    if main_report.startswith(('/', '//')) or ':' in main_report:
        raise ValueError('Main report link must be relative')
    result = _read(run / 'results.json', ('status', 'records', 'wallSeconds', 'scope', 'groundTruthUsedForFitting'))
    _number(result['wallSeconds'])
    runs = [(run, result, '', '初次')]
    if retry:
        retry_result = _read(retry / 'results.json', ('status', 'records', 'wallSeconds', 'scope', 'groundTruthUsedForFitting'))
        _number(retry_result['wallSeconds'])
        runs.append((retry, retry_result, 'retry', '重跑'))
    copies = {}

    def asset(root, relative, destination=None):
        relative = Path(relative)
        source = (root / relative).resolve()
        if relative.is_absolute() or root.resolve() not in source.parents or not source.is_file():
            raise ValueError(f'Missing or out-of-scope recorded asset: {root / relative}')
        target = Path('real2sim-data') / (destination if destination else relative)
        if target.is_absolute() or '..' in target.parts:
            raise ValueError('Asset destination must stay inside the report')
        if target in copies and copies[target] != source:
            raise ValueError('Two recorded assets map to the same report path')
        copies[target] = source
        return target.as_posix()

    # Retain the recorded diagnostics, excluding feature DBs and original rasters.
    provenance = {'runs': [], 'selectedStages': {},
        'timing': 'Original and retry runs remain separate; neither is a fresh photo-to-model latency.'}
    selected, stage_rows, cost_rows, platform_notes = {}, [], [], []
    for source, record, prefix, label in runs:
        if not isinstance(record['records'], dict) or not record['records']:
            raise ValueError('Experiment must retain its stage execution records')
        for path in sorted(source.rglob('*.json')):
            asset(source, path.relative_to(source), Path(prefix) / path.relative_to(source))
        provenance['runs'].append({'directory': source.parent.name + '/' + source.name,
            'result': (Path('real2sim-data') / prefix / 'results.json').as_posix(),
            'sha256': hashlib.sha256((source / 'results.json').read_bytes()).hexdigest(),
            'status': record['status'], 'wallSeconds': record['wallSeconds']})
        ledger = source.parent / 'spend-ledger.json'
        if ledger.is_file():
            cost = _read(ledger, ('functionSeconds', 'callSeconds', 'estimateUsd', 'callWindowEstimateUsd', 'actualBilledUsd'))
            link = asset(source.parent, ledger.name, Path(prefix) / ledger.name)
            provenance['runs'][-1]['costLedger'] = link
            cost_rows.append([label, _number(cost['functionSeconds'], 2), _number(cost['callSeconds'], 2),
                              '$' + _number(cost['estimateUsd']), '$' + _number(cost['callWindowEstimateUsd']),
                              '未知' if cost['actualBilledUsd'] is None else '$' + _number(cost['actualBilledUsd'])])
        for name in ('input-manifest.json', 'implementation-manifest.json', 'platform-events.json'):
            if (source.parent / name).is_file():
                link = asset(source.parent, name, Path(prefix) / name)
                if name == 'platform-events.json' and any(event.get('type') == 'container_preemption'
                        for event in json.loads((source.parent / name).read_text()).get('events', [])):
                    platform_notes.append(f'<p>{_esc(label)}发生平台抢占重启。函数秒与函数估算只包含返回的尝试，不含被回收的尝试；调用窗口包含等待，实际账单未知。<a href="{_esc(link)}">平台事件记录</a></p>')
        for name, stage in record['records'].items():
            _required(stage, ('status', 'seconds'), f'{label} stage {name}')
            if stage['status'] not in ('completed', 'failed'):
                raise ValueError(f'Unknown stage execution status: {name}')
            if stage['status'] == 'failed':
                _required(stage, ('error',), name)
            else:
                selected[name] = (source, prefix)
                provenance['selectedStages'][name] = {'run': len(provenance['runs']) - 1,
                    'selection': 'latest recorded completed execution; physical validation remains separate'}
            stage_rows.append([label, LABELS.get(name, name), STATUS[stage['status']], _number(stage['seconds'], 2), stage.get('error', '')])
    retry_html = ''
    if retry:
        retry_html = f'<p>后续重跑状态：{_esc(STATUS.get(retry_result["status"], retry_result["status"]))}；独立墙钟 {_number(retry_result["wallSeconds"], 2)} 秒。按阶段选择后续成功导出的结果，初次失败记录保留；两次耗时没有拼成一次完整流程。<a href="real2sim-data/retry/results.json">重跑结果</a> · <a href="real2sim-data/report-sources.json">报告来源与哈希</a></p>'

    def completed(name):
        return name in selected

    def stage_data(name, filename, fields):
        return _read(selected[name][0] / name / filename, fields)

    def stage_asset(name, relative):
        source, prefix = selected[name]
        return asset(source, Path(name) / relative, Path(prefix) / name / relative)

    choices, appearance_rows, candidate_sections, camera_sections, button_rows, holdout_rows = [], [], [], [], [], []
    floor = None
    if completed('textures') or completed('post-faces'):
        floor = _read(baseline / 'geometry.json', ('floor',))['floor']
        _required(floor, ('normal', 'offset'), 'baseline floor')
    if completed('textures'):
        data = stage_data('textures', 'texture-results.json', ('geometryUnchanged', 'objects', 'newModelCalls', 'geometryOptimizationCalls', 'wallSeconds'))
        if data['geometryUnchanged'] is not True or {row['id'] for row in data['objects']} != {'v-guard-left', 'v-guard-right'}:
            raise ValueError('Texture report requires recorded unchanged left/right guard geometry')
        for obj in data['objects']:
            _required(obj, ('id', 'mesh', 'panels', 'geometryUnchanged', 'sourceMeshSha256', 'texturedMeshSha256'), 'textured guard')
            if obj['geometryUnchanged'] is not True:
                raise ValueError('Texture stage did not preserve guard geometry')
            before = asset(baseline, obj['mesh'], Path('before') / obj['mesh'])
            after = stage_asset('textures', obj['mesh'])
            for source, digest in ((copies[Path(before)], obj['sourceMeshSha256']), (copies[Path(after)], obj['texturedMeshSha256'])):
                if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
                    raise ValueError(f'Guard model does not match the recorded texture experiment: {source.name}')
            for label, model in (('纹理前', before), ('照片纹理后', after)):
                choices.append({'label': f"{OBJECTS[obj['id']]} · {label}", 'world': 'baseline native', 'kind': 'guard-appearance',
                    'assets': [model], 'floor': floor,
                    'note': '前后使用相同顶点、面和放置；照片纹理不能验证或修正几何。不可见区域保留原外观，反面未独立重建。'})
            for panel in obj['panels']:
                _required(panel, ('node', 'texture', 'photoTexelFraction', 'sourcePhotos', 'geometryUnchanged'), 'guard panel')
                stage_asset('textures', panel['texture'])
                appearance_rows.append([OBJECTS[obj['id']], panel['node'], '是' if panel['geometryUnchanged'] else '否',
                    _number(100 * panel['photoTexelFraction'], 1) + '%', '、'.join(map(str, panel['sourcePhotos'])) or '无'])

    rejection_rows, diagnostic_images, diagnostic_html = [], [], ''
    if completed('post-faces'):
        data = stage_data('post-faces', 'post-faces.json', ('units', 'mPerNative', 'previewOnly', 'candidates', 'rejections', 'wallSeconds'))
        if data['units'] != 'native' or data['mPerNative'] is not None or data['previewOnly'] is not True:
            raise ValueError('Visible-face candidates must remain separate native-unit previews')
        if sources:
            diagnostic_images, diagnostic_rows = _edge_diagnostics(data, sources)
            provenance['detectorOverlays'] = {'sourceSha256': [row['sha256'] for row in data['sources']],
                'method': 'Draw recorded rawSegments and locally observed side fragments; no new inference or geometry.'}
            figures = ''.join(f'<figure><a href="real2sim-data/detector-overlays/{name}"><img loading="lazy" src="real2sim-data/detector-overlays/{name}" alt="照片 {index} 的左右光幕上下端边检测与侧线编号"></a><figcaption>照片 {index} · 点击查看裁切原图</figcaption></figure>' for index, (name, _) in enumerate(diagnostic_images, 1))
            diagnostic_html = '<h3>检测器实际看到了哪些边</h3><p>青色：底边；粉色：顶边。细线为该端边绑定的可见侧线片段；空检测区仅显示已记录的灰色拒绝线。侧线 ID 来自同一张图的 LSD 线段组，比较时忽略左右顺序；它不是上下方向的序号。只绘制已观测的片段，不跨越未见间隙。</p><div class="cards">' + figures + '</div>' + _table(['照片', '对象', '底边候选', '顶边候选', '底边侧线组 ID', '顶边侧线组 ID', '共同组数'], diagnostic_rows)
        rejection_counts = Counter(row['reason'] for row in data['rejections'])
        if not data['candidates']:
            diagnostic_html = f'<p>本次导出 0 个开放面。{rejection_counts.get("Top and bottom do not have the same observed side-line pair", 0)} 个上下端边组合因侧线组不同而未配对，{rejection_counts.get("No detected bottom and top edges in the same source view", 0)} 个对象视图没有同时检测到上下端边。配对失败发生在平面拟合之前；它说明当前检测与配对步骤没有建立同一可见面的边界，不能据此断言这些输入照片无法重建。</p>' + diagnostic_html
        for row in data['rejections']:
            _required(row, ('objectId', 'sourcePhoto', 'reason'), 'face rejection')
            rejection_rows.append([OBJECTS.get(row['objectId'], row['objectId']), row['sourcePhoto'], row['reason']])
        for candidate in data['candidates']:
            _required(candidate, ('id', 'objectId', 'model', 'sourcePhoto', 'views', 'plane', 'endpointEstimate',
                                  'mPerNative', 'thicknessNative', 'sourceReprojectionMaxRawPx'), 'face candidate')
            if candidate['mPerNative'] is not None or candidate['thicknessNative'] is not None:
                raise ValueError('Open-face preview cannot imply measured scale or thickness')
            if sorted(view['photo'] for view in candidate['views']) != [1, 2, 3, 4]:
                raise ValueError('Each face candidate needs four recorded source overlays')
            name = OBJECTS.get(candidate['objectId'], candidate['objectId']) + f" · 照片 {candidate['sourcePhoto']} 可见面"
            model = stage_asset('post-faces', candidate['model'])
            choices.append({'label': name, 'world': 'baseline native', 'kind': 'post-visible-face', 'assets': [model], 'floor': floor,
                'note': '仅为单照片支持的开放面候选；厚度未知。底边与下表端点估计使用同一组坐标，原生单位尚未定米制。'})
            view_rows, figures = [], []
            for view in candidate['views']:
                _required(view, ('photo', 'overlay', 'role', 'status', 'bestSameSidePair', 'thresholdRawPx', 'reason'), 'face view')
                match = view['bestSameSidePair']
                if match is not None:
                    _required(match, ('bottomSymmetricP95RawPx', 'topSymmetricP95RawPx'), 'reprojection pair')
                view_rows.append([view['photo'], '拟合源图' if view['photo'] == candidate['sourcePhoto'] else '独立相机对照',
                    _number(match['bottomSymmetricP95RawPx']) if match else '无同侧线配对',
                    _number(match['topSymmetricP95RawPx']) if match else '无同侧线配对', STATUS.get(view['status'], view['status'])])
                image = stage_asset('post-faces', view['overlay'])
                figures.append(f'<figure><a href="{_esc(image)}"><img loading="lazy" src="{_esc(image)}" alt="{_esc(name)}在照片 {view["photo"]} 的原边、候选面与旧底面对照"></a><figcaption>照片 {view["photo"]} · {_esc(STATUS.get(view["status"], view["status"]))}</figcaption></figure>')
            plane, endpoint = candidate['plane'], candidate['endpointEstimate']
            _required(plane, ('supportPoints', 'residualP95Native', 'normalToMinorSingularRatio'), 'source plane')
            _required(endpoint, ('heightNative', 'physicalValidation'), 'source endpoint')
            candidate_sections.append(f'''<article><h3>{_esc(name)}</h3><p>平面支撑 {_esc(plane['supportPoints'])} 点；点到平面 P95 {_number(plane['residualP95Native'])} 原生单位；法向/次轴奇异值比 {_number(plane['normalToMinorSingularRatio'])}。</p>
<p>同一 mesh 底边端点到推断地面的条件估计：{_number(endpoint['heightNative'])} 原生单位。单视图平面、地面和米制均未完成物理验证。</p>
{_table(['照片', '用途', '底边 P95 原图 px', '顶边 P95 原图 px', '记录状态'], view_rows)}
<p class="muted">源图四角重投影最大值 {_number(candidate['sourceReprojectionMaxRawPx'], 6)} px 是构造结果；上表使用有限端边的双向距离，不是厘米误差。</p><div class="cards">{''.join(figures)}</div><p><a href="{_esc(model)}">下载该开放面 GLB</a></p></article>''')

    if completed('camera-control'):
        data = stage_data('camera-control', 'results.json', ('beforeBa', 'afterBa', 'registeredImages', 'cameraModel', 'squareRawPixelsAssumed', 'seconds'))
        for label in ('beforeBa', 'afterBa'):
            _required(data[label], ('points', 'meanErrorPx'), label)
        camera_sections.append(f'''<h3>原姿态初始化的相机对照</h3><p>相机模型 {_esc(data['cameraModel'])}；登记 {_esc(data['registeredImages'])} 张照片；原图方形像素假设：{'是' if data['squareRawPixelsAssumed'] else '否'}。</p>''' +
            _table(['阶段', '3D 点数', '平均重投影误差（处理图 px）'], [
                ['束调整前', data['beforeBa']['points'], _number(data['beforeBa']['meanErrorPx'])],
                ['束调整后', data['afterBa']['points'], _number(data['afterBa']['meanErrorPx'])]]) +
            '<p class="muted">这些点参与相机拟合；较低训练误差不能替代独立按钮留出检验。像素基准与下方原图 px 不同。</p>')
        cameras = stage_data('camera-control', 'cameras.json', ('frames', 'worldFrame'))
        camera_rows = []
        for frame in cameras['frames']:
            _required(frame, ('photo', 'K', 'pose'), 'camera frame')
            camera_rows.append([frame['photo'], _number(frame['K'][0][0]), _number(frame['K'][1][1]),
                                _number(frame['K'][0][2]), _number(frame['K'][1][2])])
        camera_sections.append('<details><summary>实际相机内参（canonical 图像 px）</summary>' +
            _table(['照片', 'fx', 'fy', 'cx', 'cy'], camera_rows) + '</details>')
    if completed('independent-poses'):
        data = stage_data('independent-poses', 'results.json', ('status', 'reason', 'seconds'))
        camera_sections.append(f'<h3>独立姿态重建</h3><p>{_esc(STATUS.get(data["status"], data["status"]))}；{_number(data["seconds"], 2)} 秒。{_esc(data["reason"] or "")}</p>')
        if data.get('models'):
            camera_sections.append(_table(['模型', '登记照片', '3D 点数', '重投影均值（处理图 px）'], [[
                model['model'], '、'.join(model['registeredImages']), model['points'],
                _number(model['meanReprojectionErrorControlPx'])] for model in data['models']]))
    for name in ('old-selection', 'balanced-selection', 'independent-button'):
        if not completed(name):
            continue
        data = stage_data(name, 'joint-reference.json', ('status', 'reason', 'mPerNative', 'heldOutPhotos', 'diagnostics', 'wallSeconds'))
        if data['status'] != 'available' and data['mPerNative'] is not None:
            raise ValueError('Unsupported button calibration cannot publish metric scale')
        diagnostic = data['diagnostics']
        selection = diagnostic.get('trackSelection', {})
        failed = [key for key, passed in diagnostic.get('checks', {}).items() if not passed]
        button_rows.append([LABELS[name], STATUS.get(data['status'], data['status']),
            _number(diagnostic.get('sceneReprojectionRmsRawPx')), _number(diagnostic.get('sceneReprojectionP95RawPx')),
            '；'.join(failed) or data['reason'] or '记录门槛通过'])
        camera_sections.append(f'<details><summary>{_esc(LABELS[name])}的轨迹选择记录</summary><pre>{_esc(json.dumps(selection, ensure_ascii=False, indent=2))}</pre></details>')
        for holdout in data['heldOutPhotos']:
            _required(holdout, ('photo', 'fitButtonPhotos', 'cameraRetainedSceneTracks', 'maxErrorRawPx', 'toleranceRawPx', 'status'), 'whole-button holdout')
            holdout_rows.append([LABELS[name], holdout['photo'], '、'.join(map(str, holdout['fitButtonPhotos'])),
                holdout['cameraRetainedSceneTracks'], _number(holdout['maxErrorRawPx']), _number(holdout['toleranceRawPx']),
                STATUS.get(holdout['status'], holdout['status'])])

    if choices:
        viewer = baseline / 'page/viewer-assets'
        for name in ('three.module.js', 'three.core.js', 'addons/controls/OrbitControls.js',
                     'addons/loaders/GLTFLoader.js', 'addons/utils/BufferGeometryUtils.js'):
            if not (viewer / name).is_file():
                raise ValueError(f'Missing existing viewer dependency: {name}')
    stage_table = _table(['运行', '实验', '执行', '秒', '失败原因'], stage_rows)
    cost_table = ('<h3>分次资源费用记录</h3><p>读取各次运行的成本记录，金额为资源费率估算，非账单；函数、调用窗口与算法墙钟是不同口径。函数窗口不一定包含被抢占的尝试；完整调用窗口包含调度等待。平台中断记录与成本明细均保留在结果文件中。</p>' +
        _table(['运行', '函数秒', '调用窗口秒', '函数估算 USD', '窗口估算 USD', '实际账单 USD'], cost_rows) + ''.join(platform_notes)) if cost_rows else ''
    texture_table = _table(['护板', '片面', '几何不变', '照片纹理覆盖', '来源照片'], appearance_rows) if appearance_rows else '<p>本次没有完成可发布的护板纹理结果。</p>'
    post_html = ''.join(candidate_sections) or ('<p>本次没有通过源面构建条件的光幕候选。下面保留实际拒绝记录。</p>'
        if completed('post-faces') else '<p>光幕可见面阶段未完成或未执行；当前没有可审查候选。</p>')
    rejected = '<details><summary>查看光幕拒绝记录（' + str(len(rejection_rows)) + ' 条）</summary>' + _table(['对象', '源图', '原因'], rejection_rows) + '</details>' if rejection_rows else ''
    button_table = _table(['路线', '状态', '场景 RMS 原图 px', '场景 P95 原图 px', '未通过检查'], button_rows) if button_rows else '<p>未完成按钮联合拟合；请查看执行记录。</p>'
    holdout_table = _table(['路线', '留出照片', '按钮拟合照片', '该图保留场景轨迹', '最大误差 原图 px', '阈值 原图 px', '状态'], holdout_rows) if holdout_rows else '<p>没有可用的完整按钮留出结果。</p>'
    page = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>细节与几何实验 · Panoptes</title>
<style>body{{font:16px/1.65 system-ui,sans-serif;background:#f5f6f2;color:#182824;margin:0}}main{{max-width:1100px;margin:auto;padding:32px 20px}}h1{{font-size:32px;line-height:1.25}}h2{{margin:36px 0 12px}}a{{color:#14675b}}section,article{{background:white;border:1px solid #d6ddd5;border-radius:12px;padding:20px;margin:18px 0}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px}}figure{{margin:0}}img{{width:100%;border-radius:6px}}.muted{{color:#53645f}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:9px;text-align:left;border-bottom:1px solid #d6ddd5;vertical-align:top}}.scroll{{overflow:auto}}select{{font:inherit;padding:8px;max-width:100%}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}}summary{{cursor:pointer}}canvas{{border-radius:8px}}@media(max-width:600px){{main{{padding:20px 12px}}section,article{{padding:14px}}h1{{font-size:26px}}}}</style></head><body><main>
<p><a href="{_esc(main_report)}">← 返回完整 52 对象主报告</a> · <a href="real2sim-data/results.json">本次完整结果 JSON</a></p>
<h1>细节与几何实验</h1><p>初次运行状态：{_esc(STATUS.get(result['status'], result['status']))}。初次墙钟时间 {_number(result['wallSeconds'], 2)} 秒，复用冻结输入；不是从四张照片开始的完整耗时。</p>{retry_html}
<p>下方显示实际导出的模型与观测记录。源图自洽、训练误差与独立留出分开报告；本页不会把候选自动写回主场景。拟合使用离地评估真值：{'是' if result['groundTruthUsedForFitting'] else '否'}。</p>
{_preview_markup(choices)}
<section><h2>1. 护板：照片纹理前后</h2><p>上方选择器可切换同一护板的纹理前后实际 GLB。覆盖率是纹理像素来源比例，不是形状准确率；未覆盖处和反面仍有未知。</p><p><strong>尚未确认整体外观更好；本轮模型仅作对照，未接入主流程。</strong></p>{texture_table}</section>
<section><h2>2. 光幕：原图可见面候选</h2><p>开放面保留未知厚度和未接受的米制。若导出候选，其逐图叠图用青色显示候选面、粉色显示 RGB 端边、橙色显示旧模型底面。</p>{post_html}{diagnostic_html}{rejected}</section>
<section><h2>3. 相机与按钮检验</h2>{''.join(camera_sections)}{button_table}<h3>整颗按钮留一照片验证</h3>
<p>每次排除该照片全部按钮观测；该图非按钮场景轨迹仍参与相机拟合。这里只报告真实留出误差，失败形变按钮没有进入本页或主模型。</p>{holdout_table}</section>
<section><h2>4. 执行记录</h2><p>阶段可并行，阶段秒数不能相加为总耗时。</p>{stage_table}{cost_table}<p class="muted">数据范围：{_esc(result['scope'])}</p></section>
</main></body></html>'''
    # Validate all required data/assets before creating the report output.
    out.mkdir(parents=True)
    for destination, source in copies.items():
        target = out / destination; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    (out / 'real2sim-data/report-sources.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    if diagnostic_images:
        import cv2
        destination = out / 'real2sim-data/detector-overlays'; destination.mkdir()
        for name, image in diagnostic_images:
            if not cv2.imwrite(str(destination / name), image, [cv2.IMWRITE_JPEG_QUALITY, 90]):
                raise OSError('Failed to save source detector overlay')
    if choices:
        shutil.copytree(viewer, out / 'viewer-assets')
    (out / 'real2sim.html').write_text(page)
    return out / 'real2sim.html'


def check():
    with tempfile.TemporaryDirectory(prefix='workcell-report-check-') as temporary:
        root = Path(temporary); run = root / 'run'; run.mkdir()
        path = run / 'results.json'
        record = {'status': 'partial', 'records': {'textures': {'status': 'failed', 'seconds': .2, 'error': '<source unavailable>'}},
                  'scope': 'synthetic schema check', 'groundTruthUsedForFitting': False, 'wallSeconds': .3}
        path.write_text(json.dumps({key: value for key, value in record.items() if key != 'wallSeconds'}))
        try:
            build(root / 'baseline', run, root / 'missing')
        except ValueError as error:
            assert 'wallSeconds' in str(error)
        else:
            raise AssertionError('Generator accepted missing runtime evidence')
        assert not (root / 'missing').exists()
        path.write_text(json.dumps(record))
        report = build(root / 'baseline', run, root / 'report')
        body = report.read_text()
        assert '&lt;source unavailable&gt;' in body and '<source unavailable>' not in body
        assert '0.30 秒' in body and '光幕可见面阶段未完成或未执行' in body
        assert json.loads((report.parent / 'real2sim-data/results.json').read_text()) == record
        retry = root / 'retry'; (retry / 'post-faces').mkdir(parents=True)
        retry_record = {**record, 'status': 'completed', 'wallSeconds': .4,
                        'records': {'post-faces': {'status': 'completed', 'seconds': .1}}}
        (retry / 'results.json').write_text(json.dumps(retry_record))
        (retry / 'post-faces/post-faces.json').write_text(json.dumps({'units': 'native', 'mPerNative': None,
            'previewOnly': True, 'candidates': [], 'rejections': [], 'wallSeconds': .1}))
        (root / 'baseline').mkdir()
        (root / 'baseline/geometry.json').write_text(json.dumps({'floor': {'normal': [0, 1, 0], 'offset': 0}}))
        report = build(root / 'baseline', run, root / 'merged-report', retry_run=retry)
        provenance = json.loads((report.parent / 'real2sim-data/report-sources.json').read_text())
        assert provenance['selectedStages']['post-faces']['run'] == 1
        assert [row['wallSeconds'] for row in provenance['runs']] == [.3, .4]
        assert json.loads((report.parent / 'real2sim-data/results.json').read_text()) == record
        assert json.loads((report.parent / 'real2sim-data/retry/results.json').read_text()) == retry_record
        assert '0.30 秒' in report.read_text() and '0.40 秒' in report.read_text()
        record['records']['textures'] = {'status': 'completed', 'seconds': .2}
        path.write_text(json.dumps(record))
        try:
            build(root / 'baseline', run, root / 'unrecorded')
        except (ValueError, FileNotFoundError):
            pass
        else:
            raise AssertionError('Completed texture stage was accepted without its actual assets/results')
        assert not (root / 'unrecorded').exists()
    print('PASS: missing evidence fails before output; recorded failures render safely; JSON preserved; absent assets cannot become successful models')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    for name in ('baseline', 'run', 'out'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--main-report', default='../index.html')
    parser.add_argument('--retry-run', type=Path, help='Later run with the same results.json execution schema')
    parser.add_argument('--sources', type=Path, nargs=4, help='Four exact original JPEGs for saved-detector overlays')
    args = parser.parse_args()
    if args.check:
        check()
    else:
        if not all((args.baseline, args.run, args.out)):
            parser.error('--baseline, --run and --out are required unless --check is used')
        print(build(args.baseline, args.run, args.out, args.main_report, args.retry_run, args.sources))
