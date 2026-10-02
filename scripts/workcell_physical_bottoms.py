"""Reproduce source-bottom fits on a frozen workcell; no evaluation sizes enter.

The known left/right equality is an explicit model constraint. The independent
fit and source projections remain visible; equality is never an accuracy score.
"""
import hashlib
import html
import json
from pathlib import Path
import shutil
import time

import cv2
import numpy as np
import trimesh

from workcell_endpoint_estimate import _bottom_face
from workcell_photo_metrology import _fence_plane_index, _legacy, _load, _object_edges, _pixels, TARGETS
from workcell_photo_objects import _project


def _ordered_bottom(vertices, normal, offset):
    vertices = np.asarray(vertices, float)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 4 or not np.isfinite(vertices).all():
        raise ValueError('Existing lower faces need at least four finite native points')
    center = vertices.mean(0)
    _, singular, vectors = np.linalg.svd(vertices - center, full_matrices=False)
    if singular[1] < max(singular[0], 1.) * 1e-8:
        raise ValueError('Existing lower faces do not constrain a two-dimensional plane')
    source_normal = vectors[-1]
    source_normal *= 1 if source_normal @ normal >= 0 else -1
    residual = (vertices - center) @ source_normal
    tolerance = 1e-6 * max(1., float(np.linalg.norm(np.ptp(vertices, axis=0))))
    if np.ptp(residual) > tolerance:
        raise ValueError('Existing lower faces are not coplanar; one rigid initializer cannot align them')
    # ponytail: one common rigid rotation repairs a prior floor-frame change.
    # Noncoplanar or stepped lower faces require distinct observed part models.
    transform = trimesh.geometry.align_vectors(source_normal, normal)
    transform[:3, 3] = center - transform[:3, :3] @ center
    initialized = trimesh.transform_points(vertices, transform)
    height = float(initialized.mean(0) @ normal + offset)
    initializer = {'worldTransformNative': transform.tolist(), 'pivotNative': center.tolist(),
                   'sourceNormalNative': source_normal.tolist(), 'groundNormalNative': normal.tolist(),
                   'angleDegrees': float(np.degrees(np.arccos(np.clip(source_normal @ normal, -1., 1.)))),
                   'maxCoplanarResidualNative': float(np.max(abs(residual))),
                   'sourceBottomVerticesNative': vertices.tolist(), 'initializedBottomVerticesNative': initialized.tolist(),
                   'heightSpreadBeforeNative': float(np.ptp(vertices @ normal + offset)),
                   'heightSpreadAfterNative': float(np.ptp(initialized @ normal + offset)),
                   'assumption': 'Rigidly rotate all nodes of this object/section about the observed model lower-plane center to make that plane parallel to the shared inferred ground; not new RGB evidence or physical validation.'}
    vertices = initialized
    a = vertices[0] - center
    a /= np.linalg.norm(a)
    b = np.cross(normal, a)
    order = np.argsort(np.arctan2((vertices - center) @ b, (vertices - center) @ a))
    return vertices[order], height, initializer


def _items(root, geometry, catalog, diagnostics):
    ground = geometry['floor']
    normal = np.asarray(ground['normal'], float)
    norm = np.linalg.norm(normal)
    normal, offset = normal / norm, float(ground['offset']) / norm
    posts = trimesh.load(root / 'posts.glb', force='scene', process=False)
    fences = trimesh.load(root / 'fence-fitted.glb', force='scene', process=False)
    result = []
    for diagnostic in diagnostics:
        ident = diagnostic['id']
        if ident.startswith('post-box-'):
            nodes = [ident.removeprefix('post-')]
            bottom = _bottom_face(posts, nodes[0], normal)
            bottom, height, initializer = _ordered_bottom(bottom, normal, offset)
            file = 'posts.glb'
        else:
            plane = _fence_plane_index(catalog[ident])
            members = [r for r in geometry['fence']['continuations']
                       if r['plane'] == plane and 'lower' in r['role']]
            if not members:
                result.append({'id': ident, 'status': 'unsupported',
                               'reason': 'No existing lower-rail footprint for this fence plane',
                               'geometryPlaneIndex': plane, 'observations': diagnostic['bottomCandidates']})
                continue
            # The long rectangle describes the existing lower rail footprint,
            # not a newly measured complete rail or hidden physical thickness.
            nodes = [r['id'] for r in members]
            original_rail = next((row.get('meshNode') for row in geometry.get('clearances', [])
                                 if row['id'] == f'fence-plane-{plane}-lower-rail'), None)
            if original_rail:
                nodes = list(dict.fromkeys([*nodes, original_rail]))
            points = np.concatenate([_bottom_face(fences, node, normal) for node in nodes])
            try:
                points, height, initializer = _ordered_bottom(points, normal, offset)
            except ValueError as error:
                result.append({'id': ident, 'status': 'unsupported', 'reason': str(error),
                               'geometryPlaneIndex': plane, 'observations': diagnostic['bottomCandidates']})
                continue
            center = points.mean(0)
            along = np.linalg.svd(points - center, full_matrices=False)[2][0]
            along -= (along @ normal) * normal
            along /= np.linalg.norm(along)
            across = np.cross(normal, along)
            uv = np.c_[(points - center) @ along, (points - center) @ across]
            lo, hi = uv.min(0), uv.max(0)
            bottom = np.array([center + u * along + v * across
                               for u, v in ((lo[0], lo[1]), (hi[0], lo[1]),
                                            (hi[0], hi[1]), (lo[0], hi[1]))])
            file = 'fence-fitted.glb'
        result.append({'id': ident, 'modelFile': file, 'modelNodes': nodes,
                       **({'geometryPlaneIndex': plane} if ident.startswith('fence-') else {}),
                       **({'samePhysicalEdge': True} if ident.startswith('post-box-') else {}),
                       'bottomVerticesNative': bottom.tolist(), 'baselineHeightNative': height,
                       'initializerWorldNative': initializer['worldTransformNative'], 'initializer': initializer,
                       'observations': diagnostic['bottomCandidates']})
    return result


def report(runs, out):
    """Publish the recorded experiments without replacing the scene or its GLBs."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    labels = {'fence-0': '右围栏', 'fence-1': '左围栏',
              'post-box-1': '右光幕', 'post-box-2': '左光幕'}
    sections, ledgers = [], []
    for number, run in enumerate(map(Path, runs), 1):
        ledger = json.loads((run / 'spend-ledger.json').read_text())
        ledgers.append(ledger)
        target = out / f'run-{number}'
        if (run / 'physical-bottoms').is_dir():
            shutil.copytree(run / 'physical-bottoms', target)
        else:
            target.mkdir()
        for name in ('spend-ledger.json', 'implementation-manifest.json', 'input-manifest.json', 'run.json'):
            if (run / name).is_file():
                shutil.copy2(run / name, target / name)
        if not (target / 'bottom-fits.json').is_file():
            body = f'<h2>第 {number} 轮 · {html.escape(run.name)}</h2><p>运行失败，没有可用模型或完整拟合结果。函数占用 {ledger["functionSeconds"]:.2f} s，资源价格估算 ${ledger["estimateUsd"]:.3f}；失败也计入支出。</p><p><a href="run-{number}/run.json">失败记录</a> · <a href="run-{number}/spend-ledger.json">支出</a></p>'
            sections.append(body if number == len(runs) else f'<details><summary>查看第 {number} 轮失败记录</summary>{body}</details>')
            continue
        result = json.loads((target / 'bottom-fits.json').read_text())
        counts = []
        for item in result['items']:
            photos = sorted({row['photo'] for row in item['observations']})
            counts.append(f"<li>{labels[item['id']]}：{len(item['observations'])} 条候选；来源照片 {html.escape(str(photos))}。候选不等于跨图对应已确认。</li>")
        fits = []
        for name, fit in result['fits'].items():
            text = fit.get('reason') or f"候选优化 RMS {fit['optimizer']['rmsRawPx']:.2f} 原图像素；未通过物理身份验证"
            models = []
            for file in sorted({row['candidateModelFile'] for row in fit.get('modelPreview', {}).get('items', [])}):
                if (target / name / file).is_file():
                    models.append(f'<a href="run-{number}/{html.escape(name)}/{html.escape(file)}">下载候选 GLB（native 单位）</a>')
            fits.append(f'<li>{html.escape(name)}：{html.escape(text)} {" · ".join(models)}</li>')
        images = []
        for ident, photo in (('post-box-1', 4), ('post-box-2', 4), ('fence-0', 4), ('fence-1', 2)):
            name = f'{ident}-source-{photo}.jpg'
            if (target / name).is_file():
                images.append(f'<figure><a href="run-{number}/{name}"><img src="run-{number}/{name}" alt="{labels[ident]}照片{photo}原图边缘对照" loading="lazy"></a><figcaption>{labels[ident]} · 照片 {photo}</figcaption></figure>')
        body = f'''<h2>第 {number} 轮 · {html.escape(run.name)}</h2>
<p>缓存几何上的原图端边重算：{ledger['functionSeconds']:.2f} s；预留资源价格估算 ${ledger['estimateUsd']:.3f}。不是完整照片→模型的延迟；账单金额未获取。</p>
<div class="pictures">{''.join(images)}</div><p>橙线：拟合起点的模型底面投影；绿线：本轮原图候选。C 轮围栏起点包含与共同地面的刚体对齐。没有绿线表示这一视角没有通过提取检查，不能理解为旧模型已正确。</p>
<ul>{''.join(counts)}</ul><ul>{''.join(fits)}</ul>
<p><a href="run-{number}/bottom-fits.json">拟合与留出结果 JSON</a> · <a href="run-{number}/source-boundaries.json">源边缘及拒绝原因</a> · <a href="run-{number}/spend-ledger.json">本轮支出记录</a></p>'''
        sections.append(body if number == len(runs) else f'<details><summary>查看第 {number} 轮记录</summary>{body}</details>')
    cost = sum(row['estimateUsd'] for row in ledgers)
    page = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Panoptes · 物理下沿修正</title><style>body{{font:16px/1.65 system-ui,sans-serif;background:#f5f6f2;color:#182824;margin:0}}main{{max-width:1100px;margin:auto;padding:28px 20px}}h1{{font-size:30px;line-height:1.25}}h2{{font-size:22px}}a{{color:#14675b}}section,details{{background:white;border:1px solid #d6ddd5;border-radius:12px;padding:20px;margin:18px 0}}.pictures{{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:18px}}figure{{margin:0}}img{{max-width:100%;max-height:460px;object-fit:contain;background:#f5f6f2}}summary{{cursor:pointer}}@media(max-width:600px){{main{{padding:16px 12px}}section,details{{padding:14px}}}}</style></head><body><main>
<a href="../?view=model&measurement=endpoints#scene">← 返回完整 3D、照片和卡尺</a>
<h1>物理下沿：已修提取错误，主模型替换尚未通过</h1>
<section><p>之前完成了共同地面和模型卡尺，但光幕网格仍取单图深度分位数，左右围栏的下沿来源也不等价。因此此前不能宣称已经量到同一条物理下沿。</p>
<p>本轮修正：分开围栏的原图边界与深度过滤轮廓；同时处理左右围栏；光幕端边必须直接连接到壳体，排除线缆和支架；前面板、侧翼仍须跨图确认身份。</p>
<p><strong>本页实验没有替换默认场景模型，也没有证明左右高度已准确。</strong> 相等约束只作为对照；20/24 cm 检查值未参与拟合，当前米制标尺仍未通过联合验证。</p>
<p>三维写回助手会重新加载导出的 GLB 检查实际底面；但数值收敛不能让错误部位获得替换资格。现阶段的阻点是同一部位在多张原图中的稳定对应。</p></section>
<p>A/B 使用此前完整推理中的模型；C 起使用当前公开报告的模型并复用同次推理的相机、SAM 与点图。初始光幕模型不同，不能将轮次差异当作单因素消融。C 触发 600 s 上限；D 将已有的线对拒绝检查提前，避免对明显无效线对继续做立杆拓扑计算。后续同边实验要求光幕各视角共同选择一条模型边，避免每张照片独立换面。<a href="../real2sim/real2sim.html">此前 Real2sim 纹理与相机实验</a></p>
<section>{sections[-1]}</section>{''.join(sections[:-1])}
<p>本页 {len(runs)} 次临时双 A100 实验资源价格合计估算 ${cost:.3f}，非实际账单。使用冻结的相机、点图与当前共同地面；不是新一轮完整 oneshot。</p>
</main></body></html>'''
    (out / 'index.html').write_text(page)
    return {'acceptedModelReplacement': False, 'runs': len(runs), 'estimateUsd': cost}


def _matched_fence_items(items, edges):
    matched = {row['id']: row for row in edges}
    result = []
    for item in items:
        edge = matched[item['id']].get('bottomEdge')
        if not edge:
            raise ValueError(item['id'] + ': no common finite source edge: ' + str(matched[item['id']].get('reason')))
        observations = edge['observations']
        if not any(row.get('identityEvidence', {}).get('independentlySupported') is True for row in observations):
            raise ValueError(item['id'] + ': matched edge has no lower-rail identity anchor')
        result.append({**item, 'observations': observations, 'requireIdentityAnchor': True,
                       'sourceEdgeMatch': edge})
    return result


def build(root, out, sources):
    from workcell_bottom_fit import fit_bottoms, leave_one_photo_out
    from workcell_bottom_models import apply_bottom_models
    root, out = Path(root), Path(out)
    if out.exists():
        raise ValueError('Use a new output directory')
    out.mkdir(parents=True)
    started = time.monotonic()
    geometry, catalog, segmentation, frames, _, source_inputs = _load(root, sources, None)
    print(json.dumps({'phase': 'inputs_loaded', 'seconds': time.monotonic() - started}), flush=True)
    drawings = {photo: {'standard': [], 'selected': [], 'candidates': [], 'floorPixelsRaw': []}
                for photo in frames}
    up = np.asarray(geometry['floor']['normal'], float)
    up /= np.linalg.norm(up)
    legacy = {ident: _legacy(catalog[ident], geometry) for ident in TARGETS}
    edges, diagnostics = _object_edges(catalog, segmentation, geometry, frames, up, legacy, drawings,
                                      diagnostics_path=out / 'candidate-checkpoint.json')
    physical = {'ground': geometry['floor'], 'objects': edges,
                'diagnostics': {'objectEdges': diagnostics},
                'scope': 'Source boundary candidates and line fits; saved shared floor held fixed; no new floor fit or accepted physical dimensions'}
    (out / 'source-boundaries.json').write_text(json.dumps(physical, indent=2, allow_nan=False) + '\n')
    # Keep the saved shared floor fixed in this controlled bottom-edge test.
    # Re-estimating the floor here would confound the boundary correction.
    items = _items(root, geometry, catalog, physical['diagnostics']['objectEdges'])
    result = {'schemaVersion': 1, 'status': 'experiment', 'ground': geometry['floor'],
              'worldFrame': 'MapAnything native', 'units': 'native', 'sourceInputs': source_inputs,
              'evaluationTargetsUsed': False, 'referenceDimensionsUsedForFit': False,
              'baselineModified': False, 'items': items, 'fits': {},
              'method': 'RGB terminal segments fit existing primitive footprints; shared floor; curtain views jointly select one model edge under an explicit same-terminal-edge hypothesis; optional equal-height prior',
              'sourceFiles': {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                              for name in ('posts.glb', 'fence-fitted.glb', 'geometry.json', 'sam3.json')}}
    # Inspect all observed candidates directly, independently of the fit winner.
    for item in items:
        if 'bottomVerticesNative' not in item:
            continue
        for photo in frames:
            rows = [row for row in item['observations'] if row['photo'] == photo]
            image = frames[photo]['rgb'].copy()
            pixels = []
            for i, row in enumerate(rows):
                for segment in row.get('rawSegments', [row['rawEnds']]):
                    edge = np.asarray(segment)
                    pixels.extend(edge)
                    cv2.polylines(image, [np.rint(edge).astype(np.int32)], False, (40, 255, 70), 4)
                cv2.putText(image, str(i), tuple(np.rint(np.mean(row['rawEnds'], 0)).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 30, 180), 2)
            uv, depth = _project(item['bottomVerticesNative'], frames[photo])
            if (depth > 0).all():
                uv = _pixels(uv, np.linalg.inv(frames[photo]['A']))
                pixels.extend(uv)
                cv2.polylines(image, [np.rint(uv).astype(np.int32)], True, (255, 160, 30), 4)
            if not pixels:
                continue
            pixels = np.asarray(pixels)
            lo = np.maximum(np.floor(pixels.min(0) - 100).astype(int), 0)
            hi = np.minimum(np.ceil(pixels.max(0) + 100).astype(int), image.shape[1::-1])
            crop = image[lo[1]:hi[1], lo[0]:hi[0]]
            if crop.size:
                factor = min(1., 1000 / max(crop.shape[:2]))
                crop = cv2.resize(crop, None, fx=factor, fy=factor)
                cv2.imwrite(str(out / f"{item['id']}-source-{photo}.jpg"), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
    for group, prefix in (('curtains', 'post-box-'), ('fences', 'fence-')):
        selected = [item for item in items if item['id'].startswith(prefix)]
        for shared in (False, True):
            name = group + ('-shared' if shared else '-independent')
            print(json.dumps({'phase': name, 'seconds': time.monotonic() - started}), flush=True)
            try:
                expected = {ident for ident in TARGETS if ident.startswith(prefix)}
                if {item['id'] for item in selected} != expected or len(selected) != 2:
                    raise ValueError('Both distinct left and right objects are required')
                if any(item.get('status') == 'unsupported' for item in selected):
                    raise ValueError('; '.join(item['id'] + ': ' + item['reason']
                                              for item in selected if item.get('status') == 'unsupported'))
                if prefix == 'fence-':
                    selected = _matched_fence_items(selected, edges)
                result['fits'][name] = fit_bottoms(selected, frames, geometry['floor'], shared_height=shared)
                result['fits'][name]['inputObservationsByObject'] = {item['id']: item['observations'] for item in selected}
                result['fits'][name]['leaveOnePhotoOut'] = leave_one_photo_out(
                    selected, frames, geometry['floor'], shared_height=shared, max_starts=8)
                fit = result['fits'][name]
                if fit['status'] == 'conditional_fit':
                    try:
                        fit['modelPreview'] = apply_bottom_models(root, fit, selected, out=out / name)
                    except ValueError as error:
                        fit['modelPreview'] = {'status': 'rejected', 'reason': str(error)}
            except (ValueError, np.linalg.LinAlgError) as error:
                result['fits'][name] = {'status': 'unsupported', 'reason': str(error)}
            result['wallSeconds'] = time.monotonic() - started
            (out / 'bottom-fits.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    (out / 'source-boundaries.json').write_text(json.dumps(physical, indent=2, allow_nan=False) + '\n')
    result['wallSeconds'] = time.monotonic() - started
    (out / 'bottom-fits.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result
