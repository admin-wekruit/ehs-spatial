"""Adapt current oneshot artifacts to the existing validated report scene schema."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import trimesh

from ehs_spatial.platform.contracts import empty_document, validate_document, Revision
from scripts.import_report_evidence import canonical_measurements

LABELS = {'robot': '工业机器人', 'cart': '载料运输车', 'floor': '地面参考面',
          'emergency stop button': '红黄急停按钮', 'emergency_button': '红黄急停按钮',
          'yellow safety post': '黄色光幕立柱', 'black bollard': '黑色防撞柱',
          'v guard': 'V 型黑黄护板', 'fence': '安全围栏', 'safety fence': '安全围栏', 'sign': '标识牌', 'signal light': '信号灯', 'stack light': '信号灯',
          'warning sign': '警示牌', 'workcell sign': '工位标识牌', 'folding safety barrier': '折叠防护板',
          'cable tray': '线缆托架', 'instruction poster': '作业指导海报',
          'transparent safety panel': '透明护板', 'floor marking': '地面标线',
          'light curtain': '光幕', 'work platform': '平台/护板可见表面', 'control cabinet': '控制柜'}


def _source_points(frame, transform):
    """Retain predicted pixel points/colors; the sole geometry change is rigid Z-up."""
    from scripts.workcell_photo_oneshot import _array
    points, rgb, valid = (_array(frame[key]) for key in ('pts3d', 'image', 'non_ambiguous_mask'))
    if points.ndim != 3 or points.shape[-1] != 3 or rgb.shape != points.shape or valid.shape != points.shape[:2]:
        raise ValueError('Raw point, RGB and validity rasters disagree')
    valid = valid.astype(bool) & np.isfinite(points).all(-1)
    transformed = np.full(points.shape, np.nan, np.float32)
    transformed[valid] = trimesh.transform_points(points[valid], transform)
    return transformed, rgb, valid


def _observation_point_mask(observation, shape):
    """Reuse saved source polygons; do not substitute a model or bounding box."""
    import cv2
    mask = np.zeros(shape, np.uint8)
    polygons = observation.get('polygons') or ([observation['polygon']] if observation.get('polygon') else [])
    for polygon in polygons:
        pixels = np.asarray(polygon, float)
        if pixels.ndim != 2 or pixels.shape[1] != 2 or len(pixels) < 3 or not np.isfinite(pixels).all():
            raise ValueError('Invalid saved source polygon')
        cv2.fillPoly(mask, [np.rint(pixels).astype(np.int32)], 1)
    return mask.astype(bool)


def _ground_distance(item, geometry, transform):
    """Use the multiview physical edge; visible point-cloud extrema are not endpoints."""
    result = {'byPhoto': {}, 'rangeNative': None, 'sourcePhotos': [], 'feature': None,
              'source': '多视角实体底边到同一拟合地面的垂直距离。'}
    physical = geometry.get('physicalClearances', {})
    row = next((r for r in physical.get('objects', []) if r['id'] == item['id']), None)
    if row is None or row.get('status') != 'conditional':
        result['reason'] = (row or {}).get('reason', '没有跨照片确认的实体底边；离地距离未知。')
        return result
    try:
        endpoints = np.asarray([row.get('pointNative'), row.get('footNative')], float)
    except (TypeError, ValueError):
        result['reason'] = '实体底边或地面投影证据无效；离地距离未知。'
        return result
    height = row.get('heightNative'); photos = sorted(set(row.get('sourcePhotos', [])))
    if (not isinstance(height, (float, int)) or not np.isfinite(height) or height < 0
            or len(photos) < 2 or endpoints.shape != (2, 3) or not np.isfinite(endpoints).all()):
        result['reason'] = '实体底边或地面投影证据无效；离地距离未知。'
        return result
    normal = np.asarray(geometry['floor']['normal'], float)
    raw_offset = geometry['floor']['offset']
    if normal.shape != (3,) or not np.isfinite(normal).all() or np.linalg.norm(normal) < 1e-8 or not np.isfinite(raw_offset):
        raise ValueError('A finite nonzero ground plane is required')
    length = np.linalg.norm(normal); normal /= length; offset = raw_offset/length
    if (not np.isclose(endpoints[1] @ normal + offset, 0, atol=1e-7)
            or not np.allclose(endpoints[0]-endpoints[1], height*normal, atol=1e-7)):
        raise ValueError('Physical clearance and displayed floor refer to different planes')
    point, foot = trimesh.transform_points(endpoints, transform).tolist()
    result.update(sourcePhotos=photos, rangeNative=row.get('rangeNative'))
    result['feature'] = {'valueNative': height, 'pointNative': point, 'footNative': foot,
                         'sourcePhotos': photos, 'rangeNative': row.get('rangeNative'),
                         'source': result['source'], 'status': row['status'],
                         'uncertainty': row.get('uncertainty')}
    return result


def build(root):
    from scripts.workcell_photo_oneshot import _array, _frame
    root = Path(root)
    geometry = json.loads((root / 'geometry.json').read_text())
    catalog = json.loads((root / 'objects.json').read_text())
    doc = empty_document()
    doc['captureId'] = root.name
    doc['geometryBindings'] = {}
    frame_id = 'workcell-floor'
    normal = np.asarray(geometry['floor']['normal'], float)
    if normal.shape != (3,) or not np.isfinite(normal).all() or np.linalg.norm(normal) < 1e-8 or not np.isfinite(geometry['floor']['offset']):
        raise ValueError('A finite nonzero ground plane is required')
    transform = trimesh.geometry.align_vectors(normal / np.linalg.norm(normal), [0, 0, 1])
    transform[2, 3] = geometry['floor']['offset'] / np.linalg.norm(normal)
    from scripts.workcell_photo_calibration import accepted_scale
    scale = accepted_scale(geometry)
    from scripts.workcell_conditional_scale import model_measurement_scale
    measurement_scale = model_measurement_scale(root, geometry)
    doc['coordinateFrames'] = [{'id': frame_id, 'convention': 'opencv',
        'scale': {'status': 'model_estimated' if scale is not None else 'uncalibrated', 'nativeToMeters': scale,
                  'sourceRefs': [{'kind': 'user_dimension_hypothesis', 'heightM': geometry['anchor']['assumedHeightM'],
                                  'widthM': geometry['anchor']['assumedWidthM']}]},
        'ground': {'normal': [0, 0, 1], 'offset': 0, 'plane': [0, 0, 1, 0]},
        'source': 'rigid transform of current MapAnything world; original projection preserved'}]
    if geometry.get('calibration'):
        doc['coordinateFrames'][0]['scale']['sourceRefs'] = [{'kind': 'user_measured_reference',
            'primaryAxis': 'joint3DReference', **geometry['calibration']['reference']}]
    urls, scene_cache, source_frames = {}, {}, {}
    def asset(path, aid, kind, **metadata):
        path = Path(path)
        data = path.read_bytes()
        media = 'model/gltf-binary' if path.suffix == '.glb' else 'image/png'
        doc['assets'].append({'id': aid, 'kind': kind, 'mediaType': media,
                            'sizeBytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(),
                            'metadata': {'name': path.name, **metadata}})
        urls[aid] = path.name
        return aid

    def point_representation(name, xyz, rgb, refs, **provenance):
        from fast_report.layers import points_glb
        if not len(xyz):
            return None
        payload, point_metadata = points_glb(xyz, rgb, point_size=0)
        filename = f'entity-points-{name}.glb'
        (root / filename).write_bytes(payload)
        aid = asset(root / filename, 'asset-points-'+name, 'geometry',
                    **point_metadata, pointCount=len(xyz), **provenance)
        return {'id': 'rep-points-'+name, 'kind': 'point_cloud', 'assetId': aid,
                'coordinateFrameId': frame_id,
                'transform': {'coordinateFrameId': frame_id, 'position': [0, 0, 0],
                              'quaternion': [0, 0, 0, 1], 'scale': [1, 1, 1]},
                'bounds': {'min': xyz.min(0).tolist(), 'max': xyz.max(0).tolist()},
                'placementState': 'confirmed', 'sourceValidity': 'current',
                'sourceKind': 'predicted_source_point_cloud',
                'placementNote': 'Source pixel/depth correspondence only; model-inferred geometry, not surveyed ground truth',
                'sourceRefs': refs}

    context_representations, context_observations = [], []
    for i in range(1, 5):
        f = _frame(root, i)
        source_frames[i] = _source_points(f, transform)
        image = asset(root / f'photo-{i}.png', f'photo-{i}', 'source_image', photo=i)
        doc['geometryBindings'][image] = {'cameraId': f'camera-{i}', 'geometrySolutionId': 'oneshot-mapanything'}
        h, w = f['image']['shape'][:2]
        pose = transform @ _array(f['camera_poses'])
        doc['cameras'].append({'id': f'camera-{i}', 'imageId': image, 'coordinateFrameId': frame_id,
                              'width': w, 'height': h, 'K': _array(f['intrinsics']).tolist(),
                              'cameraToWorld': pose.tolist(), 'sourceRefs': [{'photo': i}]})
        oid = f'obs-source-points-{i}'
        context_observations.append(oid)
        doc['observations'].append({'id': oid, 'revision': 1, 'imageId': image,
            'originalPixelBox': [0, 0, w, h], 'originalPixelPolygons': [],
            'boxConvention': 'edges_xyxy_right_bottom_exclusive',
            'sourceRefs': [{'file': f'frame_{i:04d}.json.gz', 'photo': i}]})
        points, rgb, valid = source_frames[i]
        context = point_representation(f'capture-photo-{i}', points[valid], rgb[valid],
            [{'file': f'frame_{i:04d}.json.gz', 'photo': i, 'imageId': image, 'observationId': oid, 'revision': 1}],
            source='MapAnything saved pts3d/image/non_ambiguous_mask; all finite valid source pixels',
            sampling='none', units='native', groundTruth=False)
        if context is not None:
            context_representations.append(context)
    if context_representations:
        doc['entities'].append({'id': 'source-capture-points', 'label': '四张照片的推断点云（非真值）',
            'observationRefs': context_observations, 'representations': context_representations, 'visible': True, 'sourceContext': True,
            'associationState': 'association_pending',
            'physicalDimensionsUnknown': True, 'modelOrientationUnknown': True})

    def representation(item, spec, suffix):
        filename = spec['file']
        if filename not in scene_cache:
            scene_cache[filename] = trimesh.load(root / filename, force='scene')
        scene = scene_cache[filename]
        parts = []
        for node in spec['nodes']:
            matrix, mesh_id = scene.graph.get(node)
            original = scene.geometry[mesh_id]
            mesh = original.copy()
            if original.visual.kind == 'texture' and 'color' in original.visual.vertex_attributes:
                mesh.visual.vertex_attributes['color'] = original.visual.vertex_attributes['color'].copy()
            mesh.apply_transform(transform @ matrix)
            parts.append(mesh)
        if not parts:
            raise ValueError(f"{item['id']}: no model nodes")
        mesh = trimesh.util.concatenate(parts)
        # trimesh concatenation drops glTF COLOR_0 and can reset sidedness.
        if all(p.visual.kind == 'texture' and 'color' in p.visual.vertex_attributes for p in parts):
            mesh.visual.vertex_attributes['color'] = np.concatenate([trimesh.visual.color.to_rgba(p.visual.vertex_attributes['color']) for p in parts])
            if all(p.visual.material.doubleSided for p in parts):
                mesh.visual.material.doubleSided = True
        if not np.isfinite(mesh.vertices).all() or not len(mesh.faces):
            raise ValueError(f"{item['id']}: invalid mesh")
        center = mesh.bounds.mean(0)
        mesh.apply_translation(-center)
        name = f"entity-{item['id']}{suffix}.glb"
        (root / name).write_bytes(mesh.export(file_type='glb'))
        aid = asset(root / name, 'asset-'+item['id']+suffix, 'geometry')
        return {'id': 'rep-'+item['id']+suffix, 'kind': 'generated_mesh', 'assetId': aid,
            'coordinateFrameId': frame_id, 'transform': {'coordinateFrameId': frame_id,
                'position': center.tolist(), 'quaternion': [0, 0, 0, 1], 'scale': [1, 1, 1]},
            'bounds': {'min': mesh.bounds[0].tolist(), 'max': mesh.bounds[1].tolist()},
            'placementState': 'unconfirmed', 'placementReason': 'requires_alignment_confirmation',
            'placementNote': 'Automatic current-photo geometry; physical accuracy not independently verified',
            'sourceValidity': 'current', 'sourceKind': item['representation'],
            'sourceRefs': [{'file': filename, 'nodes': spec['nodes']}]}

    counts = {}
    for item in catalog['objects']:
        category = item['kind']
        counts[category] = counts.get(category, 0) + 1
        label = LABELS.get(category, item['label'])
        if category not in ('robot', 'cart', 'floor', 'emergency stop button', 'emergency_button'):
            label += ' ' + str(counts[category])
        if 'button' in item['id']:
            label = '红黄急停按钮'
        item['label'] = label
        observations, point_representations = [], []
        for j, obs in enumerate(item['observations']):
            oid = f"obs-{item['id']}-{obs['photo']}-{j}"
            observations.append(oid)
            doc['observations'].append({'id': oid, 'revision': 1, 'imageId': f"photo-{obs['photo']}",
                'originalPixelBox': obs['box'], 'originalPixelPolygons': obs.get('polygons') or ([obs['polygon']] if obs.get('polygon') else []),
                'polygonCoordinateConvention': 'pixel_centers', 'boxConvention': 'edges_xyxy_right_bottom_exclusive',
                'sourceRefs': [{'photo': obs['photo'], 'evidence': obs.get('source', obs.get('evidence', 'current observation'))}]})
            points, colors, valid = source_frames[obs['photo']]
            support = valid & _observation_point_mask(obs, valid.shape)
            xyz, rgb = points[support], colors[support]
            # ponytail: cap duplicated selectable subsets; the context cloud
            # above retains every valid source point without subsampling.
            stride = max(1, (len(xyz) + 11999) // 12000)
            point_rep = point_representation(f"{item['id']}-photo-{obs['photo']}-{j}", xyz[::stride], rgb[::stride],
                [{'observationId': oid, 'imageId': f"photo-{obs['photo']}", 'revision': 1,
                  'file': f"frame_{obs['photo']:04d}.json.gz"}],
                source='Saved object source polygons rasterized on the original predicted pixel grid',
                rawSupportedPointCount=len(xyz), samplingStride=stride, units='native', groundTruth=False)
            if point_rep is not None:
                point_representations.append(point_rep)
        photos = {o['photo'] for o in item['observations']}
        samples = [o for o in item['observations'] if o.get('observedMeasurements', {}).get('status') == 'available']
        item['visibleHeightByPhoto'] = {str(o['photo']): o['observedMeasurements']['dimensions_native']['height'] for o in samples}
        heights = list(item['visibleHeightByPhoto'].values())
        item['visibleHeightNative'] = float(np.median(heights)) if len(photos) > 1 and heights else None
        item['visibleHeightRangeNative'] = [min(heights), max(heights)] if len(photos) > 1 and heights else None
        item['groundDistance'] = _ground_distance(item, geometry, transform)
        measurements = {}
        if len(photos) > 1 and samples:
            # Recompute canonical axes from the same visible support after the rigid world transform.
            selected = max(samples, key=lambda o: o['observedMeasurements']['quality']['supported_points'])
            measurements = canonical_measurements(selected['observedMeasurements'], frame_id, [{'observationId': observations[item['observations'].index(selected)]}])
            basis = measurements.get('basis')
            if basis and basis.get('cornersNative'):
                basis['cornersNative'] = trimesh.transform_points(basis['cornersNative'], transform).tolist()
            if basis and basis.get('axesNative'):
                basis['axesNative'] = (np.asarray(basis['axesNative']) @ transform[:3, :3].T).tolist()
        for evidence in measurements.get('orientationEvidence', {}).values():
            for field in ('axisNative', 'normalNative'):
                if evidence.get(field) is not None:
                    evidence[field] = (transform[:3, :3] @ np.asarray(evidence[field])).tolist()
        variants = {key: representation(item, spec, '-photo-'+key) for key, spec in item.get('modelsByPhoto', {}).items()}
        rep = variants.get('4') or (representation(item, item['model'], '') if (item.get('model') or {}).get('nodes') else None)
        entity = {'id': item['id'], 'label': label, 'observationRefs': observations,
                  'associationState': 'association_pending', 'representations': ([rep] if rep else []) + point_representations,
                  'activeModelRepresentationId': rep['id'] if rep else None, 'currentModelTransform': rep['transform'] if rep else None,
                  'measurements': measurements, 'visible': True, 'sourceContext': False,
                  'physicalDimensionsUnknown': True, 'modelOrientationUnknown': True,
                  'observedExtentAvailable': len(photos) > 1 and bool(samples),
                  'modelVariants': variants, 'notes': item.get('notes', []),
                  'lineage': [{'operation': 'current_photo_oneshot', 'sourceObjectId': item['id']}]}
        doc['entities'].append(entity)
    validate_document(doc)
    timestamp = datetime.now(timezone.utc).isoformat()
    revision = {'id': root.name, 'projectId': 'workcell-photo', 'branchId': 'oneshot',
                'parentRevisionId': None, 'sourceRevisionId': None, 'createdAt': timestamp,
                'documentSha256': hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest(),
                'label': '四张照片 oneshot', 'document': doc}
    Revision.model_validate(revision)
    from ehs_spatial.platform.scene_measurements import analyze_bends
    guard_revision = {**revision, 'document': {**doc, 'entities': [e for e in doc['entities'] if e['id'].startswith('v-guard-')]}}
    def load_asset(asset_id):
        return (root / urls[asset_id]).read_bytes()
    bend_analysis = analyze_bends(guard_revision, load_asset)
    result = {'schemaVersion': 1, 'revision': revision, 'assetURLs': urls, 'objects': catalog['objects'],
              'coverage': catalog['coverage'], 'geometry': geometry, 'sceneTransformNative': transform.tolist(),
              'nativeToMetersDefault': scale, 'timing': {},
              'bendAnalysis': bend_analysis, 'modelMeasurementScale': measurement_scale}
    # Explicit inspection artifact: these named endpoints are specific to this capture.
    endpoint_path = root / 'model-endpoint-estimate.json'
    if endpoint_path.is_file():
        measured = json.loads(endpoint_path.read_text())
        for name, digest in measured['sourceFiles'].items():
            if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
                raise ValueError('Model endpoint inspection is stale: ' + name)
        if not np.allclose(measured['sceneTransformNative'], transform, atol=1e-7):
            raise ValueError('Model endpoint inspection uses a different floor')
        conditional = measurement_scale.get('evidence')
        factor = measurement_scale['nativeToMeters']
        if factor is None:
            raise ValueError('Endpoint centimetre inspection requires an explicit model measurement scale')
        lo, hi = measurement_scale['rangeNativeToMeters']
        endpoints = []
        for row in measured['objects']:
            point, foot = trimesh.transform_points([row['pointNative'], row['footNative']], transform)
            height = row['heightNative']
            if not np.isfinite([*point, *foot, height]).all() or not np.allclose(point-foot, [0, 0, height], atol=1e-7) or abs(foot[2]) > 1e-7:
                raise ValueError('Invalid model endpoint or floor projection')
            endpoints.append({**row, 'pointNative': point.tolist(), 'footNative': foot.tolist(),
                'label': '光幕底端' if row['objectId'] == 'post-box-1' else '邻近围栏下沿',
                'estimateCm': height * factor * 100,
                'rangeCm': [height * lo * 100, height * hi * 100]})
        by_id = {row['objectId']: row for row in endpoints}
        delta = by_id['post-box-1']['heightNative'] - by_id['fence-0']['heightNative']
        result['endpointEstimation'] = {'status': 'conditional_unvalidated', 'photo': 4,
            'method': '读取当前 GLB 的光幕底面中心，以及紧邻它的围栏底面中心线；围栏此段为模型推断延伸。沿同一地面法向测量。',
            'scale': {'mPerNative': factor, 'source': measurement_scale['source']},
            'endpoints': endpoints,
            'difference': {'valueNative': delta, 'valueCm': delta * factor * 100,
                'rangeCm': sorted([delta * lo * 100, delta * hi * 100]),
                'description': '光幕底端高度减去邻近围栏下沿高度。'},
            'scaleEvidence': conditional, 'sourceFiles': measured['sourceFiles'],
            'rangeMeaning': '仅为固定模型端点在标尺表面深度第 5/95 百分位下的敏感性；不包含模型、地面或轴心深度系统误差。',
            'groundTruthUsedForEstimation': False}
    if (root/'measurements.json').is_file():
        from scripts.workcell_photo_calibration import load_measurements, measurement_evaluation
        result['measurementEvaluation'] = measurement_evaluation(catalog['objects'], geometry, load_measurements(root/'measurements.json'))
        (root/'measurement-evaluation.json').write_text(json.dumps(result['measurementEvaluation'], ensure_ascii=False, indent=2)+'\n')
    (root / 'scene-report.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    return result
