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


def build(root):
    from scripts.workcell_photo_oneshot import _array, _frame
    root = Path(root)
    geometry = json.loads((root / 'geometry.json').read_text())
    catalog = json.loads((root / 'objects.json').read_text())
    doc = empty_document()
    doc['captureId'] = root.name
    doc['geometryBindings'] = {}
    frame_id = 'workcell-floor'
    transform = trimesh.geometry.align_vectors(geometry['floor']['normal'], [0, 0, 1])
    transform[2, 3] = geometry['floor']['offset']
    scale = geometry['anchor']['mPerNative']
    doc['coordinateFrames'] = [{'id': frame_id, 'convention': 'opencv',
        'scale': {'status': 'model_estimated', 'nativeToMeters': scale,
                  'sourceRefs': [{'kind': 'user_dimension_hypothesis', 'heightM': geometry['anchor']['assumedHeightM'],
                                  'widthM': geometry['anchor']['assumedWidthM']}]},
        'ground': {'normal': [0, 0, 1], 'offset': 0, 'plane': [0, 0, 1, 0]},
        'source': 'rigid transform of current MapAnything world; original projection preserved'}]
    urls, scene_cache = {}, {}
    def asset(path, aid, kind, **metadata):
        path = Path(path)
        data = path.read_bytes()
        media = 'model/gltf-binary' if path.suffix == '.glb' else 'image/png'
        doc['assets'].append({'id': aid, 'kind': kind, 'mediaType': media,
                            'sizeBytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(),
                            'metadata': {'name': path.name, **metadata}})
        urls[aid] = path.name
        return aid
    for i in range(1, 5):
        f = _frame(root, i)
        image = asset(root / f'photo-{i}.png', f'photo-{i}', 'source_image', photo=i)
        doc['geometryBindings'][image] = {'cameraId': f'camera-{i}', 'geometrySolutionId': 'oneshot-mapanything'}
        h, w = f['image']['shape'][:2]
        pose = transform @ _array(f['camera_poses'])
        doc['cameras'].append({'id': f'camera-{i}', 'imageId': image, 'coordinateFrameId': frame_id,
                              'width': w, 'height': h, 'K': _array(f['intrinsics']).tolist(),
                              'cameraToWorld': pose.tolist(), 'sourceRefs': [{'photo': i}]})

    def representation(item, spec, suffix):
        filename = spec['file']
        if filename not in scene_cache:
            scene_cache[filename] = trimesh.load(root / filename, force='scene')
        scene = scene_cache[filename]
        parts = []
        for node in spec['nodes']:
            matrix, mesh_id = scene.graph.get(node)
            mesh = scene.geometry[mesh_id].copy()
            mesh.apply_transform(transform @ matrix)
            parts.append(mesh)
        if not parts:
            raise ValueError(f"{item['id']}: no model nodes")
        mesh = trimesh.util.concatenate(parts)
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
        observations = []
        for j, obs in enumerate(item['observations']):
            oid = f"obs-{item['id']}-{obs['photo']}-{j}"
            observations.append(oid)
            doc['observations'].append({'id': oid, 'revision': 1, 'imageId': f"photo-{obs['photo']}",
                'originalPixelBox': obs['box'], 'originalPixelPolygons': obs.get('polygons') or ([obs['polygon']] if obs.get('polygon') else []),
                'polygonCoordinateConvention': 'pixel_centers', 'boxConvention': 'edges_xyxy_right_bottom_exclusive',
                'sourceRefs': [{'photo': obs['photo'], 'evidence': obs.get('source', obs.get('evidence', 'current observation'))}]})
        photos = {o['photo'] for o in item['observations']}
        samples = [o for o in item['observations'] if o.get('observedMeasurements', {}).get('status') == 'available']
        item['visibleHeightByPhoto'] = {str(o['photo']): o['observedMeasurements']['dimensions_native']['height'] for o in samples}
        heights = list(item['visibleHeightByPhoto'].values())
        item['visibleHeightNative'] = float(np.median(heights)) if len(photos) > 1 and heights else None
        item['visibleHeightRangeNative'] = [min(heights), max(heights)] if len(photos) > 1 and heights else None
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
                  'associationState': 'association_pending', 'representations': [rep] if rep else [],
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
    from ehs_spatial.platform.scene_measurements import analyze_bends, analyze_inclinations
    guard_revision = {**revision, 'document': {**doc, 'entities': [e for e in doc['entities'] if e['id'] == 'v-guard']}}
    def load_asset(asset_id):
        return (root / urls[asset_id]).read_bytes()
    bend_analysis = analyze_bends(guard_revision, load_asset)
    inclination_analysis = analyze_inclinations(guard_revision, load_asset)
    result = {'schemaVersion': 1, 'revision': revision, 'assetURLs': urls, 'objects': catalog['objects'],
              'coverage': catalog['coverage'], 'geometry': geometry, 'sceneTransformNative': transform.tolist(),
              'nativeToMetersDefault': scale, 'timing': {},
              'bendAnalysis': bend_analysis, 'inclinationAnalysis': inclination_analysis}
    (root / 'scene-report.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    return result
