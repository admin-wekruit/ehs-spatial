"""Measured button reference and independent evaluation of frozen photo geometry."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import trimesh

FEATURES = ('wholeComponentHeightM', 'mainBodyDiameterM', 'redActuatorDiameterM')
SCOPE = 'whole component: red cap, yellow body, gray lower housing; mounting bracket excluded'


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be a finite positive number')
    return float(value)


def validate_measurements(data):
    if not isinstance(data, dict) or data.get('schemaVersion') != 1:
        raise ValueError('Measurements require schemaVersion 1')
    reference = data.get('reference', {})
    if reference.get('objectId') != 'emergency-button' or reference.get('scope') != SCOPE:
        raise ValueError('Reference must identify the whole red/yellow/gray emergency-button component, excluding its bracket')
    if reference.get('scopeStatus') not in ('pending_confirmation', 'confirmed') or not reference.get('provenance'):
        raise ValueError('Reference scopeStatus and provenance are required')
    features = reference.get('features', {})
    if set(features) != set(FEATURES):
        raise ValueError('Provide separately named whole-component height, main-body diameter and red-actuator diameter')
    for name in FEATURES:
        _positive(features[name], name)
    if features['redActuatorDiameterM'] > features['mainBodyDiameterM']:
        raise ValueError('Red-actuator diameter exceeds the supplied maximum main-body diameter')
    seen = set()
    for target in data.get('evaluation', {}).get('targets', []):
        ident = target.get('objectId')
        if ident not in ('fence-0', 'post-box-1', 'post-box-2') or ident in seen:
            raise ValueError('Evaluation requires distinct recognized fence/light-curtain object IDs')
        seen.add(ident)
        _positive(target.get('groundTruthM'), f'{ident} groundTruthM')
        if not target.get('feature') or not target.get('provenance'):
            raise ValueError('Evaluation feature and provenance are required')
    return copy.deepcopy(data)


def load_measurements(path):
    return validate_measurements(json.loads(Path(path).read_text()))


def resolve_dimensions(measurements, diameter_m=None, height_m=None):
    """Validate before cloud work; the historic diameter option means envelope width."""
    if measurements is not None:
        measurements = validate_measurements(measurements)
        height = measurements['reference']['features']['wholeComponentHeightM']
        if diameter_m is not None:
            raise ValueError('--button-diameter-m is whole-envelope width and cannot override measured part diameters')
        if height_m is not None and _positive(height_m, 'button height') != height:
            raise ValueError('--button-height-m conflicts with --measurements whole-component height')
        # The cloud's legacy envelope-width diagnostic is replaced after its native bounds are saved.
        return .2, height
    return _positive(.2 if diameter_m is None else diameter_m, 'button width'), _positive(.2 if height_m is None else height_m, 'button height')


def accepted_scale(geometry):
    """One metric export contract for every report caller, including cached runs."""
    anchor = geometry['anchor']; fit = anchor.get('referenceFit', {})
    scale = anchor.get('mPerNative')
    if scale is None and fit.get('status') != 'available':
        return None
    scale = _positive(scale, 'accepted scene scale')
    if fit.get('status') != 'available' or not fit.get('camerasFixed') or fit.get('mPerNative') != scale:
        raise ValueError('Scene scale must equal an accepted fixed-camera 3D reference fit')
    return scale


def _projection_diagnostics(root, anchor, meshes):
    from scripts.workcell_photo_oneshot import _array, _frame
    points = np.concatenate([mesh.vertices for mesh in meshes.values()])
    rows = []
    for view in anchor['views']:
        frame = _frame(root, view['photo'])
        pose, K = _array(frame['camera_poses']), _array(frame['intrinsics'])
        local = (points - pose[:3, 3]) @ pose[:3, :3]
        if (local[:, 2] <= 0).any():
            raise ValueError('Measured reference model projects behind a source camera')
        pixels = local @ K.T; pixels = pixels[:, :2] / pixels[:, 2:3]
        x0, y0, x1, y1 = view['boxRaw']
        boundary = np.array([[x0, y0, 1], [x1, y0, 1], [x0, y1, 1], [x1, y1, 1]])
        observed = boundary @ np.asarray(frame['input_mask_transform']['input_to_canonical_pixel_centres']).T
        observed = observed[:, :2] / observed[:, 2:3]
        lo, hi = pixels.min(0), pixels.max(0)
        olo, ohi = observed.min(0), observed.max(0)
        intersection = np.prod(np.maximum(0, np.minimum(hi, ohi) - np.maximum(lo, olo)))
        union = np.prod(hi-lo) + np.prod(ohi-olo) - intersection
        rows.append({'photo': view['photo'], 'coordinateSpace': 'canonical image pixels',
                     'observedBBox': [*olo.tolist(), *ohi.tolist()], 'modelProjectedBBox': [*lo.tolist(), *hi.tolist()],
                     'bboxIoU': float(intersection / union),
                     'centerOffsetPx': float(np.linalg.norm((hi+lo-ohi-olo)/2)),
                     'note': 'Projected rendering envelope versus observed whole-component box; not a part-segmentation accuracy score.'})
    return rows


def apply_measurements(root, measurements):
    """Attach supplied dimensions; only an accepted 3D reference fit authorizes scale."""
    from scripts.workcell_photo_objects import button_meshes
    root = Path(root)
    measurements = validate_measurements(measurements)
    geometry = json.loads((root/'geometry.json').read_text())
    anchor = geometry['anchor']; reference = measurements['reference']; features = reference['features']
    fit = anchor.get('referenceFit', {})
    if fit.get('knownDimensions') is not None and fit['knownDimensions'] != features:
        raise ValueError('Reference dimensions changed: rerun the 3D fit before applying measurements')
    scale = _positive(fit.get('mPerNative'), 'accepted 3D reference scale') if fit.get('status') == 'available' else None
    if fit.get('status') == 'available' and not fit.get('camerasFixed'):
        raise ValueError('Main scene calibration requires the unchanged reconstruction cameras')
    if fit.get('fittedNuisanceParameters') is not None:
        from scripts.workcell_photo_oneshot import _array, _frame
        actual = []
        for photo in range(1, 5):
            raw = _frame(root, photo)
            actual.append({'photo': photo, 'K': _array(raw['intrinsics']).tolist(),
                           'pose': _array(raw['camera_poses']).tolist(),
                           'A': np.asarray(raw['input_mask_transform']['input_to_canonical_pixel_centres']).tolist()})
        signature = hashlib.sha256(json.dumps(actual, sort_keys=True).encode()).hexdigest()
        if not fit.get('camerasFixed') or fit.get('cameraProvenance', {}).get('framesSha256') != signature:
            raise ValueError('Button candidate and current scene cameras disagree; rerun calibration')
    height, width = anchor['nativeHeight'], anchor['nativeWidth']
    geometry['calibration'] = {
        'schemaVersion': 1, 'primaryAxis': 'joint3DReference', 'reference': reference,
        'status': fit.get('status', 'unsupported'), 'nativeToMeters': scale,
        'groundTruthUsedForCalibration': False,
        'observedEnvelope': {'heightNative': height, 'widthNative': width,
                             'heightM': height*scale if scale is not None else None,
                             'widthM': width*scale if scale is not None else None},
        'diagnostics': {'referenceFit': fit, 'envelopeUsedForCalibration': False},
        'renderingAssumptions': [
            'Three supplied dimensions constrain one perspective-fit 3D button; component heights and gray housing shape are fitted nuisance parameters.',
            'Saved scene cameras stay fixed. Unsupported candidates are visual hypotheses and do not establish a metric scale.']}
    anchor.update(mPerNative=scale, assumedHeightM=features['wholeComponentHeightM'],
                  assumedWidthM=features['mainBodyDiameterM'], scope=reference['scope'],
                  status='three-dimension 3D reference fit' if scale is not None else '3D reference scale unsupported',
                  relativeWidthResidual=None)
    meshes = button_meshes(geometry)
    scene = trimesh.load(root/'object-extras.glb', force='scene')
    for node, mesh in meshes.items():
        matrix, geometry_id = scene.graph.get(node)
        local = mesh.copy(); local.apply_transform(np.linalg.inv(matrix))
        scene.geometry[geometry_id] = local
    (root/'object-extras.glb').write_bytes(scene.export(file_type='glb'))
    geometry['calibration']['diagnostics']['projectionByPhoto'] = _projection_diagnostics(root, anchor, meshes)
    catalog = json.loads((root/'objects.json').read_text())
    button = next(o for o in catalog['objects'] if o['id'] == 'emergency-button')
    button['referenceGeometry'] = {'features': features, 'provenance': reference['provenance'],
                                   'scope': reference['scope'], 'scopeStatus': reference['scopeStatus'],
                                   'observedEnvelope': geometry['calibration']['observedEnvelope'],
                                   'renderingAssumptions': geometry['calibration']['renderingAssumptions']}
    for name, field in (('height', 'wholeComponentHeightM'), ('width', 'mainBodyDiameterM'), ('redActuatorDiameter', 'redActuatorDiameterM')):
        button['measurements'][name] = {'valueNative': features[field]/scale if scale is not None else None,
                                       'valueM': features[field], 'status': 'user-supplied-reference', 'feature': field,
                                       'source': 'Supplied reference dimensions; not reconstructed observed bounds.'}
    button['notes'] = geometry['calibration']['renderingAssumptions']
    for name, value in [('objects.json', catalog), ('geometry.json', geometry), ('measurements.json', measurements)]:
        (root/name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    return geometry


def _summary(by_photo, scale, eligible=True):
    values = [v for v in by_photo.values() if v is not None and np.isfinite(v) and v >= 0]
    enough = eligible and len(by_photo) >= 2 and len(values) >= 2
    median = float(np.median(values)) if enough else None
    bounds = [min(values), max(values)] if enough else None
    return {'medianNative': median, 'medianM': median*scale if median is not None and scale is not None else None,
            'rangeNative': bounds, 'rangeM': [v*scale for v in bounds] if bounds and scale is not None else None,
            'byPhoto': {str(p): {'valueNative': v, 'valueM': v*scale if v is not None and scale is not None else None} for p, v in by_photo.items()},
            'sourcePhotos': sorted(int(p) for p in by_photo), 'method': 'median of valid saved per-photo visible support; at least two views'}


def measurement_evaluation(objects, geometry, measurements):
    """GT only enters subtraction, after estimates are frozen by source evidence."""
    measurements = validate_measurements(measurements)
    scale = geometry['anchor']['mPerNative']
    estimates = []
    for item in objects:
        ground = item['groundDistance']
        pose_dependent = bool(item.get('modelsByPhoto'))
        heights = _summary(item['visibleHeightByPhoto'], scale, not pose_dependent)
        lower = _summary({p: v['valueNative'] for p, v in ground['byPhoto'].items()}, scale, not pose_dependent)
        estimates.append({'objectId': item['id'], 'label': item['label'], 'visibleHeight': heights,
                          'lowerEdge': lower, 'physicalDimensionsUnknown': True, 'poseDependent': pose_dependent,
                          'limitation': 'Visible support only; occluded full extent and physical endpoint association remain unconfirmed.'})
    comparisons = []
    for target in measurements.get('evaluation', {}).get('targets', []):
        item = next((o for o in objects if o['id'] == target['objectId']), None)
        if item is None:
            raise ValueError(f"Evaluation object missing: {target['objectId']}")
        ground = item['groundDistance']; feature = ground.get('feature') or {}
        value = feature.get('valueNative')
        bounds = feature.get('rangeNative') if value is not None else None
        photos = feature.get('sourcePhotos', [])
        method, source = 'multiview physical lower edge', feature.get('source', feature.get('reason', ground.get('reason')))
        estimate = value*scale if value is not None and scale is not None else None
        truth = target['groundTruthM']; error = estimate-truth if estimate is not None else None
        by_photo = {p: {**v, 'valueM': v['valueNative']*scale if v['valueNative'] is not None and scale is not None else None}
                    for p, v in ground['byPhoto'].items()}
        comparisons.append({'objectId': item['id'], 'label': item['label'], 'method': method,
            'estimateNative': value, 'estimateM': estimate, 'rangeNative': bounds,
            'rangeM': [v*scale for v in bounds] if bounds and scale is not None else None, 'byPhoto': by_photo,
            'byPhotoMethod': 'independently supported physical edge observations',
            'sourcePhotos': photos, 'groundTruthM': truth, 'signedErrorM': error,
            'absoluteErrorM': abs(error) if error is not None else None,
            'relativeError': error/truth if error is not None else None, 'source': source,
            'target': target, 'limitation': 'Metric result requires accepted 3D reference scale and independently supported physical lower edge.'})
    return {'schemaVersion': 1, 'scaleMPerNative': scale, 'groundTruthUsedForCalibration': False,
            'comparisons': comparisons, 'objectEstimates': estimates}
