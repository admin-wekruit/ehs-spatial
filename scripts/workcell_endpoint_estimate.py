"""Conditional terminal heights from actual display meshes and Photo 4 pointmaps.

``estimate(root)`` measures current GLBs; ``measure_edge``/``build`` separately
diagnose named RGB edges using cached depth. No surveyed comparison values or
scene scale enter either measurement chain.
"""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import trimesh

from workcell_photo_geometry import _intersect
from workcell_photo_oneshot import _array, _frame


def _bottom_face(scene, node, normal):
    transform, name = scene.graph[node]
    mesh = scene.geometry[name].copy()
    mesh.apply_transform(transform)
    alignment = mesh.face_normals @ normal
    faces = np.flatnonzero(alignment <= alignment.min() + 1e-6)
    return mesh.vertices[np.unique(mesh.faces[faces])]


def estimate(root):
    """Measure the displayed light curtain and adjacent lower rail in native units.

    The two GLBs and current floor are the entire measurement input. This is a
    model estimate, independent of the photo endpoint diagnostic and metric scale.
    """
    root = Path(root)
    ground = json.loads((root / 'physical-clearances.json').read_text())['ground']
    normal = np.asarray(ground['normal'], float)
    norm = np.linalg.norm(normal)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError('Invalid ground normal')
    normal, offset = normal / norm, float(ground['offset']) / norm
    catalog_path = root / 'objects.json'
    catalog = {item['id']: item for item in json.loads(catalog_path.read_text())['objects']} if catalog_path.is_file() else {}
    source_files = {'fence-fitted.glb', 'physical-clearances.json'}
    if catalog_path.is_file():
        source_files.add('objects.json')

    def housing_terminal(ident):
        evidence = catalog.get(ident, {}).get('physicalBottom')
        if evidence is None:
            return None
        model = catalog[ident]['model']; name = model['file']; path = (root / name).resolve()
        if (root.resolve() not in path.parents or evidence['modelFile'] != name or
                hashlib.sha256(path.read_bytes()).hexdigest() != evidence['modelSha256']):
            raise ValueError('Housing terminal model binding is stale: ' + ident)
        scene = trimesh.load(path, force='scene', process=False)
        points = []
        for ref in evidence['vertices']:
            node, index = ref['node'], ref['vertexIndex']
            if node not in model['nodes'] or type(index) is not int:
                raise ValueError('Housing terminal references another object: ' + ident)
            matrix, mesh_id = scene.graph[node]
            vertices = scene.geometry[mesh_id].vertices
            if not 0 <= index < len(vertices):
                raise ValueError('Housing terminal vertex is absent: ' + ident)
            points.append(trimesh.transform_points([vertices[index]], matrix)[0])
        points = np.asarray(points)
        if len(points) < 2 or not np.isfinite(points).all():
            raise ValueError('Housing terminal has no finite exported edge: ' + ident)
        source_files.add(name)
        return points[np.argmin(points @ normal)], points, evidence

    current_light = housing_terminal('post-box-1')
    if current_light is None:
        selected = catalog.get('post-box-1', {}).get('model')
        if selected is not None and selected != {'file': 'posts.glb', 'nodes': ['box-1']}:
            raise ValueError('Selected housing geometry needs an explicit physicalBottom terminal binding')
        posts = trimesh.load(root / 'posts.glb', force='scene')
        light_face = _bottom_face(posts, 'box-1', normal)
        light_point = light_face.mean(0)
        source_files.add('posts.glb')
    else:
        light_point, light_face, _ = current_light
    fences = trimesh.load(root / 'fence-fitted.glb', force='scene')
    rail_face = _bottom_face(fences, 'section-0-continued-3', normal)
    center = rail_face.mean(0)
    _, _, vectors = np.linalg.svd(rail_face - center, full_matrices=False)
    # The bottom face's longest axis is this rail's length. End-edge midpoints
    # lie on the actual mesh surface; clamped projection keeps the query on it.
    along = (rail_face - center) @ vectors[0]
    order = np.argsort(along)
    ends = np.array([rail_face[order[:2]].mean(0), rail_face[order[-2:]].mean(0)])
    direction = ends[1] - ends[0]
    fraction = float(np.clip((light_point - ends[0]) @ direction / (direction @ direction), 0, 1))
    rail_point = ends[0] + fraction * direction
    T = trimesh.geometry.align_vectors(normal, [0, 0, 1]); T[2, 3] = offset

    def record(ident, node, point, face, provenance):
        height = float(point @ normal + offset)
        foot = point - height * normal
        heights = face @ normal + offset
        return {'objectId': ident, 'node': node, 'status': 'conditional_model_estimate',
                'provenance': provenance, 'heightNative': height,
                'pointNative': point.tolist(), 'footNative': foot.tolist(),
                'pointReportNative': trimesh.transform_points([point], T)[0].tolist(),
                'footReportNative': trimesh.transform_points([foot], T)[0].tolist(),
                'bottomFaceVerticesNative': face.tolist(),
                'bottomFaceHeightRangeNative': [float(heights.min()), float(heights.max())],
                'rangeMeaning': 'Actual mesh bottom-face extent, not physical measurement uncertainty'}

    if current_light is None:
        light = record('post-box-1', 'box-1', light_point, light_face,
                       'Displayed upright primitive, bottom-face center; source depth percentile envelope')
    else:
        light = record('post-box-1', current_light[2]['vertices'][0]['node'], light_point, light_face,
                       'Lowest vertex of the selected visible-face terminal read from the actual source-supported GLB; same inferred floor; whole-housing minimum unverified')
        light['modelEvidence'] = current_light[2]
        light.update(measurementScope='visible_face_lower_terminal', wholeHousingMinimumVerified=False,
                     terminalPartAmbiguity=current_light[2]['terminalPartAmbiguity'])
    fence = record('fence-0', 'section-0-continued-3', rail_point, rail_face,
                   'Displayed inferred continuation of observed lower rail; nearest point to the selected curtain terminal along bottom-face centerline')
    fence.update(bottomCenterlineEndsNative=ends.tolist(), closestAlongRailFraction=fraction)
    objects = [light, fence]
    current_left = housing_terminal('post-box-2')
    if current_left is not None:
        point, vertices, evidence = current_left
        left = record('post-box-2', evidence['vertices'][0]['node'], point, vertices,
                      'Lowest vertex of the selected visible-face terminal read from the actual source-supported GLB; same inferred floor; whole-housing minimum unverified')
        left['modelEvidence'] = evidence
        left.update(measurementScope='visible_face_lower_terminal', wholeHousingMinimumVerified=False,
                    terminalPartAmbiguity=evidence['terminalPartAmbiguity'])
        objects.append(left)
    difference = light['heightNative'] - fence['heightNative']
    return {'schemaVersion': 1, 'status': 'conditional_model_estimate', 'mPerNative': None,
            'scope': 'Actual displayed model endpoints against the current saved floor; does not certify physical dimensions',
            'ground': {'normal': normal.tolist(), 'offset': offset}, 'sceneTransformNative': T.tolist(),
            'objects': objects, 'lightMinusFenceNative': difference,
            'sign': 'light_higher' if difference > 0 else 'light_lower' if difference < 0 else 'equal',
            'sourceFiles': {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                            for name in sorted(source_files)}}


def _pixels(points, affine):
    q = np.c_[points, np.ones(len(points))] @ affine.T
    return q[:, :2] / q[:, 2:3]


def measure_edge(frame, mask, raw_ends, ground, *, floor_samples=None):
    points, conf, valid = frame['points'], frame['conf'], frame['valid']
    normal = np.asarray(ground['normal'], float)
    norm = np.linalg.norm(normal)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError('Invalid ground normal')
    normal, offset = normal / norm, float(ground['offset']) / norm
    raw_ends = np.asarray(raw_ends, float)
    if raw_ends.shape != (2, 2) or not np.isfinite(raw_ends).all():
        raise ValueError('Expected two finite raw-image terminal-edge endpoints')
    tangent = raw_ends[1] - raw_ends[0]
    length = np.linalg.norm(tangent)
    if length <= 0:
        raise ValueError('Empty terminal edge')
    tangent /= length
    perpendicular = np.array([-tangent[1], tangent[0]])
    samples = raw_ends[0] + np.linspace(.1, .9, 11)[:, None] * (raw_ends[1] - raw_ends[0])
    A = frame['A']
    foot = lambda point: point - (point @ normal + offset) * normal
    T = trimesh.geometry.align_vectors(normal, [0, 0, 1]); T[2, 3] = offset

    def record(xyz, raw_uv, confidence=None):
        heights = xyz @ normal + offset
        # One actual sampled point represents the median along this named edge.
        index = int(np.argmin(abs(heights - np.median(heights))))
        point = xyz[index]; floor_point = foot(point)
        report = {'status': 'conditional_single_photo', 'pointNative': point.tolist(),
                  'footNative': floor_point.tolist(), 'heightNative': float(heights[index]),
                  'rawPixel': raw_uv[index].tolist(), 'samplePointsNative': xyz.tolist(),
                  'sampleRawPixels': raw_uv.tolist(), 'sampleHeightRangeNative': [float(heights.min()), float(heights.max())],
                  'pointReportNative': trimesh.transform_points([point], T)[0].tolist(),
                  'footReportNative': trimesh.transform_points([floor_point], T)[0].tolist()}
        if confidence is not None:
            report['depthConfidenceRaw'] = {'selected': float(confidence[index]), 'min': float(min(confidence)), 'max': float(max(confidence)),
                                          'scope': 'Saved model confidence; not a calibrated metric error probability'}
        if floor_samples is not None and len(floor_samples):
            delta = floor_samples - floor_point
            horizontal = delta - (delta @ normal)[:, None] * normal
            report['nearestSavedFloorSupportNative'] = float(np.linalg.norm(horizontal, axis=1).min())
            report['floorSupportScope'] = 'Distance to stored sampled floor inliers; no new floor fitted here'
        return report

    def direct(delta):
        xy = np.unique(np.rint(_pixels(samples + delta * perpendicular, A)).astype(int), axis=0)
        xy = xy[(xy[:, 0] >= 0) & (xy[:, 0] < mask.shape[1]) & (xy[:, 1] >= 0) & (xy[:, 1] < mask.shape[0])]
        xy = xy[valid[xy[:, 1], xy[:, 0]]]
        if not len(xy):
            return None
        result = record(points[xy[:, 1], xy[:, 0]], _pixels(xy, np.linalg.inv(A)), conf[xy[:, 1], xy[:, 0]])
        result['sampleInsideSavedObjectMask'] = mask[xy[:, 1], xy[:, 0]].tolist()
        result['sourceScope'] = 'Named RGB edge sampled at nearest saved pointmap pixels; mask membership recorded, not presumed. Boundary depth may mix foreground and background.'
        result['quantizationScope'] = 'No change under a sub-grid pixel perturbation can reflect nearest-pixel rounding; it does not establish physical precision.'
        return result

    direct_estimate = direct(0.)
    direct_trials = [direct(delta) for delta in (-1.5, 0., 1.5)]
    direct_heights = [r['heightNative'] for r in direct_trials if r]
    if direct_estimate:
        direct_estimate['pixelPerturbationRangeNative'] = [min(direct_heights), max(direct_heights)]
    else:
        direct_estimate = {'status': 'unsupported', 'reason': 'No valid saved pointmap pixels on the named source edge', 'heightNative': None}

    yy, xx = np.indices(mask.shape)
    grid = _pixels(np.c_[xx.ravel(), yy.ravel()], np.linalg.inv(A)).reshape(*mask.shape, 2)
    relative = grid - raw_ends[0]
    along, across = relative @ tangent, relative @ perpendicular
    footprint = float(np.sqrt(abs(np.linalg.det(np.linalg.inv(A)[:2, :2]))))
    interior = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & valid
    # Source-defined, fixed neighborhood: discard one-pixel boundary mixing,
    # fit the visible same-object interior within six native image pixels.
    side_masks = [interior & (along >= -2 * footprint) & (along <= length + 2 * footprint)
                  & (sign * across >= footprint) & (sign * across <= 6 * footprint) for sign in (-1, 1)]
    support = max(side_masks, key=lambda m: int(m.sum()))
    xyz = points[support]
    plane_estimate = {'status': 'unsupported', 'heightNative': None, 'reason': 'Fewer than six interior source pixels',
                      'supportPixels': int(support.sum()), 'patchSelection': 'One-pixel eroded saved object mask, 1–6 pixel footprints from edge; side with most visible interior support; no surveyed values'}
    if len(xyz) >= 6:
        center = xyz.mean(0)
        _, singular, vectors = np.linalg.svd(xyz - center, full_matrices=False)
        pn = vectors[-1]; pd = -float(center @ pn)
        residual = abs(xyz @ pn + pd)
        if singular[1] > 1e-8:
            trial_results = []
            for delta in (-1.5, 0., 1.5):
                query = samples + delta * perpendicular
                intersections = _intersect(_pixels(query, A), frame['K'], frame['pose'], pn, pd)
                if np.isfinite(intersections).all():
                    trial_results.append((delta, record(intersections, query)))
            central = next((r for delta, r in trial_results if delta == 0), None)
            if central:
                plane_estimate.update(central)
                plane_estimate['reason'] = 'Conditional local surface intersection, not an independently validated physical endpoint'
                plane_estimate['pixelPerturbationRangeNative'] = [min(r['heightNative'] for _, r in trial_results), max(r['heightNative'] for _, r in trial_results)]
        plane_estimate.update(planeNormalNative=pn.tolist(), planeOffsetNative=pd,
                              planeResidualP95Native=float(np.percentile(residual, 95)),
                              singularValuesNative=singular.tolist(), supportPointsNative=xyz.tolist(),
                              planaritySmallestToMiddleSingular=float(singular[2] / singular[1]) if singular[1] > 0 else None,
                              supportRawPixels=grid[support].tolist(),
                              depthConfidenceRaw={'min': float(conf[support].min()), 'median': float(np.median(conf[support])), 'max': float(conf[support].max())})
    return {'rawEnds': raw_ends.tolist(), 'directPointmap': direct_estimate, 'localSurface': plane_estimate,
            'rangeMeaning': '±1.5 raw-pixel transverse perturbation only, plus separately reported along-edge variation. Excludes camera/scale, depth bias and floor-systematic errors.',
            'canonicalPixelFootprintRaw': footprint, 'ground': {'normal': normal.tolist(), 'offset': offset},
            'sceneTransformNative': T.tolist()}


def build(root, out):
    from workcell_photo_report import _observation_point_mask

    root, out = Path(root), Path(out)
    physical = json.loads((root / 'physical-clearances.json').read_text())
    catalog = json.loads((root / 'objects.json').read_text())['objects']
    raw = _frame(root, 4)
    points = _array(raw['pts3d'])
    frame = {'points': points, 'conf': _array(raw['conf']),
             'valid': _array(raw['non_ambiguous_mask']).astype(bool) & np.isfinite(points).all(2),
             'A': np.array(raw['input_mask_transform']['input_to_canonical_pixel_centres']),
             'K': _array(raw['intrinsics']), 'pose': _array(raw['camera_poses'])}
    floor_samples = np.array([s['pointNative'] for view in physical['ground']['sourceSupport'] for s in view['samples'] if s['planeInlier']])
    light = next(x for x in physical['diagnostics']['objectEdges'] if x['id'] == 'post-box-1')
    light_edge = next(x for x in light['bottomCandidates'] if x['photo'] == 4 and x['sourceSegmentIndex'] == 319)
    fence = next(x for x in physical['objects'] if x['id'] == 'fence-0')
    fence_edge = next(x for x in fence['bottomEdge']['observations'] if x['photo'] == 4)
    results = []
    for ident, edge in [('post-box-1', light_edge), ('fence-0', fence_edge)]:
        item = next(x for x in catalog if x['id'] == ident)
        observation = next(x for x in item['observations'] if x['photo'] == 4)
        mask = _observation_point_mask(observation, points.shape[:2])
        estimate = measure_edge(frame, mask, edge['rawEnds'], physical['ground'], floor_samples=floor_samples)
        results.append({'objectId': ident, 'photo': 4, 'sourceSelection': edge, **estimate})
    comparisons = {}
    for method in ('directPointmap', 'localSurface'):
        light_height, fence_height = (r[method]['heightNative'] for r in results)
        difference = light_height - fence_height if light_height is not None and fence_height is not None else None
        comparisons[method] = {'lightMinusFenceNative': difference,
                               'sign': 'light_higher' if difference is not None and difference > 0 else 'light_lower' if difference is not None and difference < 0 else 'equal' if difference == 0 else 'unsupported'}
    artifact = {'schemaVersion': 1, 'status': 'conditional_single_photo', 'mPerNative': None,
                'scope': 'Named visible terminal edges in Photo4, predicted pointmap or local interior surface, same saved floor. Source observations selected before comparison. No surveyed clearance values enter this estimator.',
                'sourceFiles': {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in ('frame_0004.json.gz', 'objects.json', 'physical-clearances.json')},
                'groundResidualP95Native': physical['ground']['residualP95Native'],
                'objects': results, 'comparison': comparisons}
    out.mkdir(parents=True, exist_ok=False)
    (out / 'endpoint-estimate.json').write_text(json.dumps(artifact, indent=2, allow_nan=False) + '\n')
    return artifact


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = build(args.root, args.out)
    print(json.dumps({'estimates': [{r['objectId']: {m: r[m]['heightNative'] for m in ('directPointmap', 'localSurface')}} for r in result['objects']], 'comparison': result['comparison']}))
