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
from workcell_photo_metrology import _fence_plane_index
from workcell_photo_oneshot import _array, _frame


MEASURED_RAIL = 'measured lower rail member'
# A lower-envelope member is placed at the 1% height of all detected member endpoints of its plane: not a rail edge.
# Image evidence (docs/workcell-photo/evidence-2026-10-03/rail-identity) puts it on the left tube face, 48 px above its lower edge.
RAIL_PART = {'observed lower-envelope hypothesis': 'lower_envelope_hypothesis', 'lower-rail continuation': 'lower_edge', MEASURED_RAIL: 'lower_edge'}
HYPOTHESIS_REASON = ('The rail point is the observed lower-envelope hypothesis (1% height of detected member endpoints), '
                     'not a measured lower-rail edge; it is not compared with measured points')


def _bottom_face(scene, node, normal):
    transform, name = scene.graph[node]
    mesh = scene.geometry[name].copy()
    mesh.apply_transform(transform)
    alignment = mesh.face_normals @ normal
    faces = np.flatnonzero(alignment <= alignment.min() + 1e-6)
    return mesh.vertices[np.unique(mesh.faces[faces])]


LIGHT_KIND = 'yellow safety post'
FENCE_KIND = 'safety fence'
# A rail is "beside" a curtain only within this fraction of the curtain model's own
# vertical extent (scale-free); a farther rail is not paired, never silently compared.
ADJACENT_FRACTION = .25
PROVENANCE = {'physicalBottom': 'Lowest vertex of the selected visible-face terminal read from the accepted conditional GLB; same inferred floor; whole-housing minimum unverified',
              'modelTerminal': 'Lowest vertex of the visible lower edge read from an unaccepted visual candidate GLB (strict source-face gate failed); same inferred floor; not a physical measurement'}


def terminal_binding(item):
    """Exact model-vertex terminal of an installed housing: accepted or candidate."""
    return item.get('physicalBottom') or item.get('modelTerminal')


def _binding_kind(item):
    return 'physicalBottom' if item.get('physicalBottom') else 'modelTerminal'


def _centerline(face):
    """End-edge midpoints of an elongated bottom face; both lie on the mesh surface."""
    center = face.mean(0)
    _, _, vectors = np.linalg.svd(face - center, full_matrices=False)
    order = np.argsort((face - center) @ vectors[0])
    return np.array([face[order[:2]].mean(0), face[order[-2:]].mean(0)])


def _nearest_on(ends, point, normal):
    direction = ends[1] - ends[0]
    fraction = float(np.clip((point - ends[0]) @ direction / (direction @ direction), 0, 1))
    nearest = ends[0] + fraction * direction
    delta = point - nearest
    return nearest, fraction, float(np.linalg.norm(delta - (delta @ normal) * normal))


def estimate(root):
    """Measure every displayed light curtain and its adjacent lower fence rail.

    The current catalog models, their GLBs and the current floor are the entire
    input; nothing is labelled by object ID. Each curtain pairs with the lower
    rail member horizontally nearest to its terminal. This is a model estimate,
    independent of the photo endpoint diagnostic and of any metric scale.
    """
    root = Path(root)
    ground = json.loads((root / 'physical-clearances.json').read_text())['ground']
    normal = np.asarray(ground['normal'], float)
    norm = np.linalg.norm(normal)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError('Invalid ground normal')
    normal, offset = normal / norm, float(ground['offset']) / norm
    catalog = {item['id']: item for item in json.loads((root / 'objects.json').read_text())['objects']}
    geometry = json.loads((root / 'geometry.json').read_text())
    source_files = {'objects.json', 'geometry.json', 'physical-clearances.json'}
    scenes = {}

    def scene(name):
        if name not in scenes:
            path = (root / name).resolve()
            if root.resolve() not in path.parents or not path.is_file():
                raise ValueError('Measured model is missing: ' + name)
            scenes[name] = trimesh.load(path, force='scene', process=False)
            source_files.add(name)
        return scenes[name]

    def housing_terminal(ident):
        evidence = terminal_binding(catalog[ident])
        model = catalog[ident]['model']; name = model['file']; path = (root / name).resolve()
        if (root.resolve() not in path.parents or evidence['modelFile'] != name or
                hashlib.sha256(path.read_bytes()).hexdigest() != evidence['modelSha256']):
            raise ValueError('Housing terminal model binding is stale: ' + ident)
        points = []
        for ref in evidence['vertices']:
            node, index = ref['node'], ref['vertexIndex']
            if node not in model['nodes'] or type(index) is not int:
                raise ValueError('Housing terminal references another object: ' + ident)
            matrix, mesh_id = scene(name).graph[node]
            vertices = scene(name).geometry[mesh_id].vertices
            if not 0 <= index < len(vertices):
                raise ValueError('Housing terminal vertex is absent: ' + ident)
            points.append(trimesh.transform_points([vertices[index]], matrix)[0])
        points = np.asarray(points)
        if len(points) < 2 or not np.isfinite(points).all():
            raise ValueError('Housing terminal has no finite exported edge: ' + ident)
        return points[np.argmin(points @ normal)], points, evidence

    T = trimesh.geometry.align_vectors(normal, [0, 0, 1]); T[2, 3] = offset

    def record(ident, model, node, point, face, provenance):
        height = float(point @ normal + offset)
        foot = point - height * normal
        heights = face @ normal + offset
        return {'objectId': ident, 'node': node, 'status': 'conditional_model_estimate',
                'provenance': provenance, 'heightNative': height, 'modelFile': model['file'],
                'modelSha256': hashlib.sha256((root / model['file']).read_bytes()).hexdigest(),
                'pointNative': point.tolist(), 'footNative': foot.tolist(),
                'pointReportNative': trimesh.transform_points([point], T)[0].tolist(),
                'footReportNative': trimesh.transform_points([foot], T)[0].tolist(),
                'bottomFaceVerticesNative': face.tolist(),
                'bottomFaceHeightRangeNative': [float(heights.min()), float(heights.max())],
                'rangeMeaning': 'Actual mesh bottom-face extent, not physical measurement uncertainty'}

    def vertical_extent(model):
        heights = [trimesh.transform_points(scene(model['file']).geometry[mesh_id].vertices, matrix) @ normal
                   for matrix, mesh_id in (scene(model['file']).graph[node] for node in model['nodes'])]
        return float(np.ptp(np.concatenate(heights)))

    lights = []
    for ident in sorted(ident for ident, item in catalog.items() if item.get('kind') == LIGHT_KIND):
        item = catalog[ident]; model = item['model']
        if terminal_binding(item) is not None:
            point, vertices, evidence = housing_terminal(ident)
            light = record(ident, model, evidence['vertices'][0]['node'], point, vertices, PROVENANCE[_binding_kind(item)])
            light['modelEvidence'] = evidence
            light['terminalBinding'] = _binding_kind(item)
            light.update(measurementScope='visible_face_lower_terminal', wholeHousingMinimumVerified=False,
                         terminalPartAmbiguity=evidence['terminalPartAmbiguity'])
        else:
            # Only the untouched display primitive may be read by its bottom face.
            if model.get('file') != 'posts.glb' or len(model.get('nodes', [])) != 1:
                raise ValueError('Selected housing geometry needs an explicit terminal binding: ' + ident)
            face = _bottom_face(scene('posts.glb'), model['nodes'][0], normal)
            light = record(ident, model, model['nodes'][0], face.mean(0), face,
                           'Displayed upright primitive, bottom-face center; source depth percentile envelope')
            light['measurementScope'] = 'model_bottom_face_center'
        light['id'] = ident + ':terminal'
        light['modelVerticalExtentNative'] = vertical_extent(model)
        lights.append(light)

    rails = []
    for ident, item in sorted(catalog.items()):
        if item.get('kind') != FENCE_KIND or not item.get('model'):
            continue
        plane = _fence_plane_index(item)
        roles = {row['id']: row['role'] for row in geometry.get('fence', {}).get('continuations', [])
                 if row.get('plane') == plane and 'lower' in row.get('role', '')}
        roles.update({row['meshNode']: MEASURED_RAIL for row in geometry.get('clearances', [])
                      if row.get('meshNode') and row['id'] == f'fence-plane-{plane}-lower-rail'})
        unknown = sorted({role for role in roles.values()} - set(RAIL_PART))
        if unknown:
            raise ValueError('Unknown lower-member role (add it to RAIL_PART before measuring): ' + ', '.join(unknown))
        for node in sorted(set(roles) & set(item['model']['nodes'])):
            face = _bottom_face(scene(item['model']['file']), node, normal)
            rails.append((ident, item['model'], node, face, _centerline(face), roles[node]))
    objects, pairs, excluded = list(lights), [], []
    for light in lights:
        point = np.asarray(light['pointNative'])
        candidates = [(_nearest_on(ends, point, normal), ident, model, node, face, ends, role)
                      for ident, model, node, face, ends, role in rails]
        limit = ADJACENT_FRACTION * light['modelVerticalExtentNative']
        light['adjacencyLimitNative'] = limit
        if not candidates:
            light.update(pairedEndpointId=None, pairingStatus='no_lower_rail_model')
            continue
        (rail_point, fraction, horizontal), ident, model, node, face, ends, role = min(candidates, key=lambda row: row[0][2])
        if horizontal > limit:
            light.update(pairedEndpointId=None, pairingStatus='no_adjacent_lower_rail', nearestRailHorizontalOffsetNative=horizontal)
            continue
        light['pairingStatus'] = 'paired'
        # The bottom face's longest axis is this rail's length; the clamped
        # projection keeps the measured point on the actual mesh surface.
        fence = record(ident, model, node, rail_point, face,
                       'Displayed lower rail member; nearest point to the paired curtain terminal along its bottom-face centerline')
        fence.update(id=f"{ident}:near:{light['objectId']}", bottomCenterlineEndsNative=ends.tolist(),
                     closestAlongRailFraction=fraction, pairedObjectId=light['objectId'],
                     horizontalOffsetNative=horizontal, measurementScope='model_lower_rail_near_curtain',
                     memberRole=role, railPart=RAIL_PART[role])
        if fence['railPart'] != 'lower_edge':
            fence['provenance'] = ('Displayed lower-envelope hypothesis member (1% height of detected member ends); nearest point to the '
                                   'paired curtain terminal along its bottom-face centerline; not a lower-rail edge')
        light['pairedEndpointId'] = fence['id']
        objects.append(fence)
        pair = {'minuendId': light['id'], 'subtrahendId': fence['id'], 'valueNative': light['heightNative'] - fence['heightNative']}
        if fence['railPart'] == 'lower_edge':
            pairs.append(pair)
        else:  # shown as its own hypothesis point; never differenced against a measured terminal, no value kept
            excluded.append({'minuendId': pair['minuendId'], 'subtrahendId': pair['subtrahendId'], 'reason': HYPOTHESIS_REASON})
    return {'schemaVersion': 3, 'status': 'conditional_model_estimate', 'mPerNative': None,
            'scope': 'Actual displayed model endpoints against the current saved floor; does not certify physical dimensions',
            'ground': {'normal': normal.tolist(), 'offset': offset}, 'sceneTransformNative': T.tolist(),
            'objects': objects, 'curtainMinusRail': pairs, 'excludedCurtainMinusRail': excluded,
            'pairingRule': {'adjacentFraction': ADJACENT_FRACTION,
                            'rule': 'Each curtain pairs with the lower member (rail edge or lower-envelope hypothesis) horizontally nearest its terminal, only within '
                                    'adjacentFraction x the curtain model vertical extent; otherwise it stays unpaired'},
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
