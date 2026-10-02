"""Fit native bottom faces to visible RGB segments using fixed saved cameras.

No metric reference or evaluation height is read. Geometry and residuals are
conditional on the supplied footprint, ground, cameras and candidate identities.
"""
from copy import deepcopy
from itertools import combinations

import numpy as np
from scipy.optimize import least_squares

from workcell_photo_metrology import _pixels
from workcell_photo_objects import _project


def _prepare(items, frames, ground, shared_height):
    normal = np.asarray(ground['normal'], float)
    norm = np.linalg.norm(normal)
    if normal.shape != (3,) or not np.isfinite(normal).all() or norm <= 0 or not np.isfinite(ground['offset']):
        raise ValueError('A finite nonzero ground plane is required')
    normal, offset = normal / norm, float(ground['offset']) / norm
    axis = np.eye(3)[np.argmin(abs(normal))]
    u = np.cross(normal, axis); u /= np.linalg.norm(u)
    basis = np.column_stack((u, np.cross(normal, u)))
    if not items or len({item['id'] for item in items}) != len(items):
        raise ValueError('At least one uniquely named item is required')
    size = 2 * len(items) + (1 if shared_height else len(items))
    initial = np.zeros(size)
    prepared = []
    for index, item in enumerate(items):
        require_anchor = item.get('requireIdentityAnchor', False)
        if not isinstance(require_anchor, bool):
            raise ValueError(f"{item['id']}: requireIdentityAnchor must be a boolean")
        same_edge = item.get('samePhysicalEdge', False)
        if not isinstance(same_edge, bool):
            raise ValueError(f"{item['id']}: samePhysicalEdge must be a boolean")
        vertices = np.asarray(item['bottomVerticesNative'], float)
        if vertices.shape != (4, 3) or not np.isfinite(vertices).all():
            raise ValueError(f"{item['id']}: bottomVerticesNative must be finite 4x3")
        heights = vertices @ normal + offset
        if np.ptp(heights) > 1e-6 * max(1., np.linalg.norm(np.ptp(vertices, axis=0))) or heights.mean() < 0:
            raise ValueError(f"{item['id']}: original bottom must be planar, parallel to and above ground")
        footprint = vertices - heights.mean() * normal
        xy = (footprint - footprint.mean(0)) @ basis
        order = np.argsort(np.arctan2(xy[:, 1], xy[:, 0]))
        sides = np.roll(xy[order], -1, axis=0) - xy[order]
        turns = sides[:, 0] * np.roll(sides, -1, axis=0)[:, 1] - sides[:, 1] * np.roll(sides, -1, axis=0)[:, 0]
        if np.min(turns) <= 1e-12:
            raise ValueError(f"{item['id']}: footprint must be a convex nondegenerate quadrilateral")
        matrix = np.zeros((3, size)); matrix[:, 2 * index:2 * index + 2] = basis
        hindex = 2 * len(items) + (0 if shared_height else index)
        matrix[:, hindex] = normal
        initial[hindex] += float(heights.mean()) / (len(items) if shared_height else 1)
        views = {}
        for candidate_index, observation in enumerate(item['observations']):
            photo = int(observation['photo'])
            identity = observation.get('identityEvidence', {})
            if not isinstance(identity, dict):
                raise ValueError(f"{item['id']} photo {photo}: identityEvidence must be an object")
            source = observation.get('rawSegments')
            if source is None:
                source = [observation['rawEnds']]
            segments = np.asarray(source, float)
            if segments.ndim != 3 or segments.shape[1:] != (2, 2) or not len(segments) or not np.isfinite(segments).all():
                raise ValueError(f"{item['id']} photo {photo}: invalid observed segments")
            lengths = np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1)
            if np.any(lengths <= 1e-8):
                raise ValueError('Zero-length source segment')
            # ponytail: 17 arc-length samples cover straight visible fragments;
            # gaps contribute no evidence. Curved boundaries need another model.
            stops = np.r_[0., np.cumsum(lengths)]
            positions = np.linspace(0, stops[-1], 17)
            which = np.minimum(np.searchsorted(stops[1:], positions, side='right'), len(segments) - 1)
            fraction = (positions - stops[which]) / lengths[which]
            samples = segments[which, 0] + fraction[:, None] * (segments[which, 1] - segments[which, 0])
            candidate = {'index': candidate_index, 'id': str(observation.get('id', candidate_index)),
                         'segments': segments, 'samples': samples,
                         'identityStatus': 'source_anchor' if identity.get('independentlySupported') is True else 'not_independently_supported',
                         'sourceIdentity': {key: deepcopy(observation[key]) for key in ('identity', 'identityEvidence', 'partIdentity') if key in observation},
                         'score': float(observation['score']) if observation.get('score') is not None else None}
            if candidate['score'] is not None and not np.isfinite(candidate['score']):
                raise ValueError('Non-finite detector score')
            if photo not in views:
                frame = {key: np.asarray(frames[photo][key], float) for key in ('K', 'pose', 'A')}
                if any(frame[key].shape != shape or not np.isfinite(frame[key]).all()
                       for key, shape in [('K', (3, 3)), ('pose', (4, 4)), ('A', (3, 3))]):
                    raise ValueError('Invalid saved camera dimensions or values')
                if abs(np.linalg.det(frame['K'])) < 1e-12 or abs(np.linalg.det(frame['A'])) < 1e-12:
                    raise ValueError('Singular camera intrinsics or pixel affine')
                if not np.allclose(frame['pose'][:3, :3].T @ frame['pose'][:3, :3], np.eye(3), atol=1e-4):
                    raise ValueError('Camera pose must be rigid camera-to-world')
                views[photo] = {'photo': photo, 'frame': frame, 'inverseA': np.linalg.inv(frame['A']), 'candidates': []}
            views[photo]['candidates'].append(candidate)
        if len(views) < 2:
            raise ValueError(f"{item['id']}: at least two source photos are required")
        if require_anchor and not any(candidate['identityStatus'] == 'source_anchor'
                                      for view in views.values() for candidate in view['candidates']):
            raise ValueError(f"{item['id']}: required physical identity anchor is absent from source observations")
        prepared.append({'id': item['id'], 'original': vertices, 'footprint': footprint, 'matrix': matrix,
                         'heightIndex': hindex, 'order': order, 'views': list(views.values()),
                         'requireIdentityAnchor': require_anchor, 'samePhysicalEdge': same_edge})
    return prepared, initial, normal, offset, basis


def _match(vertices, item, view, edge_index=None):
    uv, depth = _project(vertices, view['frame'])
    if not np.isfinite(uv).all() or np.min(depth) <= 1e-8:
        return None, np.full(34, 1e6 + max(0., -float(depth.min())) * 1e3)
    raw = _pixels(uv, view['inverseA'])
    starts = raw[item['order']]
    delta = np.roll(starts, -1, axis=0) - starts
    squared_length = np.sum(delta * delta, axis=1)
    rows = []
    for candidate in view['candidates']:
        difference = candidate['samples'][:, None, :] - starts[None, :, :]
        fraction = np.clip(np.sum(difference * delta[None], axis=2) / np.maximum(squared_length, 1e-12), 0, 1)
        residuals = difference - fraction[..., None] * delta[None]
        # One detected candidate is one straight boundary; its fragments share
        # an edge, rather than switching perimeter edges at individual samples.
        edge = int(np.argmin(np.sum(residuals * residuals, axis=(0, 2)))) if edge_index is None else edge_index
        residual = residuals[:, edge]
        distances = np.linalg.norm(residual, axis=1)
        rows.append((float(np.mean(distances ** 2)), candidate, edge, residual, distances))
    rows.sort(key=lambda row: (row[0], row[1]['index']))
    _, candidate, edge, residual, distances = rows[0]
    evidence = {'photo': view['photo'], 'selectedCandidateIndex': candidate['index'], 'selectedCandidateId': candidate['id'],
                **candidate['sourceIdentity'], 'identityStatus': candidate['identityStatus'],
                'detectorScore': candidate['score'], 'rawSegments': candidate['segments'].tolist(),
                'modelEdgeVertexIndices': [int(item['order'][edge]), int(item['order'][(edge + 1) % 4])],
                'projectedBottomRaw': raw.tolist(), 'rmsRawPx': float(np.sqrt(np.mean(distances ** 2))),
                'p95RawPx': float(np.percentile(distances, 95)), 'maxRawPx': float(distances.max()),
                'candidates': [{'index': row[1]['index'], 'id': row[1]['id'], 'rmsRawPx': float(np.sqrt(row[0])),
                                'identityStatus': row[1]['identityStatus']} for row in rows]}
    return evidence, residual.ravel() / np.sqrt(len(residual))


def _match_views(vertices, item):
    # One edge label is shared by all views of this object. Candidate source
    # fragments still compete within each view; no observed gaps are filled.
    choices = [(edge, [_match(vertices, item, view, edge) for view in item['views']])
               for edge in (range(4) if item['samePhysicalEdge'] else [None])]
    edge, matches = min(choices, key=lambda choice: sum(float(error @ error) for _, error in choice[1]))
    return matches, edge, choices


def fit_bottoms(items, frames, ground, shared_height=False, *, max_starts=24):
    """Return a candidate fit, never an accepted scale or physical measurement.

    items: id, four bottomVerticesNative in any order, observations with photo,
    rawSegments (preferred) or rawEnds; optional id and detector score.
    requireIdentityAnchor=True requires supplied identityEvidence with
    independentlySupported=True and at least one such selected candidate.
    Source identity metadata is retained, not independently verified here.
    samePhysicalEdge=True jointly chooses one rectangle edge across every view
    of that item. It is a supplied physical hypothesis, not an identity test.
    frames: integer photo -> existing K, C2W pose, raw-to-canonical A.
    shared_height constrains every supplied item to the same ground clearance.
    Output corners preserve input vertex order so the caller can update a mesh.
    """
    if not isinstance(max_starts, int) or max_starts < 1:
        raise ValueError('max_starts must be a positive integer')
    prepared, initial, normal, offset, basis = _prepare(items, frames, ground, shared_height)
    hindices = sorted({item['heightIndex'] for item in prepared})
    lower = np.full(len(initial), -np.inf); lower[hindices] = 0.

    def vertices(item, parameters):
        return item['footprint'] + item['matrix'] @ parameters

    def residual(parameters):
        return np.concatenate([error for item in prepared
                               for _, error in _match_views(vertices(item, parameters), item)[0]])

    def clipped(parameters):
        result = parameters.copy(); result[hindices] = np.maximum(result[hindices], 1e-9)
        return result

    seeds = [clipped(initial)]
    seed_groups = []
    # A detected image line backprojects to a plane through its camera. Solve
    # plane/edge-midpoint incidence for starts, then optimize finite support.
    for item in prepared:
        first_seed = len(seeds)
        equations = []
        current = vertices(item, initial)
        for view in item['views']:
            frame = view['frame']
            for candidate in view['candidates']:
                points = _pixels(candidate['samples'], frame['A'])
                center = points.mean(0)
                line_normal = np.linalg.svd(points - center, full_matrices=False)[2][-1]
                line = np.r_[line_normal, -line_normal @ center]
                plane = frame['pose'][:3, :3] @ frame['K'].T @ line
                plane /= np.linalg.norm(plane)
                a = plane @ item['matrix']
                for edge in range(4):
                    midpoint = current[item['order'][[edge, (edge + 1) % 4]]].mean(0)
                    b = -float(plane @ (midpoint - frame['pose'][:3, 3]))
                    equations.append((view['photo'], a, b))
                    seeds.append(clipped(initial + a * b / max(a @ a, 1e-12)))
        # Pair the nearest incidence starts from different views. This bounded
        # search is explicit below; it does not certify the global optimum.
        nearest = sorted(equations, key=lambda row: abs(row[2]))[:24]
        for left, right in combinations(nearest, 2):
            if left[0] != right[0]:
                shift = np.linalg.lstsq(np.array([left[1], right[1]]), [left[2], right[2]], rcond=None)[0]
                seeds.append(clipped(initial + shift))
        def local_cost(seed):
            errors = [error for _, error in _match_views(vertices(item, seed), item)[0]]
            return sum(float(error @ error) for error in errors)
        group = {tuple(np.round(seed, 10)): seed for seed in seeds[first_seed:]}
        seed_groups.append(sorted(group.values(), key=local_cost))
    unique = {tuple(np.round(seed, 10)): seed for seed in seeds}
    # Preserve starts for every object: a global ranking alone can spend the
    # entire budget moving the one whose initial residual happens to be largest.
    combined = initial.copy()
    for index, group in enumerate(seed_groups):
        combined[2 * index:2 * index + 2] = group[0][2 * index:2 * index + 2]
        if not shared_height:
            combined[prepared[index]['heightIndex']] = group[0][prepared[index]['heightIndex']]
    if shared_height:
        combined[hindices[0]] = np.mean([group[0][hindices[0]] for group in seed_groups])
    ranked = [clipped(combined)] + [group[index] for index in range(max_starts)
                                   for group in seed_groups if index < len(group)]
    starts = [clipped(initial)]
    for seed in ranked:
        if len(starts) >= max_starts:
            break
        if all(np.linalg.norm(seed - old) > 1e-7 for old in starts):
            starts.append(seed)
    attempts = []
    solutions = []
    for number, start in enumerate(starts):
        fit = least_squares(residual, start, bounds=(lower, np.inf), max_nfev=250,
                            ftol=1e-9, xtol=1e-9, gtol=1e-9, x_scale='jac')
        mse = float(fit.fun @ fit.fun) / sum(len(item['views']) for item in prepared)
        edge_choices = {}
        for item in prepared:
            if item['samePhysicalEdge']:
                matches, edge, _ = _match_views(vertices(item, fit.x), item)
                edge_choices[item['id']] = {'modelEdgeVertexIndices': item['order'][[edge, (edge + 1) % 4]].tolist(),
                    'selectedCandidateIndicesByPhoto': {str(view['photo']): evidence['selectedCandidateIndex'] if evidence else None
                                                        for view, (evidence, _) in zip(item['views'], matches)}}
        attempts.append({'start': number, 'initialParameters': start.tolist(), 'parameters': fit.x.tolist(),
                         'converged': bool(fit.success), 'termination': str(fit.message), 'nfev': int(fit.nfev),
                         'rmsRawPx': float(np.sqrt(mse)), 'sameEdgeChoicesByObject': edge_choices})
        solutions.append((mse, fit))
    best_index = min(range(len(solutions)), key=lambda index: solutions[index][0])
    mse, best = solutions[best_index]
    singular = np.linalg.svd(best.jac, compute_uv=False)
    rank = int(np.sum(singular > max(singular[0], 1e-12) * 1e-7))
    result_items = []
    for index, item in enumerate(prepared):
        final = vertices(item, best.x)
        before = [evidence for evidence, _ in _match_views(item['original'], item)[0]]
        matches, selected_edge, choices = _match_views(final, item)
        after = [evidence for evidence, _ in matches]
        hypotheses = [{'modelEdgeIndex': edge, 'modelEdgeVertexIndices': item['order'][[edge, (edge + 1) % 4]].tolist(),
                       'rmsRawPx': float(np.sqrt(sum(float(error @ error) for _, error in rows) / len(rows))),
                       'views': [{key: evidence[key] for key in ('photo', 'selectedCandidateIndex', 'selectedCandidateId', 'rmsRawPx', 'candidates')}
                                 if evidence else None for evidence, _ in rows]}
                      for edge, rows in choices if edge is not None]
        anchor_photos = sorted(view['photo'] for view in after if view and view['identityStatus'] == 'source_anchor')
        result_items.append({'id': item['id'], 'originalBottomVerticesNative': item['original'].tolist(),
                             'bottomVerticesNative': final.tolist(), 'translationNative': (final[0] - item['original'][0]).tolist(),
                             'inPlaneTranslationNative': best.x[2 * index:2 * index + 2].tolist(),
                             'originalHeightNative': float(np.mean(item['original'] @ normal + offset)),
                             'heightNative': float(best.x[item['heightIndex']]), 'beforeViews': before, 'views': after,
                             'samePhysicalEdge': item['samePhysicalEdge'], 'selectedModelEdgeIndex': selected_edge,
                             'selectedModelEdgeVertexIndices': item['order'][[selected_edge, (selected_edge + 1) % 4]].tolist() if selected_edge is not None else None,
                             'edgeHypotheses': hypotheses,
                             'edgeHypothesesScope': 'Alternative edge assignments at the final fitted geometry; each row jointly uses one edge for all source views, not a separately optimized geometry.',
                             'requireIdentityAnchor': item['requireIdentityAnchor'], 'identityAnchorPhotos': anchor_photos,
                             'identityAnchorRequirementSatisfied': not item['requireIdentityAnchor'] or bool(anchor_photos)})
    visible = all(view is not None for item in result_items for view in item['views'])
    identity_satisfied = all(item['identityAnchorRequirementSatisfied'] for item in result_items)
    return {'status': 'conditional_fit' if best.success and visible and rank == len(initial) and identity_satisfied else 'unsupported',
            'identityAnchorRequirementsSatisfied': identity_satisfied,
            **({'reason': 'Required physical identity anchor was not selected for: ' + ', '.join(item['id'] for item in result_items if not item['identityAnchorRequirementSatisfied'])} if not identity_satisfied else {}),
            'sharedHeight': bool(shared_height), 'sharedHeightNative': float(best.x[hindices[0]]) if shared_height else None,
            'units': 'native', 'mPerNative': None, 'physicalValidation': 'none', 'items': result_items,
            'ground': {'normal': normal.tolist(), 'offset': offset}, 'inPlaneBasisNative': basis.tolist(),
            'assumptions': ['Saved cameras and ground fixed; ground normal points upward.',
                            'Each existing convex footprint and orientation fixed; bottom translates in ground plane and height.',
                            'Top vertices are not changed by this function; caller must check prism consistency before mesh update.',
                            'Equal height is an input constraint when enabled, not measured agreement.',
                            'Only visible source fragments constrain the fit; unobserved full-edge extent is not evidence.',
                            'samePhysicalEdge is an explicit input hypothesis; when enabled, one rectangle edge is jointly selected across all views and retained for held-out projection.',
                            'Identity anchors are supplied source evidence; this solver does not establish cross-view identity or face-plane correspondence.',
                            'Candidate search and training pixel residuals do not validate physical endpoints or metric accuracy.'],
            'optimizer': {'bestStart': best_index, 'converged': bool(best.success), 'visibleInAllSourceCameras': visible,
                          'jacobianRank': rank, 'parameterCount': len(initial), 'jacobianSingularValues': singular.tolist(),
                          'rmsRawPx': float(np.sqrt(mse)), 'attempts': attempts,
                          'generatedStarts': len(unique), 'evaluatedStarts': len(starts),
                          'searchScope': 'Single-view starts plus pairs among 24 nearest incidence equations per item; per-item residual ranking, round-robin starts and combined seed. Not exhaustive.',
                          'nearBestStarts': [i for i, (error, _) in enumerate(solutions) if np.sqrt(error) <= np.sqrt(mse) + 1.]}}


def leave_one_photo_out(items, frames, ground, shared_height=False, *, max_starts=24):
    """Refit without each photo; diagnose held-out candidate residual and shifts.

    Held-out candidate selection is still data association, not a blind known
    endpoint. At least two remaining views per object and any explicitly
    required source identity anchor must remain in each training fold.
    """
    prepared, _, _, _, _ = _prepare(items, frames, ground, shared_height)
    folds = []
    for photo in sorted({int(row['photo']) for item in items for row in item['observations']}):
        training = [{**item, 'observations': [row for row in item['observations'] if int(row['photo']) != photo]} for item in items]
        if any(len({int(row['photo']) for row in item['observations']}) < 2 for item in training):
            folds.append({'photo': photo, 'status': 'unsupported', 'reason': 'Fewer than two training views for an item'})
            continue
        try:
            fit = fit_bottoms(training, frames, ground, shared_height, max_starts=max_starts)
        except ValueError as error:
            folds.append({'photo': photo, 'status': 'unsupported', 'reason': str(error)})
            continue
        held = []
        for item, result in zip(prepared, fit['items']):
            view = next((view for view in item['views'] if view['photo'] == photo), None)
            if view:
                evidence, _ = _match(np.asarray(result['bottomVerticesNative']), item, view,
                                     result['selectedModelEdgeIndex'] if item['samePhysicalEdge'] else None)
                held.append({'id': item['id'], 'heightNative': result['heightNative'], 'view': evidence})
        folds.append({'photo': photo, 'status': fit['status'], 'fit': fit, 'heldOut': held})
    return {'meaning': 'Leave-one-photo-out candidate consistency; held-out identities unverified; not metric accuracy.', 'folds': folds}
