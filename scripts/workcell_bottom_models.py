"""Export bottom-fit previews and integrate evidence-checked housing candidates."""
from copy import deepcopy
from fnmatch import fnmatch
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import trimesh

from workcell_endpoint_estimate import _bottom_face
from workcell_photo_metrology import _fence_plane_index


def _world(scene, node):
    transform, name = scene.graph[node]
    return trimesh.transform_points(scene.geometry[name].vertices, transform)


def _bottom(scene, nodes, normal, offset, tolerance):
    points = np.concatenate([_bottom_face(scene, node, normal) for node in nodes])
    heights = points @ normal + offset
    if not np.isfinite(points).all() or np.ptp(heights) > tolerance:
        raise ValueError('Selected physical lower faces are not one horizontal plane')
    return points


def _check_footprint(points, corners, normal, tolerance):
    center = corners.mean(0)
    basis = np.linalg.svd(corners - center, full_matrices=False)[2][:2].T
    if np.max(abs(basis.T @ normal)) > 1e-5:
        raise ValueError('Bottom footprint is not parallel to the fitted ground')
    uv, expected = (points - center) @ basis, (corners - center) @ basis
    # A rectangular bounding footprint may span gaps between observed rails.
    # Its corners describe the model envelope, not additional RGB evidence.
    if not np.allclose(np.r_[uv.min(0), uv.max(0)], np.r_[expected.min(0), expected.max(0)], atol=tolerance, rtol=0):
        raise ValueError('Selected mesh lower-face envelope differs from the fitted input footprint')
    if not np.allclose(points @ normal, corners[0] @ normal, atol=tolerance, rtol=0):
        raise ValueError('Selected mesh lower-face plane differs from the fitted input footprint')


def _appearance_equal(a, b):
    if a.visual.kind != b.visual.kind:
        return False
    if a.visual.kind != 'texture':
        return np.array_equal(a.visual.vertex_colors, b.visual.vertex_colors)
    if not np.allclose(a.visual.uv, b.visual.uv, atol=1e-7, rtol=0):
        return False
    for name in ('color',):
        left, right = a.visual.vertex_attributes.get(name), b.visual.vertex_attributes.get(name)
        if (left is None) != (right is None) or (left is not None and not np.array_equal(left, right)):
            return False
    for name in ('name', 'baseColorFactor', 'metallicFactor', 'roughnessFactor', 'emissiveFactor',
                 'alphaMode', 'alphaCutoff', 'doubleSided', 'baseColorTexture', 'normalTexture',
                 'occlusionTexture', 'emissiveTexture', 'metallicRoughnessTexture'):
        left, right = getattr(a.visual.material, name, None), getattr(b.visual.material, name, None)
        if (left is None) != (right is None):
            return False
        if left is not None and not np.array_equal(np.asarray(left), np.asarray(right)):
            return False
    return True


def _shared_ground(root, supplied):
    geometry = json.loads((root / 'geometry.json').read_text())
    normal = np.asarray(geometry['floor']['normal'], float)
    length = np.linalg.norm(normal)
    if normal.shape != (3,) or not np.isfinite(normal).all() or length <= 0:
        raise ValueError('Housing integration requires a finite shared floor')
    normal, offset = normal / length, float(geometry['floor']['offset']) / length
    other = np.asarray(supplied['normal'], float)
    other_length = np.linalg.norm(other)
    if (other.shape != (3,) or not np.isfinite(other).all() or other_length <= 0 or
            not np.isfinite([offset, supplied['offset']]).all() or
            not np.allclose(other / other_length, normal, atol=1e-9, rtol=0) or
            not np.isclose(supplied['offset'] / other_length, offset, atol=1e-9, rtol=0)):
        raise ValueError('Housing candidate and report must use the same ground')
    return geometry, normal, offset


GENERATED_NAMES = ('entity-*.glb', 'workcell-*.glb')  # report build meshes and scene exports


def _verify_sources(root, source_files, catalog):
    if not {'geometry.json', 'objects.json'} <= source_files.keys():
        raise ValueError('Housing candidate requires frozen geometry and catalog hashes')
    if json.loads((root / 'objects.json').read_text()) != catalog:
        raise ValueError("The catalog argument differs from this run's objects.json")
    for name, digest in source_files.items():
        path = (root / name).resolve()
        if root not in path.parents or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError('Housing source is missing or stale: ' + name)


def _exported_terminal(ident, candidate_root, model, references, normal, offset):
    """Hash-checked exported GLB, its exact nodes, and the referenced terminal vertices."""
    source = (candidate_root / model['file']).resolve()
    if candidate_root not in source.parents or source.suffix != '.glb' or not source.is_file():
        raise ValueError(ident + ': model must be an exported candidate GLB')
    data = source.read_bytes()
    if hashlib.sha256(data).hexdigest() != model.get('sha256'):
        raise ValueError(ident + ': exported model hash differs from the evidence-checked model')
    scene = trimesh.load(source, force='scene', process=False)
    nodes = model['nodes']
    if not nodes or len(set(nodes)) != len(nodes) or set(nodes) != set(scene.graph.nodes_geometry):
        raise ValueError(ident + ': candidate model must contain exactly its named object nodes')
    world = {node: _world(scene, node) for node in nodes}
    if any(not len(points) or not np.isfinite(points).all() for points in world.values()):
        raise ValueError(ident + ': candidate model contains invalid vertices')
    if len(references) < 2 or len({(ref['node'], ref['vertexIndex']) for ref in references}) != len(references):
        raise ValueError(ident + ': distinct exported terminal vertices are required')
    points = []
    for ref in references:
        node, index = ref['node'], ref['vertexIndex']
        if node not in world or type(index) is not int or not 0 <= index < len(world[node]):
            raise ValueError(ident + ': terminal vertex does not belong to the exported model')
        points.append(world[node][index])
    points = np.asarray(points); heights = points @ normal + offset
    if np.ptp(points, axis=0).max() <= 1e-8 or np.min(heights) < 0:
        raise ValueError(ident + ': degenerate terminal or terminal below the shared floor')
    return data, world, points, heights


def install_candidate_models(root, result, candidate_root, catalog):
    """Make a candidate revision use visual housing hypotheses, never promoting them.

    ``result`` is a volume-candidates record whose source face failed or skipped
    the strict housing gate. The run directory must be a copy dedicated to the
    candidate revision: frozen sources, shared floor, exported model hash,
    nodes, terminal vertices and catalog observations are checked exactly as
    for accepted models, and the visible lower edge read back from the GLB must
    equal the recorded height. The binding is ``modelTerminal`` with
    ``acceptedForPhysicalUse`` false; physical clearances are left untouched.
    """
    root, candidate_root = Path(root).resolve(), Path(candidate_root).resolve()
    geometry, normal, offset = _shared_ground(root, result['ground'])
    source_files = result.get('sourceFiles', {})
    _verify_sources(root, source_files, catalog)
    updated = deepcopy(catalog)
    objects = {row['id']: row for row in updated['objects']}
    rows = [row for row in result.get('items', []) if row.get('model')]
    if not rows or len({row['id'] for row in rows}) != len(rows):
        raise ValueError('Candidate models need distinct object identities')
    names = [row['model']['file'] for row in rows]
    referenced = {spec['file'] for item in catalog['objects'] for spec in [item.get('model') or {}, *item.get('modelsByPhoto', {}).values()] if spec.get('file')}
    for name in names:
        # A candidate file must be new: overwriting a shared or existing model would change other objects or revisions,
        # and names the report build or export generates would be overwritten by them.
        if (Path(name).name != name or names.count(name) > 1 or name in referenced or (root / name).exists()
                or any(fnmatch(name, pattern) for pattern in GENERATED_NAMES)):
            raise ValueError('Candidate model file would overwrite an existing, shared or generated file: ' + name)
    files, records = {}, []
    for row in rows:
        ident = row['id']; item = objects.get(ident)
        if item is None or item.get('kind') != 'yellow safety post':
            raise ValueError(ident + ': candidate does not name a catalog light curtain')
        if item.get('physicalBottom') or item.get('modelTerminal'):
            raise ValueError(ident + ': already bound to an accepted physical bottom or an earlier candidate; use a fresh copy of the unaccepted revision')
        if row.get('physicalValidation') != 'none' or row.get('sourceFaceGateAccepted') is not False:
            raise ValueError(ident + ': only explicitly unaccepted visual candidates install here')
        for observation in row['sourceObservations']:
            if not any(original['photo'] == observation['photo'] and original['source'] == observation['source']
                       for original in item['observations']):
                raise ValueError(ident + ': candidate source association differs from the catalog')
        model = row['model']
        references = [{'node': model['nodes'][0], 'vertexIndex': index} for index in row['visibleLowerEdgeVertices']]
        data, world, points, heights = _exported_terminal(ident, candidate_root, model, references, normal, offset)
        if not np.isclose(np.min(heights), row['visibleLowerEdgeHeightNative'], atol=1e-6, rtol=0):
            raise ValueError(ident + ': exported lower edge differs from the recorded candidate height')
        ambiguity = row['sourceFaceFitGate']['terminalPartAmbiguity']
        evidence = {'partId': 'housing_visible_lower_edge', 'modelFile': model['file'], 'modelSha256': model['sha256'],
                    'vertices': references, 'pointsNative': points.tolist(), 'measurementScope': 'visible_face_lower_terminal',
                    'wholeHousingMinimumVerified': False, 'terminalPartAmbiguity': deepcopy(ambiguity),
                    'candidateStatus': row['status'], 'sourceFaceGateAccepted': False, 'acceptedForPhysicalUse': False,
                    'physicalValidation': 'none', 'geometryScope': row['geometryScope'],
                    'thicknessIdentifiable': row.get('thicknessIdentifiable'), 'searchAtBound': row.get('searchAtBound'),
                    'sourceObservations': deepcopy(row['sourceObservations'])}
        files[model['file']] = data
        item['model'] = {'file': model['file'], 'nodes': list(model['nodes'])}
        item['representation'] = row['geometryScope']
        item['modelDimensionsNative'] = np.ptp(np.concatenate(list(world.values())), axis=0).tolist()
        item['modelTerminal'] = evidence
        item['notes'] = [row['geometryScope'], 'Candidate revision only: the source face did not pass the strict cross-view gate; hidden geometry and metric accuracy remain unvalidated.']
        records.append({'id': ident, 'status': 'candidate_model', 'modelFile': model['file'], 'modelSha256': model['sha256'],
                        'heightNative': float(np.min(heights)), 'sourceFaceGateAccepted': False, 'acceptedForPhysicalUse': False})
    manifest = {'schemaVersion': 1, 'status': 'candidate_models_installed', 'modelUpdates': len(files), 'items': records,
                'sourceFiles': source_files, 'ground': {'normal': normal.tolist(), 'offset': offset},
                'physicalValidation': 'none', 'acceptedForPhysicalUse': False, 'mPerNative': None}
    with tempfile.TemporaryDirectory(prefix='.candidate-write-', dir=root) as temporary:
        temporary = Path(temporary)
        for name, data in files.items():
            (temporary / name).write_bytes(data)
        (temporary / 'housing-models.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
        (temporary / 'objects.json').write_text(json.dumps(updated, ensure_ascii=False, indent=2, allow_nan=False))
        # Models first, catalog last: an interruption never leaves the catalog naming an absent model.
        for name in [*files, 'housing-models.json', 'objects.json']:
            (temporary / name).replace(root / name)
    catalog.clear(); catalog.update(updated)
    return manifest


def apply_housing_models(root, result, candidate_root, catalog):
    """Install geometry that passed the strict source-face gate, retaining the exact catalog observations.

    Supported rows need model/file/nodes, geometryScope, physicalValidation=none,
    sourceSurfaceObservations and lowerBoundary(partId, scope, vertices,
    sourceObservations). Surface observations retain every fitted view; the
    terminal subset contains only actually observed bottom segments. Missing
    terminals remain empty, while real side fragments and catalog identity are
    retained. fitGate needs
    accepted, converged, parameterCount, jacobianRank, sourceViews and
    reprojectionByPhoto(photo, rmsRawPx, maxRawPx, thresholdRawPx), and recorded
    sourcePixelTransforms. Each view's preregistered limit is max(3, one saved
    canonical pixel mapped into raw pixels). These admit a conditional model, not surveyed
    dimensions; an open visible face never becomes a complete housing claim.
    """
    root, candidate_root = Path(root).resolve(), Path(candidate_root).resolve()
    geometry, normal, offset = _shared_ground(root, result['ground'])
    source_files = result.get('sourceFiles', {})
    _verify_sources(root, source_files, catalog)
    expected = {'post-box-1', 'post-box-2'}
    items = result.get('items', [])
    if len(items) != 2 or {row['id'] for row in items} != expected:
        raise ValueError('Housing result must account for both distinct light curtains')
    updated = deepcopy(catalog)
    objects = {row['id']: row for row in updated['objects']}
    if not expected <= objects.keys():
        raise ValueError('Housing catalog is missing a source object')
    measurements, files, records = {}, {}, []
    for row in items:
        ident, item = row['id'], objects[row['id']]
        if item.get('modelTerminal'):
            raise ValueError(ident + ': carries an unaccepted candidate; install accepted models on a main-revision copy')
        if row.get('status') == 'unsupported':
            if not row.get('reason'):
                raise ValueError(ident + ': unsupported geometry needs an explicit reason')
            measurements[ident] = {'id': ident, 'status': 'unsupported', 'reason': row['reason']}
            item['measurements']['groundClearance'] = {'valueNative': None, 'status': 'unsupported', 'source': row['reason']}
            records.append({'id': ident, 'status': 'unsupported', 'reason': row['reason'], 'modelUpdated': False})
            continue
        if row.get('status') != 'supported_candidate' or row.get('physicalValidation') != 'none' or not row.get('geometryScope'):
            raise ValueError(ident + ': only explicitly scoped, unvalidated supported candidates can be integrated')
        gate, boundary = row['fitGate'], row['lowerBoundary']
        count, rank = gate.get('parameterCount'), gate.get('jacobianRank')
        if (gate.get('accepted') is not True or gate.get('converged') is not True or
                gate.get('rankSource') != 'source_reprojection_without_priors' or
                type(count) is not int or count <= 0 or type(rank) is not int or rank != count):
            raise ValueError(ident + ': housing fit must converge with full parameter rank')
        surfaces, observations = row['sourceSurfaceObservations'], boundary['sourceObservations']
        photos = [entry['photo'] for entry in surfaces]
        bottom_photos = [entry['photo'] for entry in observations]
        if (boundary.get('partId') != 'housing_lower_terminal' or
                boundary.get('scope') != 'Selected observed face terminal; whole-housing minimum and front-versus-wing identity unverified' or
                len(photos) < 2 or len(set(photos)) != len(photos) or
                any(type(photo) is not int or photo not in (1, 2, 3, 4) for photo in photos) or
                sorted(gate.get('sourceViews', [])) != sorted(photos) or not bottom_photos or
                len(set(bottom_photos)) != len(bottom_photos) or not set(bottom_photos) <= set(photos)):
            raise ValueError(ident + ': distinct surface views and their actual visible-terminal subset are required')
        complete_anchors = 0
        for observation in surfaces:
            if not any(original['photo'] == observation['photo'] and original['source'] == observation.get('source')
                       for original in item['observations']):
                raise ValueError(ident + ': fitted source association differs from the catalog')
            bottom, top = observation['bottomRawSegments'], observation['topRawSegments']
            side_fragments = observation['fullSideSegmentsRaw']
            sides = np.asarray(observation['faceSideEdgesRaw'], float)
            side_ids = observation['faceSideIds']
            if (observation.get('partId') != 'selected_visible_housing_face' or
                    len(side_fragments) != 2 or not all(side_fragments) or not (bottom or top) or
                    sides.shape != (2, 2, 2) or not np.isfinite(sides).all() or np.any(np.linalg.norm(np.diff(sides, axis=1), axis=2) <= 1e-8) or
                    len(side_ids) != 2 or side_ids[0] == side_ids[1]):
                raise ValueError(ident + ': surface correspondence lacks distinct supported sides and a real terminal')
            for segments in [bottom, top, *side_fragments]:
                if not segments:
                    continue
                segments = np.asarray(segments, float)
                if (segments.ndim != 3 or segments.shape[1:] != (2, 2) or not np.isfinite(segments).all() or
                        np.any(np.linalg.norm(np.diff(segments, axis=1), axis=2) <= 1e-8)):
                    raise ValueError(ident + ': observed face boundary contains an invalid source segment')
            named = observation['observedBoundaryNames']
            if ('bottom' in named) != bool(bottom) or ('top' in named) != bool(top):
                raise ValueError(ident + ': missing source terminals must not be fabricated or relabeled')
            complete_anchors += bool(bottom and top)
        if not complete_anchors:
            raise ValueError(ident + ': joint partial surfaces need an observed complete-face anchor')
        if sorted(bottom_photos) != sorted(source['photo'] for source in surfaces if source['bottomRawSegments']):
            raise ValueError(ident + ': terminal evidence must contain exactly the actual bottom-support subset')
        for observation in observations:
            surface = next(source for source in surfaces if source['photo'] == observation['photo'])
            if (observation.get('partId') != boundary['partId'] or observation.get('source') != surface['source'] or
                    observation.get('faceSideIds') != surface['faceSideIds'] or
                    not np.array_equal(observation.get('rawSegments'), surface['bottomRawSegments']) or
                    not np.array_equal(observation.get('faceSideEdgesRaw'), surface['faceSideEdgesRaw'])):
                raise ValueError(ident + ': terminal evidence changed the selected source surface or part')
        if not isinstance(gate.get('terminalPartAmbiguity'), dict) or boundary.get('terminalPartAmbiguity') != gate['terminalPartAmbiguity']:
            raise ValueError(ident + ': terminal-part ambiguity evidence must remain attached to the selected boundary')
        residuals = gate.get('reprojectionByPhoto', [])
        if len(residuals) != len(photos) or sorted(entry['photo'] for entry in residuals) != sorted(photos):
            raise ValueError(ident + ': every source view needs a reprojection check')
        exported_residuals = gate.get('exportedModelReprojectionByPhoto', [])
        if len(exported_residuals) != len(photos) or sorted(entry['photo'] for entry in exported_residuals) != sorted(photos):
            raise ValueError(ident + ': every source view needs an exported-model reprojection check')
        for entry in [*residuals, *exported_residuals]:
            rms, maximum = entry['rmsRawPx'], entry['maxRawPx']
            affine = np.asarray(gate['sourcePixelTransforms'][str(entry['photo'])], float)
            if affine.shape != (3, 3) or not np.isfinite(affine).all() or abs(np.linalg.det(affine)) < 1e-12:
                raise ValueError(ident + ': invalid recorded source pixel transform')
            threshold = max(3., float(np.linalg.norm(np.linalg.inv(affine)[:2, :2], 2)))
            if (not np.isfinite([rms, maximum, entry['thresholdRawPx']]).all() or
                    not np.isclose(entry['thresholdRawPx'], threshold, atol=1e-8, rtol=0) or
                    not 0 <= rms <= maximum <= threshold):
                raise ValueError(ident + ': housing reprojection exceeds its preregistered source-resolution gate')
        held_out = gate.get('leaveOnePhotoOut', [])
        if len(held_out) != len(photos) or sorted(entry['photo'] for entry in held_out) != sorted(photos):
            raise ValueError(ident + ': every source photo needs an actual held-out fit')
        for entry in held_out:
            threshold = 2 * next(source['thresholdRawPx'] for source in residuals if source['photo'] == entry['photo'])
            count, rank = entry.get('parameterCount'), entry.get('jacobianRank')
            if (entry.get('converged') is not True or type(count) is not int or count != gate['parameterCount'] or
                    type(rank) is not int or rank != count or not np.isfinite([entry['maxRawPx'], entry['thresholdRawPx']]).all() or
                    not np.isclose(entry['thresholdRawPx'], threshold, atol=1e-8, rtol=0) or
                    not 0 <= entry['maxRawPx'] <= threshold):
                raise ValueError(ident + ': held-out housing geometry is unsupported by the fixed gate')
        model, references = row['model'], boundary['vertices']
        data, world, points, heights = _exported_terminal(ident, candidate_root, model, references, normal, offset)
        nodes = model['nodes']
        point = points[np.argmin(heights)]; height = float(np.min(heights)); foot = point - height * normal
        filename = ident + '-physical.glb'
        digest = hashlib.sha256(data).hexdigest()
        files[filename] = data
        evidence = {'partId': boundary['partId'], 'modelFile': filename, 'modelSha256': digest,
                    'vertices': deepcopy(references), 'pointsNative': points.tolist(),
                    'sourceObservations': deepcopy(observations), 'fitGate': deepcopy(gate),
                    'sourceSurfaceObservations': deepcopy(surfaces), 'scope': boundary['scope'],
                    'measurementScope': 'visible_face_lower_terminal', 'wholeHousingMinimumVerified': False,
                    'terminalPartAmbiguity': deepcopy(gate['terminalPartAmbiguity']),
                    'physicalValidation': 'none', 'geometryScope': row['geometryScope']}
        measurements[ident] = {'id': ident, 'status': 'conditional', 'pointNative': point.tolist(),
            'footNative': foot.tolist(), 'heightNative': height, 'sourcePhotos': sorted(bottom_photos),
            'surfaceSupportPhotos': sorted(photos), 'measurementScope': 'visible_face_lower_terminal',
            'wholeHousingMinimumVerified': False,
            'rangeNative': None, 'bottomHeightRangeNative': [float(heights.min()), float(heights.max())],
            'source': 'Selected visible-face terminal vertices to the same inferred floor; conditional geometry. Whole-housing minimum remains unverified.',
            'modelEvidence': evidence}
        item['model'] = {'file': filename, 'nodes': list(nodes)}
        item['representation'] = row['geometryScope']
        item['modelDimensionsNative'] = np.ptp(np.concatenate(list(world.values())), axis=0).tolist()
        item['physicalBottom'] = evidence
        item['measurements']['groundClearance'] = {'valueNative': height, 'status': 'conditional-model-estimate',
                                                  'source': 'physicalBottom: selected visible-face terminal; whole-housing minimum unverified; inferred common floor'}
        item['notes'] = [row['geometryScope'], 'Passed the strict source-face gate; conditional model, hidden geometry and metric accuracy remain unvalidated.']
        records.append({'id': ident, 'status': 'conditional_model', 'modelUpdated': True,
                        'modelFile': filename, 'modelSha256': digest, 'heightNative': height,
                        'measurementScope': 'visible_face_lower_terminal', 'wholeHousingMinimumVerified': False,
                        'terminalPartAmbiguity': deepcopy(gate['terminalPartAmbiguity']),
                        'exportVerified': True, 'physicalValidation': 'none'})
    physical = geometry.setdefault('physicalClearances', {})
    physical['ground'] = {**physical.get('ground', {}), 'normal': normal.tolist(), 'offset': offset}
    physical['objects'] = [row for row in physical.get('objects', []) if row['id'] not in expected] + list(measurements.values())
    manifest = {'schemaVersion': 1, 'status': 'conditional_models_applied' if files else 'unsupported',
                'modelUpdates': len(files), 'items': records, 'sourceFiles': source_files,
                'ground': {'normal': normal.tolist(), 'offset': offset}, 'physicalValidation': 'none',
                'acceptedForPhysicalUse': False, 'mPerNative': None}
    # Validate everything before touching the run. New object files preserve the
    # old initializer; subsequent report export reads these exact hashed bytes.
    with tempfile.TemporaryDirectory(prefix='.housing-write-', dir=root) as temporary:
        temporary = Path(temporary)
        for name, data in files.items():
            (temporary / name).write_bytes(data)
        writes = [('geometry.json', geometry), ('physical-clearances.json', physical), ('housing-models.json', manifest), ('objects.json', updated)]
        for name, value in writes:
            (temporary / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + ('' if name == 'objects.json' else '\n'))
        # Models first, catalog last: an interruption never leaves the catalog naming an absent model.
        for name in [*files, *(name for name, _ in writes)]:
            (temporary / name).replace(root / name)
    catalog.clear(); catalog.update(updated)
    from workcell_endpoint_estimate import estimate
    endpoints = estimate(root)
    (root / 'model-endpoint-estimate.json').write_text(json.dumps(endpoints, indent=2, allow_nan=False) + '\n')
    return manifest


def apply_bottom_models(root, fit, items, *, out):
    """Export fitted candidates into a new sibling directory and read them back.

    A single world-space affine acts on every node of each selected section:
    rigidly align its source lower plane to the shared ground, translate the
    footprint, preserve the initialized top height, and fit the bottom height.
    These are explicit display assumptions, not accepted physical geometry.
    """
    root, out = Path(root).resolve(), Path(out).resolve()
    if out.exists() or out == root or root in out.parents or out in root.parents:
        raise ValueError('Preview output must be a new directory outside the source run')
    if fit.get('status') != 'conditional_fit' or fit.get('units') != 'native':
        raise ValueError('Only an explicit conditional native fit can be previewed')
    normal = np.asarray(fit['ground']['normal'], float)
    if normal.shape != (3,) or not np.isfinite(normal).all() or np.linalg.norm(normal) <= 0 or not np.isfinite(fit['ground']['offset']):
        raise ValueError('Fit must provide a finite ground plane')
    norm = np.linalg.norm(normal)
    normal, offset = normal / norm, float(fit['ground']['offset']) / norm
    source_items = {row['id']: row for row in items}
    if len(source_items) != len(items) or not fit.get('items') or len({row['id'] for row in fit['items']}) != len(fit['items']):
        raise ValueError('Source and fitted items require unique identities')
    geometry_path = root / 'geometry.json'
    geometry = json.loads(geometry_path.read_text())
    hashes = {'geometry.json': hashlib.sha256(geometry_path.read_bytes()).hexdigest()}
    originals, scenes, records, changed = {}, {}, [], {}
    for fitted in fit['items']:
        ident = fitted['id']
        if ident not in source_items:
            raise ValueError(f'{ident}: fitted item lacks source model identity')
        item = source_items[ident]
        file = item['modelFile']
        expected_file = 'posts.glb' if ident.startswith('post-box-') else 'fence-fitted.glb' if ident.startswith('fence-') else None
        if file != expected_file:
            raise ValueError(f'{ident}: unexpected source model file')
        if file not in scenes:
            hashes[file] = hashlib.sha256((root / file).read_bytes()).hexdigest()
            originals[file] = trimesh.load(root / file, force='scene', process=False)
            scenes[file] = originals[file].copy()
            for name, mesh in originals[file].geometry.items():
                if mesh.visual.kind == 'texture' and 'color' in mesh.visual.vertex_attributes:
                    scenes[file].geometry[name].visual.vertex_attributes['color'] = mesh.visual.vertex_attributes['color'].copy()
            changed[file] = set()
        scene = scenes[file]
        old, new = np.asarray(item['bottomVerticesNative'], float), np.asarray(fitted['bottomVerticesNative'], float)
        if old.shape != (4, 3) or new.shape != (4, 3) or not np.isfinite([old, new]).all():
            raise ValueError(f'{ident}: bottom corners must be finite 4x3')
        tolerance = 2e-5 * max(1., float(np.linalg.norm(np.ptp(old, axis=0))))
        if not np.allclose(old, fitted['originalBottomVerticesNative'], atol=tolerance, rtol=0):
            raise ValueError(f'{ident}: fit belongs to another source footprint')
        delta = new - old
        if not np.allclose(delta, delta[0], atol=tolerance, rtol=0):
            raise ValueError(f'{ident}: one footprint translation cannot align all four fitted corners')
        old_height, new_height = float(np.mean(old @ normal + offset)), float(fitted['heightNative'])
        if (not np.isfinite(new_height) or new_height < 0 or
                not np.allclose(new @ normal + offset, new_height, atol=tolerance, rtol=0)):
            raise ValueError(f'{ident}: fitted bottom height and corners disagree')
        lower_nodes = list(item['modelNodes'])
        if not lower_nodes or len(set(lower_nodes)) != len(lower_nodes):
            raise ValueError(f'{ident}: distinct source lower-face nodes are required')
        nodes = lower_nodes
        if ident.startswith('fence-'):
            plane = _fence_plane_index(item)
            members = geometry['fence']['beams'] + geometry['fence']['continuations']
            nodes = list(dict.fromkeys(row['id'] for row in members
                                      if row['plane'] == plane and row['id'] in scene.graph.nodes_geometry))
            if not set(lower_nodes) <= set(nodes):
                raise ValueError(f'{ident}: source lower nodes do not belong to the selected fence section')
            original_rail = next((row.get('meshNode') for row in geometry.get('clearances', [])
                                 if row['id'] == f'fence-plane-{plane}-lower-rail'), None)
            if original_rail:
                if original_rail not in nodes:
                    raise ValueError(f'{ident}: the original observed lower rail is absent from this section')
                lower_nodes = list(dict.fromkeys([*lower_nodes, original_rail]))
        if not nodes or any(node not in scene.graph.nodes_geometry for node in nodes) or changed[file].intersection(nodes):
            raise ValueError(f'{ident}: missing or multiply assigned model nodes')
        initializer = np.asarray(item.get('initializerWorldNative'), float)
        if (initializer.shape != (4, 4) or not np.isfinite(initializer).all() or
                not np.allclose(initializer[3], [0., 0., 0., 1.], atol=1e-9) or
                not np.allclose(initializer[:3, :3].T @ initializer[:3, :3], np.eye(3), atol=1e-8) or
                not np.isclose(np.linalg.det(initializer[:3, :3]), 1., atol=1e-8)):
            raise ValueError(f'{ident}: a finite rigid shared-ground initializer is required')
        source_normal = initializer[:3, :3].T @ normal
        source_bottom = np.concatenate([_bottom_face(scene, node, source_normal) for node in lower_nodes])
        bottom = trimesh.transform_points(source_bottom, initializer)
        _check_footprint(bottom, old, normal, tolerance)
        source_points = np.concatenate([_world(scene, node) for node in nodes])
        points = trimesh.transform_points(source_points, initializer)
        original_top = float(np.max(source_points @ normal + offset))
        top_height = float(np.max(points @ normal + offset))
        if top_height <= max(old_height, new_height) + tolerance:
            raise ValueError(f'{ident}: fitted bottom must remain strictly below the preserved top')
        if ident.startswith('post-box-'):
            top_points = points[abs(points @ normal + offset - top_height) <= tolerance]
            _check_footprint(top_points, old + (top_height - old_height) * normal, normal, tolerance)
        stretch = (top_height - new_height) / (top_height - old_height)
        world = np.eye(4)
        world[:3, :3] += (stretch - 1) * np.outer(normal, normal)
        world[:3, 3] = delta[0] - (delta[0] @ normal) * normal + (stretch - 1) * (offset - top_height) * normal
        if not np.allclose(trimesh.transform_points(old, world), new, atol=tolerance, rtol=0):
            raise ValueError(f'{ident}: one upright affine cannot align the fitted bottom corners')
        bottom_affine = world.copy()
        world = bottom_affine @ initializer
        for node in nodes:
            transform, name = scene.graph[node]
            if abs(np.linalg.det(transform[:3, :3])) < 1e-12:
                raise ValueError(f'{ident}: singular source node transform')
            mesh = scene.geometry[name].copy()
            if scene.geometry[name].visual.kind == 'texture' and 'color' in scene.geometry[name].visual.vertex_attributes:
                mesh.visual.vertex_attributes['color'] = scene.geometry[name].visual.vertex_attributes['color'].copy()
            mesh.vertices = trimesh.transform_points(trimesh.transform_points(_world(scene, node), world), np.linalg.inv(transform))
            users = [other for other in scene.graph.nodes_geometry if scene.graph[other][1] == name]
            if len(users) > 1:
                replacement = f'{name}-bottom-{node}'
                if replacement in scene.geometry:
                    raise ValueError('Preview geometry name collides with an existing mesh')
                scene.geometry[replacement] = mesh
                parent = scene.graph.transforms.parents[node]
                edge = dict(scene.graph.transforms.edge_data[(parent, node)])
                edge['geometry'] = replacement
                scene.graph.update(frame_to=node, frame_from=parent, **edge)
            else:
                scene.geometry[name] = mesh
        changed[file].update(nodes)
        records.append({'id': ident, 'sourceModelFile': file, 'candidateModelFile': file.removesuffix('.glb') + '-candidate.glb',
                        'modelNodes': nodes, 'lowerFaceNodes': lower_nodes, 'worldAffineNative': world.tolist(),
                        'initializerWorldNative': initializer.tolist(), 'bottomAffineNative': bottom_affine.tolist(),
                        'initializer': item.get('initializer'),
                        'initializerMaxMovementNative': float(np.max(np.linalg.norm(points - source_points, axis=1))),
                        'originalBottomVerticesNative': trimesh.transform_points(old, np.linalg.inv(initializer)).tolist(),
                        'initializedBottomVerticesNative': old.tolist(), 'bottomVerticesNative': new.tolist(),
                        'originalBottomHeightRangeNative': [float(np.min(source_bottom @ normal + offset)), float(np.max(source_bottom @ normal + offset))],
                        'initializedHeightNative': old_height, 'heightNative': new_height,
                        'originalTopHeightNative': original_top, 'preservedTopHeightNative': top_height,
                        'initializerTopHeightChangeNative': top_height - original_top,
                        'verticalScale': stretch, 'verificationToleranceNative': tolerance})
    manifest = {'schemaVersion': 1, 'status': 'preview_candidate', 'inputFitStatus': fit['status'],
                'baselineModified': False, 'acceptedForPhysicalUse': False, 'physicalValidation': 'none',
                'units': 'native', 'mPerNative': None, 'ground': {'normal': normal.tolist(), 'offset': offset},
                'sharedHeightConstraint': bool(fit.get('sharedHeight')), 'sourceFiles': hashes, 'items': records,
                'assumptions': ['Existing coplanar lower faces are rigidly aligned to the shared inferred ground before bottom fitting; this changes section orientation and may change its top height.',
                    'Initialized footprint and horizontal thickness are retained; the whole section translates in the ground plane.',
                    'Initialized section top height is held fixed; every member is stretched vertically by the same positive affine, including the observed lower rail.',
                    'Top positions move with the footprint; intermediate heights and vertical member thicknesses change. These are display assumptions, not new physical measurements.',
                    'Equal height, when enabled, is an input prior and is not evidence of accuracy.']}
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.bottom-preview-', dir=out.parent) as temporary:
        temporary = Path(temporary)
        for file, scene in scenes.items():
            candidate = file.removesuffix('.glb') + '-candidate.glb'
            scene.export(temporary / candidate)
            loaded = trimesh.load(temporary / candidate, force='scene', process=False)
            if set(loaded.graph.nodes_geometry) != set(scene.graph.nodes_geometry):
                raise ValueError('Candidate export changed model node identities')
            for node in scene.graph.nodes_geometry:
                expected = scene if node in changed[file] else originals[file]
                T, name = expected.graph[node]; saved_T, saved_name = loaded.graph[node]
                if not np.allclose(T, saved_T, atol=1e-9, rtol=0):
                    raise ValueError(f'{node}: node transform changed during export')
                tolerance = 2e-5 * max(1., float(np.linalg.norm(np.ptp(_world(expected, node), axis=0))))
                if not np.allclose(_world(loaded, node), _world(expected, node), atol=tolerance, rtol=0):
                    raise ValueError(f'{node}: exported vertices differ from the intended candidate')
                if not np.array_equal(loaded.geometry[saved_name].faces, expected.geometry[name].faces):
                    raise ValueError(f'{node}: exported topology differs from the intended candidate')
                if not _appearance_equal(expected.geometry[name], loaded.geometry[saved_name]):
                    raise ValueError(f'{node}: exported appearance differs from the intended candidate')
            for record in [row for row in records if row['sourceModelFile'] == file]:
                tolerance = record['verificationToleranceNative']
                bottom = _bottom(loaded, record['lowerFaceNodes'], normal, offset, tolerance)
                _check_footprint(bottom, np.asarray(record['bottomVerticesNative']), normal, tolerance)
                top = max(float(np.max(_world(loaded, node) @ normal + offset)) for node in record['modelNodes'])
                if abs(top - record['preservedTopHeightNative']) > tolerance:
                    raise ValueError('Candidate export moved the preserved section top height')
                record['exportedBottomHeightRangeNative'] = [float(np.min(bottom @ normal + offset)), float(np.max(bottom @ normal + offset))]
                record['exportVerified'] = True
            manifest.setdefault('candidateFiles', {})[candidate] = hashlib.sha256((temporary / candidate).read_bytes()).hexdigest()
        if any(hashlib.sha256((root / file).read_bytes()).hexdigest() != digest for file, digest in hashes.items()):
            raise ValueError('Source files changed during candidate export')
        (temporary / 'bottom-models.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
        temporary.rename(out)
    return manifest
