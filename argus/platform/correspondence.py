"""Pure scene correspondence audit. Metadata links do not verify stored geometry."""
from argus.platform.contracts import PlatformError, digest, validate_transform
from argus.platform.identity import MODEL_KINDS, model_family, validate_part_relations
from argus.platform.spatial import primitive_mesh


def model_family_quality_binding(document, entity):
    """Pin the actual current family, including relations and each saved pose."""
    from copy import deepcopy
    assets = {asset['id']: asset for asset in document['assets']}
    observations = {observation['id']: observation for observation in document['observations']}
    members = []
    for member in sorted(model_family(document, entity['id']), key=lambda value: value['id']):
        rep = next((rep for rep in member.get('representations', [])
                    if rep['id'] == member.get('activeModelRepresentationId')), None)
        if rep is None or rep['kind'] not in MODEL_KINDS or rep.get('sourceValidity') == 'stale':
            raise PlatformError('quality_family_model_unavailable', entityId=member['id'])
        pose = member.get('currentModelTransform') or rep['transform']
        validate_transform(pose, {frame['id'] for frame in document['coordinateFrames']})
        if pose['coordinateFrameId'] != rep['coordinateFrameId']:
            raise PlatformError('quality_coordinate_frames_unregistered')
        members.append({'entityId': member['id'], 'parentEntityId': member.get('parentEntityId'),
            'partRelation': deepcopy(member.get('partRelation')), 'representationId': rep['id'],
            'assetId': rep.get('assetId'),
            'assetSha256': assets[rep['assetId']]['sha256'] if rep.get('assetId') else None,
            'assetRecordSha256': digest(assets[rep['assetId']]) if rep.get('assetId') else None,
            'primitive': deepcopy(rep.get('primitive')), 'transform': deepcopy(pose),
            'sourceRefsSha256': digest((rep.get('sourceRefs') or []) + (rep.get('provenance') or {}).get('sourceRefs', [])),
            'observations': [{'observationId':oid, 'sha256':digest(observations[oid])}
                             for oid in sorted(member.get('observationRefs', []))]})
    if len({member['transform']['coordinateFrameId'] for member in members}) != 1:
        raise PlatformError('quality_coordinate_frames_unregistered')
    return members


def model_quality_binding(document, entity, rep, *, transform=None):
    """Snapshot a just-reviewed saved model; only its recording path creates this proof.

    Use the canonical saved pose and asset hash, avoiding float32 mesh and
    matrix/quaternion roundtrip differences from the provider's scoring inputs.
    """
    from copy import deepcopy
    assets = {asset['id']: asset for asset in document['assets']}
    observations = {observation['id']: observation for observation in document['observations']}
    cameras = {camera['id']: camera for camera in document['cameras']}
    quality = rep.get('qualityEvidence') or {}
    if not isinstance(quality, dict):
        raise PlatformError('model_quality_not_accepted')
    geometric, review = quality.get('geometric') or {}, quality.get('shapeReview') or {}
    if not isinstance(geometric, dict) or not isinstance(review, dict):
        raise PlatformError('model_quality_not_accepted')
    own = entity.get('observationRefs') or []
    if (quality.get('status') != 'accepted' or quality.get('entityId') != entity['id']
            or geometric.get('status') != 'observed_consistent' or review.get('status') != 'pass'
            or quality.get('missingEvidence') or review.get('visibleShapeIssues') or review.get('nextAction') != 'none'
            or not own or sorted(review.get('observationIds') or []) != sorted(own)):
        raise PlatformError('model_quality_not_accepted')

    def asset_hash(identity):
        sha = assets.get(identity, {}).get('sha256')
        if not isinstance(sha, str) or len(sha) != 64:
            raise PlatformError('model_quality_source_unavailable')
        return sha

    evidence = quality.get('evidenceRef') or {}
    candidate = quality.get('candidateRef') or {}
    refs = (rep.get('sourceRefs') or []) + (rep.get('qualityReviewRefs') or [])
    if (not isinstance(evidence, dict) or not isinstance(candidate, dict)
            or evidence.get('sha256') != asset_hash(evidence.get('assetId'))
            or candidate.get('sha256') != asset_hash(candidate.get('assetId'))
            or evidence not in refs or candidate not in refs
            or digest({key: value for key, value in quality.items() if key != 'evidenceRef'}) != evidence['sha256']):
        raise PlatformError('model_quality_evidence_mismatch')
    pose = rep['transform'] if transform is None else transform
    validate_transform(pose, {frame['id'] for frame in document['coordinateFrames']})
    if pose['coordinateFrameId'] != rep['coordinateFrameId']:
        raise PlatformError('model_quality_source_mismatch')
    views = []
    scored = geometric.get('perView') or []
    if (not isinstance(scored, list) or any(not isinstance(view, dict) for view in scored)
            or sorted(view.get('observationId', '') for view in scored) != sorted(own)):
        raise PlatformError('model_quality_source_mismatch')
    for oid in sorted(own):
        observation = observations[oid]
        image_id = observation['imageId']
        binding = document['geometryBindings'][image_id]
        camera = cameras[binding['cameraId']]
        if (camera['imageId'] != image_id or camera['coordinateFrameId'] != rep['coordinateFrameId']
                or any(oid in other.get('observationRefs', []) for other in document['entities'] if other['id'] != entity['id'])):
            raise PlatformError('model_quality_source_mismatch')
        score = next(item for item in scored if item['observationId'] == oid)
        view = {'observationId': oid, 'observationRevision': observation['revision'], 'imageId': image_id,
            'coordinateFrameId': camera['coordinateFrameId'], 'maskComplete': observation.get('maskComplete') is True,
            'sourceHashes': {'image': asset_hash(image_id), 'mask': asset_hash(observation.get('maskAssetId')),
                'geometry': asset_hash(binding['geometrySolutionId']),
                'camera': score['sourceHashes']['camera']}}
        if score.get('status') != 'observed_consistent' or any(score.get(key) != value for key, value in view.items()):
            raise PlatformError('model_quality_source_mismatch')
        # Scoring uses the canonical camera; the scene camera retains original
        # pixel intrinsics. Both remain pinned through the geometry asset.
        view['sceneCameraSha256'] = digest(camera)
        views.append(view)
    family = None
    if any(member.get('parentEntityId') == entity['id'] for member in document['entities']):
        family = model_family_quality_binding(document, entity)
        if geometric.get('assessmentScope') != 'parent_family' or quality.get('familyBinding') != family:
            raise PlatformError('model_quality_family_mismatch')
    elif quality.get('familyBinding') is not None or geometric.get('assessmentScope') == 'parent_family':
        raise PlatformError('model_quality_family_mismatch')
    return {'schemaVersion': 1, 'entityId': entity['id'], 'representationId': rep['id'],
        'assetId': rep.get('assetId'), 'assetSha256': asset_hash(rep['assetId']) if rep.get('assetId') else None,
        'assetRecordSha256': digest(assets[rep['assetId']]) if rep.get('assetId') else None,
        'primitiveSha256': digest(rep['primitive']) if rep.get('primitive') is not None else None,
        'transformSnapshot': deepcopy(pose), 'qualityEvidenceSha256': digest(quality), 'views': views,
        'parentEntityId': entity.get('parentEntityId'), 'partRelation': deepcopy(entity.get('partRelation')),
        'familyBinding': family}


def audit_correspondence(document):
    """Report every non-context entity without changing the coverage denominator.

    Current means the declared model and its source links are current metadata;
    it does not certify mesh bytes, shape, physical placement, or CAD contours.
    """
    entities = {item['id']: item for item in document['entities']}
    observations = {item['id']: item for item in document['observations']}
    assets = {item['id']: item for item in document['assets']}
    cameras = {item['id']: item for item in document['cameras']}
    frames = {item['id'] for item in document['coordinateFrames']}
    owners = {}
    for entity in entities.values():
        for oid in entity.get('observationRefs', []):
            owners.setdefault(oid, []).append(entity['id'])
    document_errors = []
    try:
        validate_part_relations(document)
    except PlatformError as error:
        document_errors.append({'code': error.code, **error.params})

    def source_errors(refs, allowed, frame_id=None):
        errors, linked = [], False
        for ref in refs:
            if not isinstance(ref, dict):
                errors.append({'code': 'source_reference_invalid'})
                continue
            oid = ref.get('observationId')
            if oid is not None:
                observation = observations.get(oid)
                if not observation:
                    errors.append({'code': 'source_observation_missing', 'observationId': oid})
                    continue
                if oid not in allowed or len(owners.get(oid, [])) != 1:
                    errors.append({'code': 'source_observation_foreign', 'observationId': oid})
                if ref.get('revision') is None:
                    errors.append({'code': 'source_observation_revision_unrecorded', 'observationId': oid})
                elif ref['revision'] != observation.get('revision'):
                    errors.append({'code': 'source_observation_stale', 'observationId': oid})
                if ref.get('imageId') is not None and ref['imageId'] != observation['imageId']:
                    errors.append({'code': 'source_image_mismatch', 'observationId': oid})
                binding = document.get('geometryBindings', {}).get(observation['imageId']) or {}
                camera = cameras.get(binding.get('cameraId'), {})
                if camera.get('imageId') != observation['imageId']:
                    errors.append({'code': 'source_camera_unbound', 'observationId': oid})
                elif frame_id is not None and camera.get('coordinateFrameId') != frame_id:
                    errors.append({'code': 'source_coordinate_frame_mismatch', 'observationId': oid})
                if ref.get('cameraId') is not None and ref['cameraId'] != camera.get('id'):
                    errors.append({'code': 'source_camera_mismatch', 'observationId': oid})
                for field, aid in [('imageSha256', observation['imageId']), ('maskSha256', observation.get('maskAssetId')),
                                   ('geometrySolutionSha256', binding.get('geometrySolutionId'))]:
                    if field in ref and assets.get(aid, {}).get('sha256') != ref[field]:
                        errors.append({'code': 'source_asset_hash_mismatch', 'observationId': oid, 'field': field})
                linked = True
            if ref.get('assetId') is not None:
                asset = assets.get(ref['assetId'])
                if asset is None:
                    errors.append({'code': 'source_asset_missing', 'assetId': ref['assetId']})
                elif ref.get('sha256') is not None and asset.get('sha256') != ref['sha256']:
                    errors.append({'code': 'source_asset_hash_mismatch', 'assetId': ref['assetId']})
            if ref.get('role') == 'model_artifact':
                # Artifact records name recipes, source meshes or parameters;
                # they never establish a versioned photo observation input.
                if not ref.get('assetId') or oid is not None or ref.get('imageId') is not None:
                    errors.append({'code': 'source_reference_role_invalid'})
                if not ref.get('sha256'):
                    errors.append({'code': 'source_asset_hash_unrecorded', 'assetId': ref.get('assetId')})
            elif 'sourceRecordId' in ref:
                matched = any(source.get('assetId') == ref.get('assetId') and source.get('sourceRecordId') == ref['sourceRecordId'] and
                    all(key not in ref or source.get(key) == ref[key] for key in ('sourceFrameId', 'jsonPointer'))
                    for source_oid in ([oid] if oid is not None else allowed)
                    for source in observations.get(source_oid, {}).get('sourceRefs', []) if isinstance(source, dict))
                if not ref.get('assetId') or not matched:
                    errors.append({'code': 'source_record_link_unvalidated', 'assetId': ref.get('assetId'), 'sourceRecordId': ref['sourceRecordId']})
                elif oid is None:
                    errors.append({'code': 'source_observation_revision_unrecorded', 'assetId': ref['assetId'], 'sourceRecordId': ref['sourceRecordId']})
                linked |= matched
            if oid is None and ref.get('imageId') is not None:
                if not any(observations.get(oid, {}).get('imageId') == ref['imageId'] for oid in allowed):
                    errors.append({'code': 'source_image_unowned', 'imageId': ref['imageId']})
                else:
                    errors.append({'code': 'source_observation_revision_unrecorded', 'imageId': ref['imageId']})
                linked = True
        if not linked:
            errors.append({'code': 'source_observation_link_missing'})
        return errors

    def representation_errors(entity, rep):
        allowed = set(entity.get('observationRefs', []))
        refs = list(rep.get('sourceRefs') or []) + list((rep.get('provenance') or {}).get('sourceRefs') or [])
        # A validated assembly alone is not source ownership. Exact partition
        # provenance must connect the retained source model and derived part.
        placement = rep.get('placementSource') or {}
        if not document_errors and placement.get('type') == 'derived_source_partition':
            root = entity
            while root.get('parentEntityId'):
                root = entities[root['parentEntityId']]
            family = model_family(document, root['id'])
            for event in refs:
                if not isinstance(event, dict) or event.get('operation') != 'partition_model_parts':
                    continue
                source = next((r for item in family for r in item.get('representations', []) if r['id'] == event.get('sourceRepresentationId')), None)
                lineage = any(e.get('operation') == 'partition_model_parts' and e.get('representationId') == rep['id'] and
                    all(e.get(key) == event.get(key) for key in ('manifestSha256', 'sourceRepresentationId', 'sourceAssetId', 'sourceAssetSha256'))
                    for e in entity.get('lineage', []) if isinstance(e, dict))
                if source and lineage and event.get('manifestSha256') and event.get('manifestSha256') == placement.get('manifestSha256') and \
                        event.get('sourceRepresentationId') == placement.get('sourceRepresentationId') and \
                        source.get('assetId') == event.get('sourceAssetId') and assets.get(source.get('assetId'), {}).get('sha256') == event.get('sourceAssetSha256'):
                    allowed.update(oid for item in family for oid in item.get('observationRefs', []))
                    refs = refs + list(event.get('evidenceRefs') or [])
        errors = source_errors(refs, allowed, rep.get('coordinateFrameId'))
        model_errors = []
        if rep.get('sourceValidity') == 'stale':
            model_errors.append({'code': 'stale_representation'})
        pose = (entity.get('currentModelTransform') or rep.get('transform')) if rep.get('kind') in MODEL_KINDS else rep.get('transform')
        try:
            validate_transform(pose, frames)
            if pose['coordinateFrameId'] != rep.get('coordinateFrameId'):
                raise PlatformError('representation_coordinate_frame_mismatch')
            if rep.get('kind') == 'primitive':
                primitive_mesh(rep.get('primitive') or {})
            elif rep.get('assetId') not in assets:
                raise PlatformError('model_asset_missing')
        except PlatformError as error:
            model_errors.append({'code': error.code})
        return [{'representationId': rep['id'], **error} for error in errors + model_errors], not model_errors

    rows = []
    for entity in entities.values():
        if entity.get('sourceContext'):
            continue
        own = entity.get('observationRefs', [])
        observation_rows, errors = [], []
        for oid in own:
            observation = observations.get(oid)
            if observation is None:
                errors.append({'code': 'source_observation_missing', 'observationId': oid})
                continue
            binding = document.get('geometryBindings', {}).get(observation['imageId']) or {}
            camera = cameras.get(binding.get('cameraId'), {})
            observation_rows.append({'observationId': oid, 'revision': observation.get('revision'), 'imageId': observation['imageId'],
                'cameraId': camera.get('id'), 'coordinateFrameId': camera.get('coordinateFrameId')})
            errors.extend(source_errors([{'observationId': oid, 'revision': observation.get('revision')}], set(own)))
            if observation['imageId'] not in assets:
                errors.append({'code': 'source_image_missing', 'observationId': oid})
        active_id = entity.get('activeModelRepresentationId')
        active = next((r for r in entity.get('representations', []) if r['id'] == active_id), None)
        declared = active is not None and active.get('kind') in MODEL_KINDS
        reasons = []
        if active_id is None:
            reasons.append('active_model_not_selected' if any(r.get('kind') in MODEL_KINDS for r in entity.get('representations', [])) else 'no_model_representation')
        elif not declared:
            reasons.append('active_model_missing' if active is None else 'active_model_kind_invalid')
        available = False
        if declared:
            representation_issues, available = representation_errors(entity, active)
            errors.extend(representation_issues)
            if any(event.get('operation') == 'partition_model_parts' and event.get('sourceRepresentationId') == active_id
                   for event in entity.get('lineage', []) if isinstance(event, dict)):
                errors.append({'code': 'part_source_model_inactive', 'representationId': active_id})
                available = False
        current = available and not errors and not document_errors
        quality = (active or {}).get('qualityEvidence') or {}
        quality = quality if isinstance(quality, dict) else {}
        quality_current = False
        if current:
            try:
                quality_current = active.get('qualityBinding') == model_quality_binding(document, entity, active,
                    transform=entity.get('currentModelTransform') or active['transform'])
            except (PlatformError, ValueError, TypeError, KeyError):
                pass
        if declared and not available:
            reasons.extend(sorted({error['code'] for error in errors}))
        references = [r for r in entity.get('representations', []) if r.get('kind') == 'observed_surface' and
            r.get('sourceKind') == 'observed_reference_surface' and r.get('sourceValidity') != 'stale'] if active_id is None else []
        reference_ids = []
        for reference in references:
            reference_errors, _ = representation_errors(entity, reference)
            errors.extend(reference_errors)
            if not reference_errors:
                reference_ids.append(reference['id'])
        # ponytail: cache presence is metadata only; reuse a shared renderer
        # validator if extracted, never duplicate its mesh/pose/reference checks.
        caches = [r['id'] for r in entity.get('representations', []) if r.get('planProjection') is not None and
                  r.get('sourceValidity') != 'stale' and (r.get('kind') not in MODEL_KINDS or r['id'] == active_id)]
        rows.append({'entityId': entity['id'], 'parentEntityId': entity.get('parentEntityId'), 'observationRefs': observation_rows,
            'activeModelRepresentationId': active_id, 'modelDeclared': declared, 'modelCurrent': current,
            'modelAvailable': available,
            'placementState': (active or {}).get('placementState'), 'shapeStatus': (active or {}).get('shapeStatus'),
            'qualityStatus': quality.get('status', 'unreviewed'),
            'qualityCurrent': quality_current,
            'category': 'model' if available else 'reference' if reference_ids and not errors else 'unresolved',
            'referenceRepresentationIds': reference_ids, 'sourceErrors': errors, 'unmodeledReasons': reasons,
            'cad': {'status': 'unvalidated' if caches else 'absent', 'representationIds': caches,
                'reason': 'cache_metadata_requires_renderer_validation' if caches else 'no_current_projection_cache'}})

    by_id = {row['entityId']: row for row in rows}
    for row in rows:
        if row['qualityCurrent']:
            # A parent's union proof cannot survive a stale/unowned member, even
            # when the parent's own mesh and source observations are unchanged.
            row['qualityCurrent'] = all(by_id.get(member['id'], {}).get('modelCurrent', False)
                for member in model_family(document, row['entityId']))
    for row in rows:
        if row['activeModelRepresentationId'] is not None or row['category'] != 'unresolved':
            continue
        mappings = [a for a in document.get('annotations', []) if a.get('kind') == 'observation_component_mapping' and
            a.get('entityId') == row['entityId'] and a.get('representationType') == 'composite_source_evidence' and a.get('independentObject') is False]
        if len(mappings) != 1:
            continue
        mapping = mappings[0]
        boundary_errors = source_errors([{'observationId': mapping.get('unresolvedBoundaryObservationId'),
            'revision': mapping.get('unresolvedBoundaryObservationRevision')}], set(entities[row['entityId']].get('observationRefs', [])))
        targets = mapping.get('targets') or []
        valid = bool(targets) and not boundary_errors and not row['sourceErrors']
        for target in targets:
            owner = by_id.get(target.get('entityId'), {})
            valid &= owner.get('modelAvailable', False) and owner.get('activeModelRepresentationId') == target.get('activeModelRepresentationId') and \
                not source_errors([{'observationId': target.get('observationId'), 'revision': target.get('observationRevision')}],
                    set(entities.get(target.get('entityId'), {}).get('observationRefs', [])))
        if valid:
            row.update(category='composite', compositeAnnotationId=mapping['id'],
                compositeTargetEntityIds=list(dict.fromkeys(t['entityId'] for t in targets)), unmodeledReasons=['composite_evidence_not_independent_model'])
        else:
            row['sourceErrors'].append({'code': 'composite_source_invalid', 'annotationId': mapping['id']})
    for row in rows:
        row['sourceCorrespondenceStatus'] = 'current' if row['observationRefs'] and not row['sourceErrors'] and not document_errors else 'unvalidated'
    return {'schemaVersion': 1, 'documentSha256': digest(document), 'rows': rows, 'documentErrors': document_errors,
        'assetBytesVerified': False, 'scope': 'Current metadata correspondence only; geometry, CAD, shape and placement are not certified.',
        'summary': {'entityCount': len(entities), 'coverageDenominator': len(rows), 'currentSourceModelCount': sum(r['modelCurrent'] for r in rows),
            **{category + 'Count': sum(r['category'] == category for r in rows) for category in ('model', 'reference', 'composite', 'unresolved')},
            'sourceErrorCount': sum(len(r['sourceErrors']) for r in rows)}}


def validate_cad_correspondence(document, stages):
    """Read-only CPU proof of current CAD caches against byte-verified mesh topology.

    This verifies projected contours, not semantic shape or physical calibration.
    Model CAD and observed CAD retain separate coverage denominators.
    """
    from copy import deepcopy
    import numpy as np
    from shapely import LineString, Polygon, union_all
    from shapely.errors import GEOSException
    from argus.platform.blender_export import mesh_from_asset
    from argus.platform.reconstruction import _plan_plane, _plan_projection, _scene_asset_bytes

    result = audit_correspondence(document)
    entities = {entity['id']: entity for entity in document['entities']}
    observations = {observation['id']: observation for observation in document['observations']}
    assets = {asset['id']: asset for asset in document['assets']}
    cameras = {camera['id']: camera for camera in document['cameras']}
    frames = {frame['id'] for frame in document['coordinateFrames']}

    def geometry(projection):
        def points(value, ring=False):
            array = np.asarray(value)
            if (array.ndim != 2 or array.shape[1] != 2 or len(array) < (4 if ring else 2)
                    or array.dtype.kind not in 'ifu' or not np.isfinite(array).all()
                    or ring and not np.array_equal(array[0], array[-1])):
                raise PlatformError('projection_geometry_invalid')
            return array
        polygons = [Polygon(points(p['exterior'], True), [points(h, True) for h in p['holes']])
                    for p in projection['polygons']]
        lines = [LineString(points(line)) for line in projection['lines']]
        if any(not item.is_valid or item.is_empty for item in polygons + lines):
            raise PlatformError('projection_geometry_invalid')
        if not polygons and not lines:
            raise PlatformError('projection_geometry_empty')
        return union_all(polygons), union_all(lines)

    def validate(entity, row, rep):
        proof = {'representationId': rep['id'], 'status': 'invalid', 'reasons': [], 'meshBytesVerified': False}
        try:
            modeled = rep['kind'] in MODEL_KINDS
            pose = (entity.get('currentModelTransform') or rep['transform']) if modeled else rep['transform']
            validate_transform(pose, frames)
            if pose['coordinateFrameId'] != rep['coordinateFrameId']:
                raise PlatformError('projection_frame_mismatch')
            if rep.get('sourceValidity') == 'stale':
                raise PlatformError('stale_representation')
            if _plan_plane(document, rep['coordinateFrameId']) is None:
                return {**proof, 'status': 'absent', 'reasons': ['floor_plane_missing']}
            saved = rep.get('planProjection')
            if saved is None:
                return {**proof, 'status': 'absent', 'reasons': ['no_current_projection_cache']}
            snapshot = saved['transformSnapshot']
            validate_transform(snapshot, frames)
            translated = modeled and {**snapshot, 'position': pose['position']} == pose
            if snapshot != pose and not translated:
                raise PlatformError('projection_pose_mismatch')
            if saved.get('coordinateFrameId') != rep['coordinateFrameId']:
                raise PlatformError('projection_frame_mismatch')
            asset = None
            if rep['kind'] == 'primitive':
                mesh = primitive_mesh(rep['primitive'])
                if saved.get('assetId') is not None or saved.get('assetSha256') is not None or saved.get('primitiveSnapshot') != rep['primitive']:
                    raise PlatformError('projection_asset_mismatch')
            else:
                asset = stages.repo.get_asset(rep['assetId'])
                if assets.get(rep['assetId'], {}).get('sha256') != asset['sha256']:
                    raise PlatformError('scene_asset_hash_mismatch')
                payload = _scene_asset_bytes(document, rep['assetId'], stages)
                mesh = mesh_from_asset(payload, {**asset, **asset.get('metadata', {}), **assets[rep['assetId']]})
                proof['meshBytesVerified'] = True
                if saved.get('assetId') != rep['assetId'] or saved.get('assetSha256') != asset['sha256']:
                    raise PlatformError('projection_asset_mismatch')
            current = _plan_projection(document, rep, mesh, asset['sha256'] if asset else None, pose)
            if saved.get('methodVersion') != current['methodVersion']:
                raise PlatformError('projection_method_mismatch')
            plane = np.asarray(saved['nativeToPlane'], dtype=float)
            if (saved.get('groundNormalSnapshot') != current['groundNormalSnapshot'] or plane.shape != (4, 4)
                    or not np.allclose(plane, current['nativeToPlane'], rtol=1e-12, atol=1e-12)):
                raise PlatformError('projection_plane_mismatch')
            if result['documentErrors'] or any(error.get('representationId') in (None, rep['id']) for error in row['sourceErrors']):
                raise PlatformError('source_correspondence_invalid')
            if any(saved[key] != current.get(key) for key in ('observationId', 'observationRevision', 'imageId') if saved.get(key) is not None):
                raise PlatformError('projection_source_mismatch')
            if not modeled:
                observation = observations.get(saved.get('observationId'), {})
                binding = document.get('geometryBindings', {}).get(observation.get('imageId'), {})
                camera = cameras.get(binding.get('cameraId'), {})
                if (not observation or observation['id'] not in entity.get('observationRefs', [])
                        or observation['revision'] != saved.get('observationRevision') or observation['imageId'] != saved.get('imageId')
                        or camera.get('imageId') != observation['imageId'] or camera.get('coordinateFrameId') != rep['coordinateFrameId']
                        or not any(ref.get('observationId') == observation['id'] and ref.get('revision') == observation['revision']
                                   for ref in rep.get('sourceRefs', []))):
                    raise PlatformError('projection_source_mismatch')
                source = {'imageId': observation['imageId'], 'cameraId': camera['id'],
                    'imageSha256': assets.get(observation['imageId'], {}).get('sha256'),
                    'maskSha256': assets.get(observation.get('maskAssetId'), {}).get('sha256'),
                    'geometrySolutionSha256': assets.get(binding.get('geometrySolutionId'), {}).get('sha256')}
                if any(ref[field] != value for ref in rep.get('sourceRefs', []) if ref.get('observationId') == observation['id']
                       for field, value in source.items() if field in ref):
                    raise PlatformError('projection_source_mismatch')
            expected = current if snapshot == pose else _plan_projection(document, rep, mesh, asset['sha256'] if asset else None, snapshot)
            if not all(left.equals(right) for left, right in zip(geometry(saved), geometry(expected), strict=True)):
                raise PlatformError('projection_geometry_mismatch')
            proof.update(status='validated', assetId=rep.get('assetId'), assetSha256=asset['sha256'] if asset else None,
                primitiveSnapshot=deepcopy(rep.get('primitive')), groundNormalSnapshot=deepcopy(current['groundNormalSnapshot']),
                coordinateFrameId=rep['coordinateFrameId'], transformSnapshot=deepcopy(pose), cacheTransformSnapshot=deepcopy(snapshot),
                poseRelation='exact' if snapshot == pose else 'translation', sourceRefsSha256=digest(rep.get('sourceRefs', [])),
                projectionSha256=digest(saved), currentProjectionSha256=digest(current), nativeToPlane=deepcopy(current['nativeToPlane']),
                vertexCount=len(mesh.vertices), triangleCount=len(mesh.faces), polygonCount=len(current['polygons']),
                holeCount=sum(len(p['holes']) for p in current['polygons']), lineCount=len(current['lines']))
        except PlatformError as error:
            proof['reasons'].append(error.code)
        except (ValueError, TypeError, KeyError, IndexError, OSError, GEOSException):
            proof['reasons'].append('projection_geometry_invalid')
        return proof

    for row in result['rows']:
        entity = entities[row['entityId']]
        active = next((rep for rep in entity.get('representations', [])
                       if rep['id'] == entity.get('activeModelRepresentationId') and rep['kind'] in MODEL_KINDS), None)
        model = validate(entity, row, active) if active else {'status': 'absent', 'reasons': ['no_active_model']}
        observed = [validate(entity, row, rep) for rep in entity.get('representations', [])
                    if rep['kind'] == 'observed_surface' and rep.get('sourceValidity') != 'stale']
        selected = [model] if active else [proof for proof in observed if proof['representationId'] in row['referenceRepresentationIds']]
        status = 'validated' if selected and all(proof['status'] == 'validated' for proof in selected) else \
            'invalid' if any(proof['status'] == 'invalid' for proof in selected) else 'absent'
        row['cad'] = {'status': status, 'kind': 'model' if active else 'observed_reference' if selected else 'model',
            'model': model, 'observed': observed, 'representationIds': [proof['representationId'] for proof in selected],
            'reason': None if status == 'validated' else next((reason for proof in selected for reason in proof['reasons']), 'no_active_model')}
    by_id = {row['entityId']: row for row in result['rows']}
    for row in result['rows']:
        if row['category'] == 'composite':
            checks = row['cad']['observed'] + [by_id[identity]['cad']['model'] for identity in row['compositeTargetEntityIds']]
            valid = bool(row['cad']['observed']) and all(proof['status'] == 'validated' for proof in checks)
            row['cad'].update(status='validated' if valid else 'invalid', kind='component_mapping',
                reason=None if valid else 'component_cad_unvalidated', targetEntityIds=row['compositeTargetEntityIds'],
                representationIds=[proof['representationId'] for proof in checks if 'representationId' in proof])
    result.update(cadValidationMethod='byte-verified-indexed-mesh-projection-v1',
        scope='Source metadata and current CAD contour consistency. Semantic shape and physical calibration are not certified.')
    result['summary'].update(validatedModelCadCount=sum(row['cad']['model']['status'] == 'validated' for row in result['rows']),
        validatedObservedCadCount=sum(bool(row['cad']['observed']) and all(p['status'] == 'validated' for p in row['cad']['observed']) for row in result['rows']))
    return result
