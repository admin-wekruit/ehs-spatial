"""Pure scene correspondence audit. Metadata links do not verify stored geometry."""
from .contracts import PlatformError, digest, validate_transform
from .identity import MODEL_KINDS, model_family, validate_part_relations
from .spatial import primitive_mesh


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
            'category': 'model' if available else 'reference' if reference_ids and not errors else 'unresolved',
            'referenceRepresentationIds': reference_ids, 'sourceErrors': errors, 'unmodeledReasons': reasons,
            'cad': {'status': 'unvalidated' if caches else 'absent', 'representationIds': caches,
                'reason': 'cache_metadata_requires_renderer_validation' if caches else 'no_current_projection_cache'}})

    by_id = {row['entityId']: row for row in rows}
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
