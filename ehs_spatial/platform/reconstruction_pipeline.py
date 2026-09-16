"""Durable preparation and attachment around the artifact-only research worker."""
from copy import deepcopy

from .contracts import PlatformError, digest
from .correspondence import audit_correspondence, validate_cad_correspondence
from .recgen import RECGEN_PINS, validate_frozen_source
from .reconstruction import (
    UNKNOWN_OUTCOME_CODES, _Stages, _assess_generation, _capture, _establish_cad_references,
    _generation_targets, _include, _load_geometry, _load_masks, _packed, _ref,
    _record_generated_representation, _refresh_plan_projections, _unpacked, providers_from_manifest,
)
from .research_inputs import prepare_recgen_input
from .spatial import stage_cache_key


def _next(entity_ids, processed):
    return {'kind': 'reconstruct_scene', 'inputs': {'phase': 'prepare',
            'entityIds': entity_ids, 'processed': processed}, 'config': {}} if entity_ids else None


def _result(phase, processed, remaining, *, stages=(), document=None, correspondence=None, **extra):
    result = {'status': 'incomplete' if remaining or any(row['status'] not in ('accepted', 'retained') for row in processed) else 'succeeded',
              'pipelineVersion': 'revision-research-attachment-v1', 'phase': phase,
              'processed': processed, 'remainingEntityIds': remaining, 'stages': list(stages),
              'newModelCalls': sum(row.get('newModelCalls', 0) for row in stages),
              'placementConfirmedEntityIds': [], 'scope': 'research_only', **extra}
    if document is not None:
        audit = result['correspondence'] = correspondence or audit_correspondence(document)
        cad_pending = [row['entityId'] for row in audit['rows'] if row['cad']['status'] != 'validated']
        quality_pending = [row['entityId'] for row in audit['rows']
                           if row['category'] == 'model' and not row.get('qualityCurrent')]
        result.update(cadValidationStatus='incomplete' if cad_pending else 'validated',
                      cadPendingEntityIds=cad_pending, qualityPendingEntityIds=quality_pending)
        if cad_pending or quality_pending or audit['documentErrors'] or audit['summary']['sourceErrorCount'] or audit['summary']['unresolvedCount']:
            result['status'] = 'incomplete'
    return result


def _cached_output(repository, stages, research_job, frozen):
    """Bind cached bytes to the exact protocol/input, including a reused cache."""
    result = research_job.get('result') or {}
    protocol_asset = repository.get_asset(result['protocolAssetId'])
    output_asset = repository.get_asset(result['outputAssetId'])
    if any(asset.get('projectId') != research_job['projectId'] for asset in (protocol_asset, output_asset)):
        raise PlatformError('research_output_mismatch', 409)
    protocol_record = stages.load(protocol_asset)
    if (protocol_asset.get('metadata', {}).get('kind') != 'frozen_research_protocol'
            or protocol_record != {key: frozen[key] for key in ('protocol', 'providerManifest')}):
        raise PlatformError('research_output_mismatch', 409)
    payload = {**_unpacked(frozen['payload']), '_researchProtocol': frozen['protocol']}
    key = stage_cache_key('generation', [{'imageId': image['id'], 'sha256': image['sha256'],
        'pixelMapping': image.get('pixelMapping', [])} for image in frozen['images']],
        frozen['providerManifest']['generation']['pins'], {'payloadSha256': digest(_packed(payload))}, [_ref(protocol_asset)])
    metadata = output_asset.get('metadata', {})
    record = stages.load(output_asset)
    if (metadata.get('kind') != 'stage_cache' or metadata.get('stage') != 'generation'
            or metadata.get('cacheKey') != key or metadata.get('providerPins') != RECGEN_PINS
            or record.get('stage') != 'generation' or record.get('cacheKey') != key
            or not any(stage.get('stage') == 'generation' and stage.get('assetId') == output_asset['id']
                       and stage.get('cacheKey') == key and stage.get('status') in ('succeeded', 'cached')
                       for stage in result.get('stages', []))):
        raise PlatformError('research_output_mismatch', 409)
    response = record.get('output')
    if not isinstance(response, dict) or response.get('providerError'):
        raise PlatformError('research_output_invalid', 409)
    provenance = response.get('provenance') or {}
    request = validate_frozen_source(_unpacked(frozen['payload']), frozen['protocol'],
                                     repository.get_revision(research_job['baseRevisionId'])['document'])
    if (provenance.get('licenseScope') != 'noncommercial_research' or provenance.get('shapeStatus') != 'research_only'
            or provenance.get('pins') != RECGEN_PINS or provenance.get('anchorObservationId') != request.anchor_observation_id
            or provenance.get('coordinateFrameId') != request.views[0].coordinate_frame_id):
        raise PlatformError('research_output_mismatch', 409)
    expected_refs = [{'observationId': view.observation_id, 'revision': view.observation_revision,
                    'imageSha256': view.image_sha256, 'maskSha256': view.mask_sha256,
                    'geometrySolutionSha256': view.geometry_solution_sha256,
                    **({'pixelMapping': view.pixel_mapping} if view.pixel_mapping else {})} for view in request.views]
    if provenance.get('sourceRefs') != expected_refs:
        raise PlatformError('research_output_mismatch', 409)
    return response, output_asset


def run_reconstruction_pipeline(repository, blobs, job, providers=None):
    """Return a result and server-only continuation; the repository owns the CAS.

    The prepare continuation registers immutable bytes only. ``finish_job`` must
    validate their admin authority and live budget before atomically creating the
    research child. This function never submits or repeats a generation call.
    """
    inputs = job.get('inputs', {})
    phase, pending, processed = inputs.get('phase'), inputs.get('entityIds'), deepcopy(inputs.get('processed', []))
    if (job.get('kind') != 'reconstruct_scene' or phase not in ('prepare', 'attach')
            or not isinstance(pending, list) or any(not isinstance(identity, str) or not identity for identity in pending)
            or len(set(pending)) != len(pending) or not isinstance(processed, list)
            or any(not isinstance(row, dict) or not row.get('entityId') or not row.get('status') for row in processed)
            or any(row['entityId'] in pending for row in processed)):
        raise PlatformError('reconstruction_pipeline_invalid', 422)
    pending = list(pending)
    source = repository.get_revision(job['baseRevisionId'])['document']
    if phase == 'prepare':
        rows = {row['entityId']: row for row in audit_correspondence(source)['rows']}
        for index, entity_id in enumerate(pending):
            try:
                _, entities = _generation_targets(source, {'kind': 'generate_scene', 'inputs': {'entityIds': [entity_id]}})
                if rows.get(entity_id, {}).get('category') == 'composite':
                    raise PlatformError('composite_evidence_not_independent_model', 409)
                entity = entities[0]
                observations = {observation['id']: observation for observation in source['observations']}
                ids = entity.get('observationRefs', [])
                if not ids or any(oid not in observations for oid in ids):
                    raise PlatformError('generation_mask_required', 409)
                anchor = max(ids, key=lambda oid: ((observations[oid].get('geometrySupport') or {}).get('validPixelCount', 0), oid))
                prepared = prepare_recgen_input(repository, blobs, job, entity_id, ids, anchor,
                    job['config']['providerManifest'], seed=job['config'].get('seed', 0),
                    mask_erosion_enabled=job['config']['researchPreparation'].get('maskErosionEnabled', True))
            except PlatformError as error:
                retained = error.code in ('generation_reference_surface', 'generation_part_workflow_required',
                    'generation_model_already_present', 'composite_evidence_not_independent_model')
                processed.append({'entityId': entity_id, 'status': 'retained' if retained else 'needs_information', 'reason': error.code})
                continue
            stages = _Stages(repository, blobs, job, {})
            asset = stages.put(prepared['validation'], {'kind': 'recgen_validation_input', 'scope': 'research_only'})
            if asset['sha256'] != prepared['sha256']:
                raise PlatformError('research_input_hash_mismatch', 409)
            remaining = pending[index:]
            pipeline = {'phase': 'attach', 'entityIds': remaining, 'processed': processed,
                        'entityId': entity_id, 'preparationJobId': job['id']}
            continuation = {'kind': 'validate_model', 'inputs': {'validationAssetId': asset['id'], 'validationSha256': asset['sha256']},
                            'config': {'researchProtocolSha256': digest(prepared['validation']['protocol']), 'pipeline': pipeline}}
            return None, _result(phase, processed, remaining, validationAssetId=asset['id'],
                                 validationSha256=asset['sha256']), continuation
        audit = validate_cad_correspondence(source, _Stages(repository, blobs, job, {}))
        return None, _result(phase, processed, [], document=source, correspondence=audit), None

    entity_id = inputs.get('entityId')
    research = repository.get_job(inputs['researchJobId'])
    pipeline = research.get('config', {}).get('pipeline') or {}
    if (research.get('kind') != 'validate_model' or not pending or pending[0] != entity_id
            or any(research.get(key) != job.get(key) for key in ('projectId', 'branchId', 'baseRevisionId'))
            or pipeline.get('phase') != 'attach' or pipeline.get('entityId') != entity_id
            or pipeline.get('entityIds') != pending or pipeline.get('processed') != processed):
        raise PlatformError('research_attachment_context_mismatch', 409)
    if research.get('status') not in ('succeeded', 'incomplete'):
        status = 'outcome_unknown' if research.get('status') == 'outcome_unknown' else 'incomplete'
        return None, _result(phase, processed, pending, status=status,
                            error={'code': 'research_not_completed', 'researchStatus': research.get('status')}), None
    from .reconstruction import load_research_input
    frozen = load_research_input(repository, blobs, research)
    if frozen['protocol']['entityId'] != entity_id:
        raise PlatformError('research_attachment_context_mismatch', 409)
    if providers is None:
        manifest = job.get('config', {}).get('providerManifest', {})
        providers = providers_from_manifest({'model_review': manifest['model_review']}) if 'model_review' in manifest else {}
    review_providers = {key: value for key, value in providers.items() if key == 'model_review'}
    stages = _Stages(repository, blobs, job, review_providers)
    response, evidence = _cached_output(repository, stages, research, frozen)
    _, document, images = _capture(repository, blobs, {**job, 'inputs': {}})
    _, entities = _generation_targets(document, {'kind': 'generate_scene', 'inputs': {'entityIds': [entity_id]}})
    entity, unknown = entities[0], False
    _include(document, evidence)
    try:
        frames, records = _load_geometry(document, images, stages)
        masks, _ = _load_masks(document, records, stages)
        observations = {observation['id']: observation for observation in document['observations']}
        frame_id = response['provenance']['coordinateFrameId']
        response, quality, accepted = _assess_generation(document, entity, observations, response, evidence,
                                                        frames, records, masks, stages, frame_id)
        anchor = frozen['protocol']['anchorObservationId']
        selected = [observations[oid] for oid in [anchor, *[oid for oid in entity['observationRefs'] if oid != anchor]]]
        rep = _record_generated_representation(document, stages, entity, selected, frame_id, response,
                                               evidence, activate=accepted, quality=quality)
        processed.append({'entityId': entity_id, 'status': quality['status'], 'representationId': rep['id'],
                          'qualityEvidenceRef': quality['evidenceRef'], 'researchJobId': research['id']})
        unknown = quality.get('shapeReview', {}).get('reason') in UNKNOWN_OUTCOME_CODES
        _establish_cad_references(document, stages)
        _refresh_plan_projections(document, stages, entity_ids={entity_id})
    except (PlatformError, ValueError, TypeError, KeyError) as error:
        code = error.code if isinstance(error, PlatformError) else 'research_output_invalid'
        if processed and processed[-1].get('entityId') == entity_id:
            processed[-1].update(status='needs_information', reason=code)
        else:
            processed.append({'entityId': entity_id, 'status': 'needs_information', 'reason': code,
                              'researchJobId': research['id'], 'candidateRef': _ref(evidence)})
        unknown |= code in UNKNOWN_OUTCOME_CODES
    remaining = pending[1:]
    result = _result(phase, processed, remaining, stages=stages.records, document=document,
                     correspondence=validate_cad_correspondence(document, stages) if not remaining else None,
                     researchJobId=research['id'], candidateRef=_ref(evidence))
    if unknown:
        result.update(status='incomplete', stoppedReason='provider_outcome_unknown')
    return document, result, None if unknown else _next(remaining, processed)
