"""Read-only preparation of a revision-bound RecGen research envelope."""
from copy import deepcopy
import io

import numpy as np
from PIL import Image

from .contracts import PlatformError, canonical, digest
from .recgen import build_payload, source_grid_crop, validate_frozen_source
from .reconstruction import (
    _Stages, _canonical_mask, _capture, _load_geometry, _load_masks, _packed,
    _scene_asset_bytes, providers_from_manifest, validate_research_manifest,
)
from .spatial import project_native


def _original_mask(document, observation, image, record, canonical_mask, stages):
    with Image.open(io.BytesIO(_scene_asset_bytes(document, observation['maskAssetId'], stages))) as png:
        mask = np.asarray(png.convert('L')) > 0
    shape = image['rgb'].shape[:2]
    if tuple(record.get('originalShape', ())) != shape:
        raise PlatformError('recgen_pixel_mapping_invalid', 422)
    mapping = np.asarray(record['inputToCanonical'], dtype=float)
    source = 'original_pixels'
    if mask.shape != shape:
        declared = [item['matrix'] for item in observation.get('pixelMapping', [])
                    if item.get('source') == 'canonical_pixels' and item.get('target') == 'original_pixels']
        if (mask.shape != canonical_mask.shape or len(declared) != 1
                or np.asarray(declared[0]).shape != (3, 3)
                or not np.allclose(mapping @ np.asarray(declared[0]), np.eye(3), atol=1e-6)):
            raise PlatformError('mask_pixel_mapping_missing', 409)
        y, x = np.indices(shape)
        pixels = np.stack((x, y, np.ones_like(x)), axis=-1) @ mapping.T
        nearest = np.floor(pixels[..., :2] + .5).astype(int)
        valid = ((nearest[..., 0] >= 0) & (nearest[..., 0] < mask.shape[1]) &
                 (nearest[..., 1] >= 0) & (nearest[..., 1] < mask.shape[0]))
        original = np.zeros(shape, bool)
        original[valid] = mask[nearest[..., 1][valid], nearest[..., 0][valid]]
        mask, source = original, 'canonical_nearest_upsample'
    if not np.array_equal(_canonical_mask(mask, record), canonical_mask):
        raise PlatformError('mask_geometry_mapping_mismatch', 409)
    return mask, source


def prepare_recgen_input(repository, blobs, job, entity_id, observation_ids,
                         anchor_observation_id, provider_manifest, *, seed=0,
                         mask_erosion_enabled=True):
    """Freeze inputs without registering assets, reserving calls, or invoking models.

    Prerequisite: the server has verified ``job.config.researchPreparation`` and
    removed any client-supplied value. It contains authority, budgetAtPreparation,
    runtimeManifest, callLimits, metricDefinitions, policyThresholds, split and
    purpose. Copying that proof here grants no dispatch authority: submit and the
    research worker still enforce their existing admin, hash and budget gates.
    """
    preparation = deepcopy(job.get('config', {}).get('researchPreparation'))
    required = {'authority', 'budgetAtPreparation', 'runtimeManifest', 'callLimits',
                'metricDefinitions', 'policyThresholds', 'split', 'purpose'}
    if not isinstance(preparation, dict) or not required <= preparation.keys():
        raise PlatformError('frozen_research_protocol_required', 409)
    if (not isinstance(preparation['authority'], dict)
            or preparation['authority'].get('source') != 'database_admin'
            or any(not isinstance(job.get(key), str) or not job[key]
                   for key in ('id', 'projectId', 'branchId', 'baseRevisionId'))):
        raise PlatformError('admin_research_job_required', 403)
    original = repository.get_revision(job['baseRevisionId'])['document']
    if original.get('schemaVersion') != 2:
        raise PlatformError('research_scene_schema_invalid', 409)
    # The frozen revision chooses its captures; UI/job input cannot switch them.
    read_job = {**job, 'kind': 'validate_model', 'inputs': {}}
    _, document, images = _capture(repository, blobs, read_job)
    entity = next((e for e in document['entities'] if e['id'] == entity_id), None)
    observations = {o['id']: o for o in document['observations']}
    if (not entity or entity.get('sourceContext') or not isinstance(observation_ids, list)
            or not observation_ids or any(not isinstance(oid, str) for oid in observation_ids)
            or len(set(observation_ids)) != len(observation_ids) or anchor_observation_id not in observation_ids
            or any(oid not in observations or oid not in entity.get('observationRefs', [])
                   or any(oid in other.get('observationRefs', []) for other in document['entities']
                          if other['id'] != entity_id) for oid in observation_ids)):
        raise PlatformError('recgen_observation_selection_invalid', 422)
    selected_images = {observations[oid]['imageId'] for oid in observation_ids}
    asset_ids = set(selected_images)
    for oid in observation_ids:
        observation = observations[oid]
        asset_ids.update((observation.get('maskAssetId'),
                          (observation.get('maskEvidence') or {}).get('canonicalMaskAssetId'),
                          (document.get('geometryBindings', {}).get(observation['imageId']) or {}).get('geometrySolutionId')))
    historical = document.get('geometryEvidence') or {}
    for record in historical.get('frames', []):
        if record['assets']['input'] in selected_images:
            asset_ids.update(record['assets'].values())
            asset_ids.add(historical.get('manifestAssetId'))
    asset_ids.discard(None)
    assets = {a['id']: a for a in document['assets']}
    frozen_assets = []
    for identity in sorted(asset_ids):
        declared = assets.get(identity)
        if not declared:
            raise PlatformError('research_input_hash_mismatch', 409)
        asset = repository.get_asset(identity)
        if declared.get('sha256') != asset['sha256']:
            raise PlatformError('research_input_hash_mismatch', 409)
        blobs.get(asset['storageKey'], asset['sha256'], asset['sizeBytes'])
        frozen_assets.append({'assetId': identity, 'sha256': asset['sha256']})
    stages = _Stages(repository, blobs, read_job, {})
    selected_document = deepcopy(document)
    selected_document['observations'] = [observations[oid] for oid in observation_ids]
    selected_document['geometryBindings'] = {image: binding for image, binding in document.get('geometryBindings', {}).items()
                                             if image in selected_images}
    frames, records = _load_geometry(selected_document, images, stages)
    masks, mask_errors = _load_masks(selected_document, records, stages)
    image_lookup = {image['id']: image for image in images}
    crops = {}
    for oid in observation_ids:
        observation = observations[oid]
        image_id = observation['imageId']
        if oid not in masks:
            code = next((error['code'] for error in mask_errors if error['observationId'] == oid), 'recgen_mask_required')
            raise PlatformError(code, 409)
        if image_id not in frames or image_id not in records or image_id not in image_lookup:
            raise PlatformError('geometry_evidence_unavailable', 409)
        frame, record = frames[image_id], records[image_id]
        mask, mask_source = _original_mask(document, observation, image_lookup[image_id], record, masks[oid], stages)
        _, depth = project_native(frame.points, frame.camera())
        crop = source_grid_crop(image_lookup[image_id]['rgb'], mask,
                               np.where(frame.support(), depth, 0).astype(np.float32), frame.K,
                               record['inputToCanonical'])
        crop['pixelMapping']['maskSource'] = mask_source
        crops[oid] = crop
    payload = build_payload(document, entity_id, observation_ids, anchor_observation_id,
                            images, frames, records, masks, repository.get_asset, seed=seed,
                            mask_erosion_enabled=mask_erosion_enabled, source_crops=crops)
    packed = _packed(payload)
    frozen_images = [{'id': view['imageId'], 'assetId': view['imageId'], 'sha256': view['imageSha256']}
                     for view in payload['views']]
    manifest = {'generation': deepcopy(provider_manifest['generation'])}
    protocol = {key: preparation[key] for key in ('purpose', 'metricDefinitions', 'policyThresholds',
                                                 'split', 'callLimits', 'runtimeManifest')}
    protocol.update(id=f"recgen:{job['id']}:{entity_id}", baselineRevision=job['baseRevisionId'], entityId=entity_id,
                    observationIds=[view['observationId'] for view in payload['views']],
                    anchorObservationId=anchor_observation_id, maskErosionEnabled=mask_erosion_enabled,
                    inputHashes=[image['sha256'] for image in frozen_images], inputAssetHashes=frozen_assets,
                    payloadSha256=digest(packed), providerManifestSha256=digest(manifest))
    validate_research_manifest(protocol, manifest)
    validate_frozen_source(payload, protocol, original)
    providers_from_manifest(manifest, _research=True)['generation'].validate('generation', research_protocol=protocol)
    frozen = {'schemaVersion': 1, 'projectId': job['projectId'], 'branchId': job['branchId'],
              'baseRevisionId': job['baseRevisionId'], 'baseDocumentSha256': digest(original),
              'authority': preparation['authority'], 'budgetAtPreparation': preparation['budgetAtPreparation'],
              'protocol': protocol, 'providerManifest': manifest, 'payload': packed, 'images': frozen_images}
    # canonical serialization also rejects values unsuitable for an immutable asset.
    canonical(frozen)
    return {'validation': frozen, 'sha256': digest(frozen)}
