"""Typed private RecGen research inputs and native-pose conversion."""
from dataclasses import dataclass
import io
import re

import numpy as np

from argus.platform.contracts import PlatformError
from argus.platform.spatial import affine, camera_intrinsics, matrix_to_transform, MeshData, project_native, transform_points


RECGEN_PINS = {
    'model':'TRI-ML/RecGen',
    'modelRevision':'bc0df7de2e43314830039a35a720731d4c4fac65',
    'codeRevision':'fe3c9315b439c50ada8b60c12b469d739fd722db',
    'dinoCodeRevision':'7764ea0f912e53c92e82eb78a2a1631e92725fc8',
    'dinoCheckpointSha256':'36e4deffbaef061a2576705b0c36f93621e2ae20bf6274694821b0b492551b51',
    'adapter':'recgen-multiview-research-v1',
}
RECGEN_LICENSES = {'recgen_code':'Toyota Research Institute Non-Commercial',
    'recgen_weights':'CC-BY-NC-4.0', 'trellis_base':'MIT', 'dinov2':'Apache-2.0', 'flexicubes':'Apache-2.0'}
RECGEN_WEIGHTS_SHA256 = '78f89f802145ed942d7feab07b790892195ebb5a9c4bdeb7097eb7c245f13a1b'


@dataclass(frozen=True)
class RecGenView:
    observation_id: str
    observation_revision: int
    image_id: str
    image_sha256: str
    mask_sha256: str
    geometry_solution_sha256: str
    coordinate_frame_id: str
    rgb: np.ndarray
    depth: np.ndarray
    mask: np.ndarray
    K: np.ndarray
    camera_to_world: np.ndarray
    pixel_mapping: dict | None = None

    @classmethod
    def from_payload(cls, value, *, mask_erosion_enabled=True):
        fields = ('observationId', 'observationRevision', 'imageId', 'imageSha256', 'maskSha256',
                  'geometrySolutionSha256', 'coordinateFrameId', 'rgb', 'depth', 'mask', 'K', 'cameraToWorld')
        if not isinstance(value, dict) or not set(fields) <= set(value) or set(value) - set(fields) - {'pixelMapping'}:
            raise PlatformError('recgen_view_schema_invalid', 422)
        if any(not isinstance(value[key], str) or not value[key] for key in ('observationId', 'imageId', 'coordinateFrameId')) or type(value['observationRevision']) is not int or value['observationRevision'] < 1:
            raise PlatformError('recgen_view_schema_invalid', 422)
        if any(not re.fullmatch('[0-9a-f]{64}', str(value[key])) for key in ('imageSha256', 'maskSha256', 'geometrySolutionSha256')):
            raise PlatformError('research_input_hash_mismatch', 409)
        rgb, depth, mask = (np.asarray(value[key]) for key in ('rgb', 'depth', 'mask'))
        if rgb.dtype != np.uint8 or depth.dtype != np.float32 or mask.dtype != bool or depth.ndim != 2 or rgb.shape != depth.shape + (3,) or mask.shape != depth.shape or not depth.size:
            raise PlatformError('recgen_view_grid_invalid', 422)
        if not np.isfinite(depth).all() or (depth < 0).any() or (depth > 30).any():
            raise PlatformError('recgen_depth_invalid', 422)
        # Check the exact explicitly selected upstream preprocessing mode. The
        # default stays 5x5; thin structures require a separately frozen opt-out.
        from scipy.ndimage import binary_erosion
        supported_mask = binary_erosion(mask, structure=np.ones((5, 5), bool)) if mask_erosion_enabled else mask
        if not (supported_mask & (depth > 0)).any():
            raise PlatformError('recgen_mask_required', 409)
        camera_intrinsics(value['K'])
        affine(value['cameraToWorld'], rigid=True)
        mapping = value.get('pixelMapping')
        if mapping is not None:
            required = {'source', 'target', 'matrix', 'sourceToCanonical', 'sourceShapeHW', 'canonicalShapeHW', 'sourceCropXYXY', 'maskSource'}
            if not isinstance(mapping, dict) or set(mapping) != required:
                raise PlatformError('recgen_pixel_mapping_invalid', 422)
            matrix = np.asarray(mapping['matrix'])
            source_to_canonical = np.asarray(mapping['sourceToCanonical'])
            box = mapping['sourceCropXYXY']
            if (mapping['source'] != 'original_pixels' or mapping['target'] != 'model_pixels'
                    or mapping['maskSource'] not in ('original_pixels', 'canonical_nearest_upsample')
                    or not isinstance(box, list) or len(box) != 4 or any(type(v) is not int for v in box)
                    or matrix.shape != (3, 3) or not np.array_equal(matrix, [[1,0,-box[0]], [0,1,-box[1]], [0,0,1]])
                    or depth.shape != (box[3]-box[1], box[2]-box[0])
                    or source_to_canonical.shape != (3,3) or not np.isfinite(source_to_canonical).all()
                    or not np.allclose(source_to_canonical[2], [0,0,1]) or abs(np.linalg.det(source_to_canonical)) < 1e-12
                    or any(not isinstance(mapping[k], list) or len(mapping[k]) != 2 or any(type(v) is not int or v <= 0 for v in mapping[k]) for k in ('sourceShapeHW', 'canonicalShapeHW'))):
                raise PlatformError('recgen_pixel_mapping_invalid', 422)
        return cls(*(value[key] for key in fields), pixel_mapping=mapping)


@dataclass(frozen=True)
class RecGenRequest:
    entity_id: str
    anchor_observation_id: str
    views: tuple[RecGenView, ...]
    seed: int
    mask_erosion_enabled: bool | None = None

    @classmethod
    def from_payload(cls, value):
        fields = {'entityId', 'anchorObservationId', 'views', 'seed'}
        if not isinstance(value, dict) or not fields <= set(value) or set(value) - fields - {'_researchProtocol', 'maskErosionEnabled'}:
            raise PlatformError('recgen_request_schema_invalid', 422)
        if 'maskErosionEnabled' in value and type(value['maskErosionEnabled']) is not bool:
            raise PlatformError('recgen_request_schema_invalid', 422)
        if any(not isinstance(value[key], str) or not value[key] for key in ('entityId', 'anchorObservationId')) or type(value['seed']) is not int or not 0 <= value['seed'] < 2 ** 32:
            raise PlatformError('recgen_request_schema_invalid', 422)
        if not isinstance(value['views'], list) or not value['views']:
            raise PlatformError('recgen_observation_selection_invalid', 422)
        views = tuple(RecGenView.from_payload(view, mask_erosion_enabled=value.get('maskErosionEnabled', True)) for view in value['views'])
        if (value['anchorObservationId'] != views[0].observation_id
                or len({v.observation_id for v in views}) != len(views)
                or len({v.image_id for v in views}) != len(views)
                or len({v.coordinate_frame_id for v in views}) != 1):
            raise PlatformError('recgen_observation_selection_invalid', 422)
        return cls(value['entityId'], value['anchorObservationId'], views, value['seed'], value.get('maskErosionEnabled'))

    def to_npz(self):
        arrays = {'view_count':np.array(len(self.views), dtype=np.int32)}
        if self.mask_erosion_enabled is not None:
            arrays['mask_erosion_enabled'] = np.array(self.mask_erosion_enabled, dtype=bool)
        for index, view in enumerate(self.views):
            arrays.update({f'{index}_rgb':view.rgb, f'{index}_depth':view.depth,
                           f'{index}_mask':view.mask.astype(np.uint8) * 255,
                           f'{index}_camera_intrinsics':view.K})
        stream = io.BytesIO()
        np.savez_compressed(stream, **arrays)
        return stream.getvalue()


def source_grid_crop(source_rgb, source_mask, canonical_depth, canonical_K, source_to_canonical):
    """Original pixel crop, from the reviewed RecGen experiment's camera-ray mapping."""
    from PIL import Image
    rgb, mask, depth = np.asarray(source_rgb), np.asarray(source_mask), np.asarray(canonical_depth)
    mapping = np.asarray(source_to_canonical, dtype=float)
    if (rgb.dtype != np.uint8 or mask.dtype != bool or mask.ndim != 2 or rgb.shape != mask.shape + (3,)
            or not mask.any() or depth.dtype != np.float32 or depth.ndim != 2 or not depth.size
            or not np.isfinite(depth).all() or (depth < 0).any() or (depth > 30).any()
            or mapping.shape != (3,3) or not np.isfinite(mapping).all() or not np.allclose(mapping[2], [0,0,1])
            or abs(np.linalg.det(mapping)) < 1e-12):
        raise PlatformError('recgen_crop_input_invalid', 422)
    K = camera_intrinsics(canonical_K)
    y, x = np.nonzero(mask)
    lower, upper = np.array([x.min()-.5, y.min()-.5]), np.array([x.max()+.5, y.max()+.5])
    side = int(np.ceil((upper-lower).max() * 1.4))
    left, top = np.floor((lower+upper-side)/2).astype(int)
    box = [int(left), int(top), int(left+side), int(top+side)]
    yy, xx = np.indices((side, side))
    u, v = xx+left, yy+top
    canonical = np.stack([u, v, np.ones_like(u)], axis=-1) @ mapping.T
    cx, cy = np.floor(canonical[...,0]+.5).astype(int), np.floor(canonical[...,1]+.5).astype(int)
    inside = (u >= 0) & (v >= 0) & (u < rgb.shape[1]) & (v < rgb.shape[0]) & (cx >= 0) & (cy >= 0) & (cx < depth.shape[1]) & (cy < depth.shape[0])
    sampled_depth, sampled_mask = np.zeros((side, side), np.float32), np.zeros((side, side), bool)
    sampled_depth[inside] = depth[cy[inside], cx[inside]]
    sampled_mask[inside] = mask[v[inside], u[inside]]
    crop = np.array([[1.,0,-left], [0,1.,-top], [0,0,1.]])
    return {'rgb':np.asarray(Image.fromarray(rgb).crop(box)), 'depth':sampled_depth, 'mask':sampled_mask,
        'K':crop @ np.linalg.inv(mapping) @ K,
        'pixelMapping':{'source':'original_pixels', 'target':'model_pixels', 'matrix':crop.tolist(),
            'sourceToCanonical':mapping.tolist(), 'sourceShapeHW':list(mask.shape), 'canonicalShapeHW':list(depth.shape),
            'sourceCropXYXY':box, 'maskSource':'original_pixels'}}


def build_payload(document, entity_id, observation_ids, anchor_observation_id, images, frames, records, masks, get_asset, *, seed=0, mask_erosion_enabled=None, source_crops=None):
    """Freeze only explicitly reviewed same-entity observations, with no implicit views."""
    entities = {e['id']:e for e in document['entities']}
    entity = entities.get(entity_id)
    observations = {o['id']:o for o in document['observations']}
    if (not entity or entity.get('sourceContext') or not isinstance(observation_ids, list) or not observation_ids
            or any(not isinstance(oid, str) for oid in observation_ids) or len(set(observation_ids)) != len(observation_ids)
            or anchor_observation_id not in observation_ids
            or any(oid not in entity['observationRefs'] or oid not in observations for oid in observation_ids)
            or any(oid in other.get('observationRefs', []) for oid in observation_ids for other in entities.values() if other['id'] != entity_id)):
        raise PlatformError('recgen_observation_selection_invalid', 422)
    image_lookup = {image['id']:image for image in images}
    ordered = [anchor_observation_id] + [oid for oid in observation_ids if oid != anchor_observation_id]
    if source_crops is not None and (not isinstance(source_crops, dict) or set(source_crops) - set(ordered)):
        raise PlatformError('recgen_observation_selection_invalid', 422)
    views = []
    for oid in ordered:
        observation = observations[oid]
        if not observation.get('maskAssetId') or oid not in masks:
            raise PlatformError('recgen_mask_required', 409)
        image_id = observation['imageId']
        if image_id not in frames or image_id not in records or image_id not in image_lookup:
            raise PlatformError('geometry_evidence_unavailable', 409)
        frame = frames[image_id]
        geometry_id = (document.get('geometryBindings', {}).get(image_id) or {}).get('geometrySolutionId')
        if not geometry_id:
            raise PlatformError('geometry_evidence_unavailable', 409)
        _, depth = project_native(frame.points, frame.camera())
        views.append({'observationId':oid, 'observationRevision':observation['revision'],
            'imageId':image_id, 'imageSha256':image_lookup[image_id]['sha256'],
            'maskSha256':get_asset(observation['maskAssetId'])['sha256'],
            'geometrySolutionSha256':get_asset(geometry_id)['sha256'],
            'coordinateFrameId':frame.coordinate_frame_id, 'rgb':records[image_id]['rgb'],
            'depth':np.where(frame.support(), depth, 0).astype(np.float32), 'mask':masks[oid],
            'K':frame.K, 'cameraToWorld':frame.camera_to_world})
        if source_crops is not None and oid in source_crops:
            crop = source_crops[oid]
            if not isinstance(crop, dict) or set(crop) != {'rgb', 'depth', 'mask', 'K', 'pixelMapping'}:
                raise PlatformError('recgen_crop_input_invalid', 422)
            views[-1].update(crop)
    payload = {'entityId':entity_id, 'anchorObservationId':anchor_observation_id, 'views':views, 'seed':seed}
    if mask_erosion_enabled is not None:
        payload['maskErosionEnabled'] = mask_erosion_enabled
    RecGenRequest.from_payload(payload)
    return payload


def adapt_output(request, response):
    """Retain the raw mesh; compose object-to-camera and anchor-to-native once."""
    if response.get('pins') != RECGEN_PINS:
        raise PlatformError('provider_pins_unverified', 409)
    mesh = MeshData(np.asarray(response['vertices']), np.asarray(response['faces']),
                    np.asarray(response['colors']) if response.get('colors') is not None else None)
    pose = affine(response['objectToCamera'])
    official = np.asarray(response['officialPosedVertices'])
    if official.shape != mesh.vertices.shape or not np.isfinite(official).all():
        raise PlatformError('recgen_pose_mismatch', 409)
    residual = float(np.max(np.abs(transform_points(mesh.vertices, pose) - official)))
    if residual > 2e-5:
        raise PlatformError('recgen_pose_mismatch', 409, residual=residual)
    native = request.views[0].camera_to_world @ pose
    matrix_to_transform(native, request.views[0].coordinate_frame_id)
    return {**response, 'vertices':mesh.vertices, 'faces':mesh.faces, 'colors':mesh.colors,
        'proposedObjectToNative':native, 'provenance':{'shapeStatus':'research_only',
        'licenseScope':'noncommercial_research', 'licenses':RECGEN_LICENSES,
        'metricScaleKnown':False, 'placementState':'unconfirmed', 'pins':RECGEN_PINS,
        **({'maskErosionEnabled':request.mask_erosion_enabled} if request.mask_erosion_enabled is not None else {}),
        'anchorObservationId':request.anchor_observation_id, 'coordinateFrameId':request.views[0].coordinate_frame_id,
        'sourceRefs':[{'observationId':v.observation_id, 'revision':v.observation_revision,
                       'imageSha256':v.image_sha256, 'maskSha256':v.mask_sha256,
                       'geometrySolutionSha256':v.geometry_solution_sha256,
                       **({'pixelMapping':v.pixel_mapping} if v.pixel_mapping else {})} for v in request.views],
        'poseVertexMaxAbsResidual':residual}}


def validate_runtime(protocol, pins):
    runtime = (protocol.get('runtimeManifest') or {}).get('generation') or {}
    fields = {'pins', 'distribution', 'modalFunctionId', 'weightsManifestSha256', 'runtimeAuditSha256'}
    if 'maskErosionEnabled' in protocol and (type(protocol['maskErosionEnabled']) is not bool
            or runtime.get('inputContractVersion') != 'recgen-input-v2'):
        raise PlatformError('research_runtime_unpinned', 409)
    if (not fields <= set(runtime) or set(runtime)-fields-{'inputContractVersion'}
            or ('inputContractVersion' in runtime and runtime['inputContractVersion'] != 'recgen-input-v2')
            or pins != RECGEN_PINS or runtime.get('pins') != pins
            or runtime.get('distribution') != 'recgen_inference'
            or not re.fullmatch('fu-[A-Za-z0-9]+', str(runtime.get('modalFunctionId', '')))
            or runtime.get('weightsManifestSha256') != RECGEN_WEIGHTS_SHA256
            or not re.fullmatch('[0-9a-f]{64}', str(runtime.get('runtimeAuditSha256', '')))):
        raise PlatformError('research_runtime_unpinned', 409)


def validate_frozen_source(payload, protocol, document):
    request = RecGenRequest.from_payload(payload)
    if payload.get('maskErosionEnabled') is not protocol.get('maskErosionEnabled'):
        raise PlatformError('research_input_hash_mismatch', 409)
    if (protocol.get('observationIds') != [v.observation_id for v in request.views]
            or protocol.get('anchorObservationId') != request.anchor_observation_id):
        raise PlatformError('recgen_observation_selection_invalid', 409)
    entities = {e['id']:e for e in document['entities']}
    entity = entities.get(request.entity_id)
    observations = {o['id']:o for o in document['observations']}
    assets = {a['id']:a for a in document['assets']}
    declared = {a['assetId']:a['sha256'] for a in protocol['inputAssetHashes']}
    if not entity or entity.get('sourceContext'):
        raise PlatformError('recgen_observation_selection_invalid', 409)
    for view in request.views:
        observation = observations.get(view.observation_id) or {}
        if (view.observation_id not in entity['observationRefs']
                or any(view.observation_id in other.get('observationRefs', []) for other in entities.values() if other['id'] != request.entity_id)
                or observation.get('revision') != view.observation_revision or observation.get('imageId') != view.image_id):
            raise PlatformError('recgen_observation_selection_invalid', 409)
        geometry_id = (document.get('geometryBindings', {}).get(view.image_id) or {}).get('geometrySolutionId')
        for identity, expected in ((view.image_id, view.image_sha256),
                (observation.get('maskAssetId'), view.mask_sha256), (geometry_id, view.geometry_solution_sha256)):
            if not identity or assets.get(identity, {}).get('sha256') != expected or declared.get(identity) != expected:
                raise PlatformError('research_input_hash_mismatch', 409)
    return request
