"""Prepare editable surfaces from frozen source geometry and mask evidence.

No model-provider inference, scene writes, publication, back, or thickness.
Exact-copy mode adds original-photo UVs. The explicit manual planar option
infers a visible sheet plane, records fit checks, and retains rejected diagnostics.
Run with python -m scripts.research.build_observed_surface_models --help.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
from uuid import NAMESPACE_URL, uuid5

import numpy as np
from PIL import Image

from ehs_spatial.platform.blender_export import EXPORT_OBJECT_FIELDS, _export_parts, mesh_from_asset, write_glb
from ehs_spatial.platform.contracts import PlatformError, canonical, digest, validate_document
from ehs_spatial.platform.reconstruction import _load_geometry, _load_masks, _mesh, _scene_asset_bytes
from ehs_spatial.platform.spatial import MeshData, project_native, transform_matrix, unproject_pixels
from ehs_spatial.platform.storage import LocalBlobStore


METHOD = 'native-observed-topology-original-photo-uv-v1'
PLANAR_METHOD = 'manual-sheet-plane-mask-rays-v1'


def planar_surface(frame, record, mask, photo_bytes, constraint):
    """Explicit sheet modeling constraint; fit quality is evidence, not certification."""
    from ehs_spatial.video import ransac_plane

    if not isinstance(constraint, dict) or not str(constraint.get('constraint', '')).strip() or constraint.get('appearance') not in ('original_photo', 'neutral_translucent'):
        raise PlatformError('surface_manual_planar_constraint_required', 422)
    config = {'relativeDistanceThreshold': .02, 'minHoldoutInlierFraction': .9, 'maxHoldoutP95Relative': .02,
        'minSupportSpreadRatio': .08, 'minRayNormalCosine': .1, 'maxSplitNormalAngleDegrees': 5., 'seed': 7, 'iterations': 300}
    points = np.asarray(frame.points[frame.support() & mask], dtype=float)
    if len(points) < 200:
        raise PlatformError('surface_plane_insufficient_support', 422)
    diagonal = float(np.linalg.norm(np.ptp(points, axis=0)))
    if diagonal <= 1e-10:
        raise PlatformError('surface_plane_degenerate_support', 422)
    training = np.arange(len(points)) % 4 != 0
    tolerance = diagonal * config['relativeDistanceThreshold']
    fitted = ransac_plane(points[training], iterations=config['iterations'], threshold_m=tolerance, seed=config['seed'])
    if fitted is None:
        raise PlatformError('surface_plane_fit_failed', 422)
    normal, offset, _ = fitted
    residual = np.abs(points @ normal + offset)
    inliers = residual <= tolerance
    _, singular, _ = np.linalg.svd(points[inliers] - points[inliers].mean(axis=0), full_matrices=False)
    spread = float(singular[1] / singular[0])
    holdout_p95 = float(np.quantile(residual[~training], .95) / diagonal)
    holdout_fraction = float(inliers[~training].mean())
    # Check spatial sensitivity as well as the interleaved-pixel holdout.
    sy, sx = np.where(frame.support() & mask)
    angles = []
    for coordinate in (sx, sy):
        for side in (coordinate <= np.median(coordinate), coordinate > np.median(coordinate)):
            fit = ransac_plane(points[side], iterations=config['iterations'], threshold_m=tolerance, seed=config['seed'])
            if fit is not None:
                angles.append(float(np.degrees(np.arccos(np.clip(abs(normal @ fit[0]), 0, 1)))))
    # Each existing mask pixel becomes its exact square footprint. No rectangle,
    # convex hull, image-edge completion, or triangulation across mask holes.
    y, x = np.where(mask)
    squares = np.stack([np.column_stack((x, y)), np.column_stack((x + 1, y)),
        np.column_stack((x, y + 1)), np.column_stack((x + 1, y + 1))], axis=1)
    grid, indices = np.unique(squares.reshape(-1, 2), axis=0, return_inverse=True)
    indices = indices.reshape(-1, 4)
    faces = np.concatenate([indices[:, [0, 1, 2]], indices[:, [1, 3, 2]]]).astype(np.uint32)
    pixels = grid.astype(float) - .5
    original = np.column_stack((pixels, np.ones(len(pixels)))) @ np.linalg.inv(record['inputToCanonical']).T
    original = original[:, :2] / original[:, 2:3]
    camera = {'K': np.linalg.inv(record['inputToCanonical']) @ frame.K, 'cameraToWorld': frame.camera_to_world}
    origin = frame.camera_to_world[:3, 3]
    rays = unproject_pixels(original, np.ones(len(original)), camera) - origin
    denominator = rays @ normal
    if np.any(np.abs(denominator) <= 1e-10):
        raise PlatformError('anchor_ray_parallel_to_surface', 422)
    distance = -(normal @ origin + offset) / denominator
    if np.any(distance <= 0):
        raise PlatformError('anchor_surface_behind_camera', 422)
    vertices = origin + distance[:, None] * rays
    cosine = np.abs(denominator) / np.linalg.norm(rays, axis=1)
    failed = []
    for condition, reason in [(holdout_fraction >= config['minHoldoutInlierFraction'], 'insufficient_holdout_consensus'),
        (holdout_p95 <= config['maxHoldoutP95Relative'], 'holdout_plane_residual'),
        (spread >= config['minSupportSpreadRatio'], 'plane_support_rank'),
        (float(cosine.min()) >= config['minRayNormalCosine'], 'grazing_ray_condition'),
        (len(angles) == 4 and max(angles) <= config['maxSplitNormalAngleDegrees'], 'spatial_fit_instability')]:
        if not condition:
            failed.append(reason)
    material = {'name': 'Neutral translucent sheet; appearance is a modeling choice', 'baseColorFactor': [.72, .82, .88, .22],
        'alphaMode': 'BLEND', 'doubleSided': True, 'roughness': .35, 'metallic': 0.}
    uv, texture, mime = None, None, None
    if constraint['appearance'] == 'original_photo':
        with Image.open(io.BytesIO(photo_bytes)) as photo:
            width, height = photo.size
            mime = Image.MIME[photo.format]
        uv = ((original + .5) / [width, height]).astype(np.float32)
        if np.any(uv < 0) or np.any(uv > 1):
            raise PlatformError('surface_texture_outside_image', 422)
        texture = photo_bytes
        material = {'name': 'Original source photograph on inferred plane', 'doubleSided': True, 'roughness': 1., 'metallic': 0.}
    mesh = MeshData(vertices.astype(np.float32), faces, uv=uv, texture_bytes=texture, texture_mime_type=mime, material=material)
    report = {'methodVersion': PLANAR_METHOD, 'modelingCheckStatus': 'rejected' if failed else 'supported_approximation',
        'failedChecks': failed, 'manualConstraint': constraint, 'configuration': config, 'sourcePositionsIdentical': False,
        'sourceTriangleIndicesIdentical': False, 'supportedPixels': len(points), 'maskPixels': int(mask.sum()),
        'vertexCount': len(vertices), 'triangleCount': len(faces), 'sourceDiagonalNative': diagonal,
        'plane': [*normal.tolist(), float(offset)], 'thresholdNative': tolerance, 'inlierFraction': float(inliers.mean()),
        'holdoutInlierFraction': holdout_fraction, 'holdoutResidualP95Relative': holdout_p95,
        'residualMedianNative': float(np.median(residual)), 'residualP95Native': float(np.quantile(residual, .95)),
        'supportSingularValues': singular.tolist(), 'supportSpreadRatio': spread, 'splitNormalAnglesDegrees': angles,
        'minRayNormalCosine': float(cosine.min()), 'maxRayDepthAmplification': float(1 / cosine.min()),
        'planeResidualMaxNative': float(np.max(np.abs(vertices @ normal + offset))),
        'silhouetteMeaning': 'Exact union of source canonical mask pixel squares, mapped through original-pixel rays; no hull or filled holes.',
        'coverageMeaning': 'Inferred planar visible-region geometry under explicit manual sheet constraint; no thickness or back geometry.',
        'textureSha256': hashlib.sha256(texture).hexdigest() if texture else None,
        'physicalAccuracy': 'unverified; depth can describe structures behind a transparent sheet; neutral material does not estimate optical properties'}
    return mesh, report


def textured_surface(frame, record, mask, source_mesh, photo_bytes):
    """Add original-photo UVs only after proving the source topology and positions."""
    rebuilt = _mesh(frame.points, frame.support(), record['rgb'], mask)
    if rebuilt is None or not np.array_equal(source_mesh.vertices, rebuilt.vertices) or not np.array_equal(source_mesh.faces, rebuilt.faces):
        raise PlatformError('surface_source_topology_mismatch', 422)
    with Image.open(io.BytesIO(photo_bytes)) as photo:
        width, height = photo.size
        mime = Image.MIME[photo.format]
    if (height, width) != tuple(record['originalShape']) or mime not in ('image/png', 'image/jpeg'):
        raise PlatformError('surface_texture_image_mismatch', 422)
    keep = frame.support() & mask & np.isfinite(frame.points).all(axis=-1)
    y, x = np.where(keep)
    original = np.column_stack((x, y, np.ones(len(x)))) @ np.linalg.inv(record['inputToCanonical']).T
    original = original[:, :2] / original[:, 2:3]
    # Original integer pixel coordinates denote centers. glTF UV origin is top-left.
    uv = (original + .5) / [width, height]
    used = np.unique(source_mesh.faces)
    if not np.isfinite(uv).all() or np.any(uv[used] < 0) or np.any(uv[used] > 1):
        raise PlatformError('surface_texture_outside_image', 422)
    mesh = MeshData(source_mesh.vertices.copy(), source_mesh.faces.copy(), uv=uv.astype(np.float32),
        texture_bytes=photo_bytes, texture_mime_type=mime,
        material={'name': 'Original source photograph', 'doubleSided': True, 'roughness': 1., 'metallic': 0.,
                  'textureSampler': {'wrapS': 33071, 'wrapT': 33071, 'minFilter': 9729, 'magFilter': 9729}})
    edges = np.sort(np.concatenate([mesh.faces[:, [0, 1]], mesh.faces[:, [1, 2]], mesh.faces[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    pixels, _ = project_native(mesh.vertices, frame.camera())
    reprojection = np.linalg.norm(pixels - np.column_stack((x, y)), axis=1)
    metrics = {'maskPixels': int(mask.sum()), 'supportedPixels': int(keep.sum()),
        'supportFraction': float(keep.sum() / mask.sum()), 'vertexCount': len(mesh.vertices),
        'referencedVertexCount': len(used), 'triangleCount': len(mesh.faces), 'boundaryEdgeCount': int((counts == 1).sum()),
        'sourcePositionsIdentical': True, 'sourceTriangleIndicesIdentical': True, 'newVertices': 0, 'newTriangles': 0,
        'uvRange': {'min': mesh.uv[used].min(axis=0).tolist(), 'max': mesh.uv[used].max(axis=0).tolist()},
        'uvFiniteAndInsideOriginalPhoto': True, 'uvConvention': 'top_left; original_pixel_centers_plus_half',
        'textureSha256': hashlib.sha256(photo_bytes).hexdigest(), 'textureSize': [width, height],
        'nativeCameraReprojectionPixels': {'median': float(np.median(reprojection)), 'p95': float(np.quantile(reprojection, .95)), 'max': float(reprojection.max())},
        'reprojectionMeaning': 'Canonical pointmap versus source camera; measured only, no geometry correction.',
        'coverageMeaning': 'Visible masked surface only. Boundary edges and source depth noise are retained; hidden regions, back, and thickness are unobserved.'}
    return mesh, metrics


def surface_glb(source, revision_id, entity, representation, mesh, path):
    """Use the checked GLB writer with a compact, explicitly partial source context."""
    frame = next(f for f in source['coordinateFrames'] if f['id'] == representation['coordinateFrameId'])
    obj = {key: representation.get(key) for key in EXPORT_OBJECT_FIELDS}
    obj.update(entityId=entity['id'], parentEntityId=entity.get('parentEntityId'), label=entity['label'],
        observationRefs=entity['observationRefs'], visible=True, editable=True, sourceContext=False,
        coordinateFrameScale=frame['scale'], matrix=np.eye(4).tolist(), parts=_export_parts(mesh, {}))
    derivation = representation.get('sourceDerivation', {})
    context = {'documentKind': 'surface_source_context', 'sourceRevisionId': revision_id, 'sourceDocumentSha256': digest(source),
        'coordinateFrames': [frame], 'cameras': [c for c in source['cameras'] if c['imageId'] == derivation.get('imageId')],
        'sourceDerivation': derivation}
    manifest = {'schemaVersion': 1, 'sceneRevisionId': revision_id, 'documentSha256': digest(context),
        'status': 'candidate', 'sourceDocument': context, 'newModelCalls': 0, 'methodVersion': derivation.get('methodVersion', METHOD),
        'sourceDocumentMeaning': 'Partial geometry context; frozen full scene identified by sourceDocumentSha256.'}
    return write_glb({'manifest': manifest, 'document': context, 'objects': [obj], 'cameras': context['cameras']}, path)


def build_candidates(source, revision_id, blob_root, output_dir, entity_ids, planar_constraints=None):
    validate_document(source)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    assets = {a['id']: {**a, 'storageKey': a.get('storageKey', 'sha256/' + a['sha256'])} for a in source['assets']}
    stages = SimpleNamespace(repo=SimpleNamespace(get_asset=lambda identity: assets[identity]), blobs=LocalBlobStore(blob_root))
    frames, records = _load_geometry(source, [], stages)
    masks, errors = _load_masks(source, records, stages)
    observations = {o['id']: o for o in source['observations']}
    document, candidates = deepcopy(source), []
    for identity in entity_ids:
        entity = next(e for e in document['entities'] if e['id'] == identity)
        if entity.get('activeModelRepresentationId') is not None:
            raise PlatformError('surface_target_already_modeled', 422)
        image_id = (entity.get('cadReference') or {}).get('referenceImageId')
        selected = [observations[oid] for oid in entity['observationRefs'] if observations[oid]['imageId'] == image_id]
        if len(selected) != 1:
            raise PlatformError('surface_reference_observation_required', 422)
        observation = selected[0]
        oid = observation['id']
        if oid not in masks:
            raise PlatformError('surface_mask_unavailable', 422, errors=[e for e in errors if e['observationId'] == oid])
        surfaces = [r for r in entity['representations'] if r['kind'] == 'observed_surface' and r.get('sourceValidity') != 'stale'
            and any(ref.get('observationId') == oid and ref.get('revision') == observation['revision'] for ref in r.get('sourceRefs', []))]
        if len(surfaces) != 1:
            raise PlatformError('surface_current_source_required', 422)
        original_rep = surfaces[0]
        if (original_rep['coordinateFrameId'] != frames[image_id].coordinate_frame_id
                or not np.array_equal(transform_matrix(original_rep['transform']), np.eye(4))):
            raise PlatformError('surface_native_source_frame_required', 422)
        original_asset = assets[original_rep['assetId']]
        source_mesh = mesh_from_asset(_scene_asset_bytes(source, original_asset['id'], stages), original_asset)
        photo = _scene_asset_bytes(source, image_id, stages)
        mesh, metrics = textured_surface(frames[image_id], records[image_id], masks[oid], source_mesh, photo)
        constraint = (planar_constraints or {}).get(identity)
        if constraint is not None:
            mesh, metrics = planar_surface(frames[image_id], records[image_id], masks[oid], photo, constraint)
        method = PLANAR_METHOD if constraint else METHOD
        source_kind = 'inferred_planar_surface_from_observed_depth' if constraint else 'observed_depth_surface'
        provenance = {'methodVersion': method, 'sourceKind': source_kind, 'sourceRevisionId': revision_id,
            'sourceDocumentSha256': digest(source), 'sourceRepresentationId': original_rep['id'],
            'sourceMeshAssetId': original_asset['id'], 'sourceMeshSha256': original_asset['sha256'],
            'imageId': image_id, 'textureSha256': metrics['textureSha256'], 'observationId': oid,
            'observationRevision': observation['revision'], 'geometryBinding': deepcopy(source['geometryBindings'][image_id]),
            'inputToCanonical': np.asarray(records[image_id]['inputToCanonical']).tolist(),
            'maskAssetId': observation['maskAssetId'], 'maskSha256': assets[observation['maskAssetId']]['sha256'],
            'selectionReason': 'Existing explicit CAD/source-photo reference; other photos are retained as evidence, not fused.',
            'geometryChanges': 'Explicit sheet constraint; source-mask pixel-square ray/plane intersections.' if constraint else 'none; source positions and triangle indices copied exactly',
            'appearanceChanges': 'Neutral translucent display material, no behind-sheet photo texture.' if constraint and constraint['appearance'] == 'neutral_translucent' else 'Original full-resolution photograph added through source-pixel UVs; no baked lighting or synthetic texture.'}
        if constraint:
            provenance['planarModeling'] = metrics
        token = digest(provenance)
        asset_id, rep_id = [str(uuid5(NAMESPACE_URL, f'{prefix}:{token}')) for prefix in ('surface-asset', 'surface-model')]
        pose = {'coordinateFrameId': original_rep['coordinateFrameId'], 'position': [0., 0., 0.], 'quaternion': [0., 0., 0., 1.], 'scale': [1., 1., 1.]}
        representation = {'id': rep_id, 'kind': 'generated_mesh', 'assetId': asset_id,
            'coordinateFrameId': original_rep['coordinateFrameId'], 'transform': pose,
            'bounds': {'min': mesh.vertices.min(axis=0).tolist(), 'max': mesh.vertices.max(axis=0).tolist()},
            'placementState': 'unconfirmed', 'placementReason': 'imported_proposal',
            'coverage': 'inferred_planar_visible_region' if constraint else 'observed_visible_surface_only',
            'shapeStatus': 'source_inferred_planar_surface' if constraint else 'source_derived_surface',
            'sourceKind': source_kind, 'sourceDerivation': provenance,
            'sourceRefs': deepcopy(original_rep['sourceRefs']) + [{'assetId': original_asset['id']}, {'assetId': image_id}, {'assetId': observation['maskAssetId']}]}
        glb_path = output_dir / f'{identity}.glb'
        validation = surface_glb(source, revision_id, entity, representation, mesh, glb_path)
        restored = mesh_from_asset(glb_path.read_bytes(), {'sha256': validation['sha256']})
        part = restored.primitives[0] if restored.primitives else restored
        if not np.array_equal(part.vertices, mesh.vertices) or not np.array_equal(part.faces, mesh.faces) or not np.array_equal(part.uv, mesh.uv) or part.texture_bytes != mesh.texture_bytes:
            raise PlatformError('surface_glb_readback_mismatch', 422)
        metadata = {'kind': 'generated_mesh', 'format': 'glb', 'sourceKind': source_kind, 'sourceDerivation': provenance}
        asset = {'id': asset_id, 'sha256': validation['sha256'], 'sizeBytes': validation['bytes'],
            'mediaType': 'model/gltf-binary', 'storageKey': 'sha256/' + validation['sha256'], **metadata, 'metadata': metadata}
        rejected = metrics.get('modelingCheckStatus') == 'rejected'
        if not rejected:
            document['assets'].append(asset)
            entity['representations'].append(representation)
            entity['activeModelRepresentationId'], entity['currentModelTransform'] = rep_id, deepcopy(pose)
        candidates.append({'entityId': identity, 'label': entity['label'], 'asset': asset, 'representation': representation,
            'path': str(glb_path.resolve()), 'metrics': metrics, 'readback': validation, 'candidateStatus': 'rejected' if rejected else 'prepared_for_review',
            'limitations': ['Visible source surface only', 'Explicit planar approximation changes depth; source measurements remain unchanged' if constraint else 'Original segmentation boundary and depth noise retained',
                'No physical dimensional calibration', 'No inferred back, hidden surface, or thickness',
                'Translucency is a display choice, not a measured optical property' if constraint and constraint['appearance'] == 'neutral_translucent' else 'Image-edge clipping retained where present; texture text legibility limited by source photograph']})
    validate_document(document)
    summary = {'schemaVersion': 1, 'status': 'prepared_for_review', 'sourceRevisionId': revision_id,
        'sourceDocumentSha256': digest(source), 'methodVersion': PLANAR_METHOD if planar_constraints else METHOD, 'newModelCalls': 0, 'newGpuCalls': 0,
        'candidateCount': len(candidates), 'candidates': candidates}
    (output_dir / 'candidates.json').write_bytes(canonical(summary))
    (output_dir / 'document.json').write_bytes(canonical(document))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--document', type=Path, required=True)
    parser.add_argument('--revision-id', required=True)
    parser.add_argument('--blob-root', type=Path, default=Path('.platform/blobs'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--entity-id', action='append', required=True)
    parser.add_argument('--planar-constraints', type=Path, help='Explicit per-entity manual sheet constraint and appearance JSON mapping.')
    args = parser.parse_args()
    constraints = json.loads(args.planar_constraints.read_text()) if args.planar_constraints else None
    result = build_candidates(json.loads(args.document.read_text()), args.revision_id, args.blob_root, args.output_dir, args.entity_id, constraints)
    print(json.dumps({'status': result['status'], 'candidateCount': result['candidateCount'], 'newModelCalls': 0}))


if __name__ == '__main__':
    main()
