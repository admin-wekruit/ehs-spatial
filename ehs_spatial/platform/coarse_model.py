"""Explicit coarse open frames fitted to owned, registered source observations.

This is an approximate shape choice, never an automatic replacement for a failed
object model. Hidden thickness and rung count are assumptions, not measurements.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt

from .contracts import PlatformError, digest
from .model_quality import _prepare_view, _array_hash
from .spatial import MeshData, primitive_mesh, project_native


def _witnesses(views, observation_ids):
    if (not isinstance(observation_ids, list) or len(observation_ids) != 2 or any(not isinstance(oid, str) for oid in observation_ids) or
            len(set(observation_ids)) != 2):
        raise PlatformError('coarse_frame_two_owned_views_required', 422)
    by_id = {v['observationId']: v for v in views}
    if len(by_id) != len(views) or any(oid not in by_id for oid in observation_ids):
        raise PlatformError('coarse_frame_observation_not_owned', 422)
    if any('error' in _prepare_view(v) for v in views):
        raise PlatformError('invalid_quality_view', 422)
    selected = [by_id[oid] for oid in observation_ids]
    if (len({v['coordinateFrameId'] for v in views}) != 1 or
            selected[0]['imageId'] == selected[1]['imageId'] or
            np.linalg.norm(selected[0]['cameraToWorld'][:3, 3] - selected[1]['cameraToWorld'][:3, 3]) < 1e-6):
        raise PlatformError('coarse_frame_independent_registered_views_required', 422)
    return selected


def qualify_depth_views(views, observation_ids, native_points, *, tolerance_pixels=2.):
    """Source-only correspondence test; no candidate mesh enters the filter.

    Require an explicitly chosen owned witness in another image. Preserve masks,
    all views and original depth, and keep non-target occluders. The returned
    proof binds raw/qualified arrays and rejected counts, not physical accuracy.
    """
    witnesses = _witnesses(views, observation_ids)
    if not np.isfinite(tolerance_pixels) or not 0 <= tolerance_pixels <= 2:
        raise PlatformError('invalid_depth_qualification_tolerance', 422)
    rows, qualified = [], []
    for view in views:
        points = np.asarray(native_points[view['imageId']])
        if points.shape != (*view['mask'].shape, 3):
            raise PlatformError('invalid_quality_point_grid', 422)
        keep = np.zeros(view['mask'].shape, bool)
        used = []
        for other in witnesses:
            if other['imageId'] == view['imageId']:
                continue
            xy, z = project_native(points, other)
            height, width = other['mask'].shape
            finite = np.isfinite(xy).all(-1) & np.isfinite(z) & (z > 0)
            uv = np.rint(np.where(finite[..., None], xy, -1)).astype(np.int64)
            inside = finite & (uv[..., 0] >= 0) & (uv[..., 0] < width) & (uv[..., 1] >= 0) & (uv[..., 1] < height)
            witness_mask = other['mask'] & other.get('domain', np.ones_like(other['mask']))
            if witness_mask.any():
                distances = distance_transform_edt(~witness_mask)
                keep[inside] |= distances[uv[..., 1][inside], uv[..., 0][inside]] <= tolerance_pixels
            used.append(other['observationId'])
        raw = view['valid'] & np.isfinite(view['depth']) & (view['depth'] > 0)
        valid = raw & (~view['mask'] | keep)
        rows.append({'observationId': view['observationId'], 'witnessObservationIds': used,
            'rawTargetDepthPixels': int((raw & view['mask']).sum()),
            'qualifiedTargetDepthPixels': int((valid & view['mask']).sum()),
            'rejectedTargetDepthPixels': int((raw & view['mask'] & ~valid).sum()),
            'rawValidSha256': _array_hash(view['valid']), 'qualifiedValidSha256': _array_hash(valid),
            'nativePointsSha256': _array_hash(points), 'sourceHashes': view['sourceHashes']})
        qualified.append({**view, 'valid': valid})
    proof = {'method': 'owned_multiview_depth_support_v1', 'tolerancePixels': tolerance_pixels,
        'witnessObservationIds': observation_ids, 'scope': 'cross_view_supported_visible_pixels',
        'physicalAccuracyConfirmed': False, 'perView': rows}
    for view in qualified:
        view['sourceHashes'] = {**view['sourceHashes'], 'depthQualification': digest(proof)}
    return qualified, proof


def coarse_open_frame(views, observation_ids, *, bar_fraction, rung_count):
    """Triangulate a slender frame's axis from two masks; retain shape assumptions."""
    selected = _witnesses(views, observation_ids)
    if (type(rung_count) is not int or not 2 <= rung_count <= 64 or
            type(bar_fraction) not in (int, float) or not np.isfinite(bar_fraction) or not 0 < bar_fraction < .5):
        raise PlatformError('invalid_coarse_frame_parameters', 422)
    planes, pixels_list, centers, rays = [], [], [], []
    for view in selected:
        y, x = np.where(view['mask'] & view.get('domain', np.ones_like(view['mask'])))
        pixels = np.column_stack((x, y))
        if len(pixels) < 8:
            raise PlatformError('coarse_frame_insufficient_mask', 422)
        mean = pixels.mean(0)
        _, singular, axes = np.linalg.svd(pixels - mean, full_matrices=False)
        if singular[0] < 3 * max(singular[1], 1e-8):
            raise PlatformError('coarse_frame_axis_ambiguous', 422)
        line = np.r_[axes[1], -axes[1] @ mean]
        normal = view['cameraToWorld'][:3, :3] @ view['K'].T @ line
        normal /= np.linalg.norm(normal)
        center = view['cameraToWorld'][:3, 3]
        planes.append(np.r_[normal, -normal @ center])
        pixels_list.append(pixels); centers.append(center)
        local = np.column_stack((pixels, np.ones(len(pixels)))) @ np.linalg.inv(view['K']).T
        rays.append(local @ view['cameraToWorld'][:3, :3].T)
    planes = np.array(planes)
    direction = np.cross(planes[0, :3], planes[1, :3])
    condition = float(np.linalg.norm(direction))
    if condition < .02:
        raise PlatformError('coarse_frame_axis_triangulation_degenerate', 422)
    direction /= condition
    origin = np.linalg.lstsq(planes[:, :3], -planes[:, 3], rcond=None)[0]
    extents, widths = [], []
    for view, pixels, center, ray in zip(selected, pixels_list, centers, rays):
        params = np.array([np.linalg.lstsq(np.column_stack((direction, -r)), center-origin, rcond=None)[0] for r in ray])
        if np.quantile(params[:, 1], .02) <= 0:
            raise PlatformError('coarse_frame_behind_camera', 422)
        extents.append(np.quantile(params[:, 0], [.02, .98]))
        projected, _ = project_native(origin + params[:, 0, None]*direction, view)
        widths.append(float(np.quantile(np.linalg.norm(projected-pixels, axis=1), .90)*2*np.median(params[:, 1])/view['K'][0, 0]))
    low, high = min(e[0] for e in extents), max(e[1] for e in extents)
    height, width = high-low, max(widths)
    if not 0 < width < height:
        raise PlatformError('coarse_frame_extent_invalid', 422)
    normal = planes[:, :3].mean(0); normal -= direction*(normal @ direction)
    normal /= np.linalg.norm(normal)
    pose = np.eye(4)
    pose[:3, :3] = np.column_stack((np.cross(normal, direction), normal, direction))
    pose[:3, 3] = origin + (low+high)/2*direction
    thickness = width*bar_fraction
    pieces = [([thickness, thickness, height], [x, 0, 0]) for x in [-width/2, width/2]]
    pieces += [([width, thickness, thickness], [0, 0, z]) for z in np.linspace(-height/2, height/2, rung_count)]
    vertices, faces = [], []
    for size, offset in pieces:
        part = primitive_mesh({'type': 'box', 'dimensions': size})
        faces.append(part.faces + sum(map(len, vertices))); vertices.append(part.vertices+offset)
    mesh = MeshData(np.vstack(vertices).astype(np.float32), np.vstack(faces).astype(np.uint32),
                    np.tile([.65, .67, .69], (sum(map(len, vertices)), 1)).astype(np.float32))
    return mesh, pose, {'method': 'two_view_coarse_open_frame_v1', 'sourceObservationIds': observation_ids,
        'widthNative': width, 'lengthNative': height, 'planeCondition': condition,
        'assumptions': {'barFraction': bar_fraction, 'rungCount': rung_count,
                        'hiddenThicknessAndRungsMeasured': False}, 'physicalPlacementConfirmed': False}
