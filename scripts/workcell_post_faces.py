"""Independent photo-supported light-curtain face candidates in native world.

build(root, sources, out) reads a frozen run and writes only to out. The saved
camera and interior pointmap establish a conditional plane; RGB terminal and
side lines establish its displayed boundaries. A single source view does not
validate physical dimensions, hidden faces, thickness, or ground clearance.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import re
import time

import cv2
import numpy as np
import trimesh
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from workcell_photo_geometry import _clearance, _intersect, _raw_mask
from workcell_photo_metrology import _horizontal, _interval_union, _legacy, _load, _object_edges, _pixels, _rays, TARGETS
from workcell_photo_objects import _project
from workcell_photo_oneshot import _response
from workcell_photo_texture import texture_planar_mesh


def _line(points, tolerance=1.5):
    points = np.asarray(points, float).reshape(-1, 2)
    if len(points) < 2 or not np.isfinite(points).all() or np.linalg.norm(np.ptp(points, axis=0)) < 1e-8:
        raise ValueError('Degenerate source line')
    center = points.mean(0)
    normal = np.linalg.svd(points - center, full_matrices=False)[2][-1]
    line = np.r_[normal, -normal @ center]
    residual = abs(np.c_[points, np.ones(len(points))] @ line)
    if residual.max() > tolerance:
        raise ValueError(f'Source side/terminal fragments disagree by {residual.max():.3f} raw pixels (> {tolerance:g})')
    return line


def _corner(a, b):
    corner = np.cross(a, b)
    if abs(corner[2]) < 1e-8:
        raise ValueError('Source boundary lines do not have a finite intersection')
    return corner[:2] / corner[2]


def _boundary_segments(edge, line, corners, tolerance=1.5):
    """Record the <=1.5px source-to-straight-boundary fit, including clipping."""
    raw = np.asarray(edge.get('rawSegments', [edge['rawEnds']]), float)
    projected = raw - (raw @ line[:2] + line[2])[..., None] * line[:2]
    axis = corners[1] - corners[0]
    fraction = np.clip((projected - corners[0]) @ axis / (axis @ axis), 0, 1)
    snapped = corners[0] + fraction[..., None] * axis
    residual = float(np.linalg.norm(raw - snapped, axis=2).max())
    if residual > tolerance or np.any(np.linalg.norm(np.diff(snapped, axis=1), axis=2) < 1):
        raise ValueError(f'Visible terminal support is inconsistent with the bounded face ({residual:.3f} raw pixels)')
    return snapped, residual


def fit_face(frame, raw_mask, bottom, top, ground):
    """Fit one observed planar face; return open textured mesh and provenance.

    ponytail: one plane and straight source edges cover a rigid visible sheet;
    curved shells or hidden thickness need separate observed surfaces.
    """
    if sorted(bottom.get('faceSideIds', [])) != sorted(top.get('faceSideIds', [])) or len(bottom.get('faceSideIds', [])) != 2:
        raise ValueError('Top and bottom do not have the same observed side-line pair')
    if bottom.get('gapIntervalsRawPx') or top.get('gapIntervalsRawPx'):
        raise ValueError('Terminal edge includes an unobserved gap')
    A, K, pose = (np.asarray(frame[k], float) for k in ('A', 'K', 'pose'))
    points, valid = np.asarray(frame['points'], float), np.asarray(frame['valid'], bool)
    raw_mask = np.asarray(raw_mask, np.uint8)
    if points.shape != valid.shape + (3,) or raw_mask.shape != frame['rgb'].shape[:2]:
        raise ValueError('Source mask and pointmap dimensions disagree')
    if not all(np.isfinite(v).all() for v in (A, K, pose)) or abs(np.linalg.det(A)) < 1e-12:
        raise ValueError('Invalid saved camera/pixel affine')
    side_lines = []
    for ident in bottom['faceSideIds']:
        fragments = [edge['faceSideEdgesRaw'][edge['faceSideIds'].index(ident)] for edge in (bottom, top)]
        side_lines.append(_line(fragments))
    lower = _line(bottom.get('rawSegments', [bottom['rawEnds']]))
    upper = _line(top.get('rawSegments', [top['rawEnds']]))
    corners_raw = np.array([_corner(lower, side_lines[0]), _corner(lower, side_lines[1]),
                            _corner(upper, side_lines[1]), _corner(upper, side_lines[0])])
    contour = corners_raw.astype(np.float32)
    if not cv2.isContourConvex(contour) or cv2.contourArea(contour) < 16:
        raise ValueError('Observed boundary lines do not form a nondegenerate convex face')
    height, width = raw_mask.shape
    if (corners_raw.min(0) < [0, 0]).any() or (corners_raw.max(0) >= [width, height]).any():
        raise ValueError('Face boundary intersection is outside the source photo')
    lower_raw, lower_snap = _boundary_segments(bottom, lower, corners_raw[:2])
    upper_raw, upper_snap = _boundary_segments(top, upper, corners_raw[[3, 2]])
    canonical = np.zeros(valid.shape, np.uint8)
    cv2.fillConvexPoly(canonical, np.rint(_pixels(corners_raw, A)).astype(np.int32), 1)
    source_mask = cv2.warpPerspective(raw_mask, A, valid.shape[::-1], flags=cv2.INTER_NEAREST)
    support = cv2.erode(canonical & source_mask, np.ones((3, 3), np.uint8)).astype(bool)
    support &= valid & np.isfinite(points).all(2)
    cloud = points[support]
    if len(cloud) < 6:
        raise ValueError(f'Only {len(cloud)} valid interior pointmap samples support this visible face (need 6)')
    center = cloud.mean(0)
    _, singular, vectors = np.linalg.svd(cloud - center, full_matrices=False)
    ratio = float(singular[2] / max(singular[1], 1e-15))
    if singular[1] < singular[0] * 1e-4 or ratio > .35:
        raise ValueError(f'Interior pointmap does not support a plane: minor/major={singular[1]/max(singular[0], 1e-15):.5f}, normal/minor={ratio:.5f} (max 0.35)')
    normal = vectors[-1]
    normal *= 1 if normal @ (pose[:3, 3] - center) >= 0 else -1
    offset = -float(normal @ center)

    def intersect(raw):
        xyz = _intersect(_pixels(np.asarray(raw).reshape(-1, 2), A), K, pose, normal, offset)
        if not np.isfinite(xyz).all():
            raise ValueError('Source boundary rays do not intersect the supported plane in front of camera')
        return xyz.reshape(np.asarray(raw).shape[:-1] + (3,))

    corners = intersect(corners_raw)
    lower_xyz, upper_xyz = intersect(lower_raw), intersect(upper_raw)
    mesh = trimesh.Trimesh(corners, [[0, 1, 2], [0, 2, 3]], process=False)
    mesh.visual.vertex_colors = [235, 184, 24, 255]
    if (mesh.face_normals.mean(0) @ (pose[:3, 3] - corners.mean(0))) < 0:
        mesh.invert()
    photo = int(frame.get('photo', 0))
    mesh, appearance = texture_planar_mesh(mesh, {photo: {**frame, 'textureRgb': frame['rgb']}},
                                           {photo: source_mask}, max_size=1024)
    up = np.asarray(ground['normal'], float)
    length = np.linalg.norm(up)
    if not np.isfinite(length) or length <= 0 or not np.isfinite(ground['offset']):
        raise ValueError('Invalid saved ground plane')
    up, ground_offset = up / length, float(ground['offset']) / length
    endpoints = lower_xyz.reshape(-1, 3)
    point = endpoints[np.argmin(endpoints @ up)]
    distance, foot = _clearance(point, up, ground_offset)
    projected, _ = _project(corners, frame)
    residual = np.linalg.norm(_pixels(projected, np.linalg.inv(A)) - corners_raw, axis=1)
    plane_residual = abs(cloud @ normal + offset)
    return mesh, {'sourcePhoto': photo, 'coordinateSystem': 'MapAnything native', 'units': 'native',
        'mPerNative': None, 'thicknessNative': None, 'measurementStatus': 'conditional_single_photo',
        'scope': 'Open visible-face hypothesis; source-view fit is not physical or multiview validation. Back face and thickness unknown.',
        'cameraPolicy': 'unchanged saved K and C2W pose; original pointmap in the same native world',
        'rawCorners': corners_raw.tolist(), 'cornersNative': corners.tolist(),
        'bottomSegmentsNative': lower_xyz.tolist(), 'topSegmentsNative': upper_xyz.tolist(),
        'sourceBoundaries': {'bottom': bottom, 'top': top, 'sideLineCoefficientsRaw': [line.tolist() for line in side_lines],
                             'bottomSnapMaxRawPx': lower_snap, 'topSnapMaxRawPx': upper_snap,
                             'cornerScope': 'intersection of supported straight boundary lines; line fragments remain the observed evidence'},
        'plane': {'normal': normal.tolist(), 'offset': offset, 'supportPoints': len(cloud),
                  'normalToMinorSingularRatio': ratio, 'maxNormalToMinorSingularRatio': .35,
                  'singularValues': singular.tolist(), 'residualP95Native': float(np.percentile(plane_residual, 95)),
                  'support': 'one-pixel-eroded source face polygon intersected with source instance mask and valid native pointmap'},
        'sourceReprojectionMaxRawPx': float(residual.max()),
        'sourceReprojectionMeaning': 'by-construction projection identity; not independent validation',
        'appearance': appearance,
        'endpointEstimate': {'status': 'conditional_model_estimate', 'pointNative': point.tolist(),
            'footNative': foot.tolist(), 'heightNative': distance,
            'bottomHeightRangeNative': [float((endpoints @ up + ground_offset).min()), float((endpoints @ up + ground_offset).max())],
            'ground': {'normal': up.tolist(), 'offset': ground_offset},
            'source': 'lowest supported endpoint on this mesh bottom edge, projected to the saved inferred ground',
            'physicalValidation': 'none; single-photo pointmap plane and inferred ground remain unvalidated'}}


def _segment_error(predicted, observed):
    predicted, observed = np.asarray(predicted), np.asarray(observed)
    def distance(a, b):
        samples = a[0] + np.linspace(0, 1, 11)[:, None] * (a[1] - a[0])
        delta = b[1] - b[0]
        t = np.clip((samples - b[0]) @ delta / max(delta @ delta, 1e-12), 0, 1)
        return np.linalg.norm(samples - (b[0] + t[:, None] * delta), axis=1)
    return float(np.percentile(np.r_[distance(predicted, observed), distance(observed, predicted)], 95))


def _comparison(record, frame, diagnostic, old_bottom):
    corners = np.asarray(record['cornersNative'])
    uv, depth = _project(corners, frame)
    raw = _pixels(uv, np.linalg.inv(frame['A']))
    photo = frame['photo']
    candidates = {name: [edge for edge in diagnostic[name + 'Candidates'] if edge['photo'] == photo]
                  for name in ('bottom', 'top')}
    matches = []
    if np.isfinite(raw).all() and (depth > 0).all():
        for bottom in candidates['bottom']:
            for top in candidates['top']:
                if (sorted(bottom['faceSideIds']) != sorted(top['faceSideIds']) or
                        bottom.get('gapIntervalsRawPx') or top.get('gapIntervalsRawPx')):
                    continue
                b = _segment_error(raw[:2], bottom['rawEnds'])
                t = _segment_error(raw[[3, 2]], top['rawEnds'])
                matches.append({'bottomSymmetricP95RawPx': b, 'topSymmetricP95RawPx': t,
                                'maxSymmetricP95RawPx': max(b, t),
                                'bottomRawEnds': bottom['rawEnds'], 'topRawEnds': top['rawEnds']})
    best = min(matches, key=lambda row: row['maxSymmetricP95RawPx']) if matches else None
    source = photo == record['sourcePhoto']
    row = {'photo': photo, 'role': 'source_fit' if source else 'independent_camera_diagnostic',
           'projectedCornersRaw': raw.tolist() if np.isfinite(raw).all() and (depth > 0).all() else None,
           'bottomCandidates': len(candidates['bottom']), 'topCandidates': len(candidates['top']),
           'bestSameSidePair': best, 'thresholdRawPx': 3.,
           'status': 'by_construction' if source else ('consistent_with_detected_end_pair' if best and best['maxSymmetricP95RawPx'] <= 3 else 'not_supported'),
           'reason': ('Same source camera; cannot validate fitted geometry' if source else
                      'No detected top/bottom pair with the same supported sides' if best is None else
                      f"Symmetric finite-segment residual is {best['maxSymmetricP95RawPx']:.2f} raw pixels")}
    image = frame['rgb'].copy()
    drawable = []
    if row['projectedCornersRaw'] is not None:
        drawable.extend(raw)
        cv2.polylines(image, [np.rint(np.clip(raw, -10000, 20000)).astype(np.int32)], True, (0, 220, 230), 5)
    for edges in candidates.values():
        for edge in edges:
            points = np.asarray(edge['rawEnds'])
            drawable.extend(points)
            cv2.polylines(image, [np.rint(points).astype(np.int32)], False, (255, 55, 200), 4)
    if old_bottom is not None:
        old_uv, old_depth = _project(old_bottom, frame)
        if (old_depth > 0).all():
            old_raw = _pixels(old_uv, np.linalg.inv(frame['A']))
            drawable.extend(old_raw)
            hull = cv2.convexHull(old_raw.astype(np.float32))
            cv2.polylines(image, [np.rint(hull).astype(np.int32)], True, (255, 130, 20), 4)
    if drawable:
        points = np.asarray(drawable)
        lo = np.maximum(np.floor(points.min(0) - 70).astype(int), 0)
        hi = np.minimum(np.ceil(points.max(0) + 70).astype(int), image.shape[1::-1])
        if (hi > lo).all():
            image = image[lo[1]:hi[1], lo[0]:hi[0]]
    factor = min(1., 1000 / max(image.shape[:2]))
    image = cv2.resize(image, None, fx=factor, fy=factor)
    banner = np.full((70, max(580, image.shape[1]), 3), 25, np.uint8)
    cv2.putText(banner, f"Photo {photo}: {row['status']}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1)
    cv2.putText(banner, 'cyan: candidate; pink: RGB edges; orange: old bottom', (8, 48), cv2.FONT_HERSHEY_SIMPLEX, .48, (255, 255, 255), 1)
    canvas = np.full((len(banner) + image.shape[0], banner.shape[1], 3), 25, np.uint8)
    canvas[:len(banner)] = banner
    canvas[len(banner):, :image.shape[1]] = image
    return row, canvas


def build(root, sources, out):
    """Remote cached-run entry point; emits candidates, per-view JPEGs and JSON."""
    root, out = Path(root), Path(out)
    sources = [Path(source) for source in sources]
    if len(sources) != 4 or root.resolve() == out.resolve() or root.resolve() in out.resolve().parents:
        raise ValueError('Provide four original JPEGs and a separate output directory outside the frozen run')
    if out.exists() and any(out.iterdir()):
        raise ValueError('Candidate output directory must be empty')
    start = time.monotonic()
    geometry, catalog, segmentation, frames, _, metadata = _load(root, sources, None)
    ground = geometry['floor']
    up = np.asarray(ground['normal'], float); up /= np.linalg.norm(up)
    legacy = {ident: _legacy(catalog[ident], geometry) for ident in TARGETS}
    drawings = {photo: {'candidates': [], 'selected': []} for photo in frames}
    out.mkdir(parents=True, exist_ok=True)
    _, diagnostics = _object_edges(catalog, segmentation, geometry, frames, up, legacy, drawings,
        targets=('post-box-1', 'post-box-2'), match_edges=False,
        diagnostics_path=out / 'source-edge-candidates.json')
    posts = trimesh.load(root / 'posts.glb', force='scene')
    from workcell_endpoint_estimate import _bottom_face
    result = {'schemaVersion': 1, 'coordinateSystem': 'MapAnything native', 'units': 'native',
              'mPerNative': None, 'previewOnly': True, 'baselineModified': False, 'sources': metadata,
              'method': 'Original RGB terminal/side lines, unchanged camera, local interior pointmap plane; open source-textured visible face',
              'candidates': [], 'rejections': [], 'sourceEdgeDiagnostics': diagnostics}
    for diagnostic in diagnostics:
        ident = diagnostic['id']
        if not ident.startswith('post-box-'):
            continue
        old_bottom = _bottom_face(posts, ident.removeprefix('post-'), up)
        observations = {row['photo']: row for row in catalog[ident]['observations']}
        for photo, frame in frames.items():
            pairs = [(bottom, top) for bottom in diagnostic['bottomCandidates'] for top in diagnostic['topCandidates']
                     if bottom['photo'] == photo and top['photo'] == photo]
            if not pairs:
                result['rejections'].append({'objectId': ident, 'sourcePhoto': photo, 'reason': 'No detected bottom and top edges in the same source view'})
                continue
            match = re.search(r'instance (\d+)', observations[photo]['source'])
            response = _response(segmentation, photo, 'yellow safety post')
            raw_mask = _raw_mask({'rle': [response['rle'][int(match[1])]]}, frame['rgb'].shape[:2])
            accepted = []
            for bottom, top in pairs:
                try:
                    mesh, record = fit_face(frame, raw_mask, bottom, top, ground)
                    accepted.append((mesh, record))
                except (ValueError, np.linalg.LinAlgError, cv2.error) as error:
                    result['rejections'].append({'objectId': ident, 'sourcePhoto': photo,
                        'bottomRawEnds': bottom['rawEnds'], 'topRawEnds': top['rawEnds'], 'reason': str(error)})
            if not accepted:
                continue
            # Select the largest supported face, never a target physical height.
            mesh, record = max(accepted, key=lambda item: cv2.contourArea(np.asarray(item[1]['rawCorners'], np.float32)))
            name = f'{ident}-source-{photo}'
            record.update({'objectId': ident, 'id': name, 'model': name + '.glb',
                           'sourceAssociation': observations[photo], 'acceptedFaceAlternatives': len(accepted), 'views': []})
            mesh.export(out / record['model'])
            for target_photo, target in frames.items():
                row, image = _comparison(record, target, diagnostic, old_bottom)
                row['overlay'] = f'{name}-photo-{target_photo}.jpg'
                cv2.imwrite(str(out / row['overlay']), cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
                record['views'].append(row)
            result['candidates'].append(record)
    result['wallSeconds'] = time.monotonic() - start
    (out / 'post-faces.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result


def _raw_tolerance(frame):
    """Pre-registered display-fit gate: one saved camera-grid pixel, >=3 raw px."""
    return max(3., float(np.linalg.norm(np.linalg.inv(frame['A'])[:2, :2], ord=2)))


def _samples(segments, count=9):
    segments = np.asarray(segments, float).reshape(-1, 2, 2)
    lengths = np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1)
    if not len(lengths) or not np.isfinite(segments).all() or np.min(lengths) <= 1e-8:
        raise ValueError('Invalid finite RGB boundary segments')
    stops = np.r_[0., np.cumsum(lengths)]
    position = np.linspace(0., stops[-1], count)
    index = np.minimum(np.searchsorted(stops[1:], position, side='right'), len(lengths) - 1)
    return segments[index, 0] + ((position - stops[index]) / lengths[index])[:, None] * (segments[index, 1] - segments[index, 0])


def _remove_dominated_partials(rows):
    """Missing is not optional: retain a terminal already observed on that face."""
    def contained(segment, observed):
        segment = np.asarray(segment, float)
        length = np.linalg.norm(segment[1] - segment[0])
        if not np.isfinite(segment).all() or length <= 1e-8:
            return False
        axis = (segment[1] - segment[0]) / length
        intervals = []
        for target in observed:
            delta = np.asarray(target, float) - segment[0]
            position = delta @ axis
            if np.isfinite(delta).all() and np.max(np.linalg.norm(delta - position[:, None] * axis, axis=1)) <= 1e-6:
                intervals.append(np.clip(sorted(position), 0, length).tolist())
        # Compare finite observed unions: segmentation changes do not hide a
        # known terminal, and an unobserved gap is never filled by an envelope.
        return any(low <= 1e-6 and high >= length - 1e-6 for low, high in _interval_union(intervals))
    result = []
    for partial in rows:
        dominated = False
        if partial.get('rawCorners') is None:
            for complete in rows:
                if complete.get('rawCorners') is None:
                    continue
                for order in ((0, 1, 2, 3), (0, 3, 2, 1)):
                    if all(not segments or all(contained(segment, complete['boundariesRaw'][order[index]]) for segment in segments)
                           for index, segments in enumerate(partial['boundariesRaw'])):
                        dominated = True
                        break
                if dominated:
                    break
        if not dominated:
            result.append(partial)
    return result


def complete_face_observations(diagnostic, frames):
    """Join terminals to actually observed full-height side groups by pixels.

    Top/bottom local side IDs alone never establish a complete physical face.
    Each proposed side must also agree with an observed long RGB side group.
    These are complete-face hypotheses, not confirmed cross-photo identities.
    """
    observations, rejected = {}, []
    details = {row['photo']: row for row in diagnostic['views']}
    for photo, frame in frames.items():
        detail = details.get(photo, {})
        groups = detail.get('bottom', {}).get('sideSupportGroups', [])
        tolerance = _raw_tolerance(frame)
        fixed_groups = []
        for group in groups:
            points = np.asarray(group['rawSegments']).reshape(-1, 2)
            try:
                fixed_groups.append((_line(points, tolerance), points, group))
            except ValueError:
                pass
        def supported_side_groups(local):
            local = np.asarray(local).reshape(-1, 2)
            supported = []
            for fixed_line, group_points, group in fixed_groups:
                # Establish identity against the group's own line first. A
                # terminal must not rotate the reference line to admit itself.
                if np.max(abs(np.c_[local, np.ones(len(local))] @ fixed_line)) > tolerance:
                    continue
                points = np.r_[local, group_points]
                try:
                    line = _line(points, tolerance)
                    supported.append((float(np.max(abs(np.c_[points, np.ones(len(points))] @ line))), group['id'], line, group))
                except ValueError:
                    pass
            return supported
        bottoms = [row for row in diagnostic['bottomCandidates'] if row['photo'] == photo]
        tops = [row for row in diagnostic['topCandidates'] if row['photo'] == photo]
        found = {}
        for bottom, top in itertools.product(bottoms, tops):
            for order in ((0, 1), (1, 0)):
                try:
                    sides, selected_groups = [], []
                    for i, j in enumerate(order):
                        local = np.asarray([bottom['faceSideEdgesRaw'][i], top['faceSideEdgesRaw'][j]]).reshape(-1, 2)
                        supported = supported_side_groups(local)
                        if not supported:
                            raise ValueError('Terminal sides do not agree with a full-height RGB side group')
                        _, _, line, group = min(supported, key=lambda row: (row[0], row[1]))
                        sides.append(line); selected_groups.append(group)
                    if selected_groups[0]['id'] == selected_groups[1]['id']:
                        raise ValueError('Both terminal sides select the same physical RGB boundary')
                    lower = _line(bottom.get('rawSegments', [bottom['rawEnds']]))
                    upper = _line(top.get('rawSegments', [top['rawEnds']]))
                    corners = np.array([_corner(lower, sides[0]), _corner(lower, sides[1]),
                                        _corner(upper, sides[1]), _corner(upper, sides[0])])
                    if not cv2.isContourConvex(corners.astype(np.float32)) or cv2.contourArea(corners.astype(np.float32)) < 16:
                        raise ValueError('Complete boundaries do not enclose a convex face')
                    raw_shape = frame.get('rawShape', frame['rgb'].shape[:2] if 'rgb' in frame else None)
                    if raw_shape is None or (corners.min(0) < 0).any() or (corners.max(0) >= np.asarray(raw_shape)[::-1]).any():
                        raise ValueError('Complete boundary intersection leaves the source image')
                    if np.mean(np.linalg.norm(corners[[3, 2]] - corners[:2], axis=1)) < 4 * np.mean(np.linalg.norm(corners[[1, 2]] - corners[[0, 3]], axis=1)):
                        raise ValueError('Source face is not a long housing')
                    _boundary_segments(bottom, lower, corners[:2], tolerance)
                    _boundary_segments(top, upper, corners[[3, 2]], tolerance)
                    boundaries = [bottom.get('rawSegments', [bottom['rawEnds']]), selected_groups[1]['rawSegments'],
                                  top.get('rawSegments', [top['rawEnds']]), selected_groups[0]['rawSegments']]
                    # All four named boundaries stay together in every camera.
                    # No later optimizer may assign the bottom to a depth side.
                    row = {'photo': photo, 'rawCorners': corners.tolist(), 'boundariesRaw': boundaries, 'complete': True,
                           'bottom': bottom, 'top': top, 'sideGroups': selected_groups,
                           'thresholdRawPx': tolerance, 'partIdentity': 'complete visible-face hypothesis; long RGB sides jointly support both terminals; cross-view identity conditional'}
                    key = (tuple(corners.round(3).ravel()), tuple(g['id'] for g in selected_groups))
                    found[key] = row
                except (ValueError, np.linalg.LinAlgError, KeyError) as error:
                    rejected.append({'photo': photo, 'reason': str(error)})
        for kind, terminals in (('bottom', bottoms), ('top', tops)):
            for terminal in terminals:
                selected_groups = []
                for ident, local in zip(terminal['faceSideIds'], terminal['faceSideEdgesRaw']):
                    supported = supported_side_groups(local)
                    selected_groups.append(min(supported, key=lambda row: row[0])[3] if supported else
                                           {'id': ident, 'rawSegments': [local], 'supportScope': 'local observed side only'})
                if len(selected_groups) != 2 or selected_groups[0]['id'] == selected_groups[1]['id']:
                    continue
                segments = terminal.get('rawSegments', [terminal['rawEnds']])
                boundaries = [segments if kind == 'bottom' else [], selected_groups[1]['rawSegments'],
                              segments if kind == 'top' else [], selected_groups[0]['rawSegments']]
                key = (kind, tuple(np.asarray(segments).ravel()), tuple(g['id'] for g in selected_groups))
                found[key] = {'photo': photo, 'rawCorners': None, 'boundariesRaw': boundaries, 'complete': False,
                              'bottom': terminal if kind == 'bottom' else None, 'top': terminal if kind == 'top' else None,
                              'sideGroups': selected_groups, 'thresholdRawPx': tolerance,
                              'partIdentity': 'partial observed housing face; missing terminal not fabricated; identity conditional on complete anchor and joint projection'}
        if found:
            rows = [dict(row, observationId=f'photo-{photo}-face-{i}') for i, row in enumerate(found.values())]
            observations[photo] = _remove_dominated_partials(rows)
            if len(rows) != len(observations[photo]):
                rejected.append({'photo': photo, 'reason': 'Partial strict subset cannot omit a detected terminal of the same complete face',
                                 'count': len(rows) - len(observations[photo])})
    return observations, rejected


def _face_projection(corners, frame, inverse=None):
    uv, depth = _project(corners, frame)
    if not np.isfinite(uv).all() or np.min(depth) <= 0:
        raise ValueError('Face projects behind a source camera')
    return _pixels(uv, np.linalg.inv(frame['A']) if inverse is None else inverse)


def _face_errors(corners, row, frame, flipped=False, all_endpoints=False, *, projected=None, samples=None):
    raw = _face_projection(corners, frame) if projected is None else projected
    order = [0, 3, 2, 1] if flipped else [0, 1, 2, 3]
    errors = []
    for edge, source in enumerate(order):
        if not row['boundariesRaw'][source]:
            errors.append(np.empty((0, 2)) if all_endpoints else np.zeros((9, 2)))
            continue
        observed = (np.asarray(row['boundariesRaw'][source]).reshape(-1, 2) if all_endpoints else
                    _samples(row['boundariesRaw'][source]) if samples is None else samples[source])
        a, b = raw[edge], raw[(edge + 1) % 4]
        direction = b - a
        fraction = np.clip((observed - a) @ direction / max(direction @ direction, 1e-12), 0, 1)
        errors.append(observed - a - fraction[:, None] * direction)
    return errors


def _source_initializations(observations, frames, up):
    """Saved pointmap initializes depth only; no pointmap residual enters fitting."""
    initial = []
    basis = _horizontal(up)
    for photo, rows in observations.items():
        frame = frames[photo]
        if 'points' not in frame:
            continue
        points = np.asarray(frame['points'])
        for row in rows:
            if row.get('rawCorners') is None:
                continue
            mask = np.zeros(points.shape[:2], np.uint8)
            cv2.fillConvexPoly(mask, np.rint(_pixels(row['rawCorners'], frame['A'])).astype(np.int32), 1)
            valid = mask.astype(bool) & frame['valid'] & np.isfinite(points).all(2)
            cloud = points[valid]
            if len(cloud) < 3:
                continue
            center = np.median(cloud, axis=0)
            # This conditional plane is a starting pose, not an extra data term.
            vectors = np.linalg.svd((cloud - center) @ basis, full_matrices=False)[2]
            normal = basis @ vectors[-1]
            xyz = _intersect(_pixels(row['rawCorners'], frame['A']), frame['K'], frame['pose'], normal, -normal @ center)
            if np.isfinite(xyz).all():
                initial.append({'photo': photo, 'observationId': row['observationId'], 'cornersNative': xyz.tolist(),
                                'source': 'saved interior pointmap upright plane, initialization only; no pointmap fitting residual'})
    return initial


def fit_multiview_face(observations, frames, ground, *, max_starts=12, initial_corners=None,
                       orientation='upright', fixed_matches=None, log_label=None):
    """Six-parameter upright or eight-parameter freely oriented visible face.

    No old box footprint, native height, depth percentile or metric standard is
    an optimization target. In free mode ground is used only after the RGB fit.
    """
    started = time.monotonic()
    def progress(stage, **details):
        if log_label is not None:
            print(f'POST_SHELL {log_label}: {stage} ' + json.dumps({'elapsedSeconds': time.monotonic() - started, **details}), flush=True)
    progress('fit-begin')
    if orientation not in ('upright', 'free'):
        raise ValueError('orientation must be upright or free')
    parameter_count = 6 if orientation == 'upright' else 8
    observations = {photo: _remove_dominated_partials(rows) for photo, rows in observations.items()}
    up = np.asarray(ground['normal'], float); norm = np.linalg.norm(up)
    if up.shape != (3,) or not np.isfinite(up).all() or norm <= 0 or not np.isfinite(ground['offset']):
        raise ValueError('Invalid shared ground')
    up /= norm
    basis = _horizontal(up)
    photos = sorted(photo for photo, rows in observations.items() if rows)
    if len(photos) < 2:
        raise ValueError('At least two source views with observed face boundaries are required')
    complete = {photo: [row for row in observations[photo] if row.get('rawCorners') is not None] for photo in photos}
    if not any(complete.values()):
        raise ValueError('One complete observed face is required to anchor part identity')
    locked = {photo: (row, flip) for photo, row, flip in fixed_matches or []}
    if locked and set(locked) != set(photos):
        raise ValueError('Orientation comparison must fix the source assignment for every participating camera')
    inverses = {photo: np.linalg.inv(frames[photo]['A']) for photo in photos}
    sampled = {}
    for row in [row for photo in photos for row in observations[photo]] + [row for row, _ in locked.values()]:
        if id(row) in sampled:
            continue
        try:
            sampled[id(row)] = [_samples(boundary) if boundary else None for boundary in row['boundariesRaw']]
        except ValueError:
            sampled[id(row)] = None
    prepared = {}
    for photo in photos:
        rows = observations[photo] + ([locked[photo][0]] if photo in locked and all(locked[photo][0] is not row for row in observations[photo]) else [])
        options, points, present, counts, indices = [], [], [], [], {}
        for row in rows:
            if sampled[id(row)] is None:
                continue
            for flipped in (False, True):
                order = (0, 3, 2, 1) if flipped else (0, 1, 2, 3)
                indices.setdefault((id(row), flipped), len(options))
                options.append((row, flipped))
                points.append([sampled[id(row)][edge] if row['boundariesRaw'][edge] else np.zeros((9, 2)) for edge in order])
                present.append([bool(row['boundariesRaw'][edge]) for edge in order])
                counts.append(sum(bool(boundary) for boundary in row['boundariesRaw']))
        prepared[photo] = (options, np.asarray(points), np.asarray(present), np.asarray(counts), indices)
    progress('fixed-samples-ready', counts={p: {'candidates': len(observations[p]), 'complete': len(complete[p])} for p in photos})
    def corners(parameters):
        if orientation == 'upright':
            axis, vertical = basis @ np.array([np.cos(parameters[3]), np.sin(parameters[3])]), up
        else:
            rotation = Rotation.from_rotvec(parameters[3:6]).as_matrix()
            axis, vertical = rotation[:, 0], rotation[:, 1]
        width, height = np.exp(np.clip(parameters[-2:], -20, 20))
        a, b = parameters[:3] - width * axis / 2, parameters[:3] + width * axis / 2
        return np.array([a, b, b + height * vertical, a + height * vertical])
    def evaluate(parameters, chosen_photos=photos, fixed=None):
        xyz = corners(parameters)
        residual, matches = [], []
        for photo in chosen_photos:
            try:
                projected = _face_projection(xyz, frames[photo], inverses[photo])
            except ValueError:
                return np.full(len(chosen_photos) * 72, 1e6), []
            fixed_for_photo = locked.get(photo, fixed.get(photo) if fixed is not None else None)
            options, points, present, counts, indices = prepared[photo]
            index = indices.get((id(fixed_for_photo[0]), fixed_for_photo[1])) if fixed_for_photo is not None else None
            if not options or (fixed_for_photo is not None and index is None):
                return np.full(len(chosen_photos) * 72, 1e6), []
            selection = slice(index, index + 1) if index is not None else slice(None)
            direction = np.roll(projected, -1, axis=0) - projected
            delta = points[selection] - projected[None, :, None, :]
            denominator = np.maximum(np.matmul(direction[:, None, :], direction[:, :, None]).reshape(4), 1e-12)
            fraction = np.clip(np.matmul(delta, direction[None, :, :, None])[..., 0] / denominator[None, :, None], 0, 1)
            vectors = ((delta - fraction[..., None] * direction[None, :, None, :]) * present[selection, :, None, None]).reshape(-1, 72)
            scores = np.matmul(vectors[:, None, :], vectors[:, :, None]).reshape(-1) / counts[selection]
            best = int(np.argmin(scores))  # First minimum preserves source and flip ordering.
            vector = vectors[best]
            row, flipped = options[index if index is not None else best]
            residual.extend(vector); matches.append((photo, row, flipped))
        return np.asarray(residual), matches
    seeds = []
    def seed_from_corners(xyz, anchor):
        xyz = np.asarray(xyz, float)
        along = (xyz[1] - xyz[0] + xyz[2] - xyz[3]) / 2
        vertical = np.mean(xyz[[3, 2]] - xyz[:2], axis=0)
        if orientation == 'upright':
            along -= (along @ up) * up
            height = float(vertical @ up)
        else:
            vertical -= (vertical @ along) * along / max(along @ along, 1e-20)
            height = np.linalg.norm(vertical)
        width = np.linalg.norm(along)
        if width <= 1e-8 or height <= 1e-8:
            return
        if orientation == 'upright':
            projected = basis.T @ along
            angles = [np.arctan2(projected[1], projected[0])]
        else:
            axis, vertical = along / width, vertical / height
            angles = Rotation.from_matrix(np.column_stack([axis, vertical, np.cross(axis, vertical)])).as_rotvec()
        parameters = np.r_[xyz[:2].mean(0), angles, np.log([width, height])]
        residual, _ = evaluate(parameters, fixed=anchor)
        seeds.append((float(residual @ residual), parameters, anchor))
    stage_started = time.monotonic()
    progress('seed-generation-begin', potentialStereoPairs=sum(2 * len(complete[a]) * len(complete[b]) for a, b in itertools.combinations(photos, 2)), pointmapStarts=len(initial_corners or []))
    for pa, pb in itertools.combinations(photos, 2):
        for a, b in itertools.product(complete[pa], complete[pb]):
            for flip in (False, True):
                xy = np.asarray(b['rawCorners'])[[1, 0, 3, 2]] if flip else np.asarray(b['rawCorners'])
                rays = [_rays(_pixels(raw, frames[p]['A']), frames[p]['K'], frames[p]['pose'])
                        for p, raw in ((pa, a['rawCorners']), (pb, xy))]
                centers = [frames[p]['pose'][:3, 3] for p in (pa, pb)]
                xyz = []
                for ra, rb in zip(*rays):
                    matrices = [np.eye(3) - np.outer(ray, ray) / (ray @ ray) for ray in (ra, rb)]
                    M = sum(matrices)
                    if np.linalg.cond(M) > 1e8:
                        break
                    xyz.append(np.linalg.solve(M, sum(m @ c for m, c in zip(matrices, centers))))
                if len(xyz) != 4:
                    continue
                seed_from_corners(xyz, {pa: (a, False)})
        progress('stereo-pair-seeds-ready', photos=[pa, pb], generatedStarts=len(seeds), stageSeconds=time.monotonic() - stage_started)
    for initial in initial_corners or []:
        photo = initial['photo']
        row = next(row for row in complete[photo] if row.get('observationId') == initial['observationId'])
        seed_from_corners(initial['cornersNative'], {photo: (row, False)})
    progress('seed-generation-end', generatedStarts=len(seeds), stageSeconds=time.monotonic() - stage_started)
    if not seeds:
        raise ValueError('Saved cameras and source initialization do not triangulate a positive-size hypothesis')
    fits = []
    # Keep each complete anchor represented before filling extra starts. Distinct
    # real faces must not disappear merely because one side has lower residual.
    ordered = sorted(seeds, key=lambda row: row[0])
    diverse, seen = [], set()
    for seed in ordered:
        key = tuple((p, row.get('observationId', tuple(np.asarray(row['rawCorners']).ravel()))) for p, (row, _) in seed[2].items())
        if key not in seen:
            diverse.append(seed); seen.add(key)
    chosen_seeds = diverse[:max_starts]
    chosen_seeds += [seed for seed in ordered if all(seed is not selected for selected in chosen_seeds)][:max_starts-len(chosen_seeds)]
    progress('optimization-begin', generatedStarts=len(seeds), evaluatedStarts=len(chosen_seeds))
    for index, (_, seed, anchor) in enumerate(chosen_seeds):
        stage_started = time.monotonic()
        progress('optimization-start-begin', startIndex=index)
        fitted = least_squares(lambda p: evaluate(p, fixed=anchor)[0], seed, loss='soft_l1', f_scale=1.5, max_nfev=120)
        residual, matches = evaluate(fitted.x, fixed=anchor)
        fits.append((float(residual @ residual), fitted, matches))
        progress('optimization-start-end', startIndex=index, stageSeconds=time.monotonic() - stage_started, evaluations=fitted.nfev, converged=bool(fitted.success))
    _, fitted, matches = min(fits, key=lambda row: row[0])
    xyz = corners(fitted.x)
    singular = np.linalg.svd(fitted.jac, compute_uv=False)
    rank = int(np.sum(singular > singular[0] * 1e-7))
    reprojection = []
    for photo, row, flipped in matches:
        errors = np.concatenate(_face_errors(xyz, row, frames[photo], flipped, True))
        lengths = np.linalg.norm(errors, axis=1)
        reprojection.append({'photo': photo, 'rmsRawPx': float(np.sqrt(np.mean(lengths ** 2))),
                             'maxRawPx': float(lengths.max()), 'thresholdRawPx': _raw_tolerance(frames[photo])})
    fixed = {photo: (row, flipped) for photo, row, flipped in matches}
    held_out = []
    for photo in photos:
        stage_started = time.monotonic()
        progress('held-out-begin', photo=photo)
        training = [p for p in photos if p != photo]
        trial = least_squares(lambda p: evaluate(p, training, fixed)[0], fitted.x, loss='soft_l1', f_scale=1.5, max_nfev=120)
        values = np.linalg.svd(trial.jac, compute_uv=False)
        trial_rank = int(np.sum(values > values[0] * 1e-7))
        row, flipped = fixed[photo]
        errors = np.concatenate(_face_errors(corners(trial.x), row, frames[photo], flipped, True))
        lengths = np.linalg.norm(errors, axis=1)
        held_out.append({'photo': photo, 'converged': bool(trial.success), 'parameterCount': parameter_count, 'jacobianRank': trial_rank,
                         'maxRawPx': float(lengths.max()), 'rmsRawPx': float(np.sqrt(np.mean(lengths ** 2))),
                         'thresholdRawPx': 2 * _raw_tolerance(frames[photo]),
                         'heightNative': float(np.min(corners(trial.x)[:2] @ up + ground['offset'] / norm)),
                         'identityPolicy': 'full-fit complete face and left/right orientation held fixed; no held-out reassociation'})
        progress('held-out-end', photo=photo, stageSeconds=time.monotonic() - stage_started, evaluations=trial.nfev)
    observable_holdouts = [row for row in held_out if row['jacobianRank'] == parameter_count]
    accepted = bool(fitted.success and rank == parameter_count and matches and
                    np.min(xyz @ up + ground['offset'] / norm) >= 0 and
                    all(row['maxRawPx'] <= row['thresholdRawPx'] for row in reprojection) and
                    len(observable_holdouts) == len(photos) and
                    all(row['converged'] and row['maxRawPx'] <= row['thresholdRawPx'] for row in observable_holdouts))
    gate = {'accepted': accepted, 'converged': bool(fitted.success), 'parameterCount': parameter_count, 'jacobianRank': rank,
            'orientationMode': orientation, 'sameSourceAssignmentLocked': bool(locked),
            'orientationScope': ('Hard upright rectangular-surface prior from shared ground normal' if orientation == 'upright' else
                                 'Free rigid rectangular surface orientation from source RGB boundaries; ground excluded from the fitting residual'),
            'modelLongAxisNative': ((xyz[3] - xyz[0]) / np.linalg.norm(xyz[3] - xyz[0])).tolist(),
            'angleFromGroundNormalDeg': float(np.degrees(np.arccos(np.clip((xyz[3] - xyz[0]) @ up / np.linalg.norm(xyz[3] - xyz[0]), -1, 1)))),
            'rankSource': 'source_reprojection_without_priors',
            'jacobianSingularValues': singular.tolist(), 'sourceViews': photos, 'reprojectionByPhoto': reprojection,
            'leaveOnePhotoOut': held_out, 'observableHeldOutViews': len(observable_holdouts),
            'sourcePixelTransforms': {str(p): np.asarray(frames[p]['A']).tolist() for p in photos},
            'thresholdPolicy': 'Declared before optimization: max(3, spectral_norm(inv(raw_to_canonical)[:2,:2])) raw pixels for every observed edge; twice that for held-out fits. Every held-out training fit must remain identifiable.',
            'crossViewIdentity': 'one complete observed face anchors named boundaries; other views supply only actually observed fragments; missing boundaries contribute no residual',
            'search': {'generatedStarts': len(seeds), 'evaluatedStarts': len(fits), 'maxStarts': max_starts,
                       'scope': 'Complete-face RGB stereo or pointmap-only initializations; each source anchor represented; bounded non-exhaustive search'}}
    alternatives = []
    for cost, trial, chosen in sorted(fits, key=lambda row: row[0]):
        if not trial.success or not chosen:
            continue
        candidate_xyz = corners(trial.x)
        candidate_errors = []
        for photo, row, flipped in chosen:
            errors = np.concatenate(_face_errors(candidate_xyz, row, frames[photo], flipped, True))
            candidate_errors.append({'photo': photo, 'maxRawPx': float(np.linalg.norm(errors, axis=1).max()),
                                     'thresholdRawPx': _raw_tolerance(frames[photo])})
        # Preserve all observation-supported parts, including a side wing whose
        # different terminal is just as plausible as the broad housing face.
        if all(row['maxRawPx'] <= row['thresholdRawPx'] for row in candidate_errors):
            signature = [(photo, row.get('observationId'), flipped) for photo, row, flipped in chosen]
            if not any(candidate['sourceAssignment'] == signature for candidate in alternatives):
                alternatives.append({'cornersNative': candidate_xyz.tolist(), 'sourceAssignment': signature,
                                     'reprojectionByPhoto': candidate_errors, 'squaredResidualRawPx': cost,
                                     'bottomHeightNative': float(np.min(candidate_xyz[:2] @ up + ground['offset'] / norm))})
    gate['surfaceAlternatives'] = alternatives
    gate['terminalPartAmbiguity'] = {'resolved': False, 'supportedAssignments': len(alternatives),
                                    'scope': 'Selected visible-face terminal, not a proven unique whole-housing minimum; stepped front/wing identity remains explicit.'}
    gate['leaveOneOutMeaning'] = 'Fixed-association leave-one-photo-out stability, initialized from the full fit; not independent held-out part selection.'
    progress('fit-end', accepted=accepted, generatedStarts=len(seeds), evaluatedStarts=len(fits))
    return xyz, gate, matches


def build_multiview(root, sources, out, *, orientation='upright', compare_orientation=False):
    """Modal entry: source-supported shell surfaces and their actual model bottom."""
    root, out = Path(root), Path(out)
    sources = [Path(path) for path in sources]
    if len(sources) != 4 or root.resolve() == out.resolve():
        raise ValueError('Four original photos and a distinct new output directory are required')
    if out.exists() and any(out.iterdir()):
        raise ValueError('Output directory must be empty')
    started = time.monotonic()
    geometry, catalog, segmentation, frames, _, metadata = _load(root, sources, None)
    ground = geometry['floor']
    up = np.asarray(ground['normal'], float); up /= np.linalg.norm(up)
    legacy = {ident: _legacy(catalog[ident], geometry) for ident in TARGETS}
    drawings = {photo: {'candidates': [], 'selected': []} for photo in frames}
    out.mkdir(parents=True, exist_ok=True)
    _, diagnostics = _object_edges(catalog, segmentation, geometry, frames, up, legacy, drawings,
        targets=('post-box-1', 'post-box-2'), match_edges=False,
        diagnostics_path=out / 'source-edge-candidates.json')
    result = {'schemaVersion': 1, 'ground': ground, 'sourceInputs': metadata, 'units': 'native',
              'mPerNative': None, 'physicalValidation': 'none', 'baselineModified': False,
              'sourceFiles': {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                              for name in ('geometry.json', 'objects.json', 'posts.glb', 'sam3.json')},
              'method': 'Joint named RGB boundaries from complete anchor and partial views; fixed source cameras; conditional visible-face geometry',
              'orientationMode': orientation, 'compareOrientation': compare_orientation,
              'items': [], 'sourceEdgeDiagnostics': diagnostics}
    for diagnostic in diagnostics:
        ident = diagnostic['id']
        if not ident.startswith('post-box-'):
            continue
        print(f'POST_SHELL {ident}: complete-source-boundaries', flush=True)
        stage_started = time.monotonic()
        observed, rejected = complete_face_observations(diagnostic, frames)
        counts = {p: {'candidates': len(observed.get(p, [])), 'complete': sum(row.get('rawCorners') is not None for row in observed.get(p, []))} for p in frames}
        source_checkpoint = {'id': ident, 'candidatesByPhoto': observed, 'rejections': rejected,
                             'counts': counts, 'sourceEdgeDiagnostic': diagnostic,
                             'stageSeconds': time.monotonic() - stage_started}
        (out / f'{ident}-source-candidates.json').write_text(json.dumps(source_checkpoint, indent=2, allow_nan=False) + '\n')
        print(f'POST_SHELL {ident}: source-candidates-saved ' + json.dumps({'counts': counts, 'stageSeconds': source_checkpoint['stageSeconds']}), flush=True)
        source_associations = {row['photo']: row for row in catalog[ident]['observations']}
        item = {'id': ident, 'status': 'unsupported', 'physicalValidation': 'none',
                'geometryScope': f'Observed open full-height housing face; {orientation} rectangular-surface assumption. Hidden rear surface, thickness and stepped wings not reconstructed by this face.',
                'sourceFaceCandidates': observed, 'rejections': rejected}
        try:
            stage_started = time.monotonic()
            print(f'POST_SHELL {ident}: source-initializations-begin', flush=True)
            initial = _source_initializations(observed, frames, up)
            item['initializations'] = initial
            initialization_checkpoint = {'id': ident, 'initializations': initial, 'stageSeconds': time.monotonic() - stage_started}
            (out / f'{ident}-initializations.json').write_text(json.dumps(initialization_checkpoint, indent=2, allow_nan=False) + '\n')
            print(f'POST_SHELL {ident}: source-initializations-saved ' + json.dumps({'count': len(initial), 'stageSeconds': initialization_checkpoint['stageSeconds']}), flush=True)
            if compare_orientation:
                upright = fit_multiview_face(observed, frames, ground, initial_corners=initial, orientation='upright', log_label=f'{ident}/upright')
                extra = [{'photo': photo, 'observationId': row['observationId'], 'cornersNative': upright[0].tolist()}
                         for photo, row, _ in upright[2] if row.get('rawCorners') is not None]
                free = fit_multiview_face(observed, frames, ground, initial_corners=initial + extra,
                                         orientation='free', fixed_matches=upright[2], log_label=f'{ident}/free')
                item['orientationComparison'] = {
                    'scope': 'Same actual observed fragments, fixed per-photo face/side assignment and unchanged pixel gates; only orientation degrees of freedom differ. Extra rank is not physical validation.',
                    'upright': {'cornersNative': upright[0].tolist(), 'fitGate': upright[1]},
                    'free': {'cornersNative': free[0].tolist(), 'fitGate': free[1]}}
                corners, gate, matches = free if orientation == 'free' else upright
            else:
                corners, gate, matches = fit_multiview_face(observed, frames, ground, initial_corners=initial, orientation=orientation, log_label=f'{ident}/{orientation}')
            mesh = trimesh.Trimesh(corners, [[0, 1, 2], [0, 2, 3]], process=False)
            mesh.visual.vertex_colors = [235, 184, 24, 255]
            masks = {}
            for photo, row, _ in matches:
                if row.get('rawCorners') is None:
                    continue
                mask = np.zeros(frames[photo]['shape'], np.uint8)
                cv2.fillConvexPoly(mask, np.rint(_pixels(row['rawCorners'], frames[photo]['A'])).astype(np.int32), 1)
                masks[photo] = mask
            mesh, appearance = texture_planar_mesh(mesh, {p: {**frames[p], 'textureRgb': frames[p]['rgb']} for p in masks}, masks, max_size=1024)
            node = ident + '-observed-housing-face'
            scene = trimesh.Scene(); scene.add_geometry(mesh, node_name=node, geom_name=node)
            file = ident + '-housing-candidate.glb'
            scene.export(out / file)
            # Acceptance refers to exported vertices, not only optimizer memory.
            readback = trimesh.load(out / file, force='scene', process=False)
            transform, name = readback.graph[node]
            actual = trimesh.transform_points(readback.geometry[name].vertices, transform)
            if actual.shape != (4, 3) or not np.allclose(actual, corners, atol=1e-6, rtol=0):
                raise ValueError('Exported visible face changed optimizer geometry or vertex identity')
            readback_errors = []
            source_observations, surface_observations = [], []
            for photo, row, flipped in matches:
                errors = np.concatenate(_face_errors(actual, row, frames[photo], flipped, True))
                lengths = np.linalg.norm(errors, axis=1)
                readback_errors.append({'photo': photo, 'rmsRawPx': float(np.sqrt(np.mean(lengths ** 2))),
                                        'maxRawPx': float(lengths.max()), 'thresholdRawPx': _raw_tolerance(frames[photo])})
                groups = row['sideGroups']
                evidence = {**source_associations[photo],
                    'faceSideIds': [g['id'] for g in groups],
                    'faceSideEdgesRaw': [max(g['rawSegments'], key=lambda segment: np.linalg.norm(np.diff(segment, axis=0))) for g in groups],
                    'fullSideSegmentsRaw': [g['rawSegments'] for g in groups],
                    'topRawSegments': row['boundariesRaw'][2],
                    'fullFaceRawCorners': row['rawCorners'], 'partEvidence': row['partIdentity'],
                    'observedBoundaryNames': [name for name, boundary in zip(('bottom', 'side_1', 'top', 'side_0'), row['boundariesRaw']) if boundary],
                    'imageSideOrderFlipped': flipped}
                surface_observations.append({**evidence, 'partId': 'selected_visible_housing_face',
                                             'bottomRawSegments': row['boundariesRaw'][0]})
                if row['bottom'] is not None:
                    source_observations.append({**evidence, 'partId': 'housing_lower_terminal',
                                                'rawSegments': row['boundariesRaw'][0], 'globalModelEdge': [0, 1]})
            gate['exportedModelReprojectionByPhoto'] = readback_errors
            gate['accepted'] &= all(r['maxRawPx'] <= r['thresholdRawPx'] for r in readback_errors)
            item.update(status='supported_candidate' if gate['accepted'] else 'unsupported',
                        fitGate=gate, model={'file': file, 'nodes': [node],
                                           'sha256': hashlib.sha256((out / file).read_bytes()).hexdigest()},
                        appearance=appearance, cornersNative=actual.tolist(), sourceSurfaceObservations=surface_observations,
                        heightNative=float(np.min(actual[:2] @ up + ground['offset'] / np.linalg.norm(ground['normal']))),
                        lowerBoundary={'partId': 'housing_lower_terminal',
                                       'scope': 'Selected observed face terminal; whole-housing minimum and front-versus-wing identity unverified',
                                       'terminalPartAmbiguity': gate['terminalPartAmbiguity'],
                                       'vertices': [{'node': node, 'vertexIndex': i} for i in (0, 1)],
                                       'sourceObservations': source_observations})
            if not gate['accepted']:
                item['reason'] = 'Complete-face candidate did not pass convergence, data-only identifiability, observed-boundary or held-out projection gates; see fitGate'
            item['views'] = []
            for photo, frame in frames.items():
                image = frame['rgb'].copy()
                uv, depth = _project(actual, frame)
                raw = _pixels(uv, np.linalg.inv(frame['A']))
                if (depth > 0).all():
                    cv2.polylines(image, [np.rint(raw).astype(np.int32)], True, (0, 220, 230), 4)
                selected = next((row for p, row, _ in matches if p == photo), None)
                if selected:
                    for boundary in selected['boundariesRaw']:
                        for segment in boundary:
                            cv2.polylines(image, [np.rint(segment).astype(np.int32)], False, (255, 40, 200), 2)
                evidence_pixels = np.concatenate([np.asarray(boundary).reshape(-1, 2) for boundary in selected['boundariesRaw'] if boundary]) if selected else np.empty((0, 2))
                pixels = np.r_[raw, evidence_pixels]
                lo = np.maximum(np.floor(pixels.min(0) - 80).astype(int), 0)
                hi = np.minimum(np.ceil(pixels.max(0) + 80).astype(int), image.shape[1::-1])
                if (hi > lo).all():
                    image = image[lo[1]:hi[1], lo[0]:hi[0]]
                factor = min(1., 1200 / max(image.shape[:2]))
                image = cv2.resize(image, None, fx=factor, fy=factor)
                overlay = f'{ident}-multiview-photo-{photo}.jpg'
                cv2.imwrite(str(out / overlay), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                item['views'].append({'photo': photo, 'overlay': overlay, 'sourceSupported': selected is not None,
                                      'legend': 'cyan: exported full-face model; pink: actual named RGB boundary segments; missing sides/back not fabricated'})
        except (ValueError, np.linalg.LinAlgError, cv2.error) as error:
            item['reason'] = str(error)
        result['items'].append(item)
        print(f'POST_SHELL {ident}: {item["status"]}', flush=True)
        (out / 'post-shells.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    result['wallSeconds'] = time.monotonic() - started
    (out / 'post-shells.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result


def _extruded_face(corners, thickness, sign):
    """Keep the four observed vertices; close a hypothetical normal extrusion."""
    corners = np.asarray(corners, float)
    if corners.shape != (4, 3) or not np.isfinite(corners).all() or not np.isfinite(thickness) or thickness <= 0 or sign not in (-1, 1):
        raise ValueError('A finite observed quadrilateral, positive native thickness and signed normal are required')
    normal = np.cross(corners[1] - corners[0], corners[3] - corners[0])
    length = np.linalg.norm(normal)
    if length <= 1e-12:
        raise ValueError('Cannot extrude a degenerate visible face')
    normal /= length
    if np.max(abs((corners - corners[0]) @ normal)) > 1e-6:
        raise ValueError('Extrusion requires a planar visible face')
    vertices = np.r_[corners, corners + sign * thickness * normal]
    faces = [[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7]]
    for i in range(4):
        j = (i + 1) % 4
        faces.extend([[i, j, j + 4], [i, j + 4, i + 4]])
    faces = np.asarray(faces)
    if sign < 0:
        faces = faces[:, ::-1]
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    if not mesh.is_watertight or mesh.volume <= 0:
        raise ValueError('Extrusion did not produce an oriented closed prism')
    mesh.visual.vertex_colors = [235, 184, 24, 255]
    return mesh


def _volume_mask_metrics(mesh, observations, *, boundaries=False):
    from workcell_guard_silhouette import _mesh_mask
    views = []
    for observation in observations:
        mask = observation['mask']
        predicted = _mesh_mask(mesh, observation['frame'], mask.shape)
        row = {'photo': observation['photo'], 'iou': float(np.count_nonzero(predicted & mask) / max(1, np.count_nonzero(predicted | mask))),
               'sourcePixels': int(mask.sum()), 'projectedPixels': int(predicted.sum()), 'imageShape': list(mask.shape)}
        if boundaries:
            kernel = np.ones((3, 3), np.uint8)
            source_edge = mask & ~cv2.erode(mask.astype(np.uint8), kernel).astype(bool)
            predicted_edge = predicted & ~cv2.erode(predicted.astype(np.uint8), kernel).astype(bool)
            distances = []
            for name, edge, target in (('source', source_edge, predicted_edge), ('model', predicted_edge, source_edge)):
                if not edge.any() or not target.any():
                    row[name + 'BoundaryMedianPx'] = row[name + 'BoundaryP95Px'] = None
                    continue
                distance = cv2.distanceTransform((~target).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[edge]
                row[name + 'BoundaryMedianPx'] = float(np.median(distance))
                row[name + 'BoundaryP95Px'] = float(np.percentile(distance, 95))
                distances.append(distance)
            row['symmetricBoundaryMeanPx'] = float(np.mean(np.concatenate(distances))) if len(distances) == 2 else None
        views.append(row)
    return {'views': views, 'meanIoU': float(np.mean([r['iou'] for r in views])),
            'medianIoU': float(np.median([r['iou'] for r in views]))}


def build_volume_candidates(root, sources, face_result, out):
    """Preview-only closed extrusion; never promotes shell or metric evidence.

    A face-result JSON path resolves its models beside that file. A dictionary
    resolves them in out, so this can follow build_multiview without copying.
    Full-image metrics retain all SAM pixels and all rendered triangles; scene
    occlusion is not silently removed from the reported error.
    """
    root, out = Path(root), Path(out)
    if isinstance(face_result, (str, Path)):
        face_root = Path(face_result).parent
        face_result = json.loads(Path(face_result).read_text())
    else:
        face_root = out
    if root.resolve() == out.resolve() or len(sources) != 4:
        raise ValueError('Four source photos and a separate candidate output are required')
    started = time.monotonic()
    for name in ('geometry.json', 'objects.json', 'posts.glb', 'sam3.json'):
        if face_result.get('sourceFiles', {}).get(name) != hashlib.sha256((root / name).read_bytes()).hexdigest():
            raise ValueError('Visible-face input no longer matches frozen source: ' + name)
    geometry, catalog, segmentation, frames, _, metadata = _load(root, [Path(path) for path in sources], None)
    if metadata != face_result['sourceInputs'] or geometry['floor'] != face_result['ground']:
        raise ValueError('Volume must reuse the visible-face photos, saved cameras and shared ground')
    ground = geometry['floor']; up = np.asarray(ground['normal'], float)
    norm = np.linalg.norm(up); up /= norm; offset = ground['offset'] / norm
    out.mkdir(parents=True, exist_ok=True)
    result = {'schemaVersion': 1, 'ground': ground, 'units': 'native', 'mPerNative': None,
              'physicalValidation': 'none', 'previewOnly': True, 'baselineModified': False,
              'sourceFiles': face_result['sourceFiles'], 'items': [],
              'method': 'Fixed observed face, bounded thickness and signed normal extrusion; four-view same-instance SAM silhouettes',
              'shapeAssumption': 'Closed rectangular prism; unseen thickness and rear surface are hypotheses, not reconstructed physical housing detail',
              'metricScope': 'Full original image, raw pixels, projected amodal silhouette versus the visible same-instance SAM mask. No ROI crop, no dilation, no exclusion of mismatch, no scene-occlusion correction.',
              'searchPolicy': {'thicknessToVisibleWidth': [.02, 1.], 'samplesPerSign': 50, 'signs': [-1, 1],
                               'objective': 'maximum mean full-canonical-image SAM IoU across all four views',
                               'boundsMeaning': 'Explicit shape hypothesis relative to visible-face width, not measured thickness or a metric dimension'}}
    for item in face_result['items']:
        ident = item['id']
        record = {'id': ident, 'status': 'unsupported', 'physicalValidation': 'none', 'previewOnly': True}
        try:
            print(f'POST_VOLUME {ident}: begin', flush=True)
            face_path = (face_root / item['model']['file']).resolve()
            if face_root.resolve() not in face_path.parents or hashlib.sha256(face_path.read_bytes()).hexdigest() != item['model']['sha256']:
                raise ValueError('Source face model hash or path is invalid')
            source_scene = trimesh.load(face_path, force='scene', process=False)
            source_nodes = item['model']['nodes']
            if len(source_nodes) != 1:
                raise ValueError('Extrusion requires exactly one source-visible face')
            transform, name = source_scene.graph[source_nodes[0]]
            corners = trimesh.transform_points(source_scene.geometry[name].vertices, transform)
            if corners.shape != (4, 3) or not np.allclose(corners, item['cornersNative'], atol=1e-6, rtol=0):
                raise ValueError('Exported source face has changed observed vertex identity')
            observed_photos = {row['photo']: row for row in catalog[ident]['observations']}
            if set(observed_photos) != {1, 2, 3, 4}:
                raise ValueError('Volume comparison requires all four same-instance catalog associations')
            canonical, original, associations = [], [], []
            for photo, observation in sorted(observed_photos.items()):
                if not any(row['photo'] == photo and row.get('source') == observation['source'] for row in item['sourceSurfaceObservations']):
                    raise ValueError('Volume SAM identity differs from the visible-face evidence')
                match = re.fullmatch(r'SAM: yellow safety post; instance (\d+)', observation['source'])
                if match is None:
                    raise ValueError('Source observation lacks an exact light-curtain SAM instance')
                frame = frames[photo]
                response = _response(segmentation, photo, 'yellow safety post')
                encoded = response['rle'][int(match[1])]
                raw_mask = _raw_mask({'rle': [encoded]}, frame['rgb'].shape[:2]).astype(bool)
                if not raw_mask.any():
                    raise ValueError('Empty source instance mask')
                small = cv2.warpAffine(raw_mask.astype(np.uint8), frame['A'][:2], frame['shape'][::-1], flags=cv2.INTER_NEAREST).astype(bool)
                canonical.append({'photo': photo, 'frame': frame, 'mask': small})
                raw_frame = {**frame, 'K': np.linalg.inv(frame['A']) @ frame['K']}
                original.append({'photo': photo, 'frame': raw_frame, 'mask': raw_mask})
                associations.append({'photo': photo, 'source': observation['source'], 'samRleSha256': hashlib.sha256(encoded.encode()).hexdigest()})
            width = float(np.mean(np.linalg.norm(corners[[1, 2]] - corners[[0, 3]], axis=1)))
            trials = []
            for thickness in width * np.linspace(.02, 1., 50):
                for sign in (-1, 1):
                    mesh = _extruded_face(corners, float(thickness), sign)
                    score = _volume_mask_metrics(mesh, canonical)
                    trials.append({'thicknessNative': float(thickness), 'extrusionSign': sign, 'meanIoU': score['meanIoU']})
            best = max(trials, key=lambda row: row['meanIoU'])
            mesh = _extruded_face(corners, best['thicknessNative'], best['extrusionSign'])
            node = ident + '-closed-extrusion-hypothesis'; filename = ident + '-volume-candidate.glb'
            scene = trimesh.Scene(); scene.add_geometry(mesh, node_name=node, geom_name=node); scene.export(out / filename)
            readback = trimesh.load(out / filename, force='scene', process=False)
            transform, name = readback.graph[node]
            actual = readback.geometry[name].copy(); actual.apply_transform(transform)
            if (actual.vertices.shape != (8, 3) or actual.faces.shape != (12, 3) or not actual.is_watertight or
                    not np.allclose(actual.vertices, mesh.vertices, atol=1e-6, rtol=0)):
                raise ValueError('Volume export changed closed geometry or observed vertex identity')
            old_model = catalog[ident]['model']; old_path = (root / old_model['file']).resolve()
            if root.resolve() not in old_path.parents:
                raise ValueError('Baseline model path leaves the frozen input')
            old_scene = trimesh.load(old_path, force='scene', process=False); old_meshes = []
            for old_node in old_model['nodes']:
                transform, name = old_scene.graph[old_node]
                old_mesh = old_scene.geometry[name].copy(); old_mesh.apply_transform(transform); old_meshes.append(old_mesh)
            before = _volume_mask_metrics(trimesh.util.concatenate(old_meshes), original, boundaries=True)
            after = _volume_mask_metrics(actual, original, boundaries=True)
            record.update(status='visual_volume_hypothesis', model={'file': filename, 'sha256': hashlib.sha256((out / filename).read_bytes()).hexdigest(), 'nodes': [node]},
                thicknessNative=best['thicknessNative'], extrusionSign=best['extrusionSign'], before=before, after=after,
                sourceFaceSha256=item['model']['sha256'], sourceFaceCornersNative=corners.tolist(),
                sourceFaceGateAccepted=bool(item['fitGate']['accepted']), sourceFaceFitGate=item['fitGate'],
                sourceObservations=associations, visibleLowerEdgeVertices=[0, 1],
                geometryScope='Closed extrusion hypothesis of the selected visible face. Thickness, rear face and end caps are assumptions; no claim of complete physical housing reconstruction.',
                appearanceScope='Uniform yellow display material on the hypothesized prism; no unseen texture synthesized',
                visibleLowerEdgeHeightNative=float(np.min(actual.vertices[:2] @ up + offset)),
                hypotheticalVolumeMinimumHeightNative=float(np.min(actual.vertices @ up + offset)),
                minimumHeightScope='The volume minimum includes invented rear geometry and is not a physical clearance measurement',
                thicknessIdentifiable=False, searchTrials=trials, searchWinner=best,
                searchAtBound=bool(best['thicknessNative'] in (trials[0]['thicknessNative'], trials[-1]['thicknessNative'])),
                equalScoreHypotheses=[row for row in trials if abs(row['meanIoU'] - best['meanIoU']) <= 1e-12])
            # Store source-mask and mesh outlines on the original image. Metrics
            # above always use the entire original image, never this display crop.
            from workcell_guard_silhouette import _mesh_mask
            record['views'] = []
            for observation in original:
                photo = observation['photo']; image = frames[photo]['rgb'].copy()
                predicted = _mesh_mask(actual, observation['frame'], observation['mask'].shape)
                for mask, color in ((observation['mask'], (255, 40, 200)), (predicted, (0, 220, 230))):
                    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(image, contours, -1, color, 3)
                yy, xx = np.where(predicted | observation['mask'])
                if len(xx):
                    image = image[max(0, yy.min() - 60):min(image.shape[0], yy.max() + 61),
                                  max(0, xx.min() - 60):min(image.shape[1], xx.max() + 61)]
                factor = min(1., 1200 / max(image.shape[:2])); image = cv2.resize(image, None, fx=factor, fy=factor)
                overlay = f'{ident}-volume-photo-{photo}.jpg'; cv2.imwrite(str(out / overlay), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                record['views'].append({'photo': photo, 'overlay': overlay, 'legend': 'cyan: closed extrusion projection; pink: exact same-instance SAM; silhouette mismatch includes occlusion and mask errors'})
        except (KeyError, IndexError, ValueError, np.linalg.LinAlgError, cv2.error) as error:
            record['reason'] = str(error)
        result['items'].append(record)
        print(f'POST_VOLUME {ident}: {record["status"]}', flush=True)
        (out / 'volume-candidates.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    result['wallSeconds'] = time.monotonic() - started
    (out / 'volume-candidates.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--sources', type=Path, nargs=4, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = build(args.root, args.sources, args.out)
    print(json.dumps({'candidates': len(result['candidates']), 'rejections': len(result['rejections']),
                      'wallSeconds': result['wallSeconds'], 'output': str(args.out)}))
