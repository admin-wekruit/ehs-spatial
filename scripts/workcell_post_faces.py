"""Independent photo-supported light-curtain face candidates in native world.

build(root, sources, out) reads a frozen run and writes only to out. The saved
camera and interior pointmap establish a conditional plane; RGB terminal and
side lines establish its displayed boundaries. A single source view does not
validate physical dimensions, hidden faces, thickness, or ground clearance.
"""
import argparse
import json
from pathlib import Path
import re
import time

import cv2
import numpy as np
import trimesh

from workcell_photo_geometry import _clearance, _intersect, _raw_mask
from workcell_photo_metrology import _legacy, _load, _object_edges, _pixels, TARGETS
from workcell_photo_objects import _project
from workcell_photo_oneshot import _response
from workcell_photo_texture import texture_planar_mesh


def _line(points):
    points = np.asarray(points, float).reshape(-1, 2)
    if len(points) < 2 or not np.isfinite(points).all() or np.linalg.norm(np.ptp(points, axis=0)) < 1e-8:
        raise ValueError('Degenerate source line')
    center = points.mean(0)
    normal = np.linalg.svd(points - center, full_matrices=False)[2][-1]
    line = np.r_[normal, -normal @ center]
    residual = abs(np.c_[points, np.ones(len(points))] @ line)
    if residual.max() > 1.5:
        raise ValueError(f'Source side/terminal fragments disagree by {residual.max():.3f} raw pixels (> 1.5)')
    return line


def _corner(a, b):
    corner = np.cross(a, b)
    if abs(corner[2]) < 1e-8:
        raise ValueError('Source boundary lines do not have a finite intersection')
    return corner[:2] / corner[2]


def _boundary_segments(edge, line, corners):
    """Record the <=1.5px source-to-straight-boundary fit, including clipping."""
    raw = np.asarray(edge.get('rawSegments', [edge['rawEnds']]), float)
    projected = raw - (raw @ line[:2] + line[2])[..., None] * line[:2]
    axis = corners[1] - corners[0]
    fraction = np.clip((projected - corners[0]) @ axis / (axis @ axis), 0, 1)
    snapped = corners[0] + fraction[..., None] * axis
    residual = float(np.linalg.norm(raw - snapped, axis=2).max())
    if residual > 1.5 or np.any(np.linalg.norm(np.diff(snapped, axis=1), axis=2) < 1):
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
    _, diagnostics = _object_edges(catalog, segmentation, geometry, frames, up, legacy, drawings)
    posts = trimesh.load(root / 'posts.glb', force='scene')
    from workcell_endpoint_estimate import _bottom_face
    out.mkdir(parents=True, exist_ok=True)
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--sources', type=Path, nargs=4, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = build(args.root, args.sources, args.out)
    print(json.dumps({'candidates': len(result['candidates']), 'rejections': len(result['rejections']),
                      'wallSeconds': result['wallSeconds'], 'output': str(args.out)}))
