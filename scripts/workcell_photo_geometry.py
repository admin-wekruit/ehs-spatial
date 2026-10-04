"""Automatic observed fence/floor fitting in MapAnything's unchanged native frame.

Dimensions describe the entire red cap + yellow/gray housing. They remain a
hypothesis until the physical reference and its external dimensions are confirmed.
No raw-image ROI or manually picked edge is used.
"""
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
import trimesh
from scipy.spatial import ConvexHull

from workcell_photo_oneshot import _array, _frame, _mask, _response


def _unit(v):
    return v / np.linalg.norm(v)


def _edge_depth_neighborhoods(samples_raw, A, raw_shape):
    """Keep the legacy ±2-at-518 footprint in original pixels at every resolution."""
    delta = samples_raw[-1] - samples_raw[0]
    perpendicular = _unit(np.array([delta[1], -delta[0]]))
    # The baseline grid's one-pixel footprint is max(raw HW)/518 raw pixels.
    # Map this fixed source neighborhood through the actual rounded resize/crop,
    # rather than halving its source width when the canonical grid doubles.
    raw = samples_raw[None] + np.arange(-2., 3.)[:, None, None] * perpendicular * (max(raw_shape[:2]) / 518)
    return (np.c_[raw.reshape(-1, 2), np.ones(raw.size // 2)] @ A.T)[:, :2].reshape(raw.shape)


def _rays(uv, K, pose):
    return np.c_[uv, np.ones(len(uv))] @ np.linalg.inv(K).T @ pose[:3, :3].T


def _intersect(uv, K, pose, normal, offset):
    rays = _rays(uv, K, pose)
    denominator = rays @ normal
    distance = -(pose[:3, 3] @ normal + offset) / np.where(abs(denominator) > 1e-9, denominator, np.nan)
    return pose[:3, 3] + rays * np.where(distance > 0, distance, np.nan)[:, None]


def _clearance(point, normal, offset):
    height = float(point @ normal + offset)
    return height, point - height * normal


def _consensus(points, up, threshold, vertical=False):
    """Small deterministic RANSAC; thresholds follow native scene extent."""
    rng = np.random.default_rng(17)
    p = points[::max(1, len(points) // 6000)]
    best, chosen = 0, None
    for _ in range(350):
        sample = p[rng.choice(len(p), 3, replace=False)]
        n = np.cross(sample[1] - sample[0], up if vertical else sample[2] - sample[0])
        if np.linalg.norm(n) < 1e-10:
            continue
        n = _unit(n)
        if not vertical and abs(n @ up) < .94:
            continue
        if n @ up < 0:
            n = -n
        d = -float(n @ sample[0])
        count = (abs(p @ n + d) < threshold).sum()
        if count > best:
            best, chosen = count, (n, d)
    if chosen is None:
        raise ValueError('No supported geometric plane')
    n, d = chosen
    for _ in range(3):
        support = points[abs(points @ n + d) < threshold]
        center = np.median(support, axis=0)
        if vertical:
            horizontal = support - np.outer(support @ up, up)
            direction = np.linalg.svd(horizontal - horizontal.mean(0), full_matrices=False)[2][0]
            n = _unit(np.cross(direction, up))
        else:
            n = np.linalg.svd(support - center, full_matrices=False)[2][-1]
            n *= 1 if n @ up > 0 else -1
        d = -float(n @ center)
    return n, d, abs(points @ n + d) < threshold


def _raw_mask(response, shape):
    out = np.zeros(shape, np.uint8)
    for encoded in response.get('rle', []):
        r = json.loads(encoded)
        counts = np.asarray(r['counts'], np.int64)
        if tuple(r['size']) != shape or counts.sum() != np.prod(shape):
            raise ValueError('Full resolution SAM mask does not match input photo')
        out |= np.repeat(np.arange(len(counts), dtype=np.uint8) % 2, counts).reshape(shape, order='F')
    return out


def _components(mask):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    return [(i, *row) for i, row in enumerate(stats[1:], 1)
            if 50 <= row[4] <= mask.size * .005], labels


def _anchor_candidates(rgb, frame, photo):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    red = cv2.inRange(hsv, (0, 100, 90), (12, 255, 255)) | cv2.inRange(hsv, (170, 100, 90), (179, 255, 255))
    yellow = cv2.inRange(hsv, (16, 85, 80), (42, 255, 255))
    reds, red_labels = _components(red)
    yellows, yellow_labels = _components(yellow)
    A = np.asarray(frame['input_mask_transform']['input_to_canonical_pixel_centres'])
    points = _array(frame['pts3d'])
    valid = _array(frame['non_ambiguous_mask']).astype(bool)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    rows = []
    for ri, rx, ry, rw, rh, ra in reds:
        for yi, yx, yy, yw, yh, ya in yellows:
            overlap = max(0, min(rx + rw, yx + yw) - max(rx, yx))
            if overlap < .5 * rw or not ry - .2 * rh < yy < ry + rh + .6 * yh:
                continue
            if not .15 < rw / yw < 1.4 or not .2 < rh / yh < 2:
                continue
            x0, top, x1, color_bottom = min(rx, yx), min(ry, yy), max(rx + rw, yx + yw), max(ry + rh, yy + yh)
            if x0 <= 1 or x1 >= rgb.shape[1] - 1 or color_bottom - top > 2 * (x1 - x0):
                continue
            # Gray housing continues below the yellow body: select the strongest
            # downward luminance step spanning its central face, not the bracket.
            a, b = yx + int(.2 * yw), yx + int(.8 * yw)
            stop = min(gray.shape[0] - 2, color_bottom + int(.8 * yh))
            profile = np.median(gray[color_bottom:stop + 2, a:b], axis=1).astype(float)
            profile = cv2.GaussianBlur(profile[:, None], (1, 5), 0).ravel()
            delta = profile[:-2] - profile[2:]
            if len(delta) < 3:
                continue
            bottom = color_bottom + int(np.argmax(delta)) + 1
            box = np.array([x0, top, x1, bottom], int)
            raw_y, raw_x = np.nonzero((red_labels[top:bottom, x0:x1] == ri) |
                                      (yellow_labels[top:bottom, x0:x1] == yi))
            raw_support = np.c_[raw_x + x0, raw_y + top]
            canonical = (np.c_[raw_support, np.ones(len(raw_support))] @ A.T)[:, :2]
            indices = np.rint(canonical).astype(int)
            inside = (indices[:, 0] >= 0) & (indices[:, 0] < points.shape[1]) & (indices[:, 1] >= 0) & (indices[:, 1] < points.shape[0])
            indices = np.unique(indices[inside], axis=0)
            support = points[indices[:, 1], indices[:, 0]]
            good = valid[indices[:, 1], indices[:, 0]] & np.isfinite(support).all(1)
            support = support[good]
            if len(support) < 8:
                continue
            # Boundary rays include housing bottom; its span is measured in the
            # common upright front plane, never bbox_width * depth / focal_length.
            boundary = np.vstack([raw_support[::max(1, len(raw_support) // 300)],
                                  [[yx + .15 * yw, bottom], [yx + .85 * yw, bottom]]])
            rows.append({'photo': photo, 'boxRaw': box.tolist(), 'centerNative': np.median(support, axis=0),
                         'support': support, 'boundary': (np.c_[boundary, np.ones(len(boundary))] @ A.T)[:, :2],
                         'grayHousingEdgeContrast': float(delta.max()), 'redPixels': int(ra), 'yellowPixels': int(ya)})
    return rows


def _edge_overlap(a, b, axis):
    """Observed common span, invariant to edge order and detector fragmentation."""
    amin, amax = sorted(np.asarray(a) @ axis)
    bmin, bmax = sorted(np.asarray(b) @ axis)
    overlap = min(amax, bmax) - max(amin, bmin)
    shorter = min(amax - amin, bmax - bmin)
    return overlap, shorter > 0 and overlap >= .65 * shorter


def _beam(a, b, width, depth, up, normal):
    along = _unit(b - a)
    across = _unit(np.cross(normal, along))
    box = trimesh.creation.box(extents=[np.linalg.norm(b - a), width, depth])
    T = np.eye(4)
    T[:3, :3] = np.c_[along, across, normal]
    T[:3, 3] = (a + b) / 2
    box.apply_transform(T)
    box.visual.vertex_colors = [184, 188, 191, 255]
    return box


def build(root: Path, sources: list[Path], diameter_m=.2, height_m=.2, reference=None):
    start = time.monotonic()
    root = Path(root)
    if not np.isfinite([diameter_m, height_m]).all() or min(diameter_m, height_m) <= 0:
        raise ValueError('Reference width and height must be finite and positive')
    if len(sources) < 2:
        raise ValueError('At least two source photos of one scene are required')
    seg = json.loads((root / 'sam3.json').read_text())
    frames, floor_points, fence_points, candidates = [], [], [], []
    for i, source in enumerate(sources, 1):
        frame = _frame(root, i)
        points = _array(frame['pts3d'])
        good = _array(frame['non_ambiguous_mask']).astype(bool) & np.isfinite(points).all(2)
        pose, K = _array(frame['camera_poses']), _array(frame['intrinsics'])
        excluded = np.zeros(good.shape, bool)
        for prompt in seg['prompts']:
            excluded |= _mask(frame, _response(seg, i, prompt['text']))
        lower = np.indices(good.shape)[0] >= good.shape[0] * .5
        selected = points[good & lower & ~excluded]
        floor_points.append(selected[::max(1, len(selected) // 5000)])
        fence_mask = _mask(frame, _response(seg, i, 'safety fence'))
        selected = points[good & fence_mask]
        fence_points.append(selected[::max(1, len(selected) // 10000)])
        rgb = cv2.cvtColor(cv2.imread(str(source)), cv2.COLOR_BGR2RGB)
        candidates.extend(_anchor_candidates(rgb, frame, i))
        frames.append({'K': K, 'pose': pose, 'A': np.asarray(frame['input_mask_transform']['input_to_canonical_pixel_centres'])})
        del frame, rgb, points
    up = _unit(np.mean([-f['pose'][:3, 1] for f in frames], axis=0))
    fp = np.concatenate(floor_points)
    extent = float(np.linalg.norm(np.percentile(fp, 90, axis=0) - np.percentile(fp, 10, axis=0)))
    tolerance = extent * .008
    n, d, support = _consensus(fp, up, tolerance)
    up = n
    floor_support = fp[support]
    u = _unit(np.cross(up, frames[0]['pose'][:3, 2]))
    v = np.cross(up, u)
    # Floor display is local to the supported fence footprint. Remote floor
    # points may fit the plane but must not create an enormous export polygon.
    local_fence = np.concatenate(fence_points) @ np.c_[u, v]
    low, high = np.percentile(local_fence, [2, 98], axis=0)
    margin = (high - low) * .12
    uv = floor_support @ np.c_[u, v]
    uv = uv[np.all((uv >= low - margin) & (uv <= high + margin), axis=1)]
    hull = ConvexHull(uv)
    vertices = np.c_[uv[hull.vertices], np.ones(len(hull.vertices))]
    xyz = vertices[:, :1] * u + vertices[:, 1:2] * v - d * up
    floor_mesh = trimesh.Trimesh(vertices=xyz, faces=[[0, j, j + 1] for j in range(1, len(xyz) - 1)], process=False)
    floor_mesh.visual.vertex_colors = [140, 138, 128, 255]
    (root / 'floor-fitted.glb').write_bytes(floor_mesh.export(file_type='glb'))
    floor = {'normal': up.tolist(), 'offset': d, 'supportPoints': int(support.sum()),
             'residualP95Native': float(np.percentile(abs(floor_support @ up + d), 95)),
             'sourcePhotos': list(range(1, len(sources) + 1)), 'method': 'automatic lower-image unsegmented horizontal plane consensus',
             'status': 'observed fit; floor identity inferred from orientation and lower-image support'}
    # Publish the same fitted ground early so semantic panel routing need not wait for fence fitting.
    (root / 'floor-reference.json').write_text(json.dumps(floor) + '\n')
    groups = []
    for candidate in candidates:
        group = next((g for g in groups if all(c['photo'] != candidate['photo'] for c in g) and
                      np.linalg.norm(g[0]['centerNative'] - candidate['centerNative']) < extent * .04), None)
        if group is None:
            groups.append([candidate])
        else:
            group.append(candidate)
    selected = max(groups, key=lambda g: (len(g), sum(c['redPixels'] + c['yellowPixels'] for c in g)), default=[])
    anchor = {'assumedHeightM': float(height_m), 'assumedWidthM': float(diameter_m),
              'scope': 'whole component: red cap, yellow body, gray lower housing; mounting bracket excluded',
              'status': 'unavailable', 'nativeHeight': None, 'nativeWidth': None, 'mPerNative': None,
              'referenceFit': {'status': 'unsupported', 'mPerNative': None,
                               'reason': 'Complete named three-dimension reference and source support are required'},
              'views': [], 'assumptions': ['Reference identity and supplied external dimensions require physical confirmation.',
                'Shared front-plane envelopes are used for association only; axial metric scale requires a validated 3D reference fit.',
                'Gray housing bottom is an automatic luminance edge; multiview depth and silhouette errors remain.']}
    if len(selected) >= 2:
        ap = np.concatenate([c['support'] for c in selected])
        # The depth cloud's horizontal principal direction defines the visible
        # face; upright reference assumption supplies the other plane direction.
        flat = ap - np.outer(ap @ up, up)
        side = np.linalg.svd(flat - flat.mean(0), full_matrices=False)[2][0]
        an = _unit(np.cross(side, up)); ad = -float(np.median(ap @ an))
        dims = []
        for c in selected:
            f = frames[c['photo'] - 1]
            cloud = _intersect(c['boundary'], f['K'], f['pose'], an, ad)
            heights, widths = cloud @ up, cloud @ side
            nh, nw = float(np.ptp(heights)), float(np.ptp(widths))
            if not np.isfinite([nh, nw]).all() or min(nh, nw) <= 0:
                continue
            dims.append([nh, nw])
            anchor['views'].append({k: c[k] for k in ['photo', 'boxRaw', 'grayHousingEdgeContrast', 'redPixels', 'yellowPixels']} |
                                   {'nativeHeight': nh, 'nativeWidth': nw})
        if dims:
            nh, nw = np.median(dims, axis=0)
            anchor.update(nativeHeight=float(nh), nativeWidth=float(nw),
                          centerNative=np.median(ap, axis=0).tolist(), normal=an.tolist(), offset=ad,
                          status='native association only; axial reference scale unavailable',
                          relativeWidthResidual=None,
                          relativeHeightSpread=float(np.ptp(np.asarray(dims)[:, 0]) / nh),
                          relativeWidthSpread=float(np.ptp(np.asarray(dims)[:, 1]) / nw),
                          frontPlaneResidualP95Native=float(np.percentile(abs(ap @ an + ad), 95)))
            # This envelope remains useful for object association, but curved
            # silhouettes do not measure the supplied axial height.
            anchor['planarEnvelope'] = {'nativeHeight': float(nh), 'nativeWidth': float(nw),
                                       'scope': 'Association extent only; never a metric reference dimension'}
            anchor['nativeHeightScope'] = 'planar association extent, not axial height'
            if reference is not None:
                from workcell_button_bundle import fit_reference_shape, observe_reference

                compact = {i: {**frame, 'rawShape': None} for i, frame in enumerate(frames, 1)}
                observations = []
                try:
                    for candidate in selected:
                        photo = candidate['photo']
                        rgb = cv2.cvtColor(cv2.imread(str(sources[photo - 1])), cv2.COLOR_BGR2RGB)
                        compact[photo]['rawShape'] = list(rgb.shape[:2])
                        observations.append(observe_reference({'rgb': rgb}, candidate))
                        del rgb
                    fit = fit_reference_shape(compact, observations, reference, up)
                except (ValueError, KeyError, np.linalg.LinAlgError, FloatingPointError) as error:
                    # The contour hash binds a single-photo conditional scale to exactly these observations.
                    fit = {'status': 'unsupported', 'mPerNative': None, 'candidateMPerNative': None,
                           'reason': f'{type(error).__name__}: {error}', 'observations': observations,
                           'sourceContourSha256': hashlib.sha256(json.dumps(observations, sort_keys=True).encode()).hexdigest()}
                anchor['referenceFit'] = fit
                if fit['status'] == 'available':
                    shape = fit['fittedNuisanceParameters']
                    anchor.update(mPerNative=fit['mPerNative'], nativeHeight=shape['height'],
                                  nativeWidth=2 * shape['yellowRadius'],
                                  nativeHeightScope='axial height of fitted three-dimension reference',
                                  status='conditional three-dimension 3D reference fit')
    # Fit separate supported fence planes; all later image lines are ray/plane
    # intersections, so lower rails and the real open space remain geometric.
    cloud = np.concatenate(fence_points)
    fence_input_count = len(cloud)
    planes = []
    for _ in range(4):
        if len(cloud) < 500:
            break
        pn, pd, keep = _consensus(cloud, up, tolerance * 1.5, vertical=True)
        pts = cloud[keep]
        if len(pts) < 500:
            break
        duplicate = next((p for p in planes if abs(pn @ p['normal']) > .995 and
                          np.median(abs(pts @ p['normal'] + p['offset'])) < tolerance * 5), None)
        if duplicate is None:
            planes.append({'normal': pn, 'offset': pd, 'points': pts})
        else:
            duplicate['points'] = np.vstack([duplicate['points'], pts])
        cloud = cloud[~keep]
    records, line_evidence = [], []
    diagnostics = {'fenceInputPoints': fence_input_count, 'planeCount': len(planes),
                   'planeSupportPoints': [len(p['points']) for p in planes],
                   'sourceStencil': 'five raw-pixel normal offsets at (-2,-1,0,1,2) * max(raw HW)/518; exact saved affine',
                   'photos': []}
    for photo, source in enumerate(sources, 1):
        raw = cv2.imread(str(source))
        raw_mask = _raw_mask(_response(seg, photo, 'safety fence'), raw.shape[:2])
        edge_radius = max(1, round(min(raw.shape[:2]) * .003))
        raw_mask = cv2.dilate(raw_mask, np.ones((2 * edge_radius + 1,) * 2, np.uint8))
        # Long structural lines are found on a moderately sized image; pixel
        # coordinates still map exactly through the recorded input affine.
        factor = min(1., 1600 / max(raw.shape[:2]))
        gray = cv2.resize(cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY), None, fx=factor, fy=factor)
        lines = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(gray)[0]
        f = frames[photo - 1]
        frame = _frame(root, photo)
        depth_points = _array(frame['pts3d'])
        valid = _array(frame['non_ambiguous_mask']).astype(bool)
        proposals = []
        counts = {'photo': photo, 'detectedSegments': 0 if lines is None else len(lines),
                  'rawNeighborhoodHalfWidthPx': 2 * max(raw.shape[:2]) / 518,
                  'proposalsBeforeCap': 0, 'proposals': 0, 'pairedBeams': 0, 'unpairedProposals': 0,
                  'rejected': {name: 0 for name in ('short', 'outsideMask', 'outsideRaster', 'depthSupport',
                              'neighborhoodSupport', 'planeSupport', 'intersection', 'orientationOrLength')}}
        diagnostics['photos'].append(counts)
        before_pairing = len(records)
        if lines is None:
            continue
        for segment in lines.reshape(-1, 4):
            raw_end = (segment.reshape(2, 2) + .5) / factor - .5
            if np.linalg.norm(segment[2:] - segment[:2]) < min(gray.shape) * .025:
                counts['rejected']['short'] += 1
                continue
            t = np.linspace(.05, .95, 15)
            sample_raw = raw_end[:1] * (1 - t[:, None]) + raw_end[1:] * t[:, None]
            ri = np.rint(sample_raw).astype(int)
            if np.mean(raw_mask[ri[:, 1], ri[:, 0]]) < .8:
                counts['rejected']['outsideMask'] += 1
                continue
            canonical = (np.c_[sample_raw, np.ones(len(t))] @ f['A'].T)[:, :2]
            ix = np.rint(canonical).astype(int)
            inside = (ix[:, 0] >= 0) & (ix[:, 0] < depth_points.shape[1]) & (ix[:, 1] >= 0) & (ix[:, 1] < depth_points.shape[0])
            ix = ix[inside]
            if len(ix) < 10:
                counts['rejected']['outsideRaster'] += 1
                continue
            dp = depth_points[ix[:, 1], ix[:, 0]]
            vg = valid[ix[:, 1], ix[:, 0]] & np.isfinite(dp).all(1)
            if vg.sum() < 8:
                counts['rejected']['depthSupport'] += 1
                continue
            # Test both sides of an edge; interpolated pointmaps mix the
            # rail with floor/background exactly on the silhouette.
            neighborhoods = []
            for sample in _edge_depth_neighborhoods(sample_raw, f['A'], raw.shape):
                near = np.rint(sample).astype(int)
                near[:, 0] = np.clip(near[:, 0], 0, depth_points.shape[1] - 1)
                near[:, 1] = np.clip(near[:, 1], 0, depth_points.shape[0] - 1)
                values = depth_points[near[:, 1], near[:, 0]]
                good_near = valid[near[:, 1], near[:, 0]] & np.isfinite(values).all(1)
                if good_near.sum() >= 8:
                    neighborhoods.append(values[good_near])
            if not neighborhoods:
                counts['rejected']['neighborhoodSupport'] += 1
                continue
            if not planes:
                counts['rejected']['planeSupport'] += 1
                continue
            errors = [min(np.median(abs(values @ p['normal'] + p['offset'])) for values in neighborhoods) for p in planes]
            pi = int(np.argmin(errors))
            if errors[pi] > tolerance * 3:
                counts['rejected']['planeSupport'] += 1
                continue
            p = planes[pi]
            end = (np.c_[raw_end, np.ones(2)] @ f['A'].T)[:, :2]
            world = _intersect(end, f['K'], f['pose'], p['normal'], p['offset'])
            if not np.isfinite(world).all():
                counts['rejected']['intersection'] += 1
                continue
            length = np.linalg.norm(world[1] - world[0]); direction = _unit(world[1] - world[0])
            vertical = abs(direction @ up) > .985
            horizontal = abs(direction @ up) < .12
            if not (vertical or horizontal) or length > extent * 2:
                counts['rejected']['orientationOrLength'] += 1
                continue
            # Snap only the direction within the supported plane. Endpoint
            # center and length are preserved; no point is forced onto floor.
            axis = up if vertical else _unit(np.cross(up, p['normal']))
            center = world.mean(0)
            world = np.stack([center - axis * length / 2, center + axis * length / 2])
            pixel_native = length / np.linalg.norm(raw_end[1] - raw_end[0])
            proposals.append({'plane': pi, 'ends': world, 'length': length, 'horizontal': horizontal,
                              'photo': photo, 'rawEnds': raw_end, 'pixelNative': pixel_native,
                              'planeDepthResidualNative': float(errors[pi])})
        # Pair parallel image edges into observed rectangular rail faces.
        # ponytail: O(n²) bounded to 500 strongest segments per photo; spatial
        # indexing is the upgrade if substantially denser captures are required.
        counts['proposalsBeforeCap'] = len(proposals)
        proposals = sorted(proposals, key=lambda r: -r['length'])[:500]
        counts['proposals'] = len(proposals)
        line_evidence.extend({'sourcePhoto': photo, 'plane': r['plane'], 'rawEnds': r['rawEnds'].tolist(),
                              'initialHeightNative': float(r['ends'].mean(0) @ up + d),
                              'lengthNative': float(r['length'])}
                             for r in proposals if r['horizontal'])
        used = set()
        for i, row in enumerate(proposals):
            if i in used:
                continue
            axis = _unit(row['ends'][1] - row['ends'][0]); across = _unit(np.cross(planes[row['plane']]['normal'], axis))
            best = None
            for j in range(i + 1, len(proposals)):
                other = proposals[j]
                if j in used or row['plane'] != other['plane'] or row['horizontal'] != other['horizontal']:
                    continue
                distance = abs((other['ends'].mean(0) - row['ends'].mean(0)) @ across)
                overlap, supported = _edge_overlap(row['ends'], other['ends'], axis)
                if 2 * row['pixelNative'] < distance < .12 * min(row['length'], other['length']) and supported:
                    if best is None or distance < best[0]:
                        best = distance, j
            if best is None:
                continue
            width, j = best
            used.update([i, j])
            other = proposals[j]
            center = (row['ends'].mean(0) + other['ends'].mean(0)) / 2
            left = max(min(row['ends'] @ axis), min(other['ends'] @ axis))
            right = min(max(row['ends'] @ axis), max(other['ends'] @ axis))
            center += axis * ((left + right) / 2 - center @ axis)
            length = right - left
            ends = np.stack([center - axis * length / 2, center + axis * length / 2])
            # Rectangular face width is observed. Extrusion is a rendering
            # assumption, kept thinner than its observed width and disclosed.
            depth = min(width, row['pixelNative'] * 2)
            label = f'fence-p{row["plane"]}-view{photo}-beam{len(records)}'
            low_point = center - width / 2 * up if row['horizontal'] else ends[0]
            h, foot = _clearance(low_point, up, d)
            records.append({'id': label, 'plane': row['plane'], 'sourcePhoto': photo,
                            'horizontal': bool(row['horizontal']), 'lengthNative': float(length),
                            'widthNative': float(width), 'depthNative': float(depth),
                            'endsNative': ends.tolist(), 'pointNative': low_point.tolist(),
                            'footNative': foot.tolist(), 'heightNative': h,
                            'rawEdges': [row['rawEnds'].tolist(), other['rawEnds'].tolist()],
                            'planeDepthResidualNative': row['planeDepthResidualNative']})
        counts['pairedBeams'] = len(records) - before_pairing
        counts['unpairedProposals'] = len(proposals) - len(used)
        del raw, raw_mask, frame, depth_points
    diagnostics['pairedBeams'] = len(records)
    (root / 'fence-edge-diagnostics.json').write_text(json.dumps(diagnostics, indent=2) + '\n')
    if not records:
        raise ValueError('No image-supported paired fence edges; cannot fabricate a fence model')
    # A shared rail is a repeatable 3D LINE. Intersect its image interpretation
    # planes from different photos, avoiding the mixed edge-depth samples.
    refinements = []
    for pi, plane in enumerate(planes):
        horizontal = sorted([r for r in records if r['plane'] == pi and r['horizontal']], key=lambda r: r['heightNative'])
        clusters = []
        for row in horizontal:
            group = next((g for g in clusters if abs(np.median([b['heightNative'] for b in g]) - row['heightNative']) < tolerance * 5), None)
            if group is None:
                clusters.append([row])
            else:
                group.append(row)
        for group in clusters:
            if len({r['sourcePhoto'] for r in group}) < 2:
                continue
            normals, offsets = [[], []], [[], []]
            for row in group:
                f = frames[row['sourcePhoto'] - 1]
                edge_planes = []
                for edge in row['rawEdges']:
                    uv = (np.c_[edge, np.ones(2)] @ f['A'].T)[:, :2]
                    rays = _rays(uv, f['K'], f['pose'])
                    viewing_normal = _unit(np.cross(*rays))
                    xyz = _intersect(uv, f['K'], f['pose'], plane['normal'], plane['offset'])
                    edge_planes.append((float(np.mean(xyz @ up)), viewing_normal, float(viewing_normal @ f['pose'][:3, 3])))
                for j, (_, normal, offset) in enumerate(sorted(edge_planes, key=lambda r: r[0])):
                    normals[j].append(normal); offsets[j].append(offset)
            M = np.asarray(normals[0])
            _, singular, axes = np.linalg.svd(M, full_matrices=True)
            axis = axes[-1]
            if abs(axis @ up) > .08 or singular[1] < .015:
                continue
            axis = _unit(axis - (axis @ up) * up)
            along = float(np.median([np.mean(r['endsNative'], axis=0) @ axis for r in group]))
            edges3d = [np.linalg.lstsq(np.vstack([normal, axis]), np.r_[offset, along], rcond=None)[0]
                       for normal, offset in zip(normals, offsets)]
            normal = _unit(np.cross(axis, up))
            offset = -float(np.mean(edges3d, axis=0) @ normal)
            lower_h, upper_h = sorted(float(p @ up + d) for p in edges3d)
            refinements.append({'plane': pi, 'normal': normal, 'offset': offset,
                                'heightNative': lower_h, 'widthNative': upper_h - lower_h,
                                'sourcePhotos': sorted({r['sourcePhoto'] for r in group}),
                                'viewPlaneCondition': float(singular[0] / singular[1]),
                                'lineFitResidualNative': float(np.max(abs(M @ edges3d[0] - np.asarray(offsets[0])))),
                                'members': [r['id'] for r in group]})
        # The lowest multiview structural rail constrains its section plane.
        # This geometric refinement moves the entire section together.
        fitted = [r for r in refinements if r['plane'] == pi]
        if fitted:
            fit = min(fitted, key=lambda r: r['heightNative'])
            plane.update(normal=fit['normal'], offset=fit['offset'])
    scene = trimesh.Scene()
    for row in records:
        plane = planes[row['plane']]
        f = frames[row['sourcePhoto'] - 1]
        edges = []
        for edge in row['rawEdges']:
            uv = (np.c_[edge, np.ones(2)] @ f['A'].T)[:, :2]
            edges.append(_intersect(uv, f['K'], f['pose'], plane['normal'], plane['offset']))
        axis = _unit(np.cross(up, plane['normal'])) if row['horizontal'] else up
        across = _unit(np.cross(plane['normal'], axis))
        centers = np.array([e.mean(0) for e in edges])
        center = centers.mean(0)
        width = abs(float((centers[1] - centers[0]) @ across))
        left = max(min(e @ axis) for e in edges)
        right = min(max(e @ axis) for e in edges)
        center += axis * ((left + right) / 2 - center @ axis)
        length = right - left
        ends = np.stack([center - axis * length / 2, center + axis * length / 2])
        depth = min(width, row['depthNative'])
        mesh = _beam(*ends, width, depth, up, plane['normal'])
        scene.add_geometry(mesh, geom_name=row['id'], node_name=row['id'])
        point = center - width / 2 * up if row['horizontal'] else ends[0]
        h, foot = _clearance(point, up, d)
        row.update(endsNative=ends.tolist(), pointNative=point.tolist(), footNative=foot.tolist(),
                   heightNative=h, widthNative=width, lengthNative=length, depthNative=depth)
    # Observed planar fence screens retain the source silhouette and colors.
    # Triangles are only built inside the segmentation and close to the fitted
    # section's pointmap support. No face extends beneath the observed panel.
    panel_choices = {}
    for photo in range(1, len(sources) + 1):
        frame = _frame(root, photo)
        points, colors = _array(frame['pts3d']), _array(frame['image'])
        mask = _mask(frame, _response(seg, photo, 'safety fence'))
        valid = _array(frame['non_ambiguous_mask']).astype(bool) & np.isfinite(points).all(2)
        f = frames[photo - 1]
        yy, xx = np.indices(mask.shape)
        for pi, plane in enumerate(planes):
            keep = mask & valid & (abs(points @ plane['normal'] + plane['offset']) < tolerance * 7)
            # Two-pixel sampling keeps the local mesh light and preserves the
            # observed panel boundary instead of enclosing it in a cuboid.
            keep = keep[::2, ::2]
            grid = np.c_[xx[::2, ::2].ravel(), yy[::2, ::2].ravel()]
            xyz = _intersect(grid, f['K'], f['pose'], plane['normal'], plane['offset'])
            keep &= np.isfinite(xyz).all(1).reshape(keep.shape)
            ids = np.arange(keep.size).reshape(keep.shape)
            cells = keep[:-1, :-1] & keep[1:, :-1] & keep[:-1, 1:] & keep[1:, 1:]
            y, x = np.nonzero(cells)
            if len(x) < 100:
                continue
            a, b, c, e = ids[y, x], ids[y + 1, x], ids[y + 1, x + 1], ids[y, x + 1]
            mesh = trimesh.Trimesh(vertices=xyz, faces=np.r_[np.stack([a,b,c],1), np.stack([a,c,e],1)],
                                   vertex_colors=colors[::2, ::2].reshape(-1, 3), process=False)
            mesh.remove_unreferenced_vertices()
            label = f'panel-{pi}-view{photo}'
            record = {'plane': pi, 'sourcePhoto': photo, 'faces': len(mesh.faces),
                      'representation': 'observed planar screen; mesh-wire holes are not resolved'}
            if pi not in panel_choices or mesh.area > panel_choices[pi][0].area:
                panel_choices[pi] = (mesh, label, record)
        del frame, points, colors
    panel_records = []
    surface_scene = trimesh.Scene()
    for mesh, label, record in panel_choices.values():
        surface_scene.add_geometry(mesh, node_name=label, geom_name=label)
        panel_records.append(record)
    (root / 'fence-surface.glb').write_bytes(surface_scene.export(file_type='glb'))
    # Structural continuation joins multiview observations into a recognizable
    # frame. Only coordinates supported by detected lines are used; hidden
    # portions between those observations are explicitly inferred, not measured.
    structural_scene = trimesh.Scene()
    continuations = []
    for pi, plane in enumerate(planes):
        rows = [r for r in records if r['plane'] == pi]
        if not rows:
            continue
        axis = _unit(np.cross(up, plane['normal']))
        endpoints = np.concatenate([r['endsNative'] for r in rows])
        lo, hi = np.percentile(endpoints @ axis, [1, 99])
        bottom, top = np.percentile(endpoints @ up + d, [1, 99])
        frame_width = float(np.percentile([r['widthNative'] for r in rows], 90))
        fits = [r for r in refinements if r['plane'] == pi]
        measured = None
        if fits:
            fit = min(fits, key=lambda r: r['heightNative'])
            if fit['heightNative'] < bottom + (top - bottom) / 3:
                measured = max([r for r in rows if r['id'] in fit['members']], key=lambda r: r['lengthNative'])
                bottom = measured['heightNative']
                structural_scene.add_geometry(scene.geometry[measured['id']], geom_name=measured['id'], node_name=measured['id'])
        def position(along, height):
            return along * axis + (height - d) * up - plane['offset'] * plane['normal']
        def add_member(a, b, width, role):
            if np.linalg.norm(b - a) <= width:
                return
            name = f'section-{pi}-continued-{len(continuations)}'
            mesh = _beam(a, b, width, min(width, frame_width * .3), up, plane['normal'])
            structural_scene.add_geometry(mesh, geom_name=name, node_name=name)
            continuations.append({'id': name, 'plane': pi, 'role': role, 'endsNative': [a.tolist(), b.tolist()],
                                  'widthNative': width, 'status': 'inferred continuation of observed collinear members and panel envelope'})
        # Envelope members are a structural model of the observed line extent.
        # They do not imply surveyed corner locations or member thickness.
        for along in [lo, hi]:
            add_member(position(along, bottom), position(along, top), frame_width, 'end-frame')
        add_member(position(lo, top), position(hi, top), frame_width, 'top-frame')
        rail_width = measured['widthNative'] if measured else frame_width
        rail_level = bottom + rail_width / 2
        if measured:
            ma, mb = sorted(np.asarray(measured['endsNative']) @ axis)
            for a, b in [(lo, ma), (mb, hi)]:
                add_member(position(a, rail_level), position(b, rail_level), rail_width, 'lower-rail continuation')
        else:
            add_member(position(lo, rail_level), position(hi, rail_level), rail_width, 'observed lower-envelope hypothesis')
        # Merge repeated horizontal levels and repeated vertical coordinates.
        for horizontal in [True, False]:
            candidates = sorted([r for r in rows if r['horizontal'] == horizontal],
                                key=lambda r: np.mean(r['endsNative'], axis=0) @ (up if horizontal else axis))
            groups = []
            for row in candidates:
                value = float(np.mean(row['endsNative'], axis=0) @ (up if horizontal else axis))
                group = next((g for g in groups if abs(value - np.median([x[0] for x in g])) < max(frame_width, row['widthNative']) * .8), None)
                if group is None:
                    groups.append([(value, row)])
                else:
                    group.append((value, row))
            for group in groups:
                coordinate = float(np.median([x[0] for x in group]))
                width = float(np.median([x[1]['widthNative'] for x in group]))
                if horizontal:
                    height = coordinate + d
                    if bottom + frame_width * 2 < height < top - frame_width * 2:
                        add_member(position(lo, height), position(hi, height), width, 'cross-rail continuation')
                elif lo + frame_width < coordinate < hi - frame_width:
                    add_member(position(coordinate, bottom + rail_width), position(coordinate, top), width, 'vertical-member continuation')
    (root / 'fence-fitted.glb').write_bytes(structural_scene.export(file_type='glb'))
    # Use only the fitted structural footprint for the local floor display.
    # Segmentation may include distant machinery; those pixels must not enlarge
    # the floor mesh used for this workcell.
    structural_points = np.vstack([m.vertices for m in structural_scene.geometry.values()])
    local = structural_points @ np.c_[u, v]
    low, high = np.percentile(local, [1, 99], axis=0)
    margin = (high - low) * .06
    corners = np.array([[low[0]-margin[0], low[1]-margin[1]], [high[0]+margin[0], low[1]-margin[1]],
                        [high[0]+margin[0], high[1]+margin[1]], [low[0]-margin[0], high[1]+margin[1]]])
    xyz = corners[:, :1] * u + corners[:, 1:] * v - d * up
    floor_mesh = trimesh.Trimesh(vertices=xyz, faces=[[0,1,2],[0,2,3]], process=False)
    floor_mesh.visual.vertex_colors = [140, 138, 128, 255]
    (root / 'floor-fitted.glb').write_bytes(floor_mesh.export(file_type='glb'))
    floor['displayExtent'] = 'fitted structural footprint plus six percent margin; plane fit unchanged'
    clearances = []
    for pi, plane in enumerate(planes):
        fits = [r for r in refinements if r['plane'] == pi]
        if not fits:
            continue
        fit = min(fits, key=lambda r: r['heightNative'])
        # Lower-rail evidence must lie in the lower third of the section.
        # A lone middle beam is retained as geometry but never a clearance.
        section = np.concatenate([np.asarray(r['endsNative']) for r in records if r['plane'] == pi])
        low, high = np.percentile(section @ up + d, [2, 98])
        if fit['heightNative'] > low + (high - low) / 3:
            continue
        bars = [r for r in records if r['id'] in fit['members']]
        bar = max(bars, key=lambda r: r['lengthNative'])
        clearances.append({'id': f'fence-plane-{pi}-lower-rail', 'label': f'Fence section {pi + 1} observed lower rail',
                           **{k: bar[k] for k in ['pointNative', 'footNative', 'heightNative']},
                           'sourcePhotos': fit['sourcePhotos'], 'meshNode': bar['id'],
                           'quality': 'provisional multiview line triangulation; inspect source edges',
                           'lineFitResidualNative': fit['lineFitResidualNative'],
                           'viewPlaneCondition': fit['viewPlaneCondition'],
                           'observedViewPointsNative': [r['pointNative'] for r in bars],
                           'observedViewHeightRangeNative': [min(r['heightNative'] for r in bars), max(r['heightNative'] for r in bars)],
                           'method': 'modeled lower rail face to modeled floor along floor normal; section fitted by multiview image lines'})
    evidence = []
    selected_ids = {member for fit in refinements for member in fit['members']
                    if any(c['id'] == f"fence-plane-{fit['plane']}-lower-rail" for c in clearances)
                    and fit['heightNative'] < max(c['heightNative'] for c in clearances) + tolerance * 5}
    for photo, source in enumerate(sources, 1):
        raw = cv2.imread(str(source))
        for row in records:
            if row['sourcePhoto'] != photo or row['id'] not in selected_ids:
                continue
            for edge in row['rawEdges']:
                q = np.rint(edge).astype(int)
                cv2.line(raw, q[0], q[1], (0, 220, 255), max(3, raw.shape[0] // 700))
        for view in anchor['views']:
            if view['photo'] == photo:
                x0, y0, x1, y1 = view['boxRaw']
                cv2.rectangle(raw, (x0, y0), (x1, y1), (255, 100, 0), max(3, raw.shape[0] // 700))
        factor = min(1., 1600 / max(raw.shape[:2]))
        out = cv2.resize(raw, None, fx=factor, fy=factor)
        cv2.putText(out, f'Photo {photo}: cyan = whole reference; yellow = triangulated lower rail',
                    (12, 25), cv2.FONT_HERSHEY_SIMPLEX, .43, (0, 0, 0), 3)
        cv2.putText(out, f'Photo {photo}: cyan = whole reference; yellow = triangulated lower rail',
                    (12, 25), cv2.FONT_HERSHEY_SIMPLEX, .43, (255, 255, 255), 1)
        name = f'geometry-evidence-{photo}.jpg'
        cv2.imwrite(str(root / name), out, [cv2.IMWRITE_JPEG_QUALITY, 88]); evidence.append(name)
        del raw, out
    result = {'schemaVersion': 1, 'coordinateSystem': 'unchanged MapAnything native world; all meshes and cameras scale together',
              'anchor': anchor, 'floor': floor, 'clearances': clearances, 'evidenceImages': evidence,
              'meshes': {'fence': 'fence-fitted.glb', 'floor': 'floor-fitted.glb'},
              'fence': {'planes': [{'normal': p['normal'].tolist(), 'offset': p['offset'], 'supportPoints': len(p['points'])} for p in planes],
                        'beams': records, 'horizontalLineEvidence': line_evidence, 'continuations': continuations, 'optionalObservedSurface': 'fence-surface.glb', 'panels': panel_records,
                        'status': 'structural frame from observed lines with explicitly inferred continuations',
                        'assumptions': ['Each section is upright and planar.', 'Frame envelope and continuations between visible line segments are inferred from observed extent; exact CAD completeness is not claimed.',
                                        'Rail face width is image-supported; extrusion thickness is a two-source-pixel rendering assumption.',
                                        'Independent view observations may overlap. Floor/rail pointmap bias is not a confidence interval.', 'Observed screen surfaces are separate optional fence-surface.glb, not part of the primary structural model.']},
              'limitations': ['Absolute dimensions are conditional on the provisional reference.',
                              'No calibrated absolute-accuracy claim; source cameras and depth are inferred.',
                              'Two-view line residual may be algebraically zero; it is not an accuracy or uncertainty estimate.',
                              'A lower detected rail is not guaranteed to be the globally lowest continuous rail.'],
              'runtime': {'opencv': cv2.__version__, 'numpy': np.__version__},
              'wallSeconds': round(time.monotonic() - start, 3)}
    (root / 'geometry.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--images', type=Path, nargs='+', required=True, help='Two or more photos of one scene, in photo order')
    parser.add_argument('--height-m', type=float, default=.2)
    parser.add_argument('--width-m', type=float, default=.2)
    args = parser.parse_args()
    result = build(args.root, args.images, args.width_m, args.height_m)
    print(json.dumps({'wallSeconds': result['wallSeconds'], 'anchor': result['anchor'],
                      'clearances': result['clearances'], 'beamCount': len(result['fence']['beams'])}, indent=2))
