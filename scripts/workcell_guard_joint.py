"""Source-photo two-plane guard experiment; no generated mesh enters the solver.

build(root, out, cameras=None, tracks=None, sources=None, feature_method='sift') exports A2-independent
and A3-shared. Optional cameras: {frames:[{photo,K,pose}],worldFrame:...}; pose is
camera-to-world and K uses canonical pixels. All cameras must share the original
MapAnything gauge. Optional tracks: {tracks:[{observations:[{photo,uv},...]}]}.
Depth initializes/associates surfaces; RGB feature transfer determines the fit.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
import trimesh
from scipy.optimize import least_squares, linear_sum_assignment
from scipy.spatial import ConvexHull, QhullError
from scipy.spatial.transform import Rotation

from workcell_photo_geometry import _rays, _intersect
from workcell_photo_objects import _project, _inside
from workcell_photo_oneshot import _array, _frame, _mask, _response, GUARD_WORD

SIDES = ('left', 'center', 'right')


def _unit(value):
    value = np.asarray(value, float)
    length = np.linalg.norm(value)
    if not np.isfinite(length) or length < 1e-10:
        raise ValueError('Degenerate direction')
    return value / length


def _json(value):
    return json.loads(Path(value).read_text()) if isinstance(value, (str, Path)) else value


def _triangulate(observations, frames):
    rows, rays = [], []
    for observation in observations:
        frame = frames[observation['photo']]
        pose, K = frame['pose'], frame['K']
        projection = K @ np.linalg.inv(pose)[:3]
        u, v = observation['uv']
        rows.extend([u * projection[2] - projection[0], v * projection[2] - projection[1]])
        rays.append(_unit(_rays(np.asarray([[u, v]]), K, pose)[0]))
    _, _, vt = np.linalg.svd(rows)
    if abs(vt[-1, 3]) < 1e-10:
        return None
    point = vt[-1, :3] / vt[-1, 3]
    errors = []
    for observation in observations:
        uv, depth = _project([point], frames[observation['photo']])
        if depth[0] <= 0:
            return None
        errors.append(float(np.linalg.norm(uv[0] - observation['uv'])))
    parallax = max(np.rad2deg(np.arccos(np.clip(a @ b, -1, 1))) for a in rays for b in rays)
    return point, max(errors), float(parallax)


def _inputs(root, cameras=None, sources=None):
    segmentation = _json(root / 'sam3.json')
    guard = np.load(root / 'guard-input.npz')
    frames, candidates = {}, {}
    replacements = {int(f['photo']): f for f in (_json(cameras) or {}).get('frames', [])}
    if replacements and set(replacements) != set(range(1, 5)):
        raise ValueError('Camera refinement must supply all four cameras in the original world gauge')
    for photo in range(1, 5):
        raw = _frame(root, photo)
        rgb, points = _array(raw['image']), _array(raw['pts3d'])
        original_pose, original_K = _array(raw['camera_poses']), _array(raw['intrinsics'])
        replacement = replacements.get(photo, {})
        pose = np.asarray(replacement.get('pose', original_pose), float)
        K = np.asarray(replacement.get('K', original_K), float)
        if pose.shape != (4, 4) or K.shape != (3, 3) or not np.isfinite(pose).all() or not np.isfinite(K).all():
            raise ValueError('Invalid camera dimensions or nonfinite values')
        if min(K[0, 0], K[1, 1]) <= 0 or not np.allclose(pose[3], [0, 0, 0, 1]) or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-4) or np.linalg.det(pose[:3, :3]) < .9999:
            raise ValueError('Camera must have positive focal lengths and a rigid pose')
        A = np.asarray(raw['input_mask_transform']['input_to_canonical_pixel_centres'])
        analysis_rgb, canonical_to_analysis = rgb, np.eye(3)
        if sources is not None:
            full = cv2.imread(str(sources[photo - 1]))
            if full is None or list(full.shape[:2]) != [raw['original_image']['height'], raw['original_image']['width']]:
                raise ValueError('Raw source dimensions must match recorded canonical transform')
            factor = min(1., 1800 / max(full.shape[:2]))
            analysis_rgb = cv2.cvtColor(cv2.resize(full, None, fx=factor, fy=factor), cv2.COLOR_BGR2RGB)
            resize = np.array([[factor, 0, (factor - 1) / 2], [0, factor, (factor - 1) / 2], [0, 0, 1]])
            canonical_to_analysis = resize @ np.linalg.inv(A)
        frames[photo] = {'photo': photo, 'K': K, 'pose': pose, 'rgb': rgb, 'points': points,
                         'analysisRgb': analysis_rgb, 'C': canonical_to_analysis,
                         'initialPose': original_pose, 'initialK': original_K}
        rows = []
        response = _response(segmentation, photo, GUARD_WORD)
        for index, (encoded, score) in enumerate(zip(response['rle'], response['scores'])):
            mask = _mask(raw, {'rle': [encoded]}) & (guard[f'v{photo}_mask'] > 0)
            valid = mask & np.isfinite(points).all(2)
            if score >= .6 and valid.any():
                rows.append({'photo': photo, 'mask': mask, 'points': points[valid], 'instance': index, 'score': score})
        largest = max((int(row['mask'].sum()) for row in rows), default=0)
        candidates[photo] = [row for row in rows if row['mask'].sum() >= .2 * largest]
    complete = [photo for photo, rows in candidates.items() if len(rows) == 3]
    if not complete:
        raise ValueError('No source photo contains three separate supported board masks')
    anchor = max(complete, key=lambda photo: sum(row['mask'].sum() for row in candidates[photo]))
    seeds = sorted(candidates[anchor], key=lambda row: np.where(row['mask'])[1].mean())
    boards = {side: {'side': side, 'masks': {anchor: seed['mask']}, 'associations': [
        {'photo': anchor, 'instance': seed['instance'], 'score': seed['score'], 'mutualProjection': 1.}]} for side, seed in zip(SIDES, seeds)}
    # Association uses observed pointmaps, never vertices from the generated model.
    init_frames = {photo: {**frame, 'pose': frame['initialPose'], 'K': frame['initialK']} for photo, frame in frames.items()}
    for photo, rows in candidates.items():
        if photo == anchor or not rows:
            continue
        support = np.array([[min(_inside(a['points'], b, init_frames), _inside(b['points'], a, init_frames)) for b in rows] for a in seeds])
        aa, bb = linear_sum_assignment(-support)
        for a, b in zip(aa, bb):
            if support[a, b] < .2:
                continue
            row, board = rows[b], boards[SIDES[a]]
            board['masks'][photo] = row['mask']
            board['associations'].append({'photo': photo, 'instance': row['instance'], 'score': row['score'],
                                          'mutualProjection': float(support[a, b])})
    return frames, boards, {'anchorPhoto': anchor, 'candidateInstancesPerPhoto': {str(k): len(v) for k, v in candidates.items()},
                             'associationBasis': 'mutual source-pointmap reprojection into independent SAM instance masks'}


def _belongs(uv, mask):
    xy = np.rint(uv).astype(int)
    return 0 <= xy[0] < mask.shape[1] and 0 <= xy[1] < mask.shape[0] and bool(mask[xy[1], xy[0]])


def _merge_tracks(nodes, links, side):
    parent = {node: node for node in nodes}
    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node
    for a, b in links:
        aa, bb = find(a), find(b)
        if aa != bb:
            parent[bb] = aa
    groups = {}
    for node in nodes:
        groups.setdefault(find(node), []).append(node)
    tracks, conflicts = [], 0
    for group in groups.values():
        if len({node[0] for node in group}) != len(group):
            conflicts += 1
        elif len(group) >= 2:
            tracks.append({'side': side, 'observations': [nodes[node] for node in sorted(group)]})
    return tracks, conflicts


def _patch_correlation(a, uv_a, b, uv_b):
    for gray, uv in ((a, uv_a), (b, uv_b)):
        if min(uv) < 6 or uv[0] >= gray.shape[1] - 6 or uv[1] >= gray.shape[0] - 6:
            return -1.
    x = cv2.getRectSubPix(a, (11, 11), tuple(map(float, uv_a))).astype(float).ravel()
    y = cv2.getRectSubPix(b, (11, 11), tuple(map(float, uv_b))).astype(float).ravel()
    x -= x.mean(); y -= y.mean()
    return float(x @ y / max(np.linalg.norm(x) * np.linalg.norm(y), 1e-9))


def _lk_candidates(frames, boards):
    """Explicit LK ablation: bounded corners, round-trip/patch/epipolar evidence."""
    candidates, diagnostics = [], []
    gray = {photo: cv2.cvtColor(frame['analysisRgb'], cv2.COLOR_RGB2GRAY) for photo, frame in frames.items()}
    if len({image.shape for image in gray.values()}) != 1:
        raise ValueError('LK experiment requires equal source crop dimensions; no implicit image resizing')
    for side, board in boards.items():
        corners, nodes, links, locations = {}, {}, [], {p: [] for p in board['masks']}
        for photo, mask in board['masks'].items():
            frame = frames[photo]
            high_mask = cv2.warpAffine(mask.astype(np.uint8) * 255, frame['C'][:2], gray[photo].shape[::-1], flags=cv2.INTER_NEAREST)
            high_mask = cv2.erode(high_mask, np.ones((5, 5), np.uint8))
            detected = cv2.goodFeaturesToTrack(gray[photo], maxCorners=240, qualityLevel=.02, minDistance=8, mask=high_mask, blockSize=5)
            corners[photo] = detected.reshape(-1, 2) if detected is not None else np.empty((0, 2), np.float32)
            diagnostics.append({'side': side, 'photo': photo, 'method': 'lk', 'detectedRgbFeatures': len(corners[photo]),
                                'analysisShape': list(gray[photo].shape), 'canonicalShape': list(mask.shape),
                                'canonicalToAnalysisPixelCentres': frame['C'].tolist()})
        def node(photo, position):
            previous = locations[photo]
            distances = np.linalg.norm(np.asarray(previous) - position, axis=1) if previous else np.empty(0)
            if len(distances) and distances.min() <= 3.:
                return photo, int(distances.argmin())
            ident = (photo, len(previous)); previous.append(position.copy())
            uv = np.linalg.inv(frames[photo]['C']) @ np.r_[position, 1.]
            nodes[ident] = {'photo': photo, 'uv': uv[:2].tolist(), 'analysisUv': position.tolist()}
            return ident
        photos = sorted(board['masks'])
        for ai, a in enumerate(photos):
            for b in photos[ai + 1:]:
                counts = {key: 0 for key in ('attempted', 'invalidDepthInitialization', 'flowStatus', 'roundTrip', 'patchError',
                                             'mask', 'patchCorrelation', 'repeatedTextureAmbiguity', 'triangulation', 'acceptedPairs')}
                counts['attempted'] = len(corners[a])
                first = frames[a]; second = frames[b]
                uv = (np.c_[corners[a], np.ones(len(corners[a]))] @ np.linalg.inv(first['C']).T)[:, :2]
                xy = np.rint(uv).astype(int)
                valid = (xy[:, 0] >= 0) & (xy[:, 0] < first['points'].shape[1]) & (xy[:, 1] >= 0) & (xy[:, 1] < first['points'].shape[0])
                starts, estimates = [], []
                for position, pixel, inside in zip(corners[a], xy, valid):
                    point = first['points'][pixel[1], pixel[0]] if inside else np.full(3, np.nan)
                    if not np.isfinite(point).all():
                        counts['invalidDepthInitialization'] += 1; continue
                    projected, depth = _project([point], second)
                    target = (second['C'] @ np.r_[projected[0], 1])[:2]
                    if depth[0] <= 0 or min(target) < 6 or target[0] >= gray[b].shape[1] - 6 or target[1] >= gray[b].shape[0] - 6:
                        counts['invalidDepthInitialization'] += 1; continue
                    starts.append(position); estimates.append(target)
                if starts:
                    p0 = np.asarray(starts, np.float32).reshape(-1, 1, 2)
                    initial = np.asarray(estimates, np.float32).reshape(-1, 1, 2)
                    options = dict(winSize=(31, 31), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, .01),
                                   flags=cv2.OPTFLOW_USE_INITIAL_FLOW, minEigThreshold=1e-4)
                    p1, sf, ef = cv2.calcOpticalFlowPyrLK(gray[a], gray[b], p0, initial.copy(), **options)
                    back, sb, eb = cv2.calcOpticalFlowPyrLK(gray[b], gray[a], p1, p0.copy(), **options)
                    relative_R = second['pose'][:3, :3].T @ first['pose'][:3, :3]
                    t = second['pose'][:3, :3].T @ (first['pose'][:3, 3] - second['pose'][:3, 3])
                    cross = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
                    F = np.linalg.inv(second['K']).T @ cross @ relative_R @ np.linalg.inv(first['K'])
                    for index, (start, end, returned) in enumerate(zip(p0[:, 0], p1[:, 0], back[:, 0])):
                        if not sf[index, 0] or not sb[index, 0] or not np.isfinite([*end, *returned]).all():
                            counts['flowStatus'] += 1; continue
                        if np.linalg.norm(returned - start) > 1.5:
                            counts['roundTrip'] += 1; continue
                        if max(ef[index, 0], eb[index, 0]) > 20:
                            counts['patchError'] += 1; continue
                        ua = (np.linalg.inv(first['C']) @ np.r_[start, 1])[:2]
                        ub = (np.linalg.inv(second['C']) @ np.r_[end, 1])[:2]
                        if not _belongs(ua, board['masks'][a]) or not _belongs(ub, board['masks'][b]):
                            counts['mask'] += 1; continue
                        correlation = _patch_correlation(gray[a], start, gray[b], end)
                        if correlation < .85:
                            counts['patchCorrelation'] += 1; continue
                        line = np.linalg.inv(second['C']).T @ F @ np.r_[ua, 1.]
                        if np.linalg.norm(line[:2]) < 1e-10:
                            counts['triangulation'] += 1; continue
                        tangent = _unit([-line[1], line[0]])
                        alternatives = [_patch_correlation(gray[a], start, gray[b], end + shift * tangent)
                                        for shift in range(-40, 41, 5) if abs(shift) >= 10]
                        if max(alternatives) >= correlation - .03:
                            counts['repeatedTextureAmbiguity'] += 1; continue
                        observations = [{'photo': a, 'uv': ua.tolist()}, {'photo': b, 'uv': ub.tolist()}]
                        tri = _triangulate(observations, frames)
                        if tri is None or tri[1] > 2.5 or tri[2] < .25:
                            counts['triangulation'] += 1; continue
                        links.append((node(a, start), node(b, end))); counts['acceptedPairs'] += 1
                diagnostics.append({'side': side, 'photos': [a, b], 'method': 'lk', 'pairCounts': counts})
        merged, conflicts = _merge_tracks(nodes, links, side)
        kept = []
        for track in sorted(merged, key=lambda t: -len(t['observations'])):
            # ponytail: <=240 seeds/view makes pairwise dedup bounded. Spatial
            # indexing is the upgrade for larger captures. Discard nearby tracks
            # before the train/test split so one image feature cannot leak across it.
            duplicate = any(any(x['photo'] == y['photo'] and np.linalg.norm(np.asarray(x['analysisUv']) - y['analysisUv']) < 6
                                for x in track['observations'] for y in other['observations']) for other in kept)
            if not duplicate:
                kept.append(track)
        candidates.extend(kept)
        diagnostics.append({'side': side, 'method': 'lk', 'conflictingComponentsRejected': conflicts,
                            'mergedTracks': len(merged), 'nearDuplicateTracksRejected': len(merged) - len(kept),
                            'retainedDistinctTracks': len(kept)})
    return candidates, diagnostics


def _features(frames, boards, supplied=None, method='sift'):
    """The explicit observation provider changes; downstream physical gates do not."""
    if method not in ('sift', 'lk'):
        raise ValueError('feature_method must be sift or lk')
    result = {side: [] for side in SIDES}
    diagnostics = []
    if supplied is not None:
        candidates = _json(supplied)['tracks']
    elif method == 'lk':
        candidates, diagnostics = _lk_candidates(frames, boards)
    else:
        candidates = []
        sift = cv2.SIFT_create(nfeatures=5000, contrastThreshold=.012)
        matcher = cv2.BFMatcher()
        for side, board in boards.items():
            descriptors, keypoints = {}, {}
            for photo, mask in board['masks'].items():
                frame = frames[photo]
                image = frame['analysisRgb']
                high_mask = cv2.warpAffine(mask.astype(np.uint8) * 255, frame['C'][:2], image.shape[1::-1], flags=cv2.INTER_NEAREST)
                high_mask = cv2.erode(high_mask, np.ones((3, 3), np.uint8))
                keys, desc = sift.detectAndCompute(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), high_mask)
                uv = np.asarray([key.pt for key in keys], float).reshape(-1, 2)
                keypoints[photo] = (np.c_[uv, np.ones(len(uv))] @ np.linalg.inv(frame['C']).T)[:, :2]
                descriptors[photo] = desc
                diagnostics.append({'side': side, 'photo': photo, 'detectedRgbFeatures': len(keys)})
            links, nodes = [], {}
            photos = sorted(board['masks'])
            for ai, a in enumerate(photos):
                for b in photos[ai + 1:]:
                    if descriptors[a] is None or descriptors[b] is None or min(len(descriptors[a]), len(descriptors[b])) < 2:
                        continue
                    forward = matcher.knnMatch(descriptors[a], descriptors[b], k=2)
                    backward = matcher.knnMatch(descriptors[b], descriptors[a], k=2)
                    reverse = {m.queryIdx: m.trainIdx for m, n in backward if m.distance < .72 * n.distance}
                    for m, n in forward:
                        if m.distance >= .72 * n.distance or reverse.get(m.trainIdx) != m.queryIdx:
                            continue
                        obs = [{'photo': a, 'uv': keypoints[a][m.queryIdx].tolist()}, {'photo': b, 'uv': keypoints[b][m.trainIdx].tolist()}]
                        tri = _triangulate(obs, frames)
                        if tri is None or tri[1] > 2.5 or tri[2] < .25:
                            continue
                        x, y = (a, m.queryIdx), (b, m.trainIdx)
                        nodes[x], nodes[y] = obs
                        links.append((x, y))
            merged, _ = _merge_tracks(nodes, links, side)
            candidates.extend(merged)
    rejected = {'invalidViewCount': 0, 'triangulation': 0, 'instanceAmbiguity': 0}
    for index, track in enumerate(candidates):
        observations = track['observations']
        if len(observations) < 2 or len(set(o['photo'] for o in observations)) != len(observations):
            rejected['invalidViewCount'] += 1
            continue
        tri = _triangulate(observations, frames)
        if tri is None or tri[1] > 2.5 or tri[2] < .25:
            rejected['triangulation'] += 1
            continue
        owners = [side for side, board in boards.items() if all(o['photo'] in board['masks'] and _belongs(o['uv'], board['masks'][o['photo']]) for o in observations)]
        if len(owners) != 1:
            rejected['instanceAmbiguity'] += 1
            continue
        result[owners[0]].append({'id': index, 'xyz': tri[0], 'observations': observations,
                                  'triangulationResidualPx': tri[1], 'parallaxDeg': tri[2]})
    diagnostics.append({'method': 'supplied-tracks' if supplied is not None else method, 'finalTrackRejections': rejected})
    return result, diagnostics


def _plane(points, threshold, seed=7):
    rng = np.random.default_rng(seed)
    best, normal, offset = 0, None, None
    for _ in range(250):
        sample = points[rng.choice(len(points), 3, replace=False)]
        n = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        if np.linalg.norm(n) < 1e-10:
            continue
        n = _unit(n); d = -n @ sample[0]
        count = int((abs(points @ n + d) < threshold).sum())
        if count > best:
            best, normal, offset = count, n, d
    if normal is None:
        raise ValueError('No noncollinear plane support')
    keep = abs(points @ normal + offset) < threshold
    center = np.median(points[keep], axis=0)
    normal = np.linalg.svd(points[keep] - center, full_matrices=False)[2][-1]
    return _unit(normal), -normal @ center, keep


def _seed(points):
    if len(points) < 16:
        raise ValueError('Fewer than sixteen geometric support points')
    extent = float(np.linalg.norm(np.percentile(points, 95, axis=0) - np.percentile(points, 5, axis=0)))
    threshold = max(1e-6, extent * .025)
    n0, d0, keep = _plane(points, threshold)
    if (~keep).sum() < 8:
        raise ValueError('Only one independently supported plane')
    n1, d1, _ = _plane(points[~keep], threshold, 11)
    if abs(n0 @ n1) > np.cos(np.deg2rad(8)):
        raise ValueError('Two candidate planes are nearly parallel')
    for _ in range(4):
        distances = np.stack([abs(points @ n0 + d0), abs(points @ n1 + d1)], axis=1)
        owners = distances.argmin(1)
        fitted = []
        for panel in range(2):
            selected = points[(owners == panel) & (distances[:, panel] < threshold * 2)]
            if len(selected) < 6:
                raise ValueError('Second plane lacks geometric support')
            center = np.median(selected, axis=0)
            n = np.linalg.svd(selected - center, full_matrices=False)[2][-1]
            fitted.append((_unit(n), -n @ center))
        (n0, d0), (n1, d1) = fitted
    axis = _unit(np.cross(n0, n1))
    center = np.median(points, axis=0)
    origin = np.linalg.solve(np.stack([n0, n1, axis]), [-d0, -d1, axis @ center])
    rays = []
    for panel, normal in enumerate((n0, n1)):
        direction = _unit(np.cross(axis, normal))
        if (np.median(points[owners == panel], axis=0) - origin) @ direction < 0:
            direction = -direction
        rays.append(direction)
    if np.cross(rays[0], rays[1]) @ axis < 0:
        axis = -axis
    theta = np.arccos(np.clip(rays[0] @ rays[1], -1, 1))
    rotation = Rotation.from_matrix(np.column_stack([rays[0], np.cross(axis, rays[0]), axis])).as_rotvec()
    return np.r_[origin, rotation, theta], extent


def _geometry(parameters):
    rotation = Rotation.from_rotvec(parameters[3:6]).as_matrix()
    axis, first = rotation[:, 2], rotation[:, 0]
    second = rotation @ [np.cos(parameters[6]), np.sin(parameters[6]), 0]
    normals = np.stack([np.cross(axis, first), np.cross(axis, second)])
    return parameters[:3], axis, np.stack([first, second]), normals


def _assign(parameters, tracks):
    origin, _, _, normals = _geometry(parameters)
    points = np.array([track['xyz'] for track in tracks])
    return abs((points - origin) @ normals.T).argmin(1) if len(points) else np.array([], int)


def _transfer(parameters, tracks, assignments, frames, omit=None):
    origin, _, _, normals = _geometry(parameters)
    residuals, labels = [], []
    for track, panel in zip(tracks, assignments):
        observations = [o for o in track['observations'] if o['photo'] != omit]
        if len(observations) < 2:
            continue
        # Each real feature is transferred both ways; no predicted-depth term
        # can force a good fit when the measured image correspondence disagrees.
        for anchor in observations:
            source = frames[anchor['photo']]
            ray = _rays(np.asarray([anchor['uv']]), source['K'], source['pose'])[0]
            n = normals[panel]
            denominator = ray @ n
            safe = np.copysign(max(abs(denominator), 1e-7), denominator)
            depth = (origin - source['pose'][:3, 3]) @ n / safe
            point = source['pose'][:3, 3] + ray * max(depth, 1e-6)
            for observation in observations:
                if observation is anchor:
                    continue
                uv, z = _project([point], frames[observation['photo']])
                delta = np.clip(uv[0] - observation['uv'], -1000, 1000)
                if depth <= 0 or z[0] <= 0:
                    delta = np.full(2, 1000.)
                residuals.extend(delta)
                labels.append(observation['photo'])
    return np.asarray(residuals), labels


def _solve(initial, tracks, assignments, frames, extent, theta=None, omit=None, max_nfev=90):
    initial = np.asarray(initial, float)
    def expand(value):
        return np.r_[value, theta] if theta is not None else value
    def residual(value):
        parameters = expand(value)
        image, _ = _transfer(parameters, tracks, assignments, frames, omit)
        origin, axis, _, _ = _geometry(parameters)
        # The hinge origin is unobservable along its own line. Fix this gauge;
        # this term does not constrain the two physical planes or their angle.
        gauge = (origin - initial[:3]) @ axis / extent
        return np.r_[image, gauge]
    x0 = initial[:6] if theta is not None else initial
    lower, upper = np.full(len(x0), -np.inf), np.full(len(x0), np.inf)
    if theta is None:
        lower[-1], upper[-1] = np.deg2rad([8, 175])
        x0 = x0.copy(); x0[-1] = np.clip(x0[-1], lower[-1] + 1e-5, upper[-1] - 1e-5)
    fit = least_squares(residual, x0, bounds=(lower, upper), loss='soft_l1', f_scale=.7,
                        x_scale='jac', max_nfev=max_nfev, ftol=1e-7, xtol=1e-7, gtol=1e-7)
    return expand(fit.x), {'converged': bool(fit.success), 'functionEvaluations': fit.nfev,
                           'optimality': float(fit.optimality), 'cost': float(fit.cost)}


def _stats(parameters, tracks, assignments, frames):
    residual, labels = _transfer(parameters, tracks, assignments, frames)
    errors = np.linalg.norm(residual.reshape(-1, 2), axis=1)
    summary = lambda x: {'count': len(x), 'medianPx': float(np.median(x)) if len(x) else None,
                          'p95Px': float(np.percentile(x, 95)) if len(x) else None}
    return {**summary(errors), 'perView': {str(photo): summary(errors[np.asarray(labels) == photo]) for photo in sorted(set(labels))}}


def _leave_view(board, training, testing, frames, photo):
    """Reconstruct a subset from its own observations, with supplied cameras fixed."""
    excluded_tracks = sum(any(o['photo'] == photo for o in track['observations']) for track in training)
    row = {'photo': photo, 'status': 'unsupported', 'angleDeg': None,
           'excludedTrainingTracks': excluded_tracks, 'trainingTracksAfterRetriangulation': 0,
           'provenance': 'Excluded view removed before triangulation, depth initialization, plane assignment and fitting; supplied remaining cameras fixed.'}
    if not excluded_tracks:
        return {**row, 'reason': 'view has no observed training features; removing it is not an independent subset'}
    remaining = {p: frame for p, frame in frames.items() if p != photo}
    subsets = []
    for original in (training, testing):
        subset = []
        for track in original:
            observations = [o for o in track['observations'] if o['photo'] != photo]
            if len(observations) < 2:
                continue
            triangulated = _triangulate(observations, remaining)
            if triangulated is None or triangulated[1] > 2.5 or triangulated[2] < .25:
                continue
            subset.append(({**track, 'observations': observations, 'xyz': triangulated[0],
                            'triangulationResidualPx': triangulated[1], 'parallaxDeg': triangulated[2]}, track))
        subsets.append(subset)
    row['trainingTracksAfterRetriangulation'] = len(subsets[0])
    if not subsets[0]:
        return {**row, 'reason': 'no training tracks triangulatable after excluding observed view'}
    validation_pairs = [(subset, original) for group in subsets for subset, original in group
                        if any(o['photo'] == photo for o in original['observations'])]
    if not validation_pairs:
        return {**row, 'reason': 'excluded view has no validation track independently triangulatable in remaining views'}
    source = [remaining[p]['points'][mask][::max(1, int(mask.sum()) // 1200)]
              for p, mask in board['masks'].items() if p != photo]
    if not source:
        return {**row, 'reason': 'no remaining source masks'}
    source = np.concatenate(source)
    subset_training = [track for track, _ in subsets[0]]
    try:
        try:
            initial, extent = _seed(np.asarray([track['xyz'] for track in subset_training]))
        except ValueError:
            initial, extent = _seed(source)
        assignments = _assign(initial, subset_training)
        if any((assignments == panel).sum() < 4 for panel in range(2)):
            return {**row, 'reason': 'fewer than four training tracks per plane after excluding view'}
        fitted, diagnostic = _solve(initial, subset_training, assignments, remaining, extent, max_nfev=55)
    except (ValueError, np.linalg.LinAlgError) as error:
        return {**row, 'reason': str(error)}
    if not diagnostic['converged']:
        return {**row, 'reason': 'leave-one-view optimizer did not converge', 'optimizer': diagnostic}
    # The excluded image enters only this evaluation. Plane assignment comes
    # from each track re-triangulated in the remaining views, including test tracks.
    assignments = _assign(initial, [subset for subset, _ in validation_pairs])
    validation = _stats(fitted, [original for _, original in validation_pairs], assignments, frames)
    excluded_residual = validation['perView'].get(str(photo))
    if not excluded_residual or not excluded_residual['count']:
        return {**row, 'reason': 'no independently predicted excluded-view residual', 'optimizer': diagnostic}
    return {**row, 'status': 'fit', 'angleDeg': float(np.rad2deg(fitted[6])), 'optimizer': diagnostic,
            'validationIncludingExcludedView': validation, 'excludedViewResidual': excluded_residual}


def _fit_board(board, tracks, frames):
    source = np.concatenate([frames[photo]['points'][mask][::max(1, int(mask.sum()) // 1200)] for photo, mask in board['masks'].items()])
    train = [index for index in range(len(tracks)) if index % 5 != 0]
    test = [index for index in range(len(tracks)) if index % 5 == 0]
    training, testing = [tracks[i] for i in train], [tracks[i] for i in test]
    try:
        initial, extent = _seed(np.asarray([track['xyz'] for track in training]))
    except ValueError:
        initial, extent = _seed(source)
    assignments = _assign(initial, tracks)
    a_train, a_test = assignments[train], assignments[test]
    parameters, fit = initial, {'converged': False, 'reason': 'insufficient feature support'}
    if min([(a_train == panel).sum() for panel in range(2)], default=0) >= 4:
        parameters, fit = _solve(initial, training, a_train, frames, extent)
    leave_views = [_leave_view(board, training, testing, frames, photo) for photo in sorted(board['masks'])]
    return {'parameters': parameters, 'initial': initial, 'extent': extent, 'tracks': tracks,
            'assignments': assignments, 'training': training, 'testing': testing, 'trainAssignments': a_train,
            'testAssignments': a_test, 'fit': fit, 'leaveViews': leave_views}


def _shared(left, right, frames):
    boards = [left, right]
    x0 = np.r_[left['parameters'][:6], right['parameters'][:6], np.mean([left['parameters'][6], right['parameters'][6]])]
    def residual(values):
        chunks = []
        for index, board in enumerate(boards):
            p = np.r_[values[index * 6:index * 6 + 6], values[-1]]
            r, _ = _transfer(p, board['training'], board['trainAssignments'], frames)
            origin, axis, _, _ = _geometry(p)
            chunks.extend([r, [(origin - board['initial'][:3]) @ axis / board['extent']]])
        return np.concatenate(chunks)
    lower, upper = np.full(13, -np.inf), np.full(13, np.inf)
    lower[-1], upper[-1] = np.deg2rad([8, 175])
    fit = least_squares(residual, x0, bounds=(lower, upper), loss='soft_l1', f_scale=.7, x_scale='jac', max_nfev=120)
    return [np.r_[fit.x[i * 6:i * 6 + 6], fit.x[-1]] for i in range(2)], {'converged': bool(fit.success), 'functionEvaluations': fit.nfev}


def _mesh(parameters, board, frames, out):
    origin, axis, directions, normals = _geometry(parameters)
    scene, records = trimesh.Scene(), []
    for panel in range(2):
        collected = []
        for photo, mask in board['masks'].items():
            frame = frames[photo]
            yy, xx = np.nonzero(mask)
            step = max(1, len(xx) // 2500)
            uv = np.c_[xx[::step], yy[::step]]
            measured = frame['points'][uv[:, 1], uv[:, 0]]
            owners = abs((measured - origin) @ normals.T).argmin(1)
            uv = uv[owners == panel]
            points = _intersect(uv, frame['K'], frame['pose'], normals[panel], -normals[panel] @ origin)
            good = np.isfinite(points).all(1) & ((points - origin) @ directions[panel] > 0)
            points = points[good]
            if not len(points):
                continue
            hits = np.zeros(len(points), int)
            for other, other_mask in board['masks'].items():
                projected, depth = _project(points, frames[other])
                xy = np.rint(projected).astype(int)
                inside = (depth > 0) & (xy[:, 0] >= 0) & (xy[:, 0] < other_mask.shape[1]) & (xy[:, 1] >= 0) & (xy[:, 1] < other_mask.shape[0])
                hits[inside] += other_mask[xy[inside, 1], xy[inside, 0]]
            collected.append(points[hits >= min(2, len(board['masks']))])
        points = np.concatenate(collected) if collected else np.empty((0, 3))
        if len(points) < 6:
            raise ValueError(f'Panel {panel + 1} lacks a multiview supported visible extent')
        local = np.c_[(points - origin) @ directions[panel], (points - origin) @ axis]
        low, high = np.percentile(local[:, 1], [2, 98])
        # Boundary samples delimit extent; interior stripe edges never determine
        # the hinge. Convex completion and its hidden regions remain inferred.
        polygon = local[ConvexHull(local).vertices]
        polygon = np.vstack([polygon, [0, low], [0, high]])
        polygon = polygon[ConvexHull(polygon).vertices]
        vertices = origin + polygon[:, :1] * directions[panel] + polygon[:, 1:] * axis
        mesh = trimesh.Trimesh(vertices=vertices, faces=[[0, j, j + 1] for j in range(1, len(vertices) - 1)], process=False)
        # Densify in-plane so vertex colors retain the identifiable photo stripes.
        for _ in range(4):
            mesh = mesh.subdivide()
        colors = np.zeros((len(mesh.vertices), 4), np.uint8); colors[:, 3] = 255
        priority = sorted(board['masks'], key=lambda p: -int(board['masks'][p].sum()))
        colored = np.zeros(len(mesh.vertices), bool)
        for photo in priority:
            frame, mask = frames[photo], board['masks'][photo]
            uv, depth = _project(mesh.vertices, frame)
            xy = np.rint(uv).astype(int)
            valid = (depth > 0) & (xy[:, 0] >= 0) & (xy[:, 0] < mask.shape[1]) & (xy[:, 1] >= 0) & (xy[:, 1] < mask.shape[0])
            idx = np.flatnonzero(valid & ~colored)
            idx = idx[mask[xy[idx, 1], xy[idx, 0]]]
            high_uv = (np.c_[uv[idx], np.ones(len(idx))] @ frame['C'].T)[:, :2]
            hxy = np.rint(high_uv).astype(int)
            image = frame['analysisRgb']; hxy[:, 0] = hxy[:, 0].clip(0, image.shape[1] - 1); hxy[:, 1] = hxy[:, 1].clip(0, image.shape[0] - 1)
            colors[idx, :3] = image[hxy[:, 1], hxy[:, 0]]; colored[idx] = True
        colors[~colored, :3] = [115, 115, 115]
        mesh.visual.vertex_colors = colors
        scene.add_geometry(mesh, geom_name=f'{board["side"]}-panel-{panel + 1}')
        records.append({'panel': panel + 1, 'observedExtentPoints': len(points), 'widthNative': float(polygon[:, 0].max()),
                        'hingeSpanNative': [float(low), float(high)], 'photoColorFraction': float(colored.mean()),
                        'normal': normals[panel].tolist(), 'polygonNative': vertices.tolist()})
    file = f'guard-{board["side"]}.glb'; scene.export(out / file)
    vertex_sets = [np.asarray(mesh.vertices) for mesh in scene.geometry.values()]
    # Read the mesh's outward panel rays back, independently of the scalar field.
    rays = [_unit((v.mean(0) - origin) - ((v.mean(0) - origin) @ axis) * axis) for v in vertex_sets]
    mesh_angle = float(np.rad2deg(np.arccos(np.clip(rays[0] @ rays[1], -1, 1))))
    return file, records, mesh_angle


def _record(board, fitted, parameters, frames, out, anchor, floor):
    train = _stats(parameters, fitted['training'], fitted['trainAssignments'], frames)
    holdout = _stats(parameters, fitted['testing'], fitted['testAssignments'], frames)
    support = []
    for panel in range(2):
        tracks = [t for t, a in zip(fitted['tracks'], fitted['assignments']) if a == panel]
        photos = sorted({o['photo'] for t in tracks for o in t['observations']})
        cloud = np.array([t['xyz'] for t in tracks])
        spread = np.linalg.svd(cloud - cloud.mean(0), compute_uv=False) if len(cloud) >= 3 else np.zeros(3)
        test_tracks = [t for t, a in zip(fitted['testing'], fitted['testAssignments']) if a == panel]
        support.append({'panel': panel + 1, 'rgbTracks': len(tracks), 'trainingTracks': int((fitted['trainAssignments'] == panel).sum()),
                        'heldOutTracks': int((fitted['testAssignments'] == panel).sum()), 'sourcePhotos': photos,
                        'spatialSpreadRatio': float(spread[1] / max(spread[0], 1e-9)),
                        'heldOutResidual': _stats(parameters, test_tracks, np.full(len(test_tracks), panel), frames),
                        'medianParallaxDeg': float(np.median([t['parallaxDeg'] for t in tracks])) if tracks else None})
    reasons = []
    if any(s['trainingTracks'] < 6 or s['heldOutTracks'] < 2 or len(s['sourcePhotos']) < 2 for s in support):
        reasons.append('insufficient independent RGB support on both planes')
    if any(s['medianParallaxDeg'] is None or s['medianParallaxDeg'] < 1 for s in support):
        reasons.append('weak triangulation parallax')
    if any(s['spatialSpreadRatio'] < .08 for s in support):
        reasons.append('nearly collinear feature support cannot identify both plane orientations')
    if any(s['heldOutResidual']['medianPx'] is None or s['heldOutResidual']['medianPx'] > 1.5 or s['heldOutResidual']['p95Px'] > 4 for s in support):
        reasons.append('at least one plane fails its separate held-out feature check')
    if holdout['medianPx'] is None or holdout['medianPx'] > 1.5 or holdout['p95Px'] > 4:
        reasons.append('held-out feature transfer residual exceeds 1.5 px median or 4 px p95 in canonical image')
    if not fitted['fit']['converged']:
        reasons.append('independent optimizer did not converge')
    angles = [r['angleDeg'] for r in fitted['leaveViews'] if r['status'] == 'fit' and r['optimizer']['converged']]
    if len(angles) < 2:
        reasons.append('fewer than two leave-one-view fits; camera/view uncertainty unresolved')
    elif np.ptp(angles) > 12:
        reasons.append('leave-one-view angle variation exceeds 12 degrees')
    mesh_file, panels, mesh_angle = None, [], None
    try:
        mesh_file, panels, mesh_angle = _mesh(parameters, board, frames, out)
    except (ValueError, QhullError) as error:
        reasons.append(str(error))
    angle = float(np.rad2deg(parameters[6]))
    if mesh_angle is not None and abs(mesh_angle - angle) > 1e-4:
        raise ValueError('Exported mesh angle does not agree with solved geometry')
    origin, axis, directions, normals = _geometry(parameters)
    scale = anchor.get('mPerNative')
    vertices = np.concatenate([p['polygonNative'] for p in panels]) if panels else np.empty((0, 3))
    clearance = float(np.min(vertices @ np.asarray(floor['normal']) + floor['offset'])) if len(vertices) else None
    candidate_geometry = {'groundClearanceNative': clearance,
                          'groundClearanceM': clearance * scale if clearance is not None and scale else None,
                          'panelWidthsM': [p['widthNative'] * scale for p in panels] if scale else None}
    return {'id': 'v-guard-' + board['side'], 'status': 'unsupported' if reasons else 'image-supported-conditional',
            'reasons': reasons, 'candidateAngleDeg': angle, 'measurementAngleDeg': None if reasons else angle,
            'initialDepthAngleDeg': float(np.rad2deg(fitted['initial'][6])), 'mesh': mesh_file,
            'meshAngleDeg': mesh_angle, 'panels': panels, 'originNative': origin.tolist(), 'hingeAxis': axis.tolist(),
            'panelDirections': directions.tolist(), 'planeNormals': normals.tolist(), 'candidateGeometry': candidate_geometry,
            **{key: None if reasons else value for key, value in candidate_geometry.items()},
            'sourceAssociations': board['associations'], 'support': support, 'trainingResidual': train,
            'heldOutFeatureResidual': holdout, 'leaveOneView': fitted['leaveViews'], 'optimizer': fitted['fit'],
            'conditionalAngleRangeDeg': [min(angles), max(angles)] if len(angles) >= 2 else None,
            'uncertaintyScope': 'Independent-fit view-subset sensitivity conditional on fixed inferred cameras and feature associations; not an absolute confidence interval or uncertainty of the shared-angle fit.',
            'geometryScope': 'Two zero-thickness photo-colored sheets; boundary convex completion and hidden hinge extent are inferred; thickness unknown.'}


def build(root, out, cameras=None, tracks=None, sources=None, feature_method='sift'):
    start = time.monotonic(); root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if sources is not None and len(sources) != 4:
        raise ValueError('Exactly four source image paths required')
    if feature_method == 'lk' and (sources is None or tracks is not None):
        raise ValueError('Explicit LK experiment requires raw source photos and cannot also supply external tracks')
    geometry = _json(root / 'geometry.json')
    frames, boards, association = _inputs(root, cameras, sources)
    features, detection = _features(frames, boards, tracks, feature_method)
    observation_method = 'supplied-tracks' if tracks is not None else feature_method
    observation_payload = {'association': association, 'featureMethod': observation_method, 'detection': detection, 'boards': {
        side: [{**track, 'xyz': track['xyz'].tolist()} for track in rows] for side, rows in features.items()}}
    observation_text = json.dumps(observation_payload, indent=2, allow_nan=False)
    (out / 'observations.json').write_text(observation_text + '\n')
    camera_payload = {'worldFrame': 'MapAnything native', 'frames': [{k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in frame.items() if k in ('photo', 'K', 'pose')} for frame in frames.values()]}
    (out / 'cameras.json').write_text(json.dumps(camera_payload, indent=2) + '\n')
    fitted, errors = {}, {}
    for side, board in boards.items():
        try:
            fitted[side] = _fit_board(board, features[side], frames)
        except (ValueError, np.linalg.LinAlgError) as error:
            errors[side] = str(error)
    shared, shared_diagnostic = {}, {'converged': False, 'reason': 'left/right independent fits unavailable'}
    if all(side in fitted and min((fitted[side]['trainAssignments'] == p).sum() for p in range(2)) >= 4 for side in ('left', 'right')):
        values, shared_diagnostic = _shared(fitted['left'], fitted['right'], frames)
        shared = dict(zip(('left', 'right'), values))
    routes = {}
    for route in ('A2-independent', 'A3-shared'):
        destination = out / route; destination.mkdir(exist_ok=True)
        rows = []
        for side in SIDES:
            if side not in fitted:
                rows.append({'id': 'v-guard-' + side, 'status': 'unsupported', 'reasons': [errors[side]], 'mesh': None,
                             'measurementAngleDeg': None, 'groundClearanceNative': None, 'groundClearanceM': None,
                             'panelWidthsM': None, 'candidateGeometry': {}})
                continue
            parameters = shared.get(side, fitted[side]['parameters']) if route == 'A3-shared' else fitted[side]['parameters']
            row = _record(boards[side], fitted[side], parameters, frames, destination, geometry['anchor'], geometry['floor'])
            row['independentCandidateAngleDeg'] = float(np.rad2deg(fitted[side]['parameters'][6]))
            if route == 'A3-shared' and side in ('left', 'right'):
                row['sharedAngleApplied'] = side in shared
                independent_error = _stats(fitted[side]['parameters'], fitted[side]['testing'], fitted[side]['testAssignments'], frames)
                row['independentHeldOutFeatureResidual'] = independent_error
                before, after = independent_error['medianPx'], row['heldOutFeatureResidual']['medianPx']
                row['sharedPriorHoldoutDeltaMedianPx'] = after - before if before is not None and after is not None else None
                if before is not None and after is not None and after > before + max(.25, .2 * before):
                    row['reasons'].append('shared prior worsens held-out median by more than max(0.25 canonical px, 20 percent)')
                    row['status'], row['measurementAngleDeg'] = 'unsupported', None
                if side not in shared or not shared_diagnostic['converged']:
                    row['reasons'].append('shared-angle optimizer unavailable or unconverged')
                    row['status'], row['measurementAngleDeg'] = 'unsupported', None
            if row['status'] == 'unsupported':
                for key in row['candidateGeometry']:
                    row[key] = None
            rows.append(row)
        result = {'schemaVersion': 1, 'route': route, 'coordinateSystem': 'unchanged MapAnything native gauge',
                  'method': 'two intersecting planes fitted to symmetric RGB feature transfer; depth only initializes and associates',
                  'optimizedVariables': 'per-board hinge origin, orientation, interior angle; A3 shares only left/right angle',
                  'fixedVariables': 'all camera intrinsics and camera poses supplied to this route; inherited floor and reference scale',
                  'cameraSource': 'supplied refinement' if cameras else 'MapAnything initialization',
                  'featureMethod': observation_method,
                  'lkObservationGates': {'maxCornersPerBoardView': 240, 'roundTripMaxAnalysisPx': 1.5,
                                         'patchL1Max': 20, 'patchCorrelationMin': .85, 'alternateEpipolarPatchMarginMin': .03,
                                         'triangulationMaxCanonicalPx': 2.5, 'parallaxMinDeg': .25,
                                         'trackMergeAnalysisPx': 3, 'trackDedupAnalysisPx': 6,
                                         'depthUse': 'target flow initialization only; observed pixel displacement determines correspondence'} if feature_method == 'lk' else None,
                  'cameraFingerprint': hashlib.sha256(json.dumps(camera_payload, sort_keys=True).encode()).hexdigest(),
                  'observationFingerprint': hashlib.sha256(observation_text.encode()).hexdigest(), 'objects': rows,
                  'sourceImageResolution': 'raw resized to max 1800' if sources else 'canonical RGB only',
                  'association': association, 'anchor': geometry['anchor'], 'floor': geometry['floor'],
                  'sharedPrior': {'appliesTo': ['left', 'right'], 'source': 'user same-specification prior',
                                  'optimizer': shared_diagnostic, 'equalityIsAccuracyEvidence': False,
                                  'holdoutAcceptance': 'Each board must pass independent support gates and median may worsen by at most max(0.25 canonical px, 20 percent).',
                                  'supportedByCurrentEvidence': all(r['status'] == 'image-supported-conditional' for r in rows if r['id'] in ('v-guard-left', 'v-guard-right'))} if route == 'A3-shared' else None,
                  'limitations': ['No real angle or metric ground truth supplied.', 'Camera calibration, SAM boundaries, repeated-stripe correspondences and source depth are uncertain.',
                                  'Button height determines inherited global scale; its width remains an independent hypothesis check. Reference scale is not re-estimated after camera changes.', 'Ground reference is inherited in original gauge and is not re-fitted.',
                                  'Holdout tracks never enter optimization; assignments use initialization. Leave-view subsets re-triangulate and initialize after excluding that view, but supplied cameras and feature identities remain fixed.',
                                  'Guard rigidity across photos is assumed, not established. Robot/cart motion is possible; residuals alone cannot distinguish object motion from camera, correspondence or depth errors.'],
                  'wallSeconds': round(time.monotonic() - start, 3)}
        (destination / 'results.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        routes[route] = result
    return routes


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cameras', type=Path)
    parser.add_argument('--tracks', type=Path)
    parser.add_argument('--images', type=Path, nargs=4)
    parser.add_argument('--feature-method', choices=('sift', 'lk'), default='sift')
    args = parser.parse_args()
    results = build(args.root, args.out, args.cameras, args.tracks, args.images, args.feature_method)
    print(json.dumps({key: [{'id': row['id'], 'status': row['status'], 'angle': row.get('candidateAngleDeg'), 'reasons': row['reasons']} for row in value['objects']] for key, value in results.items()}, indent=2))
