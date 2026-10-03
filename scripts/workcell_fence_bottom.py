"""RGB candidates for the same lower-rail underside on either fence instance.

Paired horizontal borders with repeated upright terminations establish part
identity. Partial views only supply geometric candidates; the caller must
require a strong identity witness and finite cross-view correspondence. Saved
depth establishes the fence plane/instance, never a bottom percentile.
"""
import itertools

import cv2
import numpy as np

from workcell_photo_geometry import _edge_overlap, _intersect, _unit


def _plane_index(item):
    # refresh_catalog preserves observation.source as provenance when refreshed
    # planes reorder. Current structured identity owns the mapping.
    from workcell_photo_metrology import _fence_plane_index
    return _fence_plane_index(item)


def _picket_support(vertical, axis, up, left, right, lower_edge, upper_edge, width):
    """Repeated bars terminate above a bottom rail; middle rails fail below test."""
    groups = []
    lower_sh = np.c_[np.asarray(lower_edge) @ axis, np.asarray(lower_edge) @ up]
    upper_sh = np.c_[np.asarray(upper_edge) @ axis, np.asarray(upper_edge) @ up]
    def height(edge, along):
        return float(edge[0, 1] + (along - edge[0, 0]) * (edge[1, 1] - edge[0, 1]) / (edge[1, 0] - edge[0, 0]))
    for row in vertical:
        ends = np.asarray(row['world'])
        sh = np.c_[ends @ axis, ends @ up]
        # Incidence belongs to the observed lines at this picket. A global mean
        # rail height incorrectly rejects a sloping projected edge, and a long
        # picket midpoint is not its contact coordinate under noisy cameras.
        design = np.c_[sh[1] - sh[0], -(upper_sh[1] - upper_sh[0])]
        if abs(np.linalg.det(design)) < 1e-12:
            continue
        _, fraction = np.linalg.solve(design, upper_sh[0] - sh[0])
        along = float(upper_sh[0, 0] + fraction * (upper_sh[1, 0] - upper_sh[0, 0]))
        if not left + width < along < right - width:
            continue
        # Collinear edge fragments share identity. Rail thickness is unrelated
        # to picket spacing and must not merge several adjacent uprights.
        pixel_native = np.linalg.norm(ends[1] - ends[0]) / row['lengthRaw']
        group = next((group for group in groups if abs(group['along'] - along) < 2 * pixel_native), None)
        if group is None:
            group = {'along': along, 'segments': []}
            groups.append(group)
        group['segments'].append((float(np.min(ends @ up)), float(np.max(ends @ up)), row['index']))
    supported, continuing = [], []
    for group in groups:
        upper, lower = height(upper_sh, group['along']), height(lower_sh, group['along'])
        local_width = upper - lower
        if local_width <= 0:
            continue
        above = [index for lo, hi, index in group['segments']
                 # Uprights attach to the rail's top face, behind the front
                 # border in perspective. Contact need not coincide with the
                 # front-face upper line; this is topology, not an endpoint fit.
                 if abs(lo - upper) <= local_width and hi - upper >= local_width * 3]
        below = [index for lo, hi, index in group['segments']
                 if hi >= lower - local_width * .3 and lo <= lower - local_width * 3]
        if above:
            (continuing if below else supported).append({**group, 'aboveSegmentIds': above, 'belowSegmentIds': below})
    span = np.ptp([row['along'] for row in supported]) if len(supported) > 1 else 0.
    # ponytail: six distinct upright RGB edges require more support than two
    # two-sided end posts. They are edge groups, not six physical pickets.
    # Shorter occluded spans need additional views, not invented terminals.
    accepted = len(supported) >= 6 and span >= max(width * 4, (right - left) * .25) and not continuing
    return bool(accepted), {'terminatingPickets': supported, 'continuingPickets': continuing,
                            'supportedSpanNative': float(span)}


def _join_horizontal(rows, frame, normal, offset):
    """Pickets interrupt a rail border; preserve actual fragments across gaps."""
    from workcell_photo_metrology import _interval_union, _pixels

    groups = []
    for row in rows:
        chosen = None
        for group in groups:
            points = np.concatenate([member['raw'] for member in group] + [row['raw']])
            vx, vy, x, y = cv2.fitLine(points.astype(np.float32), cv2.DIST_L2, 0, .01, .01).ravel()
            if np.max(abs((points - [x, y]) @ [-vy, vx])) <= 1.5:
                chosen = group
                break
        if chosen is None:
            groups.append([row])
        else:
            chosen.append(row)
    joined = []
    for group in groups:
        segments = np.asarray([row['raw'] for row in group])
        vx, vy, x, y = cv2.fitLine(segments.reshape(-1, 2).astype(np.float32), cv2.DIST_L2, 0, .01, .01).ravel()
        axis, center = np.array([vx, vy]), np.array([x, y])
        intervals = _interval_union(sorted((segment - center) @ axis) for segment in segments)
        raw = center + np.asarray([intervals[0][0], intervals[-1][1]])[:, None] * axis
        visible = sum(b - a for a, b in intervals)
        # Occluding posts and pickets create real gaps. Keep fragments rather
        # than discarding a line for low envelope coverage; finite multiview
        # support is checked from rawSegments, never the gap-spanning envelope.
        world = _intersect(_pixels(raw, frame['A']), frame['K'], frame['pose'], normal, offset)
        if np.isfinite(world).all():
            joined.append({**group[0], 'raw': raw, 'world': world, 'rawSegments': segments,
                           'fragmentIds': [row['index'] for row in group], 'lengthRaw': float(visible)})
    return joined


def fence_bottom_candidates(item, geometry, frames, up):
    """Return source RGB lower edges and diagnostics, without editing a model."""
    # Lazy imports avoid a cycle when metrology dispatches here for both fences.
    from workcell_photo_metrology import _pixels, _sample

    up = np.asarray(up, float)
    if up.shape != (3,) or not np.isfinite(up).all() or np.linalg.norm(up) < 1e-9:
        raise ValueError('Expected a finite nonzero common up vector')
    up = _unit(up)
    plane_index = _plane_index(item)
    planes = geometry['fence']['planes']
    if not 0 <= plane_index < len(planes):
        raise ValueError('Fence plane identity is outside saved geometry')
    plane = planes[plane_index]
    normal = np.asarray(plane['normal'], float)
    if normal.shape != (3,) or not np.isfinite(normal).all() or np.linalg.norm(np.cross(up, normal)) < 1e-9:
        raise ValueError('Fence plane must have a supported horizontal axis')
    axis = _unit(np.cross(up, normal))
    candidates, diagnostics = [], []
    for observation in item['observations']:
        photo = int(observation['photo'])
        frame = frames[photo]
        rgb, A = np.asarray(frame['rgb']), np.asarray(frame['A'], float)
        canonical = np.zeros(frame['shape'], np.uint8)
        polygons = [np.rint(p).astype(np.int32) for p in observation.get('polygons', []) if len(p) >= 3]
        if not polygons:
            diagnostics.append({'photo': photo, 'reason': 'No instance-specific source polygons'})
            continue
        cv2.fillPoly(canonical, polygons, 1)
        instance_mask = cv2.warpPerspective(canonical, np.linalg.inv(A), rgb.shape[1::-1], flags=cv2.INTER_NEAREST)
        y, x = np.where(instance_mask)
        if len(x) < 20:
            diagnostics.append({'photo': photo, 'reason': 'Insufficient instance mask support'})
            continue
        # Four canonical cells bound the association ROI around a mask whose
        # depth filter removed mixed silhouette samples. The exact edge remains
        # an RGB observation, independent of this search margin and image crop.
        margin = max(4, int(np.ceil(4 * np.linalg.norm(np.linalg.inv(A)[:2, :2], ord=2))))
        # Catalog polygons were filtered by depth-to-plane consistency. Mixed
        # depth at a true RGB silhouette can remove the rail underside itself.
        # They locate the instance only; semantic SAM membership supports the
        # face interior. Neither mask defines the physical terminal pixel.
        semantic = np.asarray(frame['semanticMasks']['safety fence'], np.uint8)
        if semantic.shape != canonical.shape:
            raise ValueError('Saved semantic fence mask must use the canonical raster')
        mask = cv2.warpPerspective(semantic, np.linalg.inv(A), rgb.shape[1::-1], flags=cv2.INTER_NEAREST)
        origin = np.maximum([x.min() - margin, y.min() - margin], 0)
        high = np.minimum([x.max() + margin + 1, y.max() + margin + 1], rgb.shape[1::-1])
        crop = rgb[origin[1]:high[1], origin[0]:high[0]]
        factor = min(1., 1600 / max(crop.shape[:2]))
        gray = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY), None, fx=factor, fy=factor)
        detected = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(gray)[0]
        hull = np.zeros_like(instance_mask)
        cv2.fillConvexPoly(hull, cv2.convexHull(np.c_[x, y].astype(np.int32)), 1)
        loose_mask = cv2.dilate(hull, np.ones((margin * 2 + 1,) * 2, np.uint8))
        rows = []
        for index, segment in enumerate([] if detected is None else detected.reshape(-1, 2, 2)):
            raw = (segment.astype(float) + .5) / factor - .5 + origin
            length = np.linalg.norm(raw[1] - raw[0])
            if length < max(8., 8 / factor):
                continue
            samples = raw[0] + np.linspace(.05, .95, 15)[:, None] * (raw[1] - raw[0])
            if _sample(loose_mask, samples).mean() < .8:
                continue
            world = _intersect(_pixels(raw, A), frame['K'], frame['pose'], normal, plane['offset'])
            if not np.isfinite(world).all():
                continue
            direction = _unit(world[1] - world[0])
            horizontal, vertical = abs(direction @ up) < .14, abs(direction @ up) > .985
            if horizontal or vertical:
                rows.append({'index': index, 'raw': raw, 'world': world, 'horizontal': horizontal,
                             'lengthRaw': float(length)})
        # Keep all upright evidence: global longest-N selection deletes lower
        # picket contacts when upper bars are long. Only paired horizontal
        # proposals are capped after joining; LSD uses a bounded 1600px image.
        vertical = [row for row in rows if not row['horizontal']]
        horizontal_raw = [row for row in rows if row['horizontal']]
        horizontal = _join_horizontal(horizontal_raw, frame, normal, plane['offset'])
        horizontal = sorted(horizontal, key=lambda row: -row['lengthRaw'])[:200]
        detail = {'photo': photo, 'plane': plane_index, 'rgbLines': 0 if detected is None else len(detected),
                  'horizontalFragmentsBeforeJoining': len(horizontal_raw),
                  'verticalFragmentsBeforeCap': sum(not row['horizontal'] for row in rows),
                  'maskPolicy': 'Depth-filtered instance polygons locate ROI; unfiltered semantic fence mask supports RGB face interior',
                  'horizontalLines': len(horizontal), 'verticalLines': len(vertical), 'pairs': []}
        diagnostics.append(detail)
        for a, b in itertools.combinations(horizontal, 2):
            overlap, supported = _edge_overlap(a['world'], b['world'], axis)
            if not supported:
                continue
            lower, upper = sorted([a, b], key=lambda row: np.mean(row['world'] @ up))
            lo, hi = (float(np.mean(row['world'] @ up)) for row in (lower, upper))
            width = hi - lo
            pixel_native = np.linalg.norm(lower['world'][1] - lower['world'][0]) / lower['lengthRaw']
            if not 1.5 * pixel_native < width < overlap * .25:
                continue
            left = max(np.min(row['world'] @ axis) for row in (a, b))
            right = min(np.max(row['world'] @ axis) for row in (a, b))
            t = np.linspace(.1, .9, 17)
            # Actual paired-face interior must belong to this fence instance.
            samples = []
            for row in (a, b):
                along = row['world'] @ axis
                fraction = (left + t * (right - left) - along[0]) / (along[1] - along[0])
                samples.append(row['raw'][0] + fraction[:, None] * (row['raw'][1] - row['raw'][0]))
            fraction = float(_sample(mask, (samples[0] + samples[1]) / 2).mean())
            if fraction < .65:
                continue
            lower_samples, upper_samples = samples if lower is a else samples[::-1]
            below_fraction = float(_sample(mask, lower_samples + .3 * (lower_samples - upper_samples)).mean())
            pair = {'lowerSegment': lower['index'], 'upperSegment': upper['index'],
                    'rawLowerEnds': lower['raw'].tolist(), 'rawUpperEnds': upper['raw'].tolist(),
                    'instanceInteriorFraction': fraction,
                    'fenceMaskContinuesBelowFraction': below_fraction}
            # This already-required rejection is independent of picket topology.
            # Run it first: thousands of rejected middle-rail pairs must not each
            # solve intersections against thousands of upright fragments.
            if below_fraction > .35:
                detail['pairs'].append({**pair, 'accepted': False, 'independentlySupported': False,
                                        'topologySkipped': 'semantic fence continues below this candidate'})
                continue
            independently_supported, topology = _picket_support(vertical, axis, up, left, right,
                                                                 lower['world'], upper['world'], width)
            # An occluded view can observe the same underside without exposing
            # three pickets. It supplies geometry, not independent part identity:
            # the caller must require a strong view and finite multiview overlap.
            accepted = not topology['continuingPickets']
            topology['independentlySupported'] = bool(independently_supported)
            detail['pairs'].append({**pair, 'accepted': accepted, **topology})
            if not accepted:
                continue
            candidates.append({'photo': photo, 'rawEnds': lower['raw'].tolist(), 'uv': _pixels(lower['raw'], A).tolist(),
                'rawSegments': lower['rawSegments'].tolist(),
                'uvSegments': [_pixels(segment, A).tolist() for segment in lower['rawSegments']],
                'score': float(fraction * (1 - below_fraction) * min(1., len(topology['terminatingPickets']) / 10)),
                'sourceBeam': f'{item["id"]}-photo{photo}-rgb-lower{lower["index"]}-upper{upper["index"]}',
                'sourceSegmentIndex': lower['index'], 'planeIndex': plane_index,
                'rawUpperEnds': upper['raw'].tolist(), 'upperUv': _pixels(upper['raw'], A).tolist(),
                'rawUpperSegments': upper['rawSegments'].tolist(),
                'pairedWidthNative': width, 'associationPointNative': lower['world'].mean(0).tolist(),
                'identity': ('paired lower-rail underside with independently observed repeated upright terminations'
                             if independently_supported else 'partial paired underside; lower-rail identity requires transfer from a strongly supported source view'),
                'identityEvidence': topology, 'scope': 'RGB lower-rail candidate; saved plane supports association only; strong-view identity and finite multiview fit required'})
    # One RGB bottom fragment group is one observation even when several upper
    # border hypotheses paired with it. Keep the best-supported identity only.
    unique = {}
    for candidate in candidates:
        key = (candidate['photo'], tuple(sorted(tuple(np.asarray(segment).ravel())
                                                for segment in candidate['rawSegments'])))
        old = unique.get(key)
        rank = (candidate['identityEvidence']['independentlySupported'], candidate['score'])
        if old is None or rank > (old['identityEvidence']['independentlySupported'], old['score']):
            unique[key] = candidate
    return list(unique.values()), diagnostics
