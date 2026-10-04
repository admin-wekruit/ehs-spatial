"""Overhead entrance gantry: one multi-view member model per scene, in the run's native world.

The white portal frame at a workcell entrance (posts, top beam, braces) carries the signal lamps, the
overhead cell sign and the compressed-air pipe; without it their proxies float. For the photos of ONE
scene this fuses the cleaned SAM support of the gantry prompts (workcell_extra_models._clean: mask and
valid depth, 1 px erosion, camera depth within 3 MAD of its median) and fits structural members:

* line hypotheses of three families -- posts parallel to the floor normal, beams parallel to the floor,
  braces 20-70 degrees from vertical -- in sequential RANSAC (fixed seed); a pixel belongs to a line its
  camera ray passes close to (thin members are placed well across the ray, poorly along it), and a member
  is the largest gap-free run of inliers, refit within its family; with two or more photos whose
  back-projected member planes differ, the member is triangulated from them, provided the result stays
  within 5% of the pixels' measured depth (else the photos' poses and pointmaps disagree there);
* a member needs support from two photos, or strong support from one (kept and flagged singleView);
* joints: a beam end over a post axis is carried by that post (the post rises to the beam, the beam
  reaches the post); a brace joins two different posts/beams (repeats per photo merge); only frames
  with a carried beam are kept;
* posts stand on the fitted floor; members are light-gray boxes with a square section fitted to every
  photo's silhouette width;
* every scene photo is a validation view: model IoU against the gantry mask (all model pixels, and
  only those not hidden behind nearer observed depth) and model depth against that photo's pointmap.

build_gantry(root, photos, masks_by_photo, geometry) -> (trimesh.Scene | None, record)
masks_by_photo: {photo: {word: [SAM RLE, ...]}} in sam3.json order (see masks_from_segmentation), so each
member's observations carry the catalog source 'SAM: <word>; instance <k>' and outer-contour polygons.
A display/placement model in native units from pointmap support; no physical-accuracy claim.

python scripts/workcell_gantry.py --root RUN --segmentation SAM.json --photos 3 4 --name cell-090 --out DIR
"""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

GANTRY_WORDS = ('white steel beam', 'blue pipe')  # research-notes/workcell-gantry-2026-10-04 prompt probe: members; entrance top
MIN_HEIGHT = .25      # native; floor rails and base plates below this are not gantry evidence (posts are extended to the floor)
INLIER = .1           # native; line inlier radius: post half-width (~.06) plus pointmap depth noise
GAP = .4              # native; largest gap along one member (brackets, lamps and clamps occlude members)
VOXEL = .03           # native; RANSAC runs on one point per voxel, membership on every point
ITERATIONS = 300      # hypotheses per extracted member
MAX_LINES = 40
SUB_POINTS = 25       # voxel points for a RANSAC line to be extracted at all
ANGLE = 10.           # degrees; a point pair this close to the floor plane proposes a beam
BRACE = (20., 70.)    # degrees from vertical: posts below, braces within, beams above (beams are modelled level)
MIN_POINTS = 120      # cleaned support points of a member
MIN_LENGTH = .5       # native; shortest beam or brace
MIN_POST = 1.         # native; shortest visible post
VIEW_POINTS = 40      # cleaned points from one photo for that photo to support a member
STRONG_SINGLE = 250   # one-photo support kept (flagged singleView) only at or above this
JOINT = .8            # native; reach of a joint (beam end to post axis, brace end to member axis); gaps are recorded
                      # ponytail: thin members' triangulated depth differs by up to ~.8 native in the 2026-10-03 capture
MAX_RISE = .3         # a post rises to the beam it carries by at most this fraction of its height
LINK = (20, .1)       # an instance observes a member with >= 20 shared support pixels and >= 10% of its support
DEPTH_TOL = .05       # relative; a model pixel behind nearer observed depth is occluded in that photo
DEPTH_SLACK = .1      # relative; a ray's closest approach to its member may differ this much from the pixel's depth
TRI_ANGLE = 10.       # degrees; photos' back-projected member planes must differ this much to triangulate the member
TRI_RAYS = 15         # inlier rays a photo needs for its back-projected member plane
TRI_DEPTH = .05       # relative; a triangulated member must stay this close (median) to its pixels' measured depth
GRAY = [200, 200, 200, 255]


def masks_from_segmentation(segmentation, photos, words=GANTRY_WORDS):
    """{photo: {word: [RLE, ...]}} for the gantry words of a sam3.json-format result (results[photo - 1])."""
    prompts = [p['text'] for p in segmentation['prompts']]
    return {p: {w: list(segmentation['results'][p - 1][prompts.index(w)].get('rle', [])) for w in words if w in prompts}
            for p in photos}


def _unit(v):
    return np.asarray(v, float) / np.linalg.norm(v)


def _frames(root, photos):
    from scripts.workcell_photo_oneshot import _array, _frame
    frames = {}
    for p in photos:
        raw = _frame(Path(root), p)
        points = _array(raw['pts3d']).astype(float)
        frames[p] = {'raw': raw, 'points': points, 'rgb': _array(raw['image']),
                     'valid': _array(raw['non_ambiguous_mask']).astype(bool) & np.isfinite(points).all(2),
                     'pose': _array(raw['camera_poses']).astype(float), 'K': _array(raw['intrinsics']).astype(float)}
    return frames


def _support(photos, masks_by_photo, frames, geometry):
    """Cleaned support of every gantry instance, and their union per photo (each pixel once) with its camera rays."""
    from scripts.workcell_extra_models import _clean
    words = list(dict.fromkeys(w for p in photos for w in masks_by_photo.get(p, {})))
    segmentation = {'prompts': [{'text': w} for w in words],
                    'results': [[{'rle': masks_by_photo.get(p, {}).get(w, [])} for w in words] for p in range(1, max(photos) + 1)]}
    instances, rows = [], []
    for p in photos:
        width = frames[p]['valid'].shape[1]
        taken = np.zeros(frames[p]['valid'].size, bool)
        for w in words:
            for k in range(len(masks_by_photo.get(p, {}).get(w, []))):
                source = f'SAM: {w}; instance {k}'
                q, _, mask, xy = _clean({'photo': p, 'source': source}, frames[p], geometry, segmentation, None)
                flat = (xy[:, 1] * width + xy[:, 0]).astype(np.int64)
                instances.append({'photo': p, 'word': w, 'instance': k, 'source': source, 'mask': mask, 'pixels': flat})
                fresh = ~taken[flat]
                taken[flat] = True
                rows.append((q[fresh], np.full(fresh.sum(), p), flat[fresh], np.tile(frames[p]['pose'][:3, 3], (fresh.sum(), 1))))
    P, photo, pixel, C = (np.concatenate(x) for x in zip(*rows)) if rows else (np.empty((0, 3)), np.empty(0, int),
                                                                               np.empty(0, np.int64), np.empty((0, 3)))
    s = np.linalg.norm(P - C, axis=1)
    return instances, {'P': P, 'C': C, 'R': (P - C) / np.maximum(s, 1e-12)[:, None], 's': s, 'photo': photo, 'pixel': pixel}


def _take(F, index):
    return {k: v[index] for k, v in F.items()}


def _approach(F, origin, direction):
    """Closest approach of each point's camera ray to a line: lateral distance (inf beyond the depth window),
    line parameter, and the point on the ray.

    A thin member's pixel is well placed across its viewing ray and poorly along it (pointmap depth bleeds
    toward the background and differs between photos), so a point belongs to a line its ray passes within
    INLIER of, no further than DEPTH_SLACK of its measured depth from the pixel's point.
    """
    C, R, s = F['C'], F['R'], F['s']
    w = C - origin
    b, d, e = R @ direction, np.einsum('ij,ij->i', R, w), w @ direction
    denom = 1 - b * b
    along = np.where(denom > 1e-9, (b * e - d) / np.where(denom > 1e-9, denom, 1), s)  # parallel ray: measured depth
    X = C + R * along[:, None]
    t = (X - origin) @ direction
    lateral = np.linalg.norm(X - origin - t[:, None] * direction, axis=1)
    lateral[np.abs(along - s) > DEPTH_SLACK * s] = np.inf
    return lateral, t, X


def _run(F, origin, direction, candidates):
    """Indices of the largest gap-free run of a line's inliers among candidate points."""
    lateral, t, _ = _approach(_take(F, candidates), origin, direction)
    near = lateral < INLIER
    index, t = candidates[near], t[near]
    if len(index) < 2:
        return index
    order = np.argsort(t)
    pieces = np.split(order, np.flatnonzero(np.diff(t[order]) > GAP) + 1)
    return index[max(pieces, key=len)]


def _angle(direction, up):
    return float(np.degrees(np.arccos(min(1., abs(float(direction @ up))))))


def _line(kind, points, up):
    """Least-squares line through a member's inliers and its family: (kind, origin, unit direction).

    Posts are vertical. A beam keeps its support's tilt for membership (depth error tilts long receding
    members in a pointmap; a level line would keep only part of them) and is modelled level in _members.
    """
    if kind == 'post':
        return kind, np.median(points, axis=0), up
    centre = points.mean(0)
    direction = np.linalg.svd(points - centre, full_matrices=False)[2][0]
    angle = _angle(direction, up)
    if angle < BRACE[0]:
        return 'post', np.median(points, axis=0), up
    return ('beam' if angle > BRACE[1] else 'brace'), centre, direction


def _triangulate(F, run, kind, origin, direction, up):
    """Member line as the intersection of its back-projected planes (camera centre and silhouette rays), one per photo.

    A plane through the middle of a silhouette contains the member's axis, so this is the axis, independent of
    pointmap depth. None with fewer than two photos of support or planes within TRI_ANGLE of each other.
    """
    normals, offsets = [], []
    for p in np.unique(F['photo'][run]):
        mine = run[F['photo'][run] == p]
        if len(mine) >= TRI_RAYS:
            n = np.linalg.svd(F['R'][mine], full_matrices=False)[2][-1]
            normals.append(n)
            offsets.append(float(n @ F['C'][mine[0]]))
    if len(normals) < 2:
        return None
    N = np.array(normals)
    _, S, Vt = np.linalg.svd(N)  # full: Vt[-1] spans the planes' common direction even for two photos
    if S[1] < S[0] * np.tan(np.radians(TRI_ANGLE) / 2):  # two unit normals at angle a: S1 / S0 = tan(a / 2)
        return None
    d = up if kind == 'post' else Vt[-1]
    if kind == 'beam':
        d = _unit(d - up * (d @ up))
    d = d if d @ direction >= 0 else -d
    x0 = np.linalg.lstsq(np.vstack([N, d]), np.r_[offsets, d @ origin], rcond=None)[0]
    support = _take(F, run)
    along = np.linalg.norm(_approach(support, x0, d)[2] - support['C'], axis=1)
    if np.median(np.abs(along - support['s']) / support['s']) > TRI_DEPTH:
        return None  # the photos' poses and pointmaps disagree here: keep the measured depth
    return x0, d


def _refit(F, kind, origin, direction, candidates, up, rounds=2):
    """Alternate membership and a fit: the family least-squares line through the inliers moved along their rays
    onto the line, replaced by the triangulated line when two or more photos allow it."""
    run, triangulated = _run(F, origin, direction, candidates), False
    for _ in range(rounds):
        if len(run) < 2:
            break
        kind, origin, direction = _line(kind, _approach(_take(F, run), origin, direction)[2], up)
        line = _triangulate(F, run, kind, origin, direction, up)
        triangulated = line is not None
        if triangulated:
            origin, direction = line
        run = _run(F, origin, direction, candidates)
    return kind, origin, direction, run, triangulated


def _fit_lines(F, up, seed=0):
    """Sequential RANSAC over post/beam/brace line hypotheses on a voxel subsample."""
    from scipy.spatial import cKDTree
    _, first = np.unique(np.floor(F['P'] / VOXEL).astype(np.int64), axis=0, return_index=True)
    S = _take(F, np.sort(first))
    tree, alive, rng, lines = cKDTree(S['P']), np.ones(len(S['P']), bool), np.random.default_rng(seed), []
    while alive.sum() >= SUB_POINTS and len(lines) < MAX_LINES:
        index = np.flatnonzero(alive)
        best = None
        for _ in range(ITERATIONS):
            i = index[rng.integers(len(index))]
            hypotheses = [('post', S['P'][i], up)]
            near = np.asarray(tree.query_ball_point(S['P'][i], 1.5), int)
            near = near[alive[near] & (S['photo'][near] == S['photo'][i])]  # one photo: no cross-photo depth offset
            if len(near):
                d = S['P'][near[rng.integers(len(near))]] - S['P'][i]
                if np.linalg.norm(d) > .3:
                    d = _unit(d)
                    angle = _angle(d, up)
                    if angle >= 90 - ANGLE:
                        hypotheses.append(('beam', S['P'][i], d))
                    elif BRACE[0] <= angle <= BRACE[1]:
                        hypotheses.append(('brace', S['P'][i], d))
            for kind, origin, direction in hypotheses:
                run = _run(S, origin, direction, index)
                if best is None or len(run) > len(best[3]):
                    best = (kind, origin, direction, run)
        kind, origin, direction, run, _ = _refit(S, *best[:3], index, up)
        if len(run) < SUB_POINTS:
            break
        t = _approach(_take(S, run), origin, direction)[1]
        lines.append({'type': kind, 'origin': origin, 'direction': direction, 'extent': (float(t.min()), float(t.max()))})
        lateral, along, _ = _approach(_take(S, index), origin, direction)
        alive[index[(lateral < 2 * INLIER) & (along >= t.min() - INLIER) & (along <= t.max() + INLIER)]] = False
        alive[run] = False
    return lines


def _section(points, photo, direction, frames):
    """Square cross-section (width, across axis) fitting every photo's silhouette width of the member.

    Per photo the silhouette width is the 2-98% range of pixel-centre offsets across the member (perpendicular to
    its axis and the view ray, about that photo's own axis) plus 2.5 pixels at its depth: _clean erodes one pixel
    on each side, and pixel centres span one pixel less than a straight silhouette and about as much as a sloped
    one (+-0.5 px). A square of side w turned by a shows w (|cos(f - a)| + |sin(f - a)|) to a ray at angle f in the
    section plane; among turns whose squared silhouette misfit is within a pixel of the best, the widest side is
    kept (one photo: the box faces it). Members are a few pixels wide on the canonical grid, so widths carry ~1 px
    uncertainty.
    """
    b1 = _unit(np.cross(direction, [1., 0, 0] if abs(direction[0]) < .9 else [0, 1., 0]))
    b2 = np.cross(direction, b1)
    widths, angles, pixels = [], [], []
    for p in np.unique(photo):
        q = points[photo == p]
        if len(q) < VIEW_POINTS:
            continue
        camera = frames[p]['pose'][:3, 3]
        centre = np.median(q, axis=0)
        own = np.linalg.svd(q - q.mean(0), full_matrices=False)[2][0]  # this photo's axis: no cross-photo tilt leaks in
        lateral = np.cross(own if abs(own @ direction) > .9 else direction, centre - camera)
        if np.linalg.norm(lateral) < .2 * np.linalg.norm(centre - camera):
            continue  # viewed along the member: no across-axis silhouette
        lo, hi = np.percentile((q - centre) @ _unit(lateral), [2, 98])
        pixel = float(np.median((q - camera) @ frames[p]['pose'][:3, 2])) / frames[p]['K'][0, 0]
        widths.append(float(hi - lo + 2.5 * pixel))
        angles.append(np.arctan2((centre - camera) @ b2, (centre - camera) @ b1))
        pixels.append(pixel)
    if not widths:
        return INLIER, None
    s, f = np.array(widths), np.array(angles)
    if len(s) == 1:
        a, w = f[0], s[0]
    else:
        turns = np.radians(np.arange(90.))[:, None]
        shown = np.abs(np.cos(f - turns)) + np.abs(np.sin(f - turns))
        side = np.median(s / shown, axis=1)
        misfit = ((s - side[:, None] * shown) ** 2).sum(1)
        # two photos rarely fix both side and turn: among turns within a pixel of the best fit, the most face-on (widest)
        best = int(np.argmax(np.where(misfit <= misfit.min() + np.sum(np.square(pixels)), side, -np.inf)))
        a, w = turns[best, 0], side[best]
    return float(w), b1 * np.cos(a) + b2 * np.sin(a)


def _members(lines, F, frames, up, offset):
    """Every fused point to its nearest line; each line's full-support member with extent, width and photo counts."""
    distance = np.full((len(lines), len(F['P'])), np.inf)
    for n, line in enumerate(lines):
        lateral, t, _ = _approach(F, line['origin'], line['direction'])
        lateral[(t < line['extent'][0] - GAP / 2) | (t > line['extent'][1] + GAP / 2)] = np.inf
        distance[n] = lateral
    owner = distance.argmin(0) if len(lines) else np.full(len(F['P']), -1)
    if len(lines):
        owner[distance.min(0) >= INLIER] = -1
    members = []
    for n, line in enumerate(lines):
        mine = np.flatnonzero(owner == n)
        if len(mine) < 2:
            continue
        kind, origin, direction, mine, triangulated = _refit(F, line['type'], line['origin'], line['direction'], mine, up)
        if len(mine) < 2:
            continue
        _, t, q = _approach(_take(F, mine), origin, direction)
        if kind == 'post':
            heights = q @ up + offset
            top = float(np.percentile(heights, 99))
            foot = origin - up * float(origin @ up + offset)  # posts start at the fitted floor
            ends = [foot, foot + up * top]
            visible = float(np.ptp(np.percentile(heights, [1, 99])))
        else:
            lo, hi = np.percentile(t, [1, 99])
            ends = [origin + direction * lo, origin + direction * hi]
            visible = float(hi - lo)
            if kind == 'beam':  # level at the support's median height; the support's tilt is recorded
                tilt = 90 - _angle(direction, up)
                level = float(np.median(q @ up))
                ends = [e - up * (e @ up - level) for e in ends]
                direction = _unit(ends[1] - ends[0])
        photo = F['photo'][mine]
        counts = {int(p): int((photo == p).sum()) for p in np.unique(photo)}
        width, side = _section(F['P'][mine], photo, direction, frames)
        push = sum(n * _unit(v - direction * (v @ direction)) for v, n in
                   ((q[photo == p].mean(0) - frames[p]['pose'][:3, 3], counts[p]) for p in counts))
        if not triangulated and np.linalg.norm(push) > 0:  # measured depth lies on the faces the cameras see
            ends = [np.asarray(e, float) + _unit(push) * width / 2 for e in ends]  # the axis is half a width behind them
        members.append({'type': kind, 'direction': direction, 'ends': [np.asarray(e, float) for e in ends], 'visibleLength': visible,
                        'width': width, 'side': side, 'index': mine, 'supportByPhoto': counts, 'triangulated': triangulated,
                        **({'tiltDeg': tilt} if kind == 'beam' else {})})
    return members


def _segment_distance(point, a, b):
    ab = b - a
    s = np.clip((point - a) @ ab / max(ab @ ab, 1e-12), 0, 1)
    return float(np.linalg.norm(point - (a + s * ab))), a + s * ab


def _joints(members, up, offset):
    """Carry beams on posts, snap brace ends; returns joints [(i, j, kind)] and records rises/extensions in place."""
    joints = []
    posts = [i for i, m in enumerate(members) if m['type'] == 'post']
    for b, beam in enumerate(members):
        if beam['type'] != 'beam':
            continue
        a, z = beam['ends']
        d = beam['direction']
        height = float(a @ up + offset)
        for i in posts:
            post = members[i]
            foot, head = post['ends']
            top = float(head @ up + offset)
            rel = foot - a
            rel -= up * (rel @ up)
            t, across = float(rel @ d), float(np.linalg.norm(rel - d * (rel @ d)))
            length = float((z - a) @ d)
            if across > JOINT or not (-JOINT <= t <= length + JOINT) or not (.5 * top <= height <= top * (1 + MAX_RISE)):
                continue
            half = post['width'] / 2
            if t - half < 0:  # the beam reaches the far face of its carrying post
                beam['ends'][0] = a = a + d * (t - half)
                length = float((z - a) @ d)
                beam.setdefault('extendedNative', 0.); beam['extendedNative'] += float(half - t)
            elif t + half > length:
                beam['ends'][1] = z = a + d * (t + half)
                beam.setdefault('extendedNative', 0.); beam['extendedNative'] += float(t + half - length)
            rise = height + beam['width'] / 2 - top
            if rise > 0:  # the post rises to the beam it carries (lamp brackets and clamps hide this part)
                post['ends'][1] = foot + up * (top + rise)
                post['riseNative'] = max(post.get('riseNative', 0.), float(rise))
            joints.append((i, b, 'carries', across))
    for c, brace in enumerate(members):  # a brace joins two different posts/beams, nearest pair within JOINT
        if brace['type'] != 'brace':
            continue
        options = [[(gap, j, point) for j, other in enumerate(members) if other['type'] != 'brace'
                    for gap, point in [_segment_distance(end, *other['ends'])] if gap <= JOINT] for end in brace['ends']]
        pairs = [(a[0] + b[0], a, b) for a in options[0] for b in options[1] if a[1] != b[1]]
        if not pairs:
            continue
        _, *ends = min(pairs, key=lambda pair: pair[0])
        for e, (gap, j, point) in enumerate(ends):
            brace['ends'][e] = point
            joints.append((c, j, 'brace end', gap))
        brace['direction'] = _unit(brace['ends'][1] - brace['ends'][0])
        brace['joined'] = frozenset(j for _, j, _ in ends)
    pairs = {}  # one brace per joined pair: photos that disagree can each yield a copy; the best supported keeps all support
    for c, brace in enumerate(members):
        if brace.get('joined'):
            pairs.setdefault(brace['joined'], []).append(c)
    for group in pairs.values():
        keep = members[max(group, key=lambda c: len(members[c]['index']))]
        for c in group:
            if members[c] is not keep:
                members[c]['duplicateOf'] = keep
                for p, n in members[c]['supportByPhoto'].items():
                    keep['supportByPhoto'][p] = keep['supportByPhoto'].get(p, 0) + n
                keep['index'] = np.union1d(keep['index'], members[c]['index'])
        keep['singleView'] = sum(n >= VIEW_POINTS for n in keep['supportByPhoto'].values()) < 2
    return [j for j in joints if 'duplicateOf' not in members[j[0]]]


def _frame_axes(member, up, e1):
    """Box axes: the member direction, then the across direction its width was measured along."""
    d = member['direction']
    side = member['side'] if member['side'] is not None else (up if member['type'] == 'beam' else np.cross(d, up) if
                                                               member['type'] == 'brace' else e1)
    side = _unit(side - d * (side @ d))
    return np.column_stack((d, side, np.cross(d, side)))


def _box(member, axes):
    a, b = member['ends']
    length = float(np.linalg.norm(b - a))
    centre = (a + b) / 2
    half = np.array([length, member['width'], member['width']]) / 2
    mesh = trimesh.creation.box(extents=2 * half)
    mesh.vertices = mesh.vertices @ axes.T + centre
    mesh.visual.vertex_colors = np.tile(np.asarray(GRAY, np.uint8), (len(mesh.vertices), 1))
    return mesh, (centre, axes, half)


def render(boxes, frame):
    """Per-pixel nearest box index (-1 none) and its camera depth, by exact ray/oriented-box slabs on the frame grid."""
    h, w = frame['valid'].shape
    yy, xx = np.mgrid[0:h, 0:w]
    rays = np.stack([xx, yy, np.ones_like(xx)], -1).reshape(-1, 3).astype(float) @ np.linalg.inv(frame['K']).T
    directions = rays @ frame['pose'][:3, :3].T  # camera z component 1: the ray parameter is camera depth
    camera = frame['pose'][:3, 3]
    depth, owner = np.full(h * w, np.inf), np.full(h * w, -1)
    for n, (centre, axes, half) in enumerate(boxes):
        o, d = axes.T @ (camera - centre), directions @ axes
        with np.errstate(divide='ignore', invalid='ignore'):
            t1, t2 = (-half - o) / d, (half - o) / d
        near = np.nanmax(np.minimum(t1, t2), axis=1)
        far = np.nanmin(np.maximum(t1, t2), axis=1)
        t = np.where(near > 0, near, far)
        hit = (far >= near) & (far > 0) & (t < depth)
        depth[hit], owner[hit] = t[hit], n
    return owner.reshape(h, w), depth.reshape(h, w)


def record_boxes(record):
    """(centre, axes, half extents) of each modelled member of a gantry record."""
    return [(np.asarray(m['box']['centreNative']), np.asarray(m['box']['axes']).T, np.asarray(m['box']['extentsNative']) / 2)
            for m in record['members']]


def reproject(boxes, frame):
    """Rendered member index and depth, observed camera depth, and model pixels not behind nearer observed depth."""
    owner, depth = render(boxes, frame)
    observed = (frame['points'] - frame['pose'][:3, 3]) @ frame['pose'][:3, 2]
    visible = (owner >= 0) & (~frame['valid'] | (observed >= depth * (1 - DEPTH_TOL)))
    return owner, depth, observed, visible


def _validate(boxes, names, frame, gantry, everything):
    """IoU against the gantry mask and depth agreement with the pointmap, for the whole model and per member."""
    owner, depth, observed, visible = reproject(boxes, frame)
    model = owner >= 0
    def iou(a, b=gantry):
        return round(float((a & b).sum() / max(1, (a | b).sum())), 4)
    def agreement(region):
        region = region & gantry & frame['valid']
        if not region.any():
            return None
        rel = np.abs(observed[region] - depth[region]) / depth[region]
        return {'pixels': int(region.sum()), 'medianRelative': round(float(np.median(rel)), 4),
                'within5Percent': round(float((rel <= .05).mean()), 4), 'within10Percent': round(float((rel <= .10).mean()), 4)}
    per = {}
    for n, name in enumerate(names):
        mine = visible & (owner == n)
        per[name] = {'visiblePixels': int(mine.sum()), 'occludedPixels': int(((owner == n) & ~visible).sum()),
                     'insideMask': round(float((mine & gantry).sum() / max(1, mine.sum())), 4), 'depth': agreement(mine)}
    return {'iou': iou(model), 'iouVisible': iou(visible), 'iouAllPromptInstances': iou(model, everything),
            'maskPixels': int(gantry.sum()), 'modelPixels': int(model.sum()), 'visibleModelPixels': int(visible.sum()),
            'depth': agreement(visible), 'members': per}


def _refuse(reason, photos, words, extra=None):
    return None, {'status': 'refused', 'reason': reason, 'photos': list(photos), 'words': list(words), **(extra or {})}


def build_gantry(root, photos, masks_by_photo, geometry, *, seed=0):
    """One gantry model for the photos of ONE scene; (None, refusal record) when evidence is missing."""
    from scripts.workcell_extra_models import _floor_basis
    from scripts.workcell_photo_objects import _observation
    photos = sorted({int(p) for p in photos})
    words = list(dict.fromkeys(w for p in photos for w in masks_by_photo.get(p, {})))
    if len(photos) < 2:
        return _refuse('A multi-view gantry needs at least two photos of one scene', photos, words)
    if not any(masks_by_photo.get(p, {}).get(w) for p in photos for w in words):
        return _refuse('No gantry segmentation in any photo', photos, words)
    up, offset, e1, _ = _floor_basis(geometry)
    frames = _frames(root, photos)
    instances, F = _support(photos, masks_by_photo, frames, geometry)
    F = _take(F, np.flatnonzero(F['P'] @ up + offset >= MIN_HEIGHT))
    if len(F['P']) < MIN_POINTS:
        return _refuse('Too little cleaned gantry support above the floor', photos, words, {'supportPoints': int(len(F['P']))})
    lines = _fit_lines(F, up, seed)
    rejected, kept = [], []
    for m in _members(lines, F, frames, up, offset):
        views = [p for p, c in m['supportByPhoto'].items() if c >= VIEW_POINTS]
        m['singleView'] = len(views) < 2
        reason = None
        if len(m['index']) < MIN_POINTS:
            reason = f'support {len(m["index"])} < {MIN_POINTS} points'
        elif m['visibleLength'] < (MIN_POST if m['type'] == 'post' else MIN_LENGTH):
            reason = f'visible length {m["visibleLength"]:.2f} native too short'
        elif m['singleView'] and max(m['supportByPhoto'].values()) < STRONG_SINGLE:
            reason = f'supported by {len(views)} photo(s); one-photo support below {STRONG_SINGLE} points'
        (rejected if reason else kept).append(m)
        if reason:
            m['reason'] = reason
    joints = _joints(kept, up, offset)
    parent = list(range(len(kept)))
    def find(i):
        while parent[i] != i:
            i = parent[i]
        return i
    for i, j, *_ in joints:
        parent[find(i)] = find(j)
    frames_of = {}
    for i, m in enumerate(kept):
        frames_of.setdefault(find(i), []).append(i)
    survivors = set()
    for group in frames_of.values():
        carried = {b for i, b, kind, _ in joints if kind == 'carries' and i in group}
        if carried:
            survivors.update(i for i in group if kept[i]['type'] != 'brace' or
                             sum(1 for c, _, kind, _ in joints if c == i and kind == 'brace end') >= 2)
    for i, m in enumerate(kept):
        if i not in survivors:
            m['reason'] = ('not part of a frame with a beam carried by a post' if m['type'] != 'brace' else
                           'repeat of a brace joining the same members (support merged into it)' if 'duplicateOf' in m else
                           'brace not joined to two different members')
            rejected.append(m)
    order = sorted(survivors, key=lambda i: ({'post': 0, 'beam': 1, 'brace': 2}[kept[i]['type']], -len(kept[i]['index'])))
    if not order:
        return _refuse('No beam carried by a post: the gantry frame is not supported', photos, words,
                       {'rejected': [_summary(m, up, offset) for m in rejected]})
    final = [kept[i] for i in order]
    names, counters = [], {}
    for m in final:
        counters[m['type']] = counters.get(m['type'], 0) + 1
        names.append(f"gantry-{m['type']}-{counters[m['type']]}")
    final_joints = [(order.index(i), order.index(j), kind, gap) for i, j, kind, gap in joints if i in order and j in order]
    groups = [[names[order.index(i)] for i in group if i in survivors] for group in frames_of.values()]
    for m in rejected:  # diagnostic: a rejected copy of a modelled member shows cross-view pointmap disagreement
        near = [(_segment_distance((m['ends'][0] + m['ends'][1]) / 2, *f['ends'])[0], name)
                for name, f in zip(names, final) if f['type'] == m['type']]
        if near and min(near)[0] <= .6:  # ponytail: proximity only; a 2D silhouette match would confirm identity
            m['axisDistanceNative'], m['nearestModelledMember'] = min(near)
    scene, boxes = trimesh.Scene(), []
    for name, m in zip(names, final):
        mesh, box = _box(m, _frame_axes(m, up, e1))
        scene.add_geometry(mesh, node_name=name, geom_name=name)
        boxes.append(box)
    observations, records = [], []
    for name, m, (centre, axes, half) in zip(names, final, boxes):
        links = []
        for p in sorted(m['supportByPhoto']):
            mine = F['pixel'][m['index']][F['photo'][m['index']] == p]
            for inst in instances:
                if inst['photo'] != p or not len(inst['pixels']):
                    continue
                shared = int(np.isin(inst['pixels'], mine).sum())
                if shared >= LINK[0] and shared >= LINK[1] * len(inst['pixels']):
                    key = (p, inst['source'])
                    if key not in [(o['photo'], o['source']) for o in observations]:
                        o = _observation(inst['mask'], p, inst['source'])
                        o.update(maskPixels=int(inst['mask'].sum()), supportedPixels=int(len(inst['pixels'])))
                        observations.append(o)
                    links.append({'photo': p, 'source': inst['source'], 'sharedSupportPixels': shared})
        record = _summary(m, up, offset)
        record.update(id=name, node=name, observations=links,
                      box={'centreNative': centre.tolist(), 'axes': axes.T.tolist(), 'extentsNative': (2 * half).tolist()})
        records.append(record)
    # Canonical order (photo, prompt, instance): member order follows support counts, which move with float rounding
    # across platforms (container vs laptop); the observation set does not.
    observations.sort(key=lambda o: (o['photo'], *o['source'].removeprefix('SAM: ').rsplit('; instance ', 1)[:1],
                                     int(o['source'].rsplit('; instance ', 1)[1])))
    validation = {}
    for p in photos:
        gantry = np.zeros(frames[p]['valid'].shape, bool)
        for o in observations:
            if o['photo'] == p:
                gantry |= next(i['mask'] for i in instances if i['photo'] == p and i['source'] == o['source'])
        everything = np.zeros_like(gantry)
        for inst in instances:
            if inst['photo'] == p:
                everything |= inst['mask']
        validation[p] = _validate(boxes, names, frames[p], gantry, everything)
    for record in records:
        record['reprojection'] = {str(p): validation[p]['members'][record['id']] for p in photos}
    return scene, {
        'status': 'modelled', 'photos': photos, 'words': words, 'coordinateSystem': 'run native world (floor from geometry.json)',
        'floor': {'up': up.tolist(), 'offset': float(offset)},
        'members': records, 'joints': [{'member': names[i], 'to': names[j], 'kind': kind, 'gapNative': round(float(gap), 4)}
                                       for i, j, kind, gap in final_joints],
        'frames': [g for g in groups if g],
        'observations': observations,
        'validation': {str(p): {k: v for k, v in validation[p].items() if k != 'members'} for p in photos},
        'rejected': [_summary(m, up, offset) for m in rejected],
        'supportPoints': int(len(F['P'])), 'linesTried': len(lines),
        'rule': {'support': 'workcell_extra_models._clean per SAM instance (eroded mask, valid depth, 3 MAD); each pixel once per photo; '
                            f'height above the fitted floor >= {MIN_HEIGHT} native',
                 'fit': f'sequential RANSAC (seed {seed}, {ITERATIONS} hypotheses per member, {VOXEL} native voxels) over post / beam / '
                        f'brace lines; a point is an inlier when its camera ray passes within {INLIER} native of the line no further '
                        f'than {DEPTH_SLACK:.0%} of its depth from the pixel point; gap {GAP}; family least-squares refit to the '
                        'inliers moved along their rays onto the line; triangulated from the photos\' back-projected member planes '
                        f'when they differ by >= {TRI_ANGLE:.0f} degrees and the result is within {TRI_DEPTH:.0%} (median) of the '
                        'measured depth, else the axis is set half a width behind the measured faces',
                 'keep': f'>= {MIN_POINTS} points and >= {VIEW_POINTS} from each of two photos, or >= {STRONG_SINGLE} from one (singleView); '
                         f'posts >= {MIN_POST}, others >= {MIN_LENGTH} native visible; only frames with a beam carried by a post',
                 'joints': f'beam end within {JOINT} native of a post axis (gap recorded); post rises to the beam by <= {MAX_RISE} '
                           f'of its height; brace ends snap to the nearest pair of different posts/beams within {JOINT}; braces '
                           'joining the same pair merge into the best supported',
                 'box': 'square section: side and turn about the axis least-squares fit to every photo\'s silhouette width '
                        '(2-98% support spread across the axis and the view ray plus 2.5 px for the 1 px erosion each side and '
                        'pixel-centre sampling; ~1 px uncertainty); one photo: the box faces it'},
        'basis': 'display and placement model from SAM support on MapAnything pointmaps; native units; not surveyed; no physical accuracy claim',
    }


def _summary(m, up, offset):
    a, b = m['ends']
    out = {'type': m['type'], 'endsNative': [a.tolist(), b.tolist()], 'lengthNative': round(float(np.linalg.norm(b - a)), 4),
           'visibleLengthNative': round(m['visibleLength'], 4), 'widthNative': round(m['width'], 4),
           'heightsNative': [round(float(a @ up + offset), 4), round(float(b @ up + offset), 4)],
           'angleFromVerticalDeg': round(_angle(m['direction'], up), 2),
           'supportPoints': int(len(m['index'])), 'supportByPhoto': {str(k): v for k, v in m['supportByPhoto'].items()},
           'sourcePhotos': sorted(int(p) for p, c in m['supportByPhoto'].items() if c >= VIEW_POINTS),
           'singleView': bool(m.get('singleView')),
           'depthFrom': 'triangulated from the photos\' silhouette planes' if m.get('triangulated') else
                        'measured pointmap depth, axis half a width behind the seen faces'}
    for key in ('tiltDeg', 'riseNative', 'extendedNative', 'reason', 'nearestModelledMember', 'axisDistanceNative'):
        if key in m:
            out[key] = round(m[key], 4) if isinstance(m[key], float) else m[key]
    return out


def box_distance(points, record):
    """Distance (native) from points to the nearest modelled member surface (0 inside); (distances, member ids)."""
    points = np.atleast_2d(points)
    best, owner = np.full(len(points), np.inf), np.full(len(points), '', object)
    for member in record['members']:
        box = member['box']
        local = (points - box['centreNative']) @ np.asarray(box['axes']).T
        d = np.linalg.norm(np.maximum(np.abs(local) - np.asarray(box['extentsNative']) / 2, 0), axis=1)
        closer = d < best
        best[closer], owner[closer] = d[closer], member['id']
    return best, owner


def gantry_mask(record, masks_by_photo, frame, photo):
    """Union of the SAM instances the record links to its members in one photo (canonical grid)."""
    from scripts.workcell_photo_objects import decode_mask
    mask = np.zeros(frame['valid'].shape, bool)
    for o in record['observations']:
        if o['photo'] == photo:
            word, k = o['source'].removeprefix('SAM: ').rsplit('; instance ', 1)
            mask |= decode_mask(frame['raw'], masks_by_photo[photo][word][int(k)])
    return mask


def overlay(frame, names, owner, visible, gantry, title):
    """Photo with the gantry mask (blue), each member's visible reprojection (filled) and hidden part (outline)."""
    image = cv2.cvtColor(np.ascontiguousarray(frame['rgb']), cv2.COLOR_RGB2BGR)
    tint = image.copy()
    tint[gantry] = (255, 120, 0)
    image = cv2.resize(cv2.addWeighted(image, .6, tint, .4, 0), None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    palette = [(0, 0, 255), (0, 200, 0), (0, 255, 255), (255, 0, 255), (0, 128, 255), (255, 255, 0), (128, 0, 255), (0, 255, 128)]
    for n, name in enumerate(names):
        colour = palette[n % len(palette)]
        for region, fill in (((owner == n) & visible, True), ((owner == n) & ~visible, False)):
            big = cv2.resize(region.astype(np.uint8), None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
            contours, _ = cv2.findContours(big, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if fill:
                layer = image.copy()
                cv2.drawContours(layer, contours, -1, colour, -1)
                image = cv2.addWeighted(image, .55, layer, .45, 0)
            cv2.drawContours(image, contours, -1, colour, 1)
        ys, xs = np.nonzero(owner == n)
        if len(xs):
            cv2.putText(image, name.replace('gantry-', ''), (int(np.median(xs)) * 2 + 4, int(np.median(ys)) * 2),
                        cv2.FONT_HERSHEY_SIMPLEX, .45, colour, 1, cv2.LINE_AA)
    cv2.rectangle(image, (0, 0), (image.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(image, title, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def proxy_anchoring(root, record, frames, geometry, kinds=('signal light', 'sign')):
    """Catalog lamp/sign proxies and their per-photo support: distance to the nearest gantry member (native)."""
    from scripts.workcell_extra_models import _clean
    root = Path(root)
    catalog = json.loads((root / 'objects.json').read_text())
    segmentation = json.loads((root / 'sam3.json').read_text())
    proxies = trimesh.load(root / 'object-proxies.glb', force='scene')
    rows = []
    for item in catalog['objects']:
        if item['kind'] not in kinds or not any(o['photo'] in frames for o in item['observations']):
            continue
        row = {'objectId': item['id'], 'kind': item['kind'], 'proxySourcePhotos': (item.get('proxyModel') or {}).get('sourcePhotos')}
        if item.get('model') and item['model']['nodes'][0] in proxies.graph.nodes_geometry:
            node = item['model']['nodes'][0]
            transform, name = proxies.graph[node]
            mesh = proxies.geometry[name].copy()
            mesh.apply_transform(transform)
            samples = np.concatenate([mesh.vertices, trimesh.sample.sample_surface_even(mesh, 400, seed=0)[0]])
            gap, nearest = box_distance(samples, record)
            centre_gap, _ = box_distance(mesh.vertices.mean(0), record)
            row['proxy'] = {'gapNative': round(float(gap.min()), 4), 'nearestMember': nearest[gap.argmin()],
                            'centreDistanceNative': round(float(centre_gap[0]), 4),
                            'fromThisScene': bool(set(row['proxySourcePhotos'] or []) <= set(frames))}
        row['observations'] = []
        for o in item['observations']:
            if o['photo'] not in frames:
                continue
            points = _clean(o, frames[o['photo']], geometry, segmentation, lambda name: np.load(root / name))[0]
            if not len(points):
                continue
            gap, nearest = box_distance(points, record)
            centre_gap, centre_member = box_distance(np.median(points, axis=0), record)
            row['observations'].append({'photo': o['photo'], 'source': o['source'], 'supportPoints': int(len(points)),
                                        'nearestSupportGapNative': round(float(np.percentile(gap, 5)), 4),
                                        'medianSupportGapNative': round(float(np.median(gap)), 4),
                                        'centreDistanceNative': round(float(centre_gap[0]), 4), 'nearestMember': centre_member[0]})
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, required=True, help='run directory: frame_*.json.gz and geometry.json')
    parser.add_argument('--segmentation', type=Path, required=True, help='sam3.json-format result containing the gantry words')
    parser.add_argument('--photos', type=int, nargs='+', required=True, help='photos of ONE scene')
    parser.add_argument('--words', nargs='+', default=list(GANTRY_WORDS))
    parser.add_argument('--name', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    geometry = json.loads((args.root / 'geometry.json').read_text())
    masks = masks_from_segmentation(json.loads(args.segmentation.read_text()), args.photos, args.words)
    scene, record = build_gantry(args.root, args.photos, masks, geometry)
    record['segmentationFile'] = str(args.segmentation)
    args.out.mkdir(parents=True, exist_ok=True)
    if scene is not None:
        (args.out / f'gantry-{args.name}.glb').write_bytes(scene.export(file_type='glb'))
        record['model'] = {'file': f'gantry-{args.name}.glb', 'nodes': [m['node'] for m in record['members']]}
        frames = _frames(args.root, record['photos'])
        boxes, names = record_boxes(record), [m['id'] for m in record['members']]
        record['overlays'] = []
        for p in record['photos']:
            owner, _, _, visible = reproject(boxes, frames[p])
            v = record['validation'][str(p)]
            name = f'gantry-{args.name}-photo-{p}.jpg'
            cv2.imwrite(str(args.out / name), overlay(frames[p], names, owner, visible, gantry_mask(record, masks, frames[p], p),
                        f"{args.name} photo {p}: IoU {v['iou']:.2f}, unoccluded {v['iouVisible']:.2f}"), [cv2.IMWRITE_JPEG_QUALITY, 88])
            record['overlays'].append(name)
        if (args.root / 'objects.json').is_file() and (args.root / 'object-proxies.glb').is_file():
            record['proxyAnchoring'] = proxy_anchoring(args.root, record, frames, geometry)
    (args.out / f'gantry-{args.name}.json').write_text(json.dumps(record, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'status': record['status'], 'reason': record.get('reason'),
                      'members': [[m['id'], m['lengthNative'], m['widthNative'], m['sourcePhotos'], m['singleView']]
                                  for m in record.get('members', [])],
                      'validation': {p: [v['iou'], v['iouVisible']] for p, v in record.get('validation', {}).items()}}))


if __name__ == '__main__':
    main()
