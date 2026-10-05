"""Straight edges: do the photos show a line where each model has a long straight edge?

Model edges are crease and boundary edges of the displayed mesh (dihedral angle >= CREASE_DEG) after quadric decimation to
triangles of about TRIANGLE_M (at most MAX_FACES): RecGen surfaces are marching-cube noise at full resolution; decimation keeps
sharp features and flattens the rest, and triangles that small keep smooth surfaces (a 10 cm radius) below the crease angle.
Short crease edges are merged into straight chains: members within CHAIN_DEG of the chain direction and CHAIN_LATERAL_M of
its line, gaps <= CHAIN_GAP_M, members covering >= CHAIN_DENSITY of the chain, chain length >= MIN_EDGE_M. Creases are fixed
3D lines, so one edge can be found in several photos and triangulated; straight runs of a silhouette are not (on a curved
surface the contour slides with the viewpoint and two of them triangulate to a line off the surface), and outlines are already
scored by the masks and the NCC shape check. Round or smooth objects therefore have no edges here: no_lines, not a failure.

Per photo an edge is tested where it is visible (SAMPLES points ray-cast against every displayed model, itself included,
tolerance VIS_TOL_M) over a contiguous run >= MIN_RUN_PX long. Image lines are cv2 LSD segments >= LSD_MIN_PX. Candidates lie
within +-BAND_PX of the projected edge (both segment ends), within ANGLE_DEG of its direction, and overlap it; they are grouped
by offset (2 px gaps) and the group covering most of the projected edge wins. The edge is supported when that coverage is
>= MIN_COVER; ambiguous (not support) when another group > AMBIG_PX away covers >= AMBIG_RATIO as much (wire grids, ribs,
both faces of a thin plate). Offset = signed distance of the group's fitted line from the projected edge at the edge middle;
offset_cm = offset_px * z / fx * nativeToMeters * 100 at that edge's depth z. For these photos BAND_PX = 12 px is about
0.9 cm at 2 m and 2.5 cm at 6 m: an edge further off than that is unsupported, not an offset.
Chance control: the same test on the edge moved CONTROL_X * BAND_PX to either side gives the support rate of clutter alone.

An edge supported in >= 2 photos is triangulated (least-squares line of the back-projected planes) and its distance from the
model edge reported at the edge middle and ends, with the change of that distance when one image line moves 1 px (root sum
square over photos, cmPerPx); triangulations with cmPerPx > MAX_CM_PER_PX are reported but ill-conditioned and not judged.

Verdict per object:
  no_lines  no model edge tested in any photo, or no image line within +-WIDE_PX and ANGLE_DEG of any tested edge
            (nothing to compare: round, smooth or texture-free);
  partial   support fraction (supported / tested edge-photo pairs) < SUPPORT_OK, or < CHANCE_RATIO x the chance control,
            or fewer than MIN_SUPPORTED supported pairs (reason says which);
  offset    otherwise, if a photo's median |offset| > OFFSET_CM (>= 2 supported edges in that photo) or the median
            triangulated distance at edge middles > TRI_CM (>= 2 well-conditioned edges);
  ok        otherwise.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

CREASE_DEG, TRIANGLE_M, MAX_FACES = 30.0, .02, 100000
CHAIN_DEG, CHAIN_LATERAL_M, CHAIN_GAP_M, MIN_EDGE_M, CHAIN_DENSITY = 4.0, .01, .03, .10, .6
VIS_TOL_M, SAMPLES, MIN_RUN_PX = .02, 64, 60.0
LSD_MIN_PX, BAND_PX, WIDE_PX, ANGLE_DEG, MIN_COVER, AMBIG_PX, AMBIG_RATIO, CONTROL_X = 20.0, 12.0, 48.0, 2.0, .3, 4.0, .6, 3.0
MAX_CM_PER_PX = .5
SUPPORT_OK, CHANCE_RATIO, MIN_SUPPORTED, OFFSET_CM, TRI_CM = .5, 2.0, 4, 1.0, 2.0


def _core():
    try:
        import shape_core as wsc  # in the Modal image
    except ImportError:
        import workcell_shape_check as wsc
    return wsc


def decimate(V, F, target):
    import open3d as o3d
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(np.asarray(V, float)), o3d.utility.Vector3iVector(np.asarray(F, np.int32)))
    m.remove_duplicated_vertices(); m.remove_degenerate_triangles()
    if len(m.triangles) > target:
        m = m.simplify_quadric_decimation(target)
        m.remove_degenerate_triangles()
    m.remove_unreferenced_vertices()
    return np.asarray(m.vertices, float), np.asarray(m.triangles, np.int64)


def feature_edges(V, F, deg=CREASE_DEG):
    """Edges whose two faces meet at >= deg, and boundary edges, as (A, B) endpoint arrays."""
    tri = V[F]; n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]); n /= np.maximum(np.linalg.norm(n, axis=1), 1e-30)[:, None]
    E = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1); fid = np.tile(np.arange(len(F)), 3)
    order = np.lexsort((E[:, 1], E[:, 0])); E, fid = E[order], fid[order]
    _, start, count = np.unique(E[:, 0] * (len(V) + 1) + E[:, 1], return_index=True, return_counts=True)
    f0, f1 = fid[start], np.where(count >= 2, fid[np.minimum(start + 1, len(fid) - 1)], -1)
    cos = np.where(f1 >= 0, (n[f0] * n[np.maximum(f1, 0)]).sum(1), -1.0)
    keep = cos <= math.cos(math.radians(deg))
    return V[E[start[keep], 0]], V[E[start[keep], 1]]


def chain_edges(A, B, S):
    """Greedy merge of short segments into long straight 3D edges (thresholds in metres, S = nativeToMeters)."""
    lat, gap, lmin = CHAIN_LATERAL_M / S, CHAIN_GAP_M / S, MIN_EDGE_M / S
    L = np.linalg.norm(B - A, axis=1); ok = L > 1e-12; A, B, L = A[ok], B[ok], L[ok]
    D = (B - A) / L[:, None]; M = (A + B) / 2; cos_tol = math.cos(math.radians(CHAIN_DEG))
    used = np.zeros(len(A), bool); edges = []
    for seed in np.argsort(-L):
        if used[seed]:
            continue
        used[seed] = True; p, d = M[seed], D[seed]
        for _ in range(2):  # gather, refit, gather again
            free = ~used; free[seed] = True
            c = np.nonzero(free & (np.abs(D @ d) >= cos_tol))[0]; ra, rb = A[c] - p, B[c] - p
            da = np.linalg.norm(ra - (ra @ d)[:, None] * d, axis=1); db = np.linalg.norm(rb - (rb @ d)[:, None] * d, axis=1)
            member = np.zeros(len(A), bool); member[c[(da <= lat) & (db <= lat)]] = True; member[seed] = True
            w = np.repeat(L[member], 2); P = np.vstack([A[member], B[member]])
            p = (w[:, None] * P).sum(0) / w.sum(); Q = (P - p) * np.sqrt(w)[:, None]
            d = np.linalg.svd(Q, full_matrices=False)[2][0]
        idx = np.nonzero(member)[0]
        s = np.sort(np.c_[(A[idx] - p) @ d, (B[idx] - p) @ d], axis=1); order = np.argsort(s[:, 0]); s, idx = s[order], idx[order]
        run_start, end, run = s[0, 0], s[0, 1], [idx[0]]
        for (s0, s1), i in list(zip(s[1:], idx[1:])) + [((np.inf, np.inf), -1)]:
            if s0 - end > gap:
                length = end - run_start
                if length >= lmin and L[run].sum() >= CHAIN_DENSITY * length:
                    edges.append((p + run_start * d, p + end * d)); used[run] = True
                run_start, end, run = s0, s1, [i]
            else:
                end = max(end, s1); run.append(i)
    return edges


def model_edges(V, F, S):
    tri = V[F]; area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum() / 2 * S * S
    Vd, Fd = decimate(V, F, int(np.clip(area / (.433 * TRIANGLE_M ** 2), 2000, MAX_FACES)))  # ~equilateral TRIANGLE_M triangles
    return chain_edges(*feature_edges(Vd, Fd), S)


def detect_lines(gray):
    """LSD segments (N, 4) x1 y1 x2 y2 in full-resolution pixels, >= LSD_MIN_PX long."""
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    seg = lsd.detect(np.clip(gray, 0, 255).astype(np.uint8))[0]
    seg = np.zeros((0, 4)) if seg is None else seg.reshape(-1, 4).astype(float)
    return seg[np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1]) >= LSD_MIN_PX]


def fit_line(seg):
    """Length-weighted total least squares line through segment endpoints: (point, unit direction)."""
    w = np.repeat(np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1]), 2); P = seg.reshape(-1, 2)
    p = (w[:, None] * P).sum(0) / w.sum(); d = np.linalg.svd((P - p) * np.sqrt(w)[:, None], full_matrices=False)[2][0]
    return p, d


def match(a, b, seg, band=BAND_PX):
    """Best image line for projected edge a->b. Returns dict(status, offsetPx, coverage, line) or status none/far."""
    Lm = float(np.linalg.norm(b - a)); u = (b - a) / Lm; nrm = np.array([-u[1], u[0]])
    p1, p2 = seg[:, :2] - a, seg[:, 2:] - a
    v = seg[:, 2:] - seg[:, :2]; v /= np.linalg.norm(v, axis=1)[:, None]
    o1, o2, s1, s2 = p1 @ nrm, p2 @ nrm, p1 @ u, p2 @ u
    lo, hi = np.clip(np.minimum(s1, s2), 0, Lm), np.clip(np.maximum(s1, s2), 0, Lm)
    near = (np.abs(v @ u) >= math.cos(math.radians(ANGLE_DEG))) & (hi - lo > 0)
    wide = max(WIDE_PX, band)
    if not (near & (np.abs(o1) <= wide) & (np.abs(o2) <= wide)).any():
        return dict(status='none')
    cand = np.nonzero(near & (np.abs(o1) <= band) & (np.abs(o2) <= band))[0]
    if not len(cand):
        return dict(status='far')
    off = (o1[cand] + o2[cand]) / 2; order = np.argsort(off); cand, off = cand[order], off[order]
    groups = np.split(np.arange(len(cand)), np.nonzero(np.diff(off) > 2)[0] + 1)

    def cover(g):
        iv = sorted(zip(lo[cand[g]], hi[cand[g]])); total, (cs, ce) = 0.0, iv[0]
        for s, e in iv[1:]:
            if s > ce:
                total += ce - cs; cs, ce = s, e
            else:
                ce = max(ce, e)
        return (total + ce - cs) / Lm
    covers = [cover(g) for g in groups]; best = int(np.argmax(covers)); gb = groups[best]
    p, d = fit_line(seg[cand[gb]]); mid = (a + b) / 2
    cross = lambda x, y: x[0] * y[1] - x[1] * y[0]
    offset = float(cross(p - mid, d) / cross(nrm, d))  # fitted line meets the edge's normal at the edge middle + offset * nrm
    rival = max((c for i, (c, g) in enumerate(zip(covers, groups)) if i != best and abs(np.mean(off[g]) - np.mean(off[gb])) > AMBIG_PX), default=0.0)
    status = 'weak' if covers[best] < MIN_COVER else 'ambiguous' if rival >= AMBIG_RATIO * covers[best] else 'supported'
    return dict(status=status, offsetPx=offset, coverage=covers[best], rivalCoverage=rival, line=(p, d),
                angleDeg=float(math.degrees(math.acos(min(1.0, abs(d @ u))))))


def line_plane(cam, p, d):
    """Back-projected plane (4-vector, unit normal) of the image line through p with direction d."""
    l = np.cross([p[0], p[1], 1.0], [p[0] + d[0], p[1] + d[1], 1.0])
    P = cam['K'] @ np.c_[cam['R'], cam['t']]; pi = P.T @ l
    return pi / np.linalg.norm(pi[:3])


def triangulate(planes, origin):
    """Least-squares 3D line of >= 2 planes (coordinates centred at origin): point, direction, widest plane angle (deg)."""
    Pl = np.array([np.r_[pi[:3], pi[3] + pi[:3] @ origin] for pi in planes])
    Vt = np.linalg.svd(Pl)[2]
    n1, n2 = Vt[0], Vt[1]  # the two planes spanning the pencil
    d = np.cross(n1[:3], n2[:3]); d /= np.linalg.norm(d)
    X = np.linalg.lstsq(np.vstack([n1[:3], n2[:3], d]), -np.r_[n1[3], n2[3], 0.0], rcond=None)[0]
    ang = max(math.degrees(math.acos(min(1.0, abs(Pl[i, :3] @ Pl[j, :3])))) for i in range(len(Pl)) for j in range(i + 1, len(Pl)))
    return X + origin, d, ang


def point_line(X, p, d):
    r = X - p
    return np.linalg.norm(r - (r @ d)[..., None] * d, axis=-1)


def edge_view(cam, A, B, scene, S, wsc):
    """Longest visible contiguous run of edge A->B in one photo: (2D ends a, b, median depth) or None."""
    t = np.linspace(0, 1, SAMPLES); X = A + t[:, None] * (B - A)
    uv, z = wsc.project(cam, X)
    inside = (z > 0) & (uv[:, 0] >= 2) & (uv[:, 1] >= 2) & (uv[:, 0] < cam['w'] - 3) & (uv[:, 1] < cam['h'] - 3)
    dist = np.linalg.norm(X - cam['C'], axis=1)
    hit = wsc.first_hit(scene, np.repeat(cam['C'][None], len(X), 0), X)
    vis = inside & (hit >= 1 - (VIS_TOL_M / S) / dist)
    best, cur, start = (0, 0), 0, 0
    for i, v in enumerate(np.r_[vis, False]):
        if v:
            if cur == 0:
                start = i
            cur += 1
        else:
            if cur > best[1] - best[0]:
                best = (start, start + cur)
            cur = 0
    i0, i1 = best[0], best[1] - 1
    if i1 <= i0 or np.linalg.norm(uv[i1] - uv[i0]) < MIN_RUN_PX:
        return None
    return uv[i0], uv[i1], float(np.median(z[i0:i1 + 1]))


def assess(tested, supported, control, per_photo, tri, any_near):
    """(verdict, reason); see the module docstring."""
    if tested == 0:
        return 'no_lines', 'no_model_edge_tested'
    if not any_near:
        return 'no_lines', 'no_image_line_near_any_edge'
    if supported / tested < SUPPORT_OK:
        return 'partial', 'low_support'
    if supported / tested < CHANCE_RATIO * control:
        return 'partial', 'support_near_chance'
    if supported < MIN_SUPPORTED:
        return 'partial', 'too_few_supported'
    bad2d = any(p['supported'] >= 2 and p['medianAbsOffsetCm'] > OFFSET_CM for p in per_photo.values())
    good = [t['distMidCm'] for t in tri if t['wellConditioned']]
    if bad2d:
        return 'offset', 'photo_offset'
    if len(good) >= 2 and float(np.median(good)) > TRI_CM:
        return 'offset', 'triangulated_offset'
    return 'ok', ''


def check_object(edges, cams, segs, scene, S, wsc, band=BAND_PX):
    rows, per_photo, tri, any_near = [], {}, [], False
    for e, (A, B) in enumerate(edges):
        hits = {}
        for k, cam in enumerate(cams):
            view = edge_view(cam, A, B, scene, S, wsc)
            if view is None:
                continue
            a, b, z = view; m = match(a, b, segs[k], band)
            any_near |= m['status'] != 'none'
            u = (b - a) / np.linalg.norm(b - a); side = CONTROL_X * band * np.array([-u[1], u[0]])
            control = sum(match(a + sg * side, b + sg * side, segs[k], band)['status'] == 'supported' for sg in (1, -1))
            row = dict(edge=e, photo=k, uv=[a.tolist(), b.tolist()], lengthPx=float(np.linalg.norm(b - a)), depthM=z * S, status=m['status'],
                       control=int(control))
            if 'offsetPx' in m:
                row.update(offsetPx=m['offsetPx'], offsetCm=m['offsetPx'] * z / cam['K'][0, 0] * S * 100, coverage=m['coverage'],
                           rivalCoverage=m['rivalCoverage'], angleDeg=m['angleDeg'])
            rows.append(row)
            if m['status'] == 'supported':
                hits[k] = m['line']
        if len(hits) >= 2:
            ks = sorted(hits); mid = (A + B) / 2
            planes = [line_plane(cams[k], *hits[k]) for k in ks]
            p, d, ang = triangulate(planes, mid)
            ends = np.array([point_line(X, p, d) for X in (A, B)]) * S * 100
            sens = []
            for i, k in enumerate(ks):  # move one image line by 1 px along its normal
                lp, ld = hits[k]; shifted = list(planes); shifted[i] = line_plane(cams[k], lp + np.array([-ld[1], ld[0]]), ld)
                p2, d2, _ = triangulate(shifted, mid); sens.append(point_line(mid, p2, d2) - point_line(mid, p, d))
            tri.append(dict(edge=e, photos=ks, distMidCm=float(point_line(mid, p, d) * S * 100), distEndsCm=ends.tolist(),
                            angleDeg=float(math.degrees(math.acos(min(1.0, abs(d @ (B - A) / np.linalg.norm(B - A)))))),
                            maxPlaneAngleDeg=ang, cmPerPx=float(np.linalg.norm(sens) * S * 100),
                            wellConditioned=bool(np.linalg.norm(sens) * S * 100 <= MAX_CM_PER_PX)))
    for k in range(len(cams)):
        mine = [r for r in rows if r['photo'] == k]
        sup = [r for r in mine if r['status'] == 'supported']
        if mine:
            per_photo[k] = dict(tested=len(mine), supported=len(sup), ambiguous=sum(r['status'] == 'ambiguous' for r in mine),
                                medianAbsOffsetPx=float(np.median([abs(r['offsetPx']) for r in sup])) if sup else None,
                                medianAbsOffsetCm=float(np.median([abs(r['offsetCm']) for r in sup])) if sup else None)
    tested, supported = len(rows), sum(r['status'] == 'supported' for r in rows)
    control = sum(r['control'] for r in rows) / (2 * tested) if tested else 0.0
    good = [t['distMidCm'] for t in tri if t['wellConditioned']]
    return dict(modelEdges=len(edges), tested=tested, supported=supported, ambiguous=sum(r['status'] == 'ambiguous' for r in rows),
                supportFraction=supported / tested if tested else None, controlFraction=control if tested else None,
                perPhoto=per_photo, triangulated=tri, wellConditionedTriangulations=len(good),
                medianTriDistCm=float(np.median(good)) if good else None,
                **dict(zip(('verdict', 'reason'), assess(tested, supported, control, per_photo, tri, any_near))), rows=rows)


def overlay(gray, items, scale=.25):
    """Downscaled photo with projected model edges: green supported, yellow ambiguous, red unsupported."""
    img = cv2.cvtColor(cv2.resize(np.clip(gray, 0, 255).astype(np.uint8), None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA), cv2.COLOR_GRAY2BGR)
    colour = dict(supported=(60, 200, 60), ambiguous=(0, 210, 255))
    for a, b, status in items:
        cv2.line(img, tuple(np.round(a * scale).astype(int)), tuple(np.round(b * scale).astype(int)), colour.get(status, (60, 60, 230)), 2, cv2.LINE_AA)
    return cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()


def run(ctx, opts):
    wsc = _core(); S, cams = ctx['S'], ctx['cams']; band = float(opts.get('bandPx', BAND_PX))
    scene = wsc.raycast_scene([o['mesh'] for o in ctx['objects']])
    segs = [detect_lines(g) for g in ctx['gray']]
    out, draw = {}, {k: [] for k in range(len(cams))}
    for o in ctx['objects']:
        edges = model_edges(*o['mesh'], S)
        res = check_object(edges, cams, segs, scene, S, wsc, band)
        for r in res['rows']:
            draw[r['photo']].append((np.array(r['uv'][0]), np.array(r['uv'][1]), r['status']))
        if not opts.get('keepRows'):
            res.pop('rows')
        out[o['id']] = dict(label=o['label'], **res)
    files = {f'photo{k + 1}.jpg': overlay(ctx['gray'][k], draw[k]) for k in draw} if opts.get('plots', True) else {}
    counts = {}
    for v in out.values():
        counts[v['verdict']] = counts.get(v['verdict'], 0) + 1
    return dict(status='ok', thresholds=dict(bandPx=band, widePx=WIDE_PX, angleDeg=ANGLE_DEG, minCover=MIN_COVER, creaseDeg=CREASE_DEG,
                                             triangleM=TRIANGLE_M, maxFaces=MAX_FACES, minEdgeM=MIN_EDGE_M, minRunPx=MIN_RUN_PX, supportOk=SUPPORT_OK,
                                             offsetCm=OFFSET_CM, triCm=TRI_CM, minSupported=MIN_SUPPORTED, maxCmPerPx=MAX_CM_PER_PX, controlPx=CONTROL_X * band,
                                             chanceRatio=CHANCE_RATIO),
                imageSegments=[len(s) for s in segs], verdictCounts=counts, objects=out, files=files)


def _render(cams, V, F, shade, size, rng):
    """Box with one grey per face over a dark background, ray-cast per pixel, slightly blurred, with noise."""
    import open3d as o3d
    scene = o3d.t.geometry.RaycastingScene(); scene.add_triangles(o3d.core.Tensor(V.astype(np.float32)), o3d.core.Tensor(F.astype(np.uint32)))
    images = []
    for cam in cams:
        ys, xs = np.mgrid[0:size[1], 0:size[0]]
        rays = (np.linalg.inv(cam['K']) @ np.vstack([xs.ravel(), ys.ravel(), np.ones(xs.size)])).T @ cam['R']  # pixel centres
        hit = scene.cast_rays(o3d.core.Tensor(np.hstack([np.repeat(cam['C'][None], len(rays), 0), rays]).astype(np.float32)))
        prim = hit['primitive_ids'].numpy().astype(np.int64); bg = prim == np.iinfo(np.uint32).max
        img = np.where(bg, 20.0, shade[np.where(bg, 0, prim)]).reshape(size[1], size[0])
        img = cv2.GaussianBlur(img.astype(np.float32), (0, 0), .8) + rng.normal(0, 2, img.shape).astype(np.float32)
        images.append(img)
    return images


def _check():
    """Synthetic box seen by 3 cameras: the true copy matches (~0 offset); a copy moved by 2.5 cm must show that offset in
    every photo and in the triangulated edges; a copy 8 cm off finds no support."""
    wsc = _core(); rng = np.random.default_rng(1); S = 1.3; size = (960, 720)
    K = np.array([[1000, 0, 480], [0, 1000, 360], [0, 0, 1.]])

    def look(C, target=np.zeros(3)):
        z = target - C; z /= np.linalg.norm(z); x = np.cross(z, [0, -1, 0]); x /= np.linalg.norm(x); y = np.cross(z, x)
        M = np.eye(4); M[:3, :3] = np.c_[x, y, z]; M[:3, 3] = C
        return wsc.camera(dict(cameraToWorld=M.tolist(), K=K.tolist(), width=size[0], height=size[1], imageId=str(C)))
    cams = [look(np.array(c)) for c in ([-.9, -.5, -1.6], [.2, -.7, -1.8], [1.0, -.4, -1.5])]
    V, F = wsc.primitive_mesh(dict(kind='box', parameters=dict(dimensions=[.5, .4, .45])))
    shade = 60 + 30 * (np.arange(len(F)) // 2).astype(float)  # one grey per box face (triangles come in pairs)
    images = _render(cams, V, F, shade, size, rng); segs = [detect_lines(g) for g in images]
    edges = model_edges(V, F, S)
    assert len(edges) == 12, len(edges)
    true = check_object(edges, cams, segs, wsc.raycast_scene([(V, F)]), S, wsc)
    assert true['verdict'] == 'ok' and true['supportFraction'] > .8 and true['controlFraction'] < .1, true['verdict']
    assert max(abs(r['offsetPx']) for r in true['rows'] if r['status'] == 'supported') < 1.0
    shift = np.array([3, -1.5, 1.0]); shift *= .025 / S / np.linalg.norm(shift)  # 2.5 cm, not along any box edge
    moved = [(A + shift, B + shift) for A, B in edges]
    res = check_object(moved, cams, segs, wsc.raycast_scene([(V + shift, F)]), S, wsc)
    errs = []
    for r in res['rows']:
        if r['status'] != 'supported':
            continue
        A, B = moved[r['edge']]; cam = cams[r['photo']]
        a, b = (wsc.project(cam, np.array([A, B]))[0]); u = (b - a) / np.linalg.norm(b - a); nrm = np.array([-u[1], u[0]])
        (pa, pb) = wsc.project(cam, np.array([A - shift, B - shift]))[0]
        expect = (((pa + pb) / 2 - (a + b) / 2) @ nrm)  # where the true edge is, relative to the model edge, at its middle
        errs.append(r['offsetPx'] - expect)
    assert len(errs) >= 12 and np.max(np.abs(errs)) < .6, errs
    for t in res['triangulated']:
        A, B = moved[t['edge']]; d = (B - A) / np.linalg.norm(B - A); perp = (shift - (shift @ d) * d)
        assert abs(t['distMidCm'] - np.linalg.norm(perp) * S * 100) < .25, (t, np.linalg.norm(perp) * S * 100)
    assert len(res['triangulated']) >= 6 and res['verdict'] == 'offset', res['verdict']
    far = check_object([(A + 3.2 * shift, B + 3.2 * shift) for A, B in edges], cams, segs, wsc.raycast_scene([(V + 3.2 * shift, F)]), S, wsc)
    assert far['verdict'] == 'partial' and far['supportFraction'] < .2, (far['supported'], far['tested'])  # 8 cm off: outside the band
    big = [abs(r['offsetCm']) for r in res['rows'] if r['status'] == 'supported']
    print('lines self-test passed: edges', len(edges), 'true median |offset| px',
          round(float(np.median([abs(r['offsetPx']) for r in true['rows'] if r['status'] == 'supported'])), 2),
          'moved median |offset| cm', round(float(np.median(big)), 2), 'max px err', round(float(np.max(np.abs(errs))), 2),
          'triangulated', len(res['triangulated']), 'verdict', res['verdict'])


if __name__ == '__main__':
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    _check()
