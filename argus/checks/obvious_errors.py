"""Obvious errors of a four-view report's DISPLAY: what a viewer sees as wrong at a glance.

User rule: a small accuracy gap is acceptable; obvious errors are not (missing floor, objects through the floor, very bad
looking models). Every gate is generic (no object-type rules) and has one fixed threshold, stated here. ctx =
scripts/workcell_shape_check.py load_report (displayed models = the measurement layer's where it has one, else the
publication's); 'box' = the layer's box_faces record of the object (layer 'boxes'); heights are above the unified floor (the
layer's ground, else the frame's) x nativeToMeters (read, never changed). opts: api (publication origin: floor surface and
point-cloud assets), draw (default true: floor maps, crops of failed objects), drawObjects (ids also cropped). Floor-role entities (geometryRole 'floor') are judged by G1 only.

Report level
G1 floor present. Besides the models the default 3-D scene draws (viewer rules, app bundle of 2026-10-06) the floor entity's
   active model, else its observed reference surfaces (observed_surface, sourceKind observed_reference_surface, sourceRefs
   naming one of its observations at that revision), and the report's point cloud (the model point-cloud overlay, on by
   default); a representation is drawn when not stale, in the report frame, and confirmed or a candidate. Obvious when
   - there is no floor entity, or nothing displayed lies at floor level (+-5 cm);
   - coverage: of the floor the photos see in front of and under the objects (by its pixel area in those photos), < 90 % has
     displayed floor at floor level (a surface sample, or >= 2 point-cloud points, within +-5 cm of the unified floor; the
     unweighted cell fraction is reported as frontByArea). Required cells (5 cm on the unified
     floor): for each object standing on or near the floor (model bottom <= 0.5 m) and each photo with its mask, the cells
     within 0.5 m of its footprint (its footprint included) on that camera's side of the object's centre, inside that photo,
     outside every object mask of that photo and not hidden from its camera by a displayed model more than 2 cm above the
     cell (so under an object only where the photo sees under it; models with holes do not make floor 'seen');
   - flatness, per displayed floor source (surface samples; point-cloud points inside the floor entity's photo masks): its own
     plane fit (refit on the best 90 %) has residual p95 > 2 cm, or its median height is > 2 cm off the unified floor (the
     objects then stand above / in the visible floor).
   Warning: the coverage holds only with the point-cloud overlay (unchecking it leaves < 90 %).
G8 scale: each reference-object feature of the frame's scale (sourceRefs features: measuredNative, specM), re-read at the
   scale in effect, deviates <= 4 % from its specification; obvious otherwise; no features = warning (not checked).

Per object (its displayed model)
G2 through the floor: robust bottom (0.5th percentile of the vertex heights, the report's modelBottomCm) < -2 cm.
G3 floating: a floor-contact object (box floorContact, or a box bottom <= 3 cm graded medium / high) with bottom > 3 cm.
G4 interpenetration: overlap volume of two displayed models > 10 % of the smaller one's volume. Occupancy on a grid aligned
   with each model's box axes, 1 cm cells (coarser for big models: <= 2M cells each): a closed model (>= 99 % of edges shared
   by exactly two faces) occupies the cells whose centre is inside (3-ray majority), an open sheet (or a closed one thinner
   than a cell) the cells within half a cell diagonal of its surface; a cell of both counts only when it lies > 1 cm inside
   one of the two closed solids (touching is not penetrating; two sheets never overlap).
   Parent / child pairs (parentEntityId) are skipped; siblings (same parent) are warnings.
G5 photo fit: visible silhouette (first hit among all displayed models, a ray every 4 px) vs the report mask (>= 200 px) in
   each masked photo: IoU < 0.35 in every masked photo, or < 0.2 in any, is obvious; < 0.35 in some photo is a warning. An
   object with displayed parts (children by parentEntityId; its mask is the whole object) is seen as itself and its parts; a
   part is not hidden by its parent.
G6 mesh sanity: connected components (vertices merged at 1e-6 of the model span); a fragment is a component (not the largest)
   whose box misses the object's box (box_faces; without one the largest component's box) grown by 5 cm on
   every side: > 2 % of the faces in fragments is obvious. Exploded / degenerate: the full vertex extent along the box axes
   > 3 x the box size (sizes below 5 cm count as 5 cm) on any axis.
G7 dims: model extent (0.5-99.5 percentile of surface samples along the box axes) vs a box dimension graded medium / high
   differs by > 30 % and > 2 cm: obvious for L or H; for W (depth) a warning only (box_faces credits depth outline edges
   that a front view also sees, so W does not count in the box's own confidence either).
Severity of an object: 'obvious' if any gate failed as obvious, else 'warning' if any warning, else None.

modal run modal_apps/workcell_view_checks.py --checks obvious_errors ... --opts OPTS.json ({"obvious_errors": {"api": ORIGIN}}),
or as a modal_apps/workcell_layer_trial.py task (run(ctx, opts), opts {"api": ORIGIN}); no arguments = synthetic self-test."""
import io
import json
import math
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np

import argus.checks.shape_core as wsc
from argus.checks import transfer as tr
from argus.checks.box_faces import basis, floor_masks

CELL_M, NEAR_M, STAND_M, LEVEL_M, PC_MIN, HIDE_M = .05, .5, .5, .05, 2, .02
COVER_SEEN, FLAT_P95_CM, FLAT_OFF_CM = .90, 2., 2.
SINK_CM, FLOAT_CM, CONTACT_BOX_M = 2., 3., .03
VOX_M, MAX_CELLS, DEEP_M, OVERLAP, CLOSED = .01, 2_000_000, .01, .10, .99
IOU_ALL, IOU_ANY, STRIDE, MIN_MASK = .35, .20, 4, 200
FRAG, GROW_M, EXPLODE, MIN_BOX_M = .02, .05, 3., .05
DIM_REL, DIM_ABS_M = .30, .02
SCALE_DEV = .04
VERIFIED = ('medium', 'high')
THRESHOLDS = dict(G1=dict(cellM=CELL_M, nearM=NEAR_M, standingBottomM=STAND_M, floorLevelM=LEVEL_M, pointsPerCell=PC_MIN,
                          occluderAboveM=HIDE_M, frontCoverage=COVER_SEEN, planeP95Cm=FLAT_P95_CM,
                          offsetCm=FLAT_OFF_CM),
                  G2=dict(sinkCm=SINK_CM), G3=dict(floatCm=FLOAT_CM, boxContactM=CONTACT_BOX_M),
                  G4=dict(cellM=VOX_M, maxCells=MAX_CELLS, deepM=DEEP_M, overlapOfSmaller=OVERLAP, closedEdges=CLOSED),
                  G5=dict(iouEvery=IOU_ALL, iouAny=IOU_ANY, stridePx=STRIDE, minMaskPx=MIN_MASK),
                  G6=dict(fragmentFaces=FRAG, growM=GROW_M, explode=EXPLODE, minBoxM=MIN_BOX_M),
                  G7=dict(relative=DIM_REL, absoluteM=DIM_ABS_M, verified=list(VERIFIED)), G8=dict(deviation=SCALE_DEV))


# ---------------------------------------------------------------- report geometry

def fetch(api, asset_id):
    get = lambda url: urllib.request.urlopen(url, timeout=180).read()  # noqa: E731
    url = json.loads(get(f'{api}/api/assets/{asset_id}'))['url']
    return get(url if url.startswith('http') else api + url)


def geometry(ctx, rep, api):
    """(points, [(V, F)]) of a floor-surface / point-cloud representation, placed (native)."""
    meta = {a['id']: a for a in ctx['doc']['assets']}[rep['assetId']]
    fmt = meta.get('format') or (meta.get('metadata') or {}).get('format')
    data, T = fetch(api, rep['assetId']), rep['transform']
    if fmt == 'panoptes-mesh-v1':
        V, F = wsc.read_packed(data, meta.get('byteLayout') or meta['metadata']['byteLayout'])
        return np.zeros((0, 3)), [(wsc.placed(V, T), F)]
    import trimesh
    sc = trimesh.load(io.BytesIO(data), file_type='glb', force='scene'); pts, meshes = [], []
    for node in sc.graph.nodes_geometry:
        M, name = sc.graph[node]; g = sc.geometry[name]; X = wsc.placed(trimesh.transform_points(np.asarray(g.vertices, float), M), T)
        if len(getattr(g, 'faces', [])):
            meshes.append((X, np.asarray(g.faces, np.int64)))
        else:
            pts.append(X)
    return (np.vstack(pts) if pts else np.zeros((0, 3))), meshes


def drawn(e, r, frame_id):
    """Viewer rule (Le): the representation is drawn when not stale, in the report frame, confirmed or a candidate."""
    t = r.get('transform') or {}
    return (e.get('visible') is not False and r.get('sourceValidity') != 'stale' and r.get('coordinateFrameId') == frame_id
            and t.get('coordinateFrameId') == frame_id and bool(r.get('assetId') or r.get('primitive'))
            and (r.get('placementState') == 'confirmed' or r.get('placementReason') in ('requires_alignment_confirmation', 'imported_proposal')))


def floor_reps(doc):
    """[(entity, representation, source)] the default scene draws as floor: floor entities' active models, else their observed
    reference surfaces (viewer rule ge), and every point cloud."""
    fid = doc['coordinateFrames'][0]['id']; obs = {o['id']: o for o in doc['observations']}; out = []
    for e in doc['entities']:
        for r in e.get('representations') or []:
            if not drawn(e, r, fid):
                continue
            if r['kind'] == 'point_cloud':
                out.append((e, r, 'point_cloud'))
            elif e.get('geometryRole') == 'floor' and not e.get('sourceContext'):
                if e.get('activeModelRepresentationId'):
                    if r['id'] == e['activeModelRepresentationId']:
                        out.append((e, r, 'surface'))
                elif r['kind'] == 'observed_surface' and r.get('sourceKind') == 'observed_reference_surface' and any(
                        isinstance(s, dict) and s.get('observationId') in (e.get('observationRefs') or []) and s['observationId'] in obs
                        and obs[s['observationId']].get('revision') == s.get('revision') and obs[s['observationId']].get('imageId') == s.get('imageId')
                        for s in r.get('sourceRefs') or []):
                    out.append((e, r, 'surface'))
    return out


def plane_stats(X):
    """X: (x, y, h) metres in the floor frame. Own plane fit (SVD, refit on the best 90 %): residual p95 (cm), tilt to the unified
    floor (deg), median height (cm)."""
    def fit(P):
        c = P.mean(0); n = np.linalg.svd(P - c, full_matrices=False)[2][2]
        return c, n * np.sign(n[2] or 1)
    c, n = fit(X); r = np.abs((X - c) @ n)
    c, n = fit(X[r <= np.percentile(r, 90)]); r = np.abs((X - c) @ n)
    return dict(points=int(len(X)), planeP95Cm=round(float(np.percentile(r, 95)) * 100, 2),
                tiltDeg=round(math.degrees(math.acos(min(1., abs(float(n[2]))))), 2), medianCm=round(float(np.median(X[:, 2])) * 100, 2))


def g1_floor(ctx, objs, api, scene):
    """Load what the default scene draws as floor (floor_reps), then judge it (judge_floor)."""
    n, d = ctx['floor']; S = ctx['S']; e1, e2 = basis(n); doc = ctx['doc']
    floors = [e['id'] for e in doc['entities'] if e.get('geometryRole') == 'floor' and not e.get('sourceContext')]
    if not api:
        return dict(floorEntities=floors, sources=[], failed=[('not checked: no api',)], warnings=[])
    rng = np.random.default_rng(0); surf, pc, sources = [], [], []
    for e, r, src in floor_reps(doc):
        P, meshes = geometry(ctx, r, api); row = dict(entityId=e['id'], representationId=r['id'], kind=r['kind'], source=src)
        if src == 'surface':
            area = [float(np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1).sum() / 2) * S * S for V, F in meshes]
            surf += [wsc.sample_surface(V, F, int(min(4e5, max(5e4, a * 4000))), rng)[0] for (V, F), a in zip(meshes, area)]; row['areaM2'] = round(sum(area), 2)
        else:
            pc.append(P); row['points'] = int(len(P))
        sources.append(row)
    P = np.vstack(pc) if pc else np.zeros((0, 3)); inm = np.zeros(len(P), bool)
    for k, m in floor_masks(ctx).items():  # point-cloud points the photos call floor (inside a floor-entity mask)
        cam = ctx['cams'][k]; uv, z = wsc.project(cam, P); ok = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 1] >= 0) & (uv[:, 0] < cam['w']) & (uv[:, 1] < cam['h'])
        ui = uv[ok].astype(int); inm[np.nonzero(ok)[0][m[ui[:, 1], ui[:, 0]]]] = True
    out = judge_floor(ctx, objs, np.vstack(surf) if surf else np.zeros((0, 3)), P, inm, scene)
    out.update(floorEntities=floors, sources=sources, pointCloudFloorMaskPoints=int(inm.sum()))
    if not floors:
        out['failed'].insert(0, ('no floor entity',))
    return out


def judge_floor(ctx, objs, surf, P, inmask, scene):
    """G1 on displayed floor surface samples surf and point-cloud points P (inmask: inside a floor mask), native."""
    n, d = ctx['floor']; S = ctx['S']; e1, e2 = basis(n); rng = np.random.default_rng(0)
    to_f = lambda X: np.c_[X @ e1 * S, X @ e2 * S, (X @ n + d) * S]  # noqa: E731 - floor frame, metres
    surf, pcf = to_f(surf), to_f(P); out = dict(failed=[], warnings=[]); flat = {}
    if len(surf):
        flat['surface'] = plane_stats(surf)
    if inmask.sum() >= 100:
        flat['pointCloud'] = plane_stats(pcf[inmask])
    out['flatness'] = flat
    for src, st in flat.items():
        if st['planeP95Cm'] > FLAT_P95_CM:
            out['failed'].append((f'{src} floor not flat: plane-fit p95 {st["planeP95Cm"]:.1f} cm > {FLAT_P95_CM:.0f}',))
        if abs(st['medianCm']) > FLAT_OFF_CM:
            out['failed'].append((f'{src} floor {st["medianCm"]:+.1f} cm off the unified floor',))
    stand = []
    for o in objs:
        V, F = o['mesh']
        if np.percentile((V @ n + d) * S, .5) <= STAND_M:
            stand.append((o['id'], to_f(wsc.sample_surface(V, F, 20000, rng)[0])[:, :2], [k for k, m in o['masks'].items() if m.sum() >= MIN_MASK]))
    if not stand:
        out['coverage'] = None; return out
    allxy = np.vstack([xy for _, xy, _ in stand]); lo = allxy.min(0) - NEAR_M - CELL_M
    shape = tuple(np.ceil((allxy.max(0) + NEAR_M + CELL_M - lo) / CELL_M).astype(int) + 1)

    def mark(xy, need=1):
        ij = np.floor((xy - lo) / CELL_M).astype(int); ok = (ij >= 0).all(1) & (ij[:, 0] < shape[0]) & (ij[:, 1] < shape[1])
        c = np.zeros(shape, np.int32); np.add.at(c, (ij[ok, 0], ij[ok, 1]), 1); return c >= need
    cov_s = mark(surf[np.abs(surf[:, 2]) <= LEVEL_M, :2]); cov_p = mark(pcf[np.abs(pcf[:, 2]) <= LEVEL_M, :2], PC_MIN)
    if not (cov_s.any() or cov_p.any()):
        out['failed'].append(('nothing displayed at floor level',))
    foot = {oid: fill(cv2.dilate(mark(xy).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0) for oid, xy, _ in stand}
    under = np.logical_or.reduce(list(foot.values()))
    gxy = lo + (np.stack(np.mgrid[0:shape[0], 0:shape[1]], -1) + .5) * CELL_M; camxy = [to_f(c['C'][None])[0, :2] for c in ctx['cams']]
    want = {}  # (object, photo): cells within NEAR_M of its footprint on that camera's side of its centre
    for oid, xy, photos in stand:
        ring = cv2.distanceTransform((~foot[oid]).astype(np.uint8), cv2.DIST_L2, 5) * CELL_M <= NEAR_M; c = xy.mean(0)
        for k in photos:
            want[oid, k] = ring & ((gxy - c) @ (camxy[k] - c) > 0)
    vis = {}
    for k, cam in enumerate(ctx['cams']):  # inside the photo, not in any object's mask there, not hidden by a displayed model
        ms = [m for (_, kk), m in want.items() if kk == k]
        if not ms:
            continue
        ij = np.argwhere(np.logical_or.reduce(ms)); xy = lo + (ij + .5) * CELL_M; X = -d * n + np.outer(xy[:, 0] / S, e1) + np.outer(xy[:, 1] / S, e2)
        uv, z = wsc.project(cam, X); ok = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 1] >= 0) & (uv[:, 0] < cam['w']) & (uv[:, 1] < cam['h'])
        L = np.linalg.norm(X - cam['C'], axis=1) * S; t = wsc.first_hit(scene, np.repeat(cam['C'][None], len(X), 0), X); ok &= t >= 1 - HIDE_M / L
        ui = np.clip(uv.astype(int), 0, [cam['w'] - 1, cam['h'] - 1])
        for o in objs:  # the photo's own word: a masked object is in front (see-through models let rays through)
            if k in o['masks']:
                ok &= ~o['masks'][k][ui[:, 1], ui[:, 0]]
        vis[k] = np.zeros(shape); D = X - cam['C']; dist = np.linalg.norm(D, axis=1)  # pixel area of each visible cell in photo k
        vis[k][tuple(ij[ok].T)] = (cam['K'][0, 0] * cam['K'][1, 1] * (CELL_M / S) ** 2 * np.abs(D @ n) / dist ** 3)[ok]
    req = np.zeros(shape, bool); per = {}; tot = dict(all=0., s=0., p=0., b=0.); pob = {}
    both = cov_s | cov_p
    for (oid, k), m in want.items():
        r = m & (vis[k] > 0); req |= r; per[oid] = per.get(oid, np.zeros(shape, bool)) | r
        w = vis[k][r]; pob.setdefault(oid, [0., 0.]); pob[oid][0] += w.sum(); pob[oid][1] += w[both[r]].sum()
    for k in vis:  # pixel-weighted: each photo's required cells once
        r = np.logical_or.reduce([m for (_, kk), m in want.items() if kk == k]) & (vis[k] > 0); w = vis[k][r]
        tot['all'] += w.sum(); tot['b'] += w[both[r]].sum(); tot['s'] += w[cov_s[r]].sum(); tot['p'] += w[cov_p[r]].sum()
    frac = lambda c, rq: round(float((c & rq).sum() / rq.sum()), 4) if rq.sum() else None  # noqa: E731
    pfrac = lambda key: round(tot[key] / tot['all'], 4) if tot['all'] else None  # noqa: E731
    cov = dict(cells=list(shape), cellsRequired=int(req.sum()), cellsUnderSeen=int((req & under).sum()), front=pfrac('b'), frontSurfaceOnly=pfrac('s'),
               frontPointCloudOnly=pfrac('p'), frontByArea=frac(both, req), underSeenByArea=frac(both, req & under),
               objects={oid: dict(front=round(b / a, 4) if a else None, frontByArea=frac(both, r), cells=int(r.sum())) for oid, r in per.items() for a, b in [pob[oid]]})
    out['coverage'] = cov
    img = np.full(shape + (3,), 255, np.uint8); img[cov_s] = (200, 230, 200); img[cov_p & ~cov_s] = (215, 215, 160); img[under] = (120, 120, 120)
    img[req & both] = (60, 170, 60); img[req & ~both] = (40, 40, 230)
    for xy in camxy:  # cameras, if on the map
        q = np.floor((xy - lo) / CELL_M).astype(int)
        if (q >= 0).all() and (q < shape).all():
            img[max(0, q[0] - 1):q[0] + 2, max(0, q[1] - 1):q[1] + 2] = (200, 0, 200)
    out['map'] = cv2.resize(np.ascontiguousarray(img.transpose(1, 0, 2)[::-1]), None, fx=6, fy=6, interpolation=cv2.INTER_NEAREST)  # e1 right, e2 up
    out['photoMaps'] = {}
    for k in vis:  # the required cells of each photo drawn on it: green covered, red missing
        rk = np.logical_or.reduce([m for (_, kk), m in want.items() if kk == k]) & (vis[k] > 0); ij = np.argwhere(rk)
        if 'gray' not in ctx or not len(ij):
            continue
        xy = lo + (ij + .5) * CELL_M; X = -d * n + np.outer(xy[:, 0] / S, e1) + np.outer(xy[:, 1] / S, e2); uv = wsc.project(ctx['cams'][k], X)[0]
        g = np.clip(ctx['gray'][k], 0, 255).astype(np.uint8); sc_ = 1400 / g.shape[1]
        im = cv2.cvtColor(cv2.resize(g, None, fx=sc_, fy=sc_, interpolation=cv2.INTER_AREA), cv2.COLOR_GRAY2BGR)
        for (u, v), c in zip(uv * sc_, both[rk]):
            cv2.circle(im, (int(u), int(v)), 3, (60, 170, 60) if c else (40, 40, 230), -1)
        out['photoMaps'][k] = im
    if cov['front'] is not None and cov['front'] < COVER_SEEN:
        bad = sorted((o for o, r in cov['objects'].items() if r['front'] is not None and r['front'] < COVER_SEEN), key=lambda o: cov['objects'][o]['front'])
        out['failed'].append((f'floor the photos see in front of / under the objects covered {100 * cov["front"]:.0f} % < {100 * COVER_SEEN:.0f} % (worst: {", ".join(b[:8] for b in bad[:5])})',))
    if not out['failed'] and (cov['frontSurfaceOnly'] or 0) < COVER_SEEN:
        out['warnings'].append((f'only with the point-cloud overlay: without it the floor covers {100 * (cov["frontSurfaceOnly"] or 0):.0f} %',))
    return out


def fill(m):
    from scipy.ndimage import binary_fill_holes
    return binary_fill_holes(m)


def g8_scale(ctx):
    S = ctx['S']; rows = []
    for ref in ((ctx['doc']['coordinateFrames'][0].get('scale') or {}).get('sourceRefs') or []):
        for f in ref.get('features') or []:
            if f.get('measuredNative') and f.get('specM'):
                rows.append(dict(entityId=ref.get('entityId'), photo=f.get('photo'), feature=f.get('feature'), specM=f['specM'],
                                 measuredM=round(f['measuredNative'] * S, 5), deviation=round(f['measuredNative'] * S / f['specM'] - 1, 4)))
    worst = max((abs(r['deviation']) for r in rows), default=None)
    out = dict(nativeToMeters=S, features=rows, maxDeviation=worst, failed=[], warnings=[])
    if worst is None:
        out['warnings'].append(('no reference-object features: scale not checked',))
    elif worst > SCALE_DEV:
        out['failed'].append((f'reference feature off by {100 * worst:.1f} % > {100 * SCALE_DEV:.0f} %',))
    return out


# ---------------------------------------------------------------- per-object geometry

def floor_axes(V, n):
    """Rows l, w, u: u = floor normal, l along the minimum-area rectangle of the horizontal projection (no box)."""
    e1, e2 = basis(n); (_, _), (a, b), ang = cv2.minAreaRect(np.c_[V @ e1, V @ e2].astype(np.float32))
    t = math.radians(ang + (90 if b > a else 0)); l = math.cos(t) * e1 + math.sin(t) * e2
    return np.array([l, np.cross(n, l), n])


def merged(V, F):
    """Faces on vertices merged at 1e-6 of the model span; vertex count."""
    key = np.round(V / max(float(np.ptp(V, 0).max()), 1e-12) * 1e6).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True); inv = inv.reshape(-1)
    return inv[F], int(inv.max()) + 1


def closed_fraction(Fm, nv):
    E = np.sort(np.r_[Fm[:, [0, 1]], Fm[:, [1, 2]], Fm[:, [2, 0]]], 1); _, c = np.unique(E[:, 0] * nv + E[:, 1], return_counts=True)
    return float((c == 2).mean())


def fragments(V, F, A, S, rec=None, faces_out=False):
    """G6: faces in components (all but the largest) whose box misses the object's box grown by GROW_M: the box_faces box when
    there is one, else the largest component's box. Returns (fraction, components, fragments[, per-face fragment flags])."""
    from scipy import ndimage
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    Fm, nv = merged(V, F)
    r, c = np.r_[Fm[:, 0], Fm[:, 1]], np.r_[Fm[:, 1], Fm[:, 2]]
    k, lab = connected_components(coo_matrix((np.ones(len(r), np.int8), (r, c)), shape=(nv, nv)), directed=False)
    faces = np.bincount(lab[Fm[:, 0]], minlength=k)
    if (faces > 0).sum() == 1:
        return (0., 1, 0) + ((np.zeros(len(F), bool),) if faces_out else ())
    Vm = np.zeros((nv, 3)); Vm[Fm.ravel()] = V[F.ravel()]; X = Vm @ A.T * S
    used = np.unique(Fm); idx = np.arange(k)
    lo = np.stack([ndimage.minimum(X[used, i], lab[used], idx) for i in range(3)], 1)
    hi = np.stack([ndimage.maximum(X[used, i], lab[used], idx) for i in range(3)], 1)
    if rec:
        ctr = np.array(rec['centerNative'], float) @ A.T * S; half = np.array(rec['sizeM'], float) / 2; blo, bhi = ctr - half, ctr + half
    else:
        blo, bhi = lo[int(np.argmax(faces))], hi[int(np.argmax(faces))]
    frag = (faces > 0) & (idx != int(np.argmax(faces))) & ((lo > bhi + GROW_M) | (hi < blo - GROW_M)).any(1)
    out = (float(faces[frag].sum() / faces.sum()), int((faces > 0).sum()), int(frag.sum()))
    return out + ((frag[lab[Fm[:, 0]]],) if faces_out else ())


def solid(V, F, A, S):
    """Occupied cells of one displayed model on a grid aligned with axes A: centres (native), deep flags, cell (m), volume (m3)."""
    import open3d as o3d
    X = V @ A.T; lo, hi = X.min(0), X.max(0); size = np.maximum((hi - lo) * S, VOX_M)
    v = max(VOX_M, float(np.prod(size) / MAX_CELLS) ** (1 / 3)); cnt = np.maximum(1, np.ceil(size / v).astype(int))
    g = [lo[i] + (np.arange(cnt[i]) + .5) * v / S for i in range(3)]
    G = np.stack(np.meshgrid(*g, indexing='ij'), -1).reshape(-1, 3) @ A
    scene = o3d.t.geometry.RaycastingScene(); scene.add_triangles(o3d.core.Tensor(V.astype(np.float32)), o3d.core.Tensor(F.astype(np.uint32)))
    closed = closed_fraction(*merged(V, F)) >= CLOSED
    occ, deep = occupancy(scene, G, closed, v, S)
    if closed and not occ.any():  # closed but thinner than a cell: a shell
        closed = False; occ, deep = occupancy(scene, G, closed, v, S)
    return dict(P=G[occ], deep=deep[occ], cell=v, closed=closed, volume=float(occ.sum()) * v ** 3, scene=scene, A=A, lo=lo, hi=hi)


def occupancy(scene, P, closed, v, S):
    import open3d as o3d
    t = o3d.core.Tensor(P.astype(np.float32)); dist = scene.compute_distance(t).numpy() * S
    if not closed:  # an open sheet: the cells its surface passes through
        return dist <= v * math.sqrt(3) / 2, np.zeros(len(P), bool)
    inside = scene.compute_occupancy(t, nsamples=3).numpy() > .5
    return inside, inside & (dist > DEEP_M)


def overlap(a, b, S, offset=0.):
    """Overlap volume (m3) of solids a (moved by offset, native) and b (solid()), counted on a's cells: occupied in both, > DEEP_M
    inside one of them."""
    if not len(a['P']):
        return 0.
    Q = a['P'] + offset; X = Q @ b['A'].T; m = a['cell'] / S
    near = ((X >= b['lo'] - m) & (X <= b['hi'] + m)).all(1)
    if not near.any():
        return 0.
    occ, deep = occupancy(b['scene'], Q[near], b['closed'], b['cell'], S)
    return float((occ & (deep | a['deep'][near])).sum()) * a['cell'] ** 3


def g4_pairs(ctx, objs, axes):
    S = ctx['S']; ents = {e['id']: e for e in ctx['doc']['entities']}; parent = {i: (ents.get(i) or {}).get('parentEntityId') for i in ents}
    box = {o['id']: (o['mesh'][0].min(0), o['mesh'][0].max(0)) for o in objs}; sol, rows = {}, []
    for i, a in enumerate(objs):
        for b in objs[i + 1:]:
            if parent.get(a['id']) == b['id'] or parent.get(b['id']) == a['id']:
                continue
            (la, ha), (lb, hb) = box[a['id']], box[b['id']]
            if (la > hb).any() or (lb > ha).any():
                continue
            for o in (a, b):
                if o['id'] not in sol:
                    sol[o['id']] = solid(*o['mesh'], axes[o['id']], S)
            sa, sb = sol[a['id']], sol[b['id']]
            small, big = (sa, sb) if sa['volume'] <= sb['volume'] else (sb, sa)
            ov = overlap(small, big, S); ratio = ov / small['volume'] if small['volume'] > 0 else 0.
            if ratio > OVERLAP:
                sib = parent.get(a['id']) is not None and parent.get(a['id']) == parent.get(b['id'])
                rows.append(dict(a=a['id'], b=b['id'], overlapM3=round(ov, 5), ratio=round(ratio, 3), volumesM3=[round(sa['volume'], 5), round(sb['volume'], 5)],
                                 closed=[sa['closed'], sb['closed']], siblings=sib))
    return rows, {k: dict(volumeM3=round(s['volume'], 5), closed=s['closed'], cellM=round(s['cell'], 4)) for k, s in sol.items()}


def family(ctx, objs):
    """Per displayed object (index): its displayed parts (children by parentEntityId) and its displayed parent, as indices."""
    ents = {e['id']: e for e in (ctx.get('doc') or {}).get('entities', [])}; idx = {o['id']: i for i, o in enumerate(objs)}
    par = [idx.get((ents.get(o['id']) or {}).get('parentEntityId')) for o in objs]
    return [[j for j, p in enumerate(par) if p == i] for i in range(len(objs))], par


def g5_iou(ctx, objs, scene, keep=False):
    """G5 per object and masked photo: IoU of the visible silhouette with the report mask (block-centre samples every STRIDE px).
    An object with displayed parts is seen as itself and its parts (its mask is the whole object); a part is not hidden by its
    parent. keep: also return per object the recall / precision and the silhouette in its largest-mask photo (for drawing)."""
    kids, par = family(ctx, objs); out = {o['id']: {} for o in objs}; fit = {o['id']: {} for o in objs}; sil = {}; scenes = {(): scene}
    big = {i: max(o['masks'], key=lambda k: o['masks'][k].sum()) for i, o in enumerate(objs) if o['masks']}
    for k, cam in enumerate(ctx['cams']):
        todo = [i for i, o in enumerate(objs) if k in o['masks'] and o['masks'][k].sum() >= MIN_MASK]
        labels = {}
        for i in todo:
            excl = () if par[i] is None else (par[i],)
            if excl not in labels:
                keep_ids = [j for j in range(len(objs)) if j not in excl]
                if excl not in scenes:
                    scenes[excl] = tr.scene_of([objs[j]['mesh'] for j in keep_ids])
                lab = tr.cast(cam, scenes[excl], 0, 0, -(-cam['w'] // STRIDE), -(-cam['h'] // STRIDE), STRIDE, len(keep_ids))
                labels[excl] = np.where(lab >= 0, np.array(keep_ids)[np.clip(lab, 0, None)], -1)
            label = labels[excl]; o = objs[i]
            m = o['masks'][k][STRIDE // 2::STRIDE, STRIDE // 2::STRIDE][:label.shape[0], :label.shape[1]]
            m = np.pad(m, [(0, label.shape[0] - m.shape[0]), (0, label.shape[1] - m.shape[1])]); v = np.isin(label, [i] + kids[i])
            out[o['id']][k + 1] = round(float((v & m).sum() / max(1, (v | m).sum())), 3)
            fit[o['id']][k + 1] = dict(recall=round(float((v & m).sum() / max(1, m.sum())), 3), precision=round(float((v & m).sum() / max(1, v.sum())), 3))
            if keep and big.get(i) == k:
                sil[o['id']] = (k, v, m)
    return (out, fit, sil) if keep else out


def judge_iou(ious):
    if not ious:
        return None
    v = list(ious.values())
    return 'obvious' if max(v) < IOU_ALL or min(v) < IOU_ANY else 'warning' if min(v) < IOU_ALL else None


def dims(V, F, A, S, rec, rng):
    """Model extents (0.5-99.5 percentile of surface samples, m) and full vertex extents along the box axes; G7 / G6 explode."""
    X = wsc.sample_surface(V, F, 20000, rng)[0] @ A.T * S; ext = np.percentile(X, 99.5, 0) - np.percentile(X, .5, 0)
    full = np.ptp(V @ A.T, 0) * S; out = dict(modelM=[round(float(x), 4) for x in ext], fullM=[round(float(x), 4) for x in full], g7=[], explode=[])
    if rec:
        for i, dim in enumerate(('L', 'W', 'H')):
            b = rec['sizeM'][i]; c = rec['dims'][dim]['confidence']
            if c in VERIFIED and abs(ext[i] - b) > max(DIM_REL * b, DIM_ABS_M):
                out['g7'].append(dict(dim=dim, modelM=round(float(ext[i]), 4), boxM=b, confidence=c, rel=round(float(ext[i] / b - 1), 3)))
            if full[i] > EXPLODE * max(b, MIN_BOX_M):
                out['explode'].append(dict(axis=dim, fullM=round(float(full[i]), 4), boxM=b))
    return out


# ---------------------------------------------------------------- run

def run(ctx, opts):
    if not ctx['floor']:
        return dict(status='no_floor', objects={})
    import open3d as o3d  # noqa: F401 - scene building below
    n, d = ctx['floor']; S = ctx['S']; rng = np.random.default_rng(0)
    ents = {e['id']: e for e in ctx['doc']['entities']}
    objs = [o for o in ctx['objects'] if (ents.get(o['id']) or {}).get('geometryRole') != 'floor']
    boxes = (ctx.get('layer') or {}).get('boxes') or {}
    axes = {o['id']: np.array(boxes[o['id']]['axes'], float) if o['id'] in boxes else floor_axes(o['mesh'][0], n) for o in objs}
    scene = tr.scene_of([o['mesh'] for o in objs])
    report = dict(G1=g1_floor(ctx, objs, opts.get('api'), scene), G8=g8_scale(ctx))
    pairs, solids = g4_pairs(ctx, objs, axes); ious, fit, sil = g5_iou(ctx, objs, scene, keep=True)
    objects = {}
    for o in objs:
        V, F = o['mesh']; rec = boxes.get(o['id']); h = (V @ n + d) * S * 100; bottom = float(np.percentile(h, .5)); fails = []
        contact = bool(rec and (rec['floorContact'] or (rec['dims']['bottom']['confidence'] in VERIFIED and rec['bottomM'] <= CONTACT_BOX_M)))
        if bottom < -SINK_CM:
            fails.append(dict(gate='G2', severity='obvious', bottomCm=round(bottom, 2), text=f'model bottom {bottom:.1f} cm below the floor (> {SINK_CM:.0f})'))
        if contact and bottom > FLOAT_CM:
            fails.append(dict(gate='G3', severity='obvious', bottomCm=round(bottom, 2), text=f'floor-contact object floats {bottom:.1f} cm (> {FLOAT_CM:.0f})'))
        for p in pairs:
            if o['id'] in (p['a'], p['b']):
                other = p['b'] if p['a'] == o['id'] else p['a']
                fails.append(dict(gate='G4', severity='warning' if p['siblings'] else 'obvious', other=other, ratio=p['ratio'], overlapM3=p['overlapM3'],
                                  text=f'{100 * p["ratio"]:.0f} % of the smaller model inside {other[:8]}' + (' (siblings)' if p['siblings'] else '')))
        sev = judge_iou(ious[o['id']])
        if sev:
            fails.append(dict(gate='G5', severity=sev, iou=ious[o['id']], text=f'silhouette IoU {ious[o["id"]]}'))
        frac, ncomp, nfrag = fragments(V, F, axes[o['id']], S, rec)
        if frac > FRAG:
            fails.append(dict(gate='G6', severity='obvious', fragmentFaces=round(frac, 4), components=ncomp, fragments=nfrag,
                              text=f'{100 * frac:.1f} % of the faces in {nfrag} fragments off the body'))
        dm = dims(V, F, axes[o['id']], S, rec, rng)
        for x in dm['explode']:
            fails.append(dict(gate='G6', severity='obvious', **x, text=f'exploded: {x["axis"]} extent {x["fullM"]:.2f} m > {EXPLODE:.0f} x box {x["boxM"]:.2f}'))
        for x in dm['g7']:
            fails.append(dict(gate='G7', severity='warning' if x['dim'] == 'W' else 'obvious', **x,
                              text=f'{x["dim"]} {100 * x["modelM"]:.1f} cm vs box {100 * x["boxM"]:.1f} cm ({x["confidence"]}): {100 * x["rel"]:+.0f} %'))
        severity = 'obvious' if any(f['severity'] == 'obvious' for f in fails) else 'warning' if fails else None
        objects[o['id']] = dict(label=o['label'], severity=severity, failed=fails, bottomCm=round(bottom, 2), topCm=round(float(np.percentile(h, 99.5)), 2),
                                floorContact=contact, iou=ious[o['id']], maskFit=fit[o['id']], fragmentFaces=round(frac, 4), components=ncomp, extents=dm,
                                solid=solids.get(o['id']), box=dict(sizeM=rec['sizeM'], confidence={k: v['confidence'] for k, v in rec['dims'].items()},
                                                                    capped=any(r['code'] in {'measurement.cap.coverage', 'measurement.cap.residual'} for r in rec['highlightReasons'])) if rec else None)
    for g in ('G1', 'G8'):
        r = report[g]; r['severity'] = 'obvious' if r['failed'] else 'warning' if r['warnings'] else None
    summary = dict(obviousObjects=sorted(k for k, v in objects.items() if v['severity'] == 'obvious'),
                   warningObjects=sorted(k for k, v in objects.items() if v['severity'] == 'warning'),
                   obviousReport=[g for g in ('G1', 'G8') if report[g]['severity'] == 'obvious'],
                   byGate={g: sorted(k for k, v in objects.items() if any(f['gate'] == g and f['severity'] == 'obvious' for f in v['failed']))
                           for g in ('G2', 'G3', 'G4', 'G5', 'G6', 'G7')})
    summary['obviousTotal'] = len(summary['obviousObjects']) + len(summary['obviousReport'])
    files = {}
    if opts.get('draw', True):
        if report['G1'].get('map') is not None:
            files['floor-coverage.png'] = cv2.imencode('.png', report['G1']['map'])[1].tobytes()
        for k, im in (report['G1'].get('photoMaps') or {}).items():
            files[f'floor-photo{k + 1}.jpg'] = cv2.imencode('.jpg', im, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
        for oid, row in objects.items():  # failed objects, and any named in opts drawObjects (e.g. the ones a fix changed)
            if (row['failed'] or oid in (opts.get('drawObjects') or ())) and oid in sil:
                files[f'object-{oid[:8]}.jpg'] = crop(ctx, sil[oid], row)
    report['G1'].pop('map', None); report['G1'].pop('photoMaps', None)
    out = dict(status='ok', thresholds=THRESHOLDS, nativeToMeters=S, floor=dict(normal=[float(x) for x in n], offset=float(d)),
               report=report, objects=objects, interpenetration=pairs, summary=summary)
    out = json.loads(json.dumps(out, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x)))  # numpy scalars -> JSON
    out['files'] = files
    return out


def crop(ctx, sil, row, width=640):
    """The object's largest-mask photo around its mask (green) and visible model silhouette (magenta), gates written on it."""
    k, v, m = sil; g = np.clip(ctx['gray'][k], 0, 255).astype(np.uint8); H, W = g.shape
    up = lambda a: cv2.resize(a.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)  # noqa: E731
    vf, mf = up(v), up(m); ys, xs = np.nonzero(vf | mf)
    if not len(xs):
        return b''
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max(); px, py = int(.15 * (x1 - x0)) + 20, int(.15 * (y1 - y0)) + 20
    x0, x1, y0, y1 = max(0, x0 - px), min(W, x1 + px), max(0, y0 - py), min(H, y1 + py)
    img = cv2.cvtColor(g[y0:y1, x0:x1], cv2.COLOR_GRAY2BGR); s = width / max(1, x1 - x0); img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    for a, col in ((mf, (0, 200, 0)), (vf, (255, 0, 255))):
        cs, _ = cv2.findContours(cv2.resize(a[y0:y1, x0:x1], (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cs, -1, col, 2, cv2.LINE_AA)
    for t, line in enumerate(['photo %d  mask green, model magenta' % (k + 1)] + [f"{f['gate']} {f['severity']}: {f['text']}"[:90] for f in row['failed']]):
        cv2.putText(img, line, (6, 18 + 18 * t), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, line, (6, 18 + 18 * t), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1, cv2.LINE_AA)
    return cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()


# ---------------------------------------------------------------- self-test

def _box(lo, hi):
    V, F = wsc.primitive_mesh(dict(kind='box', parameters=dict(dimensions=list(np.subtract(hi, lo)))))
    return V + (np.add(lo, hi) / 2), F


def _check():
    """Synthetic: z-up floor, S = 1. Solids, fragments, IoU, dims, plane stats and the scale gate."""
    import open3d as o3d  # noqa: F401
    A, S = np.eye(3), 1.
    a, b = solid(*_box([0, 0, 0], [.2, .2, .2]), A, S), solid(*_box([.1, 0, 0], [.3, .2, .2]), A, S)
    r = overlap(a, b, S) / a['volume']; assert a['closed'] and .35 < r < .55, r                       # half inside
    c = solid(*_box([.2, 0, 0], [.4, .2, .2]), A, S); assert overlap(a, c, S) / a['volume'] < .02     # touching: no penetration
    sheet = (np.array([[-.1, .1, -.1], [.3, .1, -.1], [.3, .1, .3], [-.1, .1, .3]]), np.array([[0, 1, 2], [0, 2, 3]]))
    sh = solid(*sheet, A, S); assert not sh['closed']
    r = overlap(sh, a, S) / sh['volume']; assert .15 < r < .3, r                                      # (0.2 / 0.4)^2 of the sheet inside, less the 1 cm skin
    V1, F1 = _box([0, 0, 0], [.2, .2, .9]); V2, F2 = _box([1, 1, 0], [1.05, 1.05, .05]); body = dict(centerNative=[.1, .1, .45], sizeM=[.2, .2, .9])
    for b in (body, None):  # with the object's box and without (the largest component's box)
        frac, k, nf = fragments(np.r_[V1, V2], np.r_[F1, F2 + len(V1)], A, S, b); assert k == 2 and nf == 1 and abs(frac - .5) < 1e-9, (frac, k, nf)
    V3, F3 = _box([.22, 0, 0], [.4, .2, .9]); assert fragments(np.r_[V1, V3], np.r_[F1, F3 + len(V1)], A, S, body)[0] == 0  # 2 cm off the box: kept
    wide = dict(centerNative=[.6, .6, .45], sizeM=[1.2, 1.2, .9]); assert fragments(np.r_[V1, V2], np.r_[F1, F2 + len(V1)], A, S, wide)[0] == 0  # pieces inside the box
    rec = dict(sizeM=[.2, .2, .6], dims={k: dict(confidence=c) for k, c in dict(L='high', W='medium', H='high', bottom='high').items()})
    dm = dims(V1, F1, A, S, rec, np.random.default_rng(0)); assert [x['dim'] for x in dm['g7']] == ['H'] and not dm['explode'], dm
    dm = dims(np.r_[V1, [[0, 0, 5.]]], np.r_[F1, [[0, 1, 8]]], A, S, rec, np.random.default_rng(0)); assert [x['axis'] for x in dm['explode']] == ['H'], dm
    rng = np.random.default_rng(1); X = np.c_[rng.random((5000, 2)) * 3, .03 + rng.normal(0, .005, 5000)]
    st = plane_stats(X); assert abs(st['medianCm'] - 3) < .1 and .8 < st['planeP95Cm'] < 1.2 and st['tiltDeg'] < .5, st
    assert judge_iou({1: .3, 2: .32}) == 'obvious' and judge_iou({1: .8, 2: .15}) == 'obvious' and judge_iou({1: .8, 2: .3}) == 'warning' and judge_iou({1: .8}) is None
    K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1.]]); M = np.eye(4); M[:3, 3] = [.1, .1, -2]
    cam = wsc.camera(dict(cameraToWorld=M.tolist(), K=K.tolist(), width=640, height=480, imageId='c'))
    Vb, Fb = _box([0, 0, 0], [.2, .2, .2]); lab = tr.cast(cam, tr.scene_of([(Vb, Fb)]), 0, 0, 160, 120, STRIDE, 1)
    full = np.kron(lab == 0, np.ones((STRIDE, STRIDE), bool)); shifted = np.roll(full, 60, 1)
    ctx = dict(cams=[cam]); objs = [dict(id='b', masks={0: full})]
    assert g5_iou(ctx, objs, tr.scene_of([(Vb, Fb)]))['b'][1] > .95
    objs[0]['masks'] = {0: shifted}; assert g5_iou(ctx, objs, tr.scene_of([(Vb, Fb)]))['b'][1] < .2
    # a parent (mask = whole object) and its part behind its front face: the parent is seen with its part, the part through it
    Vp, Fp = _box([-.3, 0, 0], [.2, .2, .2]); Vc, Fc = _box([-.25, .05, .05], [-.05, .15, .15])
    solo = lambda V, F: np.kron(tr.cast(cam, tr.scene_of([(V, F)]), 0, 0, 160, 120, STRIDE, 1) == 0, np.ones((STRIDE, STRIDE), bool))  # noqa: E731
    whole = solo(Vp, Fp) | solo(*_box([-.3, 0, 0], [.2, .2, .2])); fam = dict(doc=dict(entities=[dict(id='c', parentEntityId='p')]), cams=[cam])
    objs = [dict(id='p', mesh=(Vp, Fp), masks={0: whole}), dict(id='c', mesh=(Vc, Fc), masks={0: solo(Vc, Fc)})]
    got = g5_iou(fam, objs, tr.scene_of([o['mesh'] for o in objs])); assert got['p'][1] > .95 and got['c'][1] > .95, got
    frame = dict(scale=dict(sourceRefs=[dict(features=[dict(measuredNative=.031, specM=.04), dict(measuredNative=.062, specM=.08)])]))
    g = g8_scale(dict(S=1.29, doc=dict(coordinateFrames=[frame]))); assert not g['failed'] and g['maxDeviation'] < .01, g
    assert g8_scale(dict(S=1.40, doc=dict(coordinateFrames=[frame])))['failed']
    # G1: a post on a z = 0 floor seen from 2 m; a full floor passes, half a floor, a floor 3 cm low or a bumpy one fail
    from argus.checks.clearance import _look
    cams = [_look(np.array([x, -2., 1.5]), np.zeros(3)) for x in (-.5, .5)]; post = _box([-.05, -.05, 0], [.05, .05, .9])
    sc = tr.scene_of([post]); masks = {k: np.kron(tr.cast(c, sc, 0, 0, 160, 120, STRIDE, 1) == 0, np.ones((STRIDE, STRIDE), bool)) for k, c in enumerate(cams)}
    ctx = dict(floor=(np.array([0, 0, 1.]), 0.), S=1., cams=cams); objs = [dict(id='post', mesh=post, masks=masks)]
    g = np.random.default_rng(2).random((60000, 2)) * 3 - 1.5; plane = lambda z: np.c_[g, z]  # noqa: E731
    ok = judge_floor(ctx, objs, plane(np.zeros(len(g))), np.zeros((0, 3)), np.zeros(0, bool), sc)
    assert not ok['failed'] and ok['coverage']['front'] > .99 and ok['coverage']['cellsRequired'] > 100, ok
    half = judge_floor(ctx, objs, plane(np.zeros(len(g)))[g[:, 1] > .05], np.zeros((0, 3)), np.zeros(0, bool), sc)
    assert half['coverage']['front'] < .5 and len(half['failed']) == 1, half['failed']  # floor only behind the post
    low = judge_floor(ctx, objs, plane(np.full(len(g), -.03)), np.zeros((0, 3)), np.zeros(0, bool), sc)
    assert [f[0].split()[0] for f in low['failed']] == ['surface'] and 'off' in low['failed'][0][0], low['failed']
    bumpy = judge_floor(ctx, objs, plane(.04 * np.sin(9 * g[:, 0])), np.zeros((0, 3)), np.zeros(0, bool), sc)
    assert any('not flat' in f[0] for f in bumpy['failed']), bumpy['failed']
    pts = judge_floor(ctx, objs, np.zeros((0, 3)), plane(np.zeros(len(g))), np.ones(len(g), bool), sc)  # floor only as points
    assert not pts['failed'] and pts['warnings'] and pts['coverage']['frontPointCloudOnly'] > .99, pts
    print('obvious_errors self-test passed')


if __name__ == '__main__':
    _check()
