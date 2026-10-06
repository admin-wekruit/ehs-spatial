"""Capture plan (补拍规划): the photos a report still needs, as a GENERIC standard set of <= 10 viewpoints defined from the cell
entrance plus at most 4 targeted extras, judged on the report's SHARED FORMAT boxes (opts.boxes; this module never builds boxes).

FACES. A box (centerNative, axes [l, w, u], sizeM; front = -w, back = +w, left = -l, right = +l, bottom = -u, top = +u) is judged as
scripts/workcell_checks/box_faces.py grades it (R1: its sides / outline_px / face_grade / dim_grade, imported, not copied): through
its OUTLINE. Per photo the convex hull of its 8 projected corners is split into box edges, each credited to the two faces it lies on;
samples every 3 px (<= 40 per edge) are known when the sample and the pixel 3 px outside are in frame and no other model (mesh fences
and known un-modelled objects included; hits < 3 cm above the floor are markings; the object's own model and plate are not occluders)
lies in front of it by more than 5 cm + 3 % of the range. A face's edge (its samples on one silhouette side) counts when >= 80 % of it
is known and >= 20 px; its px/cm = fx * 1 cm / its straight-line distance; a face is credited in a photo at >= 10 px/cm. Grade:
high = credited in >= 2 photos >= 30 deg apart (horizontal directions from the face centre), medium = 1, low = only below 10 px/cm,
unverified = none; a dimension: both faces high -> high, both >= medium -> medium, one credited or only < 10 px/cm -> low, neither ->
unverified (bottom: the bottom face). Whether a photo sees the face itself (the file's 'photos' / 'status') is shown, not counted.
Before = the boxes file's grades; the existing photos' evidence = box_faces' per-photo pxPerCm and its caps (gradeCaps: capped box,
faces at most low, unpinned faces), added to the boxes by make_opts.py; the file's grades are recomputed from them and every
mismatch is listed (summary.beforeCheck: must be none). Where the file has no per-photo evidence it is predicted here (only in the
photos with the object's report mask, as box_faces) for the table and the reasons, and a medium face becomes high only by a new photo
>= 30 deg from every existing one. New photos are assumed to get report masks of what they show. A planned photo credits a face only
where it does so in >= 95 % of 200 Monte Carlo trials (+-0.2 m, +-0.1 m, +-5 deg yaw and pitch). After = box_faces' grade of all the
evidence (existing + planned), never below before; a capped box keeps its grades (photos do not fix the masks); a face box_faces caps
at low (unpinned, a bottom below the floor) is lifted only by a planned photo that credits it. A face still low / unverified says why,
over every spot considered (existing photos, standard set, extras and all their candidates): box capped; not within the 4 extras; the
spots that show it fail the Monte Carlo test; credited but not pinned; seen only far; seen only through a mesh (counted as hiding);
hidden by a model; never outlined. No 'contact' reason: an occluder within 5 cm + 3 % is not in front (box_faces), so a face against
something is still measured on its outline.

STANDARD SET (STANDARD; opts.standardFrame = the ids on the two sides of the entrance: E = their midpoint, x = left -> right as seen
facing the cell, out = away from it, the side the existing photos were taken from). az = horizontal angle from 'out' (+ = right),
distM = horizontal distance from the anchor: E; S = the scale reference (V7); KL / KR = the rear-left / rear-right corner of the
keep-out zones' bounding rectangle in this frame (V9 / V10). V1/V2 standing 45 deg 2.5 m; V3/V4 crouched 0.5 m, lens level, 45 deg
2.0 m (low edges above the lower third line); V5/V6 side views 80 deg 2.5 m (the back edge is on the outline from the side: depth; 80 not 90
so the side fences running into the cell do not cut the sightlines); V7 standing 1.8 m straight out from the scale reference
(<= 2.5 m straight-line); V8 standing 2.5 m out, tilted up at 2.6 m (overhead sign, lights); V9/V10 optional, 1 m diagonally outside
the rear corners. Fixed for every cell (not searched on any). Each is evaluated in the cell like a new photo: standable 'yes',
'unknown' (outside the modelled area, or floor no photo has seen: check on site) or 'no' (on a model, in the guarded cell,
unreachable, or the cell lacks its anchor); needed = standable yes / unknown and it credits a face or a target that is short now.

TARGETS (opts.targets + the scale reference) = entity id prefix + optional part (whole words bottom/lower/base/top/upper/left/right or
下/底/上/顶/左/右, or `select`: band, side, heightM, or photo + pixel [+ depthNative, + provenance] for a small object the model lacks)
+ edge 'bottom' or 'surface'. A bottom edge is the side-face samples in a 3 cm band at `edgeHeightM`, a MEASURED height with its
`edgeHeightSource` (field values are validation only, never inputs; where the model ends above it, on the downward extension of its
lowest cross-section), else at the model's lowest 3 cm. Faces of a target: samples whose normals lie within 25 deg (an edge lies on a
face long along it). `coveredBy` {face, depthM, bottomM}: a plate in front of that face, an occluder slab; a photo must look >= 45 deg
off the covered face's normal. A photo counts for a target when, on one face, >= 20 % of the samples face it and >= 80 % of those are
in frame (3 % from the borders) and not hidden (the rest of the same model too; hits within 3 cm of the floor are floor), median
|n.v| >= sin 15 deg, >= 10 px/cm (fx * 1 cm / straight-line distance), a bottom edge >= 1 m2 of free floor within 1 m seen, an
existing photo a report mask of the object. Scale reference (horizontal widths of vertical coaxial cylinders): whole in frame,
>= 11.1 px/cm, line of sight <= 30 deg from horizontal. Per target, on one face: >= 2 photos >= 30 deg apart; an edge below 0.5 m
also a low photo (camera 0.4-0.8 m, depression <= 15 deg); the scale reference in >= 2 photos. Planned photos use margins (PLAN:
seen >= 0.85 inside a 10 % border, |n.v| >= 0.32, >= 38 deg apart, depression <= 12 deg and, crouched with the lens level, the low
edge's middle above the lower third line up to 5 cm above the planned height, px/cm at distance + 0.2 m, floor >= 1.3 m2, >= 50 deg
off a covered face, e-stop tilt <= 25 deg) and keep counting for a target in >= 95 % of the 200 trials judged by the rules (vet).

EXTRAS: <= 4, greedy over the faces still short after the existing photos + the needed standard viewpoints (shortfall 2 low /
unverified, 1 medium, 0 high; capped boxes excluded) and the targets still short (their deficit): candidates on rings 1.4 / 2.0 /
2.6 m around every short box and target every 20 deg, camera 0.5 m (crouched, lens level) or 1.5 m (standing, aimed at its centre),
standable inside the modelled area (floor no photo has seen is allowed and warned); largest drop first (a pair for one target when
no single photo helps), each Monte Carlo vetted before it is taken, then pruned; one fallback >= 0.75 m away that keeps the plan as
good and is aimed at the same thing, else none.

STANDABLE: inside the bounds of the displayed models and cameras + 1 m, no model footprint (5 cm-1.9 m high; known un-modelled objects
included) within 0.25 m, outside the keep-out zones (convex hulls of entity groups; `movable` ids left out), reachable from the
existing camera positions over free floor. Floor 'observed' (<= 0.5 m from floor seen in an existing photo or within 1 m of a
photographer), 'seeThrough' (seen only through `seeThrough` models) or 'unobserved'. `knownUnmodelled` (footprintM in the plan floor
frame, or mirrorOf an entity across the perpendicular bisector of two `across` ids; heightM): objects the report lacks, added as
opaque occluders and footprints (a sightline through one is blocked). Landmarks must be fixed (`movable` ids refused).

OUTPUT: floor coordinates (x right and y into the cell as seen from the existing photos, metres, origin on the floor below their mean
position) and words relative to the nearest fixed landmark; standardSet (every viewpoint: standability, needed, targets, credited
faces), newPhotos (the extras: instruction, robustness, fallback), faces (every box face: before -> after, by which photos, which
photos see it, why short), dims (L / W / H / bottom per box, before -> after), targets; files: instructions.md and coverage.md (paste unchanged), a
top-down PNG, a CPU ray-cast preview per needed viewpoint and extra, photo 1 rendered from the other photos, a camera-convention test.
"""
import io
import math
import re
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np

try:
    import shape_core as wsc  # the check container (modal_apps/workcell_view_checks.py)
except ImportError:  # locally: scripts/workcell_shape_check.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import workcell_shape_check as wsc
from workcell_checks.box_faces import EDGE_FRAC, depth_credit, dim_grade, face_grade, hdir, outline_px, sides  # R1, the same functions box_faces grades with

RINGS_M, STEP_DEG, HEIGHTS_M, MAX_EXTRAS = (1.4, 2.0, 2.6), 20, (.5, 1.5), 4  # extras' candidates; horizontal distances
MIN_FACING, LOW_EDGE_M, LOW_CAM_M, SCALE_VIEWS, MIN_PASS = .2, .5, (.4, .8), 2, .95  # 0.5 m crouch: edges a few cm high
RULE = dict(name='rule', seen=.8, seenKey='seen', faceOn=math.sin(math.radians(15)), sepDeg=30., lowDepDeg=15., lowHeadroomM=-math.inf,
            frame=.03, padM=0., pxPerCm=10., floorM2=1., scalePxPerCm=11.1, scaleTiltDeg=30., offCoverDeg=45.)  # 10 px/cm = SHARED FORMAT
PLAN = dict(RULE, name='plan', seen=.85, seenKey='seenPlan', faceOn=.32, sepDeg=38., lowDepDeg=12., lowHeadroomM=.05, frame=.10, padM=.2,
            floorM2=1.3, scaleTiltDeg=25., offCoverDeg=50.)
JITTER = dict(posM=.2, heightM=.1, aimDeg=5., trials=200)
FLOOR_R_M, FLOOR_STEP_M, UNOBSERVED_M = 1., .1, .5
PERSON_R_M, BODY_BAND_M, PAD_M, GRID_M = .25, (.05, 1.9), 1., .05
EPS_M, FLOOR_HIT_M, LOWER_THIRD = .02, .03, 2 / 3
EDGE_BAND_M, PART_BAND, SIDE_FRAC, N_SAMPLES = .03, .25, 1 / 3, 400
FACE_DEG, FACE_MIN_SHARE, FACE_MIN_LEN = 25, .1, .5
WORDS = dict(bottom={'bottom', 'lower', 'base', '下', '底'}, top={'top', 'upper', '上', '顶'}, left={'left', '左'}, right={'right', '右'})
FACES = dict(front=(0, -1), back=(0, 1), left=(-1, 0), right=(1, 0))  # coveredBy: floor frame; front faces the existing photos
FACE_ORDER = ('notSeen', 'edgeOn', 'tooFar')
# SHARED FORMAT box faces, judged as box_faces.py measures them (evidence / level)
FACES6 = ('front', 'back', 'left', 'right', 'bottom', 'top')
FACE_ZH = dict(front='前面', back='后面', left='左面', right='右面', top='顶面', bottom='底面')
FACE_OF = {(2, 0): 2, (2, 1): 3, (1, 0): 0, (1, 1): 1, (0, 0): 4, (0, 1): 5}  # (corner bit, value) -> face; corner = 4 l + 2 w + u
FACE_AXIS = (1, 1, 0, 0, 2, 2)  # the box axis (l, w, u) along each face's normal
DIMS = dict(L=(2, 3), W=(0, 1), H=(4, 5), bottom=(4,))
CONF = ('unverified', 'low', 'medium', 'high')
CONF_ZH = dict(unverified='未验证', low='低', medium='中', high='高')
EVID_STEP, EVID_MAX_N, EVID_MIN_PX, HIGH_PX_CM, HIGH_DEG = 3., 40, 20., 10., 30.  # box_faces: 3 px samples, 20 px known, 10 px/cm, 30 deg
OCC_MARGIN_M, OCC_MARGIN_REL = .05, .03  # box_faces: another model is in front by > 5 cm + 3 % of the range
STANDARD = (  # generic, from the entrance (module doc); never tuned on a cell
    dict(id='V1', zh='左前 45° 站拍', en='front-left 45 deg, standing', anchor='E', az=-45, distM=2.5, heightM=1.5, aim=dict(inM=.5, heightM=1.)),
    dict(id='V2', zh='右前 45° 站拍', en='front-right 45 deg, standing', anchor='E', az=45, distM=2.5, heightM=1.5, aim=dict(inM=.5, heightM=1.)),
    dict(id='V3', zh='左前 45° 蹲拍', en='front-left 45 deg, crouched', anchor='E', az=-45, distM=2., heightM=.5, aim=dict(inM=0., heightM=.5)),
    dict(id='V4', zh='右前 45° 蹲拍', en='front-right 45 deg, crouched', anchor='E', az=45, distM=2., heightM=.5, aim=dict(inM=0., heightM=.5)),
    dict(id='V5', zh='左侧面（进深）', en='left side (depth)', anchor='E', az=-80, distM=2.5, heightM=1.5, aim=dict(inM=0., heightM=1.)),
    dict(id='V6', zh='右侧面（进深）', en='right side (depth)', anchor='E', az=80, distM=2.5, heightM=1.5, aim=dict(inM=0., heightM=1.)),
    dict(id='V7', zh='尺度参照物近拍', en='scale reference close-up', anchor='S', az=0, distM=1.8, heightM=1.5),
    dict(id='V8', zh='入口上方仰拍（标识牌、信号灯）', en='above the entrance, tilted up (sign, lights)', anchor='E', az=0, distM=2.5, heightM=1.5,
         aim=dict(inM=0., heightM=2.6)),
    dict(id='V9', zh='左后角（可选）', en='rear-left corner (optional)', anchor='KL', az=-135, distM=1., heightM=1.5, optional=True),
    dict(id='V10', zh='右后角（可选）', en='rear-right corner (optional)', anchor='KR', az=135, distM=1., heightM=1.5, optional=True),
)
ANCHOR_ZH = dict(E=('入口中心', 'the entrance centre'), S=('尺度参照物', 'the scale reference'), KL=('防护区左后角', 'the rear-left corner of the guarded cell'),
                 KR=('防护区右后角', 'the rear-right corner of the guarded cell'))
WHY_ZH = dict(blocked='在模型上（站不下人）', keepOut='在防护区内', unreachable='从现有拍照位置走不到', noAnchor='这个报告没有它的参照（尺度参照物 / 防护区）',
              outsideBounds='超出建模范围', unobserved='现有照片没拍到这块地面')


def floor_frame(ctx):
    n, d = ctx['floor']; S = ctx['S']
    Z = np.array([c['R'][2] for c in ctx['cams']]); f = (Z - (Z @ n)[:, None] * n).sum(0); f /= np.linalg.norm(f); r = np.cross(f, n)
    O = np.mean([c['C'] for c in ctx['cams']], 0); O = O - (n @ O + d) * n
    to = lambda X: np.c_[(X - O) @ r, (X - O) @ f, X @ n + d] * S
    at = lambda x, y, h: O + (np.asarray(x, float)[..., None] * r + np.asarray(y, float)[..., None] * f + np.asarray(h, float)[..., None] * n) / S
    return dict(n=n, d=d, S=S, r=r, f=f, O=O, to=to, at=at)


def look(C, target, K, w, h, up):
    z = target - C; z = z / np.linalg.norm(z); x = np.cross(z, up); x /= np.linalg.norm(x); R = np.array([x, np.cross(z, x), z])
    return dict(K=np.asarray(K, float), R=R, t=-R @ C, C=C, w=w, h=h)


def rotate(v, axis, ang):
    return v * math.cos(ang) + np.cross(axis, v) * math.sin(ang) + axis * (axis @ v) * (1 - math.cos(ang))


def entity(ctx, prefix):
    hit = [o for o in ctx['objects'] if o['id'].startswith(prefix)]
    if len(hit) != 1:
        raise ValueError(f'entity prefix {prefix!r} matches {len(hit)} displayed models')
    return hit[0]


def scene(meshes):
    import open3d as o3d
    s = o3d.t.geometry.RaycastingScene()
    for V, F in meshes:  # geometry id = index (callers pass non-empty meshes)
        s.add_triangles(o3d.core.Tensor(np.ascontiguousarray(V, np.float32)), o3d.core.Tensor(np.ascontiguousarray(F, np.uint32)))
    return s


def joined(*meshes):
    meshes = [m for m in meshes if m is not None and len(m[1])]; off = np.cumsum([0] + [len(V) for V, _ in meshes])
    return np.vstack([V for V, _ in meshes]), np.vstack([F + o for (_, F), o in zip(meshes, off)])


def selector(spec):
    """Band / side from whole words of the part phrase (no substrings: 'bright' is not 'right'); CJK characters count singly."""
    sel = dict(spec.get('select') or {}); words = set(re.findall(r'[a-z]+|[一-鿿]', (spec.get('part') or '').lower()))
    if spec.get('part') and not sel:
        for key in ('bottom', 'top'):
            if words & WORDS[key]:
                sel['band'] = key
        for key in ('left', 'right'):
            if words & WORDS[key]:
                sel['side'] = key
        if not sel:
            raise ValueError(f"part {spec['part']!r} names no band or side: give 'select' (band, side, heightM or photo + pixel)")
    return sel


def subdivide(V, F, L):
    """4-way split of every triangle with an edge longer than L (primitives have full-height side triangles; parts are
    selected by triangle centre). T-junctions are harmless for ray casting."""
    while True:
        T = V[F]; big = np.linalg.norm(T - T[:, [1, 2, 0]], axis=2).max(1) > L
        if not big.any():
            return V, F
        T = T[big]; m = np.vstack([T[:, 0] + T[:, 1], T[:, 1] + T[:, 2], T[:, 2] + T[:, 0]]) / 2; k = big.sum(); b = len(V)
        a, c, e = b + np.arange(k), b + k + np.arange(k), b + 2 * k + np.arange(k); i, j, l = F[big].T
        V = np.vstack([V, m]); F = np.vstack([F[~big], np.c_[i, a, e], np.c_[a, j, c], np.c_[e, c, l], np.c_[a, c, e]])


def prism(fr, xy, h0, h1):
    """Closed vertical prism over the convex hull of floor points xy, from height h0 to h1 (native coordinates)."""
    from scipy.spatial import ConvexHull
    ring = xy[ConvexHull(xy).vertices]; k = len(ring); i = np.arange(k); j = (i + 1) % k
    V = np.vstack([fr['at'](ring[:, 0], ring[:, 1], np.full(k, h0)), fr['at'](ring[:, 0], ring[:, 1], np.full(k, h1))])
    F = np.vstack([np.c_[i, j, k + j], np.c_[i, k + j, k + i], np.c_[np.zeros(k - 2, int), i[2:], i[1:-1]], np.c_[np.full(k - 2, k), k + i[1:-1], k + i[2:]]])
    return V, F


def faces_of(P, N):
    """Faces of a target: groups of samples whose outward normals lie within FACE_DEG of each other (greedy, densest first).
    An edge is measured on a face that is long along it: groups under FACE_MIN_SHARE of the largest group's samples or
    under FACE_MIN_LEN of the longest group's length (e.g. a rail's end, a thin top strip) are no such face."""
    out = N * np.sign(np.einsum('ij,ij->i', P - P.mean(0), N) + 1e-12)[:, None]
    A = out @ out.T >= math.cos(math.radians(FACE_DEG)); left = np.ones(len(P), bool); groups = []
    while left.any():
        i = int(((A & left).sum(1) * left).argmax()); g = np.nonzero(A[i] & left)[0]; groups.append(g); left[g] = False
    length = [math.sqrt(12 * max(float(np.linalg.eigvalsh(np.cov(P[g].T))[-1]), 0.)) if len(g) > 3 else 0. for g in groups]
    big, longest = max(map(len, groups)), max(length)
    keep = [g for g, L in zip(groups, length) if len(g) >= FACE_MIN_SHARE * big and L >= FACE_MIN_LEN * longest]
    return keep, [out[g].mean(0) / np.linalg.norm(out[g].mean(0)) for g in keep]


def plate(ctx, fr, spec):
    """coveredBy: the floor-frame outward normal of the covered face (the side of the model footprint's minimum-area
    rectangle nearest the named direction) and the plate in front of it: a slab depthM thick, as wide as that side, from
    bottomM to the model's top."""
    cb = spec['coveredBy']; P = fr['to'](entity(ctx, spec['entityId'])['mesh'][0]); d = np.array(FACES[cb['face']], float)
    box = cv2.boxPoints(cv2.minAreaRect(P[:, :2].astype(np.float32))).astype(float); c = box.mean(0)
    sides = [(box[i], box[(i + 1) % 4]) for i in range(4)]; nrm = lambda a, b: ((a + b) / 2 - c) / np.linalg.norm((a + b) / 2 - c)
    a, b = max(sides, key=lambda s: nrm(*s) @ d); nu = nrm(a, b)
    return nu, prism(fr, np.array([a, b, b + nu * cb['depthM'], a + nu * cb['depthM']]), cb['bottomM'], float(P[:, 2].max()))


def clearance(fr, objs, C, X):
    """Placement check of a small object given by a pixel: is the sightline C -> X cut by a model, and how high does it
    pass above the models under it (vertical rays down from it every 1 cm)."""
    import open3d as o3d
    scn = scene([o['mesh'] for o in objs]); n = max(2, int(np.linalg.norm(X - C) * fr['S'] / .01))
    Q = C + np.linspace(0, 1, n)[:-1, None] * (X - C)
    r = scn.cast_rays(o3d.core.Tensor(np.hstack([Q, np.broadcast_to(-fr['n'], Q.shape)]).astype(np.float32)))
    t, g = r['t_hit'].numpy().astype(float) * fr['S'], r['geometry_ids'].numpy().astype(np.int64)
    cut = bool(wsc.first_hit(scn, C[None], X[None])[0] < 1 - EPS_M / fr['S'] / np.linalg.norm(X - C))
    if not np.isfinite(t).any():
        return dict(blocked=cut, clearanceM=None, over=None)
    i = int(np.argmin(t)); o = objs[g[i]]
    return dict(blocked=cut, clearanceM=round(float(t[i]), 3), over=o['id'][:8], overLabel=o['label'],
                atXYM=fr['to'](Q[i][None])[0, :2].round(2).tolist(), sightlineHeightM=round(float(fr['to'](Q[i][None])[0, 2]), 3))


def resolve(ctx, fr, spec, rng, plates):
    """Target samples (native), its faces, the scenes that decide facing (own) and blocking (others), display meshes."""
    obj = entity(ctx, spec['entityId']); V, F = obj['mesh']; S, n = fr['S'], fr['n']; sel = selector(spec); pl = plates.get(obj['id'])
    carrier, rest, extra, check = None, None, None, None
    if 'pixel' in sel:  # a small object seen at a pixel: a spec-size cylinder (axis = floor normal) on its carrier model (+ plate)
        cam = ctx['cams'][sel['photo'] - 1]; D = cam['R'].T @ np.linalg.inv(cam['K']) @ np.r_[sel['pixel'], 1.]
        dia, hgt = sel.get('sizeM') or (.1, .1); carrier = joined((V, F), pl[1] if pl else None)
        if sel.get('depthNative'):  # depth along the optical axis (D has unit camera z)
            X = cam['C'] + sel['depthNative'] * D
        else:
            t = wsc.first_hit(scene([carrier]), cam['C'][None], (cam['C'] + D)[None])[0]
            if not np.isfinite(t):
                raise ValueError('the pixel ray misses the entity model')
            X = cam['C'] + t * D - D / np.linalg.norm(D) * dia / 2 / S
        Vc, Fc = wsc.primitive_mesh(dict(kind='cylinder', radius=dia / 2, height=hgt, segments=24))
        part = (X + (Vc[:, :1] * fr['r'] + Vc[:, 1:2] * fr['f'] + Vc[:, 2:] * n) / S, Fc)
    else:
        V, F = subdivide(V, F, .05 / S) if sel else (V, F); P, Pv = fr['to'](V[F].mean(1)), fr['to'](V); keep = np.ones(len(F), bool)
        lo, hi = np.percentile(Pv[:, 2], [1, 99])
        if sel.get('band') == 'bottom':
            keep &= P[:, 2] <= lo + PART_BAND * (hi - lo)
        if sel.get('band') == 'top':
            keep &= P[:, 2] >= hi - PART_BAND * (hi - lo)
        if 'heightM' in sel:
            keep &= (P[:, 2] >= sel['heightM'][0]) & (P[:, 2] <= sel['heightM'][1])
        if sel.get('side'):
            xl, xh = np.percentile(Pv[:, 0], [1, 99]); w = sel.get('fraction', SIDE_FRAC) * (xh - xl)
            keep &= P[:, 0] >= xh - w if sel['side'] == 'right' else P[:, 0] <= xl + w
        if not keep.any():
            raise ValueError(f"part selection of {spec['entityId']} is empty")
        part = (V, F[keep]); rest = (V, F[~keep]) if (~keep).any() else None
    edge, he, role = spec.get('edge', 'surface'), spec.get('edgeHeightM'), spec.get('role', 'target')
    Ps, Ns = wsc.sample_surface(*part, 20000 if edge == 'bottom' else 4000, rng)
    if edge == 'bottom':
        h = fr['to'](Ps)[:, 2]; side = np.abs(Ns @ n) < .7; model_lo = float(np.percentile(h, 2))  # a bottom face is never seen
        if he is not None and he < model_lo - .01:  # the model ends above the measured edge: extend its lowest cross-section down
            band = side & (h >= model_lo - .01) & (h <= model_lo + EDGE_BAND_M)
            extra = prism(fr, fr['to'](Ps[band])[:, :2], he, model_lo + EDGE_BAND_M)
            Ps, Ns = wsc.sample_surface(*extra, 20000, rng); h = fr['to'](Ps)[:, 2]; side = np.abs(Ns @ n) < .7
        lo = model_lo if he is None else float(he)
        m = (h >= lo - .01) & (h <= lo + EDGE_BAND_M)
        Ps, Ns = (Ps[m & side], Ns[m & side]) if (m & side).sum() >= 50 else (Ps[m], Ns[m])
        if not len(Ps):
            raise ValueError(f"no surface of {spec['entityId']} at the edge height {lo:.2f} m")
    cover = None
    if spec.get('coveredBy'):  # the plate hides the covered face: keep the faces it does not cover
        nu = pl[0]; out = np.sign(np.einsum('ij,ij->i', Ps - Ps.mean(0), Ns))[:, None] * Ns; hx, hy = out @ fr['r'], out @ fr['f']
        covered = (hx * nu[0] + hy * nu[1]) / (np.hypot(hx, hy) + 1e-12) >= math.cos(math.radians(45))
        Ps, Ns = Ps[~covered], Ns[~covered]
        if not len(Ps):
            raise ValueError(f"coveredBy {spec['coveredBy']} leaves nothing of {spec['entityId']}")
        cover = dict(spec['coveredBy'], normalXY=nu.round(3).tolist(), n2=nu)
    pick = rng.choice(len(Ps), min(len(Ps), N_SAMPLES), replace=False); Ps, Ns = Ps[pick], Ns[pick]
    faces, normals = faces_of(Ps, Ns) if role != 'scale' else ([np.arange(len(Ps))], [None])
    skip = {obj['id']} | ({f"plate:{obj['id']}"} if 'pixel' in sel else set())  # a pixel object sits on its carrier and plate
    other_objs = [o for o in ctx['objects'] if o['id'] not in skip]
    others = [o['mesh'] for o in other_objs] + ([rest] if rest is not None else [])
    if 'pixel' in sel:
        check = dict(photo=sel['photo'], **clearance(fr, other_objs, ctx['cams'][sel['photo'] - 1]['C'], Ps.mean(0)))
    Pm = fr['to'](Ps); centre = Ps.mean(0)
    label = obj['label'] + (f" / {spec['part']}" if spec.get('part') else '')
    key = spec.get('key') or f"{obj['id'][:8]}:{(spec.get('part') or 'whole').replace(' ', '-')}:{edge}"
    return dict(key=key, entityId=obj['id'], label=label, name=spec.get('name') or label, nameEn=spec.get('nameEn') or label,
                edge=edge, role=role, obj=obj, P=Ps, N=Ns, faces=faces, faceNormals=normals, centre=centre, cm=fr['to'](centre[None])[0],
                heightM=float(np.median(Pm[:, 2])), extendedFromM=None if extra is None else round(model_lo, 3), cover=cover,
                needLow=edge == 'bottom' and float(np.median(Pm[:, 2])) < LOW_EDGE_M, placementCheck=check, provenance=sel.get('provenance') or spec.get('provenance'),
                edgeHeightSource=None if edge != 'bottom' else spec.get('edgeHeightSource') or ('given' if he is not None else 'model geometry (its lowest 3 cm)'),
                own=scene([part] + [m for m in (extra, carrier) if m is not None]), oth=scene(others),
                display=joined(part, extra), rest=rest if rest is not None else carrier, partRest=rest)


def floor_grid(ctx, fr, keep_out, rng):
    from scipy import ndimage
    from scipy.spatial import ConvexHull
    pts = np.vstack([fr['to'](o['mesh'][0]) for o in ctx['objects']] + [fr['to'](np.array([c['C'] for c in ctx['cams']]))])
    lo = pts[:, :2].min(0) - PAD_M; nx, ny = np.ceil((pts[:, :2].max(0) + PAD_M - lo) / GRID_M).astype(int)
    ij = lambda xy: np.floor((np.atleast_2d(xy) - lo) / GRID_M).astype(int)
    occ, owner = np.zeros((ny, nx), bool), np.full((ny, nx), -1)
    for k, o in enumerate(ctx['objects']):
        Q = np.vstack([fr['to'](wsc.sample_surface(*o['mesh'], 20000, rng)[0]), fr['to'](o['mesh'][0])])
        Q = Q[(Q[:, 2] >= BODY_BAND_M[0]) & (Q[:, 2] <= BODY_BAND_M[1])]
        if len(Q):
            one = np.zeros_like(occ); c = ij(Q[:, :2]); c = c[(c[:, 0] >= 0) & (c[:, 1] >= 0) & (c[:, 0] < nx) & (c[:, 1] < ny)]
            one[c[:, 1], c[:, 0]] = True; one = ndimage.binary_fill_holes(ndimage.binary_closing(one)); owner[one & ~occ] = k; occ |= one
    keep, hulls = np.zeros((ny, nx), np.uint8), []
    for group in keep_out:
        Q = np.vstack([fr['to'](entity(ctx, i)['mesh'][0])[:, :2] for i in group]); hull = Q[ConvexHull(Q).vertices]; hulls.append(hull)
        cv2.fillPoly(keep, [np.round((hull - lo) / GRID_M).astype(np.int32)], 1)
    rr = PERSON_R_M / GRID_M; k = int(math.ceil(rr)); yy, xx = np.ogrid[-k:k + 1, -k:k + 1]; disk = xx * xx + yy * yy <= rr * rr
    b_occ, b_keep = ndimage.binary_dilation(occ, disk), ndimage.binary_dilation(keep.astype(bool), disk)

    def reachable(free):
        label, _ = ndimage.label(free); seeds = set()
        for c in ij(fr['to'](np.array([c['C'] for c in ctx['cams']]))[:, :2]):
            win = label[max(0, c[1] - 10):c[1] + 11, max(0, c[0] - 10):c[0] + 11]; seeds |= set(np.unique(win[win > 0]).tolist())
        return np.isin(label, list(seeds))
    reach = reachable(~(b_occ | b_keep))

    def why(x, y):
        """None = a person can stand here."""
        i, j = ij([x, y])[0]
        if not (0 <= i < nx and 0 <= j < ny):
            return 'outsideBounds'
        if b_keep[j, i]:
            return 'keepOut'
        return 'blocked' if b_occ[j, i] else None if reach[j, i] else 'unreachable'
    kc = np.argwhere(keep)
    return dict(lo=lo, shape=(ny, nx), occ=occ, owner=owner, keep=keep.astype(bool), reach=reach, why=why, ij=ij, keepHulls=hulls,
                keepCentre=None if not len(kc) else lo + (kc[:, ::-1].mean(0) + .5) * GRID_M)


def floor_points(fr, grid, cm):
    g = np.arange(-FLOOR_R_M, FLOOR_R_M + 1e-9, FLOOR_STEP_M); X, Y = np.meshgrid(g, g); m = X ** 2 + Y ** 2 <= FLOOR_R_M ** 2
    xy = np.c_[X[m] + cm[0], Y[m] + cm[1]]; c = grid['ij'](xy); ny, nx = grid['shape']
    ok = (c[:, 0] >= 0) & (c[:, 1] >= 0) & (c[:, 0] < nx) & (c[:, 1] < ny); xy, c = xy[ok], c[ok]
    xy = xy[~grid['occ'][c[:, 1], c[:, 0]]]
    return fr['at'](xy[:, 0], xy[:, 1], np.zeros(len(xy)))


def unblocked(fr, scn, C, P):
    t = wsc.first_hit(scn, np.broadcast_to(C, P.shape), P); ok = t >= 1 - EPS_M / fr['S'] / np.linalg.norm(P - C, axis=1)
    hit = ~ok & np.isfinite(t)
    if hit.any():  # hits within 3 cm of the floor are floor (markings), not occluders
        ok[hit] = fr['to'](C + t[hit, None] * (P[hit] - C))[:, 2] < FLOOR_HIT_M
    return ok


def frame_margin(cam, P):
    """Per point: distance to the nearest image border as a fraction of the image size (negative outside, -1 behind)."""
    uv, z = wsc.project(cam, P)
    m = np.minimum.reduce([uv[:, 0] / cam['w'], 1 - uv[:, 0] / cam['w'], uv[:, 1] / cam['h'], 1 - uv[:, 1] / cam['h']])
    return np.where(z > 0, m, -1.)


def in_frame(cam, P, margin=RULE['frame']):
    return frame_margin(cam, P) >= margin


def view(env, T, cam):
    """How one camera sees one target, per face (raw values; thresholds are applied by judge / is_low). Mesh fences hide
    (they are models like any other). Top-level values are those of the best face."""
    fr = env['fr']; S = fr['S']; P, C = T['P'], cam['C']; fm = frame_margin(cam, P); inside = fm >= RULE['frame']
    eps = EPS_M / S / np.linalg.norm(P - C, axis=1)
    facing = wsc.first_hit(T['own'], np.broadcast_to(C, P.shape), P) >= 1 - eps
    clear = facing & unblocked(fr, T['oth'], C, P); ok, okp = inside & clear, clear & (fm >= PLAN['frame'])
    Pm, Cm = fr['to'](P), fr['to'](C[None])[0]; hd = np.linalg.norm(Pm[:, :2] - Cm[:2], axis=1)
    cos = np.abs((T['N'] * (C - P)).sum(1)) / np.linalg.norm(C - P, axis=1); dep = np.degrees(np.arctan2(Cm[2] - Pm[:, 2], hd))
    uv, z = wsc.project(cam, P); faces = {}
    for f, (idx, nf) in enumerate(zip(T['faces'], T['faceNormals'])):
        away = nf is not None and float(nf @ (C - P[idx].mean(0))) <= 0  # the face turns its back: corner samples within EPS do not make it seen
        fa, o, op = (facing[idx] & False, ok[idx] & False, okp[idx] & False) if away else (facing[idx], ok[idx], okp[idx])
        use = idx[o] if o.any() else idx[facing[idx]] if facing[idx].any() else idx
        d = float(np.linalg.norm(C - P[idx].mean(0)) * S); row = float(np.median(uv[use, 1])) / cam['h']
        rise = float(np.median(cam['K'][1, 1] / np.maximum(z[use] * S, 1e-6))) / cam['h']  # screen share per metre the phone rises (lens level)
        faces[f] = dict(facing=float(fa.mean()), seen=float(o.sum() / max(fa.sum(), 1)), seenPlan=float(op.sum() / max(fa.sum(), 1)),
                        faceOn=float(np.median(cos[use])), depressionDeg=float(np.median(dep[use])), distM=d, pxPerCm=float(cam['K'][0, 0] * .01 / d),
                        edgeScreenDown=row, screenRisePerM=rise, lowerThirdAtM=float(Cm[2] + (LOWER_THIRD - row) / max(rise, 1e-9)))
    best = max(faces, key=lambda f: (faces[f]['facing'] >= MIN_FACING, faces[f]['seen'] >= RULE['seen'], faces[f]['faceOn']))
    uvc, _ = wsc.project(cam, T['centre'][None]); to_c = C - T['centre']
    v = dict(faces[best], face=best, faces=faces, inFrame=float(inside.mean()), frameMargin=float(fm.min()), fullFrame=bool(fm.min() >= RULE['frame']),
             azDeg=math.degrees(math.atan2(Cm[1] - T['cm'][1], Cm[0] - T['cm'][0])), horizM=float(np.hypot(*(Cm[:2] - T['cm'][:2]))),
             depthM=float(cam['R'][2] @ (T['centre'] - C) * S), cameraHeightM=float(Cm[2]), fx=float(cam['K'][0, 0]),
             screenDown=float(uvc[0, 1] / cam['h']))
    if T['role'] == 'scale':  # elevation of the line of sight above or below the reference's horizontal mid-plane
        v['heightDiffM'] = abs(float(fr['n'] @ to_c)) * S; v['tiltDeg'] = math.degrees(math.asin(min(1., v['heightDiffM'] / (np.linalg.norm(to_c) * S))))
    if T['cover'] is not None:
        hz = (Cm - T['cm'])[:2]; v['offCoverDeg'] = math.degrees(math.acos(float(np.clip(hz @ T['cover']['n2'] / max(np.linalg.norm(hz), 1e-9), -1, 1))))
    if T['edge'] == 'bottom':
        F = T['floorPts']
        v['floorM2'] = float((in_frame(cam, F, 0) & unblocked(fr, env['all'], C, F)).sum() * FLOOR_STEP_M ** 2) if len(F) else 0.
    return v, facing.astype(int) + ok  # 0 = faces away, 1 = faces the camera but blocked / out of frame, 2 = seen


def judge(T, v, P):
    """(faces that count, first rule failed) for one view under thresholds P (RULE or PLAN). Seen, edge-on and px/cm are
    checked on one face's samples; the failure reported is the furthest any face got. The scale reference: whole in frame."""
    memo = v.setdefault('_judge', {})
    if P['name'] not in memo:
        if T['role'] == 'scale':
            f = ('notSeen' if v['facing'] < MIN_FACING or v[P['seenKey']] < P['seen'] else 'scaleNotWholeInFrame' if v['frameMargin'] < P['frame']
                 else 'scaleTooFar' if v['fx'] * .01 / (v['distM'] + P['padM']) < P['scalePxPerCm'] else 'scaleSteep' if v['tiltDeg'] > P['scaleTiltDeg'] else None)
            memo[P['name']] = ([] if f else [0], f)
        else:
            per = {k: 'notSeen' if s['facing'] < MIN_FACING or s[P['seenKey']] < P['seen'] else 'edgeOn' if s['faceOn'] < P['faceOn']
                   else 'tooFar' if v['fx'] * .01 / (s['distM'] + P['padM']) < P['pxPerCm'] else None for k, s in v['faces'].items()}
            ok = sorted(k for k, f in per.items() if f is None); f = None if ok else max(per.values(), key=FACE_ORDER.index)
            if ok and T['cover'] is not None and v['offCoverDeg'] < P['offCoverDeg']:
                ok, f = [], 'coveredFace'
            if ok and T['edge'] == 'bottom' and v['floorM2'] < P['floorM2']:
                ok, f = [], 'floor'
            memo[P['name']] = (ok, f)
    return memo[P['name']]


def fails(T, v, P):
    return judge(T, v, P)[1]


def is_low(T, v, f, P):
    """The low photo of a low edge on face f: camera 0.4-0.8 m and median depression <= lowDepDeg; PLAN (lens level) also:
    the edge's middle stays above the lower third line until the phone is lowHeadroomM above the planned height."""
    s = v['faces'][f]
    return (LOW_CAM_M[0] <= v['cameraHeightM'] <= LOW_CAM_M[1] and s['depressionDeg'] <= P['lowDepDeg']
            and s['lowerThirdAtM'] - v['cameraHeightM'] >= P['lowHeadroomM'])


def sep(a, b):
    return abs((a['azDeg'] - b['azDeg'] + 180) % 360 - 180)


def by_face(T, views, P):
    out = {}
    for v in views:  # each view by its own thresholds: PLAN for a new photo when P is PLAN, else RULE
        for f in judge(T, v, P if v['new'] else RULE)[0]:
            out.setdefault(f, []).append(v)
    return out


def face_missing(T, f, vs, P):
    out = [] if len(vs) >= 2 else [f'seen in {len(vs)} photo(s) of one face, need 2']
    if len(vs) >= 2 and not any(sep(a, b) >= (P if a['new'] or b['new'] else RULE)['sepDeg'] for a, b in combinations(vs, 2)):
        out.append(f"no two views of one face >= {P['sepDeg']:.0f} deg apart")
    if T['needLow'] and not any(is_low(T, v, f, P if v['new'] else RULE) for v in vs):
        out.append(f"no low photo of that face (camera {LOW_CAM_M[0]}-{LOW_CAM_M[1]} m, depression <= {P['lowDepDeg']:.0f} deg)")
    return out


def missing(T, views, P):
    """Rules T still fails with these views, on its best face (views carry new=True/False; an existing photo is judged by RULE)."""
    if T['role'] == 'scale':
        return [] if len(views) >= SCALE_VIEWS else [f'scale reference in {len(views)} photo(s), need {SCALE_VIEWS}']
    return min([face_missing(T, f, vs, P) for f, vs in by_face(T, views, P).items()] or [face_missing(T, None, [], P)], key=len)


def deficit(T, views, P):
    """Fewest further photos that could meet T's rules (on its best face)."""
    if T['role'] == 'scale':
        return max(0, SCALE_VIEWS - len(views))
    return min([2 - len(vs) if len(vs) < 2 else int(bool(face_missing(T, f, vs, P))) for f, vs in by_face(T, views, P).items()], default=2)


def plain(x):
    """Plain Python (the dispatcher's client unpickles results without numpy)."""
    if isinstance(x, dict):
        return {(k if isinstance(k, str) else plain(k)): plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [plain(v) for v in x]
    if isinstance(x, np.ndarray):
        return plain(x.tolist())
    return x.item() if isinstance(x, np.generic) else x


def rnd(x):
    if isinstance(x, dict):
        return {k: rnd(v) for k, v in x.items() if not str(k).startswith('_')}
    if isinstance(x, (list, tuple)):
        return [rnd(v) for v in x]
    return round(x, 3) if isinstance(x, float) else x


def rel(dx, dy, zh):
    if zh:
        p = ([f"往{'右' if dx > 0 else '左'} {abs(dx):.1f} m"] if abs(dx) >= .1 else []) + ([f"往{'里' if dy > 0 else '外'} {abs(dy):.1f} m"] if abs(dy) >= .1 else [])
        return '、'.join(p) or '原地'
    p = ([f"{abs(dx):.1f} m {'right' if dx > 0 else 'left'}"] if abs(dx) >= .1 else []) + ([f"{abs(dy):.1f} m {'inward' if dy > 0 else 'outward'}"] if abs(dy) >= .1 else [])
    return ' and '.join(p) or 'at'


def where(lms, x, y):
    """(zh, en) words for a floor point from the nearest fixed landmark (and from the reference landmark, the first)."""
    if not lms:
        return f'x {x:.1f} m, y {y:.1f} m', f'x {x:.1f} m, y {y:.1f} m'
    near = min(lms, key=lambda L: math.hypot(x - L['xy'][0], y - L['xy'][1])); ref = lms[0]
    d = math.hypot(x - near['xy'][0], y - near['xy'][1])
    zh = f"{near['zh']}{rel(x - near['xy'][0], y - near['xy'][1], True)}处（离它约 {d:.1f} m"
    en = f"{rel(x - near['xy'][0], y - near['xy'][1], False)} of the {near['en']} (about {d:.1f} m from it"
    if near is not ref:
        zh += f"；即{ref['zh']}{rel(x - ref['xy'][0], y - ref['xy'][1], True)}"; en += f"; i.e. {rel(x - ref['xy'][0], y - ref['xy'][1], False)} of the {ref['en']}"
    return zh + '）', en + ')'


def oname(env, o):
    nm = next((v for k, v in env['names'].items() if o['id'].startswith(k)), None)
    base = re.sub(r'[（(].*?[）)]', '', o['label']).strip()
    return tuple(nm) if nm else (base, base)


def route(env, xy):
    """(zh, en) walking words when the straight line from the nearest existing photo position to the spot crosses the
    guarded cell (go round the side the spot is on) or a model outside it (go round it); None when it is a straight walk."""
    grid = env['grid']; s = min(env['camXY'], key=lambda p: math.dist(p, xy)); ny, nx = grid['shape']
    c = grid['ij'](np.linspace(s, xy, max(2, int(math.dist(s, xy) / (GRID_M / 2)))))
    c = c[(c[:, 0] >= 0) & (c[:, 1] >= 0) & (c[:, 0] < nx) & (c[:, 1] < ny)]; zh, en = [], []
    if grid['keep'][c[:, 1], c[:, 0]].any():
        left = xy[0] < grid['keepCentre'][0]
        zh.append(f"沿防护区{'左' if left else '右'}侧外面走过去，不要穿过防护区"); en.append(f"walk round the {'left' if left else 'right'} side of the guarded cell, not through it")
    out = c[~grid['keep'][c[:, 1], c[:, 0]]]
    for k in dict.fromkeys(i for i in grid['owner'][out[:, 1], out[:, 0]].tolist() if i >= 0):
        z, e = oname(env, env['ctx']['objects'][k]); zh.append(f"绕开{z}"); en.append(f"go round the {e}")
    return ('，'.join(zh), '; '.join(en)) if zh else None


def phone_range(c, lows):
    """Stated phone height range: +-0.1 m, crouched not below 0.4 m and not above where a served low edge's middle would
    drop below the lower third line (rounded inwards to 5 cm)."""
    h = c['heightM']; lo, hi = h - JITTER['heightM'], h + JITTER['heightM']
    if h < 1:
        lo = max(lo, LOW_CAM_M[0]); hi = min([hi] + [c['views'][T['key']]['faces'][g]['lowerThirdAtM'] for T, g in lows])
    return [round(math.ceil(lo / .05 - 1e-6) * .05, 2), round(math.floor(hi / .05 + 1e-6) * .05, 2)]


def instruction(env, c, label_zh, label_en, extra=((), ())):
    """One worker instruction: where (fixed landmark), route, posture and phone height range, aim (c['A']: a target or an aim
    point with name / nameEn / centre) and straight-line distance, what the screen must show, warnings. Screen cues never put
    a low edge below the lower third line. extra: more screen cues."""
    fr, lms = env['fr'], env['lms']; zh_w, en_w = where(lms, *c['xyM']); low = c['heightM'] < 1
    served = [T for T in env['targets'] if T['key'] in c['counts']]; A = c['A']
    v = c['views'].get(A.get('key')) or dict(distM=float(np.linalg.norm(c['cam']['C'] - A['centre']) * fr['S']), edgeScreenDown=None)
    pitch = math.degrees(math.asin(float(np.clip(c['cam']['R'][2] @ fr['n'], -1, 1))))
    lows = []  # (target, face) a crouched photo is the low photo of: cues use that face
    for T in served:
        g = next((g for g in judge(T, c['views'][T['key']], PLAN)[0] if T['needLow'] and is_low(T, c['views'][T['key']], g, PLAN)), None)
        if g is not None:
            lows.append((T, g))
    lo, hi = c['phoneRange'] = phone_range(c, lows if low else [])
    if low:
        pz = f"蹲下，手机离地约 {c['heightM']:.1f} m（{lo:g}–{hi:g} m 都可以），镜头保持水平（打开网格线）"
        pe = f"crouch, phone {c['heightM']:.1f} m above the floor ({lo:g}-{hi:g} m), lens level (grid lines on)"
    else:
        tz = '镜头水平' if abs(pitch) < 5 else f"镜头{'上仰' if pitch > 0 else '下俯'}约 {abs(pitch):.0f}°"
        te = 'lens level' if abs(pitch) < 5 else f"lens tilted {'up' if pitch > 0 else 'down'} about {abs(pitch):.0f} deg"
        pz = f"站直，手机举到脸前（离地约 {c['heightM']:.1f} m，{lo:g}–{hi:g} m 都可以；胸口高度只有 1.3–1.4 m，偏低），{tz}"
        pe = f"stand, phone in front of your face (about {c['heightM']:.1f} m above the floor, {lo:g}-{hi:g} m; chest height, 1.3-1.4 m, is too low), {te}"
    main = [T for T in served if T['role'] != 'scale']
    nz, ne = '、'.join(T['name'] for T in main), ', '.join(T['nameEn'] for T in main)
    show_z, show_e = ([f"{nz}至少 4/5 在画面里、离画面四边留约 1/10 的空白，而且没被前面的东西挡住（能整个拍进去最好）"],
                      [f"at least 4/5 of the {ne} in the frame, about 1/10 of the frame from every border, and not hidden (whole is better)"]) if main else ([], [])
    if low:  # lens level: the aim is centred left-right; vertical cues only where the photo serves the target (never below the line)
        g = next((g for T, g in lows if T is A), None); row = v['faces'][g]['edgeScreenDown'] if g is not None else v['edgeScreenDown']
        dz, de = (f"、从上往下约 {row:.0%} 处", f", about {row:.0%} down the screen") if any(A is T for T in served) else ('', '')
        show_z.append(f"{A['name']}在屏幕左右正中{dz}"); show_e.append(f"the {A['nameEn']} centred left-right{de}")
        for T, g in lows:
            s = c['views'][T['key']]['faces'][g]; top = s['edgeScreenDown'] + (hi - c['heightM']) * s['screenRisePerM']
            show_z.append(f"{T['name']}的中段不低于屏幕下面那条三分线（这里约 {s['edgeScreenDown']:.0%}，手机在 {hi:g} m 时约 {top:.0%}）")
            show_e.append(f"the middle of the {T['nameEn']} not below the lower third line (here {s['edgeScreenDown']:.0%}, {top:.0%} with the phone at {hi:g} m)")
    else:
        show_z.append(f"{A['name']}在屏幕正中"); show_e.append(f"the {A['nameEn']} in the middle of the screen")
    show_z += list(extra[0]); show_e += list(extra[1])
    if any(T['edge'] == 'bottom' for T in served):
        show_z.append('下沿前面约 1 m 的地面也在画面里'); show_e.append('the floor about 1 m in front of the lower edge in the frame')
    for T in served:
        if T['cover'] is not None:
            pz_, pe_ = T['cover'].get('zh', '前护板'), T['cover'].get('en', 'front plate'); off = c['views'][T['key']]['offCoverDeg']
            show_z.append(f"{T['name']}要斜着拍：视线偏离{pz_}正面至少 45°（这里约 {off:.0f}°），拍到的是侧面那一段下沿，不是{pz_}的下沿")
            show_e.append(f"the {T['nameEn']} at a slant: at least 45 deg off the {pe_}'s normal (here about {off:.0f} deg), its side's lower edge, not the {pe_}'s")
        if T['role'] == 'scale':
            s = c['views'][T['key']]; dmax = s['fx'] * .01 / PLAN['scalePxPerCm'] - PLAN['padM']; k = math.sin(math.radians(PLAN['scaleTiltDeg']))
            show_z.append(f"{T['name']}整个在画面里、离四边至少 1/10；手机到它的直线距离不超过 {dmax:.1f} m（这里约 {s['distM']:.1f} m）；"
                          f"从侧面看它：手机和它的高差不超过这段直线距离的 {k:.2f} 倍（这里高差约 {s['heightDiffM']:.2f} m，视线倾斜约 {s['tiltDeg']:.0f}°，上限 {PLAN['scaleTiltDeg']:.0f}°）")
            show_e.append(f"the whole {T['nameEn']} in the frame, 1/10 from every border; phone at most {dmax:.1f} m from it in a straight line (here about "
                          f"{s['distM']:.1f} m); seen from the side: height difference at most {k:.2f} x that distance (here about {s['heightDiffM']:.2f} m, "
                          f"{s['tiltDeg']:.0f} deg, limit {PLAN['scaleTiltDeg']:.0f})")
    warn_z, warn_e = [], []
    st, dist = c['floor']
    if c.get('beyondModel'):
        warn_z.append('这里超出了报告的建模范围（报告里没有这块地面）：能不能站人未知，到现场确认')
        warn_e.append('beyond the modelled area (the report has no floor here): whether a person can stand here is unknown, check on site')
    elif st == 'seeThrough':
        warn_z.append(f"现有照片只隔着{env['seeThroughZh']}看到过这里的地面，先过去确认能站人、视线不被挡")
        warn_e.append(f"the existing photos see this floor only through the {env['seeThroughEn']}: check that it is free first")
    elif st == 'unobserved':
        warn_z.append(f"这里离现有照片拍到的地面约 {dist:.1f} m（照片里看不到）：能不能站人未知，到现场确认")
        warn_e.append(f"this spot is {dist:.1f} m from any floor the existing photos see: whether a person can stand here is unknown, check on site")
    rt = route(env, c['xyM'])
    zh = (f"{label_zh}：站在{zh_w}，" + (f"走法：{rt[0]}；" if rt else '') + f"{pz}，手机竖拿、1 倍，朝向{A['name']}（手机到它的直线距离约 {v['distM']:.1f} m）。"
          f"屏幕上：{'；'.join(show_z)}。" + (f"注意：{'；'.join(warn_z)}。" if warn_z else ''))
    en = (f"{label_en}: stand {en_w}; " + (f"route: {rt[1]}; " if rt else '') + f"{pe}; portrait, 1x; face the {A['nameEn']} (about {v['distM']:.1f} m from the phone in a "
          f"straight line). Screen: {'; '.join(show_e)}." + (f" Note: {'; '.join(warn_e)}." if warn_e else ''))
    return zh, en


def render(env, cam, T=None, width=360, skip=()):
    """CPU ray-cast preview: photo texture where an existing photo sees the point, else shaded (models blue)."""
    import open3d as o3d
    ctx, fr = env['ctx'], env['fr']
    s = width / cam['w']; K = cam['K'] * [[s], [s], [1]]; K[0, 2] = (cam['K'][0, 2] + .5) * s - .5; K[1, 2] = (cam['K'][1, 2] + .5) * s - .5
    W, H = width, int(round(cam['h'] * s)); small = dict(cam, K=K, w=W, h=H)
    meshes = [o['mesh'] for o in ctx['objects'] if T is None or o is not T['obj']]
    if T is not None:
        meshes += ([T['rest']] if T['rest'] is not None and len(T['rest'][1]) else []) + [T['display']]
    scn = scene(meshes); ys, xs = np.mgrid[0:H, 0:W]
    D = (np.c_[xs.ravel(), ys.ravel(), np.ones(xs.size)] @ np.linalg.inv(K).T) @ cam['R']; C = cam['C']
    r = scn.cast_rays(o3d.core.Tensor(np.hstack([np.broadcast_to(C, D.shape), D]).astype(np.float32)))
    t, g, Nn = r['t_hit'].numpy().astype(float), r['geometry_ids'].numpy().astype(np.int64), r['primitive_normals'].numpy()
    n, d = fr['n'], fr['d']
    with np.errstate(divide='ignore', invalid='ignore'):
        tf = -(n @ C + d) / (D @ n)
    floor = (tf > 0) & (tf < t); t = np.where(floor, tf, t); hit = np.isfinite(t); X = C + np.where(hit, t, 0)[:, None] * D
    cosv = np.abs((Nn * D).sum(1)) / np.linalg.norm(D, axis=1)
    tex, best = np.full(len(D), np.nan), np.full(len(D), np.inf); idx = np.nonzero(hit)[0]
    for k, c in enumerate(ctx['cams']):
        if k in skip:
            continue
        inside = in_frame(c, X[idx], 0); j = idx[inside]
        if len(j):
            vis = unblocked(fr, scn, c['C'], X[j]); j = j[vis]; dist = np.linalg.norm(X[j] - c['C'], axis=1); closer = dist < best[j]; j = j[closer]
            uv, _ = wsc.project(c, X[j]); tex[j] = wsc.bilinear(np.asarray(ctx['gray'][k], np.float32), uv); best[j] = dist[closer]
    img = np.full((len(D), 3), 35.)
    model = hit & ~floor; img[floor] = 70; img[model] = (60 + 140 * cosv[model])[:, None] * [1., .8, .55]  # BGR: blue = unseen
    has = ~np.isnan(tex); img[has] = tex[has, None]
    if T is not None:
        tg = model & (g == len(meshes) - 1); img[tg] = .5 * img[tg] + .5 * np.array([0., 140., 255.])
    img = np.clip(img, 0, 255).astype(np.uint8).reshape(H, W, 3)
    for frac in (1 / 3, 2 / 3):  # the phone's grid lines: a low edge must not be below the lower one
        cv2.line(img, (0, int(H * frac)), (W - 1, int(H * frac)), (200, 200, 200), 1)
    if T is not None:
        _, state = view(env, T, cam); uv, z = wsc.project(small, T['P'])
        for (u, v), st, zz in zip(uv, state, z):
            if st and zz > 0 and 0 <= u < W and 0 <= v < H:
                cv2.circle(img, (int(u), int(v)), 1, (0, 0, 255) if st == 2 else (255, 0, 255), -1)
    return img


def topdown(env, chosen, title, std=()):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fr, grid, ctx, targets, lms = env['fr'], env['grid'], env['ctx'], env['targets'], env['lms']
    ny, nx = grid['shape']; lo = grid['lo']; ext = [lo[0], lo[0] + nx * GRID_M, lo[1], lo[1] + ny * GRID_M]
    img = np.ones((ny, nx, 3)); img[grid['unobs'] > UNOBSERVED_M] = (1, 1, .75); img[(grid['unobs'] > UNOBSERVED_M) & (grid['unobsAny'] <= UNOBSERVED_M)] = (.8, .9, 1)
    img[~grid['reach']] = .88; img[grid['keep']] = (1, .82, .82); img[grid['occ']] = .35
    fig, ax = plt.subplots(figsize=(7, 8.5), dpi=110); ax.imshow(img, origin='lower', extent=ext, interpolation='nearest')
    for U in env['unmodelled']:
        q = np.array(U['footprintM'] + U['footprintM'][:1]); ax.plot(q[:, 0], q[:, 1], ':', color='purple', lw=1.5)
        ax.annotate(U['en'], q.mean(0), fontsize=6.5, color='purple', ha='center')
    for L in lms:
        ax.annotate(L['en'], L['xy'], fontsize=6.5, color='0.25', ha='center', va='top', xytext=(0, -4), textcoords='offset points')
    for k, c in enumerate(ctx['cams']):
        p = fr['to'](c['C'][None])[0]; z = c['R'][2]; dx, dy = z @ fr['r'], z @ fr['f']; q = math.hypot(dx, dy)
        ax.plot(*p[:2], 'b^', ms=8); ax.arrow(p[0], p[1], .5 * dx / q, .5 * dy / q, color='b', width=.01, head_width=.08)
        ax.annotate(f'photo {k + 1} ({p[2]:.2f} m)', p[:2], fontsize=7, color='b', xytext=(4, -10), textcoords='offset points')
    for t, T in enumerate(targets, 1):
        ax.plot(*T['cm'][:2], 'rx', ms=7, mew=2)
        ax.annotate(f'T{t}', T['cm'][:2], fontsize=8, color='r', weight='bold', xytext=(5, 2 + 7 * (t % 2)), textcoords='offset points')
    for i, c in enumerate(chosen, 1):
        aim = c['A']['cm']; col = 'darkorange' if c['heightM'] < 1 else 'green'
        ax.plot([c['xyM'][0], aim[0]], [c['xyM'][1], aim[1]], '--', color=col, lw=1)
        ax.plot(*c['xyM'], 'o', color=col, ms=14, mec=col, mew=2.5); ax.annotate(str(i), c['xyM'], color='w', ha='center', va='center', fontsize=9, weight='bold')
        for a in c.get('alternatives') or []:
            ax.plot(*a['xyM'], 'o', mfc='none', mec=col, ms=11); ax.annotate(f'{i}', a['xyM'], color=col, ha='center', va='center', fontsize=7)
    for v in std:  # every standard viewpoint: solid = needed, dashed edge = standability unknown, grey = not needed / not standable
        if 'xyM' not in v:
            continue
        col = 'purple' if v.get('needed') else '0.55'
        if 'A' in v:
            ax.plot([v['xyM'][0], v['A']['cm'][0]], [v['xyM'][1], v['A']['cm'][1]], '-.', color=col, lw=.7, alpha=.6)
        ax.plot(*v['xyM'], 's', color=col, ms=13, mfc='w' if v['standable'] != 'unknown' else (1, .93, .7), mew=2)
        ax.annotate(v['id'] + ('' if v['standable'] != 'no' else ' x'), v['xyM'], color=col, ha='center', va='center', fontsize=6, weight='bold')
    ax.plot([], [], 's', color='purple', mfc='w', mew=2, label='standard viewpoint, needed here')
    ax.plot([], [], 's', color='purple', mfc=(1, .93, .7), mew=2, label='standard viewpoint, standability unknown (check on site)')
    ax.plot([], [], 's', color='0.55', mfc='w', mew=2, label='standard viewpoint, not needed / not standable (x)')
    ax.plot([], [], 'o', color='darkorange', label='extra, crouched (0.4-0.6 m, lens level)'); ax.plot([], [], 'o', color='green', label='extra, standing (1.5 m)')
    ax.plot([], [], 'b^', label='existing photo'); ax.plot([], [], 'rx', label='target')
    for col, lab in (((1, 1, .75), f'floor > {UNOBSERVED_M} m from any floor seen in a photo'), ((.8, .9, 1), 'floor seen only through a see-through model'),
                     ((1, .82, .82), 'keep-out (guarded cell)'), ((.88, .88, .88), 'not reachable / person does not fit'), ((.35, .35, .35), 'model footprint')):
        ax.fill([], [], color=col, label=lab)
    ax.plot([], [], 'o', mfc='none', mec='0.3', label='fallback for the same number')
    ax.plot([], [], ':', color='purple', label='known, not in the report (an occluder)')
    ax.legend(loc='upper left', bbox_to_anchor=(0, -.07), fontsize=7, ncol=2); ax.set_aspect('equal'); ax.set_title(title, fontsize=9)
    ax.text(0, -.29, '\n'.join(f"T{t} {T['nameEn']} ({T['edge']}, {T['heightM']:.2f} m)" for t, T in enumerate(targets, 1)),
            transform=ax.transAxes, fontsize=7, color='r', va='top')
    ax.set_xlabel('x (m): right, as seen from the existing photos'); ax.set_ylabel('y (m): into the cell')
    buf = io.BytesIO(); fig.savefig(buf, format='png', bbox_inches='tight'); plt.close(fig)
    return buf.getvalue()


def report_scale_reference(ctx):
    refs = (((ctx.get('doc') or {}).get('coordinateFrames') or [{}])[0].get('scale') or {}).get('sourceRefs') or []
    ent = next((r['entityId'] for r in refs if r.get('entityId')), None)
    if not ent:
        return None
    label = entity(ctx, ent)['label']
    return dict(entityId=ent, name=f'尺度参照物（本工位为{label}）', nameEn=f'scale reference ({label})')


def convention(env, rng):
    """Camera-convention test against independent data: every model with report masks, sampled where it faces the camera,
    projected with the convention used here and with wrong ones; the share of projections inside the object's mask."""
    ctx = env['ctx']; alts = dict(used=lambda c: c, transposedR=lambda c: dict(c, R=c['R'].T, t=-c['R'].T @ c['C']),
                                  flippedV=lambda c: dict(c, flip='v'), mirroredU=lambda c: dict(c, flip='u'))
    rows = []
    for o in ctx['objects']:
        if not o['masks']:
            continue
        P = wsc.sample_surface(*o['mesh'], 1500, rng)[0]; own = scene([o['mesh']]); row = dict(entityId=o['id'][:8], label=o['label'], photos={})
        for k, m in o['masks'].items():
            c0 = ctx['cams'][k]; m = np.asarray(m)
            m = (np.unpackbits(m)[:c0['h'] * c0['w']].reshape(c0['h'], c0['w']) if m.ndim == 1 else m).astype(bool)  # 1-D: a packed local cache
            vis = wsc.first_hit(own, np.broadcast_to(c0['C'], P.shape), P) >= 1 - EPS_M / env['fr']['S'] / np.linalg.norm(P - c0['C'], axis=1)
            if vis.sum() < 20:
                continue
            res = {}
            for name, f in alts.items():
                c = f(c0); uv, z = wsc.project(c, P[vis])
                if c.get('flip') == 'v':
                    uv[:, 1] = c['h'] - 1 - uv[:, 1]
                if c.get('flip') == 'u':
                    uv[:, 0] = c['w'] - 1 - uv[:, 0]
                u, v = np.round(uv).astype(int).T; inside = (z > 0) & (u >= 0) & (v >= 0) & (u < c['w']) & (v < c['h'])
                res[name] = round(float(m[v[inside], u[inside]].sum() / len(u)), 3)
            row['photos'][str(k + 1)] = res
        if row['photos']:
            rows.append(row)
    med = {name: round(float(np.median([r[name] for row in rows for r in row['photos'].values()])), 3) if rows else None for name in alts}
    best_alt = max((med[a] for a in alts if a != 'used'), default=0) or 0
    return dict(objects=rows, medianInsideMask=med, passed=bool(rows and med['used'] >= .7 and med['used'] >= best_alt + .3))


def load_boxes(env, doc):
    """SHARED FORMAT boxes of this report (the boxes file, or its 'boxes' map) as corners, face normals and centres (native).
    Refused when their scale, floor or revision is not this report's: nativeToMeters is never changed here."""
    fr, ctx = env['fr'], env['ctx']; S = fr['S']; doc = doc if 'boxes' in doc else dict(boxes=doc)
    if doc.get('nativeToMeters') is not None and abs(doc['nativeToMeters'] / S - 1) > 1e-6:
        raise ValueError(f"boxes nativeToMeters {doc['nativeToMeters']} is not the report's {S}")
    if doc.get('floor') and (float(np.asarray(doc['floor']['normal'], float) @ fr['n']) < math.cos(math.radians(.5)) or abs(doc['floor']['offset'] - fr['d']) * S > .01):
        raise ValueError("the boxes were measured on another floor than the report's")
    rev = (ctx.get('layer') or {}).get('revisionId')
    if doc.get('revisionId') and rev and doc['revisionId'] != rev:
        raise ValueError(f"boxes of revision {doc['revisionId']}, report revision {rev}")
    idx = {o['id']: k for k, o in enumerate(ctx['objects'])}; out = []
    for eid, b in doc['boxes'].items():
        k = idx.get(eid, next((i for i, o in enumerate(ctx['objects']) if o['id'].startswith(eid)), -1))
        c = np.asarray(b['centerNative'], float); ax = np.array(b['axes'], float); h = np.asarray(b['sizeM'], float) / 2 / S
        X = np.array([c + (2 * (i >> 2 & 1) - 1) * h[0] * ax[0] + (2 * (i >> 1 & 1) - 1) * h[1] * ax[1] + (2 * (i & 1) - 1) * h[2] * ax[2] for i in range(8)])
        N = np.array([-ax[1], ax[1], -ax[0], ax[0], -ax[2], ax[2]]); Pc = c + N * h[list(FACE_AXIS)][:, None]
        text = ' '.join(list(b.get('notes') or []) + list(b.get('highlightReasons') or []) + [str(f.get('need') or '') for f in b['faces'].values()])
        m = re.search(r'(?:mean mismatch|掩码与盒子平均错位|掩码与盒子不符（)\s*(\d+(?:\.\d+)?) px', text)  # v1 notes / v2 / v3 highlightReasons
        gc = b.get('gradeCaps') or {}  # box_faces' research caps (make_opts.py adds them): capped, faces at most low, unpinned faces
        cap = ('capped' if gc.get('capped') or '掩码与盒子不符' in text else 'rejected' if b.get('fitRejected') or '掩码对不上一个盒子' in text else
               'residual' if 'disagree with any box' in text or m else None)
        out.append(dict(id=eid, obj=k, plate=idx.get(f'plate:{eid}', -1), label=b['label'], X=X, N=N, Pc=Pc, ax=ax, h=h, rec=b, cap=cap,
                        resid=b.get('fitResidualPx') or (float(m.group(1)) if m else None), before=[CONF.index(b['faces'][F]['confidence']) for F in FACES6],
                        low={FACES6.index(F) for F in gc.get('low') or []}, unpinned={FACES6.index(F) for F in gc.get('unpinned') or []}))
    return out


def evidence(env, cam, detail=False):
    """R1 evidence of every box face from one camera, as box_faces measures it (box_faces.outline_px): len = known outline px,
    s = px/cm of the face's best counting outline edge (NaN: none), ok = s >= 10 (planned photos: replaced by the Monte Carlo
    result), hid = occluder counts per face."""
    import open3d as o3d
    B, fr = env['boxes'], env['fr']; S, C = fr['S'], cam['C']; nb = len(B)
    L, Sv, segs = np.zeros((nb, 6)), np.full((nb, 6), np.nan), []
    for k, b in enumerate(B):
        uv, z = wsc.project(cam, b['X'])
        if (z <= 1e-3).any():  # a corner behind the camera: no outline (as box_faces)
            continue
        idx = cv2.convexHull(uv.astype(np.float32), returnPoints=False).ravel(); poly = uv[idx]; q2 = np.roll(poly, -1, 0)
        ccw = np.sign(float((poly[:, 0] * q2[:, 1] - q2[:, 0] * poly[:, 1]).sum())); sd = sides(poly)
        for e in range(len(idx)):
            i, j = int(idx[e]), int(idx[(e + 1) % len(idx)]); d = i ^ j; p, q = uv[i], uv[j]; ln = float(np.linalg.norm(q - p))
            if d & (d - 1) or ln < .5:  # not a box edge (looking straight along one) or too short
                continue
            ns = int(min(EVID_MAX_N, max(1, round(ln / EVID_STEP)))); s = (np.arange(ns) + .5)[:, None] / ns
            X3 = b['X'][i] + s * (b['X'][j] - b['X'][i]); px = p + s * (q - p); t = (q - p) / ln; nu = ccw * np.array([t[1], -t[0]])
            fs = [FACE_OF[(bit, i >> bit & 1)] for bit in range(3) if not d >> bit & 1]
            segs.append((k, fs, X3, px, px + 3 * nu, ln / ns, sd[e]))
    out = dict(len=L, s=Sv, C=C, ok=np.zeros((nb, 6), bool), hid={})
    if not segs:
        return out
    X3, px, po = (np.vstack([sg[i] for sg in segs]) for i in (2, 3, 4)); n = len(X3)
    kk = np.concatenate([np.full(len(sg[2]), sg[0]) for sg in segs])
    inside = lambda u: (u[:, 0] >= 0) & (u[:, 1] >= 0) & (u[:, 0] < cam['w']) & (u[:, 1] < cam['h'])
    infr = inside(px); fin = infr & inside(po)
    Dp = X3 - C; r = np.linalg.norm(Dp, axis=1); Dp = Dp / r[:, None]
    Do = (np.c_[po, np.ones(n)] @ np.linalg.inv(cam['K']).T) @ cam['R']; Do = Do / np.linalg.norm(Do, axis=1)[:, None]
    D = np.vstack([Dp, Do]); res = env['all'].cast_rays(o3d.core.Tensor(np.hstack([np.broadcast_to(C, D.shape), D]).astype(np.float32)))
    t, g = res['t_hit'].numpy().astype(float), res['geometry_ids'].numpy().astype(np.int64); rr = np.r_[r, r]
    own = np.array([B[k]['obj'] for k in kk]); pl = np.array([B[k]['plate'] for k in kk]); own, pl = np.r_[own, own], np.r_[pl, pl]
    hit = np.isfinite(t) & (t < rr - (OCC_MARGIN_M / S + OCC_MARGIN_REL * rr)) & (g != own) & (g != pl)
    if hit.any():  # floor markings never hide
        hh = np.nonzero(hit)[0]; hit[hh] = fr['to'](C + t[hh, None] * D[hh])[:, 2] >= FLOOR_HIT_M
    blk = hit[:n] | hit[n:]; known = fin & ~blk; occ = np.where(hit[:n], g[:n], g[n:])
    member = np.zeros((n, 6), bool); side = np.concatenate([np.full(len(sg[2]), sg[6]) for sg in segs])
    ln = np.concatenate([np.full(len(sg[2]), sg[5]) for sg in segs]); pos = 0
    for k, fs, Xs, *_ in segs:
        m = len(Xs); member[pos:pos + m, fs] = True; hd = occ[pos:pos + m][fin[pos:pos + m] & blk[pos:pos + m]]
        if detail and len(hd):
            for f in fs:
                out['hid'].setdefault((k, f), Counter()).update(hd.tolist())
        pos += m
    for k in np.unique(kk):
        sel = kk == k; L[k] = [ln[sel & known & member[:, f]].sum() for f in range(6)]
        b = B[k]; top = float((b['X'] @ fr['n']).max() + fr['d']); cam_h = float(C @ fr['n'] + fr['d'])
        for f, v in outline_px(side[sel], member[sel], infr[sel], known[sel], ln[sel], r[sel] * S, cam['K'][0, 0])[0].items():
            if f in DIMS['W'] and not depth_credit(fr['n'], b['ax'][1], C, b['Pc'][f], top, cam_h):
                continue  # R1, depth: side views or from above only (box_faces.depth_credit)
            Sv[k, f] = v
    out['ok'] = np.nan_to_num(Sv) >= HIGH_PX_CM
    return out


def face_mc(env, c, rng):
    """Monte Carlo of a planned photo's face evidence (the targets' vet jitter): ok = >= 10 px/cm in >= MIN_PASS of the trials."""
    fe = c['fe']; fe['ok0'] = fe['ok'].copy(); rate = np.zeros(fe['ok'].shape)
    for cam in jittered(env, c, rng, JITTER['trials']):
        rate += evidence(env, cam)['ok'] & fe['ok0']
    fe['mcRate'] = rate / JITTER['trials']; fe['ok'] = fe['ok0'] & (fe['mcRate'] >= MIN_PASS); c['faceMc'] = True
    return c


def entry(fe, k, fi, label):
    """One view's evidence on face fi of box k: (px/cm, camera centre, label); below 10 px/cm unless it holds (ok); None without."""
    if np.isnan(fe['s'][k, fi]):
        return None
    s = float(fe['s'][k, fi]); return (s if fe['ok'][k, fi] else min(s, HIGH_PX_CM - 1e-3), fe['C'], label)


def level(env, b, fi, ev):
    """box_faces' grade (index into CONF) of face fi from evidence [(px/cm, camera centre, label)] (box_faces.face_grade)."""
    n, P = env['fr']['n'], b['Pc'][fi]
    return CONF.index(face_grade({i: e[0] for i, e in enumerate(ev)}, {i: hdir(n, e[1] - P) for i, e in enumerate(ev)})[0])


def regrade(env, b, new=None):
    """(face grades, {dim: grade}) as box_faces grades them (R1, its caps), from the file's per-photo evidence ev0 (+ planned
    evidence new {face: [entries]}). Caps: a capped box keeps its grades (photos do not fix the masks); faces at most low (b['low'])
    and unpinned faces (b['unpinned'], their dimensions at most low) stay so unless a planned photo credits them (>= 10 px/cm)."""
    if b['cap'] and new:
        return regrade(env, b)
    new = new or {}; lift = {fi for fi, es in new.items() if any(e[0] >= HIGH_PX_CM for e in es)}
    fg = [level(env, b, fi, b['ev0'][fi] + new.get(fi, [])) for fi in range(6)]
    fg = [min(g, 1) if fi in b['low'] - lift else g for fi, g in enumerate(fg)]
    dg = {d: CONF.index(dim_grade(*(CONF[fg[i]] for i in fs))) for d, fs in DIMS.items()}
    return fg, {d: min(g, 1) if set(DIMS[d]) & (b['unpinned'] - lift) else g for d, g in dg.items()}


def after_level(env, b, fi, new):
    """Grade after planned evidence `new`: with the file's per-photo evidence, box_faces' grade of all of it (regrade), never
    below the file's; a capped box stays. Without it (ev0 None): the grade of the new evidence alone, and a medium face becomes
    high only by a new photo >= 10 px/cm that is >= 30 deg from every existing photo (any of them may be its one good photo)."""
    bf = b['before'][fi]
    if b['cap'] or not new or bf == 3:
        return bf
    if b['ev0'] is not None:
        return max(bf, regrade(env, b, {fi: new})[0][fi])
    lv = level(env, b, fi, new)
    if bf == 2 and lv == 2:
        n, P = env['fr']['n'], b['Pc'][fi]; hz = lambda C: hdir(n, C - P)
        lv += any(e[0] >= HIGH_PX_CM and all(hz(e[1]) @ hz(c['C']) <= math.cos(math.radians(HIGH_DEG)) + 1e-12 for c in env['ctx']['cams']) for e in new)
    return max(bf, lv)


def v_anchor(env, F, s):
    """(anchor floor point, aim) of a standard viewpoint in this cell, (None, None) when the cell lacks the anchor."""
    fr = env['fr']
    if s['anchor'] == 'E':
        a = s['aim']; xy = F['E'] - a['inM'] * F['out']; nz, ne = ANCHOR_ZH['E']
        return F['E'], dict(key=None, name=f"{nz}往里 {a['inM']:.1f} m、离地 {a['heightM']:.1f} m 处" if a['inM'] else f"{nz}离地 {a['heightM']:.1f} m 处",
                            nameEn=f"point {a['inM']:.1f} m in from {ne}, {a['heightM']:.1f} m up", centre=fr['at'](*xy, a['heightM']), cm=np.r_[xy, a['heightM']])
    if s['anchor'] == 'S':
        T = next((T for T in env['targets'] if T['role'] == 'scale'), None)
        return (None, None) if T is None else (T['cm'][:2], T)
    if not env['grid']['keepHulls']:
        return None, None
    M = np.c_[F['x'], F['out']]; q = (np.vstack(env['grid']['keepHulls']) - F['E']) @ M; lat = q[:, 0].min() if s['anchor'] == 'KL' else q[:, 0].max()
    mid = F['E'] + (q[:, 0].min() + q[:, 0].max()) / 2 * F['x'] + (q[:, 1].min() + q[:, 1].max()) / 2 * F['out']
    return F['E'] + lat * F['x'] + q[:, 1].min() * F['out'], dict(key=None, name='防护区中间离地 1 m 处', nameEn='the middle of the guarded cell, 1 m up',
                                                                    centre=fr['at'](*mid, 1.), cm=np.r_[mid, 1.])


def v_words(s):
    """(zh, en) generic definition of a standard viewpoint (for the protocol)."""
    az, an = s['az'], ANCHOR_ZH[s['anchor']]; lat, out = s['distM'] * math.sin(math.radians(az)), s['distM'] * math.cos(math.radians(az))
    pz = ([f"往{'右' if lat > 0 else '左'} {abs(lat):.1f} m"] if abs(lat) >= .05 else []) + ([f"往{'外' if out > 0 else '里'} {abs(out):.1f} m"] if abs(out) >= .05 else [])
    pe = ([f"{abs(lat):.1f} m {'right' if lat > 0 else 'left'}"] if abs(lat) >= .05 else []) + ([f"{abs(out):.1f} m {'out' if out > 0 else 'in'}"] if abs(out) >= .05 else [])
    dz = f"{an[0]}{'、'.join(pz)}" + (f"（偏离正前方{'左' if az < 0 else '右'} {abs(az)}°、水平 {s['distM']:.1f} m）" if az else '')
    de = f"{' and '.join(pe)} of {an[1]}" + (f" ({abs(az)} deg {'left' if az < 0 else 'right'} of straight out, {s['distM']:.1f} m)" if az else '')
    a = s.get('aim')
    if s['heightM'] < 1:
        pz, pe = f"蹲下，手机离地 {s['heightM']:.1f} m，镜头水平，朝向{an[0]}", f"crouch, phone {s['heightM']:.1f} m, lens level, facing {an[1]}"
    elif s['anchor'] == 'S':
        pz, pe = '站直，手机离地约 1.5 m，对准尺度参照物（急停）', 'stand, phone about 1.5 m, aimed at the scale reference (e-stop)'
    elif s['anchor'] in ('KL', 'KR'):
        pz, pe = '站直，手机离地约 1.5 m，对准防护区中间离地 1 m 处', 'stand, phone about 1.5 m, aimed at the middle of the guarded cell 1 m up'
    else:
        pz = f"站直，手机离地约 1.5 m，对准入口中心{'往里 %.1f m、' % a['inM'] if a['inM'] else ''}离地 {a['heightM']:.1f} m 处"
        pe = f"stand, phone about 1.5 m, aimed {'%.1f m in from ' % a['inM'] if a['inM'] else 'at '}the entrance centre, {a['heightM']:.1f} m up"
    return f"{dz}；{pz}", f"{de}; {pe}"


def standard_view(env, F, s, rng):
    """One standard viewpoint in this cell: standability (yes / unknown / no), the targets it is relied on for (vet) and its
    face evidence (Monte Carlo); 'no' entries are not evaluated."""
    out = dict(id=s['id'], spec=s, words=v_words(s)); anchor, A = v_anchor(env, F, s)
    if anchor is None:
        return dict(out, standable='no', standWhy='noAnchor')
    a = math.radians(s['az']); xy = anchor + s['distM'] * (math.sin(a) * F['x'] + math.cos(a) * F['out']); why = env['grid']['why'](*xy)
    out['xyM'] = np.round(xy, 2).tolist()
    if why not in (None, 'outsideBounds'):
        return dict(out, standable='no', standWhy=why)
    c = candidate(env, A, float(xy[0]), float(xy[1]), s['heightM']); c.update(out)
    if why:
        c.update(beyondModel=True, floor=('beyond', None))
    c['standable'] = 'unknown' if why or c['floor'][0] == 'unobserved' else 'yes'; c['standWhy'] = why or (c['floor'][0] if c['standable'] == 'unknown' else None)
    c['fe'] = evidence(env, c['cam'], True); vet(env, c, rng); face_mc(env, c, rng)
    return c


def served_words(env, c, need):
    """Screen cues: the objects whose faces in `need` this photo is relied on for (>= 10 px/cm in >= 95 % of the trials)."""
    zh, en = [], []
    for k, b in enumerate(env['boxes']):
        fs = [F for fi, F in enumerate(FACES6) if (k, fi) in need and c['fe']['ok'][k, fi]]
        if fs:
            z, e = oname(env, env['ctx']['objects'][b['obj']]) if b['obj'] >= 0 else (b['label'], b['label'])
            zh.append(f"{z}（{'、'.join(FACE_ZH[F] for F in fs)}）"); en.append(f"{e} ({', '.join(fs)})")
    if not zh:
        return (), ()
    return ([f"{'、'.join(zh)}的轮廓边都在画面里、没被挡（括号里是这一张要定的面）"],
            [f"the outlines of {'; '.join(en)} in the frame and not hidden (the faces in brackets are what this photo measures)"])


def box_aim(env, k):
    b = env['boxes'][k]; z, e = oname(env, env['ctx']['objects'][b['obj']]) if b['obj'] >= 0 else (b['label'], b['label']); c = b['X'].mean(0)
    return dict(key=None, name=z, nameEn=e, centre=c, cm=env['fr']['to'](c[None])[0])


def plan_extras(env, have, base, short, rng):
    """<= MAX_EXTRAS photos (greedy, module doc) for the faces `short` and the targets still short after the existing photos and
    `base` (the needed standard viewpoints). Returns (chosen, pool)."""
    B, grid, targets = env['boxes'], env['grid'], env['targets']
    views = lambda T, cs: have[T['key']] + [c['views'][T['key']] for c in base + cs if T['key'] in c['counts']]
    new0 = {kf: [e for c in base for e in [entry(c['fe'], *kf, c['id'])] if e] for kf in short}
    dmap = lambda lv: 0 if lv == 3 else 1 if lv == 2 else 2
    d0 = {kf: dmap(after_level(env, B[kf[0]], kf[1], new0[kf])) for kf in short}; f0 = sum(d0.values())
    isin = lambda c, L: any(c is x for x in L)

    def touch(c):
        c['touch'] = {kf: e for kf in short for e in [entry(c['fe'], *kf, 'new')] if e}

    def total(cs):
        kfs = set().union(*(c['touch'] for c in cs)) if cs else set()
        return (sum(deficit(T, views(T, cs), PLAN) for T in targets) + f0
                + sum(dmap(after_level(env, B[kf[0]], kf[1], new0[kf] + [c['touch'][kf] for c in cs if kf in c['touch']])) - d0[kf] for kf in kfs))

    def score(c):
        return (.05 * (c['heightM'] >= 1) - .1 * min(c['floor'][1], 2) - .05 * (c['floor'][0] != 'observed')
                - .05 * min(math.dist(c['xyM'], p) for p in env['camXY']))

    def vett(c):
        if 'mc' not in c:
            vet(env, c, rng); face_mc(env, c, rng); touch(c)
    start = total([]); pool = []
    anchors = [box_aim(env, k) for k in sorted({k for k, _ in short})] + [T for T in targets if deficit(T, views(T, []), PLAN)]
    for A in anchors:
        for ring in RINGS_M:
            for a in range(0, 360, STEP_DEG):
                x, y = A['cm'][0] + ring * math.cos(math.radians(a)), A['cm'][1] + ring * math.sin(math.radians(a))
                if grid['why'](x, y):
                    continue
                for h in HEIGHTS_M:
                    c = candidate(env, A, x, y, h); c['fe'] = evidence(env, c['cam']); touch(c)
                    if total([c]) < start:
                        pool.append(c)

    def best_pair(now, chosen):
        """The pair of candidates (both counting for one short target) with the largest drop, None if none drops."""
        best, gain = None, 0
        for T in targets:
            if not deficit(T, views(T, chosen), PLAN):
                continue
            mine = sorted((c for c in pool if T['key'] in c['counts'] and not isin(c, chosen)), key=score, reverse=True)[:40]
            for a, b in combinations(mine, 2):
                g = now - total(chosen + [a, b])
                if g > gain:
                    best, gain = (a, b), g
        return best
    chosen = []
    while len(chosen) < MAX_EXTRAS and total(chosen):
        now = total(chosen)
        best = max((c for c in pool if not isin(c, chosen)), key=lambda c: (now - total(chosen + [c]), score(c)), default=None)
        if best is None or total(chosen + [best]) >= now:
            pair = best_pair(now, chosen) if len(chosen) + 2 <= MAX_EXTRAS else None
            if pair is None:
                break
            if any('mc' not in c for c in pair):
                for c in pair:
                    vett(c)
                continue
            chosen += list(pair); continue
        if 'mc' not in best:
            vett(best); continue
        chosen.append(best)
    for c in reversed(list(chosen)):
        if total([x for x in chosen if x is not c]) <= total(chosen):
            chosen = [x for x in chosen if x is not c]
    used = [c['xyM'] for c in chosen]; far = lambda a: all(math.dist(a['xyM'], e) >= .75 for e in used)
    for c in chosen:  # one fallback if the spot is blocked on site: same aim, swapping it in leaves the plan as good (and it is vetted)
        c['alternatives'], tries = [], 0
        swap = lambda a: total([a if x is c else x for x in chosen]) <= total(chosen)
        for a in sorted((a for a in pool if a['A'] is c['A'] and not isin(a, chosen) and far(a)), key=score, reverse=True):
            if tries == 3:
                break
            if swap(a):
                tries += 1; vett(a)
                if swap(a):
                    c['alternatives'].append(a); used.append(a['xyM']); break
    return chosen, pool


def face_why(env, k, fi, details, pool, full):
    """(zh, en) why face fi of box k stays low / unverified over every spot considered (module doc). details: evidence() with
    detail of the existing photos, standard viewpoints and extras; pool: every extra candidate; full: the 4 extras are used."""
    b, S = env['boxes'][k], env['fr']['S']; objs = env['ctx']['objects']
    if b['cap']:
        r = b['resid'] or 0
        return (f"掩码与盒子不符（{r:.0f} px）：尺寸取模型，补拍提不上去，先修掩码或形状",
                f"masks disagree with the box ({r:.0f} px): the size is the model's; photos do not lift it, fix the masks or the shape first")
    nominal = [c for c in pool if (c['fe'].get('ok0', c['fe']['ok']))[k, fi]]
    if any(not c.get('faceMc') or c['fe']['ok'][k, fi] for c in nominal):  # a spot that measures it was left out (by the cap, if full)
        return (f"补拍最多 {MAX_EXTRAS} 张，排不上（有能拍清楚它的站位）" if full else '有能拍清楚它的站位，但补拍时没选上',
                f"not within the {MAX_EXTRAS} extras (a spot that measures it exists)" if full else 'a spot that measures it exists but was not chosen')
    if nominal:
        return '能拍清楚它的站位在现场误差下靠不住（200 次蒙特卡洛 < 95 %）', 'the spots that measure it fail the field-error test (Monte Carlo < 95 % of 200)'
    weak = (list(b['ev0'][fi]) if b['ev0'] is not None else []) + [e for fe in list(details) + [c['fe'] for c in pool] for e in [entry(fe, k, fi, '')] if e]
    if weak:
        s = max(e[0] for e in weak)
        if s >= HIGH_PX_CM:  # credited, yet capped at low by box_faces (unpinned, or a bottom below the floor)
            return ('轮廓边拍清楚了，但盒子拟合定不住这个面（和别的面互相抵消，或底面在地面下）：考虑过的站位里没有能定住它的',
                    'its outline is credited but the fit cannot pin the face (it trades off, or the bottom sank): no spot considered pins it')
        return f"只能远拍（最多 {s:.1f} px/cm，要 ≥ 10）", f"seen only far (at most {s:.1f} px/cm, need >= 10)"
    hid = Counter()
    for d in details:
        hid.update(d['hid'].get((k, fi), {}))
    if hid:
        o = objs[hid.most_common(1)[0][0]]; z, e = oname(env, o)
        if o['id'] in env['seeThroughIds']:
            return f"只能隔着{z}看到（网按挡住算）", f"seen only through the {e} (counted as hiding)"
        return f"被{z}挡住", f"hidden by the {e}"
    return '考虑过的站位都看不到它的轮廓（朝防护区内或背对）', 'its outline is not seen from any spot considered (it faces into the guarded cell or away)'


def unmodelled(ctx, fr, spec):
    """A known object the report does not model: footprintM (plan floor frame) or the mirror image of entity mirrorOf across
    the perpendicular bisector of the two `across` ids; heightM (default: the mirrored entity's, else 0-2 m)."""
    from scipy.spatial import ConvexHull
    if spec.get('footprintM'):
        xy, h = np.array(spec['footprintM'], float), spec.get('heightM') or [0., 2.]
    else:
        P = fr['to'](entity(ctx, spec['mirrorOf'])['mesh'][0]); a, b = [fr['to'](entity(ctx, i)['mesh'][0])[:, :2].mean(0) for i in spec['across']]
        u = (b - a) / np.linalg.norm(b - a); xy = P[:, :2] - 2 * ((P[:, :2] - (a + b) / 2) @ u)[:, None] * u
        h = spec.get('heightM') or [float(P[:, 2].min()), float(P[:, 2].max())]
    return dict({k: v for k, v in spec.items() if k not in ('footprintM', 'heightM')}, footprintM=xy[ConvexHull(xy).vertices].round(3).tolist(),
                heightM=[round(float(x), 3) for x in h])


def prepare(ctx, opts):
    """Frame, plates, known un-modelled objects (occluders), floor grid, targets, landmarks, floor-observation maps, boxes."""
    from scipy import ndimage
    rng = np.random.default_rng(0); fr = floor_frame(ctx)
    movable = {entity(ctx, i)['id'] for i in opts.get('movable') or []}
    for L in opts.get('landmarks') or []:
        if any(entity(ctx, i)['id'] in movable for i in L['ids']):
            raise ValueError(f"landmark {L['en']!r} is a movable object: use fixed ones")
    specs = [dict(s) for s in opts.get('targets') or []]; plates, extra = {}, []
    for s in specs:  # a plate covering a face is an occluder like any model
        o = entity(ctx, s['entityId'])
        if s.get('coveredBy') and o['id'] not in plates:
            plates[o['id']] = plate(ctx, fr, s)
            extra.append(dict(id=f"plate:{o['id']}", label=f"{s['coveredBy'].get('zh', '前护板')}（{o['label']}）", kind='plate', mesh=plates[o['id']][1], masks={}))
    unm = [unmodelled(ctx, fr, u) for u in opts.get('knownUnmodelled') or []]
    extra += [dict(id=f'unmodelled:{k}', label=U['zh'], kind='unmodelled', mesh=prism(fr, np.array(U['footprintM']), *U['heightM']), masks={})
              for k, U in enumerate(unm)]  # opaque: a sightline through one is blocked
    ctx = dict(ctx, objects=ctx['objects'] + extra)
    keep_out = [[entity(ctx, i)['id'] for i in g] for g in opts.get('keepOut') or []]
    excluded = sorted({i[:8] for g in keep_out for i in g if i in movable}); keep_out = [[i for i in g if i not in movable] for g in keep_out]
    grid = floor_grid(ctx, fr, keep_out, rng)
    scale = opts.get('scaleReference') or report_scale_reference(ctx)
    if scale:
        specs.append(dict(scale, role='scale', edge='surface', key=scale.get('key') or 'scale-reference'))
    targets = [resolve(ctx, fr, s, rng, plates) for s in specs]
    see = [entity(ctx, i) for i in opts.get('seeThrough') or []]
    names = dict(opts.get('names') or {}); names.update({f'unmodelled:{k}': [U['zh'], U['en']] for k, U in enumerate(unm)})
    env = dict(ctx=ctx, fr=fr, grid=grid, targets=targets, byKey={T['key']: T for T in targets}, all=scene([o['mesh'] for o in ctx['objects']]),
               K0=ctx['cams'][0]['K'], wh=(ctx['cams'][0]['w'], ctx['cams'][0]['h']), names=names, keepOutExcludedMovable=excluded,
               seeThroughZh='、'.join(re.sub(r'[（(].*?[）)]', '', o['label']).strip() for o in see) or '透视的模型',
               seeThroughEn=', '.join(re.sub(r'[（(].*?[）)]', '', o['label']).strip() for o in see) or 'see-through models',
               seeThroughIds=[o['id'] for o in see], camXY=[fr['to'](c['C'][None])[0, :2].tolist() for c in ctx['cams']], unmodelled=unm)
    if len(set(env['byKey'])) != len(targets):
        raise ValueError('two targets share a key: give them distinct keys')
    for T in targets:
        T['floorPts'] = floor_points(fr, grid, T['cm'])
    ids = {o['id'] for o in see}
    ny, nx = grid['shape']; jj, ii = np.mgrid[0:ny, 0:nx]; cx, cy = grid['lo'][0] + (ii.ravel() + .5) * GRID_M, grid['lo'][1] + (jj.ravel() + .5) * GRID_M
    Fg = fr['at'](cx, cy, np.zeros(len(cx))); seen, seen_any = np.zeros(len(cx), bool), np.zeros(len(cx), bool)
    clear = scene([o['mesh'] for o in ctx['objects'] if o['id'] not in ids]) if ids else env['all']
    for cam in ctx['cams']:  # floor seen in a photo, or within 1 m of where the photographer stood
        near = np.hypot(cx - fr['to'](cam['C'][None])[0, 0], cy - fr['to'](cam['C'][None])[0, 1]) <= 1; inside = in_frame(cam, Fg, 0)
        seen |= near | inside & unblocked(fr, env['all'], cam['C'], Fg)
        seen_any |= near | inside & unblocked(fr, clear, cam['C'], Fg)
    grid['unobs'] = ndimage.distance_transform_edt(~seen.reshape(ny, nx)) * GRID_M
    grid['unobsAny'] = ndimage.distance_transform_edt(~seen_any.reshape(ny, nx)) * GRID_M

    def floor_status(x, y):
        j, i = np.clip(grid['ij']([x, y])[0][::-1], 0, [ny - 1, nx - 1]); d = float(grid['unobs'][j, i])
        return ('observed' if d <= UNOBSERVED_M else 'seeThrough' if grid['unobsAny'][j, i] <= UNOBSERVED_M else 'unobserved', round(d, 2))
    env['floorStatus'] = floor_status
    env['lms'] = [dict(L, xy=np.mean([fr['to'](entity(ctx, i)['mesh'][0])[:, :2].mean(0) for i in L['ids']], 0).round(2).tolist())
                  for L in opts.get('landmarks') or []]
    env['boxes'] = load_boxes(env, opts['boxes']) if opts.get('boxes') else []
    return env


def pose(env, x, y, h, A):
    """A planned camera at floor point (x, y), height h, facing A: level when crouched (h < 1 m), else aimed at it."""
    fr = env['fr']; C = fr['at'](x, y, h); aim = A['centre'] if h >= 1 else A['centre'] - (fr['n'] @ (A['centre'] - C)) * fr['n']
    return look(C, aim, env['K0'], *env['wh'], fr['n'])


def candidate(env, A, x, y, h):
    """A planned photo facing A (a target, or an aim point with centre / cm / name): its views and the targets it counts for (PLAN)."""
    cam = pose(env, x, y, h, A)
    c = dict(A=A, aim=A.get('key'), xyM=[round(x, 2), round(y, 2)], heightM=h, cam=cam, views={}, counts=set(), why={}, floor=env['floorStatus'](x, y))
    for T in env['targets']:
        if T is not A and not in_frame(cam, T['centre'][None], -.2)[0]:
            continue
        v = dict(view(env, T, cam)[0], new=True); c['views'][T['key']] = v; ok, f = judge(T, v, PLAN)
        if f is None and h < 1 and T['needLow'] and not any(is_low(T, v, g, PLAN) for g in ok):
            f = 'belowLowerThird'  # a crouched, lens-level photo serves only low edges it keeps above the lower third line
        c['why'][T['key']] = f
        if f is None:
            c['counts'].add(T['key'])
    return c


def jittered(env, c, rng, n):
    """n poses around a planned one: +-posM in position (uniform in a disc), +-heightM, +-aimDeg in yaw and pitch."""
    fr, nrm = env['fr'], env['fr']['n']; out = []
    for _ in range(n):
        r, a = JITTER['posM'] * math.sqrt(rng.random()), rng.random() * 2 * math.pi
        C = fr['at'](c['xyM'][0] + r * math.cos(a), c['xyM'][1] + r * math.sin(a), c['heightM'] + rng.uniform(-1, 1) * JITTER['heightM'])
        z = rotate(c['cam']['R'][2], nrm, math.radians(rng.uniform(-1, 1) * JITTER['aimDeg']))
        x = np.cross(z, nrm); z = rotate(z, x / np.linalg.norm(x), math.radians(rng.uniform(-1, 1) * JITTER['aimDeg']))
        out.append(look(C, C + z, c['cam']['K'], c['cam']['w'], c['cam']['h'], nrm))
    return out


def trial_fail(T, planned, v):
    """Why a jittered view no longer does what the planned one does (None = it does): a RULE fails, it sees none of the
    faces the plan counted it for, or it is no longer low where the plan relies on it being the low photo."""
    ok, f = judge(T, v, RULE)
    if f:
        return f
    faces = judge(T, planned, PLAN)[0]; same = [g for g in ok if g in faces]
    if not same:
        return 'otherFace'
    if T['needLow'] and any(is_low(T, planned, g, PLAN) for g in faces) and not any(is_low(T, v, g, RULE) for g in same):
        return 'notLow'
    return None


def vet(env, c, rng):
    """Monte Carlo against the RULES (200 jittered poses): drop the targets this photo cannot be relied on for (most
    failing first) until it does what it is planned for, for all remaining targets together, in >= MIN_PASS of the trials."""
    if 'mc' in c:
        return c
    rows = []
    for cam in jittered(env, c, rng, JITTER['trials']):
        row = {}
        for k in c['counts']:
            T = env['byKey'][k]; v = dict(view(env, T, cam)[0], new=True); row[k] = (v, trial_fail(T, c['views'][k], v))
        rows.append(row)
    rate = lambda ks: sum(all(r[k][1] is None for k in ks) for r in rows) / len(rows); keys = set(c['counts'])
    while keys and rate(keys) < MIN_PASS:
        worst = max(sorted(keys), key=lambda k: sum(r[k][1] is not None for r in rows)); keys.discard(worst); c['why'][worst] = 'fieldError'
    c['counts'] = keys
    c['mc'] = dict(rows=[{k: r[k] for k in keys} for r in rows], trials=len(rows), passRate=round(rate(keys), 3) if keys else None,
                   failures=dict(Counter(r[k][1] for r in rows for k in keys if r[k][1])))
    return c


def plan_rate(env, chosen, have, keys):
    """Share of the trials (trial j of every photo together) in which every target in keys (those the plan completes) is still
    complete by the RULES."""
    n = min((c['mc']['trials'] for c in chosen), default=0); ok = 0; ts = [T for T in env['targets'] if T['key'] in keys]
    for j in range(n):
        ok += all(not missing(T, have[T['key']] + [c['mc']['rows'][j][T['key']][0] for c in chosen if T['key'] in c['counts']
                                                   and not judge(T, c['mc']['rows'][j][T['key']][0], RULE)[1]], RULE) for T in ts)
    return dict(trials=n, planPassRate=round(ok / n, 3) if n and ts else None, targets=sorted(keys), jitter=JITTER, minPhotoPassRate=MIN_PASS,
                scope='the targets the plan completes; faces: per photo (>= 95 % each), no joint test')


def coverage_md(env, std, chosen, rows, title):
    """The face x viewpoint table (markdown, paste unchanged): per box face, every photo's outline credit (● >= 10 px/cm and,
    planned, it holds in the Monte Carlo test, ○ an outline edge that counts but below that), which photos see the face itself
    (not counted), before -> after and why it stays short."""
    ncam = len(env['ctx']['cams']); cols = [f'照片 {p}' for p in range(1, ncam + 1)] + [v['id'] for v in std] + [f'新 {i}' for i in range(1, len(chosen) + 1)]
    sym = lambda e: '' if e is None else '●' if e[0] >= HIGH_PX_CM else '○'
    seen_zh = dict(not_facing='背对', occluded='被挡', out_of_frame='不在画面内', seen='看到')
    out = [f"<!-- generated by capture_plan.py: {title}, coverage -->",
           '● = 这张照片里这个面的轮廓边 ≥ 80 % 在画面里、没被挡、≥ 10 px/cm（按到这条边的直线距离；补拍的还要在 200 次蒙特卡洛里 ≥ 95 % 成立）；'
           '○ = 轮廓边在画面里、没被挡，但不到 10 px/cm；— = 这个标准视角在这里站不了。看到这个面 = 面本身朝着相机、拍得到的照片（只是参考，不算置信度）。'
           '现在 = 盒子文件的置信度；补拍后 = 加上需要的标准视角和补拍以后的预计。'
           + ('现有照片那几列是按同一规则预测的（盒子文件没有逐张证据），和“现在”不一致的地方以“现在”为准。' if any(b['ev0'] is None for b in env['boxes']) else ''), '',
           '| 物体 | 面 | 看到这个面 | ' + ' | '.join(cols) + ' | 现在 → 补拍后 | 还差的原因 |', '|---|---|---|' + '---|' * len(cols) + '---|---|']
    for r in rows:
        k, fi, b = r['_k'], r['_fi'], env['boxes'][r['_k']]
        ev = b['ev0'][fi] if b['ev0'] is not None else b['evPred'][fi]
        cells = [sym(next((e for e in ev if e[2] == p), None)) for p in range(1, ncam + 1)]
        cells += [sym(entry(v['fe'], k, fi, '')) if 'fe' in v else '—' for v in std] + [sym(entry(c['fe'], k, fi, '')) for c in chosen]
        sn = '照片 ' + '、'.join(map(str, r['seenPhotos'])) if r['seenPhotos'] else seen_zh.get(r['seenStatus'], r['seenStatus'] or '—')
        out.append(f"| {r['name']} | {FACE_ZH[r['face']]} | {sn} | " + ' | '.join(cells) + f" | {CONF_ZH[r['before']]} → {CONF_ZH[r['after']]} | {r.get('whyZh', '')} |")
    return '\n'.join(out) + '\n'


def run(ctx, opts):
    if not ctx.get('floor'):
        return dict(status='no_floor')
    if not opts.get('boxes'):
        raise ValueError("opts.boxes: the report's SHARED FORMAT boxes are required (capture_plan does not build boxes)")
    env = prepare(ctx, opts); fr, grid, targets, lms, B = env['fr'], env['grid'], env['targets'], env['lms'], env['boxes']; rng = np.random.default_rng(1)
    existing, have = [], {T['key']: [] for T in targets}
    for k, cam in enumerate(ctx['cams']):
        p = fr['to'](cam['C'][None])[0]; zh, en = where(lms, p[0], p[1])
        row = dict(photo=k + 1, xyM=p[:2].round(2).tolist(), heightM=round(float(p[2]), 2), whereZh=zh, whereEn=en, views={})
        for T in targets:  # the same rules as a new photo, plus a report mask (geometry alone is not evidence)
            v = dict(view(env, T, cam)[0], new=False, reportMask=k in T['obj']['masks'])
            v['fail'] = fails(T, v, RULE) or ('noReportMask' if T['obj']['masks'] and not v['reportMask'] else None)
            v['sees'] = v['fail'] is None; v['countsOnFaces'] = judge(T, v, RULE)[0] if v['sees'] else []
            v['low'] = any(is_low(T, v, g, RULE) for g in v['countsOnFaces']) if T['role'] != 'scale' else False; row['views'][T['key']] = v
            if v['sees']:
                have[T['key']].append(v)
        existing.append(row)
    exist_fe = [evidence(env, cam, True) for cam in ctx['cams']]  # predicted: for the reasons and the table where the file has none
    for j, fe in enumerate(exist_fe):  # box_faces fits an object only in the photos with its report mask
        for k, b in enumerate(B):
            if b['obj'] < 0 or j not in ctx['objects'][b['obj']]['masks']:
                fe['len'][k], fe['ok'][k] = 0., False; fe['hid'] = {kf: v for kf, v in fe['hid'].items() if kf[0] != k}
    for k, b in enumerate(B):  # the existing photos' evidence per face: the boxes file's pxPerCm (all faces), else None (after_level)
        fs = [b['rec']['faces'][F] for F in FACES6]
        b['ev0'] = [[(float(s), ctx['cams'][int(p) - 1]['C'], int(p)) for p, s in f['pxPerCm'].items()] for f in fs] if all('pxPerCm' in f for f in fs) else None
        b['evPred'] = [[e for j, fe in enumerate(exist_fe) for e in [entry(fe, k, fi, j + 1)] if e] for fi in range(6)]
    F = entrance(env, opts['standardFrame']) if opts.get('standardFrame') else None
    std = [standard_view(env, F, s, rng) for s in STANDARD] if F else []
    need0 = {(k, fi) for k, b in enumerate(B) if not b['cap'] for fi in range(6) if b['before'][fi] <= 1}
    short_t0 = {T['key'] for T in targets if deficit(T, have[T['key']], RULE)}
    for v in std:
        v['needed'] = v['standable'] != 'no' and bool(v['counts'] & short_t0 or any(v['fe']['ok'][kf] for kf in need0))
    base = [v for v in std if v['needed']]
    lab = lambda c: c.get('id') or 'new'
    new_std = lambda k, fi: [e for c in base for e in [entry(c['fe'], k, fi, c['id'])] if e]
    short1 = sorted(kf for kf in need0 if after_level(env, B[kf[0]], kf[1], new_std(*kf)) <= 1)
    chosen, pool = plan_extras(env, have, base, short1, rng)
    for i, c in enumerate(chosen, 1):
        c['id'] = f'新 {i}'
    planned = base + chosen; views = lambda T: have[T['key']] + [c['views'][T['key']] for c in planned if T['key'] in c['counts']]
    for c in planned + [a for c in chosen for a in c['alternatives']]:  # every target's view from this photo, for the tables
        for T in targets:
            if T['key'] not in c['views']:
                c['views'][T['key']] = dict(view(env, T, c['cam'])[0], new=True)
    details = exist_fe + [c['fe'] for c in planned]; full = len(chosen) >= MAX_EXTRAS; faces, dims, mism = [], {}, []
    for k, b in enumerate(B):
        name = oname(env, ctx['objects'][b['obj']])[0] if b['obj'] >= 0 else b['label']; newf = {}
        for fi, Fn in enumerate(FACES6):
            new = newf[fi] = [e for c in planned for e in [entry(c['fe'], k, fi, lab(c))] if e]; aft = after_level(env, b, fi, new)
            by = ([e[2] for e in b['ev0'][fi] if e[0] >= HIGH_PX_CM] if b['ev0'] is not None else []) + ([e[2] for e in new if e[0] >= HIGH_PX_CM] if not b['cap'] else [])
            fr_ = b['rec']['faces'][Fn]
            r = dict(entityId=b['id'][:8], name=name, face=Fn, before=CONF[b['before'][fi]], after=CONF[aft], by=by, capped=b['cap'],
                     seenPhotos=fr_.get('photos') or [], seenStatus=fr_.get('status'), _k=k, _fi=fi)
            if aft <= 1:
                r['whyZh'], r['whyEn'] = face_why(env, k, fi, details, pool, full)
            faces.append(r)
        fd = b['rec'].get('dims') or {}; dims[b['id']] = {}
        if b['ev0'] is not None:  # box_faces' grades recomputed from its per-photo evidence (R1, its caps): must equal the file's
            fb, db = regrade(env, b); da = regrade(env, b, newf)[1]
            mism += [dict(entityId=b['id'][:8], face=FACES6[fi], file=CONF[b['before'][fi]], recomputed=CONF[g]) for fi, g in enumerate(fb) if g != b['before'][fi]]
            mism += [dict(entityId=b['id'][:8], dim=d, file=fd[d]['confidence'], recomputed=CONF[g]) for d, g in db.items() if d in fd and fd[d]['confidence'] != CONF[g]]
        lv = lambda key: [CONF.index(r[key]) for r in faces[-6:]]
        for d, fs in DIMS.items():  # before = the file's dimension grade; after = box_faces' rule on all the evidence, never lower
            bd = CONF.index((fd.get(d) or {}).get('confidence') or CONF[min(lv('before')[i] for i in fs)])
            if b['ev0'] is not None:
                ad = max(bd, da[d])
            else:  # no per-photo evidence in the file: it moves only when a face that bounds it does
                ad = max(bd, min(lv('after')[i] for i in fs)) if any(lv('after')[i] != lv('before')[i] for i in fs) else bd
            dims[b['id']][d] = dict(before=CONF[bd], after=CONF[ad])
    graded = [b for b in B if b['ev0'] is not None]
    agree = [(e0 >= HIGH_PX_CM) == (ep >= HIGH_PX_CM) for b in graded for fi in range(6) for p_ in range(1, len(ctx['cams']) + 1)
             if (b['obj'] >= 0 and p_ - 1 in ctx['objects'][b['obj']]['masks'])
             for e0, ep in [(next((e[0] for e in b['ev0'][fi] if e[2] == p_), 0.), next((e[0] for e in b['evPred'][fi] if e[2] == p_), 0.))]]
    before_check = dict(boxes=len(graded), faces=6 * len(graded), dims=4 * len(graded), mismatches=mism,
                        predictorCreditAgreement=round(sum(agree) / len(agree), 3) if agree else None, predictorPairs=len(agree))
    files = {}
    md = [f"<!-- generated by capture_plan.py: {opts.get('title') or 'capture plan'}, instructions -->", '**标准视角**（通用的一组，从入口量起；这个工位要拍 '
          + (f"{len(base)} 张：{'、'.join(v['id'] for v in base)}" if base else '0 张') + '）', '']
    shots = []
    for v in std:
        if v.get('needed'):
            zh, en = instruction(env, v, f"{v['id']}（标准视角：{v['spec']['zh']}）", f"{v['id']} (standard viewpoint: {v['spec']['en']})", extra=served_words(env, v, need0))
            zh += '标准视角没有备选：站不了就别换地方随便拍，拍一张挡路东西的照片，重新规划。'; en += ' A standard viewpoint has no fallback: if it is blocked, do not improvise; photograph what blocks it and re-plan.'
            v['instructionZh'], v['instructionEn'] = zh, en; md.append(f"- {zh}")
            img = render(env, v['cam'], v['A'] if 'P' in v['A'] else None); cv2.putText(img, f"{v['id']}: h {v['heightM']:.1f} m", (5, 16), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
            files[f"{v['id']}-preview.jpg"] = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes()
        elif v['standable'] == 'no':
            md.append(f"- {v['id']}（{v['spec']['zh']}）：站不了（{WHY_ZH.get(v['standWhy'], v['standWhy'])}），不拍。")
        else:
            md.append(f"- {v['id']}（{v['spec']['zh']}）：这个工位不需要（它能拍清楚的面和目标现有照片已经够了）。")
        shots.append(dict(id=v['id'], definitionZh=v['words'][0], definitionEn=v['words'][1], optional=bool(v['spec'].get('optional')), xyM=v.get('xyM'),
                          standable=v['standable'], standWhy=v.get('standWhy'), needed=bool(v.get('needed')),
                          heightM=v['spec']['heightM'], phoneRangeM=v.get('phoneRange'), targets=sorted(v['counts']) if 'counts' in v else [],
                          facesRelied=[f"{B[k]['id'][:8]}:{FACES6[fi]}" for k, fi in zip(*np.nonzero(v['fe']['ok']))] if 'fe' in v else [],
                          targetsPassRate=v['mc']['passRate'] if 'mc' in v else None, instructionZh=v.get('instructionZh'), instructionEn=v.get('instructionEn')))
    md += ['', f"**补拍**（最多 {MAX_EXTRAS} 张，给标准视角之后还不够的面和目标；这个工位 {len(chosen)} 张）", '']
    new = []
    for i, c in enumerate(chosen, 1):
        zh, en = instruction(env, c, f'新 {i} 号（补拍）', f'New #{i} (extra)', extra=served_words(env, c, set(short1))); alts = []
        for a in c['alternatives']:
            az, ae = instruction(env, a, f'新 {i} 号备选', f'New #{i} fallback', extra=served_words(env, a, set(short1)))
            alts.append(dict(xyM=a['xyM'], heightM=a['heightM'], phoneRangeM=a['phoneRange'], floor=a['floor'], targets=sorted(a['counts']),
                             instructionZh=az, instructionEn=ae, targetsPassRate=a['mc']['passRate']))
        if alts:
            zh += '站不了就用备选（见下）。'; en += ' If the spot is blocked, use the fallback (below).'
        else:
            zh += '没有能替代这一张的站位：站不了就别换地方随便拍，拍一张挡路东西的照片，重新规划。'
            en += ' No spot can replace this one: if it is blocked, do not improvise; photograph what blocks it and re-plan.'
        md.append(f"{i}. {zh}"); md += [f"   - {a['instructionZh']}" for a in alts]
        new.append(dict(n=i, xyM=c['xyM'], heightM=c['heightM'], phoneRangeM=c['phoneRange'], posture='crouched, lens level' if c['heightM'] < 1 else 'standing, aimed',
                        aim=c['aim'] or c['A']['nameEn'], floor=c['floor'], cameraNative=c['cam']['C'].round(4).tolist(), targets=sorted(c['counts']),
                        faces=[f"{B[k]['id'][:8]}:{FACES6[fi]}" for (k, fi) in sorted(c['touch']) if c['fe']['ok'][k, fi]],
                        instructionZh=zh, instructionEn=en, robustness=dict(trials=JITTER['trials'], targetsPassRate=c['mc']['passRate'],
                                                                            facesMinMcRate=round(float(min((c['fe']['mcRate'][kf] for kf in c['touch'] if c['fe']['ok'][kf]), default=1.)), 3)),
                        views={k: rnd(dict(v, counts=k in c['counts'], fail=fails(env['byKey'][k], v, PLAN))) for k, v in c['views'].items()}, alternatives=alts))
        T = c['A'] if 'P' in c['A'] else None; img = render(env, c['cam'], T)
        cv2.putText(img, f"new {i}: h {c['heightM']:.1f} m", (5, 16), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
        files[f'new{i}-preview.jpg'] = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes()
    files['instructions.md'] = ('\n'.join(md) + '\n').encode()
    files['coverage.md'] = coverage_md(env, std, chosen, faces, opts.get('title') or 'capture plan').encode()
    out_t = []
    for T in targets:
        out_t.append(dict(key=T['key'], entityId=T['entityId'], label=T['label'], name=T['name'], nameEn=T['nameEn'], edge=T['edge'], role=T['role'],
                          edgeHeightM=round(T['heightM'], 3), edgeHeightSource=T['edgeHeightSource'], extendedFromModelBottomM=T['extendedFromM'],
                          coveredBy=None if T['cover'] is None else {k: v for k, v in T['cover'].items() if k != 'n2'}, needsLowPhoto=T['needLow'],
                          centreM=T['cm'].round(2).tolist(), placementCheck=T['placementCheck'], provenance=T['provenance'],
                          existing={str(r['photo']): rnd(r['views'][T['key']]) for r in existing}, missingBefore=missing(T, have[T['key']], RULE),
                          photos=[c['id'] for c in planned if T['key'] in c['counts']], missingAfter=missing(T, views(T), PLAN), missingAfterRules=missing(T, views(T), RULE)))
    check = render(env, ctx['cams'][0], skip=(0,)); ph = cv2.resize(np.asarray(ctx['gray'][0], np.float32), check.shape[1::-1]).astype(np.uint8)
    files['render-check-photo1.jpg'] = cv2.imencode('.jpg', np.hstack([check, cv2.cvtColor(ph, cv2.COLOR_GRAY2BGR)]), [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()
    files['topdown.png'] = topdown(env, chosen, opts.get('title') or 'capture plan', std)
    rules = dict(rule=rnd(RULE), plan=rnd(PLAN), minFacing=MIN_FACING, lowEdgeBelowM=LOW_EDGE_M, lowCameraM=list(LOW_CAM_M), scaleViews=SCALE_VIEWS,
                 floorRadiusM=FLOOR_R_M, unobservedM=UNOBSERVED_M, jitter=JITTER, minPhotoPassRate=MIN_PASS, lowerThird=LOWER_THIRD,
                 faces=dict(evidenceStepPx=EVID_STEP, maxSamplesPerEdge=EVID_MAX_N, evidenceMinPx=EVID_MIN_PX, edgeKnownFrac=EDGE_FRAC, highPxPerCm=HIGH_PX_CM, highDeg=HIGH_DEG,
                            pxPerCm='fx * 1 cm / the edge\'s straight-line distance (box_faces.outline_px)', grading='box_faces.face_grade / dim_grade (R1)',
                            occluderMarginM=OCC_MARGIN_M, occluderMarginRel=OCC_MARGIN_REL, conf=list(CONF), dims={k: [FACES6[i] for i in v] for k, v in DIMS.items()}),
                 extras=dict(max=MAX_EXTRAS, ringsM=list(RINGS_M), stepDeg=STEP_DEG, heightsM=list(HEIGHTS_M)), personRadiusM=PERSON_R_M, boundsPadM=PAD_M,
                 standard=[{k: v for k, v in s.items()} for s in STANDARD])
    frame = dict(description='x right and y into the cell as seen from the existing photos (mean horizontal viewing direction), z up, metres; '
                             'origin = floor point below the mean existing camera position',
                 originNative=fr['O'].round(5).tolist(), xNative=fr['r'].round(5).tolist(), yNative=fr['f'].round(5).tolist(), upNative=fr['n'].round(5).tolist())
    count = lambda key, rs: {c: sum(r[key] == c for r in rs) for c in CONF}
    doc = opts['boxes'] if 'boxes' in opts['boxes'] else {}
    summary = dict(standardNeeded=[v['id'] for v in base], standardUnknown=[v['id'] for v in std if v['standable'] == 'unknown'],
                   standardNo=[v['id'] for v in std if v['standable'] == 'no'], extras=len(chosen), facesShortNow=len(need0), facesShortAfterStandard=len(short1),
                   facesBefore=count('before', faces), facesAfter=count('after', faces),
                   cappedBoxes=sorted(b['id'][:8] for b in B if b['cap']), beforeCheck=before_check,
                   dimsBefore={d: count('before', [v[d] for v in dims.values()]) for d in DIMS}, dimsAfter={d: count('after', [v[d] for v in dims.values()]) for d in DIMS},
                   targetsCompleteAfter=[T['key'] for T in out_t if not T['missingAfterRules']],
                   targetsShortAfter=[T['key'] for T in out_t if T['missingAfterRules']])
    return plain(dict(status='ok', nativeToMeters=fr['S'], frame=frame, rules=rules, landmarks=lms, keepOutExcludedMovable=env['keepOutExcludedMovable'],
                boxes=dict(count=len(B), schema=doc.get('schema'), revisionId=doc.get('revisionId'), source=opts.get('boxesSource'),
                           existingEvidence=sorted({'file pxPerCm' if b['ev0'] is not None else 'none in the file: predicted for the table and reasons only' for b in B})),
                summary=summary, standardSet=shots, newPhotos=new, faces=[{k: v for k, v in r.items() if not k.startswith('_')} for r in faces], dims=dims,
                knownUnmodelled=env['unmodelled'], existingPhotos=[dict(r, views={k: rnd(v) for k, v in r['views'].items()}) for r in existing],
                targets=out_t, robustness=plan_rate(env, planned, have, set(summary['targetsCompleteAfter'])) if planned else None,
                cameraConvention=convention(env, np.random.default_rng(2)), files=files))


def entrance(env, spec):
    """Frame of the standard viewpoints: the two entrance sides ({'left': ids, 'right': ids}), x from the left side to the
    right one, 'out' away from the cell (the side the existing photos were taken from); E = the midpoint."""
    fr, ctx = env['fr'], env['ctx']
    L, R = [np.mean([fr['to'](entity(ctx, i)['mesh'][0])[:, :2].mean(0) for i in spec[k]], 0) for k in ('left', 'right')]
    x = (R - L) / np.linalg.norm(R - L); out = np.array([x[1], -x[0]]); E = (L + R) / 2
    out = out if (np.mean(env['camXY'], 0) - E) @ out >= 0 else -out
    return dict(L=L, R=R, E=E, x=x, out=out)


def _walk(x):
    if isinstance(x, dict):
        for v in x.values():
            yield from _walk(v)
    elif isinstance(x, list):
        for v in x:
            yield from _walk(v)
    else:
        yield x


def _check():
    """Synthetic cell. A post (lower edge 24 cm) with a front plate (coveredBy), behind an occluding bollard; a wall behind
    it; a cart box on the right (movable: left out of the keep-out hull); a closed room on the left (free floor nobody can
    reach); a see-through mesh panel; an 'e-stop' on the post given by a pixel; a known un-modelled fence. Checks: the target
    rules (as before), the face evidence of a box measured through its outline (box_faces' rule), its grade, occluders and
    reasons (no false 'contact'), the un-modelled fence blocking, capped boxes, the generic standard set (mirror pairs,
    standability yes / unknown / no, the scale and keep-out anchors) and a whole run with <= 4 extras, each Monte Carlo vetted."""
    up = np.array([0, 0, 1.])

    def box(lo, hi):
        V, F = wsc.primitive_mesh(dict(kind='box', dimensions=list(np.subtract(hi, lo))))
        return V + np.add(lo, hi) / 2, F

    def rec(lo, hi, conf='unverified', **kw):  # a SHARED FORMAT box record (axis-aligned, front towards -y), as box_faces writes it
        c = (np.add(lo, hi) / 2).tolist(); return dict(label=kw.pop('label', 'box'), centerNative=c, axes=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                                                       sizeM=np.subtract(hi, lo).tolist(), bottomM=lo[2], topM=hi[2],
                                                       faces={F: dict(photos=[], status='seen', confidence=conf, need=None) for F in FACES6}, **kw)
    cyl = wsc.primitive_mesh(dict(kind='cylinder', radius=.12, height=1., segments=32)); cyl = (cyl[0] + [0, 2.3, .5], cyl[1])
    room = [box((-2.9, 1.2, 0), (-1.3, 1.25, 2)), box((-2.9, 2.75, 0), (-1.3, 2.8, 2)), box((-2.9, 1.2, 0), (-2.85, 2.8, 2)), box((-1.35, 1.2, 0), (-1.3, 2.8, 2))]
    objs = [dict(id=i, label=i, kind='x', mesh=m, masks={}) for i, m in
            (('post', box((-.05, 2.95, .24), (.05, 3.05, 1.2))), ('bollard', cyl), ('wall', box((-3, 4.2, 0), (3, 4.3, 2.5))),
             ('cart', box((2.4, 2.6, 0), (3.0, 3.2, .9))), ('mesh', box((2.2, 1.5, 0), (4.2, 1.53, 2.))), ('marker', box((-1., 1.8, 0), (-.8, 2., .3))))]
    objs += [dict(id=f'room{k}', label='room', kind='x', mesh=m, masks={}) for k, m in enumerate(room)]
    K, W, H = np.array([[2800, 0, 1439.5], [0, 2800, 1919.5], [0, 0, 1.]]), 2880, 3840  # the report phone: the rules are in px/cm
    cams = [look(np.array([0, 0, 1.5]), np.array([0, 3, .7]), K, W, H, up), look(np.array([3.2, 0, 1.5]), np.array([3.2, 3, .3]), K, W, H, up)]
    # camera convention: the optical axis projects to the principal point, up is up in the image, depth = R[2].(X - C)
    c0 = cams[0]; Xa = c0['C'] + 2 * c0['R'][2]; uv, z = wsc.project(c0, np.array([Xa, Xa + .3 * up]))
    assert np.allclose(uv[0], [1439.5, 1919.5]) and uv[1, 1] < uv[0, 1] and abs(z[0] - 2) < 1e-9 and abs(c0['R'][2] @ (Xa - c0['C']) - 2) < 1e-9, (uv, z)
    for o in (o for o in objs if o['id'] in ('post', 'marker')):  # report masks of the post and of an off-centre marker (a centred object cannot tell u from -u)
        uv, _ = wsc.project(c0, wsc.sample_surface(*o['mesh'], 3000, np.random.default_rng(5))[0])
        mask = np.zeros((H, W), np.uint8); cv2.fillPoly(mask, [cv2.convexHull(np.round(uv).astype(np.int32))], 1); o['masks'] = {0: mask.astype(bool)}
    ctx = dict(cams=cams, gray=[np.full((H, W), 128, np.float32)] * 2, objects=objs, S=1., floor=(up, 0.), doc={})
    assert selector(dict(part='lower edge')) == dict(band='bottom') and selector(dict(part='right wing')) == dict(side='right')
    for bad in ('bright panel', 'baseboard', 'uppercase'):
        try:
            selector(dict(part=bad)); raise AssertionError(f'{bad!r} matched by substring')
        except ValueError:
            pass
    rng0 = np.random.default_rng(4); Pr, Nr = wsc.sample_surface(*box((0, 0, 0), (1.2, .05, .03)), 4000, rng0)  # a rail along x: side band
    side = np.abs(Nr[:, 2]) < .7; fs, ns = faces_of(Pr[side][:400], Nr[side][:400])
    assert len(fs) == 2 and all(abs(abs(v[1]) - 1) < .05 for v in ns), [(len(g), v.round(2)) for g, v in zip(fs, ns)]  # the two long faces, not the ends
    plate_spec = dict(face='front', depthM=.03, bottomM=.2, zh='前护板', en='front plate')
    boxes = dict(nativeToMeters=1., boxes=dict(post=rec((-.05, 2.95, .24), (.05, 3.05, 1.2), 'low', label='post'),
                                               marker=rec((-1., 1.8, 0), (-.8, 2., .3), 'high', label='marker'),
                                               cart=rec((2.4, 2.6, 0), (3.0, 3.2, .9), 'unverified', label='cart', fitRejected=True)))
    opts = dict(targets=[dict(entityId='post', part='lower edge', edge='bottom', coveredBy=plate_spec)], landmarks=[dict(zh='防撞柱', en='bollard', ids=['bollard'])],
                scaleReference=dict(entityId='post', part='e-stop', name='尺度参照物（本工位为急停）', select=dict(photo=1, pixel=[1442, 1586], provenance='synthetic')),
                keepOut=[['cart', 'marker']], seeThrough=['mesh'], movable=['cart'], names={'bollard': ['防撞柱', 'bollard']}, boxes=boxes,
                knownUnmodelled=[dict(zh='假想围栏', en='made-up fence', footprintM=[[-2.6, 2.9], [-2.5, 2.9], [-2.5, 3.1], [-2.6, 3.1]], heightM=[0, 2])])  # native x -1..-0.9
    try:
        prepare(ctx, dict(opts, landmarks=[dict(zh='料车', en='cart', ids=['cart'])])); raise AssertionError('movable landmark accepted')
    except ValueError:
        pass
    try:
        prepare(ctx, dict(opts, boxes=dict(boxes, nativeToMeters=1.1))); raise AssertionError('boxes of another scale accepted')
    except ValueError:
        pass
    env = prepare(ctx, opts); fl = lambda x, y: env['fr']['to'](np.array([[x, y, 0.]]))[0, :2]  # native -> plan floor frame
    assert env['keepOutExcludedMovable'] == ['cart'] and env['grid']['why'](*fl(1., 2.4)) is None, 'the movable cart must not make a keep-out zone'
    assert env['grid']['why'](*fl(-.9, 1.9)) == 'keepOut' and any(o['id'] == 'plate:post' for o in env['ctx']['objects'])
    status = [env['floorStatus'](*fl(*p)) for p in ((0, .5), (3.2, 2.5), (-3.5, -.5))]
    assert [s[0] for s in status] == ['observed', 'seeThrough', 'unobserved'], status
    assert env['grid']['why'](*fl(-2.1, 2.0)) == 'unreachable', env['grid']['why'](*fl(-2.1, 2.0))  # free floor inside the closed room
    assert env['grid']['why'](*fl(0, 4.25)) == 'blocked' and env['grid']['why'](*fl(-.95, 3.)) == 'blocked'  # the wall; the un-modelled fence's footprint
    Tt = env['targets'][0]; T_s = env['targets'][1]
    for h in HEIGHTS_M:  # the wall hides the post from behind it, at any height
        cb = candidate(env, Tt, *fl(0, 4.7), h); assert Tt['key'] not in cb['counts'] and cb['why'][Tt['key']] == 'notSeen', cb['why']
    cu = candidate(env, Tt, *fl(-2.2, 3.0), 1.5)  # straight through the un-modelled fence: blocked (not just warned)
    assert Tt['key'] not in cu['counts'] and cu['views'][Tt['key']]['seen'] < .5, cu['views'][Tt['key']]['seen']
    assert route(env, fl(0, 2.65)) and 'bollard' in route(env, fl(0, 2.65))[1], route(env, fl(0, 2.65))
    pc = T_s['placementCheck']; assert not pc['blocked'] and pc['over'] == 'bollard' and .1 < pc['clearanceM'] < .25, pc  # ray 1.17 m over a 1.0 m bollard
    # faces measured through their outline (box_faces): a lone post, a camera in front, one to its side, one 53 deg off
    mini = [dict(id=i, label=i, kind='x', mesh=m, masks={}) for i, m in (('p', box((-.05, 2.95, 0), (.05, 3.05, 1.2))), ('blk', box((-.1, 1.6, 0), (0., 1.65, .55))),
                                                                         ('net', box((-.5, 3.6, 0), (1.5, 3.63, 2.))), ('q', box((.4, 3.8, .1), (.6, 3.82, .9))))]
    mbox = dict(p=rec((-.05, 2.95, 0), (.05, 3.05, 1.2), 'unverified', label='p'), q=rec((.4, 3.8, .1), (.6, 3.82, .9), 'unverified', label='q'))
    # photo 1 lens level in front (a high camera would measure the front through its bottom edge: depth -> height), two side views
    mc = [look(np.array(C_, float), np.array([0, 3, .6]), K, W, H, up) for C_ in ((0, 1, .6), (2., 2.9, 1.6), (-1.8, 2.2, 1.6))]
    me = prepare(dict(cams=mc, objects=mini, S=1., floor=(up, 0.), doc={}), dict(boxes=mbox, seeThrough=['net']))
    ev = [evidence(me, c, True) for c in mc]; bp = me['boxes'][0]; fid = {F: i for i, F in enumerate(FACES6)}
    # R1 from the front (lens level): the right and top outlines count (fx 2800 * 1 cm / 2 m = 14 px/cm); the front-left and
    # front-bottom edges are partly behind the block (< 80 % known: left and bottom do not count); depth faces (front / back) count only
    # from side views or from above (depth_credit), so the front view credits neither
    assert all(ev[0]['ok'][0, fid[f]] for f in ('right', 'top')) and abs(ev[0]['s'][0, fid['right']] - 14) < .3 and np.isnan(ev[0]['s'][0, fid['front']]), ev[0]['s'][0]
    assert not ev[0]['ok'][0, fid['left']] and not ev[0]['ok'][0, fid['bottom']] and np.isnan(ev[0]['s'][0, fid['back']]), ev[0]['s'][0]
    assert ev[1]['ok'][0, fid['back']] and ev[2]['ok'][0, fid['back']], (ev[1]['s'][0], ev[2]['s'][0])  # side views: the back edge (depth)
    E = lambda i, f: entry(ev[i], 0, fid[f], i + 1)
    assert level(me, bp, fid['front'], [E(1, 'front'), E(2, 'front')]) == 3 and level(me, bp, fid['front'], [E(1, 'front')]) == 2
    assert E(0, 'front') is None and E(0, 'back') is None and level(me, bp, fid['front'], []) == 0
    b3 = dict(bp, cap=None, low=set(), unpinned=set(), before=[3] * 6, ev0=[[e for e in [E(1, F), E(2, F)] if e] for F in FACES6])
    assert regrade(me, b3)[1]['W'] == 3 and regrade(me, dict(b3, unpinned={fid['front']}))[1]['W'] == 1  # box_faces' rule and caps
    assert regrade(me, dict(b3, ev0=[[e for e in [E(0, F)] if e] for F in FACES6]))[1] == dict(L=1, W=0, H=1, bottom=0), 'the front view: one face of L and H, no depth'

    bm = dict(bp, before=[2] * 6, cap=None, ev0=None)  # medium now, no per-photo evidence: high needs a new view >= 30 deg from every photo
    assert after_level(me, bm, fid['front'], [(12., np.array([0, .5, 1.]), 'n')]) == 2 and after_level(me, bm, fid['front'], [(12., np.array([1.5, 1.5, 1.]), 'n')]) == 3
    assert after_level(me, dict(bm, cap='residual'), fid['front'], [(12., np.array([1.5, 1.5, 1.]), 'n')]) == 2  # a capped box stays
    assert ev[0]['hid'], 'the opaque block hides part of the post seen from photo 1'
    assert any(me['ctx']['objects'][g]['id'] == 'blk' for d in ev[0]['hid'].values() for g in d), ev[0]['hid']
    q_hid = Counter(); [q_hid.update(d) for e in ev for kf, d in e['hid'].items() if kf[0] == 1]
    assert q_hid and me['ctx']['objects'][q_hid.most_common(1)[0][0]]['id'] == 'net', q_hid  # the panel behind the mesh: hidden by the mesh
    for b in me['boxes']:
        b['ev0'], b['evPred'] = None, [[] for _ in FACES6]
    why_q = face_why(me, 1, fid['front'], ev, [], False)
    assert why_q[1].startswith('seen only through the net') and '接触' not in why_q[0], why_q  # no false 'contact'
    # a whole run: the generic standard set from two entrance posts, <= 4 extras, the capped (rejected) box stays
    sopts = dict(opts, standardFrame=dict(left=['marker'], right=['cart'])); Fr = entrance(env, sopts['standardFrame'])
    a1, a2 = (v_anchor(env, Fr, s)[0] + s['distM'] * (math.sin(math.radians(s['az'])) * Fr['x'] + math.cos(math.radians(s['az'])) * Fr['out']) for s in STANDARD[:2])
    assert abs((a1 - Fr['E']) @ Fr['x'] + (a2 - Fr['E']) @ Fr['x']) < 1e-9 and abs((a1 - a2) @ Fr['out']) < 1e-9, 'V1 / V2 mirror each other'
    assert len(STANDARD) <= 10 and v_anchor(env, Fr, STANDARD[6])[1] is T_s and v_anchor(env, Fr, STANDARD[8])[0] is not None
    res = run(ctx, sopts); sm = res['summary']; std = {v['id']: v for v in res['standardSet']}
    assert len(std) == len(STANDARD) and all(v['standable'] in ('yes', 'unknown', 'no') for v in std.values())
    far = standard_view(env, Fr, dict(STANDARD[0], id='Vfar', distM=30.), np.random.default_rng(0))
    assert far['standable'] == 'unknown' and far['standWhy'] == 'outsideBounds' and '能不能站人未知' in instruction(env, far, 'x', 'x')[0], far['standWhy']
    dv = np.asarray(fl(-2.1, 2.0)) - Fr['E']  # a viewpoint inside the closed room: not standable (unreachable)
    vr = standard_view(env, Fr, dict(STANDARD[0], id='Vroom', az=math.degrees(math.atan2(dv @ Fr['x'], dv @ Fr['out'])), distM=float(np.linalg.norm(dv))),
                       np.random.default_rng(0))
    assert vr['standable'] == 'no' and vr['standWhy'] == 'unreachable', vr
    assert len(res['newPhotos']) <= MAX_EXTRAS and res['targets'][0]['photos'], (len(res['newPhotos']), res['targets'][0])
    for p in res['newPhotos']:
        assert p['robustness']['targetsPassRate'] is None or p['robustness']['targetsPassRate'] >= MIN_PASS, p['robustness']
        assert p['robustness']['facesMinMcRate'] >= MIN_PASS and (p['heightM'] >= 1 or p['phoneRangeM'][0] >= LOW_CAM_M[0]), p['robustness']
        assert env['fr']['at'](*p['xyM'], 0)[1] < 4.2 and not (-2.9 <= env['fr']['at'](*p['xyM'], 0)[0] <= -1.3 and 1.2 <= env['fr']['at'](*p['xyM'], 0)[1] <= 2.8)
    cart = [r for r in res['faces'] if r['entityId'] == 'cart']
    assert all(r['after'] == r['before'] and r['whyEn'].startswith('masks disagree with the box') for r in cart), cart
    assert all(CONF.index(r['after']) >= CONF.index(r['before']) for r in res['faces']) and all(r.get('whyZh') for r in res['faces'] if CONF.index(r['after']) <= 1)
    md, cov = res['files']['instructions.md'].decode(), res['files']['coverage.md'].decode()
    assert all(p['instructionZh'] in md for p in res['newPhotos']) and all(v['instructionZh'] in md for v in res['standardSet'] if v['needed'])
    assert cov.count('\n| ') == len(res['faces']) + 1 and all(f'{v["id"]}-preview.jpg' in res['files'] for v in res['standardSet'] if v['needed'])
    cc = res['cameraConvention']; assert cc['passed'] and cc['medianInsideMask']['used'] > .95, cc
    assert not sm['targetsShortAfter'] and res['robustness']['planPassRate'] >= .9, (sm['targetsShortAfter'], res['robustness'])
    assert all(a['instructionZh'] in md for p in res['newPhotos'] for a in p['alternatives'])
    T = res['targets'][0]; assert T['needsLowPhoto'] and abs(T['edgeHeightM'] - .25) < .02 and T['edgeHeightSource'].startswith('model geometry'), T
    # a measured edge below the model's end extends the lowest cross-section; a covered front face is not a target
    Te = resolve(env['ctx'], env['fr'], dict(entityId='post', edge='bottom', edgeHeightM=.1, coveredBy=plate_spec), np.random.default_rng(0),
                 {'post': plate(ctx, env['fr'], dict(entityId='post', coveredBy=plate_spec))})
    h = env['fr']['to'](Te['P'])[:, 2]; assert abs(np.median(h) - .115) < .02 and abs(Te['extendedFromM'] - .24) < .02, (np.median(h), Te['extendedFromM'])
    front, back = (np.abs(Te['N'][:, 1]) > .7) & (Te['P'][:, 1] < 2.96), (np.abs(Te['N'][:, 1]) > .7) & (Te['P'][:, 1] > 3.04)
    assert not front.any() and back.any(), 'the covered front face (towards the photos, -y) must go, the back face stay'
    import json, pickle
    assert pickle.loads(pickle.dumps(res)) == res and not any(type(v).__module__ == 'numpy' for v in _walk(res)), 'results must be plain Python'
    json.dumps({k: v for k, v in res.items() if k != 'files'})
    print('capture plan self-test passed:', sm, [(p['xyM'], p['heightM'], p['robustness']) for p in res['newPhotos']])


if __name__ == '__main__':
    _check()
