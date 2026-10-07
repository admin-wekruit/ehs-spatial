"""Gravity-aligned boxes (盒子) with named faces (面) for every object and every clicked part, on the report's one floor.

Frame: u = the report floor normal n (cameras on +), height = n.X + d (x nativeToMeters, read, never changed). A box is a
yaw and six face positions in that frame: left/right along l, front/back along w, bottom/top along u.
Shape inside the box: the convex footprint of the displayed model at that yaw, normalised to the box and extruded from bottom
to top (a prism whose bounding box is the box). A box-shaped model gives a rectangle, i.e. the box itself; a round one a
circle, so its box is D x D instead of a box squeezed until its corners match a round silhouette. A part uses the rectangle.
Prior: the displayed model's surface (0.5-99.5 percentile extents); for a part, its part-mask pixels lifted onto the model
(first hit) and the report's Pi3X points (2-98 percentile). Yaw candidates: the minimum-area footprint of the model, of the
Pi3X points inside the object's eroded masks, and the yaw facing the cameras; the silhouette cost picks one, candidates within
2 % of the best going to the smallest prior footprint (the footprint follows the model at any yaw, so silhouettes barely see
yaw: the minimum-area rectangle decides); a part keeps its object's yaw.
Fit: the six faces so that the projected shape (convex hull of its vertices) matches the object's mask (holes filled) in
every photo with a mask: cost = mask pixels outside + shape pixels on 'known' non-object pixels (L1: each silhouette edge
settles on the median of the mask boundary along it). The object's own surface along a ray is its displayed model's first
hit there, else the near side of the box + 5 cm. 'Unknown' pixels count for neither: off the frame; another object's mask
(dilated 10 px) unless that object's model / Pi3X depth there is not in front of ours (margin 5 cm + 3 % of the range);
another displayed model (same margin) or a non-floor Pi3X point (10 cm + 5 %) in front of ours (an occluder; with the
model as reference also a cart inside a U-shaped guard's box); for a part, the rest of its object's mask (an internal
boundary). Floor points never occlude. An occluded lower boundary therefore only bounds the bottom from below (the box must contain the mask) and never sets it: the lower edge
of docs/workcell-photo/LOWER-EDGE.md (lower boundary along verticals, occlusion boundaries excluded) as a silhouette edge.
Best-improvement pattern search over the faces from the prior, weak pull to it (1 px^2 per cm), a bounded Powell polish for faces that trade
off (depth against height along the rays), occlusion recomputed for the fitted box and the fit repeated.
Evidence (R1, shared with capture_plan.py: sides / outline_px / face_grade / dim_grade / grade_box): a photo credits a face's
OUTLINE when one of the face's edges on the projected silhouette (samples every 3 px on hull edges whose both ends lie on the
face, membership >= 0.5: on a box, the face's own edges where it meets a face turned the other way; consecutive edges turning
< 25 deg are one side, a face's samples on one side one edge) is >= 80 % known (in frame, the pixel 3 px outside known: not
occluded / unknown) with >= 20 px known, at >= 10 px/cm = fx * 1 cm / the edge's straight-line distance (median); the face's
px/cm in that photo = its best such edge. Whether the face itself is seen ('photos' / status: facing, >= half of a 6x6 grid on it
in frame and not unknown) is reported, not counted.
Confidence: a face: high = credited in >= 2 photos >= 30 deg apart (horizontal directions from its centre), medium = >= 1, low =
only < 10 px/cm, unverified = none. A dimension (L: left / right, W: front / back, H: bottom / top): high = both faces high,
medium = both >= medium, low = only one credited or only < 10 px/cm, unverified = neither; bottom clearance = the bottom face
(its lower outline). Depth W is bounded by front and back: a front view credits the back only where the back edge is on the
outline (from above, or from the side). The object's confidence = the lowest of L, H and bottom (W is reported in dims but not
counted: a W that is low or unverified puts '进深需要侧面照片' in the front / back faces' need).
Uncertainty: each silhouette side of each photo with >= 20 px known is one measurement of its mean outward motion (J, image motion
normal to the outline per cm of each face) with the fixed margin MARGIN_PX = 4 px (half a report mask's 7.3-7.8 px
canonical pixel); Sigma = MARGIN_PX^2 (J'J + (MARGIN_PX / 10 cm)^2 I)^-1 (10 cm: the displayed model's placement as prior),
so faces that compensate each other share it (one photo: depth and height along its rays). Per face sqrt(Sigma_ff); a size
(outward motion of both faces) var = Sigma_aa + Sigma_bb + 2 Sigma_ab; bottom = the bottom face. Unverified faces and
dimensions, faces the fit cannot pin, the dimensions they bound and capped fits carry sigma null.
No collapse: a face the photos do not pin (no known silhouette >= 20 px, or sigma >= 5 cm: less than half the prior removed)
keeps the prior size (both faces of a dimension: the prior faces; one: the prior extent from the pinned face; the other faces keep
their fit, a refit would make them absorb a wrong prior); it and every dimension it bounds are at most 'low' (sigma null).
Caps (R3): a mean mismatch of the fitted box (cost / known silhouette length, before the prior-size reset) > 12 px = 3 x the
margin, a box covering < 80 % of the mask in any masked photo, or a rejected fit (< 50 %: fitRejected) is capped: the displayed
model's box (the prior) is kept, never a flattened or partial fitted box; every face and dimension at most 'low', sigma null, no
snapping, highlight reason '掩码与盒子不符（N px）：尺寸取模型' (N = the fitted box's mean mismatch). A fitted bottom > 3 cm below
the floor puts the bottom at most 'low'.
Pixels in both this object's mask and another's count for neither (two masks claiming a pixel say nothing).
Floor contact (snapping): only an uncapped fit with -3 cm <= bottom <= 3 cm and, in >= 1 photo, >= half of the lower
silhouette's known samples with floor 2-6 px below (the floor entity's mask, geometryRole 'floor', or a Pi3X point within
3 cm of the floor): bottomM = 0 (the top stays; a box whose top is not above the floor, a flat marking, moves up whole),
snapNote '贴地：盒子上移 / 下移 x cm' (research: snappedCm = the fitted bottom it replaced); never snapped otherwise.
modelBottomCm (research) is the displayed model's own lowest point.
Faces: 'front' is the side face whose normal is closest to the mean horizontal direction from the box to ALL the cell's
cameras; within 0.05 of the next, the one facing the entrance axis (the reverse of the photos' mean horizontal viewing
direction: they are taken from the entrance); left / right as seen looking at the front. axes = [l (left->right),
w (front->back), u (up)], right-handed; faceNormals = each face's outward unit normal (native).
Need (per face, null when high; CAPTURE-PROTOCOL.md): a photo without a usable mask that still sees the face (facing, half
in frame, in front of the other models) asks for a mask / click, not a photo; the bottom face is never asked for, its lower
edge is, with a crouched photo when it is below 0.5 m and no evidence photo was taken from 0.4-0.8 m at <= 15 deg down;
depth asks for a side view 45-75 deg off the face's normal; occluded outline edges ask to step sideways; caps ask for masks first.
Highlight (CONTRACT): L, H or bottom low / unverified, or the object's layer confidence low / unverified, or its layer
'missing' list naming depth / placement / model bottom (深度 / 摆放 / 位置 / 模型最低点 / 模型底部 / 模型下部); highlightReasons in
Chinese. Only objects with highlight false are smoothed. W alone never highlights.
Output: boxes = the CONTRACT records (objects only); research = the same with the evidence behind them (faceEvidence pxPerCm =
R1 per photo, unrounded; gradeCaps: what capture_plan needs to regrade) and the parts (named part boxes stay research only: the
part's own mask pixels, not clipped to the object's mask).
Generic: no object-type rules; parts are click inputs (opts partMasks {entityId: {part: {photoIndex0: polygons}}}, from
modal_apps/workcell_part_masks.py). opts: api (point maps), partMasks, draw (bool, box overlays), only (entity id prefixes),
neutralPrior (entity ids shown as their measured box: the prior is neutral, a square footprint whose depth equals the visible
width, never the rejected candidate's shape).
Dominant plane: when one near-vertical plane holds >= 50 % of the object's own mask points (within 5 cm, RANSAC), the yaw is that
plane's (a fence's long axis), not the silhouettes' near-tie. Pinned: a horizontal size (L or W) whose box leaves > 10 % of the
object's own mask points outside it by > 3 cm is not pinned, whatever the silhouettes say (a rectangle fitted to a round post):
it keeps the prior size and is at most low (so W is 'high' only when it is really pinned). research.pointContainment reports it.
English: highlightReasonsEn, snapNoteEn and faces[f].needEn next to the Chinese texts.
Runs in the Modal check container (modal_apps/workcell_view_checks.py --checks box_faces) or on-prem; no arguments = self-test."""
import math
import sys
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np

try:
    import shape_core as wsc  # the check container (modal_apps/workcell_view_checks.py)
except ImportError:  # locally / on-prem: scripts/workcell_shape_check.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import workcell_shape_check as wsc
from workcell_checks import clearance as cl

GAP_PX, REST_PX, STRIDE = 10, 2, 4  # other masks dilated 10 px (report masks leave 7-10 px gaps), a part's rest 2 px; 4 px grid
ENLARGE_M, MARGIN_M, MARGIN_REL, PI3X_M, PI3X_REL = .05, .05, .03, .10, .05  # unknown: > 5 cm + 3 % in front of the object (Pi3X 10 cm + 5 %)
FLOOR_TOL_M, CONTACT_M = .03, .03  # a floor Pi3X point is within 3 cm of the floor; a bottom <= 3 cm may touch it
BOUND_M, BOUND_REL, BELOW_FLOOR_M = .30, .5, .15  # faces move within prior +- max(30 cm, half the extent); bottom >= -15 cm
MU, MARGIN_PX, EVID_STEP, EVID_MIN_PX, PRIOR_CM, LOOSE_CM, SIDE_DEG, RESID_PX, MIN_COVER = 1.0, 4.0, 3.0, 20.0, 10.0, 5.0, 25.0, 12.0, .5
HIGH_PX_CM, HIGH_DEG, YAW_TIE = 10.0, 30.0, .02  # yaw candidates within 2 % of the best cost tie
EDGE_FRAC, COVER_CAP = .8, .8  # R1: an outline edge counts when >= 80 % of it is known; R3: a box covering < 80 % of a mask is capped
STEPS_CM = (16, 8, 4, 2, 1, .5, .25)
GRADES = ('unverified', 'low', 'medium', 'high')
SQUARE = np.array([[0, 0], [1, 0], [1, 1], [0, 1.]])
# CAPTURE-PROTOCOL.md: lower edges below 0.5 m need one crouched photo (camera 0.4-0.8 m above the floor, <= 15 deg down to
# the edge); depth needs a side view 45-75 deg off the face's normal (need texts); face names within 0.05 tie
LOW_EDGE_M, CROUCH_M, CROUCH_DEG, NAME_TIE = .5, (.4, .8), 15., .05
GEOMETRY_WORDS = ('深度', '摆放', '位置', '模型最低点', '模型底部', '模型下部')  # the layer's words for depth / placement / model-bottom items
ZH = dict(front='前面', back='后面', left='左面', right='右面', top='顶面', bottom='底面')
DIM_ZH, LEVEL_ZH = dict(L='长度', W='进深', H='高度', bottom='离地高度'), dict(high='高', medium='中', low='低', unverified='未验证')
DIM_EN = dict(L='length', W='depth', H='height', bottom='clearance')
PLANE_TOL_M, PLANE_MAJORITY, PLANE_VERTICAL = .05, .5, .5  # one plane holds most of the points: >= 50 % within 5 cm (two-view MVS noise at 3-6 m; 5 % of the extent if less), |n . up| < 0.5
CONTAIN_TOL_M, CONTAIN_OUT = .03, .10  # a box contains its own mask's points: <= 10 % outside it by > 3 cm along a horizontal axis
CROUCH = ('；下沿离地 < 0.5 m：至少 1 张蹲下拍（手机离地 0.5–0.6 m，镜头放平，下沿在屏幕上不低于下面那条三分线）',
          '; the lower edge is below 0.5 m: crouch for at least one (phone 0.5-0.6 m above the floor, lens level, the edge not below the lower third line)')


def size_sigma(cov, a, b):
    """sigma (cm) of a size = outward motion of face a + outward motion of face b: var = caa + cbb + 2 cab."""
    return math.sqrt(max(0., cov[a, a] + cov[b, b] + 2 * cov[a, b]))


# R1, shared with capture_plan.py (it imports these, so both grade the same way)
SIDE_W_DEG = 45  # depth faces: side views only (or from above)


def sides(poly):
    """Silhouette side of each hull segment poly[e] -> poly[e + 1]: consecutive segments turning < SIDE_DEG from the side's first
    (segments under 0.5 px keep the current side)."""
    out, side, start = [], 0, None
    for e in range(len(poly)):
        d = poly[(e + 1) % len(poly)] - poly[e]
        if float(np.hypot(*d)) >= .5:
            a = math.atan2(d[1], d[0])
            if start is None:
                start = a
            elif abs(math.remainder(a - start, 2 * math.pi)) > math.radians(SIDE_DEG):
                side, start = side + 1, a
        out.append(side)
    return out


def outline_px(side, member, infr, known, ln, dist_m, fx):
    """One photo's outline credit per face. Silhouette samples (rows): side (sides()), member (n x 6 bool: on the face), infr (in
    frame), known (in frame, not occluded / unknown, nor the pixel 3 px outside), ln (length px), dist_m (straight-line distance from
    the camera, m). A face's edge = its samples on one side; it counts when >= 80 % of its length is known and >= 20 px; its px/cm =
    fx * 1 cm / its median known distance. Returns ({face: best px/cm of its counting edges}, {face: 'out_of_frame' | 'occluded'} for
    faces with an outline edge >= 20 px but none counting)."""
    px, why = {}, {}
    for f in range(6):
        for s in np.unique(side[member[:, f]]):
            e = member[:, f] & (side == s); L, Lk = float(ln[e].sum()), float(ln[e & known].sum())
            if Lk >= EVID_MIN_PX and Lk >= EDGE_FRAC * L:
                px[f] = max(px.get(f, 0.), float(fx * .01 / np.median(dist_m[e & known])))
            elif L >= EVID_MIN_PX:
                why.setdefault(f, 'out_of_frame' if float(ln[e & infr].sum()) < EDGE_FRAC * L else 'occluded')
        if f in px:
            why.pop(f, None)
    return px, why


def hdir(n, d):
    """Horizontal unit direction of d (floor normal n removed)."""
    d = d - (d @ n) * n
    return d / max(float(np.linalg.norm(d)), 1e-12)


def depth_credit(n, w_axis, C, P, top_h, cam_h):
    """R1, depth: a photo counts for a depth-bounding face (front / back) only from a side view >= SIDE_W_DEG off the front-back axis
    (horizontal directions) or from above the box top; a near-frontal view sees those outlines only along its own rays."""
    if cam_h > top_h:
        return True
    return math.degrees(math.acos(min(1., abs(float(hdir(n, C - P) @ hdir(n, w_axis)))))) >= SIDE_W_DEG


def face_grade(px, dirs):
    """A face from {photo: px/cm} (outline_px) and {photo: horizontal unit direction from the face centre to the camera}: high =
    >= 2 photos >= 10 px/cm >= 30 deg apart, medium = >= 1, low = only below 10 px/cm, unverified = none. (grade, good, apart deg)."""
    good = sorted(k for k, s in px.items() if s >= HIGH_PX_CM)
    if not good:
        return ('low' if px else 'unverified'), good, 0.
    apart = max((math.degrees(math.acos(float(np.clip(dirs[a] @ dirs[b], -1, 1)))) for a, b in combinations(good, 2)), default=0.)
    return ('high' if apart >= HIGH_DEG else 'medium'), good, apart


def dim_grade(*g):
    """A dimension from its bounding faces' grades (the bottom clearance: the bottom face alone): both high -> high, both >= medium
    -> medium, any outline evidence (one face credited, or only < 10 px/cm) -> low, none -> unverified."""
    lo = min(g, key=GRADES.index)
    return lo if GRADES.index(lo) >= 2 else 'low' if any(x != 'unverified' for x in g) else 'unverified'


def grade_box(px, dirs, dims, low=(), unpinned=()):
    """R1 with the caps for one box: px {face: {photo: px/cm}}, dirs {face: {photo: direction}}, dims {dim: its faces}; low = faces at
    most low (a capped box: all; a bottom below the floor; unpinned faces), unpinned = faces placed by the prior size (the dimensions
    they bound at most low). Returns ({face: (grade, good, apart)}, {dim: grade})."""
    cap = lambda g, c: min(g, 'low', key=GRADES.index) if c else g
    face = {f: (lambda r: (cap(r[0], f in low),) + r[1:])(face_grade(px[f], dirs[f])) for f in px}
    return face, {d: cap(dim_grade(*(face[f][0] for f in fs)), bool(set(fs) & set(unpinned))) for d, fs in dims.items()}


def dominant_plane(P, tol, seed=7, iterations=300):
    """RANSAC plane (unit normal n, offset d with n.X + d = 0) through the most points within tol, SVD refit on them, and that
    inlier fraction; None below 50 points."""
    if len(P) < 50:
        return None
    rng = np.random.default_rng(seed); best = None
    for _ in range(iterations):
        a, b, c = P[rng.choice(len(P), 3, replace=False)]; nn = np.cross(b - a, c - a)
        if np.linalg.norm(nn) < 1e-12:
            continue
        nn /= np.linalg.norm(nn); inl = np.abs((P - a) @ nn) < tol
        if best is None or inl.sum() > best.sum():
            best = inl
    c = P[best].mean(0); nn = np.linalg.svd(P[best] - c, full_matrices=False)[2][-1]
    return nn, float(-nn @ c), float((np.abs(P @ nn - nn @ c) < tol).mean())


def plane_yaw(t, P):
    """The box yaw from the object's dominant plane, when one near-vertical plane holds most of its points (PLANE_*: within 5 cm,
    or 5 % of the object's horizontal extent when that is less: a small box's two faces are not one plane), else None."""
    if len(P) < 50:
        return None, None
    flat = np.c_[(P - t.O) @ t.e1, (P - t.O) @ t.e2]
    extent = float(np.linalg.norm(np.percentile(flat, 95, 0) - np.percentile(flat, 5, 0)))
    pl = dominant_plane(P, min(PLANE_TOL_M / t.S, .05 * extent))
    if pl is None or pl[2] < PLANE_MAJORITY or abs(float(pl[0] @ t.n)) >= PLANE_VERTICAL:
        return None, pl
    d = np.cross(t.n, pl[0])  # the plane's horizontal direction
    return math.atan2(float(d @ t.e2), float(d @ t.e1)) % (math.pi / 2), pl


def basis(n):
    a = np.eye(3)[int(np.argmin(np.abs(n)))]; e1 = a - (a @ n) * n; e1 /= np.linalg.norm(e1)
    return e1, np.cross(n, e1)


def axes_of(theta, n, e1, e2):
    l = math.cos(theta) * e1 + math.sin(theta) * e2
    return l, np.cross(n, l), n


def coords(X, O, ax):
    return np.c_[(X - O) @ ax[0], (X - O) @ ax[1], (X - O) @ ax[2]]


def extents(P, lo=.5, hi=99.5):
    q = np.percentile(P, [lo, hi], axis=0)
    return np.array([q[0, 0], q[1, 0], q[0, 1], q[1, 1], q[0, 2], q[1, 2]])


def corners(x, O, ax):
    """The 8 box corners, index = 4 ia + 2 ib + ih."""
    return np.array([O + x[i >> 2] * ax[0] + x[2 + ((i >> 1) & 1)] * ax[1] + x[4 + (i & 1)] * ax[2] for i in range(8)])


def normals(ax):
    l, w, u = ax
    return np.array([-l, l, -w, w, -u, u])  # faces a0 a1 b0 b1 h0 h1 (internal; named front/back/left/right at the end)


def footprint(c, x):
    """Convex footprint of points c (box coordinates) inside the extents x, normalised to the unit square (a rectangle -> SQUARE)."""
    sel = (c[:, 0] >= x[0]) & (c[:, 0] <= x[1]) & (c[:, 1] >= x[2]) & (c[:, 1] <= x[3])
    if sel.sum() < 10 or x[1] - x[0] <= 0 or x[3] - x[2] <= 0:
        return SQUARE
    q = np.c_[(c[sel, 0] - x[0]) / (x[1] - x[0]), (c[sel, 1] - x[2]) / (x[3] - x[2])].astype(np.float32)
    h = cv2.approxPolyDP(cv2.convexHull(q), .01, True).reshape(-1, 2).astype(float)
    if len(h) < 3:
        return SQUARE
    h = (h - h.min(0)) / np.maximum(h.max(0) - h.min(0), 1e-9)  # the shape's own bounding box is the box
    return h


def prism(x, fp, O, ax):
    """Vertices of the footprint fp extruded over the box x, index = 2 v + (top)."""
    a = x[0] + fp[:, 0] * (x[1] - x[0]); b = x[2] + fp[:, 1] * (x[3] - x[2])
    base = O + a[:, None] * ax[0] + b[:, None] * ax[1]
    return np.stack([base + x[4] * ax[2], base + x[5] * ax[2]], 1).reshape(-1, 3)


def weights(fp):
    """How far each prism vertex moves per unit outward motion of each face (a0 a1 b0 b1 h0 h1)."""
    v = np.repeat(fp, 2, 0); top = np.tile([0., 1.], len(fp))
    return np.c_[1 - v[:, 0], v[:, 0], 1 - v[:, 1], v[:, 1], 1 - top, top]


def hull(cam, X):
    """Image polygon (convex hull of the projected vertices, their indices in order), or None if a vertex is behind the camera."""
    uv, z = wsc.project(cam, X)
    if (z <= 1e-3).any():
        return None, None
    idx = cv2.convexHull(uv.astype(np.float32), returnPoints=False).ravel()
    return uv[idx], idx


def box_span(C, D, x, O, ax):
    """Entry and exit depth of each unit ray through the box (inf, inf = miss)."""
    o = coords(C[None], O, ax)[0]; d = np.c_[D @ ax[0], D @ ax[1], D @ ax[2]]
    lo, hi = x[[0, 2, 4]], x[[1, 3, 5]]
    with np.errstate(divide='ignore', invalid='ignore'):
        t1, t2 = (lo - o) / d, (hi - o) / d
    t1 = np.where(np.isnan(t1), -np.inf, t1); t2 = np.where(np.isnan(t2), np.inf, t2)
    tmin, tmax = np.minimum(t1, t2).max(1), np.maximum(t1, t2).min(1)
    hit = tmax >= np.maximum(tmin, 0)
    return np.where(hit, np.maximum(tmin, 0), np.inf), np.where(hit, tmax, np.inf)


def pi3x_points(frames, k, u, v):
    """Pi3X world point under pixels (u, v) of photo k (NaN rows where none)."""
    out = np.full((len(u), 3), np.nan); f = next((f for f in frames if f['camera'] == k), None)
    if f is None or not len(u):
        return out
    A = cv2.invertAffineTransform(f['affine'].astype(np.float64)); H, W = f['P'].shape[:2]
    gj = np.round(A[0, 0] * u + A[0, 1] * v + A[0, 2]).astype(int); gi = np.round(A[1, 0] * u + A[1, 1] * v + A[1, 2]).astype(int)
    ok = (gj >= 0) & (gi >= 0) & (gj < W) & (gi < H); P = f['P'][gi[ok], gj[ok]]
    good = np.isfinite(P).all(1) & (np.abs(P).sum(1) > 0); idx = np.nonzero(ok)[0][good]; out[idx] = P[good]
    return out


def fill_holes(mask):
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros(mask.shape, np.uint8); cv2.drawContours(out, cs, -1, 1, -1)
    return out.astype(bool) | mask


class View:
    """One photo of one target: crop, region mask M, unknown map U, and the row prefix sums of W (+1 known outside, -1 mask)."""

    def __init__(self, k, cam, region, rest, contested):
        self.k, self.cam, self.full, self.region_full, self.rest_full, self.contested_full = k, cam, region.shape, region, rest, contested
        ys, xs = np.nonzero(region); self.box0 = (xs.min(), ys.min(), xs.max(), ys.max())

    def set_crop(self, polys):
        """Crop = the region's box and the bound boxes' polygons, inside the frame, + 20 px."""
        h, w = self.full; x0, y0, x1, y1 = self.box0
        for p in polys:
            x0, y0 = min(x0, p[:, 0].min()), min(y0, p[:, 1].min()); x1, y1 = max(x1, p[:, 0].max()), max(y1, p[:, 1].max())
        self.x0, self.y0 = int(max(0, math.floor(x0) - 20)), int(max(0, math.floor(y0) - 20))
        self.x1, self.y1 = int(min(w, math.ceil(x1) + 21)), int(min(h, math.ceil(y1) + 21))
        self.M = fill_holes(self.region_full[self.y0:self.y1, self.x0:self.x1])
        self.C = self.contested_full[self.y0:self.y1, self.x0:self.x1] & self.M  # also in another object's mask: says nothing
        self.M &= ~self.C

    def set_unknown(self, U):
        self.U = (U | self.C) & ~self.M
        self.W = np.where(self.M, -1., np.where(self.U, 0., 1.)).astype(np.float32)
        self.cum = np.zeros((self.W.shape[0], self.W.shape[1] + 1)); np.cumsum(self.W, axis=1, out=self.cum[:, 1:])
        self.base = float(self.M.sum())

    def cost(self, poly):
        """Mask outside the convex polygon + polygon on known non-mask pixels (4 sub-rows per pixel row, exact along rows)."""
        if poly is None:
            return self.base
        p = poly - [self.x0, self.y0]; h, w = self.W.shape
        r0, r1 = max(0, int(math.floor(p[:, 1].min() + .5))), min(h - 1, int(math.ceil(p[:, 1].max() - .5)))
        if r1 < r0:
            return self.base
        J = np.repeat(np.arange(r0, r1 + 1), 4); Y = J + np.tile([-.375, -.125, .125, .375], r1 - r0 + 1)
        q = np.roll(p, -1, axis=0); dy = q[:, 1] - p[:, 1]; keep = np.abs(dy) > 1e-9
        a, b, dy = p[keep], q[keep], dy[keep]
        t = (Y[:, None] - a[None, :, 1]) / dy[None]; ok = (t >= 0) & (t <= 1); X = a[None, :, 0] + t * (b[None, :, 0] - a[None, :, 0])
        xl = np.clip(np.where(ok, X, np.inf).min(1), -.5, w - .5); xr = np.clip(np.where(ok, X, -np.inf).max(1), -.5, w - .5)
        sel = xr > xl; J, xl, xr = J[sel], xl[sel], xr[sel]

        def F(x):
            s = x + .5; i = np.minimum(np.floor(s).astype(int), w - 1)
            return self.cum[J, i] + (s - i) * self.W[J, i]
        return self.base + float((F(xr) - F(xl)).sum()) * .25


class Target:
    """One object or one clicked part: its views, prior points, shape points, and the current yaw / prior / bound / footprint."""

    def __init__(self, ctx, obj, gid, part, regions, rests, frames, scene, floor_masks, others, contested, prior_pts, shape_pts):
        self.ctx, self.obj, self.gid, self.part, self.frames, self.scene = ctx, obj, gid, part, frames, scene
        self.floor_masks, self.others, self.prior_pts, self.shape_pts = floor_masks, others, prior_pts, shape_pts
        n, d = ctx['floor']; self.n, self.O, self.S = n, -d * n, ctx['S']; self.e1, self.e2 = basis(n)
        self.views = [View(k, ctx['cams'][k], regions[k], rests.get(k), regions[k] & contested[k]) for k in sorted(regions)]
        self.masked = set(regions); self.neutral = False; self.own_pts = np.zeros((0, 3))  # photos with a usable mask (the rest may still see it: 'add a mask', not 'take a photo')

    def set_yaw(self, theta):
        """Axes, prior extents, bound and footprint at this yaw; returns (prior, bound)."""
        S = self.S; self.theta = theta; self.ax = axes_of(theta, self.n, self.e1, self.e2)
        prior = extents(coords(self.prior_pts, self.O, self.ax), *((2, 98) if self.part else (.5, 99.5)))
        size = prior[1::2] - prior[0::2]; R = np.maximum(BOUND_M / S, BOUND_REL * size)
        bound = prior + np.repeat(R, 2) * np.tile([-1, 1], 3); bound[4] = max(bound[4], -BELOW_FLOOR_M / S)
        if self.neutral:  # a measured-box stand-in: a neutral prior, never the rejected candidate's shape or depth: a square
            cdir = np.mean([v.cam['C'] for v in self.views], 0) - self.prior_pts.mean(0)  # footprint whose depth (the axis
            k = int(abs(cdir @ self.ax[1]) > abs(cdir @ self.ax[0]))  # facing the cameras) equals the visible width
            mid, half = (prior[2 * k] + prior[2 * k + 1]) / 2, (prior[2 * (1 - k) + 1] - prior[2 * (1 - k)]) / 2
            prior = prior.copy(); prior[2 * k], prior[2 * k + 1] = mid - half, mid + half
            size = prior[1::2] - prior[0::2]; R = np.maximum(BOUND_M / S, BOUND_REL * size)
            bound = prior + np.repeat(R, 2) * np.tile([-1, 1], 3); bound[4] = max(bound[4], -BELOW_FLOOR_M / S)
        self.fp = SQUARE if self.neutral else footprint(coords(self.shape_pts, self.O, self.ax), prior) if self.shape_pts is not None else SQUARE
        self.wts = weights(self.fp)
        return prior, bound

    def cost(self, x, prior):
        X = prism(x, self.fp, self.O, self.ax)
        return sum(v.cost(hull(v.cam, X)[0]) for v in self.views) + MU * float(np.abs(x - prior).sum()) * self.S * 100

    def unknown(self, v, x, bound):
        """Unknown pixels of view v's crop for the box x (see the module doc), on a STRIDE grid, upsampled."""
        cam, k, S = v.cam, v.k, self.S; C = cam['C']
        gx, gy = np.meshgrid(np.arange(v.x0 + STRIDE / 2, v.x1, STRIDE), np.arange(v.y0 + STRIDE / 2, v.y1, STRIDE))
        u, vv = gx.ravel(), gy.ravel(); D = cl.rays(cam, u, vv)
        t_own, t_oth = cl.hits(self.scene, C, D, self.gid)
        t_ref = box_span(C, D, x + np.array([-1, 1, -1, 1, -1, 1]) * ENLARGE_M / S, self.O, self.ax)[0]
        far = ~np.isfinite(t_ref); t_ref[far] = box_span(C, D[far], bound, self.O, self.ax)[0]
        t_ref = np.where(np.isfinite(t_own), t_own, t_ref)  # the object's own surface there: its model, else the box's near side
        P = pi3x_points(self.frames, k, u, vv); r = np.linalg.norm(P - C, axis=1)
        with np.errstate(invalid='ignore'):
            floorish = np.abs((P - self.O) @ self.n) * S <= FLOOR_TOL_M
            m = MARGIN_M / S + MARGIN_REL * t_ref; mp = PI3X_M / S + PI3X_REL * t_ref
            r_np = np.where(floorish | ~np.isfinite(r), np.inf, r)
            front = (t_oth < t_ref - m) | (r_np < t_ref - mp)
            depth = np.minimum(t_oth, np.where(np.isfinite(r), r, np.inf)); behind = np.isfinite(depth) & (depth >= t_ref - m)
        oth = cv2.resize(self.others[k][v.y0:v.y1, v.x0:v.x1].astype(np.uint8), gx.shape[::-1], interpolation=cv2.INTER_AREA) > 0
        U = (front | (oth.ravel() & ~behind)).reshape(gx.shape)  # another mask: unknown unless that object lies behind ours
        U = cv2.resize(U.astype(np.uint8), (v.x1 - v.x0, v.y1 - v.y0), interpolation=cv2.INTER_NEAREST).astype(bool)
        if v.rest_full is not None:  # a part: the rest of its object is an internal boundary
            U |= cv2.dilate(v.rest_full[v.y0:v.y1, v.x0:v.x1].astype(np.uint8), np.ones((2 * REST_PX + 1,) * 2, np.uint8)) > 0
        return U

    def prepare(self, x, bound, crop_boxes):
        """Crops from the bound boxes of every yaw to be tried (crop_boxes: [(bound, axes)]), then the occlusion map for box x."""
        bp = [[hull(v.cam, corners(b, self.O, a))[0] for b, a in crop_boxes] for v in self.views]
        keep = [all(p is not None for p in ps) for ps in bp]  # a bound corner behind the camera: the photo is not used
        self.views = [v for v, k in zip(self.views, keep) if k]
        for v, ps in zip(self.views, [ps for ps, k in zip(bp, keep) if k]):
            v.set_crop(ps)
        self.refresh(x, bound)

    def refresh(self, x, bound):
        for v in self.views:
            v.set_unknown(self.unknown(v, x, bound))

    def fit(self, x, prior, bound, sweeps=6, polish=False):
        """Best-improvement pattern search (all 12 single-face moves at each step size, the best taken; a greedy face order lets
        a weak depth face absorb a missing bottom) within the bound, faces >= 1 cm apart; optionally a bounded Powell polish."""
        x = x.copy(); best = self.cost(x, prior); gap = .01 / self.S
        for _ in range(sweeps):
            start = best
            for step in STEPS_CM:
                s = step / 100 / self.S
                while True:
                    move = None
                    for f in range(6):
                        for sign in (1, -1):
                            y = x.copy(); y[f] += sign * s
                            if (f % 2 == 0 and (y[f] < bound[f] or y[f] > y[f ^ 1] - gap)) or (f % 2 == 1 and (y[f] > bound[f] or y[f] < y[f ^ 1] + gap)):
                                continue
                            c = self.cost(y, prior)
                            if c < best - 1e-6 and (move is None or c < move[0]):
                                move = (c, y)
                    if move is None:
                        break
                    best, x = move
            if best > start - 1:
                break
        if polish:  # conjugate directions for faces that trade off against each other, then one more pattern sweep
            from scipy.optimize import minimize
            pen = lambda y: self.cost(y, prior) + (1e9 if (y[1::2] - y[0::2] < gap).any() else 0)
            box = [(min(bound[f & ~1], x[f]), max(bound[f | 1], x[f])) for f in range(6)]  # inside the bound box; order by the penalty
            r = minimize(pen, x, method='Powell', bounds=box, options=dict(xtol=5e-4, ftol=1e-6, maxfev=4000))
            if r.fun < best - 1e-6:
                x, best = np.asarray(r.x, float), float(r.fun)
            return self.fit(x, prior, bound, sweeps=1)
        return x, best


def minrect_yaw(P2):
    if len(P2) < 5:
        return None
    return math.radians(cv2.minAreaRect(P2.astype(np.float32))[2]) % (math.pi / 2)


def lift(t, k, region, erode=12):
    """Pi3X points inside the eroded region of photo k, off the floor."""
    f = next((f for f in t.frames if f['camera'] == k), None)
    if f is None:
        return np.zeros((0, 3))
    H, W = f['P'].shape[:2]; jj, ii = np.meshgrid(np.arange(W), np.arange(H)); A = f['affine']
    ui = np.round(A[0, 0] * jj + A[0, 1] * ii + A[0, 2]).astype(int); vi = np.round(A[1, 0] * jj + A[1, 1] * ii + A[1, 2]).astype(int)
    m = cv2.erode(region.astype(np.uint8), np.ones((2 * erode + 1,) * 2, np.uint8)) > 0
    ok = (ui >= 0) & (vi >= 0) & (ui < m.shape[1]) & (vi < m.shape[0]); ok[ok] = m[vi[ok], ui[ok]]
    P = f['P'][ok]; P = P[np.isfinite(P).all(1) & (np.abs(P).sum(1) > 0)]
    return P[(P - t.O) @ t.n * t.S > FLOOR_TOL_M]


def model_lift(t, k, region, step=8):
    """The region's pixels (every 8 px) cast onto the object's own displayed model (first hit)."""
    ys, xs = np.nonzero(region[::step, ::step]); cam = t.ctx['cams'][k]
    D = cl.rays(cam, xs * step + .0, ys * step + .0); own = cl.hits(t.scene, cam['C'], D, t.gid)[0]; ok = np.isfinite(own)
    return cam['C'] + own[ok, None] * D[ok]


def evidence(t, x):
    """R1 per face {photo: px/cm} (outline_px: its outline edges on the silhouette, membership >= 0.5); per photo {face: why no
    edge counts}; per face {photo: known silhouette px} where >= 20 px (outline evidence: 'no collapse'); per photo the lower
    silhouette's known samples (pixels, outward normals) for floor contact; the J rows (one per silhouette side and photo, the motion
    normal to the outline per cm of each face) for the covariance."""
    X = prism(x, t.fp, t.O, t.ax); N = normals(t.ax); out = {f: {} for f in range(6)}; why = {}; seen_px = {f: {} for f in range(6)}
    low = {}; rows = []; t.known_px = {}
    for v in t.views:
        poly, idx = hull(v.cam, X)
        if poly is None:
            continue
        q2 = np.roll(poly, -1, axis=0); ccw = np.sign(float((poly[:, 0] * q2[:, 1] - q2[:, 0] * poly[:, 1]).sum()))
        smp = []; sd = sides(poly); fh, fw = v.full
        for e in range(len(idx)):
            i, j = int(idx[e]), int(idx[(e + 1) % len(idx)]); p, q = poly[e], poly[(e + 1) % len(idx)]; L = float(np.linalg.norm(q - p))
            if L < .5:
                continue
            ns = max(1, int(round(L / EVID_STEP))); s = ((np.arange(ns) + .5) / ns)[:, None]
            pts = p + s * (q - p); X3 = X[i] + s * (X[j] - X[i]); w = np.repeat(np.minimum(t.wts[i], t.wts[j])[None], ns, 0)
            w_move = (1 - s) * t.wts[i] + s * t.wts[j]; base = wsc.project(v.cam, X3)[0]
            tang = (q - p) / L; nu = ccw * np.array([tang[1], -tang[0]])  # outward image normal
            o = np.round(pts + 3 * nu).astype(int) - [v.x0, v.y0]; h, wd = v.U.shape  # the crop lies inside the frame
            infr = (pts[:, 0] >= 0) & (pts[:, 1] >= 0) & (pts[:, 0] < fw) & (pts[:, 1] < fh)
            known = infr & (o[:, 0] >= 0) & (o[:, 1] >= 0) & (o[:, 0] < wd) & (o[:, 1] < h); known[known] = ~v.U[o[known, 1], o[known, 0]]
            sens = np.stack([(wsc.project(v.cam, X3 + w_move[:, f:f + 1] * N[f] * (.01 / t.S))[0] - base) @ nu for f in range(6)], 1)
            smp.append((np.full(ns, sd[e]), pts, np.repeat(nu[None], ns, 0), w, known, sens, np.full(ns, L / ns), infr,
                        np.linalg.norm(X3 - v.cam['C'], axis=1) * t.S))
        if not smp:
            continue
        sid, pts, nus, w, known, sens, ln, infr, dist = (np.concatenate(z) for z in zip(*smp)); t.known_px[v.k] = float(ln[known].sum())
        px, why[v.k] = outline_px(sid, w >= .5, infr, known, ln, dist, v.cam['K'][0, 0])
        for f in range(6):
            if f in px:
                out[f][v.k] = px[f]
            if ln[known & (w[:, f] >= .5)].sum() >= EVID_MIN_PX:
                seen_px[f][v.k] = float(ln[known & (w[:, f] >= .5)].sum())
        for g in np.unique(sid):
            sel = known & (sid == g)
            if ln[sel].sum() >= EVID_MIN_PX:
                rows.append(np.average(sens[sel], axis=0, weights=ln[sel]))
        sel = known & (w[:, 4] >= .5)
        if sel.any():
            low[v.k] = (pts[sel], nus[sel])
    return out, why, seen_px, low, rows


def face_grid(t, x, f):
    """A 6 x 6 grid of points on face f of the box x (native)."""
    lo, hi = x[[0, 2, 4]], x[[1, 3, 5]]; g = (np.arange(6) + .5) / 6; gg = np.array(np.meshgrid(g, g)).reshape(2, -1).T
    axis = f // 2; loc = np.zeros((len(gg), 3)); loc[:, axis] = x[f]
    for c, o in enumerate(a for a in range(3) if a != axis):
        loc[:, o] = lo[o] + gg[:, c] * (hi[o] - lo[o])
    return t.O + loc @ np.array(t.ax)


def seen(t, x):
    """Per face: photos that see it (facing, >= half of its grid in frame, >= half of those not unknown), its status and the
    status per photo."""
    N = normals(t.ax); res = {}
    for f in range(6):
        P = face_grid(t, x, f); centre = P.mean(0); st = {}
        for v in t.views:
            if N[f] @ (v.cam['C'] - centre) <= 0:
                st[v.k + 1] = 'not_facing'; continue
            uv, z = wsc.project(v.cam, P); fr = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 1] >= 0) & (uv[:, 0] < v.full[1]) & (uv[:, 1] < v.full[0])
            if fr.mean() < .5:
                st[v.k + 1] = 'out_of_frame'; continue
            ui = np.round(uv[fr]).astype(int) - [v.x0, v.y0]
            ok = (ui[:, 0] >= 0) & (ui[:, 1] >= 0) & (ui[:, 0] < v.U.shape[1]) & (ui[:, 1] < v.U.shape[0])
            occ = np.ones(len(ui), bool); occ[ok] = v.U[ui[ok, 1], ui[ok, 0]]
            st[v.k + 1] = 'occluded' if occ.mean() > .5 else 'seen'
        res[f] = ([k for k, s in st.items() if s == 'seen'], next((s for s in ('seen', 'occluded', 'out_of_frame', 'not_facing') if s in st.values()), 'not_facing'), st)
    return res


def unmasked(t, x):
    """Per face: photos without a usable mask of this target in which the face is facing, >= half of its grid in frame,
    >= half of those not behind another displayed model (margin 5 cm + 3 %) and, where the photo has Pi3X points there, >= half
    of them at the face (10 cm + 5 %; else the model is not where the photo shows something): only the mask is missing."""
    N = normals(t.ax); res = {f: [] for f in range(6)}
    for k, cam in enumerate(t.ctx['cams']):
        if k in t.masked:
            continue
        for f in range(6):
            P = face_grid(t, x, f)
            if N[f] @ (cam['C'] - P.mean(0)) <= 0:
                continue
            uv, z = wsc.project(cam, P); fr = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 1] >= 0) & (uv[:, 0] < cam['w']) & (uv[:, 1] < cam['h'])
            if fr.mean() < .5:
                continue
            d = P[fr] - cam['C']; r = np.linalg.norm(d, axis=1); oth = cl.hits(t.scene, cam['C'], d / r[:, None], t.gid)[1]
            vis = oth >= r - (MARGIN_M / t.S + MARGIN_REL * r); q = np.linalg.norm(pi3x_points(t.frames, k, *uv[fr].T) - cam['C'], axis=1)
            has = vis & np.isfinite(q); at = np.abs(q[has] - r[has]) <= PI3X_M / t.S + PI3X_REL * r[has]
            if vis.mean() >= .5 and (not has.any() or at.mean() >= .5):
                res[f].append(k + 1)
    return res


def flat(t, d):
    return hdir(t.n, d)


def name_faces(t, x):
    """{name: internal face}: front = the side face whose normal is closest to the mean horizontal direction from the box to ALL
    the cell's cameras; within NAME_TIE of the next, the one facing the entrance axis (the cell's photos are taken from the
    entrance: the reverse of their mean horizontal viewing direction); left / right as seen looking at the front."""
    N, cams, c = normals(t.ax), t.ctx['cams'], corners(x, t.O, t.ax).mean(0)
    to_cam = flat(t, np.mean([flat(t, cam['C'] - c) for cam in cams], 0)); entrance = flat(t, -np.mean([flat(t, cam['R'][2]) for cam in cams], 0))
    s = sorted(range(4), key=lambda f: -float(N[f] @ to_cam))
    front = s[0] if N[s[0]] @ to_cam - N[s[1]] @ to_cam > NAME_TIE else max(s[:2], key=lambda f: float(N[f] @ entrance))
    right = max([f for f in range(4) if f // 2 != front // 2], key=lambda f: float(N[f] @ np.cross(-N[front], t.n)))
    return dict(front=front, back=front ^ 1, left=right ^ 1, right=right, bottom=4, top=5)


def face_centre(t, x, f):
    lo, hi = x[[0, 2, 4]], x[[1, 3, 5]]; c = (lo + hi) / 2; c[f // 2] = x[f]
    return t.O + c @ np.array(t.ax)


def need_of(t, x, name, f, g, px, good, apart, why, nomask, depth, loose=None):
    """(zh, en) of what would make face f high (None when high), in the words of docs/workcell-photo/CAPTURE-PROTOCOL.md: a photo
    without a mask that sees it asks for a mask, not a photo; the bottom face itself is never photographed (its lower edge is);
    a lower edge below 0.5 m needs a crouched photo; depth (front / back while W is low) needs a side view 45-75 deg off (the back
    edge on the outline); why = {photo: why its outline edge does not count}; loose = the sigma (cm) of a face credited well that
    the fit still cannot pin (it trades off against another face)."""
    if g == 'high':
        return None, None
    zf, d = ZH[name], np.mean([c['K'][0, 0] for c in t.ctx['cams']]) / HIGH_PX_CM / 100; c = face_centre(t, x, f)
    if nomask:
        p = '、'.join(map(str, nomask))
        return f'照片 {p} 拍到了它但没有掩码：补掩码/点选（不需补拍）', f'photo {p} sees it but has no mask: add a mask/click (no new photo needed)'
    if depth:
        return (f'进深需要侧面照片：从偏离{zf}法向 45–75° 的方向（侧面，{zf}的轮廓边在画面里）拍，≥ 10 px/cm（约 {d:.1f} m 以内）',
                f'depth needs a side photo: 45-75 deg off the {name} face normal (from the side, its outline edge in frame), >= 10 px/cm (within about {d:.1f} m)')
    ps = lambda ks: '、'.join(str(k + 1) for k in ks)
    best = max(px.items(), key=lambda r: r[1]) if px else None
    far = lambda k, s: t.ctx['cams'][k]['K'][0, 0] * .01 / s  # the edge's straight-line distance (m)
    if name == 'bottom':
        hgt = lambda k: (t.ctx['cams'][k]['C'] - t.O) @ t.n * t.S
        dep = lambda k: math.degrees(math.asin(np.clip((t.ctx['cams'][k]['C'] - c) @ t.n / max(np.linalg.norm(t.ctx['cams'][k]['C'] - c), 1e-12), -1, 1)))
        crouch = CROUCH if float(x[4]) * t.S < LOW_EDGE_M and not any(CROUCH_M[0] <= hgt(k) <= CROUCH_M[1] and dep(k) <= CROUCH_DEG for k in px) else ('', '')
        if loose:
            return (f'下沿和进深沿视线互相抵消（σ {loose:.1f} cm）：再要一张水平方向相差 ≥ 30° 的下沿照片' + crouch[0],
                    f'the lower edge trades off against depth along the rays (sigma {loose:.1f} cm): a lower-edge photo >= 30 deg to the side' + crouch[1])
        if g == 'unverified':
            return ('要拍清楚下沿：四周侧面的下边和它下面的地面都在画面里，前面没有东西挡（底面本身不用拍）' + crouch[0],
                    'the lower edge: the bottom of the side faces and the floor below it in frame, nothing in front (never the bottom face itself)' + crouch[1])
        if g == 'low':
            k, s = best
            return (f'下沿只有 {s:.1f} px/cm（照片 {k + 1}，{far(k, s):.1f} m）：走近到约 {d:.1f} m 以内（≥ 10 px/cm）' + crouch[0],
                    f'the lower edge has only {s:.1f} px/cm (photo {k + 1}, {far(k, s):.1f} m): within about {d:.1f} m (>= 10 px/cm)' + crouch[1])
        if len(good) >= 2:
            return (f'照片 {ps(good)} 只差 {apart:.0f}°：再要一张下沿清楚、水平方向相差 ≥ 30° 的照片' + crouch[0],
                    f'photos {ps(good)} only {apart:.0f} deg apart: one more of the lower edge >= 30 deg to the side' + crouch[1])
        return (f'再要一张下沿清楚、与照片 {ps(good)} 水平方向相差 ≥ 30° 的照片（≥ 10 px/cm）' + crouch[0],
                f'a second photo of the lower edge >= 30 deg to the side of photo {ps(good)} (>= 10 px/cm)' + crouch[1])
    if loose:
        return f'{zf}和别的面互相抵消（σ {loose:.1f} cm）：从另一侧或另一高度再拍一张', f'the {name} face trades off against other faces (sigma {loose:.1f} cm): a photo from another side or height'
    if g == 'unverified':
        for s_, zh, en in (('occluded', f'里{zf}的轮廓边被挡住：横向挪开，到它不在遮挡物后面再拍', f': the {name} outline edge is occluded; step sideways until nothing is in front'),
                           ('out_of_frame', f'里{zf}的轮廓边不全在画面内：让它整个入画（离边框 ≥ 3 %）', f': the {name} outline edge is out of frame; get all of it in (>= 3 % from the borders)')):
            ks = sorted(k + 1 for k, w in why.items() if w == s_)
            if ks:
                return f'照片 {"、".join(map(str, ks))} {zh}（≥ 10 px/cm，约 {d:.1f} m 以内）', f'photo {", ".join(map(str, ks))}{en} (>= 10 px/cm, within about {d:.1f} m)'
        return (f'没有照片拍到{zf}的轮廓边：从能看到这条边的一侧（侧面或更高处）拍 2 张，水平方向相差 ≥ 30°，≥ 10 px/cm（约 {d:.1f} m 以内）',
                f'no photo has the {name} outline edge: two from where it is on the outline (the side or higher) >= 30 deg apart, >= 10 px/cm (within about {d:.1f} m)')
    if g == 'low':
        k, s = best
        return (f'照片 {k + 1} 只有 {s:.1f} px/cm（{far(k, s):.1f} m）：走近到约 {d:.1f} m 以内（≥ 10 px/cm）', f'photo {k + 1} has only {s:.1f} px/cm ({far(k, s):.1f} m): within about {d:.1f} m (>= 10 px/cm)')
    if len(good) >= 2:
        return f'照片 {ps(good)} 只差 {apart:.0f}°：再要一张水平方向相差 ≥ 30° 的', f'photos {ps(good)} only {apart:.0f} deg apart: one more >= 30 deg to the side'
    return f'再要一张与照片 {ps(good)} 水平方向相差 ≥ 30° 的照片（≥ 10 px/cm）', f'a second photo >= 30 deg to the side of photo {ps(good)} (>= 10 px/cm)'


def covariance(rows):
    """Face covariance (cm^2) from the J rows (px per cm of each face), each with error MARGIN_PX, and a PRIOR_CM prior."""
    J = np.array(rows) if rows else np.zeros((0, 6))
    return MARGIN_PX ** 2 * np.linalg.inv(J.T @ J + (MARGIN_PX / PRIOR_CM) ** 2 * np.eye(6))


def contact(t, low):
    """Photos where >= half of the lower silhouette's known samples have floor (floor mask or floor Pi3X point) 2-6 px below."""
    hits = []
    for k, (pts, nu) in low.items():
        if len(pts) < EVID_MIN_PX / EVID_STEP:
            continue
        fm = t.floor_masks.get(k); cam = t.ctx['cams'][k]; any_floor = np.zeros(len(pts), bool)
        for off in (2, 4, 6):
            q = np.round(pts + off * nu); ok = (q[:, 0] >= 0) & (q[:, 1] >= 0) & (q[:, 0] < cam['w']) & (q[:, 1] < cam['h'])
            fl = np.zeros(len(pts), bool)
            if fm is not None:
                qi = q[ok].astype(int); fl[ok] = fm[qi[:, 1], qi[:, 0]]
            with np.errstate(invalid='ignore'):
                fl |= np.abs((pi3x_points(t.frames, k, q[:, 0], q[:, 1]) - t.O) @ t.n) * t.S <= FLOOR_TOL_M
            any_floor |= fl
        if any_floor.mean() >= .5:
            hits.append(k + 1)
    return hits


def fit_target(t, parent_theta=None):
    """Yaw, prior, fit, refit with the occlusion of the fitted box. Returns (x, prior, bound, info) or None."""
    info = {}; pts = np.vstack([lift(t, v.k, v.region_full) for v in t.views] + [np.zeros((0, 3))]); t.own_pts = pts
    if t.part:
        lifted = np.vstack([model_lift(t, v.k, v.region_full) for v in t.views] + [np.zeros((0, 3))])
        t.prior_pts = np.vstack([lifted, pts]); info['priorSource'] = f'part pixels on the model ({len(lifted)}) + Pi3X ({len(pts)})'
    else:
        info['priorSource'] = 'displayed model surface'
    if len(t.prior_pts) < 10:
        return None
    flat = lambda P: np.c_[(P - t.O) @ t.e1, (P - t.O) @ t.e2]
    th_plane, pl = plane_yaw(t, pts) if parent_theta is None else (None, None)
    info['dominantPlane'] = None if pl is None else dict(inlierFraction=round(pl[2], 3), tiltDeg=round(math.degrees(math.acos(min(1., abs(float(pl[0] @ t.n))))), 1))
    if parent_theta is not None:
        cands = {'object': parent_theta}
    elif th_plane is not None:  # one plane holds most of the object's points: its yaw, not the silhouettes' near-tie
        cands = {'plane': th_plane}
    else:
        cdir = np.mean([v.cam['C'] - t.prior_pts.mean(0) for v in t.views], axis=0)
        cands = {'model': minrect_yaw(flat(t.prior_pts)), 'pi3x': minrect_yaw(flat(pts)) if len(pts) >= 30 else None,
                 'cameras': (math.atan2(cdir @ t.e2, cdir @ t.e1) - math.pi / 2) % (math.pi / 2)}
        cands = {k: v for k, v in cands.items() if v is not None}
    crop = []
    for th in cands.values():
        b = t.set_yaw(th)[1]; crop.append((b, t.ax))
    prior, bound = t.set_yaw(next(iter(cands.values())))
    t.prepare(prior, bound, crop)
    if not t.views:
        return None
    memo = {}

    def at(th):
        if th not in memo:
            pr, bd = t.set_yaw(th); x, c = t.fit(pr, pr, bd, sweeps=3); memo[th] = (c, th, x, pr, bd)
        return memo[th]
    trials = {name: at(th) for name, th in cands.items()}
    area = lambda r: (r[3][1] - r[3][0]) * (r[3][3] - r[3][2])  # the prior's footprint area at that yaw
    best = min(r[0] for r in trials.values())  # the footprint follows the model at any yaw, so the silhouettes barely see yaw:
    near = [k for k, r in trials.items() if r[0] <= best * (1 + YAW_TIE) + 50]  # near-ties go to the minimum-area footprint
    src = min(near, key=lambda k: area(trials[k])); c, th, x, prior, bound = trials[src]
    info['yawCandidates'] = {k: dict(yawDeg=round(math.degrees(v[1]), 1), cost=round(v[0]), priorAreaCm2=round(area(v) * (t.S * 100) ** 2))
                             for k, v in trials.items()}
    info['yawSource'] = src; info['yawDeg'] = round(math.degrees(th), 2)
    prior, bound = t.set_yaw(th); t.refresh(x, bound)
    x, c = t.fit(x, prior, bound, sweeps=6, polish=True)
    t.refresh(x, bound); x, c = t.fit(x, prior, bound, sweeps=4, polish=True)
    info['pi3xPoints'] = int(len(pts)); info['cost'] = c; info['footprintVertices'] = int(len(t.fp))
    return x, prior, bound, info


def containment(t, x):
    """The object's own mask points (point maps, eroded masks, off the floor) against box x: the share outside it by more than
    CONTAIN_TOL_M along each horizontal axis, and the share inside the box grown by it (all axes)."""
    P = t.own_pts
    if len(P) < 30:
        return dict(n=int(len(P)), outsideL=0., outsideW=0., inside=None)
    c = coords(P, t.O, t.ax); tol = CONTAIN_TOL_M / t.S
    out = [(c[:, i] < x[2 * i] - tol) | (c[:, i] > x[2 * i + 1] + tol) for i in range(3)]
    return dict(n=int(len(P)), outsideL=float(out[0].mean()), outsideW=float(out[1].mean()), inside=float((~(out[0] | out[1] | out[2])).mean()))


def coverage(t, x):
    """Per photo, the fraction of the region mask (uncontested) inside the projected shape."""
    X = prism(x, t.fp, t.O, t.ax); out = {}
    for v in t.views:
        poly = hull(v.cam, X)[0]; m = np.zeros(v.M.shape, np.uint8)
        if poly is not None:
            cv2.fillConvexPoly(m, np.round(poly - [v.x0, v.y0]).astype(np.int32), 1)
        out[v.k] = float((v.M & (m > 0)).sum() / max(1, v.M.sum()))
    return out


def describe(t, x, prior, bound, info, label, layer=None):
    """CONTRACT record of one fitted box (+ 'research': the evidence behind it) and its final x. layer: the object's layer
    confidence entry ({} when the layer has none; None for a part)."""
    cm = lambda v: float(v) * t.S * 100
    cov_ph = coverage(t, x); rejected = min(cov_ph.values()) < MIN_COVER; notes, U = [], set()
    ev, why, spx, low, rows = evidence(t, x); X = prism(x, t.fp, t.O, t.ax)
    res = {v.k: v.cost(hull(v.cam, X)[0]) / max(1., t.known_px.get(v.k, 0.)) for v in t.views}  # mean mismatch width (px) of the
    resid = sum(v.cost(hull(v.cam, X)[0]) for v in t.views) / max(1., sum(t.known_px.values()))  # fitted box, before any reset
    capped = rejected or resid > RESID_PX or min(cov_ph.values()) < COVER_CAP
    if capped:  # R3: never a flattened or partial fitted box: the displayed model's box, every dimension at most low, no sigma
        x = prior.copy(); ev, why, spx, low, rows = evidence(t, x)
        notes.append(f'capped: mean mismatch {resid:.1f} px (cap {RESID_PX:.0f}), mask coverage ' + ', '.join(f'{100 * c:.0f} %' for c in cov_ph.values()) +
                     f' (photos {", ".join(str(k + 1) for k in cov_ph)}; cap < {100 * COVER_CAP:.0f} % in one)' + ('; fit rejected (< 50 %)' if rejected else '') +
                     '; the displayed model\'s box kept, every dimension at most low, sigma null')
    cov = covariance(rows)
    contain = containment(t, x)
    if not capped:  # no collapse: a face the photos do not pin keeps the prior size (both faces: the prior faces); the other
        U = {f for f in range(6) if not spx[f] or math.sqrt(cov[f, f]) >= LOOSE_CM}  # faces keep their fit (a refit would make
        for a, frac in ((0, contain['outsideL']), (2, contain['outsideW'])):  # the box contradicts its own points along this axis:
            if frac > CONTAIN_OUT:  # the silhouettes did not pin it (a round object fitted by a rectangle): the prior size
                U |= {a, a + 1}; notes.append(f'{100 * frac:.0f} % of the mask points lie outside the box along axis {a // 2} by > {100 * CONTAIN_TOL_M:.0f} cm: that size is unpinned')
        if U:  # them absorb a wrong prior)
            x = x.copy()
            for a, b in ((0, 1), (2, 3), (4, 5)):
                if a in U and b in U:
                    x[a], x[b] = prior[a], prior[b]
                elif a in U:
                    x[a] = x[b] - (prior[b] - prior[a])
                elif b in U:
                    x[b] = x[a] + (prior[b] - prior[a])
            ev, why, spx, low, rows = evidence(t, x); cov = covariance(rows)
    contain = containment(t, x)
    vis, nm = seen(t, x), name_faces(t, x); name = {f: k for k, f in nm.items()}; nomask = unmasked(t, x)
    nomask[4] = sorted({p for g in range(4) for p in nomask[g]})  # the lower edge is seen with the side faces
    bottom = cm(x[4]); sank = bottom < -CONTACT_M * 100
    if sank:
        notes.append(f'the fitted bottom was {-bottom:.1f} cm below the floor (depth or masks off); bottom at most low')
    dirs = {f: {v.k: hdir(t.n, v.cam['C'] - face_centre(t, x, f)) for v in t.views} for f in range(6)}
    fs = dict(L=(nm['left'], nm['right']), W=(nm['front'], nm['back']), H=(4, 5), bottom=(4,))
    lowf = set(range(6)) if capped else U | ({4} if sank else set())
    w_axis = normals(t.ax)[nm['front']]  # the front-back axis by the face names (front need not lie along the fit's w axis)
    for f in fs['W']:  # R1, depth: front / back outlines count only from side views or from above (depth_credit, shared)
        P = face_centre(t, x, f)
        ev[f] = {k: s for k, s in ev[f].items()
                 if depth_credit(t.n, w_axis, t.ctx['cams'][k]['C'], P, float(x[5]), float((t.ctx['cams'][k]['C'] - t.O) @ t.n))}
    rated, dimc = grade_box(ev, dirs, fs, lowf, U)  # R1 (shared with capture_plan.py)
    gr = {f: r[0] for f, r in rated.items()}; pre = {f: face_grade(ev[f], dirs[f])[0] for f in range(6)}
    loose = {f: round(math.sqrt(cov[f, f]), 1) for f in U if pre[f] in ('medium', 'high')}  # credited well, yet the fit cannot pin it
    sig = {f: None if capped or f in U or gr[f] == 'unverified' else round(math.sqrt(cov[f, f]), 2) for f in range(6)}
    dsig = {d: None if capped or dimc[d] == 'unverified' or U & set(v) else round(size_sigma(cov, *v), 2) if len(v) == 2 else sig[4] for d, v in fs.items()}
    lowcov = min(cov_ph.values()) < COVER_CAP or sum(t.known_px.values()) < 1  # a coverage cap: the px mismatch says nothing
    what = (f'掩码覆盖仅 {100 * min(cov_ph.values()):.0f}%', f'mask coverage {100 * min(cov_ph.values()):.0f} %') if lowcov else (f'{resid:.0f} px',) * 2
    cap = ((f'掩码与盒子不符（{what[0]}）：尺寸取模型', f'masks disagree with the box ({what[1]}): the size is the model\'s')
           if capped else None)
    need = {}
    for f in range(6):
        zh, en = need_of(t, x, name[f], f, gr[f], ev[f], rated[f][1], rated[f][2], {k: w[f] for k, w in why.items() if f in w}, nomask[f],
                         f in fs['W'] and dimc['W'] in ('low', 'unverified'), loose.get(f))
        if cap:  # the masks first (a missing mask is still named)
            zh, en = cap[0] + '，先修掩码' + ('；' + zh if zh and nomask[f] else ''), cap[1] + '; fix the masks first' + ('; ' + en if en and nomask[f] else '')
        elif f == 4 and sank and pre[4] in ('medium', 'high'):
            zh, en = f'拟合底面在地面下 {-bottom:.1f} cm：深度或掩码有误，需复核', 'the fitted bottom sank below the floor: depth or masks off'
        need[f] = (zh, en)
    fl = contact(t, low) if not capped and -CONTACT_M * 100 <= bottom <= CONTACT_M * 100 else []
    snapped, snap_note = None, None
    if fl:  # the box stands on the floor: snap its bottom face (the top stays, unless it is not above the floor: a flat
        snapped = round(bottom, 2); x = x.copy()  # object then moves up whole), record what was replaced
        x[5] = x[5] if x[5] > .01 / t.S else x[5] - x[4]; x[4] = 0.
        snap_note = '贴地：盒子未移动' if abs(snapped) < .05 else f'贴地：盒子{"下" if snapped > 0 else "上"}移 {abs(snapped):.1f} cm'
    snap_en = None if snap_note is None else 'on the floor: box not moved' if abs(snapped) < .05 else f'on the floor: box moved {"down" if snapped > 0 else "up"} {abs(snapped):.1f} cm'
    N = normals(t.ax); ext = lambda y, f: abs(cm(y[f | 1] - y[f & ~1])) / 100
    size = [round(ext(x, nm['right']), 4), round(ext(x, nm['front']), 4), round(ext(x, 4), 4)]
    dims = {d: dict(valueM=v, sigmaCm=dsig[d], confidence=dimc[d]) for d, v in zip(('L', 'W', 'H', 'bottom'), size + [round(cm(x[4]) / 100, 4)])}
    reasons, reasons_en = ([cap[0]], [cap[1]]) if cap else ([], [])
    reasons += [f'{DIM_ZH[d]}{"置信度低" if dimc[d] == "low" else "未验证"}' for d in ('L', 'H', 'bottom') if dimc[d] in ('low', 'unverified')]
    reasons_en += [f'{DIM_EN[d]} {"low confidence" if dimc[d] == "low" else "unverified"}' for d in ('L', 'H', 'bottom') if dimc[d] in ('low', 'unverified')]
    lay, lay_en = [], []
    if layer:
        if layer.get('level') in ('low', 'unverified'):
            lay.append(f'图层置信度{LEVEL_ZH[layer["level"]]}'); lay_en.append(f'layer confidence {layer["level"]}')
        en_missing = dict(zip(layer.get('missing') or [], layer.get('missingEn') or []))
        for m in layer.get('missing') or []:
            if any(w in m for w in GEOMETRY_WORDS):
                lay.append(f'图层：{m}'); lay_en.append(f'layer: {en_missing.get(m, m)}')
    highlight = bool(lay) or any(dimc[d] in ('low', 'unverified') for d in ('L', 'H', 'bottom'))
    order = ['front', 'back', 'left', 'right', 'bottom', 'top']
    rec = dict(label=label, centerNative=[round(float(c), 6) for c in corners(x, t.O, t.ax).mean(0)],
               axes=[[round(float(c), 6) for c in a] for a in (N[nm['right']], N[nm['back']], t.n)],
               faceNormals={k: [round(float(c), 6) for c in N[nm[k]]] for k in order},
               sizeM=size, bottomM=dims['bottom']['valueM'], topM=round(cm(x[5]) / 100, 4), floorContact=bool(fl), snapNote=snap_note, snapNoteEn=snap_en, dims=dims,
               faces={k: dict(photos=vis[nm[k]][0], status=vis[nm[k]][1], confidence=gr[nm[k]], need=need[nm[k]][0], needEn=need[nm[k]][1]) for k in order},
               highlight=highlight, highlightReasons=(reasons + lay) if highlight else [], highlightReasonsEn=(reasons_en + lay_en) if highlight else [],
               confidence=min((dimc[d] for d in ('L', 'H', 'bottom')), key=GRADES.index),
               method=f"silhouette L1 fit of a gravity-aligned box (model footprint inside, {info['footprintVertices']} vertices) in "
                      f"{len(t.views)} photo(s) ({', '.join(str(v.k + 1) for v in t.views)}), occlusion-aware; yaw {info['yawSource']}; "
                      f"prior {info['priorSource']}; faces the photos do not pin keep the prior size; a capped fit keeps the displayed model's box; "
                      f"face and dimension confidence from the faces' outline edges (>= 80 % in frame and known, >= 10 px/cm at the straight-line "
                      f"distance; 'photos' / 'status' = the photos that see the face itself, not counted); object confidence = lowest of L, H, bottom "
                      f"(W is reported, not counted)")
    rec['research'] = dict(
        part=t.part, snappedCm=snapped, floorContactPhotos=fl, unconstrainedFaces=sorted(name[f] for f in U), layerConfidence=layer,
        capped=capped, gradeCaps=dict(low=sorted(name[f] for f in lowf), unpinned=sorted(name[f] for f in U)),
        faceEvidence={name[f]: dict(pxPerCm={str(k + 1): s for k, s in sorted(ev[f].items())}, creditedPhotos=[k + 1 for k in rated[f][1]],
                                    outlineWhyByPhoto={str(k + 1): w[f] for k, w in sorted(why.items()) if f in w},
                                    knownOutlinePx={str(k + 1): round(s, 1) for k, s in sorted(spx[f].items())}, statusByPhoto=vis[f][2],
                                    unmaskedPhotos=nomask[f], sigmaCm=sig[f], movedFromPriorCm=round(cm(x[f] - prior[f]) * (1 if f % 2 else -1), 2),
                                    needEn=need[f][1]) for f in range(6)},
        yawDeg=info['yawDeg'], yawCandidates=info['yawCandidates'], yawSource=info['yawSource'], dominantPlane=info.get('dominantPlane'),
        pointContainment=dict(n=contain['n'], outsideL=round(contain['outsideL'], 3), outsideW=round(contain['outsideW'], 3),
                              inside=None if contain['inside'] is None else round(contain['inside'], 3), tolCm=100 * CONTAIN_TOL_M),
        priorSource=info['priorSource'],
        priorSizeM=[round(ext(prior, nm['right']), 4), round(ext(prior, nm['front']), 4), round(ext(prior, 4), 4)], priorBottomM=round(cm(prior[4]) / 100, 4),
        fitResidualPx=round(resid, 2), residualPxByPhoto={str(k + 1): round(r, 2) for k, r in res.items()},
        maskCoverageByPhoto={str(k + 1): round(c, 3) for k, c in cov_ph.items()}, fitRejected=rejected, notes=notes)
    return rec, x


def draw(ctx, boxes, k, width=1400):
    """Photo k with every fitted box (12 edges), coloured by confidence, numbered; JPEG bytes."""
    g = ctx['gray'][k]; s = width / g.shape[1]
    img = cv2.cvtColor(cv2.resize(np.clip(g, 0, 255).astype(np.uint8), None, fx=s, fy=s, interpolation=cv2.INTER_AREA), cv2.COLOR_GRAY2BGR)
    colour = dict(high=(0, 200, 0), medium=(0, 220, 255), low=(0, 140, 255), unverified=(0, 0, 255))
    for i, (rec, X) in enumerate(boxes):
        uv, z = wsc.project(ctx['cams'][k], X)
        if (z <= 0).any():
            continue
        p = (uv * s).astype(int); col = colour[rec['confidence']]
        for a in range(8):
            for b in range(a + 1, 8):
                if bin(a ^ b).count('1') == 1:
                    cv2.line(img, tuple(p[a]), tuple(p[b]), col, 2 if rec.get('part') else 1, cv2.LINE_AA)
        c = p.mean(0).astype(int)
        if 0 <= c[0] < img.shape[1] and 0 <= c[1] < img.shape[0]:
            cv2.putText(img, str(i + 1), tuple(c), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(img, str(i + 1), tuple(c), cv2.FONT_HERSHEY_SIMPLEX, .6, col, 1, cv2.LINE_AA)
    return cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 82])[1].tobytes()


def floor_masks(ctx):
    """Per photo, the union of the masks of the report's floor entities (geometryRole 'floor')."""
    if 'floorMasks' in ctx:
        return ctx['floorMasks']
    obs = {o['id']: o for o in ctx['doc'].get('observations', [])}; out = {}
    for e in ctx['doc'].get('entities', []):
        if e.get('geometryRole') != 'floor':
            continue
        for oid in e.get('observationRefs') or []:
            o = obs.get(oid)
            if o and o['imageId'] in ctx['index'] and o.get('originalPixelPolygons'):
                k = ctx['index'][o['imageId']]; c = ctx['cams'][k]; m = wsc.polygon_mask(o['originalPixelPolygons'], (c['h'], c['w']))
                out[k] = out[k] | m if k in out else m
    return out


def run(ctx, opts):
    if not ctx['floor']:
        return dict(status='no_floor', boxes={})
    rng = np.random.default_rng(0)
    scene = cl.scene_of([o['mesh'] for o in ctx['objects']])
    frames = ctx['pi3xFrames'] if 'pi3xFrames' in ctx else cl.pi3x_frames(ctx, opts['api']) if opts.get('api') else []
    fm = floor_masks(ctx); pm = opts.get('partMasks') or {}; count = {}
    for o in ctx['objects']:
        for k, m in o['masks'].items():
            count[k] = count.get(k, 0) + m.astype(np.uint8)
    lconf = (ctx.get('layer') or {}).get('confidence') or {}
    boxes, research, drawn, skipped = {}, {}, [], []
    kernel = np.ones((2 * GAP_PX + 1,) * 2, np.uint8)
    for gid, o in enumerate(ctx['objects']):
        if opts.get('only') and not o['id'].startswith(tuple(opts['only'])):  # fit a subset; every object still occludes
            continue
        regions = {k: m for k, m in o['masks'].items() if m.sum() >= 200}
        if not regions:
            skipped.append(dict(entityId=o['id'], reason='no_mask')); continue
        raw = {k: (c - o['masks'][k] if k in o['masks'] else c) > 0 for k, c in count.items()}
        others = {k: cv2.dilate(m.astype(np.uint8), kernel) > 0 for k, m in raw.items()}
        V, F = o['mesh']; model_pts = wsc.sample_surface(V, F, 20000, rng)[0]
        t = Target(ctx, o, gid, None, regions, {}, frames, scene, fm, others, raw, model_pts, model_pts)
        t.neutral = o['id'] in set(opts.get('neutralPrior') or [])  # shown as a measured box: refit from a neutral prior
        res = fit_target(t)
        if res is None:
            skipped.append(dict(entityId=o['id'], reason='no_prior_or_views')); continue
        rec, xs = describe(t, *res, o['label'], lconf.get(o['id']) or {}); extra = rec.pop('research')
        extra['modelBottomCm'] = round(float(np.percentile((V @ ctx['floor'][0] + ctx['floor'][1]) * ctx['S'] * 100, .5)), 2)
        drawn.append((rec, corners(xs, t.O, t.ax))); parts = []
        for part, per in (pm.get(o['id']) or {}).items():  # the clicked part's own pixels (not clipped to the object's mask)
            pr = {int(kk): wsc.polygon_mask(polys, o['masks'][int(kk)].shape) for kk, polys in per.items() if int(kk) in o['masks'] and polys}
            pr = {k: m for k, m in pr.items() if m.sum() >= 200}
            r2 = None
            if pr:
                tp = Target(ctx, o, gid, part, pr, {k: o['masks'][k] & ~pr[k] for k in pr}, frames, scene, fm, others, raw, None, None)
                r2 = fit_target(tp, parent_theta=t.theta)
            if r2 is None:
                parts.append(dict(part=part, label=o['label'], confidence='unverified', reason='no_part_mask' if not pr else 'no_prior_or_views'))
                continue
            prec, px = describe(tp, *r2, o['label']); prec = dict(prec, **prec.pop('research')); parts.append(prec); drawn.append((prec, corners(px, tp.O, tp.ax)))
        boxes[o['id']] = rec; research[o['id']] = dict(rec, **extra, parts=parts)  # parts: research only, never in the layer
    files = {}
    if opts.get('draw', True):
        for k in range(len(ctx['cams'])):
            files[f'photo{k + 1}.jpg'] = draw(ctx, drawn, k)
        files['legend.tsv'] = '\n'.join(f"{i + 1}\t{r['label']}\t{r.get('part') or ''}\t{r['confidence']}" for i, (r, _) in enumerate(drawn)).encode()
    params = dict(gapPx=GAP_PX, restPx=REST_PX, strideOcclusionPx=STRIDE, enlargeM=ENLARGE_M, marginM=MARGIN_M, marginRel=MARGIN_REL,
                  pi3xMarginM=PI3X_M, pi3xMarginRel=PI3X_REL, floorTolM=FLOOR_TOL_M, contactM=CONTACT_M, boundM=BOUND_M, boundRel=BOUND_REL,
                  belowFloorM=BELOW_FLOOR_M, priorPullPx2PerCm=MU, marginPx=MARGIN_PX, evidenceStepPx=EVID_STEP, evidenceMinPx=EVID_MIN_PX,
                  priorCm=PRIOR_CM, looseCm=LOOSE_CM, sideDeg=SIDE_DEG, residPx=RESID_PX, minCoverage=MIN_COVER, coverCap=COVER_CAP, edgeFrac=EDGE_FRAC, highPxPerCm=HIGH_PX_CM, highDeg=HIGH_DEG, yawTie=YAW_TIE, stepsCm=list(STEPS_CM),
                  lowEdgeM=LOW_EDGE_M, crouchCameraM=list(CROUCH_M), crouchDeg=CROUCH_DEG, nameTie=NAME_TIE, geometryWords=list(GEOMETRY_WORDS))
    return dict(status='ok', nativeToMeters=ctx['S'], floor=dict(normal=[float(c) for c in ctx['floor'][0]], offset=float(ctx['floor'][1])),
                params=params, pi3xPhotos=[f['camera'] + 1 for f in frames], floorMaskPhotos=sorted(k + 1 for k in fm), boxes=boxes,
                research=research, skipped=skipped, files=files)


def _check():
    """Synthetic scene, floor z = 0, S = 1, three cameras 1.5 m up (a fourth without masks): a post standing on the floor
    (rotated 20 deg) in front of a panel 24 cm up whose displayed model lacks its lower 16 cm and sits 2 cm too deep, a round
    bollard (r 6 cm), and a cube whose mask in photo 2 is shifted 80 px (masks no box matches). The post must read 10 x 10 x 90 cm on
    the floor (floorContact, snapped < 1.5 cm), the panel 120 x 5 x 76 cm with its bottom at 24 cm (its lower boundary behind the
    post unknown, not an edge), no floor contact, its depth W low at most (the back outline only from above, far: the back face's
    need asks for a side photo), the bollard 12 x 12 cm (not squeezed) with photo 4 asking for a mask, not a photo; front faces the
    cameras; the panel's bottom has evidence; the cube capped (R3): the model's box, sigma null, at most low, the mask reason first.
    Also R1's dimension rule, the size sigma's sign (var = caa + cbb + 2 cab) and the CONTRACT keys."""
    from workcell_checks import lower_edge as le
    assert abs(size_sigma(np.array([[4., -4], [-4, 4]]), 0, 1)) < 1e-9 and abs(size_sigma(np.array([[4., 4], [4, 4]]), 0, 1) - 4) < 1e-9
    assert [dim_grade(*g) for g in (('high', 'high'), ('high', 'medium'), ('medium', 'unverified'), ('low', 'low'), ('unverified', 'unverified'), ('medium',))] == \
        ['high', 'medium', 'low', 'low', 'unverified', 'medium']
    e1, e2 = np.array([1., 0, 0]), np.array([math.cos(math.radians(31)), math.sin(math.radians(31)), 0])
    assert face_grade({0: 12., 1: 11.}, {0: e1, 1: e2})[0] == 'high' and face_grade({0: 12., 1: 9.}, {0: e1, 1: e2})[0] == 'medium' and face_grade({0: 9.}, {0: e1})[0] == 'low'
    assert grade_box({0: {0: 12., 1: 11.}, 1: {}}, {0: {0: e1, 1: e2}, 1: {}}, dict(W=(0, 1)), low={0})[1] == dict(W='low')  # one face credited (capped)
    one = lambda kn, fr=None: outline_px(np.zeros(5, int), np.ones((5, 6), bool), np.ones(5, bool) if fr is None else np.array(fr, bool),
                                         np.array(kn, bool), np.full(5, 10.), np.full(5, 2.), 1000.)  # a 50 px edge 2 m away
    assert one([1, 1, 1, 1, 0]) == ({f: 5. for f in range(6)}, {}) and one([1, 1, 1, 0, 0]) == ({}, {f: 'occluded' for f in range(6)})
    assert one([1, 1, 1, 0, 0], [1, 1, 1, 0, 0])[1] == {f: 'out_of_frame' for f in range(6)}  # 80 % known; px/cm = fx * 1 cm / 2 m
    floor = (np.array([0, 0, 1.]), 0.)
    c, s = math.cos(math.radians(20)), math.sin(math.radians(20))
    pv, pf = cl._box((-.05, -.05, 0), (.05, .05, .9)); post = (pv @ np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]]) + [0, 2.0, 0], pf)
    panel = cl._box((-.6, 2.95, .24), (.6, 3.0, 1.0)); ground = cl._box((-4, -1, -.01), (4, 6, 0))
    cv_, cf = wsc.primitive_mesh(dict(kind='cylinder', parameters=dict(radius=.06, height=.8, segments=48))); bollard = (cv_ + [.8, 2.4, .4], cf)
    cube = cl._box((-1.1, 2.3, 0), (-.95, 2.45, .15))
    cams = [cl._look(np.array([x, 0, 1.5]), np.array([0, 3, .4]), f=1100., w=1280, h=960) for x in (-.9, .1, 1.1)]
    gray, masks = cl._render(cams, [post, panel, ground, bollard, cube])
    objs = [dict(id='post', label='post', kind='box', mesh=post, masks={k: masks[k][0] for k in range(3)}),
            dict(id='panel', label='panel', kind='box', mesh=cl._box((-.6, 2.97, .40), (.6, 3.02, 1.0)), masks={k: masks[k][1] for k in range(3)}),
            dict(id='bollard', label='bollard', kind='cylinder', mesh=bollard, masks={k: masks[k][3] for k in range(3)}),
            dict(id='cube', label='cube', kind='box', mesh=cube, masks={k: np.roll(masks[k][4], 80 * (k == 1), axis=1) for k in range(3)})]
    frames = [dict(camera=k, P=P, affine=A) for k, cam in enumerate(cams) for P, A in [le._pi3x_like(cam, [post, panel, ground, bollard])]]
    cams.append(cl._look(np.array([.5, .2, 1.5]), np.array([.8, 2.4, .4]), f=1100., w=1280, h=960))  # sees the bollard, no masks
    ctx = dict(cams=cams, gray=gray, images=gray, objects=objs, S=1.0, floor=floor, doc={'assets': []}, pi3xFrames=frames,
               floorMasks={k: masks[k][2] for k in range(3)}, index={})
    out = run(ctx, dict(draw=False)); res, rs = out['boxes'], out['research']
    po, pa, bo, cu = res['post'], res['panel'], res['bollard'], res['cube']
    keys = {'label', 'centerNative', 'axes', 'faceNormals', 'sizeM', 'bottomM', 'topM', 'floorContact', 'snapNote', 'snapNoteEn', 'dims', 'faces',
            'highlight', 'highlightReasons', 'highlightReasonsEn', 'confidence', 'method'}
    assert all(set(r) == keys and set(r['faces']['top']) == {'photos', 'status', 'confidence', 'need', 'needEn'} for r in res.values()), [set(r) ^ keys for r in res.values()]
    assert all(len(r['highlightReasons']) == len(r['highlightReasonsEn']) for r in res.values())
    assert po['floorContact'] and po['bottomM'] == 0 and abs(rs['post']['snappedCm']) < 1.5 and po['snapNote'].startswith('贴地'), po
    assert all(abs(a - .1) < .012 for a in po['sizeM'][:2]) and abs(po['sizeM'][2] - .9) < .012, po['sizeM']
    assert abs(pa['bottomM'] - .24) < .01 and not pa['floorContact'] and pa['snapNote'] is None and abs(pa['sizeM'][0] - 1.2) < .015 and abs(pa['topM'] - 1.0) < .01, pa
    assert abs(pa['sizeM'][1] - .05) < .015 and pa['dims']['W']['confidence'] in ('low', 'unverified'), pa['dims']  # depth seen from one side
    assert pa['faces']['back']['need'].startswith('进深需要侧面照片'), pa['faces']['back']
    assert pa['faces']['bottom']['confidence'] != 'unverified' and pa['faces']['front']['status'] == 'seen', pa['faces']
    assert np.dot(pa['axes'][1], [0, 1, 0]) > .99 and np.dot(pa['faceNormals']['front'], [0, -1, 0]) > .99, pa['axes']  # front faces the cameras
    assert all(abs(a - .12) < .012 for a in bo['sizeM'][:2]) and bo['floorContact'], bo['sizeM']
    assert '照片 4' in (bo['faces']['front']['need'] or '照片 4'), bo['faces']['front']
    assert all(r['highlight'] == bool(r['highlightReasons']) for r in res.values()), {k: r['highlightReasons'] for k, r in res.items()}
    assert rs['cube']['capped'] and not cu['floorContact'] and all(abs(a - b) < .01 for a, b in zip(cu['sizeM'], (.15, .15, .15))), (cu['sizeM'], rs['cube']['notes'])
    assert all(v['sigmaCm'] is None and v['confidence'] in ('low', 'unverified') for v in cu['dims'].values()) and cu['highlight'], cu['dims']
    assert cu['highlightReasons'][0].startswith('掩码与盒子不符（') and cu['highlightReasons'][0].endswith('）：尺寸取模型'), cu['highlightReasons']
    assert ('px' in cu['highlightReasons'][0]) != ('覆盖' in cu['highlightReasons'][0]), cu['highlightReasons']  # px mismatch or coverage, never both
    assert not any(r['capped'] for k, r in rs.items() if k != 'cube'), {k: r['notes'] for k, r in rs.items()}
    # the round bollard shown as a thin measured box (a stand-in, 12 x 2 cm): the neutral prior and the containment rule give a
    # square footprint (depth >= the visible width), never the thin box, and its depth is not 'high'
    thin = cl._box((.74, 2.39, 0), (.86, 2.41, .8))
    objs2 = [dict(o, mesh=thin) if o['id'] == 'bollard' else o for o in objs]
    b2 = run(dict(ctx, objects=objs2), dict(draw=False, neutralPrior=['bollard']))
    tb = b2['boxes']['bollard']
    assert min(tb['sizeM'][:2]) >= .10 and tb['dims']['W']['confidence'] != 'high', (tb['sizeM'], tb['dims'], b2['research']['bollard']['notes'])
    assert b2['research']['bollard']['pointContainment']['outsideW'] <= .10, b2['research']['bollard']['pointContainment']
    print('box_faces self-test passed: thin stand-in ->', tb['sizeM'], tb['dims']['W'], '| post', po['sizeM'], po['snapNote'], '| panel', pa['sizeM'], 'bottom', pa['bottomM'], 'W', pa['dims']['W'],
          '| cube', cu['sizeM'], cu['highlightReasons'][0],
          '| bollard', bo['sizeM'], bo['faces']['front']['need'], '| panel faces', {k: v['confidence'] for k, v in pa['faces'].items()},
          '| highlight', {k: r['highlightReasons'] for k, r in res.items()})


if __name__ == '__main__':
    _check()
