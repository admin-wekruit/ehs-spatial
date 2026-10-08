"""Lower edge (下沿) of an object, or of a part of it: the defined measurement of docs/archive/workcell-photo/LOWER-EDGE.md.

Part (click exemplar): one click {entityId, part (a label), clickPhoto (1-based), clickXY [u, v]} on the part. The click ray's
first hit on the object's displayed model is the 3D anchor (the report's Pi3X point on that ray where the model misses or
disagrees with Pi3X by > 5 % of the range), stored in the model's own frame (anchorModel) so the part can be reused on an
identical object: a target may give anchorModel instead of a click. The anchor is projected into every photo with the
object's mask; it is a prompt there unless another model (or the object itself, for a model anchor) is hit in front of it,
or the report's Pi3X range there is > 5 % shorter, or it falls off the object's mask (prompts(); `--prompts-out`).
modal_apps/workcell_part_masks.py --prompts segments each prompt point (SAM 3 tracker, one positive point, in a crop of the
object's mask box) and returns the part masks (opts partMasks); a text phrase as part (the earlier text mode) still works.
Region: the part mask intersected with the object's report mask (the object mask alone without a part).
Lower boundary: along projected 3D verticals (lines through the nadir vanishing point, clearance.verticals) the lowest region
pixel. A line is dropped in the region's 10 % sides, where the frame cuts it, where a pixel 1-20 px below lies in another
object's mask, for a part where the rest of the object fills >= half of the 20 px below (an internal boundary: another piece in
front or attached), or where the ray through the pixel below meets another displayed model in front (occlusion is no edge).
Height: the bottom edge of each kept boundary pixel is back-projected onto the object's own displayed model (first hit, or
the hit at the pixel's centre: the model's edge lies inside the boundary pixel). A ray that only meets the model 3-15 px above
is grazed: counted, used for the occlusion test, excluded from heights (an extrapolated range biases them high); too few
direct lines left: 'too_few_direct_lines', no value. Where the model's silhouette ends <= 2 px below the boundary on >= half
of a photo's lines, the edge read is the model's own bottom: flagged (edgeIsModelBottom), not hidden.
A photo is model-backed when >= half of its candidate lines hit the model directly and the report's Pi3X range agrees with the
model's within 5 % (median of (Pi3X - model) / model at the same region pixels, 10-150 px above the boundary every 10 px; the
gap per offset is reported, pi3xGapProfile). If any usable photo is not (see-through object, or model missing there) the
model is not used: a line is fitted to each photo's boundary pixels (Huber), the back-projected planes of two photos are
intersected and each boundary ray's closest point on that 3D line gives the height. A pair is dropped when the 3D line is
tilted > 45 deg from horizontal (it runs along the verticals), when the two photos see < 25 % of a common stretch of it, or
when < half of a photo's boundary pixels lie within 3 px of its line (post hoc, first run on the reports).
Segmentation edge (seg_step): along the same lines the photo's strongest intensity step within +-10 px (contrast >= 3,
sub-pixel); offsets that agree within 1 px on >= half the lines are a mask bias: corrected (model path: offset x the model's cm
per px; two-view: the boundary points moved onto the step), its standard error in sigma; offsets that do not agree:
hypot(offset, spread) in sigma; no step: the +-10 px window (10 / sqrt 3 px) in sigma.
Height = n.X + d on the report floor x nativeToMeters.
Uncertainty (replaces the 15 deg / 0.5 cm-per-px gates): model path, per photo sigma = hypot(|Pi3X - model range gap over the
band| x sin(depression), IQR / 1.349, segmentation term); without Pi3X there the gap is the see-through bound itself, 5 % of
the range (rangeSource no_pi3x_5pct_of_range: missing evidence never gives a small sigma). Two-view, per pair sigma =
hypot(cm per px of line a x (RMS residual a, segmentation term a), the same for b). Photos (pairs) whose medians differ by
more than max(2 cm, 3 x their combined sigma): 'photos_disagree', no value. Otherwise, over the lowest sigma tier that has
any (<= 1 cm, <= 2.5 cm, all), estimate = their median, sigma = max(RMS of their sigmas, half the spread of their medians)
(errors are largely shared: no averaging gain claimed); published when sigma <= 2.5 cm, graded 'trusted' (<= 1 cm) or
'large'; else 'sigma_too_large', the estimate +- sigma still reported. Reported apart, not in sigma: the scale term (value x
the layer's scale uncertaintyRelative) and the floor term (height on the two-view sweep floor near the object minus on the
report floor, opts sweepFloorAboveReportCm {entityId: cm}, e.g. from clearance B).

Generic: parts are inputs (a click or a phrase); no per-object rules; the scale is read, never changed. opts: {targets:
[{entityId (exact), part?, clickPhoto?, clickXY?, anchorModel?}], partMasks: {entityId: {part: {photoIndex0: polygons}}} (a
part absent from it raises: a label typo), api:
ORIGIN (Pi3X maps), crops: bool, sweepFloorAboveReportCm}. Pure numpy/cv2/open3d: the Modal check container
(modal_apps/workcell_view_checks.py --checks lower_edge) or on-prem (python lower_edge.py --view ... --opts J --out O.json;
--prompts-out P.json writes the click prompts instead; no arguments = self-test)."""
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

try:
    import shape_core as wsc  # the check container (modal_apps/workcell_view_checks.py)
except ImportError:  # locally / on-prem: scripts/workcell_shape_check.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import workcell_shape_check as wsc
from workcell_checks import clearance as cl

EDGE, GAP_PX, GRAZE_PX, MIN_LINES = .10, 20, 15, 10
BAND_PX = tuple(range(10, 151, 10))  # Pi3X vs model range at the same pixels 10-150 px above the boundary: the median over the band
MIN_HIT, SEE_THROUGH = .5, .05  # Pi3X vs solid models' bottom rays here: <= 3.5 % of range (clearance A: posts, cart, bollards, robot)
MAX_TILT_DEG, MIN_OVERLAP = 45., .25  # post hoc (first real run): the 3D line must cross the verticals and be seen in both photos
STRAIGHT_PX, MIN_STRAIGHT = 3., .5  # post hoc: the method assumes a straight edge; >= half the boundary pixels within 3 px of the line
MAX_SIGMA_CM, TRUST_SIGMA_CM = 2.5, 1.0  # a value is published at sigma <= 2.5 cm, graded 'trusted' at <= 1 cm, else 'large'
AGREE_MIN_CM, AGREE_K, IQR_SIGMA = 2.0, 3.0, 1.349  # IQR / 1.349 = the normal-equivalent sigma of the spread
ANCHOR_MARGIN_M, ANCHOR_MARGIN_REL = .02, .01  # a hit this much (2 cm + 1 % of the range) in front of the anchor occludes it
BOTTOM_PX = 2  # the boundary within 2 px of the model's silhouette bottom: the edge read is the model's own bottom (flagged)
STEP_WIN, STEP_CONTRAST, STEP_AGREE_PX = 10, 3.0, 1.0  # photo intensity step within +-10 px (clearance A's window and contrast)


def model_transform(ctx, obj):
    """The displayed model's placement, as load_report places it: the layer's model, else the entity's current transform."""
    over = ((ctx.get('layer') or {}).get('models') or {}).get(obj['id'])
    if over:
        return over['representation']['transform']
    e = next(e for e in ctx['doc']['entities'] if e['id'] == obj['id'])
    rep = next(r for r in e['representations'] if r['id'] == e['activeModelRepresentationId'])
    return e.get('currentModelTransform') or rep['transform']


def to_model(X, T):
    return (X - np.array(T['position'], float)) @ wsc.quat_matrix(T['quaternion']) / np.array(T['scale'], float)


def anchor_of(ctx, obj, gid, scene, frames, t):
    """(world anchor, source, detail) of a click target: the click ray's first hit on the object's displayed model, unless the
    report's Pi3X range at the click disagrees by > SEE_THROUGH of the range (then the Pi3X point on the ray), or the model
    misses (Pi3X point); or the stored model-frame anchorModel placed by this object's transform."""
    T = model_transform(ctx, obj)
    if t.get('anchorModel') is not None:
        return wsc.placed(np.array([t['anchorModel']], float), T)[0], 'stored', {}
    k = int(t['clickPhoto']) - 1; cam = ctx['cams'][k]; u, v = (float(x) for x in t['clickXY'])
    D = cl.rays(cam, np.array([u]), np.array([v])); own = cl.hits(scene, cam['C'], D, gid)[0][0]
    r = pi3x_range(frames, k, cam, np.ones((cam['h'], cam['w']), bool), np.array([round(u)]), np.array([round(v)]))[0]
    m = obj['masks'].get(k); detail = dict(onObjectMask=bool(m is not None and m[round(v), round(u)]),
                                          modelRangeM=round(float(own * ctx['S']), 3) if np.isfinite(own) else None,
                                          pi3xRangeM=round(float(r * ctx['S']), 3) if np.isfinite(r) else None)
    if np.isfinite(own) and not (np.isfinite(r) and abs(r - own) > SEE_THROUGH * own):
        return cam['C'] + own * D[0], 'model', detail
    if np.isfinite(r):
        return cam['C'] + r * D[0], 'pi3x', detail
    return None, None, detail


def prompt_points(ctx, obj, gid, scene, frames, t, X, source):
    """Per photo with the object's mask: the anchor's projection and whether it is a prompt there (see anchor_of)."""
    out = []
    for k in sorted(obj['masks']):
        cam = ctx['cams'][k]; C = cam['C']
        if t.get('clickPhoto') is not None and k == int(t['clickPhoto']) - 1:
            out.append(dict(photo=k + 1, point=[float(x) for x in t['clickXY']], prompt=True, why='click')); continue
        uv, z = wsc.project(cam, X[None]); u, v = (float(x) for x in uv[0]); ui, vi = round(u), round(v)
        row = dict(photo=k + 1, point=[round(u, 1), round(v, 1)], prompt=False)
        if z[0] <= 0 or not (0 <= ui < cam['w'] and 0 <= vi < cam['h']):
            out.append({**row, 'why': 'outside_frame'}); continue
        d = float(np.linalg.norm(X - C)); D = ((X - C) / d)[None]; own, oth = (x[0] for x in cl.hits(scene, C, D, gid))
        margin = ANCHOR_MARGIN_M / ctx['S'] + ANCHOR_MARGIN_REL * d
        r = pi3x_range(frames, k, cam, np.ones((cam['h'], cam['w']), bool), np.array([ui]), np.array([vi]))[0]
        why = ('other_model_in_front' if oth < d - margin else 'own_model_in_front' if source == 'model' and own < d - margin
               else 'pi3x_in_front' if np.isfinite(r) and r < d * (1 - SEE_THROUGH) else 'off_object_mask' if not obj['masks'][k][vi, ui]
               else None)
        out.append({**row, 'prompt': why is None, 'why': why or 'visible'})
    return out


def anchor(ctx, obj, gid, scene, frames, t):
    """The click target's anchor record: source, model-frame and world point, height, and the per-photo prompts."""
    X, source, detail = anchor_of(ctx, obj, gid, scene, frames, t)
    a = dict(entityId=obj['id'], part=t.get('part'), clickPhoto=t.get('clickPhoto'), clickXY=t.get('clickXY'), anchorSource=source, **detail)
    if X is None:
        return {**a, 'why': 'click_has_no_depth', 'photos': []}
    return {**a, 'anchorModel': [round(float(x), 6) for x in to_model(X, model_transform(ctx, obj))],
            'anchorWorld': [round(float(x), 6) for x in X], 'anchorHeightCm': round(float((X @ ctx['floor'][0] + ctx['floor'][1]) * ctx['S'] * 100), 2),
            'photos': prompt_points(ctx, obj, gid, scene, frames, t, X, source)}


def prompts(ctx, opts):
    """Click targets -> {anchors, prompts: [{entityId, part, photoIndex0, point}]} for modal_apps/workcell_part_masks.py --prompts."""
    scene = cl.scene_of([o['mesh'] for o in ctx['objects']])
    frames = ctx['pi3xFrames'] if 'pi3xFrames' in ctx else cl.pi3x_frames(ctx, opts['api']) if opts.get('api') else []
    anchors, points = [], []
    for t in opts.get('targets') or []:
        if t.get('clickXY') is None and t.get('anchorModel') is None:
            continue
        gid, obj = next(((i, o) for i, o in enumerate(ctx['objects']) if o['id'] == t['entityId']), (None, None))
        a = anchor(ctx, obj, gid, scene, frames, t) if obj else dict(entityId=t['entityId'], part=t.get('part'), anchorSource=None, why='no_object', photos=[])
        anchors.append(a)
        points += [dict(entityId=t['entityId'], part=t['part'], photoIndex0=p['photo'] - 1, point=p['point'], why=p['why'])
                   for p in a['photos'] if p['prompt']]
    return dict(anchors=anchors, prompts=points)


def region_of(ctx, obj, k, part, part_masks):
    mask = obj['masks'][k]
    if not part:
        return mask
    polys = ((part_masks.get(obj['id']) or {}).get(part) or {}).get(str(k))
    return mask & wsc.polygon_mask(polys, mask.shape) if polys else None


def pi3x_range(frames, k, cam, region, uq, vq):
    """Pi3X range along each ray of pixels (uq, vq) (NaN outside the region / grid / without a map for photo k)."""
    out = np.full(len(uq), np.nan); f = next((f for f in frames if f['camera'] == k), None)
    h, w = region.shape; ok = (uq >= 0) & (vq >= 0) & (uq < w) & (vq < h)
    if f is None or not ok.any():
        return out
    ok[ok] = region[vq[ok], uq[ok]]
    A = cv2.invertAffineTransform(f['affine'].astype(np.float64)); H, W = f['P'].shape[:2]
    gj = np.round(A[0, 0] * uq + A[0, 1] * vq + A[0, 2]).astype(int); gi = np.round(A[1, 0] * uq + A[1, 1] * vq + A[1, 2]).astype(int)
    ok &= (gj >= 0) & (gi >= 0) & (gj < W) & (gi < H)
    P = f['P'][gi[ok], gj[ok]]; good = np.isfinite(P).all(1) & (np.abs(P).sum(1) > 0)
    idx = np.nonzero(ok)[0][good]; out[idx] = np.linalg.norm(P[good] - cam['C'], axis=1)
    return out


def seg_step(img, u, v, du, dv, sel):
    """The mask boundary against the photo's sub-pixel intensity step along the same lines (the strongest step within +-10 px,
    contrast >= 3, parabolic peak). offsetPx > 0: the step lies below the mask's bottom edge. 'corrected': >= MIN_LINES and
    >= half of the lines have a step and their offsets agree within 1 px (IQR / 1.349): a segmentation bias, corrected, its
    uncertainty the median's standard error; 'in_sigma': they do not agree: hypot(offset, spread) goes into sigma; 'no_step':
    too few steps, the +-10 px window as a uniform error (10 / sqrt 3 px) goes into sigma."""
    h, w = img.shape; sw = np.arange(-STEP_WIN, STEP_WIN + 2); r = np.arange(len(u))
    uv = np.c_[(u[:, None] + sw * du[:, None]).ravel(), (v[:, None] + sw * dv[:, None]).ravel()]
    g = np.abs(np.diff(wsc.bilinear(img, np.clip(uv, 0, [w - 1, h - 1])).reshape(len(u), len(sw)), axis=1)); i = g.argmax(1)
    sharp = sel & (g.max(1) / (np.median(g, 1) + 1e-3) >= STEP_CONTRAST)
    j = np.clip(i, 1, g.shape[1] - 2); a, b, c = g[r, j - 1], g[r, j], g[r, j + 1]; den = a - 2 * b + c
    off = sw[i] + np.where((i == j) & (den < 0), .5 * (a - c) / np.where(den < 0, den, -1.), 0.)  # gradient i sits at sw[i] + .5, the mask edge at +.5
    n = int(sharp.sum())
    if n < max(MIN_LINES, sel.sum() / 2):
        return dict(mode='no_step', sharpLines=n, offsetPx=None, spreadPx=None, uncertaintyPx=round(STEP_WIN / math.sqrt(3), 2))
    q1, med, q3 = np.percentile(off[sharp], [25, 50, 75]); spread = (q3 - q1) / IQR_SIGMA; agree = spread <= STEP_AGREE_PX
    return dict(mode='corrected' if agree else 'in_sigma', sharpLines=n, offsetPx=round(float(med), 2), spreadPx=round(float(spread), 2),
                uncertaintyPx=round(float(1.2533 * spread / math.sqrt(n) if agree else math.hypot(med, spread)), 2))


def measure_photo(ctx, k, obj, gid, scene, region, own_part, frames):
    cam, S = ctx['cams'][k], ctx['S']; n, d = ctx['floor']; h, w = region.shape
    u, v, du, dv, coord = cl.verticals(region, cam, n)
    if len(u) < 5:
        return None
    lo, span = coord.min(), coord.max() - coord.min()
    edge = (coord < lo + EDGE * span) | (coord > lo + span - EDGE * span)
    at = lambda s: (np.round(u + s * du).astype(int), np.round(v + s * dv).astype(int))
    inside = lambda a, b: (a >= 0) & (b >= 0) & (a < w) & (b < h)
    frame = ~inside(*at(GAP_PX + 1))
    other, own_count = np.zeros(len(u), bool), np.zeros(len(u), int)
    for s in range(1, GAP_PX + 1):
        a, b = at(s); a, b = np.clip(a, 0, w - 1), np.clip(b, 0, h - 1)
        for o in ctx['objects']:
            if o is not obj and k in o['masks']:
                other |= o['masks'][k][b, a]
        own_count += obj['masks'][k][b, a] & ~region[b, a]
    own_below = (own_count >= GAP_PX / 2) if own_part else np.zeros(len(u), bool)
    C = cam['C']; ray = lambda s: cl.rays(cam, u + s * du, v + s * dv); Db = ray(.5)
    t_own = cl.hits(scene, C, Db, gid)[0]; t_oth_below = cl.hits(scene, C, ray(1.5), gid)[1]; grazed = np.zeros(len(u), int)
    for off in range(0, GRAZE_PX + 1, 3):  # the ray passes under the model's silhouette: its range at the pixel centre (1: the edge lies
        miss = ~np.isfinite(t_own)  # inside the boundary pixel, kept) or 3-15 px above (2: grazed, extrapolated, biased high: excluded)
        if not miss.any():
            break
        a, b = np.round(u[miss] - off * du[miss]), np.round(v[miss] - off * dv[miss])
        t_own[miss] = cl.hits(scene, C, cl.rays(cam, a, b), gid)[0]; grazed[miss] = np.isfinite(t_own[miss]) * (1 + (off > 0))
    R = np.full((len(u), len(BAND_PX)), np.nan); T = R.copy()  # Pi3X and model range at the same region pixels over the band
    for j, off in enumerate(BAND_PX):
        a, b = at(-off); R[:, j] = pi3x_range(frames, k, cam, region, a, b); got = np.isfinite(R[:, j])
        if got.any():
            T[got, j] = cl.hits(scene, C, cl.rays(cam, a[got].astype(float), b[got].astype(float)), gid)[0]
    r_near = R[np.arange(len(u)), np.isfinite(R).argmax(1)]  # the nearest Pi3X range above the boundary (NaN: none)
    t_ref = np.where(np.isfinite(t_own), t_own, np.where(np.isfinite(r_near), r_near, np.inf))
    occluded = t_oth_below < t_ref
    base = ~edge & ~frame
    cand = base & ~other & ~own_below & ~occluded
    direct = cand & np.isfinite(t_own) & (grazed < 2)  # the boundary pixel (its bottom edge or its centre) lies on the model
    same = cand[:, None] & np.isfinite(R) & np.isfinite(T)
    with np.errstate(invalid='ignore', divide='ignore'):
        gap = float(np.median((R - T)[same]) * S * 100) if same.sum() >= 5 else None
        rel = float(np.median(((R - T) / T)[same])) if same.sum() >= 5 else None  # same pixel, same ray (modelBacked)
    profile = {str(off): [round(float(np.median((R[:, j] - T[:, j])[same[:, j]]) * S * 100), 2), int(same[:, j].sum())]
               for j, off in enumerate(BAND_PX) if same[:, j].sum() >= 5}
    hit_frac = float(direct.sum() / cand.sum()) if cand.any() else 0.
    backed = hit_frac >= MIN_HIT and (rel is None or abs(rel) <= SEE_THROUGH)
    hgt = ((C + np.where(direct, t_own, 0)[:, None] * Db) @ n + d) * S * 100
    D_top = ray(-.5); t_top = cl.hits(scene, C, D_top, gid)[0]; both = direct & np.isfinite(t_top)
    dh = float(np.median(hgt[both] - ((C + t_top[both, None] * D_top[both]) @ n + d) * S * 100)) if both.any() else None  # cm per px down, on the model
    seg = seg_step(ctx['images'][k], u, v, du, dv, direct)
    corr = seg['offsetPx'] * dh if seg['mode'] == 'corrected' and dh is not None else 0.
    seg.update(cmPerPx=None if dh is None else round(dh, 3), correctionCm=round(corr, 2), termCm=None if dh is None else round(seg['uncertaintyPx'] * abs(dh), 2))
    st = cl.stats(hgt[direct] + corr)
    sin_dep = float(np.median(-(Db[direct] @ n))) if direct.any() else None  # sine of the ray's angle below the horizontal
    if sin_dep is None:  # no direct line: no height, no sigma
        range_cm, source = None, None
    elif gap is not None:  # a range error dr moves the height dr sin(dep)
        range_cm, source = abs(gap) * sin_dep, 'pi3x'
    else:  # no Pi3X here: the see-through bound itself (5 % of the range), never 0
        range_cm, source = SEE_THROUGH * float(np.median(t_own[direct])) * S * 100 * sin_dep, 'no_pi3x_5pct_of_range'
    terms = dict(iqrCm=round((st['iqrCm'][1] - st['iqrCm'][0]) / IQR_SIGMA, 2) if st['n'] else None,
                 rangeCm=None if range_cm is None else round(range_cm, 2), rangeSource=source, segCm=seg['termCm'])
    low = direct & ~np.isfinite(cl.hits(scene, C, ray(BOTTOM_PX + .5), gid)[0])  # the model's silhouette ends <= 2 px below the boundary
    row = dict(photo=k + 1, regionPx=int(region.sum()), lines=int(len(u)), candidates=int(cand.sum()), **st,
               sigmaCm=round(math.hypot(terms['iqrCm'], range_cm or 0, seg['termCm'] or 0), 2) if st['n'] else None, sigmaTerms=terms,
               sinDepression=None if sin_dep is None else round(sin_dep, 3), pi3xModelGapCm=None if gap is None else round(gap, 2),
               pi3xGapProfile=profile, segEdge=seg, segEdgeAll=seg_step(ctx['images'][k], u, v, du, dv, cand),
               centreHits=int((cand & (grazed == 1)).sum()), grazedHits=int((cand & (grazed == 2)).sum()),
               edgeAtModelBottom=round(float(low.sum() / direct.sum()), 3) if direct.any() else None,
               dropped=dict(edge10pct=int(edge.sum()), frameCut=int((frame & ~edge).sum()), otherMaskBelow=int((other & base).sum()),
                            ownObjectBelow=int((own_below & base & ~other).sum()), occludedBelow=int((occluded & base & ~other & ~own_below).sum())),
               hitFraction=round(hit_frac, 3), pi3xRangeDiffRel=None if rel is None else round(rel, 4),
               modelBacked=bool(backed), rangeM=round(float(np.median(t_ref[cand & np.isfinite(t_ref)]) * S), 3) if (cand & np.isfinite(t_ref)).any() else None)
    s2 = row['segEdgeAll']; s = s2['offsetPx'] if s2['mode'] == 'corrected' else 0.  # two-view: the boundary moved onto the photo's step
    pts = np.c_[u + (.5 + s) * du, v + (.5 + s) * dv][cand]
    arr = dict(u=u, v=v, kept=direct, plane_used=np.zeros(len(u), bool), other=(other | own_below) & base, occluded=occluded & base & ~other & ~own_below,
               edge=edge, nodepth=cand & ~direct)
    return row, pts, arr


def image_line(pts):
    vx, vy, x0, y0 = cv2.fitLine(pts.astype(np.float32), cv2.DIST_HUBER, 0, .01, .01).ravel()
    l = np.array([vy, -vx, vx * y0 - vy * x0], float)
    return l / np.hypot(l[0], l[1])


def back_plane(cam, l):
    P = np.hstack([cam['K'] @ cam['R'], (cam['K'] @ cam['t'])[:, None]]); pi = P.T @ l
    return pi / np.linalg.norm(pi[:3])


def line_heights(ctx, cams, lines, pts):
    """Plane-plane 3D line of two image lines, and the heights (cm) of each boundary ray's closest point on it."""
    (n, d), S = ctx['floor'], ctx['S']
    pa, pb = (back_plane(c, l) for c, l in zip(cams, lines))
    D = np.cross(pa[:3], pb[:3]); dihedral = math.degrees(math.asin(min(1., float(np.linalg.norm(D)))))
    if dihedral < 1e-3:  # the same plane: no line
        return np.zeros(0), dihedral, np.array([1., 0, 0]), [np.zeros(1), np.zeros(1)]
    D /= np.linalg.norm(D)
    X0 = np.linalg.lstsq(np.vstack([pa[:3], pb[:3]]), -np.array([pa[3], pb[3]]), rcond=None)[0]
    hs, ss = [], []
    for cam, p in zip(cams, pts):
        r = cl.rays(cam, p[:, 0], p[:, 1]); w0 = cam['C'] - X0
        b, dd, e = r @ D, r @ w0, D @ w0
        s = (e - b * dd) / np.maximum(1 - b * b, 1e-12)  # closest point on the line (unit r, unit D)
        hs.append(((X0 + s[:, None] * D) @ n + d) * S * 100); ss.append(s)
    return np.concatenate(hs), dihedral, D, ss


def two_view(ctx, a, b, pa, pb, seg_px=(0., 0.)):
    """seg_px: each photo's segmentation-edge uncertainty (px, seg_step on its candidate lines), in sigma beside the line fit."""
    cams = [ctx['cams'][a], ctx['cams'][b]]; la, lb = image_line(pa), image_line(pb); n = ctx['floor'][0]
    res = [np.abs(np.c_[p, np.ones(len(p))] @ l) for p, l in ((pa, la), (pb, lb))]  # boundary pixels' distance to their line
    straight = [float(np.mean(r <= STRAIGHT_PX)) for r in res]; rms = [float(np.sqrt(np.mean(r * r))) for r in res]
    h, dihedral, D, (sa, sb) = line_heights(ctx, cams, [la, lb], [pa, pb]); med = float(np.median(h)) if len(h) else math.nan
    sens = []
    for i in (0, 1):  # 1 px shift of one image line along its normal
        ls = [la.copy(), lb.copy()]; ls[i][2] -= 1.; h2 = line_heights(ctx, cams, ls, [pa, pb])[0]
        sens.append(abs(float(np.median(h2)) - med) if len(h2) and len(h) else math.inf)
    fit, seg = math.hypot(sens[0] * rms[0], sens[1] * rms[1]), math.hypot(sens[0] * seg_px[0], sens[1] * seg_px[1])
    sigma = math.hypot(fit, seg)  # each line's offset uncertainty = its fit's RMS residual (px) and its segmentation-edge term
    overlap = max(0., min(sa.max(), sb.max()) - max(sa.min(), sb.min())) / max(1e-12, max(sa.max(), sb.max()) - min(sa.min(), sb.min()))
    tilt = math.degrees(math.asin(min(1., abs(float(D @ n)))))
    fails = [name for name, bad in (('tilt', tilt > MAX_TILT_DEG), ('overlap', overlap < MIN_OVERLAP),
                                    ('straight', min(straight) < MIN_STRAIGHT)) if bad]
    fin = lambda x, r=3: round(x, r) if math.isfinite(x) else None
    return dict(photos=[a + 1, b + 1], **cl.stats(h), sigmaCm=fin(sigma, 2), sigmaTerms=dict(lineFitCm=fin(fit, 2), segCm=fin(seg, 2)),
                dihedralDeg=round(dihedral, 2), cmPerPx=fin(max(sens)),
                cmPerPxHypot=fin(math.hypot(*sens)), lineRmsPx=[round(x, 2) for x in rms], lineTiltDeg=round(tilt, 2), overlap=round(overlap, 3),
                straight=[round(x, 3) for x in straight], accepted=not fails, reason=','.join(fails) or None)


def combine(cands):
    """(value, sigma, estimate, reason, grade) over per-photo (or per-pair) {medianCm, sigmaCm}: 'photos_disagree' when two
    medians differ by more than max(2 cm, 3 x their combined sigma); else the median over the lowest sigma tier that has any
    (<= 1 cm, <= 2.5 cm, all), sigma = max(RMS of their sigmas, half the spread of their medians). Published when sigma <=
    2.5 cm, graded 'trusted' (<= 1 cm) or 'large'; else 'sigma_too_large' (the estimate +- sigma still reported)."""
    s = lambda c: math.inf if c.get('sigmaCm') is None else c['sigmaCm']
    for i, a in enumerate(cands):
        for b in cands[i + 1:]:
            if abs(a['medianCm'] - b['medianCm']) > max(AGREE_MIN_CM, AGREE_K * math.hypot(s(a), s(b))):
                return None, None, None, 'photos_disagree', None
    pool = [c for c in cands if s(c) <= TRUST_SIGMA_CM] or [c for c in cands if s(c) <= MAX_SIGMA_CM] or cands; m = [c['medianCm'] for c in pool]
    sigma = max(math.sqrt(float(np.mean([s(c) ** 2 for c in pool]))), (max(m) - min(m)) / 2)
    est = round(float(np.median(m)), 2); sigma = round(sigma, 2) if math.isfinite(sigma) else None
    if sigma is None or sigma > MAX_SIGMA_CM:
        return None, sigma, est, 'sigma_too_large', None
    return est, sigma, est, None, 'trusted' if sigma <= TRUST_SIGMA_CM else 'large'


def measure(ctx, obj, gid, scene, part, part_masks, frames, files=None, sweep_floor=None):
    n, d = ctx['floor']; S = ctx['S']
    out = dict(entityId=obj['id'], label=obj['label'], part=part or None,
               modelBottomCm=round(float(np.percentile((obj['mesh'][0] @ n + d) * S * 100, .5)), 2), photos=[], pairs=[])
    pts, missing = {}, []
    for k in sorted(obj['masks']):
        region = region_of(ctx, obj, k, part, part_masks)
        if region is None or region.sum() < 25:
            missing.append(k + 1); continue
        res = measure_photo(ctx, k, obj, gid, scene, region, bool(part), frames)
        if res is None:
            missing.append(k + 1); continue
        row, p, arr = res; out['photos'].append(row)
        if row['candidates'] >= MIN_LINES:
            pts[k] = p
        if files is not None:
            files[f"{obj['id'][:8]}-{(part or 'object').replace(' ', '_')[:24]}-photo{k + 1}.jpg"] = cl.crop(ctx['gray'][k], arr, f"{obj['id'][:8]} {part or 'object'} photo {k + 1}")
    out['noRegionPhotos'] = missing; out['grazedHits'] = sum(r['grazedHits'] for r in out['photos'])  # excluded from heights
    usable = [r for r in out['photos'] if r['candidates'] >= MIN_LINES]
    if not usable:
        out.update(path=None, valueCm=None, reason='no_part_mask' if part and not out['photos'] else 'too_few_lines'); return out
    if all(r['modelBacked'] for r in usable):
        used = [r for r in usable if r['n'] >= MIN_LINES]  # n: direct hits only (grazed lines excluded)
        if not used:
            out.update(path='model', valueCm=None, reason='too_few_direct_lines' if any(r['n'] + r['grazedHits'] >= MIN_LINES for r in usable)
                       else 'too_few_lines'); return out
        cands = used; out.update(path='model', usedPhotos=[r['photo'] for r in used],
                                 edgeIsModelBottom=any((r['edgeAtModelBottom'] or 0) >= .5 for r in used))
    else:
        ks = sorted(pts)  # see-through (or model missing): two-view line triangulation, model-free
        seg = {r['photo'] - 1: r['segEdgeAll']['uncertaintyPx'] for r in out['photos']}
        out['pairs'] = [two_view(ctx, a, b, pts[a], pts[b], (seg[a], seg[b])) for i, a in enumerate(ks) for b in ks[i + 1:]]
        cands = [p for p in out['pairs'] if p['accepted'] and p.get('n')]
        out.update(path='twoView', notBackedPhotos=[r['photo'] for r in usable if not r['modelBacked']], usedPairs=[p['photos'] for p in cands])
        if not cands:
            out.update(valueCm=None, reason='single_view_see_through' if len(ks) < 2 else 'two_view_rejected'); return out
    value, sigma, est, reason, grade = combine(cands)
    out.update(valueCm=value, sigmaCm=sigma, estimateCm=est, reason=reason, grade=grade)
    if est is not None:  # reported apart: the e-stop scale's uncertainty and the floor near the object
        rel = ((ctx.get('layer') or {}).get('scale') or {}).get('uncertaintyRelative')
        sweep = (sweep_floor or {}).get(obj['id'])
        out.update(scaleTermCm=None if rel is None else round(est * rel, 2), scaleRel=rel,
                   floorTermCm=None if sweep is None else round(-sweep, 2))  # height on the sweep floor minus on the report floor
    return out


def run(ctx, opts):
    if not ctx['floor']:
        return dict(status='no_floor', targets=[])
    scene = cl.scene_of([o['mesh'] for o in ctx['objects']])
    frames = ctx['pi3xFrames'] if 'pi3xFrames' in ctx else cl.pi3x_frames(ctx, opts['api']) if opts.get('api') else []
    files = {} if opts.get('crops', True) else None; rows = []; pm = opts.get('partMasks') or {}
    for t in opts.get('targets') or []:
        if t.get('part') and pm and t['part'] not in (pm.get(t['entityId']) or {}):  # workcell_part_masks.py lists every part it ran
            raise ValueError(f"part {t['part']!r} of {t['entityId']} is not in partMasks (has {sorted(pm.get(t['entityId']) or {})}): label typo?")
        gid, obj = next(((i, o) for i, o in enumerate(ctx['objects']) if o['id'] == t['entityId']), (None, None))
        if obj is None:
            rows.append(dict(entityId=t['entityId'], part=t.get('part'), valueCm=None, reason='no_object')); continue
        row = measure(ctx, obj, gid, scene, t.get('part'), pm, frames, files, opts.get('sweepFloorAboveReportCm'))
        if t.get('clickXY') is not None or t.get('anchorModel') is not None:  # the click exemplar, for reuse on an identical object
            row['anchor'] = anchor(ctx, obj, gid, scene, frames, t)
        rows.append(row)
    params = dict(edgeFraction=EDGE, gapPx=GAP_PX, pi3xBandPx=[BAND_PX[0], BAND_PX[-1], BAND_PX[1] - BAND_PX[0]], grazePx=GRAZE_PX,
                  grazedLines='counted, excluded from heights', minLines=MIN_LINES, minHitFraction=MIN_HIT,
                  seeThroughRelRange=SEE_THROUGH, maxTiltDeg=MAX_TILT_DEG, minOverlap=MIN_OVERLAP, straightPx=STRAIGHT_PX, minStraight=MIN_STRAIGHT,
                  maxSigmaCm=MAX_SIGMA_CM, trustSigmaCm=TRUST_SIGMA_CM, agreeMinCm=AGREE_MIN_CM, agreeK=AGREE_K, iqrToSigma=IQR_SIGMA,
                  anchorMarginM=ANCHOR_MARGIN_M, anchorMarginRel=ANCHOR_MARGIN_REL, modelBottomPx=BOTTOM_PX, stepWindowPx=STEP_WIN,
                  stepContrast=STEP_CONTRAST, stepAgreePx=STEP_AGREE_PX,
                  postHoc='maxTiltDeg, minOverlap, minStraight: added after the first run on the reports; band, grazed exclusion, '
                          'segmentation-edge term, missing-Pi3X bound, 2.5 cm gate: review findings 2026-10-06, set before this run')
    return dict(status='ok', nativeToMeters=ctx['S'], params=params, pi3xPhotos=[f['camera'] + 1 for f in frames], targets=rows, files=files or {})


def _pi3x_like(cam, meshes, step=4):
    """A Pi3X-like point map: true first hits on a grid of step-px cells, with its affine grid->pixel."""
    import open3d as o3d
    jj, ii = np.meshgrid(np.arange(cam['w'] // step), np.arange(cam['h'] // step)); u, v = step * jj.ravel() + step / 2, step * ii.ravel() + step / 2
    D = cl.rays(cam, u, v); t = cl.scene_of(meshes).cast_rays(o3d.core.Tensor(np.hstack([np.broadcast_to(cam['C'], D.shape), D]).astype(np.float32)))['t_hit'].numpy()
    P = np.where(np.isfinite(t)[:, None], cam['C'] + t[:, None] * D, 0).reshape(jj.shape + (3,))
    return P, np.array([[step, 0, step / 2], [0, step, step / 2]], float)


def _check():
    """(1) panel 24 cm up behind a post: 24 cm on the model +- a small sigma, lines behind the post dropped, the edge flagged as
    the model's own bottom (exact model); a part mask 3 px short of the edge: the photo's intensity step corrects it back to 24
    and the flag clears; a click on the panel where the post hides it from the other camera is no prompt there; a part label
    typo raises; no Pi3X maps: the 5 %-of-range bound, no value (never a small sigma); the panel model 1 cm too high: the grazed
    lines are excluded, no extrapolated value.
    (2) a part: a recessed piece (bottom 24 cm) beside a lower front plate (bottom 20 cm), one placed (rotated, scaled) model;
    a click on the recessed piece in photo 1 projects onto it in photo 2, its model-frame anchor re-places to the same point, and
    the part mask of the recessed piece reads 24 while the whole object reads lower. (3) a rail 20 cm up whose displayed model
    sits 30 cm too deep (every ray hits it, Pi3X disagrees): two-view reads 20 cm, sigma <= 1 cm; (4) the same rail from two
    cameras at one height and distance (planes ~parallel): sigma >> 1 cm, no value. (5) combine(): photos that disagree; the
    2.5 cm gate and its trusted / large grade."""
    floor = (np.array([0, 0, 1.]), 0.)
    look, box = cl._look, cl._box
    ident = dict(position=[0, 0, 0], quaternion=[0, 0, 0, 1], scale=[1, 1, 1])
    doc_of = lambda ids, T=ident: {'assets': [], 'entities': [{'id': i, 'activeModelRepresentationId': 'r', 'currentModelTransform': T,
                                                              'representations': [{'id': 'r', 'transform': T}]} for i in ids]}

    def ctx_of(cams, meshes, models, ids, frames=True):
        gray, masks = cl._render(cams, meshes)
        objs = [dict(id=i, label=i, kind='box', mesh=m, masks={k: masks[k][j] for k in range(len(cams)) if masks[k][j].any()}) for j, (i, m) in enumerate(zip(ids, models))]
        ctx = dict(cams=cams, gray=gray, images=gray, objects=objs, S=1.0, floor=floor, doc=doc_of(ids))
        ctx['pi3xFrames'] = [dict(camera=k, P=P, affine=A) for k, c in enumerate(cams) for P, A in [_pi3x_like(c, meshes)]] if frames else []
        return ctx
    panel, post = box((-.6, 2.95, .24), (.6, 3.0, 1.0)), box((-.05, 2.0, 0), (.05, 2.1, .9))
    cams = [look(np.array([-.4, 0, 1.5]), np.array([0, 3, .4])), look(np.array([.5, .2, 1.4]), np.array([0, 3, .4]))]
    ctx = ctx_of(cams, [panel, post], [panel, post], ['panel', 'post'])
    hidden = np.array([-.243, 2.95, .508])  # on the panel; the post hides it from camera 2
    click = [round(float(x), 1) for x in wsc.project(cams[0], hidden[None])[0][0]]
    short = {str(k): _polys(m & np.roll(m, -3, axis=0)) for k, m in ctx['objects'][0]['masks'].items()}  # 3 px short of the edge
    res = run(ctx, dict(targets=[{'entityId': 'panel', 'part': 'p', 'clickPhoto': 1, 'clickXY': click}, {'entityId': 'panel'}, {'entityId': 'post'}],
                        partMasks={'panel': {'p': short}}, crops=False))
    r = {(t['entityId'], t['part']): t for t in res['targets']}
    pan, sh = r[('panel', None)], r[('panel', 'p')]
    assert pan['path'] == 'model' and abs(pan['valueCm'] - 24) < .3 and pan['grade'] == 'trusted' and pan['edgeIsModelBottom'], pan
    assert all(p['dropped']['otherMaskBelow'] > 0 and p['modelBacked'] and p['pi3xGapProfile'] and abs(p['pi3xRangeDiffRel']) < .005 for p in pan['photos']), pan['photos']
    assert abs(sh['valueCm'] - 24) < .3 and not sh['edgeIsModelBottom'], sh
    assert all(p['segEdge']['mode'] == 'corrected' and 2 < p['segEdge']['offsetPx'] < 4 and p['segEdge']['correctionCm'] < -1 for p in sh['photos']), sh['photos']
    assert abs(r[('post', None)]['valueCm']) < .5, r[('post', None)]
    an = sh['anchor']
    assert an['anchorSource'] == 'model' and np.abs(np.array(an['anchorWorld']) - hidden).max() < .01, an
    assert [p['why'] for p in an['photos']] == ['click', 'other_model_in_front'], an['photos']
    try:
        run(ctx, dict(targets=[{'entityId': 'panel', 'part': 'P'}], partMasks={'panel': {'p': short}}, crops=False)); raise RuntimeError('a part typo passed')
    except ValueError as error:
        assert 'label typo' in str(error), error
    nf = run(ctx_of(cams, [panel, post], [panel, post], ['panel', 'post'], frames=False), dict(targets=[{'entityId': 'panel'}], crops=False))['targets'][0]
    assert nf['valueCm'] is None and nf['reason'] == 'sigma_too_large' and abs(nf['estimateCm'] - 24) < .3, nf
    assert all(p['sigmaTerms']['rangeSource'] == 'no_pi3x_5pct_of_range' and p['sigmaTerms']['rangeCm'] > 3 for p in nf['photos']), nf['photos']
    raised = (panel[0] + [0, 0, .01], panel[1])  # the panel model 1 cm too high: the boundary rays graze under it
    rg = run(ctx_of(cams, [panel, post], [raised, post], ['panel', 'post']), dict(targets=[{'entityId': 'panel'}], crops=False))['targets'][0]
    assert rg['grazedHits'] > 100 and pan['grazedHits'] == 0 and rg['valueCm'] is None, (rg, pan)
    # (2) one placed model of two pieces; the part mask (as workcell_part_masks.py would supply it) covers the recessed piece
    recessed, plate = box((-.3, 3.0, .24), (0., 3.05, 1.0)), box((0., 2.95, .20), (.3, 3.0, 1.0))
    V = np.vstack([recessed[0], plate[0]]); F = np.vstack([recessed[1], plate[1] + len(recessed[0])])
    T = dict(position=[.1, 2.9, .5], quaternion=[0, 0, math.sin(.3), math.cos(.3)], scale=[2., 2., 2.])
    local = to_model(V, T); assert np.abs(wsc.placed(local, T) - V).max() < 1e-9
    gray, masks = cl._render(cams, [recessed, plate])
    obj = dict(id='post2', label='post2', kind='mesh', mesh=(V, F), masks={k: masks[k][0] | masks[k][1] for k in range(2)})
    ctx2 = dict(cams=cams, gray=gray, images=gray, objects=[obj], S=1.0, floor=floor, doc=doc_of(['post2'], T),
                pi3xFrames=[dict(camera=k, P=P, affine=A) for k, c in enumerate(cams) for P, A in [_pi3x_like(c, [recessed, plate])]])
    ys, xs = np.nonzero(masks[0][0]); sel = ys > np.percentile(ys, 80)  # the click: on the recessed piece, just above its lower edge
    pr = prompts(ctx2, dict(targets=[{'entityId': 'post2', 'part': 'recessed piece', 'clickPhoto': 1, 'clickXY': [float(np.median(xs[sel])), float(np.percentile(ys, 85))]}]))
    a2 = pr['anchors'][0]; (u1, v1), = [p['point'] for p in pr['prompts'] if p['photoIndex0'] == 1]
    assert len(pr['prompts']) == 2 and masks[1][0][round(v1), round(u1)], (pr, 'projected prompt not on the recessed piece')
    re = prompts(ctx2, dict(targets=[{'entityId': 'post2', 'part': 'recessed piece', 'anchorModel': a2['anchorModel']}]))['anchors'][0]
    assert re['anchorSource'] == 'stored' and np.abs(np.array(re['anchorWorld']) - a2['anchorWorld']).max() < 1e-5, (re, a2)
    pm = {'post2': {'recessed piece': {str(k): _polys(masks[k][0]) for k in range(2)}}}
    r2 = {t['part']: t for t in run(ctx2, dict(targets=[{'entityId': 'post2', 'part': 'recessed piece'}, {'entityId': 'post2'}], partMasks=pm, crops=False))['targets']}
    assert abs(r2['recessed piece']['valueCm'] - 24) < .5, r2['recessed piece']
    assert r2[None]['estimateCm'] < 23, r2[None]
    # (3) rail 20 cm up, displayed model 30 cm too deep; cameras at different heights / distances
    rail = box((-.6, 3.0, .20), (.6, 3.04, .28))
    cams3 = [look(np.array([-.3, 0, 1.6]), np.array([0, 3, .3])), look(np.array([.3, 1.5, .35]), np.array([0, 3, .3]))]
    ctx3 = ctx_of(cams3, [rail], [box((-.7, 3.3, 0.), (.7, 3.34, .6))], ['rail'])  # every ray hits the model, 30 cm too far
    r3 = run(ctx3, dict(targets=[{'entityId': 'rail'}], crops=False, sweepFloorAboveReportCm={'rail': .5}))['targets'][0]
    assert all(p['hitFraction'] > .9 and p['pi3xRangeDiffRel'] < -.05 and not p['modelBacked'] for p in r3['photos']), r3['photos']
    assert r3['path'] == 'twoView' and abs(r3['valueCm'] - 20) < .5 and r3['sigmaCm'] <= TRUST_SIGMA_CM and r3['grade'] == 'trusted' and r3['floorTermCm'] == -.5, r3
    # (4) two cameras at one height and distance: the back-projected planes are nearly the same plane
    cams4 = [look(np.array([-.4, 0, 1.5]), np.array([0, 3, .3])), look(np.array([.4, 0, 1.5]), np.array([0, 3, .3]))]
    r4 = run(ctx_of(cams4, [rail], [(rail[0] + [0, .3, 0], rail[1])], ['rail']), dict(targets=[{'entityId': 'rail'}], crops=False))['targets'][0]
    assert r4['valueCm'] is None and r4['pairs'][0]['sigmaCm'] > 10 * TRUST_SIGMA_CM, r4  # (and its two photos share 2 % of the line)
    # (5) the cross-photo rule
    c = lambda m, s: dict(medianCm=m, sigmaCm=s)
    assert combine([c(24., .3), c(27., .4)])[3] == 'photos_disagree'
    assert combine([c(24., .3), c(24.6, .4)]) == (24.3, .35, 24.3, None, 'trusted') and combine([c(24., .3), c(27., 1.5)])[:2] == (24., .3)
    assert combine([c(24., 1.5), c(24.4, 2.)]) == (24.2, 1.77, 24.2, None, 'large')
    assert combine([c(24., 2.6), c(24.4, 3.)])[::3] == (None, 'sigma_too_large')
    print('lower_edge self-test passed: panel', pan['valueCm'], '+-', pan['sigmaCm'], '| 3 px short part', sh['valueCm'], '(step',
          sh['photos'][0]['segEdge']['offsetPx'], 'px) | no Pi3X: +-', nf['sigmaCm'], '| 1 cm high model: grazed', rg['grazedHits'], 'excluded |', 'part', r2['recessed piece']['valueCm'],
          'whole', r2[None]['estimateCm'], 'rail two-view', r3['valueCm'], '+-', r3['sigmaCm'], f"(dihedral {r3['pairs'][0]['dihedralDeg']})",
          'parallel planes', r4['pairs'][0]['dihedralDeg'], 'deg: sigma', r4['pairs'][0]['sigmaCm'], 'rejected')


def _polys(mask):
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    return [c.reshape(-1, 2).tolist() for c in cs if len(c) >= 3]


if __name__ == '__main__':
    if len(sys.argv) == 1:
        _check()
    else:  # on-prem: python lower_edge.py --view V.json --photos-dir D --photo ID=FILE,... --layer-url URL --api ORIGIN --opts J (--out O.json | --prompts-out P.json)
        import argparse
        a = argparse.ArgumentParser(); [a.add_argument(f'--{x}', default='') for x in ('view', 'photos-dir', 'photo', 'layer-url', 'api', 'out', 'opts', 'prompts-out')]
        a = a.parse_args(); pairs = dict(p.split('=', 1) for p in a.photo.split(','))
        ctx = wsc.load_report(Path(a.view).read_bytes(), {i: (Path(a.photos_dir) / f).read_bytes() for i, f in pairs.items()}, a.layer_url, a.api)
        opts = {'api': a.api, **(json.loads(Path(a.opts).read_text()) if a.opts else {})}
        if a.prompts_out:  # click targets -> prompt points for modal_apps/workcell_part_masks.py --prompts
            Path(a.prompts_out).write_text(json.dumps(prompts(ctx, opts), indent=1, ensure_ascii=False) + '\n'); sys.exit()
        result = run(ctx, opts)
        out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
        for name, data in result.pop('files').items():
            (out.parent / f'lower_edge-{name}').write_bytes(data)
        out.write_text(json.dumps(dict(results=dict(lower_edge=result), nativeToMeters=ctx['S'], layerRevision=(ctx['layer'] or {}).get('revisionId')),
                                  indent=1, ensure_ascii=False) + '\n')
