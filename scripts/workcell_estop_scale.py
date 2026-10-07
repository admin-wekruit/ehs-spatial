"""Metric scale of a workcell photo report from its emergency stop, starting from the e-stop's masks only (on-prem CLI).

The e-stop is a stack of coaxial round parts (red lip, yellow body, grey base). In each photo:
1. seed: from the mask, the red component the mask covers most is the head; the yellow component under it is the body; the grey
   rows under the body that span >= 80 % of its width are the base. Axis column = median centre of the body's lower half
   (sub-pixel gradient edges), row span = head top .. base bottom. No hand-placed pixels.
2. depth: with >= 2 masked photos the axis points (column, mid row) are triangulated (least-squares ray intersection); with one
   photo it is the median of the run's Pi3X points on the body pixels (needs --run).
3. fit: the coaxial symmetric cylinder fit (research-notes/cell030-sept-pipeline-2026-10-05/estop_cylinder.py fit(), copied
   unchanged): one projected axis (image of the floor normal), straight sides symmetric about it, one width per part.
4. scale: scripts/reference_object_scale.py joint_scale over every (photo, feature): width x depth / fx against the spec
   (default red lip 4 cm + yellow body 8 cm), accepted only when every feature is within 4 %.

Masks: --entity ENTITY (the report observations' polygons of that entity, view mode) and/or --mask ID=PATH,... where PATH is a
mask image (nonzero = inside; resized to the photo) or a folder of candidate PNGs tried in name order, e.g. the SAM 3 output of
modal_apps/workcell_estop_mask.py. Per photo the first mask that marks a whole e-stop (yellow body 1.6-2.6 x the red head and
centred under it, a grey base, not cut by the photo border) is used; a photo without one is skipped and listed. A photo may be
a higher-resolution original of the report photo (same framing): its camera K and the polygons are rescaled (pixel centres).

  python scripts/workcell_estop_scale.py --view VIEW.json [--layer LAYER.json] --photos-dir DIR --photo IMAGE_ID=FILE,... \
      [--entity ENTITY_ID] [--mask IMAGE_ID=MASK.png|SAM_DIR,...] [--resolution-aware] --out SCALE.json
  python scripts/workcell_estop_scale.py --run RUN [--photo FRAME_ID=FILE,...] --mask FRAME_ID=MASK.png|SAM_DIR,... --out SCALE.json
  python scripts/workcell_estop_scale.py --self-test
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
from scipy.special import ndtr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reference_object_scale import joint_scale  # noqa: E402

SPEC = 'red lip=0.04,yellow body=0.08'  # user, 2026-10-04: red head 4 cm, yellow body max diameter 8 cm (total height 10 cm)


HEAD_PX_REF = 56  # head width (px) of the e-stop on which fit()'s pixel windows were set (030 photo 2, cell030-sept-pipeline)


# --- fit(): research-notes/cell030-sept-pipeline-2026-10-05/estop_cylinder.py (sha256 5fa55456...), unchanged unless head_px ---
# head_px (opt-in, --resolution-aware): the search / row / profile windows and the pre-blur are in units of q = head_px / HEAD_PX_REF
# px (fixed in head widths); the edge-blur model (step sigma bounds, acceptance, fallback, step window) is in units of b = max(q, 1)
# px, because a camera's edge blur does not shrink below ~1 px on a small e-stop. head_px=None: q = b = 1, the original fit() exactly.
def fit(image_bgr, K, c2w, up, P3, seed_xy, top_row, bottom_row, guess=(None, None, None), head_px=None):
    q = 1.0 if head_px is None else head_px / HEAD_PX_REF; b = max(q, 1.0)
    R, C = c2w[:3, :3].T, c2w[:3, 3]
    proj = lambda X: (lambda x: x[:2] / x[2])(K @ (R @ (np.asarray(X, float) - C)))
    a0, a1 = proj(P3), proj(P3 + .05 * np.asarray(up, float)); u0 = (a1 - a0) / np.linalg.norm(a1 - a0)
    if u0[1] > 0: u0 = -u0
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32); hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV).astype(int)
    smooth = cv2.GaussianBlur(rgb, (0, 0), 1.0 * q)
    def frame(theta):
        c, s = np.cos(theta), np.sin(theta); u = np.array([c * u0[0] - s * u0[1], s * u0[0] + c * u0[1]])
        n = np.array([-u[1], u[0]]); return u, (n if n[0] > 0 else -n)
    def axis(a, theta, row):
        u, n = frame(theta); base = np.array(seed_xy, float) + a * n; t = (row - base[1]) / u[1]; return base + t * u, n
    # colour sequence along the axis (seed axis) -> spans
    seq = []
    for row in range(top_row - round(4 * q), bottom_row + round(4 * q)):
        p, _ = axis(0, 0, row); x, y = int(round(p[0])), row
        h, s, v = hsv[y, x]
        seq.append((row, 'red' if (h < 10 or h > 165) and s > 70 else 'yellow' if 15 < h < 40 and s > 70 else 'grey' if s < 60 and 60 < v < 210 else None))
    def span(kind):
        rows = [r for r, k in seq if k == kind]; return (min(rows), max(rows)) if rows else None
    red, yel, gry = span('red'), span('yellow'), span('grey')
    if not (red and yel and gry): raise ValueError(f'parts not found along the axis: {red}, {yel}, {gry}')
    parts = {'red lip': (red[0] + max(1, round(q)), red[0] + max(int(.4 * (red[1] - red[0])), round(4 * q))),
             'yellow body': (yel[0] + int(.35 * (yel[1] - yel[0])), yel[1] - round(2 * q)),
             'grey cylinder': (max(gry[0], yel[1]) + round(2 * q), gry[1] - round(2 * q))}
    gx = np.stack([cv2.Sobel(smooth[..., c], cv2.CV_32F, 1, 0, ksize=3) for c in range(3)], -1)
    gy = np.stack([cv2.Sobel(smooth[..., c], cv2.CV_32F, 0, 1, ksize=3) for c in range(3)], -1)
    _, n0 = frame(0)
    # part-colour likelihoods: a silhouette is where that part's colour starts or stops, so neighbouring wood, cables or boxes
    # (other colours) give no evidence for it
    H, Sat, Val = hsv[..., 0].astype(np.float32), hsv[..., 1].astype(np.float32) / 255, hsv[..., 2].astype(np.float32) / 255
    hue_red = np.minimum(np.abs(H - 0), np.abs(H - 180))
    like = {'red': Sat * np.exp(-np.square(hue_red / 8)) * (Val > .25),
            'yellow': Sat * np.exp(-np.square((H - 27) / 7)) * (Sat > .45),
            'grey': (1 - Sat) * np.exp(-np.square((Val - .6) / .2)) * (Sat < .3)}
    edge = {}
    for k, L in like.items():
        L = cv2.GaussianBlur(L.astype(np.float32), (0, 0), 1.0 * q); like[k] = L
        edge[k] = np.abs(cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3) * n0[0] + cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3) * n0[1])
    caps = {k: float(np.percentile(v, 99.5)) or 1.0 for k, v in edge.items()}
    part_kind = {'red lip': 'red', 'yellow body': 'yellow', 'grey cylinder': 'grey'}
    def remap(img, xs, ys):
        size = xs.size; pad = -size % 1024
        X = np.pad(xs.ravel(), (0, pad)).astype(np.float32).reshape(-1, 1024); Y = np.pad(ys.ravel(), (0, pad)).astype(np.float32).reshape(-1, 1024)
        return cv2.remap(img, X, Y, cv2.INTER_LINEAR).ravel()[:size].reshape(xs.shape)
    def scores(theta, offsets, widths, rows, kind):
        u, n = frame(theta); A_, W_, Rr = np.meshgrid(offsets, widths, rows, indexing='ij')
        t = (Rr - (seed_xy[1] + A_ * n[1])) / u[1]; cx, cy = seed_xy[0] + A_ * n[0] + t * u[0], seed_xy[1] + A_ * n[1] + t * u[1]
        return sum(np.minimum(remap(edge[kind], cx + s * W_ / 2 * n[0], cy + s * W_ / 2 * n[1]), caps[kind]).mean(-1) / caps[kind] for s in (-1, 1))
    # coarse width guesses from the colour run at mid rows
    def run_width(kind, row):
        p, _ = axis(0, 0, row); x = int(round(p[0])); h = hsv[row, :, 0]; s = hsv[row, :, 1]; v = hsv[row, :, 2]
        m = ((h < 10) | (h > 165)) & (s > 70) if kind == 'red' else (h > 15) & (h < 40) & (s > 70) if kind == 'yellow' else (s < 60) & (v > 60) & (v < 210)
        l = r = x
        while l > 0 and m[l - 1]: l -= 1
        while r < m.size - 1 and m[r + 1]: r += 1
        return r - l
    # e-stop structure (reference object): the red lip's colour run is the reliable anchor (nothing nearby is that red); the yellow
    # body and grey base are searched at 1.75-2.35 times it, so a yellowish post or a dark box beside the e-stop cannot pull them
    r_guess = guess[0] or run_width('red', (parts['red lip'][0] + parts['red lip'][1]) // 2)
    ranges = {'red lip': (.8 * r_guess, 1.2 * r_guess), 'yellow body': (1.75 * r_guess, 2.35 * r_guess), 'grey cylinder': (1.75 * r_guess, 2.35 * r_guess)}
    offsets = np.arange(-8, 8.01, .25) * q; best = None
    for theta in np.radians(np.arange(-2, 2.01, .25)):
        grids = {name: np.arange(lo, hi + .01 * q, .25 * q) for name, (lo, hi) in ranges.items()}
        per = {name: scores(theta, offsets, grids[name], np.arange(r0, r1 + 1, 1.0), part_kind[name]) for name, (r0, r1) in parts.items()}
        total = sum(v.max(1) for v in per.values()); i = int(np.argmax(total))
        if best is None or total[i] > best[0]:
            best = (float(total[i]), float(offsets[i]), float(theta), {name: float(grids[name][int(np.argmax(v[i]))]) for name, v in per.items()})
    _, a, theta, widths = best
    step = lambda x, lo, hi, x0, s: lo + (hi - lo) * ndtr((x - x0) / s)
    result = {}
    for name, (r0, r1) in parts.items():
        w = widths[name]; offs = np.arange(-8, 8.01, .25) * q; sides = {}
        for sign, label in ((-1, 'left'), (1, 'right')):
            # 1) the part's colour membership locates the silhouette robustly (not the neighbouring wood, cable or box)
            prof, rgbp = [], []
            for row in np.arange(r0, r1 + 1, 1.0):
                p, n = axis(a, theta, row); e = p + sign * w / 2 * n
                prof.append(remap(like[part_kind[name]], e[0] + offs * n[0], e[1] + offs * n[1]))
                rgbp.append(np.stack([remap(smooth[..., c], e[0] + offs * n[0], e[1] + offs * n[1]) for c in range(3)], -1))
            y = np.mean(prof, 0)
            (_, _, xc, sc), _ = curve_fit(step, offs, y, p0=[y[0], y[-1], 0, b], bounds=([-1, -1, -8 * q, .2 * b], [2, 2, 8 * q, 6 * b]), maxfev=20000)
            # 2) intensity mixes linearly at a silhouette, so its step centre is unbiased: fit it in +-4 px around the colour edge
            rgbp = np.mean(rgbp, 0); win = np.abs(offs - xc) <= 4 * b
            k = int(np.argmax(np.abs(rgbp[win][-3:].mean(0) - rgbp[win][:3].mean(0)))); yy = rgbp[win][:, k]
            try:
                (_, _, x0, s), _ = curve_fit(step, offs[win], yy, p0=[yy[0], yy[-1], xc, b], bounds=([-50, -50, xc - 4 * b, .2 * b], [350, 350, xc + 4 * b, 6 * b]), maxfev=20000)
                ok = s <= 3 * b
            except RuntimeError:
                ok = False
            sides[label] = (float(x0), float(s)) if ok else (float(xc), max(float(sc), 3.0 * b))
            sides[label + 'Source'] = 'intensity step' if ok else 'colour membership (intensity step unusable)'
        d = np.array([w / 2 - sides['left'][0], w / 2 + sides['right'][0]]); wt = 1 / np.square([sides['left'][1], sides['right'][1]])
        result[name] = {'rows': [int(r0), int(r1)], 'symmetricWidthPx': float(2 * (d * wt).sum() / wt.sum()), 'gradientWidthPx': w,
                        'sideDistancesPx': d.tolist(), 'blurPx': [sides['left'][1], sides['right'][1]], 'edgeSources': [sides['leftSource'], sides['rightSource']]}
    return {'axisOffsetPx': a, 'axisExtraDeg': float(np.degrees(theta)), 'vanishingTiltDeg': float(np.degrees(np.arctan2(u0[0], -u0[1]))),
            'spans': {'red': red, 'yellow': yel, 'grey': gry}, 'parts': result, 'axis': lambda row: axis(a, theta, row)}
# --- end of fit() --------------------------------------------------------------------------------------------------------


def _colours(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV); h, s, v = (hsv[..., i].astype(int) for i in range(3))
    return ((h <= 8) | (h >= 170)) & (s > 110) & (v > 60), (h >= 18) & (h <= 34) & (s > 110) & (v > 110), s, v


def _edge_x(profile, guess, half=6):
    """Sub-pixel column of the strongest gradient within +-half of guess (parabola through the peak)."""
    lo, hi = max(int(guess) - half, 1), min(int(guess) + half, len(profile) - 2)
    k = lo + int(np.argmax(profile[lo:hi + 1])); a, b, c = profile[k - 1], profile[k], profile[k + 1]; den = a - 2 * b + c
    return k + (.5 * (a - c) / den if den < 0 else 0.)


def seed_from_mask(image_bgr, mask, grow=.2):
    """Axis column, head-top row, base-bottom row and body pixels of the e-stop that `mask` marks (full-image coordinates).

    The mask only has to cover part of the red head (a report polygon that also holds the bracket, or a SAM mask, both work):
    the red component it covers most (mask grown by `grow` x the square root of its area) is the head; the parts below are found
    by colour. Every later window is a multiple of the head's measured width (search bands, body-pick weight, edge windows, base
    rows), so the seed does not depend on the image resolution. Raises ValueError when the marked object does not have the
    e-stop's structure (yellow body 1.6-2.6 x the head, centred under it, on a grey base) or touches the photo border (cut
    e-stop)."""
    ys, xs = np.nonzero(mask)
    if not len(ys):
        raise ValueError('empty e-stop mask')
    H, W = image_bgr.shape[:2]
    # 1) the head: the red component (whole photo, so it is never cut by a window) that the grown mask covers most
    red = _colours(np.ascontiguousarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)))[0]
    n, lab = cv2.connectedComponents(red.astype(np.uint8), connectivity=8)
    g_ = max(1, int(round(grow * np.sqrt(len(ys))))); a0, a1, b0, b1 = max(ys.min() - g_, 0), ys.max() + g_ + 1, max(xs.min() - g_, 0), xs.max() + g_ + 1
    inside = cv2.dilate(mask[a0:a1, b0:b1].astype(np.uint8), np.ones((2 * g_ + 1,) * 2, np.uint8)) > 0
    cover = np.bincount(lab[a0:a1, b0:b1][inside], minlength=n)[1:]
    if not cover.size or cover.max() == 0:
        raise ValueError('no red head under the mask')
    ys, xs = np.where(lab == 1 + int(np.argmax(cover)))
    head = xs.max() - xs.min() + 1
    # 2) everything else in a window around the head, sized in head widths (the e-stop is ~2.2 heads wide, ~2.5 tall)
    x0, y0 = max(int(xs.mean() - 3 * head), 0), max(int(ys.min() - head), 0)
    crop = image_bgr[y0:int(ys.max() + 5 * head) + 1, x0:int(xs.mean() + 3 * head) + 1]
    rgb = np.ascontiguousarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)); _, yellow, s_, v_ = _colours(rgb)
    ys, xs = ys - y0, xs - x0
    band = np.zeros_like(yellow); band[ys.max():ys.max() + int(3.4 * head), max(int(xs.mean() - 2.4 * head), 0):int(xs.mean() + 2.4 * head)] = True
    n, lab, st, cen = cv2.connectedComponentsWithStats((yellow & band).astype(np.uint8), 8)
    if n < 2:
        raise ValueError('no yellow body under the red head')
    near = np.array([xs.mean(), ys.max() + .8 * head])  # nearest to just under the head, larger parts preferred (area / head)
    body = lab == 1 + int(np.argmin([np.hypot(*(cen[i] - near)) - st[i, cv2.CC_STAT_AREA] / head for i in range(1, n)]))
    by, bx = np.where(body)
    lab_img = gaussian_filter(cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32), sigma=(1.0, 1.0, 0))
    g = np.sqrt((np.gradient(lab_img, axis=1) ** 2).sum(-1)); half = max(2, int(round(head / 8)))
    collar = []
    for r in range(by.min() + int(.5 * (by.max() - by.min())), by.max() + 1):
        cols = np.where(body[r])[0]
        if len(cols) >= max(3, .12 * head):
            collar.append((_edge_x(g[r], cols.min() - .5, half), _edge_x(g[r], cols.max() + .5, half)))
    if not collar:
        raise ValueError('yellow body too small')
    cw = np.percentile([r - l for l, r in collar], 90); axis = float(np.median([(l + r) / 2 for l, r in collar]))
    if not (1.6 <= cw / head <= 2.6 and abs(axis - xs.mean()) <= .3 * head):
        raise ValueError(f'not an e-stop: body {cw:.0f} px vs head {head} px, centres {axis - xs.mean():+.0f} px apart')
    grey = (s_ < 50) & (v_ > 70) & (v_ < 225); ring = []
    for r in range(by.max() + 1, min(by.max() + int(1.2 * head), rgb.shape[0])):  # base rows: a grey run >= 80 % of the body width
        k, _, st, _ = cv2.connectedComponentsWithStats(grey[r:r + 1].astype(np.uint8), 4)
        if k > 1 and st[1:, cv2.CC_STAT_WIDTH].max() > .8 * cw:
            ring.append(r)
        elif ring:
            break
    if not ring:
        raise ValueError('no grey base under the yellow body')
    if min(ys.min() + y0, bx.min() + x0, xs.min() + x0) < 2 or max(bx.max(), xs.max()) + x0 > W - 3 or max(ring) + y0 > H - 3:
        raise ValueError('e-stop cut by the photo border')
    return dict(axisX=axis + x0, headTopRow=int(ys.min() + y0), ringBottomRow=int(max(ring) + y0), bodyPixels=np.stack([bx + x0, by + y0], 1),
                headPx=int(head), bodyPx=float(cw))


def pick_seed(image_bgr, candidates):
    """The first (source, mask) candidate that marks an e-stop: (seed, source, rejected [reasons])."""
    rejected = []
    for source, mask in candidates:
        try:
            return seed_from_mask(image_bgr, mask), source, rejected
        except ValueError as error:
            rejected.append(f'{source}: {error}')
    raise ValueError('; '.join(rejected) or 'no mask')


def scaled_K(K, sx, sy):
    """K of the same camera on an image resized by (sx, sy), pixel-centre convention x' = (x + .5) s - .5."""
    K = np.array(K, float); return np.array([[K[0, 0] * sx, 0, (K[0, 2] + .5) * sx - .5], [0, K[1, 1] * sy, (K[1, 2] + .5) * sy - .5], [0, 0, 1]])


def polygons_mask(polygons, shape, sx=1., sy=1.):
    """Even-odd fill of pixel-centre polygons, rescaled to the image (as scripts/workcell_shape_check.py polygon_mask)."""
    m = np.zeros(shape, np.uint8)
    for poly in polygons:
        p = (np.array(poly, float) + .5) * [sx, sy] - .5; one = np.zeros(shape, np.uint8)
        cv2.fillPoly(one, [np.round(p).astype(np.int32)], 1); m ^= one
    return m.astype(bool)


def triangulate(rays):
    """Least-squares intersection of rays (origin, direction)."""
    A, b = np.zeros((3, 3)), np.zeros(3)
    for o, d in rays:
        d = d / np.linalg.norm(d); P = np.eye(3) - np.outer(d, d); A += P; b += P @ o
    return np.linalg.solve(A, b)


def measure(photos, up, specs, pointmap=None, head_unit=False):
    """photos: [{id, image (BGR), masks: [(source, mask)], K, c2w}]; pointmap(id, pixels) -> 3D points (single photo only).
    A photo none of whose masks marks a whole e-stop is left out (and listed under 'skipped'). head_unit: fit() windows in units of
    the seed's head width (opt-in resolution-aware mode; default False = the original pixel windows)."""
    seeds, skipped = {}, {}
    for p in photos:
        try:
            seeds[p['id']], p['maskSource'], p['rejectedMasks'] = pick_seed(p['image'], p['masks'])
        except ValueError as error:
            skipped[p['id']] = str(error)
    photos = [p for p in photos if p['id'] in seeds]
    if not photos:
        raise ValueError(f'no photo shows a whole e-stop: {skipped}')
    if len(photos) >= 2:
        rays = []
        for p in photos:
            s = seeds[p['id']]; M = p['c2w']
            rays.append((M[:3, 3], M[:3, :3] @ (np.linalg.inv(p['K']) @ [s['axisX'], (s['headTopRow'] + s['ringBottomRow']) / 2, 1])))
        P3, depth = triangulate(rays), 'triangulated axis points (column, mid row) of %d photos' % len(photos)
    elif pointmap:
        P3, depth = np.median(pointmap(photos[0]['id'], seeds[photos[0]['id']]['bodyPixels']), axis=0), 'median Pi3X point on the yellow body pixels'
    else:
        raise ValueError('one photo: its depth needs the run pointmap (--run)')
    rows, out = [], {}
    for p in photos:
        s = seeds[p['id']]; M = p['c2w']
        res = fit(p['image'], p['K'], M, up, P3, (s['axisX'], (s['headTopRow'] + s['ringBottomRow']) / 2), s['headTopRow'], s['ringBottomRow'],
                  head_px=s['headPx'] if head_unit else None)
        res.pop('axis')
        z = float((M[:3, :3].T @ (P3 - M[:3, 3]))[2]); fx = float(p['K'][0, 0]); w = {k: v['symmetricWidthPx'] for k, v in res['parts'].items()}
        out[p['id']] = dict(file=p.get('file'), maskSource=p['maskSource'], rejectedMasks=p['rejectedMasks'], seed={k: v for k, v in s.items() if k != 'bodyPixels'},
                            depthNative=z, fx=fx, widthsPx=w, **res)
        rows += [{'photo': p['id'], 'feature': f'{name} {spec * 100:g} cm', 'part': name, 'specM': spec, 'measuredNative': w[name] * z / fx}
                 for name, spec in specs.items()]
    result = joint_scale(rows)
    per = {name: float(np.exp(np.mean([np.log(r['specM'] / r['measuredNative']) for r in rows if r['part'] == name]))) for name in specs}
    if len(per) == 2:  # the reports' interval: half the spread of the single-feature scales over the joint scale
        a, b = per.values(); result['uncertaintyRelative'] = abs(a - b) / 2 / result['nativeToMeters']
    for p in out.values():  # every part in cm at the joint scale (parts outside --spec, e.g. the grey base, are independent)
        p['partsCm'] = {k: w * p['depthNative'] / p['fx'] * result['nativeToMeters'] * 100 for k, w in p['widthsPx'].items()}
    extra = {'fitWindows': f'--resolution-aware: windows in q = headPx / {HEAD_PX_REF} px, edge blur in max(q, 1) px'} if head_unit else {}
    return dict(**result, perFeature=per, P3=P3.tolist(), depthFrom=depth, photos=out, skipped=skipped, **extra)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_inputs(a):
    pairs = dict(x.split('=', 1) for x in a.photo.split(',')) if a.photo else {}
    masks = dict(x.split('=', 1) for x in a.mask.split(',')) if a.mask else {}
    photos, pointmap = [], None
    if a.run:
        run = Path(a.run); man = json.loads((run / 'manifest.json').read_text())
        frames = {f['frame_id']: f for f in man['frames']}
        up = np.array(json.loads((run / 'evidence/floor.json').read_text())['up_native'], float)
        A = {k: np.array(f['input_to_canonical_pixel_centres'], float) for k, f in frames.items()}
        for fid in pairs or masks:
            f = frames[fid]; d = run / 'geometry/frames' / fid
            K = np.linalg.inv(A[fid]) @ np.load(d / 'intrinsics.npy').astype(float)
            file = Path(a.photos_dir or run) / pairs[fid] if fid in pairs else run / f['input']
            photos.append(dict(id=fid, file=str(file), K=K, c2w=np.load(d / 'camera_to_world.npy').astype(float), size=(int(f['width']), int(f['height']))))

        def pointmap(fid, pixels):
            d = run / 'geometry/frames' / fid; P = np.load(d / 'pts3d.npy'); valid = np.load(d / 'content_valid_mask.npy')
            uv = np.round((A[fid] @ np.c_[pixels, np.ones(len(pixels))].T)[:2].T).astype(int)
            return P[uv[valid[uv[:, 1], uv[:, 0]], 1], uv[valid[uv[:, 1], uv[:, 0]], 0]]
    else:
        doc = json.loads(Path(a.view).read_text())['publication']['snapshot']['revision']['document']
        ground = (json.loads(Path(a.layer).read_text()).get('ground') if a.layer else None) or doc['coordinateFrames'][0]['ground']
        up = np.array(ground.get('normal') or ground['plane'][:3], float)
        for c in doc['cameras']:
            if c['imageId'] in pairs:
                photos.append(dict(id=c['imageId'], file=str(Path(a.photos_dir or '.') / pairs[c['imageId']]), K=np.array(c['K'], float),
                                   c2w=np.array(c['cameraToWorld'], float), size=(int(c['width']), int(c['height']))))
        obs = {o['id']: o for o in doc['observations']}
        entity = next((e for e in doc['entities'] if a.entity and e['id'].startswith(a.entity)), None) if a.entity else None
        if a.entity and not entity:
            raise ValueError(f'no entity {a.entity} in the view')
        polys = {}
        for oid in (entity or {}).get('observationRefs') or []:
            if oid in obs and obs[oid].get('originalPixelPolygons'):
                polys.setdefault(obs[oid]['imageId'], []).extend(obs[oid]['originalPixelPolygons'])
    used = []
    for p in photos:
        p['image'] = cv2.imread(p['file'], cv2.IMREAD_COLOR)
        if p['image'] is None:
            raise FileNotFoundError(p['file'])
        H, W = p['image'].shape[:2]; sx, sy = W / p['size'][0], H / p['size'][1]
        if (sx, sy) != (1, 1):
            p['K'] = scaled_K(p['K'], sx, sy)
        p['masks'] = []
        if not a.run and p['id'] in polys:
            p['masks'].append((f'report observation polygons of entity {entity["id"]}', polygons_mask(polys[p['id']], (H, W), sx, sy)))
        if p['id'] in masks:  # a mask image, or a folder of candidates tried in name order (modal_apps/workcell_estop_mask.py)
            path = Path(masks[p['id']])
            for f in sorted(path.glob('*.png')) if path.is_dir() else [path]:
                m = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
                p['masks'].append((f'mask file {f.parent.name}/{f.name} (sha256 {_sha(f)[:12]})',
                                   (cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST) if m.shape != (H, W) else m) > 0))
        if not p['masks']:
            print(f"skip {p['id']}: no e-stop mask", file=sys.stderr); continue
        used.append(p)
    return used, up, pointmap


def main(argv=None):
    a = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    a.add_argument('--view'); a.add_argument('--layer'); a.add_argument('--run'); a.add_argument('--photos-dir'); a.add_argument('--photo', default='')
    a.add_argument('--entity'); a.add_argument('--mask', default=''); a.add_argument('--spec', default=SPEC); a.add_argument('--out')
    a.add_argument('--resolution-aware', action='store_true', help='opt-in study mode: fit() windows in head widths (not used for published scales)')
    a.add_argument('--self-test', action='store_true')
    a = a.parse_args(argv)
    if a.self_test:
        _check(); print('workcell_estop_scale self-test passed'); return
    if not (a.view or a.run) or not a.out:
        raise SystemExit('need --view or --run, and --out')
    photos, up, pointmap = load_inputs(a)
    if not photos:
        raise SystemExit('no photo has an e-stop mask')
    files = {p['id']: p['file'] for p in photos}
    specs = {k.strip(): float(v) for k, v in (x.split('=') for x in a.spec.split(','))}
    result = measure(photos, up, specs, pointmap, head_unit=a.resolution_aware)
    result['inputs'] = dict(view=a.view and {'path': a.view, 'sha256': _sha(a.view)}, layer=a.layer and {'path': a.layer, 'sha256': _sha(a.layer)},
                            run=a.run, up=up.tolist(), spec=specs, photos={k: {'file': f, 'sha256': _sha(f)} for k, f in files.items()})
    Path(a.out).write_text(json.dumps(result, indent=1, default=float) + '\n')
    print(f"nativeToMeters {result['nativeToMeters']:.6f}  maxDeviation {100 * result['maxDeviation']:.2f} %  passed {result['passed']}  ({result['depthFrom']})")
    for pid, p in result['photos'].items():
        print(f"  {pid[:12]:12s} " + '  '.join(f"{k} {v:.2f} px" for k, v in p['widthsPx'].items()) + f"  depth {p['depthNative']:.4f}  <- {p['maskSource']}"
              + (f" ({len(p['rejectedMasks'])} earlier candidate(s) rejected)" if p['rejectedMasks'] else ''))
    for pid, why in result['skipped'].items():
        print(f"  {pid[:12]:12s} skipped: {why[:200]}")


def _check():
    """Synthetic e-stop: red lip 4 cm over a narrower stem, yellow body 8 cm, grey base 8.2 cm, black bracket, on a white wall,
    seen at 2 native units by fx 3000 (1 native = 1 m); the mask is a loose box. Widths, axis and scale must come back."""
    fx, z, cx0 = 3000., 2., 400.3
    H, W = 800, 800; img = np.full((H, W, 3), (235, 235, 235), np.float32)  # RGB
    parts = [(300, 316, .040, (200, 30, 30)), (316, 330, .033, (200, 30, 30)), (330, 400, .080, (235, 200, 25)),
             (400, 420, .082, (150, 150, 150)), (420, 470, .150, (20, 20, 20))]
    xx = np.arange(W) + .0
    for r0, r1, d, col in parts:
        half = d / 2 * fx / z; cov = np.clip(half - np.abs(xx - cx0) + .5, 0, 1)  # anti-aliased columns
        img[r0:r1] = img[r0:r1] * (1 - cov[None, :, None]) + np.array(col, np.float32) * cov[None, :, None]
    img = cv2.GaussianBlur(img, (0, 0), 1.0); rng = np.random.default_rng(0)
    bgr = np.clip(img + rng.normal(0, 1.5, img.shape), 0, 255).astype(np.uint8)[..., ::-1].copy()
    K = np.array([[fx, 0, 399.5], [0, fx, 399.5], [0, 0, 1]]); M = np.eye(4); up = np.array([0, -1., 0])
    mask = np.zeros((H, W), bool); mask[290:480, 300:520] = True
    s = seed_from_mask(bgr, mask)
    assert abs(s['axisX'] - cx0) < .5 and abs(s['headTopRow'] - 300) <= 2 and abs(s['ringBottomRow'] - 419) <= 2, s
    for f in (.5, 3):  # the same e-stop at half and three times the resolution: same seed in original pixels (no fixed px windows)
        sf = seed_from_mask(cv2.resize(bgr, None, fx=f, fy=f, interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_CUBIC),
                            cv2.resize(mask.astype(np.uint8), None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST) > 0)
        back = lambda v: (v + .5) / f - .5
        assert abs(back(sf['axisX']) - cx0) < .5 and abs(back(sf['headTopRow']) - 300) <= 2 and abs(back(sf['ringBottomRow']) - 419) <= 2.5, (f, sf)
    pm =lambda pid, px: np.tile([cx0 - 399.5, 0, fx], (len(px), 1)) * z / fx
    lamp = np.zeros((H, W), bool); lamp[100:140, 100:140] = True  # a candidate on the wall only: rejected, the next one is used
    r = measure([dict(id='p', image=bgr, masks=[('wall', lamp), ('box', mask)], K=K, c2w=M)], up, {'red lip': .04, 'yellow body': .08, 'grey cylinder': .082}, pm)
    assert r['photos']['p']['maskSource'] == 'box' and len(r['photos']['p']['rejectedMasks']) == 1, r['photos']['p']
    w = r['photos']['p']['widthsPx']
    for name, d in (('red lip', .04), ('yellow body', .08), ('grey cylinder', .082)):
        assert abs(w[name] - d * fx / z) < .5, (name, w[name], d * fx / z)
    assert abs(r['nativeToMeters'] - 1) < .005 and r['passed'], r['nativeToMeters']
    # --resolution-aware: the e-stop crop upsampled 3x keeps intensity-step edges and its widths (the default windows fall back to
    # colour membership there, yellow -0.6 px); research-notes/estop-resolution-2026-10-05
    x0, y0, f = 240, 250, 3; Kc = K.copy(); Kc[:2, 2] -= (x0, y0)
    up3 = np.ascontiguousarray(cv2.resize(bgr[y0:y0 + 260, x0:x0 + 320], None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC))
    s3 = seed_from_mask(up3, cv2.resize(mask[y0:y0 + 260, x0:x0 + 320].astype(np.uint8), up3.shape[1::-1], interpolation=cv2.INTER_NEAREST) > 0)
    r3 = fit(up3, scaled_K(Kc, f, f), M, up, np.array([(cx0 - 399.5) * z / fx, 0, z]), (s3['axisX'], (s3['headTopRow'] + s3['ringBottomRow']) / 2),
             s3['headTopRow'], s3['ringBottomRow'], head_px=s3['headPx'])['parts']
    for name, d in (('red lip', .04), ('yellow body', .08), ('grey cylinder', .082)):
        assert abs(r3[name]['symmetricWidthPx'] / f - d * fx / z) < .1 and set(r3[name]['edgeSources']) == {'intensity step'}, (name, r3[name])
    P = np.array([.1, -.2, 2.]); rays = [(np.zeros(3), P), (np.array([.5, 0, 0]), P - [.5, 0, 0]), (np.array([0, .3, -.1]), P - [0, .3, -.1])]
    assert np.allclose(triangulate(rays), P)
    cut = np.ascontiguousarray(bgr[:, :420])  # the e-stop cut by the right border
    for image, m in ((bgr, lamp), (cut, mask[:, :420])):
        try:
            seed_from_mask(image, m)
        except ValueError:
            pass
        else:
            raise AssertionError('a mask with no red head, or a cut e-stop, must be rejected')


if __name__ == '__main__':
    main()
