"""Generic e-stop measurement as a projected stack of coaxial cylinders (red lip, yellow body, grey base) in one photo.

One image axis (direction = projection of the world up direction at the e-stop), straight silhouette lines symmetric about it, one
width per part, fitted to gradient summed along whole sides; each side then refined by a blurred-step fit of the profile averaged
along it, and the symmetric half-width weighted 1/blur^2. Part spans come from the colour sequence along the axis."""
import numpy as np, cv2
from scipy.optimize import curve_fit
from scipy.special import ndtr

def fit(image_bgr, K, c2w, up, P3, seed_xy, top_row, bottom_row, guess=(None, None, None)):
    R, C = c2w[:3, :3].T, c2w[:3, 3]
    proj = lambda X: (lambda x: x[:2] / x[2])(K @ (R @ (np.asarray(X, float) - C)))
    a0, a1 = proj(P3), proj(P3 + .05 * np.asarray(up, float)); u0 = (a1 - a0) / np.linalg.norm(a1 - a0)
    if u0[1] > 0: u0 = -u0
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32); hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV).astype(int)
    smooth = cv2.GaussianBlur(rgb, (0, 0), 1.0)
    def frame(theta):
        c, s = np.cos(theta), np.sin(theta); u = np.array([c * u0[0] - s * u0[1], s * u0[0] + c * u0[1]])
        n = np.array([-u[1], u[0]]); return u, (n if n[0] > 0 else -n)
    def axis(a, theta, row):
        u, n = frame(theta); base = np.array(seed_xy, float) + a * n; t = (row - base[1]) / u[1]; return base + t * u, n
    # colour sequence along the axis (seed axis) -> spans
    seq = []
    for row in range(top_row - 4, bottom_row + 4):
        p, _ = axis(0, 0, row); x, y = int(round(p[0])), row
        h, s, v = hsv[y, x]
        seq.append((row, 'red' if (h < 10 or h > 165) and s > 70 else 'yellow' if 15 < h < 40 and s > 70 else 'grey' if s < 60 and 60 < v < 210 else None))
    def span(kind):
        rows = [r for r, k in seq if k == kind]; return (min(rows), max(rows)) if rows else None
    red, yel, gry = span('red'), span('yellow'), span('grey')
    if not (red and yel and gry): raise ValueError(f'parts not found along the axis: {red}, {yel}, {gry}')
    parts = {'red lip': (red[0] + 1, red[0] + max(int(.4 * (red[1] - red[0])), 4)),
             'yellow body': (yel[0] + int(.35 * (yel[1] - yel[0])), yel[1] - 2),
             'grey cylinder': (max(gry[0], yel[1]) + 2, gry[1] - 2)}
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
        L = cv2.GaussianBlur(L.astype(np.float32), (0, 0), 1.0); like[k] = L
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
    offsets = np.arange(-8, 8.01, .25); best = None
    for theta in np.radians(np.arange(-2, 2.01, .25)):
        grids = {name: np.arange(lo, hi + .01, .25) for name, (lo, hi) in ranges.items()}
        per = {name: scores(theta, offsets, grids[name], np.arange(r0, r1 + 1, 1.0), part_kind[name]) for name, (r0, r1) in parts.items()}
        total = sum(v.max(1) for v in per.values()); i = int(np.argmax(total))
        if best is None or total[i] > best[0]:
            best = (float(total[i]), float(offsets[i]), float(theta), {name: float(grids[name][int(np.argmax(v[i]))]) for name, v in per.items()})
    _, a, theta, widths = best
    step = lambda x, lo, hi, x0, s: lo + (hi - lo) * ndtr((x - x0) / s)
    result = {}
    for name, (r0, r1) in parts.items():
        w = widths[name]; offs = np.arange(-8, 8.01, .25); sides = {}
        for sign, label in ((-1, 'left'), (1, 'right')):
            # 1) the part's colour membership locates the silhouette robustly (not the neighbouring wood, cable or box)
            prof, rgbp = [], []
            for row in np.arange(r0, r1 + 1, 1.0):
                p, n = axis(a, theta, row); e = p + sign * w / 2 * n
                prof.append(remap(like[part_kind[name]], e[0] + offs * n[0], e[1] + offs * n[1]))
                rgbp.append(np.stack([remap(smooth[..., c], e[0] + offs * n[0], e[1] + offs * n[1]) for c in range(3)], -1))
            y = np.mean(prof, 0)
            (_, _, xc, sc), _ = curve_fit(step, offs, y, p0=[y[0], y[-1], 0, 1], bounds=([-1, -1, -8, .2], [2, 2, 8, 6]), maxfev=20000)
            # 2) intensity mixes linearly at a silhouette, so its step centre is unbiased: fit it in +-4 px around the colour edge
            rgbp = np.mean(rgbp, 0); win = np.abs(offs - xc) <= 4
            k = int(np.argmax(np.abs(rgbp[win][-3:].mean(0) - rgbp[win][:3].mean(0)))); yy = rgbp[win][:, k]
            try:
                (_, _, x0, s), _ = curve_fit(step, offs[win], yy, p0=[yy[0], yy[-1], xc, 1], bounds=([-50, -50, xc - 4, .2], [350, 350, xc + 4, 6]), maxfev=20000)
                ok = s <= 3
            except RuntimeError:
                ok = False
            sides[label] = (float(x0), float(s)) if ok else (float(xc), max(float(sc), 3.0))
            sides[label + 'Source'] = 'intensity step' if ok else 'colour membership (intensity step unusable)'
        d = np.array([w / 2 - sides['left'][0], w / 2 + sides['right'][0]]); wt = 1 / np.square([sides['left'][1], sides['right'][1]])
        result[name] = {'rows': [int(r0), int(r1)], 'symmetricWidthPx': float(2 * (d * wt).sum() / wt.sum()), 'gradientWidthPx': w,
                        'sideDistancesPx': d.tolist(), 'blurPx': [sides['left'][1], sides['right'][1]], 'edgeSources': [sides['leftSource'], sides['rightSource']]}
    return {'axisOffsetPx': a, 'axisExtraDeg': float(np.degrees(theta)), 'vanishingTiltDeg': float(np.degrees(np.arctan2(u0[0], -u0[1]))),
            'spans': {'red': red, 'yellow': yel, 'grey': gry}, 'parts': result, 'axis': lambda row: axis(a, theta, row)}
