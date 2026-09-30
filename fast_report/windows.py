"""Content windows (X6): a video cut into windows by what the camera sees, not by seconds.

A window opens at an anchor keyframe. Each later 5 fps keyframe is matched to the anchor (ORB, ratio test, MAGSAC
fundamental matrix, a homography when F is degenerate); the co-visibility is the share of the anchor's textured grid
cells (8 x 4 over the middle 60 % of the 640 x 480 raster's rows, >= 3 ORB keypoints) that keep >= 2 verified matches
(one stray match per cell is what unrelated views share: runs/fx-x6-windows-time-001, far pairs 0.25-0.36 with 1, 0.13-0.25 with 2).
The top and bottom 20 % are left out: burned-in captions, logos and clocks sit there and match across any move
(detect_shot_cuts' OVERLAY_EDGE; ME340's caption band is at rows 648-704 of 720). When it drops below THRESHOLD (once the window has MIN_KEYS
new keyframes: DA3 and the lift need views), or the window reaches MAX_KEYS new keyframes, or a shot cut falls between
two keyframes, the window closes. The next window starts with the last CARRY
keyframes of the closed one (the overlap the geometry is stitched on) and its first keyframe is the new anchor. After a
cut (or a collapse: co-visibility <= COLLAPSE, nothing shared) nothing is carried: another place, another frame.

A still camera keeps co-visibility near 1, so the content rule opens no window there; the MAX_KEYS cap is a separate,
reported reason ('max'), there only to bound DA3's views per window.

    python -m fast_report.windows --self-check
"""
import sys

import numpy as np

THRESHOLD, MIN_KEYS, MAX_KEYS, CARRY, COLLAPSE = .40, 8, 40, 2, .05  # X6 VERIFY: 0.40 is the recommendation (0.55 was the old default)
GRID, MIN_CELL_KP, MIN_HITS, ORB_N, RATIO, F_PX, EDGE = (8, 4), 3, 2, 1500, .8, 1.5, .2


def features(gray):
    """640x480 grey -> (keypoint xy (n,2) float32, ORB descriptors (n,32) uint8 or None)."""
    import cv2
    kp, des = cv2.ORB_create(ORB_N, fastThreshold=10).detectAndCompute(gray, None)
    xy = np.array([k.pt for k in kp], np.float32).reshape(-1, 2)
    keep = (xy[:, 1] >= EDGE * gray.shape[0]) & (xy[:, 1] < (1 - EDGE) * gray.shape[0])
    return xy[keep], (des[keep] if des is not None else None)


def cells(xy, shape=(480, 640)):
    gx, gy = GRID
    y = (xy[:, 1] - EDGE * shape[0]) / (1 - 2 * EDGE)
    return (np.clip(y * gy // shape[0], 0, gy - 1) * gx + np.minimum(xy[:, 0] * gx // shape[1], gx - 1)).astype(int)


def covisibility(a, b):
    """Share of anchor a's textured cells with >= 1 geometrically verified match in b (0..1)."""
    import cv2
    (xa, da), (xb, db) = a, b
    textured = np.bincount(cells(xa), minlength=GRID[0] * GRID[1]) >= MIN_CELL_KP
    if da is None or db is None or len(da) < 8 or len(db) < 8 or not textured.any():
        return 0.
    good = [m[0] for m in cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db, k=2) if len(m) == 2 and m[0].distance < RATIO * m[1].distance]
    if len(good) < 12:
        return 0.
    pa, pb = xa[[m.queryIdx for m in good]], xb[[m.trainIdx for m in good]]
    try:
        _, inl = cv2.findFundamentalMat(pa, pb, cv2.USAC_MAGSAC, F_PX, .999)
    except cv2.error:  # no motion (a still camera) or pure rotation: F is degenerate, a homography is the model
        inl = None
    if inl is None:
        _, inl = cv2.findHomography(pa, pb, cv2.RANSAC, 3.)
    if inl is None:
        return 0.
    hit = np.bincount(cells(pa[inl.ravel() > 0]), minlength=GRID[0] * GRID[1]) >= MIN_HITS
    return float((hit & textured).sum() / textured.sum())


class Builder:
    """Online window rule. push(key, feats, cut_before) -> the window it closed, or None; finish() -> the last one.
    A window: {'keys': [key ids, carried first], 'carried': n, 'reason': why it closed, 'covis': [per key vs anchor]}."""

    def __init__(self, threshold=THRESHOLD, max_keys=MAX_KEYS, carry=CARRY, covis=covisibility, min_keys=MIN_KEYS):
        self.threshold, self.max_keys, self.carry, self.covis, self.min_keys = threshold, max_keys, carry, covis, min_keys
        self.cur, self.feats, self.anchor, self.carried, self.scores = [], {}, None, 0, []

    def _open(self, keys, carried):
        self.cur, self.carried = list(keys), carried
        self.anchor = self.feats[keys[0]]
        self.scores = [1.] * len(keys)

    def push(self, key, feats, cut_before=False):
        self.feats[key] = feats
        if not self.cur:
            self._open([key], 0)
            return None
        c = self.covis(self.anchor, feats)
        new = len(self.cur) - self.carried
        reason = "cut" if cut_before else "collapse" if c <= COLLAPSE else "content" if c < self.threshold and new >= self.min_keys else \
            "max" if new >= self.max_keys else None
        if reason is None:
            self.cur.append(key)
            self.scores.append(round(c, 3))
            return None
        closed = self.close(reason)
        keep = [] if reason in ("cut", "collapse") else self.cur[-self.carry:] if self.carry else []
        self._open(keep + [key], len(keep))
        if keep:  # the new key against the new anchor (a carried keyframe), for the record
            self.scores[-1] = round(self.covis(self.anchor, feats), 3)
        for k in list(self.feats):
            if k not in self.cur:
                del self.feats[k]
        return closed

    def close(self, reason):
        return {"keys": list(self.cur), "carried": self.carried, "reason": reason, "covis": list(self.scores)}

    def finish(self):
        return self.close("end") if self.cur else None


def run(keys, feats, threshold=THRESHOLD, max_keys=MAX_KEYS, carry=CARRY, cuts=(), covis=covisibility, min_keys=MIN_KEYS):
    """Offline: the same rule over a list of keyframes (cuts: frame numbers where a new shot starts)."""
    b, out = Builder(threshold, max_keys, carry, covis, min_keys), []
    for i, (k, f) in enumerate(zip(keys, feats)):
        w = b.push(k, f, cut_before=i > 0 and any(keys[i - 1] < c <= k for c in cuts))
        if w:
            out.append(w)
    last = b.finish()
    return out + ([last] if last else [])


def self_check():
    import cv2
    rng = np.random.default_rng(0)
    tex = cv2.GaussianBlur(rng.integers(0, 256, (900, 2600), np.uint8), (0, 0), 1.2)

    def view(x, noise=0.):
        g = tex[200:680, x:x + 640].astype(np.float32) + rng.normal(0, noise, (480, 640))
        return np.clip(g, 0, 255).astype(np.uint8)
    still = [features(view(300, noise=4.)) for _ in range(30)]
    assert covisibility(still[0], still[1]) > .9
    w = run(list(range(30)), still, max_keys=100)
    assert len(w) == 1 and w[0]["reason"] == "end", "a still camera opens no window"
    pan = [features(view(40 * i)) for i in range(45)]  # 40 px a key: the anchor view is gone after 16 keys
    w = run(list(range(45)), pan, threshold=.55)
    assert len(w) >= 3 and all(x["reason"] in ("content", "end") for x in w), [x["reason"] for x in w]
    assert all(a["keys"][-2:] == b["keys"][:2] and b["carried"] == 2 for a, b in zip(w, w[1:])), "2 keyframes carried"
    lengths = [len(x["keys"]) - x["carried"] for x in w[:-1]]
    assert all(n == MIN_KEYS for n in lengths), lengths  # 55 % of the anchor left after ~7 keys of 40 px: the floor holds it to 8
    w = run(list(range(45)), pan, threshold=.55, min_keys=1)
    assert all(4 <= len(x["keys"]) - x["carried"] <= 10 for x in w[:-1]), [len(x["keys"]) for x in w]
    w = run(list(range(45)), pan, threshold=.55, cuts=[20])
    cut = next(x for x in w if x["reason"] == "cut")
    assert max(cut["keys"]) == 19 and w[w.index(cut) + 1]["carried"] == 0, "no carry across a cut"
    w = run(list(range(30)), still, max_keys=10)
    assert [x["reason"] for x in w] == ["max", "max", "end"], "the cap is its own reason"
    print("windows self-check ok: still camera = 1 window, pan windows carry 2 keys, cut = no carry, cap reason")


if __name__ == "__main__":
    assert sys.argv[1:] == ["--self-check"], __doc__
    self_check()
