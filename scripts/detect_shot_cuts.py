"""Split a clip into its shots before anything reconstructs it: an edited video's cut makes one walk out of two places.

Every consecutive frame pair is matched (ORB, 1000 features, cross-checked Hamming) and a RANSAC homography counts the
inliers. Inside one shot the count moves with texture and speed, so it is judged against its own neighbourhood: a pair
is low when it falls below CUT_SHARE of the median over +-WINDOW pairs and below CUT_ABSOLUTE. A low pair seeds a
transition that grows to the neighbouring pairs still under FADE_SHARE of their median (a dissolve fades in and out
around its low middle; a hard cut's neighbours are normal). One pair is a cut; two or more are a fade, and the frames
inside a fade belong to no shot. Segments are the shots between them.

Measured on the five delivered clips (every frame): exactly ME340 {14, 226}, Sam's Club {420}, Walmart {383}, Lightning
none; the nearest miss inside a shot is Walmart's bare floor at 0.30 of its median. On real-frame splices: 20/20 hard
cuts (one as a 1-frame fade: a blurred frame next to the cut). A dissolve is found only when its inliers dip:
4 of 20 real-frame 15-frame dissolves (the synthetic one in --self-check is found).

  python scripts/detect_shot_cuts.py --clip CLIP_DIR --output segments.json
  python scripts/detect_shot_cuts.py --self-check
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

CUT_SHARE, CUT_ABSOLUTE, FADE_SHARE, WINDOW = .1, 100, .5, 15
RANSAC_PX, FEATURES = 3., 1000


def features(gray):
    keys, desc = cv2.ORB_create(FEATURES).detectAndCompute(gray, None)
    return np.float32([k.pt for k in keys]).reshape(-1, 2), desc


def inliers(a, b):
    """Homography RANSAC inliers between two frames' ORB features (0 when either has too few)."""
    (pa, da), (pb, db) = a, b
    if da is None or db is None or min(len(da), len(db)) < 4:
        return 0
    m = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(da, db)
    if len(m) < 4:
        return 0
    _, keep = cv2.findHomography(pa[[x.queryIdx for x in m]], pb[[x.trainIdx for x in m]], cv2.RANSAC, RANSAC_PX)
    return 0 if keep is None else int(keep.sum())


def pair_inliers(frames):
    """inliers[k] is the pair (frame k, frame k+1), for an iterable of grayscale frames."""
    out, last = [], None
    for gray in frames:
        now = features(gray)
        if last is not None:
            out.append(inliers(last, now))
        last = now
    return np.array(out, float)


# ponytail: pair inliers only, so a dissolve that keeps matching passes as one shot; a learned shot-boundary model is the upgrade
def transitions(counts):
    """(cuts, fades) from pair inlier counts. A cut is the first frame of the new shot; a fade is [first, last] frame
    that belongs to neither shot. Also returns the per-pair local median."""
    n = len(counts)
    median = np.array([np.median(counts[max(0, k - WINDOW):k + WINDOW + 1]) for k in range(n)])
    low = (counts < CUT_SHARE * median) & (counts < CUT_ABSOLUTE)
    soft = counts < FADE_SHARE * median
    cuts, fades, k = [], [], 0
    while k < n:
        if not low[k]:
            k += 1
            continue
        a, b = k, k
        while a > 0 and soft[a - 1]:
            a -= 1
        while b + 1 < n and soft[b + 1]:
            b += 1
        if a == b:
            cuts.append(a + 1)                  # pair (a, a+1): frame a+1 opens the new shot
        else:
            fades.append([a + 1, b])            # frames a+1..b sit between two low pairs
        k = b + 1
    return cuts, fades, median


def segments(n_frames, cuts, fades):
    """Inclusive [first, last] shots: every frame outside a fade, split at cuts and fades."""
    starts = sorted([0] + cuts + [b + 1 for _, b in fades])
    stops = sorted([c - 1 for c in cuts] + [a - 1 for a, _ in fades] + [n_frames - 1])
    return [[s, e] for s, e in zip(starts, stops) if e >= s]


def detect(frames):
    counts = pair_inliers(frames)
    cuts, fades, median = transitions(counts)
    ratio = counts / np.maximum(median, 1)
    flagged = np.zeros(len(counts), bool)
    for c in cuts:
        flagged[c - 1] = True
    for a, b in fades:
        flagged[a - 1:b + 1] = True
    closest = int(np.argmin(np.where(flagged, np.inf, ratio))) if (~flagged).any() else None
    return {"frames": len(counts) + 1, "cuts": cuts, "fades": fades, "segments": segments(len(counts) + 1, cuts, fades),
            "closestNonTransition": None if closest is None else {"frame": closest + 1, "inliers": int(counts[closest]),
                                                                  "localMedian": float(median[closest]), "share": round(float(ratio[closest]), 3)},
            "transitionShares": {str(c): round(float(ratio[c - 1]), 3) for c in cuts} |
                                {f"{a}-{b}": round(float(ratio[a - 1:b + 1].min()), 3) for a, b in fades},
            "inliers": counts.astype(int).tolist()}


def clip_frames(clip):
    rows = [line.split()[1] for line in (clip / "rgb.txt").read_text().splitlines() if line.strip() and not line.startswith("#")]
    return (cv2.imread(str(clip / r), cv2.IMREAD_GRAYSCALE) for r in rows)


def self_check():
    """Three synthetic walks over different random textures (TUM raster) spliced by a hard cut, a 15-frame dissolve and
    a fade through black; then the rule alone on known counts, including Walmart's bare-floor stretch that is no cut."""
    rng = np.random.default_rng(0)

    def walk(seed, count, dx, dy, zoom):
        """A camera sliding and zooming over its own random texture; each shot moves its own way, as real shots do."""
        big = cv2.GaussianBlur(np.random.default_rng(seed).integers(0, 256, (1400, 1800), np.uint8), (0, 0), 2.5)
        big = cv2.normalize(big, None, 0, 255, cv2.NORM_MINMAX)
        return [cv2.warpPerspective(big, np.array([[1 + i * zoom, .01, -dx * i - 200], [-.01, 1 + i * zoom, -dy * i - 200], [0, 0, 1]]),
                                    (640, 480)) for i in range(count)]

    a, b, c = walk(1, 40, 4, 1.6, .002), walk(2, 75, -3, 2, -.001), walk(3, 48, 2, -3, .003)
    dissolve = [cv2.addWeighted(b[60 + k], 1 - (k + 1) / 16, c[k], (k + 1) / 16, 0) for k in range(15)]
    black = [cv2.convertScaleAbs(c[40 + k], alpha=1 - k / 7) for k in range(8)] + [cv2.convertScaleAbs(a[k], alpha=k / 7) for k in range(8)]
    frames = a[:40] + b[:60] + dissolve + c[15:40] + black + a[8:40]  # cut opens 40; dissolve 100-114; black 140-155
    got = detect([np.clip(f + rng.normal(0, 2, f.shape), 0, 255).astype(np.uint8) for f in frames])
    assert got["cuts"] == [40], got
    assert len(got["fades"]) == 2, got["fades"]
    (d0, d1), (k0, k1) = got["fades"]
    assert 100 <= d0 and d1 <= 114 and d1 - d0 >= 4, ("the dissolve is a fade inside its blended frames", got["fades"])
    # ponytail: the dissolve's faint ends (alpha <= 3/16 and >= 13/16) stay with their shots; FADE_SHARE is the knob
    assert 140 <= k0 and k1 <= 155, ("the fade through black", got["fades"])
    assert got["segments"] == [[0, 39], [40, d0 - 1], [d1 + 1, k0 - 1], [k1 + 1, len(frames) - 1]], got["segments"]
    # the rule alone: lone low pairs are cuts (also at the first pair); a low middle with soft shoulders is one fade
    flat = np.full(60, 600.)
    flat[[0, 20]] = 5
    assert transitions(flat)[:2] == ([1, 21], []), transitions(flat)[:2]
    flat[20:24] = [250, 20, 10, 280]
    assert transitions(flat)[:2] == ([1], [[21, 23]]), transitions(flat)[:2]
    assert segments(61, [1], [[21, 23]]) == [[0, 0], [1, 20], [24, 60]]
    # walmart-190 pairs 590-639 (measured): the camera looks down at a bare floor, the count falls 10x over 12 frames
    floor = [230, 190, 139, 158, 152, 183, 156, 186, 160, 142, 187, 138, 127, 120, 74, 76, 86, 77, 76, 57, 51, 39, 38, 52, 23,
             25, 29, 30, 41, 42, 59, 58, 65, 91, 109, 131, 131, 171, 185, 161, 194, 237, 195, 230, 309, 353, 336, 383, 410, 375]
    assert transitions(np.array(floor, float))[:2] == ([], []), "a slow fall on a bland view is not a cut"
    print("detect_shot_cuts self-check: passed", {k: got[k] for k in ("cuts", "fades", "segments", "transitionShares")})


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clip", type=Path, help="clip folder with rgb.txt (TUM raster)")
    p.add_argument("--output", type=Path, help="segments.json to write")
    p.add_argument("--self-check", action="store_true")
    args = p.parse_args()
    if args.self_check:
        return self_check()
    result = {"clip": str(args.clip), "rule": {"match": f"ORB {FEATURES}, cross-checked Hamming, RANSAC homography {RANSAC_PX} px",
                                                "cut": f"inliers < {CUT_SHARE} x median of +-{WINDOW} pairs and < {CUT_ABSOLUTE}",
                                                "grow": f"neighbouring pairs < {FADE_SHARE} x their median join the transition; one pair is a cut, more is a fade"},
              **detect(clip_frames(args.clip))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: result[k] for k in ("clip", "frames", "cuts", "fades", "segments", "closestNonTransition")}))


if __name__ == "__main__":
    main()
