"""Split a clip into its shots before anything reconstructs it: an edited video's cut makes one walk out of two places.

Every consecutive frame pair is matched (ORB, 1000 features, cross-checked Hamming) and a RANSAC homography counts the
inliers. Still matches that may be overlay are dropped first: burned-in captions, logos and clocks survive a cut (39 of
the 40 inliers across ME340's cut at 226 sat on the caption line). That is every still match in the top or bottom
OVERLAY_EDGE of the frame, where banners, tickers, clocks and subtitles sit (a top banner with a bottom subtitle is two
bands), and all of them when they form one thin band anywhere. Each test is judged against the clip's own +-WINDOW
pairs, because texture and speed set the scale; a pair that matches nothing (< GEOMETRY_INLIERS inliers) is left out of
every median:

  no coverage  a frame with < BLANK_KEYPOINTS keypoints (black, flat, defocused), or one of >= MATCHLESS_RUN frames in a
               row whose pairs match nothing (grain, static, blur with keypoints), belongs to no shot and its pairs leave
               every median, so a long gap can neither pull the median to zero nor join the shots around it;
  cut / fade   a pair below CUT_SHARE of the median and below CUT_ABSOLUTE seeds a transition that grows to the
               neighbouring pairs still under FADE_SHARE; one pair is a cut, two or more a fade (frames of no shot);
  jump         the pair's inliers land >= JUMP_PX and >= JUMP_RATIO x the local median away from where both neighbour
               pairs' homographies put them: the camera jumped inside a place that still matches;
  dissolve     frames SPAN apart fall below SPAN_SHARE of their median (and CUT_ABSOLUTE) while the chained pair
               homographies say >= SPAN_OVERLAP of the view is shared: the picture changed, the camera did not move.
               The run grows to the spans under FADE_SHARE of their median, and every frame of its spans and SPAN more
               on each side is no coverage (blended frames carry both places), whatever else found it; only a jump cut or
               a blank gap explains such a run instead.

A shot shorter than MIN_SHOT frames between two transitions is part of them (a dim frame at the edge of a black gap).
Measured on the five delivered clips (every frame): exactly ME340 {14, 226}, Sam's Club {420}, Walmart {383}, Lightning
none, and on 100 real-frame splices (hard cuts, same-place jumps, 15/30-frame dissolves, black/grey/defocused gaps) in
research-notes/phase2/runs/m0-cut-eval-320. With the grain, overlay-edge and dissolve rules (runs/m0-integrate-cuts, M):
the same five clips exactly; the same splices 95/100 found, 0 false (in-sample); no detected dissolve leaves a frame of
>= 25% of the other place in a shot (the 3 missed 30-frame dissolves still merge); grain gaps of 8-32 frames at sigma
4-12 and a top banner plus bottom subtitle over four real cuts all split.

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
BLANK_KEYPOINTS, MIN_SHOT, MATCHLESS_RUN = 10, 3, 16
OVERLAY_PX, OVERLAY_BAND, OVERLAY_EDGE = 1.5, .1, .2
GEOMETRY_INLIERS, JUMP_PX, JUMP_RATIO = 12, 2., 10.
SPAN, SPAN_SHARE, SPAN_OVERLAP = 8, .15, .75


def features(gray):
    keys, desc = cv2.ORB_create(FEATURES).detectAndCompute(gray, None)
    return np.float32([k.pt for k in keys]).reshape(-1, 2), desc


def overlay(rows, height):
    """Which still matches may be overlay: those in the top or bottom OVERLAY_EDGE of the frame, and all of them when their
    rows (10th to 90th percentile) span < OVERLAY_BAND of the height (a caption line, a logo, a clock anywhere).
    ponytail: a static camera whose scene still-matches only at those edges loses them; its pairs then match nothing and
    it is no coverage, never a false cut."""
    edge = (rows < OVERLAY_EDGE * height) | (rows > (1 - OVERLAY_EDGE) * height)
    return edge | (len(rows) >= 4 and np.ptp(np.percentile(rows, [10, 90])) < OVERLAY_BAND * height)


def match(a, b, height):
    """(inlier points in a, in b, homography or None) between two frames' ORB features. Still matches (<= OVERLAY_PX) in
    overlay bands are not scene and are dropped before RANSAC."""
    (pa, da), (pb, db) = a, b
    empty = np.zeros((0, 2), np.float32)
    if da is None or db is None or min(len(da), len(db)) < 4:
        return empty, empty, None
    m = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(da, db)
    qa, qb = pa[[x.queryIdx for x in m]], pb[[x.trainIdx for x in m]]
    still = np.flatnonzero(np.linalg.norm(qb - qa, axis=1) <= OVERLAY_PX)
    drop = still[overlay(qa[still, 1], height)] if len(still) else still
    qa, qb = np.delete(qa, drop, 0), np.delete(qb, drop, 0)
    if len(qa) < 4:
        return empty, empty, None
    H, keep = cv2.findHomography(qa, qb, cv2.RANSAC, RANSAC_PX)
    if keep is None:
        return empty, empty, None
    keep = keep.ravel() > 0
    return qa[keep], qb[keep], H


def measure(frames):
    """Per frame keypoints; per pair k (frame k, k+1) inliers, homography (None under GEOMETRY_INLIERS) and jump (px);
    per span (k, k+SPAN) inliers. Streams: only the last SPAN frames' features and two pairs' points are held."""
    kept, keypoints, inliers, homographies, jump, spans, last = [], [], [], [], [], [], []
    for gray in frames:
        now = features(gray)
        if kept:
            a, b, H = match(kept[-1], now, gray.shape[0])
            inliers.append(len(a))
            homographies.append(H if len(a) >= GEOMETRY_INLIERS else None)
            jump.append(np.nan)
            last = (last + [(a, b)])[-2:]
            if len(last) == 2:                   # pair k-1 is now between two known pairs
                k = len(inliers) - 2
                (pa, pb), near = last[0], [homographies[i] for i in (k - 1, k + 1) if i >= 0 and homographies[i] is not None]
                if homographies[k] is not None and near:
                    jump[k] = min(np.median(np.linalg.norm(cv2.perspectiveTransform(pa[None], h)[0] - pb, axis=1)) for h in near)
        if len(kept) == SPAN:
            spans.append(len(match(kept[0], now, gray.shape[0])[0]))
        kept = (kept + [now])[-SPAN:]
        keypoints.append(len(now[0]))
        shape = gray.shape
    return {"keypoints": np.array(keypoints), "inliers": np.array(inliers, float), "homographies": homographies,
            "jump": np.array(jump, float), "spans": np.array(spans, float), "shape": shape}


def local_median(values):
    """Median over +-WINDOW, ignoring NaN (pairs of blank frames); NaN where the whole window is NaN."""
    out = np.full(len(values), np.nan)
    for k in range(len(values)):
        w = values[max(0, k - WINDOW):k + WINDOW + 1]
        if np.isfinite(w).any():
            out[k] = np.nanmedian(w)
    return out


def signals(m):
    """What the rules read: blank frames (too few keypoints, or inside a run of >= MATCHLESS_RUN frames whose pairs both
    match nothing); per pair inliers and jump, NaN where a pair touches a blank frame; per span inliers and the share of
    the view the chained pair homographies keep over it."""
    matchless = m["inliers"] < GEOMETRY_INLIERS
    inner = np.r_[False, matchless[:-1] & matchless[1:], False]  # both of the frame's pairs match nothing
    blank = m["keypoints"] < BLANK_KEYPOINTS
    for a, b in runs(inner | blank):
        if b - a + 1 >= MATCHLESS_RUN:
            blank[a:b + 1] = True
    touch = blank[:-1] | blank[1:]
    inliers, jump = np.where(touch, np.nan, m["inliers"]), np.where(touch, np.nan, m["jump"])
    h, w = m["shape"]
    gx, gy = np.meshgrid((np.arange(16) + .5) * w / 16, (np.arange(12) + .5) * h / 12)
    grid = np.float32(np.c_[gx.ravel(), gy.ravel()])[None]
    spans, overlap = m["spans"].copy(), np.full(len(m["spans"]), np.nan)
    for k in range(len(spans)):
        if blank[k] or blank[k + SPAN]:
            spans[k] = np.nan
        chain = np.eye(3)
        for H in m["homographies"][k:k + SPAN]:
            chain = None if H is None or chain is None else H @ chain
        if chain is not None:
            p = cv2.perspectiveTransform(grid, chain)[0]
            overlap[k] = np.mean((p[:, 0] >= 0) & (p[:, 0] < w) & (p[:, 1] >= 0) & (p[:, 1] < h))
    return {"blank": blank, "inliers": inliers, "jump": jump, "spans": spans, "overlap": overlap}


def scores(s):
    """Per pair (cut, jump) and per span (dissolve): the test's value over its firing level, >= 1 fires (the CUT_ABSOLUTE
    gate aside). The largest score outside every transition is how close a clip came to a false one."""
    with np.errstate(invalid="ignore", divide="ignore"):
        cut = CUT_SHARE * local_median(np.where(s["inliers"] < GEOMETRY_INLIERS, np.nan, s["inliers"])) / s["inliers"]
        jump = np.minimum(s["jump"] / JUMP_PX, s["jump"] / (JUMP_RATIO * local_median(s["jump"])))
        span = np.where(s["overlap"] >= SPAN_OVERLAP, SPAN_SHARE * local_median(s["spans"]) / s["spans"], 0)
    return cut, jump, span


def runs(mask):
    """[first, last] of every run of True."""
    edges = np.flatnonzero(np.diff(np.r_[0, np.asarray(mask, int), 0]))
    return [[int(a), int(b) - 1] for a, b in zip(edges[::2], edges[1::2])]


def transitions(s):
    """{'cuts', 'fades', 'noCoverage', 'why'} from signals. A cut is the first frame of the new shot; a fade and a
    no-coverage run are [first, last] frames that belong to no shot; 'why' names the test behind each."""
    cut, jump, span = scores(s)
    counts, n = s["inliers"], len(s["inliers"])
    low, soft = (cut >= 1) & (counts < CUT_ABSOLUTE), cut >= CUT_SHARE / FADE_SHARE
    blank = runs(s["blank"])
    cuts, fades, why, k = [], [], {f"{a}-{b}": "no coverage" for a, b in blank}, 0
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
            why[str(a + 1)] = "inliers"
        else:
            fades.append([a + 1, b])            # frames a+1..b sit between two low pairs
            why[f"{a + 1}-{b}"] = "inliers"
        k = b + 1
    for k in map(int, np.flatnonzero(jump >= 1)):
        if k + 1 in cuts:
            why[str(k + 1)] += "+jump"
        elif not any(a - 1 <= k <= b for a, b in fades + blank):
            cuts.append(k + 1)
            why[str(k + 1)] = "jump"
    broken, soft = (span >= 1) & (s["spans"] < CUT_ABSOLUTE), span >= SPAN_SHARE / FADE_SHARE
    for k0, k1 in runs(soft):
        # a run of spans under FADE_SHARE of their median holding one that fired reaches frames k0..k1+SPAN; a jump cut or a
        # blank gap in there explains it, anything else is a blend
        jumped = any(k0 < c <= k1 + SPAN and why[str(c)] == "jump" for c in cuts)
        if not broken[k0:k1 + 1].any() or jumped or any(a <= k1 + SPAN and b > k0 for a, b in blank):
            continue
        # the blend starts and ends further out, where frames SPAN apart still match (the 100 splices of
        # runs/m0-integrate-cuts: without the margin 4 of 17 found 30-frame dissolves left 1-5 frames of >= 25% of the
        # other place in a shot, with it none), so SPAN more on each side: no coverage, in neither shot
        a, b = max(0, k0 - SPAN), min(len(s["blank"]) - 1, k1 + 2 * SPAN)
        blank.append([a, b])
        why[f"{a}-{b}"] = "dissolve"
    return {"cuts": sorted(cuts), "fades": sorted(fades), "noCoverage": sorted(blank), "why": why}


def segments(n_frames, cuts, gaps):
    """Inclusive [first, last] shots of >= MIN_SHOT frames: every frame outside the gaps (fades, no coverage), split at cuts."""
    shot = np.ones(n_frames, bool)
    for a, b in gaps:
        shot[a:b + 1] = False
    edges = sorted({0, n_frames, *cuts, *[a for a, _ in gaps], *[b + 1 for _, b in gaps]})
    return [[s, e - 1] for s, e in zip(edges, edges[1:]) if shot[s] and e - s >= MIN_SHOT]


def closest(s, t):
    """The highest score each test reached outside every transition (1 would have fired)."""
    inside = np.zeros(len(s["inliers"]), bool)
    for c in t["cuts"]:
        inside[c - 1] = True
    for a, b in t["fades"] + t["noCoverage"]:
        inside[max(a - 1, 0):b + 1] = True
    out = {}
    for name, score, hit in zip(("cut", "jump", "dissolve"), scores(s),
                                (inside, inside, np.array([inside[k:k + SPAN].any() for k in range(len(s["spans"]))], bool))):
        score = np.where(hit | ~np.isfinite(score), -np.inf, score)
        if np.isfinite(score).any():
            k = int(np.argmax(score))
            out[name] = {"pair" if name != "dissolve" else "spanFrom": k, "score": round(float(score[k]), 3)}
    return out


def detect(frames):
    s = signals(measure(frames))
    t = transitions(s)
    n = len(s["blank"])
    return {"frames": n, **t, "segments": segments(n, t["cuts"], t["fades"] + t["noCoverage"]), "closestNonTransition": closest(s, t),
            "inliers": [None if np.isnan(x) else int(x) for x in s["inliers"]]}


def clip_frames(clip):
    rows = [line.split()[1] for line in (clip / "rgb.txt").read_text().splitlines() if line.strip() and not line.startswith("#")]
    return (cv2.imread(str(clip / r), cv2.IMREAD_GRAYSCALE) for r in rows)


def self_check():
    """Synthetic walks over random textures spliced by a hard cut, a 15-frame dissolve, a fade through black, a 20-frame
    black gap, a same-place jump and a 30-frame dissolve (no blended frame in a shot); a 20-frame gap of grain whose
    keypoints match nothing; a cut under a caption, and under a caption plus a top banner; then the rules alone on known
    signals, including Walmart's bare-floor stretch that is no cut."""
    rng = np.random.default_rng(0)

    def walk(seed, count, dx, dy, zoom):
        """A camera sliding and zooming over its own random texture; each shot moves its own way, as real shots do."""
        big = cv2.GaussianBlur(np.random.default_rng(seed).integers(0, 256, (1400, 1800), np.uint8), (0, 0), 2.5)
        big = cv2.normalize(big, None, 0, 255, cv2.NORM_MINMAX)
        return [cv2.warpPerspective(big, np.array([[1 + i * zoom, .01, -dx * i - 200], [-.01, 1 + i * zoom, -dy * i - 200], [0, 0, 1]]),
                                    (640, 480)) for i in range(count)]

    a, b, c = walk(1, 150, 4, 1.6, .002), walk(2, 75, -3, 2, -.001), walk(3, 48, 2, -3, .003)
    dissolve = [cv2.addWeighted(b[60 + k], 1 - (k + 1) / 16, c[k], (k + 1) / 16, 0) for k in range(15)]
    black = [cv2.convertScaleAbs(c[40 + k], alpha=1 - k / 7) for k in range(8)] + [cv2.convertScaleAbs(a[k], alpha=k / 7) for k in range(8)]
    frames = a[:40] + b[:60] + dissolve + c[15:40] + black + a[8:40] + [np.full_like(a[0], 8)] * 20 + b[:30] + a[70:100] + a[130:150]
    # cut opens 40; dissolve 100-114; fade through black 140-155; black gap 188-207; b opens 208, a 238; the jump opens 268
    got = detect([np.clip(f + rng.normal(0, 2, f.shape), 0, 255).astype(np.uint8) for f in frames])
    assert got["cuts"] == [40, 238, 268], got
    assert got["why"]["268"] == "jump", ("a[99] -> a[130] still matches; only its geometry jumps", got["why"])
    in_shot = lambda segs: {f for a, b in segs for f in range(a, b + 1)}  # noqa: E731
    blended = {100 + k for k in range(15) if .25 <= (k + 1) / 16 <= .75}  # frames carrying >= 25% of the other place
    assert not blended & in_shot(got["segments"]), ("a blended frame of the dissolve sits in a shot", sorted(blended & in_shot(got["segments"])))
    through_black = [[k0, k1] for k0, k1 in got["fades"] if k0 >= 130]
    assert through_black and all(140 <= k0 and k1 <= 155 for k0, k1 in through_black), ("the fade through black", got["fades"])
    assert [188, 207] in got["noCoverage"], ("a 20-frame black gap is no coverage, not the middle of one shot", got["noCoverage"])
    assert got["segments"][0] == [0, 39] and [188 - 1, 188 - 1] not in got["segments"] and got["segments"][-3:] == [[208, 237], [238, 267], [268, len(frames) - 1]], got["segments"]

    # a 30-frame dissolve keeps every pair matching; only frames SPAN apart stop matching while the camera holds its course
    slow = walk(4, 90, 1, .5, .0005)
    mixed = [cv2.addWeighted(slow[45 + k], 1 - (k + 1) / 31, b[k], (k + 1) / 31, 0) for k in range(30)]
    long = detect(slow[:45] + mixed + b[30:60])
    blended = {45 + k for k in range(30) if .25 <= (k + 1) / 31 <= .75}
    assert any(w == "dissolve" for w in long["why"].values()) and len(long["segments"]) == 2, long
    assert not blended & in_shot(long["segments"]), ("blended frames sit in a shot", sorted(blended & in_shot(long["segments"])), long["segments"])

    # 20 frames of sensor grain between two places: ~180 keypoints a frame, none matching; the shots stay apart
    grain = [np.clip(12 + np.random.default_rng(k).normal(0, 8, a[0].shape), 0, 255).astype(np.uint8) for k in range(20)]
    noisy = detect(a[:40] + grain + b[:40])
    assert noisy["segments"] == [[0, 39], [60, 99]] and [40, 59] in noisy["noCoverage"], (noisy["segments"], noisy["noCoverage"])

    # a cut under a boxed caption that stays on screen: its still matches alone hold the pair at ~300 inliers
    def caption(img):
        img = cv2.rectangle(img.copy(), (0, 424), (639, 470), 0, -1)
        for y, text in ((442, "SO THESE ARE THE ITEMS THAT WE USE FOR ALL THAT"), (464, "and over here we have the lathe, it can't vibrate")):
            cv2.putText(img, text, (6, y), cv2.FONT_HERSHEY_SIMPLEX, .6, 255, 2)
        return img
    captioned = detect([caption(f) for f in a[:30] + c[:30]])
    assert captioned["cuts"] == [30], ("the caption is overlay, not scene", captioned["cuts"], captioned["inliers"][27:32])

    def banner(img):  # a clock box and a two-line banner at the top: with the caption, still rows span the whole frame
        img = cv2.rectangle(cv2.rectangle(img.copy(), (380, 8), (632, 36), 0, -1), (0, 40), (639, 86), 0, -1)
        for x, y, text in ((386, 29, "2026-09-27 13:24:07 CAM 3"), (6, 58, "WAREHOUSE SAFETY WALK - LOADING DOCK B - SHIFT 2"),
                           (6, 80, "forklift lanes, racking and the pedestrian crossing")):
            cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .6, 255, 2)
        return img
    overlaid = detect([banner(caption(f)) for f in a[:30] + c[:30]])
    assert overlaid["cuts"] == [30], ("a top and a bottom band are both overlay", overlaid["cuts"], overlaid["inliers"][27:32])

    # the rules alone: lone low pairs are cuts (also at the first pair); a low middle with soft shoulders is one fade
    def known(counts, spans=None, overlap=1.):
        n = len(counts)
        spans = np.full(n - SPAN + 1, 600.) if spans is None else np.asarray(spans, float)
        return transitions({"blank": np.zeros(n + 1, bool), "inliers": np.asarray(counts, float), "jump": np.ones(n),
                            "spans": spans, "overlap": np.full(len(spans), overlap)})
    flat = np.full(60, 600.)
    flat[[0, 20]] = 5
    assert known(flat)["cuts"] == [1, 21], known(flat)
    flat[20:24] = [250, 20, 10, 280]
    assert (known(flat)["cuts"], known(flat)["fades"]) == ([1], [[21, 23]]), known(flat)
    assert segments(61, [1], [[21, 23]]) == [[1, 20], [24, 60]], "frame 0 alone is too short to be a shot"
    spans = np.full(53, 500.)
    spans[30:32] = 3  # spans 30 and 31 reach frames 30..39, and SPAN more on each side: 22..47 are no coverage
    assert known(np.full(60, 600.), spans)["noCoverage"] == [[22, 47]], known(np.full(60, 600.), spans)
    assert known(np.full(60, 600.), spans, overlap=.5)["noCoverage"] == [], "the same span on a fast pan is no dissolve"
    one_low = np.full(60, 600.)
    one_low[33] = 5  # one pair of the blend fell like a cut: the dissolve's frames still stay out of both shots
    assert known(one_low, spans)["cuts"] == [34] and known(one_low, spans)["noCoverage"] == [[22, 47]], known(one_low, spans)
    # 16 pairs in a row that match nothing (15 frames, short of MATCHLESS_RUN) hold half the window around their middle:
    # unless they leave the median it falls to theirs, and the middle frames would make a shot of their own
    matchless = np.full(60, 600.)
    matchless[20:36] = 5
    assert known(matchless)["fades"] == [[21, 35]], known(matchless)
    # walmart-190 pairs 590-639 (measured): the camera looks down at a bare floor, the count falls 10x over 12 frames
    floor = [230, 190, 139, 158, 152, 183, 156, 186, 160, 142, 187, 138, 127, 120, 74, 76, 86, 77, 76, 57, 51, 39, 38, 52, 23,
             25, 29, 30, 41, 42, 59, 58, 65, 91, 109, 131, 131, 171, 185, 161, 194, 237, 195, 230, 309, 353, 336, 383, 410, 375]
    assert known(floor)["cuts"] == [] and known(floor)["fades"] == [], "a slow fall on a bland view is not a cut"
    print("detect_shot_cuts self-check: passed", {k: got[k] for k in ("cuts", "fades", "noCoverage", "segments", "why")})


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clip", type=Path, help="clip folder with rgb.txt (TUM raster)")
    p.add_argument("--output", type=Path, help="segments.json to write")
    p.add_argument("--self-check", action="store_true")
    args = p.parse_args()
    if args.self_check:
        return self_check()
    result = {"clip": str(args.clip), "rule": {
        "match": f"ORB {FEATURES}, cross-checked Hamming, RANSAC homography {RANSAC_PX} px; still matches (<= {OVERLAY_PX} px) dropped as overlay in the top and bottom {OVERLAY_EDGE} of the frame, and all of them when their rows span < {OVERLAY_BAND} of the height",
        "matchless": f"pairs under {GEOMETRY_INLIERS} inliers leave every median; {MATCHLESS_RUN}+ frames in a row between such pairs are no coverage",
        "noCoverage": f"frames with < {BLANK_KEYPOINTS} keypoints; their pairs leave every median",
        "cut": f"inliers < {CUT_SHARE} x median of +-{WINDOW} pairs and < {CUT_ABSOLUTE}",
        "grow": f"neighbouring pairs < {FADE_SHARE} x their median join the transition; one pair is a cut, more is a fade",
        "jump": f"inliers >= {JUMP_PX} px and >= {JUMP_RATIO} x the local median from both neighbour homographies ({GEOMETRY_INLIERS}+ inliers each)",
        "dissolve": f"frames {SPAN} apart < {SPAN_SHARE} x their median and < {CUT_ABSOLUTE} while chained homographies keep >= {SPAN_OVERLAP} of the view; the run grown to spans < {FADE_SHARE} x median is no coverage",
        "shot": f">= {MIN_SHOT} frames"},
        **detect(clip_frames(args.clip))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: result[k] for k in ("clip", "frames", "cuts", "fades", "noCoverage", "segments", "why", "closestNonTransition")}))


if __name__ == "__main__":
    main()
