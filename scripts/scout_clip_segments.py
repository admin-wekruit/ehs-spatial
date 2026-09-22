"""Find the stretches of a long video that suit reconstruction: one continuous shot, the camera moving steadily, sharp, people in frame.

A tour video is mostly unusable for us — cuts, talking heads, static shots. This reads the video once at a coarse
stride and scores every window: a cut anywhere disqualifies it, the camera must move (median image shift per second
inside a band: too little is a tripod, too much is a whip pan), frames must be sharp, and people should be visible.
Nothing here decides quality on its own: it shortlists windows and writes the middle frame of each so a person looks.

  python scripts/scout_clip_segments.py --video V --output NEW_DIR [--window 40 --stride 1 --top 6]
  python scripts/scout_clip_segments.py --self-check
"""
import argparse
import json
from pathlib import Path

import numpy as np

CUT, MOVE_LOW, MOVE_HIGH = .55, .6, 9.  # histogram correlation below this is a cut; image shift per second, in percent of width


def samples(video, stride, width=320):
    """(time, small gray frame, colour histogram, sharpness) every `stride` seconds."""
    import cv2
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    assert fps and np.isfinite(fps), "video has no frame rate"
    step = max(1, int(round(fps * stride)))
    out, index = [], 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if index % step == 0:
            ok, bgr = cap.retrieve()
            if ok:
                small = cv2.resize(bgr, (width, round(width * bgr.shape[0] / bgr.shape[1])), interpolation=cv2.INTER_AREA)
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                hist = cv2.calcHist([cv2.cvtColor(small, cv2.COLOR_BGR2HSV)], [0, 1], None, [32, 32], [0, 180, 0, 256])
                out.append((index / fps, gray, cv2.normalize(hist, hist).flatten(), float(cv2.Laplacian(gray, cv2.CV_64F).var())))
        index += 1
    cap.release()
    return out, fps


def measure(rows, stride):
    """Per gap: histogram correlation (cut if low) and camera shift in percent of width per second."""
    import cv2
    correlation, shift = [], []
    for (_, a, ha, _), (_, b, hb, _) in zip(rows, rows[1:]):
        correlation.append(float(cv2.compareHist(ha.astype(np.float32), hb.astype(np.float32), cv2.HISTCMP_CORREL)))
        (dx, dy), _ = cv2.phaseCorrelate(np.float32(a), np.float32(b))
        shift.append(float(np.hypot(dx, dy)) / a.shape[1] * 100 / stride)
    return np.array(correlation), np.array(shift)


def people(rows, every):
    """Rough count of standing people per sampled frame with OpenCV's own HOG detector, when this build has one.

    Returns (counts, available). A headless build without HOG leaves the counts at zero: the windows are then ranked on
    motion and sharpness only, and who is in frame is decided by looking at the saved middle frames.
    """
    import cv2
    if not hasattr(cv2, "HOGDescriptor"):
        return np.zeros(len(rows)), False
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    found = np.zeros(len(rows))
    for n in range(0, len(rows), every):
        big = cv2.resize(rows[n][1], None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        found[n] = len(hog.detectMultiScale(big, winStride=(8, 8), scale=1.06)[0])
    return found, True


def windows(rows, correlation, shift, seen, stride, length):
    """Every window of `length` seconds with no cut, scored on steady motion, sharpness and people."""
    span = max(2, int(round(length / stride)))
    sharp = np.array([r[3] for r in rows])
    out = []
    for start in range(0, len(rows) - span):
        gaps = slice(start, start + span - 1)
        if correlation[gaps].min() < CUT:
            continue
        move = float(np.median(shift[gaps]))
        if not MOVE_LOW <= move <= MOVE_HIGH:
            continue
        inside = slice(start, start + span)
        crowd = float(np.mean(seen[inside] > 0))  # zero everywhere when this OpenCV build has no people detector
        out.append({"t0": round(rows[start][0], 1), "t1": round(rows[start + span - 1][0], 1), "shift_pct_per_s": round(move, 2),
                    "shift_spread": round(float(np.percentile(shift[gaps], 90) - np.percentile(shift[gaps], 10)), 2),
                    "sharpness": round(float(np.median(sharp[inside])), 1), "people_share": round(crowd, 2),
                    "score": round(crowd * 2 + min(float(np.median(sharp[inside])) / 200, 1) - abs(move - 3) / 10, 3)})
    return sorted(out, key=lambda w: -w["score"])


def pick(found, apart):
    """The best windows that do not overlap within `apart` seconds."""
    kept = []
    for w in found:
        if all(w["t1"] < k["t0"] - apart or w["t0"] > k["t1"] + apart for k in kept):
            kept.append(w)
    return kept


def run(args):
    import cv2
    rows, fps = samples(args.video, args.stride)
    correlation, shift = measure(rows, args.stride)
    seen, detector = people(rows, max(1, int(round(args.people_every / args.stride))))
    found = pick(windows(rows, correlation, shift, seen, args.stride, args.window), args.window)[:args.top]
    args.output.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(args.video))
    name = args.video.stem[:40]
    for n, w in enumerate(found):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round((w["t0"] + w["t1"]) / 2 * fps)))
        ok, bgr = cap.read()
        if ok:
            cv2.putText(bgr, f"{w['t0']:.0f}-{w['t1']:.0f}s move {w['shift_pct_per_s']}%/s people {w['people_share']:.0%}", (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 1., (0, 0, 0), 5)
            cv2.putText(bgr, f"{w['t0']:.0f}-{w['t1']:.0f}s move {w['shift_pct_per_s']}%/s people {w['people_share']:.0%}", (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 1., (255, 255, 255), 2)
            cv2.imwrite(str(args.output / f"{name}-{n:02d}-{int(w['t0'])}s.jpg"), cv2.resize(bgr, (960, round(960 * bgr.shape[0] / bgr.shape[1]))))
    cap.release()
    report = {"video": str(args.video), "duration_s": round(rows[-1][0], 1), "fps": fps, "stride_s": args.stride, "window_s": args.window,
              "samples": len(rows), "cuts": int((correlation < CUT).sum()), "median_shift_pct_per_s": round(float(np.median(shift)), 2),
              "people_detector": detector,
              "rule": f"no cut inside the window; median shift between {MOVE_LOW} and {MOVE_HIGH} percent of width per second; score = people share x2 + sharpness - distance from 3%/s",
              "windows": found}
    (args.output / f"{name}-segments.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: report[k] for k in ("video", "duration_s", "cuts", "median_shift_pct_per_s")} | {"windows": found[:4]}, ensure_ascii=False))


def self_check():
    import cv2
    rng = np.random.default_rng(0)
    texture = rng.integers(0, 255, (400, 700), dtype=np.uint8)
    scene, elsewhere = rng.random(16).astype(np.float32), rng.random(16).astype(np.float32)  # two colour histograms with nothing in common
    walk = [(n * .5, np.roll(texture, n * 6, axis=1)[:, :320].copy(), scene, 300.) for n in range(20)]  # 6 px per .5 s = 3.75%/s at 320 px
    correlation, shift = measure(walk, .5)
    assert correlation.min() > .9 and 2 < np.median(shift) < 6, (correlation.min(), np.median(shift))
    still = [(n * .5, texture[:, :320].copy(), scene, 300.) for n in range(20)]
    assert np.median(measure(still, .5)[1]) < .1, "a tripod shows no shift"
    assert windows(walk, correlation, shift, np.ones(len(walk)), .5, 4) and not windows(still, *measure(still, .5), np.ones(len(still)), .5, 4)
    other = rng.integers(0, 255, (400, 320), dtype=np.uint8)  # a different scene: its colour histogram is unlike the first
    cut = walk[:10] + [(r[0], other.copy(), elsewhere, 300.) for r in walk[10:]]
    assert all(w["t0"] >= 5. or w["t1"] <= 4.5 for w in windows(cut, *measure(cut, .5), np.ones(len(cut)), .5, 4)), "no window spans a cut"
    print("segment scout check passed: a walk scores, a tripod does not, and no window crosses a cut")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--video", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--window", type=float, default=40., help="length of the segment to look for, in seconds")
    parser.add_argument("--stride", type=float, default=1., help="sampling step in seconds")
    parser.add_argument("--people-every", type=float, default=4., help="run the people detector every N seconds")
    parser.add_argument("--top", type=int, default=6)
    a = parser.parse_args()
    self_check() if a.self_check else run(a)
