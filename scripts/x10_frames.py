"""X10 frames: the fast core's keyframe rule on each test clip, then 12 reference keyframes per video and their 5 fps
neighbours. Core rule (fast_report/core.py): 5 fps keyframe = sharpest (Laplacian variance of the 4:3 grey raster) of
each 6-frame block; object keyframe = every 3rd 5 fps keyframe (segment.OBJECT_EVERY). Reference keyframes: 12 object
keyframes spread evenly ((i + 0.5) x n / 12); neighbours: the 5 fps keyframe just before and just after each one.

  python scripts/x10_frames.py OUT_DIR     # writes OUT_DIR/frames.json + OUT_DIR/jpg/<video>-<frame>.jpg (q95, native size)
"""
import json
import sys
from pathlib import Path

import numpy as np

CLIPS = Path("/Users/adam/Desktop/panoptes-public/research-notes/phase2/data/clips")
VIDEOS = {"me340": CLIPS / "me340-165/source-full.mp4", "samsclub": CLIPS / "samsclub-337/source-full.mp4",
          "walmart": CLIPS / "walmart-190/source-full.mp4", "lightning": CLIPS / "lightning-3585/source-rgb.mp4"}
BLOCK, OBJECT_EVERY, REF = 6, 3, 12


def pick(sharp, n_ref=REF):
    """sharpness per frame -> (5 fps keys, object keys, reference keys, {ref: (before, after)})."""
    keys = [b + int(np.argmax(sharp[b:b + BLOCK])) for b in range(0, len(sharp), BLOCK)]
    obj = keys[::OBJECT_EVERY]
    ref = [obj[int((i + .5) * len(obj) / n_ref)] for i in range(n_ref)]
    nb = {r: (keys[keys.index(r) - 1] if keys.index(r) else None, keys[keys.index(r) + 1] if keys.index(r) + 1 < len(keys) else None) for r in ref}
    return keys, obj, ref, nb


def main(out):
    import cv2
    out = Path(out)
    (out / "jpg").mkdir(parents=True, exist_ok=True)
    rec = {}
    for name, path in VIDEOS.items():
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames, sharp = [], []
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            h, w = bgr.shape[:2]
            x0 = (w - h * 4 // 3) // 2  # core.raster_gray: the 4:3 centre crop at 640x480
            g = cv2.cvtColor(cv2.resize(bgr[:, x0:w - x0], (640, 480), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
            sharp.append(float(cv2.Laplacian(g, cv2.CV_32F).var()))
            frames.append(bgr)
        keys, obj, ref, nb = pick(sharp)
        want = sorted(set(ref) | {x for p in nb.values() for x in p if x is not None})
        for f in want:
            cv2.imwrite(str(out / "jpg" / f"{name}-{f:05d}.jpg"), frames[f], [cv2.IMWRITE_JPEG_QUALITY, 95])
        rec[name] = {"video": str(path), "fps": fps, "frames": len(frames), "size_wh": [frames[0].shape[1], frames[0].shape[0]],
                     "keys_5fps": keys, "object_keys": obj, "reference": ref, "neighbours": {str(r): nb[r] for r in ref},
                     "reference_t_s": [round(r / fps, 2) for r in ref]}
        print(name, len(frames), "frames,", len(keys), "5 fps keys,", len(obj), "object keys, ref", ref, flush=True)
    (out / "frames.json").write_text(json.dumps(rec, indent=1))


def self_check():
    sharp = np.zeros(60)
    sharp[[2, 7, 15, 20, 26, 33, 38, 44, 49, 57]] = 1
    keys, obj, ref, nb = pick(sharp, 3)
    assert keys == [2, 7, 15, 20, 26, 33, 38, 44, 49, 57] and obj == [2, 20, 38, 57], (keys, obj)
    assert ref == [2, 38, 57] and nb[2] == (None, 7) and nb[38] == (33, 44) and nb[57] == (49, None), (ref, nb)
    print("x10_frames self-check ok")


if __name__ == "__main__":
    self_check() if sys.argv[1:] == ["--self-check"] else main(sys.argv[1])
