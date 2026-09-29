"""X6 step 1 on the Mac (CPU): content windows of the three test clips at several co-visibility thresholds, the shot cuts
they must respect, and the still-camera test. Keyframes are the fast core's (sharpest of each 6-frame block of the
640x480 raster grey); cuts are detect_shot_cuts on every frame (the core's rule).

    python scripts/x6_windows_sweep.py OUT_DIR
"""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "scripts")]
import detect_shot_cuts as dsc  # noqa: E402
from fast_report import windows as win  # noqa: E402

PHASE2 = Path("/Users/adam/Desktop/panoptes-public/research-notes/phase2")
CLIPS = {"me340": "me340-165", "samsclub-a2": "samsclub-337", "walmart": "walmart-190"}
BLOCK = 6
THRESHOLDS = (.4, .55, .7)


def raster_gray(bgr):
    h, w = bgr.shape[:2]
    x0 = (w - h * 4 // 3) // 2
    return cv2.cvtColor(cv2.resize(bgr[:, x0:w - x0], (640, 480), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)


def keyframes(path):
    cap, grays = cv2.VideoCapture(str(path)), []
    fps = cap.get(cv2.CAP_PROP_FPS)
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        grays.append(raster_gray(bgr))
    sharp = [float(cv2.Laplacian(g, cv2.CV_32F).var()) for g in grays]
    keys = [max(range(b, min(b + BLOCK, len(grays))), key=lambda f: sharp[f]) for b in range(0, len(grays), BLOCK)]
    return grays, keys, fps


def summarise(ws, fps):
    new = [len(w["keys"]) - w["carried"] for w in ws]
    span = [(w["keys"][-1] - w["keys"][w["carried"]]) / fps for w in ws]
    return {"windows": len(ws), "reasons": {r: sum(w["reason"] == r for w in ws) for r in dict.fromkeys(w["reason"] for w in ws)},
            "new_keyframes": {"median": float(np.median(new)), "min": int(min(new)), "max": int(max(new))},
            "span_s": {"median": round(float(np.median(span)), 2), "min": round(float(min(span)), 2), "max": round(float(max(span)), 2)},
            "bounds_s": [[round(w["keys"][0] / fps, 2), round(w["keys"][-1] / fps, 2)] for w in ws],
            "keys": [w["keys"] for w in ws], "carried": [w["carried"] for w in ws], "why": [w["reason"] for w in ws],
            "covis_min": [min(w["covis"]) for w in ws]}


def cached(fn):
    memo, calls = {}, [0, 0.]

    def f(a, b):
        k = (id(a), id(b))
        if k not in memo:
            t = time.perf_counter()
            memo[k] = fn(a, b)
            calls[0] += 1
            calls[1] += time.perf_counter() - t
        return memo[k]
    return f, calls


def still_tests(gray, rng):
    """50 keyframes (10 s) of one real view: sensor noise + JPEG + 1 px jitter; then the same with a dark person-sized
    blob (25 % of the width, full height of the judged band) walking across. No MIN_KEYS floor and no cap here: the
    content rule alone."""
    def shot(occluder):
        out = []
        for i in range(50):
            g = np.roll(gray, tuple(rng.integers(-1, 2, 2)), (0, 1)).astype(np.float32) + rng.normal(0, 3, gray.shape)
            if occluder:
                x = int(-160 + i * 800 / 50)
                a, b = min(640, max(0, x)), min(640, max(0, x + 160))
                g[80:480, a:b] = 30 + rng.normal(0, 3, (400, b - a))
            g = cv2.imdecode(cv2.imencode(".jpg", np.clip(g, 0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 85])[1], cv2.IMREAD_GRAYSCALE)
            out.append(win.features(g))
        return out
    res = {}
    for name, occ in (("still_noise_jpeg_jitter", False), ("still_with_walking_occluder", True)):
        f = shot(occ)
        res[name] = {str(t): [w["reason"] for w in win.run(list(range(50)), f, threshold=t, max_keys=1000, min_keys=1)] for t in THRESHOLDS}
        res[name]["covis_min"] = round(min(win.covisibility(f[0], x) for x in f[1:]), 3)
    return res


def main(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rng, result = np.random.default_rng(0), {"rule": win.__doc__.split("\n\n")[1], "constants": {
        "max_keys": win.MAX_KEYS, "min_keys": win.MIN_KEYS, "min_hits": win.MIN_HITS, "carry": win.CARRY, "collapse": win.COLLAPSE, "grid": win.GRID, "edge": win.EDGE, "orb": win.ORB_N},
        "where": "Mac CPU (Apple, 12 cores), one thread: the rule's own cost, not the container's", "videos": {}}
    for site, clip in CLIPS.items():
        t = time.perf_counter()
        grays, keys, fps = keyframes(PHASE2 / "data/clips" / clip / "source-full.mp4")
        decode_s = time.perf_counter() - t
        t = time.perf_counter()
        cuts = dsc.detect(grays)
        cut_s = time.perf_counter() - t
        cut_frames = [int(c) for c in cuts["cuts"]]
        t = time.perf_counter()
        feats = [win.features(grays[k]) for k in keys]
        feat_ms = (time.perf_counter() - t) / len(keys) * 1e3
        row = {"frames": len(grays), "fps": fps, "keyframes": len(keys), "cuts": cut_frames, "fades": cuts.get("fades"),
               "segments": cuts.get("segments"), "orb_ms_per_keyframe": round(feat_ms, 2), "decode_s_mac": round(decode_s, 2),
               "cuts_s_mac": round(cut_s, 2), "thresholds": {}}
        for thr in THRESHOLDS:
            cov, calls = cached(win.covisibility)
            ws = win.run(keys, feats, threshold=thr, cuts=cut_frames, covis=cov)
            s = summarise(ws, fps)
            s["no_min_keys"] = {k: v for k, v in summarise(win.run(keys, feats, threshold=thr, cuts=cut_frames, covis=cov, min_keys=1), fps).items()
                                if k in ("windows", "reasons", "span_s")}
            s["match_ms_per_call"] = round(calls[1] / max(calls[0], 1) * 1e3, 2)
            s["cuts_respected"] = all(not any(w["keys"][0] < c <= w["keys"][-1] for w in ws) for c in cut_frames)
            row["thresholds"][str(thr)] = s
        # the same windows without the cut rule: does the content rule alone find the cuts?
        cov, _ = cached(win.covisibility)
        alone = win.run(keys, feats, threshold=.55, covis=cov)
        row["content_rule_alone_055"] = {"reason_at_each_cut": [next((w["reason"] for w in alone if w["keys"][-1] < c <= w["keys"][-1] + BLOCK), None)
                                                                for c in cut_frames]}
        row["still"] = still_tests(grays[keys[len(keys) // 2]], rng)
        result["videos"][site] = row
        print(site, {k: (v["windows"], v["reasons"], v["span_s"]["median"]) for k, v in row["thresholds"].items()}, row["still"], flush=True)
    (out / "windows-sweep.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
