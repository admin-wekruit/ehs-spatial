"""r5b: planted changes whose before and after are both seen (X6 VERIFY: X6's plants changed in the clip's last 0.8 s or
were seen on too few keyframes; 0 of 10 found in the timed config). scripts/x6_plant.py's boxes (0.8 m cardboard cubes on
the delivered floor, drawn with DROID's cameras where DROID's depth saw free space behind every corner), placed so that:
  - the box is seen 1.2-7 m away (X6: lifted at that range) on >= MIN_DET 5 fps keyframes on its side of the change, and its
    empty place 1.2-5 m away (the place test judges <= 5 m, timeline.MAX_RANGE) on >= MIN_JUDGE keyframes on the other side
    ('disappeared': box before, empty after; 'appeared': the reverse; 'moved': both, one place each way);
  - the change is >= AFTER_S before the end of the reference shot (>= 2 content windows of video after it);
  - nothing seen in front of the box (DROID depth nearer than 0.9 x the box's nearest corner inside its outline) and its
    volume seen through (depth beyond 1.05 x its top face and centre) on every DROID keyframe that sees it: X6's test asked
    the floor under the bottom corners to be beyond them, which a textured floor never is (ME340 could not be planted).
A plant, not a finding: every number measured on these clips is labelled 'planted'.

    python scripts/r5b_plant.py SITE OUT.mp4 TRUTH.json
    python scripts/r5b_plant.py --self-check
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "scripts"), str(REPO / "modal_apps"), str(REPO)]
import x6_plant as xp  # noqa: E402

NEAR, FAR_DET, FAR_JUDGE, MIN_DET, MIN_JUDGE, AFTER_S, KEY_EVERY = 1.2, 7., 5., 4, 3, 3.5, 6
MOVE_SHORT = .5  # m: a second 'moved' tried when X6's 1.2 m move finds no place (its two places overlap: one card that moves)


def visible(corners, poses, frames, Kr, mpn, far=FAR_JUDGE):
    """Frames where the cube is NEAR-far m away with >= 6 corners in the 640x480 raster and its centre above the bottom 15 %."""
    out = []
    for f in frames:
        xy, z = xp.project(corners, poses[f], Kr)
        inside = (xy[:, 0] > 0) & (xy[:, 0] < 640) & (xy[:, 1] > 0) & (xy[:, 1] < 480)
        if (z * mpn > NEAR).all() and (z * mpn < far).all() and inside.sum() >= 6 and xy[:, 1].mean() < .85 * 480:
            out.append(f)
    return out


def keys_of(fs):
    return len({f // KEY_EVERY for f in fs})  # 5 fps keyframes: one per 6-frame block


def change_frame(need, last_ok):
    """The change: a frame <= last_ok with, for every (frames, side, n) in need, >= n keyframes of those frames on that side
    ('before': < frame, 'after': >= frame), closest to the middle of the first list; None when there is none."""
    cands = sorted({f for fs, _, _ in need for f in fs if f <= last_ok})
    ok = [f for f in cands if all(keys_of([g for g in fs if (g < f if side == "before" else g >= f)]) >= n for fs, side, n in need)]
    if not ok:
        return None
    mid = need[0][0][len(need[0][0]) // 2]
    return min(ok, key=lambda f: abs(f - mid))


def seen_free(corners, d, frames, mpn):
    """Nothing in front of the box and its volume seen through, on every DROID keyframe that sees >= 5 of its corners (>= 1 such)."""
    keys, depth, K = d["keyframe_source_indices"], d["keyframe_final_fullres_depth"], d["keyframe_final_fullres_intrinsics"]
    top, centre = corners[4:].mean(0), corners.mean(0)
    seen = 0
    for j, f in enumerate(keys):
        if f not in frames:
            continue
        c2w = d["poses_c2w"][f].astype(np.float64)
        k = K[j] if K.ndim == 2 else K
        kk = np.array([k[0], k[1], k[2], k[3]]) if len(k) == 4 else np.array([k[0, 0], k[1, 1], k[0, 2], k[1, 2]])
        xy, z = xp.project(corners, c2w, kk)
        h, w = depth[j].shape
        ok = (z > .1) & (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
        if ok.sum() < 5:
            continue
        seen += 1
        valid = d["keyframe_final_fullres_valid"][j]
        m = np.zeros((h, w), np.uint8)
        import cv2
        cv2.fillConvexPoly(m, cv2.convexHull(np.clip(xy, -1e4, 1e4).astype(np.int32)), 1)
        inside = (m > 0) & valid
        if inside.any() and (depth[j][inside] < .9 * z[ok].min()).mean() > .02:  # something in front of the box
            return False, seen
        pxy, pz = xp.project(np.stack([top, centre]), c2w, kk)
        for (u, v), zz in zip(pxy, pz):
            if zz > .1 and 0 <= u < w and 0 <= v < h and valid[int(v), int(u)] and depth[j][int(v), int(u)] < 1.05 * zz:
                return False, seen  # its volume is not seen through: something stands there
    return seen >= 1, seen


def plan(site):
    ref, d, clip, up, p0 = xp.load(site)
    mpn, poses, frames = ref["mpn"], d["poses_c2w"].astype(np.float64), ref["segment"]
    Kr = 2 * d["keyframe_final_fullres_intrinsics"][0].astype(float)
    size, move, clear = xp.SIDE / mpn, xp.MOVE / mpn, xp.CLEAR / mpn
    fps = float(clip["playback"]["fps"])
    last_ok = frames[-1] - int(AFTER_S * fps)
    path = np.array([poses[f][:3, 3] for f in frames])
    on_floor = lambda p: p - np.multiply.outer((p - p0) @ up, up)  # noqa: E731
    events, used = [], []
    for kind, share in (("moved", .65), ("appeared", .45), ("disappeared", .2)):  # the hardest first: their places are the rarest
        best = None
        for sh, mv in [(sh, mv) for mv in ((move, MOVE_SHORT / mpn) if kind == "moved" else (move,)) for sh in sorted(np.arange(.0, .9, .05), key=lambda x: abs(x - share))]:
            f0 = frames[int(sh * len(frames))]
            c2w = poses[f0]
            fwd = on_floor(c2w[:3, 3] + c2w[:3, 2]) - on_floor(c2w[:3, 3])
            fwd /= np.linalg.norm(fwd)
            side = np.cross(up, fwd)
            for dist in (4., 4.5, 3.5, 5., 3., 5.5, 6., 2.5):
                for lat in (0., .6, -.6, .9, -.9, 1.2, -1.2, 1.5, -1.5):
                    c = on_floor(c2w[:3, 3]) + (dist * fwd + lat * side) / mpn
                    cs = [c] + ([c + mv * side * (1 if lat <= 0 else -1)] if kind == "moved" else [])
                    if any(np.min(np.linalg.norm(on_floor(path) - x, axis=1)) < clear for x in cs) or \
                            any(np.linalg.norm(x - u) < 2 * size for x in cs for u in used):  # (a short move's two places overlap: allowed)
                        continue
                    det = [visible(xp.cube(x, up, fwd, size), poses, frames, Kr, mpn, FAR_DET) for x in cs]
                    jud = [visible(xp.cube(x, up, fwd, size), poses, frames, Kr, mpn, FAR_JUDGE) for x in cs]
                    need = {"disappeared": [(det[0], "before", MIN_DET), (jud[0], "after", MIN_JUDGE)],
                            "appeared": [(det[0], "after", MIN_DET), (jud[0], "before", MIN_JUDGE)]}.get(kind) or \
                        [(det[0], "before", MIN_DET), (jud[0], "after", MIN_JUDGE), (det[1], "after", MIN_DET), (jud[1], "before", MIN_JUDGE)]
                    fc = change_frame(need, last_ok) if all(det) and all(jud) else None
                    if fc is None:
                        continue
                    vis = det
                    oks = [seen_free(xp.cube(x, up, fwd, size), d, set(frames), mpn) for x in cs]
                    if all(o for o, _ in oks):
                        best = (cs, dist, lat, [n for _, n in oks], vis, fc, fwd, mv * mpn)
                        break
                if best:
                    break
            if best:
                break
        if best is None:
            events.append({"kind": kind, "placed": False})
            continue
        cs, dist, lat, seen, vis, fc, fwd = best[:7]
        used += cs
        events.append({"kind": kind, "placed": True, "centre_native": [x.tolist() for x in cs], "centre_m": [(x * mpn).tolist() for x in cs],
                       "change_frame": int(fc), "visible_frames": [[int(v[0]), int(v[-1])] for v in vis],
                       "keyframes_before_after_within_7m": [[keys_of([f for f in v if f < fc]), keys_of([f for f in v if f >= fc])] for v in vis],
                       "droid_keyframes_judged": seen, "distance_m": dist, "lateral_m": lat, "side_dir": fwd.tolist(),
                       **({"move_m": round(best[7], 2)} if kind == "moved" else {})})
    return events, (ref, d, clip, up, p0)


def render(site, out_mp4, truth_json):
    xp.plan = plan  # x6_plant.render draws whatever plan() placed
    t = xp.render(site, out_mp4, truth_json)
    t.update(rule=f"r5b plants: the box {NEAR}-{FAR_DET} m away on >= {MIN_DET} keyframes on its side of the change, its empty place {NEAR}-{FAR_JUDGE} m "
                  f"away on >= {MIN_JUDGE} on the other; the change >= {AFTER_S} s before the reference shot's end", label="planted: rendered boxes, not a real change")
    for e in t["events"]:
        if e.get("placed"):
            e["visible_frames_first_place"] = e["visible_frames"][0]
    Path(truth_json).write_text(json.dumps(t, indent=1))
    return t


def self_check():
    r = list(range(0, 120))
    f = change_frame([(r, "before", 4), (r, "after", 3)], 200)
    assert f is not None and keys_of([g for g in r if g < f]) >= 4 and keys_of([g for g in r if g >= f]) >= 3
    assert change_frame([(list(range(0, 30)), "before", 4), (list(range(0, 30)), "after", 3)], 200) is None, "5 keyframes cannot hold 4 + 3"
    assert change_frame([(r, "before", 4), (r, "after", 3)], 10) is None, "no change after the last allowed frame"
    assert change_frame([(r, "before", 4), (r, "after", 3), (list(range(119, 200)), "before", 3)], 400) is None, "the second place is seen before too"
    print("r5b plant self-check ok: MIN_DET keyframes of the box on its side of the change, MIN_JUDGE of its empty place on the other, room "
          "before the shot's end")


if __name__ == "__main__":
    if sys.argv[1:] == ["--self-check"]:
        self_check()
    else:
        t = render(*sys.argv[1:4])
        print(json.dumps([{k: e.get(k) for k in ("kind", "placed", "change_s", "visible_frames", "keyframes_before_after_within_7m", "distance_m", "frames_drawn")}
                          for e in t["events"]], indent=1))
