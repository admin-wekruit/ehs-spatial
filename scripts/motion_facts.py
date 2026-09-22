"""What a moving object did, as facts an agent (or a person) can read: path, speed, stops, and which objects it passed.

Input is the object's per-view surface centroid over time (mono_room.py dynamic) and the static objects of the same
map. Positions are projected onto the floor plane, so walking speed is not mixed with crouching. The centroid is of
the visible side only, so a turn shifts it a little; stops are periods slower than STILL of the object's own height
per second. Units are the report's: metres when the map is calibrated, native units otherwise, and the text says which.

  python scripts/motion_facts.py --self-check
"""
import numpy as np

STILL, NEAR_MIN_SECONDS = .15, .5  # speed below this share of a person's height per second counts as standing; shorter visits are not listed
IGNORED = ("floor", "wall", "ceiling", "unnamed surface")


def summarise(samples, up, origin, statics, unit="原生单位", height=None):
    """samples: [(t, xyz)] in time order; statics: [(entity id, label, centroid xyz)]. Returns a JSON-able dict with a Chinese text."""
    up = np.asarray(up, float) / np.linalg.norm(up)
    a = np.cross(up, [1., 0, 0]) if abs(up[0]) < .9 else np.cross(up, [0, 1., 0]); a /= np.linalg.norm(a)
    b = np.cross(up, a)
    t = np.array([s[0] for s in samples], float)
    raw = np.array([s[1] for s in samples], float)
    # A visible-side centroid jumps when the mask changes from frame to frame; a 5-sample running median keeps the walk, not the jumps.
    xyz = np.array([np.median(raw[max(0, n - 2):n + 3], 0) for n in range(len(raw))])
    plan = np.stack([(xyz - origin) @ a, (xyz - origin) @ b], 1)
    steps = np.linalg.norm(np.diff(plan, axis=0), axis=1)
    gaps = np.diff(t)
    # speed over +-2 samples (about half a second at the usual sampling), not over one 0.1 s step that one mask change can double
    lo, hi = np.maximum(np.arange(len(gaps)) - 2, 0), np.minimum(np.arange(len(gaps)) + 3, len(t) - 1)
    span = t[hi] - t[lo]
    speed = np.divide(np.linalg.norm(plan[hi] - plan[lo], axis=1), span, out=np.zeros(len(gaps)), where=span > 0)
    height = height or float(np.percentile((xyz - origin) @ up, 90))  # the visible top of the object, as its own size scale
    still = float(gaps[speed < STILL * height].sum()) if len(gaps) else 0.
    visits = []
    for (ti, p) in zip(t, xyz):
        near = [(np.linalg.norm(((np.asarray(c) - p) @ np.stack([a, b], 1))), ident, label) for ident, label, c in statics
                if not any(w in (label or "").lower() for w in IGNORED)]
        if not near:
            continue
        _, ident, label = min(near)
        if visits and visits[-1]["entityId"] == ident:
            visits[-1]["until"] = float(ti)
        else:
            visits.append({"entityId": ident, "label": label, "from": float(ti), "until": float(ti)})
    visits = [v for v in visits if v["until"] - v["from"] >= NEAR_MIN_SECONDS]
    facts = {"from": float(t[0]), "until": float(t[-1]), "samples": len(t), "unit": unit,
             "pathLength": round(float(steps.sum()), 3), "netDisplacement": round(float(np.linalg.norm(plan[-1] - plan[0])), 3),
             "meanSpeed": round(float(steps.sum() / max(t[-1] - t[0], 1e-6)), 3), "maxSpeed": round(max(float(speed.max()) if len(speed) else 0., float(steps.sum() / max(t[-1] - t[0], 1e-6))), 3),  # never below the mean
             "stillSeconds": round(still, 1), "near": [{**v, "from": round(v["from"], 1), "until": round(v["until"], 1)} for v in visits],
             "trajectory": [[round(float(ti), 2), *[round(float(v), 3) for v in p]] for ti, p in zip(t, xyz)],
             "basis": "centroid of the visible surface per sampled view, 5-sample running median, projected on the floor plane; nearest named static object by floor-plan distance"}
    route = " → ".join(f"{v['label']}（{v['from']:.1f}–{v['until']:.1f} 秒）" for v in facts["near"]) or "没有停留超过0.5秒的具名物体"
    facts["text"] = (f"出现 {facts['from']:.1f}–{facts['until']:.1f} 秒（{len(t)} 个采样视图）；地面上走了 {facts['pathLength']:.2f} {unit}，"
                     f"起点到终点直线 {facts['netDisplacement']:.2f} {unit}；平均 {facts['meanSpeed']:.2f} {unit}/秒，最快 {facts['maxSpeed']:.2f}；"
                     f"基本不动的时间共 {facts['stillSeconds']:.1f} 秒。依次靠近：{route}。")
    return facts


def self_check():
    up, origin = [0., 0, 1], np.zeros(3)
    walk = [(i * .1, [i * .1, 0., .9]) for i in range(31)] + [(3.1 + i * .1, [3., 0., .9]) for i in range(1, 21)]  # 1 unit/s for 3 s, then stands 2 s
    statics = [("desk", "desk", [0., .5, .7]), ("door", "door", [3., .5, 1.]), ("floor", "floor", [1.5, 0, 0])]
    f = summarise(walk, up, origin, statics)
    assert abs(f["pathLength"] - 3.) < .15 and abs(f["netDisplacement"] - 3.) < .15  # the running median trims a little at the ends
    assert abs(f["stillSeconds"] - 2.) < .35, f["stillSeconds"]  # stands 2 s; the half-second speed baseline blurs the start of the stop
    assert [v["label"] for v in f["near"]] == ["desk", "door"], f["near"]  # the floor is never "near"
    assert "desk" in f["text"] and "door" in f["text"]
    jumpy = [(t, [x + (.5 if n == 12 else 0.), y, z]) for n, (t, (x, y, z)) in enumerate(walk)]  # one mask glitch mid-walk
    assert summarise(jumpy, up, origin, statics)["maxSpeed"] < 1.5, "a single centroid jump is not a sprint"
    print("motion facts check passed: path, stops and the objects passed, floor ignored")


if __name__ == "__main__":
    self_check()
