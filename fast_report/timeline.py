"""Objects over time across content windows (X6): one identity per physical object, a state per window interval.

Input, one window at a time in video order (fast_report.windows cut them): its keyframes' cameras in the window's map
frame, their depth (z, metres 'estimated', 0 = none) and person masks on a grid, and its lifted instances (points,
centroid, robust box, SigLIP 2 embedding, word). Windows of one map frame are comparable by position; a new frame (a
shot cut, a refused stitch) starts objects afresh.

Identity: an instance joins the object whose last position is within K_SIGMA x sigma_pair (sigma(z) = sqrt(POSE_M^2 +
(DEPTH_REL z)^2) per observation at its camera range z: the measured pose ~4 cm and depth ~5 %), or whose robust box
overlaps it by IOU_MIN (a partial view moves the centroid by up to half the object), one to one (Hungarian).

States of an object in a later window where it was not detected, from its own points (box_free_space's per-pixel rule
on points instead of a box: a box of an object that is not axis-aligned claims air around it). A point is judged only
inside the view's inner 90 % (BORDER), within MAX_RANGE of the camera, off person pixels; an object is judged only when
it was observed on >= MIN_OBS_KEYS keyframes and covers >= MIN_PIX grid pixels there:
  free        the window's depth, minimum over NEIGH px around where the object's points project (a pose or stitch error
              of a pixel or two must not put a point on the background beside the object), lies > 2 sigma + REL_MARGIN z
              beyond them (depth errors between windows are relative: a stitched scale, a far view) in
              >= MIN_VIEWS keyframes on >= FREE_SHARE of the judged points: the camera saw through its place -> 'disappeared', or
              'moved' when an unmatched instance of this or a later window matches its appearance (cos >= the video's
              negative-pair 99th percentile, cascade.calibrate) and size, elsewhere;
  occupied    depth at the points' depth: still there, not detected -> 'not-observed' (reason occupied-undetected);
  occluded    something (or a person) in front -> 'not-observed' (occluded);
  out-of-view / unjudged (< MIN_JUDGED points, or an object under MIN_EXTENT) -> 'not-observed'.
A new instance is 'appeared' only when an earlier window of its frame saw its place free by the same rule; otherwise
it is first seen (new content), not a change. Every change carries before/after keyframes: the evidence frames.

    python -m fast_report.timeline --self-check
"""
import sys

import numpy as np

POSE_M, DEPTH_REL = .04, .05
K_SIGMA, IOU_MIN = 3., .2
FREE_SHARE, MIN_VIEWS, MIN_JUDGED, SEEN_SHARE, MIN_EXTENT = .6, 2, 30, .5, .1
BORDER, MAX_RANGE, NEIGH, MIN_PIX, MIN_OBS_KEYS = .05, 5., 2, 25, 3  # run fx-x6-windows-time-002: 10 of 10 ME340 claims false without them
REL_MARGIN, PERSON_GROW = .2, 2  # run 003: 5 of 5 claims left were far/edge places seen < 20 % past the object, or a person's rim
SIZE_RATIO = 2.  # a moved object keeps its size within this factor (robust box diagonal)
LOOKBACK = 6     # windows an 'appeared' test looks back (the ones that could have seen the place)


def sigma(z):
    return np.sqrt(POSE_M ** 2 + (DEPTH_REL * np.asarray(z, float)) ** 2)


def iou3(a, b):
    lo, hi = np.maximum(a["lo"], b["lo"]), np.minimum(a["hi"], b["hi"])
    inter = np.prod(np.clip(hi - lo, 0, None))
    va, vb = np.prod(a["hi"] - a["lo"]), np.prod(b["hi"] - b["lo"])
    return float(inter / max(va + vb - inter, 1e-9))


def diag(o):
    return float(np.linalg.norm(o["hi"] - o["lo"]))


def project(points, c2w, K):
    cam = (points - c2w[:3, 3]) @ c2w[:3, :3]
    z = cam[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        return K[0, 0] * cam[:, 0] / z + K[0, 2], K[1, 1] * cam[:, 1] / z + K[1, 2], z


def near_min(w):
    """Per keyframe depth, minimum over the (2 NEIGH + 1)^2 neighbourhood (0 = no depth wins: unknown is never free), and
    the person masks grown by PERSON_GROW px (a person's rim is neither free nor the object)."""
    if w.get("_dmin_key") != (NEIGH, PERSON_GROW):
        from scipy.ndimage import maximum_filter, minimum_filter
        w["_dmin"] = minimum_filter(np.asarray(w["depth"], np.float32), size=(1, 2 * NEIGH + 1, 2 * NEIGH + 1))
        w["_person"] = maximum_filter(np.asarray(w["person"], bool), size=(1, 2 * PERSON_GROW + 1, 2 * PERSON_GROW + 1))
        w["_dmin_key"] = (NEIGH, PERSON_GROW)
    return w["_dmin"], w["_person"]


def place(points, w):
    """How window w sees the place `points` occupied. -> {state, views: [per judged keyframe], best_key}."""
    h, wd = w["depth"].shape[1:]
    dmin, grown = near_min(w)
    views, seen = [], 0
    for j, key in enumerate(w["keys"]):
        u, v, z = project(points, w["c2w"][j], w["K"][j])
        inside = (z > .1) & (u >= BORDER * wd) & (u < (1 - BORDER) * wd) & (v >= BORDER * h) & (v < (1 - BORDER) * h)
        if inside.mean() < SEEN_SHARE:
            continue
        seen += 1
        ui, vi, zz = u[inside].astype(int), v[inside].astype(int), z[inside]
        d, person = w["depth"][j][vi, ui], grown[j][vi, ui]
        valid = (d > 0) & ~person & (zz <= MAX_RANGE)
        m = 2 * sigma(zz)
        free, front = valid & (dmin[j][vi, ui] > zz * (1 + REL_MARGIN) + m), valid & (d < zz - m)
        pix = len(np.unique(vi[valid] * wd + ui[valid]))
        views.append({"key": int(key), "judged": int(valid.sum()) if pix >= MIN_PIX else 0, "free": int(free.sum()), "front": int(front.sum() + person.sum()),
                      "occupied": int((valid & ~free & ~front).sum()), "inside": int(inside.sum()), "pixels": pix})
    judged = [x for x in views if x["judged"] >= max(MIN_JUDGED, .3 * len(points))]
    n_free = sum(x["free"] >= FREE_SHARE * x["judged"] for x in judged)
    n_occ = sum(x["occupied"] >= .5 * x["judged"] for x in judged)
    if n_free >= MIN_VIEWS and n_free > n_occ:
        state = "free"
    elif n_occ and n_occ >= n_free:
        state = "occupied"
    elif not seen:
        state = "out-of-view"
    elif sum(x["front"] >= .5 * x["inside"] for x in views) > len(views) / 2:
        state = "occluded"
    else:
        state = "unjudged"
    best = max(judged, key=lambda x: x["free"] / x["judged"], default=None)
    return {"state": state, "views": len(judged), "free_views": int(n_free), "occupied_views": int(n_occ),
            "best_key": best["key"] if best else None, "best_free_share": round(best["free"] / best["judged"], 3) if best else None}


class Tracker:
    """add(window) in video order; objects: every identity with its observations and per-window states."""

    def __init__(self, k_sigma=K_SIGMA, iou_min=IOU_MIN, app_min=None, tau_move=.99):
        self.k_sigma, self.iou_min, self.app_min, self.tau_move = k_sigma, iou_min, app_min, tau_move
        self.objects, self.windows = [], []

    def _new(self, w, inst, how):
        o = {"id": f"g{len(self.objects)}", "frame": w["frame"], "obs": [(w["index"], inst)], "emb": inst["emb"].copy(),
             "votes": dict(inst.get("votes") or {inst["label"]: 1.}), "states": {w["index"]: how}, "gone": None, "moves": []}
        self.objects.append(o)
        return o

    def _observe(self, o, w, inst, state):
        o["obs"].append((w["index"], inst))
        e = o["emb"] * (len(o["obs"]) - 1) + inst["emb"]
        o["emb"] = e / max(np.linalg.norm(e), 1e-9)
        for k, v in (inst.get("votes") or {inst["label"]: 1.}).items():
            o["votes"][k] = o["votes"].get(k, 0.) + v
        o["states"][w["index"]] = state

    def add(self, w):
        from scipy.optimize import linear_sum_assignment
        self.windows.append(w)
        insts = w["instances"]
        live = [o for o in self.objects if o["frame"] == w["frame"]]
        cost = np.full((len(insts), len(live)), np.inf)
        for i, a in enumerate(insts):
            for j, o in enumerate(live):
                b = o["obs"][-1][1]
                s = float(np.hypot(sigma(a["range_m"]), sigma(b["range_m"])))
                d = float(np.linalg.norm(a["centroid"] - b["centroid"]))
                if (d <= self.k_sigma * s or iou3(a, b) >= self.iou_min) and (self.app_min is None or float(a["emb"] @ o["emb"]) >= self.app_min):
                    cost[i, j] = d / s
        rows, cols = linear_sum_assignment(np.where(np.isfinite(cost), cost, 1e9)) if cost.size else ([], [])
        pairs = [(r, c) for r, c in zip(rows, cols) if np.isfinite(cost[r, c])]
        used_i, used_o = {r for r, _ in pairs}, {c for _, c in pairs}
        for r, c in pairs:
            o = live[c]
            if o["gone"] is not None:  # detected again where it was said to be gone: the claim is withdrawn
                o["gone"].update(withdrawn_in_window=w["index"])
                o["gone"] = None
            self._observe(o, w, insts[r], {"state": "static"})
        free = []
        for j, o in enumerate(live):
            if j in used_o:
                continue
            last = o["obs"][-1][1]
            if o["gone"] is not None:
                o["states"][w["index"]] = {"state": o["gone"]["kind"], "since_window": o["gone"]["window"]}
                continue
            if diag(last) < MIN_EXTENT:
                o["states"][w["index"]] = {"state": "not-observed", "reason": "too small to judge"}
                continue
            if len({k for _, x in o["obs"] for k in x["keys"]}) < MIN_OBS_KEYS:
                o["states"][w["index"]] = {"state": "not-observed", "reason": "too few observations to judge"}
                continue
            p = place(last["points"], w)
            if p["state"] == "free":
                free.append((o, p))
            else:
                o["states"][w["index"]] = {"state": "not-observed", "reason": {"occupied": "occupied-undetected"}.get(p["state"], p["state"]), "place": p}
        new = [i for i in range(len(insts)) if i not in used_i]
        # moved: a place seen free, and its appearance and size again elsewhere in this window
        for o, p in free:
            last = o["obs"][-1][1]
            best = None
            for i in new:
                a = insts[i]
                cos = float(a["emb"] @ o["emb"])
                far = np.linalg.norm(a["centroid"] - last["centroid"]) > self.k_sigma * float(np.hypot(sigma(a["range_m"]), sigma(last["range_m"])))
                size = max(diag(a), 1e-3) / max(diag(last), 1e-3)
                if cos >= self.tau_move and far and 1 / SIZE_RATIO <= size <= SIZE_RATIO and (best is None or cos > best[1]):
                    best = (i, cos)
            change = {"object": o["id"], "window": w["index"], "t_after": w["t"][0], "t_before": self.windows_by(o["obs"][-1][0])["t"][1],
                      "before_key": last["best_key"], "after_key": p["best_key"], "place": p, "from_centroid": last["centroid"].tolist()}
            if best:
                i, cos = best
                new.remove(i)
                change.update(kind="moved", to_centroid=insts[i]["centroid"].tolist(), appearance_cos=round(cos, 4),
                              after_key_new_place=insts[i]["best_key"], distance_m=round(float(np.linalg.norm(insts[i]["centroid"] - last["centroid"])), 3))
                o["moves"].append(change)
                self._observe(o, w, insts[i], {"state": "moved", "change": change})
            else:
                change.update(kind="disappeared")
                o["gone"] = change
                o["states"][w["index"]] = {"state": "disappeared", "change": change}
        # new instances: appeared (an earlier window saw the place free) or first seen
        for i in new:
            a = insts[i]
            how = {"state": "first-seen"}
            if diag(a) >= MIN_EXTENT and len(a["keys"]) >= MIN_OBS_KEYS:
                for prev in [x for x in self.windows[:-1] if x["frame"] == w["frame"]][::-1][:LOOKBACK]:
                    p = place(a["points"], prev)
                    if p["state"] == "free":
                        how = {"state": "appeared", "change": {"kind": "appeared", "window": w["index"], "t_before": prev["t"][1], "t_after": w["t"][0],
                                                               "before_key": p["best_key"], "after_key": a["best_key"], "place": p,
                                                               "to_centroid": a["centroid"].tolist()}}
                        break
                    if p["state"] in ("occupied", "occluded"):
                        how = {"state": "first-seen", "reason": f"place {p['state']} in window {prev['index']}"}
                        break
            o = self._new(w, a, how)
            if how["state"] == "appeared":
                how["change"]["object"] = o["id"]

    def windows_by(self, index):
        return next(w for w in self.windows if w["index"] == index)

    def changes(self):
        out = []
        for o in self.objects:
            for s in o["states"].values():
                if "change" in s and s["state"] in ("appeared", "disappeared", "moved"):
                    out.append(s["change"])
        return out

    def timelines(self):
        """Per object: firstSeen/lastSeen (keyframe times), positions per window, states as time intervals."""
        rows = []
        for o in self.objects:
            wins = [w for w in self.windows if w["frame"] == o["frame"] and w["index"] >= o["obs"][0][0]]
            iv = []
            for w in wins:
                s = o["states"].get(w["index"], {"state": "not-observed", "reason": "no record"})
                name = s["state"] if s["state"] != "not-observed" else f"not-observed:{s.get('reason')}"
                if iv and iv[-1]["state"] == name:
                    iv[-1]["t1"] = w["t"][1]
                    iv[-1]["windows"].append(w["index"])
                else:
                    iv.append({"state": name, "t0": w["t"][0], "t1": w["t"][1], "windows": [w["index"]]})
            times = [t for _, inst in o["obs"] for t in inst["times"]]
            rows.append({"id": o["id"], "frame": o["frame"], "label": max(o["votes"], key=o["votes"].get),
                         "first_seen_s": round(min(times), 2), "last_seen_s": round(max(times), 2),
                         "positions": [{"window": wi, "centroid": np.round(inst["centroid"], 3).tolist()} for wi, inst in o["obs"]],
                         "intervals": iv, "moves": len(o["moves"]), "windows_observed": len({wi for wi, _ in o["obs"]})})
        return rows


# ---------- self-check: a synthetic room rendered by ray casting ----------

def _render(boxes, c2w, K, hw, wall=6.):
    """z-depth of axis-aligned boxes in front of a wall at world z = wall (a camera looking along +z)."""
    h, w = hw
    v, u = np.mgrid[0:h, 0:w] + .5
    d = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1) @ c2w[:3, :3].T
    o = c2w[:3, 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        depth = np.where(d[..., 2] > 0, (wall - o[2]) / d[..., 2], 0.)
        for lo, hi in boxes:
            t0, t1 = (lo - o) / d, (hi - o) / d
            near, far = np.minimum(t0, t1).max(-1), np.maximum(t0, t1).min(-1)
            hit = (near <= far) & (near > 0)
            depth = np.where(hit & (near < depth), near, depth)
    return depth  # the ray parameter is z-depth: d has camera z = 1


def _cam(x, yaw=0.):
    c, s = np.cos(yaw), np.sin(yaw)
    m = np.eye(4)
    m[:3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
    m[:3, 3] = [x, 0, 0]
    return m


def _inst(lo, hi, emb, label, key, t):
    lo, hi = np.array(lo, float), np.array(hi, float)
    g = np.stack(np.meshgrid(*[np.linspace(a, b, 9) for a, b in zip(lo, hi)]), -1).reshape(-1, 3)
    front = g[g[:, 2] <= lo[2] + 1e-9]  # what a camera at z = 0 sees of it: its front face
    return {"points": front, "centroid": front.mean(0), "lo": lo, "hi": hi, "emb": emb, "label": label, "times": [t], "best_key": key,
            "keys": [key, key + 1, key + 2],
            "range_m": float(lo[2])}


def self_check():
    K, hw = np.array([[200., 0, 160], [0, 200, 120], [0, 0, 1]]), (240, 320)
    rng = np.random.default_rng(0)
    e = {k: v / np.linalg.norm(v) for k, v in {n: rng.normal(size=16) for n in "ABCDEFP"}.items()}
    box = {"A": ([-1.2, -.3, 3.], [-.6, .3, 3.5]),   # static
           "B": ([-.2, -.3, 3.], [.3, .3, 3.5]),      # moves to B2
           "B2": ([1.6, -.3, 3.], [2.1, .3, 3.5]),
           "C": ([.8, -.3, 2.8], [1.2, .3, 3.2]),     # disappears
           "D": ([-1.9, -.3, 3.2], [-1.5, .3, 3.6]),  # appears
           "E": ([2.6, -.2, 4.5], [3.4, .2, 4.8]),    # hidden by an occluder in window 1
           "F": ([4.5, -.3, 3.], [5., .3, 3.5])}      # out of view in window 1

    def window(i, frame_boxes, insts, cams, occ=(), pose_error=0.):
        c2w = np.stack([_cam(x) for x in cams])
        depth = np.stack([_render([box[b] for b in frame_boxes] + list(occ), c, K, hw) for c in c2w])
        c2w[:, 0, 3] += pose_error  # the recorded camera, off by pose_error from the one that saw
        return {"index": i, "frame": 0, "t": [i * 2., i * 2. + 1.8], "keys": [10 * i + j for j in range(len(cams))], "c2w": c2w,
                "K": np.repeat(K[None], len(cams), 0), "depth": depth, "person": np.zeros(depth.shape, bool),
                "instances": [_inst(*box[b], e[b[0]], b[0], 10 * i, i * 2.) for b in insts]}
    w0 = window(0, "ABCEF", "ABCEF", [0., .1, .2, 3.8, 4.])  # the two cameras at x ~ 4 see F
    occluder = ([1.3, -.5, 2.5], [2., .5, 2.7])
    w1 = window(1, ["A", "B2", "D", "E"], ["A", "B2", "D"], [0., .1, .2], occ=[occluder])  # C gone, B moved, D new, E behind
    t = Tracker(tau_move=.9)
    t.add(w0)
    t.add(w1)
    by = {o["obs"][0][1]["label"]: o for o in t.objects}
    st = {k: o["states"][1]["state"] for k, o in by.items()}
    assert st["A"] == "static", st
    assert st["B"] == "moved" and len(by["B"]["obs"]) == 2 and len(t.objects) == 6, "B keeps one identity with two places"
    assert st["C"] == "disappeared", st
    assert st["D"] == "appeared" and by["D"]["obs"][0][0] == 1, st
    assert by["E"]["states"][1] == {"state": "not-observed", "reason": "occluded", "place": by["E"]["states"][1]["place"]}, by["E"]["states"][1]
    assert by["F"]["states"][1]["reason"] == "out-of-view", by["F"]["states"][1]
    kinds = sorted(c["kind"] for c in t.changes())
    assert kinds == ["appeared", "disappeared", "moved"], kinds
    assert all(c["before_key"] is not None and c["after_key"] is not None for c in t.changes()), "every change has evidence frames"
    # C detected again at its place later: the disappearance is withdrawn, never silently kept
    w2 = window(2, "ACE", "AC", [0., .1, .2])
    t.add(w2)
    assert by["C"]["gone"] is None and by["C"]["states"][2]["state"] == "static" and "withdrawn_in_window" in by["C"]["states"][1]["change"]
    # a missed detection with the object still there is not a change
    t2 = Tracker(tau_move=.9)
    t2.add(window(0, "AB", "AB", [0., .1, .2]))
    t2.add(window(1, "AB", "A", [0., .1, .2]))
    assert t2.objects[1]["states"][1]["reason"] == "occupied-undetected" and not t2.changes()
    # a thin pole (4 px), still there, its recorded camera 4.5 cm off (3 px at 3 m): its edge pixels see the wall, it is not gone
    box["P"] = ([.3, -.3, 3.], [.36, .3, 3.06])
    t3 = Tracker(tau_move=.9)
    t3.add(window(0, "P", "P", [0., .1, .2]))
    t3.add(window(1, "P", "", [0., .1, .2], pose_error=.045))
    assert t3.objects[0]["states"][1]["state"] == "not-observed" and not t3.changes(), t3.objects[0]["states"][1]
    tl = {r["label"]: r for r in t.timelines()}
    assert [iv["state"] for iv in tl["C"]["intervals"]] == ["first-seen", "disappeared", "static"]
    assert tl["B"]["positions"][1]["centroid"][0] > 1.5 and tl["B"]["moves"] == 1
    print("timeline self-check ok: static, moved (one identity), disappeared (withdrawn when seen again), appeared, occluded, "
          "out of view, missed detection, a thin object under a 3 px pose error")


if __name__ == "__main__":
    assert sys.argv[1:] == ["--self-check"], __doc__
    self_check()
