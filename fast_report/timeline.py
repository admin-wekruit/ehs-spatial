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
Two windows are compared only when the stitches between them changed the scale by <= CHAIN_MAX in all (w['chain'], the
running sum of |log s|; 0 when one DA3 run holds the whole shot): the windows' own floor-plane scales then agree.

    python -m fast_report.timeline --self-check
"""
import sys

import numpy as np

POSE_M, DEPTH_REL = .04, .05
K_SIGMA, IOU_MIN = 3., .2
FREE_SHARE, MIN_VIEWS, MIN_JUDGED, SEEN_SHARE, MIN_EXTENT = .6, 2, 30, .5, .1
BORDER, MAX_RANGE, NEIGH, MIN_PIX, MIN_OBS_KEYS = .05, 5., 2, 25, 3  # run fx-x6-windows-time-002: 10 of 10 ME340 claims false without them
REL_MARGIN, PERSON_GROW = .2, 2  # run 003: 5 of 5 claims left were far/edge places seen < 20 % past the object, or a person's rim
MIN_OBS_WINDOWS = 1  # windows an object must be seen in before its place is judged (2: one window's depth fluke is not a place)
CHAIN_MAX = .1  # stitched windows: sum of |log Sim3 scale| between the two windows compared (runs 003/004: Walmart's 0.87 link)
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


def place(points, w, idx=None, min_views=MIN_VIEWS, min_pix=MIN_PIX, min_judged=MIN_JUDGED, rel_margin=REL_MARGIN, free_px=False):
    """How window w sees the place `points` occupied. -> {state, views: [per judged keyframe], best_key}. r5b: idx = the
    keyframes of w to read (default all: a card's shot dict is read in place, never copied per window); min_* = the width
    ends' test (cards) judges its small probes on fewer pixels and in the object's own keyframes (one depth map: no
    margin between windows, rel_margin 0), the defaults are X6's."""
    h, wd = w["depth"].shape[1:]
    dmin, grown = near_min(w)
    views, seen = [], 0
    for j in (range(len(w["keys"])) if idx is None else idx):
        key = w["keys"][j]
        u, v, z = project(points, w["c2w"][j], w["K"][j])
        inside = (z > .1) & (u >= BORDER * wd) & (u < (1 - BORDER) * wd) & (v >= BORDER * h) & (v < (1 - BORDER) * h)
        if inside.mean() < SEEN_SHARE:
            continue
        seen += 1
        ui, vi, zz = u[inside].astype(int), v[inside].astype(int), z[inside]
        d, person = w["depth"][j][vi, ui], grown[j][vi, ui]
        valid = (d > 0) & ~person & (zz <= MAX_RANGE)
        m = 2 * sigma(zz)
        free, front = valid & (dmin[j][vi, ui] > zz * (1 + rel_margin) + m), valid & (d < zz - m)
        pix = len(np.unique(vi[valid] * wd + ui[valid]))
        views.append({"key": int(key), "judged": int(valid.sum()) if pix >= min_pix else 0, "free": int(free.sum()), "front": int(front.sum() + person.sum()),
                      "occupied": int((valid & ~free & ~front).sum()), "inside": int(inside.sum()), "pixels": pix,
                      **({"j": j, "px": (vi[free], ui[free])} if free_px else {})})
    judged = [x for x in views if x["judged"] >= max(min_judged, .3 * len(points))]
    n_free = sum(x["free"] >= FREE_SHARE * x["judged"] for x in judged)
    n_occ = sum(x["occupied"] >= .5 * x["judged"] for x in judged)
    if n_free >= min_views and n_free > n_occ:
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
    out = {"state": state, "views": len(judged), "free_views": int(n_free), "occupied_views": int(n_occ),
           "best_key": best["key"] if best else None, "best_free_share": round(best["free"] / best["judged"], 3) if best else None}
    if free_px:  # r5b: where the camera saw through the place, per free view (the look-alike test reads what is there)
        out["free_px"] = [(x["j"], x["px"]) for x in judged if x["free"] >= FREE_SHARE * x["judged"]]
    return out


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
            if len({k for _, x in o["obs"] for k in x["keys"]}) < MIN_OBS_KEYS or len({wi for wi, _ in o["obs"]}) < MIN_OBS_WINDOWS:
                o["states"][w["index"]] = {"state": "not-observed", "reason": "too few observations to judge"}
                continue
            if abs(w.get("chain", 0.) - self.windows_by(o["obs"][-1][0]).get("chain", 0.)) > CHAIN_MAX:
                o["states"][w["index"]] = {"state": "not-observed", "reason": "loose stitch chain"}
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
        # new instances: appeared (an earlier window saw the place free) or first seen; an object with too few keyframes
        # for the test when first seen is tested once it has them (its first window is still the one that counts)
        for i in new:
            o = self._new(w, insts[i], {"state": "first-seen", "pending": True})
        for o in self.objects:
            first = o["states"][o["obs"][0][0]]
            if first.get("pending") and o["frame"] == w["frame"] and o["obs"][-1][0] == w["index"]:
                self._appeared(o)

    def _appeared(self, o):
        wi0, a = o["obs"][0]
        pts = np.concatenate([x["points"] for _, x in o["obs"]])
        keys = {k for _, x in o["obs"] for k in x["keys"]}
        first = o["states"][wi0]
        if diag(a) < MIN_EXTENT or len(keys) < MIN_OBS_KEYS:
            return
        first.pop("pending")
        w0 = self.windows_by(wi0)
        for prev in [x for x in self.windows if x["frame"] == o["frame"] and x["index"] < wi0 and abs(x.get("chain", 0.) - w0.get("chain", 0.)) <= CHAIN_MAX][::-1][:LOOKBACK]:
            p = place(pts[::max(1, len(pts) // 400)], prev)
            if p["state"] == "free":
                o["states"][wi0] = {"state": "appeared", "change": {"kind": "appeared", "window": wi0, "t_before": prev["t"][1], "t_after": w0["t"][0],
                                                                    "before_key": p["best_key"], "after_key": a["best_key"], "place": p, "object": o["id"],
                                                                    "to_centroid": a["centroid"].tolist(), "decided_in_window": o["obs"][-1][0]}}
                return
            if p["state"] in ("occupied", "occluded"):
                first["reason"] = f"place {p['state']} in window {prev['index']}"
                return

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


# ---------- r5b: a card's timeline over its shot's content windows ----------
# The core lifts a shot once (one DA3 forward, one map frame), so a card is one object of one shot. Its timeline cuts the
# shot into fast_report.windows' content windows (a window's own keyframes: the carried ones belong to the one before) and
# says per window what the object was: seen there ('first seen' / 'appeared' / 'static' / 'moved') with values measured on
# that window's points alone, or not seen, and then why, from place() on its last seen points over the unseen windows so far
# ('disappeared' once the camera saw through its place in >= MIN_VIEWS keyframes, withdrawn if it is seen there again; else
# 'not observed' with the place's state). A value change between two windows is flagged only when it exceeds both windows'
# u (the shared-scale term left out: one shot, one scale); a position change is a 'moved' claim only when the old place (the
# earlier points away from the new ones) is seen empty too. Intervals are the runs of windows with one state and no flagged
# change; each carries its values pooled over its windows. Compact: values are [value, u] (a position [[x, y], u]).

TL_MIN_POINTS, TL_PLACE_KEYS, TL_PLACE_POINTS = 8, 24, 400
TL_MOVE_SIGMA = 3.  # an earlier window's point farther than this x sigma from every later point is part of the old place
TL_FIELDS = ("position_xy", "top_above_floor", "base_above_floor", "height", "width", "depth")
TL_EVENTS = ("appeared", "first seen", "moved")
TL_RULE = ("content windows (ORB co-visibility, fast_report.windows, threshold 0.40); per window: seen (values from its own points, "
           "each +-u) or the place test on its last seen points over the unseen windows so far (timeline.place: 'disappeared' once "
           "the camera saw through the place in >= 2 keyframes, withdrawn if it is seen there again); a change is flagged only when "
           "it exceeds both windows' u (without the shared scale term); 'moved' needs the old place seen empty too; 'appeared' needs "
           "its place seen empty before")


def window_rows(windows, times, end_s):
    """fast_report.windows' windows (local keyframe indices, carried first) -> [{w, keys (own), t: [t0, t1), reason}]."""
    own = [list(w["keys"][w["carried"]:]) or list(w["keys"]) for w in windows]
    return [{"w": i, "keys": ks, "t": [round(float(times[ks[0]]), 2), round(float(times[own[i + 1][0]]) if i + 1 < len(own) else float(end_s), 2)],
             "reason": windows[i].get("reason")} for i, ks in enumerate(own)]


def _sample(Q, n=TL_PLACE_POINTS):
    return Q[:: max(1, len(Q) // n)]


def _spread_keys(ks, n=TL_PLACE_KEYS):
    ks = sorted(set(ks))
    return ks if len(ks) <= n else [ks[int(i)] for i in np.linspace(0, len(ks) - 1, n)]


def _t(s, j):
    return round(float(s["times"][j]), 2)


def _t_key(s, key):
    ks = [int(k) for k in s["keys"]]
    return _t(s, ks.index(int(key))) if key is not None and int(key) in ks else None


def _brief(p):
    return {k: p.get(k) for k in ("state", "views", "free_views", "occupied_views", "best_key")}


def _compact(vals):
    return None if not vals else {f: [vals[f]["value"], vals[f]["u"]] for f in TL_FIELDS if isinstance(vals.get(f), dict) and "value" in vals[f]}


LOOKALIKE_SHARE = .3  # r5b: a place seen through onto a same-word object on this share of its free pixels is no claim


def lookalike(s, p, word, own):
    """What the camera saw through a place judged free (p from place(..., free_px=True)): the share of its free pixels that
    the pick maps (s['labels'], per local keyframe on DA3's raster, object index + 1) give to another object detected with the
    same word (s['label_words'], s['label_ids']). A static object lifted twice from far and near views (a depth estimate that
    drifts along a walk: Walmart's planted boxes, 1-2 m) is seen through its first place onto its second card: that is no
    change. -> (share, the look-alike's id or None)."""
    labels, words, ids = s.get("labels"), s.get("label_words"), s.get("label_ids")
    if labels is None or not words or not word or not p.get("free_px"):
        return 0., None
    head = lambda w: str(w or "").lower().split()[-1:]  # noqa: E731  'cardboard box' and 'box' are one kind
    hit, n, other = 0, 0, {}
    for j, (vi, ui) in p["free_px"]:
        codes = np.asarray(labels[j])[vi, ui].astype(int)
        n += len(codes)
        for c in codes[codes > 0]:
            if c - 1 < len(words) and ids[c - 1] not in own and head(words[c - 1]) == head(word):
                hit += 1
                other[ids[c - 1]] = other.get(ids[c - 1], 0) + 1
    return (hit / n if n else 0.), (max(other, key=other.get) if other else None)


NOT_SEEN = {"occupied": "its place is occupied: there, not detected", "occluded": "occluded", "out-of-view": "out of view",
            "unjudged": "not judged (too few pixels of its place)", "free": "its place seen empty"}


def card_timeline(s, observed, frame, world, measure, compare, word=None, own=()):
    """One card over its shot's content windows (s['windows'], cards' shot dict, read in place).
    observed: local keyframes the object was seen on (pick-map detections or lifted points); frame, world: its points'
    local keyframe and shot-frame position; measure(local keys) -> {field: value dict with u_rel} or None (too few points);
    compare(a, b) -> {field: {delta, u, flagged}} for two windows' values; word, own: the object's detected word and its ids (its
    card and the objects merged into it) for the look-alike test on every place seen free. -> {windows, intervals, changes, rule} or None."""
    alike = lambda p: lookalike(s, p, word, set(own)) if p["state"] == "free" else (0., None)  # noqa: E731
    if not s.get("windows"):
        return None
    rows = window_rows(s["windows"], s["times"], float(s["times"][-1]) + .2)
    obs = set(int(j) for j in observed)
    seen = [sorted(obs & set(r["keys"])) for r in rows]
    first = next((i for i, x in enumerate(seen) if x), None)
    if first is None:
        return None
    pts_of = lambda ks: world[np.isin(frame, ks)]  # noqa: E731
    enough = lambda Q: Q if len(Q) >= MIN_JUDGED else world  # noqa: E731
    first_pts = _sample(enough(pts_of(seen[first])))
    out, changes = [], []
    last, gone, unseen = None, None, []  # last: (row index, its values, its points); unseen: keyframes since the last sighting
    for i, (r, ks) in enumerate(zip(rows, seen)):
        row = {"w": r["w"], "t": r["t"], "seen": len(ks)}
        if i < first:  # before its first sighting: is its place seen empty (evidence for 'appeared') or not seen yet
            p = place(first_pts, s, _spread_keys(r["keys"]))
            row.update(state="not observed", reason="not seen yet" + ("; " + NOT_SEEN[p["state"]] if p["state"] in ("free", "occupied", "occluded") else ""),
                       place=_brief(p))
        elif ks:
            vals = measure(ks)
            row["v"] = _compact(vals)
            if gone is not None:  # seen again after a 'disappeared': the claim is withdrawn, never kept
                for x in out[gone:]:
                    if x["state"] == "disappeared":
                        x.update(state="not observed", reason="its place was seen empty, but it was seen there again later: withdrawn")
                changes[:] = [c for c in changes if not (c["kind"] == "disappeared" and c["window"] == out[gone]["w"])]
                gone = None
            if i == first:  # the keyframes before its first sighting, this window's too (a change inside a window is still seen)
                row.update(_appeared(s, [j for x in rows[:first + 1] for j in x["keys"] if j < ks[0]], first_pts, ks, alike))
                if row["state"] == "appeared":
                    changes.append({"kind": "appeared", "window": r["w"], **row["evidence"]})
            else:
                row["state"] = "static"
                ch = compare(last[1], vals) if (last and last[1] and vals) else None
                if ch:
                    row["d"] = {f: [c["delta"], c["u"][0], c["u"][1], c["flagged"]] for f, c in ch.items()}
                if ch and ch.get("position_xy", {}).get("flagged"):
                    now, old = pts_of(ks), last[2]
                    if len(old) and len(now):
                        from scipy.spatial import cKDTree
                        cam = s["c2w"][ks[0]][:3, 3]
                        old = old[cKDTree(now).query(old)[0] > TL_MOVE_SIGMA * sigma(np.linalg.norm(old - cam, axis=1))]
                    p = place(_sample(old), s, _spread_keys(r["keys"]), free_px=True) if len(old) >= MIN_JUDGED else {"state": "unjudged"}
                    share, other = alike(p)
                    if p["state"] == "free" and share >= LOOKALIKE_SHARE:
                        p = {**p, "state": f"seen through onto a look-alike ({other})"}
                    if p["state"] == "free":
                        j0 = seen[last[0]][-1]
                        row.update(state="moved", evidence={"t_before": _t(s, j0), "t_after": _t(s, ks[0]), "before_key": int(s["keys"][j0]),
                                                            "after_key": p["best_key"], "new_place_key": int(s["keys"][ks[0]]), "free_views": p["free_views"],
                                                            "distance_m": ch["position_xy"]["delta"], "distance_u_m": max(ch["position_xy"]["u"])})
                        changes.append({"kind": "moved", "window": r["w"], **row["evidence"]})
                    else:
                        row["note"] = f"its position differs by more than both u, but its old place was not seen empty ({p['state']}): no claim of motion"
            last = (i, vals, pts_of(ks))
            unseen = [j for j in r["keys"] if j > ks[-1]]  # this window's keyframes after its last sighting count as unseen too
        else:  # after it was seen, not seen here: the place test over the keyframes since its last sighting
            unseen += r["keys"]
            if gone is not None:
                row.update(state="disappeared", reason=f"gone since window {out[gone]['w']}")
            else:
                p = place(_sample(enough(last[2])), s, _spread_keys(unseen), free_px=True)
                row["place"] = _brief(p)
                share, other = alike(p)
                if p["state"] == "free" and share >= LOOKALIKE_SHARE:
                    row.update(state="not observed", reason=f"its place was seen through onto a look-alike ({other}, {share:.0%} of the pixels): "
                                                            "likely this object placed apart by the depth estimate, not a disappearance")
                elif p["state"] == "free":
                    gone, j0 = i, seen[last[0]][-1]
                    ev = {"t_before": _t(s, j0), "t_after": _t_key(s, p["best_key"]), "before_key": int(s["keys"][j0]), "after_key": p["best_key"],
                          "free_views": p["free_views"]}
                    row.update(state="disappeared", evidence=ev)
                    changes.append({"kind": "disappeared", "window": r["w"], **ev})
                else:
                    row.update(state="not observed", reason=NOT_SEEN[p["state"]])
        out.append(row)
    ivs = intervals(out, seen)
    for x in ivs:
        ks = x.pop("_keys")
        x["v"] = _compact(measure(sorted(set(ks)))) if ks else None
    return {"windows": out, "intervals": ivs, "changes": changes, "rule": TL_RULE}


def _appeared(s, before, pts, ks, alike=lambda p: (0., None)):
    """'appeared' when the keyframes before its first sighting (the latest 2 x TL_PLACE_KEYS of them, <= LOOKBACK windows' worth)
    saw its place empty; 'first seen' when they saw it occupied or occluded (there before, not detected), saw through it onto a
    look-alike (the same object placed apart), or never saw it."""
    if before:
        p = place(pts, s, _spread_keys(before[-2 * TL_PLACE_KEYS:]), free_px=True)
        share, other = alike(p)
        if p["state"] == "free" and share >= LOOKALIKE_SHARE:
            return {"state": "first seen", "reason": f"its place was seen through onto a look-alike ({other}, {share:.0%} of the pixels): "
                                                     "likely this object placed apart by the depth estimate, not an appearance"}
        if p["state"] == "free":
            return {"state": "appeared", "evidence": {"t_before": _t_key(s, p["best_key"]), "t_after": _t(s, ks[0]), "before_key": p["best_key"],
                                                      "after_key": int(s["keys"][ks[0]]), "free_views": p["free_views"]}}
        if p["state"] in ("occupied", "occluded"):
            return {"state": "first seen", "reason": f"its place was {p['state']} before its first sighting: there before, not detected"}
    return {"state": "first seen", "reason": "new content (its place was not seen before)"}


def intervals(rows, seen):
    """Runs of windows with one state and no flagged change between them (an event and the static windows after it are one
    run); '_keys': the run's seen keyframes (the caller measures them)."""
    out = []
    for row, ks in zip(rows, seen):
        flagged = any(d[3] for d in (row.get("d") or {}).values())
        if out and not flagged and (out[-1]["state"] == row["state"] and row["state"] not in TL_EVENTS or
                                    out[-1]["state"] in TL_EVENTS and row["state"] == "static"):
            out[-1]["t"][1] = row["t"][1]
            out[-1]["w"][1] = row["w"]
            out[-1]["_keys"] += ks
        else:
            out.append({"state": row["state"], "t": list(row["t"]), "w": [row["w"], row["w"]], "_keys": list(ks),
                        **({"reason": row["reason"]} if row.get("reason") else {}), **({"d": row["d"]} if flagged else {}),
                        **({"evidence": row["evidence"]} if row.get("evidence") else {})})
    return out


def shown(tl, mobility, raw_states):
    """The timeline a card shows for its class (apply_name): a deformable object changes shape (no place claims: it is
    re-measured per window) and an agent moves by nature (present / not observed); others keep the raw states."""
    if tl is None:
        return None
    for key in ("windows", "intervals"):
        for x, st in zip(tl[key], raw_states[key]):
            x["state"] = st
    tl["changes"] = list(raw_states["changes"])
    if mobility == "deformable":
        for x in tl["windows"] + tl["intervals"]:
            if x["state"] in ("disappeared", "moved"):
                x["state"] = "not observed" if x["state"] == "disappeared" else "static"
                x["reason"] = "a deformable object changes shape: re-measured per window, no place claim"
        tl["changes"] = []
    elif mobility == "agent":
        for x in tl["windows"] + tl["intervals"]:
            x["state"] = "present" if x["state"] in ("static", "moved") + TL_EVENTS else "not observed"
        tl["changes"] = []
    tl["mobility_rule"] = {"deformable": "re-measured per window, no place claims", "agent": "an agent: present / not observed, no place claims"}.get(mobility)
    return tl


def no_claims(tl, why):
    """A timeline whose changes are not claims (why: e.g. a screen-fixed overlay): its states stay, its events become notes."""
    if not tl:
        return tl
    for x in tl["windows"] + tl["intervals"]:
        if x["state"] in ("appeared", "disappeared", "moved"):
            x["state"], x["reason"] = {"appeared": "first seen", "disappeared": "not observed", "moved": "static"}[x["state"]], why
    tl["changes"], tl["claims"] = [], why
    return tl


def raw_states(tl):
    return None if tl is None else {"windows": [x["state"] for x in tl["windows"]], "intervals": [x["state"] for x in tl["intervals"]],
                                    "changes": list(tl["changes"])}


def interval_at(tl, t):
    """The interval of a timeline holding video time t (the viewer's scrubber), or None."""
    for x in (tl or {}).get("intervals", []):
        if x["t"][0] <= t < x["t"][1]:
            return x
    return None


def _u_rel(f):
    """A position's u without its shared-scale part (both places are in one shot: the scale scales their distance, it does not
    make one)."""
    return float(np.sqrt(max(float(f["u"]) ** 2 - float((f.get("parts") or {}).get("scale", 0.)) ** 2, 0.)))


def link_moves(cards, emb=None, tau=None, max_gap=1):
    """r5b: across cards of one shot, a card that 'disappeared' and one that 'appeared' within max_gap windows of it, never
    seen at the same time, sizes within SIZE_RATIO, places apart by more than both positions' u (without the shared scale),
    and the same kind of thing (appearance cos >= tau when both have a vector, else the same detected word) are one object
    that moved: both cards say so, each with the other's id; a card joins one pair at most, the nearest in windows, then in
    place (Walmart planted: two identical boxes, one gone at 19.6 s, one moved 0.5 m at 23.9 s: the move is the near pair).
    -> the pairs [(from id, to id, record)]."""
    by_shot = {}
    for c in cards:
        tl = (c.get("time") or {}).get("timeline")
        if c.get("kind") != "object" or not tl:
            continue
        for ch in tl["changes"]:
            if ch["kind"] in ("disappeared", "appeared"):
                by_shot.setdefault(c["shot"], {"disappeared": [], "appeared": []})[ch["kind"]].append((c, ch))
    pairs, used = [], set()
    for sh in by_shot.values():
        cand = []
        for a, ca in sh["disappeared"]:
            for b, cb in sh["appeared"]:
                if a["id"] == b["id"] or abs(cb["window"] - ca["window"]) > max_gap:
                    continue
                ta, tb = a["time"], b["time"]
                if tb["first_seen_s"] < ta["last_seen_s"]:  # seen at the same time: two objects
                    continue
                sa, sb = (a.get("raw") or {}).get("size") or {}, (b.get("raw") or {}).get("size") or {}
                la, lb = sa.get("longest"), sb.get("longest")
                if not la or not lb or not 1 / SIZE_RATIO <= la / lb <= SIZE_RATIO:
                    continue
                pa, pb = (a["physical"].get("position_xy") or {}), (b["physical"].get("position_xy") or {})
                if "value" not in pa or "value" not in pb:
                    continue
                d = float(np.linalg.norm(np.subtract(pa["value"], pb["value"])))
                if d <= _u_rel(pa) or d <= _u_rel(pb):
                    continue
                va, vb = (emb or {}).get(a["id"]), (emb or {}).get(b["id"])
                if va is not None and vb is not None and tau is not None:
                    cos = float(np.dot(va, vb) / max(np.linalg.norm(va) * np.linalg.norm(vb), 1e-9))
                    if cos < tau:
                        continue
                    how = f"appearance cos {cos:.3f} >= {tau:.3f}"
                else:
                    wa, wb = (a.get("identity") or {}).get("detector_words") or [], (b.get("identity") or {}).get("detector_words") or []
                    if not (wa and wb and wa[0] == wb[0]):
                        continue
                    how, cos = f"the same detected word '{wa[0]}'", None
                cand.append((abs(cb["window"] - ca["window"]), d, a, ca, b, cb, d, how))
        for _, _, a, ca, b, cb, d, how in sorted(cand, key=lambda x: (x[0], x[1])):
            if a["id"] in used or b["id"] in used:
                continue
            used |= {a["id"], b["id"]}
            rec = {"from": a["id"], "to": b["id"], "t_before": ca["t_before"], "t_after": cb["t_after"], "old_place_empty_key": ca["after_key"],
                   "before_key": ca["before_key"], "new_place_key": cb["after_key"], "distance_m": round(d, 3),
                   "distance_u_m": round(float(max(_u_rel(a["physical"]["position_xy"]), _u_rel(b["physical"]["position_xy"]))), 3), "same_object_by": how}
            ca.update(kind="moved", to=b["id"], link=rec)
            cb.update(kind="moved", **{"from": a["id"]}, link=rec)
            pairs.append((a["id"], b["id"], rec))
    return pairs


def move_threshold(emb, together, q=99):
    """The appearance cos a move needs: the q-th percentile over pairs of cards seen at the same time (known different
    objects) in this video; None without such pairs."""
    cs = [float(np.dot(emb[a], emb[b]) / max(np.linalg.norm(emb[a]) * np.linalg.norm(emb[b]), 1e-9)) for a, b in together if a in emb and b in emb]
    return float(np.percentile(cs, q)) if len(cs) >= 20 else None


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
    t4 = Tracker(tau_move=.9)  # the same disappearance across a stitch that changed the scale by 15 %: not judged
    t4.add(window(0, "AC", "AC", [0., .1, .2]))
    t4.add(dict(window(1, "A", "A", [0., .1, .2]), chain=.14))
    assert not t4.changes() and t4.objects[1]["states"][1]["reason"] == "loose stitch chain"
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
    t5 = Tracker(tau_move=.9)  # D first seen on one keyframe: tested once it has three, and dated to its first window
    t5.add(window(0, "A", "A", [0., .1, .2]))
    w1 = window(1, "AD", "AD", [0., .1, .2])
    w1["instances"][1]["keys"] = [10]
    t5.add(w1)
    assert t5.objects[1]["states"][1] == {"state": "first-seen", "pending": True}
    t5.add(window(2, "AD", "AD", [0., .1, .2]))
    assert t5.objects[1]["states"][1]["state"] == "appeared" and t5.objects[1]["states"][1]["change"]["decided_in_window"] == 2
    tl = {r["label"]: r for r in t.timelines()}
    assert [iv["state"] for iv in tl["C"]["intervals"]] == ["first-seen", "disappeared", "static"]
    assert tl["B"]["positions"][1]["centroid"][0] > 1.5 and tl["B"]["moves"] == 1
    print("timeline self-check ok: static, moved (one identity), disappeared (withdrawn when seen again), appeared, occluded, "
          "out of view, missed detection, a thin object under a 3 px pose error, a deferred appearance")


def self_check_cards():
    """r5b: card timelines on a rendered shot (12 keyframes, 3 windows of 4): a box that stays, one taken away after window 0,
    one put down in window 2, one pushed 0.9 m in window 1; a missed detection is not a change; per-window values carry u."""
    K, hw = np.array([[200., 0, 160], [0, 200, 120], [0, 0, 1]]), (240, 320)
    box = {"A": ([-1.2, -.3, 3.], [-.6, .3, 3.5]), "C": ([.8, -.3, 2.8], [1.2, .3, 3.2]), "D": ([-1.9, -.3, 3.2], [-1.5, .3, 3.6]),
           "M0": ([-.3, -.3, 3.3], [.1, .3, 3.7]), "M1": ([.6, -.3, 3.3], [1., .3, 3.7])}
    scene = [["A", "C", "M0"]] * 4 + [["A", "M1"]] * 4 + [["A", "D", "M1"]] * 4  # what stands there in each keyframe
    c2w = np.stack([_cam(.02 * j) for j in range(12)])
    s = {"keys": list(range(0, 72, 6)), "times": [j * .2 for j in range(12)], "c2w": c2w, "K": np.repeat(K[None], 12, 0),
         "depth": np.stack([_render([box[b] for b in sc], c, K, hw) for sc, c in zip(scene, c2w)]), "person": np.zeros((12, *hw), bool),
         "windows": [{"keys": list(range(4 * i, 4 * i + 4)), "carried": 0, "reason": "content"} for i in range(3)]}

    def card(name, seen, ks_pts=None):
        lo, hi = np.array(box[name][0]), np.array(box[name][1])
        g = np.stack(np.meshgrid(*[np.linspace(a, b, 7) for a, b in zip(lo, hi)]), -1).reshape(-1, 3)
        front = g[g[:, 2] <= lo[2] + 1e-9]
        return front, seen

    def measure(ks, W, F):
        m = np.isin(F, ks)
        if m.sum() < TL_MIN_POINTS:
            return None
        c = W[m].mean(0)
        return {"position_xy": {"value": [round(c[0], 3), round(c[2], 3)], "u": .1, "u_rel": .1}}

    def compare(a, b):
        d = float(np.linalg.norm(np.subtract(b["position_xy"]["value"], a["position_xy"]["value"])))
        return {"position_xy": {"delta": d, "u": [.1, .1], "flagged": d > .1}}

    def run(parts):  # parts: [(box name, keyframes it is seen on)]
        W = np.concatenate([card(n, ks)[0] for n, ks in parts for _ in ks])
        F = np.concatenate([[k] * len(card(n, ks)[0]) for n, ks in parts for k in ks])
        return card_timeline(s, sorted({k for _, ks in parts for k in ks}), F, W, lambda ks: measure(ks, W, F), compare)
    a = run([("A", range(12))])
    assert [x["state"] for x in a["windows"]] == ["first seen", "static", "static"] and not a["changes"], a["windows"]
    assert a["windows"][1]["v"]["position_xy"][1] == .1 and a["windows"][1]["d"]["position_xy"][3] is False
    c = run([("C", range(4))])
    assert [x["state"] for x in c["windows"]] == ["first seen", "disappeared", "disappeared"] and c["changes"][0]["kind"] == "disappeared", c["windows"]
    assert c["changes"][0]["before_key"] == 18 and c["changes"][0]["after_key"] >= 24, c["changes"]
    d = run([("D", range(9, 12))])  # put down at keyframe 8, first detected on keyframe 9 (the window's first keyframes saw its place)
    assert [x["state"] for x in d["windows"]][2] == "appeared" and d["changes"][0]["kind"] == "appeared", d["windows"]
    assert d["intervals"][-1]["state"] == "appeared" and d["intervals"][0]["state"] == "not observed"
    m = run([("M0", range(4)), ("M1", range(4, 12))])  # one lifted object at two places: moved by 0.9 m, its old place seen empty
    assert [x["state"] for x in m["windows"]] == ["first seen", "moved", "static"] and m["changes"][0]["kind"] == "moved", m["windows"]
    assert abs(m["changes"][0]["distance_m"] - .9) < .05 and len(m["intervals"]) == 2
    miss = run([("A", [0, 1, 2, 3, 10, 11])])  # not detected in window 1: its place is occupied, no change
    assert miss["windows"][1]["state"] == "not observed" and "occupied" in miss["windows"][1]["reason"] and not miss["changes"], miss["windows"]
    assert interval_at(m, 1.0)["state"] == "moved" and interval_at(m, .1)["state"] == "first seen"
    tl = run([("C", range(4))])
    shown(tl, "deformable", raw_states(tl))
    assert not tl["changes"] and tl["windows"][1]["state"] == "not observed", "a deformable object gets no place claims"
    tl2 = run([("C", range(4))])
    no_claims(tl2, "overlay")
    assert not tl2["changes"] and tl2["claims"] == "overlay"
    # a static box lifted twice (its first card placed 0.6 m too near by a drifting depth estimate): the camera later sees through
    # the first card's place onto the box itself, which the pick maps give to its second card with the same word: no claim
    box["A0"] = ([.8, -.3, 2.2], [1.2, .3, 2.6])  # where the first card's points are (the true box C stands at z 2.8-3.2)
    still = [["A", "C"]] * 12
    s["depth"] = np.stack([_render([box[b] for b in sc], c, K, hw) for sc, c in zip(still, c2w)])
    bare = np.stack([_render([box["A"]], c, K, hw) for c in c2w])
    s["labels"], s["label_words"], s["label_ids"] = np.where(s["depth"] < bare - .01, 2, 0).astype(np.int16), ["box", "cardboard box"], ["obj-a", "obj-b"]
    W, F = card("A0", range(4))[0], np.zeros(0, int)
    W = np.concatenate([W] * 4)
    F = np.repeat(np.arange(4), len(W) // 4)
    drift = card_timeline(s, list(range(4)), F, W, lambda ks: measure(ks, W, F), compare, "box", {"obj-a"})
    assert not drift["changes"] and "look-alike" in drift["windows"][1]["reason"], drift["windows"][1]
    bare_s = {k: v for k, v in s.items() if k not in ("labels",)}
    assert card_timeline(bare_s, list(range(4)), F, W, lambda ks: measure(ks, W, F), compare, "box", {"obj-a"})["changes"][0]["kind"] == "disappeared", \
        "without the pick maps the drifted place reads as gone (the failure the look-alike test stops)"
    # two identical boxes: one gone in window 1, one moved 0.5 m in window 3 (its new place appeared then): the move pairs the near two
    def fake(cid, kind, w, xy, first, last):
        return {"id": cid, "kind": "object", "shot": 0, "raw": {"size": {"longest": .8}}, "identity": {"detector_words": ["box"]},
                "physical": {"position_xy": {"value": xy, "u": .9, "parts": {"scale": .8}}},
                "time": {"first_seen_s": first, "last_seen_s": last, "timeline": {"changes": [{"kind": kind, "window": w, "t_before": 1., "t_after": 2.,
                                                                                               "before_key": 1, "after_key": 2}]}}}
    cs = [fake("gone", "disappeared", 1, [6.9, .2], 0., 4.), fake("old", "disappeared", 3, [9.8, .1], 5., 8.), fake("new", "appeared", 3, [10.4, .6], 8.5, 9.)]
    pairs = link_moves(cs)
    assert [(a, b) for a, b, _ in pairs] == [("old", "new")] and cs[0]["time"]["timeline"]["changes"][0]["kind"] == "disappeared", pairs
    print("timeline card self-check ok: static, disappeared with evidence keys, appeared inside a window, moved (one lifted object), "
          "a missed detection, the scrubber's interval at t, deformable and overlay claims dropped, a drifted second lift is no claim, "
          "the move pairs the nearest look-alike")


if __name__ == "__main__":
    assert sys.argv[1:] == ["--self-check"], __doc__
    self_check()
    self_check_cards()
