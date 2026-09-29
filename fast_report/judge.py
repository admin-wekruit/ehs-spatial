"""Judgement engine (MVP spec section 5, builder B): the EHS checks J0-J8 over the object cards. Each row is PASS / FAIL /
NEEDS_REVIEW / NO_DATA with value +- u against its threshold, the VLM's option probabilities per view set, reasons and
evidence frames; written as the `judgements` layer (v1 geometry only, v2 with the VLM answers).

Reuse: video.banded_verdict is the numeric rule with band = the fact's own u; video.worst_verdict (fixed there: NO_DATA ranks
above PASS) aggregates over time and per object; rules._assess_clearance merges guard fragments and measures J6's gap;
policy's thresholds (MAX_HEIGHT 2.5 m, MIN_SEPARATION 0.711 m, MAX_TILT); X8's decider (vlm.options). Every metric value
carries a 20% scale part in u (spec 0.1); people rules R1-R3 pass through with video.scale_gated unchanged.

Cards read (spec 4.9): id, kind, shot, identity {name, confidence, calibrated}, class {category, mobility},
physical {top_above_floor, base_above_floor, height, width, depth, position_xy, footprint_xy, overhang,
principal_axis_tilt_deg, planar_slope_deg: {value, u, n_subsets, parts, scale} or {status, reason}; size_check {status};
flags ['fragmented support']}, views {best, azimuth_spread_deg}, time {first_seen_s, last_seen_s}. Person cards: track,
points [{t, frame, bbox (0-1), score, foot_surface}], rules. ctx: judge.context()'s shape (the contract in spec 9 A, plus
per-shot 'room_floor' points for J5's scan).

  python -m fast_report.judge --self-check                  # no GPU: the verdict table, gap asymmetry, NO_DATA paths, J5
  python -m fast_report.judge --calibrate ANSWERS.json      # set-d answers (modal_apps/judge_decider.py) -> calibration.json
"""
import io
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path

import numpy as np

from ehs_spatial.video import BAND_M, FAIL, NO_DATA, PASS, PERSON_HEIGHT_M, REVIEW, banded_verdict, worst_verdict

CALIBRATION = Path(__file__).with_name("calibration.json")
SCALE_REL = .20  # spec 4.4: assumed 1.6 m camera height (-19%/+12% for 1.3-1.8 m) and 0.99-1.17 against the delivered scale
GRID_M, SCAN_M = .05, 5.  # J5: floor grid, how far each side is scanned
# J5 rev. 2 (round-1 review R6): the camera's corridor (the operator, and a cart pushed ahead) is free; a surface cell needs
# 3 TSDF points (a floater is one or two); an unseen floor run up to 0.3 m between seen cells is bridged (TSDF holes on
# glossy floors); a footprint with half its area on the corridor moves with the camera (the operator's cart)
CORRIDOR_M, CART_AHEAD_M, PATH_STEP_M, OCC_MIN_PTS, GAP_BRIDGE_M, CART_SHARE, PERSON_MIN_W = .30, 1.2, .10, 3, .30, .5, .45
DEPTH_REL = .05  # DA3's depth error, a share of the range (spec 4.4)
AISLE_MIN_M = .711  # OSHA 1910.36(g)(2): exit access at least 28 in
NEAR_PATH_M, TRIP_PATH_M, ON_FLOOR_M, FLOOR_BASE_M, GUARD_NEAR_M, ON_FLOOR_PASS_M = 1., .5, .05, .10, 1.5, .30
TOP_BIAS_M = .20  # R3: tops read low on retail (signed median -0.20 m Sam's Club, -0.12 m Walmart): J1 PASS allows for it
WALKED_M = .5  # a camera or person path shorter than this is someone standing, not a walked path
OBSTACLE_H_M = (.1, 1.8)  # J5: a room-mesh point at this height above the floor occupies its cell
STACK_MAX_M, OVERHANG_MAX_M, STACK_TILT_DEG, LADDER_TILT_DEG, GUARD_CLEAR_M, STAND_M = 2.5, .10, 5., 10., .6, .30
CLIMB_TOP_M, CLIMB_SIDE_M, CLIMB_SLOPE_DEG = (.3, 1.2), .3, 10.
FACE_MIN_H_M, FACE_MIN_PTS, FACE_SLOPE_DEG, TSDF_VOXEL_M = .5, 60, 60., .03  # J2 from observed faces (face_overhang)
MIN_MASS = .5  # a Qwen answer whose option letters hold less than this is unanswered
CAL_MIN_N, CAL_MIN_EACH, CAL_MAX_ECE = 30, 5, .10  # a question decides only when calibrated on this much, this well (spec 10)
HAZARD_WAIT_S = 40.  # Gemini answers are waited for this long after the wave is sent (typically 10-25 s, stragglers 57-63 s:
# hazard-001); later ones fall back to Qwen in that run and are carried into the next
ONE_SET = "one view set: uncertainty from model terms only (likely understated)"
# J3a's foot-height cue decides nothing yet: people standing on the floor read 0.12-0.62 m (median 0.36 m, 15 visible resting
# feet, run mvp-b-judge-samsclub-002; ME340 one foot at 0.40 m) against u of about 0.1 m, and no clip here has a positive
FOOT_CUE_VALIDATED = False
FOOT_CUE_NOTE = "foot-height cue not validated: people on the floor read +0.36 m median (Sam's Club), more than its u"
LABELS = ["verdicts: PASS/FAIL only when value +- u clears the threshold; a VLM answer alone never makes a FAIL",
          "scale estimated: floor plane + an assumed 1.6 m camera height (a 20% scale part is in every metric u)",
          "VLM probabilities are model outputs; 'calibrated' = Platt-fitted on X8 set d (agent-labelled frames)"]

# words (a name matches when it is the word or ends with ' ' + word: 'cardboard box' is a box)
STACK = ("stacked boxes", "stack of boxes", "pallet of goods", "loaded pallet", "stacked pallets", "boxes", "pallet")
CLIMBABLE = ("shelf", "shelving", "shelves", "rack", "pallet", "box", "boxes", "carton", "crate", "cart", "trolley", "machine",
             "lathe", "mill", "cnc machine", "table", "desk", "workbench", "bench")
GUARD = ("guard", "machine guard", "fence", "safety fence", "barrier", "railing", "guard rail", "cage")
LADDER = ("ladder", "step ladder", "stepladder")
DEFORMABLE = ("cable", "hose", "cord", "wire", "rope", "chain", "strap", "curtain", "wrap")
AGENT = ("person", "forklift", "pallet jack", "agv", "robot arm")
FIXED = ("wall", "shelf", "shelving", "shelves", "rack", "machine", "lathe", "mill", "cnc machine", "door", "column", "conveyor",
         "cabinet", "sign", "locker", "duct", "pipe")

SCENE = "These are crops from a video of a workplace (a shop floor, warehouse, store, lab or office)."
MARKS = ("In the first image of each pair, white outlines with numbers mark detected things; the second image is the same crop "
         "without marks. Judge only what is inside outline [1]: touching neighbours and "
         "side-by-side panels or boxes are separate things. Judge a striped surface by its geometry (horizontal: a work "
         "surface, sloped: a guide, vertical: a barrier), not by its pattern.")  # agent.AGENT_LESSONS 1, 3, 7 in English
YNC = ["yes", "no", "cannot tell"]
QUESTIONS = {  # X8 set d's wording (verbatim, so set d's calibration transfers as far as it can) -> options, hazard options
    "q1": ("Is there a cable or hose lying across the floor where people walk?", YNC, (0,)),
    "q2": ("Are boxes or goods stacked unstably, so that they could fall?", YNC, (0,)),
    "q3": ("Is a person standing on a shelf, rack or pallet?", YNC, (0,)),
    "q4": ("Is the aisle or exit route blocked or narrowed by an object standing in it?", YNC, (0,)),
    "q5": ("Is a person standing on a raised platform, step or box instead of the floor?", YNC, (0,)),
    "hard hat": ("Is person [1] wearing a hard hat?", YNC, (1,)),
    "hi-vis vest": ("Is person [1] wearing a high-visibility vest?", YNC, (1,)),
}
CHECKS = {  # id -> title, severity, questions, rule source
    "J1": ("stack height", "major", (), "policy MAX_HEIGHT; starter rule 'stacked pallets <= 2.5 m'"),
    "J2": ("stack stability", "major", ("q2",), "MVP: overhang <= 0.10 m, tilt <= 5 deg (geometry._spatial_state overhang)"),
    "J3a": ("person standing or climbing on an object", "critical", ("q3", "q5"), "MVP; OSHA 1910.22 (walking-working surfaces)"),
    "J3b": ("climbable surface", "minor", (), "photo product: climbability is REVIEW only (providers/gemini.py)"),
    "J4": ("cable or hose across a walked path", "major", ("q1",), "MVP; OSHA 1910.22(a) (surfaces free of trip hazards)"),
    "J5": ("blocked aisle or exit access", "major", ("q4",), "policy MIN_SEPARATION arithmetic; OSHA 1910.36(g)(2) (28 in)"),
    "J6": ("clearance to guarding", "major", (), "rules._assess_clearance; starter s01 (0.6 m)"),
    "J7": ("ladder lean", "major", (), "policy MAX_TILT on the physical axis (10 deg)"),
    "J8": ("person rules", "major", (), "ehs_spatial.video R1-R3 (live_people), unchanged"),
}


# ---------- card access ----------

def name_of(card):
    return ((card.get("identity") or {}).get("name") or "").lower().strip()


def is_a(name, words):
    return any(name == w or name.endswith(" " + w) for w in words)


def mobility(card):
    m = (card.get("class") or {}).get("mobility")
    if m:
        return m
    n = name_of(card)
    return "deformable" if is_a(n, DEFORMABLE) else "agent" if is_a(n, AGENT) else "fixed" if is_a(n, FIXED) else "movable rigid"


def fact(card, key):
    f = (card.get("physical") or {}).get(key)
    return f if isinstance(f, dict) and f.get("value") is not None else None


def missing(card, key):
    f = (card.get("physical") or {}).get(key)
    what = key.replace("_", " ")
    if isinstance(f, dict) and f.get("status"):
        return f"{what}: {f['status']}" + (f" ({f['reason']})" if f.get("reason") else "")
    return f"{what}: not in the card"


def observed(card, key):
    return fact(card, key) is not None


def u_rel(f):
    """u without its scale part (a gap between two things does not carry their scale error from the origin); the
    calibration factor k in u = k sqrt(sum parts^2) is kept by scaling u."""
    parts = {k: v for k, v in (f.get("parts") or {}).items() if v is not None}
    total = math.sqrt(sum(v * v for v in parts.values()))
    return float(f["u"]) * math.sqrt(max(0., total ** 2 - parts.get("scale", 0.) ** 2)) / total if total > 0 else float(f["u"])


def doubts(card):
    ph = card.get("physical") or {}
    out = []
    size = ph.get("size_check") or {}
    if size.get("status") == "implausible":
        out.append("implausible size for its class: no rule uses it (needs review)")
    if ph.get("fragmented_support") or "fragmented support" in (ph.get("flags") or []):
        out.append("fragmented support: its sizes need review")
    return out


def footprint(card):
    """The card's footprint polygon in the shot's floor frame (shapely), None without one."""
    from shapely.geometry import Point, Polygon
    fp = (card.get("physical") or {}).get("footprint_xy")
    if isinstance(fp, dict):
        fp = fp.get("value")
    if fp and len(fp) >= 3:
        poly = Polygon(fp)
        return poly if poly.is_valid and poly.area > 0 else poly.buffer(0)
    pos = fact(card, "position_xy")
    return Point(pos["value"]).buffer(.05) if pos else None


def one_set(card):
    """A footprint-based check (J4 distance, J5 width, J6 gap) is one view set when the card's position is (the stub's always;
    run mvp-b-judge-walmart-001 let a one-set J5 FAIL through before this)."""
    pos = fact(card, "position_xy")
    return [ONE_SET] if pos is None or (pos.get("n_subsets") or 0) < 2 else []


def follow_name(card):
    """The card as its final name reads it (round-1 review R1: class, size check and so the checks followed the SAM 3 word, not
    the decider's name; 51/155/51 cards differed). Class = cards.kind_of(name) unless observed behaviour overrode it; the size
    check re-run on the card's own box for the new class; 'needs review' marks left by the old class's size check are lifted
    when the new class finds the size plausible (fragmented support keeps them). Returns the card unchanged when the name's
    class word is the card's."""
    if card.get("kind") == "person":
        return card
    from fast_report import cards as A
    name, cls = name_of(card), card.get("class") or {}
    word = A.head_match(name, A.KIND)
    if not name or word == cls.get("class_word") or cls.get("mobility_source") == "observed":
        return card
    ph = dict(card.get("physical") or {})
    box = (ph.get("box") or {}).get("size_m")
    old = ph.get("size_check") or {}
    if box and old.get("status") in ("plausible", "implausible"):
        h = fact(card, "height")
        hv = float(h["value"]) if h else float(box[2])
        base = fact(card, "base_above_floor")
        seen = all(isinstance(ph.get(k), dict) and ph[k].get("value") is not None and ph[k].get("status") not in ("at least", "not observed")
                   for k in ("height", "width", "depth"))
        new = A.size_check(name, max(max(box[:2]), hv), max(box[:2]), hv, None if base is None else float(base["value"]), seen)
        ph["size_check"] = {**new, "rechecked_for": "final name (judge)", "was": old.get("status")}
        if old.get("status") == "implausible" and new["status"] == "plausible" and not ph.get("fragmented_support"):
            for k, f in list(ph.items()):
                if isinstance(f, dict) and f.get("status") == "needs review" and "implausible" in (f.get("reason") or ""):
                    ph[k] = {kk: vv for kk, vv in f.items() if kk not in ("status", "reason")}
    return {**card, "class": {**A.kind_of(name), "followed": "final name (judge)", "was": cls.get("class_word")}, "physical": ph}


def usable(card):
    """Plausible and not fragmented: its footprint may bound a gap for another object's check."""
    return not doubts(card)


# ---------- geometry ----------

def geo(quantity, value, u, unit, threshold, direction, result, reasons, scale=None, **extra):
    return {"quantity": quantity, "value": None if value is None else round(float(value), 3), "u": None if u is None else round(float(u), 3),
            "unit": unit, "threshold": threshold, "direction": direction, "result": result,
            "scale": scale or ("estimated" if unit in ("m", "m2") else "scale-free"), "reasons": list(reasons), **extra}


def forced(g, reasons):
    """Spec 5.4: one view subset, an implausible size, fragmented support or the gap asymmetry force NEEDS_REVIEW."""
    if reasons:
        g["reasons"] += reasons
        if g["result"] in (PASS, FAIL):
            g["before_forced"], g["result"] = g["result"], REVIEW
    return g


def numeric(card, key, threshold, direction, quantity, unit, bias=0.):
    """value +- u against the threshold (video.banded_verdict). bias: a one-sided allowance against a PASS on a maximum (a
    measurement known to read low: PASS needs v + u + bias < T). g['hazard_side']: the value itself is past the threshold,
    from two or more view subsets on a plausible card, and not a bound that could put it back (combine() may then FAIL it
    with a calibrated 'likely hazard' picture)."""
    f = fact(card, key)
    if f is None:
        return geo(quantity, None, None, unit, threshold, direction, NO_DATA, [missing(card, key)])
    v, u = float(f["value"]), float(f["u"])
    r = banded_verdict(v, threshold, u, fail_low=direction == "min")
    reasons = []
    if r == PASS and bias and direction == "max" and v + u + bias >= threshold:
        r = REVIEW
        reasons.append(f"reads low by up to {bias:.2f} m (round-1 review R3: tops shaved): {v:.2f} + {u:.2f} + {bias:.2f} m does not clear {threshold:g} m")
    word = {PASS: "clears", FAIL: "breaks", REVIEW: "straddles"}[r]
    g = geo(quantity, v, u, unit, threshold, direction, r, [f"{quantity} {v:.2f} +- {u:.2f} {unit} {word} {threshold:g} {unit}"] + reasons,
            f.get("scale"), n_subsets=f.get("n_subsets"))
    bound_back = f.get("status") in (("at most",) if direction == "max" else ("at least",)) or f.get("status") == "needs review"
    g["hazard_side"] = bool((v > threshold if direction == "max" else v < threshold) and not doubts(card) and not bound_back
                            and (f.get("n_subsets") or 0) >= 2)
    return forced(g, doubts(card) + fact_doubts(f, r, direction))


def fact_doubts(f, r, direction="max"):
    """One view subset; a value A marked 'needs review'; a bound on the wrong side ('at least' cannot PASS a maximum or FAIL a
    minimum: the true value may be larger; 'at most' the reverse)."""
    out = [ONE_SET] if (f.get("n_subsets") or 0) < 2 else []
    st = f.get("status")
    if st == "needs review":
        out.append(f"needs review ({f.get('reason')})")
    lower = (st == "at least" and (r, direction) in ((PASS, "max"), (FAIL, "min"))) or (st == "at most" and (r, direction) in ((FAIL, "max"), (PASS, "min")))
    if lower:
        out.append(f"only a bound ({st}: {f.get('reason')})")
    return out


def to_floor(frame, pts):
    o, x, z = (np.asarray(frame[k], float) for k in ("origin_m", "x", "z"))
    return (np.asarray(pts, float).reshape(-1, 3) - o) @ np.stack([x, np.cross(z, x), z]).T


def path_distance(card, ctx):
    """Nearest walked path (the camera's floor path, every person path of the shot) -> (distance, u, path id, the path
    point nearest the footprint) or None. u: the card's position u without scale, the path's own (camera: the shot's pose u;
    a person: the mono band), and 20% of the distance for scale."""
    from shapely.geometry import LineString, Point
    from shapely.ops import nearest_points
    poly = footprint(card)
    walked = (ctx.get("walked") or {}).get(card.get("shot")) or {}
    if poly is None or not walked:
        return None
    best = None
    for pid, xy in walked.items():
        xy = np.asarray(xy, float)
        if len(xy) == 0 or path_length(xy) < WALKED_M:
            continue
        line = LineString(xy) if len(xy) > 1 else Point(xy[0])
        d = poly.distance(line)
        if best is None or d < best[0]:
            best = (d, pid, np.asarray(nearest_points(line, poly)[0].coords[0]), xy)
    if best is None:
        return None
    d, pid, near, xy = best
    shot = shot_of(ctx, card.get("shot"))
    pos = fact(card, "position_xy")
    u_card = u_rel(pos) if pos else .1
    u_path = shot.get("u_pose_m", .04) if pid == "camera" else BAND_M
    return d, math.sqrt(u_card ** 2 + u_path ** 2 + (SCALE_REL * d) ** 2), pid, near, xy


def shot_of(ctx, index):
    return next((s for s in ctx.get("shots", []) if s.get("index") == index), {})


def g_j4(card, ctx, cards):
    """Cable or hose across a walked path. PASS off the floor only when base - u > 0.30 m: a thin hose lying on the floor read a
    base of 0.28-0.39 m in round 1 (its floor part eroded or trimmed; ME340 obj-1-474 / obj-1-91, agent-labelled on the floor,
    were PASS there); PASS far from paths (distance - u > 1 m); FAIL on the floor within 0.5 m. hazard_side: not clearly off
    the floor, within 0.5 m, two view subsets, plausible."""
    base, pd = fact(card, "base_above_floor"), path_distance(card, ctx)
    if base is None or pd is None:
        return geo("distance to walked path", None, None, "m", TRIP_PATH_M, "min", NO_DATA,
                   [missing(card, "base_above_floor") if base is None else "no walked path or footprint in this shot"])
    d, u, pid, _, _ = pd
    b, ub = float(base["value"]), float(base["u"])
    on = banded_verdict(b, ON_FLOOR_M, ub, fail_low=False)  # FAIL here = clearly off the floor
    off = b - ub > ON_FLOOR_PASS_M  # off the floor by more than a thin hose's base error
    reasons = [f"base {b:.2f} +- {ub:.2f} m above the floor", f"{d:.2f} +- {u:.2f} m from walked path '{pid}'"]
    if off:
        r, reasons = PASS, reasons + [f"hangs {b - ub:.2f} m or more above the floor"]
    elif d - u > NEAR_PATH_M:
        r = PASS
    elif on == PASS and d + u < TRIP_PATH_M:
        r = FAIL
    else:
        r = REVIEW
        if on == FAIL:
            reasons.append(f"off the floor, but not by more than {ON_FLOOR_PASS_M:g} m (a thin hose's base reads high)")
    g = geo("distance to walked path", d, u, "m", TRIP_PATH_M, "min", r, reasons, path=pid)
    if off:  # integration: the PASS rests on the base height, so the geometry line shows that quantity
        g = geo("base above the floor (off the floor: no trip hazard)", b, ub, "m", ON_FLOOR_PASS_M, "min", r, reasons,
                base.get("scale"), path=pid, path_distance_m=round(float(d), 3))
    gap = ["footprint seen from one side: the gap may be smaller"] if r == PASS and not off and not observed(card, "depth") else []
    extra = doubts(card) + gap + fact_doubts(base, on, "max") + (one_set(card) if r != PASS or not off else [])
    g["hazard_side"] = bool(not off and b <= ON_FLOOR_PASS_M and d < TRIP_PATH_M and not doubts(card) and not one_set(card)
                            and (base.get("n_subsets") or 0) >= 2)
    return forced(g, extra)


def resample(xy, step=None):
    """A floor path at `step` spacing -> (points (n, 2), unit tangents (n, 2) over +-0.3 m, arc length (n,))."""
    step = step or PATH_STEP_M
    xy = np.asarray(xy, float)
    s = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    if len(xy) < 2 or s[-1] < step:
        return np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0)
    t = np.arange(0., s[-1] + 1e-9, step)
    pts = np.c_[np.interp(t, s, xy[:, 0]), np.interp(t, s, xy[:, 1])]
    k = max(1, int(round(.3 / step)))
    d = pts[np.minimum(np.arange(len(pts)) + k, len(pts) - 1)] - pts[np.maximum(np.arange(len(pts)) - k, 0)]
    return pts, d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9), t


def corridor(shot, walked):
    """The camera's own floor path buffered by CORRIDOR_M, and CART_AHEAD_M past its end: the operator (and a cart pushed ahead
    of the camera) went there, so nothing there is an obstacle (the TSDF keeps the cart, smeared along the walk: the 'free width
    0.00 m' rows of round 1). None without a moving camera."""
    from shapely.geometry import LineString
    cam = (walked or {}).get("camera")
    if cam is None or path_length(cam) < WALKED_M:
        return None
    pts, tan, _ = resample(cam)
    if not len(pts):
        return None
    return LineString(np.r_[pts, pts[-1:] + CART_AHEAD_M * tan[-1:]]).buffer(CORRIDOR_M)


def aisle(shot, cards, walked):
    """J5's free width along every walked path of one shot, from observed surfaces (spec 5.1's scan, rev. 2): on a 5 cm floor
    grid a cell is occupied by >= OCC_MIN_PTS room (TSDF) points 0.1-1.8 m above the floor, or by a usable object footprint
    standing in that band (a footprint is a hull of observed points too); seen when any room point below 1.8 m falls in it
    (dilated one cell); the camera's corridor is free and seen. Every PATH_STEP_M along a path, both sides are scanned across
    it to the first occupied cell ('surface'), or to the start of an unseen run longer than GAP_BRIDGE_M ('unobserved': the
    width is then a lower bound), or SCAN_M ('open'). Person paths keep only samples whose own cell is free and seen (mono
    person paths run through racks). -> {'samples': [...], 'corridor', 'cart': {card index: overlap}} or None."""
    from shapely import contains_xy
    pts = shot.get("room_floor")
    if pts is None and shot.get("room_points") is not None and shot.get("floor_frame"):
        pts = to_floor(shot["floor_frame"], shot["room_points"])
    pts = np.asarray(pts if pts is not None else np.zeros((0, 3)), float)
    if len(pts) < 100 or not walked:
        return None
    pts = pts[pts[:, 2] < OBSTACLE_H_M[1]]
    lo = pts[:, :2].min(0) - 1.
    shape = tuple(np.ceil((pts[:, :2].max(0) + 1. - lo) / GRID_M).astype(int) + 1)
    ij = ((pts[:, :2] - lo) / GRID_M).astype(int)
    seen = np.zeros(shape, bool)
    seen[ij[:, 0], ij[:, 1]] = True
    seen[1:] |= seen[:-1].copy()
    seen[:-1] |= seen[1:].copy()
    seen[:, 1:] |= seen[:, :-1].copy()
    seen[:, :-1] |= seen[:, 1:].copy()
    band = pts[:, 2] >= OBSTACLE_H_M[0]
    count = np.zeros(shape, np.int32)
    np.add.at(count, (ij[band, 0], ij[band, 1]), 1)
    occ, owner = count >= OCC_MIN_PTS, np.zeros(shape, np.int32)
    corr, under = corridor(shot, walked), None
    centres = lambda a0, a1: np.meshgrid(np.arange(a0[0], a1[0] + 1), np.arange(a0[1], a1[1] + 1), indexing="ij")  # noqa: E731
    cart = {}
    for k, c in enumerate(cards):
        base, top, poly = fact(c, "base_above_floor"), fact(c, "top_above_floor"), footprint(c)
        if poly is None or poly.is_empty or c.get("kind") == "person":
            continue
        if corr is not None and poly.area > 0 and poly.intersection(corr).area >= CART_SHARE * poly.area:
            cart[k] = round(poly.intersection(corr).area / poly.area, 2)  # moves with the camera: the operator's cart (or a misplaced footprint)
            continue
        if not usable(c) or (base is not None and base["value"] > OBSTACLE_H_M[1]) or (top is not None and top["value"] < OBSTACLE_H_M[0]):
            continue
        a0 = np.maximum(((np.asarray(poly.bounds[:2]) - lo) / GRID_M).astype(int), 0)
        a1 = np.minimum(((np.asarray(poly.bounds[2:]) - lo) / GRID_M).astype(int) + 1, np.array(shape) - 1)
        gi, gj = centres(a0, a1)
        inside = contains_xy(poly, lo[0] + (gi + .5) * GRID_M, lo[1] + (gj + .5) * GRID_M)
        occ[gi[inside], gj[inside]] = True
        owner[gi[inside], gj[inside]] = k + 1
        seen[gi[inside], gj[inside]] = True
    if corr is not None:
        a0 = np.maximum(((np.asarray(corr.bounds[:2]) - lo) / GRID_M).astype(int), 0)
        a1 = np.minimum(((np.asarray(corr.bounds[2:]) - lo) / GRID_M).astype(int) + 1, np.array(shape) - 1)
        gi, gj = centres(a0, a1)
        inside = contains_xy(corr, lo[0] + (gi + .5) * GRID_M, lo[1] + (gj + .5) * GRID_M)
        under = np.zeros(shape, bool)  # room surfaces the corridor erases: the camera passed over them (held over a bench)
        under[gi[inside], gj[inside]] = count[gi[inside], gj[inside]] >= OCC_MIN_PTS
        occ[gi[inside], gj[inside]], owner[gi[inside], gj[inside]], seen[gi[inside], gj[inside]] = False, 0, True
    K, R = int(SCAN_M / GRID_M), int(round(GAP_BRIDGE_M / GRID_M)) + 1
    steps = np.arange(1, K + 1) * GRID_M
    out = []
    for pid, xy in walked.items():
        p, t, arc = resample(xy)
        if not len(p):
            continue
        n = np.c_[-t[:, 1], t[:, 0]]
        c0 = ((p - lo) / GRID_M).astype(int)
        inside = (c0 >= 0).all(1) & (c0 < shape).all(1)
        ok = inside.copy()
        ok[inside] = ~occ[c0[inside, 0], c0[inside, 1]] & seen[c0[inside, 0], c0[inside, 1]]
        if pid == "camera":
            ok = inside
        p, n, arc = p[ok], n[ok], arc[ok]
        sides, over = [], np.zeros(len(p), bool)
        if pid == "camera" and corr is not None:  # a surface under the camera's line: its floor path is not where the feet were
            q = p[:, None, :] + n[:, None, :] * np.arange(-CORRIDOR_M, CORRIDOR_M + 1e-9, GRID_M)[None, :, None]
            c = np.clip(((q - lo) / GRID_M).astype(int), 0, np.array(shape) - 1)
            over = under[c[..., 0], c[..., 1]].any(1)
        for sgn in (1., -1.):
            q = p[:, None, :] + sgn * n[:, None, :] * steps[None, :, None]  # (S, K, 2)
            c = ((q - lo) / GRID_M).astype(int)
            valid = (c >= 0).all(2) & (c[..., 0] < shape[0]) & (c[..., 1] < shape[1])
            ci, cj = np.clip(c[..., 0], 0, shape[0] - 1), np.clip(c[..., 1], 0, shape[1] - 1)
            o = occ[ci, cj] & valid
            un = ~(seen[ci, cj] & valid)
            cs = np.c_[np.zeros(len(p), int), np.cumsum(un, 1)]
            run = np.zeros_like(un)
            run[:, :K - R + 1] = (cs[:, R:] - cs[:, :K - R + 1]) == R
            run[:, K - R + 1:] = un[:, K - R + 1:] & ~valid[:, K - R + 1:]  # the grid's edge: nothing seen beyond
            first_o = np.where(o.any(1), o.argmax(1), K)
            first_u = np.where(run.any(1), run.argmax(1), K)
            stop = np.minimum(first_o, first_u)
            dist = np.where(stop < K, (stop + .5) * GRID_M, SCAN_M)
            kind = np.where((first_o <= first_u) & (first_o < K), "surface", np.where(first_u < K, "unobserved", "open"))
            si = np.minimum(stop, K - 1)
            who = np.where(kind == "surface", owner[ci[np.arange(len(p)), si], cj[np.arange(len(p)), si]] - 1, -1)
            sides.append((dist, kind, who, p + sgn * n * dist[:, None]))
        out.append({"path": pid, "xy": p, "n": n, "arc": arc, "sides": sides, "over": over})
    return {"samples": out, "corridor": corr, "cart": cart}


def aisle_of(ctx, card, cards):
    """The shot's aisle model, built once per run (cached on the shot record, like guards_of)."""
    shot = shot_of(ctx, card.get("shot"))
    if "_aisle" not in shot:
        shot_cards = [c for c in cards if c.get("shot") == card.get("shot")]
        shot["_aisle"] = (aisle(shot, shot_cards, (ctx.get("walked") or {}).get(card.get("shot")) or {}), shot_cards)
    return shot["_aisle"]


def g_j5(card, ctx, cards):
    """Free width across the walked paths beside the object, from observed surfaces (aisle()). Samples 'at' the object: their
    cross line meets its footprint grown by max(0.15 m, its position u without scale); 'bounded' by it: a side stops on its
    footprint or on a surface cell within that zone. value = the narrowest bounded sample (else the narrowest at it: the width
    there is set by other surfaces). u: 5% of each side's distance (the lateral share of DA3's depth error for a point seen from
    the path line), half a cell per side, the shot's pose u per side, 20% scale. PASS when width - u > 0.711 m (a lower bound
    may pass); FAIL only when width + u < 0.711 m with observed surfaces on both sides, this object bounding one side on >= 2
    neighbouring samples; the gap asymmetry no longer applies (the scan measures the faces seen from the path), nor does the
    card's view-subset count (the width comes from the fused room surfaces, not from the card's subsets)."""
    import shapely
    from shapely import contains_xy
    from shapely.geometry import LineString
    q = ("free width across the walked path", "m", AISLE_MIN_M, "min")
    model, shot_cards = aisle_of(ctx, card, cards)
    poly = footprint(card)
    if model is None or poly is None:
        return geo(q[0], None, None, *q[1:], NO_DATA, ["no room points or walked path in this shot" if model is None else "no footprint"])
    me = shot_cards.index(card)
    if me in model["cart"]:
        return geo(q[0], None, None, *q[1:], NO_DATA, [f"{model['cart'][me]:.0%} of its footprint lies on the camera's own path: the "
                                                       "operator's cart or a misplaced footprint, not an aisle obstacle"])
    pos = fact(card, "position_xy")
    zone = poly.buffer(min(max(.15, u_rel(pos) if pos else .15), .6))
    best = None
    for smp in model["samples"]:  # every walked path beside it: the narrowest one it bounds decides (a PASS on the camera's wide
        p, n = smp["xy"], smp["n"]  # aisle must not hide a person squeezing past it on another path)
        if not len(p):
            continue
        (dl, kl, wl, el), (dr, kr, wr, er) = smp["sides"]
        w = dl + dr
        lines = shapely.linestrings(np.stack([p - n * SCAN_M, p + n * SCAN_M], 1))
        at = shapely.intersects(poly.buffer(GRID_M), lines)
        if not at.any():
            at = shapely.intersects(zone, lines)
        if smp["path"] != "camera":
            at &= w >= PERSON_MIN_W  # narrower than a person: the mono person path is misplaced there
        if not at.any():
            continue
        by_l = (wl == me) | ((kl == "surface") & contains_xy(zone, el[:, 0], el[:, 1]))
        by_r = (wr == me) | ((kr == "surface") & contains_xy(zone, er[:, 0], er[:, 1]))
        bounded = at & (by_l | by_r)
        sel = bounded if bounded.any() else at
        i = int(np.flatnonzero(sel)[np.argmin(w[sel])])
        run = bounded & (w <= w[i] + 2 * GRID_M)
        pairs = bool(bounded.any() and ((run[1:] & run[:-1]).any()))
        rec = {"path": smp["path"], "w": float(w[i]), "l": float(dl[i]), "r": float(dr[i]), "kinds": [str(kl[i]), str(kr[i])],
               "bounded": bool(bounded.any()), "pairs": pairs, "at": int(at.sum()), "xy": p[i], "ends": [el[i], er[i]],
               "who": [int(wl[i]), int(wr[i])], "over": int((smp["over"] & at).sum())}
        if best is None or (rec["bounded"], -rec["w"], rec["path"] == "camera") > (best["bounded"], -best["w"], best["path"] == "camera"):
            best = rec
    if best is None:
        return geo(q[0], None, None, *q[1:], NO_DATA, ["no walked-path sample passes beside it"])
    shot = shot_of(ctx, card.get("shot"))
    u_pose = shot.get("u_pose_m", .04)
    w, l, r_ = best["w"], best["l"], best["r"]
    u = math.sqrt((DEPTH_REL * l) ** 2 + (DEPTH_REL * r_) ** 2 + 2 * (GRID_M / 2) ** 2 + 2 * u_pose ** 2 + (SCALE_REL * w) ** 2)
    lower = any(k != "surface" for k in best["kinds"])
    bound_by = [("this object" if who == me else f"card:{shot_cards[who]['id']}" if who >= 0 else k) for k, who in zip(best["kinds"], best["who"])]
    r = banded_verdict(w, AISLE_MIN_M, u, fail_low=True)
    reasons = [f"free width {w:.2f} +- {u:.2f} m across path '{best['path']}' ({' / '.join(bound_by)})" + (" (a lower bound)" if lower else "")]
    if not best["bounded"]:
        reasons.append("this object does not bound the free width here: other surfaces are nearer the path")
    if best["over"] and r in (PASS, FAIL):
        r, reasons = REVIEW, reasons + [f"the camera passed over a surface at {best['over']} of these samples (held over a bench?): its floor "
                                        "path is not where the operator walked"]
    if r == FAIL and lower:
        r, reasons = REVIEW, reasons + ["one side ends where nothing was observed: the width may be larger"]
    elif r == FAIL and not (best["bounded"] and best["pairs"]):
        r, reasons = REVIEW, reasons + ["narrow here, but this object does not bound it on two neighbouring samples"]
    hz = bool(w < AISLE_MIN_M and best["bounded"] and best["pairs"] and not lower and not best["over"] and not doubts(card))
    g = geo(q[0], w, u, *q[1:], r, reasons, path=best["path"], bounded_by=bound_by, lower_bound=lower, hazard_side=hz,
            scan={"at_xy": np.round(best["xy"], 3).tolist(), "ends_xy": [np.round(e, 3).tolist() for e in best["ends"]]},
            samples_beside=best["at"], object_distance_m=round(float(poly.distance(LineString(best["ends"]))), 3) if w > 0 else None)
    return forced(g, doubts(card))


def g_j6(card, ctx, cards):
    from ehs_spatial.contracts import AssessmentStatus, Criterion, Entity3D
    from ehs_spatial.rules import FENCE_LABEL, _assess_clearance
    guards = [c for c in guards_of(ctx, cards, card.get("shot")) if c is not card and usable(c) and footprint(c) is not None]

    def entity(c, label):
        poly, h, pos = footprint(c), fact(c, "height"), fact(c, "position_xy")
        n = max((f.get("n_subsets") or 0) for f in [pos or {}, h or {}]) if (pos or h) else 0
        ring = [tuple(map(float, p)) for p in list(poly.exterior.coords)[:-1]]
        return Entity3D(entity_id=c["id"], label=label, observation_ids=[], centroid_xyz=(*map(float, poly.centroid.coords[0]), 0.),
                        footprint_xy=ring, height_m=max(0., float(h["value"])) if h else 0., evidence_frame_ids=[f"subset-{i}" for i in range(n)])
    q = ("clearance to guarding", "m", GUARD_CLEAR_M, "min")
    if footprint(card) is None:
        return geo(q[0], None, None, *q[1:], NO_DATA, ["no footprint"])
    res = _assess_clearance([entity(g, FENCE_LABEL) for g in guards] + [entity(card, "material cart")], Criterion(), capture_frame_count=2)
    if res.assessment.status == AssessmentStatus.INSUFFICIENT_EVIDENCE or res.assessment.approximate_distance_m is None:
        return geo(q[0], None, None, *q[1:], NO_DATA, ["at least 2 view subsets on both sides needed: " + "; ".join(res.warnings)])
    d = float(res.assessment.approximate_distance_m)
    u_g = max(u_rel(fact(g, "position_xy")) if fact(g, "position_xy") else .1 for g in guards)
    u_c = u_rel(fact(card, "position_xy")) if fact(card, "position_xy") else .1
    u = math.sqrt(u_c ** 2 + u_g ** 2 + (SCALE_REL * d) ** 2)
    r = banded_verdict(d, GUARD_CLEAR_M, u, fail_low=True)
    g = geo(q[0], d, u, *q[1:], r, [f"gap {d:.2f} +- {u:.2f} m to the guard hull", *res.warnings])
    gap = ["footprint seen from one side: the gap may be smaller"] if r == PASS and not all(observed(c, "depth") for c in [card, *guards]) else []
    return forced(g, doubts(card) + gap + one_set(card))


def guards_of(ctx, cards, shot):
    """The shot's guards, found once per run (a scan per card was 163k name matches: 10 s beside the cascade on Walmart)."""
    if "_guards" not in ctx:
        ctx["_guards"] = {}
        for c in cards:
            if c.get("kind") != "person" and is_guard(c):
                ctx["_guards"].setdefault(c.get("shot"), []).append(c)
    return ctx["_guards"].get(shot, [])


def is_guard(card):
    return (card.get("class") or {}).get("category", "").startswith("C") or is_a(name_of(card), GUARD)


def face_overhang(card, ctx):
    """J2's overhang from the observed faces (round-1 review R6: the cards carry none): the room's TSDF points inside the
    footprint (+5 cm) from 5 cm above the base to the top, in a frame facing the camera path (n: away from the path, t: along
    the face). Upper band (above 60% of the height) against the base band (below 40%): 'front' = how far the upper face comes
    nearer the path than the base face (p10 of n each); 'side' = how far the upper band reaches past the base band's ends
    (p3/p97 of t). value = the larger. u: the spread of 'front' over three lateral segments (the view-subset term's stand-in:
    the TSDF fuses the views), the TSDF voxel (3 cm) per face, the up direction's uncertainty over the height (the walls' p90
    plumb reading), 20% scale. -> value record, or {status: not measurable, reason}."""
    import shapely
    from shapely.geometry import LineString, Point
    shot = shot_of(ctx, card.get("shot"))
    pts, base, top, poly = shot.get("room_floor"), fact(card, "base_above_floor"), fact(card, "top_above_floor"), footprint(card)
    nm = lambda why: {"status": "not measurable", "reason": why}  # noqa: E731
    if pts is None or base is None or top is None or poly is None:
        return nm("no room points, base, top or footprint")
    b, t = float(base["value"]), float(top["value"])
    h = t - b
    if h < FACE_MIN_H_M:
        return nm(f"too low for an overhang ({h:.2f} m)")
    P = np.asarray(pts, float)
    P = P[shapely.contains_xy(poly.buffer(GRID_M), P[:, 0], P[:, 1]) & (P[:, 2] > b + GRID_M) & (P[:, 2] < t)]
    lo, up = P[:, 2] < b + .4 * h, P[:, 2] > b + .6 * h
    if lo.sum() < FACE_MIN_PTS or up.sum() < FACE_MIN_PTS:
        return nm(f"too few observed surface points in its box (base band {int(lo.sum())}, upper band {int(up.sum())})")
    cen = np.asarray(poly.centroid.coords[0])
    cam = ((ctx.get("walked") or {}).get(card.get("shot")) or {}).get("camera")
    if cam is not None and len(cam) > 1:
        line = LineString(cam)
        near = np.asarray(line.interpolate(line.project(Point(cen))).coords[0])
    else:
        near = np.zeros(2)  # a still camera: the shot's origin is the camera
    n = cen - near
    if np.linalg.norm(n) < 1e-6:
        return nm("on the camera's path: no face direction")
    n /= np.linalg.norm(n)
    tv = np.array([-n[1], n[0]])
    a, s_ = (P[:, :2] - cen) @ n, (P[:, :2] - cen) @ tv
    front = float(np.percentile(a[lo], 10) - np.percentile(a[up], 10))
    side = float(max(np.percentile(s_[up], 97) - np.percentile(s_[lo], 97), np.percentile(s_[lo], 3) - np.percentile(s_[up], 3)))
    edges = np.percentile(s_, [0, 100 / 3, 200 / 3, 100])
    seg = []
    for k in range(3):
        m = (s_ >= edges[k]) & (s_ <= edges[k + 1])
        if (m & lo).sum() >= FACE_MIN_PTS // 3 and (m & up).sum() >= FACE_MIN_PTS // 3:
            seg.append(float(np.percentile(a[m & lo], 10) - np.percentile(a[m & up], 10)))
    v = max(0., front, side)
    plumb = shot.get("plumb_u_deg") or shot.get("plumb_deg") or 2.
    parts = {"segments": (max(seg) - min(seg)) if len(seg) >= 2 else None, "surface": TSDF_VOXEL_M * math.sqrt(2),
             "up": h * math.tan(math.radians(max(1., plumb))), "scale": SCALE_REL * v}
    u = math.sqrt(sum(x * x for x in parts.values() if x is not None))
    return {"value": round(v, 3), "u": round(u, 3), "unit": "m", "level": "coarse", "scale": "estimated (floor plane + assumed 1.6 m camera height)",
            "n_subsets": len(seg), "parts": {k: None if x is None else round(x, 4) for k, x in parts.items()},
            "front_m": round(front, 3), "side_m": round(side, 3), "segments_front_m": [round(x, 3) for x in seg],
            "note": "observed faces only (room TSDF points in the footprint): an unseen side may overhang",
            "faces_seen": 2 if observed(card, "depth") else 1}


def g_j2(card, ctx, cards):
    """Stack stability (spec 5.1 J2): overhang <= 0.10 m (face_overhang) and lean <= 5 deg. The lean is the card's own
    planar slope when its plane is a face (>= 60 deg from horizontal: lean = 90 - slope, gated by >= 2 agreeing view subsets
    and the plumb check, spec 4.5), else the principal axis of a tall stack (h > 1.5 w, below 45 deg). PASS needs both parts
    to clear; FAIL needs one to break; a part not measurable leaves NEEDS_REVIEW (or NO_DATA when both are). A PASS on one
    seen face asks for a clear picture as well (g['needs_clear_picture']: combine() makes it NEEDS_REVIEW otherwise)."""
    oh = face_overhang(card, ctx)
    card = {**card, "physical": {**(card.get("physical") or {}), "overhang": oh}}
    parts = [numeric(card, "overhang", OVERHANG_MAX_M, "max", "overhang (observed faces)", "m")]
    if oh.get("value") is not None and oh["front_m"] <= oh["value"] - 1e-9:  # the side reading sets the value
        o = parts[0]
        o["hazard_side"] = False  # an occluded base band reads as a side overhang (Sam's Club obj-0-6: side 0.28 m, front -0.02 m)
        if o["result"] == FAIL and oh["front_m"] + oh["u"] < OVERHANG_MAX_M:
            o.update(result=REVIEW, before_forced=FAIL)
            o["reasons"].append(f"only its side reaches past the base ({oh['side_m']:.2f} m; front {oh['front_m']:.2f} m): a base hidden "
                                "behind something reads the same way")
    slope, tilt, h, w = (fact(card, k) for k in ("planar_slope_deg", "principal_axis_tilt_deg", "height", "width"))
    if slope is not None and slope["value"] >= FACE_SLOPE_DEG:
        lean = {**slope, "value": round(90. - float(slope["value"]), 2), "unit": "deg"}
        parts.append(numeric({**card, "physical": {**card["physical"], "lean": lean}}, "lean", STACK_TILT_DEG, "max", "lean of its front face from vertical", "deg"))
    elif h and w and h["value"] > 1.5 * w["value"] and tilt is not None and tilt["value"] < 45.:
        parts.append(numeric(card, "principal_axis_tilt_deg", STACK_TILT_DEG, "max", "principal-axis tilt", "deg"))
    else:
        why = missing(card, "planar_slope_deg") if slope is None else f"its plane is not a face (slope {slope['value']:.0f} deg)"
        parts.append(geo("lean", None, None, "deg", STACK_TILT_DEG, "max", NO_DATA, [f"lean not measurable: {why}"]))
    got = [x for x in parts if x["result"] != NO_DATA]
    if not got:
        return geo("overhang (observed faces)", None, None, "m", OVERHANG_MAX_M, "max", NO_DATA, [r for x in parts for r in x["reasons"]])
    rank = {FAIL: 3, REVIEW: 2, NO_DATA: 1, PASS: 0}
    top = max(parts, key=lambda x: rank[x["result"]])
    if top["result"] == NO_DATA:  # one part passes, the other is not measurable: not a PASS
        top = {**max(got, key=lambda x: rank[x["result"]]), "result": REVIEW}
        top["reasons"] = top["reasons"] + [r for x in parts if x["result"] == NO_DATA for r in x["reasons"]]
    out = {**top, "parts": parts, "hazard_side": any(x.get("hazard_side") for x in parts)}
    if out["result"] == PASS and (oh.get("faces_seen") or 1) < 2:
        out["needs_clear_picture"] = "one face seen: an unseen side may overhang"
    return out


def g_j3b(card, ctx, cards):
    slope, top, w, d = (fact(card, k) for k in ("planar_slope_deg", "top_above_floor", "width", "depth"))
    if not (slope and top and w and d):
        return None  # a finding, not a pass: nothing to say without a measured flat top and both sides
    if slope["value"] + slope["u"] <= CLIMB_SLOPE_DEG and CLIMB_TOP_M[0] <= top["value"] <= CLIMB_TOP_M[1] and min(w["value"], d["value"]) >= CLIMB_SIDE_M:
        return geo("flat top height", top["value"], top["u"], "m", list(CLIMB_TOP_M), "range", REVIEW,
                   [f"flat top (slope {slope['value']:.0f} deg) {top['value']:.2f} m up, short side {min(w['value'], d['value']):.2f} m: "
                    "a climbable surface (always review)"])
    return None


def foot_surface(mask, depth, K, c2w, up, p0, u_floor=.02):
    """J3a's physical cue, per person detection: the height above the floor of the surface the feet rest on (the median 3D
    point of the mask's lowest pixels) and whether they rest on what is below them (the patch just below the feet is not
    nearer than them by more than max(0.3 m, 10%): else something in front hides the feet). People's footWorld cannot serve:
    live_people puts it on the floor plane by construction. The lowest pixels are feet only when the visible body spans a
    standing height (video.PERSON_HEIGHT_M, 1.3-2.1 m, from those pixels to the mask's p98 height): run mvp-b-judge-me340-001
    read people behind benches at 0.5-1.5 m with the patch below at their own depth. -> {h_m, u_m, contact, span_m,
    feet_visible, bbox (0-1)}; h_m None when the feet are not seen at all."""
    ys, xs = np.nonzero(mask)
    if not len(ys):
        return None
    H, W = mask.shape
    bbox = [round(xs.min() / W, 4), round(ys.min() / H, 4), round((xs.max() + 1) / W, 4), round((ys.max() + 1) / H, 4)]
    if ys.max() >= H - 2:
        return {"h_m": None, "reason": "feet cut by the frame edge", "bbox": bbox}
    low = ys >= ys.max() - max(2, .03 * (ys.max() - ys.min()))
    fy, fx = ys[low], xs[low]
    z = depth[fy, fx]
    ok = z > 0
    if ok.sum() < 3:
        return {"h_m": None, "reason": "no depth at the feet", "bbox": bbox}
    kinv = np.linalg.inv(K)
    cam = (np.c_[fx[ok], fy[ok], np.ones(ok.sum())] @ kinv.T) * z[ok, None]
    world = cam @ c2w[:3, :3].T + c2w[:3, 3]
    h = float(np.median((world - p0) @ up))
    dz = float(abs(np.median((c2w[:3, 3] - world) @ up)))
    zm = depth[ys, xs]
    body = (zm > 0) & (np.abs(zm - np.median(zm[zm > 0])) <= max(.5, .15 * np.median(zm[zm > 0]))) if (zm > 0).any() else zm > 0
    pts = (np.c_[xs[body], ys[body], np.ones(body.sum())] @ kinv.T) * zm[body, None]
    span = float(np.percentile((pts @ c2w[:3, :3].T + c2w[:3, 3] - p0) @ up, 98) - h) if body.sum() >= 10 else None
    u = math.sqrt((.05 * dz) ** 2 + u_floor ** 2 + (SCALE_REL * h) ** 2)
    rows = slice(ys.max() + 1, min(H, ys.max() + 4))
    cols = slice(max(0, int(np.median(fx)) - 6), min(W, int(np.median(fx)) + 7))
    below = depth[rows, cols][(depth[rows, cols] > 0) & ~mask[rows, cols]]
    zf = float(np.median(z[ok]))
    contact = bool(len(below) >= 3 and np.median(below) >= zf - max(.3, .1 * zf))  # the floor in front slopes ~5 cm a row at 4.5 m
    visible = span is not None and PERSON_HEIGHT_M[0] <= span <= PERSON_HEIGHT_M[1]
    return {"h_m": round(h, 3), "u_m": round(u, 3), "contact": contact, "span_m": None if span is None else round(span, 3),
            "feet_visible": bool(visible), "bbox": bbox}


def person_points(card, ctx):
    """A person card's detections: its own 'points', else its track's in ctx['people'] (A's cards carry no points)."""
    if card.get("points"):
        return card["points"]
    track = card.get("track") or (card.get("identity") or {}).get("track") or card["id"].split(":", 1)[-1]
    return next((t["points"] for t in ((ctx.get("people") or {}).get("tracks") or []) if t["id"] == track), [])


def g_j3a(card, ctx, cards):
    """Per detection: FAIL cue when the feet rest on a surface more than 0.30 m +- u up and over an object's footprint + 0.1 m;
    PASS when they rest within 0.30 m -+ u of the floor; over the track: any FAIL fails, PASS needs every detection to pass."""
    from shapely.geometry import Point
    pts = person_points(card, ctx)
    polys = [(c, footprint(c)) for c in cards if c.get("kind") != "person" and c.get("shot") == card.get("shot") and usable(c)]
    frame = shot_of(ctx, card.get("shot")).get("floor_frame")
    per, worst = [], None
    for p in pts:
        fs = p.get("foot_surface") or {}
        if fs.get("h_m") is None:
            per.append(NO_DATA)
            continue
        h, u = fs["h_m"], fs["u_m"]
        if fs.get("feet_visible") is False:  # legs hidden (or crouching): the lowest pixels are not feet
            per.append(NO_DATA)
            continue
        if not fs.get("contact"):
            per.append(REVIEW)
            continue
        r = banded_verdict(h, STAND_M, u, fail_low=False)
        if r == FAIL:
            xy = to_floor(frame, p["xyz"])[0, :2] if frame and p.get("xyz") else None
            over = [c["id"] for c, poly in polys if poly is not None and xy is not None and poly.buffer(.1).contains(Point(xy))]
            r = FAIL if over else REVIEW
        per.append(r)
        if worst is None or h > worst[0]:
            worst = (h, u, p.get("t"))
    if not per:
        return geo("feet above the floor", None, None, "m", STAND_M, "max", NO_DATA, ["no detections"])
    r = FAIL if FAIL in per else PASS if all(v == PASS for v in per) else worst_verdict(dict(enumerate(per)))
    counts = {v: per.count(v) for v in (FAIL, REVIEW, PASS, NO_DATA) if per.count(v)}
    reasons = [f"detections: {counts}"] + ([f"highest feet {worst[0]:.2f} +- {worst[1]:.2f} m at t = {worst[2]} s"] if worst else [])
    if r in (PASS, FAIL) and not FOOT_CUE_VALIDATED:
        return forced(geo("feet above the floor", worst[0] if worst else None, worst[1] if worst else None, "m", STAND_M, "max", r, reasons),
                      [FOOT_CUE_NOTE])
    return geo("feet above the floor", worst[0] if worst else None, worst[1] if worst else None, "m", STAND_M, "max", r, reasons)


def applicable(card):
    """-> [check id] for this card (spec 5.1's 'applies to' column)."""
    if card.get("kind") == "person":
        return ["J3a", "J8"]
    n, mob, out = name_of(card), mobility(card), []
    if is_a(n, STACK):
        out += ["J1", "J2"]
    if is_a(n, CLIMBABLE):
        out.append("J3b")
    if mob == "deformable":
        out.append("J4")
    if mob in ("movable rigid", "fixed"):
        out.append("J5")
    if mob == "movable rigid":
        out.append("J6")
    if is_a(n, LADDER):
        out.append("J7")
    return out


def geometry(check, card, ctx, cards):
    """-> the geometry record, or None when the check does not apply here (J3b without a finding, J5 far from any path, J6
    without a guard)."""
    if check == "J1":
        base = fact(card, "base_above_floor")
        if base is not None and base["value"] - base["u"] > FLOOR_BASE_M:  # on a rack beam or shelf: the stack's own height
            return numeric(card, "height", STACK_MAX_M, "max", "stack height (off the floor: its own height)", "m", bias=TOP_BIAS_M)
        return numeric(card, "top_above_floor", STACK_MAX_M, "max", "top above the floor", "m", bias=TOP_BIAS_M)
    if check == "J7":
        return numeric(card, "principal_axis_tilt_deg", LADDER_TILT_DEG, "max", "principal-axis tilt from vertical", "deg")
    if check == "J5":
        pd, base = path_distance(card, ctx), fact(card, "base_above_floor")
        if pd is None or pd[0] - pd[1] > NEAR_PATH_M or base is not None and base["value"] - base["u"] > FLOOR_BASE_M:
            return None  # no walked path near it, or not on the floor
    if check == "J6":  # only beside a guard: 220 far objects on Sam's Club all read NO_DATA against two rack guards (round 1)
        poly = footprint(card)
        near = [c for c in guards_of(ctx, cards, card.get("shot")) if c is not card and poly is not None and footprint(c) is not None
                and poly.distance(footprint(c)) <= GUARD_NEAR_M]
        if not near:
            return None
    return {"J2": g_j2, "J3a": g_j3a, "J3b": g_j3b, "J4": g_j4, "J5": g_j5, "J6": g_j6}[check](card, ctx, cards)


# ---------- VLM answers and the verdict rule (spec 5.4) ----------

def load_calibration(path=CALIBRATION):
    try:
        raw = path.read_bytes()
    except OSError:
        return {"file_sha256": None, "questions": {}}
    import hashlib
    return {**json.loads(raw), "file_sha256": hashlib.sha256(raw).hexdigest()}


def platt(p, ab):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return 1 / (1 + math.exp(-min(50., max(-50., ab[0] * math.log(p / (1 - p)) + ab[1]))))


def combine(geo_result, vlm, visual_ok=False, hazard_side=False, clear_needed=False):
    """Round-2 verdict table. vlm: 'hazard' (a calibrated 'likely hazard'), 'likely' (at the veto cut, not calibrated), 'clear'
    (a calibrated 'very unlikely'), 'unsure' or None.
      geometry FAIL:        clear -> NEEDS_REVIEW (they disagree); else FAIL
      geometry PASS:        hazard / likely -> NEEDS_REVIEW; clear_needed (J2 on one face) without clear -> NEEDS_REVIEW; else PASS
      geometry NEEDS_REVIEW: hazard + hazard_side (the measured value itself is past the threshold, two view subsets, plausible)
                            -> FAIL; else NEEDS_REVIEW
      no geometry / NO_DATA: hazard / likely -> NEEDS_REVIEW (a hint); else NO_DATA: a VLM answer alone never makes a FAIL or a
                            PASS. visual_ok is kept for the call signature of round 1 and no longer decides."""
    if geo_result == FAIL:
        return REVIEW if vlm == "clear" else FAIL
    if geo_result == PASS:
        if vlm in ("hazard", "likely") or (clear_needed and vlm != "clear"):
            return REVIEW
        return PASS
    if geo_result == REVIEW:
        return FAIL if vlm == "hazard" and hazard_side else REVIEW
    return REVIEW if vlm in ("hazard", "likely") else NO_DATA


# ---------- set-of-marks ----------

def som(frame, polygons_by_mark, subject=1, marks=True, side=448, scale=1.6):
    """BGR frame + {mark: [polygon (source px)]} -> JPEG of 1.6 x the subject's box, long side `side` px; with marks, every
    polygon as a 2 px white-over-black stroke and its number in a black tag (never red: X2's red outlines read as 'fire
    extinguisher')."""
    import cv2
    pts = np.concatenate([np.asarray(p, float).reshape(-1, 2) for p in polygons_by_mark[subject]])
    (x0, y0), (x1, y1) = pts.min(0), pts.max(0)
    H, W = frame.shape[:2]
    half = max(x1 - x0, y1 - y0, 48) * scale / 2
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    a, b, c, d = int(max(0, cx - half)), int(min(W, cx + half)), int(max(0, cy - half)), int(min(H, cy + half))
    s = side / max(b - a, d - c)
    img = cv2.resize(frame[c:d, a:b], (max(1, round((b - a) * s)), max(1, round((d - c) * s))), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    if marks:
        for mark in sorted(polygons_by_mark, key=lambda m: m == subject):  # the subject last, on top
            polys = [np.round((np.asarray(p, float).reshape(-1, 2) - [a, c]) * s).astype(np.int32) for p in polygons_by_mark[mark]]
            me = mark == subject  # the subject a little heavier, so [1] reads first among many outlines
            cv2.polylines(img, polys, True, (0, 0, 0), 5 if me else 4, cv2.LINE_AA)
            cv2.polylines(img, polys, True, (255, 255, 255), 3 if me else 2, cv2.LINE_AA)
            allp = np.concatenate(polys)
            inside = allp[(allp[:, 0] >= 0) & (allp[:, 0] < img.shape[1]) & (allp[:, 1] >= 0) & (allp[:, 1] < img.shape[0])]
            if not len(inside):
                continue
            tx, ty = inside[np.argmin(inside[:, 1])]  # the outline's highest point inside the crop
            label, fs = str(mark), .7 if me else .5
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 1 + me)
            tx, ty = int(min(max(tx, 0), img.shape[1] - tw - 4)), int(min(max(ty, th + 4), img.shape[0] - 1))
            cv2.rectangle(img, (tx, ty - th - 4), (tx + tw + 4, ty), (0, 0, 0), -1)
            cv2.putText(img, label, (tx + 2, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1 + me, cv2.LINE_AA)
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()


def thumb(jpeg, side=320, max_bytes=30000):
    import cv2
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    s = side / max(img.shape[:2])
    img = cv2.resize(img, (max(1, round(img.shape[1] * s)), max(1, round(img.shape[0] * s))), interpolation=cv2.INTER_AREA)
    for q in (80, 65, 50, 35):
        out = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1].tobytes()
        if len(out) <= max_bytes:
            break
    return out


def view_sets(card, ctx=None):
    """-> [[keyframe]] : two sets (independent answers) when two best views are >= 15 deg apart, else one."""
    if card.get("kind") == "person":
        pts = sorted((p for p in person_points(card, ctx or {}) if (p.get("foot_surface") or {}).get("bbox")),
                     key=lambda p: -(lambda b: (b[2] - b[0]) * (b[3] - b[1]))(p["foot_surface"]["bbox"]))
        best = [pts[0]] if pts else []
        best += [p for p in pts[1:] if abs(p["t"] - pts[0]["t"]) >= 1.][:1]  # ponytail: 1 s apart stands in for 15 deg for people
        return [[p["frame"]] for p in best]
    v = card.get("views") or {}
    best = list(v.get("best") or [])
    if len(best) >= 2 and (v.get("azimuth_spread_deg") or 0) >= 15:
        return [[best[0]], [best[1]]]
    return [best[:1]] if best else []


def marks_on(ctx, key, card):
    """{1: the subject's polygons, 2..: the other entities on that keyframe} (source px), or None if the subject is not there."""
    entries = (ctx.get("outlines_by_frame") or {}).get(key) or []
    if card.get("kind") == "person":
        p = next((p for p in person_points(card, ctx) if p["frame"] == key), None)
        b = (p or {}).get("foot_surface", {}).get("bbox")
        if not b:
            return None
        W, H = ctx.get("source_wh", (1280, 720))
        subj = [[[b[0] * W, b[1] * H], [b[2] * W, b[1] * H], [b[2] * W, b[3] * H], [b[0] * W, b[3] * H]]]
    else:
        subj = next((e["polygons"] for e in entries if e["entityId"] == card["id"]), None)
        if not subj:
            return None
    out = {1: subj}
    for e in entries:
        if e["entityId"] != card["id"]:
            out[len(out) + 1] = e["polygons"]
    return out


def state_line(card, row):
    """What the geometry found, in one sentence (spec 5.3), estimated metres rounded to 0.1 m; only facts from two or more view
    subsets on a plausible card (a one-set or inflated box would tell the model something wrong)."""
    if card.get("kind") == "person":
        return "The question is about the person marked [1]."
    bits = []
    firm = lambda k: (lambda f: f if f and (f.get("n_subsets") or 0) >= 2 and not doubts(card) else None)(fact(card, k))  # noqa: E731
    top, base = firm("top_above_floor"), firm("base_above_floor")
    if top:
        bits.append(f"its top is about {top['value']:.1f} m above the floor")
    if base and base["value"] + base["u"] < FLOOR_BASE_M:
        bits.append("it stands on the floor")
    g = row.get("geometry") or {}
    if g.get("result") in (None, NO_DATA) or ONE_SET in g.get("reasons", []) or doubts(card):
        pass
    elif g.get("path") and g.get("object_distance_m") is not None:
        bits.append(f"it is about {g['object_distance_m']:.1f} m from where people walked")
    elif g.get("quantity") == "distance to walked path" and g.get("value") is not None:
        bits.append(f"it is about {g['value']:.1f} m from where people walked")
    return "The question is about object [1]" + (": " + ", ".join(bits) if bits else "") + " (estimated metres)."


def prompt(card, row, q):
    from fast_report import vlm
    question, opts, _ = QUESTIONS[q]
    return vlm.qwen_prompt(" ".join([SCENE, MARKS, state_line(card, row)]), question, opts)


# ---------- the run ----------

def evaluate(cards, ctx):
    """Geometry only (judgements v1): one row per (applicable check, subject)."""
    rows = []
    cards = [follow_name(c) for c in cards]
    for card in cards:
        t = card.get("time") or {}
        for check in applicable(card):
            if check == "J8":
                rows += person_rule_rows(card)
                continue
            g = geometry(check, card, ctx, cards)
            if g is None:
                continue
            title, severity, questions, source = CHECKS[check]
            rows.append({"id": f"{check}:{card['id']}", "check": check, "title": title, "subject": card["id"],
                         "subject_name": name_of(card) or card.get("kind"), "mobility": mobility(card), "object": g.get("path"),
                         "verdict": g["result"], "severity": severity, "geometry": g, "vlm": None, "questions": list(questions),
                         "reasons": list(g["reasons"]), "evidence": [{"key": k[0], "t": key_time(ctx, k[0])} for k in view_sets(card, ctx)],
                         "rule_source": source, "interval_s": [t.get("first_seen_s"), t.get("last_seen_s")]})
    return rows


def frame_at(ctx, key):
    try:
        return ctx["frames"][key]
    except (IndexError, KeyError, TypeError):
        return None


def key_time(ctx, key):
    fps = ctx.get("fps") or 30.
    return round(key / fps, 3)


def person_rule_rows(card):
    """J8: R1-R3 from the people layer, unchanged (scale_gated inside), aggregated over the track's findings."""
    rows, by_rule = [], {}
    for r in card.get("rules") or []:
        by_rule.setdefault(r["rule"], []).append(r)
    for rule, rs in sorted(by_rule.items()):
        v = worst_verdict({i: r["verdict"] for i, r in enumerate(rs)})
        vals = [r["value"] for r in rs if r.get("value") is not None]
        rows.append({"id": f"J8-{rule}:{card['id']}", "check": "J8", "title": f"person rule {rule}", "subject": card["id"],
                     "subject_name": "person", "mobility": "agent", "object": None, "verdict": v, "severity": "major",
                     "geometry": {"quantity": rule, "value": max(vals) if vals else None, "result": v, "scale": "estimated",
                                  "reasons": sorted({r.get("reason") or r["verdict"] for r in rs})}, "vlm": None, "questions": [],
                     "reasons": sorted({f"{r['verdict']}: {r.get('reason') or 'scale not measured (scale_gated)'}" for r in rs}),
                     "evidence": [{"key": r["frame"], "t": r["t"]} for r in rs[:3]], "rule_source": CHECKS["J8"][3],
                     "interval_s": [rs[0]["t"], rs[-1]["t"]]})
    return rows


def layer(rows, cal, extra=None):
    by_object = {}
    for r in rows:
        if r["severity"] != "advisory":
            by_object.setdefault(r["subject"], {})[r["id"]] = r["verdict"]
    counts = {v: sum(r["verdict"] == v for r in rows) for v in (FAIL, REVIEW, PASS, NO_DATA)}
    by_check = {}
    for r in rows:
        by_check.setdefault(r["check"], {v: 0 for v in (FAIL, REVIEW, PASS, NO_DATA)})[r["verdict"]] += 1
    status = {q: (c or {}).get("status") for q, c in (cal.get("questions") or {}).items()}
    return {"schema": "panoptes-judgements-v1", "checks_version": "mvp-1",
            "vlm_note": "per_view: probs = raw option probabilities; p_hazard_raw = the hazard options' sum; calibrated = Platt's "
                        "p(hazard), null when the question is not calibrated (then the VLM is shown, never decides)",
            "calibration": {"file_sha256": cal.get("file_sha256"), "questions": status, "fitted_on": cal.get("fitted_on")},
            "rows": rows, "by_object": {k: worst_verdict(v) for k, v in by_object.items()}, "counts": counts, "by_check": by_check,
            **(extra or {})}


def hazard_cut(cal, decider, q):
    """calibration.json's hazard record for (decider, question), or the uncalibrated veto alone."""
    return ((cal.get("hazard") or {}).get(decider) or {}).get(q) or {"hazard": None, "clear": None, "veto": {"t": .8, "calibrated": False}}


def qwen_p(a):
    """A Qwen yes/no/cannot-tell answer -> p(yes), None when unanswered, letter mass < 0.5 or 'cannot tell' >= 0.5."""
    if not a or a.get("probs") is None or (a.get("mass") or 0) < MIN_MASS or a["probs"][2] >= .5:
        return None
    return float(a["probs"][0])


def run(cards, ctx, writer, clock, vlm_on=True, ask=None, cal=None, pool=None, carried=None, hazard_ask=None):
    """Judgements v1 (geometry) at once; then the hazard judge (round 2): per object with a check, its check's question(s) to
    Gemini on one 2 x 2 evidence image (fast_report.hazard, 10 objects a request, all requests at once, answers waited for up
    to HAZARD_WAIT_S), Qwen's letter probabilities (vlm.options, priority 'judgement') for what Gemini did not answer and for
    people; each answer read at calibration.json's hazard cuts and joined to the geometry by combine(); then v2.
    ask: a stand-in for vlm.submit; hazard_ask: reqs -> {key: Future of the provider output} (hazard.Asker.ask), None = no
    Gemini (Qwen alone); cal: for fast_report/calibration.json (tests). carried: a dict shared by the runs of one analysis: a
    later cards version asks only what is new. pool: a process pool for the rules. Returns counts and times."""
    cal = load_calibration() if cal is None else cal
    outl = ctx.get("outlines") or []
    outl = outl.get("frames", []) if isinstance(outl, dict) else outl  # A passes the outlines analysis, the stub its frames
    ctx = {**ctx, "outlines_by_frame": ctx.get("outlines_by_frame") or {f["sourceFrame"]: f["objects"] for f in outl}}
    with clock.stage("judge.rules", n={"cards": len(cards), "process": pool is not None}):
        lite = {k: v for k, v in ctx.items() if k not in ("frames", "outlines", "outlines_by_frame")}  # what the rules read
        rows = pool.submit(evaluate, cards, lite).result() if pool is not None else evaluate(cards, ctx)
    writer.put("judgements", layer(rows, cal, {"version_of": ctx.get("version_of"), "vlm_answers": False}), None, "estimated+inferred", LABELS)
    clock.mark("judgements_v1_put")
    rec = {"rows": len(rows), "counts_v1": layer(rows, cal)["counts"]}
    if not vlm_on:
        return rec
    from fast_report import hazard, vlm
    ask = ask or vlm.submit
    carried = {} if carried is None else carried
    by_id = {c["id"]: c for c in (follow_name(c) for c in cards)}
    want = {}  # object id -> [question]: the hazard judge's
    for r in rows:
        q = hazard.CHECK_Q.get(r["check"])
        if q and by_id[r["subject"]].get("kind") != "person" and q not in want.setdefault(r["subject"], []):
            want[r["subject"]].append(q)
    blobs, got, t0 = {}, {}, time.perf_counter()
    for oid, qs in want.items():
        for q in qs:
            if ("gemini", oid, q) in carried:
                got[(oid, q)] = {**carried[("gemini", oid, q)], "carried": True}
    todo = {oid: [q for q in qs if (oid, q) not in got] for oid, qs in want.items()}
    todo = {k: v for k, v in todo.items() if v}
    stats = {"objects": len(want), "asked_gemini": 0, "requests": 0, "carried": len(got), "gemini_unanswered": 0, "qwen_questions": 0,
             "prompt_tokens": 0}
    with clock.stage("judge.evidence", n={"objects": len(todo)}):
        def build(oid):
            return oid, hazard.evidence(by_id[oid], ctx, frame_at, marks_on, som)
        with ThreadPoolExecutor(8) as tp:
            ev = dict((oid, x) for oid, x in tp.map(build, list(todo)) if x[0] is not None)
    for oid, (jpg, keys) in ev.items():
        blobs[f"ev-hz-{oid}".replace(":", "_")] = (thumb(jpg), {"mediaType": "image/jpeg", "format": "jpeg", "note": "hazard-judge evidence (2 x 2), 320 px"})
    if hazard_ask is not None and ev:
        with clock.stage("judge.gemini", n={"objects": len(ev)}, sync=False):
            items = [{"id": oid, "name": name_of(by_id[oid]) or "object", "questions": todo[oid], "jpeg": ev[oid][0]} for oid in ev]
            reqs = hazard.batches(items)
            stats.update(asked_gemini=sum(len(i["questions"]) for i in items), requests=len(reqs))
            try:
                futs = hazard_ask(reqs)
            except Exception as error:  # noqa: BLE001  no relay: Qwen answers every question
                futs, stats["gemini_error"] = {}, repr(error)[:200]
            deadline = time.perf_counter() + HAZARD_WAIT_S

            def keep(r, out):  # an answer goes to `carried`: a later judge run of this analysis reads it without asking
                for (oid, q), a in hazard.parse(out.get("output_text") or "{}", r["ids"]).items():
                    carried[("gemini", oid, q)] = {"p": a["p"], "why": a["why"], "decider": "gemini", "keys": ev[oid][1]}
            for r in reqs:
                f = futs.get(r["key"])
                try:
                    out = f.result(timeout=max(.1, deadline - time.perf_counter())) if f is not None else None
                except TimeoutError:  # late: this run falls back to Qwen, the answer still lands in `carried` for the next run
                    f.add_done_callback(lambda f_, r=r: keep(r, f_.result()) if not f_.exception() and f_.result() else None)
                    out, stats["gemini_late"] = None, stats.get("gemini_late", 0) + 1
                except Exception as error:  # noqa: BLE001  failed: its questions fall back to Qwen
                    out, stats["gemini_last_error"] = None, repr(error)[:200]
                if not out:
                    continue
                stats["prompt_tokens"] += ((out.get("usage") or {}).get("prompt_token_count") or 0)
                keep(r, out)
                for (oid, q) in r["ids"]:
                    if ("gemini", oid, q) in carried:
                        got[(oid, q)] = carried[("gemini", oid, q)]
            stats["gemini_unanswered"] = sum(1 for oid, qs in todo.items() for q in qs if (oid, q) not in got and oid in ev)
            stats["gemini_s"] = round(time.perf_counter() - t0, 3)
    # Qwen: what Gemini did not answer (the fallback), and people (J3a's q3 / q5 on the person's views)
    jobs = []
    for row in rows:
        card = by_id[row["subject"]]
        qs = [q for q in ([hazard.CHECK_Q.get(row["check"])] if card.get("kind") != "person" else row["questions"])
              if q and q in QUESTIONS and (card.get("kind") == "person" or (row["subject"], q) not in got)]
        vs = view_sets(card, ctx)[:1]
        for q in qs:
            key = ("qwen", row["id"], q)
            if key in carried:
                jobs.append((row, q, None, carried[key]))
                continue
            for keys in vs:
                marks, frame = marks_on(ctx, keys[0], card), frame_at(ctx, keys[0])
                if marks is None or frame is None:
                    continue
                jobs.append((row, q, keys, ask([som(frame, marks), som(frame, marks, marks=False)], prompt(card, row, q), len(QUESTIONS[q][1]), "judgement")))
    stats["qwen_questions"] = sum(1 for j in jobs if j[2] is not None)
    with clock.stage("judge.qwen", gpu=1, sync=False, n={"questions": stats["qwen_questions"]}):
        qwen = {}
        for row, q, keys, fut in jobs:
            if keys is None:
                qwen[(row["id"], q)] = fut
                continue
            try:
                a = fut.result()
            except Exception as error:  # noqa: BLE001  unanswered: the row stays as the geometry has it
                a = {"probs": None, "mass": 0., "error": repr(error)[:200]}
            qwen[(row["id"], q)] = carried[("qwen", row["id"], q)] = {"p": qwen_p(a), "why": None, "decider": "qwen", "keys": keys,
                                                                        "probs": a.get("probs"), "mass": a.get("mass")}
    names = {"gemini": f"{hazard.GEMINI} stated p(yes) (hazard judge v2)", "qwen": f"{vlm.QWEN} option-letter p(yes) (fallback)"}
    for row in rows:
        card = by_id[row["subject"]]
        q = hazard.CHECK_Q.get(row["check"]) if card.get("kind") != "person" else next(iter(row["questions"]), None)
        a = got.get((row["subject"], q)) if card.get("kind") != "person" else None
        a = a or qwen.get((row["id"], q))
        if not q or not a:
            continue
        cut = hazard_cut(cal, a["decider"], q)
        verdict = hazard.verdict(a["p"], cut)
        g = row["geometry"] or {}
        before = row["verdict"]
        row["verdict"] = combine(before, verdict, hazard_side=bool(g.get("hazard_side")) and row["check"] != "J3a",
                                 clear_needed=bool(g.get("needs_clear_picture")))
        text = hazard.QUESTIONS.get(q) if a["decider"] == "gemini" else QUESTIONS[q][0]
        row["vlm"] = {"question": q, "text": text, "decider": names[a["decider"]], "p_yes": a["p"], "why": a.get("why"), "answer": verdict,
                      "cut": {k: (cut.get(k) or {}).get("t") for k in ("hazard", "clear", "veto")}, "calibrated": bool(cut.get("hazard")),
                      "keys": a.get("keys"), "carried": bool(a.get("carried"))}
        why = f": '{a['why']}'" if a.get("why") else ""
        row["reasons"] = list(g.get("reasons") or row["reasons"]) + [
            f"picture ({a['decider']}): p(yes) {'n/a' if a['p'] is None else format(a['p'], '.2f')} -> {verdict}{why}"]
        if row["verdict"] != before:
            row["reasons"].append({(PASS, REVIEW): "the picture disagrees with the geometry's PASS",
                                   (FAIL, REVIEW): "the picture disagrees with the geometry's FAIL",
                                   (REVIEW, FAIL): "measured value past the threshold and a calibrated 'likely hazard' picture",
                                   (NO_DATA, REVIEW): "not measured; the picture hints at a hazard"}.get((before, row["verdict"]), f"{before} -> {row['verdict']}"))
        name = f"ev-hz-{row['subject']}".replace(":", "_")
        if name in blobs and row["evidence"]:
            row["evidence"][0]["image"] = name
    stats["ask_s"] = round(time.perf_counter() - t0, 3)
    stats["evidence_images"] = len(blobs)
    writer.put("judgements", layer(rows, cal, {"version_of": ctx.get("version_of"), "vlm_answers": True, "vlm": stats}), blobs,
               "estimated+inferred", LABELS)
    clock.mark("judgements_v2_put")
    return {**rec, **stats, "counts_v2": layer(rows, cal)["counts"], "by_check_v2": layer(rows, cal)["by_check"]}


def path_length(xy):
    return float(np.linalg.norm(np.diff(np.asarray(xy, float), axis=0), axis=1).sum()) if len(xy) > 1 else 0.


def context(cam_rows, outline_frames, people, frames, room_points, fps, source_wh=(1280, 720), version_of=None):
    """The ctx contract (spec 9 A) from the core's own pieces: per shot the floor frame (origin = the first camera dropped onto
    the floor, +z = the floor normal, +x = the first camera's forward on the floor), u_pose, room points in that frame; the
    walked paths (camera and every person track that moved at least WALKED_M) as floor xy."""
    shots, walked = [], {}
    for s in cam_rows:
        fl = s.get("floor") or {}
        if not fl.get("normal"):
            continue
        c2w = np.asarray(s["c2w"], float)
        up, p0 = np.asarray(fl["normal"], float), np.asarray(fl["point_m"], float)
        o = c2w[0, :3, 3] - ((c2w[0, :3, 3] - p0) @ up) * up
        fwd = c2w[0, :3, 2] - (c2w[0, :3, 2] @ up) * up
        frame = {"origin_m": o.tolist(), "x": (fwd / np.linalg.norm(fwd)).tolist(), "z": up.tolist()}
        room = room_points.get(s["index"]) if room_points else None
        shots.append({"index": s["index"], "keys": s["keys"], "times": s["times"], "floor_frame": frame, "u_pose_m": .04, "u_floor_m": .02,
                      "room_floor": to_floor(frame, room) if room is not None and len(room) else None})
        cam = to_floor(frame, c2w[:, :3, 3])[:, :2]
        walked[s["index"]] = {"camera": cam} if path_length(cam) >= WALKED_M else {}
    frames_of = {s["index"]: s["floor_frame"] for s in shots}
    for tr in (people or {}).get("tracks", []):
        if tr["shot"] in frames_of and tr["points"]:
            xy = to_floor(frames_of[tr["shot"]], [p["xyz"] for p in tr["points"]])[:, :2]
            if path_length(xy) >= WALKED_M:
                walked[tr["shot"]][f"person:{tr['id']}"] = xy
    return {"shots": shots, "frames": frames, "outlines": outline_frames, "people": people, "walked": walked, "fps": fps,
            "source_wh": tuple(source_wh), "version_of": version_of}


# ---------- calibration (spec 5.3: Platt per question on set d, cross-validated by source clip) ----------

def fit_platt(p, y, iters=100, ridge=1e-3):
    """Logistic regression of y on logit(p) -> (a, b): Newton steps halved until the log-loss drops (plain Newton overshot to
    a = 4e5 on q4, where Qwen's p(yes) spans 1e-9 to 1). Platt's (1999) smoothed targets (N+ + 1) / (N+ + 2) and 1 / (N- + 2)
    keep a finite when a fold separates perfectly."""
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    x = np.c_[np.log(p / (1 - p)), np.ones(len(p))]
    y = np.asarray(y, float)
    t = np.where(y > .5, (y.sum() + 1) / (y.sum() + 2), 1 / ((1 - y).sum() + 2))
    loss = lambda w: float(np.sum(np.logaddexp(0, x @ w) - t * (x @ w)) + ridge * w @ w / 2)  # noqa: E731
    w = np.array([1., 0.])
    for _ in range(iters):
        z = 1 / (1 + np.exp(-np.clip(x @ w, -50, 50)))
        step = np.linalg.solve((x * (z * (1 - z))[:, None]).T @ x + ridge * np.eye(2), x.T @ (z - t) + ridge * w)
        k, now = 1., loss(w)
        while loss(w - k * step) > now and k > 1e-6:
            k /= 2
        w = w - k * step
        if np.abs(k * step).max() < 1e-9:
            break
    return float(w[0]), float(w[1])


def scores(p, y):
    p, y = np.asarray(p, float), np.asarray(y, int)
    brier = float(np.mean((p - y) ** 2))
    bins = np.minimum((p * 10).astype(int), 9)
    ece = float(sum(abs(p[bins == b].mean() - y[bins == b].mean()) * (bins == b).mean() for b in range(10) if (bins == b).any()))
    pos, neg = p[y == 1], p[y == 0]
    auroc = float(((pos[:, None] > neg[None]).sum() + .5 * (pos[:, None] == neg[None]).sum()) / (len(pos) * len(neg))) if len(pos) and len(neg) else None
    return {"brier": round(brier, 4), "ece_10": round(ece, 4), "auroc": None if auroc is None else round(auroc, 4)}


def calibrate(items):
    """items: [{question_id, source, truth, p}] (p = p(hazard) from the deployed prompt) -> per question: a, b (fitted on
    all), leave-one-source-out calibrated scores beside the raw ones, and a status: 'calibrated' only with n >= 30, >= 5 of
    each class and a cross-validated ECE <= 0.10; else 'advisory' (the probabilities are shown, they never decide)."""
    out = {}
    for q in sorted({x["question_id"] for x in items}):
        xs = [x for x in items if x["question_id"] == q and x.get("p") is not None]
        y = np.array([x["truth"] for x in xs])
        p = np.array([x["p"] for x in xs])
        rec = {"n": len(xs), "positives": int(y.sum()), "raw": scores(p, y)}
        if len(xs) < CAL_MIN_N or y.sum() < CAL_MIN_EACH or (1 - y).sum() < CAL_MIN_EACH:
            out[q] = {**rec, "status": f"advisory (n {len(xs)}, {int(y.sum())} positives: too few to calibrate)"}
            continue
        cv = np.zeros(len(xs))
        src = np.array([x["source"] for x in xs])
        for s in set(src):
            train = src != s
            if y[train].sum() == 0 or y[train].sum() == train.sum():  # one class left: the held-out clip keeps its base rate
                cv[~train] = y[train].mean()
                continue
            ab = fit_platt(p[train], y[train])
            cv[~train] = [platt(v, ab) for v in p[~train]]
        a, b = fit_platt(p, y)
        c = scores(cv, y)
        rec.update(a=round(a, 5), b=round(b, 5), calibrated_cv=c, sources=sorted(set(src)))
        rec["status"] = "calibrated" if c["ece_10"] <= CAL_MAX_ECE else f"advisory (cross-validated ECE {c['ece_10']} > {CAL_MAX_ECE})"
        out[q] = rec
    return out


# ---------- self-check ----------

def contract_check():
    """After A merges (fast_report.cards): its synthetic scene through its build(), then evaluate() on the real card shapes.
    Run by hand on A's branch (2026-09-29): J1 PASS 0.59 +- 0.13 m on a two-subset stack; J4 NEEDS_REVIEW on its one-side cable."""
    try:
        from fast_report import cards as A
    except ImportError:
        return "A's cards not on this branch: contract check skipped"
    cams, objs = A._scene()
    K = np.repeat(np.array([[262., 0, 252], [0, 262., 140], [0, 0, 1]])[None], 6, 0)
    shot = {"index": 0, "keys": list(range(0, 36, 6)), "times": [i / 5 for i in range(6)], "c2w": cams, "K": K, "normal": [0., -1., 0.],
            "point_m": [0., 1.6, 3.], "u_floor_m": .02, "mpu": 1., "sharp": np.ones(6), "plumb_deg": 1., "depth": None, "person": None}
    objects = [{"id": f"obj-0-{i}", "shot": 0, "word": w, "votes": {w: 1.}} for i, w in enumerate(["stacked boxes", "box", "cable"])]
    points = [{"world": P, "frame": f, "z": P[:, 2], "sample_ratio": 1., "views": {int(v): [500, 0, 0, 0, 0] for v in np.unique(f)}} for P, f in objs]
    out = A.build({"shots": [shot], "objects": objects, "points": points, "people": None, "calibration": {},
                   "counts": {"obj-0-0": {j: [300, 0] for j in range(6)}, "obj-0-2": {0: [100, 0], 1: [100, 0]}}})
    ctx = {"shots": [{"index": 0, "floor_frame": out["shots"][0]["floor_frame"], "u_pose_m": out["shots"][0]["u_pose_m"]}],
           "outlines": {"frames": []}, "walked": out["walked"]}
    got = {r["id"]: r["verdict"] for r in evaluate(out["cards"], ctx)}
    assert got.get("J1:obj-0-0") == PASS and got.get("J4:obj-0-2") == REVIEW and got.get("J2:obj-0-0") == NO_DATA, got
    return "A's cards: contract ok"


class _Clock:
    def stage(self, *a, **k):
        import contextlib
        return contextlib.nullcontext()

    def mark(self, *_):
        pass


class _Writer:
    def __init__(self):
        self.puts = []

    def put(self, layer, data, blobs=None, status=None, labels=()):
        self.puts.append((layer, data, blobs or {}))


def _val(v, u, n=2, **kw):
    return {"value": v, "u": u, "unit": "m", "n_subsets": n, "scale": "estimated", "parts": {"views": u * .6, "scale": u * .8}, **kw}


def self_check():
    from concurrent.futures import Future
    # the round-2 verdict table, every cell (hazard_side / clear_needed where they matter)
    for g, v, hs, cn, want in [(FAIL, "hazard", 0, 0, FAIL), (FAIL, "clear", 0, 0, REVIEW), (FAIL, "unsure", 0, 0, FAIL), (FAIL, "likely", 0, 0, FAIL),
                               (PASS, "hazard", 0, 0, REVIEW), (PASS, "likely", 0, 0, REVIEW), (PASS, "clear", 0, 0, PASS), (PASS, "unsure", 0, 0, PASS),
                               (PASS, "unsure", 0, 1, REVIEW), (PASS, "clear", 0, 1, PASS), (PASS, None, 0, 0, PASS),
                               (REVIEW, "hazard", 1, 0, FAIL), (REVIEW, "hazard", 0, 0, REVIEW), (REVIEW, "likely", 1, 0, REVIEW),
                               (REVIEW, "clear", 1, 0, REVIEW), (REVIEW, "unsure", 1, 0, REVIEW),
                               (NO_DATA, "hazard", 1, 0, REVIEW), (NO_DATA, "likely", 0, 0, REVIEW), (NO_DATA, "clear", 0, 0, NO_DATA), (NO_DATA, "unsure", 0, 0, NO_DATA)]:
        assert combine(g, v, hazard_side=bool(hs), clear_needed=bool(cn)) == want, (g, v, hs, cn)
    # worst_verdict fixed: one PASS and 59 NO_DATA is NO_DATA, not PASS
    assert worst_verdict({0: PASS, **{i: NO_DATA for i in range(1, 60)}}) == NO_DATA
    assert worst_verdict({0: PASS, 1: REVIEW, 2: NO_DATA}) == REVIEW and worst_verdict({0: PASS, 1: PASS}) == PASS
    assert qwen_p({"probs": [.7, .2, .1], "mass": .9}) == .7 and qwen_p({"probs": [.7, .2, .1], "mass": .3}) is None
    assert qwen_p({"probs": [.1, .3, .6], "mass": 1.}) is None and qwen_p(None) is None  # letter mass < 0.5, cannot tell
    # numeric rule with u, one view subset, implausible size, NO_DATA when not observed
    stack = {"id": "o1", "shot": 0, "identity": {"name": "stacked boxes"}, "physical": {"top_above_floor": _val(3.1, .4)}}
    assert numeric(stack, "top_above_floor", 2.5, "max", "top", "m")["result"] == FAIL
    stack["physical"]["top_above_floor"] = _val(2.7, .4)
    assert numeric(stack, "top_above_floor", 2.5, "max", "top", "m")["result"] == REVIEW
    stack["physical"]["top_above_floor"] = _val(1.2, .4, n=1)
    g = numeric(stack, "top_above_floor", 2.5, "max", "top", "m")
    assert g["result"] == REVIEW and g["before_forced"] == PASS and ONE_SET in g["reasons"]
    stack["physical"].update(top_above_floor=_val(1.2, .4), size_check={"status": "implausible"})
    assert numeric(stack, "top_above_floor", 2.5, "max", "top", "m")["result"] == REVIEW
    assert numeric({"physical": {"height": {"status": "not observed", "reason": "seen from one side"}}}, "height", 1, "max", "h", "m")["result"] == NO_DATA
    assert applicable(stack) == ["J1", "J2", "J3b", "J5", "J6"] and applicable({"identity": {"name": "power cable"}}) == ["J4"]
    # u without its scale part keeps the calibration factor: parts views .6, scale .8 of u=1 -> .6
    assert abs(u_rel({"u": 1., "parts": {"views": .6, "scale": .8}}) - .6) < 1e-9
    # A's shapes: parts holding None, 'at least' (cut by the frame edge) cannot PASS a maximum but can FAIL it,
    # fragmented_support, a person card without points (its track's points come from ctx['people'])
    assert abs(u_rel({"u": .5, "parts": {"views": None, "depth": .3, "scale": .4}}) - .3) < 1e-9
    cut = {"physical": {"top_above_floor": {**_val(1.2, .3), "status": "at least", "reason": "cut by the frame edge"}}}
    assert numeric(cut, "top_above_floor", 2.5, "max", "top", "m")["result"] == REVIEW
    cut["physical"]["top_above_floor"]["value"] = 3.5
    assert numeric(cut, "top_above_floor", 2.5, "max", "top", "m")["result"] == FAIL
    assert doubts({"physical": {"fragmented_support": True}}) and not doubts({"physical": {"size_check": {"status": "plausible"}}})
    assert person_points({"id": "person:0-person-0", "kind": "person"}, {"people": {"tracks": [{"id": "0-person-0", "points": [1]}]}}) == [1]
    # a synthetic shot: the camera walked along +x at y = 0, a person along y = -2; floor seen on y in [-3, 3]; a wall (TSDF
    # points 0.2-1.6 m up) at y = +1.5
    xs, ys = np.meshgrid(np.arange(-1, 9, .04), np.arange(-3, 3, .04))
    floor = np.c_[xs.ravel(), ys.ravel(), np.zeros(xs.size)]
    wall = np.c_[np.repeat(np.arange(-1, 9, .04), 20), np.full(250 * 20, 1.5), np.tile(np.linspace(.2, 1.6, 20), 250)]
    frame = {"origin_m": [0, 0, 0], "x": [1, 0, 0], "z": [0, 0, 1]}

    def shot_ctx(extra=None):
        pts = np.r_[floor, wall] if extra is None else np.r_[floor, wall, extra]
        return {"shots": [{"index": 0, "floor_frame": frame, "u_pose_m": .04, "plumb_u_deg": 1., "room_floor": pts}], "fps": 30.,
                "walked": {0: {"camera": np.c_[np.linspace(0, 8, 40), np.zeros(40)], "person:0-p": np.c_[np.linspace(0, 8, 40), np.full(40, -2.)]}}}
    ctx = shot_ctx()

    def box(i, y0, y1, depth_seen=True, name="box", x0=3, x1=4):
        return {"id": f"obj-0-{i}", "kind": "object", "shot": 0, "identity": {"name": name},
                "physical": {"footprint_xy": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], "position_xy": _val([(x0 + x1) / 2, (y0 + y1) / 2], .05),
                             "base_above_floor": _val(0., .03), "top_above_floor": _val(.8, .1), "height": _val(.8, .1),
                             "width": _val(1., .1), "depth": _val(y1 - y0, .05) if depth_seen else {"status": "not observed"}}}
    # J5: free width 0.4 (camera path to the box) + 1.5 (to the wall) = 1.9 m -> PASS, seen from one side or not (the scan
    # measures the faces seen from the path: round 1's gap asymmetry no longer applies)
    wide = box(0, -.9, -.4, depth_seen=False)
    g = g_j5(wide, ctx, [wide])
    assert g["result"] == PASS and abs(g["value"] - 1.9) < .06 and "this object" in g["bounded_by"] and "surface" in g["bounded_by"], g
    # the person squeezes between two stacks 0.5 m apart (y -1.75 and -2.25): the narrowest bounded path decides -> FAIL
    ctx = shot_ctx()
    p1, q1 = box(0, -1.75, -1.2), box(1, -2.8, -2.25)
    g = g_j5(p1, ctx, [p1, q1])
    assert g["result"] == FAIL and abs(g["value"] - .5) < .06 and g["path"] == "person:0-p" and g["hazard_side"], g
    ctx = shot_ctx()  # without the second stack the person's side runs to the edge of what was seen: a lower bound may PASS
    g = g_j5(p1, ctx, [p1])
    assert g["result"] == PASS and g["lower_bound"] and not g["hazard_side"], g
    ctx = shot_ctx()
    cart = box(0, -.3, .3, x0=6.8, x1=7.6)  # on the camera's own path: the operator's cart, not an obstacle
    g = g_j5(cart, ctx, [cart])
    assert g["result"] == NO_DATA and "cart" in g["reasons"][0], g
    # J2 from the observed faces: a 1.5 m stack whose front face (y = -1.0, facing the camera path) steps 0.2 m out above 0.9 m
    # -> overhang 0.2 -> FAIL; a straight one -> PASS on its overhang and lean, one face seen -> the picture must say clear
    fx, fz = np.meshgrid(np.arange(3.02, 4, .02), np.arange(.07, 1.5, .02))
    face = np.c_[fx.ravel(), np.where(fz.ravel() > .9, -.8, -1.), fz.ravel()]
    stack = {"id": "obj-0-7", "kind": "object", "shot": 0, "identity": {"name": "stacked boxes"},
             "physical": {"footprint_xy": [[3, -1.5], [4, -1.5], [4, -.75], [3, -.75]], "position_xy": _val([3.5, -1.1], .05),
                          "base_above_floor": _val(0., .03), "top_above_floor": _val(1.5, .1), "height": _val(1.5, .1), "width": _val(1., .1),
                          "depth": {"status": "not observed"}, "planar_slope_deg": {**_val(89., 2.), "unit": "deg"}}}
    ctx = shot_ctx(face)
    g = g_j2(stack, ctx, [stack])
    assert g["result"] == FAIL and abs(g["parts"][0]["value"] - .2) < .03, g
    wide_top = np.c_[face[:, 0] + np.where(face[:, 2] > .9, .3, 0.), np.full(len(face), -1.), face[:, 2]]  # the top band 0.3 m wider
    ctx = shot_ctx(wide_top)
    stack["physical"]["footprint_xy"] = [[3, -1.5], [4.4, -1.5], [4.4, -.75], [3, -.75]]
    g = g_j2(stack, ctx, [stack])
    assert g["result"] == REVIEW and g["parts"][0].get("before_forced") == FAIL and not g["hazard_side"], g  # side only: never a FAIL
    stack["physical"]["footprint_xy"] = [[3, -1.5], [4, -1.5], [4, -.75], [3, -.75]]
    ctx = shot_ctx(np.c_[face[:, 0], np.full(len(face), -1.), face[:, 2]])
    g = g_j2(stack, ctx, [stack])
    assert g["result"] == PASS and g["needs_clear_picture"] and len(g["parts"]) == 2 and combine(g["result"], "unsure", clear_needed=True) == REVIEW, g
    low = {**stack, "physical": {**stack["physical"], "height": _val(.3, .05), "top_above_floor": _val(.3, .05)}}  # too low for an overhang
    assert g_j2(low, ctx, [low])["result"] == REVIEW  # the lean passes, the overhang is not measurable: not a PASS
    low["physical"]["planar_slope_deg"] = {"status": "not measurable", "reason": "view sets disagree"}
    assert g_j2(low, ctx, [low])["result"] == NO_DATA
    # J1: a stack reading 2.25 +- 0.1 m clears 2.5 m, but not with the 0.2 m allowance for tops that read low (R3)
    assert numeric(stack | {"physical": {"top_above_floor": _val(2.25, .1)}}, "top_above_floor", 2.5, "max", "top", "m", bias=TOP_BIAS_M)["result"] == REVIEW
    assert numeric(stack | {"physical": {"top_above_floor": _val(2.05, .1)}}, "top_above_floor", 2.5, "max", "top", "m", bias=TOP_BIAS_M)["result"] == PASS
    ctx = shot_ctx()
    # J4: a cable on the floor 0.2 m from the path -> FAIL; 2 m away -> PASS; hanging (base 1 m) -> PASS; a thin hose whose base
    # reads 0.28 +- 0.13 m near the path (round 1's false PASS) -> NEEDS_REVIEW on the hazard side; no base -> NO_DATA
    cable = {"id": "obj-0-5", "kind": "object", "shot": 0, "identity": {"name": "power cable"},
             "physical": {"footprint_xy": [[2, .2], [3, .2], [3, .25], [2, .25]], "position_xy": _val([2.5, .22], .05),
                          "base_above_floor": _val(0., .02), "depth": _val(.05, .02)}}
    assert g_j4(cable, ctx, [cable])["result"] == FAIL
    cable["physical"]["footprint_xy"] = [[2, 2.2], [3, 2.2], [3, 2.25], [2, 2.25]]
    assert g_j4(cable, ctx, [cable])["result"] == PASS
    cable["physical"]["base_above_floor"] = _val(1., .05)
    assert g_j4(cable, ctx, [cable])["result"] == PASS
    cable["physical"].update(footprint_xy=[[2, .44], [3, .44], [3, .48], [2, .48]], base_above_floor=_val(.28, .13))
    g = g_j4(cable, ctx, [cable])
    assert g["result"] == REVIEW and g["hazard_side"], g
    cable["physical"]["base_above_floor"] = {"status": "not observed"}
    assert g_j4(cable, ctx, [cable])["result"] == NO_DATA
    # J3a: feet 0.6 m up with contact over a box -> FAIL; on the floor -> PASS; feet hidden -> NEEDS_REVIEW
    person = {"id": "person:0-person-0", "kind": "person", "shot": 0, "points": [
        {"t": 1., "frame": 30, "xyz": [3.5, -.6, 0], "score": .9, "foot_surface": {"h_m": .6, "u_m": .08, "contact": True, "bbox": [.1, .1, .3, .9]}}]}
    tall = box(0, -.9, -.4)
    global FOOT_CUE_VALIDATED
    assert g_j3a(person, ctx, [person, tall])["result"] == REVIEW  # not validated: the cue asks for review
    FOOT_CUE_VALIDATED = True
    assert g_j3a(person, ctx, [person, tall])["result"] == FAIL
    person["points"][0]["foot_surface"].update(h_m=.02)
    assert g_j3a(person, ctx, [person, tall])["result"] == PASS
    person["points"][0]["foot_surface"].update(contact=False)
    assert g_j3a(person, ctx, [person, tall])["result"] == REVIEW
    # foot_surface on a synthetic depth map: feet on a 0.5 m box (camera 1.6 m up, looking down +z world? here: camera frame =
    # world, up = -y, floor at y = 1.6): the feet's pixels read 0.5 m up with contact
    K = np.array([[200., 0, 100], [0, 200, 100], [0, 0, 1]])
    depth = np.full((200, 200), 4.)
    mask = np.zeros((200, 200), bool)
    mask[74:150, 90:110] = True  # a 1.7 m person standing 4.49 m away on the 0.5 m surface: head at row 74, feet at row 149
    yy = np.arange(200)[:, None]
    depth[:] = np.where(yy > 100, 1.1 * 200 / np.maximum(yy - 100, 1), 8.)  # a floor 1.1 m below the camera (1.6 - 0.5)
    depth[mask] = 4.49
    fs = foot_surface(mask, depth, K, np.eye(4), np.array([0., -1, 0]), np.array([0., 1.6, 0]))
    assert fs["contact"] and fs["feet_visible"] and abs(fs["h_m"] - .5) < .05 and 1.5 < fs["span_m"] < 1.8, fs
    upper = mask.copy()
    upper[112:] = False  # legs hidden: the lowest visible pixels are the waist
    fs = foot_surface(upper, depth, K, np.eye(4), np.array([0., -1, 0]), np.array([0., 1.6, 0]))
    assert not fs["feet_visible"], fs
    person["points"][0]["foot_surface"].update(contact=True, feet_visible=False, h_m=.6)
    assert g_j3a(person, ctx, [person, tall])["result"] == NO_DATA
    FOOT_CUE_VALIDATED = False
    # the whole run with a fake Gemini relay and a fake Qwen: v1 then v2, evidence images, the combined verdicts
    frame_img = np.full((720, 1280, 3), 90, np.uint8)
    cable.update(physical={**cable["physical"], "footprint_xy": [[2, .2], [3, .2], [3, .25], [2, .25]], "base_above_floor": _val(0., .02)},
                 views={"best": [30, 60], "azimuth_spread_deg": 40, "keyframes": [30, 60]}, time={"first_seen_s": 1., "last_seen_s": 2.})
    hose = {**cable, "id": "obj-0-6", "physical": {**cable["physical"], "footprint_xy": [[2, .44], [3, .44], [3, .48], [2, .48]],
                                                    "position_xy": _val([2.5, .46], .05), "base_above_floor": _val(.02, .02)}}
    poly, poly2 = [[[500, 300], [700, 300], [700, 340], [500, 340]]], [[[500, 400], [700, 400], [700, 440], [500, 440]]]
    ctx = {**shot_ctx(), "frames": [frame_img] * 90,
           "outlines": [{"sourceFrame": k, "objects": [{"entityId": cable["id"], "polygons": poly}, {"entityId": hose["id"], "polygons": poly2}]} for k in (30, 60)]}
    cal = {"hazard": {"gemini": {"q1": {"hazard": {"t": .5}, "clear": {"t": .1}, "veto": {"t": .5}}},
                      "qwen": {"q1": {"hazard": {"t": .18}, "clear": {"t": .002}, "veto": {"t": .18}}}}}
    calls, gem = [], {"p": .9, "fail": False, "reqs": 0}

    def fake(jpegs, p, n, priority):
        calls.append((len(jpegs), priority, p))
        f = Future()
        f.set_result({"probs": [.9, .09, .01], "mass": .95, "s": 0., "prompt_tokens": 100})
        return f

    def fake_gemini(reqs):
        out = {}
        for r in reqs:
            gem["reqs"] += 1
            f = Future()
            if gem["fail"]:
                f.set_exception(RuntimeError("relay down"))
            else:
                f.set_result({"status": "completed", "usage": {"prompt_token_count": 1000}, "output_text": json.dumps(
                    {"answers": [{"id": i, "question": q, "p_yes": gem["p"], "why": "lies on the floor"} for i, q in r["ids"]]})})
            out[r["key"]] = f
        return out
    w = _Writer()
    out = run([cable, hose], ctx, w, _Clock(), ask=fake, cal=cal, hazard_ask=fake_gemini)
    assert [x[0] for x in w.puts] == ["judgements", "judgements"] and not w.puts[0][1]["vlm_answers"] and w.puts[1][1]["vlm_answers"]
    rows = {r["subject"]: r for r in w.puts[1][1]["rows"] if r["check"] == "J4"}
    assert not calls and out["requests"] == 1 and out["asked_gemini"] == 2  # Gemini answered: Qwen not asked
    assert rows[cable["id"]]["verdict"] == FAIL and rows[hose["id"]]["verdict"] == FAIL, rows  # hose: REVIEW on the hazard side + hazard
    assert "calibrated 'likely hazard'" in rows[hose["id"]]["reasons"][-1] and rows[hose["id"]]["vlm"]["p_yes"] == .9
    assert rows[cable["id"]]["evidence"][0]["image"] in w.puts[1][2] and all(len(b) <= 30000 for b, _ in w.puts[1][2].values())
    gem.update(p=.05)  # a calibrated 'clear' picture: the geometry FAIL becomes NEEDS_REVIEW (they disagree), the REVIEW stays
    w = _Writer()
    run([cable, hose], ctx, w, _Clock(), ask=fake, cal=cal, hazard_ask=fake_gemini)
    rows = {r["subject"]: r["verdict"] for r in w.puts[1][1]["rows"] if r["check"] == "J4"}
    assert rows == {cable["id"]: REVIEW, hose["id"]: REVIEW}, rows
    gem.update(fail=True)  # the relay fails: Qwen answers (p 0.9 >= its fitted cut 0.18 -> hazard), one question per row
    w = _Writer()
    out = run([cable, hose], ctx, w, _Clock(), ask=fake, cal=cal, hazard_ask=fake_gemini)
    rows = {r["subject"]: r for r in w.puts[1][1]["rows"] if r["check"] == "J4"}
    assert len(calls) == 2 and out["gemini_unanswered"] == 2 and rows[hose["id"]]["verdict"] == FAIL and "fallback" in rows[hose["id"]]["vlm"]["decider"]
    calls.clear()
    gem.update(fail=False, p=.9, reqs=0)  # answers carried between the runs of one analysis: the second run asks nothing new
    carried = {}
    run([cable, hose], ctx, _Writer(), _Clock(), ask=fake, cal=cal, carried=carried, hazard_ask=fake_gemini)
    out = run([cable, hose], ctx, _Writer(), _Clock(), ask=fake, cal=cal, carried=carried, hazard_ask=fake_gemini)
    assert gem["reqs"] == 1 and out["carried"] == 2 and not calls, (gem, out)
    far = {**cable, "id": "obj-0-8", "physical": {**cable["physical"], "footprint_xy": [[2, 2.2], [3, 2.2], [3, 2.25], [2, 2.25]]}}
    ctx["outlines"][0]["objects"].append({"entityId": far["id"], "polygons": poly})
    gem.update(p=.6)  # a PASS (2 m from the path) with a 'likely hazard' picture -> NEEDS_REVIEW (they disagree)
    w = _Writer()
    run([far], ctx, w, _Clock(), ask=fake, cal=cal, hazard_ask=fake_gemini)
    assert next(r for r in w.puts[1][1]["rows"] if r["check"] == "J4")["verdict"] == REVIEW
    jpg = som(frame_img, {1: poly, 2: [[[710, 300], [800, 300], [800, 400]]]})
    import cv2
    img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
    assert max(img.shape[:2]) == 448 and (img[..., 2] > 200).any() and not ((img[..., 2] > 200) & (img[..., 1] < 80)).any()  # white marks, no red
    # calibration: a clean monotone set calibrates; a set without positives stays advisory
    rng = np.random.default_rng(0)
    items = [{"question_id": "qa", "source": f"s{i % 4}", "truth": int(t), "p": float(np.clip(.2 + .6 * t + rng.normal(0, .15), .01, .99))}
             for i, t in enumerate(rng.random(120) < .4)]
    items += [{"question_id": "qb", "source": "s0", "truth": 0, "p": .1}] * 40
    c = calibrate(items)
    assert c["qa"]["status"] == "calibrated" and c["qa"]["a"] > 0 and c["qb"]["status"].startswith("advisory"), c
    contract = contract_check()
    print(f"judge self-check ok ({contract}): verdict table (20 cells), worst_verdict fix, Qwen p rules, numeric rule with u and bias, one "
          "view set, implausible size, NO_DATA paths, J5 aisle (wide / narrow person path / cart), J2 faces (overhang FAIL / one-face PASS), "
          "J1 top allowance, J4 (thin-hose base), J3a + foot surface, run v1 -> v2 (Gemini, clear, relay down -> Qwen, carried, veto), "
          "set-of-marks (white, no red), calibration")


if __name__ == "__main__":
    if "--calibrate" in sys.argv:
        src = Path(sys.argv[sys.argv.index("--calibrate") + 1])
        items = json.loads(src.read_text())
        q = calibrate(items["items"])
        old = json.loads(CALIBRATION.read_text()) if CALIBRATION.exists() else {}
        CALIBRATION.write_text(json.dumps({**old, "schema": "panoptes-calibration-v1", "fitted_on": items["fitted_on"], "decider": items["decider"],
                                           "prompt": items["prompt"], "questions": q,
                                           "note": "Platt per question on the deployed prompt's p(hazard); CV = leave one source clip out; "
                                                   "labels agent-made (X8 set d). Transfer from whole frames to set-of-marks crops is untested."}, indent=1))
        print(json.dumps({k: {kk: v[kk] for kk in ("n", "positives", "status", "raw") if kk in v} | ({"cv": v["calibrated_cv"]} if "calibrated_cv" in v else {})
                          for k, v in q.items()}, indent=1))
    else:
        self_check()
