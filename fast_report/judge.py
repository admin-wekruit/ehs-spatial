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
AISLE_MIN_M = .711  # OSHA 1910.36(g)(2): exit access at least 28 in
NEAR_PATH_M, TRIP_PATH_M, ON_FLOOR_M, FLOOR_BASE_M = 1., .5, .05, .10
WALKED_M = .5  # a camera or person path shorter than this is someone standing, not a walked path
OBSTACLE_H_M = (.1, 1.8)  # J5: a room-mesh point at this height above the floor occupies its cell
STACK_MAX_M, OVERHANG_MAX_M, STACK_TILT_DEG, LADDER_TILT_DEG, GUARD_CLEAR_M, STAND_M = 2.5, .10, 5., 10., .6, .30
CLIMB_TOP_M, CLIMB_SIDE_M, CLIMB_SLOPE_DEG = (.3, 1.2), .3, 10.
HAZARD_P, CLEAR_P, IDENTITY_P, PERSON_SCORE, MIN_MASS, SCREEN_P = .90, .10, .7, .5, .5, .5
CAL_MIN_N, CAL_MIN_EACH, CAL_MAX_ECE = 30, 5, .10  # a question decides only when calibrated on this much, this well (spec 10)
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
    "J0": ("Does object [1] show a safety problem?", ["none visible", "could fall", "trip hazard on the floor",
                                                    "blocks a path or exit", "damaged", "cannot tell"], (1, 2, 3, 4)),
}
CHECKS = {  # id -> title, severity, questions, rule source
    "J0": ("hazard screen (advisory)", "advisory", ("J0",), "MVP: a VLM hint only; it never decides"),
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


def numeric(card, key, threshold, direction, quantity, unit):
    f = fact(card, key)
    if f is None:
        return geo(quantity, None, None, unit, threshold, direction, NO_DATA, [missing(card, key)])
    v, u = float(f["value"]), float(f["u"])
    r = banded_verdict(v, threshold, u, fail_low=direction == "min")
    word = {PASS: "clears", FAIL: "breaks", REVIEW: "straddles"}[r]
    g = geo(quantity, v, u, unit, threshold, direction, r, [f"{quantity} {v:.2f} +- {u:.2f} {unit} {word} {threshold:g} {unit}"],
            f.get("scale"), n_subsets=f.get("n_subsets"))
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
    base, pd = fact(card, "base_above_floor"), path_distance(card, ctx)
    if base is None or pd is None:
        return geo("distance to walked path", None, None, "m", TRIP_PATH_M, "min", NO_DATA,
                   [missing(card, "base_above_floor") if base is None else "no walked path or footprint in this shot"])
    d, u, pid, _, _ = pd
    on = banded_verdict(float(base["value"]), ON_FLOOR_M, float(base["u"]), fail_low=False)  # FAIL here = clearly off the floor
    reasons = [f"base {base['value']:.2f} +- {base['u']:.2f} m above the floor", f"{d:.2f} +- {u:.2f} m from walked path '{pid}'"]
    if on == FAIL:
        r, reasons = PASS, reasons + ["not lying on the floor"]
    elif d - u > NEAR_PATH_M:
        r = PASS
    elif on == PASS and d + u < TRIP_PATH_M:
        r = FAIL
    else:
        r = REVIEW
    g = geo("distance to walked path", d, u, "m", TRIP_PATH_M, "min", r, reasons, path=pid)
    if on == FAIL:  # integration: the PASS rests on the base height, so the geometry line shows that quantity
        g = geo("base above the floor (off the floor: no trip hazard)", float(base["value"]), float(base["u"]), "m", ON_FLOOR_M, "min", r, reasons,
                base.get("scale"), path=pid, path_distance_m=round(float(d), 3))
    gap = ["footprint seen from one side: the gap may be smaller"] if r == PASS and on != FAIL and not observed(card, "depth") else []
    return forced(g, doubts(card) + gap + fact_doubts(base, on, "max") + (one_set(card) if r != PASS or on != FAIL else []))


def floor_grid(shot, cards):
    """J5's 5 cm floor grid for one shot: seen (a room point there), occupied (a room point 0.1-1.8 m up), owner (1 + the index
    of a usable card whose footprint covers the cell, cards on the floor only)."""
    from shapely import contains_xy
    pts = shot.get("room_floor")
    if pts is None and shot.get("room_points") is not None and shot.get("floor_frame"):
        pts = to_floor(shot["floor_frame"], shot["room_points"])
    pts = np.asarray(pts if pts is not None else np.zeros((0, 3)), float)
    if len(pts) < 100:
        return None
    lo = pts[:, :2].min(0) - 1.
    shape = tuple(np.ceil((pts[:, :2].max(0) + 1. - lo) / GRID_M).astype(int))
    ij = ((pts[:, :2] - lo) / GRID_M).astype(int)
    seen, occ, owner = np.zeros(shape, bool), np.zeros(shape, bool), np.zeros(shape, np.int32)
    seen[ij[:, 0], ij[:, 1]] = True
    up = (pts[:, 2] >= OBSTACLE_H_M[0]) & (pts[:, 2] <= OBSTACLE_H_M[1])
    occ[ij[up, 0], ij[up, 1]] = True
    for k, c in enumerate(cards):
        base, poly = fact(c, "base_above_floor"), footprint(c)
        if poly is None or not usable(c) or (base is not None and base["value"] > OBSTACLE_H_M[1]):
            continue
        a0 = np.maximum(((np.asarray(poly.bounds[:2]) - lo) / GRID_M).astype(int), 0)
        a1 = np.minimum(((np.asarray(poly.bounds[2:]) - lo) / GRID_M).astype(int) + 1, np.array(shape) - 1)
        gi, gj = np.meshgrid(np.arange(a0[0], a1[0] + 1), np.arange(a0[1], a1[1] + 1), indexing="ij")
        inside = contains_xy(poly, lo[0] + (gi + .5) * GRID_M, lo[1] + (gj + .5) * GRID_M)
        owner[gi[inside], gj[inside]] = k + 1
    return {"lo": lo, "seen": seen, "occ": occ, "owner": owner}


def scan(grid, p, n):
    """From floor point p along unit n in GRID_M steps -> (free distance, what stopped it: 'card:<k>' | 'mesh' | 'unobserved' |
    'open')."""
    for k in range(1, int(SCAN_M / GRID_M) + 1):
        i, j = ((p + n * k * GRID_M - grid["lo"]) / GRID_M).astype(int)
        if not (0 <= i < grid["seen"].shape[0] and 0 <= j < grid["seen"].shape[1]):
            return (k - .5) * GRID_M, "unobserved"
        if grid["owner"][i, j]:
            return (k - .5) * GRID_M, f"card:{grid['owner'][i, j] - 1}"
        if grid["occ"][i, j]:
            return (k - .5) * GRID_M, "mesh"
        if not grid["seen"][i, j]:
            return (k - .5) * GRID_M, "unobserved"
    return SCAN_M, "open"


def g_j5(card, ctx, cards):
    """Free width across the nearest walked path at the path point nearest the object (spec 5.1's scan rule)."""
    pd, base = path_distance(card, ctx), fact(card, "base_above_floor")
    q = ("free width across the walked path", "m", AISLE_MIN_M, "min")
    if pd is None or base is None:
        return geo(*q[:1], None, None, *q[1:], NO_DATA, ["no walked path, footprint or base in this shot"])
    d, u_d, pid, near, xy = pd
    shot_cards = [c for c in cards if c.get("shot") == card.get("shot")]
    shot = shot_of(ctx, card.get("shot"))
    grid = shot.setdefault("_grid", floor_grid(shot, shot_cards)) if "_grid" not in shot else shot["_grid"]
    if grid is None:
        return geo(*q[:1], None, None, *q[1:], NO_DATA, ["no room points for this shot"])
    k = int(np.argmin(np.linalg.norm(xy - near, axis=1)))
    tangent = xy[min(k + 1, len(xy) - 1)] - xy[max(k - 1, 0)]
    if np.linalg.norm(tangent) < 1e-6:
        return geo(*q[:1], None, None, *q[1:], NO_DATA, ["the walked path does not move here: no direction to scan across"])
    t = tangent / np.linalg.norm(tangent)
    n = np.array([-t[1], t[0]])
    i, j = ((near - grid["lo"]) / GRID_M).astype(int)
    me = shot_cards.index(card)
    if d <= 0 or grid["owner"][i, j] or grid["occ"][i, j]:  # somebody walked through it
        g = geo(*q[:1], 0., None, *q[1:], REVIEW, ["a footprint or the room mesh covers a point people walked through: the footprint "
                                                   "or the path is wrong, or the object moved"], path=pid)
        return forced(g, doubts(card))
    sides = [scan(grid, near, n), scan(grid, near, -n)]
    w = sides[0][0] + sides[1][0]
    bounds = [s for _, s in sides]
    if bounds[0] == bounds[1] and bounds[0].startswith("card:"):  # one footprint on both sides of where somebody walked
        g = geo(q[0], w, None, *q[1:], REVIEW, [f"the walked path runs inside {bounds[0]}'s footprint: the footprint or the path is "
                                                "wrong, or the object moved"], path=pid, bounded_by=bounds)
        return forced(g, doubts(card))
    u_side = max(u_rel(fact(card, "position_xy")) if fact(card, "position_xy") else .1, shot.get("u_pose_m", .04))
    u = math.sqrt(2 * u_side ** 2 + GRID_M ** 2 + (SCALE_REL * w) ** 2)
    lower = any(s in ("unobserved", "open") for s in bounds)
    r = banded_verdict(w, AISLE_MIN_M, u, fail_low=True)
    subject = f"card:{me}" in bounds
    reasons = [f"free width {w:.2f} +- {u:.2f} m across path '{pid}' ({' / '.join(bounds)})" + (" (a lower bound)" if lower else "")]
    if r == FAIL and lower:
        r, reasons = REVIEW, reasons + ["one side ends where nothing was observed: the width may be larger"]
    if r == FAIL and not subject:
        r, reasons = REVIEW, reasons + ["narrow here, but this object does not bound the width"]
    g = geo(q[0], w, u, *q[1:], r, reasons, path=pid, bounded_by=bounds, lower_bound=lower, object_distance_m=round(d, 3))
    others = [shot_cards[int(s[5:])] for s in bounds if s.startswith("card:")]
    gap = ["footprint seen from one side: the gap may be smaller"] if r == PASS and not all(observed(c, "depth") for c in [card, *others]) else []
    return forced(g, doubts(card) + gap + one_set(card) + [x for c in others if c is not card for x in one_set(c)][:1])


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


def g_j2(card, ctx, cards):
    parts = [numeric(card, "overhang", OVERHANG_MAX_M, "max", "overhang", "m")]
    h, w = fact(card, "height"), fact(card, "width")
    tilt = fact(card, "principal_axis_tilt_deg")
    # a principal axis is a lean only on a tall stack (a wide one lies flat: 90 deg), and only below 45 deg: a standing stack
    # whose axis reads 89 deg (Sam's Club, a pallet seen from one side) is measuring its visible face, not a lean
    if h and w and h["value"] > 1.5 * w["value"] and (tilt is None or tilt["value"] < 45.):
        parts.append(numeric(card, "principal_axis_tilt_deg", STACK_TILT_DEG, "max", "principal-axis tilt", "deg"))
    got = [p for p in parts if p["result"] != NO_DATA]
    if not got:
        return geo("overhang", None, None, "m", OVERHANG_MAX_M, "max", NO_DATA, [r for p in parts for r in p["reasons"]])
    top = max(got, key=lambda p: {FAIL: 3, REVIEW: 2, PASS: 0}[p["result"]])
    return {**top, "parts": got} if len(got) > 1 else top


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
            return numeric(card, "height", STACK_MAX_M, "max", "stack height (off the floor: its own height)", "m")
        return numeric(card, "top_above_floor", STACK_MAX_M, "max", "top above the floor", "m")
    if check == "J7":
        return numeric(card, "principal_axis_tilt_deg", LADDER_TILT_DEG, "max", "principal-axis tilt from vertical", "deg")
    if check == "J5":
        pd, base = path_distance(card, ctx), fact(card, "base_above_floor")
        if pd is None or pd[0] - pd[1] > NEAR_PATH_M or base is not None and base["value"] - base["u"] > FLOOR_BASE_M:
            return None  # no walked path near it, or not on the floor
    if check == "J6" and not [c for c in guards_of(ctx, cards, card.get("shot")) if c is not card]:
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


def calibrated(q, p, cal):
    c = (cal.get("questions") or {}).get(q) or {}
    return platt(p, (c["a"], c["b"])) if c.get("status") == "calibrated" else None


def answer(q, views, cal):
    """views: [{"probs", "mass"}] (one per view set) -> ('hazard' | 'clear' | 'unsure', per-view records). hazard: calibrated
    p >= 0.90 in every set; clear: <= 0.10 in every set; anything else (cannot tell >= 0.5, letter mass < 0.5, uncalibrated,
    unanswered) is unsure."""
    _, opts, hz = QUESTIONS[q]
    out, cal_p = [], []
    for v in views:
        rec = {k: v.get(k) for k in ("keys", "probs", "mass", "error")}
        if v.get("probs") is None or (v.get("mass") or 0) < MIN_MASS or ("cannot tell" in opts and v["probs"][opts.index("cannot tell")] >= .5):
            rec["calibrated"] = None
        else:
            p = sum(v["probs"][i] for i in hz)
            rec["p_hazard_raw"] = round(p, 4)
            rec["calibrated"] = None if (c := calibrated(q, p, cal)) is None else round(c, 4)
        cal_p.append(rec["calibrated"])
        out.append(rec)
    if not cal_p or any(p is None for p in cal_p):
        return "unsure", out
    return ("hazard" if min(cal_p) >= HAZARD_P else "clear" if max(cal_p) <= CLEAR_P else "unsure"), out


def combine(geo_result, vlm, visual_ok):
    """Spec 5.4's table. visual_ok: the subject's identity is confirmed for this check and both view sets were asked."""
    if geo_result in (FAIL, PASS, REVIEW):
        if geo_result == FAIL:
            return REVIEW if vlm == "clear" else FAIL
        if geo_result == PASS:
            return REVIEW if vlm == "hazard" else PASS
        return REVIEW
    if vlm == "hazard":
        return FAIL if visual_ok else REVIEW
    return PASS if vlm == "clear" else NO_DATA


def identity_confirmed(card, ctx=None):
    if card.get("kind") == "person":
        return max([p.get("score") or 0 for p in person_points(card, ctx or {})] + [0]) >= PERSON_SCORE
    ident = card.get("identity") or {}
    return bool(ident.get("calibrated")) and (ident.get("confidence") or 0) >= IDENTITY_P


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


def run(cards, ctx, writer, clock, vlm_on=True, ask=None, cal=None, pool=None, carried=None):
    """Judgements v1 (geometry) at once, then every question on every view set through vlm.options (priority 'judgement', J0
    'screen'), then v2. ask: a stand-in for vlm.submit, cal: for fast_report/calibration.json (tests). carried: a dict shared by
    the runs of one analysis, (row id, question, keyframes) -> answer: a later cards version asks only what is new (on Sam's
    Club the v3 run re-asked every question beside the identity pass and slowed it 1.3x). pool: a process pool for
    the rules (in the core's main process they took 0.5-10.9 s for the same 0.2 s of work, by what else held the GIL).
    Returns counts and times."""
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
    from fast_report import vlm
    ask = ask or vlm.submit
    by_id = {c["id"]: c for c in cards}
    screen = [c for c in cards if c.get("kind") != "person" and identity_confirmed(c) and len((c.get("views") or {}).get("keyframes") or []) >= 3]
    for c in screen:  # J0: asked, and a row only when it flags something
        rows.append({"id": f"J0:{c['id']}", "check": "J0", "title": CHECKS["J0"][0], "subject": c["id"], "subject_name": name_of(c),
                     "mobility": mobility(c), "object": None, "verdict": NO_DATA, "severity": "advisory", "geometry": None, "vlm": None,
                     "questions": ["J0"], "reasons": [], "evidence": [{"key": k[0], "t": key_time(ctx, k[0])} for k in view_sets(c)],
                     "rule_source": CHECKS["J0"][3], "interval_s": [(c.get("time") or {}).get("first_seen_s"), (c.get("time") or {}).get("last_seen_s")]})
    blobs, jobs = {}, []

    def images(row):
        card = by_id[row["subject"]]
        out = []
        for keys in view_sets(card, ctx):
            k = keys[0]
            marks, frame = marks_on(ctx, k, card), frame_at(ctx, k)
            if marks is None or frame is None:
                continue
            out.append((keys, som(frame, marks), som(frame, marks, marks=False)))
        return out
    with clock.stage("judge.som", n={"rows": len(rows)}):
        with ThreadPoolExecutor(8) as pool:
            views = list(pool.map(images, rows))
    t_ask = time.perf_counter()
    with clock.stage("judge.vlm", gpu=1, sync=False):  # vLLM's own process on GPU 1
        for row, vs in zip(rows, views):
            for i, (keys, marked, _plain) in enumerate(vs):
                name = f"ev-{row['check']}-{row['subject']}-{i}".replace(":", "_")
                blobs[name] = (thumb(marked), {"mediaType": "image/jpeg", "format": "jpeg", "note": "set-of-marks evidence, 320 px"})
                ev = next((e for e in row["evidence"] if e["key"] == keys[0]), None)
                if ev is not None:
                    ev["image"] = name
            for q in row["questions"]:
                card = by_id[row["subject"]]
                p = prompt(card, row, q)
                n = len(QUESTIONS[q][1])
                decides = ((cal.get("questions") or {}).get(q) or {}).get("status") == "calibrated"
                for keys, marked, plain in vs if decides else vs[:1]:  # an advisory question never decides: one view set shows it
                    key = (row["id"], q, tuple(keys))
                    if carried is not None and key in carried:
                        fut = Future()
                        fut.set_result({**carried[key], "carried": True})
                    else:
                        fut = ask([marked, plain], p, n, "screen" if q == "J0" else "judgement")
                    jobs.append((row, q, keys, fut))
        answers = {}
        for row, q, keys, fut in jobs:
            try:
                a = {**fut.result(), "keys": keys}
                if carried is not None:
                    carried[(row["id"], q, tuple(keys))] = {k: v for k, v in a.items() if k not in ("keys", "carried")}
            except Exception as error:  # noqa: BLE001  unanswered: the row stays 'unsure'
                a = {"keys": keys, "probs": None, "mass": 0., "error": repr(error)[:200]}
            answers.setdefault(row["id"], {}).setdefault(q, []).append(a)
    ask_s = round(time.perf_counter() - t_ask, 3)
    decider = f"{vlm.QWEN} option-letter log-probs"
    for row in rows:
        got = answers.get(row["id"])
        if not got:
            continue
        card = by_id[row["subject"]]
        per_q = {q: answer(q, a, cal) for q, a in got.items()}
        if row["check"] == "J0":
            flagged = [(QUESTIONS["J0"][1][max(QUESTIONS["J0"][2], key=lambda i: a["probs"][i])], sum(a["probs"][i] for i in QUESTIONS["J0"][2]))
                       for a in got["J0"] if a.get("probs")]
            hits = [f for f in flagged if f[1] >= SCREEN_P]
            row["vlm"] = {"question": "J0", "text": QUESTIONS["J0"][0], "options": QUESTIONS["J0"][1], "decider": decider, "per_view": per_q["J0"][1],
                          "answer": "advisory", "also": []}
            if hits:
                row["verdict"] = REVIEW
                row["reasons"] = [f"screen hint (uncalibrated, advisory): '{hits[0][0]}', p = {hits[0][1]:.2f}"]
            continue
        verdicts = [a for a, _ in per_q.values()]
        vlm_ans = "hazard" if "hazard" in verdicts else "clear" if verdicts and all(v == "clear" for v in verdicts) else "unsure"
        both = all(len(a) >= 2 for a in got.values())
        before = row["verdict"]
        row["verdict"] = combine(before, vlm_ans, identity_confirmed(card, ctx) and both)
        asked = [{"question": q, "text": QUESTIONS[q][0], "options": QUESTIONS[q][1], "per_view": pv, "answer": a,
                  "calibration": ((cal.get("questions") or {}).get(q) or {}).get("status") or "none"} for q, (a, pv) in per_q.items()]
        row["vlm"] = {**asked[0], "decider": decider, "answer": vlm_ans, "also": asked[1:]}  # spec 5.6's shape; J3a asks two questions
        raw = [f"{q}: " + ", ".join(f"p(hazard) raw {v.get('p_hazard_raw')}" + (f" -> {v['calibrated']:.2f} calibrated" if v.get("calibrated") is not None else " (uncalibrated)")
                                    for v in pv) for q, (_, pv) in per_q.items()]
        row["reasons"] = row["geometry"]["reasons"] + [f"picture: {vlm_ans} ({'; '.join(raw)})"] + (
            [f"geometry {before} and picture {vlm_ans} disagree"] if row["verdict"] == REVIEW and before in (PASS, FAIL) else [])
    rows = [r for r in rows if r["check"] != "J0" or r["verdict"] == REVIEW]
    stats = {"questions": len(jobs), "carried": sum(1 for *_, f in jobs if (f.result() if f.done() and not f.exception() else {}).get("carried")), "ask_s": ask_s, "unanswered": sum(1 for r in answers.values() for a in r.values() for v in a if v.get("probs") is None),
             "prompt_tokens": sum(v.get("prompt_tokens") or 0 for r in answers.values() for a in r.values() for v in a),
             "evidence_images": len(blobs), "screened": len(screen)}
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
    # the verdict table (spec 5.4), every cell
    for g, v, ok, want in [(FAIL, "hazard", 0, FAIL), (FAIL, "clear", 0, REVIEW), (FAIL, "unsure", 0, FAIL),
                           (PASS, "hazard", 0, REVIEW), (PASS, "clear", 0, PASS), (PASS, "unsure", 0, PASS),
                           (REVIEW, "hazard", 1, REVIEW), (REVIEW, "clear", 1, REVIEW), (REVIEW, "unsure", 1, REVIEW),
                           (NO_DATA, "hazard", 1, FAIL), (NO_DATA, "hazard", 0, REVIEW), (NO_DATA, "clear", 0, PASS), (NO_DATA, "unsure", 1, NO_DATA)]:
        assert combine(g, v, bool(ok)) == want, (g, v, ok)
    # worst_verdict fixed: one PASS and 59 NO_DATA is NO_DATA, not PASS
    assert worst_verdict({0: PASS, **{i: NO_DATA for i in range(1, 60)}}) == NO_DATA
    assert worst_verdict({0: PASS, 1: REVIEW, 2: NO_DATA}) == REVIEW and worst_verdict({0: PASS, 1: PASS}) == PASS
    # VLM answers: calibrated hazard needs >= 0.9 in every set; uncalibrated / low mass / cannot tell -> unsure
    cal = {"questions": {"q1": {"status": "calibrated", "a": 1., "b": 0.}, "q2": {"status": "advisory (too few)"}}}
    assert answer("q1", [{"probs": [.95, .04, .01], "mass": .9}, {"probs": [.97, .02, .01], "mass": .9}], cal)[0] == "hazard"
    assert answer("q1", [{"probs": [.95, .04, .01], "mass": .9}, {"probs": [.5, .49, .01], "mass": .9}], cal)[0] == "unsure"
    assert answer("q1", [{"probs": [.02, .97, .01], "mass": .9}], cal)[0] == "clear"
    assert answer("q1", [{"probs": [.02, .97, .01], "mass": .3}], cal)[0] == "unsure"  # letter mass < 0.5
    assert answer("q1", [{"probs": [.02, .38, .6], "mass": .9}], cal)[0] == "unsure"  # cannot tell
    assert answer("q2", [{"probs": [.99, 0, .01], "mass": 1.}], cal)[0] == "unsure"  # uncalibrated never decides
    assert answer("hard hat", [{"probs": [.01, .98, .01], "mass": 1.}], {"questions": {"hard hat": {"status": "calibrated", "a": 1, "b": 0}}})[0] == "hazard"
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
    # a synthetic shot: walked along +x at y = 0; floor seen on y in [-3, 3]; a wall (mesh) at y = +1.5; a box at y in [-0.9, -0.4]
    xs, ys = np.meshgrid(np.arange(-1, 9, .04), np.arange(-3, 3, .04))
    floor = np.c_[xs.ravel(), ys.ravel(), np.zeros(xs.size)]
    wall = np.c_[np.repeat(np.arange(-1, 9, .04), 20), np.full(250 * 20, 1.5), np.tile(np.linspace(.2, 1.6, 20), 250)]
    frame = {"origin_m": [0, 0, 0], "x": [1, 0, 0], "z": [0, 0, 1]}
    ctx = {"shots": [{"index": 0, "floor_frame": frame, "u_pose_m": .04, "room_floor": np.r_[floor, wall]}],
           "walked": {0: {"camera": np.c_[np.linspace(0, 8, 40), np.zeros(40)]}}, "fps": 30.}

    def box(i, y0, y1, depth_seen=True, name="box"):
        return {"id": f"obj-0-{i}", "kind": "object", "shot": 0, "identity": {"name": name},
                "physical": {"footprint_xy": [[3, y0], [4, y0], [4, y1], [3, y1]], "position_xy": _val([3.5, (y0 + y1) / 2], .05),
                             "base_above_floor": _val(0., .03), "top_above_floor": _val(.8, .1), "height": _val(.8, .1),
                             "width": _val(1., .1), "depth": _val(y1 - y0, .05) if depth_seen else {"status": "not observed"}}}
    wide = box(0, -.9, -.4)  # free width = 0.4 (to the box) + 1.5 (to the wall) = 1.9 m -> PASS
    g = g_j5(wide, ctx, [wide])
    assert g["result"] == PASS and abs(g["value"] - 1.9) < .06 and "mesh" in g["bounded_by"] and "card:0" in g["bounded_by"], g
    ctx["shots"][0].pop("_grid")
    near = box(0, -.45, -.2)  # 0.2 + 1.5 = 1.7 m still wide; now a second box at y in [0.3, 0.9] narrows it to 0.2 + 0.3 = 0.5 m
    other = box(1, .3, .9)
    g = g_j5(near, ctx, [near, other])
    assert g["result"] == FAIL and abs(g["value"] - .5) < .06, g
    ctx["shots"][0].pop("_grid")
    near1 = box(0, -.45, -.2)
    near1["physical"]["position_xy"]["n_subsets"] = 1  # one view set: the same narrow aisle can only ask for review
    assert g_j5(near1, ctx, [near1, other])["result"] == REVIEW
    ctx["shots"][0].pop("_grid")
    on_path = box(0, -.2, .2)  # the camera walked through its footprint
    assert g_j5(on_path, ctx, [on_path])["result"] == REVIEW
    ctx["shots"][0].pop("_grid")
    one_side = box(0, -.9, -.4, depth_seen=False)  # gap asymmetry: a PASS on a footprint seen from one side is NEEDS_REVIEW
    g = g_j5(one_side, ctx, [one_side])
    assert g["result"] == REVIEW and g["before_forced"] == PASS, g
    ctx["shots"][0].pop("_grid")
    # J4: a cable on the floor 0.2 m from the path -> FAIL; 2 m away -> PASS; hanging (base 1 m) -> PASS; no base -> NO_DATA
    cable = {"id": "obj-0-5", "kind": "object", "shot": 0, "identity": {"name": "power cable"},
             "physical": {"footprint_xy": [[2, .2], [3, .2], [3, .25], [2, .25]], "position_xy": _val([2.5, .22], .05),
                          "base_above_floor": _val(0., .02), "depth": _val(.05, .02)}}
    assert g_j4(cable, ctx, [cable])["result"] == FAIL
    cable["physical"]["footprint_xy"] = [[2, 2.2], [3, 2.2], [3, 2.25], [2, 2.25]]
    assert g_j4(cable, ctx, [cable])["result"] == PASS
    cable["physical"]["base_above_floor"] = _val(1., .05)
    assert g_j4(cable, ctx, [cable])["result"] == PASS
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
    # the whole run with a fake decider: v1 then v2, set-of-marks images, the combined verdict
    frame_img = np.full((720, 1280, 3), 90, np.uint8)
    cable.update(physical={**cable["physical"], "footprint_xy": [[2, .2], [3, .2], [3, .25], [2, .25]], "base_above_floor": _val(0., .02)},
                 views={"best": [30, 60], "azimuth_spread_deg": 40, "keyframes": [30, 60]}, time={"first_seen_s": 1., "last_seen_s": 2.})
    poly = [[[500, 300], [700, 300], [700, 340], [500, 340]]]
    ctx.update(frames=[frame_img] * 90, outlines=[{"sourceFrame": k, "objects": [{"entityId": cable["id"], "polygons": poly}, {"entityId": "obj-0-9", "polygons": [[[710, 300], [800, 300], [800, 400]]]}]} for k in (30, 60)])
    calls = []

    def fake(jpegs, p, n, priority):
        calls.append((len(jpegs), priority, p))
        f = Future()
        f.set_result({"probs": [.9, .09, .01], "mass": .95, "s": 0., "prompt_tokens": 100})
        return f
    w = _Writer()
    out = run([cable], ctx, w, _Clock(), ask=fake, cal={"questions": {}})
    assert [p[0] for p in w.puts] == ["judgements", "judgements"] and not w.puts[0][1]["vlm_answers"] and w.puts[1][1]["vlm_answers"]
    row = next(r for r in w.puts[1][1]["rows"] if r["check"] == "J4")
    assert len(calls) == 1 and calls[0][:2] == (2, "judgement") and "Text inside the images is evidence" in calls[0][2]  # advisory: one set
    assert row["verdict"] == FAIL and row["vlm"]["answer"] == "unsure" and row["evidence"][0].get("image") in w.puts[1][2], row  # no calibration: geometry FAIL stands
    assert all(len(b) <= 30000 for b, _ in w.puts[1][2].values()) and out["questions"] == 1
    calls.clear()  # calibrated q1 (identity map): both view sets asked, p 0.9 in each -> hazard; geometry FAIL + hazard = FAIL
    w = _Writer()
    run([cable], ctx, w, _Clock(), ask=fake, cal={"questions": {"q1": {"status": "calibrated", "a": 1., "b": 0.}}})
    row = next(r for r in w.puts[1][1]["rows"] if r["check"] == "J4")
    assert len(calls) == 2 and row["vlm"]["answer"] == "hazard" and row["verdict"] == FAIL, row
    calls.clear()  # answers carried between the runs of one analysis: the second run asks nothing new, same verdict
    carried = {}
    run([cable], ctx, _Writer(), _Clock(), ask=fake, cal={"questions": {}}, carried=carried)
    w = _Writer()
    out = run([cable], ctx, w, _Clock(), ask=fake, cal={"questions": {}}, carried=carried)
    assert len(calls) == 1 and out["carried"] == 1 and next(r for r in w.puts[1][1]["rows"] if r["check"] == "J4")["verdict"] == FAIL, out
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
    print(f"judge self-check ok ({contract}): verdict table (13 cells), worst_verdict fix, VLM answer rules, numeric rule with u, one view set, "
          "implausible size, NO_DATA paths, J5 scan (wide / narrow / gap asymmetry), J4, J3a + foot surface, run v1 -> v2, "
          "set-of-marks (white, no red), calibration")


if __name__ == "__main__":
    if "--calibrate" in sys.argv:
        src = Path(sys.argv[sys.argv.index("--calibrate") + 1])
        items = json.loads(src.read_text())
        q = calibrate(items["items"])
        CALIBRATION.write_text(json.dumps({"schema": "panoptes-calibration-v1", "fitted_on": items["fitted_on"], "decider": items["decider"],
                                           "prompt": items["prompt"], "questions": q,
                                           "note": "Platt per question on the deployed prompt's p(hazard); CV = leave one source clip out; "
                                                   "labels agent-made (X8 set d). Transfer from whole frames to set-of-marks crops is untested."}, indent=1))
        print(json.dumps({k: {kk: v[kk] for kk in ("n", "positives", "status", "raw") if kk in v} | ({"cv": v["calibrated_cv"]} if "calibrated_cv" in v else {})
                          for k, v in q.items()}, indent=1))
    else:
        self_check()
