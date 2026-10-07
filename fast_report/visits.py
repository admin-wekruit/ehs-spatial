"""Cross-visit analysis (r5b): a new visit's video registered into the first visit's site map, its objects matched to the
site map's by position + type + DINOv2 appearance, identities and display models carried over, and the differences as a
`visits` layer on the new visit's report: moved / new / missing / changed height or angle, each with evidence frames from
both visits and +-u. A place the other visit never saw clearly is 'not observed', never 'missing' or 'new'.

A visit is one finished report (the layers Volume or a local mirror: same layout). Per shot of the new visit (B):
  1. retrieval: DINOv2-L global descriptors of every keyframe of both visits; the site map (A) shot most B frames look like;
  2. registration (register_cut_shot's method): B_FRAMES B keyframes and their A_PER_B most similar A keyframes through
     DA3-GIANT any-view in one forward; DA3's A cameras are carried onto A's own (rotation average, scale from the depth
     ratio against A's pick depth, translation from the centres) and the register_cut_shot gate refuses the registration
     when they do not agree (centre residual <= 0.1 of their spread, rotation residual <= 3 deg); B's own cameras then give
     the similarity B shot -> A shot the same way (a second gate), and the two floors are compared: within FLOOR_TILT_MAX_DEG
     and FLOOR_DZ_MAX_M the similarity is snapped to the floors (a turn about the vertical, a floor shift, a scale), so heights
     are comparable; else the registration is refused ('floors disagree');
  3. matching: every card of the two shots in A's floor frame; a pair is possible within K_SIGMA of the pooled position u
     (cards' geometry u, registration u) or on overlapping footprints, with overlapping height ranges; cost = distance / u +
     appearance (1 - cos of the naming pass's DINOv2 vectors) + a type mismatch (two specific families that differ); one to one;
  4. differences: a matched pair is 'static' unless its top / base (both measured, not bounds) differ by more than
     max(K_CHANGE u, MIN_DZ_M) ('changed height') or its tilt / slope by more than max(K_CHANGE u, MIN_DDEG) ('changed angle');
     an unmatched object's points (its pick masks lifted with its depth) go through the other visit's see-through test
     (timeline.place with the registration's pose u): 'free' -> 'missing' (A's object) or 'new' (B's), 'occupied' -> present but
     delineated otherwise (no change), anything else -> 'not observed'. A missing object and a new one of a compatible type,
     appearance cos >= the pair's negative 99th percentile and size within 2x are one object that 'moved';
  5. carry-over: every B card gets a site id (A's card id when matched or moved), A's specific name / type when B has none,
     and A's display model placed in B's frame; A's missing objects keep their model there as a ghost.

    python -m fast_report.visits --self-check
"""
import json
import time
from pathlib import Path

import numpy as np

from fast_report import cards, ondemand, timeline

SCHEMA = "panoptes-visits-v1"
MAX_CENTRE_RESIDUAL_SHARE, MAX_ROTATION_RESIDUAL_DEG = .1, 3.  # scripts/register_cut_shot.py's gate
SPREAD_FLOOR_M = .25       # a centre residual share is taken over at least this spread (views close together)
MAX_DEPTH_RATIO_SPREAD = .15  # per-frame depth ratio (own depth / DA3's) spread (MAD / median) a registration tolerates
FLOOR_TILT_MAX_DEG, FLOOR_DZ_MAX_M = 5., .2
B_FRAMES, A_PER_B, MIN_GAP_KEYS, MIN_KEYS = 10, 2, 4, 10
A_SPREAD_KEYS, A_MAX = 5, 20  # the site map's views get their neighbours +-5 keyframes (1 s) up to 20: the fit's cameras spread out
SCALES = ("depth", "centres")  # the similarity's scale: the median depth ratio, or the camera centres (rotation fixed); of those that
# pass the gate, the one whose floor lands nearer the site map's (r5b dev, GT: the depth ratio read 7-11 % small on 5 of 8 TUM pairs,
# the centres on the other 3; the floor shift before snapping picked the better one on 7 of 8)
DESC_SIDE = 224
K_SIGMA, K_CHANGE, MIN_DZ_M, MIN_DDEG = 3., 2., .10, 5.
POSE_M, NEAR_SIZE, NEAR_MIN_M, PART_SHARE, Z_SHARE, Z_TOL_M = .04, .5, .15, .5, .3, .05
COS_MIN, COS_TYPES = .5, .7  # the same object looks alike across visits: below COS_MIN never one object; two specific types that differ need COS_TYPES
TAU_MOVE_FALLBACK, NEG_MIN_M, SIZE_RATIO = .8, 3., 2.
PLACE_KEYS, OBJ_POINTS, OBJ_VIEWS = 60, 400, 4
# the see-through test across visits (timeline.place's rule with these three): r5b dev on 4020 TUM cards against the GT oracle (GT
# depth, GT poses): X6's 0.2 / 2 px / 0.6 called 12 places free (10 right, recall 9 % of GT's 116); 0.1 / 1 / 0.5 called 22 (18 right,
# recall 16 %) but on our own clip (two overlapping windows of one Sam's Club walk: nothing moved but the filmer's cart) its claims
# went from 3 to 10, the new ones all on static shelves (by eye). X6's rule stays: a false change costs more than a missed one
REL_MARGIN, NEIGH, FREE_SHARE = timeline.REL_MARGIN, timeline.NEIGH, timeline.FREE_SHARE
GENERIC = (None, "misc", "material")  # families that say little (the review: 'material / part' is a catch-all)
TILE_W, TILE_H = 480, 270


# ---------------------------------------------------------------- geometry

def rot_avg(Ra, Rb):
    """R minimising sum |Ra_i - R Rb_i|_F (Ra_i ~ R Rb_i)."""
    u, _, vt = np.linalg.svd(sum(a @ b.T for a, b in zip(Ra, Rb)))
    return u @ np.diag([1, 1, np.sign(np.linalg.det(u @ vt))]) @ vt


def angle_deg(Ra, Rb):
    return float(np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1))))


def fit(known, pred, ratio, scale="depth", trim=True):
    """Similarity carrying `pred` cameras (c2w) onto `known` ones: rotation average, scale = median depth ratio (known depth /
    pred depth, per frame medians given) or from the centres (scale='centres'), translation from the centres. -> {s, R, t, residuals}.
    trim: once fitted, views whose centre residual is over TRIM_K x the median (and TRIM_MIN_M) are left out and the fit is made
    again on the rest (one view DA3 placed wrong must not pull the similarity); at least TRIM_KEEP of the views stay."""
    f = _fit(known, pred, ratio, scale)
    if not trim or f["centre_residual_m"] is None or len(known) < 6:
        return f
    res = np.linalg.norm(known[:, :3, 3] - (f["s"] * pred[:, :3, 3] @ f["R"].T + f["t"]), axis=1)
    keep = res <= max(TRIM_K * float(np.median(res)), TRIM_MIN_M)
    if keep.all() or keep.sum() < max(4, TRIM_KEEP * len(keep)):
        return f
    g = _fit(known[keep], pred[keep], np.asarray(ratio)[keep] if len(ratio) == len(keep) else ratio, scale)
    return {**g, "views_used": int(keep.sum()), "views": int(len(keep))}


TRIM_K, TRIM_MIN_M, TRIM_KEEP = 3., .05, .75


def _fit(known, pred, ratio, scale="depth"):
    R = rot_avg(known[:, :3, :3], pred[:, :3, :3])
    ratio = np.asarray(ratio, float)[np.isfinite(ratio)]
    if not len(ratio):
        return {"s": float("nan"), "R": R, "t": np.zeros(3), "centre_residual_m": None, "rotation_residual_deg": None, "spread_m": None,
                "centre_residual_share": None, "ratio_spread": None, "reason": "no depth overlap between the report's depth and DA3's"}
    kc, pc = known[:, :3, 3] - known[:, :3, 3].mean(0), (pred[:, :3, 3] - pred[:, :3, 3].mean(0)) @ R.T
    s_centres = float((kc * pc).sum() / max((pc ** 2).sum(), 1e-12))
    s = float(np.median(ratio)) if scale == "depth" else s_centres
    t = (known[:, :3, 3] - s * pred[:, :3, 3] @ R.T).mean(0)
    cres = np.linalg.norm(known[:, :3, 3] - (s * pred[:, :3, 3] @ R.T + t), axis=1)
    rres = [angle_deg(k[:3, :3], R @ p[:3, :3]) for k, p in zip(known, pred)]
    spread = float(np.linalg.norm(known[:, :3, 3] - known[:, :3, 3].mean(0), axis=1).mean())
    return {"s": s, "R": R, "t": t, "centre_residual_m": float(np.median(cres)), "rotation_residual_deg": float(np.median(rres)),
            "spread_m": spread, "centre_residual_share": float(np.median(cres) / max(spread, SPREAD_FLOOR_M)),
            "ratio_spread": float(np.median(np.abs(ratio - np.median(ratio))) / max(float(np.median(ratio)), 1e-9)),
            "s_depth": float(np.median(ratio)), "s_centres": s_centres}


def passes(f):
    return (f["centre_residual_share"] is not None and f["centre_residual_share"] <= MAX_CENTRE_RESIDUAL_SHARE
            and f["rotation_residual_deg"] <= MAX_ROTATION_RESIDUAL_DEG and f["ratio_spread"] <= MAX_DEPTH_RATIO_SPREAD)


def _num(f):
    """A fit's numbers for the layer (JSON: no NaN)."""
    return {k: (v if isinstance(v, int) else round(v, 4) if isinstance(v, float) and np.isfinite(v) else None)
            for k, v in f.items() if k not in ("R", "t", "reason")} | (
        {"reason": f["reason"]} if f.get("reason") else {})


def move(T, c2w):
    """A similarity {s, R, t} applied to cameras (n,4,4)."""
    out = np.array(c2w, float, copy=True)
    out[:, :3, :3] = T["R"] @ out[:, :3, :3]
    out[:, :3, 3] = T["s"] * out[:, :3, 3] @ T["R"].T + T["t"]
    return out


def pts(T, P):
    return T["s"] * np.asarray(P, float) @ T["R"].T + T["t"]


def inverse(T):
    Ri = T["R"].T
    return {"s": 1 / T["s"], "R": Ri, "t": -(Ri @ T["t"]) / T["s"]}


def compose(T2, T1):
    """T2 after T1."""
    return {"s": T2["s"] * T1["s"], "R": T2["R"] @ T1["R"], "t": T2["s"] * T2["R"] @ T1["t"] + T2["t"]}


def floor_T(fr):
    """A cards floor frame {R rows = axes, origin} as the similarity shot -> floor."""
    return {"s": 1., "R": np.asarray(fr["R"], float), "t": -np.asarray(fr["R"], float) @ np.asarray(fr["origin"], float)}


def snap(Tf):
    """B floor -> A floor similarity: its tilt and floor shift, and the snapped one (a turn about +z, a shift in the floor
    plane, the scale): both floors are z = 0."""
    z = Tf["R"][:, 2]
    tilt = float(np.degrees(np.arccos(np.clip(z[2], -1, 1))))
    yaw = float(np.arctan2(Tf["R"][1, 0] - Tf["R"][0, 1], Tf["R"][0, 0] + Tf["R"][1, 1]))
    c, s_ = np.cos(yaw), np.sin(yaw)
    return tilt, float(Tf["t"][2]), yaw, {"s": Tf["s"], "R": np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1.]]), "t": np.array([Tf["t"][0], Tf["t"][1], 0.])}


def grid_K(K):
    """K at DA3's 504x280 -> K on the pick layer's depth grid (4x4 blocks)."""
    K = np.asarray(K, float)
    return np.array([[K[0, 0] / 4, 0, (K[0, 2] + .5) / 4 - .5], [0, K[1, 1] / 4, (K[1, 2] + .5) / 4 - .5], [0, 0, 1.]])


def pool_min(d, k=4):
    """(n, H, W) depth -> (n, H/k, W/k): minimum over positive values per block (0 = none), the pick depth grid's rule."""
    n, h, w = d.shape
    b = np.where(d > 0, d, np.inf).reshape(n, h // k, k, w // k, k).min((2, 4))
    return np.where(np.isfinite(b), b, 0.)


def ratio_per_frame(known, pred):
    """Per frame: median of known / pred over cells where both have depth."""
    out = []
    for a, b in zip(known, pred):
        ok = (a > 0) & (b > 0)
        out.append(float(np.median(a[ok] / b[ok])) if ok.sum() >= 50 else np.nan)
    return np.array(out)


# ---------------------------------------------------------------- one visit (a report)

def load(root, report):
    """The report's newest video, cameras, pick, object_cards (and models) layers, its cards and the naming passes' DINOv2
    vectors (reports/<id>/naming-*.npz, when the Volume holds them)."""
    root = Path(root)
    newest = {}
    for p in sorted((root / "reports" / report / "patches").glob("*.json")):
        newest[p.stem.split("-", 1)[1]] = p
    P = {k: json.loads(v.read_text()) for k, v in newest.items() if k in ("video", "cameras", "pick", "object_cards", "models")}
    oc = P["object_cards"]["data"]
    cl = oc["cards"] if isinstance(oc.get("cards"), list) else json.loads(ondemand.blob(root, P["object_cards"]["blobs"]["cards"]["sha256"]))
    V = {"root": str(root), "report": report, "video": root / "blobs" / "sha256" / P["video"]["blobs"]["video"]["sha256"],
         "fps": P["video"]["data"]["fps"], "window_s": P["video"]["data"].get("window_s"), "pick": P["pick"],
         "cameras": {s["index"]: s for s in P["cameras"]["data"]["shots"]}, "shots": {s["index"]: s for s in oc.get("shots") or []},
         "chunks": {}, "frames": {}, "aliases": oc.get("aliases") or {}, "models": P.get("models"),
         "cards": [c for c in cl if c.get("kind") == "object"], "cards_seq": P["object_cards"]["seq"]}
    ent = P["pick"]["data"]["entities"]
    V["owner"] = [resolve(V["aliases"], e) if isinstance(e, str) else None for e in ent]
    V["pick_at"] = {(f["shot"], f["key"]): i for i, f in enumerate(P["pick"]["data"]["frames"]) if f.get("key") is not None}
    V["pick_of_frame"] = {f["frame"]: i for i, f in enumerate(P["pick"]["data"]["frames"])}
    V["dino"] = dino_vectors(root, report, V)
    return V


def resolve(aliases, e):
    seen = set()
    while e in aliases and aliases[e] != e and e not in seen:
        seen.add(e)
        e = aliases[e]
    return e


def dino_vectors(root, report, V):
    """card id -> unit DINOv2-L vector (the naming passes' own: the plain and masked crops of up to 5 views); a card merged
    from several first-pass objects sums theirs; the densify pass overrides the first."""
    vec = {}
    for tag in ("first", "densify"):
        p = Path(root) / "reports" / report / f"naming-{tag}.npz"
        if p.exists():
            z = np.load(p)
            vec.update({str(i): v.astype(np.float32) for i, v in zip(z["ids"], z["dino"]) if np.any(v)})
    out = {}
    for c in V["cards"]:
        mem = [c["id"], *[x for x in (c.get("physical") or {}).get("merged_from") or [] if isinstance(x, str)]]
        vs = [vec[x] for x in mem if x in vec]
        if vs:
            v = np.sum(vs, 0)
            out[c["id"]] = v / max(np.linalg.norm(v), 1e-9)
    return out


def floor_of(V, shot):
    return ondemand.floor_of(V, shot)


def decode(V, frames):
    """{video frame index: BGR} for the wanted frames (one sequential pass; ondemand.decode_keyframes' way)."""
    import cv2
    want, cap, q = set(int(f) for f in frames) - set(V["frames"]), cv2.VideoCapture(str(V["video"])), 0
    while want:
        if not cap.grab():
            break
        if q in want:
            V["frames"][q] = cap.retrieve()[1]
            want.discard(q)
        q += 1
    cap.release()
    return V["frames"]


def shot_arrays(V, shot, keys_idx=None):
    """One shot's keyframes as the see-through test wants them: frame ids, c2w (shot frame, estimated metres), K on the depth
    grid, depth (metres) and person pixels on the grid; every keyframe, or those indexed."""
    import cv2
    cam = V["cameras"][shot]
    idx = range(len(cam["keys"])) if keys_idx is None else keys_idx
    keys, c2w, K, depth, person = [], [], [], [], []
    g = V["pick"]["data"]["depth"]
    for j in idx:
        i = V["pick_at"].get((shot, j))
        if i is None:
            continue
        d = ondemand.chunk_of(V, i, "depth").astype(np.float32) / 1000.
        ids = ondemand.chunk_of(V, i, "pick")
        per = np.isin(ids, [k for k, e in enumerate(V["pick"]["data"]["entities"]) if isinstance(e, str) and e.startswith("person")])
        keys.append(int(cam["keys"][j]))
        c2w.append(np.asarray(cam["c2w"][j], float))
        K.append(grid_K(cam["K"][j]))
        depth.append(d)
        person.append(cv2.resize(per.astype(np.uint8), (g["w"], g["h"]), interpolation=cv2.INTER_AREA) > 0)
    return {"keys": keys, "c2w": np.array(c2w), "K": np.array(K), "depth": np.array(depth), "person": np.array(person), "shot": shot}


def card_codes(V, cid):
    return [k for k, o in enumerate(V["owner"]) if o == cid]


def card_views(V, c, k=OBJ_VIEWS):
    """The card's best views first, then others spread over its keyframes (video frame ids with a pick frame)."""
    vw = c.get("views") or {}
    order = list(vw.get("best") or []) + [f for f in (vw.get("keyframes") or []) if f not in (vw.get("best") or [])]
    have = [f for f in order if f in V["pick_of_frame"]]
    if len(have) <= k:
        return have
    head, rest = have[:min(3, k)], have[min(3, k):]
    return head + [rest[int(i)] for i in np.linspace(0, len(rest) - 1, k - len(head))]


def card_points(V, c, n=OBJ_POINTS):
    """The card's pick-mask pixels on its views lifted with the pick depth grid -> (m, 3) shot-frame points and, per view,
    (frame, box in source px). ondemand.lift's cells: well inside the mask first, depth outliers trimmed."""
    import cv2
    codes = card_codes(V, c["id"])
    cam = V["cameras"][c["shot"]]
    g = V["pick"]["data"]["depth"]
    sw, sh = V["pick"]["data"]["source_wh"]
    P, boxes = [], {}
    for f in card_views(V, c):
        i = V["pick_of_frame"][f]
        pf = V["pick"]["data"]["frames"][i]
        if pf["shot"] != c["shot"] or pf.get("key") is None:
            continue
        ids = ondemand.chunk_of(V, i, "pick")
        mask = np.isin(ids, codes)
        if not mask.any():
            continue
        ys, xs = np.nonzero(mask)
        sx, sy = sw / mask.shape[1], sh / mask.shape[0]
        boxes[f] = [float(xs.min() * sx), float(ys.min() * sy), float((xs.max() + 1) * sx), float((ys.max() + 1) * sy)]
        d = ondemand.chunk_of(V, i, "depth").astype(np.float32) / 1000.
        cov = cv2.resize(mask.astype(np.float32), (g["w"], g["h"]), interpolation=cv2.INTER_AREA)
        cells = (d > 0) & (cov >= ondemand.INTERIOR)
        if cells.sum() < ondemand.MIN_CELLS:
            cells = (d > 0) & (cov >= ondemand.LOOSE)
        if cells.sum() < ondemand.MIN_CELLS:
            continue
        z = d[cells]
        med = float(np.median(z))
        keep = np.abs(z - med) <= max(3 * 1.4826 * float(np.median(np.abs(z - med))), .15 * med)
        vy, vx = np.nonzero(cells)
        vy, vx, z = vy[keep], vx[keep], z[keep]
        Kg, M = grid_K(cam["K"][pf["key"]]), np.asarray(cam["c2w"][pf["key"]], float)
        pc = np.stack([(vx - Kg[0, 2]) / Kg[0, 0] * z, (vy - Kg[1, 2]) / Kg[1, 1] * z, z], 1)
        P.append(pc @ M[:3, :3].T + M[:3, 3])
    if not P:
        return np.zeros((0, 3)), boxes
    P = np.concatenate(P)
    if len(P) > n:
        P = P[np.random.default_rng(0).choice(len(P), n, replace=False)]
    return P, boxes


# ---------------------------------------------------------------- the cards as comparable records

def geo_u(f):
    """A physical field's u without its scale part (the two visits share one registered scale)."""
    if not isinstance(f, dict) or f.get("u") is None:
        return None
    sc = float((f.get("parts") or {}).get("scale") or 0.)
    return float(np.sqrt(max(f["u"] ** 2 - sc ** 2, 0.)))


def val(f):
    """(value, u without scale, measured?) of a scalar field; measured = a value that is no bound."""
    if not isinstance(f, dict) or f.get("value") is None or isinstance(f["value"], list):
        return None, None, False
    bound = f.get("status") in ("at least", "at most", "needs review") or f.get("bound")
    return float(f["value"]), geo_u(f), not bound


def family(c):
    t = (c.get("identity") or {}).get("type") or {}
    return t.get("family")


def specific(c):
    n = (c.get("identity") or {}).get("name") or ""
    return bool(n) and cards.TYPE_ONLY not in n and "unidentified" not in n and n not in cards.SHAPES


def record(V, c, T=None):
    """A card in the site map's floor frame (T: this visit's floor -> the site map's floor; None = already there)."""
    ph = c.get("physical") or {}
    pos = ph.get("position_xy") or {}
    if pos.get("value") is None:
        return None
    top, u_top, top_ok = val(ph.get("top_above_floor"))
    base, u_base, base_ok = val(ph.get("base_above_floor"))
    top = top if top is not None else (base or 0.)
    base = base if base is not None else top
    s = T["s"] if T else 1.
    xy = np.array([*pos["value"], (top + base) / 2])
    xy = pts(T, xy[None])[0] if T else xy
    ext = [val(ph.get(n))[0] for n in ("height", "width", "depth")]
    foot = np.asarray((ph.get("footprint_xy") or {}).get("value") or [], float).reshape(-1, 2)
    if T is not None and len(foot):
        foot = pts(T, np.c_[foot, np.zeros(len(foot))])[:, :2]
    ang = {}
    for n in ("principal_axis_tilt_deg", "planar_slope_deg"):
        a = ph.get(n) or {}
        if a.get("value") is not None and not isinstance(a["value"], list):
            ang[n] = (float(a["value"]), float(a.get("u") or 0.))
    return {"id": c["id"], "shot": c["shot"], "xy": xy[:2], "u_xy": geo_u(pos) or .1, "top": s * top, "base": s * base,
            "u_top": s * (u_top or .1), "u_base": s * (u_base or .1), "top_ok": top_ok, "base_ok": base_ok,
            "size": s * max([e for e in ext if e is not None] or [0.]), "foot": foot, "fam": family(c), "angles": ang,
            "name": (c.get("identity") or {}).get("name"), "type": ((c.get("identity") or {}).get("type") or {}).get("label"),
            "specific": specific(c), "dino": V["dino"].get(c["id"]), "card": c}


def compatible(fa, fb):
    return fa in GENERIC or fb in GENERIC or fa == fb


def cos(a, b):
    return float(a["dino"] @ b["dino"]) if a["dino"] is not None and b["dino"] is not None else None


def match(A, B, u_reg):
    """One-to-one pairs of A and B records (lists) at the same place. -> [(i, j, cost, d, gate)].
    Same place: centres within gate = max(K_SIGMA x the pose u (registration, both cameras), NEAR_SIZE x the smaller object's size
    (a partial view moves a centre by up to half the object), NEAR_MIN_M), or the smaller footprint >= PART_SHARE inside the larger
    (one visit delineated a part of what the other saw whole); height ranges overlapping by >= Z_SHARE of the shorter one.
    Same thing: appearance cos >= COS_MIN, and two specific types that differ only at cos >= COS_TYPES."""
    from scipy.optimize import linear_sum_assignment
    if not A or not B:
        return []
    XA, XB = np.array([a["xy"] for a in A]), np.array([b["xy"] for b in B])
    D = np.linalg.norm(XA[:, None] - XB[None], axis=2)
    pose = K_SIGMA * float(np.sqrt(u_reg ** 2 + 2 * POSE_M ** 2))
    SZ = np.minimum(np.array([a["size"] for a in A])[:, None], np.array([b["size"] for b in B])[None])
    GATE = np.maximum(np.maximum(pose, NEAR_SIZE * SZ), NEAR_MIN_M)
    RA, RB = np.array([_radius(a) for a in A]), np.array([_radius(b) for b in B])
    cost = np.full(D.shape, 1e6)
    for i, j in zip(*np.nonzero((D <= GATE) | (D <= RA[:, None] + RB[None]))):
        a, b = A[i], B[j]
        if D[i, j] > GATE[i, j] and part_share(a, b) < PART_SHARE:
            continue
        lo, hi = max(a["base"], b["base"]), min(a["top"], b["top"])
        uz = max(Z_TOL_M, float(np.hypot(max(a["u_top"], a["u_base"]), max(b["u_top"], b["u_base"]))))
        if hi - lo < Z_SHARE * max(min(a["top"] - a["base"], b["top"] - b["base"]), .02) - uz:
            continue  # height ranges apart (a sticky note on a box is not the box)
        cs = cos(a, b)
        if cs is not None and (cs < COS_MIN or not compatible(a["fam"], b["fam"]) and cs < COS_TYPES):
            continue
        iou = prism_iou(a, b, pose / K_SIGMA)
        if iou < MIN_IOU and not (part_share(a, b) >= PART_KEEP and cs is not None and cs >= COS_PART):
            continue  # one object fills one volume in both visits (a part of it only when it also looks like it)
        same_name = a["specific"] and b["specific"] and cards.canonical(a["name"]) == cards.canonical(b["name"])
        cost[i, j] = (D[i, j] / GATE[i, j] + 2 * (1 - (cs if cs is not None else .5)) + (0. if compatible(a["fam"], b["fam"]) else 1.)
                      + W_OVERLAP * (1 - iou) - (.5 if same_name else 0.))
    rows, cols = linear_sum_assignment(cost)
    return [(int(i), int(j), float(cost[i, j]), float(D[i, j]), float(GATE[i, j])) for i, j in zip(rows, cols) if cost[i, j] < 1e6]


W_OVERLAP = 1.5  # the cost of two objects that do not share a volume (a neighbour of similar look at a similar place)
# r5b dev (928 matched pairs of 9 TUM + 2 ARKit pairs against GT points): the grown volumes' IoU told right from wrong best
# (medians 0.37 vs 0.11); >= 0.1, or a part >= 80 % inside that looks alike (cos >= 0.7), kept 559 of 598 right pairs at 75 % (64 % before)
MIN_IOU, PART_KEEP, COS_PART = .1, .8, .7


def prism_iou(a, b, grow):
    """IoU of the two objects' volumes (footprint hull x height range), each grown by `grow` (the pose u): a swap between two
    neighbours of similar look costs more than the distance of their centres says."""
    if len(a["foot"]) < 3 or len(b["foot"]) < 3:
        return 0.
    from shapely.geometry import MultiPoint
    pa, pb = (MultiPoint([tuple(p) for p in r["foot"]]).convex_hull.buffer(grow) for r in (a, b))
    fp = pa.intersection(pb).area / max(pa.union(pb).area, 1e-9)
    lo, hi = max(a["base"], b["base"]) - grow, min(a["top"], b["top"]) + grow
    v = max(hi - lo, 0.) / max(max(a["top"], b["top"]) - min(a["base"], b["base"]) + 2 * grow, 1e-9)
    return float(fp * v)


def part_share(a, b):
    """The share of the smaller footprint (convex hull) inside the larger one."""
    if len(a["foot"]) < 3 or len(b["foot"]) < 3:
        return 0.
    from shapely.geometry import MultiPoint
    pa, pb = MultiPoint([tuple(p) for p in a["foot"]]).convex_hull, MultiPoint([tuple(p) for p in b["foot"]]).convex_hull
    small = min(pa.area, pb.area)
    return float(pa.intersection(pb).area / small) if small > 1e-9 else 0.


def _radius(r):
    return float(np.linalg.norm(r["foot"] - r["foot"].mean(0), axis=1).max()) if len(r["foot"]) else 0.


def tau_move(A, B, pairs):
    """The pair's appearance threshold for a move: the 99th percentile of cos between A and B cards >= NEG_MIN_M apart that
    are not matched (the video's negative-pair rule of fast_report.timeline); TAU_MOVE_FALLBACK with under 20 such pairs."""
    ia = [i for i, a in enumerate(A) if a["dino"] is not None]
    jb = [j for j, b in enumerate(B) if b["dino"] is not None]
    if not ia or not jb:
        return TAU_MOVE_FALLBACK, 0
    C = np.array([A[i]["dino"] for i in ia]) @ np.array([B[j]["dino"] for j in jb]).T
    D = np.linalg.norm(np.array([A[i]["xy"] for i in ia])[:, None] - np.array([B[j]["xy"] for j in jb])[None], axis=2)
    ok = D >= NEG_MIN_M
    for i, j, *_ in pairs:
        if i in ia and j in jb:
            ok[ia.index(i), jb.index(j)] = False
    neg = C[ok]
    return (float(np.percentile(neg, 99)), int(len(neg))) if len(neg) >= 20 else (TAU_MOVE_FALLBACK, int(len(neg)))


def differences(a, b):
    """A matched pair's measured differences (B - A, site-map metres), each +-u, and the claims they support."""
    out, claims = {}, []
    d = b["xy"] - a["xy"]
    out["position_xy_m"] = {"value": np.round(d, 3).tolist(), "norm": round(float(np.linalg.norm(d)), 3),
                            "u": round(float(np.hypot(a["u_xy"], b["u_xy"])), 3)}
    for n, ok in (("top_above_floor", a["top_ok"] and b["top_ok"]), ("base_above_floor", a["base_ok"] and b["base_ok"])):
        k = n.split("_")[0]
        dv, u = b[k] - a[k], float(np.hypot(a["u_" + k], b["u_" + k]))
        out[n] = {"value": round(float(dv), 3), "u": round(u, 3), "both_measured": bool(ok)}
        if ok and abs(dv) > max(K_CHANGE * u, MIN_DZ_M):
            claims.append("changed_height")
    for n in set(a["angles"]) & set(b["angles"]):
        dv, u = b["angles"][n][0] - a["angles"][n][0], float(np.hypot(a["angles"][n][1], b["angles"][n][1]))
        out[n] = {"value": round(dv, 1), "u": round(u, 1)}
        if abs(dv) > max(K_CHANGE * u, MIN_DDEG):
            claims.append("changed_angle")
    return out, sorted(set(claims))


# ---------------------------------------------------------------- registration

def keyframes(V):
    """Every keyframe with a pick frame: (video frame ids, shot per frame)."""
    ids, shot = [], []
    for si, cam in V["cameras"].items():
        for j, f in enumerate(cam["keys"]):
            if (si, j) in V["pick_at"]:
                ids.append(int(f))
                shot.append(si)
    return ids, shot


def descriptors(V, gpu):
    """(frame ids, (n, D) unit DINOv2-L descriptors, shot per frame) over every keyframe of every shot."""
    ids, shot = keyframes(V)
    frames = decode(V, ids)
    return np.array(ids), gpu.embed([frames[f] for f in ids]), np.array(shot)


def select(S, fa, fb, sa, sb, b_shot):
    """B frames of one shot (the B_FRAMES most like the site map, MIN_GAP_KEYS keyframes apart) and the A shot they vote for,
    with each one's A_PER_B most similar A frames of that shot (2 keyframes apart). S: (nb, na) cosine."""
    rows = np.flatnonzero(sb == b_shot)
    a_shot = int(np.bincount(sa[S[rows].argmax(1)], weights=S[rows].max(1)).argmax())
    cols = np.flatnonzero(sa == a_shot)
    pos_b, pos_a = {r: k for k, r in enumerate(rows)}, {c: k for k, c in enumerate(cols)}
    picked = []
    for r in rows[np.argsort(-S[rows][:, cols].max(1), kind="stable")]:
        if all(abs(pos_b[r] - pos_b[q]) >= MIN_GAP_KEYS for q in picked):
            picked.append(r)
        if len(picked) == B_FRAMES:
            break
    a_rows = []
    for r in picked:
        got = 0
        for c in cols[np.argsort(-S[r, cols], kind="stable")]:
            if c in a_rows:
                got += 1
            elif all(abs(pos_a[c] - pos_a[q]) >= 2 for q in a_rows):
                a_rows.append(c)
                got += 1
            if got >= A_PER_B:
                break
    sim = float(np.median(S[picked][:, a_rows].max(1))) if picked and a_rows else 0.
    for c in list(a_rows):  # neighbours of the matched views: a wider camera spread for the fit, still seeing the same things
        for dk in (-A_SPREAD_KEYS, A_SPREAD_KEYS):
            k = pos_a[c] + dk
            if len(a_rows) < A_MAX and 0 <= k < len(cols) and all(abs(k - pos_a[q]) >= 2 for q in a_rows):
                a_rows.append(cols[k])
    return a_shot, [int(fb[r]) for r in picked], [int(fa[c]) for c in a_rows], sim


def own(V, shot, frames):
    """The report's own cameras and depth grid (metres) for these video frames of one shot."""
    cam = V["cameras"][shot]
    j = [cam["keys"].index(f) for f in frames]
    c2w = np.array([cam["c2w"][k] for k in j], float)
    depth = np.array([ondemand.chunk_of(V, V["pick_at"][(shot, k)], "depth").astype(np.float32) / 1000. for k in j])
    return c2w, depth


def register(A, B, gpu, b_shot, S, desc):
    """One B shot into the site map. -> the registration record (accepted or refused, with its reasons and numbers)."""
    fa, sa, fb, sb = desc
    a_shot, bf, af, sim = select(S, fa, fb, sa, sb, b_shot)
    rec = {"b_shot": int(b_shot), "a_shot": a_shot, "b_frames": bf, "a_frames": af, "retrieval_cos_median": round(sim, 3)}
    if len(bf) < 3 or len(af) < 3:
        return {**rec, "accepted": False, "refused_because": "too few frames to register"}
    fa_img, fb_img = [decode(A, af)[f] for f in af], [decode(B, bf)[f] for f in bf]
    t0 = time.perf_counter()
    out = gpu.da3(fa_img + fb_img)
    rec["da3_s"] = round(time.perf_counter() - t0, 3)
    n = len(af)
    ka, da = own(A, a_shot, af)
    kb, db = own(B, b_shot, bf)
    pred = pool_min(out["depth"])
    rec["gate"] = {"max_centre_residual_share": MAX_CENTRE_RESIDUAL_SHARE, "max_rotation_residual_deg": MAX_ROTATION_RESIDUAL_DEG,
                   "spread_floor_m": SPREAD_FLOOR_M, "max_depth_ratio_spread": MAX_DEPTH_RATIO_SPREAD,
                   "floor_tilt_max_deg": FLOOR_TILT_MAX_DEG, "floor_dz_max_m": FLOOR_DZ_MAX_M, "rule": "scripts/register_cut_shot.py's gate on both fits"}
    FA, FB = floor_of(A, a_shot), floor_of(B, b_shot)
    if FA is None or FB is None:
        return {**rec, "accepted": False, "refused_because": "a shot without a floor: heights cannot be compared"}
    ratio_a = ratio_per_frame(da, pred[:n])
    tried = []
    for scale in SCALES:  # the similarity's scale from the depth ratio or from the camera centres: the floors say which (below)
        fitA = fit(ka, out["c2w"][:n], ratio_a, scale)
        v = {"scale": scale, "fit_a": _num(fitA)}
        tried.append(v)
        if not passes(fitA):
            v["refused_because"] = "the site map's cameras do not agree after the similarity"
            continue
        carried = move(fitA, out["c2w"][n:])
        fitB = fit(carried, kb, ratio_per_frame(fitA["s"] * pred[n:], db), scale)  # B's own cameras onto their carried places
        v["fit_b"] = _num(fitB)
        if not passes(fitB):
            v["refused_because"] = "this visit's own cameras do not agree with their registered places"
            continue
        T_shot = {k: fitB[k] for k in ("s", "R", "t")}  # B shot -> A shot
        tilt, dz, yaw, Tsnap = snap(compose(floor_T(FA), compose(T_shot, inverse(floor_T(FB)))))  # B floor -> A floor
        v["floor"] = {"tilt_deg": round(tilt, 2), "dz_m": round(dz, 3)}
        if tilt > FLOOR_TILT_MAX_DEG or abs(dz) > FLOOR_DZ_MAX_M:
            v["refused_because"] = f"floors disagree (tilt {tilt:.1f} deg, shift {dz:.2f} m)"
            continue
        v.update(_fits=(fitA, fitB), _T=(Tsnap, yaw, T_shot))
    ok = [v for v in tried if "_T" in v]
    rec["scales"] = [{k: x for k, x in v.items() if not k.startswith("_")} for v in tried]
    if not ok:
        return {**rec, **{k: tried[0].get(k) for k in ("fit_a", "fit_b", "floor")}, "accepted": False, "refused_because": tried[0]["refused_because"]}
    best = min(ok, key=lambda v: abs(v["floor"]["dz_m"]))  # the scale that puts this visit's floor on the site map's
    (fitA, fitB), (Tsnap, yaw, T_shot) = best["_fits"], best["_T"]
    rec.update(fit_a=best["fit_a"], fit_b=best["fit_b"], floor=best["floor"], scale_from=best["scale"])
    zd = float(np.median(db[db > 0])) if (db > 0).any() else 2.
    u_pose = [(V["shots"].get(sh) or {}).get("u_pose_m") or POSE_M for V, sh in ((A, a_shot), (B, b_shot))]  # each map's own pose u
    u_reg = float(np.sqrt(fitA["centre_residual_m"] ** 2 + fitB["centre_residual_m"] ** 2 + u_pose[0] ** 2 + u_pose[1] ** 2
                          + (np.radians(max(fitA["rotation_residual_deg"], fitB["rotation_residual_deg"])) * zd) ** 2))
    u_floor = float(np.hypot((A["shots"].get(a_shot) or {}).get("u_floor_m") or .02, (B["shots"].get(b_shot) or {}).get("u_floor_m") or .02))
    T_shot_snapped = compose(inverse(floor_T(FA)), compose(Tsnap, floor_T(FB)))
    return {**rec, "accepted": True, "refused_because": None, "u_m": round(u_reg, 3), "u_floor_m": round(u_floor, 3), "scene_depth_m": round(zd, 2),
            "floor_transform": {"s": round(Tsnap["s"], 5), "yaw_deg": round(float(np.degrees(yaw)), 3), "t_m": np.round(Tsnap["t"][:2], 4).tolist(),
                                "note": "B floor frame -> A floor frame: a turn about +z, a shift in the floor plane, a scale (floors snapped)"},
            "_T_floor": Tsnap, "_T_shot": T_shot_snapped, "_T_shot_raw": T_shot}


# ---------------------------------------------------------------- the comparison

def place(points, V, arr, u):
    """timeline.place of shot-frame points (already in V's shot frame) against V's keyframes (arr from shot_arrays)."""
    if len(points) < 30:
        return {"state": "unjudged", "views": 0, "free_views": 0, "occupied_views": 0, "best_key": None, "best_free_share": None, "reason": "too few points"}
    return timeline.place(points, arr, pose_m=max(timeline.POSE_M, u), rel_margin=REL_MARGIN, neigh=NEIGH, free_share=FREE_SHARE)  # arr caches its minima


def corners(r):
    """A record's volume (footprint hull corners x base / top, site-map floor frame) and its centre: points to project."""
    f = r["foot"] if len(r["foot"]) else r["xy"][None]
    return np.concatenate([np.c_[f, np.full(len(f), z)] for z in (r["base"], r["top"])] + [[[*r["xy"], (r["base"] + r["top"]) / 2]]])


def in_view(P, arr, margin=.05):
    """Any of the points in front of any of the keyframes and inside its image (the depth grid)."""
    h, w = arr["depth"].shape[1:]
    for c2w, K in zip(arr["c2w"], arr["K"]):
        cam = (P - c2w[:3, 3]) @ c2w[:3, :3]
        z = cam[:, 2]
        ok = z > .1
        if not ok.any():
            continue
        u, v = K[0, 0] * cam[ok, 0] / z[ok] + K[0, 2], K[1, 1] * cam[ok, 1] / z[ok] + K[1, 2]
        if ((u >= -margin * w) & (u < (1 + margin) * w) & (v >= -margin * h) & (v < (1 + margin) * h)).any():
            return True
    return False


def project_box(P, c2w, K, wh, src_wh):
    """Shot-frame points -> their box in source px on a keyframe (K at DA3's wh), or None when mostly behind the camera."""
    cam = (np.asarray(P) - c2w[:3, 3]) @ c2w[:3, :3]
    ok = cam[:, 2] > .05
    if ok.mean() < .5:
        return None
    u = (K[0, 0] * cam[ok, 0] / cam[ok, 2] + K[0, 2] + .5) * src_wh[0] / wh[0] - .5
    v = (K[1, 1] * cam[ok, 1] / cam[ok, 2] + K[1, 2] + .5) * src_wh[1] / wh[1] - .5
    return [float(np.percentile(u, 2)), float(np.percentile(v, 2)), float(np.percentile(u, 98)), float(np.percentile(v, 98))]


def cam_at(V, shot, frame):
    cam = V["cameras"][shot]
    j = cam["keys"].index(frame)
    return np.asarray(cam["c2w"][j], float), np.asarray(cam["K"][j], float), cam["wh"]


def mesh_of(V, cid):
    """The models layer's accepted mesh of a card (its blob and pose), when the report has one."""
    ml = V.get("models")
    if not ml:
        return None
    for m in (ml.get("data") or {}).get("models") or []:
        if resolve(V["aliases"], m.get("object")) == cid and (ml.get("blobs") or {}).get("model-" + m["object"]):
            t = m["transform"]
            return {"sha256": ml["blobs"]["model-" + m["object"]]["sha256"],
                    "transform": {"position": t["position"], "quaternion": t["quaternion"], "scale": t.get("scale", [1, 1, 1])}}
    return None


def carry_model(model, T):
    """A card's display model (position, quaternion in its shot frame, sizes) carried by a similarity into another shot frame."""
    if not model or model.get("position") is None:
        return None
    from scipy.spatial.transform import Rotation
    q = Rotation.from_matrix(T["R"] @ Rotation.from_quat(model["quaternion"]).as_matrix()).as_quat()
    out = {k: v for k, v in model.items() if k not in ("position", "quaternion")}
    out.update(position=np.round(pts(T, np.asarray(model["position"], float)[None])[0], 4).tolist(), quaternion=np.round(q, 6).tolist())
    for k in ("size_m", "radius_m", "length_m", "scale"):
        if model.get(k) is not None:
            out[k] = np.round(np.asarray(model[k], float) * T["s"], 4).tolist()
    if model.get("parts"):
        out["parts"] = [[*np.round(np.asarray(p[:6], float) * T["s"], 4).tolist(), *p[6:]] for p in model["parts"]]
    return out


OUT_OF_VIEW = {"state": "out-of-view", "views": 0, "free_views": 0, "occupied_views": 0, "best_key": None, "best_free_share": None,
               "reason": "its volume is in none of the other visit's views"}


def classify(recA, recB, u_reg, points_a, points_b, place_in_b, place_in_a, height_a=None, height_b=None, seen_in_b=None, seen_in_a=None):
    """The differences between two registered shots, from their records (site-map floor frame), their objects' points
    (points_a(i) / points_b(j): shot-frame points), the see-through tests (place_in_b(P) of A's points in B's views,
    place_in_a(P) of B's in A's) and the points' heights above the common floor (height_a / height_b, site-map metres).
    -> {pairs: [(i, j, cost, d, gate, diff, claims, unconfirmed)], moves, status_a {i: status}, status_b {j: status}, ...}.
    A height difference is a claim only when the part one visit has and the other lacks is seen through by the other visit
    (a partial view lowers a top without any change); an angle difference only on a strong match (look, type, place)."""
    pairs = match(recA, recB, u_reg)
    tau, n_neg = tau_move(recA, recB, pairs)
    pose = K_SIGMA * float(np.sqrt(u_reg ** 2 + 2 * POSE_M ** 2))
    placeB, placeA, broken = {}, {}, []

    def in_b(i):
        if i not in placeB:
            placeB[i] = place_in_b(points_a(i)) if seen_in_b is None or seen_in_b(i) else dict(OUT_OF_VIEW)
        return placeB[i]

    def in_a(j):
        if j not in placeA:
            placeA[j] = place_in_a(points_b(j)) if seen_in_a is None or seen_in_a(j) else dict(OUT_OF_VIEW)
        return placeA[j]
    kept = []
    for x in pairs:  # one object at one place in both visits: neither visit may see the other's object's place empty
        (kept if in_b(x[0])["state"] != "free" and in_a(x[1])["state"] != "free" else broken).append(x)
    pairs = kept
    out_pairs, checks = [], {}
    for i, j, cost, d, gate in pairs:
        a, b = recA[i], recB[j]
        diff, claims = differences(a, b)
        kept, unconfirmed = [], []
        for c in claims:
            if c == "changed_height" and height_a is not None:
                chk = _height_check(a, b, diff, lambda: points_a(i), lambda: points_b(j), height_a, height_b, place_in_b, place_in_a)
                checks[(i, j)] = chk
                (kept if chk["confirmed"] else unconfirmed).append(c)
            elif c == "changed_angle":
                cs = cos(a, b)
                strong = cs is not None and cs >= tau and d <= pose and compatible(a["fam"], b["fam"])
                (kept if strong else unconfirmed).append(c)
            else:
                kept.append(c)
        out_pairs.append((i, j, cost, d, gate, diff, kept, unconfirmed))
    mi, mj = {i for i, *_ in pairs}, {j for _, j, *_ in pairs}
    # seen_in_b(i) / seen_in_a(j): False when the object's volume projects into none of the other visit's views (skips lifting it)
    placeB = {i: in_b(i) for i in range(len(recA)) if i not in mi}
    placeA = {j: in_a(j) for j in range(len(recB)) if j not in mj}
    gone = [i for i, p in placeB.items() if p["state"] == "free"]
    came = [j for j, p in placeA.items() if p["state"] in ("free", "out-of-view", "occluded", "unjudged")]
    moves = _moves(recA, recB, gone, came, tau, u_reg)
    ma, mb = {m[0] for m in moves}, {m[1] for m in moves}
    status_a = {i: "missing" if p["state"] == "free" else "delineated_otherwise" if p["state"] == "occupied" else "not_observed_in_b"
                for i, p in placeB.items() if i not in ma}
    status_b = {j: "new" if p["state"] == "free" else "delineated_otherwise" if p["state"] == "occupied" else "not_observed_in_a"
                for j, p in placeA.items() if j not in mb}
    return {"pairs": out_pairs, "moves": moves, "status_a": status_a, "status_b": status_b, "placeA": placeA, "placeB": placeB,
            "gone": gone, "came": came, "tau": tau, "negatives": n_neg, "height_checks": checks,
            "broken": [(i, j) for i, j, *_ in broken]}


def _height_check(a, b, diff, pa, pb, height_a, height_b, place_in_b, place_in_a, min_points=30):
    """A top that dropped: A's points above B's top must be seen through in B; a top that rose: B's points above A's top seen
    through in A; a base that rose / dropped likewise below. -> {confirmed, tests: [{what, points, place}]}."""
    tests = []
    dt, db = diff["top_above_floor"], diff["base_above_floor"]
    if dt["both_measured"] and abs(dt["value"]) > max(K_CHANGE * dt["u"], MIN_DZ_M):
        m = max(dt["u"], .05)
        if dt["value"] < 0:
            P = pa()
            tests.append(("top lower in B: A's part above B's top, seen from B", P[height_a(P) > b["top"] + m], place_in_b))
        else:
            P = pb()
            tests.append(("top higher in B: B's part above A's top, seen from A", P[height_b(P) > a["top"] + m], place_in_a))
    if db["both_measured"] and abs(db["value"]) > max(K_CHANGE * db["u"], MIN_DZ_M):
        m = max(db["u"], .05)
        if db["value"] > 0:
            P = pa()
            tests.append(("base higher in B: A's part below B's base, seen from B", P[height_a(P) < b["base"] - m], place_in_b))
        else:
            P = pb()
            tests.append(("base lower in B: B's part below A's base, seen from A", P[height_b(P) < a["base"] - m], place_in_a))
    rows = []
    for what, P, fn in tests:
        st = fn(P) if len(P) >= min_points else {"state": "unjudged", "reason": "too few points in the part that differs"}
        rows.append({"what": what, "points": int(len(P)), "place": _slim(st)})
    return {"confirmed": any(r["place"]["state"] == "free" for r in rows) and not any(r["place"]["state"] == "occupied" for r in rows), "tests": rows}


def stored(A, B, registration):
    """A layer's registration records back as the ones register() returns (the comparison again without the GPU)."""
    out = []
    for r in registration:
        r = dict(r)
        if r.get("accepted"):
            T = {"s": float(r["shot_transform"]["s"]), "R": np.asarray(r["shot_transform"]["R"], float), "t": np.asarray(r["shot_transform"]["t"], float)}
            r["_T_shot"] = T
            r["_T_floor"] = compose(floor_T(floor_of(A, r["a_shot"])), compose(T, inverse(floor_T(floor_of(B, r["b_shot"])))))
        out.append(r)
    return out


def compare(A, B, gpu, clock=None, evidence=True, regs=None):
    """The whole comparison of visit B against the site map A. -> (layer data, blobs {role: (bytes, meta)}, record).
    regs: registrations from an earlier layer (stored()): no descriptors and no DA3, the rest again."""
    stage = (lambda name, **k: clock.stage(name, **k)) if clock is not None else (lambda name, **k: _Null())
    t_all = time.perf_counter()
    if regs is not None:
        return _compare(A, B, regs, stage, evidence, t_all)
    with stage("visits.descriptors"):
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(2) as pool:  # the two videos decode side by side (OpenCV releases the GIL)
            list(pool.map(lambda V: decode(V, keyframes(V)[0]), (A, B)))
        fa, da, sa = descriptors(A, gpu)
        fb, db_, sb = descriptors(B, gpu)
    S = db_ @ da.T
    usable = np.array([len(A["cameras"][x]["keys"]) >= MIN_KEYS and A["cameras"][x].get("scale_status") == "estimated" for x in sa], bool)
    S[:, ~usable] = -1.  # a site-map shot without a floor or with too few keyframes is never a registration target
    regs = []
    with stage("visits.register"):
        for b_shot, cam in B["cameras"].items():
            if len(cam["keys"]) < MIN_KEYS or cam.get("scale_status") != "estimated":
                regs.append({"b_shot": int(b_shot), "accepted": False, "refused_because": f"shot with {len(cam['keys'])} keyframes or no floor scale"})
                continue
            regs.append(register(A, B, gpu, b_shot, S, (fa, sa, fb, sb)))
    return _compare(A, B, regs, stage, evidence, t_all)


def _compare(A, B, regs, stage, evidence, t_all):
    objects, tiles, counts = [], [], {}
    b_done = set()
    tau_rec = []
    with stage("visits.compare"):
        for r in regs:
            if not r["accepted"]:
                continue
            a_shot, b_shot = r["a_shot"], r["b_shot"]
            T_BA, T_AB = r["_T_shot"], inverse(r["_T_shot"])
            recA = [x for x in (record(A, c) for c in A["cards"] if c["shot"] == a_shot) if x]
            recB = [x for x in (record(B, c, r["_T_floor"]) for c in B["cards"] if c["shot"] == b_shot) if x]
            arrA = shot_arrays(A, a_shot, _spread(len(A["cameras"][a_shot]["keys"])))
            arrB = shot_arrays(B, b_shot, _spread(len(B["cameras"][b_shot]["keys"])))
            ptsA, ptsB = {}, {}
            FA, FB = floor_T(floor_of(A, a_shot)), floor_T(floor_of(B, b_shot))
            A_to_B = compose(T_AB, inverse(FA))  # site-map floor frame -> B's shot frame
            got = classify(recA, recB, r["u_m"],
                           lambda i: ptsA[i][0] if i in ptsA else ptsA.setdefault(i, card_points(A, recA[i]["card"]))[0],
                           lambda j: ptsB[j][0] if j in ptsB else ptsB.setdefault(j, card_points(B, recB[j]["card"]))[0],
                           lambda P: place(pts(T_AB, P), B, arrB, r["u_m"]), lambda P: place(pts(T_BA, P), A, arrA, r["u_m"]),
                           lambda P: pts(FA, P)[:, 2], lambda P: r["_T_floor"]["s"] * pts(FB, P)[:, 2],
                           lambda i: in_view(pts(A_to_B, corners(recA[i])), arrB), lambda j: in_view(pts(inverse(FA), corners(recB[j])), arrA))
            pairs, moves, gone, came, placeA, placeB = (got[k] for k in ("pairs", "moves", "gone", "came", "placeA", "placeB"))
            tau_rec.append({"b_shot": b_shot, "tau_move": round(got["tau"], 4), "negatives": got["negatives"]})
            ctx = {"A": A, "B": B, "r": r, "T_AB": T_AB, "T_BA": T_BA, "recA": recA, "recB": recB, "ptsA": ptsA, "ptsB": ptsB,
                   "placeA": placeA, "placeB": placeB}
            for i, j, cost, d, gate, diff, claims, unconfirmed in pairs:
                row = _row(ctx, claims[0] if claims else "static", i, j, diff=diff, claims=claims, cost=cost)
                if unconfirmed:
                    row["unconfirmed"] = unconfirmed  # a measured difference the see-through test (or a strong match) did not back
                if (i, j) in got["height_checks"]:
                    row["height_check"] = got["height_checks"][(i, j)]
                objects.append(row)
            for i, j, cs in moves:
                a, b = recA[i], recB[j]
                d = b["xy"] - a["xy"]
                diff = {"position_xy_m": {"value": np.round(d, 3).tolist(), "norm": round(float(np.linalg.norm(d)), 3),
                                          "u": round(float(np.sqrt(a["u_xy"] ** 2 + b["u_xy"] ** 2 + r["u_m"] ** 2)), 3)}}
                objects.append(_row(ctx, "moved", i, j, diff=diff, appearance=cs))
            for i, st in got["status_a"].items():
                objects.append(_row(ctx, st, i, None))
            for j, st in got["status_b"].items():
                objects.append(_row(ctx, st, None, j))
            b_done |= {x["id"] for x in recB}
    objects = one_row_per_object(objects)
    a_done = {o["a"] for o in objects if o.get("a")}
    for c in B["cards"]:  # a shot that was not registered: nothing compared
        if c["id"] not in b_done:
            objects.append({"status": "not_compared", "a": None, "b": c["id"], "site_id": "v1:" + c["id"], "shot_b": c["shot"],
                            "name_b": (c.get("identity") or {}).get("name"), "reason": "its shot was not registered to the site map"})
    for c in A["cards"]:
        if c["id"] not in a_done:
            objects.append({"status": "not_compared", "a": c["id"], "b": None, "site_id": "v0:" + c["id"], "shot_a": c["shot"],
                            "name_a": (c.get("identity") or {}).get("name"), "reason": "no shot of this visit was registered to its shot"})
    for o in objects:
        counts[o["status"]] = counts.get(o["status"], 0) + 1
    if evidence:
        with stage("visits.evidence"):
            tiles = _evidence(A, B, objects)
    blobs = {f"evidence-{k}": (jpg, {"mediaType": "image/jpeg", "format": "visit A | visit B"}) for k, jpg in tiles}
    for o in objects:
        o.pop("_ev", None)
    data = {"schema": SCHEMA, "site_map": A["report"], "this": B["report"],
            "visits": [{"id": "v0", "report": A["report"], "role": "site map", "window_s": A["window_s"], "video": A["video"].name},
                       {"id": "v1", "report": B["report"], "role": "this visit", "window_s": B["window_s"], "video": B["video"].name}],
            "registration": [{k: v for k, v in r.items() if not k.startswith("_")} | _transforms(r) for r in regs],
            "objects": objects, "counts": counts, "tau_move": tau_rec,
            "rules": {"k_sigma": K_SIGMA, "k_change": K_CHANGE, "min_dz_m": MIN_DZ_M, "min_ddeg": MIN_DDEG, "size_ratio": SIZE_RATIO,
                      "see_through": "fast_report.timeline.place on the other visit's pick depth grid (126x70, min per 4x4 DA3 block), pose u = the registration's",
                      "not_observed": "an unmatched object whose place the other visit did not see clearly (out of view, occluded, unjudged) is 'not observed', never missing or new",
                      "scale": "both visits' metres are estimated (floor + assumed 1.6 m camera); B is carried into A's metres by the registration's scale"},
            "s": round(time.perf_counter() - t_all, 3)}
    rec = {"registrations": len(regs), "accepted": sum(r["accepted"] for r in regs), "counts": counts, "s": data["s"]}
    return data, blobs, rec


RANK = {"delineated_otherwise": 1, "missing": 2, "not_observed_in_b": 3}
MATCHED = ("static", "changed_height", "changed_angle", "moved")


def one_row_per_object(rows):
    """Several B shots compared with one A shot: an A object matched in any of them keeps its matched rows (one per B card: the
    same site object seen in two shots) and loses its unmatched ones; an A object matched nowhere keeps its strongest unmatched
    row (delineated otherwise > missing > not observed): a place one shot saw empty and another saw occupied is no claim."""
    matched = {o["a"] for o in rows if o.get("a") and o["status"] in MATCHED}
    best = {}
    for k, o in enumerate(rows):
        if o.get("a") and o["a"] not in matched and o["status"] in RANK:
            if o["a"] not in best or RANK[o["status"]] < RANK[rows[best[o["a"]]]["status"]]:
                best[o["a"]] = k
    return [o for k, o in enumerate(rows) if not o.get("a") or o["status"] in MATCHED or best.get(o["a"]) == k]


class _Null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _spread(n, k=PLACE_KEYS):
    return list(range(n)) if n <= k else [int(i) for i in np.linspace(0, n - 1, k)]


def _transforms(r):
    if not r.get("accepted"):
        return {}
    T = r["_T_shot"]
    return {"shot_transform": {"s": round(T["s"], 6), "R": np.round(T["R"], 6).tolist(), "t": np.round(T["t"], 5).tolist(),
                               "note": "B shot frame -> A shot frame (estimated metres), floors snapped"}}


def _moves(recA, recB, gone, came, tau, u_reg):
    """A missing object and a new one (or one A never saw) that look alike, of a compatible type and size, far apart."""
    from scipy.optimize import linear_sum_assignment
    if not gone or not came:
        return []
    M = np.full((len(gone), len(came)), -1.)
    for x, i in enumerate(gone):
        for y, j in enumerate(came):
            a, b = recA[i], recB[j]
            cs = cos(a, b)
            if cs is None or cs < tau:
                continue
            if a["specific"] and b["specific"] and a["fam"] != b["fam"] or not compatible(a["fam"], b["fam"]):
                continue
            size = max(b["size"], 1e-3) / max(a["size"], 1e-3)
            far = np.linalg.norm(a["xy"] - b["xy"]) > K_SIGMA * float(np.sqrt(a["u_xy"] ** 2 + b["u_xy"] ** 2 + u_reg ** 2))
            if 1 / SIZE_RATIO <= size <= SIZE_RATIO and far:
                M[x, y] = cs
    rows, cols = linear_sum_assignment(-M)
    return [(gone[x], came[y], float(M[x, y])) for x, y in zip(rows, cols) if M[x, y] >= 0]


def _row(ctx, status, i, j, diff=None, claims=(), cost=None, appearance=None):
    """One object's line of the layer: A's record i and / or B's record j (indices into ctx's lists)."""
    r = ctx["r"]
    a = ctx["recA"][i] if i is not None else None
    b = ctx["recB"][j] if j is not None else None
    ca, cb = (a or {}).get("card"), (b or {}).get("card")
    row = {"status": status, "claims": list(claims), "a": ca["id"] if ca else None, "b": cb["id"] if cb else None,
           "site_id": "v0:" + ca["id"] if ca else "v1:" + cb["id"], "shot_a": r["a_shot"], "shot_b": r["b_shot"],
           "name_a": a["name"] if a else None, "name_b": b["name"] if b else None, "type_a": a["type"] if a else None, "type_b": b["type"] if b else None,
           "xy_a": np.round(a["xy"], 3).tolist() if a else None, "xy_b": np.round(b["xy"], 3).tolist() if b else None, "frame": "site map floor frame (A's)"}
    if diff:
        row["delta"] = diff
    if cost is not None:
        row["match_cost"] = round(cost, 3)
    cs = appearance if appearance is not None else (cos(a, b) if a and b else None)
    if cs is not None:
        row["appearance_cos"] = round(cs, 4)
    if i in ctx["placeB"]:
        row["place_in_b"] = _slim(ctx["placeB"][i])
    if j in ctx["placeA"]:
        row["place_in_a"] = _slim(ctx["placeA"][j])
    if cb is not None and ca is not None:  # carry-over: identity, A's name / type when B has none, A's model in B's frame
        carried = {"site_id": "v0:" + ca["id"]}
        if a["specific"] and not b["specific"]:
            carried["name"] = {"value": a["name"], "from": "v0:" + ca["id"], "reason": "matched to the site map's object, which has a name"}
        if a["fam"] not in GENERIC and b["fam"] in GENERIC:
            carried["type"] = {"value": a["type"], "family": a["fam"], "from": "v0:" + ca["id"]}
        m = carry_model(ca.get("model"), ctx["T_AB"])
        if m:
            carried["model"] = {**m, "frame": f"shot {r['b_shot']} of this visit", "from": "v0:" + ca["id"]}
        g = mesh_of(ctx["A"], ca["id"])
        if g:  # the site map's accepted SAM 3D mesh, placed in this visit's frame (a model is made once per site, not per visit)
            carried["mesh"] = {"sha256": g["sha256"], "transform": carry_model({**g["transform"], "kind": "mesh"}, ctx["T_AB"]),
                               "from": "v0:" + ca["id"]}
        row["carried"] = carried
    if status == "missing":  # the ghost: where it stood, in this visit's frame
        m = carry_model(ca.get("model"), ctx["T_AB"])
        if m:
            row["ghost_model"] = {**m, "frame": f"shot {r['b_shot']} of this visit", "from": "v0:" + ca["id"]}
    if status in CHANGES:
        row["_ev"] = (ctx, i, j)
    return row


CHANGES = ("moved", "new", "missing", "changed_height", "changed_angle")


def _slim(p):
    return {k: p.get(k) for k in ("state", "views", "free_views", "occupied_views", "best_key", "best_free_share", "reason") if p.get(k) is not None}


def _evidence(A, B, objects):
    """Per change, one JPEG: the site map's keyframe (left) and this visit's (right), the object's outline box where it was
    segmented, its projected box (dashed) where only the other visit's points fall."""
    import cv2
    out = []
    for k, o in enumerate(x for x in objects if "_ev" in x):
        ctx, i, j = o["_ev"]
        r = ctx["r"]
        (Pa, boxes_a) = (ctx["ptsA"].get(i) or card_points(A, ctx["recA"][i]["card"])) if i is not None else (None, {})
        (Pb, boxes_b) = (ctx["ptsB"].get(j) or card_points(B, ctx["recB"][j]["card"])) if j is not None else (None, {})
        fa_key, fb_key = next(iter(boxes_a), None), next(iter(boxes_b), None)
        box_a, box_b, dash_a, dash_b = boxes_a.get(fa_key), boxes_b.get(fb_key), False, False
        if o["status"] == "missing" and (o.get("place_in_b") or {}).get("best_key") is not None:  # B's clearest view of the empty place
            fb_key = o["place_in_b"]["best_key"]
            c2w, K, wh = cam_at(B, r["b_shot"], fb_key)
            box_b, dash_b = project_box(pts(ctx["T_AB"], Pa), c2w, K, wh, B["pick"]["data"]["source_wh"]), True
        if o["status"] == "new" and (o.get("place_in_a") or {}).get("best_key") is not None:  # A's clearest view of the then-empty place
            fa_key = o["place_in_a"]["best_key"]
            c2w, K, wh = cam_at(A, r["a_shot"], fa_key)
            box_a, dash_a = project_box(pts(ctx["T_BA"], Pb), c2w, K, wh, A["pick"]["data"]["source_wh"]), True
        panels = []
        for V, key, box, dash, label in ((A, fa_key, box_a, dash_a, "site map (A)"), (B, fb_key, box_b, dash_b, "this visit (B)")):
            img = decode(V, [key]).get(key) if key is not None else None
            if img is None:
                panels.append(np.full((TILE_H, TILE_W, 3), 40, np.uint8))
                continue
            sx, sy = TILE_W / img.shape[1], TILE_H / img.shape[0]
            p = cv2.resize(img, (TILE_W, TILE_H), interpolation=cv2.INTER_AREA)
            if box is not None:
                x0, y0, x1, y1 = [int(round(v)) for v in (box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy)]
                if dash:
                    for x in range(x0, x1, 12):
                        for y_ in (y0, y1):
                            cv2.line(p, (x, y_), (min(x + 6, x1), y_), (60, 60, 255), 2)
                    for y in range(y0, y1, 12):
                        for x_ in (x0, x1):
                            cv2.line(p, (x_, y), (x_, min(y + 6, y1)), (60, 60, 255), 2)
                else:
                    cv2.rectangle(p, (x0, y0), (x1, y1), (0, 200, 255), 2)
            cv2.putText(p, f"{label}  t={key / V['fps']:.1f} s", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 2, cv2.LINE_AA)
            panels.append(p)
        tile = np.hstack(panels)
        cap = f"{o['status']}: {o.get('name_a') or o.get('name_b') or ''}"[:80]
        tile = np.vstack([np.full((28, tile.shape[1], 3), 20, np.uint8), tile])
        cv2.putText(tile, cap, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1, cv2.LINE_AA)
        o["evidence"] = {"tile": f"evidence-{k}",
                         "a": {"frame": fa_key, "t": round(fa_key / A["fps"], 2) if fa_key is not None else None, "box": _r(box_a), "projected": dash_a},
                         "b": {"frame": fb_key, "t": round(fb_key / B["fps"], 2) if fb_key is not None else None, "box": _r(box_b), "projected": dash_b}}
        out.append((k, cv2.imencode(".jpg", tile, [cv2.IMWRITE_JPEG_QUALITY, 82])[1].tobytes()))
    return out


def _r(box):
    return None if box is None else [round(v, 1) for v in box]


# ---------------------------------------------------------------- GPU and entry points

class Gpu:
    """DINOv2-L (the naming cascade's encoder, GPU 1) and DA3-GIANT any-view (core.Da3, GPU 0), resident in the report container."""

    def __init__(self, da3, enc):
        self.da3_model, self.enc = da3, enc

    def embed(self, frames):
        import cv2
        import torch
        out = []
        for k in range(0, len(frames), 64):
            x = np.stack([cv2.resize(f, (DESC_SIDE, DESC_SIDE), interpolation=cv2.INTER_AREA)[..., ::-1] for f in frames[k:k + 64]])
            t = torch.from_numpy(np.ascontiguousarray(x)).to(self.enc.dev).permute(0, 3, 1, 2).float().div_(255)
            out.append(self.enc.dino_embed(t).cpu().numpy())
        return np.concatenate(out).astype(np.float32)

    def da3(self, frames):
        import torch
        kf = torch.from_numpy(np.stack(frames)).to(self.da3_model.dev)
        with torch.inference_mode():
            g = self.da3_model.shot(kf)
        return {"depth": g["depth"].float().cpu().numpy(), "c2w": g["c2w"].double().cpu().numpy(), "K": g["K"].float().cpu().numpy()}


class Replay:
    """GPU outputs recorded by Recorder, served in call order (the local CPU loop on mirrored reports)."""

    def __init__(self, path):
        z = np.load(path, allow_pickle=True)
        self.calls = list(z["calls"])

    def embed(self, frames):
        kind, out = self.calls.pop(0)
        assert kind == "embed" and len(out) == len(frames), "the recorded call does not match"
        return out

    def da3(self, frames):
        kind, out = self.calls.pop(0)
        assert kind == "da3" and len(out["depth"]) == len(frames), "the recorded call does not match"
        return out


class Recorder:
    def __init__(self, gpu):
        self.gpu, self.calls = gpu, []

    def embed(self, frames):
        out = self.gpu.embed(frames)
        self.calls.append(("embed", out))
        return out

    def da3(self, frames):
        out = self.gpu.da3(frames)
        self.calls.append(("da3", out))
        return out

    def save(self, path):
        arr = np.empty(len(self.calls), object)
        arr[:] = self.calls
        np.savez_compressed(path, calls=arr)


def run(root, a_report, b_report, gpu, writer=None, clock=None, site=None, b_frames=None):
    """Compare report b with the site map a (both under root); with a writer, the `visits` layer goes on b's report and the
    site's index (sites/<site>/visits.json) lists the visit. b_frames: b's decoded frames when the caller holds them (the
    analysis that just ran), so only the site map's video is decoded again. -> (data, blobs, record)."""
    A, B = load(root, a_report), load(root, b_report)
    if b_frames is not None:
        B["frames"].update({f: b_frames[f] for f in keyframes(B)[0] if f < len(b_frames)})
    data, blobs, rec = compare(A, B, gpu, clock)
    if writer is not None:
        writer.put("visits", data, blobs, "estimated+inferred", ["differences between two visits of one site: registered (estimated), matched (inferred)"])
    if site:
        p = Path(root) / "sites" / site / "visits.json"
        idx = json.loads(p.read_text()) if p.exists() else {"site": site, "site_map": a_report, "visits": []}
        idx["visits"] = [v for v in idx["visits"] if v["report"] != b_report] + [{"report": b_report, "registered_to": a_report, "counts": rec["counts"],
                                                                                 "accepted": rec["accepted"], "at_unix": time.time()}]
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(idx, indent=1))
    return data, blobs, rec


# ---------------------------------------------------------------- self-check

def self_check():
    rng = np.random.default_rng(0)
    rot = lambda v: __import__("cv2").Rodrigues(np.asarray(v, float))[0]  # noqa: E731
    # 1. registration: DA3's frame is a similarity G of the site map's; B's shot is T of A's. Both come back exactly.
    cams = lambda n: np.stack([np.block([[rot(rng.normal(0, .3, 3)), rng.normal(0, 1, (3, 1))], [np.zeros((1, 3)), np.ones((1, 1))]]) for _ in range(n)])  # noqa: E731
    kA, kB = cams(8), cams(6)
    T = {"s": 1.3, "R": rot([.05, -.4, .02]), "t": np.array([.4, -1., .2])}
    G = {"s": .37, "R": rot([.3, .1, -.2]), "t": np.array([2., .5, -1.])}
    Gi = inverse(G)
    predA, predB = move(Gi, kA), move(Gi, move(T, kB))
    dA, dB = rng.uniform(1, 4, (8, 70, 126)), rng.uniform(1, 4, (6, 70, 126))
    fA = fit(kA, predA, ratio_per_frame(dA, dA / G["s"]))
    assert abs(fA["s"] - G["s"]) < 1e-9 and fA["centre_residual_m"] < 1e-9 and passes(fA), fA
    carried = move(fA, predB)
    fB = fit(carried, kB, ratio_per_frame(fA["s"] * (T["s"] * dB / G["s"]), dB))
    assert abs(fB["s"] - T["s"]) < 1e-9 and np.allclose(fB["R"], T["R"]) and np.allclose(fB["t"], T["t"]) and passes(fB), fB
    noisy = predA.copy()
    noisy[:, :3, 3] += rng.normal(0, .5, (8, 3))
    assert not passes(fit(kA, noisy, ratio_per_frame(dA, dA / G["s"]))), "cameras that do not agree are refused"
    Tz = {"s": 1.1, "R": rot([0, 0, .7]), "t": np.array([1., 2., 0.])}
    tilt, dz, yaw, Ts = snap(Tz)
    assert tilt < 1e-6 and abs(dz) < 1e-9 and abs(yaw - .7) < 1e-9 and np.allclose(Ts["R"], Tz["R"])
    tilt, dz, *_ = snap(compose(Tz, {"s": 1., "R": rot([.1, 0, 0]), "t": np.array([0, 0, .3])}))  # B's floor tilted and raised
    assert abs(tilt - np.degrees(.1)) < 1e-6 and abs(dz - 1.1 * .3) < 1e-9, (tilt, dz)
    assert np.allclose(pts(inverse(T), pts(T, dA[0, :3, :3])), dA[0, :3, :3]) and np.allclose(pool_min(np.arange(16.).reshape(1, 4, 4), 2), [[[1, 2], [8, 10]]])
    # 2. the differences on a rendered room (fast_report.timeline's ray caster): static, moved, missing, new, changed height,
    # out of view, and a desk one visit delineated as two
    K, hw = np.array([[200., 0, 160], [0, 200, 120], [0, 0, 1]]), (240, 320)
    e = {k: v / np.linalg.norm(v) for k, v in {n: rng.normal(size=32) for n in "SMGNHUDP"}.items()}
    boxA = {"S": ([-1.2, -.3, 3.], [-.6, .3, 3.5]), "M": ([-.2, -.3, 3.], [.3, .3, 3.5]), "G": ([.8, -.3, 2.8], [1.2, .3, 3.2]),
            "H": ([2.6, -.2, 4.5], [3.4, .2, 4.8]), "U": ([4.5, -.3, 3.], [5., .3, 3.5]), "D": ([-1.2, -.3, 4.2], [0., .3, 4.6]),
            "P": ([-1., -1.2, 3.], [-.6, -.8, 3.3])}  # on a high shelf above S
    boxB = {"S": boxA["S"], "M": ([1.6, -.3, 3.], [2.1, .3, 3.5]), "N": ([-1.9, -.3, 3.2], [-1.5, .3, 3.6]), "H": ([2.6, -.6, 4.5], [3.4, .2, 4.8]),
            "U": boxA["U"], "D1": ([-1.2, -.3, 4.2], [-.6, .3, 4.6]), "D2": ([-.6, -.3, 4.2], [0., .3, 4.6]), "P": boxA["P"]}

    def window(boxes, xs):
        c2w = np.stack([timeline._cam(x) for x in xs])
        return {"keys": list(range(len(xs))), "c2w": c2w, "K": np.repeat(K[None], len(xs), 0),
                "depth": np.stack([timeline._render(list(boxes.values()), c, K, hw) for c in c2w]), "person": np.zeros((len(xs), *hw), bool)}

    def rec(name, lo, hi):
        lo, hi = np.array(lo), np.array(hi)
        g = np.stack(np.meshgrid(*[np.linspace(a, b, 9) for a, b in zip(lo, hi)]), -1).reshape(-1, 3)
        foot = np.array([[lo[0], lo[2]], [hi[0], lo[2]], [hi[0], hi[2]], [lo[0], hi[2]]])
        v = e[name[0]] + (.3 * e["N"] if name == "D2" else 0)
        return {"xy": np.array([(lo[0] + hi[0]) / 2, (lo[2] + hi[2]) / 2]), "u_xy": .05, "top": .5 - lo[1], "base": .5 - hi[1], "u_top": .03,
                "u_base": .03, "top_ok": True, "base_ok": True, "size": float(np.max(hi - lo)), "foot": foot, "fam": "goods" if name == "N" else "furniture",
                "angles": {}, "name": name, "type": None, "specific": False, "dino": v / np.linalg.norm(v), "card": {"id": name},
                "points": g[g[:, 2] <= lo[2] + 1e-9]}
    wA, wB = window(boxA, [0., .1, .2, 3.8, 4.]), window(boxB, [0., .1, .2])
    RA, RB = [rec(k, *b) for k, b in boxA.items()], [rec(k, *b) for k, b in boxB.items() if k != "U"]  # B's cameras never see U
    pB = next(r_ for r_ in RB if r_["name"] == "P")  # B saw only P's lower half (a partial view): its top reads 0.2 m low, nothing changed
    pB.update(top=1.5, points=pB["points"][.5 - pB["points"][:, 1] <= 1.5])
    got = classify(RA, RB, .05, lambda i: RA[i]["points"], lambda j: RB[j]["points"],
                   lambda P: timeline.place(P, wB, pose_m=.05), lambda P: timeline.place(P, wA, pose_m=.05), lambda P: .5 - P[:, 1], lambda P: .5 - P[:, 1])
    st = {RA[i]["name"]: s_ for i, s_ in got["status_a"].items()} | {RB[j]["name"]: s_ for j, s_ in got["status_b"].items()}
    pairs = {(RA[i]["name"], RB[j]["name"]) for i, j, *_ in got["pairs"]}
    moves = {(RA[i]["name"], RB[j]["name"]) for i, j, _ in got["moves"]}
    assert ("S", "S") in pairs and ("H", "H") in pairs and moves == {("M", "M")}, (pairs, moves)
    assert st["G"] == "missing" and st["N"] == "new" and st["U"] == "not_observed_in_b", st
    assert any(p_[0] == "D" for p_ in pairs) and st[({"D1", "D2"} - {p_[1] for p_ in pairs}).pop()] == "delineated_otherwise", (pairs, st)
    h = next(x for x in got["pairs"] if RA[x[0]]["name"] == "H")
    assert h[6] == ["changed_height"] and abs(h[5]["top_above_floor"]["value"] - .4) < 1e-9, h
    pp = next(x for x in got["pairs"] if RA[x[0]]["name"] == "P")
    assert pp[6] == [] and pp[7] == ["changed_height"], ("a partial view's lower top is not a change: its upper part is still seen there", pp)
    assert differences(RA[0], RB[0])[1] == [], "a static object claims nothing"
    rows = one_row_per_object([{"a": "x", "b": None, "status": "missing"}, {"a": "x", "b": "y", "status": "static"}, {"a": None, "b": "z", "status": "new"},
                               {"a": "x", "b": "y2", "status": "static"}, {"a": "w", "b": None, "status": "not_observed_in_b"},
                               {"a": "w", "b": None, "status": "missing"}])
    assert [(r_["status"], r_["b"]) for r_ in rows] == [("static", "y"), ("new", "z"), ("static", "y2"), ("missing", None)], rows
    m = carry_model({"kind": "box", "position": [1., 0, 0], "quaternion": [0, 0, 0, 1.], "size_m": [1., 2, 3]}, T)
    assert np.allclose(m["position"], pts(T, np.array([[1., 0, 0]]))[0], atol=1e-4) and np.allclose(m["size_m"], [1.3, 2.6, 3.9])
    print("visits self-check ok: registration recovers both similarities and refuses disagreeing cameras, floor snap; on a rendered "
          "room: static, moved (one identity), missing, new, changed height (+0.40 m, its new part seen through by the first visit), "
          "a partial view's lower top unconfirmed, out of view -> not observed, a desk split in two -> delineated otherwise")


if __name__ == "__main__":
    import sys
    assert sys.argv[1:] == ["--self-check"], __doc__
    self_check()
