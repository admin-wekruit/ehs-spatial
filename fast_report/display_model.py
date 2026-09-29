"""r4 (models): every object card's display model, 'generated, display only' (never a measurement: the card's physical fields
stay the facts). A primitive fitted to the object's observed points in the shot's floor frame (z up, the floor at z = 0),
chosen by the card's type and by the fit residual (the median distance of the points to the primitive's surface):
  box         gravity-aligned, yaw by the least median surface distance (X7's parametric box), extents at the points' p2 / p98
              (the card's robust box)
  cylinder    vertical, or along the points' long axis; the circle by an algebraic fit on the section (poles, pipes, bins)
  plane       any orientation (the points' PCA), a slab as thick as the points are (panels, doors, boards)
  open frame  posts at the footprint's corners (2 when it is flat) and a plate at each height the points pile up at (shelves,
              racks, ladders, carts, tables)
A face (a frame member, a 10 deg sector of a cylinder) is 'seen' when a camera that saw the object faced it and points lie on
it; the rest is 'guessed' (drawn translucent). Nothing is added where no point was: an object seen from one side is as deep
as its visible part (a lower bound), never given an invented thickness.

The record's pose is in the shot's frame (the viewer's 'shot-<i>'): position, quaternion (x, y, z, w), local sizes.

    python -m fast_report.display_model --self-check
"""
import numpy as np

STATUS = "generated, display only"
MAX_POINTS, MIN_T = 3000, .01  # points a fit uses (a seeded sample); the thinnest drawn side (m)
SEEN_COS = np.cos(np.radians(75))  # a camera faces a face when it is less than 75 deg off the face's normal
SECTORS = 36  # a cylinder's arc in 10 deg sectors
PREFER_BOX = {"plane": .7, "cylinder": .5}  # without a type another shape wins only below this share of the box's residual
MIN_ARC = 9  # ... and a cylinder only when its circle was fitted on >= 90 deg of seen arc (offline dev: bags and shoes went round)
TYPE_SLACK_M = .03  # the type's shape stays unless its residual is over 2 x the best one's and 3 cm more
TYPE_KIND = {
    "cylinder": ("drum", "bucket", "trash can", "bottle", "can", "fire extinguisher", "column", "bollard", "safety cone", "pipe", "duct",
                 "cup", "pole", "barrel", "tank", "cylinder", "roll", "paper towel roll", "toilet paper roll"),
    "plane": ("door", "window", "wall panel", "whiteboard", "sign", "safety sign", "exit sign", "label", "wooden board", "metal sheet", "monitor",
              "curtain", "clipboard", "paper", "floor marking", "fence", "barrier", "mat", "board", "partition"),  # not 'panel': a control panel is a box
    "open frame": ("shelf", "rack", "display rack", "ladder", "cart", "hand truck", "step stool", "work platform", "stairs", "table", "desk",
                   "workbench", "chair", "stool", "railing", "cable tray", "pallet jack"),  # not 'tool holder': ME340's are CNC tool cones
}
FACES = ("-x", "+x", "-y", "+y", "-z", "+z")


def sample(P, n=MAX_POINTS):
    P = np.asarray(P, float)
    return P[np.sort(np.random.default_rng(0).choice(len(P), n, replace=False))] if len(P) > n else P


def box_distance(L, lo, hi):
    """Distance of local points to the surface of the box [lo, hi] (inside points: to the nearest face)."""
    c, h = (lo + hi) / 2, (hi - lo) / 2
    d = np.abs(L - c) - h
    return np.abs(np.linalg.norm(np.maximum(d, 0), axis=1) + np.minimum(d.max(1), 0))


def yaw_axes(deg):
    x = np.array([np.cos(np.radians(deg)), np.sin(np.radians(deg)), 0.])
    return np.stack([x, np.cross([0, 0, 1.], x), [0, 0, 1.]])


def extents(L):
    lo, hi = np.percentile(L, 2, 0), np.percentile(L, 98, 0)
    return lo, np.maximum(hi, lo + MIN_T)


def box_faces(L, lo, hi, cams_l):
    """'seen' / 'guessed' per face (FACES order): a camera faced it and >= 2 % of the points lie on it."""
    tol = np.maximum(.02, .1 * (hi - lo))
    out = {}
    for i, name in enumerate(FACES):
        k, s = i // 2, (1. if i % 2 else -1.)
        n = np.eye(3)[k] * s
        fc = (lo + hi) / 2
        fc[k] = hi[k] if s > 0 else lo[k]
        d = cams_l - fc
        faced = len(d) and ((d @ n) / np.maximum(np.linalg.norm(d, axis=1), 1e-9) > SEEN_COS).any()
        on = (np.abs(L[:, k] - fc[k]) <= tol[k]).sum() >= max(3, .02 * len(L))
        out[name] = "seen" if faced and on else "guessed"
    return out


def fit_box(P, cams):
    """X7's parametric box: yaw 0-87 deg by 3, the one with the least median surface distance; p2 / p98 extents."""
    best = None
    for deg in range(0, 90, 3):
        A = yaw_axes(deg)
        L = P @ A.T
        lo, hi = extents(L)
        r = float(np.median(box_distance(L, lo, hi)))
        if best is None or r < best[0]:
            best = (r, A, lo, hi, L)
    r, A, lo, hi, L = best
    return {"kind": "box", "axes": A, "lo": lo, "hi": hi, "residual_m": r, "faces": box_faces(L, lo, hi, cams @ A.T)}


def fit_plane(P, cams):
    """A slab on the points' PCA axes (u horizontal where the plane is not flat on the floor, v in the plane, n its normal)."""
    c0 = np.median(P, 0)
    n = np.linalg.svd(P - c0, full_matrices=False)[2][2]
    u = np.cross(n, [0, 0, 1.])
    u = u / np.linalg.norm(u) if np.linalg.norm(u) > .3 else np.cross([0, 1., 0], n) / max(np.linalg.norm(np.cross([0, 1., 0], n)), 1e-9)
    A = np.stack([u, np.cross(n, u), n])
    L = P @ A.T
    lo, hi = extents(L)
    return {"kind": "plane", "axes": A, "lo": lo, "hi": hi, "residual_m": float(np.median(box_distance(L, lo, hi))),
            "faces": box_faces(L, lo, hi, cams @ A.T)}


def circle(xy):
    """Algebraic (Kasa) circle through 2d points -> (centre, radius) or None."""
    M = np.c_[2 * xy, np.ones(len(xy))]
    try:
        (a, b, c), *_ = np.linalg.lstsq(M, (xy ** 2).sum(1), rcond=None)
    except np.linalg.LinAlgError:
        return None
    r2 = c + a * a + b * b
    return (np.array([a, b]), float(np.sqrt(r2))) if np.isfinite(r2) and r2 > 0 else None


def axis_frame(a):
    """Right-handed rows (e1, e2, a) for a unit axis a."""
    a = np.asarray(a, float) / np.linalg.norm(a)
    e1 = np.cross(a, [0, 0, 1.]) if abs(a[2]) < .95 else np.cross(a, [1., 0, 0])
    e1 /= np.linalg.norm(e1)
    return np.stack([e1, np.cross(a, e1), a])


def fit_cylinder(P, cams):
    """Vertical axis, or along the points' long axis (a pipe, lying or sloping): the section's circle. A fit whose radius
    exceeds the section's extent, or whose centre sits on the cameras' side (a concave face), falls back to the section's width
    across the view as the diameter, the centre one radius behind the near side ('fitted' False: drawn for a cylinder-like type
    only). The residual is
    the distance to the cylinder's surface, caps included. The better of the two axes by residual; points 2.5 x longer one way
    than the next (a pipe, a pole) take their long axis (ME340's ceiling pipes went vertical as flat discs)."""
    best = None
    d = P - P.mean(0)
    ev, vec = np.linalg.eigh(d.T @ d)
    long = vec[:, 2]
    elongated = ev[2] >= 2.5 ** 2 * max(ev[1], 1e-12)
    for axis, A in (("long axis", axis_frame(long)),) if elongated else (("vertical", yaw_axes(0.)), ("long axis", axis_frame(long))):
        L, C = P @ A.T, cams @ A.T
        sec = L[:, :2]
        slo, shi = np.percentile(sec, 2, 0), np.percentile(sec, 98, 0)
        got = circle(sec)
        cam = C[:, :2].mean(0) if len(C) else None
        fitted = not (got is None or got[1] > float((shi - slo).max()) or
                      (cam is not None and np.linalg.norm(cam - got[0]) < np.median(np.linalg.norm(sec - got[0], axis=1))))
        if fitted:
            c, r = got
        else:  # the section's width across the view is the diameter (always seen); the centre one radius behind the near side
            m = np.median(sec, 0)
            v = (cam - m) / max(np.linalg.norm(cam - m), 1e-9) if cam is not None else np.array([0, -1.])
            w = np.array([-v[1], v[0]])
            wl, wh = np.percentile(sec @ w, [2, 98])
            r = float((wh - wl) / 2)
            c = w * (wl + wh) / 2 + v * (float(np.percentile(sec @ v, 98)) - r)
        r = max(r, MIN_T / 2)
        zlo, zhi = extents(L[:, 2:])
        dd = np.c_[np.linalg.norm(sec - c, axis=1) - r, np.abs(L[:, 2] - (zlo[0] + zhi[0]) / 2) - (zhi[0] - zlo[0]) / 2]
        res = float(np.median(np.abs(np.linalg.norm(np.maximum(dd, 0), axis=1) + np.minimum(dd.max(1), 0))))
        if best is None or res < best["residual_m"]:
            ang = np.degrees(np.arctan2(*(sec - c).T[::-1])) % 360
            cnt = np.bincount((ang // (360 / SECTORS)).astype(int).clip(0, SECTORS - 1), minlength=SECTORS)
            near = np.linalg.norm(sec - c, axis=1) <= r + max(.02, .1 * r)
            caps = {}
            for name, z, s in (("bottom", zlo[0], -1.), ("top", zhi[0], 1.)):
                faced = len(C) and ((C[:, 2] - z) * s > 0).any()
                caps[name] = "seen" if faced and ((np.abs(L[:, 2] - z) <= max(.02, .1 * (zhi[0] - zlo[0]))) & near).sum() >= max(3, .02 * len(L)) else "guessed"
            best = {"kind": "cylinder", "axis": axis, "axes": A, "centre": c, "radius": r, "zlo": float(zlo[0]), "zhi": float(zhi[0]), "fitted": fitted,
                    "residual_m": res, "arc": "".join("1" if n_ >= max(2, .005 * len(L)) else "0" for n_ in cnt), "caps": caps}
    return best


def fit_frame(P, cams, box):
    """Open frame on the box's axes: posts at the footprint's corners (2 at the ends of the long side when the footprint is
    flat: a ladder, a fence-like rack), a plate at each height where the points pile up (a 2 cm histogram's peaks over
    1.5 x its mean, >= 8 cm apart, up to 8); each member 'seen' when >= 1 % of the points lie on it."""
    A, lo, hi = box["axes"], box["lo"], box["hi"]
    L = P @ A.T
    side = hi[:2] - lo[:2]
    short, lk = float(side.min()), int(np.argmax(side))
    t = float(np.clip(.08 * max(short, .1), .02, .06))
    flat = short < max(.15, .15 * float(side.max()))
    corners = [(a, b) for a in (lo[0] + t / 2, hi[0] - t / 2) for b in (lo[1] + t / 2, hi[1] - t / 2)]
    if flat:
        mid = (lo + hi) / 2
        corners = [tuple(np.where(np.arange(2) == lk, e, mid[:2])) for e in (lo[lk] + t / 2, hi[lk] - t / 2)]
    members = [(np.r_[x - t / 2, y - t / 2, lo[2]], np.r_[x + t / 2, y + t / 2, hi[2]], "post") for x, y in corners]
    z = L[:, 2]
    bins = np.arange(lo[2], hi[2] + .02, .02)
    h = np.convolve(np.histogram(z, bins)[0], [1, 1, 1], "same") if len(bins) > 2 else np.array([len(z)])
    peaks = [i for i in np.argsort(-h) if h[i] >= 1.5 * h.mean() and (i == 0 or h[i] >= h[i - 1]) and (i == len(h) - 1 or h[i] >= h[i + 1])]
    levels = []
    for i in peaks:
        zc = float(np.clip((bins[i] + bins[min(i + 1, len(bins) - 1)]) / 2, lo[2] + t / 2, hi[2] - t / 2))
        if all(abs(zc - q) >= .08 for q in levels):
            levels.append(zc)
        if len(levels) == 8:
            break
    members += [(np.r_[lo[0], lo[1], q - t / 2], np.r_[hi[0], hi[1], q + t / 2], "level") for q in sorted(levels)]
    D = np.stack([box_distance(L, a, b) for a, b, _ in members], 1)
    tol = max(.02, t)
    seen = [bool((D[:, j] <= tol).sum() >= max(3, .01 * len(L))) for j in range(len(members))]
    return {"kind": "open frame", "axes": A, "lo": lo, "hi": hi, "residual_m": float(np.median(D.min(1))), "t": t, "flat": bool(flat),
            "members": [(a, b, what, s) for (a, b, what), s in zip(members, seen)]}


def fits(P, cams):
    """Every candidate primitive for these floor-frame points; cams: the floor-frame centres of the cameras that saw them."""
    P, cams = sample(P), np.asarray(cams, float).reshape(-1, 3)
    box = fit_box(P, cams)
    return {"box": box, "cylinder": fit_cylinder(P, cams), "plane": fit_plane(P, cams), "open frame": fit_frame(P, cams, box)}


def type_kind(cls):
    from fast_report.cards import head_match
    if not cls:
        return None
    for kind, words in TYPE_KIND.items():
        if head_match(cls, words):
            return kind
    return None


def choose(fs, cls=None):
    """-> (kind, why). The type's shape (TYPE_KIND on the class) unless the points clearly fit another better; without a type,
    the box unless the plane or a fitted cylinder (MIN_ARC) fits under PREFER_BOX x its residual (an open frame only by type)."""
    r = {k: f["residual_m"] for k, f in fs.items()}
    cyl = fs["cylinder"].get("fitted") and fs["cylinder"]["arc"].count("1") >= MIN_ARC
    ok = [k for k in ("cylinder", "plane") if (k == "plane" or cyl) and r[k] < PREFER_BOX[k] * r["box"]]
    best = min(ok, key=r.get) if ok else "box"
    want = type_kind(cls)
    if want is None:
        return best, "residual" if best != "box" else "default (the box; no other shape fits the points clearly better)"
    if r[want] <= max(2 * r[best], r[best] + TYPE_SLACK_M):
        return want, f"type ({cls})"
    return best, f"residual: a {cls} would be a {want}, but it fits the points worse ({r[want] * 100:.1f} vs {r[best] * 100:.1f} cm)"


def world_pose(A, c_local, fr):
    """Local axes (rows, floor frame) and a local centre -> the shot-frame position and quaternion (x, y, z, w)."""
    from scipy.spatial.transform import Rotation
    Rw = fr["R"].T  # floor -> shot
    M = Rw @ A.T
    return (Rw @ (A.T @ c_local) + fr["origin"]).tolist(), Rotation.from_matrix(M).as_quat().tolist()


def record(fs, kind, why, fr, depth_seen=True, n=None):
    """The card's 'model' (JSON): the chosen primitive's pose and sizes, what was seen and guessed, every candidate's residual."""
    f = fs[kind]
    rnd = lambda v, d=3: np.round(np.asarray(v, float), d).tolist()  # noqa: E731
    out = {"status": STATUS, "kind": kind, "source": "primitive fit to the observed points", "chosen_by": why,
           "residual_m": round(f["residual_m"], 4), "fits_m": {k: round(v["residual_m"], 4) for k, v in fs.items()}, "frame": "shot", "points": n}
    if kind == "cylinder":
        c = np.r_[f["centre"], (f["zlo"] + f["zhi"]) / 2]
        pos, q = world_pose(f["axes"], c, fr)
        out.update(axis=f["axis"], radius_m=round(f["radius"], 3), length_m=round(f["zhi"] - f["zlo"], 3), arc_seen=f["arc"], caps=f["caps"],
                   seen_share=round(f["arc"].count("1") / SECTORS, 2))
    else:
        lo, hi = f["lo"], f["hi"]
        c = (lo + hi) / 2
        pos, q = world_pose(f["axes"], c, fr)
        out["size_m"] = rnd(hi - lo)
        if kind == "open frame":
            out["parts"] = [rnd([*((a + b) / 2 - c), *(b - a)], 3) + [1 if s else 0] for a, b, _, s in f["members"]]
            out["seen_share"] = round(float(np.mean([s for *_, s in f["members"]])), 2)
        else:
            out["faces"] = f["faces"]
            out["seen_share"] = round(sum(v == "seen" for v in f["faces"].values()) / 6, 2)
    out.update(position=rnd(pos), quaternion=rnd(q, 6))
    if not depth_seen:
        out["depth"] = "seen from one side: the far side is guessed; the drawn depth is the visible part (a lower bound), no thickness is invented"
    return out


def compact(fs):
    """The fits as a card keeps them (raw['model_fits']): JSON, enough for record() after a rename."""
    r = lambda v: np.round(v, 4).tolist()  # noqa: E731
    out = {}
    for k, f in fs.items():
        g = {kk: (r(v) if isinstance(v, np.ndarray) else v) for kk, v in f.items() if kk not in ("members",)}
        if k == "open frame":
            g["members"] = [[r(a), r(b), w, s] for a, b, w, s in f["members"]]
        out[k] = g
    return out


def raw_fields(P, cams, fr, depth_seen):
    """What a card keeps (in raw) to draw its model after any rename: the fits, the shot's floor frame, one side or not."""
    return {"model_fits": compact(fits(P, cams)), "model_frame": {"R": np.round(fr["R"], 6).tolist(), "origin": np.round(fr["origin"], 4).tolist()},
            "depth_seen": bool(depth_seen), "model_points": int(len(P))}


def for_card(raw, cls):
    """cards.apply_name's step: the card's 'model' for its current class (the name decides the type; the fits do not change)."""
    if not raw.get("model_fits"):
        return {"status": STATUS, "kind": None, "reason": "no model: seen in 2D only (no 3D points to fit a primitive to)"}
    fs = expand(raw["model_fits"])
    fr = {k: np.asarray(v, float) for k, v in raw["model_frame"].items()}
    return record(fs, *choose(fs, cls), fr, raw.get("depth_seen", True), raw.get("model_points"))


def expand(fs):
    out = {}
    for k, f in fs.items():
        g = {kk: (np.asarray(v, float) if kk in ("axes", "lo", "hi", "centre") else v) for kk, v in f.items()}
        if k == "open frame":
            g["members"] = [(np.asarray(a, float), np.asarray(b, float), w, s) for a, b, w, s in f["members"]]
        out[k] = g
    return out


WELL_OBSERVED = ("SAM 3D goes to the well-observed cards only, best first (views x angular size): >= 3 views >= 15 deg apart in "
                 "azimuth, never cut by the frame edge (top, base, width), >= 0.2 m and <= 3 m, >= 0.12 rad across at its nearest view, "
                 "its points one cluster (dropped share < 0.3), not an open frame (shelves and racks failed every generator in X7)")


def well_observed(card):
    """-> (score or None, why not): SAM 3D's eligibility on a card (WELL_OBSERVED)."""
    v, ph, raw = card.get("views") or {}, card.get("physical") or {}, card.get("raw") or {}
    size = (raw.get("size") or {}).get("longest")
    near = (v.get("distance_m") or [None])[0]
    checks = [(v.get("n", 0) >= 3, "fewer than 3 views"), ((v.get("azimuth_spread_deg") or 0) >= 15, "views within 15 deg"),
              (not any((ph.get(n) or {}).get("status") == st for n, st in (("top_above_floor", "at least"), ("base_above_floor", "at most"),
                                                                            ("width", "at least"))), "cut by the frame edge"),
              (size is not None and .2 <= size <= 3., "size outside 0.2-3 m"), (bool(near) and size is not None and size / near >= .12, "small in view"),
              ((ph.get("dropped_share") or 0) < .3, "fragmented points"), ((card.get("model") or {}).get("kind") != "open frame", "an open frame")]
    bad = [why for ok, why in checks if not ok]
    return (None, "; ".join(bad)) if bad else (round(float(v["n"] * size / near), 3), None)


# ---------------------------------------------------------------- self-check (synthetic points, one check per fitter)
def _box_points(size, yaw, faces, rng, n=600, noise=.004):
    A = yaw_axes(yaw)
    pts = []
    for name in faces:
        i = FACES.index(name)
        k, s = i // 2, (1. if i % 2 else -1.)
        q = (rng.random((n, 3)) - .5) * size
        q[:, k] = s * size[k] / 2
        pts.append(q)
    L = np.concatenate(pts) + rng.normal(0, noise, (n * len(faces), 3))
    L[:, 2] += size[2] / 2
    return L @ A + [1., 2., 0.]


def self_check():
    rng = np.random.default_rng(3)
    fr = {"R": np.eye(3), "origin": np.zeros(3)}
    # box: a carton seen from the front only, yaw 30 deg: its width and height, the visible depth, the far faces guessed
    size = np.array([.6, .4, .5])
    P = _box_points(size, 30., ("-y",), rng)
    cam = (np.array([0., -3., 1.6]) @ yaw_axes(30.)) + [1., 2., 0.]
    b = fit_box(P, cam[None])
    got = b["hi"] - b["lo"]
    assert abs(got[0] - .6) < .05 and abs(got[2] - .5) < .05, got  # p2 / p98: 4 % inside a uniform face
    assert got[1] < .05, "seen from the front: the depth is the visible part, not an invented slab"
    assert b["faces"]["-y"] == "seen" and b["faces"]["+y"] == "guessed" and b["faces"]["-z"] == "guessed", b["faces"]
    P2 = _box_points(size, 30., ("-y", "+x", "+z"), rng)
    b2 = fit_box(P2, cam[None])
    assert np.allclose(b2["hi"] - b2["lo"], size, atol=.05), b2["hi"] - b2["lo"]
    assert b2["residual_m"] < .01
    # cylinder: a drum (r 0.3 m, 0.9 m tall) seen on its front half: the radius, the vertical axis, the back arc guessed
    th = rng.uniform(-np.pi, 0, 2000)
    zz = rng.uniform(0, .9, 2000)
    D = np.c_[.3 * np.cos(th) + 2, .3 * np.sin(th) + 1, zz] + rng.normal(0, .004, (2000, 3))
    cams = np.array([[2., -3., 1.6]])
    c = fit_cylinder(D, cams)
    assert c["axis"] == "vertical" and abs(c["radius"] - .3) < .03 and np.allclose(c["centre"], [2, 1], atol=.04), (c["radius"], c["centre"])
    assert .4 <= c["arc"].count("1") / SECTORS <= .6 and c["caps"]["bottom"] == "guessed", c["arc"]
    assert c["fitted"]
    fs = fits(D, cams)
    assert choose(fs, None)[0] == "cylinder" and choose(fs, "barrel")[0] == "cylinder"
    # a pipe (r 5 cm, 2 m) sloping 30 deg, seen on its near half: the long axis, its radius
    a = np.array([np.cos(np.radians(30)), 0, np.sin(np.radians(30))])
    E = axis_frame(a)
    th, t_ = rng.uniform(np.pi / 2, 3 * np.pi / 2, 2000), rng.uniform(0, 2, 2000)
    Pp = (np.c_[.05 * np.cos(th), .05 * np.sin(th), t_] @ E) + [0, 3, .5] + rng.normal(0, .002, (2000, 3))
    cp = fit_cylinder(Pp, np.array([[1., 0., 1.6]]))
    assert cp["axis"] == "long axis" and abs(abs(cp["axes"][2] @ a) - 1) < .01 and abs(cp["radius"] - .05) < .01, (cp["axis"], cp["radius"])
    assert choose(fits(P2, cam[None]), None)[0] == "box"
    # plane: a door leaning 10 deg from vertical: the plane's normal, a thin slab, the type picks it
    u = rng.uniform(-.45, .45, 3000)
    v = rng.uniform(0, 2., 3000)
    tilt = np.radians(10)
    Q = np.c_[u, v * np.sin(tilt), v * np.cos(tilt)] + rng.normal(0, .003, (3000, 3))
    p = fit_plane(Q, np.array([[0., -3., 1.6]]))
    assert abs(abs(p["axes"][2] @ [0, np.cos(tilt), -np.sin(tilt)]) - 1) < .01 and (p["hi"] - p["lo"])[2] < .03
    assert p["residual_m"] < fit_box(Q, np.zeros((0, 3)))["residual_m"] * PREFER_BOX["plane"]
    assert choose(fits(Q, np.array([[0., -3., 1.6]])), "door")[0] == "plane"
    # open frame: a 4-level shelf (1.2 x 0.4 x 1.8 m, posts + plates): its levels found, the frame chosen by type, members seen
    pts = [np.c_[rng.uniform(0, 1.2, 500), rng.uniform(0, .4, 500), np.full(500, z)] for z in (.1, .6, 1.1, 1.7)]
    pts += [np.c_[np.full(300, x), np.full(300, y), rng.uniform(0, 1.8, 300)] for x in (0., 1.2) for y in (0., .4)]
    S = np.concatenate(pts) + rng.normal(0, .004, (3200, 3))
    fs = fits(S, np.array([[.6, -3., 1.6]]))
    f = fs["open frame"]
    lv = sorted(round(float((a[2] + b[2]) / 2), 1) for a, b, w, _ in f["members"] if w == "level")
    assert lv == [.1, .6, 1.1, 1.7], lv
    assert f["residual_m"] < .7 * fs["box"]["residual_m"] and all(s for *_, s in f["members"]) and not f["flat"]
    assert choose(fs, "shelf") == ("open frame", "type (shelf)") and choose(fs, None)[0] != "open frame"
    # the record: a pose in the shot frame that puts the box back where the points are; JSON round trip (compact / expand)
    import json
    fr2 = {"R": yaw_axes(20.), "origin": np.array([.5, -.2, .1])}
    Pw = P2 @ fr2["R"] + fr2["origin"]  # the same box in a shot frame
    fsw = expand(json.loads(json.dumps(compact(fits(P2, cam[None])))))
    rec = record(fsw, "box", "type (box)", fr2, n=len(P2))
    assert np.allclose(rec["position"], np.median(Pw, 0) * [1, 1, 0] + [0, 0, rec["position"][2]], atol=.15), rec["position"]
    from scipy.spatial.transform import Rotation
    corners = (np.array([[i, j, k] for i in (-.5, .5) for j in (-.5, .5) for k in (-.5, .5)]) * rec["size_m"]) @ Rotation.from_quat(rec["quaternion"]).as_matrix().T + rec["position"]
    assert np.allclose(corners.min(0), np.percentile(Pw, 0, 0), atol=.06) and np.allclose(corners.max(0), np.percentile(Pw, 100, 0), atol=.06)
    assert rec["status"] == STATUS and rec["faces"]["-z"] == "guessed"
    rec1 = record(fsw, "open frame", "type", fr2, depth_seen=False)
    assert len(rec1["parts"]) >= 4 and "lower bound" in rec1["depth"] and len(json.dumps(rec1)) < 2000
    # SAM 3D's eligibility: a well-seen carton goes, the same cut by the frame edge or seen from one direction does not
    card = {"views": {"n": 5, "azimuth_spread_deg": 40, "distance_m": [1.5, 3.]}, "physical": {"top_above_floor": {"value": 1.}, "dropped_share": .05},
            "raw": {"size": {"longest": .6}}, "model": {"kind": "box"}}
    assert well_observed(card) == (2.0, None)
    assert well_observed({**card, "physical": {"top_above_floor": {"status": "at least"}}})[1] == "cut by the frame edge"
    assert "views within 15 deg" in well_observed({**card, "views": {**card["views"], "azimuth_spread_deg": 5}})[1]
    # degenerate clouds (8 points, one point, flat, a vertical line, sub-millimetre; no camera) give finite poses for every kind
    for t in range(60):
        D = rng.normal(size=(8 + t % 20, 3)) * rng.uniform(0, 1, 3)
        D = [D, D * 0 + D[0], D * [1, 1, 0], D * [0, 0, 1], D * 1e-4][t % 5]
        fsd = fits(D, rng.normal(size=(t % 3, 3)) * 3)
        assert all(np.isfinite(record(fsd, k, "t", fr, n=len(D))["quaternion"]).all() for k in fsd)
    print("display_model self-check ok: box (yaw, sizes, visible depth, faces), cylinder (radius, axis, arc), plane (tilted normal), "
          "open frame (levels, members), choice by type and residual, shot-frame pose")


if __name__ == "__main__":
    import sys
    assert sys.argv[1:] == ["--self-check"], __doc__
    self_check()
