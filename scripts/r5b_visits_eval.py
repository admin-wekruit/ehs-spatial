"""r5b (visits): the visits layer against ground truth, and the results page.

  registration  B's keyframe cameras as the layer registers them into A's map, against where GT puts them there: A's shot
                frame -> the GT world by the Sim3 of A's cameras onto A's GT cameras (S_A, accuracy_gt's rule), B's GT
                cameras carried back by S_A^-1. Error in GT metres (cm) and degrees. TUM fr1: every sequence shares one
                motion-capture frame; ARKit sessions do not (their pair is a static revisit: false changes only).
  objects       every card's GT points: its pick masks on every segmented keyframe, eroded 1 px, lifted with GT depth and the
                GT pose (accuracy_gt.regions' rule, per-view depth tails trimmed); 5 cm voxels. A GT pair = two cards of the two
                visits that are each other's best voxel IoU, IoU >= GT_IOU. A predicted pair is right when its two cards'
                GT points share >= GT_SHARE of the smaller one's voxels (the same place in the world).
  changes       a card's place in the other visit by GT depth and GT poses (the see-through test of fast_report.timeline on
                GT depth, 1 cm pose u): free / occupied / occluded / out of view. A 'missing' / 'new' claim is right when GT
                sees the place free; a GT change (for recall) is a card of one visit with >= GT_MIN_POINTS GT points whose
                place GT sees free in the other.

    python scripts/r5b_visits_eval.py registration RUN_DIR [--gt-root DIR]      # local: poses only
    python scripts/r5b_visits_eval.py --self-check
Modal (GT depth stays on the Volume): modal run modal_apps/r5b_visit_data.py::gt --pairs PAIRS.json --out OUT
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "scripts")]
GT_IOU, GT_SHARE, GT_MIN_POINTS, VOX_M = .3, .3, 50, .05
GT_POSE_M, GT_REL, GT_ABS_M = .01, .05, .05
LOCAL_KEYS = 15  # 'local' registration error: A's map near the views the registration used (A's own drift elsewhere left out)
PHASE2 = Path("/Users/adam/Desktop/panoptes-public/research-notes/phase2")


# ---------------------------------------------------------------- GT per card (Modal: GT depth on the Volume)

def card_points_gt(V, seq, a0, hold, cap=3000):
    """{card id: (n, 3) GT world points} from every segmented pick frame of report V (fast_report.visits.load)."""
    import cv2
    from fast_report import ondemand
    from accuracy_gt import tail_trim
    rng = np.random.default_rng(0)
    out = {}
    sw, sh = V["pick"]["data"]["source_wh"]
    cards = {c["id"] for c in V["cards"]}
    for i, f in enumerate(V["pick"]["data"]["frames"]):
        if f.get("source") != "segmented":
            continue
        m = ondemand.chunk_of(V, i, "pick")
        sy, sx = sh / m.shape[0], sw / m.shape[1]
        for code in np.unique(m):
            cid = V["owner"][code] if code < len(V["owner"]) else None
            if cid not in cards:
                continue
            mk = cv2.erode((m == code).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            ys, xs = np.nonzero(mk)
            if not len(ys):
                continue
            if len(ys) > 600:
                k = rng.choice(len(ys), 600, replace=False)
                ys, xs = ys[k], xs[k]
            P, z = seq.lift(a0 + int(f["frame"]) // hold, (xs + .5) * sx - .5, (ys + .5) * sy - .5)
            keep = tail_trim(z)
            if keep.sum():
                out.setdefault(cid, []).append(P[keep])
    res = {}
    for cid, parts in out.items():
        P = np.concatenate(parts)
        res[cid] = P[rng.choice(len(P), cap, replace=False)] if len(P) > cap else P
    return res


def gt_views(seq, frames, cap=80):
    """The other visit's frames for gt_place, loaded once: {frame: (GT depth, its 3x3 minimum, c2w, depth K)}; at most `cap`."""
    from scipy.ndimage import minimum_filter
    frames = list(frames)
    if len(frames) > cap:
        frames = [frames[int(i)] for i in np.linspace(0, len(frames) - 1, cap)]
    out = {}
    for i in frames:
        d, c2w = seq.depth(i), seq.c2w(i)
        if d is not None and c2w is not None:
            out[i] = (d, minimum_filter(np.where(d > 0, d, np.inf), size=3), c2w, seq.depth_k(i))
    return out


def gt_place(P, views_in, border=.05, min_pts=30):
    """The see-through test on GT depth: where GT puts the points, what GT depth sees there, over the other visit's frames
    (gt_views)."""
    views = []
    for i, (d, dmin_all, c2w, k) in views_in.items():
        cam = (P - c2w[:3, 3]) @ c2w[:3, :3]
        z = cam[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u, v = k[0] * cam[:, 0] / z + k[2], k[1] * cam[:, 1] / z + k[3]
        h, w = d.shape
        inside = (z > .1) & (u >= border * w) & (u < (1 - border) * w) & (v >= border * h) & (v < (1 - border) * h)
        if inside.sum() < min_pts or inside.mean() < .5:
            continue
        ui, vi, zz = u[inside].astype(int), v[inside].astype(int), z[inside]
        dmin = dmin_all[vi, ui]
        dd = d[vi, ui]
        valid = (dd > 0) & np.isfinite(dmin)
        if valid.sum() < min_pts:
            continue
        m = np.maximum(GT_ABS_M, GT_REL * zz) + 2 * GT_POSE_M
        free = valid & (dmin > zz + m)
        front = valid & (dd < zz - m)
        occ = valid & ~free & ~front
        n = valid.sum()
        views.append({"frame": int(i), "judged": int(n), "free": int(free.sum()), "occupied": int(occ.sum()), "front": int(front.sum())})
    n_free = sum(x["free"] >= .6 * x["judged"] for x in views)
    n_occ = sum(x["occupied"] >= .5 * x["judged"] for x in views)
    state = ("free" if n_free >= 2 and n_free > n_occ else "occupied" if n_occ and n_occ >= n_free else
             "out-of-view" if not views else "occluded" if sum(x["front"] >= .5 * x["judged"] for x in views) > len(views) / 2 else "unjudged")
    return {"state": state, "views": len(views), "free_views": int(n_free), "occupied_views": int(n_occ)}


def voxels(P, vox=VOX_M):
    return set(map(tuple, np.floor(np.asarray(P) / vox).astype(np.int64)))


def overlaps(GA, GB):
    """{(a, b): (IoU, share of the smaller)} for card pairs whose GT voxels meet."""
    VA = {k: voxels(p) for k, p in GA.items() if len(p) >= GT_MIN_POINTS}
    VB = {k: voxels(p) for k, p in GB.items() if len(p) >= GT_MIN_POINTS}
    index = {}
    for b, vs in VB.items():
        for x in vs:
            index.setdefault(x, []).append(b)
    out = {}
    for a, va in VA.items():
        hits = {}
        for x in va:
            for b in index.get(x, ()):
                hits[b] = hits.get(b, 0) + 1
        for b, n in hits.items():
            out[(a, b)] = (n / (len(va) + len(VB[b]) - n), n / min(len(va), len(VB[b])))
    return out


def gt_heights(P, floor):
    """GT top / base above the GT floor (p98 / p2 of the points' heights)."""
    n, p0 = np.asarray(floor["normal"], float), np.asarray(floor["point"], float)
    h = (np.asarray(P) - p0) @ n
    return float(np.percentile(h, 98)), float(np.percentile(h, 2))


def kabsch(Q, P):
    """R, t minimising |P - (R Q + t)|."""
    mq, mp = Q.mean(0), P.mean(0)
    U, _, Vt = np.linalg.svd((P - mp).T @ (Q - mq))
    R = U @ np.diag([1., 1., np.sign(np.linalg.det(U @ Vt))]) @ Vt
    return R, mp - R @ mq


def icp(P, Q, R, t, steps=(.3, .15, .08, .04), iters=12):
    """Rigid ICP carrying cloud Q onto cloud P from (R, t), with a shrinking inlier distance. -> R, t, fitness (share of Q within
    5 cm of P), rmse of those."""
    from scipy.spatial import cKDTree
    tree = cKDTree(P)
    for thr in steps:
        for _ in range(iters):
            d, idx = tree.query(Q @ R.T + t)
            m = d < thr
            if m.sum() < 200:
                break
            R, t = kabsch(Q[m], P[idx[m]])
    d, _ = tree.query(Q @ R.T + t)
    m = d < .05
    return R, t, float(m.mean()), float(np.sqrt(np.mean(d[m] ** 2))) if m.any() else None


def gt_alignment(SA, SB, a, b):
    """B's GT world -> A's GT world. One world (TUM's mocap, one ARKit session): the identity. Two ARKit sessions (both
    gravity-aligned, +z up): rigid ICP of their fused LiDAR (confident) depth from 36 starting turns about +z (the clouds'
    medians put together), the best kept; accepted with fitness >= 0.3 and rmse <= 3 cm. Independent of the visits layer."""
    if a["gt_world"] == b["gt_world"]:
        return {"R": np.eye(3), "t": np.zeros(3), "how": "one GT world"}
    PA = SA.fused(stride=4, frames=list(range(0, len(SA.g["frames"]), 3)))
    PB = SB.fused(stride=4, frames=list(range(0, len(SB.g["frames"]), 3)))
    vox = lambda P, v=.05: P[np.unique(np.floor(P / v).astype(np.int64), axis=0, return_index=True)[1]]  # noqa: E731
    PA, PB = vox(PA), vox(PB)
    sub = PB[np.random.default_rng(0).choice(len(PB), min(len(PB), 20000), replace=False)]
    best = None
    for yaw in np.radians(np.arange(0, 360, 10)):
        c, s_ = np.cos(yaw), np.sin(yaw)
        R0 = np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1.]])
        t0 = np.median(PA, 0) - R0 @ np.median(PB, 0)
        R, t, fit_, rmse = icp(PA, sub, R0, t0, steps=(.5, .25, .1, .05), iters=8)
        if best is None or fit_ > best[2]:
            best = (R, t, fit_, rmse, yaw)
    R, t, fit_, rmse = icp(PA, PB, best[0], best[1])
    return {"R": R, "t": t, "how": "ICP of the two sessions' LiDAR depth from 36 turns about +z", "fitness": round(fit_, 3),
            "rmse_m": None if rmse is None else round(rmse, 4), "accepted": bool(fit_ >= .3 and rmse is not None and rmse <= .03),
            "points": [len(PA), len(PB)], "start_yaw_deg": round(float(np.degrees(best[4])), 1)}


def gt_pair(layers_root, data_root, a, b):
    """Everything GT says about one pair: a / b = {report, seq (folder under data_root), a0, hold, gt_world}. Needs GT depth.
    B's GT world is carried into A's (gt_alignment); per card of each visit its GT point count, its place in the other visit
    (GT see-through test) and its GT top / base above A's GT floor; the card pairs whose GT voxels meet."""
    from accuracy_gt import Seq
    from fast_report import visits as fv
    A, B = fv.load(layers_root, a["report"]), fv.load(layers_root, b["report"])
    SA, SB = Seq(Path(data_root) / a["seq"]), Seq(Path(data_root) / b["seq"])
    al = gt_alignment(SA, SB, a, b)
    out = {"a": a, "b": b, "alignment": None if al is None else {k: (np.round(v, 6).tolist() if isinstance(v, np.ndarray) else v) for k, v in al.items()},
           "cards_a": {}, "cards_b": {}, "overlaps": []}
    GA = card_points_gt(A, SA, a["a0"], a["hold"])
    GB = card_points_gt(B, SB, b["a0"], b["hold"])
    if al is None or al.get("accepted") is False:
        for side, G in (("cards_a", GA), ("cards_b", GB)):
            out[side] = {cid: {"gt_points": int(len(P))} for cid, P in G.items()}
        return out
    R, t = np.asarray(al["R"], float), np.asarray(al["t"], float)
    to_a, to_b = (lambda P: P @ R.T + t), (lambda P: (P - t) @ R)
    fa = gt_views(SA, sorted({a["a0"] + int(k) // a["hold"] for s_ in A["cameras"].values() for k in s_["keys"]}))
    fb = gt_views(SB, sorted({b["a0"] + int(k) // b["hold"] for s_ in B["cameras"].values() for k in s_["keys"]}))
    box = lambda P: np.round(np.percentile(P, [2, 98], axis=0), 3).tolist()  # noqa: E731  GT box in A's GT world (2 / 98 %)
    for cid, P in GA.items():
        row = {"gt_points": int(len(P)), "box": box(P)}
        if len(P) >= GT_MIN_POINTS:
            row["place_in_other"] = gt_place(to_b(P), fb)
            if SA.floor:
                row["top_base"] = gt_heights(P, SA.floor)
        out["cards_a"][cid] = row
    GBa = {cid: to_a(P) for cid, P in GB.items()}
    for cid, P in GB.items():
        row = {"gt_points": int(len(P)), "box": box(GBa[cid])}
        if len(P) >= GT_MIN_POINTS:
            row["place_in_other"] = gt_place(GBa[cid], fa)
            if SA.floor:
                row["top_base"] = gt_heights(GBa[cid], SA.floor)
        out["cards_b"][cid] = row
    out["overlaps"] = [[x, y, round(i, 4), round(s_, 4)] for (x, y), (i, s_) in overlaps(GA, GBa).items() if s_ >= .05]
    return out


# ---------------------------------------------------------------- scoring (local)

def poses_of(gt_file):
    """source frame -> GT c2w (None where GT has none), from gt.json or gt-poses.json."""
    g = json.loads(Path(gt_file).read_text())
    rows = g["frames"] if "frames" in g and isinstance(g["frames"], list) else g["poses"]
    return [None if r.get("c2w") is None else np.asarray(r["c2w"], float) for r in rows]


def registration_error(layer, cams_a, cams_b, gt_a, gt_b, a0, hold_a, b0, hold_b, alignment=None):
    """Per accepted registration: B's keyframe cameras carried by the layer's B shot -> A shot transform, against GT's place
    for them in A's map (S_A^-1 of B's GT cameras). alignment: B's GT world -> A's (two ARKit sessions: gt_alignment's ICP).
    -> [{b_shot, n, cm (median, p90), deg (median, p90), s_A, ate_A_cm}]."""
    if alignment is not None:
        Rg, tg = np.asarray(alignment["R"], float), np.asarray(alignment["t"], float)
        gt_b = [None if g is None else np.block([[Rg @ g[:3, :3], (Rg @ g[:3, 3] + tg)[:, None]], [np.zeros((1, 3)), np.ones((1, 1))]]) for g in gt_b]
    from fast_report.core import umeyama
    from fast_report import visits as fv
    rows = []
    for r in layer["registration"]:
        if not r.get("accepted"):
            continue
        T = {k: np.asarray(r["shot_transform"][k], float) for k in ("R", "t")} | {"s": float(r["shot_transform"]["s"])}
        sa, sb = cams_a[r["a_shot"]], cams_b[r["b_shot"]]
        CA = np.asarray(sa["c2w"], float)
        GA = [gt_a[a0 + int(k) // hold_a] for k in sa["keys"]]
        ok = [i for i, g in enumerate(GA) if g is not None]
        used = {sa["keys"].index(f) for f in r.get("a_frames") or [] if f in sa["keys"]}
        near = [i for i in ok if any(abs(i - u) <= LOCAL_KEYS for u in used)]  # the A views around the ones the registration used
        CB = fv.move(T, np.asarray(sb["c2w"], float))
        GB = [gt_b[b0 + int(k) // hold_b] for k in sb["keys"]]
        okb = [i for i, g in enumerate(GB) if g is not None]
        row = {"b_shot": r["b_shot"], "a_shot": r["a_shot"], "n": len(okb), "u_m": r.get("u_m")}
        for tag, sel in (("", ok), ("local_", near if len(near) >= 5 else ok)):
            s, R, t = umeyama(CA[sel, :3, 3], np.array([GA[i][:3, 3] for i in sel]))
            ate = np.linalg.norm(s * CA[sel, :3, 3] @ R.T + t - np.array([GA[i][:3, 3] for i in sel]), axis=1)
            RA = fv.rot_avg([GA[i][:3, :3] for i in sel], [CA[i][:3, :3] for i in sel])  # A's rotation onto GT's (the cameras' own)
            est = s * CB[okb, :3, 3] @ R.T + t  # in the GT world
            pos = np.linalg.norm(est - np.array([GB[i][:3, 3] for i in okb]), axis=1) * 100
            rot = [fv.angle_deg(RA @ CB[i][:3, :3], GB[i][:3, :3]) for i in okb]
            row.update({tag + "cm_median": round(float(np.median(pos)), 1), tag + "cm_p90": round(float(np.percentile(pos, 90)), 1),
                        tag + "deg_median": round(float(np.median(rot)), 2), tag + "deg_p90": round(float(np.percentile(rot, 90)), 2),
                        tag + "s_A_gt_per_est": round(float(s), 4), tag + "ate_A_cm": round(float(np.sqrt(np.mean(ate ** 2))) * 100, 1),
                        tag + "a_views": len(sel)})
        rows.append(row)
    return rows


def score_pair(layer, gt):
    """Object match and change precision / recall of one pair's visits layer against gt_pair's output."""
    ca, cb = gt["cards_a"], gt["cards_b"]
    ov = {(x, y): (i, s) for x, y, i, s in gt["overlaps"]}
    # GT pairs: mutual best voxel IoU >= GT_IOU
    best_a, best_b = {}, {}
    for (x, y), (i, s) in ov.items():
        if i > best_a.get(x, (None, 0))[1]:
            best_a[x] = (y, i)
        if i > best_b.get(y, (None, 0))[1]:
            best_b[y] = (x, i)
    gt_pairs = {(x, y) for x, (y, i) in best_a.items() if i >= GT_IOU and best_b.get(y, (None,))[0] == x}
    rows = layer["objects"]
    matched = [o for o in rows if o["status"] in ("static", "changed_height", "changed_angle")]
    scored = [o for o in matched if ca.get(o["a"], {}).get("gt_points", 0) >= GT_MIN_POINTS and cb.get(o["b"], {}).get("gt_points", 0) >= GT_MIN_POINTS]
    right = [o for o in scored if ov.get((o["a"], o["b"]), (0, 0))[1] >= GT_SHARE]
    right_box = [o for o in scored if ov.get((o["a"], o["b"]), (0, 0))[1] >= GT_SHARE
                 or box_iou(ca.get(o["a"], {}).get("box"), cb.get(o["b"], {}).get("box")) >= GT_BOX_IOU]
    pred_pairs = {(o["a"], o["b"]) for o in matched}
    by_a = {o["a"]: o for o in rows if o.get("a")}
    # a GT pair is found when either card is matched to a card GT puts at the same place (a twin of the other one: one object
    # delineated in two shots of a visit, each shot its own copy)
    partners_a, partners_b = {}, {}
    for o in matched:
        partners_a.setdefault(o["a"], set()).add(o["b"])
        partners_b.setdefault(o["b"], set()).add(o["a"])
    found = {(x, y) for x, y in gt_pairs if any(ov.get((x, y2), (0, 0))[1] >= GT_SHARE for y2 in partners_a.get(x, ()))
             or any(ov.get((x2, y), (0, 0))[1] >= GT_SHARE for x2 in partners_b.get(y, ()))}
    out = {"match": {"predicted": len(matched), "scored": len(scored), "right": len(right),
                     "precision": round(len(right) / len(scored), 3) if scored else None,
                     "precision_box": round(len(right_box) / len(scored), 3) if scored and any("box" in v for v in ca.values()) else None,
                     "gt_pairs": len(gt_pairs), "found": len(found), "found_exact": len(gt_pairs & pred_pairs),
                     "recall": round(len(found) / len(gt_pairs), 3) if gt_pairs else None,
                     "gt_pairs_matched_elsewhere": sum(1 for x, y in gt_pairs if (x, y) not in pred_pairs and by_a.get(x, {}).get("status") in ("static", "changed_height", "changed_angle")),
                     "gt_pairs_status": _count(by_a.get(x, {}).get("status", "absent") for x, y in gt_pairs - found)}}
    claims = []
    for o in rows:
        if o["status"] == "missing":
            g = (ca.get(o["a"]) or {}).get("place_in_other", {}).get("state")
            claims.append(("missing", o, g, "right" if g == "free" else "wrong" if g == "occupied" else "unverifiable"))
        elif o["status"] == "new":
            g = (cb.get(o["b"]) or {}).get("place_in_other", {}).get("state")
            claims.append(("new", o, g, "right" if g == "free" else "wrong" if g == "occupied" else "unverifiable"))
        elif o["status"] == "moved":
            ga = (ca.get(o["a"]) or {}).get("place_in_other", {}).get("state")
            gb = (cb.get(o["b"]) or {}).get("place_in_other", {}).get("state")
            claims.append(("moved", o, f"{ga}/{gb}", "right" if ga == "free" and gb != "occupied" else "wrong" if ga == "occupied" or gb == "occupied" else "unverifiable"))
        elif o["status"] in ("changed_height", "changed_angle"):
            ta, tb = (ca.get(o["a"]) or {}).get("top_base"), (cb.get(o["b"]) or {}).get("top_base")
            same = ov.get((o["a"], o["b"]), (0, 0))[1] >= GT_SHARE or box_iou((ca.get(o["a"]) or {}).get("box"), (cb.get(o["b"]) or {}).get("box")) >= GT_BOX_IOU
            if ta and tb and "top_above_floor" in (o.get("delta") or {}):
                d_gt = tb[0] - ta[0]
                d = o["delta"]["top_above_floor"]["value"]
                claims.append((o["status"], o, round(d_gt, 3), "right" if same and abs(d_gt) > .05 and np.sign(d_gt) == np.sign(d) else "wrong"))
            else:
                claims.append((o["status"], o, None, "unverifiable"))
    gt_gone = {x for x, r in ca.items() if (r.get("place_in_other") or {}).get("state") == "free"}
    gt_came = {y for y, r in cb.items() if (r.get("place_in_other") or {}).get("state") == "free"}
    large = lambda r: bool(r.get("box")) and float(np.max(np.diff(np.asarray(r["box"], float), axis=0))) >= LARGE_M  # noqa: E731
    said_gone = {o["a"] for o in rows if o["status"] in ("missing", "moved")}
    said_came = {o["b"] for o in rows if o["status"] in ("new", "moved")}
    status_a = {o["a"]: o["status"] for o in rows if o.get("a")}
    status_b = {o["b"]: o["status"] for o in rows if o.get("b")}
    n = lambda v: sum(1 for c in claims if c[3] == v)  # noqa: E731
    out["changes"] = {"claims": len(claims), "right": n("right"), "wrong": n("wrong"), "unverifiable": n("unverifiable"),
                      "precision": round(n("right") / (n("right") + n("wrong")), 3) if n("right") + n("wrong") else None,
                      "gt_gone": len(gt_gone), "gt_gone_found": len(gt_gone & said_gone), "gt_came": len(gt_came), "gt_came_found": len(gt_came & said_came),
                      "recall": round((len(gt_gone & said_gone) + len(gt_came & said_came)) / (len(gt_gone) + len(gt_came)), 3) if gt_gone or gt_came else None,
                      "gt_gone_status": _count(status_a.get(x, "absent") for x in gt_gone), "gt_came_status": _count(status_b.get(y, "absent") for y in gt_came),
                      "gt_large": sum(large(ca[x]) for x in gt_gone) + sum(large(cb[y]) for y in gt_came),
                      "gt_large_found": sum(large(ca[x]) for x in gt_gone & said_gone) + sum(large(cb[y]) for y in gt_came & said_came),
                      "by_kind": _count(c[0] + ":" + c[3] for c in claims)}
    out["claims"] = [{"kind": k, "a": o.get("a"), "b": o.get("b"), "name": o.get("name_a") or o.get("name_b"), "gt": g, "verdict": v,
                      "tile": (o.get("evidence") or {}).get("tile")} for k, o, g, v in claims]
    return out


# ---------------------------------------------------------------- the runs' sites, their GT and the pairs

RUNS = PHASE2 / "runs"
SITES = {  # site -> (sequence folder on the Volume under /data, first source frame, hold, GT world, local GT poses)
    "gt-tum-w0": ("tum/room", 0, 1, "tum-fr1-mocap", RUNS / "mvp2-accuracy-inputs/tum/gt.json"),
    "gt-tum-w1": ("tum/room", 900, 1, "tum-fr1-mocap", RUNS / "mvp2-accuracy-inputs/tum/gt.json"),
    **{f"gt-{n}-w0": (f"tum/{n}", 0, 1, "tum-fr1-mocap", RUNS / f"r5b-visit-inputs/tum/{n}/gt-poses.json") for n in ("desk", "desk2", "xyz", "360", "plant", "teddy")},
    "gt-arkit47-w0": ("arkit/47333932/gt", 0, 6, "arkit-47333932", RUNS / "mvp2-accuracy-inputs/arkit47/gt.json"),
    "gt-arkit47-w1": ("arkit/47333932/gt", 128, 6, "arkit-47333932", RUNS / "mvp2-accuracy-inputs/arkit47/gt.json"),
    "gt-arkit31-w0": ("arkit/47333931/gt", 0, 6, "arkit-47333931", RUNS / "r5b-visit-inputs/arkit/arkit47333931/gt-poses.json"),
    "gt-arkit31-w1": ("arkit/47333931/gt", 128, 6, "arkit-47333931", RUNS / "r5b-visit-inputs/arkit/arkit47333931/gt-poses.json"),
}
KIND = {"tum": "GT revisit (one mocap frame; real changes between recordings)", "arkit": "static revisit (one venue, minutes apart)",
        "own": "our clips (overlapping windows of one walk: no real change)"}


def reports_of(run_dirs):
    """site -> newest report id over the runs' summary.json (accuracy entrypoint), with its run dir."""
    out = {}
    for d in run_dirs:
        sm = json.loads((Path(d) / "summary.json").read_text())
        for r in sm["runs"]:
            if not r.get("error"):
                out[r["site"]] = (r["report"], Path(d))
    return out


def gt_pairs(pairs, reports):
    """[(a_site, b_site)] -> gt_one's inputs for the pairs whose sites have GT."""
    todo = []
    for a, b in pairs:
        if a in SITES and b in SITES and a in reports and b in reports:
            side = lambda s: {"report": reports[s][0], "seq": SITES[s][0], "a0": SITES[s][1], "hold": SITES[s][2], "gt_world": SITES[s][3], "site": s}  # noqa: E731
            todo.append({"a": side(a), "b": side(b)})
    return todo


def layer_of(root, report, visit_of=None):
    """The newest visits patch on a report (optionally the one against a given site map)."""
    best = None
    for p in sorted((Path(root) / "reports" / report / "patches").glob("*-visits.json")):
        x = json.loads(p.read_text())
        if visit_of in (None, x["data"]["site_map"]):
            best = x
    return best


PAIRS = [  # (site map, revisit, kind): TUM fr1 against the room's first 30 s, two desk revisits among themselves, ARKit's venue, our clip
    ("gt-tum-w0", "gt-tum-w1", "tum"), ("gt-tum-w0", "gt-desk-w0", "tum"), ("gt-tum-w0", "gt-desk2-w0", "tum"), ("gt-tum-w0", "gt-xyz-w0", "tum"),
    ("gt-tum-w0", "gt-360-w0", "tum"), ("gt-tum-w0", "gt-plant-w0", "tum"), ("gt-tum-w0", "gt-teddy-w0", "tum"),
    ("gt-desk-w0", "gt-desk2-w0", "tum"), ("gt-xyz-w0", "gt-desk-w0", "tum"),
    ("gt-arkit47-w0", "gt-arkit47-w1", "arkit"), ("gt-arkit47-w0", "gt-arkit31-w0", "arkit"), ("gt-arkit47-w0", "gt-arkit31-w1", "arkit"),
    ("own-samsclub-337", "own-samsclub-352", "own")]
CLAIMS = ("moved", "new", "missing", "changed_height", "changed_angle")


def evaluate(a_site, b_site, kind, reports, layer, gt=None, labels=None):
    """One pair's row: registration (and its GT error), counts, object match and change precision / recall (GT depth), false
    changes of a static revisit (every claim: GT verdict, else the agent's label by eye)."""
    from fast_report import visits as fv
    (a_rep, a_run), (b_rep, b_run) = reports[a_site], reports[b_site]
    row = {"a": a_site, "b": b_site, "kind": kind, "a_report": a_rep, "b_report": b_rep, "counts": layer["counts"], "s": layer.get("s")}
    regs = layer["registration"]
    row["registration"] = [{k: r.get(k) for k in ("b_shot", "a_shot", "accepted", "refused_because", "u_m", "floor", "floor_transform", "fit_a", "fit_b",
                                                  "retrieval_cos_median", "da3_s")} for r in regs]
    if a_site in SITES and b_site in SITES:
        A, B = fv.load(a_run, a_rep), fv.load(b_run, b_rep)
        same = SITES[a_site][3] == SITES[b_site][3]
        al = (gt or {}).get("alignment")
        if same or (al and al.get("accepted")):
            row["registration_gt"] = registration_error(layer, A["cameras"], B["cameras"], poses_of(SITES[a_site][4]), poses_of(SITES[b_site][4]),
                                                        SITES[a_site][1], SITES[a_site][2], SITES[b_site][1], SITES[b_site][2], None if same else al)
        row["gt_alignment"] = None if same else al
    if gt and (gt.get("overlaps") or gt["cards_a"] and any("place_in_other" in v for v in gt["cards_a"].values())):
        row.update(score_pair(layer, gt))
    claims = [o for o in layer["objects"] if o["status"] in CLAIMS]
    judged = sum(1 for o in layer["objects"] if o["status"] not in ("not_compared", "not_observed_in_a", "not_observed_in_b"))
    lab = labels or {}
    by_gt = {(c["kind"], c["a"], c["b"]): c["verdict"] for c in row.get("claims", [])}
    verdicts = []
    for o in claims:
        v = by_gt.get((o["status"], o.get("a"), o.get("b")))
        tile = (o.get("evidence") or {}).get("tile")
        if v in (None, "unverifiable"):
            v = lab.get(tile, "unlabelled")  # by eye (agent-labelled): the evidence tile of both visits
        verdicts.append({"status": o["status"], "a": o.get("a"), "b": o.get("b"), "name": o.get("name_a") or o.get("name_b"), "tile": tile, "verdict": v})
    row["claims_all"] = verdicts
    row["false_changes"] = {"claims": len(claims), "wrong": sum(v["verdict"] == "wrong" for v in verdicts), "judged_objects": judged,
                            "rate": round(sum(v["verdict"] == "wrong" for v in verdicts) / judged, 4) if judged else None}
    return row


LARGE_M = .3  # GT changes of objects whose GT box is >= 30 cm on a side, apart from the small ones (a pen on a desk is below the
# registration's u: its place cannot be told empty)
GT_BOX_IOU = .2  # the looser 'same object' test: GT boxes (2-98 % of the GT points) overlapping (two sides of one desk share few voxels)


def box_iou(a, b):
    if not a or not b:
        return 0.
    a, b = np.asarray(a, float), np.asarray(b, float)
    inter = np.prod(np.clip(np.minimum(a[1], b[1]) - np.maximum(a[0], b[0]), 0, None))
    va, vb = np.prod(np.clip(a[1] - a[0], 1e-3, None)), np.prod(np.clip(b[1] - b[0], 1e-3, None))
    return float(inter / max(va + vb - inter, 1e-9))


def results(run, bench, gt_dir, labels_file, out):
    """The results page: every pair's row (evaluate) on the bench's final visits layers, the per-video table, times, spend."""
    import shutil
    from fast_report.instrument import usd_per_s
    run, bench, out = Path(run), Path(bench), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "evidence").mkdir(exist_ok=True)
    reports = reports_of([run])
    labels = json.loads(Path(labels_file).read_text()) if labels_file and Path(labels_file).exists() else {}
    vs = json.loads((bench / "visits-summary.json").read_text())
    rv = json.loads((bench / "revisits-summary.json").read_text()) if (bench / "revisits-summary.json").exists() else {"runs": []}
    timing = {(r["a"], r["b"]): r for r in vs["pairs"]}
    rows = []
    for sa, sb, kind in PAIRS:
        if sa not in reports or sb not in reports:
            continue
        a_rep, b_rep = reports[sa][0], reports[sb][0]
        lay = layer_of(bench, b_rep, a_rep)
        if lay is None:
            continue
        g = Path(gt_dir) / f"gt-{a_rep}--{b_rep}.json"
        gt = json.loads(g.read_text()) if g.exists() else None
        row = evaluate(sa, sb, kind, reports, lay["data"], gt, labels.get(f"{sb}--{sa}"))
        t = timing.get((a_rep, b_rep), {})
        row["timing"] = {"visit_call_written_s": (t.get("written_s") or [None])[0], "stages_s": t.get("stages"), "gpu_peak_gib": t.get("gpu_peak_gib"),
                         "clock": "FastReport.visit: t0 = the call (the two analyses done), written = the Volume commit"}
        for c in row["claims_all"]:
            if c.get("tile") and lay["blobs"].get(c["tile"]):
                src = bench / "blobs" / "sha256" / lay["blobs"][c["tile"]]["sha256"]
                if src.exists():
                    dst = out / "evidence" / f"{sb}--{sa}--{c['tile']}.jpg"
                    shutil.copyfile(src, dst)
                    c["file"] = f"evidence/{dst.name}"
        rows.append(row)
    revisits = []
    for r in rv["runs"]:
        m = r["milestones"]
        revisits.append({"site": r["site"], "report": r["report"], "visit_of": r["visit_of"], "error": r["error"], "visits_error": r.get("visits_error"),
                         "cards_final_written_s": m.get("cards_v3") or m.get("cards_v2"), "visits_written_s": m.get("visits"), "visits_stage_s": r["visits_stage_s"],
                         "gpu_peak_gib": r["gpu_peak_gib"], "first_call_after_boot": r["first_call"]})
    boot = vs["boot"]
    ends = [r.get("ended_unix") for r in vs["pairs"] if r.get("ended_unix")]
    life = (max(ends) - boot["client_submitted_unix"]) if ends else None
    summary = {"schema": "r5b-visits-results-v1", "pairs": rows, "revisits": revisits,
               "bench": {"boot_ready_s": boot.get("ready_s"), "container_life_s": round(life, 1) if life else None,
                         "usd_upper": round((life + 60) * usd_per_s(), 2) if life else None, "gpus": boot.get("gpus")}}
    (out / "results.json").write_text(json.dumps(summary, indent=1, default=float))
    (out / "tables.md").write_text(tables_md(summary))
    return summary


def gt_err(x):
    return f"{x['cm_median']} / {x['cm_p90']}, {x['deg_median']}"


def tables_md(sm):
    f = lambda v, d=2: "—" if v is None else (f"{v:.{d}f}" if isinstance(v, float) else str(v))  # noqa: E731
    md = "## Per revisit video (B against its site map A)\n\n| B (revisit) | A (site map) | kind | shots registered | u (cm) | GT error cm (median / p90), deg | " \
         "objects compared | match P (surface / box) / R | claims: right / wrong / unverified / unlabelled | GT changes found (all / >= 30 cm) | false changes / judged objects | visit s (to commit) |\n" \
         "|---|---|---|---|---|---|---|---|---|---|---|---|\n"
    for r in sm["pairs"]:
        regs = r["registration"]
        acc = [x for x in regs if x.get("accepted")]
        rg = r.get("registration_gt") or []
        m, c, fc = r.get("match") or {}, r.get("changes") or {}, r["false_changes"]
        v = [x["verdict"] for x in r["claims_all"]]
        md += (f"| {r['b']} | {r['a']} | {r['kind']} | {len(acc)}/{len(regs)} | {', '.join(f(x['u_m'] * 100, 0) for x in acc) or '—'} | "
               f"{'; '.join(map(gt_err, rg)) or '—'} | "
               f"{fc['judged_objects']} | {f(m.get('precision'))} / {f(m.get('precision_box'))} / {f(m.get('recall'))} | "
               f"{v.count('right')} / {v.count('wrong')} / {v.count('unverifiable') + v.count('unclear')} / {v.count('unlabelled')} | "
               f"{f(c.get('gt_gone_found', 0) + c.get('gt_came_found', 0) if c else None)} of {f(c.get('gt_gone', 0) + c.get('gt_came', 0) if c else None)} / "
               f"{f(c.get('gt_large_found'))} of {f(c.get('gt_large'))} | {fc['wrong']} / {fc['judged_objects']} | {f(r['timing']['visit_call_written_s'], 1)} |\n")
    md += "\n## Revisits analysed with their visit step (s from the MP4 bytes in the container to the Volume commit)\n\n| B | cards final written | visits written | visits stage | GPU peak GiB (0 / 1) |\n|---|---|---|---|---|\n"
    for r in sm["revisits"]:
        md += f"| {r['site']} | {f(r['cards_final_written_s'], 1)} | {f(r['visits_written_s'], 1)} | {f(r['visits_stage_s'], 1)} | {' / '.join(f(x, 1) for x in r['gpu_peak_gib'])} |\n"
    return md


def _count(xs):
    out = {}
    for x in xs:
        out[x] = out.get(x, 0) + 1
    return out


def self_check():
    layer = {"objects": [{"status": "static", "a": "a1", "b": "b1"}, {"status": "static", "a": "a2", "b": "b3"}, {"status": "missing", "a": "a3", "b": None},
                         {"status": "new", "a": None, "b": "b2"}, {"status": "not_observed_in_b", "a": "a4", "b": None}]}
    gt = {"cards_a": {"a1": {"gt_points": 100}, "a2": {"gt_points": 100}, "a3": {"gt_points": 100, "place_in_other": {"state": "free"}},
                      "a4": {"gt_points": 100, "place_in_other": {"state": "free"}}},
          "cards_b": {"b1": {"gt_points": 100}, "b2": {"gt_points": 100, "place_in_other": {"state": "occupied"}}, "b3": {"gt_points": 100}},
          "overlaps": [["a1", "b1", .6, .8], ["a2", "b2", .5, .7], ["a2", "b3", .01, .02]]}
    s = score_pair(layer, gt)
    assert s["match"]["precision"] == .5 and s["match"]["gt_pairs"] == 2 and s["match"]["recall"] == .5, s["match"]
    c = s["changes"]
    assert c["right"] == 1 and c["wrong"] == 1 and c["precision"] == .5 and c["gt_gone"] == 2 and c["gt_gone_found"] == 1 and c["recall"] == .5, c
    assert c["gt_gone_status"] == {"missing": 1, "not_observed_in_b": 1}, c
    P = np.random.default_rng(0).uniform(0, 1, (500, 3))
    ov = overlaps({"x": P}, {"y": P + [0, 0, .01], "z": P + 5})
    assert ov[("x", "y")][0] > .7 and ("x", "z") not in ov
    import cv2
    rng = np.random.default_rng(1)
    room = np.concatenate([np.c_[rng.uniform(0, 4, 3000), rng.uniform(0, 3, 3000), np.zeros(3000)],  # a floor, two walls, a box
                           np.c_[rng.uniform(0, 4, 2000), np.zeros(2000), rng.uniform(0, 2.5, 2000)],
                           np.c_[np.zeros(2000), rng.uniform(0, 3, 2000), rng.uniform(0, 2.5, 2000)],
                           np.c_[rng.uniform(1, 1.5, 800), rng.uniform(1, 1.4, 800), rng.uniform(0, .8, 800)]])
    Rt, tt = cv2.Rodrigues(np.array([.02, -.03, .4]))[0], np.array([.5, -.3, .1])
    Q = (room - tt) @ Rt  # the same room in another session's frame: Rt Q + tt = room
    R0 = cv2.Rodrigues(np.array([.0, .0, .45]))[0]  # a start 3 deg and 10 cm off
    R, t, fit_, rmse = icp(room, Q, R0, tt + [.08, -.06, 0])
    assert np.allclose(R, Rt, atol=1e-3) and np.allclose(t, tt, atol=5e-3) and fit_ > .95, (R, t, fit_)
    print("r5b_visits_eval self-check ok: GT pairs by mutual best IoU, match precision / recall, change claims right / wrong / unverifiable, "
          "recall of GT changes, ICP carries one session's room onto the other's from a start 3 deg / 10 cm off")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", nargs="?", choices=("results",))
    ap.add_argument("--run", type=Path)
    ap.add_argument("--bench", type=Path)
    ap.add_argument("--gt", type=Path)
    ap.add_argument("--labels", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--self-check", action="store_true")
    a = ap.parse_args()
    if a.self_check:
        self_check()
    elif a.cmd == "results":
        sm = results(a.run, a.bench, a.gt, a.labels, a.out)
        print((a.out / "tables.md").read_text())
