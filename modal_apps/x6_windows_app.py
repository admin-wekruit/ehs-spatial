"""X6: a video as content windows processed like workcell photo sets, in parallel on 2 x A100-80GB, and objects that
carry change over time (fast_report.windows cuts the windows, fast_report.timeline keeps identities and states).

One container, both cards, MPS on, models resident (boot = cold start, recorded apart; never in the analysis time):
DA3-GIANT-1.1, SAM 3 and SigLIP 2 on EACH card (no vLLM: the per-video vocabulary is the fast core's own, reused from
its runs). run() takes the MP4 bytes (t0) and returns the result; analysis time = t0 -> the last window's facts.

Decode (5 fps keyframes, the core's rule) feeds the window rule on a thread; shot cuts are measured in processes as the
core does and a window is only dispatched once the cut measure covers its frames (a cut inside it splits it). Per window:
  (b) per-window DA3 on its keyframes, SAM 3 person/floor + vocabulary, floor-plane scale, lift, embeddings; windows
      are stitched in order by Sim3 on their carried keyframes (rotation from the cameras, scale from the carried
      keyframes' depth ratio, translation from the centres) under register_cut_shot's gate (centre residual <= 10 % of
      the previous window's camera spread (>= 4 cm), rotation <= 3 deg); a refused stitch starts a new map frame;
  (a) SAM 3 per window during decode; per-shot DA3 on all the shot's keyframes after decode (today's core); the lift
      per window on the shot's geometry, in the shot's frame.
Then fast_report.timeline in window order, and three EHS facts with time intervals.

  modal run modal_apps/x6_windows_app.py --out RUNS/fx-x6-windows-time-NNN [--plan me340:a:0.4,me340:b:0.4,...]
  modal run modal_apps/x6_windows_app.py::sweep --out RUNS/... --runs ID,ID     # CPU: association thresholds on saved windows
"""
import json
import os
import pickle
import queue
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import modal
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "scripts"), str(HERE.parent)]
from fast_report_app import DA3_MODEL, DA3_REV, VOLUMES, gpu_listing, image as core_image  # noqa: E402  the core's image, weights
import sam3_app  # noqa: E402

image = core_image.add_local_python_source("fast_report_app")

PHASE2 = Path("/Users/adam/Desktop/panoptes-public/research-notes/phase2")
RUNS = PHASE2 / "runs"
PROCS, CPU, MEMORY_GIB = 24, 32, 160
CLIPS = {"me340": "me340-165", "samsclub-a2": "samsclub-337", "walmart": "walmart-190"}
STRIDE = 2  # the state tests' grid: DA3's 280 x 504 at stride 2
app = modal.App("panoptes-x6-windows")


# ---------- per-window pieces (inside the container) ----------

def instances(dev, voc, kept, votes_of, words, local_of, dyn, times, keys_frames, depth_m, K, c2w_m):
    """Lift (E7/E9) on this window's geometry -> per confirmed component: points (<= 400, 5 cm voxel centres), median
    centroid, p10/p90 box, camera range, keyframes seen, word vote, member mask indices. One GPU pass for all members'
    pixels, then numpy per component (a per-component GPU loop held the GIL for 0.3-2.5 s a window, run 003)."""
    import torch
    from fast_report import segment
    kept_t = torch.as_tensor(kept, device=dev)
    if len(kept_t) < 2:
        return [], {}
    frame_local = local_of[voc["frame"][kept_t]]
    comp, arr, stats = segment.lift(voc["mask"][kept_t], frame_local, depth_m, K, c2w_m, dyn)
    if arr is None:
        return [], stats
    conf = np.flatnonzero(arr["frames"] >= segment.CONFIRMED)
    sel = np.flatnonzero(np.isin(comp, conf))
    if not len(sel):
        return [], stats
    cid, gi = comp[sel], kept[sel]
    gi_t = torch.as_tensor(gi, device=dev)
    fl = frame_local[torch.as_tensor(sel, device=dev)]
    m = voc["mask"][gi_t][:, ::STRIDE, ::STRIDE] & (depth_m[:, ::STRIDE, ::STRIDE][fl] > 0) & ~dyn[:, ::STRIDE, ::STRIDE][fl]
    mid, vy, vx = torch.nonzero(m, as_tuple=True)
    pts = segment.backproject(depth_m, K, c2w_m, fl[mid], vy, vx, STRIDE).cpu().numpy()
    cam = c2w_m[fl[mid], :3, 3].cpu().numpy()
    pc = cid[mid.cpu().numpy()]
    order = np.argsort(pc, kind="stable")
    pc, pts, cam = pc[order], pts[order], cam[order]
    score = voc["score"][gi_t].float().cpu().numpy() * np.sqrt(voc["mask"][gi_t].sum((1, 2)).cpu().numpy())
    fl_np = fl.cpu().numpy()
    rng, out = np.random.default_rng(0), []
    for c in conf:
        a, b = np.searchsorted(pc, c, "left"), np.searchsorted(pc, c, "right")
        if b - a < 20:
            continue
        P = pts[a:b]
        p = (np.unique(np.floor(P / .05).astype(np.int64), axis=0) + .5) * .05
        if len(p) > 400:
            p = p[rng.choice(len(p), 400, replace=False)]
        own = cid == c
        members, f_own = gi[own], fl_np[own]
        vv = {}
        for q in members:
            for w_, s_ in votes_of.get(int(q), []):
                vv[words[w_]] = vv.get(words[w_], 0.) + s_
        best = int(np.argmax(score[own]))
        f_loc = sorted(set(f_own.tolist()))
        out.append({"points": p.astype(np.float64), "centroid": np.median(P, 0), "lo": np.percentile(P, 10, 0), "hi": np.percentile(P, 90, 0),
                    "range_m": float(np.median(np.linalg.norm(P - cam[a:b], axis=1))), "label": segment.name(vv, words) or "object",
                    "votes": {k: round(v, 3) for k, v in sorted(vv.items(), key=lambda x: -x[1])[:5]},
                    "keys": [int(keys_frames[j]) for j in f_loc], "times": [float(times[j]) for j in f_loc],
                    "best_key": int(keys_frames[int(f_own[best])]), "members": members, "best_member": int(members[best])})
    return out, stats


def people_points(person, local, depth_m, K, c2w_m, keys_frames, times):
    """Per keyframe: each person (core.person_masks' dedupe) as the median 3D point of its mask (window frame)."""
    import torch
    from fast_report import core, segment
    rows = []
    for lf, masks in core.person_masks(person, local).items():
        for mk, score, _ in masks:
            m = torch.as_tensor(mk, device=depth_m.device)[::STRIDE, ::STRIDE] & (depth_m[lf, ::STRIDE, ::STRIDE] > 0)
            vy, vx = torch.nonzero(m, as_tuple=True)
            if len(vy) < 20:
                continue
            p = segment.backproject(depth_m, K, c2w_m, torch.full_like(vy, lf), vy, vx, STRIDE).cpu().numpy()
            rows.append({"key": int(keys_frames[lf]), "t": float(times[lf]), "xyz": np.median(p, 0), "low": np.percentile(p, 5, 0),
                         "score": round(score, 3)})
    return rows


def compact(depth_m, dyn, c2w_m, K):
    return {"depth": depth_m[:, ::STRIDE, ::STRIDE].half().cpu().numpy(), "person": dyn[:, ::STRIDE, ::STRIDE].cpu().numpy(),
            "c2w": c2w_m.double().cpu().numpy(), "K": (K.double().cpu().numpy() * np.array([[1 / STRIDE], [1 / STRIDE], [1]]))}


def sam_chunk(m, dev, kf, obj_local, words, clock, tag):
    """SAM 3 on <= 8 keyframes of one window: person/floor on all, the vocabulary on its object keyframes."""
    import torch
    from fast_report import segment
    gi = dev.index
    with clock.stage(f"sam3.person@gpu{gi}", gpu=gi, n={"frames": len(kf), "window": tag}, sync=True):
        vision = m.sams[dev].vision(kf)
        person = m.sams[dev].detect(vision, len(kf), ("person", "floor"), segment.PERSON_SCORE, top=segment.PERSON_TOP)
    voc = None
    if obj_local:
        with clock.stage(f"sam3.vocab.wave1@gpu{gi}", gpu=gi, n={"frames": len(obj_local), "words": len(words), "window": tag}, sync=True):
            voc = m.sams[dev].detect(m.sams[dev].pick(vision, obj_local), len(obj_local), words, segment.VOCAB_SCORE)
            voc["frame"] = torch.tensor(obj_local, device=dev)[voc["frame"]]
    return person, voc


def gather(parts, dev):
    """[(chunk start, dev, result dict or None)] -> one dict on dev, frames offset to the window."""
    import torch
    rows = [{k: (v + s if k == "frame" else v).to(dev) for k, v in r.items()} for s, _, r in sorted(parts, key=lambda x: x[0]) if r is not None]
    return {k: torch.cat([r[k] for r in rows]) for k in rows[0]} if rows else None


def objects_window(m, dev, w, kf, person, voc, geo, words, clock, tag):
    """dedupe + lift + embeddings + people for one window on `geo` (depth_m, K, c2w_m of the window's keyframes)."""
    import torch
    import torch.nn.functional as F
    from fast_report import core, segment
    gi = dev.index
    n = len(w["keys"])
    depth_m, K, c2w_m = geo["depth_m"], geo["K"], geo["c2w_m"]
    is_p = person["word"] == 0 if person is not None else None
    dyn = torch.zeros((n, *core.DA3_HW), dtype=torch.bool, device=dev)
    if person is not None and is_p.any():
        dyn = F.max_pool2d(core.union_by_frame(person["frame"][is_p], person["mask"][is_p], n)[:, None].float(), 5, 1, 2)[:, 0] > 0
    local = torch.arange(n, device=dev)
    with clock.stage("lift", gpu=gi, n={"window": tag, "masks_in": int(len(voc["frame"])) if voc else 0}, sync=True):
        if voc is not None and len(voc["frame"]):
            kept, votes = segment.dedupe(voc["frame"], voc["word"], voc["score"], voc["mask"])
        else:
            kept, votes = np.zeros(0, int), []
        vote_of = {}
        for a_, w_, s_ in votes:
            vote_of.setdefault(int(a_), []).append((int(w_), float(s_)))
        insts, stats = instances(dev, voc, kept, vote_of, words, local, dyn, w["times"], w["keys"], depth_m, K, c2w_m) if voc else ([], {})
    with clock.stage("embed", gpu=gi, n={"window": tag, "instances": len(insts)}, sync=True):
        if insts:
            mem = np.concatenate([x["members"] for x in insts])
            owner = torch.as_tensor(np.repeat(np.arange(len(insts)), [len(x["members"]) for x in insts]), device=dev)
            mt = torch.as_tensor(mem, device=dev)
            with m.emb_locks[dev]:  # the embedder keeps the last frames' RGB: one window at a time per card
                emb = m.embs[dev].crops(kf, voc["frame"][mt], voc["mask"][mt])
                m.embs[dev].release()
            e = torch.nn.functional.normalize(torch.zeros((len(insts), emb.shape[1]), device=dev).index_add_(0, owner, emb), dim=1).cpu().numpy()
            for x, v in zip(insts, e):
                x["emb"] = v.astype(np.float64)
                x.pop("members")
    people = []
    if person is not None and is_p.any():
        pm = {k: v for k, v in person.items()}
        people = people_points(pm, {j: j for j in range(n)}, depth_m, K, c2w_m, w["keys"], w["times"])
    return {"instances": insts, "lift": stats, "people": people, **compact(depth_m, dyn, c2w_m, K)}


def scale(g, person, n, dev):
    import torch
    from fast_report import core
    floor = torch.zeros((n, *core.DA3_HW), dtype=torch.bool, device=dev)
    if person is not None and (person["word"] == 1).any():
        f = person["word"] == 1
        floor = core.union_by_frame(person["frame"][f], person["mask"][f], n)
    plane = core.floor_plane(g["depth"], g["K"], g["c2w"], floor) if len(g["depth"]) >= 2 else None
    mpu = core.CAMERA_HEIGHT_M / plane["camera_height_units"] if plane and plane["camera_height_units"] > 0 else 1.
    c2w_m = g["c2w"].clone()
    c2w_m[:, :3, 3] *= mpu
    up = pt = None
    if plane:
        up, pt = plane["normal"].double().cpu().numpy(), (plane["point"].double() * mpu).cpu().numpy()
    return {"depth_m": g["depth"] * mpu, "K": g["K"], "c2w_m": c2w_m, "mpu": mpu, "plane": {"up": up, "point": pt} if plane else None,
            "scale_status": "estimated" if plane else "uncalibrated"}


# ---------- stitching (b) ----------

def rot_deg(r):
    return float(np.degrees(np.arccos(np.clip((np.trace(r) - 1) / 2, -1, 1))))


def stitch(prev, cur):
    """Sim3 cur-window frame -> prev-window frame on the keyframes they share, and the register_cut_shot gate."""
    shared = [k for k in cur["keys"][:cur["carried"]] if k in prev["keys"]]
    if not shared:
        return None
    ip, ic = [prev["keys"].index(k) for k in shared], [cur["keys"].index(k) for k in shared]
    Rp, Rc = prev["c2w"][ip, :3, :3], cur["c2w"][ic, :3, :3]
    u, _, vt = np.linalg.svd(sum(a @ b.T for a, b in zip(Rp, Rc)))
    R = u @ np.diag([1, 1, np.sign(np.linalg.det(u @ vt))]) @ vt
    dp, dc = prev["depth"][ip].astype(np.float64), cur["depth"][ic].astype(np.float64)
    ok = (dp > 0) & (dc > 0) & ~prev["person"][ip] & ~cur["person"][ic]
    if ok.sum() < 500:
        return None
    ratio = dp[ok] / dc[ok]
    s = float(np.median(ratio))
    cp, cc = prev["c2w"][ip, :3, 3], cur["c2w"][ic, :3, 3]
    t = (cp - s * cc @ R.T).mean(0)
    centre = np.linalg.norm(cp - (s * cc @ R.T + t), axis=1)
    turn = [rot_deg(a.T @ R @ b) for a, b in zip(Rp, Rc)]
    spread = float(np.linalg.norm(prev["c2w"][:, :3, 3] - prev["c2w"][:, :3, 3].mean(0), axis=1).mean())
    limit = max(.1 * spread, .04)
    return {"s": s, "R": R, "t": t, "shared": shared, "centre_residual_m": float(np.median(centre)), "centre_limit_m": limit,
            "rotation_residual_deg": float(np.median(turn)), "depth_ratio_mad": float(np.median(np.abs(np.log(ratio / s)))),
            "accepted": bool(np.median(centre) <= limit and np.median(turn) <= 3.)}


def to_frame(w, T):
    """Carry a window's geometry, instances and people into its map frame by Sim3 T = (s, R, t)."""
    s, R, t = T
    mv = lambda p: s * np.asarray(p) @ R.T + t  # noqa: E731
    c2w = w["c2w"].copy()
    c2w[:, :3, :3] = R @ w["c2w"][:, :3, :3]
    c2w[:, :3, 3] = mv(w["c2w"][:, :3, 3])
    w.update(c2w=c2w, depth=(w["depth"].astype(np.float32) * s))
    for x in w["instances"]:
        pts = mv(x["points"])
        x.update(points=pts, centroid=mv(x["centroid"]), range_m=x["range_m"] * s)
        corners = mv(np.array(np.meshgrid(*zip(x["lo"], x["hi"]))).reshape(3, -1).T)
        x.update(lo=corners.min(0), hi=corners.max(0))
    for p in w["people"]:
        p.update(xyz=mv(p["xyz"]), low=mv(p["low"]))
    if w.get("plane"):
        w["plane"] = {"up": R @ w["plane"]["up"], "point": mv(w["plane"]["point"])}
    return w


def compose(A, B):
    """A o B for Sim3 tuples (s, R, t): x -> A(B(x))."""
    return A[0] * B[0], A[1] @ B[1], A[0] * A[1] @ B[2] + A[2]


# ---------- facts ----------

# words in priority order: the first class with a candidate wins
STACK = ("stacked boxes", "pallet", "boxes", "box", "carton", "crate", "paper towel", "toilet paper", "tissue", "packaging", "package",
         "shelves", "shelf", "rack", "machine", "cabinet", "tool box")
UPRIGHT = ("shelving", "shelves", "shelf", "rack", "display stand", "cabinet", "door", "machine", "control panel", "stacked boxes", "boxes")
HAZARD = ("forklift", "pallet jack", "trolley", "hand truck", "ladder", "machine", "shopping cart", "stroller", "cable", "hose", "spill",
          "fire extinguisher", "control panel", "workbench")
NEAR_M = 2.  # a person-hazard fact only when someone came this close; otherwise the narrowest aisle


def has(label, words):
    return any(w in (label or "") for w in words)


def first_class(objs, words, lab):
    for w in words:
        c = [o for o in objs if w in lab(o)]
        if c:
            return c
    return []


def facts(tracker, planes, people):
    """Three facts per video, each (object, property, value +- uncertainty, time interval, evidence keyframes):
      1. top height above the floor of the highest stack of the first stack class present (boxes, pallets, shelves...);
      2. tilt from vertical of the main face of the first upright class present (shelving, racks, cabinets, machines);
      3. a person's closest approach on the floor plan to the first hazard class someone came within NEAR_M of, with the
         interval spent within 1 m; when nobody did, the narrowest aisle: free width across the walking direction between
         object points 0.1-2 m above the floor.
    Values are 'estimated' (scale from the floor plane and an assumed 1.6 m camera height: the +- leaves that assumption
    out); +- is the spread over the windows (or keyframes) that measured it, combined with sigma(z) of the timeline."""
    from fast_report import timeline as tl
    out = []
    objs = [o for o in tracker.objects if o["frame"] in planes]
    lab = lambda o: max(o["votes"], key=o["votes"].get)  # noqa: E731

    def evidence(o):
        return sorted({inst["best_key"] for _, inst in o["obs"]})[:4]

    def interval(o):
        ts = [t for _, inst in o["obs"] for t in inst["times"]]
        return [round(min(ts), 2), round(max(ts), 2)]

    def flat(p, pl):
        return p - np.outer((p - pl["point"]) @ pl["up"], pl["up"])

    height = lambda inst, pl: float(np.percentile((inst["points"] - pl["point"]) @ pl["up"], 95))  # noqa: E731
    cands = first_class([o for o in objs if len(o["obs"]) >= 2], STACK, lab) or first_class(objs, STACK, lab)
    if cands:
        o = max(cands, key=lambda o: np.median([height(i, planes[o["frame"]]) for _, i in o["obs"]]))
        hs = [height(i, planes[o["frame"]]) for _, i in o["obs"]]
        rng = np.median([i["range_m"] for _, i in o["obs"]])
        out.append({"object": o["id"], "label": lab(o), "property": "top height above floor (m)", "value": round(float(np.median(hs)), 2),
                    "uncertainty": round(float(np.hypot(np.std(hs) if len(hs) > 1 else 0., tl.sigma(rng))), 2), "interval_s": interval(o),
                    "windows": len(hs), "evidence_keys": evidence(o), "status": "estimated"})

    def tilt(inst, pl):
        p = inst["points"] - inst["points"].mean(0)
        if len(p) < 30:
            return None
        ev, vec = np.linalg.eigh(p.T @ p)
        if ev[0] > .2 * ev[1]:  # not a face: no normal to speak of
            return None
        a = float(np.degrees(np.arccos(abs(vec[:, 0] @ pl["up"]))))
        return 90. - a if a > 45 else None
    for w_ in UPRIGHT:
        cands = [(o, [x for x in (tilt(i, planes[o["frame"]]) for _, i in o["obs"]) if x is not None]) for o in objs if w_ in lab(o)]
        cands = [(o, v) for o, v in cands if len(v) >= 2] or [(o, v) for o, v in cands if v]
        if cands:
            o, v = max(cands, key=lambda c: (len(c[1]), np.median([len(i["points"]) for _, i in c[0]["obs"]])))
            out.append({"object": o["id"], "label": lab(o), "property": "tilt of its main face from vertical (deg)", "value": round(float(np.median(v)), 1),
                        "uncertainty": round(float(np.std(v)) if len(v) > 1 else 0., 1), "interval_s": interval(o), "windows": len(v),
                        "evidence_keys": evidence(o), "status": "estimated", "note": "sign-free; +- is the spread over windows"})
            break
    near = None
    for w_ in HAZARD:
        for o in (o for o in objs if w_ in lab(o)):
            pl = planes[o["frame"]]
            fo = flat(np.concatenate([i["points"] for _, i in o["obs"]]), pl)
            rows = [(float(np.min(np.linalg.norm(fo - flat(p["xyz"][None], pl), axis=1))), p) for p in people.get(o["frame"], [])]
            if rows and min(r[0] for r in rows) <= NEAR_M and (near is None or min(r[0] for r in rows) < near[0]):
                near = (min(r[0] for r in rows), o, rows)
        if near:
            break
    if near:
        d, o, rows = near
        close = sorted((r for r in rows if r[0] <= 1.), key=lambda r: r[1]["t"]) or [min(rows, key=lambda r: r[0])]
        p = min(rows, key=lambda r: r[0])[1]
        out.append({"object": o["id"], "label": lab(o), "property": "closest person on the floor plan (m)", "value": round(d, 2),
                    "uncertainty": round(float(np.hypot(tl.sigma(np.linalg.norm(p["xyz"] - o["obs"][0][1]["centroid"])), .15)), 2),
                    "interval_s": [round(close[0][1]["t"], 2), round(close[-1][1]["t"], 2)], "within_1m_keyframes": sum(r[0] <= 1. for r in rows),
                    "evidence_keys": sorted({r[1]["key"] for r in close})[:4], "status": "estimated",
                    "note": "person = median 3D point of the SAM 3 person mask; +- adds 0.15 m for the body's half-width"})
    else:
        best = None
        for w in tracker.windows:
            f = w.get("frame")
            if f not in planes:
                continue
            pl = planes[f]
            pts = np.concatenate([i["points"] for o in objs if o["frame"] == f for wi, i in o["obs"]]) if objs else np.zeros((0, 3))
            h = (pts - pl["point"]) @ pl["up"]
            pts = flat(pts[(h > .1) & (h < 2.)], pl)
            widths = []
            for j in range(0, len(w["keys"]), 2):
                c = flat(w["c2w"][j][:3, 3][None], pl)[0]
                fwd = flat(w["c2w"][j][:3, 3][None] + w["c2w"][j][:3, 2][None], pl)[0] - c
                fwd /= max(np.linalg.norm(fwd), 1e-9)
                side = np.cross(pl["up"], fwd)
                rel = pts - c
                along, lat = rel @ fwd, rel @ side
                sel = (np.abs(along) <= .75) & (np.abs(lat) <= 4.)
                left, right = lat[sel & (lat > .2)], -lat[sel & (lat < -.2)]
                if len(left) >= 5 and len(right) >= 5:
                    widths.append((float(np.percentile(left, 5) + np.percentile(right, 5)), w["keys"][j]))
            if len(widths) >= 2:
                m = float(np.median([x for x, _ in widths]))
                if best is None or m < best[0]:
                    best = (m, float(np.std([x for x, _ in widths])), w, [k for _, k in widths])
        if best:
            m, sd, w, ks = best
            out.append({"object": None, "label": "aisle", "property": "free width across the walking direction, 0.1-2 m above the floor (m)",
                        "value": round(m, 2), "uncertainty": round(float(np.hypot(sd, 2 * tl.sigma(m / 2))), 2), "interval_s": w["t"],
                        "window": w["index"], "evidence_keys": ks[:4], "status": "estimated",
                        "note": "the narrowest window; bounded by detected objects only (an undetected obstacle would narrow it)"})
    return out


# ---------- the run ----------

def analyse(m, mp4, words, opts, clock):
    import cv2
    import torch
    import detect_shot_cuts as dsc
    from fast_report import core, segment, timeline as tl, windows as win
    from fast_report.cascade import calibrate
    option = opts["geometry"]
    marks, lock = {}, threading.Lock()

    def mark(k):
        with lock:
            marks.setdefault(k, clock.now())
    path = f"/tmp/in-{threading.get_ident()}.mp4"
    Path(path).write_bytes(mp4)
    cap = cv2.VideoCapture(path)
    fps, n_total = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames, grays, futures, ranges, keys = [], [], [], [], []
    key_q, cv, decoded, built, stop = queue.Queue(), threading.Condition(), threading.Event(), threading.Event(), threading.Event()
    tasks, dev_q = queue.PriorityQueue(), {d: queue.PriorityQueue() for d in m.devs}
    seq = iter(range(10 ** 9))
    wins, results, errors, split_log = [], {}, [], []
    res_cv = threading.Condition()
    builder = win.Builder(opts["threshold"], opts.get("max_keys", win.MAX_KEYS), opts.get("carry", win.CARRY), min_keys=opts.get("min_keys", win.MIN_KEYS))
    cut_cache = {"n": -1, "cuts": []}
    # per window: SAM 3 chunks done, DA3 result (b) or shot geometry (a); the lift runs once both are in
    sam_parts, geo_raw, shot_of, lifted = {}, {}, {}, set()
    TASK_RANK = {"lift": 0, "da3": 1, "sam": 2}

    def put(kind, i, extra=None, dev=None):
        (dev_q[dev] if dev is not None else tasks).put((i * 4 + TASK_RANK[kind], next(seq), kind, i, extra))

    def maybe_lift(i):
        with lock:
            w = wins[i]
            ready = len(sam_parts.get(i, [])) == w["chunks"] and (i in geo_raw if option == "b" else i in shot_of)
            if not ready or i in lifted:
                return
            lifted.add(i)
        put("lift", i)

    def prefix_cuts(need):
        """Cuts the measure already covers up to frame `need` (+ margin), from the chunks done so far."""
        with cv:
            while True:
                done = 0
                while done < len(futures) and futures[done].done():
                    done += 1
                covered = ranges[done - 1][1] if done else 0
                final = decoded.is_set() and done == len(futures)
                if covered >= need or final:
                    break
                cv.wait(.02)
        if done != cut_cache["n"]:
            try:
                pm = core.stitch([f.result() for f in futures[:done]])  # a prefix: pairs and spans cut to its frames
                pm.update(keypoints=pm["keypoints"][:covered], inliers=pm["inliers"][:covered - 1], jump=pm["jump"][:covered - 1],
                          homographies=pm["homographies"][:covered - 1], spans=pm["spans"][:max(0, covered - dsc.SPAN)])
                c = core.cuts_from(pm, covered)["cuts"]
                cut_cache.update(n=done, cuts=[int(x) for x in c if final or x < covered - 25])
            except Exception as error:  # noqa: BLE001  recorded; the final cuts are checked after decode
                split_log.append({"prefix_cut_error": repr(error)[:200], "covered": covered})
        return cut_cache["cuts"]

    def dispatch(w, cuts):
        parts, cur = [], list(w["keys"])
        for c in cuts:  # a cut the prefix found late splits the window; fragments of carried keys only are dropped
            if cur and cur[0] < c <= cur[-1]:
                parts.append(([k for k in cur if k < c], w["carried"] if not parts else 0))
                cur = [k for k in cur if k >= c]
                split_log.append({"window_keys": [w["keys"][0], w["keys"][-1]], "cut": c})
        parts.append((cur, 0 if parts else w["carried"]))
        for ks, carried in parts:
            if not ks or len(ks) <= carried:
                continue
            with lock:
                i = len(wins)
                wins.append({"index": i, "keys": ks, "carried": carried, "reason": w["reason"], "times": [k / fps for k in ks],
                             "t": [round(ks[0] / fps, 3), round(ks[-1] / fps, 3)], "dispatched_s": clock.now(), "covis": w["covis"][-len(ks):],
                             "chunks": (len(ks) + segment.PERSON_FRAMES - 1) // segment.PERSON_FRAMES})
            mark("first_window_dispatched")
            if option == "b":
                put("da3", i)
            for s in range(0, len(ks), segment.PERSON_FRAMES):
                put("sam", i, s)
            with res_cv:
                res_cv.notify_all()

    def build():
        try:
            prev = None
            while True:
                k = key_q.get()
                if k is None:
                    break
                cuts = prefix_cuts(k + 30)
                with clock.stage("windows.rule", n={"key": k}):
                    f = win.features(grays[k].result()[0])
                    closed = builder.push(k, f, cut_before=prev is not None and any(prev < c <= k for c in cuts))
                if closed:
                    dispatch(closed, prefix_cuts(closed["keys"][-1] + 30))
                prev = k
            last = builder.finish()
            if last:
                dispatch(last, prefix_cuts(10 ** 9))
        except Exception:  # noqa: BLE001
            errors.append(traceback.format_exc()[-2000:])
        finally:
            built.set()
            with res_cv:
                res_cv.notify_all()

    def upload(i, dev, s=0, e=None):
        return torch.from_numpy(np.stack([frames[k] for k in wins[i]["keys"][s:e]])).to(dev)

    def work(dev):
        gi = dev.index
        with torch.cuda.device(dev), torch.inference_mode():
            while not stop.is_set():
                try:
                    item = dev_q[dev].get_nowait()
                except queue.Empty:
                    try:
                        item = tasks.get(timeout=.01)
                    except queue.Empty:
                        continue
                _, _, kind, i, s = item
                w = wins[i]
                try:
                    if kind == "sam":
                        w.setdefault("started_s", clock.now())
                        ks = w["keys"][s:s + segment.PERSON_FRAMES]
                        obj = [j for j, k in enumerate(ks) if (k // core.BLOCK) % opts.get("object_every", segment.OBJECT_EVERY) == 0]
                        person, voc = sam_chunk(m, dev, upload(i, dev, s, s + segment.PERSON_FRAMES), obj, words, clock, i)
                        with lock:
                            sam_parts.setdefault(i, []).append((s, dev, person, voc))
                        with res_cv:
                            res_cv.notify_all()
                        maybe_lift(i)
                    elif kind == "da3":
                        w.setdefault("started_s", clock.now())
                        with clock.stage(f"da3.shot{i}@gpu{gi}", gpu=gi, n={"views": len(w["keys"])}, sync=True):
                            g = m.da3s[dev].shot(upload(i, dev))
                        with lock:
                            geo_raw[i] = g
                        maybe_lift(i)
                    else:  # lift, on this card: SAM 3 chunks and geometry are brought here
                        person = gather([(p[0], p[1], p[2]) for p in sam_parts[i]], dev)
                        voc = gather([(p[0], p[1], p[3]) for p in sam_parts[i]], dev)
                        n = len(w["keys"])
                        if option == "b":
                            g = scale({k: v.to(dev) for k, v in geo_raw.pop(i).items()}, person, n, dev)
                            note = "window"
                        else:
                            si, pos = shot_of[i]
                            G = geo_shots[si]
                            mine = [j for j, q in enumerate(pos) if q >= 0]
                            src = torch.as_tensor([pos[j] for j in mine], device=G["depth_m"].device)
                            g = {"depth_m": torch.zeros((n, *core.DA3_HW), device=dev), "K": G["K"][src[:1]].repeat(n, 1, 1).to(dev),
                                 "c2w_m": torch.eye(4, device=dev).repeat(n, 1, 1)}
                            for k in ("depth_m", "K", "c2w_m"):  # keys of another shot keep depth 0: nothing of them is lifted or judged
                                g[k][mine] = G[k][src].to(dev)
                            g.update({k: G[k] for k in ("mpu", "plane", "scale_status")})
                            note = f"shot {si}"
                        r = objects_window(m, dev, w, upload(i, dev), person, voc, g, words, clock, i)
                        r.update(plane=g["plane"], mpu=g["mpu"], scale_status=g["scale_status"], frame_note=note, gpu=gi)
                        with lock:
                            sam_parts.pop(i, None)
                        done(i, r)
                except Exception:  # noqa: BLE001
                    errors.append(f"window {i} {kind}: " + traceback.format_exc()[-2500:])
                    done(i, None)

    def done(i, r):
        with res_cv:
            if i in results:
                return
            wins[i]["done_s"] = clock.now()
            results[i] = r
            res_cv.notify_all()

    workers = [threading.Thread(target=work, args=(d,), daemon=True) for d in m.devs for _ in range(m.per_gpu)]
    for t in workers:
        t.start()
    bt = threading.Thread(target=build, daemon=True)
    bt.start()

    # decode on this thread (the core's loop): grey + sharpness on threads, cut chunks in processes, keyframes one block behind
    with clock.stage("decode", n={"frames": n_total}):
        a, b0 = 0, 0

        def pick(b):
            keys.append(max(range(b, min(b + core.BLOCK, len(frames))), key=lambda f: grays[f].result()[1]))
            key_q.put(keys[-1])
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            frames.append(bgr)
            grays.append(m.cpu_pool.submit(core.gray_sharp, bgr))
            if len(frames) == a + core.CHUNK + dsc.SPAN + 1:
                a0 = max(0, a - 2)
                with cv:
                    futures.append(m.proc_pool.submit(core.measure_chunk, [g.result()[0] for g in grays[a0:]], a, a + core.CHUNK, a0, len(frames)))
                    ranges.append((a, a + core.CHUNK))
                    futures[-1].add_done_callback(lambda _: cv.acquire() and (cv.notify_all(), cv.release()))
                a += core.CHUNK
            if len(frames) == b0 + 2 * core.BLOCK:
                pick(b0)
                b0 += core.BLOCK
        cap.release()
        n = len(frames)
        while b0 < n:
            pick(b0)
            b0 += core.BLOCK
        gray_all = [g.result()[0] for g in grays]
        with cv:
            for a1 in range(a, n, core.TAIL):
                b1, lo = min(a1 + core.TAIL, n), max(0, a1 - 2)
                futures.append(m.proc_pool.submit(core.measure_chunk, gray_all[lo:min(n, b1 + dsc.SPAN + 1)], a1, b1, lo, min(n, b1 + dsc.SPAN + 1)))
                ranges.append((a1, b1))
                futures[-1].add_done_callback(lambda _: cv.acquire() and (cv.notify_all(), cv.release()))
            decoded.set()
            cv.notify_all()
        key_q.put(None)
    mark("decoded")

    geo_shots, overlap_rows, shots, cuts = {}, [], None, None

    def final_cuts():
        with clock.stage("cuts"):
            c = core.cuts_from(core.stitch([f.result() for f in futures]), n)
        mark("cuts_final")
        return c, [(s, e) for s, e in c["segments"] if e - s + 1 >= core.MIN_SHOT]

    if option == "a":  # per-shot DA3 on GPU 0 (all the shot's keyframes) while both cards work through the SAM 3 queue
        cuts, shots = final_cuts()
        dev0 = m.devs[0]
        with torch.cuda.device(dev0), torch.inference_mode():
            for si, (s, e) in enumerate(shots):
                ks = [k for k in keys if s <= k <= e]
                with clock.stage(f"da3.shot{si}", gpu=0, n={"views": len(ks)}, sync=True):
                    g = m.da3s[dev0].shot(torch.from_numpy(np.stack([frames[k] for k in ks])).to(dev0))
                geo_shots[si] = {"keys": ks, "g": g}
        mark("da3_done")
        bt.join()
        for si, G in geo_shots.items():  # the shot's floor from its windows' SAM 3 'floor' masks, then the scale
            idx = {k: j for j, k in enumerate(G["keys"])}
            floor = torch.zeros((len(idx), *core.DA3_HW), dtype=torch.bool, device=dev0)
            for i, w in enumerate(wins):
                if not any(k in idx for k in w["keys"]):
                    continue
                with res_cv:
                    res_cv.wait_for(lambda: len(sam_parts.get(i, [])) == w["chunks"] or i in results)
                for s, _, person, _ in list(sam_parts.get(i, [])):
                    f = person["word"] == 1
                    for fr, mk in zip(person["frame"][f].tolist(), person["mask"][f]):
                        k = w["keys"][s + fr]
                        if k in idx:
                            floor[idx[k]] |= mk.to(dev0)
            with torch.cuda.device(dev0), torch.inference_mode(), clock.stage(f"scale.shot{si}", gpu=0, sync=True):
                G.update(scale(G["g"], {"word": torch.ones(len(idx), dtype=torch.long, device=dev0), "frame": torch.arange(len(idx), device=dev0),
                                        "mask": floor}, len(idx), dev0))
        for i, w in enumerate(wins):  # each window on its shot (the one holding most of its keys)
            best = max(geo_shots.items(), key=lambda kv: sum(k in kv[1]["keys"] for k in w["keys"]), default=None)
            if best is None or not any(k in best[1]["keys"] for k in w["keys"]):
                w["skipped"] = "no geometry (short shot)"
                done(i, None)
                continue
            si, G = best
            w["foreign_keys"] = [k for k in w["keys"] if k not in G["keys"]]
            with lock:
                shot_of[i] = (si, [G["keys"].index(k) if k in G["keys"] else -1 for k in w["keys"]])
            maybe_lift(i)

    # windows in order -> map frames -> the tracker (starts as soon as window 0 is lifted)
    tracker, frame_of, T, comp = tl.Tracker(), {}, None, -1
    stitches, planes, people, per_window_inst, seen_inst, together = [], {}, {}, [], [], {}
    i = 0
    while True:
        with res_cv:
            res_cv.wait_for(lambda: i in results or (built.is_set() and i >= len(wins)))
        if i >= len(wins):
            break
        r, w = results[i], wins[i]
        if r is None:
            w["frame"] = None
            i += 1
            continue
        w.update(r)
        with clock.stage("stitch", n={"window": i}):
            if option == "a":
                si = shot_of[i][0]
                w["frame"] = frame_of.setdefault(si, len(frame_of))
                T = (1., np.eye(3), np.zeros(3))
            else:
                prev = next((x for x in reversed(wins[:i]) if x.get("frame") is not None), None)
                st = stitch(prev["raw"], w) if prev is not None and w["carried"] else None
                if st and st["accepted"] and prev["frame"] == comp:
                    T = compose(T, (st["s"], st["R"], st["t"]))
                    w["chain"] = prev["chain"] + abs(float(np.log(st["s"])))
                else:
                    comp += 1
                    T = (1., np.eye(3), np.zeros(3))
                    w["chain"] = 0.
                w["frame"] = comp
                stitches.append({"window": i, **({k: v for k, v in st.items() if k not in ("R", "t")} if st else
                                                 {"accepted": None, "why": "no carried keys" if not w["carried"] else "no shared depth"})})
                w["raw"] = {k: (w[k].copy() if hasattr(w[k], "copy") else w[k]) for k in ("keys", "carried", "c2w", "depth", "person")}
            to_frame(w, T)
            if w.get("plane") and w["frame"] not in planes:
                planes[w["frame"]] = w["plane"]
            seen = {p["key"] for p in people.get(w["frame"], [])}
            people.setdefault(w["frame"], []).extend(p for p in w["people"] if p["key"] not in seen)
        with clock.stage("timeline", n={"window": i, "instances": len(w["instances"])}):
            for x in w["instances"]:
                for k in x["keys"]:
                    together.setdefault((i, k), []).append(len(seen_inst))
                seen_inst.append((i, x))
            if seen_inst:  # 'moved' needs the appearance to match better than 99 % of this video's known negatives so far
                tracker.tau_move = calibrate(np.array([x["emb"] for _, x in seen_inst]), list(together.values()))[0]
            tracker.add(w)
            for x in w["instances"]:
                per_window_inst.append({"window": i, "frame": w["frame"], "label": x["label"], "centroid": np.round(x["centroid"], 3).tolist(),
                                        "lo": np.round(x["lo"], 3).tolist(), "hi": np.round(x["hi"], 3).tolist(), "keys": x["keys"], "best_key": x["best_key"]})
        w["facts_s"] = clock.now()
        mark("first_window_facts")
        if w["instances"]:
            mark("first_objects")
        i += 1
    with clock.stage("facts"):
        tau, calib = calibrate(np.array([x["emb"] for _, x in seen_inst]), list(together.values())) if seen_inst else (1.01, {})
        fct = facts(tracker, planes, people)
    mark("all_windows_facts")
    stop.set()
    if cuts is None:
        cuts, shots = final_cuts()  # (b): only to check the windows against them, after the facts
    if option == "a":  # the geometric co-visibility the window rule is compared with: after the facts, never on their clock
        dev0 = m.devs[0]
        for si, G in geo_shots.items():
            with torch.cuda.device(dev0), torch.inference_mode(), clock.stage("overlap.da3", gpu=0, sync=True):
                overlap_rows.append({"shot": si, "keys": G["keys"], "rows": da3_overlap(G["depth_m"], G["K"], G["c2w_m"])})
    for t in workers:
        t.join()
    summary = {"marks": marks, "stitches": stitches, "splits": split_log, "tau_move_calibrated": tau, "calibration": calib,
               "cuts_final": [int(c) for c in cuts["cuts"]], "shots": shots, "errors": errors, "overlap_da3": overlap_rows,
               "windows_spanning_final_cut": [w["index"] for w in wins if any(w["keys"][0] < c <= w["keys"][-1] for c in cuts["cuts"])]}
    return {"wins": wins, "tracker": tracker, "facts": fct, "summary": summary, "per_window_instances": per_window_inst, "people": people,
            "planes": planes, "fps": fps, "frames": n, "keys": keys}


def da3_overlap(depth_m, K, c2w_m, stride=8, span=60):
    """DA3's geometric co-visibility (a): for each keyframe i and later j <= i + span, the share of i's pixels (a stride
    grid, depth > 0) that land inside j's view and agree with j's depth within 2 sigma (seen, not hidden)."""
    import torch
    from fast_report import segment, timeline as tl
    n = len(depth_m)
    H, W = depth_m.shape[1:]
    w2c = torch.linalg.inv(c2w_m.double()).float()
    rows = []
    for i in range(n - 1):
        vy, vx = torch.nonzero(depth_m[i, ::stride, ::stride] > 0, as_tuple=True)
        p = segment.backproject(depth_m, K, c2w_m, torch.full_like(vy, i), vy, vx, stride)
        js = torch.arange(i + 1, min(n, i + span + 1), device=depth_m.device)
        c = torch.einsum("jab,nb->jna", w2c[js, :3, :3], p) + w2c[js, None, :3, 3]
        z = c[..., 2]
        u = (K[js, 0, 0, None] * c[..., 0] / z + K[js, 0, 2, None]).round().long()
        v = (K[js, 1, 1, None] * c[..., 1] / z + K[js, 1, 2, None]).round().long()
        ok = (z > .1) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        d = depth_m[js[:, None].expand_as(u), v.clamp(0, H - 1), u.clamp(0, W - 1)]
        sig = 2 * torch.sqrt(tl.POSE_M ** 2 + (tl.DEPTH_REL * z) ** 2)
        rows.append((ok & (d > 0) & ((d - z).abs() <= sig)).float().mean(1).cpu().numpy().round(3).tolist())
    return rows


# ---------- the container ----------

@app.cls(image=image, gpu="A100-80GB:2", cpu=CPU, memory=MEMORY_GIB * 1024, volumes=VOLUMES, timeout=3000, retries=0, max_containers=1,
         scaledown_window=30)
class X6:
    @modal.enter()
    def boot(self):
        import copy
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor
        from fast_report import core
        from fast_report.instrument import Vram
        entered, t0 = time.time(), time.perf_counter()
        sys.setswitchinterval(1e-3)
        b = self.boot_record = {"entered_unix": entered}
        lap = lambda k: b.__setitem__(k, round(time.perf_counter() - t0, 2))  # noqa: E731
        env = {"CUDA_MPS_PIPE_DIRECTORY": "/tmp/mps-pipe", "CUDA_MPS_LOG_DIRECTORY": "/tmp/mps-log"}
        for d in env.values():
            Path(d).mkdir(parents=True, exist_ok=True)
        os.environ.update(env)
        b["mps"] = subprocess.run(["nvidia-cuda-mps-control", "-d"], capture_output=True, text=True).returncode == 0
        self.proc_pool = ProcessPoolExecutor(PROCS, mp_context=multiprocessing.get_context("spawn"))
        self.proc_pool.map(core.warm_worker, range(PROCS))
        import torch
        from depth_anything_3.api import DepthAnything3
        from transformers import Sam3Model, Sam3Processor
        from fast_report import cascade, segment
        lap("imports_s")
        self.devs = [torch.device("cuda:0"), torch.device("cuda:1")]
        self.per_gpu = 2
        self.emb_locks = {d: threading.Lock() for d in self.devs}
        da3 = DepthAnything3.from_pretrained(DA3_MODEL, revision=DA3_REV, cache_dir="/v/da3/huggingface/hub").eval()
        self.da3s = {d: core.Da3(copy.deepcopy(da3).to(d), d) for d in self.devs}
        lap("da3_both_s")
        proc = Sam3Processor.from_pretrained(sam3_app.MODEL_ID, revision=sam3_app.REVISION, cache_dir="/v/sam3/huggingface/hub")
        sam = Sam3Model.from_pretrained(sam3_app.MODEL_ID, revision=sam3_app.REVISION, cache_dir="/v/sam3/huggingface/hub", torch_dtype=torch.bfloat16).eval()
        self.sams = {d: segment.Sam3(copy.deepcopy(sam).to(d), proc, d) for d in self.devs}
        self.embs = {d: cascade.Embedder(d, "/v/models/hf") for d in self.devs}
        lap("sam3_siglip_both_s")
        self.cpu_pool = ThreadPoolExecutor(CPU)
        with torch.inference_mode():
            for d in self.devs:
                with torch.cuda.device(d):
                    for k in (10, 40):
                        self.da3s[d].shot(torch.randint(0, 255, (k, 720, 1280, 3), dtype=torch.uint8, device=d))
                    noise = torch.randint(0, 255, (segment.PERSON_FRAMES, 720, 1280, 3), dtype=torch.uint8, device=d)
                    v = self.sams[d].vision(noise)
                    self.sams[d].detect(v, segment.PERSON_FRAMES, ("person", "floor"), segment.PERSON_SCORE, top=segment.PERSON_TOP)
                    self.sams[d].detect(self.sams[d].pick(v, [0, 3, 6]), 3, [f"word {i}" for i in range(57)], segment.VOCAB_SCORE)
                    masks = torch.zeros((300, 280, 504), dtype=torch.bool, device=d)
                    masks[:, 50:150, 100:300] = True
                    self.embs[d].crops(noise[:4], torch.randint(0, 4, (300,), device=d), masks)
                    self.embs[d].release()
                    torch.cuda.synchronize(d)
                    torch.cuda.empty_cache()
        lap("warm_s")
        b["process_pool_pids"] = len(set(self.proc_pool.map(core.warm_worker, range(PROCS))))
        self.vram = Vram([0, 1]).start()
        b.update(ready_s=round(time.perf_counter() - t0, 2), ready_unix=time.time(), gpus=gpu_listing(), torch=str(torch.__version__),
                 resident_gb=[round((torch.cuda.mem_get_info(d)[1] - torch.cuda.mem_get_info(d)[0]) / 2 ** 30, 2) for d in self.devs])

    @modal.method()
    def boot_info(self):
        return self.boot_record

    @modal.method()
    def run(self, mp4: bytes, words: list, opts: dict):
        import torch
        from fast_report.instrument import Clock, usd_per_s
        clock = Clock()  # t0: the bytes are in the container
        self.per_gpu = opts.get("per_gpu", 2)
        for d in self.devs:
            torch.cuda.reset_peak_memory_stats(d)
        try:
            out, error = analyse(self, mp4, words, opts, clock), None
        except Exception:  # noqa: BLE001  reported, never retried
            out, error = None, traceback.format_exc()[-4000:]
        rep = clock.report(self.vram, price_per_s=usd_per_s())
        rep["flags"] = [f for f in rep["flags"] if not f.startswith("unknown stage name")]
        result = {"error": error, "opts": opts, "clock": rep, "boot": self.boot_record,
                  "torch_reserved_peak_gb": [round(torch.cuda.max_memory_reserved(d) / 2 ** 30, 2) for d in self.devs]}
        if out is not None:
            result.update(export(out))
            run_id = opts["run_id"]
            p = Path("/v/layers/x6") / run_id
            p.mkdir(parents=True, exist_ok=True)
            with open(p / "windows.pkl", "wb") as f:  # the sweep's input: windows in their map frames, as the tracker saw them
                pickle.dump({"wins": [{k: v for k, v in w.items() if k != "raw" and not k.startswith("_")} for w in out["wins"]], "planes": out["planes"],
                             "people": out["people"], "tau": out["summary"]["tau_move_calibrated"]}, f)
            VOLUMES["/v/layers"].commit()
        for d in self.devs:
            with torch.cuda.device(d):
                torch.cuda.empty_cache()
        return json.loads(json.dumps(result, default=plain))


def plain(o):
    if isinstance(o, np.ndarray):
        return np.round(o, 4).tolist()
    return o.item() if hasattr(o, "item") else str(o)


def export(out):
    """The run as JSON: windows (no arrays), cameras per keyframe, objects with timelines, changes with the points to
    draw them, facts, per-window instances, marks."""
    tracker, wins = out["tracker"], out["wins"]
    cams = {}
    for w in wins:
        if w.get("frame") is None:
            continue
        for j, k in enumerate(w["keys"]):
            if k in (w.get("foreign_keys") or []):
                continue
            cams.setdefault(str(k), {"frame": w["frame"], "window": w["index"], "c2w": np.round(w["c2w"][j], 5).tolist(),
                                     "K_grid_stride2": np.round(w["K"][j], 3).tolist()})
    objs = {o["id"]: o for o in tracker.objects} if tracker else {}
    changes = []
    for c in (tracker.changes() if tracker else []):
        o = objs[c["object"]]
        c = dict(c)
        c["points_before"], c["points_after"] = pts(tracker, c, True), pts(tracker, c, False)
        c["label"] = max(o["votes"], key=o["votes"].get)
        changes.append(c)
    fact_points = {f["object"]: np.round(np.concatenate([i["points"] for _, i in objs[f["object"]]["obs"]])[::6], 3).tolist() for f in out["facts"] if f["object"]}
    keep = ("index", "keys", "carried", "reason", "t", "dispatched_s", "started_s", "done_s", "facts_s", "gpu", "frame", "mpu", "scale_status",
            "frame_note", "lift", "skipped", "foreign_keys", "covis", "chunks", "chain")
    return {"windows": [{k: w.get(k) for k in keep} | {"instances": len(w.get("instances", []))} for w in wins], "cameras": cams,
            "objects": tracker.timelines() if tracker else [], "changes": changes, "facts": out["facts"], "fact_points": fact_points,
            "per_window_instances": out["per_window_instances"], "summary": out["summary"], "fps": out["fps"], "frames": out["frames"],
            "keyframes": out["keys"], "planes": {str(k): {kk: np.round(v, 4).tolist() for kk, v in p.items()} for k, p in out["planes"].items()},
            "people": {str(k): [{"key": p["key"], "t": p["t"], "xyz": np.round(p["xyz"], 3).tolist()} for p in v] for k, v in out["people"].items()}}


# ---------- CPU sweep of the association thresholds on saved windows ----------

cpu_image = modal.Image.debian_slim(python_version="3.11").pip_install("numpy", "scipy").add_local_python_source("fast_report", "fast_report_app", "sam3_app")


def pts(tracker, c, before):
    """The object's points on the change's before (last window before it) or after (its window) side, every 2nd."""
    o = next(x for x in tracker.objects if x["id"] == c["object"])
    side = [inst for wi, inst in o["obs"] if (wi < c["window"] if before else wi == c["window"])]
    return np.round(side[-1 if before else 0]["points"][::2], 3).tolist() if side else None


@app.function(image=cpu_image, cpu=8, memory=16384, volumes={"/v/layers": VOLUMES["/v/layers"]}, timeout=1800, retries=0)
def sweep(run_id: str, grid: list):
    from fast_report import timeline as tl
    with open(f"/v/layers/x6/{run_id}/windows.pkl", "rb") as f:
        d = pickle.load(f)
    rows = []
    defaults = {k: getattr(tl, k) for k in ("FREE_SHARE", "NEIGH", "MIN_OBS_KEYS", "MAX_RANGE", "REL_MARGIN", "PERSON_GROW", "MIN_OBS_WINDOWS", "CHAIN_MAX")}
    for cfg in grid:
        t = time.perf_counter()
        for k, v in defaults.items():  # the change rule's constants, per config
            setattr(tl, k, cfg.get(k.lower(), v))
        tr = tl.Tracker(k_sigma=cfg["k_sigma"], iou_min=cfg["iou_min"], app_min=cfg.get("app_min"), tau_move=d["tau"])
        for w in d["wins"]:
            if w.get("frame") is not None:
                tr.add(w)
        ch = tr.changes()
        rows.append({"cfg": cfg, "s": round(time.perf_counter() - t, 2), "objects": [{"id": o["id"], "frame": o["frame"], "label": max(o["votes"], key=o["votes"].get),
                     "centroid": np.round(o["obs"][0][1]["centroid"], 3).tolist(), "windows": len({wi for wi, _ in o["obs"]})} for o in tr.objects],
                     "changes": [dict(c, points_before=pts(tr, c, True), points_after=pts(tr, c, False)) for c in ch],
                     "timelines": [{k: r[k] for k in ("id", "frame", "label", "positions", "windows_observed")} for r in tr.timelines()]})
        if cfg.get("facts"):  # the final rule: facts, their objects' points, full timelines
            t0 = time.perf_counter()
            fs = facts(tr, d["planes"], d["people"])
            objs = {o["id"]: o for o in tr.objects}
            rows[-1].update(facts=fs, facts_s=round(time.perf_counter() - t0, 3), timelines=tr.timelines(),
                            fact_points={f["object"]: np.round(np.concatenate([i["points"] for _, i in objs[f["object"]]["obs"]])[::6], 3).tolist()
                                         for f in fs if f["object"]})
    return json.loads(json.dumps(rows, default=plain))


@app.local_entrypoint()
def sweep_main(out: str, runs: str, mode: str = "all"):
    """The association and change-rule grid on saved windows (CPU containers, one per run)."""
    out = Path(out)
    ids, grids = [], []
    for rid in runs.split(","):
        run = json.loads(next(out.parent.glob(f"*/{rid}.json")).read_text())
        q = run["summary"]["calibration"].get("negative_cos_q50_q90_q99_max") or [None, None]
        if mode == "facts":
            grid = [{"k_sigma": 3., "iou_min": .2, "facts": True}]
        elif mode == "change":
            grid = [{"k_sigma": 3., "iou_min": .2, "neigh": nb, "rel_margin": rm, "max_range": mr, "min_obs_windows": mw, "chain_max": cm}
                    for nb in (0, 2) for rm in (0., .1, .2, .3) for mr in (5., 8.) for mw in (1, 2) for cm in (.1, 9.)]
        else:
            grid = [{"k_sigma": k, "iou_min": i, "app_min": a} for k in (2., 3., 5.) for i in (.1, .2, .3, 9.) for a in (None, q[0], q[1])]
            grid += [{"k_sigma": 3., "iou_min": .2, "neigh": nb, "rel_margin": rm, "max_range": mr, "person_grow": pg}
                     for nb in (0, 2) for rm in (0., .1, .2, .3) for mr in (5., 8.) for pg in (0, 2)]
        ids.append(rid)
        grids.append(grid)
    for rid, rows in zip(ids, sweep.starmap(list(zip(ids, grids)))):
        (out / f"sweep-{rid}.json").write_text(json.dumps(rows))
        print(rid, len(rows), "configs", flush=True)


# ---------- local ----------

def words_for(site):
    if site == "me340":  # the fast core's own vocabulary of this clip (fb-a-core-011, Qwen3-VL v1 prompt, 5 frames + core)
        p = RUNS / "fb-a-core-011/fb-me340-e84efffd-1790639941/patches/000008-objects.json"
        return json.loads(p.read_text())["data"]["words"], str(p.relative_to(PHASE2))
    g = json.loads((RUNS / "fb-d-harness-gaps-001/summary.json").read_text())["words"][site]["qwen-v1-5+core"]
    return g, "runs/fb-d-harness-gaps-001/summary.json words[site]['qwen-v1-5+core']"


@app.local_entrypoint()
def main(out: str, plan: str = "me340:b:0.4", per_gpu: int = 2, video_dir: str = ""):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    x = X6()
    submitted = time.time()
    boot = x.boot_info.remote()
    boot.update(client_submitted_unix=submitted, submit_to_ready_s_two_clocks=round(boot["ready_unix"] - submitted, 1))
    (out / "boot.json").write_text(json.dumps(boot, indent=1, default=plain))
    print("ready:", json.dumps({k: v for k, v in boot.items() if k.endswith("_s") or k in ("mps", "resident_gb")}), flush=True)
    for item in plan.split(","):
        site, geo, thr, *flags = item.split(":")  # flags: 'planted' (VIDEO_DIR/planted-SITE.mp4), 'p1' (one worker per card),
        # 'every1' (every 5 fps keyframe is an object keyframe, not every 3rd: the plants are seen close for ~2 s only)
        planted = "planted" in flags
        mp4 = (Path(video_dir) / f"planted-{site}.mp4" if planted else PHASE2 / "data/clips" / CLIPS[site] / "source-full.mp4").read_bytes()
        words, src = words_for(site)
        run_id = f"x6-{site}-{geo}-{thr}{''.join('-' + f for f in flags)}-{int(time.time())}"
        opts = {"geometry": geo, "threshold": float(thr), "run_id": run_id, "per_gpu": 1 if "p1" in flags else per_gpu, "words_source": src,
                "site": site, "planted": planted, **({"object_every": 1} if "every1" in flags else {})}
        t = time.time()
        r = x.run.remote(mp4, words, opts)
        r["client_wall_s"] = round(time.time() - t, 2)
        (out / f"{run_id}.json").write_text(json.dumps(r, indent=1))
        s = r.get("summary", {})
        print(run_id, "error" if r["error"] else "ok", json.dumps({"marks": s.get("marks"), "windows": len(r.get("windows", [])),
              "objects": len(r.get("objects", [])), "changes": len(r.get("changes", [])), "facts": len(r.get("facts", [])),
              "peaks": [g["peak_gb"] for g in r["clock"]["gpu_peak"]], "flags": r["clock"]["flags"], "errors": s.get("errors")})[:3000], flush=True)
        if r["error"]:
            print(r["error"][-3000:], flush=True)


def self_check():
    """Stitching on synthetic windows: a known Sim3 is recovered from two carried keyframes, the gate accepts it and
    refuses a rotated copy; compose and to_frame agree."""
    from fast_report import timeline as tl
    rng = np.random.default_rng(0)

    def rot(axis, deg):
        a = np.radians(deg)
        k = np.array(axis, float) / np.linalg.norm(axis)
        K_ = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        return np.eye(3) + np.sin(a) * K_ + (1 - np.cos(a)) * K_ @ K_
    c2w = np.stack([np.block([[rot([0, 1, 0], 5 * i), np.array([[.3 * i], [0], [0]])], [np.zeros((1, 3)), np.ones((1, 1))]]) for i in range(6)])
    depth = rng.uniform(2, 5, (6, 70, 126))
    prev = {"keys": list(range(6)), "carried": 0, "c2w": c2w, "depth": depth, "person": np.zeros(depth.shape, bool)}
    s, R, t = .8, rot([1, 2, 3], 20), np.array([1., -2, .5])  # cur frame -> prev frame
    inv = lambda p: (np.asarray(p) - t) @ R / s  # noqa: E731  prev -> cur
    cur_c2w = c2w[4:].copy()
    cur_c2w[:, :3, :3] = R.T @ c2w[4:, :3, :3]
    cur_c2w[:, :3, 3] = inv(c2w[4:, :3, 3])
    cur = {"keys": [4, 5, 6], "carried": 2, "c2w": np.concatenate([cur_c2w, cur_c2w[-1:]]), "depth": np.concatenate([depth[4:] / s, depth[-1:]]),
           "person": np.zeros((3, 70, 126), bool)}
    st = stitch(prev, cur)
    assert st["accepted"] and abs(st["s"] - s) < 1e-6 and np.allclose(st["R"], R) and np.allclose(st["t"], t), st
    bad = dict(cur, c2w=cur["c2w"].copy())
    bad["c2w"][1, :3, :3] = rot([0, 0, 1], 8) @ bad["c2w"][1, :3, :3]
    assert not stitch(prev, bad)["accepted"], "an 8 deg disagreement on a carried keyframe fails the 3 deg gate"
    A, B = (2., rot([0, 0, 1], 30), np.array([1., 0, 0])), (.5, rot([1, 0, 0], 10), np.array([0, 1., 0]))
    x = rng.normal(size=3)
    ab = compose(A, B)
    assert np.allclose(ab[0] * ab[1] @ x + ab[2], A[0] * A[1] @ (B[0] * B[1] @ x + B[2]) + A[2])
    pts = rng.normal(size=(20, 3))
    w = {"c2w": cur["c2w"].copy(), "depth": cur["depth"].astype(np.float16), "people": [],
         "instances": [{"points": pts, "centroid": pts.mean(0), "lo": pts.min(0), "hi": pts.max(0), "range_m": 2.}]}
    to_frame(w, (s, R, t))
    assert np.allclose(w["c2w"][:2], c2w[4:]) and np.allclose(w["instances"][0]["points"], s * pts @ R.T + t)
    assert np.allclose(w["depth"][:2], depth[4:], rtol=2e-3), "depth carried by the scale"
    assert w["instances"][0]["range_m"] == 2. * s and tl.sigma(0) == tl.POSE_M
    # facts on a made-up tracker: a shelf face 5 deg off vertical, boxes topping at 2.0 m, a person 0.5 m from a pallet jack
    import types
    up = np.array([0., -1, 0])
    g = np.stack(np.meshgrid(np.linspace(0, 2, 12), np.linspace(0, 1.8, 12)), -1).reshape(-1, 2)
    shelf = np.stack([g[:, 0], -g[:, 1], 3 + np.tan(np.radians(5)) * g[:, 1]], 1)
    stack = np.stack(np.meshgrid(np.linspace(-2, -1, 6), -np.linspace(0, 2., 11), np.linspace(2, 3, 6)), -1).reshape(-1, 3)
    jack = np.stack(np.meshgrid(np.linspace(3, 4, 6), -np.linspace(0, .3, 4), np.linspace(1, 1.5, 4)), -1).reshape(-1, 3)
    inst = lambda p, k: {"points": p, "centroid": p.mean(0), "range_m": 3., "times": [k / 10], "best_key": k}  # noqa: E731
    obj = lambda i, lb, p: {"id": f"g{i}", "frame": 0, "votes": {lb: 1.}, "obs": [(0, inst(p, 10)), (1, inst(p, 20))]}  # noqa: E731
    tr = types.SimpleNamespace(objects=[obj(0, "shelves", shelf), obj(1, "stacked boxes", stack), obj(2, "pallet jack", jack)], windows=[])
    fs = {f["property"].split(" ")[0]: f for f in facts(tr, {0: {"up": up, "point": np.zeros(3)}}, {0: [{"xyz": np.array([3.5, -1., 2.]), "t": 1.5, "key": 15}]})}
    assert abs(fs["top"]["value"] - 1.92) < .1 and fs["top"]["label"] == "stacked boxes", fs["top"]  # the 95th percentile of 0-2 m
    assert abs(fs["tilt"]["value"] - 5.) < .5 and fs["tilt"]["label"] == "shelves", fs["tilt"]
    assert abs(fs["closest"]["value"] - .5) < .05 and fs["closest"]["interval_s"] == [1.5, 1.5], fs["closest"]
    print("x6 app self-check ok: Sim3 from carried keyframes, 3 deg gate, compose, to_frame, facts (height, tilt, person)")


if __name__ == "__main__":
    assert sys.argv[1:] == ["--self-check"], __doc__
    self_check()
