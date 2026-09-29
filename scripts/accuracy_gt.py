"""Physical accuracy of the click MVP's cards against independent metric ground truth (R5).

Three RGB-D sequences with metric camera truth: ARKitScenes raw 47333932 (ARKit poses, LiDAR depth, confident pixels
only), ARKitScenes 42445448 (ARKit poses, laser-scan depth at 1920x1440, annotated boxes) and TUM fr1 room (motion-capture
poses, Kinect depth). The MVP sees RGB only: each sequence becomes 16:9 1280x720 MP4 windows (centre crop, the MVP's input
contract). ARKit captures have irregular frame times (0.1-15 s apart): every capture frame is held for 6 video frames at
30 fps, so each one is exactly one 5 fps keyframe (the pipeline's rule: the sharpest of each 6-frame block); held frames
read as a cut every 0.2 s to the cut rule, so those runs pass cuts='none' (the captures are unedited). TUM is a real
30 fps video and runs with the cut rule on. Depth and poses are never uploaded.

Ground truth per card: the card's own pick-map regions (the segmented keyframes of its entity id, eroded 1 px at
640x360, GT depth edges dropped) are lifted with GT depth + GT poses into a GT floor frame built exactly like the card's
(origin = the shot's first keyframe camera dropped onto the floor, +z = floor normal, +x = that camera's forward on the
floor). The GT floor is the lowest horizontal plane of the fused GT depth. Values use the card's definitions: p2/p98
heights, footprint sides along the card's own footprint axes (p2-p98), the footprint centre, measure_observed_points'
angles. Errors are reported (a) as delivered (scale from the assumed 1.6 m camera height) and (b) at true scale (the
card's metres x s, s = the Sim3 scale of the shot's cameras onto the GT cameras): (a) - (b) is the scale error, (b) the
geometry error.

    python scripts/accuracy_gt.py prepare --out RUNS/mvp2-accuracy-inputs
    python scripts/accuracy_gt.py score RUN_DIR [--inputs RUNS/mvp2-accuracy-inputs]    # every mirrored report in RUN_DIR
    python scripts/accuracy_gt.py --self-check
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "scripts")]
PHASE2 = Path("/Users/adam/Desktop/panoptes-public/research-notes/phase2")
DATA, RUNS = PHASE2 / "data", PHASE2 / "runs"
W, H, HOLD = 1280, 720, 6
SEQUENCES = {  # name -> source, windows of source frames [a, b)
    "arkit47": {"kind": "arkit", "frames": DATA / "arkit-raw-47333932/clip", "poses": RUNS / "arkit-47333932-cameras-080/prediction.npz",
                "depth": "evaluation_only/lowres_depth_confident", "depth_wh": (256, 192), "rot180": False, "windows": [(0, 128), (128, 257)]},
    "arkit42": {"kind": "arkit", "frames": DATA / "arkit-42445448", "poses": RUNS / "arkit-42445448-cameras-040/prediction.npz",
                "depth": "evaluation_only/highres_depth", "depth_wh": (1920, 1440), "rot180": True, "windows": [(0, 47)]},
    "tum": {"kind": "tum", "frames": DATA / "rgbd_dataset_freiburg1_room", "windows": [(0, 900), (900, 1362)]},
}
TUM_DEPTH_K = (525., 525., 319.5, 239.5)  # TUM's registered depth: the ROS default set, no undistortion (TUM file-format page)
TUM_DEPTH_SCALE = 5000.


# ---------------------------------------------------------------- sources

def arkit_rows(seq):
    """[(stamp, image path, c2w metric (z up), K at the source raster)] for an ARKit clip (prepare_arkit_clip's layout)."""
    names = [ln.split() for ln in (seq["frames"] / "rgb.txt").read_text().splitlines() if ln and not ln.startswith("#")]
    pred = np.load(seq["poses"])
    c2w, k = pred["poses_c2w"].astype(float), pred["keyframe_final_fullres_intrinsics"].astype(float) * 6  # 320x240 model K -> 1920x1440
    assert len(names) == len(c2w) == len(k), (len(names), len(c2w), len(k))
    return [(float(s), seq["frames"] / p, c2w[i], k[i]) for i, (s, p) in enumerate(names)]


def tum_rows(seq):
    """TUM: RGB frames with the GT pose interpolated at their stamp and the nearest depth within 20 ms."""
    from scipy.spatial.transform import Rotation, Slerp
    root = seq["frames"]
    read = lambda f: [ln.split() for ln in (root / f).read_text().splitlines() if ln and not ln.startswith("#")]  # noqa: E731
    rgb, depth, gt = read("rgb.txt"), read("depth.txt"), np.loadtxt(root / "groundtruth.txt")
    dt = np.array([float(s) for s, _ in depth])
    slerp = Slerp(gt[:, 0], Rotation.from_quat(gt[:, 4:8]))
    out = []
    for s, p in rgb:
        t = float(s)
        j = int(np.argmin(np.abs(dt - t)))
        if not gt[0, 0] <= t <= gt[-1, 0]:
            c2w = None
        else:
            c2w = np.eye(4)
            c2w[:3, :3] = slerp([t]).as_matrix()[0]
            c2w[:3, 3] = [np.interp(t, gt[:, 0], gt[:, i]) for i in (1, 2, 3)]
        out.append((t, root / p, c2w, np.array([517.306408, 516.469215, 318.643040, 255.313989]),
                    root / depth[j][1] if abs(dt[j] - t) <= .02 else None))
    return out


def crop_params(src_w, src_h):
    """Centre 16:9 crop of a 4:3 source, scaled to 1280x720: (y0, scale)."""
    ch = src_w * H / W
    return (src_h - ch) / 2, W / src_w


def to_video(img, y0, sc):
    import cv2
    ch = int(round(img.shape[1] * H / W))
    return cv2.resize(img[int(round(y0)):int(round(y0)) + ch], (W, H), interpolation=cv2.INTER_AREA if sc < 1 else cv2.INTER_LINEAR)


def prepare(out):
    """MP4 windows + gt.json per sequence (never uploaded): per source frame its pose, depth file and the raster maps."""
    import cv2
    out.mkdir(parents=True, exist_ok=False)
    for name, seq in SEQUENCES.items():
        d = out / name
        d.mkdir()
        if seq["kind"] == "arkit":
            rows = [(s, p, c, k, seq["frames"] / seq["depth"] / f"{s:.3f}.png") for s, p, c, k in arkit_rows(seq)]
        else:
            rows = tum_rows(seq)
        first = cv2.imread(str(rows[0][1]))
        sh, sw = first.shape[:2]
        y0, sc = crop_params(sw, sh)
        frames = []
        for s, p, c2w, k, dp in rows:
            if c2w is not None and seq.get("rot180"):  # upside-down capture: the image turned 180 deg, the camera about its axis
                c2w = c2w @ np.diag([-1., -1, 1, 1])
                k = np.array([k[0], k[1], sw - 1 - k[2], sh - 1 - k[3]])
            frames.append({"stamp": s, "image": str(p), "c2w": None if c2w is None else np.round(c2w, 6).tolist(), "K_src": np.round(k, 4).tolist(),
                           "depth": str(dp) if dp is not None and Path(dp).exists() else None})
        if seq["kind"] == "arkit":
            dw, dh = seq["depth_wh"]
            depth = {"wh": [dw, dh], "K": "K_src x depth_w / source_w", "unit_m": .001, "rot180": seq["rot180"]}
        else:
            depth = {"wh": [640, 480], "K": list(TUM_DEPTH_K), "unit_m": 1 / TUM_DEPTH_SCALE, "rot180": False}
        windows = []
        for a, b in seq["windows"]:
            hold = HOLD if seq["kind"] == "arkit" else 1
            path = d / f"window-{a}-{b}.mp4"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"avc1"), 30., (W, H))
            assert writer.isOpened(), "no H.264 encoder"
            for f in frames[a:b]:
                img = cv2.imread(f["image"])
                if seq.get("rot180"):
                    img = img[::-1, ::-1]
                v = to_video(img, y0, sc)
                for _ in range(hold):
                    writer.write(v)
            writer.release()
            windows.append({"mp4": path.name, "source_frames": [a, b], "hold": hold, "cuts": "none" if hold > 1 else "rule"})
        (d / "gt.json").write_text(json.dumps({"name": name, "kind": seq["kind"], "source_wh": [sw, sh], "crop_y0": y0, "scale": sc,
                                               "rot180": bool(seq.get("rot180")), "depth": depth, "windows": windows, "frames": frames}))
        print(name, len(frames), "frames", [w["mp4"] for w in windows], flush=True)


# ---------------------------------------------------------------- GT geometry

class Seq:
    """One prepared sequence: GT depth lookups for 1280x720 video pixels, GT poses, the GT floor."""

    def __init__(self, folder):
        self.folder = Path(folder)
        self.g = json.loads((self.folder / "gt.json").read_text())
        self.sw, self.sh = self.g["source_wh"]
        self.dw, self.dh = self.g["depth"]["wh"]
        self._depth = {}
        floor = self.folder / "floor.json"
        self.floor = json.loads(floor.read_text()) if floor.exists() else None

    def c2w(self, i):
        c = self.g["frames"][i]["c2w"]
        return None if c is None else np.asarray(c, float)

    def depth_k(self, i):
        d = self.g["depth"]
        if isinstance(d["K"], list):
            return np.asarray(d["K"], float)
        k = np.asarray(self.g["frames"][i]["K_src"], float)
        s = self.dw / self.sw
        return np.array([k[0] * s, k[1] * s, (k[2] + .5) * s - .5, (k[3] + .5) * s - .5])

    def depth(self, i):
        """GT depth (metres, 0 = none) of source frame i on the depth raster (turned like the image)."""
        import cv2
        if i not in self._depth:
            p = self.g["frames"][i]["depth"]
            if p is None:
                self._depth[i] = None
            else:
                d = cv2.imread(p, cv2.IMREAD_UNCHANGED).astype(np.float32) * self.g["depth"]["unit_m"]
                if self.g["depth"]["rot180"]:
                    d = d[::-1, ::-1].copy()
                if len(self._depth) > 64:
                    self._depth.pop(next(iter(self._depth)))
                self._depth[i] = d
        return self._depth[i]

    def video_to_depth_px(self, u, v):
        """1280x720 video pixel centres -> depth raster pixel (float)."""
        sx = (np.asarray(u, float) + .5) / self.g["scale"] - .5
        sy = (np.asarray(v, float) + .5) / self.g["scale"] - .5 + self.g["crop_y0"]
        return (sx + .5) * self.dw / self.sw - .5, (sy + .5) * self.dh / self.sh - .5

    def lift(self, i, u, v, edge=.05):
        """World points (GT) and their camera depth for video pixels (u, v) on source frame i: nearest depth pixel, dropped
        where the 3x3 depth neighbourhood jumps by more than `edge` of the depth (mixed pixels at object borders) or has no
        depth."""
        d, c2w = self.depth(i), self.c2w(i)
        if d is None or c2w is None or not len(u):
            return np.zeros((0, 3)), np.zeros(0)
        x, y = self.video_to_depth_px(u, v)
        xi, yi = np.clip(np.round(x).astype(int), 1, self.dw - 2), np.clip(np.round(y).astype(int), 1, self.dh - 2)
        z = d[yi, xi]
        nb = np.stack([d[yi + a, xi + b] for a in (-1, 0, 1) for b in (-1, 0, 1)], 1)
        ok = (z > 0) & (nb.min(1) > 0) & ((nb.max(1) - nb.min(1)) <= edge * z)
        k = self.depth_k(i)
        cam = np.stack([(x[ok] - k[2]) / k[0] * z[ok], (y[ok] - k[3]) / k[1] * z[ok], z[ok]], 1)
        return cam @ c2w[:3, :3].T + c2w[:3, 3], z[ok]

    def floor_frame(self, i):
        """The card's floor frame (cards.floor_frame) built on GT: source frame i's camera, the GT floor."""
        from fast_report import cards
        return cards.floor_frame(self.c2w(i), self.floor["normal"], self.floor["point"])

    def fused(self, stride=8, frames=None):
        """GT points of whole frames (every `stride`-th depth pixel)."""
        out = []
        for i in frames if frames is not None else range(len(self.g["frames"])):
            d, c2w = self.depth(i), self.c2w(i)
            if d is None or c2w is None:
                continue
            ys, xs = np.mgrid[1:self.dh - 1:stride, 1:self.dw - 1:stride]
            z = d[ys, xs]
            ok = z > 0
            k = self.depth_k(i)
            cam = np.stack([(xs[ok] - k[2]) / k[0] * z[ok], (ys[ok] - k[3]) / k[1] * z[ok], z[ok]], 1)
            out.append(cam @ c2w[:3, :3].T + c2w[:3, 3])
        return np.concatenate(out) if out else np.zeros((0, 3))


def fit_floor(P, cams, up_axis=2, bin_m=.02, min_share=.01):
    """The lowest horizontal plane of the GT points: the lowest height bin (along the world's up axis) holding >= min_share
    of the points below the cameras, refined by a least-squares plane on the points within 3 cm of it (normal within 10 deg
    of up). -> {normal (towards the cameras), point, residual_p90_m, inliers, camera_height_m (median)}."""
    up = np.eye(3)[up_axis]
    h = P @ up
    below = h < np.median(cams @ up)
    hb = h[below]
    edges = np.arange(hb.min(), hb.max() + bin_m, bin_m)
    cnt, _ = np.histogram(hb, edges)
    lvl = edges[np.flatnonzero(cnt >= min_share * len(hb))[0]] + bin_m / 2
    near = P[np.abs(h - lvl) <= .03]
    n = up
    for _ in range(3):
        c = near.mean(0)
        e, v = np.linalg.eigh((near - c).T @ (near - c))
        n = v[:, 0] * np.sign(v[:, 0] @ up)
        assert np.degrees(np.arccos(min(1., n @ up))) < 10, "floor plane not horizontal"
        r = (near - c) @ n
        near = near[np.abs(r) <= max(.01, 2.5 * np.median(np.abs(r)))]
    c = near.mean(0)
    r = np.abs((P - c) @ n)
    return {"normal": n.tolist(), "point": c.tolist(), "residual_p90_m": float(np.percentile(np.abs((near - c) @ n), 90)),
            "inliers": int(len(near)), "points": int(len(P)), "camera_height_m": float(np.median((cams - c) @ n)),
            "share_within_3cm": float((r <= .03).mean())}


def gt_floor(folder):
    s = Seq(folder)
    idx = [i for i in range(len(s.g["frames"])) if s.c2w(i) is not None]
    P = s.fused(stride=8 if s.dw <= 640 else 32, frames=idx[:: max(1, len(idx) // 300)])
    cams = np.array([s.c2w(i)[:3, 3] for i in idx])
    f = fit_floor(P, cams)
    (Path(folder) / "floor.json").write_text(json.dumps(f, indent=1))
    return f


# ---------------------------------------------------------------- scoring

FIELDS = {"top_above_floor": "height", "base_above_floor": "height", "height": "extent", "width": "extent", "depth": "extent",
          "position_xy": "position", "planar_slope_deg": "angle", "principal_axis_tilt_deg": "angle", "visible_length": "extent"}
MIN_GT_POINTS, MIN_GT_COVER = 50, .5


def to_card_frame(P, ff):
    """Shot-frame points -> a cards-layer floor frame {origin_m, x, z}."""
    z, x = np.asarray(ff["z"], float), np.asarray(ff["x"], float)
    return (np.asarray(P, float) - ff["origin_m"]) @ np.stack([x, np.cross(z, x), z]).T


def rigid2(A, B):
    """2D rotation R and translation t minimising sum |B - (R A + t)|^2."""
    ma, mb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((B - mb).T @ (A - ma))
    R = U @ np.diag([1., np.sign(np.linalg.det(U @ Vt))]) @ Vt
    return R, mb - R @ ma


def tail_trim(z):
    """The card's per-mask depth-tail trim (section 4.2 step 2): keep [p15 - m, p85 + m], m = 0.25 (p85 - p15) + 0.05 z_med."""
    if len(z) < 10:
        return np.ones(len(z), bool)
    a, b = np.percentile(z, [15, 85])
    m = .25 * (b - a) + .05 * np.median(z)
    return (z >= a - m) & (z <= b + m)


def regions(pick, seq, src, cap=2500):
    """{entity: {'P': GT world points, 'views': n, 'px': mask pixels sampled, 'ok': GT points kept}} from the pick layer's
    segmented frames: each entity's region eroded 1 px at the map's 640x360, sampled, lifted with GT depth and pose,
    per-view depth tails trimmed (the card's rule)."""
    import cv2
    rng = np.random.default_rng(0)
    out = {}
    ent = pick.data["entities"]
    for i, f in enumerate(pick.frames):
        if f.get("source") != "segmented":
            continue
        m = pick.map(i)
        sy, sx = H / f["h"], W / f["w"]
        for code in np.unique(m):
            if code == 0 or not str(ent[code]).startswith("obj-"):
                continue
            mk = cv2.erode((m == code).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            ys, xs = np.nonzero(mk)
            if not len(ys):
                continue
            if len(ys) > cap:
                pick_ix = rng.choice(len(ys), cap, replace=False)
                ys, xs = ys[pick_ix], xs[pick_ix]
            P, z = seq.lift(src(f["frame"]), (xs + .5) * sx - .5, (ys + .5) * sy - .5)
            keep = tail_trim(z)
            r = out.setdefault(ent[code], {"P": [], "views": 0, "px": 0, "ok": 0, "frames": []})
            r["px"] += len(ys)
            r["ok"] += int(keep.sum())
            if keep.sum():
                r["P"].append(P[keep])
                r["views"] += 1
                r["frames"].append(int(f["frame"]))
    return out


def gt_card(P, card, fr):
    """GT values of one card from its GT points (world): the card's definitions in the GT floor frame `fr`, on the card's
    own footprint axes. -> {field: value} (angles only when measure_observed_points gives them)."""
    from fast_report import cards as fc
    Q = fc.to_floor(P, fr)
    Q = Q[fc.main_cluster(Q, fc.EPS_MIN)]
    ph = card["physical"]
    c = np.asarray(ph["footprint_xy"]["value"], float)
    e = np.stack([c[1] - c[0], c[3] - c[0]])
    sides = np.linalg.norm(e, axis=1)
    axes = e / np.maximum(sides[:, None], 1e-9) if sides.min() > 1e-6 else fc.footprint_axes(Q[:, :2])
    b = fc.box_of(Q, axes)
    out = {"top_above_floor": b["top"], "base_above_floor": b["base"], "height": b["top"] - b["base"], "position_xy": b["centre_xy"].tolist(),
           "n_points": int(len(Q))}
    w = (ph.get("width") or {}).get("value", (ph.get("width") or {}).get("visible_m"))
    wi = int(np.argmin(np.abs(sides - w))) if w is not None else 0
    out["width"], out["depth"] = float(b["sides"][wi]), float(b["sides"][1 - wi])
    out["visible_length"] = float(max(b["sides"]))  # r4: the GT points are the card's own segmented regions: the length seen
    o = fc.orient(Q, 1.)
    for name in ("planar_slope_deg", "principal_axis_tilt_deg"):
        if o[name][0] is not None and (o[name][1] or 0) <= fc.FIT_MAX_DEG:
            out[name] = float(o[name][0])
    return out


def score_report(run_dir, report, inputs):
    """Rows (one per card x field) of one mirrored report against its sequence's GT, plus per-shot scale records."""
    import fast_report_eval as ev
    from fast_report import cards as fc
    from fast_report.core import umeyama, CAMERA_HEIGHT_M
    run = json.loads((run_dir / "reports" / report / "run.json").read_text())
    call = run["call"]
    name, wi = call["site"].split("-")[1], int(call["site"].rsplit("-w", 1)[1])
    geometry = call["options"].get("label", call["options"].get("geometry", "shot"))
    seq = Seq(inputs / name)
    w = seq.g["windows"][wi]
    a0, hold = w["source_frames"][0], w["hold"]
    src = lambda f: a0 + int(f) // hold  # noqa: E731
    L = ev.load_layers(run_dir, report)
    pick = ev.run_picks(run_dir, report, L)[-1][1]
    oc = L["object_cards"]
    k = (oc.get("calibration") or {}).get("k") or {}
    alias = oc.get("aliases") or {}
    shots = {}
    for s in L["cameras"]["shots"]:
        keys, C = s["keys"], np.asarray(s["c2w"], float)[:, :3, 3]
        G = [seq.c2w(src(f)) for f in keys]
        ok = np.array([g is not None for g in G])
        Gc = np.array([g[:3, 3] for g in G if g is not None])
        sc, R, t = umeyama(C[ok], Gc)
        n, p0 = np.asarray(seq.floor["normal"]), np.asarray(seq.floor["point"])
        h_true = float(np.median((Gc - p0) @ n))
        fr = seq.floor_frame(src(keys[0]))
        ff = next((x["floor_frame"] for x in oc["shots"] if x["index"] == s["index"]), None)
        # the card frame's yaw is set by the first camera's forward on the floor: a camera looking down makes it fragile
        # (arkit47 w0 starts on a poster on the floor), so positions are compared after a floor-plane rigid fit of the
        # shot's cameras (card frame -> GT frame): (a) at the card's scale, (b) at the true camera height's
        mine = to_card_frame(C[ok], ff) if ff else None
        gtf = fc.to_floor(Gc, fr)
        shots[s["index"]] = {"s": h_true / CAMERA_HEIGHT_M, "s_sim3": sc, "ate_sim3_m": float(np.sqrt(np.mean(np.sum((sc * C[ok] @ R.T + t - Gc) ** 2, 1)))),
                             "path_m": float(np.linalg.norm(np.diff(Gc, axis=0), axis=1).sum()), "keys": len(keys),
                             "gt_camera_height_m": h_true, "mpu": s.get("mpu"), "scale_status": s.get("scale_status"), "fr": fr,
                             "align_a": rigid2(mine[:, :2], gtf[:, :2]) if ff else None,
                             "align_b": rigid2(h_true / CAMERA_HEIGHT_M * mine[:, :2], gtf[:, :2]) if ff else None}
        for key in ("align_a", "align_b"):
            if shots[s["index"]][key]:
                shots[s["index"]][key + "_yaw_deg"] = float(np.degrees(np.arctan2(shots[s["index"]][key][0][1, 0], shots[s["index"]][key][0][0, 0])))
    reg = regions(pick, seq, src)
    merged = {}
    for e, r in reg.items():
        cid = alias.get(e, e)
        while cid in alias and alias[cid] != cid:
            cid = alias[cid]
        m = merged.setdefault(cid, {"P": [], "views": 0, "px": 0, "ok": 0})
        for key in ("views", "px", "ok"):
            m[key] += r[key]
        m["P"] += r["P"]
    rows, bounds = [], []
    for c in oc["cards"]:
        if c.get("kind") != "object" or "footprint_xy" not in (c.get("physical") or {}) or c["id"] not in merged:
            continue
        m, s = merged[c["id"]], shots.get(c["shot"])
        if s is None or not m["P"]:
            continue
        P = np.concatenate(m["P"])
        cover = m["ok"] / max(m["px"], 1)
        if len(P) < MIN_GT_POINTS:
            continue
        gt = gt_card(P, c, s["fr"])
        ph = c["physical"]
        for f, fam in FIELDS.items():
            if f not in gt:
                continue
            got = ev.fact(ph.get(f), k.get(fam, 1.), angle=fam == "angle")
            x = ph.get(f) or {}
            if f == "visible_length" and x.get("value") is not None:  # r4: always a lower bound ('needs review' overwrites its status)
                got, x = None, {**x, "status": "at least"}
            if got is None and x.get("status") in ("at least", "at most") and x.get("value") is not None and fam in ("height", "extent"):
                lo = x["status"] == "at least"
                u_ns = float(np.sqrt(max(x["u"] ** 2 - float((x.get("parts") or {}).get("scale", 0.)) ** 2, 0.)))  # (b): without the scale term
                bounds.append({"report": report, "seq": name, "geometry": geometry, "card": c["id"], "field": f, "status": x["status"],
                               "value": x["value"], "u": x["u"], "gt": gt[f], "s": s["s"], "gt_cover": round(cover, 3), "first_call": bool(run.get("first_call")),
                               "holds_a": bool(gt[f] >= x["value"] - x["u"]) if lo else bool(gt[f] <= x["value"] + x["u"]),
                               "holds_b": bool(gt[f] >= s["s"] * (x["value"] - u_ns)) if lo else bool(gt[f] <= s["s"] * (x["value"] + u_ns))})
            if got is None:
                continue
            v, u, u_ns = got
            parts = (ph.get(f) or {}).get("parts") or {}
            if fam == "position":
                if not s["align_a"]:
                    continue
                (Ra, ta), (Rb, tb) = s["align_a"], s["align_b"]
                e_a = float(np.linalg.norm(Ra @ np.asarray(v) + ta - gt[f]))
                e_b = float(np.linalg.norm(Rb @ (s["s"] * np.asarray(v)) + tb - gt[f]))
                sa = sb = None
            elif fam == "angle":
                e_a = e_b = float(v - gt[f])
                sa = sb = e_a
            else:
                sa, sb = float(v - gt[f]), float(s["s"] * v - gt[f])
                e_a, e_b = abs(sa), abs(sb)
            rows.append({"report": report, "seq": name, "window": wi, "geometry": geometry, "card": c["id"], "shot": c["shot"],
                         "name": (c.get("identity") or {}).get("name"), "field": f, "family": fam, "value": v, "u": u, "u_noscale": u_ns,
                         "gt": gt[f], "s": s["s"], "err_a": abs(e_a), "signed_a": sa, "err_b": abs(e_b), "signed_b": sb,
                         "cov_a": bool(abs(e_a) <= u), "cov_b": bool(abs(e_b) <= (s["s"] * u_ns if fam != "angle" else u)),
                         "status": (ph.get(f) or {}).get("status"), "n_subsets": (ph.get(f) or {}).get("n_subsets"),
                         "parts": parts, "views": (c.get("views") or {}).get("n"), "distance_m": (c.get("views") or {}).get("distance_m"),
                         "plausible": (ph.get("size_check") or {}).get("status") != "implausible",
                         "gt_points": gt["n_points"], "gt_views": m["views"], "gt_cover": round(cover, 3), "scale_status": s["scale_status"],
                         "first_call": bool(run.get("first_call"))})
    shot_rec = {i: {k2: v2 for k2, v2 in x.items() if k2 not in ("fr", "align_a", "align_b")} for i, x in shots.items()}
    return rows, {"report": report, "seq": name, "window": wi, "geometry": geometry, "first_call": run.get("first_call"), "shots": shot_rec, "bounds": bounds,
                  "milestones": {k2: v2.get("written_s") for k2, v2 in (run.get("milestones") or {}).items()},
                  "gpu_peak_gib": [g["peak_gb"] for g in run.get("gpu_peak", [])], "flags": run.get("flags"),
                  "stages": {st["stage"]: {"s": round(st["end_s"] - st["start_s"], 3), "end_s": st["end_s"], "peak_gb": st.get("peak_gb")}
                             for st in run.get("stages", []) if str(st.get("stage", "")).startswith(("da3", "scale", "cards", "lift", "densify"))},
                  "geometry_record": ((run.get("summary") or {}).get("geometry"))}


def table(rows, gate=lambda r: True):
    """Per (seq, geometry, field): n, median / p90 |err|, signed median, coverage, for (a) and (b)."""
    out = {}
    for r in rows:
        if gate(r):
            out.setdefault((r["seq"], r["geometry"], r["field"]), []).append(r)
    res = []
    for (sq, g, f), rs in sorted(out.items()):
        ea, eb = np.array([r["err_a"] for r in rs]), np.array([r["err_b"] for r in rs])
        sa = [r["signed_a"] for r in rs if r["signed_a"] is not None]
        sb = [r["signed_b"] for r in rs if r["signed_b"] is not None]
        res.append({"seq": sq, "geometry": g, "field": f, "n": len(rs),
                    "a_med": float(np.median(ea)), "a_p90": float(np.percentile(ea, 90)), "a_signed_med": float(np.median(sa)) if sa else None,
                    "a_cov": float(np.mean([r["cov_a"] for r in rs])),
                    "b_med": float(np.median(eb)), "b_p90": float(np.percentile(eb, 90)), "b_signed_med": float(np.median(sb)) if sb else None,
                    "b_cov": float(np.mean([r["cov_b"] for r in rs])), "u_med": float(np.median([r["u"] for r in rs]))})
    return res


def score(run_dir, inputs):
    run_dir = Path(run_dir)
    rows, recs, failed = [], [], []
    for rj in sorted((run_dir / "reports").glob("*/run.json")):
        report = rj.parent.name
        if not (rj.parent / "patches").exists():
            continue
        err = json.loads(rj.read_text()).get("error")
        if err:  # a failed call (e.g. degenerate injected cameras): recorded, not scored
            failed.append({"report": report, "error": err.strip().splitlines()[-1][:300]})
            continue
        r, rec = score_report(run_dir, report, inputs)
        rows += r
        recs.append(rec)
        print(report, len(r), "rows", json.dumps({i: {k: round(v, 3) if isinstance(v, float) else v for k, v in x.items()} for i, x in rec["shots"].items()}), flush=True)
    good = lambda r: r["gt_cover"] >= MIN_GT_COVER and r["scale_status"] == "estimated" and not r["first_call"]  # noqa: E731
    out = {"reports": recs, "failed": failed, "table": table(rows, good), "table_all_cover": table(rows), "gate": {"min_gt_points": MIN_GT_POINTS, "min_gt_cover": MIN_GT_COVER}}
    (run_dir / "accuracy").mkdir(exist_ok=True)
    (run_dir / "accuracy" / "rows.json").write_text(json.dumps(rows, default=float))
    (run_dir / "accuracy" / "score.json").write_text(json.dumps(out, indent=1, default=float))
    for t in out["table"]:
        print(f"{t['seq']:8s} {t['geometry']:8s} {t['field']:24s} n={t['n']:4d}  (a) med {t['a_med']:.3f} p90 {t['a_p90']:.3f} signed {t['a_signed_med'] if t['a_signed_med'] is None else round(t['a_signed_med'], 3)} cov {t['a_cov']:.2f}"
              f"  (b) med {t['b_med']:.3f} p90 {t['b_p90']:.3f} cov {t['b_cov']:.2f}  u_med {t['u_med']:.3f}")
    return out


MAIN_FIELDS = ("top_above_floor", "base_above_floor", "height", "width", "depth", "position_xy", "planar_slope_deg")


def options_table(rows, recs):
    """Per (sequence, geometry option): camera ATE after Sim3 (key-weighted over calibrated shots), the true-height scale s,
    and per field n / median |error| (a) as delivered, (b) at the true camera height / today's coverage of (a) by u."""
    good = lambda r: r["gt_cover"] >= MIN_GT_COVER and r["scale_status"] == "estimated" and not r["first_call"] and shown(r)  # noqa: E731
    out = []
    for sq in sorted({r["seq"] for r in recs}):
        for g in sorted({r["geometry"] for r in recs if r["seq"] == sq}):
            rr = [r for r in recs if r["seq"] == sq and r["geometry"] == g and not r["first_call"]]
            sh = [s for r in rr for s in r["shots"].values() if s["scale_status"] == "estimated"]
            keys = np.array([s["keys"] for s in sh], float)
            row = {"seq": sq, "geometry": g, "windows": len(rr), "ate_sim3_m": float(np.average([s["ate_sim3_m"] for s in sh], weights=keys)) if sh else None,
                   "s_true_height": [round(s["s"], 3) for s in sh], "fields": {}}
            for f in MAIN_FIELDS:
                x = [r for r in rows if r["seq"] == sq and r["geometry"] == g and r["field"] == f and good(r)]
                if x:
                    row["fields"][f] = {"n": len(x), "a_med": float(np.median([r["err_a"] for r in x])), "b_med": float(np.median([r["err_b"] for r in x])),
                                        "a_p90": float(np.percentile([r["err_a"] for r in x], 90)), "b_p90": float(np.percentile([r["err_b"] for r in x], 90)),
                                        "a_signed_med": float(np.median([r["signed_a"] for r in x])) if x[0]["signed_a"] is not None else None,
                                        "b_signed_med": float(np.median([r["signed_b"] for r in x])) if x[0]["signed_b"] is not None else None,
                                        "cov_R0": float(np.mean([r["cov_a"] for r in x]))}
            out.append(row)
    return out


def timing_table(recs):
    """Analysis seconds (MP4 in the container -> layer written) per call: cameras, objects, cards v1 / v3, and the DA3 stage."""
    out = []
    for r in recs:
        st = r.get("stages") or {}
        out.append({"seq": r["seq"], "window": r["window"], "geometry": r["geometry"], "first_call": bool(r["first_call"]),
                    **{k: r["milestones"].get(k) for k in ("cameras", "objects", "cards_v1", "cards_v3", "judgements_v3")},
                    "da3_s": round(sum(v["s"] for k, v in st.items() if k.startswith("da3.shot")), 2), "gpu_peak_gib": r["gpu_peak_gib"],
                    "stage_peak_over_72": [k for k, v in st.items() if any((p or 0) > 72 for p in (v.get("peak_gb") or []))]})
    return out


def end_to_end(rows):
    """Coverage of the as-delivered errors by the u the cards carry (a run made with the calibrated rule), per family,
    view-set state and sequence, on shown values."""
    use = [r for r in rows if r["gt_cover"] >= MIN_GT_COVER and r["scale_status"] == "estimated" and not r["first_call"] and shown(r)]
    out = []
    for fam in ("height", "extent", "position", "angle"):
        for st in ("sets", "one_set"):
            for sq in sorted({r["seq"] for r in use}):
                x = [r for r in use if r["family"] == fam and r["seq"] == sq and one_set(r) == (st == "one_set")]
                if x:
                    out.append({"family": fam, "view_sets": st, "seq": sq, "n": len(x), "coverage": float(np.mean([r["cov_a"] for r in x])),
                                "u_med": float(np.median([r["u"] for r in x])), "err_med": float(np.median([r["err_a"] for r in x]))})
    return out


def results(out, run_dirs, droid_files, inputs, e2e_dir=None):
    """summary.json + summary.md from scored run folders (score first), the DROID timings and the u rule calibration."""
    from fast_report.core import umeyama
    rows, recs = [], []
    for d in run_dirs:
        rows += json.loads((Path(d) / "accuracy" / "rows.json").read_text())
        recs += json.loads((Path(d) / "accuracy" / "score.json").read_text())["reports"]
    droid = []
    for f in droid_files:
        for j in json.loads(Path(f).read_text())["jobs"]:
            base = j["name"].replace("-30fps", "")
            name, wi = base.split("-")[1], int(base.rsplit("-w", 1)[1])
            seq = Seq(inputs / name)
            w = seq.g["windows"][wi]
            C = np.asarray(j["c2w"], float)[:, :3, 3]
            G = np.array([seq.c2w(w["source_frames"][0] + k // w["hold"])[:3, 3] for k in j["keys"]])
            s, R, t = umeyama(C, G)
            droid.append({"job": j["name"], "frames": len(j["keys"]), "seconds": j["seconds"], "track_s": j["track_s"], "terminate_s": j["terminate_s"],
                          "peak_gib": j["peak_gib"], "gpu": j["gpu"], "ate_sim3_m": float(np.sqrt(np.mean(np.sum((s * C @ R.T + t - G) ** 2, 1))))})
    a_rows = [r for r in rows if r["geometry"] == "shot"]
    summary = {"options": options_table(rows, recs), "timing": timing_table(recs), "droid": droid,
               "u_rule": {"arkit_to_tum": calibrate(a_rows, ["arkit47", "arkit42"], ["tum"]), "tum_to_arkit": calibrate(a_rows, ["tum"], ["arkit47", "arkit42"])},
               "bounds": [b for r in recs if r["geometry"] == "shot" and not r["first_call"] for b in r.get("bounds", [])], "floors": {n: Seq(inputs / n).floor for n in SEQUENCES},
               "gate": {"min_gt_points": MIN_GT_POINTS, "min_gt_cover": MIN_GT_COVER, "shown_only": True, "warm_calls_only": True}}
    if e2e_dir:
        e2e = json.loads((Path(e2e_dir) / "accuracy" / "score.json").read_text())
        summary["end_to_end"] = {"run": str(e2e_dir), "rows": end_to_end(json.loads((Path(e2e_dir) / "accuracy" / "rows.json").read_text())),
                                 "timing": timing_table(e2e["reports"])}
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    return summary


def fmt(x, nd=3):
    return "-" if x is None else f"{x:.{nd}f}"


def summary_tables(summary):
    """The markdown tables of summary.md (the prose around them is written by hand in notes)."""
    out = ["## Geometry options (warm calls; errors on shown values with GT cover >= 0.5)", "",
           "median |error| in m: (a) as delivered (assumed 1.6 m camera height) / (b) at the true camera height; cov = share of (a) inside the ±u the cards carried in these runs (the u before this change). posed-droid / posed-gt timings exclude DROID itself (table below).", "",
           "| sequence | option | camera ATE (Sim3) | s = true h / 1.6 | top (a)/(b), cov | base (a)/(b), cov | height (a)/(b), cov | width (a)/(b), cov | position (a)/(b), cov |",
           "|---|---|---|---|---|---|---|---|---|"]
    for t in summary["options"]:
        f = t["fields"]
        cell = lambda k: f"{fmt(f[k]['a_med'])} / {fmt(f[k]['b_med'])}, {f[k]['cov_R0']:.0%} (n {f[k]['n']})" if k in f else "-"  # noqa: E731
        out.append(f"| {t['seq']} | {t['geometry']} | {fmt(t['ate_sim3_m'])} | {', '.join(f'{s:.2f}' for s in t['s_true_height'])} | {cell('top_above_floor')} | "
                   f"{cell('base_above_floor')} | {cell('height')} | {cell('width')} | {cell('position_xy')} |")
    out += ["", "## Timing (s from the MP4 bytes in the container; display layers off)", "",
            "| sequence | window | option | call | cameras | objects | cards v1 | cards v3 | judgements v3 | DA3 stage | GPU peaks GiB |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for t in sorted(summary["timing"], key=lambda t: (t["geometry"], t["seq"], t["window"], not t["first_call"])):
        out.append(f"| {t['seq']} | {t['window']} | {t['geometry']} | {'first' if t['first_call'] else 'warm'} | {fmt(t['cameras'], 1)} | {fmt(t['objects'], 1)} | "
                   f"{fmt(t['cards_v1'], 1)} | {fmt(t['cards_v3'], 1)} | {fmt(t['judgements_v3'], 1)} | {t['da3_s']} | {'/'.join(f'{p:.1f}' for p in t['gpu_peak_gib'])} |")
    out += ["", "## DROID-SLAM on the same frames (A100-80GB, DA3's intrinsics, 384x216)", "",
            "| job | frames | seconds (track + terminate) | ATE vs GT (Sim3, m) | peak GiB |", "|---|---|---|---|---|"]
    for d in summary["droid"]:
        out.append(f"| {d['job']} | {d['frames']} | {d['seconds']} ({d['track_s']} + {d['terminate_s']}) | {d['ate_sim3_m']:.3f} | {d['peak_gib']} |")
    for key, title in (("arkit_to_tum", "fitted on ARKit (47333932 x2 windows, 42445448), tested on TUM fr1 room (held out)"),
                       ("tum_to_arkit", "reverse check: fitted on TUM, tested on ARKit")):
        c = summary["u_rule"][key]
        out += ["", f"## u rule, {title}", "",
                f"k_geo = {json.dumps(c['k_geo'])}; scale part {c['r_scale']} x |value| (largest |1 - s| on the fit set {c['calib_max_scale_error']:.3f}, on the test set {c['held_max_scale_error']:.3f}).", "",
                "| family | view sets | split | n | coverage, today's u | coverage, new u | median u today / new (m) | geometry-only coverage at true height, new |", "|---|---|---|---|---|---|---|---|"]
        for fam, rows in c["families"].items():
            for k, v in rows.items():
                st, split = k.split("/")
                pct = lambda x: "-" if x is None else f"{x:.0%}"  # noqa: E731
                out.append(f"| {fam} | {st} | {split} | {v['n']} | {pct(v['R0'])} | {pct(v.get('new'))} | {fmt(v['R0_u_med'])} / {fmt(v.get('new_u_med'))} | "
                           f"{pct(v.get('new_geometry_only_b'))} |")
    if summary.get("end_to_end"):
        e = summary["end_to_end"]
        out += ["", "## End to end: cards made with the calibrated u (run " + Path(e["run"]).name + "); TUM held out, ARKit in-sample", "",
                "| family | view sets | sequence | n | coverage by the card's u | median u (m or deg) | median error |", "|---|---|---|---|---|---|---|"]
        for r in e["rows"]:
            out.append(f"| {r['family']} | {r['view_sets']} | {r['seq']} | {r['n']} | {r['coverage']:.0%} | {fmt(r['u_med'])} | {fmt(r['err_med'])} |")
        out += ["", "| sequence | window | call | cameras | objects | cards v1 | cards v3 | judgements v3 | DA3 stage | GPU peaks GiB |", "|---|---|---|---|---|---|---|---|---|---|"]
        for t in sorted(e["timing"], key=lambda t: (t["seq"], t["window"], not t["first_call"])):
            out.append(f"| {t['seq']} | {t['window']} | {'first' if t['first_call'] else 'warm'} | {fmt(t['cameras'], 1)} | {fmt(t['objects'], 1)} | {fmt(t['cards_v1'], 1)} | "
                       f"{fmt(t['cards_v3'], 1)} | {fmt(t['judgements_v3'], 1)} | {t['da3_s']} | {'/'.join(f'{p:.1f}' for p in t['gpu_peak_gib'])} |")
    b = summary["bounds"]
    if b:
        out += ["", f"Bounds ('at least' / 'at most', cut by the frame edge): {sum(x['holds_a'] for x in b)} of {len(b)} hold within u against GT."]
    return "\n".join(out) + "\n"


def decides(r):
    """A value that could decide a check: shown as a number (no bound, no review), two or more view sets, plausible size."""
    return r["status"] in (None, "") and (r["n_subsets"] or 1) >= 2 and r["plausible"]


def smallest_k(err, u, target=.9):
    """Smallest k >= 1 with share(err <= k u) >= target."""
    ratio = np.sort(np.asarray(err, float) / np.maximum(np.asarray(u, float), 1e-9))
    return max(1., float(ratio[min(len(ratio) - 1, int(np.ceil(target * len(ratio))) - 1)])) if len(ratio) else None


def coverage(rows, rule):
    """Share of rows whose error is within the rule's u. rule(r) -> (err, u)."""
    got = [rule(r) for r in rows]
    return float(np.mean([e <= u for e, u in got])) if got else None


def geo_raw(r):
    """The u's geometry part without any k: every part except scale (views, depth, pose, floor, up, resolution, fit)."""
    return float(np.sqrt(sum(float(v) ** 2 for k, v in (r["parts"] or {}).items() if k != "scale")))


def magnitude(r):
    return float(np.linalg.norm(r["value"])) if r["family"] == "position" else abs(float(r["value"]))


def one_set(r):
    return "views" not in (r["parts"] or {})


def shown(r):
    """A number on the card (not a bound, not 'needs review'), plausible size, in a shot with a floor."""
    return r["status"] in (None, "") and r["plausible"]


def calibrate(rows, calib, held, r_scale=.25, target=.9):
    """The u rule against metric ground truth: u = sqrt((k_geo[family][view sets] x geometry parts)^2 + (r_scale |v|)^2).
    k_geo is fitted on `calib` only (the smallest k >= 1 with >= target of the (b) errors, i.e. at the true camera height, inside
    k x the geometry part in true metres), apart for values with >= 2 view sets and with one; r_scale is a stated prior
    (camera height 1.2-1.8 m for 1.6 m assumed: values up to 25 % too large or 12.5 % too small), reported beside the largest
    |1 - s| on calib shots. Coverage of (a) errors (as delivered) is reported on calib and on `held` (never tuned on) for
    R0 (the u the cards carry today) and the new rule, on shown values, split by view-set state."""
    use = [r for r in rows if r["gt_cover"] >= MIN_GT_COVER and r["scale_status"] == "estimated" and not r["first_call"] and shown(r)]
    cal = [r for r in use if r["seq"] in calib]
    out = {"calib": calib, "held": held, "target": target, "r_scale": r_scale,
           "calib_max_scale_error": float(max(abs(1 - r["s"]) for r in cal)), "held_max_scale_error": float(max(abs(1 - r["s"]) for r in use if r["seq"] in held)),
           "k_geo": {}, "families": {}}
    for fam in ("height", "extent", "position"):
        out["k_geo"][fam] = {}
        for st, pick in (("sets", lambda r: not one_set(r)), ("one_set", one_set)):
            c = [r for r in cal if r["family"] == fam and pick(r)]
            out["k_geo"][fam][st] = round(smallest_k([r["err_b"] for r in c], [r["s"] * geo_raw(r) for r in c], target), 3) if c else None
    k = out["k_geo"]
    new = lambda r: float(np.hypot((k[r["family"]]["one_set" if one_set(r) else "sets"] or 1.) * geo_raw(r), r_scale * magnitude(r)))  # noqa: E731
    for fam in ("height", "extent", "position", "angle"):
        fo = {}
        for st, pick in (("sets", lambda r: not one_set(r)), ("one_set", one_set)):
            for split, rs in (("calib", [r for r in cal if r["family"] == fam and pick(r)]),
                              *((sq, [r for r in use if r["seq"] == sq and r["family"] == fam and pick(r)]) for sq in held)):
                if not rs:
                    continue
                row = {"n": len(rs), "R0": coverage(rs, lambda r: (r["err_a"], r["u"])), "R0_u_med": float(np.median([r["u"] for r in rs]))}
                if fam != "angle":
                    row.update(new=coverage(rs, lambda r: (r["err_a"], new(r))), new_u_med=float(np.median([new(r) for r in rs])),
                               new_geometry_only_b=coverage(rs, lambda r: (r["err_b"], r["s"] * (k[fam]["one_set" if one_set(r) else "sets"] or 1.) * geo_raw(r))))
                fo[f"{st}/{split}"] = row
        out["families"][fam] = fo
    return out


def sheet(run_dir, report, rows, out_jpg, n=12, seed=0):
    """Contact sheet: n cards (seeded draw among rows with GT), each on its largest segmented pick region, outlined, with
    card vs GT top / base / width (card as delivered, then at the true camera height)."""
    import cv2
    import fast_report_eval as ev
    run = json.loads((Path(run_dir) / "reports" / report / "run.json").read_text())
    L = ev.load_layers(run_dir, report)
    pick = ev.run_picks(run_dir, report, L)[-1][1]
    by = {}
    for r in rows:
        if r["report"] == report and r["family"] in ("height", "extent") and r["gt_cover"] >= MIN_GT_COVER:
            by.setdefault(r["card"], {})[r["field"]] = r
    ids = sorted(by)
    ids = [ids[i] for i in np.random.default_rng(seed).choice(len(ids), min(n, len(ids)), replace=False)]
    ent = pick.data["entities"]
    best = {}
    for i, f in enumerate(pick.frames):
        if f.get("source") != "segmented":
            continue
        m = pick.map(i)
        for cid in ids:
            if cid in ent:
                a = int((m == ent.index(cid)).sum())
                if a > best.get(cid, (0,))[0]:
                    best[cid] = (a, i)
    cap = cv2.VideoCapture(run["call"]["mp4"])
    tiles = []
    for cid in ids:
        if cid not in best:
            continue
        i = best[cid][1]
        f = pick.frames[i]
        cap.set(cv2.CAP_PROP_POS_FRAMES, f["frame"])
        img = cv2.resize(cap.read()[1], (640, 360))
        mk = (pick.map(i) == ent.index(cid)).astype(np.uint8)
        cs, _ = cv2.findContours(mk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cs, -1, (255, 255, 255), 3)
        cv2.drawContours(img, cs, -1, (0, 160, 255), 1)
        d = by[cid]
        lines = [f"{cid} {next(iter(d.values()))['name']}"]
        for fld, lab in (("top_above_floor", "top"), ("base_above_floor", "base"), ("width", "width")):
            if fld in d:
                r = d[fld]
                lines.append(f"{lab}: card {r['value']:.2f}+-{r['u']:.2f} (true-h {r['s'] * r['value']:.2f}) GT {r['gt']:.2f}")
        for j, t in enumerate(lines):
            cv2.putText(img, t, (8, 22 + 22 * j), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 0), 4)
            cv2.putText(img, t, (8, 22 + 22 * j), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1)
        tiles.append(img)
    while len(tiles) % 3:
        tiles.append(np.zeros_like(tiles[0]))
    grid = np.vstack([np.hstack(tiles[k:k + 3]) for k in range(0, len(tiles), 3)])
    cv2.imwrite(str(out_jpg), grid, [cv2.IMWRITE_JPEG_QUALITY, 80])


def droid_jobs(run_dir, inputs, out, jpeg_dir):
    """DROID jobs from option A's reports (one per window, the last call of each): that run's keyframes decoded from the
    same MP4, and DA3's median intrinsics from its cameras layer (at 504x280 -> 1280x720)."""
    import cv2
    import fast_report_eval as ev
    jpeg_dir.mkdir(parents=True, exist_ok=True)
    jobs, seen = [], {}
    for rj in sorted((Path(run_dir) / "reports").glob("*/run.json")):
        run = json.loads(rj.read_text())
        if run["call"]["options"].get("geometry", "shot") != "shot" or not (rj.parent / "patches").exists():
            continue
        seen[run["call"]["site"]] = (rj.parent.name, run["call"])
    for site, (report, call) in sorted(seen.items()):
        L = ev.load_layers(run_dir, report)
        keys = [f for s in L["cameras"]["shots"] for f in s["keys"]]
        Ks = np.array([k for s in L["cameras"]["shots"] for k in s["K"]])
        sx, sy = 1280 / 504, 720 / 280
        K720 = [float(np.median(Ks[:, 0, 0])) * sx, float(np.median(Ks[:, 1, 1])) * sy, 639.5, 359.5]
        cap, paths, want, f = cv2.VideoCapture(call["mp4"]), [], set(keys), 0
        while True:
            ok, img = cap.read()
            if not ok:
                break
            if f in want:
                p = jpeg_dir / f"{site}-{f:05d}.jpg"
                cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                paths.append(str(p))
            f += 1
        assert len(paths) == len(keys), (site, len(paths), len(keys))
        jobs.append({"name": site, "keys": keys, "K720": K720, "jpegs": paths, "from_report": report, "mp4": call["mp4"], "window_s": call["window_s"],
                     "options": call["options"]})
    Path(out).write_text(json.dumps(jobs, indent=1))
    return jobs


def posed_plan(jobs_path, droid_out, plan_out):
    """Option C calls: each window with opts.geometry = 'posed' and DROID's keyframe cameras (K at DA3's 504x280)."""
    jobs = {j["name"]: j for j in json.loads(Path(jobs_path).read_text())}
    calls = []
    for r in json.loads(Path(droid_out).read_text())["jobs"]:
        j = jobs[r["name"]]
        fx, fy, cx, cy = j["K720"]
        K = [[fx * 504 / 1280, 0, (cx + .5) * 504 / 1280 - .5], [0, fy * 280 / 720, (cy + .5) * 280 / 720 - .5], [0, 0, 1]]
        calls.append({"mp4": j["mp4"], "site": j["name"], "window_s": j["window_s"],
                      "options": {**j["options"], "geometry": "posed", "poses": {"keys": r["keys"], "c2w": r["c2w"], "K": K, "source": "DROID-SLAM keyframes"}}})
    Path(plan_out).write_text(json.dumps(calls))
    return calls


def self_check():
    rng = np.random.default_rng(0)
    floor = np.c_[rng.uniform(-3, 3, (4000, 2)), rng.normal(0, .005, 4000)]
    table = np.c_[rng.uniform(0, 1, (1500, 2)), .75 + rng.normal(0, .003, 1500)]
    tilt = np.radians(3)
    R = np.array([[1, 0, 0], [0, np.cos(tilt), -np.sin(tilt)], [0, np.sin(tilt), np.cos(tilt)]])
    P = (np.r_[floor, table] + [0, 0, -.2]) @ R.T
    f = fit_floor(P, np.array([[0, 0, 1.4], [1, 1, 1.5]]) @ R.T)
    assert abs(np.degrees(np.arccos(np.asarray(f["normal"]) @ R[:, 2])) ) < .5 and abs(f["camera_height_m"] - 1.65) < .05, f
    y0, sc = crop_params(1920, 1440)
    assert (y0, sc) == (180., 2 / 3)
    assert smallest_k([1, 2, 3, 4, 5, 6, 7, 8, 9, 30], [1] * 10) == 9. and smallest_k([.1] * 10, [1] * 10) == 1.
    R2 = np.array([[0., -1], [1, 0]])
    Rf, tf = rigid2(np.array([[0., 0], [1, 0], [0, 2]]), np.array([[0., 0], [1, 0], [0, 2]]) @ R2.T + [3, 4])
    assert np.allclose(Rf, R2) and np.allclose(tf, [3, 4])
    z = np.r_[np.full(50, 2.), [9.]]
    assert not tail_trim(z)[-1] and tail_trim(z)[:50].all()
    r = {"family": "height", "value": 1., "parts": {"views": .03, "depth": .04, "scale": .25}, "err_a": .2, "err_b": .1, "s": 1.}
    assert abs(geo_raw(r) - .05) < 1e-9 and magnitude(r) == 1. and not one_set(r)
    print("accuracy_gt self-check passed: floor fit (tilted, with a table above it), 16:9 crop, smallest k, floor-plane rigid fit, depth-tail trim, u parts")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", nargs="?", choices=("prepare", "floor", "score", "droid-jobs", "posed-plan", "results"))
    ap.add_argument("--runs", nargs="*", type=Path, default=[])
    ap.add_argument("--droid", nargs="*", type=Path, default=[])
    ap.add_argument("--e2e", type=Path)
    ap.add_argument("path", nargs="?", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--inputs", type=Path, default=RUNS / "mvp2-accuracy-inputs")
    ap.add_argument("--jpegs", type=Path)
    ap.add_argument("--jobs", type=Path)
    ap.add_argument("--self-check", action="store_true")
    a = ap.parse_args()
    if a.self_check:
        self_check()
    elif a.cmd == "prepare":
        prepare(a.out)
    elif a.cmd == "floor":
        for d in sorted(p for p in a.path.iterdir() if (p / "gt.json").exists()):
            print(d.name, json.dumps(gt_floor(d)))
    elif a.cmd == "score":
        score(a.path, a.inputs)
    elif a.cmd == "droid-jobs":
        droid_jobs(a.path, a.inputs, a.out, a.jpegs)
    elif a.cmd == "posed-plan":
        posed_plan(a.jobs, a.path, a.out)
    elif a.cmd == "results":
        sm = results(a.out, a.runs, a.droid, a.inputs, a.e2e)
        (a.out / "tables.md").write_text(summary_tables(sm))
        print((a.out / "tables.md").read_text())
