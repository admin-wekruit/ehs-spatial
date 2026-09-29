"""Fast report core (E9's resident pipeline, split into stages): decode, shot cuts, 5 fps keyframes, DA3 per shot,
floor-plane scale, TSDF + points, people (ehs_spatial.live_people.PeopleLoop), and analyse(): the one run that ties
every stage together and hands each layer to the writer the moment it exists.

Everything heavy is imported inside functions: the cut workers are spawned processes that import this module.
"""
import copy
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import queue
import struct
import threading
import time
from pathlib import Path

import numpy as np

import detect_shot_cuts as dsc  # cut rules, unchanged

BLOCK, MIN_SHOT, CHUNK, TAIL = 6, 30, 16, 8  # E9: sharpest of each 6-frame block; shots under 1 s get no geometry
DENSIFY_BATCH = 4  # click MVP section 7: keyframes per densify SAM 3 task
NAMER_WAIT_S, NAMER_HEDGE_S = 36., 15.  # mvp2/identity: Gemini answers awaited this long after the requests went out (then the Qwen
# decider names what is left); a request unanswered after NAMER_HEDGE_S (or answered with an error) is sent once more (tail latency)
DA3_HW = (280, 504)
CAMERA_HEIGHT_M = 1.6  # the reference's own assumption: every metre here is 'estimated'
LICENSE = "DA3-GIANT-1.1 (CC BY-NC 4.0): research licence, not for commercial use"
SCALE_LABEL = "scale estimated: floor plane + an assumed 1.6 m camera height; no measurement"


# ---------- CPU helpers (E9; the self-check runs them) ----------

def raster_gray(bgr):
    """16:9 source frame -> the clip's 4:3 centre crop at 640x480, grey (what detect_shot_cuts reads)."""
    import cv2
    h, w = bgr.shape[:2]
    x0 = (w - h * 4 // 3) // 2
    return cv2.cvtColor(cv2.resize(bgr[:, x0:w - x0], (640, 480), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)


def raster_rgb(bgr):
    import cv2
    h, w = bgr.shape[:2]
    x0 = (w - h * 4 // 3) // 2
    return cv2.resize(bgr[:, x0:w - x0], (640, 480), interpolation=cv2.INTER_AREA)


def gray_sharp(bgr):
    import cv2
    gray = raster_gray(bgr)
    return gray, float(cv2.Laplacian(gray, cv2.CV_32F).var())


def measure_chunk(gray, a, b, a0, b1):
    """detect_shot_cuts.measure over grey frames [a0, b1) (a0 = a - 2), keeping what starts in [a, b) (E9, unchanged)."""
    import cv2
    cv2.setNumThreads(1)
    t = time.perf_counter()
    m = dsc.measure(gray)
    pairs, spans = slice(a - a0, min(b, b1 - 1) - a0), slice(a - a0, max(a, min(b, b1 - dsc.SPAN)) - a0)
    return {"keypoints": m["keypoints"][a - a0:b - a0], "inliers": m["inliers"][pairs], "homographies": m["homographies"][pairs],
            "jump": m["jump"][pairs], "spans": m["spans"][spans], "shape": m["shape"], "s": time.perf_counter() - t}


def warm_worker(_):
    import cv2
    from fast_report import cards, judge  # noqa: F401  the cards' and the judge's work runs in these processes (MVP A, B)
    cv2.setNumThreads(1)
    time.sleep(.2)
    return __import__("os").getpid()


def stitch(parts):
    cat = lambda k: np.concatenate([p[k] for p in parts])  # noqa: E731
    return {"keypoints": cat("keypoints"), "inliers": cat("inliers"), "jump": cat("jump"), "spans": cat("spans"),
            "homographies": [h for p in parts for h in p["homographies"]], "shape": parts[0]["shape"]}


def cuts_from(m, n):
    s = dsc.signals(m)
    t = dsc.transitions(s)
    return {"frames": n, "cuts": t["cuts"], "fades": t["fades"], "noCoverage": t["noCoverage"],
            "segments": dsc.segments(n, t["cuts"], t["fades"] + t["noCoverage"])}


# ---------- layer encodings ----------

def pack_mesh(vertices, faces, colors, dev):
    """panoptes-mesh-v1 (reconstruction._save_mesh's layout): float32 xyz, normal, rgb per vertex (stride 9), uint32
    indices. Normals on the GPU. -> (bytes, blob meta)."""
    import torch
    v = torch.as_tensor(np.asarray(vertices, np.float32), device=dev)
    f = torch.as_tensor(np.asarray(faces, np.int64), device=dev)
    tri = v[f]
    fn = torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=1)
    n = torch.zeros_like(v)
    for i in range(3):
        n.index_add_(0, f[:, i], fn)
    n = torch.nn.functional.normalize(n, dim=1, eps=1e-20)
    colors = np.asarray(colors, np.float32)
    c = torch.as_tensor(colors if len(colors) == len(v) else np.ones((len(v), 3), np.float32), device=dev)
    verts = torch.cat([v, n, c], 1).cpu().numpy().astype("<f4")
    idx = f.cpu().numpy().astype("<u4").ravel()
    layout = {"stride": 9, "byteOffset": 0, "vertexCount": len(verts), "indexByteOffset": verts.nbytes, "indexCount": len(idx), "indexType": "uint32"}
    bounds = {"min": v.amin(0).tolist(), "max": v.amax(0).tolist()} if len(v) else None
    return verts.tobytes() + idx.tobytes(), {"mediaType": "application/octet-stream", "format": "panoptes-mesh-v1", "byteLayout": layout, "bounds": bounds}


def points_glb(xyz, rgb, size):
    """GLB points: float32 POSITION + normalised uint8 RGBA COLOR_0 (lingbot_dense_map.write_points_glb's layout).
    ponytail: a copy, because that module imports a Modal app chain the container does not need."""
    n = len(xyz)
    pos = np.ascontiguousarray(xyz, "<f4")
    col = np.ascontiguousarray(np.concatenate([rgb, np.full((n, 1), 255, np.uint8)], 1))
    blob = pos.tobytes() + col.tobytes()
    gltf = {"asset": {"version": "2.0", "generator": "panoptes fast_report"}, "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}],
            "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "COLOR_0": 1}, "mode": 0}]}], "buffers": [{"byteLength": len(blob)}],
            "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": 12 * n, "target": 34962},
                            {"buffer": 0, "byteOffset": 12 * n, "byteLength": 4 * n, "target": 34962}],
            "accessors": [{"bufferView": 0, "componentType": 5126, "count": n, "type": "VEC3",
                           "min": pos.min(0).astype(float).tolist(), "max": pos.max(0).astype(float).tolist()},
                          {"bufferView": 1, "componentType": 5121, "normalized": True, "count": n, "type": "VEC4"}]}
    head = json.dumps(gltf, separators=(",", ":")).encode()
    head += b" " * (-len(head) % 4)
    blob += b"\0" * (-len(blob) % 4)
    out = struct.pack("<III", 0x46546C67, 2, 28 + len(head) + len(blob)) + struct.pack("<II", len(head), 0x4E4F534A) + head
    return out + struct.pack("<II", len(blob), 0x004E4942) + blob, {"mediaType": "model/gltf-binary", "format": "glb-points", "pointSizeNative": size}


def ribbon(points, up, width=.15, lift=.01):
    """A flat strip along a floor path (packed-mesh inputs): 2 vertices per point, 2 triangles per step."""
    p = np.asarray(points, np.float64)
    up = np.asarray(up, np.float64)
    if len(p) < 2:
        return None
    d = np.gradient(p, axis=0)
    side = np.cross(up, d)
    side /= np.maximum(np.linalg.norm(side, axis=1, keepdims=True), 1e-9)
    base = p + lift * up
    v = np.concatenate([base - side * width / 2, base + side * width / 2])
    n = len(p)
    i = np.arange(n - 1)
    faces = np.concatenate([np.stack([i, i + 1, n + i], 1), np.stack([i + 1, n + i + 1, n + i], 1)])
    return v, faces, np.tile([[1., .55, .1]], (2 * n, 1))


# ---------- models ----------

class Da3:
    """DA3-GIANT-1.1 any-view resident on one GPU; shot(): one forward per shot at 504x280 (E1/E9)."""

    def __init__(self, model, dev):
        import torch
        self.model, self.dev = model, dev
        self.mean = torch.tensor([.485, .456, .406], device=dev).view(1, 3, 1, 1)
        self.std = torch.tensor([.229, .224, .225], device=dev).view(1, 3, 1, 1)

    def keep_host_copy(self):
        """Pinned host copies of every weight, made once at boot, so offload() is a pointer swap: a 5 GB copy off GPU 0
        at the end of DA3 stalled SAM 3's wave 2 there (runs fb-integrate-me340-002/003: 0.53 s a task against 0.36)."""
        import itertools
        self.host = [(t, t.detach().to("cpu").pin_memory()) for t in itertools.chain(self.model.parameters(), self.model.buffers())]

    def offload(self):
        """GPU 0's DA3 weights freed after the run's last shot (spec section 5 lever 3, for SAM 3D beside the core)."""
        for t, h in self.host:
            t.data = h

    def restore(self):
        import torch
        for t, h in self.host:
            t.data = h.to(self.dev, non_blocking=True)
        torch.cuda.synchronize(self.dev)

    def shot(self, kf):
        """(k,720,1280,3) uint8 BGR on this GPU -> colors (k,H,W,3) [0,1], depth (k,H,W), c2w (k,4,4), K (k,3,3); DA3 units."""
        import torch
        import torch.nn.functional as F
        with torch.inference_mode():
            x = F.interpolate(kf.permute(0, 3, 1, 2).flip(1).float() / 255, size=DA3_HW, mode="area")
            raw = self.model.forward(((x - self.mean) / self.std)[None], None, None, [], False, False, "saddle_balanced")
            k = len(kf)
            w2c = torch.eye(4, device=self.dev, dtype=torch.float64).repeat(k, 1, 1)
            w2c[:, :3] = raw["extrinsics"].reshape(k, -1, 4)[:, :3].double()
            return {"colors": x.permute(0, 2, 3, 1).contiguous(), "depth": raw["depth"].reshape(k, *DA3_HW).float(),
                    "c2w": torch.linalg.inv(w2c).float(), "K": raw["intrinsics"].reshape(k, 3, 3).float()}


def floor_plane(depth, K, c2w, floor, stride=4):
    """Least-squares plane through the floor-masked points, trimmed three times; camera height above it (DA3 units). E9."""
    import torch
    from fast_report.segment import backproject
    sel = floor[:, ::stride, ::stride] & (depth[:, ::stride, ::stride] > 0)
    f, vy, vx = torch.nonzero(sel, as_tuple=True)
    if len(f) < 2000:
        return None
    pts = backproject(depth, K, c2w, f, vy, vx, stride).double()
    keep = pts
    for _ in range(3):
        c = keep.mean(0)
        n = torch.linalg.eigh((keep - c).T @ (keep - c))[1][:, 0]
        r = ((keep - c) @ n).abs()
        keep = keep[r <= 2.5 * r.median().clamp(min=1e-9)]
    h = (c2w[:, :3, 3].double() - c) @ n
    if h.median() < 0:
        n, h = -n, -h
    return {"normal": n.float(), "point": c.float(), "camera_height_units": float(h.median()), "points": int(len(pts)),
            "inliers": int(len(keep)), "camera_height_spread_rel": float((h - h.median()).abs().median() / h.median().abs()),
            "residual_p90_units": float(((keep - c) @ n).abs().quantile(.9))}  # click MVP section 4.4: u_floor


def union_by_frame(frame, masks, n):
    import torch
    out = torch.zeros((n, *masks.shape[1:]), dtype=torch.int32, device=masks.device)
    if len(masks):
        out.index_add_(0, frame, masks.int())
    return out > 0


def person_masks(person, frames_local):
    """{local keyframe: [(mask numpy, score)]}, E3's fast5_dedupe: a mask at least half inside a higher-scoring kept one
    is the same person seen twice and is dropped."""
    out = {}
    is_p = (person["word"] == 0).cpu().numpy()
    fr = person["frame"].cpu().numpy()
    sc = person["score"].float().cpu().numpy()
    for j in np.flatnonzero(is_p):
        if fr[j] in frames_local:
            out.setdefault(frames_local[fr[j]], []).append(j)
    masks = {}
    for lf, idx in out.items():
        kept, union = [], None
        for j in sorted(idx, key=lambda j: -sc[j]):
            m = person["mask"][j].cpu().numpy()
            if union is not None and (m & union).sum() >= .5 * m.sum():
                continue
            union = m if union is None else union | m
            kept.append((m, float(sc[j]), int(j)))
        masks[lf] = kept
    return masks


# mvp2: X12's association gate at 5 fps (4 m/s split fast movers; 12 m/s cost nothing on walkers, VERIFY.md) over one keyframe
# step only; longer gaps keep 4 m/s (12 m/s over a 3 s gap joined Sam's Club's man on a cart to a shopper 45 m away, run 004)
PEOPLE_GATE_MPS, PEOPLE_GATE_WINDOW_S = 12., .2
PEOPLE_UP_DEG = 2.  # the up direction's u for people (the cards use the walls' plumb p90, not known yet here: ME340 read 1-2 deg)


def people_shot(si, keys, fps, g, depth_m, c2w_m, masks, plane, mpu, floor=None, pool=None):
    """PeopleLoop (the live rules, video.judge_frame) over the shot's 5 fps keyframes: tracks and rule rows. mvp2 (R4): each
    SAM 3 person mask is measured first (cards.person_geometry: feet and head along their rays at the body's range); a mask
    that cannot be a person (a picture or print: cards.PERSON_H_M rules) never reaches the loop and is returned in `rejected`.
    floor: the shot's SAM 3 'floor' masks per keyframe (numpy), the heights' local reference beside each person's feet.
    pool: the core's process pool for the measurement, one task per keyframe (in this thread it cost ME340 +3.8 s under the GIL)."""
    from ehs_spatial.live_people import PeopleLoop
    from fast_report import cards
    if plane:
        up, p0 = plane["normal"].cpu().numpy().astype(float), plane["point"].cpu().numpy().astype(float) * mpu
        scale = {"status": "model_estimated", "nativeToMeters": mpu,
                 "anchor": {"kind": "stated_carry_height", "metres": CAMERA_HEIGHT_M, "measured": False}}
    else:
        up, p0, scale = np.array([0., -1., 0.]), np.zeros(3), {"status": "uncalibrated"}
    rgb = (g["colors"] * 255).clamp(0, 255).byte().cpu().numpy()
    depth = depth_m.cpu().numpy()
    K, c2w = g["K"].cpu().numpy().astype(float), c2w_m.cpu().numpy().astype(float)
    u_floor = plane["residual_p90_units"] * mpu if plane else .02
    t0 = time.perf_counter()
    body = {i: None for kept in masks.values() for _, _, i in kept}
    if plane and body:
        fr = [j for j in masks if masks[j]]
        args = ([[(i, cards.person_crop(mk, depth[j], None if floor is None else floor[j])) for mk, _, i in masks[j]] for j in fr],
                [depth.shape[1:]] * len(fr), [K[j] for j in fr], [c2w[j] for j in fr], [up] * len(fr), [p0] * len(fr), [u_floor] * len(fr),
                [PEOPLE_UP_DEG] * len(fr))
        for part in (pool.map(cards.people_frame, *args) if pool is not None else map(cards.people_frame, *args)):
            body.update(part)
    rejected = [{"t": round(keys[j] / fps, 4), "frame": int(keys[j]), "source": f"sam3-person-{i}", "score": round(sc, 3), "reason": body[i]["reason"],
                 "geometry": body[i]} for j, kept in masks.items() for _, sc, i in kept if body[i] and body[i].get("plausible") is False]
    out_ = {r["source"] for r in rejected}

    def detector(frame):
        return [{"label": "person", "source": f"sam3-person-{j}", "score": s, "mask": m} for m, s, j in masks.get(frame["local"], [])
                if f"sam3-person-{j}" not in out_]
    t1 = time.perf_counter()
    loop = PeopleLoop(p0, up, scale, detector, None, world_epoch=si, max_speed_mps=PEOPLE_GATE_MPS, fast_window_s=PEOPLE_GATE_WINDOW_S)
    rows, findings = [], []
    for j, f in enumerate(keys):
        r, fnd = loop.step({"t": f / fps, "frame": int(f), "local": j, "streamGap": None, "rgb": rgb[j], "depth": depth[j], "K": K[j],
                            "cameraToWorld": c2w[j], "trackingState": "normal", "trackingStateReason": None, "worldOriginEpoch": si})
        for row in r:  # MVP J3a: the surface the feet rest on (footWorld is on the floor plane by construction)
            src = row.get("source") or ""
            row["footSurface"] = body.get(int(src.rsplit("-", 1)[1])) if src.startswith("sam3-person-") else None
        rows += r
        findings += fnd
    tracks = {}
    for r in rows:
        if r["track"] and r["centroidWorld"]:
            c = np.asarray(r["centroidWorld"])
            ground = c - ((c - p0) @ up) * up
            tracks.setdefault(r["track"], []).append({"t": r["t"], "frame": r["frame"], "xyz": np.round(ground, 3).tolist(),
                                                      "foot": r["footWorld"], "accepted_foot": r["accepted"], "score": r.get("score"),
                                                      "foot_surface": r.get("footSurface")})
    return tracks, rows, findings, up, rejected, {"measure_s": round(t1 - t0, 3), "loop_s": round(time.perf_counter() - t1, 3), "masks": len(body)}


def object_points(pts, frame_of_lifted, conf, cap=20000):
    """Per confirmed lift component (in `conf` order): its cleaned points (<= cap, a seeded random sample, with the
    share kept) and per view [mask pixels, touches top, bottom, left, right], on the CPU for the cards (section 4.2)."""
    import torch
    lab = pts["label"]
    if not len(conf):
        return []
    want = torch.full((int(lab.max()) + 1,), -1, dtype=torch.long, device=lab.device)
    want[torch.as_tensor(conf, device=lab.device)] = torch.arange(len(conf), device=lab.device)
    oi_pt = want[lab[pts["mid"]]]
    k = oi_pt >= 0
    oi_pt, world, fr_pt, z = (t.cpu().numpy() for t in (oi_pt[k], pts["world"][k], frame_of_lifted[pts["mid"][k]], pts["z"][k]))
    oi_m, fm = want[lab].cpu().numpy(), frame_of_lifted.cpu().numpy()
    px, bd = pts["pixels"].cpu().numpy(), pts["border"].cpu().numpy()
    order = np.argsort(oi_pt, kind="stable")
    rng = np.random.default_rng(0)
    out = []
    for ix in np.split(order, np.cumsum(np.bincount(oi_pt, minlength=len(conf)))[:-1]):
        ratio = min(1., cap / max(len(ix), 1))
        if ratio < 1:
            ix = np.sort(rng.choice(ix, cap, replace=False))
        out.append({"world": world[ix], "frame": fr_pt[ix].astype(np.int32), "z": z[ix], "sample_ratio": ratio, "views": {}})
    for j in np.flatnonzero(oi_m >= 0):
        v = out[oi_m[j]]["views"].setdefault(int(fm[j]), [0, False, False, False, False])
        v[0] += int(px[j])
        v[1:] = [a or bool(b) for a, b in zip(v[1:], bd[j])]
    for name, (m, w) in (pts.get("edge") or {}).items():  # mvp2: the un-eroded top / bottom edges (segment.edges), all kept
        oi = want[lab[m]]
        k = oi >= 0
        oi, w, f = oi[k].cpu().numpy(), w[k].cpu().numpy(), frame_of_lifted[m[k]].cpu().numpy().astype(np.int32)
        order = np.argsort(oi, kind="stable")
        for i, ix in enumerate(np.split(order, np.cumsum(np.bincount(oi, minlength=len(conf)))[:-1])):
            out[i][f"{name}_world"], out[i][f"{name}_frame"] = w[ix], f[ix]
    return out


def object_voxels(obj_voxel, conf, first):
    """The first lift's (component, voxel code) pairs -> (sorted unique codes, object index of each) for the densify join;
    objects are numbered from `first` in `conf` order; a voxel two objects share goes to one of them."""
    import torch
    oc, codes = obj_voxel
    want = torch.full((int(oc.max()) + 1 if len(oc) else 1,), -1, dtype=torch.long, device=codes.device)
    want[torch.as_tensor(conf, device=codes.device)] = torch.arange(len(conf), device=codes.device) + first
    o = want[oc]
    keep = o >= 0
    codes, o = codes[keep], o[keep]
    u, inv = torch.unique(codes, return_inverse=True)
    owner = torch.full((len(u),), -1, dtype=torch.long, device=codes.device).scatter_reduce(0, inv, o, "amax")
    return u, owner


def merge_points(parts, cap=20000):
    """Point dicts of one object (object_points' layout) -> one, subsampled to `cap` with a seeded draw."""
    world = np.concatenate([p["world"] for p in parts])
    frame = np.concatenate([p["frame"] for p in parts])
    z = np.concatenate([p["z"] for p in parts])
    ratio = float(np.mean([p.get("sample_ratio", 1.) for p in parts]))
    if len(world) > cap:
        ix = np.sort(np.random.default_rng(1).choice(len(world), cap, replace=False))
        world, frame, z, ratio = world[ix], frame[ix], z[ix], ratio * cap / len(world)
    views = {}
    for p in parts:
        for v, m in (p.get("views") or {}).items():
            a = views.setdefault(v, [0, False, False, False, False])
            a[0] += m[0]
            a[1:] = [x or bool(y) for x, y in zip(a[1:], m[1:])]
    edge = {f"{e}_{k}": np.concatenate([p[f"{e}_{k}"] for p in parts if f"{e}_{k}" in p] or [np.zeros((0, 3) if k == "world" else 0)])
            for e in ("top", "bottom") for k in ("world", "frame")}
    return {"world": world, "frame": frame, "z": z, "sample_ratio": ratio, "views": views, **edge}


def cards_calibration():
    """fast_report/calibration.json (D writes it): k per family and k_pose; defaults 1 without it (spec section 4.4)."""
    path = Path(__file__).with_name("calibration.json")
    out = {"file_sha256": None, "k": {"height": 1., "extent": 1., "position": 1., "angle": 1.}, "k_pose": 1.}
    if path.exists():
        raw = path.read_bytes()
        d = json.loads(raw)
        ks = {f: float(v["k"] if isinstance(v, dict) else v) for f, v in (d.get("k") or {}).items()}  # D writes {family: {k, n, coverage}}
        out.update(file_sha256=sha256(raw), k={**out["k"], **ks}, k_pose=float(d.get("k_pose", 1.)))
    return out


def box_stats(objects, cards_v1, limit=3.):
    """L1 inflation: longest box side over `limit` m, p90 and max, for the lift's voxel boxes (fb/integrate's), the v1 robust
    boxes and the cards' boxes shown as plausible."""
    def row(sides):
        a = np.asarray([x for x in sides if x is not None], float)
        return {"n": int(len(a)), "over_3m_share": round(float((a > limit).mean()), 4) if len(a) else None,
                "p90_m": round(float(np.percentile(a, 90)), 3) if len(a) else None, "max_m": round(float(a.max()), 3) if len(a) else None}
    out = {"voxel_box": row([max(np.subtract(o["voxel_box_m"][1], o["voxel_box_m"][0])) for o in objects if "voxel_box_m" in o]),
           "v1_robust_box": row([max(o["box"]["size_m"]) for o in objects if "box" in o])}
    if cards_v1:
        shown = [c["physical"] for c in cards_v1["cards"] if c["kind"] == "object" and "box" in c["physical"]]
        out["cards_all"] = row([max(p["box"]["size_m"]) for p in shown])
        out["cards_shown_plausible"] = row([max(p["box"]["size_m"]) for p in shown if p["size_check"].get("status") != "implausible"])
        out["implausible_by_class"] = {}
        for p in shown:
            if p["size_check"].get("status") == "implausible":
                c = p["size_check"]["class"]
                out["implausible_by_class"][c] = out["implausible_by_class"].get(c, 0) + 1
    return out


def shot_floor(gg):
    """(normal, point in metres) of the shot's floor; without a floor plane the first camera's down vector 1.6 m below it."""
    if gg["plane"]:
        return gg["plane"]["normal"].cpu().numpy().astype(float), gg["plane"]["point"].cpu().numpy().astype(float) * gg["mpu"]
    c = gg["c2w_m"][0].cpu().numpy().astype(float)
    return -c[:3, 1], c[:3, 3] + CAMERA_HEIGHT_M * c[:3, 1]


# ---------- the run ----------

def sha256(b):
    return hashlib.sha256(b).hexdigest()


GENERATED = ["a generated display layer: never used for measurement"]


def share_frames(frames, clock):
    """The decoded frames (cv2 BGR) as one .npy in RAM (/dev/shm) that the SAM 3D gate and splat processes map. Plain file
    writes: the copy happens in the kernel without the GIL (filling a memmap from this thread took 15.8 s beside SAM 3)."""
    import os
    from fast_report import sam3d
    path = sam3d.SHARED / f"fb-frames-{os.getpid()}-{time.time_ns()}.npy"
    with clock.stage("frames.shared", n={"frames": len(frames)}), open(path, "wb") as fh:
        np.lib.format.write_array_header_1_0(fh, {"descr": "|u1", "fortran_order": False, "shape": (len(frames), *frames[0].shape)})
        for f in frames:
            fh.write(memoryview(np.ascontiguousarray(f)))
    return str(path)


def splat_job(m, gg, frames_ab, shared, opts, clock, writer, release):
    """GPU 1's splat process on the longest shot: set up now, train once `release` is set (the SAM 3 queue is empty), a
    preview after opts['splat_preview_s'] (B: 150 s keeps ME340 above 27 dB), then optional background snapshots. Held-out
    frames (opts['eval_holdout'], the delivered splat's) stay out of training and are scored after it; the preview's score
    comes as a second version of the layer (same file)."""
    from fast_report import splat
    a, b = frames_ab
    held = sorted(f for f in opts.get("eval_holdout") or [] if a <= f <= b)
    shot = {"keys": gg["keys"], "frames": [a, b], "c2w_m": gg["c2w_m"], "K": gg["K"], "person": gg["person"]}
    out, released, preview = {"held_out_frames": len(held), "versions": []}, None, None
    for r in m.splat.start(shared.result(), shot, gg["seeds"], opts.get("splat_preview_s") or splat.PREVIEW_S, background_s=opts.get("background_s") or 0,
                           held=held, hold=True):
        now = time.time()
        if r["kind"] == "ready":
            clock.external("splat.setup", 1, r["end_unix"] - r["setup_s"], r["end_unix"])
            release.wait()
            released = time.time()
            m.splat.release()
            clock.mark("splat_released")
        elif r["kind"] == "score":
            clock.external("splat.score", 1, now - r["score_s"], now, n={"of": r["of"]})
            out["versions"].append({"score_of": r["of"], "seconds": r["seconds"], "held_out": r["held_out"]})
            got = (r["held_out"] or {}).get("corrected_pose")
            if r["of"] == "preview" and preview and got:
                writer.put("splat", {**preview[0], "holdout": {k: got.get(k) for k in ("psnr", "ssim", "lpips", "frames")}}, preview[1], "generated", GENERATED)
        else:
            blob = r.pop("splat32")
            clock.external(f"splat.{r['kind']}", 1, released, r["end_unix"], n={"steps": r["steps"], "count": r["count"]})
            data = {"format": "splat32", "shot": gg["index"], "count": r["count"], "kind": r["kind"], "train_s": r["seconds"], "steps": r["steps"],
                    "trained_frames": r.get("trained_frames"), "held_out_frames": len(held)}
            blobs = {"splat": (blob, {"mediaType": "application/octet-stream", "format": "splat32"})}
            writer.put("splat", data, blobs, "generated", GENERATED)
            clock.mark(f"splat_{r['kind']}_put")
            if r["kind"] == "preview":
                preview = (data, blobs)
            out["versions"].append(r)
    out["worker"] = getattr(m.splat, "last", None)
    return out


def models_job(m, inputs, geo, shared, words, clock, writer, dev):
    """SAM 3D's first pass (fast_report.sam3d.gate: the first 30 ranked objects, one try each, GPU 0 under MPS): each
    accepted model at once as a new 'models' version (cumulative), and a last one ('final') when the pass is judged."""
    import os
    import shutil
    import torch
    from fast_report import sam3d
    with torch.cuda.device(dev):
        torch.cuda.empty_cache()  # the core's cached blocks back to the device before SAM 3D's two processes generate beside it
    objs = inputs()
    records, models, blobs = [], [], {}
    shots = [{k: gg[k] for k in ("index", "keys", "depth_m", "c2w_m", "K", "person")} for gg in geo]
    judged = lambda: sum(r.get("stage") == "assess" for r in records)  # noqa: E731
    try:
        for x in sam3d.gate(objs, shots, shared.result(), clock, m.sam3d, m.gate_pool, vocab=words, records=records):
            models.append({"object": x["object"], "transform": {"position": [float(v) for v in x["transform"][:3, 3]], "quaternion": [0, 0, 0, 1],
                                                                "scale": [1, 1, 1]}, "bounds": x["bounds"], "gate": x["gate"]})
            blobs[f"model-{x['object']}"] = (x["glb"], {"mediaType": "model/gltf-binary", "format": "glb"})
            writer.put("models", {"models": list(models), "attempted": judged(), "final": False}, dict(blobs), "generated", GENERATED)
            clock.mark("first_model_put")
    finally:
        summary = {"ranked": len(sam3d.rank(objs, words)), "attempted": judged(), "accepted": len(models),
                   "prepare_rejected": sum("rejected" in r for r in records), "errors": [r for r in records if "error" in r][:5]}
        writer.put("models", {"models": models, "attempted": judged(), "final": True, "first_pass": summary}, dict(blobs), "generated", GENERATED)
        clock.mark("models_final_put")
        for d in sam3d.SHARED.glob(f"fb-gate-{os.getpid()}-*"):  # this run's staged gate inputs
            shutil.rmtree(d, ignore_errors=True)
    return {**summary, "records": records}


def analyse(m, mp4, opts, clock, writer, log):
    """m: the resident models (FastReport): sams, da3, emb, devs, pools. Every layer goes to `writer` as soon as it
    exists. Returns the run summary; layers are the product."""
    import cv2
    import torch
    import torch.nn.functional as F
    import video_events
    from fast_report import cards, cascade, layers, segment, vlm
    dev_geo, dev_seg = m.dev_geo, m.dev_seg
    video_sha = sha256(mp4)
    site = opts.get("site") or "unknown"
    use_cache = opts.get("cache", True)  # the cross-video label cache (and remembering this video's words)
    # the site's earlier words in wave 1 (spec section 11): opt-in, because the words of two visits add up (runs 004/005:
    # 57 + 39 = 96 words, objects 7 s later than one visit's 56)
    site_vocab = use_cache and opts.get("site_vocab", False)
    site_path = f"/v/layers/sites/{site}/vocab.json"
    cached_words = vlm.site_words(site_path, video_sha) if site_vocab else []
    wave1 = list(dict.fromkeys(vlm.CORE + cached_words))
    summary = {"video_sha256": video_sha, "site": site, "cache": use_cache, "site_vocab": site_vocab, "wave1_words": wave1,
               "site_cached_words": len(cached_words)}
    Path("/tmp/in.mp4").write_bytes(mp4)
    cap = cv2.VideoCapture("/tmp/in.mp4")
    fps, n_total = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer.put("video", {"fps": fps, "frames": n_total, "width": W, "height": H, "sha256": video_sha, "window_s": opts.get("window_s")},
               {"video": (mp4, {"mediaType": "video/mp4"})}, "observed", ["the uploaded video"])
    work = segment.SamWork(m.sams, dev_geo, clock, wave1)
    for words in (("person", "floor"), tuple(wave1)):
        for s in m.sams.values():
            s.text(words)
    seg_future = m.cpu_pool.submit(work.worker, dev_seg, "seg")
    cuts_ready = threading.Event()
    early = m.cpu_pool.submit(work.worker, dev_geo, "geo before cuts", cuts_ready) if dev_geo != dev_seg else None
    vlm_frames = sorted({int((i + .5) * n_total / vlm.VOCAB_FRAMES) for i in range(vlm.VOCAB_FRAMES)})
    frames, grays, futures, keys, chunk_at, chunk_done = [], [], [], [], [], {}
    decoded_all = threading.Event()
    results = {}

    def vocab_then_events():
        """GPU 1's vLLM: the vocabulary first (it gates wave 2), then events (they gate nothing)."""
        try:
            with clock.stage("vlm.vocab.frames", n={"frames": len(vlm_frames)}):  # seek: no waiting for the decoder to get there
                reader = cv2.VideoCapture("/tmp/in.mp4")
                for f in vlm_frames:
                    reader.set(cv2.CAP_PROP_POS_FRAMES, f)
                    seeked.append(reader.read()[1])
                reader.release()
            with clock.stage("vlm.vocab", n={"frames": len(vlm_frames)}):
                pngs = [cv2.imencode(".png", raster_rgb(img))[1].tobytes() for img in seeked]
                words, rec = vlm.vocab(pngs)
            results["vocab"] = {**rec, "words": words, "frames": vlm_frames}
        except Exception as error:  # noqa: BLE001  no vocabulary: wave 1 alone, recorded
            words, results["vocab"] = [], {"error": repr(error)[:500]}
        decoded_all.wait()
        results["wave2_words"] = work.set_wave2(words, len(keys))
        clock.mark("vocab_known")
        if use_cache and words:
            vlm.remember_site(site_path, words, video_sha)
        with clock.stage("vlm.events.frames"):
            windows = []
            for t0, t1 in video_events.bounds(len(frames) / fps, 12.):
                picked = [(float(t), cv2.imencode(".jpg", raster_rgb(frames[int(round(t * fps))]), [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes())
                          for t in np.arange(t0, t1, .5) if int(round(t * fps)) < len(frames)]
                windows.append((float(t0), float(t1), picked))
        with clock.stage("vlm.events", n={"windows": len(windows)}):
            ev = vlm.events(windows)
        writer.put("events", {"model": vlm.QWEN + " (vLLM)", "windows": [{"t0": e["t0"], "t1": e["t1"], **(e["parsed"] if isinstance(e["parsed"], dict) else {"caption": None, "events": []}),
                                           "raw": e["text"]} for e in ev]},
                   None, "inferred", ["events are Qwen3-VL-8B descriptions of the frames: model output, not findings"])
        clock.mark("events_put")
        return ev
    seeked = []
    vocab_future = m.vlm_pool.submit(vocab_then_events)

    # decode on this thread; grey + sharpness on threads; cut chunks measured in processes; keyframe = sharpest of each
    # BLOCK-frame block, picked one block behind decoding; every PERSON_FRAMES keyframes -> both GPUs + one SAM 3 task
    with clock.stage("decode", n={"frames": n_total}):
        a, b0 = 0, 0

        def pick(b):
            keys.append(max(range(b, min(b + BLOCK, len(frames))), key=lambda f: grays[f].result()[1]))
            if len(keys) % segment.PERSON_FRAMES == 0:
                work.add_chunk([frames[f] for f in keys[-segment.PERSON_FRAMES:]])
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            frames.append(bgr)
            grays.append(m.cpu_pool.submit(gray_sharp, bgr))
            if len(frames) == a + CHUNK + dsc.SPAN + 1:
                a0 = max(0, a - 2)
                futures.append(m.proc_pool.submit(measure_chunk, [g.result()[0] for g in grays[a0:]], a, a + CHUNK, a0, len(frames)))
                chunk_at.append(clock.now())
                futures[-1].add_done_callback(lambda f, i=len(futures) - 1: chunk_done.__setitem__(i, clock.now()))
                a += CHUNK
            if len(frames) == b0 + 2 * BLOCK:
                pick(b0)
                b0 += BLOCK
        cap.release()
        n = len(frames)
        while b0 < n:
            pick(b0)
            b0 += BLOCK
        if len(keys) % segment.PERSON_FRAMES:
            work.add_chunk([frames[f] for f in keys[len(keys) - len(keys) % segment.PERSON_FRAMES:]])
        work.seal_decode()
        decoded_all.set()
        shared = m.cpu_pool.submit(share_frames, frames, clock)
        gray_all = [g.result()[0] for g in grays]
        sharp_all = [g.result()[1] for g in grays]  # the cards' best-view ranking (section 4.3)
        for a1 in range(a, n, TAIL):  # the tail in small pieces: after decoding it is on the critical path
            b1, lo = min(a1 + TAIL, n), max(0, a1 - 2)
            futures.append(m.proc_pool.submit(measure_chunk, gray_all[lo:min(n, b1 + dsc.SPAN + 1)], a1, b1, lo, min(n, b1 + dsc.SPAN + 1)))
            chunk_at.append(clock.now())
            futures[-1].add_done_callback(lambda f, i=len(futures) - 1: chunk_done.__setitem__(i, clock.now()))
    with clock.stage("cuts"):
        parts = [f.result() for f in futures]
        cuts = cuts_from(stitch(parts), n)
        shots = [(a, b) for a, b in cuts["segments"] if b - a + 1 >= MIN_SHOT]
        shot_pos = [[i for i, f in enumerate(keys) if a <= f <= b] for a, b in shots]
        kf = torch.cat(work.chunks[dev_geo])
    clock.mark("cuts_ready")
    cuts_ready.set()
    if early is not None:
        early.result()

    shots_gpu = []
    for si, pos in enumerate(shot_pos):
        with clock.stage(f"da3.shot{si}", gpu=dev_geo, n={"views": len(pos)}):
            shots_gpu.append(m.da3.shot(kf[pos]))
    clock.mark("da3_done")
    m.da3.offload()  # spec section 5 lever 3: ~5 GB of GPU 0 for SAM 3D (a pointer swap); run() puts it back
    if dev_geo != dev_seg:
        work.worker(dev_geo, "geo", until=work.person_ready)
    work.wait(work.person_ready)
    with torch.inference_mode(), clock.stage("gather.person_floor", gpu=dev_geo):
        person = work.gathered("person")
        work.person.clear()  # gathered: the per-task pieces are copies
        is_person = person["word"] == 0
        people_u = union_by_frame(person["frame"][is_person], person["mask"][is_person], len(keys))
        dyn = F.max_pool2d(people_u[:, None].float(), 5, 1, 2)[:, 0] > 0
        floor = union_by_frame(person["frame"][~is_person], person["mask"][~is_person], len(keys))

    cam_rows, room_rows, room_blobs, light_rows, light_blobs, people_blobs, geo = [], [], {}, [], {}, {}, []
    lab = [SCALE_LABEL, LICENSE]
    for si, (pos, g) in enumerate(zip(shot_pos, shots_gpu)):  # scale first: the cameras layer needs nothing else
        p = torch.tensor(pos, device=dev_geo)
        with torch.inference_mode(), clock.stage(f"scale.shot{si}", gpu=dev_geo):
            plane = floor_plane(g["depth"], g["K"], g["c2w"], floor[p])
            mpu = CAMERA_HEIGHT_M / plane["camera_height_units"] if plane else 1.
            depth_m, c2w_m = g["depth"] * mpu, g["c2w"].clone()
            c2w_m[:, :3, 3] *= mpu
        floor_rec = {k: v for k, v in (plane or {}).items() if k not in ("normal", "point")}
        if plane:
            floor_rec.update(normal=plane["normal"].tolist(), point_m=(plane["point"] * mpu).tolist(), u_floor_m=round(plane["residual_p90_units"] * mpu, 4))
        cam_rows.append({"index": si, "frame_id": f"shot-{si}", "frames": list(shots[si]), "keys": [keys[q] for q in pos],
                         "times": [round(keys[q] / fps, 4) for q in pos], "c2w": c2w_m.cpu().numpy().round(5).tolist(),
                         "K": g["K"].cpu().numpy().round(3).tolist(), "wh": [DA3_HW[1], DA3_HW[0]], "source_wh": [W, H], "mpu": mpu,
                         "scale_status": "estimated" if plane else "uncalibrated", "floor": floor_rec})
        geo.append({"depth_m": depth_m, "c2w_m": c2w_m, "K": g["K"], "mpu": mpu, "pos": pos, "plane": plane, "index": si,
                    "keys": [keys[q] for q in pos], "person": people_u[p]})
    cameras = {"shots": cam_rows, "fps": fps, "keyframe_rule": "sharpest of each 6-frame block (5 fps)", "license": LICENSE,
               "scale": {"status": "estimated", "source": "floor plane (SAM 3 'floor') + assumed camera height 1.6 m; unit scale where no floor"}}
    for si, (gg, g) in enumerate(zip(geo, shots_gpu)):
        pos, plane, mpu, depth_m, c2w_m = gg["pos"], gg["plane"], gg["mpu"], gg["depth_m"], gg["c2w_m"]
        p = torch.tensor(pos, device=dev_geo)
        with torch.inference_mode(), clock.stage(f"tsdf.shot{si}", gpu=dev_geo, n={"views": len(pos)}):
            import m3_exp_geometry as geo_e7
            d = depth_m.clone()
            d[dyn[p]] = 0
            geo_e7.edge_filter(d)
            tsdf = geo_e7.fuse(d, g["K"].cpu().numpy(), c2w_m.cpu().numpy().astype(np.float64), g["colors"])
        with clock.stage(f"pack.room.shot{si}", gpu=dev_geo):
            mesh = tsdf["mesh"]
            faces = np.asarray(mesh.triangles)
            raw, meta = pack_mesh(np.asarray(mesh.vertices), faces, np.asarray(mesh.vertex_colors), dev_geo)
            room_blobs[f"mesh-{si}"] = (raw, {**meta, "frame": f"shot-{si}"})
            rows9 = np.frombuffer(raw, "<f4", count=9 * meta["byteLayout"]["vertexCount"]).reshape(-1, 9)
            quick = layers.light_mesh(rows9[:, :3], rows9[:, 3:6], rows9[:, 6:9], faces)
            light_blobs[f"mesh-{si}"] = layers.pack_mesh(*quick)
            raw, meta = points_glb(tsdf["points"], tsdf["colors"], .03)
            room_blobs[f"points-{si}"] = (raw, {**meta, "frame": f"shot-{si}"})
        gg["seeds"] = {"xyz": tsdf["points"], "rgb": tsdf["colors"]}
        gg["mesh_vf"] = (rows9[:, :3], faces)  # the cards' plumb check (section 4.5), off the critical path
        room_rows.append({"index": si, "frame_id": f"shot-{si}", "triangles": tsdf["n_triangles"], "points": tsdf["n_points"], "tsdf_voxel_m": .03})
        light_rows.append({"index": si, "frame_id": f"shot-{si}", "triangles": len(quick[3]), "points": 0, "light_cell_m": .06})
    # cameras and the light room in one commit (put 1 s apart, the room waited for the cameras' commit: run
    # fb-integrate-me340-005's warm call, room written 8.3 s after put); people next; the full room (77 MB on ME340, a
    # 4 s commit) only once people are written
    writer.put("cameras", cameras, None, "estimated", lab)
    writer.put("room", {"shots": light_rows, "kind": "light"}, light_blobs, "estimated", lab)
    clock.mark("cameras_put")
    clock.mark("room_put")
    # the splat (GPU 1): set up now, while SAM 3 still runs there; trains once the SAM 3 queue is empty (released below)
    longest = max(range(len(geo)), key=lambda i: len(geo[i]["pos"])) if geo else None
    release = m.release = threading.Event()  # run() sets it too if this run fails before the SAM 3 queue empties
    splat_future = m.cpu_pool.submit(splat_job, m, geo[longest], shots[longest], shared, opts, clock, writer, release) \
        if longest is not None else None

    def people_layer():  # CPU work: runs beside GPU 0's share of the SAM 3 queue
        try:
            people_tracks()
            for _ in range(750):  # people's own commit first (ponytail: polls the writer's committed rows, 15 s at most)
                if any(r["layer"] == "people" for r in writer.rows):
                    break
                time.sleep(.02)
        finally:
            writer.put("room", {"shots": room_rows, "kind": "full"}, room_blobs, "estimated", lab)
            clock.mark("room_full_put")

    def people_tracks():
        tracks_out, rules_out, per_shot, rejected_out = [], [], [], []
        for si, (gg, g) in enumerate(zip(geo, shots_gpu)):
            pos = gg["pos"]
            with clock.stage(f"people.shot{si}", n={"keyframes": len(pos)}):
                local = {q: j for j, q in enumerate(pos)}
                pm = person_masks(person, local)
                tracks, rows, findings, up, rejected, timing = people_shot(si, [keys[q] for q in pos], fps, g, gg["depth_m"], gg["c2w_m"], pm, gg["plane"],
                                                                    gg["mpu"], floor[torch.tensor(pos, device=floor.device)].cpu().numpy(), m.proc_pool)
                for j, kept in pm.items():  # the pick layer's people: each kept mask with its track (section 3.2)
                    person_kept[pos[j]] = [(gi, mk) for mk, _, gi in kept]
                for r in rows:
                    if (r.get("source") or "").startswith("sam3-person-"):
                        person_entity[int(r["source"].rsplit("-", 1)[1])] = f"person:{si}-{r['track']}" if r["track"] else "person:untracked"
                for r in rejected:  # mvp2: a picture of a person clicks through to the card that says why it is none
                    person_entity[int(r["source"].rsplit("-", 1)[1])] = "person:not-a-person"
                rejected_out.extend({**r, "shot": si} for r in rejected)
                for tid, pts in tracks.items():
                    rb = ribbon([q["xyz"] for q in pts], up)
                    if rb is not None:
                        raw, meta = pack_mesh(*rb, dev_geo)
                        people_blobs[f"track-{si}-{tid}"] = (raw, {**meta, "frame": f"shot-{si}"})
            tracks_out += [{"id": f"{si}-{t}", "shot": si, "t0": pts[0]["t"], "t1": pts[-1]["t"], "detections": len(pts), "points": pts}
                           for t, pts in tracks.items()]
            rules_out += [{**f, "shot": si} for f in findings]
            per_shot.append({"index": si, "frame_id": f"shot-{si}", "detections": len(rows), "tracks": len(tracks), "timing": timing})
        results["people"] = {"tracks": tracks_out, "rules": rules_out, "shots": per_shot, "rejected": rejected_out,
                             "association_gate_mps": PEOPLE_GATE_MPS, "note": "the fast path tracks people only: no non-person movers"}
        writer.put("people", results["people"], people_blobs, "observed+estimated", [*lab, "rules that need metres say NEEDS_REVIEW: the scale is not measured"])
        clock.mark("geometry_layers_put")

    person_kept, person_entity = {}, {}  # keyframe -> [(person mask index, DA3-grid mask)]; mask index -> entity id
    people_future = m.cpu_pool.submit(people_layer)

    if dev_geo != dev_seg:
        work.worker(dev_geo, "geo")
    work.wait(work.all_ready)
    seg_future.result()
    densify_on = opts.get("densify", True)
    if not densify_on:
        release.set()  # fb/integrate's order: the splat trains once the SAM 3 queue is empty
    people_future.result()
    clock.mark("sam3_done")
    words = work.words
    work.cache.clear()
    if not densify_on:
        for d in work.chunks:  # the keyframes on each GPU (kf is GPU 0's own copy); densify keeps them until it is done
            work.chunks[d] = []

    # ---------- objects: flood handling, lift, naming, cascade ----------
    with torch.inference_mode(), clock.stage("dedupe", gpu=dev_geo):
        voc = work.gathered("vocab")
        work.vocab.clear()  # gathered: the per-task pieces were a second copy on GPU 0 (6 GB on ME340) until the run's end
        voc_count = int(len(voc["frame"])) if voc is not None else 0
        kept, votes = segment.dedupe(voc["frame"], voc["word"], voc["score"], voc["mask"]) if voc is not None else (np.zeros(0, int), [])
        vf = voc["frame"].cpu().numpy() if voc is not None else np.zeros(0, int)
    objects, members, obj_masks_on, obj_points, shot_voxels = [], [], [], [], {}
    with torch.inference_mode(), clock.stage("lift", gpu=dev_geo, n={"masks_kept": int(len(kept)), "masks_in": int(len(voc["frame"])) if voc else 0}):
        vote_of = {}
        for a_, w_, s_ in votes:
            vote_of.setdefault(int(a_), []).append((int(w_), float(s_)))
        kept_t = torch.from_numpy(kept).to(dev_geo)
        lift_stats = []
        for si, gg in enumerate(geo):
            local = torch.full((len(keys),), -1, dtype=torch.long, device=dev_geo)
            local[torch.tensor(gg["pos"], device=dev_geo)] = torch.arange(len(gg["pos"]), device=dev_geo)
            sel = kept_t[local[voc["frame"][kept_t]] >= 0] if len(kept_t) else kept_t
            if len(sel) < 2:
                continue
            comp, arr, stats, pts = segment.lift(voc["mask"][sel], local[voc["frame"][sel]], gg["depth_m"], gg["K"], gg["c2w_m"],
                                                 dyn[torch.tensor(gg["pos"], device=dev_geo)])
            lift_stats.append(stats)
            if arr is None:
                continue
            conf = np.flatnonzero(arr["frames"] >= segment.CONFIRMED)
            obj_points += object_points(pts, pts["fr"], conf)
            shot_voxels[si] = object_voxels(pts["obj_voxel"], conf, len(objects))
            sel_np, fr_np, sc_np = sel.cpu().numpy(), voc["frame"][sel].cpu().numpy(), voc["score"][sel].float().cpu().numpy()
            area = voc["mask"][sel].sum((1, 2)).cpu().numpy()
            for c in np.flatnonzero(arr["frames"] >= segment.CONFIRMED):
                mem = np.flatnonzero(comp == c)
                vv = {}
                for gi in sel_np[mem]:
                    for w_, s_ in vote_of.get(int(gi), []):
                        vv[words[w_]] = vv.get(words[w_], 0.) + s_
                best = mem[np.argmax(sc_np[mem] * np.sqrt(area[mem]))]
                oid = f"obj-{si}-{len(objects)}"
                objects.append({"id": oid, "shot": si, "word": segment.name(vv, words),
                                "votes": {w: round(v, 3) for w, v in sorted(vv.items(), key=lambda x: -x[1])[:5]},
                                "frames": int(arr["frames"][c]), "masks": int(len(mem)), "voxels": int(arr["voxels"][c]),
                                "centroid_m": arr["centroid"][c].round(3).tolist(), "best_key": int(keys[fr_np[best]]),
                                "voxel_box_m": [(arr["lo"][c] - segment.LIFT_VOXEL / 2).round(3).tolist(), (arr["hi"][c] + segment.LIFT_VOXEL / 2).round(3).tolist()]})
                members.append(sel_np[mem])
                obj_masks_on.append((si, int(sel_np[best])))
        summary["lift"] = lift_stats
    with clock.stage("objects.boxes", n={"objects": len(objects)}):  # section 4.2 step 5 on the cleaned points (L1)
        floors = [cards.floor_frame(gg["c2w_m"][0].cpu().numpy(), *shot_floor(gg)) for gg in geo]
        ws = [np.asarray(x["world"])[:: max(1, len(x["world"]) // 4000)] for x in obj_points]  # v1_box's own stride, here: less to pickle
        # integration: in the process pool (threads held the GIL: 1.7 / 2.1 s on Sam's Club / Walmart, SAM 3 -> objects +1.2 s over fb)
        for o, (box, lo, hi) in zip(objects, m.proc_pool.map(cards.v1_box, ws, [floors[o["shot"]] for o in objects], chunksize=max(1, len(objects) // 32))):
            o.update(box=box, box_min_m=np.round(lo, 3).tolist(), box_max_m=np.round(hi, 3).tolist(),
                     box_rule="robust: p2/p98 height x min-area p2-p98 footprint of the cleaned points (click MVP 4.2)")
    clock.mark("objects_lifted")

    by_frame = {}  # object keyframe -> {object: [its masks there]}
    for oi, mem in enumerate(members):
        for gi in mem:
            by_frame.setdefault(int(vf[gi]), {}).setdefault(oi, []).append(int(gi))

    # objects v1 at once: the name is SAM 3's word ('detected word, unverified', spec section 9). On ME340 it matched the
    # delivered names more often than SigLIP zero-shot or Qwen3-VL naming (runs 005-008), so the cascade's name rides
    # along in 'cascade' (v2) and never replaces it until a namer beats it
    for o in objects:  # mvp2/identity: a hazard word (spill, cable, guard ...) is never a label before its second check (cards.hazard_gate)
        o.update(label=cards.hazard_gate(cards.identity_v1(o), None)[0], label_source="sam3 word vote",
                 status="estimated box; name is a detected word, unverified")
    obj_labels = ["object names are detected words or model outputs: unverified", SCALE_LABEL]
    casc = {"objects": len(objects)}
    writer.put("objects", copy.deepcopy({"objects": objects, "cascade": casc, "vocabulary": words}), None, "estimated+inferred", obj_labels)
    clock.mark("objects_v1_put")

    # ---------- click MVP: object cards (section 4) and the judgement hook (B's fast_report.judge, when present) ----------
    cards_out, cards_ready, pick_ready, judge_futures, cards_lock = {}, threading.Event(), threading.Event(), [], threading.Lock()
    judge_lock = threading.Lock()
    carried = {}  # the judge's VLM answers by (row, question, keyframes), shared by its runs on cards v1 and v3
    card_labels = ["physical values are estimated (floor plane + assumed 1.6 m camera height) with +-u from view-subset disagreement plus depth, "
                   "pose, floor, resolution and scale terms; 'not observed' / 'not measurable' carry their reason",
                   "identity and class are inferred: a detected word until a calibrated decider answers",
                   "time from the pick maps; 'disappeared' only with before/after keyframes"]

    def cards_put(version, out):
        data = {"schema": "panoptes-object-cards-v1", "version": version, "version_of": {"objects": {1: 1, 2: 2}.get(version, 3), "pick": 1 if version < 3 else 2},
                "note": {1: "geometry; names are detected words", 2: "identity from the decider", 3: "after densify",
                         4: "after densify, identity from the decider"}.get(version),
                "calibration": cards_calibration(), "shots": out["shots"], "aliases": out["aliases"], "stats": out["stats"]}
        body = json.dumps(out["cards"], separators=(",", ":"), default=layers._plain).encode()
        blobs = {}
        if len(body) < 1 << 20:
            data["cards"] = out["cards"]
        else:
            data["cards"] = "blob"
            blobs["cards"] = (body, {"mediaType": "application/json", "format": "panoptes-object-cards-v1 cards"})
        if out.get("diagnostics"):  # mvp2: the tops' edge-vs-points record per object (evaluation only; never shown)
            blobs["diagnostics"] = (json.dumps(out["diagnostics"], separators=(",", ":"), default=layers._plain).encode(),
                                    {"mediaType": "application/json", "format": "panoptes-object-cards diagnostics"})
        writer.put("object_cards", data, blobs or None, "estimated+inferred", card_labels)
        clock.mark(f"cards_v{version}_put")

    def cards_job():
        with clock.stage("cards.inputs", n={"shots": len(geo)}):
            shots_in = []
            for si, gg in enumerate(geo):
                normal, point = shot_floor(gg)
                shots_in.append({"index": si, "keys": gg["keys"], "times": [k_ / fps for k_ in gg["keys"]], "c2w": gg["c2w_m"].cpu().numpy().astype(float),
                                 "K": gg["K"].cpu().numpy().astype(float), "normal": normal, "point_m": point, "mpu": gg["mpu"],
                                 "u_floor_m": cam_rows[si]["floor"].get("u_floor_m"), "scale_status": cam_rows[si]["scale_status"],
                                 "sharp": np.array([sharp_all[k_] for k_ in gg["keys"]]), "plumb_deg": cards.plumb(*gg["mesh_vf"], normal),
                                 "plumb_walls": cards.plumb_walls(*gg["mesh_vf"], normal),
                                 "depth": gg["depth_m"].cpu().numpy(), "person": gg["person"].cpu().numpy()})
        with clock.stage("cards.v1", n={"objects": len(objects)}):  # the pick maps' counts are read only for the time fields
            out = cards.build({"shots": shots_in, "objects": copy.deepcopy(objects), "points": obj_points,
                               "counts": lambda: (pick_ready.wait(120), pick_counts)[1],
                               "people": results.get("people"), "calibration": cards_calibration()}, m.proc_pool, 16)
        cards_out["v1"], cards_out["shots_in"] = out, shots_in
        cards_ready.set()
        cards_put(1, out)
        judge_hook(out, 1)
        return out["stats"]

    def judge_hook(out, version):
        """B's entry point (spec section 9 A's contract out): judge.run(cards, ctx, writer, clock) on each cards version, one
        after the other (a later version's judgements are never overwritten by an earlier one's VLM pass)."""
        try:
            from fast_report import judge
        except ImportError:
            return
        if not opts.get("judge", True):
            return
        outl = results.get("outlines_v2" if version >= 3 else "outlines") or {}
        ctx = judge.context(cam_rows, outl.get("frames", []), results.get("people"), frames,
                            {gg["index"]: gg["seeds"]["xyz"] for gg in geo if "seeds" in gg}, fps, (W, H), version_of={"object_cards": version})
        by = {x["index"]: x for x in out["shots"]}
        for x in ctx["shots"]:  # the cards' own pose, floor and plumb readings (B's context has constants)
            a_ = by.get(x["index"]) or {}
            x.update(u_pose_m=a_.get("u_pose_m", x["u_pose_m"]), u_floor_m=a_.get("u_floor_m") or x["u_floor_m"], angles_usable=a_.get("angles_usable"),
                     plumb_u_deg=a_.get("plumb_u_deg"), plumb_deg=a_.get("plumb_deg"))

        def job(prev):
            try:  # round 2: the rules and the hazard judge's questions go out now, beside the previous run's wait
                rows = judge.ahead(out["cards"], ctx, clock, carried, opts.get("hazard_ask"), m.proc_pool) if opts.get("judge_vlm", True) else None
            except Exception:  # noqa: BLE001  run() redoes the rules and asks what is missing
                rows = None
            if prev is not None:
                prev.result()
            try:
                return judge.run(out["cards"], ctx, writer, clock, vlm_on=opts.get("judge_vlm", True), pool=m.proc_pool, carried=carried,
                                 hazard_ask=opts.get("hazard_ask"), rows=rows)
            except Exception:  # noqa: BLE001  the judgements are one layer: their failure is recorded, the others stand
                import traceback
                return {"error": traceback.format_exc()[-3000:]}
        with judge_lock:  # the runs chain in call order (identity and densify call from two threads)
            judge_futures.append(m.cpu_pool.submit(job, judge_futures[-1] if judge_futures else None))

    def sync_objects():
        """objects v2 carries the cards' boxes (main cluster, fragments merged) and size checks once they exist."""
        if not cards_ready.wait(20):
            return
        out = cards_out["v1"]
        by = {c["id"]: c["physical"] for c in out["cards"] if c["kind"] == "object"}
        for o in objects:
            p = by.get(o["id"])
            if p and "box" in p:
                o.update(box=p["box"], box_min_m=p["box_min_m"], box_max_m=p["box_max_m"], size_check=p["size_check"].get("status"),
                         box_rule="cards: main cluster of the cleaned points, fragments merged (click MVP 4.2)")
            if o["id"] in out["aliases"]:
                o["merged_into"] = out["aliases"][o["id"]]

    def with_identity(out, idents):
        """A cards version with these identities (by object id), every name-dependent field re-derived on this version's
        measurements (cards.apply_name: the single source of truth, mvp2/identity R1)."""
        new = copy.deepcopy(out)
        for c in new["cards"]:
            if c["kind"] == "object" and c["id"] in idents:
                c["identity"] = copy.deepcopy(idents[c["id"]])
                cards.apply_name(c)
        return new

    def publish(idents, route):
        """Every identity update (R1): the newest cards version with it (v2, or v4 once densify's v3 is out), then that
        version's judgements, so kind, size check, checks and verdicts follow the final names."""
        with cards_lock:
            cards_out["identities"] = {**cards_out.get("identities", {}), **idents}
            v = 4 if "v3" in cards_out else 2
            cards_out[f"v{v}"] = out = with_identity(cards_out["v3" if v == 4 else "v1"], cards_out["identities"])
            cards_put(v, out)
            shown = {c["id"]: c["identity"] for c in out["cards"] if c["kind"] == "object"}
            for o in objects:  # the objects layer's labels follow too (carried by its next version)
                if o["id"] in idents and o["id"] in shown:
                    o.update(label=shown[o["id"]]["name"], label_source=shown[o["id"]].get("decided_by"))
        clock.mark(f"identity_{route}_put")
        judge_hook(out, v)

    def cards_v2():
        """Identity once the cards exist: Gemini's open names through the bench's relay when it is there (mvp2/identity), the
        lettered Qwen decider for what it did not name (section 4.7). Put as v2, or merged into densify's v3 as v4. A failure is
        recorded, never raised: raised, it ended the run while densify's SAM 3 still ran, and run()'s cleanup then faulted
        both GPUs for every later call (runs mvp-integrate-*-002/003, CUDA illegal address)."""
        try:
            return identity_pass()
        except Exception:  # noqa: BLE001
            import traceback
            return {"error": traceback.format_exc()[-3000:]}

    def identity_pass():
        if not cards_ready.wait(120):
            namer["published"].set()
            return
        by = {o["id"]: o for o in objects}
        new = copy.deepcopy(cards_out["v1"]["cards"])
        for c in new:
            if c["kind"] == "object":
                c["identity"] = cards.identity_v2(c["identity"], by[c["id"]], c["physical"].get("size_check") or {})
        namer["started"].wait(90)  # the outlines start it; a failed outlines job leaves the Qwen decider alone
        named, rec = namer["future"].result() if namer.get("future") else ({}, {"namer": "none: no relay or no outlines (Qwen decider only)"})
        for c in new:
            if c["id"] in named:
                c["identity"] = cards.open_identity(c["identity"], named[c["id"]])
        if named:  # every answer is kept: an object merged into another card here may be a card of its own in densify's v3
            rows = {o["id"]: o for o in objects}
            idents = {i: cards.open_identity(cards.identity_v1(rows[i]), a) for i, a in named.items() if i in rows}
            publish({**idents, **{c["id"]: c["identity"] for c in new if c["id"] in named}}, "gemini")
        namer["published"].set()
        if named:  # what Gemini did not name in time keeps its detected word until densify's pass asks again (the Qwen decider
            return rec  # for 56 late objects put Sam's Club's identity 8 s later, with the weaker names)
        rest = [c for c in new if c["kind"] == "object"]
        for which in ("ehs", "other"):  # no Gemini at all: the Qwen decider, EHS classes' names first (the checks read them)
            if ask_identity(rest, which):
                publish({c["id"]: c["identity"] for c in rest}, f"qwen_{which}")
        return rec

    def views_for_identity(key="outlines"):
        """{entity: (keyframe, marks {1: its polygons, 2..: the others'})} on segmented outlines: the card's first best view
        that was segmented, else its largest segmented outline."""
        outl = {f["sourceFrame"]: f for f in (results.get(key) or {}).get("frames", []) if f["source"] == "segmented"}
        area = {}
        for q, f in outl.items():
            for o in f["objects"]:
                xy = np.concatenate([np.asarray(pp, float).reshape(-1, 2) for pp in o["polygons"]]) if o["polygons"] else None
                if xy is not None and len(xy):
                    area.setdefault(o["entityId"], []).append((float(np.prod(xy.max(0) - xy.min(0))), q))

        def view(c):
            best = next((k for k in (c.get("views") or {}).get("best", []) if k in outl), None)
            if best is None and area.get(c["id"]):
                best = max(area[c["id"]])[1]  # integration: the object's largest segmented outline (its best views were projected)
            if best is None:
                return None
            marks, n = {}, 2
            for o in outl[best]["objects"]:
                if o["entityId"] == c["id"]:
                    marks[1] = o["polygons"]
                else:
                    marks[n], n = o["polygons"], n + 1
            return (best, marks) if 1 in marks else None
        return view

    def ehs(c):
        i = c["identity"]
        return c["class"]["category"] != "other" or cards.head_match(i.get("proposed") or i["name"], cards.CLASS_SIZE) is not None \
            or any(cards.hazard_of(w) for w in i.get("detector_words") or [])

    def to_name(c, which=None):
        """The objects asked for a name: EHS classes (their SAM 3 words) whatever their views, every other seen on >= 3 views."""
        return c["kind"] == "object" and (ehs(c) if which == "ehs" else not ehs(c) and (c.get("views") or {}).get("n", 0) >= 3 if which else
                                          ehs(c) or (c.get("views") or {}).get("n", 0) >= 3)

    namer = {"started": threading.Event(), "published": threading.Event()}

    def start_namer():
        """mvp2/identity: the Gemini naming starts as soon as the outlines and pick counts exist (3-4 s before cards v1)."""
        if opts.get("namer") is None or not objects:
            namer["started"].set()
            return
        from concurrent.futures import Future
        namer["future"] = fut = Future()

        def run():
            try:
                fut.set_result(gemini_names())
            except Exception:  # noqa: BLE001  no names: the Qwen decider names everything, as before
                import traceback
                fut.set_result(({}, {"namer": "gemini failed", "error": traceback.format_exc()[-2000:]}))
        threading.Thread(target=run, name="namer", daemon=True).start()
        namer["started"].set()

    def gemini_names(rows=None, key="outlines", counts=None, ready=None, first=0, tag="gemini"):
        """mvp2/identity (R2): open names from Gemini (cloud) for every object seen on >= 3 keyframes (pick counts) or carrying
        an EHS word (the first-pass objects; densify's new ones on its outlines in a second pass), on its largest segmented
        outline; two a sheet (the thing in its surroundings | a close crop), 14 a request, every request at once. The requests
        go out on the event stream (writer.send), the bench relays them into the deployed report container and puts each answer
        on opts['namer'] (a modal.Queue, this report's partition). -> ({object id: {name, status, p}}, record); what is
        unanswered after NAMER_WAIT_S keeps its detected word (first pass: goes to the Qwen decider)."""
        import queue as _queue
        relay, view = opts["namer"], views_for_identity(key)
        counts = pick_counts if counts is None else counts
        (ready or pick_ready).wait(60)

        def wanted(o):
            i = cards.identity_v1(o)
            seen = sum(1 for c in (counts.get(o["id"]) or {}).values() if max(c) >= cards.MIN_PX)
            return seen >= 3 or cards.kind_of(i["proposed"])["category"] != "other" or cards.head_match(i["proposed"], cards.CLASS_SIZE) is not None \
                or any(cards.hazard_of(w) for w in i["detector_words"])
        todo = [o for o in list(objects if rows is None else rows) if wanted(o)]
        with clock.stage("identity.gemini.sheets", n={"objects": len(todo)}):
            with ThreadPoolExecutor(8) as pool:
                tiles = list(pool.map(lambda o: (lambda v: v and vlm.namer_tile(frames[v[0]], v[1][1]))(view({"id": o["id"], "views": {}})), todo))
            ids = [o["id"] for o, t in zip(todo, tiles) if t is not None]
            reqs = vlm.namer_requests(ids, [t for t in tiles if t is not None], first)

        def send(r, attempt):
            writer.send({"type": "namer_request", "report": writer.report_id, "request": r["request"], "attempt": attempt, "n": len(r["ids"]),
                         "blocks": r["blocks"]})
        sent = time.time()
        for r in reqs:
            send(r, 1)
        clock.mark(f"identity_{tag}_sent")
        got, pending, again = {}, {r["request"]: r for r in reqs}, set()
        rec = {"namer": "gemini via the bench relay", "asked": len(ids), "requests": len(reqs), "answers": []}
        with clock.stage("identity.gemini", n={"objects": len(ids), "requests": len(reqs)}):
            while pending and time.time() - sent < NAMER_WAIT_S:
                wake = NAMER_WAIT_S if len(again) >= len(reqs) or time.time() - sent >= NAMER_HEDGE_S else NAMER_HEDGE_S
                try:
                    a = relay.get(timeout=max(.1, wake - (time.time() - sent)), partition=writer.report_id)
                except _queue.Empty:
                    a = None
                r = pending.get((a or {}).get("request"))
                if r is not None:
                    ans = vlm.namer_answers(r, a.get("provider"))
                    rec["answers"].append({k: a.get(k) for k in ("request", "attempt", "s", "status", "error", "usage")} | {"at_s": round(time.time() - clock.t0_unix, 2)})
                    if ans:
                        got.update(ans)
                        del pending[r["request"]]
                    elif r["request"] not in again:  # an error or an empty answer: once more, at once
                        again.add(r["request"])
                        send(r, 2)
                if time.time() - sent >= NAMER_HEDGE_S:
                    for k, r in pending.items():  # the tail: a second copy of every request still out
                        if k not in again:
                            again.add(k)
                            send(r, 2)
        rec.update(named=len(got), late_requests=sorted(pending), resent=sorted(again))
        return got, rec

    def ask_identity(card_list, which="ehs"):
        """Section 4.7 step 3 through B's decider (vlm.options + judge.som, when both exist): the best view with the outlines
        as numbered white-over-black marks, the subject [1]. -> the number of questions asked."""
        try:
            from fast_report import judge
            som = judge.som
        except ImportError:
            return 0
        cal, view = cards_calibration(), views_for_identity()

        def build(c):  # CPU: the set-of-marks pair and the lettered prompt
            v = view(c)
            if v is None:
                return None
            best, marks = v
            opts_ = cards.identity_options(c["identity"])
            # integration fix: the options go into the prompt as letters (vlm.qwen_prompt); A's bare question listed none
            p = vlm.qwen_prompt(" ".join([judge.SCENE, judge.MARKS]), "What is the object marked [1]?", opts_)
            # 336 px crops: ~144 image tokens each instead of 256 (the identity pass is prefill-bound: Sam's Club 413 questions, 87 s)
            return opts_, [som(frames[best], marks, subject=1, side=336), som(frames[best], marks, subject=1, marks=False, side=336)], p
        # integration: every object seen on >= 3 views is asked (spec 5.3's 'identity for the other objects', after the
        # judgement questions): the SAM 3 word alone named a floor drain 'metal part' and a flammables cabinet 'machine'
        todo = [c for c in card_list if to_name(c, which)]
        if not todo:
            return 0
        with clock.stage(f"vlm.identity.{which}", n={"objects": len(todo)}):
            with ThreadPoolExecutor(8) as pool:  # never the core's cpu_pool: its threads would wait on vLLM (the judge's queue)
                built = list(pool.map(build, todo))
            asked = [(c, b[0], vlm.submit(b[1], b[2], len(b[0]), "identity" if ehs(c) else "identity_other")) for c, b in zip(todo, built) if b is not None]
            for c, opts_, fut in asked:
                try:
                    c["identity"] = cards.decide_identity(c["identity"], opts_, fut.result(), cal)
                except Exception as error:  # noqa: BLE001  one unanswered question leaves that card's detected word
                    c["identity"] = {**c["identity"], "decider": {"question": "identity", "answer": "unanswered", "error": repr(error)[:200]}}
        return len(asked)

    cards_future = m.cpu_pool.submit(cards_job)

    def sam3d_inputs():
        """B's Obj: every object's SAM 3 logits on each keyframe it was seen in (the max over its masks there, B's fixture rule)."""
        with torch.inference_mode(), clock.stage("sam3d.inputs", gpu=dev_geo, n={"objects": len(objects)}):
            pair, owner, rows = {}, [], []
            for oi, mem in enumerate(members):
                for gi in mem:
                    owner.append(pair.setdefault((oi, int(keys[vf[gi]])), len(pair)))
                    rows.append(int(gi))
            lr = torch.full((len(pair), segment.LR, segment.LR), -1e4, dtype=torch.half, device=dev_geo)
            if rows:
                lr.index_reduce_(0, torch.tensor(owner, device=dev_geo), voc["logits"][torch.tensor(rows, device=dev_geo)], "amax")
            lr = lr.cpu().numpy()
        inputs_taken.set()
        objs = [{"id": o["id"], "shot": o["shot"], "word": o["word"], "box_min_m": o["box_min_m"], "box_max_m": o["box_max_m"], "masks_lr": {}}
                for o in objects[:len(members)]]
        for (oi, f), r in pair.items():
            objs[oi]["masks_lr"][f] = lr[r]
        return objs

    inputs_taken = threading.Event()

    def release_voc():
        """The SAM 3 masks and logits off GPU 0 after their last reader (SAM 3D inputs, outlines, cascade, VLM crops), and this
        process's cache with them: SAM 3D's two processes peak at ~28 GiB each beside it (Sam's Club: GPU 0 at 75 GiB)."""
        inputs_taken.wait(300)  # ponytail: a failed models job never blocks the report
        maps_ready.wait()
        if voc is not None:
            voc.clear()
        if not densify_on:  # densify: the one empty_cache is models_job's, after its SAM 3 (see below)
            with torch.cuda.device(dev_geo):
                torch.cuda.empty_cache()

    # complete models: SAM 3D + the fit gate on GPU 0's two processes; its inputs are gathered now, the generation waits for
    # the facts (click MVP section 7: densify, cards and judgements before the display layers)
    sam3d_objs = m.cpu_pool.submit(sam3d_inputs)
    display = {}

    def release_splat():
        if not release.is_set():
            release.set()  # the splat trains on GPU 1 from here
            clock.mark("splat_started")

    def start_display():
        release_splat()
        if display:
            return
        display["models"] = m.cpu_pool.submit(models_job, m, sam3d_objs.result, geo, shared, words, clock, writer, dev_geo)
        clock.mark("display_started")
    if not densify_on or not objects:
        start_display()
    v1_labels = {o["id"]: o["label"] for o in objects}
    gpu0_free, cascade_done = threading.Event(), threading.Event()

    def densify_sam3(dev, batches, dens, lock):
        """Section 7: SAM 3 with every word on the 5 fps keyframes the objects skipped, from one queue both GPUs take from;
        per batch the flood dedupe, a label map per frame at 640x360 from the kept masks' logits (the smaller wins), and the
        kept masks bit-packed on the CPU (X1's layout)."""
        sam, words_t, oh, ow = m.sams[dev], tuple(words), H // 2, W // 2
        with torch.cuda.device(dev), torch.inference_mode():
            while True:
                try:
                    batch = batches.get_nowait()
                except queue.Empty:
                    return
                print(f"densify gpu{dev.index} frames {batch} t={clock.now():.2f}", flush=True)  # a breadcrumb in the container log
                with clock.stage(f"densify.sam3@gpu{dev.index}", gpu=dev, n={"frames": len(batch), "words": len(words_t)}):
                    x = torch.stack([work.chunks[dev][q // segment.PERSON_FRAMES][q % segment.PERSON_FRAMES] for q in batch])
                    r = sam.detect(sam.vision(x), len(batch), words_t, segment.VOCAB_SCORE, logits=True)
                    r["frame"] = torch.tensor(batch, device=dev)[r["frame"]]
                    kd, vd = segment.dedupe(r["frame"], r["word"], r["score"], r["mask"]) if len(r["frame"]) else (np.zeros(0, int), [])
                    kt = torch.from_numpy(kd).to(dev)
                    fk = r["frame"][kt]
                    maps_d = {}
                    for q in batch:
                        sel = torch.nonzero(fk == q).squeeze(1)
                        if not len(sel):
                            maps_d[q] = (np.zeros((oh, ow), np.int16), np.zeros(0, np.int64))
                            continue
                        up = F.interpolate(r["logits"][kt[sel]][None].float(), size=(oh, ow), mode="bilinear", align_corners=False)[0] > 0
                        maps_d[q] = (segment.paint(up).short().cpu().numpy(), sel.cpu().numpy())
                    packed = segment.pack(r["mask"][kt]) if len(kt) else np.zeros((0, DA3_HW[0], DA3_HW[1] // 8), np.uint8)
                    fr_np, w_np, s_np = fk.cpu().numpy(), r["word"][kt].cpu().numpy(), r["score"][kt].float().cpu().numpy()
                    at = {int(a): i for i, a in enumerate(kd)}
                with lock:
                    base = len(dens["q"])
                    dens["q"] += fr_np.tolist()
                    dens["word"] += w_np.tolist()
                    dens["score"] += s_np.tolist()
                    dens["packed"].append(packed)
                    dens["votes"] += [(base + at[int(a)], int(w_), float(s_)) for a, w_, s_ in vd]
                    for q, (lab, sel) in maps_d.items():
                        dens["maps"][q] = (lab, base + sel)

    def densify_job():
        try:
            return densify()
        finally:
            start_display()
            for d in work.chunks:
                work.chunks[d] = []

    def densify():
        """Section 7: every 5 fps keyframe segmented (X1: in-between outline IoU 0.70 -> 0.81 on ME340). New masks join an
        existing object by the lift's overlap rule (segment.join), the rest are lifted among themselves (new objects seen on
        >= 2 keyframes); existing ids never change. Then outlines v2, pick v2, objects v3 and cards v3; SAM 3D and the splat
        start after its GPU part."""
        frames_d = [q for gg in geo for q in gg["pos"] if q not in by_frame]
        batches = queue.Queue()
        for b in range(0, len(frames_d), DENSIFY_BATCH):
            batches.put(frames_d[b:b + DENSIFY_BATCH])
        dens, lock = {"q": [], "packed": [], "word": [], "score": [], "votes": [], "maps": {}}, threading.Lock()
        g1 = m.cpu_pool.submit(densify_sam3, dev_seg, batches, dens, lock) if dev_seg != dev_geo else None
        gpu0_free.wait(300)  # ponytail: a failed cascade never holds densify forever
        densify_sam3(dev_geo, batches, dens, lock)
        if g1 is not None:
            g1.result()
        clock.mark("densify_sam3_done")
        n_d = len(dens["q"])
        qs = np.asarray(dens["q"], np.int64)
        packed = np.concatenate(dens["packed"]) if dens["packed"] else np.zeros((0, DA3_HW[0], DA3_HW[1] // 8), np.uint8)
        ent = np.zeros(n_d, np.int64)  # object index + 1 per densify mask, 0 = none
        vote_d = {}
        for g_, w_, s_ in dens["votes"]:
            vote_d.setdefault(g_, []).append((w_, s_))
        n_old = len(members)
        added, new_objs, new_points, st = {}, [], [], {"frames": len(frames_d), "masks_kept": n_d, "joined": 0, "new_objects": 0}
        with torch.inference_mode(), clock.stage("densify.lift", gpu=dev_geo, n={"masks": n_d}):
            for si, gg in enumerate(geo):
                pos = gg["pos"]
                idx = np.flatnonzero(np.isin(qs, pos))
                if not len(idx):
                    continue
                local = torch.full((len(keys),), -1, dtype=torch.long, device=dev_geo)
                local[torch.tensor(pos, device=dev_geo)] = torch.arange(len(pos), device=dev_geo)
                dyn_s = dyn[torch.tensor(pos, device=dev_geo)]
                codes, owner = shot_voxels.get(si, (None, None))
                unmatched = []
                for b0 in range(0, len(idx), 4096):
                    b = idx[b0:b0 + 4096]
                    p, _ = segment.mask_points(segment.unpack(packed[b], dev_geo), local[torch.from_numpy(qs[b]).to(dev_geo)], gg["depth_m"], gg["K"],
                                               gg["c2w_m"], dyn_s)
                    if p is None:
                        continue
                    lifted = p["lifted"].cpu().numpy()
                    who = segment.join(p, codes, owner)[0] if codes is not None else torch.full((len(lifted),), -1, dtype=torch.long, device=dev_geo)
                    w_np = who.cpu().numpy()
                    ent[b[lifted[w_np >= 0]]] = w_np[w_np >= 0] + 1
                    unmatched += b[lifted[w_np < 0]].tolist()
                    got = np.unique(w_np[w_np >= 0])
                    lab = torch.where(who >= 0, who, torch.full_like(who, int(max(got.max(initial=0), 0)) + 1))
                    for oi, d in zip(got, object_points({**p, "label": lab}, p["fr"], got)):
                        added.setdefault(int(oi), []).append(d)
                if len(unmatched) >= 2:
                    um = np.asarray(unmatched)
                    comp, arr, _, pts = segment.lift(segment.unpack(packed[um], dev_geo), local[torch.from_numpy(qs[um]).to(dev_geo)], gg["depth_m"],
                                                     gg["K"], gg["c2w_m"], dyn_s)
                    if arr is not None:
                        conf = np.flatnonzero(arr["frames"] >= segment.CONFIRMED)
                        floor = cards.floor_frame(gg["c2w_m"][0].cpu().numpy(), *shot_floor(gg))
                        for c, pp in zip(conf, object_points(pts, pts["fr"], conf)):
                            mem = np.flatnonzero(comp == c)
                            vv = {}
                            for g_ in um[mem]:
                                for w_, s_ in vote_d.get(int(g_), []):
                                    vv[words[w_]] = vv.get(words[w_], 0.) + s_
                            oi = n_old + len(new_objs)
                            best = um[mem][np.argmax(np.asarray(dens["score"])[um[mem]])]
                            box, lo, hi = cards.v1_box(pp["world"], floor)
                            word = segment.name(vv, words)
                            new_objs.append({"id": f"obj-{si}-{oi}", "shot": si, "word": word, "label": word, "label_source": "sam3 word vote",
                                             "votes": {w: round(v, 3) for w, v in sorted(vv.items(), key=lambda x: -x[1])[:5]},
                                             "frames": int(arr["frames"][c]), "masks": int(len(mem)), "voxels": int(arr["voxels"][c]),
                                             "centroid_m": arr["centroid"][c].round(3).tolist(), "best_key": int(keys[qs[best]]),
                                             "voxel_box_m": [(arr["lo"][c] - segment.LIFT_VOXEL / 2).round(3).tolist(),
                                                             (arr["hi"][c] + segment.LIFT_VOXEL / 2).round(3).tolist()],
                                             "box": box, "box_min_m": np.round(lo, 3).tolist(), "box_max_m": np.round(hi, 3).tolist(),
                                             "box_rule": "robust: p2/p98 height x min-area p2-p98 footprint of the cleaned points (click MVP 4.2)",
                                             "source": "densify", "status": "estimated box; name is a detected word, unverified"})
                            new_points.append(pp)
                            ent[um[mem]] = oi + 1
            maps_v2 = [e for e in results.get("maps_v1", []) if e[0]["source"] == "segmented"]
            for q, (lab, gidx) in dens["maps"].items():
                lut = np.r_[0, ent[gidx]].astype(np.int16)
                maps_v2.append(({"timeSec": round(keys[q] / fps, 4), "sourceFrame": int(keys[q]), "source": "segmented"}, lut[lab], W / (W // 2),
                                H / (H // 2), q))
        st.update(joined=int((ent[:] > 0).sum() - sum(new_o["masks"] for new_o in new_objs)), new_objects=len(new_objs),
                  objects_gaining_views=len(added))
        release_splat()  # densify's GPU 1 share is done: the splat trains; SAM 3D (GPU 0, and 16 CPU processes) after cards v3
        points_v3 = list(obj_points)
        for oi, ds in added.items():
            points_v3[oi] = merge_points([obj_points[oi]] + ds)
        cascade_done.wait()  # the cascade indexes the first objects: new ones join the list after it
        objects.extend(new_objs)
        points_v3 += new_points
        labels_v2 = {**v1_labels, **{o["id"]: o["label"] for o in new_objs}}
        with clock.stage("outlines.polygons.v2", n={"frames": len(maps_v2)}):
            counts_v2, ready_v2 = {}, threading.Event()
            polys = m.proc_pool.map(segment.label_polygons, [x[1] for x in maps_v2], [x[2] for x in maps_v2], [x[3] for x in maps_v2])
            pick2 = pick_layer(maps_v2, counts_v2, ready_v2)
            frames_out = []
            for (entry, _, _, _, _), found in zip(maps_v2, polys):
                entry = dict(entry, objects=[{"entityId": objects[v - 1]["id"], "label": labels_v2[objects[v - 1]["id"]], "polygons": poly,
                                              "source": "segmented"} for v, poly in sorted(found.items())])
                frames_out.append(entry)
            frames_out.sort(key=lambda e: e["timeSec"])
            for e, nxt in zip(frames_out, frames_out[1:] + [None]):
                e["endTimeSec"] = nxt["timeSec"] if nxt else round(e["timeSec"] + BLOCK / fps, 4)
            results["outlines_v2"] = {"width": W, "height": H, "frames": frames_out}
            analysis = json.dumps(results["outlines_v2"], separators=(",", ":")).encode()
        pick_data, pick_blobs = pick2.result()
        writer.put("outlines", {"analysis": "blob", "frames": len(frames_out), "segmented": len(frames_out), "projected": 0, "densified": True},
                   {"analysis": (analysis, {"mediaType": "application/json", "format": "video-analysis"})}, "observed(segmented)",
                   ["every 5 fps keyframe segmented (SAM 3 masks): the densify pass", "labels as of the first objects version"])
        writer.put("pick", {**pick_data, "version_note": "v2: every keyframe segmented"}, pick_blobs, "observed(segmented)",
                   ["segmented frames: SAM 3 masks (observed)", SCALE_LABEL])
        clock.mark("pick_v2_put")
        writer.put("objects", copy.deepcopy({"objects": objects, "cascade": casc, "vocabulary": words, "densify": st}), None, "estimated+inferred", obj_labels)
        clock.mark("objects_v3_put")
        cards_ready.wait(120)
        with clock.stage("cards.v3", n={"objects": len(objects)}):
            shots_in = cards_out.get("shots_in")
            out = cards.build({"shots": shots_in, "objects": copy.deepcopy(objects), "points": points_v3, "counts": counts_v2,
                               "people": results.get("people"), "calibration": cards_calibration()}, m.proc_pool, 16)
        with cards_lock:  # the identities known now (v2's, else v1's words); a later decider pass merges in as v4
            prev = cards_out.get("identities") or {c["id"]: c["identity"] for c in cards_out.get("v1", {}).get("cards", []) if c["kind"] == "object"}
            out = with_identity(out, prev)
            cards_out["v3"] = out
            cards_put(3, out)
        start_display()  # facts before display (section 7): the gate's CPU processes slowed cards v3 by 2-3x beside it (run 005)
        judge_hook(out, 3)
        v3_ids = {c["id"] for c in out["cards"] if c["kind"] == "object"}  # densify's objects that are cards (the rest merged away)
        new_rows = [o for o in objects if o["id"] in v3_ids and o["id"] not in cards_out.get("identities", {})]
        if new_rows and opts.get("namer") is not None:  # mvp2/identity: densify's own objects named too (a second, smaller pass)
            from concurrent.futures import Future
            namer["densify"] = fut = Future()

            def name_densified():
                try:  # after the first pass's names are in: only what they left (Sam's Club asked 623 twice when v3 came first)
                    namer["published"].wait(120)
                    rows_now = [o for o in new_rows if o["id"] not in cards_out.get("identities", {})]
                    got, rec = gemini_names(rows_now, "outlines_v2", counts_v2, ready_v2, first=100, tag="gemini_densify")
                    by = {c["id"]: c for c in out["cards"] if c["kind"] == "object"}
                    idents = {i: cards.open_identity(by[i]["identity"], a) for i, a in got.items() if i in by}
                    if idents:
                        publish(idents, "gemini_densify")
                    fut.set_result(rec)
                except Exception:  # noqa: BLE001  those objects keep their detected words
                    import traceback
                    fut.set_result({"error": traceback.format_exc()[-2000:]})
            threading.Thread(target=name_densified, name="namer-densify", daemon=True).start()
        return {**st, "cards": out["stats"]}
    densify_future = m.cpu_pool.submit(densify_job) if densify_on and objects else None

    maps_ready = threading.Event()

    def outlines_job():
        """Object keyframes 'segmented' (SAM 3 logits), other 5 fps keyframes 'projected' (E6b pair); beside the cascade,
        with the first objects version's names. Per frame, every object's union of its masks in one reduction."""
        try:
            members_of = {}  # keyframe -> (object ids, their masks' indices, owner of each mask)
            for q, objs in by_frame.items():
                ids = list(objs)
                members_of[q] = (ids, torch.tensor([gi for o in ids for gi in objs[o]], device=dev_geo),
                                 torch.tensor([k for k, o in enumerate(ids) for _ in objs[o]], device=dev_geo))

            def per_object(x, q, fill):
                ids, idx, owner = members_of[q]
                return torch.full((len(ids), *x.shape[1:]), fill, dtype=torch.half, device=dev_geo).index_reduce_(0, owner, x[idx].half(), "amax")

            def global_ids(q, lab):
                return torch.cat([torch.zeros(1, dtype=torch.long, device=dev_geo), torch.tensor(members_of[q][0], device=dev_geo) + 1])[lab]
            maps = []  # (entry, label map int16 numpy (0 = nothing, i + 1 = object i), sx, sy): polygons in the process pool
            oh, ow = H // 2, W // 2
            with torch.inference_mode(), clock.stage("outlines.segmented", gpu=dev_geo, n={"frames": len(by_frame)}):
                for q in sorted(by_frame):
                    lg = per_object(voc["logits"], q, -1e4).float()
                    lab = segment.paint(F.interpolate(lg[None], size=(oh, ow), mode="bilinear", align_corners=False)[0] > 0)
                    maps.append(({"timeSec": round(keys[q] / fps, 4), "sourceFrame": int(keys[q]), "source": "segmented"},
                                 global_ids(q, lab).short().cpu().numpy(), W / ow, H / oh, q))
            with torch.inference_mode(), clock.stage("outlines.projected", gpu=dev_geo):
                for si, gg in enumerate(geo):
                    pos = gg["pos"]
                    d = torch.where(gg["depth_m"] > 0, gg["depth_m"], torch.full_like(gg["depth_m"], 1e4))
                    d4 = -F.max_pool2d(-d[:, None], 4)[:, 0]  # section 3.3: nearest visible surface per 4 x 4 block, mm
                    depth4[si] = torch.where(d4 < 1e4, (d4 * 1000).round().clamp(1, 65535), torch.zeros_like(d4)).cpu().numpy().astype("<u2")
                    obj_local = [j for j, q in enumerate(pos) if q in by_frame]
                    if not obj_local:
                        continue
                    idm = torch.zeros((len(pos), *DA3_HW), dtype=torch.long, device=dev_geo)
                    for j in obj_local:  # every object on a keyframe belongs to that keyframe's shot (the lift is per shot)
                        idm[j] = global_ids(pos[j], segment.paint(per_object(voc["mask"], pos[j], 0) > 0))
                    people_local = dyn[torch.tensor(pos, device=dev_geo)]
                    for j, q in enumerate(pos):
                        if q in by_frame:
                            continue
                        lab = segment.project_pair(idm, gg["depth_m"], gg["K"], gg["c2w_m"], obj_local, j)
                        lab[people_local[j]] = -1  # people are not projected (E6b caveat): cut out with this frame's SAM 3 person mask
                        maps.append(({"timeSec": round(keys[q] / fps, 4), "sourceFrame": int(keys[q]), "source": "projected"},
                                     lab.clamp(min=0).short().cpu().numpy(), W / DA3_HW[1], H / DA3_HW[0], q))
        finally:
            maps_ready.set()  # the cascade's SigLIP pass waits for the GPU part of the outlines, never longer
        polys = m.proc_pool.map(segment.label_polygons, [x[1] for x in maps], [x[2] for x in maps], [x[3] for x in maps])
        results["maps_v1"] = maps
        pick = pick_layer(maps, pick_counts, pick_ready)  # section 3: people painted over the same maps, RLE in the pool beside the polygons
        with clock.stage("outlines.polygons", n={"frames": len(maps)}):
            polys = list(polys)
            frames_out = []
            for (entry, _, _, _, _), found in zip(maps, polys):
                entry["objects"] = [{"entityId": objects[v - 1]["id"], "label": v1_labels[objects[v - 1]["id"]], "polygons": poly, "source": entry["source"]}
                                    for v, poly in sorted(found.items())]
                frames_out.append(entry)
            frames_out.sort(key=lambda e: e["timeSec"])
            results["outline_frames"] = frames_out
            for e, nxt in zip(frames_out, frames_out[1:] + [None]):
                e["endTimeSec"] = nxt["timeSec"] if nxt else round(e["timeSec"] + BLOCK / fps, 4)
            results["outlines"] = {"width": W, "height": H, "frames": frames_out}
            start_namer()
            analysis = json.dumps(results["outlines"], separators=(",", ":")).encode()
        pick_data, pick_blobs = pick.result()  # outlines and pick back to back: one commit (section 3.3)
        writer.put("outlines", {"analysis": "blob", "frames": len(frames_out), "segmented": sum(e["source"] == "segmented" for e in frames_out),
                                "projected": sum(e["source"] == "projected" for e in frames_out)},
                   {"analysis": (analysis, {"mediaType": "application/json", "format": "video-analysis"})}, "observed(segmented)/estimated(projected)",
                   ["'segmented' outlines are SAM 3 masks on keyframes; 'projected' ones are carried from 3D (E6b 'pair'): no accuracy claimed",
                    "labels as of the first objects version; the objects layer holds the latest"])
        clock.mark("outlines_put")
        writer.put("pick", pick_data, pick_blobs, "observed(segmented)/estimated(projected)",
                   ["segmented frames: SAM 3 masks (observed); projected frames: carried from 3D (estimated)", SCALE_LABEL])
        clock.mark("pick_put")

    depth4, pick_counts = {}, {}  # shot -> (n,70,126) mm; object id -> {local keyframe: [segmented, projected] px at 640x360}

    def pick_layer(maps, counts, ready):
        """panoptes-pick-v1 (section 3.3): one id map per 5 fps keyframe of every geometry shot (objects as outlined, people
        painted last with their tracks, the smaller person on top), RLE per frame in the process pool; nearest depth per
        4 x 4 block. -> future of (data, blobs); fills pick_counts for the cards' time (section 4.6)."""
        import gzip
        n_objects = len(objects)  # objects added later (densify) are not on these maps
        with clock.stage("pick.maps"):
            people_ids = sorted(set(person_entity.values()) | {"person:untracked"})
            ent = [None] + [o["id"] for o in objects[:n_objects]] + people_ids
            pidx = {e: n_objects + 1 + i for i, e in enumerate(people_ids)}
            by_q = {x[4]: x for x in maps}
            labs, meta = [], []
            for si, gg in enumerate(geo):
                for j, q in enumerate(gg["pos"]):
                    x = by_q.get(q)
                    lab = x[1].astype(np.uint16) if x else np.zeros(DA3_HW, np.uint16)
                    h, w = lab.shape
                    for gi, mk in sorted(person_kept.get(q, []), key=lambda t: -int(t[1].sum())):
                        mk = mk if mk.shape == (h, w) else cv2.resize(mk.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
                        lab[mk] = pidx[person_entity.get(gi, "person:untracked")]
                    labs.append(lab)
                    meta.append((si, j, q, x[0]["source"] if x else "people only", w, h))
            enc = m.proc_pool.map(segment.pick_frame, labs)

        def finish():
            try:
                return encode()
            finally:
                ready.set()  # the cards go on with what there is

        def encode():
            with clock.stage("pick.encode", n={"frames": len(labs)}):
                frames_pk, chunks, depth, offset = [], [], [], 0
                for (si, j, q, source, w, h), (rle, vals, cnts) in zip(meta, enc):
                    t = round(keys[q] / fps, 4)
                    frames_pk.append({"t": t, "frame": int(keys[q]), "shot": si, "key": j, "w": w, "h": h, "source": source, "offset": offset,
                                      "pairs": len(rle) // 4})
                    chunks.append(rle)
                    offset += len(rle) // 4
                    depth.append(depth4[si][j].tobytes())
                    sc = (W // 2) * (H // 2) / (w * h)
                    for v, c in zip(vals, cnts):
                        if 1 <= v <= n_objects:
                            counts.setdefault(objects[v - 1]["id"], {}).setdefault(j, [0, 0])[source != "segmented"] += int(round(c * sc))
                for e, nxt in zip(frames_pk, frames_pk[1:] + [None]):
                    e["t_end"] = nxt["t"] if nxt and nxt["t"] - e["t"] <= 2 * BLOCK / fps + 1e-6 else round(e["t"] + BLOCK / fps, 4)
                    e["t_end"] = min(e["t_end"], round((shots[e["shot"]][1] + 1) / fps, 4))  # never past the shot's last frame (a cut)
                data = {"format": "panoptes-pick-v1", "source_wh": [W, H], "entities": ent, "frames": frames_pk,
                        "depth": {"w": DA3_HW[1] // 4, "h": DA3_HW[0] // 4, "unit": "mm", "scale": "estimated", "grid": "DA3 504x280 / 4, min per block"},
                        "note": "segmented frames: SAM 3 masks (observed); projected frames: carried from 3D (estimated); 'people only': no object map"}
                blobs = {"pick": (gzip.compress(b"".join(chunks), 5), {"mediaType": "application/gzip", "format": "panoptes-pick-v1 uint16 (value, run) pairs"}),
                         "depth": (gzip.compress(b"".join(depth), 5), {"mediaType": "application/gzip", "format": "panoptes-pick-v1 depth uint16 mm"})}
            return data, blobs
        return m.cpu_pool.submit(finish)

    outlines_future = m.cpu_pool.submit(outlines_job)


    # cascade: every member mask -> masked crop -> SigLIP 2 (GPU 0); object = mean of its masks
    obj_kf = sorted(by_frame)
    blobs_obj, groups = {}, {}
    maps_ready.wait()  # the outlines are a spec layer (<= 30 s); the cascade only feeds objects v2
    if objects:
        kf_index = {q: j for j, q in enumerate(obj_kf)}
        all_mem = np.concatenate(members)
        best_masks = np.array([gi for _, gi in obj_masks_on])
        with torch.inference_mode(), clock.stage("cascade.embed", gpu=dev_geo, n={"crops": int(len(all_mem)), "frames": len(obj_kf)}):
            frames_obj = kf[torch.tensor(obj_kf, device=dev_geo)]
            kf = None  # its last reader
            mt = torch.from_numpy(all_mem).to(dev_geo)
            fo = torch.tensor([kf_index[int(q)] for q in vf[all_mem]], device=dev_geo)
            emb = m.emb.crops(frames_obj, fo, voc["mask"][mt])
            owner = torch.from_numpy(np.repeat(np.arange(len(members)), [len(x) for x in members])).to(dev_geo)
            obj_emb = torch.zeros((len(members), emb.shape[1]), device=dev_geo).index_add_(0, owner, emb)
            obj_emb = torch.nn.functional.normalize(obj_emb, dim=1)
            txt = m.emb.text(words)
            probs = m.emb.zero_shot(obj_emb, txt)
            bt = torch.from_numpy(best_masks).to(dev_geo)  # a record only: the best view without the mask, for comparison
            ctx = m.emb.zero_shot(m.emb.crops(frames_obj, torch.tensor([kf_index[int(q)] for q in vf[best_masks]], device=dev_geo),
                                              voc["mask"][bt], masked=False), txt)
            m.emb.release()
            if not densify_on:
                # torch.cuda.empty_cache() empties every device's cache: called here while densify's SAM 3 ran on GPU 1 it
                # faulted that GPU (XID 31, illegal address, runs mvp-a-cards-me340-002/003, both at the second GPU 1 batch)
                with torch.cuda.device(dev_geo):
                    torch.cuda.empty_cache()
    gpu0_free.set()  # densify's GPU 0 share starts after the objects' own GPU work (section 7)
    if objects:
        with clock.stage("cascade.decide", n={"objects": len(objects)}):
            cache = cascade.LabelCache("/v/layers/label-cache/siglip2-base-p16-224-v3.npz")
            e_np, p_np, c_np = obj_emb.cpu().numpy(), probs.cpu().numpy(), ctx.cpu().numpy()
            tau, calib = cascade.calibrate(e_np, [list(objs) for objs in by_frame.values()])
            recs, unsure = cascade.decide(e_np, p_np, words, [o["word"] for o in objects], cache, video_sha, tau,
                                          lambda w: segment.is_generic(w, words), use_cache)
            groups = cascade.clusters(e_np, unsure, tau)
            for i, r in enumerate(recs):
                r["zero_shot_context_top3"] = [(words[j], round(float(c_np[i, j]), 4)) for j in np.argsort(-c_np[i])[:3]]
            buf = io.BytesIO()
            np.savez(buf, ids=np.array([o["id"] for o in objects]), embedding=e_np.astype(np.float16), probs=p_np.astype(np.float16),
                     probs_context=c_np.astype(np.float16), words=np.array(words))
            blobs_obj = {"embeddings": (buf.getvalue(), {"mediaType": "application/octet-stream", "format": "npz",
                                                         "note": "SigLIP 2 object embeddings and zero-shot probabilities, for analysis"})}
        for o, r in zip(objects, recs):
            o["cascade"] = {**{k: v for k, v in r.items() if k != "sam3"}, "source": r["source"] or "uncertain: to the VLM"}
        casc.update(crops=int(len(all_mem)), calibration=calib, cache_entries_before=len(cache), cache_used=use_cache,
                    cross_video_hits=sum(r["source"] == "cache:cross-video" for r in recs),
                    zero_shot_accepted=sum(r["source"] == "zero-shot" for r in recs), in_video_hits=sum(r["source"] == "cache:in-video" for r in recs),
                    uncertain=len(unsure), vlm_requests_objects=len(groups))
    cascade_done.set()
    if objects:
        # integration: section 4.7's options question replaces the free-text naming of the uncertain clusters (it held
        # vLLM 34 s on Sam's Club, 360 crops, before the decider could start)
        casc.update(vlm="replaced by the decider's options (click MVP 4.7)", vlm_requests_objects=0, uncertain_clusters=len(groups))
        sync_objects()
        writer.put("objects", copy.deepcopy({"objects": objects, "cascade": casc, "vocabulary": words}), blobs_obj, "estimated+inferred", obj_labels)
        clock.mark("objects_v2_put")
    esc = m.vlm_pool.submit(cards_v2) if objects else None
    release_voc()

    outlines_future.result()
    ev = vocab_future.result()
    if esc is not None:
        summary["identity"] = esc.result()
    summary["cards"] = cards_future.result()
    summary["boxes"] = box_stats(objects[:len(members)], cards_out.get("v1"))
    summary["densify"] = densify_future.result() if densify_future is not None else None
    summary["identity_densify"] = namer["densify"].result() if namer.get("densify") else None  # before the judge futures: it adds one
    summary["judge"] = [f.result() for f in judge_futures]  # after densify: it adds the v3 judgements' future
    summary["cards"] = {"v1": summary["cards"], "v3": cards_out.get("v3", {}).get("stats")}
    summary["boxes_v3"] = box_stats(objects, cards_out.get("v3"))
    summary["sam3d"] = display["models"].result()
    summary["splat"] = splat_future.result() if splat_future is not None else None
    Path(shared.result()).unlink(missing_ok=True)
    summary.update(frames=n, fps=fps, wh=[W, H], cuts=cuts, keyframes=len(keys), object_keyframes=len(range(0, len(keys), segment.OBJECT_EVERY)),
                   words=len(words), wave2_words=len(work.wave2 or []), vocab=results.get("vocab"), sam3_tasks_by_worker=work.by_worker,
                   vocab_frames_equal_decoded=[bool(img is not None and np.array_equal(img, frames[f])) for img, f in zip(seeked, vlm_frames)],
                   detections={"person": int(is_person.sum()), "floor": int((~is_person).sum()), "vocabulary_masks": voc_count,
                               "vocabulary_masks_kept": int(len(kept))},
                   objects=len(objects), cascade=casc, events_windows=len(ev), vllm_engine_stats=vlm.throughput(),
                   cut_chunks={"submitted_s": chunk_at, "done_s": [chunk_done.get(i) for i in range(len(futures))],
                               "work_s": [round(f.result()["s"], 3) for f in futures]})
    return summary


def self_check():
    """Chunked cut measure == sequential (E9's check, with this CHUNK); packing; ribbons."""
    import cv2
    rng = np.random.default_rng(0)
    tex = [cv2.GaussianBlur(rng.integers(0, 256, (1100, 1700), np.uint8), (0, 0), 2.5) for _ in range(2)]
    frames = [cv2.cvtColor(cv2.resize(tex[i >= 30][40 + 3 * (i % 30): 760 + 3 * (i % 30), 60 + 5 * (i % 30): 1340 + 5 * (i % 30)], (1280, 720)),
                           cv2.COLOR_GRAY2BGR) for i in range(55)]
    gray = [raster_gray(f) for f in frames]
    whole = dsc.measure(gray)
    for chunk in (8, CHUNK, 64):
        parts = [measure_chunk(gray[max(0, a - 2):min(len(frames), a + chunk + dsc.SPAN + 1)], a, min(a + chunk, len(frames)), max(0, a - 2),
                               min(len(frames), a + chunk + dsc.SPAN + 1)) for a in range(0, len(frames), chunk)]
        mm = stitch(parts)
        for k in ("keypoints", "inliers", "jump", "spans"):
            assert np.array_equal(mm[k], whole[k], equal_nan=True), (chunk, k)
    assert 30 in cuts_from(whole, len(frames))["cuts"]
    v, f, c = ribbon([[0, 0, 0], [1, 0, 0], [2, 0, 0]], [0, 1, 0])
    assert v.shape == (6, 3) and f.shape == (4, 3) and np.allclose(v[:, 1], .01)
    raw, meta = points_glb(np.zeros((3, 3)), np.zeros((3, 3), np.uint8), .03)
    assert raw[:4] == b"glTF" and len(raw) % 4 == 0 and meta["pointSizeNative"] == .03
    try:
        import torch
    except ImportError:
        print(f"core self-check ok: chunked cut measure == sequential (chunks 8/{CHUNK}/64), ribbon, GLB points (point helpers skipped: no torch)")
        return
    # the cards' point hand-off: components by label (a sentinel label is skipped), per-view pixels and edge flags
    p = {"world": torch.rand(10, 3), "mid": torch.tensor([0, 0, 1, 1, 1, 2, 2, 2, 2, 2]), "z": torch.rand(10), "pixels": torch.tensor([5, 6, 7]),
         "border": torch.tensor([[1, 0, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0]], dtype=torch.bool), "label": torch.tensor([4, 9, 2]),
         "edge": {"top": (torch.tensor([0, 2, 2]), torch.rand(3, 3)), "bottom": (torch.tensor([1]), torch.rand(1, 3))}}
    out = object_points(p, torch.tensor([0, 1, 1]), np.array([2, 4]))
    assert [len(o["world"]) for o in out] == [5, 2] and out[0]["views"] == {1: [7, False, True, False, False]}
    assert [len(o["top_world"]) for o in out] == [2, 1] and [len(o["bottom_world"]) for o in out] == [0, 0] and out[0]["top_frame"].tolist() == [1, 1]
    u, owner = object_voxels((torch.tensor([0, 0, 1, 2, 2]), torch.tensor([10, 11, 11, 30, 31])), np.array([0, 2]), 7)
    assert u.tolist() == [10, 11, 30, 31] and owner.tolist() == [7, 7, 8, 8]  # component 1 is not an object
    mp = merge_points(out, cap=4)
    assert len(mp["world"]) == 4 and mp["views"][1][0] == 7 and mp["views"][0][1] is True and len(mp["top_world"]) == 3
    print(f"core self-check ok: chunked cut measure == sequential (chunks 8/{CHUNK}/64), ribbon, GLB points, object points, voxel owners")
