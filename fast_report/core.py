"""Fast report core (E9's resident pipeline, split into stages): decode, shot cuts, 5 fps keyframes, DA3 per shot,
floor-plane scale, TSDF + points, people (ehs_spatial.live_people.PeopleLoop), and analyse(): the one run that ties
every stage together and hands each layer to the writer the moment it exists.

Everything heavy is imported inside functions: the cut workers are spawned processes that import this module.
"""
import copy
import hashlib
import io
import json
import struct
import threading
import time
from pathlib import Path

import numpy as np

import detect_shot_cuts as dsc  # cut rules, unchanged

BLOCK, MIN_SHOT, CHUNK, TAIL = 6, 30, 16, 8  # E9: sharpest of each 6-frame block; shots under 1 s get no geometry
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
            "inliers": int(len(keep)), "camera_height_spread_rel": float((h - h.median()).abs().median() / h.median().abs())}


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


def people_shot(si, keys, fps, g, depth_m, c2w_m, masks, plane, mpu):
    """PeopleLoop (the live rules, video.judge_frame) over the shot's 5 fps keyframes: tracks and rule rows."""
    from ehs_spatial.live_people import PeopleLoop
    if plane:
        up, p0 = plane["normal"].cpu().numpy().astype(float), plane["point"].cpu().numpy().astype(float) * mpu
        scale = {"status": "model_estimated", "nativeToMeters": mpu,
                 "anchor": {"kind": "stated_carry_height", "metres": CAMERA_HEIGHT_M, "measured": False}}
    else:
        up, p0, scale = np.array([0., -1., 0.]), np.zeros(3), {"status": "uncalibrated"}
    rgb = (g["colors"] * 255).clamp(0, 255).byte().cpu().numpy()
    depth = depth_m.cpu().numpy()
    K, c2w = g["K"].cpu().numpy().astype(float), c2w_m.cpu().numpy().astype(float)

    def detector(frame):
        return [{"label": "person", "source": f"sam3-person-{j}", "score": s, "mask": m} for m, s, j in masks.get(frame["local"], [])]
    loop = PeopleLoop(p0, up, scale, detector, None, world_epoch=si)
    rows, findings = [], []
    for j, f in enumerate(keys):
        r, fnd = loop.step({"t": f / fps, "frame": int(f), "local": j, "streamGap": None, "rgb": rgb[j], "depth": depth[j], "K": K[j],
                            "cameraToWorld": c2w[j], "trackingState": "normal", "trackingStateReason": None, "worldOriginEpoch": si})
        rows += r
        findings += fnd
    tracks = {}
    for r in rows:
        if r["track"] and r["centroidWorld"]:
            c = np.asarray(r["centroidWorld"])
            ground = c - ((c - p0) @ up) * up
            tracks.setdefault(r["track"], []).append({"t": r["t"], "frame": r["frame"], "xyz": np.round(ground, 3).tolist(),
                                                      "foot": r["footWorld"], "accepted_foot": r["accepted"]})
    return tracks, rows, findings, up


# ---------- the run ----------

def sha256(b):
    return hashlib.sha256(b).hexdigest()


def analyse(m, mp4, opts, clock, writer, log):
    """m: the resident models (FastReport): sams, da3, emb, devs, pools. Every layer goes to `writer` as soon as it
    exists. Returns the run summary; layers are the product."""
    import cv2
    import torch
    import torch.nn.functional as F
    import video_events
    from fast_report import cascade, segment, vlm
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
    writer.put("video", {"fps": fps, "frames": n_total, "wh": [W, H], "sha256": video_sha, "window_s": opts.get("window_s")},
               {"mp4": (mp4, {"mediaType": "video/mp4"})}, "observed", ["the uploaded video"])
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
        writer.put("events", {"windows": [{"t0": e["t0"], "t1": e["t1"], **(e["parsed"] if isinstance(e["parsed"], dict) else {"caption": None, "events": []}),
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
        gray_all = [g.result()[0] for g in grays]
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
    if dev_geo != dev_seg:
        work.worker(dev_geo, "geo", until=work.person_ready)
    work.wait(work.person_ready)
    with torch.inference_mode(), clock.stage("gather.person_floor", gpu=dev_geo):
        person = work.gathered("person")
        is_person = person["word"] == 0
        dyn = F.max_pool2d(union_by_frame(person["frame"][is_person], person["mask"][is_person], len(keys))[:, None].float(), 5, 1, 2)[:, 0] > 0
        floor = union_by_frame(person["frame"][~is_person], person["mask"][~is_person], len(keys))

    cam_rows, room_rows, room_blobs, people_rows, people_blobs, geo = [], [], {}, [], {}, []
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
            floor_rec.update(normal=plane["normal"].tolist(), point_m=(plane["point"] * mpu).tolist())
        cam_rows.append({"index": si, "frame_id": f"shot-{si}", "frames": list(shots[si]), "keyframes": [keys[q] for q in pos],
                         "times_s": [round(keys[q] / fps, 4) for q in pos], "c2w": c2w_m.cpu().numpy().round(5).tolist(),
                         "K_grid": g["K"].cpu().numpy().round(3).tolist(), "grid_wh": [DA3_HW[1], DA3_HW[0]], "source_wh": [W, H],
                         "scale": {"metres_per_unit": mpu, "status": "estimated" if plane else "uncalibrated",
                                   "source": "floor plane (SAM 3 'floor') + assumed camera height 1.6 m" if plane else "none: unit scale",
                                   "floor": floor_rec}})
        geo.append({"depth_m": depth_m, "c2w_m": c2w_m, "K": g["K"], "mpu": mpu, "pos": pos, "plane": plane})
    writer.put("cameras", {"shots": cam_rows, "fps": fps, "keyframe_rule": "sharpest of each 6-frame block (5 fps)"}, None, "estimated", lab)
    clock.mark("cameras_put")
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
            raw, meta = pack_mesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles), np.asarray(mesh.vertex_colors), dev_geo)
            room_blobs[f"mesh-{si}"] = (raw, {**meta, "frame": f"shot-{si}"})
            raw, meta = points_glb(tsdf["points"], tsdf["colors"], .03)
            room_blobs[f"points-{si}"] = (raw, {**meta, "frame": f"shot-{si}"})
        room_rows.append({"frame_id": f"shot-{si}", "triangles": tsdf["n_triangles"], "points": tsdf["n_points"], "tsdf_voxel_m": .03,
                          "mesh": f"mesh-{si}", "points_blob": f"points-{si}"})
    writer.put("room", {"shots": room_rows}, room_blobs, "estimated", lab)
    clock.mark("room_put")
    def people_layer():  # CPU work: runs beside GPU 0's share of the SAM 3 queue
        for si, (gg, g) in enumerate(zip(geo, shots_gpu)):
            pos = gg["pos"]
            with clock.stage(f"people.shot{si}", n={"keyframes": len(pos)}):
                local = {q: j for j, q in enumerate(pos)}
                pm = person_masks(person, local)
                tracks, rows, findings, up = people_shot(si, [keys[q] for q in pos], fps, g, gg["depth_m"], gg["c2w_m"], pm, gg["plane"], gg["mpu"])
                for tid, pts in tracks.items():
                    rb = ribbon([q["xyz"] for q in pts], up)
                    if rb is not None:
                        raw, meta = pack_mesh(*rb, dev_geo)
                        people_blobs[f"track-{si}-{tid}"] = (raw, {**meta, "frame": f"shot-{si}"})
            people_rows.append({"frame_id": f"shot-{si}", "tracks": [{"id": f"{si}-{t}", "points": pts, "detections": len(pts), "ribbon": f"track-{si}-{t}"}
                                                                      for t, pts in tracks.items()],
                                "rules": findings, "detections": len(rows)})
        writer.put("people", {"shots": people_rows, "note": "the fast path tracks people only: no non-person movers"}, people_blobs,
                   "observed+estimated", [*lab, "rules that need metres say NEEDS_REVIEW: the scale is not measured"])
        clock.mark("geometry_layers_put")

    people_future = m.cpu_pool.submit(people_layer)

    if dev_geo != dev_seg:
        work.worker(dev_geo, "geo")
    work.wait(work.all_ready)
    seg_future.result()
    people_future.result()
    clock.mark("sam3_done")
    words = work.words
    work.cache.clear()

    # ---------- objects: flood handling, lift, naming, cascade ----------
    with torch.inference_mode(), clock.stage("dedupe", gpu=dev_geo):
        voc = work.gathered("vocab")
        kept, votes = segment.dedupe(voc["frame"], voc["word"], voc["score"], voc["mask"]) if voc is not None else (np.zeros(0, int), [])
        vf = voc["frame"].cpu().numpy() if voc is not None else np.zeros(0, int)
    objects, members, obj_masks_on = [], [], []
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
            comp, arr, stats = segment.lift(voc["mask"][sel], local[voc["frame"][sel]], gg["depth_m"], gg["K"], gg["c2w_m"], dyn[torch.tensor(gg["pos"], device=dev_geo)])
            lift_stats.append(stats)
            if arr is None:
                continue
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
                                "centroid_m": arr["centroid"][c].round(3).tolist(), "box_min_m": (arr["lo"][c] - segment.LIFT_VOXEL / 2).round(3).tolist(),
                                "box_max_m": (arr["hi"][c] + segment.LIFT_VOXEL / 2).round(3).tolist(), "best_key": int(keys[fr_np[best]])})
                members.append(sel_np[mem])
                obj_masks_on.append((si, int(sel_np[best])))
        summary["lift"] = lift_stats
    clock.mark("objects_lifted")

    by_frame = {}  # object keyframe -> {object: [its masks there]}
    for oi, mem in enumerate(members):
        for gi in mem:
            by_frame.setdefault(int(vf[gi]), {}).setdefault(oi, []).append(int(gi))

    labels_v1 = threading.Event()

    def outlines_job():
        """Object keyframes 'segmented' (SAM 3 logits), other 5 fps keyframes 'projected' (E6b pair); beside the cascade."""
        with torch.inference_mode(), clock.stage("outlines", gpu=dev_geo):
            maps = []  # (entry, label map int16 numpy (0 = nothing, i + 1 = object i), sx, sy): polygons in the process pool
            oh, ow = H // 2, W // 2
            for q, objs in sorted(by_frame.items()):
                ids = list(objs)
                lg = torch.stack([voc["logits"][torch.tensor(objs[o], device=dev_geo)].float().amax(0) for o in ids])
                lab = segment.paint(F.interpolate(lg[None], size=(oh, ow), mode="bilinear", align_corners=False)[0] > 0)
                lab = torch.cat([torch.zeros(1, dtype=torch.long, device=dev_geo), torch.tensor(ids, device=dev_geo) + 1])[lab]
                maps.append(({"timeSec": round(keys[q] / fps, 4), "sourceFrame": int(keys[q]), "source": "segmented"}, lab.short().cpu().numpy(), W / ow, H / oh))
            for si, gg in enumerate(geo):
                pos = gg["pos"]
                obj_local = [j for j, q in enumerate(pos) if q in by_frame]
                if not obj_local:
                    continue
                items = {j: [(o, voc["mask"][torch.tensor(g_, device=dev_geo)].any(0)) for o, g_ in by_frame[pos[j]].items() if objects[o]["shot"] == si]
                         for j in obj_local}
                idm = segment.id_maps(items, len(pos), dev_geo)
                people_local = dyn[torch.tensor(pos, device=dev_geo)]
                for j, q in enumerate(pos):
                    if q in by_frame:
                        continue
                    lab = segment.project_pair(idm, gg["depth_m"], gg["K"], gg["c2w_m"], obj_local, j)
                    lab[people_local[j]] = -1  # people are not projected (E6b caveat): cut out with this frame's SAM 3 person mask
                    maps.append(({"timeSec": round(keys[q] / fps, 4), "sourceFrame": int(keys[q]), "source": "projected"},
                                 lab.clamp(min=0).short().cpu().numpy(), W / DA3_HW[1], H / DA3_HW[0]))
            polys = list(m.proc_pool.map(segment.label_polygons, [x[1] for x in maps], [x[2] for x in maps], [x[3] for x in maps]))
            frames_out = []
            labels_v1.wait(300)  # the polygons are ready; names as of the first objects version
            for (entry, _, _, _), found in zip(maps, polys):
                entry["objects"] = [{"entityId": objects[v - 1]["id"], "label": v1_labels[objects[v - 1]["id"]], "polygons": poly, "source": entry["source"]}
                                    for v, poly in sorted(found.items())]
                frames_out.append(entry)
            frames_out.sort(key=lambda e: e["timeSec"])
            for e, nxt in zip(frames_out, frames_out[1:] + [None]):
                e["endTimeSec"] = nxt["timeSec"] if nxt else round(e["timeSec"] + BLOCK / fps, 4)
            analysis = json.dumps({"width": W, "height": H, "frames": frames_out}, separators=(",", ":")).encode()
        writer.put("outlines", {"analysis": "blob", "frames": len(frames_out), "segmented": sum(e["source"] == "segmented" for e in frames_out),
                                "projected": sum(e["source"] == "projected" for e in frames_out)},
                   {"analysis": (analysis, {"mediaType": "application/json", "format": "video-analysis"})}, "observed(segmented)/estimated(projected)",
                   ["'segmented' outlines are SAM 3 masks on keyframes; 'projected' ones are carried from 3D (E6b 'pair'): no accuracy claimed",
                    "labels as of the first objects version; the objects layer holds the latest"])
        clock.mark("outlines_put")

    outlines_future = m.cpu_pool.submit(outlines_job)

    # cascade: every member mask -> masked crop -> SigLIP 2 (GPU 0); object = mean of its masks
    casc = {"objects": len(objects)}
    obj_kf = sorted(by_frame)
    blobs_obj = {}
    if objects:
        kf_index = {q: j for j, q in enumerate(obj_kf)}
        all_mem = np.concatenate(members)
        best_masks = np.array([gi for _, gi in obj_masks_on])
        with torch.inference_mode(), clock.stage("cascade.embed", gpu=dev_geo, n={"crops": int(len(all_mem)), "frames": len(obj_kf)}):
            frames_obj = kf[torch.tensor(obj_kf, device=dev_geo)]
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
            o.update(label=r["label"] or o["word"], label_source=r["source"] or "sam3-vote (awaiting VLM)",
                     cascade={k: v for k, v in r.items() if k not in ("label", "source", "sam3")})
        casc.update(crops=int(len(all_mem)), calibration=calib, cache_entries_before=len(cache), cache_used=use_cache,
                    cross_video_hits=sum(r["source"] == "cache:cross-video" for r in recs),
                    zero_shot_accepted=sum(r["source"] == "zero-shot" for r in recs), in_video_hits=sum(r["source"] == "cache:in-video" for r in recs),
                    uncertain=len(unsure), vlm_requests_objects=len(groups))
    for o in objects:
        o["status"] = "estimated box; name is a model output, unverified"
    obj_labels = ["object names come from detectors and models: unverified", SCALE_LABEL]
    writer.put("objects", copy.deepcopy({"objects": objects, "cascade": casc, "words": words}), blobs_obj, "estimated+inferred", obj_labels)
    clock.mark("objects_v1_put")
    v1_labels = {o["id"]: o["label"] for o in objects}
    labels_v1.set()

    def vlm_crop(gi):
        """The object's best view, full resolution: SAM 3's own mask outlined in red, 1.5 x its box, long side 448 px."""
        q = int(vf[gi])
        img = frames[keys[q]].copy()
        mk = (F.interpolate(voc["logits"][gi][None, None].float(), size=(H, W), mode="bilinear", align_corners=False)[0, 0] > 0).cpu().numpy()
        if not mk.any():
            mk = cv2.resize(voc["mask"][gi].cpu().numpy().astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
        ys, xs = np.nonzero(mk)
        cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2
        half = max(ys.max() - ys.min(), xs.max() - xs.min(), 128) * .75
        y0, y1, x0, x1 = int(max(0, cy - half)), int(min(H, cy + half)), int(max(0, cx - half)), int(min(W, cx + half))
        contours, _ = cv2.findContours(mk.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, (0, 0, 255), 2)
        crop = img[y0:y1, x0:x1]
        sc = 448 / max(crop.shape[:2])
        crop = cv2.resize(crop, (max(1, int(crop.shape[1] * sc)), max(1, int(crop.shape[0] * sc))), interpolation=cv2.INTER_AREA if sc < 1 else cv2.INTER_CUBIC)
        return cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()

    # VLM for what the cascade could not settle: one crop per cluster, in the background while outlines are drawn
    def escalate():
        reps = list(groups)
        with clock.stage("vlm.name.crops", n={"crops": len(reps)}):
            items = list(m.cpu_pool.map(lambda r: vlm_crop(obj_masks_on[r][1]), reps))
        with clock.stage("vlm.name", n={"crops": len(items)}):
            names, rec = vlm.name_crops(items)
        answered = []
        for r, nm in zip(reps, names):
            for i in groups[r]:
                o = objects[i]
                if nm and nm != "none":
                    o.update(label=nm, label_source="vlm" if i == r else "cache:in-video(vlm)")
                else:
                    o.update(label=o["word"], label_source="vlm:none" if nm == "none" else "vlm:no answer")
                o["cascade"]["vlm_answer"] = nm
            if nm and nm != "none":
                answered.append(r)
        if use_cache and answered:
            cache.add(e_np[answered], [objects[r]["label"] for r in answered], video_sha, site, "vlm")
            cache.save()
        casc.update(vlm={k: v for k, v in rec.items() if k != "texts"}, vlm_answered=len(answered), cache_entries_after=len(cache))
        writer.put("objects", copy.deepcopy({"objects": objects, "cascade": casc, "words": words}), blobs_obj, "estimated+inferred", obj_labels)
        clock.mark("objects_v2_put")
    esc = m.vlm_pool.submit(escalate) if objects and groups else None

    # hand-off for the SAM 3D / splat builders (in process, GPU tensors): Shot and Obj as FAST-BUILD-SPEC section 12
    m.last = {"shots": [{"index": si, "frames": shots[si], "keys": [keys[q] for q in gg["pos"]], "depth_m": gg["depth_m"], "K": gg["K"],
                         "c2w_m": gg["c2w_m"], "colors": shots_gpu[si]["colors"], "person": dyn[torch.tensor(gg["pos"], device=dev_geo)],
                         "mpu": gg["mpu"], "scale_status": "estimated"} for si, gg in enumerate(geo)],
              "objs": [{**o, "mask_logits_lr": voc["logits"][obj_masks_on[i][1]]} for i, o in enumerate(objects)], "frames": frames}
    outlines_future.result()
    ev = vocab_future.result()
    if esc is not None:
        esc.result()
    summary.update(frames=n, fps=fps, wh=[W, H], cuts=cuts, keyframes=len(keys), object_keyframes=len(range(0, len(keys), segment.OBJECT_EVERY)),
                   words=len(words), wave2_words=len(work.wave2 or []), vocab=results.get("vocab"), sam3_tasks_by_worker=work.by_worker,
                   vocab_frames_equal_decoded=[bool(img is not None and np.array_equal(img, frames[f])) for img, f in zip(seeked, vlm_frames)],
                   detections={"person": int(is_person.sum()), "floor": int((~is_person).sum()), "vocabulary_masks": int(len(voc["frame"])) if voc else 0,
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
    print(f"core self-check ok: chunked cut measure == sequential (chunks 8/{CHUNK}/64), ribbon, GLB points")
