"""Live map core for a moving phone that streams posed RGB-D (the future ARKit app): the stream contract, a 1x replay of an
ARKitScenes raw capture acting as that phone, and a worker that keeps an incremental TSDF and emits a voxel patch about
once a second. The device's metric poses and LiDAR depth are used as they come: no SLAM, no model, no scale fit.

Stream contract, one message per ARFrame:
    u32 little-endian header length | header JSON | rgb | depth | confidence      (blob sizes are header["sizes"])
    header  seq; t_capture: wall-clock epoch seconds at capture (phone NTP-synced); t_device: ARKit timestamp;
            K: [fx, fy, cx, cy] of the 640x480 rgb raster; cameraToWorld: 16 floats row-major, metres, OpenCV axes
            (x right, y down, z forward: ARKit's camera.transform @ diag(1, -1, -1, 1))
    rgb         640x480 JPEG. May be empty: this public capture kept wide images for only 258 of its 7.5k ARFrames; the app always sends it.
    depth       optional PNG uint16 millimetres, aligned with rgb (same field of view, lower resolution, K scaled by width)
    confidence  optional PNG uint8 0/1/2 (ARConfidenceLevel), same size as depth
The worker takes the newest waiting message and drops the older ones (latest frame wins); each dropped run is a coverage gap.

Patch, patches/NNNNNN.npz: every block touched since the previous patch, whole. blocks int32 (n, 3); tsdf int8 (n, 8, 8, 8)
indexed [z, y, x], = round(127 * tsdf / truncation); weight uint16 (n, 8, 8, 8); last_observed float64 (n,), the t_capture
of the newest frame that touched the block; seq, voxel_size, t_capture_newest. Voxel i of block b sits at (8 b + i) *
voxel_size in the world. A consumer keeps the newest copy of each block.

    python scripts/live_map.py --raw DIR --output NEW_DIR [--offline-mesh MESH --offline-cameras RUN] [--max-blocks N]
    python scripts/live_map.py --compare RUN_DIR --offline-mesh MESH [--offline-cameras RUN]
    python scripts/live_map.py --self-check
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import json
import multiprocessing as mp
from multiprocessing.connection import Client, Listener
from pathlib import Path
import resource
import struct
import subprocess
from tempfile import TemporaryDirectory
import time
import zipfile

import cv2
import numpy as np

VOXEL = .02  # m; one lowres LiDAR pixel covers 1.4 cm at 3 m
TRUNC_VOXELS = 4.
DEPTH_MAX = 5.  # m, ARKit scene depth range
MIN_CONFIDENCE = 2  # ARKit high only: medium is mostly depth edges, which smear into the map
EMIT_EVERY_S = 1.
MAX_BLOCKS = 32000  # the sliding cache: 32k blocks of float tsdf + weight are 130 MB
SPARE_BLOCKS = 2000  # Open3D grows the hash (moving every buffer index) once size + a frame's blocks pass capacity; a LiDAR frame of 47333932 touches <= 910 (median 196)
MIN_WEIGHT = 3  # frames a voxel needs before its surface counts in the extracted map (Open3D's default)


def pack(header, rgb=b"", depth=b"", confidence=b""):
    text = json.dumps(dict(header, sizes=[len(rgb), len(depth), len(confidence)])).encode()
    return struct.pack("<I", len(text)) + text + rgb + depth + confidence


def unpack(message):
    """(header, rgb, depth, confidence) of one message."""
    n = struct.unpack_from("<I", message)[0]
    header, at, blobs = json.loads(message[4:4 + n]), 4 + n, []
    for size in header["sizes"]:
        blobs.append(message[at:at + size])
        at += size
    return header, *blobs


def take_latest(conn, timeout):
    """Newest waiting frame (None if nothing came), the t_capture of each older one it replaced, and whether the stream ended."""
    latest, dropped = None, []
    while conn.poll(timeout if latest is None else 0):
        message = conn.recv_bytes()
        if not message:
            return latest, dropped, True
        if latest is not None:
            dropped.append(unpack(latest)[0]["t_capture"])
        latest = message
    return latest, dropped, False


def stamp_of(name):
    """'wide/47333932_84111.868.png' -> '84111.868': the ARFrame time, shared by an ARFrame's files in every zip."""
    return name.rsplit("_", 1)[1].rsplit(".", 1)[0]


def arkit_frames(raw):
    """[(stamp text, t_device, cameraToWorld, K of 640x480)] for every ARFrame with LiDAR depth inside the pose trajectory."""
    from scipy.spatial.transform import Rotation, Slerp
    traj = np.loadtxt(raw / "lowres_wide.traj")
    # the trajectory runs world-to-camera: carrying LiDAR depth between views agrees only that way (runs/arkit-47333932-cameras-080)
    to_world = Rotation.from_rotvec(traj[:, 1:4]).inv()
    centres = -to_world.apply(traj[:, 4:7])
    slerp = Slerp(traj[:, 0], to_world)
    with zipfile.ZipFile(raw / "wide_intrinsics.zip") as pins, zipfile.ZipFile(raw / "lowres_depth.zip") as depth:
        pin = {float(stamp_of(n)): list(map(float, pins.read(n).split()))[2:] for n in pins.namelist() if n.endswith(".pincam")}
        stamps = sorted((stamp_of(n) for n in depth.namelist() if n.endswith(".png")), key=float)
    pin_times = np.array(sorted(pin))
    frames = []
    for text in stamps:
        t = float(text)
        if not traj[0, 0] <= t <= traj[-1, 0]:
            continue
        c2w = np.eye(4)
        c2w[:3, :3] = slerp(t).as_matrix()
        c2w[:3, 3] = [np.interp(t, traj[:, 0], centres[:, axis]) for axis in range(3)]
        k = pin[float(pin_times[np.abs(pin_times - t).argmin()])]  # autofocus moves f by ~1%: the nearest frame's own K
        frames.append((text, t, c2w, [v * 640 / 1920 for v in k]))
    return frames


def produce(raw, address, authkey, output):
    """The phone: sends every ARFrame at its own time (1x), stamped with the wall clock at that instant."""
    frames = arkit_frames(raw)

    def jpeg(name):
        with zipfile.ZipFile(raw / "wide.zip") as wide:
            image = cv2.imdecode(np.frombuffer(wide.read(name), np.uint8), cv2.IMREAD_COLOR)
        return cv2.imencode(".jpg", cv2.resize(image, (640, 480), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()

    with zipfile.ZipFile(raw / "wide.zip") as wide:
        names = [n for n in wide.namelist() if n.endswith(".png")]
    with ThreadPoolExecutor(4) as pool:  # the phone encodes in hardware as it captures; here it is done before the clock starts
        jpegs = dict(zip(map(stamp_of, names), pool.map(jpeg, names)))
    prefix = names[0].split("/")[1].split("_")[0]
    late, sent_bytes = [], 0
    with zipfile.ZipFile(raw / "lowres_depth.zip") as depth, zipfile.ZipFile(raw / "confidence.zip") as confidence, \
            Client(address, authkey=authkey) as conn:
        start = time.time() + .1
        for seq, (text, t_device, c2w, k) in enumerate(frames):
            t_capture = start + t_device - frames[0][1]
            time.sleep(max(0., t_capture - time.time()))
            message = pack({"seq": seq, "t_capture": t_capture, "t_device": t_device, "K": k, "cameraToWorld": c2w.ravel().tolist()},
                           jpegs.get(text, b""), depth.read(f"lowres_depth/{prefix}_{text}.png"), confidence.read(f"confidence/{prefix}_{text}.png"))
            conn.send_bytes(message)
            late.append(time.time() - t_capture)
            sent_bytes += len(message)
        conn.send_bytes(b"")
    span = frames[-1][1] - frames[0][1]
    (output / "producer.json").write_text(json.dumps({
        "frames_sent": len(frames), "frames_with_rgb": sum(f[0] in jpegs for f in frames), "capture_s": span, "stream_start": start,
        "input_fps": (len(frames) - 1) / span, "stream_mbit_s": sent_bytes * 8 / span / 1e6,
        "send_done_after_capture_s_p50_p95_max": [float(np.percentile(late, 50)), float(np.percentile(late, 95)), float(max(late))]}, indent=1))


class LiveMap:
    """Incremental TSDF on a bounded hash of 8^3 voxel blocks; each patch carries every block touched since the last one."""

    def __init__(self, output, max_blocks=MAX_BLOCKS, voxel=VOXEL):
        import open3d as o3d
        import open3d.core as o3c
        self.o3d, self.o3c = o3d, o3c
        self.grid = o3d.t.geometry.VoxelBlockGrid(("tsdf", "weight"), (o3c.float32, o3c.float32), (1, 1), voxel, 8,
                                                  max_blocks + SPARE_BLOCKS, o3c.Device("CPU:0"))
        self.capacity = self.grid.hashmap().capacity()
        self.last_observed, self.dirty = np.zeros(self.capacity), np.zeros(self.capacity, bool)
        self.output, self.max_blocks, self.voxel = output, max_blocks, voxel
        self.patch_of = np.zeros(self.capacity, int)  # the patch holding each cached block's last emitted state
        self.cold = {}  # evicted block -> its patch. ponytail: ~150 B a block, bounded by the map not the cache; an on-disk index when walks reach millions of blocks
        self.patches, self.evicted, self.restored, self.peak_blocks, self.newest = [], 0, 0, 0, None

    def integrate(self, depth_mm, k, c2w, t_capture):
        """One frame of uint16 millimetre depth with its own K (fx fy cx cy at depth resolution) and camera-to-world pose.
        False when no pixel has usable depth (Open3D refuses such a frame)."""
        if not ((depth_mm > 0) & (depth_mm <= DEPTH_MAX * 1000)).any():
            return False
        o3c = self.o3c
        image = self.o3d.t.geometry.Image(o3c.Tensor(np.ascontiguousarray(depth_mm)))
        # Open3D reads the depth pixel at floor(projection); +0.5 makes it the nearest one (synthetic box: median error 8.8 mm -> 0)
        intrinsic = o3c.Tensor(np.array([[k[0], 0, k[2] + .5], [0, k[1], k[3] + .5], [0, 0, 1.]]))
        extrinsic = o3c.Tensor(np.linalg.inv(c2w))
        blocks = self.grid.compute_unique_block_coordinates(image, intrinsic, extrinsic, 1000., DEPTH_MAX, TRUNC_VOXELS)
        self._make_room(blocks)
        self.grid.integrate(blocks, image, intrinsic, extrinsic, 1000., DEPTH_MAX, TRUNC_VOXELS)
        hashmap = self.grid.hashmap()
        assert hashmap.capacity() == self.capacity, "block hash grew: buffer indices moved"
        index = hashmap.find(blocks)[0].numpy()
        self.last_observed[index], self.dirty[index], self.newest = t_capture, True, t_capture
        self.peak_blocks = max(self.peak_blocks, hashmap.size())
        return True

    def _make_room(self, incoming):
        """Keeps the cache at max_blocks: the longest-unseen clean blocks leave (their state is in a patch already), and an
        evicted block seen again comes back from its patch before integration, so a revisit never restarts it from nothing."""
        hashmap, o3c = self.grid.hashmap(), self.o3c
        index, found = (t.numpy() for t in hashmap.find(incoming))
        excess = hashmap.size() + int((~found).sum()) - self.max_blocks
        if excess > 0:
            active = hashmap.active_buf_indices().numpy()
            clean = np.setdiff1d(active[~self.dirty[active]], index[found])
            if len(clean) < excess:
                self.emit()
                clean = np.setdiff1d(active, index[found])
            assert len(clean) >= excess, "one frame needs more blocks than the cache holds"
            gone = clean[np.argsort(self.last_observed[clean])[:excess]]
            keys = hashmap.key_tensor().numpy()[gone].copy()
            self.cold.update(zip(map(tuple, keys.tolist()), self.patch_of[gone].tolist()))
            for name in ("tsdf", "weight"):  # a reused buffer keeps its old values (checked): clear them
                self.grid.attribute(name).numpy()[gone] = 0
            hashmap.erase(o3c.Tensor(keys))
            self.evicted += excess
        back = [(key, self.cold.pop(key)) for key in map(tuple, incoming.numpy()[~found].tolist()) if key in self.cold]
        if back:
            buffers = hashmap.activate(o3c.Tensor(np.array([key for key, _ in back], np.int32)))[0].numpy()
            tsdf, weight = self.grid.attribute("tsdf").numpy(), self.grid.attribute("weight").numpy()
            for (key, seq), i in zip(back, buffers):
                t, w, seen = read_patch(self.output / "patches" / f"{seq:06d}.npz")[key]
                tsdf[i, ..., 0], weight[i, ..., 0], self.last_observed[i], self.patch_of[i] = t / 127, w, seen, seq
            self.restored += len(back)

    def blocks(self, index):
        """Patch arrays of the given buffer indices."""
        return {"blocks": self.grid.hashmap().key_tensor().numpy()[index].astype(np.int32),
                "tsdf": np.round(self.grid.attribute("tsdf").numpy()[index, ..., 0] * 127).astype(np.int8),
                "weight": np.minimum(self.grid.attribute("weight").numpy()[index, ..., 0], 65535).astype(np.uint16),
                "last_observed": self.last_observed[index]}

    def emit(self):
        """Writes the touched blocks as the next patch (atomically: a reader never sees half a file).
        ponytail: written on the integration thread, which drops the frames that arrive meanwhile; a writer thread if that matters."""
        index = np.flatnonzero(self.dirty)
        if not len(index):
            return
        t0, seq = time.perf_counter(), len(self.patches) + 1
        path = self.output / "patches" / f"{seq:06d}.npz"
        with open(path.with_suffix(".tmp"), "wb") as file:
            np.savez_compressed(file, seq=seq, voxel_size=self.voxel, t_capture_newest=self.newest, **self.blocks(index))
        path.with_suffix(".tmp").replace(path)
        self.dirty[index], self.patch_of[index] = False, seq
        t_emit = time.time()
        self.patches.append({"seq": seq, "t_emit": t_emit, "t_capture_newest": self.newest, "latency_s": t_emit - self.newest,
                             "blocks": len(index), "bytes": path.stat().st_size, "active_blocks": self.grid.hashmap().size(),
                             "write_ms": (time.perf_counter() - t0) * 1000})


def serve(output, max_blocks, authkey, ready):
    """The worker process: newest frame wins, integrate, a patch a second, stats when the stream ends."""
    live, integrate_ms, gaps, integrated, without_depth, first = LiveMap(output, max_blocks), [], [], 0, 0, None
    with Listener(("127.0.0.1", 0), authkey=authkey) as listener:
        ready.put(listener.address)
        with listener.accept() as conn:
            ended = False
            while not ended:
                message, dropped, ended = take_latest(conn, .05)
                if dropped:
                    gaps.append([dropped[0], dropped[-1], len(dropped)])
                if message is not None:
                    t0 = time.perf_counter()
                    header, _, depth, confidence = unpack(message)
                    first = first or header["t_capture"]
                    used = False
                    if depth:
                        depth = cv2.imdecode(np.frombuffer(depth, np.uint8), cv2.IMREAD_UNCHANGED)
                        if confidence:
                            depth[cv2.imdecode(np.frombuffer(confidence, np.uint8), cv2.IMREAD_UNCHANGED) < MIN_CONFIDENCE] = 0
                        k = np.array(header["K"]) * depth.shape[1] / 640
                        used = live.integrate(depth, k, np.array(header["cameraToWorld"]).reshape(4, 4), header["t_capture"])
                    if used:
                        integrated += 1
                        integrate_ms.append((time.perf_counter() - t0) * 1000)
                    else:
                        without_depth += 1
                if ended or live.dirty.any() and (not live.patches or time.time() - live.patches[-1]["t_emit"] >= EMIT_EVERY_S):
                    live.emit()
    (output / "worker.json").write_text(json.dumps({
        "frames_integrated": integrated, "frames_without_usable_depth": without_depth, "first_t_capture": first, "last_t_capture": live.newest,
        "decode_integrate_ms_p50_p95_max": [float(np.percentile(integrate_ms, q)) for q in (50, 95, 100)],
        "coverage_gaps": gaps, "evicted_blocks": live.evicted, "restored_blocks": live.restored, "peak_blocks": live.peak_blocks, "max_blocks": max_blocks,
        "worker_peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20, "patches": live.patches}, indent=1))


@lru_cache(maxsize=8)
def read_patch(path):
    """{block: (tsdf int8, weight, last_observed)} of one patch file; patches never change once written."""
    with np.load(path) as patch:
        return dict(zip(map(tuple, patch["blocks"].tolist()), zip(patch["tsdf"], patch["weight"], patch["last_observed"])))


def load_patches(run):
    """The consumer's map: the newest copy of every block, in patch order."""
    blocks, voxel = {}, VOXEL
    for path in sorted((run / "patches").glob("*.npz")):
        with np.load(path) as patch:
            voxel = float(patch["voxel_size"])
            blocks.update(zip(map(tuple, patch["blocks"].tolist()), zip(patch["tsdf"], patch["weight"])))
    return blocks, voxel


def surface_points(blocks, voxel, min_weight=MIN_WEIGHT):
    """Zero crossings of the TSDF between neighbouring voxels, in metres: where marching cubes would put its vertices."""
    keys = np.array(list(blocks), np.int64).reshape(-1, 3)
    tsdf = np.array([t for t, _ in blocks.values()], np.float32).reshape(-1) / 127
    weight = np.array([w for _, w in blocks.values()]).reshape(-1)
    z, y, x = np.indices((8, 8, 8))
    voxels = (keys[:, None, None, None] * 8 + np.stack([x, y, z], -1)).reshape(-1, 3)
    keep = weight >= min_weight
    voxels, tsdf = voxels[keep], tsdf[keep]
    code = lambda v: ((v[:, 0] + 2 ** 20) << 42) | ((v[:, 1] + 2 ** 20) << 21) | (v[:, 2] + 2 ** 20)
    order = np.argsort(codes := code(voxels))
    codes, points = codes[order], []
    for axis in range(3):
        step = np.eye(3, dtype=np.int64)[axis]
        wanted = code(voxels + step)
        at = np.searchsorted(codes, wanted).clip(max=len(codes) - 1)
        a = np.flatnonzero(codes[at] == wanted)
        b = order[at[a]]
        t0, t1 = tsdf[a], tsdf[b]
        cross = ((t0 > 0) != (t1 > 0)) & (np.abs(t0) < 1) & (np.abs(t1) < 1)  # a +1 to -1 jump is a truncation edge, not a surface
        points.append((voxels[a[cross]] + (t0[cross] / (t0[cross] - t1[cross]))[:, None] * step) * voxel)
    return np.concatenate(points)


def seen_by_views(points, cameras_run, tolerance=.05):
    """Which points lie on the confident LiDAR surface of a view the offline run fused: where both maps could look.
    The offline run fused only the 258 kept wide frames (gaps up to 16 s); the live map has every ARFrame's depth."""
    clip = json.loads((cameras_run / "run.json").read_text())["clip_definition"]
    cameras = np.load(cameras_run / "prediction.npz")
    names = [line.split()[0] for line in (Path(clip["dataset"]) / "rgb.txt").read_text().splitlines() if not line.startswith("#")]
    seen = np.zeros(len(points), bool)
    for c2w, k, name in zip(cameras["poses_c2w"].astype(float), cameras["keyframe_final_fullres_intrinsics"].astype(float), names):
        depth = cv2.imread(str(Path(clip["dataset"]) / clip["evaluation_depth"] / f"{name}.png"), cv2.IMREAD_UNCHANGED) / 1000
        k = k * 2 * depth.shape[1] / 640  # stored for the 320x240 model raster
        local = (points - c2w[:3, 3]) @ c2w[:3, :3]
        z = np.maximum(local[:, 2], 1e-6)
        u, v = (np.round(local[:, axis] / z * k[axis] + k[axis + 2]).astype(int) for axis in (0, 1))
        ok = np.flatnonzero((local[:, 2] > .1) & (u >= 0) & (u < depth.shape[1]) & (v >= 0) & (v < depth.shape[0]))
        seen[ok] |= np.abs(depth[v[ok], u[ok]] - local[ok, 2]) < tolerance  # no confident depth reads 0: not seen
    return seen


def compare(points, mesh_path, cameras_run=None, near=.05):
    """Live surface against the offline fused mesh: accuracy (live to offline), completeness (offline to live), chamfer;
    with the offline run's cameras, accuracy again on the live points those cameras also saw."""
    import open3d as o3d
    from scipy.spatial import cKDTree
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    o3d.utility.random.seed(0)
    reference = np.asarray(mesh.sample_points_uniformly(int(mesh.get_surface_area() / .01 ** 2)).points)  # 1 cm spacing
    to_offline, to_live = cKDTree(reference).query(points)[0], cKDTree(points).query(reference)[0]
    stats = lambda d: {"median_m": float(np.median(d)), "p90_m": float(np.percentile(d, 90)), f"within_{near * 100:.0f}cm": float((d < near).mean())}
    result = {"offline_mesh": str(mesh_path), "live_points": len(points), "offline_points": len(reference),
              "accuracy_live_to_offline": stats(to_offline), "completeness_offline_to_live": stats(to_live),
              "chamfer_mean_m": float((to_offline.mean() + to_live.mean()) / 2)}
    if cameras_run:
        seen = seen_by_views(points, cameras_run)
        result["live_points_seen_by_offline_views"] = {"share": float(seen.mean()), "accuracy_live_to_offline": stats(to_offline[seen]),
                                                       "accuracy_of_the_rest": stats(to_offline[~seen])}
    return result


def run(args):
    """Producer and worker as two processes over a local socket, memory sampled each second, then metrics."""
    (args.output / "patches").mkdir(parents=True)
    ctx = mp.get_context("spawn")
    authkey, ready = mp.current_process().authkey, ctx.Queue()
    worker = ctx.Process(target=serve, args=(args.output, args.max_blocks, authkey, ready))
    worker.start()
    producer = ctx.Process(target=produce, args=(args.raw, ready.get(timeout=120), authkey, args.output))
    producer.start()
    memory = []
    while worker.is_alive():
        rss = [subprocess.run(["ps", "-o", "rss=", "-p", str(p.pid)], capture_output=True, text=True).stdout.strip() for p in (worker, producer)]
        memory.append([time.time()] + [int(r) / 1024 if r else None for r in rss])
        worker.join(1.)
    producer.join()
    assert worker.exitcode == 0 and producer.exitcode == 0, (worker.exitcode, producer.exitcode)
    sent, done = (json.loads((args.output / f"{n}.json").read_text()) for n in ("producer", "worker"))
    patches, gaps = done["patches"], done["coverage_gaps"]
    latency = [p["latency_s"] for p in patches]
    worker_mb = [m[1] for m in memory if m[1]]
    capture_time = [m[0] - sent["stream_start"] for m in memory if m[1]]
    metrics = {
        "machine": "local Mac CPU (worker and producer on one host, loopback socket: no network in the latency)",
        "frames_sent": sent["frames_sent"], "frames_integrated": done["frames_integrated"],
        "frames_without_usable_depth": done["frames_without_usable_depth"], "input_fps": sent["input_fps"],
        "integrated_fps": done["frames_integrated"] / sent["capture_s"], "stream_mbit_s": sent["stream_mbit_s"],
        "time_to_first_patch_s": patches[0]["t_emit"] - sent["stream_start"],
        "patch_latency_s_p50_p95_max": [float(np.percentile(latency, 50)), float(np.percentile(latency, 95)), max(latency)],
        "patches": len(patches), "patch_interval_s_median": float(np.median(np.diff([p["t_emit"] for p in patches]))),
        "patch_bytes_total": sum(p["bytes"] for p in patches), "patch_blocks_median": float(np.median([p["blocks"] for p in patches])),
        "patch_write_ms_p50_p95": [float(np.percentile([p["write_ms"] for p in patches], q)) for q in (50, 95)],
        "decode_integrate_ms_p50_p95_max": done["decode_integrate_ms_p50_p95_max"],
        "coverage_gaps": {"runs": len(gaps), "frames_dropped": sum(g[2] for g in gaps),
                          "longest_s": max((g[1] - g[0] for g in gaps), default=0.)},
        "blocks": {"peak": done["peak_blocks"], "cap": done["max_blocks"], "evicted": done["evicted_blocks"], "restored": done["restored_blocks"]},
        "worker_rss_mb": {"first_10s_max": max(m for t, m in zip(capture_time, worker_mb) if t < 10), "max": max(worker_mb),
                          "last": worker_mb[-1], "peak_rusage": done["worker_peak_rss_mb"],
                          "every_10s": [round(m) for m in worker_mb[::10]]},
        "producer_rss_mb_max": max(m[2] for m in memory if m[2]),
        "producer_send_lateness_s_p50_p95_max": sent["send_done_after_capture_s_p50_p95_max"]}
    (args.output / "memory.json").write_text(json.dumps(memory))
    if args.offline_mesh:
        metrics["against_offline"] = evaluate(args.output, args.offline_mesh, args.offline_cameras)
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=1))
    print(json.dumps(metrics, indent=1))


def evaluate(run_dir, mesh, cameras_run=None):
    """The consumer's map from the patches alone, compared with the offline mesh; the points are kept for a look."""
    import open3d as o3d
    points = surface_points(*load_patches(run_dir))
    o3d.io.write_point_cloud(str(run_dir / "live-surface.ply"), o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points)))
    return compare(points, mesh, cameras_run)


def render_box_depth(c2w, k, shape, room, cube):
    """Exact depth of a camera inside an axis-aligned room with an axis-aligned cube on its floor."""
    v, u = np.indices(shape)
    rays = np.stack([(u - k[2]) / k[0], (v - k[3]) / k[1], np.ones(shape)], -1) @ c2w[:3, :3].T  # camera z = 1: t is depth
    o = c2w[:3, 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        leave_room = np.maximum((room[0] - o) / rays, (room[1] - o) / rays).min(-1)
        a, b = (cube[0] - o) / rays, (cube[1] - o) / rays
        enter, leave = np.minimum(a, b).max(-1), np.maximum(a, b).min(-1)
    return np.where((enter <= leave) & (enter > 0), np.minimum(enter, leave_room), leave_room)


def look_at(eye, target):
    forward = (target - eye) / np.linalg.norm(target - eye)
    right = np.cross(forward, [0, 0, 1.])
    right /= np.linalg.norm(right)
    c2w = np.eye(4)
    c2w[:3, :3], c2w[:3, 3] = np.stack([right, np.cross(forward, right), forward], 1), eye
    return c2w


def self_check():
    room, cube = np.array([[0, 0, 0], [4, 3, 2.5]]), np.array([[1.6, 1.1, 0], [2.2, 1.7, .6]])
    k, shape, voxel = [100., 100., 60., 45.], (90, 120), .04
    frames = []
    for n in range(90):  # one lap around the cube, looking at it from 1.3 m and 1.5 m up
        angle = 2 * np.pi * n / 90
        c2w = look_at(np.array([1.9 + 1.3 * np.cos(angle), 1.4 + 1.3 * np.sin(angle), 1.5]), np.array([1.9, 1.4, .3]))
        frames.append((np.round(render_box_depth(c2w, k, shape, room, cube) * 1000).astype(np.uint16), c2w, n / 30))

    def truth_distance(p):
        to_room = np.minimum(p - room[0], room[1] - p).min(1)
        outside = np.linalg.norm(np.maximum(np.maximum(cube[0] - p, p - cube[1]), 0), axis=1)
        inside = np.minimum(p - cube[0], cube[1] - p).min(1).clip(min=0)
        return np.minimum(np.abs(to_room), outside + inside)

    def cube_coverage(points):
        from scipy.spatial import cKDTree
        g = np.linspace(.01, .59, 30)
        a, b = (m.ravel() for m in np.meshgrid(g, g))
        x0, y0 = cube[0, :2]
        faces = [np.c_[x0 + a, y0 + b, np.full_like(a, .6)], np.c_[np.full_like(a, 1.6), y0 + a, b], np.c_[np.full_like(a, 2.2), y0 + a, b],
                 np.c_[x0 + a, np.full_like(a, 1.1), b], np.c_[x0 + a, np.full_like(a, 1.7), b]]
        return float((cKDTree(points).query(np.concatenate(faces))[0] < voxel).mean())

    with TemporaryDirectory() as directory:
        for name in ("whole", "sliding"):
            cap = MAX_BLOCKS if name == "whole" else int(.6 * peak)  # the lap needs 40% more blocks than the sliding cache holds
            out = Path(directory) / name
            (out / "patches").mkdir(parents=True)
            live = LiveMap(out, cap, voxel)
            for depth, c2w, t in frames:
                live.integrate(depth, k, c2w, t)
                if round(t * 30) % 10 == 9:
                    live.emit()
                assert live.grid.hashmap().size() <= cap
            live.emit()
            blocks, patch_voxel = load_patches(out)
            points = surface_points(blocks, patch_voxel)
            error, coverage = truth_distance(points), cube_coverage(points)
            assert patch_voxel == voxel and np.median(error) < .05 * voxel and np.percentile(error, 95) < .25 * voxel, (name, np.percentile(error, [50, 95]))
            assert coverage > .95, (name, coverage)  # the cube's top and four sides came out whole
            if name == "whole":
                peak, whole = live.peak_blocks, blocks
                own = live.blocks(live.grid.hashmap().active_buf_indices().numpy())
                assert live.evicted == 0 and len(own["blocks"]) == len(blocks)  # the patches carry every block the worker holds, as it holds it
                for key, t, w in zip(own["blocks"].tolist(), own["tsdf"], own["weight"]):
                    assert np.array_equal(blocks[tuple(key)][0], t) and np.array_equal(blocks[tuple(key)][1], w)
            else:  # the bounded cache gives the consumer the same map: same blocks, same weights, tsdf within one step
                assert live.evicted > 0 and live.restored > 0 and live.peak_blocks <= cap < peak, (live.peak_blocks, cap, peak)
                assert blocks.keys() == whole.keys()
                for key, (t, w) in blocks.items():
                    assert np.array_equal(w, whole[key][1]) and np.abs(t.astype(int) - whole[key][0]).max() <= 1, key
            print(f"{name}: {len(points)} surface points, error median {np.median(error) * 1000:.1f} mm, p95 {np.percentile(error, 95) * 1000:.1f} mm, "
                  f"cube covered {coverage:.3f}, peak blocks {live.peak_blocks}, evicted {live.evicted}")

        # a new block in an evicted block's buffer starts empty; an evicted block that comes back keeps its history
        out = Path(directory) / "reuse"
        (out / "patches").mkdir(parents=True)
        wall, aside = np.full(shape, 2000, np.uint16), np.eye(4)
        aside[0, 3] = 9.6  # the same wall 30 blocks along: all new blocks
        probe = LiveMap(out, 5000, voxel)
        probe.integrate(wall, k, np.eye(4), 0)
        live = LiveMap(out, int(probe.peak_blocks * 1.1), voxel)
        held = lambda: dict(zip(map(tuple, live.blocks(active := live.grid.hashmap().active_buf_indices().numpy())["blocks"].tolist()), active))
        weight_of = lambda keys: live.grid.attribute("weight").numpy()[live.grid.hashmap().find(live.o3c.Tensor(np.array(sorted(keys), np.int32)))[0].numpy()]
        live.integrate(wall, k, np.eye(4), 0)
        live.integrate(wall, k, np.eye(4), 1)
        live.emit()
        first = held()
        live.integrate(wall, k, aside, 2)
        live.emit()
        second = held()
        new, back = second.keys() - first.keys(), first.keys() - second.keys()
        assert {second[key] for key in new} & set(first.values()), "no buffer was reused: the check would prove nothing"
        assert weight_of(new).max() == 1
        live.integrate(wall, k, np.eye(4), 3)
        assert back and weight_of(back).max() == 3 and live.restored == len(back), (len(back), live.restored)

    sender, receiver = mp.Pipe()
    for t in (1., 2., 3.):
        sender.send_bytes(pack({"t_capture": t}))
    latest, dropped, ended = take_latest(receiver, 0)
    assert unpack(latest)[0]["t_capture"] == 3 and dropped == [1., 2.] and not ended
    sender.send_bytes(pack({"t_capture": 4.}))
    sender.send_bytes(b"")
    latest, dropped, ended = take_latest(receiver, 0)
    assert unpack(latest)[0]["t_capture"] == 4 and dropped == [] and ended
    header, rgb, depth, confidence = unpack(pack({"seq": 7}, b"j", b"dd", b""))
    assert header["seq"] == 7 and (rgb, depth, confidence) == (b"j", b"dd", b"")
    print("live map check passed: a moving camera around a box integrates to the box and room surfaces, patches rebuild the "
          "worker's map exactly, the sliding cache stays under its cap, and the newest frame wins")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--raw", type=Path, help="ARKitScenes raw capture: wide.zip, wide_intrinsics.zip, lowres_wide.traj, lowres_depth.zip, confidence.zip")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--compare", type=Path, help="an earlier run directory to compare again")
    parser.add_argument("--offline-mesh", type=Path, help="the offline fusion of the same capture, e.g. runs/da3-posed-arkit-raw-081/mono-anchored-mesh.ply")
    parser.add_argument("--offline-cameras", type=Path, help="the offline run's cameras (prediction.npz, run.json), e.g. runs/arkit-47333932-cameras-080")
    parser.add_argument("--max-blocks", type=int, default=MAX_BLOCKS)
    a = parser.parse_args()
    if a.self_check:
        self_check()
    elif a.compare:
        print(json.dumps(evaluate(a.compare, a.offline_mesh, a.offline_cameras), indent=1))
    else:
        run(a)
