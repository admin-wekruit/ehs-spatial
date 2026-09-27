"""Live map core for a moving phone that streams posed RGB-D (the future ARKit app): a 1x replay of an ARKitScenes raw
capture acting as that phone, and a worker that keeps an incremental TSDF and emits a voxel patch about once a second.
The device's metric poses and LiDAR depth are used as they come: no SLAM, no model, no scale fit.

The phone stream contract (header keys, blobs, the one codec) is ehs_spatial/phone_stream.py. The phone sends at most
RATE_HZ of its 60 Hz frames. Only a 'normal' frame with an int worldOriginEpoch and a rigid pose is integrated; every
other frame is a coverage gap. Each epoch is its own submap, never fused with another. The worker takes the newest
waiting message and drops the older ones (latest frame wins); each dropped run is a coverage gap.

Submap output, OUTPUT/submap-EPOCH/: submap.json (epoch, voxel_size); patches/NNNNNN.npz, every block touched since the
previous patch, whole: blocks int32 (n, 3); tsdf int8 (n, 8, 8, 8) indexed [z, y, x], = round(127 * tsdf / truncation);
weight uint16 (n, 8, 8, 8); last_observed float64 (n,), the t_capture of the newest frame that touched the block (views
through it count); seq, epoch, voxel_size, t_capture_newest; extractable_t_capture, the frames whose surface first became
extractable (weight >= MIN_WEIGHT) in this patch: a consumer's map latency is its receive time minus these. Voxel i of block
b sits at (8 b + i) * voxel_size in the world. A consumer keeps the newest copy of each block. Only the last RETAIN_PATCHES
patch files stay; blocks.sqlite holds the newest state of every block ever emitted (what a late consumer starts from, and
where an evicted block comes back from), so the disk grows with the mapped area, not with time.

    python scripts/live_map.py --raw DIR --output NEW_DIR [--rate HZ] [--offline-mesh MESH --offline-cameras RUN] [--max-blocks N]
    python scripts/live_map.py --compare SUBMAP_DIR --offline-mesh MESH [--offline-cameras RUN]
    python scripts/live_map.py --self-check
"""
import argparse
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import io
import json
import multiprocessing as mp
from multiprocessing.connection import Client, Listener
from pathlib import Path
import queue
import resource
import sqlite3
import struct
import subprocess
import sys
from tempfile import TemporaryDirectory
import threading
import time
import zipfile
import zlib

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ehs_spatial.phone_stream import TRACKING_NORMAL, depth_mm, header, pack, pose, unpack  # noqa: E402  (the Modal image carries it too)

VOXEL = .02  # m; one lowres LiDAR pixel covers 1.4 cm at 3 m
TRUNC_VOXELS = 4.
DEPTH_MAX = 5.  # m, ARKit scene depth range
EMIT_EVERY_S = 1.
MAX_BLOCKS = 32000  # the sliding cache: 32k blocks of float tsdf + weight are 130 MB
SPARE_BLOCKS = 2000  # Open3D grows the hash (moving every buffer index) once size + a frame's blocks pass capacity; a LiDAR frame of 47333932 touches <= 910 (median 196)
MIN_WEIGHT = 3  # frames a voxel needs before its surface counts in the extracted map (Open3D's default)
MAX_WEIGHT = 32  # evidence cap: a box that stood for an hour still leaves after ~MAX_WEIGHT views through it (2 s at 15 Hz)
RATE_HZ = 15.  # uplink cap: 60 Hz LiDAR frames 17 ms apart add little map for 4x the bytes
RETAIN_PATCHES = 30  # patch files kept for a consumer catching up (30 s); older states live on in blocks.sqlite
WINDOW = 20000  # per-frame and per-patch samples kept (rolling): the whole 125 s replay at 60 Hz fits, a day-long stream does not grow
PENDING_S = 30.  # a frame whose surface is not extractable this long after capture counts as never
SAMPLES = 32  # surface points per frame followed for the map latency
EXTRACTABLE_SHARE = .9  # a frame's surface is in the map once this share of its sampled points is


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


class Inbox:
    """A Connection-like stream for a transport that is not one (a websocket): the receiving task sends, serve() polls.
    ponytail: unbounded, but serve() drains all of it every loop; a cap if a stalled worker must not buffer."""

    def __init__(self):
        self.items, self.ready = deque(), threading.Condition()

    def send_bytes(self, message):
        with self.ready:
            self.items.append(message)
            self.ready.notify()

    def poll(self, timeout):
        with self.ready:
            return bool(self.ready.wait_for(lambda: self.items, timeout))

    def recv_bytes(self):
        with self.ready:
            return self.items.popleft()


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


def paced(times, rate_hz):
    """Which frames the phone sends: the first of each 1/rate_hz slot of device time (every frame when rate_hz is 0)."""
    if not rate_hz:
        return np.ones(len(times), bool)
    slot = np.floor((np.asarray(times) - times[0]) * rate_hz)
    return np.r_[True, slot[1:] != slot[:-1]]


def produce(raw, send, rate_hz=RATE_HZ):
    """The phone: sends ARFrames at their own time (1x), at most rate_hz of them, stamped with the wall clock at that instant.
    ARKitScenes keeps no tracking state and its trajectory has no step over 0.12 s (M), so every frame goes as 'normal', epoch 0."""
    frames = arkit_frames(raw)
    frames = [f for f, keep in zip(frames, paced([f[1] for f in frames], rate_hz)) if keep]

    def jpeg(name):
        with zipfile.ZipFile(raw / "wide.zip") as wide:
            image = cv2.imdecode(np.frombuffer(wide.read(name), np.uint8), cv2.IMREAD_COLOR)
        return cv2.imencode(".jpg", cv2.resize(image, (640, 480), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()

    sending = {f[0] for f in frames}
    with zipfile.ZipFile(raw / "wide.zip") as wide:
        names = [n for n in wide.namelist() if n.endswith(".png")]
    prefix = names[0].split("/")[1].split("_")[0]
    names = [n for n in names if stamp_of(n) in sending]
    with ThreadPoolExecutor(4) as pool:  # the phone encodes in hardware as it captures; here it is done before the clock starts
        jpegs = dict(zip(map(stamp_of, names), pool.map(jpeg, names)))
    late, sent_bytes, depth_bytes = deque(maxlen=WINDOW), 0, 0
    with zipfile.ZipFile(raw / "lowres_depth.zip") as depth, zipfile.ZipFile(raw / "confidence.zip") as confidence:
        start = time.time() + .1
        for seq, (text, t_device, c2w, k) in enumerate(frames):
            t_capture = start + t_device - frames[0][1]
            time.sleep(max(0., t_capture - time.time()))
            png = depth.read(f"lowres_depth/{prefix}_{text}.png")
            message = pack(header(seq, t_capture, t_device, k, c2w), jpegs.get(text, b""), png, confidence.read(f"confidence/{prefix}_{text}.png"))
            send(message)
            late.append(time.time() - t_capture)
            sent_bytes, depth_bytes = sent_bytes + len(message), depth_bytes + len(png)
        send(b"")
    span = frames[-1][1] - frames[0][1]
    return {"frames_sent": len(frames), "frames_with_rgb": sum(f[0] in jpegs for f in frames), "capture_s": span, "stream_start": start,
            "rate_cap_hz": rate_hz, "sent_fps": (len(frames) - 1) / span, "uplink_mbit_s": sent_bytes * 8 / span / 1e6,
            "uplink_depth_mbit_s": depth_bytes * 8 / span / 1e6, "bytes_per_frame_mean": sent_bytes / len(frames),
            "send_done_after_capture_s_p50_p95_max": [float(np.percentile(late, 50)), float(np.percentile(late, 95)), float(max(late))]}


def percentiles(values, qs):
    """The given percentiles of a sample, None when it is empty (a submap whose one frame had no usable depth)."""
    return [float(np.percentile(values, q)) for q in qs] if len(values) else None


def as_blocks(patch):
    """{block: (tsdf, weight)} of one patch's arrays."""
    return dict(zip(map(tuple, patch["blocks"].tolist()), zip(patch["tsdf"], patch["weight"])))


class LiveMap:
    """Incremental TSDF of one submap on a bounded hash of 8^3 voxel blocks. Each patch carries every block touched since the
    last one; a writer thread saves it, folds it into the block store and hands it to the consumer, off the frame loop."""

    def __init__(self, output, max_blocks=MAX_BLOCKS, voxel=VOXEL, epoch=0, sink=None, retain=RETAIN_PATCHES):
        import open3d as o3d
        import open3d.core as o3c
        self.o3d, self.o3c = o3d, o3c
        self.grid = o3d.t.geometry.VoxelBlockGrid(("tsdf", "weight"), (o3c.float32, o3c.float32), (1, 1), voxel, 8,
                                                  max_blocks + SPARE_BLOCKS, o3c.Device("CPU:0"))
        self.capacity = self.grid.hashmap().capacity()
        self.last_observed, self.dirty = np.zeros(self.capacity), np.zeros(self.capacity, bool)
        self.output, self.max_blocks, self.voxel, self.epoch, self.sink, self.retain = output, max_blocks, voxel, epoch, sink, retain
        (output / "patches").mkdir(parents=True, exist_ok=True)
        (output / "submap.json").write_text(json.dumps({"epoch": epoch, "voxel_size": voxel}))
        self.store = sqlite3.connect(output / "blocks.sqlite", check_same_thread=False)  # the writer upserts; a restore reads only after writes.join()
        self.store.execute("CREATE TABLE IF NOT EXISTS blocks (x INT, y INT, z INT, seq INT, last_observed REAL, tsdf BLOB, weight BLOB, "
                           "PRIMARY KEY (x, y, z)) WITHOUT ROWID")
        self.cold, self.cold_keys = set(), None  # evicted blocks, whose newest state is in the store. ponytail: ~100 B a block, bounded by the map
        self.seq, self.evicted, self.restored, self.peak_blocks, self.newest, self.first_t, self.last_emit = 0, 0, 0, 0, None, None, 0.
        self.pending = deque()  # (t_capture, voxel samples) of frames whose surface is not extractable yet: at most PENDING_S of them
        self.patches, self.latency = deque(maxlen=WINDOW), deque(maxlen=WINDOW)
        self.bytes_total, self.never, self.first_3d, self.failed = 0, 0, None, None
        self.writes = queue.Queue(2)  # a writer that falls behind holds the frame loop, whose waiting frames then drop as gaps
        self.writer = threading.Thread(target=self._write_loop, daemon=True)
        self.writer.start()

    def integrate(self, depth_mm, k, c2w, t_capture):
        """One frame of uint16 millimetre depth with its own K (fx fy cx cy at depth resolution) and camera-to-world pose.
        False when no pixel has usable depth (Open3D refuses such a frame)."""
        valid = (depth_mm > 0) & (depth_mm <= DEPTH_MAX * 1000)
        if not valid.any():
            return False
        o3c = self.o3c
        image = self.o3d.t.geometry.Image(o3c.Tensor(np.ascontiguousarray(depth_mm)))
        # Open3D reads the depth pixel at floor(projection); +0.5 makes it the nearest one (synthetic box: median error 8.8 mm -> 0)
        intrinsic = o3c.Tensor(np.array([[k[0], 0, k[2] + .5], [0, k[1], k[3] + .5], [0, 0, 1.]]))
        extrinsic = o3c.Tensor(np.linalg.inv(c2w))
        band = self.grid.compute_unique_block_coordinates(image, intrinsic, extrinsic, 1000., DEPTH_MAX, TRUNC_VOXELS).numpy()
        blocks = o3c.Tensor(np.unique(np.concatenate([band, self._seen_through(depth_mm, k, c2w)]), axis=0))
        self._make_room(blocks)
        self.grid.integrate(blocks, image, intrinsic, extrinsic, 1000., DEPTH_MAX, TRUNC_VOXELS)
        hashmap = self.grid.hashmap()
        assert hashmap.capacity() == self.capacity, "block hash grew: buffer indices moved"
        index = hashmap.find(blocks)[0].numpy()
        weight = self.grid.attribute("weight").numpy()
        weight[index] = np.minimum(weight[index], MAX_WEIGHT)
        self.last_observed[index], self.dirty[index], self.newest = t_capture, True, t_capture
        self.first_t = t_capture if self.first_t is None else self.first_t
        self.peak_blocks = max(self.peak_blocks, hashmap.size())
        v, u = np.nonzero(valid)
        pick = np.linspace(0, len(v) - 1, SAMPLES).astype(int)
        z = depth_mm[v[pick], u[pick]] / 1000
        camera = np.stack([(u[pick] - k[2]) / k[0] * z, (v[pick] - k[3]) / k[1] * z, z], 1)
        self.pending.append((t_capture, np.round((camera @ c2w[:3, :3].T + c2w[:3, 3]) / self.voxel).astype(np.int64)))
        return True

    def _seen_through(self, depth_mm, k, c2w):
        """Known blocks, held or evicted, that this frame sees in front of its surface. Open3D updates only the blocks it is
        given and its own list is the truncation band around the new depth, so without these a surface that left (a
        passer-by, a moved box) would stay in the map for good. Given them, Open3D sets each voxel it sees through to free."""
        hashmap = self.grid.hashmap()
        keys = hashmap.key_tensor().numpy()[hashmap.active_buf_indices().numpy()]
        if self.cold:
            if self.cold_keys is None:
                self.cold_keys = np.array(sorted(self.cold), np.int32)
            keys = np.concatenate([keys, self.cold_keys])
        local = ((keys * 8 + 3.5) * self.voxel - c2w[:3, 3]) @ c2w[:3, :3]
        ahead = np.flatnonzero(local[:, 2] > .05)
        z = local[ahead, 2]
        u = np.round(local[ahead, 0] / z * k[0] + k[2]).astype(int)
        v = np.round(local[ahead, 1] / z * k[1] + k[3]).astype(int)
        inside = (u >= 0) & (u < depth_mm.shape[1]) & (v >= 0) & (v < depth_mm.shape[0])
        ahead, z, u, v = ahead[inside], z[inside], u[inside], v[inside]
        surface = depth_mm[v, u] / 1000  # at the block centre's pixel. ponytail: a block whose centre pixel is occluded waits for another view
        radius = 8 * self.voxel * 3 ** .5 / 2
        return keys[ahead[(surface > 0) & (z - radius < surface - TRUNC_VOXELS * self.voxel)]].astype(np.int32).reshape(-1, 3)

    def _make_room(self, incoming):
        """Keeps the cache at max_blocks: the longest-unseen clean blocks leave (their state is in the store), and an evicted
        block seen again comes back from the store before integration, so a revisit never restarts it from nothing."""
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
            self.cold.update(map(tuple, keys.tolist()))
            self.cold_keys = None
            for name in ("tsdf", "weight"):  # a reused buffer keeps its old values (checked): clear them
                self.grid.attribute(name).numpy()[gone] = 0
            hashmap.erase(o3c.Tensor(keys))
            self.evicted += excess
        back = [key for key in map(tuple, incoming.numpy()[~found].tolist()) if key in self.cold]
        if back:
            self.writes.join()  # an evicted block's newest state may still be on its way to the store
            buffers = hashmap.activate(o3c.Tensor(np.array(back, np.int32)))[0].numpy()
            tsdf, weight = self.grid.attribute("tsdf").numpy(), self.grid.attribute("weight").numpy()
            for key, i in zip(back, buffers):
                t, w, seen = self.store.execute("SELECT tsdf, weight, last_observed FROM blocks WHERE x=? AND y=? AND z=?", key).fetchone()
                tsdf[i, ..., 0] = np.frombuffer(zlib.decompress(t), np.int8).reshape(8, 8, 8) / 127
                weight[i, ..., 0], self.last_observed[i] = np.frombuffer(zlib.decompress(w), np.uint16).reshape(8, 8, 8), seen
            self.cold.difference_update(back)
            self.cold_keys = None
            self.restored += len(back)

    def held(self):
        """Buffer indices of the blocks in the cache."""
        return self.grid.hashmap().active_buf_indices().numpy()

    def blocks(self, index):
        """Patch arrays of the given buffer indices."""
        return {"blocks": self.grid.hashmap().key_tensor().numpy()[index].astype(np.int32),
                "tsdf": np.round(self.grid.attribute("tsdf").numpy()[index, ..., 0] * 127).astype(np.int8),
                "weight": np.minimum(self.grid.attribute("weight").numpy()[index, ..., 0], 65535).astype(np.uint16),
                "last_observed": self.last_observed[index]}

    def _extractable(self):
        """t_capture of the waiting frames whose sampled surface now has weight >= MIN_WEIGHT; a frame waiting past PENDING_S gives up."""
        if not self.pending:
            return np.zeros(0)
        times = np.array([t for t, _ in self.pending])
        voxels = np.concatenate([p for _, p in self.pending])
        block = voxels // 8
        local = voxels - 8 * block
        index, found = (t.numpy() for t in self.grid.hashmap().find(self.o3c.Tensor(block.astype(np.int32))))
        weight = self.grid.attribute("weight").numpy()[np.where(found, index, 0), local[:, 2], local[:, 1], local[:, 0], 0]
        done = (np.where(found, weight, 0) >= MIN_WEIGHT).reshape(len(times), SAMPLES).mean(1) >= EXTRACTABLE_SHARE
        stale = ~done & (times < self.newest - PENDING_S)
        self.never += int(stale.sum())
        self.pending = deque(p for p, keep in zip(self.pending, ~done & ~stale) if keep)
        return times[done]

    def emit(self):
        """Hands the touched blocks to the writer thread as the next patch."""
        if self.failed:  # the writer is gone (disk, a closed socket): stop the frame loop instead of queueing into nothing
            raise self.failed
        index = np.flatnonzero(self.dirty)
        self.last_emit = time.time()
        if not len(index):
            return
        self.seq += 1
        self.writes.put(dict(seq=self.seq, epoch=self.epoch, voxel_size=self.voxel, t_capture_newest=self.newest,
                             extractable_t_capture=self._extractable(), **self.blocks(index)))
        self.dirty[index] = False

    def _write_loop(self):
        """Patch file (atomic: a reader never sees half a file), block store upsert, retention, the consumer's sink."""
        while (patch := self.writes.get()) is not None:
            try:
                if not self.failed:
                    self._write(patch)
            except Exception as error:  # raised on the frame loop at the next emit or close; task_done still runs, so no join hangs
                self.failed = error
            finally:
                self.writes.task_done()
        self.writes.task_done()

    def _write(self, patch):
        t0 = time.perf_counter()
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **patch)
        data, path = buffer.getvalue(), self.output / "patches" / f"{patch['seq']:06d}.npz"
        path.with_suffix(".tmp").write_bytes(data)
        path.with_suffix(".tmp").replace(path)
        with self.store:
            self.store.executemany("INSERT OR REPLACE INTO blocks VALUES (?, ?, ?, ?, ?, ?, ?)", [
                (*key, patch["seq"], seen, zlib.compress(t.tobytes(), 1), zlib.compress(w.tobytes(), 1))
                for key, t, w, seen in zip(patch["blocks"].tolist(), patch["tsdf"], patch["weight"], patch["last_observed"].tolist())])
        (self.output / "patches" / f"{patch['seq'] - self.retain:06d}.npz").unlink(missing_ok=True)
        if self.sink:
            self.sink(data)
        t_written = time.time()
        self.latency.extend(t_written - patch["extractable_t_capture"])
        if self.first_3d is None and len(surface_points(as_blocks(patch), self.voxel)):
            self.first_3d = t_written
        self.bytes_total += len(data)
        self.patches.append({"seq": patch["seq"], "t_written": t_written, "blocks": len(patch["blocks"]), "bytes": len(data),
                             "extractable_frames": len(patch["extractable_t_capture"]), "write_ms": (time.perf_counter() - t0) * 1000})

    def close(self):
        """Flushes the last patch, stops the writer and returns the submap's stats (samples are the rolling window)."""
        self.emit()
        self.writes.put(None)
        self.writer.join()
        self.store.close()
        if self.failed:
            raise self.failed
        return {"epoch": self.epoch, "directory": str(self.output), "patches": self.seq, "patch_bytes_total": self.bytes_total,
                "patch_files_kept": len(list((self.output / "patches").glob("*.npz"))), "store_bytes": (self.output / "blocks.sqlite").stat().st_size,
                "patch_blocks_median": percentiles([p["blocks"] for p in self.patches], [50]),
                "patch_write_ms_p50_p95": percentiles([p["write_ms"] for p in self.patches], [50, 95]),
                "first_t_capture": self.first_t, "first_3d_written": self.first_3d,
                "map_latency_s_p50_p95_max": percentiles(self.latency, [50, 95, 100]),
                "frames_extractable": len(self.latency), "frames_never_extractable": self.never, "frames_pending_at_end": len(self.pending),
                "evicted_blocks": self.evicted, "restored_blocks": self.restored, "peak_blocks": self.peak_blocks, "max_blocks": self.max_blocks}


def serve(conn, output, max_blocks=MAX_BLOCKS, sink=None):
    """The worker: newest frame wins; a frame without a 'normal' pose or without usable depth is a coverage gap; each
    world-origin epoch is its own submap; a patch a second. Returns the run's stats (samples are rolling windows)."""
    import open3d  # noqa: F401  imported before the first frame arrives: inside it, it cost 3.5 s of dropped frames (M)
    live, submaps, first, integrated, integrate_ms = None, [], None, 0, deque(maxlen=WINDOW)
    gaps, gap_frames, in_gap = deque(maxlen=WINDOW), Counter(), False  # runs of frames not integrated: [first t_capture, last t_capture, frames]

    def gap(reason, t):
        nonlocal in_gap
        gap_frames[reason] += 1
        if in_gap:
            gaps[-1][1:] = [t, gaps[-1][2] + 1]
        else:
            gaps.append([t, t, 1])
        in_gap = True

    ended = False
    while not ended:
        message, dropped, ended = take_latest(conn, .05)
        for t in dropped:
            gap("dropped", t)
        if message is not None:
            t0 = time.perf_counter()
            head, _, depth, confidence = unpack(message)
            t, epoch, state = head["t_capture"], head.get("worldOriginEpoch"), head.get("trackingState")
            first = t if first is None else first
            c2w, depth = pose(head), depth_mm(depth, confidence)
            if not isinstance(epoch, int):  # a missing field is not a credible pose either
                gap("noEpoch", t)
            elif state != TRACKING_NORMAL:
                gap(f"{state}:{head.get('trackingStateReason')}", t)
            elif live and epoch < live.epoch:
                gap("epochWentBack", t)
            elif c2w is None:  # null, NaN or not a rotation: no place to integrate at
                gap("badPose", t)
            elif depth is None:
                gap("noDepth", t)
            else:
                if live is None or epoch != live.epoch:
                    if live:
                        submaps.append(live.close())
                    live = LiveMap(output / f"submap-{epoch}", max_blocks, epoch=epoch, sink=sink)
                k = np.array(head["K"]) * depth.shape[1] / 640
                if live.integrate(depth, k, c2w, t):
                    integrated, in_gap = integrated + 1, False
                    integrate_ms.append((time.perf_counter() - t0) * 1000)
                else:
                    gap("noDepth", t)
        if live and (ended or live.dirty.any() and time.time() - live.last_emit >= EMIT_EVERY_S):
            live.emit()
    if live:
        submaps.append(live.close())
    return {"frames_integrated": integrated, "first_t_capture": first, "coverage_gap_frames": dict(gap_frames), "coverage_gap_runs": len(gaps),
            "coverage_gap_longest_s": max((g[1] - g[0] for g in gaps), default=0.), "coverage_gaps_recent": list(gaps)[-200:],
            "decode_integrate_ms_p50_p95_max": percentiles(integrate_ms, [50, 95, 100]),
            "worker_peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** (20 if sys.platform == "darwin" else 10),
            "submaps": submaps}


def listen(output, max_blocks, authkey, ready):
    """The local worker process: one producer over a loopback socket."""
    with Listener(("127.0.0.1", 0), authkey=authkey) as listener:
        ready.put(listener.address)
        with listener.accept() as conn:
            stats = serve(conn, output, max_blocks)
    (output / "worker.json").write_text(json.dumps(stats, indent=1))


def transmit(raw, address, authkey, output, rate_hz):
    """The local phone process."""
    with Client(address, authkey=authkey) as conn:
        stats = produce(raw, conn.send_bytes, rate_hz)
    (output / "producer.json").write_text(json.dumps(stats, indent=1))


def load_patches(directory):
    """A consumer that followed every patch: the newest copy of every block, in patch order."""
    blocks, voxel = {}, VOXEL
    for path in sorted((directory / "patches").glob("*.npz")):
        with np.load(path) as patch:
            voxel = float(patch["voxel_size"])
            blocks.update(as_blocks(patch))
    return blocks, voxel


def load_store(directory):
    """A consumer that joins late: the newest state of every block the submap ever emitted."""
    with closing(sqlite3.connect(directory / "blocks.sqlite")) as store:
        rows = store.execute("SELECT x, y, z, tsdf, weight FROM blocks").fetchall()
    return ({(x, y, z): (np.frombuffer(zlib.decompress(t), np.int8).reshape(8, 8, 8), np.frombuffer(zlib.decompress(w), np.uint16).reshape(8, 8, 8))
             for x, y, z, t, w in rows}, json.loads((directory / "submap.json").read_text())["voxel_size"])


def surface_points(blocks, voxel, min_weight=MIN_WEIGHT):
    """Zero crossings of the TSDF between neighbouring voxels, in metres: where marching cubes would put its vertices."""
    keys = np.array(list(blocks), np.int64).reshape(-1, 3)
    tsdf = np.array([t for t, _ in blocks.values()], np.float32).reshape(-1) / 127
    weight = np.array([w for _, w in blocks.values()]).reshape(-1)
    z, y, x = np.indices((8, 8, 8))
    voxels = (keys[:, None, None, None] * 8 + np.stack([x, y, z], -1)).reshape(-1, 3)
    keep = weight >= min_weight
    voxels, tsdf = voxels[keep], tsdf[keep]
    if not len(voxels):
        return np.zeros((0, 3))
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


def evaluate(points, directory, mesh, cameras_run=None):
    """A consumer's surface compared with the offline mesh; the points are kept for a look."""
    import open3d as o3d
    o3d.io.write_point_cloud(str(directory / "live-surface.ply"), o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points)))
    return compare(points, mesh, cameras_run)


def run(args):
    """Producer and worker as two processes over a local socket, memory sampled each second, then metrics."""
    args.output.mkdir(parents=True)
    ctx = mp.get_context("spawn")
    authkey, ready = mp.current_process().authkey, ctx.Queue()
    worker = ctx.Process(target=listen, args=(args.output, args.max_blocks, authkey, ready))
    worker.start()
    producer = ctx.Process(target=transmit, args=(args.raw, ready.get(timeout=120), authkey, args.output, args.rate))
    producer.start()
    memory = []
    while worker.is_alive():
        rss = [subprocess.run(["ps", "-o", "rss=", "-p", str(p.pid)], capture_output=True, text=True).stdout.strip() for p in (worker, producer)]
        memory.append([time.time()] + [int(r) / 1024 if r else None for r in rss])
        worker.join(1.)
    producer.join()
    assert worker.exitcode == 0 and producer.exitcode == 0, (worker.exitcode, producer.exitcode)
    sent, done = (json.loads((args.output / f"{n}.json").read_text()) for n in ("producer", "worker"))
    submap = max(done["submaps"], key=lambda s: s["peak_blocks"])
    worker_mb = [m[1] for m in memory if m[1]]
    capture_time = [m[0] - sent["stream_start"] for m in memory if m[1]]
    metrics = {
        "machine": "local Mac CPU (worker and producer on one host, loopback socket: no network in the latency)",
        "rate_cap_hz": sent["rate_cap_hz"], "frames_sent": sent["frames_sent"], "sent_fps": sent["sent_fps"],
        "frames_integrated": done["frames_integrated"], "integrated_fps": done["frames_integrated"] / sent["capture_s"],
        "uplink_mbit_s": sent["uplink_mbit_s"], "uplink_depth_mbit_s": sent["uplink_depth_mbit_s"],
        "map_latency_s_p50_p95_max": submap["map_latency_s_p50_p95_max"],
        "map_latency_meaning": f"frame capture -> first patch written in which >= {EXTRACTABLE_SHARE:.0%} of its {SAMPLES} sampled surface points have weight >= {MIN_WEIGHT}",
        "frames_extractable": submap["frames_extractable"], "frames_never_extractable_30s": submap["frames_never_extractable"],
        "frames_pending_at_end": submap["frames_pending_at_end"],
        "time_to_first_extractable_3d_s": submap["first_3d_written"] - sent["stream_start"],
        "patches": submap["patches"], "patch_bytes_total": submap["patch_bytes_total"], "patch_blocks_median": submap["patch_blocks_median"],
        "patch_files_kept": submap["patch_files_kept"], "store_bytes": submap["store_bytes"],
        "patch_write_ms_p50_p95_writer_thread": submap["patch_write_ms_p50_p95"],
        "decode_integrate_ms_p50_p95_max": done["decode_integrate_ms_p50_p95_max"],
        "coverage_gap_frames": done["coverage_gap_frames"], "coverage_gap_runs": done["coverage_gap_runs"],
        "coverage_gap_longest_s": done["coverage_gap_longest_s"], "submaps": len(done["submaps"]),
        "blocks": {"peak": submap["peak_blocks"], "cap": submap["max_blocks"], "evicted": submap["evicted_blocks"], "restored": submap["restored_blocks"]},
        "worker_rss_mb": {"first_10s_max": max(m for t, m in zip(capture_time, worker_mb) if t < 10), "max": max(worker_mb),
                          "last": worker_mb[-1], "peak_rusage": done["worker_peak_rss_mb"],
                          "every_10s": [round(m) for m in worker_mb[::10]]},
        "producer_rss_mb_max": max(m[2] for m in memory if m[2]),
        "producer_send_lateness_s_p50_p95_max": sent["send_done_after_capture_s_p50_p95_max"]}
    (args.output / "memory.json").write_text(json.dumps(memory))
    if args.offline_mesh:
        directory = Path(submap["directory"])
        metrics["against_offline"] = evaluate(surface_points(*load_store(directory)), directory, args.offline_mesh, args.offline_cameras)
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=1))
    print(json.dumps(metrics, indent=1))


def render_box_depth(c2w, k, shape, room, cube=None):
    """Exact depth of a camera inside an axis-aligned room with an axis-aligned cube (or none) on its floor."""
    v, u = np.indices(shape)
    rays = np.stack([(u - k[2]) / k[0], (v - k[3]) / k[1], np.ones(shape)], -1) @ c2w[:3, :3].T  # camera z = 1: t is depth
    o = c2w[:3, 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        leave_room = np.maximum((room[0] - o) / rays, (room[1] - o) / rays).min(-1)
        if cube is None:
            return leave_room
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
    from scipy.spatial import cKDTree
    room, cube = np.array([[0, 0, 0], [4, 3, 2.5]]), np.array([[1.6, 1.1, 0], [2.2, 1.7, .6]])
    k, shape, voxel = [100., 100., 60., 45.], (90, 120), .04
    mm = lambda depth: np.round(depth * 1000).astype(np.uint16)
    frames = []
    for n in range(90):  # one lap around the cube, looking at it from 1.3 m and 1.5 m up
        angle = 2 * np.pi * n / 90
        c2w = look_at(np.array([1.9 + 1.3 * np.cos(angle), 1.4 + 1.3 * np.sin(angle), 1.5]), np.array([1.9, 1.4, .3]))
        frames.append((mm(render_box_depth(c2w, k, shape, room, cube)), c2w, n / 30))

    def truth_distance(p):
        to_room = np.minimum(p - room[0], room[1] - p).min(1)
        outside = np.linalg.norm(np.maximum(np.maximum(cube[0] - p, p - cube[1]), 0), axis=1)
        inside = np.minimum(p - cube[0], cube[1] - p).min(1).clip(min=0)
        return np.minimum(np.abs(to_room), outside + inside)

    def cube_coverage(points):
        g = np.linspace(.01, .59, 30)
        a, b = (m.ravel() for m in np.meshgrid(g, g))
        x0, y0 = cube[0, :2]
        faces = [np.c_[x0 + a, y0 + b, np.full_like(a, .6)], np.c_[np.full_like(a, 1.6), y0 + a, b], np.c_[np.full_like(a, 2.2), y0 + a, b],
                 np.c_[x0 + a, np.full_like(a, 1.1), b], np.c_[x0 + a, np.full_like(a, 1.7), b]]
        return float((cKDTree(points).query(np.concatenate(faces))[0] < voxel).mean())

    wall = np.full(shape, 2000, np.uint16)
    with TemporaryDirectory() as directory:
        for name in ("whole", "sliding"):
            cap = MAX_BLOCKS if name == "whole" else int(.6 * peak)  # the lap needs 40% more blocks than the sliding cache holds
            out = Path(directory) / name
            live = LiveMap(out, cap, voxel, retain=10 ** 6)
            for depth, c2w, t in frames:
                live.integrate(depth, k, c2w, t)
                if round(t * 30) % 10 == 9:
                    live.emit()
                assert live.grid.hashmap().size() <= cap
            live.close()
            blocks, patch_voxel = load_patches(out)
            stored = load_store(out)[0]  # a late consumer's map is the one a consumer that followed every patch has
            assert stored.keys() == blocks.keys() and all(np.array_equal(stored[key][0], t) and np.array_equal(stored[key][1], w) for key, (t, w) in blocks.items())
            points = surface_points(blocks, patch_voxel)
            error, coverage = truth_distance(points), cube_coverage(points)
            assert patch_voxel == voxel and np.median(error) < .05 * voxel and np.percentile(error, 95) < .25 * voxel, (name, np.percentile(error, [50, 95]))
            assert coverage > .95, (name, coverage)  # the cube's top and four sides came out whole
            if name == "whole":
                peak, whole = live.peak_blocks, blocks
                own = live.blocks(live.held())
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

        # a box stands while the camera looks at it, the camera turns away long enough for a small cache to evict it, the
        # box is taken away and the camera looks back: the box leaves the map, the room stays, the floor under it appears
        box = np.array([[2.9, 1.2, 0], [3.5, 1.8, .6]])
        at_box = [look_at(np.array([1.3, 1.5 + .4 * np.sin(n / 7), 1.5]), np.array([3.2, 1.5, .3])) for n in range(100)]
        away = [look_at(np.array([1.3, 1.5, 1.5]), np.array([0, 1.5 + 1.4 * np.sin(n / 9), .2 + .6 * (n % 3)])) for n in range(90)]
        phases = [(c2w, box) for c2w in at_box + away] + [(c2w, None) for c2w in at_box[:50]]
        in_box = lambda p, pad: np.all((p > box[0] - pad) & (p < box[1] + pad), 1)
        on_box = lambda p: in_box(p, voxel) & (p[:, 2] > 1.5 * voxel)
        for name in ("whole", "sliding"):
            out = Path(directory) / f"moved-{name}"
            live = LiveMap(out, MAX_BLOCKS if name == "whole" else cap_moved, voxel)
            for n, (c2w, standing) in enumerate(phases):
                live.integrate(mm(render_box_depth(c2w, k, shape, room, standing)), k, c2w, n / 15)
                if n % 15 == 14:
                    live.emit()
                if n == 99 and name == "whole":
                    before = surface_points(as_blocks(live.blocks(live.held())), voxel)
                if n == 189 and name == "whole":  # just under what the look away touches: every block of the box has to leave
                    cap_moved = int((live.last_observed[live.held()] >= 100 / 15).sum() * .95)
            live.close()
            after = surface_points(*load_store(out))
            static = before[~in_box(before, 2 * voxel)]  # the room the first look saw, less the box and the floor it hid
            kept = float((cKDTree(after).query(static)[0] < voxel).mean())
            footprint = np.all((after[:, :2] > box[0, :2] + voxel) & (after[:, :2] < box[1, :2] - voxel), 1) & (np.abs(after[:, 2]) < voxel)
            assert on_box(before).sum() > 300 and on_box(after).sum() <= .01 * on_box(before).sum(), (name, on_box(before).sum(), on_box(after).sum())
            assert kept > .97 and footprint.sum() > 80, (name, kept, footprint.sum())
            assert name == "whole" or live.restored > 0, "the box's blocks never left the cache: the check would prove nothing"
            print(f"moved box ({name}): box surface {on_box(before).sum()} -> {on_box(after).sum()} points, room kept {kept:.3f}, "
                  f"floor under the box {footprint.sum()} points, evicted {live.evicted}, restored {live.restored}")

        # the stream contract: only 'normal' poses are integrated, the rest are gaps with their reason; a new world-origin
        # epoch starts a new submap, so a wall seen 2 m away in epoch 0 and 3 m away in epoch 1 is never fused into one map
        class OneAtATime:  # hands serve() one message per poll, so latest-wins drops none
            def __init__(self, messages):
                self.messages = deque(messages)

            def poll(self, timeout):
                return timeout > 0 and bool(self.messages)

            def recv_bytes(self):
                return self.messages.popleft()

        png = cv2.imencode(".png", wall)[1].tobytes()
        moved = np.eye(4)
        moved[2, 3] = 1.  # the camera 1 m further along z: the same wall reading lands at z = 3
        frame = lambda n, state, reason, epoch, c2w: header(n, float(n), float(n), [v * 640 / 120 for v in k], c2w, state, reason, epoch)
        stream = [frame(n, "normal", None, 0, np.eye(4)) for n in range(4)] + [frame(n, "limited", "excessiveMotion", 0, moved) for n in range(4, 8)]
        stream += [frame(8, "notAvailable", None, 0, moved), frame(9, "normal", None, None, moved)]
        stream += [frame(n, "normal", None, 1, moved) for n in range(10, 14)]
        # an older epoch after a newer one, a null pose and a NaN pose are gaps: none may reopen a submap or claim coverage
        stream += [frame(14, "normal", None, 0, np.eye(4)), frame(15, "normal", None, 1, None), frame(16, "normal", None, 1, np.full((4, 4), np.nan))]
        blank = pack(frame(17, "normal", None, 2, moved), b"", cv2.imencode(".png", 0 * wall)[1].tobytes())  # a submap whose one frame has no depth still closes
        stats = serve(OneAtATime([pack(h, b"", png) for h in stream] + [blank, b""]), Path(directory) / "contract", 5000)
        assert len(stats["submaps"]) == 3 and stats["frames_integrated"] == 8 and stats["submaps"][2]["patches"] == 0, stats
        z = [surface_points(*load_store(Path(directory) / "contract" / f"submap-{e}"))[:, 2] for e in (0, 1)]
        assert len(z[0]) and len(z[1]) and np.abs(z[0] - 2).max() < .02 and np.abs(z[1] - 3).max() < .02, (np.unique(z[0].round(2)), np.unique(z[1].round(2)))
        assert stats["coverage_gap_frames"] == {"limited:excessiveMotion": 4, "notAvailable:None": 1, "noEpoch": 1, "epochWentBack": 1, "badPose": 2,
                                                "noDepth": 1}, stats["coverage_gap_frames"]
        assert stats["coverage_gap_runs"] == 2 and stats["coverage_gap_longest_s"] == 5. and stats["first_t_capture"] == 0.  # frames 4 to 9 unbroken, then 14 to 17
        print(f"contract: 8 of 18 frames integrated into 2 submaps and an empty third (walls at {np.median(z[0]):.2f} m and {np.median(z[1]):.2f} m), gaps {stats['coverage_gap_frames']}")

        # map latency is capture -> the first patch in which the frame's surface is extractable (weight >= MIN_WEIGHT)
        out = Path(directory) / "latency"
        live = LiveMap(out, 5000, voxel, retain=10 ** 6)
        for t in (0, 1, 2, 3):
            live.integrate(wall, k, np.eye(4), float(t))
            if t != 1:
                live.emit()
        live.close()
        patches = [dict(np.load(p)) for p in sorted((out / "patches").glob("*.npz"))]
        assert [p["extractable_t_capture"].tolist() for p in patches] == [[], [0., 1., 2.], [3.]], [p["extractable_t_capture"] for p in patches]
        assert not len(surface_points(as_blocks(patches[0]), voxel)) and len(surface_points(as_blocks(patches[1]), voxel))
        assert live.first_3d == live.patches[1]["t_written"] and len(live.latency) == 4

        # patch writes run on their own thread: a slow write does not hold the frame loop
        slow = LiveMap(Path(directory) / "slow", 5000, voxel, sink=lambda data: time.sleep(.3))
        slow.integrate(wall, k, np.eye(4), 0.)
        t0 = time.perf_counter()
        slow.emit()
        handed = time.perf_counter() - t0
        slow.close()
        assert handed < .1, f"emit held the frame loop for {handed:.2f} s: the write is not on its own thread"

        def hung_up(data):
            raise ConnectionError("consumer went away")
        broken = LiveMap(Path(directory) / "broken", 5000, voxel, sink=hung_up)
        broken.integrate(wall, k, np.eye(4), 0.)
        broken.emit()
        try:
            broken.close()
            raise AssertionError("a failed patch write went unnoticed")
        except ConnectionError:
            pass

        # a static scene: the patch files stay at RETAIN, the block store stays the same size, samples are rolling windows
        out = Path(directory) / "static"
        live, sizes = LiveMap(out, 5000, voxel, retain=5), []
        for t in range(40):  # 40 s of a camera held still
            live.integrate(wall, k, np.eye(4), float(t))
            live.emit()
            live.writes.join()
            sizes.append((out / "blocks.sqlite").stat().st_size)
        live.close()
        files = len(list((out / "patches").glob("*.npz")))
        assert files == 5 and sizes[-1] <= 1.1 * sizes[9], (files, sizes[9], sizes[-1])
        assert load_store(out)[0].keys() == set(map(tuple, live.blocks(live.held())["blocks"].tolist()))
        assert all(d.maxlen for d in (live.patches, live.latency))
        print(f"static: 40 patches leave {files} files and a {sizes[-1] / 1024:.0f} KB store ({sizes[9] / 1024:.0f} KB after 10)")

    times = np.arange(0, 10, 1 / 60) + np.random.default_rng(0).normal(0, .001, 600)
    assert abs(paced(times, 15).sum() - 150) <= 1 and paced(times, 0).all()

    inbox = Inbox()
    for sender, receiver in (mp.Pipe(), (inbox, inbox)):  # latest frame wins over a socket and over a websocket's inbox
        for t in (1., 2., 3.):
            sender.send_bytes(pack({"t_capture": t}))
        latest, dropped, ended = take_latest(receiver, 0)
        assert unpack(latest)[0]["t_capture"] == 3 and dropped == [1., 2.] and not ended
        sender.send_bytes(pack({"t_capture": 4.}))
        sender.send_bytes(b"")
        latest, dropped, ended = take_latest(receiver, 0)
        assert unpack(latest)[0]["t_capture"] == 4 and dropped == [] and ended
    head, rgb, depth, confidence = unpack(pack({"seq": 7}, b"j", b"dd", b""))
    assert head["seq"] == 7 and (rgb, depth, confidence) == (b"j", b"dd", b"")
    print("live map check passed: a moving camera around a box integrates to the box and room surfaces, patches and the block "
          "store rebuild the worker's map exactly, a moved box leaves the map (from evicted blocks too), non-normal poses and "
          "epochs are kept apart, map latency counts extractable surface, writes are off the frame loop, a static scene's "
          "disk stays flat, the uplink is paced and the newest frame wins")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--raw", type=Path, help="ARKitScenes raw capture: wide.zip, wide_intrinsics.zip, lowres_wide.traj, lowres_depth.zip, confidence.zip")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--rate", type=float, default=RATE_HZ, help="uplink cap in Hz (0: every ARFrame)")
    parser.add_argument("--compare", type=Path, help="a submap directory of an earlier run to compare again")
    parser.add_argument("--offline-mesh", type=Path, help="the offline fusion of the same capture, e.g. runs/da3-posed-arkit-raw-081/mono-anchored-mesh.ply")
    parser.add_argument("--offline-cameras", type=Path, help="the offline run's cameras (prediction.npz, run.json), e.g. runs/arkit-47333932-cameras-080")
    parser.add_argument("--max-blocks", type=int, default=MAX_BLOCKS)
    a = parser.parse_args()
    if a.self_check:
        self_check()
    elif a.compare:
        print(json.dumps(evaluate(surface_points(*load_store(a.compare)), a.compare, a.offline_mesh, a.offline_cameras), indent=1))
    else:
        run(a)
