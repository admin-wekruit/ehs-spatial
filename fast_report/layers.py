"""Fast report layers (builder C, FAST-BUILD-SPEC sections 6-7): the patch store a run writes, the CLI's local mirror of what
run() yields, and the loopback HTTP endpoint the viewer (#/live/<reportId>) polls.

Store: Modal Volume `panoptes-fb-layers` mounted at `root`; the local mirror uses the same layout.
  blobs/sha256/<hex>                                  content addressed, never rewritten
  reports/<report>/patches/<seq:06d>-<layer>.json     one per put()
  reports/<report>/run.json                           the last timing (the run's final event)
  reports/<report>/written.json, served.json          mirror only: {seq: written_s} from `written` events, {seq: first-fetch unix}

Patch `data` per layer, as web/src/live-report.ts reads it. Frames are per shot ("shot-<index>"), in estimated metres.
Every version of a layer is complete (models too: cumulative); the viewer uses the newest.
  video    {fps, frames, width, height, sha256, window_s}                       blobs {video: MP4}
  cameras  {shots: [{index, keys, times, c2w, K, wh, source_wh, mpu, floor: {normal, point_m}}], scale: {source, status}, license}
  room     {shots: [{index, triangles, points}], kind: light|full}              blobs {mesh-<i>: pack_mesh, points-<i>: points_glb}
           (a light_mesh version put just before the full one, same commit: the room shows while the full one travels)
  people   {tracks: [{id, shot, t0, t1, detections, points: [{t, frame, xyz}]}], rules: [{t, frame, rule, verdict, track?}], note}
                                                                                blobs {track-<id>: ribbon}
  objects  {objects: [{id, shot, word, votes, frames, centroid_m, box_min_m, box_max_m, best_key}]}
  outlines {analysis} inline, or blobs {analysis: JSON} over 1 MB (VideoView's format; objects carry source segmented|projected)
  events   {model, windows: [{t0, t1, caption, events}]}
  models   {models: [{object, transform: {position, quaternion, scale}, bounds: {min, max}, gate}]}   blobs {model-<object>: GLB}
  splat    {format: "splat32", count, shot, kind: preview|full, train_s, steps}   blobs {splat}
  timing   {layers: [rows so far], run: run.json so far}                         the Writer adds one to every commit

Wiring (A): in run(), w = Writer(volume, report_id, clock, client_has=options["client_has"], report=lambda: clock.report(vram));
the pipeline thread puts layers (those finished together back to back) and ends with w.close(); run() does `yield from
w.events()` and then yields {"type": "run", "report": report_id, "run": run_json}. The CLI stores its own MP4 with
put_blob(out, mp4), calls mirror(event, out) on every event, and with --serve runs serve(out) for the viewer.

  python -m fast_report.layers self-check
  python -m fast_report.layers serve ROOT [--port 8793] [--replay SRC REPORT [--as NEW] [--speed 1]]
  python -m fast_report.layers fixture-e9 E9_RUN_DIR OUT_ROOT            recorded patch stream from E9 run 003 (simulated times)
  python -m fast_report.layers modal-check SRC REPORT OUT_ROOT [--port 8793]   the recording through a real Writer on the Volume
"""
import hashlib
import json
import sys
from pathlib import Path
import queue
import re
import struct
import threading
import time
from urllib.parse import parse_qs, urlsplit

SCHEMA = "panoptes-fast-patch-v1"
VOLUME = "panoptes-fb-layers"


def put_blob(root, payload):
    """Content addressed and immutable: an existing key is never rewritten. Returns the sha256."""
    sha = hashlib.sha256(payload).hexdigest()
    path = Path(root) / "blobs" / "sha256" / sha
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{sha}.{threading.get_ident()}.part")
        tmp.write_bytes(payload)
        tmp.rename(path)
    return sha


def _plain(o):
    """numpy scalars and arrays (the pipeline's floats and ints) as JSON."""
    return o.item() if hasattr(o, "item") and getattr(o, "ndim", 0) == 0 else o.tolist() if hasattr(o, "tolist") else str(o)


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{threading.get_ident()}.part")
    tmp.write_text(json.dumps(value, separators=(",", ":"), default=_plain))
    tmp.rename(path)


def _read_json(path, default):
    return json.loads(path.read_text()) if path.exists() else default


class Writer:
    """put() never blocks the pipeline. A writer thread writes each patch as it comes (blobs, then the patch JSON) and hands
    the event to run() at once (sent_s); once the writer is idle, a committer thread commits the Volume for everything written
    since its last commit, with a `timing` patch, then sends a `written` event (written_s). Layers put together share one
    commit; a layer put during a commit is still sent at once and goes into the next. clock.t0_unix is when the MP4 bytes were in the container;
    clock.external(...) records write.<layer> (put to commit) if the clock has it.
    report: a zero-argument callable returning run.json so far (D's clock.report(vram)), for the timing layer."""

    def __init__(self, volume, report_id, clock, root="/layers", client_has=(), report=None):
        self.volume, self.report_id, self.clock, self.root, self.report = volume, report_id, clock, Path(root), report
        self.sent = set(client_has)  # blobs the client already holds (its own MP4) are never sent back
        self.seq, self.versions, self.rows, self.pending, self.closing, self.busy = 0, {}, [], [], False, False
        self.lock, self.ready = threading.Lock(), threading.Condition()
        self.inbox, self.outbox = queue.Queue(), queue.Queue()
        threading.Thread(target=self._write, name="layers.write", daemon=True).start()
        self.committer = threading.Thread(target=self._commit, name="layers.commit", daemon=True)
        self.committer.start()

    def now(self):
        return round(time.time() - self.clock.t0_unix, 3)

    def put(self, layer, data, blobs=None, status="estimated", labels=()):
        """blobs: {role: (bytes, meta)}; meta (mediaType, format, byteLayout, ...) is copied into the patch. Returns nothing:
        the seq is given when the patch is written, so seqs follow write order and a viewer polling after=<last seq> misses none."""
        self.inbox.put((layer, data, blobs or {}, status, list(labels), self.now()))

    def send(self, event):
        """An event for the client that is not a layer (mvp2/identity: a naming request the bench relays), in stream order."""
        self.outbox.put(event)

    def events(self):
        """What run() yields, in order, until close(): patch, written, error."""
        while (event := self.outbox.get()) is not None:
            yield event

    def close(self):
        """After the last put(): the last commit, then events() ends."""
        self.inbox.put(None)
        self.committer.join()

    def _emit(self, layer, data, blobs, status, labels, queued):
        with self.lock:  # one writer at a time, so seq order is event order
            self.seq += 1
            self.versions[layer] = version = self.versions.get(layer, 0) + 1
            refs, send = {}, {}
            for role, (payload, meta) in blobs.items():
                sha = put_blob(self.root, payload)
                refs[role] = {"sha256": sha, "bytes": len(payload), **meta}
                if sha not in self.sent:
                    self.sent.add(sha)
                    send[sha] = payload
            patch = {"schema": SCHEMA, "report": self.report_id, "seq": self.seq, "layer": layer, "version": version, "status": status,
                     "labels": labels, "t0_unix": self.clock.t0_unix, "queued_s": queued, "sent_s": self.now(), "data": data, "blobs": refs}
            _write_json(self.root / "reports" / self.report_id / "patches" / f"{self.seq:06d}-{layer}.json", patch)
            self.outbox.put({"type": "patch", "report": self.report_id, "patch": patch, "blobs": send})
            return {"layer": layer, "seq": self.seq, "version": version, "queued_s": queued, "sent_s": patch["sent_s"],
                    "bytes": sum(len(p) for p, _ in blobs.values())}

    def _write(self):
        while (item := self.inbox.get()) is not None:
            with self.ready:
                self.busy = True
            try:
                row = self._emit(*item)
            except Exception as error:  # the run goes on; the CLI prints it and the layer stays missing
                self.outbox.put({"type": "error", "report": self.report_id, "layer": item[0], "error": repr(error)})
                row = None
            with self.ready:
                self.pending += [row] if row else []
                self.busy = not self.inbox.empty()  # more waiting: they join this commit
                self.ready.notify()
        with self.ready:
            self.closing = True
            self.ready.notify()

    def _commit(self):
        while True:
            with self.ready:
                self.ready.wait_for(lambda: self.pending and not self.busy or self.closing)
            time.sleep(.05)  # ponytail: a 50 ms grace, so layers put a moment apart still share one commit
            with self.ready:
                self.ready.wait_for(lambda: self.pending and not self.busy or self.closing)
                rows, self.pending = self.pending, []
            if not rows:
                break
            try:
                rows.append(self._emit("timing", {"layers": self.rows + [dict(r) for r in rows], "run": self.report() if self.report else None},
                                       {}, "timing", [], self.now()))
                start = time.time()
                if self.volume is not None:
                    self.volume.commit()
                written = self.now()
            except Exception as error:
                self.outbox.put({"type": "error", "report": self.report_id, "seqs": [r["seq"] for r in rows], "error": repr(error)})
                continue
            for row in rows:
                row["written_s"] = written
            self.rows += rows
            self.outbox.put({"type": "written", "report": self.report_id, "seqs": [r["seq"] for r in rows], "written_s": written,
                             "commit_s": round(time.time() - start, 3)})
            if hasattr(self.clock, "external"):
                for row in rows:
                    if row["layer"] != "timing":
                        self.clock.external(f"write.{row['layer']}", start_unix=self.clock.t0_unix + row["queued_s"], end_unix=time.time(),
                                            n={"seq": row["seq"], "bytes": row["bytes"]})
        self.outbox.put(None)


def mirror(event, root, max_bytes=None):
    """The CLI's copy of one event run() yielded: blobs checked against their sha256 and stored like the Volume, patch JSONs,
    `written` times (written.json) and the final run.json. Raises on a bad blob or a blob the mirror does not hold.
    max_bytes: larger blobs stay on the Volume only (a nearly full disk); their patches still mirror."""
    root, report = Path(root), Path(root) / "reports" / event["report"]
    if event["type"] == "patch":
        for sha, payload in event["blobs"].items():
            if hashlib.sha256(payload).hexdigest() != sha:
                raise ValueError(f"blob {sha} does not match its sha256")
            if max_bytes is None or len(payload) <= max_bytes:
                put_blob(root, payload)
        patch = event["patch"]
        missing = [r["sha256"] for r in patch["blobs"].values() if not (root / "blobs" / "sha256" / r["sha256"]).exists()
                   and (max_bytes is None or r["bytes"] <= max_bytes)]
        if missing:
            raise ValueError(f"patch {patch['seq']} needs blobs the mirror does not hold: {missing}")
        _write_json(report / "patches" / f"{patch['seq']:06d}-{patch['layer']}.json", patch)
    elif event["type"] == "written":
        written = _read_json(report / "written.json", {})
        written.update({str(s): event["written_s"] for s in event["seqs"]})
        _write_json(report / "written.json", written)
    elif event["type"] == "run":
        _write_json(report / "run.json", event["run"])


def _sniff(data):
    return ("video/mp4" if data[4:8] == b"ftyp" else "model/gltf-binary" if data[:4] == b"glTF"
            else "application/json" if data[:1] in (b"{", b"[") else "application/octet-stream")


def _poll(folder, after, lock):
    """Patches after `after`, plus every written and served time so far; a patch's first fetch is its served time."""
    with lock:
        served, now, out = _read_json(folder / "served.json", {}), time.time(), []
        for path in sorted((folder / "patches").glob("*.json")):
            if int(path.name[:6]) > after:
                patch = json.loads(path.read_text())
                out.append(patch)
                served.setdefault(str(patch["seq"]), now)
        if out:
            _write_json(folder / "served.json", served)
    return {"patches": out, "written": _read_json(folder / "written.json", {}), "served": served, "run": _read_json(folder / "run.json", None)}


def serve(root, port=8793, click=None, alive=None):
    """Loopback only, no key: GET /fast/reports/<id>/patches?after=<seq>, GET /fast/blobs/<hex> (immutable). Returns the running server.
    mvp3 (D4 b), with a report container attached: GET /fast/reports/<id>/click?i=<pick frame>&x=&y= -> click(id, i, x, y), its
    on-demand card; GET /fast/alive -> alive() (the viewer's heartbeat keeps the container up). Without one: 404."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    root, lock = Path(root), threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            url = urlsplit(self.path)
            if m := re.fullmatch(r"/fast/reports/([\w.-]+)/patches", url.path):
                after = int(parse_qs(url.query).get("after", ["0"])[0])
                return self._send(json.dumps(_poll(root / "reports" / m[1], after, lock)).encode(), "application/json", "no-store")
            if (m := re.fullmatch(r"/fast/reports/([\w.-]+)/click", url.path)) and click:
                q = parse_qs(url.query)
                try:
                    body, status = click(m[1], int(q["i"][0]), float(q["x"][0]), float(q["y"][0])), 200
                except Exception as error:  # noqa: BLE001  the viewer says why (the container is gone, the report is not on the Volume)
                    body, status = {"status": "unavailable", "reason": repr(error)[:300]}, 503
                return self._send(json.dumps(body, default=str).encode(), "application/json", "no-store", status)
            if url.path == "/fast/alive" and alive:
                return self._send(json.dumps({"alive": alive()}).encode(), "application/json", "no-store")
            if (m := re.fullmatch(r"/fast/blobs/([0-9a-f]{64})", url.path)) and (path := root / "blobs" / "sha256" / m[1]).exists():
                data = path.read_bytes()
                return self._send(data, _sniff(data), "public, max-age=31536000, immutable")
            self._send(b"not found", "text/plain", "no-store", 404)

        def _send(self, body, media, cache, status=200):
            self.send_response(status)
            for k, v in (("Content-Type", media), ("Content-Length", str(len(body))), ("Cache-Control", cache)):
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


# ---------------------------------------------------------------- recordings: replay a mirrored report (viewer tests, demos)

def recording(src, report):
    """A mirrored report as its patches (seq order), its blobs by sha256 and its written times."""
    folder = Path(src) / "reports" / report
    patches = [json.loads(p.read_text()) for p in sorted((folder / "patches").glob("*.json"))]
    have = lambda r: (Path(src) / "blobs" / "sha256" / r["sha256"]).exists()  # noqa: E731  a capped mirror (--mirror-max-mb) lacks big blobs
    blobs = {r["sha256"]: (Path(src) / "blobs" / "sha256" / r["sha256"]).read_bytes() for p in patches for r in p["blobs"].values() if have(r)}
    return patches, blobs, _read_json(folder / "written.json", {})


def replay(src, report, root, as_report=None, speed=1.0):
    """Mirror a recorded report into root at its recorded sent/written times, as a new report (t0 = now): the viewer sees the
    same arrivals in the same order and spacing. Two replays of one recording must draw the same picture."""
    patches, blobs, written = recording(src, report)
    as_report, start, seen, timeline = as_report or report, time.time(), set(), []
    for p in patches:
        if any(r["sha256"] not in blobs for r in p["blobs"].values()):  # its blobs stayed on the Volume: the layer is left out
            print(f"replay: patch {p['seq']} ({p['layer']}) skipped, a blob is not in {src}", file=sys.stderr)
            continue
        send = {r["sha256"]: blobs[r["sha256"]] for r in p["blobs"].values() if r["sha256"] not in seen}
        seen |= set(send)
        timeline.append((p["sent_s"], 0, {"type": "patch", "report": as_report, "patch": {**p, "report": as_report, "t0_unix": start}, "blobs": send}))
    by_time = {}
    for seq, t in written.items():
        by_time.setdefault(t, []).append(int(seq))
    timeline += [(t, 1, {"type": "written", "report": as_report, "seqs": sorted(seqs), "written_s": t}) for t, seqs in by_time.items()]
    for t, _, event in sorted(timeline, key=lambda x: x[:2]):
        time.sleep(max(0., start + t / speed - time.time()))
        mirror(event, root)
    if (run := _read_json(Path(src) / "reports" / report / "run.json", None)) is not None:  # the CLI mirrors run.json last
        mirror({"type": "run", "report": as_report, "run": run}, root)
    return as_report


# ---------------------------------------------------------------- geometry blobs (A and B call these; the fixture too)

def pack_mesh(vertices, normals, rgb01, faces):
    """panoptes-mesh-v1: stride 9 float32 rows (xyz, normal, rgb 0-1) then uint32 indices; returns (bytes, meta)."""
    import numpy as np
    rows = np.hstack([vertices, normals, rgb01]).astype("<f4")
    indices = np.asarray(faces, "<u4").ravel()
    layout = {"stride": 9, "byteOffset": 0, "vertexCount": len(rows), "indexByteOffset": rows.nbytes, "indexCount": len(indices), "indexType": "uint32"}
    return rows.tobytes() + indices.tobytes(), {"mediaType": "application/octet-stream", "format": "panoptes-mesh-v1", "byteLayout": layout}


def light_mesh(vertices, normals, rgb01, faces, cell=0.06):
    """A quick first version of a big mesh (vertex clustering on a `cell` grid: 6 cm takes ME340's 1.56 M-triangle walk room
    to 0.35 M, 49 -> 11 MB), put just before the full one so the room shows while the full one travels. Returns the four
    arrays for pack_mesh. ~0.3 s on the CPU; ponytail: numpy, port to torch if it ever sits on the critical path."""
    import numpy as np
    v = np.asarray(vertices, float)
    q = np.floor((v - v.min(0)) / cell).astype(np.int64)
    _, inverse, counts = np.unique((q[:, 0] << 42) | (q[:, 1] << 21) | q[:, 2], return_inverse=True, return_counts=True)
    inverse = inverse.ravel()
    mean = lambda a: np.stack([np.bincount(inverse, np.asarray(a, float)[:, j], len(counts)) for j in range(3)], 1) / counts[:, None]
    f = inverse[np.asarray(faces)]
    f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])]
    n = mean(normals)
    return mean(v), n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9), mean(rgb01), f


def points_glb(xyz, rgb, point_size):
    """GLB points: float32 POSITION + normalized uint8 RGBA COLOR_0, one buffer (lingbot_dense_map.write_points_glb's layout, as bytes).
    point_size is the cell size in native units, so the viewer draws round points at world size."""
    import numpy as np
    n = len(xyz)
    pos = np.ascontiguousarray(xyz, "<f4")
    blob = pos.tobytes() + np.ascontiguousarray(np.concatenate([rgb, np.full((n, 1), 255, np.uint8)], 1), np.uint8).tobytes()
    gltf = {"asset": {"version": "2.0", "generator": "panoptes fast_report.layers"}, "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}],
            "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "COLOR_0": 1}, "mode": 0}]}], "buffers": [{"byteLength": len(blob)}],
            "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": 12 * n}, {"buffer": 0, "byteOffset": 12 * n, "byteLength": 4 * n}],
            "accessors": [{"bufferView": 0, "componentType": 5126, "count": n, "type": "VEC3", "min": pos.min(0).tolist(), "max": pos.max(0).tolist()},
                          {"bufferView": 1, "componentType": 5121, "normalized": True, "count": n, "type": "VEC4"}]}
    head = json.dumps(gltf, separators=(",", ":")).encode()
    head += b" " * (-len(head) % 4)
    blob += b"\0" * (-len(blob) % 4)
    data = struct.pack("<III", 0x46546C67, 2, 28 + len(head) + len(blob)) + struct.pack("<II", len(head), 0x4E4F534A) + head + struct.pack("<II", len(blob), 0x004E4942) + blob
    return data, {"mediaType": "model/gltf-binary", "format": "glb", "pointSizeNative": float(point_size)}


def ribbon(points, up, width=0.12, lift=0.02, rgb=(1., .55, .1)):
    """A flat band along a person's floor path (width in native units, just above the floor) as panoptes-mesh-v1."""
    import numpy as np
    p, up = np.asarray(points, float), np.asarray(up, float) / np.linalg.norm(up)
    if len(p) < 2:
        p = np.vstack([p, p + 1e-3 * np.cross(up, [1, 0, 0] if abs(up[0]) < .9 else [0, 1, 0])])
    d = np.gradient(p, axis=0)
    side = np.cross(d, up)
    side /= np.maximum(np.linalg.norm(side, axis=1, keepdims=True), 1e-9)
    centre = p + lift * up
    vertices = np.vstack([centre - side * width / 2, centre + side * width / 2])
    n = len(p)
    faces = [[i, i + 1, n + i] for i in range(n - 1)] + [[i + 1, n + i + 1, n + i] for i in range(n - 1)]
    return pack_mesh(vertices, np.tile(up, (2 * n, 1)), np.tile(rgb, (2 * n, 1)), faces)


# ---------------------------------------------------------------- fixture: E9 run 003 as a recorded patch stream

ART = Path("/Users/adam/Desktop/panoptes-public/research-notes/phase2")
# When each layer lands in the FAST-BUILD-SPEC section 4 schedule (all [E]); a commit takes 1.5 s (E8b: 1.4-2.8 s).
SIMULATED = {"video": [2.5], "cameras": [18.5], "room": [18.5, 18.5], "people": [18.5], "objects": [28.0], "outlines": [28.0], "events": [28.0],
             "models": [60.0 + 7.0 * k for k in range(9)], "splat": [150.0]}


def align_sim3(c2w_src, c2w_dst):
    """dst ~ s R src + t, R from the camera orientations. Copy of m3_exp_geometry.align_sim3 (branch m3/fu-e9-onegpu);
    ponytail: import it once fb/build has that module."""
    import numpy as np
    u, _, vt = np.linalg.svd(sum(d @ q.T for d, q in zip(c2w_dst[:, :3, :3], c2w_src[:, :3, :3])))
    r = u @ np.diag([1, 1, np.sign(np.linalg.det(u @ vt))]) @ vt
    x, y = c2w_src[:, :3, 3], c2w_dst[:, :3, 3]
    xc, yc = x - x.mean(0), y - y.mean(0)
    s = float(((xc @ r.T) * yc).sum() / (xc ** 2).sum())
    return s, r, y.mean(0) - s * r @ x.mean(0)


def _quat_mul(a, b):
    w1, x1, y1, z1 = a.T
    w2, x2, y2, z2 = b.T
    import numpy as np
    return np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], 1)


def splat_sim3(data, s, r, t):
    """splat32 records (xyz, scale f32; rgba, quaternion wxyz u8) moved by x -> s R x + t."""
    import numpy as np
    from scipy.spatial.transform import Rotation
    rec = np.frombuffer(data, np.uint8).reshape(-1, 32).copy()
    f = rec[:, :24].view("<f4")
    f[:, :3] = f[:, :3] @ (s * r).T + t
    f[:, 3:6] *= s
    q = (rec[:, 28:32].astype(float) - 128) / 128
    x, y, z, w = Rotation.from_matrix(r).as_quat()
    q = _quat_mul(np.array([[w, x, y, z]]), q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-9))
    rec[:, 28:32] = np.clip(np.round(q * 128 + 128), 0, 255).astype(np.uint8)
    return rec.tobytes()


class _Clock:
    def __init__(self):
        self.t0_unix = time.time()


def fixture_e9(e9, out, report="fb-fixture-me340-e9-003"):
    """E9 run 003 (2 GPUs, run 3) as the fast report would stream it: real geometry, objects, people and events; the delivered
    report's clip, E6b outlines, 9 SAM 3D models and splat moved into E9's walk frame by a camera Sim3; times simulated (SIMULATED)."""
    import gzip
    import numpy as np
    import open3d as o3d
    import trimesh
    e9, out = Path(e9), Path(out)
    run = json.loads((e9 / "2gpu.json").read_text())["runs"][-1]
    fps, shots = run["fps"], run["shots"]
    w = Writer(None, report, _Clock(), root=out)
    labels = ["尺度为估计值（地面 + 假设 1.6 m 相机高）", "DA3-GIANT 研究许可（CC BY-NC）"]

    video = (ART / "data/clips/me340-165/source-full.mp4").read_bytes()
    w.put("video", {"fps": fps, "frames": run["frames"], "width": 1280, "height": 720, "sha256": hashlib.sha256(video).hexdigest(), "window_s": [165, 195]},
          {"video": (video, {"mediaType": "video/mp4"})}, "observed")
    cams = [{"index": i, "keys": s["keyframes"], "times": [k / fps for k in s["keyframes"]], "c2w": s["c2w_m"], "K": s["K_504x280"], "wh": [504, 280],
             "source_wh": [1280, 720], "mpu": s["metres_per_unit"], "floor": {"normal": s["floor"]["normal"], "point_m": s["floor"]["point_m"]}}
            for i, s in enumerate(shots)]
    w.put("cameras", {"shots": cams, "scale": {"source": shots[0]["scale_source"], "status": "estimated"}, "license": "DA3-GIANT-1.1 CC BY-NC (research)"},
          status="estimated", labels=labels)
    room_blobs, room, light_blobs, light = {}, [], {}, []
    for i in range(len(shots)):
        mesh = o3d.io.read_triangle_mesh(str(e9 / f"2gpu-3-balanced-shot{i}-mesh.ply"))
        arrays = [np.asarray(mesh.vertices), np.asarray(mesh.vertex_normals), np.asarray(mesh.vertex_colors), np.asarray(mesh.triangles)]
        room_blobs[f"mesh-{i}"] = pack_mesh(*arrays)
        light_blobs[f"mesh-{i}"] = pack_mesh(*(quick := light_mesh(*arrays)))
        pts = np.load(e9 / f"2gpu-3-balanced-shot{i}-points.npz")
        room_blobs[f"points-{i}"] = points_glb(pts["points_m"], pts["colors"], 0.03)
        room.append({"index": i, "triangles": len(mesh.triangles), "points": len(pts["points_m"])})
        light.append({"index": i, "triangles": len(quick[3]), "points": 0, "light_cell_m": 0.06})
    w.put("room", {"shots": light, "kind": "light"}, light_blobs, labels=labels)  # the full room follows at once, same commit
    w.put("room", {"shots": room, "kind": "full"}, room_blobs, labels=labels)
    tracks, ribbons = [], {}
    for i, s in enumerate(shots):
        for k, track in enumerate(s["people"]["tracks"]):
            tid = f"{i}-{k}"
            tracks.append({"id": tid, "shot": i, "t0": track[0]["frame"] / fps, "t1": track[-1]["frame"] / fps, "detections": len(track),
                           "points": [{"t": p["frame"] / fps, **p} for p in track]})
            ribbons[f"track-{tid}"] = ribbon([p["xyz"] for p in track], s["floor"]["normal"])
    w.put("people", {"tracks": tracks, "rules": [], "note": "快速版没有非人移动物；E9 没有跑 PeopleLoop 规则"}, ribbons, "observed+estimated", labels)
    objects = [{"id": f"obj-{i}-{k:03d}", "shot": i, **o} for i, s in enumerate(shots) for k, o in enumerate(s["objects"])]
    w.put("objects", {"objects": objects}, status="estimated+inferred", labels=["名字是检测词，未核", *labels])

    pair = json.load(gzip.open(ART / "runs/m3-fu-e6b-outlines-002/da3-outlines-step5.json.gz"))["pair"]["frames"]
    frames = [{"timeSec": f["frame"] / fps, "endTimeSec": (pair[n + 1]["frame"] if n + 1 < len(pair) else run["frames"]) / fps, "sourceFrame": f["frame"],
               "objects": [{"entityId": None, "label": f"E6b #{o['id']}", "polygons": o["polygons"], "source": f["source"]} for o in f["objects"]]}
              for n, f in enumerate(pair)]
    analysis = json.dumps({"width": 1280, "height": 720, "frames": frames}, separators=(",", ":")).encode()
    w.put("outlines", {"note": "fixture: E6b pair outlines of the delivered report's objects, not E9's"},
          {"analysis": (analysis, {"mediaType": "application/json"})}, "observed/estimated")
    windows = []
    for e in run["events"]:
        start, end = e["text"].find("{"), e["text"].rfind("}")
        parsed = json.loads(e["text"][start:end + 1])
        windows.append({"t0": e["t0"], "t1": e["t1"], "caption": parsed.get("caption"), "events": parsed.get("events", [])})
    w.put("events", {"model": "Qwen3-VL-8B (vLLM)", "windows": windows}, status="inferred", labels=["事件由 VLM 给出，未核"])

    walk = shots[1]
    droid = np.load(ART / "runs/droid-me340-165-171/prediction.npz")["poses_c2w"]
    s, r, t = align_sim3(droid[walk["keyframes"]].astype(float), np.asarray(walk["c2w_m"], float))
    sim3 = np.eye(4)
    sim3[:3, :3], sim3[:3, 3] = s * r, t
    merged = ART / "runs/me340-object-models-303-merged"
    chosen = [k for k, v in json.loads((merged / "merge.json").read_text())["choice"].items() if v == "sam3d"]
    walk_objects = [o for o in objects if o["shot"] == 1]
    models, model_blobs = [], {}
    for name in sorted(chosen):
        m = trimesh.load(merged / "models" / name / "model.glb", force="mesh", process=False)
        m.apply_transform(sim3)
        o3 = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(m.vertices), o3d.utility.Vector3iVector(m.faces))
        o3.vertex_colors = o3d.utility.Vector3dVector(np.asarray(m.visual.vertex_colors)[:, :3] / 255.)
        o3 = o3.simplify_quadric_decimation(40000)
        light = trimesh.Trimesh(np.asarray(o3.vertices), np.asarray(o3.triangles), vertex_colors=(np.asarray(o3.vertex_colors) * 255).astype(np.uint8), process=False)
        centre = light.bounds.mean(0)
        near = min(walk_objects, key=lambda o: np.linalg.norm(np.subtract(o["centroid_m"], centre)))
        gate = json.loads((merged / "models" / name / "validation.json").read_text())
        models.append({"object": near["id"], "transform": {"position": [0, 0, 0], "quaternion": [0, 0, 0, 1], "scale": [1, 1, 1]},
                       "bounds": {"min": light.bounds[0].tolist(), "max": light.bounds[1].tolist()},
                       "gate": {"accepted": gate["accepted_source_consistency"], "silhouette_iou": gate["silhouette_iou"], "source": f"fixture: delivered {name}"},
                       "distance_to_object_m": float(np.linalg.norm(np.subtract(near["centroid_m"], centre)))})
        model_blobs[f"model-{near['id']}"] = (light.export(file_type="glb"), {"mediaType": "model/gltf-binary", "format": "glb"})
        w.put("models", {"models": list(models)}, dict(model_blobs), "generated", ["生成的显示层，不用于测量"])
    raw = (ART / "runs/me340-splat-232/splats.splat").read_bytes()
    w.put("splat", {"format": "splat32", "count": len(raw) // 32, "shot": 1, "kind": "preview", "train_s": None, "steps": None,
                    "note": "fixture: the delivered splat (run 232) moved into E9's walk frame by the camera Sim3"},
          {"splat": (splat_sim3(raw, s, r, t), {"mediaType": "application/octet-stream", "format": "splat32"})}, "generated", ["生成的显示层，不用于测量"])
    w.close()
    for event in w.events():
        mirror(event, out)
    # Simulated arrival times (SIMULATED) replace the few ms this took; a timing patch takes the time of the layer before it.
    folder, at, written = out / "reports" / report, {}, {}
    paths = sorted((folder / "patches").glob("*.json"))
    for path in paths:
        p = json.loads(path.read_text())
        at[p["seq"]] = SIMULATED[p["layer"]][p["version"] - 1] if p["layer"] != "timing" else at[p["seq"] - 1]
        written[str(p["seq"])] = round(at[p["seq"]] + 1.5, 3)
    # E9's own stages and device peaks (it logged run peaks only, so no stage has a peak of its own).
    where = {"geo": "gpu0", "seg": "gpu1", "ev": "gpu1 (vLLM)", "cpu": "cpu"}
    peaks = run["vram"]["device_used_peak_gb"]
    timing = {"schema": "panoptes-fast-run-v1", "report": report, "fixture": "E9 run 003 stages; simulated layer times",
              "hardware": {"gpus": json.loads((e9 / "2gpu.json").read_text())["boot"]["gpus"]},
              "boot": json.loads((e9 / "2gpu.json").read_text())["boot"],
              "stages": [{**st, "where": where.get(st["stage"].split(":")[0].split(" ")[0], "other"), "peak_gb": None, "over_90": None} for st in run["stages"]],
              "gpu_peak": [{"gpu": g, "total_gb": 80, "peak_gb": v, "at_s": None} for g, v in enumerate(peaks)], "flags": []}
    for path in paths:
        p = json.loads(path.read_text())
        p["queued_s"], p["sent_s"] = round(at[p["seq"]] - .2, 3), at[p["seq"]]
        if p["layer"] == "timing":
            p["data"] = {"layers": [{**row, "queued_s": round(at[row["seq"]] - .2, 3), "sent_s": at[row["seq"]], "written_s": written[str(row["seq"])]}
                                    for row in p["data"]["layers"]], "run": timing}
        _write_json(path, p)
    _write_json(folder / "written.json", written)
    _write_json(folder / "fixture.json", {"source": str(e9), "simulated_times": SIMULATED, "commit_s": 1.5,
                                          "sim3_droid_to_walk": {"s": s, "R": r.tolist(), "t": t.tolist()}})
    return report


# ---------------------------------------------------------------- the recording through a real Writer on the Volume (Modal)

def modal_check(src, report, out, port=None, speed=1.0, as_report=None):
    """Ephemeral: a CPU container puts the recording's layers into a Writer on the Volume at their recorded times; this machine
    mirrors what it yields (and serves it, with port). Measures write+commit, sent, written and (with a viewer open) served."""
    import modal
    patches, blobs, _ = recording(src, report)
    feed = [(p["sent_s"] / speed, p["layer"], p["data"], {role: (blobs[r["sha256"]], {k: v for k, v in r.items() if k not in ("sha256", "bytes")})
                                                          for role, r in p["blobs"].items()}, p["status"], p["labels"]) for p in patches if p["layer"] != "timing"]
    app = modal.App("panoptes-fb-layers-check")
    image = modal.Image.debian_slim(python_version="3.12").add_local_python_source("fast_report")
    volume = modal.Volume.from_name(VOLUME, create_if_missing=True)

    @app.function(image=image, volumes={"/layers": volume}, cpu=2.0, memory=4096, timeout=900, max_containers=1, serialized=True)  # a generator takes no retries (none by default)
    def push(feed, report_id, client_has):
        import platform
        from fast_report.layers import Writer, _Clock, VOLUME as name
        import modal as m
        clock = _Clock()
        w = Writer(m.Volume.from_name(name), report_id, clock, client_has=client_has)

        def run():
            for at, layer, data, blobs, status, labels in feed:
                time.sleep(max(0., clock.t0_unix + at - time.time()))
                w.put(layer, data, blobs, status, labels)
            w.close()
        threading.Thread(target=run, daemon=True).start()
        yield {"type": "host", "report": report_id, "t0_unix": clock.t0_unix, "cpu": platform.processor() or platform.machine()}
        yield from w.events()

    new = as_report or f"fb-check-{int(time.time())}"
    own = [r["sha256"] for p in patches if p["layer"] == "video" for r in p["blobs"].values()]  # the CLI holds the MP4 it sent
    for sha in own:
        put_blob(out, blobs[sha])
    server = serve(out, port) if port else None
    log, start = [], time.time()
    with modal.enable_output(), app.run():
        for event in push.remote_gen(feed, new, own):
            got = time.time()
            if event["type"] in ("patch", "written", "run"):
                mirror(event, out)
            row = {k: event[k] for k in ("type", "seqs", "written_s", "commit_s", "t0_unix", "layer", "error") if k in event}
            if event["type"] == "patch":
                row.update(seq=event["patch"]["seq"], layer=event["patch"]["layer"], sent_s=event["patch"]["sent_s"],
                           blob_bytes=sum(map(len, event["blobs"].values())))
            row["received_unix"] = got
            log.append(row)
            print(json.dumps(row))
    _write_json(Path(out) / "reports" / new / "check.json", {"source": [str(src), report], "speed": speed, "events": log, "local_wall_s": time.time() - start})
    return new  # a server (port) keeps serving until this process ends


# ---------------------------------------------------------------- self-check

def self_check():
    import urllib.error
    import tempfile
    import urllib.request
    with tempfile.TemporaryDirectory() as volume, tempfile.TemporaryDirectory() as local:
        clock = _Clock()
        video = b"\0\0\0\x18ftypmp42" + b"x" * 100
        w = Writer(None, "r1", clock, root=volume, client_has={hashlib.sha256(video).hexdigest()})
        put_blob(local, video)  # the CLI already holds its own MP4
        w.put("video", {"fps": 30}, {"video": (video, {"mediaType": "video/mp4"})}, "observed")
        mesh = pack_mesh([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 0, 1]] * 3, [[1, 1, 1]] * 3, [[0, 1, 2]])
        w.put("room", {"shots": [{"index": 0}]}, {"mesh-0": mesh})
        w.put("models", {"models": [1]}, {"model-a": (b"glTF-a", {})})
        w.put("models", {"models": [1, 2]}, {"model-a": (b"glTF-a", {}), "model-b": (b"glTF-b", {})})
        w.close()
        events = list(w.events())
        patches = [e for e in events if e["type"] == "patch"]
        assert not [e for e in events if e["type"] == "error"], events
        assert [p["patch"]["layer"] for p in patches if p["patch"]["layer"] != "timing"] == ["video", "room", "models", "models"]
        assert [p["patch"]["version"] for p in patches if p["patch"]["layer"] == "models"] == [1, 2]
        assert [p["patch"]["seq"] for p in patches] == list(range(1, len(patches) + 1)), "seqs follow write order"
        sent = [sha for p in patches for sha in p["blobs"]]
        assert len(sent) == len(set(sent)) == 3, "each blob crosses once; the client's own MP4 never"
        assert all(e["written_s"] >= max(p["patch"]["sent_s"] for p in patches if p["patch"]["seq"] in e["seqs"]) for e in events if e["type"] == "written")
        assert {s for e in events if e["type"] == "written" for s in e["seqs"]} == {p["patch"]["seq"] for p in patches}
        for e in events:
            mirror(e, local)
        room = next(p for p in patches if p["patch"]["layer"] == "room")
        bad = dict(room, blobs={next(iter(room["blobs"])): b"tampered"})
        try:
            mirror(bad, local)
            raise AssertionError("a tampered blob must be refused")
        except ValueError:
            pass
        server = serve(local, 0)
        port = server.server_address[1]
        get = lambda path: urllib.request.urlopen(f"http://127.0.0.1:{port}{path}").read()
        first = json.loads(get("/fast/reports/r1/patches?after=0"))
        assert len(first["patches"]) == len(patches) and set(first["served"]) == {str(p["patch"]["seq"]) for p in patches}
        assert set(first["written"]) == set(first["served"])
        assert json.loads(get("/fast/reports/r1/patches?after=%d" % patches[-1]["patch"]["seq"]))["patches"] == []
        ref = next(p for p in first["patches"] if p["layer"] == "room")["blobs"]["mesh-0"]
        assert hashlib.sha256(get(f"/fast/blobs/{ref['sha256']}")).hexdigest() == ref["sha256"]
        assert ref["byteLayout"]["vertexCount"] == 3 and ref["byteLayout"]["indexCount"] == 3
        try:  # no report container attached: no on-demand route
            get("/fast/reports/r1/click?i=0&x=1&y=2")
            raise AssertionError("the click route needs a container")
        except urllib.error.HTTPError as e:
            assert e.code == 404
        server.shutdown()
        calls = []
        server = serve(local, 0, click=lambda r, i, x, y: calls.append((r, i, x, y)) or {"status": "card"}, alive=lambda: 1.)
        port = server.server_address[1]
        assert json.loads(get("/fast/reports/r1/click?i=3&x=10.5&y=20")) == {"status": "card"} and calls == [("r1", 3, 10.5, 20.)]
        assert json.loads(get("/fast/alive")) == {"alive": 1.}
        server.shutdown()
        replayed = replay(local, "r1", local, as_report="r2", speed=100)
        assert len(list((Path(local) / "reports" / replayed / "patches").glob("*.json"))) == len(patches)
        class SlowVolume:  # a 0.3 s commit
            commits = 0

            def commit(self):
                time.sleep(.3)
                self.commits += 1
        slow = SlowVolume()
        w = Writer(slow, "r4", clock, root=volume)
        for layer in ("cameras", "room", "people"):
            w.put(layer, {}, {"b": (layer.encode() * 1000, {})})
        w.close()
        assert [len(e["seqs"]) for e in w.events() if e["type"] == "written"] == [4], "layers put together share one commit"
        slow = SlowVolume()
        w = Writer(slow, "r3", clock, root=volume)
        w.put("a", {})
        time.sleep(.1)
        w.put("b", {})
        w.close()
        events = list(w.events())
        sent = {e["patch"]["layer"]: e["patch"]["sent_s"] for e in events if e["type"] == "patch"}
        written = [e for e in events if e["type"] == "written"]
        assert sent["b"] < written[0]["written_s"], "a layer arriving during a commit is sent at once"
        assert slow.commits == len(written) == 2 and sum(len(e["seqs"]) for e in written) == 4  # a, b and a timing patch each
        g = points_glb([[0, 0, 0], [1, 1, 1]], [[255, 0, 0], [0, 255, 0]], 0.03)[0]
        assert g[:4] == b"glTF" and struct.unpack("<I", g[8:12])[0] == len(g)
        assert len(ribbon([[0, 0, 0], [1, 0, 0], [2, 0, 0]], [0, -1, 0])[0]) == 6 * 36 + 4 * 3 * 4
    print("layers self-check passed")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("self-check")
    sv = sub.add_parser("serve")
    sv.add_argument("root")
    sv.add_argument("--port", type=int, default=8793)
    sv.add_argument("--replay", nargs=2, metavar=("SRC", "REPORT"))
    sv.add_argument("--as", dest="as_report")
    sv.add_argument("--speed", type=float, default=1.0)
    fx = sub.add_parser("fixture-e9")
    fx.add_argument("e9")
    fx.add_argument("out")
    mc = sub.add_parser("modal-check")
    mc.add_argument("src")
    mc.add_argument("report")
    mc.add_argument("out")
    mc.add_argument("--port", type=int)
    mc.add_argument("--speed", type=float, default=1.0)
    mc.add_argument("--as", dest="as_report")
    a = ap.parse_args()
    if a.cmd == "self-check":
        self_check()
    elif a.cmd == "serve":
        server = serve(a.root, a.port)
        print(f"serving {a.root} on http://127.0.0.1:{a.port}/fast/")
        if a.replay:
            name = a.as_report or f"{a.replay[1]}-replay-{int(time.time())}"
            print(f"viewer: http://127.0.0.1:5173/app.html#/live/{name}", flush=True)
            replay(a.replay[0], a.replay[1], a.root, name, a.speed)
            print("replay done", flush=True)
        threading.Event().wait()
    elif a.cmd == "fixture-e9":
        print(fixture_e9(a.e9, a.out))
    elif a.cmd == "modal-check":
        print(modal_check(a.src, a.report, a.out, a.port, a.speed, a.as_report))
        print("replay done", flush=True)
        if a.port:
            threading.Event().wait()
