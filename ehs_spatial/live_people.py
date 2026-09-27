"""Live people/mover loop over the phone's posed RGB-D stream: timestamped 3D person tracks and rule findings.

The stream contract is ehs_spatial/phone_stream.py, the same messages scripts/live_map.py integrates. decode_frame turns
a message into the frame step() takes. Only a frame phone_stream.credible() passes (the map's own rule), in the
worldOriginEpoch the floor and zone were drawn in, with rgb and depth, covers anything; any other frame is a coverage gap
(NO_DATA), never an empty scene. Feet go through video.lift_foot and rules through video.judge_frame, so this loop and
the offline video tier cannot disagree about a rule. State is bounded: speed samples older than two baselines, tracks unseen for
TRACK_KEEPALIVE_S and latencies beyond LATENCY_WINDOW frames are dropped.

  python -m ehs_spatial.live_people --self-check
"""

import base64
import io
import json
import time
from collections import Counter, deque
from collections.abc import Callable, Iterable
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from shapely.geometry import Polygon, box

from .phone_stream import credible, depth_mm, header, pack, unpack
from .providers.sam3 import SAM3_ENDPOINT, decode_coco_rle, sam_backend_revision, sam_subscribe
from .video import (
    BAND_M, COST_PER_SAM_CALL_USD, MAX_HUMAN_SPEED_MPS, MODAL_L4_USD_PER_S, MODAL_SCALEDOWN_S, NO_DATA, RULE_NAMES,
    SAM_SPEND, SPEED_BASELINE_S, BudgetExhausted, HourlySpendGuard, frame_health, image_region, judge_frame,
    lift_foot, track_speed, zone_floor_points, zone_floor_seen, zone_prism_points,
)

TRACK_KEEPALIVE_S = 5.0  # a track unseen this long is closed; a returning person gets a new id
LATENCY_WINDOW = 36000  # the last hour of frames at 10 Hz


def decode_frame(message: bytes) -> dict:
    """One stream message -> the frame step() takes: t (t_capture), frame (seq), streamGap (phone_stream.credible's
    reason, None when credible), rgb (HxWx3), K (3x3), cameraToWorld (4x4 metres), depth (metres on its own raster, 0
    where missing or below high confidence) and the header's trackingState, trackingStateReason and worldOriginEpoch. A
    blob that does not decode stays None, so the frame becomes a gap."""
    head, rgb, depth, confidence = unpack(message)
    gap, c2w, image = credible(head, rgb)
    fx, fy, cx, cy = head["K"] if gap is None else (np.nan,) * 4
    frame = {"t": float(head["t_capture"]), "frame": head.get("seq"), "streamGap": gap, "rgb": image, "depth": None,
             "K": np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]]), "cameraToWorld": c2w,
             **{key: head.get(key) for key in ("trackingState", "trackingStateReason", "worldOriginEpoch")}}
    millimetres = depth_mm(depth, confidence)
    if millimetres is not None:
        frame["depth"] = millimetres.astype(np.float32) / 1000.0
    return frame


def on_raster(depth: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Depth at each rgb pixel centre, nearest: the two rasters share one field of view."""
    if depth.shape == shape:
        return depth
    rows = ((np.arange(shape[0]) + 0.5) * depth.shape[0] / shape[0]).astype(int)
    cols = ((np.arange(shape[1]) + 0.5) * depth.shape[1] / shape[1]).astype(int)
    return depth[np.ix_(rows, cols)]


class Sam3Detector:
    """SAM 3 text-prompt detection through the providers switch (SAM3_BACKEND), under the hourly spend guard for the
    backends that bill: fal per call (booked before it), our Modal L4 by the second (booked after it, and refused up front
    once the guard is spent). http is our own GPU server, which nobody bills per call: the guard never charges or stops
    it. Returns None when the detector cannot answer and raises BudgetExhausted when the guard has nothing left: both are
    gaps."""

    def __init__(self, labels=("person",), subscriber: Callable = sam_subscribe, guard: HourlySpendGuard | None = None,
                 clock: Callable[[], float] = time.monotonic):
        self.labels, self.subscriber, self.guard, self.clock = labels, subscriber, guard or SAM_SPEND, clock
        self.backend, self.calls, self.seconds, self.last_end = sam_backend_revision(), 0, deque(maxlen=LATENCY_WINDOW), None

    def __call__(self, frame: dict) -> list[dict] | None:
        buffer = io.BytesIO()
        Image.fromarray(frame["rgb"]).save(buffer, "JPEG", quality=92)
        uri = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()
        height, width = frame["rgb"].shape[:2]
        backend = self.backend["backend"]
        modal = backend == "modal"
        found = []
        for label in self.labels:
            if backend != "http":  # anything but our own server bills: fal (and a backend we do not know) per call
                self.guard.charge(0.0 if modal else COST_PER_SAM_CALL_USD)
            started = self.clock()
            try:
                response = self.subscriber(SAM3_ENDPOINT, arguments={
                    "image_url": uri, "prompt": label, "return_multiple_masks": True,
                    "include_scores": True, "include_boxes": True, "max_masks": 32})
            except Exception:
                return None
            finally:
                ended = self.clock()
                if modal:  # one container is billed while up: this call plus the idle time since the last one
                    idle = 0.0 if self.last_end is None else min(started - self.last_end, MODAL_SCALEDOWN_S)
                    self.guard.record((ended - started + idle) * MODAL_L4_USD_PER_S)
                self.last_end = ended
            self.calls += 1
            self.seconds.append(ended - started)
            if response.get("error"):
                return None
            rles = response.get("rle") or []
            scores = response.get("scores") or []
            for ordinal, rle in enumerate([rles] if isinstance(rles, str) else rles):
                mask = decode_coco_rle(rle, height=height, width=width).astype(bool)
                found.append({"label": label, "mask": mask,
                              "score": float(scores[ordinal]) if ordinal < len(scores) else None})  # none sent: none made up
        return found


class PeopleLoop:
    """One camera's people/mover state. step() takes one decoded stream frame and returns
    (track rows, finding rows); findings are emitted when a rule's verdict changes."""

    def __init__(self, floor_point, floor_up, scale_record: dict, detector: Callable,
                 zone: Polygon | None = None, person_labels=("person",), world_epoch: int = 0) -> None:
        """floor_point, floor_up and zone are in world metres of world_epoch, the frame cameraToWorld maps into;
        scale_record says whether those metres were measured (device) or assumed, for the scale gate."""
        self.p0 = np.asarray(floor_point, float)
        self.up = np.asarray(floor_up, float) / np.linalg.norm(floor_up)
        axis = np.eye(3)[np.argmin(np.abs(self.up))]
        self.e1 = axis - (axis @ self.up) * self.up
        self.e1 /= np.linalg.norm(self.e1)
        self.e2 = np.cross(self.up, self.e1)
        self.scale, self.detector, self.zone, self.world_epoch = scale_record, detector, zone, world_epoch
        self.zone_floor = zone_floor_points(zone) if zone is not None else None
        self.zone_prism = zone_prism_points(zone) if zone is not None else None
        self.person_labels = set(person_labels)
        self.tracks: dict[str, dict] = {}  # id -> {kind, last_t, last_xy, samples (accepted, the last 2 baselines)}
        self.next_id, self.previous_gray, self.last_verdict = 0, None, {}
        self.latencies: deque = deque(maxlen=LATENCY_WINDOW)  # (gap or None, loop s without detector, detector s or None)
        self.gaps: Counter = Counter()

    def floor_xy(self, world: np.ndarray) -> tuple[float, float]:
        offset = np.asarray(world) - self.p0
        return float(offset @ self.e1), float(offset @ self.e2)

    def camera_points(self, floor_points: np.ndarray, rotation: np.ndarray, centre: np.ndarray) -> np.ndarray:
        """(n, 3) floor (x, y, height above the floor) -> camera frame."""
        x, y, h = floor_points.T
        world = self.p0 + np.outer(x, self.e1) + np.outer(y, self.e2) + np.outer(h, self.up)
        return (world - centre) @ rotation

    def _associate(self, kind: str, xy: tuple[float, float], t: float, taken: set) -> str:
        """Greedy nearest live track this detection could have walked to since last seen."""
        best, best_distance = None, None
        for track_id, track in self.tracks.items():
            if track["kind"] != kind or track_id in taken:
                continue
            distance = np.hypot(xy[0] - track["last_xy"][0], xy[1] - track["last_xy"][1])
            if distance <= 2 * BAND_M + MAX_HUMAN_SPEED_MPS * (t - track["last_t"]) and (
                    best_distance is None or distance < best_distance):
                best, best_distance = track_id, distance
        if best is None:
            best = f"{kind}-{self.next_id}"
            self.next_id += 1
            self.tracks[best] = {"kind": kind, "samples": deque()}
        self.tracks[best].update(last_t=t, last_xy=xy)
        taken.add(best)
        return best

    def _gap(self, frame: dict) -> str | None:
        """Why this frame cannot cover a rule before anything is detected; resets state a bad pose would poison."""
        if frame["streamGap"]:
            for track in self.tracks.values():
                track["samples"].clear()  # a pose that drifted and then snapped back must not feed a speed
            return frame["streamGap"]
        if frame["worldOriginEpoch"] != self.world_epoch:  # an int: credible() refused anything else
            self.tracks.clear()  # positions from another world origin are not comparable
            return f"world origin epoch {frame['worldOriginEpoch']}; floor and zone are in {self.world_epoch}"
        missing = [name for name in ("rgb", "depth") if frame.get(name) is None]
        return f"no {' or '.join(missing)}" if missing else None

    def step(self, frame: dict) -> tuple[list[dict], list[dict]]:
        started = time.perf_counter()
        t, index = float(frame["t"]), frame.get("frame")
        self.tracks = {k: v for k, v in self.tracks.items() if t - v["last_t"] <= TRACK_KEEPALIVE_S}
        gap = self._gap(frame)
        if frame.get("rgb") is not None:
            gray = np.asarray(frame["rgb"], np.float32).mean(axis=2)
            gap = gap or frame_health(gray, self.previous_gray)
            self.previous_gray = gray
        detections, detector_s = None, None
        if gap is None:
            detector_started = time.perf_counter()
            try:
                detections = self.detector(frame)
                gap = "person detector offline" if detections is None else None
            except BudgetExhausted as error:
                gap = f"detector budget exhausted ({error})"
            detector_s = time.perf_counter() - detector_started
        rows, persons, movers, speeds, zone_seen = [], [], [], [], False
        if gap is None:
            c2w = np.asarray(frame["cameraToWorld"], float)
            rotation, centre = c2w[:3, :3], c2w[:3, 3]
            K = np.asarray(frame["K"], float)
            height, width = frame["rgb"].shape[:2]
            depth = on_raster(np.asarray(frame["depth"], np.float32), (height, width))
            plane = (rotation.T @ self.up, float(self.up @ (centre - self.p0)))
            region = None
            if self.zone is not None:
                region = image_region(self.camera_points(self.zone_prism, rotation, centre), K, (width, height))
                points, cells = self.zone_floor
                zone_seen = zone_floor_seen(self.camera_points(points, rotation, centre), cells, K, depth, (width, height))
            exclude = np.any([d["mask"] for d in detections], axis=0) if detections else None
            taken: set = set()
            for detection in detections:
                ys, xs = np.nonzero(detection["mask"])
                if not len(ys):
                    continue  # an empty mask is no detection; a small one is kept for review below
                bbox = (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))
                person = detection["label"] in self.person_labels
                foot_u = float(np.median(xs[ys >= ys.max() - max(2, 0.03 * (ys.max() - ys.min()))]))  # lowest mask pixels
                foot, implied, review = lift_foot(bbox, (width, height), K, plane, depth, exclude, 1.0, person, foot_u)
                # A detection that cannot be placed stays a person: it can only ask for review, and for R1 only if it
                # overlaps where someone in the zone could appear.
                entry = {"xy": None, "review": review, "near_zone": region is None or region.intersects(box(*bbox))}
                foot_world = None
                if foot is not None:
                    foot_world = rotation @ foot + centre
                    entry["xy"] = self.floor_xy(foot_world)
                    if not person:  # near-side ground edge of a mover, for person-to-mover distance
                        rays = np.linalg.inv(K) @ np.array([[bbox[0], bbox[2]], [bbox[3], bbox[3]], [1.0, 1.0]])
                        ends = rays * (-plane[1] / (plane[0] @ rays))
                        entry["edge"] = tuple(self.floor_xy(rotation @ end + centre) for end in ends.T)
                visible = detection["mask"] & (depth > 0)
                centroid = None
                if visible.any():  # median visible-surface point, the offline person layer's own measure
                    vs, us = np.nonzero(visible)
                    points = (np.c_[us, vs, np.ones(len(us))] @ np.linalg.inv(K).T) * depth[visible][:, None]
                    centroid = (rotation @ np.median(points, axis=0) + centre).tolist()
                # Identity follows the visible body: a hidden foot lifts far away, the torso does not.
                kind = "person" if person else "mover"
                anchor = self.floor_xy(centroid) if centroid else entry["xy"]
                track_id = self._associate(kind, anchor, t, taken) if anchor is not None else None
                speed = None
                if person:
                    if review:
                        speeds.append((None, "unaccepted foot"))
                    else:
                        samples = self.tracks[track_id]["samples"]
                        samples.append((t, entry["xy"]))
                        while samples[0][0] < t - 2 * SPEED_BASELINE_S:  # older samples never enter a speed window
                            samples.popleft()
                        speed, reason = track_speed(samples, t)
                        speeds.append((speed, reason))
                    persons.append(entry)
                else:
                    movers.append(entry)
                rows.append({"t": round(t, 4), "frame": index, "track": track_id, "label": detection["label"],
                             "source": detection.get("source"), "score": detection.get("score"),
                             "footWorld": None if foot_world is None else [round(v, 5) for v in foot_world],
                             "floorXyM": None if entry["xy"] is None else [round(v, 4) for v in entry["xy"]],
                             "centroidWorld": centroid,
                             "centroidXyM": [round(v, 4) for v in self.floor_xy(centroid)] if centroid else None,
                             "impliedHeightM": None if implied is None else round(implied, 3),
                             "accepted": not review, "review": review, "nearZone": entry["near_zone"],
                             "speedMps": None if speed is None else round(speed, 3)})
        result = judge_frame(persons, movers, self.zone, gap, zone_seen, speeds, self.scale)
        loop_s = time.perf_counter() - started - (detector_s or 0.0)
        self.latencies.append((gap, loop_s, detector_s))
        if gap is not None:
            self.gaps[gap] += 1
        findings = []
        for rule in RULE_NAMES:
            outcome = result[rule]
            state = (outcome["verdict"], outcome.get("raw"))  # a pre-gate change is news too
            if self.last_verdict.get(rule) != state:
                self.last_verdict[rule] = state
                findings.append({"t": round(t, 4), "frame": index, "rule": rule, "verdict": outcome["verdict"],
                                 "beforeScaleGate": outcome.get("raw"), "value": outcome.get("value"),
                                 "reason": outcome.get("reason"), "scaleStatus": self.scale["status"],
                                 "tracks": sorted({r["track"] for r in rows if r["track"]}),
                                 "latencyS": round(loop_s + (detector_s or 0.0), 4)})
        return rows, findings


def latency_summary(loop: PeopleLoop) -> dict:
    """p50/p95 seconds over the loop's latency window: covered frames without the detector, gap frames, the detector."""
    def stats(values):
        return {"n": len(values), "p50": round(float(np.percentile(values, 50)), 4) if values else None,
                "p95": round(float(np.percentile(values, 95)), 4) if values else None}
    return {"covered_frames_without_detector": stats([s for gap, s, _ in loop.latencies if gap is None]),
            "gap_frames": stats([s for gap, s, _ in loop.latencies if gap is not None]),
            "detector": stats([d for _, _, d in loop.latencies if d is not None])}


def run(frames: Iterable[dict], loop: PeopleLoop, output: Path) -> dict:
    """Drive the loop over a stream, appending tracks.jsonl and findings.jsonl as frames arrive."""
    output.mkdir(parents=True, exist_ok=True)
    counts = {"frames": 0, "track_rows": 0, "findings": 0}
    with open(output / "tracks.jsonl", "w") as tracks, open(output / "findings.jsonl", "w") as findings:
        for frame in frames:
            rows, found = loop.step(frame)
            counts["frames"] += 1
            counts["track_rows"] += len(rows)
            counts["findings"] += len(found)
            tracks.writelines(json.dumps(row) + "\n" for row in rows)
            findings.writelines(json.dumps(row) + "\n" for row in found)
    counts["gaps"] = sum(loop.gaps.values())
    counts["gap_reasons"] = dict(loop.gaps)
    counts["loop_latency_s"] = latency_summary(loop)
    return counts


# ------------------------------------------------------------------ self-check
_SEEN_ZONE = Polygon([(-1, 5), (1, 5), (1, 7), (-1, 7)])  # floor metres (x, z) in front of the camera


def _message(t, rng, camera_height=1.6, people=(), walls=(), tracking="normal", epoch=0, posed=True, reason=None):
    """A synthetic ARFrame over a flat floor, camera looking along +z (K 600 px on the 640x480 raster). people:
    (x, z, top m, bottom m, half width m), the part between bottom and top visible; walls: (z, top m, u0, u1)
    fronto-parallel slabs standing on the floor. Depth goes out at 256x192 in millimetres, as LiDAR does.
    Returns (message, person masks on the rgb raster)."""
    vv, uu = np.mgrid[0:480, 0:640] + 0.5
    down = (vv - 240) / 600.0
    depth = np.where(down > 1e-3, camera_height / np.maximum(down, 1e-3), 0.0)
    for z, top, u0, u1 in walls:
        face = (vv >= 240 - 600 * (top - camera_height) / z) & ((depth == 0) | (depth > z)) & (uu >= u0) & (uu < u1)
        depth[face] = z
    masks = []
    for x, z, top, bottom, half in people:
        mask = ((vv >= 240 - 600 * (top - camera_height) / z) & (vv < 240 + 600 * (camera_height - bottom) / z)
                & (np.abs(uu - 320 - 600 * x / z) < 600 * half / z) & ((depth == 0) | (depth > z)))
        depth[mask] = z
        masks.append(mask)
    c2w = np.eye(4)
    c2w[:3, 3] = [0, -camera_height, 0]  # OpenCV world: y down, floor y = 0
    low = depth[((np.arange(192) + 0.5) * 2.5).astype(int)][:, ((np.arange(256) + 0.5) * 2.5).astype(int)]
    head = header(int(round(t * 1000)), t, t, [600.0, 600.0, 320.0, 240.0], c2w if posed else None, tracking, reason, epoch)
    rgb = cv2.imencode(".jpg", rng.integers(20, 235, (480, 640, 3), dtype=np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 90])[1]
    millimetres = cv2.imencode(".png", np.clip(np.rint(low * 1000), 0, 65535).astype(np.uint16))[1]
    confidence = cv2.imencode(".png", np.where(low > 0, 2, 0).astype(np.uint8))[1]
    return pack(head, rgb.tobytes(), millimetres.tobytes(), confidence.tobytes()), masks


def _step(loop, message_masks):
    """One synthetic frame through decode_frame, the detector answering with that frame's person masks."""
    message, masks = message_masks
    loop.detector = lambda frame: [{"label": "person", "mask": m} for m in masks]
    return loop.step(decode_frame(message))


def _r1(loop, message_masks):
    rows, found = _step(loop, message_masks)
    return {f["rule"]: f for f in found}.get("R1_zone") or {"verdict": loop.last_verdict["R1_zone"][0]}, rows


def self_check() -> None:
    """Synthetic ARFrames through the stream contract: every coverage and review rule the critics broke."""
    from .video import contract_scale

    measured = contract_scale({"scale_status": "device_metric", "metres_per_native_unit": 1.0})
    stated = {"status": "model_estimated", "nativeToMeters": 1.0}

    def fresh(zone=None, scale=measured):
        return PeopleLoop([0, 0, 0], [0, -1, 0], scale, lambda frame: [], zone)

    # 1. The sample rate never changes a verdict; depth arrives at 256x192 mm and is lifted on the rgb raster.
    verdicts = {}
    for rate in (1, 5):
        rng, loop, last = np.random.default_rng(0), fresh(Polygon([(-2, -3), (2, -3), (2, -1), (-2, -1)])), {}
        for t in np.arange(0, 8 + 1e-9, 1 / rate):
            rows, found = _step(loop, _message(float(t), rng, people=[(-1.5 + 0.4 * t, 6.0, 1.75, 0.0, 0.2)]))
            last.update({f["rule"]: f["verdict"] for f in found})
            assert rows and rows[0]["accepted"] and abs(rows[0]["impliedHeightM"] - 1.75) < 0.05, rows
        assert abs(rows[0]["speedMps"] - 0.4) < 0.05 and len(loop.tracks) == 1, rows[0]
        verdicts[rate] = last
    assert verdicts[1] == verdicts[5] and verdicts[1]["R3_speed"] == "PASS" and verdicts[1]["R2_min_distance"] == NO_DATA
    assert verdicts[1]["R1_zone"] == NO_DATA  # the zone lies behind the camera: never seen, so no PASS

    # 2. Coverage: a seen empty zone PASSes; no pose, tracking that is not normal or not sent, or another world epoch
    #    is NO_DATA; an offline detector too.
    rng, loop = np.random.default_rng(1), fresh(_SEEN_ZONE)
    assert _r1(loop, _message(0.0, rng))[0]["verdict"] == "PASS"
    for t, kwargs, reason in ((1, {"posed": False}, "no cameraToWorld"),
                              (2, {"tracking": "limited", "reason": "excessiveMotion"}, "ARKit tracking limited (excessiveMotion)"),
                              (3, {"tracking": "notAvailable"}, "ARKit tracking notAvailable"),
                              (4, {"tracking": None}, "ARKit tracking state not sent"),
                              (5, {"epoch": 1}, "world origin epoch 1; floor and zone are in 0")):
        finding = _r1(loop, _message(float(t), rng, **kwargs))[0]
        assert finding["verdict"] == NO_DATA and finding["reason"] == reason, (kwargs, finding)
        assert _r1(loop, _message(t + 0.5, rng))[0]["verdict"] == "PASS"
    offline = PeopleLoop([0, 0, 0], [0, -1, 0], measured, lambda frame: None, _SEEN_ZONE)
    assert offline.step(decode_frame(_message(0.0, rng)[0]))[1][0]["reason"] == "person detector offline"
    head, rgb, depth, _ = unpack(_message(0.0, rng)[0])
    medium = pack(head, rgb, depth, cv2.imencode(".png", np.ones((192, 256), np.uint8))[1].tobytes())
    assert not decode_frame(medium)["depth"].any()  # medium confidence over valid depth is no depth

    # 3. A zone hidden behind a wall is not seen, though its corners are in the image: NO_DATA, not PASS.
    hidden = _r1(fresh(_SEEN_ZONE), _message(0.0, rng, walls=[(4.5, 3.0, 0, 640)]))[0]
    assert hidden["verdict"] == NO_DATA and hidden["reason"] == "zone floor not observed", hidden
    #    A 5 m x 10 m zone with a 0.5 m crate at z = 12 and a crouched worker behind it the detector cannot see: the
    #    crate's floor shadow is about 4% of the zone (a 95%-seen rule PASSed it) but holds a footprint, so NO_DATA.
    large = Polygon([(-2.5, 6), (2.5, 6), (2.5, 16), (-2.5, 16)])
    crate = [(12.0, 0.7, 320 - 600 * 0.25 / 12, 320 + 600 * 0.25 / 12)]
    assert _r1(fresh(large), _message(0.0, rng))[0]["verdict"] == "PASS"
    shadow, rows = _r1(fresh(large), _message(0.0, rng, people=[(0.0, 13.0, 0.6, 0.0, 0.2)], walls=crate))
    assert not rows and shadow["verdict"] == NO_DATA and shadow["reason"] == "zone floor not observed", (shadow, rows)
    #    A zone, or a part of one, narrower than two grid cells left no 2 x 2 block of hidden cell centres, so it PASSed
    #    with none of that floor observed: a 0.15 m arm of the seen zone behind a post at z = 7.2, and whole 0.15 m and
    #    0.05 m strips behind a wall. Their outline samples count too: hidden, NO_DATA; open, PASS.
    arm = Polygon([(-1, 5), (1, 5), (1, 7), (0.15, 7), (0.15, 10), (0, 10), (0, 7), (-1, 7)])
    strip = lambda width: Polygon([(-width / 2, 5), (width / 2, 5), (width / 2, 7), (-width / 2, 7)])  # noqa: E731
    for zone, walls in ((arm, [(7.2, 3.0, 300, 340)]), (strip(0.15), [(4.5, 3.0, 0, 640)]), (strip(0.05), [(4.5, 3.0, 0, 640)])):
        assert _r1(fresh(zone), _message(0.0, rng))[0]["verdict"] == "PASS", zone
        narrow = _r1(fresh(zone), _message(0.0, rng, walls=walls))[0]
        assert narrow["verdict"] == NO_DATA and narrow["reason"] == "zone floor not observed", (zone, narrow)
    #    A zone half out of the image: the part in view is observed floor, the rest nobody saw. NO_DATA, never PASS.
    half = _r1(fresh(Polygon([(2, 5), (5, 5), (5, 7), (2, 7)])), _message(0.0, rng))[0]
    assert half["verdict"] == NO_DATA and half["reason"] == "zone floor not observed", half

    # 4. A person who cannot be placed is never nobody. In the zone: a 10 px head whose lowest pixel is above the
    #    horizon, a 10 px blob below it, and a head over a shelf (camera 1.2 m, shelf 1.25 m at z = 4.5) all make R1
    #    NEEDS_REVIEW. The same head well to the side of the zone leaves R1 to the zone floor, which is seen: PASS.
    for person, reason in (((0.0, 6.0, 1.75, 1.66, 0.1), "foot_above_horizon"), ((0.0, 6.0, 0.2, 0.1, 0.1), "box_too_small")):
        small, rows = _r1(fresh(_SEEN_ZONE), _message(0.0, rng, people=[person]))
        assert small["verdict"] == "NEEDS_REVIEW" and reason in rows[0]["review"], (small, rows)
    head = (0.0, 6.0, 1.75, 1.25, 0.2)
    shelf = fresh(_SEEN_ZONE)
    over_shelf, rows = _r1(shelf, _message(0.0, rng, 1.2, [head], [(4.5, 1.25, 0, 640)]))
    assert over_shelf["verdict"] == "NEEDS_REVIEW" and rows[0]["review"] == ["foot_above_horizon"], (over_shelf, rows)
    assert shelf.last_verdict["R3_speed"][0] == "NEEDS_REVIEW"  # nobody knows how fast an unplaced person moves
    aside, rows = _r1(fresh(_SEEN_ZONE), _message(0.0, rng, 1.2, [(-3.0, 6.0, 1.75, 1.25, 0.2)], [(4.5, 1.25, 0, 80)]))
    assert aside["verdict"] == "PASS" and rows[0]["review"] == ["foot_above_horizon"] and not rows[0]["nearZone"], rows
    #    Near means within the position band of the zone: a head 0.3 m outside it (u 445-455, past the zone's own image
    #    hull at u 440) is someone who could be in the zone, so R1 is NEEDS_REVIEW.
    beside, rows = _r1(fresh(_SEEN_ZONE), _message(0.0, rng, people=[(1.3, 6.0, 1.75, 1.66, 0.05)]))
    assert beside["verdict"] == "NEEDS_REVIEW" and rows[0]["nearZone"] and rows[0]["review"] == ["foot_above_horizon"], (beside, rows)

    # 5. An assumed scale decides nothing: a 2.5 m/s speed FAIL and a zone entry FAIL become NEEDS_REVIEW.
    fast, found = fresh(None, stated), []
    for t in np.arange(0, 2.01, 0.5):
        found += _step(fast, _message(float(t), rng, people=[(-2.5 + 2.5 * t, 6.0, 1.75, 0.0, 0.2)]))[1]
    speed = [f for f in found if f["rule"] == "R3_speed"]
    assert speed[-1]["verdict"] == "NEEDS_REVIEW" and speed[-1]["beforeScaleGate"] == "FAIL", speed
    inside = _r1(fresh(_SEEN_ZONE, stated), _message(0.0, rng, people=[(0.0, 6.0, 1.75, 0.0, 0.2)]))[0]
    assert inside["verdict"] == "NEEDS_REVIEW" and inside["beforeScaleGate"] == "FAIL", inside

    # 6. Spend: an empty guard is a recorded gap, not a crash; the default guard keeps a 1 Hz Modal stream up an hour.
    broke = fresh(_SEEN_ZONE)
    broke.detector = Sam3Detector(subscriber=lambda endpoint, arguments: {"rle": [], "scores": []},
                                  guard=HourlySpendGuard(0.02))
    broke.detector.backend = {"backend": "fal", "model_revision": "test"}
    found = [broke.step(decode_frame(_message(float(t), rng)[0]))[1] for t in range(3)]
    assert found[2][0]["verdict"] == NO_DATA and found[2][0]["reason"] == (
        "detector budget exhausted (budget cap: $0.02 of the $0.02/hour SAM spend guard spent in the last hour)"), found
    #    Our Modal L4 bills after the call, so it books nothing up front, but a spent guard refuses it before it starts;
    #    http is our own GPU server, billed by nobody per call: a spent guard neither stops nor charges it.
    asked = []

    def answer(endpoint, arguments):
        asked.append(endpoint)
        return {"rle": [], "scores": []}
    for backend in ("modal", "http"):
        spent = Sam3Detector(subscriber=answer, guard=HourlySpendGuard(0.0))
        spent.backend = {"backend": backend, "model_revision": "test"}
        try:
            assert spent({"rgb": np.zeros((8, 8, 3), np.uint8)}) == [] and backend == "http", f"{backend} ran on a spent guard"
        except BudgetExhausted:
            assert backend == "modal", "the spend guard stopped our own http server, which nobody bills per call"
            assert not asked, "the Modal call started before the guard refused it"
        assert not spent.guard.spent
    assert len(asked) == 1
    unscored = Sam3Detector(subscriber=lambda endpoint, arguments: {"rle": ['{"size": [8, 8], "counts": [0, 64]}']},
                            guard=HourlySpendGuard(1.0))
    unscored.backend = {"backend": "fal", "model_revision": "test"}
    assert unscored({"rgb": np.zeros((8, 8, 3), np.uint8)})[0]["score"] is None  # SAM sent no score: none is made up
    now = [0.0]

    def warm_call(endpoint, arguments):
        now[0] += 0.4  # a warm L4 call (M: 0.41 s)
        return {"rle": [], "scores": []}
    hour = Sam3Detector(subscriber=warm_call, guard=HourlySpendGuard(clock=lambda: now[0]), clock=lambda: now[0])
    hour.backend = {"backend": "modal", "model_revision": "test"}
    for _ in range(3600):
        hour({"rgb": np.zeros((8, 8, 3), np.uint8)})  # raises BudgetExhausted if the default guard stops the stream
        now[0] += 0.6
    assert 0.8 < hour.guard.usd_per_hour - hour.guard.remaining() < 1.0  # about $0.91 for the hour (E)

    # 7. Bounded state: a person standing for 10 s keeps two baselines of samples; tracks close after the keepalive.
    loop = fresh()
    for t in np.arange(0, 10, 0.1):
        _step(loop, _message(float(t), rng, people=[(0.0, 6.0, 1.75, 0.0, 0.2)]))
    assert len(loop.tracks) == 1 and len(next(iter(loop.tracks.values()))["samples"]) <= 2 * SPEED_BASELINE_S * 10 + 1
    for t in np.arange(10, 16, 1.0):
        _step(loop, _message(float(t), rng))
    assert not loop.tracks and loop.latencies.maxlen == LATENCY_WINDOW

    # 8. What a bad pose would poison is reset: no speed baseline spans a 'limited' frame (the first speed comes 2 s
    #    after it), and a new world epoch closes every track. Loop latency leaves the detector out.
    walker, walk = fresh(), lambda t, **kw: _message(float(t), rng, people=[(-1.0 + 0.4 * t, 6.0, 1.75, 0.0, 0.2)], **kw)
    for t in np.arange(0, 1.55, 0.1):
        _step(walker, walk(t))
    _step(walker, walk(1.6, tracking="limited", reason="excessiveMotion"))
    after = [_step(walker, walk(t))[0][0]["speedMps"] for t in np.arange(1.7, 3.65, 0.1)]
    assert after == [None] * len(after) and walker.tracks, after
    assert _step(walker, walk(3.7, epoch=1))[0] == [] and not walker.tracks
    slow = fresh(_SEEN_ZONE)
    slow.detector = lambda frame: time.sleep(0.05) or []
    for t in range(3):
        slow.step(decode_frame(_message(float(t), rng)[0]))
    spent = latency_summary(slow)
    assert spent["detector"]["p50"] >= 0.05 > spent["covered_frames_without_detector"]["p95"], spent
    print("live people check passed: stream contract decoded (mm depth at 256x192); 1 Hz = 5 Hz verdicts; no pose, "
          "limited tracking, a new world epoch, a hidden zone floor, a footprint-sized shadow in it, a hidden narrow zone "
          "or arm, a zone half out of view, an offline or broke detector are NO_DATA; unplaceable people within the band "
          "of the zone are NEEDS_REVIEW; a spent guard stops Modal before the call and never stops our own http server; "
          "assumed metres decide nothing; tracking loss and a new epoch reset tracks; state stays bounded")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    if parser.parse_args().self_check:
        self_check()
