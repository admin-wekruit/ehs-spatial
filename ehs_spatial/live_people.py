"""Live people/mover loop over a posed RGB-D stream: timestamped 3D tracks and rule findings.

Consumes the live-core stream contract, one dict per frame: t (seconds), rgb (HxWx3 uint8),
K (3x3), cameraToWorld (4x4, OpenCV camera axes), depth (HxW z-depth in the pose's units,
0 = none). A frame without pose or depth is a coverage gap, never an empty scene. Feet go through
video.lift_foot, rules through video.judge_frame, distances and speeds through video.scale_gated,
so this loop and the offline video tier cannot disagree about a rule.

  python -m ehs_spatial.live_people --self-check
"""

import base64
import io
import json
import time
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
from PIL import Image
from shapely.geometry import Polygon

from .providers.sam3 import SAM3_ENDPOINT, decode_coco_rle, sam_backend_revision, sam_subscribe
from .video import (
    BAND_M, COST_PER_SAM_CALL_USD, MAX_HUMAN_SPEED_MPS, MIN_BOX_HEIGHT_PX, NO_DATA, RULE_NAMES, SAM_SPEND,
    HourlySpendGuard, frame_health, judge_frame, lift_foot, track_speed, zone_in_view,
)

TRACK_KEEPALIVE_S = 5.0  # a track unseen this long is closed; a returning person gets a new id


class Sam3Detector:
    """SAM 3 text-prompt detection through the providers switch (SAM3_BACKEND), under the
    hourly spend guard. Returns None when the detector cannot answer: a coverage gap."""

    def __init__(self, labels=("person",), subscriber: Callable = sam_subscribe, guard: HourlySpendGuard | None = None):
        self.labels, self.subscriber, self.guard = labels, subscriber, guard or SAM_SPEND
        self.backend, self.calls, self.seconds = sam_backend_revision(), 0, []

    def __call__(self, frame: dict) -> list[dict] | None:
        buffer = io.BytesIO()
        Image.fromarray(frame["rgb"]).save(buffer, "JPEG", quality=92)
        uri = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()
        height, width = frame["rgb"].shape[:2]
        found = []
        for label in self.labels:
            self.guard.charge(COST_PER_SAM_CALL_USD)
            started = time.perf_counter()
            try:
                response = self.subscriber(SAM3_ENDPOINT, arguments={
                    "image_url": uri, "prompt": label, "return_multiple_masks": True,
                    "include_scores": True, "include_boxes": True, "max_masks": 32})
            except Exception:
                return None
            self.calls += 1
            self.seconds.append(time.perf_counter() - started)
            if response.get("error"):
                return None
            rles = response.get("rle") or []
            scores = response.get("scores") or []
            for ordinal, rle in enumerate([rles] if isinstance(rles, str) else rles):
                mask = decode_coco_rle(rle, height=height, width=width).astype(bool)
                found.append({"label": label, "mask": mask,
                              "score": float(scores[ordinal]) if ordinal < len(scores) else 1.0})
        return found


class PeopleLoop:
    """One camera's people/mover state. step() takes one stream frame and returns
    (track rows, finding rows); findings are emitted when a rule's verdict changes."""

    def __init__(self, floor_point, floor_up, scale_record: dict, detector: Callable,
                 zone: Polygon | None = None, person_labels=("person",)) -> None:
        self.p0 = np.asarray(floor_point, float)
        self.up = np.asarray(floor_up, float) / np.linalg.norm(floor_up)
        axis = np.eye(3)[np.argmin(np.abs(self.up))]
        self.e1 = axis - (axis @ self.up) * self.up
        self.e1 /= np.linalg.norm(self.e1)
        self.e2 = np.cross(self.up, self.e1)
        self.scale, self.detector, self.zone = scale_record, detector, zone
        self.metres = float(scale_record.get("nativeToMeters") or 1.0)
        self.person_labels = set(person_labels)
        self.tracks: dict[str, dict] = {}  # id -> {kind, last_t, last_xy, samples (accepted, for speed)}
        self.next_id, self.previous_gray, self.last_verdict = 0, None, {}
        self.latencies: list[float] = []

    def floor_xy(self, world: np.ndarray) -> tuple[float, float]:
        offset = np.asarray(world) - self.p0
        return float(offset @ self.e1 * self.metres), float(offset @ self.e2 * self.metres)

    def _associate(self, kind: str, xy: tuple[float, float], t: float, taken: set) -> str:
        """Greedy nearest live track this detection could have walked to since last seen."""
        best, best_distance = None, None
        for track_id, track in self.tracks.items():
            if track["kind"] != kind or track_id in taken or t - track["last_t"] > TRACK_KEEPALIVE_S:
                continue
            distance = np.hypot(xy[0] - track["last_xy"][0], xy[1] - track["last_xy"][1])
            if distance <= 2 * BAND_M + MAX_HUMAN_SPEED_MPS * (t - track["last_t"]) and (
                    best_distance is None or distance < best_distance):
                best, best_distance = track_id, distance
        if best is None:
            best = f"{kind}-{self.next_id}"
            self.next_id += 1
            self.tracks[best] = {"kind": kind, "samples": []}
        self.tracks[best].update(last_t=t, last_xy=xy)
        taken.add(best)
        return best

    def step(self, frame: dict) -> tuple[list[dict], list[dict]]:
        started = time.perf_counter()
        t, index = float(frame["t"]), frame.get("frame")
        c2w, depth = frame.get("cameraToWorld"), frame.get("depth")
        gray = np.asarray(frame["rgb"], np.float32).mean(axis=2)
        gap = "no camera pose or depth" if c2w is None or depth is None else frame_health(gray, self.previous_gray)
        self.previous_gray = gray
        detections = None if gap else self.detector(frame)
        if gap is None and detections is None:
            gap = "person detector offline"
        rows, persons, movers, speeds, zone_seen = [], [], [], [], False
        if gap is None:
            c2w = np.asarray(c2w, float)
            rotation, centre = c2w[:3, :3], c2w[:3, 3]
            K = np.asarray(frame["K"], float)
            height, width = depth.shape
            plane = (rotation.T @ self.up, float(self.up @ (centre - self.p0)))
            exclude = np.any([d["mask"] for d in detections], axis=0) if detections else None
            taken: set = set()
            for detection in detections:
                ys, xs = np.nonzero(detection["mask"])
                if not len(ys) or ys.max() - ys.min() < MIN_BOX_HEIGHT_PX:
                    continue
                bbox = (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))
                person = detection["label"] in self.person_labels
                foot_u = float(np.median(xs[ys >= ys.max() - max(2, 0.03 * (ys.max() - ys.min()))]))  # lowest mask pixels
                foot, implied, review = lift_foot(bbox, (width, height), K, plane, depth, exclude, self.metres, person, foot_u)
                if foot is None:
                    continue
                foot_world = rotation @ foot + centre
                entry = {"xy": self.floor_xy(foot_world), "review": review}
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
                track_id = self._associate(kind, self.floor_xy(centroid) if centroid else entry["xy"], t, taken)
                speed = None
                if person:
                    if review:
                        speeds.append((None, "unaccepted foot"))
                    else:
                        self.tracks[track_id]["samples"].append((t, entry["xy"]))
                        speed, reason = track_speed(self.tracks[track_id]["samples"], t)
                        speeds.append((speed, reason))
                    persons.append(entry)
                else:
                    movers.append(entry)
                rows.append({"t": round(t, 4), "frame": index, "track": track_id, "label": detection["label"],
                             "source": detection.get("source"), "footWorld": [round(v, 5) for v in foot_world],
                             "floorXyM": [round(v, 4) for v in entry["xy"]], "centroidWorld": centroid,
                             "centroidXyM": [round(v, 4) for v in self.floor_xy(centroid)] if centroid else None,
                             "impliedHeightM": round(implied, 3), "accepted": not review, "review": review,
                             "speedMps": None if speed is None else round(speed, 3)})
            if self.zone is not None:
                def project(xy):
                    world = self.p0 + (xy[0] * self.e1 + xy[1] * self.e2) / self.metres
                    camera = rotation.T @ (world - centre)
                    return None if camera[2] <= 1e-6 else tuple((K @ (camera / camera[2]))[:2])
                zone_seen = zone_in_view(self.zone, project, (width, height))
        result = judge_frame(persons, movers, self.zone, gap, zone_seen, speeds, self.scale)
        latency = time.perf_counter() - started
        self.latencies.append(latency)
        findings = []
        for rule in RULE_NAMES:
            outcome = result[rule]
            state = (outcome["verdict"], outcome.get("raw"))  # a pre-gate change is news too
            if self.last_verdict.get(rule) != state:
                self.last_verdict[rule] = state
                findings.append({"t": round(t, 4), "frame": index, "rule": rule, "verdict": outcome["verdict"],
                                 "beforeScaleGate": outcome.get("raw"), "value": outcome.get("value"),
                                 "reason": outcome.get("reason"), "scaleStatus": self.scale["status"],
                                 "tracks": sorted({r["track"] for r in rows}), "latencyS": round(latency, 4)})
        return rows, findings


def run(frames: Iterable[dict], loop: PeopleLoop, output: Path) -> dict:
    """Drive the loop over a stream, appending tracks.jsonl and findings.jsonl as frames arrive."""
    output.mkdir(parents=True, exist_ok=True)
    counts = {"frames": 0, "gaps": 0, "track_rows": 0, "findings": 0}
    with open(output / "tracks.jsonl", "w") as tracks, open(output / "findings.jsonl", "w") as findings:
        for frame in frames:
            rows, found = loop.step(frame)
            counts["frames"] += 1
            counts["gaps"] += frame.get("cameraToWorld") is None or frame.get("depth") is None
            counts["track_rows"] += len(rows)
            counts["findings"] += len(found)
            tracks.writelines(json.dumps(row) + "\n" for row in rows)
            findings.writelines(json.dumps(row) + "\n" for row in found)
    latencies = np.array(loop.latencies or [0.0])
    counts["loop_latency_s"] = {"p50": round(float(np.percentile(latencies, 50)), 4),
                                "p95": round(float(np.percentile(latencies, 95)), 4)}
    return counts


def self_check() -> None:
    """A synthetic walker on a flat floor, seen by a camera 1.6 m up with device-metric metres."""
    from .video import contract_scale

    measured = contract_scale({"scale_status": "device_metric", "metres_per_native_unit": 1.0})
    K = np.array([[300.0, 0, 160], [0, 300.0, 120], [0, 0, 1]])
    c2w = np.eye(4)
    c2w[:3, 3] = [0, -1.6, 0]  # OpenCV world: y down, floor y = 0, camera 1.6 m above it
    vv, uu = np.mgrid[0:240, 0:320]
    down = (vv - 120) / 300.0
    floor = np.where(down > 1e-3, 1.6 / np.maximum(down, 1e-3), 0.0)

    def frame_at(t, x, rng, pose=True):
        """Person at (x, 0, 6): a box from the foot pixel up to 1.75 m."""
        mask = np.zeros((240, 320), bool)
        foot_v, head_v = 120 + 300 * 1.6 / 6, 120 - 300 * 0.15 / 6
        u = 160 + 300 * x / 6
        mask[int(head_v):int(foot_v), int(u - 8):int(u + 8)] = True
        depth = np.where(mask, 6.0, floor)
        rgb = rng.integers(20, 235, (240, 320, 3), dtype=np.uint8)
        return {"t": t, "rgb": rgb, "K": K, "cameraToWorld": c2w if pose else None, "depth": depth,
                "_mask": mask}

    detector = lambda frame: [{"label": "person", "mask": frame["_mask"]}]  # noqa: E731
    zone = Polygon([(-2, -3), (2, -3), (2, -1), (-2, -1)])  # floor metres; never entered
    verdicts = {}
    for rate in (1, 5):
        rng = np.random.default_rng(0)
        loop = PeopleLoop([0, 0, 0], [0, -1, 0], measured, detector, zone)
        last = {}
        for t in np.arange(0, 8 + 1e-9, 1 / rate):
            rows, found = loop.step(frame_at(float(t), -1.5 + 0.4 * t, rng))
            last.update({f["rule"]: f["verdict"] for f in found})
            assert rows and rows[0]["accepted"], rows
            assert abs(rows[0]["impliedHeightM"] - 1.75) < 0.05 and len({r["track"] for r in rows}) == 1
        assert rows[0]["speedMps"] is not None and abs(rows[0]["speedMps"] - 0.4) < 0.05, rows[0]
        verdicts[rate] = last
    assert verdicts[1] == verdicts[5], verdicts  # the sample rate never changes a verdict
    assert verdicts[1]["R3_speed"] == "PASS" and verdicts[1]["R2_min_distance"] == NO_DATA
    # The zone lies behind the camera (floor y < 0 is behind it): never fully seen, so no PASS.
    assert verdicts[1]["R1_zone"] == NO_DATA

    rng = np.random.default_rng(1)
    loop = PeopleLoop([0, 0, 0], [0, -1, 0], measured, lambda frame: [], zone=Polygon([(-1, 5), (1, 5), (1, 7), (-1, 7)]))
    assert loop.step(frame_at(0.0, 0, rng))[1][0]["verdict"] == "PASS"  # zone fully in view, nobody in it
    assert loop.step(frame_at(1.0, 0, rng, pose=False))[1][0]["verdict"] == NO_DATA  # a coverage gap
    offline = PeopleLoop([0, 0, 0], [0, -1, 0], measured, lambda frame: None, zone=loop.zone)
    assert offline.step(frame_at(0.0, 0, rng))[1][0]["reason"] == "person detector offline"
    stated = {"status": "model_estimated", "nativeToMeters": 1.0}
    fast = PeopleLoop([0, 0, 0], [0, -1, 0], stated, detector)
    speed = [f for t in np.arange(0, 2.01, 0.5) for f in fast.step(frame_at(float(t), -2.5 + 2.5 * t, rng))[1]
             if f["rule"] == "R3_speed"]  # 2.5 m/s
    assert speed[-1]["verdict"] == "NEEDS_REVIEW" and speed[-1]["beforeScaleGate"] == "FAIL", speed
    print("live people check passed: 1 Hz and 5 Hz give the same verdicts; a gap or an offline detector is "
          "NO_DATA, a covered empty zone PASS; an assumed scale turns a speed FAIL into NEEDS_REVIEW")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    if parser.parse_args().self_check:
        self_check()
