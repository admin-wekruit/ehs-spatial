"""One phone stream, two live consumers: what scripts/live_map.py's producer sends, the map worker integrates and the
people loop covers. A consumer reading other header keys sees every frame as a gap, which is what this catches."""

import sys
import zipfile
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import Polygon

from ehs_spatial import live_people, phone_stream
from ehs_spatial.video import contract_scale

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import live_map  # noqa: E402


class OneAtATime:
    """serve() gets one message per poll, so latest-wins drops none."""

    def __init__(self, messages):
        self.messages = deque(messages)

    def poll(self, timeout):
        return timeout > 0 and bool(self.messages)

    def recv_bytes(self):
        return self.messages.popleft()


def raw_capture(root: Path, n: int = 3) -> Path:
    """A three-ARFrame ARKitScenes raw capture: camera 1.6 m over a flat floor, looking along the floor (the same
    synthetic frame the people self-check uses), identity world-to-camera trajectory."""
    rng, stamps = np.random.default_rng(0), [f"{100 + 0.1 * i:.3f}" for i in range(n)]
    (root / "lowres_wide.traj").write_text("99.9 0 0 0 0 0 0\n100.5 0 0 0 0 0 0\n")
    with zipfile.ZipFile(root / "wide.zip", "w") as wide, zipfile.ZipFile(root / "lowres_depth.zip", "w") as depth, \
            zipfile.ZipFile(root / "confidence.zip", "w") as confidence, zipfile.ZipFile(root / "wide_intrinsics.zip", "w") as pins:
        for i, stamp in enumerate(stamps):
            _, rgb, png, levels = phone_stream.unpack(live_people._message(0.1 * i, rng)[0])
            wide.writestr(f"wide/47333932_{stamp}.png", cv2.imencode(".png", cv2.imdecode(np.frombuffer(rgb, np.uint8), cv2.IMREAD_COLOR))[1].tobytes())
            depth.writestr(f"lowres_depth/47333932_{stamp}.png", png)
            confidence.writestr(f"confidence/47333932_{stamp}.png", levels)
            pins.writestr(f"wide_intrinsics/47333932_{stamp}.pincam", "1920 1440 1800 1800 960 720")
    return root


def test_one_codec_for_every_producer_and_consumer():
    assert live_map.pack is phone_stream.pack and live_map.unpack is phone_stream.unpack
    assert live_people.pack is phone_stream.pack and live_people.unpack is phone_stream.unpack
    assert not hasattr(live_people, "struct")  # no second copy of the byte layout


def test_the_map_producer_is_covered_by_both_consumers(tmp_path):
    sent = []
    live_map.produce(raw_capture(tmp_path), sent.append, rate_hz=0)
    assert sent[-1] == b"" and len(sent) == 4
    head = phone_stream.unpack(sent[0])[0]
    assert (head["trackingState"], head["trackingStateReason"], head["worldOriginEpoch"]) == ("normal", None, 0)

    stats = live_map.serve(OneAtATime(sent), tmp_path / "map", 5000)
    assert stats["frames_integrated"] == 3 and stats["coverage_gap_frames"] == {}, stats

    measured = contract_scale({"scale_status": "device_metric", "metres_per_native_unit": 1.0})
    zone = Polygon([(-1, 5), (1, 5), (1, 7), (-1, 7)])  # floor metres (x, z) in front of the camera
    loop = live_people.PeopleLoop([0, 1.6, 0], [0, -1, 0], measured, lambda frame: [], zone)  # identity pose: floor at y = 1.6
    for message in sent[:-1]:
        loop.step(live_people.decode_frame(message))
    assert not loop.gaps and loop.last_verdict["R1_zone"][0] == "PASS", (dict(loop.gaps), loop.last_verdict)


def test_a_bad_pose_or_confidence_plane_is_no_data():
    good = phone_stream.header(0, 0.0, 0.0, [600, 600, 320, 240], np.eye(4))
    assert phone_stream.pose(good) is not None
    for c2w in (None, np.full((4, 4), np.nan), np.diag([2.0, 1, 1, 1]), np.diag([1.0, 1, -1, 1])):  # null, NaN, scaled, mirrored
        assert phone_stream.pose(phone_stream.header(0, 0.0, 0.0, [600, 600, 320, 240], c2w)) is None, c2w
    depth = cv2.imencode(".png", np.full((4, 4), 1500, np.uint16))[1].tobytes()
    levels = np.array([[2, 1, 0, 2]] * 4, np.uint8)
    got = phone_stream.depth_mm(depth, cv2.imencode(".png", levels)[1].tobytes())
    assert (got == np.where(levels == 2, 1500, 0)).all()  # only high confidence stays
    assert phone_stream.depth_mm(depth, cv2.imencode(".png", np.full((2, 2), 2, np.uint8))[1].tobytes()) is None
    assert phone_stream.depth_mm(cv2.imencode(".png", np.full((4, 4), 9, np.uint8))[1].tobytes()) is None  # not millimetres


def test_both_consumers_call_the_same_frames_gaps(tmp_path):
    """Each consumer coded the credibility rule itself and they disagreed: epoch 0.0 PASSed the people loop while the map
    recorded a gap, epoch False was integrated into 'submap-False', and neither checked that K and rgb belong to the
    640x480 raster (ARKit's own K is for 1920x1440: rays 3x off). Both now ask phone_stream.credible: each of these is a
    gap in both, under the same reason, and the well-formed frame is covered by both."""
    rng = np.random.default_rng(0)
    head, rgb, depth, levels = phone_stream.unpack(live_people._message(0.0, rng)[0])
    full = cv2.imencode(".jpg", cv2.resize(cv2.imdecode(np.frombuffer(rgb, np.uint8), cv2.IMREAD_COLOR), (1920, 1440)))[1].tobytes()
    cases = [({}, rgb), ({"worldOriginEpoch": 0.0}, rgb), ({"worldOriginEpoch": False}, rgb), ({"worldOriginEpoch": "0"}, rgb),
             ({"worldOriginEpoch": None}, rgb), ({"trackingState": "limited"}, rgb), ({"cameraToWorld": [float("nan")] * 16}, rgb),
             ({"K": [1800.0, 1800.0, 960.0, 720.0]}, rgb), ({"K": [300.0, 300.0, 160.0, 120.0]}, rgb), ({}, full), ({}, b"not a jpeg")]
    measured = contract_scale({"scale_status": "device_metric", "metres_per_native_unit": 1.0})
    for n, (change, image) in enumerate(cases):
        message = phone_stream.pack({**head, **change}, image, depth, levels)
        mapped = live_map.serve(OneAtATime([message, b""]), tmp_path / str(n), 5000)
        people = live_people.PeopleLoop([0, 0, 0], [0, -1, 0], measured, lambda frame: [], live_people._SEEN_ZONE)
        people.step(live_people.decode_frame(message))
        if n == 0:
            assert mapped["frames_integrated"] == 1 and not people.gaps and people.last_verdict["R1_zone"][0] == "PASS"
            continue
        assert mapped["frames_integrated"] == 0 and not list((tmp_path / str(n)).glob("submap-*")), (change, mapped)
        assert people.last_verdict["R1_zone"][0] == "NO_DATA", (change, people.last_verdict)
        assert list(people.gaps) == list(mapped["coverage_gap_frames"]), (change, dict(people.gaps), mapped["coverage_gap_frames"])


def test_credible_refuses_malformed_numbers_with_a_fixed_reason_and_never_raises():
    base = phone_stream.header(0, 0., 0., [500., 500., 320., 240.], np.eye(4))
    assert phone_stream.credible(base)[0] is None
    for K in ([np.nan, np.nan, 320., 240.], [-600., -600., 320., 240.], "abc", [[1, 2], [3]], {"fx": 1}, None):
        why, c2w, _ = phone_stream.credible({**base, "K": K})
        assert why == "K not of the 640x480 raster" and c2w is None, K
    for pose in ("abc", [[1, 0], [0]], {"a": 1}):
        assert phone_stream.credible({**base, "cameraToWorld": pose})[0] == "no cameraToWorld", pose
    reasons = {phone_stream.credible({**base, "worldOriginEpoch": e})[0] for e in (.5, 1.5, "0", False)}
    assert reasons == {"world origin epoch not an int"}  # counter keys stay a fixed set, not one per bad value
    assert phone_stream.credible({**base, "trackingState": "limited", "trackingStateReason": "x" * 50})[0] == "ARKit tracking limited (other reason)"
    assert phone_stream.credible({**base, "trackingState": "limited", "trackingStateReason": "excessiveMotion"})[0] == "ARKit tracking limited (excessiveMotion)"
    assert phone_stream.credible({**base, "trackingState": "weird"})[0] == "ARKit tracking state unknown"


def test_the_map_reads_the_rgb_size_from_the_jpeg_header_without_decoding():
    base = phone_stream.header(0, 0., 0., [500., 500., 320., 240.], np.eye(4))
    small = cv2.imencode(".jpg", np.zeros((480, 640, 3), np.uint8))[1].tobytes()
    big = cv2.imencode(".jpg", np.zeros((1440, 1920, 3), np.uint8))[1].tobytes()
    assert phone_stream.jpeg_size(small) == (640, 480) and phone_stream.jpeg_size(big) == (1920, 1440) and phone_stream.jpeg_size(b"not a jpeg") is None
    assert phone_stream.credible(base, small, decode=False) [0] is None and phone_stream.credible(base, small, decode=False)[2] is None
    for decode in (False, True):
        assert phone_stream.credible(base, big, decode=decode)[0] == "rgb not a 640x480 JPEG"
    assert phone_stream.credible(base, b"not a jpeg", decode=False)[0] == "rgb not a 640x480 JPEG"
