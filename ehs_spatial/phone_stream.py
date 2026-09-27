"""The phone stream contract: one message per ARFrame the phone sends (the ARKit app, or a replay standing in for it), read
by every live consumer (scripts/live_map.py's map worker, ehs_spatial.live_people's people loop). This module is the
only codec and the only place the header's key names are spelled.

    u32 little-endian header length | header JSON | rgb | depth | confidence      (blob sizes are header["sizes"])
    header  seq; t_capture: wall-clock epoch seconds at capture (phone NTP-synced); t_device: ARKit timestamp;
            K: [fx, fy, cx, cy] of the 640x480 rgb raster; cameraToWorld: 16 floats row-major, metres, OpenCV axes
            (x right, y down, z forward: ARKit's camera.transform @ diag(1, -1, -1, 1)), null without a pose;
            trackingState: "normal" | "limited" | "notAvailable" (ARCamera.trackingState); trackingStateReason: null or
            ARKit's reason ("initializing", "excessiveMotion", "insufficientFeatures", "relocalizing");
            worldOriginEpoch: an int the app increments whenever the world origin moves (relocalisation, a loaded
            ARWorldMap, a tracking reset, setWorldOrigin).
    rgb         640x480 JPEG; may be empty (a public capture kept few wide images; the app always sends it)
    depth       optional PNG uint16 millimetres, same field of view as rgb at a lower resolution (K scaled by width)
    confidence  optional PNG uint8 0/1/2 (ARConfidenceLevel), same size as depth

Only a frame whose trackingState is "normal", with an int worldOriginEpoch, a pose that is a rigid transform, a K of the
640x480 raster and an rgb (when sent) of 640x480 is credible (credible(), which every consumer calls); every other frame
is a coverage gap for every consumer, never an empty scene. Each epoch is its own world: two epochs' poses share no
frame until a Sim3 gate joins them.
"""
import json
import struct

import cv2
import numpy as np

TRACKING_NORMAL = "normal"
TRACKING_STATES = ("normal", "limited", "notAvailable")  # ARCamera.TrackingState
TRACKING_REASONS = ("initializing", "relocalizing", "excessiveMotion", "insufficientFeatures")  # ARCamera.TrackingState.Reason
MIN_CONFIDENCE = 2  # ARKit high only: medium is mostly depth edges, which smear into a map and into a foot
RASTER_WH = (640, 480)
# K's principal point lies within this share of the raster of its centre (ARKit's is a few px off). K left at the
# camera's full 1920x1440 puts cx near 960, one scaled twice near 160: rays 3x off, so the frame is a gap, not a guess.
PRINCIPAL_POINT_SHARE = .1


def header(seq, t_capture, t_device, K, camera_to_world, tracking_state=TRACKING_NORMAL, reason=None, epoch=0) -> dict:
    """The header every producer sends."""
    return {"seq": seq, "t_capture": t_capture, "t_device": t_device, "K": [float(v) for v in K],
            "cameraToWorld": None if camera_to_world is None else np.asarray(camera_to_world, float).ravel().tolist(),
            "trackingState": tracking_state, "trackingStateReason": reason, "worldOriginEpoch": epoch}


def pack(header: dict, rgb: bytes = b"", depth: bytes = b"", confidence: bytes = b"") -> bytes:
    text = json.dumps(dict(header, sizes=[len(rgb), len(depth), len(confidence)])).encode()
    return struct.pack("<I", len(text)) + text + rgb + depth + confidence


def unpack(message: bytes) -> tuple[dict, bytes, bytes, bytes]:
    """(header, rgb, depth, confidence) of one message."""
    n = struct.unpack_from("<I", message)[0]
    header, at, blobs = json.loads(message[4:4 + n]), 4 + n, []
    for size in header["sizes"]:
        blobs.append(message[at:at + size])
        at += size
    return header, *blobs


def numbers(value) -> np.ndarray:
    """A header field as a flat float array; anything that is not a list of numbers (a string, a ragged list, a dict)
    is one NaN, so a malformed header is a gap and never an exception that stops the worker."""
    try:
        return np.array(value, float).ravel() if value is not None else np.array([np.nan])
    except (TypeError, ValueError):
        return np.array([np.nan])


def jpeg_size(data: bytes) -> tuple[int, int] | None:
    """(width, height) from a JPEG's start-of-frame marker, without decoding the pixels; None when there is none."""
    i = 2 if data[:2] == b"\xff\xd8" else len(data)
    while i + 9 <= len(data):
        if data[i] != 0xFF:
            return None
        marker, length = data[i + 1], struct.unpack(">H", data[i + 2:i + 4])[0]
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):  # SOFn, not DHT/JPG/DAC
            height, width = struct.unpack(">HH", data[i + 5:i + 9])
            return width, height
        i += 2 + length
    return None


def pose(header: dict) -> np.ndarray | None:
    """cameraToWorld as a 4x4, None when it is missing, not 16 finite numbers, or not a rigid transform: a pose that
    cannot be read is no pose, so the frame is a gap and not an integration at a made-up place."""
    values = numbers(header.get("cameraToWorld"))
    if values.size != 16 or not np.isfinite(values).all():
        return None
    c2w = values.reshape(4, 4)
    rotation = c2w[:3, :3]
    if not (np.allclose(c2w[3], [0, 0, 0, 1]) and np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3)
            and np.linalg.det(rotation) > 0):
        return None
    return c2w


def depth_mm(depth: bytes, confidence: bytes = b"") -> np.ndarray | None:
    """The depth blob as uint16 millimetres with every pixel below MIN_CONFIDENCE zeroed; None when there is no depth,
    it does not decode to uint16, or a confidence plane was sent that does not match it (unknown confidence is not high)."""
    millimetres = cv2.imdecode(np.frombuffer(depth, np.uint8), cv2.IMREAD_UNCHANGED) if depth else None
    if millimetres is None or millimetres.dtype != np.uint16 or millimetres.ndim != 2:
        return None
    if confidence:
        levels = cv2.imdecode(np.frombuffer(confidence, np.uint8), cv2.IMREAD_UNCHANGED)
        if levels is None or levels.shape != millimetres.shape:
            return None
        millimetres[levels < MIN_CONFIDENCE] = 0
    return millimetres


def credible(header: dict, rgb: bytes = b"", decode: bool = True) -> tuple[str | None, np.ndarray | None, np.ndarray | None]:
    """(why this frame is a coverage gap for every consumer or None, cameraToWorld 4x4, rgb as HxWx3 RGB or None when none
    was sent or decode is False). Credible: trackingState "normal", an int worldOriginEpoch (not a bool, a float or a
    string), a rigid finite pose, K of the 640x480 raster, and an rgb, when sent, of 640x480. The one rule both consumers
    apply. Reasons are a fixed set of strings (they are counter keys), never the frame's own values. decode=False (the
    map, which never uses the pixels) reads the rgb size from its JPEG header instead of decoding it."""
    state, why = header.get("trackingState"), header.get("trackingStateReason")
    if state != TRACKING_NORMAL:
        state = state if state in TRACKING_STATES else "state not sent" if state is None else "state unknown"
        return f"ARKit tracking {state}" + (f" ({why})" if why in TRACKING_REASONS else " (other reason)" if why else ""), None, None
    epoch = header.get("worldOriginEpoch")
    if type(epoch) is not int:
        return "world origin epoch " + ("not sent" if epoch is None else "not an int"), None, None
    c2w = pose(header)
    if c2w is None:
        return "no cameraToWorld", None, None
    K = numbers(header.get("K"))
    centre = np.array(RASTER_WH) / 2
    if K.size != 4 or not np.isfinite(K).all() or (K[:2] <= 0).any() or (np.abs(K[2:] - centre) > PRINCIPAL_POINT_SHARE * 2 * centre).any():
        return f"K not of the {RASTER_WH[0]}x{RASTER_WH[1]} raster", None, None
    if not rgb:
        return None, c2w, None
    if jpeg_size(rgb) != RASTER_WH:  # both consumers judge the size from the header, so they give the same reason
        return f"rgb not a {RASTER_WH[0]}x{RASTER_WH[1]} JPEG", None, None
    if not decode:
        return None, c2w, None
    image = cv2.imdecode(np.frombuffer(rgb, np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.shape[1::-1] != RASTER_WH:  # a body that does not decode: only a consumer of the pixels can tell
        return "rgb does not decode", None, None
    return None, c2w, cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
