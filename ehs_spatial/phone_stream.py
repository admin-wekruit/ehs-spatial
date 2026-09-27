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

Only a frame whose trackingState is "normal", with an int worldOriginEpoch and a pose that is a rigid transform, has a
credible pose; every other frame is a coverage gap for every consumer, never an empty scene. Each epoch is its own
world: two epochs' poses share no frame until a Sim3 gate joins them.
"""
import json
import struct

import cv2
import numpy as np

TRACKING_NORMAL = "normal"
MIN_CONFIDENCE = 2  # ARKit high only: medium is mostly depth edges, which smear into a map and into a foot


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


def pose(header: dict) -> np.ndarray | None:
    """cameraToWorld as a 4x4, None when it is missing, not 16 finite numbers, or not a rigid transform: a pose that
    cannot be read is no pose, so the frame is a gap and not an integration at a made-up place."""
    values = np.array(header.get("cameraToWorld") or [np.nan], float)
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
