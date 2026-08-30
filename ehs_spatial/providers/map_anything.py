import base64
from collections.abc import Callable, Mapping
import json
import math
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, build_opener

import numpy as np
from PIL import Image

from ..contracts import GeometryFrame
from .base import ProviderError


MAP_ANYTHING_MODEL_ID = (
    "vufinder/map-anything:"
    "bb68c254a65d3ce6b173909181d2dfbd044300b3ebca25ab63f07aa7eb1eebff"
)
# This pinned Replicate wrapper uses the map-anything-apache checkpoint.


def decode_encoded_array(payload: dict[str, object]) -> np.ndarray:
    dtype = np.dtype(payload["dtype"])
    shape = tuple(payload["shape"])
    raw = base64.b64decode(payload["data"], validate=True)
    expected_bytes = math.prod(shape) * dtype.itemsize
    if len(raw) != expected_bytes:
        raise ValueError(
            f"encoded array byte count mismatch: expected {expected_bytes}, got {len(raw)}"
        )
    return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()


def parse_frame_json(
    json_path: str | Path,
    output_dir: str | Path,
    frame_id: str,
) -> GeometryFrame:
    payload = json.loads(Path(json_path).read_text(encoding="utf-8"))
    image = decode_encoded_array(payload["image"])
    pts3d = decode_encoded_array(payload["pts3d"])
    conf = decode_encoded_array(payload["conf"])
    valid_mask = decode_encoded_array(payload["non_ambiguous_mask"])
    camera_to_world = decode_encoded_array(payload["camera_poses"])
    intrinsics = decode_encoded_array(payload["intrinsics"])

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"provider image must be HxWx3, got {image.shape}")
    height, width = image.shape[:2]
    if (
        pts3d.shape != (height, width, 3)
        or conf.shape != (height, width)
        or valid_mask.shape != (height, width)
    ):
        raise ValueError("image, pts3d, conf, and non_ambiguous_mask must be pixel-aligned")
    if camera_to_world.shape != (4, 4) or intrinsics.shape != (3, 3):
        raise ValueError("camera_poses must be 4x4 and intrinsics must be 3x3")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    image_path = destination / "canonical.png"
    pts3d_path = destination / "pts3d.npy"
    conf_path = destination / "conf.npy"
    valid_mask_path = destination / "valid_mask.npy"
    Image.fromarray(image).save(image_path)
    np.save(pts3d_path, pts3d)
    np.save(conf_path, conf)
    np.save(valid_mask_path, valid_mask)
    np.save(destination / "camera_to_world.npy", camera_to_world)
    np.save(destination / "intrinsics.npy", intrinsics)

    return GeometryFrame(
        frame_id=frame_id,
        canonical_image_path=str(image_path),
        pts3d_path=str(pts3d_path),
        conf_path=str(conf_path),
        valid_mask_path=str(valid_mask_path),
        camera_to_world=camera_to_world.tolist(),
        intrinsics=intrinsics.tolist(),
    )


def _default_runner(model_identifier: str, *, input: dict[str, object]) -> object:
    from ..backends import http_json, service_backend

    backend = service_backend("GEOMETRY_BACKEND", "replicate")
    if backend == "http":
        import os

        # internal GPU serving: same payload, same response schema —
        # see ehs_spatial.backends for the contract
        return http_json(os.environ["GEOMETRY_HTTP_URL"], dict(input))
    if backend == "modal":
        import modal

        MapAnything = modal.Cls.from_name(
            "mapanything-inference", "MapAnything"
        )
        return MapAnything().run.remote(dict(input))
    import replicate

    # wait=False polls with short requests instead of holding one blocking read;
    # cold-boots longer than the HTTP read timeout otherwise kill the call.
    return replicate.run(model_identifier, input=input, wait=False)


def _json_safe(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (bytes, bytearray)):
        return f"<{len(value)} bytes>"
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


# Hosts a replicate prediction may legitimately point us at. Anything else
# in a provider response is treated as hostile (review finding O3): a
# malicious response must not be able to direct reads of local files or
# arbitrary origins into run artifacts.
_ALLOWED_URL_SUFFIXES = (".replicate.delivery", ".replicate.com")


class _NoRedirectHandler(HTTPRedirectHandler):
    """The suffix check below runs on the initial URL only; a followed
    redirect could land anywhere (metadata IPs, internal services), so any
    30x from an allowed host is treated as hostile and refused outright."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError(f"provider URL redirect refused: {newurl}")


def _redirect_refusing_opener(*extra_handlers):
    return build_opener(_NoRedirectHandler(), *extra_handlers)


_OPENER = _redirect_refusing_opener()


def _read_provider_bytes(location: object, *, allow_local: bool = False) -> bytes:
    # self-hosted backends (Modal, internal http) hand bytes straight back
    if isinstance(location, (bytes, bytearray)):
        return bytes(location)
    if isinstance(location, str) and location.startswith("data:"):
        return base64.b64decode(location.split(",", 1)[1])
    if not isinstance(location, (str, Path)) and hasattr(location, "read"):
        content = location.read()
        return content if isinstance(content, bytes) else bytes(content)
    value = str(getattr(location, "url", location))
    parsed = urlparse(value)
    if parsed.scheme in {"http", "https"}:
        hostname = parsed.hostname or ""
        if not hostname.endswith(_ALLOWED_URL_SUFFIXES):
            raise ValueError(f"provider URL host not allowed: {hostname}")
        with _OPENER.open(value, timeout=60) as response:
            return response.read()
    if allow_local:
        # Test/replay seam only: replaying a saved response with local paths.
        local_path = Path(parsed.path if parsed.scheme == "file" else value)
        if local_path.is_file():
            return local_path.read_bytes()
    raise ValueError(f"unsupported provider file location: {value}")


def _download_provider_bytes(location: object, *, allow_local: bool = False) -> bytes:
    try:
        return _read_provider_bytes(location, allow_local=allow_local)
    except Exception as exc:
        raise ProviderError("replicate", "map_anything.download", str(exc)) from exc


class MapAnythingAdapter:
    def __init__(
        self,
        runner: Callable[..., object] | None = None,
        *,
        allow_local: bool = False,
    ) -> None:
        self.runner = runner or _default_runner
        self.allow_local = allow_local

    def run(
        self,
        image_paths: list[str],
        geometry_dir: str | Path,
    ) -> tuple[list[GeometryFrame], Path]:
        if not 1 <= len(image_paths) <= 4:
            raise ValueError("MapAnything requires one to four image paths")
        sources = [Path(value) for value in image_paths]
        for source in sources:
            if not source.is_file():
                raise FileNotFoundError(source)

        output_dir = Path(geometry_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        flags: dict[str, object] = {
            "normals": False,
            "to_base64": True,
            "return_pcd": True,
            "return_mesh": False,
            "point_scales": False,
            "keys_to_exclude": "",
            "alpha_blend_onto": "white",
        }
        request_metadata = {
            "model_identifier": MAP_ANYTHING_MODEL_ID,
            "input": {**flags, "inputs": image_paths},
        }
        (output_dir / "map_anything_request.json").write_text(
            json.dumps(request_metadata, indent=2) + "\n",
            encoding="utf-8",
        )
        try:
            # Data URIs instead of file handles: the replicate client uploads
            # handles to its authenticated Files API, whose URLs the model's
            # Cog wrapper cannot fetch ("No valid data, image, or video files
            # found in the input!"). Base64 payloads are self-contained.
            inputs = [
                "data:image/png;base64,"
                + base64.b64encode(source.read_bytes()).decode("ascii")
                for source in sources
            ]
            response = self.runner(
                MAP_ANYTHING_MODEL_ID,
                input={"inputs": inputs, **flags},
            )
        except Exception as exc:
            raise ProviderError("replicate", "map_anything.run", str(exc)) from exc

        try:
            response_metadata = _json_safe(response)
        except Exception as exc:
            raise ProviderError("replicate", "map_anything.response", str(exc)) from exc
        (output_dir / "map_anything_response.json").write_text(
            json.dumps(response_metadata, indent=2) + "\n",
            encoding="utf-8",
        )
        try:
            if not isinstance(response, Mapping):
                raise ValueError("MapAnything response must be an object")
            data = response.get("data")
            point_cloud_location = response.get("point_cloud")
            if not isinstance(data, (list, tuple)) or len(data) != len(image_paths):
                raise ValueError(
                    "MapAnything response must contain one data file per "
                    f"input image ({len(image_paths)})"
                )
            if point_cloud_location is None:
                raise ValueError("MapAnything response is missing point_cloud")
        except Exception as exc:
            raise ProviderError("replicate", "map_anything.response", str(exc)) from exc

        provider_dir = output_dir / "provider"
        provider_dir.mkdir(exist_ok=True)
        raw_json_paths = []
        point_cloud_path = output_dir / "point_cloud.glb"
        for index, location in enumerate(data, start=1):
            raw_json_path = provider_dir / f"frame_{index:04d}.json"
            raw_json_path.write_bytes(
                _download_provider_bytes(location, allow_local=self.allow_local)
            )
            raw_json_paths.append(raw_json_path)
        point_cloud_path.write_bytes(
            _download_provider_bytes(point_cloud_location, allow_local=self.allow_local)
        )

        frames = []
        for index, raw_json_path in enumerate(raw_json_paths, start=1):
            frame_id = f"frame_{index:04d}"
            try:
                frames.append(
                    parse_frame_json(
                        raw_json_path,
                        output_dir / "frames" / frame_id,
                        frame_id,
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ProviderError(
                    "replicate", "map_anything.decode", str(exc)
                ) from exc
        return frames, point_cloud_path
