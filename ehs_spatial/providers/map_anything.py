import base64
from collections.abc import Callable, Mapping
from contextlib import ExitStack
import json
import math
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

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
    import replicate

    return replicate.run(model_identifier, input=input)


def _json_safe(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _read_provider_bytes(location: object) -> bytes:
    if not isinstance(location, (str, Path)) and hasattr(location, "read"):
        content = location.read()
        return content if isinstance(content, bytes) else bytes(content)
    value = str(getattr(location, "url", location))
    local_path = Path(value)
    if local_path.is_file():
        return local_path.read_bytes()
    if urlparse(value).scheme in {"http", "https", "file"}:
        with urlopen(value, timeout=60) as response:
            return response.read()
    raise ValueError(f"unsupported provider file location: {value}")


def _download_provider_bytes(location: object) -> bytes:
    try:
        return _read_provider_bytes(location)
    except Exception as exc:
        raise ProviderError("replicate", "map_anything.download", str(exc)) from exc


class MapAnythingAdapter:
    def __init__(self, runner: Callable[..., object] | None = None) -> None:
        self.runner = runner or _default_runner

    def run(
        self,
        image_paths: list[str],
        geometry_dir: str | Path,
    ) -> tuple[list[GeometryFrame], Path]:
        if len(image_paths) != 4:
            raise ValueError("MapAnything requires exactly four image paths")
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
            with ExitStack() as stack:
                inputs = [stack.enter_context(source.open("rb")) for source in sources]
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
            if not isinstance(data, (list, tuple)) or len(data) != 4:
                raise ValueError("MapAnything response must contain four data files")
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
            raw_json_path.write_bytes(_download_provider_bytes(location))
            raw_json_paths.append(raw_json_path)
        point_cloud_path.write_bytes(_download_provider_bytes(point_cloud_location))

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
