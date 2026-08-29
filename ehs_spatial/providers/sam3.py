import base64
import time
from collections.abc import Callable, Mapping
import json
from pathlib import Path

import numpy as np
from PIL import Image

from ..contracts import Observation2D
from ..path_safety import validate_safe_path_segment
from .base import ProviderError


SAM3_ENDPOINT = "fal-ai/sam-3-1/image-rle"
PROMPT_VOCABULARY = (
    "factory floor",
    "safety fence",
    "industrial robot arm",
    "material cart",
    "pallet",
    "crate",
    "step ladder",
    "portable work platform",
    "safety sensor",
    "emergency stop button",
    "warning sign",
    "safety light",
)

# Ordered prompt candidates per canonical label, canonical phrase first. SAM 3 is
# trained on short noun phrases, so compound labels need short fallbacks; measured
# on the eval pack: "safety fence" -> 0 detections, "barrier" -> 0.97 recall.
LABEL_PROMPTS: dict[str, tuple[str, ...]] = {
    "factory floor": ("factory floor", "floor", "ground"),
    # "clear panel"/"polycarbonate panel" added 2026-08-26: aluminium-frame
    # clear-panel guarding (owner's real workcells + generated test set)
    # scored 0 on every earlier synonym; A/B on both image families found
    # "clear panel" hits both (8 masks, score 0.82 real / 0.66 generated).
    "safety fence": (
        "safety fence",
        "fence",
        "barrier",
        "guardrail",
        "clear panel",
        "polycarbonate panel",
    ),
    "industrial robot arm": ("industrial robot arm", "robot arm", "robot"),
    "material cart": ("material cart", "cart"),
    "pallet": ("pallet",),
    "crate": ("crate",),
    "step ladder": ("step ladder", "ladder"),
    "portable work platform": ("portable work platform", "work platform"),
    "safety sensor": (
        "safety sensor",
        "light curtain",
        "photoelectric sensor",
        "safety scanner",
    ),
    "emergency stop button": (
        "emergency stop button",
        "e-stop button",
        "red emergency button",
    ),
    "warning sign": ("warning sign", "safety sign", "hazard sign"),
    "safety light": ("safety light", "stack light", "signal tower", "andon light"),
}


def _mask_from_counts(counts: list[int], height: int, width: int) -> np.ndarray:
    total = height * width
    flat = np.zeros(total, dtype=np.uint8)
    offset = 0
    value = 0
    for count in counts:
        if count < 0 or offset + count > total:
            raise ValueError("invalid COCO RLE run lengths")
        if value:
            flat[offset : offset + count] = 1
        offset += count
        value = 1 - value
    if offset != total:
        raise ValueError(f"COCO RLE covers {offset} pixels, expected {total}")
    return flat.reshape((height, width), order="F")


def _decode_compressed_counts(value: str) -> list[int]:
    counts = []
    position = 0
    while position < len(value):
        decoded = 0
        shift = 0
        more = True
        while more:
            if position >= len(value):
                raise ValueError("truncated compressed COCO RLE")
            code = ord(value[position]) - 48
            if code < 0 or code > 63:
                raise ValueError("invalid compressed COCO RLE character")
            decoded |= (code & 0x1F) << shift
            more = bool(code & 0x20)
            position += 1
            if not more and code & 0x10:
                decoded |= -1 << (shift + 5)
            shift += 5
        if len(counts) > 2:
            decoded += counts[-2]
        if decoded < 0:
            raise ValueError("invalid negative compressed COCO RLE count")
        counts.append(decoded)
    return counts


def _mask_from_fal_pairs(value: str, height: int, width: int) -> np.ndarray:
    tokens = value.split()
    if not tokens or len(tokens) % 2:
        raise ValueError("fal RLE requires start/length pairs")
    try:
        pairs = [int(token) for token in tokens]
    except ValueError as exc:
        raise ValueError("fal RLE pairs must be integers") from exc

    total = height * width
    flat = np.zeros(total, dtype=np.uint8)
    previous_end = 0
    for start, length in zip(pairs[::2], pairs[1::2], strict=True):
        offset = start - 1
        end = offset + length
        if start < 1 or length < 1 or offset < previous_end or end > total:
            raise ValueError("invalid fal RLE start/length pairs")
        flat[offset:end] = 1
        previous_end = end
    return flat.reshape((height, width))


def encode_coco_rle(mask: np.ndarray) -> str:
    """Inverse of decode_coco_rle's object form: column-major runs starting
    with the zero run, wrapped as {"size": [H, W], "counts": [...]}."""
    mask = np.asarray(mask).astype(bool)
    flat = mask.flatten(order="F").astype(np.int8)
    boundaries = np.concatenate(
        ([0], np.flatnonzero(np.diff(flat)) + 1, [flat.size])
    )
    counts = np.diff(boundaries).tolist()
    if flat.size and flat[0] == 1:
        counts = [0, *counts]
    return json.dumps(
        {
            "size": [int(mask.shape[0]), int(mask.shape[1])],
            "counts": [int(count) for count in counts],
        }
    )


def decode_coco_rle(
    rle: str,
    *,
    height: int | None = None,
    width: int | None = None,
) -> np.ndarray:
    try:
        payload = json.loads(rle)
    except json.JSONDecodeError:
        payload = None

    if isinstance(payload, dict):
        if not {"size", "counts"} <= payload.keys():
            raise ValueError("COCO RLE object requires size and counts")
        rle_height, rle_width = payload["size"]
        counts = payload["counts"]
    elif height is not None and width is not None:
        rle_height, rle_width = height, width
        counts = payload if isinstance(payload, (list, str)) else rle
    else:
        raise ValueError("counts-only COCO RLE requires height and width")

    if not isinstance(rle_height, int) or not isinstance(rle_width, int):
        raise ValueError("COCO RLE size must contain integer height and width")
    if isinstance(counts, str) and any(character.isspace() for character in counts):
        return _mask_from_fal_pairs(counts, rle_height, rle_width)
    if isinstance(counts, str):
        decoded_counts = _decode_compressed_counts(counts)
    elif isinstance(counts, list) and all(
        isinstance(count, int) and not isinstance(count, bool) for count in counts
    ):
        decoded_counts = counts
    else:
        raise ValueError("unknown COCO RLE counts format")
    return _mask_from_counts(decoded_counts, rle_height, rle_width)


def sam_subscribe(endpoint: str, *, arguments: dict[str, object]) -> object:
    """Backend-routed SAM call, fal request/response schema on every
    backend: fal (vendor API), modal (our warm L4), http (internal GPU
    serving speaking the contract in ehs_spatial.backends)."""
    from ..backends import http_json, service_backend

    backend = service_backend("SAM3_BACKEND", "fal")
    if backend == "fal":
        import fal_client

        return fal_client.subscribe(endpoint, arguments=arguments)
    import base64 as _b64
    import os

    data_url = str(arguments["image_url"])
    image_b64 = data_url.split(",", 1)[1]
    if arguments.get("box_prompts"):
        box_prompt = arguments["box_prompts"][0]
        box = [
            box_prompt["x_min"], box_prompt["y_min"],
            box_prompt["x_max"], box_prompt["y_max"],
        ]
        prompt = {"box": box}
    else:
        prompt = {"text": str(arguments.get("prompt", ""))}
    if backend == "http":
        body = (
            {"box": prompt["box"]}
            if "box" in prompt
            else {"prompt": prompt["text"]}
        )
        return http_json(
            os.environ["SAM3_HTTP_URL"], {"image_b64": image_b64, **body}
        )
    if backend == "modal":
        import modal

        Sam3 = modal.Cls.from_name("sam3-inference", "Sam3")
        return Sam3().segment.remote(_b64.b64decode(image_b64), [prompt])[0]
    raise ValueError(f"unknown SAM3_BACKEND {backend!r}")


def _default_subscriber(endpoint: str, *, arguments: dict[str, object]) -> object:
    return sam_subscribe(endpoint, arguments=arguments)


class SAM3Adapter:
    def __init__(self, subscriber: Callable[..., object] | None = None) -> None:
        self.subscriber = subscriber or _default_subscriber

    def segment(
        self,
        image_path: str | Path,
        *,
        prompt: str,
        frame_id: str,
        output_dir: str | Path,
        label: str | None = None,
    ) -> list[Observation2D]:
        validate_safe_path_segment(frame_id, "frame_id")
        label = prompt if label is None else label
        if label not in PROMPT_VOCABULARY:
            raise ValueError(f"unsupported SAM 3 prompt: {label}")
        if prompt not in LABEL_PROMPTS[label]:
            raise ValueError(
                f"unsupported SAM 3 prompt for label {label!r}: {prompt}"
            )
        source = Path(image_path)
        raw_image = source.read_bytes()
        with Image.open(source) as canonical:
            width, height = canonical.size
            mime_type = Image.MIME.get(canonical.format or "", "image/png")
        request = {
            "image_url": (
                f"data:{mime_type};base64,"
                f"{base64.b64encode(raw_image).decode('ascii')}"
            ),
            "prompt": prompt,
            "return_multiple_masks": True,
            "include_scores": True,
            "include_boxes": True,
        }
        # fal's billing gate flaps under burst usage: a positive-balance
        # account can return "User is locked. Reason: TOP_UP" for seconds at
        # a time (single probes succeed minutes apart). Treat lock/429/5xx
        # as transient with bounded backoff, like video._subscribe_with_backoff.
        # ponytail: fixed 5 tries / linear sleep; make configurable if a
        # provider ever needs a different budget.
        last_exc: Exception | None = None
        for attempt in range(5):
            try:
                response = self.subscriber(SAM3_ENDPOINT, arguments=request)
                break
            except Exception as exc:
                message = str(exc)
                transient = any(
                    marker in message
                    for marker in ("locked", "429", "502", "503", "timeout")
                )
                if not transient or attempt == 4:
                    raise ProviderError("fal", "sam3.subscribe", message) from exc
                last_exc = exc
                time.sleep(15 * (attempt + 1))
        try:
            if not isinstance(response, Mapping):
                raise ValueError("SAM 3 response must be an object")
            rle_value = response.get("rle")
            if isinstance(rle_value, str):
                rles = [rle_value]
            elif isinstance(rle_value, list) and all(
                isinstance(item, str) for item in rle_value
            ):
                rles = rle_value
            else:
                raise ValueError(
                    "SAM 3 response rle must be a string or list of strings"
                )
            metadata = response.get("metadata") or []
            scores = response.get("scores") or []
            boxes = response.get("boxes") or []
            if (
                not isinstance(metadata, list)
                or not isinstance(scores, list)
                or not isinstance(boxes, list)
            ):
                raise ValueError("SAM 3 scores, boxes, and metadata must be lists")
        except Exception as exc:
            raise ProviderError("fal", "sam3.response", str(exc)) from exc

        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        prompt_slug = label.replace(" ", "_")
        observations = []
        seen_instance_ids: set[str] = set()
        for ordinal, serialized in enumerate(rles):
            try:
                item_metadata = metadata[ordinal] if ordinal < len(metadata) else {}
                if not isinstance(item_metadata, Mapping):
                    raise ValueError("SAM 3 mask metadata must be an object")
                instance_index = item_metadata.get("index", ordinal)
                if not isinstance(instance_index, int) or isinstance(
                    instance_index, bool
                ):
                    raise ValueError("SAM 3 mask metadata index must be an integer")
                score = (
                    scores[ordinal]
                    if ordinal < len(scores)
                    else item_metadata.get("score")
                )
                box = (
                    boxes[ordinal]
                    if ordinal < len(boxes)
                    else item_metadata.get("box")
                )
                if score is None or box is None:
                    raise ValueError(
                        "SAM 3 response is missing requested score or box"
                    )
            except Exception as exc:
                raise ProviderError("fal", "sam3.normalize", str(exc)) from exc

            try:
                mask = decode_coco_rle(serialized, height=height, width=width)
            except Exception as exc:
                raise ProviderError("fal", "sam3.decode", str(exc)) from exc
            mask_image = Image.fromarray(mask * 255)
            if mask_image.size != (width, height):
                mask_image = mask_image.resize((width, height), Image.Resampling.NEAREST)
            binary_mask = (np.asarray(mask_image) > 0).astype(np.uint8) * 255
            instance_id = str(instance_index)
            if instance_id in seen_instance_ids:
                raise ProviderError(
                    "fal",
                    "sam3.normalize",
                    f"duplicate SAM 3 mask instance id: {instance_id}",
                )
            seen_instance_ids.add(instance_id)
            mask_path = destination / f"{frame_id}_{prompt_slug}_{instance_id}.png"
            try:
                observation = Observation2D(
                    observation_id=f"{frame_id}:{prompt_slug}:{instance_id}",
                    frame_id=frame_id,
                    label=label,
                    instance_id=instance_id,
                    mask_path=str(mask_path),
                    score=score,
                    bbox=box,
                    source_prompt=prompt,
                )
            except (TypeError, ValueError) as exc:
                raise ProviderError("fal", "sam3.normalize", str(exc)) from exc
            Image.fromarray(binary_mask).save(mask_path)
            observations.append(observation)
        return observations
