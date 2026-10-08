"""Canonical RLE codecs retained for capture masks and platform evidence."""
from collections.abc import Mapping
import json
import numpy as np

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
