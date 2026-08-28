"""Semantic orientation recovery — the understanding-layer defense.

EXIF handles the deterministic case, but messengers and screenshots strip
the tag while leaving the pixels rotated. Here the VLM answers the one
question a human answers instantly — "which way is up?" — and the image
is rotated upright in place before any geometry runs. The floor-normal
gravity check in scene build remains the physical tripwire behind both.
Fail-soft: a provider hiccup leaves the image untouched (the tripwire
still guards the output).
"""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class OrientationCall(BaseModel):
    rotate_clockwise_deg: Literal[0, 90, 180, 270] = Field(
        description="clockwise rotation that makes the photo upright "
        "(floor at the bottom, walls/pillars vertical); 0 if already upright"
    )
    reason: str = ""


_PROMPT = (
    "This is an industrial factory photo. Decide which CLOCKWISE rotation "
    "(0, 90, 180 or 270 degrees) makes it upright: the floor at the "
    "bottom, walls, fence posts and pillars vertical, ceiling at the top. "
    "Answer 0 if it is already upright."
)


def ensure_upright(image_paths: list[str], adapter) -> list[int]:
    """Rotate each image file upright in place per the VLM's call.
    Returns the applied clockwise rotations (0 = untouched)."""
    from PIL import Image

    from .providers.gemini import (
        GEMINI_MODEL_ID,
        _image_block,
        _response_format,
        _text_block,
    )

    applied: list[int] = []
    for path in image_paths:
        rotation = 0
        try:
            response = adapter._create(
                "orientation.check",
                model=GEMINI_MODEL_ID,
                input=[_text_block(_PROMPT), _image_block(str(path))],
                response_format=_response_format(OrientationCall),
            )
            call, _ = adapter._parse(
                response, OrientationCall, "orientation.check"
            )
            rotation = call.rotate_clockwise_deg
        except Exception:
            rotation = 0
        if rotation:
            with Image.open(path) as image:
                upright = image.rotate(-rotation, expand=True)
                upright.save(path, quality=95)
        applied.append(rotation)
    return applied


__all__ = ["ensure_upright", "OrientationCall"]
