"""Fetch MoGe-2 (Replicate) outputs for the MoGe-3 A/B. One-off eval helper.

Usage: uv run --env-file .env python scripts/moge2_eval_fetch.py incoming/gen-01.png
Saves every output the model returns under outputs/moge3_eval/moge2/<stem>/.
"""

import base64
import json
import sys
from pathlib import Path

import replicate

from ehs_spatial.providers.moge import MOGE_VERSION


def main(image_path: str) -> None:
    path = Path(image_path)
    out_dir = Path("outputs/moge3_eval/moge2") / path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix = "png" if path.suffix.lower() == ".png" else "jpeg"
    payload = f"data:image/{suffix};base64," + base64.b64encode(
        path.read_bytes()
    ).decode("ascii")
    # wait=False -> poll instead of holding one long HTTP read (avoids ReadTimeout)
    output = replicate.run(MOGE_VERSION, input={"image": payload, "fp16": True}, wait=False)

    meta = {}
    items = output.items() if isinstance(output, dict) else enumerate(output)
    for key, value in items:
        if hasattr(value, "read"):
            name = Path(getattr(value, "url", str(key))).name or str(key)
            target = out_dir / f"{key}__{name}"
            target.write_bytes(value.read())
            meta[str(key)] = target.name
        else:
            meta[str(key)] = value
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    print(path.stem, "->", json.dumps(meta, default=str)[:500])


if __name__ == "__main__":
    main(sys.argv[1])
