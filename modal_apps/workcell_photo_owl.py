"""Reuse the existing OWLv2 coverage detector on four still workcell frames."""

import io
import json
from pathlib import Path
import sys

import modal

if not modal.is_local():
    sys.path.insert(0, "/repo")
from modal_apps.fast_report_app import image, VOLUMES

app = modal.App("workcell-photo-owl")


@app.function(image=image, gpu="A100-80GB:2", cpu=8, memory=64 * 1024,
              volumes=VOLUMES, timeout=1200, retries=0)
def detect(images: list[bytes]):
    import numpy as np
    from PIL import Image
    import torch
    from fast_report import coverage

    frames = [np.asarray(Image.open(io.BytesIO(b)).convert("RGB"))[..., ::-1].copy() for b in images]
    worker = coverage.Detector("owlv2", torch.device("cuda:0"))
    rows = worker.detect(frames, coverage.SCORE["owlv2"])
    return [[{"box": box.tolist(), "score": round(float(score), 4), "word": word}
             for box, score, word in zip(boxes, scores, words)] for boxes, scores, words in rows]


@app.local_entrypoint()
def main(images: str, out: str):
    import time

    paths = [Path(x) for x in images.split(",")]
    started = time.monotonic()
    rows = detect.remote([p.read_bytes() for p in paths])
    payload = {"images": [str(p) for p in paths], "results": rows,
               "wallSecondsIncludingColdStart": time.monotonic() - started}
    Path(out).write_text(json.dumps(payload, indent=2) + "\n")
    print(f"{len(rows)} frames; boxes {[len(x) for x in rows]}; {payload['wallSecondsIncludingColdStart']:.1f} s")
