"""SAM 3 on-prem service — implements SAM3_HTTP_URL (docs/BACKENDS.md).

    POST /sam3  {"image_b64": ..., "prompt": "safety fence"}
    POST /sam3  {"image_b64": ..., "box": [x1, y1, x2, y2]}
    ->          {"rle": [<COCO object RLE json str>, ...], "scores": [...]}

Run:  uvicorn sam3_service:app --host 0.0.0.0 --port 8801
Note: facebook/sam3 is a GATED repo — the HF account behind HF_TOKEN must
have accepted the license at https://huggingface.co/facebook/sam3.
"""

import base64
import io
import json
import threading

import numpy as np
import torch
from fastapi import FastAPI
from PIL import Image
from pydantic import BaseModel
from transformers import Sam3Model, Sam3Processor

MODEL_ID = "facebook/sam3"

app = FastAPI(title="sam3-service")
_lock = threading.Lock()  # one GPU, serialized inference
processor = Sam3Processor.from_pretrained(MODEL_ID)
model = Sam3Model.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16).to("cuda")
model.eval()


class Sam3Request(BaseModel):
    image_b64: str
    prompt: str | None = None
    box: list[float] | None = None


def _encode_coco_rle(mask: np.ndarray) -> str:
    mask = np.asarray(mask).astype(bool)
    flat = mask.flatten(order="F").astype(np.int8)
    boundaries = np.concatenate(([0], np.flatnonzero(np.diff(flat)) + 1, [flat.size]))
    counts = np.diff(boundaries).tolist()
    if flat.size and flat[0] == 1:
        counts = [0, *counts]
    return json.dumps(
        {"size": [int(mask.shape[0]), int(mask.shape[1])],
         "counts": [int(c) for c in counts]}
    )


@app.post("/sam3")
def segment(request: Sam3Request) -> dict:
    image = Image.open(io.BytesIO(base64.b64decode(request.image_b64))).convert("RGB")
    kwargs = {"images": image, "return_tensors": "pt"}
    if request.box is not None:
        kwargs["input_boxes"] = [[list(request.box)]]
    else:
        kwargs["text"] = request.prompt or ""
    with _lock, torch.inference_mode():
        inputs = processor(**kwargs).to("cuda")
        outputs = model(**inputs)
        results = processor.post_process_instance_segmentation(
            outputs, threshold=0.4, mask_threshold=0.5,
            target_sizes=[(image.height, image.width)],
        )[0]
    masks, scores = results.get("masks"), results.get("scores")
    if masks is None or len(masks) == 0:
        return {"rle": [], "scores": []}
    masks = masks.cpu().numpy()
    scores = [float(s) for s in scores.cpu().numpy()]
    order = sorted(range(len(scores)), key=lambda i: -scores[i])[:12]
    return {
        "rle": [_encode_coco_rle(masks[i] > 0.5) for i in order],
        "scores": [round(scores[i], 4) for i in order],
    }


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True, "model": MODEL_ID}
