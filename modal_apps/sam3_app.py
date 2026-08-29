"""SAM 3 inference on Modal — self-hosted replacement for the fal endpoint.

Deploy:  uv run --with modal modal deploy modal_apps/sam3_app.py
Smoke:   uv run --with modal modal run modal_apps/sam3_app.py --image-path <jpg> --prompt fence

Why: per-call API pricing plus a burst-sensitive billing gate made the
segmentation stage both the cost and the reliability bottleneck (~$1-2
and several lockouts per full run). One warm L4 serves a whole run's
prompts in a single call for a few cents, with no third-party lock.

The response schema matches fal's `sam-3-1/image-rle` shape ({"rle":
[...], "scores": [...]}) so every existing consumer decodes unchanged;
RLE is the COCO object form our decode_coco_rle already reads.
"""

import io
import json
import time
from pathlib import Path

import modal

app = modal.App("sam3-inference")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "torchvision",
        "transformers>=4.57",
        "accelerate",
        "pillow",
        "numpy",
    )
    .env({"HF_HOME": "/cache/huggingface"})
)

volume = modal.Volume.from_name("sam3-hf-cache", create_if_missing=True)

MODEL_ID = "facebook/sam3"


def _encode_coco_rle(mask) -> str:
    import numpy as np

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
            "counts": [int(c) for c in counts],
        }
    )


# facebook/sam3 is a gated repo: the HF account behind the `huggingface`
# Modal secret must have accepted the license at
# https://huggingface.co/facebook/sam3 or load() 401s.
@app.cls(
    image=image,
    gpu="L4",
    volumes={"/cache": volume},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=600,
)
class Sam3:
    @modal.enter()
    def load(self):
        import torch
        from transformers import Sam3Model, Sam3Processor

        t0 = time.time()
        self.torch = torch
        self.processor = Sam3Processor.from_pretrained(MODEL_ID)
        self.model = Sam3Model.from_pretrained(
            MODEL_ID, torch_dtype=torch.bfloat16
        ).to("cuda")
        self.model.eval()
        volume.commit()
        self.load_seconds = round(time.time() - t0, 1)

    def _segment_one(self, image, *, text=None, box=None):
        """One prompt -> masks + scores at full image resolution."""
        torch = self.torch
        kwargs = {"images": image, "return_tensors": "pt"}
        if text is not None:
            kwargs["text"] = text
        if box is not None:
            # absolute pixel xyxy box prompt
            kwargs["input_boxes"] = [[list(box)]]
        inputs = self.processor(**kwargs).to("cuda")
        with torch.inference_mode():
            outputs = self.model(**inputs)
        results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=0.4,
            mask_threshold=0.5,
            target_sizes=[(image.height, image.width)],
        )[0]
        masks = results.get("masks")
        scores = results.get("scores")
        if masks is None or len(masks) == 0:
            return [], []
        masks = masks.cpu().numpy()
        scores = [float(s) for s in scores.cpu().numpy()]
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:12]
        return (
            [_encode_coco_rle(masks[i] > 0.5) for i in order],
            [round(scores[i], 4) for i in order],
        )

    @modal.method()
    def segment(self, image_bytes: bytes, prompts: list[dict]) -> list[dict]:
        """Batch endpoint: each prompt is {"text": str} or
        {"box": [x1, y1, x2, y2]}. Returns one fal-schema response per
        prompt: {"rle": [...], "scores": [...]}. One warm call serves a
        whole run's vocabulary."""
        from PIL import Image

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        responses = []
        for prompt in prompts:
            try:
                rles, scores = self._segment_one(
                    image,
                    text=prompt.get("text"),
                    box=prompt.get("box"),
                )
            except Exception as error:  # surface per-prompt, don't kill batch
                responses.append({"rle": [], "scores": [], "error": str(error)[:200]})
                continue
            responses.append({"rle": rles, "scores": scores})
        return responses


@app.local_entrypoint()
def main(image_path: str, prompt: str = "fence"):
    result = Sam3().segment.remote(
        Path(image_path).read_bytes(), [{"text": prompt}]
    )
    first = result[0]
    print(
        f"{prompt!r}: {len(first.get('rle', []))} masks, "
        f"scores {first.get('scores', [])[:3]}, "
        f"error={first.get('error')}"
    )
