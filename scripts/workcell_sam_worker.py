"""One SAM 3 load on GPU 1; text prompts, then OWLv2 cart boxes on the same four photos."""
import json
from pathlib import Path
import sys
import time

from PIL import Image
import numpy as np
import torch
from transformers import Sam3Model, Sam3Processor

MODEL_ID = "facebook/sam3"
REVISION = "3c879f39826c281e95690f02c7821c4de09afae7"


def _response(result):
    masks, scores = result.get("masks"), result.get("scores")
    if masks is None or len(masks) == 0:
        return [], []
    masks = masks.float().cpu().numpy()
    scores = [float(x) for x in scores.float().cpu().numpy()]
    order = sorted(range(len(scores)), key=lambda i: -scores[i])[:12]
    encoded = []
    for i in order:
        mask = masks[i] > .5
        flat = mask.flatten(order="F").astype(np.int8)
        boundaries = np.concatenate(([0], np.flatnonzero(np.diff(flat)) + 1, [flat.size]))
        counts = np.diff(boundaries).tolist()
        if flat.size and flat[0]:
            counts.insert(0, 0)
        encoded.append(json.dumps({"size": list(mask.shape), "counts": [int(x) for x in counts]}))
    return encoded, [round(scores[i], 4) for i in order]


def main(root):
    started = time.monotonic()
    processor = Sam3Processor.from_pretrained(MODEL_ID, revision=REVISION, cache_dir="/v/sam3/huggingface/hub")
    model = Sam3Model.from_pretrained(MODEL_ID, revision=REVISION, cache_dir="/v/sam3/huggingface/hub", torch_dtype=torch.bfloat16).to("cuda").eval()
    words = json.loads((root / "words.json").read_text())
    prompts = [{"text": w} for w in words]
    images = [Image.open(root / f"source-{i}.jpg").convert("RGB") for i in range(1, 5)]
    results = []
    for image in images:
        base = processor(images=image, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            vision = model.get_vision_features(pixel_values=base["pixel_values"])
        view = []
        for word in words:
            kwargs = processor(text=word, original_sizes=base["original_sizes"], return_tensors="pt").to("cuda")
            with torch.inference_mode():
                output = model(vision_embeds=vision, **kwargs)
            masks, scores = _response(processor.post_process_instance_segmentation(
                output, threshold=.4, mask_threshold=.5, target_sizes=[(image.height, image.width)])[0])
            view.append({"rle": masks, "scores": scores})
        results.append(view)
    (root / "sam3.json").write_text(json.dumps({"prompts": prompts, "results": results}))
    deadline = time.monotonic() + 600
    while not (root / "cart-boxes.json").exists():
        if time.monotonic() > deadline:
            raise TimeoutError("OWLv2 cart boxes did not arrive")
        time.sleep(.1)
    boxes = json.loads((root / "cart-boxes.json").read_text())["results"]
    if len(boxes) != 4:
        raise ValueError("Expected one cart box for each image")
    cart_results = []
    for i, box_rows in enumerate(boxes, 1):
        # OWLv2 boxes are on MapAnything's canonical raster, not the raw JPEG.
        image = Image.open(root / f"photo-{i}.png").convert("RGB")
        base = processor(images=image, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            vision = model.get_vision_features(pixel_values=base["pixel_values"])
        # Evaluate all object proposals: the highest-scoring "work platform"
        # can be the folded guard rather than the cart itself.
        masks, scores = [], []
        for pick in box_rows:
            kwargs = processor(input_boxes=[[pick["box"]]], original_sizes=base["original_sizes"], return_tensors="pt").to("cuda")
            kwargs["input_boxes"] = kwargs["input_boxes"].to(model.dtype)
            with torch.inference_mode():
                output = model(vision_embeds=vision, **kwargs)
            candidate_masks, candidate_scores = _response(processor.post_process_instance_segmentation(
                output, threshold=.4, mask_threshold=.5, target_sizes=[(image.height, image.width)])[0])
            masks.extend(candidate_masks); scores.extend(candidate_scores)
        cart_results.append({"rle": masks, "scores": scores})
    (root / "cart-masks.json").write_text(json.dumps({"results": cart_results}))
    (root / "sam-timing.json").write_text(json.dumps({"containerSeconds": time.monotonic() - started}))
    print(json.dumps({"sam3": "ok", "seconds": time.monotonic() - started}), flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
