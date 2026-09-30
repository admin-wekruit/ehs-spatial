"""MapAnything subprocess on GPU 0 inside the one-shot two-GPU container."""

import base64
import gzip
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, "/repo")
from modal_apps.mapanything_app import CODE_REV, MODEL_ID, _encode_array, load_images_with_metadata
from mapanything.models import MapAnything


def run(images, out):
    started = time.monotonic()
    model = MapAnything.from_pretrained(MODEL_ID).to("cuda")
    model.eval()
    loaded = time.monotonic() - started
    views, metadata = load_images_with_metadata(images)
    with torch.inference_mode():
        predictions = model.infer(views, memory_efficient_inference=False,
                                  use_amp=True, amp_dtype="bf16",
                                  apply_mask=False, mask_edges=True)
    if len(predictions) != len(images):
        raise ValueError("Geometry returned a different frame count")
    def array(tensor):
        return tensor[0].float().cpu().numpy()
    for i, (prediction, info) in enumerate(zip(predictions, metadata), 1):
        rgb = (np.clip(array(prediction["img_no_norm"]), 0, 1) * 255 + .5).astype(np.uint8)
        if not np.array_equal(rgb, info.pop("_canonical_rgb")):
            raise ValueError("Geometry changed the verified image raster")
        points = array(prediction["pts3d"]).astype(np.float32)
        confidence = array(prediction["conf"]).astype(np.float32)
        mask = array(prediction["non_ambiguous_mask"]).astype(bool)
        depth = array(prediction["depth_z"]).astype(np.float32)
        if depth.ndim == 3:
            depth = depth[..., 0]
        dy, dx = np.gradient(depth)
        mask &= np.hypot(dx, dy) / np.maximum(depth, 1e-6) < .08
        frame = {"image": _encode_array(rgb), "pts3d": _encode_array(points),
                 "conf": _encode_array(confidence), "non_ambiguous_mask": _encode_array(mask),
                 "camera_poses": _encode_array(array(prediction["camera_poses"]).astype(np.float32)),
                 "intrinsics": _encode_array(array(prediction["intrinsics"]).astype(np.float32)),
                 **info, "model_id": MODEL_ID, "code_revision": CODE_REV, "backend": "modal-one-shot"}
        with gzip.open(out / f"frame_{i:04d}.json.gz", "wt") as stream:
            json.dump(frame, stream)
        Image.fromarray(rgb).save(out / f"photo-{i}.png")
    (out / "geometry-timing.json").write_text(json.dumps({
        "wallSecondsInContainer": time.monotonic() - started, "modelLoadSeconds": loaded,
        "frames": len(predictions)}) + "\n")
    print(json.dumps({"status": "ok", "frames": len(predictions), "seconds": time.monotonic() - started}), flush=True)


if __name__ == "__main__":
    run(sys.argv[2:], Path(sys.argv[1]))
