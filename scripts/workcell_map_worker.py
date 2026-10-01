"""MapAnything subprocess on GPU 0 inside the one-shot two-GPU container."""

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, "/repo")
from modal_apps.mapanything_app import CODE_REV, MODEL_ID, _encode_array, _relative_depth_gradient, load_images_with_metadata
from mapanything.models import MapAnything


def run(images, out, *, fixed_size=None, model_revision=None, seed=0):
    started = time.monotonic()
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.reset_peak_memory_stats()
    options = {"revision": model_revision} if model_revision else {}
    model = MapAnything.from_pretrained(MODEL_ID, **options).to("cuda")
    model.eval()
    torch.cuda.synchronize()
    loaded = time.monotonic() - started
    views, metadata = load_images_with_metadata(images, fixed_size=fixed_size)
    prepared = time.monotonic()
    with torch.inference_mode():
        predictions = model.infer(views, memory_efficient_inference=False,
                                  use_amp=True, amp_dtype="bf16",
                                  apply_mask=False, mask_edges=True)
    torch.cuda.synchronize()
    inferred = time.monotonic()
    if len(predictions) != len(images):
        raise ValueError("Geometry returned a different frame count")
    def array(tensor):
        return tensor[0].float().cpu().numpy()
    summaries = []
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
        upstream_valid = int(mask.sum())
        mask &= _relative_depth_gradient(depth, info) < .08
        frame = {"image": _encode_array(rgb), "pts3d": _encode_array(points),
                 "conf": _encode_array(confidence), "non_ambiguous_mask": _encode_array(mask),
                 "depth_z": _encode_array(depth),
                 "camera_poses": _encode_array(array(prediction["camera_poses"]).astype(np.float32)),
                 "intrinsics": _encode_array(array(prediction["intrinsics"]).astype(np.float32)),
                 **info, "model_id": MODEL_ID, "model_revision": model_revision,
                 "code_revision": CODE_REV, "backend": "modal-one-shot"}
        with gzip.open(out / f"frame_{i:04d}.json.gz", "wt") as stream:
            json.dump(frame, stream)
        Image.fromarray(rgb).save(out / f"photo-{i}.png")
        A = np.asarray(info["input_mask_transform"]["input_to_canonical_pixel_centres"])
        K = array(prediction["intrinsics"])
        summaries.append({"photo": i, "sourceSha256": hashlib.sha256(Path(images[i-1]).read_bytes()).hexdigest(),
            "canonicalShapeHW": list(rgb.shape[:2]), "inputToCanonicalPixelCentres": A.tolist(),
            "K": K.tolist(), "rawK": (np.linalg.inv(A) @ K).tolist(),
            "cameraPose": array(prediction["camera_poses"]).tolist(),
            "upstreamValidPixels": upstream_valid, "validPixels": int(mask.sum()), "totalPixels": int(mask.size),
            "depthQuantiles": np.percentile(depth[mask & np.isfinite(depth)], [5, 50, 95]).tolist()})
    (out / "geometry-timing.json").write_text(json.dumps({
        "wallSecondsInContainer": time.monotonic() - started, "modelLoadSeconds": loaded,
        "preprocessSeconds": prepared - started - loaded, "inferenceSeconds": inferred - prepared,
        "serializeSeconds": time.monotonic() - inferred,
        "peakAllocatedGiB": torch.cuda.max_memory_allocated() / 2**30,
        "peakReservedGiB": torch.cuda.max_memory_reserved() / 2**30,
        "gpu": torch.cuda.get_device_name(), "torchVersion": torch.__version__,
        "cudaVersion": torch.version.cuda, "seed": seed, "modelRevision": model_revision,
        "fixedSizeWH": fixed_size, "frames": len(predictions), "frameSummaries": summaries,
        "maskPolicy": {"upstreamMaskEdges": True,
                       "localRelativeDepthGradientCutoff": .08, "referenceLongestSide": 518,
                       "localGradientFrame": "original image via saved affine",
                       "scope": "Upstream mask_edges remains part of the model preprocessing and may depend on raster resolution; not a pure neural-output ablation."}}) + "\n")
    print(json.dumps({"status": "ok", "frames": len(predictions), "seconds": time.monotonic() - started}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('out', type=Path)
    parser.add_argument('images', nargs='+')
    parser.add_argument('--fixed-size', type=int, nargs=2, metavar=('WIDTH', 'HEIGHT'))
    parser.add_argument('--model-revision')
    args = parser.parse_args()
    run(args.images, args.out, fixed_size=args.fixed_size, model_revision=args.model_revision)
