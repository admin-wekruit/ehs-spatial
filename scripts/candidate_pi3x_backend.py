"""Pinned local Pi3X runner with native point maps and measured pinhole-fit error.

The upstream network and intrinsic recovery are unchanged. Canonical input pixels
are preserved, and native learned metric scaling is retained for a separate audit.
"""

import argparse
import base64
from contextlib import nullcontext
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image


CODE_REVISION = "9fa3ddb3f8d53041f8b2738df404f62223bbaa7b"
MODEL_REVISION = "bb1deea4d7423de5b30691739cb451a3f57dc1d5"
WEIGHTS_SHA256 = "69972d6e1c4492cb4d737a84fe940e357087d81c52f5c9b7c160b49c1f41669a"


def pi3x_prediction_to_response(raw, images):
    """Fit K using upstream code; serialize original points, never refitted rays."""
    import torch
    from pi3.utils.geometry import recover_intrinsic_from_rays_d
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from candidate_geometry_backend import prediction_to_response
    from ehs_spatial.providers.map_anything import decode_encoded_array

    local = raw["local_points"].detach().float().cpu()
    points = raw["points"].detach().float().cpu().numpy()[0]
    poses = raw["camera_poses"].detach().float().cpu().numpy()[0]
    if local.ndim != 5 or local.shape[0] != 1 or local.shape[-1] != 3:
        raise ValueError("Expected one batch of native Pi3X local point maps")
    n, h, w = local.shape[1:4]
    if points.shape != (n, h, w, 3) or poses.shape != (n, 4, 4):
        raise ValueError("Pi3X native points and camera poses must match the image batch")
    expected = np.einsum("nhwj,nij->nhwi", local.numpy()[0], poses[:, :3, :3]) + poses[:, None, None, :3, 3]
    finite = np.isfinite(points).all(-1) & np.isfinite(expected).all(-1)
    if not finite.any() or not np.allclose(points[finite], expected[finite], rtol=1e-4, atol=1e-4):
        raise ValueError("Pi3X world/local point maps disagree with its native c2w poses")
    # Official example_mm.py recovers K from normalized local points and fixes
    # the principal point at ((W-1)/2, (H-1)/2). This is an estimated pinhole fit.
    rays = torch.nn.functional.normalize(local, dim=-1)
    k = recover_intrinsic_from_rays_d(rays, force_center_principal_point=True).numpy()[0]
    confidence = torch.sigmoid(raw["conf"].detach().float().cpu())[0, ..., 0].numpy()
    prediction = SimpleNamespace(
        depth=local.numpy()[0, ..., 2], conf=confidence,
        processed_images=np.asarray(images), intrinsics=k,
        extrinsics=np.linalg.inv(poses)[:, :3],
    )
    response = prediction_to_response(prediction, native_world_points=points)
    yy, xx = np.indices((h, w))
    pixels = np.stack([xx, yy], axis=-1)
    diagnostics = []
    for index, frame_bytes in enumerate(response["data"]):
        mask = decode_encoded_array(json.loads(frame_bytes)["non_ambiguous_mask"])
        if not mask.any():
            raise ValueError("Pi3X frame has no valid pixels for pinhole-fit diagnostics")
        projected = local.numpy()[0, index] @ k[index].T
        uv = projected[..., :2] / projected[..., 2:]
        residual = np.linalg.norm(uv - pixels, axis=-1)[mask]
        diagnostics.append({
            "median_px": float(np.median(residual)),
            "p95_px": float(np.percentile(residual, 95)),
            "max_px": float(residual.max()),
            "p95_image_height_fraction": float(np.percentile(residual, 95) / h),
            "valid_fraction": float(mask.mean()),
        })
    response["native_pinhole_fit"] = diagnostics
    return response


class Pi3XRunner:
    def __init__(self, vendor_dir, model_dir, *, device="cpu"):
        if device not in {"cpu", "mps", "cuda"}:
            raise ValueError("Device must be cpu/mps/cuda")
        vendor_dir, model_dir = Path(vendor_dir).resolve(), Path(model_dir).resolve()
        revision = subprocess.check_output(
            ["git", "-C", str(vendor_dir), "rev-parse", "HEAD"], text=True).strip()
        if revision != CODE_REVISION:
            raise ValueError(f"Pi3X source must be pinned at {CODE_REVISION}")
        weights = model_dir / "model.safetensors"
        with weights.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != WEIGHTS_SHA256:
            raise ValueError("Pi3X checkpoint SHA256 does not match the recorded revision")
        sys.path.insert(0, str(vendor_dir))
        import torch
        from safetensors.torch import load_model
        from pi3.models.pi3x import Pi3X

        start = time.perf_counter()
        model = Pi3X().eval()
        load_model(model, str(weights), strict=True)
        # Official image-only inference explicitly removes unused prior branches.
        model.disable_multimodal(free_cuda_cache=False)
        self.model = model.to(device)
        self.torch, self.device = torch, device
        self.metadata = {
            "model_id": "yyfz233/Pi3X", "model_revision": MODEL_REVISION,
            "weights_sha256": digest, "code_revision": revision,
            "torch_version": torch.__version__, "device": device,
            "precision": "autocast" if device == "cuda" else "float32",
            "model_load_seconds": time.perf_counter() - start,
            "require_identity_grid": True, "process_res_method": "none; canonical RGB retained",
            "reference_view_strategy": "permutation-equivariant; input frame order preserved",
            "scale_semantics": "native learned approximate metric prior; not measured ground truth",
            "confidence_semantics": "upstream sigmoid(logits), not an empirically calibrated probability",
            "mask_semantics": "finite positive z-depth and finite confidence; relative depth edge < 0.08",
            "camera_convention": "native OpenCV c2w; right/down/forward",
            "intrinsics_semantics": "native recover_intrinsic_from_rays_d; fixed centered principal point",
            "points_semantics": "native metric-scaled point maps retained; no pinhole refitting of points",
        }

    def __call__(self, model_identifier, *, input):
        sources = input.get("inputs")
        if not isinstance(sources, list) or not 1 <= len(sources) <= 4:
            raise ValueError("Expected one to four input image data URIs")
        start = time.perf_counter()
        images, records = [], []
        for uri in sources:
            if not isinstance(uri, str) or not uri.startswith("data:image/") or ";base64," not in uri:
                raise ValueError("Only self-contained image data URIs are accepted")
            content = base64.b64decode(uri.split(",", 1)[1], validate=True)
            with Image.open(io.BytesIO(content)) as source:
                image = np.array(source.convert("RGB"))
            h, w = image.shape[:2]
            if min(h, w) < 28 or h % 14 or w % 14:
                raise ValueError("Pi3X canonical inputs must have dimensions divisible by 14, at least 28")
            images.append(image)
            records.append({"sha256": hashlib.sha256(content).hexdigest(),
                            "input_role": "baseline canonical image",
                            "source_size_wh": [w, h], "processed_size_wh": [w, h],
                            "source_to_processed": np.eye(3).tolist()})
        if len({image.shape for image in images}) != 1:
            raise ValueError("Pi3X canonical images in one scene must have identical shapes")
        rgb = np.stack(images)
        torch = self.torch
        tensors = torch.from_numpy(rgb).permute(0, 3, 1, 2)[None].to(self.device).float() / 255
        amp = nullcontext()
        if self.device == "cuda":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            amp = torch.autocast("cuda", dtype=dtype)
        with torch.inference_mode(), amp:
            raw = self.model(imgs=tensors)
        response = pi3x_prediction_to_response(raw, rgb)
        metric = float(raw["metric"].detach().float().cpu().reshape(-1)[0])
        if not np.isfinite(metric) or metric <= 0:
            raise ValueError("Pi3X metric head must produce a finite positive scaling factor")
        self.metadata.update(frames=records, native_metric_factor=metric,
                             native_pinhole_fit=response["native_pinhole_fit"],
                             inference_and_encoding_seconds=time.perf_counter() - start)
        response["candidate"] = self.metadata
        return response


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vendor-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        parser.error("Output must be new or empty")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from ehs_spatial.providers.map_anything import MapAnythingAdapter

    runner = Pi3XRunner(args.vendor_dir, args.model_dir, device=args.device)
    frames, cloud = MapAnythingAdapter(runner=runner).run(args.inputs, output)
    (output / "candidate_manifest.json").write_text(json.dumps(runner.metadata, indent=2) + "\n")
    print(json.dumps({"frame_count": len(frames), "point_cloud": str(cloud), **runner.metadata}, indent=2))


if __name__ == "__main__":
    main()
