"""Local, image-only DA3 experiment through MapAnythingAdapter's runner seam.

No model download or service call occurs here. Supply a local official checkout
and checkpoint directory. DA3 Small/Base/Large predict relative-scale geometry;
the production pipeline's existing scale anchor must remain in an A/B.
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

import numpy as np
from PIL import Image


CHECKPOINTS = {
    "depth-anything/DA3-SMALL": (
        "da3-small", "e08cab65ca0ec38e7826075418411ab90cab4da3",
        "364492e38a3a06d221ac75da7f6621ada3f2361cd24fde11ba79091e9f40efcf",
    ),
    "depth-anything/DA3-BASE": (
        "da3-base", "f4a6c9b3c95e41c82048423d3493a81ec3fa810e",
        "e01067dc1659613083d9145a9a2547ccdbe6ccbbf83c4fe7b3e8a4e2bdae78b5",
    ),
    "depth-anything/DA3-LARGE-1.1": (
        "da3-large", "0e109ae307c5982f319a67cf6f9f99ccdc0ec97c",
        "739905c423cf0d6ccaf9e61a8401d82ba1ac32d7f4d3ee6dca8f92b377633f64",
    ),
}
DA3_CODE_REVISION = "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4"


def _encode_array(array):
    array = np.ascontiguousarray(array)
    return {"shape": list(array.shape), "dtype": str(array.dtype),
            "data": base64.b64encode(array.tobytes()).decode("ascii")}


def prediction_to_response(prediction, *, native_world_points=None):
    """Convert native DA3 z-depth and OpenCV w2c into the frozen frame schema."""
    import trimesh

    depth = np.asarray(prediction.depth, dtype=np.float32)
    rgb = np.asarray(prediction.processed_images)
    conf = np.asarray(prediction.conf, dtype=np.float32)
    intrinsics = np.asarray(prediction.intrinsics, dtype=np.float32)
    extrinsics = np.asarray(prediction.extrinsics, dtype=np.float32)
    if depth.ndim != 3 or min(depth.shape) < 1:
        raise ValueError("DA3 depth must have shape N,H,W")
    n, h, w = depth.shape
    if native_world_points is not None:
        native_world_points = np.asarray(native_world_points, dtype=np.float32)
        if native_world_points.shape != (n, h, w, 3):
            raise ValueError("Native world points must be aligned with depth")
    if h < 2 or w < 2 or rgb.shape != (n, h, w, 3) or rgb.dtype != np.uint8:
        raise ValueError("DA3 processed RGB must be uint8 and aligned with depth")
    if conf.shape != depth.shape or intrinsics.shape != (n, 3, 3):
        raise ValueError("DA3 confidence and intrinsics have incompatible shapes")
    if extrinsics.shape not in ((n, 3, 4), (n, 4, 4)):
        raise ValueError("DA3 must predict one w2c matrix per image")
    if not np.isfinite(intrinsics).all() or not np.isfinite(extrinsics).all():
        raise ValueError("DA3 camera matrices must be finite")
    if not np.allclose(intrinsics[:, 2, :], [0, 0, 1]):
        raise ValueError("DA3 intrinsics must use homogeneous pixel coordinates")
    if np.any(intrinsics[:, (0, 1), (0, 1)] <= 0):
        raise ValueError("DA3 focal lengths must be positive")
    w2c = np.broadcast_to(np.eye(4), (n, 4, 4)).copy()
    w2c[:, :extrinsics.shape[1]] = extrinsics
    if not np.allclose(w2c[:, 3], [0, 0, 0, 1]):
        raise ValueError("DA3 extrinsics must be homogeneous rigid transforms")
    rotations = w2c[:, :3, :3]
    if not np.allclose(rotations @ rotations.transpose(0, 2, 1), np.eye(3), atol=1e-3):
        raise ValueError("DA3 extrinsic rotations must be orthonormal")
    if not np.allclose(np.linalg.det(rotations), 1, atol=1e-3):
        raise ValueError("DA3 extrinsic rotations must preserve handedness")
    c2w = np.linalg.inv(w2c).astype(np.float32)
    yy, xx = np.indices((h, w), dtype=np.float32)
    pixels = np.stack([xx, yy, np.ones_like(xx)], axis=-1)
    frames, points, colors = [], [], []
    for index in range(n):
        if native_world_points is None:
            rays = pixels @ np.linalg.inv(intrinsics[index]).T
            camera_points = rays * depth[index, ..., None]
            world_points = camera_points @ c2w[index, :3, :3].T + c2w[index, :3, 3]
        else:
            world_points = native_world_points[index]
        mask = (np.isfinite(world_points).all(axis=-1)
                & np.isfinite(conf[index]) & (depth[index] > 0))
        # Same relative z-depth edge threshold as modal_apps/mapanything_app.py.
        gy, gx = np.gradient(depth[index])
        mask &= np.hypot(gx, gy) / np.maximum(depth[index], 1e-6) < 0.08
        payload = {
            "image": rgb[index], "pts3d": world_points.astype(np.float32),
            "conf": conf[index], "non_ambiguous_mask": mask,
            "camera_poses": c2w[index], "intrinsics": intrinsics[index],
        }
        frames.append(json.dumps({key: _encode_array(value)
                                  for key, value in payload.items()}).encode())
        points.append(world_points[mask])
        colors.append(rgb[index][mask])
    if not any(len(value) for value in points):
        raise ValueError("DA3 produced no valid points after the declared edge filter")
    cloud = trimesh.PointCloud(np.concatenate(points), colors=np.concatenate(colors))
    return {"data": frames, "point_cloud": cloud.export(file_type="glb")}


class DA3Runner:
    """A callable accepted by MapAnythingAdapter(runner=...)."""

    def __init__(self, vendor_dir, model_dir, *, model_id="depth-anything/DA3-LARGE-1.1",
                 device="cpu", process_res=518):
        if model_id not in CHECKPOINTS:
            raise ValueError("Select a checkpoint recorded in CHECKPOINTS")
        if device not in {"cpu", "mps", "cuda"} or process_res < 28 or process_res % 14:
            raise ValueError("Device must be cpu/mps/cuda; resolution must be a multiple of 14 >= 28")
        vendor_dir, model_dir = Path(vendor_dir).resolve(), Path(model_dir).resolve()
        code_revision = subprocess.check_output(
            ["git", "-C", str(vendor_dir), "rev-parse", "HEAD"], text=True).strip()
        if code_revision != DA3_CODE_REVISION:
            raise ValueError(f"DA3 source must be pinned at {DA3_CODE_REVISION}")
        model_name, revision, expected_sha = CHECKPOINTS[model_id]
        weights_path = model_dir / "model.safetensors"
        with weights_path.open("rb") as weights_file:
            actual_sha = hashlib.file_digest(weights_file, "sha256").hexdigest()
        if actual_sha != expected_sha:
            raise ValueError(f"Checkpoint SHA256 does not match {model_id} at {revision}")
        if json.loads((model_dir / "config.json").read_text())["model_name"] != model_name:
            raise ValueError("Checkpoint config does not match the selected model")
        sys.path.insert(0, str(vendor_dir / "src"))
        import torch
        from safetensors.torch import load_model
        from depth_anything_3.cfg import create_object, load_config
        from depth_anything_3.registry import MODEL_REGISTRY
        from depth_anything_3.utils.io.input_processor import InputProcessor
        from depth_anything_3.utils.io.output_processor import OutputProcessor

        start = time.perf_counter()
        # ponytail: load only the native geometry network; GS/CLI exporters are
        # outside this experiment. Its official checkpoint retains the model. prefix.
        wrapper = torch.nn.Module()
        wrapper.add_module("model", create_object(load_config(MODEL_REGISTRY[model_name])))
        load_model(wrapper, str(weights_path), strict=True)
        self.model = wrapper.model.to(device).eval()
        self.processor, self.output_processor = InputProcessor(), OutputProcessor()
        self.torch, self.device, self.process_res = torch, device, process_res
        self.metadata = {
            "model_id": model_id, "model_revision": revision, "weights_sha256": actual_sha,
            "code_revision": code_revision, "torch_version": torch.__version__,
            "device": device, "precision": "autocast" if device == "cuda" else "float32",
            "process_res": process_res, "process_res_method": "upper_bound_resize",
            "reference_view_strategy": "first", "require_identity_grid": True,
            "scale_semantics": "relative; no metric calibration applied",
            "camera_convention": "OpenCV w2c inverted to c2w; right/down/forward",
            "mask_semantics": "finite positive z-depth and finite confidence; relative depth edge < 0.08",
            "confidence_semantics": "native depth confidence; not a calibrated probability",
            "model_load_seconds": time.perf_counter() - start,
        }

    def __call__(self, model_identifier, *, input):
        sources = input.get("inputs")
        if not isinstance(sources, list) or not 1 <= len(sources) <= 4:
            raise ValueError("Expected one to four input image data URIs")
        torch = self.torch
        tensors, transforms, source_records = [], [], []
        start = time.perf_counter()
        for uri in sources:
            if not isinstance(uri, str) or not uri.startswith("data:image/") or ";base64," not in uri:
                raise ValueError("Only self-contained image data URIs are accepted")
            content = base64.b64decode(uri.split(",", 1)[1], validate=True)
            with Image.open(io.BytesIO(content)) as source:
                image = source.convert("RGB")
            # Passing identity K through the official processor measures its
            # source-to-grid transform; it is metadata, never a model condition.
            tensor, _, transform = self.processor(
                [image], intrinsics=np.eye(3, dtype=np.float32)[None],
                process_res=self.process_res, process_res_method="upper_bound_resize",
                sequential=True,
            )
            height, width = tensor.shape[-2:]
            if (width, height) != image.size or not np.array_equal(transform[0].numpy(), np.eye(3)):
                raise ValueError("This fixed-mask A/B requires baseline canonical images and identity preprocessing")
            tensors.append(tensor[0])
            transforms.append(transform[0].numpy())
            source_records.append({
                "sha256": hashlib.sha256(content).hexdigest(),
                "input_role": "baseline canonical image; original photo provenance belongs in experiment manifest",
                "source_size_wh": list(image.size), "processed_size_wh": [width, height],
                "source_to_processed": transforms[-1].tolist(),
                "pixel_convention": "DA3/OpenCV integer pixel coordinates",
            })
        if len({tuple(tensor.shape) for tensor in tensors}) != 1:
            raise ValueError("Canonical images within one scene must have identical shapes")
        images_cpu = torch.stack(tensors)
        images = images_cpu[None].to(self.device)
        amp = nullcontext()
        if self.device == "cuda":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            amp = torch.autocast("cuda", dtype=dtype)
        with torch.inference_mode(), amp:
            raw = self.model(images, infer_gs=False, use_ray_pose=False,
                             ref_view_strategy="first")
        prediction = self.output_processor(raw)
        rgb = images_cpu.permute(0, 2, 3, 1).numpy()
        rgb = rgb * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        prediction.processed_images = np.rint(np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        response = prediction_to_response(prediction)
        self.metadata.update(frames=source_records, inference_and_encoding_seconds=time.perf_counter() - start)
        response["candidate"] = self.metadata
        return response


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vendor-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--model-id", choices=CHECKPOINTS, default="depth-anything/DA3-LARGE-1.1")
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--process-res", type=int, default=518)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        parser.error("Output must be new or empty; existing experiment artifacts are immutable")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from ehs_spatial.providers.map_anything import MapAnythingAdapter

    runner = DA3Runner(args.vendor_dir, args.model_dir, model_id=args.model_id,
                       device=args.device, process_res=args.process_res)
    frames, cloud = MapAnythingAdapter(runner=runner).run(args.inputs, output)
    (output / "candidate_manifest.json").write_text(json.dumps(runner.metadata, indent=2) + "\n")
    print(json.dumps({"frame_count": len(frames), "point_cloud": str(cloud), **runner.metadata}, indent=2))


if __name__ == "__main__":
    main()
