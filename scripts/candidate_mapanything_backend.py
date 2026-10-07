"""Pinned MapAnything Apache through the existing runner seam, entirely local.

Only baseline canonical RGB PNGs are accepted. Native metric world points and
c2w poses are retained; the existing downstream scale anchor remains unchanged.
"""

import argparse
import base64
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from candidate_geometry_backend import _encode_array

MODEL_ID = "facebook/map-anything-apache"
MODEL_REVISION = "00f9c245bbcb60522d1ed7f9e9d88462c6e3f38a"
CODE_REVISION = "3d10cf7a3016fc0f9bb13a071ee66c47b10be0d9"
WEIGHTS_SHA256 = "fa06c0fdccefc5048e072c85935d5789b1e36b307f3859033c17f9dcb9fd5201"


def _numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    value = np.asarray(value)
    if value.shape[0] != 1:
        raise ValueError("Expected exactly one image per native prediction batch")
    return value[0]


def _require_identity_grid(rgb, source):
    if not np.isfinite(rgb).all() or np.any((rgb < 0) | (rgb > 1)):
        raise ValueError("MapAnything RGB must be finite in [0,1]")
    decoded = np.rint(rgb * 255).astype(np.uint8)
    if decoded.shape != source.shape or not np.array_equal(decoded, source):
        raise ValueError("Fixed-mask A/B requires an identity pixel grid and unchanged RGB")
    return decoded


def predictions_to_response(predictions, source_images):
    """Serialize native world points without deriving replacement pinhole geometry."""
    import trimesh

    if len(predictions) != len(source_images) or not 1 <= len(predictions) <= 4:
        raise ValueError("Expected one native prediction for each of one to four images")
    frames, points, colors = [], [], []
    for prediction, source in zip(predictions, source_images):
        image = _require_identity_grid(_numpy(prediction["img_no_norm"]), source)
        h, w = image.shape[:2]
        pts = _numpy(prediction["pts3d"]).astype(np.float32)
        conf = _numpy(prediction["conf"]).astype(np.float32)
        native_mask = _numpy(prediction["non_ambiguous_mask"])
        depth = _numpy(prediction["depth_z"])
        if depth.shape == (h, w, 1):
            depth = depth[..., 0]
        pose = _numpy(prediction["camera_poses"]).astype(np.float32)
        k = _numpy(prediction["intrinsics"]).astype(np.float32)
        if (min(h, w) < 2 or pts.shape != (h, w, 3) or conf.shape != (h, w)
                or native_mask.shape != (h, w) or depth.shape != (h, w)):
            raise ValueError("Native geometry, confidence and masks must share the image grid")
        if not np.isin(native_mask, [0, 1]).all():
            raise ValueError("Native non-ambiguous mask must be binary")
        if (pose.shape != (4, 4) or not np.isfinite(pose).all()
                or not np.allclose(pose[3], [0, 0, 0, 1])
                or not np.allclose(pose[:3, :3] @ pose[:3, :3].T, np.eye(3), atol=1e-3)
                or not np.isclose(np.linalg.det(pose[:3, :3]), 1, atol=1e-3)):
            raise ValueError("Native camera pose must be a finite rigid c2w transform")
        if (k.shape != (3, 3) or not np.isfinite(k).all()
                or not np.allclose(k[2], [0, 0, 1]) or min(k[0, 0], k[1, 1]) <= 0):
            raise ValueError("Native intrinsics must be finite with positive focal lengths")
        mask = (native_mask.astype(bool) & np.isfinite(pts).all(-1)
                & np.isfinite(conf) & np.isfinite(depth) & (depth > 0))
        # Same declared relative z-depth edge trim as the existing Modal wrapper.
        gy, gx = np.gradient(depth)
        mask &= np.hypot(gx, gy) / np.maximum(depth, 1e-6) < 0.08
        payload = {"image": image, "pts3d": pts, "conf": conf,
                   "non_ambiguous_mask": mask, "camera_poses": pose, "intrinsics": k}
        frames.append(json.dumps({key: _encode_array(value)
                                  for key, value in payload.items()}).encode())
        points.append(pts[mask])
        colors.append(image[mask])
    if not any(len(value) for value in points):
        raise ValueError("MapAnything produced no valid points after the declared edge filter")
    cloud = trimesh.PointCloud(np.concatenate(points), colors=np.concatenate(colors))
    return {"data": frames, "point_cloud": cloud.export(file_type="glb")}


class MapAnythingRunner:
    def __init__(self, vendor_dir, model_dir, *, device="cpu"):
        if device not in {"cpu", "mps", "cuda"}:
            raise ValueError("Device must be cpu/mps/cuda")
        vendor_dir, model_dir = Path(vendor_dir).resolve(), Path(model_dir).resolve()
        revision = subprocess.check_output(
            ["git", "-C", str(vendor_dir), "rev-parse", "HEAD"], text=True).strip()
        if revision != CODE_REVISION:
            raise ValueError(f"MapAnything source must be pinned at {CODE_REVISION}")
        weights = model_dir / "model.safetensors"
        with weights.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        if digest != WEIGHTS_SHA256:
            raise ValueError("Checkpoint SHA256 does not match the pinned Apache weights")
        if importlib.metadata.version("uniception") != "0.1.7":
            raise ValueError("Pinned MapAnything requires uniception==0.1.7")
        sys.path.insert(0, str(vendor_dir))
        import torch
        from mapanything.models import MapAnything
        from mapanything.models.external.dinov2.hub.backbones import dinov2_vitg14
        from mapanything.utils.image import load_images, rgb

        def local_encoder(repo, model, *args, **kwargs):
            if (repo != "facebookresearch/dinov2" or model != "dinov2_vitg14" or args
                    or kwargs.get("pretrained") is not False
                    or set(kwargs) - {"pretrained", "force_reload"}
                    or kwargs.get("force_reload", False)):
                raise ValueError("Unrecognized Torch Hub call; offline loading refuses downloads")
            return dinov2_vitg14(pretrained=False)

        start = time.perf_counter()
        # ponytail: single-process loader only. Route the one hard-coded Hub I/O
        # to the pinned vendored factory, restore immediately; concurrent loading
        # would need an upstream local-source parameter instead of a global hook.
        with patch.object(torch.hub, "load", side_effect=local_encoder):
            self.model = MapAnything.from_pretrained(
                str(model_dir), local_files_only=True, strict=True).to(device).eval()
        self.torch, self.device = torch, device
        self.load_images, self.rgb = load_images, rgb
        self.metadata = {
            "model_id": MODEL_ID, "model_revision": MODEL_REVISION, "weights_sha256": digest,
            "code_revision": revision, "torch_version": torch.__version__, "uniception_version": "0.1.7",
            "device": device, "precision": "float32; use_amp=False", "process_res": 518,
            "resize_mode": "fixed_size at baseline canonical size", "require_identity_grid": True,
            "hub_source": f"{vendor_dir}/mapanything/models/external/dinov2/hub/backbones.py@{revision}",
            "strict_state_dict": True, "scale_semantics": "native predicted metric world points; anchor unchanged",
            "camera_convention": "native OpenCV c2w; right/down/forward; no pose inversion",
            "mask_semantics": "native non_ambiguous_mask AND finite positive depth/confidence/points AND relative depth edge <0.08",
            "confidence_semantics": "native learned exp confidence; not calibrated probability",
            "memory_efficient_inference": True, "minibatch_size": 1,
            "model_load_seconds": time.perf_counter() - start,
        }

    def __call__(self, model_identifier, *, input):
        sources = input.get("inputs")
        if not isinstance(sources, list) or not 1 <= len(sources) <= 4:
            raise ValueError("Expected one to four canonical image data URIs")
        start = time.perf_counter()
        images, records = [], []
        with tempfile.TemporaryDirectory() as temporary:
            paths = []
            for index, uri in enumerate(sources):
                if not isinstance(uri, str) or not uri.startswith("data:image/png;base64,"):
                    raise ValueError("Only self-contained canonical PNG data URIs are accepted")
                content = base64.b64decode(uri.split(",", 1)[1], validate=True)
                with Image.open(io.BytesIO(content)) as image:
                    if image.format != "PNG" or image.mode != "RGB" or image.getexif().get(274, 1) != 1:
                        raise ValueError("Input must be a canonical RGB PNG with no EXIF rotation")
                    if max(image.size) != 518 or any(value % 14 for value in image.size):
                        raise ValueError("Baseline canonical grid must have longest side 518 and 14-pixel alignment")
                    images.append(np.array(image))
                path = Path(temporary) / f"view_{index:02d}.png"
                path.write_bytes(content)
                paths.append(str(path))
                records.append({"sha256": hashlib.sha256(content).hexdigest(),
                                "source_size_wh": list(images[-1].shape[1::-1]),
                                "processed_size_wh": list(images[-1].shape[1::-1]),
                                "source_to_processed": np.eye(3).tolist(),
                                "input_role": "baseline canonical RGB; original provenance in run manifest"})
            if len({image.shape for image in images}) != 1:
                raise ValueError("All canonical frames in a run must have identical dimensions")
            views = self.load_images(paths, resize_mode="fixed_size", size=images[0].shape[1::-1],
                                     norm_type="dinov2", patch_size=14)
            if len(views) != len(images):
                raise ValueError("Official preprocessing dropped an input image")
            for view, image in zip(views, images):
                _require_identity_grid(self.rgb(view["img"], "dinov2")[0], image)
        with self.torch.inference_mode():
            predictions = self.model.infer(views, memory_efficient_inference=True, minibatch_size=1,
                                          use_amp=False, apply_mask=False, mask_edges=False)
        response = predictions_to_response(predictions, images)
        self.metadata.update(frames=records, inference_and_encoding_seconds=time.perf_counter() - start)
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
        parser.error("Output must be new or empty; existing experiment artifacts are immutable")
    from ehs_spatial.providers.map_anything import MapAnythingAdapter

    runner = MapAnythingRunner(args.vendor_dir, args.model_dir, device=args.device)
    frames, cloud = MapAnythingAdapter(runner=runner).run(args.inputs, output)
    (output / "candidate_manifest.json").write_text(json.dumps(runner.metadata, indent=2) + "\n")
    print(json.dumps({"frames": len(frames), "point_cloud": str(cloud), **runner.metadata}, indent=2))


if __name__ == "__main__":
    main()
