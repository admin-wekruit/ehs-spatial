"""MapAnything multiview reconstruction on Modal — self-hosted geometry.

Deploy:  uv run modal deploy modal_apps/mapanything_app.py
Smoke:   uv run modal run modal_apps/mapanything_app.py --image-paths a.jpg,b.jpg

Why: replicate cold boots add 1-2 unpredictable minutes to every run and
per-second pricing meters the whole boot. One warm A100 here serves a
4-view capture in seconds, scale-to-zero between runs.

Contract: same request/response the replicate Cog wrapper speaks
(ehs_spatial/backends.py). Input {"inputs": [dataURI, ...], **flags};
output {"data": [<per-frame JSON bytes>, ...], "point_cloud": <GLB bytes>}
where each frame JSON carries base64 arrays {"shape", "dtype", "data"}
for the keys parse_frame_json reads: image, pts3d, conf,
non_ambiguous_mask, camera_poses, intrinsics.
"""

import base64
import io
import json

import modal

app = modal.App("mapanything-inference")
CODE_REV = "3d10cf7a3016fc0f9bb13a071ee66c47b10be0d9"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.4.0",
        "torchvision==0.19.0",
        "torchaudio==2.4.0",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "numpy",
        "pillow",
        "trimesh",
        "huggingface_hub",
        "protobuf>=5,<7",
        f"git+https://github.com/facebookresearch/map-anything.git@{CODE_REV}",
    )
    .run_commands("python -c 'import numpy as np, torch, torchvision, torchaudio; from mapanything.models import MapAnything; from mapanything.utils.image import load_images; assert torch.__version__.startswith(\"2.4.0\"); assert torchaudio.__version__.startswith(\"2.4.0\"); assert np.array_equal(torch.from_numpy(np.arange(3)).numpy(), np.arange(3)); print(\"PASS: pinned runtime imports and NumPy tensor roundtrip; no model loaded\")'")
    .env({"HF_HOME": "/cache/huggingface"})
)

volume = modal.Volume.from_name("mapanything-hf-cache", create_if_missing=True)

MODEL_ID = "facebook/map-anything"


def _encode_array(array) -> dict:
    """Inverse of ehs_spatial decode_encoded_array."""
    import numpy as np

    array = np.ascontiguousarray(array)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "data": base64.b64encode(array.tobytes()).decode("ascii"),
    }


def _relative_depth_gradient(depth, metadata):
    """Relative depth change over a fixed original-image pixel footprint."""
    import numpy as np

    dy, dx = np.gradient(depth)
    A = np.asarray(metadata['input_mask_transform']['input_to_canonical_pixel_centres'])
    raw_gradient = np.stack([dx, dy], axis=-1) @ A[:2, :2]
    original = metadata['original_image']
    raw_step = max(original['height'], original['width']) / 518
    return np.linalg.norm(raw_gradient, axis=-1) * raw_step / np.maximum(depth, 1e-6)


def load_images_with_metadata(paths, *, fixed_size=None):
    """Observe the exact upstream raster transform; never infer it from aspect ratio."""
    import numpy as np
    from PIL import Image, ImageOps
    from mapanything.utils.image import load_images, rgb
    from mapanything.utils.cropping import rescale_image_and_other_optional_info, crop_image_and_other_optional_info

    if fixed_size is not None:
        if (not isinstance(fixed_size, (tuple, list)) or len(fixed_size) != 2 or
                any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 or v % 14 for v in fixed_size)):
            raise ValueError("fixed_size must be positive width/height multiples of 14")
        views = load_images(paths, resize_mode="fixed_size", size=tuple(fixed_size))
    else:
        views = load_images(paths)
    if len(views) != len(paths):
        raise ValueError("Upstream skipped an input image")
    metadata = []
    for path, view in zip(paths, views):
        with Image.open(path) as image:
            original = ImageOps.exif_transpose(image).convert("RGB")
        h, w = map(int, view["true_shape"][0])
        # Replay the official resize helper to get actual rounded raster size;
        # its scalar camera-intrinsics scaling is not the pixel-centre affine.
        resized = rescale_image_and_other_optional_info(original, np.array([w, h]))[0]
        rw, rh = resized.size
        left, top = (rw-w)//2, (rh-h)//2
        crop = [left, top, left+w, top+h]
        canonical = np.asarray(crop_image_and_other_optional_info(resized, crop)[0])
        loaded = np.rint(rgb(view["img"][0], view["data_norm_type"][0])*255).astype(np.uint8)
        if not np.array_equal(canonical, loaded):
            raise ValueError("Recorded preprocessing does not reproduce the actual upstream model input")
        sx, sy = rw/original.width, rh/original.height
        affine = [[sx, 0, (sx-1)/2-left], [0, sy, (sy-1)/2-top], [0, 0, 1]]
        metadata.append({"original_image": {"width": original.width, "height": original.height},
            "preprocessing": {"resize_mode": "fixed_size" if fixed_size else "fixed_mapping",
                              "requested_size_wh": list(fixed_size) if fixed_size else None},
            "alpha_mask": _encode_array(np.ones((h, w), np.uint8)),
            "input_mask_transform": {"resized_shape_hw": [rh, rw], "crop_xyxy": crop,
                "input_to_canonical_pixel_centres": affine,
                "source": f"mapanything.utils.image.load_images@{CODE_REV}",
                "rgb_replay_pixel_exact": True}, "_canonical_rgb": canonical})
    return views, metadata


@app.cls(
    image=image,
    gpu="A100-80GB:2",
    volumes={"/cache": volume},
    secrets=[modal.Secret.from_name("huggingface")],
    scaledown_window=180,
    timeout=900,
)
class MapAnything:
    @modal.enter()
    def load(self):
        import torch
        from mapanything.models import MapAnything as Model

        self.torch = torch
        self.model = Model.from_pretrained(MODEL_ID).to("cuda")
        self.model.eval()
        volume.commit()

    @modal.method()
    def run(self, input: dict) -> dict:
        import tempfile
        from pathlib import Path

        import numpy as np
        import trimesh

        torch = self.torch
        sources = input.get("inputs") or []
        if not sources:
            raise ValueError("input.inputs must hold at least one data URI")
        with tempfile.TemporaryDirectory() as workdir:
            paths = []
            for index, uri in enumerate(sources):
                encoded = str(uri).split(",", 1)[1]
                path = Path(workdir) / f"view_{index:02d}.png"
                path.write_bytes(base64.b64decode(encoded))
                paths.append(str(path))
            views, metadata = load_images_with_metadata(paths)
            with torch.inference_mode():
                # apply_mask=False: zeroing masked points would blank the
                # returned mask and leave 0,0,0 landmines in pts3d — the
                # pipeline filters by valid_mask, so keep points intact and
                # the model's own non_ambiguous_mask authoritative
                predictions = self.model.infer(
                    views,
                    memory_efficient_inference=False,
                    use_amp=True,
                    amp_dtype="bf16",
                    apply_mask=False,
                    mask_edges=True,
                )

        def _np(tensor):
            return tensor[0].float().cpu().numpy()

        frames, cloud_points, cloud_colors = [], [], []
        if len(predictions) != len(metadata):
            raise ValueError("Prediction count disagrees with input frames")
        for prediction, source_metadata in zip(predictions, metadata):
            rgb = np.clip(_np(prediction["img_no_norm"]), 0.0, 1.0)
            image_u8 = (rgb * 255.0 + 0.5).astype(np.uint8)
            if not np.array_equal(image_u8, source_metadata.pop("_canonical_rgb")):
                raise ValueError("Prediction image grid changed after the verified preprocessing")
            pts3d = _np(prediction["pts3d"]).astype(np.float32)
            conf = _np(prediction["conf"]).astype(np.float32)
            mask = _np(prediction["non_ambiguous_mask"]).astype(bool)
            # the model's mask carries no edge trim; depth-discontinuity
            # pixels are flying-point noise for downstream plane fits —
            # cut them the way the reference wrapper does
            depth = _np(prediction["depth_z"]).astype(np.float32)
            if depth.ndim == 3:
                depth = depth[..., 0]
            relative = _relative_depth_gradient(depth, source_metadata)
            mask &= relative < 0.08
            pose = _np(prediction["camera_poses"]).astype(np.float32)
            intrinsics = _np(prediction["intrinsics"]).astype(np.float32)
            frames.append(
                json.dumps(
                    {
                        "image": _encode_array(image_u8),
                        "pts3d": _encode_array(pts3d),
                        "conf": _encode_array(conf),
                        "non_ambiguous_mask": _encode_array(mask),
                        "camera_poses": _encode_array(pose),
                        "intrinsics": _encode_array(intrinsics),
                        **source_metadata,
                        "model_id": MODEL_ID,
                        "code_revision": CODE_REV,
                        "backend": "modal",
                    }
                ).encode("utf-8")
            )
            cloud_points.append(pts3d[mask])
            cloud_colors.append(image_u8[mask])

        cloud = trimesh.PointCloud(
            np.concatenate(cloud_points) if cloud_points else np.zeros((0, 3)),
            colors=np.concatenate(cloud_colors) if cloud_colors else None,
        )
        glb = cloud.export(file_type="glb")
        if isinstance(glb, str):
            glb = glb.encode("utf-8")
        return {"data": frames, "point_cloud": bytes(glb)}


@app.local_entrypoint()
def main(image_paths: str, out: str = ""):
    import gzip
    from pathlib import Path
    import time

    started = time.monotonic()
    uris = [
        "data:image/png;base64,"
        + base64.b64encode(Path(p).read_bytes()).decode("ascii")
        for p in image_paths.split(",")
    ]
    result = MapAnything().run.remote({"inputs": uris})
    elapsed = time.monotonic() - started
    first = json.loads(result["data"][0])
    if out:
        destination = Path(out)
        destination.mkdir(parents=True, exist_ok=True)
        for index, payload in enumerate(result["data"], 1):
            with gzip.open(destination / f"frame_{index:04d}.json.gz", "wb") as stream:
                stream.write(payload)
        (destination / "point_cloud.glb").write_bytes(result["point_cloud"])
        (destination / "geometry-timing.json").write_text(json.dumps({
            "sourceImages": image_paths.split(","), "wallSecondsIncludingColdStart": elapsed,
            "frames": len(result["data"]), "pointCloudBytes": len(result["point_cloud"]),
        }, indent=2) + "\n")
    print(
        f"{len(result['data'])} frames; frame0 keys {sorted(first)}; "
        f"pts3d shape {first['pts3d']['shape']}; "
        f"glb {len(result['point_cloud'])} bytes"
    )
