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

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.4.0",
        "torchvision==0.19.0",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "numpy",
        "pillow",
        "trimesh",
        "huggingface_hub",
        "git+https://github.com/facebookresearch/map-anything.git",
    )
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


@app.cls(
    image=image,
    gpu="A100",
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
        from mapanything.utils.image import load_images

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
            views = load_images(paths)
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
        for prediction in predictions:
            rgb = np.clip(_np(prediction["img_no_norm"]), 0.0, 1.0)
            image_u8 = (rgb * 255.0 + 0.5).astype(np.uint8)
            pts3d = _np(prediction["pts3d"]).astype(np.float32)
            conf = _np(prediction["conf"]).astype(np.float32)
            mask = _np(prediction["non_ambiguous_mask"]).astype(bool)
            # the model's mask carries no edge trim; depth-discontinuity
            # pixels are flying-point noise for downstream plane fits —
            # cut them the way the reference wrapper does
            depth = _np(prediction["depth_z"]).astype(np.float32)
            if depth.ndim == 3:
                depth = depth[..., 0]
            grad_y, grad_x = np.gradient(depth)
            relative = np.hypot(grad_x, grad_y) / np.maximum(depth, 1e-6)
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
def main(image_paths: str):
    from pathlib import Path

    uris = [
        "data:image/png;base64,"
        + base64.b64encode(Path(p).read_bytes()).decode("ascii")
        for p in image_paths.split(",")
    ]
    result = MapAnything().run.remote({"inputs": uris})
    first = json.loads(result["data"][0])
    print(
        f"{len(result['data'])} frames; frame0 keys {sorted(first)}; "
        f"pts3d shape {first['pts3d']['shape']}; "
        f"glb {len(result['point_cloud'])} bytes"
    )
