"""MapAnything on-prem service — implements GEOMETRY_HTTP_URL
(docs/BACKENDS.md).

    POST /geometry  {"inputs": [<data URI>, ...], **flags}
    ->              {"data": [<data:application/json;base64 URI per frame>],
                     "point_cloud": <data:model/gltf-binary;base64 URI>}

The client adapter accepts data: URIs for every file location, so this
service needs no file hosting. Per-frame JSON carries base64 arrays
{"shape","dtype","data"} for: image, pts3d, conf, non_ambiguous_mask,
camera_poses, intrinsics — identical to modal_apps/mapanything_app.py,
which is the authoritative inference body to port here.

Run:  uvicorn mapanything_service:app --host 0.0.0.0 --port 8802
"""

import base64
import json
import tempfile
import threading
from pathlib import Path

import numpy as np
import torch
import trimesh
from fastapi import FastAPI
from mapanything.models import MapAnything as Model
from mapanything.utils.image import load_images
from pydantic import BaseModel

MODEL_ID = "facebook/map-anything"

app = FastAPI(title="mapanything-service")
_lock = threading.Lock()
model = Model.from_pretrained(MODEL_ID).to("cuda")
model.eval()


class GeometryRequest(BaseModel):
    inputs: list[str]

    class Config:
        extra = "allow"


def _encode_array(array) -> dict:
    array = np.ascontiguousarray(array)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "data": base64.b64encode(array.tobytes()).decode("ascii"),
    }


@app.post("/geometry")
def run(request: GeometryRequest) -> dict:
    with tempfile.TemporaryDirectory() as workdir:
        paths = []
        for index, uri in enumerate(request.inputs):
            encoded = str(uri).split(",", 1)[1]
            path = Path(workdir) / f"view_{index:02d}.png"
            path.write_bytes(base64.b64decode(encoded))
            paths.append(str(path))
        views = load_images(paths)
        with _lock, torch.inference_mode():
            predictions = model.infer(
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
        depth = _np(prediction["depth_z"]).astype(np.float32)
        if depth.ndim == 3:
            depth = depth[..., 0]
        grad_y, grad_x = np.gradient(depth)
        mask &= np.hypot(grad_x, grad_y) / np.maximum(depth, 1e-6) < 0.08
        payload = json.dumps({
            "image": _encode_array(image_u8),
            "pts3d": _encode_array(pts3d),
            "conf": _encode_array(conf),
            "non_ambiguous_mask": _encode_array(mask),
            "camera_poses": _encode_array(
                _np(prediction["camera_poses"]).astype(np.float32)
            ),
            "intrinsics": _encode_array(
                _np(prediction["intrinsics"]).astype(np.float32)
            ),
        }).encode("utf-8")
        frames.append(
            "data:application/json;base64,"
            + base64.b64encode(payload).decode("ascii")
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
    return {
        "data": frames,
        "point_cloud": "data:model/gltf-binary;base64,"
        + base64.b64encode(bytes(glb)).decode("ascii"),
    }


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True, "model": MODEL_ID}
