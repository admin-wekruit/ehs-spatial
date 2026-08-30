"""MoGe-3 on-prem service — implements MOGE_HTTP_URL (docs/BACKENDS.md).

    POST /moge  {"image_b64": ...}
    ->          {"ply_b64": ..., "intrinsics": [[...]], "fov_x_deg": float}

Run:  uvicorn moge_service:app --host 0.0.0.0 --port 8803
Same model + inference body as modal_apps/moge3_app.py (Ruicheng/moge-3-vitl,
MIT). NVIDIA GPU required (FlexGEMM -> Triton).
"""

import base64
import threading

import cv2
import numpy as np
import torch
import trimesh
from fastapi import FastAPI
from moge.model.v3 import MoGeModel
from pydantic import BaseModel

MAX_SIDE = 2048

app = FastAPI(title="moge-service")
_lock = threading.Lock()
model = MoGeModel.from_pretrained("Ruicheng/moge-3-vitl").to("cuda")
model.eval()


class MoGeRequest(BaseModel):
    image_b64: str


@app.post("/moge")
def infer(request: MoGeRequest) -> dict:
    raw = np.frombuffer(base64.b64decode(request.image_b64), dtype=np.uint8)
    bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("could not decode image bytes")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    if max(h, w) > MAX_SIDE:
        scale = MAX_SIDE / max(h, w)
        rgb = cv2.resize(
            rgb, (round(w * scale), round(h * scale)),
            interpolation=cv2.INTER_AREA,
        )
    tensor = (
        torch.from_numpy(rgb.copy()).float().permute(2, 0, 1) / 255.0
    ).cuda()
    with _lock:
        out = model.infer(tensor, use_fp16=True)

    points = out["points"].cpu().numpy().astype(np.float32)
    mask = out["mask"].cpu().numpy().astype(bool)
    intrinsics = out["intrinsics"].cpu().numpy().astype(np.float64)
    fov_x_deg = float(np.degrees(2 * np.arctan(0.5 / intrinsics[0, 0])))

    valid = mask & np.isfinite(points).all(axis=-1)
    cloud_pts, cloud_rgb = points[valid], rgb[valid]
    if len(cloud_pts) > 2_000_000:
        idx = np.random.default_rng(0).choice(
            len(cloud_pts), 2_000_000, replace=False
        )
        cloud_pts, cloud_rgb = cloud_pts[idx], cloud_rgb[idx]
    ply = trimesh.PointCloud(cloud_pts, colors=cloud_rgb).export(file_type="ply")
    if isinstance(ply, str):
        ply = ply.encode("utf-8")

    return {
        "ply_b64": base64.b64encode(bytes(ply)).decode("ascii"),
        "intrinsics": intrinsics.tolist(),
        "fov_x_deg": fov_x_deg,
    }


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True, "model": "Ruicheng/moge-3-vitl"}
