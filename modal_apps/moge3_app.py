"""MoGe-3 inference on Modal.

Deploy:  modal deploy modal_apps/moge3_app.py
Eval:    modal run modal_apps/moge3_app.py --image-path incoming/gen-01.png

MoGe-3 (Ruicheng/moge-3-vitl) needs FlexGEMM->Triton, so NVIDIA GPU only.
Weights cache in a modal.Volume so warm starts skip the HF download.
"""

import io
import json
import time
from pathlib import Path

import modal

app = modal.App("moge3-inference")

# ponytail: latest default PyPI torch wheels (cu12x + bundled triton) instead of
# a pinned cu121 index — FlexGEMM's Aug-2026 pin wants a modern triton; pin
# torch==2.5.1+cu121 here if this ever breaks.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential")
    .pip_install("torch", "torchvision")
    .pip_install(
        "git+https://github.com/microsoft/MoGe.git",
        "huggingface_hub",
        "opencv-python-headless",
        "trimesh",
    )
    .env({"HF_HOME": "/cache/huggingface"})
)

volume = modal.Volume.from_name("moge3-hf-cache", create_if_missing=True)

MAX_SIDE = 2048  # keep payloads sane; SAM masks live at <=1448px anyway


@app.cls(image=image, gpu="L4", volumes={"/cache": volume}, timeout=900)
class MoGe3:
    @modal.enter()
    def load(self):
        import torch
        from moge.model.v3 import MoGeModel

        t0 = time.time()
        self.torch = torch
        self.model = MoGeModel.from_pretrained("Ruicheng/moge-3-vitl").to("cuda")
        self.model.eval()
        volume.commit()
        self.load_seconds = round(time.time() - t0, 1)

    @modal.method()
    def infer(self, image_bytes: bytes) -> dict:
        import cv2
        import numpy as np
        import trimesh

        raw = np.frombuffer(image_bytes, dtype=np.uint8)
        bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("could not decode image bytes")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        if max(h, w) > MAX_SIDE:
            scale = MAX_SIDE / max(h, w)
            rgb = cv2.resize(
                rgb, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA
            )

        tensor = (
            self.torch.from_numpy(rgb.copy()).float().permute(2, 0, 1) / 255.0
        ).cuda()
        t0 = time.time()
        out = self.model.infer(tensor, use_fp16=True)  # refine_steps default (3)
        infer_seconds = round(time.time() - t0, 2)

        points = out["points"].cpu().numpy().astype(np.float32)
        depth = out["depth"].cpu().numpy().astype(np.float32)
        normal = out["normal"].cpu().numpy().astype(np.float32)
        mask = out["mask"].cpu().numpy().astype(bool)
        intrinsics = out["intrinsics"].cpu().numpy().astype(np.float64)
        fov_x_deg = float(np.degrees(2 * np.arctan(0.5 / intrinsics[0, 0])))

        def npz(**arrays) -> bytes:
            buf = io.BytesIO()
            np.savez_compressed(buf, **arrays)
            return buf.getvalue()

        valid = mask & np.isfinite(points).all(axis=-1)
        cloud_pts = points[valid]
        cloud_rgb = rgb[valid]
        if len(cloud_pts) > 2_000_000:
            idx = np.random.default_rng(0).choice(
                len(cloud_pts), 2_000_000, replace=False
            )
            cloud_pts, cloud_rgb = cloud_pts[idx], cloud_rgb[idx]
        ply = trimesh.PointCloud(cloud_pts, colors=cloud_rgb).export(file_type="ply")

        return {
            "points_npz": npz(points=points, mask=mask),
            "depth_npz": npz(depth=depth, mask=mask),
            "normals_npz": npz(normal=normal),
            "ply": ply,
            "intrinsics": intrinsics.tolist(),
            "fov_x_deg": fov_x_deg,
            "shape": [int(points.shape[0]), int(points.shape[1])],
            "infer_seconds": infer_seconds,
            "model_load_seconds": self.load_seconds,
        }


@app.local_entrypoint()
def main(image_path: str):
    path = Path(image_path)
    out_dir = Path("outputs/moge3_eval") / path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    result = MoGe3().infer.remote(path.read_bytes())

    (out_dir / "points.npz").write_bytes(result.pop("points_npz"))
    (out_dir / "depth.npz").write_bytes(result.pop("depth_npz"))
    (out_dir / "normals.npz").write_bytes(result.pop("normals_npz"))
    (out_dir / "cloud.ply").write_bytes(result.pop("ply"))
    (out_dir / "meta.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"{path.name}: {result['shape']} in {result['infer_seconds']}s "
          f"(load {result['model_load_seconds']}s) -> {out_dir}")
