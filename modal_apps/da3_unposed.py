"""Depth Anything 3 on unposed views: it predicts every view's camera (w2c, K) and depth in one shared frame.

Used to place a second shot of an edited clip (another camera) into the walk's map: run it on a few frames of that shot
together with walk frames whose cameras are known, then align the predicted walk cameras to the known ones
(scripts/register_cut_shot.py). One A100-80GB call, no retries.
"""
import io

import modal
import numpy as np

app = modal.App("panoptes-da3-unposed")
volume = modal.Volume.from_name("moge3-hf-cache", create_if_missing=True)  # the DA3 weights mono_room.py already cached
image = (modal.Image.debian_slim(python_version="3.11").apt_install("git", "libgl1", "libglib2.0-0")
         .pip_install("torch", "torchvision", "xformers")
         .pip_install("git+https://github.com/ByteDance-Seed/Depth-Anything-3.git", "addict")  # same image recipe as mono_room.py's DA3
         .env({"HF_HOME": "/cache/huggingface"}))


@app.function(image=image, gpu="A100-80GB", volumes={"/cache": volume}, timeout=1200, retries=0, max_containers=1)
def infer_unposed(frames, model_name):
    """frames: [png bytes] of 640x480 rasters -> npz bytes: w2c (N,3,4) OpenCV, K (N,3,3) on 640x480, depth/conf (N,480,640)."""
    import cv2
    import torch
    from depth_anything_3.api import DepthAnything3
    model = DepthAnything3.from_pretrained(model_name).to("cuda")
    images = [cv2.cvtColor(cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) for png in frames]
    with torch.inference_mode():
        out = model.inference(images, process_res=504)
    n, h, w = out.depth.shape
    k = np.asarray(out.intrinsics, np.float64).copy()
    k[:, 0] *= 640 / w
    k[:, 1] *= 480 / h
    buffer = io.BytesIO()
    np.savez_compressed(buffer, w2c=np.asarray(out.extrinsics, np.float64)[:, :3], K=k,
                        depth=np.stack([cv2.resize(d, (640, 480), interpolation=cv2.INTER_LINEAR) for d in out.depth]).astype(np.float32),
                        conf=np.stack([cv2.resize(c, (640, 480), interpolation=cv2.INTER_LINEAR) for c in out.conf]).astype(np.float16),
                        process_hw=np.array([h, w]), peak_gb=torch.cuda.max_memory_allocated() / 1e9)
    return buffer.getvalue()
