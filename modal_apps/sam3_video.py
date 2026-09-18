"""Bounded native SAM 3.1 video experiment; leaves the SAM 3 image service intact.

Access only: modal run modal_apps/sam3_video.py::access
One GPU job: modal run modal_apps/sam3_video.py::main --video-path clip.mp4 --output-path observations.json
Local check: python -m pytest tests/test_sam3_video_contract.py
"""

import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time
import urllib.error
import urllib.request

import modal

SOURCE_COMMIT = "660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7"
MODEL_ID = "facebook/sam3.1"
MODEL_REVISION = "daa63191845a41281374e725f4c9e51c7a824460"
CHECKPOINT = "sam3.1_multiplex.pt"
SAM3_REVISION = "3c879f39826c281e95690f02c7821c4de09afae7"
MAX_FRAMES = 180
MAX_SECONDS = 6.0
MAX_BYTES = 64 * 1024 * 1024
MAX_OBJECTS = 16
GPU_TIMEOUT_SECONDS = 900


def check_access(version: str = "3.1") -> dict:
    """HEAD only; do not follow the signed weight redirect or reveal credentials."""
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None

    if version not in ("3", "3.1"):
        raise ValueError("Expected SAM version 3 or 3.1")
    model_id, revision, checkpoint = ((MODEL_ID, MODEL_REVISION, CHECKPOINT) if version == "3.1"
                                     else ("facebook/sam3", SAM3_REVISION, "sam3.pt"))
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    url = f"https://huggingface.co/{model_id}/resolve/{revision}/{checkpoint}"
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    request = urllib.request.Request(url, headers=headers, method="HEAD")
    error_code = error_message = None
    try:
        with urllib.request.build_opener(NoRedirect).open(request, timeout=20) as response:
            status = response.status
    except urllib.error.HTTPError as error:
        status = error.code
        error_code = error.headers.get("X-Error-Code")
        error_message = error.headers.get("X-Error-Message")
        if error_message:
            error_message = re.sub(r"https?://\S+|hf_[\w-]+|Bearer\s+\S+", "[redacted]", error_message)[:500]
    except urllib.error.URLError:
        status = None
    return {"model_id": model_id, "revision": revision,
            "token_present": bool(token), "http_status": status,
            "error_code": error_code, "error_message": error_message,
            "accessible": status in (200, 302, 307), "downloaded_weights": False}


def inspect_clip(path: Path) -> dict:
    """Check actual decoded frame metadata before allocating a GPU."""
    if not path.is_file() or not 0 < path.stat().st_size <= MAX_BYTES:
        raise ValueError("Input must be a nonempty video of at most 64 MiB")
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_frames",
         "-show_entries", "stream=width,height:frame=best_effort_timestamp_time:format=duration",
         "-of", "json", str(path)], check=True, capture_output=True, text=True, timeout=30,
    )
    data = json.loads(probe.stdout)
    frames = data.get("frames", [])
    duration = float(data["format"]["duration"])
    if not 0 < len(frames) <= MAX_FRAMES or not 0 < duration <= MAX_SECONDS + 0.001:
        raise ValueError("Clip must contain 1–180 frames and last at most 6 seconds")
    timestamps = [float(frame["best_effort_timestamp_time"]) for frame in frames]
    if not all(math.isfinite(t) for t in timestamps) or any(
        b <= a for a, b in zip(timestamps, timestamps[1:])
    ):
        raise ValueError("Clip requires finite, strictly increasing frame timestamps")
    stream = data["streams"][0]
    width, height = int(stream["width"]), int(stream["height"])
    if width <= 0 or height <= 0:
        raise ValueError("Invalid video dimensions")
    return {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "frame_count": len(frames), "width": width, "height": height,
            "duration_seconds": duration, "frame_timestamps_seconds": timestamps}


def encode_frame(response: dict, clip: dict, source_start_frame: int = 0) -> dict:
    """Preserve native IDs/order/boxes; validate and reuse the existing mask codec."""
    import numpy as np
    from ehs_spatial.providers.sam3 import decode_coco_rle, encode_coco_rle

    index = response["frame_index"]
    if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
        raise ValueError("Native frame_index must be an integer")
    if not 0 <= index < clip["frame_count"]:
        raise ValueError("Native frame_index is outside the input clip")
    out = response["outputs"]
    ids, scores, boxes, masks = [np.asarray(out[key]) for key in (
        "out_obj_ids", "out_probs", "out_boxes_xywh", "out_binary_masks")]
    n = len(ids)
    if ids.shape != (n,) or ids.dtype.kind not in "iu" or len(set(ids.tolist())) != n:
        raise ValueError("Native IDs must be a unique integer vector")
    if n > MAX_OBJECTS or np.any(ids < 0):
        raise ValueError("Native IDs exceed the object contract")
    if scores.shape != (n,) or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("Invalid native scores")
    if boxes.shape != (n, 4) or not np.isfinite(boxes).all() or np.any((boxes < 0) | (boxes > 1)):
        raise ValueError("Invalid normalized native xywh boxes")
    if masks.dtype != np.bool_ or masks.shape != (n, clip["height"], clip["width"]):
        raise ValueError("Native masks must be boolean arrays at input resolution")
    objects = []
    for obj_id, score, box, mask in zip(ids, scores, boxes, masks, strict=True):
        rle = encode_coco_rle(mask)
        if not np.array_equal(decode_coco_rle(rle).astype(bool), mask):
            raise ValueError("Mask RLE failed exact round-trip validation")
        objects.append({"track_id": int(obj_id), "label": "person", "score": float(score),
                        "box_xywh_normalized": box.tolist(), "rle": rle,
                        "mask_area_pixels": int(mask.sum())})
    return {"frame_index": int(index), "source_frame_index": source_start_frame + int(index),
            "timestamp_seconds": clip["frame_timestamps_seconds"][index], "objects": objects}


def validate_frames(frames: list[dict], clip: dict) -> None:
    if [frame["frame_index"] for frame in frames] != list(range(clip["frame_count"])):
        raise ValueError("Expected exactly one ordered output for every input frame, including empty frames")


# Separate apps keep the access-only command independent of the GPU image/model.
access_app = modal.App("sam3-video-access")
access_image = modal.Image.debian_slim(python_version="3.12")


@access_app.function(image=access_image, secrets=[modal.Secret.from_name("huggingface")],
                     timeout=30, retries=0, max_containers=1)
def access_status() -> dict:
    return {"models": [check_access("3"), check_access("3.1")]}


@access_app.local_entrypoint()
def access():
    result = access_status.remote()
    print(json.dumps(result, indent=2))
    if not any(model["accessible"] for model in result["models"]):
        raise SystemExit(1)


app = modal.App("sam3-native-video-mvp")
video_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "ffmpeg")
    .pip_install("torch==2.10.0", "torchvision==0.25.0",
                 index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("setuptools==80.9.0", "numpy==1.26.4", "pydantic>=2,<3", "pillow",
                 "pycocotools", "scipy", "psutil", "opencv-python-headless<4.12", "einops")
    .pip_install(f"sam3 @ git+https://github.com/facebookresearch/sam3.git@{SOURCE_COMMIT}")
    .env({"HF_HOME": "/cache/huggingface", "HF_HUB_DISABLE_PROGRESS_BARS": "1"})
    .add_local_python_source("ehs_spatial")
)
cache = modal.Volume.from_name("sam3-hf-cache")


@app.function(image=video_image, gpu="A100-40GB", timeout=GPU_TIMEOUT_SECONDS,
              retries=0, max_containers=1, min_containers=0, scaledown_window=2,
              volumes={"/cache": cache}, secrets=[modal.Secret.from_name("huggingface")])
def run_native(video_bytes: bytes, expected_clip: dict, source_start_frame: int = 0) -> dict:
    import torch
    from huggingface_hub import hf_hub_download
    from sam3.model_builder import build_sam3_multiplex_video_predictor

    if isinstance(source_start_frame, bool) or not isinstance(source_start_frame, int) or source_start_frame < 0:
        raise ValueError("source_start_frame must be a nonnegative integer")
    if not 0 < len(video_bytes) <= MAX_BYTES:
        raise ValueError("Video payload exceeds the 64 MiB limit")
    started = time.perf_counter()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "clip.mp4"
        path.write_bytes(video_bytes)
        clip = inspect_clip(path)
        if clip != expected_clip:
            raise ValueError("Remote clip metadata differs from the locally validated input")
        download_started = time.perf_counter()
        checkpoint_path = hf_hub_download(MODEL_ID, CHECKPOINT, revision=MODEL_REVISION)
        cache.commit()
        download_seconds = time.perf_counter() - download_started
        load_started = time.perf_counter()
        predictor = build_sam3_multiplex_video_predictor(
            checkpoint_path=checkpoint_path, max_num_objects=MAX_OBJECTS, multiplex_count=16,
            use_fa3=False, compile=False, warm_up=False,
        )
        torch.cuda.synchronize()
        load_seconds = time.perf_counter() - load_started
        inference_started = time.perf_counter()
        session_id = predictor.handle_request({
            "type": "start_session", "resource_path": str(path),
            "offload_video_to_cpu": True, "offload_state_to_cpu": True,
        })["session_id"]
        frames = []
        encode_seconds = 0.0
        try:
            predictor.handle_request({"type": "add_prompt", "session_id": session_id,
                                      "frame_index": 0, "text": "person"})
            for response in predictor.handle_stream_request({
                "type": "propagate_in_video", "session_id": session_id,
                "propagation_direction": "forward", "start_frame_index": 0,
                "max_frame_num_to_track": clip["frame_count"],
            }):
                encode_started = time.perf_counter()
                frames.append(encode_frame(response, clip, source_start_frame))
                encode_seconds += time.perf_counter() - encode_started
            torch.cuda.synchronize()
            inference_seconds = time.perf_counter() - inference_started
            validate_frames(frames, clip)
        finally:
            predictor.handle_request({"type": "close_session", "session_id": session_id})
        return {"schema_version": "sam3-native-video-observations-v1", "status": "complete",
                "method": {"name": "SAM 3.1 Object Multiplex", "source_commit": SOURCE_COMMIT,
                           "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
                           "checkpoint": CHECKPOINT, "prompt": "person", "compile": False,
                           "warm_up": False, "use_fa3": False, "max_num_objects": MAX_OBJECTS,
                           "gpu": torch.cuda.get_device_name(), "torch_version": torch.__version__},
                "limits": {"max_frames": MAX_FRAMES, "max_seconds": MAX_SECONDS,
                           "timeout_seconds": GPU_TIMEOUT_SECONDS, "retries": 0},
                "input": clip | {"source_start_frame": source_start_frame},
                "identity_scope": "video_session_only", "session_id": session_id,
                "coordinates": "2d_input_pixels_masks_and_normalized_xywh_boxes",
                "global_map_alignment": "not_estimated", "frames": frames,
                "timing_seconds": {"weight_access_and_download": download_seconds,
                                   "model_load": load_seconds,
                                   "video_inference_including_encoding": inference_seconds,
                                   "mask_encoding_and_validation": encode_seconds,
                                   "remote_total": time.perf_counter() - started}}


@app.local_entrypoint()
def main(video_path: str, output_path: str, source_start_frame: int = 0):
    if source_start_frame < 0:
        raise ValueError("source_start_frame must be nonnegative")
    output = Path(output_path)
    if output.exists() or not output.parent.is_dir():
        raise ValueError("Output must be a new file in an existing directory")
    clip_path = Path(video_path)
    clip = inspect_clip(clip_path)
    started = time.perf_counter()
    # ponytail: one already-trimmed clip and one class; no resampling or hidden tracker.
    result = run_native.remote(clip_path.read_bytes(), clip, source_start_frame)
    result["timing_seconds"]["client_call_wall"] = time.perf_counter() - started
    result["input"]["filename"] = clip_path.name
    with output.open("x") as stream:
        json.dump(result, stream, allow_nan=False)
    print(json.dumps({"output": str(output.resolve()), "frames": len(result["frames"]),
                      "timing_seconds": result["timing_seconds"]}, indent=2))
