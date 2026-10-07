"""mvp2 accuracy lane: DROID-SLAM (the pinned build already on the panoptes-droid-room-001 volume, droid_room.py) on a run's
5 fps keyframes, with DA3's intrinsics from that run's cameras layer (RGB only: no calibration is given). Frames at
384x216 (16:9, divisible by 8; ~ the 320x240 of droid_room's runs), no depth upsampling (only poses are used), official
test_tum settings otherwise. One A100-80GB; each job is timed from its JPEG bytes in the container to its poses.

    modal run modal_apps/droid_keyframes.py --jobs JOBS.json --out OUT.json
    # JOBS.json: [{"name": str, "keys": [video frame], "K720": [fx, fy, cx, cy] at 1280x720, "jpegs": [path, ...]}]
"""
import json
import sys
import time
from pathlib import Path

import modal

app = modal.App("panoptes-mvp2-droid-keyframes")
image = modal.Image.from_registry("pytorch/pytorch:2.7.0-cuda12.6-cudnn9-devel").entrypoint([])  # the build's own image (droid_room)
volume = modal.Volume.from_name("panoptes-droid-room-001")
WH = (384, 216)


@app.function(image=image, gpu="A100-80GB", cpu=8, memory=32768, timeout=1200, retries=0, max_containers=1, volumes={"/artifact": volume})
def track(jobs):
    import argparse
    import os
    sys.path[:0] = ["/artifact/site", "/artifact/DROID-SLAM/droid_slam"]
    os.environ["MPLBACKEND"] = "Agg"
    import cv2
    import numpy as np
    import torch
    torch.multiprocessing.set_start_method("spawn", force=True)
    from droid import Droid
    out = []
    for job in jobs:
        t0 = time.time()
        sx, sy = WH[0] / 1280, WH[1] / 720
        fx, fy, cx, cy = job["K720"]
        K = torch.tensor([fx * sx, fy * sy, (cx + .5) * sx - .5, (cy + .5) * sy - .5], dtype=torch.float32)
        ims = [torch.from_numpy(cv2.resize(cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR), WH, interpolation=cv2.INTER_AREA))
               .permute(2, 0, 1)[None] for b in job["jpegs"]]
        args = argparse.Namespace(weights="/artifact/droid.pth", buffer=len(ims) + 38, image_size=[WH[1], WH[0]], disable_vis=True, beta=.3,
                                  filter_thresh=1.5, warmup=12, keyframe_thresh=2.0, frontend_thresh=12.0, frontend_window=25, frontend_radius=2,
                                  frontend_nms=1, backend_thresh=20.0, backend_radius=2, backend_nms=3, motion_damping=.5, upsample=False,
                                  stereo=False, asynchronous=False, frontend_device="cuda:0", backend_device="cuda:0")
        t_load = time.time()
        droid = Droid(args)
        for i, im in enumerate(ims):
            droid.track(i, im, intrinsics=K)
        t_track = time.time()
        traj = droid.terminate(((i, im, K) for i, im in enumerate(ims)))  # official filler: camera-to-world per input frame
        import lietorch
        c2w = lietorch.SE3(torch.as_tensor(traj, device="cuda")).matrix().cpu().numpy()
        torch.cuda.synchronize()
        out.append({"name": job["name"], "keys": job["keys"], "c2w": c2w.tolist(), "droid_keyframes": int(droid.video.counter.value),
                    "seconds": round(time.time() - t0, 2), "decode_s": round(t_load - t0, 2), "track_s": round(t_track - t_load, 2),
                    "terminate_s": round(time.time() - t_track, 2), "K_droid": K.tolist(), "wh": list(WH),
                    "peak_gib": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2), "gpu": torch.cuda.get_device_name()})
        print(json.dumps({k: v for k, v in out[-1].items() if k != "c2w"}), flush=True)
        del droid
        torch.cuda.empty_cache()
    return out


@app.local_entrypoint()
def main(jobs: str, out: str):
    spec = json.loads(Path(jobs).read_text())
    for j in spec:
        j["jpegs"] = [Path(p).read_bytes() for p in j["jpegs"]]
    t = time.time()
    res = track.remote(spec)
    Path(out).write_text(json.dumps({"call_wall_s_two_clocks": round(time.time() - t, 1), "jobs": res}))
    print("done", len(res), round(time.time() - t, 1), "s")
