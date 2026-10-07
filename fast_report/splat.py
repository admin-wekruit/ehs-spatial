"""Gaussian splats for the fast report (FAST-BUILD-SPEC section 12 B): E5b's one-GPU recipe (splat_train's loss, pose knots, MCMC,
elongation cap 4, with the sync-free pose and separable SSIM: 36.5 -> 11.5 ms a step on an A100) in its own process on GPU1,
seeded from the shot's DA3/TSDF points. A preview after `budget_s` of training, then, if asked, the same process goes on from
it and hands out `full` snapshots. Generated display layers: never used for measurement.

The shot's DA3 keyframes are the cameras. train="all" (the default) trains on every frame of the shot, poses in between
interpolated (rotation slerp, centre linear) and corrected by the pose knots, a frame's person mask the union of its two
keyframes' masks; train="keyframes" trains on the keyframes only (exact poses and masks). ME340, A100, 120 s, E5b's 84
held-out frames: all 26.94 dB, keyframes 25.88 dB. Either way the presenter (dilated DILATE px) and the burnt-in caption box
stay out of the loss. The preview needs PREVIEW_S = 150 s to stay above 27.0 dB (E5b with DROID cameras: 27.48 at 120 s).

Venv /opt/splat: splat_train's pins (Python 3.10, torch 2.4.1 cu124, gsplat 1.5.3 prebuilt), plus modal because splat_train
and fast_splat import it at module level (nothing here calls it).

  python -m fast_report.splat --self-check
"""
import subprocess
import sys
import threading
import time

import numpy as np

from fast_report.sam3d import ROOT, caption_box, frames_path, host, log_tail, recv, send, serve, worker_env

SPLAT_PY = "/opt/splat/bin/python"
PREVIEW_S = 150  # ME340, A100, DA3 cameras, all frames: 120 s 26.94 dB; 140 s 26.94-27.46 (4 runs); 150 s 27.37 / 27.43 (2 runs, splat alone)
PREVIEW_CAP, FULL_CAP = 500_000, 2_500_000  # E5b: 500k is best at 120-180 s, 2.5M for the long run (31.0 dB at 1200 s on an H100)
SEED_VOXEL_M, SEED_SIZE_M = .02 * 2.8591949, .03  # E5b's pick: DA3 points one per .02 native voxel (ME340: 2.859 m/native), 3 cm TSDF samples
SNAPSHOTS = (300, 600, 1200, 1800)  # full snapshots at these total training seconds
DILATE, CAPTION_WINDOW = 15, 3  # splat_train: the presenter's mask grows 15 px; a caption box holds +-3 frames (a line changing)
GSPLAT = "1.5.3"


def poses(keys, c2w, n):
    """Camera-to-world of every frame from the keyframes': rotation slerped and centre linear in between, the nearest keyframe's
    outside them."""
    from scipy.spatial.transform import Rotation, Slerp
    keys, c2w = np.asarray(keys), np.asarray(c2w, np.float64)
    f = np.clip(np.arange(n), keys[0], keys[-1])
    out = np.repeat(np.eye(4)[None], n, 0)
    out[:, :3, :3] = Slerp(keys, Rotation.from_matrix(c2w[:, :3, :3]))(f).as_matrix()
    for d in range(3):
        out[:, d, 3] = np.interp(f, keys, c2w[:, d, 3])
    out[keys] = c2w  # the keyframes' own poses exactly
    return out


def person_of(keys, person, i):
    """A frame's person mask on the raster: its own at a keyframe, else the union of the keyframes either side."""
    j = int(np.searchsorted(keys, i))
    if j < len(keys) and keys[j] == i:
        return person[j]
    return person[max(j - 1, 0)] | person[min(j, len(keys) - 1)]


def full_k(k_raster, raster_wh, full_wh):
    """The raster's K on the whole frame (pixel centres as fast_report.sam3d's grid)."""
    (w, h), (W, H) = raster_wh, full_wh
    sx, sy = W / w, H / h
    return np.array([[sx, 0, (sx - 1) / 2], [0, sy, (sy - 1) / 2], [0, 0, 1.]]) @ np.asarray(k_raster, np.float64)


def excluded(frame_bgr, person_raster, box):
    """Pixels out of the loss: the person (raster mask carried to the frame, dilated) and the caption box."""
    import cv2
    H, W = frame_bgr.shape[:2]
    out = cv2.resize(person_raster.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
    out = cv2.dilate(out, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * DILATE + 1, 2 * DILATE + 1)))
    if box is not None:
        out[max(box[1], 0):box[3] + 1, max(box[0], 0):box[2] + 1] = 1
    return out > 0


def union_box(boxes):
    boxes = [b for b in boxes if b is not None]
    return None if not boxes else [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]


# ---------------------------------------------------------------- the training process (/opt/splat)
def splat_worker():
    sys.path[:0] = [str(ROOT / "modal_apps"), str(ROOT / "modal_apps" / "m3_e5b"), str(ROOT / "scripts")]
    state = {}

    def boot():
        t = time.time()
        import torch
        import gsplat
        import fast_splat as fs
        import splat_train as st
        state.update(torch=torch, fs=fs, st=st)
        imported = time.time() - t
        t = time.time()
        warm(torch, fs, st)  # CUDA context, cuBLAS, the rasteriser's kernels at the real frame size
        return {"ready": True, "import_s": round(imported, 1), "warm_s": round(time.time() - t, 1), "gsplat": gsplat.__version__,
                "torch": str(torch.__version__), "gpu": torch.cuda.get_device_name(0)}

    serve(lambda message, emit: train(message, emit, **state), boot)


def warm(torch, fs, st, n=4):
    st.use_frame({"W": 1280, "H": 720, "SIDE": 0, "TOP": 0, "ZOOM": 1.})
    rng = np.random.default_rng(0)
    c2w = np.repeat(np.eye(4)[None], n, 0)
    c2w[:, 0, 3] = np.linspace(0, .2, n)
    data = {"frames": torch.randint(0, 255, (n, 720, 1280, 3), dtype=torch.uint8, device="cuda"), "excluded": torch.zeros((n, 720, 1280), dtype=torch.bool, device="cuda"),
            "c2w": torch.tensor(c2w, dtype=torch.float32, device="cuda"), "K": torch.tensor([[900., 0, 640], [0, 900, 360], [0, 0, 1]], device="cuda"),
            "train": list(range(n)), "held": [], "xyz": (rng.normal(0, 1, (5000, 3)) + [0, 0, 4]).astype(np.float32),
            "rgb": rng.random((5000, 3)).astype(np.float32), "scale": np.full(5000, .02, np.float32), "depth": None}
    cfg = {"gpus": 1, "cap": 10000, "pose": True, "max_elongation": 4., "fast": True, "schedule": [(1, 1.)], "seconds": None, "steps": 30, "snapshots": ()}
    fs.check_equivalence(data["c2w"][0])
    fs.fit(cfg, data)
    del data
    torch.cuda.empty_cache()


def train(message, emit, torch, fs, st):
    import cv2
    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    frames = np.load(message["frames"], mmap_mode="r")
    n, H, W = frames.shape[:3]
    st.use_frame({"W": W, "H": H, "SIDE": 0, "TOP": 0, "ZOOM": 1.})
    shot, held = message["shot"], sorted(message.get("held") or [])
    keys, (a, b) = [int(k) for k in shot["keys"]], shot["frames"]
    person = np.asarray(shot["person"], bool)
    c2w = poses(keys, shot["c2w"], n)
    use = [k for k in keys if k not in set(held)] if message["train"] == "keyframes" else [i for i in range(a, b + 1) if i not in set(held)]
    given = message.get("held_excluded") is not None  # else the held-out frames are scored under the training frames' own rule
    masked = use + ([] if given else held)
    boxes = {i: caption_box(frames[i]) for i in range(max(a, min(masked) - CAPTION_WINDOW), min(b, max(masked) + CAPTION_WINDOW) + 1)}
    box_of = lambda i: union_box([boxes.get(j) for j in range(i - CAPTION_WINDOW, i + CAPTION_WINDOW + 1)])
    span = sorted(set(use) | set(held))
    gpu_frames = torch.zeros((n, H, W, 3), dtype=torch.uint8, device="cuda")
    for s in range(span[0], span[-1] + 1, 64):  # the whole span in blocks: one copy each, colour order flipped on the GPU
        block = torch.from_numpy(np.ascontiguousarray(frames[s:min(s + 64, span[-1] + 1)])).cuda()
        gpu_frames[s:s + len(block)] = block.flip(-1)
    out_mask = torch.zeros((n, H, W), dtype=torch.bool, device="cuda")
    ellipse = torch.from_numpy(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * DILATE + 1,) * 2)).half().cuda()[None, None]
    for s in range(0, len(masked), 32):  # excluded() on the GPU: nearest upsampling, then the ellipse dilation as a convolution
        part = masked[s:s + 32]
        raster = torch.from_numpy(np.stack([person_of(keys, person, i) for i in part])).cuda()
        full = torch.nn.functional.interpolate(raster[:, None].half(), size=(H, W), mode="nearest")
        out_mask[part] = (torch.nn.functional.conv2d(full, ellipse, padding=DILATE) > 0)[:, 0]
    for i in masked:
        box = box_of(i)
        if box is not None:
            out_mask[i, max(box[1], 0):box[3] + 1, max(box[0], 0):box[2] + 1] = True
    check = [int((out_mask[i].cpu().numpy() != excluded(frames[i], person_of(keys, person, i), box_of(i))).sum()) for i in use[::max(1, len(use) // 4)]]
    if held and given:  # scoring only (D's eval_holdout): the frames' exclusion as the reference scored them
        scored = np.unpackbits(message["held_excluded"], axis=1)[:, :H * W].reshape(len(held), H, W).astype(bool)
        out_mask[held] = torch.from_numpy(scored).cuda()
    rgb = np.asarray(message["seeds"]["rgb"], np.float32)
    rgb = rgb / 255 if rgb.max() > 1 else rgb
    xyz, rgb, scale = st.voxel_seeds(np.asarray(message["seeds"]["xyz"], np.float32), rgb, np.full(len(rgb), SEED_SIZE_M, np.float32), SEED_VOXEL_M, .5)
    cuda = lambda x: torch.tensor(np.asarray(x), dtype=torch.float32, device="cuda")
    k = full_k(np.median(np.asarray(shot["K"], np.float64).reshape(-1, 3, 3), 0), person.shape[1:][::-1], (W, H))
    data = {"frames": gpu_frames, "excluded": out_mask, "c2w": cuda(c2w), "train": use, "held": held, "xyz": xyz, "rgb": rgb, "scale": scale,
            "K": cuda(k + [[0, 0, .5], [0, 0, .5], [0, 0, 0]]), "depth": None}  # gsplat's pixel centres, as splat_train.load
    fs.check_equivalence(data["c2w"][use[0]])
    setup_s = time.time() - started
    if message.get("hold"):  # set up while the GPU is still busy (A: SAM 3's queue); train only once released
        emit({"kind": "ready", "setup_s": round(setup_s, 2), "end_unix": time.time()})
        released = recv(sys.stdin.buffer)
        assert released.get("go"), released
    budget, background = float(message["budget_s"]), float(message.get("background_s") or 0)
    cfg = {"gpus": 1, "cap": max(PREVIEW_CAP, len(xyz)), "pose": True, "max_elongation": 4., "fast": True, "schedule": [(1, 1.)],
           "seconds": budget, "steps": None, "snapshots": ()}
    base = {"train": message["train"], "trained_frames": len(use), "seeds": len(xyz), "setup_s": round(setup_s, 2),
            "exclusion_pixels_off_cpu_rule": check}  # the GPU exclusion against excluded() on a few frames (expected all 0)
    params, knots, stats, _ = fs.fit(cfg, data)
    scored = []

    def hand_out(kind, params_now, knots_now, seconds, steps, **extra):
        t = time.time()
        records, _ = st.export(params_now, {"cap": max(FULL_CAP if kind == "full" else cfg["cap"], len(params_now["means"]))}, data)
        blob = records.tobytes()
        emit({"kind": kind, "seconds": round(seconds, 1), "steps": steps, "splat32": blob, "count": len(records), "export_s": round(time.time() - t, 2),
              "end_unix": time.time(), "max_reserved_gb": round(torch.cuda.max_memory_reserved() / 1e9, 2), **base, **extra})
        if held:
            scored.append((kind, seconds, blob, knots_now))

    hand_out("preview", params, knots["pose"], stats["train_seconds"], stats["steps_done"], gaussians=stats["gaussians"])
    if background > 0:
        steps0, used = stats["steps_done"], knots["used"]
        data["init"] = {"params": params, "knots": knots["pose"]}
        on_snapshot = lambda s: hand_out("full", s["params"], s["knots"], budget + s["elapsed_s"], steps0 + s["step"], gaussians=len(s["params"]["means"]))
        more = {**cfg, "cap": FULL_CAP, "seconds": background, "on_snapshot": on_snapshot,
                "snapshots": [t - budget for t in SNAPSHOTS if budget < t < budget + background]}
        params, knots, stats, _ = fs.fit(more, data)
        hand_out("full", params, knots["pose"], budget + stats["train_seconds"], steps0 + stats["steps_done"], gaussians=stats["gaussians"], final=True)
        knots["used"] = used
    if held:  # off the clock, after training: each handed-out file scored on the held-out frames at its own corrected cameras
        import lpips
        net = lpips.LPIPS(net="alex", spatial=True, verbose=False).cuda().eval()
        for kind, seconds, blob, pose in scored:
            t = time.time()
            summary = st.evaluate(st.as_model(blob), {"pose": pose, "used": knots["used"]}, {"pose": True, "exposure": False}, data, net)[0]["summary"]
            emit({"kind": "score", "of": kind, "seconds": seconds, "held_out": summary, "score_s": round(time.time() - t, 1)})
    data = gpu_frames = out_mask = params = None  # the tensors go (a closure still names data)
    torch.cuda.empty_cache()  # the frames and Gaussians' cache back to GPU 1 before the next report's core
    return {"done": True, "max_reserved_gb": round(torch.cuda.max_memory_reserved() / 1e9, 2), "wall_s": round(time.time() - started, 1)}


# ---------------------------------------------------------------- main side
class Worker:
    """The splat process on one GPU, started at boot (imports and a few real steps happen there)."""

    def __init__(self, gpu=1, python=SPLAT_PY, torch_home=None):
        env = worker_env(python, CUDA_VISIBLE_DEVICES=gpu, **({"TORCH_HOME": torch_home} if torch_home else {}))
        self.gpu, self.log, self.lock, self.boot = gpu, f"/tmp/fb-splat-{gpu}.log", threading.Lock(), None
        self.proc = subprocess.Popen([python, "-c", "from fast_report.splat import splat_worker; splat_worker()"], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=open(self.log, "wb"), env=env, cwd=str(ROOT))

    def _recv(self):
        try:
            reply = recv(self.proc.stdout)
        except EOFError as error:
            raise RuntimeError(f"splat process died: {error}\n{log_tail(self.log)}") from None
        if isinstance(reply, dict) and "error" in reply:
            raise RuntimeError("splat process: " + reply["error"])
        return reply

    def ready(self):
        if self.boot is None:
            self.boot = self._recv()
        return self.boot

    def start(self, frames_host, shot, seeds, budget_s=PREVIEW_S, background_s=0, train="all", held=(), held_excluded=None, hold=False):
        """{"kind": "preview" | "full" | "score", "seconds", "steps", "splat32", "count", ...} as they are made. shot: A's Shot (keys,
        frames (a, b), c2w_m, K, person); seeds: {"xyz" (m, 3) metres, "rgb" (m, 3)}; held/held_excluded: frames kept out of
        training and scored after it, with the exclusion masks to score them by ((m, H, W) bool; None: the training frames' own rule,
        person and caption). With hold, the process sets up
        (frames and masks onto the GPU, 8-16 s), yields {"kind": "ready"} and trains only after release()."""
        self.ready()
        message = {"frames": frames_path(frames_host), "budget_s": budget_s, "background_s": background_s, "train": train, "held": list(held), "hold": hold,
                   "shot": {"keys": [int(k) for k in shot["keys"]], "frames": [int(x) for x in shot["frames"]], "c2w": host(shot["c2w_m"]).astype(np.float64),
                            "K": host(shot["K"]).astype(np.float64), "person": host(shot["person"]).astype(bool)},
                   "seeds": {"xyz": host(seeds["xyz"]).astype(np.float32), "rgb": host(seeds["rgb"])}}
        if held and held_excluded is not None:
            message["held_excluded"] = np.packbits(np.asarray(held_excluded, bool).reshape(len(held), -1), axis=1)
        with self.lock:
            send(self.proc.stdin, message)
            while True:
                reply = self._recv()
                if reply.get("done"):
                    self.last = reply
                    return
                yield reply

    def release(self):
        """Start training after start(hold=True) has yielded "ready" (any thread; start() only reads the process's stdout)."""
        send(self.proc.stdin, {"go": True})

    def close(self):
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=20)
        except Exception:
            self.proc.kill()


def with_envs(image):
    """A's image plus /opt/splat (splat_train's pins; uv brings Python 3.10, which gsplat 1.5.3's wheels need)."""
    wheel = f"https://github.com/nerfstudio-project/gsplat/releases/download/v{GSPLAT}/gsplat-{GSPLAT}%2Bpt24cu124-cp310-cp310-linux_x86_64.whl"
    return image.run_commands("python -m pip install uv==0.8.22", "uv venv --python 3.10 /opt/splat",
                              "uv pip install --python /opt/splat/bin/python torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124",
                              f"uv pip install --python /opt/splat/bin/python numpy==1.26.4 opencv-python-headless==4.10.0.84 scipy==1.13.1 lpips==0.1.4 "
                              f"packaging==24.2 setuptools==75.8.0 modal {wheel}")  # gsplat loads its kernels through torch.utils.cpp_extension, which imports setuptools


def self_check():
    from scipy.spatial.transform import Rotation
    keys = [2, 8, 14]
    c2w = np.repeat(np.eye(4)[None], 3, 0)
    c2w[:, :3, :3] = Rotation.from_euler("y", [[0], [30], [60]], degrees=True).as_matrix()
    c2w[:, :3, 3] = [[0, 0, 0], [1, 0, 0], [1, 1, 0]]
    p = poses(keys, c2w, 20)
    assert np.allclose(p[keys], c2w) and np.allclose(p[0], c2w[0]) and np.allclose(p[19], c2w[2]), "keyframes kept, ends held"
    assert np.allclose(p[5, :3, 3], [.5, 0, 0]) and abs(Rotation.from_matrix(p[5, :3, :3]).as_euler("xyz", degrees=True)[1] - 15) < 1e-6
    person = np.zeros((3, 4, 5), bool)
    person[0, 0, 0], person[1, 1, 1] = True, True
    assert person_of(keys, person, 8)[1, 1] and not person_of(keys, person, 8)[0, 0] and person_of(keys, person, 5)[[0, 1], [0, 1]].all()
    assert person_of(keys, person, 0)[0, 0] and person_of(keys, person, 19).sum() == 0
    k = np.array([[226.6, 0, 252], [0, 222.9, 140], [0, 0, 1.]])
    kf = full_k(k, (504, 280), (1280, 720))
    point = np.array([.3, .1, 2.])
    assert np.allclose((kf @ point / 2)[:2], [(1280 / 504) * (k @ point / 2)[0] + (1280 / 504 - 1) / 2, (720 / 280) * (k @ point / 2)[1] + (720 / 280 - 1) / 2])
    frame = np.zeros((720, 1280, 3), np.uint8)
    person_raster = np.zeros((280, 504), bool)
    person_raster[100:120, 200:220] = True
    out = excluded(frame, person_raster, [300, 650, 900, 690])
    assert out[300, 540] and out[670, 600] and not out[0, 0] and out[:, :].sum() > (20 * 1280 / 504) * (20 * 720 / 280)
    assert union_box([None, [1, 2, 3, 4], [0, 5, 2, 9]]) == [0, 2, 3, 9] and union_box([None]) is None
    print("fast_report.splat self-check passed: pose interpolation, person masks between keyframes, K on the frame, exclusion")


if __name__ == "__main__":
    assert sys.argv[1:] == ["--self-check"], __doc__
    self_check()
