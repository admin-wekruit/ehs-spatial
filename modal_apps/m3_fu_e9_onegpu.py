"""M3 follow-up E9: the whole fast-path core (FAST-PATH-PLAN.md section 2) in ONE resident Modal container, on one A100-80GB
and then on two. Research probe, not a stage.

One class loads everything once at container start: DA3-GIANT-1.1 any-view, SAM 3 (image, bf16), Open3D CUDA TSDF + the
E7 GPU mask lift, and Qwen3-VL-8B behind a vLLM server (its own venv in the same image: vLLM pins its own torch). Per call
the ME340 MP4 bytes go in and the core layers come out:
  decode (CPU) || shot cuts (detect_shot_cuts rules, chunks measured in worker processes while decoding)
  -> 5 fps keyframes per shot (sharpest of each 6-frame block; shots under 1 s get none) -> keyframes uploaded once
  -> DA3 any-view per shot (GPU-side resize/normalise, raw forward: depth, poses, K never leave the GPU)
  -> SAM 3 {person, floor} on every keyframe; 40-word vocabulary on every 3rd (vision features reused); all SAM 3 work is one
     task queue ({person, floor} chunks first), so a second GPU can take any of it
  -> floor plane + 1.6 m camera height = metric scale -> TSDF (people cut out) + points -> people 3D + tracks
  -> vocabulary masks lifted, merged by voxel overlap, named by score-weighted word vote -> events (2 windows, vLLM)

Configurations (one container each, all run at once; "-mps" = NVIDIA MPS daemon started before any CUDA context, so the
vLLM process's kernels and this process's run side by side instead of time-slicing; E4 found MPS works on Modal):
  1gpu[-mps]  1 x A100-80GB  serial    one stage after another, events last
                             overlap   geometry thread || SAM 3 thread on its own CUDA stream || events from end of decode
  2gpu[-mps]  2 x A100-80GB  split     the plan: GPU 0 DA3 + TSDF + lift, GPU 1 SAM 3 + events (vLLM)
                             balanced  split, and GPU 0 takes SAM 3 tasks from the shared queue whenever it would wait

  modal run modal_apps/m3_fu_e9_onegpu.py --out RUNS/m3-fu-e9-onegpu-NNN [--configs 1gpu,1gpu-mps,2gpu,2gpu-mps] [--smoke]
  python modal_apps/m3_fu_e9_onegpu.py --self-check           # chunked cut measure == sequential; keyframes; track linking
  python modal_apps/m3_fu_e9_onegpu.py --evaluate RUN_DIR     # poses vs DROID, objects vs the object map, events vs 197
"""
import base64
import copy
import itertools
import json
import os
import queue
import subprocess
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from pathlib import Path

import modal
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "scripts")]
import detect_shot_cuts as dsc  # noqa: E402  cut rules, unchanged
import m3_exp_geometry as geo  # noqa: E402  E1/E7: Open3D CUDA TSDF (fuse), edge filter, Sim3 alignment
import sam3_app  # noqa: E402  SAM 3 revision pin
import video_events  # noqa: E402  events prompt, windows, JSON parsing

PHASE2 = Path("/Users/adam/Desktop/panoptes-public/research-notes/phase2")
CLIP = PHASE2 / "data/clips/me340-165"
DA3_CODE = "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4"  # m3_e5/cold_start.py
DA3_MODEL, DA3_REV = "depth-anything/DA3-GIANT-1.1", "72ee9f89ce4e50d704e9d55ee9c646ec8dc25a19"  # mono_room.py pins
QWEN, VLLM_SHARE, VLLM_PORT = "Qwen/Qwen3-VL-8B-Instruct", .35, 8000
# E2's 40-word list (runs/m3-exp-e2-e6-segment-probe-4-vocab40: 20 EHS + 20 generic); no E2b list existed when this ran
EHS = ["machine", "cabinet", "shelf", "workbench", "cart", "ladder", "fire extinguisher", "bin", "box", "pallet", "forklift",
       "vise", "drill press", "lathe", "control panel", "hose", "cable", "door", "sign", "chair"]
GENERIC = ["bar", "board", "bucket", "clamp", "container", "guard", "handle", "lamp", "light", "mirror", "monitor", "panel",
           "part", "pipe", "plate", "rack", "screen", "table", "tool", "window"]
TEXTS = ["person", "floor"] + EHS + GENERIC  # prompt ids: 0 person, 1 floor, 2.. vocabulary
BLOCK, OBJECT_EVERY, MIN_SHOT, CHUNK, PROCS = 6, 3, 30, 64, 12
SCORE, TOP = .4, 12  # production rule (E2)
PERSON_FRAMES = 8  # frames per SAM 3 forward in the {person, floor} pass (x2 prompts = 16 pairs)
DA3_HW, SAM_SIDE = (280, 504), 1008
LIFT_VOXEL, MIN_PIXELS, MOVING, MATCH_MIN, MATCH_MAX, CONFIRMED = .05, 16, .5, .5, .2, 2  # E7's rule; 2 frames at ~1.7 fps
CAMERA_HEIGHT_M = 1.6  # same assumption as the reference's metric scale
TRACK_GAP_FRAMES, TRACK_STEP_M = 15, 1.0
PRICE = {"A100-80GB": .000694, "cpu_core": .0000131, "gib": .00000222}  # Modal list $/s
CPU, MEMORY_GIB = 16, 64

app = modal.App("panoptes-m3-fu-e9-onegpu")
VOLUMES = {"/v/da3": modal.Volume.from_name("moge3-hf-cache"), "/v/sam3": modal.Volume.from_name("sam3-hf-cache"),
           "/v/vlm": modal.Volume.from_name("panoptes-vlm-cache")}
image = (modal.Image.debian_slim(python_version="3.11").apt_install("git", "libgl1", "libglib2.0-0", "libgomp1")
         # m3_e5/cold_start.py's /opt/vis recipe (DA3 + SAM 3 proven together) plus Open3D 0.19 CUDA (E7)
         .pip_install("torch", "torchvision", "xformers", "transformers>=4.57", "accelerate", "addict", "pillow", "scipy",
                      "open3d==0.19.0", f"git+https://github.com/ByteDance-Seed/Depth-Anything-3.git@{DA3_CODE}")
         .run_commands("python -m venv /opt/vllm && PIP_EXTRA_INDEX_URL= /opt/vllm/bin/pip install -q vllm==0.11.0 transformers==4.57.1 pillow")
         .env({"HF_HUB_OFFLINE": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
         .add_local_python_source("detect_shot_cuts", "m3_exp_geometry", "sam3_app", "video_events"))


# ---------- CPU helpers (the self-check runs these locally) ----------

def raster_gray(bgr):
    """16:9 source frame -> the clip's 4:3 centre crop at 640x480, grey (what detect_shot_cuts reads)."""
    import cv2
    h, w = bgr.shape[:2]
    x0 = (w - h * 4 // 3) // 2
    return cv2.cvtColor(cv2.resize(bgr[:, x0:w - x0], (640, 480), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)


def gray_sharp(bgr):
    import cv2
    gray = raster_gray(bgr)
    return gray, float(cv2.Laplacian(gray, cv2.CV_32F).var())


def measure_chunk(gray, a, b, a0, b1):
    """detect_shot_cuts.measure over the grey frames [a0, b1) (a0 = a - 2), keeping the frames, pairs and spans that start in
    [a, b). Two frames of lead give pair a its left neighbour (jump test); b1 = b + SPAN + 1 gives the last kept pair its
    right neighbour and the last kept span its far frame. The last chunk ends at the clip's end, where sequential ends too.
    Runs in a worker process: dsc.measure holds the GIL between OpenCV calls, so threads do not scale."""
    import cv2
    cv2.setNumThreads(1)
    m = dsc.measure(gray)
    pairs, spans = slice(a - a0, min(b, b1 - 1) - a0), slice(a - a0, max(a, min(b, b1 - dsc.SPAN)) - a0)
    return {"keypoints": m["keypoints"][a - a0:b - a0], "inliers": m["inliers"][pairs], "homographies": m["homographies"][pairs],
            "jump": m["jump"][pairs], "spans": m["spans"][spans], "shape": m["shape"]}


def warm_worker(_):
    import cv2
    cv2.setNumThreads(1)
    time.sleep(.2)  # so every worker gets one
    return os.getpid()


def stitch(parts):
    cat = lambda k: np.concatenate([p[k] for p in parts])  # noqa: E731
    return {"keypoints": cat("keypoints"), "inliers": cat("inliers"), "jump": cat("jump"), "spans": cat("spans"),
            "homographies": [h for p in parts for h in p["homographies"]], "shape": parts[0]["shape"]}


def cuts_from(m, n):
    s = dsc.signals(m)
    t = dsc.transitions(s)
    return {"frames": n, "cuts": t["cuts"], "fades": t["fades"], "noCoverage": t["noCoverage"],
            "segments": dsc.segments(n, t["cuts"], t["fades"] + t["noCoverage"])}


def link_tracks(dets):
    """Greedy: each detection (frame order) joins the nearest track seen <= TRACK_GAP_FRAMES earlier within TRACK_STEP_M.
    ponytail: nearest-neighbour only, no motion model or re-identification; E3's tracker is the full version."""
    tracks = []
    for d in sorted(dets, key=lambda d: (d["frame"], -d["score"])):
        best, best_dist = None, TRACK_STEP_M
        for t in tracks:
            last = t[-1]
            dist = float(np.linalg.norm(np.subtract(d["xyz"], last["xyz"])))
            if 0 < d["frame"] - last["frame"] <= TRACK_GAP_FRAMES and dist <= best_dist:
                best, best_dist = t, dist
        (best.append(d) if best is not None else tracks.append([d]))
    return tracks


# ---------- GPU helpers ----------

def backproject(depth, K, c2w, f, vy, vx, stride):
    """Pixels (frame f, row vy, col vx of a stride grid over depth's full resolution) -> world points."""
    import torch
    u, v = vx.float() * stride + (stride - 1) / 2, vy.float() * stride + (stride - 1) / 2
    z = depth[f, vy * stride, vx * stride]
    k = K[f]
    cam = torch.stack([(u - k[:, 0, 2]) / k[:, 0, 0] * z, (v - k[:, 1, 2]) / k[:, 1, 1] * z, z], 1)
    return (c2w[f, :3, :3] @ cam[:, :, None])[:, :, 0] + c2w[f, :3, 3]


def floor_plane(depth, K, c2w, floor, stride=4):
    """Least-squares plane through the floor-masked points, trimmed three times; camera height above it (DA3 units)."""
    import torch
    sel = floor[:, ::stride, ::stride] & (depth[:, ::stride, ::stride] > 0)
    f, vy, vx = torch.nonzero(sel, as_tuple=True)
    if len(f) < 2000:
        return None
    pts = backproject(depth, K, c2w, f, vy, vx, stride).double()
    keep = pts
    for _ in range(3):
        c = keep.mean(0)
        n = torch.linalg.eigh((keep - c).T @ (keep - c))[1][:, 0]
        r = ((keep - c) @ n).abs()
        keep = keep[r <= 2.5 * r.median().clamp(min=1e-9)]
    h = (c2w[:, :3, 3].double() - c) @ n
    if h.median() < 0:
        n, h = -n, -h
    return {"normal": n.float(), "point": c.float(), "camera_height_units": float(h.median()), "points": int(len(pts)),
            "inliers": int(len(keep)), "camera_height_spread_rel": float((h - h.median()).abs().median() / h.median().abs())}


def lift(masks, frame_of, word_of, score_of, depth_m, K, c2w_m, dyn, stride=2):
    """E7's lift on SAM 3 masks: pixels -> 5 cm voxels -> cross-frame voxel overlap -> connected components -> objects,
    each named by its masks' score-weighted word vote. masks (M,H,W) bool; frame_of (M,) index into depth_m (N,H,W)."""
    import torch
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    dev, n = depth_m.device, len(depth_m)
    m = masks[:, ::stride, ::stride]
    dy = dyn[:, ::stride, ::stride][frame_of]
    moving = (m & dy).sum((1, 2)) >= MOVING * m.sum((1, 2)).clamp(min=1)
    m = m & (depth_m[:, ::stride, ::stride] > 0)[frame_of] & ~dy
    idx = torch.nonzero(~moving & (m.sum((1, 2)) >= MIN_PIXELS)).squeeze(1)
    stats = {"masks_in": len(masks), "masks_dropped_moving": int(moving.sum()), "masks_lifted": len(idx)}
    if len(idx) < 2:
        return [], stats
    mid, vy, vx = torch.nonzero(m[idx], as_tuple=True)
    fr = frame_of[idx]
    world = backproject(depth_m, K, c2w_m, fr[mid], vy, vx, stride)
    ijk = torch.floor(world / LIFT_VOXEL).long()
    B, OFF = 1 << 21, 1 << 20
    vox, vid = torch.unique(((ijk[:, 0] + OFF) * B + (ijk[:, 1] + OFF)) * B + (ijk[:, 2] + OFF), return_inverse=True)
    pair = torch.unique(mid * len(vox) + vid)
    pm, pv = pair // len(vox), pair % len(vox)
    size = torch.bincount(pm, minlength=len(idx)).float()
    A = torch.sparse_coo_tensor(torch.stack([pm, pv]), torch.ones_like(pm, dtype=torch.float32), (len(idx), len(vox))).coalesce()
    inter = torch.sparse.mm(A, A.t()).coalesce()
    (a, b), c = inter.indices(), inter.values()
    edge = (a < b) & (fr[a] != fr[b]) & (c >= MATCH_MIN * torch.minimum(size[a], size[b])) & (c >= MATCH_MAX * torch.maximum(size[a], size[b]))
    a, b = a[edge].cpu().numpy(), b[edge].cpu().numpy()
    ncomp, label = connected_components(coo_matrix((np.ones(len(a)), (a, b)), shape=(len(idx), len(idx))), directed=False)
    lab = torch.from_numpy(label).to(dev)
    ov = torch.unique(lab[pm] * len(vox) + pv)
    oc, ovid = ov // len(vox), ov % len(vox)
    centre = torch.stack([vox // (B * B) - OFF, (vox // B) % B - OFF, vox % B - OFF], 1).float()[ovid] * LIFT_VOXEL + LIFT_VOXEL / 2
    nvox = torch.bincount(oc, minlength=ncomp).float()
    cent = torch.zeros(ncomp, 3, device=dev).index_add_(0, oc, centre) / nvox.clamp(min=1)[:, None]
    lo = torch.full((ncomp, 3), 1e9, device=dev).scatter_reduce(0, oc[:, None].expand(-1, 3), centre, "amin")
    hi = torch.full((ncomp, 3), -1e9, device=dev).scatter_reduce(0, oc[:, None].expand(-1, 3), centre, "amax")
    nframes = torch.bincount(torch.unique(lab * n + fr) // n, minlength=ncomp)
    nframes, nvox, cent, lo, hi = (x.cpu().numpy() for x in (nframes, nvox, cent, lo, hi))
    votes = np.zeros((ncomp, len(TEXTS)))
    np.add.at(votes, (label, word_of[idx].cpu().numpy()), score_of[idx].float().cpu().numpy())
    objects = []
    for i in np.flatnonzero(nframes >= CONFIRMED):
        top = np.argsort(-votes[i])[:3]
        objects.append({"word": TEXTS[top[0]], "votes": {TEXTS[w]: round(float(votes[i, w]), 3) for w in top if votes[i, w] > 0},
                        "frames": int(nframes[i]), "masks": int((label == i).sum()), "voxels": int(nvox[i]), "centroid_m": cent[i].round(3).tolist(),
                        "box_min_m": (lo[i] - LIFT_VOXEL / 2).round(3).tolist(), "box_max_m": (hi[i] + LIFT_VOXEL / 2).round(3).tolist()})
    stats.update(points=int(len(mid)), voxels=int(len(vox)), edges=int(len(a)), components=int(ncomp), confirmed=len(objects))
    return objects, stats


def union_by_frame(frame, masks, n):
    """OR of the masks per frame -> (n,H,W) bool."""
    import torch
    out = torch.zeros((n, *masks.shape[1:]), dtype=torch.int32, device=masks.device)
    if len(masks):
        out.index_add_(0, frame, masks.int())
    return out > 0


# ---------- vLLM sidecar ----------

def start_vllm(gpu):
    env = {k: v for k, v in os.environ.items() if k != "PYTORCH_CUDA_ALLOC_CONF"}
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), HF_HOME="/v/vlm/huggingface", HF_HUB_OFFLINE="1")
    cmd = ["/opt/vllm/bin/vllm", "serve", QWEN, "--host", "127.0.0.1", "--port", str(VLLM_PORT), "--served-model-name", "qwen",
           "--max-model-len", "16384", "--gpu-memory-utilization", str(VLLM_SHARE), "--max-num-seqs", "4",
           "--limit-mm-per-prompt", json.dumps({"image": 40, "video": 0}), "--enforce-eager", "--seed", "0"]
    return subprocess.Popen(cmd, env=env, stdout=open("/tmp/vllm.log", "w"), stderr=subprocess.STDOUT)


def vllm_log_tail(n=30):
    lines = Path("/tmp/vllm.log").read_text(errors="replace").splitlines()[-n:]
    return "\n".join(l for l in lines if not any(w in l.lower() for w in ("token", "secret", "capabilit")))


def wait_vllm(proc, timeout=900):
    started = time.time()
    while time.time() - started < timeout:
        if proc.poll() is not None:
            raise RuntimeError("vLLM exited:\n" + vllm_log_tail())
        try:
            if urllib.request.urlopen(f"http://127.0.0.1:{VLLM_PORT}/health", timeout=2).status == 200:
                return
        except Exception:
            pass
        time.sleep(.25)
    raise TimeoutError("vLLM not healthy:\n" + vllm_log_tail())


def chat(window, max_tokens=900):
    t0, t1, frames = window
    content = []
    for t, jpg in frames:
        content += [{"type": "text", "text": f"[t = {t:.1f} s]"},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jpg).decode()}}]
    content.append({"type": "text", "text": video_events.PROMPT.format(n=len(frames), t0=t0, t1=t1)})
    body = {"model": "qwen", "messages": [{"role": "user", "content": content}], "temperature": 0, "max_tokens": max_tokens, "seed": 0}
    req = urllib.request.Request(f"http://127.0.0.1:{VLLM_PORT}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    r = json.loads(urllib.request.urlopen(req, timeout=600).read())
    return {"t0": t0, "t1": t1, "text": r["choices"][0]["message"]["content"], "prompt_tokens": r["usage"]["prompt_tokens"],
            "completion_tokens": r["usage"]["completion_tokens"]}


# ---------- run bookkeeping ----------

class Log:
    def __init__(self):
        self.t0, self.rows, self.marks = time.perf_counter(), [], {}

    def now(self):
        return round(time.perf_counter() - self.t0, 3)

    def mark(self, name):
        self.marks[name] = self.now()

    def add(self, name, start):
        end = self.now()
        self.rows.append({"stage": name, "start_s": start, "end_s": end, "s": round(end - start, 3)})

    @contextmanager
    def stage(self, name, dev=None):
        """dev: synchronise that device's current stream at the end, so the row is GPU time, not launch time."""
        start = self.now()
        yield
        if dev is not None:
            import torch
            torch.cuda.current_stream(dev).synchronize()
        self.rows.append({"stage": name, "start_s": start, "end_s": self.now(), "s": round(self.now() - start, 3)})


class Vram(threading.Thread):
    """Whole-device memory in use (all processes: main, Open3D, vLLM), sampled every 50 ms."""

    def __init__(self, devs):
        super().__init__(daemon=True)
        self.devs, self.peak, self.halt = devs, [0.] * len(devs), threading.Event()

    def run(self):
        import torch
        while not self.halt.is_set():
            for i, d in enumerate(self.devs):
                free, total = torch.cuda.mem_get_info(d)
                self.peak[i] = max(self.peak[i], (total - free) / 1e9)
            time.sleep(.05)


# ---------- the resident container ----------

@app.cls(image=image, gpu="A100-80GB", cpu=CPU, memory=MEMORY_GIB * 1024, volumes=VOLUMES, timeout=1500, retries=0,
         max_containers=1, scaledown_window=20)
class Core:
    mps: int = modal.parameter(default=0)  # 1: NVIDIA MPS daemon first, so vLLM's and this process's kernels share SMs
    compile: int = modal.parameter(default=0)  # 1: torch.compile SAM 3's vision encoder, DETR encoder/decoder, mask decoder
    # ponytail: compile=1 crashes at boot in this image (Inductor/Triton: KeyError triton.language.bfloat16, run 003) and Modal
    # restarts the crashing container until the app is stopped; needs a torch/Triton pin before it can be measured

    @modal.enter()
    def boot(self):
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor
        entered, t0 = time.time(), time.perf_counter()
        b = self.boot_record = {"entered_unix": entered, "mps": self.mps, "compile": self.compile}
        lap = lambda k: b.__setitem__(k, round(time.perf_counter() - t0, 2))  # noqa: E731
        listing = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.splitlines()
        self.n_gpu = sum(line.startswith("GPU") for line in listing)
        if self.mps:  # E4: works on Modal; must run before any CUDA context (vLLM's or ours)
            env = {"CUDA_MPS_PIPE_DIRECTORY": "/tmp/mps-pipe", "CUDA_MPS_LOG_DIRECTORY": "/tmp/mps-log"}
            for d in env.values():
                Path(d).mkdir(parents=True, exist_ok=True)
            os.environ.update(env)
            started = subprocess.run(["nvidia-cuda-mps-control", "-d"], capture_output=True, text=True)
            b["mps_start"] = {"code": started.returncode, "out": (started.stdout + started.stderr)[-300:]}
            assert started.returncode == 0, b["mps_start"]
        self.vllm = start_vllm(self.n_gpu - 1)  # first: its imports and load overlap everything below
        lap("vllm_spawned_s")
        self.proc_pool = ProcessPoolExecutor(PROCS, mp_context=multiprocessing.get_context("spawn"))
        self.proc_pool.map(warm_worker, range(PROCS))  # workers import in the background
        import torch
        import cv2  # noqa: F401
        import open3d  # noqa: F401
        import transformers
        from depth_anything_3.api import DepthAnything3
        from transformers import Sam3Model, Sam3Processor
        lap("imports_s")
        da3 = DepthAnything3.from_pretrained(DA3_MODEL, revision=DA3_REV, cache_dir="/v/da3/huggingface/hub").eval()
        lap("da3_cpu_load_s")
        self.proc = Sam3Processor.from_pretrained(sam3_app.MODEL_ID, revision=sam3_app.REVISION, cache_dir="/v/sam3/huggingface/hub")
        sam = Sam3Model.from_pretrained(sam3_app.MODEL_ID, revision=sam3_app.REVISION, cache_dir="/v/sam3/huggingface/hub",
                                        torch_dtype=torch.bfloat16).eval()
        lap("sam3_cpu_load_s")
        self.dev_geo, self.dev_seg = torch.device("cuda:0"), torch.device(f"cuda:{self.n_gpu - 1}")
        self.devs = sorted({self.dev_geo, self.dev_seg}, key=str)
        if self.n_gpu > 1:  # GPU 0 carries no vLLM: no need to wait for its memory profiling
            self.da3 = da3.to(self.dev_geo)
            lap("da3_to_gpu0_s")
        # vLLM sizes its KV cache from what is free on its GPU while it profiles: nothing else allocates there until it is up
        wait_vllm(self.vllm)
        lap("vllm_ready_s")
        self.cpu_pool = ThreadPoolExecutor(CPU)
        vllm_warm = self.cpu_pool.submit(self.warm_vllm)
        if self.n_gpu == 1:
            self.da3 = da3.to(self.dev_geo)
        self.sam = {self.dev_seg: sam.to(self.dev_seg)}
        if self.n_gpu > 1:  # "balanced" lets GPU 0 run vocabulary frames too
            self.sam[self.dev_geo] = copy.deepcopy(sam).to(self.dev_geo)
        lap("to_gpu_s")
        ip = self.proc.image_processor
        assert (ip.size["height"], ip.size["width"]) == (SAM_SIDE, SAM_SIDE), ip.size
        self.norm = {d: {"da3": (torch.tensor([.485, .456, .406], device=d).view(1, 3, 1, 1), torch.tensor([.229, .224, .225], device=d).view(1, 3, 1, 1)),
                         "sam": (torch.tensor(ip.image_mean, device=d).view(1, 3, 1, 1), torch.tensor(ip.image_std, device=d).view(1, 3, 1, 1))}
                     for d in self.devs}
        tok = self.proc(text=TEXTS, return_tensors="pt")
        self.text = {}
        for d, model in self.sam.items():
            t = tok.to(d)
            with torch.cuda.device(d), torch.inference_mode():
                enc = model.get_text_features(input_ids=t["input_ids"], attention_mask=t["attention_mask"])
            wrapped = hasattr(enc, "pooler_output")  # newer transformers wrap it in an output object
            self.text[d] = (enc.pooler_output if wrapped else enc, t["attention_mask"], wrapped)
        lap("text_features_s")
        if self.compile:  # static shapes: person chunks are padded to PERSON_FRAMES; warm-up below compiles every shape used
            torch._dynamo.config.cache_size_limit = 32
            for m in self.sam.values():
                for name in ("vision_encoder", "detr_encoder", "detr_decoder", "mask_decoder"):
                    setattr(m, name, torch.compile(getattr(m, name), dynamic=False))
        self.seg_stream = torch.cuda.Stream(self.dev_seg)
        with torch.inference_mode():  # kernels, allocator, cuBLAS handles at the shapes the runs use
            self.da3.forward(torch.rand(1, 16, 3, *DA3_HW, device=self.dev_geo), None, None, [], False, False, "saddle_balanced")
            torch.cuda.synchronize(self.dev_geo)
            lap("warm_da3_s")
            for d in self.sam:
                with torch.cuda.device(d):
                    noise = torch.randint(0, 255, (PERSON_FRAMES, 720, 1280, 3), dtype=torch.uint8, device=d)
                    vision = self.sam[d].get_vision_features(pixel_values=self.sam_pixels(d, noise))
                    self.sam_forward(d, vision, PERSON_FRAMES, [0, 1])
                    self.sam_forward(d, self.vision_frames(vision, [0]), 1, list(range(2, len(TEXTS))))
                    self.sam[d].get_vision_features(pixel_values=self.sam_pixels(d, noise[:1]))  # a vocabulary frame without cached features
                    torch.cuda.synchronize(d)
            lap("warm_sam3_s")
        depth = torch.full((2, *DA3_HW), 2., device=self.dev_geo)
        geo.fuse(depth, np.repeat(np.array([[[300., 0, 252], [0, 300, 140], [0, 0, 1]]]), 2, 0), np.repeat(np.eye(4)[None], 2, 0),
                 torch.full((2, *DA3_HW, 3), .5, device=self.dev_geo))
        lap("warm_open3d_s")
        b["vllm_warm"] = vllm_warm.result()
        b["process_pool_pids"] = len(set(self.proc_pool.map(warm_worker, range(PROCS))))
        lap("warm_vllm_done_s")
        for d in self.devs:
            with torch.cuda.device(d):
                torch.cuda.empty_cache()
        b.update(ready_s=round(time.perf_counter() - t0, 2), ready_unix=time.time(), n_gpu=self.n_gpu, gpus=listing,
                 torch=str(torch.__version__), transformers=transformers.__version__, open3d=open3d.__version__,
                 resident_gb=[round(torch.cuda.mem_get_info(d)[1] / 1e9 - torch.cuda.mem_get_info(d)[0] / 1e9, 2) for d in self.devs])

    def warm_vllm(self):
        """Two concurrent requests shaped like the real windows (24 and 36 frames of 640x480)."""
        import cv2
        t = time.perf_counter()
        noise = cv2.imencode(".jpg", np.random.default_rng(0).integers(0, 255, (480, 640, 3), np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes()
        windows = [(0., 12., [(i / 2, noise) for i in range(24)]), (12., 30., [(12 + i / 2, noise) for i in range(36)])]
        with ThreadPoolExecutor(2) as pool:
            list(pool.map(lambda w: chat(w, max_tokens=8), windows))
        return round(time.perf_counter() - t, 2)

    @modal.exit()
    def stop(self):
        if getattr(self, "vllm", None) is not None:
            self.vllm.terminate()

    @modal.method()
    def boot_info(self):
        return self.boot_record

    # --- SAM 3 ---
    def sam_pixels(self, dev, frames_bgr):
        """uint8 (n,H,W,3) BGR on dev -> the processor's resize + rescale + normalise, on the GPU, bf16 (E2's validated path)."""
        import torch
        import torch.nn.functional as F
        x = frames_bgr.permute(0, 3, 1, 2).flip(1).float() / 255
        x = F.interpolate(x, size=(SAM_SIDE, SAM_SIDE), mode="bilinear", antialias=True, align_corners=False)
        mean, std = self.norm[dev]["sam"]
        return ((x - mean) / std).to(torch.bfloat16)

    @staticmethod
    def vision_frames(vision, idx):
        return type(vision)(**{k: tuple(t[idx] for t in v) for k, v in vision.items() if k.startswith("fpn_")})

    def sam_forward(self, dev, vision, n_frames, prompt_ids):
        """n_frames x prompts in one forward, then the production rule (score >= SCORE, top TOP per pair), vectorised: only
        kept queries' masks are upsampled, straight to DA3's grid."""
        import types
        import torch
        import torch.nn.functional as F
        feats, mask, wrapped = self.text[dev]
        k = len(prompt_ids)
        ids = torch.tensor(prompt_ids, device=dev)
        f = feats[ids].repeat(n_frames, 1, 1)
        out = self.sam[dev](vision_embeds=type(vision)(**{n: tuple(t.repeat_interleave(k, 0) for t in v) for n, v in vision.items() if n.startswith("fpn_")}),
                            attention_mask=mask[ids].repeat(n_frames, 1), text_embeds=types.SimpleNamespace(pooler_output=f) if wrapped else f)
        scores = out.pred_logits.float().sigmoid()
        if getattr(out, "presence_logits", None) is not None:
            scores = scores * out.presence_logits.float().sigmoid()
        top, which = scores.topk(TOP, dim=1)
        pair, rank = torch.nonzero(top >= SCORE, as_tuple=True)
        q = which[pair, rank]
        if len(pair):
            masks = F.interpolate(out.pred_masks[pair, q][None].float().sigmoid(), size=DA3_HW, mode="bilinear", align_corners=False)[0] > .5
        else:
            masks = torch.zeros((0, *DA3_HW), dtype=torch.bool, device=dev)
        return {"frame": pair // k, "prompt": ids[pair % k], "score": top[pair, rank], "mask": masks}

    # --- one call: MP4 bytes in, core layers out ---
    @modal.method()
    def run(self, mp4, mode, save=False):
        import torch
        entered = time.time()
        assert mode in (("serial", "overlap") if self.n_gpu == 1 else ("split", "balanced")), (mode, self.n_gpu)
        log = Log()
        for d in self.devs:
            torch.cuda.reset_peak_memory_stats(d)
        vram = Vram(self.devs)
        vram.start()
        try:
            out = self.pipeline(mp4, mode, log)
        finally:
            vram.halt.set()
        log.mark("core_ready")
        files = {}
        if save:
            with log.stage("encode outputs (PLY, NPZ; not part of core)"):
                files = self.encode(out.pop("_fused"))
        out.pop("_fused", None)
        out.update(mode=mode, n_gpu=self.n_gpu, stages=log.rows, marks=log.marks, core_ready_s=log.marks["core_ready"],
                   vram={"device_used_peak_gb": [round(x, 2) for x in vram.peak],
                         "main_process_torch_reserved_peak_gb": [round(torch.cuda.max_memory_reserved(d) / 1e9, 2) for d in self.devs],
                         "main_process_torch_allocated_peak_gb": [round(torch.cuda.max_memory_allocated(d) / 1e9, 2) for d in self.devs],
                         "note": "device_used includes the vLLM process's fixed share (%.2f of its GPU) and Open3D" % VLLM_SHARE},
                   entered_unix=entered, remote_wall_s=round(time.time() - entered, 3), files=files)
        torch.cuda.empty_cache()
        return out

    def pipeline(self, mp4, mode, log):
        import cv2
        import torch
        import torch.nn.functional as F
        dev_geo, dev_seg = self.dev_geo, self.dev_seg
        overlap = mode != "serial"
        Path("/tmp/in.mp4").write_bytes(mp4)

        # SAM 3 needs no shot boundaries: its tasks are fed while decoding. One priority queue, {person, floor} chunks of
        # PERSON_FRAMES keyframes (0) before vocabulary frames (1); a worker on either GPU may take any task.
        tasks, order, lock = queue.PriorityQueue(), itertools.count(), threading.Lock()
        sealed, cuts_ready, person_ready, vocab_ready = (threading.Event() for _ in range(4))
        keys, kf_chunks, total, done, by_worker = [], {d: [] for d in self.devs}, {}, {"person": 0, "vocab": 0}, {}
        seg, cache, vocab_ids = {"person": [], "vocab": []}, {}, list(range(2, len(TEXTS)))

        def check_ready():  # under lock
            if sealed.is_set():
                if done["person"] == total["person"] and not person_ready.is_set():
                    log.mark("person_floor_masks_ready")
                    person_ready.set()
                if done["vocab"] == total["vocab"] and not vocab_ready.is_set():
                    log.mark("vocabulary_masks_ready")
                    vocab_ready.set()

        def flush():
            """Upload the newest (possibly partial) chunk of keyframes to every GPU and queue its tasks."""
            start = len(kf_chunks[dev_geo]) * PERSON_FRAMES
            idx = keys[start:start + PERSON_FRAMES]
            if idx:
                chunk = torch.from_numpy(np.stack([frames[f] for f in idx]))
                for d in self.devs:
                    kf_chunks[d].append(chunk.to(d))
                tasks.put((0, next(order), "person", start))
                for i in range(start, start + len(idx)):
                    if i % OBJECT_EVERY == 0:
                        tasks.put((1, next(order), "vocab", i))

        def sam_worker(dev, role, until=None):
            """Take tasks until the queue is sealed and empty (or `until` is set). Vision features of vocabulary frames met in
            a person chunk are kept (cloned: a view would pin the whole chunk) for their vocabulary task on the same GPU."""
            model, got, start = self.sam[dev], {"person": 0, "vocab": 0}, log.now()
            with torch.cuda.device(dev), torch.inference_mode():
                while not (until is not None and until.is_set()):
                    try:
                        _, _, kind, x = tasks.get(timeout=.01)
                    except queue.Empty:
                        if sealed.is_set():
                            break
                        continue
                    chunk = kf_chunks[dev][x // PERSON_FRAMES]
                    if kind == "person":
                        real = len(chunk)
                        if real < PERSON_FRAMES:  # pad the last chunk: fixed shapes (compiled graphs, allocator)
                            chunk = torch.cat([chunk, chunk[-1:].expand(PERSON_FRAMES - real, -1, -1, -1)])
                        vision = model.get_vision_features(pixel_values=self.sam_pixels(dev, chunk))
                        r = self.sam_forward(dev, vision, PERSON_FRAMES, [0, 1])
                        r = {k: v[r["frame"] < real] for k, v in r.items()}
                        for j in range(real):
                            if (x + j) % OBJECT_EVERY == 0:
                                cache[(dev, x + j)] = type(vision)(**{k: tuple(t[j:j + 1].clone() for t in v) for k, v in vision.items() if k.startswith("fpn_")})
                    else:
                        vision = cache.pop((dev, x), None)
                        if vision is None:
                            o = x % PERSON_FRAMES
                            vision = model.get_vision_features(pixel_values=self.sam_pixels(dev, chunk[o:o + 1]))
                        r = self.sam_forward(dev, vision, 1, vocab_ids)
                    r["frame"] = r["frame"] + x
                    seg[kind].append({k: v.to(dev_geo) for k, v in r.items()})
                    torch.cuda.current_stream(dev).synchronize()  # results visible to the other thread before it is told
                    got[kind] += 1
                    with lock:
                        done[kind] += 1
                        check_ready()
            log.add(f"{role}: SAM 3 worker on {dev} ({got['person']} person/floor chunks of {PERSON_FRAMES}, {got['vocab']} vocabulary frames)", start)
            with lock:
                w = by_worker.setdefault(f"{role} {dev}", {"person_chunks": 0, "vocab_frames": 0})
                w["person_chunks"] += got["person"]
                w["vocab_frames"] += got["vocab"]

        def seg_thread():
            with (torch.cuda.stream(self.seg_stream) if overlap and dev_seg == dev_geo else nullcontext()):
                sam_worker(dev_seg, "seg")

        seg_future, early = None, None
        if overlap:  # SAM 3 on the seg GPU from the first keyframe chunk on
            seg_future = self.cpu_pool.submit(seg_thread)
        if mode == "balanced":  # GPU 0 has nothing else to do until the cuts are known
            early = self.cpu_pool.submit(sam_worker, dev_geo, "geo (before cuts)", cuts_ready)

        # decode on this thread; grey + sharpness on threads; keyframe = sharpest of each global BLOCK-frame block, picked one
        # block behind decoding; cut chunks measured in processes as soon as their frames exist
        with log.stage("cpu: decode || grey (threads) || keyframes -> GPU + SAM 3 tasks || cut measure (processes)"):
            cap, frames, grays, futures, a, b0 = cv2.VideoCapture("/tmp/in.mp4"), [], [], [], 0, 0
            fps = cap.get(cv2.CAP_PROP_FPS)

            def pick(b):
                keys.append(max(range(b, min(b + BLOCK, len(frames))), key=lambda f: grays[f].result()[1]))
                if len(keys) % PERSON_FRAMES == 0:
                    flush()
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                frames.append(bgr)
                grays.append(self.cpu_pool.submit(gray_sharp, bgr))
                if len(frames) == a + CHUNK + dsc.SPAN + 1:
                    a0 = max(0, a - 2)
                    futures.append(self.proc_pool.submit(measure_chunk, [g.result()[0] for g in grays[a0:]], a, a + CHUNK, a0, len(frames)))
                    a += CHUNK
                if len(frames) == b0 + 2 * BLOCK:
                    pick(b0)
                    b0 += BLOCK
            cap.release()
            n = len(frames)
            while b0 < n:
                pick(b0)
                b0 += BLOCK
            if len(keys) % PERSON_FRAMES:
                flush()
            with lock:
                total.update(person=len(kf_chunks[dev_geo]), vocab=len(range(0, len(keys), OBJECT_EVERY)))
                sealed.set()
                check_ready()
            log.mark("decoded_keyframes_queued")
            ev = self.cpu_pool.submit(self.events, frames, fps, log) if overlap else None  # needs frames only
            futures.append(self.proc_pool.submit(measure_chunk, [g.result()[0] for g in grays[max(0, a - 2):]], a, n, max(0, a - 2), n))
            parts = [f.result() for f in futures]
        with log.stage("cpu: cut rules -> shots"):
            cuts = cuts_from(stitch(parts), n)
            shots = [(a, b) for a, b in cuts["segments"] if b - a + 1 >= MIN_SHOT]
            shot_pos = [[i for i, f in enumerate(keys) if a <= f <= b] for a, b in shots]
            kf = torch.cat(kf_chunks[dev_geo])
        cuts_ready.set()
        if early is not None:
            early.result()  # GPU 0 finishes its current SAM 3 task, then DA3

        def wait(event):
            """Wait for a SAM 3 signal, or raise what killed the SAM 3 thread."""
            while not event.wait(.05):
                if seg_future is not None and seg_future.done():
                    seg_future.result()
                    if not event.is_set():
                        raise RuntimeError("SAM 3 thread ended without its signal")

        # geometry: DA3 per shot, everything stays on dev_geo
        def da3_shots():
            out = []
            mean, std = self.norm[dev_geo]["da3"]
            for si, pos in enumerate(shot_pos):
                with torch.inference_mode(), log.stage(f"geo: DA3 any-view shot {si} ({len(pos)} views)", dev_geo):
                    x = F.interpolate(kf[pos].permute(0, 3, 1, 2).flip(1).float() / 255, size=DA3_HW, mode="area")
                    raw = self.da3.forward(((x - mean) / std)[None], None, None, [], False, False, "saddle_balanced")
                    k = len(pos)
                    w2c = torch.eye(4, device=dev_geo, dtype=torch.float64).repeat(k, 1, 1)
                    w2c[:, :3] = raw["extrinsics"].reshape(k, -1, 4)[:, :3].double()
                    out.append({"colors": x.permute(0, 2, 3, 1).contiguous(), "depth": raw["depth"].reshape(k, *DA3_HW).float(),
                                "c2w": torch.linalg.inv(w2c).float(), "K": raw["intrinsics"].reshape(k, 3, 3).float()})
            log.mark("cameras_ready")
            return out

        shots_gpu = da3_shots()
        if not overlap:
            seg_thread()
        if mode == "balanced":  # GPU 0 takes SAM 3 tasks once DA3 is done, until the person/floor masks are complete
            sam_worker(dev_geo, "geo", until=person_ready)
        wait(person_ready)
        with torch.inference_mode(), log.stage("geo: gather person/floor masks", dev_geo):
            person = {k: torch.cat([r[k] for r in seg["person"]]).to(dev_geo) for k in ("frame", "prompt", "score", "mask")}
            is_person = person["prompt"] == 0
            people_union = union_by_frame(person["frame"][is_person], person["mask"][is_person], len(keys))
            dyn = F.max_pool2d(people_union[:, None].float(), 5, 1, 2)[:, 0] > 0  # 2 px margin at 504x280
            floor = union_by_frame(person["frame"][~is_person], person["mask"][~is_person], len(keys))

        result_shots, fused = [], []
        for si, (pos, g) in enumerate(zip(shot_pos, shots_gpu)):
            p = torch.tensor(pos, device=dev_geo)
            with torch.inference_mode(), log.stage(f"geo: floor plane + scale shot {si}", dev_geo):
                plane = floor_plane(g["depth"], g["K"], g["c2w"], floor[p])
                mpu = CAMERA_HEIGHT_M / plane["camera_height_units"] if plane else 1.
                depth_m, c2w_m = g["depth"] * mpu, g["c2w"].clone()
                c2w_m[:, :3, 3] *= mpu
            with torch.inference_mode(), log.stage(f"geo: TSDF + points shot {si}", dev_geo):
                d = depth_m.clone()
                d[dyn[p]] = 0
                geo.edge_filter(d)
                tsdf = geo.fuse(d, g["K"].cpu().numpy(), c2w_m.cpu().numpy().astype(np.float64), g["colors"])
            with torch.inference_mode(), log.stage(f"geo: people 3D + tracks shot {si}", dev_geo):
                local = {int(q): j for j, q in enumerate(pos)}
                dets = []
                for j in torch.nonzero(is_person).squeeze(1).tolist():
                    fk = int(person["frame"][j])
                    if fk not in local:
                        continue
                    sel = person["mask"][j][::2, ::2] & (depth_m[local[fk]][::2, ::2] > 0)
                    vy, vx = torch.nonzero(sel, as_tuple=True)
                    if len(vy) < 20:
                        continue
                    pts = backproject(depth_m, g["K"], c2w_m, torch.full_like(vy, local[fk]), vy, vx, 2)
                    xyz = pts.median(0).values
                    if plane:  # foot point: drop onto the floor plane
                        nrm, pt = plane["normal"], plane["point"] * mpu
                        xyz = xyz - ((xyz - pt) @ nrm) * nrm
                    dets.append({"frame": keys[fk], "t": round(keys[fk] / fps, 3), "xyz": xyz.round(decimals=3).tolist(), "score": round(float(person["score"][j]), 3)})
                tracks = link_tracks(dets)
            result_shots.append({"shot": list(shots[si]), "keyframes": [keys[q] for q in pos], "object_keyframes": [keys[q] for q in pos if q % OBJECT_EVERY == 0],
                                 "metres_per_unit": mpu, "scale_source": "floor plane, camera 1.6 m above it" if plane else "none (unit scale)",
                                 "floor": {k: v for k, v in (plane or {}).items() if k not in ("normal", "point")} |
                                          ({"normal": plane["normal"].tolist(), "point_m": (plane["point"] * mpu).tolist()} if plane else {}),
                                 "c2w_m": c2w_m.cpu().numpy().round(5).tolist(), "K_504x280": g["K"].cpu().numpy().round(3).tolist(),
                                 "tsdf": {"timing": tsdf["timing"], "points": tsdf["n_points"], "triangles": tsdf["n_triangles"]},
                                 "people": {"detections": len(dets), "tracks": [[{k: d[k] for k in ("frame", "xyz")} for d in t] for t in tracks],
                                            "tracks_ge_3": sum(len(t) >= 3 for t in tracks)}})
            fused.append({"points": tsdf["points"], "colors": tsdf["colors"], "mesh": tsdf["mesh"]})
        log.mark("mesh_points_people_ready")

        if mode == "balanced":
            sam_worker(dev_geo, "geo")
        wait(vocab_ready)
        if seg_future is not None:
            seg_future.result()  # re-raise anything the SAM 3 thread hit
        with torch.inference_mode(), log.stage("geo: lift + merge + name objects (all shots)", dev_geo):
            vocab = {k: torch.cat([r[k] for r in seg["vocab"]]) for k in ("frame", "prompt", "score", "mask")}
            for si, (pos, g) in enumerate(zip(shot_pos, shots_gpu)):
                local = torch.full((len(keys),), -1, dtype=torch.long, device=dev_geo)
                local[torch.tensor(pos, device=dev_geo)] = torch.arange(len(pos), device=dev_geo)
                sel = local[vocab["frame"]] >= 0
                mpu = result_shots[si]["metres_per_unit"]
                c2w_m = g["c2w"].clone()
                c2w_m[:, :3, 3] *= mpu
                objects, stats = lift(vocab["mask"][sel], local[vocab["frame"][sel]], vocab["prompt"][sel], vocab["score"][sel],
                                      g["depth"] * mpu, g["K"], c2w_m, dyn[torch.tensor(pos, device=dev_geo)])
                result_shots[si].update(objects=objects, lift=stats)
        log.mark("objects_ready")

        if overlap:
            events = ev.result()
        else:
            events = self.events(frames, fps, log)
        log.mark("events_ready")
        words = {}
        for w, c in zip(*np.unique(vocab["prompt"].cpu().numpy(), return_counts=True)):
            words[TEXTS[w]] = int(c)
        return {"frames": n, "fps": fps, "cuts": cuts, "keyframes": len(keys), "object_keyframes": total["vocab"],
                "keyframes_outside_shots": len(keys) - sum(map(len, shot_pos)),
                "detections": {"person": int(is_person.sum()), "floor": int((~is_person).sum()), "vocabulary": int(len(vocab["prompt"])), "by_word": words},
                "sam3_tasks_by_worker": by_worker, "shots": result_shots, "events": events, "_fused": fused}

    def events(self, frames, fps, log):
        import cv2
        with log.stage("ev: window frames -> JPEG (CPU)"):
            windows = []
            for t0, t1 in video_events.bounds(len(frames) / fps, 12.):
                picked = []
                for t in np.arange(t0, t1, .5):  # the reference's 2 fps, 4:3 centre at 640x480
                    i = int(round(t * fps))
                    if i < len(frames):
                        h, w = frames[i].shape[:2]
                        x0 = (w - h * 4 // 3) // 2
                        small = cv2.resize(frames[i][:, x0:w - x0], (640, 480), interpolation=cv2.INTER_AREA)
                        picked.append((float(t), cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes()))
                windows.append((float(t0), float(t1), picked))
        with log.stage(f"ev: Qwen3-VL-8B via vLLM ({len(windows)} windows at once)"):
            with ThreadPoolExecutor(len(windows)) as pool:
                return list(pool.map(chat, windows))

    def encode(self, fused):
        import io
        import open3d as o3d
        files = {}
        for si, f in enumerate(fused):
            buf = io.BytesIO()
            np.savez_compressed(buf, points_m=f["points"].astype(np.float32), colors=f["colors"])
            files[f"shot{si}-points.npz"] = buf.getvalue()
            o3d.io.write_triangle_mesh(f"/tmp/shot{si}.ply", f["mesh"])
            files[f"shot{si}-mesh.ply"] = Path(f"/tmp/shot{si}.ply").read_bytes()
        return files


# ---------- local ----------

PLANS = {"1gpu": ["serial", "overlap", "serial", "overlap"], "1gpu-mps": ["serial", "overlap", "serial", "overlap"],
         "2gpu": ["split", "balanced", "split", "balanced"], "2gpu-mps": ["split", "balanced", "split", "balanced"],
         "1gpu-compile": ["overlap", "overlap", "overlap"], "2gpu-compile": ["balanced", "balanced", "balanced"]}


def plain(o):
    return o.item() if hasattr(o, "item") else str(o)


def usd_per_s(n_gpu):
    return n_gpu * PRICE["A100-80GB"] + CPU * PRICE["cpu_core"] + MEMORY_GIB * PRICE["gib"]


def drive(cfg, out, mp4, plan):
    cls = Core if cfg.startswith("1gpu") else Core.with_options(gpu="A100-80GB:2")
    core = cls(mps=int(cfg.endswith("-mps")), compile=int(cfg.endswith("-compile")))
    rec = {"config": cfg, "plan": plan, "runs": []}
    path = out / f"{cfg}.json"
    submitted = time.time()
    try:
        boot = core.boot_info.remote()
    except Exception as error:  # a failed boot must not lose the other config
        rec["boot_error"] = repr(error)[:3000]
        path.write_text(json.dumps(rec, indent=1, default=plain))
        return rec
    rec["boot"] = {**boot, "client_submitted_unix": submitted, "client_returned_unix": time.time(),
                   "submit_to_enter_s_two_clocks": round(boot["entered_unix"] - submitted, 1),
                   "submit_to_ready_s_two_clocks": round(boot["ready_unix"] - submitted, 1)}
    path.write_text(json.dumps(rec, indent=1, default=plain))
    print(cfg, "ready:", json.dumps({k: v for k, v in rec["boot"].items() if k.endswith("_s") or k.endswith("_two_clocks")}), flush=True)
    for i, mode in enumerate(plan):
        t = time.time()
        try:
            r = core.run.remote(mp4, mode, save=(i == len(plan) - 1))
        except Exception as error:
            rec["runs"].append({"mode": mode, "error": repr(error)[:3000]})
            path.write_text(json.dumps(rec, indent=1, default=plain))
            print(cfg, mode, "failed:", repr(error)[:300], flush=True)
            continue
        for name, blob in r.pop("files").items():
            (out / f"{cfg}-{i}-{mode}-{name}").write_bytes(blob)
        r.update(index=i, first_call_after_boot=(i == 0), client_wall_s=round(time.time() - t, 3))
        r["transfer_and_call_overhead_s"] = round(r["client_wall_s"] - r["remote_wall_s"], 3)
        rec["runs"].append(r)
        rec["container_s_so_far_two_clocks"] = round(time.time() - boot["entered_unix"], 1)
        rec["usd_estimate_so_far"] = round((time.time() - submitted) * usd_per_s(2 if cfg.startswith("2gpu") else 1), 3)  # upper bound: queue is not billed
        path.write_text(json.dumps(rec, indent=1, default=plain))
        print(cfg, i, mode, json.dumps({"core_ready_s": r["core_ready_s"], "marks": r["marks"], "vram": r["vram"]["device_used_peak_gb"],
                                        "client_wall_s": r["client_wall_s"]}), flush=True)
    return rec


@app.local_entrypoint()
def main(out: str, configs: str = "1gpu,1gpu-compile,2gpu,2gpu-compile", smoke: bool = False):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)  # never reuse a run folder
    mp4 = (CLIP / "source-full.mp4").read_bytes()
    plans = {c: PLANS[c][1:2] for c in PLANS} if smoke else PLANS
    with ThreadPoolExecutor(4) as pool:
        list(pool.map(lambda c: drive(c, out, mp4, plans[c]), configs.split(",")))


def evaluate(run_dir):
    """Quality next to the speed: walk-shot poses vs DROID (Sim3), objects vs today's object map, events vs runs/me340-events-197."""
    sys.path.insert(0, str(HERE / "m3_e5"))
    import vllm_events  # E5: temporal matching of events
    run_dir, runs = Path(run_dir), PHASE2 / "runs"
    droid = np.load(runs / "droid-me340-165-171/prediction.npz")["poses_c2w"].astype(np.float64)
    mpn = json.loads((runs / "da3-posed-me340-223-shotc/metric-scale.json").read_text())["metres_per_native_unit"]
    names = json.loads((runs / "me340-entity-names-200/names.json").read_text())
    omap = json.loads((runs / "me340-entity-names-200/object-map.json").read_text())
    walk = (226, 899)
    ref = [e for e in omap["entities"] if e["entityId"].startswith("object-") and sum(walk[0] <= f < walk[1] for f in e["sourceFrames"]) >= 3
           and not any(14 <= f < walk[0] for f in e["sourceFrames"])]  # E1's filter: seen >= 3 times in the walk shot, never in the cut-away
    ref_xyz = np.array([e["centroidNative"] for e in ref]) * mpn
    ref_cat = [names.get(e["entityId"], {}).get("category") for e in ref]
    reference_events = json.loads((runs / "me340-events-197/events.json").read_text())["windows"]
    report = {}
    for path in sorted(run_dir.glob("*gpu*.json")):
        rec = json.loads(path.read_text())
        for r in rec.get("runs", []):
            if "shots" not in r:
                continue
            walk_shot = next(s for s in r["shots"] if s["shot"][0] <= 500 <= s["shot"][1])
            ours = np.array(walk_shot["c2w_m"], np.float64)
            target = droid[walk_shot["keyframes"]].copy()
            target[:, :3, 3] *= mpn
            s, R, t = geo.align_sim3(ours, target)
            err = np.linalg.norm((s * (R @ ours[:, :3, 3].T)).T + t - target[:, :3, 3], axis=1)
            path_m = float(np.linalg.norm(np.diff(target[:, :3, 3], axis=0), axis=1).sum())
            objs = walk_shot["objects"]
            row = {"timing": {"core_ready_s": r["core_ready_s"], "marks": r["marks"], "stages": {x["stage"]: x["s"] for x in r["stages"]},
                              "first_call_after_boot": r["first_call_after_boot"], "transfer_and_call_overhead_s": r["transfer_and_call_overhead_s"]},
                   "vram": r["vram"], "counts": {"keyframes": r["keyframes"], "object_keyframes": r["object_keyframes"], "detections": r["detections"],
                                                 "cuts": r["cuts"]["segments"], "sam3_tasks_by_worker": r.get("sam3_tasks_by_worker")},
                   "walk_shot_poses": {"ate_rmse_m": round(float(np.sqrt((err ** 2).mean())), 4), "path_m_reference": round(path_m, 3),
                                       "our_metres_over_reference_metres": round(1 / s, 4),
                                       "note": "both scales assume a 1.6 m camera height; ours from the SAM 3 floor plane, the reference from its own floor rule"},
                   "objects": {"ours_confirmed": len(objs), "reference_walk_entities": len(ref)}}
            if objs:
                xyz = (s * (R @ np.array([o["centroid_m"] for o in objs]).T)).T + t
                d = np.linalg.norm(ref_xyz[:, None] - xyz[None], axis=2)
                near = d.argmin(1)
                for th in (.3, .5):
                    row["objects"][f"reference_recall_{th}m"] = round(float((d.min(1) < th).mean()), 3)
                    row["objects"][f"ours_near_reference_{th}m"] = round(float((d.min(0) < th).mean()), 3)
                hit = d.min(1) < .5
                row["objects"]["matched_word_in_reference_name"] = round(float(np.mean([objs[near[i]]["word"] in (ref_cat[i] or "").lower() for i in np.flatnonzero(hit)])), 3) if hit.any() else None  # free-form names: substring
                row["objects"]["words"] = dict(sorted({o["word"]: sum(p["word"] == o["word"] for p in objs) for o in objs}.items(), key=lambda kv: -kv[1]))
            parsed = [{"t0": e["t0"], "t1": e["t1"], **(video_events.parse(e["text"]) or {"caption": None, "events": []})} for e in r["events"]]
            row["events_vs_197"] = vllm_events.agreement(reference_events, parsed)
            report[f"{path.stem}-{r['index']}-{r['mode']}"] = row
    (run_dir / "evaluation.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


def self_check():
    import cv2
    rng = np.random.default_rng(0)
    tex = [cv2.GaussianBlur(rng.integers(0, 256, (1100, 1700), np.uint8), (0, 0), 2.5) for _ in range(2)]
    frames = [cv2.cvtColor(cv2.resize(tex[i >= 30][40 + 3 * (i % 30): 760 + 3 * (i % 30), 60 + 5 * (i % 30): 1340 + 5 * (i % 30)], (1280, 720)),
                           cv2.COLOR_GRAY2BGR) for i in range(55)]  # two walks spliced at 30
    whole = dsc.measure([raster_gray(f) for f in frames])
    for chunk in (8, 16, 64):
        starts = list(range(0, len(frames), chunk))
        gray = [raster_gray(f) for f in frames]
        m = stitch([measure_chunk(gray[max(0, a - 2):min(len(frames), a + chunk + dsc.SPAN + 1)], a, min(a + chunk, len(frames)), max(0, a - 2),
                                  min(len(frames), a + chunk + dsc.SPAN + 1)) for a in starts])
        for k in ("keypoints", "inliers", "jump", "spans"):
            assert np.array_equal(m[k], whole[k], equal_nan=True), (chunk, k)
        assert [h is None for h in m["homographies"]] == [h is None for h in whole["homographies"]]
        assert all(h is None or np.allclose(h, g) for h, g in zip(m["homographies"], whole["homographies"]))
    assert 30 in cuts_from(whole, len(frames))["cuts"], cuts_from(whole, len(frames))
    tracks = link_tracks([{"frame": 0, "xyz": [0, 0, 0], "score": .9}, {"frame": 0, "xyz": [3, 0, 0], "score": .8},
                          {"frame": 6, "xyz": [.3, 0, 0], "score": .9}, {"frame": 6, "xyz": [3.2, 0, 0], "score": .9},
                          {"frame": 40, "xyz": [.4, 0, 0], "score": .9}])
    assert [[d["frame"] for d in t] for t in tracks] == [[0, 6], [0, 6], [40]], tracks
    print("self-check ok: chunked cut measure == sequential (chunks 8/16/64), track linking")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--evaluate"]:
        evaluate(sys.argv[2])
    else:
        assert sys.argv[1:] == ["--self-check"], __doc__
        self_check()
