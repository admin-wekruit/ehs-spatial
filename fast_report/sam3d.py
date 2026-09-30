"""Complete object models for the fast report (FAST-BUILD-SPEC section 12 B): SAM 3D Objects in its E4 s1cfg12 setting, two
processes on GPU0 (under the container's MPS), each model judged by complete_video_objects' gate with its decision code
unchanged, accepted models handed out one by one as 40k-face GLBs. Generated display layers: never used for measurement.

Processes, all started at boot, each in its own venv (the image recipe is with_envs), talking over pipes. Arrays cross as raw
bytes, so the venvs' numpy versions never meet:
  /opt/sam3d  E4's SAM 3D recipe (torch 2.5.1 cu121, pytorch3d, kaolin, flash_attn, spconv)   Workers(gpu=0, n=2)
  /opt/gate   E4's gate pins (numpy 2.5.1, open3d 0.19, trimesh 5.1, scipy 1.18, py3.12)      GatePool(n)
The main process only orchestrates: prepare (pool, per object) -> generate (SAM 3D, rank order) -> assess (pool) -> yield.

The gate is complete_video_objects' own code (usable/failing/spread/pick_views/reliable/assess and its FIT_GATE) fed from
memory: the shot's DA3 keyframes stand in for the DROID views and the object's SAM 3 masks for the object map's. Rewritten
are only the two places that read files (view_metrics, build_input: same rules, any grid size) and the order of work (every
object prepared in parallel instead of reading all 312 posed views per object; assess in a process pool). The tolerance is
the delivered ME340 gate's own voxel in metres (VOXEL_M), since the fast path's units are estimated metres.

Inputs from A, beyond the spec's Shot and Obj: Obj["masks_lr"] = {source frame: SAM 3 mask logits (288, 288), the 1008^2
squash of the whole frame} for every object keyframe the object was seen in; the gate picks its own best view and judges
the fit on the other views, so it needs all of them. frames: (n, 720, 1280, 3) uint8 BGR (cv2's decode), best made with
frames_buffer() so every process maps the same memory instead of a copy.

  python -m fast_report.sam3d --self-check
"""
from concurrent.futures import Future
import itertools
import json
import os
from pathlib import Path
import pickle
import queue
import struct
import subprocess
import sys
import threading
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SAM3D_PY, GATE_PY = "/opt/sam3d/bin/python", "/opt/gate/bin/python"
WEIGHTS = "/weights"  # the panoptes-sam3d-weights volume, as modal_apps/sam3d_research.py mounts it
TORCH_HUB = "/opt/torch-hub"  # DINOv2 (code + weights) for SAM 3D's embedders, baked into the image
S1CFG12 = {"stage1_inference_steps": 12, "use_stage2_distillation": True, "stage2_inference_steps": 4}  # E4: 3.49 s a call, gate rate kept
VOXEL_M = .01399 * 2.8591949  # ME340's delivered gate: its fused voxel (fuse-metrics voxel_native) x its metres per native unit
DISPLAY_FACES, FIRST_PASS, PREPARING, NICE, IDLE_S = 40_000, 30, 8, 10, 3.
CORE = ("fire extinguisher", "exit sign", "forklift", "ladder", "spill", "cable", "hose", "guard")  # E2b's EHS core words
SHARED = Path("/dev/shm") if Path("/dev/shm").is_dir() else Path("/tmp")
for _p in (ROOT / "scripts", ROOT / "modal_apps"):  # complete_video_objects and the pinned recipes, in every venv
    if str(_p) not in sys.path:
        sys.path.append(str(_p))


# ---------------------------------------------------------------- pipes (shared with fast_report.splat)
class _Pickler(pickle.Pickler):
    def reducer_override(self, obj):
        if isinstance(obj, np.ndarray) and obj.dtype != object:
            return _array, (np.ascontiguousarray(obj).tobytes(), obj.dtype.str, obj.shape)
        if isinstance(obj, np.generic):
            return _same, (obj.item(),)
        return NotImplemented


def _array(data, dtype, shape):
    return np.frombuffer(bytearray(data), dtype).reshape(shape)


def _same(value):
    return value


def send(stream, message):
    import io
    buffer = io.BytesIO()
    _Pickler(buffer, protocol=4).dump(message)
    stream.write(struct.pack("<Q", buffer.tell()) + buffer.getvalue())
    stream.flush()


def recv(stream):
    head = stream.read(8)
    if len(head) < 8:
        raise EOFError("worker pipe closed")
    return pickle.loads(stream.read(struct.unpack("<Q", head)[0]))


def serve(handler, boot):
    """Worker side: stdout carries only replies (library prints go to stderr); boot() once, then one reply per message."""
    import traceback
    out = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    send(out, boot())
    while True:
        try:
            message = recv(sys.stdin.buffer)
        except EOFError:
            return
        try:
            reply = handler(message, lambda m: send(out, m))
        except Exception:
            reply = {"error": traceback.format_exc()[-4000:]}
        send(out, reply)


def log_tail(path, lines=30):
    text = Path(path).read_text(errors="replace").splitlines()[-lines:] if Path(path).exists() else []
    return "\n".join(l for l in text if not any(w in l.lower() for w in ("capabilit", "token", "secret")))


class Pool:
    """Persistent worker processes behind one priority queue: submit(message, priority tuple) -> Future. Each process sends
    its boot record first (ready()). A process that dies fails its job; when none is left, every queued job fails."""

    def __init__(self, argv, n, env, name):
        self.jobs, self.order, self.name, self.live = queue.PriorityQueue(), itertools.count(), name, n
        self.boot, self.procs, self.logs, self.lock = [Future() for _ in range(n)], [], [], threading.Lock()
        for i in range(n):
            self.logs.append(f"/tmp/fb-{name}-{i}.log")
            self.procs.append(subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=open(self.logs[i], "wb"),
                                               env=env[i] if isinstance(env, list) else env, cwd=str(ROOT)))  # a list: one env a process (its GPU)
            threading.Thread(target=self._serve, args=(i,), daemon=True).start()

    def _serve(self, i):
        proc = self.procs[i]
        try:
            self.boot[i].set_result(recv(proc.stdout))
            while True:
                _, _, future, message = self.jobs.get()
                if future is None:
                    return
                if not future.set_running_or_notify_cancel():
                    continue
                try:
                    send(proc.stdin, message)
                    reply = recv(proc.stdout)
                except (EOFError, OSError) as error:
                    future.set_exception(RuntimeError(f"{self.name}-{i} died: {error}\n{log_tail(self.logs[i])}"))
                    raise
                if isinstance(reply, dict) and "error" in reply:
                    future.set_exception(RuntimeError(f"{self.name}-{i}: {reply['error']}"))
                else:
                    future.set_result(reply)
        except Exception as error:
            if not self.boot[i].done():
                self.boot[i].set_exception(RuntimeError(f"{self.name}-{i} failed at boot: {error}\n{log_tail(self.logs[i])}"))
            with self.lock:
                self.live -= 1
                if self.live == 0:  # nobody left to take the queue
                    while not self.jobs.empty():
                        future = self.jobs.get()[2]
                        if future is not None and not future.done():
                            future.set_exception(RuntimeError(f"every {self.name} process died"))

    def ready(self, timeout=None):
        return [f.result(timeout) for f in self.boot]

    def submit(self, message, priority=(0,)):
        future = Future()
        self.jobs.put((tuple(priority), next(self.order), future, message))
        return future

    def close(self):
        for _ in self.procs:
            self.jobs.put(((float("inf"),), next(self.order), None, None))
        for proc in self.procs:
            try:
                proc.stdin.close()
                proc.wait(timeout=20)
            except Exception:
                proc.kill()


def worker_env(python, **extra):
    """The parent's environment for a worker; its PYTHONPATH (in a Modal container: where the modal client lives) only when the
    worker runs the parent's own interpreter, since in another venv it would shadow that venv's packages."""
    env = {k: v for k, v in os.environ.items() if k != "PYTORCH_CUDA_ALLOC_CONF"}  # the main env's allocator settings are its own
    inherited = [env["PYTHONPATH"]] if python == sys.executable and env.get("PYTHONPATH") else []
    env.update(PYTHONPATH=os.pathsep.join([str(ROOT), *inherited]), PYTHONUNBUFFERED="1", **{k: str(v) for k, v in extra.items()})
    return env


# ---------------------------------------------------------------- shared frames
def frames_buffer(n, height=720, width=1280, name=None):
    """A (n, height, width, 3) uint8 array in a RAM-backed file that every process maps (A decodes straight into it)."""
    path = SHARED / (name or f"fb-frames-{os.getpid()}-{time.time_ns()}.npy")
    return np.lib.format.open_memmap(path, mode="w+", dtype=np.uint8, shape=(n, height, width, 3))


def frames_path(frames):
    """The mapped file behind `frames`; a plain array is copied into one once."""
    if isinstance(frames, (str, Path)):
        return str(frames)
    if isinstance(frames, np.memmap) and frames.filename:
        frames.flush()
        return str(frames.filename)
    buffer = frames_buffer(*np.shape(frames)[:3])
    buffer[:] = frames
    buffer.flush()
    return str(buffer.filename)


def host(x):
    """A (possibly CUDA) tensor or array as a numpy array."""
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def caption_box(frame):
    """complete_video_objects.subtitle_box on a whole video frame: its 640x480 clip-frame constants scaled to the frame's height."""
    import complete_video_objects as cvo
    s = frame.shape[0] / 480
    return cvo.subtitle_box(frame, band=round(420 * s), minimum=round(150 * s * s), pad=round(6 * s))


def mask_from_logits(logits, size):
    """SAM 3's low-resolution logits (the square squash of the whole frame) -> a bool mask of (width, height) `size`, as the
    core's masks are made (sigmoid, bilinear, > 0.5)."""
    import cv2
    prob = 1 / (1 + np.exp(-np.asarray(logits, np.float32)))
    return cv2.resize(prob, size, interpolation=cv2.INTER_LINEAR) > .5


# ---------------------------------------------------------------- SAM 3D processes (/opt/sam3d)
def load_pipeline():
    """modal_apps/e4_sam3d_fast.load_pipeline (that module imports modal, which this venv lacks): mesh decoder only,
    internal depth disabled so only our pointmap places the object."""
    from huggingface_hub import snapshot_download
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    root = Path(snapshot_download("facebook/sam-3d-objects", revision=os.environ["SAM3D_MODEL_REVISION"]))
    settings = OmegaConf.load(root / "checkpoints/pipeline.yaml")
    settings.rendering_engine, settings.compile_model, settings.workspace_dir = "pytorch3d", False, str(root / "checkpoints")
    pipeline = instantiate(settings, depth_model=None, decode_formats=["mesh"], slat_decoder_gs_config_path=None, slat_decoder_gs_ckpt_path=None,
                           slat_decoder_gs_4_config_path=None, slat_decoder_gs_4_ckpt_path=None)

    def forbidden_depth(*args, **kwargs):
        raise RuntimeError("external pointmap required; internal depth disabled")
    pipeline.depth_model = forbidden_depth
    return pipeline


def run_once(pipeline, rgb, mask, pointmap, seed, options):
    """modal_apps/e4_sam3d_fast.run_once: one call with the arm's options; the posed mesh in the pointmap's PyTorch3D camera."""
    import torch
    from pytorch3d.transforms import quaternion_to_matrix
    from sam3d_objects.data.dataset.tdfy.transforms_3d import compose_transform
    started = time.monotonic()
    rgb, mask, pointmap = np.asarray(rgb, np.uint8), np.asarray(mask, bool), np.asarray(pointmap, np.float32)
    rgba = np.concatenate([rgb[..., :3], (mask.astype(np.uint8) * 255)[..., None]], -1)
    result = pipeline.run(rgba, None, seed=seed, pointmap=torch.from_numpy(pointmap).cuda(), estimate_plane=False, decode_formats=["mesh"],
                          with_mesh_postprocess=False, with_texture_baking=False, with_layout_postprocess=False, use_vertex_color=True, **options)
    mesh = result["glb"]
    basis = np.eye(4, dtype=np.float32)
    basis[:3, :3] = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
    pose = compose_transform(scale=result["scale"], rotation=quaternion_to_matrix(result["rotation"]), translation=result["translation"])
    return {"vertices": np.asarray(mesh.vertices, np.float32), "faces": np.asarray(mesh.faces, np.uint32),
            "colors": np.asarray(mesh.visual.vertex_colors)[:, :3].astype(np.uint8),
            "objectToCamera": (pose.get_matrix()[0].detach().cpu().numpy().T @ basis).astype(np.float64),
            "seconds": time.monotonic() - started}


def plane_input(h=720, w=1280):
    """A synthetic call at the real frame size for the warm-up: a box on a plane 2 m away."""
    rgb = np.full((h, w, 3), 90, np.uint8)
    rgb[h // 2 - 80:h // 2 + 80, w // 2 - 100:w // 2 + 100] = (200, 60, 40)
    mask = np.zeros((h, w), bool)
    mask[h // 2 - 80:h // 2 + 80, w // 2 - 100:w // 2 + 100] = True
    v, u = np.indices((h, w), dtype=np.float64)
    z = np.where(mask, 1.8, 2.)
    return rgb, mask, np.stack([-(u - w / 2) / 900 * z, -(v - h / 2) / 900 * z, z], -1).astype(np.float32)


def sam3d_worker():
    import torch
    state = {}

    def boot():
        t = time.time()
        state["pipeline"] = load_pipeline()
        loaded = time.time() - t
        t = time.time()
        run_once(state["pipeline"], *plane_input(), 42, S1CFG12)  # the decoder's one-off set-up (~14 s in E4) happens here
        torch.cuda.synchronize()
        warm_reserved = torch.cuda.memory_reserved()
        torch.cuda.empty_cache()  # idle through the core's 25 s beside DA3 and SAM 3 on the same GPU: hold the weights only
        return {"ready": True, "load_s": round(loaded, 1), "warm_s": round(time.time() - t, 1), "gpu": torch.cuda.get_device_name(0),
                "reserved_after_warm_gb": round(warm_reserved / 1e9, 2), "idle_reserved_gb": round(torch.cuda.memory_reserved() / 1e9, 2),
                "allocated_gb": round(torch.cuda.memory_allocated() / 1e9, 2)}

    def handle(message, _):
        if state.get("idle"):
            state["idle"].cancel()
        torch.cuda.reset_peak_memory_stats()
        start = time.time()
        out = run_once(state["pipeline"], message["rgb"], message["mask"], message["pointmap"], message["seed"], S1CFG12)
        # idle for IDLE_S (the pass is over): the ~17 GB a call caches goes back to the device, else the next report's
        # core runs beside two full caches (run fb-integrate-me340-002's second call: GPU 0 at 79 GiB, SAM 3 twice as slow)
        state["idle"] = threading.Timer(IDLE_S, torch.cuda.empty_cache)
        state["idle"].daemon = True
        state["idle"].start()
        return {**out, "start_unix": start, "end_unix": time.time(), "gpu": torch.cuda.get_device_name(0),
                "max_reserved_gb": round(torch.cuda.max_memory_reserved() / 1e9, 2)}
    serve(handle, boot)


class Workers:
    """SAM 3D processes on one GPU. Start them after the MPS daemon (A's boot): they inherit its pipe directory."""

    def __init__(self, gpu=0, n=2, python=SAM3D_PY):
        from complete_video_objects import SAM3D
        env = worker_env(python, CUDA_VISIBLE_DEVICES=gpu, HF_HOME=f"{WEIGHTS}/huggingface", HF_HUB_OFFLINE=1, LIDRA_SKIP_INIT="true", TORCH_HOME=TORCH_HUB,
                         CUDA_HOME="/usr/local/cuda", SAM3D_MODEL_REVISION=SAM3D["modelRevision"])
        self.gpu, self.pool = gpu, Pool([python, "-c", "from fast_report.sam3d import sam3d_worker; sam3d_worker()"], n, env, "sam3d")

    def ready(self, timeout=None):
        return self.pool.ready(timeout)

    def submit(self, job, priority=(0,)):
        return self.pool.submit(job, priority)

    def close(self):
        self.pool.close()


# ---------------------------------------------------------------- the gate, in memory (/opt/gate)
class Rules:
    """The two mono_room rules complete_video_objects.reliable() calls (mono_room itself imports modal): its depth-edge rule,
    unchanged, and the moving person, here the SAM 3 person masks of the keyframe."""

    def __init__(self, person):
        self.person = person

    @staticmethod
    def unreliable(depth, conf, conf_floor, edge_jump):
        pad = np.pad(depth, 1, mode="edge")
        jump = np.max([np.abs(depth - pad[a:a + depth.shape[0], b:b + depth.shape[1]]) for a, b in [(0, 1), (2, 1), (1, 0), (1, 2)]], 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            bad = jump / depth > edge_jump
        return bad | (conf < conf_floor if conf is not None and conf_floor is not None else False)

    def moving_mask(self, masks, source_index):
        return self.person[source_index]


def fast_clip(k_raster, raster_wh, full_wh, frames, person):
    """complete_video_objects.Clip for a DA3 shot: mask grid, depth raster and camera are one grid over the whole 16:9 frame
    (no 4:3 crop, no undistortion), the source frame the same view scaled."""
    import cv2
    import complete_video_objects as cvo

    class FastClip(cvo.Clip):
        def __init__(self):
            (w, h), (W, H) = raster_wh, full_wh
            sx, sy = W / w, H / h
            self.clip_to_full = np.array([[sx, 0, (sx - 1) / 2], [0, sy, (sy - 1) / 2], [0, 0, 1.]])  # pixel centres, as Clip
            self.k_raster = np.asarray(k_raster, np.float64)
            self.full_to_raster = np.linalg.inv(self.clip_to_full)
            self.k_full = self.clip_to_full @ self.k_raster
            self.full_size, self.clip_size, self.mono_room = (W, H), (w, h), Rules(person)

        def raster_mask(self, path):
            return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) > 0

        def frames(self, wanted):
            for index in sorted(wanted):
                yield index, np.ascontiguousarray(frames[index][..., ::-1])  # BGR -> RGB, as Clip.frames
    return FastClip()


class FastSource:
    """One shot of a staged report (stage()): rows, clip, entities and arguments as complete_video_objects' code reads them."""
    undistorted = wider = False  # the raster is the whole frame: no undistortion border, nothing beyond the frame's sides

    def __init__(self, spec, shot):
        import argparse
        import cv2
        root = Path(spec["dir"])
        folder = root / f"shot{shot}"
        depth, c2w, person = (np.load(folder / f"{n}.npy", mmap_mode="r") for n in ("depth", "c2w", "person"))
        keys = json.loads((folder / "keys.json").read_text())
        self.frames = np.load(spec["frames"], mmap_mode="r")
        self.rows = {f: {"source_index": f, "c2w": np.asarray(c2w[i], np.float64), "mono": np.asarray(depth[i], np.float32), "scale": 1.}
                     for i, f in enumerate(keys)}
        self.clip = fast_clip(np.load(folder / "K.npy"), depth.shape[1:][::-1], self.frames.shape[1:3][::-1], self.frames,
                              {f: np.asarray(person[i]) for i, f in enumerate(keys)})
        self.objects = {o["key"]: o for o in json.loads((root / "objects.json").read_text()) if o["shot"] == shot}
        self.entities = {k: {"entityId": k, "label": o["word"], "labelStatus": "clear", "observations": [f"{k}:{f}:0" for f in o["frames"]],
                             "sourceFrames": o["frames"], "boundsNative": [o["box_min_m"], o["box_max_m"]], "supportPoints": 0}
                         for k, o in self.objects.items()}
        self.root, self.cv2 = root, cv2
        self.args = argparse.Namespace(masks=root / "masks", dynamic_masks=None, no_captions=False, skip=frozenset(), metres_per_native=1.,
                                       generator="sam3d", output=root, invoke=False)

    def caption(self, frame):
        box = caption_box(self.frames[frame])
        if box is None:
            return None
        (x0, y0), (x1, y1) = (self.clip.full_to_raster @ [[box[0], box[2]], [box[1], box[3]], [1, 1]])[:2].T
        return [int(np.floor(x0)), int(np.floor(y0)), int(np.ceil(x1)), int(np.ceil(y1))]

    def write_masks(self, key):
        """The object's raster masks as PNG files in the masks layout complete_video_objects.mask_path() finds."""
        logits, w, h = np.load(self.root / "logits" / f"{key}.npy"), *self.clip.clip_size
        for frame, lr in zip(self.objects[key]["frames"], logits):
            path = self.root / "masks" / f"{key}-x" / f"frame-{frame:05d}" / "instance-0-mask.png"
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                self.cv2.imwrite(str(path), mask_from_logits(lr, (w, h)).astype(np.uint8) * 255)

    def full_mask(self, view):
        import complete_video_objects as cvo
        key, frame = view["observation"].split(":")[0], view["frame"]
        logits = np.load(self.root / "logits" / f"{key}.npy", mmap_mode="r")[self.objects[key]["frames"].index(frame)]
        return cvo.main_parts(mask_from_logits(logits, self.clip.full_size))


class E4Source:
    """The delivered ME340 inputs E4 judged (DROID rows, the object map's masks, 241's re-segmentations): the equivalence test's
    source, read exactly as complete_video_objects.prepare reads them (needs mono_room, so modal, in the venv)."""
    undistorted = wider = True

    def __init__(self, spec, _):
        import argparse
        import cv2
        import complete_video_objects as cvo
        import mono_room
        a = argparse.Namespace(**{k: Path(v) for k, v in spec["paths"].items()}, skip=cvo.frame_set(spec["skip"]), no_captions=False,
                               generator="sam3d", invoke=False, explicit=set(spec["entities"]))
        a.metres_per_native = json.loads((a.depth_run / "metric-scale.json").read_text())["metres_per_native_unit"]
        self.clip = cvo.Clip(a.droid_run, a.clip)
        rows = {r["source_index"]: r for r in mono_room.load(a.droid_run, None, a.depth_run)}
        self.rows = {i: r for i, r in rows.items() if i not in a.skip}
        for row in self.rows.values():
            for key in ("conf", "droid", "retained"):
                row.pop(key, None)
        self.entities = {e["entityId"]: e for e in json.loads((a.object_map / "object-map.json").read_text())["entities"]}
        self.args, self.cv2 = a, cv2

    def caption(self, frame):
        import complete_video_objects as cvo
        return cvo.subtitle_box(self.cv2.imread(str(self.clip.clip_frames[frame])))

    def write_masks(self, key):
        pass

    def full_mask(self, view):
        import complete_video_objects as cvo
        imread = lambda p: self.cv2.imread(p, self.cv2.IMREAD_GRAYSCALE) > 0
        return cvo.main_parts(imread(view["fullMask"])) if view.get("fullMask") else self.clip.full_mask(cvo.main_parts(imread(view["mask"])))


def view_metrics(src, entity, cache):
    """complete_video_objects.view_metrics on any grid: its 640x480 limits become the grid's size, the area is counted in that
    raster's pixels (so failing()'s 1500 px stays the same share of the source frame), and the raster's own edge counts only where
    the raster is undistorted (the DA3 raster is the whole frame, so its edge is the frame's)."""
    import cv2
    import complete_video_objects as cvo
    clip, args, (w, h) = src.clip, src.args, src.clip.clip_size
    border, out = cvo.BORDER, []
    unit = clip.clip_to_full[0, 0] * clip.clip_to_full[1, 1] / 1.5 ** 2  # source pixels a raster pixel covers, over the 640x480 raster's 1.5^2
    for observation in entity["observations"]:
        frame = int(observation.rsplit(":", 2)[1])
        path = cvo.mask_path(args.masks, observation)
        if frame not in src.rows or path is None:
            continue
        if frame not in cache:
            cache.clear()
            depth, moving = cvo.reliable(src.rows[frame], clip, args.dynamic_masks)
            cache[frame] = depth, moving, cvo.facing(depth, clip.k_raster), None if args.no_captions else src.caption(frame)
        depth, moving, cos, caption = cache[frame]
        clip_mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) > 0
        raster = clip.raster_mask(path)
        ys, xs = np.nonzero(clip_mask)
        if not len(xs) or not raster.any():
            continue
        ry, rx = np.nonzero(raster)
        py, px = np.nonzero(cvo.main_parts(clip_mask))
        edge = src.undistorted and (rx.min() < 2 or ry.min() < 2 or rx.max() >= raster.shape[1] - 2 or ry.max() >= raster.shape[0] - 2)
        out.append({"observation": observation, "frame": frame, "mask": str(path), "area": int(round(raster.sum() * unit)),  # failing()'s 1500 px
                    "clipBox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                    "sourceShortSide": float(min((np.ptp(px) + 1) * clip.clip_to_full[0, 0], (np.ptp(py) + 1) * clip.clip_to_full[1, 1])),
                    "solidity": float(raster.sum() / max(cv2.contourArea(cv2.convexHull(np.column_stack([rx, ry]).astype(np.int32))), 1)),
                    "cut": [side for side, hit in (("left", xs.min() < border), ("right", xs.max() >= w - border), ("top", ys.min() < border),
                                                   ("bottom", ys.max() >= h - border), ("raster edge", edge)) if hit],
                    "depthShare": float((depth[raster] > 0).mean()), "personShare": float(moving[raster].mean()),
                    "captionShare": float(clip_mask[caption[1]:caption[3] + 1, max(caption[0], 0):caption[2] + 1].sum() / clip_mask.sum()) if caption else 0.,
                    "frontal": float(np.nanmedian(cos[raster])) if np.isfinite(cos[raster]).sum() >= 20 else 0.})
    return out


def build_input(src, view, frame_rgb):
    """complete_video_objects.build_input's SAM 3D branch (the whole frame, its mask, the view's reliable depth carried onto the
    frame grid, the frame's K), without the RecGen crop and file digests nothing here reads."""
    import complete_video_objects as cvo
    row = src.rows[view["frame"]]
    depth, _ = cvo.reliable(row, src.clip, src.args.dynamic_masks)
    full = src.full_mask(view)
    mapping = np.asarray(src.clip.full_to_raster, float)
    v, u = np.indices(full.shape, dtype=np.float64)
    cx = np.floor(mapping[0, 0] * u + mapping[0, 1] * v + mapping[0, 2] + .5).astype(int)
    cy = np.floor(mapping[1, 0] * u + mapping[1, 1] * v + mapping[1, 2] + .5).astype(int)
    inside = (cx >= 0) & (cy >= 0) & (cx < depth.shape[1]) & (cy < depth.shape[0])
    full_depth = np.zeros(full.shape, np.float32)
    full_depth[inside] = depth[cy[inside], cx[inside]]
    return {"cameraToWorld": row["c2w"], "fullRgb": frame_rgb, "fullMask": full, "fullDepth": full_depth,
            "fullK": np.linalg.inv(mapping) @ src.clip.k_raster}


def journal_key(view, seed):
    """complete_video_objects.sam3d_obtain's content key of one SAM 3D input (E4's journal is filed under it)."""
    import hashlib
    import complete_video_objects as cvo
    rgb, mask, depth = (np.ascontiguousarray(np.asarray(view[n])) for n in ("fullRgb", "fullMask", "fullDepth"))
    return hashlib.sha256(b"".join([json.dumps({"seed": seed, **{k: cvo.SAM3D[k] for k in ("modelRevision", "codeRevision")}}).encode(),
                                    rgb.tobytes(), mask.astype(bool).tobytes(), depth.astype(np.float32).tobytes()])).hexdigest()[:24]


def prepare_object(src, key):
    """complete_video_objects.prepare for one object: usable views (else the 4:3 crop's side-cut views, segmented again), up to
    ATTEMPT_VIEWS spread views best first, and each one's SAM 3D input. Returns the tries (views) and their jobs."""
    import complete_video_objects as cvo
    entity = src.entities[key]
    if not set(entity["sourceFrames"]) - src.args.skip:
        return {"rejected": "skipped: cut-away shot only"}
    src.write_masks(key)
    metrics = view_metrics(src, entity, {})
    views = cvo.usable(metrics)
    side = [m for m in metrics if "side" in cvo.failing(m) and cvo.failing(m) <= {"side", "raster edge"}] if src.wider else []
    if not views and not side:
        least = min((cvo.failing(m) for m in metrics), key=len, default={"no mask with depth"})
        return {"rejected": "no usable view: the least-failing view fails " + ", ".join(sorted(least))}
    chosen = {"entity": entity, "resegment": not views, "tries": cvo.pick_views({key: views or side}, src.clip)[key]}
    if chosen["resegment"]:
        kept, lost = cvo.resegment([chosen], src.clip, src.args)
        if not kept:
            return {"rejected": lost[0]["reason"]}
    frames = dict(src.clip.frames({v["frame"] for v in chosen["tries"]}))
    jobs = []
    for view in chosen["tries"]:
        payload = build_input(src, view, frames[view["frame"]])
        jobs.append({"rgb": payload["fullRgb"], "mask": payload["fullMask"].astype(bool), "c2w": payload["cameraToWorld"],
                     "pointmap": cvo.sam3d_pointmap(payload["fullDepth"], payload["fullK"].astype(float)),
                     "keys": {s: journal_key(payload, s) for s in (cvo.SEED, cvo.EXTRA_SEED)}})
    return {"tries": chosen["tries"], "jobs": jobs}


def judge(src, key, view, c2w, generated):
    """complete_video_objects.assess on one generated mesh (unchanged: its gate, its FIT_GATE); an accepted model also comes back
    as the display GLB: decimated to DISPLAY_FACES, observed vertices opaque, the rest translucent, centred, with its placement."""
    import complete_video_objects as cvo
    from scipy.spatial import cKDTree
    vertices = cvo.sam3d_to_world(generated["vertices"], generated["objectToCamera"], c2w)
    record = {"pins": {k: cvo.SAM3D[k] for k in ("model", "modelRevision", "codeRevision")},
              "telemetry": {"workerElapsedSeconds": generated["seconds"], "gpuElapsedSeconds": generated["seconds"]},
              "runtimeEvidence": {"hardware": {"gpu": generated.get("gpu")}}}
    started = time.time()
    result = cvo.assess(src.entities[key], view, {}, (vertices, np.asarray(generated["faces"]), np.asarray(generated["colors"])), record,
                        src.rows, src.clip, src.args)
    v, out = result["validation"], {"assess_start_unix": started, "assess_end_unix": time.time()}
    out["gate"] = {k: v.get(k) for k in ("accepted_source_consistency", "rejectionReasons", "silhouette_iou", "relative_depth_median",
                                         "relative_depth_p95", "fitResidualCm", "observedCoverage", "entityCoverage", "viewsAgreeing",
                                         "judgedPoints", "icpScale", "icpRotationDeg", "observedShare", "triangles", "sourceFrame", "observation")}
    if v["accepted_source_consistency"]:
        light, light_faces, light_colors = cvo.decimate(result["vertices"], result["faces"], result["colors"] / 255, DISPLAY_FACES)
        alpha = np.where(result["flags"], cvo.OBSERVED_ALPHA, cvo.INFERRED_ALPHA)[cKDTree(result["vertices"]).query(light)[1]]
        centre = light.mean(0)
        transform = np.eye(4)
        transform[:3, 3] = centre
        out.update(glb=cvo.glb(light - centre, light_faces, np.column_stack([(light_colors * 255).round(), alpha]).astype(np.uint8), blend=True),
                   transform=transform, bounds={"min": light.min(0).tolist(), "max": light.max(0).tolist()}, faces=len(light_faces),
                   decimate_end_unix=time.time())
    return out


def gate_worker():
    """One gate process: sources cached per (staged report, shot); 'prepare' and 'assess' messages. Niced: the splat's step loop,
    the SAM 3D processes and the main process come first on the CPU (a busy pool cost the splat 8% of its steps)."""
    os.nice(NICE)
    import complete_video_objects as cvo
    sources = {}

    def source(spec, shot):
        key = (json.dumps(spec, sort_keys=True), shot)
        if key not in sources:
            if spec["kind"] == "fast":  # the gate's tolerances in the fast path's units (estimated metres)
                cvo.VOXEL, cvo.OCCLUSION = VOXEL_M, VOXEL_M / 2
                cvo.FIT_GATE["max_fit_median_native"] = VOXEL_M
            sources[key] = (FastSource if spec["kind"] == "fast" else E4Source)(spec, shot)
        return sources[key]

    def handle(message, _):
        start = time.time()
        src = source(message["src"], message.get("shot"))
        if message["op"] == "prepare":
            out = prepare_object(src, message["key"])
        else:
            out = judge(src, message["key"], message["view"], message["c2w"], message["mesh"])
        return {**out, "start_unix": start, "end_unix": time.time(), "pid": os.getpid()}
    serve(handle, lambda: {"ready": True, "pid": os.getpid()})


class GatePool(Pool):
    def __init__(self, n=16, python=GATE_PY):
        env = worker_env(python, OMP_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1, MKL_NUM_THREADS=1)
        super().__init__([python, "-c", "from fast_report.sam3d import gate_worker; gate_worker()"], n, env, "gate")


# ---------------------------------------------------------------- main side
def rank(objs, vocab=(), eligible=None):
    """First-pass order (spec 12 B): core words and EHS equipment first, then the VLM's own order, then views seen. Excluded
    labels (floor, wall, person, lights ...) and objects seen in fewer views than the gate needs to agree never go; an object
    whose 3D box overlaps one already taken (IoU >= 0.25) is the same thing again (complete_video_objects' selection rule).
    r4 (models): eligible {id: score} (the cards' well-observed objects, display_model.well_observed): only those go, best
    observed first."""
    import complete_video_objects as cvo
    order, terms = {w: i for i, w in enumerate(vocab)}, [t for ts in cvo.RELEVANT.values() for t in ts]
    ehs = lambda w: w in CORE or cvo.matches(w, terms) is not None
    keep = [o for o in objs if not cvo.matches(o["word"], cvo.EXCLUDED) and len(o["masks_lr"]) >= cvo.FIT_GATE["min_agreeing_views"]
            and (eligible is None or o["id"] in eligible)]
    key = (lambda o: (-eligible[o["id"]], str(o["id"]))) if eligible is not None else \
        (lambda o: (not ehs(o["word"]), order.get(o["word"], len(order)), -len(o["masks_lr"]), str(o["id"])))
    taken = []
    for o in sorted(keep, key=key):
        box = np.array([o["box_min_m"], o["box_max_m"]], float)
        if all(t["shot"] != o["shot"] or cvo.box_iou(box, np.array([t["box_min_m"], t["box_max_m"]], float)) < .25 for t in taken):
            taken.append(o)
    return taken


def stage(objs, shots, frames, root=None):
    """What the gate processes read, once per report, in RAM: per shot the keyframes' depth, cameras, K (the shot's median: DA3
    gives each view its own, within 0.2%) and person masks, per object its keys, frames, box and mask logits."""
    root = Path(root or SHARED / f"fb-gate-{os.getpid()}-{time.time_ns()}")
    for s in shots:
        folder = root / f"shot{s['index']}"
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / "depth.npy", host(s["depth_m"]).astype(np.float32))
        np.save(folder / "c2w.npy", host(s["c2w_m"]).astype(np.float64))
        np.save(folder / "K.npy", np.median(host(s["K"]).astype(np.float64).reshape(-1, 3, 3), 0))
        np.save(folder / "person.npy", host(s["person"]).astype(bool))
        (folder / "keys.json").write_text(json.dumps([int(k) for k in s["keys"]]))
    (root / "logits").mkdir(exist_ok=True)
    listing = []
    for i, o in enumerate(objs):
        frames_of = sorted(int(f) for f in o["masks_lr"])
        np.save(root / "logits" / f"o{i}.npy", np.stack([host(o["masks_lr"][f]).astype(np.float16) for f in frames_of]))
        listing.append({"key": f"o{i}", "id": o["id"], "shot": int(o["shot"]), "word": o["word"], "frames": frames_of,
                        "box_min_m": [float(x) for x in o["box_min_m"]], "box_max_m": [float(x) for x in o["box_max_m"]]})
    (root / "objects.json").write_text(json.dumps(listing))
    return {"kind": "fast", "dir": str(root), "frames": frames_path(frames)}, {o["id"]: f"o{i}" for i, o in enumerate(objs)}


def _external(clock, name, gpu, start, end, **n):
    if clock is not None and hasattr(clock, "external"):
        clock.external(name, gpu=gpu, start_unix=start, end_unix=end, n=n)


def gate(objs, shots, frames_host, clock, workers, pool, vocab=(), first=FIRST_PASS, background=False, deadline=None, records=None,
         eligible=None):
    """Accepted complete models, one dict each as soon as the gate passes it: {"object", "glb", "transform", "bounds", "gate"}.

    First pass: one try each (the best view, seed 42) for the first `first` ranked objects that have a usable view; objects are
    prepared in rank order, PREPARING at a time (1-4 s each), so first-pass assesses never queue behind a flood of prepares;
    SAM 3D calls go in rank order. With `background`, until `deadline` (unix)
    nothing new is started: the other views and seed 43 of the first pass's rejected ones (runner order), then the rest of the
    ranked objects. Every object's outcome and every try's gate record go to `records` (a list) if given."""
    import complete_video_objects as cvo
    ranked = rank(objs, vocab, eligible)
    src, keys = stage(objs, shots, frames_host)
    todo, slots, fed, preparing = ranked, 0, 0, 0
    events, pending, prepared = queue.Queue(), 0, {}
    records = [] if records is None else records
    late = lambda: deadline is not None and time.time() > deadline

    def feed():
        """The next prepares in rank order: only as many as can still fill the first pass, unless `background`."""
        nonlocal fed, preparing, pending
        while fed < len(todo) and preparing < PREPARING and (background or slots + preparing < first) and not late():
            pool.submit({"op": "prepare", "src": src, "shot": int(todo[fed]["shot"]), "key": keys[todo[fed]["id"]]}, (0, fed)) \
                .add_done_callback(lambda f, r=fed: events.put(("prepared", r, 0, f)))
            fed, preparing, pending = fed + 1, preparing + 1, pending + 1
    feed()

    def generate(r, n):
        """Try n (0-based, runner order: views best first, then the best view with seed 43) of ranked object r."""
        p, extra = prepared[r], n == len(prepared[r]["tries"])
        job, seed = p["jobs"][0 if extra else n], cvo.EXTRA_SEED if extra else cvo.SEED
        priority = (0 if n == 0 and p["first"] else 1 if p["first"] else 2, n, r)
        workers.submit({"rgb": job["rgb"], "mask": job["mask"], "pointmap": job["pointmap"], "seed": seed}, priority) \
            .add_done_callback(lambda f: events.put(("generated", r, n, f)))

    while pending:
        kind, r, n, future = events.get()
        o = todo[r]
        base = {"object": o["id"], "word": o["word"], "rank": r, "attempt": n + 1}
        if kind == "prepared":
            preparing -= 1
        try:
            result = future.result()
        except Exception as error:
            records.append({**base, "stage": kind, "error": str(error)[-2000:]})
            pending -= 1
            feed()
            continue
        if kind == "prepared":
            _external(clock, "sam3d.prepare", None, result["start_unix"], result["end_unix"], object=o["id"])
            if "rejected" in result or late():
                records.append({**base, "stage": "prepare", "rejected": result.get("rejected", "deadline")})
                pending -= 1
                feed()
                continue
            prepared[r] = {**result, "first": slots < first}
            slots += prepared[r]["first"]
            if prepared[r]["first"] or background:
                generate(r, 0)
            else:
                records.append({**base, "stage": "prepare", "untried": "the first pass is full"})
                pending -= 1
            feed()
        elif kind == "generated":
            _external(clock, "sam3d.generate", workers.gpu, result["start_unix"], result["end_unix"], object=o["id"], attempt=n + 1)
            p, extra = prepared[r], n == len(prepared[r]["tries"])
            view = p["tries"][0 if extra else n]
            message = {"op": "assess", "src": src, "shot": int(o["shot"]), "key": keys[o["id"]], "view": view,
                       "c2w": p["jobs"][0 if extra else n]["c2w"], "mesh": {k: result[k] for k in ("vertices", "faces", "colors", "objectToCamera", "seconds", "gpu")}}
            prepared[r]["generated"] = {k: result[k] for k in ("seconds", "start_unix", "end_unix", "max_reserved_gb")}
            phase = 0 if n == 0 and p["first"] else 1 if p["first"] else 2
            pool.submit(message, (phase, n, r)).add_done_callback(lambda f, r=r, n=n: events.put(("assessed", r, n, f)))
        else:
            _external(clock, "sam3d.assess", None, result["assess_start_unix"], result["assess_end_unix"], object=o["id"], attempt=n + 1)
            if "glb" in result:
                _external(clock, "sam3d.decimate", None, result["assess_end_unix"], result["decimate_end_unix"], object=o["id"])
            p, extra = prepared[r], n == len(prepared[r]["tries"])
            seed = cvo.EXTRA_SEED if extra else cvo.SEED
            g = result["gate"]
            records.append({**base, "stage": "assess", "first": p["first"], "seed": seed, "view": p["tries"][0 if extra else n]["frame"],
                            "accepted": g["accepted_source_consistency"], "reasons": g["rejectionReasons"], "iou": g["silhouette_iou"],
                            "fitCm": g["fitResidualCm"], "generate_s": p["generated"]["seconds"], "generated_unix": p["generated"]["end_unix"],
                            "assessed_unix": result["end_unix"], "max_reserved_gb": p["generated"]["max_reserved_gb"],
                            "key": p["jobs"][0 if extra else n]["keys"][seed]})
            if g["accepted_source_consistency"]:
                yield {"object": o["id"], "glb": result["glb"], "transform": result["transform"], "bounds": result["bounds"],
                       "gate": {**g, "attempt": n + 1, "seed": seed, "word": o["word"], "faces": result["faces"],
                                "status": "generated display model: never used for measurement"}}
                pending -= 1
            elif background and n + 1 <= len(p["tries"]) and not late():  # the next view, then the best view with seed 43
                generate(r, n + 1)
            else:
                pending -= 1


# ---------------------------------------------------------------- image
def with_envs(image):
    """A's image plus the two venvs this module runs (and the apt packages they need). Build-time settings are set per
    command, so nothing leaks into the main process's environment."""
    import sam3d_research as s  # the pinned SAM 3D recipe (E4 ran it)
    venv, pip, py = "/opt/sam3d", "/opt/sam3d/bin/pip", "/opt/sam3d/bin/python"
    build = "TORCH_CUDA_ARCH_LIST='8.0;9.0' FORCE_CUDA=1 MAX_JOBS=8 CC=gcc CXX=g++ CUDA_HOME=/usr/local/cuda"
    index = (f"PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu121 "
             f"PIP_FIND_LINKS=https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html")
    return (image.apt_install("git", "curl", "build-essential", "ninja-build", "libgl1", "libglib2.0-0", "libegl1", "libxrender1", "libxext6",
                              "libsm6", "libxi6", "libxkbcommon0", "libxxf86vm1", "libx11-6", "libgomp1", "ffmpeg")
            # the venv's compilers look for Python.h where the standalone interpreter was built (/install): point it at the real headers
            .run_commands("mkdir -p /install/include && ln -sfn \"$(python -c 'import glob, sys; print(glob.glob(sys.base_prefix + \"/include/python3.11\")[0])')\" "
                          "/install/include/python3.11",
                          f"python -m venv {venv}",
                          f"{pip} install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121",
                          f"{pip} install {s.FLASH_ATTN} hatchling hatch-requirements-txt setuptools wheel ninja")
            .run_commands(f"{build} {pip} install --no-build-isolation 'pytorch3d @ git+https://github.com/facebookresearch/pytorch3d.git@{s.PYTORCH3D}'")
            .run_commands(f"{build} {pip} install --no-build-isolation 'gsplat @ git+https://github.com/nerfstudio-project/gsplat.git@{s.GSPLAT}'")
            # upstream's requirements, resolved to exactly what E4's image has (unpinned transitive packages sent today's pip
            # backtracking through transformers for over 10 minutes)
            .add_local_file(ROOT / "fast_report/sam3d-constraints.txt", "/opt/prep/sam3d-constraints.txt", copy=True)
            .run_commands(f"for f in requirements.txt requirements.p3d.txt requirements.inference.txt; do curl -fsSL {s.UPSTREAM}/$f; echo; done "
                          "| grep -v -e '^nvidia-pyindex' -e '^pytorch3d' -e '^gsplat' -e '^flash_attn' > /tmp/sam3d-requirements.txt && "
                          f"{build} {index} {pip} install --no-build-isolation -r /tmp/sam3d-requirements.txt -c /opt/prep/sam3d-constraints.txt")
            .run_commands(f"{pip} install --no-build-isolation --no-deps 'sam3d_objects @ git+https://github.com/facebookresearch/sam-3d-objects.git@{s.CODE_REVISION}'")
            .run_commands(f"{py} -c \"import hydra; assert hydra.__version__ == '1.3.2'\" && "
                          f"curl -fsSL {s.HYDRA_UTILS} -o $({py} -c 'import hydra, os; print(os.path.join(os.path.dirname(hydra.__file__), \"core\", \"utils.py\"))')")
            .add_local_file(ROOT / "scripts/prepare_sam3d_mesh_source.py", "/opt/prep/scripts/prepare_sam3d_mesh_source.py", copy=True)
            .add_local_file(ROOT / "modal_apps/sam3d_mesh_only.patch", "/opt/prep/modal_apps/sam3d_mesh_only.patch", copy=True)
            .run_commands(f"{py} /opt/prep/scripts/prepare_sam3d_mesh_source.py")  # the reviewed mesh-only patch, with its receipt
            # SAM 3D's image and mask embedders torch.hub.load DINOv2 from GitHub (code and 1.2 GB of weights) on every start into a
            # per-container cache, and two processes starting together race on unpacking it: fetched once here instead
            .run_commands(f"TORCH_HOME={TORCH_HUB} {py} -c \"import torch; torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg', source='github', verbose=False)\"")
            # the gate: E4's gate image pins (the Mac venv's versions of what the gate imports), Python 3.12 as they need
            .run_commands("python -m pip install uv==0.8.22", "uv venv --python 3.12 /opt/gate",
                          "uv pip install --python /opt/gate/bin/python numpy==2.5.1 opencv-python-headless==5.0.0.93 scipy==1.18.0 "
                          "trimesh==5.1.0 open3d==0.19.0 pydantic==2.13.4 pillow==12.3.0"))


# ---------------------------------------------------------------- self-check (local, CPU)
def self_check():
    import io
    import tempfile
    import complete_video_objects as cvo
    # pipes: arrays of every kind and numpy scalars round-trip, and the receiving side may write into them
    message = {"a": np.arange(12, dtype=np.float32).reshape(3, 4), "b": np.zeros((2, 2), bool), "c": np.float64(1.5), "d": [np.int64(3)],
               "e": np.eye(4)[:, :2]}
    buffer = io.BytesIO()
    send(buffer, message)
    buffer.seek(0)
    back = recv(buffer)
    assert np.array_equal(back["a"], message["a"]) and back["b"].dtype == bool and back["c"] == 1.5 and type(back["d"][0]) is int
    assert np.array_equal(back["e"], np.eye(4)[:, :2]) and back["a"].flags.writeable
    # the DA3 grid: a raster pixel's centre lands on the centre of the source pixels it covers; K carried consistently
    k = np.array([[226.6, 0, 252], [0, 222.9, 140], [0, 0, 1.]])
    frames = np.zeros((3, 720, 1280, 3), np.uint8)
    clip = fast_clip(k, (504, 280), (1280, 720), frames, {})
    x = clip.clip_to_full @ [0, 0, 1]
    assert abs(x[0] - (1280 / 504 - 1) / 2) < 1e-9 and np.allclose(clip.full_to_raster @ clip.clip_to_full, np.eye(3))
    point = np.array([.4, -.2, 3.])
    assert np.allclose((clip.k_full @ point)[:2] / 3, (clip.clip_to_full @ (k @ point / 3))[:2])
    mask = np.zeros((280, 504), bool)
    mask[100:150, 200:260] = True
    full = clip.full_mask(mask)
    assert abs(full.sum() / mask.sum() - (1280 / 504) * (720 / 280)) < .1 and np.array_equal(clip.clip_mask(full), mask)
    # logits: positive inside, the right size on both grids
    logits = np.full((288, 288), -8, np.float16)
    logits[100:180, 50:120] = 8
    small, big = mask_from_logits(logits, (504, 280)), mask_from_logits(logits, (1280, 720))
    assert small.shape == (280, 504) and big.shape == (720, 1280) and abs(big.mean() - small.mean()) < .01 and abs(small.mean() - 80 * 70 / 288 ** 2) < .01
    # the depth-edge rule is mono_room's, and a keyframe's person stands in for the moving mask
    rows = {7: {"source_index": 7, "c2w": np.eye(4), "mono": np.full((280, 504), 2., np.float32), "scale": 1.}}
    rows[7]["mono"][:, 250:] = 3.
    person = np.zeros((280, 504), bool)
    person[10:20, 10:20] = True
    clip = fast_clip(k, (504, 280), (1280, 720), frames, {7: person})
    depth, moving = cvo.reliable(rows[7], clip, None)
    assert (depth[:, 249:251] == 0).all() and depth[:, 100].min() == 2 and moving[8:22, 8:22].all() and moving.sum() > person.sum()
    # captions: the 640x480 rule scaled to 720 rows finds a burnt-in line
    frame = np.zeros((720, 1280, 3), np.uint8)
    frame[661:689, 300:980] = 255
    box = caption_box(frame)
    assert box and box[1] < 661 and box[3] > 688 and box[0] < 300 and box[2] > 979
    # the fast prepare and gate plumbing on a synthetic shot: a box seen from three keyframes gets tries and SAM 3D inputs
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        keys, n = [0, 6, 12], 13
        buffer = np.lib.format.open_memmap(tmp / "frames.npy", mode="w+", dtype=np.uint8, shape=(n, 720, 1280, 3))
        rng = np.random.default_rng(0)
        buffer[:] = rng.integers(0, 255, (1, 720, 1280, 3), dtype=np.uint8)
        c2w = np.repeat(np.eye(4)[None], 3, 0)
        c2w[:, 0, 3] = [-.05, 0, .05]
        depth = np.full((3, 280, 504), 3., np.float32)
        depth[:, 110:170, 200:300] = 2.
        logits = np.full((288, 288), -8, np.float32)
        logits[113:175, 114:172] = 8
        shot = {"index": 1, "keys": keys, "depth_m": depth, "c2w_m": c2w, "K": np.repeat(k[None], 3, 0), "person": np.zeros((3, 280, 504), bool)}
        objs = [{"id": "box", "shot": 1, "word": "cabinet", "box_min_m": [-.3, -.2, 2], "box_max_m": [.3, .2, 2.2], "masks_lr": {f: logits for f in keys}},
                {"id": "again", "shot": 1, "word": "box", "box_min_m": [-.3, -.2, 2], "box_max_m": [.3, .2, 2.2], "masks_lr": {f: logits for f in keys}},
                {"id": "floor", "shot": 1, "word": "floor", "box_min_m": [0] * 3, "box_max_m": [1] * 3, "masks_lr": {f: logits for f in keys}}]
        assert [o["id"] for o in rank(objs)] == ["again"], "one of the two boxes (a tie goes by id); the same box again and the floor never go"
        assert [o["id"] for o in rank(objs, eligible={"box": 2.})] == ["box"] and rank(objs, eligible={}) == [], "r4: only the well-observed"
        spec, names = stage(objs, [shot], str(tmp / "frames.npy"), tmp / "staged")
        src = FastSource(spec, 1)
        out = prepare_object(src, names["box"])
        assert "tries" in out and len(out["jobs"]) >= 1, out
        job = out["jobs"][0]
        assert job["rgb"].shape == (720, 1280, 3) and job["mask"].shape == (720, 1280) and job["pointmap"].shape == (720, 1280, 3)
        z = job["pointmap"][..., 2][job["mask"]]
        assert np.nanmedian(z) == 2 and set(job["keys"]) == {cvo.SEED, cvo.EXTRA_SEED}
        assert (tmp / "staged/masks/o0-x/frame-00006/instance-0-mask.png").exists()
    print("fast_report.sam3d self-check passed: pipes, DA3 grid and K, logits, reliable depth and person, captions, rank, prepare")


if __name__ == "__main__":
    assert sys.argv[1:] == ["--self-check"], __doc__
    self_check()
