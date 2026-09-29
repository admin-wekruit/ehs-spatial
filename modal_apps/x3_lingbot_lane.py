"""X3, the LingBot fine lane (fast_report/lingbot_lane.py) measured next to the fast core (fb/a-core).

Analysis time = seconds from the MP4 bytes in the container to the dense layer written (the core's stub Writer: patch +
Volume commit returned). Boot (model load, CUDA-graph capture, vLLM) is never analysis time. Ephemeral `modal run` only.

  modal run modal_apps/x3_lingbot_lane.py::lane_only --videos me340,samsclub,walmart --strides 2,1 [--compile] --out RUN_DIR
      # one A100-80GB: LingBot alone (the 'third GPU' of a core container is this, in its own container)
  modal run modal_apps/x3_lingbot_lane.py::parallel --plan "me340:core,me340:lb1@1,..." --out RUN_DIR
      # 3 x A100-80GB: the core on GPUs 0-1 (its own boot); LingBot (stride s) on GPU g: 0/1 shared with the core (MPS), 2 its own
  modal run modal_apps/x3_lingbot_lane.py::evaluate --lane RUN_TAG --core RUN_TAG --out RUN_DIR      # alignment, detail, fusion
"""
import json
import multiprocessing
import os
import sys
import threading
import time
from pathlib import Path

import modal
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "scripts"), str(HERE.parent)]
import fast_report_app as fra  # noqa: E402  the core's image steps, volumes, boot
import lingbot_room as lbr  # noqa: E402  LingBot pins

CLIPS = {"me340": "me340-165", "samsclub": "samsclub-337", "walmart": "walmart-190"}  # data/clips/<clip>/source-full.mp4 (the bench's MP4)
SITE = {"me340": "me340", "samsclub": "samsclub-a2", "walmart": "walmart"}  # fast_report_eval.reference names
DATA = fra.PHASE2 / "data/clips"
app = modal.App("panoptes-fx-x3-lingbot")
VOLS = {**fra.VOLUMES, "/v/lingbot": lbr.volume, "/v/x3": modal.Volume.from_name("panoptes-fx-x3-lingbot", create_if_missing=True)}
# fra.image's steps (cached layers), then LingBot at lingbot_room's pin, then the local sources
image = (modal.Image.from_registry("nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04", add_python="3.11")
         .apt_install("git", "libgl1", "libglib2.0-0", "libgomp1")
         .pip_install("torch==2.14.0", "torchvision", "xformers", "transformers==5.17.0", "accelerate", "addict", "pillow", "scipy",
                      "open3d==0.19.0", "shapely", "pydantic", "opencv-python-headless", "sentencepiece",
                      f"git+https://github.com/ByteDance-Seed/Depth-Anything-3.git@{fra.DA3_CODE}")
         .run_commands("python -m venv /opt/vllm && PIP_EXTRA_INDEX_URL= /opt/vllm/bin/pip install -q vllm==0.11.0 transformers==4.57.1 pillow")
         .env({"HF_HUB_OFFLINE": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
         .pip_install("einops==0.8.1", "safetensors", "huggingface_hub", "trimesh", "tqdm")
         .run_commands("git init /opt/lingbot", "git -C /opt/lingbot remote add origin https://github.com/Robbyant/lingbot-map.git",
                       "git -C /opt/lingbot sparse-checkout init --cone", "git -C /opt/lingbot sparse-checkout set lingbot_map",
                       f"git -C /opt/lingbot fetch --depth=1 --filter=blob:none origin {lbr.REV}", "git -C /opt/lingbot checkout --detach FETCH_HEAD")
         .add_local_python_source("detect_shot_cuts", "m3_exp_geometry", "sam3_app", "video_events", "fast_report", "ehs_spatial",
                                  "fast_report_app", "lingbot_room", "lingbot_dense_map", "lingbot_icp_refine", "report_runner"))


def weights_path():
    from huggingface_hub import hf_hub_download
    return hf_hub_download("robbyant/lingbot-map", "lingbot-map.pt", revision=lbr.WEIGHTS_REV, cache_dir="/v/lingbot/hf")


def start_worker(gpu, compile_, procs):
    """One spawned LingBot process per GPU; returns (process, jobs, results, ready record)."""
    from fast_report import lingbot_lane
    ctx = multiprocessing.get_context("spawn")
    jobs, results = ctx.Queue(), ctx.Queue()
    proc = ctx.Process(target=lingbot_lane.worker, args=(gpu, compile_, jobs, results, weights_path(), procs))  # not daemonic: it has a process pool
    proc.start()
    ready = results.get(timeout=1500)
    if ready["kind"] != "ready":
        raise RuntimeError(ready.get("error"))
    return {"proc": proc, "jobs": jobs, "results": results, "ready": ready, "gpu": gpu}


LABELS = ["LingBot-Map weights: demo only until written licence confirmation", "scale estimated (DA3 floor plane + assumed 1.6 m)",
          "display geometry: one view's LingBot depth per point, multi-view filtered, one layer per surface; not a measurement"]


def lingbot_side(w, writer, clock=None):
    """Worker replies -> the 'dense' layer, one version per finished shot (cumulative: rows + blobs so far) -> the record."""
    rows, blobs, rec = [], {}, {"versions": []}
    while True:
        msg = w["results"].get(timeout=1500)
        if msg["kind"] == "shot":
            rows.append(msg["row"])
            if msg["blob"]:
                name, path, meta = msg["blob"]
                blobs[name] = (Path(path).read_bytes(), meta)
            seq = writer.put("dense", {"shots": rows, "source": "LingBot-Map (lingbot_room pins), fast lane X3"}, dict(blobs), "estimated", LABELS)
            rec["versions"].append({"seq": seq, "shot": msg["row"]["shot"]})
        elif msg["kind"] == "points":
            rec["points"] = msg
        elif msg["kind"] == "done":
            rec["done"] = msg
            return rec
        else:
            rec["error"] = msg.get("error")
            return rec


def add_worker_stages(clock, msg, gpu):
    for s in msg["stamps"].get("stages", []):
        clock.external(s["stage"], gpu, s["start_unix"], s["end_unix"], s.get("n"))


def usd(seconds, gpus, cpu, gib):
    return round(seconds * (gpus * fra.PRICE["A100-80GB"] + cpu * fra.PRICE["cpu_core"] + gib * fra.PRICE["gib"]), 4)


# ---------------------------------------------------------------- 1 GPU: the lane alone

@app.cls(image=image, gpu="A100-80GB", cpu=16, memory=64 * 1024, volumes=VOLS, timeout=3600, retries=0, max_containers=1, scaledown_window=20)
class Lane:
    compile_: bool = modal.parameter(default=False)

    @modal.enter()
    def boot(self):
        import torch
        t = time.perf_counter()
        torch.cuda.mem_get_info(0)  # this process's CUDA context (the Vram sampler's) at boot, not in a run
        self.w = start_worker(0, self.compile_, 14)
        self.boot_record = {**self.w["ready"], "boot_s": round(time.perf_counter() - t, 2), "gpus": fra.gpu_listing()}

    @modal.exit()
    def stop(self):
        self.w["jobs"].put(None)
        self.w["proc"].join(10)

    @modal.method()
    def boot_info(self):
        return self.boot_record

    @modal.method()
    def run(self, mp4: bytes, name: str, stride: int, tag: str):
        from fast_report.stubs import Clock, Vram, Writer
        clock = Clock()  # t0: the bytes are in the container
        vram = Vram([0], clock)
        vram.start()
        rid = f"fx-x3-{tag}-{name}-s{stride}"
        tmp = Path("/tmp/x3") / rid
        tmp.mkdir(parents=True, exist_ok=True)
        (tmp / "in.mp4").write_bytes(mp4)
        writer = Writer(VOLS["/v/layers"], "/v/layers", rid, clock)
        self.w["jobs"].put({"id": rid, "mp4": str(tmp / "in.mp4"), "stride": stride, "t0_unix": clock.t0_unix, "tmp": str(tmp),
                            "out": f"/v/x3/{tag}/{name}/s{stride}", "align": None})
        side = {}
        th = threading.Thread(target=lambda: side.update(lingbot_side(self.w, writer)), daemon=True)
        th.start()
        while "points" not in side and "error" not in side and th.is_alive():
            time.sleep(.01)
        written_by_then = clock.now()
        writer.close()
        events = [e for e in writer.events() if e["type"] == "written"]
        vram.stop()
        th.join()
        if "error" in side or "points" not in side:
            return {"error": side.get("error")}
        msg = side["points"]
        add_worker_stages(clock, msg, 0)
        VOLS["/v/x3"].commit()
        rep = clock.report(vram)
        wall = time.time() - clock.t0_unix
        return {"report": rid, "video": name, "stride": stride, "dense_written_s": [e["written_s"] for e in events], "worker_done_s": written_by_then,
                "layers": writer.layers, **rep, "worker_peak_alloc_gb": msg["stamps"].get("peak_alloc_gb"),
                "worker_peak_reserved_gb": msg["stamps"].get("peak_reserved_gb"), "picked_up_s": round(msg["stamps"]["picked_up_unix"] - clock.t0_unix, 3),
                "shots": msg["shot_stats"], "cuts": {k: msg["cuts"][k] for k in ("cuts", "segments")}, "fps": msg["fps"],
                "raw_saved_s": side["done"].get("saved_s"), "raw_dir": f"panoptes-fx-x3-lingbot:/{tag}/{name}/s{stride}",
                "call_wall_s": round(wall, 3), "usd_estimate": usd(wall, 1, 16, 64)}


def video_bytes(name):
    return (DATA / CLIPS[name] / "source-full.mp4").read_bytes()


@app.local_entrypoint()
def lane_only(videos: str = "me340", strides: str = "2", compile: bool = False, out: str = "", tag: str = ""):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    tag = tag or out.name
    lane = Lane(compile_=compile)
    submitted = time.time()
    boot = lane.boot_info.remote()
    boot["client_submit_to_ready_s"] = round(time.time() - submitted, 1)
    (out / f"boot{'-compile' if compile else ''}.json").write_text(json.dumps(boot, indent=1))
    print("ready:", json.dumps(boot), flush=True)
    for v in videos.split(","):
        mp4 = video_bytes(v)
        for s in (int(x) for x in strides.split(",")):
            r = lane.run.remote(mp4, v, s, tag)
            (out / f"lane-{v}-s{s}{'-compile' if compile else ''}.json").write_text(json.dumps(r, indent=1, default=fra.plain))
            print(json.dumps({k: r.get(k) for k in ("report", "dense_written_s", "flags", "worker_peak_alloc_gb", "gpu_peak", "error")}), flush=True)
            print(json.dumps([{k: s_.get(k) for k in ("shot", "frames", "fps_infer", "conf_threshold", "points_stats")} for s_ in r.get("shots", [])])[:1500], flush=True)


# ---------------------------------------------------------------- 3 GPUs: the core and the lane together

_CORE = fra.FastReport._get_user_cls()


@app.cls(image=image, gpu="A100-80GB:3", cpu=fra.CPU, memory=fra.MEMORY_GIB * 1024, volumes=VOLS, timeout=3600, retries=0, max_containers=1,
         scaledown_window=20)
class Parallel:
    warm_vllm = _CORE.warm_vllm
    lb_gpus: str = modal.parameter(default="1,2")

    @modal.enter()
    def boot(self):
        _CORE.__dict__["boot"]._get_raw_f()(self)  # the core's own boot: MPS daemon, vLLM on GPU 1, DA3/SAM 3/SigLIP, pools
        t = time.perf_counter()
        from concurrent.futures import ThreadPoolExecutor
        gpus = [int(x) for x in self.lb_gpus.split(",")]
        with ThreadPoolExecutor(len(gpus)) as pool:
            self.workers = dict(zip(gpus, pool.map(lambda g: start_worker(g, False, 8), gpus)))
        self.boot_record.update(lingbot_boot_s=round(time.perf_counter() - t, 2), lingbot={g: w["ready"] for g, w in self.workers.items()},
                                gpus=fra.gpu_listing())

    @modal.exit()
    def stop(self):
        if getattr(self, "vllm", None) is not None:
            self.vllm.terminate()
        for w in getattr(self, "workers", {}).values():
            w["jobs"].put(None)
            w["proc"].join(10)

    @modal.method()
    def boot_info(self):
        return self.boot_record

    @modal.method()
    def run(self, mp4: bytes, name: str, site: str, tag: str, lb_gpu: int = -1, stride: int = 2, save_da3: bool = False):
        import torch
        from fast_report import core
        from fast_report.stubs import Clock, Vram, Writer
        from lingbot_dense_map import read_points_glb
        clock = Clock()
        vram = Vram([0, 1, 2], clock)
        vram.start()
        for d in (0, 1):
            torch.cuda.reset_peak_memory_stats(d)
        rid = f"fx-x3-{tag}-{name}-{'core' if lb_gpu < 0 else f'lb{stride}at{lb_gpu}'}-{int(time.time())}"
        tmp = Path("/tmp/x3") / rid
        (tmp / "align").mkdir(parents=True, exist_ok=True)
        writer = Writer(VOLS["/v/layers"], "/v/layers", rid, clock)
        w = self.workers.get(lb_gpu)
        if lb_gpu >= 0:
            (tmp / "in.mp4").write_bytes(mp4)
            w["jobs"].put({"id": rid, "mp4": str(tmp / "in.mp4"), "stride": stride, "t0_unix": clock.t0_unix, "tmp": str(tmp),
                           "out": f"/v/x3/{tag}/{name}/par-s{stride}-g{lb_gpu}", "align": str(tmp / "align")})
        options = {"cache": False, "site_vocab": False, "window_s": None, "client_has": [], "site": site}
        job = self.run_pool.submit(core.analyse, self, mp4, options, clock, writer, None)
        lb = {}
        side = threading.Thread(target=lambda: lb.update(lingbot_side(w, writer)), daemon=True) if w else None
        if side:
            side.start()

        def closer():
            try:
                job.result()
            except Exception:  # noqa: BLE001  reported below
                pass
            if side:
                side.join()
            writer.close()
        threading.Thread(target=closer, daemon=True).start()
        got, handed = {}, False
        for e in writer.events():
            if e["type"] == "patch" and e["patch"]["layer"] in ("cameras", "room"):
                got[e["patch"]["layer"]] = e
            if not handed and len(got) == 2 and w:  # the core's DA3 frame -> the LingBot worker (files it polls for)
                cams = got["cameras"]["patch"]["data"]["shots"]
                da3 = {"shots": [{"index": s["index"], "keyframes": s["keyframes"], "c2w_m": s["c2w"], "floor": s["scale"]["floor"] if s["scale"].get("floor", {}).get("normal") else None}
                                 for s in cams]}
                for s in cams:
                    (tmp / "g.glb").write_bytes(got["room"]["blobs"][got["room"]["patch"]["blobs"][f"points-{s['index']}"]["sha256"]])
                    np.save(tmp / "align" / f"da3-points-{s['index']}.npy", read_points_glb(tmp / "g.glb")[0].astype(np.float64))
                (tmp / "align" / "da3.json").write_text(json.dumps(da3))
                (tmp / "align" / "ready").touch()
                clock.mark("da3_handed_to_lingbot")
                handed = True
        vram.stop()
        try:
            summary, error = job.result(), None
        except Exception:  # noqa: BLE001
            import traceback
            summary, error = None, traceback.format_exc()[-3000:]
        if w and "points" in lb:
            add_worker_stages(clock, lb["points"], lb_gpu)
        rep = clock.report(vram)
        wall = time.time() - clock.t0_unix
        saved = self.save_da3(tag, name, got) if save_da3 and summary else None
        return {"report": rid, "video": name, "mode": "core" if lb_gpu < 0 else f"core + LingBot s{stride} on GPU {lb_gpu}", "error": error,
                "layers": writer.layers, **rep, "summary_objects": (summary or {}).get("objects"),
                "lingbot": None if not w else {"error": lb.get("error"), "versions": lb.get("versions"), "shots": lb.get("points", {}).get("shot_stats"),
                                               "picked_up_s": round(lb["points"]["stamps"]["picked_up_unix"] - clock.t0_unix, 3) if "points" in lb else None,
                                               "worker_peak_alloc_gb": lb.get("points", {}).get("stamps", {}).get("peak_alloc_gb"),
                                               "worker_peak_reserved_gb": lb.get("points", {}).get("stamps", {}).get("peak_reserved_gb")},
                "main_process_torch_reserved_peak_gb": [round(torch.cuda.max_memory_reserved(d) / 1e9, 2) for d in (0, 1)],
                "da3_saved": saved, "call_wall_s": round(wall, 3), "usd_estimate": usd(wall, 3, fra.CPU, fra.MEMORY_GIB)}

    def save_da3(self, tag, name, got):
        """The core's DA3 shots (m.last) for the evaluation: depth (m), K, c2w, colours, person masks, keys; objects; not timed."""
        last = self.last
        out = Path(f"/v/x3/{tag}/{name}/da3")
        out.mkdir(parents=True, exist_ok=True)
        for s in last["shots"]:
            np.savez(out / f"shot{s['index']}.npz", depth_m=s["depth_m"].half().cpu().numpy(), K=s["K"].cpu().numpy(), c2w_m=s["c2w_m"].cpu().numpy(),
                     colors=(s["colors"] * 255).clamp(0, 255).byte().cpu().numpy(), person=s["person"].cpu().numpy(), keys=np.array(s["keys"]),
                     frames=np.array(s["frames"]), mpu=s["mpu"])
        (out / "cameras.json").write_text(json.dumps(got["cameras"]["patch"]["data"]))
        objs = [{k: v for k, v in o.items() if k != "mask_logits_lr"} for o in last["objs"]]
        (out / "objects.json").write_text(json.dumps(objs, default=fra.plain))
        VOLS["/v/x3"].commit()
        return str(out)


@app.local_entrypoint()
def parallel(plan: str = "me340:core", out: str = "", tag: str = "", lb_gpus: str = "1,2"):
    """plan: comma list of VIDEO:MODE, MODE = core | lbS@G (LingBot stride S on GPU G) [+save] (save the DA3 shots)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    tag = tag or out.name
    par = Parallel(lb_gpus=lb_gpus)
    submitted = time.time()
    boot = par.boot_info.remote()
    boot["client_submit_to_ready_s"] = round(time.time() - submitted, 1)
    (out / "boot-parallel.json").write_text(json.dumps(boot, indent=1, default=fra.plain))
    print("ready:", json.dumps({k: v for k, v in boot.items() if k.endswith("_s") or k in ("gpus", "resident_gb")}, default=fra.plain), flush=True)
    for i, item in enumerate(plan.split(",")):
        v, mode = item.split(":")
        save = mode.endswith("+save")
        mode = mode.removesuffix("+save")
        stride, gpu = (int(x) for x in mode[2:].split("@")) if mode.startswith("lb") else (2, -1)
        r = par.run.remote(video_bytes(v), v, SITE[v], tag, gpu, stride, save)
        r["plan_index"] = i
        (out / f"par-{i:02d}-{v}-{mode.replace('@', 'at')}.json").write_text(json.dumps(r, indent=1, default=fra.plain))
        lay = {f"{x['layer']}.v{x['version']}": x["written_s"] for x in r["layers"]}
        print(json.dumps({"i": i, "video": v, "mode": r["mode"], "layers": lay, "gpu_peak": [g["peak_gb"] for g in r["gpu_peak"]], "flags": r["flags"],
                          "error": (r["error"] or "")[-500:], "lingbot": (r["lingbot"] or {}).get("error")}), flush=True)


# ---------------------------------------------------------------- evaluation: alignment, detail, noise, fusion (not timed)

THIN = ("cable", "cord", "wire", "hose", "pipe", "rail", "railing", "handrail", "pole", "post", "ladder", "chain", "rope", "strap",
        "bar", "rod", "shelf", "shelving", "rack", "box", "boxes", "carton", "pallet", "bin", "cart", "stack")
DELIVERED = {"me340": "runs/me340-lingbot-map-222", "samsclub": "runs/report-runner/adopted/decisions/samsclub-a2/dense_gate/icp",
             "walmart": "runs/report-runner/adopted/decisions/walmart/dense_gate/icp"}  # the dense_gate decisions: raw / icp


def fuse_groups(groups, voxel, block_count):
    """m3_exp_geometry.fuse's recipe (Open3D CUDA VoxelBlockGrid, depth_max 15 m, weight >= 3) at any voxel, over several
    depth groups each at its own resolution: [(depth (N,H,W) metres, torch cuda), K (N,3,3), c2w (N,4,4), colors (N,H,W,3) [0,1] torch)]."""
    import open3d as o3d
    import open3d.core as o3c
    import torch.utils.dlpack as tdl
    import m3_exp_geometry as geo
    vbg = o3d.t.geometry.VoxelBlockGrid(attr_names=("tsdf", "weight", "color"), attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
                                        attr_channels=((1), (1), (3)), voxel_size=voxel, block_resolution=16, block_count=block_count,
                                        device=o3c.Device("CUDA:0"))
    t = time.perf_counter()
    for d, K, c2w, col in groups:
        for i in range(len(d)):
            di = o3d.t.geometry.Image(o3c.Tensor.from_dlpack(tdl.to_dlpack(d[i].contiguous())))
            ci = o3d.t.geometry.Image(o3c.Tensor.from_dlpack(tdl.to_dlpack(col[i].contiguous())))
            k, e = o3c.Tensor(np.asarray(K[i], np.float64)), o3c.Tensor(np.linalg.inv(np.asarray(c2w[i], np.float64)))
            coords = vbg.compute_unique_block_coordinates(di, k, e, depth_scale=1., depth_max=geo.TSDF_DEPTH_MAX)
            vbg.integrate(coords, di, ci, k, e, depth_scale=1., depth_max=geo.TSDF_DEPTH_MAX)
    pcd = vbg.extract_point_cloud(weight_threshold=3.)
    mesh = vbg.extract_triangle_mesh(weight_threshold=3.).to_legacy()
    o3c.cuda.synchronize()
    return {"xyz": pcd.point.positions.cpu().numpy().astype(np.float64), "rgb": (pcd.point.colors.cpu().numpy() * 255).clip(0, 255).astype(np.uint8),
            "mesh": mesh, "s": round(time.perf_counter() - t, 3), "voxel_m": voxel}


def render_points(xyz, rgb, K, c2w, wh=(1280, 720)):
    """z-buffered 1-px points, then empty pixels take their nearest-depth neighbour in 5x5 (display splat). -> HxWx3 uint8."""
    import torch
    import torch.nn.functional as F
    W, H = wh
    p = torch.as_tensor(np.asarray(xyz, np.float32), device="cuda")
    c = torch.as_tensor(np.asarray(rgb, np.uint8), device="cuda")
    cw = torch.as_tensor(np.asarray(c2w, np.float32), device="cuda")
    cam = (p - cw[:3, 3]) @ cw[:3, :3]
    z = cam[:, 2]
    ok = z > .05
    u = torch.round(K[0][0] * cam[:, 0] / z.clamp(min=1e-6) + K[0][2]).long()
    v = torch.round(K[1][1] * cam[:, 1] / z.clamp(min=1e-6) + K[1][2]).long()
    ok &= (u >= 0) & (u < W) & (v >= 0) & (v < H)
    idx, z, c = v[ok] * W + u[ok], z[ok], c[ok]
    zb = torch.full((H * W,), float("inf"), device="cuda").scatter_reduce(0, idx, z, "amin")
    win = z <= zb[idx]
    img = torch.zeros((H * W, 3), dtype=torch.uint8, device="cuda")
    img[idx[win]] = c[win]
    zb = zb.view(1, 1, H, W)
    near = -F.max_pool2d(-torch.where(torch.isfinite(zb), zb, torch.full_like(zb, 1e9)), 5, 1, 2)
    empty = ~torch.isfinite(zb)[0, 0]
    img = img.view(H, W, 3)
    if empty.any():  # fill: copy the colour of the pixel holding the 5x5 minimum depth
        pad = F.pad(torch.where(torch.isfinite(zb), zb, torch.full_like(zb, 1e9)), (2, 2, 2, 2), value=1e9)[0, 0]
        best = torch.zeros((H, W, 3), dtype=torch.uint8, device="cuda")
        for dy in range(5):
            for dx in range(5):
                m = empty & (pad[dy:dy + H, dx:dx + W] == near[0, 0]) & (near[0, 0] < 1e9)
                ys, xs = torch.nonzero(m, as_tuple=True)
                sy, sx = (ys + dy - 2).clamp(0, H - 1), (xs + dx - 2).clamp(0, W - 1)
                best[ys, xs] = img[sy, sx]
        img = torch.where(empty[..., None], best, img)
    return img.cpu().numpy()


def render_mesh(mesh, K, c2w, wh=(1280, 720)):
    """Open3D ray casting of a legacy triangle mesh, vertex colours interpolated. -> HxWx3 uint8."""
    import open3d as o3d
    import open3d.core as o3c
    W, H = wh
    scene = o3d.t.geometry.RaycastingScene()
    v, f = np.asarray(mesh.vertices, np.float32), np.asarray(mesh.triangles, np.uint32)
    if not len(f):
        return np.zeros((H, W, 3), np.uint8)
    scene.add_triangles(o3c.Tensor(v), o3c.Tensor(f))
    rays = scene.create_rays_pinhole(o3c.Tensor(np.asarray(K, np.float64)), o3c.Tensor(np.linalg.inv(np.asarray(c2w, np.float64))), W, H)
    hit = scene.cast_rays(rays)
    prim, uv = hit["primitive_ids"].numpy(), hit["primitive_uvs"].numpy()
    ok = prim != scene.INVALID_ID
    col = np.asarray(mesh.vertex_colors, np.float32)
    tri = f[prim[ok]]
    a, b = uv[ok][:, :1], uv[ok][:, 1:]
    out = np.zeros((H, W, 3), np.float32)
    out[ok] = (1 - a - b) * col[tri[:, 0]] + a * col[tri[:, 1]] + b * col[tri[:, 2]]
    return (out * 255).clip(0, 255).astype(np.uint8)


def full_k_from(K, hw):
    """K at a grid (h, w) of the whole 1280x720 frame -> K at 1280x720 (pixel-centre convention)."""
    h, w = hw
    sx, sy = 1280 / w, 720 / h
    K = np.asarray(K, np.float64)
    return np.array([[K[0, 0] * sx, 0, (K[0, 2] + .5) * sx - .5], [0, K[1, 1] * sy, (K[1, 2] + .5) * sy - .5], [0, 0, 1]])


def sim3_apply(p, s, R, t):
    return s * np.asarray(p, np.float64) @ np.asarray(R).T + np.asarray(t)


def c2w_apply(c2w, s, R, t):
    """A camera under x -> s R x + t (rotation R Rc, centre s R c + t)."""
    out = np.array(c2w, np.float64, copy=True)
    out[..., :3, :3] = np.asarray(R) @ out[..., :3, :3]
    out[..., :3, 3] = (s * out[..., :3, 3] @ np.asarray(R).T) + np.asarray(t)
    return out


def cloud_metrics(xyz, dl_tree, room_tree, objects, floor_n):
    from lingbot_dense_map import self_spread, spread_summary
    rng = np.random.default_rng(0)
    q = xyz[rng.choice(len(xyz), min(400_000, len(xyz)), replace=False)]
    out = {"points": int(len(xyz)), "thickness_3cm_patches": spread_summary(self_spread(xyz, .03), 1.),
           "thickness_10cm_patches": spread_summary(self_spread(xyz, .10), 1.)}
    for name, tree in (("delivered_lingbot", dl_tree), ("da3_tsdf", room_tree)):
        if tree is None:
            continue
        d, _ = tree.query(q, distance_upper_bound=.25, workers=16)
        out[f"share_within_{name}"] = {f"{c}cm": round(float((d <= c / 100).mean()), 4) for c in (2, 5, 10, 25)}
    per = []
    for o in objects:
        lo, hi = np.array(o["box_min_m"]) - .05, np.array(o["box_max_m"]) + .05
        inside = xyz[np.all((xyz >= lo) & (xyz <= hi), 1)]
        row = {"id": o["id"], "word": o["word"], "points": int(len(inside))}
        if len(inside) >= 30:
            import open3d as o3d
            sub = inside[rng.choice(len(inside), min(20000, len(inside)), replace=False)]  # the plane fit on a 20k sample
            pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(sub))
            o3d.utility.random.seed(0)
            model, inl = pc.segment_plane(.02, 3, 400)
            if len(inl) >= 10:
                n = np.asarray(model[:3]) / np.linalg.norm(model[:3])
                dist = (sub[inl] @ n + model[3])
                row.update(plane_inliers_of_sample=len(inl), plane_share=round(len(inl) / len(sub), 3),
                           plane_thickness_cm=round(float(np.subtract(*np.percentile(dist, [90, 10]))) * 100, 2),
                           plane_tilt_to_floor_deg=round(float(np.degrees(np.arccos(min(1., abs(n @ floor_n))))), 1) if floor_n is not None else None)
        per.append(row)
    out["objects"] = per
    return out


@app.function(image=image, gpu="A100-80GB", cpu=16, memory=96 * 1024, volumes=VOLS, timeout=3600, retries=0)
def evaluate_remote(video: str, lane_tag: str, core_tag: str, delivered: dict):
    """One video: every cloud of the comparison window in the core's DA3 metre frame; metrics; one comparison JPG."""
    import cv2
    import torch
    from scipy.spatial import cKDTree
    from fast_report import lingbot_lane as ll
    from lingbot_dense_map import read_points_glb
    import m3_exp_geometry as geo
    VOLS["/v/x3"].reload()
    ll.self_check()  # the lane's rules, before any cost
    T0 = time.perf_counter()
    root = Path(f"/v/x3/{core_tag}/{video}/da3")
    cams = json.loads((root / "cameras.json").read_text())["shots"]
    objs = json.loads((root / "objects.json").read_text())
    lo, hi = delivered["window"]
    main = max(cams, key=lambda s: len(set(range(s["frames"][0], s["frames"][1] + 1)) & set(range(lo, hi + 1))))
    si = main["index"]
    win = (max(lo, main["frames"][0]), min(hi, main["frames"][1]))
    z = np.load(root / f"shot{si}.npz")
    keys = z["keys"]
    kin = (keys >= win[0]) & (keys <= win[1])
    floor = main["scale"].get("floor") if main["scale"].get("floor", {}).get("normal") else None
    floor_n = np.array(floor["normal"]) if floor else None
    shot = {"index": si, "keyframes": main["keyframes"], "c2w_m": main["c2w"], "floor": floor}
    dev = torch.device("cuda:0")
    rec = {"video": video, "da3_shot": si, "window": list(win), "da3_keyframes_in_window": int(kin.sum()), "notes": []}
    # C0: the core's DA3 TSDF, rebuilt from its keyframes in the window with the core's recipe (people and edges out, 3 cm)
    d = torch.as_tensor(z["depth_m"][kin].astype(np.float32), device=dev)
    person = torch.as_tensor(z["person"][kin], device=dev)
    d[torch.nn.functional.max_pool2d(person[:, None].float(), 5, 1, 2)[:, 0] > 0] = 0
    geo.edge_filter(d)
    da3_col = torch.as_tensor(z["colors"][kin].astype(np.float32) / 255, device=dev)
    da3_group = (d, z["K"][kin], z["c2w_m"][kin], da3_col)
    C0 = fuse_groups([da3_group], .03, 80000)
    room_tree = cKDTree(C0["xyz"])
    clouds = {"da3_tsdf_3cm": C0}
    # the delivered LingBot layer (DROID native) -> DA3 metres: camera Sim3 on the shot's keyframes, then the capped ICP
    dl_xyz, dl_rgb = read_points_glb(Path(delivered["dir"]) / "dense-points.glb")
    droid = np.load(Path(delivered["dir"]) / "droid.npz")
    dk = [k for k in keys[kin] if k < len(droid["poses_c2w"])]
    da3_c = np.array([main["c2w"][main["keyframes"].index(int(k))] for k in dk])
    s_, R_, t_ = geo.align_sim3(droid["poses_c2w"][dk].astype(np.float64), da3_c)
    res = np.linalg.norm(sim3_apply(droid["poses_c2w"][dk][:, :3, 3], s_, R_, t_) - da3_c[:, :3, 3], axis=1)
    dl = sim3_apply(dl_xyz, s_, R_, t_)
    (s2, R2, t2), steps, refused = ll.icp(dl, C0["xyz"])
    if not refused:
        dl = sim3_apply(dl, s2, R2, t2)
        s_, R_, t_ = s2 * s_, R2 @ R_, s2 * R2 @ t_ + t2
    rec["delivered_alignment"] = {"camera_sim3_residual_m": {"median": float(np.median(res)), "p90": float(np.percentile(res, 90))},
                                  "icp": {"iterations": len(steps), "refused": refused, "last_median_cm": steps[-1]["median_cm"]},
                                  "gate_vs_da3": ll.gate_metrics(dl, C0["xyz"], floor), "points": int(len(dl))}
    dl_T = (s_, R_, t_)
    clouds["delivered_lingbot"] = {"xyz": dl, "rgb": dl_rgb}
    dl_tree = cKDTree(dl)
    # the fast lane's LingBot, per stride: points (filtered layer) + a LingBot TSDF + a joint DA3 + LingBot TSDF
    lanes = {}
    for stride in delivered["strides"]:
        f = Path(f"/v/x3/{lane_tag}/{video}/s{stride}")
        shots = sorted(f.glob("shot*.npz"))
        best = None
        for p in shots:
            y = np.load(p)
            ov = int(((y["source_frames"] >= win[0]) & (y["source_frames"] <= win[1])).sum())
            if best is None or ov > best[0]:
                best = (ov, p)
        y = dict(np.load(best[1]))
        fin = (y["source_frames"] >= win[0]) & (y["source_frames"] <= win[1])
        sf, c2w_l, K_l = y["source_frames"][fin], y["c2w"][fin].astype(np.float64), y["K"][fin]
        thr = float(y["thr"])
        lcol = frames_rgb(delivered["video"], sf, ll.LB_HW)  # the LingBot input colours, re-made (518x294 crop mode on the GPU)
        raw_d = torch.as_tensor(y["depth"][fin].astype(np.float32), device=dev)
        raw_c = torch.as_tensor(y["conf"][fin].astype(np.float32), device=dev)
        cw_t, k_t = torch.as_tensor(c2w_l, device=dev).float(), torch.as_tensor(K_l, device=dev).float()
        ol_nomask = ll.points(raw_d, raw_c, cw_t, k_t, lcol.permute(0, 3, 1, 2), thr, ll.offsets(stride), one_layer=True)  # as the lane ran
        # people out: the core's SAM 3 person masks (5x5-grown, DA3 keyframes) at the nearest keyframe, grown 9 px more
        t_v = time.perf_counter()
        near_kf = np.abs(sf[:, None] - keys[None]).argmin(1)
        pm = torch.as_tensor(z["person"][near_kf], device=dev)[:, None].float()
        pm = torch.nn.functional.max_pool2d(torch.nn.functional.interpolate(pm, size=ll.LB_HW, mode="nearest"), 19, 1, 9)[:, 0] > 0
        raw_d = raw_d.masked_fill(pm, 0.)
        mask_s = round(time.perf_counter() - t_v, 3)
        t_v = time.perf_counter()
        ol = ll.points(raw_d, raw_c, cw_t, k_t, lcol.permute(0, 3, 1, 2), thr, ll.offsets(stride), one_layer=True)  # + people masks
        ol_s = round(time.perf_counter() - t_v, 3)
        t_v = time.perf_counter()
        full = frames_full(delivered["video"], sf)
        t_up = time.perf_counter()
        dx, dr = ll.densify(raw_d, k_t, cw_t, torch.as_tensor(ol["frame"], device=dev).long(), torch.as_tensor(ol["pixel"], device=dev).long(), full)
        torch.cuda.synchronize()
        x2_s = {"decode_and_upload_full_frames_s": round(t_up - t_v, 3), "densify_s": round(time.perf_counter() - t_up, 3)}
        del full
        t_v = time.perf_counter()
        vo = ll.points(raw_d, raw_c, cw_t, k_t, lcol.permute(0, 3, 1, 2), thr, ll.offsets(stride), one_layer=False)  # no one-layer pass
        vo_s = round(time.perf_counter() - t_v, 3)
        rec.setdefault("filter_stats", {})[f"s{stride}"] = {"one_layer": ol["stats"], "one_layer_s": ol_s, "voxel_only": vo["stats"], "voxel_only_s": vo_s,
                                                             "one_layer_no_people_mask": ol_nomask["stats"], "people_mask_s": mask_s,
                                                             "people_mask_share_of_pixels": round(float(pm.float().mean()), 4),
                                                             "people_mask_nearest_keyframe_gap_frames": {"median": float(np.median(np.abs(sf - keys[near_kf]))),
                                                                                                         "max": int(np.abs(sf - keys[near_kf]).max())}}
        lbf = {int(k): i for i, k in enumerate(sf)}
        t_v = time.perf_counter()
        al = ll.align_points(ol["xyz"], ol["conf"].astype(np.float32), thr, c2w_l, lbf, shot, C0["xyz"], icp_always=True)
        al["align_s_with_icp_always"] = round(time.perf_counter() - t_v, 3)
        T = (al["transform"]["s"], al["transform"]["R"], al["transform"]["t"])
        clouds[f"lingbot_s{stride}_points"] = {"xyz": sim3_apply(ol["xyz"], *T), "rgb": ol["rgb"]}
        clouds[f"lingbot_s{stride}_points_voxel_only"] = {"xyz": sim3_apply(vo["xyz"], *T), "rgb": vo["rgb"]}
        clouds[f"lingbot_s{stride}_points_no_people_mask"] = {"xyz": sim3_apply(ol_nomask["xyz"], *T), "rgb": ol_nomask["rgb"]}
        clouds[f"lingbot_s{stride}_points_x2_fullres_colour"] = {"xyz": sim3_apply(dx.cpu().numpy(), *T), "rgb": dr.cpu().numpy()}
        rec["filter_stats"][f"s{stride}"]["densify_x2"] = {**x2_s, "points": int(len(dx))}
        ld = raw_d.clone()
        ld[~ll.usable(raw_d, raw_c, thr)] = 0
        ld *= T[0]
        grp = (ld, K_l, c2w_apply(c2w_l, *T), lcol)
        for vox in (.015,):
            clouds[f"lingbot_s{stride}_tsdf_{vox * 100:g}cm"] = fuse_groups([grp], vox, 150000)
            clouds[f"da3+lingbot_s{stride}_tsdf_{vox * 100:g}cm"] = fuse_groups([da3_group, grp], vox, 150000)
        lanes[stride] = {"alignment": al, "cams": (sf, c2w_l, K_l, T), "native": (ol["xyz"], ol["rgb"], ol["frame"]), "x2": (dx.cpu().numpy(), dr.cpu().numpy()),
                         "native_nomask": (ol_nomask["xyz"], ol_nomask["rgb"])}
        rec.setdefault("lingbot", {})[f"s{stride}"] = {"alignment": {k: v for k, v in al.items() if k != "transform"}, "scale_m_per_native": T[0],
                                                       "cell_m": ol["cell"] * T[0], "frames_in_window": int(fin.sum()), "conf_threshold": thr}
        del raw_d, raw_c, cw_t, k_t
        del ld
        torch.cuda.empty_cache()
    in_shot = [o for o in objs if o["shot"] == si]
    rec["clouds"] = {}
    for name, c in clouds.items():
        try:  # one metric's bug must not lose the evaluation
            m = cloud_metrics(np.asarray(c["xyz"], np.float64), dl_tree if name != "delivered_lingbot" else None, room_tree if name != "da3_tsdf_3cm" else None,
                              in_shot, floor_n)
        except Exception as e:  # noqa: BLE001
            m = {"error": repr(e)[:400], "points": int(len(c["xyz"]))}
        m["tsdf_s"] = c.get("s")
        rec["clouds"][name] = m
    # renders: up to 3 thin / EHS objects of the shot, each at its best keyframe, every layer from its own camera there
    ranked = sorted(in_shot, key=lambda o: (min([THIN.index(w) for w in THIN if w in o["word"].lower().split()] or [99]), -o["frames"]))
    def span_px(o):  # the object's projected box at its best keyframe, longest side (px at 1280x720)
        kc = np.array(main["c2w"][main["keyframes"].index(int(o["best_key"]))])
        Kd = full_k_from(main["K_grid"][main["keyframes"].index(int(o["best_key"]))], (280, 504))
        cs = np.array([[x, y_, zz] for x in (o["box_min_m"][0], o["box_max_m"][0]) for y_ in (o["box_min_m"][1], o["box_max_m"][1])
                       for zz in (o["box_min_m"][2], o["box_max_m"][2])])
        cam = (cs - kc[:3, 3]) @ kc[:3, :3]
        if (cam[:, 2] <= .05).any():
            return 1e9
        uv = np.stack([Kd[0, 0] * cam[:, 0] / cam[:, 2], Kd[1, 1] * cam[:, 1] / cam[:, 2]], 1)
        return float((uv.max(0) - uv.min(0)).max())
    picks = [o for o in ranked if win[0] <= o["best_key"] <= win[1] and 60 <= span_px(o) <= 480][:6]  # compact ones: the first 3 that render
    rows, rec["renders"] = [], []
    for o in picks:
        if len(rows) == 3:
            break
        try:
            fr = int(o["best_key"])
            kc = np.array(main["c2w"][main["keyframes"].index(fr)])
            Kd = full_k_from(main["K_grid"][main["keyframes"].index(fr)], (280, 504))
            corners = np.array([[x, y_, zz] for x in (o["box_min_m"][0], o["box_max_m"][0]) for y_ in (o["box_min_m"][1], o["box_max_m"][1])
                                for zz in (o["box_min_m"][2], o["box_max_m"][2])])
            cam = (corners - kc[:3, 3]) @ kc[:3, :3]
            cam = cam[cam[:, 2] > .05]
            if not len(cam):
                raise ValueError('object behind the camera')
            uv = np.stack([Kd[0, 0] * cam[:, 0] / cam[:, 2] + Kd[0, 2], Kd[1, 1] * cam[:, 1] / cam[:, 2] + Kd[1, 2]], 1)
            (x0, y0), (x1, y1) = uv.min(0), uv.max(0)
            cx, cy, half = (x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, y1 - y0, 160) * .65
            box = (int(max(0, cx - half)), int(max(0, cy - half)), int(min(1280, cx + half)), int(min(720, cy + half)))
            if box[2] - box[0] < 40 or box[3] - box[1] < 40:
                raise ValueError('object crop under 40 px')
            panels = [("video frame", frame_bgr(delivered["video"], fr)[..., ::-1])]
            panels.append(("DA3 TSDF 3cm (core)", render_mesh(C0["mesh"], Kd, kc)))
            for stride, L in lanes.items():
                sf, c2w, K, T = L["cams"]
                j = int(np.argmin(np.abs(sf - fr)))
                Kl = full_k_from(K[j], ll.LB_HW)
                xyz, rgb, fi = L["native"]
                at = f" (frame {sf[j]})" if sf[j] != fr else ""
                panels.append((f"LingBot s{stride} points{at}", render_points(xyz, rgb, Kl, c2w[j])))
                if stride == delivered["strides"][0]:  # the TSDF variants and the unmasked layer for one stride (the sheet's width)
                    panels.append((f"LingBot s{stride} x2 samples, full-res colour", render_points(*L["x2"], Kl, c2w[j])))
                    panels.append((f"LingBot s{stride} pts, people kept", render_points(*L["native_nomask"], Kl, c2w[j])))
                    lcw = c2w_apply(c2w[j], *T)
                    panels.append((f"LingBot s{stride} TSDF 1.5cm", render_mesh(clouds[f"lingbot_s{stride}_tsdf_1.5cm"]["mesh"], Kl, lcw)))
                    panels.append((f"DA3+LingBot s{stride} TSDF 1.5cm", render_mesh(clouds[f"da3+lingbot_s{stride}_tsdf_1.5cm"]["mesh"], Kd, kc)))
            dK = np.asarray(delivered["clip_k"], np.float64)  # the 640x480 raster (x0 160, scale 1.5) -> 1280x720
            Kdl = np.array([[dK[0] * 1.5, 0, dK[2] * 1.5 + 160.25], [0, dK[1] * 1.5, dK[3] * 1.5 + .25], [0, 0, 1]])
            panels.append(("delivered LingBot (full report)", render_points(dl_xyz, dl_rgb, Kdl, droid["poses_c2w"][fr])))
            crops = []
            for label, img in panels:
                c = np.ascontiguousarray(img[box[1]:box[3], box[0]:box[2]])
                c = cv2.resize(c, (int(240 * c.shape[1] / c.shape[0]), 240), interpolation=cv2.INTER_AREA)
                c = cv2.copyMakeBorder(c, 22, 2, 2, 2, cv2.BORDER_CONSTANT, value=(255, 255, 255))
                cv2.putText(c, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, .42, (0, 0, 0), 1, cv2.LINE_AA)
                crops.append(c)
            row = np.concatenate(crops, 1)
            cv2.putText(row, f"{o['word']} ({o['id']}) frame {fr}", (4, row.shape[0] - 6), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 0), 1, cv2.LINE_AA)
            rows.append(row)
            rec["renders"].append({"object": o["id"], "word": o["word"], "frame": fr, "crop_xyxy_1280x720": box, "panels": [p[0] for p in panels]})
        except Exception as e:  # noqa: BLE001  a render bug must not lose the metrics
            rec["renders"].append({"object": o["id"], "error": repr(e)[:400]})
    jpg = None
    if rows:
        width = max(r.shape[1] for r in rows)
        sheet = np.concatenate([cv2.copyMakeBorder(r, 0, 0, 0, width - r.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)) for r in rows], 0)
        jpg = cv2.imencode(".jpg", sheet[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()
    rec["eval_wall_s"] = round(time.perf_counter() - T0, 1)
    return rec, jpg


_frames_cache = {}


def frame_bgr(video, f):
    import cv2
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
    ok, img = cap.read()
    cap.release()
    assert ok, f
    return img


def frames_full(video, idx):
    """Source frames idx at full resolution, RGB uint8 (n,720,1280,3) on the GPU."""
    import cv2
    import torch
    want, got, cap, i = set(int(x) for x in idx), {}, cv2.VideoCapture(video), 0
    while len(got) < len(want):
        ok, img = cap.read()
        if not ok:
            break
        if i in want:
            got[i] = img[..., ::-1].copy()
        i += 1
    cap.release()
    return torch.as_tensor(np.stack([got[int(x)] for x in idx]), device="cuda")


def frames_rgb(video, idx, hw):
    """The LingBot input colours of source frames idx: decode, then lingbot_lane.preprocess -> (n,H,W,3) [0,1] cuda."""
    import cv2
    import torch
    from fast_report import lingbot_lane as ll
    want, got, cap, i = set(int(x) for x in idx), {}, cv2.VideoCapture(video), 0
    while len(got) < len(want):
        ok, img = cap.read()
        if not ok:
            break
        if i in want:
            got[i] = img
        i += 1
    cap.release()
    return ll.preprocess([got[int(x)] for x in idx], torch.device("cuda:0")).permute(0, 2, 3, 1).contiguous()


@app.local_entrypoint()
def evaluate(videos: str = "me340,samsclub,walmart", lane_tag: str = "", core_tag: str = "", strides: str = "2,1", out: str = "", upload: bool = True):
    import sys as _s
    _s.path.insert(0, str(HERE.parent / "scripts"))
    import fast_report_eval as fe
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    vol = VOLS["/v/x3"]
    for v in videos.split(","):
        ref = fe.reference(SITE[v])
        src = fra.PHASE2 / DELIVERED[v]
        remote = f"delivered/{v}"
        if upload:
            droid = np.load(ref["droid"])
            tmp = out / f".droid-{v}.npz"
            np.savez(tmp, poses_c2w=droid["poses_c2w"])
            with vol.batch_upload(force=True) as b:
                b.put_file(src / "dense-points.glb", f"{remote}/dense-points.glb")
                b.put_file(tmp, f"{remote}/droid.npz")
                b.put_file(DATA / CLIPS[v] / "source-full.mp4", f"{remote}/source-full.mp4")
            tmp.unlink()
        clip = json.loads((DATA / CLIPS[v] / "clip.json").read_text())
        spec = {"dir": f"/v/x3/{remote}", "video": f"/v/x3/{remote}/source-full.mp4", "window": [int(min(ref["frames"])), int(max(ref["frames"]))],
                "clip_k": clip["K"], "strides": [int(s) for s in strides.split(",")], "delivered_run": DELIVERED[v], "mpn_delivered": ref["mpn"]}
        rec, jpg = evaluate_remote.remote(v, lane_tag, core_tag, spec)
        rec["delivered"] = {k: spec[k] for k in ("delivered_run", "window", "mpn_delivered")}
        (out / f"eval-{v}.json").write_text(json.dumps(rec, indent=1, default=fra.plain))
        if jpg:
            (out / f"compare-{v}.jpg").write_bytes(jpg)
        print(json.dumps({"video": v, "wall": rec.get("eval_wall_s"), "clouds": {k: (c["points"], c.get("thickness_3cm_patches", {}).get("median_cm"),
                                                                                     c.get("share_within_delivered_lingbot")) for k, c in rec["clouds"].items()}},
                         default=fra.plain), flush=True)


@app.function(image=image, gpu="A100-80GB", cpu=8, memory=64 * 1024, volumes=VOLS, timeout=900, retries=0)
def debug_one_layer(path: str, frames: int = 60):
    """The one-layer pass's per-frame decisions on a saved shot (first `frames` LingBot frames)."""
    import torch
    from fast_report import lingbot_lane as ll
    y = np.load(path)
    dev = torch.device("cuda:0")
    d = torch.as_tensor(y["depth"][:frames].astype(np.float32), device=dev)
    c = torch.as_tensor(y["conf"][:frames].astype(np.float32), device=dev)
    cw = torch.as_tensor(y["c2w"][:frames], device=dev).float()
    K = torch.as_tensor(y["K"][:frames], device=dev).float()
    good = ll.usable(d, c, float(y["thr"])).reshape(frames, -1)
    tally = []
    fr, px = ll.one_layer_pass(d, K, cw, good, tally)
    return {"good": int(good.sum()), "survivors": int(len(fr)), "tally": tally[:12] + tally[-5:], "K0": y["K"][0].tolist(), "c2w0": y["c2w"][0].tolist(),
            "c2w1": y["c2w"][1].tolist(), "depth_median": float(d.median())}


@app.local_entrypoint()
def debug(path: str = "/v/x3/fx-x3-001/me340/s2/shot1.npz", frames: int = 60):
    print(json.dumps(debug_one_layer.remote(path, frames), indent=1))
