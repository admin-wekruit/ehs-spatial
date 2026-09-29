"""Fast report (docs/phase2/FAST-BUILD-SPEC.md): upload a 30 s video, the report's layers appear one by one.

One resident Modal container, 2 x A100-80GB, MPS on. boot() loads everything once (cold start, recorded, never part of
the analysis time); run() takes the MP4 bytes and yields every layer patch as the writer sends it, then run.json.
Analysis time = seconds from the MP4 bytes in the container to each layer written (Volume commit returned).
Core (A): video, cameras, room, people, events, objects, outlines; SAM 3D models on GPU 0 and the splat on GPU 1 (B);
patch store, local mirror and loopback endpoint (C, fast_report.layers); clock and memory (D, fast_report.instrument).

  modal run modal_apps/fast_report_app.py --video PATH --start S --end E --site NAME --out RUNS/fb-NNN \
      [--serve] [--eval-site me340|samsclub-a2|walmart] [--windows "S-E[:nocache|:site|:nodensify],S-E,..."] [--background-s 0]
      [--mirror-max-mb 8]
      # --serve: the viewer's endpoint on 127.0.0.1:8793 (web: npm run dev, then #/live/<report>), polled like the viewer;
      # --eval-site: the delivered splat's held-out frames stay out of training, and the quality table runs after the call;
      # windows of the same video share one boot (the first = the first call); nocache: no label cache; site: the
      # site's earlier words in wave 1
  modal run modal_apps/fast_report_app.py::setup      # image check + SigLIP 2 weights to the models volume (once)
  python modal_apps/fast_report_app.py --self-check    # CPU only: cuts, flood rules, naming, cascade, parsing
  python modal_apps/fast_report_app.py --evaluate RUN_DIR   # local numpy: poses vs DROID, naming vs the delivered names
"""
import json
import os
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import modal
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent if (HERE.parent / "fast_report").is_dir() else Path("/repo")  # the container mounts the repo at /repo
sys.path[:0] = [str(REPO), str(REPO / "scripts"), str(REPO / "modal_apps"), str(HERE)]
import sam3_app  # noqa: E402  SAM 3 revision pin

PHASE2 = Path("/Users/adam/Desktop/panoptes-public/research-notes/phase2")
DA3_CODE = "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4"
DA3_MODEL, DA3_REV = "depth-anything/DA3-GIANT-1.1", "72ee9f89ce4e50d704e9d55ee9c646ec8dc25a19"
PROCS, CPU, MEMORY_GIB, GPU, GATE_PROCS = 24, 32, 160, "A100-80GB:2", 16
VLLM_MPS = False  # vLLM outside MPS; this process, SAM 3D (E4) and the splat inside

app = modal.App("panoptes-fast-report")
VOLUMES = {"/v/da3": modal.Volume.from_name("moge3-hf-cache"), "/v/sam3": modal.Volume.from_name("sam3-hf-cache"),
           "/v/vlm": modal.Volume.from_name("panoptes-vlm-cache"),
           "/v/models": modal.Volume.from_name("panoptes-fb-models", create_if_missing=True),
           "/v/layers": modal.Volume.from_name("panoptes-fb-layers", create_if_missing=True),
           "/weights": modal.Volume.from_name("panoptes-sam3d-weights"),  # SAM 3D (sam3d_research's volume)
           "/ckpt": modal.Volume.from_name("panoptes-splat-train"),  # LPIPS' AlexNet for the splat's held-out score (torch hub cache)
           "/v/x13": modal.Volume.from_name("panoptes-x13-models"),  # r4/naming: DINOv2-L and PE-Core-L (x13's weights, read only)
           "/v/r4": modal.Volume.from_name("panoptes-r4-naming")}  # r4/naming: YOLOE-26L with the taxonomy baked in (modal_apps/r4_naming.py)


def build_image():
    """CUDA devel base (nvcc for SAM 3D's pytorch3d); E9's main environment, pinned to the versions E9 ran (torch
    2.14.0+cu130, transformers 5.17.0, open3d 0.19.0), vLLM in its own venv as in E9; then B's venvs (/opt/sam3d,
    /opt/gate, /opt/splat); the repo's code mounted at /repo (the SAM 3D, gate and splat processes run it from there)."""
    from fast_report import sam3d, splat
    base = (modal.Image.from_registry("nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04", add_python="3.11")
            .apt_install("git", "libgl1", "libglib2.0-0", "libgomp1")
            .pip_install("torch==2.14.0", "torchvision", "xformers", "transformers==5.17.0", "accelerate", "addict", "pillow", "scipy",
                         "open3d==0.19.0", "shapely", "pydantic", "opencv-python-headless", "sentencepiece",
                         f"git+https://github.com/ByteDance-Seed/Depth-Anything-3.git@{DA3_CODE}")
            .run_commands("python -m venv /opt/vllm && PIP_EXTRA_INDEX_URL= /opt/vllm/bin/pip install -q vllm==0.11.0 transformers==4.57.1 pillow"))
    out = splat.with_envs(sam3d.with_envs(base)).env({"HF_HUB_OFFLINE": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    # r4/naming: PE-Core-L (open_clip, x13's version) and YOLOE (Ultralytics, AGPL-3.0: accepted by the user for now); ultralytics
    # without its opencv-python dependency (the image has opencv-python-headless: two cv2 packages would overwrite each other)
    out = (out.pip_install("open_clip_torch==3.3.0", "matplotlib", "pyyaml", "requests", "psutil", "polars", "ultralytics-thop")
           .run_commands("python -m pip install --no-deps ultralytics==8.4.165"))
    for d in ("fast_report", "scripts", "modal_apps", "ehs_spatial"):
        out = out.add_local_dir(REPO / d, f"/repo/{d}", ignore=["**/__pycache__/**", "**/*.pyc"])
    return out


image = build_image() if modal.is_local() else modal.Image.debian_slim()  # a container runs the image it was built from


def gpu_listing():
    return subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.splitlines()


@app.function(image=image, gpu="A100-80GB", timeout=1200, retries=0, volumes=VOLUMES)
def setup():
    """The image imports and sees CUDA; r4/naming: the cascade's encoders load from their volumes and give the signals of a
    synthetic object, the bank loads (no download: x13's and r4_naming's volumes hold the weights)."""
    import open3d.core as o3c
    import torch
    import transformers
    from fast_report import cascade
    t = time.time()
    from depth_anything_3.api import DepthAnything3  # noqa: F401
    from transformers import Sam3Model  # noqa: F401
    from ehs_spatial.live_people import PeopleLoop  # noqa: F401
    from fast_report import segment
    segment.self_check()  # the flood rules need torch
    enc = cascade.Encoders(torch.device("cuda:0"))
    enc.warm()
    frames = torch.randint(0, 255, (3, 720, 1280, 3), dtype=torch.uint8, device="cuda:0")
    masks = torch.zeros((6, 280, 504), dtype=torch.bool, device="cuda:0")
    masks[:, 50:150, 100:300] = True
    masks[3:, 0:40, 0:60] = True
    sig = cascade.signals(enc, frames, [0, 1, 2, 0, 1, 2], masks, [0, 0, 0, 1, 1, 1], 2, lambda i: frames[i].cpu().numpy())
    bank = cascade.Bank()
    probs = bank.zero_shot(sig["pe"])
    vllm = subprocess.run(["/opt/vllm/bin/python", "-c", "import vllm, torch; print(vllm.__version__, torch.__version__)"], capture_output=True, text=True)
    out = {"torch": str(torch.__version__), "cuda": torch.cuda.is_available(), "transformers": transformers.__version__, "o3d_cuda": o3c.cuda.is_available(),
            "signals": sig["record"], "dino_norms": [float(np.linalg.norm(x)) for x in sig["dino"]], "zero_shot_top": [cascade.CLASSES[int(i)] for i in probs.argmax(1)],
            "bank_rows": len(bank.meta), "vram_gib": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2),
            "vllm": (vllm.stdout + vllm.stderr)[-300:], "gpus": gpu_listing(), "s": round(time.time() - t, 1)}
    print(json.dumps(out))
    return out


from fast_report.instrument import Clock, Vram, torch_peaks, usd_per_s  # noqa: E402  D
from fast_report.layers import Writer  # noqa: E402  C


@app.cls(image=image, gpu=GPU, cpu=CPU, memory=MEMORY_GIB * 1024, volumes=VOLUMES, timeout=3600, retries=0, max_containers=1,
         scaledown_window=60)
class FastReport:
    @modal.enter()
    def boot(self):
        import copy
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor
        from fast_report import core, sam3d, splat, vlm
        entered, t0 = time.time(), time.perf_counter()
        sys.setswitchinterval(1e-3)  # the decode thread gets the GIL back sooner from the two SAM 3 threads (E9: 1.7 -> 4.5 s)
        b = self.boot_record = {"entered_unix": entered}
        lap = lambda k: b.__setitem__(k, round(time.perf_counter() - t0, 2))  # noqa: E731
        self.listing = gpu_listing()
        env = {"CUDA_MPS_PIPE_DIRECTORY": "/tmp/mps-pipe", "CUDA_MPS_LOG_DIRECTORY": "/tmp/mps-log"}
        for d in env.values():
            Path(d).mkdir(parents=True, exist_ok=True)
        os.environ.update(env)
        started = subprocess.run(["nvidia-cuda-mps-control", "-d"], capture_output=True, text=True)  # before any CUDA context
        b["mps"] = started.returncode == 0
        self.vllm = vlm.start(1, mps=VLLM_MPS)  # first: its load overlaps everything below; GPU 1 stays empty until it has profiled
        b["vllm_mps"] = VLLM_MPS
        lap("vllm_spawned_s")
        self.sam3d = sam3d.Workers(gpu=0, n=2)  # B: two SAM 3D processes under MPS on GPU 0 (~60 s load + 16 s warm-up, beside vLLM's load)
        self.gate_pool = sam3d.GatePool(GATE_PROCS)  # the gate's prepare/assess processes (niced)
        self.proc_pool = ProcessPoolExecutor(PROCS, mp_context=multiprocessing.get_context("spawn"))
        self.proc_pool.map(core.warm_worker, range(PROCS))
        import torch
        import open3d  # noqa: F401
        import transformers
        from depth_anything_3.api import DepthAnything3
        from transformers import Sam3Model, Sam3Processor
        from fast_report import cascade, segment
        lap("imports_s")
        self.dev_geo, self.dev_seg = torch.device("cuda:0"), torch.device("cuda:1")
        da3 = DepthAnything3.from_pretrained(DA3_MODEL, revision=DA3_REV, cache_dir="/v/da3/huggingface/hub").eval()
        self.da3 = core.Da3(da3.to(self.dev_geo), self.dev_geo)
        self.da3.keep_host_copy()
        lap("da3_gpu0_s")
        proc = Sam3Processor.from_pretrained(sam3_app.MODEL_ID, revision=sam3_app.REVISION, cache_dir="/v/sam3/huggingface/hub")
        sam = Sam3Model.from_pretrained(sam3_app.MODEL_ID, revision=sam3_app.REVISION, cache_dir="/v/sam3/huggingface/hub",
                                        torch_dtype=torch.bfloat16).eval()
        sam0 = copy.deepcopy(sam).to(self.dev_geo)
        lap("sam3_gpu0_s")
        vlm.wait(self.vllm)
        lap("vllm_ready_s")
        self.namer_enc = cascade.Encoders(self.dev_seg)  # r4/naming: DINOv2-L, PE-Core-L, YOLOE on GPU 1, after vLLM sized its share
        lap("naming_encoders_gpu1_s")
        self.splat = splat.Worker(gpu=1, torch_home="/ckpt/torch")  # after vLLM sized its cache from GPU 1's free memory (B)
        b["sam3d"] = self.sam3d.ready()  # before this process warms up on GPU 0: SAM 3D's warm-up holds ~20 GB a process until it is done
        lap("sam3d_ready_s")
        self.sams = {self.dev_geo: segment.Sam3(sam0, proc, self.dev_geo), self.dev_seg: segment.Sam3(sam.to(self.dev_seg), proc, self.dev_seg)}
        self.cpu_pool, self.vlm_pool, self.run_pool = ThreadPoolExecutor(CPU), ThreadPoolExecutor(4), ThreadPoolExecutor(1)
        vllm_warm = self.cpu_pool.submit(self.warm_vllm)
        with torch.inference_mode():  # kernels, allocator, cuBLAS handles at the shapes the runs use
            self.da3.shot(torch.randint(0, 255, (16, 720, 1280, 3), dtype=torch.uint8, device=self.dev_geo))
            torch.cuda.synchronize(self.dev_geo)
            lap("warm_da3_s")
            filler = [f"word {i}" for i in range(58)]
            for d, s in self.sams.items():
                with torch.cuda.device(d):
                    noise = torch.randint(0, 255, (segment.PERSON_FRAMES, 720, 1280, 3), dtype=torch.uint8, device=d)
                    v = s.vision(noise)
                    s.detect(v, segment.PERSON_FRAMES, ("person", "floor"), segment.PERSON_SCORE, top=segment.PERSON_TOP)
                    s.detect(s.pick(v, [0, 3, 6]), 3, vlm.CORE, segment.VOCAB_SCORE, logits=True)
                    s.detect(s.pick(v, [0]), 1, filler, segment.VOCAB_SCORE, logits=True)
                    torch.cuda.synchronize(d)
            lap("warm_sam3_s")
            frames = torch.randint(0, 255, (4, 720, 1280, 3), dtype=torch.uint8, device=self.dev_geo)
            masks = torch.zeros((600, 280, 504), dtype=torch.bool, device=self.dev_geo)
            masks[:, 50:150, 100:300] = True
            cascade.signals(self.namer_enc, frames, np.arange(600) % 4, masks, np.arange(600) // 5, 120, lambda i: frames[i].cpu().numpy())
            lap("warm_naming_s")
        import m3_exp_geometry as geo
        geo.fuse(torch.full((2, 280, 504), 2., device=self.dev_geo), np.repeat(np.array([[[300., 0, 252], [0, 300, 140], [0, 0, 1]]]), 2, 0),
                 np.repeat(np.eye(4)[None], 2, 0), torch.full((2, 280, 504, 3), .5, device=self.dev_geo))
        lap("warm_open3d_s")
        b["vllm_warm_s"] = vllm_warm.result()
        b["process_pool_pids"] = len(set(self.proc_pool.map(core.warm_worker, range(PROCS))))
        b["splat"] = self.splat.ready()
        b["gate_processes"] = len(self.gate_pool.ready())
        lap("sam3d_splat_gate_ready_s")
        for d in (self.dev_geo, self.dev_seg):
            with torch.cuda.device(d):
                torch.cuda.empty_cache()
        b.update(ready_s=round(time.perf_counter() - t0, 2), ready_unix=time.time(), gpus=self.listing, torch=str(torch.__version__),
                 transformers=transformers.__version__, open3d=open3d.__version__,
                 resident_gb=[round((torch.cuda.mem_get_info(d)[1] - torch.cuda.mem_get_info(d)[0]) / 1e9, 2) for d in (self.dev_geo, self.dev_seg)])
        self.calls = 0

    def warm_vllm(self):
        """Requests shaped like the real ones: a 5-frame vocabulary list and two event windows (24 and 36 frames)."""
        import cv2
        from fast_report import vlm
        t = time.perf_counter()
        noise = cv2.imencode(".jpg", np.random.default_rng(0).integers(0, 255, (480, 640, 3), np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes()
        png = cv2.imencode(".png", np.random.default_rng(1).integers(0, 255, (480, 640, 3), np.uint8))[1].tobytes()
        with ThreadPoolExecutor(3) as pool:
            jobs = [pool.submit(vlm.chat, [vlm.image_block(png, "image/png")] * 5 + [{"type": "text", "text": "List objects."}], max_tokens=8)]
            jobs += [pool.submit(vlm.chat, [vlm.image_block(noise)] * k + [{"type": "text", "text": "Describe."}], max_tokens=8) for k in (24, 36)]
            [j.result() for j in jobs]
        # mvp2: the decider's own shape (two 336 / 448 px crops, one token, log-probs), 64 at MAX_SEQS in flight: the first call's
        # identity and judgement passes ran 1.2-1.4x slower than the warm call's (Sam's Club 24 / 31 s vs 20 / 22 s)
        crops = {side: cv2.imencode(".jpg", np.random.default_rng(side).integers(0, 255, (side, side, 3), np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()
                 for side in (336, 448)}
        p = vlm.qwen_prompt("These are crops.", "What is the object marked [1]?", ["box", "shelf", "another kind of object"])
        futs = [vlm.submit([crops[(336, 448)[i % 2]]] * 2, p, 3, "identity") for i in range(64)]
        [f.result() for f in futs]
        warm_s = round(time.perf_counter() - t, 2)
        t = time.perf_counter()  # idle decode speed, one sequence, 256 new tokens: the reference for the runs' vLLM numbers
        _, usage = vlm.chat([{"type": "text", "text": "Count from 1 to 500, separated by commas."}], max_tokens=256, ignore_eos=True)
        self.boot_record["vllm_idle_one_sequence_decode_per_s"] = round(usage["completion_tokens"] / (time.perf_counter() - t), 1)
        return warm_s

    @modal.exit()
    def stop(self):
        for name in ("sam3d", "gate_pool", "splat"):
            if getattr(self, name, None) is not None:
                getattr(self, name).close()
        if getattr(self, "vllm", None) is not None:
            self.vllm.terminate()

    @modal.method()
    def boot_info(self):
        return self.boot_record

    @modal.method()
    def run(self, mp4: bytes, site: str, report_id: str, options: dict):
        """MP4 bytes in -> every writer event as it happens -> run.json last."""
        import torch
        from fast_report import core
        clock = Clock()  # t0: the bytes are in the container
        vram = Vram([0, 1], source=options.get("vram_source", "auto")).start()  # whole-device memory, every process
        for d in (self.dev_geo, self.dev_seg):
            torch.cuda.reset_peak_memory_stats(d)
        self.calls += 1
        pool_note = None
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor
        if getattr(self.proc_pool, "_broken", False):  # mvp2/integrate: a worker died in the last call (Sam's Club 003, in cards.v1_box):
            self.proc_pool = ProcessPoolExecutor(PROCS, mp_context=multiprocessing.get_context("spawn"))  # every later call failed
            list(self.proc_pool.map(core.warm_worker, range(PROCS)))  # at its first submit; a new pool, warmed (not analysis time)
            pool_note = "the process pool broke in an earlier call and was recreated before this one"
        price = usd_per_s(2, CPU, MEMORY_GIB)
        writer = Writer(VOLUMES["/v/layers"], report_id, clock, root="/v/layers", client_has=options.get("client_has", ()),
                        report=lambda: clock.report(vram, price))
        asker = None
        if options.get("hazard_queues"):  # round 2: the hazard judge's Gemini questions go out through the CLI's relay
            from fast_report import hazard
            asker = hazard.Asker(*options["hazard_queues"])
        job = self.run_pool.submit(core.analyse, self, mp4, {**options, "site": site, "hazard_ask": asker.ask if asker else None}, clock, writer, None)
        job.add_done_callback(lambda _: writer.close())
        if asker is not None:
            job.add_done_callback(lambda _: asker.close())
        yield from writer.events()
        vram.stop()
        try:
            summary, error = job.result(), None
        except Exception:  # noqa: BLE001  reported, never retried
            summary, error = None, traceback.format_exc()[-4000:]
            if getattr(self, "release", None) is not None:
                self.release.set()  # a splat still holding for the SAM 3 queue trains and ends instead of waiting forever
        try:
            with clock.stage("da3.restore", gpu=self.dev_geo):  # after every layer (off the clock): the core offloaded DA3 for SAM 3D
                self.da3.restore()
        except Exception:  # noqa: BLE001  a dead CUDA context (a GPU fault) still returns this call's run.json
            error = (error or "") + "\nda3.restore failed: " + traceback.format_exc()[-1500:]
        run = {**clock.report(vram, price), "report": report_id, "site": site, "error": error, "process_pool": pool_note,
               "video": {"sha256": core.sha256(mp4), "bytes": len(mp4), "window_s": options.get("window_s")},
               "hardware": {"gpus": self.listing, "cpu": CPU, "memory_gib": MEMORY_GIB, "mps": self.boot_record.get("mps")},
               "boot": {**self.boot_record, "first_call_after_boot": self.calls == 1, "note": "cold start: never part of the analysis time"},
               "layers": writer.rows, "summary": summary, "main_process_torch_reserved_peak_gib": torch_peaks()}
        path = Path("/v/layers/reports") / report_id / "run.json"
        path.write_text(json.dumps(run, indent=1, default=plain))
        VOLUMES["/v/layers"].commit()
        try:
            for d in (self.dev_geo, self.dev_seg):
                with torch.cuda.device(d):
                    torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
        yield {"type": "run", "report": report_id, "run": run}


def plain(o):
    return o.tolist() if hasattr(o, "tolist") else str(o)


# ---------- local ----------

def cut(video, start, end, out_path, max_width=1280):
    """prepare_video_clip.full_video's rule: first frame round(S x fps), round((E - S) x fps) frames, avc1, <= 1280 wide;
    MP4 frame i is clip frame i. Not timed."""
    import cv2
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    first, count = int(round(start * fps)), int(round((end - start) * fps))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    s = min(1., max_width / w)
    size = (int(round(w * s)) // 2 * 2, int(round(h * s)) // 2 * 2)
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"avc1"), fps, size)
    assert writer.isOpened(), "H.264 encoder unavailable"
    written = 0
    while written < count:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(cv2.resize(frame, size, interpolation=cv2.INTER_AREA) if s < 1 else frame)
        written += 1
    cap.release(), writer.release()
    assert written == count, (written, count)
    return {"first_frame": first, "frames": written, "fps": fps, "size": size}


MILESTONES = {  # name -> (layer, which version): the report's moments, each at its patch's written time
    "cameras": ("cameras", lambda d: True), "first_3d": ("room", lambda d: True), "room_full": ("room", lambda d: d.get("kind") == "full"),
    "people": ("people", lambda d: True),
    "events": ("events", lambda d: True), "objects": ("objects", lambda d: True), "outlines": ("outlines", lambda d: True),
    "first_model": ("models", lambda d: bool(d.get("models"))), "all_models": ("models", lambda d: d.get("final")),
    "splat_preview": ("splat", lambda d: d.get("kind") == "preview"),
    "pick": ("pick", lambda d: True), "cards_v1": ("object_cards", lambda d: d.get("version") == 1),
    "cards_v2": ("object_cards", lambda d: d.get("version") in (2, 4)), "judgements_v1": ("judgements", lambda d: not d.get("vlm_answers")),
    "judgements_v2": ("judgements", lambda d: bool(d.get("vlm_answers"))),
    "outlines_v2": ("outlines", lambda d: bool(d.get("densified"))), "pick_v2": ("pick", lambda d: "version_note" in d),
    "objects_v3": ("objects", lambda d: "densify" in d), "cards_v3": ("object_cards", lambda d: d.get("version") == 3),
    "judgements_v3": ("judgements", lambda d: bool(d.get("vlm_answers")) and (d.get("version_of") or {}).get("object_cards") == 3)}


def milestones(root, report, t0_unix):
    """Per milestone: seq, sent_s and written_s (container clock from t0), served_s (the endpoint's first fetch: this machine's
    clock minus t0_unix, two clocks)."""
    folder = Path(root) / "reports" / report
    written = json.loads((folder / "written.json").read_text()) if (folder / "written.json").exists() else {}
    served = json.loads((folder / "served.json").read_text()) if (folder / "served.json").exists() else {}
    out = {}
    for path in sorted((folder / "patches").glob("*.json")):
        patch = json.loads(path.read_text())
        for name, (layer, want) in MILESTONES.items():
            if name not in out and patch["layer"] == layer and want(patch["data"] or {}):
                seq = str(patch["seq"])
                out[name] = {"seq": patch["seq"], "version": patch["version"], "sent_s": patch["sent_s"], "written_s": written.get(seq),
                             "served_s": round(served[seq] - t0_unix, 3) if seq in served else None}
    return out


def poll_like_the_viewer(report, stop, port=8793):
    """GET the new patches every 0.5 s, as web/src/live-report.ts does: the endpoint records each patch's first fetch."""
    import urllib.request
    after = 0
    while not stop.is_set():
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/fast/reports/{report}/patches?after={after}", timeout=5) as r:
                after = max([after] + [p["seq"] for p in json.loads(r.read())["patches"]])
        except OSError:
            pass
        stop.wait(.5)


@app.local_entrypoint()
def main(video: str, start: float = 0., end: float = 0., site: str = "site", out: str = "", windows: str = "", vocab: str = "qwen",
         serve: bool = False, eval_site: str = "", splat_preview_s: float = 0., background_s: float = 0., vram_source: str = "auto",
         mirror_max_mb: float = 0.):
    import hashlib
    import threading
    from fast_report import layers
    assert vocab == "qwen", "only the Qwen vocabulary is wired (fast_report/vlm.py docstring)"
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)  # never reuse a run folder
    plan = []
    for w in (windows or f"{start:g}-{end:g}").split(","):
        span, _, flag = w.partition(":")
        a, b = (float(x) for x in span.split("-"))
        path = out / f"input-{a:g}-{b:g}.mp4"
        if not path.exists():
            clip = cut(video, a, b, path)
            (out / f"input-{a:g}-{b:g}.json").write_text(json.dumps({"video": video, "start_s": a, "end_s": b, **clip}, indent=1))
        plan.append((a, b, flag, path.read_bytes()))
    ev = None
    if eval_site:
        sys.path.append(str(REPO / "scripts"))
        import fast_report_eval as ev  # D: the quality table (CPU rows) against the delivered report
    if serve:
        layers.serve(out, 8793)  # loopback only; web/vite.config.ts proxies /fast here
    fr = FastReport()
    submitted = time.time()
    boot = fr.boot_info.remote()
    boot.update(client_submitted_unix=submitted, submit_to_ready_s_two_clocks=round(boot["ready_unix"] - submitted, 1))
    (out / "boot.json").write_text(json.dumps(boot, indent=1, default=plain))
    print("ready:", json.dumps({k: v for k, v in boot.items() if k.endswith("_s") or k in ("mps", "gpus", "resident_gb")}), flush=True)
    rows = []
    for i, (a, b, flag, mp4) in enumerate(plan):
        use_cache = flag != "nocache"
        digest = hashlib.sha256(mp4).hexdigest()
        report_id = f"fb-{site}-{digest[:8]}-{int(time.time())}"
        layers.put_blob(out, mp4)  # the client holds its own MP4: it is never sent back
        options = {"cache": use_cache, "site_vocab": flag == "site", "window_s": [a, b], "client_has": [digest], "background_s": background_s,
                   "densify": flag != "nodensify",
                   "vram_source": vram_source,
                   "splat_preview_s": splat_preview_s or None, "eval_holdout": ev.holdout_frames(eval_site) if ev else None}
        if serve:
            print(f"viewer: http://127.0.0.1:5173/app.html#/live/{report_id}", flush=True)
        stop = threading.Event()
        poller = threading.Thread(target=poll_like_the_viewer, args=(report_id, stop), daemon=True)
        if serve:
            poller.start()
        called, received, errors, run = time.time(), {}, [], None
        for e in fr.run.remote_gen(mp4, site, report_id, options):
            now = time.time()
            if e["type"] in ("patch", "written", "run"):
                layers.mirror(e, out, int(mirror_max_mb * 1e6) if mirror_max_mb else None)
            if e["type"] == "patch":
                received[e["patch"]["seq"]] = now
                print(f"  [{i}] {e['patch']['layer']} v{e['patch']['version']}: sent {e['patch']['sent_s']} s", flush=True)
            elif e["type"] == "written":
                print(f"  [{i}] written {e['written_s']} s (seqs {e['seqs']}, commit {e['commit_s']} s)", flush=True)
            elif e["type"] == "error":
                errors.append(e)
                print("  writer error:", json.dumps(e)[:1500], flush=True)
            elif e["type"] == "run":
                run = e["run"]
        time.sleep(1.5 if serve else 0)  # one more poll after the last patch
        stop.set()
        for r in run["layers"]:
            r["received_s_two_clocks"] = round(received[r["seq"]] - run["t0_unix"], 3) if r["seq"] in received else None
        run.update(client={"called_unix": called, "returned_unix": time.time(), "wall_s": round(time.time() - called, 2), "cache": use_cache,
                           "first_call": i == 0, "upload_and_dispatch_s_two_clocks": round(run["t0_unix"] - called, 3), "writer_errors": errors},
                   milestones=milestones(out, report_id, run["t0_unix"]))
        if ev is not None:
            try:
                q = ev.evaluate(ev.load_layers(out, report_id), eval_site)
                run["quality"] = {"verdict": q["verdict"], "rows": q["rows"], "not_scored": q["not_scored"]}
            except Exception:  # noqa: BLE001  a quality failure must not lose the timing
                run["quality"] = {"error": traceback.format_exc()[-2000:]}
        layers._write_json(out / "reports" / report_id / "run.json", run)
        rows.append({"report": report_id, "window_s": [a, b], "cache": use_cache, "first_call": i == 0, "error": run["error"],
                     "milestones": {k: v["written_s"] for k, v in run["milestones"].items()}, "flags": run["flags"],
                     "gpu_peak_gib": [g["peak_gb"] for g in run["gpu_peak"]], "gpus": run["hardware"]["gpus"],
                     "quality_verdict": (run.get("quality") or {}).get("verdict"), "usd_estimate": run["usd_estimate"]})
        print(json.dumps(rows[-1], default=plain)[:3000], flush=True)
        if run["error"]:
            print(run["error"][-3000:], flush=True)
    (out / "summary.json").write_text(json.dumps({"boot": boot, "runs": rows}, indent=1, default=plain))


@app.local_entrypoint()
def accuracy(plan: str, out: str, mirror_max_mb: float = 8.):
    """mvp2 accuracy (scripts/accuracy_gt.py): ground-truth sequence windows through one container, in the order of `plan`
    (a JSON list of {"mp4": path, "site": name, "window_s": [a, b], "options": {...}}); the first call is the first call
    after boot. Options add cuts / geometry / poses / display to the core's; no label cache between videos."""
    import hashlib
    from fast_report import layers
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    calls = json.loads(Path(plan).read_text())
    fr = FastReport()
    submitted = time.time()
    boot = fr.boot_info.remote()
    boot.update(client_submitted_unix=submitted, submit_to_ready_s_two_clocks=round(boot["ready_unix"] - submitted, 1))
    (out / "boot.json").write_text(json.dumps(boot, indent=1, default=plain))
    rows = []
    for i, c in enumerate(calls):
        mp4 = Path(c["mp4"]).read_bytes()
        digest = hashlib.sha256(mp4).hexdigest()
        report_id = f"acc-{c['site']}-{c['options'].get('label', c['options'].get('geometry', 'shot'))}-{digest[:8]}-{int(time.time())}"
        layers.put_blob(out, mp4)
        options = {"cache": False, "window_s": c["window_s"], "client_has": [digest], "display": False, **c["options"]}
        run = None
        for e in fr.run.remote_gen(mp4, c["site"], report_id, options):
            if e["type"] in ("patch", "written", "run"):
                layers.mirror(e, out, int(mirror_max_mb * 1e6) if mirror_max_mb else None)
            if e["type"] == "run":
                run = e["run"]
            elif e["type"] == "error":
                print("  writer error:", json.dumps(e)[:1500], flush=True)
        run.update(first_call=i == 0, call=c, milestones=milestones(out, report_id, run["t0_unix"]))
        layers._write_json(out / "reports" / report_id / "run.json", run)
        rows.append({"report": report_id, "site": c["site"], "geometry": options.get("label", options.get("geometry", "shot")), "first_call": i == 0, "error": run["error"],
                     "milestones": {k: v["written_s"] for k, v in run["milestones"].items()}, "flags": run["flags"],
                     "gpu_peak_gib": [g["peak_gb"] for g in run["gpu_peak"]], "usd_estimate": run["usd_estimate"]})
        print(json.dumps(rows[-1], default=plain)[:2000], flush=True)
        if run["error"]:
            print(run["error"][-3000:], flush=True)
        (out / "summary.json").write_text(json.dumps({"boot": boot, "runs": rows}, indent=1, default=plain))


# ---------- evaluation (local numpy) ----------

def patches(report_dir):
    """latest patch per layer, plus every version of objects."""
    out, versions = {}, {}
    for p in sorted((report_dir / "patches").glob("*.json")):
        x = json.loads(p.read_text())
        out[x["layer"]] = x
        versions.setdefault(x["layer"], []).append(x)
    return out, versions


def norm_name(s):
    words = [w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w for w in (s or "").lower().replace("-", " ").split()]
    return " ".join(words)


def name_match(ours, theirs):
    """The spec's rule (section 10): the same head noun (last word), or one name inside the other as whole words."""
    a, b = norm_name(ours), norm_name(theirs)
    if not a or not b:
        return False
    return a.split()[-1] == b.split()[-1] or f" {a} " in f" {b} " or f" {b} " in f" {a} "


def cascade_label(o):
    """The cascade's name: in 'cascade' since run 009, in 'label' before (runs 001-008 displayed it); SAM 3's word when
    the cascade gave none."""
    c = o.get("cascade", {})
    if "source" in c:
        return c.get("label") or o["word"]
    return o.get("label") or o["word"]


def cascade_source(o):
    c = o.get("cascade", {})
    return (c.get("source") if "source" in c else o.get("label_source", "?")).split(" ")[0]


def evaluate(run_dir):
    """Per report of ME340 165-195 s in RUN_DIR: walk-shot ATE vs DROID (E9's rule), and naming vs the delivered names:
    delivered named objects (clear/partial) matched to our objects by mask IoU >= 0.5 on shared frames (our segmented
    outline frames vs today's masks on the nearest 10 fps frame, <= 1 frame away), then the name rule, for the SAM 3
    vote word, SigLIP zero-shot top-1 and the cascade's final name, by cascade source."""
    import cv2
    sys.path.insert(0, str(HERE))
    import m3_exp_geometry as geo
    runs = PHASE2 / "runs"
    droid = np.load(runs / "droid-me340-165-171/prediction.npz")["poses_c2w"].astype(np.float64)
    mpn = json.loads((runs / "da3-posed-me340-223-shotc/metric-scale.json").read_text())["metres_per_native_unit"]
    names = json.loads((runs / "me340-entity-names-200/names.json").read_text())
    omap = json.loads((runs / "me340-entity-names-200/object-map.json").read_text())
    masks_dir = runs / "me340-masks-194/object-a"
    have = {int(p.name.split("-")[1]) for p in masks_dir.iterdir() if p.name.startswith("frame-")}
    obs_by_frame = {}
    for e in omap["entities"]:
        spec = names.get(e["entityId"])
        if not spec or spec["status"] not in ("clear", "partial"):
            continue
        for o in e["observations"]:
            kind, f, inst = o.split(":")
            obs_by_frame.setdefault(int(f), []).append((e["entityId"], int(inst)))
    report = {}
    base = Path(run_dir) / "reports" if (Path(run_dir) / "reports").is_dir() else Path(run_dir)  # layers.mirror's layout, or the stub's
    for rdir in sorted(p for p in base.iterdir() if (p / "patches").is_dir()):
        runj = json.loads((rdir / "run.json").read_text()) if (rdir / "run.json").exists() else {}
        if (runj.get("video", {}).get("window_s") or [None])[0] != 165:
            continue
        latest, versions = patches(rdir)
        row = {}
        cams = latest["cameras"]["data"]["shots"]
        walk = next(s for s in cams if s["frames"][0] <= 500 <= s["frames"][1])
        ours = np.array(walk["c2w"], np.float64)
        walk_keys = walk.get("keyframes") or walk["keys"]
        target = droid[walk_keys].copy()
        target[:, :3, 3] *= mpn
        s, R, t = geo.align_sim3(ours, target)
        err = np.linalg.norm((s * (R @ ours[:, :3, 3].T)).T + t - target[:, :3, 3], axis=1)
        row["walk_shot"] = {"ate_m": round(float(np.sqrt((err ** 2).mean())), 4), "our_metres_over_reference": round(1 / s, 4),
                            "path_m_reference": round(float(np.linalg.norm(np.diff(target[:, :3, 3], axis=0), axis=1).sum()), 3)}
        objs = {o["id"]: o for o in latest["objects"]["data"]["objects"]}
        blob = latest["outlines"]["blobs"]["analysis"]["sha256"]
        analysis = json.loads(next(p for p in (Path(run_dir) / "blobs/sha256" / blob, rdir / "blobs" / blob) if p.exists()).read_text())
        best = {}  # entity -> (iou, our object id)
        pairs = 0
        for fr in analysis["frames"]:
            if fr["source"] != "segmented":
                continue
            g = min(have, key=lambda x: abs(x - fr["sourceFrame"]))
            if abs(g - fr["sourceFrame"]) > 1 or g not in obs_by_frame:
                continue
            ours_m = {}
            for o in fr["objects"]:
                m = np.zeros((480, 640), np.uint8)
                for poly in o["polygons"]:
                    p = np.array(poly, np.float64)
                    p = np.stack([(p[:, 0] - 160) * 640 / 960, p[:, 1] * 480 / 720], 1)
                    cv2.fillPoly(m, [np.round(p).astype(np.int32)], 1)
                ours_m[o["entityId"]] = m.astype(bool)
            for ent, inst in obs_by_frame[g]:
                path = masks_dir / f"frame-{g:05d}" / f"instance-{inst}-mask.png"
                if not path.exists():
                    continue
                theirs = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) > 0
                if theirs.sum() < 300:
                    continue
                pairs += 1
                for oid, m in ours_m.items():
                    iou = (m & theirs).sum() / max((m | theirs).sum(), 1)
                    if iou > best.get(ent, (0, None))[0]:
                        best[ent] = (iou, oid)
        matched = {e: oid for e, (iou, oid) in best.items() if iou >= .5 and oid in objs}
        # what the namer saw: each object's best view (the VLM crop's frame), matched there to one delivered object
        seen_pairs = {}
        seg_frames = {fr["sourceFrame"]: fr for fr in analysis["frames"] if fr["source"] == "segmented"}
        for oid, o in objs.items():
            fr = seg_frames.get(o["best_key"])
            g = min(have, key=lambda x: abs(x - o["best_key"]))
            if fr is None or abs(g - o["best_key"]) > 1 or g not in obs_by_frame:
                continue
            m = np.zeros((480, 640), np.uint8)
            for x in fr["objects"]:
                if x["entityId"] == oid:
                    for poly in x["polygons"]:
                        pp = np.array(poly, np.float64)
                        cv2.fillPoly(m, [np.round(np.stack([(pp[:, 0] - 160) * 640 / 960, pp[:, 1] * 480 / 720], 1)).astype(np.int32)], 1)
            m = m.astype(bool)
            if not m.any():
                continue
            top = (0, None)
            for ent, inst in obs_by_frame[g]:
                path = masks_dir / f"frame-{g:05d}" / f"instance-{inst}-mask.png"
                if path.exists():
                    theirs = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) > 0
                    iou = (m & theirs).sum() / max((m | theirs).sum(), 1)
                    top = max(top, (iou, ent), key=lambda x: x[0])
            if top[0] >= .5:
                seen_pairs[oid] = top[1]
        tally = {"sam3_word": [], "zero_shot_top1": [], "cascade_final": []}
        by_source = {}
        detail = []
        for ent, oid in matched.items():
            o, theirs = objs[oid], names[ent]["category"]
            zs = (o.get("cascade", {}).get("zero_shot_top3") or [[None]])[0][0]
            ok = {"sam3_word": name_match(o["word"], theirs), "zero_shot_top1": name_match(zs, theirs),
                  "cascade_final": name_match(cascade_label(o), theirs)}
            for k, v in ok.items():
                tally[k].append(v)
            by_source.setdefault(cascade_source(o), []).append(ok["cascade_final"])
            detail.append({"entity": ent, "delivered": theirs, "object": oid, "word": o["word"], "zero_shot": zs, "cascade_label": cascade_label(o),
                           "source": cascade_source(o), "iou": round(best[ent][0], 3), **ok})
        from fast_report import segment
        vocab_words = latest["objects"]["data"].get("vocabulary") or latest["objects"]["data"].get("words", [])
        bv = {"sam3_word": [], "zero_shot_top1": [], "zero_shot_context_top1": [], "cascade_final": [], "displayed_label": [],
              "what_if_vlm_only_for_generic_sam3_words": []}
        bv_source = {}
        for oid, ent in seen_pairs.items():
            o, theirs = objs[oid], names[ent]["category"]
            c = o.get("cascade", {})
            ok = {"sam3_word": name_match(o["word"], theirs), "zero_shot_top1": name_match((c.get("zero_shot_top3") or [[None]])[0][0], theirs),
                  "zero_shot_context_top1": name_match((c.get("zero_shot_context_top3") or [[None]])[0][0], theirs),
                  "cascade_final": name_match(cascade_label(o), theirs), "displayed_label": name_match(o.get("label"), theirs),
                  "what_if_vlm_only_for_generic_sam3_words": name_match(cascade_label(o) if segment.is_generic(o["word"], vocab_words) else o["word"], theirs)}
            for k, v in ok.items():
                bv[k].append(v)
            bv_source.setdefault(cascade_source(o), []).append(ok["cascade_final"])
        row["naming_best_view"] = {"pairs": len(seen_pairs), "rule": "each object's best view (the VLM crop's frame), IoU >= 0.5 with one delivered object there",
                                   "accuracy": {k: {"correct": int(sum(v)), "of": len(v), "share": round(sum(v) / max(len(v), 1), 3)} for k, v in bv.items()},
                                   "final_by_source": {k: {"correct": int(sum(v)), "of": len(v)} for k, v in bv_source.items()},
                                   "pairs_detail": [{"object": oid, "delivered": names[e]["category"], "cascade_label": cascade_label(objs[oid]),
                                                     "word": objs[oid]["word"], "source": cascade_source(objs[oid])} for oid, e in seen_pairs.items()]}
        row["naming"] = {"delivered_named_on_our_frames": len({e for e in best}), "matched_iou_0.5": len(matched), "observation_pairs": pairs,
                         "accuracy": {k: {"correct": int(sum(v)), "of": len(v), "share": round(sum(v) / max(len(v), 1), 3)} for k, v in tally.items()},
                         "final_by_source": {k: {"correct": int(sum(v)), "of": len(v)} for k, v in by_source.items()},
                         "rule": "head noun equal, or one name inside the other (FAST-BUILD-SPEC section 10)"}
        casc = latest["objects"]["data"].get("cascade", {})
        n_obj = max(casc.get("objects", 0), 1)
        row["cascade"] = {"objects": casc.get("objects"), "cross_video_hit_rate": round(casc.get("cross_video_hits", 0) / n_obj, 3),
                          "in_video_hit_rate": round(casc.get("in_video_hits", 0) / n_obj, 3),
                          "zero_shot_accept_rate": round(casc.get("zero_shot_accepted", 0) / n_obj, 3),
                          "escalated_share": round(casc.get("uncertain", 0) / n_obj, 3),
                          "vlm_crops_after_clustering": casc.get("vlm_requests_objects"), "vlm_s": (casc.get("vlm") or {}).get("s")}
        # the accuracy / escalation trade-off, read off the saved probabilities (no rerun): zero-shot accepted at (p, margin)
        sweep = []
        for p_min in (.3, .4, .5, .6, .7, .8):
            for margin in (0., .1, .25, .4):
                acc = []
                for ent, oid in matched.items():
                    top = objs[oid].get("cascade", {}).get("zero_shot_top3") or []
                    if top and top[0][1] >= p_min and top[0][1] - (top[1][1] if len(top) > 1 else 0) >= margin:
                        acc.append(name_match(top[0][0], names[ent]["category"]))
                all_top = [o.get("cascade", {}).get("zero_shot_top3") or [] for o in objs.values()]
                share = np.mean([bool(t) and t[0][1] >= p_min and t[0][1] - (t[1][1] if len(t) > 1 else 0) >= margin for t in all_top]) if all_top else 0
                sweep.append({"p_min": p_min, "margin": margin, "accepted_share": round(float(share), 3),
                              "accepted_correct": f"{sum(acc)}/{len(acc)}"})
        row["zero_shot_sweep"] = sweep
        row["detail"] = detail
        report[rdir.name] = row
    (Path(run_dir) / "evaluation.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: {kk: (vv if kk != "naming_best_view" else {a: b for a, b in vv.items() if a != "pairs_detail"})
                          for kk, vv in v.items() if kk not in ("detail", "zero_shot_sweep")} for k, v in report.items()}, indent=1))


def self_check():
    from fast_report import cascade, core, segment, vlm
    core.self_check()
    segment.self_check()
    cascade.self_check()
    vlm.self_check()
    assert name_match("cabinet", "tool cabinet") and name_match("lathes", "lathe") and name_match("drill press", "drill")
    assert not name_match("cabinet", "lathe") and not name_match(None, "lathe")
    print("fast_report self-check ok")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--evaluate"]:
        evaluate(sys.argv[2])
    else:
        assert sys.argv[1:] == ["--self-check"], __doc__
        self_check()
