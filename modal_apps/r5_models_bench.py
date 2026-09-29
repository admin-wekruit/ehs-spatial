"""r5 (models): which model every object card gets, measured on the r5 report dumps (fast_report.fixture, runs of
fast_report_app.py::accuracy with options dump) of ME340, Sam's Club and Walmart. Three arms, each one container, every model
resident from boot (loads and warm-ups are cold start, never analysis time); t0 = the dump staged in the container's memory:
  cpu         32 cores, the gate venv x24 (the report container's process count): every card's (a) observed-surface mesh and
              (d) planar parts, timed; X7's view selection (a held-out view >= 15 deg from <= 4 generation views; 8 deg where
              no 15 deg set exists, flagged); the held-out test of (a), of the r4 primitive refitted on the same points and of (c)
              the splat cut by the card's box; tiles
  internal    2 x A100-80GB, MPS: RecGen x4 (2 a GPU; TRI non-commercial licence: the internal profile only), X7's recipe:
              generate_multiview on the generation views with depth + mask + K, bounded placement, the held-out gate
  commercial  2 x A100-80GB, MPS: SAM 3D s1cfg12 x2, TRELLIS-image-large x2 (multi-image), TripoSR x2 (one of each a GPU)
Every accepted generated model is also measured the workcell's way (scene_measurements.fitted_plane on the placed surface
next to each observed planar part) against (d).

  M=/Users/adam/Desktop/Tesla/panoptes-platform/.venv/bin/modal
  $M run modal_apps/r5_models_bench.py::prep                                    # TripoSR weights to panoptes-r5-models, image check
  $M run modal_apps/r5_models_bench.py --out RUNS/r5-models-bench-NNN [--names a,b] [--arms cpu,internal,commercial] [--top 70]
  python modal_apps/r5_models_bench.py --self-check
"""
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import time

import modal
import numpy as np

WT = Path(__file__).resolve().parents[1]
LOCAL = modal.is_local()
sys.path[:0] = [str(WT), str(WT / "modal_apps"), str(WT / "scripts")] if LOCAL else ["/repo", "/repo/scripts", "/repo/modal_apps"]
DUMP = "r5-models-001"
NAMES = ("me340", "samsclub-a2", "walmart")
CPU, CPU_PROCS, GPU_CPU_PROCS = 32, 24, 16
PRICE = {"A100-80GB": .000694, "cpu": .0000131, "gib": .00000222}
RECGEN_GPUS, SAM3D_GPUS, TRELLIS_GPUS, TRIPOSR_GPUS = (0, 0, 1, 1), (0, 1), (0, 1), (0, 1)

app = modal.App("panoptes-r5-models-bench")
r5_volume = modal.Volume.from_name("panoptes-r5-models", create_if_missing=True)
VOLS = {"/weights": modal.Volume.from_name("panoptes-sam3d-weights"), "/cache": modal.Volume.from_name("panoptes-lucida-weights"),
        "/v/layers": modal.Volume.from_name("panoptes-fb-layers"), "/v/r5": r5_volume}
if LOCAL:
    from fast_report import gen3d, sam3d as fb_sam3d, x7 as fx7
    base = (modal.Image.from_registry("nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04", add_python="3.11")  # X7's base: its layers are cached
            .pip_install("numpy==2.2.6", "opencv-python-headless==4.11.0.86", "scipy==1.15.3", "pydantic==2.11.7", "trimesh==4.6.10"))
    image = gen3d.with_generators(fx7.with_recgen(fb_sam3d.with_envs(base)))
    for d in ("fast_report", "scripts", "modal_apps", "ehs_spatial"):
        image = image.add_local_dir(WT / d, f"/repo/{d}", ignore=["**/__pycache__/**", "**/*.pyc"])
    cpu_image = fb_sam3d.with_envs(base)
    for d in ("fast_report", "scripts", "modal_apps", "ehs_spatial"):
        cpu_image = cpu_image.add_local_dir(WT / d, f"/repo/{d}", ignore=["**/__pycache__/**", "**/*.pyc"])
else:
    image = cpu_image = modal.Image.debian_slim()


def stats(values):
    v = np.asarray([x for x in values if x is not None], float)
    if not len(v):
        return None
    return {"n": int(len(v)), "median": round(float(np.median(v)), 3), "p90": round(float(np.percentile(v, 90)), 3), "max": round(float(v.max()), 3),
            "sum": round(float(v.sum()), 2)}


def load_dump(name):
    """The dump, staged the SAM 3D gate's way (frames in /dev/shm), and the latest splat of its report as a small npz."""
    from fast_report import fixture, sam3d
    folder = Path("/v/layers/r5") / DUMP / name
    fx = fixture.load(folder)
    meta = fx["meta"]
    frames = sam3d.frames_buffer(meta["n_frames"], name=f"r5-{name}-frames.npy")
    for f, img in zip(fx["z"]["frame_ids"], fx["z"]["frames"]):
        frames[int(f)] = img
    frames.flush()
    spec, keys = sam3d.stage(fx["objs"], fx["shots"], str(frames.filename))
    splat = None
    pdir = Path("/v/layers/reports") / meta["report"] / "patches"
    ps = sorted(pdir.glob("*-splat.json")) if pdir.is_dir() else []
    if ps:
        import splat_to_web
        p = json.loads(ps[-1].read_text())
        blob = Path("/v/layers/blobs/sha256") / p["blobs"]["splat"]["sha256"]
        if blob.exists():
            u = splat_to_web.unpack(blob.read_bytes())
            splat = f"/dev/shm/r5-{name}-splat.npz"
            np.savez(splat, shot=p["data"]["shot"], **u)
    return {"fx": fx, "folder": str(folder), "spec": spec, "keys": keys, "splat": splat, "frames": frames}


def boot_pools(self, arm):
    from fast_report import gen3d, sam3d, x7
    env = {"CUDA_MPS_PIPE_DIRECTORY": "/tmp/mps-pipe", "CUDA_MPS_LOG_DIRECTORY": "/tmp/mps-log"}
    for d in env.values():
        Path(d).mkdir(parents=True, exist_ok=True)
    os.environ.update(env)
    rec = {"entered_unix": time.time(), "arm": arm}
    if arm != "cpu":
        rec["mps"] = subprocess.run(["nvidia-cuda-mps-control", "-d"], capture_output=True, text=True).returncode == 0
    cpu_env = sam3d.worker_env(sam3d.GATE_PY, OMP_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1, MKL_NUM_THREADS=1)
    pools = {"cpu": sam3d.Pool([sam3d.GATE_PY, "-c", "from fast_report.r5_bench import cpu_worker; cpu_worker()"],
                               CPU_PROCS if arm == "cpu" else GPU_CPU_PROCS, cpu_env, "cpu")}
    if arm == "internal":
        rg_env = []
        for g in RECGEN_GPUS:
            e = sam3d.worker_env(x7.RECGEN_PY, CUDA_VISIBLE_DEVICES=g, ATTN_BACKEND="xformers", SPCONV_ALGO="native", HF_HUB_OFFLINE=1)
            e["PYTHONPATH"] += os.pathsep + x7.RECGEN_DIR
            rg_env.append(e)
        pools["recgen"] = sam3d.Pool([x7.RECGEN_PY, "-c", "from fast_report.x7 import recgen_worker; recgen_worker()"], len(RECGEN_GPUS), rg_env, "recgen")
    elif arm == "commercial":
        import complete_video_objects as cvo
        sam_env = [sam3d.worker_env(sam3d.SAM3D_PY, CUDA_VISIBLE_DEVICES=g, HF_HOME=f"{sam3d.WEIGHTS}/huggingface", HF_HUB_OFFLINE=1, LIDRA_SKIP_INIT="true",
                                    TORCH_HOME=sam3d.TORCH_HUB, CUDA_HOME="/usr/local/cuda", SAM3D_MODEL_REVISION=cvo.SAM3D["modelRevision"]) for g in SAM3D_GPUS]
        pools["sam3d"] = sam3d.Pool([sam3d.SAM3D_PY, "-c", "from fast_report.sam3d import sam3d_worker; sam3d_worker()"], len(SAM3D_GPUS), sam_env, "sam3d")
        tr_env = [sam3d.worker_env(gen3d.TRELLIS_PY, CUDA_VISIBLE_DEVICES=g, ATTN_BACKEND="xformers", SPCONV_ALGO="native", HF_HUB_OFFLINE=1)
                  for g in TRELLIS_GPUS]
        pools["trellis"] = sam3d.Pool([gen3d.TRELLIS_PY, "-c", "from fast_report.gen3d import worker; worker('trellis')"], len(TRELLIS_GPUS), tr_env, "trellis")
        ts_env = [sam3d.worker_env(gen3d.TRIPOSR_PY, CUDA_VISIBLE_DEVICES=g, HF_HOME=gen3d.HF_HOME, HF_HUB_OFFLINE=1) for g in TRIPOSR_GPUS]
        pools["triposr"] = sam3d.Pool([gen3d.TRIPOSR_PY, "-c", "from fast_report.gen3d import worker; worker('triposr')"], len(TRIPOSR_GPUS), ts_env, "triposr")
    t = time.time()
    for k, p in pools.items():
        try:
            rec[k] = p.ready(1200)
        except Exception as error:  # noqa: BLE001  a dead pool is reported, the others still run
            rec[k] = {"error": str(error)[-3000:]}
        rec[f"{k}_ready_s"] = round(time.time() - t, 1)
    rec["ready_s"] = round(time.time() - rec["entered_unix"], 1)
    rec["note"] = "cold start: model loads and warm-up calls, never part of the analysis time"
    self.pools, self.boot_record = pools, rec


def events_loop(pending, events, handle):
    while pending[0]:
        kind, r, method, fut = events.get()
        pending[0] -= 1
        try:
            res = fut.result()
        except Exception as error:  # noqa: BLE001  one object's failure is recorded, the rest go on
            res = {"error": str(error)[-1500:]}
        handle(kind, r, method, res)


def plumb_u(fx, shot):
    sh = next((s for s in fx["meta"]["cards_shots"] if s["index"] == shot), {})
    pw = sh.get("plumb_walls") or {}
    return pw.get("p90", sh.get("plumb_deg"))


@app.function(image=cpu_image, cpu=4, memory=8192, volumes={"/v/r5": r5_volume}, timeout=1800, retries=0)
def setup():
    """TripoSR (MIT) and its DINO tokenizer (Apache-2.0) into the panoptes-r5-models volume (the TripoSR venv reads them offline)."""
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub==0.24.7"], check=True)
    from huggingface_hub import snapshot_download
    out = {"triposr": snapshot_download("stabilityai/TripoSR", local_dir="/v/r5/hf/triposr", allow_patterns=["config.yaml", "model.ckpt", "README.md"]),
           "dino": snapshot_download("facebook/dino-vitb16", cache_dir="/v/r5/hf/hub")}
    r5_volume.commit()
    return out


@app.function(image=image, cpu=8, memory=32768, volumes=VOLS, timeout=1800, retries=0)
def check():
    """No GPU: each venv imports its generator's code (the image is right before a GPU container boots)."""
    from fast_report import gen3d, x7
    cmds = {"recgen": [x7.RECGEN_PY, "-c", "from recgen_inference import build_recgen, generate, generate_multiview; print('ok')"],
            "trellis": [gen3d.TRELLIS_PY, "-c", "import sys; sys.path.insert(0, '/repo'); from fast_report import gen3d; import types, os; "
                        "gen3d._stubs(); "
                        "[sys.modules.__setitem__(n, type(sys)(n)) or setattr(sys.modules[n], '__path__', [os.path.join(gen3d.TRELLIS_DIR, *n.split('.'))]) "
                        "for n in ('trellis', 'trellis.pipelines')]; "
                        "from trellis.pipelines.trellis_image_to_3d import TrellisImageTo3DPipeline; import trellis.models.structured_latent_vae; print('ok')"],
            "triposr": [gen3d.TRIPOSR_PY, "-c", "import sys; sys.path[:0] = ['/repo', '/opt/triposr']; from fast_report import gen3d; "
                        "sys.modules['torchmcubes'] = gen3d._mcubes_stand_in(); import types; sys.modules['rembg'] = types.ModuleType('rembg'); "
                        "from tsr.system import TSR; print('ok')"],
            "gate": ["/opt/gate/bin/python", "-c", "import sys; sys.path[:0] = ['/repo', '/repo/scripts']; from fast_report import r5_bench, surface; "
                     "surface.self_check(); r5_bench.self_check()"]}
    out = {}
    for k, c in cmds.items():
        r = subprocess.run(c, capture_output=True, text=True, env={**os.environ, "ATTN_BACKEND": "xformers", "SPCONV_ALGO": "native",
                                                                    "PYTHONPATH": x7.RECGEN_DIR if k == "recgen" else ""})
        out[k] = {"rc": r.returncode, "out": r.stdout[-600:], "err": r.stderr[-2500:]}
    out["weights"] = {p: os.path.exists(p) for p in (gen3d.TRELLIS_PIPE + "/pipeline.json", gen3d.DINO_WEIGHTS, gen3d.TRIPOSR_WEIGHTS + "/model.ckpt")}
    return out


@app.function(image=image, gpu="A100-80GB", cpu=8, memory=65536, volumes=VOLS, timeout=1200, retries=0)
def smoke():
    """One A100: a TRELLIS and a TripoSR process, their boot (load + a synthetic warm-up) and three timed synthetic calls each."""
    from fast_report import gen3d, sam3d
    env = {"trellis": sam3d.worker_env(gen3d.TRELLIS_PY, CUDA_VISIBLE_DEVICES=0, ATTN_BACKEND="xformers", SPCONV_ALGO="native", HF_HUB_OFFLINE=1),
           "triposr": sam3d.worker_env(gen3d.TRIPOSR_PY, CUDA_VISIBLE_DEVICES=0, HF_HOME=gen3d.HF_HOME, HF_HUB_OFFLINE=1)}
    out = {}
    for kind, py in (("trellis", gen3d.TRELLIS_PY), ("triposr", gen3d.TRIPOSR_PY)):
        pool = sam3d.Pool([py, "-c", f"from fast_report.gen3d import worker; worker('{kind}')"], 1, [env[kind]], kind)
        try:
            out[kind] = {"boot": pool.ready(900)}
            rgb = np.full((400, 400, 3), 90, np.uint8)
            rgb[100:300, 120:280] = (40, 60, 200)
            mask = np.zeros((400, 400), bool)
            mask[100:300, 120:280] = True
            crop = gen3d.rgba_crop(rgb, mask)
            calls = []
            for views in ([crop], [crop, crop], [crop, crop, crop]):
                r = pool.submit({"views": views, "seed": 42}).result()
                calls.append({"views": len(views), "s": round(r["seconds"], 2), "faces": int(len(r["faces"])), "max_reserved_gib": r.get("max_reserved_gib")})
            out[kind]["calls"] = calls
        except Exception as error:  # noqa: BLE001
            out[kind] = {**out.get(kind, {}), "error": str(error)[-3000:]}
        finally:
            pool.close()
    return out


@app.cls(image=image, cpu=CPU, memory=96 * 1024, volumes=VOLS, timeout=3600, retries=0, max_containers=1, scaledown_window=20)
class CpuBench:
    @modal.enter()
    def boot(self):
        boot_pools(self, "cpu")

    @modal.exit()
    def stop(self):
        for p in getattr(self, "pools", {}).values():
            p.close()

    @modal.method()
    def boot_info(self):
        return self.boot_record

    @modal.method()
    def run(self, name: str, options: dict):
        """Every card's (a) + (d) (timed on CPU_PROCS processes), then the held-out tests on every card X7's selection admits."""
        from fast_report.instrument import Clock
        t_load = time.time()
        d = load_dump(name)
        fx, keys, spec = d["fx"], d["keys"], d["spec"]
        cards = [c for c in fx["meta"]["cards"] if c.get("kind") == "object"]
        objs = {o["id"]: o for o in fx["meta"]["objects"]}
        load_s = round(time.time() - t_load, 2)
        cpu = self.pools["cpu"]
        clock = Clock()
        rec = {c["id"]: {"id": c["id"], "shot": c["shot"], "name": c["identity"].get("name"), "class": (c.get("class") or {}).get("class_word"),
                         "views": c["views"].get("n"), "longest_m": (c["raw"].get("size") or {}).get("longest"), "near_m": (c["views"].get("distance_m") or [None])[0],
                         "model_r4": (c.get("model") or {}).get("kind"), "angles_r4": {k: c["physical"].get(k, {}).get("status", "value" if "value" in c["physical"].get(k, {}) else None)
                                                                                     for k in ("planar_slope_deg", "principal_axis_tilt_deg")}}
               for c in cards}
        events, pending = queue.Queue(), [0]
        by = {c["id"]: c for c in cards}
        lite = lambda c: {"id": c["id"], "shot": c["shot"], "physical": {"merged_from": c["physical"].get("merged_from", []), "box": c["physical"].get("box")},  # noqa: E731
                          "views": {"subsets": c["views"].get("subsets", [])}, "class": c.get("class")}

        def submit(msg, prio, kind, r, method=None):
            pending[0] += 1
            cpu.submit(msg, prio).add_done_callback(lambda f: events.put((kind, r, method, f)))
        for i, c in enumerate(cards):
            submit({"op": "surface", "src": spec, "shot": c["shot"], "fixture": d["folder"], "card": lite(c), "plumb_u": plumb_u(fx, c["shot"])},
                   (0, i), "surface", c["id"])
        first_wave = [None]

        def handle(kind, cid, method, res):
            R = rec[cid]
            if "error" in res and kind != "select":
                R.setdefault("errors", []).append(f"{kind}: {res['error'][-600:]}")
                return
            if kind == "surface":
                R["surface"] = {k: res[k] for k in ("points", "mesh", "parts", "parts_s")}
                R["surface_done_s"] = clock.unix_to_s(res["end_unix"])
                if all("surface" in x or "errors" in x for x in rec.values()) and first_wave[0] is None:
                    first_wave[0] = clock.now()
                    tilted = [x["id"] for x in rec.values() if any(15 <= p["tilt_deg"]["value"] <= 75 for p in ((x.get("surface") or {}).get("parts") or {}).get("parts", []))]
                    rest = [x["id"] for x in rec.values() if ((x.get("surface") or {}).get("parts") or {}).get("parts") and x["id"] not in tilted]
                    rest = [rest[i] for i in sorted(np.random.default_rng(7).choice(len(rest), min(len(rest), 20), replace=False))] if rest else []
                    for cid2 in tilted[:40] + rest:
                        c2 = by[cid2]
                        best = (c2["views"].get("best") or [None])[0]
                        if best is not None:
                            submit({"op": "parts_tile", "src": spec, "shot": c2["shot"], "fixture": d["folder"], "card": lite(c2), "frame_key": int(best)},
                                   (3,), "parts_tile", cid2)
                    for cid2 in [c["id"] for c in cards if c["id"] in objs]:
                        c2 = by[cid2]
                        submit({"op": "select", "src": spec, "shot": c2["shot"], "key": keys[cid2], "obj": {"centroid_m": objs[cid2]["centroid_m"]},
                                "min_sep": 15.}, (1,), "select", cid2, 15.)
            elif kind == "select":
                if "error" in res:
                    R["select"] = {"error": res["error"][-300:]}
                    return
                sel = {k: res.get(k) for k in ("eligible", "reason", "views_with_depth", "good_views", "min_pair_deg", "select_s")}
                sel.update(gen=[g["frame"] for g in res.get("gen") or []], held=(res.get("held") or {}).get("frame"), sep=method, crop=res.get("held_crop"))
                if not res["eligible"] and method == 15.:
                    R["select_15"] = sel
                    c2 = by[cid]
                    submit({"op": "select", "src": spec, "shot": c2["shot"], "key": keys[cid], "obj": {"centroid_m": objs[cid]["centroid_m"]}, "min_sep": 8.},
                           (1,), "select", cid, 8.)
                    return
                R["select"] = sel
                if res["eligible"]:
                    R["held_tile_crop"] = res["held_tile"]
                    c2 = by[cid]
                    submit({"op": "heldout", "src": spec, "shot": c2["shot"], "fixture": d["folder"], "card": lite(c2), "key": keys[cid],
                            "gen": sel["gen"], "held": sel["held"], "crop": sel["crop"], "splat": d["splat"], "sep": method}, (2,), "heldout", cid)
            elif kind == "heldout":
                R["heldout"] = {k: v for k, v in res.items() if k not in ("tiles", "start_unix", "end_unix", "pid")}
                R["tiles"] = {**R.get("tiles", {}), **res.get("tiles", {}), **({"crop": R.pop("held_tile_crop")} if "held_tile_crop" in R else {})}
            elif kind == "parts_tile" and res.get("tile"):
                R.setdefault("tiles", {})["parts"] = res["tile"]
        events_loop(pending, events, handle)
        surf = [r["surface"] for r in rec.values() if "surface" in r]
        summary = {"cards": len(cards), "with_mesh": sum(r["mesh"].get("triangles", 0) > 0 for r in surf),
                   "with_parts": sum("parts" in r["parts"] for r in surf),
                   "mesh_s": stats(r["mesh"]["s"] for r in surf), "parts_s": stats(r["parts_s"] for r in surf),
                   "all_surface_wall_s": round(first_wave[0], 2) if first_wave[0] else None, "processes": CPU_PROCS}
        return {"name": name, "report": fx["meta"]["report"], "load_s_not_analysis": load_s, "summary": summary, "cards": list(rec.values()),
                "elapsed_s": round(clock.now(), 2), "splat": d["splat"] is not None}


@app.cls(image=image, gpu="A100-80GB:2", cpu=CPU, memory=128 * 1024, volumes=VOLS, timeout=3600, retries=0, max_containers=2, scaledown_window=20)
class GpuBench:
    arm: str = modal.parameter()

    @modal.enter()
    def boot(self):
        from fast_report.instrument import Vram
        self.vram = Vram([0, 1]).start()
        boot_pools(self, self.arm)
        self.boot_record["vram_after_boot_gib"] = self.vram.now()

    @modal.exit()
    def stop(self):
        for p in getattr(self, "pools", {}).values():
            p.close()

    @modal.method()
    def boot_info(self):
        return self.boot_record

    @modal.method()
    def run(self, name: str, picks: list, options: dict):
        """picks: [{id, sep}] best first. Selection again here (it builds the generators' inputs), every generator of the arm on
        each selected object as soon as its selection is back, placement and the gate as soon as a model is back."""
        from fast_report.instrument import Clock
        t_load = time.time()
        d = load_dump(name)
        fx, keys, spec = d["fx"], d["keys"], d["spec"]
        cards = {c["id"]: c for c in fx["meta"]["cards"] if c.get("kind") == "object"}
        objs = {o["id"]: o for o in fx["meta"]["objects"]}
        up = {s["index"]: s["normal"] for s in fx["meta"]["cards_shots"]}
        load_s = round(time.time() - t_load, 2)
        methods = ("recgen",) if self.arm == "internal" else ("sam3d", "trellis", "triposr")
        clock = Clock()
        vram = self.vram
        rec = {p["id"]: {"id": p["id"], "sep": p["sep"], "methods": {}} for p in picks}
        events, pending, tiles = queue.Queue(), [0], {}
        lite = lambda c: {"id": c["id"], "shot": c["shot"], "physical": {"merged_from": c["physical"].get("merged_from", []), "box": c["physical"].get("box")},  # noqa: E731
                          "views": {"subsets": c["views"].get("subsets", [])}, "class": c.get("class")}
        sel = {}

        def submit(pool, msg, prio, kind, r, method=None):
            pending[0] += 1
            self.pools[pool].submit(msg, prio).add_done_callback(lambda f: events.put((kind, r, method, f)))
        for i, p in enumerate(picks):
            c = cards[p["id"]]
            submit("cpu", {"op": "select", "src": spec, "shot": c["shot"], "key": keys[p["id"]], "obj": {"centroid_m": objs[p["id"]]["centroid_m"]},
                           "min_sep": p["sep"], "jobs": True}, (1, i), "selected", p["id"])

        def handle(kind, cid, method, res):
            R = rec[cid]
            c = cards[cid]
            if "error" in res:
                (R["methods"].setdefault(method, {}) if method else R).setdefault("errors", []).append(f"{kind}: {res['error'][-800:]}")
                return
            if kind == "selected":
                clock.external("r5.select", None, res["start_unix"], res["end_unix"])
                if not res["eligible"]:
                    R["select"] = {"eligible": False, "reason": res.get("reason")}
                    return
                gen, held = [g["frame"] for g in res["gen"]], res["held"]["frame"]
                sel[cid] = {"src": spec, "shot": c["shot"], "key": keys[cid], "gen": gen, "held": held, "crop": res["held_crop"], "fixture": d["folder"],
                            "card": lite(c), "up": up[c["shot"]]}
                R["select"] = {"eligible": True, "gen": gen, "held": held}
                tiles.setdefault(cid, {})["crop"] = res["held_tile"]
                now = clock.now()
                if "recgen" in methods and res.get("recgen_job"):
                    R["methods"]["recgen"] = {"dispatched_s": now, "views": len(res["recgen_job"]["views"])}
                    sel[cid]["anchor"] = res["recgen_views"][0]
                    submit("recgen", res["recgen_job"], (0,), "generated", cid, "recgen")
                if "sam3d" in methods and res.get("sam3d_job"):
                    j = res["sam3d_job"]
                    R["methods"]["sam3d"] = {"dispatched_s": now, "view": j["frame"]}
                    sel[cid]["sam3d_view"] = j["frame"]
                    submit("sam3d", {k: j[k] for k in ("rgb", "mask", "pointmap", "seed")}, (0,), "generated", cid, "sam3d")
                crops = res.get("rgba") or []
                for m, vs in (("trellis", crops), ("triposr", crops[:1])):
                    if m in methods and vs:
                        R["methods"][m] = {"dispatched_s": now, "views": len(vs)}
                        submit(m, {"views": vs, "seed": 42}, (0,), "generated", cid, m)
            elif kind == "generated":
                M = R["methods"][method]
                M.update(generate_s=round(res["seconds"], 3), gpu=res.get("gpu", res.get("worker")), started_s=clock.unix_to_s(res["start_unix"]),
                         generated_s=clock.unix_to_s(res["end_unix"]), faces_raw=int(len(res["faces"])),
                         max_reserved_gib=res.get("max_reserved_gib", res.get("max_reserved_gb")))
                clock.external(f"r5.{method}.generate", None, res["start_unix"], res["end_unix"])
                mesh = {k: res.get(k) for k in ("vertices", "faces", "colors")}
                if method == "sam3d":
                    mesh["objectToCamera"] = res["objectToCamera"]
                s = sel[cid]
                kind_ = {"recgen": "recgen", "sam3d": "sam3d"}.get(method, method)
                submit("cpu", {"op": "gate_free" if method in ("trellis", "triposr") else "gate", **s, "kind": kind_, "mesh": mesh,
                               "source_frame": s.get("anchor") if method == "recgen" else s.get("sam3d_view")}, (0,), "gated", cid, method)
            elif kind == "gated":
                M = R["methods"][method]
                clock.external(f"r5.{method}.gate", None, res["start_unix"], res["end_unix"])
                if res.get("tile"):
                    tiles.setdefault(cid, {})[method] = res.pop("tile")
                M.update({k: v for k, v in res.items() if k not in ("start_unix", "end_unix", "pid")})
                M.update(accepted=bool((res.get("gate") or {}).get("accepted_source_consistency")), decided_s=clock.unix_to_s(res["end_unix"]))
        events_loop(pending, events, handle)
        rep = clock.report(vram, price_per_s=2 * PRICE["A100-80GB"] + CPU * PRICE["cpu"] + 128 * PRICE["gib"], report=f"r5-{name}-{self.arm}", site=name)
        summary = {}
        for m in methods:
            rows = [R["methods"][m] for R in rec.values() if m in R["methods"]]
            acc = sorted(r["decided_s"] for r in rows if r.get("accepted"))
            summary[m] = {"dispatched": len(rows), "generated": sum("generate_s" in r for r in rows), "gated": sum("gate" in r for r in rows),
                          "accepted": len(acc), "first_accepted_s": acc[0] if acc else None, "last_accepted_s": acc[-1] if acc else None,
                          "last_decided_s": max((r["decided_s"] for r in rows if "decided_s" in r), default=None),
                          "generate_s": stats(r.get("generate_s") for r in rows), "gate_s": stats(r.get("gate_s") for r in rows),
                          "queue_wait_s": stats(r["started_s"] - r["dispatched_s"] for r in rows if "started_s" in r),
                          "max_reserved_gib_process": max((r.get("max_reserved_gib") or 0 for r in rows), default=None)}
        return {"name": name, "arm": self.arm, "load_s_not_analysis": load_s, "picks": len(picks), "summary": summary, "objects": list(rec.values()),
                "tiles": tiles, "elapsed_s": rep["elapsed_s"], "gpu_peak": rep["gpu_peak"], "flags": [f for f in rep["flags"] if "unknown stage" not in f],
                "usd_estimate_analysis": rep.get("usd_estimate")}


# ---------------------------------------------------------------- local
def visibility(r):
    return (r.get("views") or 0) * (r.get("longest_m") or 0) / max(r.get("near_m") or 1e9, 1e-3)


@app.local_entrypoint()
def main(out: str, names: str = ",".join(NAMES), arms: str = "cpu,internal,commercial", top: int = 70, cpu_from: str = ""):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    arms_ = arms.split(",")
    names_ = names.split(",")
    cpu_res = {}
    if cpu_from:
        for n in names_:
            cpu_res[n] = json.loads((Path(cpu_from) / f"{n}-cpu.json").read_text())
    started = {}
    if "internal" in arms_:
        started["internal"] = GpuBench(arm="internal")
        started["internal"].boot_info.spawn()
    if "commercial" in arms_:
        started["commercial"] = GpuBench(arm="commercial")
        started["commercial"].boot_info.spawn()
    if "cpu" in arms_:
        cb = CpuBench()
        called = time.time()
        boot = cb.boot_info.remote()
        boot["client_called_unix"] = called
        (out / "boot-cpu.json").write_text(json.dumps(boot, indent=1, default=str))
        for n in names_:
            t = time.time()
            res = cb.run.remote(n, {})
            res["client_wall_s"] = round(time.time() - t, 1)
            tl = {r["id"]: r.pop("tiles") for r in res["cards"] if "tiles" in r}
            np.savez_compressed(out / f"{n}-cpu-tiles.npz", **{f"{cid}__{k}": np.frombuffer(v, np.uint8) for cid, t_ in tl.items() for k, v in t_.items()})
            (out / f"{n}-cpu.json").write_text(json.dumps(res, indent=1, default=str))
            cpu_res[n] = res
            print(n, "cpu", json.dumps(res["summary"], default=str), "elapsed", res["elapsed_s"], flush=True)
    for arm, g in started.items():
        boot = g.boot_info.remote()
        (out / f"boot-{arm}.json").write_text(json.dumps(boot, indent=1, default=str))
        print("boot", arm, json.dumps({k: v for k, v in boot.items() if k.endswith("_s") or k in ("mps",)}), flush=True)
        for k in ("recgen", "sam3d", "trellis", "triposr", "cpu"):
            if isinstance(boot.get(k), dict) and boot[k].get("error"):
                print("  pool error", k, boot[k]["error"][-1500:], flush=True)
    calls = []
    for n in names_:
        if n not in cpu_res:
            continue
        ranked = sorted([r for r in cpu_res[n]["cards"] if (r.get("select") or {}).get("eligible")], key=lambda r: -visibility(r))[:top]
        picks = [{"id": r["id"], "sep": r["select"]["sep"]} for r in ranked]
        (out / f"{n}-picks.json").write_text(json.dumps(picks, indent=1))
        for arm, g in started.items():
            calls.append((n, arm, g.run.spawn(n, picks, {})))
    for n, arm, call in calls:
        res = call.get()
        tiles = res.pop("tiles")
        np.savez_compressed(out / f"{n}-{arm}-tiles.npz", **{f"{cid}__{k}": np.frombuffer(v, np.uint8) for cid, t_ in tiles.items() for k, v in t_.items()})
        (out / f"{n}-{arm}.json").write_text(json.dumps(res, indent=1, default=str))
        print(n, arm, json.dumps(res["summary"], default=str)[:2500], "elapsed", res["elapsed_s"], flush=True)


@app.local_entrypoint()
def smoke_run():
    print(json.dumps(smoke.remote(), indent=1, default=str), flush=True)


@app.local_entrypoint()
def prep():
    """The weights (setup) then the image check (check), printed."""
    print(json.dumps(setup.remote(), indent=1), flush=True)
    print(json.dumps(check.remote(), indent=1), flush=True)


def self_check():
    rows = [{"views": 10, "longest_m": 1., "near_m": 2.}, {"views": 3, "longest_m": .2, "near_m": 4.}]
    assert visibility(rows[0]) == 5. and visibility(rows[1]) == .15 and stats([1, None, 3])["median"] == 2.
    print("r5_models_bench self-check ok: visibility ranking, stats")


if __name__ == "__main__":
    assert sys.argv[1:] == ["--self-check"], __doc__
    self_check()
