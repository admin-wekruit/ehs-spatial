"""recgen/fast bench: RecGen per stage, speed-up arms, views 1-4, forward batching and processes under MPS, on ONE
A100-80GB (isolated), on X7's saved inputs: the fx-x7-001 fixtures of ME340 / Sam's Club / Walmart and X7's sep5 view
selection (x7.select re-run here, checked frame for frame against runs/fx-x7-object-models-001/models2/*-sep5.json).

Every arm runs the same OBJECTS (19: 6 / 5 / 4 / 4 with 1 / 2 / 3 / 4 generation views) in one warm process after one untimed
warm-up call. Each mesh goes to fast_report/recgen_fast.judge on the CPU pool while the GPU works on: X7's placement and
held-out gate (IoU >= 0.65, depth median <= 0.04, p95 <= 0.10, >= 103 px), and the Chamfer distance to the default arm's mesh
(seed 42, same object, same anchor camera). RecGen: TRI non-commercial licence, internal use only.

  M=/Users/adam/Desktop/Tesla/panoptes-platform/.venv/bin/modal
  $M run modal_apps/recgen_fast_bench.py --plan smoke --out RUNS/recgen-fast-001     # ephemeral, retries 0
  $M run modal_apps/recgen_fast_bench.py --plan main --out RUNS/recgen-fast-002
  $M run modal_apps/recgen_fast_bench.py --plan confirm --out RUNS/recgen-fast-003   # FAST x 3 seeds, backends, SAM 3D s1cfg12
  $M run modal_apps/recgen_fast_bench.py --plan mps --out RUNS/recgen-fast-004       # 1-3 processes under MPS
  $M run modal_apps/recgen_fast_bench.py --plan compile --out RUNS/recgen-fast-005   # torch.compile of the SS model
  python modal_apps/recgen_fast_bench.py --summarize RUNS/recgen-fast-00N     # local: that run's tables -> summary.json
  python modal_apps/recgen_fast_bench.py --results RUNS                        # local: the report's numbers -> RUNS/recgen-fast-results
  python modal_apps/recgen_fast_bench.py --self-check
"""
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time

import modal
import numpy as np

WT = Path(__file__).resolve().parents[1]
LOCAL = modal.is_local()
sys.path[:0] = [str(WT), str(WT / "modal_apps"), str(WT / "scripts")] if LOCAL else ["/repo", "/repo/scripts", "/repo/modal_apps"]
PHASE2 = Path("/Users/adam/Desktop/panoptes-public/research-notes/phase2")
X7_RUN = PHASE2 / "runs/fx-x7-object-models-001/models2"
CPU, MEM_GIB, CPU_PROCS = 16, 80, 10
TIMEOUT = int(os.environ.get("RF_TIMEOUT", 5400))  # explicit per run: RF_TIMEOUT=1500 modal run ... (smoke)
USD_PER_S = .000694 + CPU * .0000131 + MEM_GIB * .00000222  # Modal list prices: A100-80GB + cores + GiB (estimate, not an invoice)
FLASH_WHL = ("https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/"
             "flash_attn-2.6.3+cu123torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl")
UTILS3D = "utils3d @ git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8"
OBJECTS = {"me340-165": ["obj-1-140", "obj-1-191", "obj-1-122", "obj-1-131", "obj-1-143", "obj-1-178", "obj-1-95", "obj-1-111"],
           "samsclub-337": ["obj-1-386", "obj-1-422", "obj-1-489", "obj-0-103", "obj-1-475", "obj-0-71"],
           "walmart-190": ["obj-1-517", "obj-1-516", "obj-0-172", "obj-1-531", "obj-1-461"]}
FAST = {"formats": ("mesh",), "fusion": "fused", "cache_cond": True}  # lossless for a mesh + vertex colours deliverable
RF_FAST = {"ss_steps": 12, "slat_steps": 8, "slat_cfg": 0, "formats": ("mesh",), "fusion": "fused", "cache_cond": True, "fast_out": True}
ARMS = {  # name -> (setting changes, seed)
    "default": ({}, 42), "default_s43": ({}, 43),
    "mesh_only": ({"formats": ("mesh",)}, 42), "fused": ({"fusion": "fused"}, 42),
    "ss12": ({"ss_steps": 12}, 42), "ss8": ({"ss_steps": 8}, 42), "slat12": ({"slat_steps": 12}, 42), "slat8": ({"slat_steps": 8}, 42),
    "slat_nocfg": ({"slat_cfg": 0}, 42), "nocfg": ({"ss_cfg": 0, "slat_cfg": 0}, 42),
    "stochastic": ({"fusion": "stochastic"}, 42), "views2": ({"max_views": 2}, 42), "lossless": (FAST, 42),
    "fast12": ({**FAST, "ss_steps": 12, "slat_steps": 12}, 42),
    "fast12_s2nocfg": ({**FAST, "ss_steps": 12, "slat_steps": 12, "slat_cfg": 0}, 42),
    "fast12_4": ({**FAST, "ss_steps": 12, "slat_steps": 4}, 42),
    "fast8": ({**FAST, "ss_steps": 8, "slat_steps": 8}, 42),
    "fast8_s2nocfg": ({**FAST, "ss_steps": 8, "slat_steps": 8, "slat_cfg": 0}, 42),
    "fast4": ({**FAST, "ss_steps": 4, "slat_steps": 4}, 42),
    "fast12_v2": ({**FAST, "ss_steps": 12, "slat_steps": 12, "max_views": 2}, 42),
    "fast12_stoch": ({**FAST, "fusion": "fused_stochastic", "ss_steps": 12, "slat_steps": 12}, 42),
    "fast12_s43": ({**FAST, "ss_steps": 12, "slat_steps": 12}, 43),
    "default_s44": ({}, 44),
    # the recommendation (after recgen-fast-002): recgen_fast.FAST
    "rec": (RF_FAST, 42), "rec_s43": (RF_FAST, 43), "rec_s44": (RF_FAST, 44),
}
XFORMERS = {"ATTN_BACKEND": "xformers"}  # X7's
SDPA = {"ATTN_BACKEND": "sdpa", "SPARSE_ATTN_BACKEND": "xformers"}  # dense attention on torch's flash kernels; sparse stays
FLASH = {"ATTN_BACKEND": "flash_attn"}


def arm(name, reference=False, export=0, op="run"):
    change, seed = ARMS[name]
    return {"name": name, "setting": change, "seed": seed, "reference": reference, "export": export, "op": op}


def plans(name):
    if name == "smoke":
        objects = {"me340-165": ["obj-1-140"], "samsclub-337": ["obj-1-386", "obj-1-475"]}
        procs = [{"name": "xformers", "env": XFORMERS, "micro": True,
                  "arms": [arm("default", reference=True, export=1), arm("default", op="x7") | {"name": "x7_call"}, arm("fused"),
                           arm("mesh_only"), arm("fast12")],
                  "sweep": {"arms": ["default"], "min_views": 3}}]
        return {"objects": objects, "procs": procs}
    if name == "main":
        singles = ["default_s43", "mesh_only", "fused", "lossless", "ss12", "ss8", "slat12", "slat8", "slat_nocfg", "nocfg", "stochastic", "views2"]
        combos = ["fast12", "fast12_s43", "fast12_s2nocfg", "fast12_4", "fast8", "fast8_s2nocfg", "fast4", "fast12_v2", "fast12_stoch"]
        procs = [{"name": "xformers", "env": XFORMERS, "micro": True,
                  "arms": [arm("default", reference=True, export=5), arm("default", op="x7") | {"name": "x7_call"}]
                          + [arm(a) for a in singles + combos],
                  "sweep": {"arms": ["default", "fast12"], "min_views": 3}},
                 {"name": "sdpa", "env": SDPA, "arms": [arm("default"), arm("fast12")]},
                 {"name": "flash_attn", "env": FLASH, "arms": [arm("default"), arm("fast12")]},
                 {"name": "sdpa_compile", "env": {**SDPA, "COMPILE_SS": "1"}, "arms": [arm("default"), arm("fast12")]},
                 {"name": "sam3d", "kind": "sam3d", "arms": [{"name": "s1cfg12", "setting": {}, "seed": 42, "reference": False, "op": "sam3d"}]}]
        return {"objects": OBJECTS, "procs": procs}
    if name == "mps":  # resident processes on one GPU under MPS: the default and the recommendation
        return {"objects": OBJECTS, "procs": [], "mps": {"env": XFORMERS, "procs": [1, 2, 3], "arms": ["default", "rec"], "calls": 20}}
    if name == "compile":  # torch.compile of the SS flow model (triton added), against the same setting uncompiled in this container
        procs = [{"name": "xformers", "env": XFORMERS, "arms": [arm("rec", reference=True)]},
                 {"name": "sdpa_compile", "env": {**SDPA, "COMPILE_SS": "1"}, "arms": [arm("rec")]}]
        return {"objects": OBJECTS, "procs": procs}
    if name == "confirm":  # the recommendation at 3 seeds against the default at a third seed; what recgen-fast-002 lost to preemption
        procs = [{"name": "xformers", "env": XFORMERS,
                  "arms": [arm("default", reference=True), arm("rec"), arm("rec_s43"), arm("rec_s44"), arm("default_s44")]},
                 {"name": "flash_attn", "env": FLASH, "arms": [arm("fast12"), arm("rec")]},
                 {"name": "sdpa_compile", "env": {**SDPA, "COMPILE_SS": "1"}, "arms": [arm("default"), arm("fast12"), arm("rec")]},
                 {"name": "sam3d", "kind": "sam3d", "arms": [{"name": "s1cfg12", "setting": {}, "seed": 42, "reference": False, "op": "sam3d"}]}]
        return {"objects": OBJECTS, "procs": procs}
    raise ValueError(name)


app = modal.App("panoptes-recgen-fast-bench")
RUNS = modal.Dict.from_name("panoptes-recgen-fast-runs", create_if_missing=True)  # plan token -> first entry (preemption guard)
VOLS = {"/cache": modal.Volume.from_name("panoptes-lucida-weights"), "/v/layers": modal.Volume.from_name("panoptes-fb-layers"),
        "/weights": modal.Volume.from_name("panoptes-sam3d-weights")}
if LOCAL:
    from fast_report import sam3d as fb_sam3d, x7 as fx7
    base = (modal.Image.from_registry("nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04", add_python="3.11")  # X7's image: cached layers
            .pip_install("numpy==2.2.6", "opencv-python-headless==4.11.0.86", "scipy==1.15.3", "pydantic==2.11.7", "trimesh==4.6.10"))
    image = (fx7.with_recgen(fb_sam3d.with_envs(base))
             .run_commands(f"uv pip install --python {fx7.RECGEN_PY} {FLASH_WHL}",
                           f"{fx7.RECGEN_PY} -c 'import flash_attn; print(\"flash_attn\", flash_attn.__version__)'")
             .run_commands(f"uv pip install --python {fx7.RECGEN_PY} --no-deps '{UTILS3D}'",  # RecGen's PLY export (its pin)
                           f"{fx7.RECGEN_PY} -c 'import utils3d; print(\"utils3d ok\")'")
             .run_commands(f"uv pip install --python {fx7.RECGEN_PY} triton==3.0.0 setuptools",  # inductor (compile arm): triton imports setuptools
                           f"{fx7.RECGEN_PY} -c 'import torch, triton; from torch.utils._triton import has_triton_package; "
                           f"print(\"torch\", torch.__version__, \"triton\", triton.__version__, has_triton_package())'")
             .add_local_dir(WT / "fast_report", "/repo/fast_report", ignore=["**/__pycache__/**"])
             .add_local_dir(WT / "scripts", "/repo/scripts", ignore=["**/__pycache__/**"])
             .add_local_dir(WT / "modal_apps", "/repo/modal_apps", ignore=["**/__pycache__/**"])
             .add_local_dir(WT / "ehs_spatial", "/repo/ehs_spatial", ignore=["**/__pycache__/**"]))
else:
    image = modal.Image.debian_slim()


def load_clip(name, fixture="fx-x7-001"):
    """X7's run(): the core's hand-off into memory, ranked and staged as X7 staged it (same keys)."""
    from fast_report import sam3d
    folder = Path("/v/layers/x7") / fixture / name
    meta = json.loads((folder / "fixture.json").read_text())
    z = np.load(folder / "fixture.npz")
    shots = [{"index": s["index"], "keys": z[f"s{s['index']}_keys"].tolist(), "depth_m": z[f"s{s['index']}_depth"], "c2w_m": z[f"s{s['index']}_c2w"],
              "K": z[f"s{s['index']}_K"], "person": z[f"s{s['index']}_person"]} for s in meta["shots"]]
    objs = []
    for i, o in enumerate(meta["objects"]):
        fr, lg = z[f"o{i}_frames"], z[f"o{i}_logits"]
        objs.append({**o, "masks_lr": {int(f): lg[j] for j, f in enumerate(fr)}})
    frames = sam3d.frames_buffer(meta["n_frames"], name=f"rf-{name}-frames.npy")
    for f, img in zip(z["frame_ids"], z["frames"]):
        frames[int(f)] = img
    frames.flush()
    ranked = sam3d.rank(objs)[:100]
    spec, keys = sam3d.stage(ranked, shots, str(frames.filename))
    return {"spec": spec, "keys": keys, "by_id": {o["id"]: o for o in ranked}}


def stats(values):
    v = np.asarray([x for x in values if x is not None], float)
    return None if not len(v) else {"n": int(len(v)), "median": round(float(np.median(v)), 3), "mean": round(float(v.mean()), 3),
                                    "p90": round(float(np.percentile(v, 90)), 3), "sum": round(float(v.sum()), 2)}


@app.function(image=image, gpu="A100-80GB", cpu=CPU, memory=MEM_GIB * 1024, volumes=VOLS, timeout=TIMEOUT)  # a generator: Modal never retries it
def bench(plan: dict):
    from fast_report import recgen_fast as rf, sam3d, x7
    from fast_report.instrument import Vram
    entered = time.time()
    if RUNS.get(plan["token"]) is not None:  # Modal restarts a preempted function with the same input: never pay twice
        yield {"kind": "done", "entered_unix": entered, "done_unix": time.time(), "usd_estimate": 0., "error": "restart after preemption: skipped"}
        return
    RUNS[plan["token"]] = entered
    if plan.get("mps"):
        env = {"CUDA_MPS_PIPE_DIRECTORY": "/tmp/mps-pipe", "CUDA_MPS_LOG_DIRECTORY": "/tmp/mps-log"}
        for d in env.values():
            Path(d).mkdir(parents=True, exist_ok=True)
        os.environ.update(env)
        assert subprocess.run(["nvidia-cuda-mps-control", "-d"], capture_output=True).returncode == 0, "MPS daemon"
    vram = Vram([0]).start()
    cpu = sam3d.Pool([sam3d.GATE_PY, "-c", "from fast_report.recgen_fast import cpu_worker; cpu_worker()"], CPU_PROCS,
                     sam3d.worker_env(sam3d.GATE_PY, OMP_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1, MKL_NUM_THREADS=1), "cpu")
    t = time.time()
    clips = {name: load_clip(name) for name in plan["objects"]}
    fixtures_s = round(time.time() - t, 1)
    cpu.ready(900)
    yield {"kind": "boot", "entered_unix": entered, "fixtures_s": fixtures_s, "ready_s": round(time.time() - entered, 1),
           "gpu": subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.strip()}
    # ---- X7's selection (sep5 arm) for the benchmark objects, checked against X7's records
    objs, futures = {}, {}
    for name, ids in plan["objects"].items():
        c = clips[name]
        for oid in ids:
            o = c["by_id"][oid]
            futures[oid] = (name, cpu.submit({"op": "select", "src": c["spec"], "shot": int(o["shot"]), "key": c["keys"][oid],
                                              "obj": {"centroid_m": o["centroid_m"]}, "min_sep": 5.}))
    summary = []
    for oid, (name, fut) in futures.items():
        res, c = fut.result(), clips[name]
        want = plan["expected"][oid]
        got = {"gen": [g["frame"] for g in res["gen"]], "held": res["held"]["frame"], "recgen_views": res["recgen_views"]}
        objs[oid] = {"clip": name, "views": res["recgen_job"]["views"], "held_tile": res["held_tile"], "sam3d_job": res.get("sam3d_job"),
                     "sam3d_rejected": res.get("sam3d_b_rejected"),
                     "sel": {"src": c["spec"], "shot": int(c["by_id"][oid]["shot"]), "key": c["keys"][oid], "gen": got["gen"], "held": got["held"],
                             "crop": res["held_crop"], "anchor": got["recgen_views"][0]}}
        summary.append({"id": oid, "clip": name, "label": c["by_id"][oid].get("word"), **got, "matches_x7": got == want, "x7": want,
                        "crop_px": [v["rgb"].shape[:2] for v in objs[oid]["views"]], "select_s": round(res["select_s"], 2)})
    yield {"kind": "objects", "objects": summary, "held_tiles": {oid: o["held_tile"] for oid, o in objs.items()}}
    refs = {}

    def judge(oid, mesh, ref, anchor=None):
        return cpu.submit({"op": "judge", **objs[oid]["sel"], **({"anchor": anchor} if anchor else {}), "mesh": mesh, "ref": ref})

    def one_arm(worker, proc, a, ids, max_views=None, sam=False):
        """RecGen (setting) or, sam=True, SAM 3D s1cfg12 on X7's SAM 3D view (its prepare rule; a rejection is recorded)."""
        s = None if sam else rf.setting(**{**a["setting"], **({"max_views": max_views} if max_views else {})})

        def message(oid):
            if sam:
                return {k: objs[oid]["sam3d_job"][k] for k in ("rgb", "mask", "pointmap", "seed")}
            return {"op": a["op"], "views": objs[oid]["views"], "seed": a["seed"], "setting": s}
        run = [oid for oid in ids if not sam or objs[oid]["sam3d_job"]]
        t0 = time.time()
        warm = worker.submit({**message(run[0]), **({} if sam else {"check_out": True})}).result()
        records = {oid: {"prepare_rejected": objs[oid]["sam3d_rejected"]} for oid in ids if oid not in run}
        judged = {}
        for i, oid in enumerate(run):
            m = message(oid)
            if i < a.get("export", 0):
                m["export_dir"] = f"/tmp/recgen-export/{proc}/{oid}"
            t = time.time()
            res = worker.submit(m).result()
            mesh = {k: res.pop(k) for k in ("vertices", "faces", "colors", "objectToCamera") if k in res}
            if sam:
                j = objs[oid]["sam3d_job"]
                res.update(max_reserved_gib=round(res.pop("max_reserved_gb") * 1e9 / 2 ** 30, 2), view=j["frame"], method=j["method"],
                           faces_n=int(len(mesh["faces"])))
            if a["reference"]:
                refs[oid] = {"vertices": mesh["vertices"], "faces": mesh["faces"]}
            records[oid] = {**{k: v for k, v in res.items() if k not in ("start_unix", "end_unix")}, "round_trip_s": round(time.time() - t, 3)}
            judged[oid] = judge(oid, mesh, None if a["reference"] or sam else refs.get(oid), anchor=objs[oid]["sam3d_job"]["frame"] if sam else None)
        tiles = {}
        for oid, f in judged.items():
            try:
                j = f.result()
            except Exception as error:  # noqa: BLE001  one object's failure is recorded, the rest go on
                records[oid]["judge_error"] = str(error)[-1500:]
                continue
            tiles[oid] = j.pop("tile")
            records[oid].update({k: v for k, v in j.items() if k not in ("start_unix", "end_unix")})
        return {"kind": "arm", "proc": proc, "arm": a["name"], "setting": s, "seed": a["seed"], "max_views": max_views, "records": records,
                "fast_out_check": warm.get("fast_out_check"),
                "tiles": tiles, "warm_up_s": round(warm["seconds"], 3), "arm_wall_s": round(time.time() - t0, 1),
                "gpu_peak_gib": vram.window(time.perf_counter() - (time.time() - t0), time.perf_counter())}

    for proc in plan["procs"]:
        sam = proc.get("kind") == "sam3d"
        t = time.time()
        if sam:  # X7's SAM 3D process: s1cfg12, the mesh-only decoder, internal depth disabled
            import complete_video_objects as cvo
            env = sam3d.worker_env(sam3d.SAM3D_PY, CUDA_VISIBLE_DEVICES=0, HF_HOME=f"{sam3d.WEIGHTS}/huggingface", HF_HUB_OFFLINE=1,
                                   LIDRA_SKIP_INIT="true", TORCH_HOME=sam3d.TORCH_HUB, CUDA_HOME="/usr/local/cuda",
                                   SAM3D_MODEL_REVISION=cvo.SAM3D["modelRevision"])
            worker = sam3d.Pool([sam3d.SAM3D_PY, "-c", "from fast_report.sam3d import sam3d_worker; sam3d_worker()"], 1, [env], "sam3d")
        else:
            env = sam3d.worker_env(x7.RECGEN_PY, CUDA_VISIBLE_DEVICES=0, SPCONV_ALGO="native", HF_HUB_OFFLINE=1, **proc["env"])
            env["PYTHONPATH"] += os.pathsep + x7.RECGEN_DIR
            worker = sam3d.Pool([x7.RECGEN_PY, "-c", "from fast_report.recgen_fast import recgen_worker; recgen_worker()"], 1, [env], "rg-" + proc["name"])
        try:
            boot = worker.ready(1800)[0]
        except Exception as error:  # noqa: BLE001  a process that cannot boot is reported, the next one runs
            yield {"kind": "proc", "proc": proc["name"], "error": str(error)[-3000:]}
            worker.close()
            continue
        yield {"kind": "proc", "proc": proc["name"], "boot": boot, "boot_wall_s": round(time.time() - t, 1)}
        ids = [oid for ids in plan["objects"].values() for oid in ids]
        for a in proc["arms"]:
            try:
                yield one_arm(worker, proc["name"], a, ids, sam=sam)
            except Exception as error:  # noqa: BLE001
                yield {"kind": "arm", "proc": proc["name"], "arm": a["name"], "error": str(error)[-3000:]}
        sweep = proc.get("sweep")
        if sweep:  # views 1 .. n-1 of every object with >= min_views generation views (n itself: the arms above)
            many = [oid for oid in ids if len(objs[oid]["views"]) >= sweep["min_views"]]
            for name in sweep["arms"]:
                for n in range(1, max(len(objs[oid]["views"]) for oid in many)):
                    todo = [oid for oid in many if len(objs[oid]["views"]) > n]
                    try:
                        yield one_arm(worker, proc["name"], arm(name), todo, max_views=n) | {"kind": "sweep"}
                    except Exception as error:  # noqa: BLE001
                        yield {"kind": "sweep", "proc": proc["name"], "arm": name, "max_views": n, "error": str(error)[-3000:]}
        if proc.get("micro"):
            yield {"kind": "micro", "proc": proc["name"], **worker.submit({"op": "micro"}).result()}
        worker.close()
    if plan.get("mps"):
        yield from mps_block(plan, objs, vram)
    cpu.close()
    vram.stop()
    yield {"kind": "done", "entered_unix": entered, "done_unix": time.time(), "usd_estimate": round((time.time() - entered) * USD_PER_S, 3)}


def mps_block(plan, objs, vram):
    """Resident RecGen processes on GPU 0 under MPS: for each arm and P in procs, `calls` objects (the benchmark objects in
    order, cycled) from one shared queue to the first P processes after one untimed call each -> wall, objects/s, per call."""
    import os
    from fast_report import recgen_fast as rf, sam3d, x7
    m = plan["mps"]
    env = sam3d.worker_env(x7.RECGEN_PY, CUDA_VISIBLE_DEVICES=0, SPCONV_ALGO="native", HF_HUB_OFFLINE=1, **m["env"])
    env["PYTHONPATH"] += os.pathsep + x7.RECGEN_DIR
    t = time.time()
    workers = [sam3d.Pool([x7.RECGEN_PY, "-c", "from fast_report.recgen_fast import recgen_worker; recgen_worker()"], 1, [env], f"mps{i}")
               for i in range(max(m["procs"]))]
    yield {"kind": "mps_boot", "boots": [w.ready(1800)[0] for w in workers], "boot_wall_s": round(time.time() - t, 1), "vram_idle_gib": vram.now()}
    ids = [oid for ids in plan["objects"].values() for oid in ids]
    for name in m["arms"]:
        change, seed = ARMS[name]
        s = rf.setting(**change)
        message = lambda oid: {"op": "run", "views": objs[oid]["views"], "seed": seed, "setting": s}  # noqa: E731
        for n in m["procs"]:
            active = workers[:n]
            warm = [w.submit(message(ids[0])) for w in active]
            for f in warm:
                f.result()
            todo, done, lock = queue.Queue(), [], threading.Lock()
            for i in range(m["calls"]):
                todo.put(ids[i % len(ids)])

            def drain(w):
                while True:
                    try:
                        oid = todo.get_nowait()
                    except queue.Empty:
                        return
                    t = time.time()
                    try:
                        r = w.submit(message(oid)).result()
                        row = {"id": oid, "views": r["views"], "seconds": round(r["seconds"], 3), "stages": r["stages"], "max_reserved_gib": r["max_reserved_gib"]}
                    except Exception as error:  # noqa: BLE001
                        row = {"id": oid, "error": str(error)[-800:]}
                    with lock:
                        done.append({**row, "round_trip_s": round(time.time() - t, 3)})
            threads = [threading.Thread(target=drain, args=(w,)) for w in active]
            a = time.perf_counter()
            for th in threads:
                th.start()
            for th in threads:
                th.join()
            b = time.perf_counter()
            yield {"kind": "mps", "arm": name, "procs": n, "calls": len(done), "errors": sum("error" in d for d in done), "wall_s": round(b - a, 2),
                   "objects_per_s": round(len(done) / (b - a), 4), "per_call_s": stats(d.get("seconds") for d in done), "per_call": done,
                   "gpu_peak_gib": vram.window(a, b)}
    for w in workers:
        w.close()


# ---------------------------------------------------------------- local
def expected():
    """X7's sep5 records of the benchmark objects: generation frames, held-out frame, RecGen's views (anchor first)."""
    out = {}
    for name, ids in OBJECTS.items():
        rec = {R["id"]: R for R in json.loads((X7_RUN / f"{name}-sep5.json").read_text())["objects"]}
        for oid in ids:
            s = rec[oid]["selection"]
            out[oid] = {"gen": [g["frame"] for g in s["gen"]], "held": s["held"]["frame"], "recgen_views": s["recgen_views"]}
    return out


def jpegs_npz(path, tiles):
    np.savez(path, **{k: np.frombuffer(v, np.uint8) for k, v in tiles.items()})


@app.local_entrypoint()
def main(plan: str = "smoke", out: str = "", mps_arms: str = "", mps_env: str = ""):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    for d in ("arms", "tiles"):
        (out / d).mkdir()
    p = plans(plan)
    p["expected"] = expected()
    p["token"] = f"{out.name}-{time.time_ns()}"
    if p.get("mps") and mps_arms:
        p["mps"]["arms"] = mps_arms.split(",")
    if p.get("mps") and mps_env:
        p["mps"]["env"] = {"xformers": XFORMERS, "sdpa": SDPA, "flash_attn": FLASH}[mps_env]
    (out / "plan.json").write_text(json.dumps({**p, "arms_defined": ARMS}, indent=1, default=str))
    called = time.time()
    for e in bench.remote_gen(p):
        kind = e["kind"]
        if kind in ("arm", "sweep") and "records" in e:
            stem = f"{e['proc']}--{e['arm']}" + (f"--v{e['max_views']}" if e.get("max_views") else "")
            jpegs_npz(out / "tiles" / f"{stem}.npz", e.pop("tiles"))
            (out / "arms" / f"{stem}.json").write_text(json.dumps(e, indent=1, default=str))
            sec = stats(r.get("seconds") for r in e["records"].values())
            acc = sum(bool((r.get("gate") or {}).get("accepted_source_consistency")) for r in e["records"].values())
            ch = stats(r.get("chamfer_rel") for r in e["records"].values())
            print(f"{stem}: s/object {sec and sec['median']} (mean {sec and sec['mean']}), accepted {acc}/{len(e['records'])}, "
                  f"chamfer_rel median {ch and ch['median']}, arm wall {e['arm_wall_s']} s", flush=True)
        elif kind == "objects":
            jpegs_npz(out / "tiles" / "held.npz", e.pop("held_tiles"))
            (out / "objects.json").write_text(json.dumps(e, indent=1, default=str))
            print("objects:", [(o["id"], len(o["recgen_views"]), o["matches_x7"]) for o in e["objects"]], flush=True)
        elif kind == "mps":
            (out / f"mps-{e['arm']}-p{e['procs']}.json").write_text(json.dumps(e, indent=1, default=str))
            print(f"mps {e['arm']} x{e['procs']}: {e['calls']} calls in {e['wall_s']} s = {e['objects_per_s']} objects/s, "
                  f"per call median {(e['per_call_s'] or {}).get('median')} s, errors {e['errors']}, GPU peak {e['gpu_peak_gib']} GiB", flush=True)
        else:
            (out / f"{kind}-{e.get('proc', '')}.json".replace("-.json", ".json")).write_text(json.dumps(e, indent=1, default=str))
            print(kind, json.dumps(e, default=str)[:1500], flush=True)
    (out / "client.json").write_text(json.dumps({"client_called_unix": called, "client_done_unix": time.time()}, indent=1))


STAGE_ORDER = ("preprocess", "cond_ss", "ss_flow", "ss_decode", "cond_slat", "slat_flow", "mesh_decode", "gs_decode", "mesh_out")
ok = lambda r: bool((r.get("gate") or {}).get("accepted_source_consistency"))  # noqa: E731


def stage_medians(rs):
    rs = [r for r in rs if "stages" in r]
    return {k: round(float(np.median([r["stages"].get(k, 0.) for r in rs])), 3) for k in STAGE_ORDER} if rs else None


def arm_row(e, base):
    """One arm against a base arm on the same objects: time per object (total, by stage), held-out acceptance and its change,
    Chamfer to the default mesh."""
    recs, brecs = e["records"], base["records"]
    ids = [o for o in recs if o in brecs]
    both = [o for o in ids if recs[o].get("seconds") is not None and brecs[o].get("seconds") is not None]
    t, bt = [recs[o]["seconds"] for o in both], [brecs[o]["seconds"] for o in both]
    return {"objects": len(ids), "s_per_object": stats(recs[o].get("seconds") for o in ids),
            "speedup_of_mean": round(float(np.mean(bt) / np.mean(t)), 2) if both else None,
            "stages_median": stage_medians(recs[o] for o in ids), "accepted": sum(ok(recs[o]) for o in ids),
            "accepted_base": sum(ok(brecs[o]) for o in ids), "gained": [o for o in ids if ok(recs[o]) and not ok(brecs[o])],
            "lost": [o for o in ids if ok(brecs[o]) and not ok(recs[o])],
            "iou": stats((recs[o].get("gate") or {}).get("silhouette_iou") for o in ids),
            "chamfer_rel": stats(recs[o].get("chamfer_rel") for o in ids), "chamfer_cm": stats(recs[o].get("chamfer_cm") for o in ids),
            "max_reserved_gib": max((recs[o].get("max_reserved_gib") or 0 for o in ids), default=None),
            "errors": [o for o in ids if "judge_error" in recs[o]]}


def summarize(run_dir):
    """The report's tables from run_dir/arms/*.json -> run_dir/summary.json. Base of every arm: its own process's default
    (backend processes) or X7's call in X7's process (xformers--default)."""
    run_dir = Path(run_dir)
    ev = {p.stem: json.loads(p.read_text()) for p in sorted((run_dir / "arms").glob("*.json"))}
    ev = {k: e for k, e in ev.items() if "records" in e}
    ref = ev.get("xformers--default") or next(iter(ev.values()))  # a run without the default: its first arm
    out = {"base": ref["arm"], "arms": {}, "arms_vs_own_process_default": {}, "per_views_default": {}, "sweep": {}}
    for stem, e in ev.items():
        if e.get("kind") == "sweep":
            continue
        out["arms"][stem] = arm_row(e, ref)
        own = ev.get(f"{e['proc']}--default")
        if own and e["proc"] != "xformers":
            out["arms_vs_own_process_default"][stem] = arm_row(e, own)
    by = {}
    for r in ref["records"].values():
        by.setdefault(r["views"], []).append(r)
    for n, rs in sorted(by.items()):
        out["per_views_default"][str(n)] = {"objects": len(rs), "s_per_object": stats(r["seconds"] for r in rs), "stages_median": stage_medians(rs),
                                            "voxels": stats(r.get("voxels") for r in rs), "faces": stats(r.get("faces_n") for r in rs)}
    out["export"] = {"s": stats(r.get("export_s") for r in ref["records"].values()),
                     "files": next((r["export_files"] for r in ref["records"].values() if r.get("export_files")), None)}
    for stem, e in ev.items():  # views 1 .. n-1 (sweep events) against all n views (the arm's own records, same objects)
        if e.get("kind") != "sweep":
            continue
        full = ev[f"{e['proc']}--{e['arm']}"]["records"]
        rows = {f"{e['arm']}@{e['max_views']}v": e["records"], f"{e['arm']}@all(objects with >{e['max_views']}v)": {o: full[o] for o in e["records"]}}
        for key, rs in rows.items():
            out["sweep"][key] = {"objects": len(rs), "views": stats(r.get("views") for r in rs.values()), "s_per_object": stats(r.get("seconds") for r in rs.values()),
                                 "stages_median": stage_medians(rs.values()), "accepted": sum(ok(r) for r in rs.values()),
                                 "chamfer_rel": stats(r.get("chamfer_rel") for r in rs.values())}
    out["colour"] = colour_table(run_dir, [k for k, e in ev.items() if e.get("kind") != "sweep" and (run_dir / "tiles" / f"{k}.npz").exists()],
                                 base=f"{ref['proc']}--{ref['arm']}")
    for p in sorted(run_dir.glob("*.json")):
        if p.stem.startswith(("micro", "boot", "done", "proc")):
            out[p.stem] = json.loads(p.read_text())
    (run_dir / "summary.json").write_text(json.dumps(out, indent=1, default=str))
    return out


BACKGROUND = 40  # x7.render_tile's background grey


def rendered(tile):
    """Pixels of a held-out render tile that show the model (differ from the flat background grey after JPEG)."""
    return np.abs(tile.astype(np.int16) - BACKGROUND).max(-1) > 12


def colour_errors(crop, tile, base=None):
    """Appearance on the held-out view (the gate sees only geometry): mean |RGB| difference (0-255) of the model's rendered
    pixels against the held-out photo crop, and against a base model's render where both show a model (same camera, crop)."""
    m = rendered(tile)
    out = {"vs_photo": round(float(np.abs(tile.astype(np.int16) - crop.astype(np.int16))[m].mean()), 2) if m.sum() >= 50 else None}
    if base is not None:
        both = m & rendered(base)
        out["vs_default"] = round(float(np.abs(tile.astype(np.int16) - base.astype(np.int16))[both].mean()), 2) if both.sum() >= 50 else None
    return out


def colour_table(run_dir, stems, base="xformers--default"):
    """colour_errors for every object of each arm -> {stem: {"vs_photo": stats, "vs_default": stats}}."""
    import cv2
    run_dir = Path(run_dir)
    held = np.load(run_dir / "tiles" / "held.npz")
    dec = lambda z, k: cv2.imdecode(z[k], cv2.IMREAD_COLOR)  # noqa: E731
    b = np.load(run_dir / "tiles" / f"{base}.npz")
    out = {}
    for stem in stems:
        t = np.load(run_dir / "tiles" / f"{stem}.npz")
        rows = [colour_errors(dec(held, k), dec(t, k), dec(b, k) if k in b.files else None) for k in t.files if k in held.files]
        out[stem] = {"vs_photo": stats(r["vs_photo"] for r in rows), "vs_default": stats(r.get("vs_default") for r in rows)}
    return out


X7_VIEW_MIX = {1: 76, 2: 30, 3: 7, 4: 8}  # RecGen's generation views per object over X7's three sep5 runs (121 objects)
GATE_TAIL_S = 2.  # the last object's held-out judge on the CPU pool (X7: 1.2-2.1 s median; here 0.7-3.4 s)


def timing_model(arm_records, mps_rows, gpus=2, counts=(20, 40, 70)):
    """Wall seconds from the first object handed to RecGen to the last one judged, for N objects on `gpus` GPUs with P
    processes each under MPS: the mean single-process seconds per object on X7's view mix (per view count, from the arm's
    records) x the measured MPS latency factor at P, in ceil(N / (gpus P)) rounds, + one judge. ponytail: rounds of equal
    objects; the real queue interleaves sizes, so this is within one object's time of a greedy schedule."""
    import math
    by = {}
    for r in arm_records.values():
        if r.get("seconds") is not None:
            by.setdefault(r["views"], []).append(r["seconds"])
    per_view = {n: float(np.mean(v)) for n, v in by.items()}
    mix = sum(X7_VIEW_MIX[n] * per_view.get(n, per_view[max(per_view)]) for n in X7_VIEW_MIX) / sum(X7_VIEW_MIX.values())
    one = next(r for r in mps_rows if r["procs"] == 1)
    out = {"s_per_object_x7_mix_one_process": round(mix, 2), "per_view_mean_s": {str(k): round(v, 2) for k, v in sorted(per_view.items())},
           "by_procs": {}}
    for r in mps_rows:
        p, gain = r["procs"], r["objects_per_s"] / one["objects_per_s"]
        latency = p * mix / gain  # each of p sharing processes takes this long per object
        out["by_procs"][str(p)] = {"throughput_gain_vs_1": round(gain, 2), "latency_s": round(latency, 2),
                                   "objects_per_s_2gpu": round(gpus * p / latency, 3),
                                   "wall_s": {str(n): round(math.ceil(n / (gpus * p)) * latency + GATE_TAIL_S, 1) for n in counts}}
    return out


def contact_sheet(run_dir, columns, path, ids=None, tile=128):
    """Per object: held-out crop | each column's model rendered from the held-out camera (X7's render_tile), with the gate
    mark (+ accepted, - rejected) and IoU; columns: [(label, arm file stem)]."""
    import cv2
    run_dir = Path(run_dir)
    held = np.load(run_dir / "tiles" / "held.npz")
    arms = {stem: (json.loads((run_dir / "arms" / f"{stem}.json").read_text())["records"], np.load(run_dir / "tiles" / f"{stem}.npz"))
            for _, stem in columns}
    rows = []
    for o in json.loads((run_dir / "objects.json").read_text())["objects"]:
        if ids and o["id"] not in ids:
            continue
        cells, notes = [cv2.imdecode(held[o["id"]], cv2.IMREAD_COLOR)], [f"{o['id']} {len(o['recgen_views'])}v"]
        for label, stem in columns:
            recs, tiles = arms[stem]
            g = recs.get(o["id"], {}).get("gate") or {}
            cells.append(cv2.imdecode(tiles[o["id"]], cv2.IMREAD_COLOR) if o["id"] in tiles.files else np.full((tile, tile, 3), 235, np.uint8))
            notes.append(f"{label} {'+' if g.get('accepted_source_consistency') else '-'}{g.get('silhouette_iou') or 0:.2f}")
        strip = np.hstack([cv2.resize(c, (tile, tile)) for c in cells])
        head = np.full((16, strip.shape[1], 3), 255, np.uint8)
        for i, n in enumerate(notes):
            cv2.putText(head, n[:22], (3 + i * tile, 12), cv2.FONT_HERSHEY_SIMPLEX, .33, (0, 0, 0), 1, cv2.LINE_AA)
        rows.append(np.vstack([head, strip, np.full((4, strip.shape[1], 3), 255, np.uint8)]))
    top = np.full((22, rows[0].shape[1], 3), 255, np.uint8)
    cv2.putText(top, "held-out crop | " + " | ".join(label for label, _ in columns) + "   (+/- held-out gate, IoU)", (4, 15),
                cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), np.vstack([top, *rows]), [cv2.IMWRITE_JPEG_QUALITY, 80])
    return len(rows)


def results(runs):
    """The report's numbers from recgen-fast-002..005 -> runs/recgen-fast-results/results.json: the seed-pooled comparison of
    FAST against the default (3 seeds each), SAM 3D s1cfg12, MPS and the 2-GPU timing model, compile, spend (list-price estimates)."""
    load = lambda run, stem: json.loads((runs / run / "arms" / f"{stem}.json").read_text())["records"]  # noqa: E731
    dflt = [load("recgen-fast-003", "xformers--default"), load("recgen-fast-002", "xformers--default_s43"), load("recgen-fast-003", "xformers--default_s44")]
    fast = [load("recgen-fast-003", f"xformers--{a}") for a in ("rec", "rec_s43", "rec_s44")]
    ids = list(dflt[0])
    iou = lambda R: np.array([(R[o].get("gate") or {}).get("silhouette_iou", 0.) for o in ids])  # noqa: E731
    acc = lambda R: np.array([ok(R[o]) for o in ids])  # noqa: E731
    d = np.mean([iou(r) for r in fast], 0) - np.mean([iou(r) for r in dflt], 0)
    pd, pf = np.mean([acc(r) for r in dflt], 0), np.mean([acc(r) for r in fast], 0)
    ch = lambda R: stats(R[o].get("chamfer_rel") for o in ids)  # noqa: E731
    sam = load("recgen-fast-003", "sam3d--s1cfg12")
    mps = {arm: [json.loads((runs / "recgen-fast-004" / f"mps-{arm}-p{n}.json").read_text()) for n in (1, 2, 3)] for arm in ("default", "rec")}
    comp = {k: load("recgen-fast-005", k) for k in ("xformers--rec", "sdpa_compile--rec")}
    spend = {}
    for run in ("recgen-fast-001", "recgen-fast-003", "recgen-fast-004", "recgen-fast-005"):
        spend[run] = json.loads((runs / run / "done.json").read_text())["usd_estimate"]
    spend["recgen-fast-002"] = round((75.7 * 60) * USD_PER_S, 3)  # preempted: 16:37:04-17:41:42, restarted 17:41:47-17:52:55 (stopped)
    out = {"fast_setting": RF_FAST, "objects": len(ids),
           "seed_pooled": {"iou_fast_minus_default": {"mean": round(float(d.mean()), 4), "se": round(float(d.std(ddof=1) / np.sqrt(len(d))), 4)},
                           "accepted_per_seed": {"default_s42_s43_s44": [int(acc(r).sum()) for r in dflt], "fast_s42_s43_s44": [int(acc(r).sum()) for r in fast]},
                           "expected_accepted": {"default": round(float(pd.sum()), 2), "fast": round(float(pf.sum()), 2)},
                           "objects_moving": {o: [round(float(x), 2), round(float(y), 2)] for o, x, y in zip(ids, pd, pf) if abs(x - y) >= .5},
                           "chamfer_rel_to_default_s42": {"default_s43": ch(dflt[1]), "default_s44": ch(dflt[2]), "fast_s42": ch(fast[0]),
                                                          "fast_s43": ch(fast[1]), "fast_s44": ch(fast[2])},
                           "s_per_object_pcie": {"default": [stats(r[o]["seconds"] for o in ids)["mean"] for r in (dflt[0], dflt[2])],
                                                 "fast": [stats(r[o]["seconds"] for o in ids)["mean"] for r in fast]}},
           "sam3d_s1cfg12": {"accepted": int(acc(sam).sum()), "prepare_rejected": sum("prepare_rejected" in sam[o] for o in ids),
                             "s_per_object": stats(sam[o].get("seconds") for o in ids), "iou_generated": stats((sam[o].get("gate") or {}).get("silhouette_iou") for o in ids)},
           "mps_sxm4": {arm: [{k: r[k] for k in ("procs", "wall_s", "objects_per_s", "per_call_s", "gpu_peak_gib")} for r in rows] for arm, rows in mps.items()},
           "timing_model_2x_a100_sxm4": {arm: timing_model({f"c{i}": r for i, r in enumerate(rows[0]["per_call"])}, rows) for arm, rows in mps.items()},
           "compile_ss": {k: stats(r[o]["seconds"] for o in ids) for k, r in comp.items()},
           "spend_usd_list_price": {**spend, "total": round(sum(spend.values()), 2)}}
    dst = runs / "recgen-fast-results"
    dst.mkdir(exist_ok=True)
    (dst / "results.json").write_text(json.dumps(out, indent=1, default=str))
    return out


def self_check():
    assert sum(map(len, OBJECTS.values())) == 19 and len({o for ids in OBJECTS.values() for o in ids}) == 19
    from fast_report import recgen_fast as rf
    for name, (change, seed) in ARMS.items():
        rf.setting(**change)  # every arm is a valid setting
        assert seed in (42, 43, 44), name
    assert plans("main")["procs"][0]["arms"][0]["reference"] and all(not a["reference"] for p in plans("main")["procs"] for a in p["arms"][1:])
    assert RF_FAST == rf.FAST, "the bench measures recgen_fast.FAST"
    exp = expected()
    assert set(exp) == {o for ids in OBJECTS.values() for o in ids}
    counts = {}
    for oid, e in exp.items():
        counts[len(e["recgen_views"])] = counts.get(len(e["recgen_views"]), 0) + 1
        assert e["held"] not in e["gen"] and e["recgen_views"][0] == e["gen"][0], oid
    assert counts == {1: 6, 2: 5, 3: 4, 4: 4}, counts
    # timing model: 1 view 4 s, 2 views 6 s -> X7 mix mean (76*4 + (30+7+8)*6) / 121; 2 processes at 1.5x throughput
    recs = {"a": {"views": 1, "seconds": 4.}, "b": {"views": 2, "seconds": 6.}}
    m = timing_model(recs, [{"procs": 1, "objects_per_s": .2}, {"procs": 2, "objects_per_s": .3}])
    mix = (76 * 4 + 45 * 6) / 121
    assert abs(m["s_per_object_x7_mix_one_process"] - round(mix, 2)) < 1e-9, m
    two = m["by_procs"]["2"]
    assert two["throughput_gain_vs_1"] == 1.5 and abs(two["latency_s"] - round(2 * mix / 1.5, 2)) < .01
    assert two["wall_s"]["20"] == round(5 * (2 * mix / 1.5) + GATE_TAIL_S, 1), two  # 20 objects on 2 x 2 processes: 5 rounds
    # colour: a grey model on a grey photo -> 0 against the photo; the same render against itself -> 0; one shifted by 30 -> 30
    crop = np.full((64, 64, 3), 120, np.uint8)
    tile = np.full((64, 64, 3), BACKGROUND, np.uint8)
    tile[16:48, 16:48] = 120
    shifted = tile.copy()
    shifted[16:48, 16:48] = 150
    assert colour_errors(crop, tile, tile) == {"vs_photo": 0., "vs_default": 0.} and colour_errors(crop, shifted, tile) == {"vs_photo": 30., "vs_default": 30.}
    print("recgen_fast_bench self-check passed: arms, plans, X7's expected selection (6/5/4/4 objects with 1/2/3/4 views), timing model, colour")


def print_tables(s):
    short = {"cond_ss": "condSS", "ss_flow": "SS", "cond_slat": "condSL", "slat_flow": "SLAT", "mesh_decode": "mesh", "gs_decode": "GS",
             "mesh_out": "out"}
    print("| arm | s/object mean (median) | x default | " + " | ".join(short.values()) + " | accepted (default) | +/- | IoU mean | "
          "Chamfer % diag median (p90) | cm median | colour vs default | colour vs photo |")
    print("|---" * (10 + len(short)) + "|")
    for stem, r in s["arms"].items():
        st, c = r["stages_median"] or {}, s["colour"].get(stem, {})
        pct = lambda x: None if x is None else round(100 * x, 2)  # noqa: E731
        print(f"| {stem} | {r['s_per_object']['mean']} ({r['s_per_object']['median']}) | {r['speedup_of_mean']} | "
              + " | ".join(str(st.get(k, "")) for k in short) + f" | {r['accepted']} ({r['accepted_base']}) | +{len(r['gained'])}/-{len(r['lost'])} | "
              f"{(r['iou'] or {}).get('mean')} | {pct((r['chamfer_rel'] or {}).get('median'))} ({pct((r['chamfer_rel'] or {}).get('p90'))}) | "
              f"{(r['chamfer_cm'] or {}).get('median')} | {(c.get('vs_default') or {}).get('mean')} | {(c.get('vs_photo') or {}).get('mean')} |")
    print("\nper number of views (default, X7's view counts):")
    for n, r in s["per_views_default"].items():
        print(f"  {n} views: n={r['objects']} s/object mean {r['s_per_object']['mean']} stages {r['stages_median']} voxels median {r['voxels']['median']}")
    print("\nsweep (the multi-view objects with fewer views):")
    for k, r in s["sweep"].items():
        print(f"  {k}: n={r['objects']} s/object mean {(r['s_per_object'] or {}).get('mean')} accepted {r['accepted']} "
              f"chamfer_rel median {(r['chamfer_rel'] or {}).get('median')} stages {r['stages_median']}")
    print("\nexport (RecGenResult.save):", s.get("export"))


if __name__ == "__main__":
    if sys.argv[1:2] == ["--self-check"]:
        self_check()
    elif sys.argv[1:2] == ["--summarize"]:
        print_tables(summarize(sys.argv[2]))
    elif sys.argv[1:2] == ["--results"]:
        print(json.dumps(results(Path(sys.argv[2])), indent=1)[:6000])
    else:
        raise SystemExit(__doc__)
