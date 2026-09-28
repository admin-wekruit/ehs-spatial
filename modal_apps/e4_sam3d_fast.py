"""E4 of docs/phase2/FAST-PATH-PLAN.md: SAM 3D Objects fast mode against today's settings on 30 ME340 objects.

The 30 objects are the first 30 that runs/me340-object-models-241-sam3d modelled (its priority order: EHS equipment
first). Every arm is judged by the unchanged gate of scripts/complete_video_objects.py (read-back only: its own
prepare, assess and try order, no --invoke); only what sits in the SAM 3D journal differs between arms.

  arms      baseline  241's own journaled meshes (every try it made), re-judged here on Linux
            current   upstream defaults, what 241 ran: stage 1 25 steps + CFG 7, stage 2 25 steps + CFG 5, rerun here
            fast      stage 1 shortcut (use_stage1_distillation) 4 steps, stage 2 12 steps
            fast-s2d  fast, and stage 2 shortcut (use_stage2_distillation) at 4 steps too
            fast-crop fast on the source-grid crop and its own K instead of the whole frame (attempt 1 only)
            every other arm runs every try the runner would make: up to 4 views, then the best view with the extra seed
  GPU       one A100-80GB container; the parent never touches CUDA, each SAM 3D process is a subprocess:
            A 1 process (all arms, then a 20-call throughput block), B 2 processes (same 20 calls), C 2 processes under MPS

State lives in the Modal volume panoptes-e4-sam3d, mounted at the phase2 path so every path matches the Mac's.
Results land in runs/m3-exp-e4-sam3d-1/. Each step is one ephemeral run (never modal deploy):

  modal run modal_apps/e4_sam3d_fast.py::upload           # the 241 inputs of the 30 objects and their SAM 3D journal
  modal run modal_apps/e4_sam3d_fast.py::inputs_step           # CPU x30: prepare() -> every try's SAM 3D input, journal key check
  modal run modal_apps/e4_sam3d_fast.py::bench_step [--smoke]   # GPU: the arms, 1 vs 2 processes, MPS
  modal run modal_apps/e4_sam3d_fast.py::gate_step              # CPU x30: the gate's assess() on every try of every arm
  python modal_apps/e4_sam3d_fast.py summary              # tables -> summary.json (local, no Modal)
  python modal_apps/e4_sam3d_fast.py self-check
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import modal

sys.path[:0] = [str(Path(__file__).parent), "/repo", "/repo/scripts", "/repo/modal_apps"]
import sam3d_research  # noqa: E402  the pinned image, weights volume and revisions

ROOT = Path(__file__).resolve().parents[1]
ART = Path("/Users/adam/Desktop/panoptes-public/research-notes/phase2")
E4, SEED = ART / "e4", ART / "e4" / "seed241"
LOCAL_OUT = ART / "runs" / "m3-exp-e4-sam3d-1"
SEED_RUN = ART / "runs" / "me340-object-models-241-sam3d"
ENTITIES = ("object-019 object-035 object-001 object-017 object-106 object-012 object-022 object-120 object-073 object-018 "
            "object-131 object-009 object-148 object-130 object-105 object-039 object-127 object-007 object-043 object-099 "
            "object-109 object-189 object-094 object-182 object-104 object-034 object-033 object-023 object-026 object-036").split()
FAST = {"use_stage1_distillation": True, "stage1_inference_steps": 4, "stage2_inference_steps": 12}
ARMS = {"current": {}, "fast": FAST, "fast-crop": FAST,
        "fast-s2d": {**FAST, "use_stage2_distillation": True, "stage2_inference_steps": 4}}
GATED = ("baseline", "fast", "fast-s2d", "current", "fast-crop")
STAGES = ("compute_pointmap", "preprocess_image", "sample_sparse_structure", "pose_decoder", "sample_slat", "decode_slat", "postprocess_slat_output")
GPU_USD_S = .000694 + 8 * .0000131 + 64 * .00000222  # Modal list price: A100-80GB + 8 cores + 64 GiB
CPU_USD_S = 2 * .0000131 + 8 * .00000222              # 2 cores + 8 GiB
TP_CALLS = 20

app = modal.App("panoptes-e4-sam3d-fast")
data = modal.Volume.from_name("panoptes-e4-sam3d", create_if_missing=True)
gate_image = (modal.Image.debian_slim(python_version="3.12")
              .apt_install("libgl1", "libgomp1", "libglib2.0-0", "libegl1", "libx11-6", "libxext6", "libxrender1", "libsm6")
              .pip_install("numpy==2.5.1", "opencv-python-headless==5.0.0.93", "scipy==1.18.0", "trimesh==5.1.0", "open3d==0.19.0",
                           "pydantic==2.13.4", "pillow==12.3.0")  # the Mac venv's versions of what the gate imports
              .add_local_dir(ROOT / "scripts", "/repo/scripts", ignore=["**/__pycache__/**"])
              .add_local_dir(ROOT / "modal_apps", "/repo/modal_apps", ignore=["**/__pycache__/**"])
              .add_local_dir(ROOT / "ehs_spatial", "/repo/ehs_spatial", ignore=["**/__pycache__/**"]))
gpu_image = sam3d_research.image.add_local_python_source("sam3d_research")  # a mount: the pinned image's layers stay cached


# ---------------------------------------------------------------- the gate script, driven as the report runner drives it
def cvo_main(cvo, eid, out):
    """complete_video_objects.main() with 241's arguments for one explicit entity (read-back only: no --invoke)."""
    runs = ART / "runs"
    sys.argv = ["complete_video_objects.py", "--droid-run", str(runs / "droid-me340-165-171"),
                "--depth-run", str(runs / "da3-posed-me340-189-fused-dynamic"), "--object-map", str(runs / "me340-entity-names-200"),
                "--masks", str(runs / "me340-masks-194"), "--dynamic-masks", str(runs / "me340-dynamic-masks-188/masks"),
                "--clip", str(ART / "data/clips/me340-165"), "--output", str(out), "--generator", "sam3d",
                "--skip-frames", "14-225", "--entities", eid]
    shutil.copytree(SEED / "resegmented", out / "resegmented", dirs_exist_ok=True)  # 241's SAM 2 re-segmentations: never asked again
    cvo.main()


def sam3d_key(cvo, payload):
    """The journal folder name sam3d_obtain gives this input (its own formula, over the whole-frame input)."""
    import numpy as np
    view = payload["views"][0]
    rgb, mask, depth = (np.ascontiguousarray(np.asarray(view[n])) for n in ("fullRgb", "fullMask", "fullDepth"))
    return hashlib.sha256(b"".join([json.dumps({"seed": payload["seed"], **{k: cvo.SAM3D[k] for k in ("modelRevision", "codeRevision")}}).encode(),
                                    rgb.tobytes(), mask.astype(bool).tobytes(), depth.astype(np.float32).tobytes()])).hexdigest()[:24]


@app.function(image=gate_image, cpu=2, memory=8192, timeout=1800, retries=0, volumes={str(ART): data})
def inputs(eid):
    """Every try of one object as the gate would send it (the best views, then the best view with the extra seed): the
    whole-frame input SAM 3D gets today and the source-grid crop, saved for the GPU; 241's mesh for each try it made
    is filed under this run's journal key (by key, else by observation and seed)."""
    import numpy as np
    import complete_video_objects as cvo
    started, captured = time.time(), {}
    cvo.run = lambda args: captured.setdefault("args", args)  # main() parses and sets the module constants; prepare runs here
    out = Path("/tmp/e4") / eid
    cvo_main(cvo, eid, out)
    clip, rows, document, chosen, rejected = cvo.prepare(captured["args"])
    if not chosen:
        return {"eid": eid, "rejected": rejected, "seconds": time.time() - started}
    c = chosen[0]
    tries = list(zip(c["tries"], c["payloads"])) + [(c["tries"][0], {**c["payloads"][0], "seed": cvo.EXTRA_SEED})]  # run()'s order
    seeded = {}
    for folder in (SEED / "journal-sam3d" / eid).glob("sam3d-*"):
        d = json.loads((folder / "dispatch.json").read_text())
        seeded[d["observation"], d["seed"]] = folder
    listing = []
    for n, (view, payload) in enumerate(tries, 1):
        v, key = payload["views"][0], sam3d_key(cvo, payload)
        path = E4 / "inputs" / eid / f"a{n}-{key}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, seed=payload["seed"], rgb=v["fullRgb"], mask=np.asarray(v["fullMask"], bool),
                            pointmap=cvo.sam3d_pointmap(v["fullDepth"], np.asarray(v["fullK"], float)),
                            crop_rgb=v["rgb"], crop_mask=np.asarray(v["mask"], bool), crop_pointmap=cvo.sam3d_pointmap(v["depth"], np.asarray(v["K"], float)))
        source = SEED / "journal-sam3d" / eid / f"sam3d-{key}"
        how = "key" if (source / "output.npz").exists() else None
        if how is None and (view["observation"], payload["seed"]) in seeded:
            source, how = seeded[view["observation"], payload["seed"]], "observation+seed"
        if how:
            target = E4 / "arms" / "baseline" / "journal-sam3d" / eid / f"sam3d-{key}"
            target.mkdir(parents=True, exist_ok=True)
            for name in ("output.npz", "record.json", "dispatch.json"):
                shutil.copy2(source / name, target / name)
        listing.append({"eid": eid, "attempt": n, "frame": view["frame"], "observation": view["observation"], "seed": payload["seed"],
                        "key": key, "input": str(path), "baseline": how, "cropShape": list(np.shape(v["rgb"])[:2])})
    data.commit()
    return {"eid": eid, "tries": listing, "seededTries": len(seeded), "seconds": time.time() - started}


@app.function(image=gate_image, cpu=2, memory=8192, timeout=3600, retries=0, volumes={str(ART): data})
def gate(eid, arms):
    """The gate's own assess() on every try of one object in every arm. The runner stops at the first accepted try, so
    its outcome is the first accepted try in this order; judging the rest too gives per-try rates. Inputs come from the
    gate's own prepare(), meshes from its own sam3d_obtain() reading each arm's journal (no --invoke: nothing generated)."""
    import complete_video_objects as cvo
    captured, started = {}, time.time()
    cvo.run = lambda args: captured.setdefault("args", args)  # main() parses and sets the module constants
    cvo_main(cvo, eid, Path("/tmp/e4") / eid)
    args = captured["args"]
    args.metres_per_native = json.loads((args.depth_run / "metric-scale.json").read_text())["metres_per_native_unit"]  # as run()
    args.inflight = 0
    clip, rows, _, chosen, rejected = cvo.prepare(args)
    result = {"eid": eid, "prepareSeconds": round(time.time() - started, 1), "rejected": rejected, "tries": []}
    tries = list(zip(chosen[0]["tries"], chosen[0]["payloads"])) if chosen else []
    tries += [(tries[0][0], {**tries[0][1], "seed": cvo.EXTRA_SEED})] if tries else []  # run()'s order: views, then the extra seed
    for arm in arms:
        for n, (view, payload) in enumerate(tries, 1):
            mesh, record = cvo.sam3d_obtain(payload, E4 / "arms" / arm / "journal-sam3d" / eid, args)
            row = {"arm": arm, "attempt": n, "observation": view["observation"], "seed": payload["seed"], "generated": mesh is not None}
            if mesh is not None:
                t = time.time()
                v = cvo.assess(chosen[0]["entity"], view, payload["views"][0], mesh, record, rows, clip, args)["validation"]
                row.update(assessSeconds=round(time.time() - t, 1), accepted=v["accepted_source_consistency"], reasons=v["rejectionReasons"],
                           iou=v["silhouette_iou"], fitCm=v["fitResidualCm"], observedCoverage=v["observedCoverage"],
                           entityCoverage=v["entityCoverage"], triangles=v["triangles"], workerSeconds=v["gpuSeconds"])
            result["tries"].append(row)
            del mesh
    shutil.rmtree(Path("/tmp/e4") / eid, ignore_errors=True)
    return result


# ---------------------------------------------------------------- SAM 3D processes
def load_pipeline():
    """As sam3d_research.SAM3DObjects.load: mesh decoder only, internal depth disabled."""
    from huggingface_hub import snapshot_download
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    root = Path(snapshot_download("facebook/sam-3d-objects", revision=sam3d_research.MODEL_REVISION))
    settings = OmegaConf.load(root / "checkpoints/pipeline.yaml")
    settings.rendering_engine, settings.compile_model, settings.workspace_dir = "pytorch3d", False, str(root / "checkpoints")
    pipeline = instantiate(settings, depth_model=None, decode_formats=["mesh"], slat_decoder_gs_config_path=None, slat_decoder_gs_ckpt_path=None,
                           slat_decoder_gs_4_config_path=None, slat_decoder_gs_4_ckpt_path=None)

    def forbidden_depth(*args, **kwargs):
        raise RuntimeError("external pointmap required; internal depth disabled")
    pipeline.depth_model = forbidden_depth
    return pipeline


def run_once(pipeline, rgb, mask, pointmap, seed, options):
    """sam3d_research.SAM3DObjects.run with the arm's options passed to the upstream run(); same timing span."""
    import numpy as np
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


def save_output(folder, out, record):
    """The journal layout sam3d_obtain reads back (record.json seconds = this call's worker seconds)."""
    import numpy as np
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / "output.npz", **{k: out[k] for k in ("vertices", "faces", "colors", "objectToCamera")})
    (folder / "record.json").write_text(json.dumps(record))
    (folder / "dispatch.json").write_text(json.dumps({"entity": record["eid"], "observation": record["observation"], "seed": record["seed"], "arm": record["arm"]}))


def worker(spec_path):
    """One SAM 3D process: load, one warm-up call, wait at the barrier, run the jobs."""
    import torch
    spec = json.loads(Path(spec_path).read_text())
    t = time.time()
    pipeline = load_pipeline()
    load_seconds, log = time.time() - t, {}
    for name in STAGES:  # instance attributes shadow the methods run() calls; synchronised so a stage owns its kernels
        def timed(*a, _f=getattr(pipeline, name), _n=name, **k):
            torch.cuda.synchronize()
            s = time.monotonic()
            try:
                return _f(*a, **k)
            finally:
                torch.cuda.synchronize()
                log[_n] = log.get(_n, 0) + time.monotonic() - s
        setattr(pipeline, name, timed)
    results, pool = [], ThreadPoolExecutor(2)
    for phase in ("warmup", "jobs"):  # the first call pays one-off kernel set-up (~10 s): done before the barrier
        if phase == "jobs":
            Path(spec["ready"]).write_text("ready")
            while not Path(spec["go"]).exists():
                time.sleep(.02)
        run_jobs(spec, spec[phase], pipeline, log, results, pool)
    pool.shutdown(wait=True)
    Path(spec["result"]).write_text(json.dumps({"loadSeconds": load_seconds, "results": results, "gpu": torch.cuda.get_device_name(0),
                                                "maxAllocatedGB": torch.cuda.max_memory_allocated() / 1e9,
                                                "maxReservedGB": torch.cuda.max_memory_reserved() / 1e9}))


def run_jobs(spec, jobs, pipeline, log, results, pool):
    import traceback
    import numpy as np
    import torch

    for job in jobs:
        if time.time() > spec["deadline"]:
            results.append({**job, "skipped": "deadline"})
            continue
        arrays, prefix = np.load(job["input"]), "crop_" if job["arm"] == "fast-crop" else ""
        rgb, mask, pointmap, seed = arrays[prefix + "rgb"], arrays[prefix + "mask"], arrays[prefix + "pointmap"], int(arrays["seed"])
        log.clear()
        t0 = time.time()
        try:
            out = run_once(pipeline, rgb, mask, pointmap, seed, ARMS[job["arm"]])
        except Exception:
            results.append({**job, "error": traceback.format_exc()[-3000:]})
            continue
        record = {**{k: job.get(k) for k in ("eid", "attempt", "observation", "arm", "block")}, "seed": seed, "t0": t0, "t1": time.time(),
                  "seconds": out["seconds"], "workerSeconds": out["seconds"], "stages": {k: round(v, 3) for k, v in log.items()},
                  "input": "crop" if prefix else "full", "grid": list(rgb.shape[:2]), "triangles": int(len(out["faces"])),
                  "gpu": torch.cuda.get_device_name(0),
                  "pins": {"model": "facebook/sam-3d-objects", "modelRevision": sam3d_research.MODEL_REVISION,
                           "codeRevision": sam3d_research.CODE_REVISION, "options": ARMS[job["arm"]]}}
        results.append(record)
        if job.get("save"):
            pool.submit(save_output, Path(job["save"]), out, record)


def processes(name, job_lists, deadline, env=None, warmup=()):
    """Start one worker per job list, release them together once all have loaded, return their reports and the GPU's
    peak memory and mean utilisation (nvidia-smi, 1 s) over the timed block."""
    import threading
    folder = Path("/tmp/e4bench") / name
    folder.mkdir(parents=True, exist_ok=True)
    procs, specs = [], []
    for i, jobs in enumerate(job_lists):
        spec = {"jobs": jobs, "warmup": list(warmup), "deadline": deadline, "ready": str(folder / f"ready-{i}"), "go": str(folder / "go"), "result": str(folder / f"result-{i}.json")}
        (folder / f"spec-{i}.json").write_text(json.dumps(spec))
        log = open(folder / f"log-{i}.txt", "w")
        procs.append(subprocess.Popen([sys.executable, __file__, "worker", str(folder / f"spec-{i}.json")], stdout=log, stderr=subprocess.STDOUT,
                                      env={**os.environ, "HF_HUB_OFFLINE": "1", **(env or {})}))
        specs.append(spec)
    started = time.time()
    while not all(Path(s["ready"]).exists() for s in specs):
        if any(p.poll() is not None for p in procs) or time.time() > deadline:
            break
        time.sleep(.2)
    loaded = time.time() - started
    samples, stop = [], threading.Event()

    def poll():
        while not stop.is_set():
            q = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True)
            if q.returncode == 0 and q.stdout.strip():
                samples.append([float(x) for x in q.stdout.strip().splitlines()[0].split(",")])
            stop.wait(1)
    threading.Thread(target=poll, daemon=True).start()
    Path(specs[0]["go"]).write_text("go")
    for p in procs:
        try:
            p.wait(timeout=max(deadline - time.time() + 120, 10))
        except subprocess.TimeoutExpired:
            p.kill()
    stop.set()
    reports = []
    for i, s in enumerate(specs):
        if Path(s["result"]).exists():
            reports.append(json.loads(Path(s["result"]).read_text()))
        else:
            tail = (folder / f"log-{i}.txt").read_text().splitlines()[-40:]
            reports.append({"error": "no result", "exitCode": procs[i].returncode, "logTail": tail})
    return {"name": name, "processes": len(job_lists), "loadWallSeconds": round(loaded, 1), "reports": reports,
            "gpuMemoryUsedMaxMiB": max((s[0] for s in samples), default=None),
            "gpuUtilisationMeanPct": round(sum(s[1] for s in samples) / len(samples), 1) if samples else None, "smiSamples": len(samples)}


@app.function(image=gpu_image, gpu="A100-80GB", cpu=8, memory=65536, timeout=3000, retries=0,
              volumes={"/weights": sam3d_research.weights, str(ART): data}, secrets=[modal.Secret.from_name("huggingface")])
def bench(smoke=False):
    from huggingface_hub import snapshot_download
    entered = time.time()
    t = time.time()
    snapshot_download("facebook/sam-3d-objects", revision=sam3d_research.MODEL_REVISION)  # online once; the workers run offline
    report = {"entered": entered, "weightsCheckSeconds": round(time.time() - t, 1), "moduleFile": __file__, "phases": []}
    index = json.loads((E4 / "index.json").read_text())
    first = [r for r in index if r["attempt"] == 1]
    every = [r for r in index]
    if smoke:
        first, every = first[:2], first[:2]
    job = lambda arm, r, block, save=True: {"arm": arm, "input": r["input"], "eid": r["eid"], "attempt": r["attempt"], "observation": r["observation"],
                                            "block": block, **({"save": str(E4 / "arms" / arm / "journal-sam3d" / r["eid"] / f"sam3d-{r['key']}")} if save else {})}
    tp = first[:2 if smoke else TP_CALLS]
    # most needed first: a deadline cuts the tail
    warm = [job("fast", first[0], "warmup", save=False)]
    jobs = [job("fast", r, "tp1", save=False) for r in tp] + [job(arm, r, arm) for arm in ("fast", "fast-s2d", "current") for r in every]
    jobs += [job("fast-crop", r, "fast-crop") for r in first]
    report["phases"].append(processes("A-1proc", [jobs], time.time() + (300 if smoke else 2200), warmup=warm))
    data.commit()
    split = [[job("fast", r, "tp2", save=False) for r in tp[i::2]] for i in range(2)]
    report["phases"].append(processes("B-2proc", split, time.time() + (200 if smoke else 400), warmup=warm))
    mps = shutil.which("nvidia-cuda-mps-control")
    env = {"CUDA_MPS_PIPE_DIRECTORY": "/tmp/mps-pipe", "CUDA_MPS_LOG_DIRECTORY": "/tmp/mps-log"}
    report["mps"] = {"binary": mps}
    if mps:
        for d in env.values():
            Path(d).mkdir(parents=True, exist_ok=True)
        started = subprocess.run([mps, "-d"], env={**os.environ, **env}, capture_output=True, text=True)
        report["mps"]["start"] = {"code": started.returncode, "out": (started.stdout + started.stderr)[-500:]}
        if started.returncode == 0:
            split = [[job("fast", r, "tp2mps", save=False) for r in tp[i::2]] for i in range(2)]
            report["phases"].append(processes("C-2proc-mps", split, time.time() + (200 if smoke else 400), env, warm))
            servers = subprocess.run([mps], input="get_server_list\n", env={**os.environ, **env}, capture_output=True, text=True)
            report["mps"]["serverList"] = servers.stdout.strip()[-200:]
            log = Path(env["CUDA_MPS_LOG_DIRECTORY"]) / "control.log"
            report["mps"]["controlLogTail"] = log.read_text().splitlines()[-15:] if log.exists() else None
            subprocess.run([mps], input="quit\n", env={**os.environ, **env}, capture_output=True, text=True)
    report["exited"] = time.time()
    return report


# ---------------------------------------------------------------- local steps
def save_local(name, value):
    LOCAL_OUT.mkdir(parents=True, exist_ok=True)
    (LOCAL_OUT / name).write_text(json.dumps(value, indent=1, default=str) + "\n")


@app.local_entrypoint()
def upload():
    """The files the gate reads for the 30 objects (masks and frames of their observations only) and 241's SAM 3D journal."""
    runs = ART / "runs"
    document = json.loads((runs / "me340-entity-names-200/object-map.json").read_text())
    observations = [o for e in document["entities"] if e["entityId"] in ENTITIES for o in e["observations"]]
    manifest = json.loads((runs / "droid-me340-165-171/input-manifest.json").read_text())
    frames = {int(o.rsplit(":", 2)[1]) for o in observations}
    files = [runs / "droid-me340-165-171" / n for n in ("run.json", "input-manifest.json", "prediction.npz")]
    files += [runs / "da3-posed-me340-189-fused-dynamic" / n for n in ("droid-support.npz", "metric-scale.json", "fuse-metrics.json", "infer.json")]
    files += sorted((runs / "da3-posed-me340-189-fused-dynamic/mono").glob("*.npz")) + sorted((runs / "me340-dynamic-masks-188/masks").glob("*.png"))
    files += [runs / "me340-entity-names-200/object-map.json"]
    for o in observations:
        label, frame, instance = o.rsplit(":", 2)
        files += sorted((runs / "me340-masks-194").glob(f"{label}-*/frame-{int(frame):05d}/instance-{instance}-mask.png"))
    clip = ART / "data/clips/me340-165"
    files += [clip / n for n in ("clip.json", "rgb.txt", "source-full.json", "source-full.mp4")]
    files += [clip / f["relative_path"] for f in manifest["frames"] if f["source_index"] in frames]
    pairs = [(f, "/" + str(f.relative_to(ART))) for f in files]
    pairs += [(f, "/e4/seed241/" + str(f.relative_to(SEED_RUN))) for f in sorted((SEED_RUN / "resegmented").rglob("*")) if f.is_file()]
    pairs += [(f, "/e4/seed241/" + str(f.relative_to(SEED_RUN))) for e in ENTITIES for f in sorted((SEED_RUN / "journal-sam3d" / e).rglob("*")) if f.is_file()]
    started = time.time()
    with data.batch_upload(force=True) as batch:
        for local, remote in pairs:
            batch.put_file(local.resolve(), remote)  # resolved: the masks and mono folders are symlinks on the Mac
    size = sum(local.resolve().stat().st_size for local, _ in pairs)
    print(json.dumps({"files": len(pairs), "MB": round(size / 1e6, 1), "seconds": round(time.time() - started, 1)}))


@app.local_entrypoint()
def inputs_step():
    started = time.time()
    results = list(inputs.map(ENTITIES, return_exceptions=True))
    listing = [t for r in results if isinstance(r, dict) for t in r.get("tries", [])]
    with data.batch_upload(force=True) as batch:
        import io
        batch.put_file(io.BytesIO(json.dumps(listing).encode()), "/e4/index.json")
    save_local("inputs.json", {"wallSeconds": round(time.time() - started, 1), "results": [r if isinstance(r, dict) else repr(r) for r in results]})
    save_local("index.json", listing)
    print(json.dumps({"tries": len(listing), "byKey": sum(t["baseline"] == "key" for t in listing),
                      "byObservation": sum(t["baseline"] == "observation+seed" for t in listing),
                      "errors": [repr(r)[:300] for r in results if not isinstance(r, dict)],
                      "noTries": [r["eid"] for r in results if isinstance(r, dict) and not r.get("tries")],
                      "wallSeconds": round(time.time() - started, 1)}))


@app.local_entrypoint()
def bench_step(smoke: bool = False):
    called = time.time()
    report = bench.remote(smoke)
    report.update(called=called, returned=time.time(), coldStartSeconds=round(report["entered"] - called, 1),
                  containerSeconds=round(report["exited"] - report["entered"], 1))
    report["usdUpperBound"] = round((report["returned"] - called) * GPU_USD_S, 3)
    save_local("bench-smoke.json" if smoke else "bench.json", report)
    print(json.dumps({k: report[k] for k in ("coldStartSeconds", "containerSeconds", "usdUpperBound", "mps")}, default=str)[:1500])


@app.local_entrypoint()
def gate_step(arms: str = ",".join(GATED), only: str = ""):
    started = time.time()
    eids = only.split(",") if only else ENTITIES
    results = list(gate.map(eids, kwargs={"arms": arms.split(",")}, return_exceptions=True))
    save_local("gate-" + (only.replace(",", "_") or "all") + ".json",
               {"wallSeconds": round(time.time() - started, 1), "results": [r if isinstance(r, dict) else repr(r) for r in results]})
    print(json.dumps({"wallSeconds": round(time.time() - started, 1), "errors": [repr(r)[:400] for r in results if not isinstance(r, dict)]}))


# ---------------------------------------------------------------- summary
def within(tries, arm, eid, k):
    """The runner's outcome: one of the object's first k tries in this arm has a mesh and passed the gate."""
    return any(t["generated"] and t["accepted"] for t in tries if t["arm"] == arm and t["eid"] == eid and t["attempt"] <= k)


def percentiles(values):
    import numpy as np
    return {"n": len(values), **({"median": round(float(np.median(values)), 2), "p10": round(float(np.percentile(values, 10)), 2),
                                  "p90": round(float(np.percentile(values, 90)), 2), "mean": round(float(np.mean(values)), 2)} if values else {})}


def summary():
    bench_report = json.loads((LOCAL_OUT / "bench.json").read_text())
    gate_report = json.loads((LOCAL_OUT / "gate-all.json").read_text())["results"]
    seed = {o["entityId"]: o for o in json.loads((SEED_RUN / "manifest.json").read_text())["objects"]}
    calls = [r for p in bench_report["phases"] for rep in p["reports"] for r in rep.get("results", []) if "seconds" in r]
    out = {"calls": {}, "throughput": {}, "gate": {}, "cost": {}}
    for block in ("warmup", "fast", "current", "fast-s2d", "fast-crop", "tp1", "tp2", "tp2mps"):
        rows = [r for r in calls if r["block"] == block]
        if rows:
            stages = {s: percentiles([r["stages"].get(s, 0) for r in rows])["median"] for s in STAGES}
            out["calls"][block] = {**percentiles([r["seconds"] for r in rows]), "stageMedians": stages,
                                   "triangles": percentiles([r["triangles"] for r in rows]).get("median")}
    for p in bench_report["phases"]:
        rows = [r for rep in p["reports"] for r in rep.get("results", []) if r.get("block", "").startswith("tp") and "seconds" in r]
        if rows:
            wall = max(r["t1"] for r in rows) - min(r["t0"] for r in rows)
            out["throughput"][p["name"]] = {"calls": len(rows), "wallSeconds": round(wall, 1), "effectiveSecondsPerCall": round(wall / len(rows), 2),
                                            "callsPerMinute": round(60 * len(rows) / wall, 1), "perCallMedian": percentiles([r["seconds"] for r in rows]).get("median"),
                                            "gpuMemoryUsedMaxMiB": p["gpuMemoryUsedMaxMiB"], "gpuUtilisationMeanPct": p["gpuUtilisationMeanPct"],
                                            "loadWallSeconds": p["loadWallSeconds"], "maxAllocatedGB": [rep.get("maxAllocatedGB") for rep in p["reports"]]}
    ok = [r for r in gate_report if isinstance(r, dict)]
    tries, eids = [dict(t, eid=r["eid"]) for r in ok for t in r["tries"]], [r["eid"] for r in ok]
    judged = lambda arm: {(t["eid"], t["attempt"]): t["accepted"] for t in tries if t["arm"] == arm and t["generated"]}
    for arm in GATED:
        js = judged(arm)
        out["gate"][arm] = {"objects": len(eids), "triesJudged": len(js), "triesAccepted": sum(js.values()),
                            "perTryRate": round(sum(js.values()) / max(len(js), 1), 3),
                            **{f"objectsAcceptedWithin{k}": sum(within(tries, arm, e, k) for e in eids) for k in (1, 2, 5)},
                            "assessSeconds": percentiles([t["assessSeconds"] for t in tries if t["arm"] == arm and t["generated"]])}
    out["gate"]["paired"] = {}
    for a, b in (("baseline", "current"), ("current", "fast"), ("current", "fast-s2d"), ("baseline", "fast"), ("baseline", "fast-s2d"),
                 ("fast", "fast-crop")):
        ja, jb = judged(a), judged(b)
        both = sorted(ja.keys() & jb.keys())
        oa, ob = ({e for e in eids if within(tries, arm, e, 5)} for arm in (a, b))
        out["gate"]["paired"][f"{a} vs {b}"] = {"tries": len(both), "bothAccepted": sum(ja[k] and jb[k] for k in both),
                                               "onlyFirst": sum(ja[k] and not jb[k] for k in both), "onlySecond": sum(jb[k] and not ja[k] for k in both),
                                               "objectsBoth": len(oa & ob), "objectsOnlyFirst": sorted(oa - ob), "objectsOnlySecond": sorted(ob - oa)}
    out["gate"]["baselineVs241Manifest"] = {"agree": sum(within(tries, "baseline", e, 5) == bool(seed[e].get("accepted")) for e in eids), "of": len(eids),
                                            "241Accepted": sum(bool(seed[e].get("accepted")) for e in eids)}
    out["gate"]["prepareSeconds"] = percentiles([r["prepareSeconds"] for r in ok])
    gpu_seconds = bench_report["exited"] - bench_report["entered"]
    out["cost"] = {"benchContainerSeconds": round(gpu_seconds, 1), "benchColdStartSeconds": bench_report["coldStartSeconds"],
                   "benchUsdUpperBound": bench_report["usdUpperBound"]}
    save_local("summary.json", out)
    print(json.dumps(out, indent=1))


def self_check():
    tries = [{"arm": "x", "eid": "o", "attempt": 1, "generated": True, "accepted": False}, {"arm": "x", "eid": "o", "attempt": 2, "generated": False},
             {"arm": "x", "eid": "o", "attempt": 3, "generated": True, "accepted": True}, {"arm": "y", "eid": "o", "attempt": 1, "generated": True, "accepted": True}]
    assert not within(tries, "x", "o", 1) and not within(tries, "x", "o", 2) and within(tries, "x", "o", 3) and within(tries, "y", "o", 1)
    assert not within(tries, "x", "p", 5), "another object's tries never count"
    assert percentiles([1, 2, 3])["median"] == 2 and percentiles([])["n"] == 0
    print("self-check ok")


if __name__ == "__main__":
    {"worker": lambda: worker(sys.argv[2]), "summary": summary, "self-check": self_check}[sys.argv[1]]()
