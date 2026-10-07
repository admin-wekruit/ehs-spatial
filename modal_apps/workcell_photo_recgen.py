"""Ephemeral two A100 run for the four-photo workcell's first generated model."""

import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

import modal

if modal.is_local():
    from modal_apps.recgen_fast_bench import image
else:
    image = modal.Image.debian_slim()

app = modal.App("workcell-photo-recgen")
cache = modal.Volume.from_name("panoptes-lucida-weights")


@app.function(image=image, gpu="A100-80GB:2", cpu=16, memory=80 * 1024,
              volumes={"/cache": cache}, timeout=1800, retries=0)
def generate(payload: bytes, jobs: list[str]):
    import numpy as np
    import trimesh

    started = time.monotonic()
    with tempfile.TemporaryDirectory() as directory:
        source = Path(directory) / "input.npz"
        source.write_bytes(payload)
        calls = []
        for device, groups in enumerate(jobs):
            target = Path(directory) / f"gpu{device}"
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=str(device), ATTN_BACKEND="xformers",
                       SPCONV_ALGO="native", HF_HUB_OFFLINE="1",
                       PYTHONPATH="/repo:/opt/recgen")
            plan = Path(directory) / f"plan-{device}.json"
            parsed = [[name, [int(i) for i in indices.split(',')]] for name, indices in (part.split(':',1) for part in groups.split(';'))]
            plan.write_text(json.dumps([{"kind":"object", "source":str(source), "target":str(target), "groups":parsed}]))
            argv = ["/opt/recgen-venv/bin/python", "/repo/scripts/workcell_recgen_worker.py",
                    str(plan)]
            calls.append((target, subprocess.Popen(argv, env=env, stdout=subprocess.PIPE,
                                                           stderr=subprocess.PIPE, text=True)))
        z = np.load(io.BytesIO(payload))
        results = {}
        for target, proc in calls:
            stdout, stderr = proc.communicate(timeout=1500)
            if proc.returncode:
                raise RuntimeError(f"RecGen failed: {stderr[-1500:]}")
            worker = json.loads(stdout.strip().splitlines()[-1])
            for name, stats in worker["jobs"]["object"]["models"].items():
                mesh_data = np.load(f"{target}-{name}.npz")
                mesh = trimesh.Trimesh(vertices=mesh_data["vertices"], faces=mesh_data["faces"],
                                       vertex_colors=mesh_data["colors"], process=False)
                mesh.apply_transform(z[f"v{stats['views'][0]}_c2w"])
                glb = mesh.export(file_type="glb")
                results[name] = {"glb": bytes(glb), "stats": {**stats, "modelLoadSeconds": worker["modelLoadSeconds"]},
                                 "bounds": mesh.bounds.tolist()}
        return {"models": results, "wallSecondsIncludingColdStart": time.monotonic() - started}


@app.local_entrypoint()
def main(input: str, out: str, object_name: str = "robot",
         jobs: str = "v1:1;v3:3;multi:1,2,3|v2:2;v4:4"):
    started = time.monotonic()
    destination = Path(out)
    destination.mkdir(parents=True, exist_ok=True)
    groups = jobs.split("|")
    if len(groups) != 2 or any(not group for group in groups):
        raise ValueError("Exactly two GPU job groups are required")
    result = generate.remote(Path(input).read_bytes(), groups)
    for name, model in result["models"].items():
        (destination / f"{object_name}-{name}.glb").write_bytes(model["glb"])
    summary = {"wallSecondsIncludingColdStart": time.monotonic() - started,
               "containerWallSeconds": result["wallSecondsIncludingColdStart"],
               "models": {name: {k: v for k, v in model.items() if k != "glb"}
                          for name, model in result["models"].items()}}
    (destination / f"{object_name}-timing.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))
