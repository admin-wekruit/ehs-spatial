"""mvp2/identity study: Qwen3-VL-8B open naming (the core's vLLM settings) on the held-out items of scripts/identity_study.py.
One ephemeral container, one A100-80GB (an offline study, not the 2-GPU pipeline); vLLM's load is timed apart.

  modal run modal_apps/identity_study_app.py --study RUNS/mvp2-identity-study-001

B1: the context crop (outlined, 2.5x) + the close crop; B2: the pipeline's set-of-marks pair (core.ask_identity's crops).
Writes STUDY/qwen-heldout.json: {id: {b1, b2: {text, name, s}}} and the timing.
"""
import json
import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent)]
DA3_CODE = "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4"
VOLUMES = {"/v/vlm": modal.Volume.from_name("panoptes-vlm-cache")}
image = (modal.Image.from_registry("nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04", add_python="3.11")  # judge_decider.py's image, same layers
         .apt_install("git", "libgl1", "libglib2.0-0", "libgomp1")
         .pip_install("torch==2.14.0", "torchvision", "xformers", "transformers==5.17.0", "accelerate", "addict", "pillow", "scipy",
                      "open3d==0.19.0", "shapely", "pydantic", "opencv-python-headless", "sentencepiece",
                      f"git+https://github.com/ByteDance-Seed/Depth-Anything-3.git@{DA3_CODE}")
         .run_commands("python -m venv /opt/vllm && PIP_EXTRA_INDEX_URL= /opt/vllm/bin/pip install -q vllm==0.11.0 transformers==4.57.1 pillow")
         .env({"HF_HUB_OFFLINE": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
         .add_local_python_source("fast_report", "ehs_spatial"))

app = modal.App("panoptes-mvp2-identity-study")


@app.function(image=image, gpu="A100-80GB", cpu=8, memory=32768, timeout=1800, retries=0, volumes=VOLUMES, max_containers=1)
def ask(rows, crops):
    from concurrent.futures import ThreadPoolExecutor
    from fast_report import vlm
    from fast_report.instrument import Clock, Vram
    t0 = time.perf_counter()
    vram = Vram([0]).start()
    proc = vlm.start(0)
    vlm.wait(proc)
    boot_s = round(time.perf_counter() - t0, 2)
    vlm.name_open([crops[rows[0]["id"]]["ctx"], crops[rows[0]["id"]]["plain"]])  # warm-up, not timed
    clock = Clock()
    out = {}
    for variant, kinds in (("b1", ("ctx", "plain")), ("b2", ("som", "plain"))):
        with clock.stage(f"identity.open.{variant}", gpu=0, n={"questions": len(rows)}, sync=False):
            with ThreadPoolExecutor(vlm.MAX_SEQS) as pool:
                got = list(pool.map(lambda r: vlm.name_open([crops[r["id"]][k] for k in kinds]), rows))
        for r, g in zip(rows, got):
            out.setdefault(r["id"], {})[variant] = g
    run = clock.report(vram, price_per_s=None, boot={"vllm_ready_s": boot_s})
    vram.stop()
    proc.terminate()
    return {"answers": out, "run": {k: run[k] for k in ("stages", "gpu_peak", "flags") if k in run}, "boot_s": boot_s, "vllm_tail": vlm.log_tail(3)}


@app.local_entrypoint()
def main(study: str, which: str = "heldout"):
    rd = Path(study)
    rows = json.loads((rd / "items.json").read_text())["items"]
    crops = {r["id"]: {k: (rd / "crops" / f"{r['id'].replace(':', '__')}.{k}.jpg").read_bytes() for k in ("ctx", "plain", "som")} for r in rows}
    t = time.time()
    res = ask.remote(rows, crops)
    res["client_wall_s"] = round(time.time() - t, 1)
    (rd / f"qwen-{which}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({"boot_s": res["boot_s"], "wall_s": res["client_wall_s"], "stages": res["run"].get("stages"),
                      "gpu_peak": res["run"].get("gpu_peak")}, indent=1)[:3000])
