"""MVP B2: the judgement decider (fast_report.vlm.options: Qwen3-VL-8B option letters, first-token log-probs, the core's
vLLM sidecar settings) on X8's set d, with the deployed options (yes / no / cannot tell) -> p(hazard) per item, for
fast_report.judge.calibrate (Platt per question, leave one source clip out). One ephemeral container, one A100-80GB;
vLLM's load is timed apart (cold start), the questions from their inputs in the container to the last answer.

  modal run modal_apps/judge_decider.py --x8-run RUNS/fx-x8-jev-001 --out RUNS/mvp-b-judge-decider-001
  python -m fast_report.judge --calibrate RUNS/mvp-b-judge-decider-001/answers.json   # -> fast_report/calibration.json
"""
import json
import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent)]
DA3_CODE = "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4"
D_STATE = "This is a frame from a video of a workplace (a shop floor, warehouse, store, lab or office)."  # X8's set-d state line

app = modal.App("panoptes-mvp-judge-decider")
VOLUMES = {"/v/vlm": modal.Volume.from_name("panoptes-vlm-cache")}
image = (modal.Image.from_registry("nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04", add_python="3.11")  # the fast core's layers, cached
         .apt_install("git", "libgl1", "libglib2.0-0", "libgomp1")
         .pip_install("torch==2.14.0", "torchvision", "xformers", "transformers==5.17.0", "accelerate", "addict", "pillow", "scipy",
                      "open3d==0.19.0", "shapely", "pydantic", "opencv-python-headless", "sentencepiece",
                      f"git+https://github.com/ByteDance-Seed/Depth-Anything-3.git@{DA3_CODE}")
         .run_commands("python -m venv /opt/vllm && PIP_EXTRA_INDEX_URL= /opt/vllm/bin/pip install -q vllm==0.11.0 transformers==4.57.1 pillow")
         .env({"HF_HUB_OFFLINE": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
         .add_local_python_source("fast_report", "ehs_spatial"))


@app.function(image=image, gpu="A100-80GB", cpu=8, memory=32768, timeout=1800, retries=0, volumes=VOLUMES, max_containers=1)
def ask(items, crops):
    from concurrent.futures import ThreadPoolExecutor
    from fast_report import judge, vlm
    from fast_report.instrument import Clock, Vram
    t0 = time.perf_counter()
    vram = Vram([0]).start()
    proc = vlm.start(0)
    vlm.wait(proc)
    boot_s = round(time.perf_counter() - t0, 2)
    vlm.options([crops[items[0]["crop"]]], vlm.qwen_prompt(D_STATE, "Is this a warm-up?", judge.YNC), judge.YNC)  # not timed
    clock = Clock()  # t0: every input in this container
    with clock.stage("judge.vlm", gpu=0, n={"questions": len(items)}, sync=False):
        with ThreadPoolExecutor(vlm.MAX_SEQS) as pool:
            got = list(pool.map(lambda x: vlm.options([crops[x["crop"]]], vlm.qwen_prompt(D_STATE, x["question"], judge.YNC), judge.YNC), items))
    run = clock.report(vram, price_per_s=None, boot={"vllm_ready_s": boot_s})
    vram.stop()
    proc.terminate()
    return {"answers": got, "run": {k: run[k] for k in ("stages", "gpu_peak", "flags") if k in run}, "boot_s": boot_s,
            "vllm_tail": vlm.log_tail(3)}


@app.local_entrypoint()
def main(x8_run: str, out: str):
    rd, out = Path(x8_run), Path(out)
    out.mkdir(parents=True, exist_ok=False)
    items = json.loads((rd / "sets/d.json").read_text())["items"]
    crops = {x["crop"]: (rd / "sets/crops" / f"{x['crop']}.jpg").read_bytes() for x in items}
    t = time.time()
    res = ask.remote(items, crops)
    wall = round(time.time() - t, 1)
    from fast_report import judge
    rows = [{"id": x["id"], "question_id": x["question_id"], "source": x["source"], "truth": x["truth"], "clear": x["clear"],
             "probs": a["probs"], "mass": a["mass"], "s": a["s"],
             "p": None if a["mass"] < judge.MIN_MASS or a["probs"][2] >= .5 else a["probs"][0]} for x, a in zip(items, res["answers"])]
    (out / "answers.json").write_text(json.dumps({"fitted_on": f"X8 set d ({rd.name}/sets/d.json, {len(items)} items, agent-labelled)",
                                                  "decider": "Qwen/Qwen3-VL-8B-Instruct via vLLM 0.11.0, option letters, first-token log-probs",
                                                  "prompt": {"state": D_STATE, "options": judge.YNC, "p": "p(yes) over the three letters"},
                                                  "items": rows, "run": res["run"], "boot_s": res["boot_s"], "client_wall_s": wall}, indent=1))
    print(json.dumps({"boot_s": res["boot_s"], "wall_s": wall, "stages": res["run"].get("stages"), "gpu_peak": res["run"].get("gpu_peak"),
                      "unanswered": sum(r["p"] is None for r in rows)}, indent=1)[:3000])
