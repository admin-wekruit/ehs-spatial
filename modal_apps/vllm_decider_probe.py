"""mvp2 click (R9): the decider's throughput under vLLM settings, alone on one A100-80GB. The core's vLLM (fast_report.vlm.start:
Qwen3-VL-8B, 0.35 of the GPU, --max-num-seqs 16, --enforce-eager) answers identity and judgement questions (two set-of-marks
crops, one token, log-probs) at 12-18 questions/s inside a run; which setting raises that is measured here, with the core's
own request code (vlm.submit) and real crops from a run.

  modal run modal_apps/vllm_decider_probe.py --crops DIR_OF_JPEGS --out RUNS/mvp2-click-vllm-probe-NNN [--n 400]
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent)]
DA3_CODE = "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4"
VOLUMES = {"/v/vlm": modal.Volume.from_name("panoptes-vlm-cache")}
image = (modal.Image.from_registry("nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04", add_python="3.11")  # judge_decider's image (cached layers)
         .apt_install("git", "libgl1", "libglib2.0-0", "libgomp1")
         .pip_install("torch==2.14.0", "torchvision", "xformers", "transformers==5.17.0", "accelerate", "addict", "pillow", "scipy",
                      "open3d==0.19.0", "shapely", "pydantic", "opencv-python-headless", "sentencepiece",
                      f"git+https://github.com/ByteDance-Seed/Depth-Anything-3.git@{DA3_CODE}")
         .run_commands("python -m venv /opt/vllm && PIP_EXTRA_INDEX_URL= /opt/vllm/bin/pip install -q vllm==0.11.0 transformers==4.57.1 pillow")
         .env({"HF_HUB_OFFLINE": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
         .add_local_python_source("fast_report", "ehs_spatial"))

app = modal.App("panoptes-mvp2-vllm-probe")
CONFIGS = [  # name, extra/changed vLLM args, client workers
    ("base: eager, 16 seqs", [], 16),
    ("eager, 32 seqs", ["--max-num-seqs", "32"], 32),
    ("compiled (no --enforce-eager), 16 seqs", ["-eager"], 16),
    ("eager, 2 API servers", ["--api-server-count", "2"], 16),
    ("compiled, 32 seqs, 2 API servers", ["-eager", "--max-num-seqs", "32", "--api-server-count", "2"], 32),
]


@app.function(image=image, gpu="A100-80GB", cpu=16, memory=65536, timeout=3000, retries=0, volumes=VOLUMES, max_containers=1)
def probe(crops, n=400, configs=CONFIGS):
    import os
    from concurrent.futures import ThreadPoolExecutor
    from fast_report import vlm
    out = []
    for name, extra, workers in configs:
        env = {k: v for k, v in os.environ.items() if k != "PYTORCH_CUDA_ALLOC_CONF"}
        env.update(CUDA_VISIBLE_DEVICES="0", HF_HOME="/v/vlm/huggingface", HF_HUB_OFFLINE="1")
        seqs = extra[extra.index("--max-num-seqs") + 1] if "--max-num-seqs" in extra else str(vlm.MAX_SEQS)
        cmd = ["/opt/vllm/bin/vllm", "serve", vlm.QWEN, "--host", "127.0.0.1", "--port", str(vlm.VLLM_PORT), "--served-model-name", "qwen",
               "--max-model-len", "16384", "--gpu-memory-utilization", str(vlm.VLLM_SHARE), "--max-num-seqs", seqs,
               "--limit-mm-per-prompt", json.dumps({"image": 40, "video": 0}), "--seed", "0"]
        cmd += [] if "-eager" in extra else ["--enforce-eager"]
        cmd += [a for i, a in enumerate(extra) if a != "-eager" and a != "--max-num-seqs" and (i == 0 or extra[i - 1] != "--max-num-seqs")]
        t = time.perf_counter()
        proc = subprocess.Popen(cmd, env=env, stdout=open("/tmp/vllm.log", "w"), stderr=subprocess.STDOUT)
        row = {"config": name, "args": extra, "workers": workers}
        try:
            vlm.wait(proc, 900)
            row["startup_s"] = round(time.perf_counter() - t, 1)
            p = vlm.qwen_prompt("These are crops from a video of a workplace.", "What is the object marked [1]?",
                                ["box", "shelf", "pallet", "another kind of object", "not one object (a part, a surface or several things)"])
            ask = lambda i: vlm._ask([crops[i % len(crops)], crops[(i + 7) % len(crops)]], p, 5)  # noqa: E731  the core's request
            with ThreadPoolExecutor(workers) as pool:  # this setting's client concurrency (vlm.submit's workers are process-wide)
                list(pool.map(ask, range(2 * workers)))
                t0 = time.perf_counter()
                res = list(pool.map(ask, range(n)))
            s = time.perf_counter() - t0
            lat = sorted(r["s"] for r in res)
            row.update(questions=n, s=round(s, 2), per_s=round(n / s, 2), prompt_tokens_mean=round(sum(r["prompt_tokens"] for r in res) / n, 1),
                       request_p50_s=lat[n // 2], request_p90_s=lat[int(.9 * n)], engine=vlm.throughput()[-6:],
                       probs=[r["probs"] for r in res])  # the same questions in every setting (and each pair twice: i, i + len(crops))
        except Exception as error:  # noqa: BLE001  one setting that does not start is that row's result
            row["error"] = repr(error)[:600]
        finally:
            proc.terminate()
            try:
                proc.wait(60)
            except subprocess.TimeoutExpired:
                proc.kill()
            time.sleep(5)
        print(json.dumps({k: v for k, v in row.items() if k not in ("stats", "engine")}), flush=True)
        out.append(row)
    return out


SAME = [("base: eager, 16 seqs", [], 16), ("eager, 3 API servers", ["--api-server-count", "3"], 16), ("base again", [], 16)]
MORE = [("base: eager, 16 seqs", [], 16), ("eager, 3 API servers", ["--api-server-count", "3"], 16),
        ("eager, 4 API servers", ["--api-server-count", "4"], 16), ("eager, 4 API servers, 24 seqs", ["--api-server-count", "4", "--max-num-seqs", "24"], 24)]


@app.local_entrypoint()
def main(crops: str, out: str, n: int = 400, more: bool = False, same: bool = False):
    jpegs = [p.read_bytes() for p in sorted(Path(crops).glob("*.jpg"))]
    assert jpegs, f"no crops in {crops}"
    o = Path(out)
    o.mkdir(parents=True, exist_ok=False)
    t = time.time()
    rows = probe.remote(jpegs, n, SAME if same else MORE if more else CONFIGS)
    base = rows[0].get("probs") or []
    for r in rows:  # max |p - p_base| over the same questions
        r["max_prob_delta_vs_first"] = max((abs(a - b) for x, y in zip(r.get("probs") or [], base) for a, b in zip(x, y)), default=None)
    (o / "probe.json").write_text(json.dumps({"crops": len(jpegs), "n": n, "wall_s": round(time.time() - t, 1), "rows": rows}, indent=1))
    for r in rows:
        print(r["config"], r.get("per_s"), r.get("startup_s"), r.get("max_prob_delta_vs_first"), r.get("error", ""))
