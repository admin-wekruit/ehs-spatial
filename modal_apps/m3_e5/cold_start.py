"""E5(d): cold start of one container with two A100-80GB holding DA3-GIANT-1.1, SAM 3, SAM 3D Objects and Qwen3-VL-8B (vLLM), all
weights from the Modal volumes the existing apps filled. Seconds from submitting the call to every model loaded and through one
warm-up call.

One image, three Python environments, as the fast plan's resident container would need them (their torch versions conflict):
  system python (SAM 3D's pinned torch 2.5.1/cu121 stack, the sam3d_research.py image itself) -> SAM 3D, GPU 0
  /opt/vis  (latest torch, xformers, transformers, DA3 at a pinned commit)                   -> DA3-GIANT-1.1 and SAM 3, GPU 0
  /opt/vllm (vLLM 0.11.0, torch 2.8)                                                          -> Qwen3-VL-8B, GPU 1, 0.45 of the card
Each model family loads in its own process, all four at once, from the moment the function starts. single_use_containers: every call gets a
fresh container, so each run is a cold start (the first one also pulls a new image onto a worker).

  python modal_apps/m3_e5/cold_start.py --output NEW_DIR [--runs 2]
"""
import argparse
import json
from pathlib import Path
import sys
import time

import modal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import sam3d_research  # noqa: E402  its image (cached layers) and pins
import sam3_app  # noqa: E402  pin

DA3_CODE = "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4"
DA3_MODEL, DA3_REV = "depth-anything/DA3-GIANT-1.1", "72ee9f89ce4e50d704e9d55ee9c646ec8dc25a19"  # mono_room.DA3_REVISIONS
QWEN = "Qwen/Qwen3-VL-8B-Instruct"
USD_PER_S = 2 * .000694 + 8 * .0000131 + 64 * .00000222  # 2x A100-80GB + 8 cores + 64 GiB, modal.com/pricing
NO_INDEX = "PIP_EXTRA_INDEX_URL= PIP_FIND_LINKS="  # the SAM 3D image points pip at the cu121 torch index; the venvs must not use it
image = (sam3d_research.image
         .run_commands(f"python -m venv /opt/vllm && {NO_INDEX} /opt/vllm/bin/pip install -q vllm==0.11.0 transformers==4.57.1 pillow")
         .run_commands(f"python -m venv /opt/vis && {NO_INDEX} /opt/vis/bin/pip install -q torch torchvision xformers 'transformers>=4.57' accelerate "
                       f"addict pillow 'git+https://github.com/ByteDance-Seed/Depth-Anything-3.git@{DA3_CODE}'")
         .add_local_python_source("sam3d_research", "sam3_app"))
app = modal.App("panoptes-m3-e5-cold-start")
volumes = {"/v/da3": modal.Volume.from_name("moge3-hf-cache"), "/v/sam3": modal.Volume.from_name("sam3-hf-cache"),
           "/weights": sam3d_research.weights, "/v/vlm": modal.Volume.from_name("panoptes-vlm-cache")}

HEAD = "import json, time, sys\nt0 = time.time()\n"
READY = "print('READY ' + json.dumps(r), flush=True)\n"
LOADERS = {
    "da3": ("/opt/vis/bin/python", "0", "/v/da3/huggingface", HEAD + f"""
import numpy as np, torch
from depth_anything_3.api import DepthAnything3
r = {{"imported": time.time()}}
m = DepthAnything3.from_pretrained("{DA3_MODEL}", revision="{DA3_REV}").to("cuda").eval()
torch.cuda.synchronize(); r["loaded"] = time.time()
m.inference([np.random.randint(0, 255, (280, 504, 3), np.uint8) for _ in range(2)], process_res=504)
torch.cuda.synchronize(); r["warm"] = time.time(); r["gb"] = torch.cuda.max_memory_allocated() / 1e9
""" + READY),
    "sam3": ("/opt/vis/bin/python", "0", "/v/sam3/huggingface", HEAD + f"""
import numpy as np, torch
from PIL import Image
from transformers import Sam3Model, Sam3Processor
r = {{"imported": time.time()}}
p = Sam3Processor.from_pretrained("facebook/sam3", revision="{sam3_app.REVISION}")
m = Sam3Model.from_pretrained("facebook/sam3", revision="{sam3_app.REVISION}", torch_dtype=torch.bfloat16).to("cuda").eval()
torch.cuda.synchronize(); r["loaded"] = time.time()
x = p(images=Image.fromarray(np.random.randint(0, 255, (720, 1280, 3), np.uint8)), text="person", return_tensors="pt").to("cuda")
with torch.inference_mode():
    m(**{{k: (v.to(torch.bfloat16) if v.is_floating_point() else v) for k, v in x.items()}})
torch.cuda.synchronize(); r["warm"] = time.time(); r["gb"] = torch.cuda.max_memory_allocated() / 1e9
""" + READY),
    "sam3d": ("python", "0", "/weights/huggingface", HEAD + f"""
from pathlib import Path
import numpy as np, torch
from huggingface_hub import snapshot_download
from hydra.utils import instantiate
from omegaconf import OmegaConf
r = {{"imported": time.time()}}
root = Path(snapshot_download("facebook/sam-3d-objects", revision="{sam3d_research.MODEL_REVISION}"))
s = OmegaConf.load(root / "checkpoints/pipeline.yaml")
s.rendering_engine, s.compile_model, s.workspace_dir = "pytorch3d", False, str(root / "checkpoints")
pipe = instantiate(s, depth_model=None, decode_formats=["mesh"], slat_decoder_gs_config_path=None, slat_decoder_gs_ckpt_path=None,
                   slat_decoder_gs_4_config_path=None, slat_decoder_gs_4_ckpt_path=None)
torch.cuda.synchronize(); r["loaded"] = time.time()
rgb = np.full((256, 256, 3), 90, np.uint8); rgb[88:168, 88:168] = (200, 60, 40)
mask = np.zeros((256, 256), bool); mask[88:168, 88:168] = True
v, u = np.indices((256, 256), dtype=np.float64); z = np.full((256, 256), 2.); z[88:168, 88:168] = 1.8
pm = np.stack([-(u - 127.5) / 250 * z, -(v - 127.5) / 250 * z, z], -1).astype(np.float32)
rgba = np.concatenate([rgb, (mask.astype(np.uint8) * 255)[..., None]], -1)
pipe.run(rgba, None, seed=42, pointmap=torch.from_numpy(pm).cuda(), estimate_plane=False, decode_formats=["mesh"], with_mesh_postprocess=False,
         with_texture_baking=False, with_layout_postprocess=False, use_vertex_color=True)
torch.cuda.synchronize(); r["warm"] = time.time(); r["gb"] = torch.cuda.max_memory_allocated() / 1e9
""" + READY),
    "qwen3vl_vllm": ("/opt/vllm/bin/python", "1", "/v/vlm/huggingface", HEAD + f"""
import torch
from vllm import LLM, SamplingParams
r = {{"imported": time.time()}}
llm = LLM(model="{QWEN}", max_model_len=16384, limit_mm_per_prompt={{"image": 40, "video": 0}}, gpu_memory_utilization=.45, max_num_seqs=8, seed=0)
r["loaded"] = time.time()
llm.generate(["Describe a workshop in one sentence."], SamplingParams(temperature=0, max_tokens=16), use_tqdm=False)
r["warm"] = time.time()
""" + READY)}


@app.function(image=image, gpu="A100-80GB:2", cpu=8, memory=65536, timeout=1200, retries=0, single_use_containers=True, volumes=volumes)
def cold_remote():
    import os
    import subprocess
    entered = time.time()
    procs = {}
    for name, (python, gpu, hf_home, code) in LOADERS.items():
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "HF_HOME": hf_home, "HF_HUB_OFFLINE": "1", "LIDRA_SKIP_INIT": "true"}
        procs[name] = subprocess.Popen([python, "-c", code], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out = {"entered_unix": entered, "models": {}}
    for name, p in procs.items():
        stdout, stderr = p.communicate(timeout=1100)
        line = next((l for l in stdout.splitlines() if l.startswith("READY ")), None)
        if line is None:
            tail = [l for l in stderr.splitlines()[-40:] if "token" not in l.lower() and "secret" not in l.lower()]
            out["models"][name] = {"error": "\n".join(tail)[-3000:], "returncode": p.returncode}
            continue
        r = json.loads(line[6:])
        out["models"][name] = {"import_s": r["imported"] - entered, "loaded_s": r["loaded"] - entered, "warm_s": r["warm"] - entered,
                               **({"peak_gb": r["gb"]} if "gb" in r else {})}
    import torch
    out["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    out["returned_unix"] = time.time()
    return out


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    runs = []
    t_app = time.time()
    with modal.enable_output(), app.run():
        app_ready = time.time()
        for n in range(args.runs):
            submitted = time.time()
            r = cold_remote.remote()
            got = time.time()
            ok = [m for m in r["models"].values() if "warm_s" in m]
            row = {"run": n, "submit_to_entry_s_M": round(r["entered_unix"] - submitted, 1), "gpus": r["gpus"],
                   "models": {k: {kk: round(vv, 1) for kk, vv in m.items()} if "error" not in m else m for k, m in r["models"].items()},
                   "all_loaded_from_entry_s_M": round(max(m["loaded_s"] for m in ok), 1) if len(ok) == len(r["models"]) else None,
                   "all_warm_from_entry_s_M": round(max(m["warm_s"] for m in ok), 1) if len(ok) == len(r["models"]) else None,
                   "client_wall_s_M": round(got - submitted, 1), "container_s_M": round(r["returned_unix"] - r["entered_unix"], 1)}
            row["all_warm_from_submit_s_M"] = row["all_warm_from_entry_s_M"] and round(row["all_warm_from_entry_s_M"] + row["submit_to_entry_s_M"], 1)
            row["usd_estimate"] = round(USD_PER_S * row["client_wall_s_M"], 3)  # billed from container start, so bounded by the client wall
            runs.append(row)
            print(json.dumps(row, indent=1), flush=True)
            (args.output / "result.json").write_text(json.dumps({"app_start_s_M": round(app_ready - t_app, 1), "runs": runs,
                                                                 "loaders": {k: v[:3] for k, v in LOADERS.items()}}, indent=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
