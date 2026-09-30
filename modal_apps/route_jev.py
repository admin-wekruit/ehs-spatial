"""route/jev: the display-model router's GPU parts (scripts/route_jev.py). Labels never enter a container: it returns
embeddings and option probabilities; everything is scored locally.

  embed       PE-Core-L and SigLIP 2 so400m (both on the panoptes-x13-models volume): every card's crops (plain, masked,
              the outlined view) and the text prompts, one A100-80GB batch job (the core would run this on its own GPUs).
  JevService  Jev-Omni alone on one A100-80GB (X13's service: the card's loader + X8's batched forward), called from a CPU
              container standing in for the core; each round trip is timed at the caller, its compute in the service, the
              hop is the rest. Model load is timed apart (boot, not analysis).

  modal run modal_apps/route_jev.py --data SCRATCH/dataset.pkl --questions OUT/questions.json --prompts OUT/prompts.json \
      --out SCRATCH/answers.pkl [--limit N]
"""
import json
import os
import pickle
import subprocess
import time
from pathlib import Path

import modal

ENCODERS = {"pe-core-l": "timm/PE-Core-L-14-336", "siglip2-so400m": "google/siglip2-so400m-patch16-384"}
X13_HF = "/v/x13/hf"
TEMPLATES = ("a photo of {}.", "a photo of {} in a workplace.")
BATCH = 256

app = modal.App("panoptes-route-jev")
VOLUMES = {"/v/x8": modal.Volume.from_name("panoptes-x8-models"), "/v/x13": modal.Volume.from_name("panoptes-x13-models")}
# X13 naming's image line for line (its layers are cached)
image = (modal.Image.from_registry("nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04", add_python="3.11")
         .apt_install("git", "libgl1", "libglib2.0-0", "libgomp1")
         .pip_install("torch==2.14.0", "torchvision", "xformers", "transformers==5.17.0", "accelerate", "addict", "pillow", "scipy",
                      "open3d==0.19.0", "shapely", "pydantic", "opencv-python-headless", "sentencepiece",
                      "git+https://github.com/ByteDance-Seed/Depth-Anything-3.git@3d835ec1a5802d64a8b8b15f817a1ab54809bfe4")
         .run_commands("python -m venv /opt/vllm && PIP_EXTRA_INDEX_URL= /opt/vllm/bin/pip install -q vllm==0.11.0 transformers==4.57.1 pillow")
         .apt_install("ffmpeg")
         .pip_install("soundfile", "librosa", "laya==0.3.21")
         .pip_install("open_clip_torch")
         .env({"HF_HUB_OFFLINE": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "USE_TF": "0", "HF_HOME": "/v/x8/hf"}))


# ---------------------------------------------------------------- (b) zero-shot embeddings

class Encoder:
    """X13 naming's encoder, trimmed to the two text-image models: image(uint8 NHWC, masks or None) -> unit embeddings (the
    background outside the mask mid grey); text(phrases) -> unit embeddings (TEMPLATES averaged per phrase)."""

    def __init__(self, key):
        import torch
        self.key, self.torch = key, torch
        if key == "pe-core-l":
            import open_clip
            self.model, _, pre = open_clip.create_model_and_transforms("hf-hub:" + ENCODERS[key], cache_dir=X13_HF)
            self.model = self.model.cuda().eval().to(torch.bfloat16)
            self.tok = open_clip.get_tokenizer("hf-hub:" + ENCODERS[key])
            norm = [t for t in pre.transforms if type(t).__name__ == "Normalize"][0]
            self.side, self.mean, self.std = 336, tuple(norm.mean), tuple(norm.std)
        else:
            from transformers import AutoModel, AutoTokenizer
            self.model = AutoModel.from_pretrained(ENCODERS[key], cache_dir=X13_HF, torch_dtype=torch.bfloat16).cuda().eval()
            self.tok = AutoTokenizer.from_pretrained(ENCODERS[key], cache_dir=X13_HF)
            self.side, self.mean, self.std = self.model.config.vision_config.image_size, (.5,) * 3, (.5,) * 3
        self.scale = float(self.model.logit_scale.exp())

    def image(self, arr, masks=None):
        import numpy as np
        torch = self.torch
        F = torch.nn.functional
        mean = torch.tensor(self.mean, device="cuda")[:, None, None]
        std = torch.tensor(self.std, device="cuda")[:, None, None]
        out = []
        for s in range(0, len(arr), BATCH):
            x = torch.from_numpy(np.ascontiguousarray(arr[s:s + BATCH])).cuda().permute(0, 3, 1, 2).float() / 255
            if masks is not None:
                x = torch.where(torch.from_numpy(masks[s:s + BATCH]).cuda()[:, None], x, .5)
            x = F.interpolate(x, size=(self.side, self.side), mode="bicubic", antialias=True, align_corners=False).clamp(0, 1)
            x = ((x - mean) / std).to(torch.bfloat16)
            with torch.inference_mode():
                e = self.model.encode_image(x) if self.key == "pe-core-l" else self.model.get_image_features(pixel_values=x)
                e = getattr(e, "pooler_output", e)
            out.append(F.normalize(e.float(), dim=-1).cpu())
        return torch.cat(out).numpy()

    def text(self, phrases):
        torch = self.torch
        F = torch.nn.functional
        rows = []
        for p in phrases:
            prompts = [t.format(p) for t in TEMPLATES]
            with torch.inference_mode():
                if self.key == "pe-core-l":
                    e = self.model.encode_text(self.tok(prompts).cuda())
                else:
                    t = self.tok(prompts, padding="max_length", max_length=64, truncation=True, return_tensors="pt").to("cuda")
                    e = self.model.get_text_features(**t)
                    e = getattr(e, "pooler_output", e)
            rows.append(F.normalize(F.normalize(e.float(), dim=-1).mean(0), dim=0).cpu())
        return torch.stack(rows).numpy()


@app.function(image=image, gpu="A100-80GB", cpu=8, memory=65536, timeout=1800, retries=0, volumes=VOLUMES)
def embed(items, phrases):
    """items: [(card key, [(plain jpeg, mask png)], som jpeg)] -> pickle {emb: {(enc, var): (cards, D) f16, views mean-pooled},
    text: {enc: (phrases, D)}, scale, timing}."""
    os.environ["HF_HUB_CACHE"] = X13_HF  # before any Hugging Face import: open_clip's hf-hub lookups land in this volume
    import cv2
    import numpy as np
    import torch
    t = time.perf_counter()
    imgs, masks, owner, soms = [], [], [], []
    for i, (_, views, som) in enumerate(items):
        for plain, mask in views:
            imgs.append(cv2.imdecode(np.frombuffer(plain, np.uint8), 1)[:, :, ::-1])
            masks.append(cv2.imdecode(np.frombuffer(mask, np.uint8), 0) > 127)
            owner.append(i)
        s = cv2.imdecode(np.frombuffer(som, np.uint8), 1)[:, :, ::-1]
        pad = np.full((max(s.shape[:2]),) * 2 + (3,), 128, np.uint8)
        pad[:s.shape[0], :s.shape[1]] = s
        soms.append(cv2.resize(pad, (384, 384), interpolation=cv2.INTER_AREA))
    imgs, masks, owner, soms = np.stack(imgs), np.stack(masks), np.array(owner), np.stack(soms)
    res = {"keys": [k for k, _, _ in items], "emb": {}, "text": {}, "scale": {}, "timing": {"decode_s": round(time.perf_counter() - t, 2),
                                                                                             "cards": len(items), "views": len(imgs)}}
    for key in ENCODERS:
        t = time.perf_counter()
        enc = Encoder(key)
        enc.image(imgs[:8])  # warm-up, not timed
        tm = {"load_s": round(time.perf_counter() - t, 2)}
        for var, a, m in (("plain", imgs, None), ("masked", imgs, masks), ("som", soms, None)):
            torch.cuda.synchronize()
            t = time.perf_counter()
            e = enc.image(a, m)
            torch.cuda.synchronize()
            tm[var] = {"n": len(a), "s": round(time.perf_counter() - t, 3), "ms_per_crop": round(1000 * (time.perf_counter() - t) / len(a), 3)}
            if var != "som":  # views mean-pooled per card, renormalised
                v = np.zeros((len(items), e.shape[1]), np.float32)
                np.add.at(v, owner, e)
                e = v / np.linalg.norm(v, axis=1, keepdims=True)
            res["emb"][(key, var)] = e.astype(np.float16)
        t = time.perf_counter()
        res["text"][key] = enc.text(phrases)
        tm["text_s"], tm["peak_gib_torch"] = round(time.perf_counter() - t, 3), round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
        res["scale"][key], res["timing"][key] = enc.scale, tm
        del enc
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    res["gpu"] = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.splitlines()
    return pickle.dumps(res, protocol=4)


# ---------------------------------------------------------------- (c) Jev-Omni as a service on its own GPU (X13 naming's)

@app.cls(image=image.add_local_python_source("fast_report"), gpu="A100-80GB", cpu=4, memory=49152, timeout=1800, retries=0, volumes=VOLUMES,
         max_containers=1)
class JevService:
    """r5b: fast_report.jev's loader and forward (the report's app runs the same on its own GPU)."""

    @modal.enter()
    def load(self):
        from fast_report import jev
        self.st = jev.load()
        self.load_s = self.st["load_s"]

    @modal.method()
    def decide(self, qs):
        """qs: [(key, jpeg, state, question, options)] -> {"probs": {key: [p]}, "compute_s", "load_s", "peak_gib"}."""
        from fast_report import jev
        return jev.decide(self.st, qs)


@app.function(image=image, cpu=2, memory=8192, timeout=1800, retries=0)
def caller(questions, per_request):
    """The core's stand-in (no GPU): the questions to the service in requests of `per_request`, each round trip timed; then 20
    one-question requests (the hop's floor per call)."""
    svc = JevService()
    t = time.perf_counter()
    warm = svc.decide.remote(questions[:2])  # the service's cold start (container + model load), timed apart
    cold_s = round(time.perf_counter() - t, 2)
    probs, trips = {}, []
    t_all = time.perf_counter()
    for s in range(0, len(questions), per_request):
        chunk = questions[s:s + per_request]
        t = time.perf_counter()
        r = svc.decide.remote(chunk)
        rtt = time.perf_counter() - t
        probs.update(r["probs"])
        trips.append({"n": len(chunk), "rtt_s": round(rtt, 4), "compute_s": r["compute_s"], "hop_s": round(rtt - r["compute_s"], 4),
                      "bytes": sum(len(q[1]) for q in chunk)})
    total_s = round(time.perf_counter() - t_all, 2)
    single = []
    for q in questions[:20]:
        t = time.perf_counter()
        r = svc.decide.remote([q])
        single.append({"rtt_s": round(time.perf_counter() - t, 4), "compute_s": r["compute_s"]})
    return {"probs": probs, "trips": trips, "single": single, "cold_s": cold_s, "load_s": warm["load_s"], "peak_gib": r["peak_gib"],
            "gpu": r["gpu"], "total_s": total_s}


# ---------------------------------------------------------------- (d) the last tier: the core's Qwen3-VL-8B sidecar

@app.function(image=image.add_local_python_source("fast_report"), gpu="A100-80GB", cpu=8, memory=65536, timeout=1500, retries=0,
              volumes={"/v/vlm": modal.Volume.from_name("panoptes-vlm-cache")})
def qwen(qs):
    """The core's own vLLM start (fast_report.vlm: Qwen3-VL-8B-Instruct, its GPU share and server count) alone on one A100; the
    same image and question as Jev-Omni, answered as option-letter log-probs (X8's decider). qs: [(key, jpeg, state, q, options)]."""
    from concurrent.futures import ThreadPoolExecutor
    from fast_report import vlm
    t = time.perf_counter()
    proc = vlm.start(0)
    vlm.wait(proc)
    load_s = round(time.perf_counter() - t, 2)
    vlm._ask([qs[0][1]], vlm.qwen_prompt(*qs[0][2:]), len(qs[0][4]))  # warm-up, not timed
    t = time.perf_counter()
    with ThreadPoolExecutor(vlm.MAX_SEQS) as ex:
        got = list(ex.map(lambda q: vlm._ask([q[1]], vlm.qwen_prompt(q[2], q[3], q[4]), len(q[4])), qs))
    total = round(time.perf_counter() - t, 2)
    single = [vlm._ask([q[1]], vlm.qwen_prompt(q[2], q[3], q[4]), len(q[4]))["s"] for q in qs[:20]]
    proc.terminate()
    return {"probs": {q[0]: g["probs"] for q, g in zip(qs, got)}, "letter_mass_min": min(g["mass"] for g in got), "load_s": load_s,
            "total_s": total, "s_per_question_16_parallel": round(total / len(qs), 4), "single_s": single,
            "gpu": subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.strip()}


@app.local_entrypoint()
def main(data: str, questions: str, prompts: str, out: str, per_request: int = 64, limit: int = 0, stage: str = "jev"):
    raw = pickle.loads(Path(data).read_bytes())
    meta, crops = raw["meta"][:limit or None], raw["crops"]
    items = [(m["key"], [(crops[f"{m['key']}|{j}|plain"], crops[f"{m['key']}|{j}|mask"]) for j in range(len(m["views"]))], crops[f"{m['key']}|som"])
             for m in meta]
    keys = {m["key"] for m in meta}
    qs = [q for q in json.loads(Path(questions).read_text()) if q["card"] in keys]
    qq = [(q["key"], crops[f"{q['card']}|som"], q["state"], q["question"], q["options"]) for q in qs]
    if stage == "qwen":  # the Q2 / Q5 questions only (the yes / no form was Jev's alone)
        r = qwen.remote([q for q in qq if not q[0].endswith("|qyn")])
        Path(out).write_bytes(pickle.dumps(r, protocol=4))
        print(json.dumps({k: v for k, v in r.items() if k != "probs"}, indent=1))
        return
    phrases = [p for g in json.loads(Path(prompts).read_text()).values() for p in g["phrases"]]
    t = time.time()
    h = embed.spawn(items, phrases)
    jev = caller.remote(qq, per_request)
    emb = pickle.loads(h.get())
    Path(out).write_bytes(pickle.dumps({"jev": jev, "embed": emb, "wall_s": round(time.time() - t, 1)}, protocol=4))
    print(json.dumps({"jev": {k: v for k, v in jev.items() if k not in ("probs", "trips", "single")} | {"trips": jev["trips"][:3], "single": jev["single"][:3]},
                      "embed": {"timing": emb["timing"], "gpu": emb["gpu"]}, "wall_s": round(time.time() - t, 1)}, indent=1, default=str)[:4000])
