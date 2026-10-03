"""X13: naming without a run-time VLM. Frozen encoders + an open-vocabulary detector over the crop dataset of
scripts/x13_naming.py (features), and Jev-Omni as a service on its own GPU called from a separate CPU container (jev).
No training of any model. Labels never enter a container: it returns embeddings, detections and option probabilities.

  modal run modal_apps/x13_naming.py::setup                                   # weights -> panoptes-x13-models (CPU)
  modal run modal_apps/x13_naming.py --stage features --data DATASET.pkl --out FEATURES.pkl
  modal run modal_apps/x13_naming.py --stage jev --data DATASET.pkl --questions Q.json --out JEV.pkl

features: one A100-80GB (isolated model benchmark, not the 2-GPU production container). Model load is timed apart; each
stage is timed from its decoded inputs in the container to its result in memory.
jev: the service = one A100-80GB container (Jev-Omni only); the caller = a CPU container in the same region, standing in
for the core; each request's round trip is timed at the caller and its compute inside the service, so the hop is the rest.
"""
import io
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]

MODELS = {"siglip2-base": "google/siglip2-base-patch16-224", "siglip2-so400m": "google/siglip2-so400m-patch16-384",
          "pe-core-l": "timm/PE-Core-L-14-336", "dinov2-l": "facebook/dinov2-large", "owlv2-base": "google/owlv2-base-patch16-ensemble"}
JEV = "akhilaaa3/Jev-Omni"
JEV_FILES = ["config.json", "generation_config.json", "model*.safetensors*", "processor_config.json", "tokenizer.json", "tokenizer_config.json",
             "chat_template.jinja", "decision_config.json", "head.pt", "jev_omni.py", "sha256.json", "verification.json", "README.md"]
X13_HF = "/v/x13/hf"
TEMPLATES = ("a photo of a {}.", "a photo of a {} in a workplace.")
BATCH, OWL_BATCH, JEV_BATCH = 256, 8, 8

app = modal.App("panoptes-x13-naming")
VOLUMES = {"/v/x8": modal.Volume.from_name("panoptes-x8-models"), "/v/x13": modal.Volume.from_name("panoptes-x13-models", create_if_missing=True)}
# X8's image line for line (its layers are cached), plus open_clip for PE-Core
image = (modal.Image.from_registry("nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04", add_python="3.11")
         .apt_install("git", "libgl1", "libglib2.0-0", "libgomp1")
         .pip_install("torch==2.14.0", "torchvision", "xformers", "transformers==5.17.0", "accelerate", "addict", "pillow", "scipy",
                      "open3d==0.19.0", "shapely", "pydantic", "opencv-python-headless", "sentencepiece",
                      "git+https://github.com/ByteDance-Seed/Depth-Anything-3.git@3d835ec1a5802d64a8b8b15f817a1ab54809bfe4")
         .run_commands("python -m venv /opt/vllm && PIP_EXTRA_INDEX_URL= /opt/vllm/bin/pip install -q vllm==0.11.0 transformers==4.57.1 pillow")
         .apt_install("ffmpeg")
         .pip_install("soundfile", "librosa", "laya==0.3.21")
         .pip_install("open_clip_torch")
         .env({"HF_HUB_OFFLINE": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "USE_TF": "0", "HF_HOME": "/v/x8/hf"})
         .add_local_python_source("fast_report", "ehs_spatial"))


def card_licence(path):
    """The model card's own `license:` (README front matter), the primary source for the licence."""
    p = Path(path, "README.md")
    if not p.exists():
        return None
    head = p.read_text(errors="ignore").split("---")
    for line in (head[1] if len(head) > 2 else "").splitlines():
        if line.strip().lower().startswith("license"):
            return line.split(":", 1)[1].strip()
    return None


@app.function(image=image, cpu=8, memory=32768, timeout=3600, retries=0, volumes=VOLUMES)
def setup():
    os.environ["HF_HUB_OFFLINE"] = "0"
    from huggingface_hub import snapshot_download
    out = {}
    for k, repo in MODELS.items():
        t = time.time()
        p = snapshot_download(repo, cache_dir=X13_HF, ignore_patterns=["*.msgpack", "*.h5", "*.onnx", "flax*", "tf_*", "*.ot"])
        out[k] = {"repo": repo, "s": round(time.time() - t, 1), "licence": card_licence(p), "files": sorted(os.listdir(p))[:30],
                  "gb": round(sum(f.stat().st_size for f in Path(p).rglob("*") if f.is_file()) / 1e9, 2)}
    VOLUMES["/v/x13"].commit()
    import open_clip
    out["open_clip"] = open_clip.__version__
    print(json.dumps(out, indent=1))
    return out


# ---------------------------------------------------------------- features

def _decode(data):
    """dataset -> uint8 crops (N,S,S,3) RGB, masks (N,S,S) bool, view index [(card i, view j)], som crops (M,448,448,3)."""
    import cv2
    import numpy as np
    meta, crops = data["meta"], data["crops"]
    idx, imgs, masks, soms = [], [], [], []
    for i, m in enumerate(meta):
        for j in range(len(m["views"])):
            imgs.append(cv2.imdecode(np.frombuffer(crops[f"{m['key']}|{j}|plain"], np.uint8), 1)[:, :, ::-1])
            masks.append(cv2.imdecode(np.frombuffer(crops[f"{m['key']}|{j}|mask"], np.uint8), 0) > 127)
            idx.append((i, j))
        s = cv2.imdecode(np.frombuffer(crops[f"{m['key']}|som"], np.uint8), 1)[:, :, ::-1]
        side = max(s.shape[:2])
        pad = np.full((side, side, 3), 128, np.uint8)
        pad[:s.shape[0], :s.shape[1]] = s
        soms.append(cv2.resize(pad, (384, 384), interpolation=cv2.INTER_AREA))
    return np.stack(imgs), np.stack(masks), np.array(idx), np.stack(soms)


class Encoder:
    """One frozen image(-text) encoder: image(uint8 NHWC, masks or None) -> unit embeddings; text(classes) -> unit class
    embeddings (prompt ensemble over TEMPLATES x the class word and its synonyms), or None for image-only models."""

    def __init__(self, key):
        import torch
        self.key, self.torch = key, torch
        if key.startswith("siglip2"):
            from transformers import AutoModel, AutoTokenizer
            self.model = AutoModel.from_pretrained(MODELS[key], cache_dir=X13_HF, torch_dtype=torch.bfloat16).cuda().eval()
            self.tok = AutoTokenizer.from_pretrained(MODELS[key], cache_dir=X13_HF)
            self.side, self.mean, self.std = self.model.config.vision_config.image_size, (.5,) * 3, (.5,) * 3
            self.scale = float(self.model.logit_scale.exp())
        elif key == "pe-core-l":
            import open_clip
            self.model, _, pre = open_clip.create_model_and_transforms("hf-hub:" + MODELS[key], cache_dir=X13_HF)
            self.model = self.model.cuda().eval().to(torch.bfloat16)
            self.tok = open_clip.get_tokenizer("hf-hub:" + MODELS[key])
            norm = [t for t in pre.transforms if type(t).__name__ == "Normalize"][0]
            self.side, self.mean, self.std = 336, tuple(norm.mean), tuple(norm.std)
            self.scale = float(self.model.logit_scale.exp())
        elif key == "dinov2-l":
            from transformers import AutoModel
            self.model = AutoModel.from_pretrained(MODELS[key], cache_dir=X13_HF, torch_dtype=torch.bfloat16).cuda().eval()
            self.side, self.mean, self.std, self.scale = 224, (.485, .456, .406), (.229, .224, .225), None

    def _pool(self, out):
        return out.pooler_output if hasattr(out, "pooler_output") else out

    def image(self, arr, masks=None):
        import numpy as np
        torch = self.torch
        F = torch.nn.functional
        mean = torch.tensor(self.mean, device="cuda")[:, None, None]
        std = torch.tensor(self.std, device="cuda")[:, None, None]
        out = []
        for s in range(0, len(arr), BATCH):
            x = torch.from_numpy(np.ascontiguousarray(arr[s:s + BATCH])).cuda().permute(0, 3, 1, 2).float() / 255
            if masks is not None:  # background outside the mask to mid grey (cascade.Embedder.crops)
                x = torch.where(torch.from_numpy(masks[s:s + BATCH]).cuda()[:, None], x, .5)
            x = F.interpolate(x, size=(self.side, self.side), mode="bicubic", antialias=True, align_corners=False).clamp(0, 1)
            x = ((x - mean) / std).to(torch.bfloat16)
            with torch.inference_mode():
                if self.key == "pe-core-l":
                    e = self.model.encode_image(x)
                elif self.key == "dinov2-l":
                    e = self.model(pixel_values=x).pooler_output
                else:
                    e = self._pool(self.model.get_image_features(pixel_values=x))
            out.append(F.normalize(e.float(), dim=-1).cpu())
        return torch.cat(out).numpy()

    def text(self, classes):
        """classes: [(class, [synonyms])] -> (C, D) unit class embeddings."""
        if self.scale is None:
            return None
        torch = self.torch
        F = torch.nn.functional
        rows = []
        for c, syn in classes:
            prompts = [t.format(w) for w in [c, *syn[:4]] for t in TEMPLATES]
            with torch.inference_mode():
                if self.key == "pe-core-l":
                    e = self.model.encode_text(self.tok(prompts).cuda())
                else:
                    t = self.tok(prompts, padding="max_length", max_length=64, truncation=True, return_tensors="pt").to("cuda")
                    e = self._pool(self.model.get_text_features(**t))
            rows.append(F.normalize(F.normalize(e.float(), dim=-1).mean(0), dim=0).cpu())
        return torch.stack(rows).numpy()


def _owl(frames, classes):
    """OWLv2 over the keyframes with one text query per taxonomy class -> {frame key: [(x0,y0,x1,y1, class i, score)]}."""
    import cv2
    import numpy as np
    import torch
    from transformers import Owlv2ForObjectDetection, Owlv2Processor
    proc = Owlv2Processor.from_pretrained(MODELS["owlv2-base"], cache_dir=X13_HF)
    model = Owlv2ForObjectDetection.from_pretrained(MODELS["owlv2-base"], cache_dir=X13_HF, torch_dtype=torch.float16).cuda().eval()
    queries = [f"a photo of a {c}" for c, _ in classes]
    keys = sorted(frames)
    out, t_model = {}, 0.
    t0 = time.perf_counter()
    for s in range(0, len(keys), OWL_BATCH):
        ks = keys[s:s + OWL_BATCH]
        ims = [np.ascontiguousarray(cv2.imdecode(np.frombuffer(frames[k], np.uint8), 1)[:, :, ::-1]) for k in ks]
        inp = proc(text=[queries] * len(ims), images=ims, return_tensors="pt").to("cuda")
        inp["pixel_values"] = inp["pixel_values"].half()
        torch.cuda.synchronize()
        t = time.perf_counter()
        with torch.inference_mode():
            o = model(**inp)
        torch.cuda.synchronize()
        t_model += time.perf_counter() - t
        side = max(ims[0].shape[:2])  # OWLv2 pads to a square: boxes are in the padded square's pixels
        post = getattr(proc, "post_process_grounded_object_detection", None) or proc.post_process_object_detection  # transformers 5 renamed it
        res = post(outputs=o, threshold=.05, target_sizes=[(side, side)] * len(ims))
        for k, r in zip(ks, res):
            b, sc, lb = r["boxes"].float().cpu().numpy(), r["scores"].float().cpu().numpy(), r["labels"].cpu().numpy()
            top = np.argsort(-sc)[:300]
            out[k] = [(*map(float, b[i]), int(lb[i]), float(sc[i])) for i in top]
    return out, {"frames": len(keys), "queries": len(queries), "total_s": round(time.perf_counter() - t0, 2), "model_s": round(t_model, 2),
                 "per_frame_s": round((time.perf_counter() - t0) / len(keys), 4)}


@app.function(image=image, gpu="A100-80GB", cpu=8, memory=65536, timeout=3600, retries=0, volumes=VOLUMES)
def features(data, classes):
    os.environ["HF_HUB_CACHE"] = X13_HF  # before any Hugging Face import: open_clip's hf-hub lookups land in this volume
    import numpy as np
    import torch
    from fast_report.instrument import Vram
    vram = Vram([0]).start()
    t = time.perf_counter()
    arr, masks, idx, soms = _decode(data)
    res = {"idx": idx, "timing": {"decode_s": round(time.perf_counter() - t, 2), "views": len(arr), "cards": len(soms)}, "emb": {}, "text": {}, "load_s": {}}
    for key in ("siglip2-base", "siglip2-so400m", "pe-core-l", "dinov2-l"):
        try:
            t = time.perf_counter()
            enc = Encoder(key)
            enc.image(arr[:8])  # warm-up, not timed
            res["load_s"][key] = round(time.perf_counter() - t, 2)
            tm = {}
            for var, a, m in (("plain", arr, None), ("masked", arr, masks), ("som", soms, None)):
                torch.cuda.synchronize()
                t = time.perf_counter()
                res["emb"][(key, var)] = enc.image(a, m).astype(np.float16)
                torch.cuda.synchronize()
                tm[var] = {"n": len(a), "s": round(time.perf_counter() - t, 3), "ms_per_crop": round(1000 * (time.perf_counter() - t) / len(a), 3)}
            t = time.perf_counter()
            res["text"][key] = enc.text(classes)
            tm["text_s"] = round(time.perf_counter() - t, 3)
            tm["scale"], tm["peak_gib_torch"] = enc.scale, round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
            res["timing"][key] = tm
            del enc
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        except Exception as error:  # one encoder failing must show in the record, not stop the others
            res["timing"][key] = {"error": repr(error)[:500]}
    try:
        res["owl"], res["timing"]["owlv2-base"] = _owl(data["frames"], classes)
    except Exception as error:
        res["timing"]["owlv2-base"] = {"error": repr(error)[:500]}
    res["gpu"] = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.splitlines()
    res["vram_peak_gib"] = round(max((g[0] for g in vram.gb), default=0.), 2)
    vram.stop()
    return pickle.dumps(res, protocol=4)


@app.function(image=image, gpu="A100-80GB", cpu=8, memory=32768, timeout=1800, retries=0, volumes=VOLUMES)
def owl(frames, classes):
    os.environ["HF_HUB_CACHE"] = X13_HF
    det, timing = _owl(frames, classes)
    return pickle.dumps({"owl": det, "timing": timing, "gpu": subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.splitlines()})


# ---------------------------------------------------------------- Jev-Omni as a service on its own GPU

@app.cls(image=image, gpu="A100-80GB", cpu=4, memory=49152, timeout=3600, retries=0, volumes=VOLUMES, max_containers=1)
class JevService:
    @modal.enter()
    def load(self):
        import torch
        t = time.perf_counter()
        os.environ["HF_HOME"] = "/v/x8/hf"
        from huggingface_hub import snapshot_download
        path = snapshot_download(JEV, allow_patterns=JEV_FILES)
        sys.path.insert(0, path)
        import jev_omni
        self.mod, self.torch, self.path = jev_omni, torch, path
        self.clf = jev_omni.load_jev_omni()
        self.full = {}
        _, decoder = jev_omni._find_backbone(self.clf.model)
        decoder.register_forward_hook(lambda _m, _a, out: self.full.__setitem__("h", out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]))
        self.load_s = round(time.perf_counter() - t, 2)

    def _batch(self, reqs):
        """X8's batched forward (right padding, each row's last real token, the card's head). reqs: [(PIL, state, q, options)]."""
        torch, clf = self.torch, self.clf
        convs = [[{"role": "user", "content": [{"type": "image", "image": im}, {"type": "text", "text": self.mod._prompt(st, q, opts)}]}]
                 for im, st, q, opts in reqs]
        proc = clf.processor
        proc.tokenizer.padding_side = "right"
        inputs = proc.apply_chat_template(convs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt", padding=True,
                                          enable_thinking=False)
        inputs = {k: v.to("cuda", dtype=torch.bfloat16) if torch.is_floating_point(v) else v.to("cuda") for k, v in inputs.items()}
        last = inputs["attention_mask"].sum(1) - 1
        clf._capture.clear()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            clf.model(**inputs, use_cache=False, **clf._extra)
            hidden = self.full["h"][torch.arange(len(reqs), device=last.device), last].float()
            logits = clf.head(hidden, torch.tensor([len(r[3]) for r in reqs], device="cuda"))
        return [logits[i, :len(r[3])].float().softmax(-1).cpu().tolist() for i, r in enumerate(reqs)]

    @modal.method()
    def decide(self, qs):
        """qs: [(key, jpeg, state, question, options)] -> {"probs": {key: [p]}, "compute_s", "load_s", "peak_gib"}."""
        from PIL import Image
        t = time.perf_counter()
        reqs = [(Image.open(io.BytesIO(j)).convert("RGB"), st, q, o) for _, j, st, q, o in qs]
        out = {}
        for s in range(0, len(reqs), JEV_BATCH):
            for (k, *_), p in zip(qs[s:s + JEV_BATCH], self._batch(reqs[s:s + JEV_BATCH])):
                out[k] = [round(v, 6) for v in p]
        self.torch.cuda.synchronize()
        return {"probs": out, "compute_s": round(time.perf_counter() - t, 4), "load_s": self.load_s,
                "peak_gib": round(self.torch.cuda.max_memory_reserved() / 2 ** 30, 2), "licence": card_licence(self.path)}


@app.function(image=image, cpu=2, memory=8192, timeout=3600, retries=0)
def caller(questions, per_request):
    """The core's stand-in (no GPU): sends the questions to the service in requests of `per_request`, timing each round trip."""
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
    single = []  # one question per request: the hop's floor per call
    for q in questions[:20]:
        t = time.perf_counter()
        r = svc.decide.remote([q])
        single.append({"rtt_s": round(time.perf_counter() - t, 4), "compute_s": r["compute_s"]})
    return {"probs": probs, "trips": trips, "single": single, "cold_s": cold_s, "load_s": warm["load_s"], "peak_gib": r["peak_gib"],
            "licence": r["licence"], "total_s": round(time.perf_counter() - t_all, 2)}


@app.local_entrypoint()
def main(stage: str, data: str, out: str, questions: str = "", per_request: int = 64, limit: int = 0):
    raw = pickle.loads(Path(data).read_bytes())
    if limit and stage != "jev":  # smoke test: the first `limit` cards and their frames
        keep = raw["meta"][:limit]
        ks = {m["key"] for m in keep}
        fr = {f"{m['site']}|{v['frame']}" for m in keep for v in m["views"]}
        raw = {"meta": keep, "crops": {k: v for k, v in raw["crops"].items() if k.rsplit("|", 2)[0] in ks or k.rsplit("|", 1)[0] in ks},
               "frames": {k: v for k, v in raw["frames"].items() if k in fr}}
    t = time.time()
    if stage == "features":
        sys.path.insert(0, str(HERE.parent))
        from fast_report import cards
        classes = [(c, list(syn)) for fam in cards.TAXONOMY.values() for c, syn in fam.items()]
        res = features.remote({"meta": raw["meta"], "crops": raw["crops"], "frames": raw["frames"]}, classes)
        Path(out).write_bytes(res)
        r = pickle.loads(res)
        print(json.dumps({"timing": r["timing"], "load_s": r["load_s"], "gpu": r["gpu"], "client_s": round(time.time() - t, 1)}, indent=1, default=str))
    elif stage == "owl":
        sys.path.insert(0, str(HERE.parent))
        from fast_report import cards
        classes = [(c, list(syn)) for fam in cards.TAXONOMY.values() for c, syn in fam.items()]
        res = owl.remote(raw["frames"], classes)
        Path(out).write_bytes(res)
        r = pickle.loads(res)
        print(json.dumps({"timing": r["timing"], "gpu": r["gpu"], "client_s": round(time.time() - t, 1)}, indent=1))
    elif stage == "jev":
        qs = json.loads(Path(questions).read_text())
        qs = qs[:limit] if limit else qs
        qq = [(q["key"], raw["crops"][q["crop"]], q["state"], q["question"], q["options"]) for q in qs]
        r = caller.remote(qq, per_request)
        Path(out).write_bytes(pickle.dumps(r, protocol=4))
        print(json.dumps({k: v for k, v in r.items() if k != "probs"} | {"client_s": round(time.time() - t, 1)}, indent=1, default=str)[:4000])
