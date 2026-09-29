"""X13 hazards: Jev-Omni (the calibrated multiple-choice decider) and SigLIP 2 so400m (look-alike embeddings) as one HTTPS
service on its own A100-80GB (the separate server: nothing shares its GPU), and a client container on CPU that calls it over
the network the way the core container would. Latency is measured on both sides: the client's round trip and the service's
own time from request received to response ready; the difference is the network hop (plus JSON/base64 work).

Model load and warm-up happen at container start and are timed apart (boot); every request is analysis time. Labels never
enter a container: the answers come back as raw option probabilities and are scored locally (scripts/x13_hazards.py score).

  modal run modal_apps/x13_jev_service.py::setup                 # SigLIP 2 so400m weights into the panoptes-x8-models volume
  modal run modal_apps/x13_jev_service.py --work W [--limit N]   # W/items.json -> W/answers.json
"""
import base64
import io
import json
import os
import sys
import time
from pathlib import Path

import modal

JEV = "akhilaaa3/Jev-Omni"
JEV_FILES = ["config.json", "generation_config.json", "model*.safetensors*", "processor_config.json", "tokenizer.json", "tokenizer_config.json",
             "chat_template.jinja", "decision_config.json", "head.pt", "jev_omni.py", "sha256.json", "verification.json", "README.md"]
SIGLIP = "google/siglip2-so400m-patch14-384"  # Apache-2.0 (model card)
BATCH = 8  # X8's batched forward: 74-131 ms/question at 8 a batch

app = modal.App("panoptes-x13-jev-service")
VOL = modal.Volume.from_name("panoptes-x8-models")  # X8 put Jev-Omni's weights here (HF cache under /v/x8/hf)
image = (modal.Image.debian_slim(python_version="3.11")
         .apt_install("libgl1", "libglib2.0-0", "ffmpeg")
         .pip_install("torch==2.14.0", "torchvision", "transformers==5.17.0", "accelerate", "pillow", "numpy", "fastapi[standard]",
                      "nvidia-ml-py", "sentencepiece", "protobuf", "soundfile", "librosa", "av")
         .env({"HF_HUB_OFFLINE": "1", "HF_HOME": "/v/x8/hf", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "USE_TF": "0"}))
client_image = modal.Image.debian_slim(python_version="3.11").pip_install("numpy", "pillow")


@app.function(image=image, cpu=8, memory=32768, timeout=3600, retries=0, volumes={"/v/x8": VOL})
def setup():
    """SigLIP 2 so400m to the volume (public weights, no token); Jev-Omni's loader imports (no weights loaded)."""
    os.environ["HF_HUB_OFFLINE"] = "0"
    from huggingface_hub import snapshot_download
    t = time.time()
    sig = snapshot_download(SIGLIP)
    VOL.commit()
    jev = snapshot_download(JEV, allow_patterns=JEV_FILES)
    src = Path(jev, "jev_omni.py").read_text()
    sys.path.insert(0, jev)
    import jev_omni  # noqa: F401
    return {"s": round(time.time() - t, 1), "siglip_files": sorted(os.listdir(sig)), "jev_imports": [l for l in src.splitlines() if l.startswith(("import ", "from "))]}


class Jev:
    """X8's wrapper (modal_apps/x8_decider.py, fx-x8-jev): the card's loader (jev_omni.load_jev_omni) plus a batched forward of the
    same prompt and head, right padding, each row's last real token."""

    def __init__(self):
        import torch
        from huggingface_hub import snapshot_download
        path = snapshot_download(JEV, allow_patterns=JEV_FILES)
        sys.path.insert(0, path)
        import jev_omni
        self.mod, self.torch, self.path = jev_omni, torch, path
        self.clf = jev_omni.load_jev_omni()
        self.full = {}
        _, decoder = jev_omni._find_backbone(self.clf.model)
        decoder.register_forward_hook(lambda _m, _a, out: self.full.__setitem__(
            "h", out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]))

    def batch(self, reqs):
        """reqs: [(PIL images, state, question, options)] -> [probabilities]."""
        torch, clf = self.torch, self.clf
        convs = [[{"role": "user", "content": [{"type": "image", "image": im} for im in ims] +
                   [{"type": "text", "text": self.mod._prompt(st, q, opts)}]}] for ims, st, q, opts in reqs]
        proc = clf.processor
        proc.tokenizer.padding_side = "right"
        inputs = proc.apply_chat_template(convs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
                                          padding=True, enable_thinking=False)
        inputs = {k: v.to("cuda", dtype=torch.bfloat16) if torch.is_floating_point(v) else v.to("cuda") for k, v in inputs.items()}
        last = inputs["attention_mask"].sum(1) - 1
        clf._capture.clear()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            clf.model(**inputs, use_cache=False, **clf._extra)
            hidden = self.full["h"][torch.arange(len(reqs), device=last.device), last].float()
            logits = clf.head(hidden, torch.tensor([len(r[3]) for r in reqs], device="cuda"))
        return [logits[i, :len(r[3])].float().softmax(-1).cpu().tolist() for i, r in enumerate(reqs)]


def pil(b64):
    from PIL import Image
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


@app.function(image=image, gpu="A100-80GB", cpu=8, memory=65536, timeout=3600, volumes={"/v/x8": VOL}, max_containers=1,  # web: no retries at all
              scaledown_window=600, startup_timeout=1200)
@modal.asgi_app()
def service():
    """One A100-80GB: Jev-Omni + SigLIP 2 so400m, loaded and warmed at start (boot timed apart). POST /decide {items: [{images:
    [b64 JPEG], state, question, options}]} -> {probs, server_s, gpu_s}; POST /embed {images: [b64]} -> {emb (float16 b64, unit
    vectors), dim, server_s}; GET /stats -> boot, memory (nvml whole device, torch peaks)."""
    import numpy as np
    import pynvml
    import torch
    from fastapi import FastAPI
    from transformers import AutoModel, AutoProcessor
    t0 = time.perf_counter()
    boot = {}
    jev = Jev()
    boot["jev_loaded_s"] = round(time.perf_counter() - t0, 2)
    sig = AutoModel.from_pretrained(SIGLIP, torch_dtype=torch.bfloat16).to("cuda").eval()
    sproc = AutoProcessor.from_pretrained(SIGLIP)
    boot["siglip_loaded_s"] = round(time.perf_counter() - t0, 2)
    from PIL import Image
    im = Image.new("RGB", (448, 448), (120, 120, 120))
    jev.batch([([im], "warm up", "Is this grey?", ["yes", "no"])] * 2)
    with torch.inference_mode():
        sig.get_image_features(**{k: v.to("cuda") for k, v in sproc(images=[im], return_tensors="pt").items()})
    torch.cuda.synchronize()
    boot["warm_s"] = round(time.perf_counter() - t0, 2)
    boot["resident_gib_torch_reserved"] = round(torch.cuda.memory_reserved() / 2 ** 30, 2)
    boot["resident_gib_torch_allocated"] = round(torch.cuda.memory_allocated() / 2 ** 30, 2)
    ver = json.loads(Path(jev.path, "verification.json").read_text())  # the card's own cases: does this load reproduce them?
    got = [jev.clf.predict(**c)["probabilities"] for c in ver["cases"]]
    boot["card_verification_worst_abs_diff"] = round(max(abs(g[k] - r[k]) for g, r in zip(got, ver["reference"]) for k in r), 5)
    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    boot["gpu"] = pynvml.nvmlDeviceGetName(h)
    torch.cuda.reset_peak_memory_stats()
    stats = {"boot": boot, "nvml_peak_gib": 0., "requests": 0}

    def nvml():
        stats["nvml_peak_gib"] = max(stats["nvml_peak_gib"], round(pynvml.nvmlDeviceGetMemoryInfo(h).used / 2 ** 30, 2))

    api = FastAPI()

    @api.post("/decide")
    def decide(body: dict):
        try:
            return _decide(body)
        except Exception:  # noqa: BLE001  the traceback to the caller (a smoke test found a 500 with nothing to read)
            import traceback
            return {"error": traceback.format_exc()[-3000:]}

    def _decide(body):
        t = time.perf_counter()
        reqs = [([pil(b) for b in x["images"]], x["state"], x["question"], x["options"]) for x in body["items"]]
        g = time.perf_counter()
        out = []
        for i in range(0, len(reqs), int(body.get("batch") or BATCH)):
            out += jev.batch(reqs[i:i + int(body.get("batch") or BATCH)])
        torch.cuda.synchronize()
        nvml()
        stats["requests"] += 1
        return {"probs": out, "gpu_s": round(time.perf_counter() - g, 4), "server_s": round(time.perf_counter() - t, 4)}

    @api.post("/embed")
    def embed(body: dict):
        t = time.perf_counter()
        ims = [pil(b) for b in body["images"]]
        with torch.inference_mode():
            x = sproc(images=ims, return_tensors="pt")
            e = sig.get_image_features(**{k: v.to("cuda") for k, v in x.items()})
            e = getattr(e, "pooler_output", e)
            e = torch.nn.functional.normalize(e.float(), dim=-1).cpu().numpy().astype(np.float16)
        nvml()
        return {"emb": base64.b64encode(e.tobytes()).decode(), "dim": int(e.shape[1]), "server_s": round(time.perf_counter() - t, 4)}

    @api.get("/stats")
    def get_stats():
        nvml()
        return {**stats, "torch_peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2 ** 30, 2),
                "torch_peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)}

    return api


def post(url, path, body, timeout=900):
    import urllib.request
    data = json.dumps(body).encode()
    t = time.perf_counter()
    req = urllib.request.Request(url.rstrip("/") + path, data=data, headers={"Content-Type": "application/json"})
    out = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
    if "error" in out:
        raise RuntimeError(out["error"])
    return out, round(time.perf_counter() - t, 4), len(data)


@app.function(image=client_image, cpu=4, memory=8192, timeout=3600, retries=0)
def client(url, specs, images, opts):
    """The core container's side (CPU, another container): calls the service over HTTPS. 1) per-question requests (one question a
    request, sequential): round trip vs the service's own time; 2) one request per video and question family (the batched
    pattern: everything a judge wave has to ask); 3) embeddings in chunks. -> answers, latency records."""
    import urllib.request
    import numpy as np
    requests, embeds = expand(specs, images)
    t0 = time.perf_counter()
    for _ in range(60):  # the service loads Jev + SigLIP at start: wait for it (boot, not analysis)
        try:
            urllib.request.urlopen(url.rstrip("/") + "/stats", timeout=600).read()
            break
        except Exception:  # noqa: BLE001
            time.sleep(10)
    ready_s = round(time.perf_counter() - t0, 1)
    post(url, "/decide", {"items": [r["req"] for r in requests[:2]]})  # warm the path
    single = []
    for r in requests[:opts.get("single", 60)]:
        out, rtt, nbytes = post(url, "/decide", {"items": [r["req"]], "batch": 1})
        single.append({"key": r["key"], "rtt_s": rtt, "server_s": out["server_s"], "gpu_s": out["gpu_s"], "bytes": nbytes})
    groups = {}
    for r in requests:
        groups.setdefault(r["group"], []).append(r)
    answers, batched = {}, []
    for g, rs in groups.items():
        for i in range(0, len(rs), opts.get("per_request", 400)):
            chunk = rs[i:i + opts.get("per_request", 400)]
            out, rtt, nbytes = post(url, "/decide", {"items": [r["req"] for r in chunk], "batch": BATCH})
            for r, p in zip(chunk, out["probs"]):
                answers[r["key"]] = p
            batched.append({"group": g, "n": len(chunk), "rtt_s": rtt, "server_s": out["server_s"], "gpu_s": out["gpu_s"], "bytes": nbytes})
    emb, emb_lat = {}, []
    for i in range(0, len(embeds), 64):
        chunk = embeds[i:i + 64]
        out, rtt, nbytes = post(url, "/embed", {"images": [b for _, b in chunk]})
        e = np.frombuffer(base64.b64decode(out["emb"]), np.float16).reshape(len(chunk), out["dim"])
        for (k, _), v in zip(chunk, e):
            emb[k] = base64.b64encode(v.tobytes()).decode()
        emb_lat.append({"n": len(chunk), "rtt_s": rtt, "server_s": out["server_s"], "bytes": nbytes})
    stats = json.loads(urllib.request.urlopen(url.rstrip("/") + "/stats", timeout=60).read())
    return {"answers": answers, "embeddings": emb, "single": single, "batched": batched, "embed_latency": emb_lat, "stats": stats,
            "client_ready_wait_s": ready_s, "client_total_s": round(time.perf_counter() - t0, 1)}


# ---------- local: the requests from items.json ----------

GRID_STATE = ("This image comes from a video of a workplace (a shop floor, warehouse, store, lab or office). It shows one object in four "
              "tiles: top left, a close view with the object outlined in white and tagged [1] (other things carry other numbers); top "
              "right, the same view without marks; bottom left, the object outlined in another view (black: no other view); bottom "
              "right, the whole camera frame of the first view with only the object outlined, to show its surroundings.")
TILES_STATE = ("These images come from a video of a workplace (a shop floor, warehouse, store, lab or office). Image 1: a close view with "
               "the object outlined in white and tagged [1] (other things carry other numbers). Image 2: the same view without marks. "
               "Image 3: the whole camera frame with only the object outlined, to show its surroundings.")
FRAME_STATE = "This is a frame from a video of a workplace (a shop floor, warehouse, store, lab or office)."  # X8's D_STATE


def b64(raw):
    return base64.b64encode(raw).decode()


def tiles(raw):
    """The 2 x 2 evidence -> (marked view, plain view, context) JPEGs (hazard.evidence's layout, 384 px tiles)."""
    from PIL import Image
    im = Image.open(io.BytesIO(raw)).convert("RGB")
    t = im.width // 2
    out = []
    for box in ((0, 0, t, t), (t, 0, 2 * t, t), (t, t, 2 * t, 2 * t)):
        b = io.BytesIO()
        im.crop(box).save(b, format="JPEG", quality=90)
        out.append(b.getvalue())
    return out


def expand(specs, images):
    """(specs, {image id: JPEG bytes}) -> (decide requests [{key, group, req}], embed requests [(key, b64)]) (in the client)."""
    cut, reqs, embeds, seen = {}, [], [], set()
    for sp in specs:
        raw = images[sp["image"]]
        if sp["variant"] == "tiles" or sp.get("embed_tiles"):
            cut.setdefault(sp["image"], tiles(raw))
        ims = [b64(r) for r in cut[sp["image"]]] if sp["variant"] == "tiles" else [b64(raw)]
        reqs.append({"key": sp["key"], "group": sp["group"], "req": {"images": ims, "state": sp["state"], "question": sp["question"], "options": sp["options"]}})
        for v in (("plain", "context") if sp.get("embed_tiles") else ("frame",)):
            k = f"{sp['image']}|{v}"
            if k not in seen:
                seen.add(k)
                embeds.append((k, b64(cut[sp["image"]][1 if v == "plain" else 2]) if v != "frame" else b64(raw)))
    return reqs, embeds


def build(items, limit=0):
    """items.json -> (specs, {image id: bytes}): per object item two variants (the 2 x 2 grid as one image, as Gemini sees it; its
    three tiles as separate images), per frame one; object images are embedded as their plain and context tiles, frames whole."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from fast_report import hazard
    specs, images = [], {}
    for x in items[:limit or None]:
        images.setdefault(x["image"], Path(x["image"]).read_bytes())
        if x["form"] == "frame":
            variants = {"frame": (FRAME_STATE, x["question"], ["Yes", "No"])}
        else:
            q = hazard.QUESTIONS[x["q"]]
            variants = {"grid": (GRID_STATE, q, ["yes", "no"]), "tiles": (TILES_STATE, q, ["yes", "no"])}
        for v, (st, q, opts) in variants.items():
            specs.append({"key": f"{x['key']}|{v}", "group": f"{x['form']}|{x['video']}|{v}", "image": x["image"], "variant": v, "state": st,
                          "question": q, "options": opts, "embed_tiles": x["form"] != "frame"})
    return specs, images


@app.local_entrypoint()
def main(work: str, limit: int = 0, single: int = 60, local_calls: int = 10):
    import urllib.request
    w = Path(work)
    items = json.loads((w / "items.json").read_text())
    specs, images = build(items, limit)
    reqs, _ = expand(specs[:local_calls], images)
    print(f"{len(specs)} questions, {len(images)} images ({sum(map(len, images.values())) / 1e6:.1f} MB)", flush=True)
    url = service.get_web_url()
    t = time.time()
    out = client.remote(url, specs, images, {"single": single})
    out["client_call_wall_s"] = round(time.time() - t, 1)
    local = []  # the same per-question call from this Mac (another network) for reference
    for r in reqs[:local_calls]:
        o, rtt, nbytes = post(url, "/decide", {"items": [r["req"]], "batch": 1})
        local.append({"key": r["key"], "rtt_s": rtt, "server_s": o["server_s"], "bytes": nbytes})
    out["local_single"] = local
    out["url_kind"] = "modal ephemeral web endpoint (HTTPS), service and client in different containers"
    (w / "answers.json").write_text(json.dumps(out))
    print(json.dumps({k: out[k] for k in ("stats", "client_ready_wait_s", "client_total_s", "client_call_wall_s")}, indent=1))
    print(json.dumps({"single": out["single"][:5], "batched": out["batched"][:6]}, indent=1)[:3000])
