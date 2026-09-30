"""r5b/vocab: the VLM-free word sources measured off the pipeline, and the text embeddings the pipeline's zero-shot needs.

  pe_yoloe(): one A100-80GB (the pipeline's image and volumes). PE-Core-L (x13's weights) text embeddings with x13's prompt
      ensemble for every taxonomy class (the cascade's zero-shot; the old 102 are checked against the r4 bank's) and for the
      'pe' word list (fast_report.vocab.candidates over LVIS + Objects365 names from Ultralytics' dataset files); then per video
      the 'pe' source (fast_report.vocab.pe_scores on FRAMES frames) and YOLOE-26L prompt-free (Ultralytics' release weights,
      AGPL-3.0: internal only) per-frame scores, each timed on a second pass (the first warms the kernels).
  ram(): one A100-80GB, its own image (recognize-anything's pins): RAM++ swin-L tags per frame with its per-class thresholds
      and their sigmoid scores, timed the same way. Apache-2.0 code and weights.
Raw per-frame scores come back; ranking (fast_report.vocab.rank) runs locally, so every source is ranked by one rule.

  modal run modal_apps/r5b_vocab.py --out RUNS/r5b-vocab-tags-001
"""
import io
import json
import subprocess
import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parent if (HERE.parent / "fast_report").is_dir() else Path("/repo")
sys.path[:0] = [str(REPO), str(REPO / "modal_apps")]
try:
    from modal_apps.fast_report_app import VOLUMES, image  # noqa: E402  the pipeline's image: PE-Core (open_clip), Ultralytics 8.4.165
except ImportError:  # the RAM++ container (its own image): neither is used there
    VOLUMES, image = {"/v/models": modal.Volume.from_name("panoptes-fb-models")}, modal.Image.debian_slim()

PHASE2 = Path("/Users/adam/Desktop/panoptes-public/research-notes/phase2")
CLIPS = {"me340": "me340-165", "samsclub-a2": "samsclub-337", "walmart": "walmart-190"}
TEMPLATES = ("a photo of a {}.", "a photo of a {} in a workplace.")  # x13's prompt ensemble (modal_apps/x13_naming.py)
PF, YOLO_DIR, PF_CONF, PF_IMGSZ = "yoloe-26l-seg-pf.pt", "/v/r4/yolo", .1, 960
OUT_DIR = "/v/layers/label-bank/r5b"
app = modal.App("panoptes-r5b-vocab")
ram_image = (modal.Image.debian_slim(python_version="3.10").apt_install("git", "libgl1", "libglib2.0-0")
             .pip_install("torch==2.0.1", "torchvision==0.15.2", "timm==0.4.12", "transformers==4.25.1", "fairscale==0.4.4", "pillow",
                          "opencv-python-headless", "numpy<2", "huggingface_hub==0.16.4", "scipy")
             .pip_install("git+https://github.com/xinyu1205/recognize-anything.git", extra_options="--no-deps")
             .env({"HF_HOME": "/v/models/hf-r5b"}))


def frames_of(mp4, k):
    """k evenly spaced frames (fast_report.vocab.pick_frames) of an MP4 -> ([BGR], [frame index], decode s)."""
    import cv2
    import numpy as np
    from fast_report import vocab
    t = time.perf_counter()
    Path("/tmp/v.mp4").write_bytes(mp4)
    cap = cv2.VideoCapture("/tmp/v.mp4")
    idx = vocab.pick_frames(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), k)
    out = []
    for f in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        out.append(np.ascontiguousarray(cap.read()[1]))
    cap.release()
    return out, idx, round(time.perf_counter() - t, 3)


@app.function(image=image, gpu="A100-80GB", cpu=8, memory=32768, timeout=1800, retries=0, volumes=VOLUMES)
def pe_yoloe(videos: dict, k: int = 16):
    import hashlib
    import os
    import numpy as np
    import open_clip
    import torch
    import yaml
    from ultralytics.utils import ROOT
    from fast_report import cards, vocab
    from fast_report.cascade import PE_CORE, X13_HF, Encoders

    class Enc(Encoders):  # the cascade's PE path (Encoders.pe_embed) without DINOv2 and YOLOE
        def __init__(self, dev, pe, pre):
            norm = [x for x in pre.transforms if type(x).__name__ == "Normalize"][0]
            self.dev, self.torch, self.pe, self.pe_mean, self.pe_std = dev, torch, pe, tuple(norm.mean), tuple(norm.std)
    rec = {"gpu": subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.splitlines()}
    t = time.perf_counter()
    dev = torch.device("cuda:0")
    model, _, pre = open_clip.create_model_and_transforms(PE_CORE, cache_dir=X13_HF)
    model = model.to(dev).eval().to(torch.bfloat16)
    tok = open_clip.get_tokenizer(PE_CORE, cache_dir=X13_HF)  # offline: the config x13 cached
    scale = float(model.logit_scale.exp())

    def text(groups):
        """[[word, synonyms ...]] -> (n, D) unit rows: each prompt unit, averaged over TEMPLATES x the group's words, unit again
        (x13's Encoder.text), the prompts encoded 512 at a time."""
        prompts = [(g, tt.format(w)) for g, ws in enumerate(groups) for w in ws for tt in TEMPLATES]
        acc = None
        for s in range(0, len(prompts), 512):
            with torch.inference_mode():
                e = torch.nn.functional.normalize(model.encode_text(tok([p for _, p in prompts[s:s + 512]]).to(dev)).float(), dim=-1)
            acc = torch.zeros((len(groups), e.shape[1]), device=dev) if acc is None else acc
            acc.index_add_(0, torch.tensor([g for g, _ in prompts[s:s + 512]], device=dev), e)
        return torch.nn.functional.normalize(acc, dim=-1)
    classes = [(c, list(syn)) for fam in cards.TAXONOMY.values() for c, syn in fam.items()]
    zs = text([[c, *syn[:4]] for c, syn in classes])
    names = {}
    for f in ("lvis.yaml", "Objects365.yaml"):
        d = yaml.safe_load((ROOT / "cfg" / "datasets" / f).read_text())["names"]
        names[f] = [d[i] for i in sorted(d)]
    words = vocab.candidates(names["lvis.yaml"], names["Objects365.yaml"])
    wt = text([[w] for w in words])
    rec.update(text_s=round(time.perf_counter() - t, 2), words=len(words), lvis=len(names["lvis.yaml"]), o365=len(names["Objects365.yaml"]),
               zs_classes=len(classes), logit_scale=scale)
    buf = io.BytesIO()
    np.savez(buf, words=np.array(words), word_text=wt.cpu().numpy().astype(np.float16), zs_classes=np.array([c for c, _ in classes]),
             zs_text=zs.cpu().numpy().astype(np.float32), scale=np.float32(scale), templates=np.array(TEMPLATES))
    raw = buf.getvalue()
    sha = hashlib.sha256(raw).hexdigest()
    os.makedirs(OUT_DIR, exist_ok=True)
    Path(f"{OUT_DIR}/pe-text-{sha[:12]}.npz").write_bytes(raw)
    VOLUMES["/v/layers"].commit()
    rec["text_file"] = f"{OUT_DIR}/pe-text-{sha[:12]}.npz"
    enc = Enc(dev, model, pre)
    wt_dev = wt.to(dev)
    os.chdir(YOLO_DIR)  # the release weights download here once (Ultralytics' GitHub assets)
    from ultralytics import YOLOE
    t = time.perf_counter()
    pf = YOLOE(PF)
    pf_names = pf.names if isinstance(pf.names, list) else [pf.names[i] for i in range(len(pf.names))]
    VOLUMES["/v/r4"].commit()
    rec["pf_load_s"], rec["pf_classes"] = round(time.perf_counter() - t, 2), len(pf_names)
    out = {"record": rec, "videos": {}}
    for rnd in (0, 1):  # round 0 warms the kernels; round 1 is timed
        for name, mp4 in videos.items():
            fr, idx, dec_s = frames_of(mp4, k)
            torch.cuda.synchronize()
            t = time.perf_counter()
            pe = vocab.pe_scores(enc, fr, wt_dev, words, scale)
            torch.cuda.synchronize()
            pe_s = time.perf_counter() - t
            t = time.perf_counter()
            yo = []
            for r in pf.predict(fr, imgsz=PF_IMGSZ, conf=PF_CONF, half=True, verbose=False, max_det=300, device="cuda:0"):
                d = {}
                for c, p in zip(r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                    d[pf_names[int(c)]] = max(d.get(pf_names[int(c)], 0.), float(p))
                yo.append(d)
            torch.cuda.synchronize()
            yo_s = time.perf_counter() - t
            if rnd:
                out["videos"][name] = {"frames": idx, "decode_s": dec_s, "pe": pe, "pe_s": round(pe_s, 3), "yoloe_pf": yo, "yoloe_pf_s": round(yo_s, 3)}
    rec["peak_gib"] = round(torch.cuda.max_memory_reserved() / 2 ** 30, 2)
    return {**out, "text_npz": raw}


@app.function(image=ram_image, gpu="A100-80GB", cpu=8, memory=32768, timeout=1800, retries=0, volumes={"/v/models": VOLUMES["/v/models"]})
def ram(videos: dict, k: int = 16):
    import cv2
    import numpy as np
    import torch
    from huggingface_hub import hf_hub_download
    from PIL import Image
    from ram import get_transform
    from ram.models import ram_plus
    rec = {"gpu": subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.splitlines()}
    t = time.perf_counter()
    ckpt = hf_hub_download("xinyu1205/recognize-anything-plus-model", "ram_plus_swin_large_14m.pth")
    model = ram_plus(pretrained=ckpt, image_size=384, vit="swin_l").eval().to("cuda")
    VOLUMES["/v/models"].commit()
    tf = get_transform(image_size=384)
    got = []
    model.fc.register_forward_hook(lambda m, i, o: got.append(o.detach()))
    rec["load_s"] = round(time.perf_counter() - t, 2)
    tags, thr = model.tag_list, model.class_threshold.cpu().numpy()
    drop = set(np.asarray(getattr(model, "delete_tag_index", []), int).tolist())
    out = {"record": rec, "videos": {}}

    def pick(mp4):
        Path("/tmp/v.mp4").write_bytes(mp4)
        cap = cv2.VideoCapture("/tmp/v.mp4")
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idx = sorted({int((i + .5) * n / k) for i in range(k)})
        fr = []
        for f in idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            fr.append(cap.read()[1])
        return fr, idx
    for rnd in (0, 1):
        for name, mp4 in videos.items():
            fr, idx = pick(mp4)
            torch.cuda.synchronize()
            t = time.perf_counter()
            x = torch.stack([tf(Image.fromarray(f[..., ::-1])) for f in fr]).to("cuda")
            got.clear()
            with torch.inference_mode():
                model.generate_tag(x)
            p = torch.sigmoid(got[-1].squeeze(-1).float()).cpu().numpy()
            torch.cuda.synchronize()
            s = time.perf_counter() - t
            per = [{str(tags[j]): round(float(q[j]), 4) for j in np.flatnonzero(q > thr) if j not in drop} for q in p]
            if rnd:
                out["videos"][name] = {"frames": idx, "ram": per, "ram_s": round(s, 3)}
    rec["peak_gib"] = round(torch.cuda.max_memory_reserved() / 2 ** 30, 2)
    return out


@app.local_entrypoint()
def main(out: str, sources: str = "pe,ram", extra: str = ""):
    """extra: name=path.mp4 pairs (comma list) tagged beside the three bench videos."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    videos = {s: (PHASE2 / "data/clips" / c / "source-full.mp4").read_bytes() for s, c in CLIPS.items()}
    for kv in filter(None, extra.split(",")):
        k, v = kv.split("=")
        videos[k] = Path(v).read_bytes()
    t = time.time()
    calls = {}
    if "pe" in sources:
        calls["pe_yoloe"] = pe_yoloe.spawn(videos)
    if "ram" in sources:
        calls["ram"] = ram.spawn(videos)
    for k, c in calls.items():
        try:
            r = c.get()
        except Exception as error:  # noqa: BLE001  one source failing keeps the other's
            (out / f"{k}-error.txt").write_text(repr(error)[:4000])
            print(k, "failed:", repr(error)[:800], flush=True)
            continue
        if "text_npz" in r:
            (out / "pe-text.npz").write_bytes(r.pop("text_npz"))
        (out / f"{k}.json").write_text(json.dumps(r, indent=1))
        print(k, json.dumps(r["record"])[:1500], flush=True)
    (out / "client.json").write_text(json.dumps({"wall_s": round(time.time() - t, 1), "videos": list(videos)}, indent=1))
