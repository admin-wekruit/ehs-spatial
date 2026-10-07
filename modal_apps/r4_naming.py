"""Round 4 naming (r4/naming): YOLOE-26L with the canonical taxonomy baked in as its classes, and its detections on the x13
crop dataset's keyframes (the offline fit of the naming cascade's detector vote). Ultralytics YOLOE is AGPL-3.0: accepted
by the user for now (prefer an Apache detector where equal; OWLv2 was the x13 alternative).

  modal run modal_apps/r4_naming.py --data X13/dataset.pkl --out X13/yoloe.pkl

One A100-80GB, ephemeral. The baked model (/v/r4/yolo/TAXONOMY_PT) is what fast_report loads: no text encoder at run time.
Each class is the mean of the text embeddings of its word and up to 4 synonyms (the zero-shot's prompt ensemble).
"""
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
ULTRA = "ultralytics==8.4.165"
CLIP = "git+https://github.com/ultralytics/CLIP.git"  # YOLOE's text model imports it (the tokenizer); only the bake needs it
YOLOE_W, YOLO_DIR, TAXONOMY_PT, IMGSZ, CONF = "yoloe-26l-seg.pt", "/v/r4/yolo", "yoloe-26l-seg-taxonomy.pt", 960, .05

app = modal.App("panoptes-r4-naming")
VOLUMES = {"/v/r4": modal.Volume.from_name("panoptes-r4-naming", create_if_missing=True)}
image = (modal.Image.from_registry("nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04", add_python="3.11")
         .apt_install("git", "libgl1", "libglib2.0-0")
         .pip_install("torch==2.14.0", "torchvision", "opencv-python-headless", ULTRA, CLIP))


def bake(classes, out):
    """classes: [(class, [synonyms])] -> the YOLOE model with one class per taxonomy entry, saved to `out`."""
    import torch
    from ultralytics import YOLOE
    model = YOLOE(YOLOE_W)
    rows = []
    for c, syn in classes:
        pe = model.get_text_pe([c, *syn[:4]])  # (1, n, D)
        rows.append(torch.nn.functional.normalize(pe.float().mean(1), dim=-1))
    names = [c for c, _ in classes]
    model.set_classes(names, torch.stack(rows, 1).to(pe.dtype))
    model.save(out)
    return names


@app.function(image=image, gpu="A100-80GB", cpu=8, memory=32768, timeout=1800, retries=0, volumes=VOLUMES)
def yolo(frames, classes):
    """Bake (once) and detect: {frame key: [(x0, y0, x1, y1, class i, score)]} in source pixels, conf >= CONF."""
    import cv2
    import numpy as np
    import torch
    from ultralytics import YOLOE
    os.makedirs(YOLO_DIR, exist_ok=True)
    os.chdir(YOLO_DIR)  # the weights and the text encoder download here
    rec = {"gpu": subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.splitlines()}
    t = time.perf_counter()
    if not Path(TAXONOMY_PT).exists():
        bake(classes, TAXONOMY_PT)
        VOLUMES["/v/r4"].commit()
    rec["bake_s"] = round(time.perf_counter() - t, 2)
    t = time.perf_counter()
    model = YOLOE(TAXONOMY_PT)
    names = model.names if isinstance(model.names, list) else [model.names[i] for i in range(len(model.names))]
    assert names == [c for c, _ in classes], names[:5]
    rec["load_s"] = round(time.perf_counter() - t, 2)
    keys = sorted(frames)
    ims = [cv2.imdecode(np.frombuffer(frames[k], np.uint8), 1) for k in keys]  # BGR, as ultralytics expects numpy input
    model.predict(ims[:2], imgsz=IMGSZ, conf=CONF, half=True, verbose=False)  # warm-up
    torch.cuda.synchronize()
    t = time.perf_counter()
    out = {}
    for s in range(0, len(ims), 16):
        for k, r in zip(keys[s:s + 16], model.predict(ims[s:s + 16], imgsz=IMGSZ, conf=CONF, half=True, verbose=False, max_det=300)):
            b = r.boxes
            out[k] = [(*map(float, xy), int(c), float(p)) for xy, c, p in zip(b.xyxy.cpu().numpy(), b.cls.cpu().numpy(), b.conf.cpu().numpy())]
    torch.cuda.synchronize()
    rec.update(frames=len(keys), total_s=round(time.perf_counter() - t, 2), per_frame_s=round((time.perf_counter() - t) / len(keys), 4),
               peak_gib=round(torch.cuda.max_memory_reserved() / 2 ** 30, 2), imgsz=IMGSZ, conf=CONF, model=TAXONOMY_PT, ultralytics=ULTRA)
    return pickle.dumps({"yolo": out, "timing": rec}, protocol=4)


@app.local_entrypoint()
def main(data: str, out: str):
    sys.path.insert(0, str(HERE.parent))
    from fast_report import cards
    classes = [(c, list(syn)) for fam in cards.TAXONOMY.values() for c, syn in fam.items()]
    raw = pickle.loads(Path(data).read_bytes())
    t = time.time()
    res = yolo.remote(raw["frames"], classes)
    Path(out).write_bytes(res)
    r = pickle.loads(res)
    print(json.dumps({**r["timing"], "detections": sum(len(v) for v in r["yolo"].values()), "client_s": round(time.time() - t, 1)}, indent=1))
