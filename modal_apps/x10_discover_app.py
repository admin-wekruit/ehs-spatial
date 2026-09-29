"""X10: can we segment 'everything' without categories? Every discovery method on the same reference keyframes (and
their 5 fps neighbours), masks saved at the evaluation grid, analysis time per method per frame on an A100.

One container, 2 x A100-80GB (no MPS: one process per GPU context + vLLM outside), models resident, loaded in boot()
(cold start, reported apart). run_video(): the MP4 bytes in -> every method's masks packed (npz bytes) -> returned and
written to the x10 volume. Methods (the task's letters):
  vocab        the fast core's own vocabulary: Qwen3-VL-8B list on 5 frames (vlm.vocab) + vlm.CORE -> SAM 3   (baseline)
  amg16/32[c1] (a) SAM 2.1 automatic masks, 16 / 32 points a side, c1 = one crop layer (crop_n_layers=1)
  generic      (b) SAM 3 with catch-all words (GENERIC)
  ram          (c) RAM++ tags per frame (full frame + 2x2 tiles; ram_tags(), its own image) -> SAM 3 words
  owlw         (c) OWLv2 over RAM's 4585-tag list: the top label of each objectness box -> SAM 3 words
  owlbox       (c) OWLv2 objectness boxes (class-agnostic) -> SAM 2.1 box prompts
  geo, geosam  (d) DA3 on a 9-keyframe window (the core's model and grid), large planes by RANSAC (+ the SAM 3 floor
               plane, 1.6 m camera height: 'estimated'), residual pixels not on any plane, split at depth edges,
               connected components -> raw masks (geo) and SAM 2.1 box + point prompts (geosam)
  ridge        (e) Hessian ridge filter (dark and bright thin lines, 3 scales) -> thin components -> SAM 2.1 points
  listing      Qwen3-VL-8B grounding on the full frame + 4 upscaled tiles -> SAM 2.1 box prompts (reference input)
  aux          SAM 3 person / floor / wall / ceiling (filters and 'background' evidence, not a method)

  modal run modal_apps/x10_discover_app.py::setup                     # OWLv2 weights to the models volume, probes
  modal run modal_apps/x10_discover_app.py::ram_probe                 # RAM++ image + weights check
  modal run modal_apps/x10_discover_app.py --frames FRAMES_JSON --jpg JPG_DIR --out RUN_DIR [--videos a,b]
"""
import io
import json
import os
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import modal
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "scripts"), str(HERE.parent)]

DA3_CODE = "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4"
DA3_MODEL, DA3_REV = "depth-anything/DA3-GIANT-1.1", "72ee9f89ce4e50d704e9d55ee9c646ec8dc25a19"
SAM2_MODEL, SAM2_REV = "facebook/sam2.1-hiera-large", "665f8e2ad61cf5f53d65644ff27c8ee525124610"
OWL_MODEL = "google/owlv2-large-patch14-ensemble"
RAM_REPO, RAM_WEIGHTS = "xinyu1205/recognize-anything-plus-model", "ram_plus_swin_large_14m.pth"
LICENCES = {"SAM 2.1": "Apache-2.0", "SAM 3": "SAM License (Meta, custom; commercial use allowed with conditions)",
            "OWLv2": "Apache-2.0", "RAM++": "Apache-2.0 (code and weights)", "Qwen3-VL-8B": "Apache-2.0",
            "DA3-GIANT-1.1": "CC BY-NC 4.0 (research only)",
            "YOLO-World": "not used: GPL-3.0 code, AGPL-3.0 via ultralytics", "Grounding DINO": "not used (Apache-2.0); OWLv2 covered the box route"}
CPU, MEMORY_GIB, GPU = 16, 96, "A100-80GB:2"
PRICE = {"A100-80GB": .000694, "cpu_core": .0000131, "gib": .00000222}

GENERIC = ["object", "item", "tool", "cable", "hose", "debris", "container", "equipment", "sign"]
AUX = ["person", "floor", "wall", "ceiling"]
STOP = {  # tags that are not one physical thing: people, surfaces, scenes, materials, colours, abstract words
    "person", "man", "woman", "people", "boy", "girl", "child", "customer", "worker", "shopper", "employee", "guy", "stand", "walk",
    "floor", "wall", "ceiling", "ground", "room", "indoor", "interior", "building", "warehouse", "store", "supermarket", "shop",
    "retail", "grocery", "market", "aisle", "factory", "workshop", "office", "garage", "shopping", "sale", "hall", "corridor",
    "hallway", "space", "area", "scene", "view", "image", "photo", "picture", "light", "lighting", "shadow", "reflection",
    "white", "black", "red", "blue", "green", "yellow", "orange", "grey", "gray", "silver", "colorful", "color", "metal",
    "wood", "wooden", "plastic", "glass", "steel", "concrete", "tile", "paper", "stuff", "things", "lot", "many", "full",
    "stack", "row", "line", "pile", "variety", "assortment", "display", "product", "goods", "merchandise", "stock",
    "industry", "manufacturing", "technology", "business", "job", "work", "training", "class", "classroom", "lab", "laboratory"}
SAVE_SCORE, VOCAB_SCORE, PERSON_SCORE = .2, .3, .4
GENERIC_SAVE = .1  # generic words score low: saved from 0.1, the evaluation sweeps the floor
AMG = {"amg16": (16, 0), "amg32": (32, 0), "amg16c1": (16, 1), "amg32c1": (32, 1)}
AMG_SETTINGS = {"pred_iou_thresh": .8, "stability_score_thresh": .92, "points_per_batch": 256}  # X2 / today's settings
OWL_TOP, OWL_MIN_OBJ, OWL_WORD_MIN, OWL_WORDS = 100, .05, .2, 25  # input side: the model's (1008 for large/14)
RIDGE_SIGMAS, RIDGE_MIN, RIDGE_LEN, RIDGE_WIDTH, RIDGE_TOP = (1.5, 2.5, 4.), .05, 40, 12., 40
WINDOW, PLANE_MAX, PLANE_SHARE, PLANE_TAU_M, PLANE_TAU_REL, FAR_M = 4, 10, .02, .05, .015, 12.
GEO_MIN_PX, GEO_MAX_SHARE, DEPTH_EDGE = 30, .2, .04
GROUND_TOKENS, GROUND_LIMIT, TILE = 1500, 60, .6  # run 000: 2500 tokens -> 56/60 answers hit the cap, 330 s a video

app = modal.App("panoptes-x10-discover")
VOLUMES = {"/v/da3": modal.Volume.from_name("moge3-hf-cache"), "/v/sam3": modal.Volume.from_name("sam3-hf-cache"),
           "/v/vlm": modal.Volume.from_name("panoptes-vlm-cache"),
           "/v/models": modal.Volume.from_name("panoptes-fb-models", create_if_missing=True),
           "/v/x10": modal.Volume.from_name("panoptes-x10", create_if_missing=True)}
image = (modal.Image.from_registry("nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04", add_python="3.11")  # fx/x2-discover's chain
         .apt_install("git", "libgl1", "libglib2.0-0", "libgomp1")
         .pip_install("torch==2.14.0", "torchvision", "xformers", "transformers==5.17.0", "accelerate", "addict", "pillow", "scipy",
                      "open3d==0.19.0", "shapely", "pydantic", "opencv-python-headless", "sentencepiece",
                      f"git+https://github.com/ByteDance-Seed/Depth-Anything-3.git@{DA3_CODE}")
         .run_commands("python -m venv /opt/vllm && PIP_EXTRA_INDEX_URL= /opt/vllm/bin/pip install -q vllm==0.11.0 transformers==4.57.1 pillow")
         .run_commands("SAM2_BUILD_CUDA=0 pip install -q git+https://github.com/facebookresearch/sam2.git@c2ec8e14a185632b0a5d8b161928ceb50197eddc")
         .env({"HF_HUB_OFFLINE": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
         .add_local_python_source("detect_shot_cuts", "m3_exp_geometry", "sam3_app", "video_events", "fast_report", "ehs_spatial"))
ram_image = (modal.Image.debian_slim(python_version="3.10").apt_install("git", "libgl1", "libglib2.0-0")
             .pip_install("torch==2.1.2", "torchvision==0.16.2", "numpy<2", "timm==0.4.12", "transformers==4.36.2", "fairscale==0.4.4",
                          "scipy", "pillow", "huggingface_hub==0.20.3", "opencv-python-headless<4.11")
             .pip_install("ftfy", "regex", "tqdm")
             .run_commands("pip install -q --no-deps git+https://github.com/openai/CLIP.git",
                           "pip install -q --no-deps git+https://github.com/xinyu1205/recognize-anything.git",
                           "pip show recognize-anything || true"))


# ---------- pure helpers (self_check) ----------

def tiles(w, h, side=TILE):
    """2 x 2 overlapping tiles, each side x the frame: [(x0, y0, x1, y1)] in pixels."""
    tw, th = int(round(w * side)), int(round(h * side))
    return [(x, y, x + tw, y + th) for y in (0, h - th) for x in (0, w - tw)]


def clean_words(words, cap=40):
    """Lower case, stripped, not a STOP word (whole phrase or its head noun), deduplicated, in order."""
    out = []
    for w in words:
        w = " ".join(str(w).lower().strip(" .,;:\"'").split())
        if w and w not in STOP and w.split()[-1] not in STOP and w not in out:
            out.append(w)
    return out[:cap]


def box_iou_np(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ix = np.maximum(0, np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]))
    iy = np.maximum(0, np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]))
    inter = ix * iy
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])  # noqa: E731
    return inter / np.maximum(area(a)[:, None] + area(b)[None] - inter, 1e-9)


def merge_boxes(items, iou=.7):
    """[(box xyxy px, label, source)] -> greedy: a box with IoU >= iou to a kept one is that one (tiles first: tighter)."""
    order = sorted(range(len(items)), key=lambda i: items[i][2] == "full")
    kept = []
    for i in order:
        if not kept or box_iou_np([items[i][0]], [items[k][0] for k in kept]).max() < iou:
            kept.append(i)
    return [items[i] for i in sorted(kept)]


def pack(masks):
    """(n, h, w) bool -> (n, h, ceil(w/8)) uint8."""
    return np.packbits(np.asarray(masks, bool), axis=-1)


def unpack(bits, w):
    return np.unpackbits(bits, axis=-1, count=w).astype(bool)


def thin_components(strength, thr, min_len, max_width, top):
    """Ridge strength (H, W) numpy -> [(pixel rows, pixel cols, length px, mean strength)] of thin components: bbox
    diagonal >= min_len, area / diagonal <= max_width, the `top` longest x strongest."""
    import cv2
    n, lab, stats, _ = cv2.connectedComponentsWithStats((strength >= thr).astype(np.uint8), connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        diag = float(np.hypot(w, h))
        if diag >= min_len and a / diag <= max_width:
            ys, xs = np.nonzero(lab[y:y + h, x:x + w] == i)
            out.append((ys + y, xs + x, diag, float(strength[ys + y, xs + x].mean())))
    return sorted(out, key=lambda c: -c[2] * c[3])[:top]


def along(ys, xs, k=5):
    """k pixels of a component spread along its principal axis (quantiles of the projection) -> (k, 2) x, y."""
    p = np.stack([xs, ys], 1).astype(float)
    c = p - p.mean(0)
    axis = np.linalg.svd(c, full_matrices=False)[2][0] if len(p) > 1 else np.array([1., 0.])
    order = np.argsort(c @ axis)
    return p[order[np.linspace(0, len(p) - 1, k).round().astype(int)]]


def straightness(ys, xs):
    """RMS distance of a component's pixels from its principal line, over its length: ~0 for a straight edge."""
    p = np.stack([xs, ys], 1).astype(float)
    c = p - p.mean(0)
    _, sv, vt = np.linalg.svd(c, full_matrices=False)
    proj = c @ vt[0]
    return float(np.sqrt(np.mean((c @ vt[1]) ** 2)) / max(proj.max() - proj.min(), 1.))


def thinnest(masks, scores, min_score=.5):
    """(n, 3, H, W) multimask output -> per prompt the smallest mask with score >= min_score (else the best)."""
    area = masks.reshape(*masks.shape[:2], -1).sum(-1)
    ok = scores >= min_score
    area = np.where(ok, area, np.inf)
    pick = np.where(ok.any(1), area.argmin(1), scores.argmax(1))
    return pick


def self_check_cpu():
    assert tiles(1280, 720) == [(0, 0, 768, 432), (512, 0, 1280, 432), (0, 288, 768, 720), (512, 288, 1280, 720)]
    assert clean_words(["Floor", "power cord", "store", "cord", "white", "pallet jack", "Power Cord"]) == ["power cord", "cord", "pallet jack"]
    m = merge_boxes([([0, 0, 10, 10], "a", "full"), ([0, 0, 10, 11], "a", "tile0"), ([50, 50, 60, 60], "b", "full")])
    assert [x[2] for x in m] == ["tile0", "full"], m
    x = np.random.default_rng(0).random((3, 7, 13)) > .5
    assert (unpack(pack(x), 13) == x).all()
    s = np.zeros((100, 200), np.float32)
    s[50, 20:180] = 1   # a thin line: kept
    s[5:25, 5:25] = 1   # a blob: too wide
    s[90, 10:20] = 1    # too short
    comps = thin_components(s, .5, 40, 12, 10)
    assert len(comps) == 1 and abs(comps[0][2] - np.hypot(160, 1)) < 1, [(c[2]) for c in comps]
    pts = along(comps[0][0], comps[0][1], 5)
    assert pts[0][0] == 20 and pts[-1][0] == 179 and (pts[:, 1] == 50).all(), pts
    yy = np.arange(100.)
    assert straightness(yy, 2 * yy) < 1e-9 and straightness(yy, 20 * np.sin(yy / 15)) > .02
    ms = np.zeros((1, 3, 4, 4), bool)
    ms[0, 0, :2, :2] = True
    ms[0, 1] = True
    ms[0, 2, 0, 0] = True
    assert thinnest(ms, np.array([[.9, .95, .3]]))[0] == 0 and thinnest(ms, np.array([[.1, .2, .3]]))[0] == 2
    print("x10 cpu self-check ok")


# ---------- GPU helpers ----------

def ridge_map(gray, sigmas=RIDGE_SIGMAS):
    """(H, W) float [0, 1] tensor -> scale-normalised ridge strength: dark or bright lines a few px wide (Hessian: one
    large eigenvalue across the line, a small one along it)."""
    import torch
    import torch.nn.functional as F
    out = torch.zeros_like(gray)
    x = gray[None, None]
    k2 = torch.tensor([1., -2., 1.], device=gray.device)
    for s in sigmas:
        r = int(np.ceil(3 * s))
        t = torch.arange(-r, r + 1, device=gray.device, dtype=torch.float32)
        g = torch.exp(-t ** 2 / (2 * s * s))
        g = g / g.sum()
        b = F.conv2d(F.pad(x, (r, r, 0, 0), mode="replicate"), g.view(1, 1, 1, -1))
        b = F.conv2d(F.pad(b, (0, 0, r, r), mode="replicate"), g.view(1, 1, -1, 1))
        dxx = F.conv2d(F.pad(b, (1, 1, 0, 0), mode="replicate"), k2.view(1, 1, 1, 3))[0, 0]
        dyy = F.conv2d(F.pad(b, (0, 0, 1, 1), mode="replicate"), k2.view(1, 1, 3, 1))[0, 0]
        dxy = F.conv2d(F.pad(b, (1, 1, 1, 1), mode="replicate"),
                       torch.tensor([[1., 0, -1], [0, 0, 0], [-1, 0, 1]], device=gray.device).view(1, 1, 3, 3) / 4)[0, 0]
        gx = F.conv2d(F.pad(b, (1, 1, 0, 0), mode="replicate"), torch.tensor([-.5, 0, .5], device=gray.device).view(1, 1, 1, 3))[0, 0]
        gy = F.conv2d(F.pad(b, (0, 0, 1, 1), mode="replicate"), torch.tensor([-.5, 0, .5], device=gray.device).view(1, 1, 3, 1))[0, 0]
        root = torch.sqrt((dxx - dyy) ** 2 + 4 * dxy ** 2)
        hi, lo = (dxx + dyy + root) / 2, (dxx + dyy - root) / 2
        dark = torch.where((hi > 0) & (lo.abs() < .5 * hi), hi, torch.zeros_like(hi))
        bright = torch.where((lo < 0) & (hi.abs() < .5 * -lo), -lo, torch.zeros_like(lo))
        # minus the scaled gradient: beside a step edge the curvature is as large as the gradient (both c x phi(1));
        # at a line's centre the gradient is ~0
        out = torch.maximum(out, s * s * torch.maximum(dark, bright) - s * torch.sqrt(gx ** 2 + gy ** 2))
    return out


def ransac_planes(pts, tau, max_planes=PLANE_MAX, min_share=PLANE_SHARE, iters=512, sample=20000, seed=0):
    """(N, 3) tensor -> [(unit normal, offset d)] with n.x + d = 0, the largest first; each takes >= min_share of all
    points within tau; least-squares refit on its inliers."""
    import torch
    gen = torch.Generator(device=pts.device).manual_seed(seed)
    n_all, left, planes = len(pts), pts, []
    for _ in range(max_planes):
        if len(left) < max(3, min_share * n_all):
            break
        idx = torch.randint(len(left), (iters, 3), device=pts.device, generator=gen)
        a, b, c = left[idx[:, 0]], left[idx[:, 1]], left[idx[:, 2]]
        nrm = torch.linalg.cross(b - a, c - a)
        ln = nrm.norm(dim=1, keepdim=True)
        nrm = nrm / ln.clamp(min=1e-12)
        d = -(nrm * a).sum(1)
        sub = left[torch.randperm(len(left), device=pts.device, generator=gen)[:sample]]
        cnt = ((sub @ nrm.T + d).abs() < tau).sum(0) * (ln[:, 0] > 1e-12)
        best = int(cnt.argmax())
        inl = (left @ nrm[best] + d[best]).abs() < tau
        if inl.sum() < min_share * n_all:
            break
        p = left[inl]
        cen = p.mean(0)
        u = torch.linalg.eigh((p - cen).T @ (p - cen))[1][:, 0]
        dd = -(u * cen).sum()
        planes.append((u, dd))
        left = left[(left @ u + dd).abs() >= tau]
    return planes


def self_check_gpu(dev="cuda:0"):
    """Ridges find a drawn dark line and not a flat step edge; RANSAC recovers two planes and not a box on them."""
    import cv2
    import torch
    img = np.full((200, 300), .7, np.float32)
    img[:, 150:] = .4                                        # step edge: not a ridge
    cv2.line(img, (20, 30), (280, 170), .2, 3)               # dark line, 3 px
    s = ridge_map(torch.tensor(img, device=dev)).cpu().numpy()
    comps = thin_components(s, RIDGE_MIN, 40, 12, 10)
    assert comps and comps[0][2] > 200, [c[2] for c in comps]
    assert s[0:60, 140:160].max() < RIDGE_MIN, ("step edge seen as a ridge", float(s[0:60, 140:160].max()))
    rng = np.random.default_rng(0)
    floor = np.c_[rng.uniform(-5, 5, (6000, 2)), np.zeros(6000)]
    wall = np.c_[rng.uniform(-5, 5, 4000), np.full(4000, 5.), rng.uniform(0, 3, 4000)]
    box = rng.uniform([0, 0, .1], [.5, .5, .6], (300, 3))
    pts = torch.tensor(np.r_[floor, wall, box], dtype=torch.float32, device=dev)
    planes = ransac_planes(pts, .02)
    normals = [p[0].abs().cpu().numpy().round(2) for p in planes[:2]]
    assert any((n == [0, 0, 1]).all() for n in normals) and any((n == [0, 1, 0]).all() for n in normals), normals
    on = torch.stack([(pts @ u + d).abs() < .02 for u, d in planes]).any(0).cpu().numpy()
    assert on[:10000].mean() > .98 and on[10000:].mean() < .3, (on[:10000].mean(), on[10000:].mean())
    print("x10 gpu self-check ok")
    return True


# ---------- setup / probes ----------

@app.function(image=image, gpu="A100-80GB", timeout=1800, retries=0, volumes=VOLUMES)
def setup():
    """OWLv2 weights to the models volume (public, no token); SAM 2.1 weights present; GPU self-check; OWLv2 API probe."""
    os.environ["HF_HUB_OFFLINE"] = "0"
    import torch
    from huggingface_hub import hf_hub_download, snapshot_download
    t = time.time()
    snapshot_download(OWL_MODEL, cache_dir="/v/models/hf")
    VOLUMES["/v/models"].commit()
    from sam2.build_sam import HF_MODEL_ID_TO_FILENAMES
    hf_hub_download(SAM2_MODEL, HF_MODEL_ID_TO_FILENAMES[SAM2_MODEL][1], revision=SAM2_REV, cache_dir="/v/da3/huggingface/hub")
    self_check_cpu()
    self_check_gpu()
    owl = Owl(torch.device("cuda:0"))
    q = owl.encode(["cat", "power cord", "forklift"])
    rgb = np.random.default_rng(0).integers(0, 255, (720, 1280, 3), np.uint8)
    boxes, obj, top_word, top_score = owl.detect(torch.tensor(rgb, device="cuda:0"), q)
    out = {"owl_boxes": list(boxes.shape), "obj_max": float(obj.max()), "words": top_word[:3].tolist(), "s": round(time.time() - t, 1)}
    print(json.dumps(out, default=str))
    return out


@app.function(image=ram_image, gpu="A100-80GB", timeout=1800, retries=0, volumes={"/v/models": VOLUMES["/v/models"]})
def ram_tags(frames: dict):
    """RAM++ (Swin-L, 384 px) on each frame and its 2 x 2 tiles -> {key: {'full': [tags], 'tiles': [tags]}}, per-frame
    time (the model resident, batch of 5 images per frame), the 4585-tag list, the installed version."""
    import cv2
    import torch
    from huggingface_hub import hf_hub_download
    from PIL import Image
    from ram import get_transform
    from ram.models import ram_plus
    t0 = time.perf_counter()
    path = hf_hub_download(RAM_REPO, RAM_WEIGHTS, cache_dir="/v/models/hf")
    VOLUMES["/v/models"].commit()
    model = ram_plus(pretrained=path, image_size=384, vit="swin_l").eval().to("cuda")
    tf = get_transform(image_size=384)
    load_s = time.perf_counter() - t0
    tag_list = [str(x) for x in model.tag_list]

    def batch(jpg):
        bgr = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        crops = [rgb] + [rgb[y0:y1, x0:x1] for x0, y0, x1, y1 in tiles(w, h)]
        return torch.stack([tf(Image.fromarray(c)) for c in crops])
    with torch.inference_mode():
        model.generate_tag(batch(next(iter(frames.values()))).cuda())  # warm-up (cold start)
        torch.cuda.synchronize()
    out, times = {}, []
    for key, jpg in frames.items():
        t = time.perf_counter()
        x = batch(jpg).cuda()
        with torch.inference_mode():
            tags = model.generate_tag(x)[0]
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t)
        split = [[s.strip() for s in tg.split("|") if s.strip()] for tg in tags]
        out[key] = {"full": split[0], "tiles": sorted({s for sp in split[1:] for s in sp})}
    ver = subprocess.run([sys.executable, "-m", "pip", "show", "ram"], capture_output=True, text=True).stdout
    return {"tags": out, "tag_list": tag_list, "load_s": round(load_s, 1), "per_frame_s": [round(t, 4) for t in times],
            "note": "per frame = decode + 5 crops (full + 2x2 tiles) + one batched forward", "pip_show": ver[-600:],
            "gpu": subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.strip()}


@app.local_entrypoint()
def ram_probe():
    rng = np.random.default_rng(0)
    import cv2
    jpg = cv2.imencode(".jpg", rng.integers(0, 255, (720, 1280, 3), np.uint8))[1].tobytes()
    r = ram_tags.remote({"noise": jpg})
    print(json.dumps({k: v for k, v in r.items() if k != "tag_list"} | {"tags_in_list": len(r["tag_list"])}, default=str)[:3000])


# ---------- OWLv2 ----------

class Owl:
    """OWLv2 large (ensemble) on one GPU: image features once, objectness per box, class logits against any number of
    encoded queries (RAM's 4585 tags encoded once)."""

    def __init__(self, dev):
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor
        self.dev = dev
        self.model = Owlv2ForObjectDetection.from_pretrained(OWL_MODEL, cache_dir="/v/models/hf", torch_dtype=torch.float16).to(dev).eval()
        self.proc = Owlv2Processor.from_pretrained(OWL_MODEL, cache_dir="/v/models/hf")
        ip = self.proc.image_processor
        self.mean = torch.tensor(ip.image_mean, device=dev).view(1, 3, 1, 1)
        self.std = torch.tensor(ip.image_std, device=dev).view(1, 3, 1, 1)
        self.side = self.model.config.vision_config.image_size

    def encode(self, words, chunk=512):
        import torch
        out = []
        with torch.inference_mode():
            for i in range(0, len(words), chunk):
                tok = self.proc.tokenizer(words[i:i + chunk], padding="max_length", max_length=16, truncation=True, return_tensors="pt").to(self.dev)
                e = self.model.owlv2.get_text_features(input_ids=tok["input_ids"], attention_mask=tok["attention_mask"])
                out.append(getattr(e, "pooler_output", e))
        return torch.cat(out)

    def detect(self, rgb, queries, top=OWL_TOP):
        """rgb (H, W, 3) uint8 tensor on this GPU -> boxes xyxy px (top by objectness), objectness, top word index and
        its score per box."""
        import torch
        import torch.nn.functional as F
        h, w = rgb.shape[:2]
        side = max(h, w)
        x = torch.full((1, 3, side, side), .5, device=self.dev)
        x[0, :, :h, :w] = rgb.permute(2, 0, 1).float() / 255
        x = F.interpolate(x, size=(self.side, self.side), mode="bilinear", antialias=True, align_corners=False)
        x = ((x - self.mean) / self.std).half()
        with torch.inference_mode():
            m = self.model
            emb, _ = m.image_embedder(pixel_values=x)[:2]
            b, gh, gw, d = emb.shape
            feats = emb.reshape(b, gh * gw, d)
            boxes = m.box_predictor(feats, emb)[0]  # cx, cy, w, h in [0, 1] of the padded square
            obj = m.objectness_predictor(feats)[0].float().sigmoid()
            logits = m.class_predictor(feats, queries[None].to(feats.dtype), None)[0][0].float().sigmoid()
            order = obj.argsort(descending=True)[:top]
            bx = boxes[order].float()
            xyxy = torch.stack([bx[:, 0] - bx[:, 2] / 2, bx[:, 1] - bx[:, 3] / 2, bx[:, 0] + bx[:, 2] / 2, bx[:, 1] + bx[:, 3] / 2], 1) * side
            xyxy[:, 0::2] = xyxy[:, 0::2].clamp(0, w)
            xyxy[:, 1::2] = xyxy[:, 1::2].clamp(0, h)
            s, word = logits[order].max(1)
        return xyxy, obj[order], word, s


# ---------- the container ----------

@app.cls(image=image, gpu=GPU, cpu=CPU, memory=MEMORY_GIB * 1024, volumes=VOLUMES, timeout=5400, retries=0, max_containers=1,
         scaledown_window=60)
class X10:
    @modal.enter()
    def boot(self):
        import copy
        import sam3_app  # SAM 3 revision pin (not importable in the RAM++ image: imported here)
        from fast_report import core, segment, vlm
        t0 = time.perf_counter()
        b = self.boot_record = {"entered_unix": time.time()}
        lap = lambda k: b.__setitem__(k, round(time.perf_counter() - t0, 2))  # noqa: E731
        self.vllm = vlm.start(1)
        lap("vllm_spawned_s")
        import torch
        from depth_anything_3.api import DepthAnything3
        from huggingface_hub import hf_hub_download
        from sam2.build_sam import HF_MODEL_ID_TO_FILENAMES, build_sam2
        from transformers import Sam3Model, Sam3Processor
        self.devs = [torch.device("cuda:0"), torch.device("cuda:1")]
        lap("imports_s")
        self.da3 = core.Da3(DepthAnything3.from_pretrained(DA3_MODEL, revision=DA3_REV, cache_dir="/v/da3/huggingface/hub").eval().to(self.devs[0]), self.devs[0])
        lap("da3_s")
        proc = Sam3Processor.from_pretrained(sam3_app.MODEL_ID, revision=sam3_app.REVISION, cache_dir="/v/sam3/huggingface/hub")
        sam = Sam3Model.from_pretrained(sam3_app.MODEL_ID, revision=sam3_app.REVISION, cache_dir="/v/sam3/huggingface/hub", torch_dtype=torch.bfloat16).eval()
        self.sam3 = {self.devs[0]: segment.Sam3(copy.deepcopy(sam).to(self.devs[0]), proc, self.devs[0]),
                     self.devs[1]: segment.Sam3(sam.to(self.devs[1]), proc, self.devs[1])}
        lap("sam3_s")
        config, ckpt = HF_MODEL_ID_TO_FILENAMES[SAM2_MODEL]
        path = hf_hub_download(SAM2_MODEL, ckpt, revision=SAM2_REV, cache_dir="/v/da3/huggingface/hub")
        self.sam2 = {}
        for d in self.devs:
            with torch.cuda.device(d):
                self.sam2[d] = build_sam2(config, path, device=str(d))
        lap("sam2_s")
        self.owl = {d: Owl(d) for d in self.devs}
        lap("owl_s")
        vlm.wait(self.vllm)
        lap("vllm_ready_s")
        self_check_cpu()
        b["gpu_self_check"] = self_check_gpu("cuda:0")
        self.warm()
        lap("warm_s")
        self.tag_queries = None
        b.update(ready_s=round(time.perf_counter() - t0, 2), gpus=subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.splitlines(),
                 torch=str(torch.__version__), resident_gb=[round((torch.cuda.mem_get_info(d)[1] - torch.cuda.mem_get_info(d)[0]) / 1e9, 2) for d in self.devs])

    def warm(self):
        """Every model once per GPU at the run's shapes (kernels, allocator): cold start, never analysis time."""
        import torch
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        from fast_report import vlm
        rgb = np.random.default_rng(0).integers(0, 255, (720, 1280, 3), np.uint8)
        for d in self.devs:
            with torch.cuda.device(d), torch.inference_mode():
                x = torch.tensor(rgb, device=d)
                v = self.sam3[d].vision(x[None].flip(-1))
                self.sam3[d].detect(v, 1, tuple(GENERIC), .5)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    SAM2AutomaticMaskGenerator(self.sam2[d], points_per_side=16, **AMG_SETTINGS).generate(rgb)
                    p = SAM2ImagePredictor(self.sam2[d])
                    p.set_image(rgb)
                    p._predict(torch.tensor([[[100., 100.]]], device=d), torch.ones((1, 1), device=d, dtype=torch.int), multimask_output=False)
                self.owl[d].detect(x, self.owl[d].encode(["cat"]))
                ridge_map(torch.rand((720, 1280), device=d))
        with torch.inference_mode():
            self.da3.shot(torch.randint(0, 255, (9, 720, 1280, 3), dtype=torch.uint8, device=self.devs[0]))
        noise = __import__("cv2").imencode(".jpg", rgb)[1].tobytes()
        vlm.chat([vlm.image_block(noise), {"type": "text", "text": "List objects."}], max_tokens=8)
        for d in self.devs:
            with torch.cuda.device(d):
                torch.cuda.synchronize(d)
                torch.cuda.empty_cache()

    @modal.exit()
    def stop(self):
        if getattr(self, "vllm", None) is not None:
            self.vllm.terminate()

    @modal.method()
    def boot_info(self):
        return self.boot_record

    @modal.method()
    def run_video(self, name: str, mp4: bytes, plan: dict, ram: dict, tag_list: list, run_id: str):
        """plan: reference, neighbours ({ref: [before, after]}), keys_5fps. -> (npz bytes of every method's masks at the
        evaluation grid, record with per-stage times and per-GPU peaks)."""
        import torch
        from fast_report.stubs import Clock, Vram
        clock = Clock()
        vram = Vram(self.devs, clock)
        vram.start()
        for d in self.devs:
            torch.cuda.reset_peak_memory_stats(d)
        try:
            out, rec = self._run(name, mp4, plan, ram, tag_list, clock)
            rec["error"] = None
        except Exception:  # noqa: BLE001  reported, never retried
            out, rec = {}, {"error": traceback.format_exc()[-4000:]}
        vram.stop()
        rec["timing"] = clock.report(vram)
        rec["torch_reserved_peak_gb"] = [round(torch.cuda.max_memory_reserved(d) / 1e9, 2) for d in self.devs]
        buf = io.BytesIO()
        np.savez_compressed(buf, **out)
        path = Path("/v/x10") / run_id / f"{name}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(buf.getvalue())
        (path.parent / f"{name}.json").write_text(json.dumps(rec, default=str))
        VOLUMES["/v/x10"].commit()
        for d in self.devs:
            with torch.cuda.device(d):
                torch.cuda.empty_cache()
        return buf.getvalue(), json.loads(json.dumps(rec, default=str))

    # ---- one video ----
    def _run(self, name, mp4, plan, ram, tag_list, clock):
        import cv2
        import torch
        from fast_report import core, segment, vlm
        rec = {"video": name, "stages": {}}
        with clock.stage("decode"):
            tmp = Path(f"/tmp/{name}.mp4")
            tmp.write_bytes(mp4)
            cap = cv2.VideoCapture(str(tmp))
            frames = []
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                frames.append(bgr)
        H, W = frames[0].shape[:2]
        hw = (H // 2, W // 2) if W >= 1280 else (H, W)
        segment.DA3_HW = hw  # SAM 3 masks straight to the evaluation grid (detect() reads the module global)
        ref = plan["reference"]
        nb = sorted({x for p in plan["neighbours"].values() for x in p if x is not None} - set(ref))
        allf = ref + nb
        rec.update(size_wh=[W, H], eval_hw=list(hw), reference=ref, neighbours=nb)
        out = {}
        put = lambda f, m, masks, score, label: out.update({f"{f}/{m}/bits": pack(masks), f"{f}/{m}/score": np.asarray(score, np.float32),  # noqa: E731
                                                            f"{f}/{m}/label": np.array(label if label is not None else [""] * len(score), dtype="U64")})
        times = {}  # method -> [per-frame seconds]

        def timed(method, f, t):
            times.setdefault(method, []).append({"frame": f, "s": round(time.perf_counter() - t, 4)})

        def both(fs, fn):
            """fn(dev, frame) for every frame, GPU 0 takes the even positions, GPU 1 the odd; one thread per GPU."""
            def worker(d, part):
                with torch.cuda.device(d):
                    return [fn(d, f) for f in part]
            with ThreadPoolExecutor(2) as pool:
                jobs = [pool.submit(worker, self.devs[i], fs[i::2]) for i in range(2)]
                return [x for j in jobs for x in j.result()]

        # vocabulary (the core's: 5 frames of the whole clip, 4:3 raster PNGs) -> words
        t = time.perf_counter()
        with clock.stage("vocab.qwen"):
            n = len(frames)
            pngs = [cv2.imencode(".png", core.raster_rgb(frames[int((i + .5) * n / vlm.VOCAB_FRAMES)]))[1].tobytes() for i in range(vlm.VOCAB_FRAMES)]
            words, vrec = vlm.vocab(pngs)
        vocab_words = list(dict.fromkeys(vlm.CORE + words))
        rec["vocab"] = {"words": vocab_words, "qwen_s": round(time.perf_counter() - t, 3), "record": {k: v for k, v in vrec.items() if k != "text"}}

        # SAM 3 vision features (kept per frame on its GPU)
        feats = {}

        def vision(d, f):
            t = time.perf_counter()
            x = torch.from_numpy(frames[f]).to(d)
            with torch.inference_mode():
                feats[f] = (d, self.sam3[d].vision(x[None]))
            torch.cuda.synchronize(d)
            timed("sam3_vision", f, t)
        with clock.stage("sam3.vision", n=len(allf)):
            both(allf, vision)

        # OWLv2 over RAM's tag list: boxes (owlbox) and words (owlw)
        if self.tag_queries is None or self.tag_queries[0] != len(tag_list):
            t = time.perf_counter()
            self.tag_queries = (len(tag_list), {d: self.owl[d].encode(tag_list) for d in self.devs})
            rec["owl_tag_encode_s_once"] = round(time.perf_counter() - t, 2)  # once per container: cold start
        owl = {}

        def owl_run(d, f):
            t = time.perf_counter()
            x = torch.from_numpy(np.ascontiguousarray(frames[f][..., ::-1])).to(d)
            boxes, obj, word, s = self.owl[d].detect(x, self.tag_queries[1][d])
            keep = obj >= OWL_MIN_OBJ
            owl[f] = (boxes[keep].cpu().numpy(), obj[keep].cpu().numpy(), [tag_list[i] for i in word[keep].tolist()], s[keep].cpu().numpy())
            torch.cuda.synchronize(d)
            timed("owl", f, t)
        with clock.stage("owl", n=len(allf)):
            both(allf, owl_run)
        owl_words = {f: clean_words([w for w, s in sorted(zip(owl[f][2], owl[f][3]), key=lambda x: -x[1]) if s >= OWL_WORD_MIN], OWL_WORDS) for f in allf}
        ram_words = {f: clean_words(ram[str(f)]["full"] + ram[str(f)]["tiles"]) for f in allf}
        rec["words"] = {"owlw": {str(f): owl_words[f] for f in allf}, "ram": {str(f): ram_words[f] for f in allf},
                        "ram_raw": {str(f): ram[str(f)] for f in allf}}

        # SAM 3 detections: aux, vocab, generic, ram, owlw (per-frame words)
        sets = [("aux", lambda f: AUX), ("vocab", lambda f: vocab_words), ("generic", lambda f: GENERIC),
                ("ram", lambda f: ram_words[f]), ("owlw", lambda f: owl_words[f])]

        def detect(d, f, method, wl):
            t = time.perf_counter()
            dd, v = feats[f]
            wl = tuple(wl)
            if not wl:
                put(f, method, np.zeros((0, *hw), bool), [], [])
                timed(method, f, t)
                return
            floor = GENERIC_SAVE if method == "generic" else SAVE_SCORE
            with torch.inference_mode():
                r = self.sam3[dd].detect(v, 1, wl, floor, top=PERSON_TOP_AUX if method == "aux" else None)
            m, s, w = r["mask"].cpu().numpy(), r["score"].float().cpu().numpy(), [wl[i] for i in r["word"].tolist()]
            put(f, method, m, s, w)
            timed(method, f, t)
        for method, wf in sets:
            with clock.stage(f"sam3.{method}", n=len(allf)):
                both(allf, lambda d, f: detect(d, f, method, wf(f)))

        # (a) SAM 2.1 AMG variants
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        import torch.nn.functional as F

        def down(masks_full, d):
            """full-res bool masks (numpy or tensor) -> eval grid bool numpy (area > 0.3)."""
            if len(masks_full) == 0:
                return np.zeros((0, *hw), bool)
            mk = torch.as_tensor(masks_full, device=d)
            return (F.interpolate(mk[:, None].half(), size=hw, mode="area")[:, 0] > .3).cpu().numpy()

        def amg(d, f, method):
            side, crops = AMG[method]
            t = time.perf_counter()
            gen = SAM2AutomaticMaskGenerator(self.sam2[d], points_per_side=side, crop_n_layers=crops, **AMG_SETTINGS)
            rgb = np.ascontiguousarray(frames[f][..., ::-1])
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                found = gen.generate(rgb)
            m = down(np.stack([x["segmentation"] for x in found]) if found else [], d)
            torch.cuda.synchronize(d)
            timed(method, f, t)
            put(f, method, m, [x["predicted_iou"] * x["stability_score"] for x in found], None)
        for method in AMG:
            fs = allf if method in ("amg32", "amg32c1") else ref
            with clock.stage(f"sam2.{method}", n=len(fs)):
                both(fs, lambda d, f: amg(d, f, method))

        # SAM 2.1 prompts: owlbox (c), ridge (e) on every frame; geo (d) on reference frames (DA3 first)
        preds = {d: SAM2ImagePredictor(self.sam2[d]) for d in self.devs}

        def prompt(d, pts=None, labels=None, boxes=None, multi=False):
            """batched prompts on the predictor's current image -> (masks eval-grid numpy, scores numpy)."""
            p = preds[d]
            hw0 = p._orig_hw[-1]
            pc = p._transforms.transform_coords(torch.as_tensor(pts, dtype=torch.float32, device=d), normalize=True, orig_hw=hw0) if pts is not None else None
            pl = torch.as_tensor(labels, dtype=torch.int, device=d) if labels is not None else None
            bx = p._transforms.transform_boxes(torch.as_tensor(boxes, dtype=torch.float32, device=d), normalize=True, orig_hw=hw0) if boxes is not None else None
            n = len(pts) if pts is not None else len(boxes)
            ms, ss = [], []
            for i in range(0, n, 64):
                sl = slice(i, i + 64)
                m, s, _ = p._predict(pc[sl] if pc is not None else None, pl[sl] if pl is not None else None, bx[sl] if bx is not None else None,
                                     multimask_output=multi)
                if multi:
                    best = s.argmax(1)
                    m, s = m[torch.arange(len(m)), best], s[torch.arange(len(s)), best]
                else:
                    m, s = m[:, 0], s[:, 0]
                ms.append(down(m > 0 if m.dtype != torch.bool else m, d))
                ss.append(s.float().cpu().numpy())
            return (np.concatenate(ms), np.concatenate(ss)) if ms else (np.zeros((0, *hw), bool), np.zeros(0, np.float32))

        geo_comps = {}
        with clock.stage("da3.geo", gpu=self.devs[0], n=len(ref)):
            for f in ref:
                t = time.perf_counter()
                geo_comps[f], grec = self.geometry(frames, plan["keys_5fps"], f, out, hw)
                rec.setdefault("geo", {})[str(f)] = grec
                timed("geo_da3_and_planes", f, t)

        def prompts(d, f):
            t = time.perf_counter()
            rgb = np.ascontiguousarray(frames[f][..., ::-1])
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                preds[d].set_image(rgb)
                torch.cuda.synchronize(d)
                timed("sam2_embed", f, t)
                t = time.perf_counter()
                boxes = owl[f][0]
                m, s = prompt(d, boxes=boxes) if len(boxes) else (np.zeros((0, *hw), bool), np.zeros(0))
                put(f, "owlbox", m, s * 0 + owl[f][1] if len(s) else s, owl[f][2])
                torch.cuda.synchronize(d)
                timed("owlbox_decode", f, t)
                t = time.perf_counter()
                g = torch.from_numpy(cv2.cvtColor(frames[f], cv2.COLOR_BGR2GRAY)).to(d).float() / 255
                strength = ridge_map(g).cpu().numpy()
                sc = W / 1280
                comps = thin_components(strength, RIDGE_MIN, RIDGE_LEN * sc, RIDGE_WIDTH * sc, RIDGE_TOP)
                timed("ridge_filter", f, t)
                t = time.perf_counter()
                if comps:
                    pts = np.stack([along(ys, xs, 5) for ys, xs, _, _ in comps])
                    m, s = prompt(d, pts=pts, labels=np.ones(pts.shape[:2]))
                else:
                    m, s = np.zeros((0, *hw), bool), np.zeros(0)
                put(f, "ridge", m, s, [f"len {c[2]:.0f} str {c[3]:.3f}" for c in comps])
                torch.cuda.synchronize(d)
                timed("ridge_decode", f, t)
                if f in geo_comps:
                    t = time.perf_counter()
                    cs = geo_comps[f]
                    if cs:
                        m, s = prompt(d, pts=np.array([[c["point"]] for c in cs], float), labels=np.ones((len(cs), 1)), boxes=np.array([c["box"] for c in cs], float))
                    else:
                        m, s = np.zeros((0, *hw), bool), np.zeros(0)
                    put(f, "geosam", m, s, None)
                    torch.cuda.synchronize(d)
                    timed("geosam_decode", f, t)
        with clock.stage("sam2.prompts", n=len(allf)):
            both(allf, prompts)

        # listing (reference input): Qwen3-VL-8B grounding on the full frame and 4 upscaled tiles
        from fast_report import discover as dsc
        with clock.stage("listing.qwen", n=len(ref)):
            t = time.perf_counter()
            jobs = []
            for f in ref:
                jobs.append((f, "full", (0, 0, W, H), frames[f]))
                for k, (x0, y0, x1, y1) in enumerate(tiles(W, H)):
                    jobs.append((f, f"tile{k}", (x0, y0, x1, y1), cv2.resize(frames[f][y0:y1, x0:x1], (W, H), interpolation=cv2.INTER_CUBIC)))

            def ask(job):
                f, src, (x0, y0, x1, y1), img = job
                jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()
                t1 = time.perf_counter()
                text, usage = vlm.chat([vlm.image_block(jpg), {"type": "text", "text": dsc.GROUND_PROMPT}], max_tokens=GROUND_TOKENS)
                boxes = [([x0 + b[0] / 1000 * (x1 - x0), y0 + b[1] / 1000 * (y1 - y0), x0 + b[2] / 1000 * (x1 - x0), y0 + b[3] / 1000 * (y1 - y0)], lab, src)
                         for b, lab in dsc.parse_boxes(text, limit=GROUND_LIMIT)]
                return f, boxes, usage, time.perf_counter() - t1
            with ThreadPoolExecutor(vlm.MAX_SEQS) as pool:
                got = list(pool.map(ask, jobs))
            lrec = {"requests": len(jobs), "wall_s": round(time.perf_counter() - t, 2), "per_request_s": [round(g[3], 2) for g in got],
                    "completion_tokens": sum(g[2]["completion_tokens"] for g in got), "hit_cap": sum(g[2]["completion_tokens"] >= GROUND_TOKENS for g in got)}
            listing = {f: merge_boxes([b for g in got if g[0] == f for b in g[1]]) for f in ref}
            lrec["boxes_per_frame"] = {str(f): len(v) for f, v in listing.items()}
            rec["listing"] = lrec

        def listing_masks(d, f):
            items = listing[f]
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                preds[d].set_image(np.ascontiguousarray(frames[f][..., ::-1]))
                m, s = prompt(d, boxes=np.array([b for b, _, _ in items], float)) if items else (np.zeros((0, *hw), bool), np.zeros(0))
            put(f, "listing", m, s, [f"{lab}|{src}" for _, lab, src in items])
        with clock.stage("sam2.listing", n=len(ref)):
            both(ref, listing_masks)

        rec["per_frame_s"] = {k: {"median": round(float(np.median([x["s"] for x in v])), 4), "mean": round(float(np.mean([x["s"] for x in v])), 4),
                                  "frames": len(v), "rows": v} for k, v in times.items()}
        rec["counts"] = {m: {str(f): int(len(out[f"{f}/{m}/score"])) for f in allf if f"{f}/{m}/score" in out}
                         for m in ["aux", "vocab", "generic", "ram", "owlw", "owlbox", "ridge", "geo", "geosam", "listing", *AMG]}
        return out, rec

    @modal.method()
    def stack_time(self, name: str, mp4: bytes, keys: list, keys_5fps: list, components: list, tag_list: list):
        """A discovery stack on every given keyframe of a video, both GPUs (frames alternate), timed as the fast core
        would add it: keyframes already decoded and on their GPU, SAM 3 vision features, person/floor masks and DA3
        depth already computed (the core's own work: timed here, reported apart, before t0). t0 -> both GPUs done =
        added seconds. -> record (per component seconds, masks per frame, per-GPU peaks)."""
        import cv2
        import torch
        from fast_report import segment
        from fast_report.stubs import Clock, Vram
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        tmp = Path(f"/tmp/{name}-stack.mp4")
        tmp.write_bytes(mp4)
        cap, frames = cv2.VideoCapture(str(tmp)), []
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            frames.append(bgr)
        H, W = frames[0].shape[:2]
        hw = (H // 2, W // 2) if W >= 1280 else (H, W)
        segment.DA3_HW = hw
        if any(c == "owlw" for c in components) and (self.tag_queries is None or self.tag_queries[0] != len(tag_list)):
            self.tag_queries = (len(tag_list), {d: self.owl[d].encode(tag_list) for d in self.devs})
        dev_of = {f: self.devs[i % 2] for i, f in enumerate(keys)}
        pre = {}
        t = time.perf_counter()
        gpu_frames, feats, aux = {}, {}, {}
        for f in keys:  # the core's state: keyframes on their GPU, SAM 3 features, person/floor masks
            d = dev_of[f]
            gpu_frames[f] = torch.from_numpy(frames[f]).to(d)
            with torch.cuda.device(d), torch.inference_mode():
                feats[f] = self.sam3[d].vision(gpu_frames[f][None])
                r = self.sam3[d].detect(feats[f], 1, tuple(AUX), SAVE_SCORE, top=PERSON_TOP_AUX)
                aux[f] = (r["mask"].cpu().numpy(), r["score"].float().cpu().numpy(), np.array([AUX[i] for i in r["word"].tolist()]))
        for d in self.devs:
            torch.cuda.synchronize(d)
        pre["core_state_s"] = round(time.perf_counter() - t, 2)
        clock = Clock()
        vram = Vram(self.devs, clock)
        vram.start()
        preds = {d: SAM2ImagePredictor(self.sam2[d]) for d in self.devs}
        comp_s = {}
        counts = {}
        lock = __import__("threading").Lock()

        def add(k, t0, n=0):
            with lock:
                comp_s[k] = comp_s.get(k, 0.) + time.perf_counter() - t0
                counts[k] = counts.get(k, 0) + n

        def one(d, f):
            rgb_gpu = gpu_frames[f].flip(-1)
            rgb = None
            need_embed = any(c in ("owlbox", "ridge", "ridge2", "geosam") for c in components)
            if "generic" in components:
                t0 = time.perf_counter()
                with torch.inference_mode():
                    r = self.sam3[d].detect(feats[f], 1, tuple(GENERIC), VOCAB_SCORE)
                torch.cuda.synchronize(d)
                add("generic", t0, len(r["score"]))
            if "labelw" in components:
                t0 = time.perf_counter()
                with torch.inference_mode():
                    r = self.sam3[d].detect(feats[f], 1, tuple(LABEL_WORDS), VOCAB_SCORE)
                torch.cuda.synchronize(d)
                add("labelw", t0, len(r["score"]))
            owl_out = None
            if "owlbox" in components or "owlw" in components:
                t0 = time.perf_counter()
                q = self.tag_queries[1][d] if self.tag_queries else self.owl[d].encode(["object"])
                owl_out = self.owl[d].detect(rgb_gpu, q)
                torch.cuda.synchronize(d)
                add("owl", t0)
            if "owlw" in components:
                t0 = time.perf_counter()
                boxes, obj, word, sc = owl_out
                keep = obj >= OWL_MIN_OBJ
                ws = clean_words([tag_list[i] for i, x in sorted(zip(word[keep].tolist(), sc[keep].tolist()), key=lambda z: -z[1]) if x >= OWL_WORD_MIN], OWL_WORDS)
                if ws:
                    with torch.inference_mode():
                        r = self.sam3[d].detect(feats[f], 1, tuple(ws), VOCAB_SCORE)
                    torch.cuda.synchronize(d)
                    add("owlw_sam3", t0, len(r["score"]))
            if need_embed:
                t0 = time.perf_counter()
                rgb = np.ascontiguousarray(frames[f][..., ::-1])
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    preds[d].set_image(rgb)
                torch.cuda.synchronize(d)
                add("sam2_embed", t0)
            p = preds[d]

            def decode(pts=None, labels=None, boxes=None):
                hw0 = p._orig_hw[-1]
                pc = p._transforms.transform_coords(torch.as_tensor(pts, dtype=torch.float32, device=d), normalize=True, orig_hw=hw0) if pts is not None else None
                pl = torch.as_tensor(labels, dtype=torch.int, device=d) if labels is not None else None
                bx = p._transforms.transform_boxes(torch.as_tensor(boxes, dtype=torch.float32, device=d), normalize=True, orig_hw=hw0) if boxes is not None else None
                n, k = (len(pts) if pts is not None else len(boxes)), 0
                for i in range(0, n, 64):
                    sl = slice(i, i + 64)
                    m, _, _ = p._predict(pc[sl] if pc is not None else None, pl[sl] if pl is not None else None, bx[sl] if bx is not None else None, multimask_output=False)
                    small = torch.nn.functional.interpolate(m[:, :1].half(), size=hw, mode="area")[:, 0] > .3
                    k += int(small.shape[0])
                return k
            if "owlbox" in components:
                t0 = time.perf_counter()
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    k = decode(boxes=owl_out[0]) if len(owl_out[0]) else 0
                torch.cuda.synchronize(d)
                add("owlbox_decode", t0, k)
            if "ridge" in components:
                t0 = time.perf_counter()
                g = cv2.cvtColor(frames[f], cv2.COLOR_BGR2GRAY)
                strength = ridge_map(torch.from_numpy(g).to(d).float() / 255).cpu().numpy()
                sc = W / 1280
                comps = thin_components(strength, RIDGE_MIN, RIDGE_LEN * sc, RIDGE_WIDTH * sc, RIDGE_TOP)
                k = 0
                if comps:
                    pts = np.stack([along(ys, xs, 5) for ys, xs, _, _ in comps])
                    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                        k = decode(pts=pts, labels=np.ones(pts.shape[:2]))
                torch.cuda.synchronize(d)
                add("ridge", t0, k)
            if "ridge2" in components:
                t0 = time.perf_counter()
                g = cv2.cvtColor(frames[f], cv2.COLOR_BGR2GRAY)
                strength = ridge_map(torch.from_numpy(g).to(d).float() / 255).cpu().numpy()
                sc = W / 1280
                comps = [c for c in thin_components(strength, RIDGE_MIN, RIDGE_LEN * sc, RIDGE_WIDTH * sc, 4 * 60) if straightness(c[0], c[1]) >= .012][:60]
                k = 0
                if comps:
                    pts = np.stack([along(ys, xs, 5) for ys, xs, _, _ in comps])
                    hw0 = p._orig_hw[-1]
                    pc = p._transforms.transform_coords(torch.as_tensor(pts, dtype=torch.float32, device=d), normalize=True, orig_hw=hw0)
                    pl = torch.ones(pts.shape[:2], dtype=torch.int, device=d)
                    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                        for i in range(0, len(pts), 64):
                            m, sco, _ = p._predict(pc[i:i + 64], pl[i:i + 64], None, multimask_output=True)
                            small = (torch.nn.functional.interpolate(m.flatten(0, 1)[:, None].float(), size=hw, mode="area")[:, 0] > 0).view(m.shape[0], m.shape[1], *hw).cpu().numpy()
                            thinnest(small, sco.float().cpu().numpy())
                            k += len(m)
                torch.cuda.synchronize(d)
                add("ridge2", t0, k)
            for c in components:
                if c in AMG:
                    t0 = time.perf_counter()
                    side, crops = AMG[c]
                    rgb = rgb if rgb is not None else np.ascontiguousarray(frames[f][..., ::-1])
                    gen = SAM2AutomaticMaskGenerator(self.sam2[d], points_per_side=side, crop_n_layers=crops, **AMG_SETTINGS)
                    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                        found = gen.generate(rgb)
                    if found:
                        mk = torch.from_numpy(np.stack([x["segmentation"] for x in found])).to(d)
                        (torch.nn.functional.interpolate(mk[:, None].half(), size=hw, mode="area")[:, 0] > .3).cpu()
                    torch.cuda.synchronize(d)
                    add(c, t0, len(found))
            return f

        def worker(d, part):
            with torch.cuda.device(d):
                return [one(d, f) for f in part]
        t0 = time.perf_counter()
        with ThreadPoolExecutor(2) as pool:
            jobs = [pool.submit(worker, dv, [f for f in keys if dev_of[f] == dv]) for dv in self.devs]
            [j.result() for j in jobs]
        added = time.perf_counter() - t0
        vram.stop()
        rep = clock.report(vram)
        for d in self.devs:
            with torch.cuda.device(d):
                torch.cuda.empty_cache()
        return {"video": name, "components": components, "keyframes": len(keys), "added_s": round(added, 2), "pre_core_state_s_not_added": pre,
                "component_gpu_s_sum": {k: round(v, 2) for k, v in comp_s.items()}, "masks_per_frame": {k: round(v / len(keys), 1) for k, v in counts.items()},
                "gpu_peak": rep["gpu_peak"], "flags": rep["flags"],
                "note": "added_s: wall from t0 (core state ready) until both GPUs finish; component sums are GPU-thread seconds (two threads)"}

    @modal.method()
    def ridge2(self, name: str, mp4: bytes, frames_idx: list, straight_max: float = .012, top: int = 60):
        """(e) v2 on the given frames: dark and bright ridges, straight components dropped (shelf edges, lights, floor
        lines: straightness < straight_max), 5 points along each, SAM 2.1 multimask -> the thinnest mask with score
        >= 0.5. -> (npz bytes of masks at the evaluation grid, per-frame seconds)."""
        import cv2
        import torch
        import torch.nn.functional as F
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        tmp = Path(f"/tmp/{name}-r2.mp4")
        tmp.write_bytes(mp4)
        cap, frames = cv2.VideoCapture(str(tmp)), []
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            frames.append(bgr)
        H, W = frames[0].shape[:2]
        hw = (H // 2, W // 2) if W >= 1280 else (H, W)
        out, times = {}, {}
        preds = {d: SAM2ImagePredictor(self.sam2[d]) for d in self.devs}

        def one(d, f):
            t = time.perf_counter()
            p = preds[d]
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                p.set_image(np.ascontiguousarray(frames[f][..., ::-1]))
                g = torch.from_numpy(cv2.cvtColor(frames[f], cv2.COLOR_BGR2GRAY)).to(d).float() / 255
                strength = ridge_map(g).cpu().numpy()
                sc = W / 1280
                comps = [c for c in thin_components(strength, RIDGE_MIN, RIDGE_LEN * sc, RIDGE_WIDTH * sc, 4 * top)
                         if straightness(c[0], c[1]) >= straight_max][:top]
                masks, scores = np.zeros((0, *hw), bool), np.zeros(0, np.float32)
                if comps:
                    pts = np.stack([along(ys, xs, 5) for ys, xs, _, _ in comps])
                    hw0 = p._orig_hw[-1]
                    pc = p._transforms.transform_coords(torch.as_tensor(pts, dtype=torch.float32, device=d), normalize=True, orig_hw=hw0)
                    pl = torch.ones(pts.shape[:2], dtype=torch.int, device=d)
                    ms, ss = [], []
                    for i in range(0, len(pts), 64):
                        m, sco, _ = p._predict(pc[i:i + 64], pl[i:i + 64], None, multimask_output=True)
                        small = (F.interpolate(m.flatten(0, 1)[:, None].float(), size=hw, mode="area")[:, 0] > 0).view(m.shape[0], m.shape[1], *hw).cpu().numpy()
                        sco = sco.float().cpu().numpy()
                        pick = thinnest(small, sco)
                        ms.append(small[np.arange(len(pick)), pick])
                        ss.append(sco[np.arange(len(pick)), pick])
                    masks, scores = np.concatenate(ms), np.concatenate(ss)
            torch.cuda.synchronize(d)
            times[f] = round(time.perf_counter() - t, 4)
            out[f"{f}/ridge2/bits"], out[f"{f}/ridge2/score"] = pack(masks), scores.astype(np.float32)
            out[f"{f}/ridge2/label"] = np.array([""] * len(scores), "U64")

        def worker(d, part):
            with torch.cuda.device(d):
                for f in part:
                    one(d, f)
        with ThreadPoolExecutor(2) as pool:
            [j.result() for j in [pool.submit(worker, self.devs[i], frames_idx[i::2]) for i in range(2)]]
        buf = io.BytesIO()
        np.savez_compressed(buf, **out)
        return buf.getvalue(), times

    @modal.method()
    def words_run(self, name: str, mp4: bytes, frames_idx: list, sets: dict):
        """SAM 3 word sets on the given frames, whole frame or on 2 x 2 upscaled tiles (tiles(): 60% of the frame each,
        each tile through SAM 3's own 1008 px resize, masks pasted back). sets: {method: [words, tiled]}. -> (npz bytes
        at the evaluation grid, per method per-frame seconds (vision included: tiles need their own features))."""
        import cv2
        import torch
        from fast_report import segment
        tmp = Path(f"/tmp/{name}-w.mp4")
        tmp.write_bytes(mp4)
        cap, frames = cv2.VideoCapture(str(tmp)), []
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            frames.append(bgr)
        H, W = frames[0].shape[:2]
        hw = (H // 2, W // 2) if W >= 1280 else (H, W)
        out, times = {}, {}
        tl = tiles(hw[1], hw[0])  # tile boxes on the evaluation grid
        src = tiles(W, H)         # the same tiles in source pixels
        for method, (words, tiled) in sets.items():
            segment.DA3_HW = (tl[0][3] - tl[0][1], tl[0][2] - tl[0][0]) if tiled else hw

            def one(d, f):
                t = time.perf_counter()
                with torch.inference_mode():
                    if tiled:
                        x = torch.stack([torch.from_numpy(np.ascontiguousarray(frames[f][y0:y1, x0:x1])) for x0, y0, x1, y1 in src]).to(d)
                        v = self.sam3[d].vision(x)
                        r = self.sam3[d].detect(v, len(src), tuple(words), SAVE_SCORE)
                        m = r["mask"].cpu().numpy()
                        full = np.zeros((len(m), *hw), bool)
                        for i, k in enumerate(r["frame"].tolist()):
                            x0, y0, x1, y1 = tl[k]
                            full[i, y0:y1, x0:x1] = m[i]
                        masks = full
                    else:
                        v = self.sam3[d].vision(torch.from_numpy(frames[f]).to(d)[None])
                        r = self.sam3[d].detect(v, 1, tuple(words), SAVE_SCORE)
                        masks = r["mask"].cpu().numpy()
                torch.cuda.synchronize(d)
                times.setdefault(method, {})[f] = round(time.perf_counter() - t, 4)
                out[f"{f}/{method}/bits"], out[f"{f}/{method}/score"] = pack(masks), r["score"].float().cpu().numpy()
                out[f"{f}/{method}/label"] = np.array([words[i] for i in r["word"].tolist()], "U64")

            def worker(d, part):
                with torch.cuda.device(d):
                    for f in part:
                        one(d, f)
            with ThreadPoolExecutor(2) as pool:
                [j.result() for j in [pool.submit(worker, self.devs[i], frames_idx[i::2]) for i in range(2)]]
        buf = io.BytesIO()
        np.savez_compressed(buf, **out)
        return buf.getvalue(), times

    def geometry(self, frames, keys, f, out, hw):
        """(d) for reference frame f: DA3 on the 9 5-fps keyframes around it (one forward, the core's model and grid),
        metric scale from the floor plane (SAM 3 'floor' on f) + 1.6 m camera height (estimated), large planes by
        RANSAC over the window's points, pixels of f not within tau of any plane (not people, not beyond FAR_M),
        split at depth edges -> components -> raw masks (method 'geo') + SAM 2.1 prompts (box + interior point)."""
        import cv2
        import torch
        import torch.nn.functional as F
        from fast_report import core
        d = self.devs[0]
        i = keys.index(f)
        win = keys[max(0, i - WINDOW): i + WINDOW + 1]
        at = win.index(f)
        t = time.perf_counter()
        with torch.inference_mode():
            kf = torch.from_numpy(np.stack([frames[k] for k in win])).to(d)
            g = self.da3.shot(kf)
        torch.cuda.synchronize(d)
        da3_s = time.perf_counter() - t
        t = time.perf_counter()
        gh, gw = core.DA3_HW
        aux_bits, aux_lab = out[f"{f}/aux/bits"], out[f"{f}/aux/label"]
        aux = unpack(aux_bits, hw[1]) if len(aux_bits) else np.zeros((0, *hw), bool)
        grid = lambda m: F.interpolate(torch.as_tensor(m[None, None], device=d).float(), size=(gh, gw), mode="area")[0, 0] > .5  # noqa: E731
        floor_m = grid(aux[aux_lab == "floor"].any(0)) if (aux_lab == "floor").any() else torch.zeros((gh, gw), dtype=torch.bool, device=d)
        person = grid(aux[aux_lab == "person"].any(0)) if (aux_lab == "person").any() else torch.zeros((gh, gw), dtype=torch.bool, device=d)
        floor = torch.zeros((len(win), gh, gw), dtype=torch.bool, device=d)
        floor[at] = floor_m
        plane = core.floor_plane(g["depth"], g["K"], g["c2w"], floor, stride=2)
        mpu = core.CAMERA_HEIGHT_M / plane["camera_height_units"] if plane else None
        grec = {"window": win, "da3_s": round(da3_s, 3), "floor_plane": plane is not None}
        if mpu is None:
            out[f"{f}/geo/bits"], out[f"{f}/geo/score"], out[f"{f}/geo/label"] = pack(np.zeros((0, *hw), bool)), np.zeros(0, np.float32), np.array([], "U64")
            grec["planes_s"] = round(time.perf_counter() - t, 3)
            return [], grec
        s = 4
        fi, vy, vx = torch.nonzero(g["depth"][:, ::s, ::s] > 0, as_tuple=True)
        from fast_report.segment import backproject
        pts = backproject(g["depth"], g["K"], g["c2w"], fi, vy, vx, s) * mpu
        planes = ransac_planes(pts, PLANE_TAU_M)
        planes.append((plane["normal"].to(d), -(plane["normal"].to(d) * plane["point"].to(d)).sum() * mpu))
        yy, xx = torch.meshgrid(torch.arange(gh, device=d), torch.arange(gw, device=d), indexing="ij")
        allp = backproject(g["depth"], g["K"], g["c2w"], torch.full((gh * gw,), at, device=d), yy.flatten(), xx.flatten(), 1) * mpu
        z = g["depth"][at].flatten() * mpu
        tau = torch.clamp(z * PLANE_TAU_REL, min=PLANE_TAU_M)
        dist = torch.stack([(allp @ u.float() + dd).abs() for u, dd in planes]).amin(0)
        resid = ((dist > tau) & (z < FAR_M)).view(gh, gw) & ~person
        lz = torch.log(g["depth"][at].clamp(min=1e-6))
        edge = torch.zeros_like(resid)
        edge[:, 1:] |= (lz[:, 1:] - lz[:, :-1]).abs() > DEPTH_EDGE
        edge[1:, :] |= (lz[1:, :] - lz[:-1, :]).abs() > DEPTH_EDGE
        r = (resid & ~edge).cpu().numpy().astype(np.uint8)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(r, connectivity=4)
        H, W = frames[f].shape[:2]
        comps, masks = [], []
        for c in range(1, n):
            x, y, w, h, a = stats[c]
            if a < GEO_MIN_PX or a > GEO_MAX_SHARE * gh * gw:
                continue
            m = lab == c
            dt = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 3)
            py, px = np.unravel_index(dt.argmax(), dt.shape)
            comps.append({"box": [x * W / gw, y * H / gh, (x + w) * W / gw, (y + h) * H / gh], "point": [(px + .5) * W / gw, (py + .5) * H / gh], "px": int(a)})
            masks.append(cv2.resize(m.astype(np.uint8), (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST) > 0)
        out[f"{f}/geo/bits"] = pack(np.array(masks, bool).reshape(-1, *hw))
        out[f"{f}/geo/score"] = np.array([c["px"] for c in comps], np.float32)
        out[f"{f}/geo/label"] = np.array([""] * len(comps), "U64")
        torch.cuda.synchronize(d)
        grec.update(planes=len(planes), metres_per_unit=round(mpu, 4), residual_share=round(float(resid.float().mean()), 3), components=len(comps),
                    planes_s=round(time.perf_counter() - t, 3), scale="estimated (floor plane + assumed 1.6 m camera height)")
        return comps, grec


PERSON_TOP_AUX = 12


@app.local_entrypoint()
def main(frames: str, jpg: str, out: str, videos: str = "me340,samsclub,walmart,lightning", run_id: str = ""):
    """frames.json (scripts/x10_frames.py) -> RAM++ tags (own container) -> X10 on each video -> OUT/<video>.npz + .json."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    plan = json.loads(Path(frames).read_text())
    names = videos.split(",")
    run_id = run_id or out.name
    meta = {"run_id": run_id, "videos": names, "started_unix": time.time(), "licences": LICENCES}
    ram_in = {}
    for v in names:
        p = plan[v]
        fs = p["reference"] + [x for pair in p["neighbours"].values() for x in pair if x is not None]
        for f in sorted(set(fs)):
            ram_in[f"{v}/{f}"] = (Path(jpg) / f"{v}-{f:05d}.jpg").read_bytes()
    x10 = X10()
    t_ram = time.time()
    ram_call = ram_tags.spawn(ram_in)
    t_boot = time.time()
    boot = x10.boot_info.remote()
    meta["boot"] = {**boot, "client_submit_to_ready_s": round(time.time() - t_boot, 1)}
    ram = ram_call.get()
    meta["ram"] = {k: v for k, v in ram.items() if k not in ("tags", "tag_list")} | {"client_s": round(time.time() - t_ram, 1)}
    (out / "ram.json").write_text(json.dumps(ram, indent=1))
    print("boot", json.dumps({k: v for k, v in boot.items() if k.endswith("_s") or k == "resident_gb"}), flush=True)
    for v in names:
        p = plan[v]
        tags = {k.split("/")[1]: t for k, t in ram["tags"].items() if k.startswith(v + "/")}
        mp4 = Path(p["video"]).read_bytes()
        t = time.time()
        npz, rec = x10.run_video.remote(v, mp4, {"reference": p["reference"], "neighbours": p["neighbours"], "keys_5fps": p["keys_5fps"]}, tags, ram["tag_list"], run_id)
        rec["client_s"] = round(time.time() - t, 1)
        free = os.statvfs(str(out))
        free_gb = free.f_bavail * free.f_frsize / 1e9
        rec["local_free_gb_before_write"] = round(free_gb, 1)
        if free_gb < 8:
            print("local disk < 8 GB free: npz kept on the volume only", flush=True)
        else:
            (out / f"{v}.npz").write_bytes(npz)
        (out / f"{v}.json").write_text(json.dumps(rec, indent=1, default=str))
        print(v, "error:", rec.get("error"), "client_s", rec["client_s"], json.dumps({k: x["median"] for k, x in rec.get("per_frame_s", {}).items()}), flush=True)
    meta["finished_unix"] = time.time()
    meta["usd_upper"] = round((meta["finished_unix"] - t_boot) * (2 * PRICE["A100-80GB"] + CPU * PRICE["cpu_core"] + MEMORY_GIB * PRICE["gib"])
                              + sum(ram["per_frame_s"]) * PRICE["A100-80GB"] + (ram["load_s"] + 120) * PRICE["A100-80GB"], 3)
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str))


@app.local_entrypoint()
def stack(frames: str, out: str, jpg: str = "", videos: str = "me340,samsclub,walmart,lightning", stacks: str = "generic+owlbox;generic+owlbox+ridge;generic+owlbox+ridge+amg16;generic+owlbox+amg32",
          fps5: str = "generic+owlbox"):
    """Added seconds of discovery stacks on every object keyframe (and, for `fps5`, every 5 fps keyframe) of each
    video, both GPUs, core state ready first (not counted) + ridge v2 masks on the reference frames (and their
    neighbours) for scoring. -> OUT/stack-time.json, OUT/<video>-ridge2.npz."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    plan = json.loads(Path(frames).read_text())
    x10 = X10()
    t0 = time.time()
    boot = x10.boot_info.remote()
    rows = {"boot": boot, "runs": []}
    tag_list = json.loads((Path(frames).parent / "run001" / "ram.json").read_text())["tag_list"] if (Path(frames).parent / "run001" / "ram.json").exists() else []
    for v in videos.split(","):
        p = plan[v]
        mp4 = Path(p["video"]).read_bytes()
        ref_nb = sorted(set(p["reference"]) | {x for pair in p["neighbours"].values() for x in pair if x is not None})
        npz, rt = x10.ridge2.remote(v, mp4, ref_nb)
        (out / f"{v}-ridge2.npz").write_bytes(npz)
        rows["runs"].append({"video": v, "ridge2_per_frame_s": rt})
        for st in stacks.split(";"):
            r = x10.stack_time.remote(v, mp4, p["object_keys"], p["keys_5fps"], st.split("+"), tag_list)
            r["keyframes_kind"] = "object keyframes (every 3rd 5 fps keyframe)"
            rows["runs"].append(r)
            print(v, st, "object keys", r["keyframes"], "added_s", r["added_s"], r["component_gpu_s_sum"], r["flags"], flush=True)
        if fps5:
            r = x10.stack_time.remote(v, mp4, p["keys_5fps"], p["keys_5fps"], fps5.split("+"), tag_list)
            r["keyframes_kind"] = "every 5 fps keyframe"
            rows["runs"].append(r)
            print(v, fps5, "5 fps keys", r["keyframes"], "added_s", r["added_s"], flush=True)
        (out / "stack-time.json").write_text(json.dumps(rows, indent=1, default=str))
    rows["usd_upper"] = round((time.time() - t0) * (2 * PRICE["A100-80GB"] + CPU * PRICE["cpu_core"] + MEMORY_GIB * PRICE["gib"]), 3)
    (out / "stack-time.json").write_text(json.dumps(rows, indent=1, default=str))


LABEL_WORDS = ["label", "sticker", "tag", "sign", "placard", "price tag", "warning label"]


@app.local_entrypoint()
def words(frames: str, out: str, videos: str = "me340,samsclub,walmart,lightning"):
    """Word-set follow-ups on the reference frames: a label/sign word set (whole frame and tiled) and the generic
    words on tiles -> OUT/<video>-words.npz + OUT/words-time.json."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    plan = json.loads(Path(frames).read_text())
    x10 = X10()
    t0 = time.time()
    rows = {"boot": x10.boot_info.remote(), "sets": {"labelw": [LABEL_WORDS, False], "labelw_t": [LABEL_WORDS, True], "generic_t": [GENERIC, True]}, "videos": {}}
    for v in videos.split(","):
        p = plan[v]
        npz, t = x10.words_run.remote(v, Path(p["video"]).read_bytes(), p["reference"], rows["sets"])
        (out / f"{v}-words.npz").write_bytes(npz)
        rows["videos"][v] = t
        print(v, {k: round(float(np.median(list(x.values()))), 3) for k, x in t.items()}, flush=True)
    rows["usd_upper"] = round((time.time() - t0) * (2 * PRICE["A100-80GB"] + CPU * PRICE["cpu_core"] + MEMORY_GIB * PRICE["gib"]), 3)
    (out / "words-time.json").write_text(json.dumps(rows, indent=1, default=str))


if __name__ == "__main__":
    assert sys.argv[1:] == ["--self-check"], __doc__
    self_check_cpu()
