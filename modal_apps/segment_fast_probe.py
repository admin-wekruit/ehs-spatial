"""Fast-path probes E2 + E6 (docs/phase2/FAST-PATH-PLAN.md section 7) on the ME340 walk shot, all GPU work on A100-80GB.

E2: SAM 3 (encode once, text encoded once per vocabulary, frames x prompts batched into one forward) on ~2 fps walk
    keyframes with (a) {person, floor}, (b) 20 EHS words, (c) today's own names as an oracle vocabulary. Warm s/frame, and
    recall of today's named objects (runs/me340-entity-names-200): found = some SAM 3 mask has IoU >= 0.5 with one of the
    object's observed masks on a shared keyframe.
E6: SAM 2 automatic masks only on keyframes; in-between frames get the keyframe's masks carried by (i) forward
    reprojection with the reference DROID poses + DA3 posed depth, (ii) SAM 2 video propagation (sam2 main = one object at
    a time, and the last commit that still batched objects). Agreement with today's per-frame masks (me340-masks-194) on
    the in-between frames, and time against today's per-frame loop run on the same A100. Also AMG with bigger point
    batches, and box prompts encode-once vs today's re-encode-per-box loop.

  modal run modal_apps/segment_fast_probe.py --out RUNS/m3-exp-e2-e6-segment-probe-N [--only sam3,sam2,sam2old] [--smoke]
  (inputs are built once, locally, into RUNS/m3-exp-e2-e6-segment-inputs; RUNS = research-notes/phase2/runs)
  python modal_apps/segment_fast_probe.py --self-check          # no GPU
Measures, does not assert thresholds: every pass/fail is judged from the returned numbers.
"""
import io
import json
import sys
import time
from pathlib import Path

import modal
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import sam2_everything as s2  # noqa: E402  pinned SAM 2.1 checkpoint, today's AMG settings, weight volume
import sam3_app as s3  # noqa: E402  pinned SAM 3 revision, image and weight volume

PHASE2 = Path("/Users/adam/Desktop/panoptes-public/research-notes/phase2")
CLIP = PHASE2 / "data/clips/me340-165"
MASKS = PHASE2 / "runs/me340-masks-194/object-a"
NAMES = PHASE2 / "runs/me340-entity-names-200"
DROID = PHASE2 / "runs/droid-me340-165-171"
DEPTH = PHASE2 / "runs/da3-posed-me340-223-shotc"

WALK = list(range(228, 898, 3))  # today's 10 fps mask grid inside the walk shot (226 is the cut, 228 the first grid frame)
E2_KEYS = WALK[::5]  # every 15th frame, ~2 fps, all on today's mask grid
SPACINGS = (3, 6, 15, 30)  # source frames between keyframes: 10 (AMG-vs-AMG ceiling), 5, 2, 1 fps
BASIC = ["person", "floor"]
EHS = ["machine", "cabinet", "shelf", "workbench", "cart", "ladder", "fire extinguisher", "bin", "box", "pallet", "forklift",
       "vise", "drill press", "lathe", "control panel", "hose", "cable", "door", "sign", "chair"]
MIN_AREA = 300  # px on the DROID raster: today's AMG keeps nothing smaller (sam2_everything.MIN_AREA at 640x480)
SAM2_BATCHED = "c2ec8e14a185632b0a5d8b161928ceb50197eddc"  # last sam2 commit whose video predictor runs all objects as one batch
USD_PER_S = {"A100-80GB": 0.000694, "cpu_core": 0.0000131, "mem_gib": 0.00000222}  # Modal list prices

extra = ("sam2_everything", "sam3_app", "moge3_app")
S3_IMAGE = s3.image.add_local_python_source(*extra)
S2_IMAGE = s2.image.add_local_python_source("sam2_everything", "sam3_app")
S2_OLD_IMAGE = (modal.Image.debian_slim(python_version="3.11").apt_install("git", "libgl1", "libglib2.0-0")
                .pip_install("torch", "torchvision", "opencv-python-headless", "huggingface_hub")
                .env({"SAM2_BUILD_CUDA": "0", "HF_HOME": "/cache/huggingface"})
                .pip_install(f"git+https://github.com/facebookresearch/sam2.git@{SAM2_BATCHED}")
                .add_local_python_source(*extra))
app = modal.App("panoptes-segment-fast-probe")
GPU = dict(gpu="A100-80GB", retries=0, max_containers=1, scaledown_window=2)


# ---------- pure helpers (numpy; also run locally by the self-check) ----------

def agreement(today, carried, min_area=MIN_AREA):
    """Best IoU of each of today's masks (area >= min_area) with any carried mask. Both are label maps, 0 = no mask."""
    today, carried = today.astype(np.int64).ravel(), carried.astype(np.int64).ravel()
    nc = int(carried.max()) + 1
    table = np.bincount(today * nc + carried, minlength=(int(today.max()) + 1) * nc).reshape(-1, nc)
    area_t, area_c, inter = table.sum(1), table.sum(0), table[1:, 1:]
    union = area_t[1:, None] + area_c[None, 1:] - inter
    iou = inter / np.maximum(union, 1)
    best = iou.max(1) if iou.shape[1] else np.zeros(len(inter))
    keep = area_t[1:] >= min_area
    return best[keep], area_t[1:][keep]


def summarize(rows):
    """rows: [(best IoUs, areas, frames from keyframe)] -> mask-level agreement, overall and per distance."""
    if not rows:
        return {"frames": 0}
    best = np.concatenate([r[0] for r in rows])
    area = np.concatenate([r[1] for r in rows]).astype(float)
    out = {"frames": len(rows), "masks": int(best.size), "mean_best_iou": round(float(best.mean()), 4),
           "area_weighted_best_iou": round(float((best * area).sum() / area.sum()), 4),
           "share_iou_ge_0.5": round(float((best >= .5).mean()), 4), "share_iou_ge_0.85": round(float((best >= .85).mean()), 4)}
    by = {}
    for b, a, d in rows:
        by.setdefault(int(d), []).append(b)
    out["mean_best_iou_by_distance"] = {d: round(float(np.concatenate(v).mean()), 4) for d, v in sorted(by.items())}
    return out


def to_droid(label):
    """Raw 640x480 raster -> DROID's TUM raster (droid_room.prepare_image, scale 2), nearest so labels stay labels."""
    import cv2
    return cv2.resize(label, (704, 512), interpolation=cv2.INTER_NEAREST)[16:-16, 32:-32]


def droid_k(source_k):
    fx, fy, cx, cy = source_k  # droid_room.prepare_image intrinsics at resolution scale 2
    return [fx * 352 / 640 * 2, fy * 256 / 480 * 2, (cx * 352 / 640 - 16) * 2, (cy * 256 / 480 - 8) * 2]


def segments(n, step):
    """Keyframe indices every `step` grid frames; each in-between index is carried from the keyframe before it."""
    return [(k, list(range(k + 1, min(k + step, n)))) for k in range(0, n, step)]


def npz_bytes(**arrays):
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    return buffer.getvalue()


def self_check():
    square = np.zeros((40, 40), np.int32)
    square[10:30, 10:30] = 1
    best, area = agreement(square, square, min_area=1)
    assert best.tolist() == [1.0] and area.tolist() == [400], "a mask agrees with itself"
    shifted = np.roll(square, 10, axis=1) * 7  # any label id
    best, _ = agreement(square, shifted, min_area=1)
    assert abs(best[0] - 200 / 600) < 1e-9, "half-overlapping squares: IoU 1/3"
    assert agreement(square, np.zeros_like(square), min_area=1)[0].tolist() == [0.0], "nothing carried: IoU 0"
    assert agreement(square, square, min_area=401)[0].size == 0, "masks under min_area are not scored"
    assert segments(7, 3) == [(0, [1, 2]), (3, [4, 5]), (6, [])], "forward carry from the keyframe before"
    k = droid_k([377.51247406005854, 377.51247406005854, 319.5, 239.5])
    assert abs(k[0] / 2 - 207.63187) < 1e-3 and abs(k[1] / 2 - 201.33998) < 1e-3 and abs(k[2] / 2 - 159.725) < 1e-3, "matches DROID's fullres K"
    s = summarize([(np.array([1., .5]), np.array([100, 300]), 3), (np.array([0.]), np.array([100]), 6)])
    assert s["mean_best_iou"] == .5 and s["area_weighted_best_iou"] == round(250 / 500, 4) and s["mean_best_iou_by_distance"] == {3: .75, 6: 0.}
    print("segment_fast_probe self-check passed: agreement, carry segments, DROID K; no GPU invoked")


# ---------- GPU helpers ----------

def splat(labels, depth, rel, k):
    """Forward-reproject a keyframe label map (torch, DROID raster) into another view.

    rel: 4x4 keyframe camera -> target camera. Each pixel with depth lands on its nearest target pixel (z-buffer: the
    nearest surface wins); gaps that opens up (the target is closer, so magnified) take the nearest surface among 2x2
    footprints. Holes up to 3 px take a neighbour's label, larger ones stay 0 (unlabelled), as does anything the
    keyframe never saw. Pixels the depth rule dropped (object edges) borrow the farther neighbour's depth.
    ponytail: forward only from the previous keyframe; a backward pass from the next keyframe would fill new content.
    """
    import torch
    pool = torch.nn.functional.max_pool2d
    height, width = labels.shape
    fx, fy, cx, cy = k
    for _ in range(2):
        depth = torch.where(depth > 0, depth, pool(depth[None, None], 5, 1, 2)[0, 0])
    v, u = torch.meshgrid(torch.arange(height, device=labels.device, dtype=torch.float32),
                          torch.arange(width, device=labels.device, dtype=torch.float32), indexing="ij")
    ok = depth > 0
    z = depth[ok]
    points = torch.stack([(u[ok] - cx) / fx * z, (v[ok] - cy) / fy * z, z], -1) @ rel[:3, :3].T + rel[:3, 3]
    front = points[:, 2] > 1e-3
    points, label = points[front], labels[ok][front].long()
    x, y = points[:, 0] / points[:, 2] * fx + cx, points[:, 1] / points[:, 2] * fy + cy
    key = (points[:, 2] * 1e4).long().clamp(0, 2 ** 40) * 65536 + label  # z-buffer and label in one int64
    empty = torch.iinfo(torch.int64).max

    def scatter(out, xi, yi):
        inside = (xi >= 0) & (xi < width) & (yi >= 0) & (yi < height)
        return out.scatter_reduce_(0, (yi * width + xi)[inside], key[inside], reduce="amin")

    near = scatter(torch.full((height * width,), empty, dtype=torch.int64, device=labels.device), torch.round(x).long(), torch.round(y).long())
    wide = torch.full_like(near, empty)
    for dx in (0, 1):
        for dy in (0, 1):
            scatter(wide, torch.floor(x).long() + dx, torch.floor(y).long() + dy)
    out = torch.where(near != empty, near, wide)
    result = torch.where(out == empty, -1, out % 65536).view(1, 1, height, width).float()
    for _ in range(3):
        result = torch.where(result < 0, pool(result, 3, 1, 1), result)
    return result[0, 0].clamp(min=0).int()


def decode_walk(mp4, wanted):
    import cv2
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
        f.write(mp4)
        f.flush()
        cap, out = cv2.VideoCapture(f.name), {}
        for index in range(max(wanted) + 1):
            ok, bgr = cap.read()
            assert ok, "frame missing from the clip video"
            if index in wanted:
                out[index] = bgr
        cap.release()
    return out


def propagate(predictor, video_dir, labels, step, sync):
    """SAM 2 video: per keyframe, today's masks go in as one object each, and are propagated forward to the next keyframe.

    Returns {grid index: raw-raster label map} for the in-between frames and a time breakdown (GPU-synchronised)."""
    import torch
    t = time.perf_counter()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = predictor.init_state(video_path=video_dir)
    sync()
    times = {"init_state_s": time.perf_counter() - t, "add_masks_s": 0., "propagate_s": 0., "objects": 0, "keyframes": 0}
    carried = {}
    for key, between in segments(len(labels), step):
        if not between:
            continue
        t = time.perf_counter()
        ids = [int(i) for i in np.unique(labels[key]) if i]
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            predictor.reset_state(state)
            for i in ids:
                predictor.add_new_mask(state, frame_idx=key, obj_id=i, mask=labels[key] == i)
        sync()
        times["add_masks_s"] += time.perf_counter() - t
        times["objects"] += len(ids)
        times["keyframes"] += 1
        t = time.perf_counter()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for index, obj_ids, logits in predictor.propagate_in_video(state, start_frame_idx=key, max_frame_num_to_track=len(between)):
                if index == key:
                    continue
                score, which = logits[:, 0].float().max(0)
                carried[index] = torch.where(score > 0, torch.as_tensor(obj_ids, device=which.device)[which], 0).int().cpu().numpy()
        sync()
        times["propagate_s"] += time.perf_counter() - t
    times["frames_carried"] = len(carried)
    return carried, times


def video_probe(mp4, labels_npz, steps, limit=None):
    """SAM 2 video propagation for each keyframe step (grid frames), with whichever sam2 this image installed."""
    import cv2
    import torch
    from huggingface_hub import hf_hub_download
    from sam2.build_sam import HF_MODEL_ID_TO_FILENAMES, build_sam2_video_predictor
    sync = torch.cuda.synchronize
    report = {"gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__)}
    t = time.perf_counter()
    frames = decode_walk(mp4, set(WALK))
    labels = np.load(io.BytesIO(labels_npz))["labels"][:limit]
    Path("/tmp/walk").mkdir(exist_ok=True)
    for n, index in enumerate(WALK[:len(labels)]):
        cv2.imwrite(f"/tmp/walk/{n:05d}.jpg", frames[index], [cv2.IMWRITE_JPEG_QUALITY, 95])
    report["decode_and_jpeg_s"] = time.perf_counter() - t
    t = time.perf_counter()
    config, checkpoint = HF_MODEL_ID_TO_FILENAMES[s2.MODEL]
    predictor = build_sam2_video_predictor(config, hf_hub_download(s2.MODEL, checkpoint, revision=s2.REVISION))
    report["load_s"] = time.perf_counter() - t
    propagate(predictor, "/tmp/walk", labels[:4], 2, sync)  # warm-up, untimed
    report["steps"], maps = {}, {}
    for step in steps:
        carried, times = propagate(predictor, "/tmp/walk", labels, step, sync)
        rows = []
        for key, between in segments(len(labels), step):
            for n in between:
                rows.append((*agreement(to_droid(labels[n]), to_droid(carried[n].astype(np.uint16))), 3 * (n - key)))
        report["steps"][str(step)] = {"times": times, "agreement": summarize(rows)}
        maps[f"step{step}"] = np.stack([carried.get(n, np.zeros_like(labels[0], np.int32)) for n in range(len(labels))]).astype(np.uint16)
        torch.cuda.empty_cache()
    report["peak_gpu_mb"] = round(torch.cuda.max_memory_allocated() / 2 ** 20)
    return report, npz_bytes(**maps)


# ---------- Modal functions ----------

@app.function(image=S3_IMAGE, volumes={"/cache": s3.volume}, secrets=[modal.Secret.from_name("huggingface")], timeout=1200,
              cpu=4, memory=32768, **GPU)
def sam3_probe(pngs, labels_npz, named, sets, caps, loop_sets):
    import types
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from transformers import Sam3Model, Sam3Processor
    import transformers
    enter = time.time()
    sync = torch.cuda.synchronize
    t = time.perf_counter()
    processor = Sam3Processor.from_pretrained(s3.MODEL_ID, revision=s3.REVISION)
    model = Sam3Model.from_pretrained(s3.MODEL_ID, revision=s3.REVISION, torch_dtype=torch.bfloat16).to("cuda").eval()
    report = {"gpu": torch.cuda.get_device_name(), "transformers": transformers.__version__, "torch": str(torch.__version__),
              "load_s": round(time.perf_counter() - t, 2), "frames": len(pngs)}
    t = time.perf_counter()
    images = [Image.open(io.BytesIO(p)).convert("RGB") for p in pngs]
    report["png_decode_s"] = round(time.perf_counter() - t, 3)
    height, width = images[0].height, images[0].width
    labels = torch.from_numpy(np.load(io.BytesIO(labels_npz))["labels"].astype(np.int32)).cuda()
    vocab = sorted({w for words in sets.values() for w in words})

    t = time.perf_counter()  # the vocabulary is fixed: its text encoder runs once, not once per frame
    text = processor(text=vocab, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        encoded = model.get_text_features(input_ids=text["input_ids"], attention_mask=text["attention_mask"])
    wrapped = hasattr(encoded, "pooler_output")  # newer transformers return an output object, older a tensor
    text_feats, text_mask = (encoded.pooler_output if wrapped else encoded), text["attention_mask"]
    sync()
    report["text_encode_s"] = {"words": len(vocab), "s": round(time.perf_counter() - t, 3)}

    ip = processor.image_processor
    size = (ip.size["height"], ip.size["width"])
    mean = torch.tensor(ip.image_mean, device="cuda").view(1, 3, 1, 1)
    std = torch.tensor(ip.image_std, device="cuda").view(1, 3, 1, 1)

    def pixels(batch):  # the processor's resize + rescale + normalise, on the GPU for a whole batch
        x = torch.from_numpy(np.stack([np.asarray(i) for i in batch])).cuda().permute(0, 3, 1, 2).float() / 255
        x = F.interpolate(x, size=size, mode="bilinear", antialias=True, align_corners=False)
        return ((x - mean) / std).to(torch.bfloat16)

    reference = processor(images=images[:1], return_tensors="pt")["pixel_values"].cuda().float()
    report["gpu_preprocess_max_abs_diff_vs_processor"] = round(float((pixels(images[:1]).float() - reference).abs().max()), 4)

    def expand(vision, n):
        fields = {k: tuple(t.repeat_interleave(n, 0) for t in v) for k, v in vision.items() if k.startswith("fpn_")}
        return type(vision)(**fields)

    def batched(words, cap, frames=None):
        """Frames x prompts in one forward, `cap` (frame, prompt) pairs at a time; vision encoded once per frame."""
        frames = list(range(len(images))) if frames is None else frames
        idx = [vocab.index(w) for w in words]
        per = min(len(idx), cap)  # prompts per forward
        chunk_n = max(1, cap // len(idx))  # frames per forward
        out = {}
        sync()
        t = time.perf_counter()
        for s in range(0, len(frames), chunk_n):
            chunk = frames[s:s + chunk_n]
            with torch.inference_mode():
                vision = model.get_vision_features(pixel_values=pixels([images[i] for i in chunk]))
                for p0 in range(0, len(idx), per):
                    sub = idx[p0:p0 + per]
                    feats = text_feats[sub].repeat(len(chunk), 1, 1)
                    outputs = model(vision_embeds=expand(vision, len(sub)), attention_mask=text_mask[sub].repeat(len(chunk), 1),
                                    text_embeds=types.SimpleNamespace(pooler_output=feats) if wrapped else feats)
                    results = processor.post_process_instance_segmentation(outputs, threshold=.3, mask_threshold=.5,
                                                                           target_sizes=[(height, width)] * (len(chunk) * len(sub)))
                    for j, r in enumerate(results):
                        out[(chunk[j // len(sub)], sub[j % len(sub)])] = (r["scores"].float(), r["masks"].bool())
        sync()
        return out, time.perf_counter() - t

    def loop(words, frames):
        """Today's sam3_app path: processor on the CPU, encode once per frame, then one forward per prompt."""
        out = {}
        sync()
        t = time.perf_counter()
        for i in frames:
            base = processor(images=images[i], return_tensors="pt").to("cuda")
            with torch.inference_mode():
                vision = model.get_vision_features(pixel_values=base["pixel_values"].to(torch.bfloat16))
            for w in words:
                inputs = processor(text=w, original_sizes=base["original_sizes"], return_tensors="pt").to("cuda")
                inputs["vision_embeds"] = vision
                with torch.inference_mode():
                    outputs = model(**inputs)
                r = processor.post_process_instance_segmentation(outputs, threshold=.3, mask_threshold=.5, target_sizes=[(height, width)])[0]
                out[(i, vocab.index(w))] = (r["scores"].float(), r["masks"].bool())
        sync()
        return out, time.perf_counter() - t

    for name, words in sets.items():  # warm-up: kernels, allocator, cudnn autotune; untimed
        batched(words, caps[name][-1], frames=[0, 1])
    loop(BASIC, [0])
    report["batched"], report["loop"], results = {}, {}, {}
    for name, words in sets.items():
        report["batched"][name] = {"prompts": len(words)}
        for cap in caps[name]:
            torch.cuda.reset_peak_memory_stats()
            try:
                out, seconds = batched(words, cap)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                report["batched"][name][str(cap)] = "oom"
                continue
            report["batched"][name][str(cap)] = {"frames_per_forward": max(1, cap // len(words)), "s": round(seconds, 3),
                                                 "s_per_frame": round(seconds / len(images), 4),
                                                 "peak_gpu_mb": round(torch.cuda.max_memory_allocated() / 2 ** 20)}
            results[name] = out
            del out
            torch.cuda.empty_cache()
    for name in loop_sets:
        out, seconds = loop(sets[name], list(range(len(images))))
        report["loop"][name] = {"s": round(seconds, 3), "s_per_frame": round(seconds / len(images), 4)}
        agree = []  # batched (GPU preprocessing, one forward) vs loop (CPU processor, one forward per prompt)
        for key, (score, masks) in out.items():
            other_score, other_masks = results[name][key]
            same_count = int((score >= .4).sum()) == int((other_score >= .4).sum())
            if not same_count or not (score >= .4).any():
                agree.append(float(same_count))
                continue
            a, b = masks[score.argmax()], other_masks[other_score.argmax()]
            agree.append(float((a & b).sum()) / max(1., float((a | b).sum())))
        report["loop"][name]["batched_vs_loop_same_top_mask_share"] = round(float(np.mean(np.array(agree) >= .9)), 4)
        del out
        torch.cuda.empty_cache()

    # recall of today's named objects: IoU of each observed mask with every SAM 3 mask on that keyframe
    torch.backends.cuda.matmul.allow_tf32 = False  # exact pixel counts
    per_mask, masks_of_frame = [], {}
    for name, out in results.items():
        for (i, w), (score, masks) in out.items():
            for rank, m in enumerate(torch.argsort(score, descending=True).tolist()):
                per_mask.append((i, name, vocab[w], float(score[m]), rank))
                masks_of_frame.setdefault(i, []).append(masks[m])
    info = np.array([(i, list(sets).index(n), s, r) for i, n, _, s, r in per_mask], float).reshape(-1, 4)
    frame_obs = {}
    for entity, spec in named.items():
        for i, label in spec["obs"]:
            frame_obs.setdefault(i, []).append((entity, label))
    best = {entity: {} for entity in named}
    for i, obs in frame_obs.items():
        if i not in masks_of_frame:
            continue
        stack = torch.stack(masks_of_frame[i]).flatten(1).float()
        observed = torch.stack([(labels[i] == label).flatten() for _, label in obs]).float()
        inter = observed @ stack.T
        iou = (inter / (observed.sum(1)[:, None] + stack.sum(1)[None] - inter).clamp(min=1)).cpu().numpy()
        rows = info[info[:, 0] == i]  # same order as masks_of_frame[i]
        for o, (entity, _) in enumerate(obs):
            for set_index, set_name in enumerate(sets):
                for mode, keep in (("prod", (rows[:, 2] >= .4) & (rows[:, 3] < 12)), ("loose", rows[:, 2] >= .3)):
                    pick = keep & (rows[:, 1] == set_index)
                    value = float(iou[o][pick].max()) if pick.any() else 0.
                    best[entity][f"{set_name}|{mode}"] = max(best[entity].get(f"{set_name}|{mode}", 0.), value)
    report["best_iou"] = best
    report["detections_ge_0.4"] = {}
    for n, out in results.items():
        counts = report["detections_ge_0.4"].setdefault(n, {})
        for (_, w), (score, _) in out.items():
            counts[vocab[w]] = counts.get(vocab[w], 0) + int((score >= .4).sum())
    saved, masks_out = [], []  # production-rule masks of sets a and b, for later lifting experiments
    for i in range(len(images)):
        for k, (_, n, w, s, r) in enumerate(p for p in per_mask if p[0] == i):
            if n in ("a", "b") and s >= .4 and r < 12:
                saved.append((i, vocab.index(w), s))
                masks_out.append(np.packbits(masks_of_frame[i][k].cpu().numpy()))
    report["peak_gpu_mb"] = round(torch.cuda.max_memory_allocated() / 2 ** 20)
    report["function_wall_s"] = round(time.time() - enter, 2)
    report["enter_at"] = enter
    blob = npz_bytes(frame_index=np.array([r[0] for r in saved], np.int16), word=np.array(vocab), word_index=np.array([r[1] for r in saved], np.int16),
                     score=np.array([r[2] for r in saved], np.float32), masks=np.array(masks_out, np.uint8).reshape(len(masks_out), -1),
                     shape=np.array([height, width]))
    return json.dumps(report, default=str), blob  # plain JSON: nothing in the result needs torch to unpickle


@app.function(image=S2_IMAGE, volumes={"/cache": s2.volume}, timeout=1800, cpu=8, memory=32768, **GPU)
def sam2_today(mp4, labels_npz, depth_npz, video_steps, smoke=False):
    """Today's AMG loop on the A100, AMG with bigger point batches, box prompts, reprojection, and sam2-main propagation."""
    import torch
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    enter = time.time()
    sync = torch.cuda.synchronize
    report = {"gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__)}
    t = time.perf_counter()
    frames = decode_walk(mp4, set(WALK))
    rgb = [frames[i][..., ::-1].copy() for i in WALK]
    report["decode_s"] = round(time.perf_counter() - t, 2)
    labels = np.load(io.BytesIO(labels_npz))["labels"][:12 if smoke else None]
    t = time.perf_counter()
    model = s2.pinned_sam2()
    report["load_s"] = round(time.perf_counter() - t, 2)
    amp = lambda: torch.autocast("cuda", dtype=torch.bfloat16)  # noqa: E731  today's segment_remote settings

    def amg(settings, which, keep):
        generator = SAM2AutomaticMaskGenerator(model, **settings)
        times, counts, found = [], [], {}
        for n in which:
            sync()
            t = time.perf_counter()
            with torch.inference_mode(), amp():
                masks = generator.generate(rgb[n])
            sync()
            times.append(time.perf_counter() - t)
            counts.append(len(masks))
            if n in keep:  # ~70 full-frame masks per frame: keep only the frames compared below
                found[n] = masks
        return times, counts, found

    amg(s2.SETTINGS, [0, 1], ())  # warm-up
    every = list(range(len(WALK)))[: 6 if smoke else None]
    sample = every[::10]
    times, counts, found = amg(s2.SETTINGS, every, set(sample))
    report["amg_today"] = {"settings": s2.SETTINGS, "points_per_batch": 64, "frames": len(times), "s_per_frame": [round(x, 4) for x in times],
                           "masks_per_frame": counts}
    report["amg_variants"] = {}
    for name, extra_settings in (("ppb256", {"points_per_batch": 256}), ("ppb1024", {"points_per_batch": 1024}),
                                 ("side16_ppb256", {"points_per_side": 16, "points_per_batch": 256})):
        vt, _, vfound = amg({**s2.SETTINGS, **extra_settings}, sample, set(sample))
        ious = []
        for n in sample:  # each of today's masks: best IoU with any mask of the variant
            base = torch.from_numpy(np.array([m["segmentation"] for m in found[n]])).cuda().flatten(1).float()
            other = torch.from_numpy(np.array([m["segmentation"] for m in vfound[n]])).cuda().flatten(1).float()
            if len(base) and len(other):
                inter = base @ other.T
                ious.append((inter / (base.sum(1)[:, None] + other.sum(1)[None] - inter)).max(1).values.cpu().numpy())
        report["amg_variants"][name] = {"frames": len(sample), "s_per_frame": round(float(np.mean(vt)), 4),
                                        "today_s_per_frame_same_frames": round(float(np.mean([times[every.index(n)] for n in sample])), 4),
                                        "masks_per_frame": round(float(np.mean([len(vfound[n]) for n in sample])), 1),
                                        "today_masks_best_iou_mean": round(float(np.concatenate(ious).mean()), 4) if ious else None}
    del found

    # box prompts: today's box_masks_remote re-encodes the image for every box; encode once and send all boxes together
    predictor = SAM2ImagePredictor(model)
    box_frames = every[::20][:10]
    loop_s = batch_s = 0.
    same = []
    for n in box_frames:
        ids = [i for i in np.unique(labels[n]) if i and (labels[n] == i).sum() >= MIN_AREA]
        boxes = []
        for i in ids:
            ys, xs = np.nonzero(labels[n] == i)
            boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
        boxes = np.array(boxes, np.float32)
        sync()
        t = time.perf_counter()
        one = []
        for box in boxes:
            with torch.inference_mode(), amp():
                predictor.set_image(rgb[n])
                masks, _, _ = predictor.predict(box=box, multimask_output=False)
            one.append(masks[0] > 0)
        sync()
        loop_s += time.perf_counter() - t
        t = time.perf_counter()
        with torch.inference_mode(), amp():
            predictor.set_image(rgb[n])
            masks, _, _ = predictor.predict(box=boxes, multimask_output=False)
        sync()
        batch_s += time.perf_counter() - t
        masks = masks.reshape(len(boxes), *masks.shape[-2:]) > 0
        same += [((a & b).sum() / max(1, (a | b).sum())) for a, b in zip(one, masks)]
    report["box_prompts"] = {"frames": len(box_frames), "boxes": len(same), "loop_s": round(loop_s, 3), "encode_once_batched_s": round(batch_s, 3),
                             "mask_iou_loop_vs_batched_mean": round(float(np.mean(same)), 4)}
    del predictor

    # reprojection with DROID poses + DA3 posed depth, on DROID's raster
    geo = np.load(io.BytesIO(depth_npz))
    k = [float(v) for v in geo["k"]]
    depth = torch.from_numpy(geo["depth"].astype(np.float32)).cuda()
    c2w = geo["c2w"].astype(np.float64)
    droid_labels = np.stack([to_droid(x) for x in labels])
    gpu_labels = torch.from_numpy(droid_labels.astype(np.int32)).cuda()
    same_view = splat(gpu_labels[0], depth[0], torch.eye(4, device="cuda"), k).cpu().numpy()
    report["splat_identity_check"] = summarize([(*agreement(droid_labels[0], same_view), 0)])
    report["reprojection"], maps = {}, {}
    for spacing in SPACINGS:
        step = spacing // 3
        carried = {}
        sync()
        t = time.perf_counter()
        for key, between in segments(len(labels), step):
            for n in between:
                rel = torch.from_numpy((np.linalg.inv(c2w[n]) @ c2w[key]).astype(np.float32)).cuda()
                carried[n] = splat(gpu_labels[key], depth[key], rel, k)
        sync()
        seconds = time.perf_counter() - t
        rows, still = [], []
        for key, between in segments(len(labels), step):
            for n in between:
                rows.append((*agreement(droid_labels[n], carried[n].cpu().numpy()), 3 * (n - key)))
                still.append((*agreement(droid_labels[n], droid_labels[key]), 3 * (n - key)))
        report["reprojection"][str(spacing)] = {"frames_carried": len(carried), "carry_s": round(seconds, 3),
                                                "agreement": summarize(rows), "identity_no_warp_agreement": summarize(still)}
        if spacing == 15:
            maps["reproject15"] = np.stack([carried[n].cpu().numpy() if n in carried else np.zeros_like(droid_labels[0], np.int32)
                                            for n in range(len(labels))]).astype(np.uint16)
    del depth, gpu_labels
    torch.cuda.empty_cache()
    report["video_main"], video_maps = video_probe(mp4, labels_npz, video_steps, 12 if smoke else None)
    report["peak_gpu_mb"] = round(torch.cuda.max_memory_allocated() / 2 ** 20)
    report["function_wall_s"] = round(time.time() - enter, 2)
    report["enter_at"] = enter
    return json.dumps(report, default=str), npz_bytes(**maps), video_maps


@app.function(image=S2_OLD_IMAGE, volumes={"/cache": s2.volume}, timeout=1200, cpu=8, memory=32768, **GPU)
def sam2_batched(mp4, labels_npz, video_steps, smoke=False):
    enter = time.time()
    report, maps = video_probe(mp4, labels_npz, video_steps, 12 if smoke else None)
    report.update(function_wall_s=round(time.time() - enter, 2), enter_at=enter, sam2_commit=SAM2_BATCHED)
    return json.dumps(report, default=str), maps


# ---------- local side ----------

def prepare(inputs):
    """Today's masks as label maps, DA3 depth + DROID poses on DROID's raster, keyframe PNGs and named observations."""
    import cv2
    sys.path.insert(0, str(Path(__file__).parent))
    from mono_room import unreliable
    inputs.mkdir(parents=True, exist_ok=True)
    if (inputs / "labels.npz").exists():
        return
    labels, overlap = np.zeros((len(WALK), 480, 640), np.uint16), 0
    for n, index in enumerate(WALK):
        folder = MASKS / f"frame-{index:05d}"
        for path in folder.glob("instance-*-mask.png"):
            mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) > 0
            overlap += int((labels[n][mask] > 0).sum())
            labels[n][mask] = int(path.stem.split("-")[1]) + 1
    (inputs / "labels.npz").write_bytes(npz_bytes(labels=labels, frames=np.array(WALK)))
    scale = json.loads((DEPTH / "fuse-metrics.json").read_text())["scale_native_per_mono_metre_median"]
    droid = {k: v for k, v in np.load(DROID / "prediction.npz").items() if k in ("keyframe_source_indices", "keyframe_c2w", "poses_c2w")}
    keys = {int(i): n for n, i in enumerate(droid["keyframe_source_indices"])}
    poses, depths = [], []
    for index in WALK:  # mono_room.load: keyframe poses for DROID keyframes, filler poses otherwise; the object map's edge rule
        poses.append(droid["keyframe_c2w"][keys[index]] if index in keys else droid["poses_c2w"][index])
        mono = np.load(DEPTH / "mono" / f"{index:05d}.npz")
        depth = np.where(mono["mask"], mono["depth"], 0).astype(np.float32)
        depths.append(np.where(unreliable(depth, None, None, .03), 0, scale * depth).astype(np.float16))
    clip = json.loads((CLIP / "clip.json").read_text())
    (inputs / "geometry.npz").write_bytes(npz_bytes(depth=np.stack(depths), c2w=np.stack(poses).astype(np.float64), k=np.array(droid_k(clip["K"])),
                                                    frames=np.array(WALK), scale=np.array(scale)))
    frames = decode_local(set(E2_KEYS))
    for index in E2_KEYS:
        cv2.imwrite(str(inputs / f"e2-{index:05d}.png"), frames[index])
    names = json.loads((NAMES / "names.json").read_text())
    entities = {e["entityId"]: e for e in json.loads((NAMES / "object-map.json").read_text())["entities"]}
    named = {}
    for entity, spec in names.items():
        observed = [o.split(":") for o in entities[entity]["observations"]]
        if spec["status"] not in ("clear", "partial") or not any(226 <= int(f) <= 898 for _, f, _ in observed):
            continue
        assert all(kind == "object" for kind, _, _ in observed), "named observations are all SAM 2 object masks"
        named[entity] = {"category": spec["category"], "status": spec["status"],
                         "obs": [[E2_KEYS.index(int(f)), int(i) + 1] for _, f, i in observed if int(f) in E2_KEYS]}
    (inputs / "named.json").write_text(json.dumps(named, indent=1))
    (inputs / "prepare.json").write_text(json.dumps({"mask_overlap_px": overlap, "walk_frames": len(WALK), "e2_keyframes": E2_KEYS,
                                                     "depth_scale_native_per_mono": scale, "named_walk_objects": len(named),
                                                     "named_on_e2_keyframes": sum(bool(v["obs"]) for v in named.values())}, indent=1))


def decode_local(wanted):
    import cv2
    cap, out = cv2.VideoCapture(str(CLIP / "source-rgb.mp4")), {}
    for index in range(max(wanted) + 1):
        ok, bgr = cap.read()
        assert ok
        if index in wanted:
            out[index] = bgr
    cap.release()
    return out


@app.local_entrypoint()
def main(out: str, only: str = "sam3,sam2,sam2old", smoke: bool = False, inputs: str = str(PHASE2 / "runs/m3-exp-e2-e6-segment-inputs")):
    out, inputs = Path(out), Path(inputs)
    prepare(inputs)
    mp4 = (CLIP / "source-rgb.mp4").read_bytes()
    labels_npz = (inputs / "labels.npz").read_bytes()
    named = {k: v for k, v in json.loads((inputs / "named.json").read_text()).items()}
    categories = sorted({v["category"] for v in named.values()})
    sets = {"a": BASIC, "b": EHS, "c": categories}
    caps = {"a": [2, 16, 90], "b": [20, 40, 80, 160], "c": [len(categories), 2 * len(categories)]}
    keys = E2_KEYS[:4] if smoke else E2_KEYS
    pngs = [(inputs / f"e2-{i:05d}.png").read_bytes() for i in keys]
    all_labels = np.load(io.BytesIO(labels_npz))["labels"]
    e2_labels = npz_bytes(labels=all_labels[[WALK.index(i) for i in keys]])
    if smoke:
        sets, caps = {"a": BASIC, "b": EHS[:6]}, {"a": [4], "b": [12]}
        named = {k: {**v, "obs": [o for o in v["obs"] if o[0] < 4]} for k, v in named.items()}
    calls, started = {}, {}
    wanted = set(only.split(","))
    if "sam3" in wanted:
        started["sam3"] = time.time()
        calls["sam3"] = sam3_probe.spawn(pngs, e2_labels, named, sets, caps, ["a", "b"])
    if "sam2" in wanted:
        started["sam2"] = time.time()
        calls["sam2"] = sam2_today.spawn(mp4, labels_npz, (inputs / "geometry.npz").read_bytes(), [2] if smoke else [5], smoke)
    if "sam2old" in wanted:
        started["sam2old"] = time.time()
        calls["sam2old"] = sam2_batched.spawn(mp4, labels_npz, [2] if smoke else [2, 5, 10], smoke)
    out.mkdir(parents=True, exist_ok=True)
    for name, call in calls.items():
        try:
            result = call.get()
        except Exception as error:  # report, keep the other probes' results
            (out / f"{name}-error.txt").write_text(repr(error)[:4000])
            print(name, "failed:", repr(error)[:400])
            continue
        report = json.loads(result[0])
        report["client_call_wall_s"] = round(time.time() - started[name], 1)
        report["cold_start_s_approx"] = round(report["enter_at"] - started[name], 1)  # two clocks
        (out / f"{name}.json").write_text(json.dumps(report, indent=1, default=str))
        for n, blob in enumerate(result[1:]):
            (out / f"{name}-maps-{n}.npz").write_bytes(blob)
        print(name, "done in", report["client_call_wall_s"], "s")


RESERVED = {"sam3": (4, 32), "sam2": (8, 32), "sam2old": (8, 32)}  # (cpu cores, memory GiB) requested per function above


def verdicts(out):
    """Pass/fail and headline numbers from a run folder's JSON (M = measured, E = derived from measured x list price)."""
    out = Path(out)
    load = lambda name: json.loads((out / f"{name}.json").read_text()) if (out / f"{name}.json").exists() else None  # noqa: E731
    sam3, today, old = load("sam3"), load("sam2"), load("sam2old")
    result = {"usd": {}, "seconds": {}}
    for name, report in (("sam3", sam3), ("sam2", today), ("sam2old", old)):
        if report:
            cores, gib = RESERVED[name]
            per_s = USD_PER_S["A100-80GB"] + cores * USD_PER_S["cpu_core"] + gib * USD_PER_S["mem_gib"]
            result["seconds"][name] = {"client_call_wall_s_M": report["client_call_wall_s"], "function_wall_s_M": report["function_wall_s"],
                                       "cold_start_s_approx_M": report["cold_start_s_approx"]}
            result["usd"][name] = round((report["client_call_wall_s"] + 2) * per_s, 3)  # + 2 s scaledown window
    result["usd"]["total_E"] = round(sum(result["usd"].values()), 3)
    if sam3:
        named = json.loads((Path(PHASE2 / "runs/m3-exp-e2-e6-segment-inputs") / "named.json").read_text())
        best = sam3["best_iou"]
        on_keys = [e for e in named if named[e]["obs"]]

        def recall(sets, mode, pool):
            found = [e for e in pool if max((best.get(e, {}).get(f"{x}|{mode}", 0.) for x in sets), default=0.) >= .5]
            return {"found": len(found), "of": len(pool), "recall": round(len(found) / max(1, len(pool)), 4)}
        result["e2_recall"] = {f"{'+'.join(sets)}|{mode}|{pool_name}": recall(sets, mode, pool)
                               for sets in (["a"], ["b"], ["a", "b"], ["c"], ["a", "b", "c"]) if all(x in sam3["batched"] for x in sets)
                               for mode in ("prod", "loose") for pool_name, pool in (("on_keyframes", on_keys), ("walk_named", list(named)))}
        result["e2_missed_by_ab_prod"] = sorted(named[e]["category"] for e in on_keys
                                                if max(best.get(e, {}).get(f"{x}|prod", 0.) for x in "ab") < .5)
        speed = {}
        for name, runs in sam3["batched"].items():
            timed = {cap: r for cap, r in runs.items() if isinstance(r, dict)}
            if timed:
                cap, fastest = min(timed.items(), key=lambda kv: kv[1]["s_per_frame"])
                speed[name] = {"batched_best_s_per_frame_M": fastest["s_per_frame"], "pairs_per_forward": int(cap),
                               "all_caps": {c: r["s_per_frame"] for c, r in timed.items()}}
            if name in sam3["loop"]:
                speed[name]["todays_loop_s_per_frame_M"] = sam3["loop"][name]["s_per_frame"]
        result["e2_speed"] = speed
        a, b = speed.get("a", {}).get("batched_best_s_per_frame_M"), speed.get("b", {}).get("batched_best_s_per_frame_M")
        r = result["e2_recall"].get("b|prod|on_keyframes", {}).get("recall")
        result["e2_pass"] = {"a<=0.15s": a is not None and a <= .15, "b<=0.7s": b is not None and b <= .7, "recall>=0.8 (b, production rule)": r is not None and r >= .8}
    if today:
        per_frame = today["amg_today"]["s_per_frame"]
        result["e6_today_amg_a100_s_M"] = round(sum(per_frame), 2)
        result["e6_today_amg_a100_s_per_frame_M"] = round(float(np.mean(per_frame)), 4)
        rows = {}
        for spacing in SPACINGS[1:]:
            step = spacing // 3
            keys = [k for k, _ in segments(len(per_frame), step)]
            amg_keys = sum(per_frame[k] for k in keys)
            row = {"keyframes": len(keys), "keyframe_amg_s_M": round(amg_keys, 2)}
            reproject = today["reprojection"][str(spacing)]
            row["reproject"] = {"carry_s_M": reproject["carry_s"], "total_s_M": round(amg_keys + reproject["carry_s"], 2),
                                "speedup_vs_today_M": round(sum(per_frame) / (amg_keys + reproject["carry_s"]), 2), **reproject["agreement"],
                                "no_warp_mean_best_iou": reproject["identity_no_warp_agreement"]["mean_best_iou"]}
            for label, report in (("video_batched_objects", old), ("video_sam2_main", today.get("video_main"))):
                got = report and report["steps"].get(str(step))
                if got:
                    carry = got["times"]["add_masks_s"] + got["times"]["propagate_s"]
                    row[label] = {"carry_s_M": round(carry, 2), "init_state_s_M": round(got["times"]["init_state_s"], 2),
                                  "total_s_M": round(amg_keys + carry + got["times"]["init_state_s"], 2),
                                  "speedup_vs_today_M": round(sum(per_frame) / (amg_keys + carry + got["times"]["init_state_s"]), 2),
                                  **got["agreement"]}
            rows[str(spacing)] = row
        result["e6"] = rows
        result["e6_ceiling_reproject_3_frames"] = today["reprojection"]["3"]["agreement"]
        result["e6_ceiling_no_warp_3_frames"] = today["reprojection"]["3"]["identity_no_warp_agreement"]
        result["amg_variants"] = today["amg_variants"]
        result["box_prompts"] = today["box_prompts"]
    return result


if __name__ == "__main__":
    if sys.argv[1:2] == ["--verdicts"]:
        print(json.dumps(verdicts(sys.argv[2]), indent=1))
    else:
        assert sys.argv[1:] == ["--self-check"], "usage: python modal_apps/segment_fast_probe.py --self-check | --verdicts RUN_DIR"
        self_check()
