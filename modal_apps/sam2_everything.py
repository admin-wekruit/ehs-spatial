"""Class-agnostic masks for video keyframes: SAM 2.1 automatic mask generation (Apache-2.0 weights), no text prompt.

Prompted segmentation only finds what someone thought to name. ConceptGraphs' entry step is the opposite: segment
everything, lift it into the map, name it afterwards. This writes the folder layout of discover_video_keyframes.py
under the label "object", so the object map, replay and report consume it unchanged. Masks are made disjoint (a
smaller mask is cut out of any larger one) and masks mostly covered by an already prompted mask are dropped, so the
class-agnostic entities are exactly the remainder the prompts missed. One GPU call, retries 0.

  python modal_apps/sam2_everything.py --video V --frames 7 28 ... --known MASK_ROOT --output NEW_DIR
  python modal_apps/sam2_everything.py --self-check
"""
import argparse
import io
import json
from pathlib import Path
import sys
import time

import modal
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from moge3_app import volume  # noqa: E402  the project's weight cache

MODEL, REVISION = "facebook/sam2.1-hiera-large", "665f8e2ad61cf5f53d65644ff27c8ee525124610"  # HF revision pin for new runs
SETTINGS = {"points_per_side": 32, "pred_iou_thresh": .8, "stability_score_thresh": .92, "min_mask_region_area": 200}
MIN_AREA, KEEP_SHARE, COVERED, BACKGROUND_SHARE = 300, .25, .5, .08
STUFF = ("floor", "wall", "ceiling")

image = (modal.Image.debian_slim(python_version="3.11").apt_install("git", "libgl1", "libglib2.0-0")
         .pip_install("torch", "torchvision", "opencv-python-headless", "huggingface_hub")
         .env({"SAM2_BUILD_CUDA": "0", "HF_HOME": "/cache/huggingface"})
         .pip_install("git+https://github.com/facebookresearch/sam2.git")
         .add_local_python_source("moge3_app"))
app = modal.App("panoptes-sam2-everything-once")


def pinned_sam2():
    """SAM 2.1 at REVISION: sam2's from_pretrained always fetches main, so fetch the pinned checkpoint and build it as build_sam2_hf does."""
    from huggingface_hub import hf_hub_download
    from sam2.build_sam import HF_MODEL_ID_TO_FILENAMES, build_sam2
    config, checkpoint = HF_MODEL_ID_TO_FILENAMES[MODEL]
    return build_sam2(config_file=config, ckpt_path=hf_hub_download(MODEL, checkpoint, revision=REVISION))


@app.function(image=image, gpu="L4", volumes={"/cache": volume}, timeout=900, retries=0, max_containers=1)
def segment_remote(frames, settings):
    """frames: [(source_index, png_bytes)] -> [(source_index, npz_bytes)] of bit-packed masks with SAM's own scores."""
    import cv2
    import torch
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    generator = SAM2AutomaticMaskGenerator(pinned_sam2(), **settings)
    results = []
    for index, png in frames:
        rgb = cv2.cvtColor(cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            found = generator.generate(rgb)
        buffer = io.BytesIO()
        np.savez_compressed(buffer, shape=np.array(rgb.shape[:2]), masks=np.packbits(np.array([m["segmentation"] for m in found], bool).reshape(len(found), -1), axis=1),
                            predicted_iou=np.array([m["predicted_iou"] for m in found], np.float32), stability=np.array([m["stability_score"] for m in found], np.float32))
        results.append((index, buffer.getvalue()))
    return results


@app.function(image=image, gpu="L4", volumes={"/cache": volume}, timeout=900, retries=0, max_containers=1)
def box_masks_remote(items):
    """items: [(key, image bytes, [x0, y0, x1, y1])] -> [(key, npz of the one mask SAM gives for that box, its score)]."""
    import cv2
    import torch
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    predictor = SAM2ImagePredictor(pinned_sam2())
    results = []
    for key, data, box in items:
        rgb = cv2.cvtColor(cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            predictor.set_image(rgb)
            masks, scores, _ = predictor.predict(box=np.asarray(box, np.float32), multimask_output=False)
        buffer = io.BytesIO()
        np.savez_compressed(buffer, mask=masks[0].astype(bool), score=float(scores[0]))
        results.append((key, buffer.getvalue()))
    return results


def box_masks(requests, work_width=1280):
    """requests: {key: (source BGR image, box in source pixels)} -> {key: (mask in source pixels, score)}. One GPU call for all of them."""
    import cv2
    items, shapes = [], {}
    for key, (image, box) in requests.items():
        scale = min(1., work_width / image.shape[1])
        small = image if scale == 1 else cv2.resize(image, (work_width, round(image.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        items.append((key, cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes(), [float(v) * scale for v in box]))
        shapes[key] = image.shape[:2]
    with app.run():
        answers = box_masks_remote.remote(items)
    out = {}
    for key, data in answers:
        found = np.load(io.BytesIO(data))
        out[key] = (cv2.resize(found["mask"].astype(np.uint8), shapes[key][::-1], interpolation=cv2.INTER_NEAREST) > 0, float(found["score"]))
    return out


def disjoint(masks, things, stuff):
    """Masks that overlap nothing: smallest first, each keeps the pixels no smaller mask took.

    Dropped: what a prompted object mask already covers, the wall or floor itself, slivers, and a container left hollow by its parts.
    """
    taken, kept = np.zeros(masks[0].shape, bool) if len(masks) else None, []
    sliver = MIN_AREA * (masks[0].size / (640 * 480) if len(masks) else 1)  # MIN_AREA is meant at 640x480: the same share of a larger frame
    for n in sorted(range(len(masks)), key=lambda n: masks[n].sum()):
        area = masks[n].sum()
        if area < sliver or (masks[n] & things).sum() >= COVERED * area:
            continue
        if area >= BACKGROUND_SHARE * masks[n].size and (masks[n] & stuff).sum() >= COVERED * area:
            continue
        left = masks[n] & ~taken
        if left.sum() >= max(sliver, KEEP_SHARE * area):
            kept.append((n, left))
            taken |= left
    return kept


def known_masks(root, index, shape):
    import cv2
    things, stuff = np.zeros(shape, bool), np.zeros(shape, bool)
    for path in (root.glob(f"*/frame-{index:05d}/instance-*-mask.png") if root else []):
        pixels = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        mask = pixels.reshape(*pixels.shape[:2], -1).max(-1) > 0
        (stuff if path.parents[1].name.rsplit("-", 1)[0] in STUFF else things).__ior__(mask)
    return things, stuff


def run(args):
    import cv2
    args.output.mkdir(parents=True, exist_ok=False)
    wanted, frames, cap = sorted(set(args.frames)), {}, cv2.VideoCapture(str(args.video))
    for index in range(wanted[-1] + 1):
        ok, bgr = cap.read()
        assert ok, "selected frame is absent from the video"
        if index in wanted:
            frames[index] = bgr
    cap.release()
    state = {"status": "gpu_running", "model": MODEL, "settings": SETTINGS, "frames": wanted, "gpu": "L4", "timeout_s_per_48_frames": 900, "retries": 0,
             "video": str(args.video), "known_masks": str(args.known) if args.known else None}
    with app.run():
        state["app_id"] = app.app_id
        (args.output / f"segment-{int(time.time())}.json").write_text(json.dumps(state, indent=1))
        started = time.time()
        # SAM works at 1024 px inside: a 1920 px frame is sent smaller and its masks are brought back to source pixels
        small = lambda f: f if not args.work_width or f.shape[1] <= args.work_width else cv2.resize(f, (args.work_width, round(f.shape[0] * args.work_width / f.shape[1])), interpolation=cv2.INTER_AREA)
        packed = [(i, cv2.imencode(".jpg" if args.work_width else ".png", small(f), [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes()) for i, f in frames.items()]
        chunks = [packed[n:n + 48] for n in range(0, len(packed), 48)]  # bounded request size; one warm container takes them in turn
        raw = {index: data for chunk in segment_remote.map(chunks, kwargs={"settings": SETTINGS}) for index, data in chunk}
        state.update(status="complete", wall_seconds=time.time() - started)
    total = kept_total = 0
    for index, data in raw.items():
        (args.output / "raw").mkdir(exist_ok=True)
        (args.output / "raw" / f"{index:05d}.npz").write_bytes(data)  # the provider's answer, before any local filtering
        found = np.load(io.BytesIO(data))
        shape = tuple(found["shape"])
        masks = np.unpackbits(found["masks"], axis=1)[:, :shape[0] * shape[1]].reshape(-1, *shape).astype(bool)
        if shape != frames[index].shape[:2]:
            masks = np.array([cv2.resize(m.astype(np.uint8), frames[index].shape[1::-1], interpolation=cv2.INTER_NEAREST).astype(bool) for m in masks])
            shape = frames[index].shape[:2]
        kept = disjoint(list(masks), *known_masks(args.known, index, shape))
        folder = args.output / "object-a" / f"frame-{index:05d}"
        folder.mkdir(parents=True)
        cv2.imwrite(str(folder / f"frame-{index}.png"), frames[index])
        for ordinal, (n, mask) in enumerate(kept):
            cv2.imwrite(str(folder / f"instance-{ordinal}-mask.png"), mask.astype(np.uint8) * 255)
        (folder / "instances.json").write_text(json.dumps({"instances": [{"instance_index": o, "label": "object", "sam_mask": int(n), "area": int(m.sum()),
            "predicted_iou": float(found["predicted_iou"][n]), "stability": float(found["stability"][n])} for o, (n, m) in enumerate(kept)]}, indent=1))
        total += len(masks); kept_total += len(kept)
    state.update(masks_from_sam=total, masks_kept=kept_total, rule=f"disjoint smallest-first; dropped if <{MIN_AREA}px, >={COVERED:.0%} inside a prompted object mask, "
                 f"background itself (>={BACKGROUND_SHARE:.0%} of the frame and mostly wall/floor), or left with <{KEEP_SHARE:.0%} of itself")
    (args.output / "segment.json").write_text(json.dumps(state, indent=1))
    print(json.dumps({k: state[k] for k in ("frames", "masks_from_sam", "masks_kept", "wall_seconds")}))


def self_check():
    shape = (100, 100)
    def box(y0, y1, x0, x1):
        m = np.zeros(shape, bool); m[y0:y1, x0:x1] = True; return m
    desk, book, wall, named, shell, part = box(40, 100, 0, 100), box(50, 70, 10, 40), box(0, 40, 0, 100), box(45, 65, 60, 90), box(0, 30, 0, 30), box(1, 29, 1, 29)
    kept = dict(disjoint([desk, book, wall, named, shell, part], things=box(45, 65, 60, 90), stuff=box(0, 40, 0, 100) & ~shell))
    assert (kept[1] == book).all() and not (kept[0] & book).any() and kept[0].sum() == desk.sum() - book.sum(), "a contained object is cut out of its container and kept whole"
    assert 2 not in kept and 3 not in kept, "the wall itself and an already prompted object are not class-agnostic entities"
    assert 5 in kept and 4 not in kept, "a container left hollow by its part is dropped"
    print("sam2 everything check passed: disjoint smallest-first, prompted objects and background dropped, hollow containers dropped")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--video", type=Path)
    parser.add_argument("--frames", type=int, nargs="+")
    parser.add_argument("--work-width", type=int, help="send frames wider than this at this width (JPEG); masks come back in source pixels")
    parser.add_argument("--known", type=Path, help="mask root of prompted runs; what they already cover is left to them")
    parser.add_argument("--output", type=Path)
    a = parser.parse_args()
    self_check() if a.self_check else run(a)
