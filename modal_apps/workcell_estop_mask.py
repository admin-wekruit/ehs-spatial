"""SAM 3 text-prompt masks of the e-stop ("emergency stop button") on original photos, for scripts/workcell_estop_scale.py --mask.

Same pinned model, weights volume and image as modal_apps/workcell_mask_transfer.py (one ephemeral L4, offline). An e-stop is
~120 px wide in a 12 MP photo, ~30 px at SAM 3's 1008 px input, too small to be found on the whole photo: the photo is
segmented in overlapping square tiles (--tile px, 25 % overlap; 0 = whole photo) and each instance is pasted back at photo
resolution. The e-stop is rarely SAM's top instance (signal lamps and red labels score higher), so every instance above
--threshold is kept, duplicates (box IoU > .7 across tiles and prompts) dropped, and the best --keep written in score order as
OUT/<photo stem>/NN.png (255 = inside); scripts/workcell_estop_scale.py --mask ID=OUT/<stem> takes the first one that has the
e-stop's structure. results.json lists every instance. On-prem: python scripts/onprem/run_stage.py --weights DIR
modal_apps/workcell_estop_mask.py ... (/opt/sam3).

modal run modal_apps/workcell_estop_mask.py --photos-dir DIR --photo FILE,... --out NEW_DIR [--text "emergency stop button|red button"] [--tile 1008] [--threshold .1] [--keep 16]
"""
import json
from pathlib import Path
import time

import modal

REPO = Path(__file__).resolve().parents[1]
RATE = .000222 + 4 * .0000131 + 16 * .00000222  # L4 + 4 CPU + 16 GiB list rate (USD/s); not an invoice

app = modal.App('workcell-estop-mask')
image = (modal.Image.debian_slim(python_version='3.11')
         .pip_install('torch==2.14.0', 'torchvision', 'transformers==5.17.0', 'accelerate', 'pillow', 'numpy<2.3', 'opencv-python-headless==4.10.0.84')
         .env({'HF_HUB_OFFLINE': '1'})
         .add_local_file(REPO / 'scripts/workcell_sam_worker.py', '/sam/workcell_sam_worker.py'))


def tiles(width, height, size):
    """Top-left corners of overlapping size x size tiles (25 % overlap) covering the photo; one tile when size is 0."""
    if not size or size >= max(width, height):
        return [(0, 0, width, height)]
    stride = int(size * .75)
    xs = sorted({min(x, width - size) for x in range(0, width, stride)}); ys = sorted({min(y, height - size) for y in range(0, height, stride)})
    return [(x, y, size, size) for y in ys for x in xs]


def box_iou(a, b):
    w, h = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    inter = max(w, 0) * max(h, 0); area = lambda c: (c[2] - c[0]) * (c[3] - c[1])
    return inter / (area(a) + area(b) - inter)


@app.function(image=image, gpu='L4', cpu=4, memory=16 * 1024, volumes={'/v/sam3': modal.Volume.from_name('sam3-hf-cache')},
              timeout=900, retries=0, min_containers=0)
def segment(photos: dict, text: str, tile: int, threshold: float, keep: int) -> dict:
    import sys
    sys.path.insert(0, '/sam')
    import cv2
    import numpy as np
    from PIL import Image
    import torch
    import workcell_sam_worker as worker  # the pinned model id and revision
    from transformers import Sam3Model, Sam3Processor
    start = time.monotonic()
    cache = '/v/sam3/huggingface/hub'
    processor = Sam3Processor.from_pretrained(worker.MODEL_ID, revision=worker.REVISION, cache_dir=cache)
    model = Sam3Model.from_pretrained(worker.MODEL_ID, revision=worker.REVISION, cache_dir=cache, torch_dtype=torch.bfloat16).to('cuda').eval()
    out = {}
    for name, data in photos.items():
        bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)  # EXIF-oriented, as cv2.imread in the scale CLI
        H, W = bgr.shape[:2]; rows = []
        for x0, y0, w, h in tiles(W, H, tile):
            photo = Image.fromarray(np.ascontiguousarray(bgr[y0:y0 + h, x0:x0 + w, ::-1]))
            base = processor(images=photo, return_tensors='pt').to('cuda')
            with torch.inference_mode():
                vision = model.get_vision_features(pixel_values=base['pixel_values'])
            for prompt in text.split('|'):  # alternatives: every instance competes on its score
                with torch.inference_mode():
                    kwargs = processor(text=prompt, original_sizes=base['original_sizes'], return_tensors='pt').to('cuda')
                    output = model(vision_embeds=vision, **kwargs)
                result = processor.post_process_instance_segmentation(output, threshold=threshold, mask_threshold=.5, target_sizes=[(h, w)])[0]
                masks, scores = result.get('masks'), result.get('scores')
                if masks is None or not len(masks):
                    continue
                scores = scores.float().cpu().numpy(); masks = (masks.float() > .5).cpu().numpy()
                for i in np.argsort(-scores):
                    ys, xs = np.nonzero(masks[i])
                    if not len(ys):
                        continue
                    rows.append(dict(prompt=prompt, score=float(scores[i]), areaPx=int(len(ys)), tile=[x0, y0, w, h],
                                     bbox=[int(xs.min()) + x0, int(ys.min()) + y0, int(xs.max()) + 1 + x0, int(ys.max()) + 1 + y0],
                                     crop=masks[i][ys.min():ys.max() + 1, xs.min():xs.max() + 1]))
        rows.sort(key=lambda r: -r['score']); kept = []
        for r in rows:
            if len(kept) < keep and all(box_iou(r['bbox'], k['bbox']) <= .7 for k in kept):
                full = np.zeros((H, W), np.uint8); x0, y0, x1, y1 = r['bbox']; full[y0:y1, x0:x1] = r['crop']
                r['png'] = cv2.imencode('.png', full * 255)[1].tobytes(); kept.append(r)
        for r in rows:
            r.pop('crop')
        out[name] = dict(instances=rows, shape=[H, W])
    return dict(photos=out, containerSeconds=time.monotonic() - start, peakAllocatedGiB=torch.cuda.max_memory_allocated() / 2**30)


@app.local_entrypoint()
def main(photos_dir: str, photo: str, out: str, text: str = 'emergency stop button|red button', tile: int = 1008, threshold: float = .1,
         keep: int = 16):
    destination = Path(out)
    if destination.exists():
        raise ValueError('Choose a fresh output directory')
    photos = {name: (Path(photos_dir) / name).read_bytes() for name in photo.split(',')}
    start = time.monotonic()
    result = segment.remote(photos, text, tile, threshold, keep)
    destination.mkdir(parents=True)
    for name, r in result['photos'].items():
        folder = destination / Path(name).stem; folder.mkdir(); k = 0
        for i in r['instances']:
            if 'png' in i:
                i['mask'] = f'{folder.name}/{k:02d}.png'; (destination / i['mask']).write_bytes(i.pop('png')); k += 1
        print(f"{name}: {len(r['instances'])} instance(s), {k} candidate mask(s)", [(i['prompt'], round(i['score'], 3), i['bbox']) for i in r['instances'] if 'mask' in i][:4])
    result.update(text=text, tile=tile, threshold=threshold, keep=keep)
    (destination / 'results.json').write_text(json.dumps(result, indent=1) + '\n')
    ledger = {'mode': 'ephemeral modal run', 'hardware': '1 L4, 4 CPU, 16 GiB', 'functionSeconds': result['containerSeconds'],
              'callSeconds': time.monotonic() - start, 'estimateUsd': RATE * result['containerSeconds'], 'actualBilledUsd': None,
              'rateSource': 'https://modal.com/pricing'}
    (destination / 'spend-ledger.json').write_text(json.dumps(ledger, indent=2) + '\n')
    print(json.dumps(ledger))
