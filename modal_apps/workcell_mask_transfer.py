"""Transferred masks: SAM 3 box prompts where a report photo shows an object's model but has no mask for it (one ephemeral L4).

scripts/workcell_checks/transfer.py (run through modal_apps/workcell_view_checks.py) finds the candidates: the model's visible
silhouette in a photo without the object's mask, its padded box and polygons. Here SAM 3 (the pinned model of
scripts/workcell_sam_worker.py, weights from the sam3-hf-cache volume, offline) segments each box on the original photo; the
best-scoring mask that lies mostly inside the box is kept. Its IoU with the silhouette is a held-out check of the placement
(fitted only on the masked photos); masks with score >= MIN_SCORE and IoU >= MIN_IOU are written as extra masks for
modal_apps/workcell_shape_check.py --extra-masks ({entityId: {photoIndex0: polygons}}, original pixel coordinates).

modal run modal_apps/workcell_mask_transfer.py --candidates TRANSFER/results.json --photos-dir DIR --photo IMAGE_ID=FILE,... --out NEW_DIR
"""
import json
from pathlib import Path
import time

import modal

REPO = Path(__file__).resolve().parents[1]
RATE = .000222 + 4 * .0000131 + 16 * .00000222  # L4 + 4 CPU + 16 GiB list rate (USD/s); not an invoice
MIN_SCORE, MIN_IOU, INSIDE = .5, .5, .5

app = modal.App('workcell-mask-transfer')
image = (modal.Image.debian_slim(python_version='3.11')
         .pip_install('torch==2.14.0', 'torchvision', 'transformers==5.17.0', 'accelerate', 'pillow', 'numpy<2.3', 'opencv-python-headless==4.10.0.84')
         .env({'HF_HUB_OFFLINE': '1'})
         .add_local_file(REPO / 'scripts/workcell_sam_worker.py', '/sam/workcell_sam_worker.py')
         .add_local_file(REPO / 'scripts/workcell_shape_check.py', '/sam/shape_core.py')
         .add_local_file(REPO / 'scripts/workcell_checks/transfer.py', '/sam/transfer.py'))


@app.function(image=image, gpu='L4', cpu=4, memory=16 * 1024, volumes={'/v/sam3': modal.Volume.from_name('sam3-hf-cache')},
              timeout=1200, retries=0, min_containers=0)
def segment(photos: dict, candidates: list, min_score: float, min_iou: float) -> dict:
    import sys
    sys.path.insert(0, '/sam')
    import cv2
    import numpy as np
    from PIL import Image
    import torch
    import shape_core as wsc
    import transfer
    import workcell_sam_worker as worker  # the pinned model id and revision
    from transformers import Sam3Model, Sam3Processor
    start = time.monotonic()
    cache = '/v/sam3/huggingface/hub'
    processor = Sam3Processor.from_pretrained(worker.MODEL_ID, revision=worker.REVISION, cache_dir=cache)
    model = Sam3Model.from_pretrained(worker.MODEL_ID, revision=worker.REVISION, cache_dir=cache, torch_dtype=torch.bfloat16).to('cuda').eval()
    load_seconds = time.monotonic() - start
    rows, tiles = [], []
    for image_id in dict.fromkeys(c['imageId'] for c in candidates):
        bgr = cv2.imdecode(np.frombuffer(photos[image_id], np.uint8), cv2.IMREAD_COLOR)  # EXIF-oriented, as load_report reads it
        photo = Image.fromarray(np.ascontiguousarray(bgr[..., ::-1]))
        base = processor(images=photo, return_tensors='pt').to('cuda')
        with torch.inference_mode():
            vision = model.get_vision_features(pixel_values=base['pixel_values'])
        for c in (c for c in candidates if c['imageId'] == image_id):
            x0, y0, x1, y1 = c['bbox']
            kwargs = processor(input_boxes=[[[float(v) for v in c['bbox']]]], original_sizes=base['original_sizes'], return_tensors='pt').to('cuda')
            kwargs['input_boxes'] = kwargs['input_boxes'].to(model.dtype)
            with torch.inference_mode():
                output = model(vision_embeds=vision, **kwargs)
            result = processor.post_process_instance_segmentation(output, threshold=.4, mask_threshold=.5, target_sizes=[(photo.height, photo.width)])[0]
            sil = wsc.polygon_mask(c['polygons'], bgr.shape[:2])
            masks, scores = result.get('masks'), result.get('scores')
            masks = masks.float() if masks is not None else None
            row = {k: c[k] for k in ('entityId', 'label', 'photoIndex0', 'imageId', 'areaPx', 'visibleFraction', 'bbox')}
            row.update(instances=0, samScore=0.0, heldOutIoU=0.0, accepted=False, polygons=[])
            mask = None
            if masks is not None and len(masks):
                scores = scores.float().cpu().numpy(); row['instances'] = len(scores)
                inside = [float((m[y0:y1, x0:x1] > .5).sum() / max(1, (m > .5).sum())) for m in masks]
                # SAM 3 box prompts are exemplars: other instances of the same kind come back too; keep the boxed one
                order = [i for i in np.argsort(-scores) if inside[i] >= INSIDE] or list(np.argsort(-scores))
                i = int(order[0]); mask = (masks[i] > .5).cpu().numpy()
                iou = float((mask & sil).sum() / max(1, (mask | sil).sum()))
                row.update(samScore=float(scores[i]), heldOutIoU=iou, insideBox=inside[i], samAreaPx=int(mask.sum()),
                           otherInstances=[dict(score=float(scores[j]), insideBox=inside[j]) for j in np.argsort(-scores) if j != i][:5])
                row['accepted'] = row['samScore'] >= min_score and iou >= min_iou
                if row['accepted']:
                    row['polygons'] = transfer.polygons_of(mask, 1, 1.5)
            rows.append(row)
            # contact-sheet tile: silhouette (cyan) and SAM mask (magenta, if any) over the box
            w, h = x1 - x0, y1 - y0; cx0, cy0 = max(0, int(x0 - .1 * w)), max(0, int(y0 - .1 * h))
            crop = bgr[cy0:min(bgr.shape[0], int(y1 + .1 * h)), cx0:min(bgr.shape[1], int(x1 + .1 * w))].copy()
            thick = max(2, max(crop.shape[:2]) // 150)
            for m, color in ((sil, (255, 255, 0)), (mask, (255, 0, 255))):
                if m is not None:
                    cs, _ = cv2.findContours(m[cy0:cy0 + crop.shape[0], cx0:cx0 + crop.shape[1]].astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(crop, cs, -1, color, thick)
            f = 300 / max(crop.shape[:2]); crop = cv2.resize(crop, (max(1, round(crop.shape[1] * f)), max(1, round(crop.shape[0] * f))), interpolation=cv2.INTER_AREA)
            tile = np.full((330, 300, 3), 255, np.uint8); tile[:crop.shape[0], :crop.shape[1]] = crop
            text = f"{c['entityId'][:8]} P{c['photoIndex0'] + 1} s{row['samScore']:.2f} IoU{row['heldOutIoU']:.2f} {'ACC' if row['accepted'] else 'rej'}"
            cv2.putText(tile, text, (4, 322), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 120, 0) if row['accepted'] else (0, 0, 200), 1, cv2.LINE_AA)
            tiles.append(tile)
            del masks, output
        del vision
    cols = min(6, len(tiles)) or 1
    tiles += [np.full((330, 300, 3), 255, np.uint8)] * (-len(tiles) % cols)
    sheet = np.vstack([np.hstack(tiles[r:r + cols]) for r in range(0, len(tiles), cols)]) if tiles else np.full((10, 10, 3), 255, np.uint8)
    ok, jpg = cv2.imencode('.jpg', sheet, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return dict(rows=rows, sheet=jpg.tobytes(), modelLoadSeconds=load_seconds, containerSeconds=time.monotonic() - start,
                peakAllocatedGiB=torch.cuda.max_memory_allocated() / 2**30)


@app.local_entrypoint()
def main(candidates: str, photos_dir: str, photo: str, out: str, min_score: float = MIN_SCORE, min_iou: float = MIN_IOU):
    destination = Path(out)
    if destination.exists():
        raise ValueError('Choose a fresh output directory')
    found = json.loads(Path(candidates).read_text())['results']['transfer']['candidates']
    pairs = dict(item.split('=', 1) for item in photo.split(','))
    photos = {image_id: (Path(photos_dir) / name).read_bytes() for image_id, name in pairs.items() if image_id in {c['imageId'] for c in found}}
    start = time.monotonic()
    result = segment.remote(photos, found, min_score, min_iou)
    destination.mkdir(parents=True)
    (destination / 'contact-sheet.jpg').write_bytes(result.pop('sheet'))
    extra = {}
    for r in result['rows']:
        if r['accepted']:
            extra.setdefault(r['entityId'], {})[str(r['photoIndex0'])] = r['polygons']
    (destination / 'extra-masks.json').write_text(json.dumps(extra) + '\n')
    for r in result['rows']:
        r['polygonCount'] = len(r.pop('polygons'))
    result.update(minScore=min_score, minIoU=min_iou, insideBox=INSIDE)
    (destination / 'results.json').write_text(json.dumps(result, indent=1, ensure_ascii=False) + '\n')
    ledger = {'mode': 'ephemeral modal run', 'hardware': '1 L4, 4 CPU, 16 GiB', 'functionSeconds': result['containerSeconds'],
              'callSeconds': time.monotonic() - start, 'estimateUsd': RATE * result['containerSeconds'], 'actualBilledUsd': None,
              'rateSource': 'https://modal.com/pricing'}
    (destination / 'spend-ledger.json').write_text(json.dumps(ledger, indent=2) + '\n')
    for r in result['rows']:
        print(f"{'ACC' if r['accepted'] else 'rej'} {r['entityId'][:8]} P{r['photoIndex0'] + 1} score {r['samScore']:.2f} IoU {r['heldOutIoU']:.2f} "
              f"inside {r.get('insideBox', 0):.2f} instances {r['instances']} area {r['areaPx']} frac {r['visibleFraction']:.2f}")
    print(json.dumps(ledger))
