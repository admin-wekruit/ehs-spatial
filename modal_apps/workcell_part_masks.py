"""Part masks for scripts/workcell_checks/lower_edge.py, restricted to an object's report-mask box. Two modes, both generic.

Point (click exemplar, --prompts P.json from `lower_edge.py --prompts-out`): one positive point per photo where the clicked part's
3D anchor is visible (the click itself in the click photo). SAM 3's tracker head (PVS: promptable visual segmentation,
transformers Sam3TrackerModel, same pinned facebook/sam3 checkpoint, weights volume and image as the text mode) segments a
square --tile px crop centred on the point; of its 3 multimask candidates the one with the best predicted IoU that contains the
point, overlaps the object's mask and covers <= --max-cover (0.8) of the object's mask inside the crop wins; a part mask that
covers more is the whole object, not a part, and is rejected (no mask for that photo). The mask is clipped to the box (padded
PAD of its size); lower_edge.py intersects it with the object's mask.
Text (--targets T.json [{entityId, part: phrase}], the earlier mode): the mask box is covered by overlapping square tiles of
--tile px (SAM 3 sees 1008 px: a whole 12 MP photo leaves a light-curtain post ~40 px wide and its mask edge ~10 px coarse),
each segmented with the phrase; every instance with score >= --threshold is pasted back and the part mask is their union. Text
cannot tell two same-coloured neighbouring pieces apart (research-notes/workcell-lower-edge-2026-10-05), clicks can.

Entity ids are matched exactly in both modes. Writes OUT/part-masks.json {entityId: {part: {photoIndex0: polygons}}} (every part
that ran, {} where no photo got a mask; original pixel coordinates, even-odd) to pass as
lower_edge opts partMasks, results.json (every instance / candidate: score, box, area, cover), contact-sheet.jpg (cyan =
object mask, magenta = part mask, green = prompt point), spend-ledger.json. One ephemeral L4, offline weights. On-prem:
python scripts/onprem/run_stage.py --weights DIR modal_apps/workcell_part_masks.py ...

modal run modal_apps/workcell_part_masks.py --view VIEW.json --photos-dir DIR --photo IMAGE_ID=FILE,... --prompts P.json --out NEW_DIR \
    [--tile 512] [--max-cover .8]
modal run modal_apps/workcell_part_masks.py --view VIEW.json --photos-dir DIR --photo IMAGE_ID=FILE,... --targets T.json --out NEW_DIR \
    [--tile 512] [--threshold .1]
"""
import json
from pathlib import Path
import time

import modal

REPO = Path(__file__).resolve().parents[1]
RATE = .000222 + 4 * .0000131 + 16 * .00000222  # L4 + 4 CPU + 16 GiB list rate (USD/s); not an invoice
PAD = .05

app = modal.App('workcell-part-masks')
image = (modal.Image.debian_slim(python_version='3.11')
         .pip_install('torch==2.14.0', 'torchvision', 'transformers==5.17.0', 'accelerate', 'pillow', 'numpy<2.3', 'opencv-python-headless==4.10.0.84')
         .env({'HF_HUB_OFFLINE': '1'})
         .add_local_file(REPO / 'scripts/workcell_sam_worker.py', '/sam/workcell_sam_worker.py')
         .add_local_file(REPO / 'scripts/workcell_checks/transfer.py', '/sam/transfer.py')
         .add_local_file(REPO / 'scripts/workcell_shape_check.py', '/sam/shape_core.py'))


def box_tiles(box, width, height, size):
    """Square size-px tiles (>= 25 % overlap, clamped to the photo) covering box = (x0, y0, x1, y1); one centred tile per
    axis where the box is narrower than a tile."""
    size = min(size, width, height); x0, y0, x1, y1 = box; stride = max(1, int(size * .75))

    def starts(a, b, n):
        s = [(a + b - size) // 2] if b - a <= size else list(range(a, b - size, stride)) + [b - size]
        return sorted({min(max(0, v), n - size) for v in s})
    return [(x, y, size, size) for y in starts(y0, y1, height) for x in starts(x0, x1, width)]


def mask_boxes(doc, entity_id):
    """{photoIndex0: (x0, y0, x1, y1)} of an entity's report masks (observation polygons), padded by PAD of the box size."""
    index = {c['imageId']: k for k, c in enumerate(doc['cameras'])}; obs = {o['id']: o for o in doc['observations']}
    e = next(e for e in doc['entities'] if e['id'] == entity_id); out = {}
    for oid in e.get('observationRefs') or []:
        o = obs.get(oid)
        if o and o['imageId'] in index and o.get('originalPixelPolygons'):
            pts = [p for poly in o['originalPixelPolygons'] for p in poly]; k = index[o['imageId']]
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            b = out.get(k, (min(xs), min(ys), max(xs), max(ys)))
            out[k] = (min(b[0], min(xs)), min(b[1], min(ys)), max(b[2], max(xs)), max(b[3], max(ys)))
    cam = {k: doc['cameras'][k] for k in out}
    pad = lambda b, k: (max(0, int(b[0] - PAD * (b[2] - b[0]))), max(0, int(b[1] - PAD * (b[3] - b[1]))),
                        min(int(cam[k]['width']), int(b[2] + PAD * (b[2] - b[0])) + 1), min(int(cam[k]['height']), int(b[3] + PAD * (b[3] - b[1])) + 1))
    return {k: pad(b, k) for k, b in out.items()}


@app.function(image=image, gpu='L4', cpu=4, memory=16 * 1024, volumes={'/v/sam3': modal.Volume.from_name('sam3-hf-cache')},
              timeout=1200, retries=0, min_containers=0)
def segment(photos: dict, jobs: list, tile: int, threshold: float) -> dict:
    """jobs: [{entityId, part, photoIndex0, imageId, box, polygons}] -> per job the union part mask (polygons) and its instances."""
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
    rows, tiles_img = [], []
    for image_id in dict.fromkeys(j['imageId'] for j in jobs):
        bgr = cv2.imdecode(np.frombuffer(photos[image_id], np.uint8), cv2.IMREAD_COLOR)  # EXIF-oriented, as load_report reads it
        H, W = bgr.shape[:2]; vision_cache = {}
        for j in (j for j in jobs if j['imageId'] == image_id):
            x0, y0, x1, y1 = j['box']; part = np.zeros((H, W), bool); instances = []
            for tx, ty, tw, th in box_tiles(j['box'], W, H, tile):
                key = (tx, ty, tw, th)
                if key not in vision_cache:  # tiles shared by targets of the same photo are encoded once
                    photo = Image.fromarray(np.ascontiguousarray(bgr[ty:ty + th, tx:tx + tw, ::-1]))
                    base = processor(images=photo, return_tensors='pt').to('cuda')
                    with torch.inference_mode():
                        vision_cache[key] = (model.get_vision_features(pixel_values=base['pixel_values']), base['original_sizes'])
                vision, sizes = vision_cache[key]
                with torch.inference_mode():
                    kwargs = processor(text=j['part'], original_sizes=sizes, return_tensors='pt').to('cuda')
                    output = model(vision_embeds=vision, **kwargs)
                result = processor.post_process_instance_segmentation(output, threshold=threshold, mask_threshold=.5, target_sizes=[(th, tw)])[0]
                masks, scores = result.get('masks'), result.get('scores')
                if masks is None or not len(masks):
                    continue
                scores = scores.float().cpu().numpy(); masks = (masks.float() > .5).cpu().numpy()
                for m, s in zip(masks, scores):
                    ys, xs = np.nonzero(m)
                    if len(ys):
                        part[ty:ty + th, tx:tx + tw] |= m
                        polys = [[[x + tx, y + ty] for x, y in poly] for poly in transfer.polygons_of(m, 1, .5)]  # for re-selection offline
                        instances.append(dict(tile=[tx, ty, tw, th], score=float(s), areaPx=int(len(ys)), polygons=polys,
                                              bbox=[int(xs.min()) + tx, int(ys.min()) + ty, int(xs.max()) + 1 + tx, int(ys.max()) + 1 + ty]))
            clip = np.zeros_like(part); clip[y0:y1, x0:x1] = True; part &= clip
            obj = np.zeros((H, W), bool)
            for polys in j['polygons']:  # one polygon list per observation: even-odd within, union across (as load_report)
                obj |= wsc.polygon_mask(polys, (H, W))
            row = {k: j[k] for k in ('entityId', 'part', 'photoIndex0', 'imageId', 'box')}
            row.update(instances=instances, partAreaPx=int(part.sum()), insideObjectPx=int((part & obj).sum()), objectAreaPx=int(obj.sum()),
                       polygons=transfer.polygons_of(part, 1, .5) if part.any() else [])
            rows.append(row)
            tiles_img.append(sheet_tile(bgr, (x0, y0, x1, y1), obj, part, f"{j['entityId'][:8]} P{j['photoIndex0'] + 1} {j['part'][:28]}"))
        del vision_cache
    return dict(rows=rows, sheet=sheet_of(tiles_img), modelLoadSeconds=load_seconds,
                containerSeconds=time.monotonic() - start, peakAllocatedGiB=torch.cuda.max_memory_allocated() / 2**30)


def sheet_tile(bgr, box, obj, part, label, point=None, crop_box=None):
    """Contact-sheet tile: the box (or crop_box) with the object mask (cyan), the part mask (magenta) and the prompt (green)."""
    import cv2
    import numpy as np
    x0, y0, x1, y1 = crop_box or box
    crop = bgr[y0:y1, x0:x1].copy(); f = 400 / max(crop.shape[:2])
    crop = cv2.resize(crop, (max(1, round(crop.shape[1] * f)), max(1, round(crop.shape[0] * f))), interpolation=cv2.INTER_AREA)
    for m, color in ((obj, (255, 255, 0)), (part, (255, 0, 255))):
        small = cv2.resize(m[y0:y1, x0:x1].astype(np.uint8), (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST)
        cs, _ = cv2.findContours(small, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(crop, cs, -1, color, 2)
    if point is not None:
        cv2.drawMarker(crop, (round((point[0] - x0) * f), round((point[1] - y0) * f)), (0, 255, 0), cv2.MARKER_CROSS, 18, 2)
    t = np.full((430, 400, 3), 255, np.uint8); t[:crop.shape[0], :crop.shape[1]] = crop
    cv2.putText(t, label, (4, 422), cv2.FONT_HERSHEY_SIMPLEX, .42, (0, 0, 0), 1, cv2.LINE_AA)
    return t


def sheet_of(tiles):
    import cv2
    import numpy as np
    cols = min(6, len(tiles)) or 1
    tiles = tiles + [np.full((430, 400, 3), 255, np.uint8)] * (-len(tiles) % cols)
    sheet = np.vstack([np.hstack(tiles[r:r + cols]) for r in range(0, len(tiles), cols)]) if tiles else np.full((10, 10, 3), 255, np.uint8)
    return cv2.imencode('.jpg', sheet, [cv2.IMWRITE_JPEG_QUALITY, 82])[1].tobytes()


@app.function(image=image, gpu='L4', cpu=4, memory=16 * 1024, volumes={'/v/sam3': modal.Volume.from_name('sam3-hf-cache')},
              timeout=1200, retries=0, min_containers=0)
def segment_points(photos: dict, jobs: list, tile: int, max_cover: float) -> dict:
    """jobs: [{entityId, part, photoIndex0, imageId, box, polygons, point}] -> per job the SAM 3 tracker mask of one positive point."""
    import sys
    sys.path.insert(0, '/sam')
    import cv2
    import numpy as np
    from PIL import Image
    import torch
    import transformers
    import shape_core as wsc
    import transfer
    import workcell_sam_worker as worker  # the pinned model id and revision
    from transformers import Sam3TrackerModel, Sam3TrackerProcessor  # SAM 3 PVS (point prompts), in the pinned transformers
    start = time.monotonic()
    cache = '/v/sam3/huggingface/hub'
    processor = Sam3TrackerProcessor.from_pretrained(worker.MODEL_ID, revision=worker.REVISION, cache_dir=cache)
    model = Sam3TrackerModel.from_pretrained(worker.MODEL_ID, revision=worker.REVISION, cache_dir=cache).to('cuda').eval()
    load_seconds = time.monotonic() - start
    rows, tiles_img = [], []
    for image_id in dict.fromkeys(j['imageId'] for j in jobs):
        bgr = cv2.imdecode(np.frombuffer(photos[image_id], np.uint8), cv2.IMREAD_COLOR)  # EXIF-oriented, as load_report reads it
        H, W = bgr.shape[:2]
        for j in (j for j in jobs if j['imageId'] == image_id):
            x0, y0, x1, y1 = j['box']; u, v = j['point']; size = min(tile, W, H)
            tx, ty = min(max(0, round(u - size / 2)), W - size), min(max(0, round(v - size / 2)), H - size)
            obj = np.zeros((H, W), bool)
            for polys in j['polygons']:  # one polygon list per observation: even-odd within, union across (as load_report)
                obj |= wsc.polygon_mask(polys, (H, W))
            box = np.zeros((size, size), bool); box[max(0, y0 - ty):max(0, y1 - ty), max(0, x0 - tx):max(0, x1 - tx)] = True
            obj_t = obj[ty:ty + size, tx:tx + size]; px, py = min(size - 1, max(0, round(u - tx))), min(size - 1, max(0, round(v - ty)))
            inputs = processor(images=Image.fromarray(np.ascontiguousarray(bgr[ty:ty + size, tx:tx + size, ::-1])),
                               input_points=[[[[u - tx, v - ty]]]], input_labels=[[[1]]], return_tensors='pt').to('cuda')
            with torch.inference_mode():
                output = model(**inputs, multimask_output=True)
            masks = processor.post_process_masks(output.pred_masks.float().cpu(), inputs['original_sizes'].cpu())[0][0].numpy() > 0
            scores = output.iou_scores[0, 0].float().cpu().numpy(); cands = []
            for i, (m, sc) in enumerate(zip(masks, scores)):
                m = m & box; inside = int((m & obj_t).sum())
                cands.append(dict(index=i, score=float(sc), areaPx=int(m.sum()), insideObjectPx=inside, containsPoint=bool(m[py, px]),
                                  coverOfObjectInCrop=round(inside / max(1, int(obj_t.sum())), 3)))
            ok = [c for c in cands if c['containsPoint'] and c['insideObjectPx'] and c['coverOfObjectInCrop'] <= max_cover]
            best = max(ok, key=lambda c: c['score'], default=None)
            part = np.zeros((H, W), bool)
            if best:
                part[ty:ty + size, tx:tx + size] = masks[best['index']] & box
            row = {k: j[k] for k in ('entityId', 'part', 'photoIndex0', 'imageId', 'box', 'point')}
            row.update(crop=[tx, ty, size, size], candidates=cands, chosen=None if best is None else best['index'],
                       rejected=None if best else ('covers_object' if any(c['containsPoint'] and c['insideObjectPx'] for c in cands) else 'no_mask_at_point'),
                       partAreaPx=int(part.sum()), insideObjectPx=int((part & obj).sum()), objectAreaPx=int(obj.sum()),
                       polygons=transfer.polygons_of(part, 1, .5) if part.any() else [])
            rows.append(row)
            label = f"{j['entityId'][:8]} P{j['photoIndex0'] + 1} {j['part'][:16]} " + (f"s{best['score']:.2f} c{best['coverOfObjectInCrop']:.2f}" if best else row['rejected'])
            tiles_img.append(sheet_tile(bgr, None, obj, part, label, point=(u, v), crop_box=(tx, ty, tx + size, ty + size)))
    return dict(rows=rows, sheet=sheet_of(tiles_img), modelLoadSeconds=load_seconds, containerSeconds=time.monotonic() - start,
                peakAllocatedGiB=torch.cuda.max_memory_allocated() / 2**30, transformers=transformers.__version__, model='Sam3TrackerModel')


@app.local_entrypoint()
def main(view: str, photos_dir: str, photo: str, out: str, targets: str = '', prompts: str = '', tile: int = 512, threshold: float = .1,
         max_cover: float = .8):
    destination = Path(out)
    if destination.exists():
        raise ValueError('Choose a fresh output directory')
    if bool(targets) == bool(prompts):
        raise ValueError('give --targets (text parts) or --prompts (click parts)')
    doc = json.loads(Path(view).read_text())['publication']['snapshot']['revision']['document']
    obs = {o['id']: o for o in doc['observations']}
    jobs = []

    def entity(eid):  # exact ids only
        e = next((e for e in doc['entities'] if e['id'] == eid), None)
        if e is None:
            raise ValueError(f'no entity {eid!r} in {view} (give the full entity id)')
        return e

    def job(eid, part, k, **extra):
        e = entity(eid); image_id = doc['cameras'][k]['imageId']
        polys = [obs[o]['originalPixelPolygons'] for o in e.get('observationRefs') or [] if o in obs and obs[o]['imageId'] == image_id and obs[o].get('originalPixelPolygons')]
        return dict(entityId=eid, part=part, photoIndex0=k, imageId=image_id, box=list(mask_boxes(doc, eid)[k]), polygons=polys, **extra)
    if prompts:  # click parts: one point per photo (lower_edge.py --prompts-out)
        spec = json.loads(Path(prompts).read_text()); ran = [(a['entityId'], a['part']) for a in spec['anchors']]
        jobs = [job(p['entityId'], p['part'], p['photoIndex0'], point=p['point']) for p in spec['prompts']]
    else:
        spec = json.loads(Path(targets).read_text()); ran = [(entity(t['entityId'])['id'], t['part']) for t in spec]
        for eid, part in ran:
            jobs += [job(eid, part, k) for k in sorted(mask_boxes(doc, eid))]
    pairs = dict(item.split('=', 1) for item in photo.split(','))
    photos = {i: (Path(photos_dir) / name).read_bytes() for i, name in pairs.items() if i in {j['imageId'] for j in jobs}}
    start = time.monotonic()
    result = segment_points.remote(photos, jobs, tile, max_cover) if prompts else segment.remote(photos, jobs, tile, threshold)
    destination.mkdir(parents=True)
    (destination / 'contact-sheet.jpg').write_bytes(result.pop('sheet'))
    masks = {}
    for eid, part in ran:  # every target is listed, {} when no photo got a mask (lower_edge tells that from a label typo)
        masks.setdefault(eid, {}).setdefault(part, {})
    for r in result['rows']:
        if r['polygons']:
            masks[r['entityId']][r['part']][str(r['photoIndex0'])] = r['polygons']
        r['polygonCount'] = len(r.pop('polygons'))
    (destination / 'part-masks.json').write_text(json.dumps(masks) + '\n')
    result.update(tile=tile, pad=PAD, **({'maxCover': max_cover} if prompts else {'threshold': threshold}))
    (destination / 'results.json').write_text(json.dumps(result, indent=1, ensure_ascii=False) + '\n')
    ledger = {'mode': 'ephemeral modal run', 'hardware': '1 L4, 4 CPU, 16 GiB', 'functionSeconds': result['containerSeconds'],
              'callSeconds': time.monotonic() - start, 'estimateUsd': RATE * result['containerSeconds'], 'actualBilledUsd': None,
              'rateSource': 'https://modal.com/pricing'}
    (destination / 'spend-ledger.json').write_text(json.dumps(ledger, indent=2) + '\n')
    for r in result['rows']:
        if prompts:
            print(f"{r['entityId'][:8]} P{r['photoIndex0'] + 1} '{r['part']}': chosen {r['chosen']} {r['rejected'] or ''} part {r['partAreaPx']} px, inside object "
                  f"{r['insideObjectPx']} / {r['objectAreaPx']}", [(c['index'], round(c['score'], 2), c['coverOfObjectInCrop'], c['containsPoint']) for c in r['candidates']])
        else:
            print(f"{r['entityId'][:8]} P{r['photoIndex0'] + 1} '{r['part']}': {len(r['instances'])} instance(s), part {r['partAreaPx']} px, "
                  f"inside object {r['insideObjectPx']} / {r['objectAreaPx']}", [round(i['score'], 2) for i in r['instances']][:8])
    print(json.dumps(ledger))
