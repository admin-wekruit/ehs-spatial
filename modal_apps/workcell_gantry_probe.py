"""SAM 3 gantry-prompt probe: one ephemeral 2 x A100 call, N >= 2 photos, candidate text prompts.

Same image, volume and model pin as workcell_photo_all.py + workcell_sam_worker.py (raw photo grid,
no EXIF transpose, threshold .4, top-12 instances, column-major RLE), so the result is sam3.json format
and decodes with workcell_photo_objects.decode_mask. Photos alternate between the two GPUs.

modal run modal_apps/workcell_gantry_probe.py --images a.jpg,b.jpg --out DIR [--words w1,w2]
Writes DIR/gantry-sam3.json and appends the call (app id, seconds, list-rate estimate) to DIR/spend-ledger.json.
"""
import io
import json
from pathlib import Path
import re
import sys
import time

import modal

if not modal.is_local():
    sys.path[:0] = ["/repo", "/repo/scripts"]
from modal_apps.fast_report_app import build_image

app = modal.App("workcell-gantry-probe")
image = build_image(with_mapanything=True) if modal.is_local() else modal.Image.debian_slim()
USD_PER_SECOND = 0.0017752  # 2 x A100-80GB + 16 CPU + 80 GiB list rate (modal.com/pricing, checked 2026-09-30); not an invoice
WORDS = ("steel gantry frame", "overhead portal frame", "white metal frame", "door frame", "overhead beam",
         "metal post", "white post", "white pillar", "white steel beam", "gantry", "blue pipe", "steel frame")


def _segment(model, processor, device, images, words):
    import torch
    from PIL import Image
    from scripts.workcell_sam_worker import _response
    torch.cuda.set_device(device)
    rows = []
    for payload in images:
        photo = Image.open(io.BytesIO(payload)).convert("RGB")  # raw grid, as the pipeline's SAM worker
        base = processor(images=photo, return_tensors="pt").to(device)
        with torch.inference_mode():
            vision = model.get_vision_features(pixel_values=base["pixel_values"])
        view = []
        for word in words:
            kwargs = processor(text=word, original_sizes=base["original_sizes"], return_tensors="pt").to(device)
            with torch.inference_mode():
                output = model(vision_embeds=vision, **kwargs)
            masks, scores = _response(processor.post_process_instance_segmentation(
                output, threshold=.4, mask_threshold=.5, target_sizes=[(photo.height, photo.width)])[0])
            view.append({"rle": masks, "scores": scores})
        rows.append(view)
    return rows


@app.function(image=image, gpu="A100-80GB:2", cpu=16, memory=80 * 1024,
              volumes={"/v/sam3": modal.Volume.from_name("sam3-hf-cache")}, timeout=900, retries=0, min_containers=0)
def segment(images: list[bytes], words: list[str]):
    started = time.monotonic()
    try:
        import copy
        from concurrent.futures import ThreadPoolExecutor
        import torch
        from transformers import Sam3Model, Sam3Processor
        from scripts.workcell_sam_worker import MODEL_ID, REVISION
        processor = Sam3Processor.from_pretrained(MODEL_ID, revision=REVISION, cache_dir="/v/sam3/huggingface/hub")
        model = Sam3Model.from_pretrained(MODEL_ID, revision=REVISION, cache_dir="/v/sam3/huggingface/hub",
                                          torch_dtype=torch.bfloat16).eval()
        loaded = time.monotonic()
        devices = [torch.device("cuda:0"), torch.device("cuda:1")]
        models = [copy.deepcopy(model).to(devices[0]), model.to(devices[1])]
        shares = [list(range(g, len(images), 2)) for g in range(2)]
        with ThreadPoolExecutor(2) as pool:
            parts = list(pool.map(lambda g: _segment(models[g], processor, devices[g], [images[i] for i in shares[g]], words), range(2)))
        results = [None] * len(images)
        for share, part in zip(shares, parts):
            for i, view in zip(share, part):
                results[i] = view
        return {"sam3": {"prompts": [{"text": w} for w in words], "results": results},
                "modelLoadSeconds": loaded - started, "containerWallSeconds": time.monotonic() - started}
    except Exception as error:
        return {"error": f"{type(error).__name__}: {error}"[-1500:], "containerWallSeconds": time.monotonic() - started}


def _ledger_entry(destination, entry):
    path = destination / "spend-ledger.json"
    ledger = json.loads(path.read_text()) if path.is_file() else {
        "hardware": "one ephemeral 2 x A100-80GB container (16 CPU, 80 GiB)", "mode": "ephemeral modal run",
        "usdPerSecond": USD_PER_SECOND, "rateSource": "https://modal.com/pricing", "rateCheckedDate": "2026-09-30",
        "basis": "reserved-resource list-rate estimates, not invoice amounts; call window includes cold start and scheduling",
        "actualBilledUsd": None, "runs": []}
    ledger["runs"].append(entry)
    ledger["totalFunctionEstimateUsd"] = round(sum(r.get("functionEstimateUsd") or 0 for r in ledger["runs"]), 4)
    ledger["totalCallEstimateUsd"] = round(sum(r["callEstimateUsd"] for r in ledger["runs"]), 4)
    path.write_text(json.dumps(ledger, indent=2) + "\n")


@app.local_entrypoint()
def main(images: str, out: str, words: str = ",".join(WORDS)):
    paths = [Path(p) for p in images.split(",")]
    if len(paths) < 2 or len({p.resolve() for p in paths}) != len(paths) or any(not p.is_file() for p in paths):
        raise ValueError("At least two distinct readable photos are required")
    prompts = [w.strip() for w in words.split(",") if w.strip()]
    if not prompts:
        raise ValueError("No prompts")
    destination = Path(out)
    destination.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    entry = {"stage": "gantry SAM 3 prompt probe", "appId": app.app_id, "photos": [str(p) for p in paths], "prompts": prompts}
    result = {}
    try:
        result = segment.remote([p.read_bytes() for p in paths], prompts)
    except Exception as error:
        result = {"error": f"{type(error).__name__}: {error}"[-1500:]}
    finally:
        call = time.monotonic() - started
        function = result.get("containerWallSeconds")
        error = result.get("error")
        entry.update(status="failed" if error else "ok", error=error and re.sub(r"(?i)(capabilit|token|secret)\S*", "[filtered]", error),
                     functionSeconds=round(function, 2) if function is not None else None,
                     modelLoadSeconds=round(result["modelLoadSeconds"], 2) if "modelLoadSeconds" in result else None,
                     callSeconds=round(call, 2),
                     functionEstimateUsd=round(function * USD_PER_SECOND, 4) if function is not None else None,
                     callEstimateUsd=round(call * USD_PER_SECOND, 4))
        _ledger_entry(destination, entry)
    if error:
        raise RuntimeError(f"Gantry probe failed; ledger entry written: {entry['error']}")
    (destination / "gantry-sam3.json").write_text(json.dumps(result["sam3"]))
    print(json.dumps({k: entry[k] for k in ("appId", "functionSeconds", "callSeconds", "callEstimateUsd")}))
