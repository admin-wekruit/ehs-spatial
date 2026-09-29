"""X9 probe (one A100): what the reuse ladder can use in the a-core image, and what it costs.
  - SAM 3's tracker (box prompts, SAM 2 style decoder) from the cached facebook/sam3 files: loads offline? per-frame image
    embedding and per-box decode time; does it share SAM 3's vision trunk weights?
  - DINOv2 (public weights, downloaded into the Modal volume, never locally): dense features per frame, time.
  modal run modal_apps/x9_probe.py
"""
import json
import os
import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "scripts"), str(HERE.parent)]
import fast_report_app as fr  # noqa: E402

app = modal.App("panoptes-fx-x9-probe")
OUT = modal.Volume.from_name("panoptes-fx-x9-reuse", create_if_missing=True)
DINO = "facebook/dinov2-base"
image = fr.image.add_local_python_source("fast_report_app")


@app.function(image=image, gpu="A100-80GB", timeout=900, retries=0, volumes={"/v/sam3": fr.VOLUMES["/v/sam3"], "/v/out": OUT})
def probe():
    os.environ["HF_HUB_OFFLINE"] = "0"  # DINOv2 only; SAM 3 files are read from the cache with local_files_only
    import torch
    import transformers
    from huggingface_hub import snapshot_download
    import sam3_app
    out = {"transformers": transformers.__version__, "classes": [n for n in dir(transformers) if n.startswith(("Sam3Tracker", "Dinov2"))]}
    dev = torch.device("cuda:0")
    t = time.perf_counter()
    snapshot_download(DINO, cache_dir="/v/out/hf")
    OUT.commit()
    out["dino_download_s"] = round(time.perf_counter() - t, 1)
    # tracker
    try:
        from transformers import Sam3TrackerModel, Sam3TrackerProcessor
        t = time.perf_counter()
        kw = dict(revision=sam3_app.REVISION, cache_dir="/v/sam3/huggingface/hub", local_files_only=True)
        tp = Sam3TrackerProcessor.from_pretrained(sam3_app.MODEL_ID, **kw)
        tm = Sam3TrackerModel.from_pretrained(sam3_app.MODEL_ID, torch_dtype=torch.bfloat16, **kw).to(dev).eval()
        out["tracker_load_s"] = round(time.perf_counter() - t, 1)
        out["tracker_image_size"] = getattr(tp.image_processor, "size", None)
        names = [n for n, _ in tm.named_parameters()]
        out["tracker_param_prefixes"] = sorted({".".join(n.split(".")[:2]) for n in names})[:40]
        from transformers import Sam3Model
        sm = Sam3Model.from_pretrained(sam3_app.MODEL_ID, torch_dtype=torch.bfloat16, **kw).eval()
        sp = dict(sm.named_parameters())
        tpar = dict(tm.named_parameters())
        same, total = 0, 0
        for n, p in tpar.items():
            if "vision_encoder.backbone" in n:
                total += 1
                k = n.replace("vision_encoder.backbone", "vision_encoder.backbone")
                if k in sp and sp[k].shape == p.shape and torch.equal(sp[k].to(dev), p):
                    same += 1
        out["tracker_backbone_params_equal_to_sam3"] = [same, total]
        out["sam3_param_prefixes"] = sorted({".".join(n.split(".")[:2]) for n in sp})[:40]
        del sm
        x = torch.randint(0, 255, (8, 3, 1008, 1008), device=dev).float()
        mean = torch.tensor(tp.image_processor.image_mean, device=dev).view(1, 3, 1, 1)
        std = torch.tensor(tp.image_processor.image_std, device=dev).view(1, 3, 1, 1)
        pv = ((x / 255 - mean) / std).to(torch.bfloat16)
        with torch.inference_mode():
            for _ in range(2):
                emb = tm.get_image_features(pixel_values=pv)
            torch.cuda.synchronize()
            t = time.perf_counter()
            for _ in range(3):
                emb = tm.get_image_features(pixel_values=pv)
            torch.cuda.synchronize()
            out["tracker_embed_s_per_frame"] = round((time.perf_counter() - t) / 24, 4)
            out["tracker_embed_type"] = str(type(emb))[:200]
            out["tracker_embed_shapes"] = [list(e.shape) for e in (emb if isinstance(emb, (list, tuple)) else [emb])][:6] if not hasattr(emb, "keys") else {k: str(getattr(v, 'shape', type(v)))[:80] for k, v in emb.items()}
            # box prompts: 64 boxes on one frame
            one = [e[:1] for e in emb] if isinstance(emb, (list, tuple)) else emb
            boxes = torch.tensor([[[100. + i, 100., 300. + i, 400.] for i in range(64)]], device=dev)
            for _ in range(2):
                o = tm(pixel_values=pv[:1], input_boxes=boxes, multimask_output=False)
            torch.cuda.synchronize()
            t = time.perf_counter()
            for _ in range(5):
                o = tm(pixel_values=pv[:1], input_boxes=boxes, multimask_output=False)
            torch.cuda.synchronize()
            out["tracker_full_forward_64_boxes_s"] = round((time.perf_counter() - t) / 5, 4)
            out["tracker_pred_masks_shape"] = list(o.pred_masks.shape)
            try:
                t = time.perf_counter()
                for _ in range(5):
                    o2 = tm(image_embeddings=one, input_boxes=boxes, multimask_output=False)
                torch.cuda.synchronize()
                out["tracker_decode_64_boxes_on_cached_embedding_s"] = round((time.perf_counter() - t) / 5, 4)
                out["tracker_cached_equal"] = bool(torch.allclose(o2.pred_masks.float(), o.pred_masks.float(), atol=1e-2))
            except Exception as e:  # noqa: BLE001
                out["tracker_cached_error"] = repr(e)[:600]
    except Exception as e:  # noqa: BLE001
        import traceback
        out["tracker_error"] = traceback.format_exc()[-2000:]
    # DINOv2 dense
    from transformers import AutoModel
    dm = AutoModel.from_pretrained(DINO, cache_dir="/v/out/hf", torch_dtype=torch.bfloat16).to(dev).eval()
    for hw in ((504, 896), (280, 504)):
        x = torch.randn(16, 3, *hw, device=dev, dtype=torch.bfloat16)
        with torch.inference_mode():
            for _ in range(2):
                h = dm(pixel_values=x).last_hidden_state
            torch.cuda.synchronize()
            t = time.perf_counter()
            for _ in range(3):
                h = dm(pixel_values=x).last_hidden_state
            torch.cuda.synchronize()
        out[f"dino_{hw[0]}x{hw[1]}_s_per_frame"] = round((time.perf_counter() - t) / 48, 4)
        out[f"dino_{hw[0]}x{hw[1]}_tokens"] = list(h.shape)
    out["peak_gib"] = round(torch.cuda.max_memory_reserved(dev) / 2 ** 30, 2)
    return json.loads(json.dumps(out, default=str))


@app.local_entrypoint()
def main():
    print(json.dumps(probe.remote(), indent=1, default=str))


@app.function(image=image, cpu=2, timeout=300, retries=0)
def src(names: list):
    """transformers source of the SAM 3 tracker / vision entry points (to share SAM 3's trunk with the tracker)."""
    import importlib
    import inspect
    out = []
    for n in names:
        mod, attr = n.rsplit(":", 1)
        obj = importlib.import_module(mod)
        for a in attr.split("."):
            obj = getattr(obj, a)
        out.append(f"##### {n}\n" + inspect.getsource(obj))
    return "\n".join(out)


@app.local_entrypoint()
def source(names: str):
    print(src.remote(names.split(",")))
