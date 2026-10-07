"""E3 (fast path, people and movers): SAM 3 image detection and SAM 3.1 video tracking on ME340, one A100-80GB each.

Both calls run at once in one ephemeral app (retries 0, explicit timeouts) and answer with their own clocks, so warm
compute, model load and the client's wall (cold start + transfer) are reported apart.

  detect: SAM 3 (transformers, the sam3_app weights pin) on every depth frame of the walk shot and every 3rd frame of the
          cut-away, image encoded once per frame, prompts 'person' + a small mover vocabulary. Masks come back as the
          sam3_app response (top 12, threshold 0.4). Also batched 'person' throughput.
  track:  SAM 3.1 multiplex (the sam3_video pin), text 'person', walk shot 228..897: stride 3 twice in one container
          (the second is warm), then stride 1; the stride-3 masks come back and are compared to stride 1 in the container.

Frames are on the depth raster (droid_room.prepare_image), what scripts/replay_people_stream.py and
sam3_motion_tracks.py feed. Scored locally by scripts/e3_people_eval.py.

  python modal_apps/e3_people_probe.py --output NEW_DIR
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
import sam3_app  # noqa: E402  image, weights pin and response shape of our SAM 3 image service
import sam3_video  # noqa: E402  image, weights pin and session helper of SAM 3.1 video

WORDS = ["person", "forklift", "pallet jack", "cart", "vehicle"]  # person + the critique's mover vocabulary
WALK = (228, 898)  # walk shot 226..898; 228 is the first frame on the every-3rd depth grid
GPU, USD_PER_S = "A100-80GB", 0.000694  # Modal list price, GPU only; CPU/memory added in the report
app = modal.App("panoptes-e3-people-probe")
hf = modal.Secret.from_name("huggingface")


@app.function(image=sam3_app.image.add_local_python_source("sam3_app", "sam3_video"), gpu=GPU, timeout=900, retries=0,
              max_containers=1, scaledown_window=2, volumes={"/cache": sam3_app.volume}, secrets=[hf])
def detect_remote(jpegs, words, batch=8):
    import torch
    from PIL import Image
    from transformers import Sam3Model, Sam3Processor
    began = time.time()
    processor = Sam3Processor.from_pretrained(sam3_app.MODEL_ID, revision=sam3_app.REVISION)
    model = Sam3Model.from_pretrained(sam3_app.MODEL_ID, revision=sam3_app.REVISION, torch_dtype=torch.bfloat16).to("cuda").eval()
    load_s = time.time() - began

    def clock():
        torch.cuda.synchronize()
        return time.perf_counter()

    def one(jpeg):
        t0 = clock()
        image = Image.open(io.BytesIO(jpeg)).convert("RGB")
        base = processor(images=image, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            vision = model.get_vision_features(pixel_values=base["pixel_values"])
        ms, out = {"encode": (clock() - t0) * 1000}, {}
        for w in words:  # the prompt side only, on the cached encoder output (sam3_app._segment_one)
            t = clock()
            inputs = processor(text=w, original_sizes=base["original_sizes"], return_tensors="pt").to("cuda")
            inputs["vision_embeds"] = vision
            with torch.inference_mode():
                outputs = model(**inputs)
            out[w] = sam3_app._response(processor.post_process_instance_segmentation(
                outputs, threshold=0.4, mask_threshold=0.5, target_sizes=[(image.height, image.width)])[0])
            ms[w] = (clock() - t) * 1000
        return out, ms

    for jpeg in jpegs[:3]:
        one(jpeg)  # kernels and allocator, untimed
    warm = time.time()
    frames = []
    for jpeg in jpegs:
        out, ms = one(jpeg)
        frames.append({"rle": {w: r for w, (r, _) in out.items()}, "scores": {w: s for w, (_, s) in out.items()}, "ms": ms})
    per_frame_s = time.time() - warm
    batched = []  # 'person' only, `batch` images per encoder call: what a fast path would do
    for start in range(0, len(jpegs) - batch + 1, batch):
        t = clock()
        images = [Image.open(io.BytesIO(j)).convert("RGB") for j in jpegs[start:start + batch]]
        base = processor(images=images, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            vision = model.get_vision_features(pixel_values=base["pixel_values"])
            inputs = processor(text=["person"] * batch, original_sizes=base["original_sizes"], return_tensors="pt").to("cuda")
            outputs = model(vision_embeds=vision, **inputs)
        for r in processor.post_process_instance_segmentation(outputs, threshold=0.4, mask_threshold=0.5,
                                                              target_sizes=[(i.height, i.width) for i in images]):
            sam3_app._response(r)
        batched.append((clock() - t) * 1000 / batch)
    return {"gpu": torch.cuda.get_device_name(), "load_s": load_s, "per_frame_loop_s": per_frame_s, "frames": frames,
            "batched_person_ms_per_image": batched, "batch": batch, "peak_gb": torch.cuda.max_memory_allocated() / 2**30,
            "started": began, "ended": time.time()}


@app.function(image=sam3_video.video_image.add_local_python_source("sam3_app", "sam3_video"), gpu=GPU, timeout=1200,
              retries=0, max_containers=1, scaledown_window=2, volumes={"/cache": sam3_video.cache}, secrets=[hf])
def track_remote(jpegs, text="person", runs=(("stride3_first", 3), ("stride3_warm", 3), ("stride1", 1))):
    """jpegs: every frame of the window in order. -> report + npz of the warm stride-3 masks (key frame/obj, frame = window index)."""
    import tempfile
    import torch
    from huggingface_hub import hf_hub_download
    from sam3.model_builder import build_sam3_multiplex_video_predictor
    began = time.time()
    checkpoint = hf_hub_download(sam3_video.MODEL_ID, sam3_video.CHECKPOINT, revision=sam3_video.MODEL_REVISION)
    predictor = build_sam3_multiplex_video_predictor(checkpoint_path=checkpoint, max_num_objects=16, multiplex_count=16,
                                                     use_fa3=False, compile=False, warm_up=False)
    torch.cuda.synchronize()
    report, unions = {"gpu": torch.cuda.get_device_name(), "load_s": time.time() - began, "runs": {}}, {}
    arrays = {}
    for name, stride in runs:
        shown = list(range(0, len(jpegs), stride))
        with tempfile.TemporaryDirectory() as folder:
            for n, i in enumerate(shown):
                (Path(folder) / f"{n:05d}.jpg").write_bytes(jpegs[i])
            t0 = time.perf_counter()
            session = sam3_video.start_session(predictor, folder)
            t1 = time.perf_counter()
            predictor.handle_request({"type": "add_prompt", "session_id": session, "frame_index": 0, "text": text})
            t2 = time.perf_counter()
            ticks, union, objects = [], {}, set()
            try:
                for response in predictor.handle_stream_request({"type": "propagate_in_video", "session_id": session,
                                                                 "propagation_direction": "forward", "start_frame_index": 0}):
                    out = response["outputs"]
                    ids = np.asarray(out["out_obj_ids"]).astype(int)
                    masks = np.asarray(out["out_binary_masks"]).astype(bool)
                    frame = shown[int(response["frame_index"])]
                    union[frame] = masks.any(0) if len(masks) else None
                    objects.update(ids.tolist())
                    if name == "stride3_warm":
                        for i, m in zip(ids, masks):
                            arrays[f"{frame}/{int(i)}"] = np.packbits(m)
                    ticks.append(time.perf_counter())
                torch.cuda.synchronize()
            finally:
                predictor.handle_request({"type": "close_session", "session_id": session})
            t3 = time.perf_counter()
        steps = np.diff([t2] + ticks) * 1000
        report["runs"][name] = {"stride": stride, "frames": len(shown), "answered": len(ticks), "objects": len(objects),
                                "init_state_s": t1 - t0, "text_prompt_s": t2 - t1, "propagate_s": t3 - t2,
                                "fps_propagate": len(ticks) / (t3 - t2), "fps_with_init": len(ticks) / (t3 - t0),
                                "ms_per_frame_p50": float(np.median(steps)), "ms_per_frame_p95": float(np.percentile(steps, 95))}
        unions[name] = union
    # stride 3 against stride 1 on the frames both saw: does skipping frames change the person masks?
    a, b = unions.get("stride3_warm", {}), unions.get("stride1", {})
    ious = []
    for f in sorted(set(a) & set(b)):
        x, y = a[f], b[f]
        if x is None and y is None:
            continue
        x = np.zeros_like(y) if x is None else x
        y = np.zeros_like(x) if y is None else y
        ious.append(float((x & y).sum() / max((x | y).sum(), 1)))
    report["stride3_vs_stride1_union_iou"] = {"frames": len(ious), "median": float(np.median(ious)) if ious else None,
                                              "p10": float(np.percentile(ious, 10)) if ious else None}
    report["peak_gb"] = torch.cuda.max_memory_allocated() / 2**30
    report.update(started=began, ended=time.time())
    buffer = io.BytesIO()
    np.savez_compressed(buffer, report=json.dumps(report), **arrays)
    return buffer.getvalue()


def jpegs_on_raster():
    """(detect frame ids, walk frame ids 228..897, detect JPEGs q90 as the replay stream sends, walk JPEGs q95 as
    sam3_motion_tracks sends), all on the depth raster, from the clip's rgb/. Encoded one frame at a time."""
    import cv2
    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    import replay_people_stream as R
    rows = [line.split() for line in (R.CLIP / "rgb.txt").read_text().splitlines() if not line.startswith("#")]
    with_depth = {int(p.stem) for p in (R.DEPTH / "mono").glob("*.npz")}
    detect, walk = sorted(with_depth | set(range(0, 898, 3))), list(range(*WALK))

    def jpeg(i, quality):
        image = R.raster(cv2.imread(str(R.CLIP / rows[i][1])), R.calibration())[0]
        return cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tobytes()
    return detect, walk, [jpeg(i, 90) for i in detect], [jpeg(i, 95) for i in walk]


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    detect, walk, det_jpegs, trk_jpegs = jpegs_on_raster()
    report = {"gpu_requested": GPU, "timeouts_s": {"detect": 900, "track": 1200}, "retries": 0, "words": WORDS,
              "detect_frames": detect, "walk": list(WALK), "upload_mb": {"detect": sum(map(len, det_jpegs)) / 2**20,
                                                                          "track": sum(map(len, trk_jpegs)) / 2**20}}
    with modal.enable_output(), app.run():
        spawned = time.time()
        d = detect_remote.spawn(det_jpegs, WORDS)
        t = track_remote.spawn(trk_jpegs)
        det = d.get()
        det_back = time.time()
        answer = t.get()
        trk_back = time.time()
    (args.output / "tracks-stride3.npz").write_bytes(answer)
    trk = json.loads(str(np.load(io.BytesIO(answer))["report"]))
    for name, r, back in (("detect", det, det_back), ("track", trk, trk_back)):
        billed_upper = back - spawned  # container boot + load + work + return; Modal bills container time, not queueing
        report[name + "_timing"] = {"client_wall_s": billed_upper, "in_container_s": r["ended"] - r["started"],
                                    "spawn_to_function_start_s": r["started"] - spawned, "load_s": r["load_s"],
                                    "usd_upper_gpu_list": billed_upper * USD_PER_S,
                                    "usd_upper_with_cpu_mem_E": billed_upper * (USD_PER_S + 2 * .0000131 + 16 * .00000222)}
    det_frames = det.pop("frames")
    ms = {k: [f["ms"][k] for f in det_frames] for k in det_frames[0]["ms"]}
    report["detect"] = {k: v for k, v in det.items() if k not in ("started", "ended")} | {
        "ms_p50": {k: float(np.median(v)) for k, v in ms.items()},
        "person_only_ms_per_frame_p50": float(np.median(np.add(ms["encode"], ms["person"]))),
        "all_words_ms_per_frame_p50": float(np.median(np.sum([ms[k] for k in ms], 0))),
        "batched_person_ms_per_image_p50": float(np.median(det["batched_person_ms_per_image"]))}
    report["track"] = {k: v for k, v in trk.items() if k not in ("started", "ended")}
    (args.output / "detections.json").write_text(json.dumps({"frames": dict(zip(map(str, detect), det_frames)), "words": WORDS}))
    (args.output / "probe.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k not in ("detect_frames",)}, indent=1)[:6000])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
