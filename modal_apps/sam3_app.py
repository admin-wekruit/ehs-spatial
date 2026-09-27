"""SAM 3 inference on Modal — self-hosted replacement for the fal endpoint.

Deploy:  uv run --with modal modal deploy modal_apps/sam3_app.py
Smoke:   uv run --with modal modal run modal_apps/sam3_app.py::main --image-path <jpg> --prompt fence
Latency: modal run modal_apps/sam3_app.py::bench --clip <clip dir with rgb/*.png> --out <json>
Check:   python modal_apps/sam3_app.py --self-check   (no GPU)

Why: per-call API pricing plus a burst-sensitive billing gate made the
segmentation stage both the cost and the reliability bottleneck (~$1-2
and several lockouts per full run). One warm L4 serves a whole run's
prompts in a single call for a few cents, with no third-party lock.

The response schema matches fal's `sam-3-1/image-rle` shape ({"rle":
[...], "scores": [...]}) so every existing consumer decodes unchanged;
RLE is the COCO object form our decode_coco_rle already reads.
"""

import hashlib
import io
import json
import math
import time
from pathlib import Path

import modal

app = modal.App("sam3-inference")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "torchvision",
        "transformers>=4.57",
        "accelerate",
        "pillow",
        "numpy",
    )
    .env({"HF_HOME": "/cache/huggingface"})
)

volume = modal.Volume.from_name("sam3-hf-cache", create_if_missing=True)

MODEL_ID = "facebook/sam3"


def _encode_coco_rle(mask) -> str:
    import numpy as np

    mask = np.asarray(mask).astype(bool)
    flat = mask.flatten(order="F").astype(np.int8)
    boundaries = np.concatenate(
        ([0], np.flatnonzero(np.diff(flat)) + 1, [flat.size])
    )
    counts = np.diff(boundaries).tolist()
    if flat.size and flat[0] == 1:
        counts = [0, *counts]
    return json.dumps(
        {
            "size": [int(mask.shape[0]), int(mask.shape[1])],
            "counts": [int(c) for c in counts],
        }
    )


def _response(results) -> tuple[list[str], list[float]]:
    """One post-processed instance result -> top-12 (RLEs, scores)."""
    masks = results.get("masks")
    scores = results.get("scores")
    if masks is None or len(masks) == 0:
        return [], []
    masks = masks.float().cpu().numpy()  # numpy has no bfloat16
    scores = [float(s) for s in scores.float().cpu().numpy()]
    order = sorted(range(len(scores)), key=lambda i: -scores[i])[:12]
    return (
        [_encode_coco_rle(masks[i] > 0.5) for i in order],
        [round(scores[i], 4) for i in order],
    )


# facebook/sam3 is a gated repo: the HF account behind the `huggingface`
# Modal secret must have accepted the license at
# https://huggingface.co/facebook/sam3 or load() 401s.
@app.cls(
    image=image,
    gpu="L4",
    volumes={"/cache": volume},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=600,
    retries=0,
    max_containers=1,
)
class Sam3:
    @modal.enter()
    def load(self):
        import torch
        from transformers import Sam3Model, Sam3Processor

        self.enter_at = time.time()
        t0 = time.time()
        self.torch = torch
        self.processor = Sam3Processor.from_pretrained(MODEL_ID)
        self.model = Sam3Model.from_pretrained(
            MODEL_ID, torch_dtype=torch.bfloat16
        ).to("cuda")
        self.model.eval()
        volume.commit()
        self.load_seconds = round(time.time() - t0, 1)
        self._last = None  # (sha1 of image bytes, image, vision embeds, original sizes)

    def _encode(self, image_bytes: bytes):
        """Decoded image + image-encoder output for these bytes. The encoder
        is most of a prompt's cost and does not depend on the prompt, so N
        prompts on one image pay it once, whether they come in one call or
        as N single-prompt calls (providers.sam3 sends one per call).
        ponytail: one-entry cache, correct while the container serves one
        input at a time (no allow_concurrent_inputs); key it per input if
        concurrency is ever turned on."""
        from PIL import Image

        key = hashlib.sha1(image_bytes).digest()
        if self._last is None or self._last[0] != key:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            base = self.processor(images=image, return_tensors="pt").to("cuda")
            with self.torch.inference_mode():
                vision = self.model.get_vision_features(
                    pixel_values=base["pixel_values"]
                )
            self._last = (key, image, vision, base["original_sizes"])
        return self._last[1:]

    def _segment_one(self, image, prompt: dict, vision=None, sizes=None):
        """One prompt -> masks + scores at full image resolution. With the
        cached encoder output only the prompt side runs; without it the
        whole model runs (the pre-cache path, kept to measure against)."""
        kwargs = {"return_tensors": "pt"}
        if vision is None:
            kwargs["images"] = image
        else:
            kwargs["original_sizes"] = sizes
        if prompt.get("text") is not None:
            kwargs["text"] = prompt["text"]
        if prompt.get("box") is not None:
            # absolute pixel xyxy box prompt
            kwargs["input_boxes"] = [[list(prompt["box"])]]
        inputs = self.processor(**kwargs).to("cuda")
        if vision is not None:
            inputs["vision_embeds"] = vision
        with self.torch.inference_mode():
            outputs = self.model(**inputs)
        return _response(
            self.processor.post_process_instance_segmentation(
                outputs,
                threshold=0.4,
                mask_threshold=0.5,
                target_sizes=[(image.height, image.width)],
            )[0]
        )

    def _segment(self, image_bytes: bytes, prompts: list[dict]) -> list[dict]:
        image, vision, sizes = self._encode(image_bytes)
        responses = []
        for prompt in prompts:
            try:
                rles, scores = self._segment_one(image, prompt, vision, sizes)
            except Exception as error:  # surface per-prompt, don't kill batch
                responses.append({"rle": [], "scores": [], "error": str(error)[:200]})
                continue
            responses.append({"rle": rles, "scores": scores})
        return responses

    @modal.method()
    def segment(self, image_bytes: bytes, prompts: list[dict]) -> list[dict]:
        """Batch endpoint: each prompt is {"text": str} or
        {"box": [x1, y1, x2, y2]}. Returns one fal-schema response per
        prompt: {"rle": [...], "scores": [...]}. One warm call serves a
        whole run's vocabulary; the image is encoded once per call."""
        return self._segment(image_bytes, prompts)

    @modal.method()
    def bench(self, frames: list[bytes], prompt_sets: list[list[str]], batch_sizes: list[int]) -> dict:
        """Warm server-side ms per image (JPEG decode -> RLE), per prompt
        set, for the pre-cache path (whole model per prompt) and the
        encoder-once path, interleaved per frame; counts encoder runs with
        a hook and compares the two paths' masks. Then batched throughput
        for the last prompt set. Measures, does not assert: a mismatch is
        reported, not hidden."""
        from PIL import Image

        torch = self.torch
        runs = [0]
        self.model.vision_encoder.register_forward_hook(
            lambda *_: runs.__setitem__(0, runs[0] + 1)
        )

        def timed(fn):
            torch.cuda.synchronize()
            t = time.perf_counter()
            out = fn()
            torch.cuda.synchronize()
            return out, (time.perf_counter() - t) * 1000

        def decode(jpeg):
            return Image.open(io.BytesIO(jpeg)).convert("RGB")

        def fresh(fn):
            self._last = None  # every timed image pays its own encode
            return fn()

        for jpeg in frames[:3]:  # kernel/allocator warm-up, untimed
            fresh(lambda: self._segment(jpeg, [{"text": "person"}]))
        report = {"gpu": torch.cuda.get_device_name(), "load_seconds": self.load_seconds,
                  "enter_at": self.enter_at, "frames": len(frames), "sets": {},
                  "dtype": str(self.model.dtype), "attn": self.model.config._attn_implementation}
        for words in prompt_sets:
            prompts = [{"text": w} for w in words]
            ms = {"per_prompt_encoder": [], "encode_once": [], "encoder_only": []}
            encoder_runs = {"per_prompt_encoder": 0, "encode_once": 0}
            same = score_diff = masks = 0
            for jpeg in frames:
                r0 = runs[0]
                old, t_old = timed(lambda: [self._segment_one(i, p) for i in [decode(jpeg)] for p in prompts])
                r1 = runs[0]
                new, t_new = timed(lambda: fresh(lambda: self._segment(jpeg, prompts)))
                r2 = runs[0]
                _, t_enc = timed(lambda: fresh(lambda: self._encode(jpeg)))
                ms["per_prompt_encoder"].append(t_old)
                ms["encode_once"].append(t_new)
                ms["encoder_only"].append(t_enc)
                encoder_runs["per_prompt_encoder"] += r1 - r0
                encoder_runs["encode_once"] += r2 - r1
                same += [list(o) for o in old] == [[n["rle"], n["scores"]] for n in new]
                for (_, s_old), n in zip(old, new):
                    masks += len(s_old)
                    if len(s_old) == len(n["scores"]):
                        score_diff = max([score_diff, *(abs(a - b) for a, b in zip(s_old, n["scores"]))])
                    else:
                        score_diff = max(score_diff, 1.0)
            report["sets"]["+".join(words)] = {
                "ms": ms, "encoder_runs_per_image": {k: v / len(frames) for k, v in encoder_runs.items()},
                "identical_images": same, "max_score_diff": score_diff, "masks_per_image": masks / len(frames),
            }
        words = prompt_sets[-1]
        report["batched"] = {}
        for b in batch_sizes:
            per_batch = []
            for start in range(0, len(frames) - b + 1, b):
                def run_batch(chunk=frames[start:start + b]):
                    images = [decode(j) for j in chunk]
                    base = self.processor(images=images, return_tensors="pt").to("cuda")
                    with torch.inference_mode():
                        vision = self.model.get_vision_features(pixel_values=base["pixel_values"])
                        for w in words:
                            inputs = self.processor(text=[w] * len(images), original_sizes=base["original_sizes"],
                                                    return_tensors="pt").to("cuda")
                            outputs = self.model(vision_embeds=vision, **inputs)
                            for result in self.processor.post_process_instance_segmentation(
                                    outputs, threshold=0.4, mask_threshold=0.5,
                                    target_sizes=[(i.height, i.width) for i in images]):
                                _response(result)
                per_batch.append(timed(run_batch)[1])
            report["batched"][str(b)] = {"prompts": words, "ms_per_batch": per_batch}
        report["peak_gpu_mb"] = round(torch.cuda.max_memory_allocated() / 2**20)
        return report


def latency_summary(ms: list[float], images_per_call: int = 1, utilization: float = 0.8) -> dict:
    """p50/p95 and how many 1 Hz cameras one worker keeps up with: the
    worker is sequential, so throughput is set by the mean, and it is run
    at `utilization` so queueing stays bounded."""
    import numpy as np

    a = np.asarray(ms, float)
    per_image = a.mean() / images_per_call
    return {
        "n": int(a.size), "p50_ms": round(float(np.percentile(a, 50)), 1),
        "p95_ms": round(float(np.percentile(a, 95)), 1), "mean_ms": round(float(a.mean()), 1),
        "images_per_s": round(1000 / per_image, 2),
        "cameras_at_1hz": math.floor(utilization * 1000 / per_image),
    }


def self_check():
    s = latency_summary([100.0] * 19 + [300.0])
    assert s["p50_ms"] == 100 and s["mean_ms"] == 110 and s["p95_ms"] == 110, s
    assert s["cameras_at_1hz"] == 7, "0.8 * 1000 / 110 ms = 7.27 -> 7 cameras"
    assert latency_summary([400.0] * 4, images_per_call=8)["cameras_at_1hz"] == 16, "8 images per 400 ms call"
    assert json.loads(_encode_coco_rle([[1, 0], [1, 1]]))["counts"] == [0, 2, 1, 1], "F-order, leading zero run"
    print("sam3_app latency math and RLE self-check passed; no GPU invoked")


@app.local_entrypoint()
def main(image_path: str, prompt: str = "fence"):
    result = Sam3().segment.remote(
        Path(image_path).read_bytes(), [{"text": prompt}]
    )
    first = result[0]
    print(
        f"{prompt!r}: {len(first.get('rle', []))} masks, "
        f"scores {first.get('scores', [])[:3]}, "
        f"error={first.get('error')}"
    )


@app.local_entrypoint()
def bench(clip: str, out: str, frames: int = 120, prompts: str = "person|person,forklift",
          batch_sizes: str = "4,8", rpc_frames: int = 20):
    """One ephemeral L4 container: cold call, in-container warm bench, then
    laptop round trips in the provider's one-prompt-per-call pattern."""
    import numpy as np
    from PIL import Image

    pngs = sorted(Path(clip, "rgb").glob("*.png"))
    picks = [pngs[i] for i in np.linspace(0, len(pngs) - 1, frames).round().astype(int)]
    jpegs = []
    for p in picks:
        buf = io.BytesIO()
        Image.open(p).convert("RGB").save(buf, "JPEG", quality=90)
        jpegs.append(buf.getvalue())
    prompt_sets = [s.split(",") for s in prompts.split("|")]
    sam = Sam3()
    t_call = time.time()
    t = time.perf_counter()
    sam.segment.remote(jpegs[0], [{"text": "person"}])
    cold_ms = (time.perf_counter() - t) * 1000
    t = time.perf_counter()
    report = sam.bench.remote(jpegs, prompt_sets, [int(b) for b in batch_sizes.split(",")])
    report["bench_wall_s"] = round(time.perf_counter() - t, 1)
    report["cold_first_call_ms"] = round(cold_ms, 1)
    report["cold_call_to_enter_s_approx"] = round(report["enter_at"] - t_call, 1)  # two clocks
    report["jpeg_kb_mean"] = round(sum(map(len, jpegs)) / len(jpegs) / 1024, 1)
    rpc = {"two_prompts_one_call": [], "one_prompt_per_call_x2": []}
    words = prompt_sets[-1]
    for jpeg in jpegs[3:3 + rpc_frames]:
        t = time.perf_counter()
        sam.segment.remote(jpeg, [{"text": w} for w in words])
        rpc["two_prompts_one_call"].append((time.perf_counter() - t) * 1000)
    for jpeg in jpegs[3 + rpc_frames:3 + 2 * rpc_frames]:
        t = time.perf_counter()
        for w in words:
            sam.segment.remote(jpeg, [{"text": w}])
        rpc["one_prompt_per_call_x2"].append((time.perf_counter() - t) * 1000)
    report["laptop_rpc_ms"] = rpc
    summary = {name: {k: latency_summary(v) for k, v in s["ms"].items()} | {
        k: s[k] for k in ("encoder_runs_per_image", "identical_images", "max_score_diff", "masks_per_image")}
        for name, s in report["sets"].items()}
    summary["batched"] = {b: latency_summary(v["ms_per_batch"], images_per_call=int(b))
                          for b, v in report["batched"].items()}
    summary["laptop_rpc"] = {k: latency_summary(v) for k, v in rpc.items()}
    report["summary"] = summary
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(report, indent=1))
    print(json.dumps({k: report[k] for k in ("gpu", "dtype", "attn", "load_seconds", "cold_first_call_ms", "cold_call_to_enter_s_approx",
                                             "bench_wall_s", "peak_gpu_mb", "jpeg_kb_mean", "summary")}, indent=1))


if __name__ == "__main__":
    import sys

    assert sys.argv[1:] == ["--self-check"], "usage: python modal_apps/sam3_app.py --self-check"
    self_check()
