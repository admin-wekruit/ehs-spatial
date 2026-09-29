"""E5(c): Qwen3-VL-8B events through vLLM, all 12 s windows in one batch, against modal_apps/video_events.py (transformers, one
window at a time). Same frames, same prompt, greedy. Also reruns the transformers path on the same GPU type for a like-for-like time
and for how much two runs of the same model agree at all (the ceiling for any agreement number). One A100-80GB per path, retries 0.

  python modal_apps/m3_e5/vllm_events.py --output NEW_DIR [--reference runs/me340-events-197] [--no-baseline]
  python modal_apps/m3_e5/vllm_events.py --self-check
"""
import argparse
import base64
import json
from pathlib import Path
import sys
import time

import modal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import video_events  # noqa: E402  prompt, windows, frame sampling, JSON parsing, the transformers path

ART = Path("/Users/adam/Desktop/panoptes-public/research-notes/phase2")
VLLM = "0.11.0"
GPU, MEMORY_SHARE = "A100-80GB", .45  # a share, not vLLM's 0.9 default: the plan puts gsplat and SAM 3 on the same card
USD_PER_S = .000694  # A100-80GB list price, modal.com/pricing (CPU and memory extra, small)
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install(f"vllm=={VLLM}", "transformers==4.57.1", "pillow")
         .env({"HF_HOME": "/cache/huggingface", "HF_HUB_OFFLINE": "1"})  # weights from the volume video_events.py filled
         .add_local_file(video_events.__file__, "/root/video_events.py", copy=True))  # baked in: a mount of it went missing in one container
app = modal.App("panoptes-m3-e5-vllm-events")


@app.function(image=image, gpu=GPU, timeout=900, retries=0, max_containers=1, volumes={"/cache": video_events.volume})
def vllm_remote(windows, model_id, max_new_tokens=900, eager=False):
    """All windows as one batch, twice: the first batch pays any lazy warm-up, the second is the steady state."""
    entered = time.time()
    import torch
    from vllm import LLM, SamplingParams
    started = time.perf_counter()
    llm = LLM(model=model_id, max_model_len=16384, limit_mm_per_prompt={"image": max(len(f) for _, _, f in windows), "video": 0},
              gpu_memory_utilization=MEMORY_SHARE, max_num_seqs=8, seed=0, enforce_eager=eager)
    loaded = time.perf_counter() - started
    conversations = []
    for t0, t1, frames in windows:
        content = []
        for t, data in frames:
            content += [{"type": "text", "text": f"[t = {t:.1f} s]"},
                        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(data).decode()}}]
        content.append({"type": "text", "text": video_events.PROMPT.format(n=len(frames), t0=t0, t1=t1)})
        conversations.append([{"role": "user", "content": content}])
    sampling = SamplingParams(temperature=0, max_tokens=max_new_tokens, seed=0)
    batches = []
    for _ in range(2):
        t = time.perf_counter()
        outs = llm.chat(conversations, sampling, use_tqdm=False)
        batches.append({"seconds": time.perf_counter() - t, "texts": [o.outputs[0].text for o in outs],
                        "prompt_tokens": [len(o.prompt_token_ids) for o in outs], "output_tokens": [len(o.outputs[0].token_ids) for o in outs]})
    return {"entered_unix": entered, "load_seconds": loaded, "batches": batches, "gpu": torch.cuda.get_device_name(),
            "peak_torch_gb": torch.cuda.max_memory_reserved() / 1e9, "vllm": VLLM, "eager": eager, "memory_share": MEMORY_SHARE,
            "container_seconds": time.time() - entered}


@app.function(image=video_events.image.add_local_file(video_events.__file__, "/root/video_events.py", copy=True), gpu=GPU, timeout=900, retries=0, max_containers=1,
              volumes={"/cache": video_events.volume}, secrets=[modal.Secret.from_name("huggingface")])
def transformers_remote(windows, model_id):
    """The existing transformers path (video_events.describe_remote), unchanged, on this GPU type."""
    return video_events.describe_remote.local(windows, model_id)


def tiou(a, b):
    inter = max(0., min(a["t1"], b["t1"]) - max(a["t0"], b["t0"]))
    union = max(a["t1"], b["t1"]) - min(a["t0"], b["t0"])
    return inter / union if union > 0 else 0.


def agreement(reference, candidate, threshold=.3):
    """Event recall/precision by temporal IoU >= threshold (greedy, best first), and field agreement on the matched pairs."""
    ref = [e for w in reference for e in (w.get("events") or []) if isinstance(e.get("t0"), (int, float)) and isinstance(e.get("t1"), (int, float))]
    cand = [e for w in candidate for e in (w.get("events") or []) if isinstance(e.get("t0"), (int, float)) and isinstance(e.get("t1"), (int, float))]
    pairs = sorted(((tiou(r, c), i, j) for i, r in enumerate(ref) for j, c in enumerate(cand)), reverse=True)
    used_r, used_c, matched = set(), set(), []
    for score, i, j in pairs:
        if score >= threshold and i not in used_r and j not in used_c:
            used_r.add(i), used_c.add(j), matched.append((ref[i], cand[j], score))
    same = lambda key: sum(r.get(key) == c.get(key) for r, c, _ in matched)
    ppe = sum((r.get("ppe") or {}) == (c.get("ppe") or {}) for r, c, _ in matched)
    words = lambda s: set((s or "").lower().replace(",", " ").replace(".", " ").split())
    captions = [len(words(r.get("caption")) & words(c.get("caption"))) / max(1, len(words(r.get("caption")) | words(c.get("caption"))))
                for r, c in zip(reference, candidate)]
    return {"reference_events": len(ref), "candidate_events": len(cand), "matched_tiou_ge_%.1f" % threshold: len(matched),
            "recall": round(len(matched) / len(ref), 3) if ref else None, "precision": round(len(matched) / len(cand), 3) if cand else None,
            "mean_tiou_matched": round(sum(s for _, _, s in matched) / len(matched), 3) if matched else None,
            "actor_same": same("actor"), "ppe_same": ppe, "safety_note_same": same("safety_note"),
            "caption_word_jaccard": [round(c, 3) for c in captions], "windows_same_count": len(reference) == len(candidate)}


def parsed(windows, texts):
    return [{"t0": t0, "t1": t1, **(video_events.parse(text) or {"caption": None, "events": [], "unparsed": text[:400]})}
            for (t0, t1, _), text in zip(windows, texts)]


def run(args):
    import cv2
    reference = json.loads((args.reference / "events.json").read_text())
    setup = json.loads(next(args.reference.glob("events-*.json")).read_text())  # the reference's own sampling
    video = Path(setup["video"])
    cap = cv2.VideoCapture(str(video))
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    windows = []
    for t0, t1 in video_events.bounds(duration, setup["window_s"]):
        frames = video_events.marked_frames(video, video_events.np.arange(t0, t1, 1 / setup["fps"]), None, {}, setup["width"])
        windows.append((float(t0), float(t1), [(t, data) for t, data, _ in frames]))
    assert len(windows) == len(reference["windows"]) and sum(len(w[2]) for w in windows) == setup["frames"], "not the reference's frames"
    args.output.mkdir(parents=True, exist_ok=False)
    out = {"reference": str(args.reference), "video": str(video), "windows": len(windows), "frames": setup["frames"], "gpu": GPU}
    with modal.enable_output(), app.run():
        calls = {"vllm": vllm_remote.spawn(windows, video_events.MODEL, eager=args.eager)}
        if args.baseline:  # the existing app, only its GPU type changed so both paths run on the same card
            calls["transformers"] = transformers_remote.spawn(windows, video_events.MODEL)
        submitted = time.time()
        results, failed = {}, {}
        for name, call in calls.items():
            try:
                results[name] = call.get()
            except Exception as error:  # one path failing must not lose the other's result
                failed[name] = repr(error)[:500]
    assert "vllm" in results, failed
    out["failed"] = failed
    returned = time.time()
    (args.output / "raw.json").write_text(json.dumps(results, ensure_ascii=False, indent=1))
    v = results["vllm"]
    out["vllm"] = {"load_seconds_M": round(v["load_seconds"], 1), "first_batch_seconds_M": round(v["batches"][0]["seconds"], 2),
                   "warm_batch_seconds_M": round(v["batches"][1]["seconds"], 2), "prompt_tokens": v["batches"][1]["prompt_tokens"],
                   "output_tokens": v["batches"][1]["output_tokens"], "container_seconds_M": round(v["container_seconds"], 1),
                   "submit_to_entry_seconds_M": round(v["entered_unix"] - submitted, 1), "gpu": v["gpu"], "peak_torch_reserved_gb": round(v["peak_torch_gb"], 1),
                   "memory_share": v["memory_share"], "eager": v["eager"], "vllm": v["vllm"],
                   "batches_identical": v["batches"][0]["texts"] == v["batches"][1]["texts"]}
    events = {"vllm": parsed(windows, v["batches"][1]["texts"])}
    if "transformers" in results:
        b = results["transformers"]
        out["transformers_same_gpu"] = {"load_seconds_M": round(b["load_seconds"], 1), "generate_seconds_M": round(b["seconds"] - b["load_seconds"], 2),
                                        "gpu": b["gpu"], "prompt_tokens": [w[3] for w in b["windows"]]}
        events["transformers_same_gpu"] = parsed(windows, [w[2] for w in b["windows"]])
    out["reference_197"] = {"load_seconds_M": round(reference["load_seconds"], 1), "generate_seconds_M": round(reference["seconds"] - reference["load_seconds"], 2),
                            "gpu": reference["gpu"], "prompt_tokens": [w.get("promptTokens") for w in reference["windows"]]}
    out["agreement_with_197"] = {name: agreement(reference["windows"], e) for name, e in events.items()}
    if "transformers_same_gpu" in events:
        out["agreement_vllm_vs_transformers_same_gpu"] = agreement(events["transformers_same_gpu"], events["vllm"])
    out["client_wall_seconds_M"] = round(returned - submitted, 1)
    out["usd_estimate"] = round(USD_PER_S * (v["container_seconds"] + 30 + (results["transformers"]["seconds"] + 30 if "transformers" in results else 0)), 3)
    for name, e in events.items():
        (args.output / f"events-{name}.json").write_text(json.dumps(e, ensure_ascii=False, indent=1))
    (args.output / "summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps(out, ensure_ascii=False, indent=1))


def self_check():
    a = {"t0": 0., "t1": 10.}
    assert tiou(a, {"t0": 5., "t1": 15.}) == 5 / 15 and tiou(a, {"t0": 20., "t1": 30.}) == 0
    ref = [{"caption": "a man walks", "events": [a, {"t0": 12., "t1": 20., "actor": "x"}]}]
    got = agreement(ref, [{"caption": "a man stands", "events": [{"t0": 1., "t1": 9.}]}])
    assert got["recall"] == .5 and got["precision"] == 1. and got["caption_word_jaccard"] == [.5], got
    print("vllm events check passed: temporal matching and agreement counts hold")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--reference", type=Path, default=ART / "runs/me340-events-197")
    parser.add_argument("--no-baseline", dest="baseline", action="store_false", help="skip the transformers rerun on the same GPU type")
    parser.add_argument("--eager", action="store_true", help="vLLM enforce_eager: no CUDA-graph capture (faster start, slower decode)")
    parser.add_argument("--output", type=Path)
    a = parser.parse_args()
    self_check() if a.self_check else run(a)
