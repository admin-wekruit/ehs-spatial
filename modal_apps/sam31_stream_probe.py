"""Can self-hosted SAM 3.1 follow people live, one frame at a time, with a memory that stops growing?

sam3_motion_tracks.py writes a window of JPEGs to a folder and opens an offline, two-way session on it; a camera
cannot do that. Reading the model at sam3_video.SOURCE_COMMIT: the multiplex model already detects and tracks frame by
frame inside propagate_in_video and reads each frame through `img_batch[i]`, the hook its own async folder loader
uses. So a live feed is a frame list that hands out frame i once frame i has arrived. Three things stand in the way:
  - batched grounding runs the detector on 16 frames at once, 16 frames of look-ahead: batch size 1 here;
  - outputs are post-processed 16 at a time: 1 here. The 15-frame hot-start delay stays, it is the algorithm;
  - every per-frame store (tracker memory, conditioning frames, cached masks, frame-wise scores) keeps every frame
    forever. prune() drops what the forward pass can no longer read.
One session loops a clip on one GPU for a fixed time. Reported: throughput, per-frame latency, GPU memory over time
(unpruned first, then pruned), and identity agreement with offline tracks of the same frames. Timeout 2100 s, retries 0.

  modal run modal_apps/sam31_stream_probe.py --droid-run RUN --output NEW_DIR [--minutes 25] [--unpruned 1798]
      [--reference-tracks SAM3_TRACKS_RUN,...] [--reference-analysis MOTION_ANALYSIS_RUN] [--compile]
  python modal_apps/sam31_stream_probe.py --self-check
"""
import io
import json
from pathlib import Path
import sys
import time

import modal
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import sam3_video  # noqa: E402  the pinned SAM 3.1 source commit, weights revision, image and weight cache

KEEP_FRAMES = 8  # decoded frames a live feed holds; the model asking for an older one counts as a stale read
WINDOW = 64  # prune per-frame state older than this: the tracker reads 6 memory frames, 15 pointer frames, 15 frames of hot start
KEEP_COND = 4  # max_cond_frames_in_attn at SOURCE_COMMIT: going forward, the latest 4 conditioning frames are the closest 4
MAX_FPS = 30  # session length is sized for this rate; faster ends the run early, which the report shows
SAMPLE, CENSUS = 100, 3000  # memory sample and state census every this many answered frames
QUARTER = (slice(2, None, 4), slice(2, None, 4))  # 480x640 masks kept as 120x160 samples: enough to match people by IoU
IOU, WARM = .5, 100  # a stream mask is the reference object at this IoU; latency percentiles skip this many first frames
TIMEOUT = 2100
app = modal.App("panoptes-sam31-stream-probe-once")
image = sam3_video.video_image.add_local_python_source("sam3_video")


class LiveFrames:
    """The session's frame list for a live feed: frame i exists once it has arrived, and only the last KEEP_FRAMES stay.

    A camera thread decodes the looped clip into a short queue; the model takes frame i when it asks for it, never
    earlier. A request for a frame not yet arrived (look-ahead) or already dropped (stale) is served but counted,
    because either one means the session is not really forward-only.
    """

    def __init__(self, decode, count, capacity):
        import queue
        import threading
        self.decode, self.count, self.capacity = decode, count, capacity
        self.store, self.arrived, self.next, self.lookahead, self.stale = {}, {}, 0, 0, 0
        self.feed = queue.Queue(maxsize=4)
        threading.Thread(target=self._camera, daemon=True).start()

    def _camera(self):
        n = 0
        while True:
            self.feed.put(self.decode(n % self.count))
            n += 1

    def __len__(self):
        return self.capacity

    def __getitem__(self, i):
        i = int(i)
        if i in self.store:
            return self.store[i]
        if i != self.next:
            self.lookahead += i > self.next
            self.stale += i < self.next
            return self.decode(i % self.count)
        self.store[i], self.arrived[i] = self.feed.get(), time.perf_counter()
        self.next += 1
        self.store.pop(i - KEEP_FRAMES, None)
        return self.store[i]


def prune(state, frontier, window=WINDOW, keep_cond=KEEP_COND):
    """Forget per-frame entries the forward pass can no longer read; returns how many went.

    Tracking frame t reads non-conditioning memory from t-1..t-6, object pointers from t-1..t-15 and the 4
    conditioning frames closest in time (video_tracking_multiplex.py, select_closest_cond_frames); going forward those
    are the latest 4. Inputs and 'consolidated' frame sets are cut by the same frames because
    propagate_in_video_preflight asserts they match.
    """
    old, gone = frontier - window, 0

    def drop(store, keep=()):
        nonlocal gone
        for f in [f for f in store if f < old and f not in keep]:
            store.remove(f) if isinstance(store, set) else store.pop(f)
            gone += 1

    drop(state["cached_frame_outputs"])
    meta = state["tracker_metadata"]
    drop(meta.get("obj_id_to_sam2_score_frame_wise", {}))
    drop(meta.get("rank0_metadata", {}).get("suppressed_obj_ids", {}))
    for s in state["sam2_inference_states"]:
        keep = set(sorted(s["output_dict"]["cond_frame_outputs"])[-keep_cond:])
        stores = [s["frames_already_tracked"], *s["consolidated_frame_inds"].values(),
                  *s["point_inputs_per_obj"].values(), *s["mask_inputs_per_obj"].values()]
        for outputs in (s["output_dict"], *s["output_dict_per_obj"].values(), *s["temp_output_dict_per_obj"].values()):
            stores += list(outputs.values())
        for store in stores:
            drop(store, keep)
    return gone


def census(tree, path="", out=None, seen=None, depth=0):
    """{path: [entries, tensor bytes]} over the session state; integer keys (frames, objects) fold into '#' so a growing store shows by name."""
    out, seen = ({}, set()) if out is None else (out, seen)
    if hasattr(tree, "untyped_storage"):
        storage = tree.untyped_storage()
        if storage.data_ptr() not in seen:
            seen.add(storage.data_ptr())
            out.setdefault(path, [0, 0])[1] += storage.nbytes()
        return out
    if not isinstance(tree, (dict, list, tuple, set)) or depth > 8:
        return out
    out.setdefault(path, [0, 0])[0] += len(tree)
    if isinstance(tree, set) or len(tree) > 5000:  # frame-sized fixed lists: their length is the session's, not growth
        return out
    for key, value in (tree.items() if isinstance(tree, dict) else enumerate(tree)):
        census(value, f"{path}/{key if isinstance(key, str) else '#'}", out, seen, depth + 1)
    return out


@app.function(image=image, gpu=["H100", "A100-80GB"], timeout=TIMEOUT, retries=0, max_containers=1,
              volumes={"/cache": sam3_video.cache}, secrets=[modal.Secret.from_name("huggingface")])
def stream_remote(jpegs, seconds, unpruned, text, compile_model=False):
    """jpegs: one clip pass, looped. -> npz bytes: report + one row per (answered frame, object) + timing and memory series."""
    import psutil
    import torch
    from PIL import Image
    from huggingface_hub import hf_hub_download
    import sam3.model.sam3_multiplex_tracking as tracking
    from sam3.model.io_utils import _load_img_as_tensor
    from sam3.model_builder import build_sam3_multiplex_video_predictor
    report, began = {"timings": {}}, time.perf_counter()
    checkpoint = hf_hub_download(sam3_video.MODEL_ID, sam3_video.CHECKPOINT, revision=sam3_video.MODEL_REVISION)
    sam3_video.cache.commit()
    # compile: the model's own torch.compile path and its warm-up on a dummy video, both already in SOURCE_COMMIT; FA3 is not in the image
    predictor = build_sam3_multiplex_video_predictor(checkpoint_path=checkpoint, max_num_objects=16, multiplex_count=16, use_fa3=False, compile=compile_model, warm_up=False)
    model = predictor.model
    report.update(gpu=torch.cuda.get_device_name(), torch=torch.__version__, compile=compile_model, use_fa3=False,
                  hotstart_delay=model.hotstart_delay, grounding_batch_was=model.batched_grounding_batch_size, postprocess_batch_was=model.postprocess_batch_size)
    model.batched_grounding_batch_size = 1  # the detector sees frame t alone: no look-ahead
    model.postprocess_batch_size = 1  # an answer per frame, not per 16
    report["timings"]["weights_and_build"] = time.perf_counter() - began
    if compile_model:  # Sam3MultiplexVideoPredictor's own warm-up, run after the batch sizes change so it compiles the shapes a live feed uses
        warm = time.perf_counter()
        model._warm_up_complete = False
        model.warm_up_compilation()
        model._warm_up_complete = True
        report["timings"]["compile_warm_up"] = time.perf_counter() - warm

    size, width, height = model.image_size, *Image.open(io.BytesIO(jpegs[0])).size
    mean = std = torch.tensor((.5, .5, .5), dtype=torch.float16)[:, None, None]

    def decode(n):  # exactly AsyncImageFrameLoader.__getitem__ with offload_video_to_cpu, from bytes instead of a path
        img = _load_img_as_tensor(io.BytesIO(jpegs[n]), size)[0].to(dtype=torch.float16)
        img -= mean
        img /= std
        return img

    # ponytail: the loader is swapped for the live list at module level; this container runs one session and nothing else
    tracking.load_resource_as_video_frames = lambda resource_path, **_: (resource_path, height, width)
    tracking.is_image_type = lambda resource_path: False
    tracking.tqdm = lambda iterable, **_: iterable  # a progress bar sized to the session floods the log
    frames = LiveFrames(decode, len(jpegs), int(seconds * MAX_FPS) + 1000)
    stats = {"dropped": 0, "tracked_max": 0}
    inner = model._run_single_frame_inference

    def spy(*args, **kwargs):  # the model's own per-frame counts: objects tracked, and new ones refused for lack of slots
        out = inner(*args, **kwargs)
        frame_stats = out.get("frame_stats") or {}
        stats["dropped"] += int(frame_stats.get("num_obj_dropped", 0))
        stats["tracked_max"] = max(stats["tracked_max"], int(frame_stats.get("num_obj_tracked", 0)))
        return out

    model._run_single_frame_inference = spy
    started = time.perf_counter()
    state = model.init_state(resource_path=frames, offload_video_to_cpu=True)
    session = sam3_video._register(predictor, state)
    report["timings"]["init_state"] = time.perf_counter() - started
    report.update(session_frames=len(frames), allocated_after_init=torch.cuda.memory_allocated(), mask_shape=list(np.zeros((height, width), bool)[QUARTER].shape))
    rows = {"frame": [], "object": [], "prob": [], "box": [], "mask": []}
    series = {"answered": [], "period_s": [], "model_s": [], "latency_s": []}
    memory, censuses, pruned = [], {}, 0
    torch.cuda.reset_peak_memory_stats()
    try:
        began_prompt = time.perf_counter()
        predictor.handle_request({"type": "add_prompt", "session_id": session, "frame_index": 0, "text": text})
        report["timings"]["add_prompt"] = time.perf_counter() - began_prompt
        stream = iter(predictor.handle_stream_request({"type": "propagate_in_video", "session_id": session, "propagation_direction": "forward", "start_frame_index": 0}))
        started = last = time.perf_counter()
        while True:
            asked = time.perf_counter()
            response = next(stream, None)
            if response is None:
                break
            series["model_s"].append(time.perf_counter() - asked)  # inside the model's generator only
            y, out = int(response["frame_index"]), response["outputs"]
            ids = np.asarray(out["out_obj_ids"]).astype(np.int32)
            masks = np.asarray(out["out_binary_masks"]).astype(bool)
            for n, ident in enumerate(ids):
                rows["frame"].append(y)
                rows["object"].append(int(ident))
                rows["prob"].append(float(np.asarray(out["out_probs"])[n]))
                rows["box"].append(np.asarray(out["out_boxes_xywh"], np.float32)[n])
                rows["mask"].append(np.packbits(masks[n][QUARTER]))
            if frames.next > unpruned:
                pruned += prune(state, frames.next - 1)
            now = time.perf_counter()
            series["answered"].append(y)
            series["period_s"].append(now - last)  # the whole cost of one more frame: model, answer, pruning
            series["latency_s"].append(now - frames.arrived.pop(y, np.nan))  # frame in -> its masks out, hot start included
            last = now
            if y % SAMPLE == 0:
                memory.append([y, frames.next - 1, now - started, torch.cuda.memory_allocated(), torch.cuda.memory_reserved(),
                               psutil.Process().memory_info().rss, len(state["tracker_metadata"].get("obj_ids_all_gpu", []))])
            if y % CENSUS == 0:
                censuses[y] = {k: v for k, v in census(state).items() if v[0] > 20 or v[1] > 2 ** 20}
            if now - started > seconds:
                break
        stream.close()
        report["stream_seconds"] = time.perf_counter() - started
    except Exception as error:  # the frames already answered are still worth the call
        report["error"] = f"{type(error).__name__}: {error}"[:800]
        report["stream_seconds"] = time.perf_counter() - started
    report.update(entries_pruned=pruned, lookahead_reads=frames.lookahead, stale_reads=frames.stale, peak_allocated=torch.cuda.max_memory_allocated(),
                  objects_dropped_for_slots=stats["dropped"], objects_tracked_max=stats["tracked_max"], census=censuses, remote_seconds=time.perf_counter() - began)
    buffer = io.BytesIO()
    np.savez_compressed(buffer, report=json.dumps(report), memory=np.array(memory, np.float64).reshape(-1, 7),
                        **{k: np.array(v, np.float64) for k, v in series.items()},
                        row_frame=np.array(rows["frame"], np.int32), row_object=np.array(rows["object"], np.int32),
                        row_prob=np.array(rows["prob"], np.float32), row_box=np.array(rows["box"], np.float32).reshape(-1, 4),
                        row_mask=np.array(rows["mask"], np.uint8).reshape(-1, np.packbits(np.zeros((height, width), bool)[QUARTER]).size))
    return buffer.getvalue()


def agreement(reference, stream, frames):
    """Identity agreement on `frames`: {frame: {id: mask}} each. Per frame, masks pair one to one at IoU >= IOU; then one
    stream id per reference id by the most frames together. idf1 counts every miss, extra and switch;
    id_consistency only asks whether matched masks carry the mapped id."""
    from collections import Counter
    from scipy.optimize import linear_sum_assignment
    together, matched, ref_n, stream_n = Counter(), 0, 0, 0
    for f in frames:
        here, there = reference.get(f, {}), stream.get(f, {})
        ref_n, stream_n = ref_n + len(here), stream_n + len(there)
        if not here or not there:
            continue
        a, b = list(here), list(there)
        iou = np.array([[(here[i] & there[j]).sum() / max((here[i] | there[j]).sum(), 1) for j in b] for i in a])
        for m, n in zip(*linear_sum_assignment(-iou)):
            if iou[m, n] >= IOU:
                together[a[m], b[n]] += 1
                matched += 1
    refs, ours = sorted({i for i, _ in together}), sorted({j for _, j in together})
    counts = np.array([[together[i, j] for j in ours] for i in refs]).reshape(len(refs), len(ours))
    rows, cols = linear_sum_assignment(-counts) if counts.size else ((), ())
    idtp = int(sum(counts[r, c] for r, c in zip(rows, cols)))
    return {"frames": len(frames), "reference_objects": ref_n, "stream_objects": stream_n, "matched": matched,
            "detection_recall": matched / max(ref_n, 1), "id_consistency": idtp / max(matched, 1), "idf1": 2 * idtp / max(ref_n + stream_n, 1),
            "mapping": {str(refs[r]): int(ours[c]) for r, c in zip(rows, cols)}, "stream_ids": sorted({j for f in frames for j in stream.get(f, {})})}


def passes(found, count, shape):
    """Answered frames per pass through the looped clip: {pass: {source frame: {stream id: quarter mask}}}; empty frames included."""
    out = {}
    for y in found["answered"].astype(int):
        out.setdefault(y // count, {})[y % count] = {}
    for y, ident, packed in zip(found["row_frame"], found["row_object"], found["row_mask"]):
        out[int(y) // count][int(y) % count][int(ident)] = np.unpackbits(packed)[:shape[0] * shape[1]].reshape(shape).astype(bool)
    return out


def load_references(tracks_runs, analysis_run):
    """The offline answers on the same raster, as quarter masks: each sam3_motion_tracks window's own text stage, and the linked motion analysis."""
    import cv2
    refs = {}
    for tracks_run in tracks_runs:
        from stitch_track_windows import load
        window = load(tracks_run)
        state = json.loads((tracks_run / "tracks.json").read_text())
        first, shown = state["frames"][0], state["tracked_frames"] * state.get("stride", 1)
        refs[f"window {tracks_run.name}"] = ({f: {i: window.mask(k)[QUARTER] for i, k in objects.items() if i.startswith("text/")} for f, objects in window.frames.items()},
                          range(first, first + shown), str(tracks_run))
    if analysis_run:
        analysis = json.loads((analysis_run / "analysis.json").read_text())
        masks = {row["sourceFrame"]: {o["entityId"]: cv2.imread(str(analysis_run / o["maskUrl"]), 0)[QUARTER] > 0 for o in row["objects"]} for row in analysis["frames"]}
        frames = [row["sourceFrame"] for row in analysis["frames"]]
        refs["analysis"] = (masks, frames, str(analysis_run))
        # the linked offline text tracks alone: what the same prompt found offline, without the motion-seeded objects
        refs["analysis_text"] = ({f: {e: m for e, m in objects.items() if "-text-" in e} for f, objects in masks.items()}, frames, str(analysis_run) + " text tracks")
    return refs


def summarize(found, refs, count, unpruned):
    """Throughput, latency, memory trend and identity agreement from one stream answer."""
    report = json.loads(str(found["report"]))
    period, latency = found["period_s"][WARM:], found["latency_s"][WARM:]
    memory = found["memory"]
    pct = lambda values, q: float(np.nanpercentile(values, q)) if len(values) else None
    out = {"gpu": report.get("gpu"), "compile": report.get("compile"), "error": report.get("error"), "answered_frames": int(len(found["answered"])), "passes": int(len(found["answered"]) / count),
           "stream_seconds": report.get("stream_seconds"), "fps_overall": len(found["answered"]) / report["stream_seconds"] if report.get("stream_seconds") else None,
           "fps_steady": len(period) / period.sum() if len(period) else None, "model_ms_p50": pct(found["model_s"][WARM:], 50) * 1e3 if "model_s" in found.files and len(period) else None,
           "period_ms_p50": pct(period, 50) * 1e3 if len(period) else None, "period_ms_p95": pct(period, 95) * 1e3 if len(period) else None,
           "latency_ms_p50": pct(latency, 50) * 1e3 if len(latency) else None, "latency_ms_p95": pct(latency, 95) * 1e3 if len(latency) else None,
           "hotstart_delay_frames": report.get("hotstart_delay"), "lookahead_reads": report.get("lookahead_reads"), "stale_reads": report.get("stale_reads"),
           "objects_tracked_max": report.get("objects_tracked_max"), "objects_dropped_for_slots": report.get("objects_dropped_for_slots"),
           "peak_allocated_gb": report.get("peak_allocated", 0) / 2 ** 30, "timings": report.get("timings"), "remote_seconds": report.get("remote_seconds")}
    gb = lambda rows: rows[:, 3] / 2 ** 30
    grow = memory[(memory[:, 1] < unpruned) & (memory[:, 1] >= WARM)] if len(memory) else memory
    flat = memory[memory[:, 1] >= unpruned + count] if len(memory) else memory  # one pass after pruning starts, past the drop
    if len(grow) > 2:
        out["unpruned_gb_per_1000_frames"] = float(np.polyfit(grow[:, 1], gb(grow), 1)[0] * 1000)
    if len(flat) > 10:
        tenth = max(len(flat) // 10, 1)
        first, last = gb(flat[:tenth]).mean(), gb(flat[-tenth:]).mean()
        out.update(pruned_gb_first_tenth=float(first), pruned_gb_last_tenth=float(last), pruned_growth=float(last / first - 1),
                   pruned_gb_per_1000_frames=float(np.polyfit(flat[:, 1], gb(flat), 1)[0] * 1000),
                   pruned_rss_gb_first_last=[float(flat[:tenth, 5].mean() / 2 ** 30), float(flat[-tenth:, 5].mean() / 2 ** 30)])
    streams = passes(found, count, tuple(report["mask_shape"]))
    out["agreement"] = {name: {int(p): agreement(masks, streams[p], [f for f in frames if f in streams[p]]) for p in sorted(streams)}
                        for name, (masks, frames, _) in refs.items()}
    return out


@app.local_entrypoint()
def main(droid_run: str, output: str, minutes: float = 25, unpruned: int = 1798, text: str = "person",
         reference_tracks: str = "", reference_analysis: str = "", frames: int = 0, compile: bool = False):
    import cv2
    import mono_room as M
    droid_run, output = Path(droid_run), Path(output)
    timeout = int(minutes * 60) + 300 + 780 * compile  # start-up (about 25 s measured), and the compile warm-up: a warm-up longer than that is itself the answer
    if not 0 < minutes or timeout > TIMEOUT:
        raise ValueError(f"--minutes plus start-up must fit the {TIMEOUT} s timeout")
    M.use_clip(droid_run)
    manifest = json.loads((droid_run / "input-manifest.json").read_text())
    rows = manifest["frames"][:frames or None]
    refs = load_references([Path(r) for r in reference_tracks.split(",") if r], Path(reference_analysis) if reference_analysis else None)
    output.mkdir(parents=True, exist_ok=True)  # an output holding stream.npz is scored again, not paid for again
    state = {"model": sam3_video.MODEL_ID, "revision": sam3_video.MODEL_REVISION, "source_commit": sam3_video.SOURCE_COMMIT, "cameras": str(droid_run),
             "clip_frames": len(rows), "minutes": minutes, "unpruned_frames": unpruned, "text": text, "gpu": "H100, fallback A100-80GB", "compile": compile,
             "timeout_s": timeout, "retries": 0, "window": WINDOW, "keep_cond": KEEP_COND, "references": {k: v[2] for k, v in refs.items()}}
    (output / f"stream-{int(time.time())}.json").write_text(json.dumps(state, indent=1))
    began = time.time()
    if (output / "stream.npz").exists():
        answer = (output / "stream.npz").read_bytes()
    else:
        # the same frames sam3_motion_tracks.py tracked offline: same raster, same JPEG quality
        images = [M.prepare_image(cv2.imread(str(M.DATASET / r["relative_path"])), M.CALIBRATION, 2)[0] for r in rows]
        jpegs = [cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes() for f in images]
        answer = stream_remote.with_options(timeout=timeout).remote(jpegs, minutes * 60, unpruned, text, compile)
        (output / "stream.npz").write_bytes(answer)  # the provider's answer, before any local reading
        state["client_wall_seconds"] = time.time() - began
    found = np.load(io.BytesIO(answer))
    state["result"] = summarize(found, refs, len(rows), unpruned)
    state["census"] = json.loads(str(found["report"])).get("census")
    (output / "stream.json").write_text(json.dumps(state, indent=1))
    result = dict(state["result"])
    agreement = result.pop("agreement")
    print(json.dumps(result, indent=1))
    print("agreement per pass: [detection recall, id consistency, idf1, stream ids]")
    for name, by_pass in agreement.items():
        print(name, json.dumps({p: [round(a["detection_recall"], 3), round(a["id_consistency"], 3), round(a["idf1"], 3), len(a["stream_ids"])] for p, a in by_pass.items() if a["frames"]}))


def self_check():
    frames = LiveFrames(lambda n: n, 5, 100)
    assert len(frames) == 100 and [frames[i] for i in range(12)] == [i % 5 for i in range(12)], "frames arrive in order and the clip loops"
    assert frames.lookahead == frames.stale == 0 and set(frames.store) == set(range(12 - KEEP_FRAMES, 12)), "only the last KEEP_FRAMES are held"
    frames[11], frames[13], frames[1]
    assert (frames.lookahead, frames.stale) == (1, 1), "a frame not yet arrived, or already dropped, is counted"

    def tracker(cond, non_cond):
        return {"output_dict": {"cond_frame_outputs": dict.fromkeys(cond), "non_cond_frame_outputs": dict.fromkeys(non_cond)},
                "output_dict_per_obj": {0: {"cond_frame_outputs": dict.fromkeys(cond), "non_cond_frame_outputs": dict.fromkeys(non_cond)}},
                "temp_output_dict_per_obj": {0: {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}},
                "point_inputs_per_obj": {0: {}}, "mask_inputs_per_obj": {0: dict.fromkeys(cond)}, "frames_already_tracked": dict.fromkeys(non_cond),
                "consolidated_frame_inds": {"cond_frame_outputs": set(cond), "non_cond_frame_outputs": set()}}
    cond, non_cond = [0, 16, 32, 48, 64, 150, 190], [f for f in range(200) if f not in (0, 16, 32, 48, 64, 150, 190)]
    state = {"cached_frame_outputs": dict.fromkeys(range(200)), "sam2_inference_states": [tracker(cond, non_cond)],
             "tracker_metadata": {"obj_id_to_sam2_score_frame_wise": dict.fromkeys(range(200)), "rank0_metadata": {"suppressed_obj_ids": dict.fromkeys(range(200))}}}
    assert prune(state, 199, window=64, keep_cond=4) > 0
    s = state["sam2_inference_states"][0]
    assert sorted(s["output_dict"]["cond_frame_outputs"]) == [48, 64, 150, 190], "old conditioning frames go, the latest 4 stay"
    assert min(s["output_dict"]["non_cond_frame_outputs"]) == 135 and min(state["cached_frame_outputs"]) == 135, "memory the tracker still reads stays"
    inputs = set(s["mask_inputs_per_obj"][0]) | set(s["point_inputs_per_obj"][0])
    assert set().union(*s["consolidated_frame_inds"].values()) == inputs, "preflight's invariant: consolidated frames are the input frames"
    assert prune(state, 199, window=64, keep_cond=4) == 0, "pruning twice changes nothing"

    a, b = np.zeros((12, 16), bool), np.zeros((12, 16), bool)
    a[2:8, 1:4], b[2:8, 9:12] = True, True
    reference = {f: {"p": a, "q": b} for f in range(10)}
    same = agreement(reference, {f: {7: a, 9: b} for f in range(10)}, range(10))
    assert same["idf1"] == same["id_consistency"] == 1 and same["mapping"] == {"p": 7, "q": 9}, "renamed ids agree fully"
    switched = agreement(reference, {f: ({7: a, 9: b} if f < 6 else {9: a, 7: b}) for f in range(10)}, range(10))
    assert switched["detection_recall"] == 1 and abs(switched["id_consistency"] - .6) < 1e-9, "an id switch after 6 of 10 frames agrees 60%"
    missed = agreement(reference, {f: {7: a} for f in range(10)}, range(10))
    assert missed["id_consistency"] == 1 and abs(missed["idf1"] - 2 * 10 / 30) < 1e-9, "a missed person costs idf1, not consistency"
    found = {"answered": np.arange(12.), "row_frame": np.array([0, 11]), "row_object": np.array([3, 4]),
             "row_mask": np.array([np.packbits(np.ones((480, 640), bool)[QUARTER])] * 2)}
    split = passes(found, 5, (120, 160))
    assert sorted(split) == [0, 1, 2] and split[2][1][4].all() and split[1][3] == {}, "frames split into passes, empty frames kept"
    print("stream probe check passed: live frames in order, pruning keeps what the tracker reads, ids agree only when they should")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    parser.parse_args()
    self_check()
