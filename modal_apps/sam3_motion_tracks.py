"""What moved, tracked through the clip by self-hosted SAM 3.1: seeded by motion, not by words.

motion_masks.py finds, pair by pair, the regions whose image motion the camera cannot explain. That cue is precise but
momentary: a standing person moves only an arm, a seated one nothing. Here each such region seeds SAM 3.1's instance
tracker (points + object id, the SAM 2 style prompt of Sam3MultiplexVideoPredictor), which follows the whole object
through every frame in both directions: moved once, dynamic for the clip. Seeds are taken largest first and a seed already
covered by a tracked object is skipped, so one person does not become ten objects.
For comparison the same call also runs SAM 3.1's own text prompt (default "person") in a second session.
One A100 call, timeout 900 s, retries 0; every stage reports its own error instead of losing the others.

  python modal_apps/sam3_motion_tracks.py --droid-run RUN --motion MOTION_RUN --frames 500 800 --output NEW_DIR [--reference MASK_DIR]
  python modal_apps/sam3_motion_tracks.py --self-check
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
import sam3_video  # noqa: E402  the pinned SAM 3.1 source commit, weights revision, image and weight cache

MIN_SEED, COVERED, MAX_OBJECTS, POINTS = .02, .5, 8, 3  # seed area as a share of the frame; a seed this much inside a track is known; tracker slots; clicks per seed
MOVING_SHARE = .25  # a track is dynamic if, at the median sampled frame, at least this share of its pixels carries unexplained motion (a false seed returns the floor under it: 0)
app = modal.App("panoptes-sam3-motion-tracks-once")
image = sam3_video.video_image.add_local_python_source("sam3_video")


@app.function(image=image, gpu="A100-40GB", timeout=900, retries=0, max_containers=1, volumes={"/cache": sam3_video.cache},
              secrets=[modal.Secret.from_name("huggingface")])
def track_remote(jpegs, seeds, text):
    """jpegs: frames in order. seeds: [(frame, [[x, y] relative, ...], packed seed mask)] largest first. -> npz bytes of packed masks + report."""
    import tempfile
    import torch
    from huggingface_hub import hf_hub_download
    from sam3.model_builder import build_sam3_multiplex_video_predictor
    report, started = {"stages": {}}, time.perf_counter()
    checkpoint = hf_hub_download(sam3_video.MODEL_ID, sam3_video.CHECKPOINT, revision=sam3_video.MODEL_REVISION)
    sam3_video.cache.commit()
    report["weights_seconds"] = time.perf_counter() - started
    predictor = build_sam3_multiplex_video_predictor(checkpoint_path=checkpoint, max_num_objects=16, multiplex_count=16, use_fa3=False, compile=False, warm_up=False)
    report["gpu"] = torch.cuda.get_device_name()
    arrays = {}
    with tempfile.TemporaryDirectory() as folder:
        for n, data in enumerate(jpegs):
            (Path(folder) / f"{n:05d}.jpg").write_bytes(data)

        def propagate(session, start=None):
            masks = {}
            # a session holding only instance clicks must name its start frame: without a text prompt the model finds "no prompts on any frame"
            for response in predictor.handle_stream_request({"type": "propagate_in_video", "session_id": session, "propagation_direction": "both", "start_frame_index": start}):
                out = response["outputs"]
                ids, found = np.asarray(out["out_obj_ids"]).astype(int), np.asarray(out["out_binary_masks"]).astype(bool)
                masks[int(response["frame_index"])] = {int(i): m for i, m in zip(ids, found)}
            return masks

        def session_of(name, work):
            stage, began = {}, time.perf_counter()
            try:
                session = sam3_video.start_session(predictor, folder)
                try:
                    masks = work(session, stage)
                finally:
                    predictor.handle_request({"type": "close_session", "session_id": session})
                for frame, objects in masks.items():
                    for ident, mask in objects.items():
                        arrays[f"{name}/{frame}/{ident}"] = np.packbits(mask)
                        stage["shape"] = list(mask.shape)
                stage["frames_with_masks"] = sum(bool(o) for o in masks.values())
            except Exception as error:  # the other stage's answer is still worth the call
                stage["error"] = f"{type(error).__name__}: {error}"[:600]
            stage["seconds"] = time.perf_counter() - began
            report["stages"][name] = stage

        def by_motion(session, stage):
            masks, stage["seeds_used"], stage["seeds_known"] = {}, [], 0
            for frame, points, packed in seeds:
                region = np.unpackbits(np.frombuffer(packed, np.uint8))
                known = np.zeros(0, bool)
                for mask in masks.get(frame, {}).values():
                    known = mask.ravel() if not known.size else known | mask.ravel()
                if known.size and (region[:known.size].astype(bool) & known).sum() >= COVERED * region.sum():
                    stage["seeds_known"] += 1
                    continue
                if len(stage["seeds_used"]) == MAX_OBJECTS:
                    break
                ident = len(stage["seeds_used"]) + 1
                predictor.handle_request({"type": "add_prompt", "session_id": session, "frame_index": frame, "points": points, "point_labels": [1] * len(points), "obj_id": ident})
                stage["seeds_used"].append({"object": ident, "frame": frame})
                masks = propagate(session, frame)  # again with the new object, so the next seed can be recognised as already tracked
            return masks

        def by_text(session, stage):
            predictor.handle_request({"type": "add_prompt", "session_id": session, "frame_index": 0, "text": text})
            return propagate(session)

        session_of("motion", by_motion)
        if text:
            session_of("text", by_text)
    report["remote_seconds"] = time.perf_counter() - started
    buffer = io.BytesIO()
    np.savez_compressed(buffer, report=json.dumps(report), **arrays)
    return buffer.getvalue()


def moving_share(masks, first, motion):
    """Median over the motion run's sampled frames of the share of the track's pixels above that frame's motion floor; None without samples."""
    shares = []
    for frame, mask in masks.items():
        path = motion / f"{first + frame:05d}-residual.npz"
        if not path.exists() or mask.sum() < 500:
            continue
        data = np.load(path)
        inside = data["residual"].astype(float)[mask]
        inside = inside[np.isfinite(inside)]
        if inside.size:
            shares.append(float((inside > float(data["floor"])).mean()))
    return float(np.median(shares)) if shares else None


def seeds_of(moving, first, last):
    """Seed regions of the motion run inside [first, last): connected parts of each frame's moving mask, largest first, with clicks well inside them."""
    import cv2
    seeds = []
    for frame, mask in moving.items():
        if not first <= frame < last:
            continue
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
        for n in range(1, count):
            if stats[n, cv2.CC_STAT_AREA] < MIN_SEED * mask.size:
                continue
            region, points = labels == n, []
            depth = cv2.distanceTransform(region.astype(np.uint8), cv2.DIST_L2, 5)
            for _ in range(POINTS):  # deepest point, then the deepest one away from it: clicks spread over the region, never on its rim
                y, x = np.unravel_index(np.argmax(depth), depth.shape)
                if depth[y, x] < 3:
                    break
                points.append([float(x) / mask.shape[1], float(y) / mask.shape[0]])
                cv2.circle(depth, (int(x), int(y)), int(max(depth[y, x] * 1.5, 12)), 0, -1)
            if points:
                seeds.append((int(stats[n, cv2.CC_STAT_AREA]), frame - first, points, region))
    return [(frame, points, region) for _, frame, points, region in sorted(seeds, key=lambda s: -s[0])]


def run(args):
    import cv2
    import mono_room as M
    M.use_clip(args.droid_run)
    manifest = json.loads((args.droid_run / "input-manifest.json").read_text())
    first, last = args.frames
    motion = json.loads((args.motion / "motion.json").read_text())
    moving = {r["frame"]: cv2.imread(str(args.motion / f"{r['frame']:05d}-moving.png"), 0) > 0 for r in motion["frames"]}
    seeds = seeds_of(moving, first, last)
    assert seeds, "the motion run found nothing large enough to seed a track in these frames"
    frames = [M.prepare_image(cv2.imread(str(M.DATASET / manifest["frames"][i]["relative_path"])), M.CALIBRATION, 2)[0] for i in range(first, last)]
    args.output.mkdir(parents=True, exist_ok=True)  # an output holding tracks.npz is read again, not paid for again
    state = {"status": "gpu_running", "model": sam3_video.MODEL_ID, "revision": sam3_video.MODEL_REVISION, "source_commit": sam3_video.SOURCE_COMMIT, "frames": [first, last],
             "seed_candidates": len(seeds), "text": args.text, "gpu": "A100-40GB", "timeout_s": 900, "retries": 0, "motion_run": str(args.motion), "cameras": str(args.droid_run)}
    (args.output / f"tracks-{int(time.time())}.json").write_text(json.dumps(state, indent=1))
    started = time.time()
    if (args.output / "tracks.npz").exists():
        answer = (args.output / "tracks.npz").read_bytes()
    else:
        jpegs = [cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes() for f in frames]
        with app.run():
            answer = track_remote.remote(jpegs, [(f, p, np.packbits(r).tobytes()) for f, p, r in seeds], args.text)
        (args.output / "tracks.npz").write_bytes(answer)  # the provider's answer, before any local reading
    found = np.load(io.BytesIO(answer))
    report = json.loads(str(found["report"]))
    state.update(status="complete", wall_seconds=time.time() - started, remote=report)
    summary, objects = {}, {}
    for name, stage in report["stages"].items():
        if "shape" not in stage:
            continue
        h, w = stage["shape"]
        tracks = {}
        for key in found.files:
            if key.startswith(name + "/"):
                _, frame, ident = key.split("/")
                tracks.setdefault(int(ident), {})[int(frame)] = np.unpackbits(found[key])[:h * w].reshape(h, w).astype(bool)
        objects[name] = [{"object": ident, "frames": len(masks), "moving_share_median": moving_share(masks, first, args.motion)} for ident, masks in sorted(tracks.items())]
        for o in objects[name]:  # a tracker follows whatever it is clicked on; only a track that keeps moving is dynamic
            o["kept"] = o["moving_share_median"] is None or o["moving_share_median"] >= MOVING_SHARE
        union = {}
        for o in objects[name]:
            for frame, mask in (tracks[o["object"]].items() if o["kept"] else ()):
                union[frame] = union.get(frame, np.zeros((h, w), bool)) | mask
        (args.output / name).mkdir(exist_ok=True)
        for old in (args.output / name).glob("*.png"):
            old.unlink()
        for frame, mask in union.items():
            cv2.imwrite(str(args.output / name / f"{first + frame:05d}.png"), mask.astype(np.uint8) * 255)
        if args.reference:
            rows = [(union.get(n, np.zeros((h, w), bool)), M.moving_mask(args.reference, first + n)) for n in range(0, last - first, 5)]
            rows = [(said, there) for said, there in rows if there.any()]
            summary[name] = {"frames_measured": len(rows), "median_iou": float(np.median([(s & t).sum() / max((s | t).sum(), 1) for s, t in rows])),
                             "pixel_recall": sum((s & t).sum() for s, t in rows) / max(sum(t.sum() for _, t in rows), 1),
                             "pixel_precision": sum((s & t).sum() for s, t in rows) / max(sum(s.sum() for s, _ in rows), 1)}
    state["against_reference"], state["objects"] = summary, objects
    (args.output / "tracks.json").write_text(json.dumps(state, indent=1))
    print(json.dumps({"wall_s": round(state["wall_seconds"]), "stages": {k: {x: v[x] for x in v if x not in ("shape", "seeds_used")} for k, v in report["stages"].items()},
                      "objects": objects, "against_reference": summary}, indent=1))


def self_check():
    mask = np.zeros((480, 640), bool)
    mask[100:400, 200:330] = True  # a person-sized region
    mask[10:20, 10:20] = True  # a speck: not a seed
    seeds = seeds_of({508: mask, 900: mask}, 500, 800)
    assert len(seeds) == 1 and seeds[0][0] == 8, "one seed, indexed inside the clip; the frame outside the clip and the speck are ignored"
    points = np.array(seeds[0][1]) * [640, 480]
    assert len(points) == POINTS and all(mask[int(y), int(x)] for x, y in points), "clicks lie inside the region"
    assert np.linalg.norm(points[0] - points[1]) > 12, "and apart from each other"
    import tempfile
    with tempfile.TemporaryDirectory() as folder:  # a track on the floor under a false seed carries no motion; a walking one does
        residual = np.zeros((480, 640), np.float16); residual[100:400, 200:330] = 9.
        np.savez_compressed(Path(folder) / "00510-residual.npz", residual=residual, floor=2.)
        walker, floor = np.zeros((480, 640), bool), np.zeros((480, 640), bool)
        walker[100:400, 200:330], floor[400:480, :] = True, True
        assert moving_share({10: walker}, 500, Path(folder)) >= MOVING_SHARE > moving_share({10: floor}, 500, Path(folder)) and moving_share({11: walker}, 500, Path(folder)) is None
    print("motion track check passed: seeds are large regions inside the clip, clicked well inside; a still track is not dynamic")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--droid-run", type=Path)
    parser.add_argument("--motion", type=Path, help="motion_masks.py run of the same clip")
    parser.add_argument("--frames", type=int, nargs=2, help="first and one-past-last source frame of the clip to track")
    parser.add_argument("--text", default="person", help="also track this SAM 3.1 text prompt in a second session, for comparison; empty to skip")
    parser.add_argument("--reference", type=Path, help="SOURCEINDEX-*.png masks to measure against (never an input)")
    parser.add_argument("--output", type=Path)
    a = parser.parse_args()
    self_check() if a.self_check else run(a)
