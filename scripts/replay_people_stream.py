"""Replay ME340 as a posed RGB-D stream through the live people loop and score it against the offline person layer.

Stand-ins, all stated: DROID poses (run 171) for phone VIO; DA3 posed depth (run 223, x its fused scale) for LiDAR;
cached SAM 3.1 person masks (run 187) for the live detector, so no GPU is spent. Frames 14-225 are a cut-away shot from
another camera: the stream carries them without a pose and the loop records a coverage gap. Metres are DROID units x the
stated 1.6 m carry height (model_estimated), so every distance and speed verdict is NEEDS_REVIEW by design; the
pre-gate verdict is kept to compare sample rates.

  python scripts/replay_people_stream.py --output NEW_DIR [--rate 1] [--zone-wkt WKT] [--detector sam3 --max-frames 8]
  python scripts/replay_people_stream.py --compare DIR_A DIR_B
  python scripts/replay_people_stream.py --self-check
"""
import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from shapely import wkt

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "modal_apps")]
from ehs_spatial.live_people import PeopleLoop, Sam3Detector, run  # noqa: E402
from ehs_spatial.video import R3_MAX_SPEED_MPS, SPEED_BAND_MPS, banded_verdict, contract_scale, track_speed  # noqa: E402

PHASE2 = Path("/Users/adam/Desktop/panoptes-public/research-notes/phase2")
CLIP, CAMERAS = PHASE2 / "data/clips/me340-165", PHASE2 / "runs/droid-me340-165-171"
DEPTH, MASKS = PHASE2 / "runs/da3-posed-me340-223-shotc", PHASE2 / "runs/me340-motion-analysis-187"
OFFLINE = PHASE2 / "runs/me340-cutaway-dynamic-305/scene/scene.json"
CUT_AWAY = range(14, 226)  # another camera; its DROID poses belong to no shot of this map


def raster(image, calibration):
    """Source 640x480 -> the depth model's raster (official TUM resize + crop), as mono_room does for every input."""
    from droid_room import prepare_image
    return prepare_image(image, calibration, 2)


def stream(rate):
    """Frames of the posed RGB-D contract at about `rate` Hz (0 = every depth frame, about 10 Hz)."""
    clip = json.loads((CLIP / "clip.json").read_text())
    calibration = {"source_K_fx_fy_cx_cy": clip["K"], "source_distortion": clip["D"]}
    rows = [line.split() for line in (CLIP / "rgb.txt").read_text().splitlines() if not line.startswith("#")]
    cameras = np.load(CAMERAS / "prediction.npz")
    poses, keys = cameras["poses_c2w"], {int(i): k for k, i in enumerate(cameras["keyframe_source_indices"])}
    keyframe_poses = cameras["keyframe_c2w"]
    scale = json.loads((DEPTH / "fuse-metrics.json").read_text())["scale_native_per_mono_metre_median"]
    with_depth = {int(p.stem) for p in (DEPTH / "mono").glob("*.npz")}
    candidates = sorted(with_depth | set(range(CUT_AWAY.start, CUT_AWAY.stop, 3)))
    start, tick, chosen = float(rows[0][0]), 0.0, []
    for index in candidates:
        t = float(rows[index][0]) - start
        if rate == 0 or t >= tick - 1e-6:
            chosen.append(index)
            tick = t + 1.0 / rate if rate else tick
    for index in chosen:
        rgb, k = raster(cv2.imread(str(CLIP / rows[index][1])), calibration)
        frame = {"t": float(rows[index][0]) - start, "frame": index, "rgb": cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB),
                 "K": np.array([[k[0], 0, k[2]], [0, k[1], k[3]], [0, 0, 1.0]]), "cameraToWorld": None, "depth": None,
                 "calibration": calibration}
        if index not in CUT_AWAY and index in with_depth:
            mono = np.load(DEPTH / "mono" / f"{index:05d}.npz")
            frame["depth"] = np.where(mono["mask"], mono["depth"], 0).astype(np.float32) * scale
            frame["cameraToWorld"] = (keyframe_poses[keys[index]] if index in keys else poses[index]).astype(np.float64)
        yield frame


def cached_detector():
    """Cached SAM 3.1 masks as a live "person" prompt: every entity that prompt ever named, in every frame it has a
    mask (the motion re-seed relabels the presenter "moving object" from frame 560). The one motion-only track,
    180-motion-4, is the burned-in subtitle, not a mover, so ME340 has no movers and R2 stays NO_DATA."""
    analysis = json.loads((MASKS / "analysis.json").read_text())
    frames = {f["sourceFrame"]: f["objects"] for f in analysis["frames"]}
    people = {o["entityId"] for objects in frames.values() for o in objects if o["label"] == "person"}

    def detect(frame):
        return [{"label": "person", "source": o["entityId"],
                 "mask": raster(cv2.imread(str(MASKS / o["maskUrl"]), cv2.IMREAD_COLOR), frame["calibration"])[0][..., 0] > 0}
                for o in frames.get(frame["frame"], []) if o["entityId"] in people]
    return detect


def compare_offline(tracks, up, metres):
    """Per (frame, entity): the loop's visible-surface centroid vs the offline one (3D), and its foot vs the offline
    centroid on the floor plane (the offline layer has no foot point, so this also holds half a body's depth)."""
    offline = {(f["sourceFrame"], o["entityId"]): np.array(o["centroid"]) for f in json.loads(OFFLINE.read_text())["frames"]
               for o in f["objects"] if o.get("centroid")}
    up = np.asarray(up) / np.linalg.norm(up)
    flat = lambda v: v - (v @ up) * up  # noqa: E731
    centroid, foot, foot_accepted = [], [], []
    for row in tracks:
        reference = offline.get((row["frame"], row["source"]))
        if reference is None:
            continue
        if row["centroidWorld"] is not None:
            centroid.append(np.linalg.norm(np.array(row["centroidWorld"]) - reference) * metres)
        error = np.linalg.norm(flat(np.array(row["footWorld"]) - reference)) * metres
        foot.append(error)
        if row["accepted"]:
            foot_accepted.append(error)

    def stats(values):
        return {"n": len(values), "median_m": round(float(np.median(values)), 3) if values else None,
                "p90_m": round(float(np.percentile(values, 90)), 3) if values else None}
    return {"matched_offline_rows": len(offline), "centroid_3d": stats(centroid), "foot_vs_centroid_on_floor": stats(foot),
            "foot_vs_centroid_on_floor_accepted_only": stats(foot_accepted)}


def replay(args):
    scale = json.loads((DEPTH / "metric-scale.json").read_text())
    record = contract_scale(scale)
    detector, cached, ious = cached_detector(), None, []
    if args.detector == "sam3":
        live, cached = Sam3Detector(), detector

        def detector(frame):  # live SAM 3, scored against the cached SAM 3.1 mask of the same frame
            found = live(frame)
            for reference in cached(frame) if found is not None else []:
                ious.append(max((iou(d["mask"], reference["mask"]) for d in found), default=0.0))
            return found
    zone = wkt.loads(args.zone_wkt) if args.zone_wkt else None
    loop = PeopleLoop(scale["plane_point_native"], scale["up_native"], record, detector, zone)
    frames = stream(args.rate)
    if args.max_frames:
        frames = (f for i, f in enumerate(f for f in frames if f["cameraToWorld"] is not None) if i < args.max_frames)
    started = time.perf_counter()
    counts = run(frames, loop, args.output)
    wall = time.perf_counter() - started
    tracks = [json.loads(line) for line in (args.output / "tracks.jsonl").read_text().splitlines()]
    reasons = {}
    for row in tracks:
        for reason in row["review"]:
            reasons[reason] = reasons.get(reason, 0) + 1
    summary = {"stream": {"clip": str(CLIP), "cameras": str(CAMERAS), "depth": str(DEPTH), "rate_hz": args.rate or "all",
                          "detector": args.detector if args.detector == "sam3" else f"cached SAM 3.1 masks {MASKS}",
                          "cut_away_frames_as_gap": [CUT_AWAY.start, CUT_AWAY.stop - 1]},
               "scale": record, **counts, "wall_s": round(wall, 2), "measured": "M: wall and loop latency on this Mac, CPU only",
               "person_rows": sum(r["label"] == "person" for r in tracks),
               "accepted_person_rows": sum(r["label"] == "person" and r["accepted"] for r in tracks),
               "review_reasons": reasons, "tracks": sorted({r["track"] for r in tracks}),
               "offline_comparison": compare_offline(tracks, scale["up_native"], record["nativeToMeters"])}
    if args.detector == "sam3":
        seconds = live.seconds or [0.0]
        summary["sam3"] = {**live.backend, "ephemeral_modal_app": args.ephemeral_modal, "calls": live.calls,
                           "call_s_p50": round(float(np.median(seconds)), 3), "call_s_max": round(float(max(seconds)), 3),
                           "call_s_all": [round(v, 3) for v in seconds],
                           "iou_vs_cached_sam31_person_mask": [round(float(v), 3) for v in ious]}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


def iou(a, b):
    union = (a | b).sum()
    return (a & b).sum() / union if union else 0.0


def on_ephemeral_modal(run_replay):
    """SAM 3 on our own L4 as an ephemeral app (modal_apps/sam3_app.py, timeout 600 s, retries 0), reached through
    sam_subscribe's modal branch: that branch looks the class up by deployed name, so the lookup is pointed at the
    ephemeral one. Nothing is deployed."""
    import os

    import modal
    import sam3_app
    os.environ["SAM3_BACKEND"] = "modal"
    deployed = modal.Cls.from_name
    modal.Cls.from_name = lambda app, name, **kw: sam3_app.Sam3 if (app, name) == ("sam3-inference", "Sam3") else deployed(app, name, **kw)
    started = time.perf_counter()
    with modal.enable_output(), sam3_app.app.run():
        run_replay()
    print(json.dumps({"ephemeral_app_wall_s": round(time.perf_counter() - started, 1)}))


def compare(a, b):
    """Two replays of one clip at different rates: the pre-gate speed verdict at every second both cover, and speeds
    of the same entity within 0.25 s."""
    def states(folder):
        rows = [json.loads(line) for line in (folder / "findings.jsonl").read_text().splitlines()]
        return [(r["t"], r["beforeScaleGate"] or r["verdict"]) for r in rows if r["rule"] == "R3_speed"]

    def at(timeline, t):
        before = [v for s, v in timeline if s <= t + 1e-6]
        return before[-1] if before else None
    first, second = states(a), states(b)
    last = lambda folder: json.loads((folder / "tracks.jsonl").read_text().splitlines()[-1])["t"]  # noqa: E731
    end = min(last(a), last(b))  # verdicts hold until the next change, so compare up to the stream's end
    seconds = [s for s in range(int(end) + 1) if at(first, s) and at(second, s)]
    agree = [s for s in seconds if at(first, s) == at(second, s)]
    speeds = lambda folder: [json.loads(line) for line in (folder / "tracks.jsonl").read_text().splitlines()  # noqa: E731
                             if json.loads(line)["speedMps"] is not None]
    other = speeds(b)
    differences = []
    for row in speeds(a):
        near = [o for o in other if o["source"] == row["source"] and abs(o["t"] - row["t"]) <= .25]
        if near:
            differences.append(abs(min(near, key=lambda o: abs(o["t"] - row["t"]))["speedMps"] - row["speedMps"]))
    body = {}  # diagnostic, not a rule: the same regression on the visible-body centroid, which ME340 keeps in view
    for name, folder in (("a", a), ("b", b)):
        rows = [json.loads(line) for line in (folder / "tracks.jsonl").read_text().splitlines()]
        for source in {r["source"] for r in rows}:
            samples = [(r["t"], tuple(r["centroidXyM"])) for r in rows if r["source"] == source and r["centroidXyM"]]
            for r in rows:
                if r["source"] == source and r["centroidXyM"]:
                    body.setdefault(name, {})[(source, round(r["t"] * 5) / 5)] = track_speed(samples, r["t"])[0]
    shared = [k for k in body["a"] if k in body["b"] and body["a"][k] is not None and body["b"][k] is not None]
    gaps = [abs(body["a"][k] - body["b"][k]) for k in shared]
    verdict = lambda v: banded_verdict(v, R3_MAX_SPEED_MPS, SPEED_BAND_MPS, fail_low=False)  # noqa: E731
    result = {"body_centroid_speed_pairs": len(shared),
              "body_speed_abs_difference_mps_median": round(float(np.median(gaps)), 3) if gaps else None,
              "body_speed_abs_difference_mps_max": round(float(max(gaps)), 3) if gaps else None,
              "body_speed_same_verdict": sum(verdict(body["a"][k]) == verdict(body["b"][k]) for k in shared),
              "body_speed_mps_median": round(float(np.median([body["a"][k] for k in shared])), 3) if shared else None,
              "seconds_compared": len(seconds), "same_speed_verdict": len(agree),
              "disagreeing_seconds": [(s, at(first, s), at(second, s)) for s in seconds if s not in agree],
              "speed_pairs": len(differences),
              "speed_abs_difference_mps_median": round(float(np.median(differences)), 3) if differences else None,
              "speed_abs_difference_mps_p90": round(float(np.percentile(differences, 90)), 3) if differences else None}
    print(json.dumps(result, indent=1))
    return result


def self_check():
    """The stream picks one frame per tick, carries the cut-away as pose-less frames, and never invents depth."""
    frames = list(stream(1))
    assert all(b["t"] - a["t"] >= 1.0 - 1e-6 for a, b in zip(frames, frames[1:]))
    gap = [f for f in frames if f["frame"] in CUT_AWAY]
    assert gap and all(f["cameraToWorld"] is None and f["depth"] is None for f in gap)
    posed = [f for f in frames if f["cameraToWorld"] is not None]
    assert posed and all(f["depth"].shape == f["rgb"].shape[:2] == (480, 640) for f in posed)
    person = cached_detector()(posed[0])
    assert [d["label"] for d in person] == ["person"] and person[0]["mask"].any()
    late = next(f for f in frames if f["frame"] >= 600 and f["cameraToWorld"] is not None)
    assert [d["source"][-6:] for d in cached_detector()(late)] == ["text-0"], "the subtitle track is not a person"
    print(f"replay stream check passed: {len(frames)} frames at 1 Hz, {len(gap)} cut-away gap frames, masks on the depth raster")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--rate", type=float, default=0, help="Hz; 0 streams every frame that has depth (about 10 Hz)")
    parser.add_argument("--zone-wkt", help="keep-clear polygon in floor metres of the loop's floor frame")
    parser.add_argument("--detector", choices=["cached", "sam3"], default="cached")
    parser.add_argument("--max-frames", type=int, default=0, help="stop after this many posed frames (live detector budget)")
    parser.add_argument("--ephemeral-modal", action="store_true", help="with --detector sam3: run our L4 SAM 3 as an ephemeral app")
    parser.add_argument("--compare", type=Path, nargs=2)
    parser.add_argument("--self-check", action="store_true")
    a = parser.parse_args()
    if a.self_check:
        self_check()
    elif a.compare:
        compare(*a.compare)
    else:
        a.output.mkdir(parents=True, exist_ok=False)
        on_ephemeral_modal(lambda: replay(a)) if a.ephemeral_modal else replay(a)
