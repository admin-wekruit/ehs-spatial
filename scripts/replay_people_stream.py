"""Replay a capture as the live-core stream through the live people loop: ME340 (scored against the offline person
layer) or the ARKitScenes 47333932 room scan (device-metric poses and LiDAR, no people).

Every frame goes through the stream contract (ehs_spatial.live_people: header JSON, rgb JPEG, depth PNG16 mm at the
LiDAR raster 256x192, confidence PNG8) and back through decode_frame, so this loop reads what the phone sends.

ME340 stand-ins, all stated: DROID poses (run 171) for phone VIO; DA3 posed depth (run 223, x its fused scale) for
LiDAR; cached SAM 3.1 person masks (run 187) for the live detector, so no GPU is spent. Frames 14-225 are a cut-away
shot from another camera: they go out with tracking "notAvailable" and no pose, and the loop records a coverage gap.
Metres are DROID units x the stated 1.6 m carry height (model_estimated), so every verdict is NEEDS_REVIEW by design;
the pre-gate verdict is kept to compare sample rates.

ARKit: poses, K and bytes come from live-core's own producer code (scripts/live_map.py of m0/live-core, imported);
the floor is a plane fitted to confident LiDAR points, standing in for ARKit's plane detection. The capture records no
tracking state or world epoch and has no people, so those are stated ("normal", 0) and the detector is a stand-in
that reports nobody: a PASS there only means the zone floor was observed.

  python scripts/replay_people_stream.py --output NEW_DIR [--rate 1] [--zone-wkt WKT] [--detector sam3 --max-frames 8]
  python scripts/replay_people_stream.py --output NEW_DIR --arkit RAW_DIR [--zone-wkt WKT]   (needs scripts/live_map.py)
  python scripts/replay_people_stream.py --compare DIR_A DIR_B
  python scripts/replay_people_stream.py --self-check
"""
import argparse
from itertools import islice
import json
from pathlib import Path
import sys
import time
import zipfile

import cv2
import numpy as np
from shapely import wkt

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "modal_apps")]
from ehs_spatial.live_people import MIN_CONFIDENCE, PeopleLoop, Sam3Detector, decode_frame, pack, run  # noqa: E402
from ehs_spatial.video import R3_MAX_SPEED_MPS, SPEED_BAND_MPS, banded_verdict, contract_scale, track_speed  # noqa: E402

PHASE2 = Path("/Users/adam/Desktop/panoptes-public/research-notes/phase2")
CLIP, CAMERAS = PHASE2 / "data/clips/me340-165", PHASE2 / "runs/droid-me340-165-171"
DEPTH, MASKS = PHASE2 / "runs/da3-posed-me340-223-shotc", PHASE2 / "runs/me340-motion-analysis-187"
OFFLINE = PHASE2 / "runs/me340-cutaway-dynamic-305/scene/scene.json"
CUT_AWAY = range(14, 226)  # another camera; its DROID poses belong to no shot of this map
DEPTH_WH = (256, 192)  # ARKit's LiDAR raster: depth goes out lower than rgb, as the phone sends it


def raster(image, calibration):
    """Source 640x480 -> the depth model's raster (official TUM resize + crop), as mono_room does for every input."""
    from droid_room import prepare_image
    return prepare_image(image, calibration, 2)


def calibration():
    clip = json.loads((CLIP / "clip.json").read_text())
    return {"source_K_fx_fy_cx_cy": clip["K"], "source_distortion": clip["D"]}


def encode_depth(metres):
    """Depth in metres on the rgb raster -> (PNG16 mm, PNG8 confidence) at DEPTH_WH, sampled at pixel centres."""
    rows = ((np.arange(DEPTH_WH[1]) + 0.5) * metres.shape[0] / DEPTH_WH[1]).astype(int)
    cols = ((np.arange(DEPTH_WH[0]) + 0.5) * metres.shape[1] / DEPTH_WH[0]).astype(int)
    millimetres = np.rint(metres[np.ix_(rows, cols)] * 1000)
    millimetres[(millimetres > 65535) | ~np.isfinite(millimetres)] = 0
    confident = np.where(millimetres > 0, MIN_CONFIDENCE, 0).astype(np.uint8)
    return cv2.imencode(".png", millimetres.astype(np.uint16))[1].tobytes(), cv2.imencode(".png", confident)[1].tobytes()


def stream(rate):
    """ME340 as stream messages at about `rate` Hz (0 = every depth frame, about 10 Hz); t_capture counts from 0."""
    rows = [line.split() for line in (CLIP / "rgb.txt").read_text().splitlines() if not line.startswith("#")]
    cameras = np.load(CAMERAS / "prediction.npz")
    poses, keys = cameras["poses_c2w"], {int(i): k for k, i in enumerate(cameras["keyframe_source_indices"])}
    keyframe_poses = cameras["keyframe_c2w"]
    scale = json.loads((DEPTH / "fuse-metrics.json").read_text())["scale_native_per_mono_metre_median"]
    metres = json.loads((DEPTH / "metric-scale.json").read_text())["metres_per_native_unit"]
    with_depth = {int(p.stem) for p in (DEPTH / "mono").glob("*.npz")}
    candidates = sorted(with_depth | set(range(CUT_AWAY.start, CUT_AWAY.stop, 3)))
    start, tick, chosen = float(rows[0][0]), 0.0, []
    for index in candidates:
        t = float(rows[index][0]) - start
        if rate == 0 or t >= tick - 1e-6:
            chosen.append(index)
            tick = t + 1.0 / rate if rate else tick
    for index in chosen:
        rgb, k = raster(cv2.imread(str(CLIP / rows[index][1])), calibration())
        t = float(rows[index][0]) - start
        header = {"seq": index, "t_capture": t, "t_device": t, "K": [float(v) for v in k], "cameraToWorld": None,
                  "tracking": "notAvailable", "world_epoch": 0}
        depth = confidence = b""
        if index not in CUT_AWAY and index in with_depth:
            mono = np.load(DEPTH / "mono" / f"{index:05d}.npz")
            depth, confidence = encode_depth(np.where(mono["mask"], mono["depth"], 0).astype(np.float64) * scale * metres)
            c2w = (keyframe_poses[keys[index]] if index in keys else poses[index]).astype(np.float64)
            c2w[:3, 3] *= metres
            header.update(cameraToWorld=c2w.ravel().tolist(), tracking="normal")
        yield pack(header, cv2.imencode(".jpg", rgb, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes(), depth, confidence)


def cached_detector():
    """Cached SAM 3.1 masks as a live "person" prompt: every entity that prompt ever named, in every frame it has a
    mask (the motion re-seed relabels the presenter "moving object" from frame 560). The one motion-only track,
    180-motion-4, is the burned-in subtitle, not a mover, so ME340 has no movers and R2 stays NO_DATA."""
    analysis = json.loads((MASKS / "analysis.json").read_text())
    frames = {f["sourceFrame"]: f["objects"] for f in analysis["frames"]}
    people = {o["entityId"] for objects in frames.values() for o in objects if o["label"] == "person"}
    source = calibration()

    def detect(frame):
        return [{"label": "person", "source": o["entityId"],
                 "mask": raster(cv2.imread(str(MASKS / o["maskUrl"]), cv2.IMREAD_COLOR), source)[0][..., 0] > 0}
                for o in frames.get(frame["frame"], []) if o["entityId"] in people]
    return detect


def compare_offline(tracks, up, metres):
    """Per (frame, entity): the loop's visible-surface centroid vs the offline one (3D), and its foot vs the offline
    centroid on the floor plane (the offline layer has no foot point, so this also holds half a body's depth)."""
    offline = {(f["sourceFrame"], o["entityId"]): np.array(o["centroid"]) * metres
               for f in json.loads(OFFLINE.read_text())["frames"] for o in f["objects"] if o.get("centroid")}
    up = np.asarray(up) / np.linalg.norm(up)
    flat = lambda v: v - (v @ up) * up  # noqa: E731
    centroid, foot, foot_accepted = [], [], []
    for row in tracks:
        reference = offline.get((row["frame"], row["source"]))
        if reference is None:
            continue
        if row["centroidWorld"] is not None:
            centroid.append(np.linalg.norm(np.array(row["centroidWorld"]) - reference))
        if row["footWorld"] is None:
            continue
        error = np.linalg.norm(flat(np.array(row["footWorld"]) - reference))
        foot.append(error)
        if row["accepted"]:
            foot_accepted.append(error)

    def stats(values):
        return {"n": len(values), "median_m": round(float(np.median(values)), 3) if values else None,
                "p90_m": round(float(np.percentile(values, 90)), 3) if values else None}
    return {"matched_offline_rows": len(offline), "centroid_3d": stats(centroid), "foot_vs_centroid_on_floor": stats(foot),
            "foot_vs_centroid_on_floor_accepted_only": stats(foot_accepted)}


def lidar_floor(raw, frames, live_map):
    """ARKit's floor for the replay: over confident LiDAR points of every 100th frame, the densest 1 cm height band,
    refit to its points as the band narrows from 10 cm to 2 cm (heights re-measured along each new normal, starting
    from the cameras' mean image-up). Returns (point, up) in world metres."""
    points = []
    with zipfile.ZipFile(raw / "lowres_depth.zip") as depth_zip, zipfile.ZipFile(raw / "confidence.zip") as confidence_zip:
        depths = {live_map.stamp_of(n): n for n in depth_zip.namelist() if n.endswith(".png")}
        levels = {live_map.stamp_of(n): n for n in confidence_zip.namelist() if n.endswith(".png")}
        for text, _, c2w, k in frames[::100]:
            z = decode_frame(pack({"t_capture": 0.0, "K": k}, b"", depth_zip.read(depths[text]),
                                  confidence_zip.read(levels[text])))["depth"]
            fx, fy, cx, cy = np.array(k) * z.shape[1] / 640
            vs, us = np.nonzero(z > 0)
            camera = np.c_[(us + 0.5 - cx) / fx, (vs + 0.5 - cy) / fy, np.ones(len(us))] * z[vs, us][:, None]
            points.append(camera @ c2w[:3, :3].T + c2w[:3, 3])
    points = np.vstack(points)
    normal = -np.mean([c2w[:3, 1] for _, _, c2w, _ in frames], axis=0)  # the camera's image-up, on average, is up
    for band in (0.10, 0.05, 0.02, 0.02):
        normal /= np.linalg.norm(normal)
        heights = points @ normal
        counts, edges = np.histogram(heights, bins=np.arange(heights.min(), heights.min() + 1.5, 0.01))
        floor = points[np.abs(heights - edges[counts.argmax()] - 0.005) < band]
        centre = floor.mean(axis=0)
        fitted = np.linalg.svd(floor - centre, full_matrices=False)[2][-1]
        normal = fitted if fitted @ normal > 0 else -fitted
    return centre, normal


def arkit_stream(raw, frames, live_map):
    """ARKitScenes 47333932 as the phone sends it, through live-core's own code (arkit_frames, pack), every LiDAR frame
    in order; rgb only where the capture kept a wide image (about 1 frame in 29)."""
    with zipfile.ZipFile(raw / "wide.zip") as wide, zipfile.ZipFile(raw / "lowres_depth.zip") as depth, \
            zipfile.ZipFile(raw / "confidence.zip") as confidence:
        images = {live_map.stamp_of(n): n for n in wide.namelist() if n.endswith(".png")}
        prefix = next(iter(images.values())).split("/")[1].split("_")[0]
        for seq, (text, t_device, c2w, k) in enumerate(frames):
            rgb = b""
            if text in images:
                image = cv2.imdecode(np.frombuffer(wide.read(images[text]), np.uint8), cv2.IMREAD_COLOR)
                rgb = cv2.imencode(".jpg", cv2.resize(image, (640, 480), interpolation=cv2.INTER_AREA),
                                   [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()
            yield live_map.pack({"seq": seq, "t_capture": t_device - frames[0][1], "t_device": t_device, "K": k,
                                 "cameraToWorld": c2w.ravel().tolist(), "tracking": "normal", "world_epoch": 0},
                                rgb, depth.read(f"lowres_depth/{prefix}_{text}.png"),
                                confidence.read(f"confidence/{prefix}_{text}.png"))


def replay_arkit(args):
    import live_map  # scripts/live_map.py, from m0/live-core

    frames = live_map.arkit_frames(args.arkit)
    point, up = lidar_floor(args.arkit, frames, live_map)
    record = contract_scale({"scale_status": "device_metric", "metres_per_native_unit": 1.0})
    zone = wkt.loads(args.zone_wkt) if args.zone_wkt else None
    loop = PeopleLoop(point, up, record, lambda frame: [], zone)
    started = time.perf_counter()
    counts = run((decode_frame(m) for m in arkit_stream(args.arkit, frames, live_map)), loop, args.output)
    r1 = [json.loads(line) for line in (args.output / "findings.jsonl").read_text().splitlines()]
    summary = {"stream": {"capture": str(args.arkit), "producer": "scripts/live_map.py arkit_frames + pack (m0/live-core)",
                          "detector": "stand-in reporting nobody (the room scan has no people): R1 PASS = zone floor observed",
                          "tracking_and_world_epoch": "stated normal / 0: ARKitScenes keeps neither"},
               "floor": {"point": point.round(4).tolist(), "up": up.round(5).tolist(), "camera_height_m_median":
                         round(float(np.median([(c2w[:3, 3] - point) @ up for _, _, c2w, _ in frames])), 3)},
               "scale": record, "zone_wkt": args.zone_wkt, **counts, "wall_s": round(time.perf_counter() - started, 2),
               "measured": "M: wall and loop latency on this Mac, CPU only",
               "r1_changes": [(f["t"], f["verdict"], f["reason"]) for f in r1 if f["rule"] == "R1_zone"][:60]}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


def replay(args):
    scale = json.loads((DEPTH / "metric-scale.json").read_text())
    record, metres = contract_scale(scale), scale["metres_per_native_unit"]
    detector, cached, ious, live_masks = cached_detector(), None, [], []
    if args.detector == "sam3":
        live, cached = Sam3Detector(), detector

        def detector(frame):  # live SAM 3, scored against the cached SAM 3.1 masks of the same frame both ways
            found = live(frame)
            references = cached(frame) if found is not None else []
            for reference in references:  # recall: each cached person's best live mask
                ious.append(max((iou(d["mask"], reference["mask"]) for d in found), default=0.0))
            for d in found or []:  # precision: does a cached person explain each live mask?
                best = max((iou(d["mask"], r["mask"]) for r in references), default=0.0)
                live_masks.append({"frame": frame["frame"], "score": round(d["score"], 3), "best_iou": round(float(best), 3)})
            return found
    zone = wkt.loads(args.zone_wkt) if args.zone_wkt else None
    loop = PeopleLoop(np.array(scale["plane_point_native"]) * metres, scale["up_native"], record, detector, zone)
    frames = (decode_frame(message) for message in stream(args.rate))
    if args.max_frames:
        frames = islice((f for f in frames if f["cameraToWorld"] is not None), args.max_frames)
    started = time.perf_counter()
    counts = run(frames, loop, args.output)
    wall = time.perf_counter() - started
    tracks = [json.loads(line) for line in (args.output / "tracks.jsonl").read_text().splitlines()]
    reasons = {}
    for row in tracks:
        for reason in row["review"]:
            reasons[reason] = reasons.get(reason, 0) + 1
    summary = {"stream": {"clip": str(CLIP), "cameras": str(CAMERAS), "depth": str(DEPTH), "rate_hz": args.rate or "all",
                          "depth_raster": list(DEPTH_WH),
                          "detector": args.detector if args.detector == "sam3" else f"cached SAM 3.1 masks {MASKS}",
                          "cut_away_frames_as_gap": [CUT_AWAY.start, CUT_AWAY.stop - 1]},
               "scale": record, "zone_wkt": args.zone_wkt, **counts, "wall_s": round(wall, 2),
               "measured": "M: wall and loop latency on this Mac, CPU only",
               "person_rows": sum(r["label"] == "person" for r in tracks),
               "accepted_person_rows": sum(r["label"] == "person" and r["accepted"] for r in tracks),
               "unplaced_person_rows": sum(r["label"] == "person" and r["footWorld"] is None for r in tracks),
               "review_reasons": reasons, "tracks": sorted({r["track"] for r in tracks if r["track"]}),
               "offline_comparison": compare_offline(tracks, scale["up_native"], metres)}
    if args.detector == "sam3":
        seconds = list(live.seconds) or [0.0]
        warm = seconds[1:] or [0.0]
        summary["sam3"] = {**live.backend, "ephemeral_modal_app": args.ephemeral_modal, "calls": live.calls,
                           "call_s_first": round(seconds[0], 3), "call_s_warm_p50": round(float(np.median(warm)), 3),
                           "call_s_warm_max": round(float(max(warm)), 3), "call_s_all": [round(v, 3) for v in seconds],
                           "iou_vs_cached_sam31_person_mask": [round(float(v), 3) for v in ious],
                           "live_masks": len(live_masks),
                           "unmatched_live_masks": [m for m in live_masks if m["best_iou"] < 0.5]}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


def iou(a, b):
    union = (a | b).sum()
    return (a & b).sum() / union if union else 0.0


def on_ephemeral_modal(run_replay, output):
    """SAM 3 on our own L4 as an ephemeral app (modal_apps/sam3_app.py, timeout 600 s, retries 0), reached through
    sam_subscribe's modal branch: that branch looks the class up by deployed name, so the lookup is pointed at the
    ephemeral one. Nothing is deployed. The app's wall time goes into summary.json (it bounds the GPU seconds)."""
    import os

    import modal
    import sam3_app
    os.environ["SAM3_BACKEND"] = "modal"
    deployed = modal.Cls.from_name
    modal.Cls.from_name = lambda app, name, **kw: sam3_app.Sam3 if (app, name) == ("sam3-inference", "Sam3") else deployed(app, name, **kw)
    started = time.perf_counter()
    with modal.enable_output(), sam3_app.app.run():
        run_replay()
    summary = json.loads((output / "summary.json").read_text())
    summary["sam3"]["ephemeral_app_wall_s"] = round(time.perf_counter() - started, 1)
    (output / "summary.json").write_text(json.dumps(summary, indent=1))


def old_rule(folder, step):
    """The replaced speed rule on the visible-body centroid: consecutive samples, band 2 x 0.35 m per step."""
    by_source, verdicts = {}, {}
    for row in (json.loads(line) for line in (folder / "tracks.jsonl").read_text().splitlines()):
        if row["centroidXyM"]:
            by_source.setdefault(row["source"], []).append((row["t"], row["centroidXyM"]))
    for source, samples in by_source.items():
        for (t0, a), (t1, b) in zip(samples, samples[1:]):
            if t1 - t0 <= 1.5 * step:
                speed = np.hypot(b[0] - a[0], b[1] - a[1]) / (t1 - t0)
                verdicts[(source, round(t1 * 5) / 5)] = banded_verdict(speed, R3_MAX_SPEED_MPS, 0.7 / step, fail_low=False)
    return verdicts


def compare(a, b):
    """Two replays of one clip at different rates: the pre-gate speed verdict at every second both cover, speeds of the
    same entity within 0.25 s, the body-centroid regression both ways, and the replaced consecutive-sample rule."""
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
    step = lambda folder: 1.0 / json.loads((folder / "summary.json").read_text())["stream"]["rate_hz"]  # noqa: E731
    old_a, old_b = old_rule(a, step(a)), old_rule(b, step(b))
    old_shared = [k for k in old_a if k in old_b]
    result = {"body_centroid_speed_pairs": len(shared),
              "body_speed_abs_difference_mps_median": round(float(np.median(gaps)), 3) if gaps else None,
              "body_speed_abs_difference_mps_max": round(float(max(gaps)), 3) if gaps else None,
              "body_speed_same_verdict": sum(verdict(body["a"][k]) == verdict(body["b"][k]) for k in shared),
              "body_speed_mps_median": round(float(np.median([body["a"][k] for k in shared])), 3) if shared else None,
              "old_consecutive_rule_pairs": len(old_shared),
              "old_consecutive_rule_same_verdict": sum(old_a[k] == old_b[k] for k in old_shared),
              "old_consecutive_rule_verdicts": [sorted({old_a[k] for k in old_shared}), sorted({old_b[k] for k in old_shared})],
              "seconds_compared": len(seconds), "same_speed_verdict": len(agree),
              "disagreeing_seconds": [(s, at(first, s), at(second, s)) for s in seconds if s not in agree],
              "speed_pairs": len(differences),
              "speed_abs_difference_mps_median": round(float(np.median(differences)), 3) if differences else None,
              "speed_abs_difference_mps_p90": round(float(np.percentile(differences, 90)), 3) if differences else None}
    print(json.dumps(result, indent=1))
    return result


def self_check():
    """The stream picks one frame per tick, carries the cut-away as pose-less notAvailable frames, sends depth at the
    LiDAR raster in millimetres, and decode_frame gets back the metres it was given."""
    frames = [decode_frame(m) for m in stream(1)]
    assert all(b["t"] - a["t"] >= 1.0 - 1e-6 for a, b in zip(frames, frames[1:]))
    gap = [f for f in frames if f["frame"] in CUT_AWAY]
    assert gap and all(f["cameraToWorld"] is None and f["depth"] is None and f["tracking"] == "notAvailable" for f in gap)
    posed = [f for f in frames if f["cameraToWorld"] is not None]
    assert posed and all(f["depth"].shape == (192, 256) and f["rgb"].shape == (480, 640, 3) for f in posed)
    fused = json.loads((DEPTH / "fuse-metrics.json").read_text())["scale_native_per_mono_metre_median"]
    mono = np.load(DEPTH / "mono" / f"{posed[0]['frame']:05d}.npz")
    sent = np.where(mono["mask"], mono["depth"], 0) * fused * json.loads((DEPTH / "metric-scale.json").read_text())[
        "metres_per_native_unit"]
    assert abs(posed[0]["depth"][96, 128] - sent[int(96.5 * 2.5), int(128.5 * 2.5)]) <= 0.0005 + 1e-6  # mm rounding
    person = cached_detector()(posed[0])
    assert [d["label"] for d in person] == ["person"] and person[0]["mask"].shape == (480, 640) and person[0]["mask"].any()
    late = next(f for f in frames if f["frame"] >= 600 and f["cameraToWorld"] is not None)
    assert [d["source"][-6:] for d in cached_detector()(late)] == ["text-0"], "the subtitle track is not a person"
    print(f"replay stream check passed: {len(frames)} frames at 1 Hz, {len(gap)} cut-away gap frames, depth PNG16 mm at "
          "256x192 decodes to the metres sent, masks on the rgb raster")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--rate", type=float, default=0, help="Hz; 0 streams every frame that has depth (about 10 Hz)")
    parser.add_argument("--zone-wkt", help="keep-clear polygon in floor metres of the loop's floor frame")
    parser.add_argument("--detector", choices=["cached", "sam3"], default="cached")
    parser.add_argument("--max-frames", type=int, default=0, help="stop after this many posed frames (live detector budget)")
    parser.add_argument("--ephemeral-modal", action="store_true", help="with --detector sam3: run our L4 SAM 3 as an ephemeral app")
    parser.add_argument("--arkit", type=Path, help="ARKitScenes raw capture directory (lowres_wide.traj, *.zip)")
    parser.add_argument("--compare", type=Path, nargs=2)
    parser.add_argument("--self-check", action="store_true")
    a = parser.parse_args()
    if a.self_check:
        self_check()
    elif a.compare:
        compare(*a.compare)
    else:
        a.output.mkdir(parents=True, exist_ok=False)
        if a.arkit:
            replay_arkit(a)
        elif a.ephemeral_modal:
            on_ephemeral_modal(lambda: replay(a), a.output)
        else:
            replay(a)
