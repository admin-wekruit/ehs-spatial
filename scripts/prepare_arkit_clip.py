"""Wrap one ARKitScenes capture as a posed clip for the posed-depth pipeline. No SLAM, no model call.

The 1920x1440 frames stay untouched on disk; the pipeline works on their 640x480 raster. Cameras are the device's metric
ARKit poses, so no scale is assumed or fitted. Reference depth (and annotations, where they exist) are copied for
evaluation only and are never read by infer/fuse.

Two layouts of the public dataset:
  --scenes-root .../scenes --scene ID   the "upsampling" subset: sparse stills (about 2 s apart) with laser-scan depth
  --raw DIR --video-id ID               the raw capture: continuous video with LiDAR depth; DIR holds wide.zip,
                                        wide_intrinsics.zip, lowres_wide.traj, lowres_depth.zip, confidence.zip

  python scripts/prepare_arkit_clip.py --raw DIR --video-id 47333932 --data NEW_DATA_DIR --run NEW_RUN_DIR
  python scripts/prepare_arkit_clip.py --self-check
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def write_preview(data, fps=1):
    """The frames as one constant-rate H.264 file, a frame per still, so the replay page can show and scrub them like a video."""
    names = [line.split()[1] for line in (data / "rgb.txt").read_text().splitlines() if not line.startswith("#")]
    first = cv2.imread(str(data / names[0]))
    writer = cv2.VideoWriter(str(data / "preview.mp4"), cv2.VideoWriter_fourcc(*"avc1"), fps, first.shape[1::-1])
    assert writer.isOpened(), "no H.264 encoder available to OpenCV"
    for name in names:
        writer.write(cv2.imread(str(data / name)))
    writer.release()


def upsampling(args):
    """Yields (stamp, png bytes, c2w, fx fy cx cy, reference depth png bytes); laser-scan depth rendered into each still."""
    import arkitscenes_eval as arkit
    arkit.DATA = args.scenes_root.parent  # the reused loader resolves DATA / "scenes" / id
    scene = arkit._load_scene(args.scene)
    assert scene, "scene has fewer than two annotated objects or is missing"
    world_to_camera = arkit._detect_convention(scene)
    (args.data / "evaluation_only").mkdir(parents=True)
    (args.data / "evaluation_only" / "annotation.json").write_bytes((args.scenes_root / args.scene / "annotation.json").read_bytes())
    with zipfile.ZipFile(args.scenes_root / args.scene / "upsampling.zip") as archive:
        for stamp, name in scene["frames"]:
            pose, pin = arkit._nearest(scene["poses"], stamp), arkit._nearest(scene["intrinsics"], stamp)
            if pose is None or pin is None:
                continue  # no device pose within 0.1 s: the frame is not used rather than interpolated
            rotation, translation = pose
            c2w = np.eye(4)
            if world_to_camera:
                c2w[:3, :3], c2w[:3, 3] = rotation.T, -rotation.T @ translation
            else:
                c2w[:3, :3], c2w[:3, 3] = rotation, translation
            yield stamp, archive.read(name), c2w, pin, archive.read(name.replace("/wide/", "/highres_depth/"))


def agreement(depths, cameras, ks, world_to_camera):
    """Median relative depth disagreement when each depth map is carried into the next camera: small only for the right pose convention."""
    errors = []
    for (da, ca, ka), (db, cb, kb) in zip(zip(depths, cameras, ks), zip(depths[1:], cameras[1:], ks[1:])):
        a2w, b2w = (np.linalg.inv(c) if world_to_camera else c for c in (ca, cb))
        v, u = np.indices(da.shape)
        ok = da > 0
        local = np.stack([(u[ok] - ka[2]) / ka[0] * da[ok], (v[ok] - ka[3]) / ka[1] * da[ok], da[ok]], 1)
        there = (local @ a2w[:3, :3].T + a2w[:3, 3] - b2w[:3, 3]) @ b2w[:3, :3]
        front = there[:, 2] > 1e-6
        there = there[front]
        x, y = np.round(there[:, 0] / there[:, 2] * kb[0] + kb[2]).astype(int), np.round(there[:, 1] / there[:, 2] * kb[1] + kb[3]).astype(int)
        inside = (x >= 0) & (x < db.shape[1]) & (y >= 0) & (y < db.shape[0])
        seen = db[y[inside], x[inside]]
        errors += list(np.abs(there[inside, 2][seen > 0] / seen[seen > 0] - 1))
    return float(np.median(errors)) if errors else np.inf


def raw(args):
    """Yields the same tuples for a raw capture; LiDAR depth of the highest confidence only, as reference."""
    lines = [list(map(float, line.split())) for line in (args.raw / "lowres_wide.traj").read_text().splitlines() if line.strip()]
    poses = {}
    for stamp, rx, ry, rz, tx, ty, tz in lines:
        pose = np.eye(4)
        pose[:3, :3], pose[:3, 3] = cv2.Rodrigues(np.array([rx, ry, rz]))[0], [tx, ty, tz]
        poses[stamp] = pose
    times = np.array(sorted(poses))

    def pose_at(stamp, limit=.15):
        """The device pose at a frame's own time, between the two trajectory samples around it (the trajectory has its own clock ticks)."""
        from scipy.spatial.transform import Rotation, Slerp
        after = int(np.searchsorted(times, stamp))
        if after == 0 or after == len(times) or times[after] - times[after - 1] > limit:
            return None, None
        t0, t1 = times[after - 1], times[after]
        a, b, w = poses[t0], poses[t1], (stamp - t0) / (t1 - t0)
        pose = np.eye(4)
        pose[:3, :3] = Slerp([0, 1], Rotation.from_matrix([a[:3, :3], b[:3, :3]]))(w).as_matrix()
        pose[:3, 3] = (1 - w) * a[:3, 3] + w * b[:3, 3]
        return pose, float(min(stamp - t0, t1 - stamp))

    def nearest(stamp, keys, limit):
        key = keys[int(np.argmin(np.abs(keys - stamp)))]
        return float(key) if abs(key - stamp) <= limit else None

    with zipfile.ZipFile(args.raw / "wide.zip") as wide, zipfile.ZipFile(args.raw / "wide_intrinsics.zip") as pins, \
            zipfile.ZipFile(args.raw / "lowres_depth.zip") as depth, zipfile.ZipFile(args.raw / "confidence.zip") as confidence:
        stamp_of = lambda name: float(Path(name).stem.split("_")[1])
        images = sorted((n for n in wide.namelist() if n.endswith(".png")), key=stamp_of)
        pin_of = {stamp_of(n): n for n in pins.namelist() if n.endswith(".pincam")}
        depth_of = {stamp_of(n): n for n in depth.namelist() if n.endswith(".png")}
        confidence_of = {stamp_of(n): n for n in confidence.namelist() if n.endswith(".png")}
        pin_times, depth_times = np.array(sorted(pin_of)), np.array(sorted(depth_of))
        rows = []
        for name in images:
            stamp = stamp_of(name)
            (pose, offset), pin, lidar = pose_at(stamp), nearest(stamp, pin_times, .06), nearest(stamp, depth_times, .02)
            if pose is None or pin is None:
                continue  # no trajectory samples on both sides within 150 ms, or no intrinsics: the frame is not used
            rows.append((stamp, name, pose, list(map(float, pins.read(pin_of[pin]).split()))[2:], lidar, offset))
        # the trajectory file does not say which way its transform runs: carry LiDAR depth between views and see
        probe = [r for r in rows[:: max(1, len(rows) // 8)] if r[4] is not None][:8]
        maps = [cv2.imdecode(np.frombuffer(depth.read(depth_of[r[4]]), np.uint8), cv2.IMREAD_UNCHANGED).astype(np.float64) / 1000 for r in probe]
        small = [np.array(r[3]) * (maps[0].shape[1] / 1920) for r in probe]
        scores = {way: agreement(maps, [r[2] for r in probe], small, way) for way in (True, False)}
        world_to_camera = scores[True] < scores[False]
        args.convention = {"world_to_camera": bool(world_to_camera), "depth_carry_median_relative_error": {"world_to_camera": scores[True], "camera_to_world": scores[False]},
                           "pose": "interpolated between the two trajectory samples around each frame", "nearest_sample_s_max": float(max(r[5] for r in rows))}
        for stamp, name, pose, pin, lidar, _ in rows:
            reference = None
            if lidar is not None and lidar in confidence_of:
                metres = cv2.imdecode(np.frombuffer(depth.read(depth_of[lidar]), np.uint8), cv2.IMREAD_UNCHANGED)
                sure = cv2.imdecode(np.frombuffer(confidence.read(confidence_of[lidar]), np.uint8), cv2.IMREAD_UNCHANGED) == 2
                reference = cv2.imencode(".png", np.where(sure, metres, 0).astype(np.uint16))[1].tobytes()
            yield stamp, wide.read(name), np.linalg.inv(pose) if world_to_camera else pose, pin, reference


def main(args):
    (args.data / "rgb").mkdir(parents=True, exist_ok=False)
    args.run.mkdir(parents=True, exist_ok=False)
    depth_dir = args.data / "evaluation_only" / ("lowres_depth_confident" if args.raw else "highres_depth")
    frames, cameras, intrinsics, stamps, without_reference = [], [], [], [], 0
    for stamp, png, c2w, pin, reference in (raw(args) if args.raw else upsampling(args)):
        relative = f"rgb/{stamp:.3f}.png"
        (args.data / relative).write_bytes(png)
        depth_dir.mkdir(parents=True, exist_ok=True)
        if reference is None:
            without_reference += 1
        else:
            (depth_dir / f"{stamp:.3f}.png").write_bytes(reference)
        frames.append({"source_index": len(frames), "timestamp_text": f"{stamp:.3f}", "relative_path": relative, "sha256": hashlib.sha256(png).hexdigest()})
        cameras.append(c2w); intrinsics.append(pin); stamps.append(stamp)
    intrinsics = np.array(intrinsics)
    k = np.median(intrinsics, 0)  # autofocus moves the focal length slightly; per-frame values are kept, the median only describes the clip
    spread = float(np.abs(intrinsics[:, 0] / k[0] - 1).max())
    rate = 1. if not args.raw else float(round(1 / np.median(np.diff(stamps)), 2))
    (args.data / "rgb.txt").write_text("# timestamp filename\n" + "".join(f"{f['timestamp_text']} {f['relative_path']}\n" for f in frames))
    definition = {"dataset": str(args.data.resolve()), "K": k.tolist(), "D": [0., 0., 0., 0., 0.], "raster": "resize", "metric_cameras": True, "source_wh": [1920, 1440],
                  "evaluation_depth": str(depth_dir.relative_to(args.data))}
    manifest = {"frame_count": len(frames), "rgb_index_sha256": hashlib.sha256((args.data / "rgb.txt").read_bytes()).hexdigest(),
                "source_K_fx_fy_cx_cy": k.tolist(), "source_distortion": definition["D"], "frames": frames, "groundtruth_included": False, "depth_included": False}
    (args.run / "input-manifest.json").write_text(json.dumps(manifest, indent=1))
    cameras = np.array(cameras, np.float32)
    model_k = (intrinsics * (640 / 1920) / 2).astype(np.float32)  # per frame; the pipeline stores K of the 320x240 model raster and doubles it for 640x480
    np.savez(args.run / "prediction.npz", poses_c2w=cameras, keyframe_source_indices=np.arange(len(frames)), keyframe_c2w=cameras,
             keyframe_final_fullres_intrinsics=model_k)
    name = args.video_id if args.raw else args.scene
    (args.run / "run.json").write_text(json.dumps({"status": "posed_multiview_prepared", "clip": f"arkit-{name}", "clip_definition": definition,
        "cameras": "ARKit device poses, metric", "pose_convention": getattr(args, "convention", "detected from annotated objects (arkitscenes_eval)"),
        "focal_length_relative_spread": spread, "frames": len(frames), "frames_per_second": rate, "frames_without_reference_depth": without_reference,
        "evaluation_only": str((args.data / "evaluation_only").resolve()), "dataset_licence": "ARKitScenes: Apple, non-commercial research"}, indent=1))
    write_preview(args.data, rate)
    steps = np.linalg.norm(np.diff(cameras[:, :3, 3], axis=0), axis=1)
    print(json.dumps({"frames": len(frames), "frames_per_second": rate, "seconds": round(float(stamps[-1] - stamps[0]), 1), "focal_spread": round(spread, 4),
                      "camera_path_m": round(float(steps.sum()), 2), "median_step_m": round(float(np.median(steps)), 3),
                      "pose_convention": getattr(args, "convention", None), "frames_without_reference_depth": without_reference}))


def self_check():
    k, depth = np.array([100., 100., 32., 24.]), np.full((48, 64), 2.)  # a wall 2 m ahead
    turn = np.eye(4)
    turn[:3, :3], turn[0, 3] = cv2.Rodrigues(np.array([0., .3, 0.]))[0], .5  # a camera that stepped aside and turned, as camera-to-world
    v, u = np.indices(depth.shape)
    rays = np.stack([(u - k[2]) / k[0], (v - k[3]) / k[1], np.ones(depth.shape)], -1) @ turn[:3, :3].T
    seen = ((2. - turn[2, 3]) / rays[..., 2]) * 1.  # depth along the turned camera's own axis of the same wall z = 2
    local_z = seen * 1.
    right = agreement([depth, local_z], [np.eye(4), turn], [k, k], False)
    wrong = agreement([depth, local_z], [np.eye(4), turn], [k, k], True)
    assert right < .01 < wrong, (right, wrong)
    print("arkit clip check passed: depth carried between views agrees only under the right pose convention")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    for name in ("scenes-root", "raw", "data", "run"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--scene")
    parser.add_argument("--video-id")
    a = parser.parse_args()
    self_check() if a.self_check else main(a)
