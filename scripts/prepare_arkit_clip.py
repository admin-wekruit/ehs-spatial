"""Wrap one ARKitScenes capture as a posed multi-view clip for the posed-depth pipeline. No SLAM, no model call.

The 1920x1440 frames stay on disk untouched; the pipeline works on their 640x480 raster. Cameras are the device's
metric ARKit poses (convention detected as arkitscenes_eval does), so no scale is assumed or fitted. LiDAR depth and
the 3D box annotations are copied for evaluation only and are never read by infer/fuse.

  python scripts/prepare_arkit_clip.py --scenes-root .../arkitscenes/scenes --scene 42445448 --data NEW_DATA_DIR --run NEW_RUN_DIR
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arkitscenes_eval as arkit  # noqa: E402


def write_preview(data, fps=1):
    """The stills as one constant-rate H.264 file, a frame per still, so the replay page can show and scrub them like a video."""
    import cv2
    names = [line.split()[1] for line in (data / "rgb.txt").read_text().splitlines() if not line.startswith("#")]
    first = cv2.imread(str(data / names[0]))
    writer = cv2.VideoWriter(str(data / "preview.mp4"), cv2.VideoWriter_fourcc(*"avc1"), fps, first.shape[1::-1])
    assert writer.isOpened(), "no H.264 encoder available to OpenCV"
    for name in names:
        writer.write(cv2.imread(str(data / name)))
    writer.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("scenes-root", "data", "run"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--scene", required=True)
    args = parser.parse_args()
    arkit.DATA = args.scenes_root.parent  # the reused loader resolves DATA / "scenes" / id
    scene = arkit._load_scene(args.scene)
    assert scene, "scene has fewer than two annotated objects or is missing"
    world_to_camera = arkit._detect_convention(scene)
    (args.data / "rgb").mkdir(parents=True, exist_ok=False)
    (args.data / "evaluation_only" / "highres_depth").mkdir(parents=True)
    args.run.mkdir(parents=True, exist_ok=False)
    frames, cameras, intrinsics = [], [], []
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
            data = archive.read(name)
            relative = f"rgb/{stamp:.3f}.png"
            (args.data / relative).write_bytes(data)
            (args.data / "evaluation_only" / "highres_depth" / f"{stamp:.3f}.png").write_bytes(archive.read(name.replace("/wide/", "/highres_depth/")))
            frames.append({"source_index": len(frames), "timestamp_text": f"{stamp:.3f}", "relative_path": relative, "sha256": hashlib.sha256(data).hexdigest()})
            cameras.append(c2w); intrinsics.append(pin)
    intrinsics = np.array(intrinsics)
    k = np.median(intrinsics, 0)  # autofocus moves the focal length slightly; the spread is recorded, one K is used
    spread = float(np.abs(intrinsics[:, 0] / k[0] - 1).max())
    (args.data / "rgb.txt").write_text("# timestamp filename\n" + "".join(f"{f['timestamp_text']} {f['relative_path']}\n" for f in frames))
    (args.data / "evaluation_only" / "annotation.json").write_bytes((args.scenes_root / args.scene / "annotation.json").read_bytes())
    definition = {"dataset": str(args.data.resolve()), "K": k.tolist(), "D": [0., 0., 0., 0., 0.], "raster": "resize", "metric_cameras": True, "source_wh": [1920, 1440]}
    manifest = {"frame_count": len(frames), "rgb_index_sha256": hashlib.sha256((args.data / "rgb.txt").read_bytes()).hexdigest(),
                "source_K_fx_fy_cx_cy": k.tolist(), "source_distortion": definition["D"], "frames": frames, "groundtruth_included": False, "depth_included": False}
    (args.run / "input-manifest.json").write_text(json.dumps(manifest, indent=1))
    cameras = np.array(cameras, np.float32)
    model_k = (intrinsics * (640 / 1920) / 2).astype(np.float32)  # per frame; the pipeline stores K of the 320x240 model raster and doubles it for 640x480
    np.savez(args.run / "prediction.npz", poses_c2w=cameras, keyframe_source_indices=np.arange(len(frames)), keyframe_c2w=cameras,
             keyframe_final_fullres_intrinsics=model_k)
    (args.run / "run.json").write_text(json.dumps({"status": "posed_multiview_prepared", "clip": f"arkit-{args.scene}", "clip_definition": definition,
        "cameras": "ARKit device poses, metric; convention " + ("world-to-camera inverted" if world_to_camera else "camera-to-world"),
        "focal_length_relative_spread": spread, "frames": len(frames), "skipped_without_pose": len(scene["frames"]) - len(frames),
        "evaluation_only": str((args.data / "evaluation_only").resolve()), "dataset_licence": "ARKitScenes: Apple, non-commercial research"}, indent=1))
    write_preview(args.data)
    steps = np.linalg.norm(np.diff(cameras[:, :3, 3], axis=0), axis=1)
    print(json.dumps({"frames": len(frames), "skipped_without_pose": len(scene["frames"]) - len(frames), "focal_spread": round(spread, 4),
                      "camera_path_m": round(float(steps.sum()), 2), "median_step_m": round(float(np.median(steps)), 3), "annotated_objects": sorted({o["label"] for o in scene["objects"]})}))


if __name__ == "__main__":
    main()
