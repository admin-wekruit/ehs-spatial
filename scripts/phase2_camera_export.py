"""Export one final ORB map and only valid final camera poses for a viewer.

Never fuse independent maps or apply evaluation alignment to scene geometry.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from reconstruct_tum_room import associate, digest, read_rows
from phase2_camera_run import read_euroc


def to_atlas(rows, atlas_from_euroc):
    rows = np.asarray(rows, dtype=float)
    origin = np.asarray(atlas_from_euroc, dtype=float)
    if rows.ndim != 2 or rows.shape[1] != 8 or origin.shape != (4, 4):
        raise ValueError("Expected EuRoC timestamp XYZ quaternion rows and a 4x4 origin")
    if not np.isfinite(rows).all() or not np.isfinite(origin).all():
        raise ValueError("Nonfinite camera pose")
    if not np.allclose(np.linalg.norm(rows[:, 4:8], axis=1), 1, atol=1e-5):
        raise ValueError("Expected normalized xyzw quaternions")
    poses = np.tile(np.eye(4), (len(rows), 1, 1))
    poses[:, :3, :3] = Rotation.from_quat(rows[:, 4:8]).as_matrix()
    poses[:, :3, 3] = rows[:, 1:4]
    return origin @ poses


def export(run, output):
    run = run.resolve()
    metadata = json.loads((run / "run.json").read_text())
    if not metadata.get("execution_complete"):
        raise ValueError("Cannot export an incomplete execution")
    names = ["map.sparse.json", "trajectory.final.euroc.txt", "trajectory.export.json",
             "atlas.final.json", "frames.online.jsonl", "input.manifest.json"]
    for name in names:
        if digest(run / name) != metadata["artifacts_sha256"][name]:
            raise ValueError(f"Source artifact changed: {name}")
    cloud = json.loads((run / "map.sparse.json").read_text())
    origin = json.loads((run / "trajectory.export.json").read_text())
    inventory = json.loads((run / "atlas.final.json").read_text())
    if cloud["map_id"] != origin["map_id"]:
        raise ValueError("Cloud and trajectory do not share a final map")
    points = np.asarray(cloud["points"], float)
    selected_map = next(m for m in inventory["maps"] if m["map_id"] == cloud["map_id"])
    if (points.ndim != 2 or points.shape[1] != 4 or not np.isfinite(points).all()
            or len(set(points[:, 0])) != len(points)
            or set(points[:, 0]) != set(selected_map["map_point_ids"])):
        raise ValueError("Cloud does not match the final map inventory")
    ledger = [json.loads(line) for line in (run / "frames.online.jsonl").read_text().splitlines()]
    valid = [f for f in ledger if f["state"] == 2 and f["valid"]]
    raw = np.asarray(read_euroc(run / "trajectory.final.euroc.txt"), float)
    times = raw[:, 0] / 1e9
    pairs = associate([f["timestamp"] for f in valid], times, 0.000002)
    if not pairs:
        raise ValueError("No valid final poses")
    chosen = [j for _, j in pairs]
    poses = to_atlas(raw[chosen], origin["atlas_from_euroc"])
    inputs = {f["source_index"]: f for f in json.loads((run / "input.manifest.json").read_text())}
    for i, _ in pairs:
        frame = inputs[valid[i]["source_index"]]
        if digest(Path(frame["source_path"])) != frame["sha256"]:
            raise ValueError(f"Source image changed: {frame['source_path']}")
    first_stamp = float(read_rows(Path(metadata["dataset"]) / "rgb.txt", 2)[0][0])
    frames = [{"source_index": valid[i]["source_index"], "timestamp": valid[i]["timestamp"],
               "source_time_s": valid[i]["timestamp"] - first_stamp,
               "source_image": inputs[valid[i]["source_index"]]["source_path"],
               "source_image_sha256": inputs[valid[i]["source_index"]]["sha256"],
               "c2w": pose.tolist(), "state": 2}
              for (i, _), pose in zip(pairs, poses)]
    included = {f["source_index"] for f in frames}
    start, stop = metadata["selected_range_start_stop"]
    value = {"schema": "phase2-orb-native-scene-v1", "source_run": str(run),
             "source_run_sha256": digest(run / "run.json"),
             "coordinate_frame": "final_native_atlas", "map_id": cloud["map_id"],
             "units": "meters" if metadata["native_scale"] == "sensor_depth_meters" else metadata["native_scale"],
             "scale_provenance": metadata["native_scale"], "evaluation_alignment_applied": False,
             "complete_room_accepted": False, "camera_settings": str(run / "camera.yaml"),
             "point_columns": ["id", "x", "y", "z"], "points": cloud["points"], "frames": frames,
             "coverage": {"input_frames": len(ledger), "requested_rgb_frames": stop - start,
                          "source_rgb_frames": metadata["source_frame_count"], "all_maps_online_ok": len(valid),
                          "selected_map_final_ok": len(frames), "final_maps": len(inventory["maps"]),
                          "omitted_source_indices": [index for index in range(start, stop) if index not in included]},
             "limitations": ["One final map only; other maps have independent coordinates.",
                             "Gaps remain gaps. The viewer must not interpolate lost poses.",
                             "Final bundle-adjusted coordinates differ from online poses."]}
    output.mkdir(parents=True, exist_ok=False)
    path = output / "native-scene.json"
    path.write_text(json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n")
    print(json.dumps({"path": str(path), "points": len(points), "valid_final_poses": len(frames),
                      "map_id": cloud["map_id"], "sha256": digest(path)}))


def self_check():
    origin = np.eye(4)
    origin[:3, :3] = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    origin[:3, 3] = [2, 3, 4]
    result = to_atlas([[1e9, 1, 0, 0, 0, 0, 0, 1]], origin)[0]
    assert np.allclose(result[:3, 3], [2, 4, 4])
    assert np.allclose(np.linalg.inv(result) @ origin @ [1, 0, 2, 1], [0, 0, 2, 1])
    assert not np.allclose(result, np.linalg.inv(origin))
    print("camera export self-check passed: final c2w composition and camera/world reprojection")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
    elif args.run and args.output:
        export(args.run, args.output)
    else:
        parser.error("Pass --run and --output, or --self-check")
