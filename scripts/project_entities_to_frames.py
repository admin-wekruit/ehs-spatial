"""Make every video frame selectable by projecting multi-view 3D entities into it. No model call.

Each confirmed object-map entity is a set of world points (its observed surfaces). For every source frame the official
filler camera projects them into source pixels, lens distortion included; points hidden behind the fused room mesh are
dropped. What is left becomes a clickable outline carrying the entity's identity. This is geometry re-projection, not a
per-frame segmentation: where cameras or depth are off the outline is off by the same amount, and it says so.

  python scripts/project_entities_to_frames.py --droid-run RUN --scene REPLAY/scene.json --object-map DIR --mesh ROOM.glb \
      --output NEW_DIR [--merge-analysis EXISTING/analysis.json]
  python scripts/project_entities_to_frames.py --self-check
"""
import argparse
import json
import os
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "modal_apps"), str(ROOT / "scripts")]
from build_video_object_map import STUFF  # noqa: E402
CONFIRMED, SAMPLES, MIN_VISIBLE, HIDDEN = 3, 800, 40, .05  # HIDDEN: a hit more than 5% nearer than the point occludes it


def visible_outline(points, c2w, k, distortion, scene, size=(640, 480)):  # size: the source frame's
    """Source-pixel convex outline of the points seen from this camera, or None."""
    import open3d as o3d
    local = (points - c2w[:3, 3]) @ c2w[:3, :3]
    front = local[:, 2] > 1e-6
    if front.sum() < MIN_VISIBLE:
        return None
    local, world = local[front], points[front]
    pixels = cv2.projectPoints(local.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), k, distortion)[0].reshape(-1, 2)
    inside = (pixels[:, 0] >= 0) & (pixels[:, 0] < size[0]) & (pixels[:, 1] >= 0) & (pixels[:, 1] < size[1])
    if inside.sum() < MIN_VISIBLE:
        return None
    pixels, world = pixels[inside], world[inside]
    direction = world - c2w[:3, 3]
    distance = np.linalg.norm(direction, axis=1)
    rays = np.hstack([np.repeat(c2w[:3, 3][None], len(world), 0), direction / distance[:, None]]).astype(np.float32)
    hit = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
    seen = hit >= distance * (1 - HIDDEN)
    if seen.sum() < MIN_VISIBLE:
        return None
    hull = cv2.convexHull(pixels[seen].astype(np.float32)).reshape(-1, 2)
    if len(hull) < 3 or cv2.contourArea(hull) < 60:
        return None
    return hull, float(seen.sum() / len(points))


def raycaster(vertices, faces):
    import open3d as o3d
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(np.asarray(vertices, np.float32)), o3d.core.Tensor(np.asarray(faces, np.uint32)))
    return scene


def build(args):
    import trimesh
    import mono_room
    mono_room.use_clip(args.droid_run)
    fx, fy, cx, cy = mono_room.SOURCE_K
    k, distortion, size = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]]), np.asarray(mono_room.SOURCE_D, float), tuple(mono_room.SOURCE_WH)
    prediction = np.load(args.droid_run / "prediction.npz")
    cameras = prediction["poses_c2w"].astype(np.float64)

    def source_k(index):  # a device refocuses per frame: its own K, from the 320-wide model raster up to source pixels
        if not mono_room.METRIC_CAMERAS:
            return k
        f = prediction["keyframe_final_fullres_intrinsics"][index].astype(float) * 2 * (size[0] / 640)
        return np.array([[f[0], 0, f[2]], [0, f[1], f[3]], [0, 0, 1.]])

    replay = json.loads(args.scene.read_text())
    room = trimesh.load(args.mesh, force="mesh", process=False)
    scene = raycaster(room.vertices, room.faces)
    rng, entities = np.random.default_rng(0), []
    for entity in json.loads((args.object_map / "object-map.json").read_text())["entities"]:
        # a wall's or floor's convex outline would cover the frame and swallow every click meant for an object in front of it
        if len(entity["observations"]) >= CONFIRMED and entity.get("surfaces") and entity["label"] not in STUFF:
            points = np.concatenate([np.load(args.object_map / s["file"])["vertices"] for s in entity["surfaces"]])
            entities.append({"entityId": f"obs-{entity['entityId']}", "label": entity["label"], "views": len(entity["observations"]),
                             "points": points[rng.choice(len(points), min(len(points), SAMPLES), replace=False)]})
    merged = {}
    if args.merge_analysis:  # keep the existing per-frame people; their mask files stay where they are
        for frame in json.loads(args.merge_analysis.read_text())["frames"]:
            for item in frame["objects"]:
                if item.get("maskUrl"):
                    item["maskUrl"] = os.path.relpath((args.merge_analysis.parent / item["maskUrl"]).resolve(), args.output.resolve())
            merged[frame["sourceFrame"]] = frame["objects"]
    args.output.mkdir(parents=True, exist_ok=False)
    frames, outlines = [], 0
    for frame in replay["frames"]:
        index, objects = frame["sourceFrame"], []
        for entity in entities:
            found = visible_outline(entity["points"], cameras[index], source_k(index), distortion, scene, size)
            if found:
                hull, share = found
                objects.append({"entityId": entity["entityId"], "label": entity["label"], "displayName": f"{entity['label']} · {entity['views']} 个视图确认",
                                "polygons": [np.round(hull, 1).tolist()], "bbox": [float(hull[:, 0].min()), float(hull[:, 1].min()), float(hull[:, 0].max()), float(hull[:, 1].max())],
                                "visibleShare": round(share, 3), "evidence": "projected_3d_entity"})
        outlines += len(objects)
        frames.append({"timeSec": frame["timeSec"], "endTimeSec": frame["endTimeSec"], "sourceFrame": index, "objects": objects + merged.get(index, []), "absentEntityIds": []})
    analysis = {"version": 1, "coordinateSpace": "source_pixels", "width": size[0], "height": size[1], "identityScope": "video_session_only",
                "method": "Multi-view 3D object-map entities re-projected into every frame with the DROID cameras and lens distortion; occlusion against the fused room mesh. Not a per-frame segmentation.",
                "frames": frames,
                "limitations": ["对象轮廓是三维实体投影到该帧的凸包，不是逐帧分割；相机或深度有误差时轮廓同样偏移。", "被融合网格挡住超过5%距离的点视为不可见；网格有孔洞处可能多显示。",
                                "只包含被≥3个视图确认的实体；人物标注沿用已有的逐帧分割。"],
                "provenance": {"objectMap": str(args.object_map), "roomMesh": str(args.mesh), "cameras": str(args.droid_run), "mergedAnalysis": str(args.merge_analysis) if args.merge_analysis else None,
                               "samplesPerEntity": SAMPLES, "minVisiblePoints": MIN_VISIBLE, "hiddenTolerance": HIDDEN, "newModelCalls": 0}}
    (args.output / "analysis.json").write_text(json.dumps(analysis, allow_nan=False, separators=(",", ":")))
    per_frame = np.array([sum(o.get("evidence") == "projected_3d_entity" for o in f["objects"]) for f in frames])
    print(json.dumps({"frames": len(frames), "entities": len(entities), "outlines": outlines, "frames_with_outline": int((per_frame > 0).sum()),
                      "median_outlines_per_frame": float(np.median(per_frame))}))


def self_check():
    k, none = np.array([[500., 0, 320], [0, 500., 240], [0, 0, 1]]), np.zeros(5)
    grid = np.stack(np.meshgrid(np.linspace(-.2, .2, 12), np.linspace(-.2, .2, 12)), -1).reshape(-1, 2)
    target = np.column_stack([grid, np.full(len(grid), 3.)])                       # a patch 3 units in front of the camera
    wall = lambda z: (np.array([[-5, -5, z], [5, -5, z], [5, 5, z], [-5, 5, z]], float), np.array([[0, 1, 2], [0, 2, 3]]))
    behind, front = raycaster(*wall(4.)), raycaster(*wall(2.))
    hull, share = visible_outline(target, np.eye(4), k, none, behind)
    assert share == 1 and abs(hull[:, 0].min() - (320 - 500 * .2 / 3)) < .5, "unoccluded patch must project as a pinhole would"
    assert visible_outline(target, np.eye(4), k, none, front) is None, "a wall in front must hide the patch"
    away = np.eye(4); away[:3, :3] = np.diag([-1., 1, -1])
    assert visible_outline(target, away, k, none, behind) is None, "a camera facing away sees nothing"
    print("projection check passed: pinhole agreement, occlusion by a nearer surface, nothing behind the camera")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    for name in ("droid-run", "scene", "object-map", "mesh", "output", "merge-analysis"):
        parser.add_argument("--" + name, type=Path)
    a = parser.parse_args()
    self_check() if a.self_check else build(a)
