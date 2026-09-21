"""Put object-map entities (and their checked generated models) into a replay scene as selectable static objects.

The replay viewer already links video time, point cloud / mesh and selectable static objects with a model toggle.
Each confirmed entity contributes one source-bound surface: the view its model was generated from, else its largest
view. Mask and frame are the original source-pixel files, so the overlay matches the video exactly.

  python scripts/attach_entities_to_replay.py --scene BASE/scene.json --object-map DIR --masks ROOT --models DIR --output NEW_DIR
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np

CONFIRMED, LIGHT = 3, 8000  # views for a confirmed entity; triangles above which a replay surface is thinned


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def light_model(path, triangles=40000):
    """A generated model as a page can carry it: the generator returns ~750k triangles (15 MB) per object, a viewer needs a few percent.

    Quadric decimation with vertex colours; returns (glb bytes, record) with the largest distance of the light surface from the
    validated one, so the display copy is known to be the checked shape. The validated file itself is not touched.
    """
    import open3d as o3d
    import trimesh
    full = trimesh.load(path, force="mesh", process=False)
    if len(full.faces) <= triangles:
        return Path(path).read_bytes(), {"triangles": len(full.faces), "decimated": False}
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(full.vertices), o3d.utility.Vector3iVector(full.faces))
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.asarray(full.visual.vertex_colors)[:, :3] / 255.)
    light = mesh.simplify_quadric_decimation(triangles)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    deviation = float(scene.compute_distance(o3d.core.Tensor(np.asarray(light.vertices, np.float32))).numpy().max())
    data = trimesh.Trimesh(np.asarray(light.vertices), np.asarray(light.triangles), vertex_colors=(np.asarray(light.vertex_colors) * 255).astype(np.uint8), process=False).export(file_type="glb")
    return data, {"triangles": len(light.triangles), "decimated": True, "from_triangles": len(full.faces), "max_deviation_native": deviation, "validated_mesh_sha256": sha(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("scene", "object-map", "masks", "models", "output"):
        parser.add_argument("--" + name, type=Path, required=name != "models")
    parser.add_argument("--inferred-floor", type=Path, help="infer_room_floor.py output: becomes the floor entity's model")
    args = parser.parse_args()
    import trimesh
    scene = json.loads(args.scene.read_text())
    frames = {f["sourceFrame"]: f for f in scene["frames"]}
    args.output.mkdir(parents=True, exist_ok=False)
    relative = lambda path: os.path.relpath(Path(path).resolve(), args.output.resolve())
    for key in ("meshUrl", "pointCloudUrl"):  # the geometry stays where it was built
        scene[key] = relative(args.scene.parent / scene[key])
    for frame in scene["frames"]:
        for entity in frame["objects"]:
            if entity.get("surface"):
                entity["surface"]["meshUrl"] = relative(args.scene.parent / entity["surface"]["meshUrl"])
    models = {json.loads(v.read_text())["observation"]: v.parent for v in (args.models.glob("*/validation.json") if args.models else [])
              if json.loads(v.read_text())["accepted_source_consistency"]}
    # a model generated from a name-prompted whole-object mask is bound by entity: its anchor frame and mask are its own record
    prompted = {json.loads(v.read_text())["entity"]: v.parent for v in (args.models.glob("*/validation.json") if args.models else [])
                if json.loads(v.read_text())["accepted_source_consistency"] and (v.parent / "anchor.json").exists()}
    static = []
    floors = [e for e in json.loads((args.object_map / "object-map.json").read_text())["entities"] if e["label"] == "floor" and len(e["observations"]) >= CONFIRMED and e.get("surfaces")]
    floor_owner = max(floors, key=lambda e: len(e["observations"]))["entityId"] if floors else None  # several floor pieces, one floor model
    for entity in json.loads((args.object_map / "object-map.json").read_text())["entities"]:
        if len(entity["observations"]) < CONFIRMED or not entity.get("surfaces"):
            continue
        anchored = [s for s in entity["surfaces"] if s["observation"] in models]
        chosen = anchored[0] if anchored else max(entity["surfaces"], key=lambda s: s["triangles"])
        label, index, instance = chosen["observation"].split(":")
        folder = next(args.masks.glob(f"{label}-*/frame-{int(index):05d}"))
        mask_path, image_path = folder / f"instance-{instance}-mask.png", folder / f"frame-{int(index)}.png"
        if entity["entityId"] in prompted:  # shown with the frame and whole-object mask its model came from
            anchor = json.loads((prompted[entity["entityId"]] / "anchor.json").read_text())
            index, mask_path, image_path = anchor["sourceFrame"], Path(anchor["mask"]), Path(anchor["image"])
        pixels = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)  # RGBA overlay or single-band mask, both in source pixels
        ys, xs = np.nonzero(pixels.reshape(*pixels.shape[:2], -1).max(-1) > 0)
        frame = frames[int(index)]
        identity = f"obs-{entity['entityId']}"
        data = np.load(args.object_map / chosen["file"])
        surface = trimesh.Trimesh(data["vertices"], data["faces"], vertex_colors=data["colors"], process=False)
        if len(surface.faces) > LIGHT:  # hundreds of objects load with the page: a pixel-dense patch is thinned to a grid of its own size / 80
            import open3d as o3d
            dense = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(surface.vertices), o3d.utility.Vector3iVector(surface.faces))
            dense.vertex_colors = o3d.utility.Vector3dVector(np.asarray(data["colors"])[:, :3] / 255.)
            light = dense.simplify_vertex_clustering(float(surface.extents.max()) / 80)
            surface = trimesh.Trimesh(np.asarray(light.vertices), np.asarray(light.triangles), vertex_colors=(np.asarray(light.vertex_colors) * 255).astype(np.uint8), process=False)
        surface.export(args.output / f"{identity}.glb")
        (args.output / f"{identity}.json").write_text(json.dumps({"objectMapEntity": entity["entityId"], "observations": entity["observations"],
            "sourceFrames": entity["sourceFrames"], "shownObservation": chosen["observation"], "source_mask_sha256": sha(mask_path),
            "source_image_sha256": sha(image_path), "method": "visible surface of one view of a multi-view object-map entity; unseen sides not made up"}, indent=1))
        name = entity["label"]  # the prompt, or the VLM's name for a class-agnostic segment; the mask folder keeps the segmentation label
        record = {"entityId": identity, "label": name, "displayName": f"{name} · {len(entity['observations'])} 个视图确认" + ("（VLM命名）" if entity.get("labelSource") else ""), "meshUrl": f"{identity}.glb",
                  "representation": "single_frame_observed_surface", "identityScope": "independent_observation", "provenanceUrl": f"{identity}.json",
                  "objectMapEntity": {"entityId": entity["entityId"], "observations": len(entity["observations"]), "sourceFrames": entity["sourceFrames"]},
                  "source": {"sourceFrame": int(index), "timeSec": frame["timeSec"], "endTimeSec": frame["endTimeSec"], "width": int(pixels.shape[1]), "height": int(pixels.shape[0]),
                             "maskUrl": relative(mask_path), "imageUrl": relative(image_path), "bbox": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]}}
        if args.inferred_floor and entity["entityId"] == floor_owner:  # the model view shows the whole floor: the verified plane over the room's footprint, stated as inferred
            basis = json.loads((args.inferred_floor / "inferred-floor.json").read_text())
            record["generatedModel"] = {"meshUrl": relative(args.inferred_floor / "inferred-floor.glb"), "sha256": basis["mesh_sha256"], "provenanceUrl": relative(args.inferred_floor / "inferred-floor.json"),
                                        "sourceFrame": int(index), "status": "source_consistent_model_estimate",
                                        "note": f"推断模型，不是观测：已验证的地面平面铺满房间范围（{basis['area_m2']} m²），相机只看到其中 {basis['observed_share_of_this_floor']:.0%}；未观测部分上面有什么不知道。"}
            record["displayName"] += f"；模型＝推断地面 {basis['area_m2']} m²（其中观测到 {basis['observed_share_of_this_floor']:.0%}）"
        if entity["entityId"] in prompted:
            base = prompted[entity["entityId"]]
            checked = json.loads((base / "validation.json").read_text())
            data, display = light_model(base / "model.glb")
            (args.output / f"{identity}-model.glb").write_bytes(data)
            record["generatedModel"] = {"meshUrl": f"{identity}-model.glb", "sha256": hashlib.sha256(data).hexdigest(), "provenanceUrl": relative(base / "validation.json"),
                                        "sourceFrame": int(index), "status": "source_consistent_model_estimate", "display": display}
        elif anchored:
            checked = json.loads((models[chosen["observation"]] / "validation.json").read_text())
            record["generatedModel"] = {"meshUrl": relative(models[chosen["observation"]] / "model.glb"), "sha256": checked["mesh_sha256"],
                                        "provenanceUrl": relative(models[chosen["observation"]] / "validation.json"), "sourceFrame": int(index),
                                        "status": "source_consistent_model_estimate"}
        static.append(record)
    scene["staticObjects"] = static
    scene["limitations"] = scene.get("limitations", []) + ["可选中的静态对象来自对象地图中被≥3个视图确认的实体，每个只显示其中一个视图的可见表面；生成模型的未见面是生成器的估计（非商用许可）。"]
    (args.output / "scene.json").write_text(json.dumps(scene, allow_nan=False, separators=(",", ":")))
    print(json.dumps({"staticObjects": len(static), "withGeneratedModel": sum("generatedModel" in s for s in static)}))


if __name__ == "__main__":
    main()
