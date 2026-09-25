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
    colours = glb_rgba(Path(path).read_bytes(), len(full.vertices))
    colours = vertex_rgba(full) if colours is None else colours
    seethrough = bool((colours[:, 3] < 255).any())  # a model's inferred, unseen part is drawn see-through
    if len(full.faces) <= triangles:
        data = Path(path).read_bytes()
        if seethrough and getattr(full.visual, "kind", None) != "texture":
            data = split_glb(np.asarray(full.vertices), np.asarray(full.faces), colours)
        return blend_glb(data) if seethrough else data, {"triangles": len(full.faces), "decimated": False}
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(full.vertices), o3d.utility.Vector3iVector(full.faces))
    mesh.vertex_colors = o3d.utility.Vector3dVector(colours[:, :3] / 255.)
    light = mesh.simplify_quadric_decimation(triangles)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    deviation = float(scene.compute_distance(o3d.core.Tensor(np.asarray(light.vertices, np.float32))).numpy().max())
    rgba = np.column_stack([(np.asarray(light.vertex_colors) * 255).round().astype(np.uint8), np.full(len(light.vertices), 255, np.uint8)])
    if seethrough:  # decimation keeps RGB only: alpha comes from the nearest original vertex
        from scipy.spatial import cKDTree
        rgba[:, 3] = colours[cKDTree(full.vertices).query(np.asarray(light.vertices))[1], 3]
    data = (split_glb(np.asarray(light.vertices), np.asarray(light.triangles), rgba) if seethrough else
            trimesh.Trimesh(np.asarray(light.vertices), np.asarray(light.triangles), vertex_colors=rgba, process=False).export(file_type="glb"))
    return blend_glb(data) if seethrough else data, {"triangles": len(light.triangles), "decimated": True, "from_triangles": len(full.faces),
                                                    "max_deviation_native": deviation, "validated_mesh_sha256": sha(path)}


def vertex_rgba(mesh):
    """Per-vertex RGBA uint8, whether trimesh read the colours as ColorVisuals or, beside a material, as the COLOR_0 vertex attribute."""
    colours = getattr(mesh.visual, "vertex_colors", None)
    if colours is None:
        colours = getattr(mesh.visual, "vertex_attributes", {}).get("color")
    colours = np.full((len(mesh.vertices), 4), 255, np.uint8) if colours is None else np.asarray(colours)
    if colours.dtype.kind == "f":
        colours = (colours * 255).round()
    if colours.shape[1] == 3:
        colours = np.column_stack([colours, np.full(len(colours), 255)])
    return colours.astype(np.uint8)


def glb_rgba(data, count):
    """COLOR_0 of a single-primitive GLB as RGBA uint8 straight from the bytes (trimesh drops it once a material is present), else None."""
    import struct
    length = struct.unpack_from("<I", data, 12)[0]
    doc, binary = json.loads(data[20:20 + length]), data[20 + length + 8:]
    primitives = [p for m in doc.get("meshes", []) for p in m["primitives"]]
    if len(primitives) != 1 or "COLOR_0" not in primitives[0]["attributes"]:
        return None
    accessor = doc["accessors"][primitives[0]["attributes"]["COLOR_0"]]
    view, width = doc["bufferViews"][accessor["bufferView"]], {"VEC3": 3, "VEC4": 4}[accessor["type"]]
    dtype = np.dtype({5121: np.uint8, 5123: np.uint16, 5126: np.float32}[accessor["componentType"]])
    if accessor["count"] != count or view.get("byteStride", width * dtype.itemsize) != width * dtype.itemsize:
        return None
    values = np.frombuffer(binary, dtype, count * width, view.get("byteOffset", 0) + accessor.get("byteOffset", 0)).reshape(count, width).astype(np.float64)
    values *= {"u1": 1., "u2": 255 / 65535, "f4": 255.}[dtype.str[1:]]
    return np.column_stack([values, np.full(count, 255.)])[:, :4].round().astype(np.uint8) if width == 3 else values.round().astype(np.uint8)


def split_glb(vertices, faces, rgba):
    """A model's seen faces (every corner alpha 255) as their own primitive named "observed", the rest as "inferred".

    blend_glb then makes only "inferred" see-through: one BLEND primitive for both wrote no depth, so guessed back faces showed
    through the seen front in triangle order.
    """
    import trimesh
    seen, scene = (rgba[faces][:, :, 3] == 255).all(1), trimesh.Scene()
    for name, chosen in (("observed", seen), ("inferred", ~seen)):
        if chosen.any():
            used, inverse = np.unique(faces[chosen], return_inverse=True)
            scene.add_geometry(trimesh.Trimesh(vertices[used], inverse.reshape(-1, 3), vertex_colors=rgba[used], process=False), geom_name=name)
    return scene.export(file_type="glb")


def blend_glb(data):
    """The GLB with one BLEND material on every primitive (but those of a mesh named "observed", which stay opaque): the viewer
    honours vertex alpha only in BLEND, and trimesh writes no material."""
    import struct
    length = struct.unpack_from("<I", data, 12)[0]
    doc = json.loads(data[20:20 + length])
    materials = doc.setdefault("materials", [])
    for material in materials:  # an existing material (a texture, a colour) stays; only its alpha mode changes
        material["alphaMode"] = "BLEND"
    for mesh in doc["meshes"]:
        for primitive in mesh["primitives"]:
            if "material" not in primitive and mesh.get("name") != "observed":
                if not materials or materials[-1].get("name") != "vertex-alpha":
                    materials.append({"name": "vertex-alpha", "alphaMode": "BLEND", "doubleSided": True,
                                      "pbrMetallicRoughness": {"baseColorFactor": [1, 1, 1, 1], "metallicFactor": 0, "roughnessFactor": 1}})
                primitive["material"] = len(materials) - 1
    text = json.dumps(doc, separators=(",", ":")).encode()
    text += b" " * (-len(text) % 4)
    rest = data[20 + length:]
    return struct.pack("<3I", 0x46546C67, 2, 20 + len(text) + len(rest)) + struct.pack("<2I", len(text), 0x4E4F534A) + text + rest


def self_check():
    """A half see-through sphere keeps both alphas and gets a BLEND material, decimated or not."""
    import struct
    import tempfile
    import trimesh
    sphere = trimesh.creation.icosphere(4)
    rgba = np.full((len(sphere.vertices), 4), 200, np.uint8)
    rgba[sphere.vertices[:, 0] < 0, 3] = 90
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "m.glb"
        path.write_bytes(trimesh.Trimesh(sphere.vertices, sphere.faces, vertex_colors=rgba, process=False).export(file_type="glb"))
        for limit, decimated in ((100000, False), (400, True)):
            data, record = light_model(path, limit)
            doc = json.loads(data[20:20 + struct.unpack_from("<I", data, 12)[0]])
            assert record["decimated"] is decimated and all(m["alphaMode"] == "BLEND" for m in doc["materials"]), record
            assert all("material" in p for m in doc["meshes"] for p in m["primitives"])
            alphas = set(np.unique(glb_rgba(data, record["triangles"] and len(trimesh.load(trimesh.util.wrap_as_stream(data), file_type="glb", force="mesh", process=False).vertices))[:, 3]))
            assert {90, 200} <= alphas, alphas
        seen = rgba.copy()
        seen[:, 3] = np.where(sphere.vertices[:, 0] < 0, 90, 255)
        half = Path(folder) / "half-seen.glb"
        half.write_bytes(trimesh.Trimesh(sphere.vertices, sphere.faces, vertex_colors=seen, process=False).export(file_type="glb"))
        for limit in (100000, 400):  # the seen half is its own opaque primitive (writes depth), the guessed half see-through
            data = light_model(half, limit)[0]
            doc = json.loads(data[20:20 + struct.unpack_from("<I", data, 12)[0]])
            modes = {m["name"]: doc["materials"][p["material"]]["alphaMode"] if "material" in p else "OPAQUE" for m in doc["meshes"] for p in m["primitives"]}
            assert modes == {"observed": "OPAQUE", "inferred": "BLEND"}, modes
        textured = trimesh.Trimesh(sphere.vertices, sphere.faces, process=False)  # a textured primitive keeps its texture when made see-through
        textured.visual = trimesh.visual.TextureVisuals(uv=np.zeros((len(sphere.vertices), 2)), material=trimesh.visual.material.PBRMaterial(
            baseColorTexture=__import__("PIL.Image", fromlist=["Image"]).new("RGB", (4, 4), (10, 200, 30))))
        kept = blend_glb(textured.export(file_type="glb"))
        materials = json.loads(kept[20:20 + struct.unpack_from("<I", kept, 12)[0]])["materials"]
        assert len(materials) == 1 and "baseColorTexture" in materials[0]["pbrMetallicRoughness"] and materials[0]["alphaMode"] == "BLEND", materials
        blended = Path(folder) / "blended.glb"  # an input that already carries a material still keeps its alpha
        blended.write_bytes(blend_glb(path.read_bytes()))
        assert {90, 200} <= set(np.unique(light_model(blended, 400)[0] and glb_rgba(light_model(blended, 100000)[0], len(sphere.vertices))[:, 3]))
    print("light model check passed: see-through vertices keep their alpha through decimation, the GLB gets a BLEND material, a seen side stays opaque")


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
    import sys
    self_check() if sys.argv[1:] == ["--self-check"] else main()
