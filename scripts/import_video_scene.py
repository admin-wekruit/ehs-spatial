"""Offline, zero-model-call import of a video-derived scene into the platform, reusing the import_scene job flow.

What enters: rectified keyframes as the source images with their DROID cameras, the fused room mesh as source context
(observed, not accepted), the fitted floor and the scale record, and the video object map's entities with their mask
observations. Entities seen in fewer than three views stay association_pending and carry no measurements.
Run with a venv that has psycopg (the platform venv) and PANOPTES_DATABASE_URL set. Served locally only.

  python scripts/import_video_scene.py --droid-run RUN --depth-run FUSED --object-map DIR --masks ROOT [--policy DIR] --title T
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import secrets
import sys
from uuid import NAMESPACE_URL, uuid5

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "modal_apps"), str(ROOT / "scripts")]
from ehs_spatial.platform.contracts import digest, empty_document, validate_document  # noqa: E402

FRAME, CONFIRMED = "droid_final_native_world", 3
STUFF = ("floor", "wall", "ceiling")  # as build_video_object_map.STUFF: extents, not things
AGNOSTIC, EVIDENCE_PER_AGNOSTIC = "object", 8  # segment-everything label; how many of its views a report entity keeps as evidence


def rectify(image, calibration=None):
    """The exact 640x480 raster every depth, mask and camera of this pipeline lives on, for whichever clip the run names."""
    import mono_room
    return mono_room.prepare_image(image, mono_room.CALIBRATION, 2)


def extent_box(entity, plan):
    """A floor-aligned box over what was seen of a confirmed entity: the minimum rectangle of its floor footprint and its observed bottom and top.

    It is a parametric stand-in for the Blender/model view, not a measurement: nothing is extended to the floor or behind the seen side.
    """
    from scipy.spatial.transform import Rotation
    from shapely.geometry import Polygon
    rectangle = np.array(Polygon(entity["footprintPlanNative"]).minimum_rotated_rectangle.exterior.coords[:4])
    along, across = rectangle[1] - rectangle[0], rectangle[3] - rectangle[0]
    size = [float(np.linalg.norm(along)), float(np.linalg.norm(across)), max(entity["heightNative"] - entity["baseNative"], 1e-3)]
    if min(size[:2]) <= 1e-6:
        return None
    origin, up, a, b = (np.array(plan[k]) for k in ("origin_native", "up", "axis_a", "axis_b"))
    x = (along / size[0]) @ np.stack([a, b])
    rotation = np.stack([x, np.cross(up, x), up], 1)  # columns: box x, y, z in the world
    centre = rectangle.mean(0)
    position = origin + a * centre[0] + b * centre[1] + up * (entity["heightNative"] + entity["baseNative"]) / 2
    return {"type": "box", "dimensions": size}, {"coordinateFrameId": FRAME, "position": position.tolist(),
                                                 "quaternion": Rotation.from_matrix(rotation).as_quat().tolist(), "scale": [1., 1., 1.]}


def atlas(patches, limit=2048):
    """Pack image patches on shelves. Returns the atlas and each patch's (x, y) offset and scale in it."""
    area = sum(p.shape[0] * p.shape[1] for p in patches)
    shrink = min(1., limit / (1.25 * np.sqrt(area) + max(p.shape[1] for p in patches)))  # only a whole wall's worth of photos ever needs it
    patches = [p if shrink == 1 else cv2.resize(p, (max(1, int(p.shape[1] * shrink)), max(1, int(p.shape[0] * shrink))), interpolation=cv2.INTER_AREA) for p in patches]
    width = max(max(p.shape[1] for p in patches), int(1.25 * np.sqrt(sum(p.shape[0] * p.shape[1] for p in patches))))
    places, x, y, shelf = [None] * len(patches), 0, 0, 0
    for n in sorted(range(len(patches)), key=lambda n: -patches[n].shape[0]):
        h, w = patches[n].shape[:2]
        if x + w > width:
            x, y, shelf = 0, y + shelf, 0
        places[n], x, shelf = (x, y), x + w, max(shelf, h)
    sheet = np.zeros((y + shelf, width, 3), np.uint8)
    for (px, py), patch in zip(places, patches):
        sheet[py:py + patch.shape[0], px:px + patch.shape[1]] = patch
    return sheet, places, shrink


def fused_part(textured, object_map, entity, minimum=50):
    """The entity's cells cut out of the photo-textured fused mesh, or None if too little falls inside.

    One mesh with one small texture: only the pixels its triangles use are cut from each photo and packed together.
    Carrying the whole photos (8 per entity at the median) made a report's model layer ask for ~2900 full textures.
    """
    import trimesh
    from PIL import Image
    from build_video_object_map import pack
    data = np.load(object_map / entity["cells"]["file"])
    cells, cell = pack(data["cells"]), float(data["cell"])
    pieces, patches, plain = [], [], []
    for geometry in textured.geometry.values():
        inside = np.isin(pack(np.floor(np.asarray(geometry.vertices) / cell).astype(np.int64)), cells)
        chosen = np.flatnonzero(inside[geometry.faces].all(1))
        if not len(chosen):
            continue
        part = geometry.submesh([chosen], append=True)
        if getattr(part.visual, "uv", None) is None:  # triangles no photo saw fully keep their fused colour
            plain.append(part)
            continue
        photo = np.asarray(part.visual.material.baseColorTexture.convert("RGB"))
        height, width = photo.shape[:2]
        pixels = np.column_stack([part.visual.uv[:, 0] * width, (1 - part.visual.uv[:, 1]) * height])
        x0, y0 = np.maximum(np.floor(pixels.min(0)).astype(int) - 2, 0)
        x1, y1 = np.minimum(np.ceil(pixels.max(0)).astype(int) + 3, [width, height])
        pieces.append((part, pixels - [x0, y0]))
        patches.append(photo[y0:y1, x0:x1])
    parts, triangles = trimesh.Scene(), sum(len(p.faces) for p, _ in pieces) + sum(len(p.faces) for p in plain)
    if triangles < minimum:
        return None
    if pieces:
        sheet, places, shrink = atlas(patches)
        vertices, faces, uv, offset = [], [], [], 0
        for (part, pixels), (px, py) in zip(pieces, places):
            at = pixels * shrink + [px, py]
            vertices.append(part.vertices); faces.append(part.faces + offset); offset += len(part.vertices)
            uv.append(np.column_stack([at[:, 0] / sheet.shape[1], 1 - at[:, 1] / sheet.shape[0]]))
        jpeg = Image.open(io.BytesIO(cv2.imencode(".jpg", sheet[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes()))
        material = trimesh.visual.material.PBRMaterial(baseColorTexture=jpeg, metallicFactor=0., roughnessFactor=1.)
        parts.add_geometry(trimesh.Trimesh(np.vstack(vertices), np.vstack(faces), visual=trimesh.visual.TextureVisuals(uv=np.vstack(uv), material=material), process=False), geom_name="photo-atlas")
    if plain:
        parts.add_geometry(trimesh.util.concatenate(plain), geom_name="fused-colour-only")
    return parts


def png(image):
    return cv2.imencode(".png", image)[1].tobytes()


def build_document(args, put_asset, calibration, dataset):
    ident = lambda *parts: str(uuid5(NAMESPACE_URL, "video-import:" + ":".join(map(str, parts))))
    object_map = json.loads((args.object_map / "object-map.json").read_text())
    scale = json.loads((args.depth_run / "metric-scale.json").read_text())
    manifest = json.loads((args.droid_run / "input-manifest.json").read_text())
    import mono_room
    device = mono_room.METRIC_CAMERAS  # set by use_clip in run(): metric poses from the capture device, no assumed scale
    prediction = np.load(args.droid_run / "prediction.npz")
    keyframes = {int(i): c for i, c in zip(prediction["keyframe_source_indices"], prediction["keyframe_c2w"])}
    document = empty_document()

    def include(data, media_type, metadata):
        asset = put_asset(data, media_type, metadata)
        if asset["id"] not in {a["id"] for a in document["assets"]}:
            document["assets"].append({**asset, **metadata})
        return asset["id"]

    source_bytes = (args.object_map / "object-map.json").read_bytes()
    source = include(source_bytes, "application/json", {"kind": "import_source", "sourceSha256": hashlib.sha256(source_bytes).hexdigest()})
    up, origin = np.array(scale["up_native"]), np.array(scale["plane_point_native"])
    document["coordinateFrames"] = [{"id": FRAME, "convention": "opencv",
        "scale": {"status": "operator_anchored", "nativeToMeters": scale["metres_per_native_unit"], "sourceRefs": [{"assetId": source}],
                  "anchor": {"kind": "device_metric_poses", "measured": True, "note": "metres come from the capture device's own poses; nothing was assumed"} if device else
                            {"kind": "stated_carry_height", "metres": 1.6, "measured": False,
                             "modelEstimateMetresPerNative": scale.get("model_estimated_metres_per_native_unit"),
                             "sourcesDisagreeOver10pct": scale.get("scale_sources_disagree_over_10pct")}},
        "ground": {"plane": [*up.tolist(), float(-up @ origin)], "normal": up.tolist(), "offset": float(-up @ origin),
                   "sourceRefs": [{"assetId": source}], "source": "video_floor_consensus_plane"}}]

    # A dense segment-everything map holds thousands of views. The report keeps every prompted view, the largest few class-agnostic
    # views of an entity, and no class-agnostic fragment that fewer than three views confirm; the counts left out are recorded.
    agnostic = lambda o: o.startswith(AGNOSTIC + ":")
    skipped = [e["entityId"] for e in object_map["entities"] if len(e["observations"]) < CONFIRMED and all(map(agnostic, e["observations"]))]
    object_map["entities"] = [e for e in object_map["entities"] if e["entityId"] not in skipped]
    for e in object_map["entities"]:
        area = lambda o: np.prod(np.subtract(e["observationBoxes"][o][2:], e["observationBoxes"][o][:2]))
        kept = [o for o in e["observations"] if not agnostic(o)] + sorted(filter(agnostic, e["observations"]), key=area, reverse=True)[:EVIDENCE_PER_AGNOSTIC]
        e["observationsNotImported"] = len(e["observations"]) - len(kept)
        e["observationBoxes"] = {o: e["observationBoxes"][o] for o in kept}
    used = sorted({int(o.split(":")[1]) for e in object_map["entities"] for o in e["observationBoxes"]})
    images = {}
    for index in used:
        record = manifest["frames"][index]
        rectified, k = rectify(cv2.imread(str(dataset / record["relative_path"])), calibration)
        if device:  # the device refocuses per frame; the raster only knows the clip's median K
            k = prediction["keyframe_final_fullres_intrinsics"][index].astype(float) * 2
        images[index] = include(png(rectified), "image/png", {"kind": "source_image", "width": 640, "height": 480, "sourceFrame": index,
            "videoTimestamp": record["timestamp_text"], "sourceSha256": record["sha256"],
            "pixelMapping": [{"source": "original_pixels", "target": "canonical_pixels", "coordinateConvention": "pixel_centers", "matrix": np.eye(3).tolist()}]})
        document["cameras"].append({"id": ident("camera", index), "imageId": images[index], "coordinateFrameId": FRAME, "width": 640, "height": 480,
            "K": [[float(k[0]), 0., float(k[2])], [0., float(k[1]), float(k[3])], [0., 0., 1.]], "cameraToWorld": np.asarray(keyframes.get(index, prediction["poses_c2w"][index]), float).tolist(),  # a view between keyframes has DROID's filler camera, as in fusion
            "sourceRefs": [{"assetId": source, "sourceCameraId": f"droid-keyframe-{index}"}]})

    import trimesh
    from ehs_spatial.platform.reconstruction import _plan_projection
    from ehs_spatial.platform.spatial import primitive_mesh
    shell = trimesh.load(args.depth_run / "predicted-scene.glb", force="mesh", process=False)
    rows = np.hstack([shell.vertices, shell.vertex_normals, np.asarray(shell.visual.vertex_colors)[:, :3] / 255.]).astype("<f4")
    indices = np.asarray(shell.faces, "<u4").ravel()
    layout = {"stride": 9, "byteOffset": 0, "vertexCount": len(rows), "indexByteOffset": rows.nbytes, "indexCount": len(indices), "indexType": "uint32"}
    shell_asset = include(rows.tobytes() + indices.tobytes(), "application/octet-stream", {"kind": "geometry", "format": "panoptes-mesh-v1", "byteLayout": layout,
                          "sourceRecordId": "fused-room-surface"}) if not args.shell_glb else include(  # same geometry, appearance from the source photos (texture_fused_mesh.py)
        args.shell_glb.read_bytes(), "model/gltf-binary", {"kind": "geometry", "format": "glb", "sourceRecordId": "fused-room-surface-photo-textured"})
    identity = {"coordinateFrameId": FRAME, "position": [0., 0, 0], "quaternion": [0., 0, 0, 1], "scale": [1., 1, 1]}
    document["entities"].append({"id": ident("entity", "room"), "label": "observed room surface (not accepted)", "observationRefs": [],
        "associationState": "association_pending", "currentModelTransform": None, "measurements": {}, "groupId": None, "visible": True, "sourceContext": True,
        "lineage": [{"operation": "offline_import", "sourceAssetId": source, "sourceRecordId": "fused-room-surface"}],
        # placed exactly by construction (same cameras); "not accepted" is a statement about completeness, carried by the label and the provenance note
        "representations": [{"id": ident("representation", "room"), "kind": "observed_surface", "assetId": shell_asset, "coordinateFrameId": FRAME,
                             "transform": identity, "placementState": "confirmed", "primitive": None, "coverage": "observed_only_not_accepted", "sourceRefs": [{"assetId": source}],
                             "bounds": {"min": shell.bounds[0].tolist(), "max": shell.bounds[1].tolist()}},
                            {"id": ident("representation", "room-points"), "kind": "point_cloud", "coordinateFrameId": FRAME, "transform": identity, "placementState": "confirmed",
                             "primitive": None, "coverage": "cross_view_supported_pixels_only", "sourceRefs": [{"assetId": source}],
                             "assetId": include((args.depth_run / "supported-keyframe-points.glb").read_bytes(), "model/gltf-binary", {"kind": "geometry", "format": "glb-points", "sourceRecordId": "supported-keyframe-points"}),
                             "bounds": {"min": shell.bounds[0].tolist(), "max": shell.bounds[1].tolist()}}]})

    textured = trimesh.load(args.shell_glb, process=False) if args.shell_glb else None
    facts = json.loads((args.policy / "scene-document.json").read_text())["entities"] if args.policy else []
    facts = {e["id"]: e["measurements"] for e in facts}
    for entity in object_map["entities"]:
        refs = []
        for oid, box in entity["observationBoxes"].items():
            label, index, instance = oid.split(":")
            path = next(args.masks.glob(f"{label}-*/frame-{int(index):05d}/instance-{instance}-mask.png"))
            mask = (rectify(cv2.imread(str(path), cv2.IMREAD_COLOR), calibration)[0][..., 0] > 0).astype(np.uint8) * 255
            mask_asset = include(png(mask), "image/png", {"kind": "source_mask", "resolution": "original", "sourceRecordId": oid})
            observation = ident("observation", oid)
            document["observations"].append({"id": observation, "revision": 1, "imageId": images[int(index)], "originalPixelBox": box, "maskAssetId": mask_asset,
                "labelEvidence": [{"label": entity["label"], "source": "vlm_name_of_class_agnostic_segment" if entity.get("labelSource") else "sam3_text_prompt"}], "geometrySupport": None, "sourceRefs": [{"assetId": source, "sourceRecordId": oid}]})
            refs.append((oid, observation))
        measurements = {}
        for name, fact in facts.get(entity["entityId"], {}).items():  # the policy run's own metric facts, re-pointed at this document's observations
            known = dict(refs)
            kept = [known[o] for o in fact["sourceRefs"] if o in known]  # views beyond the report's evidence cap are not in this document
            if kept:
                measurements[name] = {**fact, "sourceRefs": kept}
        representations = []
        for seen in (x for x in entity.get("surfaces", []) if x["observation"] in dict(refs)):  # as reconstruction.py writes them: one observed surface per imported observation, measured in place
            data = np.load(args.object_map / seen["file"])
            part = trimesh.Trimesh(data["vertices"], data["faces"], process=False)
            packed = np.hstack([part.vertices, part.vertex_normals, data["colors"] / 255.]).astype("<f4")
            triangles = np.asarray(part.faces, "<u4").ravel()
            payload = packed.tobytes() + triangles.tobytes()
            asset = include(payload, "application/octet-stream", {"kind": "geometry", "format": "panoptes-mesh-v1", "sourceRecordId": seen["observation"],
                "byteLayout": {"stride": 9, "byteOffset": 0, "vertexCount": len(packed), "indexByteOffset": packed.nbytes, "indexCount": len(triangles), "indexType": "uint32"}})
            surface = {"id": ident("representation", seen["observation"]), "kind": "observed_surface", "assetId": asset, "coordinateFrameId": FRAME, "transform": identity,
                       "placementState": "confirmed", "primitive": None, "coverage": "visible_support_only",
                       "bounds": {"min": part.bounds[0].tolist(), "max": part.bounds[1].tolist()},
                       "sourceRefs": [{"observationId": dict(refs)[seen["observation"]], "revision": 1, "imageId": images[int(seen["observation"].split(":")[1])]}]}
            # the photo report's own CAD outline: exact triangle union on the floor. A wall or floor has no footprint worth the minutes it costs.
            # ...nor has each of the many segment-everything views: their entity's fused multi-view surface carries one outline instead (below)
            projection = None if entity["label"] in STUFF or seen["observation"].startswith(AGNOSTIC + ":") else _plan_projection(document, surface, part, hashlib.sha256(payload).hexdigest())
            if projection is not None:
                surface["planProjection"] = projection
            representations.append(surface)
        model_transform = None
        # models are bound to the observation they were generated from, so renumbered entities cannot pick up a neighbour's model
        anchored = [v for v in (args.models.glob("*/validation.json") if args.models else []) if json.loads(v.read_text())["observation"] in dict(refs)]
        generated = anchored[0].parent if len(anchored) == 1 else None
        checked = json.loads(anchored[0].read_text()) if generated else None
        if checked and checked["accepted_source_consistency"]:  # a generated shape that reprojects onto its own source view; unseen sides are the generator's estimate
            shape = trimesh.load(generated / "model.glb", force="mesh", process=False)  # already posed in the scene frame
            asset = include((generated / "model.glb").read_bytes(), "model/gltf-binary", {"kind": "geometry", "format": "glb", "sourceRecordId": entity["entityId"],
                            "generator": "RecGen (non-commercial research licence)", "meshSha256": checked["mesh_sha256"]})
            model_transform = identity
            model = {"id": ident("representation", "generated", entity["entityId"]), "kind": "generated_mesh", "assetId": asset, "coordinateFrameId": FRAME, "transform": identity,
                     "bounds": {"min": shape.bounds[0].tolist(), "max": shape.bounds[1].tolist()}, "placementState": "unconfirmed", "placementReason": "requires_alignment_confirmation",
                     "sourceRefs": [{"observationId": dict(refs)[checked["observation"]], "revision": 1, "imageId": images[int(checked["observation"].split(":")[1])]}],
                     "sourceConsistency": {k: checked[k] for k in ("silhouette_iou", "relative_depth_median", "relative_depth_p95", "supported_pixels")},
                     "modelBasis": "generated from one keyframe crop with its posed depth; checked against that view only"}
            projection = _plan_projection(document, model, shape, checked["mesh_sha256"], identity)
            if projection is not None:
                model["planProjection"] = projection
            representations.append(model)
        if args.inferred_floor and entity["label"] == "floor" and refs:  # the floor's model: its verified plane over the room's footprint, stated as inferred, not observed
            basis = json.loads((args.inferred_floor / "inferred-floor.json").read_text())
            slab = trimesh.load(args.inferred_floor / "inferred-floor.glb", force="mesh", process=False)
            asset = include((args.inferred_floor / "inferred-floor.glb").read_bytes(), "model/gltf-binary", {"kind": "geometry", "format": "glb", "sourceRecordId": "inferred-floor",
                            "generator": "parametric plane extension, no model call", "meshSha256": basis["mesh_sha256"]})
            model_transform = identity
            representations.append({"id": ident("representation", "inferred-floor"), "kind": "generated_mesh", "assetId": asset, "coordinateFrameId": FRAME, "transform": identity,
                "bounds": {"min": slab.bounds[0].tolist(), "max": slab.bounds[1].tolist()}, "placementState": "unconfirmed", "placementReason": "requires_alignment_confirmation",
                "sourceRefs": [{"observationId": refs[0][1], "revision": 1, "imageId": images[int(refs[0][0].split(":")[1])]}],
                "modelBasis": f"inferred, not observed: {basis['basis']}; {basis['observed_share_of_this_floor']:.0%} of its {basis['area_m2']} m2 was seen"})
        if model_transform is None and len(refs) >= CONFIRMED and representations:
            # No checked model: the model view shows what was actually seen, textured, as the photo report does for fences and floors.
            fused = fused_part(textured, args.object_map, entity) if textured is not None and entity.get("cells") else None
            if fused is not None:  # every view's share of the entity in one piece: the entity's cells cut out of the photo-textured fused mesh
                payload = fused.export(file_type="glb")
                whole = trimesh.util.concatenate([trimesh.Trimesh(g.vertices, g.faces, process=False) for g in fused.geometry.values()])
                representations.append({"id": ident("representation", "fused", entity["entityId"]), "kind": "observed_surface", "coordinateFrameId": FRAME, "transform": identity,
                    "assetId": include(payload, "model/gltf-binary", {"kind": "geometry", "format": "glb", "sourceRecordId": f"{entity['entityId']}:fused-surface"}),
                    "placementState": "confirmed", "primitive": None, "coverage": "visible_support_only_all_views", "sourceKind": "observed_reference_surface",
                    "bounds": {"min": fused.bounds[0].tolist(), "max": fused.bounds[1].tolist()},
                    "sourceRefs": [{"observationId": o, "revision": 1, "imageId": images[int(oid.split(":")[1])]} for oid, o in refs]})
                outline = None if entity["label"] in STUFF else _plan_projection(document, representations[-1], whole, hashlib.sha256(payload).hexdigest())
                if outline is not None:
                    representations[-1]["planProjection"] = outline
            else:  # one surface only (the largest), so the same object is not stacked from several views. White extent boxes are gone.
                largest = max((r for r in representations if r["kind"] == "observed_surface"), key=lambda r: np.prod(np.subtract(r["bounds"]["max"], r["bounds"]["min"])))
                largest["sourceKind"] = "observed_reference_surface"
        document["entities"].append({"id": ident("entity", entity["entityId"]), "label": entity["label"], "observationRefs": [o for _, o in refs],
            "associationState": "confirmed" if len(refs) >= CONFIRMED else "association_pending", "representations": representations, "currentModelTransform": model_transform,
            "measurements": measurements, "groupId": None, "visible": True, "sourceContext": False,
            "lineage": [{"operation": "offline_import", "sourceAssetId": source, "sourceRecordId": entity["entityId"], "method": object_map["associator"]}]})
    if args.video and args.analysis:  # the source video as a report view: per-frame outlines carry this document's entity ids
        by_map_id = {f"obs-{e['entityId']}": ident("entity", e["entityId"]) for e in object_map["entities"]}
        analysis = json.loads(args.analysis.read_text())
        frames = [{"timeSec": f["timeSec"], "endTimeSec": f["endTimeSec"], "sourceFrame": f["sourceFrame"], "objects": [
            {"entityId": by_map_id.get(o["entityId"]), "label": o.get("label") or o.get("displayName") or "", "polygons": [[[round(x, 1), round(y, 1)] for x, y in polygon] for polygon in o["polygons"]]}
            for o in f["objects"] if o.get("polygons")]} for f in analysis["frames"]]
        slim = json.dumps({"width": analysis["width"], "height": analysis["height"], "frames": frames, "method": analysis.get("method")}, separators=(",", ":")).encode()
        document["annotations"].append({"id": ident("annotation", "video-replay"), "kind": "video_replay", "sourceAssetId": source,
            "videoAssetId": include(args.video.read_bytes(), "video/mp4", {"kind": "source_video", "sourceRecordId": "source-video"}),
            "analysisAssetId": include(slim, "application/json", {"kind": "video_frame_outlines", "sourceRecordId": "per-frame-outlines"}),
            "note": "outlines are the scene's 3D entities re-projected into each frame, not a per-frame segmentation"})
    left_out = {"unconfirmed_class_agnostic_fragments": len(skipped), "views_not_imported": sum(e["observationsNotImported"] for e in object_map["entities"])}
    limitations = [("Metres come from the capture device's poses; depth comes from a pretrained model conditioned on them." if device else
                    "Monocular video; metres come from a stated 1.6 m carry height that disagrees with the model scale estimate by about 20% on this clip."),
                   "Only what the camera saw is present; the room surface is source context and has not been accepted.",
                   "Entities are grouped by 3D point overlap of text-prompted masks; identities were not reviewed by a person."] + object_map["limitations"]
    document["annotations"].append({"id": ident("annotation", "provenance"), "kind": "import_provenance", "sourceAssetId": source, "limitations": limitations,
        "missingArtifacts": [], "recomputeRequiresNewCapture": True, "leftOutOfReport": left_out, "pipeline": {"cameras": str(args.droid_run), "depth": str(args.depth_run), "objects": str(args.object_map)}})
    return document, hashlib.sha256(source_bytes).hexdigest()


def run(args):
    import mono_room
    from ehs_spatial.platform.config import PlatformConfig
    from ehs_spatial.platform.runtime import services
    mono_room.use_clip(args.droid_run)
    clip = {"K": mono_room.SOURCE_K, "D": mono_room.SOURCE_D, "dataset": mono_room.DATASET}
    repository, blobs = services(PlatformConfig.from_env())
    repository.migrate()
    repository.blobs = blobs
    capability = "pcap_v1_" + secrets.token_urlsafe(32)
    request = lambda name: str(uuid5(NAMESPACE_URL, f"video-import:{name}:{args.object_map}:{args.depth_run}:{args.request_suffix}"))
    created = repository.create_project(capability, {"requestId": request("create"), "title": args.title, "target": "scene"})
    project = created["project"]["id"]

    def put_asset(data, media_type, metadata):
        asset = blobs.put(data, media_type)
        asset["metadata"] = metadata
        return repository.register_asset(project, asset)

    document, sha = build_document(args, put_asset, clip, clip["dataset"])
    capture = request("capture")
    document["captureId"] = capture
    validate_document(document)
    images = [{"id": c["imageId"], "assetId": c["imageId"], "width": c["width"], "height": c["height"], "sourceCameraId": c["sourceRefs"][0]["sourceCameraId"]} for c in document["cameras"]]
    imported = {"id": capture, "images": images, "task": {"schemaVersion": 1, "kind": "offline_import", "sourceSha256": sha, "imageIds": [i["id"] for i in images],
                "missingArtifacts": [], "recomputeRequiresNewCapture": True}}
    job = repository.create_job(project, capability, {"requestId": request("job"), "branchId": created["branch"]["id"], "baseRevisionId": created["revision"]["id"],
                                "kind": "import_scene", "inputs": {"sourceSha256": sha}, "config": {"offline": True, "converter": {"version": "video-scene-v1"}}})
    job = repository.claim_job(job["id"], lease_seconds=3600)
    job = repository.finish_job(job["id"], job["attemptToken"], "succeeded", document=document, result={"sourceSha256": sha, "newModelCalls": 0}, imported_capture=imported)
    assert job["status"] == "succeeded" and job.get("resultRevisionId"), job
    # The current reader only opens schema v2; the platform's own explicit migration makes the new immutable revision.
    migrated = repository.commit_edits(project, capability, {"requestId": request("migrate"), "branchId": created["branch"]["id"],
                                       "baseRevisionId": job["resultRevisionId"], "operations": [{"type": "migrateScene", "schemaVersion": 2}]})["revision"]
    publication = repository.create_publication(project, capability, {"requestId": request("publish"), "sceneRevisionId": migrated["id"],
                                                "evaluationIds": [], "reviewIds": [], "title": args.title})
    result = {"projectId": project, "publicationId": publication["id"], "importedRevisionId": job["resultRevisionId"], "sceneRevisionId": migrated["id"], "capability": capability,
              "images": len(images), "observations": len(document["observations"]), "entities": len(document["entities"]),
              "confirmedEntities": sum(e["associationState"] == "confirmed" for e in document["entities"]), "assets": len(document["assets"]),
              "newModelCalls": 0, "documentSha256": digest(document)}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / f"video-import-{publication['id']}.json"
    path.write_text(json.dumps(result, indent=1))
    path.chmod(0o600)  # holds the project capability
    print(json.dumps({k: v for k, v in result.items() if k != "capability"}, indent=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("droid-run", "depth-run", "object-map", "masks", "policy", "models"):
        parser.add_argument("--" + name, type=Path, required=name not in ("policy", "models"))
    parser.add_argument("--video", type=Path, help="source video to show as a report view (needs --analysis)")
    parser.add_argument("--analysis", type=Path, help="project_entities_to_frames analysis.json of the same object map")
    parser.add_argument("--inferred-floor", type=Path, help="infer_room_floor.py output: becomes the floor entity's model in the model layer")
    parser.add_argument("--shell-glb", type=Path, help="photo-textured copy of the depth run's fused mesh to show as the room surface")
    parser.add_argument("--title", default="Video workcell (imported, not accepted)")
    parser.add_argument("--output-dir", type=Path, default=Path(".platform/imports"))
    parser.add_argument("--request-suffix", default="1", help="change to import the same inputs again as a new project")
    run(parser.parse_args())
