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
from evaluate_video_policy import contract_scale  # noqa: E402  one rule for what a scale record may claim

FRAME, CONFIRMED = "droid_final_native_world", 3
STUFF = ("floor", "wall", "ceiling")  # as build_video_object_map.STUFF: extents, not things
FUSED_SHARE = .5  # the fused cut of an entity is shown only if it holds at least this share of the area its best single view shows
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


def dense_points(folder, include):
    """The room's point cloud as a dense display map (lingbot_dense_map.py / dense_point_map.py): one point per cell, drawn at its cell size."""
    info = json.loads((folder / "points.json").read_text())
    cell = info.get("cell_native") or info.get("cellNative")
    assert cell and cell > 0, "points.json must give the cell size in native units"
    return {"coverage": "dense_display_points_single_view", "pointSizeNative": float(cell), "pointCount": info.get("points") or info.get("count"),
            "assetId": include((folder / "dense-points.glb").read_bytes(), "model/gltf-binary", {"kind": "geometry", "format": "glb-points", "sourceRecordId": "dense-display-points",
                                                                                          "pointSizeNative": float(cell)}),
            "note": info.get("note") or "dense display points: single-view geometry per point, not multi-view confirmed"}


def inferred_floor_points(folder, include, ident, identity, source, shell):
    """infer_room_floor.py --dense: floor points where none was seen, on the verified plane under and between things seen
    standing on it; drawn with the room's points, dimmed, and stated as inferred, never observed."""
    basis = json.loads((folder / "inferred-floor.json").read_text())
    spacing = float(basis["dense"]["point_spacing_native"])
    return {"id": ident("representation", "room-inferred-floor-points"), "kind": "point_cloud", "coordinateFrameId": FRAME, "transform": identity,
            "placementState": "unconfirmed", "primitive": None, "sourceRefs": [{"assetId": source}],
            "bounds": {"min": shell.bounds[0].tolist(), "max": shell.bounds[1].tolist()},
            "coverage": "inferred_floor_points_not_observed", "pointSizeNative": spacing, "pointCount": basis["dense"]["points"],
            "assetId": include((folder / "inferred-floor-points.glb").read_bytes(), "model/gltf-binary",
                               {"kind": "geometry", "format": "glb-points", "sourceRecordId": "inferred-floor-points", "pointSizeNative": spacing}),
            "note": f"inferred, not observed: {basis['dense']['region']}; {basis['dense']['colour_rule']}"}


def fused_part(textured, object_map, entity, minimum=50):
    """The entity's cells cut out of the photo-textured fused mesh, or None if too little falls inside.

    One mesh with one small texture: only the pixels its triangles use are cut from each photo and packed together.
    Carrying the whole photos (8 per entity at the median) made a report's model layer ask for ~2900 full textures.
    """
    import trimesh
    from PIL import Image
    from build_video_object_map import pack
    from texture_fused_mesh import tiles
    data = np.load(object_map / entity["cells"]["file"])
    cells, cell = pack(entity["walkCells"] if "walkCells" in entity else data["cells"]), float(data["cell"])  # walkCells: without a cut-away shot
    pieces, patches, plain = [], [], []
    for name, geometry in textured.geometry.items():
        if name.startswith("single-view-depth-fill"):  # one view's own depth, not the fused surface this cut claims to be
            continue
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
        # An atlas page packs many views' tiles: one box over all the cut's UVs would take most of the page. Crop per tile.
        for ids, (x0, y0, x1, y1) in tiles(pixels, np.asarray(part.faces), (width, height)):
            # the tile's own vertices and pixels, as submesh would order them; submesh also gave every tile its own copy of the whole
            # atlas page, which a 4096x3956 page (Walmart) and a floor of thousands of tiles took past 20 GB
            used, faces = np.unique(np.asarray(part.faces)[ids], return_inverse=True)
            pieces.append((trimesh.Trimesh(np.asarray(part.vertices)[used], faces.reshape(-1, 3), process=False), pixels[used] - [x0, y0]))
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
    # A cut-away shot (an edited clip jumping to another camera and lens) is not part of the walk: its frames' cameras and depth
    # place things wrongly, so nothing 3D comes from them. The video still plays them; only their moving-object masks are drawn.
    cuts = [tuple(int(v) for v in span.split(":")) for span in args.exclude_frames or []]
    assert all(a < b for a, b in cuts), "--exclude-frames spans are START:END with START < END"
    walk = lambda frame: not any(a <= frame < b for a, b in cuts)
    entities_before_cuts = len(object_map["entities"])
    for e in object_map["entities"]:
        e["observations"] = [o for o in e["observations"] if walk(int(o.split(":")[1]))]
        e["observationBoxes"] = {o: box for o, box in e["observationBoxes"].items() if walk(int(o.split(":")[1]))}
    object_map["entities"] = [e for e in object_map["entities"] if e["observations"]]
    from build_video_object_map import pack
    no_walk_3d = []  # ponytail: dropped, not re-lifted from the walk views' masks; lift them if an important object lands here
    for e in object_map["entities"]:
        if all(map(walk, e["sourceFrames"])):
            continue
        # Also seen in the cut-away shot: its cells and centre were built from those views too, so they are rebuilt from its
        # walk views' own surfaces (never more than its agreed cells), or dropped when none of those views kept a surface.
        e["sourceFrames"] = [f for f in e["sourceFrames"] if walk(f)]
        seen = [np.load(args.object_map / x["file"])["vertices"] for x in e.get("surfaces", []) if walk(int(x["observation"].split(":")[1]))]
        points = np.concatenate(seen) if seen else np.empty((0, 3))
        e["centroidNative"] = np.median(points, 0).tolist() if len(points) else None
        if e.get("cells") and len(points):
            data = np.load(args.object_map / e["cells"]["file"])
            cells = np.unique(np.floor(points / float(data["cell"])).astype(np.int64), axis=0)
            e["walkCells"] = cells[np.isin(pack(cells), pack(data["cells"]))]
        elif not len(points):
            no_walk_3d += [e["entityId"]] if e.pop("cells", None) else []
    scale = json.loads((args.depth_run / "metric-scale.json").read_text())
    manifest = json.loads((args.droid_run / "input-manifest.json").read_text())
    import mono_room
    device = mono_room.METRIC_CAMERAS  # set by use_clip in run(): metric poses from the capture device, no assumed scale
    uncalibrated = scale.get("scale_status") == "uncalibrated"  # no device metres, no floor mask, no stated height: native units, no metres claimed
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
        "scale": {**contract_scale(scale), **({} if scale.get("scale_status") == "uncalibrated" else {"sourceRefs": [{"assetId": source}]})},
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
                             **({"singleViewFill": (lambda f: {"trianglesAdded": f["triangles_added"], "trianglesFused": f["triangles_fused"], "note": f["note"]})(
                                 json.loads((args.shell_glb.parent / "fill.json").read_text()))} if args.shell_glb and (args.shell_glb.parent / "fill.json").exists() else {}),
                             "bounds": {"min": shell.bounds[0].tolist(), "max": shell.bounds[1].tolist()}},
                            {"id": ident("representation", "room-points"), "kind": "point_cloud", "coordinateFrameId": FRAME, "transform": identity, "placementState": "confirmed",
                             "primitive": None, "sourceRefs": [{"assetId": source}], "bounds": {"min": shell.bounds[0].tolist(), "max": shell.bounds[1].tolist()},
                             **(dense_points(args.dense_points, include) if args.dense_points else
                                {"coverage": "cross_view_supported_pixels_only", "assetId": include((args.depth_run / "supported-keyframe-points.glb").read_bytes(), "model/gltf-binary",
                                                                                                      {"kind": "geometry", "format": "glb-points", "sourceRecordId": "supported-keyframe-points"})})},
                            *([inferred_floor_points(args.inferred_floor, include, ident, identity, source, shell)]
                              if args.inferred_floor and (args.inferred_floor / "inferred-floor-points.glb").exists() else [])]})

    textured = trimesh.load(args.shell_glb, process=False) if args.shell_glb else None
    floors = [e for e in object_map["entities"] if e["label"] == "floor" and len(e["observationBoxes"]) >= CONFIRMED]
    floor_owner = max(floors, key=lambda e: len(e["observations"]))["entityId"] if floors else None  # several floor pieces, one floor model
    facts = json.loads((args.policy / "scene-document.json").read_text())["entities"] if args.policy else []
    facts = {e["id"]: e["measurements"] for e in facts}
    used_fused = used_view = 0
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
            if kept and all(walk(int(o.split(":")[1])) for o in fact["sourceRefs"]):  # a fact measured with cut-away views is not kept
                measurements[name] = {**fact, "sourceRefs": kept}
        representations, views = [], {}  # views: each imported single-view surface by representation id, to weigh the fused cut against
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
            views[surface["id"]] = (part, hashlib.sha256(payload).hexdigest())
        model_transform = None
        # models are bound to the observation they were generated from, so renumbered entities cannot pick up a neighbour's model
        anchored = [v for v in (args.models.glob("*/validation.json") if args.models else []) if json.loads(v.read_text())["observation"] in dict(refs)
                    or (v.parent / "anchor.json").exists() and json.loads(v.read_text())["entity"] == entity["entityId"]]  # a name-prompted model is bound by entity
        generated = anchored[0].parent if len(anchored) == 1 else None
        checked = json.loads(anchored[0].read_text()) if generated else None
        if checked and checked.get("mesh_sha256") not in (None, hashlib.sha256((generated / "model.glb").read_bytes()).hexdigest()):
            raise SystemExit(f"{generated}: model.glb is not the mesh its validation.json judged")  # a half-written or replaced model is never imported
        if checked and checked["accepted_source_consistency"]:  # a generated shape that reprojects onto its own source view; unseen sides are the generator's estimate
            shape = trimesh.load(generated / "model.glb", force="mesh", process=False)  # already posed in the scene frame
            from attach_entities_to_replay import light_model
            light, display = light_model(generated / "model.glb")  # the checked shape, light enough for a page; the validated file stays as it is
            asset = include(light, "model/gltf-binary", {"kind": "geometry", "format": "glb", "sourceRecordId": entity["entityId"],
                            "generator": checked.get("generator") or "RecGen (non-commercial research licence)", "meshSha256": checked["mesh_sha256"], "display": display})
            model_transform = identity
            model = {"id": ident("representation", "generated", entity["entityId"]), "kind": "generated_mesh", "assetId": asset, "coordinateFrameId": FRAME, "transform": identity,
                     "bounds": {"min": shape.bounds[0].tolist(), "max": shape.bounds[1].tolist()}, "placementState": "unconfirmed", "placementReason": "requires_alignment_confirmation",
                     "sourceRefs": [{"observationId": dict(refs).get(checked["observation"], refs[0][1]), "revision": 1,
                                     "imageId": images[int((checked["observation"] if checked["observation"] in dict(refs) else refs[0][0]).split(":")[1])]}],
                     "sourceConsistency": {k: checked[k] for k in ("silhouette_iou", "relative_depth_median", "relative_depth_p95", "supported_pixels")},
                     "modelBasis": (f"{checked['generator']}: generated from one full-resolution video view with its posed depth, then fitted to the object's observed "
                                    f"points (residual {checked['fitResidualCm']:.1f} cm); {checked['observedShare']:.0%} of its surface was seen, the see-through rest is the generator's estimate"
                                    if "fitResidualCm" in checked else
                                    "generated from the frame that holds the whole object, with a whole-object mask from a segmentation prompted by the entity's name; checked against that view only"
                                    if (generated / "anchor.json").exists() else "generated from one keyframe crop with its posed depth; checked against that view only"),
                     **({"inferredDisplay": {"alpha": checked["alpha"], "observedRule": checked.get("observedRule"), "observedShare": checked.get("observedShare")}} if "alpha" in checked else {})}
            projection = _plan_projection(document, model, shape, checked["mesh_sha256"], identity)
            if projection is not None:
                model["planProjection"] = projection
            representations.append(model)
        if args.inferred_floor and entity["entityId"] == floor_owner and refs:  # the floor's model: its verified plane over the room's footprint, stated as inferred, not observed
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
            whole = None if fused is None else trimesh.util.concatenate([trimesh.Trimesh(g.vertices, g.faces, process=False) for g in fused.geometry.values()])
            seen_best = max(views, key=lambda r: views[r][0].area) if views else None
            # The fused mesh can lack an object the views agree on (free-space carving removes a near object whose depth differs by a few cm
            # between far and near views): a cut showing under half of what one photo shows is shards, so that photo's surface stands in.
            if whole is not None and (seen_best is None or whole.area >= FUSED_SHARE * views[seen_best][0].area):
                used_fused += 1
                payload = fused.export(file_type="glb")
                representations.append({"id": ident("representation", "fused", entity["entityId"]), "kind": "observed_surface", "coordinateFrameId": FRAME, "transform": identity,
                    "assetId": include(payload, "model/gltf-binary", {"kind": "geometry", "format": "glb", "sourceRecordId": f"{entity['entityId']}:fused-surface"}),
                    "placementState": "confirmed", "primitive": None, "coverage": "visible_support_only_all_views", "sourceKind": "observed_reference_surface",
                    "bounds": {"min": fused.bounds[0].tolist(), "max": fused.bounds[1].tolist()},
                    "sourceRefs": [{"observationId": o, "revision": 1, "imageId": images[int(oid.split(":")[1])]} for oid, o in refs]})
                outline = None if entity["label"] in STUFF else _plan_projection(document, representations[-1], whole, hashlib.sha256(payload).hexdigest())
                if outline is not None:
                    representations[-1]["planProjection"] = outline
            elif seen_best:  # one surface only (the largest), so the same object is not stacked from several views. White extent boxes are gone.
                used_view += 1
                largest = next(r for r in representations if r["id"] == seen_best)
                largest["sourceKind"] = "observed_reference_surface"
                if "planProjection" not in largest and entity["label"] not in STUFF:  # a segment-everything view got no outline above: its entity's one outline comes from here
                    outline = _plan_projection(document, largest, *views[seen_best])
                    if outline is not None:
                        largest["planProjection"] = outline
        document["entities"].append({"id": ident("entity", entity["entityId"]), "label": entity["label"], "observationRefs": [o for _, o in refs],
            "associationState": "confirmed" if len(refs) >= CONFIRMED else "association_pending", "representations": representations, "currentModelTransform": model_transform,
            "measurements": measurements, "groupId": None, "visible": True, "sourceContext": False,
            "lineage": [{"operation": "offline_import", "sourceAssetId": source, "sourceRecordId": entity["entityId"], "method": object_map["associator"]}]})
    moving = {}  # motion track id -> report entity: the dynamic layer, one timed surface per sampled view
    if args.dynamic_scene:
        from attach_entities_to_replay import light_model
        identity = {"coordinateFrameId": FRAME, "position": [0., 0., 0.], "quaternion": [0., 0., 0., 1.], "scale": [1., 1., 1.]}
        named = {o["entityId"]: o.get("sourceLabel") for f in (json.loads(args.dynamic_analysis.read_text())["frames"] if args.dynamic_analysis else []) for o in f["objects"]}
        for frame in (f for f in json.loads(args.dynamic_scene.read_text())["frames"] if walk(f["sourceFrame"])):
            for o in (x for x in frame["objects"] if x.get("surface")):
                data, record = light_model(args.dynamic_scene.parent / o["surface"]["meshUrl"], triangles=3000)  # a person's visible side needs no more
                shape = trimesh.load(io.BytesIO(data), file_type="glb", force="mesh", process=False)
                kind = "person" if named.get(o["entityId"]) == "person" else "moving object"  # a text-prompted person track says so; a motion track only knows it moved
                entity = moving.setdefault(o["entityId"], {"id": ident("entity", "moving", o["entityId"]), "label": f"{kind} {len(moving) + 1}", "motion": "dynamic",
                    "observationRefs": [], "associationState": "confirmed", "representations": [], "currentModelTransform": None, "measurements": {}, "groupId": None, "visible": True,
                    "sourceContext": False, "lineage": [{"operation": "offline_import", "sourceAssetId": source, "sourceRecordId": o["entityId"],
                                                         "method": "motion_masks.py seed -> sam3_motion_tracks.py (SAM 3.1 instance track, no text prompt) -> mono_room.py dynamic"}]})
                entity["representations"].append({"id": ident("representation", "moving", o["entityId"], frame["sourceFrame"]), "kind": "observed_surface", "coordinateFrameId": FRAME,
                    "assetId": include(data, "model/gltf-binary", {"kind": "geometry", "format": "glb", "sourceRecordId": f"{o['entityId']}:{frame['sourceFrame']}", "decimation": record}),
                    "transform": identity, "placementState": "confirmed", "primitive": None, "coverage": "visible_support_only", "sourceKind": "moving_object_surface",
                    "timeRange": [frame["timeSec"], frame["endTimeSec"]], "sourceFrame": frame["sourceFrame"], "bounds": {"min": shape.bounds[0].tolist(), "max": shape.bounds[1].tolist()}})
        if args.skeleton_scene:  # cached 2D joints lifted onto the same surfaces (mono_room.py dynamic): bones as their own timed layer
            owners = {f["sourceFrame"]: [o for o in f["objects"] if any(p is not None for p in o.get("keypoints3d") or [])] for f in json.loads(args.skeleton_scene.read_text())["frames"]}
            lengths = [np.linalg.norm(np.subtract(o["keypoints3d"][i], o["keypoints3d"][j])) for f in owners.values() for o in f for i, j in o["bones"] if o["keypoints3d"][i] and o["keypoints3d"][j]]
            radius = .06 * float(np.median(lengths)) if lengths else .01
            for frame in (f for f in json.loads(args.dynamic_scene.read_text())["frames"] if walk(f["sourceFrame"])):
                for o in (x for x in frame["objects"] if x["entityId"] in moving):
                    near = [c for c in owners.get(frame["sourceFrame"], []) if np.linalg.norm(np.subtract(c["centroid"], o["centroid"])) < .1]  # the same surface, lifted twice
                    if not near:
                        continue
                    joints = near[0]["keypoints3d"]
                    parts = [trimesh.creation.cylinder(radius, segment=[joints[i], joints[j]], sections=6) for i, j in near[0]["bones"] if joints[i] and joints[j]]
                    parts += [trimesh.creation.icosphere(1, radius * 1.6).apply_translation(q) for q in joints if q]
                    if not parts:
                        continue
                    bones = trimesh.util.concatenate(parts)
                    bones.visual.vertex_colors = np.tile([255, 196, 64, 255], (len(bones.vertices), 1))
                    data = bones.export(file_type="glb")
                    moving[o["entityId"]]["representations"].append({"id": ident("representation", "skeleton", o["entityId"], frame["sourceFrame"]), "kind": "observed_surface",
                        "coordinateFrameId": FRAME, "assetId": include(data, "model/gltf-binary", {"kind": "geometry", "format": "glb", "sourceRecordId": f"{o['entityId']}:{frame['sourceFrame']}:skeleton"}),
                        "transform": identity, "placementState": "confirmed", "primitive": None, "coverage": "visible_support_only", "sourceKind": "moving_object_skeleton",
                        "timeRange": [frame["timeSec"], frame["endTimeSec"]], "sourceFrame": frame["sourceFrame"], "bounds": {"min": bones.bounds[0].tolist(), "max": bones.bounds[1].tolist()}})
            for entity in moving.values():  # joints come only from person tracks: a mover wearing a skeleton in most of its views is that person
                reps = entity["representations"]
                bones = sum(r["sourceKind"] == "moving_object_skeleton" for r in reps)
                if bones >= .5 * (len(reps) - bones) and entity["label"].startswith("moving object"):
                    entity["label"] = entity["label"].replace("moving object", "person")
        from motion_facts import summarise
        factor = 1. if uncalibrated else scale["metres_per_native_unit"]
        statics = [(ident("entity", e["entityId"]), e["label"], np.array(e["centroidNative"]) * factor) for e in object_map["entities"] if e.get("centroidNative")]
        tracks = {}
        for frame in (f for f in json.loads(args.dynamic_scene.read_text())["frames"] if walk(f["sourceFrame"])):
            for o in frame["objects"]:
                if o["entityId"] in moving and o.get("centroid"):
                    tracks.setdefault(o["entityId"], []).append((frame["timeSec"], np.array(o["centroid"]) * factor))
        for key, samples in tracks.items():  # what an agent reads to answer "how did this person move": path, speed, stops, objects passed
            if len(samples) >= 2:
                moving[key]["motionSummary"] = summarise(samples, up, origin * factor, statics, unit="原生单位" if uncalibrated else "米")
        document["entities"].extend(moving.values())
    if args.video and args.analysis:  # the source video as a report view: per-frame outlines carry this document's entity ids
        by_map_id = {f"obs-{e['entityId']}": ident("entity", e["entityId"]) for e in object_map["entities"]}
        analysis = json.loads(args.analysis.read_text())
        outlines = {}  # moving objects' own masks, so a pick in the video selects the mover in 3D
        for f in (json.loads(args.dynamic_analysis.read_text())["frames"] if args.dynamic_analysis else []):
            for o in f["objects"]:
                if o["entityId"] not in moving:
                    continue
                mask = cv2.imread(str(args.dynamic_analysis.parent / o["maskUrl"]), cv2.IMREAD_GRAYSCALE) > 0
                contours = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
                polygons = [cv2.approxPolyDP(c, 1.5, True)[:, 0].tolist() for c in contours if cv2.contourArea(c) > 150]
                if polygons:
                    outlines.setdefault(f["sourceFrame"], []).append({"entityId": moving[o["entityId"]]["id"], "label": moving[o["entityId"]]["label"], "polygons": polygons})
        frames = [{"timeSec": f["timeSec"], "endTimeSec": f["endTimeSec"], "sourceFrame": f["sourceFrame"], "objects": [
            {"entityId": by_map_id.get(o["entityId"]), "label": o.get("label") or o.get("displayName") or "", "polygons": [[[round(x, 1), round(y, 1)] for x, y in polygon] for polygon in o["polygons"]]}
            for o in f["objects"] if o.get("polygons") and walk(f["sourceFrame"]) and not (outlines and by_map_id.get(o["entityId"]) is None)] + outlines.get(f["sourceFrame"], [])}  # with mover outlines, the old unlinked person outlines go
            for f in analysis["frames"]]
        slim = json.dumps({"width": analysis["width"], "height": analysis["height"], "frames": frames, "method": analysis.get("method")}, separators=(",", ":")).encode()
        document["annotations"].append({"id": ident("annotation", "video-replay"), "kind": "video_replay", "sourceAssetId": source,
            "videoAssetId": include(args.video.read_bytes(), "video/mp4", {"kind": "source_video", "sourceRecordId": "source-video"}),
            "analysisAssetId": include(slim, "application/json", {"kind": "video_frame_outlines", "sourceRecordId": "per-frame-outlines"}),
            "note": "outlines are the scene's 3D entities re-projected into each frame, not a per-frame segmentation",
            # the same frames without the 4:3 crop, shown in its place; outlines stay on the raster, placed at the crop
            **({"fullFrame": (lambda full: {"videoAssetId": include((args.full_video.parent / full["video"]).read_bytes(), "video/mp4", {"kind": "source_video_uncropped", "sourceRecordId": "source-video-full"}),
                                            "width": full["width"], "height": full["height"], "rasterInVideo": full["raster_in_video_xywh"]})(json.loads(args.full_video.read_text()))}
               if args.full_video else {})})
    if args.splats:  # photo-real appearance: 3D Gaussians fitted to the video frames, an appearance layer beside the measured surfaces
        info = json.loads(args.splats.with_suffix(".json").read_text())
        assert info["format"] == "splat32" and info["coordinateFrameId"] == FRAME, "splats must be splat32 records in the report's frame"
        assert isinstance(info["count"], int) and args.splats.stat().st_size == 32 * info["count"], "the .splat file must hold count 32-byte records"
        assert len(info["bounds"]["min"]) == len(info["bounds"]["max"]) == 3, "splats.json needs 3D bounds"
        refined = args.splats.parent / "refined-cameras.json"  # the trainer's refined poses: the sharpest cameras to follow the video with
        document["annotations"].append({"id": ident("annotation", "gaussian-splats"), "kind": "gaussian_splats", "coordinateFrameId": FRAME, "format": "splat32",
            "assetId": include(args.splats.read_bytes(), "application/octet-stream", {"kind": "gaussian_splats", "format": "splat32", "sourceRecordId": "gaussian-splats"}),
            "count": info["count"], "bounds": info["bounds"], "metrics": info.get("metrics"),
            **({"refinedCamerasAssetId": include(refined.read_bytes(), "application/json", {"kind": "refined_cameras", "sourceRecordId": "splat-refined-cameras"})} if refined.exists() else {}),
            "note": "3D Gaussians fitted to the video frames: sharp where the camera looked, not measured geometry and not used for any rule"})
    if args.video_events:  # the video memory: window captions and events in the model's own words, kept as evidence beside the geometry
        memory = json.loads(args.video_events.read_text())
        by_label = {e["label"]: e for e in document["entities"] if e.get("motion") == "dynamic"}
        for window in memory["windows"]:
            for event in window.get("events") or []:
                actor = by_label.get(event.get("actor"))
                event["entityId"] = actor["id"] if actor else None
                if actor:  # what the agent's get_entity returns for a mover: its geometry facts and what the video model said it did
                    actor.setdefault("videoEvents", []).append({k: event.get(k) for k in ("t0", "t1", "action", "near", "ppe", "safety_note")})
        document["annotations"].append({"id": ident("annotation", "video-events"), "kind": "video_events", "sourceAssetId": source, "model": memory["model"], "method": memory["method"],
            "windows": [{k: w.get(k) for k in ("t0", "t1", "caption", "events")} for w in memory["windows"]],
            "note": "model descriptions of each window; evidence for review and search, never a rule verdict"})
    left_out = {"cut_away_frames": args.exclude_frames or [], "entities_whose_3d_came_only_from_cut_away_views": no_walk_3d, "entities_seen_only_in_cut_away_frames": entities_before_cuts - len(object_map["entities"]) - len(skipped),
                "entities_shown_by_fused_cut": used_fused, "entities_shown_by_best_single_view": used_view, "unconfirmed_class_agnostic_fragments": len(skipped), "views_not_imported": sum(e["observationsNotImported"] for e in object_map["entities"])}
    if args.comparison_video:  # the clip split into static and dynamic layers, rendered from the clip's own camera (render_static_dynamic_video.py)
        rendered = json.loads(args.comparison_video.with_suffix(".json").read_text())
        document["annotations"].append({"id": ident("annotation", "static-dynamic"), "kind": "static_dynamic_comparison", "sourceAssetId": source,
            "videoAssetId": include(args.comparison_video.read_bytes(), "video/mp4", {"kind": "static_dynamic_comparison_video", "sourceRecordId": "static-dynamic"}),
            "panels": rendered["panels"], "frames": rendered["frames"],
            "note": "静态层＝融合前剔除运动像素的地图；动态层＝运动对象在每个采样视图的可见表面（运动线索种子＋SAM 3.1 跟踪，未用文字提示）。从原相机视角渲染，不可旋转。"})
    limitations = [("Metres come from the capture device's poses; depth comes from a pretrained model conditioned on them." if device else
                    "Monocular video without a scale anchor: sizes and distances are in native units, not metres." if uncalibrated else
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
    request = lambda name: str(uuid5(NAMESPACE_URL, f"video-import:{name}:{args.object_map}:{args.depth_run}:{args.request_suffix}"))
    if args.republish:  # a new version of an existing report: the same workcell entry, earlier versions stay in its history
        record = json.loads(args.republish.read_text())
        capability, created = record["capability"], repository.get_project(record["projectId"])
    else:
        capability = "pcap_v1_" + secrets.token_urlsafe(32)
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
    parser.add_argument("--exclude-frames", nargs="*", metavar="START:END", help="cut-away shots (source frames START..END-1): no cameras, outlines or surfaces from them")
    parser.add_argument("--dense-points", type=Path, help="dense display point map (dense-points.glb + points.json with cell_native): replaces the room's supported-points cloud")
    parser.add_argument("--splats", type=Path, help="a splat_train.py .splat file (its .json beside it): the photo-real appearance layer")
    parser.add_argument("--full-video", type=Path, help="prepare_video_clip.py --full-video source-full.json: the uncropped frames, shown in place of --video")
    parser.add_argument("--analysis", type=Path, help="project_entities_to_frames analysis.json of the same object map")
    parser.add_argument("--inferred-floor", type=Path, help="infer_room_floor.py output: becomes the floor entity's model in the model layer")
    parser.add_argument("--shell-glb", type=Path, help="photo-textured copy of the depth run's fused mesh to show as the room surface")
    parser.add_argument("--comparison-video", type=Path, help="render_static_dynamic_video.py output (with its .json): the static/dynamic split as a report view")
    parser.add_argument("--republish", type=Path, help="an earlier .platform/imports/video-import-*.json: publish this import as that report's next version instead of a new report")
    parser.add_argument("--dynamic-scene", type=Path, help="scene.json written by mono_room.py dynamic: moving objects as timed surfaces in the 3D views")
    parser.add_argument("--video-events", type=Path, help="video_events.py events.json: window captions and events as the report's video memory")
    parser.add_argument("--skeleton-scene", type=Path, help="a replay scene.json whose moving objects carry keypoints3d/bones (same cameras): skeletons as a timed layer")
    parser.add_argument("--dynamic-analysis", type=Path, help="motion_tracks_to_analysis.py output of the same tracks: their masks become pickable outlines in the video")
    parser.add_argument("--title", default="Video workcell (imported, not accepted)")
    parser.add_argument("--output-dir", type=Path, default=Path(".platform/imports"))
    parser.add_argument("--request-suffix", default="1", help="change to import the same inputs again as a new project")
    run(parser.parse_args())
