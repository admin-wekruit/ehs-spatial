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
sys.path[:0] = [str(ROOT), str(ROOT / "modal_apps")]
from ehs_spatial.platform.contracts import digest, empty_document, validate_document  # noqa: E402

FRAME, CONFIRMED = "droid_final_native_world", 3


def rectify(image, calibration):
    """The exact raster every depth, mask and camera of this pipeline lives on: the shared DROID preprocessing at scale 2."""
    from droid_room import prepare_image
    return prepare_image(image, {"source_K_fx_fy_cx_cy": calibration["K"], "source_distortion": calibration["D"]}, 2)


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


def png(image):
    return cv2.imencode(".png", image)[1].tobytes()


def build_document(args, put_asset, calibration, dataset):
    ident = lambda *parts: str(uuid5(NAMESPACE_URL, "video-import:" + ":".join(map(str, parts))))
    object_map = json.loads((args.object_map / "object-map.json").read_text())
    scale = json.loads((args.depth_run / "metric-scale.json").read_text())
    manifest = json.loads((args.droid_run / "input-manifest.json").read_text())
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
                  "anchor": {"kind": "stated_carry_height", "metres": 1.6, "measured": False,
                             "modelEstimateMetresPerNative": scale.get("model_estimated_metres_per_native_unit"),
                             "sourcesDisagreeOver10pct": scale.get("scale_sources_disagree_over_10pct")}},
        "ground": {"plane": [*up.tolist(), float(-up @ origin)], "normal": up.tolist(), "offset": float(-up @ origin),
                   "sourceRefs": [{"assetId": source}], "source": "video_floor_consensus_plane"}}]

    used = sorted({int(o.split(":")[1]) for e in object_map["entities"] for o in e["observations"]})
    images = {}
    for index in used:
        record = manifest["frames"][index]
        rectified, k = rectify(cv2.imread(str(dataset / record["relative_path"])), calibration)
        images[index] = include(png(rectified), "image/png", {"kind": "source_image", "width": 640, "height": 480, "sourceFrame": index,
            "videoTimestamp": record["timestamp_text"], "sourceSha256": record["sha256"],
            "pixelMapping": [{"source": "original_pixels", "target": "canonical_pixels", "coordinateConvention": "pixel_centers", "matrix": np.eye(3).tolist()}]})
        document["cameras"].append({"id": ident("camera", index), "imageId": images[index], "coordinateFrameId": FRAME, "width": 640, "height": 480,
            "K": [[float(k[0]), 0., float(k[2])], [0., float(k[1]), float(k[3])], [0., 0., 1.]], "cameraToWorld": np.asarray(keyframes[index], float).tolist(),
            "sourceRefs": [{"assetId": source, "sourceCameraId": f"droid-keyframe-{index}"}]})

    import trimesh
    from ehs_spatial.platform.reconstruction import _plan_projection
    from ehs_spatial.platform.spatial import primitive_mesh
    shell = trimesh.load(args.depth_run / "predicted-scene.glb", force="mesh", process=False)
    rows = np.hstack([shell.vertices, shell.vertex_normals, np.asarray(shell.visual.vertex_colors)[:, :3] / 255.]).astype("<f4")
    indices = np.asarray(shell.faces, "<u4").ravel()
    layout = {"stride": 9, "byteOffset": 0, "vertexCount": len(rows), "indexByteOffset": rows.nbytes, "indexCount": len(indices), "indexType": "uint32"}
    shell_asset = include(rows.tobytes() + indices.tobytes(), "application/octet-stream", {"kind": "geometry", "format": "panoptes-mesh-v1", "byteLayout": layout,
                          "sourceRecordId": "fused-room-surface"})
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
                "labelEvidence": [{"label": label, "source": "sam3_text_prompt"}], "geometrySupport": None, "sourceRefs": [{"assetId": source, "sourceRecordId": oid}]})
            refs.append((oid, observation))
        measurements = {}
        for name, fact in facts.get(entity["entityId"], {}).items():  # the policy run's own metric facts, re-pointed at this document's observations
            known = dict(refs)
            measurements[name] = {**fact, "sourceRefs": [known[o] for o in fact["sourceRefs"]]}
        representations = []
        for seen in entity.get("surfaces", []):  # as reconstruction.py writes them: one observed surface per observation, measured in place
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
            projection = _plan_projection(document, surface, part, hashlib.sha256(payload).hexdigest())  # the photo report's own CAD outline: exact triangle union on the floor
            if projection is not None:
                surface["planProjection"] = projection
            representations.append(surface)
        model_transform = None
        fitted = extent_box(entity, object_map["plan"]) if len(refs) >= CONFIRMED and object_map.get("plan") else None
        if fitted:  # same fields repository.py writes for a primitive: a proposal until a person confirms the alignment
            primitive, model_transform = fitted
            local = primitive_mesh(primitive)
            model = {"id": ident("representation", "box", entity["entityId"]), "kind": "primitive", "assetId": None, "primitive": primitive, "transform": model_transform,
                     "coordinateFrameId": FRAME, "bounds": {"min": local.vertices.min(0).tolist(), "max": local.vertices.max(0).tolist()},
                     "placementState": "unconfirmed", "placementReason": "requires_alignment_confirmation", "sourceRefs": [{"observationId": o} for _, o in refs],
                     "modelBasis": "floor-aligned box over the observed extent of a multi-view confirmed entity; unseen sides are not measured"}
            projection = _plan_projection(document, model, local, None, model_transform)
            if projection is not None:
                model["planProjection"] = projection
            representations.append(model)
        document["entities"].append({"id": ident("entity", entity["entityId"]), "label": entity["label"], "observationRefs": [o for _, o in refs],
            "associationState": "confirmed" if len(refs) >= CONFIRMED else "association_pending", "representations": representations, "currentModelTransform": model_transform,
            "measurements": measurements, "groupId": None, "visible": True, "sourceContext": False,
            "lineage": [{"operation": "offline_import", "sourceAssetId": source, "sourceRecordId": entity["entityId"], "method": object_map["associator"]}]})
    limitations = ["Monocular video; metres come from a stated 1.6 m carry height that disagrees with the model scale estimate by about 20% on this clip.",
                   "Only what the camera saw is present; the room surface is source context and has not been accepted.",
                   "Entities are grouped by 3D point overlap of text-prompted masks; identities were not reviewed by a person."] + object_map["limitations"]
    document["annotations"].append({"id": ident("annotation", "provenance"), "kind": "import_provenance", "sourceAssetId": source, "limitations": limitations,
        "missingArtifacts": [], "recomputeRequiresNewCapture": True, "pipeline": {"cameras": str(args.droid_run), "depth": str(args.depth_run), "objects": str(args.object_map)}})
    return document, hashlib.sha256(source_bytes).hexdigest()


def run(args):
    from droid_room import CLIPS
    from ehs_spatial.platform.config import PlatformConfig
    from ehs_spatial.platform.runtime import services
    clip = CLIPS[json.loads((args.droid_run / "run.json").read_text()).get("clip", "fr1-room")]
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
    for name in ("droid-run", "depth-run", "object-map", "masks", "policy"):
        parser.add_argument("--" + name, type=Path, required=name != "policy")
    parser.add_argument("--title", default="Video workcell (imported, not accepted)")
    parser.add_argument("--output-dir", type=Path, default=Path(".platform/imports"))
    parser.add_argument("--request-suffix", default="1", help="change to import the same inputs again as a new project")
    run(parser.parse_args())
