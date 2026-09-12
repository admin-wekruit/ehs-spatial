"""Offline, one-way import of a version-1 public scene into canonical PostgreSQL.

No legacy schema enters the product runtime. Separate input scenes remain separate
coordinate frames/projects; labels never establish cross-view physical identity.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import secrets
from uuid import NAMESPACE_URL, UUID, uuid5

import numpy as np
from PIL import Image

from ehs_spatial.platform.contracts import PlatformError, canonical, digest, empty_document, validate_document
from ehs_spatial.platform.spatial import MeshData, camera_intrinsics, transform_matrix

CONVERTER_VERSION = "public-scene-v3"


def converter_identity():
    return {"version": CONVERTER_VERSION, "reportRendererVersion": "native-webgl-v1", "codeSha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def evidence_documents(scene_path, source):
    root = Path(scene_path).parent
    paths = set()
    if source.get("object_evidence"):
        paths.add(source_path(root, source["object_evidence"]))
    for path in root.glob("*.json"):
        if path.resolve() == Path(scene_path).resolve():
            continue
        try:
            data = json.loads(path.read_bytes())
        except (ValueError, OSError):
            continue
        if isinstance(data, dict) and data.get("scene_url") == Path(scene_path).name and data.get("reconstruction_run_id") == source.get("run_id"):
            paths.add(path.resolve())
    return [(path, path.read_bytes()) for path in sorted(paths)]


def source_path(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise PlatformError("import_asset_path_invalid", 422)
    return path


def packed_asset(root, record):
    path = source_path(root, record["path"])
    packed = path.read_bytes()
    if "packed_bytes" in record and len(packed) != record["packed_bytes"]:
        raise PlatformError("import_packed_size_mismatch", 422)
    try:
        raw = gzip.decompress(packed) if path.suffix == ".gz" else packed
    except (OSError, EOFError):
        raise PlatformError("import_compression_invalid", 422) from None
    if "bytes" in record and len(raw) != record["bytes"]:
        raise PlatformError("import_asset_size_mismatch", 422)
    if "sha256" in record and hashlib.sha256(raw).hexdigest() != record["sha256"]:
        raise PlatformError("import_asset_hash_mismatch", 422)
    return raw


def legacy_transform(value, frame_id):
    """Exact public viewer convention: T Rz Ry Rx S, quaternion XYZW."""
    if not isinstance(value, dict):
        raise PlatformError("import_transform_missing", 422)
    try:
        x, y, z = [math.radians(float(angle)) / 2 for angle in value["rotation_deg"]]
        sx, sy, sz, cx, cy, cz = math.sin(x), math.sin(y), math.sin(z), math.cos(x), math.cos(y), math.cos(z)
        quaternion = [sx*cy*cz-cx*sy*sz, cx*sy*cz+sx*cy*sz, cx*cy*sz-sx*sy*cz, cx*cy*cz+sx*sy*sz]
        result = {"coordinateFrameId": frame_id, "position": list(value["position"]), "quaternion": quaternion, "scale": list(value["scale"])}
        transform_matrix(result)
    except (KeyError, ValueError, TypeError):
        raise PlatformError("import_transform_invalid", 422) from None
    return result


def unpack_mesh(root, record):
    raw = packed_asset(root, record["asset"])
    count, index_count = record["vertex_count"], record["index_count"]
    offset, index_offset = record["byte_offset"], record["index_byte_offset"]
    if record.get("stride") != 9 or record.get("index_type") != "uint32" or any(type(value) is not int or value < 0 for value in (count, index_count, offset, index_offset)) or index_count % 3 or offset % 4 or index_offset % 4 or offset+count*36 > index_offset or index_offset+index_count*4 > len(raw):
        raise PlatformError("import_mesh_layout_invalid", 422)
    vertices = np.frombuffer(raw, dtype="<f4", count=count*9, offset=offset).reshape(-1, 9).copy()
    faces = np.frombuffer(raw, dtype="<u4", count=index_count, offset=index_offset).reshape(-1, 3).copy()
    MeshData(vertices[:, :3], faces, vertices[:, 6:9])
    if not np.isfinite(vertices).all():
        raise PlatformError("import_mesh_nonfinite", 422)
    return vertices, faces


def import_document(scene_path, put_asset):
    """Pure conversion apart from injected verified asset registration."""
    scene_path = Path(scene_path).resolve()
    root = scene_path.parent
    source_bytes = scene_path.read_bytes()
    source_sha = hashlib.sha256(source_bytes).hexdigest()
    source = json.loads(source_bytes)
    if source.get("version") != 1 or not isinstance(source.get("cameras"), list) or not source["cameras"]:
        raise PlatformError("import_scene_invalid", 422)
    namespace = uuid5(NAMESPACE_URL, "panoptes-public:" + source_sha)
    ident = lambda kind, value: str(uuid5(namespace, kind + ":" + str(value)))
    frame_id = ident("frame", "native-opencv")
    document = empty_document()
    document["target"] = source.get("target", "scene")
    manifest = {"schemaVersion": 1, "converter": converter_identity(), "sourcePath": str(scene_path), "sourceSha256": source_sha, "runId": source.get("run_id"),
                "cameraIds": {}, "entityIds": {}, "observationIds": {}, "assetIds": {}, "representationAssetIds": {}, "limitations": source.get("limitations", [])}

    def include(data, media_type, metadata, source_key=None):
        asset = put_asset(data, media_type, metadata)
        ref = {**asset, **metadata}
        if asset["id"] not in {a["id"] for a in document["assets"]}:
            document["assets"].append(ref)
        if source_key:
            manifest["assetIds"][source_key] = asset["id"]
        return asset["id"]

    source_asset = include(source_bytes, "application/json", {"kind": "import_source", "sourceSha256": source_sha}, scene_path.name)
    ground = None
    if source.get("floor_plane"):
        plane = np.asarray(source["floor_plane"], dtype=float)
        if plane.shape != (4,) or not np.isfinite(plane).all() or np.linalg.norm(plane[:3]) <= 1e-12:
            raise PlatformError("import_ground_invalid", 422)
        plane = plane / np.linalg.norm(plane[:3])
        ground = {"plane": plane.tolist(), "normal": plane[:3].tolist(), "offset": float(plane[3]), "sourceRefs": [{"assetId": source_asset}], "source": "imported_native_floor"}
    document["coordinateFrames"] = [{"id": frame_id, "convention": "opencv", "scale": {"status": "uncalibrated", "nativeToMeters": None, "sourceRefs": [{"assetId": source_asset}]}, "ground": ground}]
    cameras, camera_images, camera_hashes = {}, {}, {}
    for camera in source["cameras"]:
        use_original = all(key in camera for key in ("original_image", "original_width", "original_height", "original_K"))
        image_path = camera["original_image"] if use_original else camera["image"]
        data = source_path(root, image_path).read_bytes()
        expected = camera.get("original_sha256") if use_original else camera.get("sha256")
        if expected and hashlib.sha256(data).hexdigest() != expected:
            raise PlatformError("import_image_hash_mismatch", 422)
        with Image.open(io.BytesIO(data)) as image:
            width, height = image.size
            media_type = Image.MIME.get(image.format, "application/octet-stream")
        expected_size = (camera["original_width"], camera["original_height"]) if use_original else (camera["width"], camera["height"])
        if (width, height) != expected_size:
            raise PlatformError("import_image_dimensions_mismatch", 422)
        k = camera["original_K"] if use_original else camera["K"]
        mapping = camera_intrinsics(camera["K"]) @ np.linalg.inv(camera_intrinsics(k))
        image_id = include(data, media_type, {"kind": "source_image", "width": width, "height": height, "pixelMapping": [{"source": "original_pixels", "target": "canonical_pixels", "coordinateConvention": "pixel_centers", "matrix": mapping.tolist()}]}, image_path)
        camera_id = ident("camera", camera["id"])
        document["cameras"].append({"id": camera_id, "imageId": image_id, "coordinateFrameId": frame_id, "width": width, "height": height,
            "K": deepcopy(k), "cameraToWorld": deepcopy(camera["camera_to_world"]), "sourceRefs": [{"assetId": source_asset, "sourceCameraId": camera["id"]}]})
        manifest["cameraIds"][camera["id"]] = camera_id
        cameras[camera["id"]] = {**camera, "chosenWidth": width, "chosenHeight": height, "canonicalToOriginal": np.linalg.inv(mapping)}
        camera_images[camera["id"]] = image_id
        camera_hashes[camera["id"]] = hashlib.sha256(data).hexdigest()

    linked_views, candidates = {}, {}
    for path, raw in evidence_documents(scene_path, source):
        evidence_id = include(raw, "application/json", {"kind": "import_evidence"}, path.name)
        evidence = json.loads(raw)
        for record in evidence.get("objects", []):
            if record.get("scene_object_id", record.get("id")) == record.get("id"):
                linked_views[record["id"]] = (record.get("views", []), evidence_id)
        for record in evidence.get("candidates", []):
            candidates[record["id"]] = (record, evidence_id)

    meshes = {}
    records = {}
    for collection in ("objects", "observed_regions", "unavailable_regions", "unavailable_objects"):
        for record in source.get(collection, []):
            if not isinstance(record, dict) or not isinstance(record.get("id"), str):
                raise PlatformError("import_identity_invalid", 422)
            if record["id"] not in records:
                records[record["id"]] = record
    for record in source.get("objects", []):
        if record.get("mesh"):
            meshes[record["id"]] = unpack_mesh(root, record["mesh"])

    def save_mesh(vertices, faces, source_record):
        indices = np.asarray(faces, dtype="<u4").ravel()
        vertices = np.asarray(vertices, dtype="<f4")
        layout = {"stride": 9, "byteOffset": 0, "vertexCount": len(vertices), "indexByteOffset": vertices.nbytes, "indexCount": len(indices), "indexType": "uint32"}
        return include(vertices.tobytes()+indices.tobytes(), "application/octet-stream", {"kind": "geometry", "format": "panoptes-mesh-v1", "byteLayout": layout, "sourceRecordId": source_record})

    for old_id, record in records.items():
        entity_id = ident("entity", old_id)
        manifest["entityIds"][old_id] = entity_id
        item = {"id": entity_id, "label": record.get("label", old_id), "observationRefs": [], "associationState": "association_pending",
                "representations": [], "currentModelTransform": None, "measurements": {}, "groupId": None,
                "lineage": [{"operation": "offline_import", "sourceAssetId": source_asset, "sourceRecordId": old_id}],
                "visible": record.get("visible", True), "sourceContext": record.get("role") == "context"}
        measurements = record.get("measurements", {})
        dimensions = measurements.get("dimensions_native")
        if measurements.get("status") == "available" and isinstance(dimensions, dict) and all(type(dimensions.get(k)) in (int, float) and math.isfinite(dimensions[k]) and dimensions[k] >= 0 for k in ("height", "width", "depth")):
            item["measurements"] = {"dimensionsNative": deepcopy(dimensions), "unit": "native", "coverage": measurements.get("coverage"),
                "source": "imported_observed_measurement", "coordinateFrameId": frame_id, "uncertaintyNative": None,
                "basis": deepcopy(measurements.get("basis")), "sourceRefs": [{"assetId": source_asset, "sourceRecordId": old_id}],
                "scaleEvidence": deepcopy(measurements.get("scale")), "orientationEvidence": deepcopy(measurements.get("orientation"))}
        frame_ref = record.get("reference_frame")
        mask = record.get("mask")
        source_box = record.get("source_bbox", {})
        if frame_ref in cameras and (mask or source_box.get("bbox_xyxy")):
            camera = cameras[frame_ref]
            mask_id = None
            if mask:
                mask_data = packed_asset(root, mask)
                mask_id = include(mask_data, "image/png", {"kind": "source_mask", "resolution": mask.get("resolution"), "sourceRecordId": old_id}, mask["path"])
            original_shape = [camera["chosenHeight"], camera["chosenWidth"]]
            if source_box.get("resolution") == "original" and source_box.get("shape_hw") == original_shape:
                box = source_box["bbox_xyxy"]
            else:
                canonical_box = mask.get("bbox_xyxy") if mask else source_box.get("bbox_xyxy")
                if not canonical_box:
                    raise PlatformError("import_observation_mapping_missing", 422)
                corners = np.array([[canonical_box[0], canonical_box[1], 1], [canonical_box[2], canonical_box[3], 1]]) @ camera["canonicalToOriginal"].T
                box = [max(0., float(corners[0, 0])), max(0., float(corners[0, 1])), min(float(camera["chosenWidth"]), float(corners[1, 0])), min(float(camera["chosenHeight"]), float(corners[1, 1]))]
            observation_id = ident("observation", old_id + ":" + frame_ref)
            observation = {"id": observation_id, "revision": 1, "imageId": camera_images[frame_ref], "originalPixelBox": list(box), "maskAssetId": mask_id,
                "pixelMapping": [{"source": "canonical_pixels", "target": "original_pixels", "coordinateConvention": "pixel_centers", "matrix": camera["canonicalToOriginal"].tolist()}],
                "labelEvidence": [{"label": item["label"], "source": "legacy_import"}], "geometrySupport": None,
                "sourceRefs": [{"assetId": source_asset, "sourceRecordId": old_id}], "sourceBoxEvidence": deepcopy(source_box)}
            document["observations"].append(observation)
            item["observationRefs"].append(observation_id)
            manifest["observationIds"][old_id] = observation_id
        if not item["observationRefs"] and record.get("source") in ("generated", "parametric"):
            binding = measurements.get("source", {})
            anchor = binding.get("frame_id")
            match = None
            if binding.get("candidate_id") == old_id and anchor in camera_hashes and binding.get("image_sha256") == camera_hashes[anchor]:
                if old_id in candidates:
                    candidate, evidence_id = candidates[old_id]
                    if candidate.get("frame_id") == anchor and candidate.get("mask", {}).get("sha256") == binding.get("mask_sha256") and candidate.get("mask", {}).get("bbox"):
                        match = {"bbox": candidate["mask"]["bbox"], "resolution": candidate["mask"].get("resolution"), "shape": candidate["mask"].get("shape_hw"), "assetId": evidence_id, "maskMissing": True}
                elif old_id in linked_views:
                    views, evidence_id = linked_views[old_id]
                    view = next((view for view in views if view.get("frame_id") == anchor and view.get("bbox")), None)
                    if view:
                        match = {"bbox": view["bbox"], "resolution": "canonical", "assetId": evidence_id, "polygons": view.get("polygons", []), "maskMissing": True}
            if match:
                camera = cameras[anchor]
                box = match["bbox"]
                if match["resolution"] == "canonical":
                    p = np.array([[box[0], box[1], 1], [box[2], box[3], 1]]) @ camera["canonicalToOriginal"].T
                    box = [max(0., float(p[0, 0])), max(0., float(p[0, 1])), min(float(camera["chosenWidth"]), float(p[1, 0])), min(float(camera["chosenHeight"]), float(p[1, 1]))]
                elif match.get("shape") != [camera["chosenHeight"], camera["chosenWidth"]]:
                    raise PlatformError("import_linked_mask_dimensions_mismatch", 422)
                oid = ident("observation", old_id + ":" + anchor)
                document["observations"].append({"id": oid, "revision": 1, "imageId": camera_images[anchor], "originalPixelBox": list(box), "maskAssetId": None,
                    "pixelMapping": [{"source": "canonical_pixels", "target": "original_pixels", "matrix": camera["canonicalToOriginal"].tolist(), "coordinateConvention": "pixel_centers"}],
                    "labelEvidence": [{"label": item["label"], "source": "legacy_import"}], "geometrySupport": None,
                    "sourceRefs": [{"assetId": match["assetId"], "sourceRecordId": old_id, "binding": "exact_candidate_id_and_image_sha256", "imageSha256": camera_hashes[anchor]}],
                    "missingEvidence": ["source_mask_not_packaged"], "sourcePolygonsCanonical": match.get("polygons", [])})
                item["observationRefs"].append(oid)
                manifest["observationIds"][old_id] = oid
            else:
                item["missingEvidence"] = ["source_observation_binding_pending"]
        geometry = meshes.get(old_id)
        transform = legacy_transform(record["transform"], frame_id) if record.get("transform") else None
        if record.get("faces"):
            context_id = record.get("context_id")
            if context_id not in meshes or frame_ref not in records[context_id].get("frame_ids", []):
                raise PlatformError("import_region_context_mismatch", 422)
            context_transform = legacy_transform(records[context_id]["transform"], frame_id)
            if not np.allclose(transform_matrix(context_transform), np.eye(4), atol=1e-12):
                raise PlatformError("import_region_context_not_native", 422)
            face_record = record["faces"]
            raw = packed_asset(root, face_record["asset"])
            face_ids = np.frombuffer(raw, dtype="<u4")
            vertices, faces = meshes[context_id]
            if len(face_ids) != face_record["count"] or len(np.unique(face_ids)) != len(face_ids) or (len(face_ids) and face_ids.max() >= len(faces)):
                raise PlatformError("import_region_faces_invalid", 422)
            selected = faces[face_ids]
            vertex_ids, remapped = np.unique(selected, return_inverse=True)
            geometry = (vertices[vertex_ids], remapped.reshape(-1, 3).astype("<u4"))
            transform = context_transform
        if geometry is not None:
            if transform is None:
                raise PlatformError("import_transform_missing", 422)
            asset_id = save_mesh(*geometry, old_id)
            manifest["representationAssetIds"][old_id] = asset_id
            kind = "observed_surface" if record.get("source") == "observed" else "generated_mesh"
            item["representations"].append({"id": ident("representation", old_id), "kind": kind, "assetId": asset_id,
                "coordinateFrameId": frame_id, "transform": transform, "primitive": None, "placementState": "confirmed" if kind == "observed_surface" else "unconfirmed",
                "placementReason": "imported_observed_surface" if kind == "observed_surface" else "imported_proposal",
                "bounds": {"min": geometry[0][:, :3].min(axis=0).tolist(), "max": geometry[0][:, :3].max(axis=0).tolist()},
                "sourceRefs": [{"assetId": source_asset, "sourceRecordId": old_id}]})
            if kind == "generated_mesh":
                item["currentModelTransform"] = deepcopy(transform)
        document["entities"].append(item)
    document["annotations"].append({"id": ident("annotation", "provenance"), "kind": "import_provenance", "sourceAssetId": source_asset,
        "sourceSha256": source_sha, "converter": converter_identity(), "sourceRunId": source.get("run_id"), "sourceUnits": source.get("units"), "limitations": source.get("limitations", []),
        "identityPolicy": "Existing source IDs retained; labels and cross-view observations never merged", "measurementsPolicy": "Observed native extents retain source provenance; metric scale and physical PCA-axis meanings are not promoted",
        "missingArtifacts": ["native_pointmaps", "native_depth", "native_confidence"], "recomputeRequiresNewCapture": True})
    validate_document(document)
    manifest.update(entityCount=len(document["entities"]), observationCount=len(document["observations"]),
                    representationCount=sum(len(e["representations"]) for e in document["entities"]), assetCount=len(document["assets"]), documentSha256=digest(document))
    return document, manifest


def run_import(scene_path, repository, blobs, output_dir, title=None):
    scene_path = Path(scene_path).resolve()
    source = json.loads(scene_path.read_bytes())
    sha = hashlib.sha256(scene_path.read_bytes()).hexdigest()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    management = output_dir / (sha + ".management.json")
    if management.exists():
        if management.stat().st_mode & 0o077:
            raise PlatformError("import_management_permissions_invalid", 422)
        pending = json.loads(management.read_text())
    else:
        pending = {"requestId": str(uuid5(NAMESPACE_URL, "import-create:" + sha)), "capability": "pcap_v1_" + secrets.token_urlsafe(32), "sourceSha256": sha}
        fd = os.open(management, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as stream:
            json.dump(pending, stream)
    repository.blobs = blobs
    body = {"requestId": pending["requestId"], "title": title or source.get("label") or scene_path.stem, "target": "scene"}
    created = repository.create_project(pending["capability"], body)
    project_id = created["project"]["id"]
    converter = converter_identity()
    converter_key = digest({"converter": converter, "evidence": [hashlib.sha256(raw).hexdigest() for _, raw in evidence_documents(scene_path, source)]})
    manifest_path = output_dir / (sha + "." + converter_key + ".manifest.json")
    if manifest_path.exists():
        return json.loads(manifest_path.read_text())

    def put_asset(data, media_type, metadata):
        asset = blobs.put(data, media_type)
        asset["metadata"] = metadata
        return repository.register_asset(project_id, asset)

    document, manifest = import_document(scene_path, put_asset)
    request_id = str(uuid5(NAMESPACE_URL, "import-job:" + sha + ":" + converter_key))
    jobs = repository.list_project_records(project_id, "jobs")["items"]
    prior_job = next((job for job in jobs if job["requestId"] == request_id), None)
    prior_imports = [job for job in jobs if job["kind"] == "import_scene" and job.get("resultRevisionId")]
    branch_id, base_id = created["branch"]["id"], created["revision"]["id"]
    if prior_job:
        branch_id, base_id = prior_job["branchId"], prior_job["baseRevisionId"]
    elif prior_imports and base_id not in {job["resultRevisionId"] for job in prior_imports}:
        branch = repository.create_branch(project_id, pending["capability"], {"requestId": str(uuid5(NAMESPACE_URL, "import-branch:"+sha+":"+converter_key)), "sourceRevisionId": prior_imports[0]["resultRevisionId"], "kind": "reconstruction", "title": "Imported source " + CONVERTER_VERSION})
        branch_id, base_id = branch["id"], branch["headRevisionId"]
    capture_id = str(uuid5(NAMESPACE_URL, "import-capture:"+sha+":"+converter_key))
    document["captureId"] = capture_id
    images = [{"id": camera["imageId"], "assetId": camera["imageId"], "width": camera["width"], "height": camera["height"], "sourceCameraId": camera["sourceRefs"][0]["sourceCameraId"]} for camera in document["cameras"]]
    imported_capture = {"id": capture_id, "images": images, "task": {"schemaVersion": 1, "kind": "offline_import", "sourceSha256": sha, "imageIds": [image["id"] for image in images], "missingArtifacts": ["native_pointmaps", "native_depth", "native_confidence"], "recomputeRequiresNewCapture": True}}
    job = repository.create_job(project_id, pending["capability"], {"requestId": request_id, "branchId": branch_id, "baseRevisionId": base_id, "kind": "import_scene", "inputs": {"sourceSha256": sha}, "config": {"offline": True, "converter": converter}})
    if job["status"] in ("pending_dispatch", "queued"):
        job = repository.claim_job(job["id"], lease_seconds=3600)
        job = repository.finish_job(job["id"], job["attemptToken"], "succeeded", document=document, result={"sourceSha256": sha, "newModelCalls": 0}, imported_capture=imported_capture)
    if job["status"] != "succeeded" or not job.get("resultRevisionId"):
        raise PlatformError("import_job_requires_review", 409)
    publication = repository.create_publication(project_id, pending["capability"], {"requestId": str(uuid5(NAMESPACE_URL, "import-publish:" + sha + ":" + converter_key)), "sceneRevisionId": job["resultRevisionId"], "evaluationIds": [], "reviewIds": [], "title": body["title"]})
    manifest.update(projectId=project_id, branchId=branch_id, captureId=capture_id, sceneRevisionId=job["resultRevisionId"], publicationId=publication["id"], jobId=job["id"], newModelCalls=0, documentSha256=digest(document))
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenes", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path(".platform/imports"))
    args = parser.parse_args()
    from ehs_spatial.platform.runtime import services
    from ehs_spatial.platform.config import PlatformConfig
    repository, blobs = services(PlatformConfig.from_env())
    repository.migrate()
    for scene in args.scenes:
        manifest = run_import(scene, repository, blobs, args.output_dir)
        print(json.dumps({key: manifest[key] for key in ("projectId", "publicationId", "sceneRevisionId", "entityCount", "observationCount", "assetCount", "newModelCalls")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
