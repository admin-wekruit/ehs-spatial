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
from ehs_spatial.platform.source_cad import refresh_source_cad_links
from scripts.import_report_evidence import canonical_measurements, canonical_observation_masks, import_report_evidence, import_observation_masks, import_source_equivalences, observation_mask_sources, original_box, original_polygons, report_dependencies

CONVERTER_VERSION = "public-scene-v9"


def converter_identity():
    return {"version": CONVERTER_VERSION, "reportRendererVersion": "native-webgl-v1", "codeSha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "reportConverterSha256": hashlib.sha256(Path(__file__).with_name("import_report_evidence.py").read_bytes()).hexdigest(),
            "sourceCadConverterSha256": hashlib.sha256(Path(__file__).parents[1].joinpath("ehs_spatial/platform/source_cad.py").read_bytes()).hexdigest(),
            "geometryConverterSha256": hashlib.sha256(Path(__file__).with_name("import_geometry_evidence.py").read_bytes()).hexdigest()}


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


def parametric_definition(root, record, geometry, frame_id):
    """Restore saved cylinder parameters only when they reproduce the source mesh."""
    from ehs_spatial.platform.spatial import primitive_mesh, matrix_to_transform, transform_points
    from shapely.geometry import Polygon, box
    from shapely.ops import unary_union
    relative = (record.get('metrics') or {}).get('parameter_source')
    if not isinstance(relative, str):
        raise PlatformError('import_parameter_source_missing', 422)
    raw = source_path(root, relative).read_bytes()
    try:
        source = json.loads(raw)
    except ValueError:
        raise PlatformError('import_parameter_contract_invalid', 422) from None
    if not isinstance(source, dict) or not isinstance(source.get('objects'), list) or not all(isinstance(row, dict) for row in source['objects']):
        raise PlatformError('import_parameter_contract_invalid', 422)
    rows = [row for row in source['objects'] if row.get('object_id') == record['id']]
    template = source.get('primitive', {})
    if len(rows) != 1 or not isinstance(template, dict) or template.get('type') != 'cylinder' or template.get('local_axis') != '+Z' or template.get('radius') != 1 or template.get('height') != 1:
        raise PlatformError('import_parameter_contract_invalid', 422)
    row = rows[0]
    if not isinstance(row.get('fitted_parameters'), dict) or row['fitted_parameters'] != record.get('parameters'):
        raise PlatformError('import_parameter_values_mismatch', 422)
    params = row['fitted_parameters']
    primitive = {'kind': 'cylinder', 'radius': params.get('radius_native'), 'height': params.get('height_native'), 'segments': template.get('segments')}
    mesh = primitive_mesh(primitive)
    try:
        matrix = np.asarray(row.get('native_object_to_world'), dtype=float)
    except (TypeError, ValueError):
        raise PlatformError('import_parameter_pose_invalid', 422) from None
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise PlatformError('import_parameter_pose_invalid', 422)
    matrix = matrix @ np.diag([1 / primitive['radius'], 1 / primitive['radius'], 1 / primitive['height'], 1])
    pose = matrix_to_transform(matrix, frame_id)
    if not np.allclose(pose['scale'], [1, 1, 1], rtol=0, atol=1e-6):
        raise PlatformError('import_parameter_pose_invalid', 422)
    pose['scale'] = [1., 1., 1.]
    expected = transform_points(geometry[0][:, :3], transform_matrix(legacy_transform(record['transform'], frame_id)))
    actual = transform_points(mesh.vertices, transform_matrix(pose))
    if len(mesh.faces) != len(geometry[1]) or not 0 < len(expected) <= 3 * len(mesh.faces):
        raise PlatformError('import_parameter_mesh_mismatch', 422)
    # ponytail: cylinder-only proof is at most 3072 by 514 vertices; general
    # parametric assets need a spatial index before increasing this bound.
    errors = np.linalg.norm(expected[:, None, :] - actual[None, :, :], axis=2)
    error = float(max(errors.min(axis=0).max(), errors.min(axis=1).max()))
    tolerance = max(1e-7, float(np.max(np.abs(expected))) * float(np.finfo(np.float32).eps) * 16)
    if error > tolerance:
        raise PlatformError('import_parameter_mesh_mismatch', 422, maxVertexError=error, tolerance=tolerance)
    # Equal vertices do not prove equal surfaces. Compare each boundary facet's
    # coverage; either diagonal of a side quad represents the same surface.
    n = primitive['segments']
    facets = [[] for _ in range(n + 2)]
    for face in errors.argmin(axis=1)[geometry[1]]:
        if np.all((face < n) | (face == 2*n)):
            facet, points = n, mesh.vertices[face, :2] / primitive['radius']
        elif np.all(((face >= n) & (face < 2*n)) | (face == 2*n+1)):
            facet, points = n+1, mesh.vertices[face, :2] / primitive['radius']
        else:
            ring = sorted(set((face % n).tolist()))
            if np.any(face >= 2*n) or len(ring) != 2 or (ring[1]-ring[0]) not in (1, n-1):
                raise PlatformError('import_parameter_surface_mismatch', 422)
            facet = ring[0] if ring[1]-ring[0] == 1 else ring[1]
            points = np.column_stack((face % n != facet, face >= n)).astype(float)
        triangle = Polygon(points)
        if not triangle.is_valid or triangle.area == 0:
            raise PlatformError('import_parameter_surface_mismatch', 422)
        facets[facet].append(triangle)
    for index, triangles in enumerate(facets):
        boundary = Polygon(mesh.vertices[:n, :2] / primitive['radius']) if index >= n else box(0, 0, 1, 1)
        covered = unary_union(triangles)
        if covered.symmetric_difference(boundary).area > 1e-10 or abs(sum(triangle.area for triangle in triangles) - boundary.area) > 1e-10:
            raise PlatformError('import_parameter_surface_mismatch', 422)
    proof = {'method': 'saved-cylinder-surface-equivalence-v1', 'maxVertexErrorNative': error, 'toleranceNative': tolerance,
             'surfaceFacetCount': len(facets), 'surfaceCoverage': 'equivalent',
             'parameterSourceSha256': hashlib.sha256(raw).hexdigest(), 'parameterSource': source.get('parameter_source'),
             'limitations': source.get('limitations', [])}
    return primitive, pose, raw, proof


def floor_evidence(geometry_root, source):
    """Hash-pinned floor samples in the source scene's native camera frame."""
    if not geometry_root or not source.get("floor_reference"):
        return None
    from scripts.import_geometry_evidence import pinned_manifest
    root = Path(geometry_root).resolve()
    frozen, manifest_raw = pinned_manifest(root, source)
    reference = source["floor_reference"]
    raw = source_path(root, reference["path"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != reference.get("sha256") or frozen.get("evidence", {}).get("floor_sha256") != reference["sha256"]:
        raise PlatformError("import_floor_evidence_hash_mismatch", 422)
    floor = json.loads(raw)
    view = floor["views"][-1]
    fid = view["frame_id"]
    specs = [record for record in frozen["geometry"]["frames"] if record["frame_id"] == fid]
    cameras = [camera for camera in source["cameras"] if camera["id"] == fid]
    if len(specs) != 1 or len(cameras) != 1:
        raise PlatformError("import_floor_frame_mismatch", 422)
    files = {"manifest.json": manifest_raw, reference["path"]: raw}
    arrays = {}
    for key, name in (("pointmap_path", "pts3d.npy"), ("valid_path", "valid_mask.npy"),
                      ("content_valid_path", "content_valid_mask.npy"), ("conf_path", "conf.npy"),
                      ("K_path", "intrinsics.npy"), ("c2w_path", "camera_to_world.npy"),
                      ("canonical_mask_path", None), ("points_path", None)):
        relative = view[key]
        expected = view["sha256"].get(Path(relative).name) if name is None else specs[0]["files"].get(name)
        if name is not None and relative != f"geometry/frames/{fid}/{name}":
            raise PlatformError("import_floor_frame_mismatch", 422)
        data = source_path(root, relative).read_bytes()
        if not expected or hashlib.sha256(data).hexdigest() != expected:
            raise PlatformError("import_floor_evidence_hash_mismatch", 422)
        files[relative] = data
        arrays[key] = np.load(io.BytesIO(data), allow_pickle=False)
    if not np.array_equal(arrays["K_path"], cameras[0]["K"]) or not np.array_equal(arrays["c2w_path"], cameras[0]["camera_to_world"]):
        raise PlatformError("import_floor_camera_mismatch", 422)
    return {"view": view, "files": files, "arrays": arrays}


def floor_mesh_members(evidence, records, meshes, frame_id):
    """Prove exact indexed geometry membership, independent of object IDs/names."""
    arrays = evidence["arrays"]
    points, mask = arrays["pointmap_path"], arrays["canonical_mask_path"].astype(bool)
    if points.shape != (*mask.shape, 3) or any(arrays[key].shape != mask.shape for key in ("valid_path", "content_valid_path", "conf_path")):
        raise PlatformError("import_floor_grid_mismatch", 422)
    inverse = np.linalg.inv(arrays["c2w_path"])
    local = points @ inverse[:3, :3].T + inverse[:3, 3]
    conf = arrays["conf_path"]
    valid = mask & arrays["valid_path"].astype(bool) & arrays["content_valid_path"].astype(bool) & np.isfinite(local).all(-1) & (local[..., 2] > 0) & np.isfinite(conf) & (conf >= .1)
    if not np.array_equal(points[valid], arrays["points_path"]):
        raise PlatformError("import_floor_sample_mismatch", 422)
    grid = np.arange(mask.size).reshape(mask.shape)
    a, b, c, d = grid[:-1, :-1], grid[:-1, 1:], grid[1:, :-1], grid[1:, 1:]
    faces = np.concatenate([np.stack([a, b, c], -1).reshape(-1, 3), np.stack([b, d, c], -1).reshape(-1, 3)])
    faces = faces[valid.ravel()[faces].all(axis=1)]
    vertices, depth = points.reshape(-1, 3), local[..., 2].ravel()
    # Exact observed-mesh profile: adjacent source pixels and depth-relative
    # discontinuity rejection. No fitted plane or label can establish membership.
    lengths = np.linalg.norm(vertices[faces] - vertices[faces[:, [1, 2, 0]]], axis=2)
    faces = faces[lengths.max(axis=1) <= .04 * np.median(depth[faces], axis=1)]
    if not len(faces):
        raise PlatformError("import_floor_mesh_empty", 422)
    used, remapped = np.unique(faces, return_inverse=True)
    expected_vertices, expected_faces = vertices[used].astype("<f4"), remapped.reshape(-1, 3).astype("<u4")
    matches = []
    for old_id, (mesh_vertices, mesh_faces) in meshes.items():
        record = records[old_id]
        if record.get("source") != "observed" or record.get("frame_ids") != [evidence["view"]["frame_id"]]:
            continue
        transform = transform_matrix(legacy_transform(record["transform"], frame_id))
        native = (mesh_vertices[:, :3] @ transform[:3, :3].T + transform[:3, 3]).astype("<f4")
        if np.array_equal(native, expected_vertices) and np.array_equal(mesh_faces, expected_faces):
            matches.append(old_id)
    return matches, len(expected_vertices), len(expected_faces)


def import_document(scene_path, put_asset, *, legacy_root=None, observation_root=None, geometry_root=None):
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

    raster_bytes = {}
    def include(data, media_type, metadata, source_key=None):
        asset = put_asset(data, media_type, metadata)
        if media_type == "application/json" or (geometry_root and media_type in ("image/png", "application/x-npy")):
            raster_bytes[asset["id"]] = data
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
        declared_mapping = camera.get("input_to_canonical_pixel_centres") if use_original else None
        if declared_mapping is not None and (np.asarray(declared_mapping).shape != (3, 3) or not np.allclose(mapping, declared_mapping, rtol=1e-10, atol=1e-10)):
            raise PlatformError("import_pixel_mapping_mismatch", 422)
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
    for record in (record for collection in ("objects", "observed_regions", "unavailable_regions", "unavailable_objects") for record in source.get(collection, [])):
        if record.get("mesh"):
            meshes[record["id"]] = unpack_mesh(root, record["mesh"])
    floor_members, floor_refs = set(), []
    floor = floor_evidence(geometry_root, source)
    if floor:
        matched, vertices, faces = floor_mesh_members(floor, records, meshes, frame_id)
        floor_members = set(matched)
        for relative, raw in floor["files"].items():
            aid = include(raw, "application/json" if relative.endswith(".json") else "application/octet-stream",
                          {"kind": "floor_role_evidence", "sourcePath": relative}, "floor_role/" + relative)
            floor_refs.append({"assetId": aid, "sha256": hashlib.sha256(raw).hexdigest(), "sourcePath": relative})
        manifest["floorBinding"] = {"status": "verified" if matched else "no_matching_source_mesh", "sourceRecordIds": matched,
                                    "sourceFrameId": floor["view"]["frame_id"], "vertexCount": vertices, "faceCount": faces,
                                    "method": "exact_native_indexed_mesh_from_hash_pinned_floor_samples", "sourceRefs": floor_refs}

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
        item["measurements"] = canonical_measurements(measurements, frame_id, [{"assetId": source_asset, "sourceRecordId": old_id}])
        provenance = record.get("provenance", {})
        source_record = provenance.get("source_record", {}) if isinstance(provenance, dict) else {}
        source_refs = source_record.get("source_refs", []) if isinstance(source_record, dict) else []
        floor_sources = [ref for ref in source_refs if isinstance(ref, dict) and ref.get("type") == "text-sam" and ref.get("label") == "floor"] if isinstance(source_refs, list) else []
        label_evidence = {"label": item["label"], "source": "legacy_import"}
        if floor_sources:
            # The saved segmentation query is evidence of a floor category;
            # a display name or the existence of a ground plane is not.
            role_refs = [{"assetId": source_asset, "sourceRecordId": old_id, "sourceEvidence": deepcopy(ref)} for ref in floor_sources]
            item.update(geometryRole="floor", geometryRoleSourceRefs=role_refs)
            label_evidence.update(geometryRole="floor", source="imported_segmentation_query", sourceRefs=deepcopy(role_refs))
        if old_id in floor_members:
            role_refs = [{"assetId": source_asset, "sourceRecordId": old_id, "binding": "exact_native_floor_mesh"}, *deepcopy(floor_refs)]
            item.update(geometryRole="floor", geometryRoleSourceRefs=role_refs)
            label_evidence.update(geometryRole="floor", source="verified_floor_mesh", sourceRefs=deepcopy(role_refs))
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
                box = original_box(canonical_box, camera["canonicalToOriginal"], camera["chosenWidth"], camera["chosenHeight"])
            observation_id = ident("observation", old_id + ":" + frame_ref)
            observation = {"id": observation_id, "revision": 1, "imageId": camera_images[frame_ref], "originalPixelBox": list(box), "maskAssetId": mask_id,
                "pixelMapping": [{"source": "canonical_pixels", "target": "original_pixels", "coordinateConvention": "pixel_centers", "matrix": camera["canonicalToOriginal"].tolist()}],
                "labelEvidence": [deepcopy(label_evidence)], "geometrySupport": None,
                "sourceRefs": [{"assetId": source_asset, "sourceRecordId": old_id}], "sourceBoxEvidence": deepcopy(source_box)}
            if old_id in linked_views:
                view = next((v for v in linked_views[old_id][0] if v.get("frame_id") == frame_ref), None)
                if view:
                    observation.update(originalPixelPolygons=original_polygons(view.get("polygons", []), camera["canonicalToOriginal"]),
                                       polygonCoordinateConvention="pixel_centers", boxConvention="edges_xyxy_right_bottom_exclusive", fillRule=view.get("fill_rule", "evenodd"))
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
                    box = original_box(box, camera["canonicalToOriginal"], camera["chosenWidth"], camera["chosenHeight"])
                elif match.get("shape") != [camera["chosenHeight"], camera["chosenWidth"]]:
                    raise PlatformError("import_linked_mask_dimensions_mismatch", 422)
                oid = ident("observation", old_id + ":" + anchor)
                document["observations"].append({"id": oid, "revision": 1, "imageId": camera_images[anchor], "originalPixelBox": list(box), "maskAssetId": None,
                    "pixelMapping": [{"source": "canonical_pixels", "target": "original_pixels", "matrix": camera["canonicalToOriginal"].tolist(), "coordinateConvention": "pixel_centers"}],
                    "labelEvidence": [deepcopy(label_evidence)], "geometrySupport": None,
                    "sourceRefs": [{"assetId": match["assetId"], "sourceRecordId": old_id, "binding": "exact_candidate_id_and_image_sha256", "imageSha256": camera_hashes[anchor]}],
                    "missingEvidence": ["source_mask_not_packaged"], "originalPixelPolygons": original_polygons(match.get("polygons", []), camera["canonicalToOriginal"]),
                    "polygonCoordinateConvention": "pixel_centers", "boxConvention": "edges_xyxy_right_bottom_exclusive", "fillRule": "evenodd"})
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
            primitive, material = None, None
            source_refs = [{"assetId": source_asset, "sourceRecordId": old_id}]
            vertices = geometry[0][:, :3]
            if record.get("source") == "parametric":
                from ehs_spatial.platform.spatial import primitive_mesh
                primitive, transform, parameters_raw, proof = parametric_definition(root, record, geometry, frame_id)
                colors = geometry[0][:, 6:9]
                if not np.array_equal(colors, np.broadcast_to(colors[0], colors.shape)) or np.any((colors < 0) | (colors > 1)):
                    raise PlatformError("import_parameter_material_not_uniform", 422)
                material = {"color": colors[0].tolist()}
                parameter_id = include(parameters_raw, "application/json", {"kind": "parametric_source", "sha256": proof["parameterSourceSha256"]}, record["metrics"]["parameter_source"])
                source_refs += [{"assetId": asset_id, "sourceRecordId": old_id, "binding": "original_baked_mesh"},
                                {"assetId": parameter_id, "sourceRecordId": old_id, "sha256": proof["parameterSourceSha256"], "binding": "verified_cylinder_parameters", "proof": proof}]
                vertices = primitive_mesh(primitive).vertices
                kind, asset_id = "primitive", None
            item["representations"].append({"id": ident("representation", old_id), "kind": kind, "assetId": asset_id,
                "coordinateFrameId": frame_id, "transform": transform, "primitive": primitive, "placementState": "confirmed" if kind == "observed_surface" else "unconfirmed",
                "placementReason": "imported_observed_surface" if kind == "observed_surface" else "imported_proposal",
                "bounds": {"min": vertices.min(axis=0).tolist(), "max": vertices.max(axis=0).tolist()},
                "sourceRefs": source_refs, **({"material": material} if material else {})})
            if kind in ("generated_mesh", "primitive"):
                item["currentModelTransform"] = deepcopy(transform)
        document["entities"].append(item)
    document["annotations"].append({"id": ident("annotation", "provenance"), "kind": "import_provenance", "sourceAssetId": source_asset,
        "sourceSha256": source_sha, "converter": converter_identity(), "sourceRunId": source.get("run_id"), "sourceUnits": source.get("units"), "limitations": source.get("limitations", []),
        "identityPolicy": "Existing source IDs retained; explicit source-object views share an entity; labels never merge identities", "measurementsPolicy": "Observed native extents retain source provenance; metric scale and physical PCA-axis meanings are not promoted",
        "missingArtifacts": ["native_pointmaps", "native_depth", "native_confidence"], "recomputeRequiresNewCapture": True})
    report = import_report_evidence(scene_path, source, document, manifest, include, cameras, camera_images, ident, legacy_root, observation_root)
    if report:
        document["reportEvidence"] = report
    masks = {}
    if geometry_root:
        from scripts.import_geometry_evidence import import_geometry_evidence
        document["geometryEvidence"] = import_geometry_evidence(geometry_root, scene_path, source, document, manifest, include, ident, frame_id)
        manifest["observationMasks"] = import_observation_masks(geometry_root, source, document, manifest, include)
        _, source_records, _ = observation_mask_sources(geometry_root, source)
        if source_records:
            masks, mask_errors = canonical_observation_masks(document, raster_bytes.__getitem__)
            manifest["sourceIdentity"] = import_source_equivalences(document, source, source_asset, source_records, masks, include)
            manifest["sourceIdentity"]["maskErrors"] = mask_errors
        if report:
            geometry = document["geometryEvidence"]
            for field, label, meaning in (
                ("pointCloudAssetId", "content-point-cloud.glb", "Same-frame native points after the frozen content-valid mask; no new inference or registration"),
                ("sourcePointCloudAssetId", "source-point-cloud.glb", "Verified original same-frame point cloud, including pixels outside the source photo content"),
            ):
                asset_id = geometry[field]
                report["resources"].append({"id": asset_id, "label": label, "kind": "point_cloud", "assetId": asset_id,
                    "runId": geometry["sourceRunId"], "meaning": meaning, "sourceRefs": [{"assetId": geometry["manifestAssetId"]}]})
        provenance = next(annotation for annotation in document["annotations"] if annotation.get("kind") == "import_provenance")
        provenance["missingArtifacts"] = ["native_depth"]
        provenance["recomputeRequiresNewCapture"] = False
        provenance["recomputeStatus"] = "frozen_geometry_imported_not_a_platform_pipeline_checkpoint"
    if report and (report.get("historical") or {}).get("cad"):
        manifest["sourceCadCoverage"] = refresh_source_cad_links(document, raster_bytes.__getitem__, masks)
    validate_document(document)
    manifest.update(entityCount=len(document["entities"]), observationCount=len(document["observations"]),
                    representationCount=sum(len(e["representations"]) for e in document["entities"]), assetCount=len(document["assets"]), documentSha256=digest(document))
    return document, manifest


def run_import(scene_path, repository, blobs, output_dir, title=None, *, legacy_root=None, observation_root=None, geometry_root=None):
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
    geometry_files = []
    if geometry_root:
        from scripts.import_geometry_evidence import geometry_dependencies
        geometry_files = geometry_dependencies(geometry_root, source)
        geometry_files += observation_mask_sources(geometry_root, source)[2]
    parameter_files = set()
    for record in source.get("objects", []):
        if record.get("source") == "parametric":
            relative = (record.get("metrics") or {}).get("parameter_source")
            if not isinstance(relative, str):
                raise PlatformError("import_parameter_source_missing", 422)
            parameter_files.add(source_path(scene_path.parent, relative))
    converter_key = digest({"converter": converter, "evidence": [hashlib.sha256(raw).hexdigest() for _, raw in evidence_documents(scene_path, source)],
                            "reportDependencies": [hashlib.sha256(path.read_bytes()).hexdigest() for path in report_dependencies(scene_path, source, legacy_root, observation_root)],
                            "geometryDependencies": [hashlib.sha256(path.read_bytes()).hexdigest() for path in geometry_files],
                            "parameterDependencies": [hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(parameter_files)],
                            "floorDependencies": [hashlib.sha256(raw).hexdigest() for raw in (floor_evidence(geometry_root, source) or {}).get("files", {}).values()]})
    manifest_path = output_dir / (sha + "." + converter_key + ".manifest.json")
    if manifest_path.exists():
        return json.loads(manifest_path.read_text())

    def put_asset(data, media_type, metadata):
        asset = blobs.put(data, media_type)
        asset["metadata"] = metadata
        return repository.register_asset(project_id, asset)

    document, manifest = import_document(scene_path, put_asset, legacy_root=legacy_root, observation_root=observation_root, geometry_root=geometry_root)
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
    provenance = next(annotation for annotation in document["annotations"] if annotation.get("kind") == "import_provenance")
    imported_capture = {"id": capture_id, "images": images, "task": {"schemaVersion": 1, "kind": "offline_import", "sourceSha256": sha, "imageIds": [image["id"] for image in images],
        "missingArtifacts": provenance["missingArtifacts"], "recomputeRequiresNewCapture": provenance["recomputeRequiresNewCapture"]}}
    if geometry_root:
        imported_capture["task"].update(geometryEvidence=document["geometryEvidence"], recomputeStatus=provenance["recomputeStatus"])
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
    parser.add_argument("--legacy-root", type=Path, help="Optional original report run; all frozen report source hashes must match")
    parser.add_argument("--observation-root", type=Path, help="Optional observation run pinned by the source bridge registry SHA")
    parser.add_argument("--geometry-root", type=Path, help="Optional frozen geometry run pinned by the source manifest SHA")
    args = parser.parse_args()
    from ehs_spatial.platform.runtime import services
    from ehs_spatial.platform.config import PlatformConfig
    repository, blobs = services(PlatformConfig.from_env())
    repository.migrate()
    for scene in args.scenes:
        manifest = run_import(scene, repository, blobs, args.output_dir, legacy_root=args.legacy_root, observation_root=args.observation_root, geometry_root=args.geometry_root)
        print(json.dumps({key: manifest[key] for key in ("projectId", "publicationId", "sceneRevisionId", "entityCount", "observationCount", "assetCount", "newModelCalls")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
