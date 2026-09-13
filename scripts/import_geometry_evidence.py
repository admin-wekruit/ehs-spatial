"""Import a hash-pinned native point cloud without inference or registration."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import struct

import numpy as np
from PIL import Image

from ehs_spatial.platform.contracts import PlatformError

FRAME_FILES = ("pts3d.npy", "conf.npy", "valid_mask.npy", "content_valid_mask.npy",
               "intrinsics.npy", "camera_to_world.npy", "canonical.png")
SELECTION_RULE = "native valid & declared alpha & finite & nonzero points & conf>=0.1"


def require(condition, code):
    if not condition:
        raise PlatformError("import_geometry_" + code, 422)


def safe_path(root, name):
    path = (Path(root) / name).resolve()
    require(path.is_relative_to(Path(root).resolve()) and path.is_file(), "path_invalid")
    return path


def sha(data):
    return hashlib.sha256(data).hexdigest()


def pinned_manifest(root, source):
    raw = safe_path(root, "manifest.json").read_bytes()
    provenance = source.get("provenance", {})
    observed = provenance.get("observed_ranges", {})
    pins = [provenance.get("source_bridge", {}).get("target_manifest_sha256"),
            observed.get("source_sha256", {}).get("manifest.json")]
    require(any(pins) and all(value == sha(raw) for value in pins if value), "source_hash_mismatch")
    manifest = json.loads(raw)
    run_id = manifest.get("experiment")
    require(isinstance(run_id, str) and run_id == source.get("source_run_id", source.get("run_id")), "source_run_mismatch")
    require(not observed.get("source_run_id") or observed["source_run_id"] == run_id, "source_run_mismatch")
    return manifest, raw


def geometry_dependencies(geometry_root, source):
    """All consumed run artifacts participate in the caller's converter cache key."""
    root = Path(geometry_root).resolve()
    manifest, _ = pinned_manifest(root, source)
    paths = [safe_path(root, "manifest.json"), safe_path(root, "geometry/point_cloud.glb")]
    for frame in manifest["frames"]:
        paths.extend(safe_path(root, frame[key]) for key in ("input", "canonical", "alpha"))
        paths.extend(safe_path(root, "geometry/frames/" + frame["frame_id"] + "/" + name) for name in FRAME_FILES)
    return sorted(set(paths))


def read_cloud(raw):
    """Read the verified single-node, uncompressed POINTS export profile only."""
    # ponytail: this offline importer accepts one explicit GLB source profile;
    # another layout needs a reviewed decoder, not an implicit coordinate fit.
    require(len(raw) >= 28, "cloud_invalid")
    magic, version, total = struct.unpack_from("<4sII", raw)
    length, kind = struct.unpack_from("<II", raw, 12)
    require(magic == b"glTF" and version == 2 and total == len(raw) and kind == 0x4E4F534A and length % 4 == 0 and 28 + length <= len(raw), "cloud_invalid")
    data = json.loads(raw[20:20 + length])
    binary_length, binary_kind = struct.unpack_from("<II", raw, 20 + length)
    require(binary_kind == 0x004E4942 and 28 + length + binary_length == len(raw), "cloud_invalid")
    binary = raw[28 + length:]
    require(len(data.get("nodes", [])) == 1 and len(data.get("meshes", [])) == 1 and len(data.get("buffers", [])) == 1 and data.get("scenes") == [{"nodes": [0]}] and data.get("scene", 0) == 0, "cloud_profile_unsupported")
    node = data["nodes"][0]
    require(node.get("mesh") == 0 and not node.get("children"), "cloud_profile_unsupported")
    for key, identity in (("matrix", np.eye(4).reshape(-1).tolist()), ("translation", [0, 0, 0]), ("rotation", [0, 0, 0, 1]), ("scale", [1, 1, 1])):
        require(key not in node or node[key] == identity, "cloud_transform_not_identity")
    primitives = data["meshes"][0].get("primitives", [])
    require(len(primitives) == 1 and primitives[0].get("mode") == 0 and set(primitives[0].get("attributes", {})) == {"POSITION", "COLOR_0"} and "indices" not in primitives[0] and not primitives[0].get("extensions"), "cloud_profile_unsupported")
    require("uri" not in data["buffers"][0] and data["buffers"][0]["byteLength"] <= len(binary), "cloud_invalid")

    def array(name, component, width, dtype):
        accessor = data["accessors"][primitives[0]["attributes"][name]]
        view = data["bufferViews"][accessor["bufferView"]]
        require(accessor.get("componentType") == component and accessor.get("type") == "VEC" + str(width) and not accessor.get("sparse") and view.get("buffer") == 0 and "byteStride" not in view, "cloud_profile_unsupported")
        count = accessor["count"]
        require(type(count) is int and count > 0, "cloud_invalid")
        start = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
        size = count * width * np.dtype(dtype).itemsize
        require(start >= 0 and start + size <= len(binary) and accessor.get("byteOffset", 0) + size <= view["byteLength"], "cloud_invalid")
        return np.frombuffer(binary, dtype=dtype, count=count * width, offset=start).reshape(count, width)

    points, colors = array("POSITION", 5126, 3, "<f4"), array("COLOR_0", 5121, 4, "u1")
    require(len(points) == len(colors) and np.isfinite(points).all(), "cloud_invalid")
    return points, colors


def write_cloud(points, colors):
    """Deterministic native-coordinate GLB; no mesh conversion or resampling."""
    points = np.asarray(points, dtype="<f4")
    colors = np.asarray(colors, dtype="u1")
    binary = points.tobytes() + colors.tobytes()
    data = {"asset": {"version": "2.0", "generator": "Panoptes frozen native cloud subset"},
            "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}],
            "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "COLOR_0": 1}, "mode": 0}]}],
            "buffers": [{"byteLength": len(binary)}],
            "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": points.nbytes},
                            {"buffer": 0, "byteOffset": points.nbytes, "byteLength": colors.nbytes}],
            "accessors": [{"bufferView": 0, "componentType": 5126, "count": len(points), "type": "VEC3", "min": points.min(0).tolist(), "max": points.max(0).tolist()},
                          {"bufferView": 1, "componentType": 5121, "count": len(colors), "type": "VEC4", "normalized": True}]}
    encoded = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 4)
    binary += b"\0" * (-len(binary) % 4)
    return (struct.pack("<4sII", b"glTF", 2, 28 + len(encoded) + len(binary)) +
            struct.pack("<II", len(encoded), 0x4E4F534A) + encoded +
            struct.pack("<II", len(binary), 0x004E4942) + binary)


def import_geometry_evidence(geometry_root, scene_path, source, document, manifest, include, ident, frame_id):
    root, scene_root = Path(geometry_root).resolve(), Path(scene_path).resolve().parent
    frozen, manifest_raw = pinned_manifest(root, source)
    records = frozen["frames"]
    specs = {f["frame_id"]: f for f in frozen["geometry"]["frames"]}
    cameras = {c["id"]: c for c in source["cameras"]}
    require(len(records) == len(specs) == len(cameras) and len({f["frame_id"] for f in records}) == len(records), "frame_set_mismatch")
    require(frozen["geometry"].get("content_valid_rule") == SELECTION_RULE, "selection_rule_mismatch")
    raw_cloud = safe_path(root, "geometry/point_cloud.glb").read_bytes()
    actual_points, actual_colors = read_cloud(raw_cloud)
    points, colors, source_points, source_colors, frame_data = [], [], [], [], []
    for record in records:
        fid = record["frame_id"]
        require(fid in cameras and fid in specs and fid in manifest["cameraIds"], "frame_set_mismatch")
        camera, spec = cameras[fid], specs[fid]
        files = {}
        for name in FRAME_FILES:
            raw = safe_path(root, "geometry/frames/" + fid + "/" + name).read_bytes()
            require(sha(raw) == spec["files"].get(name), "artifact_hash_mismatch")
            files[name] = raw
        for name, key, pin in (("input", "input", "sha256"), ("canonical", "canonical", "canonical_sha256"), ("alpha", "alpha", "alpha_sha256")):
            files[name] = safe_path(root, record[key]).read_bytes()
            require(sha(files[name]) == record[pin], "artifact_hash_mismatch")
        require(files["canonical"] == files["canonical.png"] == safe_path(scene_root, camera["image"]).read_bytes(), "image_mismatch")
        require(files["input"] == safe_path(scene_root, camera["original_image"]).read_bytes(), "image_mismatch")
        arrays = {name: np.load(io.BytesIO(raw), allow_pickle=False) for name, raw in files.items() if name.endswith(".npy")}
        alpha = np.load(io.BytesIO(files["alpha"]), allow_pickle=False)
        p, valid, content, conf = (arrays[name] for name in ("pts3d.npy", "valid_mask.npy", "content_valid_mask.npy", "conf.npy"))
        with Image.open(io.BytesIO(files["canonical.png"])) as image:
            rgb = np.asarray(image.convert("RGB"))
        with Image.open(io.BytesIO(files["input"])) as image:
            require(image.size == (record["width"], record["height"]) == (camera["original_width"], camera["original_height"]), "image_dimensions_mismatch")
        shape = rgb.shape[:2]
        require(p.shape == (*shape, 3) and valid.shape == content.shape == conf.shape == alpha.shape == shape and valid.dtype == content.dtype == alpha.dtype == bool, "array_shape_invalid")
        require(shape == (camera["height"], camera["width"]), "image_dimensions_mismatch")
        rect = record["content_rect_xyxy"]
        require(len(rect) == 4 and all(type(n) is int for n in rect) and 0 <= rect[0] < rect[2] <= shape[1] and 0 <= rect[1] < rect[3] <= shape[0], "content_rect_invalid")
        declared = np.zeros(shape, bool)
        declared[rect[1]:rect[3], rect[0]:rect[2]] = True
        require(np.array_equal(alpha, declared), "alpha_rect_mismatch")
        expected = valid & alpha & np.isfinite(p).all(-1) & (np.linalg.norm(p, axis=-1) > 1e-6) & (conf >= .1)
        require(np.array_equal(content, expected) and content.any() and int(content.sum()) == spec["valid_content_points"], "content_mask_mismatch")
        k, c = arrays["intrinsics.npy"], arrays["camera_to_world.npy"]
        require(np.array_equal(k, camera["K"]) and np.array_equal(c, camera["camera_to_world"]), "camera_mismatch")
        affine = np.asarray(record["input_to_canonical_pixel_centres"])
        require(np.array_equal(affine, camera["input_to_canonical_pixel_centres"]) and np.allclose(k, affine @ np.asarray(camera["original_K"]), rtol=1e-10, atol=1e-10), "pixel_mapping_mismatch")
        current = next((v for v in document["cameras"] if v["id"] == manifest["cameraIds"][fid]), None)
        require(current is not None and current["coordinateFrameId"] == frame_id and np.array_equal(current["cameraToWorld"], c) and np.array_equal(current["K"], camera["original_K"]), "target_camera_mismatch")
        source_points.append(p[valid]); source_colors.append(rgb[valid])
        points.append(p[content]); colors.append(rgb[content]); frame_data.append((fid, files, int(valid.sum()), int(content.sum())))
    require(np.array_equal(actual_points, np.concatenate(source_points)) and np.array_equal(actual_colors[:, :3], np.concatenate(source_colors)) and (actual_colors[:, 3] == 255).all(), "cloud_native_mismatch")
    points = np.concatenate(points)
    rgb = np.concatenate(colors)
    rgba = np.column_stack((rgb, np.full(len(rgb), 255, np.uint8)))
    filtered = write_cloud(points, rgba)
    # Register only after the full evidence chain has passed.
    manifest_asset = include(manifest_raw, "application/json", {"kind": "native_geometry_manifest", "sourceRunId": frozen["experiment"]}, "native_geometry/manifest.json")
    cloud_asset = include(raw_cloud, "model/gltf-binary", {"kind": "native_point_cloud", "pointCount": len(actual_points), "coordinateFrameId": frame_id}, "native_geometry/source.glb")
    filtered_asset = include(filtered, "model/gltf-binary", {"kind": "point_cloud", "pointCount": len(points), "coordinateFrameId": frame_id, "selectionRule": SELECTION_RULE}, "native_geometry/content.glb")
    frame_refs = []
    for fid, files, raw_count, content_count in frame_data:
        refs = {}
        for name, raw in files.items():
            if name == "canonical":
                continue
            media_type = "application/x-npy" if name.endswith(".npy") or name == "alpha" else "image/png" if name.endswith(".png") else "image/jpeg"
            refs[name] = include(raw, media_type, {"kind": "native_geometry_evidence", "sourceFrameId": fid, "logicalName": name, "coordinateFrameId": frame_id}, "native_geometry/" + fid + "/" + name)
        frame_refs.append({"sourceFrameId": fid, "cameraId": manifest["cameraIds"][fid], "assets": refs, "sourcePointCount": raw_count, "contentPointCount": content_count})
    refs = [{"assetId": manifest_asset}, {"assetId": cloud_asset}]
    identity = {"coordinateFrameId": frame_id, "position": [0, 0, 0], "quaternion": [0, 0, 0, 1], "scale": [1, 1, 1]}
    entity_id = ident("entity", "native-content-point-cloud")
    require(all(e["id"] != entity_id for e in document["entities"]), "duplicate_context")
    document["entities"].append({"id": entity_id, "label": "观测点云 / Observed point cloud", "sourceContext": True, "visible": True,
        "associationState": "confirmed", "observationRefs": [], "currentModelTransform": None, "measurements": {},
        "representations": [{"id": ident("representation", "native-content-point-cloud"), "kind": "point_cloud", "assetId": filtered_asset,
            "coordinateFrameId": frame_id, "transform": identity, "placementState": "confirmed", "placementReason": "verified_native_pointmaps",
            "bounds": {"min": points.min(0).tolist(), "max": points.max(0).tolist()}, "sourceRefs": deepcopy(refs)}], "sourceRefs": refs})
    manifest["pointCloudEntityId"] = entity_id
    return {"schemaVersion": 1, "sourceRunId": frozen["experiment"], "coordinateFrameId": frame_id, "manifestAssetId": manifest_asset,
            "sourcePointCloudAssetId": cloud_asset, "pointCloudAssetId": filtered_asset, "sourcePointCount": len(actual_points),
            "contentPointCount": len(points), "pointCloudEntityId": entity_id, "frames": frame_refs,
            "registration": "identity; exact native vertices, RGB, camera matrices and frozen source hashes verified",
            "selectionRule": SELECTION_RULE, "newModelCalls": 0}
