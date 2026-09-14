"""Prepare isolated, hash-bound source completion for every frozen evaluation revision."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import numpy as np
from PIL import Image

from scripts.import_report_evidence import canonical_observation_masks, import_observation_masks, import_source_equivalences, observation_mask_sources
from scripts.research.scan_object_identity import encoded, file_ref, sha


def checked(reference):
    raw = Path(reference["path"]).read_bytes()
    if sha(raw) != reference["sha256"]:
        raise ValueError("Frozen input hash mismatch")
    return raw


def complete_source_masks(profile, document, include):
    """Restore declared detection/refinement masks from a pinned original run."""
    if not profile.get("maskSourceManifest"):
        return {"attachedObservationIds": [], "newModelCalls": 0}
    manifest_raw, objects_raw = checked(profile["maskSourceManifest"]), checked(profile["maskSourceObjects"])
    manifest, evidence = json.loads(manifest_raw), json.loads(objects_raw)
    root = Path(profile["maskSourceManifest"]["path"]).parent
    assets = {a["id"]: a for a in document["assets"]}
    source_records = {o["object_id"]: (index, o) for index, o in enumerate(evidence["objects"])}
    attached = []
    for observation in document["observations"]:
        if observation.get("maskAssetId"):
            continue
        source_ids = {r["sourceRecordId"] for r in observation["sourceRefs"] if r.get("binding") == "exact_candidate_id_and_image_sha256"}
        candidates = []
        for source_id in source_ids & source_records.keys():
            object_index, record = source_records[source_id]
            for view_index, view in enumerate(record["views"]):
                frame = next(f for f in manifest["frames"] if f["frame_id"] == view["frame_id"])
                if frame["sha256"] == assets[observation["imageId"]]["sha256"]:
                    candidates.append((source_id, object_index, view_index, view, frame))
        if not candidates:
            continue
        if len(candidates) != 1:
            raise ValueError("Source mask binding is ambiguous")
        source_id, object_index, view_index, view, frame = candidates[0]
        provenance = view["provenance"]
        if provenance.get("candidate_id") != source_id or provenance.get("mask_resolution") != "original" or not provenance.get("source_refs"):
            raise ValueError("Mask lacks original observation provenance")
        def original_file(relative, expected):
            path = (root / relative).resolve()
            if not path.is_relative_to(root.resolve()):
                raise ValueError("Source mask path escaped pinned run")
            raw = path.read_bytes()
            if sha(raw) != expected:
                raise ValueError("Source mask dependency hash mismatch")
            return raw
        original_file(view["rgb_path"], frame["sha256"])
        for ref in provenance["source_refs"]:
            if ref["type"] not in ("detection", "refinement"):
                raise ValueError("Mask source is not observed detection/refinement evidence")
            original_file(ref["path"], ref["sha256"])
        original = original_file(view["mask_path"], view["sha256"]["mask.png"])
        canonical = original_file(view["canonical_mask_path"], view["sha256"]["canonical_mask.npy"])
        camera = next(c for c in document["cameras"] if c["imageId"] == observation["imageId"])
        expected_mapping = np.linalg.inv(np.asarray(observation["pixelMapping"][0]["matrix"]))
        if not np.allclose(expected_mapping, frame["input_to_canonical_pixel_centres"], atol=1e-12, rtol=0):
            raise ValueError("Source canonical mask pixel mapping differs")
        with Image.open(io.BytesIO(original)) as mask:
            if mask.format != "PNG" or mask.size != (camera["width"], camera["height"]) or len(mask.getbands()) != 1:
                raise ValueError("Original mask dimensions differ from source image")
        grid = np.load(io.BytesIO(canonical), allow_pickle=False)
        if list(grid.shape) != frame["canonical_shape_hw"] or grid.dtype.kind not in "bu" or not np.isin(grid, [0, 1]).all():
            raise ValueError("Canonical mask is not the declared binary grid")
        manifest_id = include(manifest_raw, "application/json", {"kind": "observation_mask_source_manifest"})
        source_id_asset = include(objects_raw, "application/json", {"kind": "observation_mask_source_records"})
        original_id = include(original, "image/png", {"kind": "observation_mask", "sourceRecordId": source_id})
        canonical_id = include(canonical, "application/x-npy", {"kind": "canonical_observation_mask", "sourceRecordId": source_id})
        observation["maskAssetId"] = original_id
        observation["maskEvidence"] = {"originalMaskAssetId": original_id, "canonicalMaskAssetId": canonical_id,
            "originalShape": [camera["height"], camera["width"]], "canonicalShape": list(grid.shape),
            "inputToCanonical": frame["input_to_canonical_pixel_centres"], "geometryManifestAssetId": document["geometryEvidence"]["manifestAssetId"],
            "sourceRefs": [{"assetId": source_id_asset, "jsonPointer": f"/objects/{object_index}/views/{view_index}", "sourceRecordId": source_id,
                "sourceFrameId": frame["frame_id"], "imageSha256": frame["sha256"]}, {"assetId": manifest_id, "sha256": sha(manifest_raw)}]}
        observation["missingEvidence"] = [x for x in observation.get("missingEvidence", []) if x != "source_mask_not_packaged"]
        attached.append(observation["id"])
    return {"attachedObservationIds": attached, "newModelCalls": 0}


def prepare(scan_reference, profiles, output):
    frozen = json.loads(checked(scan_reference))
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "blobs").mkdir()
    registry = {a["id"]: {"id": a["id"], "projectId": a["project_id"], "sha256": a["sha256"].strip(),
        "sizeBytes": a["size_bytes"], "mediaType": a["media_type"], "metadata": a["metadata"], "file": a["file"]} for a in frozen["assets"]}
    result = {"schemaVersion": 1, "sourceScan": scan_reference, "evaluationRevisions": [], "assets": [],
        "captures": frozen["captures"], "newModelCalls": 0, "productionWrites": 0}
    for target in frozen["evaluationRevisions"]:
        document = json.loads(checked(target["document"]))
        before = deepcopy(document)
        source_sha = next((a.get("sourceSha256") for a in document["annotations"] if a.get("kind") == "import_provenance"), None)
        profile = next((p for p in profiles if p["sourceScene"]["sha256"] == source_sha), None)
        status, mask_result = "not_comparable_no_verified_source_geometry", None
        source_identity = {"pairCount": 0, "unavailable": [], "reason": "no_verified_source_geometry"}
        if profile:
            source = json.loads(checked(profile["sourceScene"]))
            source_root = Path(profile["sourceScene"]["path"]).parent
            assets = {a["id"]: a for a in document["assets"]}
            by_sha = {a["sha256"]: a["id"] for a in document["assets"]}

            def include(raw, media, metadata, key=None):
                checksum = sha(raw)
                identity = by_sha.get(checksum)
                if identity is None:
                    identity = str(uuid5(NAMESPACE_URL, "identity-evidence:" + checksum))
                    path = output / "blobs" / checksum
                    if not path.exists():
                        path.write_bytes(raw)
                    asset = {"id": identity, "sha256": checksum, "sizeBytes": len(raw), "mediaType": media, "metadata": metadata, **metadata}
                    document["assets"].append(asset)
                    assets[identity], by_sha[checksum] = asset, identity
                    registry.setdefault(identity, {**asset, "projectId": target["projectId"], "file": file_ref(path)})
                return identity

            camera_ids = {ref["sourceCameraId"]: c["id"] for c in document["cameras"] for ref in c.get("sourceRefs", []) if "sourceCameraId" in ref}
            camera_pairs = []
            for original in source["cameras"]:
                camera = next(c for c in document["cameras"] if c["id"] == camera_ids[original["id"]])
                raw = (source_root / original["original_image"]).read_bytes()
                if sha(raw) != assets[camera["imageId"]]["sha256"] or not np.array_equal(camera["K"], original["original_K"]) or not np.array_equal(camera["cameraToWorld"], original["camera_to_world"]):
                    raise ValueError("Source camera/image does not match frozen revision")
                camera_pairs.append((original, camera))
            frame_ids = {c["coordinateFrameId"] for _, c in camera_pairs}
            if len(frame_ids) != 1:
                raise ValueError("Source cameras do not share one verified native frame")
            coordinate_frame = next(iter(frame_ids))
            if profile.get("templateDocument"):
                template = json.loads(checked(profile["templateDocument"]))
                geometry = deepcopy(template["geometryEvidence"])
                if geometry["coordinateFrameId"] != coordinate_frame:
                    raise ValueError("Native solution frame mismatch")
                referenced = {geometry["manifestAssetId"], geometry["sourcePointCloudAssetId"], geometry["pointCloudAssetId"]}
                referenced.update(a for f in geometry["frames"] for a in f["assets"].values())
                for asset in template["assets"]:
                    if asset["id"] in referenced and asset["id"] not in assets:
                        checked(registry[asset["id"]]["file"])
                        document["assets"].append(deepcopy(asset)); assets[asset["id"]] = asset
                        by_sha[asset["sha256"]] = asset["id"]
                if geometry.get("pointCloudEntityId") not in {e["id"] for e in document["entities"]}:
                    geometry.pop("pointCloudEntityId", None)
                document["geometryEvidence"] = geometry
                bindings = {"entityIds": {ref["sourceRecordId"]: e["id"] for e in document["entities"] for ref in e.get("lineage", []) if ref.get("operation") == "offline_import"},
                    "cameraIds": camera_ids, "floorBinding": {"sourceRecordIds": [ref["sourceRecordId"] for e in document["entities"] for ref in e.get("geometryRoleSourceRefs", []) if ref.get("binding") == "exact_native_floor_mesh"]}}
                mask_result = import_observation_masks(profile["geometryRoot"], source, document, bindings, include)
            else:
                selection = "source valid & finite xyz/conf & positive camera depth & confidence>=0.1; no alpha inferred"
                frame_records, source_pins = [], []
                for original, camera in camera_pairs:
                    pin = next(f for f in profile["frames"] if f["sourceFrameId"] == original["id"])
                    raw_files = {name: checked(ref) for name, ref in pin["files"].items()}
                    if checked(pin["original"]) != checked(registry[camera["imageId"]]["file"]) or raw_files["canonical.png"] != (source_root / original["image"]).read_bytes():
                        raise ValueError("Native source image bytes differ")
                    arrays = {name: np.load(io.BytesIO(raw), allow_pickle=False) for name, raw in raw_files.items() if name.endswith(".npy")}
                    points, valid, conf, k, c2w = [arrays[n] for n in ("pts3d.npy", "valid_mask.npy", "conf.npy", "intrinsics.npy", "camera_to_world.npy")]
                    with Image.open(io.BytesIO(raw_files["canonical.png"])) as im:
                        shape = (im.height, im.width)
                    if points.shape != (*shape, 3) or valid.shape != conf.shape or valid.shape != shape or not np.array_equal(k, original["K"]) or not np.array_equal(c2w, camera["cameraToWorld"]):
                        raise ValueError("Native geometry arrays/camera mismatch")
                    local = points @ np.linalg.inv(c2w)[:3, :3].T + np.linalg.inv(c2w)[:3, 3]
                    content = valid.astype(bool) & np.isfinite(points).all(-1) & np.isfinite(conf) & (local[..., 2] > 0) & (conf >= .1)
                    buffer = io.BytesIO(); np.save(buffer, content, allow_pickle=False)
                    raw_files["content_valid_mask.npy"] = buffer.getvalue()
                    refs = {name: include(raw, "image/png" if name.endswith(".png") else "application/x-npy",
                        {"kind": "native_geometry_evidence", "sourceFrameId": original["id"], "logicalName": name,
                         "coordinateFrameId": coordinate_frame, "selectionRule": selection if name == "content_valid_mask.npy" else "original_source_bytes"}) for name, raw in raw_files.items()}
                    refs["input"] = camera["imageId"]
                    frame_records.append({"sourceFrameId": original["id"], "cameraId": camera["id"], "assets": refs})
                    source_pins.append(pin)
                manifest_raw = encoded({"schemaVersion": 1, "sourceScene": profile["sourceScene"], "frames": source_pins, "selectionRule": selection,
                    "coordinateFrameId": coordinate_frame, "registration": "exact source image bytes, K and cameraToWorld; no cross-solution registration"})
                manifest_id = include(manifest_raw, "application/json", {"kind": "native_geometry_manifest"})
                document["geometryEvidence"] = {"schemaVersion": 1, "sourceRunId": source["run_id"], "coordinateFrameId": coordinate_frame,
                    "manifestAssetId": manifest_id, "frames": frame_records, "selectionRule": selection, "newModelCalls": 0}
                mask_result = complete_source_masks(profile, document, include)
            frame = next(f for f in document["coordinateFrames"] if f["id"] == coordinate_frame)
            reference = {"assetId": document["geometryEvidence"]["manifestAssetId"], "sha256": assets[document["geometryEvidence"]["manifestAssetId"]]["sha256"]}
            if reference not in frame.setdefault("sourceRefs", []):
                frame["sourceRefs"].append(reference)
            if profile.get("geometryRoot"):
                _, identity_records, _ = observation_mask_sources(profile["geometryRoot"], source)
            else:
                identity_raw = checked(profile["maskSourceObjects"]) if profile.get("maskSourceObjects") else b'{"objects":[]}'
                identity_records = [{"kind": "objects", "sourceRecordId": row["object_id"], "view": view,
                    "sourceRaw": identity_raw, "jsonPointer": f"/objects/{i}/views/{j}"}
                    for i, row in enumerate(json.loads(identity_raw)["objects"]) for j, view in enumerate(row.get("views", []))]
            canonical_masks, mask_errors = canonical_observation_masks(document, lambda aid: checked(registry[aid]["file"]))
            if mask_errors:
                raise ValueError("Prepared source masks are incomplete: " + str(mask_errors))
            source_identity = import_source_equivalences(document, source, by_sha[profile["sourceScene"]["sha256"]], identity_records, canonical_masks, include)
            status = "ready_from_verified_source"
        assert before["entities"] == document["entities"]
        assert len(before["observations"]) == len(document["observations"])
        for old, new in zip(before["observations"], document["observations"]):
            for key in ("id", "revision", "imageId", "originalPixelBox", "originalPixelPolygons", "pixelMapping"):
                assert old.get(key) == new.get(key)
            if old.get("maskAssetId"):
                assert old["maskAssetId"] == new["maskAssetId"]
        path = output / (target["revisionId"] + ".json")
        path.write_bytes(encoded(document))
        result["evaluationRevisions"].append({**target, "baseDocument": target["document"], "document": file_ref(path), "preparationStatus": status,
            "maskCompletion": mask_result, "sourceIdentityCompletion": source_identity})
    result["assets"] = list(registry.values())
    (output / "manifest.json").write_bytes(encoded(result) + b"\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(file_ref(args.manifest), json.loads(args.sources.read_bytes()), args.output)
    print(json.dumps({"targets": len(result["evaluationRevisions"]), "manifest": str(args.output.resolve() / "manifest.json")}))
