"""One-way conversion of a saved report into versioned, asset-backed evidence.

Only explicit source IDs establish links. Historical results retain their own run;
an attached report never creates a current-scene policy evaluation.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from ehs_spatial.platform.contracts import PlatformError


def _path(root, value):
    path = (Path(root) / value).resolve()
    if not path.is_relative_to(Path(root).resolve()) or not path.is_file():
        raise PlatformError("import_report_asset_path_invalid", 422)
    return path


def _read(path, expected=None):
    raw = Path(path).read_bytes()
    if expected and hashlib.sha256(raw).hexdigest() != expected:
        raise PlatformError("import_report_source_hash_mismatch", 422)
    return raw


def _camel(value):
    """Normalize frozen legacy evidence records at the offline boundary only."""
    if isinstance(value, list):
        return [_camel(item) for item in value]
    if isinstance(value, dict):
        return {key.split("_")[0] + "".join(part[:1].upper() + part[1:] for part in key.split("_")[1:]): _camel(item)
                for key, item in value.items()}
    return value


def original_polygons(polygons, matrix):
    result = []
    for polygon in polygons:
        points = np.asarray(polygon, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
            raise PlatformError("import_report_polygon_invalid", 422)
        homogeneous = np.c_[points, np.ones(len(points))] @ np.asarray(matrix).T
        if np.any(np.abs(homogeneous[:, 2]) < 1e-12):
            raise PlatformError("import_report_pixel_mapping_invalid", 422)
        result.append((homogeneous[:, :2] / homogeneous[:, 2:]).tolist())
    return result


def original_box(box, matrix, width, height):
    # Source boxes are edge coordinates; K maps pixel centres. Translate edges
    # through the same pixel lattice rather than treating them as centre samples.
    points = original_polygons([[[box[0] - .5, box[1] - .5], [box[2] - .5, box[3] - .5]]], matrix)[0]
    return [max(0., points[0][0] + .5), max(0., points[0][1] + .5),
            min(float(width), points[1][0] + .5), min(float(height), points[1][1] + .5)]


def canonical_measurements(measurements, frame_id, source_refs):
    dimensions = measurements.get("dimensions_native", {})
    if measurements.get("status") != "available" or not isinstance(dimensions, dict):
        return {}
    values = [dimensions.get(key) for key in ("width", "depth", "height")]
    if any(type(value) not in (int, float) or not np.isfinite(value) or value < 0 for value in values):
        return {}
    return {"widthNative": values[0], "depthNative": values[1], "groundHeightNative": values[2],
            "unit": "native", "coverage": measurements.get("coverage"), "source": "imported_observed_measurement",
            "coordinateFrameId": frame_id, "uncertaintyNative": None, "basis": _camel(measurements.get("basis")),
            "sourceRefs": source_refs, "scaleEvidence": _camel(measurements.get("scale")),
            "orientationEvidence": _camel(measurements.get("orientation")), "quality": _camel(measurements.get("quality"))}


def report_documents(scene_path, source):
    matches = []
    for path in Path(scene_path).parent.glob("*.json"):
        if path.resolve() == Path(scene_path).resolve():
            continue
        try:
            value = json.loads(path.read_bytes())
        except (ValueError, OSError):
            continue
        if isinstance(value, dict) and value.get("scene_url") == Path(scene_path).name and value.get("reconstruction_run_id") == source.get("run_id"):
            matches.append((path, value))
    if len(matches) > 1:
        raise PlatformError("import_report_ambiguous", 422)
    return matches


def report_dependencies(scene_path, source, legacy_root=None, observation_root=None):
    """Hash every attached dependency into the retry-safe converter identity."""
    root = Path(scene_path).parent
    files = set()
    for path, report in report_documents(scene_path, source):
        files.add(path)
        legacy = report.get("legacy", {})
        for value in (legacy.get("evidence_url"), legacy.get("cad", {}).get("url"), legacy.get("cad", {}).get("map_url")):
            if value:
                files.add(_path(root, value))
        for value in source.get("provenance", {}).get("source_sha256", {}):
            files.add(_path(root, value))
        audit = source.get("provenance", {}).get("source_bridge", {}).get("audit")
        if audit:
            files.add(_path(root, audit))
        for name in ["comparisons.json", *[str(item["id"]) + "-comparison.json" for item in report.get("objects", [])]]:
            if (root / name).is_file():
                files.add(_path(root, name))
        if legacy_root:
            old = Path(legacy_root)
            for name in ("policies.json", "assessment.json", "scene.json", "inventory/inventory.json", "detection/detections.json", "observations.json"):
                files.add(_path(old, name))
            files.update(old.glob("geometry/frames/*/canonical.png"))
        if observation_root:
            original = Path(observation_root)
            registry = _path(original, "object-evidence.json")
            files.update((registry, _path(original, "detection/detections.json")))
            files.update(_path(original, frame["image_path"]) for frame in json.loads(registry.read_bytes())["frames"])
    return sorted(files)


def import_report_evidence(scene_path, source, document, manifest, include, cameras, camera_images, ident, legacy_root=None, observation_root=None):
    matches = report_documents(scene_path, source)
    if not matches:
        return None
    path, saved = matches[0]
    root = path.parent
    run_id = saved["reconstruction_run_id"]
    source_id = include(path.read_bytes(), "application/json", {"kind": "report_source"}, path.name)
    refs = lambda pointer="": [{"assetId": source_id, "jsonPointer": pointer}]
    result = {"schemaVersion": 1, "sourceRunId": saved.get("source_run_id"), "reconstructionRunId": run_id,
              "sourceRefs": refs(), "mappingNotes": deepcopy(saved.get("mapping_notes", [])),
              "frames": [], "objects": [], "resources": [], "imageInterpretations": [],
              "quality": {"metricMeaning": "Input-view consistency, not held-out reconstruction accuracy or physical calibration",
                          "scale": _camel(saved.get("scale", {})), "limitations": deepcopy(source.get("limitations", []))}}
    canonical_hashes = {fid: hashlib.sha256(_read(_path(Path(scene_path).parent, camera["image"]))).hexdigest() for fid, camera in cameras.items()}
    for frame in saved.get("frames", []):
        fid = frame["id"]
        if fid not in cameras:
            raise PlatformError("import_report_camera_missing", 422)
        if frame.get("url") and hashlib.sha256(_read(_path(root, frame["url"]))).hexdigest() != canonical_hashes[fid]:
            raise PlatformError("import_report_image_mismatch", 422)
        result["frames"].append({"runId": run_id, "sourceFrameId": fid, "imageId": camera_images[fid], "cameraId": manifest["cameraIds"][fid]})
    entities = {entity["id"]: entity for entity in document["entities"]}
    observations = {observation["id"]: observation for observation in document["observations"]}
    assets = {asset["id"]: asset for asset in document["assets"]}
    association_refs = {}
    for index, obj in enumerate(saved.get("objects", [])):
        source_record = obj.get("scene_object_id", obj["id"])
        entity_id = manifest["entityIds"].get(source_record)
        item = {"entityId": entity_id, "sourceRecordId": source_record, "sourceKind": obj.get("source"),
                "mappingStatus": "explicit_source_id" if entity_id else "unassociated", "views": [],
                "metricsMeaning": obj.get("metrics_source") or result["quality"]["metricMeaning"], "sourceRefs": refs(f"/objects/{index}")}
        for view_index, view in enumerate(obj.get("views", [])):
            fid = view["frame_id"]
            if fid not in cameras:
                raise PlatformError("import_report_camera_missing", 422)
            camera = cameras[fid]
            converted = {"sourceFrameId": fid, "imageId": camera_images[fid], "cameraId": manifest["cameraIds"][fid],
                "originalPixelBox": original_box(view["bbox"], camera["canonicalToOriginal"], camera["chosenWidth"], camera["chosenHeight"]),
                "originalPixelPolygons": original_polygons(view.get("polygons", []), camera["canonicalToOriginal"]),
                "coordinateConvention": "pixel_centers", "boxConvention": "edges_xyxy_right_bottom_exclusive", "fillRule": view.get("fill_rule", "evenodd")}
            if entity_id:
                entity = entities[entity_id]
                binding = {"assetId": source_id, "jsonPointer": f"/objects/{index}/views/{view_index}",
                    "sourceRecordId": source_record, "sourceFrameId": fid, "cameraId": converted["cameraId"],
                    "imageSha256": assets[converted["imageId"]]["sha256"], "canonicalImageSha256": canonical_hashes[fid],
                    "binding": "explicit_source_id_and_camera_pixel_mapping"}
                observation = next((observations[oid] for oid in entity["observationRefs"] if observations[oid]["imageId"] == converted["imageId"]), None)
                if observation is None:
                    oid = ident("observation", source_record + ":" + fid)
                    observation = {"id": oid, "revision": 1, "imageId": converted["imageId"], "maskAssetId": None,
                        "originalPixelBox": deepcopy(converted["originalPixelBox"]), "originalPixelPolygons": deepcopy(converted["originalPixelPolygons"]),
                        "polygonCoordinateConvention": "pixel_centers", "boxConvention": converted["boxConvention"], "fillRule": converted["fillRule"],
                        "pixelMapping": [{"source": "canonical_pixels", "target": "original_pixels", "coordinateConvention": "pixel_centers", "matrix": camera["canonicalToOriginal"].tolist()}],
                        "labelEvidence": [{"label": entity["label"], "source": "imported_report_observation", "sourceRefs": [deepcopy(binding)]}],
                        "geometrySupport": None, "sourceRefs": [], "missingEvidence": ["source_mask_not_packaged"]}
                    observations[oid] = observation
                    document["observations"].append(observation)
                    entity["observationRefs"].append(oid)
                    manifest["observationIds"].setdefault(source_record, oid)
                observation["sourceRefs"].append(binding)
                converted["observationId"] = observation["id"]
                association_refs.setdefault(entity_id, []).append(deepcopy(binding))
                if "source_observation_binding_pending" in entity.get("missingEvidence", []):
                    entity["missingEvidence"].remove("source_observation_binding_pending")
            item["views"].append(converted)
        item["plan"] = _camel(obj.get("plan"))
        if entity_id and item["plan"] and item["plan"].get("hull"):
            entity = entities[entity_id]
            entity.setdefault("measurements", {})["projectedHull"] = {
                "coordinateFrameId": document["coordinateFrames"][0]["id"],
                "nativeToPlane": deepcopy(saved["plan"]["native_to_floor"]), "points": deepcopy(item["plan"]["hull"]),
                "representationSnapshot": deepcopy(entity["representations"]), "modelTransformSnapshot": deepcopy(entity["currentModelTransform"]),
                "meaning": "Frozen projected hull; not ground contact or a safety zone", "sourceRefs": deepcopy(item["sourceRefs"])}
        item["metrics"] = _camel(obj.get("metrics"))
        if item["metrics"]:
            for view in item["metrics"].get("views", []):
                view["sourceFrameId"] = view.pop("frameId")
        result["objects"].append(item)
    for entity_id, bindings in association_refs.items():
        entity = entities[entity_id]
        if len({observations[oid]["imageId"] for oid in entity["observationRefs"]}) > 1:
            entity["associationState"] = "confirmed"
            entity["associationEvidence"] = {"method": "explicit_source_id", "status": "confirmed",
                "observationIds": list(entity["observationRefs"]), "sourceRefs": deepcopy(bindings)}
            entity["lineage"].append({"operation": "associate_observations", "method": "explicit_source_id_and_camera_pixel_mapping",
                "observationIds": list(entity["observationRefs"]), "sourceRefs": bindings})
    result["plan"] = {**_camel(saved.get("plan", {})), "coordinateFrameId": document["coordinateFrames"][0]["id"],
                      "meaning": "Saved native geometry projected onto the saved floor; convex hull is not a measured CAD/contact footprint"}

    def resource(path, kind, meaning, expected=None, resource_run=run_id):
        raw = _read(path, expected)
        media = "application/json" if path.suffix == ".json" else "application/x-blender" if path.suffix == ".blend" else "application/octet-stream"
        aid = include(raw, media, {"kind": "report_" + kind, "sourceRunId": resource_run}, str(path.relative_to(root)) if path.is_relative_to(root) else None)
        result["resources"].append({"id": aid, "label": path.name, "kind": kind, "assetId": aid, "runId": resource_run,
                                    "meaning": meaning, "sourceRefs": [{"assetId": aid}]})
        return aid

    legacy = saved.get("legacy", {})
    if legacy:
        legacy_run = legacy["source_run_id"]
        evidence_path = _path(root, legacy["evidence_url"])
        evidence = json.loads(evidence_path.read_bytes())
        if evidence.get("source_run_id") != legacy_run:
            raise PlatformError("import_report_run_mismatch", 422)
        evidence_id = resource(evidence_path, "historical_evidence", "Frozen historical results; not evaluated against this scene revision", resource_run=legacy_run)
        erefs = lambda pointer="": [{"assetId": evidence_id, "jsonPointer": pointer}]
        historical = {"runId": legacy_run, "status": legacy.get("summary", {}).get("status"),
            "notice": "Historical source-run result only; not current-scene compliance. No authoritative regulation verification or recorded expert approval is implied.",
            "summary": _camel(legacy.get("summary", {})), "assessment": _camel(evidence.get("assessment", {})),
            "policies": _camel(evidence.get("policies", [])), "facts": _camel(evidence.get("facts", [])),
            "entities": _camel(evidence.get("entities", [])), "inventory": [], "findings": [], "frames": [], "sourceRefs": erefs()}
        for index, record in enumerate(legacy.get("inventory", [])):
            links = record.get("mapped_object_ids", [])
            entity_ids = [manifest["entityIds"][key] for key in links if key in manifest["entityIds"]]
            historical["inventory"].append({"inventoryIndex": record["inv"], "label": record["label"], "sourceFrameId": record.get("frame_id"),
                "source": record.get("source"), "mappingStatus": record.get("mapping_status") if entity_ids else "unassociated", "entityIds": entity_ids,
                "imageBox": record.get("image_bbox"), "score": record.get("score"), "heightM": record.get("height_m"), "sizeM": record.get("size_m"),
                "cameraDistanceM": record.get("camera_dist_m"), "orientationDeg": record.get("orientation_deg"), "tiltDeg": record.get("tilt_deg"),
                "measurementMeaning": "Historical model-estimated values; orientation/tilt are not installation acceptance measurements",
                "sourceRefs": refs(f"/legacy/inventory/{index}")})
        policies = {item["id"]: item for item in historical["policies"]}
        for index, finding in enumerate(legacy.get("findings", [])):
            policy = policies.get(finding["id"], {})
            metrics = finding.get("metrics", {})
            historical["findings"].append({"id": finding["id"], "title": finding["title"], "status": finding["status"], "summary": finding.get("summary"),
                "predicate": metrics.get("predicate"), "threshold": metrics.get("threshold"), "unit": metrics.get("unit"),
                "violations": deepcopy(policy.get("violations", _camel(metrics.get("violations", [])))),
                "warnings": deepcopy(policy.get("warnings", [])), "facts": deepcopy(policy.get("facts", [])),
                "sourceFrameIds": finding.get("evidence", {}).get("frame_ids", []), "sourceRefs": refs(f"/legacy/findings/{index}") + erefs("/policies")})
        cad = legacy.get("cad")
        if cad:
            cad_path, map_path = _path(root, cad["url"]), _path(root, cad["map_url"])
            mapping = json.loads(map_path.read_bytes())
            if mapping.get("inventory_sha256") != saved.get("source_sha256", {}).get("legacy_inventory"):
                raise PlatformError("import_report_inventory_hash_mismatch", 422)
            raw = _read(cad_path, mapping.get("image_sha256"))
            with Image.open(io.BytesIO(raw)) as image:
                if image.size != (cad["width"], cad["height"]):
                    raise PlatformError("import_report_cad_dimensions_mismatch", 422)
            cad_id = include(raw, "image/png", {"kind": "historical_cad", "sourceRunId": legacy_run}, cad["url"])
            map_id = resource(map_path, "historical_cad_map", "Historical CAD pixels; separate source-run geometry and estimated scale", resource_run=legacy_run)
            inventory = {item["inventoryIndex"]: item for item in historical["inventory"]}
            historical["cad"] = {"assetId": cad_id, "width": cad["width"], "height": cad["height"],
                "regions": [{"inventoryIndex": region["inv"], "entityIds": inventory.get(region["inv"], {}).get("entityIds", []),
                             "polygon": region["polygon"], "centroid": region.get("centroid")} for region in mapping.get("objects", cad["regions"])],
                "clipToImage": True, "clippedInventoryIndices": cad.get("clipped_inventory_indices", []), "sourceRefs": [{"assetId": map_id}]}
        if legacy_root:
            _attach_legacy(Path(legacy_root), saved, evidence, historical, result, include)
        else:
            historical["missingEvidence"] = ["original_policy_specs", "historical_source_images", "image_interpretation_records"]
        result["historical"] = historical

    provenance = source.get("provenance", {})
    for relative, sha in provenance.get("source_sha256", {}).items():
        path = _path(root, relative)
        kind = "blender_source" if path.suffix == ".blend" else "quality" if "comparisons" in path.name or "verification" in path.name else "model_source"
        resource(path, kind, "Frozen original source/experiment; not a complete export of the current canonical revision", sha, source.get("source_run_id", run_id))
    audit = provenance.get("source_bridge", {}).get("audit")
    if audit:
        aid = resource(_path(root, audit), "frame_relations", "Verified source-to-target photo resampling; source 3D was not registered")
        bridge = json.loads(_path(root, audit).read_bytes())
        result["frameRelations"] = {"sourceRunId": bridge.get("source_run_id"), "targetRunId": bridge.get("target_run_id"),
            "registration": provenance["source_bridge"].get("registration"), "sourceRefs": [{"assetId": aid}],
            "frames": [{"sourceFrameId": fid, **_camel(binding)} for fid, binding in bridge.get("mapped_source_frames", {}).items()],
            "targetFramesWithoutRegistry": bridge.get("target_frames_without_registry", []),
            "sourceCandidateCount": bridge.get("source_candidates"), "mappedCandidateCount": bridge.get("mapped_candidates"),
            "omittedCandidates": _camel(bridge.get("omitted_candidates", []))}
    for name in ["comparisons.json", *[str(item["id"]) + "-comparison.json" for item in saved.get("objects", [])]]:
        path = root / name
        if path.is_file():
            resource(_path(root, name), "quality", "Frozen RecGen input-fit metrics; not metrics for later parametric replacements or physical accuracy", resource_run=source.get("source_run_id", run_id))
    if observation_root:
        if not audit:
            raise PlatformError("import_report_bridge_missing", 422)
        _attach_observations(Path(observation_root), source, bridge, result, document, manifest, include, cameras, camera_images)
    return result


def _attach_legacy(root, saved, evidence, historical, result, include):
    """Optional original run, accepted only after all frozen report hashes match."""
    pins = evidence["source_sha256"]
    for name in ("policies.json", "assessment.json", "scene.json"):
        _read(_path(root, name), pins[name])
    _read(_path(root, "inventory/inventory.json"), saved["source_sha256"]["legacy_inventory"])
    old_scene = json.loads(_path(root, "scene.json").read_bytes())
    if old_scene.get("run_id") != historical["runId"]:
        raise PlatformError("import_report_run_mismatch", 422)
    raw = _path(root, "policies.json").read_bytes()
    pid = include(raw, "application/json", {"kind": "historical_policy_source", "sourceRunId": historical["runId"]})
    specs = {spec["policy_id"]: _camel(spec) for spec in json.loads(raw)["specs"]}
    for policy in historical["policies"]:
        policy["spec"] = specs.get(policy["id"])
        policy["sourceRefs"] = [{"assetId": pid}]
    for path in sorted(root.glob("geometry/frames/*/canonical.png")):
        with Image.open(path) as image:
            width, height = image.size
        image_id = include(path.read_bytes(), "image/png", {"kind": "historical_source_image", "sourceRunId": historical["runId"], "width": width, "height": height})
        historical["frames"].append({"runId": historical["runId"], "sourceFrameId": path.parent.name, "imageId": image_id, "cameraId": None, "width": width, "height": height, "coordinateConvention": "pixel_centers"})
    detection_path = _path(root, "detection/detections.json")
    detection = json.loads(detection_path.read_bytes())
    if detection.get("run_id") != historical["runId"]:
        raise PlatformError("import_report_run_mismatch", 422)
    did = include(detection_path.read_bytes(), "application/json", {"kind": "historical_image_interpretation", "sourceRunId": historical["runId"]})
    interpretations = {"runId": historical["runId"], "items": [], "missing": _camel(detection.get("missing", [])),
        "rejected": [{key: _camel(value) for key, value in item.items() if key not in ("rle", "mask_path", "image_path")} for item in detection.get("rejected", [])],
        "notice": "Saved model detections/missing candidates, not compliance findings; unbound image context remains unassociated", "sourceRefs": [{"assetId": did}]}
    for index, item in enumerate(detection.get("detections", [])):
        interpretations["items"].append({"sourceRecordId": item.get("item_id"), "label": item.get("label"), "labelZh": item.get("zh"),
            "category": item.get("category"), "note": item.get("note"), "score": item.get("sam_score"),
            "sourceFrameId": item.get("frame_id"), "imageId": None, "entityIds": [], "mappingStatus": "unassociated",
            "sourcePixelBox": item.get("box"), "sourceRefs": [{"assetId": did, "jsonPointer": f"/detections/{index}"}]})
    result["imageInterpretations"].append(interpretations)


def _attach_observations(root, source, bridge, result, document, manifest, include, cameras, camera_images):
    """Join registry detection pointers, source-image hashes, and explicit bridge IDs."""
    registry_raw = _read(_path(root, "object-evidence.json"), source["provenance"]["source_bridge"]["source_registry_sha256"])
    registry = json.loads(registry_raw)
    detection_raw = _path(root, "detection/detections.json").read_bytes()
    detection_sha = hashlib.sha256(detection_raw).hexdigest()
    detection = json.loads(detection_raw)
    run_id = registry["run_id"]
    if run_id != bridge["source_run_id"] or detection["run_id"] != run_id:
        raise PlatformError("import_report_run_mismatch", 422)
    registry_id = include(registry_raw, "application/json", {"kind": "report_observation_registry", "sourceRunId": run_id})
    detection_id = include(detection_raw, "application/json", {"kind": "report_image_interpretation", "sourceRunId": run_id})
    by_detection = {}
    for index, candidate in enumerate(registry["candidates"]):
        for ref in candidate.get("source_refs", []):
            if ref.get("type") != "detection":
                continue
            pointer = ref.get("pointer", [])
            if ref.get("path") != "detection/detections.json" or ref.get("sha256") != detection_sha or len(pointer) != 3 or pointer[0] != "detections" or pointer[2] != "rle" or type(pointer[1]) is not int or not 0 <= pointer[1] < len(detection["detections"]):
                raise PlatformError("import_report_detection_binding_invalid", 422)
            by_detection.setdefault(pointer[1], []).append((candidate, index))
    if not by_detection:
        raise PlatformError("import_report_detection_binding_missing", 422)
    frames = {}
    for frame in registry["frames"]:
        raw = _read(_path(root, frame["image_path"]), frame["source_sha256"])
        with Image.open(io.BytesIO(raw)) as image:
            width, height = image.size
            media_type = Image.MIME.get(image.format, "application/octet-stream")
        if (width, height) != (frame["width"], frame["height"]):
            raise PlatformError("import_report_image_dimensions_mismatch", 422)
        image_id = include(raw, media_type, {"kind": "report_source_image", "sourceRunId": run_id, "width": width, "height": height})
        frames[frame["frame_id"]] = {"runId": run_id, "sourceFrameId": frame["frame_id"], "imageId": image_id, "sourceImageSha256": frame["source_sha256"], "width": width, "height": height}
    bridge_records = {record["id"]: record for record in bridge["records"]}
    assets = {asset["id"]: asset for asset in document["assets"]}
    interpretations = {"runId": run_id, "detectionVersion": detection.get("detection_version"), "items": [], "frames": list(frames.values()),
        "missing": _camel(detection.get("missing", [])), "rejected": _camel(detection.get("rejected", [])),
        "recall": detection.get("recall"), "recallReason": detection.get("recall_reason"),
        "notice": "Saved single-sweep or explicitly unverified legacy detections; no ground-truth recall. Only verified source-image and registry-to-bridge IDs link to the current scene; source 3D is not registered.",
        "sourceRefs": [{"assetId": detection_id}, {"assetId": registry_id}]}
    for index, detection_item in enumerate(detection["detections"]):
        fid = detection_item.get("frame_id")
        frame = frames.get(fid)
        image_bound = frame is not None and detection_item.get("source_image_sha256") == frame["sourceImageSha256"]
        item = {"sourceRecordId": detection_item.get("instance_id") or f"{fid}:{detection_item.get('item_id')}:{index}",
            "label": detection_item.get("label"), "labelZh": detection_item.get("zh"), "category": detection_item.get("category"),
            "note": detection_item.get("note"), "score": detection_item.get("sam_score"), "sourceFrameId": fid,
            "semanticVerification": detection_item.get("semantic_verification"), "sourceBinding": detection_item.get("source_binding"),
            "sourceImageId": frame["imageId"] if image_bound else None, "sourcePixelBox": detection_item.get("box"),
            "sourceCandidateIds": [], "entityIds": [], "imageId": None, "cameraId": None, "originalPixelBox": None, "mappingStatus": "unassociated",
            "sourceRefs": [{"assetId": detection_id, "jsonPointer": f"/detections/{index}"}]}
        for candidate, registry_index in by_detection.get(index, []):
            if candidate["frame_id"] != fid:
                raise PlatformError("import_report_detection_frame_mismatch", 422)
            item["sourceCandidateIds"].append(candidate["id"])
            item["sourceRefs"].append({"assetId": registry_id, "jsonPointer": f"/candidates/{registry_index}"})
            record = bridge_records.get(candidate["id"])
            binding = bridge.get("mapped_source_frames", {}).get(fid)
            if not image_bound or not record or not binding:
                continue
            target = binding["target_frame_id"]
            if record["source_frame_id"] != fid or record["target_frame_id"] != target or binding["source_image_sha256"] != frame["sourceImageSha256"] or target not in camera_images or assets[camera_images[target]]["sha256"] != binding["target_image_sha256"]:
                raise PlatformError("import_report_bridge_binding_invalid", 422)
            entity_id = manifest["entityIds"].get(candidate["id"])
            if entity_id is None:
                continue
            if entity_id not in item["entityIds"]:
                item["entityIds"].append(entity_id)
            item.update(imageId=camera_images[target], cameraId=manifest["cameraIds"][target], targetSourceFrameId=target,
                        mappingStatus="registry_pointer_and_verified_photo_bridge", mappingLimitation=binding["assumption"],
                        originalPixelBox=original_box(detection_item["box"], binding["original_to_original"], cameras[target]["chosenWidth"], cameras[target]["chosenHeight"]))
        if not item["entityIds"]:
            item["mappingLimitation"] = "Source-image identity unverified" if not image_bound else "No verified same-capture target-frame/entity bridge"
        interpretations["items"].append(item)
    result["imageInterpretations"].append(interpretations)
