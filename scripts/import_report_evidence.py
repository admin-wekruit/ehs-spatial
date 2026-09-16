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
from types import SimpleNamespace

import numpy as np
from PIL import Image

from ehs_spatial.platform.contracts import PlatformError
from ehs_spatial.platform.source_cad import legacy_inventory_sam_path


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


def build_source_cad_manifest(root, inventory_sha256, *, source_run_id):
    """Freeze the actual source-run SAM bytes independently of current candidates."""
    inventory = json.loads(_read(_path(root, "inventory/inventory.json"), inventory_sha256))
    files = []
    for relative in sorted({legacy_inventory_sam_path(row) for row in inventory.get("objects", []) if not row.get("refine_slug")}):
        if not (Path(root) / relative).is_file():
            continue
        raw = _read(_path(root, relative))
        files.append({"path": relative, "sha256": hashlib.sha256(raw).hexdigest(), "sizeBytes": len(raw)})
    return json.dumps({"schemaVersion": 1, "kind": "source_cad_segmentation_manifest", "sourceRunId": source_run_id,
        "inventorySha256": inventory_sha256, "files": files}, sort_keys=True, separators=(",", ":")).encode()


def observation_mask_sources(geometry_root, source):
    """Read explicit, hash-pinned segmentation records; never rasterize report polygons."""
    from scripts.import_geometry_evidence import pinned_manifest
    frozen, _ = pinned_manifest(geometry_root, source)
    evidence, records, dependencies = frozen.get("evidence", {}), [], []
    for kind in ("objects", "floor"):
        if not evidence.get(kind):
            continue
        path = _path(geometry_root, evidence[kind])
        expected = evidence.get(kind + "_sha256")
        if not expected:
            raise PlatformError("import_report_mask_source_unpinned", 422)
        raw = _read(path, expected)
        dependencies.append(path)
        value = json.loads(raw)
        rows = value["objects"] if kind == "objects" else [value]
        for index, row in enumerate(rows):
            for view_index, view in enumerate(row.get("views", [])):
                if view.get("observed_only") is not True:
                    raise PlatformError("import_report_mask_not_observed", 422)
                files = {}
                provenance = view.get("provenance", {})
                if provenance.get("source_path") and provenance.get("source_sha256") and type(provenance.get("source_instance")) is int:
                    original_source = Path(provenance["source_path"]).resolve()
                    _read(original_source, provenance["source_sha256"])
                    dependencies.append(original_source)
                for key in ("mask_path", "canonical_mask_path"):
                    file = _path(geometry_root, view[key])
                    pin = view.get("sha256", {}).get(file.name)
                    if not pin:
                        raise PlatformError("import_report_mask_source_unpinned", 422)
                    files[key] = (file, _read(file, pin))
                    dependencies.append(file)
                records.append({"kind": kind, "sourceRecordId": row.get("object_id"), "view": view,
                    "sourcePath": evidence[kind], "sourceRaw": raw,
                    "jsonPointer": (f"/objects/{index}" if kind == "objects" else "") + f"/views/{view_index}", "files": files})
    return frozen, records, sorted(set(dependencies))


def import_observation_masks(geometry_root, source, document, manifest, include):
    """Attach both original and canonical raster evidence to existing observations."""
    frozen, records, _ = observation_mask_sources(geometry_root, source)
    if not records:
        return {"attachedObservationIds": [], "sourceViewCount": 0, "newModelCalls": 0}
    frames = {row["frame_id"]: row for row in frozen["frames"]}
    cameras = {row["id"]: row for row in source["cameras"]}
    entities = {row["id"]: row for row in document["entities"]}
    observations = {row["id"]: row for row in document["observations"]}
    geometry = document["geometryEvidence"]
    prepared = []
    for record in records:
        view, files = record["view"], record["files"]
        fid = view["frame_id"]
        if fid not in frames or fid not in cameras or view.get("rgb_path") != frames[fid]["input"]:
            raise PlatformError("import_report_mask_image_mismatch", 422)
        camera, frame = cameras[fid], frames[fid]
        original_shape = [frame["height"], frame["width"]]
        canonical_shape = [camera["height"], camera["width"]]
        with Image.open(io.BytesIO(files["mask_path"][1])) as image:
            mask = np.asarray(image)
            if image.format != "PNG" or mask.ndim != 2 or list(mask.shape) != original_shape:
                raise PlatformError("import_report_mask_dimensions_mismatch", 422)
        canonical_mask = np.load(io.BytesIO(files["canonical_mask_path"][1]), allow_pickle=False)
        if list(canonical_mask.shape) != canonical_shape or canonical_mask.dtype.kind not in "bu" or not np.isin(canonical_mask, [0, 1]).all():
            raise PlatformError("import_report_canonical_mask_invalid", 422)
        source_ids = [record["sourceRecordId"]] if record["kind"] == "objects" else manifest.get("floorBinding", {}).get("sourceRecordIds", [])
        for source_id in source_ids:
            entity = entities.get(manifest["entityIds"].get(source_id))
            if entity is None:
                continue
            current_camera = next(c for c in document["cameras"] if c["id"] == manifest["cameraIds"][fid])
            candidates = [observations[oid] for oid in entity.get("observationRefs", []) if observations[oid]["imageId"] == current_camera["imageId"]]
            if len(candidates) > 1:
                raise PlatformError("import_report_mask_observation_ambiguous", 422)
            if not candidates:
                continue
            observation = candidates[0]
            if observation.get("maskAssetId"):
                continue
            prepared.append((record, observation, source_id, original_shape, canonical_shape, frame))
    # All source hashes and grids pass before any callback registers bytes.
    for record, observation, source_id, original_shape, canonical_shape, frame in prepared:
        source_asset = include(record["sourceRaw"], "application/json", {"kind": "segmentation_source", "sourceRunId": frozen["experiment"]}, "segmentation/" + record["sourcePath"])
        refs = [{"assetId": source_asset, "jsonPointer": record["jsonPointer"], "sourceRecordId": source_id,
                 "sourceFrameId": record["view"]["frame_id"], "imageSha256": frame["sha256"]}]
        saved = {}
        for key, grid, shape, media in (("mask_path", "original_pixels", original_shape, "image/png"),
                                      ("canonical_mask_path", "canonical_pixels", canonical_shape, "application/x-npy")):
            path, raw = record["files"][key]
            saved[key] = include(raw, media, {"kind": "source_mask", "resolution": grid, "shapeHw": shape,
                "sourceRecordId": source_id, "sourceFrameId": record["view"]["frame_id"], "sourceRefs": refs}, "segmentation/" + str(path.relative_to(Path(geometry_root).resolve())))
        observation["maskAssetId"] = saved["mask_path"]
        observation["maskEvidence"] = {"originalMaskAssetId": saved["mask_path"], "canonicalMaskAssetId": saved["canonical_mask_path"],
            "originalShape": original_shape, "canonicalShape": canonical_shape,
            "inputToCanonical": frame["input_to_canonical_pixel_centres"], "geometryManifestAssetId": geometry["manifestAssetId"], "sourceRefs": refs}
        observation["missingEvidence"] = [item for item in observation.get("missingEvidence", []) if item != "source_mask_not_packaged"]
    return {"attachedObservationIds": [row[1]["id"] for row in prepared], "sourceViewCount": len(records), "newModelCalls": 0}


def import_source_equivalences(document, source, source_asset_id, records, masks, include, *, read_asset=None, read_source=None, read_import_asset=None):
    """Package exact shared SAM instances, never infer identity from mask overlap."""
    from ehs_spatial.providers.sam3 import decode_coco_rle

    def walk(value, pointer=""):
        if isinstance(value, dict):
            yield pointer, value
            for key, child in value.items():
                yield from walk(child, pointer + "/" + key.replace("~", "~0").replace("/", "~1"))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                yield from walk(child, pointer + "/" + str(index))

    assets = {a["id"]: a for a in document["assets"]}
    observations = {o["id"]: o for o in document["observations"]}
    cameras = {ref["sourceCameraId"]: c for c in document["cameras"] for ref in c.get("sourceRefs", []) if ref.get("sourceCameraId")}
    source_owners = {}
    for entity in document["entities"]:
        for ref in entity.get("lineage", []):
            if ref.get("operation") == "offline_import" and ref.get("sourceRecordId"):
                source_owners.setdefault(ref["sourceRecordId"], []).append(entity)

    def observation(source_id, image_id):
        candidates = {oid for entity in source_owners.get(source_id, []) for oid in entity["observationRefs"] if observations[oid]["imageId"] == image_id}
        if len(candidates) != 1:
            return None
        return observations[next(iter(candidates))]

    # ponytail: pinned input inventories are small; scan provenance without
    # coupling identity to a run name, label, object ID or dictionary location.
    raw_records = [(pointer, value) for pointer, value in walk(source)
                   if value.get("candidate_id") and isinstance(value.get("source_mask"), dict)]
    pairs, unavailable, seen = [], [], set()
    for record in records:
        provenance = record["view"].get("provenance", {})
        instance = provenance.get("source_instance")
        checksum = provenance.get("source_sha256")
        if not checksum or type(instance) is not int or instance < 0 or not provenance.get("source_path"):
            continue
        fid = record["view"]["frame_id"]
        if fid not in cameras:
            continue
        image_id = cameras[fid]["imageId"]
        for pointer, raw_record in raw_records:
            reference = raw_record["source_mask"].get("ref", {})
            if (reference.get("encoding") != "rle" or reference.get("sha256") != checksum
                    or reference.get("pointer") != ["rle", instance]
                    or raw_record.get("source_frame_id") != provenance.get("source_frame")
                    or raw_record.get("target_frame_id") != fid):
                continue
            generated = observation(record["sourceRecordId"], image_id)
            raw_observation = observation(raw_record["candidate_id"], image_id)
            if generated is None or raw_observation is None:
                unavailable.append({"sourceRecordIds": [record["sourceRecordId"], raw_record["candidate_id"]], "sourceFrameId": fid, "reason": "existing_observation_missing_or_ambiguous"})
                continue
            ids = tuple(sorted((generated["id"], raw_observation["id"])))
            if len(set(ids)) != 2 or ids in seen:
                continue
            source_bytes = _read(Path(provenance["source_path"]), checksum)
            source_json = json.loads(source_bytes)
            if not isinstance(source_json.get("rle"), list) or instance >= len(source_json["rle"]):
                raise PlatformError("import_source_instance_invalid", 422)
            grids = [masks.get(oid) for oid in ids]
            if any(grid is None for grid in grids) or not np.array_equal(grids[0], grids[1]):
                unavailable.append({"observationIds": ids, "reason": "canonical_masks_differ_or_missing"})
                continue
            shape = list(grids[0].shape)
            if reference.get("shape_hw") != shape:
                unavailable.append({"observationIds": ids, "reason": "source_grid_not_canonical"})
                continue
            decoded = decode_coco_rle(source_json["rle"][instance], height=shape[0], width=shape[1]).astype(bool)
            if not np.array_equal(decoded, grids[0]):
                raise PlatformError("import_source_instance_mask_mismatch", 422)
            source_id = include(source_bytes, "application/json", {"kind": "identity_source_segmentation"})
            native_id = include(record["sourceRaw"], "application/json", {"kind": "segmentation_source"})
            pairs.append({"kind": "canonical_sam_rle", "observationRefs": [{"observationId": oid, "revision": observations[oid]["revision"]} for oid in ids],
                "imageId": image_id, "imageSha256": assets[image_id]["sha256"],
                "sourceRef": {"assetId": source_id, "sha256": checksum, "jsonPointer": f"/rle/{instance}"},
                "canonicalShape": shape, "canonicalMaskSha256": hashlib.sha256(np.ascontiguousarray(decoded, dtype=np.bool_).tobytes()).hexdigest(),
                "evidenceRefs": [{"role": "native_mask_provenance", "assetId": native_id, "sha256": hashlib.sha256(record["sourceRaw"]).hexdigest(), "jsonPointer": record["jsonPointer"] + "/provenance"},
                    {"role": "raw_source_reference", "assetId": source_asset_id, "sha256": assets[source_asset_id]["sha256"], "jsonPointer": pointer + "/source_mask/ref"}]})
            seen.add(ids)
    representation_pairs = []
    if read_asset is not None and document.get('geometryEvidence'):
        from ehs_spatial.platform.identity import verify_source_representation
        from ehs_spatial.platform.contracts import PlatformError
        source_records = {row['id']:(f'/{key}/{index}',row) for key in ('objects','observed_regions','unavailable_regions')
                          for index,row in enumerate(source.get(key,[])) if row.get('source')=='observed' and row.get('id')}
        geometry = document['geometryEvidence']
        def asset_ref(aid):
            return {'assetId':aid,'sha256':next(a['sha256'] for a in document['assets'] if a['id']==aid)}
        def source_ref(record_id):
            return {**asset_ref(source_asset_id),'jsonPointer':source_records[record_id][0]}
        def original_asset(spec):
            existing = next((a for a in document['assets'] if a.get('sha256')==spec['sha256']),None)
            if existing:
                return asset_ref(existing['id'])
            if read_import_asset is None:
                return None
            data = read_import_asset(spec)
            if hashlib.sha256(data).hexdigest()!=spec['sha256']:
                raise PlatformError('import_report_source_hash_mismatch',422)
            return asset_ref(include(data,'application/octet-stream',{'kind':'source_identity_geometry'}))
        def bound_reps(entity,record_id):
            return [r for r in entity.get('representations',[]) if r.get('kind')=='observed_surface' and any(
                ref.get('assetId')==source_asset_id and ref.get('sourceRecordId')==record_id for ref in r.get('sourceRefs',[]) if isinstance(ref,dict))]
        # ponytail: imported inventories are bounded; exact full-array comparison
        # avoids any spatial index or approximate identity inference.
        for imported in document['entities']:
            if imported.get('observationRefs') or imported.get('sourceContext'):
                continue
            for lineage in imported.get('lineage',[]):
                record_id = lineage.get('sourceRecordId')
                if lineage.get('operation')!='offline_import' or lineage.get('sourceAssetId')!=source_asset_id or record_id not in source_records:
                    continue
                _, mesh_record = source_records[record_id]
                if len(mesh_record.get('frame_ids',[]))!=1 or not mesh_record.get('mesh'):
                    continue
                fid = mesh_record['frame_ids'][0]
                camera = cameras.get(fid)
                if camera is None:
                    continue
                candidates = []
                for owner in document['entities']:
                    for oid in owner.get('observationRefs',[]):
                        observation = observations[oid]
                        if observation['imageId']!=camera['imageId'] or oid not in masks:
                            continue
                        for ref in observation.get('sourceRefs',[]):
                            observed_id = ref.get('sourceRecordId') if isinstance(ref,dict) and ref.get('assetId')==source_asset_id else None
                            if observed_id not in source_records:
                                continue
                            observed = source_records[observed_id][1]
                            face_count = observed['faces']['count'] if observed.get('faces') else observed.get('mesh',{}).get('index_count',-3)//3
                            if face_count != mesh_record['mesh']['index_count']//3:
                                continue
                            context = source_records.get(observed.get('context_id'),(None,observed))[1]
                            if not context.get('mesh'):
                                continue
                            geometry_ref = original_asset(context['mesh']['asset'])
                            face_ref = original_asset(observed['faces']['asset']) if observed.get('faces') else None
                            if geometry_ref is None or observed.get('faces') and face_ref is None:
                                continue
                            for sam in observed.get('provenance',{}).get('source_refs',[]):
                                if not isinstance(sam,dict) or sam.get('type')!='text-sam' or type(sam.get('instance')) is not int or sam['instance']<0 or sam.get('pointer')!=['rle',sam['instance']] or not sam.get('sha256'):
                                    continue
                                existing = next((a for a in document['assets'] if a.get('sha256')==sam['sha256']),None)
                                if existing:
                                    sam_id = existing['id']
                                elif read_source is not None and sam.get('path'):
                                    sam_raw = read_source(sam['path'],sam['sha256'])
                                    if hashlib.sha256(sam_raw).hexdigest()!=sam['sha256']:
                                        raise PlatformError('import_report_source_hash_mismatch',422)
                                    sam_id = include(sam_raw,'application/json',{'kind':'identity_source_segmentation'})
                                else:
                                    continue
                                for rep in bound_reps(imported,record_id):
                                    for original_rep in bound_reps(owner,observed_id):
                                        pair = {'kind':'same_source_indexed_mesh','entityId':imported['id'],'representationId':rep['id'],
                                            'representationAsset':asset_ref(rep['assetId']),'transformSnapshot':deepcopy(rep['transform']),
                                            'sourceRepresentationId':original_rep['id'],'sourceRepresentationAsset':asset_ref(original_rep['assetId']),
                                            'sourceGeometryAsset':geometry_ref,'sourceFaceAsset':face_ref,
                                            'sourceTransformSnapshot':deepcopy(original_rep['transform']),
                                            'observationRef':{'observationId':oid,'revision':observation['revision']},
                                            'imageId':observation['imageId'],'imageSha256':assets[observation['imageId']]['sha256'],
                                            'sourceRef':{**asset_ref(sam_id),'jsonPointer':f"/rle/{sam['instance']}"},
                                            'sourceRecordRef':source_ref(record_id),'observationRecordRef':source_ref(observed_id),
                                            'geometryBinding':{'geometrySolutionId':geometry['manifestAssetId'],'cameraId':camera['id']},
                                            'canonicalShape':list(masks[oid].shape),'canonicalMaskSha256':hashlib.sha256(np.ascontiguousarray(masks[oid],dtype=np.bool_).tobytes()).hexdigest(),
                                            'geometryRole':'floor' if sam.get('label')=='floor' else None,'evidenceRefs':[]}
                                        try:
                                            candidates.append(verify_source_representation(document,pair,masks,read_asset))
                                        except PlatformError as exc:
                                            unavailable.append({'representationId':rep['id'],'observationId':oid,'reason':exc.code})
                if len({p['observationRef']['observationId'] for p in candidates})==1:
                    unique = {p['representationId']:p for p in sorted(candidates,key=lambda p:p['sourceRepresentationId'],reverse=True)}
                    representation_pairs.extend(unique.values())
    pairs.extend(representation_pairs)
    if pairs:
        payload = json.dumps({"schemaVersion": 1, "kind": "same_source_observation_equivalences", "pairs": sorted(pairs, key=lambda p: json.dumps(p,sort_keys=True))}, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        identity = include(payload, "application/json", {"kind": "source_identity_evidence"})
        reference = {"assetId": identity, "sha256": hashlib.sha256(payload).hexdigest()}
        if reference not in document.setdefault('sourceIdentityEvidence',[]):
            document['sourceIdentityEvidence'].append(reference)
    return {"pairCount": len(pairs), "representationPairCount":len(representation_pairs), "unavailable": unavailable, "newModelCalls": 0}


def canonical_observation_masks(document, read_asset):
    """Read imported raster evidence through the production canonical-mask loader."""
    from ehs_spatial.platform.reconstruction import _load_masks
    assets = {a["id"]: a for a in document["assets"]}
    by_sha = {a["sha256"]: a for a in document["assets"]}
    def get_blob(key, checksum, size):
        raw = read_asset(by_sha[checksum]["id"])
        if key != "sha256/" + checksum or len(raw) != size or hashlib.sha256(raw).hexdigest() != checksum:
            raise PlatformError("import_report_source_hash_mismatch", 422)
        return raw
    stages = SimpleNamespace(repo=SimpleNamespace(get_asset=lambda aid: {**assets[aid], "storageKey": "sha256/" + assets[aid]["sha256"]}),
                             blobs=SimpleNamespace(get=get_blob))
    cameras = {c["id"]: c for c in document["cameras"]}
    geometry, canonical = document["geometryEvidence"], {}
    for frame in geometry["frames"]:
        camera = cameras[frame["cameraId"]]
        def array(name):
            asset = assets[frame["assets"][name]]
            return np.load(io.BytesIO(get_blob("sha256/" + asset["sha256"], asset["sha256"], asset["sizeBytes"])), allow_pickle=False)
        canonical[camera["imageId"]] = {"points": array("pts3d.npy"),
            "inputToCanonical": array("intrinsics.npy") @ np.linalg.inv(camera["K"]),
            "originalShape": [camera["height"], camera["width"]], "geometryManifestAssetId": geometry["manifestAssetId"]}
    return _load_masks(document, canonical, stages)


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
            inventory = json.loads(_path(old, "inventory/inventory.json").read_bytes())
            files.update(_path(old, relative) for relative in {legacy_inventory_sam_path(row) for row in inventory.get("objects", [])}
                         if (old / relative).is_file())
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
    scene_source_id = manifest["assetIds"][Path(scene_path).name]
    model_frames = {record['id']: record.get('frame_ids', []) for record in source.get('objects', [])}
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
                if fid in model_frames.get(source_record, []):
                    for representation in entity['representations']:
                        inputs = representation.get('sourceRefs', [])
                        if not any(ref.get('assetId') == scene_source_id and ref.get('sourceRecordId') == source_record for ref in inputs):
                            continue
                        # Only declared model input frames get a revision pin;
                        # other report views remain observation evidence.
                        if not any(ref.get('observationId') == observation['id'] for ref in inputs):
                            inputs.append({**deepcopy(binding), 'sha256': assets[source_id]['sha256'],
                                'observationId': observation['id'], 'revision': observation['revision'], 'imageId': observation['imageId']})
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
    inventory_raw = _read(_path(root, "inventory/inventory.json"), saved["source_sha256"]["legacy_inventory"])
    historical["inventoryAssetId"] = include(inventory_raw, "application/json", {"kind": "historical_inventory", "sourceRunId": historical["runId"]})
    source_manifest = build_source_cad_manifest(root, saved["source_sha256"]["legacy_inventory"], source_run_id=historical["runId"])
    historical["sourceCadManifestAssetId"] = include(source_manifest, "application/json", {"kind": "source_cad_segmentation_manifest", "sourceRunId": historical["runId"]})
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
