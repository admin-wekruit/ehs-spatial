"""Bind source drawing records by pinned segmentation evidence, never geometry guesses."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import re

import numpy as np

from argus.platform.contracts import PlatformError
from argus.platform.identity import observation_owners, resolve_entity_id
from argus.providers.sam3 import decode_coco_rle


def legacy_inventory_sam_path(row):
    """The legacy producer's file key; never an identity/label similarity score."""
    slug = re.sub(r"[^a-z0-9]+", "_", row.get("label", "")).strip("_")
    return f"inventory/sam/{row.get('frame')}__{slug}.json"


def refresh_source_cad_links(document, read_asset, masks, *, inventory_asset_id=None, source_manifest_asset_id=None):
    """Refresh current owners and an exhaustive source-record coverage ledger.

    This does not register source 3D coordinates, alter source CAD pixels, or
    change an entity's semantic/measurement/association state.
    """
    report = document.get("reportEvidence") or {}
    historical = report.get("historical") or {}
    cad = historical.get("cad")
    if not cad:
        return {"sourceRecordCount": 0, "linkedRecordCount": 0, "unresolvedRecordCount": 0, "records": []}
    assets = {a["id"]: a for a in document["assets"]}
    by_sha = {a["sha256"]: a for a in document["assets"]}
    owners = observation_owners(document)
    cache = {}

    def read(aid):
        if aid not in cache:
            asset = assets[aid]
            raw = read_asset(aid)
            if hashlib.sha256(raw).hexdigest() != asset["sha256"] or len(raw) != asset["sizeBytes"]:
                raise PlatformError("source_cad_asset_hash_mismatch", 422, assetId=aid)
            cache[aid] = json.loads(raw)
        return cache[aid]

    inventory_asset_id = inventory_asset_id or historical.get("inventoryAssetId")
    inventory = {}
    if inventory_asset_id:
        if inventory_asset_id not in assets:
            raise PlatformError("source_cad_inventory_asset_missing", 422)
        pins = {read(ref["assetId"]).get("inventory_sha256") for ref in cad.get("sourceRefs", []) if ref.get("assetId") in assets}
        if assets[inventory_asset_id]["sha256"] not in pins:
            raise PlatformError("source_cad_inventory_hash_mismatch", 422)
        rows = read(inventory_asset_id).get("objects", [])
        inventory = {row["inv"]: (index, row) for index, row in enumerate(rows)}
        if len(inventory) != len(rows):
            raise PlatformError("source_cad_inventory_duplicate_index", 422)
        historical["inventoryAssetId"] = inventory_asset_id

    source_manifest_asset_id = source_manifest_asset_id or historical.get("sourceCadManifestAssetId")
    source_files = {}
    if source_manifest_asset_id:
        if source_manifest_asset_id not in assets or not inventory_asset_id:
            raise PlatformError("source_cad_manifest_asset_missing", 422)
        manifest = read(source_manifest_asset_id)
        if (manifest.get("schemaVersion") != 1 or manifest.get("kind") != "source_cad_segmentation_manifest"
                or manifest.get("sourceRunId") != historical.get("runId")
                or manifest.get("inventorySha256") != assets[inventory_asset_id]["sha256"]):
            raise PlatformError("source_cad_manifest_mismatch", 422)
        files = manifest.get("files", [])
        if not isinstance(files, list):
            raise PlatformError("source_cad_manifest_invalid", 422)
        for index, item in enumerate(files):
            if (not isinstance(item, dict) or not isinstance(item.get("path"), str)
                    or not isinstance(item.get("sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"])
                    or type(item.get("sizeBytes")) is not int or item["sizeBytes"] <= 0 or item["path"] in source_files):
                raise PlatformError("source_cad_manifest_invalid", 422)
            source_files[item["path"]] = (index, item)
        historical["sourceCadManifestAssetId"] = source_manifest_asset_id

    def walk(value, pointer=""):
        if isinstance(value, dict):
            yield pointer, value
            for key, child in value.items():
                yield from walk(child, pointer + "/" + key.replace("~", "~0").replace("/", "~1"))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                yield from walk(child, pointer + "/" + str(index))

    # Only explicitly referenced source records may supply evidence for an observation.
    candidates, source_records = [], {}
    for obs in document["observations"]:
        for ref in obs.get("sourceRefs", []):
            aid, record_id = ref.get("assetId"), ref.get("sourceRecordId")
            if aid not in assets or not record_id or assets[aid].get("mediaType") != "application/json":
                continue
            if aid not in source_records:
                source_records[aid] = {}
                for pointer, source in walk(read(aid)):
                    if source.get("id") and isinstance(source.get("provenance"), dict) and "source_record" in source["provenance"]:
                        source_records[aid].setdefault(source["id"], []).append((pointer, source))
            for pointer, source in source_records[aid].get(record_id, []):
                provenance = source.get("provenance", {})
                record = provenance.get("source_record", {})
                mask_ref = (record.get("source_mask") or {}).get("ref", {})
                if source.get("id") == record_id and record.get("candidate_id") == record_id and mask_ref.get("encoding") == "rle":
                    candidates.append((obs, aid, pointer, provenance, record, mask_ref))

    frame_relations = (report.get("frameRelations") or {}).get("frames", [])
    inventory_rows = {row["inventoryIndex"]: row for row in historical.get("inventory", [])}
    ledger = []
    for index in sorted({region["inventoryIndex"] for region in cad.get("regions", [])}):
        row = inventory_rows.get(index, {})
        existing = set()
        prior = row.get("cadIdentityBinding") or {}
        # An exact-mask link is recomputed after mask edits; it must not become
        # an unconditional explicit binding merely because it existed last time.
        source_entity_ids = prior.get("sourceEntityIds", row.get("entityIds", []))
        for eid in source_entity_ids:
            try:
                existing.add(resolve_entity_id(document, eid))
            except PlatformError as exc:
                if exc.code not in ("identity_resolution_ambiguous", "entity_not_found"):
                    raise
        proofs = []
        reason = "source_inventory_not_packaged" if not inventory_asset_id else "no_verified_same_photo"
        pair = inventory.get(index)
        if pair:
            source_index, source = pair
            frame = source.get("frame")
            relations = [r for r in frame_relations if r.get("sourceFrameId") == frame]
            if relations:
                reason = "exact_segmentation_evidence_missing"
                path = legacy_inventory_sam_path(source)
                instance = source.get("instance")
                if source.get("refine_slug") or source.get("merged_instances") or source.get("normal_split") or type(instance) is not int:
                    reason = "source_instance_requires_review"
                elif not source_manifest_asset_id:
                    reason = "source_segmentation_manifest_missing"
                elif path not in source_files:
                    reason = "source_segmentation_not_pinned"
                else:
                    for obs, aid, pointer, provenance, record, mask_ref in candidates:
                        if record.get("source_frame_id") != frame or mask_ref.get("path") != path or mask_ref.get("pointer") != ["rle", instance]:
                            continue
                        file_index, source_file = source_files[path]
                        if mask_ref.get("sha256") != source_file["sha256"]:
                            reason = "source_segmentation_hash_mismatch"
                            continue
                        image_sha = assets.get(obs["imageId"], {}).get("sha256")
                        relations_for_photo = [r for r in relations if r.get("targetImageSha256") == image_sha and r.get("targetFrameId") == record.get("target_frame_id")]
                        mapping = record.get("capture_mapping") or {}
                        if (len(relations_for_photo) != 1 or provenance.get("source_image_sha256") != image_sha
                                or mapping.get("target_image_sha256") != image_sha
                                or mapping.get("source_image_sha256") != relations_for_photo[0].get("sourceImageSha256")):
                            reason = "source_photo_mapping_mismatch"
                            continue
                        # Exact raster identity is only valid on the same canonical grid.
                        matrix = np.asarray(mapping.get("source_canonical_to_target_canonical"), dtype=float)
                        expected_matrix = np.asarray(relations_for_photo[0].get("sourceCanonicalToTargetCanonical"), dtype=float)
                        if matrix.shape != (3, 3) or expected_matrix.shape != (3, 3) or not np.allclose(matrix, np.eye(3), atol=1e-12) or not np.allclose(matrix, expected_matrix, atol=1e-12):
                            reason = "canonical_grid_mapping_requires_review"
                            continue
                        mask = masks.get(obs["id"])
                        sam_asset = by_sha.get(mask_ref.get("sha256"))
                        if mask is None or sam_asset is None or sam_asset["sizeBytes"] != source_file["sizeBytes"] or list(mask.shape) != mask_ref.get("shape_hw"):
                            reason = "canonical_mask_or_source_missing"
                            continue
                        rles = read(sam_asset["id"]).get("rle", [])
                        if instance < 0 or instance >= len(rles):
                            raise PlatformError("source_cad_instance_invalid", 422)
                        decoded = decode_coco_rle(rles[instance], height=mask.shape[0], width=mask.shape[1]).astype(bool)
                        if not np.array_equal(mask, decoded):
                            reason = "canonical_masks_differ"
                            continue
                        if obs["id"] not in owners:
                            continue
                        proofs.append({"entityId": owners[obs["id"]], "observationId": obs["id"], "observationRevision": obs.get("revision", 1),
                            "imageId": obs["imageId"], "imageSha256": image_sha,
                            "canonicalMaskSha256": hashlib.sha256(np.ascontiguousarray(decoded).tobytes()).hexdigest(),
                            "sourceRefs": [{"assetId": inventory_asset_id, "jsonPointer": f"/objects/{source_index}"},
                                {"assetId": sam_asset["id"], "jsonPointer": f"/rle/{instance}"},
                                {"assetId": aid, "jsonPointer": pointer + "/provenance/source_record"},
                                {"assetId": source_manifest_asset_id, "jsonPointer": f"/files/{file_index}"}]})
        proof_reason = reason
        targets = existing | {p["entityId"] for p in proofs}
        if len(targets) > 1:
            linked, reason = [], "ambiguous_entity_ownership"
        elif targets:
            linked, reason = sorted(targets), "exact_source_mask" if proofs else "preserved_explicit_source_binding"
        else:
            linked = []
        binding = {"method": "source_cad_identity_v1", "status": "resolved" if linked else "requires_review", "reason": reason,
                   "proofStatus": "verified_original_source_mask" if proofs and linked else "explicit_binding_only" if linked else "unresolved",
                   "proofReason": "exact_source_mask" if proofs else proof_reason,
                   "sourceEntityIds": source_entity_ids, "entityIds": linked, "evidence": proofs}
        if row:
            row["entityIds"] = linked
            row["cadIdentityBinding"] = deepcopy(binding)
            if proofs:
                row["mappingStatus"] = "exact_source_mask" if linked else "requires_review"
        for region in cad.get("regions", []):
            if region["inventoryIndex"] == index:
                region["entityIds"] = linked
                region["cadIdentityBinding"] = deepcopy(binding)
        ledger.append({"inventoryIndex": index, "sourceFrameId": row.get("sourceFrameId"), "label": row.get("label"), **binding})
    coverage = {"method": "source_cad_identity_v1", "sourceRecordCount": len(ledger),
        "linkedRecordCount": sum(bool(r["entityIds"]) for r in ledger),
        "unresolvedRecordCount": sum(not r["entityIds"] for r in ledger),
        "geometryRegistration": "not_registered", "geometryConstraintsApplied": False, "records": ledger}
    cad["coverage"] = coverage
    return coverage
