"""Prepare one immutable research input, or submit it through the admin DB boundary.

Protocol input: id, purpose, projectId, branchId, baselineRevision, entityId,
metricDefinitions, policyThresholds, split, callLimits (maxCalls=1,
maxCostPerCallUsd, maxTotalCostUsd), and optional observationId/seed.
An explicit geometry/depth stage instead selects owned capture imageIds and
requires no prior entities, masks or geometry. Depth selects exactly one photo.
Segmentation runtime validation selects an explicit owned entityId/observationId;
its original photo and saved box are used without requiring a previous mask.
Configuration and budget come from the existing platform runtime environment.
--prepare performs reads only; --submit enqueues the prepared envelope for the
ordinary worker. Neither command invokes a model or modifies a scene head.
"""
import argparse
import json
import os
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from ehs_spatial.platform.contracts import PlatformError, canonical, digest
from ehs_spatial.platform.reconstruction import (
    _Stages, _capture, _load_geometry, _load_masks, _packed, _research_stage, _research_capture_input, _research_segmentation_input,
    provider_snapshot_from_env, providers_from_manifest, validate_research_manifest,
)
from ehs_spatial.platform.research_authority import admin_context, check_budget, validate_prepared


def prepare(protocol, repository, blobs, provider_manifest, runtime_manifest):
    allowed = {"id", "purpose", "projectId", "branchId", "baselineRevision", "entityId", "observationId",
               "metricDefinitions", "policyThresholds", "split", "callLimits", "seed", "stage", "imageIds"}
    required = allowed - {"observationId", "seed", "stage", "imageIds", "entityId"}
    if not isinstance(protocol, dict) or not required <= set(protocol) or set(protocol) - allowed:
        raise PlatformError("frozen_research_protocol_required", 409)
    stage = protocol.get('stage', 'generation')
    if stage not in ('generation', 'geometry', 'depth', 'segmentation'):
        raise PlatformError('research_stage_unsupported', 409)
    if (stage == 'generation' and ('entityId' not in protocol or 'imageIds' in protocol)
            or stage in ('geometry', 'depth') and any(k in protocol for k in ('entityId', 'observationId', 'seed'))
            or stage == 'segmentation' and (not {'entityId', 'observationId'} <= set(protocol) or
                any(k in protocol for k in ('imageIds', 'seed')) or protocol['purpose'] != 'runtime_validation')):
        raise PlatformError('frozen_research_protocol_required', 409)
    protocol = json.loads(canonical(protocol))
    project_id, branch_id, base_id = (protocol[k] for k in ("projectId", "branchId", "baselineRevision"))
    with repository._connect() as connection:
        authority, source = admin_context(repository, connection, project_id, branch_id, base_id)
    base_sha = digest(source["document"])
    job = {"id": str(uuid5(NAMESPACE_URL, protocol["id"])), "kind": "validate_model", "projectId": project_id,
           "baseRevisionId": base_id, "inputs": {}, "config": {}}
    if stage in ('geometry', 'depth', 'segmentation'):
        if stage == 'segmentation':
            payload, images, snapshot, refs = _research_segmentation_input(repository, blobs, job, protocol['entityId'], protocol['observationId'])
            protocol['sourceObservation'] = snapshot
        else:
            payload, images = _research_capture_input(repository, blobs, job, stage, protocol.get('imageIds'))
            refs = sorted([{'assetId': i['assetId'], 'sha256': i['sha256']} for i in images], key=lambda r: r['assetId'])
        if stage not in provider_manifest or stage not in runtime_manifest:
            raise PlatformError('provider_not_configured', 409, stage=stage)
        manifest = {stage: provider_manifest[stage]}
        protocol.update(imageIds=[i['id'] for i in images], inputHashes=[i['sha256'] for i in images],
                        inputAssetHashes=refs,
                        payloadSha256=digest(payload), providerManifestSha256=digest(manifest),
                        runtimeManifest={stage: runtime_manifest[stage]})
        validate_research_manifest(protocol, manifest)
        providers_from_manifest(manifest, _research=True)[stage].validate(stage, research_protocol=protocol)
        with repository._connect() as connection:
            budget = check_budget(repository, connection, protocol)
        return {'schemaVersion': 1, 'projectId': project_id, 'branchId': branch_id, 'baseRevisionId': base_id,
                'baseDocumentSha256': base_sha, 'authority': authority, 'budgetAtPreparation': budget,
                'protocol': protocol, 'providerManifest': manifest, 'payload': payload, 'images': images}
    _, document, images = _capture(repository, blobs, job)
    entity = next((e for e in document["entities"] if e["id"] == protocol["entityId"] and not e.get("sourceContext")), None)
    if entity is None:
        raise PlatformError("entity_not_found", 404)
    observations = [o for o in document["observations"] if o["id"] in entity["observationRefs"] and o.get("maskAssetId")
                    and (not protocol.get("observationId") or o["id"] == protocol["observationId"])]
    if not observations:
        raise PlatformError("generation_mask_required", 409)
    anchor = max(observations, key=lambda o: ((o.get("geometrySupport") or {}).get("validPixelCount", 0), o["id"]))
    stages = _Stages(repository, blobs, job, {})
    frames, records = _load_geometry(document, images, stages)
    masks, errors = _load_masks(document, records, stages)
    if anchor["id"] not in masks:
        raise PlatformError("research_mask_unavailable", 409)
    image = next(i for i in images if i["id"] == anchor["imageId"])
    frame = frames[image["id"]]
    payload = {"entityId": entity["id"], "image": records[image["id"]]["rgb"], "mask": masks[anchor["id"]],
               "points": frame.points, "valid": frame.valid, "K": frame.K, "cameraToWorld": frame.camera_to_world,
               "coordinateFrameId": frame.coordinate_frame_id, "imageId": image["id"], "imageSha256": image["sha256"],
               "seed": protocol.get("seed", 0)}
    asset_ids = {image["assetId"], anchor["maskAssetId"], (anchor.get("maskEvidence") or {}).get("canonicalMaskAssetId"),
                 (document.get("geometryBindings", {}).get(image["id"]) or {}).get("geometrySolutionId")}
    historical = document.get("geometryEvidence") or {}
    asset_ids.add(historical.get("manifestAssetId"))
    for saved in historical.get("frames", []):
        if saved["assets"]["input"] == image["id"]:
            asset_ids.update(saved["assets"].values())
    assets = [a for a in document["assets"] if a["id"] in asset_ids]
    for reference in assets:
        asset = repository.get_asset(reference["id"])
        if asset["sha256"] != reference["sha256"]:
            raise PlatformError("research_input_hash_mismatch", 409)
        blobs.get(asset["storageKey"], asset["sha256"], asset["sizeBytes"])
    manifest = {"generation": provider_manifest["generation"]}
    protocol.update(observationId=anchor["id"], inputHashes=[image["sha256"]],
                    inputAssetHashes=[{"assetId": a["id"], "sha256": a["sha256"]} for a in sorted(assets, key=lambda a: a["id"])],
                    payloadSha256=digest(_packed(payload)), providerManifestSha256=digest(manifest),
                    runtimeManifest={"generation": runtime_manifest["generation"]})
    validate_research_manifest(protocol, manifest)
    providers_from_manifest(manifest, _research=True)["generation"].validate("generation", research_protocol=protocol)
    with repository._connect() as connection:
        budget = check_budget(repository, connection, protocol)
    return {"schemaVersion": 1, "projectId": project_id, "branchId": branch_id, "baseRevisionId": base_id,
            "baseDocumentSha256": base_sha, "authority": authority, "budgetAtPreparation": budget,
            "protocol": protocol, "providerManifest": manifest, "payload": _packed(payload),
            "images": [{k: image[k] for k in ("id", "assetId", "sha256", "pixelMapping") if k in image}]}


def submit(prepared, repository, blobs):
    from psycopg.types.json import Jsonb
    from ehs_spatial.platform.postgres import _wire
    repository.blobs = blobs
    frozen = prepared["validation"]
    if digest(frozen) != prepared["sha256"] or frozen.get("schemaVersion") != 1 or frozen.get("authority", {}).get("source") != "database_admin":
        raise PlatformError("research_input_hash_mismatch", 409)
    protocol = frozen["protocol"]
    validate_research_manifest(protocol, frozen["providerManifest"])
    stage = _research_stage(protocol, frozen['providerManifest'])
    if digest(frozen["payload"]) != protocol["payloadSha256"]:
        raise PlatformError("research_input_hash_mismatch", 409)
    provider = providers_from_manifest(frozen["providerManifest"], _research=True)[stage]
    provider.validate(stage, research_protocol=protocol)
    if provider.paid is not True or not 0 < provider.estimated_cost_usd <= protocol["callLimits"]["maxCostPerCallUsd"]:
        raise PlatformError("research_call_budget_invalid", 409)
    pid, branch_id, base_id = (frozen[k] for k in ("projectId", "branchId", "baseRevisionId"))
    model_namespace = ('recgen' if provider.pins.get('model') == 'TRI-ML/RecGen' else 'sam3d') if stage == 'generation' else stage
    job_id = str(uuid5(NAMESPACE_URL, model_namespace + "-validation:" + pid + ":" + protocol["id"]))
    with repository._connect() as connection:
        connection.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s,0))", (job_id,))
        authority, source = admin_context(repository, connection, pid, branch_id, base_id)
        if digest(source["document"]) != frozen["baseDocumentSha256"]:
            raise PlatformError("research_input_hash_mismatch", 409)
        previous = connection.execute("SELECT * FROM jobs WHERE id=%s", (job_id,)).fetchone()
        if previous:
            if previous["request_sha256"].strip() != prepared["sha256"]:
                raise PlatformError("research_idempotency_mismatch", 409)
            return _wire(previous)
        authority = validate_prepared(repository, connection, frozen, prepared['sha256'])
        metadata = blobs.put(canonical(frozen), "application/json")
        metadata.update(id=str(uuid5(NAMESPACE_URL, model_namespace + "-validation-input:" + pid + ":" + prepared["sha256"])),
                        metadata=({'kind': model_namespace + '_validation_input', 'scope': 'research_only'} if stage == 'generation'
                                  else {'kind': 'stage_validation_input', 'stage': stage, 'scope': 'research_only'}))
        asset = repository._register_asset(connection, pid, metadata)
        inputs = {"validationAssetId": str(asset["id"]), "validationSha256": metadata["sha256"]}
        config = {"researchProtocolSha256": digest(protocol), "providerManifest": frozen["providerManifest"], "submittedBy": authority}
        row = connection.execute("""INSERT INTO jobs(id,project_id,branch_id,base_revision_id,request_id,request_sha256,kind,inputs,config,status)
            VALUES(%s,%s,%s,%s,%s,%s,'validate_model',%s,%s,'pending_dispatch') RETURNING *""",
            (job_id, pid, branch_id, base_id, job_id, prepared["sha256"], Jsonb(inputs), Jsonb(config))).fetchone()
        return _wire(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--submit", type=Path, metavar="PREPARED_JSON")
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        from ehs_spatial.platform.runtime import services
        repository, blobs = services()
        if args.prepare:
            if not args.protocol:
                parser.error("--prepare requires --protocol")
            output = args.output or args.protocol.with_name(args.protocol.stem + ".prepared.json")
            runtime = json.loads(Path(os.environ["PANOPTES_MODEL_RUNTIME_MANIFEST"]).read_text())
            frozen = prepare(json.loads(args.protocol.read_text()), repository, blobs, provider_snapshot_from_env(), runtime)
            output.parent.mkdir(parents=True, exist_ok=True)
            descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(canonical({"validation": frozen, "sha256": digest(frozen)}) + b"\n")
            print(json.dumps({"status": "prepared", "sha256": digest(frozen), "newModelCalls": 0, "databaseWrites": 0}))
        else:
            job = submit(json.loads(args.submit.read_text()), repository, blobs)
            print(json.dumps({"jobId": job["id"], "status": job["status"]}))
        return 0
    except Exception as exc:
        # Database/provider exception text can contain private URLs or credentials.
        print(json.dumps({"status": "blocked", "error": exc.code if isinstance(exc, PlatformError) else type(exc).__name__}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
