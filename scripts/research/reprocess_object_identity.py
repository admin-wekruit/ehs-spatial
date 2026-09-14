"""Reviewable admin batch for fixed revisions; ordinary jobs retain lease/CAS semantics.

Plan (files only): python -m scripts.research.reprocess_object_identity --manifest PREPARED --output PLAN
Execute reviewed plan: python -m scripts.research.reprocess_object_identity --execute PLAN --runtime-env PRIVATE_JSON --output AUDIT
No publication writes, capability reads, provider calls, or automatic database migrations.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from ehs_spatial.platform.contracts import PlatformError, canonical, digest, validate_document
from scripts.research.prepare_object_identity_inputs import checked
from scripts.research.scan_object_identity import encoded, file_ref


def identity(value):
    return str(uuid5(NAMESPACE_URL, "identity-batch:" + value))


def assert_source_conserved(source, result, *, preparing=False):
    """Association can change owners, never the observation's original evidence."""
    for key in ("revision", "imageId", "originalPixelBox", "originalPixelPolygons", "sourcePolygonsCanonical", "pixelMapping", "sourceRefs"):
        assert {o["id"]: o.get(key) for o in source["observations"]} == {o["id"]: o.get(key) for o in result["observations"]}, key
    old_masks = {o["id"]: o["maskAssetId"] for o in source["observations"] if o.get("maskAssetId")}
    assert all(next(o for o in result["observations"] if o["id"] == oid).get("maskAssetId") == aid for oid, aid in old_masks.items())
    assert {a["id"] for a in source["assets"]} <= {a["id"] for a in result["assets"]}
    assert source["cameras"] == result["cameras"]
    if preparing:
        assert source["entities"] == result["entities"]


def make_plan(manifest_path):
    manifest_ref = file_ref(Path(manifest_path).resolve())
    manifest = json.loads(checked(manifest_ref))
    scan = json.loads(checked(manifest["sourceScan"]))
    targets = {(t["projectId"], t["revisionId"]) for t in scan["evaluationRevisions"]}
    if targets != {(t["projectId"], t["revisionId"]) for t in manifest["evaluationRevisions"]}:
        raise ValueError("Prepared target coverage differs from frozen scan")
    assets = {a["id"]: a for a in manifest["assets"]}
    # ponytail: one linear pass per asset; a larger corpus can stream hash checks.
    for asset in assets.values():
        raw = checked(asset["file"])
        if len(raw) != asset["sizeBytes"] or asset["file"]["sha256"] != asset["sha256"]:
            raise ValueError("Prepared asset metadata mismatch")
    rows = []
    for target in manifest["evaluationRevisions"]:
        source = json.loads(checked(target["baseDocument"]))
        prepared = json.loads(checked(target["document"]))
        assert_source_conserved(source, prepared, preparing=True)
        validate_document(prepared)
        key = manifest_ref["sha256"] + ":" + target["projectId"] + ":" + target["revisionId"]
        capture = None
        if not prepared.get("captureId") and prepared["observations"]:
            images = [{"id": c["imageId"], "assetId": c["imageId"], "width": c["width"], "height": c["height"]} for c in prepared["cameras"]]
            if not images or not {o["imageId"] for o in prepared["observations"]} <= {i["id"] for i in images}:
                raise ValueError("Historical capture cannot cover source observations")
            capture = {"id": identity(key + ":capture"), "images": images, "target": prepared["target"],
                "task": {"schemaVersion": 1, "kind": "identity_evidence_recovery", "sourceRevisionId": target["revisionId"],
                    "sourceDocumentSha256": digest(source), "batchManifestSha256": manifest_ref["sha256"],
                    "imageSourceRefs": [{"assetId": i["id"], "sha256": assets[i["id"]]["sha256"]} for i in images]}}
        rows.append({**target, "sourceDocumentSha256": digest(source), "branchId": identity(key + ":branch"),
            "jobId": identity(key + ":job"), "requestId": identity(key + ":request"), "capture": capture,
            "comparability": "native_geometry_available" if prepared.get("geometryEvidence") else "not_comparable"})
    root = Path(__file__).resolve().parents[2]
    code = ["ehs_spatial/platform/contracts.py", "ehs_spatial/platform/identity.py", "ehs_spatial/platform/reconstruction.py", "ehs_spatial/platform/spatial.py",
        "scripts/import_report_evidence.py", "scripts/research/prepare_object_identity_inputs.py", "scripts/research/reprocess_object_identity.py"]
    return {"schemaVersion": 1, "mode": "admin_fixed_revision_batch", "manifest": manifest_ref,
        "codePins": [file_ref(root / name) for name in code], "projectIds": sorted({p for p, _ in targets}), "targets": rows,
        "publicationIds": sorted(p["id"] for p in scan["publications"]), "publicationWrites": 0, "newModelCalls": 0}


def remap_new_assets(value, mapping):
    if isinstance(value, dict):
        return {k: remap_new_assets(v, mapping) for k, v in value.items()}
    if isinstance(value, list):
        return [remap_new_assets(v, mapping) for v in value]
    return mapping.get(value, value) if isinstance(value, str) else value


def register_target(repository, blobs, plan, target, registry):
    """One scoped admin transaction creates branch, capture, prepared asset and queued job."""
    from psycopg.types.json import Jsonb
    from ehs_spatial.platform.postgres import _wire
    source = json.loads(checked(target["baseDocument"]))
    prepared = json.loads(checked(target["document"]))
    original_ids = {a["id"] for a in source["assets"]}
    project_id, base_id = target["projectId"], target["revisionId"]
    request_sha = digest({"manifestSha256": plan["manifest"]["sha256"], "target": target, "codePins": plan["codePins"]})
    with repository._connect() as connection:
        connection.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s,0))", (target["jobId"],))
        row = repository._revision(connection, project_id, base_id)
        if row["document_sha256"].strip() != target["sourceDocumentSha256"] or digest(row["document"]) != digest(source):
            raise PlatformError("identity_batch_source_changed", 409)
        previous = connection.execute("SELECT * FROM jobs WHERE id=%s", (target["jobId"],)).fetchone()
        if previous:
            if previous["request_sha256"].strip() != request_sha:
                raise PlatformError("identity_batch_idempotency_mismatch", 409)
            return _wire(previous)
        # The new branch points at the fixed immutable base, so finish_job owns CAS.
        connection.execute("""INSERT INTO scene_branches(id,project_id,kind,title,source_revision_id,head_revision_id,request_id,request_sha256)
            VALUES(%s,%s,'reconstruction',%s,%s,%s,%s,%s)""", (target["branchId"], project_id,
            "Object identity · " + base_id[:8], base_id, base_id, target["requestId"], request_sha))
        mapping, proof_metadata = {}, {}
        proof_ids = {ref["assetId"] for ref in prepared.get("sourceIdentityEvidence", [])}
        # Proofs reference the immutable source assets. Resolve those IDs first,
        # then freeze the proof bytes once with their actual registered IDs.
        for asset in sorted(prepared["assets"], key=lambda a: a["id"] in proof_ids):
            registered = registry[asset["id"]]
            raw = checked(registered["file"])
            if asset["id"] in original_ids:
                existing = repository._one(connection, "SELECT * FROM assets WHERE id=%s", (asset["id"],), code="asset_not_found")
                if existing["sha256"].strip() != registered["sha256"] or existing["size_bytes"] != len(raw):
                    raise PlatformError("identity_batch_asset_changed", 409)
                blobs.get(existing["storage_key"], registered["sha256"], len(raw))
                continue
            if asset["id"] in proof_ids:
                proof = json.loads(raw)
                if proof.get("kind") != "same_source_observation_equivalences" or proof.get("schemaVersion") != 1:
                    raise PlatformError("identity_batch_source_proof_invalid", 422)
                raw = canonical(remap_new_assets(proof, mapping))
            metadata = blobs.put(raw, registered["mediaType"])
            metadata.update(id=identity(project_id + ":asset:" + metadata["sha256"]), metadata=remap_new_assets(registered.get("metadata", {}), mapping))
            saved = repository._register_asset(connection, project_id, metadata)
            mapping[asset["id"]] = str(saved["id"])
            if asset["id"] in proof_ids:
                proof_metadata[asset["id"]] = {key: metadata[key] for key in ("sha256", "sizeBytes", "storageKey")}
        for asset in prepared["assets"]:
            if asset["id"] in proof_metadata:
                asset.update(proof_metadata[asset["id"]])
        for reference in prepared.get("sourceIdentityEvidence", []):
            if reference["assetId"] in proof_metadata:
                reference["sha256"] = proof_metadata[reference["assetId"]]["sha256"]
        prepared = remap_new_assets(prepared, mapping)
        prepared["assets"] = list({a["id"]: a for a in reversed(prepared["assets"])}.values())[::-1]
        if target["capture"]:
            capture = target["capture"]
            prepared["captureId"] = capture["id"]
            prepared["annotations"].append({"id": identity(target["jobId"] + ":capture-source"), "kind": "capture_source_binding",
                "captureId": capture["id"], "sourceRevisionId": base_id, "batchManifestSha256": plan["manifest"]["sha256"],
                "sourceRefs": capture["task"]["imageSourceRefs"]})
            connection.execute("""INSERT INTO captures(id,project_id,branch_id,base_revision_id,revision_id,request_id,request_sha256,target,images,task)
                VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""", (capture["id"], project_id, target["branchId"], base_id, base_id,
                target["requestId"], digest(capture), capture["target"], Jsonb(capture["images"]), Jsonb(capture["task"])))
        assert_source_conserved(source, prepared, preparing=True)
        validate_document(prepared)
        metadata = blobs.put(canonical(prepared), "application/json")
        metadata.update(id=identity(target["jobId"] + ":prepared"), metadata={"kind": "identity_prepared_document", "baseRevisionId": base_id,
            "baseDocumentSha256": digest(source), "batchManifestSha256": plan["manifest"]["sha256"]})
        document_asset = repository._register_asset(connection, project_id, metadata)
        inputs = {"preparedDocumentAssetId": str(document_asset["id"]), "preparedDocumentSha256": metadata["sha256"],
            "baseDocumentSha256": digest(source), "batchManifestSha256": plan["manifest"]["sha256"]}
        config = {"offline": True, "associationConfigVersion": "workcell-identity-v2", "providerManifest": {}, "codePins": plan["codePins"]}
        job = connection.execute("""INSERT INTO jobs(id,project_id,branch_id,base_revision_id,request_id,request_sha256,kind,inputs,config,status,executor_ref,dispatched_at)
            VALUES(%s,%s,%s,%s,%s,%s,'reassociate_scene',%s,%s,'queued',%s,now()) RETURNING *""", (target["jobId"], project_id,
            target["branchId"], base_id, target["requestId"], request_sha, Jsonb(inputs), Jsonb(config), "identity-batch:" + plan["manifest"]["sha256"]))
        return _wire(job.fetchone())


def execute(plan_path, repository, blobs, audit_path):
    from ehs_spatial.platform.reconstruction import run_reassociation
    plan_ref = file_ref(Path(plan_path).resolve())
    plan = json.loads(checked(plan_ref))
    for pin in plan["codePins"]:
        checked(pin)
    manifest = json.loads(checked(plan["manifest"]))
    if make_plan(plan["manifest"]["path"]) != plan:
        raise ValueError("Reviewed plan differs from frozen inputs")
    registry = {a["id"]: a for a in manifest["assets"]}
    audit = {"plan": plan_ref, "results": [], "publicationWrites": 0, "newModelCalls": 0}
    for target in plan["targets"]:
        job = register_target(repository, blobs, plan, target, registry)
        if job["status"] in ("queued", "pending_dispatch"):
            job = repository.claim_job(job["id"], lease_seconds=3600)
            try:
                document, result = run_reassociation(repository, blobs, job, {})
                source = json.loads(checked(target["baseDocument"]))
                assert_source_conserved(source, document)
                assert result["newModelCalls"] == 0
                result.update(batchManifestSha256=plan["manifest"]["sha256"], comparability=target["comparability"],
                    sourcePublicationIds=target["publicationIds"], catalogPublicationIds=target["catalogPublicationIds"])
                job = repository.finish_job(job["id"], job["attemptToken"], result["status"], document=document, result=result)
            except (PlatformError, ValueError, AssertionError) as exc:
                result = {"code": exc.code if isinstance(exc, PlatformError) else "identity_batch_validation_failed", "newModelCalls": 0}
                job = repository.finish_job(job["id"], job["attemptToken"], "failed", result=result)
        audit["results"].append({"projectId": target["projectId"], "baseRevisionId": target["revisionId"], "branchId": target["branchId"],
            "jobId": job["id"], "status": job["status"], "resultRevisionId": job.get("resultRevisionId"), "headAdvanced": job.get("headAdvanced"),
            "captureId": (target["capture"] or {}).get("id"), "comparability": target["comparability"], "result": job.get("result")})
        Path(audit_path).write_bytes(encoded(audit) + b"\n")
        if job["status"] not in ("succeeded", "incomplete") or job.get("lateResultSaved"):
            raise PlatformError("identity_batch_job_requires_review", 409, jobId=job["id"], status=job["status"])
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--manifest", type=Path)
    mode.add_argument("--execute", type=Path)
    parser.add_argument("--runtime-env", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.manifest:
        if args.output.exists():
            raise ValueError("Plan output already exists")
        plan = make_plan(args.manifest)
        args.output.write_bytes(encoded(plan) + b"\n")
        print(json.dumps({"mode": "plan_only", "projects": len(plan["projectIds"]), "targets": len(plan["targets"]), "databaseWrites": 0}))
    else:
        if args.runtime_env:
            if args.runtime_env.stat().st_mode & 0o077:
                raise ValueError("Private runtime configuration must have mode 0600")
            os.environ.update(json.loads(args.runtime_env.read_text()))
        from ehs_spatial.platform.runtime import services
        repository, blobs = services()
        execute(args.execute, repository, blobs, args.output)
