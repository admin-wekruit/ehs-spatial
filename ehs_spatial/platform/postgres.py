"""PostgreSQL business transactions. Every mutation has one authority boundary."""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime
from decimal import Decimal
from pathlib import Path
import hmac
from typing import Any
from uuid import UUID, uuid4

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from .contracts import Job, PlatformError, capability_sha, digest, empty_document, validate_document
from .repository import apply_operations


def _wire(row):
    if row is None:
        return None
    if isinstance(row, dict):
        json_fields = {"document", "operations", "inverse_operations", "images", "task", "inputs", "config", "metadata", "result", "late_results", "response", "request", "snapshot"}
        return {key.split("_")[0] + "".join(x.title() for x in key.split("_")[1:]): deepcopy(value) if key in json_fields else _wire(value)
                for key, value in row.items() if key not in ("capability_sha256", "request_sha256")}
    if isinstance(row, (list, tuple)):
        return [_wire(value) for value in row]
    if isinstance(row, UUID):
        return str(row)
    if isinstance(row, datetime):
        return row.isoformat()
    if isinstance(row, Decimal):
        return float(row)
    return row


def _job_asset_references(result):
    """Declared exporter/pipeline outputs; error params are not output assets."""
    if not result:
        return
    assets, stages = result.get("assets", []), result.get("stages", [])
    if not isinstance(assets, list) or any(not isinstance(ref, dict) or "id" not in ref for ref in assets) or not isinstance(stages, list) or any(not isinstance(stage, dict) for stage in stages):
        raise PlatformError("publication_job_asset_invalid", 422)
    yield from ((ref["id"], ref) for ref in assets)
    yield from ((stage["assetId"], stage) for stage in stages if stage.get("assetId") is not None)
    for key in ("checkpointAssetId", "protocolAssetId", "outputAssetId"):
        if result.get(key) is not None:
            yield result[key], {}


class PostgresRepository:
    def __init__(self, dsn: str, *, paid_budget: Decimal | float | str | None = None, schema: str | None = None, blob_store=None, execution_config=None):
        self.dsn = dsn
        self.paid_budget = Decimal(str(paid_budget)) if paid_budget is not None else None
        if self.paid_budget is not None and (not self.paid_budget.is_finite() or self.paid_budget < 0):
            raise ValueError("paid_budget must be nonnegative and finite")
        self.schema = schema
        self.blobs = blob_store
        self.execution_config = deepcopy(execution_config or {"providerManifest": {}})

    def _job_config(self, client_config):
        return {**deepcopy(client_config), **deepcopy(self.execution_config)}

    @contextmanager
    def _connect(self):
        with psycopg.connect(self.dsn, row_factory=dict_row) as connection:
            if self.schema:
                connection.execute(psycopg.sql.SQL("SET search_path TO {}").format(psycopg.sql.Identifier(self.schema)))
            yield connection

    def migrate(self):
        with self._connect() as connection:
            connection.execute("SELECT pg_advisory_xact_lock(728611936)")
            for path in sorted((Path(__file__).parent / "migrations").glob("*.sql")):
                connection.execute(path.read_text())

    @staticmethod
    def _one(connection, query, params=(), *, code="not_found"):
        row = connection.execute(query, params).fetchone()
        if row is None:
            raise PlatformError(code, 404)
        return row

    def _auth(self, connection, project_id, capability):
        supplied = capability_sha(capability)
        row = self._one(connection, "SELECT * FROM projects WHERE id=%s", (project_id,), code="project_not_found")
        if not hmac.compare_digest(row["capability_sha256"].strip(), supplied):
            raise PlatformError("capability_forbidden", 403)
        return row

    def authorize(self, project_id, capability):
        with self._connect() as connection:
            return _wire(self._auth(connection, project_id, capability))

    @staticmethod
    def _idempotent(connection, table, project_id, body):
        allowed = {"scene_branches", "captures", "edit_batches", "jobs", "publications", "agent_turns"}
        if table not in allowed:
            raise ValueError("Invalid idempotency table")
        # Serialize only the business request, including simultaneous lost-response retries.
        connection.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s,0))", (str(project_id) + ":" + str(body["requestId"]),))
        row = connection.execute(f"SELECT * FROM {table} WHERE project_id=%s AND request_id=%s", (project_id, body["requestId"])).fetchone()
        if row and row["request_sha256"].strip() != digest(body):
            raise PlatformError("idempotency_mismatch", 409)
        return row

    def _branch(self, connection, project_id, branch_id, base_revision_id=None):
        branch = self._one(connection, "SELECT * FROM scene_branches WHERE project_id=%s AND id=%s FOR UPDATE", (project_id, branch_id), code="branch_not_found")
        if base_revision_id is not None and str(branch["head_revision_id"]) != str(base_revision_id):
            raise PlatformError("revision_conflict", 409, baseRevisionId=str(base_revision_id), currentRevisionId=str(branch["head_revision_id"]))
        return branch

    def _revision(self, connection, project_id, revision_id):
        return self._one(connection, "SELECT * FROM scene_revisions WHERE project_id=%s AND id=%s", (project_id, revision_id), code="revision_not_found")

    def _insert_revision(self, connection, project_id, branch_id, document, *, parent=None, source=None, label=None, revision_id=None):
        validate_document(document)
        identities = [item["id"] for item in document["assets"]]
        if identities:
            if self.blobs is None:
                raise PlatformError("blob_store_not_configured", 503)
            try:
                for identity in identities:
                    UUID(identity)
            except (ValueError, TypeError):
                raise PlatformError("invalid_asset_id", 422) from None
            project = self._one(connection, "SELECT fork_source_revision_id FROM projects WHERE id=%s", (project_id,))
            inherited = set()
            if project["fork_source_revision_id"]:
                source_document = self._one(connection, "SELECT document FROM scene_revisions WHERE id=%s", (project["fork_source_revision_id"],))["document"]
                inherited = {item["id"] for item in source_document["assets"]}
            assets = connection.execute("SELECT * FROM assets WHERE id=ANY(%s::uuid[])", (identities,)).fetchall()
            if len(assets) != len(identities):
                raise PlatformError("revision_asset_not_found", 422)
            for asset in assets:
                if str(asset["project_id"]) != str(project_id) and str(asset["id"]) not in inherited:
                    raise PlatformError("revision_asset_forbidden", 403)
                self.blobs.get(asset["storage_key"], asset["sha256"].strip(), asset["size_bytes"])
        identity = revision_id or uuid4()
        return connection.execute("""INSERT INTO scene_revisions(id,project_id,branch_id,parent_revision_id,source_revision_id,document,document_sha256,label)
            VALUES(%s,%s,%s,%s,%s,%s,%s,%s) RETURNING *""", (identity, project_id, branch_id, parent, source, Jsonb(document), digest(document), label)).fetchone()

    def create_project(self, capability, body, *, source_revision_id=None):
        cap_sha = capability_sha(capability)
        with self._connect() as connection:
            connection.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s,0))", (cap_sha,))
            previous = connection.execute("SELECT * FROM projects WHERE capability_sha256=%s", (cap_sha,)).fetchone()
            if previous:
                if str(previous["request_id"]) != str(body["requestId"]) or previous["request_sha256"].strip() != digest(body):
                    raise PlatformError("idempotency_mismatch", 409)
                return self._project_response(connection, previous)
            source = None
            if source_revision_id:
                source = self._one(connection, "SELECT * FROM scene_revisions WHERE id=%s", (source_revision_id,), code="revision_not_found")
            document = deepcopy(source["document"]) if source else empty_document()
            if not source:
                document["target"] = body.get("target", "scene")
            if body.get("operations"):
                document, _ = apply_operations(document, body["operations"], base_revision_id=str(source["id"]) if source else None)
            project_id, branch_id, revision_id = uuid4(), uuid4(), uuid4()
            project = connection.execute("""INSERT INTO projects(id,title,capability_sha256,request_id,request_sha256,default_branch_id,fork_source_revision_id)
                VALUES(%s,%s,%s,%s,%s,%s,%s) RETURNING *""", (project_id, body["title"], cap_sha, body["requestId"], digest(body), branch_id, source_revision_id)).fetchone()
            connection.execute("""INSERT INTO scene_branches(id,project_id,kind,title,head_revision_id) VALUES(%s,%s,'reconstruction',%s,%s)""", (branch_id, project_id, "Reconstruction", revision_id))
            self._insert_revision(connection, project_id, branch_id, document, source=source_revision_id, label="Fork" if source else "Initial", revision_id=revision_id)
            return self._project_response(connection, project)

    def fork_project(self, source_project_id, capability, body):
        with self._connect() as connection:
            self._revision(connection, source_project_id, body["sourceRevisionId"])
        return self.create_project(capability, body, source_revision_id=body["sourceRevisionId"])

    def _project_response(self, connection, project):
        branches = connection.execute("SELECT * FROM scene_branches WHERE project_id=%s ORDER BY created_at,id", (project["id"],)).fetchall()
        branch = next(row for row in branches if row["id"] == project["default_branch_id"])
        return _wire({"project": project, "branch": branch, "revision": self._revision(connection, project["id"], branch["head_revision_id"]), "branches": branches})

    def get_project(self, project_id):
        with self._connect() as connection:
            return self._project_response(connection, self._one(connection, "SELECT * FROM projects WHERE id=%s", (project_id,), code="project_not_found"))

    def list_projects(self):
        with self._connect() as connection:
            return {"items": _wire(connection.execute("SELECT * FROM projects ORDER BY created_at DESC,id LIMIT 200").fetchall())}

    def get_revision(self, revision_id):
        with self._connect() as connection:
            return _wire(self._one(connection, "SELECT * FROM scene_revisions WHERE id=%s", (revision_id,), code="revision_not_found"))

    def list_project_records(self, project_id, kind):
        table = {"revisions": "scene_revisions", "captures": "captures", "assets": "assets", "jobs": "jobs", "agent-turns": "agent_turns", "edits": "edit_batches"}.get(kind)
        if not table:
            raise ValueError("Unsupported record type")
        with self._connect() as connection:
            self._one(connection, "SELECT id FROM projects WHERE id=%s", (project_id,), code="project_not_found")
            return {"items": _wire(connection.execute(f"SELECT * FROM {table} WHERE project_id=%s ORDER BY created_at DESC,id LIMIT 500", (project_id,)).fetchall())}

    def create_branch(self, project_id, capability, body):
        with self._connect() as connection:
            self._auth(connection, project_id, capability)
            previous = self._idempotent(connection, "scene_branches", project_id, body)
            if previous:
                return _wire(previous)
            source = self._revision(connection, project_id, body["sourceRevisionId"])
            branch_id, revision_id = uuid4(), uuid4()
            branch = connection.execute("""INSERT INTO scene_branches(id,project_id,kind,title,source_revision_id,head_revision_id,request_id,request_sha256)
                VALUES(%s,%s,%s,%s,%s,%s,%s,%s) RETURNING *""", (branch_id, project_id, body.get("kind", "planning"), body["title"], source["id"], revision_id, body["requestId"], digest(body))).fetchone()
            self._insert_revision(connection, project_id, branch_id, source["document"], parent=source["id"], source=source["id"], label=body["title"], revision_id=revision_id)
            return _wire(branch)

    def commit_edits(self, project_id, capability, body):
        with self._connect() as connection:
            self._auth(connection, project_id, capability)
            previous = self._idempotent(connection, "edit_batches", project_id, body)
            if previous:
                return _wire({"revision": self._revision(connection, project_id, previous["revision_id"]), "edit_batch": previous, "head_advanced": True})
            self._branch(connection, project_id, body["branchId"], body["baseRevisionId"])
            base = self._revision(connection, project_id, body["baseRevisionId"])
            document, operations = None, body.get("operations", [])
            if body.get("agentTurnId"):
                turn = self._one(connection, "SELECT * FROM agent_turns WHERE project_id=%s AND id=%s", (project_id, body["agentTurnId"]), code="agent_turn_not_found")
                proposal = turn["response"] or {}
                if turn["status"] != "succeeded" or proposal.get("kind") != "proposal" or str(turn["branch_id"]) != str(body["branchId"]) or str(turn["base_revision_id"]) != str(base["id"]) or proposal.get("operations") != operations:
                    raise PlatformError("agent_proposal_mismatch", 409)
            if body.get("undoOf") or body.get("redoOf"):
                batch_id = body.get("undoOf") or body["redoOf"]
                batch = self._one(connection, "SELECT * FROM edit_batches WHERE project_id=%s AND branch_id=%s AND id=%s", (project_id, body["branchId"], batch_id), code="edit_batch_not_found")
                if body.get("undoOf"):
                    if str(batch["revision_id"]) != str(base["id"]):
                        raise PlatformError("undo_conflict", 409)
                    document = deepcopy(batch["inverse_operations"][0]["document"])
                else:
                    undo = connection.execute("SELECT * FROM edit_batches WHERE project_id=%s AND revision_id=%s AND undo_of=%s", (project_id, base["id"], batch_id)).fetchone()
                    if undo is None:
                        raise PlatformError("redo_conflict", 409)
                    document = deepcopy(self._revision(connection, project_id, batch["revision_id"])["document"])
                operations = [{"type": "undo" if body.get("undoOf") else "redo", "editBatchId": str(batch_id)}]
                inverse = [{"type": "restoreDocument", "document": base["document"]}]
            else:
                document, inverse = apply_operations(base["document"], operations, base_revision_id=str(base["id"]))
            revision = self._insert_revision(connection, project_id, body["branchId"], document, parent=base["id"], label=body.get("label"))
            batch = connection.execute("""INSERT INTO edit_batches(id,project_id,branch_id,base_revision_id,revision_id,request_id,request_sha256,operations,inverse_operations,undo_of,redo_of,agent_turn_id)
                VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING *""", (uuid4(), project_id, body["branchId"], base["id"], revision["id"], body["requestId"], digest(body), Jsonb(operations), Jsonb(inverse), body.get("undoOf"), body.get("redoOf"), body.get("agentTurnId"))).fetchone()
            connection.execute("UPDATE scene_branches SET head_revision_id=%s WHERE id=%s AND project_id=%s", (revision["id"], body["branchId"], project_id))
            return _wire({"revision": revision, "edit_batch": batch, "head_advanced": True})

    def register_asset(self, project_id, metadata, job_id=None):
        required = ("storageKey", "sha256", "sizeBytes", "mediaType")
        if any(key not in metadata for key in required) or len(metadata["sha256"]) != 64 or metadata["storageKey"] != "sha256/" + metadata["sha256"] or metadata["sizeBytes"] < 0:
            raise PlatformError("invalid_asset")
        with self._connect() as connection:
            return _wire(self._register_asset(connection, project_id, metadata, job_id))

    def _register_asset(self, connection, project_id, metadata, job_id=None):
        if self.blobs is None:
            raise PlatformError("blob_store_not_configured", 503)
        self.blobs.get(metadata["storageKey"], metadata["sha256"], metadata["sizeBytes"])
        previous = connection.execute("SELECT * FROM assets WHERE project_id=%s AND storage_key=%s", (project_id, metadata["storageKey"])).fetchone()
        if previous:
            if previous["sha256"].strip() != metadata["sha256"] or previous["size_bytes"] != metadata["sizeBytes"]:
                raise PlatformError("asset_integrity_conflict", 409)
            return previous
        return connection.execute("""INSERT INTO assets(id,project_id,job_id,storage_key,sha256,size_bytes,media_type,metadata)
            VALUES(%s,%s,%s,%s,%s,%s,%s,%s) RETURNING *""", (metadata.get("id", str(uuid4())), project_id, job_id, metadata["storageKey"], metadata["sha256"], metadata["sizeBytes"], metadata["mediaType"], Jsonb(metadata.get("metadata", {})))).fetchone()

    def get_asset(self, asset_id):
        with self._connect() as connection:
            return _wire(self._one(connection, "SELECT * FROM assets WHERE id=%s", (asset_id,), code="asset_not_found"))

    def create_capture(self, project_id, capability, body, images):
        mode = body.get("captureMode", "initial")
        if mode not in ("initial", "append"):
            raise PlatformError("invalid_capture_mode", 422)
        if not isinstance(images, list) or not 1 <= len(images) <= 4:
            raise PlatformError("image_count_out_of_range", 422, minimum=1, maximum=4)
        request = {**body, "images": [{k: value for k, value in image.items() if k != "id"} for image in images]}
        with self._connect() as connection:
            self._auth(connection, project_id, capability)
            previous = self._idempotent(connection, "captures", project_id, request)
            if previous:
                job = self._one(connection, "SELECT * FROM jobs WHERE project_id=%s AND request_id=%s", (project_id, body["requestId"]))
                return _wire({"capture": previous, "revision": self._revision(connection, project_id, previous["revision_id"]), "job": job})
            self._branch(connection, project_id, body["branchId"], body["baseRevisionId"])
            base = self._revision(connection, project_id, body["baseRevisionId"])
            source = base["document"]
            if mode == "initial" and (source.get("captureId") or source.get("captureIds") or any(source[key] for key in ("entities", "observations", "coordinateFrames", "cameras", "assets", "annotations"))):
                raise PlatformError("capture_append_required", 409)
            if mode == "append" and body["target"] != source["target"]:
                raise PlatformError("capture_target_mismatch", 422)
            identity, revision_id = uuid4(), uuid4()
            asset_rows = [self._register_asset(connection, project_id, image) for image in images]
            existing_images = {a["id"] for a in source["assets"] if a.get("kind") == "source_image"}
            capture_ids = source.get("captureIds") or ([source["captureId"]] if source.get("captureId") else [])
            old_captures = connection.execute("SELECT images FROM captures WHERE project_id=%s AND id=ANY(%s::uuid[])", (project_id, capture_ids)).fetchall() if capture_ids else []
            prior_images = [image for capture in old_captures for image in capture["images"]]
            def image_mapping(image):
                return {key: image.get(key) for key in ("width", "height", "originalWidth", "originalHeight", "pixelMapping", "exifOrientation")}
            image_records = []
            for asset, image in zip(asset_rows, images):
                record = {**image.get("metadata", {}), "id": str(asset["id"]), "assetId": str(asset["id"])}
                duplicate = next((r for r in image_records if r["id"] == record["id"]), None)
                prior = [r for r in prior_images if r["assetId"] == record["assetId"]]
                if duplicate is not None and image_mapping(duplicate) != image_mapping(record) or record["id"] in existing_images and not any(image_mapping(old) == image_mapping(record) for old in prior):
                    raise PlatformError("capture_image_mapping_conflict", 422, imageId=record["id"])
                if duplicate is None:
                    record["sourceReused"] = record["id"] in existing_images
                    image_records.append(record)
            if mode == "append":
                from .identity import migrate_document
                document = migrate_document(source, base_revision_id=str(base["id"])) if source["schemaVersion"] == 1 else deepcopy(source)
                document["captureIds"].append(str(identity))
            else:
                document = empty_document()
            document.update(captureId=str(identity), target=body["target"])
            asset_ids = {a["id"] for a in document["assets"]}
            for asset in asset_rows:
                aid = str(asset["id"])
                if aid not in asset_ids:
                    document["assets"].append({"id": aid, "kind": "source_image", "mediaType": asset["media_type"], "sha256": asset["sha256"].strip(), "sizeBytes": asset["size_bytes"]})
                    asset_ids.add(aid)
                    if document["schemaVersion"] == 2:
                        document["geometryBindings"][aid] = None
            new_images = [image["id"] for image in image_records if not image["sourceReused"]]
            reused_images = [image["id"] for image in image_records if image["sourceReused"]]
            task = {"schemaVersion": 1, "kind": "capture_reconstruction", "target": body["target"], "imageIds": [image["id"] for image in image_records],
                    "captureId": str(identity), "captureMode": mode, "newImageIds": new_images, "reusedImageIds": reused_images}
            capture = connection.execute("""INSERT INTO captures(id,project_id,branch_id,base_revision_id,revision_id,request_id,request_sha256,target,images,task)
                VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING *""", (identity, project_id, body["branchId"], base["id"], revision_id, body["requestId"], digest(request), body["target"], Jsonb(image_records), Jsonb(task))).fetchone()
            revision = self._insert_revision(connection, project_id, body["branchId"], document, parent=base["id"], label="Capture", revision_id=revision_id)
            connection.execute("UPDATE scene_branches SET head_revision_id=%s WHERE id=%s", (revision_id, body["branchId"]))
            job_body = {"requestId": body["requestId"], "branchId": body["branchId"], "baseRevisionId": str(revision_id),
                        "kind": "analyze_capture", "inputs": {"captureId": str(identity), "captureMode": mode, "newImageIds": new_images, "reusedImageIds": reused_images}, "config": {}}
            job = connection.execute("""INSERT INTO jobs(id,project_id,branch_id,base_revision_id,request_id,request_sha256,kind,inputs,config,status)
                VALUES(%s,%s,%s,%s,%s,%s,'analyze_capture',%s,%s,'pending_dispatch') RETURNING *""",
                (uuid4(), project_id, body["branchId"], revision_id, body["requestId"], digest(job_body), Jsonb(job_body["inputs"]), Jsonb(self._job_config({})))).fetchone()
            return _wire({"capture": capture, "revision": revision, "job": job})

    def create_job(self, project_id, capability, body):
        if body["kind"] == "validate_model":
            raise PlatformError("admin_job_required", 403)
        with self._connect() as connection:
            self._auth(connection, project_id, capability)
            previous = self._idempotent(connection, "jobs", project_id, body)
            if previous:
                return _wire(previous)
            self._branch(connection, project_id, body["branchId"], body["baseRevisionId"])
            self._revision(connection, project_id, body["baseRevisionId"])
            for asset_id in body.get("inputs", {}).get("assetIds", []):
                self._one(connection, "SELECT id FROM assets WHERE project_id=%s AND id=%s", (project_id, asset_id), code="asset_not_found")
            return _wire(connection.execute("""INSERT INTO jobs(id,project_id,branch_id,base_revision_id,request_id,request_sha256,kind,inputs,config,status)
                VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,'pending_dispatch') RETURNING *""", (uuid4(), project_id, body["branchId"], body["baseRevisionId"], body["requestId"], digest(body), body["kind"], Jsonb(body.get("inputs", {})), Jsonb(self._job_config(body.get("config", {}))))).fetchone())

    def get_job(self, job_id):
        with self._connect() as connection:
            return _wire(self._one(connection, "SELECT * FROM jobs WHERE id=%s", (job_id,), code="job_not_found"))

    def pending_jobs(self, limit=100):
        with self._connect() as connection:
            return _wire(connection.execute("SELECT * FROM jobs WHERE status='pending_dispatch' AND kind NOT IN ('agent_turn','import_scene') AND NOT cancel_requested ORDER BY created_at LIMIT %s", (min(limit, 500),)).fetchall())

    def mark_dispatched(self, job_id, executor_ref):
        with self._connect() as connection:
            connection.execute("UPDATE jobs SET status='queued',executor_ref=%s,dispatched_at=now(),updated_at=now() WHERE id=%s AND status='pending_dispatch' AND NOT cancel_requested", (executor_ref, job_id))
        return self.get_job(job_id)

    def claim_job(self, job_id, *, lease_seconds=300):
        if not 1 <= lease_seconds <= 86400:
            raise ValueError("Invalid job lease")
        with self._connect() as connection:
            row = self._one(connection, "SELECT * FROM jobs WHERE id=%s FOR UPDATE", (job_id,), code="job_not_found")
            if row["status"] not in ("pending_dispatch", "queued") or row["cancel_requested"]:
                raise PlatformError("job_not_claimable", 409)
            return _wire(connection.execute("UPDATE jobs SET status='running',attempt=attempt+1,attempt_token=%s,heartbeat_at=now(),lease_expires_at=now()+%s*interval '1 second',updated_at=now() WHERE id=%s RETURNING *", (uuid4(), lease_seconds, job_id)).fetchone())

    def heartbeat_job(self, job_id, attempt_token, *, lease_seconds=300):
        if not 1 <= lease_seconds <= 86400:
            raise ValueError("Invalid job lease")
        with self._connect() as connection:
            return connection.execute("UPDATE jobs SET heartbeat_at=now(),lease_expires_at=now()+%s*interval '1 second',updated_at=now() WHERE id=%s AND attempt_token=%s AND status='running' AND NOT cancel_requested AND lease_expires_at>now() RETURNING id", (lease_seconds, job_id, attempt_token)).fetchone() is not None

    def recover_expired_jobs(self):
        with self._connect() as connection:
            connection.execute("""UPDATE jobs j SET status='pending_dispatch',executor_ref=NULL,dispatched_at=NULL,updated_at=now()
                WHERE j.status='queued' AND j.attempt=0 AND NOT j.cancel_requested AND j.dispatched_at<now()-interval '5 minutes'
                AND NOT EXISTS(SELECT 1 FROM model_calls m WHERE m.job_id=j.id)""")
            rows = connection.execute("""SELECT id FROM jobs WHERE (status='running' AND lease_expires_at<now())
                OR (kind='agent_turn' AND status='pending_dispatch' AND created_at<now()-interval '5 minutes') FOR UPDATE SKIP LOCKED""").fetchall()
            for row in rows:
                connection.execute("UPDATE jobs SET status='outcome_unknown',updated_at=now() WHERE id=%s", (row["id"],))
                connection.execute("UPDATE model_calls SET status='outcome_unknown',updated_at=now() WHERE job_id=%s AND status='reserved'", (row["id"],))
                connection.execute("UPDATE agent_turns SET status='outcome_unknown',updated_at=now() WHERE job_id=%s AND status IN ('pending','running')", (row["id"],))
            return [str(row["id"]) for row in rows]

    def cancel_job(self, job_id, capability):
        with self._connect() as connection:
            row = self._one(connection, "SELECT * FROM jobs WHERE id=%s FOR UPDATE", (job_id,), code="job_not_found")
            self._auth(connection, row["project_id"], capability)
            if row["status"] in ("pending_dispatch", "queued", "running"):
                row = connection.execute("UPDATE jobs SET cancel_requested=true,status=CASE WHEN status='running' THEN status ELSE 'cancelled' END,updated_at=now() WHERE id=%s RETURNING *", (job_id,)).fetchone()
            return _wire(row)

    def finish_job(self, job_id, attempt_token, status, document=None, result=None, *, imported_capture=None):
        if status not in ("succeeded", "incomplete", "failed", "outcome_unknown"):
            raise PlatformError("invalid_job_outcome")
        if document is not None:
            validate_document(document)
        with self._connect() as connection:
            job = self._one(connection, "SELECT *,lease_expires_at>now() AS lease_valid FROM jobs WHERE id=%s FOR UPDATE", (job_id,), code="job_not_found")
            lease_valid = job.pop("lease_valid")
            if str(job["attempt_token"]) != str(attempt_token) or job["status"] != "running" or job["cancel_requested"] or not lease_valid:
                late = {"attemptToken": str(attempt_token), "status": status, "result": result, "document": document}
                if late not in job["late_results"]:
                    job = connection.execute("UPDATE jobs SET late_results=late_results || %s::jsonb,updated_at=now() WHERE id=%s RETURNING *", (Jsonb([late]), job_id)).fetchone()
                if str(job["attempt_token"]) == str(attempt_token) and job["status"] == "running" and job["cancel_requested"]:
                    job = connection.execute("UPDATE jobs SET status='cancelled',result=%s,updated_at=now() WHERE id=%s RETURNING *", (Jsonb(result), job_id)).fetchone()
                return {**_wire(job), "lateResultSaved": True}
            revision = None
            advanced = False
            if document is not None and status in ("succeeded", "incomplete"):
                branch = self._branch(connection, job["project_id"], job["branch_id"])
                revision = self._insert_revision(connection, job["project_id"], job["branch_id"], document, parent=job["base_revision_id"], label=job["kind"])
                if imported_capture is not None:
                    if job["kind"] != "import_scene" or document.get("captureId") != imported_capture.get("id"):
                        raise PlatformError("import_capture_mismatch", 422)
                    image_ids = {camera["imageId"] for camera in document["cameras"]}
                    if image_ids != {image.get("assetId") for image in imported_capture.get("images", [])}:
                        raise PlatformError("import_capture_images_mismatch", 422)
                    connection.execute("""INSERT INTO captures(id,project_id,branch_id,base_revision_id,revision_id,request_id,request_sha256,target,images,task)
                        VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""", (imported_capture["id"], job["project_id"], job["branch_id"], job["base_revision_id"], revision["id"], job["request_id"], digest(imported_capture), document["target"], Jsonb(imported_capture["images"]), Jsonb(imported_capture["task"])))
                advanced = branch["head_revision_id"] == job["base_revision_id"]
                if advanced:
                    connection.execute("UPDATE scene_branches SET head_revision_id=%s WHERE id=%s", (revision["id"], job["branch_id"]))
            return _wire(connection.execute("""UPDATE jobs SET status=%s,result=%s,result_revision_id=%s,head_advanced=%s,updated_at=now() WHERE id=%s RETURNING *""", (status, Jsonb(result), revision["id"] if revision else None, advanced, job_id)).fetchone())

    def reserve_model_call(self, job_id, attempt_token, provider, model, request_key, estimated_cost, *, code_sha256=None, model_sha256=None, adapter_sha256=None, input_sha256=None, paid=True):
        estimate = Decimal(str(estimated_cost))
        if not estimate.is_finite() or estimate < 0:
            raise PlatformError("invalid_cost")
        with self._connect() as connection:
            connection.execute("SELECT pg_advisory_xact_lock(728611937)")
            job = self._one(connection, "SELECT *,lease_expires_at>now() AS lease_valid FROM jobs WHERE id=%s FOR UPDATE", (job_id,), code="job_not_found")
            existing = connection.execute("SELECT * FROM model_calls WHERE job_id=%s AND request_key=%s", (job_id, request_key)).fetchone()
            if existing:
                raise PlatformError("model_call_already_reserved", 409, modelCallId=str(existing["id"]), status=existing["status"])
            if job["status"] != "running" or job["cancel_requested"] or str(job["attempt_token"]) != str(attempt_token) or not job["lease_valid"]:
                raise PlatformError("stale_job_attempt", 409)
            if paid or estimate > 0:
                if self.paid_budget is None:
                    raise PlatformError("paid_budget_not_configured", 409)
                spent = connection.execute("SELECT COALESCE(sum(COALESCE(actual_cost,estimated_cost)),0) AS cost FROM model_calls").fetchone()["cost"]
                if spent + estimate > self.paid_budget:
                    raise PlatformError("paid_budget_exceeded", 409)
            return _wire(connection.execute("""INSERT INTO model_calls(id,project_id,job_id,attempt_token,provider,model,request_key,status,estimated_cost,code_sha256,model_sha256,adapter_sha256,input_sha256)
                VALUES(%s,%s,%s,%s,%s,%s,%s,'reserved',%s,%s,%s,%s,%s) RETURNING *""", (uuid4(), job["project_id"], job_id, attempt_token, provider, model, request_key, estimate, code_sha256, model_sha256, adapter_sha256, input_sha256)).fetchone())

    def complete_model_call(self, call_id, status, actual_cost=None, response=None):
        if status not in ("succeeded", "failed", "outcome_unknown"):
            raise PlatformError("invalid_model_outcome")
        if actual_cost is not None and (not Decimal(str(actual_cost)).is_finite() or Decimal(str(actual_cost)) < 0):
            raise PlatformError("invalid_cost")
        with self._connect() as connection:
            row = self._one(connection, "SELECT * FROM model_calls WHERE id=%s FOR UPDATE", (call_id,), code="model_call_not_found")
            if row["status"] != "reserved":
                if row["status"] == "outcome_unknown" and status in ("succeeded", "failed"):
                    previous = row["response"] or {}
                    late = {"status": status, "actualCost": str(Decimal(str(actual_cost)).normalize()) if actual_cost is not None else None, "response": response}
                    if isinstance(previous, dict) and previous.get("lateOutcome"):
                        if previous["lateOutcome"] != late:
                            raise PlatformError("model_outcome_conflict", 409)
                        return _wire(row)
                    return _wire(connection.execute("UPDATE model_calls SET actual_cost=COALESCE(%s,actual_cost),response=%s,updated_at=now() WHERE id=%s RETURNING *", (actual_cost, Jsonb({"previousResponse": previous, "lateOutcome": late}), call_id)).fetchone())
                if row["status"] != status:
                    raise PlatformError("model_outcome_conflict", 409)
                return _wire(row)
            return _wire(connection.execute("UPDATE model_calls SET status=%s,actual_cost=%s,response=%s,updated_at=now() WHERE id=%s RETURNING *", (status, actual_cost, Jsonb(response), call_id)).fetchone())

    def create_publication(self, project_id, capability, body, *, evaluations=None, reviews=None):
        with self._connect() as connection:
            self._auth(connection, project_id, capability)
            previous = self._idempotent(connection, "publications", project_id, body)
            if previous:
                return _wire(previous)
            revision = self._revision(connection, project_id, body["sceneRevisionId"])
            if (body.get("evaluationIds") and evaluations is None) or (body.get("reviewIds") and reviews is None):
                raise PlatformError("publication_evidence_unverified", 422)
            branch = self._one(connection, "SELECT kind,title,source_revision_id FROM scene_branches WHERE project_id=%s AND id=%s", (project_id, revision["branch_id"]))
            edits = connection.execute("""WITH RECURSIVE ancestry AS (
                SELECT id,parent_revision_id,0 AS depth FROM scene_revisions WHERE project_id=%s AND id=%s
                UNION ALL SELECT r.id,r.parent_revision_id,a.depth+1 FROM scene_revisions r JOIN ancestry a ON r.id=a.parent_revision_id WHERE r.project_id=%s)
                SELECT b.* FROM ancestry a JOIN edit_batches b ON b.revision_id=a.id ORDER BY a.depth DESC""", (project_id, revision["id"], project_id)).fetchall()
            scene_version = revision["document"]["schemaVersion"]
            snapshot = {"schemaVersion": 1, "reportSchemaVersion": scene_version, "rendererVersion": f"native-webgl-v{scene_version}", "revision": _wire(revision), "evaluations": evaluations or [], "reviews": reviews or [],
                        "editBatches": _wire(edits), "branchKind": branch["kind"], "branchTitle": branch["title"]}
            jobs = connection.execute("""SELECT * FROM jobs WHERE project_id=%s AND (base_revision_id=%s OR result_revision_id=%s)
                ORDER BY created_at,id FOR SHARE""", (project_id, revision["id"], revision["id"])).fetchall()
            # The public DTO excludes executor references, attempt tokens and late
            # mutable outcomes. Config contains the server's frozen, secret-free pins.
            snapshot["jobs"] = [Job.model_validate({key: value for key, value in _wire(job).items() if key in Job.model_fields}).model_dump(mode="json") for job in jobs]
            if branch["kind"] == "planning" and branch["source_revision_id"]:
                snapshot["reconstructionRevision"] = _wire(self._revision(connection, project_id, branch["source_revision_id"]))
            reconstruction = snapshot.get("reconstructionRevision", snapshot["revision"])
            snapshot["playgroundDefinitions"] = [{"kind": "observed", "revisionId": reconstruction["id"]}, {"kind": "model", "revisionId": reconstruction["id"]}]
            if branch["kind"] == "planning":
                snapshot["playgroundDefinitions"].append({"kind": "planning", "revisionId": str(revision["id"])})
            scene_asset_ids = {asset["id"] for pinned in (snapshot["revision"], reconstruction) for asset in pinned["document"]["assets"]}
            output_refs = [ref for job in snapshot["jobs"] for ref in _job_asset_references(job["result"])]
            try:
                output_ids = {str(UUID(identity)) for identity, _ in output_refs}
            except (ValueError, TypeError, AttributeError):
                raise PlatformError("publication_job_asset_invalid", 422) from None
            asset_ids = sorted(scene_asset_ids | output_ids)
            assets = connection.execute("SELECT * FROM assets WHERE id=ANY(%s::uuid[]) ORDER BY id", (asset_ids,)).fetchall()
            if len(assets) != len(asset_ids):
                raise PlatformError("publication_asset_not_found", 422)
            assets_by_id = {str(asset["id"]): _wire(asset) for asset in assets}
            for identity, ref in output_refs:
                asset = assets_by_id[str(UUID(identity))]
                if asset["projectId"] != str(project_id) and asset["id"] not in scene_asset_ids:
                    raise PlatformError("publication_job_asset_forbidden", 403)
                if any(key in ref and ref[key] != asset[key] for key in ("sha256", "sizeBytes", "mediaType", "storageKey", "projectId", "jobId")):
                    raise PlatformError("publication_job_asset_integrity_conflict", 422)
            snapshot["assetManifest"] = []
            for asset in assets:
                if self.blobs is None:
                    raise PlatformError("blob_store_not_configured", 503)
                self.blobs.get(asset["storage_key"], asset["sha256"].strip(), asset["size_bytes"])
                snapshot["assetManifest"].append({"assetId": str(asset["id"]), "sha256": asset["sha256"].strip(), "sizeBytes": asset["size_bytes"], "mediaType": asset["media_type"]})
            return _wire(connection.execute("""INSERT INTO publications(id,project_id,scene_revision_id,request_id,request_sha256,title,evaluation_ids,review_ids,snapshot)
                VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING *""", (uuid4(), project_id, body["sceneRevisionId"], body["requestId"], digest(body), body["title"], Jsonb(body.get("evaluationIds", [])), Jsonb(body.get("reviewIds", [])), Jsonb(snapshot))).fetchone())

    def list_publications(self):
        with self._connect() as connection:
            # Aggregate inside PostgreSQL: the immutable document is never sent
            # to the application or browser for a library listing.
            return {"items": _wire(connection.execute("""SELECT p.id,p.project_id,p.scene_revision_id,p.title,p.created_at,
                images.preview_image_asset_id,images.photo_count,objects.object_count,objects.spatial_object_count,
                objects.model_object_count,objects.observed_surface_object_count
                FROM (SELECT id,project_id,scene_revision_id,title,created_at,snapshot->'revision'->'document' AS document
                    FROM publications ORDER BY created_at DESC,id LIMIT 200) p
                CROSS JOIN LATERAL (
                    SELECT count(*) AS photo_count,(array_agg(image->>'id' ORDER BY ordinal))[1] AS preview_image_asset_id
                    FROM jsonb_array_elements(p.document->'assets') WITH ORDINALITY AS images(image,ordinal)
                    WHERE image->>'kind'='source_image'
                ) images
                CROSS JOIN LATERAL (
                    SELECT count(*) AS object_count,
                        count(*) FILTER (WHERE jsonb_array_length(entity->'representations')>0) AS spatial_object_count,
                        count(*) FILTER (WHERE EXISTS (SELECT 1 FROM jsonb_array_elements(entity->'representations') r
                            WHERE r->>'kind' IN ('generated_mesh','primitive'))) AS model_object_count,
                        count(*) FILTER (WHERE EXISTS (SELECT 1 FROM jsonb_array_elements(entity->'representations') r
                            WHERE r->>'kind'='observed_surface')) AS observed_surface_object_count
                    FROM jsonb_array_elements(p.document->'entities') entity
                    WHERE entity->'sourceContext' IS DISTINCT FROM 'true'::jsonb
                ) objects ORDER BY p.created_at DESC,p.id""").fetchall())}

    def get_publication(self, publication_id):
        with self._connect() as connection:
            return _wire(self._one(connection, "SELECT * FROM publications WHERE id=%s", (publication_id,), code="publication_not_found"))

    def create_agent_turn(self, project_id, capability, body):
        with self._connect() as connection:
            self._auth(connection, project_id, capability)
            previous = self._idempotent(connection, "agent_turns", project_id, body)
            if previous:
                return _wire(previous)
            self._branch(connection, project_id, body["branchId"], body["baseRevisionId"])
            document = self._revision(connection, project_id, body["baseRevisionId"])["document"]
            if body.get("imageId"):
                image = next((asset for asset in document["assets"] if asset["id"] == str(body["imageId"])), None)
                if image is None:
                    raise PlatformError("agent_image_scope_not_found", 422)
                asset = self._one(connection, "SELECT media_type FROM assets WHERE id=%s", (body["imageId"],), code="agent_image_scope_not_found")
                if not asset["media_type"].startswith("image/"):
                    raise PlatformError("agent_image_scope_not_found", 422)
            if bool(body.get("policyId")) != bool(body.get("policyRevisionId")):
                raise PlatformError("agent_policy_scope_invalid", 422)
            if body.get("policyId"):
                self._one(connection, "SELECT id FROM policy_revisions WHERE project_id=%s AND policy_id=%s AND id=%s", (project_id, body["policyId"], body["policyRevisionId"]), code="agent_policy_scope_not_found")
            for field, collection in (("entityId", "entities"), ("observationId", "observations")):
                if body.get(field) and body[field] not in {x["id"] for x in document[collection]}:
                    raise PlatformError("agent_scope_not_found", 422, field=field)
            pair = body.get("identityEntityIds")
            if pair is not None:
                entity_ids = {e["id"] for e in document["entities"] if not e.get("sourceContext")}
                if not isinstance(pair, (list, tuple)) or len(pair) != 2 or not all(isinstance(eid, str) for eid in pair) or pair[0] == pair[1] or not set(pair) <= entity_ids or (body.get("entityId") and body["entityId"] not in pair):
                    raise PlatformError("agent_identity_scope_invalid", 422)
            turn_id, job_id = uuid4(), uuid4()
            job_body = {"requestId": body["requestId"], "branchId": body["branchId"], "baseRevisionId": body["baseRevisionId"], "kind": "agent_turn", "inputs": {"turnId": str(turn_id)}, "config": {}}
            connection.execute("""INSERT INTO jobs(id,project_id,branch_id,base_revision_id,request_id,request_sha256,kind,inputs,config,status)
                VALUES(%s,%s,%s,%s,%s,%s,'agent_turn',%s,%s,'pending_dispatch')""", (job_id, project_id, body["branchId"], body["baseRevisionId"], body["requestId"], digest(job_body), Jsonb(job_body["inputs"]), Jsonb(self._job_config({}))))
            return _wire(connection.execute("""INSERT INTO agent_turns(id,project_id,conversation_id,branch_id,base_revision_id,request_id,request_sha256,request,status,job_id)
                VALUES(%s,%s,%s,%s,%s,%s,%s,%s,'pending',%s) RETURNING *""", (turn_id, project_id, body["conversationId"], body["branchId"], body["baseRevisionId"], body["requestId"], digest(body), Jsonb(body), job_id)).fetchone())

    def claim_agent_turn(self, turn_id):
        with self._connect() as connection:
            row = connection.execute("UPDATE agent_turns SET status='running',updated_at=now() WHERE id=%s AND status='pending' RETURNING *", (turn_id,)).fetchone()
            if row is None:
                raise PlatformError("agent_turn_not_claimable", 409)
            return _wire(row)

    def list_agent_turns(self, project_id, *, conversation_id=None, after_sequence=0):
        with self._connect() as connection:
            self._one(connection, "SELECT id FROM projects WHERE id=%s", (project_id,), code="project_not_found")
            rows = connection.execute("""SELECT t.*,b.id AS applied_edit_batch_id,p.id AS applied_policy_revision_id FROM agent_turns t
                LEFT JOIN edit_batches b ON b.project_id=t.project_id AND b.agent_turn_id=t.id
                LEFT JOIN policy_revisions p ON p.project_id=t.project_id AND p.agent_turn_id=t.id
                WHERE t.project_id=%s AND (%s::uuid IS NULL OR t.conversation_id=%s) AND t.sequence>%s ORDER BY t.sequence LIMIT 500""", (project_id, conversation_id, conversation_id, after_sequence)).fetchall()
            return {"items": _wire(rows)}

    def finish_agent_turn(self, turn_id, status, response):
        if status not in ("succeeded", "failed", "outcome_unknown"):
            raise PlatformError("invalid_agent_outcome")
        with self._connect() as connection:
            row = connection.execute("UPDATE agent_turns SET status=%s,response=%s,updated_at=now() WHERE id=%s AND status='running' RETURNING *", (status, Jsonb(response), turn_id)).fetchone()
            if row is None:
                raise PlatformError("agent_outcome_conflict", 409)
            return _wire(row)
