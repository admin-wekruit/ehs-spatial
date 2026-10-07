"""MongoDB business transactions with PostgresRepository semantics.

One collection per PostgreSQL table (name = PANOPTES_MONGO_PREFIX + table), documents keep the
SQL column names so the wire format is identical. No multi-document transactions are used: every
invariant is one conditional find_one_and_update/update_one (branch head CAS, job leases, unique
request ids) plus short lease locks in a `locks` collection where PostgreSQL serialises with
advisory or row locks (request-id replay windows, per-job mutations, the global paid budget).
Transliterated from postgres.py; keep the two in step.
"""
from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import hmac
import os
import time
from urllib.parse import urlsplit
from uuid import UUID, uuid4, uuid5

from bson.decimal128 import Decimal128
import pymongo
from pymongo import ReturnDocument
from pymongo.errors import DuplicateKeyError

from .contracts import Job, PlatformError, capability_sha, digest, empty_document, validate_document
from .postgres import JSON_FIELDS, _job_asset_references, _wire as _wire_row
from .repository import apply_operations

MONGO_SCHEMES = ("mongodb://", "mongodb+srv://")
LOCK_SECONDS = 60
STALE_QUEUE = timedelta(minutes=5)

# (unique, keys[, partial filter]) per table; mirrors the UNIQUE constraints and query paths of the SQL schema.
INDEXES = {
    "projects": [(True, ["capability_sha256"]), (True, ["capability_sha256", "request_id"]), (False, ["created_at"])],
    "scene_branches": [(True, ["project_id", "request_id"], {"request_id": {"$type": "string"}}), (False, ["project_id", "created_at"])],
    "captures": [(True, ["project_id", "request_id"]), (False, ["project_id", "created_at"])],
    "scene_revisions": [(False, ["project_id", "created_at"]), (False, ["project_id", "parent_revision_id"])],
    "edit_batches": [(True, ["project_id", "request_id"]), (False, ["revision_id"]), (False, ["project_id", "agent_turn_id"]), (False, ["project_id", "created_at"])],
    "jobs": [(True, ["project_id", "request_id"]), (False, ["status", "created_at"]), (False, ["project_id", "base_revision_id"]),
             (False, ["project_id", "result_revision_id"]), (False, ["project_id", "created_at"]), (False, ["lease_expires_at"])],
    "assets": [(True, ["project_id", "storage_key"]), (False, ["project_id", "metadata.kind", "metadata.cacheKey"]), (False, ["project_id", "created_at"])],
    "model_calls": [(True, ["job_id", "request_key"]), (False, ["project_id", "provider", "model", "request_key"]), (False, ["job_id", "status"])],
    "publications": [(True, ["project_id", "request_id"]), (False, ["created_at"])],
    "agent_turns": [(True, ["project_id", "request_id"]), (False, ["project_id", "sequence"]), (False, ["job_id"]), (False, ["project_id", "created_at"])],
    "policy_sources": [(False, ["project_id"])],
    "policies": [(True, ["project_id", "request_id"]), (False, ["created_at"])],
    "policy_revisions": [(True, ["project_id", "request_id"]), (True, ["agent_turn_id"], {"agent_turn_id": {"$type": "string"}}), (False, ["policy_id", "created_at"])],
    "policy_evaluations": [(True, ["project_id", "request_id"]), (False, ["project_id", "created_at"])],
    "policy_reviews": [(True, ["project_id", "request_id"]), (False, ["project_id", "created_at"])],
    "policy_evidence_requests": [(True, ["project_id", "request_id"]), (False, ["project_id", "document.evidenceRequestId"]), (False, ["project_id", "created_at"])],
    "locks": [(False, ["expires_at"])],
    "counters": [],
}


def _s(value):
    return None if value is None else str(value)


def _plain(value):
    """Mongo document -> PostgreSQL row shape: _id -> id, Decimal128 -> Decimal, naive datetimes are UTC."""
    if isinstance(value, dict):
        return {("id" if key == "_id" else key): (item if key in JSON_FIELDS else _plain(item)) for key, item in value.items()}
    if isinstance(value, list):
        return [_plain(item) for item in value]
    if isinstance(value, Decimal128):
        return value.to_decimal()
    if isinstance(value, datetime) and value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _wire(row):
    return _wire_row(_plain(row))


def _cost(value):
    return None if value is None else Decimal128(Decimal(str(value)))


class MongoRepository:
    def __init__(self, url, *, paid_budget=None, blob_store=None, execution_config=None, client=None, prefix=None, database=None):
        self.url = url
        self.paid_budget = Decimal(str(paid_budget)) if paid_budget is not None else None
        if self.paid_budget is not None and (not self.paid_budget.is_finite() or self.paid_budget < 0):
            raise ValueError("paid_budget must be nonnegative and finite")
        self.blobs = blob_store
        self.execution_config = deepcopy(execution_config or {"providerManifest": {}})
        self.prefix = os.environ.get("PANOPTES_MONGO_PREFIX", "panoptes_") if prefix is None else prefix
        self.client = client if client is not None else pymongo.MongoClient(url, tz_aware=True)
        self.db = self.client[database or urlsplit(url).path.strip("/").split("/")[0] or "panoptes"]

    # ---- infrastructure -------------------------------------------------------------------

    def _c(self, table):
        return self.db[self.prefix + table]

    _last_now = None

    def _now(self):
        # BSON dates are milliseconds (PostgreSQL has microseconds): keep created_at strictly increasing
        # within a process so ORDER BY created_at stays deterministic for records written back to back.
        now = datetime.now(timezone.utc)
        now = now.replace(microsecond=now.microsecond // 1000 * 1000)
        if self._last_now is not None and now <= self._last_now:
            now = self._last_now + timedelta(milliseconds=1)
        self._last_now = now
        return now

    def migrate(self):
        for table, indexes in INDEXES.items():
            for unique, keys, *partial in indexes:
                options = {"unique": unique, "name": "_".join(keys) + ("_unique" if unique else "")}
                if partial:
                    options["partialFilterExpression"] = partial[0]
                self._c(table).create_index([(key, pymongo.ASCENDING) for key in keys], **options)

    def ping(self):
        self.db.command("ping")

    @contextmanager
    def _lock(self, key, *, block=True):
        # ponytail: lease lock document instead of advisory/row locks; LOCK_SECONDS bounds a crashed
        # holder. Only used where PostgreSQL serialises callers, never as the sole data invariant.
        locks = self._c("locks")
        deadline = time.monotonic() + LOCK_SECONDS
        while True:
            now = self._now()
            try:
                locks.find_one_and_update({"_id": key, "expires_at": {"$lt": now}}, {"$set": {"expires_at": now + timedelta(seconds=LOCK_SECONDS)}}, upsert=True)
                acquired = True
            except DuplicateKeyError:
                acquired = False
            if acquired or not block:
                break
            if time.monotonic() > deadline:
                raise PlatformError("lock_timeout", 503, key=key)
            time.sleep(0.02)
        try:
            yield acquired
        finally:
            if acquired:
                locks.delete_one({"_id": key})

    def _sequence(self, name):
        return self._c("counters").find_one_and_update({"_id": name}, {"$inc": {"value": 1}}, upsert=True, return_document=ReturnDocument.AFTER)["value"]

    def _one(self, table, query, *, code="not_found", sort=None, projection=None):
        row = self._c(table).find_one(query, projection, sort=sort)
        if row is None:
            raise PlatformError(code, 404)
        return row

    def _insert(self, table, row):
        try:
            self._c(table).insert_one(row)
        except pymongo.errors.DocumentTooLarge:
            raise PlatformError("document_too_large", 413, table=table) from None
        return row

    def _job_config(self, client_config):
        client = {key: value for key, value in client_config.items()
                  if key not in {'providerManifest', 'researchPreparation', 'researchProtocolSha256',
                                 'pipeline', 'pipelineStage', 'pipelineStep', 'pipelineRootJobId', 'parentJobId',
                                 'continuation', 'continuationJobId', 'continuationStopped', 'submittedBy', 'captureAnalysis'}}
        return {**deepcopy(client), **deepcopy(self.execution_config)}

    def _auth(self, project_id, capability):
        supplied = capability_sha(capability)
        row = self._one("projects", {"_id": _s(project_id)}, code="project_not_found")
        if not hmac.compare_digest(row["capability_sha256"].strip(), supplied):
            raise PlatformError("capability_forbidden", 403)
        return row

    def authorize(self, project_id, capability):
        return _wire(self._auth(project_id, capability))

    @contextmanager
    def _idempotent(self, table, project_id, body, *, scope=""):
        """Serialise one business request (including simultaneous lost-response retries) and yield its prior record."""
        with self._lock(f"{project_id}:{scope}{body['requestId']}"):
            row = self._c(table).find_one({"project_id": _s(project_id), "request_id": _s(body["requestId"])})
            if row and row["request_sha256"].strip() != digest(body):
                raise PlatformError("idempotency_mismatch", 409)
            yield row

    def _branch(self, project_id, branch_id, base_revision_id=None):
        branch = self._one("scene_branches", {"_id": _s(branch_id), "project_id": _s(project_id)}, code="branch_not_found")
        if base_revision_id is not None and branch["head_revision_id"] != str(base_revision_id):
            raise PlatformError("revision_conflict", 409, baseRevisionId=str(base_revision_id), currentRevisionId=branch["head_revision_id"])
        return branch

    def _advance_head(self, project_id, branch_id, base_revision_id, revision_id):
        """Compare-and-set the branch head: the single atomic step every revision writer relies on."""
        result = self._c("scene_branches").update_one({"_id": _s(branch_id), "project_id": _s(project_id), "head_revision_id": _s(base_revision_id)},
                                                      {"$set": {"head_revision_id": revision_id}})
        return result.matched_count == 1

    def _advance_or_conflict(self, project_id, branch_id, base_revision_id, revision):
        if not self._advance_head(project_id, branch_id, base_revision_id, revision["_id"]):
            self._c("scene_revisions").delete_one({"_id": revision["_id"]})
            current = self._branch(project_id, branch_id)["head_revision_id"]
            raise PlatformError("revision_conflict", 409, baseRevisionId=str(base_revision_id), currentRevisionId=current)

    def _revision(self, project_id, revision_id):
        return self._one("scene_revisions", {"_id": _s(revision_id), "project_id": _s(project_id)}, code="revision_not_found")

    def _insert_revision(self, project_id, branch_id, document, *, parent=None, source=None, label=None, revision_id=None, project=None):
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
            project = project or self._one("projects", {"_id": _s(project_id)})
            inherited = set()
            if project["fork_source_revision_id"]:
                source_document = self._one("scene_revisions", {"_id": project["fork_source_revision_id"]})["document"]
                inherited = {item["id"] for item in source_document["assets"]}
            assets = list(self._c("assets").find({"_id": {"$in": identities}}))
            if len(assets) != len(identities):
                raise PlatformError("revision_asset_not_found", 422)
            for asset in assets:
                if asset["project_id"] != str(project_id) and asset["_id"] not in inherited:
                    raise PlatformError("revision_asset_forbidden", 403)
                self.blobs.get(asset["storage_key"], asset["sha256"].strip(), asset["size_bytes"])
        row = {"_id": _s(revision_id) or str(uuid4()), "project_id": _s(project_id), "branch_id": _s(branch_id), "parent_revision_id": _s(parent),
               "source_revision_id": _s(source), "document": deepcopy(document), "document_sha256": digest(document), "label": label, "created_at": self._now()}
        return self._insert("scene_revisions", row)

    # ---- projects and revisions -----------------------------------------------------------

    def create_project(self, capability, body, *, source_revision_id=None):
        cap_sha = capability_sha(capability)
        with self._lock("project:" + cap_sha):
            previous = self._c("projects").find_one({"capability_sha256": cap_sha})
            if previous:
                if previous["request_id"] != str(body["requestId"]) or previous["request_sha256"].strip() != digest(body):
                    raise PlatformError("idempotency_mismatch", 409)
                return self._project_response(previous)
            source = None
            if source_revision_id:
                source = self._one("scene_revisions", {"_id": _s(source_revision_id)}, code="revision_not_found")
            document = deepcopy(source["document"]) if source else empty_document()
            if not source:
                document["target"] = body.get("target", "scene")
            if body.get("operations"):
                document, _ = apply_operations(document, body["operations"], base_revision_id=source["_id"] if source else None)
            project_id, branch_id, revision_id = str(uuid4()), str(uuid4()), str(uuid4())
            project = {"_id": project_id, "title": body["title"], "capability_sha256": cap_sha, "request_id": str(body["requestId"]), "request_sha256": digest(body),
                       "default_branch_id": branch_id, "fork_source_revision_id": _s(source_revision_id), "created_at": self._now()}
            # The project document is the entry point, so it is written last: readers never see a partial project.
            self._insert_revision(project_id, branch_id, document, source=source_revision_id, label="Fork" if source else "Initial", revision_id=revision_id, project=project)
            self._insert("scene_branches", {"_id": branch_id, "project_id": project_id, "kind": "reconstruction", "title": "Reconstruction", "source_revision_id": None,
                                            "head_revision_id": revision_id, "request_id": None, "request_sha256": None, "created_at": self._now()})
            self._insert("projects", project)
            return self._project_response(project)

    def fork_project(self, source_project_id, capability, body):
        self._revision(source_project_id, body["sourceRevisionId"])
        return self.create_project(capability, body, source_revision_id=body["sourceRevisionId"])

    def _project_response(self, project):
        branches = list(self._c("scene_branches").find({"project_id": project["_id"]}).sort([("created_at", 1), ("_id", 1)]))
        branch = next(row for row in branches if row["_id"] == project["default_branch_id"])
        return _wire({"project": project, "branch": branch, "revision": self._revision(project["_id"], branch["head_revision_id"]), "branches": branches})

    def get_project(self, project_id):
        return self._project_response(self._one("projects", {"_id": _s(project_id)}, code="project_not_found"))

    def list_projects(self):
        return {"items": _wire(list(self._c("projects").find().sort([("created_at", -1), ("_id", 1)]).limit(200)))}

    def get_revision(self, revision_id):
        return _wire(self._one("scene_revisions", {"_id": _s(revision_id)}, code="revision_not_found"))

    def list_project_records(self, project_id, kind):
        table = {"revisions": "scene_revisions", "captures": "captures", "assets": "assets", "jobs": "jobs", "agent-turns": "agent_turns", "edits": "edit_batches"}.get(kind)
        if not table:
            raise ValueError("Unsupported record type")
        self._one("projects", {"_id": _s(project_id)}, code="project_not_found")
        return {"items": _wire(list(self._c(table).find({"project_id": _s(project_id)}).sort([("created_at", -1), ("_id", 1)]).limit(500)))}

    def create_branch(self, project_id, capability, body):
        self._auth(project_id, capability)
        with self._idempotent("scene_branches", project_id, body) as previous:
            if previous:
                return _wire(previous)
            source = self._revision(project_id, body["sourceRevisionId"])
            branch_id, revision_id = str(uuid4()), str(uuid4())
            self._insert_revision(project_id, branch_id, source["document"], parent=source["_id"], source=source["_id"], label=body["title"], revision_id=revision_id)
            branch = {"_id": branch_id, "project_id": _s(project_id), "kind": body.get("kind", "planning"), "title": body["title"], "source_revision_id": source["_id"],
                      "head_revision_id": revision_id, "request_id": str(body["requestId"]), "request_sha256": digest(body), "created_at": self._now()}
            return _wire(self._insert("scene_branches", branch))

    def commit_edits(self, project_id, capability, body):
        self._auth(project_id, capability)
        pid, branch_id = _s(project_id), _s(body["branchId"])
        with self._idempotent("edit_batches", project_id, body) as previous:
            if previous:
                return _wire({"revision": self._revision(project_id, previous["revision_id"]), "edit_batch": previous, "head_advanced": True})
            self._branch(project_id, branch_id, body["baseRevisionId"])
            base = self._revision(project_id, body["baseRevisionId"])
            document, operations = None, body.get("operations", [])
            if body.get("agentTurnId"):
                turn = self._one("agent_turns", {"project_id": pid, "_id": _s(body["agentTurnId"])}, code="agent_turn_not_found")
                proposal = turn["response"] or {}
                if turn["status"] != "succeeded" or proposal.get("kind") != "proposal" or turn["branch_id"] != branch_id or turn["base_revision_id"] != base["_id"] or proposal.get("operations") != operations:
                    raise PlatformError("agent_proposal_mismatch", 409)
            if body.get("undoOf") or body.get("redoOf"):
                batch_id = _s(body.get("undoOf") or body["redoOf"])
                batch = self._one("edit_batches", {"project_id": pid, "branch_id": branch_id, "_id": batch_id}, code="edit_batch_not_found")
                if body.get("undoOf"):
                    if batch["revision_id"] != base["_id"]:
                        raise PlatformError("undo_conflict", 409)
                    document = deepcopy(batch["inverse_operations"][0]["document"])
                else:
                    undo = self._c("edit_batches").find_one({"project_id": pid, "revision_id": base["_id"], "undo_of": batch_id})
                    if undo is None:
                        raise PlatformError("redo_conflict", 409)
                    document = deepcopy(self._revision(project_id, batch["revision_id"])["document"])
                operations = [{"type": "undo" if body.get("undoOf") else "redo", "editBatchId": batch_id}]
                inverse = [{"type": "restoreDocument", "document": base["document"]}]
            else:
                document, inverse = apply_operations(base["document"], operations, base_revision_id=base["_id"])
            revision = self._insert_revision(project_id, branch_id, document, parent=base["_id"], label=body.get("label"))
            self._advance_or_conflict(project_id, branch_id, base["_id"], revision)
            batch = {"_id": str(uuid4()), "project_id": pid, "branch_id": branch_id, "base_revision_id": base["_id"], "revision_id": revision["_id"],
                     "request_id": str(body["requestId"]), "request_sha256": digest(body), "operations": operations, "inverse_operations": inverse,
                     "undo_of": _s(body.get("undoOf")), "redo_of": _s(body.get("redoOf")), "agent_turn_id": _s(body.get("agentTurnId")), "created_at": self._now()}
            self._insert("edit_batches", batch)
            return _wire({"revision": revision, "edit_batch": batch, "head_advanced": True})

    # ---- assets ---------------------------------------------------------------------------

    def register_asset(self, project_id, metadata, job_id=None):
        required = ("storageKey", "sha256", "sizeBytes", "mediaType")
        if any(key not in metadata for key in required) or len(metadata["sha256"]) != 64 or metadata["storageKey"] != "sha256/" + metadata["sha256"] or metadata["sizeBytes"] < 0:
            raise PlatformError("invalid_asset")
        return _wire(self._register_asset(project_id, metadata, job_id))

    def _register_asset(self, project_id, metadata, job_id=None):
        if self.blobs is None:
            raise PlatformError("blob_store_not_configured", 503)
        self.blobs.get(metadata["storageKey"], metadata["sha256"], metadata["sizeBytes"])
        previous = self._c("assets").find_one({"project_id": _s(project_id), "storage_key": metadata["storageKey"]})
        if previous:
            if previous["sha256"].strip() != metadata["sha256"] or previous["size_bytes"] != metadata["sizeBytes"]:
                raise PlatformError("asset_integrity_conflict", 409)
            return previous
        row = {"_id": _s(metadata.get("id")) or str(uuid4()), "project_id": _s(project_id), "job_id": _s(job_id), "storage_key": metadata["storageKey"], "sha256": metadata["sha256"],
               "size_bytes": metadata["sizeBytes"], "media_type": metadata["mediaType"], "metadata": deepcopy(metadata.get("metadata", {})), "created_at": self._now()}
        try:
            return self._insert("assets", row)
        except DuplicateKeyError:
            return self._register_asset(project_id, metadata, job_id)  # concurrent identical registration

    def get_asset(self, asset_id):
        return _wire(self._one("assets", {"_id": _s(asset_id)}, code="asset_not_found"))

    def get_stage_cache(self, project_id, cache_key):
        return _wire(self._c("assets").find_one({"project_id": _s(project_id), "metadata.kind": "stage_cache", "metadata.cacheKey": cache_key}, sort=[("created_at", 1), ("_id", 1)]))

    # ---- captures and jobs ----------------------------------------------------------------

    def _new_job(self, project_id, branch_id, base_revision_id, request_id, request_sha256, kind, inputs, config, *, job_id=None):
        now = self._now()
        row = {"_id": _s(job_id) or str(uuid4()), "project_id": _s(project_id), "branch_id": _s(branch_id), "base_revision_id": _s(base_revision_id), "request_id": _s(request_id),
               "request_sha256": request_sha256, "kind": kind, "inputs": deepcopy(inputs), "config": config, "status": "pending_dispatch", "cancel_requested": False, "attempt": 0,
               "attempt_token": None, "dispatched_at": None, "executor_ref": None, "result": None, "late_results": [], "result_revision_id": None, "head_advanced": False,
               "created_at": now, "updated_at": now, "heartbeat_at": None, "lease_expires_at": None}
        return self._insert("jobs", row)

    def create_capture(self, project_id, capability, body, images):
        mode = body.get("captureMode", "initial")
        if mode not in ("initial", "append"):
            raise PlatformError("invalid_capture_mode", 422)
        if not isinstance(images, list) or not 1 <= len(images) <= 4:
            raise PlatformError("image_count_out_of_range", 422, minimum=1, maximum=4)
        request = {**body, "images": [{k: value for k, value in image.items() if k != "id"} for image in images]}
        self._auth(project_id, capability)
        pid, branch_id = _s(project_id), _s(body["branchId"])
        with self._idempotent("captures", project_id, request) as previous:
            if previous:
                job = self._one("jobs", {"project_id": pid, "request_id": str(body["requestId"])})
                return _wire({"capture": previous, "revision": self._revision(project_id, previous["revision_id"]), "job": job})
            self._branch(project_id, branch_id, body["baseRevisionId"])
            base = self._revision(project_id, body["baseRevisionId"])
            source = base["document"]
            if mode == "initial" and (source.get("captureId") or source.get("captureIds") or any(source[key] for key in ("entities", "observations", "coordinateFrames", "cameras", "assets", "annotations"))):
                raise PlatformError("capture_append_required", 409)
            if mode == "append" and body["target"] != source["target"]:
                raise PlatformError("capture_target_mismatch", 422)
            identity, revision_id = str(uuid4()), str(uuid4())
            asset_rows = [self._register_asset(project_id, image) for image in images]
            existing_images = {a["id"] for a in source["assets"] if a.get("kind") == "source_image"}
            capture_ids = source.get("captureIds") or ([source["captureId"]] if source.get("captureId") else [])
            old_captures = list(self._c("captures").find({"project_id": pid, "_id": {"$in": capture_ids}}, {"images": 1})) if capture_ids else []
            prior_images = [image for capture in old_captures for image in capture["images"]]
            def image_mapping(image):
                return {key: image.get(key) for key in ("width", "height", "originalWidth", "originalHeight", "pixelMapping", "exifOrientation")}
            image_records = []
            for asset, image in zip(asset_rows, images):
                record = {**image.get("metadata", {}), "id": asset["_id"], "assetId": asset["_id"]}
                duplicate = next((r for r in image_records if r["id"] == record["id"]), None)
                prior = [r for r in prior_images if r["assetId"] == record["assetId"]]
                if duplicate is not None and image_mapping(duplicate) != image_mapping(record) or record["id"] in existing_images and not any(image_mapping(old) == image_mapping(record) for old in prior):
                    raise PlatformError("capture_image_mapping_conflict", 422, imageId=record["id"])
                if duplicate is None:
                    record["sourceReused"] = record["id"] in existing_images
                    image_records.append(record)
            if mode == "append":
                from .identity import migrate_document
                document = migrate_document(source, base_revision_id=base["_id"]) if source["schemaVersion"] == 1 else deepcopy(source)
                document["captureIds"].append(identity)
            else:
                document = empty_document()
            document.update(captureId=identity, target=body["target"])
            asset_ids = {a["id"] for a in document["assets"]}
            for asset in asset_rows:
                aid = asset["_id"]
                if aid not in asset_ids:
                    document["assets"].append({"id": aid, "kind": "source_image", "mediaType": asset["media_type"], "sha256": asset["sha256"].strip(), "sizeBytes": asset["size_bytes"]})
                    asset_ids.add(aid)
                    if document["schemaVersion"] == 2:
                        document["geometryBindings"][aid] = None
            new_images = [image["id"] for image in image_records if not image["sourceReused"]]
            reused_images = [image["id"] for image in image_records if image["sourceReused"]]
            task = {"schemaVersion": 1, "kind": "capture_reconstruction", "target": body["target"], "imageIds": [image["id"] for image in image_records],
                    "captureId": identity, "captureMode": mode, "newImageIds": new_images, "reusedImageIds": reused_images}
            revision = self._insert_revision(project_id, branch_id, document, parent=base["_id"], label="Capture", revision_id=revision_id)
            self._advance_or_conflict(project_id, branch_id, base["_id"], revision)
            capture = self._insert("captures", {"_id": identity, "project_id": pid, "branch_id": branch_id, "base_revision_id": base["_id"], "revision_id": revision_id,
                                                "request_id": str(body["requestId"]), "request_sha256": digest(request), "target": body["target"], "images": image_records, "task": task, "created_at": self._now()})
            job_body = {"requestId": body["requestId"], "branchId": body["branchId"], "baseRevisionId": revision_id,
                        "kind": "analyze_capture", "inputs": {"captureId": identity, "captureMode": mode, "newImageIds": new_images, "reusedImageIds": reused_images}, "config": {}}
            job = self._new_job(project_id, branch_id, revision_id, body["requestId"], digest(job_body), "analyze_capture", job_body["inputs"], self._job_config({}))
            return _wire({"capture": capture, "revision": revision, "job": job})

    def create_job(self, project_id, capability, body):
        if body["kind"] in {"validate_model", "reconstruct_scene"}:
            raise PlatformError("admin_job_required", 403)
        self._auth(project_id, capability)
        with self._idempotent("jobs", project_id, body) as previous:
            if previous:
                return _wire(previous)
            self._branch(project_id, body["branchId"], body["baseRevisionId"])
            self._revision(project_id, body["baseRevisionId"])
            for asset_id in body.get("inputs", {}).get("assetIds", []):
                self._one("assets", {"project_id": _s(project_id), "_id": _s(asset_id)}, code="asset_not_found")
            return _wire(self._new_job(project_id, body["branchId"], body["baseRevisionId"], body["requestId"], digest(body), body["kind"], body.get("inputs", {}), self._job_config(body.get("config", {}))))

    def get_job(self, job_id):
        return _wire(self._one("jobs", {"_id": _s(job_id)}, code="job_not_found"))

    def pending_jobs(self, limit=100):
        rows = self._c("jobs").find({"status": "pending_dispatch", "kind": {"$nin": ["agent_turn", "import_scene"]}, "cancel_requested": False}).sort("created_at", 1).limit(min(limit, 500))
        return _wire(list(rows))

    def mark_dispatched(self, job_id, executor_ref):
        now = self._now()
        self._c("jobs").update_one({"_id": _s(job_id), "status": "pending_dispatch", "cancel_requested": False},
                                   {"$set": {"status": "queued", "executor_ref": executor_ref, "dispatched_at": now, "updated_at": now}})
        return self.get_job(job_id)

    def claim_job(self, job_id, *, lease_seconds=300):
        if not 1 <= lease_seconds <= 86400:
            raise ValueError("Invalid job lease")
        with self._lock("job:" + _s(job_id)):
            now = self._now()
            row = self._c("jobs").find_one_and_update(
                {"_id": _s(job_id), "status": {"$in": ["pending_dispatch", "queued"]}, "cancel_requested": False},
                {"$set": {"status": "running", "attempt_token": str(uuid4()), "heartbeat_at": now, "lease_expires_at": now + timedelta(seconds=lease_seconds), "updated_at": now}, "$inc": {"attempt": 1}},
                return_document=ReturnDocument.AFTER)
            if row is None:
                self._one("jobs", {"_id": _s(job_id)}, code="job_not_found")
                raise PlatformError("job_not_claimable", 409)
            return _wire(row)

    def heartbeat_job(self, job_id, attempt_token, *, lease_seconds=300):
        if not 1 <= lease_seconds <= 86400:
            raise ValueError("Invalid job lease")
        now = self._now()
        result = self._c("jobs").update_one(
            {"_id": _s(job_id), "attempt_token": _s(attempt_token), "status": "running", "cancel_requested": False, "lease_expires_at": {"$gt": now}},
            {"$set": {"heartbeat_at": now, "lease_expires_at": now + timedelta(seconds=lease_seconds), "updated_at": now}})
        return result.matched_count == 1

    def recover_expired_jobs(self):
        jobs, now = self._c("jobs"), self._now()
        stale = {"status": "queued", "attempt": 0, "cancel_requested": False, "dispatched_at": {"$lt": now - STALE_QUEUE}}
        for row in jobs.find(stale, {"_id": 1}):
            if self._c("model_calls").find_one({"job_id": row["_id"]}, {"_id": 1}) is None:
                jobs.update_one({**stale, "_id": row["_id"]}, {"$set": {"status": "pending_dispatch", "executor_ref": None, "dispatched_at": None, "updated_at": now}})
        expired = {"$or": [{"status": "running", "lease_expires_at": {"$lt": now}},
                           {"kind": "agent_turn", "status": "pending_dispatch", "created_at": {"$lt": now - STALE_QUEUE}}]}
        recovered = []
        for row in jobs.find(expired, {"_id": 1}):
            with self._lock("job:" + row["_id"], block=False) as acquired:  # SKIP LOCKED: a finishing worker holds the job
                if not acquired or jobs.update_one({**expired, "_id": row["_id"]}, {"$set": {"status": "outcome_unknown", "updated_at": now}}).matched_count == 0:
                    continue
                self._c("model_calls").update_many({"job_id": row["_id"], "status": "reserved"}, {"$set": {"status": "outcome_unknown", "updated_at": now}})
                self._c("agent_turns").update_many({"job_id": row["_id"], "status": {"$in": ["pending", "running"]}}, {"$set": {"status": "outcome_unknown", "updated_at": now}})
                recovered.append(row["_id"])
        return recovered

    def cancel_job(self, job_id, capability):
        with self._lock("job:" + _s(job_id)):
            row = self._one("jobs", {"_id": _s(job_id)}, code="job_not_found")
            self._auth(row["project_id"], capability)
            if row["status"] in ("pending_dispatch", "queued", "running"):
                status = row["status"] if row["status"] == "running" else "cancelled"
                row = self._c("jobs").find_one_and_update({"_id": row["_id"]}, {"$set": {"cancel_requested": True, "status": status, "updated_at": self._now()}}, return_document=ReturnDocument.AFTER)
            return _wire(row)

    def finish_job(self, job_id, attempt_token, status, document=None, result=None, *, imported_capture=None, continuation=None):
        if status not in ("succeeded", "incomplete", "failed", "outcome_unknown"):
            raise PlatformError("invalid_job_outcome")
        if continuation is not None:
            if (not isinstance(continuation, dict) or set(continuation) - {'kind', 'inputs', 'config'}
                    or continuation.get('kind') not in ('reconstruct_scene', 'validate_model')
                    or not isinstance(continuation.get('inputs', {}), dict)
                    or not isinstance(continuation.get('config', {}), dict)
                    or result is not None and not isinstance(result, dict)):
                raise PlatformError('invalid_job_continuation', 422)
            if continuation['kind'] == 'validate_model':
                # Frozen research dispatch proves authority through PostgreSQL table privileges (research_authority.py).
                raise PlatformError('research_authority_unsupported', 501, backend='mongodb')
            continuation = deepcopy(continuation)
        if document is not None:
            validate_document(document)
        token = _s(attempt_token)
        with self._lock("job:" + _s(job_id)):
            job = self._one("jobs", {"_id": _s(job_id)}, code="job_not_found")
            now = self._now()
            lease_valid = job["lease_expires_at"] is not None and job["lease_expires_at"] > now
            if job["attempt_token"] != token or job["status"] != "running" or job["cancel_requested"] or not lease_valid:
                late = {"attemptToken": token, "status": status, "result": result, "document": document}
                if late not in job["late_results"]:
                    job = self._c("jobs").find_one_and_update({"_id": job["_id"]}, {"$push": {"late_results": late}, "$set": {"updated_at": now}}, return_document=ReturnDocument.AFTER)
                if job["attempt_token"] == token and job["status"] == "running" and job["cancel_requested"]:
                    job = self._c("jobs").find_one_and_update({"_id": job["_id"]}, {"$set": {"status": "cancelled", "result": result, "updated_at": now}}, return_document=ReturnDocument.AFTER)
                return {**_wire(job), "lateResultSaved": True}
            revision = None
            advanced = False
            if document is not None and status in ("succeeded", "incomplete"):
                self._branch(job["project_id"], job["branch_id"])
                if imported_capture is not None:
                    if job["kind"] != "import_scene" or document.get("captureId") != imported_capture.get("id"):
                        raise PlatformError("import_capture_mismatch", 422)
                    image_ids = {camera["imageId"] for camera in document["cameras"]}
                    if image_ids != {image.get("assetId") for image in imported_capture.get("images", [])}:
                        raise PlatformError("import_capture_images_mismatch", 422)
                revision = self._insert_revision(job["project_id"], job["branch_id"], document, parent=job["base_revision_id"], label=job["kind"])
                if imported_capture is not None:
                    self._insert("captures", {"_id": str(imported_capture["id"]), "project_id": job["project_id"], "branch_id": job["branch_id"], "base_revision_id": job["base_revision_id"],
                                              "revision_id": revision["_id"], "request_id": job["request_id"], "request_sha256": digest(imported_capture), "target": document["target"],
                                              "images": deepcopy(imported_capture["images"]), "task": deepcopy(imported_capture["task"]), "created_at": now})
                advanced = self._advance_head(job["project_id"], job["branch_id"], job["base_revision_id"], revision["_id"])
            if continuation is not None and status in ('succeeded', 'incomplete'):
                result = deepcopy(result or {})
                branch = self._branch(job['project_id'], job['branch_id'])
                base_id = revision['_id'] if revision else job['base_revision_id']
                if branch['head_revision_id'] != base_id or revision is not None and not advanced:
                    result['continuationStopped'] = 'branch_changed'
                else:
                    child_id = str(uuid5(UUID(job['_id']), 'continuation'))
                    inputs = continuation.get('inputs', {})
                    config = {**deepcopy(job['config']), **continuation.get('config', {})}
                    request = {'requestId': child_id, 'branchId': job['branch_id'], 'baseRevisionId': base_id, 'kind': continuation['kind'], 'inputs': inputs, 'config': config}
                    self._new_job(job['project_id'], job['branch_id'], base_id, child_id, digest(request), continuation['kind'], inputs, config, job_id=child_id)
                    result['continuationJobId'] = child_id
            row = self._c("jobs").find_one_and_update({"_id": job["_id"]}, {"$set": {"status": status, "result": result, "result_revision_id": revision["_id"] if revision else None,
                                                                                    "head_advanced": advanced, "updated_at": now}}, return_document=ReturnDocument.AFTER)
            return _wire(row)

    # ---- model calls ----------------------------------------------------------------------

    def _spent(self):
        # ponytail: client-side sum over (estimated, actual) pairs; an aggregate pipeline if the ledger grows large.
        total = Decimal(0)
        for row in self._c("model_calls").find({}, {"estimated_cost": 1, "actual_cost": 1}):
            total += (row["actual_cost"] if row.get("actual_cost") is not None else row["estimated_cost"]).to_decimal()
        return total

    @staticmethod
    def _lease_valid(job, now):
        return job["lease_expires_at"] is not None and job["lease_expires_at"] > now

    def reserve_model_call(self, job_id, attempt_token, provider, model, request_key, estimated_cost, *, code_sha256=None, model_sha256=None, adapter_sha256=None, input_sha256=None, paid=True):
        estimate = Decimal(str(estimated_cost))
        if not estimate.is_finite() or estimate < 0:
            raise PlatformError("invalid_cost")
        with self._lock("model_calls"), self._lock("job:" + _s(job_id)):  # the budget lock serialises every reservation
            job = self._one("jobs", {"_id": _s(job_id)}, code="job_not_found")
            calls = self._c("model_calls")
            existing = calls.find_one({"job_id": job["_id"], "request_key": request_key})
            if existing is None:
                existing = calls.find_one({"project_id": job["project_id"], "provider": provider, "model": model, "request_key": request_key}, sort=[("created_at", 1)])
            if existing:
                raise PlatformError("model_call_already_reserved", 409, modelCallId=existing["_id"], status=existing["status"])
            if job["status"] != "running" or job["cancel_requested"] or job["attempt_token"] != _s(attempt_token) or not self._lease_valid(job, self._now()):
                raise PlatformError("stale_job_attempt", 409)
            if job['kind'] in ('analyze_capture', 'reconstruct_scene', 'review_models') or (job['kind'] == 'validate_model' and isinstance(job['config'].get('pipeline'), dict)):
                branch = self._branch(job['project_id'], job['branch_id'])
                if branch['head_revision_id'] != job['base_revision_id']:
                    raise PlatformError('pipeline_base_revision_changed', 409)
            if paid or estimate > 0:
                if self.paid_budget is None:
                    raise PlatformError("paid_budget_not_configured", 409)
                if self._spent() + estimate > self.paid_budget:
                    raise PlatformError("paid_budget_exceeded", 409)
            now = self._now()
            row = {"_id": str(uuid4()), "project_id": job["project_id"], "job_id": job["_id"], "attempt_token": _s(attempt_token), "provider": provider, "model": model, "request_key": request_key,
                   "code_sha256": code_sha256, "model_sha256": model_sha256, "adapter_sha256": adapter_sha256, "input_sha256": input_sha256, "status": "reserved",
                   "estimated_cost": Decimal128(estimate), "actual_cost": None, "response": None, "created_at": now, "updated_at": now}
            return _wire(self._insert("model_calls", row))

    def record_model_call_dispatch(self, call_id, attempt_token, provider_request_id):
        if not isinstance(provider_request_id, str) or not provider_request_id.strip():
            raise PlatformError('invalid_provider_request_id')
        row = self._one("model_calls", {"_id": _s(call_id)}, code='model_call_not_found')
        with self._lock("job:" + row["job_id"]):
            job = self._one("jobs", {"_id": row["job_id"]}, code='model_call_not_found')
            row = self._one("model_calls", {"_id": _s(call_id)}, code='model_call_not_found')
            if (job['status'] != 'running' or job['cancel_requested'] or not self._lease_valid(job, self._now())
                    or job['attempt_token'] != _s(attempt_token) or row['attempt_token'] != _s(attempt_token)
                    or row['status'] != 'reserved'):
                raise PlatformError('stale_job_attempt', 409)
            response = row['response'] or {}
            if response.get('providerRequestId'):
                if response['providerRequestId'] != provider_request_id:
                    raise PlatformError('model_dispatch_conflict', 409)
                return _wire(row)
            return _wire(self._c("model_calls").find_one_and_update({"_id": row["_id"]}, {"$set": {"response": {**response, 'providerRequestId': provider_request_id}, "updated_at": self._now()}},
                                                                    return_document=ReturnDocument.AFTER))

    def complete_model_call(self, call_id, status, actual_cost=None, response=None):
        if status not in ("succeeded", "failed", "outcome_unknown"):
            raise PlatformError("invalid_model_outcome")
        if actual_cost is not None and (not Decimal(str(actual_cost)).is_finite() or Decimal(str(actual_cost)) < 0):
            raise PlatformError("invalid_cost")
        calls = self._c("model_calls")
        def update(fields):
            return _wire(calls.find_one_and_update({"_id": row["_id"]}, {"$set": {**fields, "updated_at": self._now()}}, return_document=ReturnDocument.AFTER))
        with self._lock("call:" + _s(call_id)):
            row = self._one("model_calls", {"_id": _s(call_id)}, code="model_call_not_found")
            previous = row['response'] or {}
            request_id = previous.get('providerRequestId')
            if request_id:
                if (response or {}).get('providerRequestId') not in (None, request_id):
                    raise PlatformError('model_dispatch_conflict', 409)
                response = {**(response or {}), 'providerRequestId': request_id}
            else:
                request_id = (response or {}).get('providerRequestId')
            if row["status"] != "reserved":
                if row["status"] == "outcome_unknown" and status in ("succeeded", "failed"):
                    late = {"status": status, "actualCost": str(Decimal(str(actual_cost)).normalize()) if actual_cost is not None else None, "response": response}
                    if isinstance(previous, dict) and previous.get("lateOutcome"):
                        if previous["lateOutcome"] != late:
                            raise PlatformError("model_outcome_conflict", 409)
                        return _wire(row)
                    return update({"actual_cost": _cost(actual_cost) if actual_cost is not None else row["actual_cost"],
                                   "response": {**({'providerRequestId': request_id} if request_id else {}), "previousResponse": previous, "lateOutcome": late}})
                if row["status"] != status:
                    raise PlatformError("model_outcome_conflict", 409)
                if status == 'outcome_unknown' and request_id and not previous.get('providerRequestId'):
                    # Recovery may expire the reservation while spawn returns its ID.
                    return update({"response": {**previous, 'providerRequestId': request_id}})
                return _wire(row)
            return update({"status": status, "actual_cost": _cost(actual_cost), "response": {**previous, **(response or {})} if previous else response})

    # ---- publications ---------------------------------------------------------------------

    def _ancestry(self, project_id, revision, *, stop_at=None):
        """Revision ids from `revision` to the root (or to `stop_at`, inclusive); PostgreSQL's recursive CTE."""
        ids, node = [], revision
        while node is not None:
            ids.append(node["_id"])
            if node["_id"] == stop_at or not node["parent_revision_id"] or node["parent_revision_id"] in ids:
                break
            node = self._c("scene_revisions").find_one({"_id": node["parent_revision_id"], "project_id": _s(project_id)}, {"_id": 1, "parent_revision_id": 1})
        return ids

    def create_publication(self, project_id, capability, body, *, evaluations=None, reviews=None):
        self._auth(project_id, capability)
        pid = _s(project_id)
        with self._idempotent("publications", project_id, body) as previous:
            if previous:
                return _wire(previous)
            revision = self._revision(project_id, body["sceneRevisionId"])
            if (body.get("evaluationIds") and evaluations is None) or (body.get("reviewIds") and reviews is None):
                raise PlatformError("publication_evidence_unverified", 422)
            branch = self._one("scene_branches", {"project_id": pid, "_id": revision["branch_id"]})
            ancestry = self._ancestry(project_id, revision)
            by_revision = defaultdict(list)
            for batch in self._c("edit_batches").find({"revision_id": {"$in": ancestry}}).sort([("created_at", 1), ("_id", 1)]):
                by_revision[batch["revision_id"]].append(batch)
            edits = [batch for rid in reversed(ancestry) for batch in by_revision.get(rid, [])]
            scene_version = revision["document"]["schemaVersion"]
            snapshot = {"schemaVersion": 1, "reportSchemaVersion": scene_version, "rendererVersion": f"native-webgl-v{scene_version}", "revision": _wire(revision), "evaluations": evaluations or [], "reviews": reviews or [],
                        "editBatches": _wire(edits), "branchKind": branch["kind"], "branchTitle": branch["title"]}
            jobs = list(self._c("jobs").find({"project_id": pid, "$or": [{"base_revision_id": revision["_id"]}, {"result_revision_id": revision["_id"]}]}).sort([("created_at", 1), ("_id", 1)]))
            # The public DTO excludes executor references, attempt tokens and late
            # mutable outcomes. Config contains the server's frozen, secret-free pins.
            snapshot["jobs"] = [Job.model_validate({key: value for key, value in _wire(job).items() if key in Job.model_fields}).model_dump(mode="json") for job in jobs]
            if branch["kind"] == "planning" and branch["source_revision_id"]:
                snapshot["reconstructionRevision"] = _wire(self._revision(project_id, branch["source_revision_id"]))
            reconstruction = snapshot.get("reconstructionRevision", snapshot["revision"])
            snapshot["playgroundDefinitions"] = [{"kind": "observed", "revisionId": reconstruction["id"]}, {"kind": "model", "revisionId": reconstruction["id"]}]
            if branch["kind"] == "planning":
                snapshot["playgroundDefinitions"].append({"kind": "planning", "revisionId": revision["_id"]})
            scene_asset_ids = {asset["id"] for pinned in (snapshot["revision"], reconstruction) for asset in pinned["document"]["assets"]}
            output_refs = [ref for job in snapshot["jobs"] for ref in _job_asset_references(job["result"])]
            try:
                output_ids = {str(UUID(identity)) for identity, _ in output_refs}
            except (ValueError, TypeError, AttributeError):
                raise PlatformError("publication_job_asset_invalid", 422) from None
            asset_ids = sorted(scene_asset_ids | output_ids)
            assets = list(self._c("assets").find({"_id": {"$in": asset_ids}}).sort("_id", 1))
            if len(assets) != len(asset_ids):
                raise PlatformError("publication_asset_not_found", 422)
            assets_by_id = {asset["_id"]: _wire(asset) for asset in assets}
            for identity, ref in output_refs:
                asset = assets_by_id[str(UUID(identity))]
                if asset["projectId"] != pid and asset["id"] not in scene_asset_ids:
                    raise PlatformError("publication_job_asset_forbidden", 403)
                if any(key in ref and ref[key] != asset[key] for key in ("sha256", "sizeBytes", "mediaType", "storageKey", "projectId", "jobId")):
                    raise PlatformError("publication_job_asset_integrity_conflict", 422)
            snapshot["assetManifest"] = []
            for asset in assets:
                if self.blobs is None:
                    raise PlatformError("blob_store_not_configured", 503)
                self.blobs.get(asset["storage_key"], asset["sha256"].strip(), asset["size_bytes"])
                snapshot["assetManifest"].append({"assetId": asset["_id"], "sha256": asset["sha256"].strip(), "sizeBytes": asset["size_bytes"], "mediaType": asset["media_type"]})
            row = {"_id": str(uuid4()), "project_id": pid, "scene_revision_id": _s(body["sceneRevisionId"]), "request_id": str(body["requestId"]), "request_sha256": digest(body), "title": body["title"],
                   "evaluation_ids": deepcopy(body.get("evaluationIds", [])), "review_ids": deepcopy(body.get("reviewIds", [])), "snapshot": snapshot, "created_at": self._now()}
            return _wire(self._insert("publications", row))

    def list_publications(self):
        # Project only the counted fields: the immutable document is never sent whole for a library listing.
        projection = {"project_id": 1, "scene_revision_id": 1, "title": 1, "created_at": 1, "snapshot.revision.document.assets.id": 1, "snapshot.revision.document.assets.kind": 1,
                      "snapshot.revision.document.entities.sourceContext": 1, "snapshot.revision.document.entities.representations.kind": 1}
        items = []
        for row in self._c("publications").find({}, projection).sort([("created_at", -1), ("_id", 1)]).limit(200):
            document = row.get("snapshot", {}).get("revision", {}).get("document", {})
            images = [image for image in document.get("assets", []) if image.get("kind") == "source_image"]
            objects = [entity for entity in document.get("entities", []) if entity.get("sourceContext") is not True]
            kinds = [{rep.get("kind") for rep in entity.get("representations", [])} for entity in objects]
            items.append({"id": row["_id"], "project_id": row["project_id"], "scene_revision_id": row["scene_revision_id"], "title": row["title"], "created_at": row["created_at"],
                          "preview_image_asset_id": images[0].get("id") if images else None, "photo_count": len(images), "object_count": len(objects),
                          "spatial_object_count": sum(1 for entity in objects if entity.get("representations")),
                          "model_object_count": sum(1 for found in kinds if found & {"generated_mesh", "primitive"}),
                          "observed_surface_object_count": sum(1 for found in kinds if "observed_surface" in found)})
        return {"items": _wire(items)}

    def get_publication(self, publication_id):
        return _wire(self._one("publications", {"_id": _s(publication_id)}, code="publication_not_found"))

    # ---- agent turns ----------------------------------------------------------------------

    def create_agent_turn(self, project_id, capability, body):
        self._auth(project_id, capability)
        pid = _s(project_id)
        with self._idempotent("agent_turns", project_id, body) as previous:
            if previous:
                return _wire(previous)
            self._branch(project_id, body["branchId"], body["baseRevisionId"])
            document = self._revision(project_id, body["baseRevisionId"])["document"]
            if body.get("imageId"):
                image = next((asset for asset in document["assets"] if asset["id"] == str(body["imageId"])), None)
                if image is None:
                    raise PlatformError("agent_image_scope_not_found", 422)
                asset = self._one("assets", {"_id": str(body["imageId"])}, code="agent_image_scope_not_found")
                if not asset["media_type"].startswith("image/"):
                    raise PlatformError("agent_image_scope_not_found", 422)
            if bool(body.get("policyId")) != bool(body.get("policyRevisionId")):
                raise PlatformError("agent_policy_scope_invalid", 422)
            if body.get("policyId"):
                self._one("policy_revisions", {"project_id": pid, "policy_id": _s(body["policyId"]), "_id": _s(body["policyRevisionId"])}, code="agent_policy_scope_not_found")
            for field, collection in (("entityId", "entities"), ("observationId", "observations")):
                if body.get(field) and body[field] not in {x["id"] for x in document[collection]}:
                    raise PlatformError("agent_scope_not_found", 422, field=field)
            pair = body.get("identityEntityIds")
            if pair is not None:
                entity_ids = {e["id"] for e in document["entities"] if not e.get("sourceContext")}
                if not isinstance(pair, (list, tuple)) or len(pair) != 2 or not all(isinstance(eid, str) for eid in pair) or pair[0] == pair[1] or not set(pair) <= entity_ids or (body.get("entityId") and body["entityId"] not in pair):
                    raise PlatformError("agent_identity_scope_invalid", 422)
            turn_id, job_id = str(uuid4()), str(uuid4())
            job_body = {"requestId": body["requestId"], "branchId": body["branchId"], "baseRevisionId": body["baseRevisionId"], "kind": "agent_turn", "inputs": {"turnId": turn_id}, "config": {}}
            self._new_job(project_id, body["branchId"], body["baseRevisionId"], body["requestId"], digest(job_body), "agent_turn", job_body["inputs"], self._job_config({}), job_id=job_id)
            now = self._now()
            turn = {"_id": turn_id, "project_id": pid, "conversation_id": _s(body["conversationId"]), "branch_id": _s(body["branchId"]), "base_revision_id": _s(body["baseRevisionId"]),
                    "request_id": str(body["requestId"]), "request_sha256": digest(body), "request": deepcopy(body), "response": None, "status": "pending",
                    "created_at": now, "updated_at": now, "job_id": job_id, "sequence": self._sequence("agent_turns")}
            return _wire(self._insert("agent_turns", turn))

    def claim_agent_turn(self, turn_id):
        row = self._c("agent_turns").find_one_and_update({"_id": _s(turn_id), "status": "pending"}, {"$set": {"status": "running", "updated_at": self._now()}}, return_document=ReturnDocument.AFTER)
        if row is None:
            raise PlatformError("agent_turn_not_claimable", 409)
        return _wire(row)

    def list_agent_turns(self, project_id, *, conversation_id=None, after_sequence=0):
        pid = _s(project_id)
        self._one("projects", {"_id": pid}, code="project_not_found")
        query = {"project_id": pid, "sequence": {"$gt": after_sequence}}
        if conversation_id is not None:
            query["conversation_id"] = _s(conversation_id)
        rows = list(self._c("agent_turns").find(query).sort("sequence", 1).limit(500))
        ids = [row["_id"] for row in rows]
        applied = {}
        for table, key in (("edit_batches", "applied_edit_batch_id"), ("policy_revisions", "applied_policy_revision_id")):
            for link in self._c(table).find({"project_id": pid, "agent_turn_id": {"$in": ids}}, {"_id": 1, "agent_turn_id": 1}).sort([("created_at", 1), ("_id", 1)]):
                applied.setdefault((key, link["agent_turn_id"]), link["_id"])
        return {"items": _wire([{**row, "applied_edit_batch_id": applied.get(("applied_edit_batch_id", row["_id"])), "applied_policy_revision_id": applied.get(("applied_policy_revision_id", row["_id"]))} for row in rows])}

    def finish_agent_turn(self, turn_id, status, response):
        if status not in ("succeeded", "failed", "outcome_unknown"):
            raise PlatformError("invalid_agent_outcome")
        row = self._c("agent_turns").find_one_and_update({"_id": _s(turn_id), "status": "running"}, {"$set": {"status": status, "response": deepcopy(response), "updated_at": self._now()}},
                                                         return_document=ReturnDocument.AFTER)
        if row is None:
            raise PlatformError("agent_outcome_conflict", 409)
        return _wire(row)
