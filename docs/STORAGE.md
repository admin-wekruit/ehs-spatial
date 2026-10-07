# Storage backends

The platform stores records through one `Repository` / `PolicyRepository` protocol
(`ehs_spatial/platform/repository.py`) and bytes through one `BlobStore` protocol
(`ehs_spatial/platform/storage.py`). The backend is chosen by URL and nothing else
(`ehs_spatial/platform/config.py`, `ehs_spatial/platform/runtime.py`).

| Key | Value | Backend |
|---|---|---|
| `PANOPTES_DATABASE_URL` | `postgres://…` / `postgresql://…` | `PostgresRepository` + `PostgresPolicyRepository` (`postgres.py`, `policy_repository.py`) |
| `PANOPTES_DATABASE_URL` | `mongodb://…/<db>` / `mongodb+srv://…/<db>` | `MongoRepository` + `MongoPolicyRepository` (`mongo.py`, `mongo_policy.py`); database name from the URL path, default `panoptes` |
| `PANOPTES_MONGO_PREFIX` | collection name prefix, default `panoptes_` | Mongo |
| `PANOPTES_BLOB_ROOT` | a directory | `LocalBlobStore` |
| `PANOPTES_BLOB_ROOT` | `s3://bucket[/prefix]` | `S3BlobStore` (`s3_storage.py`) |
| `AWS_ENDPOINT_URL` (or `PANOPTES_S3_ENDPOINT_URL`) | S3-compatible endpoint (MinIO, Supabase Storage, …); unset for AWS | S3 |
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION` (or `PANOPTES_S3_REGION`, default `us-east-1`) | standard boto3 credentials | S3 |
| `PANOPTES_S3_URL_EXPIRY_S` | presigned GET lifetime in seconds, default `3600` | S3 |
| `PANOPTES_BLOB_PUBLIC_BASE` | if set, `url()` / `public_url()` return `<base>/<key>` instead of a presigned URL (static site hosting / CDN in front of the bucket) | S3 |
| `PANOPTES_PAID_BUDGET_USD` | paid model-call budget; unset disables paid execution | both |

`PANOPTES_BLOB_BACKEND` and `PANOPTES_S3_BUCKET` are retired; `PANOPTES_BLOB_BACKEND=s3`
without an `s3://` root fails at startup with the replacement spelled out.

Code that needs the policy repository for a repository goes through
`ehs_spatial.platform.runtime.policy_repository(repo)`; `services()` wires repository and
blob store from the environment; `repository_from_url()` / `blob_store()` do one each.

## MongoDB

One collection per PostgreSQL table, documents keep the SQL column names (`_id` holds the
`id`; JSON columns are stored as BSON sub-documents, timestamps as BSON dates in UTC, costs
as `Decimal128`). With the default prefix:

```
panoptes_projects          panoptes_scene_branches     panoptes_scene_revisions
panoptes_edit_batches      panoptes_captures           panoptes_assets
panoptes_jobs              panoptes_model_calls        panoptes_publications
panoptes_agent_turns       panoptes_policy_sources     panoptes_policies
panoptes_policy_revisions  panoptes_policy_evaluations panoptes_policy_reviews
panoptes_policy_evidence_requests
panoptes_locks             panoptes_counters           (internal: lease locks, agent_turns.sequence)
```

`MongoRepository.migrate()` creates the indexes (idempotent; needs `createIndex` on the
database). Unique ones mirror the SQL `UNIQUE` constraints:

| Collection | Unique | Other |
|---|---|---|
| projects | `capability_sha256`; `capability_sha256,request_id` | `created_at` |
| scene_branches | `project_id,request_id` (partial: string request_id; the default branch has none) | `project_id,created_at` |
| captures, edit_batches, jobs, publications, agent_turns, policies, policy_revisions, policy_evaluations, policy_reviews, policy_evidence_requests | `project_id,request_id` (request-id idempotency) | `project_id,created_at` |
| assets | `project_id,storage_key` | `project_id,metadata.kind,metadata.cacheKey` (stage cache) |
| model_calls | `job_id,request_key` | `project_id,provider,model,request_key`; `job_id,status` |
| policy_revisions | `agent_turn_id` (partial: string) | `policy_id,created_at` |
| scene_revisions | – | `project_id,created_at`; `project_id,parent_revision_id` |
| edit_batches | – | `revision_id`; `project_id,agent_turn_id` |
| jobs | – | `status,created_at` (dispatch outbox); `project_id,base_revision_id`; `project_id,result_revision_id`; `lease_expires_at` |
| agent_turns | – | `project_id,sequence`; `job_id` |
| locks | – | `expires_at` |

### Atomicity without multi-document transactions

A standalone `mongod` (no replica set) has no multi-document transactions, so every
invariant is a single-document conditional write, plus a lease lock where PostgreSQL
serialises callers:

- **Branch head** – every revision writer (`commit_edits`, `create_capture`, `finish_job`)
  inserts the revision, then compare-and-sets `scene_branches.head_revision_id` from the
  base revision to the new one (`update_one` with the base in the filter). A lost CAS
  deletes the orphan revision and raises `revision_conflict` with the current head
  (`finish_job` records `headAdvanced=false` instead, like PostgreSQL).
- **Request-id idempotency** – `(project_id, request_id)` unique indexes, and a lock on
  `<project>:<requestId>` for the read-then-write window so a simultaneous lost-response
  retry waits and replays the first result (PostgreSQL: `pg_advisory_xact_lock`).
- **Job leases** – `claim_job` / `heartbeat_job` are one conditional `find_one_and_update`
  each (status, cancel flag, attempt token, `lease_expires_at > now`). `finish_job`,
  `cancel_job`, `reserve_model_call`, `record_model_call_dispatch` hold `job:<id>` (the
  `FOR UPDATE` row lock); `recover_expired_jobs` tries that lock without waiting
  (`SKIP LOCKED`) and flips expired jobs with a conditional update.
- **Paid budget** – `reserve_model_call` holds the global `model_calls` lock while it sums
  the ledger (PostgreSQL advisory lock 728611937).
- **Policies** – draft revisions compare-and-set `policies.draft_revision_id`; activation
  and evidence fulfilment hold `policy:<id>` / `evidence:<id>`.
- **Publications** – unique `(project_id, request_id)` plus the request lock.

Locks live in `<prefix>locks` as `{_id, expires_at}` with a 60 s lease, acquired with an
upsert that fails on a live holder; a crashed holder is overtaken when the lease expires.
Nothing waits longer than 60 s (`lock_timeout`, HTTP 503).

### Differences from PostgreSQL

- No immutability triggers: the application never updates revisions, batches, captures,
  assets or publications, but the database does not enforce it.
- `finish_job` with a `validate_model` continuation (frozen research dispatch proving
  authority through PostgreSQL table privileges) answers `research_authority_unsupported`
  (501); the research scripts under `scripts/research/` talk SQL and stay PostgreSQL only.
- Timestamps have millisecond resolution; `created_at` is kept strictly increasing within a
  process so listings ordered by `created_at` stay deterministic.
- BSON limits: one document (a scene revision, a publication snapshot) must stay under
  16 MB; integers must fit int64.
- The API maps `pymongo.errors.PyMongoError` to `database_unavailable` (503) like
  `psycopg.Error`; `/api/health` calls `repository.ping()`.

## S3

Keys: `<prefix>sha256/<hex>` where `<prefix>` is the URL path with a trailing slash
(`s3://assets/panoptes` → `panoptes/sha256/<hex>`; `s3://assets` → `sha256/<hex>`), the same
layout as `LocalBlobStore` under `PANOPTES_BLOB_ROOT`. Objects carry `ContentType` and
`Metadata.sha256`. Writes are content addressed: an existing valid object is never
rewritten, bytes are verified after upload, objects above 8 MB go through boto3 multipart
upload (`upload_fileobj`). Only PUT, GET, HEAD, multipart and presigned GET are used, with
path-style addressing and SigV4, so MinIO and other S3-compatible servers work.

Artifacts (run directories, publications, layers, weight mirrors) share the bucket:

```
s3://<bucket>/<prefix>/runs/<cell>/<run_id>/…      python scripts/panoptes_artifacts.py push DIR s3://bucket/prefix/runs/090/<run_id>
s3://<bucket>/<prefix>/publications/<id>/…         python scripts/panoptes_artifacts.py pull s3://bucket/prefix/publications/<id> DIR
s3://<bucket>/<prefix>/layers/<id>.json
s3://<bucket>/<prefix>/weights/<model>/<revision>/…
```

`push` writes every file under the target as `<target>/<relative path>`, skipping files whose
sha256 already matches the remote `manifest.json` (`{path: {sha256, size}}`, written last);
`pull` downloads files whose local sha256 differs and verifies each against the manifest.

## Running the tests

`tests/test_storage_backends.py` is one contract suite parametrised over backends
(`uv sync --extra dev` installs `mongomock` and `moto`):

| Backend | Always | With environment |
|---|---|---|
| MongoDB | `mongomock` in process | `PANOPTES_TEST_MONGO_URL=mongodb://localhost:27017/panoptes_test` (collections prefixed `contract_<hex>_` are dropped afterwards; `createIndex` required) |
| PostgreSQL | skipped | `PANOPTES_TEST_DATABASE_URL=postgresql://…` (a schema per test, dropped afterwards; also runs `tests/test_platform_backend.py`) |
| S3 | `moto` in process | `PANOPTES_TEST_S3_URL=s3://bucket/prefix` plus `AWS_ENDPOINT_URL`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` (objects under `prefix/contract-<hex>/` are deleted afterwards) |

```
.venv/bin/python -m pytest tests/test_storage_backends.py -q
PANOPTES_TEST_MONGO_URL=mongodb://localhost:27017/panoptes_test \
PANOPTES_TEST_S3_URL=s3://panoptes-test/contract AWS_ENDPOINT_URL=http://localhost:9000 \
AWS_ACCESS_KEY_ID=… AWS_SECRET_ACCESS_KEY=… .venv/bin/python -m pytest tests/test_storage_backends.py -q
```

The concurrency cases (parallel identical commits, competing appends) need a real server;
`mongomock` is not thread-safe and those cases skip on it.
