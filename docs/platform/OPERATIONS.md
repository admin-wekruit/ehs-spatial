# Platform operation and verification

The new product lives in `ehs_spatial/platform`, `panoptes_worker`, and `web`.
Historical HTML/report routes are archives; they do not write platform projects.

## Run locally

1. Install locked dependencies with `uv sync --extra dev`.
2. In `web`, run `npm ci`, then `npm run build`.
3. Set `PANOPTES_DATABASE_URL` to a PostgreSQL database; the startup migration uses an advisory transaction lock.
4. Set `PANOPTES_BLOB_ROOT` to an absolute directory; `PANOPTES_BLOB_BACKEND=local` and `PANOPTES_EXECUTOR_BACKEND=local` select the local adapters.
5. Run `.venv/bin/uvicorn ehs_spatial.platform.runtime:application --factory --host 127.0.0.1 --port 8792`.
6. Open `/app.html#/projects`. Vite development mode proxies `/api` to port8792.

The local executor runs `.venv/bin/python -m panoptes_worker --job-id UUID` in a subprocess. PostgreSQL holds the job, lease, immutable inputs, configuration, results, and costs. Terminating the browser does not cancel the worker.

## Hosted configuration

Use the same application and PostgreSQL SQL schema on the selected host. For Supabase Storage, set `PANOPTES_BLOB_BACKEND=s3`, `PANOPTES_S3_BUCKET`, `PANOPTES_S3_ENDPOINT_URL`, `PANOPTES_S3_REGION`, and standard boto3 service credentials. Credentials are server environment configuration, never frontend build variables or source files.

For Modal execution set `PANOPTES_EXECUTOR_BACKEND=modal`, `PANOPTES_MODAL_APP`, `PANOPTES_MODAL_FUNCTION`. `modal_apps/platform_worker.py` requires an audited immutable worker image digest and an existing named server secret. The model worker file additionally requires exact model/code pins and audited runtime images. Deploying a worker is distinct from passing model quality gates.

Host the API behind the website's `/api` path. `PANOPTES_WEB_ROOT` may select the Vite build directory on that host. A Pages build can use a configured API origin for background communication, while all navigation remains hash routes on the website. No management capability enters URLs, report JSON, or model context.

## Models and budget

`PANOPTES_PROVIDER_MANIFEST` is a local JSON deployment manifest. Its reviewed, secret-free fields are frozen into each job at enqueue time. Workers read that snapshot; a changed deployment does not change an already accepted job. Each stage has provider pins and license/runtime/quality evidence. Missing or failed gates stop that stage. There is no automatic alternate model route.

Paid invocation additionally requires an explicit nonnegative `PANOPTES_PAID_BUDGET_USD`. No value means paid calls cannot begin. The agent also requires `PANOPTES_AGENT_MODEL`, `PANOPTES_AGENT_CALL_BUDGET_USD`, and its server-side provider credential. Browsing, editing, selection, rule evaluation, export and cached-result viewing do not call a VLM.

`PANOPTES_BLENDER_EXECUTABLE` sets the worker's Blender executable. The user does not install Blender. Blender output is verified by reopening the saved file. Unsupported camera/mesh inputs fail explicitly.

## Controlled migration

1. Stop new write/job submissions at the website/API. Wait for claimed jobs, and reconcile every external call marked `outcome_unknown`; never resubmit it simply to complete migration.
2. Stop API/worker dispatch. Use PostgreSQL `pg_dump --format=custom` with credentials supplied through the environment or a protected password file.
3. Copy immutable blob keys unchanged. A provider-specific bulk copy command is acceptable; it must preserve the logical `sha256/...` keys. Do not rewrite scene documents with destination URLs.
4. Restore into a newly created empty target database with `pg_restore --exit-on-error`. A schema-scoped dump contains `CREATE SCHEMA public`; remove only the freshly created, empty target's default `public` schema before restoring that dump. Do not use this operation on an existing populated database. Set destination database/storage/executor configuration.
5. Run `python scripts/verify_platform_store.py --output verification.json` at the destination. It rehashes every blob, checks every scene hash, and validates fixed publication references.
6. Start the API on the target host. Check old report IDs, object selection, and downloads. Submit a zero-model `export_json` job against a fixed existing revision to exercise the target executor. A model consistency test is a separate evaluation.
7. Switch the website only after these checks. Never dual-write old run files and new platform tables.

## Checks

Set `PANOPTES_TEST_DATABASE_URL` to a disposable PostgreSQL database and run `pytest`. Tests use unique isolated schemas. `node --experimental-strip-types web/checks/renderer.mjs` checks native camera pixel projection, viewport fitting and mesh validation. `npm run check` in `web` checks frontend selection/transform behavior. Browser checks complement these assertions; passing unit checks does not establish model quality.

The release verification file records tested and untested requirements. A local preview or successful build must not be labelled the completed public SaaS release while any required gate is unverified.
