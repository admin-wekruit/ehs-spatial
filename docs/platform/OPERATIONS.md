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

## Publish a fixed report for external review

The public feedback site serves a frozen publication, independently of the local
database and workers. It uses the same React report and asset-ID contracts.
Selection, CAD, 3D, language switching, evidence, history and existing downloads
remain interactive. Per-object private feedback and explicitly shared identity
suggestions use a separate feedback store. Creating projects, applying edits,
forks and new model jobs remain in the full platform.

1. Export a selected publication using public read endpoints. Each original file
   must match its frozen hash and byte count before the export becomes visible:

   ```sh
   .venv/bin/python scripts/export_platform_publication.py \
     --api http://127.0.0.1:8792 --publication PUBLICATION_UUID \
     --output .platform/publication-catalog/PUBLICATION_UUID
   .venv/bin/python tests/check_publication_site.py \
     --catalog .platform/publication-catalog \
     --source-api http://127.0.0.1:8792
   ```

2. Deploy the independent publication service. It has no platform database or
   project credentials. Scene/publication writes are rejected. Its only write
   routes are separately scoped feedback and identity suggestions; those cannot
   modify a report. Model calls require an explicit model and reserved budget.

   ```sh
   PANOPTES_PUBLICATION_CATALOG="$PWD/.platform/publication-catalog" \
     .venv/bin/modal deploy modal_apps/publication_site.py
   ```

   The publication container reserves 3 GiB of memory. On 2026-09-15, loading
   eleven frozen bundles in a fresh local Python process peaked at
   1,467,498,496 bytes of RSS (1,399.5 MiB), down from 2,361,081,856 bytes
   (2,251.7 MiB) before sharing repeated revisions. The loader compares each
   revision's complete content before sharing its read-only object across
   snapshots and routes; conflicting content and corrupt assets still reject
   the catalog. This measurement excludes Modal's runtime and concurrent
   response serialization. Reserve that headroom and remeasure after adding
   bundles. On macOS, the command below reports peak RSS in bytes:

   ```sh
   /usr/bin/time -l .venv/bin/python -c \
     'from ehs_spatial.platform.publication_site import create_app; create_app(".platform/publication-catalog", allowed_origins=[])'
   ```

3. Build the website with `VITE_PUBLICATION_ID=PUBLICATION_UUID` and
   `VITE_API_ORIGIN` set to the returned HTTPS origin. Use a separate output path
   to leave the running local platform build unchanged:

   ```sh
   cd web
   VITE_PUBLICATION_ID=PUBLICATION_UUID VITE_API_ORIGIN=HTTPS_API_ORIGIN \
     npm run build -- --outDir ../.platform/public-web
   ```

4. Preserve the frozen v1 reader under `readers/v1/` for v1 reports. Copy the new
   build's `app.html` and `assets/` into the Pages artifact repository;
   use the same `app.html` contents for `index.html`. Commit and push its `main`
   branch, then verify the Pages deployment's exact commit and the public browser
   route. Navigation stays on the website; the API origin is background transport.

Each catalog directory contains one verified publication and its referenced
assets. Preserve existing published directories. This sharing deployment does
not constitute the full SaaS/model release.
The bundle itself is portable: `create_app(bundle_dir, allowed_origins=[...])`
can run under any ASGI host, without Modal-specific domain code. Keep exported
bundles outside Git. Exporting into an existing directory is refused.

## Models and budget

`PANOPTES_PROVIDER_MANIFEST` is a local JSON deployment manifest. Its reviewed, secret-free fields are frozen into each job at enqueue time. Workers read that snapshot; a changed deployment does not change an already accepted job. Each stage has provider pins and license/runtime/quality evidence. Missing or failed gates stop that stage. There is no automatic alternate model route.

Paid invocation additionally requires an explicit nonnegative `PANOPTES_PAID_BUDGET_USD`. No value means paid calls cannot begin. The agent also requires `PANOPTES_AGENT_MODEL`, `PANOPTES_AGENT_CALL_BUDGET_USD`, and its server-side provider credential. Browsing, editing, selection, rule evaluation, export and cached-result viewing do not call a VLM.

Run `PYTHONPATH=. .venv/bin/python scripts/preflight_sam3d.py --runtime-manifest docs/platform/sam3d-runtime.example.json --provider-manifest docs/platform/sam3d-provider.example.json` to check the SAM3D configuration without loading a model or reserving work. The examples pin the official code and weight revisions inspected on 2026-09-15; unresolved values are null and all release checks remain unverified. Exit 1 lists every unmet configuration gate. This preflight never establishes runtime quality or approves evidence itself.

The runtime image must be built and audited before recording its registry content digest in `PANOPTES_MODEL_RUNTIME_MANIFEST`. The official [setup](https://github.com/facebookresearch/sam-3d-objects/blob/f91db411c50efee93d8db7aeb323885650f6f722/doc/setup.md) requires Linux and an NVIDIA GPU with at least 32 GB VRAM. Code and checkpoints use the custom [SAM License](https://github.com/facebookresearch/sam-3d-objects/blob/f91db411c50efee93d8db7aeb323885650f6f722/LICENSE). A Modal `huggingface` secret alone does not prove access to the manually gated checkpoint repository. Record actual access and mesh-only dependency checks, official posed-mesh agreement, external-pointmap/no-internal-depth execution, and quality results against the same pins. Fill the existing provider evidence fields only with their resulting artifact hashes; then deploy `modal_apps/platform_models.py`. An explicitly authorized budget and per-call reservation remain required before inference. Historical RecGen services and their non-commercial checkpoints are not the SAM3D deployment.

Build the image with the official VCS package at the example's code revision, then run `python scripts/prepare_sam3d_mesh_source.py` inside that image before sealing it. The script resolves the installed distribution, verifies both original source hashes, and applies `modal_apps/sam3d_mesh_only.patch` once. This preserves the official mesh decoder and vertex-color/axis conversion while avoiding the unconditional Gaussian-output lookup; optional Gaussian renderer loading moves to the function that uses it. The runtime overrides depth and both Gaussian decoder configurations before Hydra construction. Its existing mesh-only flags disable texture baking and layout refinement. This does not remove or approve all dependencies in the image.

The script writes `sam3d_objects/panoptes_mesh_build.json` and reports `meshSourceBuildSha256`. The example contains the receipt hash produced by the reviewed patch; it establishes source preparation only. At startup the worker checks the installed VCS revision, receipt hash, and both patched file hashes. Preserve this receipt in the image and record the resulting image digest only after building and auditing it.

For the zero-GPU source regression, run `python scripts/prepare_sam3d_mesh_source.py --fetch --source .platform/model-delivery-20260915/sam3d-mesh-source`, then `python -m pytest tests/test_sam3d_mesh_source.py -q` in an environment with CPU PyTorch and trimesh. The test executes the exact patched upstream mesh postprocessing method with real CPU tensors, checks vertex colors/topology and the official axis round trip, and checks deferred imports. It does not establish full pipeline importability, native-pose fixture agreement, runtime quality, or weight access. `tests/test_sam3d_runtime.py` separately checks pre-construction overrides, external-pointmap forwarding, failure telemetry, and source tampering rejection.

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
