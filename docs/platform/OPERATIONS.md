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
   .venv/bin/python scripts/prepare_publication_site.py \
     --catalog .platform/publication-catalog --output .platform/publication-http
   PANOPTES_PUBLICATION_CATALOG="$PWD/.platform/publication-catalog" \
     PANOPTES_PUBLICATION_HTTP="$PWD/.platform/publication-http" \
     .venv/bin/modal deploy modal_apps/publication_site.py
   ```

   Preparation verifies every frozen response and asset, then writes compressed
   HTTP bodies and an index. The immutable deployment serves those files without
   parsing historical bundles or rehashing assets on cold start. Re-run preparation
   whenever the catalog changes. The report requests `/api/publications/{id}/view`;
   complete edit events are downloaded only on demand from `/edits/{editId}`.
   Original publication and asset routes remain byte-content equivalent.

   The publication container reserves 6 GiB with four concurrent inputs. Before
   prepared serving was introduced on 2026-09-16, a
   fresh local process loading sixteen verified bundles and serving
   four simultaneous GETs of the largest 76,627,499-byte publication peaked at
   5,039,538,176 bytes RSS. All four response hashes matched. This includes local
   serialization and client buffers, but excludes Modal runtime overhead. The
   previous 3 GiB / 32-input setting had inadequate headroom. The build-time loader shares immutable revisions only after comparing
   their complete contents, and rejects conflicting records or corrupt assets.
   On macOS, the command below reports peak RSS in bytes:

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

An owned live workcell or model workbench can run **Review models** for the
selected current model. This submits the durable `review_models` job with an
explicit `entityIds` list and fixed `baseRevisionId`; it does not regenerate or
move the model. The capture pipeline also reviews retained models through this
same worker path. Parent objects are checked together with their child parts,
and that scope is shown in the result.

The result records geometry checks against the object's own saved photographs,
optional configured shape review, and CAD correspondence. Missing evidence or a
missing shape-review provider produces an incomplete result, never an approval.
Only new review evidence and derived projection metadata are written to a new
scene revision. A model/pose/source change invalidates its previous approval.
Reports display the last review with its saved scope; frozen publications never
follow a live job. The live page follows a review it started and retains the
selected object when the job advances the branch.

`PANOPTES_PROVIDER_MANIFEST` is a local JSON deployment manifest. Its reviewed, secret-free fields are frozen into each job at enqueue time. Workers read that snapshot; a changed deployment does not change an already accepted job. Each stage has provider pins and license/runtime/quality evidence. Missing or failed gates stop that stage. There is no automatic alternate model route.

Paid invocation additionally requires an explicit nonnegative `PANOPTES_PAID_BUDGET_USD`. No value means paid calls cannot begin. The agent also requires `PANOPTES_AGENT_MODEL`, `PANOPTES_AGENT_CALL_BUDGET_USD`, and its server-side provider credential. Browsing, editing, selection, rule evaluation, export and cached-result viewing do not call a VLM.

The paid budget is a cumulative ceiling against the database's `model_calls`
ledger, including estimates reserved for calls whose actual cost is unknown.
It is not a fresh allowance for each job or process restart. Set the ceiling from
the authorization for the current task, allowing separately for deployment and
other costs outside that ledger; there is no universal dollar default.
The running API and its workers must use the same authorized ceiling and ledger.
The local executor's subprocess inherits the **API process environment**: setting
variables only in a separate preparation/submission shell does not configure it.
Set them before starting the API; restart it after an environment change. Hosted
workers need the corresponding server configuration too.

RecGen dispatch also requires `PANOPTES_RECGEN_JOURNAL`, an absolute, durable,
writable directory inherited by the worker. Preserve its existing dispatch
records across restarts and reconciliation; a new directory is not a retry of an
unknown call. The current journal supports one research worker host. Multiple
hosts require moving the dispatch claim into the repository transaction before
enabling that topology. Credentials and these runtime settings stay on the
server, outside frozen report inputs.

### Prepare and submit a first geometry or depth validation

The existing `scripts.research.validate_sam3d` command also accepts explicit
`geometry` and `depth` stages. An owned capture with uploaded photos is sufficient:
its scene may have no entities, masks or meshes. Geometry selects one to four
captured image IDs; depth selects exactly one. Preparation reads the immutable
baseline revision and owned asset bytes, then freezes the selected photos,
pixel mapping, hashes, provider and runtime configuration. It accepts no caller
photo paths. Omitting `imageIds` selects all captured photos, subject to the same
stage limits. Generation retains its entity, mask and geometry requirements.

Set `PANOPTES_PROVIDER_MANIFEST` and `PANOPTES_MODEL_RUNTIME_MANIFEST` to reviewed,
secret-free JSON files. The selected provider must be paid, have a positive
`estimatedCostUsd` within the call limit, and have license evidence bound to the
same pins. `purpose: "runtime_validation"` permits the first measured run with
runtime and quality evidence still `unverified`; it does not manufacture those
receipts. Quality validation additionally requires runtime evidence, and product
execution requires all release gates. Document the scope and dependency limits
of actual license review in its artifact.

Example protocol template (replace the illustrative UUIDs with an owned capture's
project, branch, revision and image IDs; the $1 limit is illustrative and must be
separately authorized):

```json
{
  "id": "geometry-runtime-validation-001",
  "stage": "geometry",
  "purpose": "runtime_validation",
  "projectId": "00000000-0000-4000-8000-000000000001",
  "branchId": "00000000-0000-4000-8000-000000000002",
  "baselineRevision": "00000000-0000-4000-8000-000000000003",
  "imageIds": ["00000000-0000-4000-8000-000000000004"],
  "metricDefinitions": {"sourceGrid": "Returned image IDs and raster mappings correspond to every frozen input photo."},
  "policyThresholds": {},
  "split": "runtime-validation",
  "callLimits": {"maxCalls": 1, "maxCostPerCallUsd": 1, "maxTotalCostUsd": 1}
}
```

An immutable Modal image can supply the geometry/depth runtime without rebuilding
an already verified image. For example, the following runtime template uses
public Apache MapAnything pins; replace the illustrative image ID and adapter
hash with the audited image and `sha256` of `modal_apps/platform_models.py` being
deployed. Its installed `mapanything` distribution must record the
exact `codeRevision` in `direct_url.json`, not just a matching package version:

```json
{
  "geometry": {
    "pins": {
      "model": "facebook/map-anything-apache",
      "modelRevision": "00f9c245bbcb60522d1ed7f9e9d88462c6e3f38a",
      "codeRevision": "3d10cf7a3016fc0f9bb13a071ee66c47b10be0d9",
      "adapter": "capture-v1-per-image-cache"
    },
    "modalImageId": "im-AAAAAAAAAAAAAAAAAAAAAA",
    "distribution": "mapanything",
    "adapterSourceSha256": "0000000000000000000000000000000000000000000000000000000000000000"
  }
}
```

Use exactly one of `modalImageId` (`im-` followed by 22 alphanumeric characters)
or `runtimeImage` (a registry reference ending in `@sha256:` plus 64 hex digits).
The geometry provider route is `panoptes-platform-models` / `MapAnythingApache` /
`run`; depth uses `MoGe3` / `run`, distribution `moge`, model
`Ruicheng/moge-3-vitl` and its reviewed immutable code/weight revisions. Provider
pins must equal runtime pins. Deploy `modal_apps/platform_models.py` with that
runtime configuration. Each research invocation binds its frozen runtime hash to
the deployed configuration and checks the returned hash and pins. Geometry/depth
also require the exact 64-hex `adapterSourceSha256`; workers compare it with their
actual adapter file bytes before model loading and inference. Recompute that
hash and prepare a new validation envelope after changing adapter code.

```sh
.venv/bin/python -m scripts.research.validate_sam3d --prepare \
  --protocol .platform/research/protocol.json \
  --output .platform/research/prepared.json
.venv/bin/python -m scripts.research.validate_sam3d \
  --submit .platform/research/prepared.json
```

`--prepare` reads the configured database and blobs and writes a new local file;
it performs no provider calls, reservations or scene writes. `--submit` rechecks
database authority, source bytes and cumulative budget, then registers the frozen
input and enqueues one durable `validate_model` job. Submission itself does not
invoke a model; the normal dispatcher may start the worker immediately. The
worker reserves the call in the shared ledger before invocation. Repeating an
identical submission reuses the job; changed inputs under its ID are rejected.
Results are research artifacts with `scope: "research_only"`,
`productReleaseStatus: "not_changed"` and no scene revision. A successful stage
does not establish complete fresh-photo reconstruction or approve product release.

### Capture-to-model continuation

`analyze_capture` now uses `run_capture_pipeline`. It retains discovery, geometry,
masks and stable entity IDs, freezes explicit modeling targets and runs generation
and per-view quality assessment. Only geometrically consistent candidates with a
passing independent `model_review` are activated. A pass never confirms metric
scale, hidden geometry or physical placement. Missing review configuration keeps
the candidate and its exact `needs_information` reason.

The optional `model_review` provider uses the existing Gemini adapter and charged
call ledger. Its reviewed release evidence and pinned model are required just as
for discovery. The reviewer receives source photos, masks and actual rendered
candidate meshes; numerical silhouette agreement alone cannot accept a shape.

Discovery and model review use the pinned adapters
`gemini-bounded-discovery-v2` and `gemini-bounded-model-review-v3`, respectively,
with `provider: "gemini"`, `model: "gemini-3.5-flash"` and a reservation of at
least USD0.10 per call. The same complete native request is sent to CountTokens
and GenerateContent, including every image and the response schema. Inputs above
16,384 tokens are rejected; combined thinking/output is capped at 8,192 tokens.
No views are silently removed and no automatic generation retry is permitted.
Returned ID, usage and request-count evidence remain with the stage artifact.
A failed preflight conservatively retains its reservation; it is not treated as
permission to spend the same budget again. Known request/count/limit failures are
terminal failures with allowlisted phase, code, request hash and token counts;
they record `generationAttempted: false`. Transport failure after generation
starts remains unknown. No provider exception text enters these diagnostics.

Every normal capture reviews each photo's discovered inventory before
segmentation through the same `model_review` provider in inventory mode. The
review receives the original image and an immutable snapshot of owned observation
IDs, revisions, boxes and evidence. It can add visibly distinct missed instances
through the shared discovery admission path, or retain unresolved regions. It
cannot rename or merge existing identities. Review additions continue through
the ordinary segmentation, association, geometry, generation and CAD path.
The model prompt includes every observation's ID, semantic evidence and box plus
the original image dimensions. Repeated source references and pixel mappings stay
in the frozen input, outside the model prompt; no object or view is truncated.

`document.inventoryReview` retains one historical review per source image. A
review is `assessed` or `needs_information`, scoped to visible content with
`certainty: not_ground_truth`; it is not proof of hidden-object completeness.
Failed discovery/review and unresolved regions keep analysis incomplete while
valid independent observations remain available. Appending photos preserves prior
review evidence and unresolved qualifications. An unknown paid outcome stops
subsequent paid stages; cached replay uses the same source-bound input.

When analysis continues into per-object reconstruction, its original result is
carried in server-owned `captureAnalysis` through every successor. A later model
success cannot erase an earlier photo's analysis failure or a retained model's
failed review; the final result keeps their errors, stage records and review
evidence, including when the last generation fails without another successor.
The handoff does not inherit the temporary `modeling_pending` status, so a valid
analysis and review can still finish successfully after generation.
Client configuration cannot supply or replace this context.
Publications collect the declared nested analysis/review/generation artifacts,
checkpoints and prepared-input references, with the existing project ownership,
hash and blob verification. Arbitrary configuration and error strings are not
interpreted as asset references.

Owned-photo research preparation also accepts `stage: "discovery"` with
`purpose: "runtime_validation"` and exactly one captured `imageIds` entry. It
requires no prior objects or geometry and rejects arbitrary replacement payloads.
The runtime manifest binds both local adapter file hashes and google-genai
2.11.0. Empty or malformed discovery remains an incomplete artifact, never a
completed empty workcell. Runtime validation cannot enable a quality gate.

The same research entry accepts `stage: "model_review"`, `mode: "inventory"`,
`purpose: "runtime_validation"` and exactly one owned image. It freezes the
baseline scene's saved inventory, rederives it before dispatch, and admits the
review only on a disposable scene copy. It does not substitute an unattached
discovery artifact for saved observations or advance the branch.

Inventory admission failures retain the raw output and a typed rejection reason.
Coverage mismatches identify missing owned observations and count unexpected or
duplicate IDs; they never fill in the model's missing answers. The same diagnostics
flow through ordinary analysis and research validation.

For explicitly configured noncommercial RecGen research, set the server-only
`PANOPTES_RESEARCH_PREPARATION` path to a JSON record containing `authority`,
`budgetAtPreparation`, `runtimeManifest`, `callLimits`, `metricDefinitions`,
`policyThresholds`, `split` and `purpose` (`maskErosionEnabled` is optional).
Use actual audited records from research preparation; configuration is not proof
that the model has passed quality gates. This setting is frozen into new jobs and
cannot be supplied through public job configuration. Live database authority,
source hashes and remaining budget are rechecked when the research child queues.

The analysis revision commits before `reconstruct_scene` prepares its immutable
input. `validate_model` continues to return research artifacts only. A subsequent
attachment worker checks the exact protocol/cache graph, reviews the candidate,
and commits a new scene revision before preparing the next entity. Each successor
is inserted in the same transaction as its parent result. Duplicate execution
does not create a second successor; branch changes, cancellation and unknown paid
outcomes stop continuation. The result identifies `continuationJobId` or
`continuationStopped` so a partial pipeline cannot look like completed modeling.

The correspondence audit separates current source metadata, model availability,
CAD validation and shape quality. At the end of the capture chain, the worker
validates CAD using hash-verified mesh bytes and the same triangle projection
function that creates its contours. It compares geometry, frame, pose, source
observations and holes, keeping model and observed CAD coverage separate.
Missing floor reference, stale sources, missing models and unreviewed shapes
remain explicit incomplete results. A CAD cache's presence is not validation.
Accepted shape review is bound to the saved mesh hash, canonical pose, entity,
complete observation set and source camera/image/mask/geometry hashes. Changing
any reviewed input invalidates `qualityCurrent`; a historical `accepted` status
alone cannot complete a resumed job. Ground observations remain reference
surfaces and do not trigger object generation.
Synthetic transport tests verify the durable path without incurring inference
cost; real model and new-photo acceptance remain separate release gates.

Run `PYTHONPATH=. .venv/bin/python scripts/preflight_sam3d.py --runtime-manifest docs/platform/sam3d-runtime.example.json --provider-manifest docs/platform/sam3d-provider.example.json` to check the SAM3D configuration without loading a model or reserving work. The examples pin the official code and weight revisions inspected on 2026-09-15; unresolved values are null and all release checks remain unverified. Exit 1 lists every unmet configuration gate. This preflight never establishes runtime quality or approves evidence itself.

The SAM3D runtime image must be built and audited before recording its registry content digest in `PANOPTES_MODEL_RUNTIME_MANIFEST`. The official [setup](https://github.com/facebookresearch/sam-3d-objects/blob/f91db411c50efee93d8db7aeb323885650f6f722/doc/setup.md) requires Linux and an NVIDIA GPU with at least 32 GB VRAM. Code and checkpoints use the custom [SAM License](https://github.com/facebookresearch/sam-3d-objects/blob/f91db411c50efee93d8db7aeb323885650f6f722/LICENSE). A Modal `huggingface` secret alone does not prove access to the manually gated checkpoint repository. Record actual access and mesh-only dependency checks, official posed-mesh agreement, external-pointmap/no-internal-depth execution, and quality results against the same pins. Fill the existing provider evidence fields only with their resulting artifact hashes; then deploy `modal_apps/platform_models.py`. An explicitly authorized budget and per-call reservation remain required before inference. Historical RecGen services and their non-commercial checkpoints are not the SAM3D deployment.

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
