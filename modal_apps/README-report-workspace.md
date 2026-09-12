# Persistent report workspace

From this repository, with the existing Modal login:

```sh
.venv/bin/python -m modal run modal_apps/report_workspace_app.py
.venv/bin/python -m modal deploy modal_apps/report_workspace_app.py
```

The first command seeds only `runs/user-bor1-02` into the named persistent volume;
it refuses to overwrite existing remote files. The image contains application
source, reviewed compiled policies and the previously public workcell bundle,
never the whole checkout. Set `PANOPTES_PUBLIC_BUNDLE` to that bundle's directory
when deploying from another machine. `.env` supplies only `GEMINI_API_KEY`, `FAL_KEY` and the
mode-600 `.workspace-access.json` supplies `token`; they enter an ephemeral Modal
Secret and are never copied into the image or printed.
The seed also builds the evidence registry locally from the same source hashes
and stores it on the volume. For a seed created before that addition, use
`modal run modal_apps/report_workspace_app.py --registry-only` once; it also
refuses to overwrite an existing registry. Local source runs remain unchanged.

Open the deployed origin at `/reports`, `/reports/user-bor1-02`, and `/published/`.
The updated component report is `/reports/bor1-components-20260909`. Seed it
explicitly with `--run-id bor1-components-20260909`; other local runs are not
published. New uploads remain private to authenticated workspace sessions.
The product authentication middleware requires the access code for mutations and
private uploaded reports. SAM 3 uses the same existing fal provider as the
validated local detection run; MapAnything and MoGe use existing Modal backends.
The self-hosted SAM3 weights returned 403 with the configured account, so this is
an explicit deployment configuration, not an automatic provider fallback.
Deployment and seed validation do not invoke models.

Uploads commit staged input before `process_upload.spawn`. One durable worker
processes uploads with no automatic application retry; its result and all run
artifacts are committed to `panoptes-report-workspace-data`. The HTTP container
accepts up to 16 requests concurrently. An async lock serializes volume-backed
access from reload through the complete streamed response and final commit;
threaded volume operations finish before releasing this lock even on cancellation.
Session and bundled static-file requests bypass volume access, so a large model
stream does not hold the login behind its transfer.
Polling report APIs records terminal worker failures instead of silently leaving
an abandoned daemon-thread job. Parallel volume access would require coordinating
writes to the same report; upload jobs still run one at a time.

Checks: anonymous `/api/reports` contains only the authorized public seed;
anonymous write/private routes fail; the seed detail retains all instance
evidence; `/published/scene.json` and all linked mesh assets load. A live provider
upload must be tested separately with the access code and its cost recorded.
The local zero-provider queue check is `PYTHONPATH=. .venv/bin/python modal_apps/report_workspace_app.py`.
Run `.venv/bin/python tests/check_modal_workspace_concurrency.py` for blocked
model streams, concurrent login, serialized volume access and cancellation checks.

The real HTTP upload acceptance run on 2026-09-09 is
`b61ed56896064063bb845605b233609e`: one 3024×4032 source photo, 54 evidence
candidates, separate report/viewer URLs, and anonymous access denied. Its
392×518 canonical image was checked against the exact provider resize/crop
transform. The first SAM3 access failure and the SHA-bound reuse of completed
MapAnything geometry remain in the run's retry history. See
`outputs/cloud-workspace-qa-20260909/new-upload-final-check.json` and
`observed-floor-verification.json` for the recorded runtime checks. The upload
completed with `INSUFFICIENT_EVIDENCE`; successful execution does not turn that
assessment into a pass or establish physical calibration.

One selected candidate can be generated through
`POST /api/reports/{run_id}/generations/{candidate_id}`. The persisted job is the
idempotency boundary; repeated requests return its existing state. If a failed
candidate gains changed source evidence, the next explicit request archives its
old attempt and ledger before starting a new one; unchanged failed inputs do not
automatically spend again. A separate
durable CPU worker invokes the existing private RecGen function, with a 630-second
GPU budget, then packs the result for the shared object editor. The generator
uses the existing successful model-environment record and pinned code/weights.
Set `PANOPTES_RESEARCH_BUNDLE` when the research checkout is elsewhere. Results
live under `.generations` on the same volume and are served only through their
source report's access-controlled routes. Reads and editing transforms make no
model calls. The two seeded button outputs were verified against exact cached
input and output hashes; they did not require another paid inference.

Normal analysis and Agent corrections now also call the same CPU-only observed
scene exporter, after writing the object evidence registry. This does not start
an object-generation job. Each mask-backed observation keeps its own source
photo identity, native mesh support, and floor-relative H/W/D and orientation
readouts. Occluded extent is unknown; estimated metric scale is labeled as such.
Only explicitly supplied camera heights create `calibration.json`; an old
default value cannot establish calibration. Missing floor/geometry keeps a
reason and never invents a dimension or surface.

`observed/<revision>/` contains immutable packed scene assets, while
`observed-scene.json` selects the current revision. Report reads only use these
artifacts. API detail exposes `spatial` on the report and each candidate, and
the first playground is the observed scene. The source-aware Agent returns
persisted candidate IDs so its additions are selected after refresh. Repeating
an already-saved correction also retries its cached report refresh.

Local analysis needs `PANOPTES_RESEARCH_ROOT`, `PANOPTES_PUBLISHED_ROOT`, and
optionally `PANOPTES_RESEARCH_PYTHON`; the Modal image sets these paths already.
Checks: `pytest tests/test_observed_scene.py tests/test_measurements.py
tests/test_object_evidence.py tests/test_report_refresh.py tests/test_serve.py`
and `NODE_PATH=/path/to/node_modules node tests/check_workspace_observed.cjs`.

Modal references: [ASGI](https://modal.com/docs/guide/webhooks),
[durable jobs](https://modal.com/docs/guide/job-queue),
[volume commits/reloads](https://modal.com/docs/guide/volumes).

The canonical browser entry is the existing website:
`https://admin-wekruit.github.io/panoptes-workcell-report/reports.html`.
The Modal origin supplies APIs and protected assets, not a separate product entry.
Export the same frontend (no forked workspace implementation) before publishing Pages:

```sh
.venv/bin/python scripts/export_report_website.py /path/to/panoptes-workcell-pages --api-origin https://wekruit-livekit-agents--panoptes-report-workspace-web.modal.run --site-root https://admin-wekruit.github.io/panoptes-workcell-report/
```

Static deep links use `reports.html?run=<run_id>` with optional object, tab and
playground selection. The API permits only the configured website origin.
Access-code login returns a signed session valid for 24 hours; browser requests
send it in an Authorization header. The website stores this expiring session
locally so private report links also work in a new tab. Credentials never enter
URLs or public files. Private photos, downloads and saved report/viewer documents
are fetched with authentication and displayed as browser-local Blob URLs; frame
selection and language messages remain on the website origin.

Checks: `tests/check_site_client.cjs` exercises real separate HTTP origins;
`tests/check_workspace_site.cjs` checks history/login/upload navigation and
same-report selection with provider-free API fixtures. Run both with an installed
Playwright module on `NODE_PATH` and test the deployed Pages route separately.
