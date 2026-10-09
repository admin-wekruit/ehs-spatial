# Source delivery acceptance — 2026-10-08

This records the initial `2d8a419` acceptance. See [independent-review follow-up](REVIEW-FOLLOWUP-2026-10-08.md) for subsequent repairs, current checks and the old-report stage-data gap; the counts and wheel identity below are historical.

The clean delivery repository is `phase5-delivery`, branch `codex/phase5-delivery`. The retained Modal pipeline, measurement producers and viewer have passed the checks below in a fresh checkout and a newly installed locked Python environment. The independent verdict lab, frozen inputs and archived research remain outside this repository.

## Final checks

| Check | Observed result |
| --- | --- |
| Actual `make test`, fresh locked Python 3.12 environment, isolated PostgreSQL 16 UTF-8 database | **733 passed, 51 skipped**, 1 Starlette/httpx deprecation warning, 60.15 seconds |
| `npm --prefix web run check` | All five checks passed; actual producer messages, report consumers, spatial output and mounted viewer language changes exercised |
| Eight additional retained Node report/selection checks | Passed |
| `npm --prefix web run build` | TypeScript and Vite passed; existing large-chunk warning |
| Both cells' actual installed console `status` and complete `run --dry-run` | Passed |
| Six retained Modal definition modules | Imported without remote execution; SAM3D helper and patch mounts exist |
| Actual `make check-env` against a task-created configuration | 5 keys, 0 errors, 0 warnings after its writable data directory was created |
| Input mirror verification | All **1,022** files verified by SHA-256 and size; **520,797,028 bytes**; no fetch needed on the complete mirror |
| `compileall`, Ruff E9/F63/F7/F82, `scripts/check_delivery.py`, source whitespace check | Passed |
| Independent wheel installed offline into a new target; package and model cache made read-only | **16 passed** plus imports, console paths and English nested-message rendering passed |

Three inherited whitespace-only lines/endings were normalized after the final full suite; AST equality was asserted for all three files, and the final wheel was rebuilt and independently checked.

The final wheel contains 126 entries: 113 Python source files, 8 package resources and 5 metadata files. Its SHA-256 is `0ae3f4057d820307743e630338392c2b8423cc311ceb009d46dd410e213d935b`, size 658,626 bytes. Its package-source digest is `3858b96065a1b5c8ed1e656994cf2c2bfc759497262f85023f7b4672b161b23d`. The local wheel, source-version manifest and installed-check results are in sibling `phase5-artifacts`.

The fresh Python install used the unchanged `uv.lock`: a cache-only attempt stopped because the Open3D wheel was absent; the subsequent public-registry install succeeded with 130 packages. Fresh `npm ci` installed 320 packages from the existing lock. This source acceptance therefore includes network dependency installation, with no model download or GPU call.

## Physical and display regression

| Cell | Frozen publication / revision | Frame | Boxes | Saved nativeToMeters |
| --- | --- | --- | --- | --- |
| 090 | `a9a6e0a0-a77e-4ca0-b460-aa6f18d72698` / `86207849-d672-4df4-b453-22584f491894` | `3c0cfc31-267a-53db-a2b3-e5338e6ade3c` | 9 | 3.2371372068568487 |
| 030 | `fafdeb6b-7122-4434-a62d-676b3ff9e50a` / `f464b25e-9e27-4966-aca1-e21472fa2962` | `d9e3265b-74e3-54b7-84a4-cc91e69ab4db` | 8 | 3.5616493184162774 |

The actual layer builder is exercised against both frozen scenes with their recorded boxes replayed as stage inputs, and its serialized output is consumed by the actual Node loader, box/face geometry and localized message renderer. Source frame, scale, floor, object identity, box/face coordinates, dimensions, uncertainty and confidence match exactly. A separate existing regression executes the original two-scene numerical box self-check and compares its recomputed synthetic values. This does not recompute the original published field scenes. Display text changes to schema 2 do not change these physical fields.

Read-only AST comparison against the preserved source confirmed identical selection scoring/candidate construction, extent/observed-height/free-space/model-hit/coverage calculations and numerical floor/plane/size/photo gate conditions. The only removed numerical-model branch was historical RecGen. No threshold, model pin or publication scale was adjusted to force agreement.

All three locale catalogs contain **1,777 matching keys** with matching placeholders. Checks exercise 129 emitted producer messages in all three languages and a mounted native viewer through **en → zh → nl → en**, including its accessibility labels. The canonical English catalog is shared with Python diagnostics and bundled as a regular resource in the wheel. Dutch terminology still requires customer language review. No browser screenshot or customer language acceptance is claimed.

## Root causes corrected during acceptance

- The CLI resolves the selected data/run/page roots once, propagates them to children and preserves nonzero child exits. S7 uses the configured database directly.
- S4a requires a real candidate for every configured object. S4b rejects stale or incomplete selection metadata. S4c can refresh metadata only for the same candidate; a changed selection cannot relabel an old mesh. The publisher validates complete configured evidence and actual generated meshes before opening database/blob services; the layer builder also rejects missing configured comparisons or entities.
- The schema-2 producer and every retained consumer use the same code/parameter messages. Language-specific sibling prose and old-loader aliases are removed.
- Platform routes reuse the existing response DTOs. A duplicate function-local `PlatformError` import was deleted because it shadowed the shared import on a bad-source path; its test observed the original `UnboundLocalError` and now verifies the typed import error.
- A missing Modal image constant was removed from its obsolete image branch. Test isolation replaces both Python module bindings so a prior completion import cannot escape the explicit fake runner.

## Attempts and evidence limits

These are separate attempts; the final result above supersedes their counts.

| Attempt | Result and resolution |
| --- | --- |
| Early retained offline migration suite | 570 passed / 186 conditional skips; later expanded checks and real database acceptance supersede it |
| Platform checks without a database DSN | 355 passed / 121 skipped; insufficient database coverage, so an isolated real server was started |
| First temporary database, SQL_ASCII encoding | 101 failures / 12 errors; replaced with a UTF-8 database, without adding an application decoding workaround |
| Real UTF-8 platform checks | Initially 2 failed / 468 passed / 6 skipped; repaired route DTO reuse and relocated test imports, then 470 passed / 6 skipped |
| Retained combined suite with real database | 710 passed / 49 skipped / 2 failed; existing Mongo-only test assertions needed their repository type guard |
| Next combined suite | 720 passed / 51 skipped / 1 failed; repaired fake-Modal test isolation after a prior module import |
| Combined CLI/provider/cache checks | 66 passed in one process |
| First clean-checkout `make test` | 732 passed / 51 skipped; final bad-input regression and one-line import fix followed |
| Final clean-checkout `make test` | 733 passed / 51 skipped |

The final 51 skips are 6 unavailable Blender checks, 3 unavailable Torch/pinned SAM3D-source checks, 24 unavailable real MongoDB checks, 2 unavailable real S3 checks, 11 duplicate PostgreSQL replays already covered by the platform suite, 2 Mongo-specific checks on PostgreSQL, and 3 mongomock thread-safety checks. These integrations remain conditional acceptance work.

The existing npm lock reports **9 advisory-bearing packages: 7 moderate and 2 high**. The high findings are brace-expansion and source-map-js denial-of-service advisories. The raw audit is saved in [validation/npm-audit-2026-10-08.json](validation/npm-audit-2026-10-08.json). Dependencies were kept at their existing versions for this source-preservation task; exploitability and dependency remediation were not evaluated here.

No fresh GPU reconstruction, live LLM evaluation, input upload, deployment, customer Linux/Docker execution or independent physical calibration was performed. CI now declares a PostgreSQL-backed Linux source gate, but it has not been run on GitHub in this session. A successful dry run or wheel import does not establish a fresh model result.

The independent verdict lab's [RESULTS.md](../../phase4-verdict-lab/RESULTS.md) records its own offline matrix and checks. Its scenes still lack reviewer-confirmed zones, measured coverage and engineering gold; function-library compilation currently gives **0/9 bound rules**. Live Haiku remains unrun. These are research evidence gaps, not Phase 5 measurement facts.

## Reproduce the local source check

The isolated test cluster was created at `/private/tmp/panoptes-phase5-postgres/db` on port 55439, separately from the user's database. It is stopped after acceptance. Start that retained test cluster, or supply another disposable UTF-8 PostgreSQL test database, before repeating the database gate.

```sh
cd /Users/adam/Desktop/panoptes-public/phase5-delivery
LC_ALL=en_US.UTF-8 /opt/homebrew/opt/postgresql@16/bin/pg_ctl -D /private/tmp/panoptes-phase5-postgres/db \
  -l /private/tmp/panoptes-phase5-postgres/server.log \
  -o "-h 127.0.0.1 -p 55439 -k /private/tmp/panoptes-phase5-postgres/socket" start
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  UV_PROJECT_ENVIRONMENT=/private/tmp/argus-clean-env \
  UV_CACHE_DIR=/Users/adam/.cache/uv UV_OFFLINE=1 \
  PANOPTES_TEST_DATABASE_URL=postgresql://panoptes@127.0.0.1:55439/phase5test_utf8 \
  PYTEST_ADDOPTS='--basetemp=/private/tmp/argus-clean-tests-replay --tb=short -rs' \
  make test UV=/opt/homebrew/bin/uv
npm --prefix web run check
npm --prefix web run build
PYTHONDONTWRITEBYTECODE=1 /private/tmp/argus-clean-env/bin/python scripts/check_delivery.py
/opt/homebrew/opt/postgresql@16/bin/pg_ctl -D /private/tmp/panoptes-phase5-postgres/db stop -m fast
```

For a new machine use the generic setup in [HANDOFF.md](../HANDOFF.md), install from the existing locks and configure explicit storage/input settings. The local paths above document this executed acceptance only.
