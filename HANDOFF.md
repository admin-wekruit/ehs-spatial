# Run and resume

Read `docs/STATE.md` first. This checkout delivers the existing Modal source and supports explicit HTTP provider contracts; the customer's on-prem implementation remains their adaptation.

## Setup

Use Python 3.12 and the existing locked dependencies:

```sh
make env
uv sync --frozen --extra dev
# Set .env values and an explicit input mirror; do not put HF_TOKEN in a file.
uv run --env-file .env python scripts/fetch_inputs.py
make check-env
```

`PANOPTES_DATA_ROOT` selects the writable input/output root. The input mirror is explicit (`PANOPTES_INPUT_SOURCE` or `scripts/fetch_inputs.py --source`). A mirror can be a `file://` directory, an HTTP(S) directory tree, or an existing S3 content-addressed blob root. S3 reads use the existing platform storage client and environment configuration. No default remote destination is invented and no inputs are uploaded by this command.

```sh
uv run --env-file .env python scripts/fetch_inputs.py
```

Every input is verified against `data-manifest.json` before replacement. A bad download leaves the prior destination unchanged. The current frozen set contains both workcells and their original geometry artifacts. Model weights remain in the existing Modal images/volumes.

## Verify and run

```sh
make test
uv run --env-file .env panoptes status --cell 090
uv run --env-file .env panoptes status --cell 030
uv run --env-file .env panoptes run --cell 090 --dry-run
uv run --env-file .env panoptes run --cell 030 --dry-run
```

A dry run lists commands and paths without inference or publication. Offline tests use explicit fake/offline flags and disable live benchmarks. A real run uses the configured providers, needs platform database/blob settings and authenticated Modal, and spends GPU resources:

```sh
uv run --env-file .env panoptes run --cell 090
uv run --env-file .env panoptes run --cell 030
```

The CLI supports `--from` and `--only` for its existing stages. Failed children exit nonzero even when a partial result file exists. S4a requires a real candidate for every configured object, and stale or partial S4b selection is recomputed. S4c refreshes metadata for the same selected candidate; a changed candidate requires recomputing that cell's S4c–S8 outputs. Later stages reuse their recorded result files, but publication and layer building reject incomplete configured object inventories. Use the dry-run paths to clear only the deliberately selected stage outputs before recomputation.

Measurement files are written to `PANOPTES_PAGES` (default `<data root>/measurement-layer`). Publication manifests are under `<data root>/publications/<variant>/result.json`; stage evidence is under `<data root>/swap-runs/<variant>-stages`. The formal variant names are `090-v2-mvs-fill-sam3d` and `030-v2-mvs-fill-sam3d`.

## Regression

```sh
uv run python scripts/check_measurement_regression.py tests/fixtures/measurements/090.json PATH_TO_090_LAYER
uv run python scripts/check_measurement_regression.py tests/fixtures/measurements/030.json PATH_TO_030_LAYER
```

Keep published scales and algorithm parameters unchanged. Record a numerical difference and its evidence rather than tuning the implementation to force agreement. Schema-2 display messages change prose only.

The independent verdict project has its own README, tests, configs and results. Research declarations and provisional gold labels do not become observed workcell facts merely by running a matrix.

## Local session continuation

| Artifact | Local path / branch |
| --- | --- |
| Clean source repository | `/Users/adam/Desktop/panoptes-public/phase5-delivery`, `codex/phase5-delivery` |
| Migration history and prior source | `/Users/adam/Desktop/panoptes-public/phase5-portability`, `codex/phase5-portability`, original base `e59a75e` |
| Independent verdict lab | `/Users/adam/Desktop/panoptes-public/phase4-verdict-lab`, `codex/verdict-lab` |
| Frozen input mirror | `/Users/adam/Desktop/panoptes-public/phase5-inputs` |
| Preserved research and unused source | `/Users/adam/Desktop/panoptes-public/phase5-research-reference` |
| Verified wheel, source manifest and preservation checkpoint | `/Users/adam/Desktop/panoptes-public/phase5-artifacts` |
| Newly installed locked Python environment | `/private/tmp/argus-clean-env` |

The clean source and independent verdict repositories each start with a fresh root commit; the migration branch retains the original history. The delivery remote is [admin-wekruit/argus](https://github.com/admin-wekruit/argus), branch `codex/phase5-delivery`; the independent research remote is [admin-wekruit/panoptes-verdict-lab](https://github.com/admin-wekruit/panoptes-verdict-lab), branch `codex/verdict-lab`. Both new repositories are private. The migration checkpoint is pushed to [admin-wekruit/ehs-spatial](https://github.com/admin-wekruit/ehs-spatial), branch `codex/phase5-portability`. The pre-migration working patch and untracked archive are preserved in `phase5-artifacts/pre-migration`. Obtain current commit identities with `git log -1 --format=fuller` in each repository.

First read [acceptance](docs/ACCEPTANCE-2026-10-08.md) and inspect `git status`. Use the existing input mirror and actual console dry-run before choosing any paid reconstruction or publication. The isolated PostgreSQL test cluster is retained at `/private/tmp/panoptes-phase5-postgres/db`, port 55439, database `phase5test_utf8`, and is stopped after acceptance; the acceptance document records its restart and replay command. The user's existing database was not used for these checks.

For verdict continuation, read the independent lab's `RESULTS.md` and its authoritative runset metadata. The next evidence work is reviewer-confirmed polygons, observed floor coverage and engineering gold in a new benchmark version. Live Haiku remains unrun; it needs the required real inputs and credentials. No GPU reconstruction or deployment was performed during this source migration.
