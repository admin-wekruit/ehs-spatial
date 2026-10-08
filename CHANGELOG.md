# Changelog

## Unreleased — 2026-10-08

### Verdict layer lab (`ehs_spatial/verdict/`, branch verdict-lab merged)
- Contracts C1–C5 (`contracts.py`), plugin registry (`plugins.py`), signature v1, lab runner / matrix / ledger / scorecard and
  `panoptes verdict run|matrix|scorecard|plugins`; baseline plugins ported from the 2026-10-07 trial with per-rule parity on 090 / 030;
  python@1 parity engine and decision-rule variants; synthetic cells, threshold grid, metamorphic / differential harness; space-carved
  coverage from the frozen frames; sigma policy; L3 perception monitor; clause graph v0 (20 clauses, unverified numbers) with table
  functions, alignment and retrieval. `tests/verdict`: 66 passed. Scorecard v0: `docs/research/verdict-lab-scorecard-v0-2026-10-08.md`.
  New optional extra `verdict` (clingo, pyyaml). Reports still carry no machine verdicts; the lab is research tooling.

### Verdict lab, round 2 (2026-10-08, afternoon)
- Decisions recorded in `docs/research/verdict-layer-plan-2026-10-08.md` §6: enclosure without Coverage → CANNOT_DETERMINE; L5 first round
  compares three synthesis variants against the handwritten control; LLM = Claude Haiku; gold is labelled inside the report.
- `clingo@2`, `python@2`, `handwritten@2`: the enclosure rule needs coverage (no Coverage → CANNOT_DETERMINE, breach through observed
  floor → FAIL, breach only through unobserved floor → CANNOT_DETERMINE); clingo renders every numeric fact as a generic
  `<pred>(args, V, U)` atom and boolean facts as `<pred>(args)` (synthesised packs use them) plus `coverage_known.`; parity test
  clingo = python on synthetic cells with / without coverage. Benchmark v0 counts unchanged (neither cell has coverage).
- `ehs_spatial/verdict/llm.py`: Claude through the official SDK (`anthropic` added to the `verdict` extra), structured outputs,
  one response cache keyed by model + schema + prompts (the only hash in the package), `PANOPTES_FAKE_MODEL=1` = cache only.
  `env.template`: `ANTHROPIC_API_KEY` (or an `ant auth login` profile), `PANOPTES_VERDICT_MODEL`, `PANOPTES_LLM_CACHE`.
- Scorecard: rule-pack statuses (compiled / needs_input / vocabulary_gap / refused) and LLM calls per row; `panoptes verdict extract`
  and `panoptes verdict labels` subcommands.
- L4 `llm-extract@0` (`layers/l4_spec/llm_extract.py`): safety-concept text (numbered requirements + clause citations, the
  `spec/samples/` type) → clause candidates through Claude, one call per unit, then deterministic checks (vocabulary, grounding of every
  number in the unit's text, citation, duplicates, table ids) and a diff against the hand-extracted reference; `panoptes verdict extract`;
  config `llm-extract-ts.yaml`; Claude Code subagent `.claude/agents/ehs-safety-spec.md` (the EHS safety subagent: ingest a new text,
  run the extraction, read flags + diff, propose vocabulary, never change numbers).
- L5 three synthesis variants vs the handwritten control (`layers/l5_rules/`): `function-library@0` (no LLM: clause requirement → rule
  spec + ASP through the shared renderer `asp.py`), `code-synthesis@0` (Claude writes spec + ASP + its own tests; signature check,
  tests, threshold grid, metamorphic relations decide compiled / refused), `redundant-translation@0` (two framings, agreement +
  differential execution); configs `funclib-ts.yaml`, `codegen-ts.yaml`, `redundant-ts.yaml`. Function library on the merged ISO + TS
  graph (38 clauses): 11 compiled / 9 needs_input / 18 refused (exists, comparisons inside applicability, attribute requirements,
  zone kinds other than hazard_zone, not photo-checkable).
- `relations@2`: `vertical(X)` / `horizontal(X)` of light curtains and area scanners from the box aspect (Signature predicates added);
  TS 9.1.7 applies to horizontal fields, 9.1.1 / 9.1.2 to vertical ones (before: 9.1.7 failed every vertical curtain). A bare attribute
  in a clause's applicability (TS 8.1.4 `operator_interaction`) means that attribute of the first selection variable.
- L7 `html@1` (`layers/l7_report/html.py`): one self-contained verdicts.html with the labelling panel (gold status 1–6 incl.
  NOT_APPLICABLE, reason, sure / unsure, re-measure, declared inputs, reviewer; hotkeys, queue, progress, filters; export / import of
  `verdict-labels/1` JSON, no server); `panoptes verdict labels merge|declared` (`lab/labels.py`) turns reviewer files into a new
  `gold.json` version (agreements in, conflicts listed) and a declared-inputs file that `scene-json` merges (`declared:` cfg).
  Design from `docs/research/labeling-ui-survey-2026-10-08.md` (14 tools, 8 patterns adopted).
- `tests/verdict`: 95 passed (fake model: every LLM plugin is tested through a monkeypatched `llm.complete`; no live call was possible
  on the development Mac — no Claude credentials). Scorecard v1: `docs/research/verdict-lab-scorecard-v1-2026-10-08.md`.

### Test baseline note
- At c9da31a (docs reorganisation; also touched `scripts/workcell_checks/box_faces.py`, `lower_edge.py`, `scripts/workcell_photo_oneshot.py`,
  README) the full suite went from the 6 known failures to 23 (`tests/test_app.py::test_readme_links_each_provider_credential_source`,
  `tests/test_report_runner_{decisions,lightning,reproduce,stages}.py`). Verified by running the 23 ids on 75ac212 (main) and on
  verdict-lab: identical. Not caused by the verdict lab; to be triaged separately.

## 1.0.0-rc1 — 2026-10-07 (enterprise handoff candidate; `VERSION`)

The first release a customer runs end to end on their own GPUs and storage: one `git clone` → `.env` → `make up` → `panoptes run --cell 090`
(`HANDOFF.md`). Algorithms and parameters are unchanged from the 2026-10-07 module-swap publication (`research/module-swap-2026-10-07/`).

### One repository (consolidation, 2026-10-07 evening)
- `panoptes-serving` is merged into this repository and frozen upstream (its README points here): `serving/` (v0 sam3 / mapanything / moge
  and v1 sam3d :8805 / geometry-mvs :8804 services, registry, smoke), `deploy/` (compose per GPU card, jumpbox, publications),
  `docker/` recipes, `ehs_spatial/providers/` + `ehs_spatial/cli.py` (`panoptes run|status`), `research/module-swap-2026-10-07/`
  (notes, scripts, 0.5 GB frozen inputs and intermediate outputs under `data/`), `env.template`, `HANDOFF.md`, CI. `PANOPTES_WORKCELL`
  defaults to this checkout (`env.sh`, `Makefile`); `PANOPTES_SERVING` is the same path.
- `pyproject.toml`: `[project.scripts] panoptes = "ehs_spatial.cli:main"`; `uv.lock` unchanged otherwise.
- `modal_apps/geometry_clean_ab.py`: the research harness (`fair_ab_modal`) is located through `SWAP_NOTES` (env.sh) instead of a
  machine path; absent harness still tolerated (on-prem geometry image).
- Tests: the contract dry-run tests and the provider tests restore the interpreter exactly (the on-prem Modal stub, stage modules and
  research-note modules they load; `sys.meta_path`, `os.environ`) — third-party C extensions are never dropped from `sys.modules`
  (one load per process). The merged suite shows only the 6 pre-existing platform pin failures of `main`
  (`tests/test_report_runner_reproduce.py` delivered-profile hits, `tests/test_report_runner_stages.py` `versions.json` / `splat_train`
  defaults) — stale version pins from the 2026-10 platform merge, not touched here. CI sets `PANOPTES_WORKCELL` to the checkout.

### Handoff package (WP5 + WP6, this branch)
- `env.template`: the one configuration surface — every key the system reads, grouped (GPU services, client backends, storage,
  publication hosting, weights fetch, pipeline paths, tests), one comment line each, in sync with `research/module-swap-2026-10-07/env.sh`.
- `scripts/check_env.py` (+ `tests/test_check_env.py`): validates `.env` offline (`--strict`) and against the live services (`--live`).
- `Makefile.handoff`: `env`, `check-env`, `check-env-live`, `lint`, `test`, `run CELL=090` (to be included by WP3's `Makefile`).
- `HANDOFF.md` rewritten: 4-step path, jump VM section, two-A100 layout, credentials, acceptance A1–A6 with commands, troubleshooting,
  one-line rollback; the previous handoff kept under "History".
- `deploy/README.md`, `deploy/jumpbox/`: socat systemd unit template + one env file per service port (what the customer runs today),
  and the recommended nginx + TLS reverse proxy (`compose.nginx.yml`, `nginx.conf`, one `location` per service, `X-API-Key` passed through).
- `deploy/publications/`: report hosting as the nginx stack (`compose.nginx.yml` + `nginx.conf`: viewer bundle, measurement layers,
  `/api/` → publication service) or as an S3 static website (bucket policy, sync commands); how `PANOPTES_PUBLIC_BASE_URL` feeds the viewer.
- `docs/ARCHITECTURE.md`: the four layers and their replacement points with file names, one SAM 3D call end to end, the future-proof rules.
- CI (`.github/workflows/ci.yml`): ruff (`ruff.toml`, minimal rules), `pytest tests/` with `PANOPTES_FAKE_MODEL=1`, shell syntax, YAML,
  compose `config`, Dockerfile `build --check`. Under 5 minutes, no secrets, no GPU.
- `research/module-swap-2026-10-07/run_all.sh`: `"$PY"` quoted on the heredoc line (zsh -n false positive; runtime unchanged).

### Lands with the sibling work packages (the integrator ticks each on merge)
- WP1 (ehs-spatial): `MongoRepository` / `MongoPolicyRepository` (`PANOPTES_DATABASE_URL=mongodb://…`, `PANOPTES_MONGO_PREFIX`),
  `S3BlobStore` (`PANOPTES_BLOB_ROOT=s3://…`, `AWS_ENDPOINT_URL`, `AWS_ACCESS_KEY_ID/SECRET`), `scripts/panoptes_artifacts.py push/pull`.
- WP2/4: `ehs_spatial/providers/{sam3d,geometry_mvs,service_client}.py` (`http | local | modal`), `panoptes run --cell … [--from STEP]`,
  `panoptes status`, `ledger.json` per step, multi-card scheduling by `/healthz` `queue_depth`.
- WP3: `serving/sam3d_service.py` (:8805), `serving/geometry_mvs_service.py` (:8804), `serving/registry.yaml`, `deploy/compose.gpu-a.yml`,
  `deploy/compose.gpu-b.yml`, `Makefile` (`up GPU=a|b`, `down`, `smoke`, `logs`), `serving/smoke_v1.sh`, contract v1 (`docs/BACKENDS-v1.md`).

### Unchanged
- v0 services and contract (`serving/{sam3,mapanything,moge}_service.py`, `docs/BACKENDS.md`), the customer's existing jump forwards 8080/8090.
- Pi3X / RecGen stay out of the delivery; no Kubernetes; the feedback assistant (sqlite) is untouched.
