# Changelog

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
