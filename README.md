# Panoptes

Workcell photographs → metric reconstruction → measured boxes, faces and clearances → a report viewer.

This repository is the Modal source. The shipped geometry is DA3-BASE + RoMa + NumPy bundle adjustment + MoGe-3 masked fill; completion uses SAM 3D Objects and assembly v2. Customer Linux, Docker and on-prem service adaptation belongs in the customer's clone.

Read [current state](docs/STATE.md), then [run instructions](HANDOFF.md) and [delivery closure](docs/DELIVERY-PATH-2026-10-08.md). [Initial acceptance](docs/ACCEPTANCE-2026-10-08.md) and [independent-review follow-up](docs/REVIEW-FOLLOWUP-2026-10-08.md) record the executed checks and remaining limits. [Agent rules](CLAUDE.md) apply throughout.

```sh
make env
# Fill .env, including an explicit data root and input mirror.
uv sync --frozen --extra dev
uv run --env-file .env python scripts/fetch_inputs.py
make check-env
make test
uv run --env-file .env panoptes run --cell 090 --dry-run
uv run --env-file .env panoptes run --cell 030 --dry-run
```

A real run needs an authenticated Modal account and the configured platform storage. It spends GPU resources. Run instructions distinguish local verification from real reconstruction and publication.

The implementation lives in `argus/{pipeline,checks,platform,providers}`. One cell configuration lives in `argus/pipeline/cells/<cell>.json`; `argus.ROOT` is the only checkout-root definition. Large frozen inputs remain outside Git and are verified by `data-manifest.json`. The viewer defaults to English and supports English, Chinese and Dutch; display messages use one code/parameter schema.

The independent Phase 4 verdict lab and preserved research source are outside this delivery checkout. Reports here contain measurements and evidence; they do not certify machine compliance.
