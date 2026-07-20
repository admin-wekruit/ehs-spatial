# Calibrated EHS evaluation

This pack tests one deliberately narrow claim: four views of a robot workcell
can produce enough 3D evidence to distinguish a staged ladder at `0.5 m` from
one at `0.7 m` around a `0.6 m` demo fence-clearance rule. The other two cases
cover a platform inside the fence and insufficient fence evidence.

The rule is a product-validation threshold, not an official EHS standard or a
measurement-grade safety claim.

## 1. Generate and verify the oracle path

```bash
uv run python scripts/ehs_eval.py generate --output outputs/ehs_v1
uv run python scripts/ehs_eval.py offline --pack outputs/ehs_v1
```

Generation is deterministic and CPU-only. Offline mode uses generated metric
pointmaps and oracle instance masks. It validates SceneMap reconciliation,
geometry, evidence gates, and the deterministic rule, but it does **not**
validate MapAnything, SAM 3.1, or Gemini.

Useful artifacts after those commands:

- calibrated verification mesh: `outputs/ehs_v1/ladder_050/scene.ply`
- Gradio-compatible display mesh: `outputs/ehs_v1/ladder_050/scene.glb`
- four input views: `outputs/ehs_v1/ladder_050/frame_00_rgb.png` through
  `frame_03_rgb.png`
- spatial evidence: `outputs/ehs_v1/ladder_050/topdown.png`
- machine-readable result: `outputs/ehs_v1/offline_report.json`

## 2. Run one paid provider case

Put `REPLICATE_API_TOKEN`, `FAL_KEY`, and `GEMINI_API_KEY` in the ignored local
`.env`, then start with only the failing ladder case:

```bash
uv run --env-file .env python scripts/ehs_eval.py live \
  --pack outputs/ehs_v1 \
  --case ladder_050 \
  --live
```

The duplicated `live` is intentional: the first selects the command and the
`--live` flag explicitly authorizes paid calls. A live case uses the generated
RGB images only—never the oracle pointmaps or masks—and runs the production
MapAnything → SAM 3.1 → SceneMap → Gemini path without retry or fallback.

One case makes 1 MapAnything call, 28 baseline SAM calls (7 object labels × 4
images — the floor is fitted geometrically from the full point cloud, not
segmented) plus synonym-fallback calls only for frame-labels whose canonical
prompt returns nothing, and at least 1 Gemini interaction. `ladder_050` makes a
second Gemini interaction for the grounded chat check. As checked on
2026-07-15, fal lists SAM 3.1 at `$0.01/request`, so segmentation is roughly
`$0.28–0.5/case`; Replicate is usage-time billed and Gemini is token billed. A
four-case run therefore makes 4 MapAnything calls and ~112–160 SAM calls, and
at least 5 Gemini interactions. Recheck the official
[fal](https://fal.ai/models/fal-ai/sam-3-1/image-rle),
[Replicate](https://replicate.com/vufinder/map-anything), and
[Gemini](https://ai.google.dev/gemini-api/docs/pricing) pages before scaling.

The pack-level result is `outputs/ehs_v1/live_report.json`. Provider artifacts
are stored below `outputs/ehs_v1/live_runs/{run_id}/`; their exact relative paths
are listed in the report. On provider failure, the report records the provider
and failing operation with configured key values redacted. No credentials are
written to artifacts.

Run other cases individually only after inspecting the first report:

```bash
uv run --env-file .env python scripts/ehs_eval.py live --pack outputs/ehs_v1 --case ladder_070 --live
uv run --env-file .env python scripts/ehs_eval.py live --pack outputs/ehs_v1 --case platform_inside --live
uv run --env-file .env python scripts/ehs_eval.py live --pack outputs/ehs_v1 --case fence_occluded --live
```

The live hard gate is intentionally coarse: correct status, correct side of the
`0.6 m` threshold for sufficient cases, required multi-view entities, complete
provider artifacts, and grounded fact IDs. Absolute distance error is reported
for diagnosis but is not treated as certified metrology.

## 3. Opt-in pytest benchmark

The default test suite never spends provider credits. The paid test is enabled
explicitly:

```bash
EHS_LIVE_BENCHMARK=1 EHS_BENCHMARK_CASES=ladder_050 \
  uv run --env-file .env pytest -q -s tests/test_ehs_benchmark_live.py
```

To request all four cases, set
`EHS_BENCHMARK_CASES=ladder_050,ladder_070,platform_inside,fence_occluded`.
