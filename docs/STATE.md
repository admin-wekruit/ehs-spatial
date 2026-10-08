# Current state — 2026-10-08

This is the Modal source repository. Customer GPU-service, Docker and Linux adaptation is outside this delivery. Historical on-prem reproduction instructions are preserved with the research source and are superseded for this checkout.

## Delivery

| Stage | Existing implementation | Execution |
| --- | --- | --- |
| Geometry | DA3-BASE, RoMa, NumPy LM bundle adjustment, dense triangulation, MoGe-3 masked fill | Modal GPU + local CPU |
| Completion | SAM 3D Objects, candidate selection across the workcell photographs | Modal GPU + local CPU |
| Assembly | Existing 9-DOF silhouette/boundary/depth fit, v2 floor-contact penalty | CPU |
| Measurements | Existing shape/floor/lower-edge/box and G1–G8 checks | CPU |
| Report | Platform import/publication, schema-2 measurement messages, viewer | CPU + configured storage |

Pi3X, RecGen, DA3-LARGE and TRELLIS are historical comparisons. They are not model choices in this delivery. Internal historical variable names such as `pi3xFrames` describe the retained point-map interface; they do not select a Pi3X provider.

## Frozen regression references

| Workcell | Publication | Revision | Boxes | nativeToMeters |
| --- | --- | --- | --- | --- |
| 090 | `a9a6e0a0-a77e-4ca0-b460-aa6f18d72698` | `86207849-d672-4df4-b453-22584f491894` | 9 | 3.2371372068568487 |
| 030 | `fafdeb6b-7122-4434-a62d-676b3ff9e50a` | `f464b25e-9e27-4966-aca1-e21472fa2962` | 8 | 3.5616493184162774 |

These are locally preserved published MVS+fill/SAM3D/v2 reports. The earlier `4b58dbd2…` / `cd84d3fb…` references belong to Pi3X/RecGen comparison reports. A published reconstruction is evidence of the recorded computation, not independently verified field calibration.

`scripts/check_measurement_regression.py` compares exact physical geometry, object identities, scale, floor, dimensions, uncertainty and confidence. Display prose is replaced by localized messages, so whole-file byte identity is not the schema-migration criterion. `tests/fixtures/measurements` records the frozen projection and publication metadata.

## Localization

Schema 2 uses `{"code":"stable.catalog.key","params":{...}}` for names, confidence explanations, capture advice, pipeline stages and sources. Nested messages are rendered recursively using the existing translation function. Every retained consumer uses the same schema; language-specific sibling fields have been removed. Operator-entered names remain operator text. The current three catalogs have the same keys and placeholders. Dutch terminology still warrants a customer language review.

## Preservation and evidence status

The original source was preserved before migration. Frozen inputs are outside the source checkout, with every file's size and SHA-256 in `data-manifest.json`. No weights, credentials, `.platform` contents or `node_modules` are delivery data.

Phase 4 is the independent `panoptes-verdict-lab` project. Its observed scenes still lack verified zones, coverage and reviewed gold labels. Compilation counts are not actual rule bindings. Live Haiku execution requires credentials and separately supplied real declarations; offline playback is not a live model result.

See [final acceptance](ACCEPTANCE-2026-10-08.md) for executed checks and remaining external validation. No fresh GPU reconstruction or deployment is implied by offline checks.
