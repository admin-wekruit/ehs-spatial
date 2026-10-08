# Identity input scan — 2026-09-13

Read-only PostgreSQL snapshot; no model calls or production mutations. Full machine manifest: `.platform/identity-scan-20260913-01/manifest.json`. Coverage and source pins: `docs/platform/IDENTITY-SCAN-20260913.json`.

| Scope | Count |
| --- | ---: |
| assets | 369 |
| branches | 6 |
| captures | 12 |
| catalogPublications | 2 |
| evaluationRevisions | 17 |
| invalidAssets | 0 |
| projects | 5 |
| publications | 19 |
| revisions | 28 |

Every branch head and all 19 database publications / both catalog publications appear in `evaluationRevisions`; repeated publications of the same revision are evaluated once. All 28 revision documents retain exact stored hashes. All 369 DB assets and both catalogs passed byte/hash validation. Private feedback, owner capability values and model credentials are excluded.

| Fixed revision | Observations | Masks before | Saved geometry frames | Source family |
| --- | ---: | ---: | ---: | --- |
| `0e8851b9-cfb8-4f8c-9cd9-58c0fccf2800` | 58 | 58 | 0 | lucida_same_capture |
| `13c11c61-9968-4039-9351-4770166f8c48` | 67 | 58 | 0 | lucida_same_capture |
| `2c7f5987-8056-43c2-814d-ea614599d319` | 67 | 58 | 0 | lucida_same_capture |
| `45f843bb-066c-4dd5-9cf0-a460455718b6` | 67 | 58 | 3 | lucida_same_capture |
| `518e7869-3ce2-4c2e-be18-64750483846d` | 67 | 58 | 3 | lucida_same_capture |
| `58ccf6df-d773-42d0-86f4-74e958fedcd8` | 67 | 58 | 0 | lucida_same_capture |
| `5f397ec6-2a54-49f8-8dc1-f9ae9eb50094` | 82 | 80 | 0 | components_old_geometry |
| `6770adce-440e-4f06-8cd0-ea9bbf7b39f1` | 82 | 80 | 0 | components_old_geometry |
| `6a03b5ac-8877-435e-b220-49565cf55ac6` | 1 | 0 | 0 | qa_fixture |
| `89ca3522-388b-4e37-8d39-89e6d1bf0183` | 67 | 58 | 0 | lucida_same_capture |
| `8df72130-ca20-4802-ad21-c5b8dfac3a55` | 1 | 0 | 0 | qa_fixture |
| `930db000-3ad4-4fac-a02f-c86317baa1e8` | 80 | 80 | 0 | components_old_geometry |
| `96f60839-af84-48e1-8411-17627d5733e9` | 67 | 58 | 0 | lucida_same_capture |
| `a97267c4-d435-43ee-915f-a59f2bb44ed6` | 84 | 58 | 3 | lucida_same_capture |
| `bdc2923f-9ce4-4a97-88ee-1c310d2c396e` | 82 | 80 | 0 | components_old_geometry |
| `d403c6b5-049a-4162-a944-2daae2e352e5` | 67 | 58 | 0 | lucida_same_capture |
| `fe366c68-977b-421d-806e-580e84f86614` | 1 | 0 | 0 | qa_fixture |

## Source completion

The current fixed revision has 68 business records, 9 source-confirmed multiview groups, 59 unresolved single-view records and 84 observations. The independent offline copy `.platform/identity-ready-20260913-01/manifest.json` now contains 84/84 masks: all 26 original PNGs and canonical NPYs were hash-verified, with original shape 3840×2880 and canonical shape 518×518. No raster was reconstructed or resampled. Original observation IDs, revision numbers, boxes, polygons, existing pixel mappings, 58 existing mask IDs, and all entities are unchanged. Added bytes are in the isolated output directory; old publications are unchanged.

`maskEvidence` explicitly identifies originalMaskAssetId, canonicalMaskAssetId, both shapes, inputToCanonical, geometryManifestAssetId and sourceRefs. The canonical NPY is the exact segmentation grid; consumers must prefer it over re-downsampling the original PNG. Geometry uses existing `geometryEvidence.frames[].assets` NPY/PNG references and frame camera IDs.

The components source has four original images and four canonical images byte-identical to `bor1-components-20260909`; all four K/c2w arrays are exactly equal. Its old native coordinate solution must not be mixed with the newer lucida Pi3X solution. It does not save a content-valid alpha mask; no alpha is inferred. The three QA revisions lack masks/cameras and are not comparable.

All 17 prepared inputs are frozen in `.platform/identity-ready-all-20260913-02/manifest.json`. Fourteen use verified native source geometry. The components content support is explicitly derived as source-valid, finite XYZ/confidence, positive camera depth and confidence >= 0.1; the source selection and original array hashes are saved. Original source cameras remain unchanged.

Two additional components masks were recovered from `product-evidence-01/evidence/objects.json`: emergency-stop candidate `object_6dd85942d104987a83c677d6` and control-button candidate `object_ddfd00218e68ffd9e3e7f515`. Original PNGs are exactly 3024×4032, canonical masks 518×518. Each matches the existing observation through exact candidate ID, original image SHA and pixel mapping; detection/refinement records and raw mask bytes are hash-verified. All 82 observations in the three later component revisions now have masks. No original `sourceRefs`, boxes or existing mask IDs were rewritten.

Two earliest revisions have a null `captureId`. Image ID plus SHA gives eight exact historical candidates for the earliest main revision and three for the earliest component revision, so an existing capture is not chosen by guess. The reviewed admin batch will register two new real Capture records for the exact original photo assets, retaining the original target, source revision and frozen manifest provenance. The candidate evidence is in `IDENTITY-SCAN-capture-recovery-20260913.json`.

## Frozen execution and review

The unified `run_reassociation` entry point was executed against all 17 fixed revisions with a read-only repository adapter and isolated output blobs. `.platform/identity-evaluation-all-20260913-03/results.json` records 12 successes: nine main reports (67 or 84 observations) and three components reports (82 observations). Three QA results explicitly have `comparability: not_comparable` because geometry is unavailable. The two early null-Capture revisions retain the explicit migration error until the real Capture records are registered. Every output passed document validation and original observation, mask, camera and asset-reference conservation checks. No paid models ran.

The new geometry decisions cover 12 revisions. Original-photo review assets are in `.platform/identity-review-20260913-03/manifest.json`: 11 distinct crop pairs, including older and newer boxes that differ by roughly three source pixels. The nine existing source-binding groups and five requested comparisons are in `.platform/identity-source-review-20260913-01/manifest.json`. Comparisons include left/right bollards, sign/button, button/light-curtain housing, and housing/fence. Crops retain original pixels; only contact-sheet display previews are resized. None of these files supplies a manual identity decision or human ground truth.

`scripts/research/reprocess_object_identity.py` defaults to a files-only plan. It pins the complete source manifest, code and all fixed targets. Its explicit execution mode uses an admin transaction scoped to the manifest to register immutable assets, independent reconstruction branches, real Capture records where needed and durable `reassociate_scene` jobs. Normal `claim_job` and `finish_job` retain lease, cancellation and head-CAS behavior. It does not publish, rotate capabilities or rewrite old revisions. A crash or expired running job requires inspection before resumption.

```sh
.venv/bin/python -m scripts.research.prepare_object_identity_inputs --manifest .platform/identity-scan-20260913-01/manifest.json --sources docs/platform/IDENTITY-SCAN-sources-20260913.json --output NEW_PREPARED_DIR
.venv/bin/python -m scripts.research.evaluate_object_identity --manifest NEW_PREPARED_DIR/manifest.json --output NEW_EVALUATION_DIR
.venv/bin/python -m scripts.research.reprocess_object_identity --manifest NEW_PREPARED_DIR/manifest.json --output REVIEWED_PLAN.json
```

Only after reviewing the exact plan, an operator can run `--execute REVIEWED_PLAN.json --runtime-env PRIVATE_RUNTIME_JSON --output AUDIT.json`. The private runtime file must have mode 0600. The script does not read owner capability values or change capability hashes. The prepared plan `.platform/identity-batch-plan-20260913-02.json` was generated and checked without executing it in the live database by this scan task.

## Acceptance limits

No independent real multiview holdout has been verified. Other imported projects are BOR1 variants; inspected user-3view-02 / 87926d59... photographs are the same physical workcell, and phase3-multiview-mtmc is rendered warehouse imagery. Existing source identity statements remain source evidence, not newly verified human pair labels. Physical identity precision/recall remain null. An all-unresolved output cannot count as correct.

Checks: importer and batch suites passed 13 tests with no skips, including an actual temporary PostgreSQL schema. The transaction check covers idempotent registration, a persisted real Capture, prepared-document loading, asset deduplication, a concurrent head change and unchanged original publication/revision. The real 26-file and two-component-file hash/dimension/source attachment checks passed.

## Actual batch and visual acceptance

The actual batch completed all 17 fixed targets: 14 native-geometry successes and 3 QA results explicitly incomplete/not comparable. Both early null-Capture inputs now use newly registered real Capture records. The final read-only acceptance verified the old 28 revisions, 19 publication snapshots, 369 asset hashes/sizes and 6 original branch heads unchanged, and source observation/mask/camera/asset references conserved for all 17 targets. Four newly created component revisions exposed JSONB negative-zero hash drift; append-only ordinary-job successors now pass independent digest checks while the original failure audit remains intact.

Codex actually inspected original photographs and crops for 9 source groups, 5 negative comparisons, 11 offline pair variants and 4 newly accepted real-batch pairs. Same-identity reviews were supported, negative comparisons supported distinct objects, and the two single-view recovered control masks remain ambiguous for cross-view identity. This is **Codex visual review, not human or field ground truth**; it wrote no identity decisions. Review records are `.platform/identity-visual-review-20260913-01.json` and `.platform/identity-batch-visual-review-20260913-01.json`.

The main 68→65 result preserves 84 observations: 12 confirmed multiview groups cover 31 observations; 53 independent records remain pending (34 competing candidates, 19 no supported match). These are not 53 established physical objects. The competition replay found no sorting/second-candidate error, but 33/34 top-pair rejections involve same-image competitor masks with IoU≥0.80 (21≥0.95), including imported observations competing with source-bound representations. Physical identity acceptance therefore remains incomplete. Full scope, final revision mapping, limitations and exact evidence paths are in [IDENTITY-ACCEPTANCE.md](IDENTITY-ACCEPTANCE.md) and [IDENTITY-ACCEPTANCE.json](IDENTITY-ACCEPTANCE.json).

Further source tracing recovered six deterministic equivalences (two views each of left/right bollards and the right fence): both records cite the same original SAM artifact SHA and `rle` instance and their mapped canonical masks are byte-identical. These relationships survived in source provenance but were not consumed as shared identity across generated/raw namespaces. The parent task is addressing that omission; this scan does not mark physical deduplication complete. See `source-equivalence-trace.json` under the acceptance artifact directory.


## Source-v2 full fixed-revision batch acceptance

The complete frozen 5-project / 17-target set was reprocessed again after the generic exact-source proof importer and shared verifier were implemented. Result: 14 succeeded / 3 QA incomplete, zero paid model calls. Main `a97267c4` → `b638e847` has 59 business records / 84 preserved observations; component head `6770adce` → `6fca273f` has 81 / 82. All 30 source-proof pairs were independently verified from registered immutable blobs. Main has 12 confirmed groups / 37 observations and 47 independent pending records (28 competing / 19 no supported match), not 47 confirmed physical objects. All old revisions/publications/assets/heads and all original non-null measurement values are preserved; full results are in [IDENTITY-ACCEPTANCE.md](IDENTITY-ACCEPTANCE.md) and [IDENTITY-ACCEPTANCE.json](IDENTITY-ACCEPTANCE.json). The new geometry-pair set is fully covered by the prior Codex visual review gallery; no human or field ground truth is claimed. Earlier scan/batch paragraphs above describe preserved historical rounds, not the current accepted result mapping.
