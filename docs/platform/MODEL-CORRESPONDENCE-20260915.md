# Object and model correspondence: implementation record

## Delivered scene

Publication `10cb71d2-80dc-4cc8-b678-8ffc27bdf857` freezes revision
`579a660e-2d3d-4030-b92c-63e536862f02`. It derives from revision
`e211fcc8-ab63-4832-9fcc-b1604418d0e0` without modifying historical publications.

- All 28 object records, 84 original observations, and original report/CAD evidence remain.
- Twelve object records now have actual decoded model geometry, up from nine.
  Three are components cut from the existing folded guard mesh, not new inference.
- Guard #07 is an assembly with its residual geometry. Parts #22, #24 and #25
  own the central panel, right fold and left fold respectively. Selecting the
  assembly highlights its complete family; selecting a child highlights only
  that child's triangles. The full original mesh remains an inactive source.
- Photo, CAD, scene and individual-model selection share entity IDs. CAD coverage
  remains 28/28 independently of model availability. The individual model retains
  local XYZ and the light background. Actual GPU load failures are reported separately
  from declared model availability.

The independently reviewed partition conserves every source triangle exactly once:
204,510 = 67,307 center + 67,743 right + 67,517 left + 1,943 residual.
Source vertex/normal/color bytes, materials and world transforms are preserved.
Placement remains unconfirmed and units remain uncalibrated.

The accepted `.blend` and GLB contain twelve meshes and one assembly; three cameras
and two parameter objects pass the existing Blender reopen checks. Export job status
is still `incomplete` because other objects lack models and placement is unconfirmed.

| Export | SHA-256 |
| --- | --- |
| scene.glb | `62de19d55d751abf6e9000c5a17d8c8486b609812d110df11f208be64ad9b5ba` |
| scene.blend | `70087b053b9ce12ac5efd3672fbd06688925d85a2942596f305cfba0f23b514c` |

## Shared contracts

Part relations require exact source-revision and observation evidence. Cycles,
cross-frame attachment, and reactivation of the archived full-parent mesh fail.
Parent transform, visibility and material changes operate on its explicit family;
child changes affect that child. World-space TRS changes reject shear. Python and
TypeScript consume the same 25-case fixture, including valid family frame changes
and rejected isolated frame changes. Failed previews never submit a network write.

Generation now requires explicit entity IDs. Batches reject reference surfaces,
source-only context, existing active models and part families before provider calls.
Explicit regeneration of a single independent object remains available. Admin runtime
validation is separate from product generation, uses frozen configuration hashes and
the existing budget reservation/attempt rules, and cannot advance a scene branch.

## Verification

- 184 targeted Python tests pass, including real PostgreSQL worker/transaction checks.
- TypeScript compilation and Vite production build pass. Existing lazy policy-editor
  bundle-size warnings remain; no new dependency was added.
- Model-family, CAD/report, selection and renderer checks pass. Actual WebGL checks
  cover parent/child highlights, delayed asset loads, material updates during upload,
  invalid/empty assets, context loss and resource disposal.
- Computer-use inspection of the accepted publication selected all three actual
  parts: each photo outline, scene highlight and individual model matched. Default
  selection is empty; a child's feedback entry opens its own context.
- All ten publication bundles passed 612 exact read-response checks and 571 distinct
  byte-verified asset checks. The new publication contains 334 manifest entries.
- Fresh catalog startup RSS: 1,643,692,032 bytes under the existing 3 GiB reservation.

Local evidence is under `.platform/model-correspondence/`: `accepted-audit.json`,
`accepted-dispositions.json`, `guard-parts/accepted-ownership.json`,
`blender-validation.json`, `panoptes-accepted-parts-qa.json`, `final-tests.log`,
`hf-access.json`, and `budget.json`. Original binary assets and credentials are not
committed to the source repository.

Source commit: `69175b2a73e49fda67386456c636aa8649ef066c`.
Pages commit: `03e7b160e04e23605aa12f2d7e83d5b4740d744e`;
[deployment run](https://github.com/admin-wekruit/panoptes-workcell-report/actions/runs/35030500504)
completed successfully. Public HTML matches the release build byte for byte.
The [public report](https://admin-wekruit.github.io/panoptes-workcell-report/app.html#/reports/10cb71d2-80dc-4cc8-b678-8ffc27bdf857)
was reloaded after deployment and checked through computer use: 12/12 models
loaded, CAD 28/28, empty initial selection, then the whole guard and all three
parts selected with matching photo/scene/individual-model displays. The local
actual-asset acceptance also verifies that all four selections preserve the
manually adjusted main-camera matrix exactly.

## Remaining work and compute boundary

This is not an all-object model completion claim. Transparent guard panels are not
present in the existing fence mesh; copying the whole fence cannot supply them.
Signal lights, controls and other missing geometry need reviewed generation inputs.
Mixed or unresolved observations remain in the inventory and require evidence
correction or explicit identity review; they are not counted as completed models.
The 28-row reviewed disposition ledger distinguishes nine retained models, three
accepted components, one floor reference, two fence-part geometry gaps, four
identity/mask reviews (#12/14/23/28), seven missing independent geometries,
one insufficient generation input (#17), and one mixed observation (#21).
For #21, existing masks support 8,450 disjoint cart/guard pixels; 895 boundary
pixels still require focused annotation. It is not a new physical object and
does not require a new model call to correct its observation.

The authorized total compute budget is USD 20, including failed validation/generation.
No new GPU or model inference was started. Two CPU-only Hugging Face access diagnostics
ran; their actual billed cost is not yet known. The authenticated deployment account
is denied access to the pinned SAM3D checkpoints with `403 GatedRepoError`, classified
as `account_access_not_approved`. The user must obtain model access or supply an
already-approved account before checkpoint-dependent validation can proceed.

Mesh-only source preparation and four CPU checks pass, but no Linux CUDA runtime
image or new inference result is approved. Exact dependency/redistribution review,
image digest, external-pointmap and official-pose fixtures, and per-object quality
acceptance remain open. See [runtime build inputs](../../containers/sam3d/README.md).
No release gate was changed to passed to bypass these requirements.
