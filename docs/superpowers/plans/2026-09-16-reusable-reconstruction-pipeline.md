# Reusable object reconstruction and correspondence

Status: active implementation. This is the new goal, not a completion claim.

The product must accept new photographs through the same path used for existing
workcells. Each discovered entity keeps its identity through source observations,
model candidates, accepted geometry, CAD projections and the browser selection.
Generation returning a mesh is not shape acceptance. A matching projection is not
proof of hidden geometry or calibrated physical dimensions.

## Current breaks confirmed in code

- `create_capture` queues analysis only. Analysis ends after discovery, geometry,
  segmentation, association and observed surfaces; it does not reconstruct models.
- `run_generation` saves a candidate then unconditionally returns `incomplete`.
  Placement is always `unconfirmed`; no per-object image-consistency assessment is
  consumed by the job result.
- RecGen runs under a frozen, budgeted noncommercial research protocol. Its runtime
  validation is numeric and must not become an invented quality approval.
- Per-object review, pose improvement and candidate assembly exist in separate
  research scripts. Their useful geometry operations must move into the shared
  path, without copying workcell IDs or adding a second scene database.

## Implementation sequence

- [x] Freeze the current code baseline (`f4c102d`) and public scene baseline
  (`a9ec2a62-1c48-4ae4-a6cd-6807ec8b71d3`).
- [x] Implement a shared, source-bound per-object correspondence audit. Check
  observation ownership/revision, current mesh/pose, CAD provenance and part
  ownership; retain every unresolved object in the result.
- [x] Reuse existing mesh projection/depth operations for per-view geometric
  quality checks. Report silhouette, visible depth, missing support, occlusion and
  frame limits. Never certify semantic shape or physical precision using a score
  alone. Add negative checks for wrong scale/axes/shape and stale evidence.
- [x] Integrate quality checks and bounded rigid pose correction into generation. Preserve
  original candidates, reject known-bad results, reuse cached calls, stop on unknown
  paid outcomes, and never repeat an unchanged paid input as a correction.
- [x] Connect upload analysis, explicit target selection, generation, audit and
  scene assembly through the existing durable job/worker services. Keep provider
  release gates and the noncommercial research boundary intact.
- [ ] Exercise the path on current frozen workcells and new captures. Replays and
  synthetic contract tests are useful but are not fresh-model quality evidence.
- [ ] Run object-by-object browser/CAD/model correspondence checks; publish a new
  immutable report only after the resulting artifacts are verified.

## Boundaries and acceptance

No model-route substitution, fabricated CAD, duplicate parent meshes, object-count
reduction or unverified status promotion. Existing branch/CAS, cancellation,
immutable storage and public report semantics remain authoritative. Use installed
NumPy/Open3D/SciPy/Pillow and the existing pytest/frontend checks.

The existing total paid-compute cap remains USD20. At baseline, 15 model-call
reservations total USD15 and USD2 is held for overhead; actual total billing is
not yet reconciled. No new charged run is justified by an unchanged failed input.

Completion requires reproducible correspondence for each object and evidence that
the same new-photo entrypoint executes the whole path. The current 24 models,
one reference, one composite observation and two rejected shapes are a baseline,
not acceptance of this goal. Historical source-CAD registration and the separate
BOR1 030 capture remain explicit work, not hidden behind the current count.

## Verification checkpoint (2026-09-16)

- The new analysis/quality path and server-only research continuation are
  implemented. PostgreSQL tests exercise atomic child insertion, duplicate
  completion, cancellation, expired attempts and branch changes.
- Source audit found 22 metadata discrepancies in the current frozen scene.
  Exact original report/model evidence proves these links; the importer now
  distinguishes artifact provenance and writes versioned observation references.
  The proposed correction changes only sourceRefs, preserving all 24 active
  models, one floor reference, one composite and two unresolved shapes.
- The full Python suite at this checkpoint passed 898 tests (29 environment
  gated skips). Subsequent stop-condition and worker-chain regressions are
  recorded with their own runs; this count is not real model quality evidence.
- No new paid model calls have been issued for this goal checkpoint. Current
  source CAD registration, failed fence/wire shapes, all-photo replay and fresh
  real-photo generation remain outstanding. CAD caches are explicitly unvalidated
  by the pure metadata audit; browser/mesh validation must still close that gate.

## Actual-data checkpoint

- The current 090 workcell's 24 active meshes were assessed against 78 owned
  observations on three 518 by 518 canonical grids. Sixteen were geometrically
  consistent and eight were inconsistent under the recorded checks. All masks
  were partial; these numbers do not certify complete shape, hidden geometry,
  semantic identity or metric placement.
- Eight bounded rigid refinements ran 1,163 evaluations. Two candidates improved
  their own views, but both regressed parent assembly evidence. No transform was
  applied. A partitioned parent must be checked together with its actual child
  meshes as well as individually; its standalone low coverage is not evidence
  that the entire assembly is missing.
- The actual browser CAD consumer returns 28 observed contours and 25 model-layer
  contours (24 models plus the floor reference). Each uses stored triangle-union
  projection rather than a bounding-box hull. This is separate from model shape
  acceptance and from registration to the historical source CAD.
- The separate 030 workcell currently retains 82 observations and 82 non-context
  entity records, with only two active imported models. Its native geometry and
  camera import is not a fresh model run. This workcell remains incomplete.
- The local API process inspected at this checkpoint has no provider manifest,
  research preparation or paid budget configured. Committing the reusable worker
  path alone does not enable real inference for newly uploaded photographs.
  Runtime configuration, pinned provider gates and a budgeted real-photo run
  must pass before claiming the new-photo path is deployed.

## Released correspondence checkpoint

- Saved source-only corrections as immutable revisions for both workcells:
  090 `57c3906c-1f55-4ff3-9114-a1cc3b74a1ec` and
  030 `91a28b9d-07be-40f1-a710-1d165633ef42`. Their model counts remain 24 and 2;
  source errors fell from 22 and 4 respectively to zero. No geometry changed.
- Published 090 report `88e3ded8-df99-41f7-8e22-94b4d0de28cf` with Pages commit
  `e862019`. CI run `35063271810` succeeded. The public response matched its
  frozen 76,233,684 bytes and SHA256
  `f5e940a42725057ee3fa8b85a6e3f93950d676a98ff483701bb843fec6c6b4f6`.
- Computer-use checks clicked all 28 records, including six nested parts. All
  selected the matching object details and photo identity. Robot and button
  previews, axes, CAD reverse selection, photo switching, rapid selection and
  clearing selection were checked. This does not certify all 28 model shapes.
- Actual CAD verification passed for all 87 current observed representations
  and all 24 active model representations. Browser consumers match 28 observed
  contours and 24 models plus a floor. Two absent models remain absent.
- A positive synthetic new-capture test now completes analysis, object modeling,
  quality review, reference classification and actual CAD validation. A translated
  model invalidates its old quality proof even when the translated CAD is valid.
  A reference-only scene completes without manufacturing an object model.
- Fresh real-photo inference remains unverified. The current local API lacks
  provider/budget configuration; historical MapAnything is not the required
  Apache weights, historical MoGe uses a different output contract, and SAM3D
  checkpoint access was last recorded as denied. Existing pinned runtime adapter
  code must be backed by actual deployment and release evidence. Do not replace
  missing evidence with passing manifest flags.
- Paid budget remains the original USD20 total: USD15 reservations plus USD2
  overhead allowance; actual billing is not reconciled. No new paid model calls
  were made in this goal checkpoint. Keep this goal active.
- Final Python verification at this checkpoint: 966 passed, 29 environment-gated
  skips, 46 warnings in 49.10 seconds with the real PostgreSQL test schemas.
  Frontend type checking/build, interaction checks and public browser verification
  passed. These results establish code behavior, not the unrun model release gates.
