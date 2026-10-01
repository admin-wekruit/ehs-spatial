# Four-photo guard geometry experiments

User approved running all applicable routes on 2026-09-30. Immutable input is
`workcell-three-boards-complete-2026-09-30-a` plus its four original BOR1 photos.
The button is a configurable 20 cm height / width hypothesis, not surveyed truth.

## Execution

1. Freeze input hashes and A0. A1 undoes the saved affine placement and runs the
   same placement loss with seven similarity parameters. Preserve face ownership.
2. A2 fits each board independently from genuine cross-view image observations,
   initialized by MapAnything. Export three two-panel meshes and uncertainty / support.
3. A3 uses the same observations and a shared left/right fold angle. Preserve A2;
   equality under this explicit prior does not count as independent accuracy.
4. Run COLMAP point matching, triangulation and camera/point BA; run LIMAP line
   triangulation on the same data. Save actual dependency/runtime failures as results,
   repair ordinary integration failures, and compare any usable geometry fairly.
5. Integrate actual outputs into the existing viewer, review geometry and code,
   publish under `workcell-photo-direct/`, and verify interaction online.

## Checks

- One bounded synthetic check: known folded geometry, camera projection, transform
  invariance, and a genuinely held-out set. Never manually adjust angles to match.
- Freeze masks / source definitions; no stripe edge is automatically a fold line.
- Record fit and held-out errors separately, view support, angle stability, model
  parameters, scale assumption and model/camera frame. No physical accuracy claim
  without independent truth. Unsupported dimensions remain unknown.
- Models and reported angle share the same geometry. Camera updates apply to all
  dependent geometry in their experiment, never new measurements on stale meshes.
- Heavy work only in ephemeral `modal run`, two A100-80GB, finite timeout, retries 0,
  min_containers 0. Log every run and estimated spend; no extra persistent service.
- Implementer -> integrator -> spec review -> adversarial code review. No workflow
  agent messaging, no secrets, no changes to other published directories.

Research source: `research-notes/workcell-measurement-research-2026-09-30/RESEARCH.md`
in the public artifact workspace. HomeBody's LiDAR and FoundationPose's missing CAD /
metric depth are not available inputs; those are not claimed as executed methods.

## Executed findings

- A1 similarity alignment preserves both generated fold angles exactly: left
  144.3163 degrees, right 140.9024 degrees. Source-view mask IoU is 0.8212 / 0.7604.
  Optimization took 1.511 seconds. The 3.4139 degree disagreement predates alignment.
- COLMAP camera/point BA: 676 real tracks, training reprojection error
  1.3767 -> 0.3278 pixels at 3x canonical resolution, 19.09 seconds. This is not a
  held-out or physical accuracy result.
- Native LIMAP point/line BA ran successfully: two scene lines; the guard-only
  ablation produced zero supported 3D lines. Sparse SIFT and explicit LK routes
  did not support independent two-face guard measurement.
- LoFTR indoor_new ran six pairs across the two A100s. It retained 25 / 33 / 44
  left/center/right tracks, all connecting photos 3 and 4. Checkpoint SHA256 is
  recorded and checked; source pixels, rejected matches and crop transforms are saved.
- The dense run exposed a false stability check: leaving out photos without
  training observations counted duplicate fits. The shared leave-view function now
  requires a contributing removed training view and independently retriangulated
  tracks with a real excluded-view residual. A two-view/four-camera regression
  prevents recurrence. Its original center acceptance is invalidated.
- Also removed a duplicate 24-track seed threshold. The existing seed's own
  minimum and two-plane checks now decide whether RGB initialization is usable.
  A 19-training-track regression with deliberately invalid depth verifies this.
- Production change is restricted to shape-preserving guard alignment; experimental
  candidates do not replace the scene models. Full cached scene interactions pass:
  all 52 object records, mouse/touch/keyboard wipe, model/photo picking, rotation,
  ground axes, per-board annotation, unknown dimensions, and 20/40 cm uniform export.

Artifacts live under `/Users/adam/Desktop/panoptes-public/research-notes/`:
`workcell-guard-controls-2026-09-30-a`, `workcell-guard-joint-2026-09-30-{a,b}`,
`workcell-guard-dense-2026-09-30-a`, `workcell-guard-dense-replay-2026-09-30-a`.
Each cloud call has input/code hashes, logs, timings and a spend ledger. Dense
replay reuses saved correspondences; it is not a fresh full-pipeline benchmark.
The prior actual four-photo full-pipeline time remains 334.60 seconds.

Final replay completed on `ap-jkxz3l4vJyUfUJgRU3TkO2`: all three boards remain
unsupported for physical angles in A2/A3. Center's 82.62-degree candidate has no
independent third-view check and no accepted angle/range. Left lacks second-plane
support; right does not separate into two stable planes. The shared-angle solver
therefore did not obtain a supported real-data solution. This is a negative result,
not proof that the two physical guards have different angles.

Five experiment allocations total 119.398 seconds of function work, estimated
$0.21196 at the recorded fixed-resource rates; call-window estimate $0.41503,
not invoice totals. The dense model ran once; the final validation replay was
5.46 seconds of subprocess work. Final report artifact:
`research-notes/workcell-guard-experiments-2026-09-30-final/page/`.

Runnable checks: `scripts/check_workcell_guard_joint.py`,
`scripts/workcell_guard_dense.py --self-check`, `scripts/check_workcell_vguard.py`,
`scripts/check_workcell_photo_report.py`, and `web/checks/photo-report-check.mjs`.
Use `scripts/workcell_guard_experiment_report.py` to package the reviewed A1 scene
and experiment evidence with the existing report UI.
