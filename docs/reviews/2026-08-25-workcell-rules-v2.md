# Workcell rules v2 — min_height predicate + safety-device vocabulary

Date: 2026-08-25. Branch `feature/ehs-spatial-mvp`. Additive only; the
Phase 0 contract freeze covers breaking changes, not new predicates.

## What was added

- `Predicate.MIN_HEIGHT` (`ehs_spatial/contracts.py`) — closes the
  1910.36(g)(1)-class vertical-clearance gap from
  `docs/reviews/2026-08-25-osha-compiler-exam.md`: rules demanding a
  *minimum* height (guarding fences, exit-route headroom) were previously
  inexpressible.
- Evaluation branch in `ehs_spatial/policy.py`: mirrors MAX_HEIGHT with the
  band inverted — height < threshold − band → FAIL (worst-first ordering by
  shortfall); within ±band → NEEDS_REVIEW note; above → PASS. Self-predicate
  (no object labels); every subject leaves a fact behind.
- Compiler prompt (`scripts/policy_compile.py`): one-line `min_height`
  description + added to the empty-object-labels list, so the live compiler
  can emit it.
- Perception vocabulary (`ehs_spatial/providers/sam3.py`), appended without
  reordering: `safety sensor` (light curtain / photoelectric sensor / safety
  scanner), `emergency stop button` (e-stop button / red emergency button),
  `warning sign` (safety sign / hazard sign), `safety light` (stack light /
  signal tower / andon light). First-hit synonym fallback unchanged. The
  production pipeline now makes 44 SAM calls per 4-image run (was 28).
- Reviewable compiled fixtures in `docs/policies/compiled_v2/` (prose in
  `docs/policies/workcell_v2.md`); the app's `POLICIES_DIR` remains
  `outputs/policies/compiled` — placement of v2 specs into the app's dir is
  an orchestrator decision. p03 (safety sensor at cell opening) is compiled
  as an honest refusal: presence-required semantics degrade to
  INSUFFICIENT_EVIDENCE exactly when the sensor is absent — the violation
  the rule exists to catch — and "cell opening" is not a perception class.
- Probe: `scripts/workcell_probe.py` (house pattern: main-repo disk cache
  under `outputs/workcell_probe_v1`, `--live` spend gate, printed call
  counts, ~$0.01/call).

## Live probe — new classes on real cached imagery

Images: owner's factory photo (`runs/demo-real-factory/input/image_01.png`)
and two MTMC Warehouse_016 frames (Camera_003600, Camera_03_004200).

**The probe is partial.** 12 of 42 planned calls completed (~$0.12) before
the fal account locked with "Exhausted balance"; two retry passes (30 calls
each, zero cost — locked calls are refused) confirmed the lockout persists.
`ERR` rows below are unprobed synonym chains, NOT misses; re-running
`uv run --env-file .env python scripts/workcell_probe.py --live` after a
top-up completes them for ≤ $0.30 with no code changes.

| image / class | result | detail |
|---|---|---|
| factory / safety sensor | **miss** | all 4 synonyms probed, 0 masks — a complete, honest miss |
| factory / emergency stop button | ERR | canonical + "e-stop button" probed empty; "red emergency button" blocked |
| factory / warning sign | ERR | only "hazard sign" probed (empty); canonical blocked |
| factory / safety light | ERR | canonical probed empty; 3 synonyms blocked |
| mtmc-cam0 / safety sensor | ERR | "photoelectric sensor"/"safety scanner" probed empty; 2 blocked |
| mtmc-cam0 / emergency stop button | ERR | canonical probed empty; 2 blocked |
| mtmc-cam0 / warning sign | **HIT** | "hazard sign" → 3 masks, top score 0.62; crop-verified: red prohibition signs on a pillar. Canonical "warning sign" phrase unprobed (blocked) |
| mtmc-cam0 / safety light | ERR | all 4 blocked |
| mtmc-cam3 / * | ERR | all blocked (lockout hit before this frame) |

Honest reading of what did complete: SAM 3 knows "hazard sign" and finds
real signage in a warehouse frame, while the factory photo produced zero
masks across 7 probed safety-device prompts — these devices are small,
distant, or absent in that view. Nothing was tuned to force a hit.

## Fence min-height verdict (offline, production `evaluate_policies`)

`p01-guarding-fence-min-height` (min_height, safety fence, 1.8 m) vs the
cached `runs/demo-real-factory/scene.json` — mono capture, 1 frame, band
±0.35 m:

**FAIL** — worst-first violation: `entity-safety-fence-05` measured
1.18 m (limit 1.8 m). Per-segment facts:

| fence segment | height | side of the 1.8 m ± 0.35 m band |
|---|---|---|
| entity-safety-fence-05 | 1.18 m | FAIL (< 1.45 m) |
| entity-safety-fence-08 | 1.63 m | in band → review note |
| entity-safety-fence-04 | 1.76 m | in band → review note |
| entity-safety-fence-01 | 1.79 m | in band → review note |
| entity-safety-fence-07 | 1.96 m | in band → review note |
| entity-safety-fence-06 | 2.43 m | pass (> 2.15 m) |
| entity-safety-fence-02 | 4.68 m | pass (implausible; see limitations) |

(entity-safety-fence-03 was filtered by the footprint evidence gate.)

The expectation was PASS based on the earlier H≈2.43 m measurement — that
segment (fence-06) does clear the threshold individually, but the policy
grades every detected fence segment and the scene's fence is fragmented
into 8 partial detections, including a 1.18 m fragment. The FAIL is the
engine working as designed on imperfect segmentation, reported without
tuning; whether fence-05 is a genuinely short section or a truncated mask
is exactly what the NEEDS_REVIEW/violation evidence trail is for.

Companion results on the same scene: `p02-estop-within-reach-of-robot` →
INSUFFICIENT_EVIDENCE (no e-stop entities in this scene — it predates the
v2 vocabulary); `p03-safety-sensor-at-cell-opening` →
INSUFFICIENT_EVIDENCE with the unsupported_reason surfaced.

Full envelope: `outputs/workcell_probe_v1/report.json` (main repo).

## Limitations

- Heights come from a mono reconstruction (camera-height scale, factor
  1.51); the ±0.35 m mono band applies and dominates near-threshold fence
  segments — 4 of 7 sat inside it.
- Fence entities are per-mask fragments, not one merged fence: min_height
  over fragments punishes truncated detections (fence-05) and rewards
  merged tall blobs (fence-02 at 4.68 m is not a credible fence height).
- The probe grades detectability of the new classes, not their absence
  from the world: 30 of 42 cells are unprobed until the fal balance is
  restored.
- Presence-required rules ("a sensor must be present") remain out of
  vocabulary by design; p03 documents the refusal.

## Gates

- `uv run --extra dev python -m pytest -q`: 334 passed, 2 skipped (baseline
  331+2; +3 min_height tests in `tests/test_policy.py`; vocabulary-count
  updates in `tests/test_sam3.py` / `tests/test_pipeline.py`).
- `ehs_eval generate` + `offline`: pack regenerates, offline report
  `"passed": true`, all 4 cases green.
