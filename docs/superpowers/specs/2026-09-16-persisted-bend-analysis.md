# Persisted per-object bend analysis

User request: analyze intrinsic adjoining panel angles during processing and let
report readers choose whether to display them; opening the same report should
show the same saved result without requiring a manual Calculate action.

Reuse the existing same-mesh bend fitter. After a worker produces a scene, analyze
all visible in-scope entities and store each outcome in entity.bendAnalysis (derived analysis, separate from observed measurements).
Bind it to an algorithm-version + model/asset/pose/frame fingerprint. Reuse only
exact matches; a model edit cannot silently reuse old world-space annotations.
Save measured, unsupported, skipped and failed outcomes distinctly. This remains
model inference, not a field measurement or compliance judgment.

Publication preparation runs the same analysis for frozen revisions and writes a
versioned derived JSON route, preserving original frozen scene records. Existing
reports are processed through this path as well. Serving/reading this route must
never decode meshes or start model calls. The live API reads stored worker
results and reports not-processed for absent/stale analysis.

Report loads the small analysis record once per revision. The measurement panel
automatically displays the selected object's saved bend result. A detected-bend
selector switches the ordinary object selection across all views. A report-level
checkbox shows/hides angle overlays (default on for the selected object only).
Other manual measurement modes retain their existing behavior.

Checks: exact-input reuse with zero asset reads, invalidation for pose/mesh/frame
changes, single-plane/unsupported and failed outcomes, worker persistence,
prepared HTTP serving, selecting a different object, fresh-page result loading,
and angle hide/show in the public report. No additional model generation spend.

## Verification

Implemented through the shared scene-producing worker stage and publication
preparation. The prepared catalog processed 19 revisions. Current report:
26 in-scope records; 2 measured bends, 20 unsupported fits, 2 complexity-limited
calculations, 2 records without independent models. No generated geometry or
identity decisions were changed. The saved current-scene document validates
against the existing schema and identity provenance constraints.

`tests/check_scene_measurements.py` covers persisted outcomes, exact-input reuse,
pose invalidation, exclusions, worker attachment, immutable publication derivative,
and report reads that neither calculate nor load the full revision. Existing
measurement geometry, publication and report-loading checks pass. Reconstruction
pipeline tests: 23 passed; 30 DB-dependent outbox checks skipped in this environment.
Frontend checks/build pass. Fresh local page showed the left angle automatically;
hide/show removed/restored the overlay; selector changed all-view object selection
to the right wing (146.9 degrees); refreshing retained the saved result with no
Calculate click. The independent Modal worker deployment requires its preexisting
audited image digest and DB/storage secret, neither configured in this shell;
no new remote worker image is claimed. Shared worker source and publication path
are updated; public report preparation does execute the analysis before release.

Public acceptance: source 8eaa1b3, Pages 0ed1a2e, successful Pages run
35157886657. Modal publication service redeployed with the prepared derivative.
Current analysis GET is 12,886 bytes and contains all 26 in-scope outcomes. Reloaded
the user's actual public tab without calculating: left wing 145.4 degrees appears
in 04. Unchecking Show angle annotations removes it; selecting the right wing and
rechecking shows 146.9 degrees. The detected-bend selector includes both objects.
