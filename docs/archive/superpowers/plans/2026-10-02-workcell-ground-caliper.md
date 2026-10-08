# Workcell unified ground and caliper implementation plan

> **For agentic workers:** use superpowers:subagent-driven-development. The user has authorized planning and implementation in this session.

**Goal:** Every model measurement uses the final saved floor and one explicit scale. The public report supports surface-point clearance, two-point distance/height difference, and a sampled planar region's clearance range. Model estimates remain distinct from verified physical measurements.

**Architecture:** Reuse the native world, existing report Z-up transform, mesh picking and SpatialMeasurements. Finalizing the floor refreshes derived heights and feet, never moves observed meshes to force agreement. A single model-measurement scale records its accepted/conditional/unknown status; accepted physical calibration is never inferred from a conditional model scale.

**Tech Stack:** Existing Python/NumPy/trimesh, React/TypeScript native WebGL viewer; no new dependencies.

## File ownership and sequence

1. Ground builder: `scripts/workcell_photo_metrology.py`, existing ground checks. Trace all clearance consumers, refresh caches against the final normalized plane, preserve source points/mesh bytes. Add an offset/tilt regression that fails for stale feet.
2. Caliper builder: `web/src/SpatialMeasurements.tsx`, `ReportScene.tsx`, `viewer/native-math.ts`, one small runnable check. Reuse ray hits. Enable only local operations in static reports; keep unavailable API operations disabled/hidden. Validate finite points, common frame and active representation. Invalidate picks after model/photo changes. Show signed height difference separately from straight-line distance; tilted three-point regions show a range.
3. Integrator: `scripts/workcell_photo_report.py`, `workcell_photo_oneshot.py`, `web/src/PhotoReport.tsx`, existing export/scale checks. Compute one model scale from the saved reference evidence, apply it to endpoint cards, caliper, grid spacing and export metadata. Keep failed joint reference scale null. Reuse the report's exact transform for exported geometry; normalize floor normal and offset together. Scale edits recompute every model readout and export consistently. Export measurement JSON with native points, ground and scale provenance.
4. Verification/publication: replay frozen artifacts into a new output directory; assert ground feet lie on the displayed floor, displayed and exported lengths agree, scale changes update all measurements, source meshes are unchanged unless explicitly rebuilt. Exercise real browser picking/rotation/clear/reset and export. Adversarial review before scoped commits/push and update `workcell-photo-direct/` only.

## Acceptance checks

- Nonunit, translated and tilted floor equations give the same geometric distance after normalization; stale clearance caches are refreshed.
- A point's foot lies on the shared plane; its signed height follows the floor normal, not screen vertical.
- Two selected points report Euclidean distance plus signed B-minus-A height; identical height means zero even at different depths.
- Three noncollinear sampled surface points report min/max height and inclination; no invented single distance for a tilted surface.
- Only actual model ray hits are measured. Cross-frame/stale representation points are rejected or reset. Empty space does not produce a point.
- Native geometry remains native in the report; one labeled model scale converts readouts and downloadable GLB. Conditional cm values never become accepted physical dimensions.
- Existing 52-object report, source views, rotate/zoom and explicit endpoint comparison continue working.

## Physical reconstruction follow-up

The saved left/right bottoms have different evidence (photo-4 percentiles versus measured/inferred fence rails). Common ground fixes inconsistent bookkeeping; it does not establish correct physical bottoms. Multiview bottom-edge reconstruction and joint 10/8.5/4 cm calibration remain separate acceptance items. Never impose 20/24 cm evaluation values or same-height prior and claim independent accuracy. A new GPU reconstruction run is required only when changing that inference; this measurement implementation can first be verified on saved geometry without paid inference.
