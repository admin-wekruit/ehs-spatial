# Current goal: complete, positioned coarse workcell models

This goal incorporates the user's latest clarification. Recognizable approximate
geometry is sufficient; correct object identity and spatial placement take
priority over fine detail. The cumulative paid-compute ceiling is USD40, including
existing charges and unresolved reservations.

## Delivery

The current workcell scope contains 26 report records, after the user explicitly
excluded the adjacent-cell fence and floating indicator. All 28 source record IDs
remain retained. Each in-scope record must resolve to selectable, visible geometry
in the browser and to the same identity in the source photo and CAD. Ordinary
objects need approximate models; ground needs its observed reference surface;
an assembly needs an explicit link to its modeled components without duplicated
geometry. These categories remain visible rather than being counted as 28
independently generated solids. This is a target, not current completion.

Models must retain recognizable main shape, approximate size, orientation,
position and major openings. Screws, cables, textures, small joints and individual
fence bars are optional. Partial observation, assumed hidden extent and unknown
physical scale remain recorded. A coarse display model does not grant metric EHS
measurement eligibility.

## Shared implementation path

1. Discover objects and review omissions using the existing VLM inventory path.
2. Segment each owned observation on its original photo; preserve pixel mappings.
3. Select valid 3D support through that exact mask in the capture's native frame.
   Preserve cameras; reject background-contaminated support instead of silently
   using the full image or another object's points.
4. Associate observations before modeling. Do not union unregistered frames or
   incompatible poses merely because labels agree.
5. Reuse existing adequate models. For new coarse geometry, use the existing
   object-generation provider with owned mask/pointmap, or the existing box and
   cylinder representations when that shape is supported by the object evidence.
   Simple primitives are object-specific choices, not a universal replacement.
6. Fit and validate model position, orientation and scale against the owned 3D
   support and source-camera projections. Keep coarse shape generation separate
   from scene placement; provider-local coordinates must not replace the source
   coordinate frame. Fine-detail differences alone do not request regeneration.
7. Save each admitted model through the shared representation/version path;
   derive CAD from that exact active model and transform. Preserve observed CAD
   as separately labeled evidence, not an independently positioned model.
8. Connect analysis to this modeling stage with frozen entity targets, durable
   jobs, existing budget reservations and reusable per-object results. Partial
   jobs retain objects and report precise missing input; they do not claim that
   a newly uploaded workcell is fully modeled.

## Current source-level gap

`run_analysis` calls `_associate_and_surfaces`, which extracts each mask's native
3D support and writes observed meshes. The actual worker calls the outer
`run_capture_pipeline`, which already invokes generation and review, including
a durable research continuation when configured. The earlier diagnosis based on
`run_analysis` alone was incomplete. Reuse this orchestration; finish the working
provider/configuration, source-constrained placement and coarse acceptance path.
Relabeling observed surfaces as generated complete models does not close the gap.

The current shape prompt is `coarse-layout-shape-v2`. The explicitly versioned
`coarse-layout-position-v1` profile requires 90% of observed mask pixels within
5% of the mask bounding-box diagonal of the projected mesh, relative depth median
at most 5%, and at least 80% of compared depth pixels within 15%. The complete-mask
IoU/precision requirements remain unchanged. Exact coverage, IoU and depth P95
remain in every result, including failures. These are engineering checks, not
physical accuracy guarantees. Detailed-mode thresholds and historical results
remain unchanged; the profile and thresholds participate in the evidence hash.

Hosted SAM3D shape coordinates are initialized using the upstream row-vector
quaternion/export convention and scaled from the owned source depth. The shared
refiner can fit bounded positive uniform scale as well as rigid pose, then checks
every owned source view. It never replaces source camera calibration with the
provider's estimated camera. A fit does not approve semantic shape or physical
placement. Missing views and conflicting main-body depth cannot pass.

## Executable acceptance

- Freeze and retain all original 28 record IDs; explicitly excluded records stay
  archived as source context. Every in-scope row has a photo link, a
  model/reference/component binding, its native transform and model-derived CAD.
- Validate mesh bytes, entity ownership and CAD/model transform equality.
- In the deployed browser, select each row and select back from CAD/3D; verify
  the photo highlight, model highlight and detail panel refer to that same row.
- Review source-camera overlays for gross offset, wrong scale/axis, wrong object,
  merged neighbors and blocked major openings. Do not require fine-detail parity.
- Exercise the same analysis-to-model path on new photos without report-specific
  entity IDs or manual source-file edits. Reopening a result must not regenerate.
- Keep the completion count below 28 until the actual row checks pass; report
  ground and assemblies separately so a count cannot conceal missing models.

The application goal widget still contains an unfinished paused goal with the
older USD20 wording. Its tool cannot replace an unfinished goal. This document
records the updated scope and USD40 authorization without falsely completing
the previous goal.


## Delivered geometry snapshot

The latest snapshot is publication `c34292d0-74f0-431d-b35e-4d280f1c2df6`,
revision `892f684c-60d5-4c52-a214-53f9a66d62b9`: 26 model records, one ground
reference and one component-linked record, retaining all 28 identities. Explicit
coarse open-frame generation is now part of the shared job path. Detailed
acceptance evidence and remaining fresh-photo/quality gates are in
`PHOTO-PIPELINE-ACCEPTANCE-20260916.md`. Geometry coverage alone does not complete
all acceptance items above.

The browser now defaults CAD to **current model projection**, so the drawing and
3D use the same meshes/transforms. Photo-observed CAD remains an explicitly
selectable source layer. Composite rows reuse their linked components' validated
contours without inventing an enclosing solid. The header separates geometric
record coverage from independent-model, reference, and composite counts.

The current all-record CAD audit passed: 26/26 active model projections and 28/28
observed projections validated; zero source-reference errors. The final browser
check must separately confirm the composite row and ground reference, as backend
counts alone previously missed a front-end composite projection gap.

## Explicit workcell boundary, 2026-09-16

The user marked the adjacent-cell fence (d22e72d1-ca73-5ea4-a5f3-68057d7af6dd)
and floating indicator (a214c225-ba7a-5550-9af8-5bea16c2cfe4) for exclusion.
The shared `setWorkcellScope` edit records this manual decision and its base
revision, retains observations/assets/representations, and uses existing source
context and visibility semantics for model targets, CAD and the report. Explicit
scope affects safety targets; display visibility alone does not. Inclusion can
be restored and ordinary edit-batch undo preserves the full original document.

Publication d6c2d4d3-4769-4526-a0f7-73de34fa2f5b fixes revision
f5f4b1d4-bb55-4c13-ba02-b44e706340f2: 26 in-scope records, 24 independent models,
1 reference surface, 1 composite preview, 0 unresolved geometry bindings and
0 source-reference errors. Historical publications are unchanged. This is a
coverage result; physical placement and independent shape review retain their
existing uncertainty. No paid model call was required for the exclusion.
