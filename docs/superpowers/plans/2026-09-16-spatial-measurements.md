# Spatial measurements implementation plan

Goal: On-demand, linked model measurements in the report: two board planes' acute angle, exact mesh-surface minimum distance, and object projection overlap with an explicitly drawn CAD region. These are model estimates, not compliance verdicts.

Architecture: One Python calculation service reused by the live API and immutable public reader. Reuse asset decoding, transforms, and the CAD ground basis. Do not load geometry or calculate on initial report load. No paid model calls. The report inspector chooses a second object or a user-drawn region. The result includes revision/asset references and drawing primitives for the existing 3D/CAD viewers.

- [x] Implement shared measurements with finite inputs, same-frame checks, stale/hidden/excluded geometry rejection, planarity checks, and bounded exact distance traversal.
- [x] Leave a runnable geometry/API regression check: analytic angles, nonuniform scale, distance intersection/interior closest points, polygon holes/overlap, invalid region, missing frames, and unchanged publications.
- [x] Add read-only measurement routes, using existing NumPy/Shapely dependencies in the publication deployment.
- [x] Add bilingual inspector controls and 3D/CAD overlays. User-drawn regions are temporary browser state and never treated as confirmed safety zones. Abort stale requests and clear results when inputs/revision change.
- [x] Measure the actual guard components, build/test, deploy, and exercise the public report through computer use.

Acceptance: each result names the two objects or explicit region, describes the measurement convention, shows model-estimate provenance, and links visually to the displayed scene. Unknown metric scale stays native units. Angles need no uniform metric scale, but use fully transformed geometry including anisotropic scale. A plane fit that fails geometric quality checks returns unavailable rather than inventing an angle. Multiview photography improves evidence but does not bypass model/observation validation.


## Verification — 2026-09-16

- Source implementation: `db68401`; public Pages build: `b495c3d`; Pages run `35144541362` succeeded. The existing Modal publication reader was deployed with the shared calculation module.
- `python tests/check_scene_measurements.py`: analytic plane angles, nonuniform scale, closest surface points, edge/face intersections, projection holes, unchanged input documents, excluded/hidden/stale models and ground qualification passed.
- Existing `check_report_loading.py`, `check_publication_site.py`, 19 publication feedback tests, `npm --prefix web run check`, and production TypeScript/Vite build passed. Vite's existing large lazy policy-editor chunk warning remains.
- Computer use on the public report `d6c2d4d3-4769-4526-a0f7-73de34fa2f5b`: center guard panel vs right wing returned 66.32 degrees (smaller fitted-plane angle) and 0.0003531 native units minimum surface separation. Both matched the local shared service. These are model calculations, not validated field values. Their stored model-quality reviews still contain rejections.
- A region drawn through the public CAD UI returned 0.1538 native units squared, 8.1% of that specific rectangle; an empty region returned zero locally. Mouse and keyboard corner selection worked. Switching language preserved the result; changing objects cleared it. Phone width 390 px and the default desktop local viewport were checked.
- Public 3D rotation changed camera orientation and retained the correctly projected measurement overlays. Browser error log was empty. Existing 26 displayable records / 24 independent models and CAD correspondence stayed unchanged.
- Calculations run only after the user clicks. No publication rewrite, model inference, paid GPU call, history prefetch, or dataset-specific object mapping was added. Units remain native model units until a separately verified metric conversion is provided.


## Single-panel ground inclination — 2026-09-16

- Added `inclination` to the same shared measurement service and public endpoint. It fits the transformed board mesh against the coordinate frame ground normal. Horizontal = 0 degrees, vertical = 90 degrees; deviation from vertical is its complement. No second object is required.
- The blue datum passes through the board center parallel to ground; it does not claim the physical floor is at that elevation. The yellow arc runs from a ground-parallel direction to the fitted panel plane. Missing, zero or non-finite ground normals return unavailable.
- Regression checks cover 0/30/80/90 degrees, arbitrary/reversed ground normal, arc endpoints lying in their respective planes, invalid ground and an HTTP request without entity B. Geometry/API checks, frontend interaction/scheduling checks and production build passed.
- Source `8ccd37b`; Pages `1671709`; successful Pages run `35145999196`; shared Modal backend redeployed. Public computer-use calculation on the center guard board returned 57.64 degrees to ground and 32.4 degrees from vertical, matching the backend (57.63866282807513). A mouse drag rotated the scene while preserving the projected orange board, blue datum, yellow arc and numeric label. The result remains an unverified model estimate; existing model-quality evidence is retained.

## Three-point edge angle — 2026-09-16

- User clarified the requested quantity: the angle between two meeting edge lines. Added the default `edge_angle` mode: first edge point, shared vertex, second edge point. It returns 0–180 degrees, including obtuse angles, from actual 3D mesh surface coordinates; it is not a screen-space angle or plane-normal angle.
- Reuses GPU object picking, native DOMMatrix unprojection, the loaded mesh and existing measurement overlays. Triangle intersection occurs only on a measurement click. No server request, extra geometry download, dependency, or publication edit is needed. Background clicks are rejected; object/frame changes clear picks; source model references and existing placement/quality states remain attached.
- Analytic tests cover acute/right/obtuse/straight angles, invalid and coincident points, surface intersections, winding reversal, background misses and backward rays. TypeScript/build and existing interaction/render-scheduling checks passed.
- Local computer-use checks: selected three surface points, calculated both acute and obtuse examples, rotated the scene with the same result/attached markers, rejected an empty-space click without incrementing the count, and used arrow keys plus Enter to pick a surface. These manually chosen examples validate interaction, not the real installation angle.
- Public release verified: source `13313a5`, Pages `ff732e7`, successful deployment `35148779853`. On the public report at narrow viewport width, started picking, zoomed the real scene, rejected a background click, selected three mesh points, and calculated 95.98 degrees for that manual test selection. The inspector and 3D arc agreed; point labels 1/2/3 remained visible. This is an interaction check, not a claimed measurement of the user's exact annotated edges.

## Corrected annotated quantity: edge versus vertical — 2026-09-16

- User's new drawing clarified the requested green arc: one sloping edge relative to a vertical through its upper endpoint. The earlier three-point corner was the wrong reference for that request.
- Added default `edge_vertical`: pick the upper and lower endpoints of the same edge. Reuse the current frame ground normal and existing angle helper; draw a red vertical from the first point, orange edge and green arc at that first point. Angle is deviation from vertical (0–90 degrees), independent of screen orientation. Missing/invalid ground is rejected, not replaced by world Z.
- Reused the existing mesh point picker with a two-point count, keeping three-point measurement separately available. The vertex marker is now the first point in this mode. Source model, placement and quality evidence remain attached.
- Tests cover vertical/horizontal/30/60-degree edges, arbitrary and reversed ground normals, reversed endpoint order, and invalid ground. Frontend checks and production build passed. Local browser verification completed two surface picks and displayed the red reference and green arc at point 1; the manual example is not claimed as the user's exact physical edge measurement.
- Public check: source `d780aea`, Pages `76162be`, deployment `35149844316` succeeded. In the public narrow viewport, the default mode required exactly two points, completed a manual surface selection, and showed the red vertical at point 1 with the green angle arc. The inspector and 3D label agreed (70.29 degrees for those test picks). This verifies the defined measurement and UI, not exact recovery of the user's annotated physical edge.
