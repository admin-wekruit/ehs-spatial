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
