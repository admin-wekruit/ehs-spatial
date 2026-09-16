# Spatial measurements implementation plan

Goal: On-demand, linked model measurements in the report: two board planes' acute angle, exact mesh-surface minimum distance, and object projection overlap with an explicitly drawn CAD region. These are model estimates, not compliance verdicts.

Architecture: One Python calculation service reused by the live API and immutable public reader. Reuse asset decoding, transforms, and the CAD ground basis. Do not load geometry or calculate on initial report load. No paid model calls. The report inspector chooses a second object or a user-drawn region. The result includes revision/asset references and drawing primitives for the existing 3D/CAD viewers.

- [x] Implement shared measurements with finite inputs, same-frame checks, stale/hidden/excluded geometry rejection, planarity checks, and bounded exact distance traversal.
- [x] Leave a runnable geometry/API regression check: analytic angles, nonuniform scale, distance intersection/interior closest points, polygon holes/overlap, invalid region, missing frames, and unchanged publications.
- [x] Add read-only measurement routes, using existing NumPy/Shapely dependencies in the publication deployment.
- [x] Add bilingual inspector controls and 3D/CAD overlays. User-drawn regions are temporary browser state and never treated as confirmed safety zones. Abort stale requests and clear results when inputs/revision change.
- [ ] Measure the actual guard components, build/test, deploy, and exercise the public report through computer use.

Acceptance: each result names the two objects or explicit region, describes the measurement convention, shows model-estimate provenance, and links visually to the displayed scene. Unknown metric scale stays native units. Angles need no uniform metric scale, but use fully transformed geometry including anisotropic scale. A plane fit that fails geometric quality checks returns unavailable rather than inventing an angle. Multiview photography improves evidence but does not bypass model/observation validation.
