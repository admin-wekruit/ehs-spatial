# Observed geometry ownership repair

The report could select a guard subpart in the photo/CAD but not highlight its actual 3D surface. The old scene assembly removed generated-object masks from the context before extracting observed regions. Subparts therefore inherited carved remnants or no mesh. The central guard plate had 366 faces; its original mask supports 5,396. The right fold had no owned mesh; its mask supports 3,106.

## Shared change

- Build each observation's owned surface directly from its exact mask and native pointmap before context carving. Preserve the observation/image/frame references, entity ID, and original measurements.
- Keep a complete context per source photo. Scope both owned surfaces and context to that photo instead of combining different photographed poses.
- During picking, smaller confirmed observation surfaces win coplanar ties over broad surfaces; actual nearer geometry still occludes them.
- Exclude stale representations from rendering and export, retain them in immutable history. Compute display bounds from referenced vertices and fit the visible object geometry while keeping the background rendered.
- New complete reports default to the reconstructed photo scene. Generated assets remain separately available as the generated-model comparison. No generated whole mesh is silently assigned to a subpart.

## Reprocessed publication

- Project: `a2b7c04d-0162-488b-b7db-37711a37ea62`
- Revision: `5086fd7a-fb3f-4ae1-827a-c9905e134e1a`
- Publication: `a88feade-220d-4f82-b995-e0efec077a88`
- Repair job: `b500400c-8a52-4e01-9546-fc4cec850dcc`
- Coverage: 28 business records, 84 observations, 84 owned surfaces, 3 photo contexts. Existing identity decisions, masks, cameras, measurements and generated assets retained. No additional model inference.
- Export job: `efa8cc4b-75c6-4c49-8d40-00179b6c5d76`. Blender reopen validation passed for 87 observed meshes and 3 cameras; world geometry error 0. Nine existing unconfirmed generated candidates remain excluded and the overall export correctly remains incomplete.

## Checks

- Reconstruction tests: 48 passed, covering generic capture, append, mask repair, idempotent rebuilding and lossless evidence preservation.
- Blender stale-representation and existing export checks: 6 passed.
- Renderer/readers, photo scoping, picking, fit bounds, selection lifecycle, report interactions and TypeScript checks passed.
- Publication catalog: 6 bundles, 579 exact read responses and 546 byte-verified assets; history, conflicts, ranges and write rejection passed.
- Computer Use on this publication: list selection visibly highlights the robot, central guard plate and right fold; native 3D clicks returned the robot, central plate, left fold and floor-marking identities. These are actual interaction samples, not a claim of manually clicking all 84 surfaces. A click through occluding wire geometry correctly selected the foreground record.

## Limits

This fixes record-to-observed-geometry ownership, not missing observed backs or generated-mesh part segmentation. Fourteen cross-photo groups and twelve pending identity records retain their existing decisions. Generated whole-asset placement, source geometry quality, scale calibration and current EHS assessment remain separate evidence requirements.
