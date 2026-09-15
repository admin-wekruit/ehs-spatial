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

## CAD projection correction

The previous CAD/plan consumer projected the eight corners of a 3D bounding box and took their convex hull whenever a saved contour no longer matched the scene. That often produces a hexagon. The ownership repair correctly invalidated old representation snapshots but exposed this incorrect substitute for a mesh outline.

The shared producer now projects indexed mesh triangles onto the declared floor plane and unions them with the existing Shapely dependency. It retains concavities, holes, disconnected regions and genuinely edge-on line segments. Unreferenced mesh vertices do not affect geometry bounds. Each projection binds the exact asset hash (or primitive parameters), transform, coordinate frame, floor plane and source observation version. Source context remains visible in 3D but is not an object CAD contour.

CAD and plan consume the same polygon/holes/line data. No bounding-box outline substitutes for missing mesh contours. A pure position translation shifts the true contour exactly; changed rotation, scale, primitive parameters or source geometry invalidates the cached contour. Original frozen source CAD remains available as evidence.

Reprocessing revision `5086fd7a-fb3f-4ae1-827a-c9905e134e1a` produced revision `a960488a-c701-4e7f-bc42-8d253bb14950`, job `d8d84b46-fbe9-47db-8414-7ede9d3e0b2e`. All 28 records retain projections: 84 observation meshes and 9 existing active model representations. The central guard plate alone retains one polygon with 15 holes and 597 ring vertices; the right fold retains 7 components and 11 holes. These are observed mesh details, not completed hidden surfaces or new measurements. Identity decisions, original observations, measurements and model acceptance states remain unchanged.

Producer regression checks: 52 passed, including concavity, holes, disconnected components, unused vertices, rotated/nonuniform transforms, edge-on surfaces, primitive pose and ground availability. Actual full-scene projection also passed all 93 representation checks without additional model inference.

Publication `cfb403f4-d930-4a61-80e3-ad4497db1a43` preserves this revision. Export job `9b04968c-e42a-49c1-9b78-369498c5b550` passed GLB/Blender reopen checks for 87 meshes and 3 cameras (world geometry error 0); the same 9 unconfirmed generated candidates remain excluded. Catalog checks passed for all 7 publications, 586 exact read responses and 551 byte-verified assets.

Frontend TypeScript/build, interaction checks and 10 targeted checks passed. Per-photo consumption accounts for all 84 observed projections; every available new shape has `mesh_projection` provenance. Computer Use verified the unselected initial state, four-view selection/highlighting, full-screen CAD with visible concavity and holes, fit-to-selected, and a real CAD geometry click followed by the correct central-plate choice among overlapping objects.


## Whole-workcell CAD reference correction

The previous correction still used the currently viewed photo as the filter for CAD and the interactive plan. A valid 28-record scene therefore displayed 8/18/23 observed records depending on the photo; an older report/model view could show only 9. This was a scene-state selection bug, not a need to invent missing bounding polygons.

Each scene entity now persists a `cadReference`: its declared reference exposure and exact observation revisions. An existing model/source reference is retained; otherwise an explicit capture/job reference is used, and an object with only one source photo can use that source. Ambiguous sources remain unresolved. Normal reconstruction, append, mask repair, association, and GUI merge/split preserve or update the same reference contract. This is a chosen representation state, not proof that all photos describe one physical instant.

CAD and plan consume those references in a common registered native frame. Viewing another photo no longer changes the workcell drawing. Clicking a CAD object selects its real reference observation and corresponding photo/3D surface. No convex hull or bounding-box replacement is introduced. The repaired document retains 28/28 shapes in observed, model and point-cloud modes, with identical contour data across all three photo switches. The reference distribution is 6/1/21 across the three photos, retaining nine existing model/source anchors, fourteen explicit references and five sole-source references.

### Original CAD completeness and provenance

The original CAD remains a complete historical drawing with 33 plotted records. A shared importer/runtime helper now binds records through the pinned original inventory hash, an immutable original-SAM path/hash/size manifest, the exact source instance, a verified same-photo/canonical-grid mapping, and exact current-mask equality. A repeated filename or class label alone cannot establish identity. All source records remain in an auditable ledger: 9 are linked (6 current entities); 24 have no verified same-photo relation. Three additional records are now linked. Original pixels, measurements, observations, identity decisions and historical findings are preserved.

The source drawing was generated from historical estimated geometry. Its pixel/native coordinates have not been registered to the current scene, so geometric constraints are explicitly not applied. The ledger is a completeness check, not a claim that an old estimated drawing is measured ground truth. A new immutable report must be produced when those source relations are established.

### Saved result and checks

- Reference job: `ad4e7961-c9d4-417d-9274-80748d1e4989`, succeeded with head advanced and zero model calls.
- Scene revision: `38650c52-312a-49f5-b5ba-36abf47d8dcd`.
- Publication: `f292baf7-4f27-4962-b7df-6e931c614413`, “完整工位 · 全部对象 CAD 与来源核对”.
- Source manifest asset: `22378ab1-159d-45c8-9af5-e2647ec0d9ae`, SHA `fa385c1a12c66f8ff900d1e54a83e85ad8fc0c00a7f4428fc35ffda3d8c0df78`, 15 original SAM files.
- Blender export job: `1eaf75d0-5ef8-41a7-9e88-0b4d9579e08a`; GLB and Blender reopen checks passed for 87 meshes and 3 cameras, world-position error 0. Overall incomplete remains correct because nine existing unconfirmed generated candidates are excluded.
- Backend reconstruction/identity/importer tests: 63 passed, 2 database-conditional skips. Importer provenance suite separately checks missing manifest and same-path changed SHA.
- Runnable consumer check: `node --experimental-strip-types web/checks/scene-plan-reference.mjs <scene-document.json>` verifies actual production shapes, all three layers, photo-switch invariance and source observation freshness.
- Publication catalog: 8 immutable bundles, 595 exact read responses, 558 byte-verified assets. Prior report IDs and assets retained.
- Local Computer Use: default no selection; 28/28 on each of three photos; CAD #11 switches from photo 3 to its photo 1 evidence; CAD #22 switches back to photo 3 with visible green 3D surface; the newly linked source-CAD record opens the correct adjacent-fence entity. These are sampled visible interactions, alongside complete data-contract checks.

### Public release verification

- Product source: `f43e085`; Pages release: `c5db75f4fffb9aedc8259c76884f4e0dd1b5d352`, Pages workflow `35007116033` succeeded. The public document serves `app-B2zzmCf-.js` and `app-rsKcA2kU.css`.
- Modal `panoptes-publications` v11 serves the same eight immutable bundles with 2 GiB memory. Local catalog startup peaked at 926.8 MiB against the previous 1 GiB allocation. Initial public cold-start requests timed out; after startup, the actual public report loaded and completed the interactions below. This is not a latency benchmark or proof of the sole timeout cause.
- Public Computer Use on publication `f292baf7-4f27-4962-b7df-6e931c614413`: switched all three source photos and observed 28/28 in CAD and plan; clicked central guard plate #22 in CAD and verified the matching photo contour, green 3D surface, plan highlight and object details; enlarged CAD and located the selected true contour with concavities and holes.
- The full-screen CAD source button now exits full screen before navigating to the original drawing. The public interaction displayed the complete 33-record source ledger, with 9 linked and 24 awaiting verified source correspondence. Returned to four views, fitted the complete drawing and cleared selection before handoff.
- Public entry: <https://admin-wekruit.github.io/panoptes-workcell-report/app.html?report=f292baf7-4f27-4962-b7df-6e931c614413#/reports/f292baf7-4f27-4962-b7df-6e931c614413>. The document query intentionally loads the current application bundle; older immutable publications remain available.
