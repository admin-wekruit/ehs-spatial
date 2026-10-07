# Restore full photo report through existing components

> Use superpowers:subagent-driven-development. User explicitly requested correcting missing objects and restoring prior report interactions; implementation/publication authorized.

**Goal:** Four raw photos produce a complete per-object model report, with existing object selection/measurement cards and matched photo/model slider.
**Architecture:** Reuse native-world evidence and standard SceneDocument contracts; feed existing ReportScene, PhotoView, SpatialView, and ObjectFacts through an injected local asset resolver. New snapshot entry adds reference controls and comparison. Retain automatic two-A100 oneshot; existing masks never become fresh inference inputs.
**Tech:** Existing Python/trimesh/OpenCV/measure_observed_points, React/native-viewer/Vite. No new dependency.

- [x] Object builder: preserve individual masks/identities, produce objects.json and small-object meshes using existing reconstruction._mesh/spatial.primitive_mesh/measure_observed_points. Cover button, lamps, light curtains, signs, barriers, panels and cable tray if supported. Missing categories explicitly reported; one-view physical dimensions null. Add runnable node/measurement/source validation.
- [x] Integrator: extend existing SAM words, run builder automatically; assemble standard SceneDocument with validate_document and canonical_measurements. Apply one rigid world transform to meshes and cameras, retain source pixel correspondence, and export same catalog objects to complete GLB.
- [x] UI reuse: injectable SceneResources for local assets; existing components remain the renderer/selector/cards. Add same-camera image/model slider with keyboard-accessible range and persistent object list. Show selected-object height/clearance above reused details; retain adjustable reference and actual GLB export.
- [x] Verify: standard-schema/transform/unknown-size tests; fresh raw-photo run on one ephemeral2xA100; browser select button/lights/posts from list, source photo andmodel; compare at slider0/50/100; sourceposealignment; scale/card/download consistency; mobile.
- [x] Independent spec then code review, publish only workcell-photo-direct/, check live, save timings and ledger, report short Chinese with actual screenshots.

Root cause: prior path generated only five scene groups and omitted object inventory despite extra SAM classes. This was output/report integration regression, not inherent speed/coverage tradeoff. Old complete report remains evidence for expected category coverage; its coordinates/old outputs are not injected into new reconstruction.

## Reproduction and review

Build the existing shared frontend before starting one-shot inference:

```sh
cd web
node node_modules/vite/bin/vite.js build --config vite.photo.config.ts
cd ..
python scripts/workcell_photo_oneshot.py --images FIRST.jpg SECOND.jpg THIRD.jpg FOURTH.jpg --out NEW_OUTPUT_DIRECTORY --button-diameter-m .2 --button-height-m .2
```

The run snapshots the built UI before allocating one ephemeral two-A100 container. Parallel geometry and segmentation precede parallel robot/cart generation. Object catalog, source measurements, individual meshes and the shared report contract are assembled in that container. The original photos remain the inference inputs.

Runnable checks:

```sh
python scripts/check_workcell_photo_objects.py OUTPUT_DIRECTORY
python scripts/check_workcell_photo_report.py OUTPUT_DIRECTORY --require-lights
python scripts/workcell_photo_geometry_check.py OUTPUT_DIRECTORY
node web/checks/photo-report-check.mjs OUTPUT_DIRECTORY SCREENSHOT_DIRECTORY
```

Independent spec/code review corrections: retain tiny unsupported detections as selectable source records; rotate orientation evidence with the scene; suppress unsupported physical upright/complete dimensions; bind cameras using the existing geometryBindings contract; preserve every disconnected source polygon. GLB export converts report Z-up to glTF Y-up and uses the selected uniform scale.

The first fresh run produced full inference artifacts but page packaging collided with a concurrent Vite rebuild. Its output and spend ledger are retained as `workcell-photo-complete-2026-09-30-a`; no finished oneshot timing is claimed. The final run uses a frozen viewer snapshot.

Final fresh run: `research-notes/workcell-photo-complete-2026-09-30-b`, Modal `ap-tnAxPgS6WzjIdyzzO2nlea`, 225.12 seconds end-to-end. All51 source records have renderable geometry,91 observations; the count includes partial floor markings and provisional associations, not51 ground-truth unique physical objects. Both entrance lamp records were visually checked. Known lower fence rail clearance is0.34283m under the provisional20cm anchor height. Local schema, geometry, source/model clicks, mobile and exported actual geometry checks passed. No claim of full physical measurement accuracy or photorealistic small objects.

Published site commit: `816c8e1`, only `workcell-photo-direct/`. Pages deployment36779863068 succeeded. Live browser loaded51/51 models and225s timing; selected button, both entrance lamps, post and fence; slider/free3D and GLBdownload passed with no browser console errors. Live evidence saved in the final run browser-check directory. URL: https://admin-wekruit.github.io/panoptes-workcell-report/workcell-photo-direct/?v=816c8e1-verified

## V-guard and interaction correction — 2026-09-30

User regression: the middle comparison handle was decorative, free orbit was not obvious, guard was merged with cart/work-platform geometry, and offline export hid existing angle analysis.

Corrections reuse ReportScene/SpatialView and scene_measurements. Pointer capture enables actual mouse/touch dragging; two prominent modes expose comparison/free orbit; selected plane and inspector share state. SAM independently segments yellow/black guard panels, excludes their pixels from cart generation, and examines all existing OWLv2 cart proposals. No old masks/coordinates are used.

Controlled same-input probe found that restoring DEFAULT RecGen did not repair the connecting panel: capture-order views 1/2 were occluded. Automatic top-two supported-area ranking selected photos 3/4 and restored the connecting panel with FAST. The existing x7.refine then aligns the generated mesh to source masks/depth. All4 views are checked and recorded, with no physical ground-truth claim.

Fresh end-to-end oneshot: research-notes/workcell-vguard-complete-2026-09-30-b; Modal ap-KtRIBNjx8CGjHpJBlt2QKi, one ephemeral2xA100-80GB. Total184.71s; container142.95s. No human mask/placement/old model injected. 50 records,92 source observations,134 linked nodes; counts are not ground-truth unique objects. Guard generation views3/4; bounded refinement1.604s; occlusion-aware source-mask IoU0.798/0.749 on generation views; other two0.500/0.407. Those other views remain poor and are not accepted. Local plane angles are model estimates against estimated ground; ground angular error remains unknown and full bend is unsupported. Backside, texture and hidden shape remain inferred.

Checks: check_workcell_recgen_worker.py; check_workcell_vguard.py; check_workcell_photo_objects.py; check_workcell_photo_report.py --require-lights; workcell_photo_geometry_check.py; browser check includes actual center mouse/touch/keyboard drag, source/model selection, actual orbit change, 20/40cm exported geometry, saved plane card/reference line and mobile. Typecheck/build and existing frontend check pass. Independent code review found no blocker. Static server has only an unused favicon404.

Ledger: final run estimated container window$0.254, call window$0.304, actual invoice unknown. Earlier complete-a:414.89s with substantial queue, not final latency. Diagnostic default/FAST probe ap-gTdLk8NF8A1b2D7EvIp84T:99.31s wall/85.53s container; its separate ledger and exact scripts retained under workcell-guard-quality-2026-09-30. Segmentation probe/failed import run retained under workcell-vguard-routing-2026-09-30. These development costs are separate from the final single-run estimate.

Site commit1ce3c54 updates only workcell-photo-direct/. Live validation to be appended after Pages completes.

Live verified: Pages run36785242848 succeeded for1ce3c54. URL ?v=1ce3c54-verified loaded50/50models and185s timing. Center drag60→53; free-orbit canvas changed; V-guard local face5 card84.7deg toground/5.3deg fromvertical matched3Dannotation; GLBdownload workcell-photo-4-height-20cm.glb succeeded; browser console0errors. Screenshot browser-check/live-angle.png. This is model-derived orientation, not independently measured ground-angle accuracy.


## Three physical boards, intrinsic bends and grounded scene — 2026-09-30

User clarified that each of the three guard boards contains two folded faces. Ground inclination of arbitrary local patches was the wrong output. Reuse existing fitted_bend/analyze_bends and isolated model annotations: flat is180deg, right angle90deg. Split the current generated guard using three independently segmented instances in the best complete source view, then uniquely associate observations across views. Every original triangle belongs to exactly one board, with unchanged coordinates. No manual coordinate cuts, historic masks or models. Saved floor defines the free-view grid and world XYZ; cosmetic grid does not alter meshes or exported measurements.

Fresh raw-photo oneshot: research-notes/workcell-three-boards-complete-2026-09-30-a, Modal ap-IJ9Pnx5wL3K33Q9t2GysX4, one ephemeral2xA100-80GB. End-to-end334.60s; container180.32s; call including cold start317.28s. 52 records,97 observations,136 linked nodes; not ground-truth identity count. Left intrinsic bend144.9939deg, right141.5282deg; center unsupported. Angles are inferred from generated surfaces, not independent physical measurements.

Center investigation retained under workcell-center-probe-2026-09-30: independent multiview3/4 generation had no stable bend; single-view numerical fit77.8696deg failed source consistency (mask IoU~0.456/0.459), so it was rejected and not injected into the final report. Modal ap-9nPY6NQsnECqxyeaYZQik5,63.319s container,246.698s wall; separate estimated spend ledger retained. Observed depth and previous DEFAULT/FAST meshes also did not establish a stable center bend. This limitation remains open.

Checks passed: check_workcell_vguard.py (exhaustive disjoint unchanged triangles and angle references), check_workcell_photo_objects.py, check_workcell_photo_report.py --require-lights, browser check (all3 selections, left/right fold annotations, unsupported center without stale value, ground grid/camera orbit, drag, source/model clicks,20/40cm exports,mobile,noAPI). Typecheck/build/existing frontend checks passed. Independent review found and verified fixes for missing child-GLB page assets and regression-fixture symlink write-through; no remaining blocker reported.

Spend estimates: final container window$0.320; call window$0.563 including possible scheduling; actual invoice unknown. Center diagnostic separate, estimated container$0.112/call$0.438. These are development/run estimates, not actual charges.

Site commit19f08d9 changes only workcell-photo-direct/. Live verification follows.


Live validation caught browser cache mixing new hashed JS with old scene-report.json (50records/185s) and same-name GLBs. Fixed the shared photo-report resolver: JSON fetch revalidates with cache:no-cache; photo/model/export URLs carry the report document hash, which includes asset content hashes. Regression checks assert revalidation headers and revision query values. Independent review confirmed no blocker. Build/typecheck and full browser regression rerun passed. UI-only publication correction is75b0b5c; original frozen inference UI remains report-ui, actual published UI snapshot report-ui-published. No inference rerun or new latency claim for this UI correction.


Live verified75b0b5c: Pages run36788836954 succeeded. Existing browser with cached prior data now loaded52/52 models and335s timing after ordinary navigation/reload. Three board source GLBs loaded; left145.0deg/right141.5deg cards matched annotation, center explicitly unsupported. Saved bend opens isolated rotatable model. Free scene shows saved-ground grid/worldXYZ and orbit changes actual canvas; divider dragged to75. Every requested GLB URL carries report revision. Console0errors. Actual live screenshots: browser-check/live-right-fold.png and browser-check/live-ground.png. Final URL: https://admin-wekruit.github.io/panoptes-workcell-report/workcell-photo-direct/?v=75b0b5c-verified . Center reliable bend remains unresolved.
