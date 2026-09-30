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
