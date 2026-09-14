# Cross-photo identity browser verification

Date: 2026-09-13. Browser: actual Chrome extension tab, isolated from the parent agent's IAB tabs. All interactions below were read-only or temporary previews. No scene edit, feedback message, identity suggestion, or inference request was submitted.

## Intermediate snapshot inspected

- URL: `http://127.0.0.1:8792/app.html#/reports/e70ae9b4-ba4f-4586-950f-c000014e6285`
- Scene revision: `184109d2-e6f8-4b06-8c84-7b19002dcbe3`.
- Loaded script observed in DOM: `/assets/app-Cqm_wqi2.js`.
- The page showed 65 business object records, 84 photograph observations, 12 cross-photo groups and 53 records awaiting identity review. The three background/context entities were absent from the object list. These are this snapshot's counts, not final identity-scan acceptance thresholds.

## Actual interactions

1. Initial load had no selected object. Search retained access to a no-model control button.
2. Selecting `control button` selected entity `33e5c934-cbae-5053-9b71-8c7bf63a072c`. Switching photograph 1 to photograph 3 retained that entity and changed observation from `6edd754f-c0c2-560d-aa92-d601a824a286` to `7fce23b4-a071-5fdc-828c-0eaaea4c5b27` in the URL.
3. English/Chinese switching retained the entity, photograph, comparison pair, reason and temporary identity preview. Existing source labels and source-authored reasoning remained verbatim.
4. Expanding Scene 3D selected the single-view mode; Escape returned to Four views. The DOM retained one canvas. Native fullscreen requests were rejected in this background Chrome environment, including a CUA accessibility click. The app displayed its explicit unsupported-fullscreen message; successful native fullscreen entry/exit was not established.
5. Cross-photo review compared the control button with `safety sensor` (`c8de712d-db34-5c0a-a554-56d00cdce20b`) using two actual source photographs. Different-object preview was generated locally; Apply was disabled without ownership. No preview was applied.
6. The per-object feedback button opened the right feedback area. This local non-public build correctly required saving a copy for report correction. Anonymous public feedback remains a separate post-deployment check; no message was sent.
7. The original CAD attachment remained present with 33 source records and six explicit current-object links. Its provenance stated `user-bor1-02`, 1600 × 1240 source pixels, and explained that it was distinct from the current scene projection. Source CAD, GLB and Blender download controls were present; downloads were not re-executed in this browser audit.

## Defect found and corrected in reader code

The intermediate build displayed only 9/65 current CAD projections. All source cameras and ground evidence existed in the same coordinate frame. Observed surfaces retained a precise `{assetId, sourceRecordId}` source reference, also present on their owning observations, but the new photo filter only consumed direct `observationId`/`imageId` references.

The shared source-reference consumer now accepts the exact composite asset-and-record relation, restricted to the entity's current observations and the selected photograph. Existing frame and stale-source checks remain enforced. It never associates by display name, record ID alone or asset ID alone. Source-derived measurement observation memberships also require correction in the migration producer; empty memberships are not treated as all photographs by the reader.

Directly exercising the corrected shared helper against this unchanged snapshot restored model-layer projections to 27 for photograph 1 and 36 for photograph 3; photograph 2 remained nine until producer evidence restoration. These counts describe available source evidence per photograph, not a claim that all records have verified geometry.

Different/undecided previews now have decision-specific copy instead of showing model-after-merge fields. Timeline labels for reassociation and Blender export timeout are bilingual.

## Runnable verification after reader correction

Passed:

```sh
npx tsc --noEmit --project web/tsconfig.json
node --experimental-strip-types web/checks/identity-review.mjs
node --experimental-strip-types web/checks/renderer.mjs
node --experimental-strip-types web/checks/report-scene.mjs
node --experimental-strip-types web/checks/observed-plan.mjs
node --experimental-strip-types web/checks/report-interactions.mjs
node --experimental-strip-types web/tests/report-context-check.mjs
```

Checks cover exact composite source references, foreign-frame rejection, missing geometry, current observation ownership after splitting, multiple observations in one photograph, decision-specific previews, source-photo projection, current-model choice, original evidence preservation and frozen-reader routing. Final corrected publication and deployed public feedback verification are pending integration; this document does not mark those checks passed.

## Final local publication verification

- URL: `http://127.0.0.1:8792/app.html#/reports/8a61c1b8-65a0-498a-ac9a-7ed9924d6c9f`
- Scene revision: `b638e847-7b30-4443-b811-0b4349c07d5a`.
- Initial final build: `app-BaPTecDw.js`. Final reader after the additional fix below: `app-DIW26DFi.js`.
- Actual rendered list: 59 business records, 84 observations, 12 cross-photo groups, 47 awaiting identity review; no background/context rows. A fresh URL without an object parameter started with zero selected rows.

Final-snapshot interactions passed in Chrome:

1. Clicked the `control button` directly in the source-photo evidence group. Photo 1 and photo 3 retained entity `33e5c934-cbae-5053-9b71-8c7bf63a072c` and selected their respective observations.
2. The repaired measurement memberships restored source-derived ranges for no-mesh entities. Searching/selecting sensor `c8de712d-db34-5c0a-a554-56d00cdce20b` switched to its photo 1 and displayed a CAD projection of 0.1533 × 0.1607 native units. Its observed extent was 0.15 × 0.16 × 1.31, explicitly uncalibrated; model dimensions and model pose remained unknown.
3. Switching to Point cloud visibly rendered the real colored workcell cloud, retained the selected object's observed range, and labelled its axes as scene-native rather than structural. Only one canvas remained mounted. Photo 3's cloud/observed CAD had 31 records; photo 1 had 21.
4. At a 390 × 844 viewport, document scroll width equalled 390. Objects / Views / Details tabs retained the same selected sensor, search and source evidence. The viewport override was reset afterward.
5. The final report still exposed the original 33-record CAD attachment, six explicit links, source provenance and Blender/GLB downloads. Its export was honestly marked partially complete with artifacts and recorded duration 11.51 seconds; the completed reassociation task used its Chinese label.

### Additional real-browser defect fixed before release

CAD initially kept photo 1's 30-record projection after changing to photo 3, while the linked interactive plan correctly changed to 40. `CadView` memoized shapes by frame/layer but omitted `imageId`; both photographs share a frame. The shared CAD memo now depends on `imageId`, and both CAD and interactive-plan overlap pickers clear stale candidates when it changes.

The focused check executes the actual memo initializer with two source photographs in the same frame. The previous initializer fails; the corrected version chooses the second photograph's exact representation. `cad-view.mjs`, TypeScript and `npm run build` passed.

After reloading the actual `app-DIW26DFi.js` build, repeated photo changes updated CAD immediately:

| Selected photograph | Model/context CAD records | Control-button projected width × depth |
|---|---:|---:|
| 1 | 30 / 59 | 0.1341 × 0.1111 native units |
| 3 | 40 / 59 | 0.1849 × 0.1612 native units |

Across all three photo scopes, the corrected shared helper finds projections for 58 of 59 business records. The remaining record is the main observed floor reference, not an equipment-volume placeholder. Photo 2 has nine current model projections and no photo-scoped observed-object projection; unavailable evidence is not synthesized to inflate completeness.

The final local reader's checked flows passed. Native fullscreen remains limited by the browser environment described above. No feedback was sent during the local audit.

## Public release verification

The production frontend was built from source commit `5c0d549` and published through Pages commit `fab1163393d412e03dfbcf57ed354cbf12ff9201`. The browser loaded `assets/app-DDsk6t1y.js`; the public API returned the exact final revision IDs. GitHub Pages reported the commit built successfully.

- Main report: https://admin-wekruit.github.io/panoptes-workcell-report/app.html#/reports/8a61c1b8-65a0-498a-ac9a-7ed9924d6c9f
- Component report: https://admin-wekruit.github.io/panoptes-workcell-report/app.html#/reports/02592388-82fd-494b-a478-4a833613eb98

Real computer use in the in-app browser verified the public main report:

1. The report rendered 59 business records and 84 observations with one WebGL canvas and no initial object selection. Actual source photographs and the workcell meshes were visible.
2. Selecting the control button and switching to photograph 3 retained entity `33e5c934-cbae-5053-9b71-8c7bf63a072c`, changed its observation, and updated CAD from 30 to 40 photo-scoped records. The projected width/depth changed to 0.1849 × 0.1612 native units.
3. Object feedback opened the right Agent area with the current entity and revision. Cross-photo review displayed both records' actual photograph observations. A different-object preview explicitly stated that the public report accepts an identity suggestion without mutating the published scene. The final sharing action was not executed; no synthetic feedback was published and no model call was made.
4. Reloading the old `f4e5ca43-543d-4624-8ea0-27aa2843e6e4` report redirected to the same website's `readers/v1/app.html`. The frozen reader still rendered its original revision `a97267c4`, 68 records and 84 observations, original CAD and downloads.
5. The final main report was reset to its default route, with no selected object, no QA preview text, and one canvas. Temporary viewport overrides were reset and the new public report was retained as the user-facing tab.

Public feedback submission is covered by API tests rather than a fabricated live reviewer submission. Browser verification does not establish the correctness of all physical identities, uncalibrated dimensions or historical EHS findings.
