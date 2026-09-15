# Scene selection highlighting — 2026-09-15

## Reproduction and cause

In publication `e38008a7-41e2-451a-aa10-2154962617cf`, selecting floor marking
`4f584238-6ce2-5f12-97b2-4429f122b72e` highlights its photo, CAD and plan but
not its observed surface in the model-and-scene view. The selected photo's
representation is present, confirmed and in the current coordinate frame.

The visible pass used GPU upload order and `LESS`. Scene context and an object
can contain the same surface; context drawn first rejects the equally deep
selected fragments. Picking already accepted equal depth, so selection could
succeed while the visible color did not change.

## Shared repair and checks

The native viewer draws selected visible representations last with `LEQUAL`.
It preserves the depth buffer and nearer occlusion. No geometry, transform,
identity, measurement, layer visibility or bounds setting changes.

`web/checks/renderer-selection.mjs` executes the actual draw function against a
one-pixel depth model. It failed before the repair and now covers both asset
orders, source-camera/free views, nearer occlusion, current photo/frame guards,
no selection, and unchanged hidden-layer/picking behavior. The check also
verifies the reported publication's selected photo has a visible marking mesh.
Existing renderer, report-scene, Workspace selection checks and production
TypeScript/Vite build passed.

Computer Use locally confirmed the marking's teal surface in both model-and-scene
and observed-surface modes with bounds disabled. Source commit: `ed18c6e`.
Pages commit: `a8351de25e7ba27c5fb07349cf72c625129d27f3`.

Pages workflow `34990021402` succeeded. Computer Use then verified the public
page loaded `app-CRokSa0Q.js`: selecting the marking visibly turns its 3D surface
teal; clearing selection restores the yellow/black texture. Bounds remained
disabled. The page was returned to four views with no selection, and the temporary
desktop viewport override was reset. No new publication or model call was needed.

The background point cloud has no object-level point ownership map. A trial
hidden-triangle overlay was removed after browser verification failed to show
reliable point-level coloring. This release does not claim that background-only
point clouds now support per-object highlighting; existing picking is retained.
