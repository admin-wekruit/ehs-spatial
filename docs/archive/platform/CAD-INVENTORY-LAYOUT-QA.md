# CAD fit and unified object inventory

Verified 2026-09-13 against publication `f4e5ca43-543d-4624-8ea0-27aa2843e6e4`, revision `a97267c4-d435-43ee-915f-a59f2bb44ed6`.

## Cause and change

At a 1440 × 800 viewport, the previous 560 px desktop workspace left the CAD stage only 47.65625 px high. Its fixed 80 px vertical fit padding then reduced the drawing height to 32 px. The 68 projected records span approximately 6.495 × 5.108 native units; the small drawing was caused by layout and fit padding, not a need to discard scene geometry.

The desktop four-view workspace now has a 760 px minimum height. Fit padding scales with the actual canvas size, including narrow selection-focus views. No geometry, scene identity, coordinate frame, units, or publication data changed.

The complete object inventory now lives in the existing left rail: photograph references, association state, model/placement state, and observed dimensions share the same selection and measurement functions as the four views and right inspector. The duplicate lower table and its first-ten truncation are removed. Image interpretation, source records, and historical evidence remain below; their button returns to the full inventory and opens it on mobile.

## Runnable checks

- `node --experimental-strip-types checks/cad-view.mjs`: fit at normal and 47.65625 px heights, narrow focus, all geometry inside the fit, coordinate round trips, labels and pointer lifecycle.
- `node --experimental-strip-types checks/report-scene.mjs`: 68 records without truncation, missing-mesh selection, evidence and dimensions, native/metric distinction, mobile inventory navigation, shared selection.
- `node tests/report-context-check.mjs`: inventory navigation preserves object context; no duplicate table.
- `node tests/report-evidence-check.mjs`: retained source interpretation and evidence.
- `node --experimental-strip-types tests/entity-evidence-check.ts`: photo identity and missing geometry states remain distinct.
- `npm run build`: TypeScript and Vite production build.

## Computer-use verification

- Desktop 1440 × 800: actual CAD SVG viewBox is `0 0 436.5 164`, with `data-cad-shapes=68`. The full diagram fills the height with adaptive padding, rather than appearing as a tiny cluster.
- Selected record 06, the material cart, from the left inventory: CAD and plan show it selected; the right inspector and photo context show the same entity.
- Used CAD Focus selection, expanded the CAD pane, then returned to Full drawing and Four views: selection remains intact.
- Mobile 390 × 844: View all objects opens the same inventory. Searching `control button` returns records 30 and 68. Selecting record 68 switches to its source photo 3; CAD highlights `#68 control button` with its dimensions.
- Switched Chinese to English on mobile: selected entity, source photograph and CAD view remain unchanged.
- No browser error/warning logs were reported during these checks.

This change improves presentation and navigation only; it does not assert improved reconstruction accuracy or resolve pending associations/placements.
