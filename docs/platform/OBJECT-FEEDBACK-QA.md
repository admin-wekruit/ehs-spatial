# Object feedback and optional outlines — 2026-09-13

## Behavior

The shared photo view starts with contours hidden but retains object hit-testing. Report and model-workbench controls expose object outlines; a report without an object query starts unselected, and Clear selection clears object/observation context without replacing the scene. Explicit object deep links remain supported.

Every inventory row opens the existing right pane for that object’s feedback. The modeling workbench uses the existing project Agent. Its history and pending replies are isolated by project, branch, revision, and entity; policy conversations retain their policy scope. Public feedback uses the same Deep Chat component with independent visitor credentials and immutable publication/entity scope. Objects without observations do not attach an unrelated current photo.

Public feedback is persisted separately from the published scene. It cannot apply edits. A text-only Gemini adapter is available but deployment currently has model and budget disabled pending an explicit budget; saved feedback says that no Agent reply is enabled. No real model call was made for this change. A successful persistence test is not a model-answer validation.

## Executable checks

- `.venv/bin/python -m pytest tests/test_publication_feedback.py -q` — 16 passed; authorization, CORS, immutable routes, scope, idempotency, disabled budgets, reservation races, unknown calls, cancellation and persistence.
- `node --experimental-strip-types web/checks/agent-panel.mjs` — actual handlers: A→B delayed replies, history paging, mixed old conversations, fixed revisions, same-request retry, public scope and zero-observation objects.
- `node --experimental-strip-types web/checks/report-scene.mjs` — default outline state, linked 68-row feedback selection, mobile pane change, retained CAD/3D geometry and camera behavior.
- `node web/tests/photo-draw-check.mjs` — invisible outlines still allow normal selection and drawing.
- `node --experimental-strip-types web/tests/report-context-check.mjs` — feedback/clear context, preserved publication isolation and original downloads.
- TypeScript, production build and `git diff --check` passed.

## Browser checks

Computer Use exercised the real publication f4e5ca43-543d-4624-8ea0-27aa2843e6e4 against a local public-mode API at 8793 and the full local app at 8792. Desktop 1440×900 and mobile 390×844 were checked. A plain report had no selected object and no photo/3D outline. The outline toggle restored contours and projected bounds; Clear selection removed the object query. A cart feedback message was saved; selecting the right bollard showed a separate empty conversation, and returning to the cart restored its saved feedback. Reload retained it. Mobile feedback switched directly to the detail pane.

The first mobile pass found a clipped input. The corrected flex layout sizes the actual Deep Chat shadow container to the host rather than inheriting its CSS height. The final measured input bottom was 617 px inside the 646 px panel bottom at 390×844.

## Runtime boundary

`modal_apps/publication_site.py` adds a dedicated feedback volume and one writing container. Feedback requests serialize volume reload/write/commit; immutable report metadata and asset streams bypass this lock. This is a first-release throughput ceiling, not a scalable multi-container database. A model call is reserved and checkpointed before issue, has SDK retries disabled, and an unknown outcome never automatically repeats. The deployment budget is conservative reserved spend, not a live provider invoice.
