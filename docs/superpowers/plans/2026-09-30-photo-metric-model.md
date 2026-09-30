# Photo metric model Implementation Plan

> **For agentic workers:** Use superpowers:subagent-driven-development. User approved implementation on 2026-09-30; proceed through integration, review and publication.

**Goal:** One raw-photo run produces actual models, a tunable button scale, fitted fence/floor geometry and a clickable measurement report.

**Architecture:** Preserve the existing two-A100 run. Add one geometry worker using full-resolution image features and existing multiview geometry. Scene and camera coordinates share one scale. Browser changes the scale and exports the same geometry; it never presents supplied dimensions as validated ground truth.

**Tech Stack:** Existing NumPy, SciPy, OpenCV, trimesh, Three.js, Modal.

- [x] Geometry: add scripts/workcell_photo_geometry.py with build(root, sources, diameter_m, height_m). Read saved frames and raw photos, fit local floor, extract supported fence framework, recover repeatable anchor geometry across views. Export fitted fence/floor GLBs plus geometry.json with native floor, anchor estimates, scale residuals and clearance lines. No manually selected pixels. Keep unknown axes unmeasured. Include --self-check covering perspective, transforms and known line-plane distance.
- [x] Integrate: call geometry worker in modal_apps/workcell_photo_all.py alongside RecGen after geometry and masks arrive. Pass dimensions from scripts/workcell_photo_oneshot.py. Package all meshes and evidence. Use finite positive input validation and explicit failure state. Regenerate page/data automatically.
- [x] Report: update workcell-photo-direct/index.html only. Default to orbit view showing actual meshes and structural model; editable component height/width, common scene scale, click selection, fence clearance line, photo overlay, GLB scene download. All DOM values and export transforms derive from the same scale. Preserve evidence and label unverified scale. Reject invalid inputs visibly.
- [x] Check: self-check geometry, synthetic scale invariance, run all four raw photos from scratch on ephemeral 2x A100, record latency/spend. Inspect exported geometry and browser interactions at default and altered scale; photo projections must not move when units change.
- [x] Review: independent spec review then code quality review. Resolve material findings, publish only workcell-photo-direct/ at existing Pages URL, inspect live model, capture screenshot and return concise Chinese results.

The prior design/research is /Users/adam/Desktop/panoptes-public/research-notes/workcell-photo-real2sim-plan-2026-09-30.md. No new external API, VLM measurement, model download, local GPU work, manual scene placement or unrelated refactor.

## Execution evidence — 2026-09-30

- Output: `/Users/adam/Desktop/panoptes-public/research-notes/workcell-photo-metric-2026-09-30-b`. Four original workcell JPEGs, every GPU stage rerun in one ephemeral `A100-80GB:2` allocation, app `ap-gkzD2NdKzP3ZKbWFyMF44e`.
- Full oneshot wall time185.26s, including startup and report packaging. Geometry40.14s and segmentation41.15s overlap; complex object modeling65.50s. Metric CPU work overlaps generation.
- Primary outputs: robot pose meshes, cart mesh, fence structural mesh, posts, floor, and combined `workcell-metric.glb`. Actual mesh viewer with photo overlays, object picking, editable reference size and current-size export.
- Conditional reference: whole red/yellow/gray component height0.2m, width0.2m; width/height residual25.3%, no absolute-accuracy claim. Photos3/4 support the lower rail. Model clearance0.334775m at20cm reference and0.167387m at10cm; exported GLBs verified against these values and floorY=0.
- RunA195.0s preserved with missing clearance. Root fix: match fragmented parallel edges using their common overlap relative to the shorter supported span; preserve clipping through final mesh export. No target clearance or source pixel ROI hardcoded. CloudB(OpenCV4.11.0/NumPy1.26.4) passes nonempty multiview and mesh consistency checks.
- Independent specification and code reviews passed, followed by adversarial review of the overlap fix. Local browser: model loading, scale updates, export/reload, source overlays, picking, mobile layout and error checks.
- `spend-ledger.json` records estimates, not invoices: B function window$0.243/call window$0.307; A$0.261/$0.317. Actual billed amounts unknown.
- Website commit `a7f8369` changes only `workcell-photo-direct/`; Pages deployment built; live actual mesh loading,20→10cm size change,33.5→16.7cm clearance,side view and zero console errors verified.
