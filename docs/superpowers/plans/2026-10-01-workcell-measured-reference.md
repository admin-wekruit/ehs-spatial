# Workcell measured reference implementation plan

> **For agentic workers:** Use superpowers:subagent-driven-development. Keep the existing isolated worktree and branch; do not alter previous run directories.

**Goal:** Apply the user's measured emergency-stop dimensions to the existing pipeline, publish independent fence/light-curtain comparisons and the full clickable model report.

**Architecture:** Keep camera, depth, fitted floor, guard structure and native observed geometry frozen. Apply one uniform scale from the 10 cm whole-component height. Record 8.5 cm main-body diameter and 4 cm red actuator diameter as distinct physical features; the old whole-envelope width is not either diameter. Store the 20 cm fence and 24 cm light-curtain distances only as evaluation targets, never fitting inputs. Preserve observations and residuals rather than deforming the scene to match targets.

**Tech Stack:** Existing Python/NumPy/trimesh report functions, React/Three viewer, static GitHub Pages.

## Tasks

- [x] Backend: Add an explicit JSON measured-reference input to both existing report packaging and oneshot entries, with a small reusable calibration/evaluation function and one runnable check. Reject invalid dimensions before any cloud job. Ensure measured-reference metadata, default scale, report JSON and metric GLB agree. Use the same button-construction helper for the known reference model; measured diameters apply only to their corresponding red/main cylindrical components, and internal height partitions remain rendering assumptions. Never relabel observed envelope width as the main-body diameter. Preserve arbitrary scene object geometry.
- [x] Evaluation: Preserve all per-photo lower-edge distances for fence and the two existing light-curtain housings. Report the same multiview median rule for both housings, plus range and active-photo estimate. Use the recognized fence rail feature where available. Store estimates, truth, signed/absolute/relative error, source features and visibility limits. Verify changing evaluation targets cannot change scale, geometry or estimated measurements. Add summaries for the other existing multiview objects; keep one-view full dimensions unknown.
- [x] Viewer: Reuse current scale/card/measurement controls. Display the three distinct reference dimensions and provenance, explicit primary height scale and observed-envelope mismatch. Show independent estimate-vs-measurement comparisons, source photos and adjustable-scale behavior. Retain full model selection/orbit/GLB export. Add browser verification of actual 10 cm defaults and reference mesh dimensions.
- [x] Integration: Build a new artifact from saved data, run source/schema/metric/browser checks, spec review then code-quality review. Log this as cached recalibration with zero GPU inference; do not call it a newly timed oneshot run.
- [x] Publication: Update the existing public source branch and only workcell-photo-direct/ in the Pages repository. Correct stale private-repo wording in the handoff. Publish measured input, computed results and reproducible commands; verify the live page and remote source commit.

Published source: `e588917bfa927627b447aeb942fd80d78b5600f6`, tag `workcell-measured-reference-2026-10-01`. Pages: `694c350a3671e94062711866a279ee3193f197db`; deployment run `36902294901` succeeded. Anonymous release asset downloads matched all local SHA-256 hashes. Live browser verified 52 models, the 10 cm baseline, comparison-row selection, 0.171 m fence line, actual 3D orbit, and doubling the scale without changing independent truth; report JSON and metric GLB return HTTP 200. Final cached packaging took 8.457 s, zero new GPU inference and $0 GPU spend.

## Measurement scope

User wording: red actuator diameter 4 cm; main circular part max diameter 8.5 cm; height 10 cm. A clarification about height including the gray base is pending. Implement separately named fields and disclose that scope until confirmed. Do not allow the 20/24 cm check measurements to determine the calibration choice.

## Verification

Use the existing Python environment at /Users/adam/Desktop/Tesla/panoptes-platform/.venv/bin/python. Focused checks: existing ground-distance, metric export, photo report (52 objects), and browser interaction checks, plus new calibration test. The test must include invalid values, no target leakage, preservation of native geometry, and readback of the reference GLB dimensions. Cached rebuilding and simple local checks are allowed; no heavy inference or global geometry fitting on the Mac.
