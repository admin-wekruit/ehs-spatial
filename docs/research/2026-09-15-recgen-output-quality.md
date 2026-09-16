# RecGen output quality audit — 2026-09-15

Scope: independently inspect eight completed research outputs (#12, #14, #15, #16, #17, #26, #27, #28), their frozen inputs and shared post-transform stage. This audit used local CPU computation and read-only repository/billing queries; it made no scene edits, provider calls, or quality promotions. Later #23 work is outside this snapshot.

**Runtime checks pass for all eight. Shape/placement acceptance is separate.** Provider mesh vertices, faces and colors match the shared stage exactly. Manifest/input/output hashes match; all positions/colors are finite and indices valid. Native pose is exactly `anchor.cameraToWorld @ objectToCamera`. Maximum discrepancy against official posed GLB vertices is `4.7683655246544276e-7` native units, below the existing `2e-5` tolerance. Maximum native TRS round-trip error is `7.842550542314086e-8`. Source depth remains estimated and physically uncalibrated.

| Object | Vertices / triangles | Source-mask IoU | Quality finding |
|---|---:|---|---|
| #12 | 91,010 / 182,200 | 0.511 (original photo domain) | Original-photo-domain IoU 0.511 and rendered precision 0.890; source-mask coverage 0.545. Visible-region alignment and yellow-frame/top-hardware ownership need review. Completion outside the photo is not itself failure. |
| #14 | 146,148 / 292,682 | 0.682 | Recognizable open bars/crossmembers; edge and attachment spill needs assembled-scene review. |
| #15 | 139,822 / 279,644 | 0.553 / 0.356 | Recognizable signal column; second source is displaced left and mounting bracket facing/size differs. Cross-view placement needs repair. |
| #16 | 488,684 / 977,472 | 0.717 / 0.586 | Recognizable emergency-stop assembly; bracket width/tilt and label tab differ in second view. Nearly one million triangles for a small object. |
| #17 | 113,074 / 226,232 | 0.511 | Small recognizable beacon-like body, but only a 104×104 source crop; generated stalk/shell and color fidelity remain limited. |
| #26 | 119,552 / 239,120 | 0.731 | Recognizable red signal column; inferred mount/wires exceed selected silhouette. Keep placement unconfirmed. |
| #27 | 81,304 / 162,598 | 0.713 | Recognizable printed panel; 0.960 mask coverage. Generated border/frame, thickness and writing are inferred, not faithful original-photo texture. |
| #28 | 129,972 / 259,944 | 0.363 | Reject as delivery-quality fence: rendered opaque black panel replaces visible white open slats. High mask coverage hides wrong shape and overextent. |

The existing perspective-correct CPU raster uses every triangle, the frozen crop intrinsics and native camera, with at most 518×518 pixel-center samples. Fence segmentation masks contain openings/background; overlap is not a physical-bar completeness score. Occlusion and inferred hidden geometry also affect IoU. No new automatic acceptance threshold was invented.

**Correction for #12:** full padded-crop IoU 0.205 / precision 0.248 included 31,109 rendered pixels outside the original photo. Those figures do not prove wrong visible shape. Original-photo-only IoU 0.511 / precision 0.890 are the appropriate visible-domain figures; 54.5% filled-mask coverage still requires visual interpretation against the actual rails/openings.

Validation command: `.venv/bin/python -m pytest tests/test_recgen_research.py -q` — **25 passed in 0.43s**, including `test_pose_uses_raw_mesh_and_anchor_camera_once_with_nonunit_scale`. Independent real-output residuals above were also recalculated; they are not inferred from the test fixture.

Accounting snapshot at **2026-09-16 02:59 UTC**: eight successful calls plus one prior cancelled/outcome-unknown attempt retained **$9** reservations, with the root’s **$2** overhead allowance under the authorized **$20** total. No reservation was released by this audit.

Modal billing retrieved at **2026-09-16 03:04:57 UTC** reports **$0.43748322** for `lucida-private-assets` (`ap-K00o5rOaI896qdICTqwlv0`), interval **02:00–03:00 UTC**: A100 $0.32258968, CPU $0.04882717, memory $0.06606637. This app-hour contains the nine recorded attempts but does not allocate cost per call. Adding the previously attributed two HF diagnostics ($0.00001680) gives **$0.43750002** for these two known categories. Full task cost remains **unknown** because storage/other overhead and later work are not fully attributed. Whole-account reported $1.02002177 is context, not task cost.

Private, source-bearing evidence stays under `.platform/model-correspondence/model-completion/output-quality/`:

| Evidence | SHA-256 |
|---|---|
| `summary.json` | `eaa4ce88393b7b9dabc71d744a2510b1abb5572b374418f7256279abac771395` |
| `manual-review.json` | `bcf376cc64e53f3173ddf2249c44070d7296919932b3086565b3846d13d9241d` |
| `cost-reconciliation.json` | `5b4a21145ca9c95392af0677105bd18ff6830e15d44480acd7fe43eba76b8b59` |
| `billing-latest.json` | `a54e0868379321d505d90d68826894db324e95401072daa7b43e5115f71c9ac8` |
| `audit.py` | `f6c41d9a316c1800b59f101484621a2e93cd4d36046eca66f4c7cfa457b85960` |

`object-<index>.json` contains per-view measurements and exact output hashes; `object-<index>-view-<n>.png` shows source crop, generated native-pose mesh and overlap. `manual-review.json` findings are bound to each GLB hash.

Recompute local numeric evidence without inference:
```sh
PYTHONPATH=. .venv/bin/python .platform/model-correspondence/model-completion/output-quality/audit.py
```
