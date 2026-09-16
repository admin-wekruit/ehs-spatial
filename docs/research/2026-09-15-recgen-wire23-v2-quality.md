# RecGen #23 explicit no-erosion audit — 2026-09-15

**Runtime succeeded; shape and placement rejected.** Job `3e1eb172-674a-578f-a1f1-baf3e34792bc` returned a broad guard/fence panel with foot plates, rather than the narrow cable wire tray beside the entrance post. The new result corrects the previous near-zero extent collapse but does not establish the target's shape. Keep this result as research evidence; do not activate it as the current wire-tray model.

The exact frozen RGB, depth, mask, intrinsics and camera transform are unchanged from the first #23 attempt. The mask still labels only **7,201 visible wire-core pixels**, of which **4,712** have positive depth. It is not a complete-object ground-truth mask. The reviewed setting is now explicitly `maskErosionEnabled=false`; its frozen runtime audit, deployed wrapper and deployment-log hashes all match. The actual provider response attests **`recgen-input-v2` and `mask_erosion_enabled=false`**, and the persisted stage provenance agrees.

| Independent CPU check | Result |
|---|---|
| Mesh | 114,860 vertices / 230,776 triangles; 11 connected components; finite vertices/colors; original provider mesh and colors preserved exactly |
| Native pose | Anchor camera-to-world × official object-to-camera exactly matches persisted proposed pose |
| Posed-GLB maximum discrepancy | `1.1920900266915169e-7`, below existing `2e-5` tolerance |
| Native TRS round-trip discrepancy | `1.7148338282702014e-7` |
| Native extents | `[2.78860, 3.69964, 2.65131]`, uncalibrated estimated units |
| Original-photo projected bounds | x `2008.50–8905.88`, y `-12529.95–3829.11`; source cores occupy x `2711–2864`, y `521–3230` |
| Exact original-pixel strip | All 7,201 labeled core pixels included; 2,957 covered (**41.06%**); model renders 237,537 strip pixels |
| Shape evidence | Large generated lattice panel and support feet; unsupported width and severe projection mismatch against the narrow wire tray |

At the 518-pixel audit grid, original-photo-domain core-mask IoU is `0.002444` and precision `0.002452`. These compare against partial wire cores, so they are **not full-object completeness scores**. Rejection rests on the visible wrong object form and placement. Original-resolution strip panels retain the unmasked source context. No pose fitting, geometry rescaling or mask expansion was used in this audit.

At **2026-09-16 03:50:56 UTC**, read-only accounting shows **12 succeeded + 1 outcome_unknown = $13 reserved**, plus the root's $2 overhead allowance within the authorized $20. The new provider function took 44.596 seconds; this is not an actual dollar charge. The latest available app-hour bill remains $0.43748322 for 02:00–03:00 UTC and excludes this attempt. Per-call and full-task actual costs remain unknown; no reservation was released.

All new evidence is isolated in `.platform/model-correspondence/model-completion/output-quality/followup-wire-v2/`; earlier batch audits remain unchanged.

| Evidence | SHA-256 |
|---|---|
| `summary.json` | `7ec5d4a6e670b23d044f3d460a96a9c2fd099ead83e30f0ec093cd9f6e8cc6a8` |
| `full-source-comparison.json` | `4dcf238c5e86292ebe5bfcceca28029055d273c68fbba61920463f7cfb193e1d` |
| `topology.json` | `79c89cd83569e3563fbc274395e9589c544291420d4b606a0d1f07912451c89a` |
| `deployment-input-check.json` | `8c7d49ece2ea8ee9d03e88a8033a9d4c5b92a0f17e9d9135242444745ef8dd81` |
| `cost-snapshot.json` | `0faa6e4e2baa6ba90ac5bf978fb8460f1df25e655fe895d8c9f965e63759066e` |
| `manual-review.json` | `e1eecf1031d4ad4f599f4319e3ddbdcf47c684847c253c0b39a95e87ea1f80ba` |

Visual evidence: `object-23-view-0.png`, `object-topology.png`, and `original-strip-0.png` through `original-strip-3.png`. The audited raw GLB SHA-256 is `1486113c273a302a48e81c75af544493c86f6e0d33b96e94f200e76016ed89f6`. Audit execution made no GPU calls and no database writes.
