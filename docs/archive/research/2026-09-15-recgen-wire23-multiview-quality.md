# RecGen #23 two-view wire audit — 2026-09-15

**Runtime succeeded; shape and placement rejected.** Job `6195457d-c23e-5e80-8d2b-2922512d2557` used the reviewed photo1 observation plus the existing photo3 observation, rectangular original-pixel crops, and explicit `maskErosionEnabled=false`. The result remains a broad railing/grid panel with a blue top beam and an unsupported mixed-color mass. Its sparse transverse connections do not reproduce the slender vertical wire tray's repeated short rungs. The unchanged native pose also projects it too widely and into the upper context in both photographs. Root independently reviewed the topology image and confirmed rejection.

The frozen baseline is `e16e9851-5045-4984-8b3e-c8eac49dac69`. Observation IDs are `70b778ef-4841-5c81-a525-d35181e9e24d` and `8e23c9d9-9d76-5dca-bb6d-aa31dabcbaaf`, both revision 1. Both frozen RGB crops exactly equal the original photo pixels. Photo1 labels 22,452 reviewed visible-metal pixels; photo3 retains 7,201 partial visible-core pixels. These are not complete-object ground-truth masks.

| Independent CPU check | Result |
|---|---|
| Mesh | 39,236 vertices / 78,440 triangles; finite vertices/colors; zero degenerate triangles; provider geometry and colors preserved exactly |
| Native pose | Persisted proposed pose equals anchor camera-to-world × official object-to-camera exactly |
| Posed-GLB maximum discrepancy | `1.1919589404385533e-7`, below existing `2e-5` tolerance |
| Native TRS round-trip discrepancy | `1.1549973422120274e-7` |
| Native extents | `[1.92860, 2.03718, 1.70612]`, uncalibrated estimated units |
| Photo1 original-pixel coverage | 5,800 / 22,452 (**25.83%**); 348,882 model pixels in the compared real-photo domain |
| Photo3 original-pixel coverage | 891 / 7,201 (**12.37%**); 113,667 model pixels in the compared real-photo domain |
| Projected bounds, photo1 | x `2354.00–3360.12`, y `-603.82–2173.71` |
| Projected bounds, photo3 | x `2667.52–3817.68`, y `-626.91–2301.72` |

The original-resolution comparisons use every triangle, unchanged cameras and unchanged provider pose. They include every positive mask pixel and retain the unmasked photo context. The masks are partial, so low IoU or precision alone is not a rejection criterion. The evidence for rejection is the visibly wrong wire topology and misalignment in both photographs. No pose fitting, geometry rescaling, mask expansion or altered colors were used. Passing pose arithmetic does not establish correct physical placement.

The packed payload, frozen protocol, actual NPZ, actual response and persisted provenance all agree on erosion **false**. The response attests **`recgen-input-v2`**, two views, and the reviewed model/code/weight pins. Frozen runtime-audit SHA `4fcd2004398aa7cc068323fe9579f7e55f8e72e181c417894dfd8398a4e18053` matches the v2 evidence file; the reviewed wrapper and deployment receipt also match. Provider function time is 63.555079419 seconds; **actual per-call cost remains unknown**. The audit made no GPU calls, database writes or reservation changes.

All evidence is isolated in `.platform/model-correspondence/model-completion/output-quality/followup-wire-multiview/`; earlier audits remain unchanged. Raw GLB SHA-256: `cc27915720dd32f09eca07db5fba62ae38b7a9c3b703c39e48d9c976b0a1e8fa`.

| Evidence | SHA-256 |
|---|---|
| `summary.json` | `e8cd04e0dbd9a01655f4913ac1d5b9d91ba2e9c83ea636f6b431c07f2d1e927c` |
| `full-source-comparison.json` | `f3f7514b4afa27180e6874081eba4672dd6a388256665c6cc1972e9f0c554132` |
| `runtime-input-check.json` | `d74dd4f66601ce5f6f295caf8402c917eda627ef7e5f2b86478e286c989df999` |
| `manual-review.json` | `c83fed4aa5222518bd6b4b7d10c27247c61d40255e3f4338ec2ecd6040531b34` |

Visual evidence: `object-topology.png`, `object-23-view-0.png`, `object-23-view-1.png`, and each `view-0-original-strip-*` / `view-1-original-strip-*` image. Preserve the rejected result and original observations in history; do not activate this mesh or repeat the same paid input.
