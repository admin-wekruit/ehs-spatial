# RecGen #12 visible-body audit — 2026-09-15

**Runtime succeeded; object quality rejected.** Job `fed77685-15fe-53e2-b8a2-19449940ebd6` used the accepted observation revision 3 on scene revision `e6701cde-003f-436d-8323-4b32edc7e0ab`. The conspicuous separate yellow light-curtain post from the earlier output is no longer recognizable. However, the new mesh introduces a red hood-like component above the fence and turns most of its see-through regions into opaque gray panels. The source shows equipment and vertical members through those regions. Both the independent review and the root's topology review reject this result as the current #12 model.

The source body mask contains **238,269 original pixels**, with **214,061** positive estimated-depth samples. It includes transmitted background appearance; those pixels do not establish ownership of the background machinery. The right photo boundary truncates the fence, and the foreground tray obscures some upper-left boundaries. Neither right-side completion outside the photograph nor geometry behind known occlusion is, by itself, a rejection reason. The failure is the visibly wrong component and material geometry within the observed context.

| Independent CPU check | Result |
|---|---|
| Mesh | 160,794 vertices / 322,092 triangles; finite vertices/colors; zero degenerate triangles; provider geometry and colors preserved exactly |
| Native pose | Persisted proposed pose equals anchor camera-to-world × official object-to-camera exactly |
| Posed-GLB maximum discrepancy | `2.384015500567216e-7`, below the existing `2e-5` tolerance |
| Native TRS round-trip discrepancy | `6.376496264337561e-8` |
| Original-pixel projection | All triangles rendered without pose fitting; source domain x `2416–2880`, y `550–2553` includes the mask and all projected geometry inside the photograph |
| Body-mask coverage | 221,277 / 238,269 pixels (**92.87%**); 454,393 rendered pixels in that real-photo region |
| Material | Raw GLB has no explicit materials, so its default material is opaque; audit did not change transparency |

Coverage measures a broad visible-body extent, **not complete-object or material fidelity**. At the 518-pixel audit grid, median absolute relative depth disagreement is 11.28%; source depth is estimated and may sample seen-through equipment. This is not a calibrated physical-placement test. Pose arithmetic passing does not establish physical placement.

The frozen request retains historical runtime-audit SHA `8ea54730a69cba96dc2556a64fad45218256be7b4b8f0055b63f6eefdbd0f14d` and contains no explicit erosion flag. The actual returned response attests **`recgen-input-v2` and `mask_erosion_enabled=true`**, matching the official default. Separately verified v2 deployment evidence records wrapper SHA `ca2e4fa7464bfe340651c4715beee6beda238ca4fcbd8c7d5b2e50cc20213595` and deployment-log SHA `2b4ebab8d7783c5f16c4896aa84fa9cf00c3ecae7464d0e0e9dd5eb403f5fad1`. The frozen manifest was not rewritten or presented as evidence that the old wrapper remained deployed. No metadata-only rerun occurred.

The provider function reports 52.68892822 seconds. **Actual per-call cost remains unknown**; runtime seconds are not a bill. This audit made no GPU calls, database writes or reservation changes.

All evidence is isolated in `.platform/model-correspondence/model-completion/output-quality/followup-panel12-body/`; earlier batch audits remain unchanged. Raw GLB SHA-256: `433c6291ff9fb8e5720e8811060d6ed1d927fe2dbf6aa19080e9e0211352742d`.

| Evidence | SHA-256 |
|---|---|
| `summary.json` | `341c38946df99099a27b058f159457dcb29fe6e2bb4ef1e330d809cb5258ce67` |
| `full-source-comparison.json` | `1c678bf65590d0eddb119d0566c874cf606fb6e935d93d8d0134584b7245ade2` |
| `runtime-execution-check.json` | `a0f0352718e641755372df5e51d7231923eebd73fb67342537647941bd1db835` |
| `manual-review.json` | `3c4978eb5d0875d4f7e568d7e98d76678d4e64cb7658d51402d11db3f2d20500` |

Visual evidence: `object-topology.png`, `object-12-view-0.png`, and `original-strip-0.png` through `original-strip-2.png`. Preserve this rejected output and source evidence in history; do not activate it as a successful corresponding model.
