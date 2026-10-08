# Fence-fragment merging in the clearance rule

Date 2026-07-20. Change: `_assess_clearance` (ehs_spatial/rules.py) previously
required exactly one fence entity to pass the evidence gates and returned
INSUFFICIENT_EVIDENCE whenever reconstruction split the fence into multiple
gated clusters. It now merges all gate-passing fence fragments into a single
boundary via `unary_union(...).convex_hull`.

## Why this is a geometry-evidence improvement, not verdict tuning

- One physical fence routinely reconstructs as several clusters (thin rails,
  occlusion, single-view monocular depth). The merged hull is exactly the
  footprint clustering would have produced had the fragments connected —
  fragment count is a reconstruction artifact, not scene structure.
- Measuring against only one fragment is the false-PASS mode; the merged hull
  closes the gap-between-fragments case (regression-tested: a movable sitting
  in the gap FAILs at 0.0 m).
- Single-valid-fence behavior is byte-identical; every merge emits the
  warning "N safety fence segments merged into a single boundary hull".
- Facts cite the largest fragment's entity_id; evidence frames union across
  all merged fragments.

## Measured effect (single-photo demo, run demo-mono-ladder050-v4)

Single photo of the ladder_050 synthetic scene (GT clearance 0.5 m,
0.6 m rule → expected FAIL):

| | before merge | after merge |
|---|---|---|
| verdict | INSUFFICIENT_EVIDENCE | **FAIL (correct side)** |
| distance | — | 0.037 m |
| warnings | reduced capture | reduced capture + 2 segments merged + 13 fragments discarded |

**Residual error stated plainly: 0.037 m measured vs 0.5 m GT (~46 cm off).**
Monocular depth smears the thin fence toward the ladder, so the merged hull
overreaches. The merge fixes the *abstention* failure (two valid fragments,
no ruling); it does not and cannot fix single-view metric error — that is an
input-quality/reconstruction problem (V2 ladder: multi-view DSLR reaches
14.4 cm MAE). Single-photo verdicts remain flagged by the reduced-capture
warning and should be treated as screening, not compliance-grade measurement.

## Regression status

- Full suite 175 passed / 2 skipped.
- Offline oracle 4/4 with unchanged distances (0.548 / 0.756 / 0.0 /
  abstain); `fence_occluded` still abstains — the merge does not swallow the
  designed abstention case.
