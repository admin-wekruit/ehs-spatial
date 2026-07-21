# Statistical outlier removal before entity clustering

Date 2026-07-20. Change: `_remove_outliers` in ehs_spatial/geometry.py —
Open3D `remove_statistical_outlier` applied to each observation's mask
point group (after the above-floor filter, before voxel downsampling and
DBSCAN). Constants `_OUTLIER_NB_NEIGHBORS = 20`, `_OUTLIER_STD_RATIO = 2.0`;
groups smaller than nb_neighbors are passed through untouched.

## Why

Monocular/multi-view depth smears thin structures (fence rails) toward the
background. Those smear tails inflate cluster hulls toward neighboring
objects — the direct cause of the single-photo demo measuring 0.037 m
against a 0.500 m ground truth.

## Measured effect

| Case | metric | before | after (nb=20/std=2.0) |
|---|---|---|---|
| single-photo ladder_050 (GT 0.500 m) | ruled distance | 0.037 m (err 46 cm) | **0.197 m (err 30 cm)**, verdict FAIL unchanged (correct side) |
| real factory photo | fence clusters / valid | 8 / 1 | 6 / 1 (noise clusters gone) |
| real factory photo | fence area / H / d_cam | 3.61 m² / 2.43 / 4.74 | 3.62 / 2.42 / 4.74 (real structure untouched) |
| offline oracle | 4/4, distance gate ±0.1 | 0.548 / 0.756 / 0.0 / abstain | **0.550 / 0.756 / 0.0 / abstain** |
| test suite | | 176 passed / 2 skipped | 176 passed / 2 skipped |
| Redwood V2 (camera_height) | MAE | 27.5 cm | 27.5 cm |
| ETH3D V2 (camera_height) | MAE | 14.4 cm | 14.4 cm |

## Honesty notes

- The V2 metric evals compute pair gaps from annotation boxes directly and
  do not pass through entity clustering, so their invariance is by
  construction, not evidence of SOR quality on that path. The real
  regression coverage here is the oracle pack + unit suite + the two demo
  runs above.
- Offline sweep on the single-photo case showed harsher settings recover
  more of the smear (nb=50/std=0.8 → 0.340 m, err 16 cm) but keep only
  ~17 % of the fence points. The shipped setting is deliberately
  conservative; revisit the constant only with a broader eval, not one case.
- Residual single-photo error stays large (30 cm at 0.5 m GT). SOR shrinks
  the smear tail; it cannot fix single-view depth. Multi-view capture
  remains the measurement-grade path (14.4 cm MAE on DSLR input).
