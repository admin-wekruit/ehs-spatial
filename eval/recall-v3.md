# V3 / E1 — per-class recall of fal SAM 3.1 on real EHS imagery

Date: 2026-07-20. Runner: `scripts/recall_eval.py` (bbox + mask modes, per-class
prompt ensemble, responses disk-cached so re-scoring is free). n = 25 images per
class, seed 0, `max_masks=32` (fal cap), inputs resized to ≤1024 px. Total spend
for the full sweep ≈ $2.7 including smokes.

## Results (n=25/class)

| Class (dataset, license) | Metric | Best single prompt | Ensemble |
|---|---|---|---|
| fence (IITKGP, Apache-2.0, real chain-link, pixel GT) | presence / mask coverage / IoU | "chain link fence": 0.96 / 0.847 / 0.449 | **1.00 / 0.897 / 0.436** |
| gas_cylinder (CylinDeRS, CC BY 4.0) | recall@IoU0.3 / @0.5 / fp-per-img | "gas cylinder": 0.779 / 0.766 / 0.40 | **0.844 / 0.844** / 1.20 |
| pallet (LOCO, CC0) | recall@IoU0.3 / @0.5 / fp-per-img | "pallet": 0.326 / 0.225 / 1.64 | 0.326 / 0.229 / 2.76 |
| forklift (LOCO, CC0) | recall@IoU0.3 / @0.5 / fp-per-img | "forklift": 0.471 / 0.382 / 0.40 | **0.529 / 0.412** / 0.76 |

Raw reports: `outputs/recall_v1/report_{fence,cylinders,loco}.json`; response
cache: `outputs/recall_v1/cache-v2/`.

## Findings

1. **Real-fence recall is solved at presence level**: 100% presence, 90% pixel
   coverage on real chain-link imagery. The synthetic yellow-fence zero-recall
   was a domain artifact, not a model blindspot.
2. **Prompt ensembles are load-bearing in both directions**: "barrier" is the
   only prompt that hits the synthetic fence and scores 0.00 on real fences;
   "fence"/"chain link fence" are the reverse. No single prompt wins both
   domains. Ensemble lift elsewhere: cylinder 0.779→0.844, forklift
   0.471→0.529.
3. **Weak spot located precisely: dense far-field small objects** (LOCO
   warehouse racking, ~30 GT pallets/image). Not the near-field compliance-
   check regime, but the number to beat if the product enters aisle/racking
   scenes — and the target for the V4 detector swap (LLMDet-class).
4. Ensemble unions raise false positives roughly additively (cylinder 0.4→1.2
   fp/img) — verdict-side gates (≥2 evidence frames, area/height minima)
   absorb some; per-rule FPR reporting stays mandatory (first-principles E1).

## Caveats

- IITKGP fences are see-through foreground chain-link (not factory guard
  fencing); treat as chain-link proxy until owner-site capt