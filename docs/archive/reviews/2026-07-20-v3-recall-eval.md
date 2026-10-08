# V3 / E1 — per-class segmentation recall on real EHS imagery

Date 2026-07-20. Harness `scripts/recall_eval.py` (bbox + mask modes,
prompt-ensemble scoring, disk cache — re-scores free). Provider: fal SAM 3.1,
same request shape as production, `max_masks=32` (the fal cap). n=25 images per
class, seed 0. Reports in `outputs/recall_v1/report_*.json`.

## Results (best-prompt and ensemble)

| Class | Dataset (license) | Best single prompt | Ensemble | Notes |
|---|---|---|---|---|
| gas cylinder | CylinDeRS real (CC BY) | "gas cylinder" R@0.3 0.779 | **R@0.3 0.844 / R@0.5 0.844**, presence 0.92, fp/img 1.2 | strong; ensemble +6.5 pts recall at the cost of more false masks |
| fence | IITKGP chain-link real, pixel GT (Apache) | "chain link fence" cover 0.847 | **presence 1.00, coverage 0.897**, IoU 0.44 | real fence is NOT a recall problem for SAM3 |
| forklift | LOCO warehouse (CC0) | "forklift" R@0.3 0.471 | R@0.3 0.529, presence 0.52 | mid; far-field, partial occlusion |
| pallet | LOCO warehouse (CC0) | "pallet" R@0.3 0.326 | R@0.3 0.326, presence 1.00, fp/img 2.76 | weak — dense far-field stacks, ~30 GT/img vs 32-mask cap |

## Findings

1. **The synthetic-fence panic is fully retired.** On real chain-link imagery
   with pixel ground truth, SAM 3.1 hits presence 1.00 and 90% mask coverage.
   The earlier 0-detection result was a synthetic-domain artifact (flat yellow
   render), not a model blind spot. Fence recall is not the top model risk;
   dense small-object recall is.

2. **Prompt-ensemble is load-bearing, proven both directions.** "barrier"
   scores 0 on real fences but was the ONLY hit on the synthetic fence;
   "chain link fence" is the opposite. No single prompt wins both domains — the
   synonym fallback (`LABEL_PROMPTS` in providers/sam3.py) is a requirement, not
   an optimization. On cylinders the ensemble lifts recall 0.779 → 0.844.

3. **Ensembles trade recall for false positives** (cylinder fp/img 0.36–0.44 →
   1.2; pallet 1.1–1.6 → 2.76). For an asymmetric-failure product (recall @
   fixed FPR) this is the right trade at the perception layer — the rule
   engine's evidence gates and INSUFFICIENT_EVIDENCE path absorb spurious
   masks downstream. Quantified here so the tradeoff is a dial, not a surprise.

4. **The real weakness is dense far-field small objects** (warehouse pallet
   stacks: ~30 instances/image against a 32-mask API cap, so recall is capped
   near 0.33 by construction). This is the LLMDet-boxes → SAM-HQ-masks target
   for V4, and the reason the geometry+rule product should stay in
   near/mid-field EHS clearance scenes for MVP, not aisle-scale inventory.

## Method honesty / limits

- Predicted boxes are derived from mask bounding boxes; a mask covering two
  touching objects counts as one box (depresses crowded-scene recall — real,
  not a bug).
- LOCO GT includes tiny/background instances; no min-area filter applied, so
  these are conservative (worst-case) recall numbers.
- IITKGP fences fill the frame, so bbox scoring is meaningless there — mask
  coverage/IoU is the only valid metric and is what is reported.
- n=25/class is a decision-grade sample, not a publication-grade one; the
  harness caches, so scaling to n=100+ is cheap when a citable number is needed.
- CylinDeRS license shows a CC-BY vs NC-ND discrepancy (see dataset manifest) —
  resolve before any commercial-facing citation of its numbers.
