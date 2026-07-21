# Pipeline-mode V2 eval — real ground truth for the binding+clustering layer

Date 2026-07-20. `scripts/redwood_v2_eval.py --pipeline`: distances scored
through the PRODUCTION binding path (SAM masks → floor transform → SOR →
voxel → DBSCAN → hull-to-hull shapely distance) against the same laser
ground truth the box-based modes use. Eval-only prompt map (`EVAL_PROMPTS`)
is separate from the production vocabulary; SAM responses disk-cached under
`<pack>/sam_cache_v1` (re-scoring free); ~28 calls/pack behind `--live`.

Before this, SOR / fence-merge / DBSCAN had no real-data exam — the V2
modes measure from annotation boxes and bypass clustering entirely.

## Results

| Pack | box-based (camera_height) | pipeline (clustering path) | delta |
|---|---|---|---|
| Redwood VGA | 27.5 cm MAE | **25.4 cm MAE** | clustering path -2.1 cm |
| ETH3D DSLR | 14.4 cm MAE | **18.5 cm MAE** (n=2 of 3) | +4.1 cm |

Per-pair (pipeline mode):

| Pack | pair | GT m | predicted | err cm |
|---|---|---|---|---|
| Redwood | sofa-plant | 0.172 | 0.002 | 17.0 |
| Redwood | plant-wicker | 0.400 | 0.000 | 40.0 |
| Redwood | plant-table | 0.542 | 0.683 | 14.1 |
| Redwood | sofa-wicker | 0.707 | 0.402 | 30.5 |
| ETH3D | desk-sofa | 0.737 | 0.389 | 34.8 |
| ETH3D | sofa-chair | 1.349 | 1.372 | **2.3** |
| ETH3D | sofa-bin | 1.667 | — (SAM missed bin) | — |

Entity counts: Redwood plant split into 7 clusters, sofa 2, table 2,
wicker 1; ETH3D chair 5, desk 2, sofa 1.

## Reading

1. **The clustering layer costs little on top of localization**: within
   ~4 cm of the box-based upper bound on DSLR, and actually better on VGA
   (hull distance is more robust than the base-band percentile against
   VGA noise). SOR/DBSCAN/hull parameters now have a real regression
   number to guard.
2. **Failure modes are the known ones, now quantified on this path**:
   adjacent-object mask bleed collapses gaps to 0.0 (plant-wicker) — the
   conservative direction for a clearance rule (false-FAIL, not
   false-PASS); small-object recall misses produce an honest None
   (sofa-bin — same far-field weakness V3 located).
3. Multi-cluster splits (plant ×7) did not break scoring because matching
   picks the cluster nearest the GT centroid; production rules would see
   them as separate entities — fence-style merging generalized beyond
   fences is the obvious follow-up if this bites a real rule.

## Caveats

- n=7 pairs across 2 scenes: regression-grade, not benchmark-grade.
- GT-centroid matching mildly flatters the pipeline (production has no
  oracle centroid); it isolates geometry quality from association quality
  on purpose.
- ETH3D CC-BY-NC-SA: internal eval only.
