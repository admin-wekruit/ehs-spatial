# MoGe-2 closes the scale-anchor gap — with no operator input

Date 2026-08-24. Runner `scripts/moge_anchor_eval.py`. Benchmark: NVIDIA
PhysicalAI-Spatial-Intelligence-Warehouse val, the pure-measurement subset
(105 distance questions, benchmark-provided GT masks, 1–11 m, CC BY-4.0).

## The gap this attacks

The boundary map (2026-07-21) named one real weakness: MapAnything's
unanchored mono scale is off by a tight constant — gt/pred 2.50, MAD 0.24 —
so shape is right and *metres* are wrong. A single global correction took
held-out median relative error from 60% to 8.5%. That correction came from
the operator typing a camera height. The open question was whether a second
metric model could supply it instead.

## Method

MoGe-2 (MIT, `jasonod888/moge2` on Replicate) returns a metric point cloud
in the camera frame. MapAnything's cached cloud is transformed to the same
frame. One scalar per image = ratio of their median ranges. MapAnything
distances are multiplied by it; the 105 questions are re-scored. No camera
height, no operator input, no per-scene tuning.

## Result

| Configuration | median rel err | ≤10% | ≤25% |
|---|---|---|---|
| Unanchored MapAnything | 60.1% | 0.0% | 0.0% |
| **MoGe-2 auto-anchored (centroid)** | **9.0%** | **53.3%** | **85.7%** |
| Operator anchor, split-half upper bound | 8.5% | 50.9% | 90.6% |
| Gemini 3.5 Flash, same questions | 33.2% | 8.6% | 31.4% |
| MoGe-2 auto-anchored (min-distance) | 24.2% | 11.4% | 52.4% |

**The recovered scale factor is 2.504 (MAD 0.212) against a ground-truth
constant of 2.50** — a 0.16% match, from a model that never saw the
benchmark's answers.

## Reading

1. **The operator-supplied camera height is no longer structurally
   required.** Auto-anchoring lands within 0.5 points of the operator-anchor
   upper bound (9.0% vs 8.5% median) and beats the VLM baseline 3.7×.
2. Centroid-distance beats min-distance here (9.0% vs 24.2%) because this
   benchmark's GT *is* centroid distance. That is a protocol match, not a
   quality finding — the clearance rule wants min-distance and must be
   scored against min-distance GT.
3. The scale ratio's spread (MAD 0.21, p10 2.13 / p90 3.00) is the residual:
   per-image agreement is good but not perfect, and the ~15% of questions
   still above 25% error track the images where the two models disagree
   most.

## Real photos, laser ground truth — the gate is passed

Same auto-anchor, same code path, run on the two V2 packs where ground
truth is a laser scan and the imagery is real photographs
(`scripts/redwood_v2_eval.py --moge-anchor`, 4 MoGe calls per pack):

| Pack | operator anchor (camera height) | **MoGe auto-anchor** | model native |
|---|---|---|---|
| ETH3D office (DSLR, 2048 px) | 14.4 cm MAE | **14.7 cm** | 15.0 cm |
| Redwood boardroom (VGA, dim) | 27.5 cm MAE | **22.9 cm** | 24.0 cm |

- ETH3D: parity — 0.3 cm apart, and the two scale estimates agree to 1.4%
  (MoGe 1.0006 vs camera-height 0.9868).
- Redwood: **auto-anchor beats the operator anchor by 4.6 cm.** The two
  scale estimates differ by 17% there (0.932 vs 1.126) and MoGe's is the
  one closer to the laser truth — on that pack the measured camera height
  is itself the weaker number.

So the result holds on real imagery, not just renders: **no operator input,
same or better accuracy.**

## Limits — do not over-claim

- Two real packs, 7 pairs total, plus 105 synthetic questions. Decision-
  grade, not publication-grade.
- n=105 questions over 104 images, one benchmark, one model pair.
- MoGe-2 adds a second inference per image (cost + latency). It is MIT and
  self-hostable, so this is an on-prem cost, not a licence problem.
- MoGe-3 (2026-08-18, MIT, metric, 370M/1.25B) is the successor and is
  **not on any cloud API yet** — verified by probing Replicate. Testing it
  needs a rented GPU; expect it to be at least as good.

## Reproduce

```
uv run --env-file .env python scripts/moge_anchor_eval.py          # cost preview
uv run --env-file .env python scripts/moge_anchor_eval.py --live   # ~104 calls
```

Per-image scalars are cached in `outputs/nvwarehouse_v1/moge_scale/`, so
re-scoring is free. Point clouds (2M points each) are streamed and
discarded.
