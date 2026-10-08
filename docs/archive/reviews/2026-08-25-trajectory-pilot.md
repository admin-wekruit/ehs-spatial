# MTMC trajectory pilot: states and trajectories judged deterministically over time

Date 2026-08-25. Runner `scripts/trajectory_eval.py`. Dataset:
`nvidia/PhysicalAI-SmartSpaces`, `MTMC_Tracking_2025/val/Warehouse_016`
(CC-BY-4.0, ungated). Builds directly on the n=140 distance eval
(`docs/reviews/2026-08-25-mtmc-eval.md`).

## Honesty notes, up front

- **The imagery is SYNTHETIC** (Omniverse/Cosmos Transfer). The
  collection's only real-world captures ship without ground truth.
- **Tracking identity comes from GT in this pilot.** Object ids and the
  per-camera visible boxes are the dataset's own annotations. This
  validates the metrology (world-position lifting) and the judgment
  layer (deterministic temporal rules), NOT a tracker; the tracker slot
  is a commodity being scanned separately.
- **Estimated positions are surface points.** We take the median 3D
  point of the GT visible-box region — the object's visible surface —
  while GT locations are object centres. That surface-to-centre offset
  (roughly half an object depth, toward the camera) is inside every
  reported error, not corrected away.
- **Camera_07 stays broken and stays in the numbers.** No parameter was
  tuned to hide it.

## Setup

15 timestamps evenly spread over the 5-minute sequence (frame indices
0..8400 step 600, 20 s apart), 4 cameras -> 60 mono frames; the baseline's
12 frames and their cached geometry are a subset. Per frame: MapAnything
mono + MoGe-2 auto anchor (cached under `outputs/mtmc_v1/`). At 4 of the
15 timestamps (0/2400/4800/7200) all four same-timestamp frames also go
to MapAnything jointly (multiview tier, `outputs/mtmc_v1/geometry_mv/`).
ZERO SAM calls (GT boxes are the regions), ZERO Gemini calls.

World lifting: box-region median 3D point -> MapAnything camera frame ->
metres via the MoGe anchor scale -> dataset world frame via the inverse
of the per-camera calibration extrinsic. The map overlay uses the
verified floor-map convention `px=(x+tx)*s, py=H-(y+ty)*s` (checked by
inverse-projecting taped-zone corners from Camera_000000.png onto
map.png).

## Layer 1 — the distance protocol at 5x the samples

714 pair-distance questions over 60 images (vs 140 over 12); 712
answered, no image failures.

| Configuration | n=712 median | n=140 baseline | n=712 <=25% | baseline |
|---|---|---|---|---|
| Unanchored mono, centroid | 61.5% | 60.5% | 3.9% | 5.7% |
| Unanchored mono, min-dist | 68.1% | 66.8% | 3.1% | 3.6% |
| **MoGe-anchored, centroid** | **17.4%** | **17.4%** | 61.2% | 61.4% |
| MoGe-anchored, min-dist | 27.1% | 25.3% | 47.1% | 50.0% |

**The anchored 17.4% median holds exactly at 5x the samples.** Per-camera
anchored centroid medians: `Camera` 11.1% (was 10.4%), `Camera_01` 11.7%
(10.0%), `Camera_03` 17.1% (18.3%), **`Camera_07` 68.0% (was 107.9%)** —
still broken, as expected; the larger sample softens the tail but the
view remains 4-6x worse than its siblings. Nothing was tuned.

## Layer 2 — trajectories: mono lift vs multiview lift

452 lifted positions out of 453 visible GT slots (99.8% coverage), 43
per-(camera, object) series. Errors are XY (floor-plane) distances to the
GT object centre.

Mono (all 15 timestamps), per camera:

| Camera | n | median XY err | p90 |
|---|---|---|---|
| Camera | 118 | 1.24 m | 1.86 m |
| Camera_01 | 110 | 0.91 m | 1.28 m |
| Camera_03 | 109 | 1.31 m | 2.84 m |
| **Camera_07** | 115 | **5.25 m** | 9.69 m |

Pooled: median 1.28 m XY (1.94 m 3D; the 3D number carries the
surface-vs-centre offset plus Camera_07's scale wreckage). Median
per-object ATE 1.34 m.

Multiview at the 4 joint timestamps, same objects, side by side:

| Lift | n | median XY | p90 | mean |
|---|---|---|---|---|
| Mono (same timestamps) | 118 | 1.23 m | 5.57 m | 2.18 m |
| Multiview, native scale | 118 | 6.47 m | 10.94 m | 6.81 m |
| Multiview, MoGe-anchored | 118 | 1.63 m | 5.48 m | 2.08 m |

Two findings, reported as measured:

1. **MapAnything's multiview gauge is NOT self-calibrating here.** The
   joint reconstruction needs the same ~2.5x anchor as mono (per-timestamp
   MoGe/joint ratios 2.45-2.75); at native scale it is unusable (6.5 m
   median).
2. **"Seen from many angles is best" holds for robustness, not for the
   median.** Anchored multiview is *worse* than mono at the median
   (1.63 vs 1.23 m) — the joint solve drags the two good views down
   (Camera 1.91 vs 1.22, Camera_01 1.80 vs 0.94) — but it *repairs the
   broken view* (Camera_07: 2.00 vs 5.34 m, a 2.7x recovery; Camera_03:
   0.41 vs 1.31 m) and wins on mean and p90. Multiview buys a floor under
   the worst camera, not a better best case.

## Layer 3 — deterministic temporal judgment

Rules are pure shapely/numpy over per-timestamp positions; the mono error
band (±0.35 m, `ehs_spatial.rules.ERROR_BUDGET_MONO_M`) is the honesty
buffer: a measurement within the band of a boundary/threshold becomes
NEEDS_REVIEW rather than a razor-edge verdict. GT verdicts use band 0
(perfect tracking, binary) — the delta between the two verdict streams
is the finding.

Choices, documented:

- **R1 keep-clear zone**: a 4 x 4 m square centred on the parked
  Transporter AMR (object 349, static at (-1.77, -9.13) all sequence) —
  "pedestrians keep clear of the AMR staging area", the AMR itself
  exempt. Chosen because it is a plausible EHS zone AND GT traffic
  actually crosses it at sampled timestamps, so PASS, FAIL and the band
  all get exercised. Verdict per timestamp over the objects both series
  cover (same evidence set for both sides).
- **R2 min distance over time**: Person 699 vs Transporter 349,
  threshold 2.0 m — their GT separation sweeps ~1.7-11 m across the
  samples, crossing the threshold in both directions.
- **R3 speed**: displacement/Δt per 20 s window vs a 0.25 m/s movement
  limit, for every object, from BOTH the GT trajectories and ours. GT
  window speeds span 0-0.58 m/s, so the limit exercises both sides; the
  banded rule semantics are what is validated, not any specific site
  speed limit. Speed band = 2 x 0.35 m / 20 s = 0.035 m/s (two banded
  endpoints).

Results — GT verdict (perfect tracking) vs our estimated verdict:

| Rule | timestamps | est NEEDS_REVIEW | decided | agree | hard disagreements |
|---|---|---|---|---|---|
| R1 keep-clear zone | 15 | 6 | 9 | 7 (77.8%) | 2 |
| R2 min distance | 15 | 4 | 11 | 11 (100%) | 0 |
| R3 speed (145 windows) | 145 | 28 | 117 | 104 (88.9%) | 13 |

The confusion detail is the real story:

- **No GT violation was ever passed.** Across all three rules, GT FAIL ->
  estimated PASS occurred ZERO times (R1: 8 GT FAILs -> 4 est FAIL +
  4 NEEDS_REVIEW; R2: 2 GT FAILs -> both NEEDS_REVIEW; R3: 30 GT FAILs ->
  20 est FAIL + 10 NEEDS_REVIEW). With the band in place, our metrology's
  failure mode is review queues and false alarms, never silently missed
  violations — on this sample.
- **All hard disagreements are false alarms near the line.** R1's two
  (t=80 s, t=180 s) are GT depths of -0.11 and -0.15 m — objects standing
   15 cm outside the zone that our ~1 m position noise pushed inside.
  R3's thirteen are all est-FAIL on GT-PASS windows where position jitter
  inflated displacement (worst: GT 0.23 m/s read as 0.85 m/s).
- **The speed band was too tight, knowingly.** 2 x 0.35 m / 20 s assumes
  endpoint errors bounded by the clearance band, but the measured mono
  position error (median 1.28 m) is ~4x the band, so R3 kept 13 false
  alarms out of the review bucket. Deriving the temporal band from the
  measured ATE instead (2 x 1.28 / 20 = 0.13 m/s) is the obvious next
  calibration; we report the miscalibration rather than retrofit it.

## Reading

1. **The harness thesis extends to time.** Per-timestamp geometry plus
   deterministic rules (shapely, zero model calls in the judgment layer)
   produces state timelines, min-distance-over-time, and speed verdicts
   whose banded form never contradicted GT in the dangerous direction on
   452 positions and 175 rule evaluations.
2. **Metrology degradation shows up as review volume, not wrong verdicts.**
   ~27-40% of timestamps land in NEEDS_REVIEW at mono accuracy; decided
   verdicts agree 78-100% with perfect tracking. The band is doing its
   job; the cost of mono-grade geometry is review load.
3. **The multiview trade is now measured on trajectories** (finding 2 of
   Layer 2): it floors the worst camera rather than improving the best.
4. **Camera_07 stays broken through every layer** (68% distances, 5.2 m
   positions) and multiview is what recovers it (2.0 m) — consistent with
   the production stance: multi-view capture + per-run scale confidence.

## Spend

48 MapAnything mono + 4 MapAnything multiview (4 images each) + 48
MoGe-2 calls on Replicate, GPU-second billed — same per-call class as the
baseline run (~190 mono < $1), so well under $2 total. 0 SAM, 0 Gemini
calls. All cached under `outputs/mtmc_v1/`; re-scores are free.

## Reproduce

```
uv run --env-file .env --with imageio-ffmpeg python scripts/trajectory_eval.py          # free preview
uv run --env-file .env --with imageio-ffmpeg python scripts/trajectory_eval.py --live   # spend once
```

Outputs: `outputs/mtmc_v1/trajectory_report.json` (scorecard) and
`outputs/mtmc_v1/trajectory_map.png` (GT vs estimated trajectories, zone,
violation timestamps on the floor map). Offline tests:
`tests/test_trajectory_eval.py` (transform round-trip, banded verdicts,
speed windows, agreement math, scorecard schema).
