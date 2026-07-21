# Benchmark adoption sweep — three new public exams in one day

Date 2026-07-21. Owner directive: adopt everything adoptable. All runners
cache providers on disk (re-scoring free); MapAnything is GPU-second
billed (~free at this scale); the two new mask-provided benchmarks cost
zero fal SAM.

## Scoreboard (all: success = rel err <= 25%, plus median rel err)

| Benchmark | Regime | Harness (no anchor) | VLM (Gemini 3.5 Flash) | Verdict |
|---|---|---|---|---|
| NVIDIA warehouse val (n=105) | synthetic warehouse 1-11 m, GT masks | 60.1% err / 0% ... **anchored: 8.5% err / 90.6%** | 33.2% err / 31.4% | **Anchored harness 3x ahead in-domain** |
| SpatialRGPT-Bench quant. distance (n=359) | mixed real+synthetic, 5 sources | 34.7% err / 40.2% | 32.7% err / 38.7% | **Tie** — unanchored mono already matches frontier VLM |
| ARKitScenes (n=20 pairs, 29 scenes) | real handheld iPhone photos, box-centroid GT | mono 22.4% / 55%; multi 22.3% / 55% | 23.9% / 55% | **Three-way tie** on product-form imagery |
| Q-Spatial++ (n=101) | close-range household, tape-measured | 54% success@2x, 40 abstentions | 90% success@2x | VLM home turf (from boundary map) |

Per-source pattern (SpatialRGPT): hypersim room-scale 52.3% vs SUNRGBD
close-range 31.7% — the room-scale-vs-close-range boundary reproduces
inside a single benchmark.

## Readings

1. **Unanchored mono has caught up to frontier VLM everywhere except
   close-range clutter** — and the harness has an anchoring path the VLM
   lacks: the warehouse anchored result (8.5% median) is the ceiling the
   product reaches with its camera-height input; nothing comparable exists
   for the VLM.
2. **ARKitScenes multi ≈ mono here** (unlike ETH3D 1.3%): evenly-sampled
   handheld frames give weak baselines around specific pairs, and the
   region measurement still reads one frame's pointmap. Protocol note:
   GT is box-centroid distance while we measure visible-surface centroids
   — an inherent ~10-20% protocol noise floor; treat these numbers as
   comparative (harness vs VLM on identical protocol), not absolute.
3. Coverage: 17/376 SpatialRGPT images failed provider-side (skipped,
   logged); ARKitScenes yield is 20 pairs from 29 scenes after the
   occlusion + overlap gates — strict on purpose after the close-up
   mis-selection bug (caught visually, fixed with FARO-depth visibility
   checks).

## Probe verdicts (why the others were dropped)

- ANavS warehouse: LiDAR-only, no camera — nothing to photograph.
- SODA: 2D boxes only, zero metric GT; research-only license, messy hosting.
- Rohbau3D: no real photos (point-cloud renders only) — parked; CC-BY-4.0
  data license is clean if renders ever become useful.
- ARKitScenes license: Apple custom — commercial branch exists for
  <700M-MAU licensees; have legal confirm before external use.
- SpatialRGPT-Bench: no license declared anywhere — internal use only.

## Runners

`scripts/nvwarehouse_eval.py`, `scripts/spatialrgpt_eval.py`,
`scripts/arkitscenes_eval.py`, `scripts/qspatial_eval.py` — same skeleton:
disk cache, --live spend gate with call count, --vlm baseline, graceful
per-image skip, transient 5xx retry.
