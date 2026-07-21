# Q-Spatial-Bench adoption — smoke result (harness loses this one; recorded honestly)

Date 2026-07-20. Runner `scripts/qspatial_eval.py` (Apache-2.0 benchmark,
101 tape-measured Q-Spatial++ questions; full provider caching; benchmark
protocol: success = max(pred/gt, gt/pred) <= 2).

Harness path evaluated: single photo -> MapAnything native metric mono ->
SAM masks for the two question phrases -> robust min 3D distance. No floor
fit and no production evidence gates (many scenes are tabletop close-ups).

## Smoke (n=6 distinct-image questions, ~$1)

| | answered | success@2x | median ratio |
|---|---|---|---|
| Harness (mono primitive) | 4/6 | **0.25** | 2.21 |
| Gemini 3.5 Flash (raw) | 6/6 | **0.83** | 1.21 |

**The VLM wins on this benchmark slice.** No spin: frontier-2026 Gemini is
far stronger on close-range household scenes than the 2024 models the
benchmark paper originally embarrassed.

## Diagnosis (verified per-question, not guessed)

1. Zero-depth garbage under masks: one mask's 3D centroid sat at range
   0.00 m (invalid mono reconstruction region) — initially produced a 43x
   wrong answer. Fixed: near-zero-range points are dropped and the harness
   abstains (median ratio 13.6 -> 2.21). Wrong -> abstain, per house rule.
2. Centimetre gaps are beyond mono: GT 1.5 cm between adjacent tabletop
   objects measured as 0.38 m — mono depth smoothing cannot resolve
   between-object gaps at that scale. Fundamental, not a bug.
3. Distribution favors priors: household object spacings are exactly what
   VLM priors memorize; room-scale geometric scenes (our domain evals,
   laser GT) show the opposite ordering (harness 18.2 cm vs VLM 49.5 cm
   MAE, 7/9 wins; verdict separability the VLM lacks).

## Standing decisions

- Q-Spatial++ stays in the eval suite as an ADVERSARIAL benchmark — it
  cleanly exposes the mono-scale close-range boundary of the harness, which
  is exactly what an eval harness is for. Not cherry-picked away.
- Full 101-question run (~$9 MapAnything) blocked: replicate credit is
  under $5 (throttled to 6 req/min, burst 1; backoff added to the runner).
  Run after top-up: `uv run --env-file .env python scripts/qspatial_eval.py --live --vlm`.
- NVIDIA PhysicalAI-Spatial-Intelligence-Warehouse (the in-domain 19k-QA
  benchmark) is gated on HF: needs the owner's HF account acceptance +
  token before download.
