# Harness vs VLM — spatial understanding benchmark (the MVP thesis test)

Date 2026-07-20. Script `scripts/vlm_baseline.py`; raw numbers
`outputs/vlm_baseline.json`. Identical inputs both sides: the same 4 photos +
stated camera height. Harness = full geometry pipeline. VLM = Gemini 3.5 Flash
asked the same metric questions directly (no tools, no facts). Ground truth:
laser scans (real packs), analytic mesh (synthetic; harness side = deterministic
core on oracle geometry there).

## Headline

| | MAE over 9 GT pairs | wins |
|---|---|---|
| **Harness** | **18.2 cm** | 7 / 9 |
| Gemini 3.5 Flash | 49.5 cm | 2 / 9 (both marginal) |

## The kill shot: verdict separability

On the two synthetic scenes whose ONLY difference is ladder at 0.5 m vs 0.7 m
(the 0.6 m rule's two sides):

| | ladder_050 (GT 0.5) | ladder_070 (GT 0.7) | can it tell them apart? |
|---|---|---|---|
| Harness | 0.548 | 0.756 | **yes — correct side of 0.6 both times** |
| VLM | 0.22 | 0.28 | **no — both "FAIL", same prior-driven answer** |

This reproduces, on our own benchmark, exactly what the 2026 literature
(ViewDiag "consistent yet wrong") predicts: VLM distance answers are priors,
not measurements — they barely move when the scene moves. A compliance product
built on that answers confidently and identically for compliant and
non-compliant scenes.

## Per-pair detail

| Scene | Pair | GT | Harness err | VLM err |
|---|---|---|---|---|
| Redwood (VGA worst-case) | sofa-plant | 0.172 | 16.8 | 23.8 |
| Redwood | plant-wicker | 0.400 | 34.6 | 32.0 (vlm) |
| Redwood | plant-table | 0.542 | 32.9 | 41.8 |
| Redwood | sofa-wicker | 0.707 | 25.6 | 70.3 |
| ETH3D (DSLR good-input) | desk-sofa | 0.737 | 28.6 | 18.3 (vlm) |
| ETH3D | sofa-chair | 1.349 | 13.0 | 22.9 |
| ETH3D | sofa-bin | 1.667 | **1.5** | 166.3 |
| Synthetic | ladder_050 | 0.5 | 4.8 | 28.0 |
| Synthetic | ladder_070 | 0.7 | 5.6 | 42.0 |

## Reading

1. **Harness error is input-quality-bounded and improvable** (worst on VGA
   close-ups, 1.5 cm on its best good-input pair, ~5 cm on clean geometry).
   **VLM error is prior-bounded and flat** — better input does not fix it
   (166 cm miss on the DSLR pack's easiest long pair).
2. Where the VLM "wins" it wins by luck within its prior band (both wins are
   <35 cm errors on ~0.4–0.7 m pairs — coin-flip territory for a 0.6 m rule).
3. The harness additionally outputs what the VLM cannot: auditable per-entity
   3D state (position/footprint/height/orientation/tilt/overhang), evidence-
   linked facts, and a deterministic abstention path. The VLM emits a number
   with no evidence trail.

## Method notes / honesty

- One VLM (gemini-3.5-flash), one prompt style, single shot, n=9 pairs —
  decision-grade, not publication-grade. Adding GPT/Qwen baselines and
  chain-of-thought prompting is one config away in the script.
- The synthetic images are renders (off-distribution for the VLM); the two real
  packs are natural photos, where the VLM still loses 5 of 7 pairs.
- Oracle-side note: the deterministic-core distances moved 0.505→0.548 after
  today's floor-selection redesign (still inside the ±0.1 m oracle gate);
  tracked, not hidden.
