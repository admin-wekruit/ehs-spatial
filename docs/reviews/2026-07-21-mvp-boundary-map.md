# MVP boundary map — where the harness wins, loses, and abstains

Date 2026-07-21. Owner directive: spend all remaining provider credit
mapping the MVP's limits. Everything below is measured, cached on disk,
and re-scoreable for free. Total burn this cycle: ~$10 replicate
(87 Q-Spatial + 3 warehouse mono reconstructions), ~$1.5 fal SAM,
cents of Gemini.

## The map

| Regime | Evidence | Harness | VLM (Gemini 3.5 Flash) | Boundary verdict |
|---|---|---|---|---|
| Room-scale, multi-view, real photos (laser GT) | V2: ETH3D DSLR 14.4 cm MAE (best 1.5 cm); harness-vs-VLM 9 pairs | **18.2 cm MAE, 7/9 wins, verdict separability at 0.6 m** | 49.5 cm, prior-collapsed (0.22/0.28 on 0.5/0.7 scenes) | **Harness home turf** |
| Room-scale, single photo | mono ladder: 46→30 cm err (SOR); plan-view demo | Screening-grade; correct rule side; explicit reduced-capture warning | untested here | Usable for triage, not measurement |
| Household close-range, single photo (Q-Spatial++, n=101, tape-measured GT) | full run | 54.1% ≤2×, median ratio 1.88, **40/101 honest abstentions** | **90.1% ≤2×, median 1.25, answers everything** | **VLM home turf** — cm-scale gaps beyond mono depth; object priors dominate |
| Synthetic warehouse, single photo, no scale anchor (NVIDIA val, n=2 before credit death) | preliminary | **~3.3× systematic underscale** (8.15→2.44, 9.81→3.28 m) | untested | **Unanchored mono native scale is the broken link** — consistent factor, so an anchor (camera height / floor prior / one known length) likely fixes most of it. n=2: hypothesis, not conclusion |
| Far field (>15 m) | storage rack: negative height; LOCO pallets R@0.3 0.33 | Quality gates reject / recall capped | VLM answers regardless | Both fail; harness fails honestly |
| Dense small objects | V3 LOCO: ~30 GT/image vs 32-mask API cap | recall ceiling by construction | — | Detector-swap (V4) territory |

## Reading the map

1. **The thesis survives, sharpened**: in the product's regime (room-scale
   clearance, real photos, multi-view or anchored single-view) the harness
   measures and the VLM guesses. Outside it (close-range household
   clutter) the 2026 frontier VLM's priors beat unanchored mono geometry —
   90% vs 54%, measured, not argued away.
2. **The single broken link is scale anchoring, not geometry shape**: the
   warehouse failure is a consistent factor, and Q-Spatial's worst misses
   are close-range where mono depth smooths gaps. Camera height (our
   production anchor) is exactly the fix the benchmarks deny us — the
   product should keep demanding it, and V4 should add fallback anchors
   (known object dimensions, floor-plane priors).
3. **Abstention is working as designed**: 40/101 Q-Spatial abstentions and
   the far-field rejections are the C5 behavior — the VLM's 100% answer
   rate on questions it gets wrong at 10% is the failure mode we sell
   against.

## Resume paths (all cached, free until the paid step)

- NVIDIA full 104-image run after replicate top-up (~$10):
  `uv run --env-file .env python scripts/nvwarehouse_eval.py --live`
- Anchored-scale variant for warehouse (floor fit + assumed camera height,
  or single-scene calibration constant — must be disclosed as calibrated):
  next experiment, likely closes most of the 3.3×.
- Q-Spatial re-scores are free; adding GPT/Qwen baselines is a config.
