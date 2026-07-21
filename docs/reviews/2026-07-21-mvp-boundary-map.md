# MVP boundary map — where the harness wins, loses, and abstains

Date 2026-07-21. Owner directive: map the MVP's limits with real spend.
Everything below is measured, cached on disk, and re-scoreable for free.
Actual burn correction: MapAnything on replicate is billed in GPU-seconds
— ~190 mono reconstructions cost well under $1 (billing page: $0.12 usage
at the 90-run mark), not the ~$10 first estimated. The earlier "stopped
by credit exhaustion" claim was wrong: the stop was a transient 502,
since fixed with retry+skip. fal SAM ~$1.5, Gemini cents.

## The map

| Regime | Evidence | Harness | VLM (Gemini 3.5 Flash) | Boundary verdict |
|---|---|---|---|---|
| Room-scale, multi-view, real photos (laser GT) | V2: ETH3D DSLR 14.4 cm MAE (best 1.5 cm); harness-vs-VLM 9 pairs | **18.2 cm MAE, 7/9 wins, verdict separability at 0.6 m** | 49.5 cm, prior-collapsed (0.22/0.28 on 0.5/0.7 scenes) | **Harness home turf** |
| Room-scale, single photo | mono ladder: 46→30 cm err (SOR); plan-view demo | Screening-grade; correct rule side; explicit reduced-capture warning | untested here | Usable for triage, not measurement |
| Household close-range, single photo (Q-Spatial++, n=101, tape-measured GT) | full run | 54.1% ≤2×, median ratio 1.88, **40/101 honest abstentions** | **90.1% ≤2×, median 1.25, answers everything** | **VLM home turf** — cm-scale gaps beyond mono depth; object priors dominate |
| Synthetic warehouse, single photo (NVIDIA val, FULL n=105, GT masks, 1–11 m) | full run + VLM head-to-head | Unanchored: median rel err **60.1%**, ≤25% success 0. **With ONE global scale anchor (split-half, held-out): median 8.5%, ≤25% success 90.6%** | median 33.2%, ≤25% success 31.4% (regions drawn, SpatialRGPT protocol) | **The boundary in one number: scale anchor.** gt/pred ratio 2.50 ± 0.24 MAD — a constant gauge error, not geometry error. Anchored harness beats VLM 3× in-domain; unanchored loses |
| Far field (>15 m) | storage rack: negative height; LOCO pallets R@0.3 0.33 | Quality gates reject / recall capped | VLM answers regardless | Both fail; harness fails honestly |
| Dense small objects | V3 LOCO: ~30 GT/image vs 32-mask API cap | recall ceiling by construction | — | Detector-swap (V4) territory |

## Reading the map

1. **The thesis survives, sharpened**: in the product's regime (room-scale
   clearance, real photos, multi-view or anchored single-view) the harness
   measures and the VLM guesses. Outside it (close-range household
   clutter) the 2026 frontier VLM's priors beat unanchored mono geometry —
   90% vs 54%, measured, not argued away.
2. **The single broken link is scale anchoring, not geometry shape — now
   proven, not hypothesized**: warehouse gt/pred is a tight constant
   (2.50, MAD 0.24); one global scale learned on half the images takes the
   held-out half from 60% median error to **8.5%**, beating the VLM
   (33.2%) three-fold in-domain. Camera height (our production anchor) is
   exactly this gauge — the product keeps demanding it, and V4 adds
   fallback anchors (known object dimensions, floor-plane priors).
   Q-Spatial's close-range misses are the same story at a scale where no
   anchor exists at all.
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
