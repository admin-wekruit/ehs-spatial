# route/jev: a primitive or a generated model, per object card

Question: can the display model be routed per object (SIMPLE: a primitive; COMPLEX: RecGen / SAM 3D) by rules, zero-shot embeddings, Jev-Omni or a cascade, accurately and cheaply? Code: scripts/route_jev.py, modal_apps/route_jev.py (branch route/jev). Labels: the agent's, by looking at blind sheets; not ground truth. Every threshold / temperature is fitted on two videos and scored on the third (rotated); nothing is trained.

**Answer.** Jev-Omni can route: alone it is the best single router held out (0.875, COMPLEX precision 0.83 / recall 0.74, ECE 0.075, no fitted number) against 0.712 for the best rules, 0.802 for zero-shot embeddings and 0.837 for the core's Qwen3-VL-8B. Recommended: the cascade, a class decides where it fixes the shape and Jev-Omni Q5 decides the rest, on the well-observed cards only, no VLM: 0.904 (post-hoc class list, see the caveats), Jev asked about 32 / 40 / 152 cards per video (3.3 / 4.1 / 15.7 s of the service), 38 / 0 / 111 RecGen calls per video (ME340 / Sam's Club / Walmart). Spend 1.56 USD.

## Labels

| video | cards | labelled clear | complex | simple | unclear (left out) | labels' estimate of complex cards (weighted) |
|---|---|---|---|---|---|---|
| me340 | 254 | 107 | 59 | 48 | 23 | 112 |
| samsclub-a2 | 623 | 102 | 3 | 99 | 3 | 4 |
| walmart | 721 | 104 | 33 | 71 | 1 | 308 |

Sample: 100 cards per video stratified by class (sqrt allocation, seed 0) plus a top-up draw (seed 1) of 30 / 5 / 5 for the unclear ones; weights N / n per stratum for the per-card numbers. Primitive labels on the SIMPLE ones: box 160, cylinder 9, open frame 8, plane 41. Baseline 'every card a primitive': 0.696 (weighted 0.724).

## Comparison (held out by video, pooled over the three test videos)

| router | acc | acc per card (weighted) | COMPLEX precision / recall | missed / false complex | ECE | tiers | acc ME340 / Sam's / Walmart | routed to RecGen, all cards ME340 / Sam's / Walmart | of those well-observed | latency per object |
|---|---|---|---|---|---|---|---|---|---|---|
| a0 rules: class only | 0.617 | 0.500 | 0.39 / 0.44 | 53 / 67 | 0.331 | rules 100% | 0.757 / 0.627 / 0.462 | 113 / 229 / 184 | 29 / 40 / 51 | <1 us CPU |
| a1 rules: geometry only | 0.655 | 0.681 | 0.24 / 0.06 | 89 / 19 | 0.252 | rules 100% | 0.458 / 0.843 / 0.673 | 9 / 96 / 122 | 3 / 20 / 39 | <1 us CPU |
| a2 rules: class, else geometry | 0.712 | 0.626 | 0.52 / 0.61 | 37 / 53 | 0.324 | rules 100% | 0.776 / 0.843 / 0.519 | 159 / 86 / 320 | 38 / 20 / 99 | <1 us CPU |
| b zero-shot (encoder, crop, prompts picked on training videos) | 0.802 | 0.761 | 0.76 / 0.51 | 47 / 15 | 0.089 | zero-shot 100% | 0.794 / 0.931 / 0.683 | 138 / 29 / 19 | 39 / 2 / 3 | 6 ms A100 |
| b zero-shot pe-core-l masked classes | 0.722 | 0.807 | 0.56 / 0.40 | 57 / 30 | 0.086 | zero-shot 100% | 0.514 / 0.853 / 0.808 | 22 / 51 / 387 | 7 / 5 / 125 | 6 ms A100 |
| b zero-shot siglip2-so400m masked classes | 0.642 | 0.679 | 0.34 / 0.20 | 76 / 36 | 0.157 | zero-shot 100% | 0.458 / 0.765 / 0.712 | 3 / 114 / 260 | 1 / 17 / 102 | 6 ms A100 |
| c jev q2 raw | 0.671 | 0.686 | 0.48 / 0.98 | 2 / 101 | 0.317 | jev 100% | 0.692 / 0.618 / 0.702 | 207 / 265 / 522 | 54 / 43 / 145 | 2.00 q x 0.103 s = 0.206 s |
| c jev q2 calibrated | 0.875 | 0.902 | 0.81 / 0.77 | 22 / 17 | 0.032 | jev 100% | 0.804 / 0.961 / 0.865 | 116 / 14 / 271 | 40 / 1 / 97 | 2.00 q x 0.103 s = 0.206 s |
| c jev q5 raw | 0.875 | 0.912 | 0.83 / 0.74 | 25 / 14 | 0.075 | jev 100% | 0.766 / 0.980 / 0.885 | 117 / 15 / 329 | 37 / 1 / 111 | 1.00 q x 0.103 s = 0.103 s |
| c jev q5 calibrated | 0.885 | 0.925 | 0.90 / 0.69 | 29 / 7 | 0.097 | jev 100% | 0.776 / 0.980 / 0.904 | 92 / 12 / 302 | 27 / 0 / 105 | 1.00 q x 0.103 s = 0.103 s |
| c jev qyn calibrated | 0.661 | 0.657 | 0.39 / 0.32 | 55 / 40 | 0.121 | jev 100% | 0.524 / 0.701 / 0.737 | - | - | 1.00 q x 0.103 s = 0.103 s |
| c' vlm qwen3-vl-8b q2 calibrated (reference: a VLM for every card) | 0.802 | 0.818 | 0.69 / 0.63 | 35 / 27 | 0.083 | vlm 100% | 0.692 / 0.922 / 0.798 | 77 / 69 / 384 | 24 / 10 / 121 | 0.042 s in the core |
| c' vlm qwen3-vl-8b q5 raw (reference) | 0.837 | 0.865 | 0.71 / 0.79 | 20 / 31 | 0.119 | vlm 100% | 0.748 / 0.980 / 0.788 | 141 / 23 / 338 | 35 / 2 / 106 | 0.021 s in the core |
| d cascade: class -> jev q2 | 0.815 | 0.787 | 0.65 / 0.83 | 16 / 42 | 0.217 | jev 34%, rules 66% | 0.822 / 0.961 / 0.663 | 165 / 13 / 366 | 49 / 1 / 132 | 0.68 q x 0.103 s = 0.071 s |
| d cascade: class -> jev q5 | 0.812 | 0.806 | 0.65 / 0.82 | 17 / 42 | 0.252 | jev 34%, rules 66% | 0.804 / 0.922 / 0.712 | 152 / 40 / 395 | 45 / 5 / 138 | 0.34 q x 0.103 s = 0.035 s |
| d cascade: class -> zero-shot | 0.684 | 0.593 | 0.48 / 0.52 | 46 / 53 | 0.334 | rules 66%, zero-shot 34% | 0.776 / 0.804 / 0.471 | 136 / 117 / 189 | 34 / 14 / 53 | 0.00 q x 0.103 s = 0.000 s + 6 ms |
| d jev q2 calibrated -> vlm on jev's band (P 0.3-0.7) | 0.872 | 0.904 | 0.82 / 0.74 | 25 / 15 | 0.040 | jev 87%, vlm 13% | 0.766 / 0.980 / 0.875 | 104 / 19 / 315 | 35 / 1 / 103 | 2.00 q x 0.103 s = 0.206 s + VLM |
| d cascade: class -> jev q2 -> vlm on jev's band | 0.827 | 0.804 | 0.67 / 0.84 | 15 / 39 | 0.214 | jev 29%, rules 66%, vlm 5% | 0.832 / 0.971 / 0.683 | 159 / 15 / 391 | 49 / 2 / 136 | 0.68 q x 0.103 s = 0.071 s + VLM |
| d cascade: class -> jev q2 (POST HOC class table) | 0.885 | 0.889 | 0.82 / 0.79 | 20 / 16 | 0.113 | jev 48%, rules 52% | 0.841 / 0.971 / 0.846 | 151 / 8 / 199 | 45 / 0 / 80 | 0.96 q x 0.103 s = 0.099 s |
| **d cascade: class -> jev q5 raw (POST HOC class table)** | 0.904 | 0.923 | 0.86 / 0.82 | 17 / 13 | 0.089 | jev 48%, rules 52% | 0.850 / 0.971 / 0.894 | 138 / 10 / 320 | 38 / 0 / 111 | 0.48 q x 0.103 s = 0.049 s |

Rows: a = rules (the pre-registered class prior over cards.TAXONOMY; geometry = the best primitive's fit residual over the longest side, cut fitted on the training videos); b = zero-shot text prompts (Platt on the training videos); c = Jev-Omni (Q2: simple shape or detailed model, both option orders averaged; Q5: which of four shapes or none; qyn: yes / no), raw or Platt-calibrated; c' = the core's Qwen3-VL-8B on the same image and questions; d = cascades. POST HOC rows use the class list revised after labelling.

Paired bootstrap (resampled within each video), accuracy minus that of 'c jev q2 calibrated':

- a0 rules: class only: -0.259 (95 % -0.313 to -0.201)
- a2 rules: class, else geometry: -0.163 (95 % -0.214 to -0.112)
- b zero-shot (encoder, crop, prompts picked on training videos): -0.073 (95 % -0.118 to -0.029)
- c jev q5 raw: +0.000 (95 % -0.038 to +0.038)
- d cascade: class -> jev q2: -0.061 (95 % -0.099 to -0.022)
- d cascade: class -> jev q2 (POST HOC class table): +0.010 (95 % -0.022 to +0.042)
- c' vlm qwen3-vl-8b q2 calibrated (reference: a VLM for every card): -0.073 (95 % -0.118 to -0.032)
- d jev q2 calibrated -> vlm on jev's band (P 0.3-0.7): -0.003 (95 % -0.029 to +0.022)

## Recommended rule

```
-> (display model, why): 'primitive', 'generated' or 'ask jev' (then call again with Jev-Omni's Q5 probabilities: raw,
    one question on the card's outlined best view). 1) A card no generator takes (display_model.well_observed fails) keeps its
    primitive, no question. 2) A class that fixes the shape decides (FIXED_SHAPE_POST_HOC: machines, tools, carts, furniture,
    cables -> generated; boxes, pallets, shelves, racks, signs, pipes, boards -> primitive). 3) Jev-Omni Q5 for the rest:
    P(none of the simple shapes) > cut -> generated. No VLM.

generated: families furniture, handling, machine, ppe, tool; classes cable, cup, emergency stop button, eyewash station, fan, fire alarm, fire extinguisher, guard, hose, keyboard, ladder, mannequin, phone, printer, rag, railing, safety cone, stairs, step stool, work platform, wrap
primitive: bollard, book, box, cabinet, cable tray, can, clipboard, column, control panel, door, drum, duct, electrical outlet, exit sign, first aid kit, floor drain, floor marking, label, locker, metal sheet, pallet, paper, pipe, rack, refrigerator, safety sign, shelf, sign, spill, stacked boxes, switch, trash can, vent, wall panel, whiteboard, window, wooden board
Jev-Omni (every other class, and cards with no class):
  state: This image is cropped from a video walk-through of a workplace (a workshop, warehouse or store). A white outline with a black edge, tagged 1, marks one region.
  question: Which shape best represents the outlined object in a 3D model of the place?
  A. a flat panel, board, sign, wall or door
  B. a plain box, carton, block or stack of boxes
  C. a cylinder: a pipe, pole, drum or roll
  D. an open frame: a shelf or rack
  E. none of these simple shapes: a machine, cart, tool, equipment, furniture or an irregular object
  generated when P(E) > 0.5 (raw probabilities, no fitted number)
```

Pooled 0.904, COMPLEX precision 0.86 / recall 0.82, per video 0.850 / 0.971 / 0.894; on the well-observed cards (the ones it routes) 0.920 (n 75). The class tier decides 52% of the labelled cards. The cut is the cost knob (P > cut goes to the generator); its effect with the class tier fixed:

| cut | acc | COMPLEX precision / recall | routed, all cards ME340 / Sam's / Walmart | of those well-observed (RecGen calls) |
|---|---|---|---|---|
| 0.2 | 0.859 | 0.71 / 0.90 | 160 / 58 / 426 | 44 / 7 / 134 |
| 0.3 | 0.882 | 0.78 / 0.85 | 150 / 22 / 381 | 41 / 1 / 122 |
| 0.4 | 0.901 | 0.83 / 0.84 | 143 / 13 / 348 | 38 / 0 / 115 |
| 0.5 | 0.904 | 0.86 / 0.82 | 138 / 10 / 320 | 38 / 0 / 111 |
| 0.6 | 0.914 | 0.90 / 0.81 | 130 / 8 / 296 | 33 / 0 / 105 |
| 0.7 | 0.914 | 0.91 / 0.80 | 127 / 6 / 265 | 32 / 0 / 99 |

## Latency

- Jev-Omni service, alone on one NVIDIA A100 80GB PCIe, called from a CPU container (the core's stand-in) through Modal: 0.0881 s GPU per question (batches of 8), 0.1031 s per question round trip in 64-question requests (hop 0.789 s a request of 2.35 MB); one question alone 0.488 s round trip (0.117 s of it compute). Cold start 43.37 s (load 28.91 s), 22.94 GiB.
- The recommended rule asks Jev about 32 / 40 / 152 cards per video (well-observed 60 / 117 / 175, all cards 254 / 623 / 721): 3.3 / 4.1 / 15.7 s of the service per video.
- The VLM (the core's Qwen3-VL-8B sidecar alone on an A100): 0.0212 s per question at 16 parallel inside the core (no hop), load 90.55 s. Zero-shot: PE-Core-L 3.131 ms, SigLIP 2 so400m 2.977 ms per crop on an A100. Rules: 0.80 us per card.

## RecGen load per video (the recommended rule, cut 0.5)

| video | cards | well-observed | RecGen calls (routed and well-observed) | routed, any card | GPU s at X7's 8.2 s median | labels' estimate of complex cards |
|---|---|---|---|---|---|---|
| me340 | 254 | 60 | 38 | 138 | 312 | 112 |
| samsclub-a2 | 623 | 117 | 0 | 10 | 0 | 4 |
| walmart | 721 | 175 | 111 | 320 | 911 | 308 |

## Which primitive (the SIMPLE-labelled cards)

| picker | agrees with the label | ME340 / Sam's / Walmart | top confusions (label -> picked) |
|---|---|---|---|
| display model (cards) | 0.711 | 0.542 / 0.798 / 0.704 | plane -> box 21, box -> cylinder 15, box -> plane 11 |
| jev q5 | 0.628 | 0.438 / 0.697 / 0.662 | box -> plane 37, box -> open frame 23, plane -> open frame 11 |
| zero-shot pe-core-l masked | 0.821 | 0.729 / 0.939 / 0.718 | plane -> box 23, plane -> cylinder 5, box -> cylinder 5 |
| zero-shot siglip2-so400m masked | 0.702 | 0.646 / 0.788 / 0.620 | box -> cylinder 22, plane -> box 19, box -> plane 10 |

## Spend

1.56 USD upper bound (fast_report.instrument.PRICE (Modal list prices) x each ephemeral app's wall time (modal app list) for every container of it): ap-Xi5OzB9alMGmLdwgWXcPwn jev + embed smoke 0.14, ap-1ZRbWyiSU9hhigfzyWTiuw jev + embed 1.11, ap-HuRpnB4gBtMvnT7ygV1PUw qwen smoke 0.103, ap-4JDeG9ZLrs5xkioIBW2Va9 qwen 0.208. No Gemini. The cost ledger was not edited.

## Caveats and open issues

- Labels are the agent's (Claude), by looking at blind sheets (the outlined view and a wider one, a code only), not ground truth. 27 of 340 were unclear (a region across objects, a sliver, a floor line) and are left out; 23 of them on ME340, whose cards are often parts or fragments of machines and benches.
- Load-bearing labelling calls: full pet-food bags and shrink-wrapped packs are boxes (SIMPLE); shelf boards / lips and signs are planes; workbenches are furniture (COMPLEX, the user's list). If the 23 labelled bags were COMPLEX instead, Jev Q5 alone would drop 0.875 -> 0.834 (Walmart 0.885 -> 0.760), the pre-registered cascade (bag -> complex) would rise 0.815 -> 0.888 and the recommended one fall 0.904 -> 0.863: bags are a policy decision, so put 'bag' in the class list whichever way it goes.
- The recommended class list is POST HOC: the pre-registered list (commit 1b61ed1, before any label) minus 5 entries the labels showed to be shape-ambiguous (bag, tool box, tool tray, crate, display rack). Its 0.904 is optimistic; the held-out numbers are 0.815 for the pre-registered list and 0.875 for Jev alone. On the cards the post-hoc list decides, the class is right 0.975 vs Jev 0.920 (ME340 0.948 vs 0.793: Jev calls workbenches, hand tools and machine parts simple). Only 15 of the list's classes had labelled cards here (box, stacked boxes, pallet, shelf, rack, sign, spill, pipe, metal sheet, wooden board; machine, hand tool, power tool, cable, workbench); the other entries are the user's definition, untested.
- Per video n is 102-107 labelled; a difference under ~5 points is under ~5 cards; the bootstrap intervals are above.
- Jev-Omni's raw Q2 answers lean hard to 'a detailed 3D model' and move with the option order (raw 0.744 vs 0.591 by order): Q2 needs its two-number Platt calibration, whose intercept moved from -2.6 to -4.2 with the training videos' mix. Q5's raw probabilities need none (ECE 0.075), hence Q5 in the rule. The yes / no form ('a simple geometric shape?') is unusable (Jev says no to almost all).
- RecGen load is dominated by Walmart's wall of shoes: ~300 COMPLEX cards, ~100 of them well-observed -> ~100 generations per video (~15 min of one GPU at X7's 8.2 s median). A per-video cap or reuse of one generated model across same-type cards (X9) is needed whatever the router.
- The rule routes only well-observed cards (display_model.well_observed, the generators' own view gate): 352 of 1598 cards. If a single-view generator (SAM 3D from one image, TRELLIS) is adopted, drop the gate: Jev then answers every non-class card (~0.1 s each batched).
- Which primitive is a separate, weaker spot: the cards' display model agrees with the SIMPLE label 0.71 (planes drawn as boxes, boxes as cylinders), PE-Core-L zero-shot's shape group 0.82, Jev's Q5 argmax 0.63. Many plane / box disagreements are thin boards and lips, where a thin box and a slab look alike.
- Data: the r4-models-bench-001 warm calls (the only r4 runs with primitive fits); no r4b-* run existed; r4-physical-results has no fits.
- The VLM tier was measured, not assumed: the core's own Qwen3-VL-8B (same image and questions, letter log-probs) is worse than Jev alone (0.802-0.837) and deciding Jev's uncertain band (P 0.3-0.7, 13 % of cards) with it changes nothing (0.872 vs 0.875).
- Jev's one-question round trip (0.49 s) is mostly the hop: batch each cards version's questions (0.10 s a question in 64-question requests).
