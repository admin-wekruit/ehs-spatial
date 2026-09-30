# Click MVP (mvp/integrate): results

All numbers measured on 2 x A100-SXM4-80GB (MPS), ephemeral Modal runs, analysis time only (cold start apart). Physical values are at estimated scale (floor plane + assumed 1.6 m camera height); 'agreement' is with the delivered report, itself model-made and at estimated scale, not ground truth. Labels on clicks, boxes and judgements are agent-made.

## What this is

Branch `mvp/integrate` (worktree `/Users/adam/.codex/worktrees/panoptes-phase2-video-mvp-integrate`), from `fb/integrate`, with
`mvp/d-validate`, `mvp/c-viewer`, `mvp/a-cards` and `mvp/b-judge` merged in that order, then fixed in place. One command per
video runs the whole thing on 2 x A100-80GB (a first call after boot, a warm call, and a +5 s shifted-window call for
repeatability):

    cd /Users/adam/.codex/worktrees/panoptes-phase2-video-mvp-integrate
    PYTHONPATH=. /Users/adam/Desktop/Tesla/panoptes-platform/.venv/bin/python scripts/fast_report_bench.py \
        --out RUNS/mvp-integrate-<site>-NNN --sites <me340|samsclub-a2|walmart> --plan first,warm,shifted --mirror-max-mb 8 --no-gpu-eval

Final runs (round 5): `runs/mvp-integrate-me340-005`, `-samsclub-004`, `-walmart-004`. Scored together (one pooled k, one
judgement table) with D's harness into `runs/mvp-results/eval` (`fast_report_eval.py --mvp ... --labels ... --no-gpu`,
reusing D1's SAM 3 click references). Viewer checks and screenshots: `runs/mvp-results/viewer/<site>/`.

## Open the viewer

    cd /Users/adam/.codex/worktrees/panoptes-phase2-video-mvp-integrate
    PY=/Users/adam/Desktop/Tesla/panoptes-platform/.venv/bin/python; R=/Users/adam/Desktop/panoptes-public/research-notes/phase2/runs
    PYTHONPATH=. $PY -m fast_report.layers serve $R/mvp-integrate-me340-005/mirror --port 8793 \
        --replay $R/mvp-integrate-me340-005/mirror mvp-me340-e84efffd-1790666683 --as me340-demo --speed 4
    cd web && npx vite --host 127.0.0.1        # second terminal; proxies /fast to 8793
    open "http://127.0.0.1:5173/app.html#/live/me340-demo"

Sam's Club: `mvp-integrate-samsclub-004` / `mvp-samsclub-a2-d5e0c855-1790666685`; Walmart: `mvp-integrate-walmart-004` /
`mvp-walmart-c0761a2a-1790666653`. Click anything in the video: the card opens with what it is, its physical values
(each ± u, 'estimated' on every metre, 'not observed' / 'not measurable' with the reason) and its judgements. The mirrors keep
blobs under 8 MB only, so the full room and the splat stay on the Modal Volume; the replay leaves those two layers out.

## Main changes made while integrating (all general; no per-video case)

- Judge wired to A's cards v1 and v3 (B's stand-in removed), its rules in the process pool, VLM answers carried from the v1
  run to the v3 run.
- Identity: the decider's prompt now lists the options as letters (A's prompt listed none); both crops sent; asked for
  every object seen on >= 3 views (EHS classes first) on 336 px crops; replaces the free-text naming (spec 4.7), which
  held vLLM 34 s on Sam's Club; identity that lands after densify is merged into cards v3 as v4.
- Heights carry an `up` part (the walls' p90 plumb reading x horizontal distance): warm-vs-shifted height coverage
  72% -> 93% on ME340. Angles with a fit term over 15 deg are 'not measurable' (bulky objects read 70 ± 43 deg).
- k from repeatability written to `fast_report/calibration.json` (round 2, pooled): height 1.071, extent 1.345, position 1.024,
  angle 1.0; round 5 re-measured 1.028 / 1.0 / 1.001 / 1.0 with it applied (same clips: in-sample).
- J1 on goods off the floor uses the stack's own height (rack goods at 3.6-4.1 m had failed a floor-stack rule, 20 false
  FAILs on Sam's Club); J2 ignores principal axes at >= 45 deg; J4 shows the base height when its PASS rests on it; J5 scans
  from the nearest free path point within the path's band.
- Pick maps end at a shot cut (viewer, eval and core agree); objects v1 boxes in the process pool; the identity pass never
  raises (a raise ended round 3's runs while densify ran and faulted both GPUs for every later call).
- Eval: blob cards read; still-camera shots are not Sim3-aligned (ME340 shot 0: 2 cm path, scale 0.80); part/whole pairs
  (a 2 m stack vs one box of it) are not counted as repeats (8 / 23 / 18 pairs).

## Reading the numbers

- Everything metric is at estimated scale (floor plane + an assumed 1.6 m camera height). 'Agreement' is with the
  delivered reports, themselves model-made and at estimated scale. Sam's Club's delivered heights are known to sit
  0.7-1.5 m off (L2), which is where the MVP agrees least (top median 0.24 m).
- Identity 'agreement' is exact-name agreement with the delivered report's product-level names after a 0.5 m match
  ('stacked boxes' vs 'paper towel pack' counts as wrong). The agent audit (40 cards a video on their best view) is the
  direct measure of 'does it recognise what things are'.
- Judgements: no FAIL on any of the three videos; PASS only where value - u clears the threshold (J1, J4). J5 never
  decides: the spec's gap rule turns every PASS into NEEDS_REVIEW while the far side of a footprint is unseen (79-99% of
  cards, forward walks), and person paths from mono depth run through footprints. The 60 audited rows found no hazard
  present; the audit shows 0 false PASS but cannot measure FAIL precision (n = 0).
- Timing misses: objects v1's Volume commit queues behind events and the full room (sent at fb + ~0.5 s, written 1-2 s
  later); the first SAM 3D model on ME340 (168 s) and Sam's Club (160 s) waits for the facts (spec 7 moved it after cards v3);
  pick v2's fetch in the browser waits 0.7-2.6 s behind the densify burst (v1 at load: 89-107 ms).
- Earlier rounds: round 1 `mvp-integrate-me340-001`; round 2 `-me340-002`, `-samsclub-001`, `-walmart-001` (k fitted here);
  round 3 `-me340-003`, `-samsclub-002`, `-walmart-002` failed (the identity-pass TypeError); round 4 `-me340-004`,
  `-samsclub-003`, `-walmart-003`.


## Analysis time to each layer (s from the MP4 bytes in the container; warm call, first call after boot in brackets)

| | ME340 | Sam's Club | Walmart |
|---|---|---|---|
| first 3D (cameras + light room) | 17.7 (22.6) ✓ | 15.2 (18.6) ✓ | 15.1 (18.3) ✓ |
| objects v1 | 36.3 (37.8) ✗ | 27.2 (28.6) ✗ | 20.2 (29.3) ✓ |
| pick v1 | 39.9 (42.1) ✓ | 27.2 (30.5) ✓ | 22.4 (37.9) ✓ |
| cards v1 | 42.0 (44.1) ✓ | 33.8 (35.1) ✓ | 25.4 (37.9) ✓ |
| judgements v1 | 43.9 (44.1) ✓ | 35.4 (36.6) ✓ | 27.1 (37.9) ✓ |
| cards v2 (identity) | 52.1 (59.4) ✓ | 62.5 (77.6) ✓ | 53.9 (64.2) ✓ |
| judgements v2 (VLM) | 54.2 (61.2) ✓ | 78.6 (96.5) ✓ | 28.7 (62.4) ✓ |
| pick v2 (densified) | 73.8 (76.7) ✓ | 56.1 (60.7) ✓ | 43.0 (46.8) ✓ |
| cards v3 | 75.5 (78.6) ✓ | 60.1 (63.9) ✓ | 45.6 (50.5) ✓ |
| first SAM 3D model | 168.3 (175.5) ✗ | 159.7 (173.3) ✗ | 71.0 (92.4) ✓ |
| splat preview | 223.3 (226.3) ✓ | 205.9 (208.0) ✓ | 190.8 (196.7) ✓ |
| cold start (s, not analysis) | 133.6 | 141.8 | 128.7 |
| GPU peaks GiB (warm; first) | 62.3/56.7; 61.7/56.4 | 65.2/55.0; 64.2/55.0 | 62.1/56.4; 62.6/56.2 |

✓/✗: CLICK-MVP-SPEC section 7 target (cameras/people/events: fb/integrate ± 1 s; pick ≤ outlines + 1 s; cards v1 ≤ objects + 10 s; judgements v1 ≤ objects + 12 s; cards/judgements v2 ≤ objects + 60 s; pick v2 / cards v3 ≤ objects + 40 s; first model ≤ 120 s; splat ≤ 230 s).

## Click accuracy (SAM 3 references on X1's 60 eval frames a video, seed-0 clicks; D's rule: IoU ≥ 0.3 or covers ≥ 50%)

| | ME340 | Sam's Club | Walmart |
|---|---|---|---|
| baseline pick v1 (from outlines): object clicks correct | 47% | 40% | 31% |
| baseline pick v1 (from outlines): correct, IoU ≥ 0.3 only | 24% | 30% | 27% |
| baseline pick v1 (from outlines): person clicks correct | 0% | 0% | 0% |
| baseline pick v1 (from outlines): background false hits | 1% | 3% | 5% |
| pick v1: object clicks correct | 52% | 41% | 32% |
| pick v1: correct, IoU ≥ 0.3 only | 29% | 31% | 28% |
| pick v1: person clicks correct | 94% | 43% | 0% |
| pick v1: background false hits | 1% | 2% | 3% |
| pick v2: object clicks correct | 53% | 44% | 39% |
| pick v2: correct, IoU ≥ 0.3 only | 33% | 35% | 33% |
| pick v2: person clicks correct | 93% | 43% | 0% |
| pick v2: background false hits | 1% | 2% | 6% |

## Identity (1:1 match ≤ 0.5 m to the delivered report's clear objects, after the camera Sim3; agreement with its model-made names)

| | ME340 | Sam's Club | Walmart |
|---|---|---|---|
| matched | 41 | 61 | 51 |
| name agreement, MVP | 15% | 28% | 43% |
| agent audit: name right / right or close (n decided) | 26% / 74% (38) | 65% / 98% (40) | 85% / 98% (40) |
| audit, route 'sam3 vote': right / right or close (n) | 20% / 73% (15) | 40% / 100% (15) | 88% / 100% (24) |
| audit, route 'vlm options': right / right or close (n) | 30% / 74% (23) | 80% / 96% (25) | 81% / 94% (16) |
| name agreement, fb/integrate (D1) | 8% | 46% | 25% |

## Physical info vs the delivered report (median / p90 |Δ| m, same definitions, after the camera Sim3; coverage = share with |Δ| ≤ u without its scale part)

| | ME340 | Sam's Club | Walmart |
|---|---|---|---|
| top | 0.05 / 0.20 (cov 74%, n 39); fb 0.14 / 0.87 | 0.24 / 0.75 (cov 28%, n 61); fb 0.10 / 0.68 | 0.12 / 0.34 (cov 74%, n 46); fb 0.11 / 0.54 |
| base | 0.06 / 0.25 (cov 76%, n 41); fb 0.16 / 1.04 | 0.15 / 0.72 (cov 52%, n 61); fb 0.18 / 0.72 | 0.08 / 0.26 (cov 80%, n 51); fb 0.19 / 0.43 |
| long_side | 0.59 / 2.26 (cov 31%, n 16); fb 0.56 / 3.12 | — / — (cov —, n 0); fb 0.53 / 3.32 | 0.53 / 0.94 (cov 25%, n 4); fb 0.39 / 5.42 |
| short_side | 0.16 / 1.24 (cov 31%, n 16); fb 0.28 / 1.81 | — / — (cov —, n 0); fb 0.13 / 1.35 | 0.08 / 0.16 (cov 50%, n 4); fb 0.21 / 0.73 |
| boxes > 3 m, all / shown non-large | 3% / 1% (fb 29% / 15%) | 1% / 0% (fb 11% / 2%) | 2% / 0% (fb 5% / 1%) |
| known verticals shown / outside ±u | 0 / 0 | 0 / 0 | 0 / 0 |

## Repeatability (warm call vs the +5 s shifted window, matched through the Sim3 between their cameras)

| | ME340 | Sam's Club | Walmart |
|---|---|---|---|
| median / p90 |Δ| top (m) | 0.025 / 0.168 (n 112) | 0.115 / 0.412 (n 388) | 0.026 / 0.205 (n 541) |
| median / p90 |Δ| height (m) | 0.035 / 0.251 (n 106) | 0.032 / 0.393 (n 383) | 0.013 / 0.132 (n 532) |
| median / p90 |Δ| long_side (m) | 0.096 / 0.336 (n 28) | 0.472 / 0.528 (n 2) | 0.029 / 0.434 (n 17) |
| median / p90 |Δ| position (m) | 0.090 / 0.510 (n 114) | 0.170 / 0.424 (n 397) | 0.055 / 0.271 (n 566) |
| coverage k=1, height | 93% (n 226) | 82% (n 785) | 94% (n 1107) |
| coverage k=1, extent | 91% (n 162) | 88% (n 387) | 91% (n 566) |
| coverage k=1, position | 74% (n 114) | 94% (n 397) | 90% (n 566) |
| coverage k=1, angle | 100% (n 5) | 100% (n 65) | — (n 0) |

k_family (smallest k ≥ 1 with coverage ≥ 0.9, pooled over the videos): {"pooled": {"height": {"k": 1.028, "n": 2118, "coverage_before": 0.8947, "coverage_after": 0.9004}, "extent": {"k": 1.0, "n": 1115, "coverage_before": 0.8996, "coverage_after": 0.9004}, "position": {"k": 1.001, "n": 1077, "coverage_before": 0.8997, "coverage_after": 0.9006}, "angle": {"k": 1.0, "n": 70, "coverage_before": 1.0, "coverage_after": 1.0}}, "source": "warm vs shifted window"}

## Judgements

| | ME340 | Sam's Club | Walmart |
|---|---|---|---|
| FAIL rows | 0 | 0 | 0 |
| NEEDS_REVIEW rows | 17 | 143 | 50 |
| PASS rows | 8 | 121 | 0 |
| NO_DATA rows | 16 | 186 | 10 |

Agent-labelled accuracy (agent-made labels (contact sheets); not ground truth): 60 rows labelled; false PASS total 0.

| check | rows | labelled | verdict shares | FAIL precision (n) | false PASS |
|---|---|---|---|---|---|
| J1 | 174 | 18 | NEEDS_REVIEW 30%, PASS 70% | — (0) | 0 |
| J2 | 174 | 0 | NO_DATA 100% | — (0) | 0 |
| J3a | 39 | 9 | NEEDS_REVIEW 33%, NO_DATA 67% | — (0) | 0 |
| J3b | 1 | 1 | NEEDS_REVIEW 100% | — (0) | 0 |
| J4 | 24 | 16 | NEEDS_REVIEW 42%, NO_DATA 25%, PASS 33% | — (0) | 0 |
| J5 | 109 | 8 | NEEDS_REVIEW 100% | — (0) | 0 |
| J8 | 30 | 8 | NEEDS_REVIEW 80%, NO_DATA 20% | — (0) | 0 |

## Cards

| | ME340 | Sam's Club | Walmart |
|---|---|---|---|
| cards / fields / contract violations | 269 / 2358 / 0 | 645 / 5607 / 0 | 730 / 6480 / 0 |

## Viewer (headless Chromium, recorded layer times)

| | ME340 | Sam's Club | Walmart |
|---|---|---|---|
| click → card p50 / p95 ms (200 clicks) | 5.4 / 26.1 | 6.4 / 29.6 | 7.0 / 32.5 |
| pick decode ms, v1 at load / v2 after densify (fetch + inflate + index) | 106.8 / 702.1 | 89.1 / 1939.2 | 104.2 / 2554.1 |
| clicks on an entity / unknown region (200 random) | 84 / 116 | 64 / 136 | 62 / 138 |

Screenshots:
- ME340: `me340/mvp-me340-e84efffd-1790666683-click-1.png`: aimed obj-1-211 → card 'workbench NEEDS REVIEW', verdict chips NEEDS REVIEW
- ME340: `me340/mvp-me340-e84efffd-1790666683-click-2.png`: aimed obj-1-190 → card 'workbench NEEDS REVIEW', verdict chips NEEDS REVIEW
- ME340: `me340/mvp-me340-e84efffd-1790666683-click-3.png`: aimed obj-1-82 → card 'power tool PASS', verdict chips PASS
- ME340: `me340/mvp-me340-e84efffd-1790666683-click-4.png`: aimed obj-1-104 → card 'workbench no checks', verdict chips no checks
- ME340: `me340/mvp-me340-e84efffd-1790666683-click-5.png`: aimed miss → card 'Unknown region', verdict chips none
- ME340: `me340/mvp-me340-e84efffd-1790666683-click-6.png`: aimed person:1-person-0 → card 'Person 1-person-0 NEEDS REVIEW', verdict chips NEEDS REVIEW, NO DATA
- Sam's Club: `samsclub-a2/mvp-samsclub-a2-d5e0c855-1790666685-click-1.png`: aimed obj-0-110 → card 'stacked boxes NEEDS REVIEW', verdict chips NEEDS REVIEW, NO DATA
- Sam's Club: `samsclub-a2/mvp-samsclub-a2-d5e0c855-1790666685-click-2.png`: aimed obj-0-80 → card 'stacked boxes NEEDS REVIEW', verdict chips NEEDS REVIEW, NO DATA
- Sam's Club: `samsclub-a2/mvp-samsclub-a2-d5e0c855-1790666685-click-3.png`: aimed obj-0-804 → card 'cable NEEDS REVIEW', verdict chips NEEDS REVIEW
- Sam's Club: `samsclub-a2/mvp-samsclub-a2-d5e0c855-1790666685-click-4.png`: aimed obj-1-389 → card 'stacked boxes NEEDS REVIEW', verdict chips NEEDS REVIEW, PASS, NO DATA
- Sam's Club: `samsclub-a2/mvp-samsclub-a2-d5e0c855-1790666685-click-5.png`: aimed miss → card 'Unknown region', verdict chips none
- Sam's Club: `samsclub-a2/mvp-samsclub-a2-d5e0c855-1790666685-click-6.png`: aimed person:0-person-0 → card 'Person 0-person-0 NEEDS REVIEW', verdict chips NEEDS REVIEW, NO DATA
- Walmart: `walmart/mvp-walmart-c0761a2a-1790666653-click-1.png`: aimed obj-0-124 → card 'display rack NEEDS REVIEW', verdict chips NEEDS REVIEW
- Walmart: `walmart/mvp-walmart-c0761a2a-1790666653-click-2.png`: aimed obj-1-483 → card 'shelf NEEDS REVIEW', verdict chips NEEDS REVIEW
- Walmart: `walmart/mvp-walmart-c0761a2a-1790666653-click-3.png`: aimed obj-1-496 → card 'box NEEDS REVIEW', verdict chips NEEDS REVIEW
- Walmart: `walmart/mvp-walmart-c0761a2a-1790666653-click-4.png`: aimed obj-1-474 → card 'shelf no checks', verdict chips no checks
- Walmart: `walmart/mvp-walmart-c0761a2a-1790666653-click-5.png`: aimed miss → card 'Unknown region', verdict chips none
- Walmart: `walmart/mvp-walmart-c0761a2a-1790666653-click-6.png`: aimed person:1-person-0 → card 'Person 1-person-0 NEEDS REVIEW', verdict chips NEEDS REVIEW, NO DATA

## Acceptance (CLICK-MVP-SPEC 10, D's harness; ✓ pass, ✗ fail, — not judged)

| | ME340 | Sam's Club | Walmart |
|---|---|---|---|
| L1_no_implausible_number | ✓ | ✓ | ✓ |
| L1_shown_non_large_over_3m | ✓ | ✓ | ✓ |
| cards_complete | ✓ | ✓ | ✓ |
| clicks_background_false_hits | ✓ | ✓ | ✓ |
| clicks_v2_beats_v1 | ✓ | ✓ | ✓ |
| known_verticals_within_u | — | — | — |
| memory | ✓ | ✓ | ✓ |
| times | ✗ (missed: first SAM 3D model, objects v1) | ✗ (missed: events, first SAM 3D model, objects v1, people) | ✓ |
| repeat_coverage_k_le_2 (all videos) | ✓ {"k": {"height": 1.028, "extent": 1.0, "position": 1.001, "angle": 1.0}} | | |
| decider_ece_le_0.10 (all videos) | ✗ {"decider": "qwen deployed (B)", "ece_p_yes_cv": {"q1": 0.0948, "q4": 0.1827, "q5": 0.2548}, "false_alarm_only": {"q2": 0, "q3": 0}} | | |
| judgements_zero_false_pass (all videos) | ✓ {"false_pass": 0} | | |

## Experiments plugged in, and not

- **X1 frame rates (plugged, via A)**: densify: SAM 3 on every 5 fps keyframe after objects v1 (pick v2 / cards v3); X1's robust per-view extents and int64 lift codes. Measured here as pick v2 vs pick v1 and cards v3 timing.
- **X6 windows + object time (plugged, via A)**: fast_report/timeline.py is identical to fx/x6-windows-time's head: the association rule (3 sigma, sigma = sqrt(0.04^2 + (0.05 z)^2), or box IoU >= 0.2) for the fragment merge, and the free-space test for 'disappeared'. X6's per-window geometry (b) is not plugged: its verification found the stitch-chain gate blocks most comparisons and the '0 false changes' in-sample on few judgements.
- **X7 per-object models (plugged, via A)**: parametric primitives only from >= 2 views >= 15 deg apart that pass the held-out gate (A4).
- **X8 decider (plugged, Qwen; Jev-Omni not)**: Qwen3-VL-8B option-letter log-probs (B's vlm.options) decide identity and ask the judgement questions; Platt calibration on set d. Jev-Omni is not swapped in: spec 5.5 needs Brier -0.02 and ECE <= 0.05 against calibrated Qwen; cross-fitted it is Brier 0.175 vs 0.138 and ECE 0.129 (D), and it needs ~22 GiB that GPU 0 does not have beside the integrated core (60-66 GiB peaks; X8's verification).
- **X9 same-object reuse (not plugged)**: its verification: the outline gain measures lift vs no lift, not reuse (blocker); the totals leave out 17-25 s of 15 fps DA3 depth; worse look-alike identity on Walmart and Sam's Club. Densify (X1) stays.
- **X10 class-agnostic discovery (not plugged)**: verified +0.16 own-look recall (0.69 -> 0.85) for the recommended stack, but it adds 16.5-22.5 s of GPU work on the object keyframes (before objects, or on top of densify after them), two more resident models (OWLv2-L, SAM 2.1-L), and its masks' precision was never audited; its names would be generic words ('object', 'item') that the decider would have to name. Not without slowing the first layers; a later coverage pass.
- **X11 local bundle adjustment (not plugged)**: no spot confirmed on both frame sets; BA alone not significant; the held-out outline IoU got worse (0.555 -> 0.501). The u model keeps the pose term instead.
- **X12 motion fps**: people at 5 fps (unchanged).

Spend: Modal billing (complete hours, 2026-09-29 UTC) for the 13 integrate apps: $18.52 so far; list-price upper bound over the containers' lives $22.98 (5 rounds x 1-3 videos, boot included; the final round alone $5.36). Viewer checks and scoring ran locally ($0). The cost ledger was not edited.
