# Click MVP validation

Agreement with the delivered reports (themselves at estimated scale), not ground truth. Labels are agent-made.

| row | me340 | samsclub-a2 | walmart |
|---|---|---|---|
| source of the physical rows | object_cards | object_cards | object_cards |
| pick v1: object clicks correct / unknown / wrong (n) | 51.5% / 44.0% / 4.5% (11512) | 40.5% / 48.1% / 11.4% (13064) | 31.7% / 56.9% / 11.4% (8258) |
| pick v1: object clicks correct by IoU >= 0.3 alone | 29.2% | 31.4% | 28.1% |
| pick v1: correct on segmented / projected pick frames | 52.4% (3546) / 52.4% (7766) | 42.2% (3468) / 41.8% (9158) | 35.1% (3692) / 31.1% (4264) |
| pick v1: person clicks correct (n) | 94.2% (120) | 42.9% (42) | 0.0% (18) |
| pick v1: background false hits (<= 10%) | 0.7% | 2.3% | 3.3% |
| pick v2: object clicks correct / unknown / wrong (n) | 52.8% / 41.8% / 5.4% (11512) | 44.3% / 44.0% / 11.7% (13064) | 39.1% / 47.8% / 13.1% (8258) |
| pick v2: object clicks correct by IoU >= 0.3 alone | 33.0% | 34.7% | 32.6% |
| pick v2: correct on segmented / projected pick frames | 53.7% (11312) / None (0) | 45.9% (12626) / None (0) | 40.6% (7956) / None (0) |
| pick v2: person clicks correct (n) | 93.3% (120) | 42.9% (42) | 0.0% (18) |
| pick v2: background false hits (<= 10%) | 1.3% | 2.0% | 6.3% |
| click audit: auto rule agrees with the agent | — | — | — |
| click audit, agent 'same object' only: spec rule / IoU >= 0.3 / IoU >= 0.2 agree | — | — | — |
| L1 audit: flagged boxes truly inflated (inflated / size right / unclear) | — | — | — |
| L1 audit: 30 largest unflagged boxes inflated (inflated / size right / unclear) | — | — | — |
| identity: matched / delivered clear; agreement | 41/57; 14.6% | 61/98; 27.9% | 51/81; 43.1% |
| identity: SAM 3 word on the same pairs | — | — | — |
| vs delivered: top median / p90 abs delta m (coverage; n) | 0.052 / 0.202 (74.4%; 39) | 0.242 / 0.751 (27.9%; 61) | 0.123 / 0.343 (73.9%; 46) |
| vs delivered: base median / p90 abs delta m (coverage; n) | 0.061 / 0.253 (75.6%; 41) | 0.152 / 0.719 (52.5%; 61) | 0.080 / 0.262 (80.4%; 51) |
| vs delivered: long_side median / p90 abs delta m (coverage; n) | 0.589 / 2.261 (31.2%; 16) | None / None (no u; 0) | 0.526 / 0.938 (25.0%; 4) |
| vs delivered: short_side median / p90 abs delta m (coverage; n) | 0.156 / 1.241 (31.2%; 16) | None / None (no u; 0) | 0.084 / 0.160 (50.0%; 4) |
| scale: ours / delivered | 0.990 | 1.124 | 1.006 |
| L1: longest side > 3 m, all boxes | 2.7% of 262; p90 2.077 m, max 5.894 m | 1.4% of 623; p90 1.607 m, max 5.369 m | 1.8% of 720; p90 0.945 m, max 5.816 m |
| L1: shown non-large boxes > 3 m (<= 5%) | 0.5% of 190 | 0.3% of 594 | 0.0% of 639 |
| L1: implausible sizes flagged | 20 | 10 | 6 |
| known verticals: shown of n / misses | 0 of 2 / 0 | 0 of 3 / 0 | 0 of 2 / 0 |
| repeat (shifted call 2): matched; median abs delta top / long side / position m | 114 of [262, 226]; 0.025 / 0.096 / 0.090 | 397 of [623, 563]; 0.115 / 0.472 / 0.170 | 566 of [720, 721]; 0.026 / 0.029 / 0.055 |
| repeat (shifted call 2): coverage at k=1 height / extent / position / angle | 92.9% / 90.7% / 73.7% / 100.0% | 81.7% / 88.1% / 94.2% / 100.0% | 94.3% / 91.0% / 90.3% / no u |
| time: cameras, s (target) | 17.713 (fb +- 1 s) ✓ | 15.182 (fb +- 1 s) ✓ | 15.115 (fb +- 1 s) ✓ |
| time: people, s (target) | 19.673 (fb +- 1 s) ✓ | 20.065 (fb +- 1 s) ✗ | 15.115 (fb +- 1 s) ✓ |
| time: events, s (target) | 26.186 (fb +- 1 s) ✓ | 24.714 (fb +- 1 s) ✗ | 12.618 (fb +- 1 s) ✓ |
| time: objects v1, s (target) | 36.289 (lift <= fb + 0.5 s; SAM 3 done -> objects <= fb + 0.5 s) ✗ | 27.164 (lift <= fb + 0.5 s; SAM 3 done -> objects <= fb + 0.5 s) ✗ | 20.2 (lift <= fb + 0.5 s; SAM 3 done -> objects <= fb + 0.5 s) ✓ |
| time: pick v1, s (target) | 39.889 (40.889) ✓ | 27.164 (28.164) ✓ | 22.359 (23.359) ✓ |
| time: cards v1, s (target) | 42.018 (46.289) ✓ | 33.765 (37.164) ✓ | 25.363 (30.2) ✓ |
| time: judgements v1, s (target) | 43.911 (48.289) ✓ | 35.373 (39.164) ✓ | 27.092 (32.2) ✓ |
| time: cards v2, s (target) | 52.144 (96.289) ✓ | 62.504 (87.164) ✓ | 53.888 (80.2) ✓ |
| time: judgements v2, s (target) | 54.166 (96.289) ✓ | 78.565 (87.164) ✓ | 28.671 (80.2) ✓ |
| time: pick v2, s (target) | 73.781 (76.289) ✓ | 56.091 (67.164) ✓ | 43.022 (60.2) ✓ |
| time: cards v3, s (target) | 75.526 (76.289) ✓ | 60.098 (67.164) ✓ | 45.649 (60.2) ✓ |
| time: first SAM 3D model, s (target) | 168.293 (120.0) ✗ | 159.717 (120.0) ✗ | 71.05 (120.0) ✓ |
| time: splat preview, s (target) | 223.331 (230.0) ✓ | 205.892 (230.0) ✓ | 190.797 (230.0) ✓ |
| GPU peaks GiB (flags) | [62.31, 56.69] (0) | [65.16, 55.0] (0) | [62.07, 56.4] (0) |

## Decider on X8 set d (Platt per question, cross-fitted by source clip; agent labels)

| decider | n | Brier raw → cal | ECE raw → cal | AUROC raw → cal | said yes raw → cal |
|---|---|---|---|---|---|
| qwen | 154 | 0.2058 → 0.1378 | 0.2117 → 0.055 | 0.8706 → 0.8321 | 0.0909 → 0.2338 |
| jev | 154 | 0.1767 → 0.1747 | 0.1276 → 0.129 | 0.7384 → 0.7186 | 0.0974 → 0.1299 |
| gemini-stated-p | 154 | 0.1389 → 0.1477 | 0.0836 → 0.1207 | 0.8462 → 0.8088 | 0.3312 → 0.3312 |
| qwen deployed (B) | 153 | 0.1958 → 0.1387 | 0.1985 → 0.0635 | 0.8791 → 0.7908 | 0.085 → 0.2288 |

Jev-Omni swap test (spec 5.5): {"brier_qwen": 0.1378, "brier_jev": 0.1747, "ece_jev": 0.129, "jev_s_per_question_batched": 0.0632, "swap": false, "rule": "spec 5.5: Brier lower by >= 0.02, ECE <= 0.05, <= 0.2 s a question, all cross-fitted on set d"}

k_family (warm vs shifted window): {"height": {"k": 1.028, "n": 2118, "coverage_before": 0.8947, "coverage_after": 0.9004}, "extent": {"k": 1.0, "n": 1115, "coverage_before": 0.8996, "coverage_after": 0.9004}, "position": {"k": 1.001, "n": 1077, "coverage_before": 0.8997, "coverage_after": 0.9006}, "angle": {"k": 1.0, "n": 70, "coverage_before": 1.0, "coverage_after": 1.0}}

## Judgements (agent labels)

| check | rows | shares | FAIL precision (n) | false PASS |
|---|---|---|---|---|
| J1 | 174 | {'NEEDS_REVIEW': 0.3046, 'PASS': 0.6954} | None (0) | 0 |
| J2 | 174 | {'NO_DATA': 1.0} | None (0) | 0 |
| J3a | 39 | {'NEEDS_REVIEW': 0.3333, 'NO_DATA': 0.6667} | None (0) | 0 |
| J3b | 1 | {'NEEDS_REVIEW': 1.0} | None (0) | 0 |
| J4 | 24 | {'NEEDS_REVIEW': 0.4167, 'NO_DATA': 0.25, 'PASS': 0.3333} | None (0) | 0 |
| J5 | 109 | {'NEEDS_REVIEW': 1.0} | None (0) | 0 |
| J8 | 30 | {'NEEDS_REVIEW': 0.8, 'NO_DATA': 0.2} | None (0) | 0 |

## Acceptance (spec 10; — = not judged by these runs)

| criterion | me340 | samsclub-a2 | walmart |
|---|---|---|---|
| times | ✗ | ✗ | ✓ |
| memory | ✓ | ✓ | ✓ |
| L1_shown_non_large_over_3m | ✓ | ✓ | ✓ |
| clicks_v2_beats_v1 | ✓ | ✓ | ✓ |
| clicks_background_false_hits | ✓ | ✓ | ✓ |
| known_verticals_within_u | — | — | — |
| cards_complete | ✓ | ✓ | ✓ |
| L1_no_implausible_number | ✓ | ✓ | ✓ |
| repeat_coverage_k_le_2 (all videos) | ✓ {"k": {"height": 1.028, "extent": 1.0, "position": 1.001, "angle": 1.0}, "source": "warm vs shifted window"} | | |
| decider_ece_le_0.10 (all videos) | ✗ {"decider": "qwen deployed (B)", "ece_p_yes_cv": {"q1": 0.0948, "q4": 0.1827, "q5": 0.2548}, "false_alarm_only": {"q2": 0, "q3": 0}} | | |
| judgements_zero_false_pass (all videos) | ✓ {"false_pass": 0} | | |
