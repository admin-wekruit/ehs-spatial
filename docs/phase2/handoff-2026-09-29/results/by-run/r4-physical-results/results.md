# r4 physical: results

Runs: r4-physical-bench-001 (3 videos, first + warm call, display off, no VLM, SAM 3 words only), r4-physical-gt-002 (6 GT calls, same options). Code: branch r4/physical.

## Completeness: every card carries every physical field (a number, a bound, or a status with its reason)

Contract violations: 0 on every cards version of all 12 bench reports and 12 GT reports (mvp2-integrate-*-007 final cards: 56 / 136 / 64 on me340 / samsclub / walmart, person cards 0 % complete).

Depth, object cards (warm call): a number / 'not observed' with a visible lower bound (depth.visible, 'at least') / status only: me340 50 / 122 / 81 of 253; samsclub 5 / 337 / 280 of 622; walmart 32 / 412 / 277 of 721. The table counts the visible bound as a status.

Cells: share of cards with a number (value or bound) / share with a status and its reason (not observed, not measurable).

### mvp-me340-e84efffd-1790708539 (warm call, cards v3)

| cards | n | complete | position_xy | top_above_floor | base_above_floor | height | width | depth | principal_axis_tilt_deg | planar_slope_deg |
|---|---|---|---|---|---|---|---|---|---|---|
| object | 253 | 1.00 | 1.00 / 0.00 | 1.00 / 0.00 | 1.00 / 0.00 | 0.91 / 0.09 | 0.94 / 0.06 | 0.20 / 0.80 | 0.10 / 0.90 | 0.03 / 0.97 |
| object: unidentified | 15 | 1.00 | 1.00 / 0.00 | 1.00 / 0.00 | 1.00 / 0.00 | 0.87 / 0.13 | 0.87 / 0.13 | 0.00 / 1.00 | 0.20 / 0.80 | 0.13 / 0.87 |
| object: detector word | 238 | 1.00 | 1.00 / 0.00 | 1.00 / 0.00 | 1.00 / 0.00 | 0.91 / 0.09 | 0.94 / 0.06 | 0.21 / 0.79 | 0.10 / 0.90 | 0.03 / 0.97 |
| person | 6 | 1.00 | 0.83 / 0.17 | 0.83 / 0.17 | 0.17 / 0.83 | 0.17 / 0.83 | 0.00 / 1.00 | 0.00 / 1.00 | 0.00 / 1.00 | 0.00 / 1.00 |
| not a person | 1 | 1.00 | 0.00 / 1.00 | 0.00 / 1.00 | 0.00 / 1.00 | 0.00 / 1.00 | 0.00 / 1.00 | 0.00 / 1.00 | 0.00 / 1.00 | 0.00 / 1.00 |

### mvp-samsclub-a2-d5e0c855-1790708652 (warm call, cards v3)

| cards | n | complete | position_xy | top_above_floor | base_above_floor | height | width | depth | principal_axis_tilt_deg | planar_slope_deg |
|---|---|---|---|---|---|---|---|---|---|---|
| object | 622 | 1.00 | 1.00 / 0.00 | 1.00 / 0.00 | 1.00 / 0.00 | 0.98 / 0.02 | 0.92 / 0.08 | 0.01 / 0.99 | 0.09 / 0.91 | 0.21 / 0.79 |
| object: detector word | 619 | 1.00 | 1.00 / 0.00 | 1.00 / 0.00 | 1.00 / 0.00 | 0.98 / 0.02 | 0.92 / 0.08 | 0.01 / 0.99 | 0.09 / 0.91 | 0.21 / 0.79 |
| object: unidentified | 3 | 1.00 | 1.00 / 0.00 | 1.00 / 0.00 | 1.00 / 0.00 | 1.00 / 0.00 | 1.00 / 0.00 | 0.00 / 1.00 | 0.33 / 0.67 | 0.00 / 1.00 |
| person | 16 | 1.00 | 0.94 / 0.06 | 0.94 / 0.06 | 0.25 / 0.75 | 0.25 / 0.75 | 0.00 / 1.00 | 0.00 / 1.00 | 0.00 / 1.00 | 0.00 / 1.00 |
| not a person | 1 | 1.00 | 0.00 / 1.00 | 0.00 / 1.00 | 0.00 / 1.00 | 0.00 / 1.00 | 0.00 / 1.00 | 0.00 / 1.00 | 0.00 / 1.00 | 0.00 / 1.00 |

### mvp-walmart-c0761a2a-1790708758 (warm call, cards v3)

| cards | n | complete | position_xy | top_above_floor | base_above_floor | height | width | depth | principal_axis_tilt_deg | planar_slope_deg |
|---|---|---|---|---|---|---|---|---|---|---|
| object | 721 | 1.00 | 1.00 / 0.00 | 1.00 / 0.00 | 1.00 / 0.00 | 0.98 / 0.02 | 0.96 / 0.04 | 0.04 / 0.96 | 0.00 / 1.00 | 0.00 / 1.00 |
| object: detector word | 717 | 1.00 | 1.00 / 0.00 | 1.00 / 0.00 | 1.00 / 0.00 | 0.98 / 0.02 | 0.96 / 0.04 | 0.04 / 0.96 | 0.00 / 1.00 | 0.00 / 1.00 |
| object: unidentified | 4 | 1.00 | 1.00 / 0.00 | 1.00 / 0.00 | 1.00 / 0.00 | 1.00 / 0.00 | 1.00 / 0.00 | 0.00 / 1.00 | 0.00 / 1.00 | 0.00 / 1.00 |
| person | 7 | 1.00 | 0.86 / 0.14 | 0.86 / 0.14 | 0.00 / 1.00 | 0.00 / 1.00 | 0.00 / 1.00 | 0.00 / 1.00 | 0.00 / 1.00 | 0.00 / 1.00 |
| not a person | 1 | 1.00 | 0.00 / 1.00 | 0.00 / 1.00 | 0.00 / 1.00 | 0.00 / 1.00 | 0.00 / 1.00 | 0.00 / 1.00 | 0.00 / 1.00 | 0.00 / 1.00 |

## Accuracy against ground truth (r4-physical-gt-002; mvp2-integrate-gt-003 beside it in gt-mvp2-integrate-gt-003.md)

| sequence | field | n | median abs err (a) assumed 1.6 m | (b) true height | cov (a) | cov (b) | u median |
|---|---|---|---|---|---|---|---|
| arkit42 | base_above_floor | 58 | 18.7 cm | 1.2 cm | 1.00 | 0.97 | 24.3 cm |
| arkit42 | depth | 35 | 4.6 cm | 1.6 cm | 0.91 | 0.89 | 8.6 cm |
| arkit42 | height | 39 | 6.1 cm | 2.3 cm | 0.82 | 0.92 | 9.3 cm |
| arkit42 | position_xy | 58 | 43.0 cm | 6.2 cm | 1.00 | 1.00 | 63.9 cm |
| arkit42 | principal_axis_tilt_deg | 3 | 0.8 deg | 0.8 deg | 1.00 | 1.00 | 13.0 deg |
| arkit42 | top_above_floor | 55 | 21.8 cm | 1.9 cm | 0.96 | 0.96 | 26.4 cm |
| arkit42 | width | 44 | 9.3 cm | 1.7 cm | 0.91 | 0.89 | 14.9 cm |
| arkit47 | base_above_floor | 150 | 4.0 cm | 3.5 cm | 0.98 | 0.91 | 20.7 cm |
| arkit47 | depth | 25 | 2.8 cm | 3.9 cm | 0.88 | 0.88 | 15.4 cm |
| arkit47 | height | 86 | 3.3 cm | 3.0 cm | 0.91 | 0.88 | 10.1 cm |
| arkit47 | position_xy | 150 | 24.4 cm | 12.9 cm | 0.99 | 0.84 | 82.1 cm |
| arkit47 | principal_axis_tilt_deg | 3 | 2.2 deg | 2.2 deg | 1.00 | 1.00 | 14.5 deg |
| arkit47 | top_above_floor | 143 | 5.5 cm | 3.3 cm | 0.98 | 0.85 | 26.8 cm |
| arkit47 | width | 102 | 4.8 cm | 3.5 cm | 0.94 | 0.90 | 11.9 cm |
| tum | base_above_floor | 270 | 7.7 cm | 2.7 cm | 0.98 | 0.91 | 21.6 cm |
| tum | depth | 70 | 3.7 cm | 2.6 cm | 0.90 | 0.86 | 16.1 cm |
| tum | height | 124 | 2.4 cm | 2.2 cm | 0.90 | 0.88 | 8.9 cm |
| tum | planar_slope_deg | 15 | 1.0 deg | 1.0 deg | 1.00 | 1.00 | 3.5 deg |
| tum | position_xy | 270 | 18.3 cm | 7.9 cm | 0.99 | 0.97 | 80.0 cm |
| tum | principal_axis_tilt_deg | 9 | 2.1 deg | 2.1 deg | 1.00 | 1.00 | 11.0 deg |
| tum | top_above_floor | 259 | 7.5 cm | 2.8 cm | 0.98 | 0.91 | 23.5 cm |
| tum | width | 169 | 3.1 cm | 2.3 cm | 0.92 | 0.89 | 12.5 cm |

True camera height over the floor (the delivered scale assumes 1.6 m): arkit42 1.23-1.23 m; arkit47 1.48-1.50 m; tum 1.42-1.52 m

| bounds (warm calls, GT on >= 50 % of the mask) | n | hold (a) | hold (b) |
|---|---|---|---|
| arkit42 depth at least | 15 | 13 | 14 |
| arkit42 depth at most | 5 | 4 | 4 |
| arkit42 height at least | 3 | 3 | 3 |
| arkit42 height at most | 14 | 14 | 14 |
| arkit42 top_above_floor at least | 3 | 3 | 3 |
| arkit42 visible_length at least | 3 | 3 | 3 |
| arkit42 width at least | 5 | 5 | 5 |
| arkit42 width at most | 8 | 7 | 6 |
| arkit47 depth at least | 77 | 72 | 70 |
| arkit47 depth at most | 10 | 9 | 9 |
| arkit47 height at least | 14 | 13 | 12 |
| arkit47 height at most | 47 | 44 | 43 |
| arkit47 top_above_floor at least | 7 | 7 | 6 |
| arkit47 visible_length at least | 1 | 1 | 1 |
| arkit47 width at least | 8 | 8 | 8 |
| arkit47 width at most | 30 | 27 | 26 |
| tum depth at least | 117 | 103 | 102 |
| tum depth at most | 17 | 16 | 16 |
| tum height at least | 16 | 15 | 15 |
| tum height at most | 120 | 115 | 114 |
| tum top_above_floor at least | 11 | 10 | 9 |
| tum visible_length at least | 7 | 3 | 3 |
| tum width at least | 17 | 11 | 10 |
| tum width at most | 79 | 69 | 68 |

## Audit by eye: 30 random cards per video (warm call; sheets and labels in audit-*/)

| video | cards looked at | plausible | implausible | unclear |
|---|---|---|---|---|
| me340 | 30 | 26 | 0 | 4 |
| samsclub | 30 | 25 | 0 | 5 |
| walmart | 30 | 29 | 0 | 1 |
