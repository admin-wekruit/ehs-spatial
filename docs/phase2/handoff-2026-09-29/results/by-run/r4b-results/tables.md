## 1. Types (warm call; first call in brackets)

| | ME340 | Sam's Club | Walmart |
|---|---|---|---|
| object cards | 376 (370) | 917 (923) | 896 (906) |
| typed by family (a specific name or '<family> (type only)') | 95% (98%) | 100% (100%) | 98% (98%) |
| with a specific name | 20% (48%) | 53% (61%) | 64% (62%) |
| shape type only (no family) | 5% (2%) | 0% (0%) | 2% (2%) |
| round 3 warm: typed by family / specific name | 95% / 98% | 100% / 100% | 99% / 99% |
| fresh card audit: type right / close / wrong / unclear (n) | 14 / 5 / 3 / 8 (30) | 21 / 2 / 3 / 4 (30) | 29 / 1 / 0 / 0 (30) |
| held-out items (round 2's labels), warm: type (family) right / n | 26/51 = 0.51 | 26/26 = 1.00 | 24/29 = 0.83 |
| held-out items, warm: specific names right / right-or-close / named | 3 / 6 / 10 | 14 / 15 / 15 | 13 / 13 / 15 |
| held-out items (round 2's labels), first: type (family) right / n | 29/47 = 0.62 | 26/26 = 1.00 | 24/29 = 0.83 |
| held-out items, first: specific names right / right-or-close / named | 9 / 18 / 23 | 16 / 17 / 17 | 12 / 12 / 14 |
| held-out items (round 2's labels), round 3 warm: type (family) right / n | 24/55 = 0.44 | 28/28 = 1.00 | 24/29 = 0.83 |
| held-out items, round 3 warm: specific names right / right-or-close / named | 20 / 30 / 51 | 24 / 26 / 28 | 21 / 24 / 25 |

## 2. Segmentation and clicks (warm call)

| | ME340 | Sam's Club | Walmart |
|---|---|---|---|
| object cards with a pick region (clickable) | 100% | 100% | 100% |
| fresh card audit: outline right / partial / wrong | 17 / 8 / 1 | 28 / 0 / 0 | 28 / 0 / 2 |
| 60 random clicks: right card / nothing / wrong / background right / background hit | 27 / 4 / 0 / 28 / 1 | 35 / 11 / 0 / 13 / 1 | 33 / 6 / 1 / 20 / 0 |
| the same clicks on round 3's warm call: right now and nothing then / miss now and something then | 6 / 0 | 18 / 0 | 18 / 0 |
| clicks on a thing that opened the right card | 27/31 = 0.87 | 35/46 = 0.76 | 33/40 = 0.82 |
| delivered objects (clean): covered, in pieces, wrong-merge cards; round 3 warm in brackets | 80/105, 7, 4 (53/105, 1, 3) | 79/91, 10, 4 (41/91, 5, 0) | 61/74, 2, 0 (38/74, 3, 0) |
| delivered objects (all): covered, in pieces, wrong-merge cards; round 3 warm in brackets | 125/161, 20, 14 (88/161, 13, 11) | 129/152, 31, 9 (75/152, 24, 5) | 134/157, 26, 7 (95/157, 22, 4) |
| delivered objects (models): covered, in pieces, wrong-merge cards; round 3 warm in brackets | 22/22, 4, 1 (20/22, 7, 2) | 66/67, 21, 5 (53/67, 22, 2) | 39/40, 9, 1 (33/40, 8, 0) |

## 3. Physical (warm call)

| | ME340 | Sam's Club | Walmart |
|---|---|---|---|
| object cards complete (every required field: value +-u, bound, or status + reason) | 100% | 100% | 100% |
| person cards complete | 100% | 100% | 100% |
| contract violations, every cards version (both calls) | 0 | 0 | 0 |
| object cards with a number or bound: position_xy | 100% | 100% | 100% |
| object cards with a number or bound: top_above_floor | 100% | 100% | 100% |
| object cards with a number or bound: base_above_floor | 100% | 100% | 100% |
| object cards with a number or bound: height | 89% | 95% | 97% |
| object cards with a number or bound: width | 91% | 89% | 94% |
| object cards with a number or bound: depth | 15% | 2% | 6% |
| object cards with a number or bound: principal_axis_tilt_deg | 12% | 10% | 0% |
| fresh card audit: physical plausible / implausible / unclear | 24 / 1 / 5 | 29 / 1 / 0 | 28 / 2 / 0 |

### Physical against ground truth (TUM fr1 room, ARKitScenes; warm calls)

r4b-gt-001:

| sequence | field | n | median abs err (a) assumed 1.6 m | (b) true height | cov (a) | cov (b) | u median |
|---|---|---|---|---|---|---|---|
| tum | base_above_floor | 326 | 8.3 cm | 3.0 cm | 0.96 | 0.87 | 21.8 cm |
| tum | depth | 66 | 5.5 cm | 4.5 cm | 0.94 | 0.86 | 18.6 cm |
| tum | height | 148 | 2.8 cm | 2.5 cm | 0.89 | 0.86 | 11.3 cm |
| tum | planar_slope_deg | 14 | 1.4 deg | 1.4 deg | 1.00 | 1.00 | 3.2 deg |
| tum | position_xy | 326 | 19.4 cm | 8.4 cm | 0.98 | 0.97 | 86.9 cm |
| tum | principal_axis_tilt_deg | 12 | 1.0 deg | 1.0 deg | 1.00 | 1.00 | 10.0 deg |
| tum | top_above_floor | 308 | 7.4 cm | 3.1 cm | 0.97 | 0.89 | 24.2 cm |
| tum | width | 205 | 2.9 cm | 2.9 cm | 0.94 | 0.91 | 14.4 cm |

True camera height over the floor (the delivered scale assumes 1.6 m): tum 1.42-1.52 m

| bounds (warm calls, GT on >= 50 % of the mask) | n | hold (a) | hold (b) |
|---|---|---|---|
| tum depth at least | 152 | 132 | 129 |
| tum depth at most | 28 | 27 | 27 |
| tum height at least | 16 | 15 | 15 |
| tum height at most | 150 | 140 | 139 |
| tum top_above_floor at least | 18 | 18 | 15 |
| tum visible_length at least | 19 | 12 | 12 |
| tum width at least | 11 | 10 | 10 |
| tum width at most | 98 | 81 | 74 |

r4b-gt-002:

| sequence | field | n | median abs err (a) assumed 1.6 m | (b) true height | cov (a) | cov (b) | u median |
|---|---|---|---|---|---|---|---|
| arkit42 | base_above_floor | 64 | 15.8 cm | 1.4 cm | 0.98 | 0.94 | 23.2 cm |
| arkit42 | depth | 35 | 5.6 cm | 2.0 cm | 0.89 | 0.83 | 10.1 cm |
| arkit42 | height | 42 | 7.0 cm | 2.1 cm | 0.90 | 0.95 | 11.9 cm |
| arkit42 | position_xy | 64 | 43.7 cm | 6.4 cm | 0.98 | 1.00 | 68.7 cm |
| arkit42 | principal_axis_tilt_deg | 4 | 1.2 deg | 1.2 deg | 1.00 | 1.00 | 12.4 deg |
| arkit42 | top_above_floor | 61 | 21.1 cm | 1.6 cm | 0.93 | 0.97 | 26.2 cm |
| arkit42 | width | 43 | 9.0 cm | 2.7 cm | 0.95 | 0.93 | 15.5 cm |
| arkit47 | base_above_floor | 178 | 4.9 cm | 3.4 cm | 0.97 | 0.88 | 21.6 cm |
| arkit47 | depth | 24 | 3.1 cm | 3.6 cm | 0.88 | 0.88 | 16.8 cm |
| arkit47 | height | 97 | 3.8 cm | 3.0 cm | 0.91 | 0.89 | 11.5 cm |
| arkit47 | planar_slope_deg | 4 | 0.6 deg | 0.6 deg | 1.00 | 1.00 | 3.8 deg |
| arkit47 | position_xy | 178 | 24.9 cm | 12.8 cm | 0.99 | 0.84 | 88.2 cm |
| arkit47 | principal_axis_tilt_deg | 3 | 3.2 deg | 3.2 deg | 1.00 | 1.00 | 9.0 deg |
| arkit47 | top_above_floor | 160 | 5.7 cm | 4.2 cm | 0.97 | 0.81 | 26.9 cm |
| arkit47 | width | 111 | 4.5 cm | 3.7 cm | 0.93 | 0.86 | 12.2 cm |

True camera height over the floor (the delivered scale assumes 1.6 m): arkit42 1.23-1.23 m; arkit47 1.48-1.50 m

| bounds (warm calls, GT on >= 50 % of the mask) | n | hold (a) | hold (b) |
|---|---|---|---|
| arkit42 depth at least | 12 | 11 | 11 |
| arkit42 depth at most | 15 | 15 | 14 |
| arkit42 height at least | 5 | 5 | 5 |
| arkit42 height at most | 14 | 14 | 14 |
| arkit42 top_above_floor at least | 3 | 3 | 3 |
| arkit42 visible_length at least | 5 | 4 | 4 |
| arkit42 width at least | 4 | 4 | 4 |
| arkit42 width at most | 13 | 12 | 11 |
| arkit47 depth at least | 94 | 91 | 90 |
| arkit47 depth at most | 16 | 16 | 16 |
| arkit47 height at least | 20 | 18 | 16 |
| arkit47 height at most | 51 | 49 | 48 |
| arkit47 top_above_floor at least | 18 | 18 | 14 |
| arkit47 visible_length at least | 10 | 8 | 8 |
| arkit47 width at least | 22 | 20 | 18 |
| arkit47 width at most | 36 | 29 | 26 |

## 4. Display models (warm call)

| | ME340 | Sam's Club | Walmart |
|---|---|---|---|
| object cards with a display model | 100% | 100% | 100% |
| model kinds | box 317, plane 33, cylinder 20, open frame 6 | box 818, cylinder 42, plane 33, open frame 24 | box 781, plane 55, cylinder 31, open frame 29 |
| SAM 3D accepted meshes / eligible / attempted | 1 / 68 / 30 | 5 / 167 / 30 | 4 / 234 / 30 |
| fresh card audit: model plausible / implausible / absent / unclear | 26 / 3 / 0 / 1 | 19 / 9 / 0 / 2 | 19 / 11 / 0 / 0 |

## Times (s from the MP4 bytes in the container to the Volume commit; warm call, first call after boot in brackets)

| | ME340 | Sam's Club | Walmart |
|---|---|---|---|
| first 3D | 18.0 (23.7) | 15.3 (15.8) | 13.1 (16.2) |
| objects v1 | 32.2 (36.5) | 24.7 (25.3) | 21.8 (25.1) |
| cards v1 | 43.1 (49.4) | 30.1 (36.4) | 31.2 (38.6) |
| types (first pass) | 60.3 (68.4) | 65.1 (70.9) | 60.3 (64.8) |
| objects v3 | 74.9 (81.1) | 54.7 (62.5) | 62.5 (68.4) |
| cards v3 | 83.5 (89.9) | 74.8 (81.3) | 83.0 (89.6) |
| types (densify pass) | 97.8 (104.4) | 87.5 (97.2) | 93.1 (99.2) |
| first SAM 3D model | 174.8 (133.8) | 103.7 (120.3) | 139.6 (154.5) |
| models final | 181.2 (188.7) | 178.7 (189.8) | 179.2 (191.3) |
| splat preview | 223.3 (229.5) | 222.6 (222.5) | 222.7 (222.7) |
| call end | 230.2 (237.2) | 228.5 (245.6) | 226.7 (227.5) |
| cold start (not analysis), s | 181.63 | 157.25 | 171.85 |

## GPU peaks (GiB, max over stages, GPU 0 / GPU 1; stages over 72 GiB)

| | ME340 | Sam's Club | Walmart |
|---|---|---|---|
| warm | 70.0 / 58.1; over 72: none | 70.2 / 65.7; over 72: none | 64.0 / 61.9; over 72: none |
| first | 67.7 / 58.9; over 72: none | 68.8 / 65.5; over 72: none | 63.8 / 61.8; over 72: none |

## VLM requests per call (warm; first in brackets)

| | ME340 | Sam's Club | Walmart |
|---|---|---|---|
| naming (cluster medoids) | 48 (67) | 32 (32) | 19 (18) |
| naming escalated | 0 (0) | 0 (0) | 0 (0) |
| scene vocabulary (one request) | 1 (1) | 1 (1) | 1 (1) |
| event captions (windows) | 2 (2) | 2 (2) | 2 (2) |
| hazard / judge | 0 (0) | 0 (0) | 0 (0) |
| qwen decider (identity_vlm) | 0 (0) | 0 (0) | 0 (0) |

## Spend (Modal list-price upper bound over each container's life)

| | ME340 | Sam's Club | Walmart |
|---|---|---|---|
| bench, $ | 1.77 | 2.15 | 1.41 |
