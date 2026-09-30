## Analysis time (s from the MP4 bytes in the container to the Volume commit)

| s from MP4 bytes in, warm (first call) | ME340 | Sam's Club | Walmart | round 1 warm |
|---|---|---|---|---|
| first 3D | 18.1 (22.6) | 14.4 (18.1) | 18.2 (22.4) | 17.7 / 15.2 / 15.1 |
| objects v1 | 36.6 (40.2) | 21.1 (30.5) | 23.2 (32.8) | 36.3 / 27.2 / 20.2 |
| pick v1 | 41.0 (43.3) | 23.1 (38.7) | 26.5 (32.8) | 39.9 / 27.2 / 22.4 |
| cards v1 | 43.4 (46.3) | 30.8 (46.0) | 30.9 (36.8) | 42.0 / 33.8 / 25.4 |
| identity complete | 68.2 (66.2) | 53.0 (70.7) | 57.2 (66.5) | 60.3 / 88.3 / 69.5 |
| densify names | 91.6 (94.2) | 75.1 (85.6) | 73.0 (75.3) | — |
| final judgements | 102.2 (106.5) | 83.7 (107.0) | 80.7 (83.0) | 82.2 / 117.3 / 55.9 |
| pick v2 | 66.6 (71.8) | 43.0 (64.9) | 41.4 (48.0) | 73.8 / 56.1 / 43.0 |
| cards v3 | 70.2 (74.1) | 48.2 (66.6) | 45.0 (53.8) | 75.5 / 60.1 / 45.6 |
| first SAM 3D model | — (165.3) | 136.4 (414.8) | 70.5 (86.2) | 168.3 / 159.7 / 71.0 |
| splat preview | 222.7 (222.6) | 222.5 (222.7) | 218.4 (220.7) | 223.3 / 205.9 / 190.8 |
| GPU peaks GiB gpu0/gpu1, warm (first) | 63.2/53.2 (61.2/53.0) | 67.6/59.0 (65.3/59.0) | 64.2/56.3 (61.5/56.3) | — |
| stages over 72 GiB | none | none | none | — |

Variant 'discover (X10 words, runs 004)' (same code, one flag), warm (first call):

| | ME340 | Sam's Club | Walmart |
|---|---|---|---|
| objects v1 | 38.6 (41.4) | 48.4 (35.8) | 25.4 (25.3) |
| cards v1 | 50.9 (49.9) | 75.5 (63.3) | 28.7 (38.2) |
| identity complete | 79.2 (83.9) | 116.6 (98.6) | 71.1 (73.8) |
| densify names | 94.0 (97.3) | 165.5 (125.0) | 110.6 (94.6) |
| final judgements | 101.7 (108.0) | 183.4 (159.4) | 110.6 (101.2) |
| pick v2 | 72.0 (81.3) | 102.8 (90.2) | 43.6 (49.6) |
| cards v3 | 75.0 (83.9) | 111.9 (108.9) | 48.9 (55.1) |

## Card checks (every cards version of every call)

| | ME340 | Sam's Club | Walmart |
|---|---|---|---|
| final judgement rows on a check the card's final class does not apply (R1), warm (first) | 0 (0) | 0 (0) | 0 (0) |
| final cards apply_name would change (R1), warm (first) | 0 (0) | 0 (0) | 0 (0) |
| card contract violations over every cards version, warm (first) | 0 (0) | 0 (0) | 0 (0) |
| cards checked (all versions), warm (first) | 916 (916) | 2398 (2398) | 2639 (2630) |

## Identity on the held-out items (runs/mvp2-identity-final-001, agent-labelled blind before this branch)

| call | ME340 | Sam's Club | Walmart | all |
|---|---|---|---|---|
| warm: right / close / wrong (n) | 45 / 9 / 8 (62) | 25 / 0 / 3 (28) | 26 / 0 / 3 (29) | 96 / 9 / 14 (119) |
| warm: right, right or close | 0.73, 0.87 | 0.89, 0.89 | 0.90, 0.90 | 0.81, 0.88 |
| first: right / close / wrong (n) | 40 / 2 / 13 (55) | 24 / 0 / 4 (28) | 27 / 0 / 2 (29) | 91 / 2 / 19 (112) |
| first: right, right or close | 0.73, 0.76 | 0.86, 0.86 | 0.93, 0.93 | 0.81, 0.83 |
| round 1 on the same objects: right, right or close | 0.44, 0.67 (n 54) | 0.89, 0.93 (n 28) | 0.72, 0.83 (n 29) | 0.63, 0.78 (n 111) |

Hazard-class names on these items (warm): 7 shown, 7 right or same family, 0 on items labelled unclear; 4 held back, 1 of them true.

## Physical values against metric ground truth (runs/mvp2-integrate-gt-*, 3 indoor scenes)

Median |error| (a) as delivered (assumed 1.6 m camera height) -> (b) at the true camera height; n = shown values with GT depth on >= 50 % of their mask; cov = share within +-u (as delivered).

| field | ARKit 47333932 | ARKit 42445448 | TUM fr1 room (held out) |
|---|---|---|---|
| top_above_floor | 5.5 -> 3.2 cm (n 142, cov 0.98) | 22.1 -> 1.9 cm (n 54, cov 0.96) | 7.5 -> 2.8 cm (n 256, cov 0.98) |
| base_above_floor | 4.2 -> 3.5 cm (n 149, cov 0.98) | 18.6 -> 1.1 cm (n 57, cov 1.00) | 7.5 -> 2.7 cm (n 269, cov 0.97) |
| height | 3.3 -> 3.0 cm (n 85, cov 0.91) | 6.1 -> 2.3 cm (n 39, cov 0.82) | 2.2 -> 2.0 cm (n 129, cov 0.91) |
| width | 4.8 -> 3.3 cm (n 100, cov 0.93) | 9.2 -> 1.6 cm (n 45, cov 0.91) | 3.3 -> 2.4 cm (n 176, cov 0.95) |
| depth | 2.8 -> 3.9 cm (n 24, cov 0.92) | 4.9 -> 1.5 cm (n 34, cov 0.88) | 3.9 -> 2.9 cm (n 64, cov 0.91) |
| position_xy | 24.5 -> 12.4 cm (n 149, cov 0.99) | 42.8 -> 6.2 cm (n 57, cov 1.00) | 18.3 -> 7.5 cm (n 269, cov 0.99) |
| planar_slope_deg | — | — | 0.9 -> 0.9 deg (n 13, cov 1.00) |
| principal_axis_tilt_deg | 2.2 -> 2.2 deg (n 3, cov 1.00) | 0.7 -> 0.7 deg (n 3, cov 1.00) | 2.2 -> 2.2 deg (n 10, cov 1.00) |

u coverage by family and view-set state (k_geo fitted on ARKit only; TUM held out), as delivered:

| family / view sets | ARKit 47333932 | ARKit 42445448 | TUM (held out) |
|---|---|---|---|
| height / sets | 0.99 (n 229) | 0.99 (n 67) | 0.98 (n 399) |
| height / one_set | 0.96 (n 49) | 1.00 (n 39) | 0.95 (n 108) |
| extent / sets | 0.94 (n 177) | 0.85 (n 82) | 0.94 (n 302) |
| extent / one_set | 0.80 (n 30) | 0.94 (n 32) | 0.87 (n 63) |
| position / sets | 0.99 (n 118) | 1.00 (n 34) | 1.00 (n 205) |
| position / one_set | 0.97 (n 29) | 1.00 (n 21) | 0.95 (n 62) |
| angle / sets | 1.00 (n 3) | 1.00 (n 2) | 1.00 (n 23) |

Bounds holding against GT (warm calls): at least/arkit47 28/29; at least/arkit42 11/11; at least/tum 42/49; at most/arkit47 90/98; at most/arkit42 28/30; at most/tum 220/236.

## Physical values against the delivered reports (same-object pairs; agreement, not truth), warm call

| | ME340 | Sam's Club | Walmart |
|---|---|---|---|
| same-object pairs | 13 | 40 | 14 |
| top m: signed median / median / p90 abs | -0.02 / 0.047 / 0.304 (cov 0.833, n 12) | -0.013 / 0.044 / 0.284 (cov 0.85, n 40) | -0.101 / 0.101 / 0.141 (cov 0.929, n 14) |
| base m: signed median / median / p90 abs | 0.008 / 0.04 / 0.354 (cov 0.846, n 13) | -0.01 / 0.029 / 0.79 (cov 0.75, n 40) | -0.077 / 0.095 / 0.133 (cov 0.929, n 14) |
| people: feet read (n), median signed m, share within u of 0 | 2, 0.106, 0.5 | 17, 0.284, 0.647 | 0, None, None |
| person masks measured as pictures | 1 | 285 | 26 |
| long objects: visible length shown / short side not measurable | 37 / 5 | 109 / 25 | 63 / 11 |

## Click audit (independent of SAM 3: random pixel on a random frame, the viewer's pick rule; held-out seed 2, agent-labelled)

| warm call | clicks | correct | wrong | miss | background | background-hit | resolved right | picked precision |
|---|---|---|---|---|---|---|---|---|
| ME340 | 60 | 21 | 0 | 9 | 27 | 3 | 0.80 | 0.88 |
| Sam's Club | 60 | 28 | 0 | 19 | 13 | 0 | 0.68 | 1.00 |
| Walmart | 60 | 22 | 1 | 21 | 16 | 0 | 0.63 | 0.96 |

| person clicks (inside SAM 3 person references, seed 11) | real people | real person opens the person | pictures of people | pictures opened as a person |
|---|---|---|---|---|
| ME340 | 60 | 0.98 | 1 | 0 |
| Sam's Club | 53 | 0.89 | 37 | 4 |
| Walmart | 3 | 0.67 | 13 | 0 |

## Judgements (final judgements patch; audit labels agent-made, blind to verdicts)

| call | objects with a check | verdicts (rows) | PASS right / audited (+ unverifiable) | FAIL right / audited (+ unverifiable) |
|---|---|---|---|---|
| me340 first | 33 / 260 (13%) | {"NO_DATA": 12, "PASS": 13, "NEEDS_REVIEW": 20, "FAIL": 1} | 13 / 13 (+0) | 1 / 1 (+0) |
| me340 warm | 25 / 254 (10%) | {"NO_DATA": 11, "PASS": 8, "NEEDS_REVIEW": 16, "FAIL": 2} | 8 / 8 (+0) | 2 / 2 (+0) |
| samsclub first | 190 / 622 (30%) | {"NEEDS_REVIEW": 165, "PASS": 141, "NO_DATA": 96, "FAIL": 1} | 139 / 139 (+2) | 0 / 1 (+0) |
| samsclub warm | 187 / 622 (30%) | {"NEEDS_REVIEW": 153, "PASS": 142, "NO_DATA": 95} | 139 / 139 (+3) | 0 / 0 (+0) |
| walmart first | 63 / 717 (9%) | {"PASS": 49, "NEEDS_REVIEW": 17, "NO_DATA": 11} | 49 / 49 (+0) | 0 / 0 (+0) |
| walmart warm | 70 / 720 (10%) | {"PASS": 57, "NO_DATA": 11, "NEEDS_REVIEW": 18} | 57 / 57 (+0) | 0 / 0 (+0) |

| check, warm call | ME340 | Sam's Club | Walmart |
|---|---|---|---|
| J1 | — | 137 rows, decided 57% ({"NEEDS_REVIEW": 56, "PASS": 78, "NO_DATA": 3}); PASS 75/75+3, FAIL 0/0+0 | 3 rows, decided 33% ({"NEEDS_REVIEW": 2, "PASS": 1}); PASS 1/1+0, FAIL 0/0+0 |
| J2 | — | 137 rows, decided 0% ({"NEEDS_REVIEW": 64, "NO_DATA": 73}); PASS 0/0+0, FAIL 0/0+0 | 3 rows, decided 0% ({"NEEDS_REVIEW": 1, "NO_DATA": 2}); PASS 0/0+0, FAIL 0/0+0 |
| J3a | 6 rows, decided 0% ({"NO_DATA": 5, "NEEDS_REVIEW": 1}); PASS 0/0+0, FAIL 0/0+0 | 16 rows, decided 0% ({"NEEDS_REVIEW": 4, "NO_DATA": 12}); PASS 0/0+0, FAIL 0/0+0 | 7 rows, decided 0% ({"NO_DATA": 7}); PASS 0/0+0, FAIL 0/0+0 |
| J4 | 11 rows, decided 36% ({"NO_DATA": 2, "NEEDS_REVIEW": 5, "FAIL": 2, "PASS": 2}); PASS 2/2+0, FAIL 2/2+0 | 3 rows, decided 67% ({"NEEDS_REVIEW": 1, "PASS": 2}); PASS 2/2+0, FAIL 0/0+0 | — |
| J5 | 14 rows, decided 43% ({"PASS": 6, "NEEDS_REVIEW": 8}); PASS 6/6+0, FAIL 0/0+0 | 82 rows, decided 76% ({"NEEDS_REVIEW": 15, "PASS": 62, "NO_DATA": 5}); PASS 62/62+0, FAIL 0/0+0 | 70 rows, decided 80% ({"PASS": 56, "NO_DATA": 2, "NEEDS_REVIEW": 12}); PASS 56/56+0, FAIL 0/0+0 |
| J8 | 6 rows, decided 0% ({"NO_DATA": 4, "NEEDS_REVIEW": 2}); PASS 0/0+0, FAIL 0/0+0 | 15 rows, decided 0% ({"NO_DATA": 2, "NEEDS_REVIEW": 13}); PASS 0/0+0, FAIL 0/0+0 | 3 rows, decided 0% ({"NEEDS_REVIEW": 3}); PASS 0/0+0, FAIL 0/0+0 |

## Viewer (headless Chromium, the recording replayed at recorded speed; screenshots in viewer/)

| report | click -> card p50 / p95 ms | pick decode ms | aimed clicks (screenshot: what the card says) |
|---|---|---|---|
| mvp-me340-e84efffd-1790690775 | 7.0 / 34.8 | 88 | mvp-me340-e84efffd-1790690775-click-1.png: aimed obj-1-482 -> 'power cord FAIL' [FAIL, FAIL]; mvp-me340-e84efffd-1790690775-click-2.png: aimed obj-1-249 -> 'cnc machine NEEDS REVIEW' [NEEDS REVIEW, NEEDS REVIEW]; mvp-me340-e84efffd-1790690775-click-3.png: aimed obj-1-110 -> 'tool racks no checks' [no checks]; mvp-me340-e84efffd-1790690775-click-4.png: aimed miss -> 'Unknown region' []; mvp-me340-e84efffd-1790690775-click-5.png: aimed person:1-person-0 -> 'Person 1-person-0 NEEDS REVIEW' [NEEDS REVIEW, NO DATA, NO DATA, NEEDS REVIEW, NEEDS REVIEW, NO DATA, NO DATA, NEEDS REVIEW] |
| mvp-samsclub-a2-d5e0c855-1790691934 | 6.6 / 31.8 | 82 | mvp-samsclub-a2-d5e0c855-1790691934-click-1.png: aimed obj-0-610 -> 'paper towels NEEDS REVIEW' [NEEDS REVIEW, NEEDS REVIEW]; mvp-samsclub-a2-d5e0c855-1790691934-click-2.png: aimed obj-0-110 -> 'pallet of paper towels NEEDS REVIEW' [NEEDS REVIEW, NEEDS REVIEW, NEEDS REVIEW, PASS]; mvp-samsclub-a2-d5e0c855-1790691934-click-3.png: aimed obj-1-389 -> 'pallet of merchandise NEEDS REVIEW' [NEEDS REVIEW, NEEDS REVIEW, NEEDS REVIEW, PASS]; mvp-samsclub-a2-d5e0c855-1790691934-click-4.png: aimed miss -> 'Unknown region' []; mvp-samsclub-a2-d5e0c855-1790691934-click-5.png: aimed person:0-person-0 -> 'Person 0-person-0 NEEDS REVIEW' [NEEDS REVIEW, NO DATA, NO DATA, NEEDS REVIEW, NEEDS REVIEW, NO DATA, NO DATA, NEEDS REVIEW] |
| mvp-walmart-c0761a2a-1790692552 | 7.1 / 36.3 | 156 | mvp-walmart-c0761a2a-1790692552-click-1.png: aimed obj-1-496 -> 'pallet NEEDS REVIEW' [NEEDS REVIEW, NEEDS REVIEW, NEEDS REVIEW, NEEDS REVIEW]; mvp-walmart-c0761a2a-1790692552-click-2.png: aimed obj-1-1030 -> 'plastic pallet NEEDS REVIEW' [NEEDS REVIEW, PASS, NO DATA, NEEDS REVIEW]; mvp-walmart-c0761a2a-1790692552-click-3.png: aimed obj-1-435 -> 'bag of dog food PASS' [PASS, PASS]; mvp-walmart-c0761a2a-1790692552-click-4.png: aimed miss -> 'Unknown region' []; mvp-walmart-c0761a2a-1790692552-click-5.png: aimed person:1-person-3 -> 'Person 1-person-3 NEEDS REVIEW' [NEEDS REVIEW, NEEDS REVIEW, NO DATA, NEEDS REVIEW] |
