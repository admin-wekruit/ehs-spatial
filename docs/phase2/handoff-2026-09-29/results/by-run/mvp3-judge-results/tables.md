## Judgement coverage per call (final judgements patch)

| call | objects with a check | objects with an overall PASS / FAIL | objects with any PASS / FAIL row | verdict rows | 'likely hazard' priority rows |
|---|---|---|---|---|---|
| me340 first | 27 / 261 (10.3%) | 12 / 261 (4.6%) | 13 / 261 (5.0%) | {"NO_DATA": 11, "PASS": 13, "NEEDS_REVIEW": 17} | 3 |
| me340 warm | 33 / 254 (13.0%) | 16 / 254 (6.3%) | 17 / 254 (6.7%) | {"NO_DATA": 12, "PASS": 17, "NEEDS_REVIEW": 18} | 3 |
| samsclub-a2 first | 386 / 620 (62.3%) | 42 / 620 (6.8%) | 105 / 620 (16.9%) | {"NEEDS_REVIEW": 412, "PASS": 111, "NO_DATA": 57} | 5 |
| samsclub-a2 warm | 380 / 621 (61.2%) | 43 / 621 (6.9%) | 103 / 621 (16.6%) | {"NEEDS_REVIEW": 402, "PASS": 109, "NO_DATA": 61} | 3 |
| walmart first | 97 / 719 (13.5%) | 46 / 719 (6.4%) | 57 / 719 (7.9%) | {"PASS": 61, "NEEDS_REVIEW": 58, "NO_DATA": 15} | 4 |
| walmart warm | 100 / 722 (13.9%) | 49 / 722 (6.8%) | 63 / 722 (8.7%) | {"PASS": 66, "NEEDS_REVIEW": 54, "NO_DATA": 16} | 0 |

| call | per check | objects with no row, by placement | FAIL rows |
|---|---|---|---|
| me340 first | {"J1": {"PASS": 5}, "J3a": {"NO_DATA": 5, "NEEDS_REVIEW": 1}, "J4": {"NO_DATA": 2, "NEEDS_REVIEW": 8, "PASS": 3}, "J5": {"NEEDS_REVIEW": 4, "PASS": 5}, "J8": {"NO_DATA": 4, "NEEDS_REVIEW": 2}, "J9": {"NEEDS_REVIEW": 2}} | {"off the floor, no walked path near": 100, "on the floor, no walked path near": 17, "not an object": 3, "off the floor, at a walked path": 114} | 0 |
| me340 warm | {"J1": {"PASS": 5}, "J3a": {"NO_DATA": 5, "NEEDS_REVIEW": 1}, "J4": {"NO_DATA": 3, "NEEDS_REVIEW": 9, "PASS": 4}, "J5": {"PASS": 8, "NEEDS_REVIEW": 4}, "J8": {"NO_DATA": 4, "NEEDS_REVIEW": 2}, "J9": {"NEEDS_REVIEW": 2}} | {"off the floor, no walked path near": 96, "on the floor, no walked path near": 15, "not an object": 1, "off the floor, at a walked path": 109} | 0 |
| samsclub-a2 first | {"J1": {"NEEDS_REVIEW": 313, "PASS": 63, "NO_DATA": 3}, "J2": {"NEEDS_REVIEW": 67, "NO_DATA": 33, "PASS": 1}, "J3a": {"NEEDS_REVIEW": 4, "NO_DATA": 12}, "J4": {"PASS": 1, "NEEDS_REVIEW": 1}, "J5": {"NEEDS_REVIEW": 10, "PASS": 46, "NO_DATA": 7}, "J8": {"NO_DATA": 2, "NEEDS_REVIEW": 13}, "J9": {"NEEDS_REVIEW": 4}} | {"off the floor, at a walked path": 72, "off the floor, no walked path near": 143, "on the floor, no walked path near": 17, "not an object": 2} | 0 |
| samsclub-a2 warm | {"J1": {"NEEDS_REVIEW": 306, "PASS": 60, "NO_DATA": 3}, "J2": {"NEEDS_REVIEW": 68, "NO_DATA": 35}, "J3a": {"NEEDS_REVIEW": 4, "NO_DATA": 12}, "J4": {"NEEDS_REVIEW": 1}, "J5": {"NEEDS_REVIEW": 6, "PASS": 49, "NO_DATA": 9}, "J8": {"NO_DATA": 2, "NEEDS_REVIEW": 13}, "J9": {"NEEDS_REVIEW": 4}} | {"off the floor, at a walked path": 74, "off the floor, no walked path near": 147, "on the floor, no walked path near": 19, "not an object": 1} | 0 |
| walmart first | {"J1": {"NEEDS_REVIEW": 42, "PASS": 11}, "J2": {"NO_DATA": 2}, "J3a": {"NO_DATA": 7}, "J5": {"PASS": 50, "NEEDS_REVIEW": 12, "NO_DATA": 6}, "J8": {"NEEDS_REVIEW": 3}, "J9": {"NEEDS_REVIEW": 1}} | {"off the floor, at a walked path": 471, "off the floor, no walked path near": 114, "not an object": 5, "on the floor, no walked path near": 32} | 0 |
| walmart warm | {"J1": {"NEEDS_REVIEW": 44, "PASS": 10}, "J2": {"NO_DATA": 3}, "J3a": {"NO_DATA": 7}, "J5": {"PASS": 56, "NEEDS_REVIEW": 6, "NO_DATA": 6}, "J8": {"NEEDS_REVIEW": 3}, "J9": {"NEEDS_REVIEW": 1}} | {"off the floor, at a walked path": 468, "off the floor, no walked path near": 118, "on the floor, no walked path near": 32, "not an object": 4} | 0 |

## Walked paths the judge used (D2), contract counts (D6)

| call | walked paths per shot | person cards: walked path used | 'needs review' sizes with u 0 | person moved / path with u >= max(value, 1 m) | card contract violations (all versions) | rows on a check the final class does not apply |
|---|---|---|---|---|---|---|
| me340 first | {"0": [], "1": ["camera", "person:1-person-0"]} | 1 of 5 | 0 | 0 | 0 | 0 |
| me340 warm | {"0": [], "1": ["camera", "person:1-person-0"]} | 1 of 5 | 0 | 0 | 0 | 0 |
| samsclub-a2 first | {"0": ["camera", "person:0-person-0"], "1": ["camera"]} | 1 of 15 | 0 | 0 | 0 | 0 |
| samsclub-a2 warm | {"0": ["camera", "person:0-person-0"], "1": ["camera"]} | 1 of 15 | 0 | 0 | 0 | 0 |
| walmart first | {"0": ["camera"], "1": ["camera", "person:1-person-5"]} | 1 of 6 | 0 | 0 | 0 | 0 |
| walmart warm | {"0": ["camera"], "1": ["camera", "person:1-person-5"]} | 1 of 6 | 0 | 0 | 0 | 0 |

## Analysis time (s from the MP4 bytes in the container), warm (first call)

| | ME340 | Sam's Club | Walmart |
|---|---|---|---|
| objects v1 | 32.1 (39.1) | 24.3 (25.1) | 20.3 (24.0) |
| cards v1 | 37.6 (45.4) | 34.5 (40.2) | 27.4 (33.7) |
| identity complete | 59.2 (79.3) | 68.2 (69.4) | 60.4 (58.5) |
| densify names | 82.4 (88.2) | 75.8 (78.0) | 79.2 (73.4) |
| final judgements | 99.0 (98.1) | 86.8 (106.9) | 87.0 (83.4) |
| cards v3 | 63.3 (74.9) | 51.1 (60.2) | 44.9 (53.0) |
| GPU peaks GiB [gpu0, gpu1] per stage max, warm (first) | [62.98, 52.54] ([61.96, 52.77]); over 72 GiB: none | [66.68, 59.34] ([65.89, 59.01]); over 72 GiB: none | [64.2, 56.46] ([59.87, 56.12]); over 72 GiB: none |
| Modal list-price upper bound, $ | 1.78 | 1.49 | 1.37 |
| cold start (boot, not analysis), s | 146.74 | 115.05 | 144.83 |

## Offline replay on round 2's run 007 cards (CPU; the run's own Gemini answers; no new questions)

| call | objects with a check | overall PASS / FAIL | verdict rows | FAIL rows |
|---|---|---|---|---|
| me340 first | 32 / 260 (12.3%) | 14 / 260 (5.4%) | {"NO_DATA": 12, "PASS": 16, "NEEDS_REVIEW": 19} | 0 |
| me340 warm | 28 / 254 (11.0%) | 14 / 254 (5.5%) | {"NO_DATA": 11, "PASS": 15, "NEEDS_REVIEW": 16} | 0 |
| samsclub-a2 first | 396 / 622 (63.7%) | 50 / 622 (8.0%) | {"NEEDS_REVIEW": 416, "PASS": 118, "NO_DATA": 60} | 0 |
| samsclub-a2 warm | 384 / 622 (61.7%) | 45 / 622 (7.2%) | {"NEEDS_REVIEW": 402, "PASS": 111, "NO_DATA": 58} | 0 |
| walmart first | 80 / 717 (11.2%) | 42 / 717 (5.9%) | {"PASS": 57, "NEEDS_REVIEW": 43, "NO_DATA": 15} | 0 |
| walmart warm | 83 / 720 (11.5%) | 50 / 720 (6.9%) | {"PASS": 62, "NO_DATA": 15, "NEEDS_REVIEW": 35} | 0 |

## Fresh audit (new seed, every sampled item looked at by eye; agent-labelled, blind to verdicts)

| call | PASS right / audited (+ unverifiable), of PASS items | FAIL right / audited (+ unverifiable), of FAIL items | per check |
|---|---|---|---|
| me340 first | 13 / 13 (+0), of 13 | 0 / 0 (+0), of 0 | {"J4": {"pass_audited": 3, "pass_right": 3}, "J1": {"pass_audited": 5, "pass_right": 5}, "J5": {"pass_audited": 5, "pass_right": 5}} |
| me340 warm | 16 / 16 (+0), of 16 | 0 / 0 (+0), of 0 | {"J5": {"pass_audited": 8, "pass_right": 8}, "J4": {"pass_audited": 4, "pass_right": 4}, "J1": {"pass_audited": 4, "pass_right": 4}} |
| samsclub first | 29 / 29 (+0), of 78 | 0 / 0 (+0), of 0 | {"J5": {"pass_audited": 17, "pass_right": 17}, "J1": {"pass_audited": 11, "pass_right": 11}, "J2": {"pass_audited": 1, "pass_right": 1}} |
| samsclub warm | 30 / 30 (+0), of 76 | 0 / 0 (+0), of 0 | {"J5": {"pass_audited": 20, "pass_right": 20}, "J1": {"pass_audited": 10, "pass_right": 10}} |
| walmart first | 30 / 30 (+0), of 58 | 0 / 0 (+0), of 0 | {"J1": {"pass_audited": 6, "pass_right": 6}, "J5": {"pass_audited": 24, "pass_right": 24}} |
| walmart warm | 30 / 30 (+0), of 62 | 0 / 0 (+0), of 0 | {"J5": {"pass_audited": 26, "pass_right": 26}, "J1": {"pass_audited": 4, "pass_right": 4}} |
