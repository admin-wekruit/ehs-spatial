### Types (warm calls; r4b in brackets)

| | ME340 | Sam's Club | Walmart |
|---|---|---|---|
| object cards | 265 (376) | 822 (917) | 842 (896) |
| typed by family | 93% (95%) | 98% (100%) | 99% (98%) |
| specific name | 13% (20%) | 77% (53%) | 64% (64%) |
| 'unidentified (<shape>)' | 7% (0%) | 2% (0%) | 1% (0%) |
| held-out family right / n (round 4's scorer: 'other:' labels never right) | 19/40 = 0.47 (26/51 = 0.51) | 22/22 = 1.00 (26/26 = 1.00) | 22/28 = 0.79 (24/29 = 0.83) |
| held-out family right / n (r5b scorer: 'other:' labels by the r5b taxonomy) | 24/40 = 0.60 (26/51 = 0.51) | 22/22 = 1.00 (26/26 = 1.00) | 22/28 = 0.79 (24/29 = 0.83) |
| held-out: typed / precision (ext) | 40 / 0.6 (50 / 0.52) | 22 / 1.0 (26 / 1.0) | 25 / 0.88 (26 / 0.923) |

### Clean delivered objects (the instances scorer; r4b in brackets)

| | ME340 | Sam's Club | Walmart |
|---|---|---|---|
| covered / delivered, in pieces, wrong-merge cards | 73/105, 2, 5 (80/105, 7, 4) | 79/91, 7, 3 (79/91, 10, 4) | 58/74, 2, 1 (61/74, 2, 0) |

### VLM requests by purpose (warm call: count [analysis s spans]; r4b in brackets)

| | ME340 | Sam's Club | Walmart |
|---|---|---|---|
| scene vocabulary | 0  (1) | 0  (1) | 0  (1) |
| event captions | 0  (2) | 0  (2) | 0  (2) |
| naming (cluster medoids) | 31 [[44.7, 46.1], [62.9, 63.7]] (48) | 27 [[50.2, 50.9], [72.4, 73.4]] (32) | 20 [[46.6, 47.6], [67.9, 68.7]] (19) |
| hazard / judge | 0  (0) | 0  (0) | 0  (0) |
| qwen identity decider | 0  (0) | 0  (0) | 0  (0) |
| before cards v1 (all purposes) | 0 (3) | 0 (3) | 0 (3) |

### Times (s from the MP4 bytes in the container to the Volume commit; warm call, r4b warm in brackets)

| | ME340 | Sam's Club | Walmart |
|---|---|---|---|
| first 3D | 19.7 (18.0) | 12.9 (15.3) | 12.0 (13.1) |
| objects v1 | 27.3 (32.2) | 20.4 (24.7) | 19.0 (21.8) |
| cards v1 | 33.9 (43.1) | 26.3 (30.1) | 26.4 (31.2) |
| types (first pass) | 48.6 (60.3) | 55.0 (65.1) | 54.6 (60.3) |
| objects v3 | 56.4 (77.5) | 57.5 (59.1) | 52.9 (65.3) |
| cards v3 | 60.6 (83.5) | 67.3 (74.8) | 62.5 (83.0) |
| types (densify pass) | 66.9 (97.8) | 78.8 (87.5) | 74.4 (93.1) |
| first SAM 3D model | None (176.6) | None (105.6) | None (141.9) |
| models final | None (182.8) | None (180.3) | None (182.4) |
| splat preview | None (226.7) | None (226.7) | None (226.3) |
| call end | 67.2 (230.2) | 79.0 (228.5) | 74.7 (226.7) |
| vocab known (mark) | 6.4 (8.6) | 3.5 (4.3) | 3.1 (3.5) |
| decode end | 6.4 (4.2) | 3.5 (4.3) | 3.1 (3.5) |
| GPU peak GiB (0 / 1); stages > 72 | 62.9 / 58.4; none | 69.0 / 64.2; none | 67.1 / 63.4; none |
