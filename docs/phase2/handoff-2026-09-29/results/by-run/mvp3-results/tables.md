## Per video (warm call; first call after boot in brackets)

| | ME340 | Sam's Club | Walmart |
|---|---|---|---|
| first 3D, s | 20.1 (23.0) | 14.1 (19.0) | 15.6 (17.8) |
| objects v1, s | 38.6 (40.3) | 27.4 (29.3) | 25.6 (25.4) |
| cards v1, s | 45.5 (48.3) | 29.4 (34.1) | 28.4 (31.9) |
| all names, s | 86.3 (88.5) | 71.7 (81.5) | 73.6 (78.3) |
| final judgements, s | 88.0 (90.2) | 76.7 (83.5) | 75.8 (80.0) |
| GPU peak GiB gpu0/gpu1 (max over stages) | 60.7/54.0 (63.3/53.9) | 67.1/60.5 (67.3/60.2) | 61.2/57.2 (61.2/57.4) |
| stages over 72 GiB | none | none | none |
| cold start (not analysis), s | 146.9 | 91.1 | 146.0 |
| clicks: random, real objects opening the right thing, report only -> with on-demand | 27/32 -> 29/32 | 22/43 -> 40/43 | 21/42 -> 36/42 |
| clicks: background kept unknown / fixture named / false object card | 12 / 9 / 7 of 28 | 14 / 1 / 2 of 17 | 14 / 0 / 4 of 18 |
| clicks: picked precision (report's own entities) | 0.93 | 1.00 | 1.00 |
| on-demand time to card p50 / p95 / max, s | 1.11/1.46/3.65 | 1.15/1.67/3.5 | 1.11/1.54/2.8 |
| on-demand names on right/coarse masks: right / close / wrong / none | 1 / 1 / 0 / 0 | 4 / 8 / 5 / 1 | 2 / 4 / 5 / 0 |
| identity held-out right, right-or-close (warm) | 0.36, 0.55 (n 55) | 0.86, 0.93 (n 28) | 0.72, 0.83 (n 29) |
| identity held-out right, right-or-close (first) | 0.36, 0.55 (n 55) | 0.86, 0.93 (n 28) | 0.72, 0.83 (n 29) |
| objects with a check (warm) | 27/261 (10%) | 417/622 (67%) | 76/720 (11%) |
| objects with an overall PASS / FAIL (warm) | 11/261 (4.2%) | 36/622 (5.8%) | 51/720 (7.1%) |
| verdict rows (warm) | {"NO_DATA": 12, "PASS": 12, "NEEDS_REVIEW": 17} | {"NEEDS_REVIEW": 450, "PASS": 116, "NO_DATA": 67} | {"PASS": 65, "NO_DATA": 11, "NEEDS_REVIEW": 29} |
| judgement audit (warm): right / audited | PASS 11/11 (of 11); FAIL 0/0 (of 0) | PASS 20/20 (of 89); FAIL 0/0 (of 0) | PASS 20/20 (of 64); FAIL 0/0 (of 0) |
| judgement audit (first): right / audited | PASS 11/11 (of 11); FAIL 0/0 (of 0) | PASS 19/19 (of 87); FAIL 0/0 (of 0) | PASS 20/20 (of 64); FAIL 0/0 (of 0) |
| card audit (20 random, warm): name right/close/wrong/unclear; physical plausible/implausible/unclear; judgement right/wrong/undecided/none | 8/8/4/0; 19/0/1; 0/0/4/16 | 13/7/0/0; 20/0/0; 2/0/14/4 | 16/4/0/0; 19/0/1; 0/0/1/19 |
| Modal list-price upper bound per bench, $ | 1.38 | 1.96 | 1.35 |

## What goes to the VLMs (both calls of a video)

| | ME340 | Sam's Club | Walmart |
|---|---|---|---|
| round 2 measured, Gemini naming: requests (with copies) / images / prompt tokens | 61 / 822 / 464397 | 143 / 1961 / 1105342 | 140 / 1926 / 1086059 |
| round 2 measured, Gemini hazard: requests / counted input tokens | 10 / 90355 | 69 / 797951 | 21 / 206180 |
| replay on round 2's cards, round 2's rule: hazard requests / images / questions | 10 / 75 / 75 | 50 / 465 / 573 | 21 / 170 / 170 |
| replay, this branch's rule: hazard requests / images / questions (input tokens, est.) | 10 / 57 / 57 (68670) | 46 / 430 / 448 (737890) | 8 / 36 / 38 (43662) |
| this round's benches: Gemini requests | 0 | 0 | 0 |
| this round's benches: Qwen identity questions (prompt tokens) | 500 (184260) | 1316 (485372) | 1504 (551576) |
| this round's benches: Qwen hazard questions new / reused / images new | 58 / 118 / 116 | 757 / 770 / 1514 | 52 / 172 / 104 |
