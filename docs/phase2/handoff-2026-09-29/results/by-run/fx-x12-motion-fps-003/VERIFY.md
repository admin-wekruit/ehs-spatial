# Adversarial verification of x12-motion-fps: holds-with-corrections

These corrections take precedence over results.json conclusions.

The numbers reproduce, but several interpretations need correcting before the claim holds. The directional answers survive: 5 fps is enough for walkers; the 4 m/s association gate explains fast-mover failures; a 12 m/s gate at 5 fps recovers most of it at no cost; the adaptive trigger needs a noise filter; and extra frames cost seconds, not minutes.

**What checks out**
- evaluate() re-run from the saved loop outputs (sheet writer turned off, output in the scratchpad) is bit-identical to results-eval.json.
- The self-check passes, the adaptive frame lists reproduce exactly, and every quoted table value matches results.json.
- Boot and analysis times, SAM 3 and loop per-frame costs, NVML memory peaks, GPU and VLM spend, tracker agreement (98/56/75/77%), foot-acceptance shares, hand failure rates and gesture coverage all verify.

**Corrections needed**
- **Evidence base:** all fast-mover evidence comes from two objects, both time-compressed. One is the camera operator's own cart at the bottom edge in Sam's Club (100% of those steps). The other is one guide walking next to the camera in the factory clip (100%).
- **Path error:** it is measured against noisy every-frame centroids. At 15 fps the p90 roughly equals the frame-to-frame jitter (6.7 cm, 24-29 cm, 25 cm), so the 'path accuracy' and 'how close a vehicle came' statements are unsupported.
- **R3 flips:** they are disagreement ticks summed over unequal run counts. Per run, 5 fps beats 12.5 fps (5.17 vs 6.25).
- **Sam's Club 1/236 vs 9/457:** these compare different objects. The 5 fps steps are the operator's cart; the 12.5 fps failures are distant-shopper track pieces that 5 fps barely samples.
- **Speed table:** it silently drops steps without a speed estimate (all 9 failures at 12.5 fps) and omits the 12.5 fps column and the >12 m/s bin.
- **Image-space rule:** the 0.5 w cliff follows from the IoU-0.3 gate, and the omitted mover row fails 20% at 0.1-0.25 w.
- **Cost per extra frame:** it is understated about 1.2-1.5x. DA3's real marginal cost is 0.125-0.18 GPU-s per in-between frame, so uniform 15 fps adds about +35-45 s and adaptive about +11-14 s.
- **VLM 4 fps vs 8 fps:** the two scans had different budgets (6 vs 13 windows, 2x the tokens), and both misses at 4 fps were empty answers.
- **Licences:** none are stated. DA3-GIANT-1.1 is CC BY-NC 4.0 (non-commercial); SAM 3 is under the SAM License; Qwen3-VL-8B is Apache-2.0.
- **Delivered references:** c40fbd08 / 913daf2a / 32cec650 are never used.

**My label audit:** 44 of 53 sampled track rows agree (83%), 8 are uncertain and 1 I read differently. The label behind the Sam's Club result cannot be seen in its own crops; the failures sheet confirms it.

No GPU jobs were run and nothing was spent. Scripts are in /private/tmp/claude-501/-Users-adam-Desktop-panoptes-public/1fd9a1db-e580-4bfc-8110-119a1cc38a99/scratchpad/: reeval.py, chains.py, comp.py, jitter.py and adapt.py.

## Problems
- [major] All fast-mover evidence (the k=6 proxy and the 3 m/s-and-up bins) comes from two objects that move with or next to the camera. Sam's Club is 100% the camera operator's own cart at the bottom edge of the frame (0/mover-2, 1/mover-0: 23/23 steps at 5 fps, 106/106 at 15 fps). The factory clip is 100% one guide walking next to the camera (person-1). The ME340 instructor stays under about 1 m/s even at k=6. Steps are counted up to 3 times (phase offsets), frames are reused across k, and no confidence intervals are given (0/27 still allows up to 12.8%).
  Fix: State that n is effectively 2 objects plus time compression. Report counts per object with Wilson CIs and without pseudo-replication. Present 'fast movers: 15 fps better' as a gate-algebra result shown on two proxy objects, not measured on vehicles.
- [major] The 'path accuracy' numbers compare the gaps a run skipped, filled by interpolation, against the noisy every-frame centroids of the same detections (path_errors uses xy for both sides). At 15 fps the p90 is about the size of the frame-to-frame centroid jitter: ME340 6.7 cm, Sam's cart 24-29 cm, factory guide 25 cm (p90 deviation from a ±0.2 s mean). So '15 fps gives a tighter path' and the point about how close a vehicle came to a person are not established.
  Fix: Compare against a smoothed or independent reference path, or report the jitter floor next to each p90. Drop the vehicle-proximity implication.
- [major] The R3 'flips' 25/31/59/195 are summed over unequal numbers of runs (4 at 12.5 fps, 6 at the other rates). They also count video-frame ticks where the run and the every-frame run disagree PASS vs FAIL, not verdict changes. Per run they are 6.25 at 12.5 fps vs 5.17 at 5 fps, so 5 fps is more stable than 12.5 fps.
  Fix: Normalise per run or per second. Call them disagreement ticks. Restate the result: verdict stability improves only below 5 fps.
- [major] Sam's Club walkers '1/236 vs 9/457' compares different objects. 232 of the 236 steps at 5 fps are the operator's own cart. All 9 failures at 12.5 fps are on pieces of distant shoppers' tracks (chains of 0.2-0.3 s), and those pieces give only 4 steps at 5 fps. The step metric hides short chains at low rates.
  Fix: Compare rates on the same truth links over the same time spans, or only on chains that have at least 2 samples at every rate compared.
- [major] The speed x rate table leaves out steps without a speed estimate (chains shorter than 0.5 s). That drops all 9 failures at 12.5 fps, 5 of 21 at 15 fps, 4 of 21 at 10 fps, and 2 of 22 at 1.67 fps. The 12.5 fps column and the >12 m/s bin (1/2 at 10 fps, 2/6 at 15 fps) are omitted from the claim.
  Fix: Add an 'unknown speed' row and the 12.5 fps column, or merge 12.5 into 15 with a note.
- [major] The image-space rule is overgeneralised. The cliff near 0.5 w is algebra of the chosen IoU >= 0.3 gate (equal boxes: IoU = (1-x)/(1+x)); SAM 2/3 mask propagation was never tested. The mover class breaks the 'about 2% or less below 0.25 w' rule: real movers failed 63/1532, and 44/218 (20%) at 0.1-0.25 w. That row is missing from the claim. The persons row is the agent-real subset, while the hands and arms rows are unverified detections scored against the tracker's own every-frame output.
  Fix: Present the rule as a property of an IoU-0.3 box tracker only. Add the mover row. Label which rows are verified.
- [major] Cost per extra frame is understated. The DA3 figure of 0.076-0.099 GPU-s is per view in a chunk, and every chunk re-runs all 113-150 anchors. The measured marginal cost per in-between frame is 0.125-0.180 GPU-s. The total is about 0.23-0.30 GPU-s per extra frame, about 0.12-0.15 s of analysis (every-frame pass measured at 0.124-0.164 s per frame). That makes uniform 15 fps about +35-45 s, not +30 s, and adaptive about +11-14 s, not +9 s.
  Fix: Use chunk GPU-s divided by in-between frames, add the align stage, and redo the projections. The 'minutes, not tens of minutes' conclusion still holds.
- [major] The VLM result (2/5 gestures at 4 fps vs 4/5 at 8 fps) does not use the same prompt budget. The 8 fps scan had 13 windows and 67.8k prompt tokens; the 4 fps scan had 6 windows and 33.6k. Both gestures missed at 4 fps fell in windows where the VLM returned an empty answer. Decoding was sampled (not greedy), there was one run, and n=5.
  Fix: Hold windows, duration and token budget equal (for example 16 s windows at both rates), repeat the scan several times, and report the empty-answer rate separately.
- [major] No licences are stated. From primary sources: DA3-GIANT-1.1 is CC BY-NC 4.0 (non-commercial only; Hugging Face card); SAM 3 is under the SAM License (GitHub LICENSE; gated on Hugging Face); Qwen3-VL-8B-Instruct is Apache-2.0. The DA3 licence blocks commercial use of this pipeline as it stands.
  Fix: Add a licence block to results.json and flag the DA3-GIANT restriction.
- [major] The reference leaves out the hard cases. Consensus keeps only 56% (Sam's Club), 75% (Walmart) and 77% (factory) of world-loop links. The 'real' label excludes the hopping red-shirt guide, the hardest near-camera case. So failure rates are measured on the easy subset (a lower bound). The rule lines (fps >= 2 x (v - 2.5), the 0.25 w rule) are fitted by eye on the same data, with nothing held out.
  Fix: Report failure rates on all moving chains next to the real subset. Name the rules as fits to this data, not measured thresholds.
- [minor] '15 fps is fine up to about 10 m/s' is not supported: the 8-12 m/s bin at 15 fps fails 6/71 (8.5%, above the 5% line). The 5-8 m/s bin at 10 fps is 6/117 (5.1%).
  Fix: Say 15 fps holds up to 8 m/s, and that the 8-12 m/s bin was not resolved.
- [minor] '12 m/s gate adds no errors on walkers' holds only for the real tracks. On all moving chains Walmart went from 1/84 to 4/84. At 5 fps the new gate is 3.1 m, and crowded scenes were never tested.
  Fix: Qualify the claim and test it on a clip with several people.
- [minor] Gesture boundaries come from sheets of every 2nd frame with ±2 frames (0.07 s) precision, while the fast phases last only 0.17-0.23 s. My reading of the E2 peak is f196-206 (0.37 s) rather than f198-204. With that reading, 5 fps gets at least 2 frames in about 5 of 6 phases, not 17%. The fps >= 2/D rule is arithmetic, not a measurement.
  Fix: Label on every frame, report the coverage range across ±2-frame boundaries, and present 2/D as a sampling identity.
- [minor] The 'moving' filter (leaves a 0.5 m disc) lets in distant workers who are standing still: factory person-42/49/51/56 have 0.1-0.5 s chains that spread 0.57-0.9 m from depth jitter. They account for 3 of the 6 factory failures at 15 fps.
  Fix: Require net displacement or a consistent direction, not only spread.
- [minor] The contact-sheet labels only partly carry the load. I checked 53 of 123 track rows: 44 agree, 8 are uncertain (the first crop is blur, or the boxes are tiny and static), and 1 I read differently (factory person-56 as mixed). The label that carries the Sam's Club result (0/mover-2) cannot be seen in its own crops. On the SAM 3 words sheet, 3 of the top-6 'vehicle' masks look like real parked cars through the window, 2 of the top-6 'cart' masks are a distant person, and only the top 6 of 56/111/16 detections were inspected for 'all false'.
  Fix: Crop edge objects so they stay visible. Use more crops per track. Qualify 'all false' as based on the top 6.
- [minor] Memory: '36-43 GiB per GPU' describes GPU0 only; GPU1 peaked at 29.2-35.7 GiB. The VLM stage logged no GPU memory. The Volume commit is outside the analysis clock.
  Fix: Report both GPUs, log memory for the VLM stage, and put the commit inside the clock.
- [minor] 'Switched to 15 fps more than a third of the time' is wrong for Sam's Club: 47 of 148 base intervals (32%). The factory clip is 36%. Four adaptive variants were run; the cheapest (4 m/s trigger, 2 intervals) is reported without saying so. The variants are identical at k=6.
  Fix: Correct the share and mention the variants.
- [minor] Spend: container seconds for the CPU loops recompute to $0.72, not $0.825. The total is about $2.42, not $2.53, so the claim overstates slightly (the safe direction).
  Fix: Recompute the CPU line.
- [minor] The delivered references (c40fbd08 / 913daf2a / 32cec650, the publications for me340 / samsclub-a2 / walmart per fb-results/summary.json) are not used anywhere. The reference is internal only.
  Fix: State that no delivered reference applies, or compare with the delivered people and rule findings.

## Corrected table

All numbers were re-derived from fx-x12-motion-fps-003/results.json. evaluate() was re-run from the saved loops-*.json.gz with the contact-sheet writer turned off, into the scratchpad: the output is bit-identical to results-eval.json. The self-check passes, and the adaptive frame lists reproduce exactly. Every figure the claim quotes matches results.json unless a correction is marked below. Speeds and distances are ESTIMATED (floor plane + assumed 1.6 m camera height).

**World loop, real moving steps, speed x rate (4 m/s gate)**

| speed (m/s) | 1.67 fps | 2.5 | 5 | 10 | 12.5 (Sam's k=1 only, omitted from the claim) | 15 |
|---|---|---|---|---|---|---|
| <=2 | 2/312 | 0/494 | 0/1022 | 1/1635 | 0/367 | 3/1807 |
| 2-3 | 0/16 | 1/31 | 1/70 | 2/136 | 0/20 | 1/201 |
| 3-5 | 7/28 | 4/47 | 1/116 | 1/230 | 0/13 | 2/373 |
| 5-8 | 7/7 | 13/15 | 19/49 | 6/117 | - | 2/179 |
| 8-12 | 4/4 | 5/5 | 7/13 | 6/39 | - | 6/71 |
| >12 (omitted from the claim) | - | - | - | 1/2 | - | 2/6 |
| steps left out (no speed estimate: truth chain shorter than 0.5 s) | 12 (2 failed) | 15 (2) | 26 (1) | 21 (4) | 66 (9) | 35 (5) |

- The last row is the correction that matters: every one of Sam's 9 failures at 12.5 fps falls outside the table.
- Everything at 3 m/s and above comes from two objects:
  - Sam's Club: the camera operator's own cart at the bottom edge of the frame (tracks 0/mover-2 and 1/mover-0). It is 100% of the Sam's steps at k=6.
  - Factory: one guide walking next to the camera, in a grey hoodie (person-1). It is 100% of the factory steps at k=6.
- The ME340 instructor never passes about 1 m/s, even at k=6.
- Each step is counted up to 3 times (once per phase offset), and the same frames are reused across k (k=3 at 5 fps uses the same frames as k=1 at 1.67 fps).
- No confidence intervals are given. For scale: 0/27 still allows up to 12.8% (95%).

**Fast proxy (k=6)**
- The table's numbers are verified:
  - Sam's Club: 11/23, 7/64, 3/106, 1/23, 0/27.
  - Factory: 16/47, 8/101, 5/158, 3/47, 2/52.
  - Coverage, path p90 and frames per run also match.
- What "path p90" measures: the frames a run skipped are filled in by interpolation and compared with the noisy every-frame centroids. It is not path accuracy. The 15 fps value is about the same size as the frame-to-frame jitter of the centroid (p90 of its deviation from a ±0.2 s mean):
  - ME340 instructor: 6.7 cm
  - Sam's Club cart: 24-29 cm
  - Factory guide: 25 cm

**Walkers (k=1)**
- ME340: 0/331 vs 0/668. This is one person.
- Sam's Club: 1/236 at 5 fps vs 9/457 at 12.5 fps, but these are not the same objects:
  - At 5 fps, 232 of the 236 steps are the operator's own cart.
  - At 12.5 fps, all 9 failures are on pieces of distant shoppers' tracks, and those pieces give only 4 steps at 5 fps.
- Speed-verdict "flips" (R3 on body centroids): the claim's 25 / 31 / 59 / 195 at 12.5 / 5 / 2.5 / 1.67 fps are:
  - summed over 4 / 6 / 6 / 6 runs;
  - counts of video-frame ticks where the run and the every-frame run disagree PASS vs FAIL, not verdict changes.
  - Per run they are 6.25 / 5.17 / 9.8 / 32.5, so 5 fps beats 12.5 fps.
- Factory: 0/327 vs 6/675. Three of the 6 are a distant worker who is standing still (person-42); only his position jitter makes him count as moving.

**Image-space association, real subset (box IoU >= 0.3)**

| object | < 0.25 w | 0.25-0.5 w | 0.5-0.75 w |
|---|---|---|---|
| persons (agent-real) | 0/4402 | 5/141 | 8/26 |
| movers (agent-real, missing from the claim) | 63/1532 (0.1-0.25 w: 44/218 = 20%) | 7/44 | 15/15 |
| hands (all detections, unverified) | 4/284 | 77/377 | 182/211 |
| arms (all detections, unverified) | 5/4025 | 74/1137 | 220/391 |

- For hands and arms, the reference is the image tracker's own every-frame output.
- The cliff near 0.5 w follows from the chosen gate: for two equal boxes, IoU = (1-x)/(1+x) >= 0.3 means a shift x <= 0.54 w.
- Hand failures by rate, 21% / 47% / 80% at 15 / 10 / 5 fps: verified.

**Gestures (agent-labelled)**
- Frame counts: verified.
- Labels were made on sheets of every 2nd frame, ±2 frames. My reading of the E2 peak is f196-206 (0.37 s), not f198-204 (0.23 s).
- VLM: 2/5 gestures with 4 fps input used 6 windows and 33.6k prompt tokens; 4/5 with 8 fps input used 13 windows and 67.8k tokens.
  - Both gestures missed at 4 fps fell in windows where the VLM returned an empty answer.
  - Empty answers: 7 of 13 windows at 8 fps, 2 of 6 at 4 fps (verified).

**Cost per extra frame (2x A100-SXM4-80GB), corrected**

| part | cost per frame |
|---|---|
| SAM 3 person/floor | 0.062-0.064 GPU-s |
| SAM 3 mover words | +0.045 GPU-s (7 words); 0.061 (9 words, factory) |
| DA3, per in-between frame | 0.125 (Sam's, Walmart), 0.142 (ME340), 0.180 (factory) GPU-s. The claim's 0.076-0.099 is per view in a chunk, and every chunk re-runs all anchors. |
| align | about 0.6 s per shot |
| loop | 0.0085-0.023 CPU-s |
| total | about 0.23-0.30 GPU-s, about 0.12-0.15 s of analysis on 2 GPUs |

- Measured analysis per frame on the every-frame pass: 0.124-0.164 s. This includes decode and shot cuts.
- Projections:
  - Uniform 15 fps instead of 5 on a 30 s clip: about +35-45 s (the claim says +30 s).
  - Adaptive at k=1: Sam's Club and factory about +11-14 s each (claim: +9 s); Walmart about +2 s; ME340 +0 s.
  - 12 m/s gate: 0 s.
- Adaptive trigger share: 47/148 base intervals on Sam's Club (32%, below the claimed "more than a third") and 46/129 on the factory clip (36%).

**Run facts**
- Analysis 93-128 s per video and boot 33-47 s: verified. The Volume commit falls outside the analysis clock.
- GPU memory peaks, device-level NVML:
  - GPU0: 36.3-43.4 GiB
  - GPU1: 29.2-35.7 GiB
  - No peak is over 90%.
  - The VLM stage logged no memory.
- Spend:
  - GPU $1.507 and VLM $0.194: verified.
  - CPU loops recompute to $0.72, not $0.825, so the total is about $2.42 against the claimed $2.53. The claim errs on the high side.
- Licences (none are stated in results.json):
  - DA3-GIANT-1.1: CC BY-NC 4.0, non-commercial use only (Hugging Face card).
  - SAM 3: SAM License (GitHub LICENSE; gated on Hugging Face, tag "other").
  - Qwen3-VL-8B-Instruct: Apache-2.0.
- The delivered publications c40fbd08 / 913daf2a / 32cec650 are never used. The reference is the internal consensus of the two trackers, which keeps 97.6% / 56.4% / 74.6% / 77.2% of links (verified).

**Contact-sheet audit (mine)**
- 53 of 123 labelled track rows checked: 44 agree (83%), 8 uncertain, 1 disagree (factory person-56: I read it as mixed).
- The label that carries the Sam's Club result (0/mover-2, the operator's cart) cannot be seen in its own crops. The failures sheet confirms it.
- SAM 3 words sheet:
  - 'cart': 4 of the top 6 are the real cart; 2 are a distant person's legs.
  - 'vehicle': 3 of the top 6 look like real parked cars through the window.
  - forklift, pallet jack, hand, arm: agree.
