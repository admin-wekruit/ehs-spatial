# mvp2/accuracy: card physical values against independent metric ground truth (R5)

Branch `mvp2/accuracy` (worktree `/Users/adam/.codex/worktrees/panoptes-phase2-video-mvp2-accuracy`). All GPU work: ephemeral `modal run`, 2 x A100-80GB (MPS), retries 0, `min_containers=0`; analysis time = MP4 bytes in the container -> layer written; display layers (SAM 3D, splat) off in these runs (`display: False`), so the fact timings are not slowed by them.

## What was scored

- **Sequences (RGB only to the MVP; depth and poses never uploaded).** ARKitScenes raw 47333932 (257 capture frames 0.1-15.8 s apart, ARKit poses, LiDAR depth, confident pixels only; two windows of 128/129 frames), ARKitScenes 42445448 (47 stills ~2 s apart, laser-scan depth 1920x1440, captured upside down and turned 180 deg), TUM fr1 room (30 fps, motion-capture poses, Kinect depth with TUM's registered-depth intrinsics; windows 0-30 s and 30-45.4 s). 16:9 centre crop to 1280x720 (the MVP's input contract). ARKit frames are held 6 video frames each so every capture frame is exactly one 5 fps keyframe; those runs pass `cuts: none` (held frames read as a cut every 0.2 s; the captures are unedited). TUM runs with the cut rule (it found a 9-keyframe second shot in window 0).
- **GT per card.** The card's own pick-v2 regions (segmented keyframes, eroded 1 px at 640x360, GT depth edges > 5 % dropped, the card's per-view depth-tail trim and main cluster) lifted with GT depth + GT poses into a GT floor frame built like the card's (origin = the shot's first keyframe camera on the floor, +z = floor normal, +x = that camera's forward). Same definitions as the card: p2/p98 heights, footprint sides along the card's own footprint axes, footprint centre, `measure_observed_points` angles. Masks are not scored (the GT uses the same masks), identity is not scored.
- **GT checks.** GT depth carried between views agrees to 0.6-0.8 % median (p90 2.4-3.6 %). GT floors: lowest horizontal plane of the fused GT depth, normal within 0.8 deg of gravity, p90 residual 0.9-1.4 cm. True camera heights 1.48/1.50 m (arkit47 windows), 1.23 m (arkit42), 1.51/1.42 m (TUM windows): all below the assumed 1.6 m. Gate: GT depth on >= 50 % of the mask pixels sampled, >= 50 GT points; shots without a floor excluded; first calls excluded from accuracy tables (they duplicate a warm call).
- **(a) vs (b).** (a) = the card as delivered (scale from the assumed 1.6 m camera height). (b) = the card x s, s = true camera height / 1.6 (what the MVP's own scale rule gives with the right height). (a) - (b) is the scale error, (b) is the geometry error. Positions are compared after a floor-plane rigid fit of the shot's camera path (card frame -> GT frame; (b) after scaling by s): the card frame's yaw is set by one camera, and arkit47 w0 starts looking down at a poster on the floor, which turned the card frame 14.5 deg (distances between things are unaffected; position_xy coordinates are).
- **Geometry options** (same videos, same code, `opts.geometry`): `shot` = today's per-shot DA3-GIANT any-view (A); `windows` = X6 content windows (threshold 0.40) with per-window DA3 + Sim3 stitch (B); `posed-droid` = DROID-SLAM on the run's 5 fps keyframes (DA3's intrinsics, 384x216, A100-80GB) whose cameras condition DA3 (C; DROID timed apart, it would run beside on GPU 1); `posed-gt` = the same posed DA3 call with the true cameras (a diagnostic upper bound for any pose source).

## Results in one paragraph

Geometry at the true camera height is already accurate with today's option A: median |error| 1.8-3.3 cm for tops and bases, 1.7-2.9 cm for heights, 1.7-3.4 cm for widths, 6-13 cm for positions (after the camera-path fit), 0.8 deg for the 12 plane slopes shown on TUM. What makes the delivered numbers wrong is the assumed 1.6 m camera height: s = 0.77-0.94 on all five shots, so delivered tops read +4 to +22 cm high (median 5.5 / 21.8 / 7.5 cm |error| on arkit47 / arkit42 / TUM). B loses (stitch scales swing 0.6-2.0 per link; ATE 18/39/7 cm vs A 5/8/4 cm; tops at true height 15 / 185 / 3 cm) and is not faster here (DA3 runs after decode in both). C loses: at the MVP's 5 fps keyframes DROID loses track on 4 of 5 windows (ATE 0.65-0.92 m; the arkit47 w0 run then failed in the TSDF), and where it tracks (TUM w1, ATE 2.8 cm) the cards are no better than A (tops 1.9 vs 2.0 cm, widths 2.1 vs 1.9 cm); at 30 fps it takes 18-48 s on one A100 for 15-30 s of video with ATE 3.0/17.2 cm against DA3's 1.9/5.7 cm. Even true cameras did not improve posed DA3 over A (tops at true height 2.4/4.1/4.2 vs 1.8/3.3/2.9 cm) -- with the caveat that my posed call's depth-to-camera scale disagreed by up to 28 % (floor-derived camera height 1.25-1.92 m with true poses), so posed DA3 itself is not settled, but no pose source in budget beats A's 4-8 cm ATE. **Recommendation: keep A (per-shot DA3-GIANT any-view).**

Today's u covers well except where the scale assumption is off by more than 20 % (arkit42, camera at 1.23 m: heights 79 %, extents 78 %, positions 68 %) and for one-view-set extents (64-74 % everywhere). **New u rule (implemented in the cards):** u = sqrt((k_geo[family][view-set state] x sqrt(sum of the non-scale parts^2))^2 + (0.25 |value|)^2), k_geo fitted on the ARKit sequences only (height 1.0 / 1.41, extent 2.39 / 4.52, position 2.23 / 4.33 for >= 2 view sets / one set), scale term 0.20 -> 0.25 (a stated 1.2-1.8 m camera-height range; the largest ARKit |1 - s| was 0.231). **Held out (TUM, cards made by the new code, run 003):** heights 98 % / 95 %, extents 95 % / 85 %, positions 100 % / 95 %, angles 13/13; ARKit in-sample 89-100 %. Cost: median u +14 % on heights and +38 % on extents on TUM (>= 2 view sets); on the three MVP videos' round-1 cards +11-13 % heights, +65-71 % extents, +24-30 % positions. On these GT scenes the judgement counts barely moved (J4 PASS 66 -> 64, J5 NEEDS_REVIEW 35 -> 30 over the same five windows, runs 001 vs 003, which are not bit-identical runs). The 0.25 scale term is a prior chosen after arkit42 showed 0.20 failing; its only clean held-out test is TUM, where 0.20 also worked.

## Also fixed

- A shot without a floor plane (TUM w0's 9-keyframe second shot) showed DA3 units as 'estimated' metres; the Sim3 to truth says they were 1.67x off. Every metric field and angle of such a card is now `not measurable (no floor plane in this shot: the scale and the floor are unknown)`, its size check `no data`, and the identity veto no longer reads that size.
- The judge kept its own 20 % scale term for gaps and free widths; it now reads the cards' one constant. Its scale-free u (`u_rel`) is exact for the new rule.

## Open

- The card frame's x axis (first camera forward) is fragile when that camera looks down (14.5 deg on arkit47 w0); the median forward of the shot's cameras would fix it (spec 4.1 change, not made here).
- The scale assumption is the whole of the remaining error; a measured scale (a known-size object, a metric depth head) would shrink u by more than any geometry option.
- 'at least' / 'at most' bounds hold within u against GT in 100 of 115 cases (option A, warm).
- Only three indoor scenes (homes, one office); none is a warehouse or a shop.

## Geometry options (warm calls; errors on shown values with GT cover >= 0.5)

median |error| in m: (a) as delivered (assumed 1.6 m camera height) / (b) at the true camera height; cov = share of (a) inside the ±u the cards carried in these runs (the u before this change). posed-droid / posed-gt timings exclude DROID itself (table below).

| sequence | option | camera ATE (Sim3) | s = true h / 1.6 | top (a)/(b), cov | base (a)/(b), cov | height (a)/(b), cov | width (a)/(b), cov | position (a)/(b), cov |
|---|---|---|---|---|---|---|---|---|
| arkit42 | posed-droid | 0.817 | 0.77 | 0.362 / 0.258, 100% (n 30) | 0.324 / 0.220, 97% (n 31) | 0.131 / 0.088, 24% (n 29) | 0.173 / 0.066, 36% (n 28) | 1.783 / 1.361, 0% (n 32) |
| arkit42 | posed-gt | 0.000 | 0.77 | 0.219 / 0.024, 70% (n 40) | 0.193 / 0.022, 85% (n 41) | 0.071 / 0.020, 68% (n 40) | 0.091 / 0.029, 68% (n 41) | 0.389 / 0.112, 67% (n 42) |
| arkit42 | shot | 0.053 | 0.77 | 0.218 / 0.018, 73% (n 56) | 0.193 / 0.019, 71% (n 58) | 0.036 / 0.017, 78% (n 55) | 0.094 / 0.017, 80% (n 56) | 0.432 / 0.061, 63% (n 59) |
| arkit42 | windows | 0.177 | 0.77 | 0.058 / 0.151, 73% (n 56) | 0.067 / 0.100, 81% (n 57) | 0.026 / 0.034, 77% (n 56) | 0.032 / 0.055, 82% (n 55) | 0.108 / 0.290, 84% (n 57) |
| arkit47 | posed-droid | 0.919 | 0.94 | 0.649 / 0.630, 51% (n 119) | 0.660 / 0.615, 51% (n 121) | 0.102 / 0.097, 45% (n 115) | 0.153 / 0.144, 38% (n 112) | 1.366 / 1.265, 21% (n 125) |
| arkit47 | posed-gt | 0.000 | 0.93, 0.94 | 0.077 / 0.041, 96% (n 155) | 0.075 / 0.051, 94% (n 146) | 0.041 / 0.040, 84% (n 141) | 0.045 / 0.033, 92% (n 145) | 0.174 / 0.203, 71% (n 160) |
| arkit47 | shot | 0.077 | 0.93, 0.94 | 0.055 / 0.033, 99% (n 154) | 0.052 / 0.028, 98% (n 152) | 0.031 / 0.028, 88% (n 145) | 0.046 / 0.034, 85% (n 146) | 0.246 / 0.130, 85% (n 162) |
| arkit47 | windows | 0.387 | 0.93, 0.94 | 2.014 / 1.853, 47% (n 161) | 1.547 / 1.411, 48% (n 165) | 0.241 / 0.224, 37% (n 155) | 0.292 / 0.262, 27% (n 157) | 1.861 / 1.740, 33% (n 171) |
| tum | posed-droid | 0.326 | 0.94, 0.89 | 0.101 / 0.072, 93% (n 232) | 0.109 / 0.067, 94% (n 231) | 0.033 / 0.033, 81% (n 220) | 0.042 / 0.031, 87% (n 230) | 0.305 / 0.204, 84% (n 245) |
| tum | posed-gt | 0.000 | 0.94, 0.89 | 0.051 / 0.042, 99% (n 279) | 0.056 / 0.036, 99% (n 278) | 0.027 / 0.029, 84% (n 267) | 0.057 / 0.043, 87% (n 278) | 0.226 / 0.164, 88% (n 292) |
| tum | shot | 0.044 | 0.94, 0.89 | 0.075 / 0.029, 98% (n 256) | 0.084 / 0.031, 96% (n 255) | 0.026 / 0.029, 80% (n 243) | 0.037 / 0.027, 90% (n 257) | 0.183 / 0.077, 99% (n 270) |
| tum | windows | 0.069 | 0.94, 0.89 | 0.066 / 0.032, 98% (n 240) | 0.078 / 0.026, 96% (n 238) | 0.026 / 0.029, 87% (n 228) | 0.033 / 0.029, 90% (n 242) | 0.164 / 0.151, 98% (n 253) |

## Timing (s from the MP4 bytes in the container; display layers off)

| sequence | window | option | call | cameras | objects | cards v1 | cards v3 | judgements v3 | DA3 stage | GPU peaks GiB |
|---|---|---|---|---|---|---|---|---|---|---|
| arkit42 | 0 | posed-droid | warm | 10.3 | 15.0 | 17.1 | 24.0 | 26.1 | 2.64 | 65.3/54.3 |
| arkit47 | 1 | posed-droid | warm | 19.1 | 24.5 | 29.3 | 51.7 | 56.8 | 7.46 | 71.3/54.9 |
| tum | 0 | posed-droid | warm | 16.9 | 29.4 | 35.2 | 64.9 | 78.2 | 8.65 | 66.3/52.9 |
| tum | 1 | posed-droid | warm | 9.5 | 18.5 | 22.1 | 37.2 | 40.9 | 3.3 | 65.2/52.7 |
| arkit42 | 0 | posed-gt | warm | 7.0 | 11.4 | 11.4 | 18.6 | 18.6 | 1.7 | 62.9/52.0 |
| arkit47 | 0 | posed-gt | warm | 14.9 | 20.3 | 24.1 | 39.5 | 39.5 | 7.07 | 64.2/51.5 |
| arkit47 | 1 | posed-gt | warm | 14.6 | 24.1 | 28.2 | 50.2 | 52.0 | 7.29 | 67.3/52.5 |
| tum | 0 | posed-gt | warm | 17.1 | 31.4 | 36.3 | 67.8 | 77.1 | 8.68 | 67.9/52.6 |
| tum | 1 | posed-gt | warm | 8.4 | 17.9 | 22.0 | 38.3 | 42.9 | 3.27 | 67.3/52.6 |
| arkit42 | 0 | shot | warm | 6.4 | 10.0 | 11.9 | 18.6 | 18.6 | 1.73 | 56.2/52.1 |
| arkit47 | 0 | shot | first | 16.1 | 21.2 | 24.8 | 38.8 | 38.8 | 6.84 | 57.6/51.4 |
| arkit47 | 0 | shot | warm | 13.3 | 19.4 | 21.0 | 33.3 | 35.4 | 6.68 | 60.9/53.5 |
| arkit47 | 1 | shot | warm | 15.8 | 22.5 | 27.1 | 49.0 | 49.0 | 7.43 | 58.6/52.7 |
| tum | 0 | shot | warm | 17.6 | 27.7 | 33.7 | 58.6 | 66.9 | 8.3 | 62.5/52.7 |
| tum | 1 | shot | warm | 8.1 | 16.0 | 18.1 | 35.8 | 35.8 | 3.24 | 60.2/52.7 |
| arkit42 | 0 | windows | warm | 7.8 | 12.0 | 13.9 | 19.9 | 19.9 | 2.82 | 59.4/52.0 |
| arkit47 | 0 | windows | warm | 13.7 | 19.3 | 20.9 | 34.8 | 34.8 | 7.22 | 60.3/53.2 |
| arkit47 | 1 | windows | warm | 15.0 | 22.8 | 24.7 | 47.6 | 49.4 | 6.72 | 61.3/52.5 |
| tum | 0 | windows | warm | 17.1 | 27.2 | 33.3 | 58.3 | 62.7 | 7.77 | 62.8/52.7 |
| tum | 1 | windows | warm | 9.6 | 17.0 | 18.6 | 32.2 | 34.5 | 3.87 | 60.6/52.4 |

## DROID-SLAM on the same frames (A100-80GB, DA3's intrinsics, 384x216)

| job | frames | seconds (track + terminate) | ATE vs GT (Sim3, m) | peak GiB |
|---|---|---|---|---|
| gt-arkit42-w0 | 47 | 12.64 (10.44 + 1.84) | 0.817 | 0.81 |
| gt-arkit47-w0 | 128 | 12.08 (7.39 + 3.58) | 0.842 | 1.04 |
| gt-arkit47-w1 | 129 | 9.74 (5.8 + 3.01) | 0.919 | 1.04 |
| gt-tum-w0 | 150 | 13.68 (7.68 + 4.91) | 0.650 | 1.19 |
| gt-tum-w1 | 77 | 6.46 (3.54 + 2.39) | 0.028 | 1.19 |
| gt-tum-w0-30fps | 900 | 47.6 (36.32 + 8.46) | 0.172 | 2.43 |
| gt-tum-w1-30fps | 462 | 18.13 (12.24 + 4.37) | 0.030 | 2.43 |

## u rule, fitted on ARKit (47333932 x2 windows, 42445448), tested on TUM fr1 room (held out)

k_geo = {"height": {"sets": 1.0, "one_set": 1.408}, "extent": {"sets": 2.391, "one_set": 4.522}, "position": {"sets": 2.229, "one_set": 4.328}}; scale part 0.25 x |value| (largest |1 - s| on the fit set 0.231, on the test set 0.111).

| family | view sets | split | n | coverage, today's u | coverage, new u | median u today / new (m) | geometry-only coverage at true height, new |
|---|---|---|---|---|---|---|---|
| height | sets | calib | 310 | 95% | 99% | 0.213 / 0.238 | 91% |
| height | sets | tum | 403 | 99% | 99% | 0.196 / 0.223 | 93% |
| height | one_set | calib | 110 | 81% | 98% | 0.211 / 0.255 | 90% |
| height | one_set | tum | 108 | 92% | 94% | 0.191 / 0.226 | 83% |
| extent | sets | calib | 369 | 88% | 94% | 0.109 / 0.136 | 90% |
| extent | sets | tum | 492 | 90% | 95% | 0.098 / 0.135 | 92% |
| extent | one_set | calib | 108 | 69% | 91% | 0.042 / 0.084 | 91% |
| extent | one_set | tum | 99 | 69% | 87% | 0.056 / 0.124 | 83% |
| position | sets | calib | 160 | 84% | 99% | 0.574 / 0.730 | 89% |
| position | sets | tum | 207 | 100% | 100% | 0.545 / 0.725 | 99% |
| position | one_set | calib | 61 | 67% | 98% | 0.442 / 0.667 | 90% |
| position | one_set | tum | 63 | 95% | 95% | 0.646 / 0.916 | 95% |
| angle | sets | calib | 6 | 100% | - | 13.730 / - | - |
| angle | sets | tum | 13 | 100% | - | 3.041 / - | - |

## u rule, reverse check: fitted on TUM, tested on ARKit

k_geo = {"height": {"sets": 1.0, "one_set": 1.797}, "extent": {"sets": 2.048, "one_set": 6.736}, "position": {"sets": 1.427, "one_set": 2.844}}; scale part 0.25 x |value| (largest |1 - s| on the fit set 0.111, on the test set 0.231).

| family | view sets | split | n | coverage, today's u | coverage, new u | median u today / new (m) | geometry-only coverage at true height, new |
|---|---|---|---|---|---|---|---|
| height | sets | calib | 403 | 99% | 99% | 0.196 / 0.223 | 93% |
| height | sets | arkit47 | 243 | 99% | 99% | 0.213 / 0.236 | 90% |
| height | sets | arkit42 | 67 | 79% | 99% | 0.221 / 0.242 | 96% |
| height | one_set | calib | 108 | 92% | 95% | 0.191 / 0.234 | 91% |
| height | one_set | arkit47 | 63 | 95% | 100% | 0.172 / 0.224 | 87% |
| height | one_set | arkit42 | 47 | 62% | 100% | 0.235 / 0.320 | 98% |
| extent | sets | calib | 492 | 90% | 94% | 0.098 / 0.121 | 90% |
| extent | sets | arkit47 | 272 | 91% | 93% | 0.112 / 0.133 | 88% |
| extent | sets | arkit42 | 97 | 78% | 89% | 0.095 / 0.113 | 82% |
| extent | one_set | calib | 99 | 69% | 92% | 0.056 / 0.176 | 90% |
| extent | one_set | arkit47 | 55 | 64% | 93% | 0.037 / 0.098 | 91% |
| extent | one_set | arkit42 | 53 | 74% | 98% | 0.053 / 0.141 | 96% |
| position | sets | calib | 207 | 100% | 100% | 0.545 / 0.678 | 90% |
| position | sets | arkit47 | 126 | 88% | 99% | 0.650 / 0.796 | 54% |
| position | sets | arkit42 | 34 | 68% | 94% | 0.474 / 0.586 | 91% |
| position | one_set | calib | 63 | 95% | 95% | 0.646 / 0.852 | 89% |
| position | one_set | arkit47 | 36 | 75% | 97% | 0.416 / 0.528 | 56% |
| position | one_set | arkit42 | 25 | 56% | 100% | 0.456 / 0.611 | 96% |
| angle | sets | calib | 13 | 100% | - | 3.041 / - | - |
| angle | sets | arkit47 | 4 | 100% | - | 13.730 / - | - |
| angle | sets | arkit42 | 2 | 100% | - | 13.031 / - | - |

## End to end: cards made with the calibrated u (run mvp2-accuracy-003); TUM held out, ARKit in-sample

| family | view sets | sequence | n | coverage by the card's u | median u (m or deg) | median error |
|---|---|---|---|---|---|---|
| height | sets | arkit42 | 67 | 99% | 0.222 | 0.180 |
| height | sets | arkit47 | 233 | 99% | 0.246 | 0.053 |
| height | sets | tum | 397 | 98% | 0.223 | 0.079 |
| height | one_set | arkit42 | 45 | 100% | 0.292 | 0.230 |
| height | one_set | arkit47 | 53 | 96% | 0.232 | 0.056 |
| height | one_set | tum | 110 | 95% | 0.224 | 0.089 |
| extent | sets | arkit42 | 97 | 91% | 0.120 | 0.060 |
| extent | sets | arkit47 | 262 | 95% | 0.145 | 0.041 |
| extent | sets | tum | 484 | 95% | 0.133 | 0.031 |
| extent | one_set | arkit42 | 51 | 94% | 0.104 | 0.045 |
| extent | one_set | arkit47 | 46 | 89% | 0.069 | 0.032 |
| extent | one_set | tum | 102 | 85% | 0.124 | 0.034 |
| position | sets | arkit42 | 34 | 100% | 0.615 | 0.436 |
| position | sets | arkit47 | 120 | 99% | 0.835 | 0.244 |
| position | sets | tum | 204 | 100% | 0.730 | 0.170 |
| position | one_set | arkit42 | 24 | 100% | 0.714 | 0.426 |
| position | one_set | arkit47 | 31 | 97% | 0.522 | 0.254 |
| position | one_set | tum | 64 | 95% | 0.907 | 0.216 |
| angle | sets | arkit42 | 2 | 100% | 13.031 | 0.856 |
| angle | sets | arkit47 | 3 | 100% | 14.536 | 2.081 |
| angle | sets | tum | 13 | 100% | 3.041 | 0.836 |

| sequence | window | call | cameras | objects | cards v1 | cards v3 | judgements v3 | DA3 stage | GPU peaks GiB |
|---|---|---|---|---|---|---|---|---|---|
| arkit42 | 0 | warm | 7.2 | 13.4 | 13.4 | 19.0 | 19.0 | 1.68 | 56.8/52.0 |
| arkit47 | 0 | warm | 15.6 | 20.7 | 24.6 | 39.1 | 40.9 | 7.13 | 60.4/53.1 |
| arkit47 | 1 | warm | 14.5 | 24.4 | 29.2 | 52.7 | 58.3 | 7.27 | 60.1/52.3 |
| tum | 0 | first | 23.9 | 36.6 | 42.0 | 72.2 | 80.1 | 8.99 | 60.3/52.7 |
| tum | 0 | warm | 18.7 | 30.1 | 36.7 | 65.5 | 73.0 | 8.76 | 60.1/52.7 |
| tum | 1 | warm | 9.4 | 18.7 | 22.8 | 38.5 | 40.4 | 3.3 | 58.2/52.5 |

Bounds ('at least' / 'at most', cut by the frame edge): 100 of 115 hold within u against GT.
