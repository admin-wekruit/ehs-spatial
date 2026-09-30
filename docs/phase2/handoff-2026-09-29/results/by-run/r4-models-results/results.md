# r4 models: every object card gets a display model

User acceptance (2026-09-29): most objects are recognised at least by type, and every object comes with segmentation, physical info and a model. This round builds the model part. Judgement is paused, hazard VLM questions stay off, and the VLM comes last.

Branch: `r4/models` (from `mvp3/integrate`, with `r4/physical` merged). Runs: `r4-models-bench-003` is the final code (3 videos, first and warm call, `--judge off --display on`, no VLM, SAM 3 words only). `r4-models-bench-001` ran the code before the review, and `r4-models-bench-002` was an experiment stopped after one call (see SAM 3D below). The baseline for the cards' time is `r4-physical-bench-001` (same videos and calls, display off). Labels by eye are agent-labelled.

## What was built

- **The fitter.** `fast_report/display_model.py` fits primitives to the card's floor-frame points:
  - a gravity box (X7's yaw search, p2/p98 extents like the card's box);
  - a cylinder, vertical or along the points' long axis;
  - a plane slab on the points' PCA axes;
  - an open frame: posts at the footprint corners (2 when flat) and a plate at each height where the points pile up.
- **Choosing the shape.** The card's type picks the kind: pipes, drums and bottles become cylinders; doors, boards and signs become planes; shelves, racks, ladders, carts and tables become open frames. The type is overruled only when its shape fits clearly worse (more than 2x the best residual and 3 cm more). Without a type, the box is the default. The plane replaces it below 0.7x the box's residual, and the cylinder below 0.5x, but only on a circle fitted over at least 90 deg of seen arc.
- **Seen and guessed faces.** A face (or a cylinder's 10 deg sector, or a frame member) is "seen" when a camera faced it and at least 2 % of the points lie on it. Otherwise it is "guessed" and drawn faint.
- **One-sided objects.** An object seen from one side is as deep as its visible part, a lower bound. It gets no invented thickness, and the card says so.
- **Pose.** The pose is in the shot frame.
- The self-check covers each fitter, the choice rule, the pose round trip and degenerate clouds.
- **Where the model lives.** Fits are stored name-free in `card.raw`. `cards.apply_name` draws `card.model` for the current name, so a rename re-shapes it. Every object card gets one, including on-demand click cards (one view, so one side).
  - Each card also carries its SAM 3D eligibility, `well_observed`: at least 3 views at least 15 deg apart, not cut by the frame edge, 0.2-3 m, at least 0.12 rad across at its nearest view, one cluster, not an open frame.
- **SAM 3D.** It runs s1cfg12 on GPU 0 after the facts, as before, but only on the well-observed cards, best observed first. It is gated by the unchanged complete_video_objects gate on held-out views. The `models` layer now says per object what SAM 3D did (`tried`: accepted, rejected with the gate's reasons, or not generated with the prepare reason).
- **Viewer.**
  - New `faces` and `sectors` primitives with per-face alpha. One model per object: SAM 3D's accepted mesh, else the card's model, else the old box.
  - The selected model shows in full, the others faint. The selected object's model is also shown alone in the 3D pane's corner (the viewer's own preview capture), opening its shot.
  - The card has a Model block: kind, source, how it was chosen, residual, seen share, one-side note, and SAM 3D's result (accepted with its held-out IoU, rejected with reasons, not generated, not tried with the eligibility reason).
  - Check: `web/tests/r4-models-browser.mjs`.
- **Scripts.** `scripts/r4_models.py` makes the contact sheets (crop | model from the same camera, on the card's own best view), the tables and a self-check.

## Measurements (bench 003, final code)

| video | call | object cards | with a model | box / cylinder / plane / open frame | by type (overruled) | residual median / p90 cm | seen share (median) | one side | SAM 3D eligible / tried / accepted | first model / final models s (display start) | analysis s | GPU peaks GiB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| me340 | first | 261 | 261 (100%) | 167 / 38 / 45 / 11 | 65 (8) | 1.93 / 5.7 | 0.33 | 207 | 60 / 30 / 2 | 81.2 / 140.0 (59.0) | 230.1 | [61.2, 53.49] |
| me340 | warm | 254 | 254 (100%) | 175 / 33 / 35 / 11 | 45 (10) | 1.94 / 6.67 | 0.33 | 202 | 62 / 30 / 0 | - / 140.7 (55.9) | 230.3 | [66.8, 53.4] |
| samsclub-a2 | first | 623 | 623 (100%) | 553 / 34 / 32 / 4 | 33 (11) | 1.56 / 3.37 | 0.33 | 618 | 116 / 30 / 5 | 56.3 / 125.3 (39.5) | 227.8 | [69.08, 60.38] |
| samsclub-a2 | warm | 624 | 624 (100%) | 557 / 34 / 29 / 4 | 33 (11) | 1.56 / 3.34 | 0.33 | 619 | 117 / 30 / 5 | 58.9 / 129.9 (40.5) | 228.3 | [67.75, 60.08] |
| walmart | first | 719 | 719 (100%) | 577 / 46 / 58 / 38 | 58 (37) | 1.69 / 2.91 | 0.33 | 682 | 169 / 30 / 6 | 90.5 / 113.3 (34.9) | 226.7 | [61.92, 57.22] |
| walmart | warm | 721 | 721 (100%) | 582 / 47 / 57 / 35 | 55 (39) | 1.69 / 2.9 | 0.33 | 685 | 175 / 30 / 6 | 91.5 / 114.9 (32.9) | 226.4 | [63.04, 57.36] |

- **A model on every object card.** 100 % in all 6 calls, across detector-word and unidentified cards (`models-by-identity.json`). People and "not a person" cards get none.
- **Card contract.** No card contract violations on any cards version (`completeness-bench-003.json`).
- **Accepted SAM 3D meshes, per call on the cards.** 2 / 0 / 5 / 5 / 6 / 6, that is 0-0.8 % of the object cards (24 of 180 tries). This is the same rate as mvp3's EHS-first ranking (26 of 180 in mvp3-integrate). Choosing the well-observed cards did not raise acceptance: the gate's "source-view silhouette/depth" test rejects most. The first mesh does land sooner: 56-91 s against 97-183 s in mvp3-integrate on the same videos.
- **GPU peaks.** Per GPU, at most 69.08 GiB on GPU 0 and 60.38 GiB on GPU 1. No stage went over 72 GiB, and the bench raised no flags.
- **Cold start (not analysis time).** 90.3 s in the container, 199.6 s from submit to ready.

### Time added to the cards (bench 003 against r4-physical-bench-001, same calls; s)

| video | call | cards.v1 | cards.v3 | model fits CPU s (16 processes) | cards v3 put | cards bytes (all versions) |
|---|---|---|---|---|---|---|
| me340 | first | 1.63 (base 1.84) | 2.53 (base 1.97) | 3.84 | 58.9 (base 57.1) | 4.2 MB (base 3.1) |
| me340 | warm | 1.69 (base 1.68) | 2.16 (base 1.59) | 3.78 | 55.8 (base 52.6) | 4.2 MB (base 3.1) |
| samsclub-a2 | first | 4.36 (base 4.24) | 3.63 (base 2.68) | 9.11 | 39.2 (base 38.0) | 9.9 MB (base 7.4) |
| samsclub-a2 | warm | 5.36 (base 4.77) | 3.30 (base 2.61) | 9.16 | 39.6 (base 35.4) | 9.9 MB (base 7.4) |
| walmart | first | 4.02 (base 3.67) | 3.27 (base 2.84) | 9.52 | 34.6 (base 42.2) | 10.2 MB (base 7.5) |
| walmart | warm | 3.60 (base 3.61) | 3.06 (base 2.69) | 9.19 | 32.6 (base 34.6) | 10.2 MB (base 7.6) |

- **Cards build.** Each build takes 0-0.6 s longer for v1 and 0.4-1.0 s for v3. The fits cost 3.8-9.5 CPU s spread over the pool (about 10-15 ms a card).
- **Card size.** The cards JSON grows by about 35 %, from 1.6 KB of fits per card.
- **Cards put times.** The put times move with run-to-run noise, from -8 s to +4 s against the baseline run.
- **Models.** They start after cards v3. The first accepted mesh lands at 56-91 s and the models layer is final at 113-141 s. The call still ends at 226-230 s, on the splat preview, as before.

### Audit by eye: 30 random models per video (warm call; each on the card's own best view)

| video | looked at | plausible | implausible | unclear |
|---|---|---|---|---|
| me340 | 30 | 19 | 7 | 4 |
| samsclub-a2 | 30 | 27 | 3 | 0 |
| walmart | 30 | 26 | 3 | 1 |

- **Accepted SAM 3D meshes** (every one in the sheets: ME340 first call, Sam's Club and Walmart warm calls): 11 of 13 plausible. They include the ME340 control panel and a Walmart boot and clog with their shapes and colours. Of the 2 misses, 1 lies off its region and 1 is not in view from the audit camera.
- **Rule used.** Plausible means the model sits on the object's region at no more than about 2x its size, with a sensible shape. Implausible means off the region, over about 2x its size, or covering only a small part. Unclear means a fragmented, tiny or frame-cut region. Sheets and per-item notes are in `audit-*/` and `sam3d-accepted-*/`.
- **Where models go wrong.** Nearly all implausible models inherit the card's own point cloud:
  - close-ups (ME340's tool holders and gears, Walmart's shoes), where the pooled points of many views do not line up in one view;
  - thin things whose visible depth reads large;
  - long pipes where only part was lifted.
- **Checking the placement.** Over all 1178 region/keyframe pairs of bench 001's ME340 first call, the card box projects onto its mask with a median offset of 1 px. So the placement is right on average, and the misses are per-view misregistration.

### Viewer

`web/tests/r4-models-browser.mjs` was run on the bench-003 mirror: Walmart with a SAM 3D shoe, a display-rack frame, a cylinder, a plane and a box, and ME340 with the control panel mesh and a workbench frame. In all 7, the model line said "generated, display only" and SAM 3D's result, the 3D pane redrew when the object was selected, and the model appeared alone in the pane's corner. There were no page errors. Screenshots are in `browser-*/`.

## SAM 3D in the background: tried and turned off (bench 002)

While the splat trains, GPU 0 idles after SAM 3D's first pass, which is judged by 113-141 s. Bench 002 ran the gate's existing background mode until 205 s (other views, seed 43, then the rest of the eligible cards). On ME340's first call it made 81 tries on 43 objects and got 3 meshes accepted, against 2 of 30. But the generations queued before the deadline still ran: the models finished at 255 s, the call took 26 s longer, and GPU 0 peaked at 70.7 GiB. The bench was stopped after that call and the mode was turned off (`MODELS_UNTIL_S = None`).

## Spend

Bench 001 cost $3.28, bench 002 (one call) $1.07 and bench 003 $3.47, for about **$7.82** in total. These are list-price upper bounds from the bench, not invoices.

## Known gaps

- On-demand click cards get a model record and the card line, but no drawing in the 3D pane.
- On shelves full of goods the open frame is often overruled for a box: 35-39 of the Walmart shelves and racks.
- The "seen" test can understate what was seen for very noisy clouds. One Walmart shoe had every face marked guessed.
- `web/tests/live-report-check.ts`: its default fixture (`fb-c-fixture-001`) is missing, and its people-track assertion fails on the mvp3 mirror. Both are pre-existing. The r4 block before them passes.
