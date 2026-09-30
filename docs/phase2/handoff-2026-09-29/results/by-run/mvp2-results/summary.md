# Click MVP round 2, integrated (mvp2/integrate): results

Branch `mvp2/integrate` (worktree `/Users/adam/.codex/worktrees/panoptes-phase2-video-mvp2-integrate`), from `mvp/integrate`, with
`mvp2/judge`, `mvp2/identity`, `mvp2/physical`, `mvp2/accuracy` and `mvp2/click` merged in that order, then fixed in place
(general causes only, no per-video case). Everything runs on 2 x A100-SXM4-80GB (MPS), ephemeral `modal run`, retries 0,
min_containers 0. Times are analysis time: seconds from the MP4 bytes in the container to the Volume commit; cold start is
separate. Metres are at estimated scale (floor plane + an assumed 1.6 m camera height) unless marked 'true scale'. Labels on
clicks, names and judgements are agent-made from contact sheets, not ground truth.

One command per video (a first call after boot, then a warm call; Gemini names and hazard questions relayed through the
deployed report container, as `scripts/name_video_entities.py` does, so no key leaves it):

    cd /Users/adam/.codex/worktrees/panoptes-phase2-video-mvp2-integrate
    PYTHONPATH=. /Users/adam/Desktop/Tesla/panoptes-platform/.venv/bin/python scripts/fast_report_bench.py \
        --out RUNS/mvp2-integrate-<site>-NNN --sites <me340|samsclub-a2|walmart> --plan first,warm --mirror-max-mb 8 --no-gpu-eval

Ground-truth sequences (ARKitScenes 47333932 two windows, 42445448, TUM fr1 room two windows; display layers off):

    PYTHONPATH=. .../modal run modal_apps/fast_report_app.py::accuracy --plan RUNS/mvp2-accuracy-inputs/plan-003.json --out RUNS/mvp2-integrate-gt-NNN
    PYTHONPATH=.:scripts .../python scripts/accuracy_gt.py score RUNS/mvp2-integrate-gt-NNN --inputs RUNS/mvp2-accuracy-inputs

This page: `scripts/mvp2_results.py` (tables.md, summary.json) plus the labelled audits in this folder.

## Open the viewer

    cd /Users/adam/.codex/worktrees/panoptes-phase2-video-mvp2-integrate
    PY=/Users/adam/Desktop/Tesla/panoptes-platform/.venv/bin/python; R=/Users/adam/Desktop/panoptes-public/research-notes/phase2/runs
    PYTHONPATH=. $PY -m fast_report.layers serve $R/mvp2-integrate-me340-007/mirror --port 8793 \
        --replay $R/mvp2-integrate-me340-007/mirror mvp-me340-e84efffd-1790690775 --as me340-mvp2 --speed 1
    cd web && npx vite --host 127.0.0.1        # second terminal; proxies /fast to 8793 (web/node_modules: a symlink to the platform's)
    open "http://127.0.0.1:5173/app.html#/live/me340-mvp2"

Sam's Club: `mvp2-integrate-samsclub-a2-007` / `mvp-samsclub-a2-d5e0c855-1790691934`; Walmart: `mvp2-integrate-walmart-007` /
`mvp-walmart-c0761a2a-1790692552` (warm calls). `--speed 1` replays the layers at their recorded times (`--speed 4` for a quick
look). Click anything in the video: the card says what it is (name, who named it, alternatives), its physical values (each
+-u, 'estimated' on every metre; 'not observed' / 'not measurable' / a bound, with the reason) and its judgements (verdict,
the measured value against the threshold, the picture's p(yes) and reason). The mirrors keep blobs under 8 MB only: the
full room and the splat stay on the Modal Volume and the replay leaves those two layers out.

## What changed while integrating (general causes only)

Merge decisions:
- **Final name decides everything (R1):** `cards.apply_name` (identity) is the one source; the judge's own re-derivation
  (`judge.follow_name`) now defers to it. physical's size check (long objects not observed whole, the measured size's u) and
  review marks (visible length) moved into `apply_name`; accuracy's no-floor rule gives it no size to check.
- **Judge:** the judge branch's run (Gemini hazard judge, Qwen fallback that can only veto a PASS) with click's version guard
  (runs overlap; an older run never puts over a newer one), provisional priority and set-of-marks rendered in the process pool.
- **People:** physical's measured picture rejection (`cards.person_geometry`, the not-a-person card) instead of click's
  extent rule; click's nearest-keyframe pick and chunked pick layer kept.
- **u:** accuracy's ground-truth rule (k_geo per family and view-set state, scale 0.25 apart) also sets the drawn box's u.
- **Bench:** one command, both Gemini relays (names and hazard questions) on by default.

Fixes found by running the merge (each with an assert self-check):
- **Stale judgements (R1 regression):** cards v4 is put twice (first-pass names, then densify's). Both runs carried version
  4, so the older run, finishing last, put stale rows: 25 rows on checks the final class does not apply (Sam's Club,
  round 001). Judge runs are now ordered by the cards put (`put_order`). Result: 0 such rows in every final call.
- **Identity timing:** densify's naming pass waited for the first pass's names; it now starts at cards v3 on its own queue
  partition (the two passes had shared one and could take each other's answers). The first pass's leftovers are named
  beside it.
- **Naming load:** every request was copied at 15 s (42 of 44 on Sam's Club). The doubled load slowed the report container
  to 16-24 s a request and queued copies in the relay, where both copies of one request were lost. Copies now go only to
  a pass's tail (<= 25 % out after 15 s), to small passes (<= 8 requests) at 15 s, or at 24 s. Nothing is copied while
  nothing is back.
- **Slow service vs lost request:** a pass with more than half its requests out waits up to 60 s instead of 30 s. On Walmart
  006 every request took 32-34 s; at 30 s the pass gave up and the Qwen decider named all 538 objects.
- **Process pool:** a worker that died in one call (`cards.v1_box`, Sam's Club 003) failed every later call. A broken pool
  is now recreated before the next call.
- **Numbers that say nothing:** a size whose +-u reaches zero is now the bound 'at most value + u'. Under the ground-truth
  u rule, small tools and goods 2-5 m away had read e.g. 'width 0.04 +- 0.07 m' on 51-70 % of widths. A distance whose
  +-u is >= max(value, 1 m) is 'not measurable' (far figures had read 'nearest object 0.00 +- 20 m').
- **J2 false FAIL:** a stable racked pallet of water (Sam's Club obj-0-6, first calls) FAILed on a side overhang of
  0.33 +- 0.12 m with a straddling front. The judge's own stated rule is now enforced: a side reading FAILs only when the
  front itself breaks the limit. Committed after the final runs (4ed7782). Replaying the judge offline on run 007's cards
  turns that row into NEEDS_REVIEW, and no other verdict changes.
- **Bench/viewer check:** the browser check waits for every recorded judgements patch (the last run is on cards v4) and
  aims at each check's most severe verdict.

Plugged or not:
- **X10 discovery, words part only (`--discover`, off):** SAM 3 catch-all + label words in wave 1, measured on all three
  videos (runs 004). The held-out click audit did not move: correct / miss 21 / 9, 28 / 19, 22 / 21, the same as without.
  One Sam's Club miss became a price-tag hit, and one hit became a miss. It added 11-100 objects and 0-4 naming requests.
  Final judgements landed at 159-183 s on Sam's Club against 84-113 s without, but that run also had a slow host (boot
  imports 50 s). On Walmart they landed at 101-111 s against 81-104 s; on ME340 there was no change. The flag stays
  off. X10's OWLv2 boxes + SAM 2.1 were not tried: they need two more resident models, and the words part already showed
  no coverage gain for its time. The retail coverage gap (goods, rack structure and hanging items under 19-21 of 60
  random clicks) stays open.
- **Kept from the branches:** X1 densify, X7 primitives, X12 12 m/s gate (one keyframe step), Gemini names (identity),
  Gemini hazard judge (judge), accuracy's per-shot DA3-GIANT any-view geometry.

## Reading the numbers

- Final runs: `mvp2-integrate-{me340,samsclub-a2,walmart}-007` (one boot each, a first call after boot, then a warm call; run
  one at a time: three benches at once shared the report container and slowed every Gemini request to 30 s, round 001),
  ground truth `mvp2-integrate-gt-003`. Earlier rounds 001-006 are the debugging trail; 004 is the `--discover` variant.
- **Time to facts:** first 3D, objects, pick and cards v1 land within 18-46 s. Identity for the first-pass objects
  lands at 53-71 s. Densify's own objects are named at 73-94 s, and the final judgements follow one hazard wave later
  (10-19 s).
  - Warm: 102 / 84 / 81 s for ME340 / Sam's Club / Walmart.
  - First call: 107 / 107 / 83 s.
  - The ~90 s target is met on retail warm and missed on ME340: densify's SAM 3 (cards v3 at 70 s) and its naming pass
    (17-20 s) are the long pole.
  - Variation between runs of the same code is large. ME340's final judgements were 80.6 s warm on run 006 against 102.2 s
    on run 007, and Sam's Club's 110 s against 84 s. The causes are the host (boot imports 17 s against 34-54 s; Sam's
    Club's first call ran on a slow host: cold start 420 s, cards v1 46 s) and Gemini's latency.
- **Display layers:** ME340's warm calls accept no SAM 3D model at all: 0 of 30 tries in all 7 integrated warm calls,
  against 3-4 on all 7 first calls. The cause was not found; the ranking puts thin cables first, and most fail prepare or
  the fit gate. The splat preview lands at 218-223 s, under the
  230 s target, after the facts or at 72 s.
- **Identity:** the fresh held-out items were agent-labelled blind by mvp2/identity before this branch existed, and no rule
  here was tuned on them. ME340's names vary from run to run: right / right-or-close was 0.73 / 0.87 on run 007, 0.75 /
  0.80 on 006 and 0.70 / 0.80 on 003.
- **Physical, ground truth:** only TUM is held out. k_geo was fitted on the two ARKit sequences by mvp2/accuracy; these runs
  are new. As delivered, the error is dominated by the assumed 1.6 m camera height: the true heights were 1.23-1.51 m,
  so arkit42 reads about 22 cm high on tops.
  - One-view-set extents are the weak family: 0.80-0.94 coverage, 0.87 on TUM.
  - The 'at most' bounds introduced here hold within u on 92-93 % against GT: 90/98, 28/30 and 220/236.
  - The 'at least' bounds hold on 28/29, 11/11 and 42/49.
- **Physical, delivered reports:** these are same-object pairs matched by image boxes, so this is agreement, not truth.
  Walmart keeps its 8-10 cm floor offset against its delivered report.
  - The people rows are the automatic ones and match mvp2/physical's (the code path is unchanged). Its agent-labelled
    analysis (on-floor feet median +0.02 m; 22 % of 'feet seen' actually hidden) still applies.
- **Clicks:** the held-out seed-2 random clicks were labelled on round-1 outputs before any mvp2 run. Here a label carries
  when it cannot differ: nothing picked in both runs (the label is about the scene), picked before and nothing now
  (correct -> miss), or the same region picked (IoU >= 0.5). The 24 ME340 clicks whose picks changed were looked at on
  this branch's sheets.
  - Precision of what gets picked is 0.88-1.00.
  - The gap is coverage: 9 / 19 / 21 of 60 clicks land on things no entity covers.
  - ME340's 3 background hits are burned-in subtitles.
- **Judgements:** audit labels are agent-made and blind to verdicts. They come from three sources: carried from mvp2/judge's
  labels (same id and place), carried from this branch's pre-final runs, and made here on blind sheets (215 rows).
  - The rule constants were tuned by mvp2/judge on these same videos, and the hazard cuts are held out by video.
  - Over both calls: 405 of 405 audited PASS right (+5 unverifiable ground-bay J1 tops), 0 false PASS.
  - FAIL 3 of 4 right. The wrong one is the J2 row fixed after the runs (above). All three right FAILs are J4 cables on
    ME340's floor and mat.
  - J2, J3a and J8 still decide nothing (J2: no unstable-stack example to calibrate the picture; J3a's foot cue is
    unvalidated).
  - Objects with a check: 10 / 30 / 10 %.

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

## Spend

**Modal list-price upper bound over each container's life (GPU queue waits included):** about $31.

| Runs | Modal ($) |
|---|---|
| ME340 001-007 | 3.57, 1.23, 1.41, 1.40, 1.24, 1.21, 1.41 |
| Sam's Club 001-004, 006, 007 | 1.62, 1.26, 0.30, 1.79, 1.41, 2.47 |
| Walmart 001-004, 006, 007 | 1.60, 1.20, 1.40, 1.39, 1.75, 1.31 |
| ground truth 001-003 | 0.59, 0.58, 0.59 |
| **Sum** | **30.73** |

- A Sam's Club 005 bench was stopped after boot; its container is not in the sum (under $1, estimated).
- ME340 001's $3.57 includes a long wait for GPUs.

**Gemini (gemini-3.5-flash, through the deployed report container):**

| Requests | Input tokens | Output tokens |
|---|---|---|
| Naming | 20.6 M | 1.8 M |
| Hazard questions | 5.1 M | not counted |

- This is about $12 at Gemini 2.5 Flash list prices (the repo records no price for gemini-3.5-flash), an estimate.

**Total: about $43.** The cost ledger was not edited.

## Open issues

- **Time to facts on ME340:** final judgements at 102-107 s, over ~90 s. Densify's SAM 3 (cards v3 at 70-74 s) plus its
  naming pass (17-20 s) plus one hazard wave (10-13 s). A faster path would name densify's objects progressively, or
  skip naming the new objects densify finds on the last keyframes.
- **Gemini runs through the bench's relay:** both relays run in the local bench process, into the deployed report
  container. Its latency depends on that container's load: 30 s a request with three benches at once, 12-15 s alone. A
  product needs the calls server-side with the key in ATM/Infisical and its own concurrency.
- **Run-to-run variance:** host and Gemini swing identity by ±10 s and final judgements by ±20 s between runs of the same
  code. ME340's names swing too: right-or-close 0.80-0.87 across runs 003-007.
- **The J2 side-overhang fix** landed after the final runs. It was verified by an offline replay on run 007's cards, not
  by a bench.
- **Retail click coverage:** 19-21 of 60 random clicks land on goods, rack structure or hanging items with no entity.
  X10's words did not help; its box + SAM 2.1 path is untried.
- **J2, J3a and J8 decide nothing, and J6 was never triggered.** People's feet stay unvalidated for J3a: 22 % of 'feet
  seen' are hidden (mvp2/physical).
- **SAM 3D on ME340's warm calls:** 0 accepted in 7 of 7, against 3-4 in 7 of 7 first calls.
- **One-view-set extents:** they cover the true values only 80-94 % of the time (0.87 on held-out TUM).
- **Web live-report check:** `web/tests/live-report-check.ts` needs a fixture folder (`runs/fb-c-fixture-001`) that is no
  longer on disk. The typecheck passes and the browser check passes on all three final runs.
