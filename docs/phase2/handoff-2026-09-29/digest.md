# Panoptes phase 2 digest: video → clickable 3D object layer (research, experiments, state) — 2026-09-29

Companion to `HANDOFF.md` in this folder (mission, user decisions, hard rules, next steps). This file is the evidence base:
what is built (A), what each technology gave us (B), every experiment (C), round-by-round acceptance (D), dead ends (E),
review findings still open (F), and what this digest did not read (G). Compiled read-only at ~22:45 on 2026-09-29, while the
round-5 integration (`r5b/integrate`, workflow `H/workflows/r5d.js`) was still running (only `runs/r5b-int-me340-001/mirror/`
existed, empty).

**Path roots.** `R/` = `/Users/adam/Desktop/panoptes-public/research-notes/phase2/`; `runs/` = `R/runs/`;
`H/` = `R/handoff-2026-09-29/`; `ledger` = `R/video-mvp/cost-ledger.json` (key named); `W/` =
`/Users/adam/.codex/worktrees/panoptes-phase2-video-r5b-integrate/` (code; docs in `W/docs/phase2/`).

**Conventions.** Videos: ME340 workshop / Sam's Club / Walmart (bench sites `me340`, `samsclub-a2`, `walmart`, 30 s clips);
"a / b / c" triples are always in that order. "warm (first)" = warm call (first call after boot). Times = analysis seconds
from the MP4 bytes in the container to the Volume commit (`written_s`) unless stated; cold start is separate. Metres are
"estimated" (floor plane + assumed 1.6 m camera height) unless "true height". "By eye" = agent-labelled, not ground truth.
$ = Modal list-price upper bounds unless stated. **(inferred)** marks my own inference.

---

## A. System as built (branch lineage fb → mvp → mvp2 → mvp3 → r4 → r4b → r5b builders → r5b/integrate)

### A.1 Hardware, entry points, clock

- One Modal container, app `panoptes-fast-report`, class `FastReport`: `gpu="A100-80GB:2"`, 32 CPU, 160 GiB, 24-process
  pool, 16 gate processes, `retries=0`, `timeout=3600`, `max_containers=1`, MPS daemon started before any CUDA context;
  `profile` parameter default `internal` (RecGen) vs `commercial` (SAM 3D) (`W/modal_apps/fast_report_app.py` l.41, 163-200).
- **Jev-Omni** runs in a separate class `Jev` on its own A100-80GB (4 CPU, 48 GiB), never on the pipeline's two GPUs
  (`W/modal_apps/fast_report_app.py` l.87-106; `W/fast_report/jev.py`).
- Entrypoints: `main` (a video), `accuracy` (GT plans), `visits` / `visit_dev` (revisits), `setup` (same file l.558-743).
  Bench: `W/scripts/fast_report_bench.py --sites <site> --plan first,warm[,shifted] --judge off --display on --vocab <src>
  --mirror-max-mb 8` → `call-*.json`, `eval-*.json`, `mirror/` (`runs/mvp-results/summary.md`; `H/HANDOFF.md` §4).
- Clock and memory: `W/fast_report/instrument.py` (Clock, Vram, `run.json`); a stage's peak is the device peak in its window;
  > 72 GiB (90 % of 80 GiB) is flagged; nvml and torch agree to 0.01 GiB under MPS (docstring, `runs/fb-d-harness-vram-001`).
- Ephemeral `modal run` only, `min_containers=0` in all research runs (every results page above states it). FAST-PATH-PLAN
  wanted `min_containers=1` for a live demo because queueing for 2 GPUs took 367-631 s in 2 of 3 submits
  (`H/scratch/fast-round1-synth.md` §1 E5d, §2).

### A.2 What sits on which GPU (boot order, `FastReport.boot`, `W/modal_apps/fast_report_app.py` l.170-262)

| device | resident |
|---|---|
| GPU 0 (`cuda:0`, `dev_geo`) | DA3-GIANT-1.1 (host copy kept; offloaded while generators run, `da3.restore` after); SAM 3 (bf16); **internal**: 2 RecGen-FAST worker processes (`recgen_fast.recgen_worker`); **commercial**: 2 SAM 3D processes (s1cfg12); coverage: OWLv2 + SAM 3 tracker head |
| GPU 1 (`cuda:1`, `dev_seg`) | vLLM Qwen3-VL-8B (loads first, outside MPS: `VLLM_MPS=False`); SAM 3 (bf16); naming encoders DINOv2-L, PE-Core-L, YOLOE-26L; PE-Core text vocabulary (5 MB); RAM++ (own venv); gsplat worker; SAM 3 point tracker for on-demand clicks (+0.9 GB); OWLv2 + tracker head; r5b-integrate: a second RecGen pair parked in host memory until its first job |
| own A100 | Jev-Omni (router's decider), Modal class `Jev`, scale-down 300 s (`runs/r5b-models-results/results.md`) |
| CPU | 24 spawn processes (cards, fits, surfaces, timelines), 16 gate processes (held-out-view gate, own venv), thread pools |

Boot timings on one r4 container: imports 16.97 s, DA3 on GPU 0 31.31 s, SAM 3 39.15 s, vLLM ready 72.84 s, SAM 3D load
51.6 s + warm 14.6 s (20.29 GB reserved after warm-up), coverage models 95.71 s (`runs/r4-coverage-results/results.json`
key `boot`). Cold start by round: fb 131-147 s (`W/docs/phase2/CLICK-MVP-SPEC.md` §1.1); mvp1 128.7-141.8 s
(`runs/mvp-results/summary.md`); mvp3 91.1-146.9 s (`runs/mvp3-results/tables.md`); r4 models 90.3 s in container, 199.6 s
submit→ready (`runs/r4-models-results/results.md`); r4b 157-182 s (`runs/r4b-results/tables.md`); r5b/time 102-110 s
(`runs/r5b-time-results/results.md`); r5b/visit 108 s (`runs/r5b-visit-results/results.md`).

### A.3 Pipeline stages in order (latest measured times: r5b/vocab final warm runs, display off, unless another source is named)

| # | stage | model / method | where | code | warm s (ME340 / Sam's / Walmart) |
|---|---|---|---|---|---|
| 1 | decode, shot cuts, 5 fps keyframes | decode + `detect_shot_cuts` (overlay-edge rule) | CPU | `core.py`, `scripts/detect_shot_cuts.py` | decode end 6.4 / 3.5 / 3.1 (`runs/r5b-vocab-results/final.md`) |
| 2 | cameras + depth, per shot | DA3-GIANT-1.1 any-view, 504×280, cam head, one forward per shot | GPU 0 | `core.py` (`Da3`) | in "first 3D" |
| 3 | floor + scale | floor plane; scale from the assumed 1.6 m camera height; a shot without a floor → every metric "not measurable" | CPU | `core.py`, `cards.py` | — |
| 4 | room | GPU TSDF 3 cm (Open3D CUDA) + points; light room first, full later | GPU 0 | `core.py` | first 3D 19.7 / 12.9 / 12.0 (`final.md`) |
| 5 | people | SAM 3 {person, floor} batched on 5 fps keyframes; `ehs_spatial.live_people.PeopleLoop`; containment dedupe; 12 m/s speed gate (X12) | both GPUs + CPU | `core.py` | people 20.4 / 18.0 / 14.7 in fb (`CLICK-MVP-SPEC.md` §1.1) |
| 6 | words | wave 1 = EHS core + the site's cached words (known at t = 0); wave 2 = **RAM++** tags kept when they are nouns of the PE word list (r5b default, 9-16 words, 2.4-2.9 s); Qwen scene vocabulary only as opt-in | GPU 1 | `vocab.py`, `vlm.py`, `segment.py` | `runs/r5b-vocab-results/final.json`, `extra-runs.json` |
| 7 | objects v1 | SAM 3 detect on object keyframes, one priority queue over both GPUs; dedupe; lift (eroded, depth-tail trim, int64 codes); instances (seams, part/whole) | both GPUs + CPU | `segment.py`, `instances.py` | 27.3 / 20.4 / 19.0 |
| 8 | outlines + pick v1 | keyframes "segmented", in-between keyframes "projected" (E6b pair rule); pick maps `panoptes-pick-v1` | GPU + CPU | `segment.py`, `layers.py` | ≈ objects + 2-4 (`runs/mvp2-results/summary.md`) |
| 9 | cards v1 | points first; u rule; time; name = SAM 3 word ("detected word, unverified") | CPU pool | `cards.py` | 33.9 / 26.3 / 26.4 |
| 10 | tier 0 models + parts | observed-surface TSDF mesh per card (one GLB per shot); primitives only for fixed-shape classes passing overflow ≤ 35 % / cover ≥ 60 %; planar parts + angles ±u | CPU pool | `surface.py`, `display_model.py`, `observed.py` | cards v1 + 2.5-4.7 (`runs/r5b-models-results/results.md`) |
| 11 | types, first pass | naming cascade: SAM 3 word, YOLOE-26L, PE-Core-L zero-shot, DINOv2-L bank k-NN; accept on two independent votes; otherwise clusters → one Qwen3-VL-8B question per medoid; else "<family> (type only)"; else "unidentified (<shape>)" | GPU 1 | `cascade.py`, `vlm.py` | 48.6 / 55.0 / 54.6 |
| 12 | densify + coverage | SAM 3 on every 5 fps keyframe (X1); OWLv2 objectness ≥ 0.1 boxes → SAM 3 tracker masks on the shared backbone; objects v3, pick v2, cards v3; densify naming pass | both GPUs | `core.py`, `coverage.py` | objects v3 56.4 / 57.5 / 52.9; cards v3 60.6 / 67.3 / 62.5; types (densify) 66.9 / 78.8 / 74.4 |
| 13 | timelines | ORB content windows (co-visibility 0.40) per shot; per card: per-window states, per-interval values, change flags with evidence | CPU | `windows.py`, `timeline.py` | +0.3-0.9 s wall (`runs/r5b-time-results/results.md` §3e) |
| 14 | tier 1 generated models | `route.py`: fixed-shape class list → Jev-Omni Q5 (P(none of the simple shapes) > 0.5 → generated); RecGen FAST (internal) / SAM 3D s1cfg12 (commercial); held-out-view gate; look-alike reuse; ≤ 60 groups; no new generation after 135 s | GPU 0 (+ GPU 1 pair in r5b-integrate) + Jev GPU | `route.py`, `recgen_fast.py`, `recgen_models.py`, `sam3d.py`, `r5_bench.py` | tier 1 done: internal 146 (first) / 106 / 118; commercial 189 warm, 199 first / 123 / 171 (`runs/r5b-models-results/results.md`) |
| 15 | splat preview | gsplat, E5b recipe, 150 s budget, shot frames | GPU 1 | `splat.py` | 223.3 / 222.6 / 222.7 (`runs/r4b-results/tables.md`) |
| 16 | visits (optional, `visit_of`) | DINOv2-L retrieval → DA3 any-view registration + register_cut_shot gate → Hungarian matching → see-through test | GPUs + CPU | `visits.py` | visits stage 10.9-17.3 (`runs/r5b-visit-results/results.md`) |
| 17 | on-demand click | SAM 3 tracker point prompt → lift → one-view card → cascade name (one Qwen question last, shared by look-alike clicks) | GPU 1 | `ondemand.py` | p50 1.11-1.15, p95 1.46-1.67 (`runs/mvp3-results/tables.md`) |
| — | judgement (PAUSED) | J0-J9 rules, VLM decider, Gemini/Qwen hazard judge | CPU + GPU 1 | `judge.py`, `hazard.py`, `calibration.py` | off (`--judge off`) |
| — | event captions | Qwen3-VL-8B window captions | GPU 1 | `vlm.py`, `modal_apps/video_events.py` | off in r5b: its prompt asks for PPE/safety (`runs/r5b-time-results/results.md`) |

Round-4 full-display timings for comparison (warm): cards v1 43.1 / 30.1 / 31.2, types (densify) 97.8 / 87.5 / 93.1, models
final 181.2 / 178.7 / 179.2, call end 230.2 / 228.5 / 226.7 (`runs/r4b-results/tables.md`). GPU peaks: r4b ≤ 70.2 GiB
(same); r5b/models max 70.3 / 66.4 (`runs/r5b-models-results/results.md`); r5b/time > 72 GiB on GPU 0 in 5 of 20 retail
calls (neutral vocabulary, Sam's Club 35k densify masks) (`runs/r5b-time-results/results.md`); RAM++ ∪ PE union 71.54
GiB and PE-only 73.06 GiB with 9+ stages over 72 (`runs/r5b-vocab-results/extra-runs.json`).

### A.4 Layers, patch store, viewer

- Store: Modal Volume `panoptes-fb-layers`; `blobs/sha256/<hex>` (content-addressed, never rewritten);
  `reports/<report>/patches/<seq:06d>-<layer>.json` (one per put); `run.json`; `sites/<site>/vocab.json`; the mirror adds
  `written.json`, `served.json`. Every layer version is complete (cumulative); the viewer takes the newest
  (`W/fast_report/layers.py`). Patch schema `panoptes-fast-patch-v1`; one writer thread, one Volume commit per batch;
  endpoint on 127.0.0.1:8793 `GET /fast/reports/<id>/patches?after=<seq>` and `GET /fast/blobs/<hex>`; `#/live/<id>` polls
  every 500 ms (`W/docs/phase2/FAST-BUILD-SPEC.md` §7).
- `run.json` fields: `schema`, `report`, `site`, `video`, `hardware`, `boot`, `stages[{stage, where, start_s, end_s, s,
  peak_gb, over_90, n}]`, `gpu_peak`, `flags`, `layers[{sent_s, written_s, served_s_unix}]`, `usd_estimate`
  (`FAST-BUILD-SPEC.md` §8). fb-era settings: keyframe = sharpest of every 6 frames (5 fps); SAM 3 80 (frame, word) pairs
  per forward, score ≥ 0.3, no per-word cap; SAM 3 cost ≈ 0.05 s + 0.0075 s per word per frame; vLLM share 0.35 of GPU 1
  (≈ 28 GB) in the spec vs 0.45 in E5 (`FAST-BUILD-SPEC.md` §1, §4-5, §12).
- Layers seen in results: `video`, `cameras`, `room` (light|full), `people`, `events`, `objects` (v1, v3), `outlines`,
  `pick` (v1, v2), `object_cards` (v1-v4), `judgements`, `models`, `surfaces` (r5b), `splat`, `visits` (r5b)
  (`W/fast_report/layers.py`; `CLICK-MVP-SPEC.md` §0; r5b results pages).
- Write cost: a patch write 1.53-2.80 s, readable after 2.70-5.55 s; 26.5 MB GLB and 52 MB splat readable within 4.6 s
  (E8b, `H/scratch/fast-round1-synth.md` §1). ~600-700 small GLBs per report cost 10-13 s of commit → one GLB per shot
  (`H/scratch/round5-notes.md`).
- Viewer: `W/web/src/LiveReport.tsx`, `live-report.ts`, `viewer/native-viewer.ts`, `viewer/splat-layer.ts`,
  `VisitsPanel.tsx`; page `app.html#/live/<report>`; served by `python -m fast_report.layers serve <mirror> --port 8793`
  (+ `--replay ... --speed N`) and vite proxying `/fast` (`runs/mvp-results/summary.md`; `H/HANDOFF.md` §4 uses 8794/5185).
  Click → card p50 5.4-7.1 ms, p95 26.1-36.3 ms over 200 clicks; pick v1 decode 89-107 ms at load, pick v2 702-2554 ms
  after densify (`runs/mvp-results/summary.md`; `runs/mvp2-results/summary.md`).
- Shown-model priority (r5b): generated (and look-alike copies) > checked primitive > the card's node of its shot's
  observed-surface GLB > see-through box; 3D labels = the card's shown name/type; lettered planar parts
  (`runs/r5b-models-results/results.md`). Per-interval drawing of moved objects and a timeline bar (r5b/time), Visits tab
  (r5b/visit).
- Browser checks: `W/web/tests/r4-models-browser.mjs`, `r5-surfaces-check.ts`, `r5b-models-browser.mjs`,
  `r5b-time-shots.mjs`, `r5b-visits-browser.mjs`; `live-report-check.ts` needs `runs/fb-c-fixture-001`, missing since mvp2
  (every round reports it).

### A.5 Card contract and switches

- Card: what it is (name, who named it, alternatives), kind, physical fields in the shot's floor frame (`position_xy`,
  `top_above_floor`, `base_above_floor`, `height`, `width`, `depth`, `principal_axis_tilt_deg`, `planar_slope_deg`, planar
  parts), each a value ±u with evidence level and scale label, or a bound ("at least"/"at most") or a status with reason;
  time; model record (`W/fast_report/cards.py` docstring; `CLICK-MVP-SPEC.md` §0.1, §4).
- One name decides everything: `cards.apply_name` derives class, kind, size check, model and time labels from the final
  name; raw name-free measurements stay in `card.raw` (`runs/mvp2-identity-results/summary.md`, R1).
- Core options: `judge`, `display`, `events`, `vocab` (`ram` default; `qwen`/`pe`/`taxonomy`/`yoloe-pf` comparisons),
  `coverage`, `bank_save`/`bank_write` (False = read-only naming bank), `visit_of`/`visit_site`, `fixture_dump`, profile
  (`runs/r5b-visit-results/results.md` "Options"; `runs/r5b-time-results/results.md`; `W/fast_report/vocab.py`).
- Measurement scripts (`W/scripts/`): `fast_report_eval.py`, `accuracy_gt.py` (GT harness), `r4b_results.py`,
  `r5b_models.py`, `r5b_render.py`, `r5b_timeline_eval.py`, `r5b_width_gt.py`, `r5b_plant.py`, `x6_plant.py`,
  `x6_evaluate.py`, `r5b_visits_eval.py`, `r5b_vocab.py`, `route_jev.py`, `r5_models.py`; `r5b_align.py` and
  `r5b_quick.py` belong to the running integration (overlay alignment, ME340 quick report) (inferred from names and
  `H/HANDOFF.md` §6).

---

## B. Technologies: what each does for us, numbers, verdict

Verdict key: **USE** (in the pipeline), **OPTIONAL** (switch / comparison / later), **DROPPED**.

**Geometry**

- **DA3-GIANT-1.1 any-view (poses + depth in one forward per shot)** — USE. 504×280 cam head: forward 5.66 s for 112
  keyframes (9.18 s for 150), 17.9 GB allocated / 22.8 GB reserved; photometric reprojection 1.20× / 1.16× (neighbours /
  122 held-out frames); ATE 0.043 m vs the bar 0.057 m (DROID TUM 0.038 m × 1.5); rotation error median 0.57°, max 1.73°.
  672×378: 14.85 s, 1.10×, ATE 0.063 m (`H/scratch/fast-round1-synth.md` §1 E1). On GT: ATE 0.053 / 0.077 / 0.044 m
  (arkit42 / arkit47 / TUM); at the true camera height tops/bases 1.8-3.3 cm, heights 1.7-2.9 cm, widths 1.7-3.4 cm,
  positions 6-13 cm median |error| (`runs/mvp2-accuracy-results/summary.md`). Licence CC BY-NC: demo/internal only
  (`W/docs/phase2/FAST-PATH-PLAN.md` §6).
- **DA3-BASE (commercial candidate)** — DROPPED: forward 1.04 s, 6.1 / 7.3 GB, but photometric 1.84× / 1.74×, ATE
  0.16-0.17 m (`fast-round1-synth.md` §1). MapAnything-apache and VGGT-1B-Commercial were never tested (same §2).
- **DROID-SLAM** — DROPPED from the fast path. At the 5 fps keyframes it loses track on 4 of 5 GT windows (ATE 0.65-0.92 m);
  where it tracks (TUM w1, 2.8 cm) cards are no better than per-shot DA3 (tops 1.9 vs 2.0 cm); 30 fps TUM w0: ATE 0.172 m in
  47.6 s (`runs/mvp2-accuracy-results/summary.md`). Still the reference camera for planted-event scoring
  (`runs/r5b-time-results/results.md` §3a).
- **Windows + Sim3 stitch geometry (X6 option B)** — DROPPED: stitch scales swing 0.6-2.0 per link; ATE 18 / 39 / 7 cm vs
  per-shot 5 / 8 / 4 cm; tops at true height 15 / 185 / 3 cm (`runs/mvp2-accuracy-results/summary.md`).
- **GPU TSDF + GPU lift (E7)** — USE: TSDF 3 cm 0.49-0.54 s warm (0.93 s first), ~880k points, 1.6 M triangles; lift + voxel
  merge 0.10 s for 5,915 masks (`fast-round1-synth.md` §1).
- **Assumed 1.6 m camera height** — USE for now (user decision, `H/HANDOFF.md` §2). True heights 1.23-1.52 m on GT
  (s = 0.77-0.94), so delivered tops read +4 to +22 cm high; "the scale assumption is the whole of the remaining error"
  (`runs/mvp2-accuracy-results/summary.md`). Scale term in u = 0.25 |value| (stated 1.2-1.8 m range) (same).
- **u rule and calibration k** — USE. mvp1: k from warm vs +5 s shifted window (round 2) height 1.071, extent 1.345,
  position 1.024, angle 1.0; round 5 re-measured 1.028 / 1.0 / 1.001 / 1.0 in-sample (`runs/mvp-results/summary.md`).
  mvp2/accuracy rule: u = sqrt((k_geo · sqrt(Σ non-scale²))² + (0.25 |v|)²), k_geo height 1.0 / 1.41, extent 2.39 / 4.52,
  position 2.23 / 4.33 (≥ 2 view sets / one set), fitted on ARKit only; TUM held out: heights 98 % / 95 %, extents 95 % / 85 %,
  positions 100 % / 95 % covered (`runs/mvp2-accuracy-results/summary.md`). Planar-part angle u uses k 1.75
  (`runs/r5-models-results/results.json` notes). Files: `W/fast_report/calibration.json`, `calibration.py`.
- **GT harness (ARKitScenes 47333932 two windows + 42445448 LiDAR; TUM fr1-room two windows, mocap + Kinect)** — USE. Only RGB
  goes to the pipeline; GT per card = the card's own pick regions lifted with GT depth and poses into a GT floor frame; (a)
  as delivered vs (b) at the true camera height (`runs/mvp2-accuracy-results/summary.md`; `W/scripts/accuracy_gt.py`).
  Visits add TUM fr1 desk/desk2/xyz/360/plant/teddy and ARKit 47333931 on Volume `panoptes-r5b-visit`
  (`runs/r5b-visit-results/results.md`). Limit: 3 indoor scenes (homes, one office), no warehouse or shop
  (`runs/mvp2-accuracy-results/summary.md` "Open").
- **ORB content windows (X6)** — USE for time: co-visibility share of textured grid cells with ≥ 2 verified matches,
  threshold 0.40; top/bottom 20 % ignored (burned-in captions); unrelated far pairs share 0.25-0.36 with 1 match
  (`W/fast_report/windows.py` docstring, `runs/fx-x6-windows-time-001`).
- **LingBot (-Map)** — DROPPED from the object layer. X3: "not needed" (`CLICK-MVP-SPEC.md` §0, §1.2); FAST-PATH-PLAN
  kept it as a 3-6 min background lane for dense points, an independent cross-check and inferred-floor validation; licence:
  not commercial until written confirmation (`FAST-PATH-PLAN.md` §4, §6). X3 spent 6.62 USD
  (ledger `fine_fast_experiments_result`).
- **Inference policy** — USE as the rule for anything inferred: infer only what a physical reason supports and a test
  against the video could refute; observed / inferred-from-structure / inferred-by-model / unobserved always shown apart;
  ME340 floor: test noise 0.37 %, raised-plane control flagged 53.8 %, 187.7 m² floor of which 57.9 m² seen
  (`W/docs/phase2/INFERENCE-POLICY.md`).

**Segmentation and discovery**

- **SAM 3 (image, text prompts)** — USE. Batched {person, floor} 0.066 s/frame (loop 0.160); 20 EHS words 0.191 s/frame at
  34 GB peak (loop 0.941); text encoding once per vocabulary (95 words 0.76 s). Recall of the delivered named objects
  (IoU ≥ 0.5): 51 % with 20 words, 73 % with 40, 81 % relaxed, 85 % upper bound with the 80 delivered names — the gap is
  the vocabulary, not the detector (`fast-round1-synth.md` §1 E2). Generic words flood masks (E2b caveat,
  `W/fast_report/segment.py` docstring). SAM License: no military, nuclear or ITAR use (`FAST-PATH-PLAN.md` §6).
- **SAM 3 tracker (box / point prompts on the shared backbone)** — USE for coverage masks and on-demand clicks; shared
  backbone vs the tracker's own pixels: mask IoU median 0.993 (n 16) (`runs/r4-coverage-results/results.json` `probe`).
- **SAM 3.1 video** — OPTIONAL (background movers). Stride 3 propagation 9.9 fps (8.1 with init), 27.8 s on the walking
  shot, stride 1 100 s, stride-3 vs stride-1 mask IoU median 0.995; "pallet jack" gave 23 false detections; no true
  non-person mover on ME340 (`fast-round1-synth.md` §1 E3b-c).
- **SAM 2 / 2.1 AMG ("everything")** — DROPPED from the fast path. E6: keyframe AMG + reprojection 2.0× / 4.9× / 9.4× faster at
  5 / 2 / 1 fps with IoU 0.80 / 0.73 / 0.66; video propagation 0.8-1.5×, IoU 0.75-0.83; AMG itself is unstable between
  frames (ceiling 0.80-0.83). Side result kept: batched box prompts 0.64 s vs 20.7 s for 526 boxes (32×, IoU 0.9985)
  (`fast-round1-synth.md` §1). X10 (OWLv2-L + SAM 2.1-L): own-look recall 0.69 → 0.85 for +16.5-22.5 s GPU and two more
  resident models; precision never audited (`runs/mvp-results/summary.md`).
- **OWLv2 (large ensemble, objectness)** — USE (coverage source, `COVERAGE_SRCS = ("owlv2",)`). r4 probe on 96 click misses /
  117 background clicks: covers 52/96 misses at 9/117 background; detect 0.0495 s/frame + decode 0.0451; 0.91 GB weights;
  Apache-2.0. In the pipeline (score ≥ 0.1), real-object clicks opening the right card: 0.75 → 0.825 / 0.514 → 0.676 /
  0.727 → 0.909 (off → on); cost +11.8 / +11.8 / +13.9 s to cards v3 (`runs/r4-coverage-results/results.json` keys `probe`,
  `audit`, `added_warm_on_minus_off_s`).
- **YOLOE-26L / YOLO11x / YOLO26x (ultralytics, AGPL-3.0)** — OPTIONAL, internal only. r4 probe misses covered: YOLOE text
  prompts 11/96, YOLOE prompt-free 17/96 (19/117 background), YOLO11 2/96, YOLO26 1/96 (`runs/r4-coverage-results/results.json`).
  YOLOE-26L is one vote in the naming cascade (`W/fast_report/cascade.py`); YOLOE prompt-free as the word source typed
  fewest held-out items right (ME340 19/42, Walmart 12/27) (`runs/r5b-vocab-results/compare.md`).
- **RAM++ (Recognize Anything Plus, Apache-2.0)** — USE: default wave-2 word source since r5b, chosen by a rule written before
  results were read (`runs/r5b-vocab-results/selection-rule.md`). Final: 0 VLM calls before cards v1 (was 3), held-out family
  0.60 / 1.00 / 0.79, clean delivered objects covered 73/105, 79/91, 58/74 (r4b: 80, 79, 61)
  (`runs/r5b-vocab-results/final.md`). Note: X10 rejected RAM++ as a *discovery stage* (own-look recall 0.30 / 0.39 alone,
  0.46 s/frame, tags are scene words) (`runs/fx-x10-discover-all-001/results.json`); r5b uses it only as a word source whose
  tags must be nouns of the PE word list (`W/fast_report/vocab.py`).
- **PE-Core-L (zero-shot)** — USE as a naming vote; OPTIONAL as a word source. As word source it found ME340's workshop words
  (held-out right 22-29 of 43-44) but missed Walmart's shoes/boxes (9/22) (`runs/r5b-vocab-results/compare.md`,
  `extra-runs.json`). RAM++ ∪ PE on ME340: clean coverage 81/105, held-out 0.571, cards v1 47.2 s, GPU 0 71.54 GiB
  (`extra-runs.json`). Routing as a zero-shot router: 0.722 (`runs/route-jev-001/results.md`).
- **SigLIP 2** — DROPPED as namer: X2 zero-shot called slippers and sign letters "spill" (`CLICK-MVP-SPEC.md` §1.2); 0.642 as a
  router (`runs/route-jev-001/results.md`). (Timeline docstring still mentions SigLIP 2 embeddings for association.)

**Naming, deciding, VLMs**

- **Naming cascade (X13 → r4/naming)** — USE (`W/fast_report/cascade.py`). Cheapest first: SAM 3 word share, YOLOE-26L
  class, PE-Core-L zero-shot, DINOv2-L k-NN over earlier VLM answers of other videos of the same domain family; accept when a
  second independent vote agrees at thresholds fitted on the family's other videos (`naming_calibration.json`, folds by
  site); a family with no other audited video accepts nothing cheaply (ME340 at the pooled 0.80 point agreed with the VLM
  only 0.23); the rest clustered (DINOv2-L, average linkage) with ONE Qwen question per medoid, copied only to members whose
  SAM 3 word gives the same class (docstring). X13 spent 0.66 (naming) + 0.81 (hazards) USD (ledger `vlm_light_cascade_result`).
- **DINOv2-L label bank** — USE (read-only). Caveat: shared mutable state on the Volume (grew 1798 → 1998 during benches;
  r4b review finding 7) → `bank_write False` in r5b builders (`H/workflows/r5b.js` l.11; `runs/r5b-time-results/results.md`).
  Also the visits retrieval descriptor (`W/fast_report/visits.py`).
- **Qwen3-VL-8B via vLLM (GPU 1)** — USE, last and rarely. Events (E5c): 4.44-4.63 s warm vs 23.2 s with transformers on the same
  GPU, 4/4 events matched; load eager 39 s, compiled 144 s; settings `enforce_eager`, `gpu_memory_utilization 0.45`
  (`fast-round1-synth.md` §1-2). As identity decider (option-letter log-probs) dev study 0.64 / 0.80 right / right-or-close vs
  Gemini 0.86 / 0.92 (`runs/mvp2-identity-results/summary.md`); Qwen-only mvp3 identity 0.36 / 0.55 on ME340
  (`runs/mvp3-results/tables.md`). X2: as a yes/no judge it says yes 94 % of the time; red outlines make it say "fire
  extinguisher" (`CLICK-MVP-SPEC.md` §1.2). Today: 20-31 cluster-medoid naming requests per video (`runs/r5b-vocab-results/final.md`)
  plus, accepted by the user on 2026-09-29, one scene-vocabulary request per video (`H/scratch/round5-notes.md`).
- **Gemini (gemini-3.5-flash through the deployed report-workspace container)** — DROPPED. Best names measured: dev 0.86 / 0.92
  (`runs/mvp2-identity-results/summary.md`), held-out 0.81 / 0.88 over 119 items in mvp2 (`runs/mvp2-results/summary.md`);
  hazard judge bal. acc 0.83, AUROC 0.85 on X8 set d (`W/fast_report/hazard.py` docstring). Cost: 20.6 M input tokens for
  naming + 5.1 M for hazards ≈ 12 USD est.; 12-15 s per request alone, 30 s with three benches (`runs/mvp2-results/summary.md`).
  Dropped because no named Modal secret exists (the relay ran through the local bench) and the user wants VLM last
  (`runs/mvp3-results/summary.json` key `secret_route`; `H/HANDOFF.md` §2-3).
- **Jev-Omni** — USE as the display-model router only, on its own GPU. X8 as decider: cross-fitted Brier 0.175 vs calibrated Qwen
  0.138, ECE 0.129, ~22 GiB that GPU 0 did not have → not swapped in (`runs/mvp-results/summary.md`). Router (route/jev,
  held out by video, ~313 labelled cards): Jev Q5 raw 0.875 (COMPLEX precision 0.83 / recall 0.74, ECE 0.075) vs best rules
  0.712, zero-shot 0.802, Qwen3-VL-8B q5 0.837; cascade class → Jev Q5 raw 0.904 (post-hoc class list; honest 0.815-0.875);
  0.103 s per question; RecGen load 38 / 0 / 111 calls per video (`runs/route-jev-001/results.md`); 0.088 s per question in
  batches of 8; first request 48-60 s before warm-up, 3-5 s after (`W/fast_report/jev.py`).
- **Set-of-marks / decider calibration (mvp1)** — judgement decider ECE failed the ≤ 0.10 bar (q4 0.183, q5 0.255)
  (`runs/mvp-results/summary.md`). Paused with judgement.

**Display models**

- **Observed-surface mesh (tier 0)** — USE (default shown model). 0.014-0.025 s per object median (p90 ≤ 0.136); by eye on 90
  objects 38 right / 52 partial / 0 wrong (`runs/r5-models-results/results.json`; `runs/r5-models-bench-001/sheets/labels.json`).
  r5b rim growth: outline coverage 0.60-0.67 → 0.84 / 0.91 / 0.90, silhouette IoU median 0.53 / 0.67 / 0.69, overflow up
  (ME340 0.19 → 0.36) (`runs/r5b-models-results/results.md`).
- **Primitives (box, cylinder, plane, open frame)** — USE for simple structure only. X7: param box 58/87 accepted on the
  held-out gate, plane 2/4, shelf 0/4; 44/58 box fits used one view; bootstrap CIs > 10× too small (`CLICK-MVP-SPEC.md` §1.2).
  r4: a primitive on 100 % of cards, residual median 1.56-1.94 cm, by eye plausible 19 / 27 / 26 of 30
  (`runs/r4-models-results/results.md`); r5 bench by eye 7 / 41 / 42 (`labels.json`). r5b: primitive only when the class fixes
  the shape and the fit passes overflow ≤ 35 % and cover ≥ 60 % (`runs/r5b-models-results/results.md`).
- **RecGen (TRI; TRELLIS-image-large fine-tuned; code non-commercial, weights CC-BY-NC-4.0)** — USE (internal profile,
  user decision 2026-09-29). Default 6.3-8.4 s/object. FAST setting: SS flow 12 steps CFG 5, SLAT flow 8 steps CFG off,
  mesh-only decode with vertex colours, ≤ 4 views, batched view-pair fusion with one shared CFG negative, DINOv2 once per image,
  posed mesh from GPU tensors, skip `save()` (9-24 s): 2.53-2.57 s vs 8.2-8.4 s (A100 PCIe), 1.96 vs 6.32 s (SXM4); acceptance
  9.33 vs 10.00 of 19 (gate noise: seed flips 6/19); 2 A100 → 20 / 40 / 70 objects in 19 / 36 / 62 s with 2 processes per GPU
  (~20 GiB each); torch.compile of SS +1.17× after 208 s compile (`H/scratch/round5-notes.md`); 2 processes per GPU +17 %, 3
  +32 % (`H/workflows/recgen-fast2-alternatives-stopped.js` l.9). By eye on 90 objects 58 / 27 / 5 at ~7 s
  (`runs/r5-models-bench-001/sheets/labels.json`; `runs/r5-models-results/results.json`). In r5b: Walmart 8 right / 1 partial
  vs r4b boxes 0 / 7 / 2; ME340 machines, tools, benches pass neither generator (0-1 of 20-23 tried); only 0-9 generated per
  video (`runs/r5b-models-results/results.md`). Known loss: thin pallet obj-0-71 (`round5-notes.md`).
- **SAM 3D Objects** — OPTIONAL (commercial profile). E4: the plan's fast mode 3.45 s/try but 4/30 accepted (FAIL); s1cfg12
  3.49 s/try, 7/30 (PASS within noise); 2 processes per GPU 1.14× without MPS, 1.63× with (2.09 s effective, 53.5 GB)
  (`fast-round1-synth.md` §1). Gate CPU was the bottleneck (prepare 46 s/object, assess 23 s) (same §0). In the pipeline:
  24 of 180 tries accepted (0-0.8 % of cards), 11/13 accepted meshes plausible (`runs/r4-models-results/results.md`); ME340
  warm calls accepted 0 of 30 in 7 of 7 calls (`runs/mvp2-results/summary.md`); r5 bench by eye 29 / 36 / 8 (+17 none),
  gen 4.4-5.1 s + gate 1.5-2.0 s (`runs/r5-models-results/results.json`; `labels.json`).
- **TRELLIS-image-large (MIT) / TripoSR (MIT)** — DROPPED. By eye TRELLIS 16 / 36 / 38, TripoSR 7 / 59 / 24; TripoSR 1.0 s gen but
  7.1-8.4 s gate; TRELLIS 4.0-4.4 s gen (`runs/r5-models-results/results.json`; `labels.json`).
- **Affostruction (MIT, posed multi-view RGB-D)** — untested; flagged "worth a test" (`H/scratch/round5-notes.md`); the
  recgen/fast2 alternatives workflow was stopped (`H/workflows/recgen-fast2-alternatives-stopped.js`).
- **Gaussian splats (gsplat)** — OPTIONAL display layer only. E5a 5k/7k steps: 159-262 s, 25.3-26.6 dB vs delivered 30.82 dB
  (FAIL); half size 56-81 s, 23.4-23.8 dB. E5b feed-forward (DA3's GS head): 14.5-17.2 dB, ghosting, CC BY-NC (FAIL)
  (`fast-round1-synth.md` §1). E5b follow-up recipe: 36.5 → 11.5 ms/step, ME340 120 s → 26.94 dB on 84 held-out frames; preview
  needs 150 s to stay above 27.0 dB (`W/fast_report/splat.py` docstring). A splat crop as an object model: by eye 2 / 31 / 26
  (+31 none) (`labels.json`).

**People, time, visits**

- **People (PeopleLoop)** — USE. E3: 5 fps path vs the full layer median 0.007 m, p90 0.022, max 0.136; R1/R2/R3 tick agreement
  100 / 100 / 96.5 % (`fast-round1-synth.md` §1). mvp2: real person clicks open the person 0.98 / 0.89 / 0.67; pictures of
  people opened as a person 0 / 4 / 0; 22 % of "feet seen" are hidden (`runs/mvp2-results/summary.md`). X12: 5 fps enough;
  12 m/s gate kept (`runs/mvp-results/summary.md`; `runs/mvp2-results/summary.md`).
- **Timelines / free-space place test (X6 → r5b/time)** — USE. Association: centroids within 3σ,
  σ = sqrt(0.04² + (0.05 z)²), or robust-box IoU ≥ 0.2; "disappeared" only by the see-through test (≥ 2 views)
  (`runs/mvp-results/summary.md`; `W/fast_report/timeline.py`). Results in D.
- **Visit registration (register_cut_shot method)** — USE (r5b/visit): registration gate centre residual ≤ 0.1 of camera
  spread (≥ 0.25 m), rotation ≤ 3°, depth-ratio spread ≤ 15 %, floors within 5° / 0.2 m (`runs/r5b-visit-results/results.md`).
- **MPS** — USE: needed for 2 generator processes per GPU (1.63× vs 1.14×, E4); vLLM kept outside MPS (`VLLM_MPS=False`).
- **Legacy import (`import_video_scene`)** — DROPPED: 920-1186 s, 47 % exact shapely outline unions on full meshes, 38 % mesh
  simplification; display-mesh outlines 35 s vs 562 s (`fast-round1-synth.md` §1 E8a) → patch store (E8b).

**Pre-fast-path background (2026-09-17 … 09-27; the hand-built "P3" pipeline).** Roots: `D/` = `W/docs/phase2/`,
`N/` = `/Users/adam/Desktop/panoptes-public/research-notes/`, `S/` = `H/scratch/`.
- DA3 posed (GIANT-1.1, `mono_room.py`): fr1/room 177 keyframes median depth error 3.1 %, 88 % within 10 %, 117 s A100,
  scale vs DROID 0.995; 352 views: support 90.7 %, 2.6 %, 90 %; BASE 3.2 % / 82.5 %, and on moving people 14.8 % error vs
  GIANT 4.8 % (`D/HANDOFF.md`, `D/PLAN.md` §2b). Depth-edge pixels (2.1 %) carry 12 % median error; confidence filtering
  deleted correct walls/doors → rejected; far views read 5-7 cm deeper (`D/HANDOFF.md`). DA3 licence: GIANT/Nested/Large CC
  BY-NC; Base/Small/Metric-Large/Mono-Large Apache-2.0 (`N/2026-09-17-video-scene-algorithms.md`).
- DROID-SLAM full clip: fr1/room 1362 frames 197 s, ATE 0.0408 m, but its dense mesh fell into 388 pieces (53.6 % of rays
  hit); 640×480 did not help; Lightning lost track at frames 460-630 → cameras only (`D/VIDEO-MVP.md`, `D/HANDOFF.md`,
  `D/ARCHITECTURE.md` §7). Code BSD-3, weights licence unverified (`S/stream-dossier.md`, `S/m2-design.md` §6).
- ORB-SLAM3: monocular fr1/room 0.729 m global error; RGB-D with dynamic masks 0.861 → 0.0153 m ATE; GPLv3 → dropped. ORB
  survives in `detect_shot_cuts.py`: 20/20 hard cuts, 20/20 15-frame fades, 17/20 30-frame fades, 18/20 jump cuts, 0 false
  positives, 34 ms/frame (`D/VIDEO-MVP.md`; `D/COMPONENTS.md` C1).
- MapAnything: plain RGB ATE 1.842 m (8 keyframes) / 0.641 m (32); with DROID cameras it re-predicts intrinsics (fx off
  32-39 %) → `rejected_camera_contract_drift`; default weights CC BY-NC, an `-apache` variant exists (`D/README.md`,
  `D/VIDEO-MVP.md`, `N/2026-09-17-video-scene-algorithms.md`).
- MoGe-3: single-frame depth made ghost layers (rejected); scale estimate 1.332 vs GT 1.334 on fr1/room; kept for scale
  cross-check, moving-pixel depth and FOV (`D/HANDOFF.md`; `D/ARCHITECTURE.md` §3).
- LingBot-Map: walking 22/23 flow-check pairs; fr1/room 5-9 of 13-19 pairs, below the 75 % gate; ME340 dense map 2.91 M points,
  93.6 % of the walk's view for 0.28 USD; H100 12.3-12.5 fps, A100 7.1 fps, but 16-24 min CPU post-processing; live test
  drifted within 10 min; code Apache-2.0 since ed0aee8b, weights carry no licence (`D/HANDOFF.md`, `D/ARCHITECTURE.md` §7,
  `D/STREAMING-PLAN.md` §4, §11, `S/stream-dossier.md`).
- SAM 3.1 self-hosted "person": IoU 0.975, recall 98.0 %, precision 99.3 %; fal `sam-3-1/video-rle` merged both people into one
  mask with no track id → fal retired; live SAM 3.1 kept only 90 % consistent ids; SAM 3 on L4 323 ms/frame (`D/HANDOFF.md`,
  `D/VIDEO-MVP.md`, `D/STREAMING-PLAN.md` §11).
- SAM 2 AMG "everything" over 352 views: 517 entities, 297 confirmed, 90.4 % of room surface vs 28.3 % with word prompts,
  ~2.5 s/view (`D/HANDOFF.md`; `S/stream-dossier.md`).
- SAM 3D Objects on full frames (tight crops flipped objects, depth off 12 %; full frame raised a CNC panel IoU 0.56 → 0.81);
  final hand-built models per scene (SAM 3D / RecGen / box): ME340 22 (9/11/2), Sam's 67 (41/5/21), Walmart 40 (17/14/9)
  (`D/ARCHITECTURE.md` §7). SAM 3D Body: optional/legacy (83 of 118 keyframes passed) (`D/VIDEO-MVP.md`; `D/COMPONENTS.md` C9).
- RecGen hand-built: gate silhouette IoU ≥ 0.65, depth median ≤ 4 %, p95 ≤ 10 %; 10.55 / 3.84 / 9.22 USD and 1.1-3.1 h per
  clip, ~80 % of all spend at the time (`D/HANDOFF.md`; `S/stream-dossier.md`).
- Splats hand-built: ME340 1.64 M Gaussians 31.33 dB held out (30.82 with elongation cap 4); Sam's / Walmart 27.23 / 31.07 dB;
  19-20 min per H100 run; LingBot-seeded splats rejected (holes, floaters) (`D/ARCHITECTURE.md` §7; `S/stream-dossier.md`).
- Cuts and lenses: edit cuts (ME340 {14, 226}, Sam's {420}, Walmart {383}) were the root cause of earlier failures; per-shot
  lens fixed Sam's 60.1° → 55.3° (floor inliers 72.7 → 96.5 %); Walmart's 0.884 fails the ≥ 0.90 inlier gate; the cut-away
  register gate passed ME340 at 0.92 % / 2.38° (`D/ARCHITECTURE.md` §7; `S/m2-design.md`). Docs disagree on ME340's cut-away
  lens (56° vs 95° vs 51.3° vs 80.57°) (same sources).
- 1.6 m assumption history: user decision 2026-09-20; +19.9 % off on fr1/room vs 0.1 % on an ARKit clip; the live workbench
  defaults to 1.5 m treated as calibrated (blocker A0) (`D/PLAN.md` §0.1; `D/ARCHITECTURE.md` §1, §5).
- Older architecture: six coexisting paths (P0a photo upload, P0b Gradio workbench, P1 fixed-camera video R1-R3, P2 video-mvp
  retired 2026-09-21, P3 video → report by hand, platform with 16 Postgres tables + ZEN rules); target: capture → geometry →
  one frame → memory store → rules (four-state verdicts) (`D/ARCHITECTURE.md` §1-2; components C1-C10 in `D/COMPONENTS.md`).
- Streaming M0 (2026-09-27, 4.97 USD): live ARKit map p50 0.67 s / p95 1.14 s; SAM 3 L4 works at 1 fps; SAM 3.1 and LingBot per
  frame not feasible; box auto-review not good enough (6/10 dropped, 27/32 kept); M1 stopped by the user
  (`D/STREAMING-PLAN.md` §11; ledger `streaming_m0_probes_result`, `streaming_m1_result`).
- Researched, never run: DA3-Streaming (non-commercial GIANT + GPL SALAD), MASt3R-SLAM, SLAM3R, CUT3R, StreamVGGT, MonoGS
  (non-commercial), WorldMirror 2.0, DPVO, VGGT-SLAM 2.0 (`N/2026-09-17-video-scene-algorithms.md`; `S/stream-dossier.md`).

**Research memos (conclusions; `R/` originals, each ends with a verifier/corrector section that overrides the body).**
- **Prior art, fine vs coarse** (`R/fine-expand-prior-art-2026-09-28.md`): other systems use models for perception and hand-set
  rules for every decision, so our explainable rule layer is mainstream (§0; corr.: ESAM and MoonSeg3R learn association). An
  object = a SAM 3 track lifted to 3D passing spatial, semantic and temporal/multi-view votes (§1.2, §3.2). Mobility in four
  classes: fixed (TSDF), movable rigid (own submap, persistent/absent/unobserved), deformable (polyline per window, never in the
  TSDF, re-detected each window), actors (tracks) (§3.4). 24 starting thresholds, all to be recalibrated on our footage (§2).
  fine_voxel = max(tolerance/2, z/f_eff); at DA3 504 px f_eff ≈ 262 px, 1 px ≈ 1.9 cm at 5 m, 3 cm voxels lose pixel support
  beyond ~8 m (§3.1, estimate). Change detection prior art is imprecise (Khronos precision 21.3-31.3 %, SceneDiff AP 10.6-22.8 %),
  so "moved/absent" are candidates to confirm (§1.4). Industry EHS products are 2D zones + thresholds (§1.5). Biggest risk: DA3-GIANT
  licence; only DA3-BASE/SMALL are Apache-2.0 and output poses (§4; corr.).
- **Reuse of photo-workcell judgement** (`R/reuse-judgement-memo-2026-09-28.md`): the workcell's `measure_observed_points`
  (pure numpy) was never called by the video path — the cheapest gap (§0.2); fragmentation and misses (ME340 13 of 20 covered
  objects in pieces; Sam's covered 14-25 of 67) (§0.3, corr.); tilt from video not usable yet (door 6.8-7.2°, shelf 12.9° vs 1.4°)
  (§0.4); evidence levels E0 2D only / E1 support points / E2 coarse layout / E3 held-out model (§2.1); per-interval judgement
  scheme: one fact per (object, static interval), median over windows, u = hypot(window spread, σ), any covered interval FAIL →
  FAIL (§3.4); proposed V1-V6 pass bars unvalidated (§5.2). "safety fence" 0 detections vs "barrier" recall 0.97 (§1.1).
- **VLM-light** (`R/vlm-light-memo-2026-09-29.md`): swap parts inside `cascade.py`, not add models (§0). Held-out agreement with
  Gemini's class: SAM 3 word + bank neighbour 0.88 (331 cards), + PE-Core/OWLv2 zero-shot 0.76 (538), SAM 3 word alone 0.28;
  cluster answers copied to verified members 0.95 vs 0.42 unverified; bank alone across site types 0.15-0.25; writing cheap names
  back into the bank dropped agreement to 0.03 (§2). At the 0.80 point VLM share 37.4 / 0.3 / 12.6 % (ME340 / Sam's / Walmart),
  cheap-path audited accuracy 0.70 / 0.82 vs Gemini 0.81 / 0.89 on 119 samples (§3.1). Hazard questions 820 → ~461 → 140-150 with
  policy cuts (§3.2). Encoders per card on A100: DINOv2-L 10.2 ms, PE-Core-L 15.2 ms, OWLv2 12.9 ms; Jev 0.076-0.085 s/question,
  cold 34-45 s, 27.2 GiB peak (§4). Licences: do not use YOLO-World (GPL), OV-DEIM, Objects365-only weights; OWLv2 pending legal
  review (research output, Objects365/VG training); "do not use YOLOE (AGPL)" (§5) — the code uses YOLOE-26L "accepted for now"
  (`W/fast_report/cascade.py`). Fine-tuning deferred; x13 crops/features (322 MB / 125 MB) sit only in a session scratchpad (§6).
- **Rule library** (`R/rule-library-2026-09-29.md`, research draft, NOT implemented): judge coverage is low because only J1-J9
  exist (ME340: 140 of 255 objects get no rule) and u swamps thresholds (J1 on Sam's: 239 of 369 rows straddle; the 25 % scale
  term ≈ ±0.55 m at 2.2 m) (§0). 37 rule headings in 9 categories (aisles/exits, floor, storage, electrical, fire, machines, access,
  handling, retail), each with source clause, family, fields, T_pass/T_fail, value ± u verdict, image question; 9 new card fields
  (`clear_under`, `over_aisle`, `ceiling_gap`, `front_clearance`, `protrusion_m`, `stack_base`, `zone_hits`, `route_m`,
  `low_clutter`), a 9-key site profile, a NOT_RESOLVABLE outcome and N/A rows (§1-3). Projected rows with any judgement ME340
  33 → 115, Sam's 380 → 454, Walmart 100 → 376 (§4). Code check: `judge.CHECKS` still J1-J9 (`W/fast_report/judge.py` l.96-107);
  name-mapping fixes only partly landed (`W/fast_report/cards.py` l.175-246).
- **Where a VLM may be used today** (user decisions + memos): one scene-vocabulary request per video (accepted 2026-09-29);
  one Qwen question per cluster medoid for cards without an agreeing second vote, answer copied only to same-word members;
  on-demand clicks only after cheap votes fail; never per object/click by default, never hazard/PPE prompts while judgement is
  paused; a picture never makes a PASS or FAIL alone (`H/scratch/round5-notes.md`; `R/vlm-light-memo-2026-09-29.md` §1-2;
  `R/rule-library-2026-09-29.md` §1.2; `W/fast_report/ondemand.py`).

**Licence summary.** DA3-GIANT CC BY-NC (demo/internal); SAM 3 / 3.1 / SAM 3D: SAM License (no military, nuclear, ITAR);
SAM 2.1, Qwen3-VL, gsplat, Open3D: OK; LingBot: not until written confirmation (`FAST-PATH-PLAN.md` §6); RecGen: TRI
non-commercial, weights CC-BY-NC-4.0 (`W/fast_report/recgen_models.py`); YOLOE / YOLO11 / YOLO26: AGPL-3.0; OWLv2: Apache-2.0
(`runs/r4-coverage-results/results.json`); RAM++, PE-Core weights: Apache-2.0 (`W/fast_report/vocab.py`); TRELLIS, TripoSR:
MIT (`W/fast_report/gen3d.py`).

---

## C. Experiment log

### C.1 Fast-path validation, round 1 (E1-E8, ME340, 2026-09-28) and follow-ups

Plan and pass criteria: `W/docs/phase2/FAST-PATH-PLAN.md` §7. Measured: `H/scratch/fast-round1-synth.md` §1-2 (below
"synth"). Run folders: `runs/m3-exp-e1-e7-geometry-*`, `m3-exp-e2-e6-segment-*`, `m3-exp-e3-people-*`, `m3-exp-e4-sam3d-1`,
`m3-exp-e5-splats-events-cold-*`, `m3-exp-e8-import-*`. Spend E1/E7 0.80, E2/E6 1.57, E3 0.26, E4 4.17, E5 2.90, E8 0.60 USD
(ledger `m3_fast_path_result`).

| ID | question | key numbers (synth §1 unless noted) | verdict |
|---|---|---|---|
| E1 | DA3 any-view per shot: time, VRAM, pose vs DROID | GIANT 504: 5.66 s, 1.20× / 1.16×, ATE 0.043 m; GIANT 672: 14.85 s, ATE 0.063; BASE: 1.84×, ATE 0.16-0.17 m | PARTIAL: GIANT go, BASE no-go |
| E2 | SAM 3 batched: speed; recall of named objects ≥ 80 % | 0.066 s/frame (2 classes), 0.191 s/frame (20 words); recall 51 % (20 words), 73 % (40), 85 % ceiling | PARTIAL: speed yes, recall no (vocabulary) |
| E3 | 5 fps people vs today; SAM 3.1 stride 3; non-person movers | path diff median 0.007 m; rules 100 / 100 / 96.5 %; SAM 3.1 9.9 fps; movers untestable on ME340 | PASS (weak test, ME340 only) |
| E4 | SAM 3D fast mode on 30 objects; processes per GPU | plan setting 4/30 (FAIL); s1cfg12 7/30 at 3.49 s; MPS 2 procs 1.63× | PARTIAL: s1cfg12 + MPS |
| E5 | gsplat 5k/7k; feed-forward splat; vLLM events; cold start | 159-262 s, 25.3-26.6 dB (FAIL); GS head 14.5-17.2 dB (FAIL); events 4.44-4.63 s (PASS); cold 96.4 s eager / 188-197 s compiled; queue 367-631 s | PARTIAL |
| E6 | SAM 2 keyframes + projection/propagation ≥ 5× at IoU ≥ 0.85 | 2.0-9.4× at IoU 0.66-0.80; propagation 0.8-1.5×; batched boxes 32× | FAIL (box batching kept) |
| E7 | GPU TSDF and lift ≤ 10 s | 0.49-0.54 s; 0.10 s | PASS |
| E8 | import profile; cloud patch write ≤ 10 s | import 920-1186 s; write 1.53-2.80 s, readable 2.70-5.55 s | PASS |
| sum | | PASS 3 (E3, E7, E8), PARTIAL 4 (E1, E2, E4, E5), FAIL 1 (E6) (synth §1) | |
| E2b | per-video VLM vocabulary for SAM 3 | qwen-v1-3 (50 words) 0.385 s/frame, recall 81 % (production rule) / 89 % (score ≥ 0.3); + 8 core words (56) same recall; Gemini v1-5 + core (29 words) 66 % / 78 %; Qwen warm call 9.44 s (`runs/m3-fu-e2b-vocab-1/summary.json`); review: recall comes mostly from generic words, no fixed-list control (`FAST-BUILD-SPEC.md` §1); 1.7 USD (ledger `m3_followups_result`) | adopted (Qwen top 50 + 8 core words), replaced by RAM++ in r5b |
| E5b | splat speed-ups (dense init, 2 GPUs, H100, coarse-to-fine) | step 36.53 → 11.49 ms on A100 (`runs/m3-fu-e5b-splat-bench-a100/result.json`); A100 120 s 27.48 dB (with pose correction; 23.29 on report poses), 180 s 28.00, 2×A100 120 s 27.70, H100 120 s 28.10, H100 2.5 M cap 1200 s 31.02 dB (each run's `result.json`); 7.5 USD (ledger) | adopted: single-GPU fast step with DA3 seeds, preview only |
| E6b | keyframe masks + projected in-between outlines | 3.6-4.1 ms/frame (`FAST-BUILD-SPEC.md` §1); step-5 "pair" arm 4.84× faster, keyframe static IoU 0.774 (0.826 area-weighted), every `pass_at_5x` false (`runs/m3-fu-e6b-outlines-002/verdicts.json`); passed only after the metric changed (spec §1); 1.6 USD | adopted only as a layer labelled `projected`, no accuracy claim |
| E9 | one resident core on one vs two GPUs | one A100: serial 49.2 s, overlapped 37.4-37.6 s (peak 77.5 GB); two A100: static split 29.3 s, dynamic SAM 3 queue 20.4-20.9 s (peaks 35.5 / 48.0 GB); SAM 3 alone 24.0 s; MPS gives the core nothing (39.4 vs 40.3 s); cold start 91-116 s; ATE 0.041 m (`W/docs/phase2/M3-FU-E9-ONEGPU.md`); 4.5 USD (ledger) | adopted: dynamic 2-GPU SAM 3 queue = base of `core.py` |
| fb-a | builder A (core) | run 008 warm: cameras 14.1-14.9 s, room 18.0-18.3, people 19.3-20.0, objects v1 30.7-31.5 (`runs/fb-a-core-008/summary.json`) | merged |
| fb-b | builder B (SAM 3D gate rewrite + splat worker) | SAM 3D load 60.0 s + warm-up 16.3 s; prepare median 0.75 s (was 46 s), generate 4.25 s, assess 23.97 s; 6/30 accepted; ≤ 28.72 GB per process; splat 140 s 26.99 dB, 4.2 GB (`runs/fb-b-models-splats-bench-004/summary.json`) | merged |
| fb-d | builder D (VRAM harness) | torch and NVML agree on peaks 30.92 and 70.48 GB at 17.7 Hz (`runs/fb-d-harness-vram-001/vram-check.json`) | merged |
| fb | four builders + integrate (fb/integrate cdf9fb1) | A 5.8, B 7.34, C 0.02, D 0.30, integrate 10.6 = 24.06 USD (ledger `m3_fast_build_result_2`); warm call 0.370-0.412 USD (`runs/fb-results/summary.json`); VRAM before fixes 75.2-79.2 GiB → after 60.1-66.5 (DA3 to pinned host copy, SAM 3 buffers freed, SAM 3D cache released after 3 s idle) (`runs/fb-results/summary.md`); times in D | adopted (base of all later rounds) |

### C.2 Fine + fast experiments X1-X13 (2026-09-28/29; branches `fx/*`, folders `runs/fx-x*-00N`, canonical = the highest number with `VERIFY.md`)

Numbers are the verifiers' corrected ones (`VERIFY.md` in the canonical folder) where one exists; questions from
`H/scratch/fine-fast-plan.md` (X1-X4) and `H/scratch/windows-time-plan.md` (X6-X7); "spec" = `CLICK-MVP-SPEC.md` §1.2,
"mvp1" = `runs/mvp-results/summary.md` "Experiments plugged in". Paths below are relative to `runs/`.

| ID | question | key numbers (source) | verdict | canonical folder / spend |
|---|---|---|---|---|
| X1 fps | does a higher keyframe rate (1.67 → 5/10/15/30 fps) change objects, geometry, people? | in-between outline px-IoU 0.70 / 0.66 / 0.37 → 0.81 / 0.78 / 0.65 at 5 fps; recall@0.5 m 0.550 → 0.649 but **0.547 at equal object count**, and the 1 m-displaced null also rises 0.384 → 0.486 (so the gain is outline quality, not recall); added SAM 3 time 19.0 / 6.1 / 4.8 s [M×n]; geometry G5 ATE 4.11 / 22.2 / 18.1 cm with 8.1 s DA3, G10 4.04 / 19.6 / 18.4 cm with 22.2 s; people path diff 7.2 / 64.3 / 15.2 cm at 5 fps vs 8.3 / 64.9 / 10.2 at 15 fps (`fx-x1-fps-004/VERIFY.md`); int32 lift codes overflow at 30 fps → int64 (spec) | ADOPT 5 fps for objects (densify), geometry, people; ≥ 10 fps rejected (+0.02 recall for 2× time) | `fx-x1-fps-004` (Sam's from 003); 4.20 USD |
| X2 discover | does AMG → new words → another SAM 3 wave catch what the vocabulary misses? | amg16:vlm position recall 83 → 85 / 94, 96 → 107 / 121, 85 → 93 / 116; only 8 of 21 gains carry the right name; warm +21.9 / 21.3 / 20.7 s; masks per frame 195 → 409 (Sam's); vlm-ground 31.5-33.4 s for +0; `sam3-generic` 2.0-3.8 s found the one real hanging cord; no equal-mask-count control (`fx-x2-discover-005/VERIFY.md`); X2 lessons: SigLIP "spill", red marks → "fire extinguisher", Qwen yes 94 % (spec) | OPTIONAL / superseded (X10 marks AMG not recommended) | `fx-x2-discover-005`; ~6.2-6.3 USD |
| X3 LingBot | LingBot-Map as a parallel dense lane without slowing the core? | on a third GPU core change −1.0 / −1.6 / −2.0 s, gated layer at 61.3 / 43.6 s, Walmart never passed (ICP 2.60° > 2° cap); sharing GPU 1 slows the core +8.2 / +16.7 / +12.0 s; lane alone stride 2 36.9-56.4 s; points within 5 cm of the delivered layer 67 / 57 / 70 % vs DA3 TSDF 33 / 50 / 46 %; fused agreement only 35 / 43 / 53 % (`fx-x3-lingbot-005/VERIFY.md`) | fine-fast plan: separate dense display layer, never fused, demo weights; click MVP: "not needed" (spec) → not in the object layer | `fx-x3-lingbot-005` (timings 002); 6.62 USD |
| X4 refine | DA3 crops + 1 cm TSDF at ≤ 5 spots: do spots pass multi-view ≤ 1.5 cm? | 0 / 15 confirmed; multi-view median 4.6 cm; 2.38 s per spot; held-out outline IoU 0.56 vs 0.41; tilt differs up to 22.7° across methods, RANSAC swings 10.7° / 8.6° on identical points; LingBot per spot reliable on 2 / 15 (`fx-x4-refine-006/VERIFY.md`) | OPTIONAL display patch only (always NEEDS_REVIEW); tilt reporting rejected | `fx-x4-refine-006`; 4.22 USD |
| X5 | — | no X5 exists: the plans cover X1-X4 and X6-X7; no `fx-x5*` folder; the ID was left unused (inferred) | — | — |
| X6 windows + time | content windows processed in parallel + per-object time states | windows at 0.40: 10 / 11 / 16, median 2.0 / 2.0 / 1.7 s (`windows-time-plan.md` §2); per-window (b) vs per-shot (a): first objects 1.6-2.5× sooner, all facts only 3-19 % sooner; ATE ME340 4.09 vs 4.29, Sam's 22.1 vs 14.5, Walmart 4.19 vs 3.95 cm; change rule 0 claims on 8 unplanted runs **from only 123 in-view judgements** (the chain gate blocked 297 / 405 / 106 object-windows); planted recall 0 / 5 (recommended setting), 1 / 5 (every keyframe) (`fx-x6-windows-time-010/VERIFY.md`); association 3σ, σ = sqrt(0.04² + (0.05 z)²), or box IoU ≥ 0.2; door read 6.8-7.2°, one shelf 12.9° vs 1.4° (spec) | ADOPT the 0.40 window rule and position-only identity; change detection INCONCLUSIVE then (→ r5b/time); per-window geometry not plugged (mvp1) | `fx-x6-windows-time-010` (sweeps in 004, 008); ~4 USD |
| X7 object models | one display model per unique object from its best views, held-out gate (IoU ≥ 0.65, depth median ≤ 0.04, p95 ≤ 0.10, 103 px floor) | eligible at sep15 16 / 6 / 1 of 100, at sep5 30 / 58 / 33; boxes at sep5 parametric 58 / 87, RecGen 45 / 87, SAM 3D (b) 37 / 87, SAM 3D (c) 0 / 53; RecGen 8.2-11.8 s/object, SAM 3D 5.6-7.0 s; accepts fall from 36 / 36 / 44 to 23 / 21 / 26 at a 300 px floor (`fx-x7-object-models-001/VERIFY.md`); shelves 0 / 4; point-bootstrap CIs > 10× too small (spec) | ADOPT parametric fits first + RecGen for display only; reject SAM 3D path (c) | `fx-x7-object-models-001`; 2.65 USD |
| X8 Jev-Omni | can Jev judge same-object / label / part / EHS yes-no / rule? | (a) same object Jev 0.890 vs Qwen 0.934; (b) label 0.519 shortlist / 0.416 full; (c) 0.662 vs 0.547; (d) EHS AUROC Jev 0.74, Qwen 0.87, Gemini 0.85; (e) 0.991 rule / 0.771 severity on an easy synthetic set; Jev weights ~22.3 GiB, GPU 0 would reach 83-91 GiB (`fx-x8-jev-001/VERIFY.md`); as decider Brier 0.175 vs Qwen 0.138, ECE 0.129 (mvp1) | OPTIONAL, narrow: never the only namer; own GPU only | `fx-x8-jev-001`; 2.13 USD |
| X9 reuse | skip re-segmenting the same object (depth-warped predictions + DINOv2 check)? | L6-n06 SAM 3 37.5 / 26.5 / 24.8 s vs B15 103.6 / 44.6 / 37.4 s; totals exclude +17-25 s of 15 fps DA3 depth; outline results not comparable (lift vs no lift); check AUC .94 / .89 / .90; Walmart look-alike switches 7.2 vs 3.6 per 100 (`fx-x9-reuse-003/VERIFY.md`) | INCONCLUSIVE → not adopted (5 fps densify is cheaper) | `fx-x9-reuse-003`; 7.54 USD |
| X10 discover-all | segment "everything" without categories? | AMG own-look recall 0.27 / 0.47 / 0.60 (16², 32², +crop); recommended stack (catch-all + label words + OWLv2 boxes + ridge v2) own-look 0.69 → 0.85 (+0.16 [0.11-0.22]), +15.0-19.8 s GPU (lower bound); AMG16 on top +0 items at 24-30 s; RAM++ alone 0.30 / 0.39 at 0.46 s/frame, OWLv2 words +4 points, Qwen listing 0.43 / 0.53 at 16 s/frame — all "not recommended" as stages; stack designed on the scoring data (`fx-x10-discover-all-001/VERIFY.md`, `results.json`) | stack parts adopted later (OWLv2 boxes in r4/coverage); words-only `--discover` gave no click gain in mvp2 | `fx-x10-discover-all-001`; 8.63 USD |
| X11 local BA | can local BA at X4's spots get multi-view < 1.5 cm? | multi-view median anchor → BA → BA+DA3 → +tri 7.28 → 4.08 → 3.84 → 3.22 cm (n 30); confirmed spots: 1 on frame set A, 1 on B, none on both; held-out outline IoU 0.555 → 0.501; fast spot finder 4.7-7.9× faster (`fx-x11-local-ba-008/VERIFY.md`) | INCONCLUSIVE (development-set result); not used | `fx-x11-local-ba-008`; 6.18 USD billed |
| X12 motion fps | is 15 fps better than 5 for fast motion; adaptive cost? | walkers ≤ 2 m/s: 0 / 1022 failed steps at 5 fps; time-compressed proxy k = 6: Sam's 11 / 23 failures at 5 fps → 3 / 106 at 15; a 12 m/s gate at 5 fps: 11 / 23 → 1 / 23, factory 16 / 47 → 3 / 47; adaptive 15 fps +11-14 s, uniform +35-45 s; evidence rests on 2 objects (`fx-x12-motion-fps-003/VERIFY.md`) | ADOPT 5 fps + 12 m/s gate; adaptive 15 fps optional | `fx-x12-motion-fps-003`; ~2.4-2.5 USD |
| X13 naming | name cards without per-object VLM or fine-tuning, held out by video | 1,596 cards; SAM 3 word alone 0.69 class agreement (ME340 0.17); cross-video bank 0.15-0.25, in-video kNN 0.79; zero-shot best 0.33; OWLv2 0.39; Jev 0 cards routed; unchecked cluster copy 0.42 (`x13-naming-001/results.json`); verified 0.80 point VLM share 37.4 / 0.3 / 12.6 %, audited cheap names 0.55/0.74, 0.86/0.93, 0.86/0.90 vs Gemini 0.74/0.89, 0.89/0.89, 0.90/0.90 (`R/vlm-light-memo-2026-09-29.md` §3.1; results.json's 2.8 % Walmart share is superseded) | ADOPT for retail once a calibrated same-family video exists; new domain → VLM first | `x13-naming-001`; 0.66 USD |
| X13 hazards | judge hazards without a run-time VLM | Gemini questions 820 → 461 after safe steps → 140-150 with policy cuts; Jev q1 AUROC 0.976, q4 0.337 (`x13-hazards-001/results.json`) | safe steps adoptable; policy cuts need a user decision; shelved with judgement | `x13-hazards-001`; 0.81 USD |
| fx-synth | where do the hours go? | ME340 one-shot 10,128 s to dense_gate: SAM 3D 4,548 s (44.9 %), splat 1,390.6 s, LingBot 1,124.7 s; fact steps 4,265.8 s vs display 5,982.7 s; RecGen ≥ 4,781 s unfinished; SAM 3D 12 of 151 accepted (`fx-synth-001/results.json`) | basis of the fine-fast plan | `fx-synth-001`; 0 USD |
| windows-plan | estimates for the X6/X7 schedule | all facts 20-38 s, first objects 12-16 s, all models for 30-58 objects 110-215 s (estimates) (`fx-windows-plan-001/results.json`) | estimates only | `fx-windows-plan-001` |

Additional verified details for the model studies (same agent pass): route/jev pre-registered cascade 0.815 vs post-hoc
0.904; PE-Core-L picks the right primitive 0.821; Jev single call 0.488 s round trip, cold 43.37 s, 22.94 GiB
(`route-jev-001/results.md`); recgen/fast IoU change vs default −0.0101 (se 0.0095), SAM 3D s1cfg12 10 accepted at 2.95 s
median (IoU 0.706), compile 2.285 vs 2.598 s after 208 s warm-up (`recgen-fast-results/results.json`,
`recgen-fast-005/summary.json`). No `VERIFY.md` exists for x13-*, route-jev-001 or recgen-fast-* (unverified builder output).

### C.3 Click MVP rounds, object-layer rounds, model studies

| ID | question | setup | key numbers | verdict / path |
|---|---|---|---|---|
| mvp1 (mvp/integrate) | click → what it is, physical ±u, judgement | 4 builders (a-cards, b-judge, c-viewer, d-validate) + integrate, 5 rounds; final `runs/mvp-integrate-{me340-005,samsclub-004,walmart-004}` | see D; decider ECE failed; 0 false PASS in 60 labelled rows | accepted with misses: `runs/mvp-results/summary.md` |
| mvp2 (mvp2/*) | identity (Gemini), physical vs LiDAR/Kinect GT, judgements (Gemini hazard), click audit, timing | builders judge, identity, physical, accuracy, click; final `runs/mvp2-integrate-*-007`, GT `mvp2-integrate-gt-003` | see D; round total 142.03 USD (ledger `click_mvp_v2_result`) | `runs/mvp2-results/summary.md`, `tables.md`; builder pages `runs/mvp2-{accuracy,identity,judge,physical,click}-results/` |
| mvp3 (mvp3/*) | judge rule + walked-path gates, click coverage + on-demand segmentation, server-side Gemini, timing | final `runs/mvp3-integrate-{me340-003,samsclub-a2-003,walmart-001}` | Gemini route impossible → Qwen only; see D; 23.71 USD (ledger `click_mvp_v3_result`) | `runs/mvp3-results/tables.md`, `summary.json`; `runs/mvp3-judge-results/tables.md` |
| r4/physical | every card carries every physical field | `runs/r4-physical-bench-001`, GT `r4-physical-gt-002` | 0 contract violations on 12 bench + 12 GT reports; by eye plausible 26 / 25 / 29 of 30; GT arkit42 top 21.8 → 1.9 cm (assumed → true height) | `runs/r4-physical-results/results.md` |
| r4/instances | one real object = one card (seams, part/whole) | before/after on the same calls | clean delivered objects covered ME340 53 → 77 / 105 (first), Sam's 41 → 71 / 91, Walmart 38 → 51 / 74; audit "O" (one object) 20 → 28 / 22 → 24 / 23 → 25 of 30 | `runs/r4-instances-results/tables.md`, `summary.json` |
| r4/coverage | objects where SAM 3's words found nothing | probe of 5 box sources, then OWLv2 in pipeline | see B (OWLv2, YOLO rows); 9.88 USD | `runs/r4-coverage-results/results.json` |
| r4/naming | VLM-last naming cascade | X13 cascade in pipeline | measured through r4b types (D) | `runs/r4-naming-results/audit/` |
| r4/models | every card gets a display model | `runs/r4-models-bench-003` | 100 % cards with a primitive; SAM 3D 24/180 accepted; by eye 19 / 27 / 26 of 30 plausible; background SAM 3D (bench 002) +26 s, 70.7 GiB → turned off; 7.82 USD | `runs/r4-models-results/results.md` |
| r4b (r4b/integrate 40a153e) | integrated object layer | `runs/r4b-{me340-002,samsclub-001,walmart-001}`, GT `r4b-gt-001/002` | see D; review PARTIAL (F.1); 12.04 USD (ledger `object_layer_r4b_integrate_result`) | `runs/r4b-results/tables.md`, `summary.json` |
| r5/models (804a4ca) | better models than cuboids, fast | `runs/r5-models-bench-001`, 3 videos, 90 objects by eye | observed 38/52/0, primitive 7/41/42, RecGen 58/27/5, SAM 3D 29/36/8 (+17 none), TRELLIS 16/36/38, TripoSR 7/59/24, splat crop 2/31/26 (+31 none); GT angles on matched parts median 2.3-3.8° (30° gate: biased, see F); 18.1 USD | `runs/r5-models-results/results.json`, `runs/r5-models-bench-001/sheets/labels.json` |
| recgen/fast (1c137cc) | make RecGen fast | 19 X7 objects, A100 PCIe and SXM4 | 2.5 s vs 8.2-8.4 s/object; 20/40/70 objects in 19/36/62 s on 2 A100; 7.38 USD | `runs/recgen-fast-00{1..5}`, `runs/recgen-fast-results/results.json`; also `H/scratch/round5-notes.md`, ledger `recgen_fast_result` (no VERIFY.md) |
| route/jev (1b61ed1, 6cc6f27) | primitive vs generated model per card | ~313 agent-labelled cards, held out by video | cascade 0.904 post hoc (honest 0.815-0.875); Jev q5 raw alone 0.875; 1.56 USD | `runs/route-jev-001/results.md` |
| r5b/vocab (c03e686) | SAM 3 words without a VLM | sources qwen / taxonomy / pe / ram / yoloe-pf + ram ∪ pe | RAM++ chosen by the pre-registered rule; see D | `runs/r5b-vocab-results/{final,compare}.md`, `selection-rule.md`, `extra-runs.json` |
| r5b/models (18a9022) | right outline for every model; readable angles | `runs/r5b-models-{commercial,internal}-002`, GT `r5b-models-gt-001` | see D; 18.74 USD | `runs/r5b-models-results/results.md` |
| r5b/time (ce62d95) | cross-time within a video; honest width bounds | `runs/r5b-time-{gt-001,retail-001,retail-002}` | see D; 6.14 USD | `runs/r5b-time-results/results.md` |
| r5b/visit (e06573a) | same site visited again | `runs/r5b-visit-001` (14 videos), `-002` (13 pairs) | see D; ≤ 7.1 USD | `runs/r5b-visit-results/results.md` |
| r5b/integrate (854e3e7 +) | RecGen for all non-simple objects, clean 3D pane, overlay alignment, false-tilt fix | IN PROGRESS (`H/workflows/r5d.js`) | no results yet | `runs/r5b-results/` (not yet written) |

### C.4 Other threads (stopped or superseded; ledger keys)

- Streaming M0 probes (SAM 3 on L4, SAM 3.1 incremental, LingBot windowed, live map): 4.94 USD; M1 phone path stopped by the
  user 2026-09-27 ("video first") (`streaming_m0_probes_result`, `streaming_m1_result`; plan `W/docs/phase2/STREAMING-PLAN.md`).
- M2 one-command full-quality reports: Lightning run imported but unpublished (5.10); three-video one-shot stopped during
  ME340 at RecGen after 4 h 15 min (7.87) (`m2_lightning_oneshot_*`, `m2_three_videos_oneshot_result`).
- M4 "digital twin" (HomeBody-style): v1 below the bar (Gemini critique 0.504 USD), v2 stopped: the reconstruction must serve
  judgement, not a twin (`m4_twin_prototype_result`, `m4_twin_v2_result`).
- Older hand-built quality rebuilds (DROID / DA3 / LingBot / RecGen + SAM 3D per video), e.g. ME340 SAM 3D 52 objects, 154
  calls, 12 accepted; Walmart RecGen 9.22 USD (86 + 24 calls) (`sam3d_objects_result`, `walmart_quality_rebuild_result`).

---

## D. Round-by-round acceptance history

"Types" = identity/type measures; "clicks" = random-click audits (60 per video, agent-labelled); "phys" = completeness and
GT error; times warm (first) in s. Sources per row in the last column.

| round | types / identity | clicks, segmentation | physical | models | times (warm) | VLM calls | spend | sources |
|---|---|---|---|---|---|---|---|---|
| fb/integrate | name agreement with delivered report 8 / 46 / 25 % | outlines 38.6 / 26.7 / 22.6 s; boxes > 3 m 28.7 / 10.9 / 5.4 % | camera ATE 0.041 / 0.221 / 0.178 m; Sam's scale 1.124, people path diff 0.64 m | first SAM 3D 85.6 / 67.5 / 49.9 s; accepted 3 / 4 / 8 of 30; all models judged 129.7 / 142.5 / 99.5 s; splat 27.17 / 21.62 dB / not scored | cameras 18.1 / 14.8 / 14.7; objects 35.4 / 24.7 / 20.2; splat 186.1 / 174.2 / 170.6; ME340 missed people/objects/outlines/splat targets; GPU 60.1/52.9, 66.5/43.8, 63.4/44.3 GiB | Qwen vocab, events (Walmart events 0/4), names | 24.06 | `CLICK-MVP-SPEC.md` §1.1; `runs/fb-results/summary.md`; `runs/mvp-results/summary.md`; ledger |
| mvp1 | audit right / right-or-close 26/74 %, 65/98 %, 85/98 %; name agreement 15 / 28 / 43 % | pick v2 object clicks correct 53 / 44 / 39 %; person 93 / 43 / 0 %; background false hits 1 / 2 / 6 % | vs delivered top median 0.05 / 0.24 / 0.12 m; boxes > 3 m 3 / 1 / 2 %; repeat k pooled height 1.028; 0 contract violations (269 / 645 / 730 cards) | SAM 3D display only | cards v1 42.0 / 33.8 / 25.4; cards v3 75.5 / 60.1 / 45.6; first SAM 3D 168.3 / 159.7 / 71.0; splat 223.3 / 205.9 / 190.8; times check failed on ME340 and Sam's | Qwen decider (identity + judgement) | 42.98 | `runs/mvp-results/summary.md`; ledger `click_mvp_result` |
| mvp2 | held-out right / right-or-close 0.73/0.87, 0.89/0.89, 0.90/0.90 (all 0.81/0.88, n 119) | correct / miss / background-hit of 60: 21/9/3, 28/19/0, 22/21/0; picked precision 0.88 / 1.00 / 0.96 | GT TUM (held out) top 7.5 → 2.8 cm, height 2.2 → 2.0 cm (assumed → true height); one-view-set extents cover 0.80-0.94; judgements 405/405 PASS right, FAIL 3/4 | SAM 3D ME340 warm 0/30 in 7/7 calls | final judgements 102.2 / 83.7 / 80.7 (first 106.5 / 107.0 / 83.0); cards v1 43.4 / 30.8 / 30.9; GPU ≤ 67.6 GiB | Gemini names 20.6 M tokens + hazards 5.1 M | 142.03 round (integrate ~43) | `runs/mvp2-results/summary.md`; ledger `click_mvp_v2_result` |
| mvp3 | held-out 0.36/0.55, 0.86/0.93, 0.72/0.83 (Qwen only) | real objects opening the right thing, report → with on-demand 27/32 → 29/32, 22/43 → 40/43, 21/42 → 36/42; on-demand p50 1.11-1.15 s | card audit physical plausible 19 / 20 / 19 of 20 | — | final judgements 88.0 / 76.7 / 75.8; all names 86.3 / 71.7 / 73.6; cold 146.9 / 91.1 / 146.0 | Gemini 0; Qwen identity questions 500 / 1316 / 1504 | 23.71 | `runs/mvp3-results/tables.md`, `summary.json`; ledger |
| r4 builders | naming cascade (X13) | instances: clean covered ME340 53 → 77/105; OWLv2: right card 0.75→0.825, 0.514→0.676, 0.727→0.909 | 100 % complete, 0 violations; by eye 26 / 25 / 29 of 30 plausible | primitive on 100 %; SAM 3D 24/180; by eye 19 / 27 / 26 of 30 | cards v1 +0-0.6 s from fits; models final 113-141 s | judge off, hazard VLM off | 21.14 + 7.82 | `runs/r4-*-results/`; ledger `object_layer_r4_result`, `object_models_r4_result` |
| r4b | family-typed 95 / 100 / 98 %; specific name 20 / 53 / 64 %; held-out family right 0.51 / 1.00 / 0.83; fresh audit type right 14 / 21 / 29 of 30 | clicks on a thing opening the right card 27/31 = 0.87, 35/46 = 0.76, 33/40 = 0.82; clean delivered covered 80/105, 79/91, 61/74 | 100 % complete, 0 violations; TUM top 7.4 → 3.1 cm, height 2.8 → 2.5 cm; arkit42 top 21.1 → 1.6 cm; audit plausible 24 / 29 / 28 of 30 | 100 % (boxes 317 / 818 / 781); SAM 3D 1 / 5 / 4 of 30; audit model implausible 3 / 9 / 11 of 30 | cards v1 43.1 / 30.1 / 31.2; types (densify) 97.8 / 87.5 / 93.1; models final 181.2 / 178.7 / 179.2; call end 226.7-230.2; GPU ≤ 70.2 GiB | naming 48 / 32 / 19, scene vocab 1, event captions 2, hazard 0 | 1.77 / 2.15 / 1.41 per bench; integrate 12.04 | `runs/r4b-results/tables.md`, `summary.json`; ledger |
| r5b/vocab | family-typed 93 / 98 / 99 %; specific 13 / 77 / 64 %; held-out family 0.60 / 1.00 / 0.79 | clean covered 73/105, 79/91, 58/74 (r4b 80, 79, 61); union RAM++ ∪ PE ME340 81/105 | — | display off in final runs | cards v1 33.9 / 26.3 / 26.4; types (densify) 66.9 / 78.8 / 74.4; GPU 0 ≤ 69.0 GiB | 0 before cards v1; naming 31 / 27 / 20 | ~6 est. | `runs/r5b-vocab-results/final.md`, `extra-runs.json`; `H/HANDOFF.md` §9 |
| r5b/models | — | outline by eye (50 cards) right / partial / wrong 20/25/5, 26/19/2, 37/12/1 (r4b box 16/23/11, 26/17/4, 17/31/2); silhouette IoU median 0.53 / 0.67 / 0.69 (r4b 0.45 / 0.65 / 0.59) | angles, every part counted: pair error median 2.6°, p90 17.6°, ±u covers 0.85 of pairs but 0.37 of all our parts; ME340 26 of 27 parts read 15-75° are not tilted; Walmart no angles (plumb gate) | tier shares observed / primitive / generated 95.4/4.6/0, 64.8/35.2/0, 82.2/17.4/0.4 %; generated 0-9 per video | tier 0 at cards v1 + 2.5-4.7 s; tier 1 done internal 146 / 106 / 118 | Jev 31 / 20 / 23 questions (3.1-3.7 s) | 18.74 | `runs/r5b-models-results/results.md` |
| r5b/time | — | — | timeline on 276/276, 1052/1052, 975/975 cards; 0 false change claims on unplanted footage (8 calls, 5,239 cards) and GT static scenes (799 cards); planted 5 of 9 found, 3 false (Walmart); widths with a number or bound 55 / 32 / 48 % (r4b 91 / 89 / 94 %); new "at least" holds 122/124 on GT | — | cards stage ±(−1.8 … +2.7) s | events off, neutral vocab | 6.14 | `runs/r5b-time-results/results.md` |
| r5b/visit | 193 names and 130 types carried to revisits | match precision 0.70, recall 0.76 (0.86-1.00 where shots registered) | registration vs GT 4.9-12.0 cm, 0.7-3.0°; change claims 13/16 right; change recall 10 / 137 GT places; false changes 3 / 1,660 judged | 1,097 models carried, 4 ghosts | visit step 5.7-17.4 s on stored analyses | events off | ≤ 7.1 | `runs/r5b-visit-results/results.md` |
| r5b/integrate | — | — | — | RecGen for all non-simple objects (target) | targets: cards ≤ 30 s, all typed ≤ 60 s, all models ≤ 90-120 s (`H/HANDOFF.md` §7) | one scene-vocab request allowed | not recorded | IN PROGRESS |

Budget: authorization 500 USD, accounted + reserved ceiling 669.71 USD (ledger `authorization_usd`,
`accounted_and_reserved_ceiling_usd`); the user allowed exceeding it on 2026-09-29 (ledger
`authorization_history_note_2026_09_29`); round 5 reserve 82 USD (`object_layer_r5b_reserve_usd`).

---

## E. Dead ends and failure modes (with evidence)

**Models and methods that lost**
1. DA3-BASE as the commercial pose model: photometric 1.84×, ATE 0.16-0.17 m (`fast-round1-synth.md` §1 E1).
2. DROID-posed DA3 at 5 fps keyframes: loses track on 4/5 GT windows, ATE 0.65-0.92 m (`runs/mvp2-accuracy-results/summary.md`).
3. Window + Sim3 stitched geometry: scale swings 0.6-2.0 per link, tops at true height 15 / 185 / 3 cm (same).
4. SAM 2 keyframes + reprojection/propagation (E6): IoU 0.66-0.80, propagation only 0.8-1.5× (`fast-round1-synth.md` §1).
5. SAM 3D plan fast mode (4/30 accepted) (same, E4); background SAM 3D generation (+26 s, 70.7 GiB) (`runs/r4-models-results/results.md`).
6. Feed-forward splats from DA3's GS head: 14.5-17.2 dB and non-commercial; 5k-step splats 25.3-26.6 dB (`fast-round1-synth.md` §1 E5).
7. Legacy import `import_video_scene`: 920-1186 s (E8a) → replaced by the patch store.
8. X4 refine and X11 local BA: never confirmed; outline IoU 0.555 → 0.501 with BA (`runs/mvp-results/summary.md`).
9. X9 reuse ladder: measured the wrong gain (lift vs no lift) and omitted 17-25 s of depth (same).
10. TRELLIS, TripoSR, splat crops as object models: by eye 16/36/38, 7/59/24, 2/31/26 (`runs/r5-models-bench-001/sheets/labels.json`).
11. Primitives everywhere (r4/r4b): 84-89 % boxes; fresh audit implausible 3 / 9 / 11 of 30 (`runs/r4b-results/tables.md`; `H/workflows/r5b.js` l.11) → tier 0 observed surface.
12. Gemini as namer/hazard judge: best quality but no server-side secret, relay through the laptop, 30 s/request under load (`runs/mvp3-results/summary.json` `secret_route`; `runs/mvp2-results/summary.md`).
13. Jev-Omni as the judgement decider (X8): lost to calibrated Qwen (`runs/mvp-results/summary.md`).
14. SigLIP zero-shot naming ("spill"), red set-of-marks ("fire extinguisher"), VLM yes/no judges (yes 94 %) (`CLICK-MVP-SPEC.md` §1.2).
15. X10 words-only discovery (`--discover`): no click gain, slower final judgements (`runs/mvp2-results/summary.md`).
16. Qwen scene vocabulary as the source of wave-2 words ("VLM first": words that found 77-90 % of objects) and a neutral
    vocabulary prompt (16-50 words, repeated "warehouse X", object counts swing 1.6× between calls) (`H/workflows/r5b.js` l.11;
    `runs/r5b-time-results/results.md` issue 3).
17. Dishonest width "at most" bounds (fail 8-28 % vs GT) → honest rule, but coverage 91-94 % → 32-55 % (`H/workflows/r5b.js` l.11; `runs/r5b-time-results/results.md`).
18. Full-quality one-shot pipeline: stopped at RecGen after 4 h 15 min on ME340 (ledger `m2_three_videos_oneshot_result`);
    root causes of the old 2-3 h: one GPU stage at a time, one machine per app, 2 s idle shutdown, ~290 frames × 1024 prompts
    for SAM 2, 12-13 min CPU steps on the Mac, SAM 3D 38 min, RecGen 1-3 h serial, 60k-step splats (`FAST-PATH-PLAN.md` §1).
19. Streaming (M1) and the digital-twin line (M4): stopped by the user (ledger `streaming_m1_result`, `m4_twin_v2_result`).
20. Older hand-built path (paths as in B, "Pre-fast-path background"): ORB-SLAM3 monocular (0.729 m global error, GPLv3);
    LingBot room geometry (below the 75 % pair gate; live pose drift within 10 min); DROID dense geometry (388 mesh pieces;
    higher resolution did not help); MoGe single-frame depth + scale/affine alignment (ghost layers: "do not retry");
    MapAnything with given cameras (`rejected_camera_contract_drift`); DA3 confidence filtering (deletes correct walls and doors);
    global depth-bias correction (worse overall); fal SAM 3.1 (merged class masks, no identities); SAM 3.1 streamed per frame
    (90 % consistent ids); SAM 3D on tight crops (flipped objects, depth 12 % off); models from VLM names (1 of 10 passed);
    LingBot-seeded splats (holes, floaters); box auto-review (did not reach "reject all 10, keep ≥ 30 of 32"); the first inferred
    floor reached behind walls and was withdrawn (`D/HANDOFF.md`, `D/VIDEO-MVP.md`, `D/ARCHITECTURE.md` §7, `D/STREAMING-PLAN.md`
    §11, `D/COMPONENTS.md` C4, `D/INFERENCE-POLICY.md`).
21. Fine-fast lanes that did not pay: AMG-driven discovery (+21 s for 8 correctly named gains) (`runs/fx-x2-discover-005/VERIFY.md`);
    LingBot fused into the TSDF (agreement 35-53 %) (`runs/fx-x3-lingbot-005/VERIFY.md`); ≥ 10 fps keyframes (+0.02 recall for 2×
    time) (`runs/fx-x1-fps-004/VERIFY.md`); SAM 3D path (c) 0 / 53 (`runs/fx-x7-object-models-001/VERIFY.md`); Laya zero-shot
    0.415 / 0.479 (`runs/fx-x8-jev-001/VERIFY.md`); Qwen as X9's decider 120-338 s for no outline gain
    (`runs/fx-x9-reuse-003/results-notes.json`).

**Failure modes seen in runs (keep guarding against them)**
- Identity pass raised a TypeError while densify ran; it faulted both GPUs for every later call (round 3 of mvp1)
  (`runs/mvp-results/summary.md`). A worker dying in one call broke the process pool for every later call (`runs/mvp2-results/summary.md`).
- Stale judgements: two cards puts with the same version, the older run finishing last (25 rows on wrong checks) → order by put
  (`runs/mvp2-results/summary.md`).
- Duplicated VLM requests at 15 s doubled load (42/44 copied on Sam's Club) and lost both copies in the relay (same).
- Burned-in captions: background hits and a "sign" that appeared/moved → top/bottom band guards (`runs/mvp2-results/summary.md`;
  `runs/r5b-time-results/results.md`).
- Depth drift re-lifts one static box 1-2 m along an aisle → several cards per object and false "disappeared" claims
  (`runs/r5b-time-results/results.md` §3c).
- Thin / receding / close-up objects: pooled multi-view points misalign in one view (tool holders, shoes), thin things read deep,
  pipes lifted in part (`runs/r4-models-results/results.md`); tier 0 loses ends of pipes and cables (`runs/r5b-models-results/results.md`).
- Planar parts over-found on pipes, cables, rod clusters (57 % of our GT-sequence parts have no GT part) → false tilts
  (`runs/r5b-models-results/results.md`).
- Host variance: boot imports 17 vs 34-54 s; slow Volume writes (median 27 s vs 3 s) made Sam's Club final judgements 150 s
  (`runs/mvp2-results/summary.md`; `runs/mvp3-results/summary.json` `final_runs`).
- Shared mutable naming bank changes names between runs (Sam's Club primitives 324 vs 418 between two profiles' runs)
  (`runs/r5b-models-results/results.md` issue 6).
- ME340 held-out names swing run to run (right-or-close 0.80-0.87 across mvp2 runs 003-007) (`runs/mvp2-results/summary.md`).

---

## F. Review findings still open

### F.1 r4b independent review: PARTIAL (`H/workflows/r5b.js` l.11; derived data `/private/tmp/claude-501/-Users-adam-Desktop-panoptes-public/1fd9a1db-e580-4bfc-8110-119a1cc38a99/scratchpad/rev/derived.json`)

| # | finding | status after round-5 builders |
|---|---|---|
| 1 | VLM first: one Qwen scene-vocabulary request at 0.4-8.6 s set the words that found 77-90 % of objects; on-demand clicks asked Qwen once per click | builder fixed (RAM++, 0 VLM before cards v1: `runs/r5b-vocab-results/final.md`); but coverage fell on ME340 (73 vs 80) and Walmart (58 vs 61); union measured (ME340 81/105); user then accepted one scene-vocab request → default word source still to be decided in integration |
| 2 | hazard prompts: event captions ask "ppe" and "safety_note"; VOCAB_PROMPT asks "ehs_relevant" types | builders ran `events False` and a neutral prompt (`runs/r5b-time-results/results.md`; `runs/r5b-visit-results/results.md`); integration must keep it |
| 3 | judge defaulted on in core and bench | runs used `--judge off`; code default change not verified here (inferred open) |
| 4 | ME340 family right on 0.51 of held-out items; "material / part (type only)" catch-all on 104/376 cards | 0.60 with RAM++ (r5b scorer); still the weakest video (`H/HANDOFF.md` §7 item 7) |
| 5 | models 84-89 % boxes; 3/5/8 of 20 random implausible; cylinders wider than the mask; frames without seen uprights; "seen" by point count without a facing test | tier 0 + checked primitives + `observed.py` rule (`runs/r5b-models-results/results.md`); tier 0 partial on 25/50 ME340 cards |
| 6 | width "at most" bounds (40-62 % of cards) fail 8-28 % vs GT | fixed honestly; width coverage now 55 / 32 / 48 % (open) |
| 7 | naming bank is shared mutable state (1798 → 1998) | `bank_write/bank_save False` option exists; names still differ between runs (r5b/models issue 6) |
| 8 | 3D pane labels boxes with objects-layer words | r5b/models viewer uses card names; user wants no always-on labels or wireframes at all (`H/HANDOFF.md` §2) |
| 9 | times must be Volume-commit `written_s`, not queue time | adopted in r5b pages |

### F.2 r5/models review (`H/workflows/r5b.js` l.12; `H/scratch/round5-notes.md`)
- Angle GT stats counted only parts matched within 30° (selection bias) → re-scored in r5b/models with every part: median 2.6°,
  p90 17.6°, ±u covers 0.37 of all our parts (`runs/r5b-models-results/results.md`) — **open: over-found parts, false tilts**.
- Tilted parts on the three videos never audited → audited in r5b: 1 true tilt per video; 26 of 27 ME340 15-75° parts false
  (same) — **open: show a part's tilt only with small u from enough views** (INTERIM 2, `H/workflows/r5b.js` l.47).
- Generated-mesh "observed" flag was a distance test (2 × VOXEL_M) → one facing + clear-sight + on-surface rule
  (`W/fast_report/observed.py`).
- Internal profile hit 75.9 GiB on GPU 0 → 804a4ca cache fix, 51.8-59.0 GiB on GPU 0 in r5b/models internal runs.
- 600-700 small GLBs cost 10-13 s of commit → one GLB per shot (done).
- r4/models verifier: cylinder arcs and frame members "seen" by point count alone (`round5-notes.md`) → `observed.py` (done).

### F.3 Round-5 builder issues (each builder's own list; all open unless integration fixed them)
- **models** (`runs/r5b-models-results/results.md` Issues): (1) SAM 3D overruns the 135 s tier-1 deadline (queued jobs run
  on; commercial ME340 189-199 s); (2) Walmart has no angles: plumb gate 2 deg vs walls 2.3-2.5 deg; (3) planar parts
  over-found, u median 10.6° on GT, 12-17° on videos; (4) tier 0 partial on thin/receding objects, overflow 0.36 on ME340;
  (5) cards v1 1-1.5 s later (primitive checks 23-36 CPU s); (6) names differ between profiles (shared bank); (7) on-demand
  surface not clicked live; (8) web fixture missing. Plus INTERIM 3: tier 1 rare (0-9 per video) because route + view gate
  starve complex objects; target tier 1 ≤ 150 s warm (`H/workflows/r5b.js` l.47).
- **time** (`runs/r5b-time-results/results.md` Issues): false "disappeared" from depth drift (3 on planted Walmart); move
  links name the nearest look-alike; neutral vocabulary unstable; width coverage 32-55 %; real moves 0 of 2; GPU 0 > 72 GiB in
  5 of 20 retail calls; cross-visit timelines not in this branch.
- **visit** (`runs/r5b-visit-results/results.md` Issues): change recall 7 % of GT changes (6 % of ≥ 30 cm); match precision
  0.51-0.81; refusals on short shots and one ARKit window at 0.106 vs 0.1; the filmer's own cart reads as a change; u covers the
  GT registration error at 1u on half the registrations; single-threaded comparison 4-12 s; runs used events off + neutral
  prompt (not r4b settings).
- **vocab**: RAM++ loses clean coverage vs the Qwen words on ME340/Walmart; PE finds workshop words but misses retail goods
  (`runs/r5b-vocab-results/compare.md`); union costs +13 s to cards v1 on ME340 and 71.5 GiB (`extra-runs.json`).
- **recgen/fast**: gate noisy (seed flips 6/19) — need more objects (sep15) before claiming parity; thin pallet loss; budget:
  GPU 0 peaks ~69 GiB with SAM 3D and RecGen processes need ~20 GiB each → schedule after SAM 3 frees memory or replace SAM 3D
  on GPU 0 (`H/scratch/round5-notes.md`).
- **route/jev**: "bag" policy (simple/box vs generated), cap RecGen per video (Walmart shoes ~111 objects) or reuse one model
  per look-alike group (`H/scratch/round5-notes.md`) — r5b/models implemented groups and a 60-group cap.

### F.4 Older items still open
- Gemini would need a named Modal secret via ATM/Infisical (`runs/mvp3-results/summary.json` `secret_route`) — moot while
  VLM-light holds.
- Judgement (paused): J2, J3a, J8 decided nothing, J6 never triggered, people's feet unvalidated for J3a
  (`runs/mvp2-results/summary.md` Open issues); decider ECE > 0.10 (`runs/mvp-results/summary.md`). Rule library for later:
  `R/rule-library-2026-09-29.md` (not read, see G).
- Card frame x axis from the first camera is fragile when it looks down (14.5° on arkit47 w0) — median forward suggested
  (`runs/mvp2-accuracy-results/summary.md` Open).
- Measured scale (known-size object or metric depth head) would shrink u more than any geometry option (same).
- Retail click coverage: goods, rack structure, hanging items (19-21 of 60 random clicks in mvp2); improved by OWLv2 and
  instances but on-object "nothing" still 4 / 11 / 6 of 60 in r4b (`runs/r4b-results/tables.md`).
- `web/tests/live-report-check.ts` fixture `runs/fb-c-fixture-001` missing (every round since mvp2).

---

### F.5 Fine-fast verifier findings still open (short quotes from each `VERIFY.md`, paths relative to `runs/`)
- X1: "No end-to-end analysis time was measured per rate" (`fx-x1-fps-004/VERIFY.md`).
- X2: "No control with the same number of masks was run"; the veto should check that the word's SAM 3 masks covered the
  region (`fx-x2-discover-005/VERIFY.md`).
- X3: "Walmart never produced a passing timed layer"; completeness never measured; the final lane needs re-timing
  (`fx-x3-lingbot-005/VERIFY.md`).
- X4: no comparison with the delivered reports; tilt estimator unstable "by up to 10.7° on identical points" (`fx-x4-refine-006/VERIFY.md`).
- X6: "Validate on a held-out clip with a real revisit"; "The branch head was never run on GPU"; add a scale-consistency term to
  the stitch gate; Mac vs container window counts differ, cause not found; the (a) re-join not built
  (`fx-x6-windows-time-010/VERIFY.md`; `H/scratch/windows-time-plan.md` §2, §8).
- X7: "Report a footprint only when two or more faces were observed"; write parametric GLBs inside the clock
  (`fx-x7-object-models-001/VERIFY.md`).
- X8: "Jev does not fit on GPU 0 at all"; rule accuracy is an upper bound on an easy synthetic set (`fx-x8-jev-001/VERIFY.md`).
- X9: add B5-raw / B15-raw baselines; "Reused share has a bug"; tuning not held out (`fx-x9-reuse-003/VERIFY.md`).
- X10: report 15.0-19.8 s as a lower bound; a human should audit the W/T/K tiles (`fx-x10-discover-all-001/VERIFY.md`).
- X11: "No spot is confirmed on both frame sets by any method" (`fx-x11-local-ba-008/VERIFY.md`).
- X12: n is effectively 2 objects plus time compression; "The DA3 licence blocks commercial use of this pipeline as it stands"
  (`fx-x12-motion-fps-003/VERIFY.md`).
- X13: the rule structure was not held out; encoder memory on the production A100 not measured; the policy cuts' miss rate cannot be
  measured; `x13-naming-001/results.json` still says the vocabulary came from Gemini (it came from Qwen) and states a 2.8 %
  Walmart VLM share (verified: 12.6 %) (`R/vlm-light-memo-2026-09-29.md` §3, §7).
- route/jev: the recommended class list is post hoc; "bags are a policy decision" (`route-jev-001/results.md` Caveats).

### F.6 Older and memo-level open items (not yet addressed by any round)
- Scale: one shared scale gate for the live product (A0) deferred; the live workbench defaults to 1.5 m treated as calibrated;
  a measured-reference scale (≥ 3 measured points, one held out, residual ≤ 3 %) planned; RL: a scale anchor must be validated per
  shot (20 Sam's pallets read 1.22 m ± 0.50, consistent with 48-inch pallets) (`D/ARCHITECTURE.md` §1, §8;
  `D/STREAMING-PLAN.md` §3; `R/rule-library-2026-09-29.md` §1.4).
- Rule engine: gaps A4-A20 (coverage input, free-space operator, time predicates, zones/CAD, object relations, engine defects,
  demo vs platform verdicts, ZEN not live, duplicates, sim export) (`D/ARCHITECTURE.md` §6); rule library not implemented;
  RJ's V1-V6 pass bars unvalidated (`R/reuse-judgement-memo-2026-09-28.md` §5.2); `policy.py:303` treats an assumed camera height
  as qualified scale (RJ corr.).
- Licences to settle: LingBot weights (written confirmation), SAM in an appliance, MapAnything `-apache` swap, MoGe-3 and DROID
  weights, DA3-GIANT for any commercial profile (`D/STREAMING-PLAN.md` §9; `S/m2-design.md` §6; `R/fine-expand-prior-art-2026-09-28.md` §4);
  OWLv2 training-data review; YOLOE AGPL "accepted for now" in code vs "do not use" in the VLM memo (`R/vlm-light-memo-2026-09-29.md` §5).
- Provenance: DA3, MoGe-3, SAM 2.1, SAM 3 image and Qwen3-VL not pinned to a revision in the older scripts (`S/m2-design.md` §6)
  (the fast app pins DA3 and SAM 3 revisions: `W/modal_apps/fast_report_app.py` l.37-39).
- Privacy/security: every GET needs a read capability before real footage; retention and face blurring undecided
  (`S/stream-plan.md` §7; `D/STREAMING-PLAN.md` §9.8).
- Published older reports carry a false "~20 % disagreement" sentence (`S/m2-design.md` §0.5).
- Deformables: X6 records a cable deforming in place as "static" (RJ corr.); r5b/time re-measures deformables per window but
  makes no place claims for them (`runs/r5b-time-results/results.md`).
- Data for later fine-tuning: x13 crops (322 MB) and features (125 MB) live only in a session scratchpad and should be moved to
  persistent storage (`R/vlm-light-memo-2026-09-29.md` §6).

---

## G. Sources and coverage of this digest

Read directly: the round result pages (`runs/mvp-results`, `mvp2-results`, `mvp2-identity-results`, `mvp2-accuracy-results`,
`mvp3-results`, `mvp3-judge-results`, `r4-*-results`, `r4b-results`, `r5-models-results` + `r5-models-bench-001/sheets/labels.json`,
`r5b-vocab-results`, `r5b-models-results`, `r5b-time-results`, `r5b-visit-results`, `route-jev-001/results.md`), the ledger,
`FAST-PATH-PLAN.md`, `INFERENCE-POLICY.md`, `CLICK-MVP-SPEC.md` §0-2 and §7, `H/scratch/fast-round1-synth.md` §0-2,
`H/scratch/round5-notes.md`, `H/HANDOFF.md`, `H/workflows/r5b.js` (review findings), module docstrings in `W/fast_report/` and
`W/modal_apps/fast_report_app.py`.

Read through four read-only extraction passes whose findings are merged above (numbers carry the paths they reported):
the fine-fast runs (`runs/fx-x*`, `x13-*`, `route-jev-001`, `recgen-fast-*`, `fx-synth-001`, `fx-windows-plan-001`, their
`VERIFY.md` files, `fine-fast-plan.md`, `windows-time-plan.md`); the fast-path runs and docs (`m3-exp-*`, `m3-fu-*`, `fb-*`,
`fb-results/summary.md`, `FAST-BUILD-SPEC.md` §0-11, `E3-PEOPLE.md`, `E4-SAM3D-FAST.md`, `M3-FU-E9-ONEGPU.md`, speed/latency/
twotier plans); the older docs (`ARCHITECTURE.md`, `COMPONENTS.md`, `PLAN.md`, `README.md`, `HANDOFF.md`, `VIDEO-MVP.md`,
`TECHNICAL-ROADMAP.md`, `DYNAMIC-SCENE.md`, `HUMAN-MOTION-RESEARCH.md`, `STREAMING-PLAN.md`, the three `2026-09-17-*.md` notes,
`stream-dossier.md`, `stream-plan.md`, `m2-design.md`, `plan_draft.md`, `memo.md`); and the four research memos.

Not read or only partly read: large raw JSONs (X1 raw-*, X3 eval-*, X4/X11 per-video, X9 stage2-*, X8 decisions, route-jev and
x13 question/answer files; `fb-results/summary.json` beyond lines 2370-2530); images and contact sheets; the verifiers'
scratchpad scripts; `fb-d-harness-gaps-001/summary.json`; mvp2 builder pages `mvp2-judge`, `-physical`, `-click` (skimmed);
`CLICK-MVP-SPEC.md` §3-6 and §8-12. Unreconciled: DA3 GS-head per-frame PSNRs of ~12 dB in
`runs/m3-exp-e5-splats-events-cold-da3-gs/result.json` vs 14.5-17.2 dB in `fast-round1-synth.md`; "Cosmos events",
"LingBot-World" and "Pi3X as person descriptor" do not appear in the listed sources (Pi3X is a geometry model in
`S/speed-dossier.md`, CC BY-NC). `S/memo.md` and `S/twin-v2-research-memo.md` are identical. The r4b review has no written
report on disk: its findings are the list injected into `H/workflows/r5b.js` l.11; the scratchpad `rev/` folder holds only
derived data (`derived.json`: per-call type counts) and audit sheets. Round-5 integration results (`runs/r5b-results/`) did
not exist yet when this was compiled.
