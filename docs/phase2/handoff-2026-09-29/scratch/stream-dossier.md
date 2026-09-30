### stages
**Panoptes offline pipeline: stage inventory and dependencies (branch `codex/phase2-video`)**

Repo: `/Users/adam/.codex/worktrees/panoptes-phase2-video`. Run outputs: `/Users/adam/Desktop/panoptes-public/research-notes/phase2/runs`.

There is no orchestrator in the code. Every step is run by hand; `docs/phase2/ARCHITECTURE.md` §3 says so too. Nothing in the code detects hard cuts. The timings below come from the Sam's Club A2 run (420 frames, 30 s clip) via `runs/samsclub-quality-manifest.json` and the per-run JSON files. `da3_unposed.py` and `register_cut_shot.py` are not committed yet.

Mode column: **B** = batch, needs the whole shot (a global solve or a vote across all views). **W** = could run on a sliding window with some lag. **I** = per frame or per entity, can run on new data almost as is.

| # | Stage | File | Inputs (from stage) | Outputs | Compute (measured) | Mode today → streaming | Hard deps |
|---|---|---|---|---|---|---|---|
| S0 | Clip prep and lens | `scripts/prepare_video_clip.py` (+`moge3_app.py`) | MP4, `--start/--end`, `--fov-deg` or a MoGe-3 FOV estimate (5 frames; 15 by hand on A2) | `data/clips/NAME/{rgb/*.png, rgb.txt, clip.json(K), source-rgb.mp4}`, `--full-video` → `source-full.mp4/json` | Mac CPU + L4 per FOV frame | B (cuts one window, center-crops to 4:3, resizes to 640×480). Frame extraction is I. K is fixed once, but the shot-split clip (`-a2`, hard links) got a hand-re-estimated K | – |
| S1 | Cameras | `modal_apps/droid_room.py execute` | S0 archive, `--frames` shot | `prediction.npz` (poses_c2w for every frame, keyframe c2w/depth/K/indices), `input-manifest.json` | A100-40GB, 30 s remote | B. The DROID frontend itself is causal, but the wrapper uploads a tar of the whole shot, sizes the buffer to frames+38, and consumers read the state after `terminate()` (global BA, PoseTrajectoryFiller, final upsample). Does not mask moving pixels | S0 |
| S2 | Motion masks | `modal_apps/motion_masks.py` | S1 poses/K (or `--fixed-camera --video`) | `NNNNN-moving.png`, `residual.npz` every 10th frame | 2 L4 calls (RAFT, then SAM 2.1 boxes), about 2 min | I per frame pair (gap 8) once poses exist | S1 (moving camera) |
| S3 | Mover tracks | `modal_apps/sam3_motion_tracks.py` | S2 seeds, `--frames A B` window, `--text person` | `tracks.json` + masks | A100-40GB, 160–200 s per window of about 300 frames | W. Propagates in both directions inside a window. Streaming needs forward-only tracking with a fixed lag | S2, S1 raster |
| S4 | Stitch, analysis, union masks | `scripts/stitch_track_windows.py`, `motion_tracks_to_analysis.py`, `assemble_dynamic_masks.py` | S3 windows in order | stitched ids, `analysis.json` (per object, per frame), `masks/SOURCEINDEX-N.png` | CPU | I per window (matches on mask IoU where windows overlap) | S3 |
| S5 | Posed depth | `modal_apps/mono_room.py infer` | S1 cameras. Views = keyframes + `--every/--midframes`, minus `--exclude-frames` | `mono/NNNNN.npz` | A100-80GB, 72 s for 155 views (DA3-GIANT, non-commercial, used in every demo); MoGe-3 option on L4 | B. DA3 does one posed forward pass over all views (up to 400). Could be W in chunks. Commercial use requires DA3-BASE | S1 |
| S6 | Floor masks | `scripts/discover_video_keyframes.py --prompt floor` (SAM 3 on L4) | S0 video, up to 12 hand-picked frames | `floor-*/frame-*/instance-*-mask.png` | L4 | I | S0 |
| S7 | Scale / floor plane | `mono_room.py metric` | S5 depth + S1 support, S6 masks, `--camera-height 1.6` (operator's stated guess) | `metric-scale.json` (plane, up, m/native, scale_status) | CPU | B, but only needs to run once per site (calibration) | S1, S5, S6 |
| S8 | TSDF fuse | `mono_room.py fuse` (Open3D) | S1, S5, `--dynamic-masks` S4, `--floor-plane` S7, `--voxel-length-native` computed by hand from S7 | `mono-anchored-mesh.ply`, points, `predicted-scene.glb`, `scene.json`, `fuse-metrics.json`, `droid-support.npz` | Mac CPU | B. The support vote runs over all overlapping keyframes (`--support-all-views`), and carving and floor support use all views. The TSDF itself can grow incrementally; the vote could be W with lag | S1, S4, S5, S7 |
| S9 | Moving-person surfaces | `mono_room.py dynamic` | S4 `analysis.json`, S8 `scene.json`, S5 mono. The run dir is put together by hand with symlinks | `dynamic/entity-*.glb`, timed `scene.json` | CPU | I per frame | S4, S5, S8 |
| S10 | Texture | `scripts/texture_fused_mesh.py` | S8 mesh, S1, `source-full.mp4`, S4 masks, `--overlay-rows` | `textured-scene.glb`, `texture-report.json` | CPU, 20 s | B per mesh version (picks the best view per triangle across all views) | S0, S1, S4, S8 |
| S11 | Hole fill | `scripts/fill_scene_holes.py` | S10 shell, S5 depth, S4, `--carve --step 2 --video` | `textured-scene.glb` (+`single-view-depth-fill*`), `fill.json` | CPU | B. Walks views in order, and the carve uses every view | S4, S5, S10 |
| S12 | Class-agnostic masks | `modal_apps/sam2_everything.py` | S0 frames, which must be S5 depth views (masks without depth are dropped), `--known` S6 | `object-*/frame-*/instance-N-mask.png`. The mask root is a hand-made symlink directory | L4 | I per frame | S0, S6. The frame list is coupled to S5 |
| S13 | Object map | `scripts/build_video_object_map.py --method overlap` | S1, S5, S12, `--floor` S7, `--dynamic-masks` S4 | `object-map.json`, `surfaces/*.npz`, cells | CPU | I in design (a ConceptGraphs-style accumulator: each entity keeps world points, a new mask joins one). Today it runs once as B | S1, S4, S5, S7, S12 |
| S14 | Naming | `scripts/name_video_entities.py` → Gemini, called through the deployed report container (`modal container exec`) | S13 entities seen in at least 3 views, S12 crops | labelled `object-map.json`, `names.json` | Cloud API, not on-prem | I per entity once it is confirmed | S12, S13 |
| S15 | Per-frame outlines | `scripts/project_entities_to_frames.py` | S1, S8 `scene.json`, S14, room GLB | `analysis.json` | CPU | I per frame | S1, S8, S11, S14 |
| S16 | Video memory (events) | `modal_apps/video_events.py` (Qwen3-VL-8B) | video, `--marks` S4, `--names` | `events.json` | A100-40GB | W (12 s windows) | S4 (optional S14) |
| S17 | LingBot inference | `modal_apps/lingbot_room.py prepare/execute` | `source-full.mp4`, `--frames` | predictions on a Modal volume | A100-80GB/H100/A100-40GB, 127 s (34 s inference) | B wrapper, limited to 8–768 frames. The model itself has a streaming mode (window 64). Licence undeclared | S0 |
| S18 | Dense point map | `scripts/lingbot_dense_map.py build` | S17, S1, S4, S5, S8 mesh, `--exclude-frames`, `--overlay-rows` | `dense-points.glb`, `point-attributes.npz`, `points.json` | Modal CPU, 510 s | B (Sim3 fit on all camera centers, multi-view contradiction vote) | S1, S4, S5, S8, S17 |
| S19 | ICP refine | `scripts/lingbot_icp_refine.py` | S18, S8 mesh, S7 scale | new map dir | CPU | B. The accept rule (at least 45% of points within 25 cm, floor within 5 cm) is applied by hand; it is not in the code | S7, S8, S18 |
| S20 | Splats | `modal_apps/splat_train.py` (+`--clean`, `--pick`) | S0 1280×720 frames, S1, S4, S8 mesh, S11 fill, S7 scale | `splats.splat/json`, `refined-cameras.json` | H100 (falls back to A100), 19 min for 60k steps, 2.5M Gaussians | B | S0, S1, S4, S7, S8, S11 |
| S21 | Complete object models | `scripts/complete_video_objects.py --generator sam3d\|recgen\|box` (+`modal_apps/sam3d_research.py`, `lucida-private-assets` RecGen, SAM 2.1 re-segmentation on L4) | S1, S5, S14 labels (selection needs a clear label and at least 8 views), S12, S4, full frames | `models/<id>/{model.glb, anchor.json, validation.json}`, `review/`, `manifest.json` | SAM 3D on A100-80GB: 2811 s for 145 calls. RecGen on A100-80GB (non-commercial): 4068 s for 34 calls. Box: CPU | B today. Could be I per object once an object stops getting new views (the gate needs at least 3 agreeing held-out views). Merging uses a one-off `merge-at-execution.py`, and box models are reviewed by eye | S0, S1, S4, S5, S12, S14 |
| S22 | Inferred floor | `scripts/infer_room_floor.py` | S7/S8, `--dense` S19, S1, `--skip-frames` | `inferred-floor.json` (+points GLB if it passes) | CPU | B. Checks itself and withheld the result on Sam's Club and Walmart | S1, S7, S8, S19 |
| S23 | EHS rules | `scripts/evaluate_video_policy.py` (ZEN engine) | S14, S7 | `policy-findings.json` | CPU | B, but cheap enough to rerun on every update. **Not wired into the ME340, Sam's Club or Walmart imports** (`--policy` left out) | S7, S14 |
| S24 | Import | `scripts/import_video_scene.py` (Postgres) | about 17 paths from S0–S22, `--exclude-frames`, `--title`, `--republish` | scene document and publication | CPU; 285 s up to about 60 min (Walmart with box models) | B. Builds a whole document; there is no append/patch path | all |
| S25 | Publish | `scripts/export_platform_publication.py` → `prepare_publication_site.py` → `modal deploy modal_apps/{publication_site,video_publication_site}.py` → `web/` build → push to GitHub Pages | S24 publication id, `--api 127.0.0.1:8792` | static catalog and site | CPU + deploy | B, a frozen static snapshot | S24 |
| Sx | Cut-shot registration (edited clips only) | `scripts/register_cut_shot.py` + `modal_apps/da3_unposed.py` | S1, S5, S0, `--shot`, `--walk-count` | camera registration, then a dynamic stage | A100-80GB | B | S0, S1, S5 |

**Dependency graph**
```
S0 clip/K ─┬─> S1 DROID ─┬─> S2 motion ─> S3 SAM3.1 windows ─> S4 stitch/analysis/union masks ─┐
           │             ├─> S5 DA3 posed depth ─────────────────────────────────────────────┐   │
           ├─> S6 floor masks ──> S7 metric(S1,S5,S6) ──────────────────────────────────┐   │   │
           ├─> S12 SAM2 masks (frames == S5 views)                                     │   │   │
           └─> S17 LingBot infer                                                        v   v   v
                                                    S8 FUSE(S1,S4,S5,S7) ── the join point for everything below
  S8 ─> S9 dynamic(S4,S5)      S8 ─> S10 texture(S4) ─> S11 fill(S5,S4)
  S12,S4,S5,S7 ─> S13 object map ─> S14 naming (Gemini) ─> S15 outlines(S8,S11)
                                                        └─> S21 objects(S5,S12,S4) ─> merge + eye review
  S17,S1,S4,S5,S8 ─> S18 dense map ─> S19 ICP(S8) ─> S22 inferred floor(S7,S1)
  S0,S1,S4,S7,S8,S11 ─> S20 splats (19 min)
  S4(+S14) ─> S16 events        S14,S7 ─> S23 policy (not wired in)
  {S1,S5,S8,S9,S11,S13-S16,S19-S22} ─> S24 import ─> S25 export/deploy/Pages
```
- **Critical path to first output:** S0 → S1 → S2 → S3 (per window) → S4 → S8 → S13 → S14 → S24. S8 also needs S5, and S7, which needs S6.
- **Longest tails:** S21 (47–68 min of generator wall time), S24 (up to 60 min) and S20 (19 min).
- **Coupling points that force batch today:**
  - S1 is read only after the final global BA.
  - S8 votes across all views and needs S4's masks before it can start.
  - S12's frames are tied to S5's depth views.
  - S21's selection waits for S14's names.
  - S18 and S19 fit against the finished S8 mesh.
  - S24 builds the whole document at once.
- **Fixed camera:** S1 has no parallax to work with, and S5/S7/S8 need a `droid_run`, or a `run.json` with `clip_definition.metric_cameras` and a constant pose. S2 and S3 already accept `--fixed-camera`. The only way to place a static camera today is Sx, and it needs an existing walk map. `ehs_spatial/video.py` (path P1) already applies per-frame zone, person-to-vehicle distance and speed rules to a fixed camera's tracks: SAM masks, ByteTrack, then a MoGe-2 floor-plane lift.
- **Not usable in a commercial or on-prem product:** DA3-GIANT (non-commercial) in S5 and Sx, RecGen (non-commercial) in S21, LingBot (licence undeclared) in S17–S19 and S22, and Gemini (cloud only) in S14.

**Per-clip values that are hard-coded or passed by hand**
1. **Absolute paths:** `ART=/Users/adam/Desktop/panoptes-public/research-notes/phase2` is written into `droid_room.py`, `prepare_video_clip.py` and `splat_train.py`. Run ids and output dirs are named by hand for every stage (for example `droid-samsclub-a2-280`).
2. **Shots and cuts:** cuts are found by hand (a mean absolute frame difference, computed in a one-off snippet), then passed as:
   - `--frames` (S1, S17)
   - `--exclude-frames` (S5, S7, S2, S18, S24)
   - `--skip-frames` (S21, S22)
   - `--skip` (S20)
   - `--shot 14:226` (default in `register_cut_shot.py`), `--walk-count 14`, `--cut-frames`, `--merge OLD=NEW`, `--person-tracks`

   A new clip dir with hard links and a re-estimated K is also made by hand.
3. **Lens and raster:**
   - The 640×480 4:3 raster is a constant everywhere.
   - FOV frame count: `FOV_FRAMES=5`, raised to 15 by hand on A2.
   - `lingbot_dense_map.py`: `FULL_WH=(1280,720)`, `CROP_X0=160.25`, `CROP_SCALE=1.5`.
   - `splat_train.py`: `W,H,SIDE=1280,720,160`.
4. **Burned-in captions (ME340 defaults):**
   - `splat_train.CAPTIONS=(648,704,160,1120)`
   - `lingbot_dense_map.SUBTITLE_ROWS=(646,706)`
   - `--overlay-rows` in S10, S11 and S18
   - `--no-captions` in S21
5. **Scale:**
   - `--camera-height 1.6` is the operator's guess.
   - `--voxel-length-native` is computed by hand from the metric scale (0.00567 = 4 cm / 7.055).
   - `complete_video_objects.VOXEL=.014` is ME340's value; `--voxel-native` is copied by hand from `fuse-metrics.json`.
   - Fuse settings: `--support-relative .02 --support-all-views --edge-jump .03 --carve`.
   - `fill_scene_holes.ON_SHELL=.6` was tuned on ME340.
6. **Frame and window picks:**
   - `discover_video_keyframes --frames`: at most 12, hand-picked.
   - `sam2_everything --frames`.
   - The mask root is a symlink directory made by hand.
   - `sam3_motion_tracks --frames A B` window bounds (for example 0–300 and 260–420), plus `--stride`.
   - `motion_masks --every 10 --gap 8`.
   - `mono_room --every/--midframes`.
   - `lingbot_room --stride` (default 3, 8–768 frames).
7. **`splat_train.py` defaults point at ME340 runs:** `CLIP`, `DROID`, `MESH`, `FILL`, `MASKS`, `SCALE`. `SHOW` held-out frames and `MODERATE` bench views are ME340 pixels. The cleanup rule for `--pick` is chosen by hand.
8. **Models (S21):**
   - `--function-id fu-Hh2leT3x1kprDaWpWsZ09l` is hard-coded.
   - `--entities` / `--exclude` review lists, `--count/--all`, `--max-usd`, and the `EXCLUDED` label list.
   - SAM 3D + RecGen + box results are merged by a one-off script.
   - Box models are dropped by eye (10 this session).
9. **Hand-applied gates:** the LingBot ICP accept rule. Which layers go into the import (floor withheld, `--policy` left out).
10. **Import and publish:**
    - About 17 path arguments, `--title`, `--republish`, and `PANOPTES_DATABASE_URL`.
    - Export uses `--api http://127.0.0.1:8792 --publication ID`.
    - Deploy needs the catalog/HTTP env vars and `modal deploy`, then a web build and a Pages push.
    - Naming requires exactly one deployed report container (`modal container list`).
    - The S5 dynamic run dir is assembled with symlinks by hand.

### latency
## Per-stage time and cost of the Panoptes offline pipeline, for the three demo clips

Every number comes from the run manifests and the cost ledger. For each clip I counted only the run that was delivered, not the attempts before it.

**How to read the numbers**
- **M (measured):** read from a field in a manifest (`wall_seconds`, `elapsed_seconds`, `remote_seconds`, `container_seconds`, `spend.gpuSeconds/wallSeconds`, `client_wall_seconds`, `build_seconds`).
- **F (file timestamps):** worked out from file creation and modification times. These are estimates, and they are distorted because several agents were running at once on the Mac.
- **E (my estimate):** Modal list price × seconds.
- **USD:** every dollar figure is list price × wall seconds, so it is an upper bound. The ledger has `actual_billing_reconciled: false`. The only billed figure is ME340's splat track: $13.29 "billed by Modal".
- **Rates used for E:** H100 $0.001097/s, A100‑80GB $0.000694/s (about $0.00094/s with 8 cores and 64 GiB), A100‑40GB $0.000583/s, L4 about $0.00025/s. Source: `walmart-splat-260/train.json` `usd_rates` and the cost ledger.
- **Clips:** ME340 is 899 frames at 30 fps (30 s; 687 of those frames are the walk shot). Sam's Club is shot A, 420 frames at 25 fps (16.8 s). Walmart is its second shot, 367 frames at 25 fps (14.7 s). **The Sam's Club and Walmart reconstructions cover only 15–17 s of video, not 30 s.**

| # | Stage (runs on) | ME340: wall / GPU s / USD | Sam's Club: wall / GPU s / USD | Walmart: wall / GPU s / USD | Basis |
|---|---|---|---|---|---|
| 1 | DROID-SLAM cameras (A100-40GB) | 90 s F / 55 M / ~$0.05 E | 50 s F / 30 M / $0.06 M | 61 s F / 30 M / $0.04 M | `remote-run.json elapsed_seconds` |
| 2 | DA3-GIANT posed depth (A100-80GB) | 150 M / ≤150 / ~$0.14 E | 72 M / ≤72 / $0.06 M | 80 M / ≤80 / $0.10 M | `infer.json wall_seconds` (client) |
| 3 | TSDF fuse (local CPU) | ~190 s F / 0 / 0 | ~50 s F | ~570 s F (competing with other jobs) | file times only |
| 4 | Texture (local) | 14.9 s M | 19.8 s M | 10.6 s M | `texture-report.json seconds` |
| 5 | Hole fill (local) | ~27 s F (+178 s held-out check M) | ~10 s F (+30.5 s check M) | not recorded | |
| 6 | Motion masks, RAFT (L4) | 143 s M / 143 / ~$0.04 E | 51 s M / 51 / included in the $0.26 in row 7 | 65 s M / 65 / $0.03 M | `motion.json flow+box_wall_seconds` |
| 7 | SAM 3.1 person/mover tracks (A100-40GB) | 1365 s M, 3 windows / 1298 M / ~$1.1 E | 356 s M, 2 windows / 320 M / ~$0.22 (with masks $0.26 M) | 310 s M / 264 M / $0.20 M | `tracks.json wall/remote_seconds` |
| 8 | Stitch, motion analysis, dynamic masks (local) | under 1 min F | under 1 min F | under 1 min F | |
| 9 | Dynamic person surfaces (local, reuses the DA3 depth) | ~215 s F | ~76 s F | not recorded (file times unusable) | |
| 10 | SAM 2.1 automatic masks (L4) | 755 s M / 755 / ~$0.19 E | 694 s M (run 160, reused) / ~$0.17 E | 773 s M (run 181, reused) / ~$0.19 E | `segment.json wall_seconds` |
| 11 | Object map build (local) | ~101 s F | ~76 s F | ~98 s F | |
| 12 | Gemini naming (cloud, requests sent one after another) | 239 s M (17 requests) / – / ≤$0.10 E | 190 s M (16) / – / $0.10 M | 189 s M (16) / – / $0.03 M | sum of `receipt.json client_wall_seconds` |
| 13 | Video events, Qwen3-VL-8B (A100-40GB) | 42 s M (incl. 11 s load) / ~$0.03 E | 49 s M | 63 s M | `events.json` |
| 14 | LingBot-Map: GPU call + diagnose + Modal-CPU build + check renders | 134 (inference 37) + 166 + 467 s M, about 13 min+ / $0.28 M | 127 (34) + 407 + 418 M, then rebuild 502 M: ~24 min / $0.40 M | 130 (52) + 420 + 405 M, ~24 min in file times / $0.21 M | `lingbot-run.json`, `diagnose.json`, `remote.json` |
| 14b | LingBot ICP refine (local) | – | not recorded | not recorded | |
| 15 | Gaussian splats, gsplat 60k steps (+ cleanup) | delivered run: 3091 s M on A100-80GB / 3067 / $2.34 M, + 43 s cleanup $0.12. On H100 the same run took 1344 s, $1.59. Whole splat track: **$13.29 billed + $4.28** | 1162 s M H100 / 1144 / $1.38 M, + 14 s cleanup $0.09 (two runs, old and new lens: $3.15) | 1214 s M H100 / 1198 / $1.45 M, + 19 s cleanup $0.09 | `train.json`, `clean.json` |
| 16 | RecGen (A100-40/80GB, one call at a time) | 11186 s M (153 calls, 2 rounds) / 8076 M / $10.55 M | 4068 s M (35 calls) / 1855 M / $3.84 M | 9800 s M (129 calls) / 6353 M / $9.22 M; directory spans 5.1 h F | `manifest.json spend` |
| 17 | SAM 3D Objects (A100-80GB, one call at a time) | 2573 s M (156 calls, median 14 s) / 1618 worker s M / $2.10 (+ $0.27 test run 240) | 2811 s M (146) / 1986 M / $2.30 M | 2567 s M (157) / 1613 M / $2.10 M; directories span 58–69 min F | journal `record.json` |
| 18 | Box fallback (local CPU) + checking each box by eye | ~743 s F + manual review | ~280 s F + manual review | ~1656 s F + manual review | review time never recorded |
| 19 | Inferred floor (local) | not recorded | not recorded | not recorded | |
| 20 | Cut-shot registration, DA3 unposed (ME340 only) | not recorded (reserve $3) | – | – | |
| 21 | Import, `build_document` | not recorded | 1439 s M (offline check) | 773 s M (offline check, before box models); the real import with box models took about 60 min per ARCHITECTURE.md | `*-quality-manifest.json` |
| 22 | Export, publication checks, Modal deploy, Pages | not recorded | not recorded | not recorded | |
| – | One-time setup: DROID image build | 1338 s CPU, cached and shared by all clips | | | `build.json` |

### End to end for one clip (delivered runs only, run one after another, machine time, no human waiting, no publishing)

| | Stages 1–15 (no object models) | Object models (RecGen + SAM 3D + box) | Import | **Total** | USD |
|---|---|---|---|---|---|
| ME340 (30 s of video) | ~2.1 h (the A100 splat run is 51 min of that) | ~4.0 h | not recorded | **≥ 6 h** | ~$17 delivered, ~$32 including all splat attempts (E) |
| Sam's Club (16.8 s) | ~68 min | ~2.0 h | 24 min | **~3.5 h** | $10.32 M (includes duplicate old-lens runs; about $8.5 delivered only, E) |
| Walmart (14.7 s) | ~86 min | ~3.9 h | 13–60 min | **~5.5–6.3 h** | $13.50 M |

Actual elapsed time by the clock (F):
- **Walmart:** 09-25 16:17 → 22:20, then a second session on 09-27 from 00:30 (boxes, review, merge, import).
- **Sam's Club:** 09-25 16:58 → 19:50, then 09-27 00:11 → 00:40.
- **ME340:** 09-24 00:59 → 09-27, with many iterations in between.

Even with every manual step automated but the same models kept, the longest chain of dependent stages (DROID → DA3 → fuse → SAM 2.1 → object map → names → SAM 3D → RecGen → box → import) is still about 4.6 h for Walmart (E).

### What dominates latency
1. **Per-object generation, one call at a time.** RecGen takes 1.1–3.1 h per clip and SAM 3D 43–47 min. Together that is about 55–65% of machine time and about 80% of the money (Walmart: $11.3 of $13.5). The per-call work is short (RecGen about 45 s GPU with 54–110 s wall; SAM 3D about 10–13 s worker time), but 130–160 calls run in series because of the "one GPU call at a time" budget rule. RecGen is also non-commercial, so it cannot ship in the product anyway.
2. **Gaussian splat training.** It takes 19–20 min on an H100 (51 min for ME340's delivered A100 run) and $1.4–2.3. It is a 60k-step batch job and cannot be made incremental. ME340 needed about 15 attempts, costing $17.6.
3. **LingBot-Map is slow in its CPU post-processing, not in the model.** Inference is only 34–52 s (7–12 fps). The remaining ~16–24 min is diagnose (166–420 s), the Modal-CPU build (405–502 s) and check renders (~6 min).
4. **Import.** `build_document` takes 13–24 min. The real import with box models took about 60 min, because merging the many box triangles when projecting them onto the floor plane is slow.
5. **SAM 2.1 automatic masks on an L4.** 11.5–13 min, about 2.5 s per view. It sits on the object-map path, so everything after it waits.
6. **SAM 3.1 tracks.** 5–23 min, about 0.7–1.4 fps; ME340's three windows took 1365 s.
7. **Human steps are the biggest clock-time cost, and none are measured.** These are picking cut frames, re-estimating the lens, ICP, merging model sets by hand, checking boxes by eye (10 dropped), and the import and publish commands.
8. **The stages that are already cheap:**
   - DROID: 30–55 s GPU (12–16 fps), 50–90 s wall.
   - DA3: 72–150 s.
   - Motion masks: 1–2.5 min.
   - Naming: 3–4 min, only because requests are sent one after another; they could run in parallel.
   - Events: under 1 min.
   - Texture and fill: under 30 s each.
   - Overhead beyond model time: DROID wall is 1.7–2× its GPU time, a LingBot GPU call is 130 s against 34–52 s of inference, and RecGen wall is 1.2–2.2× its GPU time.

Numbers that were never recorded: ICP refine, inferred floor, cut-shot registration, ME340 import, and export/publish/deploy. Every stage above also used DA3-GIANT (CC BY-NC), LingBot (licence undeclared) and cloud Gemini, so none of these timings come from a commercially licensed, on-prem-ready pipeline.

Sources:
- Run manifests: /Users/adam/Desktop/panoptes-public/research-notes/phase2/runs/ (droid-*, da3-posed-*, sam2-*-everything-*, the me340-*, samsclub-a*- and walmart-* runs listed above; the Sam's Club/Walmart cost breakdowns are in samsclub-quality-manifest.json and walmart-quality-manifest.json).
- Cost ledger: /Users/adam/Desktop/panoptes-public/research-notes/phase2/video-mvp/cost-ledger.json
- Docs: /Users/adam/.codex/worktrees/panoptes-phase2-video/docs/phase2/ARCHITECTURE.md

### manual
# Panoptes phase 2: how each manual step becomes automatic

Two probes ran on CPU and read files only (scripts in the scratchpad). One checks the cut detector. The other checks whether the current box gate fields can replace the eye review.

- **Cut detector:** it found all four hand-chosen cuts and nothing else among the candidates it checked (`cuts2.py`). For each pair of consecutive frames it counts ORB matches and the inliers of a RANSAC fundamental matrix (1.5 px).
  - Hand cuts: ME340 14 had 7 inliers where its neighbours had 1204, and 226 had 33 vs 1259. Sam's Club 420 had 16 vs 952. Walmart 383 had 7 vs 898.
  - Every non-cut frame with a large grey-level change kept at least 541 inliers. Lightning, which has no cut, gave nothing.
  - Histogram difference alone is not enough. Walmart 383 scores 0.29 and a non-cut frame scores 0.23; ME340 14 scores 0.28.
  - Limit: I only computed inliers for the 8 frames per clip with the largest grey-level change. The real test must run on every frame.
- **Box eye review:** the fields already in `validation.json` do not reproduce it (`boxes.py`). 45 boxes passed the gate: 32 were kept, 10 were dropped by eye and 3 were duplicates.
  - The best single cut, `observedAreaShare < 0.28`, drops only 4 of the 10 (ME340 184, Sam's Club 128 and 115, Walmart 041) and loses no kept box.
  - `silhouette_iou`, `unrefined.silhouette_iou`, `relative_depth_*`, `fitResidual*`, `entityCoverage` and `icpRotationDeg` overlap completely between kept and dropped. Examples: kept 002 has a p90 fit of 13.75 cm and dropped 021 has 18.29 cm; kept 167 and dropped 145 both have an unrefined IoU of 0.55.
  - So the rule needs new per-pixel measures (row 11).
- **Stage times on Sam's Club a2 (420 frames), from the run manifests:**
  - DROID-SLAM: 30.5 s on Modal, plus a one-time image build of 1339 s.
  - DA3 posed depth: 72 s. Texture: 20 s.
  - LingBot-Map: 502 s.
  - Gaussian splats: 1160–1350 s per training call.
  - SAM 3D Objects: 2811 s in total, about 106 s of wall time (27 s of GPU) per attempt.
  - Import: 285 s for Sam's Club v1, about 60 min for Walmart with box models.

**Streaming tiers** in the last column: **F** = every frame, under 100 ms. **K** = every keyframe, about 1 s. **B** = background, minutes, published as a new version of the report.

| # | Manual step | Automatic rule or model (code it builds on) | Test that must reproduce the hand decision | If the rule is wrong | Tier |
|---|---|---|---|---|---|
| 1 | Choosing cut frames (`--exclude-frames 14:226`, Sam's Club 420, Walmart 383) | Put a cut detector in `prepare_video_clip.py` and write the result to one `segments.json`. For each consecutive frame pair: ORB with 1500 features, ratio test 0.75, RANSAC fundamental matrix at 1.5 px. Frame i is a cut if its inlier count to frame i−1 is below 0.1 × the median of the pairs at i±2..3 and below 100 in absolute terms. A cross-fade shows up as a run of low-inlier pairs and becomes a transition span. Every stage reads `segments.json`. Today the syntaxes differ: import takes `14:226` with the end excluded, while `--skip-frames` takes `14-225` with the end included, which invites off-by-one errors. | Run on every frame of the 4 prepared clips. Must give exactly {14, 226}, {420}, {383} and {} for Lightning. Also splice two TUM clips at a known frame, with a hard cut and with a 15-frame fade: both must be flagged at the right place. | Missed cut: two places are merged into one map. Earlier this showed up as the "height drift" and "floor bend", and a whole shot got wrong poses. False cut: the walk is split in two; row 2 can merge it back. Row 5 is the second safety net. | F (10–20 ms on CPU) |
| 2 | Choosing the mapping shot (ME340 0–13 plus 226–898; Sam's Club 0–419; Walmart 383–749) | Merge segments that relocalise into each other: `pick_walk` finds SIFT matches, and `register_cut_shot.align` must pass its gate (centre residual ≤ 10% of the camera spread, rotation ≤ 3°). The mapping shot is the merged group with the largest DROID camera travel. Every other group goes to row 4. Live, this is "relocalise into the existing map, else start a new submap". | Must reproduce the three hand choices. ME340 0–13 must join 226–898. Sam's Club 0–419 must not join 420–749 (a different aisle). | Similar-looking aisles merged on appearance alone put a wrong aisle into the map. That is why merging needs the geometric gate, not match counts. | F/K |
| 3 | Estimating the lens again for each shot (Sam's Club: 60.1° when both shots were mixed, 55.3° from its own shot; ME340's 5-frame sample included a 51.3° frame from the cut-away) | FOV comes from MoGe-3 (MIT) on at least 15 frames of **one segment only**. The spread inside a single shot can be large: Sam's Club a2 per-frame values run 50.7–63.7° in two groups. So an interquartile-range cap alone would have rejected the value the hand accepted. Instead, score the candidates {median, Q1, Q3} by the fused floor-plane inlier share and the DROID residual, and keep the best. If none gives at least 90% floor inliers, or MoGe and DA3-BASE differ by more than 8%, mark the intrinsics `uncertain`. Then no measurements are taken; the scale contract stays at `model_estimated` or lower. A fixed streaming camera is calibrated once at install and its intrinsics are stored per camera. | Sam's Club a2 must choose about 55° (hand value 55.3°), and floor inliers must go from 72.7% to at least 95% (hand result 96.5%). ME340's mapping shot must not use the 51.3° frame. Walmart must stay at 54.1°. | A wrong focal length makes scale and angles wrong and shelves bend; the floor inlier share catches the large errors. A zoom during a shot looks like two FOV groups, so split the segment where the group changes. | Once per segment or camera |
| 4 | Placing the cut-away camera by hand (`register_cut_shot.py --shot 14:226`) | Run it automatically for every segment that is not the mapping shot. Take 3 frames spread evenly and choose walk frames with `pick_walk`. Keep the current gate, plus a median depth ratio in [0.9, 1.1]. **Licence:** the script calls `depth-anything/DA3-GIANT-1.1`, which is CC BY-NC; switch to DA3-BASE or the Apache-2.0 MapAnything. If a segment is refused, it produces no 3D. | ME340 14–226 with the Apache model must pass and land within 5 cm and 2° of `runs/me340-cutaway-register-304` `cutC2w`. The margin is thin: the rotation residual was 2.38° against a 3° gate, so a refusal, which leaves a blank, is an acceptable outcome. Negative control: a Walmart frame must be refused in the ME340 map. | A static camera in the wrong place puts its people in the wrong 3D position, so EHS distances are wrong. The residual gate is the guard. | Once per fixed camera |
| 5 | Deciding by eye how much of the camera path to trust (Lightning frames 96–450; "no jumps" for Sam's Club) | Mark frames untrusted when any of these holds: the step between frames is more than 5× its rolling median; camera height along gravity is more than 0.3 m from its 2 s rolling median; per-keyframe scale spread is above 5%. Keep the longest trusted span. | Lightning must give 96–450 ±15 frames. The ME340, Sam's Club and Walmart mapping shots must be fully trusted (Sam's Club worst jump 0.97×, Lightning 8.66×). | A smooth pitch drift passes the jump test; the Sam's Club height going from 0.9 to 2.6 m turned out to be the cut. Row 1 comes first so this does not happen. | F |
| 6 | The `--no-captions` flag, and the 1280×720 frame size and caption rows hard-coded in `lingbot_dense_map.py` | Detect overlays: pixels that stay almost constant for 2 s or more while the camera moves, plus the existing `subtitle_box` glyph test when it fires on at least 30% of frames. Write the overlay mask and the source size into `clip.json`, and have every stage read them there. A CCTV on-screen timestamp is caught by the same test. | ME340 must be detected as having captions. The result for every other clip must match the flag used in its run manifests. A clip with a bright floor must not trigger it. | A false caption drops object views. A missed caption lets caption text win the sharpness contest and end up as object texture. | Once per stream |
| 7 | Turning the dynamic and person layer on or off per clip (skipped at "0% people"; Sam's Club's standing worker went into the static map; ME340 #32 "man" is actually floor) | Person tracking is always on: SAM 3.1 person tracks, stitched with `stitch_track_windows.py`. A static entity whose masks overlap person masks by at least 30% in at least half its views becomes dynamic. A person-type name needs the same overlap; otherwise the name is refuted and the entity stays unnamed. | The Sam's Club worker in the first ~3 s must leave the static map. ME340 #32 must lose the name "man" (its overlap is 0–1%). The ME340 presenter must remain one identity across 687 frames. | A standing worker baked into the map raises false "blocked aisle" findings. A poster labelled as a person creates a false mover. | F |
| 8 | Naming with Gemini (cloud) and reviewing `selection.jpg` (`--entities` / `--exclude`) | Name on-prem with Qwen3-VL embedding plus reranker, fused by reciprocal rank (k=60), against a fixed EHS vocabulary. Name only when the score is above τ, set so precision is at least 0.9 on a labelled set; otherwise "unnamed object". Selection uses the `--all` rule in `candidates()`. Shape generation does not need names; EHS rules only read names that passed τ. | A labelled set of about 300 crops from the 3 scenes, labelled once, gives τ. On the 14-query probe, fused P@10 must stay at least 0.207. ME340 #32 must not be named "man". | A wrong EHS name, such as a false "fire extinguisher", applies the wrong rule. Policy: a blank is better than a wrong name. | B (seconds per entity) |
| 9 | Running RecGen, SAM 3D and box as separate hand-started runs | One job per entity. Box (CPU) and SAM 3D Objects (GPU, commercial use allowed) run in parallel through the same `assess()` gate. RecGen runs only behind a research-licence flag that is off by default. Every call is journaled, `--max-usd` stays, and `--reassess` still works. | Replaying the existing journals with no new calls must give exactly the per-object accept/reject of runs 231, 241, 264, 267, 291, 292, 300, 301 and 302. | Nothing new beyond the current gate; the risk is spend, so the budget guard stays. | B (~106 s per SAM 3D attempt) |
| 10 | Merging model sets with one-off snippets | Turn the `rule` text in `merge.json` into code. Keep models accepted by the gate. If both generators passed, keep the lower held-out `fitResidualNative`. Use a box only where no generator passed, and not when it is the same object as, or part of, a chosen model: 3D box IoU, or at least 50% of the box within the recorded distance (0.01134 native in Sam's Club), `PART_SHARE`. | Must reproduce the `choice` of all three `*-303-merged/merge.json` files exactly, including the duplicates Sam's Club 051 and 052 (to 030) and 095 (to 028), and the totals 22 / 67 / 40. | A duplicate or a part counted twice inflates the EHS inventory. | B (instant) |
| 11 | Reviewing box models by eye (10 dropped) | Apply the policy's free-space test to each box's silhouette, in the source view and in every held-out view. Label each rendered silhouette pixel as: on the mask; **overhang** (reliable depth beyond the box's far face by 8% + 5 cm); **foreign inside** (depth between the box's near and far faces, but outside the mask); **occluded** (depth in front of the near face); or no depth. Add **sourceFaceBacking**: the area share of box faces that face the source camera (n·v > 0.3) and are marked observed. Reject when overhang > a, foreign > b in at least 2 views, occluded > c, or sourceFaceBacking < d. Choose a–d like the floor gate: quiet on kept boxes, and flagging at least 50% of the silhouette of a control box that is 15% larger or shifted 10 cm vertically. It runs on CPU through `--reassess`. | All 10 eye-dropped boxes must be rejected, and at least 30 of the 32 kept boxes kept. A lost kept box is only a blank; a dropped box that survives fails the test. Thresholds fitted on two scenes must pass on the third. Each hand reason must match its measure: "sticks out" (ME340 021; Walmart 007, 041, 061) → overhang; hose coil and plug among cords (ME340 145, 175) → foreign inside; "seen face not backed" (ME340 184, Sam's Club 128, Walmart 059) → sourceFaceBacking; "only a triangle past a shelf beam" (Sam's Club 115) → occluded. | Too loose: a box that swallows the bench is shown as an object and misleads visually, even though it is never used for measurement. Too tight: fewer models, which is only a blank. | B (seconds per object) |
| 12 | Running LingBot ICP refine and its display gate by hand (≥45% of points within 25 cm, floor within 5 cm) | Always run `lingbot_icp_refine.py` after the camera-centre Sim3. Show the dense map only if the display gate passes **and** ICP moved it by at most 2° and 0.3 m. Otherwise show the fused surface only. | Sam's Club (61.6%, about 0 cm) and Walmart (74%, 0.9 cm) must pass. Walmart before ICP (−8.7 cm floor offset) must fail. A control shifted 10 cm vertically and turned 3° must fail before ICP and come back within 1 cm and 0.2° after it. | ICP sliding along a repetitive aisle passes the floor check, which only constrains height, and shows doubled objects. That is why the step cap is there. **LingBot-Map has no declared licence and took 502 s per clip**, so it stays off in the product; use DA3-BASE fused points instead. | B |
| 13 | Choosing splat settings and held-out frames by comparing runs | Fixed settings: long axis at most 4× the middle axis, mapping-shot frames only, held-out frames chosen by the runner, LingBot seed off. Show splats only if held-out PSNR is at least 27 dB; otherwise the viewer shows the mesh (it already falls back to the mesh away from the camera path). | Must reproduce 30.8 / 27.2 / 31.1 dB within ±0.5 dB. | Poor splats presented as a photo. | B (~20 min per training call) |
| 14 | Invoking floor inference by hand | Already decided by its own rule (noise at most 1%, raised-plane control at least 50%, hidden-block false rate at most 5%). The runner only has to call it. | ME340 must reproduce noise 0.37%, control 53.8% and a chosen gap of 1.0 m. Sam's Club and Walmart must switch it off themselves (Walmart control 46%). | Nothing new. | B |
| 15 | Hand-built import and publish commands, and inputs checked by hand (`--voxel-native`; an old `dynamic-masks.tar` reused; import checks only the frame name, not the pose digest) | One runner driven by manifests. Each stage records the sha256 of its inputs, including a digest of the poses, and reruns only when an input changes. Import refuses a pose mismatch. The existing publication and export checks gate publishing. Live, it publishes a new version of the scene document instead of re-importing everything. Box meshes reduced to 12 triangles for the planar projection would fix the ~60 min Walmart import. | Given only the MP4, the runner must reproduce 123/78/22 (ME340), 164/111/67 (Sam's Club) and 166/102/40 (Walmart) for detected / qualifying / modelled objects, and the export byte check must pass. Swapping in a DROID run with the same frame names must make import refuse. | Stale files from another run are reused silently and the report mixes two reconstructions. | B |

**A fixed streaming camera** needs the heavy 3D only once: rows 3 and 4 at install, then rows 9–14 in the background whenever the free-space test says something changed. Per frame it only runs rows 1, 5 and 7: tracking loss or camera switch, trajectory trust, and people and movers. EHS rules then run on the people against the stored static map. That is what gets output latency down to seconds.

Four parts cannot ship on-prem commercially until they are replaced:
- RecGen: off.
- DA3-GIANT in `register_cut_shot.py`: switch to DA3-BASE or MapAnything-apache.
- LingBot-Map: licence undeclared.
- Gemini naming: cloud only, replaced by Qwen3-VL.

Probe scripts (read-only):
- /private/tmp/claude-501/-Users-adam-Desktop-panoptes-public/1fd9a1db-e580-4bfc-8110-119a1cc38a99/scratchpad/cuts.py
- /private/tmp/claude-501/-Users-adam-Desktop-panoptes-public/1fd9a1db-e580-4bfc-8110-119a1cc38a99/scratchpad/cuts2.py
- /private/tmp/claude-501/-Users-adam-Desktop-panoptes-public/1fd9a1db-e580-4bfc-8110-119a1cc38a99/scratchpad/boxes.py

Code the rules build on:
- /Users/adam/.codex/worktrees/panoptes-phase2-video/scripts/complete_video_objects.py (`assess`, `box_mesh`, `FIT_GATE`)
- /Users/adam/.codex/worktrees/panoptes-phase2-video/scripts/register_cut_shot.py
- /Users/adam/.codex/worktrees/panoptes-phase2-video/scripts/prepare_video_clip.py
- /Users/adam/.codex/worktrees/panoptes-phase2-video/scripts/lingbot_icp_refine.py
- /Users/adam/.codex/worktrees/panoptes-phase2-video/scripts/infer_room_floor.py
- /Users/adam/.codex/worktrees/panoptes-phase2-video/scripts/import_video_scene.py

Hand-decision records:
- /Users/adam/Desktop/panoptes-public/research-notes/phase2/runs/{me340,samsclub-a2,walmart}-object-models-303-merged/merge.json

### streaming-models
# Streaming Panoptes: research on replacing each offline stage (2026-09-27)

Read-only research. I modified no files and made no GPU or Modal calls. I checked licences against GitHub and Hugging Face directly where noted, and took the local numbers from `research-notes/phase2/runs/*`.

## 0. Findings

1. **LingBot-Map is already a streaming model, but we run it as a batch job.** `modal_apps/lingbot_room.py` decodes the whole clip, uploads a tar and makes one GPU call. That call runs `model.inference_streaming(images)` over the whole tensor, and we get output only after the last frame. Upstream, `inference_streaming` is a causal loop: the first 8 frames run together as scale frames, then one `forward(num_frame_per_block=1, causal_inference=True)` per frame with a KV cache. So a live wrapper is small: keep the model and cache loaded and run one forward per incoming frame.
2. **Our LingBot speed is about 5–7 fps, against about 20 fps upstream.** We use the SDPA backend with no FlashInfer, no `--compile`, `camera_num_iterations=4`, and copy every frame back to the CPU. Cold starts roughly double wall time (below).
3. **LingBot-Map's licence is no longer undeclared.** The GitHub repo has an Apache-2.0 `LICENSE.txt`, added in commit ed0aee8b on 2026-04-17. Our pinned REV 849e690 (2026-09-08) is later than that. The Hugging Face model card has no licence field (`cardData.license = null`), so confirm the weights licence in writing.
4. **Licence blockers are already in the current demos:**
   - They use **DA3-GIANT-1.1**, which is CC BY-NC (`da3-posed-me340-176-fused/infer.json`; `mono_room.py:771`, `register_cut_shot.py:257`).
   - RecGen is non-commercial.
   - Gemini is cloud-only.
   - DA3-Streaming, used as shipped, pulls in DA3NESTED-GIANT (CC BY-NC) and SALAD (GPL-3.0).
5. **Split the product by camera type.**
   - **Fixed camera:** no SLAM. Build the static map rarely. Only people, movers and EHS rules run live.
   - **Moving camera:** needs live SLAM plus depth.
6. **Some stages cannot be live.** Splat training (about 1,400 s per run on an H100) and learned object completion (SAM 3D: 2,573 s for the ME340 object set) belong in a background tier. The live view is the observed, coloured LingBot point map, which follows the policy anyway: only observed geometry is used for measurement.
7. **"One-shot" means turning each manual step into an automatic gate** (section 5). The ~10 boxes dropped by eye this session become a labelled calibration set for two automatic checks.

## 1. Measured baseline (this repo's runs)

| Stage | Run | GPU | Time | Mode |
|---|---|---|---|---|
| LingBot-Map | `lingbot-room-official-cache-003-launch3-preview` | A100/H100 pool | 113 s inference for 681 frames (6.0 fps); 262 s wall | streaming algorithm, batch execution |
| LingBot-Map | `lingbot-room-original-cache-review` | same | 138 s for 681 frames (4.9 fps); 286 s wall | same |
| LingBot-Map | `lingbot-room-windowed-005-preview` | same | 96 s for 681 frames (7.1 fps); 227 s wall | windowed |
| LingBot-Map | `lingbot-walking-rgb-003` | same | 63 s for 287 frames (4.6 fps); 175 s wall | streaming |
| DA3 posed depth | `da3-posed-me340-176-fused` | A100-80GB | 150 s wall | batch, DA3-GIANT-1.1 |
| SAM 3.1 tracks | `lightning-sam31-tracks-143` | A100-40GB | 321 s wall (motion 224 s, text 51 s) | `propagation_direction: both`, compile off, FA3 off |
| Splat training | `me340-splat-232` | H100 | 1,405–1,461 s per run (plus 366 s anneal) | offline gsplat |
| SAM 3D Objects | `me340-object-models-241-sam3d` | – | 2,573 s wall for the object set | offline, one at a time |

Peak GPU memory for LingBot was 15–17 GB. About 110–150 s of each LingBot wall time is container start, weight hashing and tar extraction, not inference.

## 2. Stage tables

### 2.1 Online SLAM / streaming 3D reconstruction (moving camera)

| Option | Throughput | Latency | Licence (commercial?) | Maturity | Fits our rules? |
|---|---|---|---|---|---|
| **LingBot-Map (streaming, per frame)** | ~20 FPS at 518×378 over >10k frames with FlashInfer + compile (upstream). Ours: 5–7 fps (SDPA, no compile) | 1 frame after 8 warm-up frames | Apache-2.0 (GitHub LICENSE.txt); HF card has no tag | ECCV 2026 oral; repo updated 2026-09-08; 17k stars | **Yes.** Already integrated. Gives per-frame depth + K + pose. No loop closure; quality drops past 320 cached frames without keyframes; windowed mode for >3,000 frames |
| VGGT-SLAM 2.0 | 8.4 FPS on RTX 3090 (16-frame submaps); 6.3 FPS with open-set SAM 3; 3.5 FPS on Jetson Thor | ~2 s (one submap) | Code BSD-2. Default checkpoint `facebook/VGGT-1B` is non-commercial; `VGGT-1B-Commercial` is gated, commercial except military | MIT-SPARK; `main_realtime.py` with RealSense | Yes, if swapped to the commercial checkpoint. Has loop closure, which LingBot lacks |
| MASt3R-SLAM | 15 FPS | per frame | CC BY-NC-SA 4.0 | CVPR 2025 | **No** (non-commercial) |
| StreamVGGT | causal KV cache, reported faster than CUT3R; memory grows with sequence length | per frame | CC BY-NC-SA 4.0 | ICLR 2026 | **No** |
| CUT3R | ~16.6 FPS (KITTI, as reported by LONG3R) | per frame | CC BY-NC-SA 4.0 | CVPR 2025 | **No** |
| Spann3R | >50 FPS | per frame | CC BY-NC-SA 4.0 | 2024 | **No** |
| DROID-SLAM (online frontend) | frontend runs online; we run offline with global BA (`asynchronous: false`) | per frame (frontend) | BSD-3 | mature, repo idle since 2025-05 | Yes, but it gives poses plus low-res depth only, and dynamic scenes need masks |
| DROID-W (CVPR 2026) | ~10 FPS on RTX 3090; per-pixel dynamic uncertainty removes people and cars from the map | per frame | Repo Apache-2.0, but adapts DUSt3R code (CC BY-NC-SA) and uses Metric3D (BSD-2). **Verify** | new, 441 stars | Maybe; licence must be checked first |
| DPVO / DPV-SLAM | DPVO 60 FPS average on RTX 3090 with 4.9 GB; DPV-SLAM 1–3× real time with loop closure | per frame | MIT | mature (Princeton) | Poses only (sparse), so it needs a separate metric depth model. Good cheap fallback |
| DA3-Streaming | 8.5–10 FPS on A100 | 120-frame chunks (overlap 60), so seconds of latency | Code Apache-2.0, but defaults to DA3NESTED-GIANT-LARGE-1.1 (CC BY-NC) and SALAD (GPL-3.0) | Nov 2025 | **No as shipped.** Batch-over-chunks, needs ffmpeg frames first |

**Pick: LingBot-Map as a live per-frame service.** It is licence-clean, already wired into `lingbot_dense_map.py` and ICP, and its loop is causal already. First test FlashInfer, `--compile` and `camera_num_iterations` 1 vs 4 against the 20 fps claim. **Backup for 24/7 drift:** VGGT-SLAM 2.0 with VGGT-1B-Commercial, or its loop-closure idea applied over LingBot windows. **Cheap pose-only fallback:** DPVO (MIT). **Fixed cameras:** no SLAM. Set K once per camera ID and build geometry from a background plate (temporal median of motion-free frames).

### 2.2 Streaming depth

| Option | Throughput | Latency | Licence | Maturity | Fits? |
|---|---|---|---|---|---|
| **LingBot depth head** | comes free with the pose pass | per frame | Apache-2.0 | in use | **Yes.** Depth and pose are consistent by construction |
| DA3METRIC-LARGE / DA3MONO-LARGE (0.35B) | single-image, fast | per frame | **Apache-2.0** | Nov 2025 | Yes. Metric depth: `focal * out / 300` |
| DA3-BASE (0.12B) / DA3-SMALL | posed multi-view | batch of views | **Apache-2.0** | – | Yes. Commercial-safe replacement for DA3-GIANT in the async densify tier |
| DA3-GIANT/LARGE/NESTED | – | – | **CC BY-NC 4.0** | – | **No.** Used in the current demos |
| Video Depth Anything, Small (+ metric Small) | 9.1 ms/frame FP32 on A100 (32×518² window) | streaming mode caches temporal attention; 1 frame | Small: Apache-2.0; Base/Large: CC BY-NC | streaming mode is experimental, training-free; ScanNet d1 drops 0.926 → 0.836 | Yes (Small only), for fixed cameras |
| FlashDepth | 24 FPS at 2044×1148 | per frame | Code and HF card Apache-2.0; built on Depth Anything V2 (the ViT-L lineage is NC). **Verify** checkpoint lineage | ICCV 2025 | Maybe |
| MoGe-3 | single-image | per frame | MIT (per brief) | in `mono_room.py` | Yes |

**Pick:** LingBot's own depth for live. For the background densify tier, replace DA3-GIANT-1.1 with DA3-BASE (posed) or DA3METRIC-LARGE. **Fixed cameras:** DA3METRIC-LARGE or MoGe-3 on the background plate, recomputed hourly or when the scene changes, plus one scale calibration knob (a known dimension or floor plane).

### 2.3 Streaming segmentation / tracking (people, movers)

| Option | Throughput | Latency | Licence | Maturity | Fits? |
|---|---|---|---|---|---|
| **SAM 3.1 Object Multiplex** | 16 → 32 FPS on one H100 for medium-density scenes; ~7× faster at 128 objects than the Nov 2025 release. SAM 3 detector ~30 ms/image with 100+ objects on H200 | per frame with forward-only propagation | SAM License (commercial OK), gated on HF | Mar 2026; repo active | **Yes.** Today we run it bidirectionally and offline, with compile and FA3 off |
| sam3-realtime (community) | – | incremental frames | – | 1 star; OOMs after ~5 min at 480p on RTX 3090 | Reference only. Shows memory must be bounded |
| SAM 2.1 (streaming memory) | 39.7–91.5 FPS on A100 compiled (Large → Tiny) | per frame | Apache-2.0 | mature; community camera predictor (Apache-2.0, 600 stars) | Yes. Tracking only, no text detection |
| EdgeTAM | 22× faster than SAM 2; 16 FPS on iPhone 15 Pro Max | per frame | Apache-2.0 | 2026 | Yes. For edge / on-prem low-cost hardware |

**Pick: SAM 3.1 in forward-only streaming mode**, with compile and FA3 on, an EHS text vocabulary (person, forklift, pallet jack, cart, ...), re-detection every N frames and a bounded memory bank. Fallback is SAM 2.1 or EdgeTAM for tracking, with the SAM 3 detector only on keyframes. The first test is forward-only vs bidirectional quality on the Lightning clip (same seeds, compare track IDs).

### 2.4 Online open-vocabulary object mapping and naming

| Option | Throughput / mode | Licence | Maturity | Fits? |
|---|---|---|---|---|
| ConceptGraphs | needs offline reconstruction first; LLaVA captions per object | MIT | 2023–25 | **No** (not online) |
| HOV-SG | offline, hierarchical | MIT | 2024–26 | **No** (not online) |
| Clio (+ Hydra) | real-time scene graph onboard; posed RGB-D, ROS, CLIP | BSD-2 | MIT-SPARK, Hydra active | Partial. Needs RGB-D + ROS; task-driven clustering |
| OVO | online; posed RGB-D; SAM 2 + CLIP/SigLIP/PE | MIT, but its primary backend ORB-SLAM3 is GPL-3.0 | 100 stars | Partial. Copyleft backend |
| DualMap | online, dynamic scenes, ROS 2 | Apache-2.0, but uses YOLO-World (GPL-3.0) and FastSAM | 215 stars | **No** (copyleft deps) |
| VGGT-SLAM 2.0 open-set | SAM 3 + Perception Encoder, 368 ms per submap on RTX 3090 | BSD-2 + SAM License | – | Yes, as a design reference |
| Naming: SAM 3 concept prompts | fixed EHS vocabulary at detector speed | SAM License | – | **Yes** (on-prem) |
| Naming: Qwen3-VL | long-tail names, asynchronous, once per object | Apache-2.0 | – | **Yes** (on-prem replacement for Gemini) |

**Pick: build it in-house incrementally**, reusing the merge logic in `build_video_object_map.py`. Per keyframe: SAM masks → lift with LingBot depth and pose → merge into instances by 3D overlap plus track ID. An object is "stable" after at least 3 agreeing views, which matches policy tier B. Naming uses the SAM 3 vocabulary first and Qwen3-VL for the rest; Gemini becomes an optional cloud-only path. No open-source system is monocular, ROS-free and licence-clean.

### 2.5 Incremental Gaussian splatting

| Option | Throughput | Licence | Fits? |
|---|---|---|---|
| gsplat (current) | ~1,400 s per run on H100 | Apache-2.0 | Background only |
| Photo-SLAM | >30 fps (RTX 4090 in MGSO's comparison) | GPL-3.0 (ORB-SLAM3-based) | Risky for on-prem distribution |
| MonoGS | ~3 fps live monocular | Imperial non-commercial | **No** |
| MonoGS++ | 5.57× MonoGS | paper CC BY-NC-SA; code unclear | **No** |
| Gaussian-SLAM | not real time; RGB-D | MIT | **No** (no RGB-D) |
| Flash-Mono (ICLR 2026) | feed-forward poses + 2D Gaussian surfels, 10× faster than optimisation | code status unclear | Watch |
| StreamSplat (ICLR 2026) | feed-forward, 1200× faster than optimisation, dynamic scenes, up to 1,024 views | repo has no licence | **No** until licensed |
| AnySplat | feed-forward, batch | MIT | Batch only |
| DA3 Gaussian head | – | only on GIANT/NESTED (CC BY-NC) | **No** |

**Pick: no live splats.** The live display is LingBot's observed, coloured point map. Splats are display-only: a background gsplat job seeded from LingBot poses and points, on a short schedule, re-run per area when coverage changes.

### 2.6 Fast image-to-3D for objects

| Option | Speed | Licence | Fits? |
|---|---|---|---|
| **Gravity-aligned box (existing)** | milliseconds, no model | ours | **Yes. Default live answer**, with automatic gates (section 5) |
| SAM 3D Objects | ~31 s per object reported (4.1 + 9.7 + 13.8 + 3.4 s); ours 2,573 s for the set | SAM License | Yes, as an async tier |
| Fast-SAM3D | up to 2.67× faster, training-free | paper CC BY 4.0; repo has no licence | Not until licensed |
| TRELLIS.2 (4B) | ~3 s (512³) / 17 s (1024³) / 60 s (1536³) on H100; needs ≥24 GB | MIT, but nvdiffrast/nvdiffrec are NVIDIA Source Code License (non-commercial) | Geometry-only path, same gate |
| SF3D / SPAR3D | 0.5 s / <1 s | Stability Community (free under $1M revenue) | Conditional |
| TripoSR | <0.5 s, low quality | MIT | Weak |
| RecGen | – | non-commercial | **No** |

**Pick: box first, then learned models asynchronously.** Every stable object gets a box at once, judged by the same out-of-sample gate. SAM 3D (and a TRELLIS.2 geometry-only trial) runs in a queue for stable, high-EHS-value objects. The policy doc already shows the bottleneck is gate pass rate (pose and scale), not generator speed.

## 3. How LingBot-Map runs today (`modal_apps/lingbot_room.py`)

- **Flow:** `prepare` → `execute` → `collect`. `decode_frames` writes every `stride`-th frame (default 3) as a PNG, capped at 768 frames to stay under the 1,024-frame positional limit. With `--video`, decoding happens on Modal CPU.
- **Configuration:** `inference_streaming`, or `inference_windowed` with window ≤ 64 and overlap ≥ 8. `keyframe_interval = ceil(n/320)`, `num_scale_frames=8`, `kv_cache_sliding_window=64`, `camera_num_iterations=4`, `use_sdpa=True`, bf16 aggregator, `output_device=cpu`.
- **Modal settings:** one GPU call with a 900 s cap, `retries=0`, `max_containers=1`, `scaledown_window=2`. Every run is a cold start: model load, SHA of the weights, tar extraction.
- **When output arrives:** nothing is written until the whole sequence finishes. Then `native-prediction.pt` and per-frame NPZ files are saved.
- **Verdict:** streaming algorithm, offline execution.
- **What a live service needs:**
  - A warm container that loads the model once.
  - Frame ingest (RTSP or chunked upload).
  - Upstream's phase-1 / phase-2 loop run incrementally, emitting pose and depth per frame.
  - FlashInfer and compile turned on.
  - Keyframe interval plus windowed resets with Sim(3) stitching across window overlaps for unbounded streams.

## 4. Proposed tiers (latency budget)

| Tier | Target | Contents |
|---|---|---|
| T0 | ≤ 1 s | SAM 3.1 tracks → 3D foot points on the known floor → EHS events (zone intrusion, person–forklift proximity). Observed geometry only |
| T1 | ≤ 5 s | LingBot pose and depth → incremental observed point map / TSDF |
| T2 | ≤ 1 min | Object instances (stable at ≥ 3 views), names, box models through the automatic gates |
| T3 | minutes, async | SAM 3D completion, inferred floor/walls with refutation tests, gsplat, publication snapshot |

Streaming changes one rule: a later frame can refute an earlier inference. The free-space test counts "seen past it" views per cell as they accumulate, so the scene document needs versioned retraction (delete an inferred element when ≥ 2 views see past it).

## 5. Replacing each manual step with an automatic gate

| Manual today | Automatic replacement |
|---|---|
| Choosing cut frames (`--exclude-frames`) | A single live camera has no cuts. For edited uploads: PySceneDetect (BSD-3) or TransNetV2 (MIT), plus a LingBot pose-jump / confidence-drop detector, starts a new segment, then `register_cut_shot.py` runs automatically |
| Re-estimating the lens per shot | LingBot predicts K per frame: take the per-segment median. Fixed camera: cache per camera ID |
| ICP refine of the LingBot map | Runs on every window overlap; accepted only if the residual improves, otherwise keep the raw map |
| Running generators one by one, ad-hoc merges | A job queue fed by "object became stable". One merge rule per object ID: gate-passed learned > gate-passed box > blank |
| Eye review of boxes (~10 dropped) | Two automatic checks matching the two eye-rejection reasons: (a) the box silhouette projected into every view stays inside the object's mask dilated by N px (overflow ≤ ε); (b) the face the source camera sees is backed by observed points (coverage ≥ τ within ~2 cm). Calibrate N, ε, τ on this session's reviewed boxes. Switch eye review off only when the checks reproduce every eye rejection with 0 false rejections |
| Hand-built import / publish | Scene-document diff on every tier tick; publish a snapshot on a schedule |

## 6. Licence actions for a commercial / on-prem build

- **Remove:**
  - DA3-GIANT-1.1 from `mono_room.py` and `register_cut_shot.py` (use DA3-BASE or DA3METRIC-LARGE).
  - RecGen.
  - MASt3R-, CUT3R-, Spann3R- and StreamVGGT-based methods.
  - MonoGS.
  - DA3-Streaming's default weights.
- **Swap:**
  - VGGT-1B → VGGT-1B-Commercial if VGGT-SLAM is used.
  - Gemini → SAM 3 vocabulary + Qwen3-VL.
- **Verify:**
  - The LingBot weights licence (HF card has no tag).
  - DROID-W's DUSt3R-derived code.
  - FlashDepth's checkpoint lineage.
  - nvdiffrast in any TRELLIS.2 or SAM 3D texture path.
- **Copyleft to avoid in a shipped product:** Photo-SLAM, ORB-SLAM3, YOLO-World, SALAD.

## 7. Measure first (each is a bounded GPU call and needs the user's approval)

1. LingBot on a warm Modal H100 with FlashInfer + compile, `camera_num_iterations` 1 vs 4, per-frame loop. Target: ≥ 10 fps at 518×378.
2. LingBot drift over a 30+ minute walk without loop closure, to decide whether VGGT-SLAM-style loop closure is needed.
3. SAM 3.1 forward-only vs bidirectional on the Lightning clip.
4. The automatic box gates against the eye-review labels.
5. Warm-container cost for 24/7 on Modal vs an on-prem GPU per camera group.

## Sources

- https://github.com/Robbyant/lingbot-map ; https://raw.githubusercontent.com/Robbyant/lingbot-map/main/lingbot_map/models/gct_stream.py ; https://huggingface.co/robbyant/lingbot-map ; https://technology.robbyant.com/lingbot-map
- https://arxiv.org/abs/2601.19887 ; https://arxiv.org/html/2601.19887 ; https://github.com/MIT-SPARK/VGGT-SLAM
- https://github.com/facebookresearch/vggt (VGGT-1B-Commercial: https://huggingface.co/facebook/VGGT-1B-Commercial)
- https://edexheim.github.io/mast3r-slam/ ; https://github.com/rmurai0610/MASt3R-SLAM
- https://github.com/wzzheng/StreamVGGT ; https://arxiv.org/abs/2507.11539
- https://github.com/CUT3R/CUT3R ; https://arxiv.org/html/2507.18255 (LONG3R comparison) ; https://github.com/HengyiWang/spann3r
- https://github.com/princeton-vl/DROID-SLAM ; https://github.com/MoyangLi00/DROID-W ; https://openaccess.thecvf.com/content/CVPR2026/html/Li_DROID-SLAM_in_the_Wild_CVPR_2026_paper.html
- https://github.com/princeton-vl/DPVO ; https://arxiv.org/abs/2408.01654
- https://github.com/ByteDance-Seed/Depth-Anything-3 ; https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/da3_streaming/README.md ; https://huggingface.co/depth-anything/DA3-BASE
- https://github.com/DepthAnything/Video-Depth-Anything ; https://github.com/Eyeline-Labs/FlashDepth ; https://arxiv.org/html/2504.07093v2
- https://ai.meta.com/blog/segment-anything-model-3/ ; https://github.com/facebookresearch/sam3 ; https://github.com/facebookresearch/sam3/blob/main/RELEASE_SAM3p1.md ; https://github.com/Jeffjewett27/sam3-realtime
- https://github.com/facebookresearch/sam2 ; https://github.com/Gy920/segment-anything-2-real-time ; https://github.com/facebookresearch/EdgeTAM
- https://arxiv.org/pdf/2309.16650 (ConceptGraphs) ; https://github.com/hovsg/HOV-SG ; https://github.com/MIT-SPARK/Clio ; https://arxiv.org/abs/2404.13696 ; https://github.com/MIT-SPARK/Hydra ; https://github.com/tberriel/OVO ; https://arxiv.org/abs/2411.15043 ; https://github.com/Eku127/DualMap
- https://arxiv.org/html/2409.13055v3 (MGSO; Photo-SLAM / MonoGS speeds) ; https://github.com/HuajianUP/Photo-SLAM ; https://github.com/muskie82/MonoGS ; https://arxiv.org/abs/2504.02437 ; https://github.com/VladimirYugay/Gaussian-SLAM ; https://arxiv.org/abs/2604.03092 ; https://arxiv.org/abs/2608.01659 ; https://github.com/DSL-Lab/StreamSplat ; https://github.com/InternRobotics/AnySplat ; https://github.com/nerfstudio-project/gsplat
- https://github.com/facebookresearch/sam-3d-objects ; https://learnopencv.com/sam-3d/ ; https://arxiv.org/abs/2602.05293 ; https://github.com/microsoft/TRELLIS.2 ; https://github.com/NVlabs/nvdiffrast ; https://stability.ai/news-updates/introducing-stable-fast-3d ; https://huggingface.co/stabilityai/stable-point-aware-3d ; https://github.com/VAST-AI-Research/TripoSR
- https://github.com/QwenLM/Qwen3-VL ; https://github.com/Breakthrough/PySceneDetect ; https://github.com/soCzech/TransNetV2 ; https://github.com/serizba/salad ; https://github.com/AILab-CVC/YOLO-World ; https://github.com/UZ-SLAMLab/ORB_SLAM3

Local files read:
- /Users/adam/.codex/worktrees/panoptes-phase2-video/modal_apps/lingbot_room.py
- /Users/adam/.codex/worktrees/panoptes-phase2-video/modal_apps/sam3_motion_tracks.py
- /Users/adam/.codex/worktrees/panoptes-phase2-video/modal_apps/mono_room.py
- /Users/adam/.codex/worktrees/panoptes-phase2-video/docs/phase2/INFERENCE-POLICY.md
- /Users/adam/Desktop/panoptes-public/research-notes/phase2/runs/{lingbot-room-*,lingbot-walking-rgb-003,da3-posed-me340-176-fused,lightning-sam31-tracks-143,me340-splat-232,me340-object-models-241-sam3d}/

### platform
# Live updates for the Panoptes report platform and viewer: current architecture, gaps and a minimal design

**Summary**
- There is no push channel anywhere today: no SSE, WebSocket or LISTEN/NOTIFY. Each change produces a new full immutable scene document. A viewer sees a change only by polling jobs, then downloading the whole head revision and reloading every GPU asset.
- Public reports are a separate offline product. You export a publication, verify every byte, pre-encode the responses (59 s on the real catalog), then run `modal deploy` with the catalog built into the image. That path can never reach "seconds".
- Proposed fix: split a **live plane** from a **record plane**.
  - Live plane: one append-only, gap-free event log per branch, small patches and short track segments, delivered to viewers by long-poll.
  - Record plane: periodic immutable checkpoint revisions and publications, built from the log in the background.
- The audit trail becomes the immutable log plus its checkpoints. The existing immutability triggers, content-addressed blobs and publication byte checks are reused unchanged.

Everything below was found read-only. Paths are relative to `/Users/adam/.codex/worktrees/panoptes-phase2-video`.

---

## 1. Current architecture

### 1.1 Storage (Postgres + content-addressed blobs)

**Tables** (`ehs_spatial/platform/migrations/001_core.sql`)
- `projects`: one write capability hash per project. There is no read capability.
- `scene_branches`: a mutable `head_revision_id`.
- `scene_revisions`: the full scene document stored as `jsonb`, plus `document_sha256`.
- `edit_batches`, `captures`, `jobs` (lease and heartbeat added in `003_job_leases.sql`), `assets`, `model_calls`, `publications`, `agent_turns`.
- Immutability: the `panoptes_immutable()` trigger, created at 001_core.sql:99-104, blocks UPDATE and DELETE on `scene_revisions`, `edit_batches`, `publications`, `assets` and `captures`. Policy tables have their own immutable trigger in `002_policy.sql`.

**How changes are committed**
- Branch updates use compare-and-swap: `_branch()` (postgres.py:125-129) raises `revision_conflict` 409 unless `baseRevisionId == head`.
- `finish_job` advances the head only if the head has not moved since the job started (postgres.py:461-463). Otherwise the job's revision is stored but `head_advanced=false`.
- Every edit applies the whole document: `apply_operations()` deep-copies it and returns an inverse that is a **full copy of the previous document** (`restoreDocument`, repository.py:329). `commit_edits` stores both copies (postgres.py:228-262).

**Blob checks**
- `_insert_revision` reads and re-hashes **every blob the document references** on every new revision (postgres.py:156).
- `_register_asset` already verifies the bytes once, when the asset is registered (postgres.py:274).
- `create_publication` re-reads every blob again (postgres.py:631) and freezes an `assetManifest` of `{assetId, sha256, sizeBytes, mediaType}`.

**Blob store** (`ehs_spatial/platform/storage.py`)
- Keys are `sha256/<hex>`, bytes are verified on every `get`, and both the local and S3 stores are write-once.
- S3 URLs are presigned for 300 s.

### 1.2 Write path used by the video pipeline

`scripts/import_video_scene.py:504-548` writes straight to the repository, not through HTTP:
1. `create_project` (the capability is saved to `.platform/imports/*.json`).
2. `create_job`, then `claim_job`.
3. `finish_job(document=…)` with the whole built document.
4. `commit_edits([migrateScene v2])`.
5. `create_publication`.

Moving people are stored as per-frame representations with `timeRange`, one asset each (lines 391-419).

### 1.3 Platform API (`ehs_spatial/platform/api.py`, served by `runtime.application()`)

- REST only. All GET routes are **unauthenticated**, for example `GET /api/projects` (line 116, lists every project) and `/api/revisions/{id}` (line 136).
- The background "outbox" loop polls every 2 s, and only for job dispatch and recovery (lines 76-82).
- The only cursor-style feed is `GET /api/projects/{id}/agent-turns?afterSequence=` (lines 246-247). It is backed by the `agent_turns.sequence` identity column (003_job_leases.sql).
- `GET /api/blobs/sha256/{sha}` serves content-addressed bytes with `immutable` cache headers.

### 1.4 Publication site (the static audit and sharing path)

1. `scripts/export_platform_publication.py` writes `bundle.json` plus `blobs/<sha>` **per publication directory** (line 105), so bytes are duplicated across publications.
2. `publication_site.read_catalog()` re-verifies identities, the manifest and every blob hash. `compile_catalog()` pre-gzips each route response and writes `index.json`.
3. `scripts/prepare_publication_site.py` runs step 2.
4. `modal_apps/video_publication_site.py` builds the catalog into the image with `add_local_dir(catalog)` (line 27). It runs with `max_containers=1` and `@modal.concurrent(max_inputs=4)` (line 44). Every new snapshot therefore means a redeploy.

Measured in `docs/platform/REPORT-LOADING-20260916.md`: preparation takes 59.4 s, and the first request after a redeploy takes 6.8 s to first byte.

### 1.5 Web viewer (`web/src`)

- **Loading a report** (`WorkcellReport.tsx:212-330`): a publication loads `/api/publications/{id}/view` plus `/api/publications`; a project loads `/api/projects/{id}` plus `/api/revisions/{id}`. On load, `core.ts:56 currentPublicationURL` redirects to the newest publication on the same branch.
- **The only "live" behaviour**: in project mode, the job list is polled every 4 s, and only while jobs are pending or running (`WorkcellReport.tsx:339-400`, `App.tsx:2153-2215 TaskStrip`). When a job advances the head, the viewer downloads the full project and the full revision again.
- **Auth**: `api.ts:103` sends `Authorization` only on non-GET requests.
- **3D scene updates**: `viewer/native-viewer.ts:332-371 setScene`. Any change to a representation or asset signature triggers `release()` of **all** GPU meshes and a re-download of every asset.
- **Moving objects**: visibility is keyed to `timeRange` and driven by `videoClock` and the `panoptes:video-time` event (`VideoView.tsx`, `native-viewer.ts:14-21`).
- **Reusable piece**: `setStreamMesh(entityId, mesh)` (`native-viewer.ts:316`) replaces one entity's buffers in place. Only `web/experiments/video-mvp/scene.ts:219,247` uses it.

### 1.6 Measured size of one 30 s clip

Publication `d4fea10b` in `.platform/video-publication-catalog`:

| Item | Value |
|---|---|
| Scene document | 5.8 MB (static part alone 5.5 MB) |
| Entities | 169, of which 5 dynamic |
| Timed surfaces | 433, averaging 64 KB each |
| Assets | 2,537 (364 MB) |
| Cameras / observations | 293 / 1,245 |
| Document breakdown | entities 3.2 MB, asset list 1.8 MB, observations 0.57 MB |
| On disk (3 bundles in the catalog) | 389–725 MB each; `bundle.json` 38–100 MB |

---

## 2. Gaps for a continuously streaming camera

| # | Gap | Where | Consequence |
|---|---|---|---|
| G1 | The unit of change is the whole document, plus a second full copy as the inverse | repository.py:329, postgres.py:228-262 | About 11 MB written per update at the 30 s size, and the document keeps growing. One update every 2 s is roughly 20 GB/h, getting worse over time. |
| G2 | Every revision re-hashes every referenced blob | postgres.py:156, 631 | Commit latency grows with total scene bytes (2,537 S3 GETs, about 364 MB per commit). |
| G3 | One head pointer with strict compare-and-swap | postgres.py:125-129, 461-463 | Concurrent producers (tracker, namer, geometry) get 409s or `head_advanced=false`, and their results are silently dropped. |
| G4 | Moving people are stored as per-frame document representations with one asset per frame | import_video_scene.py:391-419 | The document grows without bound (about 14 surfaces/s). There is no rolling window. |
| G5 | Cameras, observations and the asset list grow with keyframes | same document | Even the static part grows linearly on a moving camera. |
| G6 | No push channel | api.py, web/src | The viewer notices nothing unless a job is running. It then pulls MBs and `setScene` reloads the whole GPU scene. |
| G7 | Publications need export, full byte verification, pre-encoding and a Modal redeploy; blobs are copied per publication | export_platform_publication.py:105, video_publication_site.py:27 | Minutes per snapshot, gigabytes duplicated. It is also unusable for live traffic at `max_inputs=4`. |
| G8 | Reads are unauthenticated | api.py GET routes, api.ts:103 | Live footage and body meshes of workers would be readable by anyone who has an ID. `/api/projects` lists every project. |
| G9 | No ingest API; producers need direct DB access and a capability file | import_video_scene.py | Producers cannot push without hand-run steps. |
| G10 | Policy evaluation is tied to one static `scene_revision_id` | policy_repository.py:161-172 | Time-based EHS events (person in zone at time t) have no place in the store. |

---

## 3. Minimal design: live plane plus record plane

```
producers (fixed/moving camera workers)
  │  POST blob (bytes)         → blobs.put + register_asset   (verified once, as today)
  │  POST event {patch|segment} → validate + append under branch row lock
  ▼
scene_events (append-only, immutable trigger, gap-free per-branch seq)
  │                                   │
  │ long-poll GET ?after=seq (≤0.5 s) │ materializer in existing outbox loop (every ~60 s)
  ▼                                   ▼
live viewer: merge patch, stream     scene_revisions checkpoint (full doc, immutable)
segments via setStreamMesh           → create_publication(checkpoint) when asked or on schedule
                                     → export/prepare/deploy stays offline and asynchronous
```

### 3.1 Migration `005_live_events.sql` (one table, one column, reuses the existing trigger)

```sql
ALTER TABLE scene_branches ADD COLUMN IF NOT EXISTS live_sequence bigint NOT NULL DEFAULT 0;
CREATE TABLE IF NOT EXISTS scene_events (
 project_id uuid NOT NULL REFERENCES projects(id), branch_id uuid NOT NULL,
 sequence bigint NOT NULL,                       -- per-branch, gap-free (see below)
 stream_id uuid NOT NULL, producer text NOT NULL, producer_seq bigint NOT NULL,
 kind text NOT NULL CHECK(kind IN ('patch','segment','checkpoint')),
 t0 double precision, t1 double precision,       -- source time; segments only
 body jsonb NOT NULL, body_sha256 char(64) NOT NULL,
 created_at timestamptz NOT NULL DEFAULT now(),
 PRIMARY KEY(branch_id, sequence),
 UNIQUE(branch_id, stream_id, producer, producer_seq),  -- idempotent producer retries
 FOREIGN KEY(project_id,branch_id) REFERENCES scene_branches(project_id,id));
-- plus: add 'scene_events' to the panoptes_immutable() trigger loop
```

**Ordering:** the sequence is assigned with `UPDATE scene_branches SET live_sequence=live_sequence+1 … RETURNING` inside the append transaction.

**Why not an identity column** (like `agent_turns.sequence`): identity values can commit out of order, so a reader using `after=N` could skip an event forever. The row lock makes the order per branch equal to the commit order.

*Ceiling:* one lock per branch allows a few hundred appends/s. That is plenty for a handful of producers per camera.

### 3.2 Event kinds (three)

**`patch`**: static scene changes.
- Body: `{upsert:{entities:[…],observations:[…],assets:[…],cameras:[…],annotations:[…]}, remove:{entities:[ids],…}}`.
- The server merges the patch into its cached head and runs the existing `validate_document()` (with identity and part-relation validation). This is the same trust level as today's `import_scene` job documents.
- Merging by id is about 10 lines on each side, so the viewer never needs a TypeScript port of `apply_operations`.
- Producers should batch changes into at most about one patch per second.

**`segment`**: one moving entity's track (a person, forklift or the moving camera) over `[t0,t1)`, about 1–2 s.
- Body: `{entityId, assetId, sha256, provenance:'observed'|'inferred', gate?:{name,version,result}}`.
- The asset is one packed file for the whole window (poses, keypoints and surfaces for N frames).
- Segments are **not** added to the document. The document holds only the dynamic entity (`motion:"dynamic"`, label). This fixes G4.

**`checkpoint`**: `{revisionId}` records that the head at this sequence was frozen as a `scene_revision`.

**Inference-policy hooks** (server-side, rejected with 422):
- An `inferred` segment or representation is rejected unless it carries a `gate` naming a refutation test that passed.
- Measurement routes (`scene_measurements.register_measurement_routes`) keep reading only observed kinds.
- Patches may not change `placementState` to `confirmed`. Confirmation stays a human `confirmPlacement` edit.

### 3.3 Server changes (all in the existing `api.py` and `postgres.py`)

1. **Blob upload:** `POST /api/projects/{id}/blobs` (Capability auth) calls `blobs.put`, then `register_asset`. Bytes are verified once here.
2. **Append:** `POST /api/projects/{id}/branches/{bid}/events` (Capability auth). It:
   - checks idempotency on `(stream, producer, producer_seq)`;
   - takes the branch row lock;
   - for a `patch`, applies it to the cached head and validates;
   - for a `segment`, checks that the asset row exists in this project and its sha256 matches (**no re-read of the bytes**);
   - inserts the row and bumps `live_sequence`.

   The head cache is keyed by `(branch, live_sequence)`. On a miss it is rebuilt from the latest checkpoint plus the patch events after it, so several API processes stay consistent.
3. **Read:** `GET /api/branches/{bid}/events?after=N&wait=25` is a long-poll.
   - It returns rows as soon as `sequence>N` exists, or an empty result after 25 s.
   - The handler must be **`async def`**: it awaits `asyncio.sleep(0.5)` between `asyncio.to_thread` DB checks. A sync handler would hold one of anyio's roughly 40 threadpool slots per waiting viewer.

   *Ceiling:* one DB poll every 0.5 s per viewer. Switch to one `LISTEN`ing connection per process that wakes all waiters once there are dozens of viewers.
4. **Initial load:** `GET /api/branches/{bid}/head` returns `{liveSequence, document}` from the cache. The viewer loads this once, then follows the events.
5. **Materializer:** added to the existing outbox loop (api.py:76). Every about 60 s, or on request, it:
   - freezes the cached head with `_insert_revision(parent=previous checkpoint, label="live_checkpoint")`;
   - moves `head_revision_id` to it;
   - appends a `checkpoint` event.

   No `edit_batch` is written: the events between the two checkpoints are the audit trail. This removes the full-copy inverse (G1).
6. **Stop re-hashing already-verified blobs:** in `_insert_revision` (postgres.py:156), re-read only the asset IDs that are **not** in the parent revision. Previously referenced assets were verified at registration or in the parent. This keeps the guarantee and makes cost proportional to new bytes (G2).
7. **Read capability** (G8):
   - Add `view_capability_sha256` to `projects` and require `Capability` on the live GETs (`/head`, `/events`).
   - `api.ts` sends the header on GET when a capability is passed explicitly (today it does so only for non-GET, at line 103).
   - Do not use `EventSource`: it cannot send headers, and putting a capability in the URL would leak it into logs.
   - The old unauthenticated GETs stay as they are until you decide to lock them down (see question 3).
8. **Bounded head** (G5): the live head keeps at most K keyframe cameras and at most 8 observations per entity (importer constant `EVIDENCE_PER_AGNOSTIC=8`), selected by the producer. Older keyframes live only in checkpoints and the log.

### 3.4 Viewer changes (`web/src`)

1. **Live route** `#/live/<projectId>/<branchId>`: load `/head`, then loop on `request("/api/branches/…/events?after="+seq+"&wait=25")` using the existing `request()` and its error handling. This replaces the job-status polling for this route.
   - `patch`: merge by id into `revision.document` and call `setScene`.
   - `segment`: fetch `/api/blobs/sha256/<sha>`. It already exists and is cached as immutable, so there is no per-asset metadata lookup. Keep a rolling window of about 2 minutes per entity.
   - `checkpoint`: record the revision ID for "freeze / share this moment".
2. **Incremental `setScene`** (`native-viewer.ts:332-371`): replace `release()` plus reload-all with a keyed diff on the existing `taskKey` entries `[entityId, repId, assetId]`. Keep resident meshes whose key is unchanged, release keys that disappeared, and load only new ones. This is the largest single edit, and it stays inside one function.
3. **Live clock:** drive the existing `videoClock` and `panoptes:video-time` from wall time minus a small buffer (about 2 s, so segments have arrived) instead of a `<video>` element. Moving entities call the existing `setStreamMesh(entityId, meshAt(t))`, as the `video-mvp` experiment already does, so there is no new rendering path.
4. **Live video:** show the latest keyframe image, or an HLS/WebRTC stream from the camera gateway. Do not put video segments in the scene store.

### 3.5 Immutable audit snapshots (record plane, kept off the latency path)

- **Audit record** = the append-only `scene_events` rows plus the checkpoint `scene_revisions`. Both are protected by `panoptes_immutable()`.
- **"Publish this moment"** = `create_publication(sceneRevisionId=<checkpoint>)`, unchanged, with one addition. The snapshot gets `liveEvents:{branchId, fromSequence, toSequence}`, and the segment assets in that range are added to `assetManifest`. The existing sha and size checks, `export_platform_publication.py`, `read_catalog` and `compile_catalog` then cover them without changes.
- For an on-prem live product, the platform API already serves `/api/publications/{id}` and `/view` with immutable cache headers, so the Modal static site is needed only for public, database-less sharing.
- If the static site keeps being used for frequent snapshots, make two changes:
  - use one shared `catalog/blobs/<sha>` instead of one copy per publication (export_platform_publication.py:105);
  - mount the catalog from a Volume instead of building it into the image (video_publication_site.py:27).

### 3.6 Latency budget (platform part only; estimates, not measured)

| Step | Estimate |
|---|---|
| Upload a 64 KB–0.5 MB segment | 0.1–0.3 s |
| Append (lock + validate; about 50–200 ms for a patch on a ~3–5 MB head) | 0.05–0.2 s |
| Long-poll wake | ≤ 0.5 s |
| Viewer fetches the segment | 0.1–0.3 s |
| **Total** | **about 1–1.5 s after the producer finishes a segment** |

Everything else is model compute, which the other agents cover.

### 3.7 Checks to leave behind (one each, small)

- Concurrent appends from 2 producers give a gap-free `sequence`, and a reader using `after=N` never skips a row.
- Replaying the latest checkpoint plus the patches gives a document whose `digest()` equals the cached head.
- An `inferred` segment without a `gate` returns 422. A `patch` setting `placementState:"confirmed"` returns 422.
- `_insert_revision` reads only the new blob IDs (injected blob store that counts `get` calls).
- A web check that merges a patch by id and confirms the incremental `setScene` keeps unchanged GPU keys resident.

### 3.8 Deliberately skipped

- **WebSocket/SSE:** long-poll through the existing `request()` avoids the header and CORS problems and works through any proxy. Add SSE only if 0.5 s turns out not to be enough.
- **TypeScript port of `apply_operations`:** patches already carry the resulting entities.
- **Separate time-series database:** Postgres rows plus content-addressed segment blobs handle about 5 segments/s per camera (roughly 18k rows/h).
- **Continuous EHS evaluation (G10):** a later step. Time-based findings would be a fourth event kind, `finding`, pointing at segment ranges.
- **Retention and deletion of body data:** needs a policy decision from you, not code.

---

## 4. Decisions needed from you

1. **Fixed vs moving camera first.** A fixed camera makes G5 trivial (one camera, and the static head barely changes). A moving camera needs the keyframe cap from 3.3 item 8.
2. **Checkpoint cadence.** Time-based (for example every 60 s), event-based (for example when a new confirmed object appears), or both. This sets how fine-grained the audit trail is.
3. **Lock down the existing unauthenticated GETs** (`/api/projects`, `/api/revisions/{id}`) before any worker footage lands in this database, not just add auth on the new live routes.