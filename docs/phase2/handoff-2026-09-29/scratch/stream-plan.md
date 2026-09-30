# Panoptes Live: how we get to one-shot streaming

**直接回答：** 把现在的离线流程全自动化也不够快：就算没有任何手工步骤，Walmart 最长的依赖链仍要约 4.6 h (E)。所以要拆成两层。**实时层**只输出观测到的东西：人、移动设备、EHS 告警、不断长大的观测地图，秒级。**精修层**就是现在整条流程，自动化后在后台按窗口/区域持续跑；它的结果只有通过自动闸门，才会替换进场景。固定摄像头的重 3D 只在安装时做一次。**"一次成型"**的意思是：每个摄像头一行配置、一条命令；每个手工决定都变成代码规则，并带一个必须重现本次手工结果的回放测试；闸门还没证明的推断默认关闭（留空）。

(M) means measured in our runs. (E) means an estimate or a vendor claim we have not reproduced.

## 1. Verdict

| Criterion (5 = best) | Latency-first | Two-tier | On-prem-ops |
|---|---|---|---|
| Time to first output | 5 | 4 | 4 |
| Steady-state latency | 5 | 4 | 4 |
| Correctness under the inference policy | 3 | 5 | 5 |
| Completeness (LingBot-level map, complete objects, people) | 4 | 5 | 3 |
| Commercial / on-prem viability | 3 | 4 | 5 |
| Cost per camera-hour | 3 | 4 | 5 |
| Implementation risk / effort | 2 | 4 | 3 |
| **Total / 35** | **25** | **30** | **29** |

- **Latency-first:** it has the tightest budgets, but it has three weaknesses:
  - Its first milestone only measures things.
  - It does not hold boxes off until their gate passes.
  - It depends on vendor fps we have not reproduced. SAM 3.1 runs at 0.7–1.4 fps for us (M), against 16–32 fps claimed (E).
- **Two-tier (the winner):** it is the only plan that keeps today's quality (complete models, splats, dense map) and still reaches seconds, because today's pipeline becomes the refine tier. It also handles late results best: `basedOnSeq`, a refutation re-check and a Sim3 entry gate.
- **On-prem-ops:** it has the strongest policy details and the lowest cost, but the moving camera slips to about week 8.

**Grafted in:**
- From on-prem-ops: `site.yaml` plus one command, `--profile commercial`, on-box alerts that do not wait for the viewer, "no pass without coverage", spool-to-disk, retention, and boxes off until their gate passes.
- From latency-first: latest-frame-wins, an fps heartbeat, always-warm workers, `finding` events from day one, and go/no-go probes.

**One change to the winner:** its first milestone, a batch runner that reproduces all three demos, is not a one-week job and shows no stream. Our first milestone is a fixed-camera live loop built from the existing `ehs_spatial/video.py`.

## 2. Target architecture

```
site.yaml ──> `panoptes up`    per camera: id, rtsp, kind fixed|moving, zones, optional 2 scale points

RTSP ─> ingest: decode 640x480, latest-frame-wins, 72 h ring, cut / tamper check (CPU)
  ├─> LIVE (warm GPU; never waits on refine; observed data only)
  │     T0 people/movers: SAM 3 detect+track → foot point on floor → rules → finding + on-box alert
  │     T1 geometry: fixed = install map (plate depth + floor); moving = LingBot per frame → trust rules → voxel patch
  │     T2 objects: SAM 3 EHS vocabulary on keyframes → 3D accumulator → stable at ≥3 views → name if score ≥ τ
  └─> REFINE (job queue, preemptible, spend cap) = today's pipeline, automated, per window/area
        R1 DROID BA + DA3-BASE → TSDF → texture/fill
        R2 SAM 2.1 masks → object map → box ∥ SAM 3D → assess() gate → coded merge
        R3 inferred floor, splats (display only)
        every result = patch {provenance, gate, basedOnSeq, supersedes}
                 │
platform: POST blob / POST event → scene_events (append-only, gap-free seq) ← acceptor: 422 + refutation re-check
  ├─ GET events?after=N (long-poll) → viewer #/live (keyed-diff setScene, setStreamMesh)
  └─ materializer every 60 s → checkpoint revision → publication (served by the API)
```

A fixed camera runs the heavy 3D once at install. It runs it again only when a plate-difference test says the scene changed. On every frame, only T0 runs.

## 3. Stage by stage

| Stage | Tier | Model (licence) | GPU | Speed | Mode |
|---|---|---|---|---|---|
| Cut / tamper / tracking loss | ingest | ORB + RANSAC, OpenCV (Apache-2.0) | CPU | 10–20 ms (E); found 4/4 hand-chosen cuts on the frames probed (M) | per frame |
| People, movers | T0 | Day 1: SAM 3 per image through the existing `SAM3_BACKEND=modal`/`http` switch, plus ByteTrack. Target: SAM 3.1 forward-only, with compile and FA3 (SAM License) | L4 → H100 | 0.7–1.4 fps today (M); 16–32 fps vendor (E) | per frame |
| Lens, plate depth, floor | install | MoGe-3 (MIT), `fit_floor_from_keyframes` | L4 | seconds (E) | once per camera |
| Lift + rules | T0 | `lift_tracks`, `judge` / ZEN (ours) | CPU | <5 ms (E) | per frame |
| Pose + depth (moving camera) | T1 | LingBot-Map as a warm per-frame loop (code Apache-2.0, weights licence unconfirmed). Fallback: DPVO (MIT) + DA3METRIC-LARGE (Apache) | H100/L40S, 15–17 GB (M) | 4.6–7.1 fps (M); 20 fps vendor (E) | per frame |
| Observed map | T1 | incremental TSDF with movers masked (Open3D) | CPU | one patch per second (E) | incremental |
| Objects + names | T2 | SAM 3 vocabulary + the `build_video_object_map` accumulator; Qwen3-VL-8B (Apache) for the long tail; Gemini only as an optional cloud path | shared | ~30 ms per image vendor; 1–3 s per entity (E) | per keyframe / entity |
| Cameras + depth rebuild | R1 | DROID-SLAM (BSD) + DA3-BASE (Apache), replacing DA3-GIANT | A100 | DROID 30–55 s GPU per clip; DA3-GIANT 72–150 s (M) | per window |
| Class-agnostic masks | R2 | SAM 2.1 (Apache) | L4 | 2.5 s per view (M) | per window |
| Complete objects | R2 | box (CPU) ∥ SAM 3D Objects (SAM License) in parallel containers; RecGen off | A100-80GB | ~106 s wall per attempt (M) | per stable object |
| Splats | R3 | gsplat (Apache), display only, shown only at held-out PSNR ≥27 dB | H100 | 1,162–1,461 s per run (M) | per area |
| LingBot dense map + ICP | off | – | – | 16–24 min CPU per clip (M) | replaced by the T1 points; ICP only where submaps join |

## 4. End-to-end latency

| Path | Budget | Today |
|---|---|---|
| Fixed camera: frame → on-box alert | wait for the next sample 100–200 ms + decode 20–40 ms + SAM 30–60 ms + rules <5 ms + 3-frame debounce 300–600 ms (+50–150 ms on Modal) = **0.5–1 s p95 (E)** | 3.5–6+ h per clip (M) |
| Fixed camera: frame → viewer | the above + a 1 s segment + upload/append/long-poll/fetch 1–1.5 s = **2.5–3 s (E)** | 3.5–6+ h (M) |
| Moving camera: frame → map in viewer | LingBot 140–220 ms (M) + 1 s patch + 1–1.5 s delivery = **3–5 s (E)** | 3.5–6+ h (M) |
| Object named and boxed | ≤60 s after its 3rd agreeing view (E) | hours (M) |
| Complete model / splat for an area | ~2–5 min / ~20–25 min after the area is covered (E, from measured per-call times) | hours (M) |

A cold start takes 110–150 s (M), so live workers stay warm. If SAM 3.1 stays near 1 fps, the fixed-camera path still works but alerts slow to about 3–4 s (E). The MEVA PoC already ran at 1 fps sampling, with a 0.7 m/s speed band (M).

## 5. Automation: each manual step → rule → acceptance test

These are CPU replays unless noted. Each test must reproduce the decision made by hand this session.

| Manual today | Rule | Must reproduce |
|---|---|---|
| Choosing cut frames | Frame is a cut when its inliers are <0.1× the local median and <100; a run of low-inlier pairs is a fade. One `segments.json` with one span syntax | On every frame: {14, 226} for ME340, {420} for Sam's Club, {383} for Walmart, {} for Lightning. A spliced TUM hard cut and a 15-frame fade are both flagged |
| Choosing the mapping shot, placing the cut-away camera | The `register_cut_shot.align` gate (centre ≤10% of camera spread, ≤3°, depth ratio 0.9–1.1) using DA3-BASE; otherwise start a new submap | ME340 0–13 joins 226–898. ME340 14–226 lands within 5 cm / 2° of run 304, or is refused. Sam's Club 0–419 stays separate from 420–749. A Walmart frame is refused in the ME340 map |
| Lens per shot | MoGe-3 on ≥15 frames of one segment; score Q1 / median / Q3 by floor-inlier share; below 90% the intrinsics are `uncertain` and nothing is measured | Sam's Club a2 ≈55° (hand value 55.3°) with ≥95% floor inliers; Walmart 54.1°; ME340 leaves out the 51.3° frame |
| Deciding which part of the camera path to trust | Untrusted if the step is >5× its rolling median, the height is >0.3 m off its 2 s median, or the scale spread is >5% | Lightning trusts frames 96–450 ±15; all three mapping shots are fully trusted |
| Caption flags | Pixels that stay static while the camera moves, plus a glyph test on ≥30% of frames → mask stored in `clip.json` | ME340 is flagged; every other clip matches its run flags |
| Turning the people layer on or off | Always on. An entity with ≥30% person-mask overlap in at least half its views becomes dynamic; a person name needs the same overlap | The Sam's Club worker leaves the static map; ME340 #32 loses the name "man" |
| Guessed camera height (1.6 m) | Two tape-measured floor points at install; otherwise the scale stays `model_estimated` | A held-out dimension within 3%; MEVA median ≤0.5 m (0.45 m (M)) |
| Reviewing names | Name only above τ, set for precision ≥0.9 on ~300 labelled crops | ME340 #32 stays unnamed; fused P@10 ≥0.207 (M) |
| Running generators and merging with snippets | One job per stable object. The `merge.json` rule in code: gate-passed learned model > box > blank, with `PART_SHARE` de-duplication | Journal replay gives identical accept/reject for runs 231…302; totals 22/67/40, including duplicates 051 and 052 → 030 and 095 → 028 |
| Reviewing box models by eye | Per-pixel free-space test on each box silhouette: overhang, foreign pixels inside, occluded, `sourceFaceBacking`. **Boxes stay off until this passes** | All 10 eye-dropped boxes rejected and ≥30 of 32 kept boxes kept; thresholds fitted on 2 scenes must pass on the 3rd. Today's fields catch only 4 of the 10 (M) |
| Accepting ICP | Display gate (≥45% of points within 25 cm, floor within 5 cm) plus a step cap of 2° / 0.3 m | Sam's Club and Walmart pass; Walmart before ICP fails |
| Splat settings | Fixed settings; the runner chooses the held-out frames | 30.8 / 27.2 / 31.1 dB, ±0.5 (M) |
| Per-clip constants, paths, run ids | Derived from `clip.json` and the metric scale; run directories named by content hash | No stage keeps an ME340 default |
| Import and publish | The runner hashes all inputs, including a pose digest; live mode uses events plus checkpoints | From the MP4 alone: 123/78/22, 164/111/67, 166/102/40; a swapped DROID run is refused |

Those object counts came from DA3-GIANT. The research profile must match them exactly. The commercial profile (DA3-BASE) must pass every gate. If it finds fewer objects, we report the drop and do not tune it away.

## 6. Inference policy, online

1. **Every event is labelled** with:
   - `provenance`: observed, inferred-structure or inferred-model;
   - `producer@version`;
   - `gate{name, version, result, metrics}`;
   - `basedOnSeq` and `supersedes`.
2. **The acceptor returns 422** in two cases: inferred data without a passed gate, and any patch that sets `confirmed`. Confirmation stays a human step.
3. **Rules and measurements read observed geometry only.** A `model_estimated` scale or `uncertain` intrinsics give at best 需复核.
4. **No pass without coverage.** A rule passes only if its zone was observed for the whole window. A camera that is offline, moved or dark gives "no data".
5. **Findings are debounced.** They need 3 agreeing frames, and any lag over 5 s marks them `degraded`.
6. **Later views can refute earlier ones.**
   - When ≥2 views see past an inferred element, it is retracted. The retraction stays in the log.
   - Refine results are re-checked against every view that arrived after their `basedOnSeq`.
7. **Refine geometry enters the live frame only through a gate:** the camera-centre Sim3 plus the ICP display gate. If it fails, it is dropped.
8. **Output stops on trouble.** On a cut, tracking loss or an untrusted pose, 3D output goes blank until the camera relocalises. 2D tracks continue but are not lifted to 3D.
9. **Each inference kind tests itself** at install and nightly: noise ≤1%, raised-plane control ≥50%, hidden-block false rate ≤5%. A failure switches that kind off and raises an alert.
10. **A single fixed camera never gets completed models**, because it has no held-out views. It shows the observed 2.5D surface and named detections.

## 7. Live path to the viewer, and immutable snapshots

We adopt the platform design as written:
- **Event log:** `scene_events` holds patch, segment, finding and checkpoint events.
  - The sequence is gap-free per branch, using the branch row lock.
  - `producer_seq` makes appends idempotent.
  - The table is protected by `panoptes_immutable()`.
- **Blobs** are verified once, at upload. `_insert_revision` re-hashes only assets that are new since the parent revision. Today every commit re-reads every blob, 364 MB for one clip (M).
- **Delivery:**
  - an async long-poll, `?after=N&wait=25`;
  - `GET head` for the first load;
  - a bounded head: K keyframes and at most 8 observations per entity.
- **Moving people** are `segment` blobs kept outside the document. Today about 14 surfaces per second pile up inside it (derived from measured counts).
- **Viewer:**
  - a `#/live` route;
  - a keyed-diff `setScene`, the biggest single edit;
  - the existing `setStreamMesh`;
  - a clock set to wall time minus 2 s.
- **Snapshots:**
  - a checkpoint revision every 60 s;
  - "publish this moment" is `create_publication(checkpoint)` plus its event range, served by the API;
  - the static Modal site (59 s to prepare (M), plus a redeploy) stays for public sharing only.
- **Outages:** producers spool to disk and replay idempotently.
- **Before any real footage:** every GET needs a read capability, including today's open `/api/projects`.

## 8. Hardware and cost per camera-hour (E unless marked)

| Camera | Cloud (Modal list price) | On-prem |
|---|---|---|
| Fixed, detector at 1 Hz + tracker at 5 fps | one L4 ($0.90/h list) shared by 2–4 cameras: **$0.23–0.45** | **$0.11–0.21** |
| Fixed, full-rate SAM 3.1 | one H100 ($3.95/h list) shared by 2–3 cameras: $1.3–2.0 | 2–4 cameras per 48 GB GPU |
| Moving | one H100 + a $2/h refine cap: **$4–6** | ~$0.85 |
| Refine extras | SAM 3D $0.013–0.016 per call; splats $1.38–1.45 per area (M) | runs on idle GPU time |
| Today's offline pipeline | $8.5–13.5 per 15–17 s clip (M), about $1.8–3.3k per hour of video | – |

- **Reference on-prem box:** 2× 48 GB GPUs (L40S or RTX 6000 Ada), 32 cores, 128 GB RAM, 8 TB NVMe. About $40k, or about $1.70 per box-hour over 36 months including power.
- **Storage:** a 72 h video ring is about 130 GB per camera.
- **Warm cloud workers** are billed every hour, so past a few cameras on-prem is cheaper.

## 9. Roadmap

| Milestone | Build (reusing existing code) | Exit criteria | Demo |
|---|---|---|---|
| **M0, about 1 week** | A rolling-window loop around existing `video.py` functions: `sam_stage` (`SAM3_BACKEND=modal`), `bytetrack`, `fit_floor_from_keyframes` (run once), `lift_tracks`, `judge`, `render_overlay`. Input is a file looped by ffmpeg as RTSP, latest-frame-wins; findings go to JSONL. On CPU: the cut detector on every frame of the 4 clips, and the merge rule as code with a journal replay. Two bounded GPU probes, **only with your approval**: SAM 3.1 forward-only vs bidirectional on Lightning, and warm LingBot per frame with FlashInfer + compile | One command, zero hand flags. The verdict timeline equals offline `run_video_assessment` on the same frames. Frame-to-finding p95 is measured. Cut and merge tests pass. Go/no-go: SAM 3.1 ≥10 fps with ≥95% ID agreement; LingBot ≥10 fps, otherwise switch to DPVO | The MEVA fixed-camera segment streams in; zone, proximity and speed findings stream out |
| M1, 2–3 weeks | `scene_events`, blob upload, long-poll, `#/live`, keyed-diff `setScene`, read capability. Install calibration: MoGe-3 plate, floor, zones, scale points. Alert webhook. SAM 3.1 forward-only if M0 passed | Alert ≤1 s p95 and viewer ≤3 s p95. A 72 h soak on 4 replayed streams with <10% memory growth. Uncalibrated distances show 需复核. A moved camera goes blank | A 4-camera wall with alerts on the 3D install map |
| M2, 2–3 weeks | The refine tier as a one-shot runner: `segments.json`, automatic lens, automatic cut-away registration (DA3-BASE), parallel SAM 3D, coded merge, pose digests, import as patches, `--profile commercial` | All §5 tests pass. The research profile reproduces the three count triples. ≤60 min per clip. The commercial image refuses non-commercial weights | Drop in an MP4 and get a report link |
| M3, 3–4 weeks | Moving camera: warm LingBot service, trust rules, submaps on cuts or tracking loss, voxel patches, T2 objects, Qwen3-VL naming | Sam's Club a2 and Walmart replayed at 1× speed: first 3D ≤5 s, map lag ≤5 s, floor inliers ≥95%, a spliced cut opens a new submap. Drift over a 30-minute walk is measured | Walk a store and watch the map, the people and named objects grow live |
| M4, 2–3 weeks | Refine feeding live: acceptor re-check, Sim3 entry, SAM 3D queue, splats per area, box gate | Live p95 stays within ±10% under refine load. An injected wrong box is retracted. Journal replays are exact. The box test passes, so eye review turns off | Complete models and splats appear minutes after an area is covered |

## 10. Decisions only you can make

1. **Fixed or moving camera first?** I recommend fixed. The first live demo would then be MEVA or the ME340 cut-away, not a store walk.
2. **LingBot-Map weights licence.** The research note found Apache-2.0 on the GitHub code, but the Hugging Face card declares nothing.
   - Someone has to ask Robbyant in writing.
   - Without it, moving cameras use DPVO + DA3METRIC, with less detail.
3. **Swap DA3-GIANT for DA3-BASE everywhere**, even if that finds fewer objects.
4. **Input at install:** zones and two tape-measured floor points per camera. Without them every distance rule gives 需复核. Is that acceptable, or does it count as manual pushing for you?
5. **Cloud or on-prem** for the first site.
6. **Worker body data:** keeping it conflicts with the immutable-asset trigger. Choose purge or tombstones, and decide whether faces are blurred.
7. **SAM License:** can the gated weights ship inside an appliance?
8. **Licence problem found in this review:** `serving/mapanything_service.py` and the Modal path of `ehs_spatial/providers/map_anything.py` both load `facebook/map-anything`, which is the CC BY-NC variant. Switch to the Apache variant before any on-prem build.
9. **Approve the M0 GPU probes:** two bounded calls, a few dollars (E).
10. **Still unmeasured:** vendor fps, 24/7 drift without loop closure, on-prem naming precision, and every cost figure above.

Files checked for this plan:
- /Users/adam/.codex/worktrees/panoptes-phase2-video/ehs_spatial/video.py
- /Users/adam/.codex/worktrees/panoptes-phase2-video/ehs_spatial/providers/sam3.py
- /Users/adam/.codex/worktrees/panoptes-phase2-video/ehs_spatial/providers/map_anything.py
- /Users/adam/.codex/worktrees/panoptes-phase2-video/serving/