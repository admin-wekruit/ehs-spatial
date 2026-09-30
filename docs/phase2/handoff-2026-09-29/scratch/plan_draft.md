# Panoptes Live: how we get to one-shot streaming

**直接回答：** 把现在的离线流程全自动化也不够快：就算没有任何手工步骤，Walmart 最长的依赖链仍要约 4.6 h (E)。所以要拆成两层。**实时层**只输出观测到的东西：人、移动设备、EHS 告警、不断长大的观测地图，秒级。**精修层**就是现在整条流程，自动化后在后台按窗口/区域持续跑；它的结果只有通过自动闸门，才会替换进场景。固定摄像头的重 3D 只在安装时做一次。**"一次成型"**的意思是：每个摄像头一行配置、一条命令；每个手工决定都变成代码规则，并带一个必须重现本次手工结果的回放测试；闸门还没证明的推断默认关闭（留空）。

(M) = measured in our runs; (E) = estimate or vendor claim we have not reproduced.

## 1. Verdict

| Criterion (5 = best) | Latency-first | Two-tier | On-prem-ops |
|---|---|---|---|
| Time to first output | 5 | 4 | 4 |
| Steady-state latency | 5 | 4 | 4 |
| Correctness under inference policy | 3 | 5 | 5 |
| Completeness (LingBot-level map, complete objects, people) | 4 | 5 | 3 |
| Commercial / on-prem viability | 3 | 4 | 5 |
| Cost per camera-hour | 3 | 4 | 5 |
| Implementation risk / effort | 2 | 4 | 3 |
| **Total / 35** | **25** | **30** | **29** |

- **Latency-first:** tightest budgets, but its first milestone only measures, boxes are not held off until their gate passes, and it leans on vendor fps we have not reproduced (SAM 3.1: 0.7–1.4 fps (M) vs 16–32 claimed (E)).
- **Two-tier (winner):** the only plan that keeps today's quality (complete models, splats, dense map) *and* reaches seconds, because today's pipeline becomes the refine tier. Best handling of late results (`basedOnSeq`, refutation re-check, Sim3 entry gate).
- **On-prem-ops:** strongest policy details and lowest cost; moving camera slips to about week 8.

**Grafted in.** From on-prem-ops: `site.yaml` + one command, `--profile commercial`, on-box alerts that do not wait for the viewer, "no pass without coverage", spool-to-disk, retention, boxes off until gated. From latency-first: latest-frame-wins, fps heartbeat, always-warm workers, `finding` events from day one, go/no-go probes.

**One change to the winner:** its first milestone (a batch runner reproducing all three demos) is not a one-week job and shows no stream. Ours is a fixed-camera live loop built from the existing `ehs_spatial/video.py`.

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

A fixed camera runs the heavy 3D once at install, and again only when a plate-difference test says the scene changed. Per frame, only T0 runs.

## 3. Stage by stage

| Stage | Tier | Model (licence) | GPU | Speed | Mode |
|---|---|---|---|---|---|
| Cut / tamper / tracking loss | ingest | ORB + RANSAC, OpenCV (Apache-2.0) | CPU | 10–20 ms (E); 4/4 hand cuts on probed frames (M) | per frame |
| People, movers | T0 | Day 1: SAM 3 per image via existing `SAM3_BACKEND=modal`/`http` + ByteTrack. Target: SAM 3.1 forward-only, compile + FA3 (SAM License) | L4 → H100 | 0.7–1.4 fps today (M); 16–32 vendor (E) | per frame |
| Lens, plate depth, floor | install | MoGe-3 (MIT), `fit_floor_from_keyframes` | L4 | seconds (E) | once per camera |
| Lift + rules | T0 | `lift_tracks`, `judge` / ZEN (ours) | CPU | <5 ms (E) | per frame |
| Pose + depth (moving) | T1 | LingBot-Map warm per-frame loop (code Apache-2.0, weights unconfirmed); fallback DPVO (MIT) + DA3METRIC-LARGE (Apache) | H100/L40S, 15–17 GB (M) | 4.6–7.1 fps (M); 20 vendor (E) | per frame |
| Observed map | T1 | incremental TSDF, movers masked (Open3D) | CPU | patch per 1 s (E) | incremental |
| Objects + names | T2 | SAM 3 vocabulary + `build_video_object_map` accumulator; Qwen3-VL-8B (Apache) for the long tail; Gemini optional cloud only | shared | ~30 ms/image vendor; 1–3 s/entity (E) | per keyframe / entity |
| Cameras + depth rebuild | R1 | DROID-SLAM (BSD) + DA3-BASE (Apache) replacing DA3-GIANT | A100 | DROID 30–55 s GPU/clip; DA3-GIANT 72–150 s (M) | per window |
| Class-agnostic masks | R2 | SAM 2.1 (Apache) | L4 | 2.5 s/view (M) | per window |
| Complete objects | R2 | box (CPU) ∥ SAM 3D Objects (SAM License), parallel containers; RecGen off | A100-80GB | ~106 s wall/attempt (M) | per stable object |
| Splats | R3 | gsplat (Apache), display only, PSNR ≥27 dB | H100 | 1,162–1,461 s/run (M) | per area |
| LingBot dense map + ICP | off | – | – | 16–24 min CPU/clip (M) | replaced by T1 points; ICP only at submap joins |

## 4. End-to-end latency

| Path | Budget | Today |
|---|---|---|
| Fixed: frame → on-box alert | sample wait 100–200 ms + decode 20–40 + SAM 30–60 + rules <5 + 3-frame debounce 300–600 (+50–150 on Modal) = **0.5–1 s p95 (E)** | 3.5–6+ h per clip (M) |
| Fixed: frame → viewer | + 1 s segment + upload/append/long-poll/fetch 1–1.5 s = **2.5–3 s (E)** | same |
| Moving: frame → map in viewer | LingBot 140–220 ms (M) + 1 s patch + 1–1.5 s = **3–5 s (E)** | same |
| Object named and boxed | ≤60 s after its 3rd agreeing view (E) | ~2 h |
| Complete model / splat of an area | ~2–5 min / ~20–25 min after coverage (E, from per-call M) | 1–4 h |

Cold start is 110–150 s (M), so live workers stay warm. If SAM 3.1 stays near 1 fps, the fixed path still works (the MEVA PoC sampled at 1 fps, speed band 0.7 m/s (M)), but alerts slow to ~3–4 s (E).

## 5. Automation: manual step → rule → acceptance test

CPU replays unless noted; each must reproduce this session's hand decisions.

| Manual today | Rule | Must reproduce |
|---|---|---|
| Cut frames | Cut when inliers <0.1× local median and <100; low-inlier runs = fade. One `segments.json`, one span syntax | Every frame: {14, 226} ME340, {420} Sam's Club, {383} Walmart, {} Lightning; spliced TUM hard cut and 15-frame fade flagged |
| Mapping shot, cut-away placement | `register_cut_shot.align` gate (centre ≤10% spread, ≤3°, depth ratio 0.9–1.1) with DA3-BASE; else new submap | ME340 0–13 joins 226–898; 14–226 within 5 cm / 2° of run 304 or refused; Sam's Club 0–419 ≠ 420–749; Walmart frame refused in ME340 |
| Lens per shot | MoGe-3 on ≥15 frames of one segment; score Q1/median/Q3 by floor inliers; <90% → `uncertain`, no measurement | Sam's Club a2 ≈55° (hand 55.3°), ≥95% inliers; Walmart 54.1°; ME340 skips the 51.3° frame |
| Trusted path span | Untrusted if step >5× rolling median, height >0.3 m off 2 s median, or scale spread >5% | Lightning 96–450 ±15; mapping shots fully trusted |
| Caption flags | Static-while-moving pixels + glyph test ≥30% of frames → mask in `clip.json` | ME340 flagged; others match their run flags |
| People layer on/off | Always on; ≥30% person-mask overlap in ≥half its views → dynamic; person names need the same | Sam's Club worker leaves the static map; ME340 #32 loses "man" |
| Camera height 1.6 m guess | Two tape-measured floor points at install; else `model_estimated` | Held-out dimension within 3%; MEVA ≤0.5 m (0.45 m (M)) |
| Name review | Name only above τ (precision ≥0.9 on ~300 labelled crops) | ME340 #32 unnamed; fused P@10 ≥0.207 (M) |
| Generator runs, merge snippets | One job per stable object; `merge.json` rule in code: gated learned > box > blank, `PART_SHARE` dedupe | Journal replay identical for runs 231…302; totals 22/67/40 incl. 051, 052→030, 095→028 |
| Box eye review | Per-pixel free-space test (overhang, foreign-inside, occluded, `sourceFaceBacking`). **Boxes off until it passes** | All 10 eye-dropped rejected, ≥30/32 kept, fit on 2 scenes passes the 3rd (today's fields: 4/10 (M)) |
| ICP accept | Display gate (≥45% within 25 cm, floor ≤5 cm) + step cap 2° / 0.3 m | Sam's Club, Walmart pass; Walmart pre-ICP fails |
| Splat settings | Fixed settings, runner-chosen held-out frames | 30.8 / 27.2 / 31.1 dB ±0.5 (M) |
| Per-clip constants, paths, run ids | Derived from `clip.json` + metric scale; content-addressed run dirs | No stage keeps an ME340 default |
| Import / publish | Runner hashes all inputs incl. pose digest; live = events + checkpoints | From the MP4 alone: 123/78/22, 164/111/67, 166/102/40; swapped DROID run refused |

Those counts came from DA3-GIANT. The research profile must match exactly; the commercial profile (DA3-BASE) must pass every gate, and any drop in objects is reported, not tuned away.

## 6. Inference policy, online

1. **Labels on every event:** `provenance` (observed / inferred-structure / inferred-model), `producer@version`, `gate{name, version, result, metrics}`, `basedOnSeq`, `supersedes`.
2. **Acceptor returns 422** for inferred data without a passed gate and for any patch setting `confirmed` (confirmation stays human).
3. **Rules and measurements read observed geometry only.** `model_estimated` scale or `uncertain` intrinsics → at best 需复核.
4. **No pass without coverage.** Pass only if the zone was observed for the whole window; offline, moved or dark camera → "no data".
5. **Debounce:** 3 agreeing frames; lag >5 s marks findings `degraded`.
6. **Later views refute.** ≥2 "seen past" views retract an inferred element (the retraction stays in the log). Refine results are re-checked against views after their `basedOnSeq`.
7. **Refine geometry enters the live frame** only through the camera-centre Sim3 + ICP display gate; failure → dropped.
8. **Stop on trouble.** Cut, tracking loss or untrusted pose → 3D blank until relocalised; 2D tracks continue unlifted.
9. **Self-tests** at install and nightly (noise ≤1%, raised-plane control ≥50%, hidden-block false rate ≤5%); failure switches that kind off and alerts.
10. **A single fixed camera never gets completed models** (no held-out views); it shows observed 2.5D and named detections.

## 7. Live path to the viewer, and immutable snapshots

Adopt the platform design as written:
- **Event log:** `scene_events` (patch, segment, finding, checkpoint), gap-free per-branch sequence via the branch row lock, idempotent `producer_seq`, under `panoptes_immutable()`.
- **Blobs** verified once at upload; `_insert_revision` re-hashes only assets new since the parent (today every commit re-reads all blobs, 364 MB for one clip (M)).
- **Delivery:** async long-poll `?after=N&wait=25`, `GET head` for first load, bounded head (K keyframes, ≤8 observations per entity).
- **Moving people** as `segment` blobs outside the document (today ~14 surfaces/s accumulate in it (M-derived)).
- **Viewer:** `#/live`, keyed-diff `setScene` (biggest edit), existing `setStreamMesh`, clock = wall time − 2 s.
- **Snapshots:** checkpoint revision every 60 s; "publish this moment" = `create_publication(checkpoint)` + its event range, served by the API. The static Modal site (59 s prepare (M) + redeploy) stays for public sharing only.
- **Outages:** producers spool to disk and replay idempotently.
- **Before real footage:** read capability on every GET, including today's open `/api/projects`.

## 8. Hardware and cost per camera-hour (E unless marked)

| Camera | Cloud (Modal list) | On-prem |
|---|---|---|
| Fixed, detector 1 Hz + tracker 5 fps | L4 ($0.90/h list) shared by 2–4: **$0.23–0.45** | **$0.11–0.21** |
| Fixed, full-rate SAM 3.1 | H100 ($3.95/h list) shared by 2–3: $1.3–2.0 | 2–4 per 48 GB GPU |
| Moving | H100 + $2/h refine cap: **$4–6** | ~$0.85 |
| Refine extras | SAM 3D $0.013–0.016/call; splats $1.38–1.45/area (M) | idle GPU |
| Today, offline | $8.5–13.5 per 15–17 s clip (M) ≈ $1.8–3.3k per video-hour | – |

Reference box: 2× 48 GB GPUs (L40S / RTX 6000 Ada), 32 cores, 128 GB RAM, 8 TB NVMe, ~$40k ≈ $1.70/box-hour over 36 months with power. Video ring 72 h ≈ 130 GB/camera. Warm cloud workers bill every hour; past a few cameras on-prem is cheaper.

## 9. Roadmap

| Milestone | Build (reuse) | Exit criteria | Demo |
|---|---|---|---|
| **M0, ~1 week** | Rolling-window loop around existing `video.py`: `sam_stage` (`SAM3_BACKEND=modal`), `bytetrack`, `fit_floor_from_keyframes` once, `lift_tracks`, `judge`, `render_overlay`; ffmpeg-looped file as RTSP, latest-frame-wins, findings JSONL. CPU: cut detector on every frame of the 4 clips; merge rule as code + journal replay. Two bounded GPU probes **with your approval**: SAM 3.1 forward-only vs bidirectional on Lightning; warm LingBot per frame with FlashInfer + compile | One command, zero hand flags; verdict timeline equals offline `run_video_assessment` on the same frames; frame→finding p95 measured; cut and merge tests pass; go/no-go: SAM 3.1 ≥10 fps with ≥95% ID agreement, LingBot ≥10 fps else DPVO | MEVA fixed-camera segment streams in; zone, proximity and speed findings stream out |
| M1, 2–3 wk | `scene_events`, blob upload, long-poll, `#/live`, keyed-diff `setScene`, read capability; install calibration (MoGe-3 plate, floor, zones, scale points); alert webhook; SAM 3.1 forward-only if M0 passed | Alert ≤1 s p95, viewer ≤3 s p95; 72 h soak on 4 replayed streams, memory growth <10%; uncalibrated distances show 需复核; moved camera → blank | 4-camera wall, alerts on the 3D install map |
| M2, 2–3 wk | Refine tier as one-shot runner: `segments.json`, auto lens, auto cut-away registration (DA3-BASE), parallel SAM 3D, coded merge, pose digests, import as patches, `--profile commercial` | All §5 tests pass; research profile reproduces the three count triples; ≤60 min per clip; commercial image refuses NC weights | Drop an MP4, get a report link |
| M3, 3–4 wk | Moving camera: warm LingBot service, trust rules, submaps on cut/loss, voxel patches, T2 objects, Qwen3-VL naming | Sam's Club a2 and Walmart replayed at 1×: first 3D ≤5 s, map lag ≤5 s, floor inliers ≥95%, spliced cut opens a submap; 30-min drift measured | Walk a store; map, people and named objects grow live |
| M4, 2–3 wk | Refine feeding live: acceptor re-check, Sim3 entry, SAM 3D queue, splats per area, box gate | Live p95 within ±10% under refine load; injected wrong box retracted; journal replays exact; box test passes → eye review off | Complete models and splats appear minutes after an area is covered |

## 10. Decisions only you can make

1. **Fixed or moving first?** I recommend fixed. The first live demo is then MEVA or the ME340 cut-away, not a store walk.
2. **LingBot-Map weights licence.** GitHub code shows Apache-2.0 (per the research note); the HF card declares nothing. Ask Robbyant in writing; otherwise moving cameras use DPVO + DA3METRIC, with less detail.
3. **DA3-GIANT → DA3-BASE everywhere**, even if fewer objects result.
4. **Install-time input:** zones and two tape-measured floor points per camera. Without them every distance rule is 需复核. Acceptable, or is that "manual pushing"?
5. **Cloud or on-prem** for the first site.
6. **Worker body data:** retention conflicts with the immutable-asset trigger (purge vs tombstone; face blurring).
7. **SAM License:** may gated weights ship inside an appliance?
8. **Licence issue found in this review:** `serving/mapanything_service.py` and the Modal path of `ehs_spatial/providers/map_anything.py` load `facebook/map-anything`, the CC BY-NC variant. Switch to the Apache variant before any on-prem build.
9. **Approve the M0 GPU probes** (two bounded calls, a few dollars (E)).
10. **Still unmeasured:** vendor fps, 24/7 drift without loop closure, on-prem naming precision, every cost figure.
