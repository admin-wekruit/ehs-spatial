# Panoptes streaming: two-tier plan (live in seconds, refine in minutes)

## 1. Principles
- **One-shot means one runner plus gates, with no hand flags.** Every hand decision becomes a coded rule, and each rule has a replay test against this session's hand result. The refine tier *is* today's offline pipeline, automated and run per window or area instead of per clip, so M0 is also the first half of streaming.
- **Live never waits on refine.** Live writes observed geometry, tracks and findings. Refine writes versioned upgrades, which are accepted only if their gate passed and no later view refutes them.
- **Fixed camera first.** It needs no live SLAM. At install, the runner processes a 1-minute walk video and places the camera in that map with the `register_cut_shot` gate. With no walk, the camera gets 2.5D from background-plate depth. `ehs_spatial/video.py` already does SAM, ByteTrack, a floor lift and banded zone/distance/speed rules.

## 2. Components
```
RTSP -> edge: decode, sample 5-10 fps, cut/tracking-loss + overlay detect, ring buffer (drop oldest)
  |
  +-> LIVE (warm GPU)                          REFINE (queue, autoscaled, $ cap)
  |    T0 SAM 3.1 fwd-only people/movers        R1 window rebuild: DROID BA + DA3-BASE, TSDF, texture
  |       -> floor foot point -> rules          R2 SAM 2.1 AMG, object map, box + SAM 3D, coded merge
  |    T1 LingBot pose+depth (moving cam)       R3 inferred floor/walls, splats per area
  |    T2 SAM 3 vocab keyframes -> instances    R4 naming: SAM 3 vocab, Qwen3-VL >= tau
  |       (>=3 views) -> box + gate                   | patch {gate, basedOnSeq, supersedes}
  v        | segment/patch/finding (observed)         v
 platform: POST blob, POST event -> scene_events (append-only, gap-free seq per branch)
   acceptor: policy checks (422) + refutation re-check | materializer 60 s -> checkpoint -> publication
   GET events?after=N long-poll -> viewer: keyed-diff setScene, setStreamMesh
```

## 3. Stages and latency

| Stage | Tier | Model (licence) | Throughput | Budget |
|---|---|---|---|---|
| Capture, sample, uplink; cut detector (ORB + RANSAC F, found 4/4 hand cuts) | edge | OpenCV (Apache) | 10–20 ms CPU | 0.35 s |
| People/mover tracks, bounded memory, periodic re-detect | T0 | SAM 3.1 forward-only (SAM Licence); fallback SAM 2.1 / EdgeTAM (Apache) | upstream 16–32 fps/H100; ours 0.7–1.4 fps offline, compile/FA3 off | 0.05 s |
| Floor lift, banded rule, 3-frame hysteresis | T0 | ours | CPU | 0.4 s |
| Append, long-poll, render | platform | – | – | 0.7 s |
| **Fixed camera: finding on screen** | | | | **≈1.5 s p50, 2.5 s p95** |
| Person surface (1 s segment) | T0 | mask × depth | – | ≈3 s |
| Pose + depth | T1 | LingBot-Map (repo Apache-2.0, weights licence unconfirmed); fallback DPVO (MIT) + DA3METRIC-LARGE (Apache) | ours 5–7 fps, upstream 20 | 0.1–0.2 s |
| Observed map chunk, people masked | T1 | incremental TSDF | CPU | 0.7 s |
| **Moving camera: map growth on screen** | | | | **≈2.5–3.5 s** |
| Object instance, box, gate | T2 | SAM 3 concept prompts at 1 Hz; `build_video_object_map` merge | ~30 ms/image upstream | ≤10 s after 3rd agreeing view |
| Posed depth | R1 | DA3-BASE (Apache), replacing DA3-GIANT (NC) | 72–150 s/clip today | minutes |
| Complete models | R2 | SAM 3D Objects (SAM Licence), parallel; RecGen off | ~106 s wall/attempt | 2–5 min/object |
| Naming | R4 | Qwen3-VL (Apache); Gemini optional cloud | seconds | ≤1 min |
| Splats, display only | R3 | gsplat (Apache), per area | ~20 min/H100 | 20–30 min |

## 4. Manual step → rule → acceptance test

| Manual | Rule | Must reproduce |
|---|---|---|
| Cut frames | edge detector → `segments.json`; live tracking loss → new submap | {14,226}, {420}, {383}, {} on every frame; spliced TUM cut and 15-frame fade flagged |
| Mapping shot, cut-away placement | relocalise via `register_cut_shot` gate (≤10%, ≤3°, depth ratio 0.9–1.1) on DA3-BASE, else new submap | ME340 14–226 within 5 cm/2° of run 304 or refused; Walmart frame refused in ME340; Sam's Club 0–419 not merged with 420–749 |
| Lens per shot | MoGe-3 on 15 frames at install (fixed) or LingBot K per segment; choose Q1/median/Q3 by floor inliers; <90% → `uncertain` | Sam's Club a2 ≈55°, inliers ≥95%; Walmart 54.1° |
| Camera height guess | install: one known dimension; else `model_estimated`, distance rules ≤ 需复核 | a held-out tape-measured dimension within 3% |
| Path trust | step >5× rolling median, height jump >0.3 m, scale spread >5% → untrusted | Lightning 96–450 ±15; mapping shots fully trusted |
| Captions | static-pixel + glyph mask in `clip.json` | matches every run's flag |
| People layer | always on; static entity ≥30% person-mask overlap in ≥half its views → dynamic | Sam's Club worker leaves map; ME340 #32 loses "man" |
| Generators, merges | job per stable entity; learned > box > blank, `PART_SHARE` dedupe | journal replay reproduces runs 231…302 and 22/67/40 |
| Box eye review | free-space silhouette test (overhang, foreign, occluded, sourceFaceBacking) | 10/10 drops rejected, ≥30/32 kept, holds on the unseen 3rd scene. **Until then boxes stay unpublished** |
| ICP accept | display gate + step cap 2°/0.3 m | Sam's Club, Walmart pass; Walmart pre-ICP fails |
| Import/publish | runner hashes inputs incl. pose digest | MP4 alone gives 123/78/22, 164/111/67, 166/102/40; swapped DROID run refused |

## 5. Inference policy online
1. Every representation carries `provenance` (observed / inferred-structure / inferred-model), `producer@version`, `gate{name,version,result,metrics}`, `basedOnSeq`, `supersedes`.
2. The acceptor rejects (422) inferred data without a passed gate, and any patch setting `confirmed`. Measurements and EHS rules read observed data only. A rule touching inferred geometry or `model_estimated` scale is at most 需复核.
3. Late refine results: the acceptor re-runs free-space refutation (CPU, seconds) on views after `basedOnSeq`; ≥2 see-past views drop the result. Live keeps counting see-past views per inferred cell, and at 2 it emits a retraction.
4. The tiers disagree on poses. Refine geometry enters the live frame through the camera-centre Sim3 (the existing `lingbot_dense_map` fit) and must pass the ICP display gate. If it fails, it is dropped and the live geometry stays.
5. Each inference kind self-tests per site (noise ≤1%, raised-plane ≥50%, hidden-block false ≤5%). A failure turns that kind off.
6. A finding needs 3 agreeing frames, and a flip needs the band crossed.

## 6. Live-update path
Adopt the platform design: `scene_events` (patch, segment, checkpoint, plus `finding`), blobs verified once, long-poll, 2-minute segment window, 60 s checkpoints, publication = checkpoint. Refine appends patches rather than racing the head CAS. `_insert_revision` re-hashes only new assets. **Blocker:** add read capability before any worker footage lands.

## 7. Cost per camera-hour (Modal list price, unmeasured)

| | Live | Refine | Total |
|---|---|---|---|
| Fixed | H100 shared by ~4 cameras at 5 fps, or L4 (SAM 2.1 tracks + SAM 3 at 1 Hz) for 2–3 | install ≈$3–6 once; hourly change check ≈$0.02 | **≈$0.4–1.0** |
| Moving | 1 H100 (LingBot 15–17 GB + SAM 3.1), $3.95/h | SAM 3D ≈$0.014/call, splat ≈$1.4/area, cap $2/h | **≈$4–6** |

Containers must stay warm (cold start 110–150 s). On-prem: one 48 GB GPU per moving camera, one per ~4 fixed cameras, one shared refine GPU per site.

## 8. Failure handling
- **Backpressure:** drop the oldest frames, never queue. If lag exceeds 5 s, findings are marked `degraded`.
- **Restart:** `producer_seq` is idempotent, so resume from the last checkpoint. Track ids restart and the break is recorded, never bridged.
- **Cut, tracking loss, or a bumped camera (plate diff):** start a new submap. The static map is invalidated and distance rules drop to 需复核 until re-install passes.
- **Refine failure or overspend:** it is journaled and the blank stays.
- **Network loss:** the edge buffers and replays with source timestamps.
- **Licences:** `--profile commercial` refuses NC weights.

## 9. Roadmap

| M | Build | Exit criteria | Demo |
|---|---|---|---|
| M0, 1–2 wk | batch one-shot runner: segments, auto lens, coded merge, pose digests, parallel SAM 3D, auto import/publish | MP4 alone reproduces the three triples and the export byte check; zero hand flags; ≤60 min/clip (now 3.5–6 h) | MP4 in, link out |
| M1, 2 wk | live plane, incremental viewer, read capability | platform §3.7 tests; a replayed clip at 1× shows people ≤3 s behind | clip replayed live |
| M2, 2–3 wk | fixed camera: install calibration, SAM 3.1 forward-only, `video.py` findings | forward-only vs bidirectional identity agreement ≥95% on Lightning; 1 h RTSP soak with p95 ≤2.5 s, flat memory, measured $/h | ME340 static cut-away (14–226) looped as RTSP → live zone alerts on the 3D map |
| M3, 3–4 wk | moving camera: warm per-frame LingBot, Sim3 window stitch, observed map, T2 | ≥10 fps at 518×378; Walmart replay passes display gate vs offline map; 30-min drift measured (decides VGGT-SLAM 2.0 loop closure) | map grows in ~3 s |
| M4, 2–3 wk | refine on live: acceptor, refutation, Sim3 entry | live p95 within ±10% under refine load; injected wrong box retracted | models and splats swap in |
| M5 | commercial on-prem | box gate passes (eye review off); Qwen3-VL τ at P≥0.9 on 300 crops; LingBot weights licence in writing | on-prem box |

## 10. Not possible yet
- Live complete models or splats: these take minutes.
- A real-time moving camera: ours is 5–7 fps, with no loop closure and unknown drift.
- Live class-agnostic discovery: AMG takes 2.5 s/view.
- Metric distances without calibration.
- Retiring box eye review: the current fields catch only 4/10.
- On-prem naming quality: unknown.
- All cost and throughput numbers here are estimates. The M2 and M3 measurements are GPU calls that need approval.
