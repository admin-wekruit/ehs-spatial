# Panoptes Live: latency-first one-shot plan

**Position.** The product is the live layer: people, movers and EHS findings in 3D within ~2 s, and a map that grows every second. Whatever cannot keep up moves to a budgeted background tier or is dropped. **One-shot** = one command per camera (`panoptes live add --rtsp URL --mode fixed|moving`): no run ids, no hand-passed paths, no eye review.

## 1. Components

```
camera (RTSP) -> gateway: decode 640x480, 30 s ring buffer, latest-frame-wins,
                 cut/tracking-loss detector (15 ms CPU)
   | 10 fps fixed / 5-7 fps moving
   v
GPU worker, always warm (one per camera group)
 T0 tracker : SAM 3.1 forward-only (person, forklift, pallet jack, cart)
              -> foot points on floor -> EHS rules -> `finding` event
 T1 geometry: fixed  = background-plate depth (once, again on change)
              moving = LingBot step (pose + K + depth)
              -> observed voxel map, movers masked -> `patch` every 1 s
 T2 objects : per keyframe, async. SAM 3 vocabulary + slow class-agnostic masks
              -> 3D accumulator -> stable at >=3 views -> box + automatic gate
              -> Qwen3-VL name if score >= tau
 T3 queue   : minutes, budget-capped. SAM 3D Objects, inferred floor, gsplat
   | every tier appends to
   v
Postgres scene_events (append-only, gap-free seq per branch) + sha256 blobs
   |- long-poll ?after=seq -> viewer #/live (keyed-diff setScene, setStreamMesh)
   '- materializer every 60 s -> checkpoint revision -> publication (async)
```

## 2. Model per stage

| Stage | Pick | Licence | Throughput | Fallback |
|---|---|---|---|---|
| Cut / tracking loss | ORB + RANSAC-F inliers | Apache-2.0 (OpenCV) | 10-20 ms CPU; found all 4 hand cuts | - |
| Moving pose + depth | LingBot-Map, warm per-frame loop | code Apache-2.0; **weights unconfirmed** | measured 5-7 fps; upstream claims 20 | DPVO (MIT) + DA3METRIC-LARGE (Apache) |
| Fixed static depth | MoGe-3 on median background plate | MIT | ~1 s per change | DA3METRIC-LARGE |
| People / movers | SAM 3.1 Multiplex, forward-only, compile + FA3 | SAM License | claimed 16-32 fps H100; measured ~1 fps (offline, bidirectional) | SAM 2.1 / EdgeTAM (Apache) + SAM 3 detector every N frames |
| Objects, vocabulary | SAM 3 concept prompts, EHS list | SAM License | ~30 ms/image claimed | - |
| Objects, class-agnostic | SAM 2.1 masks every ~10 s | Apache-2.0 | 2.5 s/view L4, measured | skip |
| Naming, long tail | Qwen3-VL-8B embed + rerank | Apache-2.0 | seconds, async | "unnamed object" |
| Shape | gravity box + gate; SAM 3D Objects in T3 | ours / SAM License | ms / ~106 s per attempt | blank |
| Splats | gsplat in T3, per area | Apache-2.0 | ~20 min H100 | mesh |

Removed: DA3-GIANT, RecGen, Gemini, LingBot dense-map post-processing, DROID global BA, whole-document import.

## 3. Latency budget (p95 targets, warm workers)

| Step | Fixed | Moving |
|---|---|---|
| Decode + uplink (on-prem ~5 ms; Modal +50-150 ms) | 0.2 s | 0.2 s |
| Geometry step | 0 (plate cached) | 0.15-0.2 s |
| SAM 3.1 step | 0.05-0.1 s | 0.05-0.1 s |
| Lift + rules | 0.01 s | 0.01 s |
| Debounce, 3 agreeing frames | 0.3 s | 0.6 s |
| Append + long-poll wake + fetch | 0.5-1.0 s | 0.5-1.0 s |
| **EHS alert on screen** | **1.1-1.6 s** | **1.5-2.1 s** |
| Moving-person mesh (1 s segment + 0.5 s buffer) | ~3 s | ~3.5 s |

**From camera connect:** about 3-4 s to first 3D in both modes (fixed: movers masked, MoGe-3, floor fit; moving: 8 LingBot warm-up frames = 1.6 s at 5 fps). People from the first frames. **Objects within 5 s only as vocabulary detections lifted to observed points;** stable objects need 3 agreeing views, so 5-15 s on a walk. A cold Modal container takes 110-150 s (measured): the live tier must stay warm.

## 4. Manual step -> automatic rule

| Manual today | Automatic | Acceptance test |
|---|---|---|
| Cut frames | gateway detector opens a new segment | every frame of 4 clips gives exactly {14,226}, {420}, {383}, {}; spliced TUM hard cut and 15-frame fade caught at the right frame |
| Mapping shot, cut-away placement | relocalise via `register_cut_shot.align` gate on DA3-BASE, else new submap | ME340 14-226 within 5 cm / 2 deg of run 304 or refused; Walmart frame refused in ME340 map |
| Lens per shot | moving: median LingBot K per segment; fixed: MoGe-3 at install | Sam's Club a2 ~55 deg, floor inliers >=95% |
| Trusting the path | step >5x rolling median, height >0.3 m off 2 s median, scale spread >5% -> no geometry | Lightning 96-450 +/-15; mapping shots fully trusted |
| Dynamic layer on/off | always on; >=30% person overlap in >=half the views -> dynamic, person name refuted | Sam's Club worker leaves static map; ME340 #32 loses "man" |
| Name review | tau for precision >=0.9 on ~300 labelled crops | ME340 #32 unnamed |
| Generator runs, merges | per-object queue; `merge.json` rule in code | journal replay reproduces runs 231-302 and 22/67/40 |
| Box eye review | per-pixel overhang / foreign-inside / occluded / sourceFaceBacking | all 10 eye-dropped rejected, >=30/32 kept, fitted on 2 scenes, passes the 3rd |
| ICP, voxel, scale knobs | from camera config; ICP only at submap joins, capped 2 deg / 0.3 m | Walmart pre-ICP fails, post-ICP passes |
| Import / publish | event log, 60 s checkpoints | export byte check passes on a live checkpoint |

## 5. Inference policy, online

- Every event carries `provenance`; an inferred one without a passed `gate {name, version, result}` gets 422.
- Rules and measurements read observed geometry only. A rule touching inferred geometry or uncalibrated scale reports 需复核, never pass or fail. Scale stays `model_estimated` until a one-time install knob (known dimension or camera height).
- Confirmation before emission: people 3 frames, objects 3 agreeing views, names >= tau.
- Later frames refute earlier ones: free-space counts accumulate per cell; >=2 views seeing past an inferred element emit a `remove` patch, kept in the audit log.
- Each test self-checks on a camera's first minute (<=1% flags on seen floor, catches the +15 cm control), else that inference kind stays off for that camera.
- Cut, tracking loss, untrusted pose or stream gap: 3D output stops (blank) until relocalised; 2D tracks continue unlifted.

## 6. Live data path

Adopt the platform dossier's live plane: `scene_events` (patch, segment, checkpoint) plus a `finding` kind from day one, because the alert is the product. Blobs verified once, async long-poll, bounded head, `_insert_revision` hashing only new blobs, keyed-diff `setScene`, `setStreamMesh` for movers. **Before any real footage:** read capability on live GETs, and close the open `/api/projects` listing.

## 7. Sizing (estimates from Modal list prices, not measured)

| Camera | GPU plan | Modal $/camera-hour | On-prem |
|---|---|---|---|
| Fixed, 10 fps | one H100 runs SAM 3.1 for 2-3 cameras | $1.3-2.0 | one 48 GB GPU per ~3 cameras |
| Fixed, low cost | one L4: SAM 2.1 tracking + SAM 3 detector at 1 Hz, 2-3 cameras | $0.30-0.45 | same, or EdgeTAM at the edge |
| Moving | one H100: LingBot (15-17 GB) + SAM 3.1 | $4-5 | one GPU per camera |

T3 extra: SAM 3D ~$0.016/attempt, splat ~$1.4/run, capped per camera-day. Always-warm 24/7 makes on-prem cheaper beyond a few cameras.

## 8. Failure handling

- **GPU behind:** latest-frame-wins; fps drops, latency does not. Real fps goes out in a heartbeat event.
- **Worker crash:** restart from last checkpoint; gaps marked, never interpolated.
- **Memory:** bounded SAM memory bank (a community SAM 3 streaming port OOMs after ~5 min); re-detect every N frames.
- **Drift:** LingBot has no loop closure; the trust monitor forces a new submap, joined only through the geometric gate.
- **Spend:** `--max-usd` per camera-day. **Viewer drop:** resumes from `after=seq`.

## 9. Roadmap

| Milestone | Scope | Exit criteria | Demo |
|---|---|---|---|
| M0, 1 wk | Measure (each GPU call needs your approval): SAM 3.1 forward-only vs bidirectional on Lightning; warm LingBot per frame with FlashInfer + compile. CPU: cut detector on every frame | SAM 3.1 >=10 fps at 640x480, >=95% track-ID agreement; LingBot >=10 fps, else DPVO fallback; cut test passes | go/no-go table |
| M1, 2-3 wk | Fixed camera live: gateway, plate depth, SAM 3.1, floor lift, zone/proximity rules (reuse `ehs_spatial/video.py` `lift_tracks`/`judge`, minus fal/Replicate), event log, `#/live`, read capability | one command; shell + people <=5 s p95; alert <=2 s p95; 1 h soak, flat memory; uncalibrated distances show 需复核 | ME340 cut-away (frames 14-226, our only static shot) replayed as RTSP, plus a new 1 h tripod recording |
| M2, 3-4 wk | Moving camera: warm LingBot loop, masked voxel map, trust monitor, cuts -> submaps, automatic cut-away registration | Sam's Club a2 and Walmart replayed in real time: first 3D <=5 s, patch every <=1 s, floor inliers >=95%, spliced cut -> new submap, 30 min sustained | map builds live during a walk, with people |
| M3, 3-4 wk | Live objects: vocabulary detection, accumulator, stability, box gate, Qwen3-VL naming | within 10% of offline detected/qualifying (123/78, 164/111, 166/102); box-gate test; name precision >=0.9 | labelled objects appear during the walk |
| M4, 2-3 wk | T3: SAM 3D queue, inferred floor, splats, checkpoint publication | journal replay matches; splat PSNR >=27 dB else mesh; 8 h soak | "freeze this moment" -> shareable report |
| M5 | On-prem box, no cloud calls | written LingBot weights licence; image scan finds no non-commercial weights | site install |

## 10. Not possible yet

- Splats and learned shapes in seconds: they arrive minutes later, or never.
- Fixed camera: front surfaces only; the box and completion gates need held-out viewpoints, so completion is off.
- LingBot 20 fps and SAM 3.1 32 fps are vendor claims; ours are 5-7 fps and ~1 fps. M0 decides.
- 24/7 moving-camera drift is unmeasured; LingBot has no loop closure.
- Metric EHS distances need a one-time calibration; without it they stay 需复核.
- Today's object counts rest on DA3-GIANT depth; parity on Apache depth is unproven.
