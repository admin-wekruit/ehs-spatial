# E6b: keyframe masks + projected outlines (fast path follow-up)

`outline_projection_probe.py` tests one idea on the ME340 walk shot: run full-quality segmentation only on keyframes, and
give every in-between frame outlines projected from 3D. It measures speed and agreement with today's per-frame masks.
All GPU work runs on Modal A100-80GB (ephemeral `modal run`, retries 0).

It reuses round 1's work:
- today's masks as label maps, and DROID poses + DA3 posed depth: `runs/m3-exp-e2-e6-segment-inputs`;
- the segment probe's helpers and SAM 2 image: `segment_fast_probe.py`, copied from `m3/exp-e2-e6-segment`;
- E1's DA3 image and pins, and E7's GPU lift rule: `m3_exp_geometry.py`, copied from `m3/exp-e1-e7-geometry`.

## How it works

- **Keyframes:** every 2, 4, 5, 6, 8 or 10 frames of today's 10 fps mask grid (5 to 1 fps). They keep today's SAM 2 AMG
  masks. These outlines are labelled `segmented`.
- **Lift:** keyframe masks are lifted to 3D objects with round 1's GPU rule: 5 cm voxels, then merge by voxel overlap
  across frames. Masks that are mostly on people are dropped.
- **Projection:** in-between frames get outlines labelled `projected`. The variants:

| Variant | What it does |
|---|---|
| `prev_r1` | Round 1's forward reprojection from the previous keyframe (reproduces round 1: 0.7316 / 0.7688 at 2 fps). |
| `prev` | The same, with people left out. |
| `pair` | The two keyframes around the frame. The nearer keyframe's points win, and the other keyframe fills only what the nearer one cannot see. |
| `fused` | All keyframes' points in one z-buffer, at half resolution. This is the "fused points per 3D object" version. |
| `pair_vis`, `fused_vis` | The same points, used for visibility only. Among the points within 5% of the nearest surface, the keyframe closest in time gives the label. |
| `*_snap` | Depth-edge snapping. A projected label stays only where its depth agrees with the frame's own depth (tau 0.04, 0.08 or 0.16). Other pixels refill from neighbours on the same depth surface. Snapping uses DA3 depth before the edge rule. |
| `dec_box`, `dec_pick`, `dec_pt` | Prompts for the SAM 2 mask decoder alone, from the `pair` projection: the box of each projected keyframe mask, box + interior point, or the point alone. `dec_pick` and `dec_pt` keep the candidate mask closest to the projection. The decoder runs on image embeddings computed once per frame and cached on the GPU (10 MB/frame fp32). |

- **Geometry chains:**
  - `da3` is the fast path's own geometry: DA3-GIANT any-view on all 224 grid frames. It is cached in
    `runs/m3-fu-e6b-outlines-da3`.
  - `ref` is round 1's chain: DROID poses + DA3 posed depth.
- **Metrics:** all are computed on the in-between frames, on the DROID raster, against `me340-masks-194`.
  - **M1:** the best IoU of each of today's masks (at least 300 px) with one projected region. This is round 1's
    metric.
    - "Per keyframe mask" regions keep round 1's granularity. "Per 3D object" regions use the lift's objects.
    - "Static" leaves out masks that are at least 50% on people. In the fast path, people come from SAM 3.
  - **M2:** object-map entities (`me340-entity-names-200`). Each entity's mask on an in-between frame is compared with
    the union of its projected keyframe masks.
- **Speed:** speedup = 224 × AMG s/frame / (keyframes × AMG + lift + keyframe points + render [+ snap]
  [+ encoder + decoder]). AMG is timed on the same A100.
  - Depth and poses are not charged, because the geometry layer computes them anyway.
  - Model load, cold start and transfer are listed separately below.

## Rerun

```
PY=/Users/adam/Desktop/Tesla/panoptes-platform/.venv/bin/python
$PY modal_apps/outline_projection_probe.py --self-check          # no GPU
/Users/adam/Desktop/Tesla/panoptes-platform/.venv/bin/modal run modal_apps/outline_projection_probe.py \
    --out /Users/adam/Desktop/panoptes-public/research-notes/phase2/runs/m3-fu-e6b-outlines-NNN [--chains ref,da3] [--steps 2,5] [--smoke]
$PY modal_apps/outline_projection_probe.py --verdicts RUN_DIR    # speed, IoU, pass flags, USD estimate
```

- The first run builds two caches:
  - `runs/m3-fu-e6b-outlines-inputs/ref-raw-depth.npz`: about 7 s on the Mac, 64 MB.
  - `runs/m3-fu-e6b-outlines-da3`: one A100 call.
- Each chain runs in its own A100 container, and the two run in parallel.
- At `--save-step` (default 5 = 2 fps), each run writes:
  - `<chain>-maps-step5.npz`: 3D-object label maps for all 224 frames;
  - `<chain>-outlines-step5.json.gz`: polygons in 1280x720 source pixels. Each frame has `source: segmented | projected`.
    `dec_pt` frames are `segmented` with `prompt: projected`.
  - `<chain>-sheet-step5.png`: today's outlines in green, the variant's in magenta.

## Results, 2026-09-28

Official run: `runs/m3-fu-e6b-outlines-002`. Run `-001` is the same code without the `*_vis` variants; its numbers are
identical, and it holds the DA3 call. Smoke runs are `-smoke-1` to `-smoke-5`. M = measured.

**`pair` vs keyframe rate**, DA3 any-view chain. Values are mean / area-weighted.

| Keyframes (of 224) | Speedup (M) | M1 per keyframe mask, static (M) | Same, `ref` chain (M) | M1 per 3D object, static (M) | M2 object outlines (M) |
|---|---|---|---|---|---|
| 5 fps (112) | 1.98x | 0.796 / 0.849 | 0.804 / 0.853 | 0.730 / 0.787 | 0.620 / 0.758 |
| 2.5 fps (56) | 3.91x | 0.780 / 0.825 | 0.790 / 0.830 | 0.745 / 0.787 | 0.611 / 0.739 |
| 2 fps (45) | 4.84x | 0.774 / 0.826 | 0.783 / 0.830 | 0.743 / 0.790 | 0.608 / 0.743 |
| **1.7 fps (38)** | **5.73x** | **0.766 / 0.813** | 0.776 / 0.817 | 0.745 / 0.801 | 0.598 / 0.728 |
| 1.25 fps (28) | 7.67x | 0.741 / 0.803 | 0.755 / 0.810 | 0.723 / 0.792 | 0.590 / 0.732 |
| 1 fps (23) | 9.25x | 0.731 / 0.785 | 0.743 / 0.791 | 0.717 / 0.777 | 0.580 / 0.711 |

**All variants at 1.7 fps**, DA3 chain. Each cell is mean / area-weighted.

| Variant | Total s (M) | Speedup (M) | M1 per keyframe mask, static | M1, all masks (mean) | M1 per 3D object, static | M2 |
|---|---|---|---|---|---|---|
| `prev_r1` (round 1) | 28.7 | 5.79x | 0.717 / 0.776 | 0.706 | 0.695 / 0.764 | 0.560 / 0.691 |
| `prev` | 28.6 | 5.79x | 0.718 / 0.776 | 0.681 | 0.696 / 0.765 | 0.560 / 0.692 |
| **`pair`** | 29.0 | 5.73x | **0.766 / 0.813** | 0.729 | 0.745 / 0.801 | 0.598 / 0.728 |
| `pair_snap04` | 29.6 | 5.60x | 0.759 / 0.812 | 0.721 | 0.737 / 0.802 | 0.592 / 0.729 |
| `pair_snap` (0.08) | 29.8 | 5.56x | 0.763 / 0.814 | 0.725 | 0.742 / 0.804 | 0.596 / 0.732 |
| `pair_snap16` | 29.8 | 5.56x | 0.766 / 0.815 | 0.728 | 0.744 / 0.805 | 0.599 / 0.733 |
| `pair_vis` | 29.0 | 5.72x | 0.727 / 0.786 | 0.692 | 0.709 / 0.778 | 0.578 / 0.714 |
| `fused` | 29.3 | 5.66x | 0.138 / 0.127 | 0.133 | 0.294 / 0.336 | 0.400 / 0.517 |
| `fused_snap` | 30.2 | 5.50x | 0.156 / 0.141 | 0.150 | 0.325 / 0.372 | 0.447 / 0.577 |
| `fused_vis` | 30.0 | 5.53x | 0.489 / 0.534 | 0.468 | 0.484 / 0.541 | 0.408 / 0.539 |
| `dec_box` | 38.8 | 4.27x | 0.771 / 0.803 | 0.733 | 0.748 / 0.800 | 0.596 / 0.729 |
| `dec_pick` | 39.2 | 4.23x | 0.768 / 0.815 | 0.731 | 0.746 / 0.808 | 0.595 / 0.737 |
| `dec_pt` | 39.1 | 4.24x | 0.771 / 0.802 | 0.738 | 0.746 / 0.791 | 0.591 / 0.710 |

At 1 fps the decoder starts to pay: `dec_box` gives 0.750 / 0.806 at 5.8x, against `pair`'s 0.731 / 0.785 at 9.25x.

**Seconds (M, warm, `ref` / `da3` containers, 1.7 fps):**
- Today's AMG on the same GPU: 0.73–0.74 s/frame, so 164–166 s for 224 frames (A100 PCIe).
- Keyframe AMG: 28 s. This is almost all of the fast path's cost.
- Lift: 0.01–0.03 s.
- Keyframe points: 0.14–0.17 s.
- Per in-between frame:
  - `pair` render: 3.6–4.1 ms;
  - snapping: 4.4–4.6 ms more;
  - SAM 2 encoder: 27 ms (batch 16), plus 26–28 ms for prompts and decoder.
- DA3-GIANT any-view forward (A100 SXM4): 5.7 s for 112 views and 17.8 s for all 224. Depth on every 10 fps frame
  therefore costs 12 s more than the 5 fps geometry pass.
- Cold start and transfer:
  - DA3 model load: 18 s.
  - SAM 2 load: 3.8 s.
  - Container ready: 15–50 s after the call.
  - Inputs uploaded per call: 125 MB for `ref` (55 MB round-1 geometry + 64 MB raw depth + 2 MB masks + 3 MB video)
    and 40 MB for `da3` (34 MB DA3 output). The DA3 output came back as 34 MB.
- USD (E, seconds × list price): run 001 $0.72, run 002 $0.61, smoke runs about $0.27. Total about $1.6.

## Verdict: nothing reaches 0.8 mean IoU at 5x

- **The mean cannot reach 0.8.** Mean IoU per mask stays below 0.8 at every keyframe rate, even at 5 fps, where the
  carry spans only 0.1 s (0.796 on `da3`, 0.804 on `ref`). Today's per-frame AMG masks change granularity from frame to
  frame (round 1's finding), so **0.80 is the ceiling for any carry**. The `pair` projection at 1.7 fps keeps 96% of
  that ceiling.
- **On area-weighted IoU, `pair` passes.** At 1.7 fps it gives 0.813 at 5.7x on `da3` (0.817 on `ref`). At 1.25 fps
  it gives 0.803 at 7.7x on `da3` (0.810 on `ref`).
- **M2 is capped the same way.** Entity outlines score 0.60–0.62 mean even at 0.1 s, because the object map picks a
  different mask per frame. M2 cannot tell the variants apart above that noise.
- **Recommendation for the fast path:**
  - Project with `pair` at about 1.7 fps.
  - Use DA3 any-view geometry. It costs only 0.01 IoU against the DROID chain.
  - Label in-between outlines `projected` and keyframe outlines `segmented`.
  - The render costs 4 ms/frame. The real speed lever is the keyframe segmenter: AMG is 97% of the time.
- **Snapping does not help.** It rejects 5–13% of projected pixels, depending on tau, and moves results by -0.007 to
  +0.002. AMG boundaries follow colour, not DA3 depth edges.
- **Fusing all keyframes hurts.** The keyframes' per-frame depths disagree by a few %, so a global z-buffer lets
  other keyframes' surfaces win (z-fighting and bleed-through). A visibility tolerance (`fused_vis`) recovers only part
  of the loss. The nearest keyframe must win.
- **The SAM 2 decoder is not worth its cost at 1.7–2 fps.** Its IoU is the same as `pair` and speed drops to about 4x.
  At 1 fps it recovers 0.02.

## Caveats

- **People:** people are not projected. Projected background outlines are drawn over a person who has moved into the
  view (see the sheet, frame 729). The fast path should cut projected outlines with the SAM 3 person masks.
- **Poses for in-between frames:** the `da3` chain ran DA3 on all 224 frames, so in-between frames have their own
  poses. At 5 fps geometry, these poses would be interpolated (E1: held-out photometric 1.16x). This is untested here.
- **Keyframe masks:** keyframes use today's SAM 2 AMG masks, so that the reference is like for like. SAM 3 vocabulary
  masks on keyframes would project the same way, but were not scored.
- **Scope:** one clip only (ME340 walk). Metric scale is borrowed from the reference: 1.6 m camera height.
