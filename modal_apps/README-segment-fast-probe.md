# E2 + E6 segmentation probe (fast path)

`segment_fast_probe.py` measures, on the ME340 walk shot, the two segmentation experiments of
`docs/phase2/FAST-PATH-PLAN.md` section 7. All GPU work runs on Modal A100-80GB (ephemeral `modal run`, retries 0).

- **E2:** SAM 3 on ~2 fps keyframes (frames 228, 243, ..., 888; 45 frames). The vision encoder runs once per frame and the
  text encoder once per vocabulary. Many (frame, prompt) pairs go through one forward pass. It measures warm s/frame for
  {person, floor}, the 20 EHS words, and today's own object names (an oracle vocabulary). It also reports recall of
  today's named objects: an object counts as found when some SAM 3 mask has IoU >= 0.5 with one of its observed masks
  on a shared keyframe.
- **E6:** SAM 2 automatic masks run only on keyframes (every 6, 15 or 30 frames). Today's keyframe masks are carried to
  the in-between frames of today's 10 fps grid in two ways:
  - forward reprojection with the reference DROID poses and DA3 posed depth;
  - SAM 2 video propagation, run twice: once with sam2 main (one object at a time) and once with sam2 `c2ec8e1`, the
    last commit that batched all objects.

  The carried masks are scored against today's per-frame masks (best IoU per mask, on DROID's raster). Timing is
  compared with today's AMG loop on the same A100. The probe also runs AMG with bigger point batches, and box prompts
  with one image encode against today's re-encode-per-box loop.

## Rerun

```
cd <worktree>
PY=/Users/adam/Desktop/Tesla/panoptes-platform/.venv/bin/python
$PY modal_apps/segment_fast_probe.py --self-check                   # no GPU
/Users/adam/Desktop/Tesla/panoptes-platform/.venv/bin/modal run modal_apps/segment_fast_probe.py \
    --out /Users/adam/Desktop/panoptes-public/research-notes/phase2/runs/m3-exp-e2-e6-segment-probe-N [--only sam3,sam2,sam2old] [--smoke]
$PY modal_apps/segment_fast_probe.py --verdicts <that run dir>     # pass/fail, speedups, USD estimate
```

The first run builds the inputs locally into `runs/m3-exp-e2-e6-segment-inputs/`:
- today's masks as label maps;
- DA3 depth and DROID poses on DROID's raster;
- the E2 keyframe PNGs;
- named observations.

This takes about 25 s of Mac CPU. The three probes run in parallel, one A100 container each. Each writes `<name>.json`
and its maps as `.npz` (`sam3-maps-0.npz` holds the production-rule SAM 3 masks for {person, floor} + EHS).

## Results, 2026-09-28

All numbers are warm, on the ME340 walk shot. M = measured, E = derived from measured seconds x list price.
Run folders: `runs/m3-exp-e2-e6-segment-probe-1` (all three probes) and `-probe-4-vocab40` (the SAM 3 40-word
follow-up). Each has a `verdicts.json`.

**E2 (SAM 3, A100 80GB PCIe):**

| Vocabulary | Batched s/frame (M) | Today's loop s/frame (M) | Recall, production rule (M) | Recall, score >= 0.3 with no cap (M) |
|---|---|---|---|---|
| {person, floor} | 0.066 | 0.160 | – | – |
| 20 EHS words | 0.191 | 0.941 | 47/93 = 51% | 52/93 = 56% |
| 40 words (20 EHS + 20 generic, chosen after seeing the misses) | 0.316 | – | 68/93 = 73% | 75/93 = 81% |
| Today's 80 names (oracle, upper bound) | 0.619 | – | 79/93 = 85% | 86/93 = 92% |

- The production rule is score >= 0.4 and the top 12 masks per word.
- Recall counts the named objects (clear or partial) that have an observed mask on at least one of the 45 keyframes.
  5 more named walk objects are never seen on a 2 fps keyframe.

**E6 (SAM 2, keyframes plus carried masks):**

Today's AMG loop on the A100 takes 0.670 s/frame, which is 150 s for the 224 walk frames at 10 fps (M).

| Keyframes | Reprojection: speedup, mean / area-weighted IoU | Video propagation, batched objects: speedup, IoU |
|---|---|---|
| 5 fps | 2.0x, 0.80 / 0.84 | 0.8x, 0.83 / 0.89 |
| 2 fps | 4.9x, 0.73 / 0.77 | 1.3x, 0.80 / 0.86 |
| 1 fps | 9.4x, 0.66 / 0.69 | 1.5x, 0.75 / 0.82 |

- IoU is the best IoU of each of today's masks against the carried masks.
- Reprojection costs about 2 ms per frame. The rest of its time is keyframe AMG.
- Even a carry from the previous 10 fps frame (0.1 s earlier) agrees with today's masks at only 0.80 (reprojection) or
  0.83 (video). Copying the masks with no warp gives 0.70. Today's per-frame AMG masks are not stable frame to frame,
  so a bar of >= 0.85 mean IoU against them is out of reach for any carry.
- sam2 main (one object at a time) gives the same masks as the batched commit, but runs 0.65 s per propagated frame
  against 0.26 s. The sam2 main run used a PCIe A100 and the batched run an SXM4 A100.
- Adding today's ~53 masks per keyframe costs 0.4–0.75 s per keyframe.

**Other SAM 2 measurements:**
- AMG with points_per_batch 256 or 1024 is only 5% faster, with identical masks.
- A 16x16 point grid runs at 0.19 s/frame but keeps only 56% of today's masks.
- Box prompts: encoding the image once and batching all boxes took 0.64 s for 526 boxes, against 20.7 s for today's
  re-encode-per-box loop (32x). The masks are identical (IoU 0.9985).
