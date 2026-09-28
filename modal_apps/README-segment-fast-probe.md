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
