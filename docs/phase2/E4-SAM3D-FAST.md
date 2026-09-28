# E4: SAM 3D Objects fast mode (FAST-PATH-PLAN section 7)

Probe: `modal_apps/e4_sam3d_fast.py`. Results: `research-notes/phase2/runs/m3-exp-e4-sam3d-1/`
(`index.json`, `inputs.json`, `bench.json`, `gate-all.json`, `summary.json`).

## What it tests

- **Objects:** the first 30 objects that `runs/me340-object-models-241-sam3d` modelled, in its priority order (EHS equipment first).
- **Gate:** the unchanged gate. `scripts/complete_video_objects.py` runs read-back only, with 241's arguments: `--generator sam3d --skip-frames 14-225 --entities <one id>`, and no `--invoke`. Only the SAM 3D journal differs between arms.

| Arm | Settings | Tries |
|---|---|---|
| baseline | 241's own journaled meshes (Mac inputs), judged again on Linux | the ones 241 made |
| current | upstream defaults: stage 1 25 steps + CFG 7, stage 2 25 steps + CFG 5 | every try |
| fast | `use_stage1_distillation`, stage 1 4 steps, stage 2 12 steps | every try |
| fast-s2d | fast, plus `use_stage2_distillation` with stage 2 at 4 steps | every try |
| fast-crop | fast, on the source-grid crop and its K instead of the whole frame | attempt 1 |

"Every try" is the runner's own order: up to 4 views, best first, then the best view with seed 43. The runner stops at the first accepted try.

GPU: one A100-80GB container (8 cores, 64 GiB). Each SAM 3D process is a subprocess that loads the model, makes one warm-up call, and waits at a barrier.
- A: 1 process runs every arm, plus a 20-call throughput block.
- B: 2 processes run the same 20 calls.
- C: 2 processes under NVIDIA MPS, same 20 calls.

## Rerun

All steps are ephemeral runs; nothing is deployed. State lives in the Modal volume `panoptes-e4-sam3d`, mounted at the phase2 path.

```
M=/Users/adam/Desktop/Tesla/panoptes-platform/.venv/bin/modal
$M run modal_apps/e4_sam3d_fast.py::upload        # 241's inputs for the 30 objects + its SAM 3D journal (~1.2 GB, no compute)
$M run modal_apps/e4_sam3d_fast.py::inputs_step   # CPU x30, ~9 min wall: every try's SAM 3D input; maps 241's meshes to them
$M run modal_apps/e4_sam3d_fast.py::bench_step    # GPU, ~40 min (--smoke: 2 objects, ~5 min)
$M run modal_apps/e4_sam3d_fast.py::gate_step     # CPU x30: gate per object and arm (--only ID --arms a,b for one)
python modal_apps/e4_sam3d_fast.py summary        # summary.json from the local results
python modal_apps/e4_sam3d_fast.py self-check
```

## Caveats

- On Linux, FFmpeg decodes the source frames within ±3 levels of the Mac decode. The SAM 3D journal keys therefore differ from 241's.
- 241's meshes are mapped to the new tries by (observation, seed): 89 of its 92 tries mapped.
- One object (object-026) picked different views on Linux, because the sharpness scores were close.
- The current rerun and baseline differ only in those decode levels and GPU nondeterminism. Together they show the run-to-run noise of the gate.
