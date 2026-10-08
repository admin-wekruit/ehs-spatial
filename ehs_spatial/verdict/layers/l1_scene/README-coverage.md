# L1 — coverage (`coverage.py`) and the sigma policy (`sigma.py`)

Both produce C1 fields the measurement layer does not: `Scene.coverage` (the observed region the enclosure rule needs) and a
sigma for every box dimension (the guard band needs one). Neither reads meshes or runs a model; `coverage.py` reads the frozen
per-photo geometry (points behind pixels + camera pose), `sigma.py` reads nothing but the Scene.

## 1. Coverage: what the photos actually saw

### Why

`enclosed(Z)` asks whether a person can walk from outside into the hazard zone. On the trial's occupancy grid every cell outside
an object footprint was *unknown*, so the rule could only answer CANNOT_DETERMINE ("the photos did not cover the region and a
real opening look the same"). The fix is Hydra's *places* idea and `docs/archive/phase2/INFERENCE-POLICY.md`'s free-space
refutation turned around: a cell is **observed** when some ray from a camera passed through, or ended on, the slab 0–2 m above
the floor over that cell. Everything else stays **unknown** and is never treated as empty.

### Inputs

One directory per photo, the frozen geometry of the published variant
(`research/module-swap-2026-10-07/data/checks/bbab-geom/<cell>-mvs-fill-padded/geometry/frames/frame_000k`; the
`candidate_manifest.json` beside `frames/` names the model and the cameras it used):

| file | content |
|---|---|
| `pts3d.npy` | H × W × 3 (518 × 518) surface point behind every pixel, world frame of the producer |
| `valid_mask.npy` | H × W bool, pixels that have a surface point (MVS kept 49 / 46 / 51 % of 090's pixels, 37 / 21 % of 030's) |
| `camera_to_world.npy` | 4 × 4, the camera centre is column 3; frame 1 is the world origin |
| `conf.npy` | 1 on MVS pixels, 0.3 on MoGe-3-filled pixels inside object masks; `min_conf=0.5` excludes the filled ones |
| `intrinsics.npy` | unused: `pts3d` already is the end of every pixel's ray |

**Frame convention (checked 2026-10-08).** The frames are in the producer's *native* world frame; the measurement layer (and
so the trial's `out/scene-<cell>.json`) is that frame scaled by `scale.nativeToMeters` of the published layer
(`a9a6e0a0…json`: 3.2371 for 090, `fafdeb6b…json`: 3.5616 for 030) — same rotation, same origin, no translation
(`center_m == centerNative × nativeToMeters` for every box). The layer's `ground.normal` equals the scene's `ground_normal`,
and its `ground.offset` (0.4513 / 0.4089 native) times the scale gives the floor plane at n·p = −1.461 / −1.456 m, which is
exactly what `floor_level(scene)` recovers from the boxes (n·centre − H/2 − bottom). Photo id k of `Obj.views` is `frame_000k`.
So the adapter has to pass `native_to_m=nativeToMeters`; with it, the frames and the Scene share one metric frame.

### Method (`coverage_from_frames`)

1. Keep the valid pixels of each frame at `stride` (default 4: on 518 px that is one ray per ~6 cm at 6 m, finer than the
   10 cm cell) with `conf ≥ min_conf`; scale points and camera centre by `native_to_m`.
2. March every ray from the camera centre to its surface point in steps of `cell_m / 2`; the last sample is the surface point
   (rays longer than `max_range_m` = 15 m are cut; memory, not physics). Rays are processed in chunks of 1024 so the largest
   array is 1024 × 300 × 3 doubles (7 MB).
3. A sample counts when its height above the floor plane is in `[h_min, h_max]` = [−0.10, 2.0] m (−0.10 keeps noisy floor
   hits; 2.0 is the reach height the rules care about). The plan cell under every counted sample is marked observed:
   free space along the ray, the surface at its end.
4. `Coverage(cell_m, origin_xy, basis, shape, observed=[[ix, iy], …] sorted, method="…")`. `attach_coverage(scene, cov)`
   returns the Scene with it.

The plan frame is the trial's convention (`relations.plan_basis` / `plan_grid`): `plan_frame(scene, cell_m=0.1, margin_m=1.0)`
gives `(basis, origin_xy, shape)` from the box corners plus a margin and reproduces the trial's grids exactly (090: 75 × 55,
030: 83 × 49). L2 should build its `Grid` with the same call so `Grid.observed` can be copied from `Scene.coverage`.
`floor_level(scene)` gives the plane offset from the Scene; `estimate_floor_level(frames, …)` (most populated 5 cm height bin,
refined by the median around it) is the fallback when there is no Scene, and agrees with the Scene value to 1.4 mm on 090.

### Result on the benchmark frames (stride 4, min_conf 0.5, 0.2 s per cell)

| cell | frames | grid | floor level n·p (scene / frames) | observed cells | ratio | per frame |
|---|---|---|---|---|---|---|
| 090 | 3 | 75 × 55 = 4125 | −1.4608 / −1.4594 m | 1667 | **0.404** | 1362 / 1435 / 1529 |
| 030 | 2 | 83 × 49 = 4067 | −1.4562 / −1.4562 m | 1241 | **0.305** | 925 / 1080 |

The maps are a fan from the camera positions over the cell's floor; the cells of every floor-contact bollard are observed in
both scenes, the far side of the robot and the whole margin behind the fences are unknown. The photos overlap heavily
(three viewpoints a few decimetres apart), which is why the union is only ~10–20 % larger than the best single frame.
`tests/verdict/test_coverage.py::test_frozen_090_frames_carve_the_floor_under_the_bollards` reruns 090 at stride 8.

### Limits

- **A column is not a slab.** A cell is observed when *some* ray crossed the 0–2 m slab above it; a ray 1.9 m up says nothing
  about a 0.4 m obstacle below it. For the enclosure rule this means "observed" cells can still hide a low barrier; a per-cell
  minimum observed height is the next step (needs a `Coverage` field, see §3).
- **Observed ≠ free.** Cells under object surfaces are observed too (the ray ended there). L2 must keep `blocked` from the
  footprints and treat `observed` only as "known", never as "free".
- **One wrong depth carves through real things.** No multi-view consensus: a far outlier depth marks every cell on its way.
  `valid_mask` and `conf` are the only gates (MVS kept pixels, filled pixels excluded by default).
- **Stride pinholes.** At stride 4 the rays are ~6 cm apart at 6 m; thin unknown gaps between rays are possible beyond ~8 m.
  Use stride 2 for a final map; the synthetic tests use stride 1 on 64 px frames.
- **One flat floor.** Heights are measured from the Scene's single floor plane; steps and ramps are not modelled.
- **Native units.** Without `native_to_m` the frozen MVS frames are ~3.2–3.6× too small and the slab test is meaningless.

## 2. Sigma policy (`sigma.py`)

The trial used a flat 5 cm default for every unmeasured dimension (flag `default_sigma`), which alone turned two floor-gap
verdicts into NEEDS_MEASUREMENT. `fill_sigma(scene)` instead assigns, per object confidence, the median of the sigmas the
measurement layer *did* produce in the two benchmark scenes (all dimensions L / W / H / bottom pooled); a level without data
takes the next worse level's value; `unverified` is at least 5 cm.

Calibration (`calibrate()` on `research/verdict-layer-trial-2026-10-07/out/scene-090.json` + `scene-030.json`, 2026-10-08;
`tests/verdict/test_sigma.py` recomputes it):

| confidence | measured values | median sigma | table value |
|---|---|---|---|
| high | 0 | — | 0.0037 m (inherits medium) |
| medium | 17 (two bollards, three light curtains) | 0.0037 m | **0.0037 m** |
| low | 6 (one bollard, one light curtain) | 0.00395 m | **0.00395 m** |
| unverified | 2 (030 right bollard: L 0.0052, H 0.1022) | 0.0537 m | **0.0537 m** (≥ 0.05 floor) |

Filled keys are recorded in `scene.declared_inputs['sigma_filled']` as a JSON list of `"<obj_id>:<dim>"` (Obj has no flags
field); the input Scene is not mutated and the call is idempotent.

Caveat, stated plainly: the table is calibrated on the dimensions that *were* measurable (posts and curtains with clean
edges), so 4 mm for a `low` fence whose every dimension went unmeasured is optimistic — survivorship, not evidence. It is the
policy that was asked for; the honest next step is a per-class or per-dimension table once more scenes carry sigmas, and L2
flagging facts built on filled sigmas (read `declared_inputs['sigma_filled']`) the way the trial flagged `default_sigma`.

## 3. Contract requests (not changed here)

- `Obj`: a `flags: list[str]` (or `sigma_source: dict[str, str]`) so filled sigmas are visible on the object, not in a JSON string.
- `Coverage`: optional `occupied: list[list[int]]` (cells where rays ended) and `min_height_m: list[float]` per observed cell,
  so the enclosure rule can require the floor itself to have been seen.
- `Scene`: `floor_offset_m: float | None` (the plane's n·p) straight from the layer's `ground.plane[3] × nativeToMeters`
  instead of deriving it from the boxes.
- The plan-grid convention (`plan_frame`) should live in `contracts.py` (a `GridSpec` shared by `Coverage` and `Grid`) rather
  than being duplicated by L1 and L2.
