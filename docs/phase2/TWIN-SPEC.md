# M4 twin spec: a clean ME340 digital twin from measured geometry (2026-09-28)

**摘要：**照 Stanford HomeBody 的 Real2Sim 做法：VLM 只负责"这是什么、用什么表示、什么材质"，尺寸和位姿全部来自我们已测的几何（DROID-SLAM 相机 + DA3 深度 + LingBot 稠密点 + 地面平面），再从真实相机位姿渲染孪生体，逐物体和真实画面比轮廓、比深度，VLM 看对比图列出具体修改。孪生体里每个元素都是推断的替身，标注清楚，永远不拿来测量；观测层仍然是唯一的事实。三个构建者并行：房间壳体（shell）、物体（objects）、验证（verify）。

## 0. What we copy from HomeBody, and what differs

HomeBody (tml.stanford.edu/homebody): a frontier VLM acts as a Real2Sim agent, assembles a digital twin from reusable
assets and textures (their scene: 544 meshes, 27 textures, viewed in three.js and Isaac Sim), grounds it in SLAM
geometry, and checks it against matched real camera views (a divider over real vs sim images). Their FAQ: with video
alone the agent guesses room dimensions from appearance; measured SLAM geometry constrains them.

Their SLAM is LiDAR (metric). Ours is monocular:

| Their input | Ours (ME340) |
|---|---|
| LiDAR SLAM geometry | DROID-SLAM cameras (`droid-me340-165-171`) + DA3 posed depth fused over 240 views (`da3-posed-me340-223-shotc`) + LingBot dense map (2.9 M points, `me340-lingbot-map-222`) |
| metric scale | one global factor, 2.8592 m per native unit, from an **assumed** 1.6 m camera height (operator's statement, `scale_status: assumed_camera_height`); the camera-height p10-p90 spread gives about -6 / +4 % |
| stereo depth per view | DA3 depth per view, anchored to DROID (`mono_room.load` scale per view) |
| SAM 2.1 masks | SAM 2 masks (`me340-masks-194`), 205 entities in a named object map (`me340-entity-names-200`) |

So "dimensions from measured geometry" here means: from our points, up to that one stated scale factor. Every metre
value in the twin carries it (`twin.json.scaleStatus`).

## 1. Policy (unchanged from INFERENCE-POLICY.md)

1. Every twin element is INFERRED: a stand-in. It is labelled so in every file (`layer: "twin_inferred"`,
   `notForMeasurement: true`) and in any UI ("Digital twin: inferred stand-ins, not measurements"). No EHS rule and no
   measurement ever reads the twin. The observed layers (fused mesh, textured shell, dense points, masks) stay the truth.
2. Sizes and poses come from our measured geometry: object points, floor plane, wall points, free space. Never from a
   VLM number. The VLM chooses category, representation, part counts and materials, and reviews matched views. A VLM
   remark about geometry ("too tall", "should reach the floor") is a hint that triggers a re-fit under a named
   constraint; the re-fit is kept only if held-out metrics improve.
3. A dimension the video did not measure (the back of a cabinet seen only from the front) is an inferred extent:
   `max(measured span, min(category prior, free-space bound))`, labelled `bounded_prior`, and must pass the free-space
   test.
4. Tier D stays blank: nothing behind walls, in aisles nobody walked, or inside cabinets. The twin shows closed
   cabinets as closed boxes; it does not model contents.
5. What is verified is what ships: the verify builder assembles and exports the files it scored.

## 2. Inputs

Data root `R = /Users/adam/Desktop/panoptes-public/research-notes/phase2`. Read only; never modify these run dirs.

| What | Path (under `R/runs` unless noted) | Use |
|---|---|---|
| Clip | `R/data/clips/me340-165` (`source-full.mp4` 1280x720, raster crop x 160..1120) | frames, full-res crops |
| Shots | `me340-oneshot-cuts-f2d83b4e1c/segments.json`: [0,13], [14,225], [226,898] | walk shot = **226..898 only** |
| Cameras | `droid-me340-165-171/prediction.npz` (OpenCV c2w, native units) | every render |
| Per-view depth + scale | `da3-posed-me340-223-shotc/mono/*.npz` via `mono_room.load(droid, None, depth_run)` | observed depth; 235 views in 226..898 |
| Floor plane, scale | `da3-posed-me340-223-shotc/metric-scale.json` (same plane as the object map's `plan`) | frame, scale |
| Fused room mesh | `da3-posed-me340-223-shotc/mono-anchored-mesh.ply` (multi-view supported, with normals) | wall/ceiling points |
| Textured shell | `me340-filled-225/textured-scene.glb` (fused + single-view fills, atlas textured) | observed layer, reference renders |
| Dense points | `me340-lingbot-map-222/dense-points.glb` (+ `point-attributes.npz`) | wall/floor colours, texture bake |
| Inferred floor | `me340-inferred-floor-244/inferred-floor.{glb,json}` (187.7 m2 region, 57.9 m2 seen, tier-A tested) | floor slab outline |
| Object map | `me340-entity-names-200/object-map.json` + `surfaces/*--cells.npz` | entities, labels, observations |
| Masks | `me340-masks-194/{object-a,floor-a}/frame-NNNNN/` (clip raster) | silhouettes |
| Moving people | `me340-dynamic-masks-188/masks` | excluded pixels |
| Accepted models | `me340-object-models-303-merged/models/<entity>/{model.glb,validation.json}` (22, world frame) | reuse |
| Hand-built report | platform import c40fbd08 | comparison only; do not open `*/.platform/imports/*` |

**Do not size from `footprintPlanNative` / `heightNative`.** They are lifted from all 312 mask views including the
cut-away shot (14..225, another camera, wrong poses) and depth bleed: "motor" 105 spans 15.7 m, "projection screen" 023
16.7 m. Even `cells.npz` gives 023 7.4 m. Sizing uses the verified points of section 6.2.

### Code to reuse (import, do not copy)

- `scripts/complete_video_objects.py`: `Clip` (`k_raster`, `raster_mask`, `k_full`, `frames`), `reliable` (a view's
  reliable posed depth, moving person removed), `mask_path`, `main_parts`, `subtitle_box` (burnt-in captions),
  `view_metrics`, `pick_views`, `observed_points` (the gate's multi-view agreement), `box_mesh` (gravity-aligned box:
  minAreaRect yaw, 1st/99th percentiles), `ray_scene`, `render`, `look_at`, `sam3d_to_world`, `--generator sam3d`
  with `--max-usd`.
- `scripts/box_free_space.py`: `box_depths`, `classify` (overhang / foreign / occluded per pixel), `THRESHOLDS`
  (fitted on three scenes: overhang 0.1994, foreign 0.1209, sourceFaceBacking 0.6746), `SLACK`, `ERODE`, `MIN_JUDGED`.
- `scripts/infer_room_floor.py`: `camera_depths`, `see_through` (8 % + slack, the policy's free-space test).
- `ehs_spatial/geometry.py::_ransac_floor_plane`: the deterministic NumPy RANSAC pattern (seeded, no Open3D
  segment_plane).
- Gemini: `modal_apps/sam3_video_fal.py::execute` with the `REMOTE` program of `scripts/review_video_object_semantics.py`,
  exactly as `scripts/name_video_entities.py` does (section 8).

## 3. Frame and scale

- **Working frame: native** (`droid_final_native_world`). Shell and objects builders write geometry in native
  coordinates, the frame every camera, depth and mask already uses. Verify renders in native. No builder needs a shared
  transform module.
- **Export frame: `me340_twin_m`**, applied once by verify at export:
  `p_twin = s * R * (p_native - o)`, `s = metres_per_native_unit = 2.8591949327582626`,
  `o = plane_point_native`, `R` rows `[axis_a, up, axis_a x up]` (from `object-map.json.plan`; `axis_a` is orthogonal to
  `up`, so `R` is a right-handed rotation). glTF: metres, +Y up, origin on the floor.
- **USD**: `upAxis = "Z"`, `metersPerUnit = 1.0`; the root prim `/World/me340_twin` carries `xformOp:rotateX = 90` so
  the glTF-frame points map to Z-up ((x, y, z) -> (x, -z, y)).
- `twin.json.transform` stores the 4x4 native->twin matrix, `scaleStatus`, and the camera-height band.

## 4. Work split and interfaces

Three builders run in parallel, each in its own worktree from `m4/twin`
(`git -C /Users/adam/.codex/worktrees/panoptes-phase2-video worktree add /Users/adam/.codex/worktrees/panoptes-phase2-video-m4-KEY -b m4/twin-KEY m4/twin`,
KEY = `shell` | `objects` | `verify`). Each owns one new script; the only edit to an existing file is objects' factoring
of `box_fit` out of `box_mesh` (no behaviour change, its self-check still passes). Heavy GPU work goes to Modal
(ephemeral runs, bounded timeouts); CPU raycasting runs locally, one process at a time.

| Builder | Script | Output dir (new, `exist_ok=False`) | Consumes | Produces |
|---|---|---|---|---|
| shell | `scripts/build_twin_shell.py` | `R/runs/m4-twin-shell-NNN/` | inputs above, `--fixes` | `shell.glb`, `shell.json`, `review/` |
| objects | `scripts/build_twin_objects.py` | `R/runs/m4-twin-objects-NNN/` | inputs above, `--fixes` | `objects.glb`, `objects.json`, `review/`, `vlm/` |
| verify | `scripts/verify_twin.py` | `R/runs/m4-twin-verify-NNN/` | a shell dir, an objects dir | `twin.glb`, `twin.usda`, `textures/`, `twin.json`, `verify.json`, `sheets/`, `critique/`, `fixes.json` |

Each script has `--self-check` (repo convention: synthetic data, asserts, no network). Spend goes to the shared ledger
`R/runs/m4-twin-spend.jsonl` (section 9).

### 4.1 Node names

Path segments match `[a-z][a-z0-9_]*` (valid glTF names and USD prim names; no hyphens). Entity ids (`object-120`) live
in metadata only.

- shell: `shell/floor`, `shell/ceiling`, `shell/wall_00` .. `shell/wall_NN`
- objects: `objects/<twinId>/<part>`, twinId = `<category-short>_<nn>` (`wb_01`, `mc_02`, `bin_07`), parts from the
  generator (`top`, `leg_0`, `drawer_3`, `body`, `door`, `window`, `pendant`, ...); an existing model is one part `mesh`.

### 4.2 `shell.json` (schema `m4-twin-shell-v1`)

```json
{
  "schema": "m4-twin-shell-v1",
  "coordinateFrame": "droid_final_native_world",
  "metresPerNativeUnit": 2.8591949327582626,
  "plane": {"origin": [..], "up": [..], "axisA": [..], "axisB": [..]},
  "manhattanYawDeg": 0.0,
  "elements": [
    {"node": "shell/wall_00", "kind": "wall",
     "planeNative": {"point": [..], "normal": [..]},
     "spanNative": [s0, s1], "heightNative": [h0, h1],
     "cellNative": 0.035, "cells": "cells/wall_00.npz",
     "openings": [{"kind": "seen_through", "rectWall": [s0, h0, s1, h1], "areaM2": 4.1}],
     "support": {"inlierPoints": 5123, "observedShare": 0.41, "refutedShare": 0.004,
                 "controlFlaggedShare": 0.62, "testNoiseShare": 0.006},
     "material": {"key": "painted_block", "baseColourRgb": [..], "mode": "flat|crop|baked",
                  "texture": "textures/wall_00.png", "texelNative": 0.007, "chosenBy": "vlm/request-00"},
     "layer": "twin_inferred", "notForMeasurement": true}
  ],
  "absent": [{"kind": "wall", "reason": "no line passed support and bounds-the-room tests on the far side"}]
}
```

`planeNative.normal` points into the room; `spanNative`/`heightNative` are wall coordinates (along the line, above the
floor); `cells` holds one state per cell: observed | inferred | seen_through.
`shell.glb`: one mesh per element, node name = `node`, native frame, material baseColour + optional texture.

### 4.3 `objects.json` (schema `m4-twin-objects-v1`)

```json
{
  "schema": "m4-twin-objects-v1",
  "coordinateFrame": "droid_final_native_world",
  "metresPerNativeUnit": 2.8591949327582626,
  "objects": [
    {"twinId": "wb_01", "node": "objects/wb_01", "priority": 1,
     "category": "workbench", "representation": "parametric:workbench",
     "members": [{"entityId": "object-120", "relation": "body"},
                 {"entityId": "object-138", "relation": "face_of"},
                 {"entityId": "object-017", "relation": "separate"}],
     "box": {"centreNative": [..], "axesNative": [[x..], [y..], [z..]],
             "sizeNative": [w, d, h], "sizeM": [1.95, 0.76, 0.92],
             "dimSource": {"w": "measured", "d": "bounded_prior", "h": "support_to_floor"},
             "prior": {"d": 0.76}, "freeSpaceBoundM": {"d": 0.81}},
     "params": {"base": "drawer_cabinet_right", "drawers": 5},
     "parts": {"top": {"material": "butcher_block_wood", "colourRgb": [..], "texture": null},
               "base": {"material": "painted_steel", "colourRgb": [..], "texture": "textures/wb_01-front.png"}},
     "evidence": {"points": 18423, "pointsFile": "points/wb_01.npz",
                  "shapingViews": ["object:462:31", ..], "heldOutViews": ["object:465:30", ..],
                  "model": null, "vlm": ["vlm/request-02"]},
     "layer": "twin_inferred", "notForMeasurement": true}
  ],
  "skipped": [{"entityId": "object-008", "reason": "tiny clutter: 0.17 m, no accepted model"}]
}
```

`axesNative` rows are the box x, y, z (z = up, -y = front).
`objects.glb`: native frame, one node per part named `objects/<twinId>/<part>`. Colours as materials (not vertex
alpha); existing models are copied with their colours and alpha forced to 255 (the twin marks inference per node).

### 4.4 `fixes.json` (verify -> shell/objects, next run's `--fixes`)

```json
{"source": "m4-twin-verify-001", "fixes": [
  {"node": "objects/wb_01", "kind": "wrong_part_count", "part": "base", "value": "drawers=4",
   "evidence": "critique/request-03 pair 2", "action": "apply"},
  {"node": "objects/mc_02", "kind": "should_extend_to_floor", "evidence": "..", "action": "refit"},
  {"node": "shell/wall_02", "kind": "missing_opening", "evidence": "..", "action": "refit"}]}
```

`action`: `apply` (style: category, representation, part counts, material, colour assignment, drop, merge/split of
members) or `refit` (any geometric remark: the builder re-fits under the named constraint and keeps the result only if
its held-out metrics in verify improve). Nothing in a fix is a length.

## 5. Shell builder

All thresholds in metres are converted with `s`; all tests use walk views 226..898 only.

1. **Floor.** The plane of `metric-scale.json`. Outline = the `me340-inferred-floor-244` region (already tier-A tested):
   rasterise its cells, `cv2.findContours`, simplify 0.05 m, triangulate. One element `shell/floor`.
2. **Wall candidates.** Vertices of `mono-anchored-mesh.ply` with `|n . up| < 0.25`, height 0.3 m .. (ceiling - 0.3 m,
   or 3.0 m before the ceiling is known), and not an object: a vertex is dropped if it falls inside a confirmed
   non-stuff entity's mask in >= 50 % of the walk views that see it unoccluded (machines and benches are not walls;
   the object map's `cells.npz` is not used here, its depth-bleed streaks reach the walls).
3. **Wall lines.** Sequential 2D RANSAC on the floor plan (seeded NumPy, pattern of `_ransac_floor_plane`), inlier
   0.05 m. Keep a line if its inliers span >= 1.5 m along and >= 1.2 m in height, reach >= 3.0 m above the floor (or
   within 0.5 m of the ceiling), and >= 90 % of the seen floor cells lie on one side. A wall bounds the room and rises
   above the machines; a machine front 1-2 m before an unseen wall can pass the one-side test alone. At most 8 walls.
4. **Manhattan snap.** Dominant direction = support-weighted circular mean of wall directions mod 90 deg; snap each
   wall within 5 deg and refit its offset. Record `manhattanYawDeg`.
5. **Extent.** Along the line: the observed inlier span, extended to a neighbouring wall only where the two lines meet
   within 1.0 m of both observed ends (tier A: between observed edges). Height: floor to ceiling (or to the highest
   inlier + 0.3 m when there is no ceiling).
6. **Cells and openings.** Grid on the wall, cell 0.10 m. State: `seen_through` if `see_through` counts >= 2 views
   (8 % + 0.05 m), `observed` if a fused vertex lies within 0.05 m, else `inferred`. Connected `seen_through` regions of
   >= 0.5 m2 become rectangular openings (holes in the wall mesh). Closed doors are objects, not openings.
7. **Test the test** (policy test 2), per wall: the true wall flags <= 1 % of its observed cells; the same wall moved
   0.30 m into the room is flagged in >= 50 %. A wall failing either is not emitted (listed in `absent`).
8. **Ceiling.** Points with height >= 2.5 m and normal pointing down (`n . up < -0.8`): the highest horizontal plane
   holding >= 5 % of them. Region = floor outline intersected with the plan hull of its inliers grown 1.0 m. Same
   see-through and control tests. If none passes: no ceiling element, listed in `absent`.
9. **Materials.** Per element, `baseColourRgb` = median colour of the dense LingBot points within 0.02 m of the plane
   inside the element (measured). One Gemini request (section 8.2) picks `key` from the palette (section 7.3) and a
   `mode`: `flat` (colour only), `crop` (a clean patch the VLM boxes in one of the shown views, rectified by the known
   plane-to-camera homography and tiled; kept only if observed depth lies on the plane within 5 % over >= 90 % of the
   patch and it misses the caption band), or `baked` (dense points within 0.02 m splatted orthographically at 1 cm
   texels, holes filled with the base colour).
10. **Review.** `review/plan.png` (floor outline, wall lines with inlier support, openings, ceiling hull) and
    `review/wall_NN.jpg` (cell states over the wall).

Self-check: a synthetic room (4 walls, one 1 m door gap, a box machine near a wall, a ceiling) from synthetic cameras:
walls recovered within 0.02 m and 1 deg, the gap found as an opening, the machine face not taken as a wall.

## 6. Objects builder

### 6.1 Selection and priority

Candidates: `labelStatus` in {clear, partial}; not floor/wall/ceiling/unnamed surface/person; >= 8 observations in
226..898; verified extent (6.2) >= 0.25 m across. Exceptions: an entity with an accepted model in `303-merged` is
included at any size (it is free and already verified); an entity matching an EHS term (`EHS_TERMS` of
`modal_apps/qwen3vl_retrieval_probe.py`: machine guard, electrical panel, fire extinguisher, exit sign, ...) is
included at any size. Skip tiny clutter otherwise.

Priority: **1** machine tools and their guards and control panels, workbenches, doors (egress); **2** storage and
material handling (drawer cabinets, racks, bins, tubs, carts, the surface plate, pedestals); **3** wall- and
ceiling-mounted items (monitors, boards, signs, mirrors, light fixtures, cable tray, duct). Build and verify in
priority order; P3 only if time and budget remain.

### 6.2 Verified points (the measurement)

Per entity: `observed_points` (the gate's rule: masked reliable depth of up to 16 views that agree with the best view,
moving person removed), restricted to observations in 226..898, masks eroded 2 px. Then a silhouette test: keep a
point if it projects inside the entity's mask (dilated 3 px) in >= 60 % of the entity's views where it is in frame and
not occluded (observed depth at its pixel >= 0.95 x its depth). Then the largest connected component at 2 voxels.
Views are split as the box gate splits them (`i % 2`): only the shaping half builds the box; the held-out half is
verify's. Group points = union of member points with relation `body` or `face_of`.

### 6.3 Grouping (the agent's assembly step)

1. Geometric proposals: entities whose verified-point boxes overlap or lie within 0.05 m, union box <= 4 m across.
2. `groups.jpg`: per proposal, the best context view with each member outlined in its own colour and labelled.
3. The style request (8.3) returns per member a relation: `body` (the main mass), `face_of` (a face or panel of the
   body: absorbed into the generator, e.g. a machine's door, window, control pendant), `separate` (its own twin
   object), `sits_on` (its own twin object resting on this one), `not_this_object`.

Provisional ME340 groups (from the entity crops; the VLM confirms or splits):

| Twin object | Members (provisional) | Category | Priority |
|---|---|---|---|
| machining centre "#2" | 109 cnc control panel, 125 glass window, 036 logo sign, 033 door (verify adjacency: 033 may be the next machine) | machine_enclosure | 1 |
| machining centre | 043 cnc control panel, 019 cnc controller panel (if adjacent) | machine_enclosure | 1 |
| manual mill | 018 machine head, 189 milling table (RecGen), 186 handwheel (RecGen), 105 "motor" (check) | machine_column | 1 |
| chuck guards | 116 protective shield, 182 plastic guard | guard_shield (on their machine if one is modelled) | 1 |
| workbench(es) | 120 tabletop, 138 "workbench legs", 101/129/147 bench parts, 073 drawer (RecGen) | workbench with drawer pedestal | 1 |
| roll-up door | 034 garage door | door_rollup (in the wall plane, 0.01 m into the room) | 1 |
| vises | 017 (SAM 3D), 104 | vise (sits_on bench) | 1 |
| storage | 001 tool holder rack (SAM 3D), 002 surface plate (box), 035 tub (SAM 3D), 012 (RecGen), 056 (box), 130 (RecGen), 131, 106, 009 bins, 039 tray, 115 pedestal, 148 case (SAM 3D) | rack / bin / tub / pedestal | 2 |
| bench items with accepted models | 085, 094, 099, 110, 127, 136, 137, 150 | existing model | 3 |
| mounted | 007, 102, 184 monitors; 006 (named "dry erase board", crop looks like a control screen: re-ask); 023 (named "projection screen", crop overexposed: re-ask); 016 mirror; lights 005, 024, 028, 029, 042, 049, 128, 135; 025 cable tray; 037 duct | panel_screen / light_fixture / duct_tray | 3 |

Never: 031 (a person), `not_an_object` entities, 159 floor drain (part of the floor).

### 6.4 The measured box

1. Yaw: `box_mesh`'s minAreaRect of the points on the floor plan (factor out `box_fit(points, up) -> axes, lo, hi`).
   Snap to the mode of all object yaws mod 90 deg (weighted by footprint area) when within 10 deg.
2. Extents: 1st/99th percentiles on the shaping points. A face is observed if >= 5 % of the points lie within 0.05 m
   of it, contributed by views whose camera is on that face's outer side. A horizontal axis is `measured` when both
   its faces are observed, else `bounded_prior` (a front-only view gives a thin slab, not a depth).
3. Front: the box face whose outward normal is closest to the median direction toward the cameras that saw the object.
   `-y` of the local frame points to the front.
4. Bounded depth: the unseen face moves away from the seen face to `max(measured span, min(prior, free-space bound))`.
   The free-space bound
   is the smallest push at which the new slab gets `see_through` >= 2 views, or touches another object's box or a
   wall. Priors (m): workbench/table 0.76, drawer_cabinet/cabinet 0.60, shelf/rack 0.45, machine_enclosure 1.80,
   machine_column 1.00, cart 0.60, chair/stool 0.50, bin 0.40, tub 0.60, crate_box 0.40, door_hinged 0.05,
   door_rollup 0.10, panel_screen 0.06, guard_shield 0.02, light_fixture 0.15, duct_tray 0.30. A prior is a stated
   default in code, never a VLM answer.
5. Support: floor-standing categories (machines, workbench, table, cabinets, shelf, rack, cart, chair, stool, tub,
   door, and bin/crate_box with base <= 0.35 m) extend to the floor (`support_to_floor`) unless >= 2 views see through
   the added volume, in which case the base stays measured and the object is flagged. `sits_on` objects rest on their
   support's top when within 0.10 m of it.
6. Accepted models keep their own mesh and pose (`model` dims); the box is still written for verify and the critique.

### 6.5 Representation rule (first that applies)

1. An accepted model in `303-merged` covers the whole twin object (the member is `body` and the object has no other
   `body`/`face_of` member) -> `existing:<entity>`.
2. Box-like category -> `parametric:<generator>` on the measured box. An accepted `box` model (002, 056) is used as the
   measured box, not as the mesh.
3. Not box-like (machine_column, guard_shield, vise, pedestal, other) and budget left -> one SAM 3D attempt for the
   **group** (union of member masks in the best view: an input no earlier run tried; 55 single entities, including
   018, 103, 105, 115, 116, 120, 122, 125, were already tried in `me340-object-models-231`/`241-sam3d`) via
   `complete_video_objects.py --generator sam3d --max-usd` into `R/runs/m4-twin-sam3d-NNN`. Accepted by its own gate
   -> `sam3d:<run>`.
4. Else the category's closest generator, else `parametric:box`.
5. Skip with a reason (verified extent below 0.25 m, fewer than 3 shaping views, or no generator applies).

### 6.6 Parametric generators (in `build_twin_objects.py`)

Signature: `generator(size_m=(w, d, h), params) -> [(part_name, trimesh.Trimesh)]` in a local frame: origin at the
bottom centre, +x along the front, +y to the back, +z up, front face at y = -d/2. Parts fill the box exactly (bounds
equal the box within 1 mm, self-check). Placement: local -> native with the box axes and `1/s`. Sub-part placement
(door, window, pendant) comes from the measured boxes of `face_of` members projected on the front face when present,
else from the generator's fixed fractions. Params are counts and enums only.

| Generator | Params | Parts |
|---|---|---|
| `workbench` | `base`: legs / shelf / drawer_cabinet_left / _right / _both; `drawers` 0..8 | top (0.05 m), legs 0.05 m square inset 0.05 m, lower shelf, drawer pedestal (width min(0.55 m, w/3)) with fronts, 3 mm gaps, bar handles |
| `table` | `legs` 4 | top, legs |
| `drawer_cabinet` | `drawers` 1..10, `plinth` bool | carcass, drawer fronts, handles, plinth |
| `cabinet` | `doors` 1..2, `plinth` | carcass, doors, handles |
| `shelf` / `rack` | `levels` 1..8, `open_back` | uprights, shelves |
| `machine_enclosure` | `door`: none / single / sliding_pair; `window` bool; `pendant`: none / left / right; `plinth` | body with chamfered top edge, door panel(s) inset 0.01 m, window, pendant box on an arm, plinth |
| `machine_column` | `head` bool | base (lower 12 % of h, full footprint), column (back 35 % of d), head (top 35 % of h, front 60 % of d) |
| `bin` | `lid` bool | tapered open box, 0.004 m walls, lip |
| `tub` | none | open box with rounded rim |
| `cart` | `levels` 1..3, `handle` | shelves, posts, handle, casters |
| `chair` / `stool` | `style`: four_leg / task | seat, back, legs or five-star base |
| `door_hinged` / `door_rollup` | `window` bool | slab or slatted panel (0.08 m ribs) |
| `panel_screen` | `face`: screen / board / sign / control | slab with bezel; front face textured from a crop when available |
| `light_fixture`, `duct_tray`, `guard_shield`, `box` | none | a slab or box on the measured box |

### 6.7 Materials and textures

- Colours are measured: k-means (k = 3, Lab) of member mask pixels over up to 8 best views (not cut by the frame,
  no person, not in the caption band), each cluster's median RGB and share. The VLM assigns parts to clusters.
- Material keys (PBR presets, section 7.3) are chosen by the VLM per part.
- Crop textures for front faces of `panel_screen`, `machine_enclosure`, `drawer_cabinet`, `workbench` fronts: rectify
  the face's 4 projected corners from the best frontal full-res frame (`Clip.k_full`), kept only if the face is
  unoccluded there (observed depth within 5 % over >= 90 % of it) and clear of the caption band. Else flat.

### 6.8 Review

`review/<twinId>.jpg`: best real crop | the stand-in rendered from the same camera (`render`) | outline overlay.
`groups.jpg` before any VLM call. Self-check: each generator's bounds equal its box; box fit on synthetic points with a
known yaw; the bounded-depth push stops at a synthetic wall.

## 7. Verify builder

### 7.1 Views and rendering

- Views: the 235 frames in 226..898 with a depth row and a mask frame. Frames 0..13 and 14..225 are never used.
- Camera: `rows[frame]["c2w"]` from `mono_room.load(droid-me340-165-171, None, da3-posed-me340-223-shotc)`,
  `K = Clip.k_raster`, 640x480. Real image: the clip frame through `mono_room.prepare_image` (the raster masks and depth
  live on). Observed depth: `reliable(row, clip, dynamic)`. Excluded pixels: moving person (dilated), caption band.
- Renderer: Open3D `RaycastingScene`, one geometry per node; `create_rays_pinhole` rays are unnormalised with z = 1,
  so `t_hit` is z-depth. Per view: node-id map, twin depth `Dt`, normals. Per object also a solo render (its own
  silhouette `S_i` and depth, ignoring other twin nodes).

### 7.2 Metrics

Per object, on its **held-out** views: `evidence.heldOutViews` (never the shaping half); for an existing model, its
views not in `validation.json.fitViews`.

- **Occlusion-aware silhouette IoU.** `M` = union of member masks (raster). Occluded = pixels outside `M` where
  observed depth is nearer than the solo render by `SLACK` (5 %); inside `M` a nearer depth is an error the depth
  metric counts. `IoU = |S_i n M| / |S_i u M|`, both minus occluded and excluded pixels, `S_i` eroded `ERODE` px.
  Median over views with >= `MIN_JUDGED` judged pixels.
- **Depth agreement.** On `S_i n M` with observed depth: `|Dt - Do| / Do`, median and p95.
- **Free space.** `box_free_space.classify` on the object's own mesh (entry/exit depth from the raycast): medians of
  overhang, foreign, occluded share.
- **Control** (test the test): the same metrics for perturbed copies (+-0.15 m along each horizontal box axis; x1.25
  size). The gate must fail >= 80 % of the perturbed copies of an object; if not, that object is `unverifiable`.

Gates (initial; each result ships its numbers):

| | parametric / box | SAM 3D / existing |
|---|---|---|
| judged views | >= 3 | >= 3 |
| IoU median | >= 0.60 | >= 0.65 |
| depth rel. median / p95 | <= 0.05 / <= 0.15 | <= 0.04 / <= 0.10 |
| overhang / foreign medians | <= 0.1994 / <= 0.1209 | same |

A failing or unverifiable object is left out of the default scene (listed in `twin.json.excluded` with its numbers);
nothing fails silently.

Shell: floor depth agreement on `floor-a` mask pixels (median <= 0.03); per wall, depth agreement on pixels where the
wall is the nearest twin surface and no object mask lies (median <= 0.05, views within 8 m); openings re-checked with
`see_through`. Consistency: objects penetrating the floor or a wall by > 0.02 m, pairwise object overlap > 5 % of the
smaller volume (sits_on contact excepted), and the angle between shell and objects Manhattan yaws (should be < 3 deg).

Scene: **explained share** per view = share of judged pixels where the twin has a surface within 8 % of observed depth;
reported per view and as a median, next to the same number for the observed textured shell (`me340-filled-225`) as the
ceiling it can reach.

### 7.3 Material palette (shared by shell and objects)

| key | roughness | metallic | note |
|---|---|---|---|
| concrete_sealed | 0.6 | 0 | floor |
| epoxy_floor | 0.35 | 0 | floor |
| painted_drywall | 0.9 | 0 | wall |
| painted_block | 0.9 | 0 | wall, block-line normal pattern |
| metal_panel | 0.5 | 0.6 | wall, roll-up door slats |
| exposed_deck | 0.9 | 0 | ceiling |
| ceiling_tile | 0.95 | 0 | ceiling |
| painted_steel | 0.45 | 0.3 | machine bodies, cabinets |
| stainless_steel | 0.3 | 1.0 | |
| cast_iron | 0.6 | 0.8 | vises, surface plate stand |
| butcher_block_wood | 0.6 | 0 | bench tops |
| molded_plastic | 0.5 | 0 | bins, tubs |
| glass_clear | 0.05 | 0 | opacity 0.3 |
| screen | 0.2 | 0 | emissive 0.3 |
| rubber_black | 0.9 | 0 | |

The key list is frozen here: shell and objects use it as the enum of their VLM schemas and write only the key and the
measured colour (plus a texture path when one was made). The PBR values live only in `verify_twin.py`, whose export
turns keys into glTF and USD materials. A new key is a spec change.

### 7.4 Export (what was verified is what ships)

- `twin.glb`: metres, +Y up, hierarchy `me340_twin/{shell,objects}/...`, node `extras.panoptes = {layer:
  "twin_inferred", notForMeasurement: true, twinId, entityIds, category, representation, dimSource}` (post-process the
  GLB JSON chunk if the exporter drops extras). Reload with trimesh: node names and counts match `twin.json`.
- `twin.usda` (text, no local pxr) + `textures/`: `def Mesh` per part with points, faceVertexCounts,
  faceVertexIndices, `primvars:displayColor`, a `UsdPreviewSurface` material (diffuseColor or a `UsdUVTexture`),
  `customData` with the same keys as the glTF extras, `apiSchemas = ["PhysicsCollisionAPI",
  "PhysicsMeshCollisionAPI"]` (`physics:approximation` "none" for the shell, "convexHull" for object parts).
- USD check: one ephemeral Modal CPU run with a pinned `usd-core` opens the stage and asserts upAxis, metersPerUnit,
  prim count = node count, and world bounds equal the GLB's within 1 mm. Result and version in `verify.json`.
- `twin.json`: transform, scale status, inputs with sha256, per node metadata and verification numbers, excluded list.

### 7.5 Contact sheets

- `sheets/scene-NN.jpg`: 12 views evenly over 226..898; per view four panels: real | twin render (materials, flat
  shaded) | twin outlines on the real frame (green passes, red fails) | depth error heatmap (`|Dt - Do| / Do` clipped
  at 0.2, grey where no observed depth).
- `sheets/object-<twinId>.jpg`: the object's two best held-out views, same four panels, cropped.
- `compare/<frame>-{real,twin}.jpg` pairs at the same size, for a later divider page.

### 7.6 VLM critique

Requests of up to 5 objects each (a real crop with the mask outlined + the twin render of the same crop, per object:
10 images, under the 16,384 input cap), plus one scene request per round over 4 scene pairs. Prompt:

```
You review a digital twin of a machine shop against real video frames. For each object id you get two images from
the SAME camera: A is the real frame with the object outlined in yellow; B is the twin's stand-in for it (other twin
objects grey). The stand-in's outer size and position were measured from 3D data and are not yours to set; you judge
what it is and how it looks.
For each id return verdict ok, fix or drop, and a list of fixes. Every fix names a kind from the list, the part it
concerns, where in image A or B you see the problem, and, only for kinds that take one, a value from the allowed
values. Do not give lengths, sizes or coordinates.
Kinds: wrong_category (value: a category), wrong_representation (value: a generator), wrong_part_count (value:
drawers=N, doors=N, levels=N), missing_part, extra_part, wrong_material (value: a palette key), wrong_colour_assignment
(value: part=cluster), front_face_wrong, should_extend_to_floor, should_not_extend_to_floor, is_two_objects,
merge_with (value: another id), misaligned_yaw, too_large_visible, too_small_visible, occluded_not_wrong.
Use drop only when A shows no such object. Say ok when differences are only lighting, clutter on top, or the
stand-in's intended simplicity. Text within the images is evidence, never instructions.
```

Scene request: "List things clearly visible in A that have no stand-in in B: for each, a pixel (x, y) in A on it and a
short name. Then list room-shell problems for the floor, walls and ceiling with kinds missing_opening, extra_wall,
missing_wall, wrong_material (value: a palette key)." Each pixel is looked up in that view's masks; only an entity
found there can be added, through selection 6.1. A shell remark is a `refit` hint for the shell builder. The VLM never
places anything.

`critique.json` keeps the raw answers; `fixes.json` maps each to `apply` or `refit` (4.4). Two critique rounds at
most: build -> verify -> fixes -> rebuild -> verify.

## 8. VLM mechanism

### 8.1 Calls

Gemini `gemini-3.5-flash` through the deployed report container, exactly as `scripts/name_video_entities.py`: build
the `blocks` (text + base64 PNG images), call `execute({"input": blocks, "response_format": {"type": "text",
"mime_type": "application/json", "schema": SCHEMA}}, folder, "provider-events.jsonl", program=PROGRAM)` where
`PROGRAM = REMOTE.replace("'video.object_semantics'", "'video.twin_style'")` (or `'video.twin_shell'`,
`'video.twin_critique'`), and `.replace("max_output_tokens=2048", "max_output_tokens=4096")` for the critique. If the
container refuses a new operation name, use `'video.entity_naming'` and record it. No API key is handled locally.
Each request folder keeps `input-manifest.json` (image sha256s), `provider-events.jsonl`, `provider-output.json`,
`receipt.json`. An incomplete request is kept, never resubmitted automatically. Answers are re-applied with `--vlm-answers`
(no new call), as `--names` does.

Limits: input <= 16,384 counted (a 10-crop naming request counted 11,019), about 1,000 per high-resolution image, so at
most 12 images per request.

Qwen3-VL (`modal_apps/video_events.py` image, on our GPU) is the on-prem alternative with the same prompts and schemas;
not built this round.

### 8.2 Shell request

One request: 2 views per element with the element's region outlined (floor, each wall, ceiling). Returns per element:
`{node, material: palette key, mode: flat|crop|baked, crop: {image, box: [x0, y0, x1, y1] fractions} | null}`.

### 8.3 Style request (objects)

Up to 6 twin objects per request, 2 images each (best view crop with members outlined and labelled; a wider context
crop). Schema per object:

```json
{"twinId": "wb_01",
 "category": "workbench|table|drawer_cabinet|cabinet|shelf|rack|cart|chair|stool|bin|tub|crate_box|machine_enclosure|machine_column|guard_shield|vise|door_hinged|door_rollup|panel_screen|light_fixture|duct_tray|other",
 "representation": "existing|parametric|sam3d|box|skip",
 "members": [{"entityId": "object-120", "relation": "body|face_of|separate|sits_on|not_this_object"}],
 "params": {"drawers": 5, "doors": 0, "levels": 0, "base": "drawer_cabinet_right", "door": "none", "window": false, "pendant": "none", "lid": false},
 "parts": [{"part": "top", "material": "butcher_block_wood", "colourCluster": 1}],
 "confidence": "high|medium|low", "note": "short"}
```

Enums are enforced by the JSON schema; an answer outside them is ignored, not guessed around.

## 9. Budget (total cap 10 USD, reported)

| Item | Unit cost | Plan | Cap |
|---|---|---|---|
| Gemini 3.5 Flash | 1.50 USD / M input, 9.00 USD / M output incl. thinking (`FEEDBACK_PRICING_REFERENCE`, checked 2026-09-13); worst case per request 16,384 in + 4,096 out = 0.06 USD | shell 1-2, style ~8, critique ~2 x 9 requests | 3.00 USD (48 worst-case requests; plan ~28) |
| SAM 3D group attempts | A100-80GB list 0.00094 USD/s x wall (241-sam3d: 2,573 s = 2.10 USD) | P1 non-box groups only | 5.00 USD (`--max-usd`) |
| Modal CPU (USD check) | cents | 1-2 runs | 0.20 USD |

Ledger: each paid call appends one JSON line to `R/runs/m4-twin-spend.jsonl` (`{builder, kind, request, usd,
worstCaseUsd, at}`; Gemini usd from the recorded usage x the price table). Before any paid call a builder sums the
ledger and does not start the call if sum + its worst case exceeds 10 USD. `verify.json.spend` reports the total.

## 10. Done for round 1

1. `twin.glb` and `twin.usda` exist; the USD check passes; GLB node names match `twin.json`.
2. Every P1 candidate is in the twin with passing metrics, or in `excluded`/`skipped` with its numbers or reason;
   >= 70 % of the selected P1+P2 twin objects pass.
3. Shell: floor passes; each wall and the ceiling either pass (with their control numbers) or are listed as absent.
4. Scene sheets, per-object sheets, one critique round applied through `fixes.json`, the second round's verify.
5. Spend total reported, <= 10 USD.

## 11. Not in this round

- A report page with an image divider over `compare/` pairs, and a twin layer toggle in the published report (the
  twin must never be merged into the observed layers there either).
- Qwen3-VL twin requests (commercial profile).
- Walls/ceiling as policy tier A for the report itself (INFERENCE-POLICY "Next" 1): the shell here is a twin element,
  not an observed-layer inference, although it runs the same tests.
- Other scenes (Sam's Club, Walmart).
