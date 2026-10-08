# EHS Calibrated Visual-Spatial Eval Design

## Decision

The first EHS proof is one bounded workcell task: four still images of a robot
arm inside one closed safety fence, with a step ladder or portable work platform
nearby. The system must reconstruct a shared local scene, reconcile the movable
object across views, compute its relation to the fence, and distinguish the two
sides of the demo `0.6 m` rule.

The benchmark intentionally separates two claims:

1. **Calibrated geometry:** exact pointmaps, camera poses, masks, and metric
   ground truth prove the Open3D/Shapely/rule path.
2. **Provider inference:** the rendered RGB images go through MapAnything, SAM,
   and Gemini to show whether the current models preserve the same result.

Passing the first layer does not claim visual recognition works. Passing a
provider smoke test without metric ground truth does not claim distance works.

## Scenarios

All scenes use metres, `z` up, a `0.6 m` demo threshold, four calibrated camera
slots, one closed approximately convex fence, and a robot arm as context.

| Case | Movable object | Ground truth | Expected result |
| --- | --- | ---: | --- |
| `ladder_050` | step ladder | about `0.5 m` outside fence | `FAIL` |
| `ladder_070` | step ladder | about `0.7 m` outside fence | `PASS` |
| `platform_inside` | portable work platform | footprint intersects fence interior | `FAIL`, `0.0 m` |
| `fence_occluded` | step ladder | fence visible in only two useful views | `INSUFFICIENT_EVIDENCE` |

The robot arm must become scene evidence but must not become the subject of the
clearance fact. The selected subject must be the ladder or platform.

## Generated eval pack

The pack is generated locally from analytic Open3D meshes. This avoids external
asset licensing and produces exact metric truth without CUDA or a new
dependency. Open3D `RaycastingScene` produces aligned RGB, label masks,
per-pixel world points, valid masks, camera intrinsics, and camera poses.

Each generated case contains:

```text
case_id/
  manifest.json
  scene.ply
  scene.glb
  frame_0001/
    rgb.png
    pts3d.npy
    confidence.npy
    valid_mask.npy
    factory_floor.png
    safety_fence.png
    industrial_robot_arm.png
    step_ladder.png | portable_work_platform.png
  ... frame_0004/
```

`scene.ply` is the calibrated mesh used for exact Open3D verification.
`scene.glb` contains the same vertices, faces, normals, and colors for Gradio
6.20 browser display. Gradio's PLY path is not used because its loader rejects
triangle-face `property list` data. Open3D's Assimp reader does not round-trip
its data-URI GLB, so the generator validates the GLB container and a real Gradio
browser test validates rendering. The production MapAnything GLB is unchanged.

Generated binary artifacts live under ignored `outputs/`. Source code,
manifests-as-schema, documentation, and tests are committed.

## Benchmark paths

### Offline calibrated benchmark

The loader converts the generated pointmaps and oracle masks into the existing
`GeometryFrame` and `Observation2D` contracts, then calls the existing
`build_scene_and_assess`. It writes top-down evidence and a JSON report.

Hard gates:

- exact expected status for all four cases;
- ladder `0.5 m` and `0.7 m` remain on opposite sides of `0.6 m`;
- reported external clearance is within `0.1 m` of mesh ground truth;
- one fence entity has at least three evidence views in sufficient cases;
- one ladder/platform entity has at least two evidence views;
- the robot arm is present but never selected as the rule subject;
- the PLY reads back with non-empty colored geometry and the GLB renders in Gradio.

### Opt-in live provider benchmark

The same four RGB sets are passed unchanged to
`EHSAssessmentPipeline.run_assessment`. No provider fallback or oracle mask is
used. The report records provider observations, reconciled entities, status,
distance error, warnings, and artifact paths. One `ladder_050` chat question
checks that Gemini cites SceneMap fact IDs.

The live hard gate is deliberately coarse: `0.5 m` and `0.7 m` must land on the
correct sides of the threshold. Absolute error is recorded but is not a
measurement-grade acceptance claim.

Live execution is paid and opt-in. One case performs one MapAnything call,
32 SAM calls, and at least one Gemini call. As checked on 2026-07-15, fal lists
SAM 3.1 at `$0.01/request`, so segmentation alone is about `$0.32/case`.
Four cases make four MapAnything calls and 128 SAM calls. Keys are read only
from local environment variables and are never written to reports.

## Provider versions

- Keep the currently advertised Replicate MapAnything version
  `bb68c254...eebff` with the Apache checkpoint.
- Move fal image segmentation from `fal-ai/sam-3/image-rle` to the current
  `fal-ai/sam-3-1/image-rle` endpoint. Its schema still accepts one text prompt
  per request, so the fixed eight-class vocabulary remains 32 calls per case.
- Keep stable GA `gemini-3.5-flash` explanation/chat behavior unchanged.

## Explicit limits

This benchmark covers XY clearance around one closed, approximately convex
fence. It does not validate an open gate, a missing fence panel, a fence gap,
multiple cells, non-convex boundaries, formal EHS standards, CAD comparison,
video, OCR, S3, robotics, or certified measurement. The current convex-hull
geometry would close a real gap, so those claims remain out of scope.
