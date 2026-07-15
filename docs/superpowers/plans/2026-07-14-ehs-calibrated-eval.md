# EHS Calibrated Eval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible CPU-only robot-workcell eval pack and compare its metric truth with both the deterministic spatial pipeline and the opt-in cloud-provider pipeline.

**Architecture:** An Open3D raycaster generates four-view RGB, exact world pointmaps, masks, camera calibration, a colored PLY for metric verification, and a matching GLB for browser display across four fixed EHS cases. A small benchmark module loads those artifacts into the existing contracts and invokes the existing scene/rule pipeline offline or the existing provider pipeline live; it writes evidence and a machine-readable report without changing production APIs.

**Tech Stack:** Python 3.12, NumPy, Pillow, Open3D 0.19, existing Pydantic contracts, pytest, Replicate MapAnything, fal SAM 3.1, Gemini 3.5 Flash.

---

### Task 1: Upgrade the fal endpoint to SAM 3.1

**Files:**
- Modify: `tests/test_sam3.py`
- Modify: `ehs_spatial/providers/sam3.py`
- Modify: `README.md`

- [ ] **Step 1: Change the provider contract test to require the current endpoint**

Change the endpoint assertion to:

```python
assert seen["endpoint"] == "fal-ai/sam-3-1/image-rle"
```

- [ ] **Step 2: Run the focused test and verify RED**

Run: `uv run pytest -q tests/test_sam3.py::test_adapter_normalizes_complete_fal_response_and_resizes_masks_nearest`

Expected: failure showing the adapter still called `fal-ai/sam-3/image-rle`.

- [ ] **Step 3: Make the minimal endpoint and documentation change**

Set:

```python
SAM3_ENDPOINT = "fal-ai/sam-3-1/image-rle"
```

Update the README provider label and link to SAM 3.1. Do not change the eight
prompts or add batching; the official fal schema accepts one text prompt.

- [ ] **Step 4: Verify GREEN**

Run: `uv run pytest -q tests/test_sam3.py`

Expected: all SAM adapter tests pass.

- [ ] **Step 5: Commit**

```bash
git add README.md ehs_spatial/providers/sam3.py tests/test_sam3.py
git commit -m "chore: move EHS segmentation to SAM 3.1"
```

### Task 2: Generate four calibrated EHS scenes

**Files:**
- Create: `ehs_spatial/eval_pack.py`
- Create: `tests/test_eval_pack.py`
- Modify: `.gitignore`

- [ ] **Step 1: Write generator contract tests**

Add tests that call `generate_eval_pack(tmp_path)` and assert:

```python
assert set(manifest["cases"]) == {
    "ladder_050", "ladder_070", "platform_inside", "fence_occluded"
}
assert manifest["cases"]["ladder_050"]["expected_status"] == "FAIL"
assert manifest["cases"]["ladder_070"]["expected_status"] == "PASS"
assert manifest["cases"]["platform_inside"]["expected_distance_m"] == 0.0
assert len(manifest["cases"]["ladder_050"]["frames"]) == 4
```

For every normal frame, load RGB, pointmap, valid mask, and label masks and
assert identical pixel dimensions. Read `scene.ply` with Open3D and assert that
it has vertices, triangles, and vertex colors. Validate the `scene.glb` header,
then browser-test it in Gradio. Assert each of the four normal
views contains visible robot, fence, floor, and movable pixels; for
`fence_occluded`, assert exactly two frames contain a usable fence mask.

- [ ] **Step 2: Run generator tests and verify RED**

Run: `uv run pytest -q tests/test_eval_pack.py -k generate`

Expected: import failure because `ehs_spatial.eval_pack` does not exist.

- [ ] **Step 3: Implement the minimum CPU raycaster**

Implement analytic mesh builders for:

```text
factory floor
closed safety fence
industrial robot arm
step ladder
portable work platform
```

Use `open3d.t.geometry.RaycastingScene` with fixed `512x384` intrinsics and four
camera poses. Derive RGB, `pts3d`, valid mask, label masks, and normalized boxes
from the same `geometry_ids` result. Use simple per-label colors plus normal
shading; do not call a generative image model. Write a colored `scene.ply` for
metric verification and matching `scene.glb` for Gradio. Read the PLY back and
validate the GLB container before declaring generation successful.

For `fence_occluded`, make the final two camera frames point away from the cell
so their RGB and masks genuinely lack fence evidence; do not merely delete an
otherwise-visible mask.

Add `/outputs/` to `.gitignore`. Add a `ponytail:` comment explaining that the
analytic meshes are intentionally domain-simplified and real capture is the
upgrade path.

- [ ] **Step 4: Run generator tests and verify GREEN**

Run: `uv run pytest -q tests/test_eval_pack.py -k generate`

Expected: all generator tests pass without CUDA or provider variables.

- [ ] **Step 5: Commit**

```bash
git add .gitignore ehs_spatial/eval_pack.py tests/test_eval_pack.py
git commit -m "feat: generate calibrated EHS workcell scenes"
```

### Task 3: Run the deterministic spatial benchmark

**Files:**
- Modify: `ehs_spatial/eval_pack.py`
- Modify: `tests/test_eval_pack.py`

- [ ] **Step 1: Write offline benchmark tests**

Generate the pack, call `run_offline_benchmark(pack_root)`, and assert:

```python
assert report["passed"] is True
assert report["cases"]["ladder_050"]["actual_status"] == "FAIL"
assert report["cases"]["ladder_070"]["actual_status"] == "PASS"
assert report["cases"]["platform_inside"]["actual_distance_m"] == 0.0
assert report["cases"]["fence_occluded"]["actual_status"] == "INSUFFICIENT_EVIDENCE"
```

For sufficient cases, assert absolute external distance error is at most
`0.1 m`, exactly one movable entity is reconciled from at least two views, the
robot arm exists, and the clearance fact subject is the movable entity rather
than the robot. Assert `offline_report.json` and one top-down PNG per case exist.

- [ ] **Step 2: Run the benchmark test and verify RED**

Run: `uv run pytest -q tests/test_eval_pack.py -k offline`

Expected: failure because `run_offline_benchmark` is missing.

- [ ] **Step 3: Load generated artifacts into existing contracts**

Build existing `GeometryFrame` and `Observation2D` values directly from each
case manifest and mask. Call the existing `build_scene_and_assess`; do not copy
geometry or rule logic into the benchmark. Save `scene.json`,
`assessment.json`, `topdown.png`, and a pack-level `offline_report.json`.

The report case object contains:

```json
{
  "expected_status": "FAIL",
  "actual_status": "FAIL",
  "expected_distance_m": 0.5,
  "actual_distance_m": 0.503,
  "absolute_error_m": 0.003,
  "entity_labels": ["industrial robot arm", "safety fence", "step ladder"],
  "warnings": [],
  "passed": true
}
```

- [ ] **Step 4: Verify GREEN**

Run: `uv run pytest -q tests/test_eval_pack.py`

Expected: all eval-pack tests pass.

- [ ] **Step 5: Commit**

```bash
git add ehs_spatial/eval_pack.py tests/test_eval_pack.py
git commit -m "test: benchmark EHS spatial facts against metric truth"
```

### Task 4: Add one CLI for generate, offline, and live eval

**Files:**
- Create: `scripts/ehs_eval.py`
- Create: `tests/test_ehs_eval_cli.py`
- Create: `eval/README.md`
- Modify: `README.md`

- [ ] **Step 1: Write CLI behavior tests**

Test `main(["generate", "--output", str(path)])` and
`main(["offline", "--pack", str(path)])`. Capture stdout and assert it prints
the absolute manifest/report path and returns `0` only when the offline report
passes. Test that `live` refuses to run unless `--live` is present and all three
provider variables exist.

- [ ] **Step 2: Run CLI tests and verify RED**

Run: `uv run pytest -q tests/test_ehs_eval_cli.py`

Expected: import failure because `scripts/ehs_eval.py` does not exist.

- [ ] **Step 3: Implement the minimal CLI and live report**

Use `argparse` with three subcommands:

```text
generate --output outputs/ehs_v1
offline --pack outputs/ehs_v1
live --pack outputs/ehs_v1 --case ladder_050 --live
```

Live mode calls the existing `EHSAssessmentPipeline.run_assessment` with the
four generated RGB paths and manifest camera height. It loads the produced
SceneMap, compares status and threshold side with ground truth, and writes
`live_report.json`. When `ladder_050` runs, call
`answer_question(run_id, "Why did this workcell fail the clearance check?")`
and record only the grounded answer and cited fact IDs. Never record keys.

No retries, fallback, queue, database, or benchmark service.

- [ ] **Step 4: Document exact claims and commands**

`eval/README.md` must state that offline mode uses oracle masks/pointmaps and
does not validate MapAnything or SAM. Document the per-case paid call count and
show how to run one case before the full four-case set. Link the generated PLY,
GLB, RGB, top-down, and report paths.

- [ ] **Step 5: Verify GREEN**

Run:

```bash
uv run pytest -q tests/test_ehs_eval_cli.py
uv run python scripts/ehs_eval.py generate --output outputs/ehs_v1
uv run python scripts/ehs_eval.py offline --pack outputs/ehs_v1
```

Expected: tests pass, generation completes on CPU, and offline exits `0` with
all four cases passing.

- [ ] **Step 6: Commit**

```bash
git add README.md eval/README.md scripts/ehs_eval.py tests/test_ehs_eval_cli.py
git commit -m "feat: add reproducible EHS eval command"
```

### Task 5: Add an opt-in paid benchmark test

**Files:**
- Create: `tests/test_ehs_benchmark_live.py`
- Modify: `eval/README.md`

- [ ] **Step 1: Write the opt-in test around the existing live runner**

Mark the module skipped unless `EHS_LIVE_BENCHMARK=1`. Parameterize case IDs
from `EHS_BENCHMARK_CASES`, defaulting to `ladder_050`, so one case can be paid
and debugged before four cases. Assert the report has provider artifacts,
expected status, threshold-side correctness, entity evidence, and grounded chat
fact IDs for `ladder_050`.

- [ ] **Step 2: Verify the default suite skips without cost**

Run: `uv run pytest -q tests/test_ehs_benchmark_live.py`

Expected: skipped; zero provider calls.

- [ ] **Step 3: Document the paid command without embedding credentials**

Document:

```bash
EHS_LIVE_BENCHMARK=1 EHS_BENCHMARK_CASES=ladder_050 \
  uv run --env-file .env pytest -q -s tests/test_ehs_benchmark_live.py
```

The four-case command sets
`EHS_BENCHMARK_CASES=ladder_050,ladder_070,platform_inside,fence_occluded`.

- [ ] **Step 4: Commit**

```bash
git add eval/README.md tests/test_ehs_benchmark_live.py
git commit -m "test: add opt-in paid EHS benchmark"
```

### Task 6: Validate artifacts, browser rendering, and the full repository

**Files:**
- Modify only if verification reveals a scoped defect.

- [ ] **Step 1: Run all offline quality gates**

Run:

```bash
uv run pytest -q
uv run python -m compileall -q app.py ehs_spatial scripts tests
uv lock --check
uv pip check
```

Expected: all tests pass with only opt-in provider tests skipped; all other
commands exit `0`.

- [ ] **Step 2: Inspect generated evidence**

Open one RGB from each case and each top-down image. Confirm that the 0.5/0.7
scenes differ only in ladder placement, the platform is inside the fence, and
the insufficient case lacks fence evidence in exactly two views.

- [ ] **Step 3: Browser-test the calibrated GLB**

Serve one generated `scene.glb` through Gradio `Model3D`, then use a real
browser to verify load, rotate, and zoom. Save a screenshot under `outputs/`.
The PLY remains the metric validation artifact; Gradio 6.20 rejects its triangle
face `property list`, while the generated GLB renders in the same viewer used by
the MapAnything result.

- [ ] **Step 4: Run one paid case only after rotated keys exist locally**

If `.env` contains newly rotated credentials, run `ladder_050` first. Do not use
credentials pasted into chat. If the provider result fails, preserve the report
and raw artifacts and report the exact failing layer; do not add a fallback.

- [ ] **Step 5: Request independent spec and code-quality review**

Review against
`docs/superpowers/specs/2026-07-14-ehs-calibrated-eval-design.md`, then resolve
all important findings and rerun the full offline gates.
