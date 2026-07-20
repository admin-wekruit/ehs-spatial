# Independent spec/code review — 2026-07-20

Scope: calibrated-eval work on `feature/ehs-spatial-mvp` @ `b94e6d9`, reviewed against
`docs/superpowers/specs/2026-07-14-ehs-calibrated-eval-design.md`.

Method: 6 independent fresh-eyes reviewers (rules, geometry, providers, eval-pack,
security/path, spec-conformance), then every finding adversarially verified by 2
independent refuters instructed to disprove it (default = refuted). 22 agents total.
Result: **7 confirmed (2-of-2 votes), 1 plausible (1-of-2), 0 findings survived
un-refuted that were wrong**. Spec-conformance reviewer found zero deviations: every
spec hard gate is enforced by `run_offline_benchmark` or tests, call-count claims
(1 MapAnything + 32 SAM + ≥1 Gemini/case) match the pipeline loop, provider pins match
code constants. Two confirmations were reproduced by execution on this machine, not
just code reading.

The two SAM findings share one root cause and are merged below → 6 unique issues.

## Fixed in this session (with regression tests; suite now 140 passed / 2 skipped)

### F1. SAM 3.1 instance-id collision silently overwrote masks — FIXED
`ehs_spatial/providers/sam3.py` derived `instance_id` from provider metadata `index`
with an ordinal fallback and never checked uniqueness. Colliding ids (duplicate
provider indices, or partial metadata making the fallback collide with an explicit
index) made two observations share one `mask_path`; the second `Image.save` silently
overwrote the first mask, corrupting footprints feeding the 0.6 m verdict. Both
refuters reproduced end-to-end (one by executing `segment()` with a fake subscriber).
Fix: duplicate id now raises `ProviderError("fal", "sam3.normalize", ...)`, consistent
with every other malformed-response path. Test:
`tests/test_sam3.py::test_adapter_rejects_colliding_instance_ids`.

### F2. Stale `live_report.json` survived unwrapped post-spend crashes — FIXED
`run_live_benchmark_case` catches only `ProviderError`; a non-ProviderError after paid
calls (e.g. `OSError` from `np.save` — `map_anything.py:210` wraps only
KeyError/TypeError/ValueError — or the run_id-mismatch `ValueError`) wrote no report,
leaving the previous invocation's possibly-`"passed": true` report as current. Fix:
the runner now unlinks any existing `live_report.json` after validation, before the
paid chain. Test:
`tests/test_ehs_eval_cli.py::test_live_runner_clears_stale_report_before_unwrapped_crash`.

### F3. Displayed clearance rounded across the verdict threshold — FIXED
`app.py` verdict card and `topdown.py` stamp used `:.1f`, so a failing 0.55–0.59 m
clearance displayed as "0.6 m" beside a FAIL verdict (indistinguishable from a passing
0.60 m). Fix: both display two decimals. Updated assertion in `tests/test_app.py`.

## Confirmed, open — owner decision needed (each has a real failure path)

### O1 (major). Floor RANSAC is nondeterministic despite seeding — scale jitters
`geometry.py:129`: `o3d.utility.random.seed(0)` does not make Open3D 0.19
`segment_plane` deterministic (OpenMP-parallel RANSAC race). Both refuters reproduced
on this machine with the repo venv: identical noisy input → 2–3 distinct
`scale_factor`s across back-to-back runs (~0.2–0.33% spread); `OMP_NUM_THREADS=1`
makes it fully deterministic. Oracle eval masks this (exactly planar floors); real
MapAnything captures will not. Near-threshold real scenes can flip PASS/FAIL between
runs on identical input, and the "reproducible eval" claim does not hold for live.
Options: (a) pin `OMP_NUM_THREADS=1` at process entry (simple; slows all Open3D ops),
(b) replace with a small deterministic NumPy RANSAC (~15 lines, vectorized; changes
offline numbers within tolerance), (c) document as known limitation for MVP.
Recommendation: (b) after the first live case, so the live run stays comparable to
the current baseline.

### O2 (major). Fence shatter → partial footprint → silent false PASS — FIXED (minimal) 2026-07-20
Minimal fix landed: `rules.py` counts fence-labeled entities discarded by the
evidence gates and, whenever a verdict is still produced, emits
"N safety fence fragment(s) discarded by evidence gates; clearance may be
measured against a partial fence". Tests:
`tests/test_rules.py::test_discarded_fence_fragments_surface_a_warning_instead_of_vanishing`
(+ clean-fence no-warning case). Live-run relevance confirmed empirically (30
fragments on the real reconstruction). The larger same-label cluster-merge
remains a future option. Original finding kept below for the record.

Original finding:
`geometry.py:235`: `cluster_dbscan(eps=0.15)` splits a fence with any >0.15 m
reconstructed point gap into fragments; `rules.py` silently drops fragments failing
its gates (a thin far-rail hull ~0.12 m² < 0.25 m²), and with exactly one surviving
fragment the exactly-one-fence guard does not trip. Clearance is then measured to the
partial hull: a movable 0.3 m from the dropped rail can report PASS with empty
warnings, and `topdown.py` never draws the discarded fragment. Directly relevant to
the fence-recall risk: on real imagery, sparse fence reconstruction makes this the
false-negative mode the product must not have. Minimal fix: emit a warning (and/or
demote to INSUFFICIENT_EVIDENCE) whenever fence-labeled entities are discarded while
one passes the gates; larger fix: merge same-label clusters before gating.
Recommendation: minimal warning fix before the first live case is *interpreted*;
verify offline gates don't assert empty warnings first.

### O3 (minor, security). Provider response can direct reads of arbitrary local files
`map_anything.py:109` `_read_provider_bytes` treats any response string as a candidate
local path and follows `file://` (plus arbitrary http/https → SSRF). A malicious or
compromised model output (pinned id is a community `vufinder/...` wrapper, not an
official account) can copy e.g. `~/.ssh/id_rsa` bytes into `geometry/point_cloud.glb`
— and GLB evidence artifacts have been committed before (`ef2f5ec`). Both refuters
reproduced the read locally. Fix: allowlist https (or known delivery hosts); keep the
bare-local-path branch only behind the injectable runner used by tests.
Recommendation: fix before any live run whose artifacts might be committed/shared.

### O4 (plausible, minor). `Criterion.object_label` is dead configuration
`rules.py:51` filters on hardcoded `FENCE_LABEL`; `criterion.object_label` is never
read, yet the full criterion (including the ignored label) is forwarded into the
Gemini grounding payload. Unreachable from the app/eval (always default), so one
refuter called it unused generality, not a defect. Cheapest honest fix: drop the field
or add a validator pinning it to "safety fence" until the rule engine generalizes.

## Known boundaries recorded (not defects, do not "fix")

- Exact 0.6 m clearance → PASS (`distance < minimum` fails); demo-rule edge, accepted.
- Offline eval never tests the threshold boundary (cases sit ~0.1 m away); accepted.
- fal fence recall 0 on synthetic smoke is a model-quality risk, not a code bug —
  tracked via the fence-recall eval below, not via prompt/rule tweaks.

## Fence recall eval — design (build-gated on live-run fence check failing)

Purpose: measure SAM 3.1 fence detection on real, licensed, multi-angle imagery —
NOT tuned to the synthetic yellow fence.

- **Sourcing:** 10–20 physical fences/guardrails × 2–4 angles each. Candidates:
  Open Images V7 "Fence" class (CC-BY images, Apache annotations), Roboflow Universe
  industrial fence/guardrail sets (license checked per set before use), hand-curated
  CC-BY industrial photos. Prefer factory/workcell-like scenes; record per-image
  license + source URL in the manifest.
- **Format:** `outputs/fence_recall_v1/manifest.json` + per-fence case dirs with
  `frame_XX_rgb.png` and optional hand-traced oracle masks, mirroring `eval_pack.py`
  manifest/report conventions so `scripts/ehs_eval.py` grows at most one subcommand.
- **Metric:** per-view bbox-level detection recall (SAM instance present ∩ oracle
  bbox) as the primary number; mask IoU only where oracle masks exist. Report
  per-view and per-fence; the product-relevant number is "fraction of fences detected
  in ≥3 of N views" (the rule's evidence gate).
- **Cost:** 1 fal call per view per prompt; fence prompt only → ~$0.01/view,
  ≈ $0.40–0.80 for the full set.

## Verification

- `uv run pytest -q` → 140 passed, 2 skipped.
- `uv run python -m compileall -q app.py ehs_spatial scripts tests` → OK.
- `uv lock --check`, `uv pip check` → OK.
