# WS-B1 — OSHA 1910 policy-corpus and compiler exam

Date: 2026-08-25. Scope: feed the policy compiler real regulation text and
prove the pipeline refuses what the closed predicate vocabulary cannot
express. No behavior changes to `ehs_spatial/policy.py` or
`scripts/policy_compile.py`.

## What landed

- `scripts/oshacorpus.py` — fetches 9 sections of 29 CFR 1910 from the eCFR
  API (curl, matching the nvwarehouse workaround for this env's broken
  urllib SSL), caches raw XML under `tests/fixtures/oshacorpus/`, extracts
  per-paragraph text with citation ids via the fixed CFR marker ladder
  `(a)(1)(i)(A)(1)(i)`, and classifies each paragraph. Offline by default;
  `--fetch` refreshes the cache (pinned to eCFR date 2026-08-01).
- `tests/fixtures/oshacorpus/corpus.json` — 505 paragraphs across
  1910.36/.37/.157/.159/.176/.212/.253/.303/.333:
  **3 compile / 10 refuse / 492 skip** (skip = not a spatial requirement).
- `docs/policies/osha1910.md` — 13-line exam sheet in the exact format
  `policy_compile.py --policies … --live` reads; exam line N corresponds to
  the Nth non-skip corpus row, so the orchestrator can diff live compiler
  output against the expected columns.
- 10 new tests in `tests/test_policy.py` (existing assertions untouched).

## The three designed refusals (must set unsupported_reason)

1. **1910.176(a)** — "…sufficient safe clearances shall be allowed for
   aisles, at loading docks, through doorways…" — no number exists; any
   threshold would be invented.
2. **1910.333(c)(3)(ii)** — "…may not approach or take any conductive
   object … closer to exposed energized parts than shown in Table S-5…" —
   threshold keyed to circuit voltage, a non-spatial variable the scene
   cannot measure.
3. **1910.157(d)(2)** — "…travel distance for employees to any
   extinguisher is 75 feet (22.9 m) or less." — walking-path distance;
   Euclidean max_separation would systematically understate it.

## Refusal taxonomy the corpus surfaced (vocabulary gaps, not bugs)

- **No minimum-height/headroom predicate.** 1910.36(g)(1) exit-route
  ceiling ≥ 2.3 m; also 1910.303(h)(2)(ii) fence ≥ 2.13 m and
  1910.303(g)(2)(i)(D) live parts elevated ≥ 2.44 m. Vocabulary has
  `max_height` only. A `min_height` twin would convert several real rules.
- **Path/travel distance.** All of 1910.157(d).
- **Conditional thresholds.** Voltage tables (1910.333 Table S-5, 1910.303
  Table S-1) and equipment-relative dimensions (1910.303(g)(1)(i)(B):
  "762 mm or the width of the equipment, whichever is greater").
- **Alternative-compliance disjunction.** 1910.253(b)(4)(iii): 20 ft
  separation OR a rated barrier — compiling only the distance arm would
  raise false violations wherever a compliant barrier exists.
- **Vertical vs plan-space separation.** 1910.159(c)(10): 18-inch
  clearance between sprinklers *above* and material *below*. The
  vocabulary could name it min_separation, but `evaluate_policy` measures
  XY footprint gaps, so an object under a sprinkler reads gap ≈ 0 and
  always fails. Classified refuse; an honest fix needs either a
  `vertical_clearance` predicate or 3D-aware separation (out of scope
  here).
- **Width of free passage.** 1910.36(g)(2): 28-inch exit-access width is a
  property of empty space between unspecified obstructions, not a labeled
  subject-object pair.

## Positives proven end-to-end (corpus numbers run through the evaluator)

- 1910.253(b)(2)(ii) → `min_separation` 6.1 m (cylinders vs highly
  combustible materials).
- 1910.253(f)(5)(i)(B) → `min_separation` 3.0 m (portable acetylene
  generators vs combustible material).
- 1910.303(h)(3) → `keep_clear` 0.914 m (clear work space about over-600V
  equipment).

## Notes for the orchestrator's --live run

- `policy_compile.py` caches compiled specs as
  `outputs/policies/compiled/p{NN}.json`, keyed **by line index only**, not
  by policy file. The starter.md cache (p01–p09 in the main repo) would be
  served for the first nine OSHA exam lines. Run the exam with a clean
  `outputs/policies/compiled/` (or from a fresh worktree). This is a
  pre-existing compiler behavior; per workstream rules it was documented,
  not changed.
- No other compiler change is required: `PolicySpec.unsupported_reason`
  plus the prompt's "do not approximate" instruction already cover every
  refusal class above — the exam exists to check the model actually obeys.
