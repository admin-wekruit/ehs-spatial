# L5 — clauses -> RulePack (C4)

`handwritten@2` (handwritten.py): the trial's rules.lp (2026-10-07) as a `RulePack` `handwritten-trial@1`, ignoring L4's clause graph; @2 = the enclosure rule reads the engine's `coverage_known` atom (no coverage → cannot_determine, breach through observed floor → fail).

| rule | spec requirement | threshold | clause (unverified numbers) |
|---|---|---|---|
| `fence_height` | top_height(F) ≥ | 1400 mm | ISO 13857:2019 Table 2 note |
| `floor_gap` (fence, guard) | bottom_height(F) ≤ | 180 mm | ISO 13857:2019 4.4 |
| `lc_lowest_beam` | bottom_height(L) ≤ | 300 mm | ISO 13855 (2010 numbers) |
| `crush_gap` (robot × fixed) | min_distance_3d(R, X) ≥ | 500 mm | ISO 13854:2017 Table 1 / ISO 10218-2 (clause unverified) |
| `reach_over` (robot × fixed) | reach_over(H, S) table | — | ISO 13857:2019 Table 2: status `needs_input`, inputs `table_lookup`, `risk_level` |
| `enclosure` | enclosed(Z) | — | topology on the occupancy grid (outside -> hazard reachability) |

Each `Rule` carries `spec` {selection, applicability, requirement, exceptions, notes} for engine-neutral evaluation and `asp` for clingo;
`common_asp` (`asp.COMMON_ASP`, shared with the synthesis variants below) has the guard-band decision (`status` pass / fail / needs_meas /
cannot_determine, `margin`, inclusive `ge`/`le` and strict `gt`/`lt`) and the `#show`s.
Params: none (the guard-band coverage factor `k` is the engine's). Limits: numbers not verified against the purchased texts; no zones,
no declared inputs consumed; `floor_gap` evaluates `bottom_height` as the trial did (a box sunk below the fitted floor gives a negative value).

## Synthesis variants (first round of the lab, 2026-10-08)

Three L5 plugins compile L4's clause graph (`clauses-v0.json` + `clauses-ts0011963-v0.json`, 38 clauses) against the handwritten control.
Shared code: `asp.py` (COMMON_ASP; `render(rule)`: spec → ASP over clingo@2's generic atoms; `unsupported(rule)`: why not; `rule_id()`: the
clause id as a clingo constant, e.g. `ISO13857:2019/4.4` → `iso13857_2019_4_4`) and `common.py` (clause → Rule skeleton, vocabulary gaps from
the Alignment, the status decision `finish()` = render → `verify.signature_check` → status, the LLM output schema, the stable system prompt,
`synthetic_facts()` = synthetic-facts@1 plus the `perimeter_of(X, hazard_zone)` / `covers_opening(lc, C)` facts the synthetic cell has by
construction, and the model-test runner). Every variant sorts its rules by `rule_id` and keeps the clause id in `Rule.clause`.

| variant | needs LLM | what decides `compiled` |
|---|---|---|
| `function-library@0` | no | `asp.render` succeeds and `signature_check` is clean |
| `code-synthesis@0` | one call per clause | the above, then the model's own tests (python engine; clingo too when the model wrote the ASP), the threshold grid, metamorphic rigid + inflate |
| `redundant-translation@0` | two calls per clause | two differently framed translations agree after normalisation, then the above and an empty differential |

Statuses, all variants: `compiled`; `needs_input` (requirement has a `table`, a `formula` or `inputs` — the ASP is one `needs_input` line, the python
engine reports the undeclared inputs); `vocabulary_gap` (a term of the rule has an Alignment row at 0.0 — none on the merged graph today);
`refused` (not photo-checkable, semantic clause without a bool predicate, unrenderable: `exists` / `not_exists`, `== 0`, attribute requirements,
comparisons inside applicability such as `floor_gap(F) > 35`, three selection variables, a zone variable over a kind other than `hazard_zone`;
plus, for the LLM variants, a model refusal or a failed check — `unsupported_reason` names it). What `render` supports: 1–2 selection
variables over classes (one `subject/2` rule per class combination, `A != B` when the lists overlap, as the python engine never binds two
variables to one object); a zone variable only as the implicit hazard zone (rendered as the id `"hazard_zone"` guarded by `hazard(_)` — zone
kinds are not facts yet, a declared zone of another kind is unsupported); applicability atoms verbatim, unbound capitalised arguments as `_`
(they join the subject tuple when the requirement uses them); exceptions as `not attr(X)`; numeric `>= <= > <` → `thr/dir/subject/meas` as
handwritten; `== 1` on a bool predicate → pass when the atom holds, cannot_determine otherwise. The subject is the selection tuple, so verdict
subjects match the python engine's (e.g. `(F, "hazard_zone")` for ISO 13857 4.4).

### `function-library@0` (function_library.py)
FuncMapper pattern (the typed scene API is the Signature; the clause's structured `requirement` is the function call): no LLM, the spec is the
clause's `selection` / `applicability` / `requirement` / `exceptions` as L4 wrote them (tables keep their id and operator, inputs stay declared).
cfg: `all` (every clause instead of the retrieved ones). Merged graph: 11 compiled, 9 needs_input, 18 refused, 0 vocabulary_gap. Parity
with handwritten@2 (`tests/verdict/test_rules_funclib.py`): the floor-gap (ISO 13857 4.4), fence-height (Table 2 min height) and light-curtain
(TS 9.1.2) rules give handwritten's statuses on six `scenes.cell()` variants under both engines once the cell's `perimeter_of` /
`covers_opening` facts are present; the crush-gap clause (ISO 10218-2 trapping clearance) binds the same (robot, fixed) pairs minus the
light curtain (the clause lists `control_panel` instead) and answers NEEDS_INPUT because it declares `restricted_space` (lab rule 4).
Limits: the clause's prose notes (e.g. TS 9.1.7 "only for orientation=horizontal") are not applicability atoms, so the rule fires on every
light curtain; rules whose applicability needs `perimeter_of` / `covers_opening` bind nothing on the benchmark scenes until L2 computes them.

### `code-synthesis@0` (code_synthesis.py)
TUM ACC / CodeAct pattern (the model writes the check, tests decide): per clause one `llm.complete` with the stable system prompt (spec
convention, Signature vocabulary, `scenes.cell` parameters, one worked example — cached on the API side) and the clause as user prompt;
structured output `common.Synthesis` = spec, optional asp, tests (`cell_params` as `{name, value}` pairs — structured outputs allow no free-form
dicts — plus the expected status), needs, refuse_reason. Checks in order: `finish` → tests → `verify.threshold_grid` (compiled rules with a
synthetic knob) → `verify.metamorphic` rigid + inflate on `scenes.cell()`. The first failure refuses the rule, the model output stays in
`rule.review["llm"]`. cfg: `model` (default `llm.MODEL`, Haiku 5.5), `effort`, `cache_dir` (default `<runs>/llm-cache`), `all`. `llm_calls`
= uncached calls (the ledger sums them). Limits: the grid and the metamorphic relations run the python engine on the spec, so they can only
catch an engine/spec mismatch; a wrong threshold is caught only when the model's tests are right.

### `redundant-translation@0` (redundant.py)
ARc-style redundancy + differential execution: per clause two calls, framing A = structured requirement + paraphrase, framing B = clause
number, title, definitions and tags only (reconstruct the requirement from the words); no sampling parameters (Haiku 5.5 rejects non-default
temperature). Both specs are normalised (sorted class lists, canonical operator, float threshold, sorted inputs incl. `needs`) and compared field
by field; a disagreement refuses (`redundant translations disagree: <fields>`, both candidates in `review`); agreement → `finish` →
`verify.differential` of the two candidate packs on four cells (nominal, open, floor gap 0.2, fence 1.2) under the python engine must be empty.
Same cfg keys; `llm_calls` counts both calls. Limit: agreement is agreement with each other, not with the standard.

### First live runs (not possible on this machine: no credentials)
```
export ANTHROPIC_API_KEY=...      # or `ant auth login`; never written to a file
PANOPTES_WORKCELL=$PWD .venv/bin/python -m ehs_spatial.cli verdict run --config ehs_spatial/verdict/configs/codegen-ts.yaml   --item 090 --runs-dir runs --run-id codegen-ts-090
PANOPTES_WORKCELL=$PWD .venv/bin/python -m ehs_spatial.cli verdict run --config ehs_spatial/verdict/configs/redundant-ts.yaml --item 090 --runs-dir runs --run-id redundant-ts-090
```
Expected: 24 retrieved clauses → 24 calls (codegen) / 48 calls (redundant) on the first run, 0 afterwards (responses cached under
`runs/llm-cache/`, so `PANOPTES_FAKE_MODEL=1` re-runs are free); the ledger line carries `llm_calls`. Without an LLM:
`PANOPTES_FAKE_MODEL=1 ... --config ehs_spatial/verdict/configs/funclib-ts.yaml --item 090` (run `wp3-funclib-090`: 8 compiled, 9 needs_input,
7 refused; 24 verdicts, 22 NEEDS_INPUT + 2 FAIL; 9 live rules bound nothing for lack of `perimeter_of` / `covers_opening` facts).
