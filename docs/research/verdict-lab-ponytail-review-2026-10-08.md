# ponytail-review of the verdict lab (2026-10-08, tree 616f8e5) — over-engineering only

Run by a read-only review agent on `ehs_spatial/verdict` + `tests/verdict` (5,485 lines) with the lab's own rules in hand (README rules 1–9 are
not flagged). Kept here so the cuts are applied deliberately, not forgotten. Status column: **todo** = not applied yet; **policy** = the owner
decides (the finding names a trade-off).

## Whole-module cuts (biggest first)

| Finding | Status |
|---|---|
| `layers/l4_spec/tables.py:L1-131`: delete: eight ISO/TS lookup functions no engine calls (python.py refuses tables by design, asp.py renders needs_input, `VERIFIED=False`); only the table tests touch them. Table ids stay as JSON data; `contracts.Table.function/inputs/output_unit` go with it; re-add when a purchased text makes `verified` true. | policy (keep until a verified text; they document the numbers we must verify) |
| `synth/facts.py:L1-182`: native: a second Scene→Facts beside relations@2, kept because L5 may not import L2 — but `common.engine()` already reaches L6 through the registry. Use `plugins.get("L2","relations")` in `common.synthetic_facts`; port the 3-line grid-origin snap (facts.py L89-92) into relations (relations@3), or keep the file and drop the parity claim (`facts_parity` is never run against relations). | policy |
| `layers/l4_spec/extract_cli.py:L1-68` (+ lab/cli.py extract wiring): delete: second entry point for one plugin; `panoptes verdict run --config configs/llm-extract-ts.yaml` already writes L4/diff.json + extraction_report.json. | todo |
| `layers/l1_scene/measurement_layer.py:L1-53`: delete: no config, test or import references it; README.md L41 wrongly calls it the baseline L1 (baseline.yaml uses scene-json). | todo |
| `synth/pack.py:L1-50`: delete: the same six trial rules as `handwritten.rules()` minus ASP; `Handwritten().run(...)["rule_pack"]` as the parity tests already do; python@2 ignores asp. | todo |
| `layers/l5_rules/verify.py:L220-237` `grid_table()`: delete (only test_rules_grid uses it). | todo |
| `verify.py:L241-253` `facts_parity()`: delete or one real test `facts_parity(relations.build(cell), F.facts_of(cell))`. | todo |
| `layers/l1_scene/coverage.py` + `sigma.py`: yagni: no plugin/config reaches either (benchmark scenes have no coverage, 22/21 unfilled sigmas) — research tools inside a layer package. Wire as scene-json cfg (`coverage_frames:`, `fill_sigma:`) or park in research/. Hard deletes inside: `attach_coverage` (= `scene.model_copy(update=...)`), `estimate_floor_level` (every caller passes `floor_level(scene)`). | todo (wire as cfg: it is Phase B's point) |

## Two renderers of the same rules

| Finding | Status |
|---|---|
| `handwritten.py`: delete the hand-typed ASP of the four threshold rules + reach_over; `asp.render(rule)` reproduces it (test_rules_funclib proves it); keep only the enclosure text; handwritten@3. | todo |
| `clingo.py:L28,L40-54`: delete legacy atoms bottom/top/dist/reach_over + `tops` dict + LEGACY (only the hand ASP uses them; `untrusted` is emitted by the generic bool path); clingo@3; verdicts unchanged (baseline parity proves it). | todo (together with the line above) |
| `asp.py:L78-91`: shrink: `render()` swallows Unsupported, then `common.finish` calls `unsupported()` to re-render for the message. Make `_render` public and catch `Unsupported` once in `finish`. | todo |

## Platform already does it / config nobody sets

| Finding | Status |
|---|---|
| `python.py:L280-286`: native: builds its own Provenance from cfg keys no config sets, ignoring `inputs["provenance"]` the runner hands every layer (clingo uses it). | todo |
| `html.py:L20`: native: `DECLARED` hardcodes 8 input names; signature has 9. Use the signature's declared_inputs (give L7 the signature / whole state). | todo |
| `runner.py:L17-19,L85`: yagni: `NEEDS` re-states plugins.py's docstring table to slice state per layer. Pass `{**state, "provenance": prov}`. | todo |
| `runner.py:L41-43` `version:` config check; `llm.py` effort plumbed through four files; `plugins.py` runtime_checkable Protocol, entry-point group: nothing sets or uses them. | todo (effort stays: the first live run tunes it) |
| `common.py:L24-25,L108,L151`: native: hand-rolled OPERATOR alias map on a free-form `operator: str`; structured outputs enforce an enum (llm_extract does). `Literal[...]`. | todo |
| `common.py` deepcopy per key → `clause.model_dump(include=...)`; `RULE_CLASS` identity dict; term collector duplicated with `alignment.clause_terms` → one `Clause.terms()` on the contract. | todo |
| `alignment.coverage()` only used by one test; status tuples copied 3× (`get_args(Status)`); SYMMETRIC / ATOM regex 4× and 3× → contracts.py; mm / corners / plan_basis / u_mm written 2-3× → one `synth/geom.py`. | todo |
| `contracts.py`: `read_json`, `Rule.tests`, `Obj.t`, `Fact.t`, `ObjSource/Obj.source` set by nothing. | policy (`t` / `source` are the 4D and believed-object hooks of the architecture doc; keep, document) |
| `python.py` decision `simple` / `conservative`: only tests use them. | policy (they are the decision-rule variants of the plan; keep) |
| `python.py` per-rule candidates/not_applicable/exception counters: read by nothing but tests. | todo (keep `skipped`) |
| `matrix.py` ProcessPoolExecutor for 9 configs × 2 items: a for loop until the matrix takes minutes. | policy (the plan asked for parallel; keep) |
| `labels.py` + `lab/cli.py`: own ArgumentParser forwarded through REMAINDER → `labels.add_arguments(sub.add_parser("labels"))` like extract. | todo |
| `retrieve.py`: a module for one 4-line function → fold into clause_kg.py. | todo |
| `html.py:L74,L151-153`: delete the clipboard copy + execCommand fallback (Export + textarea already are the fallback). | todo |
| `verify.py:L34` engine_of accepts class or instance (every caller passes the class); `ok()` → assert. | todo |

## Tests that duplicate another

| Finding | Status |
|---|---|
| `test_engine_parity.py:L17-68` re-derives Facts from the trial's scene-graph.json to prove python@2 = trial; python-engine.yaml + relations@2 already produce those Facts → parametrize test_baseline_parity over [baseline, python-engine] × items; keep the synth enclosure parity. | todo |
| `faithful` fake ×2, `inputs` fixture ×3, `SIGNATURE=Signature.load` ×6, `load_trial_scene` ×2, box/obj helpers ×3 → one `tests/verdict/conftest.py`; use `scenes.box`. | todo |
| `test_report_html` vs `test_determinism` both run the runner twice and diff → parametrize test_determinism over [baseline, html-report]. | todo |
| `KNOWN_GAPS` / `KNOWN_TAG_GAPS` / `GAP_TERMS` are empty sets with "now resolved" comments → plain asserts. | todo |

## Overall verdict (the reviewer's)

The skeleton is proportionate: registry + config + runner + matrix + scorecard + CLI is ~370 lines for nine configs × two cells; contracts are
259 lines with four dead fields; the three L5 variants are cheap (152 lines over common.py). The weight is in things that exist twice: two
evaluators of one Rule.spec (asp.py + clingo.py = 318 lines vs python.py = 305), two Scene→Facts producers (relations.py 202 vs synth/facts.py
182), two copies of the six trial rules and two renderings of their ASP, two CLIs for L4 extraction, a 244-line table library nothing executes,
and a 253-line verification harness of which only signature_check, run_tests, threshold_grid and the rigid/inflate relations gate anything. The
html labelling report is fine if labels are being collected. If it had to be half the size: keep python@2 as the only engine, freeze clingo.py /
asp.py / the hand ASP as a research script (−450); run the harness on relations@2 through the registry and drop synth/facts.py + synth/pack.py
(−230); drop tables.py + tests until a verified text exists (−244); drop extract_cli.py and measurement_layer.py (−127); fold verify.py into
common.py keeping only the gating relations, and matrix/config/cli into runner (−100). ≈ 1,150 lines, about 21 %; reaching half would mean
removing the experiments themselves (stpl, coverage, the two LLM variants), which is the point of a lab.

**net: −930 lines possible** (about −480 mechanical; tables.py, synth/facts.py and the decision variants are policy calls).

Owner's reading (2026-10-08): the two-engine design stays (clingo vs python parity is the lab's equivalence test, by plan); the mechanical
items marked **todo** are applied in the next lab round; the lab itself is off the delivery path (`docs/REVIEW-ARGUS-2026-10-08.md` §2) and
is not part of the handoff-back cleanup.
