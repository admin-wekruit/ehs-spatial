# ehs_spatial.verdict — the verdict layer as a plugin lab

Design: `docs/research/verdict-layer-architecture-2026-10-08.md` (seven layers, four contracts) and
`docs/research/verdict-layer-plan-2026-10-08.md` §7 (plugins, parallel matrix, provenance). This package is the implementation.

## Layout

```
contracts.py          C1 Scene, C2 Facts, C3 Signature, C4 RulePack, C5 Verdict (pydantic). The ONLY module plugins may import.
plugins.py            Plugin protocol, @register(layer, name, version), REGISTRY, entry points, tag()
signature-v1.json     C3 of this lab; L2 must produce only these predicates, L4/L5 may only reference them
layers/l1_scene/      producers -> Scene    (measurement-layer adapter today; DAAAM / Hydra / WorldSGG adapters later)
layers/l2_facts/      Scene -> Facts        (the relation library: distances, gaps, overlaps, reach-over, line of sight, occupancy)
layers/l3_monitor/    Scene, Facts -> Facts (perception spec: persistence, size consistency, floor contact, coverage; flags 'untrusted')
layers/l4_spec/       spec_dir -> clause graph, alignment table, retrieved clause ids
layers/l5_rules/      clauses -> RulePack   (handwritten pack today; synthesis + verification plugins)
layers/l6_engine/     Facts, RulePack -> VerdictSet (clingo; python parity engine; decision rules)
layers/l7_report/     VerdictSet -> report files with provenance
lab/                  runner (one run = one YAML), matrix (benchmark x configs, parallel), ledger, scorecard
benchmark/v0/         frozen inputs + gold (versioned folder, in git; bump the version when it changes)
```

## Rules of the lab

1. A plugin imports `ehs_spatial.verdict.contracts` and its own package only. `tests/verdict/test_imports.py` fails otherwise.
2. A plugin is a class with `run(inputs, cfg, workdir) -> dict`, registered with `@register('L6', 'clingo', '1')`; `tag()` = `clingo@1`
   goes into every verdict's provenance and the scorecard. Bump `version` when the output can change.
3. Determinism: same config, same inputs -> byte-identical outputs. The runner's test runs twice and diffs. No clocks, no random
   seeds without a fixed value, no dict-order dependence (sort what you serialise).
4. Missing is a value: a rule that lacks a fact or a declared input yields NEEDS_INPUT / CANNOT_DETERMINE, never PASS.
5. Caching = reuse by run id: a layer in the config may say `reuse: <run_id>` and the runner loads `runs/<run_id>/<layer>/`.
   No content hashing. The only hash in this package is the LLM response cache file name (model + prompt text).
6. Parity first: a new plugin for an existing layer ships with a test that it agrees with the baseline where it should
   (e.g. a new L6 engine must reproduce clingo's verdicts on benchmark/v0 exactly) before any claimed improvement is measured.
7. Units: millimetres in Facts and Verdicts; metres in Scene. Uncertainty `u` is expanded (k=2).
8. Every plugin directory has a README: what it does, where the method comes from (paper / standard clause), parameters, known limits.
9. No defensive excess: validate at the contract boundary (pydantic) and nowhere else.

## Baseline plugins (ported from research/verdict-layer-trial-2026-10-07)

L1 `measurement-layer@1` (adapter_measurement_layer.py) · L2 `relations@1` (scene_graph.py + relations.py) · L6 `clingo@1`
(engine.py + rules.lp, guard band k=2) · L7 `markdown@1` (verdicts.md with provenance). L3 `none@1`, L4/L5 `handwritten@1`
(the trial's five rules as a RulePack).
