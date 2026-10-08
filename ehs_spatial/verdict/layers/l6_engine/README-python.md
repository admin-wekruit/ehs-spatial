# L6 `python@1` — rule specs evaluated directly on Facts (parity engine for `clingo@1`)

`python.py`. Registered as `@register('L6', 'python', '1')`; cfg keys: `decision` (`guard_band` default | `simple` | `conservative`),
`k` (guard band only, default 2; Facts carry U at k=2 so U is scaled by k/2), `run_id`, `benchmark`, `plugins` (tags of the other
layers for the provenance line; L2 / L1 are filled from `Facts.producer` / `Scene.producer` when absent).

## What it does

Reads only `Rule.spec` (never `Rule.asp`), one verdict per (rule, binding of the selection variables):

1. **Selection** — variables bound to objects whose class is listed (`obj(id, cls)` facts, args = `[id, cls]`; plus `Scene.objects` when a
   scene is given) and to zones whose kind is listed (`Scene.zones`; the grid's hazard cells are the implicit zone `hazard_zone`).
2. **Applicability** — every `pred(Var, ...)` atom must hold as a fact (value `None` or non-zero); capitalised args not bound by the
   selection are bound by the fact (join), lower-case args are constants.
3. **Exceptions** — a truthy `attr(X)` fact or `Scene` attribute on a bound object switches the rule off for that binding
   (`coverage[rule]['exception']` counts them).
4. **Requirement** — the fact keyed (`predicate`, bound args); symmetric pair predicates (`min_distance_3d`, `horizontal_gap`, `z_overlap`,
   `line_of_sight`) are also looked up in reverse order. Threshold = `threshold` (number, or the name of a `Rule.thresholds` entry) |
   `formula` (arithmetic over `Scene.declared_inputs`, symbols mapped through `requirement['bindings']`) | `table` (a value declared under
   the table id; python@1 has no table functions).
5. **Status** (first that applies): untrusted subject → `CANNOT_DETERMINE`; listed input / formula symbol / table value not declared →
   `NEEDS_INPUT` (`unknown_inputs`); no fact → `CANNOT_DETERMINE`; else the decision rule.

Decision rules (`decide()`), V ± U with U = the fact's expanded uncertainty:

| decision | PASS | FAIL | otherwise |
|---|---|---|---|
| `guard_band` (rules.lp semantics) | whole interval on the passing side, boundary inclusive (`V − U ≥ T` for `≥`) | whole interval on the failing side, strict (`V + U < T`) | `NEEDS_MEASUREMENT` |
| `simple` | point value passes | point value fails | — |
| `conservative` | whole interval passes | anything else | — |

`margin` = V − T for `≥`/`>`, T − V for `≤`/`<`. Topology: `enclosed(hazard_zone)` without an `enclosed` fact is computed on `Facts.grid`
by BFS from the outside cells through unblocked cells: hazard reached through observed cells → 0 (FAIL); reached only through
unobserved cells, or `Grid.observed is None` → `CANNOT_DETERMINE`; not reached → 1 (PASS); no hazard cells → `CANNOT_DETERMINE`.
`evidence` carries the facts used (or the grid summary: openings, via_unobserved), `coverage` counts candidates / verdicts / not-applicable /
exceptions per rule, `gaps` lists unknown inputs, untrusted subjects, missing facts, skipped (`vocabulary_gap` / `refused`) rules.
Verdicts are sorted by (rule_id, subjects); same inputs and cfg give a byte-identical `VerdictSet`.

## Where the semantics come from

`research/verdict-layer-trial-2026-10-07/{engine.py,rules.lp}` (guard band after ILAC-G8 / ISO 14253-1). `tests/verdict/test_engine_parity.py`
rebuilds the trial's six rules as specs (`ehs_spatial/verdict/synth/pack.py`), converts `out/{090,030}/scene-graph.json` into Facts and asserts
the exact status, measured, U, threshold and margin of every row of `out/{090,030}/verdicts.json` (reach_over rows are `NEEDS_INPUT`,
enclosure is `CANNOT_DETERMINE` because the trial has no coverage).

## Known limits

- No table functions: a `table` requirement is `NEEDS_INPUT` until a verified lookup exists (then: declare the value under the table id, or
  add the L4 table function import to this engine behind a cfg switch).
- `reach_over` needs a (hazard top), b (structure top) besides c: `Fact` has no field for them (contract request in the WP-D report).
- `enclosed` on a declared zone (not the grid's implicit `hazard_zone`) needs an `enclosed(Z)` fact from L2.
- A numeric fact with `u = None` is decided with U = 0 (noted in the verdict).
