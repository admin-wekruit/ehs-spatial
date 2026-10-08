# L6 — Facts, RulePack -> VerdictSet (C5)

`clingo@1` (clingo.py): port of the trial engine (research/verdict-layer-trial-2026-10-07/engine.py + rules.lp, clingo 5.8).
Program = `RulePack.common_asp` + every rule's `asp` + the Facts rendered as `obj/2`, `bottom/3`, `top/3`, `dist/4` (both orders),
`reach_over/6` (a / b from the fact's `a_mm=` / `b_mm=` flags, U = max of the three), the grid as `cell/adj/blocked/hazard/outside/observed`,
and `untrusted/1`. One stable model; `status/3`, `margin/3`, `opening/1` become Verdicts with measured / u / threshold / margin / evidence
(the facts used, their views; for topology rules the grid summary, opening count and directions) and the run's provenance.

Params (cfg): `k` — guard-band coverage factor (default 2). Facts carry u at k = 2; the rendered half width is u·k/2, so k = 2 is the
trial's decision, k = 1 halves the band, k = 0 is simple acceptance. `decision_rule` = `guard_band_k<k>`.
Also writes `program.lp` (what the solver saw) into the layer directory.

Limits: integer millimetres; `measured` is looked up by the rule's `spec.requirement.predicate` and the subject ids (either order for
pairs); a rule whose spec names no predicate gets status only; `untrusted` is rendered but the baseline pack does not use it.
