# L5 verification harness — `verify.py`

Automatic checks a candidate `RulePack` goes through before a human sees it (`docs/research/verdict-spec-to-check-2026-10-08.md` §3 stage 5;
relations from `docs/research/verdict-evaluation-protocol-2026-10-08.md`). Pure functions over the contracts; the engine and the L2 are
parameters, so any L6 plugin (`engine_of(plugin, cfg)`) and any Scene → Facts callable (an L2 plugin's `run`, or
`ehs_spatial.verdict.synth.facts.facts_of`) can be plugged in.

| function | what it checks | output |
|---|---|---|
| `signature_check(pack, signature)` | every selection class / zone, applicability and requirement predicate (name and arity), exception attribute, declared input exists in the Signature; requirement variables are bound; operator known; a threshold, table or formula is present | list of problem strings (empty = ok) |
| `differential(pack_a, pack_b, scenes, engine, facts_of)` | status disagreements between two packs on the same facts | list of {scene_id, rule_id, subjects, a, b} |
| `metamorphic(pack, scenes, engine, facts_of)` | (1) rigid transform — rotation about the floor normal (90/180/270° by default, any angle with a real L2) + cell-multiple translation — keeps every status; (2) deleting an object no rule selects keeps every status; (3) inflating every U by 10× never flips PASS↔FAIL (only to NEEDS_MEASUREMENT); (4) moving a measured value from −5σ to +5σ across its threshold gives FAIL → NEEDS_MEASUREMENT → PASS monotonically | dict of violation lists per relation; `ok(report)` |
| `threshold_grid(pack, engine, cases=None)` | `synth/grid.py` cases (threshold ± ε, ± 2σ, ± 5σ per numeric rule) against the expected guard-band k=2 status | {rows, mismatches}; `grid_table(rows)` renders rule × offset |
| `facts_parity(a, b, tol_mm)` | value / U disagreements between two Facts (the real L2 vs `synthetic-facts@1` on axis-aligned scenes) | {value, u, only_a, only_b} |

`transform(scene, angle_deg, shift_xy)` is the rigid motion used by (1): objects, `Coverage` (basis rotated, origin shifted) and zone polygons
move together; it assumes `ground_normal = +z`.

Run on the trial pack: `tests/verdict/test_rules_verify.py` (signature, differential, metamorphic, parity helpers) and
`tests/verdict/test_rules_grid.py` (threshold grid under `python@2` with `guard_band` k=2, `simple`, `conservative`).

Limits: `signature_check` reads `Rule.spec` only (the ASP text is the clingo engine's to validate); the monotone relation perturbs one
subject per rule (all subjects share the decision function); synthetic-facts handles axis-aligned boxes, so rotations are multiples of 90°
until the real L2 is plugged in.
