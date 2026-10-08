# `ehs_spatial.verdict.synth` — synthetic scenes for the lab

| module | role |
|---|---|
| `scenes.py` | `cell(fence_height_m, floor_gap_m, robot_fixed_gap_m, lc_bottom_m, enclosed, opening_width_m, coverage, sigma_m, ...)` → contract `Scene`: robot, four fence segments (ring, optional opening), two bollards, light curtain, one unreferenced cart; axis-aligned; `Coverage` on the facts grid frame when `coverage='full'` |
| `facts.py` | `synthetic-facts@1`: Scene → Facts for axis-aligned boxes (`obj(id, cls)`, `top_height`, `bottom_height`, `floor_gap`, `min_distance_3d`, `horizontal_gap`, `z_overlap`, `reach_over`, plan grid with `observed` resampled from the Coverage). U mirrors the trial's `u_mm`; integer mm; deterministic. Not a registered plugin: the oracle the real L2 must agree with on these scenes (`verify.facts_parity`) |
| `grid.py` | threshold grid: per numeric rule, cells at threshold ± ε, ± 2σ, ± 5σ (σ = U/2 of the fact) with the guard-band k=2 expectation; ± 2σ sits exactly on `V ∓ U = T` |
| `pack.py` | the trial's six rules (`rules.lp`) as `Rule.spec` entries — the reference pack of these tests, not the lab's reviewed pack |

Scenes are small (nine boxes, ≈ 50 × 90 cells of 0.1 m) for the memory-limited machine.
