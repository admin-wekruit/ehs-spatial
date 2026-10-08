# L5 — clauses -> RulePack (C4)

`handwritten@1` (handwritten.py): the trial's rules.lp (2026-10-07) as a `RulePack` `handwritten-trial@0`, ignoring L4's clause graph.

| rule | spec requirement | threshold | clause (unverified numbers) |
|---|---|---|---|
| `fence_height` | top_height(F) ≥ | 1400 mm | ISO 13857:2019 Table 2 note |
| `floor_gap` (fence, guard) | bottom_height(F) ≤ | 180 mm | ISO 13857:2019 4.4 |
| `lc_lowest_beam` | bottom_height(L) ≤ | 300 mm | ISO 13855 (2010 numbers) |
| `crush_gap` (robot × fixed) | min_distance_3d(R, X) ≥ | 500 mm | ISO 13854:2017 Table 1 / ISO 10218-2 (clause unverified) |
| `reach_over` (robot × fixed) | reach_over(H, S) table | — | ISO 13857:2019 Table 2: status `needs_input`, inputs `table_lookup`, `risk_level` |
| `enclosure` | enclosed(Z) | — | topology on the occupancy grid (outside -> hazard reachability) |

Each `Rule` carries `spec` {selection, applicability, requirement, exceptions, notes} for engine-neutral evaluation and `asp` for clingo;
`common_asp` has the guard-band decision (`status` pass / fail / needs_meas / cannot_determine, `margin`) and the `#show`s.
Params: none (the guard-band coverage factor `k` is the engine's). Limits: numbers not verified against the purchased texts; no zones,
no declared inputs consumed; `floor_gap` evaluates `bottom_height` as the trial did (a box sunk below the fitted floor gives a negative value).
