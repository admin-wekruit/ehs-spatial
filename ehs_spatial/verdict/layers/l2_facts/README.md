# L2 — Scene -> Facts (C2)

`relations@1` (relations.py): the trial's relation library (research/verdict-layer-trial-2026-10-07/relations.py + scene_graph.py,
2026-10-07) emitting the predicates of signature-v1.json.

- per object: `top_height`, `bottom_height`, `floor_gap` (= bottom_height clamped at 0)
- per pair (a hazard-class object involved, or both fixed-class): `min_distance_3d` (closest points of 9x9-per-face box samples, 0 on overlap),
  `horizontal_gap` (plan footprint convex hulls), `z_overlap`, `above` (stacked, footprints intersect), `line_of_sight` (segment between the
  closest points vs every other box; `blocked_by=<id>` flags), `reach_over` (hazard x fixed: value = c, flags `a_mm=` hazard top, `b_mm=` structure top)
- `Grid`: plan occupancy, 0.10 m cells, 1 m margin; blocked = fixed-class footprints, hazard = hazard-class footprints (minus blocked),
  outside = border cells, observed = Scene.coverage.observed mapped by coordinate (None when the Scene has no coverage = unknown everywhere)
- u = k·sqrt(Σσ² + (rel·|v|)²), k = 2 (expanded); a missing σ is 0.05 m (flag `default_sigma`), a missing Scene.scale_rel_unc is 2 % (flag `default_scale_unc`)

Params (cfg): `k`, `default_sigma_m`, `default_scale_rel`, `cell_m`, `margin_m`, `face_samples`, `fixed`, `hazard`, `signature_version`.

Limits: distances come from sampled surface points (error up to the face spacing); grid cells without coverage are unknown, not free;
no zone predicates yet (`in_zone`, `zone_distance`, `perimeter_of` need Scene.zones); classes are the Scene's, not re-derived. The trial
engine ignored Scene.scale_rel_unc for top / bottom; this port uses it everywhere (identical numbers on benchmark/v0, where it is None).

Self-check: `python -m ehs_spatial.verdict.layers.l2_facts.relations`.
