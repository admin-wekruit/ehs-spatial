# L3 `stpl@1` — perception monitor

Reads the Scene (C1) and the Facts (C2), returns the Facts annotated. It never computes geometry; it decides which objects the
rules may judge. Named after the STPL / SGSM-style perception specs of `docs/research/verdict-layer-architecture-2026-10-08.md`
(L3): "is the scene graph itself believable before it is judged". This version is the static-scene subset; the cross-frame
checks (persistence, size consistency over time) wait for a 4D producer.

## Checks (per object; any failure -> untrusted)

| check | fails when | cfg |
|---|---|---|
| `views` | fewer than `min_views` photos / frames saw the object | `min_views` = 2 |
| `floor` | `floor_contact` is claimed but the base is not on the floor: abs(bottom_m) > `floor_tol_m` | `floor_tol_m` = 0.05 |
| `size` | a side is outside the class's plausible range below (applied to min(L, W), max(L, W), H) | `SIZE_RANGES_M` |

Plausible sizes (metres; thin side × long side × height; classes not listed are not size-checked):

| class | thin side | long side | height |
|---|---|---|---|
| fence | 0.05–0.6 | 0.3–10 | 0.8–3.5 |
| guard | 0.02–3 | 0.1–10 | 0.1–3 |
| bollard | 0.05–0.5 | 0.05–0.5 | 0.3–1.5 |
| light_curtain | 0.02–0.3 | 0.02–0.3 | 0.1–2.5 |
| robot | 0.3–3 | 0.3–3 | 0.3–4 |
| cart | 0.3–2 | 0.3–3 | 0.3–2 |
| estop | 0.02–0.3 | 0.02–0.3 | 0.02–0.3 |
| interlocked_door | 0.02–0.3 | 0.5–3 | 1.0–3 |
| control_panel | 0.1–1.5 | 0.1–2 | 0.2–2.5 |
| person | 0.2–1 | 0.2–1 | 0.5–2.2 |

The ranges are engineering bounds for a workcell (fence panels, posts, curtains, industrial robots), not standard clauses; a
box outside them is a reconstruction error far more often than an unusual object. Widen a row rather than drop the check.

## Output

- `facts.facts` gains `Fact(pred='untrusted', args=[id], value=1, unit='bool', flags=[failed checks])` per untrusted object, and
  every existing fact whose args include an untrusted object gets the flag `'untrusted'`.
- `facts.quality = {'monitor': 'stpl@1', 'untrusted': [ids, scene order], 'coverage_ratio': observed cells / all cells of
  `Scene.coverage` (None without coverage), 'checks': {id: {views, bottom_m, floor_contact, size_m, failed}}}`.

Deterministic: object order of the Scene, no clocks, no randomness. The input Facts is copied, not mutated.

## Limits

- `coverage_ratio` is reported, not judged: how much coverage a topology rule needs is the rule's threshold (L5), not L3's.
- A single photo of an object with a measured sigma is still 'untrusted' here (views < 2); set `min_views: 1` in the run
  config to let one-view scenes through when that is the intended experiment.
- No cross-frame persistence yet (static scenes only).
