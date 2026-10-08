# L1 — producers -> Scene (C1)

| plugin | what | source | params | limits |
|---|---|---|---|---|
| `scene-json@1` (scene_json.py) | loads an already-converted contract Scene JSON (benchmark/v0/scenes) | — | `declared`: path to a declared.json (`panoptes verdict labels declared`, `{scene_id: {input: value}}`); its entry for the scene is merged into `Scene.declared_inputs`, the file winning. Without it the output is unchanged | no conversion; only the contract's validation |
| `measurement-layer@1` (measurement_layer.py) | published measurement layer JSON (web/src/measurement-layer.ts shape) -> Scene: boxes to metres, `sigmaCm` to `sigma_m`, face photo ids to `views` | trial adapter_measurement_layer.py (2026-10-07) | `scene_id` (default: the file's stem) | classes from the English labels only (placeholder semantics); no zones, no coverage, no declared inputs |
