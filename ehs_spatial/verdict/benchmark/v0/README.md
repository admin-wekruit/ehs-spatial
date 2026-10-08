# benchmark/v0 — frozen inputs + provisional gold

- `items.yaml`: two items, 090 (`090-a9a6e0a0`, 9 objects) and 030 (`030-fafdeb6b`, 8 objects); each `source` is a contract Scene JSON.
- `scenes/scene-*.json`: the trial's `research/verdict-layer-trial-2026-10-07/out/scene-*.json` as contract Scenes (`views` as strings,
  `producer` added; no zones, no coverage, no declared inputs, `scale_rel_unc` null).
- `gold.json`: `provisional: true`, `source: "trial 2026-10-07, not human-labelled"`: the trial's verdict status per (scene, rule, subjects),
  i.e. what the baseline reproduces, not a human label. Replace with the engineer-labelled set (docs/research/verdict-layer-plan §3) under a new version.

Frozen: changing any file here means a new version folder (v1), never an edit in place.
