# L7 — VerdictSet -> report files with provenance

| plugin | what | source | params | limits |
|---|---|---|---|---|
| `markdown@1` (markdown.py) | `verdicts.md` in the layer directory: header with run id, benchmark, rule pack, decision rule, plugin tags and status counts; one table line per verdict (rule@version, subjects, status, measured ± u, threshold, margin, evidence facts / openings / needed inputs, notes) | trial engine.py `table` (2026-10-07), with provenance added | — | no figures, no per-verdict photo crops |
| `html@1` (html.py) | `verdicts.html`: the same header, then the labeling UI — a queue of all verdict rows and a focus panel for the current one (subjects with class / size / id, measured ± u, threshold, margin, needed inputs, evidence facts, notes, views) where a reviewer records the gold status, a reason, sure / unsure and a re-measure flag; a declared-inputs form for the scene; export / import of a `verdict-labels/1` JSON | docs/research/labeling-ui-survey-2026-10-08.md (Argilla focus view, Prodigy / Potato number keys, Label Studio queue + skip, Braintrust / Encord label + reason, VIA single-file offline) | — | one HTML file, inline CSS + JS, nothing fetched (opens from file:// and from Pages); phone width works; dark mode via `prefers-color-scheme`; no photos |

## html@1

Hotkeys: `1` … `6` = PASS, FAIL, NEEDS_MEASUREMENT, NEEDS_INPUT, CANNOT_DETERMINE, NOT_APPLICABLE · `n` / `p` next / previous row ·
`r` toggle re-measure · `u` toggle unsure · `Enter` save + next (also inside the reason field). Hotkeys are off while typing in other
fields. The filter shows all rows, the unlabelled ones (the skip queue) or one machine status. The progress bar counts rows with a gold
status. State is kept in memory and in `localStorage["verdict-labels/<run_id>"]` (best effort, wrapped in try / catch).

Export downloads `labels-<scene_id>-<run_id>.json`; Copy puts the same JSON on the clipboard; the textarea at the bottom always shows
it as the fallback. Import loads a previous labels file (rows matched by rule@version + subjects; other rows are ignored and counted).

Labels JSON (`schema: verdict-labels/1`):

```json
{"schema": "verdict-labels/1", "run_id": "…", "scene_id": "090-a9a6e0a0", "benchmark": "v0/090", "reviewer": "ab", "rule_pack": "handwritten-trial@0",
 "labels": [{"rule_id": "fence_height", "rule_version": "0", "subjects": ["<object id>"], "labels": ["left fence"],
             "status": "PASS", "reason": "2.0 m, reads fine", "confidence": "sure", "remeasure": false}],
 "declared_inputs": {"stop_time_ms": 500, "risk_level": "high", "restricted_space": "0,0; 3,0; 3,2; 0,2"}}
```

`labels[].labels` are the object names (informational; the key is `rule_id`, `rule_version`, `subjects`). Only rows with a status are
exported. `declared_inputs` keeps numbers as numbers, everything else as text (`restricted_space` is a free-text polygon).

Flow: reviewer opens `runs/<run_id>/L7/verdicts.html` → labels → Export → sends the JSON file (one per reviewer and run) →
`panoptes verdict labels merge --labels a.json b.json --out gold.json [--gold-in benchmark/v0/gold.json]` → copy the result to a **new**
benchmark version folder (`benchmark/v<N>/gold.json`, never into v0; disagreements are listed under `conflicts` and left out of
`verdicts`) → `panoptes verdict labels declared --labels a.json b.json --out declared.json` → L1 `{plugin: scene-json, declared: declared.json}`
puts the declared inputs into `Scene.declared_inputs`.
