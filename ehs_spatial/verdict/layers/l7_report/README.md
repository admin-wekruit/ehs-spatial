# L7 — VerdictSet -> report files with provenance

| plugin | what | source | params | limits |
|---|---|---|---|---|
| `markdown@1` (markdown.py) | `verdicts.md` in the layer directory: header with run id, benchmark, rule pack, decision rule, plugin tags and status counts; one table line per verdict (rule@version, subjects, status, measured ± u, threshold, margin, evidence facts / openings / needed inputs, notes) | trial engine.py `table` (2026-10-07), with provenance added | — | no figures, no per-verdict photo crops |
