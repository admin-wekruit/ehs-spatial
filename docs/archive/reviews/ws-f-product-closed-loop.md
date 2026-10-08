# WS-F — product closed loop (history, disposition, report, drift)

A run is no longer a dead end: the inspector can browse cached runs, rule on
a NEEDS_REVIEW verdict, and print a report — all from artifacts already on
disk, no provider calls.

## What landed

- `ArtifactStore.list_runs()` — scans `runs/`, summarizes each run from
  manifest.json + assessment.json + review.json. Missing/corrupt files
  degrade to `None` fields; one bad run never hides the rest. Verified
  against the real main-repo `runs/` (legacy runs without manifests render
  with None fields).
- **History tab** (app.py) — existing workbench wrapped in
  `gr.Tabs(Workbench | History)`; components, events, `_status_copy`,
  and the analyze 8-tuple untouched. Refresh → `gr.Dataframe` of
  `list_runs()`; row select loads the verdict card (via `_status_copy`,
  read-only), top-down PNG, and the current disposition. The selected run id
  lives in a read-only Textbox, not a second `gr.State` (the config test
  pins exactly one state component).
- **Disposition panel** — reviewer / confirmed-vs-overridden radio /
  override-status dropdown / reason. Validated through the frozen
  `ReviewDisposition`; overriding without a reason (or without a target
  status) raises `gr.Error`. Written to `review.json`; `assessment.json` is
  never rewritten — the disposition is a separate auditable layer.
- `ehs_spatial/report.py` — stdlib-only printable page per run: header +
  provider pins, verdict with `d ± budget` (never a bare decimal), policy
  table (handles both the `{"specs","results"}` envelope and legacy bare-list
  policies.json), warnings, disposition block, and topdown / plan_view /
  first overlay embedded as base64 data URIs. Everything optional: a partial
  run yields a partial report. All artifact text is HTML-escaped.
  CLI: `python -m ehs_spatial.report runs/<id>`.
- `scripts/drift_check.py` — offline mode replays the oracle pack via
  `run_offline_benchmark` (zero paid calls) and diffs statuses/distances
  against stored expectations; tolerance defaults to the pack's own 0.1 m
  gate, and an INSUFFICIENT_EVIDENCE expectation means "no distance"
  (mirrors the pack's sufficiency rule — naive comparison false-alarms on
  fence_occluded). `--live` re-runs ONE case via `run_live_benchmark_case`
  and diffs against the cached offline baseline; it prints the planned spend
  and refuses without the flag, without a pack, or without a baseline.
  Offline mode run here: `offline replay: stable, no drift`. `--live` was
  never executed (no paid calls).

## Deliberate simplifications

- Disposition form fields are not prefilled from an existing review.json;
  the existing ruling is displayed as copy above the form. Add prefill if
  reviewers ask for edit-in-place.
- The history table does not auto-refresh after saving a disposition; the
  refresh button re-reads the index.
