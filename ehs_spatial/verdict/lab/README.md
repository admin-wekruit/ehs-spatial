# lab — run, matrix, ledger, scorecard

One run = one YAML config x one benchmark item, L1..L7 in order (`runner.py`):

```
panoptes verdict run --config ehs_spatial/verdict/configs/baseline.yaml --item 090 --runs-dir runs [--run-id ID]
panoptes verdict matrix --configs ehs_spatial/verdict/configs/*.yaml --jobs 4      # items x configs, one process per task, then the scorecard
panoptes verdict scorecard --runs-dir runs                                          # scorecard.md / scorecard.json from the ledger
panoptes verdict plugins                                                            # what is registered, per layer
```

Human labels (`labels.py`; the L7 `html@1` report exports one `verdict-labels/1` JSON per reviewer and run, see layers/l7_report/README.md):

```
panoptes verdict labels merge --labels a.json b.json --out gold.json [--gold-in old/gold.json] [--version v1]   # agreements -> verdicts, disagreements -> conflicts; copy to a NEW benchmark/v<N>/, never into v0
panoptes verdict labels declared --labels a.json b.json --out declared.json                                     # {scene_id: declared inputs} for L1 {plugin: scene-json, declared: declared.json}
```

Config (`config.py`): `benchmark: <dir>` (relative to the config file), optional `item`, and per layer `{plugin, version?, params..., reuse?}`.
`extends: base.yaml` merges a base config (layer dicts key by key). `reuse: <run_id>` loads that run's `<layer>/` outputs instead of computing
(the plugin name stays in the config: it is the run's signature). No Hydra, no content hashing: the run directory is the cache.

Run directory:

```
runs/<run_id>/config.json                   the resolved config (+ item, run_id)
runs/<run_id>/L1/scene.json                 each layer's outputs: contract models through their dump, dicts / lists as sorted JSON
runs/<run_id>/L2/facts.json … L5/rule_pack.json, L6/verdicts.json + program.lp, L7/verdicts.md
runs/ledger.jsonl                           one line per run: run_id, config_file, benchmark, item, started, seconds, plugins {L1: tag, …}, reused, llm_calls
```

run_id = `<YYYYmmdd-HHMMSS>-<git short sha>` unless given (`--run-id`, which the tests fix); the matrix appends `-<config stem>-<item>`.
Every layer also receives `provenance` (run id, `<benchmark>/<item>`, plugin tags) in its inputs; L6 writes it into each verdict.
A plugin may return `llm_calls: n` in its outputs; the runner sums it into the ledger.

Scorecard (`scorecard.py`): rows = runs in ledger order; columns = plugin tag per layer, counts per status, agreement with
`benchmark/<v>/gold.json` per gold status (marked provisional when the gold says so), and `diff vs previous row` = the item and the layers
whose plugin tag or params changed (`reuse` is not a change).
