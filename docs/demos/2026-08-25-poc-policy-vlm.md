# POC — Written policy in, grounded VLM explanation out (2026-08-25)

One owner-facing loop, end to end: OSHA prose → reviewable compiled spec (or
an explicit refusal) → spec selected in the workbench UI → deterministic
verdict card with predicate, threshold and source prose → "why did it fail?"
answered by Gemini with cited fact ids and locally rendered numbers.

Every step below was executed against real artifacts; the replay commands are
exact.

## 1. Prose → compiled spec (offline, reviewed)

Compilation is an offline step (`scripts/policy_compile.py`), never a UI
action. The model translates one rule of prose into a `PolicySpec` — a closed,
diffable JSON structure — or refuses with a reason. The compiled specs are the
artifact a safety engineer signs off on; the UI states this
("Compiled specs are reviewed before use") and only offers precompiled specs.

A real compile, from the live OSHA exam
(`outputs/policies/osha_exam_compiled/p09.json`, referenced read-only):

> `[1910.253(f)(5)(i)(B)]` Portable generators shall not be used within 10
> feet (3 m) of combustible material other than the floor.

```json
{
  "policy_id": "p09-portable-generator-combustibles",
  "predicate": "min_separation",
  "subject_labels": ["portable generator"],
  "object_labels": ["combustible material"],
  "threshold": 3.0,
  "unit": "m",
  "severity": "critical",
  "unsupported_reason": null
}
```

A real refusal, from the same exam
(`outputs/policies/osha_exam_compiled/p03.json`):

> `[1910.157(d)(2)]` … travel distance for employees to any extinguisher is
> 75 feet (22.9 m) or less.

```json
{
  "policy_id": "p03-portable-fire-extinguishers-travel-distance",
  "unsupported_reason": "The rule requires measuring actual path-based travel distance around obstacles rather than direct Euclidean separation, and relies on labels ('employee', 'portable fire extinguisher') outside the current perception vocabulary."
}
```

The refusal is the feature: a rule the vocabulary cannot measure is declared
unmeasurable at compile time, not approximated at verdict time. Refused specs
are never evaluated (`evaluate_policy` returns `INSUFFICIENT_EVIDENCE` with
the reason), and the UI shows them disabled with that reason.

Replay (re-evaluation is free; only uncompiled rules would need `--live`):

```bash
uv run python scripts/policy_compile.py --policies docs/policies/starter.md --run demo-mono-ladder050-v4
```

This reuses the cached specs in `outputs/policies/compiled/` and writes
`runs/demo-mono-ladder050-v4/policies.json` in the `{"specs", "results"}`
envelope every reader consumes. Against that cached ladder scene it prints,
among others:

```
FAIL  p01-movable-equipment-clearance
      entity-step-ladder-04 vs entity-safety-fence-07: measured 0.0366m (limit 0.6m)
FAIL  p06-portable-ladder-tilt-limit
      entity-step-ladder-02 vs -: measured 21.5716deg (limit 10.0deg)
```

## 2. Selecting policies in the UI

```bash
uv run --env-file .env python app.py
```

The workbench Capture rail now has a **Policies** accordion:

- A checkbox group lists every precompiled spec found in
  `outputs/policies/compiled/*.json` at app build time, labelled
  `policy_id — first 60 chars of the source prose`. For the starter set that
  is 7 selectable specs (p01, p03–p08).
- Unsupported specs appear as **disabled** checkboxes carrying the compiler's
  refusal (p02 "conditional enclosure clause…", p09 "cannot verify paint
  color…").
- The accordion copy states the review discipline: compiled specs are
  reviewed before use; there is no live compilation from the UI.

Selected specs go into `CaptureRun.policies`; `analyze_run` evaluates them
deterministically (`ehs_spatial/policy.py`, pure shapely over the SceneMap,
with the same ±error-band discipline as the demo clearance rule) and the
verdict card lists each result worst-first with predicate, threshold, worst
measurement and the source prose excerpt:

```
- `FAIL` p01-movable-equipment-clearance — min_separation 0.6 m — worst 0.0366m (limit 0.6m) — "Movable equipment must be kept at least 0.6 m clear of any machine guar..."
```

## 3. Asking the VLM why a policy failed

Policy measurements are `SpatialFact`s (`PolicyResult.facts`). When a run has
a `policies.json`, `EHSAssessmentPipeline.answer_question` now hands those
facts to the Gemini grounded-answer path as additional citable context. The
grounding discipline is unchanged: the model only *selects* fact ids, the
application renders the numbers locally, and an unknown id is a hard
`ProviderError` — no invented numbers, ever.

Live exchange, captured 2026-08-25 against the cached run (2 Gemini calls,
zero SAM/MapAnything):

```python
# uv run --env-file .env python - <<'PY'
from ehs_spatial.pipeline import EHSAssessmentPipeline
answer = EHSAssessmentPipeline().answer_question(
    "demo-mono-ladder050-v4",
    "Why did policy p01-movable-equipment-clearance fail?",
)
print(answer.model_dump_json(indent=2))
# PY
```

Actual response:

```json
{
  "answer": "SceneMap facts: min_separation(entity-step-ladder-02, entity-safety-fence-01) = 0.542 m [fact-p01-movable-equipment-clearance-12]; min_separation(entity-step-ladder-02, entity-safety-fence-07) = 0.4498 m [fact-p01-movable-equipment-clearance-16]; min_separation(entity-step-ladder-04, entity-safety-fence-01) = 0.3576 m [fact-p01-movable-equipment-clearance-23]; min_separation(entity-step-ladder-04, entity-safety-fence-07) = 0.0366 m [fact-p01-movable-equipment-clearance-27]",
  "fact_ids": [
    "fact-p01-movable-equipment-clearance-12",
    "fact-p01-movable-equipment-clearance-16",
    "fact-p01-movable-equipment-clearance-23",
    "fact-p01-movable-equipment-clearance-27"
  ],
  "evidence_frame_ids": ["frame_0001"]
}
```

And for the tilt policy ("Why did policy p06-portable-ladder-tilt-limit
fail?") the model cited exactly the two tilt facts:
`fact-p06-portable-ladder-tilt-limit-01` (21.5716 deg) and
`…-02` (19.0823 deg) — the same measurements the verdict card shows against
the 10 deg limit.

In the UI this is the existing "Ask about this run" box: after an analysis
with selected policies, type the question; the chat shows the rendered facts
plus `Fact IDs: …`.

## 4. Where each piece lives

| Step | Code | Artifact |
| --- | --- | --- |
| Compile / refuse | `scripts/policy_compile.py` (model, cached) | `outputs/policies/compiled/*.json`, `outputs/policies/osha_exam_compiled/*.json` (read-only exam) |
| Select in UI | `ehs_spatial/app.py` — Policies accordion, `load_policy_specs`, `analyze_run` | `CaptureRun.policies` |
| Deterministic verdict | `ehs_spatial/policy.py` — `evaluate_policies` (no model) | `runs/<id>/policies.json` |
| Grounded explanation | `ehs_spatial/pipeline.py` `answer_question` + `ehs_spatial/providers/gemini.py` `answer(policy_facts=…)` | `runs/<id>/chat.jsonl` |

Gates at time of capture: `uv run python -m pytest -q` → 288 passed,
2 skipped; `uv run python scripts/ehs_eval.py generate --output outputs/ehs_v1`
and `… offline --pack outputs/ehs_v1` → pass.
