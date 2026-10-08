# layers/l4_spec — the specification side (L4)

L4 turns specification text into three things L5 consumes: a **clause graph** (`ClauseGraph`: clauses, typed tables, definitions),
an **alignment table** (`Alignment`: clause terms ↔ Signature entries with confidence) and a **retrieved set** (clause ids that
apply to the scene at hand). Design: `docs/research/verdict-spec-to-check-2026-10-08.md` §3 levels 1–3.

```
clause_kg.py   plugin L4 clause-kg@0: load spec/clauses-v0.json -> align -> retrieve
tables.py      typed lookup functions the clauses point at (ISO 13857 Tables 2/4/7, ISO 13855 S and C, ISO 13854 Table 1)
alignment.py   align(ClauseGraph, Signature) -> Alignment; coverage(Alignment)
retrieve.py    retrieve(ClauseGraph, Scene | None) -> [clause id]
llm_extract.py plugin L4 llm-extract@0: safety-concept TEXT -> ClauseGraph through Claude, verified, aligned, retrieved, diffed
extract_cli.py `panoptes verdict extract`: llm-extract@0 standalone on one text (clauses + report + diff files, markdown summary)
../../spec/samples/*.md           the texts llm-extract reads; ../../spec/clauses-ts0011963-v0.json the hand-extracted reference
../../spec/clauses-v0.json   the hand-extracted v0 graph (20 clauses, 7 tables, 32 definitions)
```

Inputs: `spec_dir` (default: `ehs_spatial/verdict/spec/`), `signature` (default: `signature-v1.json`), `scene` (optional).
Config: `clauses_file` (default `clauses-v0.json`). Output: `{'clauses': ClauseGraph, 'alignment': Alignment, 'retrieved': [ids]}`.
No LLM, no network, no randomness: the same inputs give byte-identical outputs.

## How clauses-v0 was extracted

By hand, from the research note, not from the standards (the ISO texts are paywalled and were not bought):

- `docs/research/verdict-layer-rules-2026-10-07.md` §C (requirements table, caveats) and §D6 (first rule pack),
- `docs/research/verdict-layer-rules-2026-10-07-survey-B-specs-harness.md` PART 1 (every number, its vendor source URL, the
  agent's own **unverified** marks),
- `research/verdict-layer-trial-2026-10-07/compiled_v1_draft.json` for the selection / applicability / requirement shape (r05–r10).

Each clause is one `Clause` of `contracts.py`: `id` = `<standard>:<edition>/<clause or table>`; `paraphrase` in our words (no
quotes); `rule_class`; `selection` (variable → Signature classes / zones); `applicability` (Signature predicate calls that must
hold, e.g. `perimeter_of(F, Z)`, `floor_gap(F) > 35`); `requirement` = `{predicate | attribute, args, operator, threshold | table |
formula, unit, inputs}` plus optional `table_args` (how to feed the table from predicates / declared inputs), `attributes` (object
attributes read), `predicates` (helper predicates named in `formula`), `examples_mm` / `variants_mm` (the note's numbers, kept
for diffing); `exceptions` (Signature attributes that switch a case off); `definitions` (keys of `ClauseGraph.definitions`);
`tags` (classes / zones for retrieval); `photo_checkable`; `verified`; `source_url` (the vendor page the number was read from).

Where the clause number is not in the research note the id uses a slug instead of inventing one
(`ISO13855:2010/normal-approach`, `ISO10218-2:2011/trapping-clearance`, `NFPA101/aisle-width-existing`, …).
Edition years marked "assumed" in `standards` (IEC 60204-1 → 2016, ISO 3691-4 → 2020) are not from the note either.

### Where each number comes from

| clause | numbers | source (as cited in survey B) | caveat |
|---|---|---|---|
| ISO13857:2019/4.4 | floor gap ≤ 180 slot / 240 square-round | Axelent FAQ, Troax guide | clause number as reported by vendors |
| ISO13857:2019/Table2 | c(a, b): b=1400: a=2000→1100, a=1000→1000, a=400→400; b=1800: a=1400→800, a=1000→0; b=2000: a=2600→500, a≤1000→0; b=2200: a=2600→400, a≤1600→0; b=2500: a=2600→100; b=2700→0 | Troax reproduction | **`tables.py` fills the rest of the grid from recollection of EN ISO 13857 Table 2; those cells are unverified and only the survey cells are asserted by tests.** Table 1 (low risk) not extracted → `None` |
| ISO13857:2019/Table2-min-height | structure ≥ 1400 | Table 2 note, Troax | — |
| ISO13857:2019/Table4 | e → sr (slot / square / round), 9 rows up to 120 mm; 40×40 → 200 | Troax, ABB limiting-devices PDF | mesh pitch rarely resolvable from photos |
| ISO13857:2019/Table7 | slot 35–60 → 180, 60–80 → 650, 80–95 → 1100, 95–180 → 1100 | Troax, Axelent | rows e ≤ 35 and square / round not reported → `None` |
| ISO13855:2010/normal-approach | S = K·T + C, K 2000 / 1600, C = 8(d−14) ≤ 40 mm else 850, minima 100 / 500 | Reer, Omron FAQ | 2010 numbers; ISO 13855:2024 rewrites the formula |
| ISO13855:2010/parallel-approach | C = 1200 − 0.4·H ≥ 850, H ≤ 1000, crawl-under if H > 300 | Reer | crawl-under assessment not encoded |
| ISO13855:2010/multi-beam | 300/600/900/1200 · 300/700/1100 · 400/900 · 750; C = 850 | Datalogic, Reer | ±50 mm tolerance is the note's (D6), not the standard's |
| ISO13854:2017/Table1 | body 500, head 300, leg 180, foot 120, toes 50, arm 120, hand 100, finger 25 | ISO preview (body/head/leg), EN 349 white paper (rest) | finger value unverified |
| ISO10218-2:2011/trapping-clearance | ≥ 500 mm (20 in); older 18 in | OSHA robotics page, RIA intro, vendor restatement | **clause number unverified** |
| ISO10218-2:2011/perimeter-safeguarding | enclosure; openings only with interlocked door / ESPE | ABB PDF; trial rule r10 | clause number not in the note |
| ISO14120:2015/5.18 | no climbing aids (no number) | BSI summary | needs a `climbing_aids` attribute (gap) |
| ISO13850:2015/4.3 | red actuator, yellow background | SE FAQ, machinerysafety101 | latching / reset / PL not photo-checkable |
| IEC60204-1:2016/10.7 | e-stop readily accessible at each operator station | JJ Keller | operator stations are declared zones |
| IEC60204-1:2016/10.1.2 | actuator ≥ 0.6 m above servicing level | IET discussion | **no traceable upper bound** (the 1.7 m band is unverified; NEC 2.0 m is for switches) |
| OSHA:1910.176(a)-marked / -clear | aisles marked, kept clear | law.cornell.edu | colour (1910.144(a)(3), Z535.1) in paraphrase only |
| OSHA:1910.36(g)(2) | exit route ≥ 28 in = 711 mm | law.cornell.edu | needs an `aisle_width` predicate (gap); exit-route status is a human input |
| NFPA101/aisle-width-existing | 36 in = 914 (existing), 44 in = 1118 (new) | JJ Keller | clause and edition not in the note |
| ISO3691-4:2020/path-clearance | 0.5 m both sides up to 2.1 m | Certifico, TÜV | no driverless-truck class in the Signature (tag `agv`) |

Not extracted (in the note, out of this v0): ISO 13857 Table 3 (reaching around: 850 / 550 / 230 / 130), Table 1 (low risk),
ISO 10218-2 perimeter guard vs restricted space, ISO 13855's angled approach, OSHA 1910.144 colour as its own clause, everything
the note lists as not photo-checkable (ISO/TS 15066, ISO 13849-1, IEC 62046, ISO 11161).

### What `verified` means

`verified: true` on a clause or table means its numbers and clause number were compared with the purchased standard text.
Nothing in v0 is verified (`tables.VERIFIED = False`, every `verified: false`): all numbers are vendor reproductions, some marked
unverified by the survey itself. L5 / L6 must carry that flag into the verdicts' confidence. Flip per clause when a text is checked,
and bump `version`.

## Alignment (`alignment.py`)

Every term the clauses use — selection values, applicability predicates, requirement predicate / attribute / `predicates` /
`attributes` / inputs / `table_args` predicates, exception attributes, definitions keys and aliases — gets `AlignmentRow`s:

- exact Signature name → one row, confidence 1.0, `source: exact`; the row's `kind` is the kind the term was used as (so
  `resolution_mm` has an attribute row and an input row);
- not in the Signature but a `definitions` key (case / space / hyphen-insensitive) → one row per alias that is in the Signature,
  confidence 0.8, `source: synonym`;
- otherwise one row with `target: ''`, confidence 0.0 — the gap list. v0 gaps: `aisle_width` / "aisle width" (OSHA / NFPA widths),
  `climbing_aids` / "climbing aids" (ISO 14120 5.18), "AGV" (ISO 3691-4). `coverage()` = share of distinct terms with a row > 0 (v0 ≈ 0.93).

No embeddings, no classifier: this is the baseline the plan's `align: ginsign-v1` and embedding variants are measured against.

## Retrieval (`retrieve.py`)

A clause is retrieved when its `tags` intersect the set of object classes and zone kinds present in the scene; with no scene,
every clause. Graph order, no ranking. On the two benchmark scenes (fence, guard, bollard, light curtain, robot, cart; no zones)
this returns the 13857 / 13855 / 13854 / 10218-2 / 14120 clauses and not the e-stop, aisle or AGV ones. Zones (`hazard_zone`,
`operator_station`, `aisle`) widen the set once L1 / a reviewer declares them.

## How the LLM extraction plugin reproduces this file

A later `L4` plugin (plan: `clause-kg-v1`, LLM + layout parsing over the vendor PDFs or the purchased texts) must emit the same
`ClauseGraph` model and dump it with `ClauseGraph.dump` (`model_dump_json(indent=1)`), so `spec/clauses-v0.json` and its output
diff line by line. Conventions to keep so the diff is about content, not formatting:

1. clause ids: `<STD><part>:<edition>/<clause|TableN>`; a slug only where the text gives no number (then the LLM output, which
   has the text, will carry the number — that diff is expected and is the id mapping);
2. `requirement` keys exactly as above; numbers in `examples_mm` / `variants_mm` / `threshold` in mm, inputs by their
   `declared_inputs` names, tables by `ClauseGraph.tables` ids whose `function` points into `tables.py` (an extracted table that has
   no function yet is still a `Table` entry with `function: ''`);
3. `definitions` keys are the standard's own terms (singular, lower case except acronyms), aliases are Signature entries only —
   empty alias lists are the honest way to show a gap;
4. `verified` stays false until a human compares with the purchased text; `source_url` is the page the number was read from.

Scoring the LLM output against v0: per-clause field diff (ids matched exactly, then by `title`), `coverage()` of its alignment,
retrieval recall on the benchmark scenes against the ids in `tests/verdict/test_spec_plugin.py`.

## Known limits

- Table 2 outside the survey cells is recalled, not sourced (see the table above); Table 7 covers slot rows only; Table 1 absent.
- `ISO13857:2019/Table4` fixes `shape: square` (mesh); bar guards with slots need the slot column, which L5 may select from the
  object's attributes once L1 reports an opening shape.
- 13855 clauses use `zone_distance(L, Z)` to the hazard zone / restricted space; the benchmark scenes have no zones, so these
  clauses end in NEEDS_INPUT (restricted space, T, d) by design (README rule 4).
- The aisle-width and climbing-aid clauses cannot be evaluated until the Signature grows `aisle_width` and `climbing_aids`;
  they are retained so the gap is visible in the alignment, not hidden by omission.
- `plugins.py` documents the L4 output as `{'clauses': dict, 'alignment': dict, ...}`; this plugin returns the contract models
  (`ClauseGraph`, `Alignment`) like the other layers do — the runner serialises with `.dump`.

## llm-extract@0 (`llm_extract.py`, `extract_cli.py`)

Turns a safety-concept **text** into a `ClauseGraph` through Claude. Input type: markdown like
`spec/samples/manual-loading-station-concept-2026-10-08.md` — numbered requirement paragraphs and device-note bullets, each ending in
citations such as `（TS-0011963 Rev 10, 8.1.4–8.1.6）`. Config: `spec_text` (relative to `spec_dir`, falling back to the package `spec/`
like clause-kg), `reference` (hand-extracted clauses file for the diff; default `clauses-ts0011963-v0.json`), `example` (reference
clause used as the worked example, default `TS0011963:Rev10/8.1.4`), `model`, `effort`, `cache_dir`. Output: `clauses`, `alignment`,
`retrieved` (as clause-kg@0) plus `extraction_report`, `diff` (when the reference exists) and `llm_calls` (non-cached calls; the runner
sums it into the ledger). Run config: `configs/llm-extract-ts.yaml`.

Three steps:

1. **Split (deterministic).** A unit is every numbered item (the "Configuration requirements" list) and every bullet under a heading
   that says Measure / device / 装置, provided it carries a citation parenthetical naming a standard; hazard and purpose bullets and the
   picture description are not requirements. The sample gives 11 units (9 + 2). The standards named in citations become
   `ClauseGraph.standards` with ids in the clauses files' convention (`TS-0011963 Rev 10` → `TS0011963:Rev10`, `ISO 13849-1:2023` →
   `ISO13849-1:2023`).
2. **Extract (one `llm.complete` per unit, cached).** System prompt = the extraction conventions + the Signature vocabulary (classes, zones,
   predicates with arg kinds / units / notes, attributes, declared inputs) + the document's standard ids + one worked example rendered
   from the reference file (the unit citing it and that clause in the output shape). It is the long, stable part, so the API caches it
   too (`llm.py` sets `cache_control`). User prompt = the unit id, its citations and its text. Output schema (`Extraction` → `Candidate`
   list; pydantic, no dict fields because structured outputs allow `additionalProperties: false` only): clause_id, title, paraphrase,
   rule_class, selection as `[{var, values}]`, applicability, requirement `{predicate, args, operator, threshold, table, formula, unit,
   inputs}`, exceptions, tags, photo_checkable, source_quote, vocabulary_gaps.
3. **Verify (deterministic, no LLM).** Per candidate: every selection value, applicability predicate (or bare attribute), requirement
   predicate / attribute, input, exception and tag must be a Signature name, else `vocabulary_gap:<term>` and `photo_checkable: false`
   (the clause stays, so the alignment shows its 0.0 row); a numeric threshold must appear as digits in the unit text (metre values
   ground their millimetre thresholds: 0.6 → 600), else `ungrounded:<n>`; the clause number must be in the unit's citations of that
   standard (ranges include their members: 8.1.4–8.1.6 covers 8.1.5; `Table 2 note c` cites Table2; a slug suffix such as `4.3.4-upper`
   is ignored), else `uncited`; a `table` id must be one of the `Table` ids of the `clauses-*.json` files, else `unknown_table:<id>`;
   a repeated clause id keeps the first candidate and lists the rest under `duplicates`. Tables and definitions are copied from the
   reference (the model never invents tables; the aliases help alignment). Every clause is `verified: false`.

The **diff** against the reference compares ids and the requirement key `{predicate | attribute, operator, threshold, table, formula}`:
`ids_only_llm`, `ids_only_reference`, `same_id_different_requirement` (both keys shown), `same_id_same_requirement`, one-line summary.
It is a finding, never a target: do not change thresholds to make it match.

**Cache.** `cache_dir` → `$PANOPTES_LLM_CACHE` → `<runs dir>/llm-cache` (the plugin passes `workdir.parents[1] / 'llm-cache'`, i.e.
`runs/llm-cache` for `panoptes verdict run`; the CLI defaults to the same). A hit never calls the API, so a matrix re-run or a second
variant on the same text is free. **Fake mode** (`PANOPTES_FAKE_MODEL=1`) answers from the cache only and raises `LookupError` on a
miss; `tests/verdict/test_spec_llm_extract.py` monkeypatches `llm.complete` instead and answers from the reference file.

**First live command** (credentials: `ANTHROPIC_API_KEY` or an `ant auth login` profile; `PANOPTES_FAKE_MODEL` unset; ~11 Haiku calls):

```
.venv/bin/python -m ehs_spatial.verdict.layers.l4_spec.extract_cli \
  --spec-text ehs_spatial/verdict/spec/samples/manual-loading-station-concept-2026-10-08.md --out runs/llm-extract/ts/clauses.json
.venv/bin/python -m ehs_spatial.cli verdict run --config ehs_spatial/verdict/configs/llm-extract-ts.yaml --item 090   # cache hits: 0 calls
```

The second command reuses the cache because the prompts are identical (same text, Signature and reference).

**Known limits.** The splitter is built for this document shape (numbered items, device-note headings, full-width or ASCII
parentheses); a text with numbered hazards yields hazard units the model must answer with zero candidates. Grounding checks digits,
not meaning (a number quoted for another purpose grounds a wrong threshold). Citation checking cannot see a wrong standard when both
are cited in the unit with the same clause number. The worked example is one of the reference clauses, so that clause's diff row is
not evidence. `definitions` per clause stay empty (the graph-level definitions come from the reference). Nothing is verified against a
purchased standard text.
