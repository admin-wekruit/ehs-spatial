> **已被取代（2026-10-08）**：逐句整理进固定 RASE 模板 + 词表缺口即拒绝的做法被 `docs/research/verdict-spec-to-check-2026-10-08.md` 的架构取代
> （条款图抽取 → 对齐表 → 按场景检索 → 对有类型场景 API 合成检查 → 自动验证 → 字面化审阅 → 确定性执行）。本文件留作历史。

# Compile prompt (production): natural-language safety rule → arranged RASE rule over the closed vocabulary

Use with the customer's model key (the August path: `scripts/policy_compile.py --live` calls Gemini; swap the prompt for this one and
the validator for `rase_schema.py`). One call per rule. The model arranges; it never judges. Everything it emits is checked by
`rase_schema.validate`, so an invented class / edge / input is rejected before it can run.

```
You arrange one written workplace-safety rule into a strict JSON structure. A separate deterministic engine evaluates the
structure against a measured 3D scene; you never decide compliance and you never invent measurements.

VOCABULARY (exact strings; nothing else exists):
  classes (available): {classes.available}      classes (planned, usable but flagged as gaps): {classes.planned}
  zones   (available): {zones.available}        zones   (planned): {zones.planned}
  edges   (available): {edges.available with unit and endpoints}   edges (planned): {edges.planned}
  attributes (available): {attributes.available}   attributes (planned): {attributes.planned}
  declared inputs (never visible in photos; the engine asks a human / datasheet for them): {declared_inputs}
  operators: <=, >=, ==, exists, forall, not_exists      units: mm, deg, bool, cell

ARRANGE the rule into four parts:
  selection     – variables and the classes / zones each may bind to ("which objects")
  applicability – relations (edge names between the variables) and zones that must hold for the rule to apply ("when")
  requirement   – one edge, one operator, a threshold with unit, or a formula naming its declared inputs ("what must hold")
  exception     – attributes that switch the rule off ("unless")
Rules of arrangement:
  1. Use only vocabulary strings. If the rule needs a class, zone, edge or attribute that is only PLANNED, use the planned name
     (the validator reports it as a vocabulary gap). If it needs something that is neither available nor planned, set
     unsupported_reason instead of approximating.
  2. Numbers: convert to mm / deg. Keep the clause reference in "clause". Do not change thresholds; if the standard gives a table
     or a formula, write the formula and list its declared inputs.
  3. A quantity that photographs cannot show (stopping time, resolution, PLr, restricted space, risk level, body part,
     reach radius, table values) is a declared input, never a guess.
  4. Procedures, records, training, time-based duties, people / PPE in a static workcell, electrical state: unsupported_reason.
  5. One rule → one JSON object; a sentence with several thresholds for several cases becomes a formula over a declared input
     (e.g. Table1(body_part)), not several guesses.

Respond with ONLY one JSON object with keys: rule_id (slug), source_text (verbatim), clause, selection {variables}, applicability
{relations, zones, note}, requirement {edge, operator, threshold, unit, formula, inputs} or null, exception {attributes, note},
unsupported_reason (string or null).

Rule: {text}
```

After the run: `python rase_schema.py <compiled.json>` prints per rule compiled / compiled_needs_input / vocabulary_gap / refused /
invalid, and the harness records the three KPIs of the research note (compile rate, refusal rate, hand-fix rate). Rules with
status compiled or compiled_needs_input are then rendered into `rules.lp`; vocabulary gaps go to the representation backlog.
