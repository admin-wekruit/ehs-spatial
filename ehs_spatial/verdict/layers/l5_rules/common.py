"""Shared by the L5 synthesis plugins (function-library, code-synthesis, redundant-translation): the clause -> Rule skeleton, the
vocabulary-gap test against L4's Alignment, the status decision once a spec is in hand (render -> signature check -> status), and for
the two LLM variants the structured-output schema, the stable system prompt and the synthetic-cell test runner."""
from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from ehs_spatial.verdict import plugins
from ehs_spatial.verdict.contracts import Alignment, Clause, ClauseGraph, Fact, Facts, Rule, RulePack, Scene, Signature
from ehs_spatial.verdict.layers.l5_rules import asp
from ehs_spatial.verdict.layers.l5_rules.verify import engine_of, signature_check
from ehs_spatial.verdict.synth import facts as synth_facts
from ehs_spatial.verdict.synth import scenes

VERSION = "0"
SPEC_KEYS = ("selection", "applicability", "requirement", "exceptions")
RULE_CLASS = {"geometry": "geometry", "topology": "topology", "semantic": "semantic", "procedural": "procedural"}
OPERATOR = {">=": ">=", "≥": ">=", "=>": ">=", "ge": ">=", "<=": "<=", "≤": "<=", "=<": "<=", "le": "<=", ">": ">", "gt": ">", "<": "<",
            "lt": "<", "==": "==", "=": "==", "eq": "==", "!=": "!=", "ne": "!=", "table": "table", "exists": "exists"}
_CALL = re.compile(r"\b([a-z_][a-z0-9_]*)\(")


# ---------------------------------------------------------------- clause -> Rule
def clause_spec(clause: Clause) -> dict:
    return {k: copy.deepcopy(getattr(clause, k)) for k in SPEC_KEYS}


def selected(graph: ClauseGraph, retrieved: list[str], cfg: dict) -> list[Clause]:
    """The clauses a plugin compiles: every clause with cfg `all: true`, else the retrieved ids, in graph order."""
    ids = set(retrieved)
    return [c for c in graph.clauses if cfg.get("all") or c.id in ids]


def gaps(alignment: Alignment) -> set[str]:
    return {r.term for r in alignment.rows if r.confidence == 0.0}


def terms(spec: dict, clause: Clause) -> set[str]:
    """The vocabulary a rule rests on, as alignment.py lists it: selection values, applicability / requirement predicates, attributes,
    inputs, table_args predicates, exception attributes and the clause's definition keys."""
    req = spec.get("requirement", {})
    out = {v for vs in spec.get("selection", {}).values() for v in vs} | set(spec.get("exceptions", [])) | set(clause.definitions)
    out |= {p for a in spec.get("applicability", []) for p in _CALL.findall(str(a))}
    out |= {str(req[key]) for key in ("predicate", "attribute") if req.get(key) and req[key] != "exists"}
    out |= {str(x) for key in ("predicates", "attributes", "inputs") for x in req.get(key, [])}
    out |= {p for v in req.get("table_args", {}).values() for p in _CALL.findall(str(v))}
    return out


def rule_of(clause: Clause, spec: dict, plugin: str) -> Rule:
    req = spec.get("requirement", {})
    first = next(iter(spec.get("selection", {})), None)   # spec convention: a bare attribute in applicability = that attribute of the first variable
    spec = {**spec, "applicability": [a if "(" in str(a) or first is None else f"{a}({first})" for a in spec.get("applicability", [])]}
    rid, T = asp.rule_id(clause.id), req.get("threshold")
    return Rule(rule_id=rid, version=VERSION, clause=clause.id, standard=clause.standard, edition=clause.edition,
                rule_class=RULE_CLASS[clause.rule_class], source_text=clause.paraphrase, spec=spec, inputs=sorted(req.get("inputs", [])),
                thresholds={rid: float(T)} if asp.numeric(T) else {}, provenance={"plugin": plugin, "clause": clause.id})


def finish(rule: Rule, clause: Clause, signature: Signature, gap_terms: set[str]) -> Rule:
    """Sets asp / status / unsupported_reason: refused (not photo-checkable, semantic without a bool predicate, unrenderable, signature
    problems), vocabulary_gap (a term with a 0.0 alignment row), needs_input (table / formula / inputs), else compiled."""
    req = rule.spec.get("requirement", {})
    pred = signature.predicates.get(str(req.get("predicate")))
    gap = sorted(terms(rule.spec, clause) & gap_terms)
    if not clause.photo_checkable:
        rule.status, rule.unsupported_reason = "refused", "clause is not photo-checkable"
    elif gap:
        rule.status, rule.unsupported_reason = "vocabulary_gap", "no Signature entry for: " + ", ".join(gap)
    elif clause.rule_class in ("semantic", "procedural") and not (pred and pred.unit == "bool"):
        rule.status, rule.unsupported_reason = "refused", f"{clause.rule_class} clause without a bool predicate"
    else:
        rule.asp = rule.asp or asp.render(rule)
        problems = ([asp.unsupported(rule)] if not rule.asp else
                    signature_check(RulePack(pack_id="check", version="0", signature_version=signature.version, rules=[rule]), signature))
        if problems:
            rule.asp, rule.status, rule.unsupported_reason = "", "refused", "; ".join(problems)
        else:
            rule.status = "needs_input" if asp.needs_input(req) else "compiled"
    return rule


def pack(pack_id: str, signature: Signature, rules: list[Rule]) -> RulePack:
    return RulePack(pack_id=pack_id, version=VERSION, signature_version=signature.version, common_asp=asp.COMMON_ASP,
                    rules=sorted(rules, key=lambda r: r.rule_id))


# ---------------------------------------------------------------- LLM output schema (structured outputs: no free-form dicts, so typed pairs)
class Var(BaseModel):
    var: str
    classes: list[str]


class Binding(BaseModel):
    symbol: str
    input: str


class Requirement(BaseModel):
    predicate: str
    args: list[str]
    operator: str
    threshold: float | None
    table: str | None
    formula: str | None
    unit: str | None
    inputs: list[str]
    bindings: list[Binding]


class Spec(BaseModel):
    selection: list[Var]
    applicability: list[str]
    requirement: Requirement
    exceptions: list[str]


class Param(BaseModel):
    name: str
    value: float | bool | str


class Test(BaseModel):
    cell_params: list[Param]
    expected: Literal["PASS", "FAIL", "NEEDS_MEASUREMENT", "NEEDS_INPUT", "CANNOT_DETERMINE"]


class Translation(BaseModel):
    spec: Spec
    asp: str
    needs: list[str]
    refuse_reason: str | None


class Synthesis(Translation):
    tests: list[Test]


def spec_dict(spec: Spec, needs: list[str] = ()) -> dict:
    """Pydantic output -> the spec convention, normalised: sorted class lists, canonical operator, float threshold, sorted inputs."""
    r = spec.requirement
    req: dict[str, Any] = {"predicate": r.predicate, "args": list(r.args), "operator": OPERATOR.get(r.operator.strip(), r.operator.strip()),
                           "unit": r.unit, "inputs": sorted(set(r.inputs) | set(needs))}
    if r.threshold is not None:
        req["threshold"] = float(r.threshold)
    if r.table:
        req["table"] = r.table
    if r.formula:
        req["formula"] = r.formula
    if r.bindings:
        req["bindings"] = {b.symbol: b.input for b in sorted(r.bindings, key=lambda b: b.symbol)}
    return {"selection": {v.var: sorted(set(v.classes)) for v in spec.selection}, "applicability": list(spec.applicability),
            "requirement": req, "exceptions": sorted(set(spec.exceptions))}


# ---------------------------------------------------------------- prompts
CELL_PARAMS = ("fence_height_m (default 2.0): top of the four fence segments", "floor_gap_m (0.10): bottom of the fences above the floor",
               "robot_fixed_gap_m (1.0): gap between the robot box and the fence ring", "lc_bottom_m (0.25): bottom of the light curtain 'lc' (top = bottom + 1.8)",
               "enclosed (true): false opens the east fence by opening_width_m (0.8)", "coverage ('full' | 'none'): floor coverage of the scene",
               "sigma_m (0.01): 1-sigma of every box dimension, U = 2 sigma", "declared_inputs ({}): Scene.declared_inputs, e.g. stop_time_ms")
EXAMPLE = json.dumps({
    "spec": {"selection": [{"var": "F", "classes": ["fence", "guard"]}], "applicability": ["perimeter_of(F, Z)"],
             "requirement": {"predicate": "floor_gap", "args": ["F"], "operator": "<=", "threshold": 180, "table": None, "formula": None,
                             "unit": "mm", "inputs": [], "bindings": []}, "exceptions": []},
    "asp": "", "needs": [], "refuse_reason": None,
    "tests": [{"cell_params": [{"name": "floor_gap_m", "value": 0.10}], "expected": "PASS"},
              {"cell_params": [{"name": "floor_gap_m", "value": 0.25}], "expected": "FAIL"},
              {"cell_params": [{"name": "floor_gap_m", "value": 0.185}], "expected": "NEEDS_MEASUREMENT"}]}, indent=1)


def system_prompt(signature: Signature, with_tests: bool) -> str:
    preds = "\n".join(f"  {name}({', '.join(p.args)}) unit={p.unit}" + (f": {p.note}" if p.note else "") for name, p in signature.predicates.items())
    return f"""You translate one safety clause into an engine-neutral check over a typed scene vocabulary (the Signature). Answer only through the schema.

Spec convention:
  selection      variables (capitalised) over Signature classes or zone kinds; the engine binds every object of those classes
  applicability  atoms pred(Var, Var2) that must hold as facts; an unbound capitalised argument is bound by the fact
  requirement    predicate(args) operator threshold | table id | formula, unit; inputs = declared inputs the check needs (NEEDS_INPUT until declared)
  exceptions     attributes that switch the check off for an object
Operators: >= <= > < (numeric, integer millimetres), == 1 on a bool predicate. A clause that cannot be checked from the scene (colour,
procedure, 'exists' quantification, attributes the Signature lacks) is refused with refuse_reason. Never invent vocabulary.
Decision rule of the engines: guard band with U (k=2): PASS only when the whole interval V -+ U satisfies the operator, FAIL only when the whole
interval violates it, NEEDS_MEASUREMENT in between; no fact -> CANNOT_DETERMINE; undeclared input -> NEEDS_INPUT.

Signature classes: {', '.join(signature.classes)}
Zone kinds: {', '.join(signature.zones)} (only hazard_zone exists as a fact today: the robot's footprint)
Predicates:
{preds}
Attributes: {', '.join(signature.attributes)}
Declared inputs: {', '.join(signature.declared_inputs)}

Synthetic test cell (scenes.cell): a 1 x 1 x 2 m robot at the origin inside a ring of four fences (fence_e, fence_w, fence_n, fence_s), two bollards
and one light curtain 'lc' east of the ring (further than 1.5 m from the robot), one cart west of it. On this cell every fence, guard and bollard
is perimeter_of(X, hazard_zone) and the light curtain covers_opening(lc, C). Parameters:
  {chr(10).join('  ' + p for p in CELL_PARAMS)}
{"tests: 2-4 cells with the status every verdict of this rule must have on that cell (a cell without a verdict fails the test)." if with_tests else ""}
Worked example (ISO 13857:2019 4.4, floor gap at most 180 mm):
{EXAMPLE}
"""


def clause_prompt(clause: Clause, graph: ClauseGraph, requirement: bool) -> str:
    parts = [f"id: {clause.id}", f"title: {clause.title}", f"standard: {clause.standard} {clause.edition}", f"rule_class: {clause.rule_class}",
             f"tags: {', '.join(clause.tags)}", "definitions: " + "; ".join(f"{d} = {', '.join(graph.definitions.get(d, [])) or '(no Signature alias)'}" for d in clause.definitions)]
    if requirement:
        parts += [f"paraphrase: {clause.paraphrase}", "requirement as given: " + json.dumps(clause.requirement, sort_keys=True),
                  "selection as given: " + json.dumps(clause.selection, sort_keys=True), "applicability as given: " + json.dumps(clause.applicability),
                  "exceptions as given: " + json.dumps(clause.exceptions)]
    else:
        parts.append("Reconstruct the requirement from the clause number, title, definitions and tags alone; the structured requirement is withheld.")
    return "\n".join(parts) + "\n"


# ---------------------------------------------------------------- synthetic-cell checks
def synthetic_facts(scene: Scene) -> Facts:
    """synthetic-facts@1 plus the applicability facts the synthetic cell has by construction: every fixed object is on the hazard zone's
    perimeter, the light curtain covers the (east) opening. L2 computes neither yet; without them no rule with such an applicability binds."""
    fx = synth_facts.facts_of(scene)
    for o in scene.objects:
        if o.cls in synth_facts.FIXED:
            fx.facts.append(Fact(pred="perimeter_of", args=[o.id, asp.IMPLICIT_ZONE], unit="bool", value=1))
        if o.cls == "light_curtain":
            fx.facts.append(Fact(pred="covers_opening", args=[o.id, "c_east"], unit="bool", value=1))
    return fx


def engine(name: str, workdir: Path):
    """An L6 engine through the registry (a plugin may not import another layer); clingo writes its program.lp under workdir."""
    Path(workdir).mkdir(parents=True, exist_ok=True)
    return engine_of(plugins.get("L6", name), {"k": 2}, Path(workdir))


def run_tests(rule: Rule, tests: list[Test], engines: list[str], workdir: Path) -> str:
    """The model's own tests on synthetic cells under the given engines; "" when all pass, else the first failure."""
    pk = RulePack(pack_id="test", version="0", signature_version="1", common_asp=asp.COMMON_ASP, rules=[rule])
    for t in tests:
        params = {p.name: p.value for p in t.cell_params}
        try:
            scene = scenes.cell(**params)
        except (TypeError, ValueError) as e:
            return f"tests: cell {params} is not a valid synthetic cell ({e})"
        fx = synthetic_facts(scene)
        for name in engines:
            got = sorted({v.status for v in engine(name, workdir)(fx, pk, scene).verdicts if v.rule_id == rule.rule_id})
            if got != [t.expected]:
                return f"tests ({name}): cell {params} expected {t.expected}, got {got or 'no verdict'}"
    return ""
