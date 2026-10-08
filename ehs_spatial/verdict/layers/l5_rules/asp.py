"""Shared ASP of the L5 plugins: COMMON_ASP (the guard-band decision rules, inclusive and strict operators) and render(rule), the
engine-neutral `spec` (synth/pack.py convention) written as clingo@2 rules over the generic atoms the engine renders from Facts:
  <pred>("id", ..., V, U)   every numeric fact under its Signature name (symmetric pair predicates in both orders)
  <pred>("id", ...)         every boolean / valueless fact that holds        obj("id", class)   hazard(C) ... (grid)
Rendered shape, same as handwritten.py: thr/dir, one subject/2 rule per class combination of the selection, one meas/4 rule; the subject
is the selection tuple (one variable: the variable itself; several: (A, B)), so verdict subjects match the python engine's.

Supported: 1-2 selection variables over Signature classes; a zone variable only as the implicit hazard zone (zone kinds are not facts
today: it is rendered as the id "hazard_zone", guarded by hazard(_), the python engine's implicit zone; a declared zone of another kind
is unsupported); applicability atoms name(args) verbatim (unbound capitalised args become `_`, or join the subject tuple when the
requirement uses them); exceptions as `not attr(X)`; a numeric threshold with >= <= > <; `table` / `formula` / inputs -> needs_input;
`== 1` on a bool-unit predicate -> pass when the atom holds, cannot_determine otherwise. Anything else (exists, == 0, attributes,
comparisons inside applicability, non-integer thresholds) is unsupported: render returns "" and unsupported(rule) says why.
Rule ids in ASP must be constants: rule_id("ISO13857:2019/4.4") = "iso13857_2019_4_4"; the clause id stays in Rule.clause.
"""
from __future__ import annotations

import itertools
import re
from typing import get_args

from ehs_spatial.verdict.contracts import Rule, ZoneKind

ZONE_KINDS = set(get_args(ZoneKind)) - {"other"}   # 'other' is a class as well: rendered as obj(X, other)
IMPLICIT_ZONE = "hazard_zone"
DIR = {">=": "ge", "<=": "le", ">": "gt", "<": "lt"}
_ATOM = re.compile(r"^\s*(\w+)\s*\(([^)]*)\)\s*$")

COMMON_ASP = """\
% Facts rendered by the engine (integer millimetres, U = guard-band half width):
%   obj(ID, Class).  bottom(ID, V, U).  top(ID, V, U).  dist(A, B, D, U).  reach_over(H, S, A, B, C, U).
%   cell(C).  adj(C1, C2).  blocked(C).  hazard(C).  outside(C).  observed(C).  coverage_known.  untrusted(ID).
fixed(fence).  fixed(guard).  fixed(bollard).  fixed(light_curtain).

% Guard-banded decision (ILAC-G8 / ISO 14253-1 style): a side is taken only when the whole interval V +- U is on it.
status(R, S, pass)       :- meas(R, S, V, U), thr(R, T), dir(R, ge), V - U >= T.
status(R, S, fail)       :- meas(R, S, V, U), thr(R, T), dir(R, ge), V + U <  T.
status(R, S, needs_meas) :- meas(R, S, V, U), thr(R, T), dir(R, ge), V - U <  T, V + U >= T.
status(R, S, pass)       :- meas(R, S, V, U), thr(R, T), dir(R, le), V + U <= T.
status(R, S, fail)       :- meas(R, S, V, U), thr(R, T), dir(R, le), V - U >  T.
status(R, S, needs_meas) :- meas(R, S, V, U), thr(R, T), dir(R, le), V + U >  T, V - U <= T.
status(R, S, cannot_determine) :- subject(R, S), thr(R, _), not meas(R, S, _, _).
margin(R, S, V - T) :- meas(R, S, V, _), thr(R, T), dir(R, ge).
margin(R, S, T - V) :- meas(R, S, V, _), thr(R, T), dir(R, le).

% Strict operators (> and <): the boundary itself is not on the passing side.
status(R, S, pass)       :- meas(R, S, V, U), thr(R, T), dir(R, gt), V - U >  T.
status(R, S, fail)       :- meas(R, S, V, U), thr(R, T), dir(R, gt), V + U <= T.
status(R, S, needs_meas) :- meas(R, S, V, U), thr(R, T), dir(R, gt), V - U <= T, V + U >  T.
status(R, S, pass)       :- meas(R, S, V, U), thr(R, T), dir(R, lt), V + U <  T.
status(R, S, fail)       :- meas(R, S, V, U), thr(R, T), dir(R, lt), V - U >= T.
status(R, S, needs_meas) :- meas(R, S, V, U), thr(R, T), dir(R, lt), V + U >= T, V - U <  T.
margin(R, S, V - T) :- meas(R, S, V, _), thr(R, T), dir(R, gt).
margin(R, S, T - V) :- meas(R, S, V, _), thr(R, T), dir(R, lt).

#show status/3.
#show margin/3.
#show opening/1.
"""


class Unsupported(ValueError):
    pass


def rule_id(clause_id: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", clause_id.lower()).strip("_")


def needs_input(req: dict) -> bool:
    return "table" in req or "formula" in req or bool(req.get("inputs"))


def numeric(threshold) -> bool:
    return isinstance(threshold, (int, float)) and not isinstance(threshold, bool)


def render(rule: Rule) -> str:
    try:
        return _render(rule)
    except Unsupported:
        return ""


def unsupported(rule: Rule) -> str:
    """Why render(rule) returns "" ("" when it is supported)."""
    try:
        _render(rule)
        return ""
    except Unsupported as e:
        return str(e)


def _render(rule: Rule) -> str:
    rid, spec = rule.rule_id, rule.spec
    sel, req = dict(spec.get("selection", {})), dict(spec.get("requirement", {}))
    if not 1 <= len(sel) <= 2:
        raise Unsupported(f"{len(sel)} selection variables (1-2 supported)")
    if not req.get("predicate"):
        raise Unsupported("requirement has no predicate" + (f" (attribute {req['attribute']})" if req.get("attribute") else ""))
    terms: dict[str, str] = {}                      # variable -> ASP term
    alts: list[list[str]] = []                      # per variable: one body atom per class
    for var, classes in sel.items():
        zones = [c for c in classes if c in ZONE_KINDS]
        if zones and (IMPLICIT_ZONE not in zones or len(zones) != len(classes)):
            raise Unsupported(f"zone variable {var} over {classes}: zone kinds are not facts, only the implicit hazard_zone is rendered")
        terms[var] = f'"{IMPLICIT_ZONE}"' if zones else var
        alts.append(["hazard(_)"] if zones else [f"obj({var}, {c})" for c in classes])
    body: list[str] = []
    names = list(sel)
    if len(names) == 2 and set(sel[names[0]]) & set(sel[names[1]]):
        body.append(f"{names[0]} != {names[1]}")   # the python engine never binds two variables to the same object
    req_args = [str(a) for a in req.get("args", [])]
    subject_vars = list(sel)
    for text in spec.get("applicability", []):
        m = _ATOM.match(str(text))
        if not m:
            raise Unsupported(f"applicability {text!r} is not name(args)")
        args = []
        for a in [x.strip() for x in m.group(2).split(",") if x.strip()]:
            if a not in terms and a[:1].isupper() and a in req_args:
                terms[a] = a
                subject_vars.append(a)
            args.append(terms.get(a, "_" if a[:1].isupper() else a))
        body.append(f"{m.group(1)}({', '.join(args)})")
    for attr in spec.get("exceptions", []):
        body += [f"not {attr}({terms[v]})" for v in subject_vars]
    if any(a not in terms for a in req_args):
        raise Unsupported(f"requirement argument not bound by the selection: {[a for a in req_args if a not in terms]}")
    S = terms[subject_vars[0]] if len(subject_vars) == 1 else "(" + ", ".join(terms[v] for v in subject_vars) + ")"
    subjects = "".join(f"subject({rid}, {S}) :- {', '.join(list(combo) + body)}.\n" for combo in itertools.product(*alts))
    call = ", ".join(terms[a] for a in req_args)
    op, T = req.get("operator"), req.get("threshold")
    if needs_input(req):
        return subjects + f"status({rid}, {S}, needs_input) :- subject({rid}, {S}).\n"
    if op in DIR and numeric(T):
        if T != int(T):
            raise Unsupported(f"threshold {T} is not an integer (facts are integer millimetres)")
        return (f"thr({rid}, {int(T)}).  dir({rid}, {DIR[op]}).\n" + subjects
                + f"meas({rid}, {S}, V, U) :- subject({rid}, {S}), {req['predicate']}({call}, V, U).\n")
    if op == "==" and T == 1 and req.get("unit") == "bool":
        return (subjects + f"status({rid}, {S}, pass) :- subject({rid}, {S}), {req['predicate']}({call}).\n"
                + f"status({rid}, {S}, cannot_determine) :- subject({rid}, {S}), not {req['predicate']}({call}).\n")
    raise Unsupported(f"operator {op!r} with threshold {T!r} (unit {req.get('unit')!r}) is not renderable")
