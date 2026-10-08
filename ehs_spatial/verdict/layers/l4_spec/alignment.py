"""Alignment of the clause graph's terms to the Signature: exact names 1.0, aliases from `definitions` 0.8, no match 0.0 with an
empty target. No embeddings, no LLM: the 0.0 rows are the vocabulary gaps a reviewer (or a later aligner plugin) must close."""
from __future__ import annotations

import re

from ehs_spatial.verdict.contracts import Alignment, AlignmentRow, ClauseGraph, Signature

_CALL = re.compile(r"\b([a-z_][a-z0-9_]*)\(")   # predicate names in 'perimeter_of(F, Z)' / 'top_height(H)'
_KINDS = ("class", "zone", "predicate", "attribute", "input")


def _norm(term: str) -> str:
    return re.sub(r"[\s\-]+", "_", term.strip().lower())


def clause_terms(graph: ClauseGraph) -> set[tuple[str, str]]:
    """(term, kind-as-used) pairs: selection values, applicability predicates, requirement predicate / attribute(s) / predicates /
    table_args predicates / inputs, exception attributes, definitions keys and aliases (kind 'class' until the Signature says
    otherwise). Formula strings are not parsed."""
    terms: set[tuple[str, str]] = set()
    for c in graph.clauses:
        for values in c.selection.values():
            terms.update((v, "class") for v in values)
        for a in c.applicability:
            terms.update((p, "predicate") for p in _CALL.findall(a))
        r = c.requirement
        if r.get("predicate") and r["predicate"] != "exists":   # "exists" is an operator keyword, not a Signature predicate
            terms.add((str(r["predicate"]), "predicate"))
        if r.get("attribute"):
            terms.add((str(r["attribute"]), "attribute"))
        terms.update((str(p), "predicate") for p in r.get("predicates", []))
        terms.update((str(a), "attribute") for a in r.get("attributes", []))
        terms.update((str(i), "input") for i in r.get("inputs", []))
        for v in r.get("table_args", {}).values():
            terms.update((p, "predicate") for p in _CALL.findall(str(v)))
        terms.update((e, "attribute") for e in c.exceptions)
    for key, aliases in graph.definitions.items():
        terms.add((key, "class"))
        terms.update((a, "class") for a in aliases)
    return terms


def align(clauses: ClauseGraph, signature: Signature) -> Alignment:
    names = {"class": set(signature.classes), "zone": set(signature.zones), "predicate": set(signature.predicates),
             "attribute": set(signature.attributes), "input": set(signature.declared_inputs)}

    def kind_of(term: str, used_as: str) -> str:   # the kind the term was used as, if the Signature has it there; else first hit
        return used_as if term in names[used_as] else next(k for k in _KINDS if term in names[k])

    def known(term: str) -> bool:
        return any(term in names[k] for k in _KINDS)

    defs = {_norm(k): v for k, v in clauses.definitions.items()}
    rows: list[AlignmentRow] = []
    for term, used_as in sorted(clause_terms(clauses)):
        if known(term):
            rows.append(AlignmentRow(term=term, kind=kind_of(term, used_as), target=term, confidence=1.0, source="exact"))
            continue
        hits = [a for a in defs.get(_norm(term), []) if known(a)]
        if hits:
            rows += [AlignmentRow(term=term, kind=kind_of(a, used_as), target=a, confidence=0.8, source="synonym") for a in hits]
        else:
            rows.append(AlignmentRow(term=term, kind=used_as, target="", confidence=0.0, source=""))
    return Alignment(signature_version=signature.version, rows=rows)


def coverage(alignment: Alignment) -> float:
    """Share of distinct terms that have at least one row above 0.0 (the README's 'alignment coverage')."""
    best: dict[str, float] = {}
    for r in alignment.rows:
        best[r.term] = max(best.get(r.term, 0.0), r.confidence)
    return sum(v > 0 for v in best.values()) / len(best)
