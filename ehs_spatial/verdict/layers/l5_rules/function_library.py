"""L5 function-library@0: the clause graph compiled without an LLM (FuncMapper-style: the typed scene API is the Signature, the clause's
structured `requirement` is the function call). Per retrieved clause one Rule whose spec is the clause's selection / applicability /
requirement / exceptions as L4 wrote them (tables keep their id and operator, inputs stay declared), asp = asp.render(spec), status from
common.finish: compiled (rendered, signature-clean), needs_input (table / formula / inputs), vocabulary_gap (a term with a 0.0 alignment
row), refused (not photo-checkable, semantic without a bool predicate, unrenderable). cfg: all (true = every clause, not only the
retrieved). Deterministic: same graph -> byte-identical pack (rules sorted by rule_id)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ehs_spatial.verdict.layers.l5_rules import common
from ehs_spatial.verdict.plugins import register, tag


@register("L5", "function-library", "0")
class FunctionLibrary:
    def run(self, inputs: dict[str, Any], cfg: dict[str, Any], workdir: Path) -> dict[str, Any]:
        signature, gap = inputs["signature"], common.gaps(inputs["alignment"])
        rules = [common.finish(common.rule_of(c, common.clause_spec(c), tag(self)), c, signature, gap)
                 for c in common.selected(inputs["clauses"], inputs["retrieved"], cfg)]
        return {"rule_pack": common.pack("funclib", signature, rules)}
