"""L5 code-synthesis@0: the LLM writes the check, deterministic tests decide (TUM ACC / CodeAct pattern: generate, execute, keep only what
passes). Per retrieved clause ONE llm.complete call (system = spec convention + Signature vocabulary + synthetic-cell parameters + one
worked example; user = the clause) returns common.Synthesis: spec, optional asp, the model's own tests (synthetic cell parameters ->
expected status), declared needs, or a refuse_reason. Pipeline, all deterministic: common.finish (render when the model gave no ASP,
signature check, status) -> the model's tests through the python engine (and clingo when the model wrote the ASP itself) on
common.synthetic_facts cells -> verify.threshold_grid when numeric -> verify.metamorphic rigid + inflate on scenes.cell(). The first
failing check refuses the rule with its name in unsupported_reason; the model's output stays in rule.review. llm_calls counts uncached
calls. cfg: model, effort, cache_dir (default <runs>/llm-cache), all (every clause, not only the retrieved)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ehs_spatial.verdict import llm
from ehs_spatial.verdict.layers.l5_rules import common, verify
from ehs_spatial.verdict.plugins import register, tag
from ehs_spatial.verdict.synth import grid, scenes


def check(rule, out: common.Synthesis, signature, workdir: Path) -> str:
    """"" when every deterministic check passes, else "<check>: <detail>"."""
    if rule.status not in ("compiled", "needs_input"):
        return ""
    failed = common.run_tests(rule, out.tests, ["python"] + (["clingo"] if out.asp.strip() else []), workdir)
    if failed:
        return failed
    py = common.engine("python", workdir)
    pk = common.pack("check", signature, [rule])
    if rule.status == "compiled":
        cases = grid.cases(pk)
        for c in cases:
            if "scene" in c:
                c["facts"] = common.synthetic_facts(c["scene"])
        bad = verify.threshold_grid(pk, py, cases)["mismatches"]
        if bad:
            return f"threshold_grid: {bad[0]['offset']} expected {bad[0]['expected']}, got {bad[0]['actual']}"
    rep = verify.metamorphic(pk, [scenes.cell()], py, common.synthetic_facts)
    for relation in ("rigid", "inflate_u"):   # the lead's two relations; delete / monotone are computed but not gating
        if rep[relation]:
            v = rep[relation][0]
            return f"metamorphic {relation}: {v['relation']} turned {v['before']} into {v['after']}"
    return ""


@register("L5", "code-synthesis", "0")
class CodeSynthesis:
    def run(self, inputs: dict[str, Any], cfg: dict[str, Any], workdir: Path) -> dict[str, Any]:
        signature, graph, gap = inputs["signature"], inputs["clauses"], common.gaps(inputs["alignment"])
        cache_dir = Path(cfg.get("cache_dir") or Path(workdir).parents[1] / "llm-cache")
        system = common.system_prompt(signature, with_tests=True)
        rules, calls = [], 0
        for clause in common.selected(graph, inputs["retrieved"], cfg):
            out, meta = llm.complete(system, common.clause_prompt(clause, graph, requirement=True), common.Synthesis,
                                     model=cfg.get("model"), effort=cfg.get("effort"), cache_dir=cache_dir)
            calls += 0 if meta["cached"] else 1
            rule = common.rule_of(clause, common.spec_dict(out.spec, out.needs), tag(self))
            rule.provenance["model"] = meta["model"]
            rule.review = {"llm": json.dumps(out.model_dump(mode="json"), sort_keys=True)}
            if out.refuse_reason:
                rule.status, rule.unsupported_reason = "refused", f"model refused: {out.refuse_reason}"
            else:
                rule.asp = out.asp.strip()
                failed = check(common.finish(rule, clause, signature, gap), out, signature, workdir)
                if failed:
                    rule.asp, rule.status, rule.unsupported_reason = "", "refused", failed
            rules.append(rule)
        return {"rule_pack": common.pack("codegen", signature, rules), "llm_calls": calls}
