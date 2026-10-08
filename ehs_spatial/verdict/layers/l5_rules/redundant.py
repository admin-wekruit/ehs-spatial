"""L5 redundant-translation@0: two independent translations of every clause must agree (ARc-style redundancy), then differential
execution. Per retrieved clause TWO llm.complete calls with deliberately different framings: A sees the clause's structured requirement
and paraphrase; B sees only the clause number, title, definitions and tags and reconstructs the requirement from the words (no sampling
parameters: Haiku 5.5 rejects non-default temperature). Both specs are normalised (common.spec_dict: sorted class lists, canonical
operator, float threshold, sorted inputs) and compared field by field: a disagreement refuses the rule ("redundant translations disagree:
<fields>", both candidates in rule.review); agreement renders the ASP (common.finish), then verify.differential of the two candidate packs
on four synthetic cells under the python engine must be empty (the ARc check; it is when the specs are equal) -> compiled / needs_input.
llm_calls counts uncached calls. cfg: model, effort, cache_dir (default <runs>/llm-cache), all."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ehs_spatial.verdict import llm
from ehs_spatial.verdict.layers.l5_rules import common, verify
from ehs_spatial.verdict.plugins import register, tag
from ehs_spatial.verdict.synth import scenes

CELLS = lambda: [scenes.cell(), scenes.cell(enclosed=False), scenes.cell(floor_gap_m=0.2), scenes.cell(fence_height_m=1.2)]


def disagreement(a: dict, b: dict) -> list[str]:
    fields = [k for k in common.SPEC_KEYS if k != "requirement" and a[k] != b[k]]
    ra, rb = a["requirement"], b["requirement"]
    return fields + [f"requirement.{k}" for k in sorted(set(ra) | set(rb)) if ra.get(k) != rb.get(k)]


@register("L5", "redundant-translation", "0")
class Redundant:
    def run(self, inputs: dict[str, Any], cfg: dict[str, Any], workdir: Path) -> dict[str, Any]:
        signature, graph, gap = inputs["signature"], inputs["clauses"], common.gaps(inputs["alignment"])
        cache_dir = Path(cfg.get("cache_dir") or Path(workdir).parents[1] / "llm-cache")
        system = common.system_prompt(signature, with_tests=False)
        rules, calls = [], 0
        for clause in common.selected(graph, inputs["retrieved"], cfg):
            outs = []
            for framing in (True, False):
                out, meta = llm.complete(system, common.clause_prompt(clause, graph, requirement=framing), common.Translation,
                                         model=cfg.get("model"), effort=cfg.get("effort"), cache_dir=cache_dir)
                calls += 0 if meta["cached"] else 1
                outs.append(out)
            a, b = outs
            specs = [common.spec_dict(o.spec, o.needs) for o in outs]
            rule = common.rule_of(clause, specs[0], tag(self))
            rule.provenance["model"] = meta["model"]
            rule.review = {"candidate_a": json.dumps(a.model_dump(mode="json"), sort_keys=True), "candidate_b": json.dumps(b.model_dump(mode="json"), sort_keys=True)}
            if a.refuse_reason or b.refuse_reason:
                rule.status, rule.unsupported_reason = "refused", "model refused: " + " | ".join(r for r in (a.refuse_reason, b.refuse_reason) if r)
            elif fields := disagreement(*specs):
                rule.status, rule.unsupported_reason = "refused", "redundant translations disagree: " + ", ".join(fields)
            else:
                common.finish(rule, clause, signature, gap)
                if rule.status in ("compiled", "needs_input"):
                    spec_b = {**specs[1], "selection": {v: specs[1]["selection"][v] for v in specs[0]["selection"]}}   # A's variable order = same subjects
                    other = common.rule_of(clause, spec_b, tag(self))
                    other.asp, other.status = rule.asp, rule.status
                    diff = verify.differential(common.pack("a", signature, [rule]), common.pack("b", signature, [other]), CELLS(),
                                               common.engine("python", workdir), common.synthetic_facts)
                    if diff:
                        rule.asp, rule.status, rule.unsupported_reason = "", "refused", f"differential: {diff[0]}"
            rules.append(rule)
        return {"rule_pack": common.pack("redundant", signature, rules), "llm_calls": calls}
