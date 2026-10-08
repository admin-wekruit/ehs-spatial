"""The trial rule pack (research/verdict-layer-trial-2026-10-07/rules.lp, 2026-10-07) as engine-neutral Rule.spec entries.

Reference pack of the synth tests and the python-engine parity test. The lab's reviewed pack is L5's (layers/l5_rules/handwritten.py);
this one deliberately carries no ASP text. Thresholds are the trial's numbers (docs/research/verdict-layer-rules-2026-10-07.md section C),
unverified against the purchased texts.

Spec convention (shared with L5 plugins):
  selection      {Var: [classes or zone kinds]}            Var names start with a capital letter
  applicability  ["pred(Var, Var2)"]                        every atom must hold as a fact; unbound capitalised args are bound by the fact;
                                                            a bare attribute name means attr(FirstVar) (L5 rewrites it before the engines)
  requirement    {predicate, args: [Var...], operator, threshold | table | formula, unit, inputs: [declared inputs]}
                 formula symbols are declared-input names, or mapped through requirement['bindings'] = {symbol: declared input}
  exceptions     [attributes]                               a truthy attr(X) fact or Scene attribute on a bound object switches the rule off
"""
from __future__ import annotations

from ehs_spatial.verdict.contracts import Rule, RulePack

FIXED = ["fence", "guard", "bollard", "light_curtain"]


def rule(rule_id, clause, standard, edition, rule_class, selection, requirement, inputs=(), status="compiled", text=""):
    return Rule(rule_id=rule_id, version="0", clause=clause, standard=standard, edition=edition, rule_class=rule_class, source_text=text,
                spec={"selection": selection, "applicability": [], "requirement": requirement, "exceptions": []},
                inputs=list(inputs), thresholds={} if "threshold" not in requirement else {rule_id: requirement["threshold"]}, status=status,
                provenance={"plugin": "synth.pack@1", "origin": "research/verdict-layer-trial-2026-10-07/rules.lp"})


def num(predicate, args, operator, threshold):
    return {"predicate": predicate, "args": args, "operator": operator, "threshold": threshold, "unit": "mm", "inputs": []}


def trial_pack() -> RulePack:
    rules = [
        rule("fence_height", "ISO 13857:2019 Table 2 note", "ISO 13857", "2019", "geometry", {"F": ["fence"]},
             num("top_height", ["F"], ">=", 1400), text="structures under 1400 mm are not used without additional measures"),
        rule("floor_gap", "ISO 13857:2019 4.4", "ISO 13857", "2019", "geometry", {"F": ["fence", "guard"]},
             num("bottom_height", ["F"], "<=", 180), text="gap under a guard <= 180 mm (slot)"),
        rule("lc_lowest_beam", "ISO 13855:2010", "ISO 13855", "2010", "geometry", {"L": ["light_curtain"]},
             num("bottom_height", ["L"], "<=", 300), text="lowest beam / field height; above 300 mm crawl-under must be assessed"),
        rule("crush_gap", "ISO 13854:2017 Table 1 (body); ISO 10218-2 clause unverified", "ISO 13854", "2017", "geometry",
             {"R": ["robot"], "X": FIXED}, num("min_distance_3d", ["R", "X"], ">=", 500), text="robot to fixed structure >= 500 mm"),
        rule("reach_over", "ISO 13857:2019 Table 2", "ISO 13857", "2019", "geometry", {"H": ["robot"], "S": FIXED},
             {"predicate": "reach_over", "args": ["H", "S"], "operator": ">=", "table": "ISO13857:2019/Table2", "unit": "mm",
              "inputs": ["table_lookup"]}, inputs=["table_lookup"], status="needs_input",
             text="horizontal distance c vs the table of hazard height a and structure height b; table not encoded until verified"),
        rule("enclosure", "topology: no outside -> hazard path through unblocked observed cells", "lab", "0", "topology",
             {"Z": ["hazard_zone"]}, {"predicate": "enclosed", "args": ["Z"], "operator": "==", "threshold": 1, "unit": "bool", "inputs": []}),
    ]
    return RulePack(pack_id="trial-v0", version="0", signature_version="1", rules=rules)
