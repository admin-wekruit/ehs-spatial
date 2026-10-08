"""L5 handwritten@2: the trial's rule pack (research/verdict-layer-trial-2026-10-07/rules.lp) as a RulePack.

Five rules, each with an engine-neutral `spec` (selection / applicability / requirement / exceptions, plus reviewer notes) and its
ASP text; the reach-over rule declares its missing inputs (NEEDS_INPUT); `common_asp` holds the guard-band decision rules.
L4's clause graph is ignored (hand-written pack). Thresholds are the numbers of docs/research/verdict-layer-rules-2026-10-07.md
section C: verify against the purchased texts before any production use (several are vendor reproductions).
@2 (2026-10-08): the enclosure rule reads the engine's `coverage_known` atom: no Coverage -> cannot_determine (the contract's meaning of
Coverage = None), a breach through observed floor -> fail, a breach only through unobserved floor -> open (cannot_determine).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ehs_spatial.verdict.contracts import Rule, RulePack
from ehs_spatial.verdict.layers.l5_rules.asp import COMMON_ASP   # the decision rules live in asp.py (shared with the synthesis plugins)
from ehs_spatial.verdict.plugins import register

VERSION = "1"                                   # the trial's rules.lp v0 (2026-10-07) + the enclosure coverage reading (2026-10-08)
FIXED = ["fence", "guard", "bollard", "light_curtain"]
ORIGIN = {"origin": "research/verdict-layer-trial-2026-10-07/rules.lp", "plugin": "handwritten@2"}


def threshold_rule(rule_id, clause, standard, edition, selection, requirement, asp, notes=(), source_text="") -> Rule:
    return Rule(rule_id=rule_id, version=VERSION, clause=clause, standard=standard, edition=edition, rule_class="geometry", source_text=source_text,
                spec={"selection": selection, "applicability": [], "requirement": requirement, "exceptions": [], "notes": list(notes)},
                asp=asp, thresholds={rule_id: requirement["threshold"]}, provenance=ORIGIN)


def rules() -> list[Rule]:
    return [
        threshold_rule("fence_height", "ISO 13857:2019 Table 2 note", "ISO 13857", "2019", {"F": ["fence"]},
                       {"predicate": "top_height", "args": ["F"], "operator": ">=", "threshold": 1400, "unit": "mm"},
                       "thr(fence_height, 1400).  dir(fence_height, ge).\n"
                       "subject(fence_height, F) :- obj(F, fence).\n"
                       "meas(fence_height, F, V, U) :- subject(fence_height, F), top(F, V, U).\n",
                       source_text="protective structures lower than 1400 mm are not used without additional safety measures"),
        threshold_rule("floor_gap", "ISO 13857:2019 4.4", "ISO 13857", "2019", {"F": ["fence", "guard"]},
                       {"predicate": "bottom_height", "args": ["F"], "operator": "<=", "threshold": 180, "unit": "mm"},
                       "thr(floor_gap, 180).  dir(floor_gap, le).\n"
                       "subject(floor_gap, F) :- obj(F, fence).\n"
                       "subject(floor_gap, F) :- obj(F, guard).\n"
                       "meas(floor_gap, F, V, U) :- subject(floor_gap, F), bottom(F, V, U).\n",
                       source_text="gap under a guard at most 180 mm (slot opening)"),
        threshold_rule("lc_lowest_beam", "ISO 13855 (2010 numbers)", "ISO 13855", "2010", {"L": ["light_curtain"]},
                       {"predicate": "bottom_height", "args": ["L"], "operator": "<=", "threshold": 300, "unit": "mm"},
                       "thr(lc_lowest_beam, 300).  dir(lc_lowest_beam, le).\n"
                       "subject(lc_lowest_beam, L) :- obj(L, light_curtain).\n"
                       "meas(lc_lowest_beam, L, V, U) :- subject(lc_lowest_beam, L), bottom(L, V, U).\n",
                       source_text="lowest beam / field height; above 300 mm crawling under must be assessed"),
        threshold_rule("crush_gap", "ISO 13854:2017 Table 1; ISO 10218-2 / R15.06 trapping clearance (clause unverified)", "ISO 13854", "2017",
                       {"R": ["robot"], "X": FIXED},
                       {"predicate": "min_distance_3d", "args": ["R", "X"], "operator": ">=", "threshold": 500, "unit": "mm"},
                       "thr(crush_gap, 500).  dir(crush_gap, ge).\n"
                       "subject(crush_gap, (R, X)) :- obj(R, robot), obj(X, C), fixed(C).\n"
                       "meas(crush_gap, (R, X), D, U) :- subject(crush_gap, (R, X)), dist(R, X, D, U).\n",
                       notes=["robot box = the pose in the photos, not the motion envelope; the real verdict needs the restricted space"],
                       source_text="minimum gap to avoid crushing the body: 500 mm"),
        Rule(rule_id="reach_over", version=VERSION, clause="ISO 13857:2019 Table 2", standard="ISO 13857", edition="2019", rule_class="geometry",
             source_text="reaching over a protective structure: hazard height a, structure height b, horizontal distance c -> table lookup",
             spec={"selection": {"H": ["robot"], "S": FIXED}, "applicability": [], "exceptions": [],
                   "requirement": {"predicate": "reach_over", "args": ["H", "S"], "operator": "table", "table": "ISO13857:2019/Table2",
                                   "unit": "mm", "inputs": ["risk_level", "table_lookup"]},
                   "notes": ["the three inputs are measured; the Table 2 lookup and the low / high risk choice are not verified against "
                             "the purchased text -> no verdict"]},
             asp="subject(reach_over, (H, S)) :- reach_over(H, S, _, _, _, _).\n"
                 "status(reach_over, (H, S), needs_input) :- reach_over(H, S, _, _, _, _).\n",
             inputs=["table_lookup", "risk_level"], status="needs_input",
             unsupported_reason="ISO 13857 Table 2 is not encoded until verified against the purchased text", provenance=ORIGIN),
        Rule(rule_id="enclosure", version=VERSION, clause="topology: no path from outside to a hazard cell through unblocked cells",
             standard="panoptes-lab", edition="2026-10-07", rule_class="topology",
             source_text="the hazard zone is enclosed by protective structures",
             spec={"selection": {"Z": ["hazard_zone"]}, "applicability": [], "exceptions": [],
                   "requirement": {"predicate": "enclosed", "args": ["Z"], "operator": "==", "threshold": 1, "unit": "bool"},
                   "notes": ["hazard zone = plan footprint of the robot pose box, not the restricted space (needs the controller configuration)"]},
             asp="reach(C)  :- outside(C), not blocked(C).\n"
                 "reach(C2) :- reach(C1), adj(C1, C2), not blocked(C2).\n"
                 "breach(C) :- hazard(C), reach(C).\n"
                 "opening(C) :- reach(C), adj(C, C2), hazard(C2), not hazard(C).\n"
                 "seen(C)  :- outside(C), not blocked(C), observed(C).\n"
                 "seen(C2) :- seen(C1), adj(C1, C2), not blocked(C2), observed(C2).\n"
                 "breach_seen(C) :- hazard(C), seen(C).\n"
                 "status(enclosure, hazard_zone, fail)             :- breach_seen(_).\n"
                 "status(enclosure, hazard_zone, open)             :- breach(_), not breach_seen(_).\n"
                 "status(enclosure, hazard_zone, pass)             :- hazard(_), not breach(_), coverage_known.\n"
                 "status(enclosure, hazard_zone, cannot_determine) :- hazard(_), not breach(_), not coverage_known.\n"
                 "status(enclosure, hazard_zone, cannot_determine) :- not hazard(_).\n",
             provenance=ORIGIN),
    ]


@register("L5", "handwritten", "2")
class Handwritten:
    def run(self, inputs: dict[str, Any], cfg: dict[str, Any], workdir: Path) -> dict[str, Any]:
        return {"rule_pack": RulePack(pack_id="handwritten-trial", version=VERSION, signature_version=inputs["signature"].version,
                                      common_asp=COMMON_ASP, rules=rules())}
