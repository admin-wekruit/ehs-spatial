"""Threshold grid (docs/research/verdict-evaluation-protocol-2026-10-08.md): for every numeric rule of a pack, synthetic cells whose measured
quantity sits at threshold +- eps, +- 2 sigma, +- 5 sigma, with the status the guard band (k = 2, U = 2 sigma) must return.

sigma is the 1-sigma of the FACT (U / 2 as synthetic-facts@1 computes it from the object sigmas), so +-2 sigma lands exactly on the
guard-band boundary V -+ U = T: the inclusive side (V - U = T for >=) is PASS, the other side (V + U = T) is still NEEDS_MEASUREMENT
because FAIL needs V + U < T strictly (rules.lp). Offsets are signed toward compliance (+ = further into the allowed side).
Rules whose (predicate, class) has no scene parameter are reported as skipped, not silently dropped.
"""
from __future__ import annotations

from ehs_spatial.verdict.contracts import RulePack
from ehs_spatial.verdict.synth import facts as F
from ehs_spatial.verdict.synth import scenes

PARAM = {("top_height", "fence"): "fence_height_m", ("bottom_height", "fence"): "floor_gap_m", ("floor_gap", "fence"): "floor_gap_m",
         ("bottom_height", "light_curtain"): "lc_bottom_m", ("floor_gap", "light_curtain"): "lc_bottom_m",
         ("min_distance_3d", "fence"): "robot_fixed_gap_m"}
# label, offset in sigma toward compliance, expected status for inclusive (>=, <=) operators, expected for strict (>, <) operators
OFFSETS = [("+eps", 0.2, "NEEDS_MEASUREMENT", "NEEDS_MEASUREMENT"), ("-eps", -0.2, "NEEDS_MEASUREMENT", "NEEDS_MEASUREMENT"),
           ("+2s", 2.0, "PASS", "NEEDS_MEASUREMENT"), ("-2s", -2.0, "NEEDS_MEASUREMENT", "FAIL"),
           ("+5s", 5.0, "PASS", "PASS"), ("-5s", -5.0, "FAIL", "FAIL")]
SYMMETRIC = {"min_distance_3d", "horizontal_gap", "z_overlap", "line_of_sight"}


def parameter(rule):
    """(index of the varied argument, its class, scene parameter) or None when the rule has no synthetic knob."""
    req, sel = rule.spec.get("requirement", {}), rule.spec.get("selection", {})
    for i, var in enumerate(req.get("args", [])):
        for cls in sel.get(var, []):
            if (req.get("predicate"), cls) in PARAM:
                return i, cls, PARAM[(req["predicate"], cls)]
    return None


def fact_u(facts, scene, pred, idx, cls):
    classes = {o.id: o.cls for o in scene.objects}
    for f in facts.facts:
        if f.pred != pred or f.u is None:
            continue
        if classes.get(f.args[idx]) == cls or (pred in SYMMETRIC and len(f.args) == 2 and classes.get(f.args[1 - idx]) == cls):
            return f.u
    raise LookupError(f"no {pred} fact on a {cls} in the nominal scene")


def cases(pack: RulePack, sigma_m: float = 0.01) -> list[dict]:
    out = []
    for rule in pack.rules:
        req = rule.spec.get("requirement", {})
        op, T = req.get("operator"), req.get("threshold")
        if op not in (">=", "<=", ">", "<") or not isinstance(T, (int, float)):
            continue
        hit = parameter(rule)
        if hit is None:
            out.append({"rule_id": rule.rule_id, "skipped": f"no synthetic parameter for {req.get('predicate')} on {rule.spec.get('selection')}"})
            continue
        idx, cls, param = hit
        sign = 1 if op in (">=", ">") else -1
        nominal = scenes.cell(sigma_m=sigma_m, **{param: T / 1000})
        u = fact_u(F.facts_of(nominal), nominal, req["predicate"], idx, cls)
        for label, off, inclusive, strict in OFFSETS:
            v = T + sign * off * u / 2
            scene = scenes.cell(sigma_m=sigma_m, scene_id=f"grid-{rule.rule_id}-{label}", **{param: v / 1000})
            out.append({"rule_id": rule.rule_id, "offset": label, "expected": inclusive if op in (">=", "<=") else strict, "arg_index": idx,
                        "subject_cls": cls, "value_mm": v, "u_mm": u, "threshold": T, "scene": scene, "facts": F.facts_of(scene)})
    return out
