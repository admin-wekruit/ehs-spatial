"""Verification harness for a candidate RulePack, run before any human review (docs/research/verdict-spec-to-check-2026-10-08.md section 3
stage 5; metamorphic relations and threshold grid from docs/research/verdict-evaluation-protocol-2026-10-08.md).

  signature_check(pack, signature)                     every class / zone / predicate (with arity) / attribute / declared input a rule
                                                       references exists in the Signature; variables of the requirement are bound
  differential(pack_a, pack_b, scenes, engine, facts_of)   status disagreements per (scene, rule, subjects) between two packs
  metamorphic(pack, scenes, engine, facts_of)          rigid transform (rotation about the floor normal + translation) keeps every status;
                                                       deleting an object no rule selects keeps every status; inflating every U by 10x never
                                                       flips PASS <-> FAIL (only moves to NEEDS_MEASUREMENT); moving a measured value across
                                                       its threshold from -5 sigma to +5 sigma flips FAIL -> NEEDS_MEASUREMENT -> PASS monotonically
  threshold_grid(pack, engine, cases=None)             runs synth/grid.py cases and reports expected-vs-actual mismatches
  facts_parity(a, b, tol_mm)                           value / U disagreements between two Facts (the real L2 vs synthetic-facts@1)

`engine` is a callable (facts, pack, scene) -> VerdictSet; engine_of(plugin, cfg) wraps an L6 plugin. `facts_of` is a callable Scene -> Facts
(an L2 plugin's run, or ehs_spatial.verdict.synth.facts.facts_of). The harness imports no layer: rotation only needs axis-aligned facts
when facts_of is the synthetic one, so angles default to multiples of 90 degrees; any angle is fine with the real L2.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

from ehs_spatial.verdict.contracts import Facts, RulePack, Scene, Signature, VerdictSet

ORDER = {"FAIL": 0, "NEEDS_MEASUREMENT": 1, "PASS": 2}
OPERATORS = (">=", "<=", ">", "<", "==", "!=")
SYMMETRIC = {"min_distance_3d", "horizontal_gap", "z_overlap", "line_of_sight"}
_ATOM = re.compile(r"^\s*(\w+)\s*\(([^)]*)\)\s*$")


def engine_of(plugin, cfg: dict | None = None, workdir: Path = Path(".")):
    p = plugin() if isinstance(plugin, type) else plugin
    return lambda facts, pack, scene=None: p.run({"facts": facts, "rule_pack": pack, "scene": scene}, cfg or {}, workdir)["verdicts"]


def statuses(vs: VerdictSet) -> dict[tuple[str, tuple[str, ...]], str]:
    return {(v.rule_id, tuple(v.subjects)): v.status for v in vs.verdicts}


# ---------------------------------------------------------------- (a) signature
def signature_check(pack: RulePack, signature: Signature) -> list[str]:
    problems = []
    preds = signature.predicates
    for r in pack.rules:
        where, spec = f"{r.rule_id}@{r.version}", r.spec
        sel = spec.get("selection", {})
        bound = set(sel)
        for var, classes in sel.items():
            for c in classes:
                if c not in signature.classes and c not in signature.zones:
                    problems.append(f"{where}: selection {var}: unknown class/zone {c!r}")
        for text in spec.get("applicability", []):
            m = _ATOM.match(text)
            if not m:
                problems.append(f"{where}: applicability {text!r} is not name(args)")
                continue
            name, args = m.group(1), [a.strip() for a in m.group(2).split(",") if a.strip()]
            if name not in preds:
                problems.append(f"{where}: applicability predicate {name!r} not in signature")
            elif len(args) != len(preds[name].args):
                problems.append(f"{where}: {name} takes {len(preds[name].args)} args, got {len(args)}")
            bound |= {a for a in args if a[:1].isupper()}
        req = spec.get("requirement", {})
        if not req:
            problems.append(f"{where}: no requirement")
            continue
        name, args = req.get("predicate"), list(req.get("args", []))
        if name not in preds:
            problems.append(f"{where}: requirement predicate {name!r} not in signature")
        elif len(args) != len(preds[name].args):
            problems.append(f"{where}: {name} takes {len(preds[name].args)} args, got {len(args)}")
        for a in args:
            if a[:1].isupper() and a not in bound:
                problems.append(f"{where}: requirement variable {a} is bound by neither selection nor applicability")
        if req.get("operator") not in OPERATORS and not (req.get("operator") == "table" and "table" in req):
            problems.append(f"{where}: operator {req.get('operator')!r} not in {OPERATORS} (or 'table' with a table id)")
        if not any(k in req for k in ("threshold", "table", "formula")):
            problems.append(f"{where}: requirement has no threshold, table or formula")
        for i in list(req.get("inputs", [])) + list(r.inputs) + list(req.get("bindings", {}).values()):
            if i not in signature.declared_inputs:
                problems.append(f"{where}: input {i!r} not a declared input of the signature")
        for a in spec.get("exceptions", []):
            if a not in signature.attributes:
                problems.append(f"{where}: exception attribute {a!r} not in signature")
    return problems


# ---------------------------------------------------------------- (b) differential
def differential(pack_a: RulePack, pack_b: RulePack, scenes, engine, facts_of) -> list[dict]:
    out = []
    for scene in scenes:
        facts = facts_of(scene)
        sa, sb = statuses(engine(facts, pack_a, scene)), statuses(engine(facts, pack_b, scene))
        for key in sorted(set(sa) | set(sb)):
            if sa.get(key) != sb.get(key):
                out.append({"scene_id": scene.scene_id, "rule_id": key[0], "subjects": list(key[1]), "a": sa.get(key), "b": sb.get(key)})
    return out


# ---------------------------------------------------------------- (c) metamorphic
def transform(scene: Scene, angle_deg: float, shift_xy=(0.0, 0.0)) -> Scene:
    """Rigid motion: rotate about the floor normal (must be +z) then translate in the plan; objects, coverage and zones move together."""
    if not all(abs(a - b) < 1e-9 for a, b in zip(scene.ground_normal, (0.0, 0.0, 1.0))):
        raise ValueError("transform assumes ground_normal = +z")
    c, s = math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))
    rot = lambda v: [c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]]
    t = [shift_xy[0], shift_xy[1], 0.0]
    new = scene.model_copy(deep=True)
    for o in new.objects:
        p = rot(o.center_m)
        o.center_m = [p[0] + t[0], p[1] + t[1], p[2]]
        o.axes = [rot(a) for a in o.axes]
    if new.coverage is not None:
        cov = new.coverage
        b1, b2 = rot(cov.basis[0]), rot(cov.basis[1])
        cov.basis = [b1, b2]
        cov.origin_xy = [cov.origin_xy[0] + sum(x * y for x, y in zip(t, b1)), cov.origin_xy[1] + sum(x * y for x, y in zip(t, b2))]
    e1, e2 = (0.0, 1.0, 0.0), (-1.0, 0.0, 0.0)   # plan basis of +z (relations.plan_basis): plan coords rotate like the world
    for z in new.zones:
        z.polygon_xy_m = [[c * x - s * y + sum(a * b for a, b in zip(t, e1)), s * x + c * y + sum(a * b for a, b in zip(t, e2))] for x, y in z.polygon_xy_m]
    new.scene_id = f"{scene.scene_id}~rot{angle_deg:g}"
    return new


def _diffs(scene_id: str, relation: str, base: dict, got: dict) -> list[dict]:
    return [{"scene_id": scene_id, "relation": relation, "rule_id": k[0], "subjects": list(k[1]), "before": base.get(k), "after": got.get(k)}
            for k in sorted(set(base) | set(got)) if base.get(k) != got.get(k)]


def _find(facts: Facts, pred: str, args: list[str]):
    for f in facts.facts:
        if f.pred == pred and (f.args == args or (pred in SYMMETRIC and len(args) == 2 and f.args == args[::-1])):
            return f
    return None


def metamorphic(pack: RulePack, scenes, engine, facts_of, angles=(90, 180, 270), shift_xy=(3.1, -2.7), inflate: float = 10.0,
                steps=(-5, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 5)) -> dict[str, list[dict]]:
    v: dict[str, list[dict]] = {"rigid": [], "delete_unreferenced": [], "inflate_u": [], "monotone": []}
    selected = {c for r in pack.rules for cs in r.spec.get("selection", {}).values() for c in cs}
    for scene in scenes:
        facts = facts_of(scene)
        base = statuses(engine(facts, pack, scene))
        for ang in angles:
            moved = transform(scene, ang, shift_xy)
            v["rigid"] += _diffs(scene.scene_id, f"rot{ang:g}+shift", base, statuses(engine(facts_of(moved), pack, moved)))
        for o in scene.objects:
            if o.cls in selected:
                continue
            less = scene.model_copy(deep=True)
            less.objects = [x for x in less.objects if x.id != o.id]
            v["delete_unreferenced"] += _diffs(scene.scene_id, f"delete {o.id}", base, statuses(engine(facts_of(less), pack, less)))
        fat = facts.model_copy(deep=True)
        for f in fat.facts:
            if f.u is not None:
                f.u *= inflate
        got = statuses(engine(fat, pack, scene))
        for key, s in base.items():
            g = got.get(key)
            if (s in ("PASS", "FAIL") and g not in (s, "NEEDS_MEASUREMENT")) or (s not in ("PASS", "FAIL") and g != s):
                v["inflate_u"].append({"scene_id": scene.scene_id, "relation": f"u x{inflate:g}", "rule_id": key[0], "subjects": list(key[1]), "before": s, "after": g})
        for rule in pack.rules:
            req = rule.spec.get("requirement", {})
            op, T = req.get("operator"), req.get("threshold")
            if op not in (">=", "<=", ">", "<") or not isinstance(T, (int, float)):
                continue
            keys = sorted(k for k in base if k[0] == rule.rule_id)
            if not keys:
                continue
            key = keys[0]   # ponytail: one subject per rule; every subject goes through the same decision function
            b = dict(zip(rule.spec.get("selection", {}), key[1]))
            fact = _find(facts, req["predicate"], [b.get(a, a) for a in req.get("args", [])])
            if fact is None or fact.value is None:
                continue
            sigma = (fact.u or 2.0) / 2
            sign = 1 if op in (">=", ">") else -1
            seq = []
            for off in steps:
                fx = facts.model_copy(deep=True)
                _find(fx, fact.pred, fact.args).value = T + sign * off * sigma
                seq.append(statuses(engine(fx, pack, scene)).get(key))
            ranks = [ORDER.get(s_, -1) for s_ in seq]
            if min(ranks) < 0 or any(a > b_ for a, b_ in zip(ranks, ranks[1:])) or seq[0] != "FAIL" or seq[-1] != "PASS":
                v["monotone"].append({"scene_id": scene.scene_id, "relation": "cross threshold", "rule_id": rule.rule_id, "subjects": list(key[1]),
                                      "steps_sigma": list(steps), "sequence": seq})
    return v


def ok(report: dict[str, list]) -> bool:
    return not any(report.values())


# ---------------------------------------------------------------- (d) threshold grid
def threshold_grid(pack: RulePack, engine, cases=None) -> dict:
    if cases is None:
        from ehs_spatial.verdict.synth.grid import cases as synth_cases   # lab tooling, not a layer
        cases = synth_cases(pack)
    rows, mismatches = [], []
    for c in cases:
        if "skipped" in c:
            rows.append(dict(c))
            continue
        vs = engine(c["facts"], pack, c["scene"])
        classes = {o.id: o.cls for o in c["scene"].objects}
        got = sorted({v.status for v in vs.verdicts if v.rule_id == c["rule_id"] and len(v.subjects) > c["arg_index"]
                      and classes.get(v.subjects[c["arg_index"]]) == c["subject_cls"]})
        row = {"rule_id": c["rule_id"], "offset": c["offset"], "expected": c["expected"], "actual": got, "value_mm": c["value_mm"],
               "u_mm": c["u_mm"], "threshold": c["threshold"], "ok": got == [c["expected"]]}
        rows.append(row)
        if not row["ok"]:
            mismatches.append(row)
    return {"rows": rows, "mismatches": mismatches}


def grid_table(rows: list[dict]) -> str:
    """Markdown: one row per rule, one column per offset (status, '!= expected' on a mismatch)."""
    offsets = list(dict.fromkeys(r["offset"] for r in rows if "offset" in r))
    by = {}
    for r in rows:
        if "offset" in r:
            by.setdefault(r["rule_id"], {})[r["offset"]] = r
    lines = ["| rule | " + " | ".join(offsets) + " |", "|---|" + "---|" * len(offsets)]
    for rid, cols in by.items():
        cells = []
        for o in offsets:
            r = cols.get(o)
            cells.append("" if r is None else ("/".join(r["actual"]) or "-") + ("" if r["ok"] else f" != {r['expected']}"))
        lines.append(f"| {rid} | " + " | ".join(cells) + " |")
    for r in rows:
        if "skipped" in r:
            lines.append(f"| {r['rule_id']} | skipped: {r['skipped']} |")
    return "\n".join(lines)


# ---------------------------------------------------------------- facts parity (real L2 vs synthetic-facts@1)
def facts_parity(a: Facts, b: Facts, tol_mm: float = 2.0) -> dict[str, list]:
    def key(f):
        args = sorted(f.args) if f.pred in SYMMETRIC else list(f.args)
        return (f.pred, tuple(args))
    ia, ib = {key(f): f for f in a.facts}, {key(f): f for f in b.facts}
    out: dict[str, list] = {"value": [], "u": [], "only_a": sorted(k for k in ia if k not in ib), "only_b": sorted(k for k in ib if k not in ia)}
    for k in sorted(set(ia) & set(ib)):
        fa, fb = ia[k], ib[k]
        for field, lst in (("value", out["value"]), ("u", out["u"])):
            x, y = getattr(fa, field), getattr(fb, field)
            if (x is None) != (y is None) or (x is not None and abs(x - y) > tol_mm):
                lst.append({"pred": k[0], "args": list(k[1]), "a": x, "b": y})
    return out
