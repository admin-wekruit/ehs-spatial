"""L6 python@2: evaluates each rule's engine-neutral `spec` directly on Facts. Parity engine for clingo@2 (the trial's rules.lp).

Per rule: bind the selection variables to objects / zones of the listed classes (classes from obj(id, cls) facts, args = [id, cls], and
from Scene.objects when a scene is given); keep the bindings whose applicability atoms hold as facts; drop the ones an exception attribute
switches off; then for each binding look up the requirement fact keyed (predicate, args) -- symmetric pair predicates (min_distance_3d,
horizontal_gap, z_overlap, line_of_sight) are also tried in reverse order -- and decide:
  untrusted subject (untrusted(X) fact, Facts.quality['untrusted'], or a fact flagged 'untrusted')  -> CANNOT_DETERMINE
  a declared input of the rule missing from Scene.declared_inputs, a formula symbol without a value,
  or a table with no declared value under its id                                                      -> NEEDS_INPUT (unknown_inputs lists them)
  no fact                                                                                            -> CANNOT_DETERMINE
  otherwise the decision rule (cfg):
    decision: guard_band, k (default 2; Facts carry U at k=2, so U is scaled by k/2): PASS / FAIL only when the whole interval
              V -+ U lies on one side, boundary inclusive on the passing side (V - U >= T passes, V + U = T is NEEDS_MEASUREMENT), as rules.lp
    decision: simple        compares the point value
    decision: conservative  FAIL unless the whole interval passes
  margin = V - T for >= / >, T - V for <= / <, None for == / !=.
Topology: enclosed(Z) with no enclosed fact and Z = 'hazard_zone' (the implicit zone of the grid's hazard cells) is computed on Facts.grid:
BFS from the outside cells through unblocked cells; a hazard cell reached through observed cells -> 0 (FAIL); reached only through
unobserved cells -> CANNOT_DETERMINE; not reached -> 1 (PASS) when the grid carries coverage; Grid.observed None (no Coverage in the
scene) -> CANNOT_DETERMINE whatever the BFS says (@2: the contract reads Coverage = None as unknown everywhere; @1 said PASS when the
reconstructed footprints closed the ring); no hazard cells -> CANNOT_DETERMINE.
Output is sorted by (rule_id, subjects); same inputs + cfg -> byte-identical VerdictSet. No geometry is computed here.
"""
from __future__ import annotations

import ast
import operator
import re
from collections import deque
from itertools import product
from pathlib import Path
from typing import Any

from ehs_spatial.verdict.contracts import Facts, Grid, Provenance, Rule, Scene, Verdict, VerdictSet
from ehs_spatial.verdict.plugins import register, tag

SYMMETRIC = {"min_distance_3d", "horizontal_gap", "z_overlap", "line_of_sight"}
IMPLICIT_ZONE = "hazard_zone"
_ATOM = re.compile(r"^\s*(\w+)\s*\(([^)]*)\)\s*$")
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg}


def atom(text: str) -> tuple[str, list[str]]:
    m = _ATOM.match(text)
    if not m:
        raise ValueError(f"applicability atom {text!r} is not name(args)")
    return m.group(1), [a.strip() for a in m.group(2).split(",") if a.strip()]


def decide(v: float, u: float | None, op: str, T: float, decision: str = "guard_band", k: float = 2.0) -> str:
    u = (u or 0.0) * (k / 2.0 if decision == "guard_band" else 1.0)
    lo, hi = (v, v) if decision == "simple" else (v - u, v + u)
    if op == ">=":
        ok, bad = lo >= T, hi < T
    elif op == ">":
        ok, bad = lo > T, hi <= T
    elif op == "<=":
        ok, bad = hi <= T, lo > T
    elif op == "<":
        ok, bad = hi < T, lo >= T
    elif op == "==":
        ok, bad = lo == hi == T, (hi < T or lo > T)
    elif op == "!=":
        ok, bad = (hi < T or lo > T), lo == hi == T
    else:
        raise ValueError(f"unknown operator {op!r}")
    return "PASS" if ok else "FAIL" if (bad or decision == "conservative") else "NEEDS_MEASUREMENT"


def formula(expr: str, env: dict) -> float:
    """Arithmetic only: + - * / ** unary -, min, max, numbers, names from env. An unknown name raises KeyError(name)."""
    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return float(n.value)
        if isinstance(n, ast.Name):
            return float(env[n.id])
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.operand))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in ("min", "max"):
            return {"min": min, "max": max}[n.func.id](*map(ev, n.args))
        raise ValueError(f"unsupported formula node {type(n).__name__} in {expr!r}")
    return ev(ast.parse(expr, mode="eval"))


def enclosure(grid: Grid) -> tuple[int | None, dict]:
    nx, ny = grid.shape
    blocked = {tuple(c) for c in grid.blocked}
    hazard = {tuple(c) for c in grid.hazard} - blocked
    outside = {tuple(c) for c in grid.outside} - blocked
    observed = None if grid.observed is None else {tuple(c) for c in grid.observed}
    info: dict[str, Any] = {"shape": list(grid.shape), "cell_m": grid.cell_m, "hazard_cells": len(hazard), "openings": 0, "via_unobserved": False,
                            "no_coverage": observed is None}
    if not hazard:
        return None, info

    def bfs(passable):
        seen = {c for c in outside if passable(c)}
        q = deque(seen)
        while q:
            x, y = q.popleft()
            for n in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if 0 <= n[0] < nx and 0 <= n[1] < ny and n not in seen and passable(n):
                    seen.add(n)
                    q.append(n)
        return seen

    reach = bfs(lambda c: c not in blocked)
    info["openings"] = sum(1 for (x, y) in reach - hazard if any(n in hazard for n in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))))
    if observed is None:
        info["via_unobserved"] = bool(reach & hazard)
        return None, info
    if not reach & hazard:
        return 1, info
    if bfs(lambda c: c not in blocked and c in observed) & hazard:
        return 0, info
    info["via_unobserved"] = True
    return None, info


class _Eval:
    def __init__(self, facts: Facts, scene: Scene | None, decision: str, k: float):
        self.decision, self.k, self.grid = decision, k, facts.grid
        self.index: dict[tuple[str, tuple[str, ...]], Any] = {}
        self.by_pred: dict[str, list] = {}
        for f in facts.facts:
            self.index.setdefault((f.pred, tuple(f.args)), f)
            self.by_pred.setdefault(f.pred, []).append(f)
        self.classes = {f.args[0]: f.args[1] for f in facts.facts if f.pred == "obj" and len(f.args) > 1}
        self.labels: dict[str, str] = {}
        self.zones: dict[str, str] = {}
        self.objects: dict[str, Any] = {}
        self.declared: dict[str, Any] = {}
        if scene is not None:
            for o in scene.objects:
                self.classes.setdefault(o.id, o.cls)
                self.labels[o.id], self.objects[o.id] = o.label, o
            self.zones = {z.id: z.kind for z in scene.zones}
            self.declared = dict(scene.declared_inputs)
        if self.grid is not None and self.grid.hazard and IMPLICIT_ZONE not in self.zones.values():
            self.zones[IMPLICIT_ZONE] = IMPLICIT_ZONE
        self.untrusted = {f.args[0] for f in self.by_pred.get("untrusted", []) if f.args and (f.value is None or f.value)}
        self.untrusted |= {str(x) for x in facts.quality.get("untrusted", []) or []}

    def candidates(self, classes: list[str]) -> list[str]:
        return sorted([i for i, c in self.classes.items() if c in classes] + [z for z, kind in self.zones.items() if kind in classes])

    def lookup(self, pred: str, args: list[str]):
        f = self.index.get((pred, tuple(args)))
        if f is None and pred in SYMMETRIC and len(args) == 2:
            f = self.index.get((pred, (args[1], args[0])))
        return f

    def holds(self, pred: str, args: list[str]) -> bool:
        f = self.lookup(pred, args)
        return f is not None and (f.value is None or f.value != 0)

    def bindings(self, rule: Rule, cov: dict) -> list[dict[str, str]]:
        sel = rule.spec.get("selection", {})
        out = [dict(zip(sel, combo)) for combo in product(*(self.candidates(c) for c in sel.values())) if len(set(combo)) == len(combo)]
        cov["candidates"] = len(out)
        for text in rule.spec.get("applicability", []):
            name, args = atom(text)
            nxt, seen = [], set()
            for b in out:
                for f in self.by_pred.get(name, []):
                    if len(f.args) != len(args) or not (f.value is None or f.value != 0):
                        continue
                    e = dict(b)
                    for a, fa in zip(args, f.args):
                        if a in e:
                            if e[a] != fa:
                                break
                        elif a[:1].isupper():
                            e[a] = fa
                        elif a != fa:
                            break
                    else:
                        key = tuple(sorted(e.items()))
                        if key not in seen:
                            seen.add(key)
                            nxt.append(e)
            cov["not_applicable"] += max(0, len(out) - len(nxt))
            out = nxt
        for attr in rule.spec.get("exceptions", []):
            keep = [b for b in out if not any(self.holds(attr, [x]) or getattr(self.objects.get(x), attr, None) for x in b.values())]
            cov["exception"] += len(out) - len(keep)
            out = keep
        return out

    def threshold(self, rule: Rule, req: dict, notes: list[str]) -> tuple[float | None, list[str]]:
        needed = list(dict.fromkeys(list(req.get("inputs", [])) + list(rule.inputs)))
        missing = [i for i in needed if i not in self.declared]
        T = req.get("threshold")
        if isinstance(T, str):
            T = rule.thresholds.get(T)
        if T is None and "formula" in req and not missing:   # evaluated only once every listed input is declared
            env = {**self.declared, **{sym: self.declared[name] for sym, name in req.get("bindings", {}).items() if name in self.declared}}
            try:
                T = formula(req["formula"], env)
            except KeyError as e:
                missing.append(str(e.args[0]))
        if T is None and "table" in req:
            tid = req["table"]
            if tid in self.declared:
                T = float(self.declared[tid])
            else:
                missing.append(tid)
                notes.append(f"table {tid}: python@2 has no verified lookup; declare its value under that id to evaluate")
        return (None if T is None else float(T)), list(dict.fromkeys(missing))

    def verdict(self, rule: Rule, b: dict[str, str], subjects: list[str], prov: Provenance) -> Verdict:
        req = rule.spec["requirement"]
        pred, args, op = req.get("predicate"), [b.get(a, a) for a in req.get("args", [])], req.get("operator", ">=")
        v = Verdict(rule_id=rule.rule_id, rule_version=rule.version, status="CANNOT_DETERMINE", subjects=subjects,
                    labels=[self.labels[s] for s in subjects if self.labels.get(s)], unit=req.get("unit"), provenance=prov)
        bad = sorted(s for s in subjects if s in self.untrusted)
        if bad:
            v.notes.append("untrusted subject: " + ", ".join(bad))
            v.evidence = {"untrusted": bad}
            return v
        T, missing = self.threshold(rule, req, v.notes)
        value = u = None
        f = self.lookup(pred, args)
        if f is None and pred == "enclosed" and self.grid is not None and args == [IMPLICIT_ZONE]:
            value, info = enclosure(self.grid)
            v.evidence = {"grid": info}
            if value is None:
                v.notes.append("no hazard cells in the grid" if not info["hazard_cells"] else
                               "no floor coverage in the scene: blocked cells are only what was reconstructed, an unobserved gap cannot be "
                               f"excluded -> enclosure not decided ({info['openings']} opening cells in the reconstructed footprints)" if info["no_coverage"] else
                               f"{info['openings']} opening cells reachable from outside only through unobserved cells -> cannot decide")
        elif f is None:
            v.notes.append(f"no fact {pred}({', '.join(args)})")
        elif "untrusted" in f.flags:
            v.notes.append(f"fact {pred}({', '.join(args)}) flagged untrusted")
            v.evidence = {"facts": [f.model_dump()]}
        else:
            value, u = f.value, f.u
            v.evidence = {"facts": [f.model_dump()], "views": list(f.views)}
            if value is not None and u is None and op in (">=", ">", "<=", "<") and self.decision != "simple":
                v.notes.append("fact carries no uncertainty; U taken as 0")
        v.measured, v.u, v.threshold = value, u, T
        if op == "table" and not missing:   # handwritten spells table rules with operator 'table': the comparison direction is the table's
            missing.append(f"{req.get('table')}#direction")
            v.notes.append("operator 'table': python@2 needs the comparison direction (>= / <=) besides the table value")
        if missing:
            v.status, v.unknown_inputs = "NEEDS_INPUT", missing
            return v
        if value is None:
            return v
        v.status = decide(value, u, op, T, self.decision, self.k)
        v.margin = value - T if op in (">=", ">") else T - value if op in ("<=", "<") else None
        return v

    def rule(self, rule: Rule, prov: Provenance) -> tuple[list[Verdict], dict]:
        cov = {"status": rule.status, "candidates": 0, "verdicts": 0, "not_applicable": 0, "exception": 0}
        if not rule.spec.get("requirement"):
            cov["skipped"] = "no requirement in spec"
            return [], cov
        sel = rule.spec.get("selection", {})
        verdicts = [self.verdict(rule, b, [b[v] for v in sel], prov) for b in self.bindings(rule, cov)]
        cov["verdicts"] = len(verdicts)
        return verdicts, cov


@register("L6", "python", "2")
class PythonEngine:
    """cfg: decision ('guard_band' | 'simple' | 'conservative'), k (guard band, default 2), run_id, benchmark, plugins {layer: tag}."""

    def run(self, inputs: dict[str, Any], cfg: dict[str, Any], workdir: Path) -> dict[str, Any]:
        cfg = cfg or {}
        facts, pack, scene = inputs["facts"], inputs["rule_pack"], inputs.get("scene")
        decision, k = str(cfg.get("decision", "guard_band")), float(cfg.get("k", 2))
        if decision not in ("guard_band", "simple", "conservative"):
            raise ValueError(f"unknown decision rule {decision!r}")
        plugins = {str(a): str(b) for a, b in dict(cfg.get("plugins", {})).items()}
        if facts.producer:
            plugins.setdefault("L2", facts.producer)
        if scene is not None and scene.producer:
            plugins.setdefault("L1", scene.producer)
        plugins["L6"] = tag(self)
        prov = Provenance(run_id=str(cfg.get("run_id", "")), benchmark=str(cfg.get("benchmark", "")), plugins=plugins)
        ev = _Eval(facts, scene, decision, k)
        verdicts, coverage = [], {}
        gaps: dict[str, set] = {"unknown_inputs": set(), "untrusted": set(), "missing_facts": set(), "rules_skipped": set()}
        for rule in pack.rules:
            if rule.status in ("vocabulary_gap", "refused"):
                coverage[rule.rule_id] = {"status": rule.status, "skipped": rule.unsupported_reason or rule.status}
                gaps["rules_skipped"].add(rule.rule_id)
                continue
            vs, cov = ev.rule(rule, prov)
            verdicts += vs
            coverage[rule.rule_id] = cov
            for v in vs:
                gaps["unknown_inputs"] |= set(v.unknown_inputs)
                gaps["untrusted"] |= set(v.evidence.get("untrusted", []))
                gaps["missing_facts"] |= {n[len("no fact "):] for n in v.notes if n.startswith("no fact ")}
        verdicts.sort(key=lambda v: (v.rule_id, v.subjects))
        name = f"guard_band_k{k:g}" if decision == "guard_band" else decision
        return {"verdicts": VerdictSet(scene_id=facts.scene_id, rule_pack=f"{pack.pack_id}@{pack.version}", decision_rule=name, verdicts=verdicts,
                                       coverage=coverage, gaps={a: sorted(b) for a, b in gaps.items()}, provenance=prov)}
