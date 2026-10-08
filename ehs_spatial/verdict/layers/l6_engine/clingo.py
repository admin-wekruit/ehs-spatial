"""L6 clingo@1: Facts + RulePack -> VerdictSet through clingo (port of research/verdict-layer-trial-2026-10-07/engine.py).

Program = RulePack.common_asp + every rule's `asp` + the Facts rendered as
  obj(Id, Class).  bottom(Id, V, U).  top(Id, V, U).  dist(A, B, D, U) (both orders).  reach_over(H, S, A, B, C, U).
  cell(c(X,Y)).  adj(c(X,Y), c(X2,Y2)).  blocked(c(X,Y)).  hazard(...).  outside(...).  observed(...).  untrusted(Id).
Integer millimetres. Decision rule: guard band with coverage factor k (cfg `k`, default 2). Facts carry u expanded at k = 2, so the
rendered U is u * k / 2: k = 2 reproduces the trial, k = 1 halves the band, k = 0 is simple acceptance. Verdict.u reports the fact's u.
The solved status/3, margin/3 and opening/1 atoms become Verdicts; measured / threshold / evidence come from the rule's spec and the Facts.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import clingo

from ehs_spatial.verdict.contracts import Fact, Facts, Grid, Provenance, Rule, RulePack, Scene, Verdict, VerdictSet
from ehs_spatial.verdict.plugins import register

STATUS = {"pass": "PASS", "fail": "FAIL", "needs_meas": "NEEDS_MEASUREMENT", "needs_input": "NEEDS_INPUT",
          "cannot_determine": "CANNOT_DETERMINE", "open": "CANNOT_DETERMINE"}
FLAG_NOTES = {"default_sigma": "no sigma for this dimension: default 0.05 m used", "default_scale_unc": "no scale uncertainty: default 2 % used"}


def q(s: str) -> str:
    return '"' + s + '"'


def render(scene: Scene, facts: Facts, k: float) -> str:
    scale = lambda u: 0 if u is None else int(round(u * k / 2))
    tops = {f.args[0]: f for f in facts.facts if f.pred == "top_height"}
    lines = [f"obj({q(o.id)},{o.cls})." for o in scene.objects]
    for f in facts.facts:
        a, v = [q(x) for x in f.args], None if f.value is None else int(round(f.value))
        if f.pred == "bottom_height":
            lines.append(f"bottom({a[0]},{v},{scale(f.u)}).")
        elif f.pred == "top_height":
            lines.append(f"top({a[0]},{v},{scale(f.u)}).")
        elif f.pred == "min_distance_3d":
            lines += [f"dist({a[0]},{a[1]},{v},{scale(f.u)}).", f"dist({a[1]},{a[0]},{v},{scale(f.u)})."]
        elif f.pred == "reach_over":
            ab = dict(x.split("=", 1) for x in f.flags if "=" in x)
            u = max([scale(f.u)] + [scale(tops[i].u) for i in f.args if i in tops])
            lines.append(f"reach_over({a[0]},{a[1]},{ab['a_mm']},{ab['b_mm']},{v},{u}).")
        elif f.pred == "untrusted":
            lines.append(f"untrusted({a[0]}).")
    if facts.grid:
        g, c = facts.grid, (lambda x, y: f"c({x},{y})")
        nx, ny = g.shape
        lines += [f"cell({c(x, y)})." for x in range(nx) for y in range(ny)]
        lines += [f"adj({c(x, y)},{c(x + dx, y + dy)})." for x in range(nx) for y in range(ny)
                  for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)) if 0 <= x + dx < nx and 0 <= y + dy < ny]
        for name in ("blocked", "hazard", "outside", "observed"):
            lines += [f"{name}({c(x, y)})." for x, y in getattr(g, name) or []]
    return "\n".join(lines) + "\n"


def solve(program: str) -> list[clingo.Symbol]:
    ctl = clingo.Control(["--warn=none"])
    ctl.add("base", [], program)
    ctl.ground([("base", [])])
    atoms: list[clingo.Symbol] = []
    ctl.solve(on_model=lambda m: atoms.extend(m.symbols(shown=True)))
    return atoms


def ids_of(sym: clingo.Symbol) -> list[str]:
    parts = sym.arguments if sym.type == clingo.SymbolType.Function and sym.name == "" else [sym]
    return [p.string if p.type == clingo.SymbolType.String else str(p) for p in parts]


def fact_dict(f: Fact) -> dict:
    return f.model_dump(include={"pred", "args", "value", "unit", "u", "flags", "views"})


def topology_evidence(g: Grid | None, openings: list[tuple[int, int]]) -> dict:
    if g is None:
        return {"grid": None}
    ev: dict[str, Any] = {"grid": {"shape": g.shape, "cell_m": g.cell_m, "blocked": len(g.blocked), "hazard": len(g.hazard),
                                   "observed": None if g.observed is None else len(g.observed)}, "openings": len(openings)}
    if openings and g.hazard:
        lo, cm = g.origin_xy, g.cell_m
        centre = lambda x, y: (lo[0] + (x + .5) * cm, lo[1] + (y + .5) * cm)
        hz = [centre(x, y) for x, y in g.hazard]
        cx, cy = sum(p[0] for p in hz) / len(hz), sum(p[1] for p in hz) / len(hz)

        def direction(x, y):
            px, py = centre(x, y)
            if py - cy > abs(px - cx):
                return "+e2"
            if cy - py > abs(px - cx):
                return "-e2"
            return "+e1" if px > cx else "-e1"

        ev["open_directions"] = sorted({direction(x, y) for x, y in openings})
    return ev


def verdicts_of(atoms: list[clingo.Symbol], scene: Scene, facts: Facts, pack: RulePack, prov: Provenance) -> list[Verdict]:
    rules = {r.rule_id: r for r in pack.rules}
    labels = {o.id: o.label for o in scene.objects}
    by = {(f.pred, tuple(f.args)): f for f in facts.facts}
    margins = {(str(a.arguments[0]), str(a.arguments[1])): a.arguments[2].number for a in atoms if a.name == "margin"}
    openings = [(a.arguments[0].arguments[0].number, a.arguments[0].arguments[1].number) for a in atoms if a.name == "opening"]
    out = []
    for a in sorted((a for a in atoms if a.name == "status"), key=str):
        rule_id, subj, st = str(a.arguments[0]), a.arguments[1], str(a.arguments[2])
        ids = ids_of(subj)
        rule = rules.get(rule_id) or Rule(rule_id=rule_id, version="?", clause="", standard="", edition="", rule_class="geometry")
        req = rule.spec.get("requirement", {})
        v = Verdict(rule_id=rule_id, rule_version=rule.version, status=STATUS[st], subjects=ids, labels=[labels[i] for i in ids if i in labels],
                    unit=req.get("unit"), notes=list(rule.spec.get("notes", [])), provenance=prov)
        if rule.rule_class == "topology":
            v.evidence = topology_evidence(facts.grid, openings)
            if st == "open":
                seen = "unknown (no coverage)" if facts.grid is None or facts.grid.observed is None else "partial"
                v.notes.insert(0, f"the outside reaches the hazard cells through {len(openings)} opening cells, directions "
                                  f"{v.evidence.get('open_directions', [])}; floor coverage {seen}: unobserved floor and real openings "
                                  "are indistinguishable -> no conclusion")
        else:
            f = by.get((req.get("predicate"), tuple(ids))) or by.get((req.get("predicate"), tuple(reversed(ids))))
            if f is not None:
                tops = [fact_dict(by[("top_height", (i,))]) for i in ids if st == "needs_input" and ("top_height", (i,)) in by]
                v.evidence = {"facts": [fact_dict(f)] + tops, "views": f.views}
                v.notes += [FLAG_NOTES[x] for x in f.flags if x in FLAG_NOTES]
            if st == "needs_input":
                v.unknown_inputs = list(rule.inputs)
            elif f is not None:
                v.measured, v.u, v.threshold, v.margin = f.value, f.u, req.get("threshold"), margins.get((rule_id, str(subj)))
        out.append(v)
    return out


@register("L6", "clingo", "1")
class Clingo:
    """cfg: k (guard-band coverage factor, default 2)."""

    def run(self, inputs: dict[str, Any], cfg: dict[str, Any], workdir: Path) -> dict[str, Any]:
        facts, pack, scene = inputs["facts"], inputs["rule_pack"], inputs["scene"]
        k = float(cfg.get("k", 2))
        prov = inputs.get("provenance") or Provenance(run_id="")
        program = pack.common_asp + "\n" + "\n".join(r.asp for r in pack.rules if r.asp) + "\n" + render(scene, facts, k)
        Path(workdir, "program.lp").write_text(program)
        verdicts = verdicts_of(solve(program), scene, facts, pack, prov)
        g = facts.grid
        return {"verdicts": VerdictSet(
            scene_id=facts.scene_id, rule_pack=f"{pack.pack_id}@{pack.version}", decision_rule=f"guard_band_k{k:g}", verdicts=verdicts,
            coverage={"rules_evaluated": [r.rule_id for r in pack.rules if r.asp],
                      "rules_skipped": {r.rule_id: r.unsupported_reason or r.status for r in pack.rules if not r.asp}},
            gaps={"unknown_inputs": sorted({i for v in verdicts for i in v.unknown_inputs}),
                  "untrusted": sorted(f.args[0] for f in facts.facts if f.pred == "untrusted"),
                  "grid_observed": bool(g is not None and g.observed is not None)},
            provenance=prov)}
