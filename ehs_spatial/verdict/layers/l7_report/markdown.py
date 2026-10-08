"""L7 markdown@1: VerdictSet -> verdicts.md with a provenance header (run id, benchmark, plugin tags, rule pack, decision rule)
and one line per verdict with its evidence facts."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from ehs_spatial.verdict.contracts import Verdict, VerdictSet
from ehs_spatial.verdict.plugins import register

ORDER = ("PASS", "FAIL", "NEEDS_MEASUREMENT", "NEEDS_INPUT", "CANNOT_DETERMINE")


def fmt(x) -> str:
    return "—" if x is None else f"{x:g}"


def line(v: Verdict) -> str:
    who = " ↔ ".join(v.labels) if v.labels else " ".join(v.subjects)
    meas = f"{fmt(v.measured)} ± {fmt(v.u)}" if v.measured is not None else "—"
    ev = "; ".join(f"{f['pred']}({', '.join(f['args'])}) = {fmt(f.get('value'))} ± {fmt(f.get('u'))}" for f in v.evidence.get("facts", []))
    if "openings" in v.evidence:
        ev = f"openings {v.evidence['openings']} {v.evidence.get('open_directions', [])}"
    if v.unknown_inputs:
        ev += "; needs " + ", ".join(v.unknown_inputs)
    return f"| {v.rule_id}@{v.rule_version} | {who} | **{v.status}** | {meas} | {fmt(v.threshold)} | {fmt(v.margin)} | {ev} | {'; '.join(v.notes)} |"


def report(vs: VerdictSet) -> str:
    p, counts = vs.provenance, Counter(v.status for v in vs.verdicts)
    head = [f"# {vs.scene_id}: {len(vs.verdicts)} verdicts", "",
            f"run `{p.run_id}` · benchmark `{p.benchmark}` · rule pack `{vs.rule_pack}` · decision `{vs.decision_rule}`  ",
            "plugins: " + " · ".join(f"{layer} {tag}" for layer, tag in sorted(p.plugins.items())) + "  ",
            "counts: " + " · ".join(f"{s} {counts[s]}" for s in ORDER), "",
            "| rule | subjects | status | measured ± u (mm) | threshold | margin | evidence | notes |", "|---|---|---|---|---|---|---|---|"]
    return "\n".join(head + [line(v) for v in vs.verdicts]) + "\n"


@register("L7", "markdown", "1")
class Markdown:
    def run(self, inputs: dict[str, Any], cfg: dict[str, Any], workdir: Path) -> dict[str, Any]:
        out = Path(inputs["workdir"]) / "verdicts.md"
        out.write_text(report(inputs["verdicts"]))
        return {"report": out}
