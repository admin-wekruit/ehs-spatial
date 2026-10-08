"""Scorecard: one row per run of the ledger: plugin tag per layer, rule-pack statuses (compiled / needs_input / vocabulary_gap / refused,
the compile rate of an L5 variant), LLM calls, verdict counts per status, agreement with the benchmark's
gold.json per status when it exists, and what changed versus the previous row (item; per layer the plugin tag or its params;
`reuse` is not a change). Writes scorecard.md and scorecard.json next to the ledger."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from ehs_spatial.verdict.contracts import RulePack, VerdictSet
from ehs_spatial.verdict.plugins import LAYERS

STATUSES = ("PASS", "FAIL", "NEEDS_MEASUREMENT", "NEEDS_INPUT", "CANNOT_DETERMINE")
RULE_STATUSES = ("compiled", "needs_input", "vocabulary_gap", "refused")
NOT_PARAMS = {"plugin", "version", "reuse"}


def diff(prev: dict | None, cur: dict) -> str:
    if prev is None:
        return "—"
    changes = [f"item {prev.get('item')}→{cur.get('item')}"] if prev.get("item") != cur.get("item") else []
    for layer in LAYERS:
        a, b = prev.get(layer, {}), cur.get(layer, {})
        if (a.get("plugin"), a.get("version")) != (b.get("plugin"), b.get("version")):
            changes.append(f"{layer} {a.get('plugin')}→{b.get('plugin')}")
        changes += [f"{layer} {key} {a.get(key)}→{b.get(key)}" for key in sorted((set(a) | set(b)) - NOT_PARAMS) if a.get(key) != b.get(key)]
    return "; ".join(changes) or "same"


def gold_agreement(gold: dict | None, vs: VerdictSet) -> dict[str, str]:
    """{gold status: 'agree/total'} over the gold verdicts of this scene, matched by (rule_id, subjects)."""
    if not gold:
        return {}
    mine = {(v.rule_id, tuple(v.subjects)): v.status for v in vs.verdicts}
    agree, total = Counter(), Counter()
    for g in gold["verdicts"]:
        if g["scene_id"] == vs.scene_id:
            total[g["status"]] += 1
            agree[g["status"]] += mine.get((g["rule_id"], tuple(g["subjects"]))) == g["status"]
    return {s: f"{agree[s]}/{total[s]}" for s in STATUSES if total[s]}


def rows(runs_dir: Path) -> list[dict]:
    out, prev = [], None
    for line in (runs_dir / "ledger.jsonl").read_text().splitlines():
        entry = json.loads(line)
        root = runs_dir / entry["run_id"]
        if not (root / "L6/verdicts.json").exists():
            continue
        vs = VerdictSet.load(root / "L6/verdicts.json")
        cfg = json.loads((root / "config.json").read_text())
        gold_path = Path(entry["benchmark"]) / "gold.json"
        gold = json.loads(gold_path.read_text()) if gold_path.exists() else None
        counts = Counter(v.status for v in vs.verdicts)
        pack = RulePack.load(root / "L5/rule_pack.json") if (root / "L5/rule_pack.json").exists() else None
        rules = Counter(r.status for r in pack.rules) if pack else Counter()
        out.append({"run_id": entry["run_id"], "item": entry.get("item"), "scene_id": vs.scene_id, "plugins": entry["plugins"],
                    "rules": {s: rules[s] for s in RULE_STATUSES}, "counts": {s: counts[s] for s in STATUSES}, "gold": gold_agreement(gold, vs),
                    "gold_provisional": bool(gold and gold.get("provisional")), "diff": diff(prev, cfg),
                    "seconds": entry["seconds"], "llm_calls": entry.get("llm_calls", 0)})
        prev = cfg
    return out


def markdown(rows_: list[dict]) -> str:
    head = ["run", "item", *LAYERS, "rules compiled/needs_input/gap/refused", "llm calls", *STATUSES, "gold agreement", "diff vs previous row"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows_:
        gold = (" · ".join(f"{s} {v}" for s, v in r["gold"].items()) + (" (provisional)" if r["gold_provisional"] else "")) if r["gold"] else "—"
        cells = [r["run_id"], r["item"], *(r["plugins"][layer] for layer in LAYERS), "/".join(str(r["rules"][s]) for s in RULE_STATUSES),
                 r["llm_calls"], *(r["counts"][s] for s in STATUSES), gold, r["diff"]]
        lines.append("| " + " | ".join(str(c) for c in cells) + " |")
    return "\n".join(lines) + "\n"


def write(runs_dir: str | Path) -> str:
    runs_dir = Path(runs_dir)
    rows_ = rows(runs_dir)
    md = markdown(rows_)
    (runs_dir / "scorecard.md").write_text(md)
    (runs_dir / "scorecard.json").write_text(json.dumps(rows_, indent=1, sort_keys=True, ensure_ascii=False))
    return md
