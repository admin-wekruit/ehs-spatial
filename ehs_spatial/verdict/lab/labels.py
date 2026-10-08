"""Human labels -> gold. Merges `verdict-labels/1` files (exported from the L7 html@1 report, one per reviewer and run) into a benchmark
gold.json, and extracts the declared inputs per scene for L1 scene-json's `declared:` cfg. `panoptes verdict labels merge | declared`
(also `python -m ehs_spatial.verdict.lab.labels`). Benchmark folders are frozen: the output goes where `--out` says, never over an
existing gold.json of a benchmark version; the command prints where the new gold belongs."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

SCHEMA = "verdict-labels/1"


def load(paths) -> list[dict]:
    files = [json.loads(Path(p).read_text()) for p in paths]
    bad = [str(p) for p, f in zip(paths, files) if f.get("schema") != SCHEMA]
    if bad:
        raise ValueError(f"not {SCHEMA}: {', '.join(bad)}")
    return files


def merge_gold(label_files, gold_in: dict | None = None, version: str | None = None) -> dict:
    """gold.json: one row per (scene_id, rule_id, subjects). Reviewers who agree share the row; a key labelled with different statuses
    goes to `conflicts` (every label of it) and stays out of `verdicts`. Rows of `gold_in` not labelled again are carried over, and the
    result stays provisional while any carried row is. NOT_APPLICABLE is recorded as a status like the others."""
    by_key: dict[tuple, list[dict]] = {}
    for f in load(label_files):
        for l in f["labels"]:
            row = {"scene_id": f["scene_id"], "rule_id": l["rule_id"], "subjects": list(l["subjects"]), "labels": list(l.get("labels", [])),
                   "status": l["status"], "reason": l.get("reason", ""), "confidence": l.get("confidence", "sure"), "reviewer": f.get("reviewer", "")}
            by_key.setdefault((f["scene_id"], l["rule_id"], tuple(l["subjects"])), []).append(row)
    old = gold_in or {}
    kept = [g for g in old.get("verdicts", []) if (g["scene_id"], g["rule_id"], tuple(g["subjects"])) not in by_key]
    verdicts, conflicts = [], []
    for k in sorted(by_key):
        rows = by_key[k]
        if len({r["status"] for r in rows}) == 1:
            verdicts.append({**rows[0], "reviewer": ", ".join(sorted({r["reviewer"] for r in rows}))})
        else:
            conflicts += rows
    provisional = bool(kept) and bool(old.get("provisional"))
    return {"version": version or re.sub(r"\d+$", lambda m: str(int(m.group()) + 1), old.get("version", "v0")), "provisional": provisional,
            "source": f"human labels; {len(kept)} rows still from: {old.get('source', '')}" if provisional else "human labels",
            "verdicts": sorted(kept + verdicts, key=lambda g: (g["scene_id"], g["rule_id"], g["subjects"])), "conflicts": conflicts}


def declared_inputs(label_files) -> dict[str, dict]:
    """{scene_id: declared inputs} over the label files; a later file wins per key."""
    out: dict[str, dict] = {}
    for f in load(label_files):
        out.setdefault(f["scene_id"], {}).update(f.get("declared_inputs", {}))
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="panoptes verdict labels", description=__doc__)
    sub = p.add_subparsers(dest="labels_command", required=True)
    m = sub.add_parser("merge", help="label files -> gold.json (agreements in, conflicts listed)")
    m.add_argument("--gold-in", help="previous gold.json: its rows survive unless labelled again")
    m.add_argument("--version", help="version of the new gold (default: the previous one + 1, or v1)")
    d = sub.add_parser("declared", help="label files -> declared.json {scene_id: declared inputs} for L1 {plugin: scene-json, declared: <path>}")
    for q in (m, d):
        q.add_argument("--labels", nargs="+", required=True, help="verdict-labels/1 files exported from verdicts.html")
        q.add_argument("--out", required=True)
    a = p.parse_args(argv)
    out = Path(a.out)
    if out.exists() and (out.parent / "items.yaml").exists():
        raise SystemExit(f"{out} belongs to a frozen benchmark version; write the new gold to a new version folder")
    if a.labels_command == "merge":
        gold = merge_gold(a.labels, json.loads(Path(a.gold_in).read_text()) if a.gold_in else None, a.version)
        out.write_text(json.dumps(gold, indent=1, ensure_ascii=False) + "\n")
        print(f"{out}: {len(gold['verdicts'])} verdicts, {len(gold['conflicts'])} conflicting labels left out. Benchmark folders are frozen: copy it to"
              f" ehs_spatial/verdict/benchmark/{gold['version']}/gold.json in a new folder next to copies of items.yaml and scenes/, never into v0.")
    else:
        dec = declared_inputs(a.labels)
        out.write_text(json.dumps(dec, indent=1, sort_keys=True, ensure_ascii=False) + "\n")
        print(f"{out}: declared inputs for {', '.join(sorted(dec)) or 'no scene'}; use it as L1: {{plugin: scene-json, declared: {out}}}")


if __name__ == "__main__":
    main()
