"""panoptes verdict extract (also `python -m ehs_spatial.verdict.layers.l4_spec.extract_cli`): L4 llm-extract@0 standalone on one
safety-concept text. Writes the clause graph to --out and the extraction report + diff next to it (<stem>.report.json,
<stem>.diff.json); prints the summary, the flagged clauses, the diff as a markdown table and llm_calls. Only argparse at import."""
from __future__ import annotations

import argparse
from pathlib import Path


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--spec-text", required=True, help="markdown safety-concept text: numbered requirements with clause citations")
    p.add_argument("--out", required=True, help="clause graph JSON to write (the report and the diff land next to it)")
    p.add_argument("--reference", help="hand-extracted clauses JSON to diff against (default: spec/clauses-ts0011963-v0.json)")
    p.add_argument("--model", help="Claude model (default: PANOPTES_VERDICT_MODEL, else ehs_spatial.verdict.llm.MODEL)")
    p.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"])
    p.add_argument("--cache-dir", help="response cache (default: PANOPTES_LLM_CACHE, else runs/llm-cache)")
    p.add_argument("--scene", help="Scene JSON for retrieval (default: every clause is retrieved)")
    p.add_argument("--signature", help="Signature JSON (default: the package's signature-v1.json)")


def _req(k: dict) -> str:
    value = next((v for v in (k["threshold"], k["table"], k["formula"]) if v is not None), "")
    return f"{k['predicate']} {k['operator']} {value}".strip()


def run(args: argparse.Namespace) -> dict:
    import json
    import os

    from ehs_spatial.verdict.contracts import ClauseGraph, Scene, Signature
    from ehs_spatial.verdict.layers.l4_spec.llm_extract import REFERENCE, SIGNATURE_PATH, SPEC_DIR, LLMExtract, requirement_key

    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    cfg = {"spec_text": str(Path(args.spec_text).resolve()), "reference": str(Path(args.reference).resolve()) if args.reference else REFERENCE,
           "model": args.model, "effort": args.effort, "cache_dir": args.cache_dir or os.environ.get("PANOPTES_LLM_CACHE") or "runs/llm-cache"}
    inputs = {"spec_dir": SPEC_DIR, "signature": Signature.load(args.signature or SIGNATURE_PATH),
              "scene": Scene.load(args.scene) if args.scene else None}
    result = LLMExtract().run(inputs, cfg, out.parent)
    result["clauses"].dump(out)
    report = result["extraction_report"]
    out.with_name(f"{out.stem}.report.json").write_text(json.dumps(report, indent=1, sort_keys=True, ensure_ascii=False))
    print(report["summary"])
    for cid, info in report["clauses"].items():
        if info["flags"]:
            print(f"- {cid} ({info['unit']}): {' '.join(info['flags'])}")
    if "diff" in result:
        d = result["diff"]
        out.with_name(f"{out.stem}.diff.json").write_text(json.dumps(d, indent=1, sort_keys=True, ensure_ascii=False))
        llm_req = {c.id: _req(requirement_key(c.requirement)) for c in result["clauses"].clauses}
        ref_req = {c.id: _req(requirement_key(c.requirement)) for c in ClauseGraph.load(SPEC_DIR / cfg["reference"]).clauses}
        rows = ([(i, llm_req[i], "=", "same") for i in d["same_id_same_requirement"]]
                + [(x["id"], _req(x["llm"]), _req(x["reference"]), "DIFFERENT") for x in d["same_id_different_requirement"]]
                + [(i, llm_req[i], "-", "only llm") for i in d["ids_only_llm"]] + [(i, "-", ref_req[i], "only reference") for i in d["ids_only_reference"]])
        print(f"\n{d['summary']}\n\n| clause | llm | reference | status |\n|---|---|---|---|")
        print("\n".join(f"| {a} | {b} | {c} | {s} |" for a, b, c, s in rows))
    print(f"\nllm_calls: {result['llm_calls']}")
    return result


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="panoptes verdict extract", description=__doc__)
    add_arguments(p)
    run(p.parse_args(argv))


if __name__ == "__main__":
    main()
