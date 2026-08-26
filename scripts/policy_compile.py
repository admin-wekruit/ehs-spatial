"""Compile written EHS policy into reviewable predicates, then evaluate.

Stage 1 (model, cached): prose -> PolicySpec JSON, validated against the
closed predicate vocabulary in ehs_spatial.policy. A model that cannot
express a rule must set unsupported_reason instead of approximating it.
Stage 2 (no model, ever): evaluate the reviewed specs against a run's
SceneMap with the deterministic engine.

The compiled specs are the artefact a safety engineer signs off on. They
are plain JSON, diffable, and re-evaluated without another model call.

Usage:
  uv run --env-file .env python scripts/policy_compile.py --policies docs/policies/starter.md --live
  uv run python scripts/policy_compile.py --policies docs/policies/starter.md --run demo-real-factory
"""

import argparse
import json
import re
import sys
from pathlib import Path

from ehs_spatial.contracts import SceneMap
from ehs_spatial.policy import PolicySpec, evaluate_policies

CACHE_DIR = Path("outputs/policies")
COMPILER_MODEL = "gemini-3.5-flash"


def _read_policies(path: Path) -> list[str]:
    """One rule per non-empty, non-heading line."""
    lines = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        lines.append(re.sub(r"^[-*]\s*", "", line))
    return lines


def _prompt(text: str, index: int, vocabulary: list[str]) -> str:
    from ehs_spatial.policy import Predicate

    return (
        "You translate written workplace-safety rules into a strict JSON "
        "structure. You never decide compliance; a separate deterministic "
        "engine measures the scene.\n\n"
        f"Predicates you may use (exact strings): "
        f"{[p.value for p in Predicate]}\n"
        "  min_separation: subject must be at least `threshold` metres from object\n"
        "  max_separation: subject must be within `threshold` metres of object\n"
        "  keep_clear: nothing of the subject labels may come within "
        "`threshold` metres of object\n"
        "  not_inside: subject footprint must not overlap object footprint "
        "(threshold is a small positive tolerance)\n"
        "  max_height: subject height must not exceed `threshold` metres\n"
        "  min_height: subject height must be at least `threshold` metres\n"
        "  max_tilt: subject tilt must not exceed `threshold` degrees\n\n"
        f"Object labels the perception layer can currently produce: "
        f"{vocabulary}. Prefer these; a label outside the list is allowed but "
        "means the rule cannot be checked until that class is added.\n\n"
        "Respond with ONLY one JSON object with keys: policy_id (slug), "
        "predicate, subject_labels (array), object_labels (array, empty for "
        "max_height/min_height/max_tilt), threshold (number > 0), unit ('m' or 'deg'), "
        "severity (critical|major|minor|advisory), rationale (one sentence), "
        "unsupported_reason (string or null).\n"
        "Set unsupported_reason and keep the other fields plausible when the "
        "rule needs something this vocabulary cannot express — signage, "
        "training records, time, colour, electrical state, or a measurement "
        "no predicate above covers. Do not approximate such a rule.\n\n"
        f"policy_id must start with 'p{index:02d}-'.\n\nRule: {text}"
    )


def _compile_one(text: str, index: int, vocabulary: list[str], *, live: bool):
    cache = CACHE_DIR / "compiled" / f"p{index:02d}.json"
    if cache.exists():
        return PolicySpec.model_validate_json(cache.read_text())
    if not live:
        return None
    from google import genai

    client = genai.Client()
    response = client.models.generate_content(
        model=COMPILER_MODEL, contents=_prompt(text, index, vocabulary)
    )
    match = re.search(r"\{.*\}", response.text or "", re.S)
    if not match:
        raise ValueError(f"compiler returned no JSON for rule {index}")
    payload = json.loads(match.group(0))
    payload["source_text"] = text
    # Validation is the guardrail: an invented predicate or a missing
    # threshold raises here, before anything can be evaluated.
    spec = PolicySpec.model_validate(payload)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(spec.model_dump_json(indent=2) + "\n")
    return spec


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policies", required=True)
    parser.add_argument("--run", help="run id under runs/ to evaluate against")
    parser.add_argument(
        "--scene",
        help="explicit SceneMap json (defaults to runs/<run>/scene.json); "
        "point it at inventory/scene.json to use the full enumerated vocabulary",
    )
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args(argv)

    from ehs_spatial.providers.sam3 import PROMPT_VOCABULARY

    texts = _read_policies(Path(args.policies))
    scene_path = (
        Path(args.scene)
        if args.scene
        else (Path("runs") / args.run / "scene.json" if args.run else None)
    )
    # Compile against the labels this scene can actually produce, so the
    # compiler picks names the evaluator will find.
    vocabulary = sorted(PROMPT_VOCABULARY)
    if scene_path and scene_path.exists():
        scene_labels = {
            entity["label"]
            for entity in json.loads(scene_path.read_text()).get("entities", [])
        }
        vocabulary = sorted(set(vocabulary) | scene_labels)
    pending = sum(
        1
        for index in range(1, len(texts) + 1)
        if not (CACHE_DIR / "compiled" / f"p{index:02d}.json").exists()
    )
    print(f"{len(texts)} rule(s) | {pending} uncompiled")
    if pending and not args.live:
        print("pass --live to compile them (re-evaluation is free)", file=sys.stderr)
        return 2

    specs = []
    for index, text in enumerate(texts, start=1):
        spec = _compile_one(text, index, vocabulary, live=args.live)
        if spec is not None:
            specs.append(spec)

    print("\ncompiled specs:")
    for spec in specs:
        flag = " [UNSUPPORTED]" if spec.unsupported_reason else ""
        target = " / ".join(spec.object_labels) or "-"
        print(
            f"  {spec.policy_id:<26} {spec.predicate.value:<15} "
            f"{' | '.join(spec.subject_labels):<28} vs {target:<20} "
            f"{spec.threshold}{spec.unit}{flag}"
        )
        if spec.unsupported_reason:
            print(f"      reason: {spec.unsupported_reason}")

    missing = sorted(
        label
        for spec in specs
        for label in spec.requires_labels()
        if label not in set(vocabulary)
    )
    if missing:
        print(f"\nlabels not yet in the perception vocabulary: {sorted(set(missing))}")

    if scene_path is None or not scene_path.exists():
        return 0

    scene = SceneMap.model_validate_json(scene_path.read_text())
    frame_count = len({f for e in scene.entities for f in e.evidence_frame_ids}) or 1
    results = evaluate_policies(specs, scene, capture_frame_count=frame_count)
    # The same {"specs", "results"} envelope pipeline.py writes, at the
    # location every reader consumes (app verdict/history cards, report.py):
    # runs/<id>/policies.json. Evaluating a bare --scene has no run dir, so
    # those verdicts land in the cache dir in the same envelope.
    report = {
        "specs": [spec.model_dump(mode="json") for spec in specs],
        "results": [r.model_dump(mode="json") for r in results],
    }
    out = (
        Path("runs") / args.run / "policies.json"
        if args.run
        else CACHE_DIR / f"report_{scene_path.stem}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")

    print(f"\nevaluated against {scene_path} ({frame_count} frame(s)):")
    for spec, result in zip(specs, results):
        head = f"  {result.status.value:<22} {spec.policy_id}"
        print(head)
        for violation in result.violations[:3]:
            print(
                f"      {violation.subject_id} vs "
                f"{violation.object_id or '-'}: measured "
                f"{violation.measured}{violation.unit} "
                f"(limit {violation.threshold}{violation.unit})"
            )
        for warning in result.warnings[:1]:
            print(f"      {warning}")
    print("\nwrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
