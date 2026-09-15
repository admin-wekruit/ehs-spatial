"""Check SAM3D deployment inputs without loading a model or reserving paid work.

This checks configuration, not the truth of release evidence. Passed evidence
must come from the recorded license audit and actual pinned runtime fixtures.
"""
import argparse
from decimal import Decimal, InvalidOperation
import json
import os
from pathlib import Path
import re

from ehs_spatial.platform.contracts import PlatformError
from ehs_spatial.platform.reconstruction import ProviderSpec, provider_snapshot_from_env
from ehs_spatial.platform.spatial import SAM3DMeshAdapter


def preflight(runtime, provider, budget):
    if not isinstance(runtime, dict) or not isinstance(provider, dict):
        raise ValueError("Manifests must be objects")
    runtime = runtime.get("generation") or {}
    provider = provider.get("generation") or {}
    if not isinstance(runtime, dict) or not isinstance(provider, dict):
        raise ValueError("Generation configuration must be an object")
    checks = []

    def check(name, passed):
        checks.append({"gate": name, "status": "configured" if passed else "missing_or_invalid"})

    pins = provider.get("pins") or {}
    if not isinstance(pins, dict):
        raise ValueError("Pins must be an object")
    check("generation_provider", bool(provider.get("provider")))
    check("sam3d_model", pins.get("model") == "facebook/sam-3d-objects")
    for key in ("codeRevision", "modelRevision"):
        check(key, bool(re.fullmatch(r"[0-9a-f]{40}", str(pins.get(key) or ""))))
    check("runtime_pins_match", bool(pins) and runtime.get("pins") == pins)
    check("runtime_image_digest", bool(re.fullmatch(r"[^\s]+@sha256:[0-9a-f]{64}", str(runtime.get("runtimeImage") or ""))))
    check("runtime_distribution", runtime.get("distribution") == "sam3d_objects")
    check("mesh_source_build_receipt", bool(re.fullmatch(r"[0-9a-f]{64}", str(runtime.get("meshSourceBuildSha256") or ""))))
    checkpoint = Path(str(runtime.get("checkpointConfig") or ""))
    check("checkpoint_config", bool(runtime.get("checkpointConfig")) and not checkpoint.is_absolute() and ".." not in checkpoint.parts)
    for key in ("modalApp", "modalClass", "modalMethod"):
        check(key, bool(provider.get(key)))
    cost = provider.get("estimatedCostUsd")
    check("per_call_cost_reservation", type(cost) in (int, float) and cost > 0 and cost < float("inf"))
    try:
        value = Decimal(budget) if budget is not None else None
        budget_ok = value is not None and value.is_finite() and value > 0
    except (InvalidOperation, ValueError, TypeError):
        budget_ok = False
    check("explicit_paid_budget", budget_ok)
    check("budget_covers_one_reservation", budget_ok and type(cost) in (int, float) and cost > 0 and cost < float("inf") and value >= Decimal(str(cost)))
    check("paid_reservation_required", provider.get("paid", True) is True)
    evidence = provider.get("releaseEvidence") or {}
    native = provider.get("nativePoseEvidence") or {}
    if not isinstance(evidence, dict) or not isinstance(native, dict):
        raise ValueError("Evidence must be an object")
    for section, record, names in (
        ("releaseEvidence", evidence, ("license", "runtime", "quality")),
        ("nativePoseEvidence", native, ("licenseAudit", "meshOnlyDependencyAudit", "officialPoseFixture", "externalPointmapNoDepth")),
    ):
        check(section + ".pins", bool(pins) and record.get("pins") == pins)
        for name in names:
            gate = record.get(name) or {}
            if not isinstance(gate, dict):
                raise ValueError("Evidence gate must be an object")
            check(section + "." + name, gate.get("status") == "passed" and bool(re.fullmatch(r"[0-9a-f]{64}", str(gate.get("artifactSha256") or ""))))
    # Reuse the production validators; the callable is never invoked here.
    try:
        ProviderSpec(provider.get("provider"), pins, None, cost, evidence).validate("generation")
        production = True
    except (PlatformError, ValueError, TypeError):
        production = False
    check("production_release_contract", production)
    try:
        SAM3DMeshAdapter(None, pins, native, provider.get("providerToOpenCV"))
        pose_contract = True
    except (PlatformError, ValueError, TypeError):
        pose_contract = False
    check("native_pose_and_basis_contract", pose_contract)
    return {"status": "configuration_ready" if all(c["status"] == "configured" for c in checks) else "blocked",
            "newModelCalls": 0, "checks": checks,
            "scope": "Local configuration only; does not verify deployment, gated weight access, evidence authenticity, or model quality."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-manifest", default=os.environ.get("PANOPTES_MODEL_RUNTIME_MANIFEST"))
    parser.add_argument("--provider-manifest", default=os.environ.get("PANOPTES_PROVIDER_MANIFEST"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        runtime = json.loads(Path(args.runtime_manifest).read_text()) if args.runtime_manifest else {}
        if args.provider_manifest:
            os.environ["PANOPTES_PROVIDER_MANIFEST"] = args.provider_manifest
        provider = provider_snapshot_from_env()
        result = preflight(runtime, provider, os.environ.get("PANOPTES_PAID_BUDGET_USD"))
    except (OSError, ValueError, TypeError, PlatformError) as error:
        result = {"status": "blocked", "newModelCalls": 0, "error": getattr(error, "code", type(error).__name__)}
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    return 0 if result["status"] == "configuration_ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
