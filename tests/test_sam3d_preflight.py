"""A plausible-looking manifest must not invent model release readiness."""
from scripts.preflight_sam3d import preflight
from ehs_spatial.platform.contracts import digest
import numpy as np


def test_preflight_reports_missing_gates_and_never_calls_models():
    pins = {"model": "facebook/sam-3d-objects", "codeRevision": "a" * 40, "modelRevision": "b" * 40}
    runtime = {"generation": {"pins": pins, "runtimeImage": None, "distribution": "sam3d_objects", "checkpointConfig": "checkpoints/pipeline.yaml"}}
    provider = {"generation": {"provider": "modal", "pins": pins, "paid": True}}
    for budget in (None, "NaN", "Infinity", "-1", "0"):
        result = preflight(runtime, provider, budget)
        failed = {c["gate"] for c in result["checks"] if c["status"] != "configured"}
        assert result["status"] == "blocked" and result["newModelCalls"] == 0
        assert {"explicit_paid_budget", "runtime_image_digest", "mesh_source_build_receipt", "releaseEvidence.license", "releaseEvidence.runtime", "releaseEvidence.quality", "nativePoseEvidence.officialPoseFixture", "native_pose_and_basis_contract"} <= failed
    runtime["generation"]["checkpointConfig"] = "../untrusted.yaml"
    assert next(c for c in preflight(runtime, provider, "1")["checks"] if c["gate"] == "checkpoint_config")["status"] == "missing_or_invalid"


def research_configuration():
    pins = {"model":"facebook/sam-3d-objects","codeRevision":"a"*40,"modelRevision":"b"*40}
    passed = {"status":"passed","artifactSha256":"c"*64}
    runtime = {"generation":{"pins":pins,"runtimeImage":"registry.example/sam3d@sha256:"+"d"*64,
        "distribution":"sam3d_objects","meshSourceBuildSha256":"e"*64,"checkpointConfig":"checkpoints/pipeline.yaml"}}
    provider = {"generation":{"provider":"modal","pins":pins,"paid":True,"estimatedCostUsd":.01,
        "modalApp":"test-only","modalClass":"SAM3DObjects","modalMethod":"run","providerToOpenCV":np.eye(4).tolist(),
        "releaseEvidence":{"pins":pins,"license":passed,"runtime":{"status":"unverified"},"quality":{"status":"unverified"}},
        "nativePoseEvidence":{"pins":pins,"licenseAudit":passed,"meshOnlyDependencyAudit":passed,
            "officialPoseFixture":{"status":"unverified"},"externalPointmapNoDepth":{"status":"unverified"}}}}
    protocol = {"id":"test-only","purpose":"runtime_validation","inputHashes":["f"*64],"baselineRevision":"baseline",
        "metricDefinitions":{"runtime":"mesh response and official pose contract"},"policyThresholds":{},"split":"runtime_fixture",
        "entityId":"test-entity","inputAssetHashes":[{"assetId":"test-image","sha256":"f"*64}],
        "payloadSha256":digest({}),"providerManifestSha256":digest(provider),"runtimeManifest":runtime,
        "callLimits":{"maxCalls":1,"maxCostPerCallUsd":.01,"maxTotalCostUsd":.01}}
    return runtime, provider, protocol


def test_runtime_preflight_does_not_require_the_release_evidence_it_will_measure():
    runtime, provider, protocol = research_configuration()
    result = preflight(runtime, provider, ".01", purpose="runtime_validation", protocol=protocol)
    assert result["status"] == "configuration_ready" and result["newModelCalls"] == 0
    assert all(c["status"] != "configured" and not c["required"] for c in result["checks"] if c["gate"] in (
        "releaseEvidence.runtime", "releaseEvidence.quality", "nativePoseEvidence.officialPoseFixture", "production_release_contract"))
    assert preflight(runtime, provider, ".01")["status"] == "blocked"
    protocol["purpose"] = "quality_validation"
    assert preflight(runtime, provider, ".01", purpose="quality_validation", protocol=protocol)["status"] == "blocked"
    protocol["purpose"] = "runtime_validation"
    for budget in (None, ".009"):
        assert preflight(runtime, provider, budget, purpose="runtime_validation", protocol=protocol)["status"] == "blocked"


def test_product_runtime_gate_cannot_be_lowered_by_research_fields():
    import pytest
    from ehs_spatial.platform.contracts import PlatformError
    from ehs_spatial.platform.reconstruction import ProviderSpec
    _, manifest, protocol = research_configuration()
    config = manifest["generation"]
    evidence = config["releaseEvidence"]
    evidence["quality"] = evidence["license"]
    evidence["purpose"] = "runtime_validation"
    spec = ProviderSpec("test-only",config["pins"],None,.01,evidence)
    with pytest.raises(PlatformError,match="provider_release_gate_unverified") as error:
        spec.validate("generation")
    assert error.value.params["gate"] == "runtime"
    spec.validate("generation",research_protocol=protocol)
