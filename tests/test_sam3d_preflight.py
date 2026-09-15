"""A plausible-looking manifest must not invent model release readiness."""
from scripts.preflight_sam3d import preflight


def test_preflight_reports_missing_gates_and_never_calls_models():
    pins = {"model": "facebook/sam-3d-objects", "codeRevision": "a" * 40, "modelRevision": "b" * 40}
    runtime = {"generation": {"pins": pins, "runtimeImage": None, "distribution": "sam3d_objects", "checkpointConfig": "checkpoints/pipeline.yaml"}}
    provider = {"generation": {"provider": "modal", "pins": pins, "paid": True}}
    for budget in (None, "NaN", "Infinity", "-1", "0"):
        result = preflight(runtime, provider, budget)
        failed = {c["gate"] for c in result["checks"] if c["status"] != "configured"}
        assert result["status"] == "blocked" and result["newModelCalls"] == 0
        assert {"explicit_paid_budget", "runtime_image_digest", "releaseEvidence.license", "releaseEvidence.runtime", "releaseEvidence.quality", "nativePoseEvidence.officialPoseFixture", "native_pose_and_basis_contract"} <= failed
    runtime["generation"]["checkpointConfig"] = "../untrusted.yaml"
    assert next(c for c in preflight(runtime, provider, "1")["checks"] if c["gate"] == "checkpoint_config")["status"] == "missing_or_invalid"
