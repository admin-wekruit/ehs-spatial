import asyncio
import importlib.util
import json
from pathlib import Path
import re

import gradio as gr
from gradio.state_holder import SessionState
from PIL import Image
import pytest

from ehs_spatial.artifacts import ArtifactStore
from ehs_spatial.contracts import (
    Assessment,
    AssessmentStatus,
    ClimbReview,
    GroundedAnswer,
    PolicyResult,
    PolicySpec,
    ReviewDisposition,
    SceneMap,
    SpatialFact,
    Violation,
)
from ehs_spatial.providers.base import ProviderError


class FakePipeline:
    def __init__(
        self,
        root: Path,
        *,
        run_error: Exception | None = None,
        question_error: Exception | None = None,
    ) -> None:
        self.store = ArtifactStore(root)
        self.run_error = run_error
        self.question_error = question_error
        self.assessment_calls = []
        self.question_calls = []

    def run_assessment(self, capture):
        self.assessment_calls.append(capture)
        if self.run_error is not None:
            raise self.run_error
        paths = self.store.paths(capture.run_id)
        paths.geometry_dir.mkdir(parents=True, exist_ok=True)
        paths.point_cloud_glb.write_bytes(b"real-glb-artifact")
        Image.new("RGB", (32, 32), "white").save(paths.topdown_png)
        fact = SpatialFact(
            fact_id="fact-clearance",
            predicate="minimum_boundary_clearance",
            subject_id="pallet-1",
            object_id="fence-1",
            value=0.5,
            unit="m",
            evidence_frame_ids=["frame-1", "frame-2"],
        )
        scene = SceneMap(
            run_id=capture.run_id,
            floor_plane=[0, 0, 1, 0],
            scale_source="camera_height",
            scale_factor=1,
            fence_polygon=[[0, 0], [2, 0], [2, 2]],
            entities=[],
            facts=[fact],
            warnings=[],
        )
        self.store.save_json(paths.scene_json, scene)
        return Assessment(
            status="FAIL",
            fact_ids=[fact.fact_id],
            evidence_frame_ids=["frame-1", "frame-2"],
            approximate_distance_m=0.5,
            climb_review=ClimbReview(
                verdict="uncertain",
                rationale=(
                    "REVIEW only: semantic climb hint verdict=uncertain. "
                    "SceneMap facts: minimum_boundary_clearance=0.5 m."
                ),
                fact_ids=[fact.fact_id],
            ),
        )

    def answer_question(self, run_id, question):
        self.question_calls.append((run_id, question))
        if self.question_error is not None:
            raise self.question_error
        return GroundedAnswer(
            answer="SceneMap facts: minimum boundary clearance = 0.5 m.",
            fact_ids=["fact-clearance"],
            evidence_frame_ids=["frame-1", "frame-2"],
        )


def _images(tmp_path: Path) -> list[str]:
    paths = []
    for index in range(1, 5):
        path = tmp_path / f"view-{index}.png"
        Image.new("RGB", (8, 8), (index, index, index)).save(path)
        paths.append(str(path))
    return paths


def _load_root_app():
    spec = importlib.util.spec_from_file_location(
        "ehs_root_app", Path(__file__).parents[1] / "app.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _assert_analysis_failure(
    result: tuple[object, ...],
    expected_error: str,
    *,
    forbidden: tuple[str, ...] = (),
) -> None:
    assert result[0] is None
    assert result[1]["elem_classes"] == ["result-status", "status-error"]
    assert "### RUN ERROR" in result[1]["value"]
    assert "### FAIL" not in result[1]["value"]
    assert "No assessment was produced" in result[1]["value"]
    assert "No fallback result was generated" in result[1]["value"]
    assert expected_error in result[1]["value"]
    assert result[2:4] == (None, None)
    assert result[4] == {
        "status": "RUN_ERROR",
        "error": expected_error,
        "assessment": None,
        "scene_map": None,
    }
    assert result[5] == []
    assert result[6] == {
        "value": "",
        "interactive": False,
        "__type__": "update",
    }
    assert result[7] == {"interactive": False, "__type__": "update"}
    visible_copy = f"{result[1]['value']}\n{result[4]['error']}"
    for value in forbidden:
        assert value not in visible_copy


@pytest.mark.parametrize("missing_value", [None, ""], ids=["none", "empty"])
def test_analysis_requires_at_least_one_image_before_provider_call(
    tmp_path, missing_value
):
    from ehs_spatial.app import analyze_run

    pipeline = FakePipeline(tmp_path / "runs")

    failed = analyze_run(pipeline, *([missing_value] * 4), 1.5)

    _assert_analysis_failure(
        failed, "Upload at least one workcell view before analysis."
    )
    assert pipeline.assessment_calls == []


@pytest.mark.parametrize("provided_count", [1, 2, 3])
def test_analysis_accepts_partial_captures(tmp_path, provided_count):
    from ehs_spatial.app import analyze_run

    pipeline = FakePipeline(tmp_path / "runs")
    images = _images(tmp_path)[:provided_count] + [None] * (4 - provided_count)

    analyze_run(pipeline, *images, 1.5)

    [capture] = pipeline.assessment_calls
    assert len(capture.image_paths) == provided_count


def test_analysis_returns_real_artifacts_grounded_data_and_demo_copy(tmp_path):
    from ehs_spatial.app import analyze_run

    pipeline = FakePipeline(tmp_path / "runs")

    (
        run_id,
        status_update,
        point_cloud_path,
        topdown_path,
        structured_result,
        chat_history,
        question_update,
        ask_update,
    ) = analyze_run(pipeline, *_images(tmp_path), 1.5)

    assert re.fullmatch(r"[0-9a-f]{32}", run_id)
    [capture] = pipeline.assessment_calls
    assert capture.run_id == run_id
    assert capture.camera_height_m == 1.5
    assert capture.criterion.minimum_clearance_m == 0.6
    paths = pipeline.store.paths(run_id)
    assert point_cloud_path == str(paths.point_cloud_glb)
    assert topdown_path == str(paths.topdown_png)
    assert Path(point_cloud_path).is_file()
    assert Path(point_cloud_path).suffix == ".glb"
    assert Path(topdown_path).is_file()
    assert "approximate" in status_update["value"].lower()
    assert "Approximate boundary clearance: **0.50 m**." in status_update["value"]
    assert "**0.5 m**" not in status_update["value"]
    assert "0.6 m demo rule — not an official EHS standard" in status_update["value"]
    assert "REVIEW only" in status_update["value"]
    assert status_update["elem_classes"] == ["result-status", "status-fail"]
    assert structured_result["assessment"]["status"] == "FAIL"
    assert structured_result["scene_map"]["run_id"] == run_id
    assert structured_result["scene_map"]["facts"][0]["fact_id"] == "fact-clearance"
    assert chat_history == []
    assert question_update == {
        "value": "",
        "interactive": True,
        "__type__": "update",
    }
    assert ask_update == {"interactive": True, "__type__": "update"}


def test_analysis_card_shows_band_warnings_and_ordered_policy_lines(tmp_path):
    from ehs_spatial.app import APP_CSS, analyze_run

    class PolicyPipeline(FakePipeline):
        def run_assessment(self, capture):
            assessment = super().run_assessment(capture)
            paths = self.store.paths(capture.run_id)
            scene = self.store.load_json(paths.scene_json, SceneMap)
            self.store.save_json(
                paths.scene_json,
                scene.model_copy(
                    update={
                        "scale_source": "moge_anchor",
                        "scale_confidence": 0.87,
                        "warnings": [
                            "warning one",
                            "warning two",
                            "warning three",
                            "warning four",
                        ],
                    }
                ),
            )
            results = [
                PolicyResult(policy_id="policy-pass", status="PASS"),
                PolicyResult(
                    policy_id="policy-fail",
                    status="FAIL",
                    violations=[
                        Violation(
                            subject_id="pallet-1",
                            object_id="exit-1",
                            measured=0.4,
                            threshold=0.9,
                            unit="m",
                        )
                    ],
                ),
                PolicyResult(policy_id="policy-review", status="NEEDS_REVIEW"),
            ]
            spec = PolicySpec(
                policy_id="policy-fail",
                source_text=(
                    "Movable equipment must be kept at least 0.9 m clear of "
                    "any marked exit route at all times so egress is never "
                    "obstructed during an emergency."
                ),
                predicate="min_separation",
                subject_labels=["pallet"],
                object_labels=["exit route"],
                threshold=0.9,
            )
            paths.policies_json.write_text(
                json.dumps(
                    {
                        "specs": [spec.model_dump(mode="json")],
                        "results": [r.model_dump(mode="json") for r in results],
                    }
                ),
                encoding="utf-8",
            )
            return assessment.model_copy(
                update={
                    "status": AssessmentStatus.NEEDS_REVIEW,
                    "distance_error_budget_m": 0.2,
                }
            )

    pipeline = PolicyPipeline(tmp_path / "runs")

    result = analyze_run(pipeline, *_images(tmp_path), 1.5)
    copy = result[1]["value"]

    assert "### NEEDS_REVIEW" in copy
    assert "**0.50 m ± 0.20 m**" in copy
    assert "Scale source: `moge_anchor`, confidence 0.87." in copy
    # Worst-first ordering: FAIL before NEEDS_REVIEW before PASS.
    assert (
        copy.index("`FAIL` policy-fail")
        < copy.index("`NEEDS_REVIEW` policy-review")
        < copy.index("`PASS` policy-pass")
    )
    assert "worst 0.4m (limit 0.9m)" in copy
    assert "min_separation 0.9 m" in copy
    assert '— "Movable equipment must be kept at least 0.9 m clear' in copy
    assert "..." in copy  # 90-char source excerpt is truncated
    assert "warning one" in copy
    assert "warning three" in copy
    assert "warning four" not in copy
    assert "(+1 more in the structured output)" in copy
    assert result[1]["elem_classes"] == ["result-status", "status-needs-review"]
    assert result[4]["policy_results"][1]["policy_id"] == "policy-fail"
    assert result[4]["policy_specs"]["policy-fail"]["predicate"] == "min_separation"
    assert ".status-needs-review" in APP_CSS
    assert ".status-needs-review h3" in APP_CSS


def _write_compiled_spec(
    directory: Path, policy_id: str, *, unsupported: str | None = None
) -> PolicySpec:
    spec = PolicySpec(
        policy_id=policy_id,
        source_text=(
            "Movable equipment must be kept at least 0.6 m clear of any "
            "machine guarding fence at all times."
        ),
        predicate="min_separation",
        subject_labels=["pallet"],
        object_labels=["safety fence"],
        threshold=0.6,
        unsupported_reason=unsupported,
    )
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{policy_id}.json").write_text(
        spec.model_dump_json(indent=2), encoding="utf-8"
    )
    return spec


def test_load_policy_specs_reads_compiled_dir_and_skips_unreadable(tmp_path):
    from ehs_spatial.app import load_policy_specs

    compiled = tmp_path / "compiled"
    _write_compiled_spec(compiled, "p02-later")
    _write_compiled_spec(compiled, "p01-first")
    (compiled / "broken.json").write_text("{", encoding="utf-8")

    specs = load_policy_specs(compiled)

    assert [spec.policy_id for spec in specs] == ["p01-first", "p02-later"]
    assert load_policy_specs(tmp_path / "missing") == []


def test_analyze_run_puts_selected_build_time_specs_into_capture_policies(
    tmp_path,
):
    from ehs_spatial.app import analyze_run

    pipeline = FakePipeline(tmp_path / "runs")
    spec = _write_compiled_spec(tmp_path / "compiled", "p01-clearance")
    available = {spec.policy_id: spec}

    analyze_run(
        pipeline, *_images(tmp_path), 1.5, ["p01-clearance", "ghost"], available
    )
    no_selection = analyze_run(pipeline, *_images(tmp_path), 1.5)

    first, second = pipeline.assessment_calls
    # Only reviewed build-time specs make it in; unknown ids degrade silently.
    assert [policy.policy_id for policy in first.policies] == ["p01-clearance"]
    assert first.policies[0] == spec
    assert second.policies == []
    assert no_selection[0] is not None


def test_build_app_policy_accordion_offers_specs_and_shows_refusals(tmp_path):
    from ehs_spatial.app import POLICY_REVIEW_COPY, build_app

    compiled = tmp_path / "compiled"
    supported = _write_compiled_spec(compiled, "p01-clearance")
    refused = _write_compiled_spec(
        compiled,
        "p02-repaint",
        unsupported="cannot verify paint colour or a two-year schedule",
    )
    (compiled / "broken.json").write_text("{", encoding="utf-8")
    expected_label = f"{supported.policy_id} — {supported.source_text[:60]}"

    demo = build_app(FakePipeline(tmp_path / "runs"), policies_dir=compiled)
    config = demo.get_config_file()
    components = config["components"]

    [selector] = [
        component
        for component in components
        if component["type"] == "checkboxgroup"
    ]
    assert [list(choice) for choice in selector["props"]["choices"]] == [
        [expected_label, "p01-clearance"]
    ]
    assert selector["props"]["value"] == []
    # The refusal is a feature: the unsupported spec appears as a disabled
    # checkbox carrying the compiler's reason, never as a selectable option.
    [refusal_box] = [
        component
        for component in components
        if component["type"] == "checkbox"
        and component["props"].get("label", "").startswith("p02-repaint")
    ]
    assert refusal_box["props"]["interactive"] is False
    assert refused.unsupported_reason in refusal_box["props"]["info"]
    assert "Compiler refusal" in refusal_box["props"]["info"]
    # Reviewed-offline copy is stated in the UI.
    assert any(
        POLICY_REVIEW_COPY in (component["props"].get("value") or "")
        for component in components
        if component["type"] == "markdown"
    )
    # The selector feeds the analyze event.
    [analyze_dependency] = [
        dependency
        for dependency in config["dependencies"]
        if dependency.get("api_name") == "analyze_workcell"
    ]
    assert selector["id"] in analyze_dependency["inputs"]


def test_app_event_carries_policy_selection_into_capture_run(tmp_path):
    from ehs_spatial.app import build_app

    pipeline = FakePipeline(tmp_path / "runs")
    spec = _write_compiled_spec(tmp_path / "compiled", "p01-clearance")
    demo = build_app(pipeline, policies_dir=tmp_path / "compiled")
    state = SessionState(demo)
    analyze = next(
        function
        for function in demo.fns.values()
        if function.api_name == "analyze_workcell"
    )
    upload_data = [
        {"path": path, "meta": {"_type": "gradio.FileData"}}
        for path in _images(tmp_path)
    ]

    asyncio.run(
        demo.process_api(
            analyze,
            [*upload_data, 1.5, ["p01-clearance"]],
            state=state,
            session_hash="policy-selection",
        )
    )

    [capture] = pipeline.assessment_calls
    assert [policy.policy_id for policy in capture.policies] == [spec.policy_id]
    assert capture.policies[0].threshold == 0.6


def test_failed_reanalysis_returns_atomic_cleared_state_without_fallback(tmp_path):
    from ehs_spatial.app import APP_CSS, analyze_run

    pipeline = FakePipeline(tmp_path / "runs")
    images = _images(tmp_path)
    previous_run = analyze_run(pipeline, *images, 1.5)[0]
    original_message = (
        "service unavailable; Authorization: Bearer SECRET_SENTINEL"
    )
    error = ProviderError(
        "replicate",
        "map_anything.run",
        original_message,
    )
    pipeline.run_error = error

    failed = analyze_run(pipeline, *images, 1.5)

    assert previous_run is not None
    assert error.original_message == original_message
    _assert_analysis_failure(
        failed,
        (
            "replicate map_anything.run failed. Check provider credentials, "
            "quota, and service status, then retry."
        ),
        forbidden=("service unavailable", "SECRET_SENTINEL"),
    )
    assert ".status-error" in APP_CSS


def test_analysis_returns_cleared_state_for_capture_validation_error(tmp_path):
    from ehs_spatial.app import analyze_run

    pipeline = FakePipeline(tmp_path / "runs")

    failed = analyze_run(pipeline, *_images(tmp_path), 0)

    _assert_analysis_failure(
        failed,
        (
            "Local processing failed (ValidationError). Check the uploaded "
            "files and local artifacts, then retry."
        ),
    )
    assert pipeline.assessment_calls == []


def test_analysis_returns_cleared_state_for_corrupt_scene_artifact(tmp_path):
    from ehs_spatial.app import analyze_run

    class CorruptScenePipeline(FakePipeline):
        def run_assessment(self, capture):
            assessment = super().run_assessment(capture)
            self.store.paths(capture.run_id).scene_json.write_text("{")
            return assessment

    pipeline = CorruptScenePipeline(tmp_path / "runs")

    failed = analyze_run(pipeline, *_images(tmp_path), 1.5)

    _assert_analysis_failure(
        failed,
        (
            "Local processing failed (JSONDecodeError). Check the uploaded "
            "files and local artifacts, then retry."
        ),
    )


def test_chat_requires_successful_run_before_provider_call(tmp_path):
    from ehs_spatial.app import answer_run_question

    pipeline = FakePipeline(tmp_path / "runs")

    with pytest.raises(gr.Error, match="Analyze a workcell"):
        answer_run_question(pipeline, "How far is the pallet?", [], None)

    assert pipeline.question_calls == []


def test_chat_rejects_blank_question_before_provider_call(tmp_path):
    from ehs_spatial.app import answer_run_question

    pipeline = FakePipeline(tmp_path / "runs")

    with pytest.raises(gr.Error, match="Enter a question"):
        answer_run_question(pipeline, "   ", [], "run-1")

    assert pipeline.question_calls == []


def test_chat_appends_message_dicts_and_locally_grounded_fact_ids(tmp_path):
    from ehs_spatial.app import analyze_run, answer_run_question

    pipeline = FakePipeline(tmp_path / "runs")
    run_id = analyze_run(pipeline, *_images(tmp_path), 1.5)[0]

    history, cleared_question = answer_run_question(
        pipeline, "  How far is the pallet?  ", [], run_id
    )

    assert pipeline.question_calls == [(run_id, "How far is the pallet?")]
    assert history == [
        {"role": "user", "content": "How far is the pallet?"},
        {
            "role": "assistant",
            "content": (
                "SceneMap facts: minimum boundary clearance = 0.5 m.\n\n"
                "Fact IDs: `fact-clearance`"
            ),
        },
    ]
    assert cleared_question == ""


def test_chat_never_exposes_provider_original_message(tmp_path):
    from ehs_spatial.app import answer_run_question

    provider_failures = [
        ("basic", "Basic BASIC_SENTINEL", "BASIC_SENTINEL"),
        (
            "aws",
            "AWS4-HMAC-SHA256 Credential=example/request, "
            "SignedHeaders=content-type;host;x-amz-date, "
            "Signature=AWS_SENTINEL",
            "AWS_SENTINEL",
        ),
        (
            "digest",
            'Digest username="operator", '
            'nonce="nonce", response="DIGEST_SENTINEL"',
            "DIGEST_SENTINEL",
        ),
        (
            "query key",
            "request failed: /endpoint?key=QUERY_SENTINEL&status=401",
            "QUERY_SENTINEL",
        ),
        (
            "escaped quote",
            r'upstream said \"credential=ESCAPED_SENTINEL\"',
            "ESCAPED_SENTINEL",
        ),
        (
            "arbitrary",
            "opaque upstream diagnostic ARBITRARY_SENTINEL",
            "ARBITRARY_SENTINEL",
        ),
    ]
    pipeline = FakePipeline(tmp_path / "runs")
    history = [{"role": "user", "content": "Earlier question"}]
    expected_public_copy = (
        "gemini chat.create failed. Check provider credentials, quota, and "
        "service status, then retry."
    )

    for case_name, original_message, sentinel in provider_failures:
        error = ProviderError("gemini", "chat.create", original_message)
        pipeline.question_error = error

        with pytest.raises(gr.Error) as caught:
            answer_run_question(
                pipeline, "What is the clearance?", history, "run-1"
            )

        assert error.original_message == original_message, case_name
        assert caught.value.message == expected_public_copy, case_name
        assert sentinel not in caught.value.message, case_name

    assert history == [{"role": "user", "content": "Earlier question"}]


def test_successful_new_analysis_replaces_run_and_clears_chat(tmp_path):
    from ehs_spatial.app import analyze_run, answer_run_question

    pipeline = FakePipeline(tmp_path / "runs")
    images = _images(tmp_path)
    first_run = analyze_run(pipeline, *images, 1.5)[0]
    history, _ = answer_run_question(pipeline, "How far?", [], first_run)
    assert history

    second = analyze_run(pipeline, *images, 1.5)

    assert second[0] != first_run
    assert second[5] == []
    assert second[6]["value"] == ""


@pytest.mark.parametrize(
    "failure_mode",
    ["provider", "runtime", "incomplete"],
    ids=["provider-error", "runtime-error", "incomplete-capture"],
)
def test_failed_reanalysis_clears_gradio_state_and_blocks_old_run_questions(
    tmp_path, failure_mode
):
    from ehs_spatial.app import build_app

    pipeline = FakePipeline(tmp_path / "runs")
    demo = build_app(pipeline)
    state = SessionState(demo)
    analyze = next(
        function
        for function in demo.fns.values()
        if function.api_name == "analyze_workcell"
    )
    ask = next(
        function
        for function in demo.fns.values()
        if function.api_name == "ask_about_run"
    )
    upload_data = [
        {"path": path, "meta": {"_type": "gradio.FileData"}}
        for path in _images(tmp_path)
    ]

    async def exercise_events() -> None:
        await demo.process_api(
            analyze,
            [*upload_data, 1.5, []],
            state=state,
            session_hash="failed-reanalysis-regression",
        )
        previous_run = state[analyze.outputs[0]._id]
        assert previous_run is not None

        replacement_inputs = [*upload_data, 1.5, []]
        if failure_mode == "provider":
            pipeline.run_error = ProviderError(
                "replicate",
                "map_anything.run",
                "provider detail PROVIDER_SENTINEL",
            )
            expected_error = (
                "replicate map_anything.run failed. Check provider credentials, "
                "quota, and service status, then retry."
            )
            forbidden = "PROVIDER_SENTINEL"
        elif failure_mode == "runtime":
            pipeline.run_error = RuntimeError(
                "local detail LOCAL_RUNTIME_SENTINEL"
            )
            expected_error = (
                "Local processing failed (RuntimeError). Check the uploaded "
                "files and local artifacts, then retry."
            )
            forbidden = "LOCAL_RUNTIME_SENTINEL"
        else:
            replacement_inputs[:4] = [None] * 4
            expected_error = "Upload at least one workcell view before analysis."
            forbidden = ""

        failed = await demo.process_api(
            analyze,
            replacement_inputs,
            state=state,
            session_hash=f"failed-reanalysis-{failure_mode}",
        )

        assert state[analyze.outputs[0]._id] is None
        assert failed["data"][1]["elem_classes"] == [
            "result-status",
            "status-error",
        ]
        assert failed["data"][2:4] == [None, None]
        assert failed["data"][4].root == {
            "status": "RUN_ERROR",
            "error": expected_error,
            "assessment": None,
            "scene_map": None,
        }
        assert failed["data"][5] == []
        assert failed["data"][6]["interactive"] is False
        assert failed["data"][7]["interactive"] is False
        visible_copy = (
            f"{failed['data'][1]['value']}\n{failed['data'][4].root['error']}"
        )
        assert expected_error in visible_copy
        if forbidden:
            assert forbidden not in visible_copy

        calls_before_ask = list(pipeline.question_calls)
        with pytest.raises(gr.Error, match="Analyze a workcell"):
            await demo.process_api(
                ask,
                ["What is the clearance?", [], None],
                state=state,
                session_hash=f"failed-reanalysis-{failure_mode}",
            )
        assert pipeline.question_calls == calls_before_ask

    asyncio.run(exercise_events())


def test_build_app_has_required_gradio_620_components_events_and_serialization(tmp_path):
    from ehs_spatial.app import build_app

    demo = build_app(FakePipeline(tmp_path / "runs"))
    config = demo.get_config_file()
    components = config["components"]

    uploads = [
        component
        for component in components
        if component["type"] == "image"
        and component["props"].get("type") == "filepath"
        and component["props"].get("sources") == ["upload"]
        and component["props"].get("interactive") is True
    ]
    assert len(uploads) == 4
    assert [component["props"]["label"] for component in uploads] == [
        "View 1 — workcell front (required)",
        "View 2 — workcell right (optional)",
        "View 3 — workcell rear (optional)",
        "View 4 — workcell left (optional)",
    ]
    assert any(
        component["type"] == "number" and component["props"].get("value") == 1.5
        for component in components
    )
    assert any(
        component["type"] == "model3d"
        and component["props"].get("display_mode") == "point_cloud"
        for component in components
    )
    assert any(
        component["type"] == "image"
        and component["props"].get("label") == "Top-down evidence"
        for component in components
    )
    assert any(
        component["type"] == "json"
        and component["props"].get("label") == "Assessment + SceneMap"
        for component in components
    )
    # two chatbots now: workbench grounded QA and the report-page agent hub
    [chatbot] = [
        component for component in components
        if component["type"] == "chatbot"
        and component["props"].get("label") == "Grounded answers"
    ]
    assert "type" not in chatbot["props"]
    assert chatbot["props"]["value"] == []

    buttons = {
        component["props"].get("value"): component["id"]
        for component in components
        if component["type"] == "button"
    }
    assert "Analyze workcell" in buttons
    assert "Ask about this run" in buttons
    [question] = [
        component
        for component in components
        if component["type"] == "textbox"
        and component["props"].get("label") == "Question"
    ]
    targets = {
        tuple(target)
        for dependency in config["dependencies"]
        for target in dependency["targets"]
    }
    assert (buttons["Analyze workcell"], "click") in targets
    assert (buttons["Ask about this run"], "click") in targets
    assert (question["id"], "submit") in targets
    assert demo.api_open is False
    assert demo._queue.default_concurrency_limit == 1
    # Three deliberate lanes: one serializing provider spend, one keeping
    # local History/disposition handlers responsive during an analysis, and
    # one for the read-only report view so it never queues behind the
    # History table's full-runs scan on page load.
    assert {function.concurrency_id for function in demo.fns.values()} == {
        "ehs-provider-pipeline",
        "ehs-local-ui",
        "ehs-report-ui",
    }
    assert {function.concurrency_limit for function in demo.fns.values()} == {1}
    provider_lanes = {
        function.api_name: function.concurrency_id
        for function in demo.fns.values()
        if function.api_name
        in {"analyze_workcell", "ask_about_run", "ask_about_run_from_enter"}
    }
    # Every provider-touching handler shares exactly the one provider lane.
    assert set(provider_lanes.values()) == {"ehs-provider-pipeline"}
    assert len(provider_lanes) == 3


def test_load_run_evidence_lists_overlays_and_viewer_download(tmp_path):
    from ehs_spatial.app import load_run_evidence

    pipeline = FakePipeline(tmp_path / "runs")
    paths = pipeline.store.paths("run-9")
    paths.evidence_dir.mkdir(parents=True)
    for name in ("frame_0002_overlay.png", "frame_0001_overlay.png"):
        Image.new("RGB", (4, 4), "white").save(paths.evidence_dir / name)
    (paths.evidence_dir / "notes.txt").write_text("not an overlay")
    paths.viewer_html.write_text("<!doctype html>")

    overlays, viewer = load_run_evidence(pipeline, "run-9")

    assert overlays == [
        (str(paths.evidence_dir / "frame_0001_overlay.png"), "frame_0001"),
        (str(paths.evidence_dir / "frame_0002_overlay.png"), "frame_0002"),
    ]
    assert viewer == str(paths.viewer_html)


def test_load_run_evidence_is_empty_without_run_or_artifacts(tmp_path):
    from ehs_spatial.app import load_run_evidence

    pipeline = FakePipeline(tmp_path / "runs")

    # Cleared run id (failed re-analysis) and a run whose fail-soft evidence
    # was never written both leave the section empty, not broken.
    assert load_run_evidence(pipeline, None) == ([], None)
    assert load_run_evidence(pipeline, "never-ran") == ([], None)


def test_build_app_wires_evidence_section_to_run_id_changes(tmp_path):
    from ehs_spatial.app import build_app

    demo = build_app(FakePipeline(tmp_path / "runs"))
    config = demo.get_config_file()
    components = config["components"]

    assert any(
        component["type"] == "gallery"
        and component["props"].get("label") == "Mask overlays on the captured views"
        for component in components
    )
    assert any(
        component["type"] == "file"
        and "viewer.html" in (component["props"].get("label") or "")
        for component in components
    )
    [evidence_fn] = [
        function
        for function in demo.fns.values()
        if function.api_name == "load_run_evidence"
    ]
    # Local disk read: it must not queue behind a provider analysis.
    assert evidence_fn.concurrency_id == "ehs-local-ui"
    assert evidence_fn.concurrency_limit == 1
    [state] = [component for component in components if component["type"] == "state"]
    targets = {
        tuple(target)
        for dependency in config["dependencies"]
        for target in dependency["targets"]
    }
    assert (state["id"], "change") in targets


def test_root_app_import_does_not_launch(monkeypatch):
    launches = []
    monkeypatch.setattr(gr.Blocks, "launch", lambda self, **kwargs: launches.append(kwargs))

    _load_root_app()

    assert launches == []


def test_root_main_launches_built_app_with_css(monkeypatch):
    module = _load_root_app()

    class Demo:
        def __init__(self):
            self.launches = []

        def launch(self, **kwargs):
            self.launches.append(kwargs)

    demo = Demo()
    monkeypatch.setattr(module, "build_app", lambda: demo)

    module.main()

    assert demo.launches == [{"css": module.APP_CSS, "footer_links": []}]


def _write_history_run(
    store: ArtifactStore,
    run_id: str,
    *,
    status: str = "NEEDS_REVIEW",
    distance: float = 0.58,
) -> None:
    paths = store.paths(run_id)
    paths.root.mkdir(parents=True)
    paths.manifest_json.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "created_at": "2026-08-25T10:00:00+00:00",
                "operator": "inspector-a",
                "capture_tier": "mono",
            }
        ),
        encoding="utf-8",
    )
    store.save_json(
        paths.assessment_json,
        Assessment(
            status=status,
            fact_ids=[],
            evidence_frame_ids=[],
            approximate_distance_m=distance,
            distance_error_budget_m=0.05,
        ),
    )
    Image.new("RGB", (16, 16), "white").save(paths.topdown_png)


def test_list_history_rows_come_from_the_store_index(tmp_path):
    from ehs_spatial.app import list_history

    pipeline = FakePipeline(tmp_path / "runs")
    _write_history_run(pipeline.store, "run-a")

    assert list_history(pipeline) == [
        [
            "run-a",
            "2026-08-25T10:00:00+00:00",
            "inspector-a",
            "mono",
            "NEEDS_REVIEW",
            None,
            "0.58 ± 0.05 m",
            None,
        ]
    ]


def test_list_history_surfaces_worst_policy_and_banded_distance(tmp_path):
    """A demo-rule PASS with a FAILing compiled policy must not scan as a
    clean PASS row, and the distance column carries its error budget."""
    from ehs_spatial.app import list_history

    pipeline = FakePipeline(tmp_path / "runs")
    _write_history_run(pipeline.store, "run-a", status="PASS")
    pipeline.store.paths("run-a").policies_json.write_text(
        json.dumps(
            {
                "specs": [],
                "results": [
                    {"policy_id": "p-pass", "status": "PASS"},
                    {"policy_id": "p-fail", "status": "FAIL"},
                ],
            }
        ),
        encoding="utf-8",
    )
    _write_history_run(pipeline.store, "run-no-budget")
    paths = pipeline.store.paths("run-no-budget")
    pipeline.store.save_json(
        paths.assessment_json,
        Assessment(
            status="NEEDS_REVIEW",
            fact_ids=[],
            evidence_frame_ids=[],
            approximate_distance_m=0.58,
        ),
    )

    rows = {row[0]: row for row in list_history(pipeline)}

    assert rows["run-a"][4] == "PASS"
    assert rows["run-a"][5] == "FAIL"
    assert rows["run-a"][6] == "0.58 ± 0.05 m"
    # Without a budget the distance still carries its unit, never a bare 0.58.
    assert rows["run-no-budget"][6] == "0.58 m"


def test_load_history_run_loads_card_topdown_and_disposition(tmp_path):
    from ehs_spatial.app import list_history, load_history_run

    pipeline = FakePipeline(tmp_path / "runs")
    _write_history_run(pipeline.store, "run-a")
    paths = pipeline.store.paths("run-a")
    pipeline.store.save_json(
        paths.review_json,
        ReviewDisposition(
            run_id="run-a",
            reviewer="casey",
            decision="confirmed",
            created_at="2026-08-25T11:00:00+00:00",
        ),
    )
    rows = list_history(pipeline)
    evt = gr.SelectData(None, {"index": (0, 0), "value": "run-a"})

    (
        run_id,
        card,
        topdown,
        disposition,
        reviewer_reset,
        decision_reset,
        override_reset,
        reason_reset,
    ) = load_history_run(pipeline, rows, evt)

    assert run_id == "run-a"
    assert "### NEEDS_REVIEW" in card["value"]
    assert "0.58 m ± 0.05 m" in card["value"]
    assert "status-needs-review" in card["elem_classes"]
    assert topdown == str(paths.topdown_png)
    assert "casey" in disposition["value"]
    assert "confirmed" in disposition["value"]
    # Selecting a row resets the disposition form so a ruling composed
    # against another run cannot be one-click saved onto this one.
    assert reviewer_reset["value"] == ""
    assert decision_reset["value"] == "confirmed"
    assert override_reset["value"] is None
    assert reason_reset["value"] == ""


def test_load_history_run_degrades_when_artifacts_are_missing(tmp_path):
    from ehs_spatial.app import load_history_run

    pipeline = FakePipeline(tmp_path / "runs")
    pipeline.store.paths("run-empty").root.mkdir(parents=True)
    evt = gr.SelectData(None, {"index": (0, 0), "value": "run-empty"})

    run_id, card, topdown, disposition, *form_resets = load_history_run(
        pipeline, [["run-empty"]], evt
    )

    assert run_id == "run-empty"
    assert "NO ASSESSMENT" in card["value"]
    assert topdown is None
    assert "No disposition recorded" in disposition["value"]
    assert len(form_resets) == 4

    with pytest.raises(gr.Error, match="Select a run"):
        load_history_run(
            pipeline, [], gr.SelectData(None, {"index": (3, 0), "value": None})
        )


@pytest.mark.parametrize(
    ("policies_payload", "expected_line"),
    [
        # Result row missing policy_id defaults to '?', like report.py.
        ({"specs": [], "results": [{"status": "PASS"}]}, "`PASS` ?"),
        # Legacy bare list: the non-dict entry is skipped, the rest renders.
        (["oops", {"policy_id": "p-1", "status": "FAIL"}], "`FAIL` p-1"),
    ],
    ids=["missing-keys", "non-dict-row"],
)
def test_load_history_run_tolerates_malformed_policies_like_report(
    tmp_path, policies_payload, expected_line
):
    """The card degrades on a malformed policies.json exactly where report.py
    already does, instead of KeyError/AttributeError-ing the whole tab."""
    from ehs_spatial.app import load_history_run

    pipeline = FakePipeline(tmp_path / "runs")
    _write_history_run(pipeline.store, "run-a")
    pipeline.store.paths("run-a").policies_json.write_text(
        json.dumps(policies_payload), encoding="utf-8"
    )
    evt = gr.SelectData(None, {"index": (0, 0), "value": "run-a"})

    run_id, card, *_ = load_history_run(pipeline, [["run-a"]], evt)

    assert run_id == "run-a"
    assert "### NEEDS_REVIEW" in card["value"]
    assert "**Policies:**" in card["value"]
    assert expected_line in card["value"]
    assert "oops" not in card["value"]


def test_save_disposition_confirm_writes_a_valid_review_layer(tmp_path):
    from ehs_spatial.app import save_disposition

    pipeline = FakePipeline(tmp_path / "runs")
    _write_history_run(pipeline.store, "run-a")
    paths = pipeline.store.paths("run-a")
    assessment_before = paths.assessment_json.read_bytes()

    save_disposition(pipeline, "run-a", "casey", "confirmed", None, "")

    review = pipeline.store.load_json(paths.review_json, ReviewDisposition)
    assert review.run_id == "run-a"
    assert review.reviewer == "casey"
    assert review.decision == "confirmed"
    assert review.overridden_status is None
    # created_at is real UTC ISO-8601.
    from datetime import datetime, timezone

    parsed = datetime.fromisoformat(review.created_at)
    assert parsed.utcoffset() == timezone.utc.utcoffset(None)
    # The machine verdict is never rewritten: disposition is a separate layer.
    assert paths.assessment_json.read_bytes() == assessment_before


def test_save_disposition_override_requires_reason_and_records_status(tmp_path):
    from ehs_spatial.app import save_disposition

    pipeline = FakePipeline(tmp_path / "runs")
    _write_history_run(pipeline.store, "run-a")
    paths = pipeline.store.paths("run-a")

    with pytest.raises(gr.Error, match="requires a reason"):
        save_disposition(pipeline, "run-a", "casey", "overridden", "PASS", "  ")
    assert not paths.review_json.exists()

    with pytest.raises(gr.Error, match="status the override asserts"):
        save_disposition(pipeline, "run-a", "casey", "overridden", None, "why")

    with pytest.raises(gr.Error, match="reviewer name"):
        save_disposition(pipeline, "run-a", " ", "overridden", "PASS", "why")

    with pytest.raises(gr.Error, match="Select a run"):
        save_disposition(pipeline, None, "casey", "confirmed", None, "")

    save_disposition(
        pipeline, "run-a", "casey", "overridden", "PASS", "fence moved"
    )

    review = pipeline.store.load_json(paths.review_json, ReviewDisposition)
    assert review.decision == "overridden"
    assert review.overridden_status == AssessmentStatus.PASS
    assert review.reason == "fence moved"


def test_save_disposition_rejects_override_to_the_machine_status(tmp_path):
    from ehs_spatial.app import save_disposition

    pipeline = FakePipeline(tmp_path / "runs")
    _write_history_run(pipeline.store, "run-a", status="NEEDS_REVIEW")
    paths = pipeline.store.paths("run-a")

    with pytest.raises(gr.Error, match="matches the machine verdict"):
        save_disposition(
            pipeline, "run-a", "casey", "overridden", "NEEDS_REVIEW", "why"
        )
    assert not paths.review_json.exists()


def test_save_disposition_rejects_runs_without_an_assessment(tmp_path):
    """No machine verdict means nothing to confirm or override — and a save
    must not recreate a deleted run directory as a ghost run."""
    from ehs_spatial.app import save_disposition

    pipeline = FakePipeline(tmp_path / "runs")
    paths = pipeline.store.paths("run-partial")
    paths.root.mkdir(parents=True)

    with pytest.raises(gr.Error, match="no machine assessment"):
        save_disposition(pipeline, "run-partial", "casey", "confirmed", None, "")
    assert not paths.review_json.exists()

    ghost = pipeline.store.paths("run-deleted")
    with pytest.raises(gr.Error, match="no machine assessment"):
        save_disposition(
            pipeline, "run-deleted", "casey", "overridden", "PASS", "why"
        )
    assert not ghost.root.exists()


def test_save_disposition_returns_refreshed_history_rows(tmp_path):
    from ehs_spatial.app import HISTORY_HEADERS, save_disposition

    pipeline = FakePipeline(tmp_path / "runs")
    _write_history_run(pipeline.store, "run-a")

    disposition_update, rows = save_disposition(
        pipeline, "run-a", "casey", "confirmed", None, ""
    )

    assert "casey" in disposition_update["value"]
    [row] = rows
    assert row[0] == "run-a"
    assert row[HISTORY_HEADERS.index("disposition")] == "confirmed"


def test_build_app_report_is_the_history(tmp_path):
    """Product decision: the History tab is gone — the report tab's
    time-sorted dropdown IS the run history."""
    from ehs_spatial.app import build_app

    demo = build_app(FakePipeline(tmp_path / "runs"))
    config = demo.get_config_file()
    components = config["components"]

    # no History surface remains
    assert not [c for c in components if c["type"] == "dataframe"]
    buttons = {
        c["props"].get("value") for c in components if c["type"] == "button"
    }
    assert "Refresh history" not in buttons
    assert "Save disposition" not in buttons
    api_names = {
        fn.api_name for fn in demo.fns.values() if fn.api_name
    }
    assert "list_history" not in api_names
    assert "save_disposition" not in api_names

    # report dropdown exists and evidence loading keeps its local lane
    [report_dropdown] = [
        c for c in components
        if c["type"] == "dropdown"
        and "历史" in str(c["props"].get("label"))
    ]
    assert report_dropdown["props"].get("allow_custom_value") is True
    local_lanes = {
        fn.api_name: fn.concurrency_id
        for fn in demo.fns.values()
        if fn.api_name == "load_run_evidence"
    }
    assert set(local_lanes.values()) == {"ehs-local-ui"}

def test_readme_links_each_provider_credential_source():
    readme = (Path(__file__).parents[1] / "README.md").read_text()
    credential_urls = {
        "REPLICATE_API_TOKEN": "https://replicate.com/account/api-tokens",
        "FAL_KEY": "https://fal.ai/dashboard/keys",
        "GEMINI_API_KEY": "https://aistudio.google.com/apikey",
    }

    for variable, url in credential_urls.items():
        assert re.search(
            rf"(?m)^- `{variable}`: \[[^]]+\]\({re.escape(url)}\)$",
            readme,
        )


# --------------------------------------------------------------- Video tab
def _video_report(run_id: str, *, abstained: str | None = None) -> dict:
    frames = [0, 30, 60]
    timeline = (
        {str(f): "NO_DATA" for f in frames}
        if abstained
        else {"0": "PASS", "30": "FAIL", "60": "NEEDS_REVIEW"}
    )
    return {
        "run_id": run_id,
        "video": "clip.avi",
        "sampled_frame_ids": frames,
        "step_seconds": 1.0,
        "tier": {
            "capture_tier": "video-mono",
            "band_m": 0.35,
            "calibrated_reference": "calibrated fixed-camera tier reference",
        },
        "floor": (
            {"fitted": False, "reason": "no keyframe produced a floor fit"}
            if abstained
            else {
                "fitted": True,
                "camera_height_m": 5.2,
                "inlier_fraction": 0.61,
                "height_spread_m": 0.3,
            }
        ),
        "abstained": abstained,
        "spend": {"sam_calls": 6, "sam_cost_usd": 0.06, "moge_keyframes": 3},
        "timelines": {
            "R1_zone": timeline,
            "R2_min_distance": timeline,
            "R3_speed": timeline,
        },
        "verdicts": {
            "R1_zone": "NO_DATA" if abstained else "FAIL",
            "R2_min_distance": "NO_DATA" if abstained else "FAIL",
            "R3_speed": "NO_DATA" if abstained else "NEEDS_REVIEW",
            "overall": "NO_DATA" if abstained else "FAIL",
        },
    }


def _write_video_run(
    store: ArtifactStore, run_id: str, created_at: str, report: dict | None = None
) -> None:
    from ehs_spatial.video import video_paths

    paths = video_paths(store, run_id)
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.report_json.write_text(
        json.dumps(report or _video_report(run_id)), encoding="utf-8"
    )
    paths.manifest_json.write_text(
        json.dumps({"created_at": created_at, "capture_tier": "video-mono"}),
        encoding="utf-8",
    )
    Image.new("RGB", (16, 16), "gray").save(paths.topdown_png)
    Image.new("RGB", (16, 16), "gray").save(paths.overlay_gif, format="GIF")


def test_video_cost_copy_is_explicit_about_live_spend():
    from ehs_spatial.app import video_cost_copy

    copy = video_cost_copy(1.0, "person, forklift")
    assert "~120 SAM calls" in copy
    assert "$1.20" in copy
    assert "live provider spend" in copy
    # Fewer labels and slower sampling change the estimate honestly.
    assert "~60 SAM calls" in video_cost_copy(0.5, "person")
    assert "120 s" in video_cost_copy(0.5, "person")


def test_analyze_video_requires_upload_and_labels(tmp_path):
    from ehs_spatial.app import analyze_video

    pipeline = FakePipeline(tmp_path / "runs")
    status, gif, topdown, payload, _ = analyze_video(
        pipeline, None, 1.0, "person", ""
    )
    assert "RUN ERROR" in status["value"]
    assert "Upload a video" in status["value"]
    assert gif is None and topdown is None
    assert payload["status"] == "RUN_ERROR"
    status, *_ = analyze_video(pipeline, "clip.avi", 1.0, " , ", "")
    assert "at least one object label" in status["value"]


def test_analyze_video_provider_error_card_without_fallback(
    tmp_path, monkeypatch
):
    import ehs_spatial.app as app_module
    from ehs_spatial.app import analyze_video

    def broken(*args, **kwargs):
        raise ProviderError("fal", "sam3.video", "quota exhausted")

    monkeypatch.setattr(app_module, "run_video_assessment", broken)
    pipeline = FakePipeline(tmp_path / "runs")
    status, gif, topdown, payload, dropdown = analyze_video(
        pipeline, str(tmp_path / "clip.avi"), 1.0, "person, forklift", ""
    )
    assert "RUN ERROR" in status["value"]
    assert "fal sam3.video failed" in status["value"]
    assert "No fallback result was generated" in status["value"]
    assert "quota exhausted" not in status["value"]  # raw provider text hidden
    assert gif is None and topdown is None
    assert payload["report"] is None


def test_analyze_video_success_renders_report_and_refreshes_replay(
    tmp_path, monkeypatch
):
    import ehs_spatial.app as app_module
    from ehs_spatial.app import analyze_video

    pipeline = FakePipeline(tmp_path / "runs")
    captured = {}

    def fake_runner(video_path, *, store, run_id, **kwargs):
        captured.update({"video_path": video_path, "run_id": run_id, **kwargs})
        report = _video_report(run_id)
        _write_video_run(store, run_id, "2026-08-25T10:00:00+00:00", report)
        return report

    monkeypatch.setattr(app_module, "run_video_assessment", fake_runner)
    status, gif, topdown, payload, dropdown = analyze_video(
        pipeline,
        str(tmp_path / "clip.avi"),
        0.5,
        "person, forklift",
        "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))",
    )
    assert captured["sample_fps"] == 0.5
    assert captured["labels"] == ("person", "forklift")
    assert captured["zone_wkt"].startswith("POLYGON")
    assert "### FAIL" in status["value"]
    assert "R2 person-to-vehicle" in status["value"]
    assert "camera height 5.2 m" in status["value"]
    assert "video-mono" in status["value"]
    assert "status-fail" in status["elem_classes"]
    assert gif is not None and Path(gif).is_file()
    assert topdown is not None and Path(topdown).is_file()
    assert payload["verdicts"]["overall"] == "FAIL"
    run_id = captured["run_id"]
    assert dropdown["value"] == run_id
    assert any(choice[1] == run_id for choice in dropdown["choices"])


def test_video_abstention_card_is_honest_not_a_fallback(tmp_path):
    from ehs_spatial.app import _video_status_copy

    card, status_class = _video_status_copy(
        _video_report(
            "run-1",
            abstained="floor fit failed: no keyframe produced a floor fit",
        )
    )
    assert "NO VERDICT" in card
    assert "floor fit failed" in card
    assert "no fallback result was generated" in card.lower()
    assert status_class == "status-insufficient-evidence"


def test_list_video_runs_newest_first_and_degrades(tmp_path):
    from ehs_spatial.app import list_video_runs
    from ehs_spatial.video import video_paths

    pipeline = FakePipeline(tmp_path / "runs")
    _write_video_run(pipeline.store, "older", "2026-08-24T10:00:00+00:00")
    _write_video_run(pipeline.store, "newer", "2026-08-25T10:00:00+00:00")
    # Corrupt manifest: still listed, label degrades to the bare run id.
    corrupt = video_paths(pipeline.store, "corrupt-manifest")
    corrupt.root.mkdir(parents=True, exist_ok=True)
    corrupt.report_json.write_text("{}", encoding="utf-8")
    corrupt.manifest_json.write_text("not json", encoding="utf-8")
    # A photo run (no video_report.json) is not offered for video replay.
    photo = pipeline.store.paths("photo-run")
    photo.root.mkdir(parents=True, exist_ok=True)
    photo.manifest_json.write_text("{}", encoding="utf-8")

    choices = list_video_runs(pipeline)
    assert [run_id for _, run_id in choices] == [
        "newer",
        "older",
        "corrupt-manifest",
    ]
    assert choices[0][0].startswith("newer — 2026-08-25")
    assert choices[2][0] == "corrupt-manifest"


def test_load_video_run_replays_cached_artifacts_without_spend(tmp_path):
    from ehs_spatial.app import load_video_run

    pipeline = FakePipeline(tmp_path / "runs")
    _write_video_run(pipeline.store, "cached", "2026-08-25T10:00:00+00:00")
    status, gif, topdown, payload = load_video_run(pipeline, "cached")
    assert "### FAIL" in status["value"]
    assert gif is not None and topdown is not None
    assert payload["run_id"] == "cached"
    assert not pipeline.assessment_calls  # replay never triggers providers

    status, gif, topdown, payload = load_video_run(pipeline, None)
    assert "No run selected" in status["value"]
    assert gif is None and payload is None

    broken = pipeline.store.paths("broken").root
    broken.mkdir(parents=True, exist_ok=True)
    (broken / "video_report.json").write_text("not json", encoding="utf-8")
    status, gif, topdown, payload = load_video_run(pipeline, "broken")
    assert "NO REPORT" in status["value"]
    assert payload is None


def test_build_app_wires_video_tab_components_and_lanes(tmp_path):
    from ehs_spatial.app import build_app

    demo = build_app(FakePipeline(tmp_path / "runs"))
    config = demo.get_config_file()
    components = config["components"]

    [video] = [c for c in components if c["type"] == "video"]
    assert video["props"].get("sources") == ["upload"]
    [rate] = [
        c
        for c in components
        if c["type"] == "dropdown"
        and c["props"].get("label") == "Sample rate (fps)"
    ]
    assert [choice[1] for choice in rate["props"]["choices"]] == [0.5, 1.0, 2.0]
    assert rate["props"]["value"] == 1.0
    [labels] = [
        c
        for c in components
        if c["type"] == "textbox"
        and c["props"].get("label") == "Object labels"
    ]
    assert labels["props"]["value"] == "person, forklift"
    assert any(
        c["type"] == "textbox"
        and c["props"].get("label") == "Keep-clear zone WKT (optional)"
        for c in components
    )
    [replay] = [
        c
        for c in components
        if c["type"] == "dropdown"
        and c["props"].get("label") == "Replay a processed video run"
    ]
    assert any(
        c["type"] == "markdown"
        and "live provider spend" in str(c["props"].get("value", ""))
        for c in components
    )
    buttons = {
        c["props"].get("value"): c["id"]
        for c in components
        if c["type"] == "button"
    }
    assert "Analyze video" in buttons
    assert "Refresh processed runs" in buttons
    targets = {
        tuple(target)
        for dependency in config["dependencies"]
        for target in dependency["targets"]
    }
    assert (buttons["Analyze video"], "click") in targets
    assert (buttons["Refresh processed runs"], "click") in targets
    assert (replay["id"], "input") in targets

    lanes = {
        function.api_name: function.concurrency_id
        for function in demo.fns.values()
        if function.api_name
        in {"analyze_video", "list_video_runs", "load_video_run"}
    }
    # Paid analysis shares the provider lane; replay stays local and
    # responsive while an analysis holds that lane.
    assert lanes["analyze_video"] == "ehs-provider-pipeline"
    assert lanes["list_video_runs"] == "ehs-local-ui"
    assert lanes["load_video_run"] == "ehs-local-ui"
