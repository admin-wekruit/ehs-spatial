import asyncio
import importlib.util
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
            self.store.save_json(
                paths.policies_json,
                [
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
                    PolicyResult(
                        policy_id="policy-review", status="NEEDS_REVIEW"
                    ),
                ],
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
    assert "warning one" in copy
    assert "warning three" in copy
    assert "warning four" not in copy
    assert "(+1 more in the structured output)" in copy
    assert result[1]["elem_classes"] == ["result-status", "status-needs-review"]
    assert result[4]["policy_results"][1]["policy_id"] == "policy-fail"
    assert ".status-needs-review" in APP_CSS
    assert ".status-needs-review h3" in APP_CSS


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
            [*upload_data, 1.5],
            state=state,
            session_hash="failed-reanalysis-regression",
        )
        previous_run = state[analyze.outputs[0]._id]
        assert previous_run is not None

        replacement_inputs = [*upload_data, 1.5]
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
    [chatbot] = [component for component in components if component["type"] == "chatbot"]
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
    assert {function.concurrency_id for function in demo.fns.values()} == {
        "ehs-provider-pipeline"
    }
    assert {function.concurrency_limit for function in demo.fns.values()} == {1}


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
    assert evidence_fn.concurrency_id == "ehs-provider-pipeline"
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
