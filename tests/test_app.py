import importlib.util
from pathlib import Path
import re

import gradio as gr
from PIL import Image
import pytest

from ehs_spatial.artifacts import ArtifactStore
from ehs_spatial.contracts import (
    Assessment,
    ClimbReview,
    GroundedAnswer,
    SceneMap,
    SpatialFact,
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


@pytest.mark.parametrize("missing_value", [None, ""], ids=["none", "empty"])
@pytest.mark.parametrize("missing_index", range(4))
def test_analysis_requires_all_four_images_before_provider_call(
    tmp_path, missing_index, missing_value
):
    from ehs_spatial.app import analyze_run

    pipeline = FakePipeline(tmp_path / "runs")
    images = _images(tmp_path)
    images[missing_index] = missing_value

    with pytest.raises(gr.Error, match="Upload all four"):
        analyze_run(pipeline, *images, 1.5)

    assert pipeline.assessment_calls == []


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


def test_analysis_surfaces_raw_provider_error_without_fallback(tmp_path):
    from ehs_spatial.app import analyze_run

    error = ProviderError("replicate", "map_anything.run", "service unavailable")
    pipeline = FakePipeline(tmp_path / "runs", run_error=error)

    with pytest.raises(gr.Error, match=str(error)):
        analyze_run(pipeline, *_images(tmp_path), 1.5)


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


def test_chat_surfaces_raw_provider_error_without_changing_history(tmp_path):
    from ehs_spatial.app import answer_run_question

    error = ProviderError("gemini", "chat.create", "service unavailable")
    pipeline = FakePipeline(tmp_path / "runs", question_error=error)
    history = [{"role": "user", "content": "Earlier question"}]

    with pytest.raises(gr.Error, match=str(error)):
        answer_run_question(pipeline, "What is the clearance?", history, "run-1")

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
        "View 1 — workcell front",
        "View 2 — workcell right",
        "View 3 — workcell rear",
        "View 4 — workcell left",
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

    assert demo.launches == [
        {"css": module.APP_CSS, "footer_links": [], "show_error": True}
    ]
