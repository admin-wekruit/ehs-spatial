import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
from pydantic import ValidationError

from ehs_spatial.artifacts import ArtifactStore
from ehs_spatial.contracts import (
    Assessment,
    CaptureRun,
    ClimbReview,
    GeometryFrame,
    GroundedAnswer,
    Observation2D,
    SceneMap,
    SpatialFact,
)
from ehs_spatial.providers.base import ProviderError
from ehs_spatial.providers.sam3 import PROMPT_VOCABULARY


IDENTITY_4 = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 1.5),
    (0.0, 0.0, 0.0, 1.0),
)
INTRINSICS = ((100.0, 0.0, 1.0), (0.0, 100.0, 1.0), (0.0, 0.0, 1.0))


class FakeMapAnything:
    def __init__(self, *, error=None):
        self.calls = []
        self.error = error

    def run(self, image_paths, geometry_dir):
        self.calls.append((list(image_paths), Path(geometry_dir)))
        if self.error:
            raise self.error
        geometry_dir = Path(geometry_dir)
        geometry_dir.mkdir(parents=True, exist_ok=True)
        glb_path = geometry_dir / "point_cloud.glb"
        glb_path.write_bytes(b"glb")
        frames = []
        for index in range(1, len(image_paths) + 1):
            canonical = geometry_dir / f"canonical-{index}.png"
            Image.new("RGB", (2, 2), (index, index, index)).save(canonical)
            frames.append(
                GeometryFrame(
                    frame_id=f"frame-{index}",
                    canonical_image_path=str(canonical),
                    pts3d_path=str(geometry_dir / f"points-{index}.npy"),
                    conf_path=str(geometry_dir / f"conf-{index}.npy"),
                    valid_mask_path=str(geometry_dir / f"valid-{index}.npy"),
                    camera_to_world=IDENTITY_4,
                    intrinsics=INTRINSICS,
                )
            )
        return frames, glb_path


class FakeSAM3:
    def __init__(self):
        self.calls = []

    def segment(self, image_path, *, prompt, frame_id, output_dir, label=None):
        label = prompt if label is None else label
        self.calls.append((str(image_path), prompt, frame_id, Path(output_dir)))
        ordinal = len(self.calls)
        return [
            Observation2D(
                observation_id=f"obs-{ordinal}",
                frame_id=frame_id,
                label=label,
                instance_id=str(ordinal),
                mask_reference="fake-rle",
                score=0.9,
                bbox=[0, 0, 1, 1],
                source_prompt=prompt,
            )
        ]


class FakeGemini:
    def __init__(self, *, answer_error=None):
        self.review_calls = []
        self.answer_calls = []
        self.answer_error = answer_error

    def review_climb(self, scene, assessment, criterion, frames):
        self.review_calls.append((scene, assessment, criterion, list(frames)))
        return (
            ClimbReview(
                verdict="uncertain",
                rationale="Use the measured facts for human review.",
                fact_ids=["fact-clearance"],
            ),
            "interaction-initial",
        )

    def answer(self, question, scene, *, previous_interaction_id):
        self.answer_calls.append((question, scene, previous_interaction_id))
        if self.answer_error:
            raise self.answer_error
        return (
            GroundedAnswer(
                answer="The clearance is approximately 0.5 m.",
                fact_ids=["fact-clearance"],
                evidence_frame_ids=["frame-1"],
            ),
            "interaction-next",
        )


class FakeMoGe:
    def __init__(self, anchor=None):
        self.anchor = anchor
        self.calls = 0

    def anchor_scale(self, frames, geometry_dir):
        self.calls += 1
        return self.anchor


class FakeSceneBuilder:
    def __init__(self):
        self.calls = []

    def __call__(
        self,
        run_id,
        frames,
        observations,
        camera_height_m,
        criterion,
        topdown_path=None,
        plan_view_path=None,
        semantic_ply_path=None,
        cloud_views_paths=None,
        scale_factor_override=None,
        scale_source="camera_height",
        scale_confidence=None,
        scale_warnings=None,
    ):
        self.calls.append(
            (
                run_id,
                list(frames),
                list(observations),
                camera_height_m,
                criterion,
                Path(topdown_path),
            )
        )
        Path(topdown_path).write_bytes(b"png")
        fact = SpatialFact(
            fact_id="fact-clearance",
            predicate="minimum_boundary_clearance",
            subject_id="pallet-1",
            object_id="fence-1",
            value=0.5,
            unit="m",
            evidence_frame_ids=["frame-1"],
        )
        scene = SceneMap(
            run_id=run_id,
            floor_plane=[0, 0, 1, 0],
            scale_source="camera_height",
            scale_factor=1,
            fence_polygon=[[0, 0], [2, 0], [2, 2]],
            entities=[],
            facts=[fact],
            warnings=[],
        )
        assessment = Assessment(
            status="FAIL",
            fact_ids=[fact.fact_id],
            evidence_frame_ids=["frame-1"],
            approximate_distance_m=0.5,
        )
        return scene, assessment


def _capture(tmp_path):
    images = []
    for index in range(1, 5):
        path = tmp_path / f"upload-{index}.png"
        Image.new("RGB", (2, 2), (index, index, index)).save(path)
        images.append(str(path))
    return CaptureRun(run_id="run-1", image_paths=images)


def _pipeline(tmp_path, *, map_adapter=None, gemini=None):
    from ehs_spatial.pipeline import EHSAssessmentPipeline

    store = ArtifactStore(tmp_path / "runs")
    map_adapter = map_adapter or FakeMapAnything()
    sam = FakeSAM3()
    gemini = gemini or FakeGemini()
    scene_builder = FakeSceneBuilder()
    pipeline = EHSAssessmentPipeline(
        store=store,
        map_anything=map_adapter,
        sam3=sam,
        gemini=gemini,
        moge=FakeMoGe(),
        scene_builder=scene_builder,
    )
    return pipeline, store, map_adapter, sam, gemini, scene_builder


def test_run_assessment_executes_one_map_call_and_stable_44_sam_calls(tmp_path):
    pipeline, store, map_adapter, sam, gemini, scene_builder = _pipeline(tmp_path)

    result = pipeline.run_assessment(_capture(tmp_path))

    assert len(map_adapter.calls) == 1
    assert [Path(path).name for path in map_adapter.calls[0][0]] == [
        "image_01.png",
        "image_02.png",
        "image_03.png",
        "image_04.png",
    ]
    # "factory floor" is fitted geometrically, never segmented: 11 labels x 4 frames.
    assert [(call[2], call[1]) for call in sam.calls] == [
        (f"frame-{frame}", prompt)
        for frame in range(1, 5)
        for prompt in PROMPT_VOCABULARY
        if prompt != "factory floor"
    ]
    assert len(scene_builder.calls) == 1
    scene_call = scene_builder.calls[0]
    assert len(scene_call[1]) == 4
    assert len(scene_call[2]) == 44
    assert scene_call[3] == 1.5
    assert result.model_dump(exclude={"climb_review"}) == Assessment(
        status="FAIL",
        fact_ids=["fact-clearance"],
        evidence_frame_ids=["frame-1"],
        approximate_distance_m=0.5,
    ).model_dump(exclude={"climb_review"})
    assert result.climb_review == ClimbReview(
        verdict="uncertain",
        rationale="Use the measured facts for human review.",
        fact_ids=["fact-clearance"],
    )
    [review_call] = gemini.review_calls
    assert review_call[1].climb_review is None

    paths = store.paths("run-1")
    assert all(
        path.is_file()
        for path in [
            paths.observations_json,
            paths.scene_json,
            paths.assessment_json,
            paths.topdown_png,
            paths.point_cloud_glb,
            paths.chat_jsonl,
        ]
    )
    assert len(json.loads(paths.observations_json.read_text(encoding="utf-8"))) == 44
    assert json.loads(paths.assessment_json.read_text(encoding="utf-8"))[
        "status"
    ] == "FAIL"
    assert store.latest_chat_cursor("run-1") == "interaction-initial"


def test_answer_question_recovers_cursor_in_a_new_pipeline_and_appends_after_validation(
    tmp_path,
):
    pipeline, store, *_ = _pipeline(tmp_path)
    pipeline.run_assessment(_capture(tmp_path))
    chat_before = store.paths("run-1").chat_jsonl.read_text(encoding="utf-8")
    next_gemini = FakeGemini()
    from ehs_spatial.pipeline import EHSAssessmentPipeline

    fresh_pipeline = EHSAssessmentPipeline(store=store, gemini=next_gemini)
    append_calls = []
    original_append = store.append_chat

    def record_single_append(run_id, entry):
        append_calls.append((run_id, entry))
        original_append(run_id, entry)

    store.append_chat = record_single_append

    answer = fresh_pipeline.answer_question("run-1", "How far is the pallet?")

    assert answer.fact_ids == ["fact-clearance"]
    assert next_gemini.answer_calls[0][2] == "interaction-initial"
    assert store.latest_chat_cursor("run-1") == "interaction-next"
    appended = store.paths("run-1").chat_jsonl.read_text(encoding="utf-8")[
        len(chat_before) :
    ]
    entries = [json.loads(line) for line in appended.splitlines()]
    assert entries == [
        {
            "type": "chat_turn",
            "user": {
                "role": "user",
                "content": "How far is the pallet?",
            },
            "assistant": {
                "role": "assistant",
                "content": "The clearance is approximately 0.5 m.",
                "fact_ids": ["fact-clearance"],
                "evidence_frame_ids": ["frame-1"],
            },
            "interaction_id": "interaction-next",
        }
    ]
    assert len(append_calls) == 1
    assert append_calls[0][1] == entries[0]


def _policy_fact_payload() -> dict:
    return {
        "fact_id": "fact-p06-portable-ladder-tilt-limit-01",
        "predicate": "max_tilt",
        "subject_id": "entity-step-ladder-02",
        "object_id": "entity-step-ladder-02",
        "value": 21.5716,
        "unit": "deg",
        "evidence_frame_ids": ["frame-1"],
    }


def test_answer_question_hands_policy_facts_to_the_grounded_answer(tmp_path):
    """Policy measurements become citable context for Q&A: the facts stored
    in policies.json ride into gemini.answer as SpatialFacts."""

    class PolicyAwareFakeGemini(FakeGemini):
        def answer(
            self, question, scene, *, previous_interaction_id, policy_facts=None
        ):
            self.policy_facts = policy_facts
            return super().answer(
                question, scene, previous_interaction_id=previous_interaction_id
            )

    pipeline, store, *_ = _pipeline(tmp_path)
    pipeline.run_assessment(_capture(tmp_path))
    store.paths("run-1").policies_json.write_text(
        json.dumps(
            {
                "specs": [],
                "results": [
                    {
                        "policy_id": "p06-portable-ladder-tilt-limit",
                        "status": "FAIL",
                        "facts": [_policy_fact_payload()],
                    },
                    {"policy_id": "p05-pallet-max-height", "status": "PASS"},
                ],
            }
        ),
        encoding="utf-8",
    )
    gemini = PolicyAwareFakeGemini()
    from ehs_spatial.pipeline import EHSAssessmentPipeline

    fresh = EHSAssessmentPipeline(store=store, gemini=gemini)

    fresh.answer_question("run-1", "Why did the tilt policy fail?")

    assert gemini.policy_facts == [
        SpatialFact.model_validate(_policy_fact_payload())
    ]


def test_answer_question_without_usable_policy_facts_keeps_legacy_call(tmp_path):
    """A run with no policies.json — or a malformed one — keeps the exact
    legacy gemini.answer call, so adapters without the policy_facts
    parameter (like this fake) still work."""
    pipeline, store, *_ = _pipeline(tmp_path)
    pipeline.run_assessment(_capture(tmp_path))
    store.paths("run-1").policies_json.write_text("{", encoding="utf-8")
    from ehs_spatial.pipeline import EHSAssessmentPipeline

    fresh = EHSAssessmentPipeline(store=store, gemini=FakeGemini())

    answer = fresh.answer_question("run-1", "How far is the pallet?")

    assert answer.fact_ids == ["fact-clearance"]


def test_answer_question_does_not_append_chat_when_provider_validation_fails(tmp_path):
    pipeline, store, *_ = _pipeline(tmp_path)
    pipeline.run_assessment(_capture(tmp_path))
    before = store.paths("run-1").chat_jsonl.read_text(encoding="utf-8")
    error = ProviderError("gemini", "chat.grounding", "unknown fact id")
    from ehs_spatial.pipeline import EHSAssessmentPipeline

    failing = EHSAssessmentPipeline(
        store=store, gemini=FakeGemini(answer_error=error)
    )

    with pytest.raises(ProviderError) as caught:
        failing.answer_question("run-1", "Invent an answer")

    assert caught.value is error
    assert store.paths("run-1").chat_jsonl.read_text(encoding="utf-8") == before


def test_run_assessment_propagates_provider_error_without_fallback(tmp_path):
    error = ProviderError("replicate", "map_anything.run", "service unavailable")
    pipeline, store, *_ = _pipeline(
        tmp_path, map_adapter=FakeMapAnything(error=error)
    )

    with pytest.raises(ProviderError) as caught:
        pipeline.run_assessment(_capture(tmp_path))

    assert caught.value is error
    paths = store.paths("run-1")
    assert not paths.observations_json.exists()
    assert not paths.assessment_json.exists()


def test_run_assessment_rejects_malformed_provider_climb_review(tmp_path):
    class MalformedClimbGemini(FakeGemini):
        def review_climb(self, scene, assessment, criterion, frames):
            return (
                {"verdict": "maybe", "rationale": "invalid", "fact_ids": []},
                "interaction-initial",
            )

    pipeline, store, *_ = _pipeline(tmp_path, gemini=MalformedClimbGemini())

    with pytest.raises(ValidationError):
        pipeline.run_assessment(_capture(tmp_path))

    assert not store.paths("run-1").assessment_json.exists()
    assert not store.paths("run-1").chat_jsonl.exists()


def test_answer_question_rejects_malformed_provider_answer_before_chat_write(tmp_path):
    pipeline, store, *_ = _pipeline(tmp_path)
    pipeline.run_assessment(_capture(tmp_path))
    chat_before = store.paths("run-1").chat_jsonl.read_text(encoding="utf-8")

    class MalformedAnswerGemini(FakeGemini):
        def answer(self, question, scene, *, previous_interaction_id):
            return (
                {"answer": "unvalidated", "fact_ids": ["fact-clearance"]},
                "interaction-next",
            )

    from ehs_spatial.pipeline import EHSAssessmentPipeline

    failing = EHSAssessmentPipeline(store=store, gemini=MalformedAnswerGemini())

    with pytest.raises(ValidationError):
        failing.answer_question("run-1", "How far?")

    assert store.paths("run-1").chat_jsonl.read_text(encoding="utf-8") == chat_before


def test_pipeline_falls_back_through_synonym_prompts_until_hit(tmp_path):
    from ehs_spatial.pipeline import EHSAssessmentPipeline
    from ehs_spatial.providers.sam3 import LABEL_PROMPTS

    class SynonymFakeSAM3(FakeSAM3):
        def segment(self, image_path, *, prompt, frame_id, output_dir, label=None):
            if label == "safety fence" and prompt != "barrier":
                self.calls.append((str(image_path), prompt, frame_id, Path(output_dir)))
                return []
            return super().segment(
                image_path,
                prompt=prompt,
                frame_id=frame_id,
                output_dir=output_dir,
                label=label,
            )

    store = ArtifactStore(tmp_path / "runs")
    sam = SynonymFakeSAM3()
    scene_builder = FakeSceneBuilder()
    pipeline = EHSAssessmentPipeline(
        store=store,
        map_anything=FakeMapAnything(),
        sam3=sam,
        gemini=FakeGemini(),
        scene_builder=scene_builder,
    )

    pipeline.run_assessment(_capture(tmp_path))

    fence_prompts = [
        call[1] for call in sam.calls if call[1] in LABEL_PROMPTS["safety fence"]
    ]
    # Per frame: canonical fails, "fence" fails, "barrier" hits, "guardrail" skipped.
    assert fence_prompts[:3] == ["safety fence", "fence", "barrier"]
    assert "guardrail" not in fence_prompts

    observations = scene_builder.calls[0][2]
    fence_observations = [
        observation for observation in observations
        if observation.label == "safety fence"
    ]
    assert len(fence_observations) == 4
    assert all(
        observation.source_prompt == "barrier" for observation in fence_observations
    )


def test_run_assessment_accepts_a_single_image_capture(tmp_path):
    pipeline, store, map_adapter, sam, gemini, scene_builder = _pipeline(tmp_path)
    path = tmp_path / "upload-1.png"
    Image.new("RGB", (2, 2), (1, 1, 1)).save(path)
    capture = CaptureRun(run_id="run-1", image_paths=[str(path)])

    pipeline.run_assessment(capture)

    assert len(map_adapter.calls) == 1
    assert len(map_adapter.calls[0][0]) == 1
    # 11 segmented labels x 1 frame (floor is fitted geometrically).
    assert len(sam.calls) == 11
    assert len(scene_builder.calls[0][1]) == 1


def test_capture_run_rejects_zero_and_five_images(tmp_path):
    with pytest.raises(ValidationError):
        CaptureRun(run_id="run-1", image_paths=[])
    with pytest.raises(ValidationError):
        CaptureRun(run_id="run-1", image_paths=["a.png"] * 5)


def test_scale_chain_prefers_auto_anchor_and_records_source(tmp_path):
    from ehs_spatial.pipeline import EHSAssessmentPipeline
    from ehs_spatial.providers.moge import ScaleAnchor

    store = ArtifactStore(tmp_path / "runs")
    scene_builder = FakeSceneBuilder()
    pipeline = EHSAssessmentPipeline(
        store=store,
        map_anything=FakeMapAnything(),
        sam3=FakeSAM3(),
        gemini=FakeGemini(),
        moge=FakeMoGe(anchor=ScaleAnchor(2.5, 0.92, [2.5])),
        scene_builder=scene_builder,
    )

    pipeline.run_assessment(_capture(tmp_path))

    scale = pipeline._resolve_scale(
        _capture(tmp_path), [], tmp_path
    )
    assert scale["source"] == "moge_anchor"
    assert scale["override"] == 2.5
    assert scale["confidence"] == 0.92


def test_scale_chain_falls_back_to_camera_height_with_a_warning(tmp_path):
    pipeline, *_ = _pipeline(tmp_path)  # FakeMoGe returns None

    scale = pipeline._resolve_scale(_capture(tmp_path), [], tmp_path)

    assert scale["source"] == "camera_height"
    assert scale["override"] is None
    assert any("fell back" in w for w in scale["warnings"])


def test_scale_chain_degrades_to_model_native_when_no_source_exists(tmp_path):
    pipeline, *_ = _pipeline(tmp_path)
    capture = _capture(tmp_path).model_copy(update={"camera_height_m": None})

    scale = pipeline._resolve_scale(capture, [], tmp_path)

    assert scale["source"] == "model_native"
    assert scale["override"] == 1.0
    assert scale["confidence"] == 0.2
    assert any("native scale" in w for w in scale["warnings"])


def test_scale_chain_discards_low_confidence_anchor_for_measured_height(tmp_path):
    from ehs_spatial.pipeline import EHSAssessmentPipeline
    from ehs_spatial.providers.moge import ScaleAnchor

    pipeline = EHSAssessmentPipeline(
        store=ArtifactStore(tmp_path / "runs"),
        map_anything=FakeMapAnything(),
        sam3=FakeSAM3(),
        gemini=FakeGemini(),
        moge=FakeMoGe(anchor=ScaleAnchor(2.5, 0.3, [2.5])),
        scene_builder=FakeSceneBuilder(),
    )

    scale = pipeline._resolve_scale(_capture(tmp_path), [], tmp_path)

    # A shaky anchor must not silently beat the operator's tape measure.
    assert scale["source"] == "camera_height"
    assert scale["override"] is None
    assert any("discarded" in w for w in scale["warnings"])

    # Without a measured height the shaky anchor is still the best gauge.
    no_height = _capture(tmp_path).model_copy(update={"camera_height_m": None})
    scale = pipeline._resolve_scale(no_height, [], tmp_path)
    assert scale["source"] == "moge_anchor"


def test_scale_chain_honours_explicit_camera_height_preference(tmp_path):
    from ehs_spatial.pipeline import EHSAssessmentPipeline
    from ehs_spatial.providers.moge import ScaleAnchor

    pipeline = EHSAssessmentPipeline(
        store=ArtifactStore(tmp_path / "runs"),
        map_anything=FakeMapAnything(),
        sam3=FakeSAM3(),
        gemini=FakeGemini(),
        moge=FakeMoGe(anchor=ScaleAnchor(2.5, 0.92, [2.5])),
        scene_builder=FakeSceneBuilder(),
    )
    capture = _capture(tmp_path).model_copy(
        update={"scale_preference": "camera_height"}
    )

    scale = pipeline._resolve_scale(capture, [], tmp_path)

    # The operator explicitly chose their tape measure: the anchor is not
    # even consulted.
    assert scale["source"] == "camera_height"
    assert pipeline.moge.calls == 0


def test_run_assessment_writes_a_provenance_manifest(tmp_path):
    from ehs_spatial.contracts import RunManifest

    pipeline, store, *_ = _pipeline(tmp_path)

    pipeline.run_assessment(_capture(tmp_path))

    manifest = store.load_json(store.paths("run-1").manifest_json, RunManifest)
    assert manifest.run_id == "run-1"
    assert manifest.capture_tier == "multiview"
    assert manifest.operator == "unknown"
    assert manifest.created_at.endswith("+00:00")
    assert manifest.providers.mapanything_model_id.startswith("vufinder/")
    assert manifest.providers.moge_version.startswith("jasonod888/")


def test_run_assessment_emits_reviewer_evidence_artifacts(tmp_path):
    """Every run gets its visual evidence from the standard pipeline: mask
    overlays per frame plus the interactive viewer.html."""
    from test_viewer import write_box_mask, write_synthetic_frame

    from ehs_spatial.pipeline import EHSAssessmentPipeline

    class EvidenceFakeMapAnything:
        def run(self, image_paths, geometry_dir):
            geometry_dir = Path(geometry_dir)
            geometry_dir.mkdir(parents=True, exist_ok=True)
            glb_path = geometry_dir / "point_cloud.glb"
            glb_path.write_bytes(b"glb")
            frames = [
                write_synthetic_frame(
                    geometry_dir / "frames" / f"frame_{index:04d}",
                    f"frame_{index:04d}",
                )
                for index in range(1, len(image_paths) + 1)
            ]
            return frames, glb_path

    class MaskWritingFakeSAM3(FakeSAM3):
        def segment(self, image_path, *, prompt, frame_id, output_dir, label=None):
            if label != "pallet":
                return []
            mask_path = Path(output_dir) / f"{frame_id}_pallet.png"
            write_box_mask(mask_path)
            [observation] = super().segment(
                image_path,
                prompt=prompt,
                frame_id=frame_id,
                output_dir=output_dir,
                label=label,
            )
            return [
                observation.model_copy(
                    update={"mask_path": str(mask_path), "mask_reference": None}
                )
            ]

    store = ArtifactStore(tmp_path / "runs")
    pipeline = EHSAssessmentPipeline(
        store=store,
        map_anything=EvidenceFakeMapAnything(),
        sam3=MaskWritingFakeSAM3(),
        gemini=FakeGemini(),
        moge=FakeMoGe(),
        scene_builder=FakeSceneBuilder(),
    )

    pipeline.run_assessment(_capture(tmp_path))

    paths = store.paths("run-1")
    overlays = sorted(path.name for path in paths.evidence_dir.glob("*_overlay.png"))
    assert overlays == [f"frame_{index:04d}_overlay.png" for index in range(1, 5)]
    with Image.open(paths.evidence_dir / "frame_0001_overlay.png") as overlay_image:
        overlay = np.asarray(overlay_image.convert("RGB"))
    with Image.open(paths.geometry_dir / "frames" / "frame_0001" / "canonical.png") as base_image:
        base = np.asarray(base_image.convert("RGB"))
    assert not np.array_equal(overlay, base)
    assert paths.viewer_html.is_file()
    assert '"label":"pallet"' in paths.viewer_html.read_text(encoding="utf-8")


def test_evidence_failures_warn_but_never_fail_the_run(tmp_path, monkeypatch):
    """Fail-soft: a broken overlay renderer and an unbuildable viewer both
    reduce to warnings; the assessment itself is untouched."""
    import ehs_spatial.viewer as viewer_module

    pipeline, store, *_ = _pipeline(tmp_path)

    def broken_overlays(*args, **kwargs):
        raise RuntimeError("overlay renderer exploded")

    monkeypatch.setattr(viewer_module, "render_frame_overlays", broken_overlays)

    with pytest.warns(UserWarning) as caught:
        result = pipeline.run_assessment(_capture(tmp_path))

    messages = [str(warning.message) for warning in caught]
    assert any("evidence overlays failed" in message for message in messages)
    # The standard fakes have no reconstructed geometry on disk, so the
    # viewer build also degrades to a warning here.
    assert any("3D viewer build failed" in message for message in messages)
    assert result.status.value == "FAIL"
    paths = store.paths("run-1")
    assert paths.assessment_json.is_file()
    assert not paths.evidence_dir.exists()
    assert not paths.viewer_html.exists()


def test_mono_capture_manifest_records_mono_tier(tmp_path):
    from ehs_spatial.contracts import RunManifest

    pipeline, store, *_ = _pipeline(tmp_path)
    path = tmp_path / "upload-1.png"
    Image.new("RGB", (2, 2), (1, 1, 1)).save(path)

    pipeline.run_assessment(
        CaptureRun(run_id="run-1", image_paths=[str(path)], operator="adam")
    )

    manifest = store.load_json(store.paths("run-1").manifest_json, RunManifest)
    assert manifest.capture_tier == "mono"
    assert manifest.operator == "adam"
