from collections.abc import Callable
from typing import Any

from .artifacts import ArtifactStore
from .contracts import Assessment, CaptureRun, GroundedAnswer, SceneMap
from .providers.base import ProviderError
from .providers.gemini import GeminiAdapter
from .providers.map_anything import MapAnythingAdapter
from .providers.sam3 import PROMPT_VOCABULARY, SAM3Adapter
from .scene import build_scene_and_assess


class EHSAssessmentPipeline:
    def __init__(
        self,
        *,
        store: ArtifactStore | None = None,
        map_anything: Any | None = None,
        sam3: Any | None = None,
        gemini: Any | None = None,
        scene_builder: Callable[..., tuple[SceneMap, Assessment]] | None = None,
    ) -> None:
        self.store = store if store is not None else ArtifactStore()
        self.map_anything = (
            map_anything if map_anything is not None else MapAnythingAdapter()
        )
        self.sam3 = sam3 if sam3 is not None else SAM3Adapter()
        self.gemini = gemini if gemini is not None else GeminiAdapter()
        self.scene_builder = scene_builder or build_scene_and_assess

    def run_assessment(self, capture: CaptureRun) -> Assessment:
        prepared = self.store.prepare_run(capture)
        paths = self.store.paths(prepared.run_id)
        frames, point_cloud_path = self.map_anything.run(
            prepared.image_paths, paths.geometry_dir
        )
        if len(frames) != 4:
            raise ProviderError(
                "replicate",
                "map_anything.response",
                f"expected four geometry frames, got {len(frames)}",
            )
        if point_cloud_path.resolve() != paths.point_cloud_glb.resolve() or not (
            paths.point_cloud_glb.is_file()
        ):
            raise ProviderError(
                "replicate",
                "map_anything.response",
                "point cloud was not saved at the run artifact path",
            )

        observations = []
        for frame in frames:
            for prompt in PROMPT_VOCABULARY:
                observations.extend(
                    self.sam3.segment(
                        frame.canonical_image_path,
                        prompt=prompt,
                        frame_id=frame.frame_id,
                        output_dir=paths.geometry_dir / "masks" / frame.frame_id,
                    )
                )
        self.store.save_json(paths.observations_json, observations)

        scene, assessment = self.scene_builder(
            prepared.run_id,
            frames,
            observations,
            prepared.camera_height_m,
            prepared.criterion,
            topdown_path=paths.topdown_png,
        )
        self.store.save_json(paths.scene_json, scene)
        climb_review, interaction_id = self.gemini.review_climb(
            scene, assessment, prepared.criterion, frames
        )
        if not isinstance(interaction_id, str) or not interaction_id:
            raise ProviderError(
                "gemini", "climb.cursor", "response is missing an interaction id"
            )
        final_assessment = assessment.model_copy(
            update={"climb_review": climb_review}
        )
        self.store.save_json(paths.assessment_json, final_assessment)
        self.store.append_chat(
            prepared.run_id,
            {"type": "gemini_cursor", "interaction_id": interaction_id},
        )
        return final_assessment

    def answer_question(self, run_id: str, question: str) -> GroundedAnswer:
        paths = self.store.paths(run_id)
        scene = self.store.load_json(paths.scene_json, SceneMap)
        interaction_id = self.store.latest_chat_cursor(run_id)
        if interaction_id is None:
            raise ProviderError(
                "gemini", "chat.cursor", "run has no stored interaction id"
            )
        answer, next_interaction_id = self.gemini.answer(
            question,
            scene,
            previous_interaction_id=interaction_id,
        )
        if not isinstance(next_interaction_id, str) or not next_interaction_id:
            raise ProviderError(
                "gemini", "chat.cursor", "response is missing an interaction id"
            )
        self.store.append_chat(
            run_id,
            {"type": "message", "role": "user", "content": question},
        )
        self.store.append_chat(
            run_id,
            {
                "type": "message",
                "role": "assistant",
                "content": answer.answer,
                "fact_ids": answer.fact_ids,
                "evidence_frame_ids": answer.evidence_frame_ids,
            },
        )
        self.store.append_chat(
            run_id,
            {"type": "gemini_cursor", "interaction_id": next_interaction_id},
        )
        return answer


__all__ = ["EHSAssessmentPipeline"]
