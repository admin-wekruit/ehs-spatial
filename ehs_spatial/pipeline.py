import json
import subprocess
import warnings
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from .artifacts import ArtifactStore
from .contracts import (
    Assessment,
    CaptureRun,
    ClimbReview,
    GroundedAnswer,
    ProviderManifest,
    RunManifest,
    SceneMap,
    SpatialFact,
)
from .providers.base import ProviderError
from .providers.gemini import GeminiAdapter
from .providers.map_anything import MapAnythingAdapter
from .providers.moge import MoGeAnchorAdapter
from .geometry import FLOOR_LABEL
from .providers.sam3 import LABEL_PROMPTS, PROMPT_VOCABULARY, SAM3Adapter
from .scene import build_scene_and_assess


class EHSAssessmentPipeline:
    def __init__(
        self,
        *,
        store: ArtifactStore | None = None,
        map_anything: Any | None = None,
        sam3: Any | None = None,
        gemini: Any | None = None,
        moge: Any | None = None,
        scene_builder: Callable[..., tuple[SceneMap, Assessment]] | None = None,
    ) -> None:
        self.store = store if store is not None else ArtifactStore()
        self.map_anything = (
            map_anything if map_anything is not None else MapAnythingAdapter()
        )
        self.sam3 = sam3 if sam3 is not None else SAM3Adapter()
        self.gemini = gemini if gemini is not None else GeminiAdapter()
        self.moge = moge if moge is not None else MoGeAnchorAdapter()
        self.scene_builder = scene_builder or build_scene_and_assess

    def _resolve_scale(self, prepared: CaptureRun, frames, geometry_dir) -> dict:
        """Scale chain: auto anchor first, operator camera height as the
        explicit preference or fallback, model-native scale as last resort.
        Every path records its source, confidence and warnings — the run
        never crashes for lack of a scale, it degrades and says so."""
        if (
            prepared.scale_preference == "camera_height"
            and prepared.camera_height_m is not None
        ):
            return {
                "override": None,
                "source": "camera_height",
                "confidence": 0.9,
                "warnings": [],
            }
        anchor = None
        try:
            anchor = self.moge.anchor_scale(frames, geometry_dir)
        except Exception:
            anchor = None
        if anchor is not None:
            if anchor.confidence < 0.5 and prepared.camera_height_m is not None:
                return {
                    "override": None,
                    "source": "camera_height",
                    "confidence": 0.9,
                    "warnings": [
                        "auto scale anchor discarded (confidence "
                        f"{anchor.confidence:.2f} < 0.5); using the "
                        "operator-supplied camera height instead"
                    ],
                }
            return {
                "override": anchor.scale,
                "source": "moge_anchor",
                "confidence": anchor.confidence,
                "warnings": [],
            }
        if prepared.camera_height_m is not None:
            return {
                "override": None,
                "source": "camera_height",
                "confidence": 0.9,
                "warnings": [
                    "auto scale anchor unavailable; fell back to the "
                    "operator-supplied camera height"
                ],
            }
        return {
            "override": 1.0,
            "source": "model_native",
            "confidence": 0.2,
            "warnings": [
                "no scale anchor available; distances use the model's "
                "native scale and may be off by a large factor"
            ],
        }

    def _write_manifest(self, prepared: CaptureRun, paths) -> None:
        from .providers.gemini import GEMINI_MODEL_ID
        from .providers.map_anything import MAP_ANYTHING_MODEL_ID
        from .providers.moge import MOGE_VERSION
        from .providers.sam3 import SAM3_ENDPOINT

        try:
            code_version = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip() or "unknown"
        except Exception:
            code_version = "unknown"
        manifest = RunManifest(
            run_id=prepared.run_id,
            created_at=datetime.now(timezone.utc).isoformat(),
            operator=prepared.operator,
            capture_tier="multiview" if len(prepared.image_paths) >= 2 else "mono",
            providers=ProviderManifest(
                mapanything_model_id=MAP_ANYTHING_MODEL_ID,
                sam_endpoint=SAM3_ENDPOINT,
                gemini_model=GEMINI_MODEL_ID,
                moge_version=MOGE_VERSION,
                code_version=code_version,
            ),
        )
        self.store.save_json(paths.manifest_json, manifest)

    def run_assessment(self, capture: CaptureRun) -> Assessment:
        prepared = self.store.prepare_run(capture)
        # three-layer orientation defense: EXIF was baked at ingest
        # (deterministic); this VLM pass recovers rotation on tag-stripped
        # images (semantic — any image, any source); the floor-normal
        # gravity check in scene build is the physical tripwire behind both
        from .orientation import ensure_upright

        ensure_upright(prepared.image_paths, self.gemini)
        paths = self.store.paths(prepared.run_id)
        self._write_manifest(prepared, paths)
        frames, point_cloud_path = self.map_anything.run(
            prepared.image_paths, paths.geometry_dir
        )
        if len(frames) != len(prepared.image_paths):
            raise ProviderError(
                "replicate",
                "map_anything.response",
                f"expected {len(prepared.image_paths)} geometry frames, "
                f"got {len(frames)}",
            )
        if point_cloud_path.resolve() != paths.point_cloud_glb.resolve() or not (
            paths.point_cloud_glb.is_file()
        ):
            raise ProviderError(
                "replicate",
                "map_anything.response",
                "point cloud was not saved at the run artifact path",
            )

        # Segmentation dominates wall-clock: frames x labels independent
        # provider calls that used to run one after another (30+ serial
        # round-trips on a 3-view capture). Each (frame, label) unit keeps
        # its first-hit synonym fallback sequential — that IS the ensemble
        # semantics — but units run concurrently, bounded so fal's burst
        # billing gate is not tripped (its lock flaps under bursts; the
        # adapter's transient retry is the backstop, not the plan).
        import os
        from concurrent.futures import ThreadPoolExecutor

        def _segment_unit(frame, label):
            for prompt in LABEL_PROMPTS[label]:
                label_observations = self.sam3.segment(
                    frame.canonical_image_path,
                    prompt=prompt,
                    label=label,
                    frame_id=frame.frame_id,
                    output_dir=paths.geometry_dir / "masks" / frame.frame_id,
                )
                if label_observations:
                    return label_observations
            return []

        units = [
            (frame, label)
            for frame in frames
            for label in PROMPT_VOCABULARY
            # The floor is fitted geometrically from the full point cloud;
            # segmenting it would spend provider calls on a class SAM does
            # not need to recognise.
            if label != FLOOR_LABEL
        ]
        workers = max(1, int(os.environ.get("EHS_SAM_CONCURRENCY", "4")))
        observations = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            # map() preserves unit order, so artifacts stay deterministic
            for unit_observations in pool.map(
                lambda unit: _segment_unit(*unit), units
            ):
                observations.extend(unit_observations)
        self.store.save_json(paths.observations_json, observations)
        # Reviewer evidence is fail-soft like the plan-view renders: a broken
        # overlay must never fail an otherwise sound assessment.
        try:
            from .viewer import render_frame_overlays

            render_frame_overlays(frames, observations, paths.evidence_dir)
        except Exception as error:
            warnings.warn(f"evidence overlays failed ({error}); run continues")

        scale = self._resolve_scale(prepared, frames, paths.geometry_dir)
        scene, assessment = self.scene_builder(
            prepared.run_id,
            frames,
            observations,
            prepared.camera_height_m,
            prepared.criterion,
            topdown_path=paths.topdown_png,
            plan_view_path=paths.plan_view_png,
            semantic_ply_path=paths.semantic_ply,
            cloud_views_paths=(
                paths.cloud_perspective_png,
                paths.cloud_topdown_png,
            ),
            scale_factor_override=scale["override"],
            scale_source=scale["source"],
            scale_confidence=scale["confidence"],
            scale_warnings=scale["warnings"],
        )
        self.store.save_json(paths.scene_json, scene)
        if prepared.policies:
            from .policy import evaluate_policies

            results = evaluate_policies(
                prepared.policies, scene, capture_frame_count=len(frames)
            )
            # Specs travel with results so the card can cite the compiled
            # predicate and the prose it came from without re-reading input.
            paths.policies_json.write_text(
                json.dumps(
                    {
                        "specs": [
                            spec.model_dump(mode="json")
                            for spec in prepared.policies
                        ],
                        "results": [
                            result.model_dump(mode="json") for result in results
                        ],
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        climb_review, interaction_id = self.gemini.review_climb(
            scene, assessment, prepared.criterion, frames
        )
        climb_review = ClimbReview.model_validate(climb_review)
        if not isinstance(interaction_id, str) or not interaction_id:
            raise ProviderError(
                "gemini", "climb.cursor", "response is missing an interaction id"
            )
        final_assessment = Assessment.model_validate(
            {
                **assessment.model_dump(mode="python"),
                "climb_review": climb_review,
            }
        )
        self.store.save_json(paths.assessment_json, final_assessment)
        self.store.append_chat(
            prepared.run_id,
            {"type": "gemini_cursor", "interaction_id": interaction_id},
        )
        try:
            from .viewer import build_viewer_html

            build_viewer_html(
                paths.root,
                frames=frames,
                observations=observations,
                camera_height_m=prepared.camera_height_m,
                scale_factor_override=scale["override"],
                out_path=paths.viewer_html,
            )
        except Exception as error:
            warnings.warn(f"3D viewer build failed ({error}); run continues")
        return final_assessment

    def _policy_facts(self, paths) -> list[SpatialFact]:
        """PolicyResult facts from policies.json, so a grounded answer can
        cite the measurements behind a policy verdict. Fail-soft like every
        other policies.json reader: missing or malformed means no policy
        facts, never a broken chat."""
        try:
            payload = json.loads(paths.policies_json.read_text(encoding="utf-8"))
            return [
                SpatialFact.model_validate(fact)
                for result in payload["results"]
                for fact in result.get("facts", [])
            ]
        except Exception:
            return []

    def answer_question(self, run_id: str, question: str) -> GroundedAnswer:
        paths = self.store.paths(run_id)
        scene = self.store.load_json(paths.scene_json, SceneMap)
        interaction_id = self.store.latest_chat_cursor(run_id)
        if interaction_id is None:
            raise ProviderError(
                "gemini", "chat.cursor", "run has no stored interaction id"
            )
        policy_facts = self._policy_facts(paths)
        answer, next_interaction_id = self.gemini.answer(
            question,
            scene,
            previous_interaction_id=interaction_id,
            # Only added when the run has policy facts: adapters without the
            # parameter (and runs without policies) keep the exact legacy call.
            **({"policy_facts": policy_facts} if policy_facts else {}),
        )
        answer = GroundedAnswer.model_validate(answer)
        if not isinstance(next_interaction_id, str) or not next_interaction_id:
            raise ProviderError(
                "gemini", "chat.cursor", "response is missing an interaction id"
            )
        self.store.append_chat(
            run_id,
            {
                "type": "chat_turn",
                "user": {"role": "user", "content": question},
                "assistant": {
                    "role": "assistant",
                    "content": answer.answer,
                    "fact_ids": answer.fact_ids,
                    "evidence_frame_ids": answer.evidence_frame_ids,
                },
                "interaction_id": next_interaction_id,
            },
        )
        return answer


__all__ = ["EHSAssessmentPipeline"]
