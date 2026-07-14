import os
from pathlib import Path
import re
from uuid import uuid4

import pytest

from ehs_spatial.contracts import AssessmentStatus, CaptureRun, SceneMap


pytestmark = pytest.mark.skipif(
    os.getenv("EHS_LIVE_SMOKE") != "1",
    reason="set EHS_LIVE_SMOKE=1 to run the opt-in paid provider smoke test",
)


def test_default_pipeline_live_smoke(monkeypatch, tmp_path):
    required_keys = ("REPLICATE_API_TOKEN", "FAL_KEY", "GEMINI_API_KEY")
    missing_keys = [name for name in required_keys if not os.getenv(name)]
    assert not missing_keys, f"missing required provider variables: {missing_keys}"

    image_keys = tuple(f"EHS_SMOKE_IMAGE_{index}" for index in range(1, 5))
    configured_image_keys = {
        name
        for name, value in os.environ.items()
        if value and re.fullmatch(r"EHS_SMOKE_IMAGE_\d+", name)
    }
    assert configured_image_keys == set(image_keys), (
        "live smoke requires exactly EHS_SMOKE_IMAGE_1 through "
        "EHS_SMOKE_IMAGE_4"
    )
    image_paths = [Path(os.environ[name]).expanduser().resolve() for name in image_keys]
    missing_images = [
        name
        for name, path in zip(image_keys, image_paths)
        if not path.is_file()
    ]
    assert not missing_images, (
        f"live smoke image paths are not local files: {missing_images}"
    )

    monkeypatch.chdir(tmp_path)
    from ehs_spatial.pipeline import EHSAssessmentPipeline

    pipeline = EHSAssessmentPipeline()
    run_id = uuid4().hex
    assessment = pipeline.run_assessment(
        CaptureRun(run_id=run_id, image_paths=[str(path) for path in image_paths])
    )

    paths = pipeline.store.paths(run_id)
    scene = pipeline.store.load_json(paths.scene_json, SceneMap)
    assert assessment.status in AssessmentStatus
    assert assessment.climb_review is not None
    assert scene.run_id == run_id
    assert set(assessment.fact_ids) <= {fact.fact_id for fact in scene.facts}
    assert set(assessment.evidence_frame_ids) <= {
        frame_id for fact in scene.facts for frame_id in fact.evidence_frame_ids
    }
    assert all(
        path.is_file() and path.stat().st_size > 0
        for path in (
            paths.observations_json,
            paths.scene_json,
            paths.assessment_json,
            paths.topdown_png,
            paths.point_cloud_glb,
            paths.chat_jsonl,
        )
    )
