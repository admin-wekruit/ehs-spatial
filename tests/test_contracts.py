import importlib

import pytest
from pydantic import ValidationError


def test_capture_run_accepts_one_to_four_images_and_has_clearance_defaults():
    contracts = importlib.import_module("ehs_spatial.contracts")

    capture = contracts.CaptureRun(
        run_id="run-1",
        image_paths=["a.jpg", "b.jpg", "c.jpg", "d.jpg"],
    )

    assert capture.camera_height_m == 1.5
    assert capture.criterion.model_dump() == {
        "id": "fence_clearance",
        "subject": "closest movable obstruction",
        "object_label": "safety fence",
        "minimum_clearance_m": 0.6,
    }
    assert contracts.CaptureRun(run_id="mono", image_paths=["a"]).image_paths == ["a"]
    with pytest.raises(ValidationError):
        contracts.CaptureRun(run_id="short", image_paths=[])
    with pytest.raises(ValidationError):
        contracts.CaptureRun(
            run_id="tall",
            image_paths=["a", "b", "c", "d"],
            camera_height_m=0,
        )


def test_geometry_frame_validates_camera_and_intrinsic_matrix_shapes():
    contracts = importlib.import_module("ehs_spatial.contracts")
    camera_to_world = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]
    intrinsics = [[100, 0, 2], [0, 100, 2], [0, 0, 1]]

    frame = contracts.GeometryFrame(
        frame_id="frame-1",
        canonical_image_path="image.png",
        pts3d_path="pts3d.npy",
        conf_path="conf.npy",
        valid_mask_path="valid.npy",
        camera_to_world=camera_to_world,
        intrinsics=intrinsics,
    )

    assert frame.camera_to_world[3] == (0.0, 0.0, 0.0, 1.0)
    assert frame.intrinsics[0] == (100.0, 0.0, 2.0)
    with pytest.raises(ValidationError):
        contracts.GeometryFrame(
            frame_id="bad",
            canonical_image_path="image.png",
            pts3d_path="pts3d.npy",
            conf_path="conf.npy",
            valid_mask_path="valid.npy",
            camera_to_world=[[1, 0], [0, 1]],
            intrinsics=intrinsics,
        )


def test_observation_requires_one_mask_location_and_serializes_cleanly():
    contracts = importlib.import_module("ehs_spatial.contracts")

    observation = contracts.Observation2D(
        observation_id="obs-1",
        frame_id="frame-1",
        label="pallet",
        instance_id="0",
        mask_reference='{"size":[2,2],"counts":[0,4]}',
        score=0.92,
        bbox=[0.5, 0.5, 0.25, 0.25],
        source_prompt="pallet",
    )

    assert observation.model_dump(mode="json")["bbox"] == [0.5, 0.5, 0.25, 0.25]
    with pytest.raises(ValidationError):
        contracts.Observation2D(
            observation_id="obs-2",
            frame_id="frame-1",
            label="pallet",
            instance_id="1",
            score=0.5,
            bbox=[0, 0, 1, 1],
            source_prompt="pallet",
        )


def test_scene_map_round_trips_entities_and_evidence_backed_facts():
    contracts = importlib.import_module("ehs_spatial.contracts")
    entity = contracts.Entity3D(
        entity_id="entity-pallet-1",
        label="pallet",
        observation_ids=["obs-1"],
        centroid_xyz=[1, 2, 0.4],
        footprint_xy=[[0, 0], [1, 0], [1, 1], [0, 1]],
        height_m=0.8,
        evidence_frame_ids=["frame-1"],
    )
    fact = contracts.SpatialFact(
        fact_id="fact-1",
        predicate="clearance_to",
        subject_id="entity-pallet-1",
        object_id="fence-1",
        value=0.42,
        unit="m",
        evidence_frame_ids=["frame-1"],
    )
    scene = contracts.SceneMap(
        run_id="run-1",
        floor_plane=[0, 0, 1, 0],
        scale_source="camera_height",
        scale_factor=1.1,
        fence_polygon=[[0, 0], [2, 0], [2, 2]],
        entities=[entity],
        facts=[fact],
        warnings=["single-view edge"],
    )

    restored = contracts.SceneMap.model_validate_json(scene.model_dump_json())

    assert restored.entities[0].centroid_xyz == (1.0, 2.0, 0.4)
    assert restored.facts[0].value == 0.42
    assert restored.model_dump(mode="json")["fence_polygon"][2] == [2.0, 2.0]


def test_scene_map_represents_missing_floor_and_scale_without_inventing_values():
    contracts = importlib.import_module("ehs_spatial.contracts")

    scene = contracts.SceneMap(
        run_id="run-insufficient",
        floor_plane=None,
        scale_source=None,
        scale_factor=None,
        fence_polygon=[],
        entities=[],
        facts=[],
        warnings=["floor evidence requires at least 2 frames"],
    )

    assert scene.floor_plane is None
    assert scene.scale_source is None
    assert scene.scale_factor is None


def test_assessment_and_answer_keep_typed_grounding_references():
    contracts = importlib.import_module("ehs_spatial.contracts")
    assessment = contracts.Assessment(
        status="FAIL",
        fact_ids=["fact-1"],
        evidence_frame_ids=["frame-1"],
        approximate_distance_m=0.42,
        climb_review={
            "verdict": "uncertain",
            "rationale": "ladder top is partly occluded",
            "fact_ids": ["fact-2"],
        },
    )
    answer = contracts.GroundedAnswer(
        answer="The pallet is approximately 0.42 m from the fence.",
        fact_ids=["fact-1"],
        evidence_frame_ids=["frame-1"],
    )

    assert assessment.model_dump(mode="json")["status"] == "FAIL"
    assert assessment.climb_review.verdict == "uncertain"
    assert answer.evidence_frame_ids == ["frame-1"]
    with pytest.raises(ValidationError):
        contracts.ClimbReview(verdict="maybe", rationale="", fact_ids=[])
