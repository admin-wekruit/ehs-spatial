import importlib
import json
from pathlib import Path

import pytest

from ehs_spatial.contracts import CaptureRun, Observation2D, SceneMap


def test_prepare_run_creates_layout_and_copies_four_images_to_stable_names(tmp_path):
    artifacts = importlib.import_module("ehs_spatial.artifacts")
    source_paths = []
    for index in range(4):
        source = tmp_path / f"source-{index}.jpg"
        source.write_bytes(f"image-{index}".encode())
        source_paths.append(str(source))
    store = artifacts.ArtifactStore(tmp_path / "runs")

    prepared = store.prepare_run(CaptureRun(run_id="run-1", image_paths=source_paths))

    paths = store.paths("run-1")
    assert paths.input_dir.is_dir()
    assert paths.geometry_dir.is_dir()
    assert [path.name for path in map(Path, prepared.image_paths)] == [
        "image_01.jpg",
        "image_02.jpg",
        "image_03.jpg",
        "image_04.jpg",
    ]
    assert [Path(path).read_bytes() for path in prepared.image_paths] == [
        b"image-0",
        b"image-1",
        b"image-2",
        b"image-3",
    ]


def test_artifact_paths_and_pydantic_json_round_trip(tmp_path):
    artifacts = importlib.import_module("ehs_spatial.artifacts")
    store = artifacts.ArtifactStore(tmp_path / "runs")
    paths = store.paths("run-1")
    paths.geometry_dir.mkdir(parents=True)
    scene = SceneMap(
        run_id="run-1",
        floor_plane=[0, 0, 1, 0],
        scale_source="camera_height",
        scale_factor=1,
        fence_polygon=[],
        entities=[],
        facts=[],
        warnings=[],
    )

    store.save_json(paths.scene_json, scene)
    restored = store.load_json(paths.scene_json, SceneMap)

    assert restored == scene
    assert paths.observations_json.name == "observations.json"
    assert paths.assessment_json.name == "assessment.json"
    assert paths.topdown_png.name == "topdown.png"
    assert paths.chat_jsonl.name == "chat.jsonl"
    assert paths.point_cloud_glb == paths.geometry_dir / "point_cloud.glb"


def test_append_chat_writes_one_json_object_per_line(tmp_path):
    artifacts = importlib.import_module("ehs_spatial.artifacts")
    store = artifacts.ArtifactStore(tmp_path / "runs")

    store.append_chat("run-1", {"role": "user", "content": "How far?"})
    store.append_chat("run-1", {"role": "assistant", "content": "0.42 m"})

    lines = store.paths("run-1").chat_jsonl.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in lines] == [
        {"role": "user", "content": "How far?"},
        {"role": "assistant", "content": "0.42 m"},
    ]


def test_save_json_accepts_a_list_of_pydantic_models(tmp_path):
    artifacts = importlib.import_module("ehs_spatial.artifacts")
    store = artifacts.ArtifactStore(tmp_path / "runs")
    observations = [
        Observation2D(
            observation_id="obs-1",
            frame_id="frame-1",
            label="pallet",
            instance_id="0",
            mask_reference="rle",
            score=0.9,
            bbox=[0, 0, 1, 1],
            source_prompt="pallet",
        )
    ]

    store.save_json(store.paths("run-1").observations_json, observations)

    payload = json.loads(
        store.paths("run-1").observations_json.read_text(encoding="utf-8")
    )
    assert payload == [observations[0].model_dump(mode="json")]


def test_chat_cursor_uses_the_latest_persisted_interaction_id(tmp_path):
    artifacts = importlib.import_module("ehs_spatial.artifacts")
    store = artifacts.ArtifactStore(tmp_path / "runs")

    store.append_chat("run-1", {"type": "gemini_cursor", "interaction_id": "old"})
    store.append_chat("run-1", {"type": "message", "role": "user"})
    store.append_chat(
        "run-1",
        {
            "type": "chat_turn",
            "user": {"content": "How far?"},
            "assistant": {"content": "0.5 m"},
            "interaction_id": "new",
        },
    )

    assert store.latest_chat_cursor("run-1") == "new"
    assert store.latest_chat_cursor("missing") is None


@pytest.mark.parametrize("run_id", ["../escape", "..", "/tmp/escape", "back\\slash"])
def test_run_id_cannot_escape_artifact_root(tmp_path, run_id):
    artifacts = importlib.import_module("ehs_spatial.artifacts")
    store = artifacts.ArtifactStore(tmp_path / "runs")

    with pytest.raises(ValueError, match="run_id"):
        store.paths(run_id)


def test_list_runs_summarizes_complete_runs_and_degrades_corrupt_ones(tmp_path):
    artifacts = importlib.import_module("ehs_spatial.artifacts")
    store = artifacts.ArtifactStore(tmp_path / "runs")
    complete = store.paths("run-complete")
    complete.root.mkdir(parents=True)
    complete.manifest_json.write_text(
        json.dumps(
            {
                "run_id": "run-complete",
                "created_at": "2026-08-25T10:00:00+00:00",
                "operator": "inspector-a",
                "capture_tier": "mono",
            }
        ),
        encoding="utf-8",
    )
    complete.assessment_json.write_text(
        json.dumps(
            {
                "status": "NEEDS_REVIEW",
                "approximate_distance_m": 0.58,
                "distance_error_budget_m": 0.05,
            }
        ),
        encoding="utf-8",
    )
    complete.policies_json.write_text(
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
    complete.review_json.write_text(
        json.dumps({"decision": "confirmed"}), encoding="utf-8"
    )
    corrupt = store.paths("run-corrupt")
    corrupt.root.mkdir(parents=True)
    corrupt.manifest_json.write_text("{not json", encoding="utf-8")
    (store.root / "stray-file.txt").write_text("not a run", encoding="utf-8")

    assert store.list_runs() == [
        {
            "run_id": "run-complete",
            "created_at": "2026-08-25T10:00:00+00:00",
            "operator": "inspector-a",
            "capture_tier": "mono",
            "status": "NEEDS_REVIEW",
            "worst_policy": "FAIL",
            "distance": 0.58,
            "distance_error_budget_m": 0.05,
            "disposition": "confirmed",
        },
        {
            "run_id": "run-corrupt",
            "created_at": None,
            "operator": None,
            "capture_tier": None,
            "status": None,
            "worst_policy": None,
            "distance": None,
            "distance_error_budget_m": None,
            "disposition": None,
        },
    ]


def test_list_runs_is_empty_for_a_store_that_never_wrote(tmp_path):
    artifacts = importlib.import_module("ehs_spatial.artifacts")

    assert artifacts.ArtifactStore(tmp_path / "never").list_runs() == []


def test_list_runs_degrades_wrong_typed_manifest_fields_per_field(tmp_path):
    """A manifest whose created_at is a number must not TypeError the sort
    and hide every other run: the bad field degrades to None, the rest of
    the row and the index survive."""
    artifacts = importlib.import_module("ehs_spatial.artifacts")
    store = artifacts.ArtifactStore(tmp_path / "runs")
    good = store.paths("run-good")
    good.root.mkdir(parents=True)
    good.manifest_json.write_text(
        json.dumps({"created_at": "2026-08-01T00:00:00+00:00"}),
        encoding="utf-8",
    )
    bad = store.paths("run-bad")
    bad.root.mkdir(parents=True)
    bad.manifest_json.write_text(
        json.dumps({"created_at": 123, "operator": ["not", "a", "string"]}),
        encoding="utf-8",
    )
    bad.assessment_json.write_text(
        json.dumps({"status": 7, "approximate_distance_m": "0.5"}),
        encoding="utf-8",
    )

    runs = {run["run_id"]: run for run in store.list_runs()}

    assert set(runs) == {"run-good", "run-bad"}
    assert runs["run-good"]["created_at"] == "2026-08-01T00:00:00+00:00"
    assert runs["run-bad"]["created_at"] is None
    assert runs["run-bad"]["operator"] is None
    assert runs["run-bad"]["status"] is None
    assert runs["run-bad"]["distance"] is None


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        # Legacy bare-list file.
        ([{"policy_id": "p", "status": "NEEDS_REVIEW"}], "NEEDS_REVIEW"),
        # Current envelope; worst-first across mixed statuses.
        (
            {
                "specs": [],
                "results": [
                    {"status": "PASS"},
                    {"status": "INSUFFICIENT_EVIDENCE"},
                ],
            },
            "INSUFFICIENT_EVIDENCE",
        ),
        # Malformed shapes degrade to None instead of raising.
        ("{not json", None),
        ({"results": "nope"}, None),
        ({"results": ["not-a-dict"]}, None),
        ({"results": []}, None),
    ],
)
def test_list_runs_worst_policy_status_tolerates_all_shapes(
    tmp_path, payload, expected
):
    artifacts = importlib.import_module("ehs_spatial.artifacts")
    store = artifacts.ArtifactStore(tmp_path / "runs")
    paths = store.paths("run-a")
    paths.root.mkdir(parents=True)
    text = payload if isinstance(payload, str) else json.dumps(payload)
    paths.policies_json.write_text(text, encoding="utf-8")

    [run] = store.list_runs()

    assert run["worst_policy"] == expected
