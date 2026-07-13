import importlib
import json
from pathlib import Path

import pytest

from ehs_spatial.contracts import CaptureRun, SceneMap


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


@pytest.mark.parametrize("run_id", ["../escape", "..", "/tmp/escape", "back\\slash"])
def test_run_id_cannot_escape_artifact_root(tmp_path, run_id):
    artifacts = importlib.import_module("ehs_spatial.artifacts")
    store = artifacts.ArtifactStore(tmp_path / "runs")

    with pytest.raises(ValueError, match="run_id"):
        store.paths(run_id)
