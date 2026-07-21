import json
from dataclasses import dataclass
from pathlib import Path
from shutil import copy2
from typing import TypeVar

from pydantic import BaseModel

from .contracts import CaptureRun
from .path_safety import validate_safe_path_segment


ModelT = TypeVar("ModelT", bound=BaseModel)


@dataclass(frozen=True)
class RunPaths:
    root: Path
    input_dir: Path
    geometry_dir: Path
    observations_json: Path
    scene_json: Path
    assessment_json: Path
    topdown_png: Path
    plan_view_png: Path
    cloud_perspective_png: Path
    cloud_topdown_png: Path
    chat_jsonl: Path
    point_cloud_glb: Path
    semantic_ply: Path


class ArtifactStore:
    def __init__(self, root: str | Path = "runs") -> None:
        self.root = Path(root)

    def paths(self, run_id: str) -> RunPaths:
        validate_safe_path_segment(run_id, "run_id")
        run_root = self.root / run_id
        return RunPaths(
            root=run_root,
            input_dir=run_root / "input",
            geometry_dir=run_root / "geometry",
            observations_json=run_root / "observations.json",
            scene_json=run_root / "scene.json",
            assessment_json=run_root / "assessment.json",
            topdown_png=run_root / "topdown.png",
            plan_view_png=run_root / "plan_view.png",
            cloud_perspective_png=run_root / "cloud_perspective.png",
            cloud_topdown_png=run_root / "cloud_topdown.png",
            chat_jsonl=run_root / "chat.jsonl",
            point_cloud_glb=run_root / "geometry" / "point_cloud.glb",
            semantic_ply=run_root / "geometry" / "semantic_cloud.ply",
        )

    def prepare_run(self, capture: CaptureRun) -> CaptureRun:
        paths = self.paths(capture.run_id)
        paths.input_dir.mkdir(parents=True, exist_ok=True)
        paths.geometry_dir.mkdir(parents=True, exist_ok=True)
        copied = []
        for index, source_value in enumerate(capture.image_paths, start=1):
            source = Path(source_value)
            destination = paths.input_dir / f"image_{index:02d}{source.suffix.lower()}"
            copy2(source, destination)
            copied.append(str(destination))
        return capture.model_copy(update={"image_paths": copied})

    def save_json(
        self, path: str | Path, model: BaseModel | list[BaseModel]
    ) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = (
            [item.model_dump(mode="json") for item in model]
            if isinstance(model, list)
            else model.model_dump(mode="json")
        )
        destination.write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )

    def load_json(self, path: str | Path, model_type: type[ModelT]) -> ModelT:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return model_type.model_validate(payload)

    def append_chat(self, run_id: str, entry: dict[str, object] | BaseModel) -> None:
        path = self.paths(run_id).chat_jsonl
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = entry.model_dump(mode="json") if isinstance(entry, BaseModel) else entry
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload) + "\n")

    def latest_chat_cursor(self, run_id: str) -> str | None:
        path = self.paths(run_id).chat_jsonl
        if not path.exists():
            return None
        for line in reversed(path.read_text(encoding="utf-8").splitlines()):
            entry = json.loads(line)
            if entry.get("type") not in {"gemini_cursor", "chat_turn"}:
                continue
            interaction_id = entry.get("interaction_id")
            if not isinstance(interaction_id, str) or not interaction_id:
                raise ValueError("gemini cursor requires a non-empty interaction_id")
            return interaction_id
        return None
