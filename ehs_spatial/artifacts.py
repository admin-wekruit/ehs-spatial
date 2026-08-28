import json
from dataclasses import dataclass
from pathlib import Path
from shutil import copy2
from typing import TypeVar

from pydantic import BaseModel

from .contracts import CaptureRun
from .path_safety import validate_safe_path_segment


ModelT = TypeVar("ModelT", bound=BaseModel)


def _read_json_dict(path: Path) -> dict:
    """Best-effort read of one run artifact. History browsing must survive a
    missing or corrupt file, so anything unreadable degrades to {}."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _str_or_none(value: object) -> str | None:
    """Per-field degradation: a wrong-typed field becomes None, never a crash."""
    return value if isinstance(value, str) else None


def _number_or_none(value: object) -> float | int | None:
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


# Worst first — mirrors the app's status ordering.
_POLICY_STATUS_ORDER = ("FAIL", "NEEDS_REVIEW", "INSUFFICIENT_EVIDENCE", "PASS")


def _worst_policy_status(path: Path) -> str | None:
    """Worst status across policies.json results. Handles the current
    {"specs", "results"} envelope and legacy bare-list files; anything
    missing or malformed degrades to None."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    results = payload.get("results") if isinstance(payload, dict) else payload
    if not isinstance(results, list):
        return None
    statuses = {
        result.get("status") for result in results if isinstance(result, dict)
    }
    for status in _POLICY_STATUS_ORDER:
        if status in statuses:
            return status
    return None


@dataclass(frozen=True)
class RunPaths:
    root: Path
    input_dir: Path
    geometry_dir: Path
    observations_json: Path
    scene_json: Path
    assessment_json: Path
    manifest_json: Path
    policies_json: Path
    review_json: Path
    topdown_png: Path
    plan_view_png: Path
    cloud_perspective_png: Path
    cloud_topdown_png: Path
    chat_jsonl: Path
    point_cloud_glb: Path
    semantic_ply: Path
    evidence_dir: Path
    viewer_html: Path


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
            manifest_json=run_root / "manifest.json",
            policies_json=run_root / "policies.json",
            review_json=run_root / "review.json",
            topdown_png=run_root / "topdown.png",
            plan_view_png=run_root / "plan_view.png",
            cloud_perspective_png=run_root / "cloud_perspective.png",
            cloud_topdown_png=run_root / "cloud_topdown.png",
            chat_jsonl=run_root / "chat.jsonl",
            point_cloud_glb=run_root / "geometry" / "point_cloud.glb",
            semantic_ply=run_root / "geometry" / "semantic_cloud.ply",
            evidence_dir=run_root / "evidence",
            viewer_html=run_root / "viewer.html",
        )

    def list_runs(self) -> list[dict]:
        """Summaries of every cached run, newest first, for history browsing.
        Fields come from manifest.json, assessment.json and review.json when
        present; a missing or corrupt file leaves its fields None rather than
        raising, so one bad run never hides the rest."""
        summaries: list[dict] = []
        if not self.root.is_dir():
            return summaries
        for run_dir in self.root.iterdir():
            if not run_dir.is_dir():
                continue
            try:
                paths = self.paths(run_dir.name)
            except ValueError:
                # A name paths() refuses is not a run this store wrote.
                continue
            manifest = _read_json_dict(paths.manifest_json)
            assessment = _read_json_dict(paths.assessment_json)
            review = _read_json_dict(paths.review_json)
            summaries.append(
                {
                    "run_id": run_dir.name,
                    "created_at": _str_or_none(manifest.get("created_at")),
                    "operator": _str_or_none(manifest.get("operator")),
                    "capture_tier": _str_or_none(manifest.get("capture_tier")),
                    "status": _str_or_none(assessment.get("status")),
                    "worst_policy": _worst_policy_status(paths.policies_json),
                    "distance": _number_or_none(
                        assessment.get("approximate_distance_m")
                    ),
                    "distance_error_budget_m": _number_or_none(
                        assessment.get("distance_error_budget_m")
                    ),
                    "disposition": _str_or_none(review.get("decision")),
                }
            )
        summaries.sort(
            key=lambda summary: (summary["created_at"] or "", summary["run_id"]),
            reverse=True,
        )
        return summaries

    def prepare_run(self, capture: CaptureRun) -> CaptureRun:
        paths = self.paths(capture.run_id)
        paths.input_dir.mkdir(parents=True, exist_ok=True)
        paths.geometry_dir.mkdir(parents=True, exist_ok=True)
        copied = []
        for index, source_value in enumerate(capture.image_paths, start=1):
            source = Path(source_value)
            destination = paths.input_dir / f"image_{index:02d}{source.suffix.lower()}"
            copy2(source, destination)
            self._normalize_orientation(destination)
            copied.append(str(destination))
        return capture.model_copy(update={"image_paths": copied})

    @staticmethod
    def _normalize_orientation(path: Path) -> None:
        """Bake the EXIF orientation into the pixels at ingest. A portrait
        phone photo stores landscape pixels plus a rotation tag; every
        downstream consumer (geometry provider, SAM, renders) reads pixels
        only, so an unapplied tag rotates the entire reconstruction 90°.
        One deterministic fix at the single entry point."""
        try:
            from PIL import Image, ImageOps

            with Image.open(path) as image:
                if image.getexif().get(274, 1) == 1:
                    return
                upright = ImageOps.exif_transpose(image)
                upright.save(path, quality=95)
        except Exception:
            # never block an ingest on a malformed EXIF block
            return

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
