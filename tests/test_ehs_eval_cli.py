import importlib.util
import json
from pathlib import Path

import pytest

from ehs_spatial.artifacts import ArtifactStore
from ehs_spatial.contracts import (
    Assessment,
    CaptureRun,
    ClimbReview,
    Entity3D,
    GroundedAnswer,
    SceneMap,
    SpatialFact,
)
from ehs_spatial.eval_pack import run_live_benchmark_case
from ehs_spatial.providers.base import ProviderError


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "ehs_eval.py"
SPEC = importlib.util.spec_from_file_location("ehs_eval_cli", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
ehs_eval = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ehs_eval)


PROVIDER_KEYS = ("REPLICATE_API_TOKEN", "FAL_KEY", "GEMINI_API_KEY")
CASE_TRUTH = {
    "ladder_050": ("FAIL", 0.5, "step ladder"),
    "ladder_070": ("PASS", 0.7, "step ladder"),
    "platform_inside": ("FAIL", 0.0, "portable work platform"),
    "fence_occluded": ("INSUFFICIENT_EVIDENCE", 0.5, "step ladder"),
}


def _write_small_pack(root: Path) -> None:
    root.mkdir()
    frames = []
    for index in range(4):
        image = root / f"image-{index}.png"
        image.write_bytes(b"png")
        frames.append({"frame_id": index, "rgb": image.name})
    scene_ply = root / "scene.ply"
    scene_ply.write_bytes(b"ply")
    scene_glb = root / "scene.glb"
    scene_glb.write_bytes(b"glTF")
    manifest = {
        "camera_height_m": 1.65,
        "cases": {
            case_id: {
                "expected_status": status,
                "expected_distance_m": distance,
                "movable_label": label,
                "scene_ply": scene_ply.name,
                "scene_glb": scene_glb.name,
                "frames": frames,
            }
            for case_id, (status, distance, label) in CASE_TRUTH.items()
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


class FakeLivePipeline:
    def __init__(self, root: Path, *, answer_fact_id: str = "fact-clearance"):
        self.store = ArtifactStore(root / "live_runs")
        self.captures: list[CaptureRun] = []
        self.answer_fact_id = answer_fact_id

    def run_assessment(self, capture: CaptureRun) -> Assessment:
        self.captures.append(capture)
        paths = self.store.paths(capture.run_id)
        paths.geometry_dir.mkdir(parents=True)
        paths.point_cloud_glb.write_bytes(b"glb")
        paths.observations_json.write_text("[]", encoding="utf-8")
        paths.topdown_png.write_bytes(b"png")
        evidence = ["frame-00", "frame-01", "frame-02", "frame-03"]
        ladder = Entity3D(
            entity_id="step-ladder-1",
            label="step ladder",
            observation_ids=["ladder-0", "ladder-1"],
            centroid_xyz=(2.1, 0.0, 0.7),
            footprint_xy=[(2.0, -0.3), (2.2, -0.3), (2.2, 0.3)],
            height_m=1.4,
            evidence_frame_ids=evidence[:2],
        )
        fence = Entity3D(
            entity_id="safety-fence-1",
            label="safety fence",
            observation_ids=["fence-0", "fence-1", "fence-2"],
            centroid_xyz=(0.0, 0.0, 0.8),
            footprint_xy=[(-1.5, -1.2), (1.5, -1.2), (1.5, 1.2)],
            height_m=1.6,
            evidence_frame_ids=evidence[:3],
        )
        robot = Entity3D(
            entity_id="industrial-robot-arm-1",
            label="industrial robot arm",
            observation_ids=["robot-0", "robot-1"],
            centroid_xyz=(0.0, 0.0, 0.8),
            footprint_xy=[(-0.2, -0.2), (0.2, -0.2), (0.2, 0.2)],
            height_m=1.3,
            evidence_frame_ids=evidence[:2],
        )
        fact = SpatialFact(
            fact_id="fact-clearance",
            predicate="minimum_boundary_clearance",
            subject_id=ladder.entity_id,
            object_id=fence.entity_id,
            value=0.52,
            unit="m",
            evidence_frame_ids=evidence[:2],
        )
        scene = SceneMap(
            run_id=capture.run_id,
            floor_plane=(0.0, 0.0, 1.0, 0.0),
            scale_source="camera_height",
            scale_factor=1.0,
            fence_polygon=[(-1.5, -1.2), (1.5, -1.2), (1.5, 1.2)],
            entities=[fence, robot, ladder],
            facts=[fact],
            warnings=[],
        )
        assessment = Assessment(
            status="FAIL",
            fact_ids=[fact.fact_id],
            evidence_frame_ids=evidence[:2],
            approximate_distance_m=0.52,
            climb_review=ClimbReview(
                verdict="yes",
                rationale="REVIEW only",
                fact_ids=[fact.fact_id],
            ),
        )
        self.store.save_json(paths.scene_json, scene)
        self.store.save_json(paths.assessment_json, assessment)
        self.store.append_chat(
            capture.run_id,
            {"type": "gemini_cursor", "interaction_id": "cursor-1"},
        )
        return assessment

    def answer_question(self, run_id: str, question: str) -> GroundedAnswer:
        assert question == "Why did this workcell fail the clearance check?"
        answer = GroundedAnswer(
            answer="The measured clearance is 0.52 m.",
            fact_ids=[self.answer_fact_id],
            evidence_frame_ids=["frame-00", "frame-01"],
        )
        self.store.append_chat(
            run_id,
            {
                "type": "chat_turn",
                "interaction_id": "cursor-2",
                "fact_ids": answer.fact_ids,
            },
        )
        return answer


def test_generate_prints_absolute_manifest_path(tmp_path, monkeypatch, capsys):
    output = tmp_path / "pack"

    def fake_generate(root):
        root = Path(root)
        root.mkdir(parents=True)
        (root / "manifest.json").write_text("{}", encoding="utf-8")
        return {}

    monkeypatch.setattr(ehs_eval, "generate_eval_pack", fake_generate)

    assert ehs_eval.main(["generate", "--output", str(output)]) == 0
    assert capsys.readouterr().out.strip() == str(
        (output / "manifest.json").resolve()
    )


@pytest.mark.parametrize(("passed", "expected_code"), [(True, 0), (False, 1)])
def test_offline_returns_report_status_and_prints_absolute_path(
    tmp_path,
    monkeypatch,
    capsys,
    passed,
    expected_code,
):
    pack = tmp_path / "pack"
    pack.mkdir()

    def fake_offline(root):
        report = {"passed": passed, "cases": {}}
        (Path(root) / "offline_report.json").write_text(
            json.dumps(report), encoding="utf-8"
        )
        return report

    monkeypatch.setattr(ehs_eval, "run_offline_benchmark", fake_offline)

    assert ehs_eval.main(["offline", "--pack", str(pack)]) == expected_code
    assert capsys.readouterr().out.strip() == str(
        (pack / "offline_report.json").resolve()
    )


def test_live_requires_explicit_paid_flag_before_credentials(
    tmp_path,
    monkeypatch,
    capsys,
):
    for name in PROVIDER_KEYS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(
        ehs_eval,
        "run_live_benchmark_case",
        lambda *_args, **_kwargs: pytest.fail("live runner must not start"),
    )

    code = ehs_eval.main(
        ["live", "--pack", str(tmp_path), "--case", "ladder_050"]
    )

    captured = capsys.readouterr()
    assert code == 2
    assert "--live" in captured.err
    assert captured.out == ""


def test_live_requires_all_provider_variables_without_printing_values(
    tmp_path,
    monkeypatch,
    capsys,
):
    secret = "must-not-appear"
    monkeypatch.setenv("REPLICATE_API_TOKEN", secret)
    monkeypatch.delenv("FAL_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(
        ehs_eval,
        "run_live_benchmark_case",
        lambda *_args, **_kwargs: pytest.fail("live runner must not start"),
    )

    code = ehs_eval.main(
        [
            "live",
            "--pack",
            str(tmp_path),
            "--case",
            "ladder_050",
            "--live",
        ]
    )

    captured = capsys.readouterr()
    assert code == 2
    assert "FAL_KEY" in captured.err
    assert "GEMINI_API_KEY" in captured.err
    assert secret not in captured.err
    assert captured.out == ""


@pytest.mark.parametrize(("passed", "expected_code"), [(True, 0), (False, 1)])
def test_live_prints_report_path_and_returns_report_status(
    tmp_path,
    monkeypatch,
    capsys,
    passed,
    expected_code,
):
    pack = tmp_path / "pack"
    pack.mkdir()
    for index, name in enumerate(PROVIDER_KEYS):
        monkeypatch.setenv(name, f"secret-{index}")

    calls = []

    def fake_live(root, case_id):
        calls.append((Path(root), case_id))
        report = {"passed": passed, "case_id": case_id}
        (Path(root) / "live_report.json").write_text(
            json.dumps(report), encoding="utf-8"
        )
        return report

    monkeypatch.setattr(ehs_eval, "run_live_benchmark_case", fake_live)

    code = ehs_eval.main(
        [
            "live",
            "--pack",
            str(pack),
            "--case",
            "ladder_050",
            "--live",
        ]
    )

    captured = capsys.readouterr()
    assert code == expected_code
    assert calls == [(pack, "ladder_050")]
    assert captured.out.strip() == str((pack / "live_report.json").resolve())
    assert all(value not in captured.out for value in ("secret-0", "secret-1", "secret-2"))


def test_live_prints_the_recorded_provider_failure_layer(
    tmp_path,
    monkeypatch,
    capsys,
):
    pack = tmp_path / "pack"
    pack.mkdir()
    for index, name in enumerate(PROVIDER_KEYS):
        monkeypatch.setenv(name, f"secret-{index}")

    def fake_live(root, case_id):
        report = {
            "passed": False,
            "case_id": case_id,
            "error": {
                "provider": "fal",
                "operation": "sam3.segment",
                "message": "request rejected",
            },
        }
        (Path(root) / "live_report.json").write_text(
            json.dumps(report), encoding="utf-8"
        )
        return report

    monkeypatch.setattr(ehs_eval, "run_live_benchmark_case", fake_live)

    code = ehs_eval.main(
        [
            "live",
            "--pack",
            str(pack),
            "--case",
            "ladder_050",
            "--live",
        ]
    )

    captured = capsys.readouterr()
    assert code == 1
    assert "fal sam3.segment: request rejected" in captured.err
    assert all(value not in captured.err for value in ("secret-0", "secret-1", "secret-2"))


def test_live_runner_reconciles_provider_result_with_metric_truth(tmp_path):
    pack = tmp_path / "pack"
    _write_small_pack(pack)
    pipeline = FakeLivePipeline(pack)

    report = run_live_benchmark_case(pack, "ladder_050", pipeline=pipeline)

    assert report == json.loads(
        (pack / "live_report.json").read_text(encoding="utf-8")
    )
    assert report["passed"] is True
    assert report["expected_status"] == report["actual_status"] == "FAIL"
    assert report["threshold_side_matches"] is True
    assert report["entity_evidence_passed"] is True
    assert report["absolute_error_m"] == pytest.approx(0.02)
    assert report["criterion_m"] == pytest.approx(0.6)
    assert report["observations"] == {"count": 0, "by_label": {}}
    assert report["entity_labels"] == [
        "safety fence",
        "industrial robot arm",
        "step ladder",
    ]
    assert report["entities"][2] == {
        "entity_id": "step-ladder-1",
        "label": "step ladder",
        "observation_ids": ["ladder-0", "ladder-1"],
        "evidence_frame_ids": ["frame-00", "frame-01"],
    }
    assert report["assessment_fact_ids"] == ["fact-clearance"]
    assert report["chat"]["fact_ids"] == ["fact-clearance"]
    assert report["chat"]["grounded"] is True
    assert report["source_artifacts"]["scene_glb"] == "scene.glb"
    assert pipeline.captures[0].camera_height_m == pytest.approx(1.65)
    assert pipeline.captures[0].criterion.minimum_clearance_m == pytest.approx(0.6)
    assert pipeline.captures[0].image_paths == [
        str((pack / f"image-{index}.png").resolve()) for index in range(4)
    ]
    assert all(
        (pack / relative_path).is_file()
        for relative_path in report["provider_artifacts"].values()
    )


def test_live_runner_rejects_pack_path_escape_before_provider_call(tmp_path):
    pack = tmp_path / "pack"
    _write_small_pack(pack)
    manifest_path = pack / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["cases"]["ladder_050"]["frames"][0]["rgb"] = "../outside.png"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    pipeline = FakeLivePipeline(pack)

    with pytest.raises(ValueError, match="relative path inside the pack root"):
        run_live_benchmark_case(pack, "ladder_050", pipeline=pipeline)

    assert pipeline.captures == []


def test_live_runner_writes_redacted_provider_failure_report(
    tmp_path,
    monkeypatch,
):
    pack = tmp_path / "pack"
    _write_small_pack(pack)
    secret = "provider-secret-that-must-not-be-written"
    monkeypatch.setenv("FAL_KEY", secret)

    class FailingPipeline:
        def __init__(self):
            self.store = ArtifactStore(pack / "live_runs")

        def run_assessment(self, _capture):
            raise ProviderError(
                "fal",
                "sam3.segment",
                f"request rejected for credential {secret}",
            )

    report = run_live_benchmark_case(
        pack,
        "ladder_050",
        pipeline=FailingPipeline(),
    )

    saved = (pack / "live_report.json").read_text(encoding="utf-8")
    assert report["passed"] is False
    assert report["error"]["provider"] == "fal"
    assert report["error"]["operation"] == "sam3.segment"
    assert "[REDACTED]" in report["error"]["message"]
    assert secret not in saved


def test_live_runner_clears_stale_report_before_unwrapped_crash(tmp_path):
    pack = tmp_path / "pack"
    _write_small_pack(pack)
    (pack / "live_report.json").write_text(
        json.dumps({"passed": True, "case_id": "ladder_050"}),
        encoding="utf-8",
    )

    class CrashingPipeline:
        def __init__(self):
            self.store = ArtifactStore(pack / "live_runs")

        def run_assessment(self, _capture):
            raise OSError("disk full after paid provider calls")

    with pytest.raises(OSError):
        run_live_benchmark_case(pack, "ladder_050", pipeline=CrashingPipeline())

    # Only ProviderError writes a failure report; an unwrapped crash must not
    # leave the previous invocation's verdict behind as the current report.
    assert not (pack / "live_report.json").exists()
