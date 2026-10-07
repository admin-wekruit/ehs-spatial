"""Replay the frozen four-run geometry experiment without changing the pipeline.

Use the project .venv/bin/python directly: research model dependencies are local
extras, deliberately absent from the production pyproject. All output stays in
a new private directory; source runs and cached model responses are preserved.
"""

import argparse
from contextlib import redirect_stdout, redirect_stderr
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
RUN_IDS = ("real-clean-01", "real-clean-02", "real-clean-03", "user-bor1-02")
sys.path[:0] = [str(REPO), str(REPO / "scripts")]


def hashes(root):
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            with path.open("rb") as stream:
                result[str(path.relative_to(root))] = hashlib.file_digest(stream, "sha256").hexdigest()
    return result


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=["archive", "replay", "da3", "mapanything", "pi3x"], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--runs", nargs="+", choices=RUN_IDS, default=RUN_IDS)
    parser.add_argument("--model-id", default="depth-anything/DA3-LARGE-1.1")
    parser.add_argument("--model-dir")
    parser.add_argument("--vendor-dir")
    parser.add_argument("--device", choices=["mps", "cpu", "cuda"], default="mps")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        parser.error("Output must not exist: each attempt keeps its own evidence")
    vendor = Path(args.vendor_dir).resolve() if args.vendor_dir else None
    model_dir = Path(args.model_dir).resolve() if args.model_dir else None
    output.mkdir(parents=True)
    (output / "runs").mkdir()
    source_hashes = {run: hashes(REPO / "runs" / run) for run in args.runs}
    save(output / "source_sha256.json", source_hashes)
    for run in args.runs:
        # ponytail: whole-run copies avoid a fragile cache allowlist; four
        # fixture runs total 766 MB. Larger packs should use filesystem clones.
        shutil.copytree(REPO / "runs" / run, output / "runs" / run)
    source_code = {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
                   for pattern in ("ehs_spatial/**/*.py", "scripts/*.py", "tests/test_geometry_invariants.py", "tests/test_cell_rect.py")
                   for p in REPO.glob(pattern)}
    manifest = {"arm": args.arm, "runs": list(args.runs), "code_sha256": source_code,
                "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "source_commit": subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip(),
                "scope": "fixed-cache geometry slot; no new VLM, SAM, MoGe or climb calls",
                "privacy": "model weights local; socket connections rejected by Python audit hook",
                "criterion": {"minimum_clearance_m": 0.6}, "camera_height_m": 1.5,
                "results": {}}
    save(output / "experiment.json", manifest)
    attempts = []

    def offline(event, event_args):
        if event in {"socket.connect", "socket.getaddrinfo", "socket.sendto"}:
            attempts.append(event)
            raise RuntimeError("Experiment prohibits network access")

    os.environ.update(HF_HUB_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1", TRANSFORMERS_OFFLINE="1")
    sys.addaudithook(offline)
    os.chdir(output)
    from ehs_spatial.contracts import CaptureRun, Observation2D, PolicySpec
    from ehs_spatial.pipeline import EHSAssessmentPipeline
    from ehs_spatial.policy import evaluate_policies
    from ehs_spatial.providers.map_anything import MapAnythingAdapter
    from ehs_spatial.providers.moge import MoGeAnchorAdapter
    from ehs_spatial.reproject import verify_reprojection
    from ehs_spatial.scene import build_scene_and_assess
    from scene_inventory import _frames, main as inventory_main
    from candidate_metrics import collect_metrics

    runner = None
    if args.arm == "da3":
        from candidate_geometry_backend import DA3Runner
        runner = DA3Runner(vendor, model_dir, model_id=args.model_id,
                           device=args.device, process_res=518)
    elif args.arm == "mapanything":
        from candidate_mapanything_backend import MapAnythingRunner
        runner = MapAnythingRunner(vendor, model_dir, device=args.device)
    elif args.arm == "pi3x":
        from candidate_pi3x_backend import Pi3XRunner
        runner = Pi3XRunner(vendor, model_dir, device=args.device)

    for run_id in args.runs:
        run = output / "runs" / run_id
        record = {"status": "running", "source_run_id": run_id}
        manifest["results"][run_id] = record
        begin = time.perf_counter()
        with (output / f"{run_id}.log").open("w") as log, redirect_stdout(log), redirect_stderr(log):
            try:
                frames = _frames(run)
                original_rgb = [np.asarray(Image.open(f.canonical_image_path).convert("RGB")) for f in frames]
                for name in ("moge3_normals.npz", "moge3_points.npz"):
                    assert (run / "geometry" / name).is_file(), "Missing frozen MoGe maps"
                for frame in frames:
                    assert (run / "geometry" / "moge" / f"{frame.frame_id}.json").is_file()
                if runner is not None:
                    model_start = time.perf_counter()
                    frames, _ = MapAnythingAdapter(runner=runner).run(
                        [f.canonical_image_path for f in frames], run / "geometry")
                    record["geometry_adapter_seconds"] = time.perf_counter() - model_start
                    for before, frame in zip(original_rgb, frames, strict=True):
                        after = np.asarray(Image.open(frame.canonical_image_path).convert("RGB"))
                        assert before.shape == after.shape, "Candidate changed the fixed cache grid"
                        assert np.abs(before.astype(int) - after.astype(int)).max() <= 1, "Candidate RGB mapping changed"
                    record["model"] = dict(runner.metadata)
                    save(run / "geometry" / "candidate_manifest.json", record["model"])
                # Never let an old score survive a failed verification.
                (run / "inventory" / "reprojection.json").unlink(missing_ok=True)
                (run / "inventory" / "reprojection.png").unlink(missing_ok=True)
                post_start = time.perf_counter()
                if args.arm != "archive":
                    envelope = json.loads((run / "policies.json").read_text())
                    specs = [PolicySpec.model_validate(x) for x in envelope.get("specs", [])]
                    prepared = CaptureRun(run_id=run_id,
                        image_paths=[str(p) for p in sorted((run / "input").glob("image_*"))], policies=specs)
                    pipeline = EHSAssessmentPipeline(gemini=object(), sam3=object(),
                        map_anything=object(), moge=MoGeAnchorAdapter())
                    scale = pipeline._resolve_scale(prepared, frames, run / "geometry")
                    assert scale["source"] == "moge_anchor", "Frozen anchor did not resolve"
                    observations = [Observation2D.model_validate(x) for x in json.loads((run / "observations.json").read_text())]
                    scene, assessment = build_scene_and_assess(run_id, frames, observations,
                        prepared.camera_height_m, prepared.criterion,
                        scale_factor_override=scale["override"], scale_source=scale["source"],
                        scale_confidence=scale["confidence"], scale_warnings=scale["warnings"])
                    save(run / "scene.json", scene.model_dump(mode="json"))
                    save(run / "assessment.json", assessment.model_dump(mode="json"))
                    envelope["results"] = [r.model_dump(mode="json") for r in
                        evaluate_policies(specs, scene, capture_frame_count=len(frames))]
                    save(run / "policies.json", envelope)
                    assert inventory_main(["--run", run_id]) == 0
                    assert json.loads((run / "policies.json").read_text())["specs"] == envelope["specs"], "Policy specs changed"
                verify_reprojection(run_id)
                record["postprocessing_seconds"] = time.perf_counter() - post_start
                assert not attempts, "A hidden network request was attempted"
                record["metrics"] = collect_metrics(run)
                record["status"] = "measured"
                save(output / f"{run_id}-metrics.json", record["metrics"])
            except Exception as error:
                record.update(status="failed", error=f"{type(error).__name__}: {error}")
                traceback.print_exc()
            record["elapsed_seconds"] = time.perf_counter() - begin
        save(output / "experiment.json", manifest)
        print(run_id, record["status"], f"{record['elapsed_seconds']:.2f}s", record.get("error", ""), flush=True)
    with (output / "geometry-tests.log").open("w") as log:
        test = subprocess.run([sys.executable, "-m", "pytest", str(REPO / "tests/test_geometry_invariants.py"),
            str(REPO / "tests/test_cell_rect.py"), "-q", "-rs", f"--junitxml={output / 'geometry-tests.xml'}"],
            cwd=output, stdout=log, stderr=subprocess.STDOUT)
    manifest["geometry_tests_exit_code"] = test.returncode
    manifest["network_attempts"] = attempts
    manifest["source_unchanged"] = all(hashes(REPO / "runs" / r) == source_hashes[r] for r in args.runs)
    assert manifest["source_unchanged"], "Frozen source artifacts changed during experiment"
    save(output / "experiment.json", manifest)
    print("geometry tests:", test.returncode, "source unchanged:", manifest["source_unchanged"], flush=True)
    return int(any(r["status"] != "measured" for r in manifest["results"].values()))


if __name__ == "__main__":
    raise SystemExit(main())
