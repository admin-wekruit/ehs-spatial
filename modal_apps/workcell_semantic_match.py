"""One ephemeral two-A100 experiment; reuses frozen RGB, masks and geometry.

modal run modal_apps/workcell_semantic_match.py --root BASELINE --out OUTPUT \
    --config docs/workcell-photo/semantic-match-protocol.json
"""
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import modal

if not modal.is_local():
    sys.path[:0] = ["/repo", "/repo/scripts", "/repo/modal_apps"]
from modal_apps.fast_report_app import build_image

app = modal.App("workcell-semantic-match-experiment")
image = build_image() if modal.is_local() else modal.Image.debian_slim()
RATE = 2 * .000694 + 8 * .0000131 + 32 * .00000222


def unpack(payload, destination):
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        if any(not m.isfile() or Path(m.name).is_absolute() or ".." in Path(m.name).parts
               for m in archive.getmembers()):
            raise ValueError("Unsafe experiment archive")
        archive.extractall(destination, filter="data")


@app.function(image=image, gpu="A100-80GB:2", cpu=8, memory=32 * 1024,
              volumes={"/v/x13": modal.Volume.from_name("panoptes-x13-models")},
              timeout=1200, retries=0, min_containers=0)
def experiment(payload: bytes):
    started = time.monotonic()
    with tempfile.TemporaryDirectory() as directory:
        root, out = Path(directory) / "input", Path(directory) / "output"
        root.mkdir(); out.mkdir(); unpack(payload, root)
        command = [sys.executable, "/repo/scripts/workcell_semantic_match.py", "--root", str(root),
                   "--out", str(out), "--config", str(root / "protocol.json")]
        env = {**os.environ, "PYTHONPATH": "/repo:/repo/scripts:/repo/modal_apps", "HF_HUB_OFFLINE": "1",
               "HF_HUB_CACHE": "/v/x13/hf", "HF_HOME": "/v/x13/hf", "OMP_NUM_THREADS": "2",
               "OPENBLAS_NUM_THREADS": "2", "PYTHONDONTWRITEBYTECODE": "1"}

        def run(args, gpu=""):
            print("semantic stage start:", " ".join(args), "GPU", gpu or "CPU", flush=True)
            completed = subprocess.run(command + args, env={**env, "CUDA_VISIBLE_DEVICES": gpu},
                                       capture_output=True, text=True, timeout=900)
            if completed.returncode:
                safe = [line for line in (completed.stdout + completed.stderr).splitlines()
                        if not any(word in line.lower() for word in ("token", "secret", "capabilit"))]
                raise RuntimeError(" ".join(args) + " failed: " + "\n".join(safe[-25:]))
            print("semantic stage complete:", " ".join(args), flush=True)

        error = None
        try:
            run(["--prepare"])
            with ThreadPoolExecutor(max_workers=2) as pool:
                jobs = [pool.submit(run, ["--encode", key], str(i))
                        for i, key in enumerate(("pe-core-l", "siglip2-so400m"))]
                for job in jobs:
                    job.result()
            run(["--analyze"])
        except Exception as exc:
            error = str(exc)
            (out / "failure.json").write_text(json.dumps({"error": error}))
        elapsed = time.monotonic() - started
        (out / "container-timing.json").write_text(json.dumps({"containerSeconds": elapsed}))
        archive = io.BytesIO()
        with tarfile.open(fileobj=archive, mode="w:gz") as bundle:
            for path in sorted(out.rglob("*")):
                if path.is_file():
                    bundle.add(path, arcname=path.relative_to(out).as_posix())
        return {"archive": archive.getvalue(), "containerSeconds": elapsed, "error": error}


@app.local_entrypoint()
def main(root: str, out: str, config: str):
    baseline, destination = Path(root), Path(out)
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "spend-ledger.json").exists():
        raise ValueError("Choose a fresh output directory; never overwrite a spend ledger")
    frames = sorted(baseline.glob("frame_*.json.gz"))
    files = [baseline / name for name in ("objects.json", "scene-report.json")]
    files += [baseline / f"photo-{i}.png" for i in range(1, len(frames) + 1)]
    files += frames
    if (len(frames) < 2 or not all(path.is_file() for path in files)
            or [f.name for f in frames] != [f"frame_{i:04d}.json.gz" for i in range(1, len(frames) + 1)]):
        raise ValueError("Expected frozen catalog, scene, and photo/depth frame 1..N (N >= 2) of one scene")
    archive, hashes = io.BytesIO(), {}
    with tarfile.open(fileobj=archive, mode="w:gz") as bundle:
        for path, name in [(path, path.name) for path in files] + [(Path(config), "protocol.json")]:
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
            bundle.add(path, arcname=name)
    (destination / "input-manifest.json").write_text(json.dumps({"sha256": hashes}, indent=2))
    ledger = {"mode": "ephemeral modal run", "hardware": "2 x A100-80GB; 8 CPU; 32 GiB",
              "rateSource": "https://modal.com/pricing", "rateCheckedDate": "2026-10-02",
              "reservedResourceUsdPerSecond": RATE, "actualBilledUsd": None, "status": "started",
              "scope": "incremental semantic experiment; cached reconstruction; no generative VLM",
              "estimateBasis": "requested-resource list rate; excludes build, load before function, storage, egress and teardown; not invoice"}
    start = time.monotonic()
    try:
        result = experiment.remote(archive.getvalue())
        unpack(result["archive"], destination)
        ledger.update(status="failed" if result["error"] else "completed",
                      functionSeconds=result["containerSeconds"], estimateUsd=RATE * result["containerSeconds"])
        if result["error"]:
            raise RuntimeError(result["error"])
    except Exception as exc:
        ledger.update(status="failed", errorType=type(exc).__name__)
        raise
    finally:
        ledger["callSeconds"] = time.monotonic() - start
        ledger["callWindowEstimateUsd"] = RATE * ledger["callSeconds"]
        (destination / "spend-ledger.json").write_text(json.dumps(ledger, indent=2) + "\n")
    print(json.dumps(ledger))
